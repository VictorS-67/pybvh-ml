"""PyTorch Dataset classes for motion capture data."""
from __future__ import annotations

import multiprocessing
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from pybvh import read_bvh_file
from pybvh_ml.convert import convert_arrays
from pybvh_ml.packing import pack_to_ctv, pack_to_flat, pack_to_tvc
from pybvh_ml.preprocessing import extract_repr
from pybvh_ml.sequences import standardize_length, uniform_temporal_sample
from pybvh_ml.pipeline import AugmentationPipeline


def rng_for(
    seed: int | None,
    epoch: int,
    idx: int,
) -> np.random.Generator:
    """Build the per-sample generator for one ``(seed, epoch, idx)`` triple.

    The seeding scheme both Dataset classes use, exposed because any
    Dataset needs it — not only subclasses of the two shipped here.  A
    ``SeedSequence([seed, epoch, idx])`` makes each sample's stream
    independent of the order samples are drawn in and of which worker
    draws them, so a clip augments identically whether it lands in
    worker 0 or worker 3, and shuffling doesn't change what any sample
    receives.

    Parameters
    ----------
    seed : int or None
        Base seed.  ``None`` returns a generator seeded from fresh OS
        entropy — reproducibility is off, and the *(epoch, idx)* pair is
        ignored.
    epoch : int
        Current epoch; see :class:`EpochState` for propagating it to
        DataLoader workers.
    idx : int
        Sample index.  Must be non-negative — resolve Python negative
        indexing before calling.

    Returns
    -------
    numpy.random.Generator

    Examples
    --------
    >>> from pybvh_ml.torch import EpochState, rng_for
    >>> class MyFeeder(torch.utils.data.Dataset):
    ...     def __init__(self, seed=0):
    ...         self.seed = seed
    ...         self.epoch_state = EpochState()
    ...     def set_epoch(self, epoch):
    ...         self.epoch_state.set(epoch)
    ...     def __getitem__(self, idx):
    ...         rng = rng_for(self.seed, self.epoch_state.current, idx)
    ...         ...
    """
    if seed is None:
        return np.random.default_rng()
    ss = np.random.SeedSequence([int(seed), int(epoch), int(idx)])
    return np.random.default_rng(ss)


def _replay_augmentation_params(
    dataset,
    idx: int,
    epoch: int | None,
    cls_name: str,
) -> list[dict]:
    """Re-run one sample's augmentation and report what it drew.

    Shared by both dataset classes.  The pipeline is genuinely re-run
    rather than the draws recomputed in isolation: steps like
    :func:`~pybvh_ml.add_joint_noise` consume an amount of randomness
    that depends on the clip's shape, so the later steps of a pipeline
    only report truthfully when the earlier ones ran on the real arrays.
    """
    idx = _normalize_index(idx, len(dataset), cls_name)
    if dataset.augmentation is None:
        return []
    if dataset.seed is None:
        raise ValueError(
            f"{cls_name} was built without a seed, so its augmentation "
            f"draws came from fresh OS entropy and are not recoverable — "
            f"any answer for sample {idx} would be a fresh draw, not the "
            f"one that ran. Rebuild with seed=... to make draws "
            f"replayable.")
    if epoch is None:
        epoch = dataset._epoch_state.current
    elif epoch < 0:
        raise ValueError(f"epoch must be >= 0, got {epoch}")

    root_pos, joint_data = dataset._clip_arrays(idx)
    _, _, params = dataset.augmentation(
        root_pos=root_pos, joint_data=joint_data,
        rng=rng_for(dataset.seed, epoch, idx), return_params=True)
    return params


def _normalize_index(idx: int, length: int, cls_name: str) -> int:
    """Resolve Python negative indexing and bounds-check.

    Keeps ``ds[-1]`` and ``ds[len(ds) - 1]`` on the same
    ``(seed, epoch, idx)`` rng stream — a raw negative index would
    otherwise crash ``SeedSequence`` (non-negative integers only), but
    only when seeded augmentation was active.
    """
    if idx < 0:
        idx += length
    if not 0 <= idx < length:
        raise IndexError(
            f"{cls_name} index {idx - length if idx < 0 else idx} out of "
            f"range for {length} clips")
    return idx


_MISSING_SET_EPOCH_MSG = (
    "{cls} was seeded (seed={seed!r}) but set_epoch() was never called; "
    "every epoch will draw identically per sample. "
    "Call dataset.set_epoch(epoch) at the start of each epoch, or pass "
    "seed=None for fresh OS entropy each call.")


class EpochState:
    """Shared-memory epoch counter for DataLoader-worker visibility.

    Pairs with :func:`rng_for`: it supplies the *epoch* term that makes
    a seeded sample's draw change from one epoch to the next.  Public
    because a Dataset that isn't a :class:`MotionDataset` subclass needs
    exactly this to honor the ``set_epoch`` contract — hold one, call
    :meth:`set` from ``set_epoch``, and read :attr:`current` in
    ``__getitem__``.

    The epoch lives in a ``multiprocessing.Value`` so that
    ``set_epoch()`` in the main process is observed by DataLoader
    workers — including persistent ones (``persistent_workers=True``),
    which are created once and never re-receive the dataset.  Workers
    inherit the shared handle when the DataLoader passes the dataset
    as ``Process`` args, which works under both fork and spawn start
    methods.

    The Value is built from an explicit **spawn** context rather than
    the process default, and that choice is load-bearing: a
    fork-context lock is an anonymous semaphore, unlinked at creation,
    whose handle is meaningless in a spawn-started child — it unpickles
    without complaint and segfaults the worker on first use.  A
    spawn-context lock is named, so it survives both inheritance (fork)
    and reopen-by-name (spawn).  Linux defaults to fork, so the
    mismatch is reachable with a plain
    ``DataLoader(..., multiprocessing_context="spawn")``.

    ``-1`` is the never-set sentinel (replaces a separate boolean).
    Deliberately no ``__getstate__``/``__setstate__``: swapping the
    Value for a plain int during pickling would silently break sharing
    under spawn (worker creation uses the same pickle machinery).  The
    cost is that holders cannot be ``copy.deepcopy``-ed or
    ``torch.save``-ed directly — shared ctypes only travel via process
    inheritance.

    Note that a spawn DataLoader additionally pickles the whole dataset,
    so everything it holds must be picklable — in particular a
    :class:`~pybvh_ml.AugmentationPipeline` whose kwargs are ``lambda``
    callables cannot cross a spawn boundary; use module-level functions
    there.
    """

    def __init__(self) -> None:
        self._epoch = multiprocessing.get_context("spawn").Value("i", -1)
        # Warn-once bookkeeping stays per-process: one warning per
        # worker is acceptable, and a shared flag would need a lock
        # dance for no real benefit.
        self._warned = False

    def set(self, epoch: int) -> None:
        epoch = int(epoch)
        if epoch < 0:
            raise ValueError(f"epoch must be >= 0, got {epoch}")
        with self._epoch.get_lock():
            self._epoch.value = epoch

    def _raw(self) -> int:
        """Stored value, ``-1`` when never set."""
        with self._epoch.get_lock():
            return self._epoch.value

    @property
    def current(self) -> int:
        """Current epoch (``0`` when never set), with no warning.

        For read-only callers: the warn-once budget in :meth:`effective`
        belongs to the training path, and a diagnostic read must not
        spend it and mask the real warning later.
        """
        return max(self._raw(), 0)

    def _effective(self, cls_name: str, seed: int | None,
                   draws_randomness: bool) -> int:
        """Current epoch, warning once if it was never set.

        Internal to the shipped Dataset classes: the warn-once budget is
        theirs to spend, and the signature is shaped for their call
        site.  Outside callers want :attr:`current`.
        """
        epoch = self._raw()
        if epoch >= 0:
            return epoch
        if draws_randomness and seed is not None and not self._warned:
            warnings.warn(
                _MISSING_SET_EPOCH_MSG.format(cls=cls_name, seed=seed),
                UserWarning, stacklevel=3)
            self._warned = True
        return 0


# =========================================================================
# Shared layout / temporal machinery
# =========================================================================

_LAYOUT_PACKERS = {
    "flat": pack_to_flat,
    "ctv": pack_to_ctv,
    "tvc": pack_to_tvc,
}

_TEMPORAL_MODES = ("pad", "crop", "resample", "resample_deterministic")


def _validate_layout_and_temporal(
    layout: str, temporal: str, target_length: int | None, cls_name: str,
) -> None:
    """Reject unknown layout / temporal tokens and no-op combinations."""
    if layout not in _LAYOUT_PACKERS:
        raise ValueError(
            f"{cls_name} layout must be one of {list(_LAYOUT_PACKERS)}, "
            f"got {layout!r}")
    if temporal not in _TEMPORAL_MODES:
        raise ValueError(
            f"{cls_name} temporal must be one of {list(_TEMPORAL_MODES)}, "
            f"got {temporal!r}")
    if temporal != "pad" and target_length is None:
        raise ValueError(
            f"{cls_name} temporal={temporal!r} has no length to "
            f"standardize to — pass target_length=..., or leave temporal "
            f"at its 'pad' default, which is the no-op that a missing "
            f"target_length implies.")


def _apply_temporal(
    root_pos: np.ndarray,
    joint_data: np.ndarray,
    target_length: int | None,
    temporal: str,
    rng: np.random.Generator | None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Standardize a clip to *target_length* frames.

    Runs on the ``(F, 3)`` / ``(F, J, C)`` arrays rather than on the
    packed tensor: ``(C, T, V)`` puts time on axis 1, so standardizing
    before packing keeps one implementation for all three layouts.  For
    the flat layout the two orders are equivalent — packing is a
    per-frame concatenation, and zero padding commutes with it.

    Returns ``(root_pos, joint_data, length)``, where *length* is the
    number of valid (non-padding) frames in the result.
    """
    F = root_pos.shape[0]
    if target_length is None:
        return root_pos, joint_data, F

    if temporal in ("pad", "crop"):
        return (
            standardize_length(root_pos, target_length, method=temporal),
            standardize_length(joint_data, target_length, method=temporal),
            # Both modes discard frames when the clip is too long
            # ("pad" from the end, "crop" from both ends) and zero-pad
            # when it is too short — so report only the valid frames
            # actually present, or collate masks go wrong.
            min(F, target_length),
        )

    if F == 0:
        raise ValueError(
            f"temporal={temporal!r} cannot resample a clip with 0 frames; "
            f"use temporal='pad' to zero-fill it instead.")
    mode = "train" if temporal == "resample" else "test"
    # The deterministic mode gets rng=None rather than the sample's
    # generator.  uniform_temporal_sample honors a supplied rng in test
    # mode too, so threading the augmentation-advanced stream here would
    # make "deterministic" quietly change with the epoch — the exact
    # trap that makes mode="test" alone not a reproducibility guarantee.
    indices = uniform_temporal_sample(
        F, target_length, mode=mode,
        rng=rng if mode == "train" else None) % F
    # Resampling indexes into the clip, so every returned frame is real
    # data: no padding, and the valid length is always the full budget.
    return root_pos[indices], joint_data[indices], target_length


def _finalize(
    root_pos: np.ndarray,
    joint_data: np.ndarray,
    *,
    layout: str,
    temporal: str,
    target_length: int | None,
    rng: np.random.Generator | None,
) -> tuple[torch.Tensor, int]:
    """Temporal standardization then packing — the tail of ``__getitem__``.

    ``center_root=False`` throughout: both Dataset classes have already
    made that choice on the raw arrays, and packing must not silently
    re-center.
    """
    root_pos, joint_data, length = _apply_temporal(
        root_pos, joint_data, target_length, temporal, rng)
    packed = _LAYOUT_PACKERS[layout](root_pos, joint_data, center_root=False)
    return torch.tensor(packed, dtype=torch.float32), length


class MotionDataset(Dataset):
    """Dataset that loads preprocessed motion clips.

    Designed to work with the output of
    :func:`pybvh_ml.preprocessing.load_preprocessed`.

    Parameters
    ----------
    clips : list of dict
        Each dict must have ``root_pos`` (F, 3) and ``joint_data``
        (F, J, C).
    labels : array-like or None
        Per-clip integer labels.
    target_length : int or None
        If given, standardize all clips to this length using
        ``temporal``.  The ``length`` reported by ``__getitem__`` is the
        number of valid frames actually present in the returned tensor,
        so padded frames are excluded.
    temporal : {"pad", "crop", "resample", "resample_deterministic"}
        How ``target_length`` is reached.  These are genuinely different
        operations, not styles: ``"pad"`` (default) and ``"crop"`` keep
        a fixed *window* of the clip — truncating from the end and from
        the center respectively, zero-padding when the clip is shorter —
        while the two ``resample`` modes keep the whole *arc* of the
        clip at a fixed frame budget, sampling ``target_length`` frame
        indices spread across its full duration
        (:func:`~pybvh_ml.sequences.uniform_temporal_sample`).  Resample
        when the shape of the whole clip carries the signal and clips
        vary in duration; crop or pad when a fixed-duration window does.
        ``"resample"`` draws a random offset within each segment (a
        temporal augmentation, and it consumes the sample's rng);
        ``"resample_deterministic"`` takes each segment's first frame,
        so a clip yields the same frames on every read — the evaluation
        counterpart.  Both report ``length == target_length``: every
        returned frame is real data.
    layout : {"flat", "ctv", "tvc"}
        Tensor layout of the returned ``data``.  ``"flat"`` (default)
        gives ``(T, D)`` via
        :func:`~pybvh_ml.packing.pack_to_flat`; ``"ctv"`` and ``"tvc"``
        give the graph layouts ``(C, T, V)`` / ``(T, V, C)`` that GCN
        and skeleton-transformer models consume.  Only ``"flat"`` works
        with :func:`~pybvh_ml.torch.collate_motion_batch`, which pads a
        time-major axis 0; the graph layouts are fixed-size by
        construction (they pair with ``target_length``) so they stack
        with :func:`torch.utils.data.default_collate`.
    source_repr, target_repr : str or None
        Convert each clip's ``joint_data`` from ``source_repr`` to
        ``target_repr`` before augmentation, so a dataset stored in one
        representation can train in another without a second
        preprocessing pass.  Both are required together;
        ``target_repr=None`` (default) returns the stored representation
        untouched.  :meth:`from_preprocessed` fills ``source_repr`` from
        the dataset metadata, which is where it should come from —
        restating it at the call site is how it ends up wrong.
    euler_orders : list of str or None
        Per-joint Euler orders, required when either end of the
        conversion is ``"euler"``.  From
        ``skeleton_info["euler_orders"]``.
    augmentation : AugmentationPipeline or None
        Applied on-the-fly during ``__getitem__``, after any
        ``target_repr`` conversion — so the pipeline's declared
        ``representation`` is ``target_repr`` when one is set.
    center_root : bool
        If True, subtract each clip's first-frame root position in
        ``__getitem__``.  Default ``False`` — clips from
        :func:`~pybvh_ml.preprocessing.load_preprocessed` are already
        centered when the dataset was saved with ``center_root=True``
        (check the loaded ``center_root`` metadata); set ``True`` for
        hand-built raw clip dicts, mirroring
        :class:`OnTheFlyDataset`.
    seed : int or None
        Base seed for reproducible augmentation.  When set, combined
        with the current epoch (see :meth:`set_epoch`) and the sample
        index into a ``SeedSequence`` so each ``(seed, epoch, idx)``
        triple produces a distinct but reproducible stream.  Set
        ``None`` for fresh OS entropy each call.

    Notes
    -----
    **Per-epoch augmentation variety**: call
    ``dataset.set_epoch(epoch)`` at the start of each training epoch
    so the seeded augmentation changes across epochs — same contract
    as :class:`torch.utils.data.distributed.DistributedSampler`.  The
    epoch lives in shared memory, so this works with
    ``num_workers > 0`` including ``persistent_workers=True``.  When
    ``seed`` is set and ``set_epoch`` is never called, every epoch
    sees the same augmentation per sample index (useful for
    debugging, harmful for training dynamics).

    **Pickling**: because of the shared-memory epoch, instances cannot
    be ``copy.deepcopy``-ed or ``torch.save``-ed directly — shared
    state only travels via process inheritance (which is exactly how
    the DataLoader hands the dataset to its workers).
    """

    def __init__(
        self,
        clips: list[dict],
        labels: np.ndarray | None = None,
        target_length: int | None = None,
        augmentation: AugmentationPipeline | None = None,
        seed: int | None = None,
        *,
        center_root: bool = False,
        temporal: str = "pad",
        layout: str = "flat",
        source_repr: str | None = None,
        target_repr: str | None = None,
        euler_orders: list[str] | None = None,
    ) -> None:
        _validate_layout_and_temporal(
            layout, temporal, target_length, "MotionDataset")
        if target_repr is not None and source_repr is None:
            raise ValueError(
                "target_repr requires source_repr — conversion needs to "
                "know what the stored joint_data is in, and clip dicts "
                "don't carry that. Pass source_repr=..., or build the "
                "dataset with MotionDataset.from_preprocessed(loaded), "
                "which reads it from the dataset metadata.")
        self.clips = clips
        self.labels = labels
        self.target_length = target_length
        self.temporal = temporal
        self.layout = layout
        self.source_repr = source_repr
        self.target_repr = target_repr
        self.euler_orders = euler_orders
        self.augmentation = augmentation
        self.center_root = center_root
        self.seed = seed
        self._epoch_state = EpochState()

    @classmethod
    def from_preprocessed(cls, loaded: dict, **kwargs) -> "MotionDataset":
        """Build a dataset from a :func:`~pybvh_ml.preprocessing.load_preprocessed` result.

        Wires the metadata that would otherwise be restated by hand at
        the call site: the clips and labels, the stored
        ``representation`` (as ``source_repr``, which ``target_repr``
        conversion needs), and ``skeleton_info["euler_orders"]``.
        ``center_root`` defaults to ``False`` because the stored arrays
        already reflect the choice made at preprocessing time — centering
        again here would be a second, unrecorded transform.

        Parameters
        ----------
        loaded : dict
            The dict returned by
            :func:`~pybvh_ml.preprocessing.load_preprocessed`.
        **kwargs
            Forwarded to :class:`MotionDataset`; anything passed here
            overrides what the metadata supplies.

        Examples
        --------
        >>> loaded = load_preprocessed("train.npz")          # stored as euler
        >>> ds = MotionDataset.from_preprocessed(
        ...     loaded, target_repr="6d", layout="ctv",
        ...     temporal="resample", target_length=64, seed=0)
        """
        skeleton_info = loaded.get("skeleton_info") or {}
        defaults = {
            "labels": loaded.get("labels"),
            "source_repr": loaded.get("representation"),
            "euler_orders": skeleton_info.get("euler_orders"),
            "center_root": False,
        }
        return cls(loaded["clips"], **{**defaults, **kwargs})

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch for per-epoch reproducible augmentation.

        Mirrors :meth:`torch.utils.data.distributed.DistributedSampler.set_epoch`;
        reaches DataLoader workers (persistent ones included) via
        shared memory.
        """
        self._epoch_state.set(epoch)

    def __len__(self) -> int:
        return len(self.clips)

    def explain_augmentation(self, idx: int, *,
                             epoch: int | None = None) -> list[dict]:
        """Report what the augmentation did to sample *idx*.

        Re-runs this sample's augmentation on the same
        ``(seed, epoch, idx)`` rng the loader used, so the records
        describe the draw that actually ran rather than a fresh one.
        Their layout is the pipeline's ``return_params`` format:
        ``{"name", "applied", "params"}`` per step.

        Parameters
        ----------
        idx : int
            Sample index; negative indexing works as in ``__getitem__``.
        epoch : int, optional
            Epoch to replay.  Defaults to the dataset's current epoch —
            pass it explicitly to ask about an earlier one.

        Returns
        -------
        list of dict
            One record per configured augmentation step, or ``[]`` when
            the dataset has no augmentation.

        Raises
        ------
        ValueError
            If the dataset was built without a ``seed``.  Unseeded draws
            come from fresh OS entropy and cannot be reconstructed;
            answering with a new draw would describe an augmentation
            that never ran.

        Notes
        -----
        The replay is truthful only while its inputs are unchanged: the
        same pipeline (same steps, probabilities and ranges) over the
        same clip arrays.  Rebuild the pipeline differently and the
        records describe a run that no longer exists.
        """
        return _replay_augmentation_params(
            self, idx, epoch, "MotionDataset")

    @property
    def _draws_randomness(self) -> bool:
        """Whether ``__getitem__`` consumes the per-sample rng at all."""
        return self.augmentation is not None or self.temporal == "resample"

    def _clip_arrays(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Sample *idx*'s arrays as the augmentation receives them.

        Shared by ``__getitem__`` and ``explain_augmentation`` so the
        replay can never drift from what the loader actually fed the
        pipeline — which is why the ``target_repr`` conversion lives
        here rather than in ``__getitem__``.
        """
        clip = self.clips[idx]
        root_pos = clip["root_pos"].copy()
        joint_data = clip["joint_data"].copy()
        if self.center_root and root_pos.shape[0] > 0:
            root_pos = root_pos - root_pos[0:1]
        if self.target_repr is not None:
            joint_data = convert_arrays(
                joint_data, self.source_repr, self.target_repr,
                euler_orders=self.euler_orders)
        return root_pos, joint_data

    def __getitem__(self, idx: int) -> dict:
        idx = _normalize_index(idx, len(self.clips), "MotionDataset")
        root_pos, joint_data = self._clip_arrays(idx)

        rng = None
        if self._draws_randomness:
            epoch = self._epoch_state._effective(
                "MotionDataset", self.seed, True)
            rng = rng_for(self.seed, epoch, idx)

        # Augmentation draws first and temporal resampling continues on
        # the same generator.  That order is what lets
        # explain_augmentation replay a sample exactly: it re-runs only
        # the augmentation, from the head of an identical stream.
        if self.augmentation is not None:
            root_pos, joint_data = self.augmentation(
                root_pos=root_pos, joint_data=joint_data, rng=rng)

        tensor, length = _finalize(
            root_pos, joint_data, layout=self.layout,
            temporal=self.temporal, target_length=self.target_length,
            rng=rng)

        result: dict = {"data": tensor, "length": length}
        if self.labels is not None:
            result["label"] = int(self.labels[idx])
        return result


class OnTheFlyDataset(Dataset):
    """Dataset that loads BVH files on-the-fly for maximum augmentation variety.

    Slower than :class:`MotionDataset` but avoids pre-extracting arrays,
    so every epoch sees freshly augmented data.

    Parameters
    ----------
    bvh_paths : list of str or Path
        Paths to BVH files (coerced to :class:`~pathlib.Path`).
    representation : str
        Rotation representation for joint data.  Extraction happens per
        clip, so this is already the "target" representation —
        :class:`MotionDataset`'s ``source_repr`` / ``target_repr`` pair
        has no counterpart here.
    target_length : int or None
        If given, standardize to this length using ``temporal``.  The
        reported ``length`` counts only the valid frames present in the
        returned tensor (see :class:`MotionDataset`).
    temporal : {"pad", "crop", "resample", "resample_deterministic"}
        How ``target_length`` is reached — see :class:`MotionDataset`.
    layout : {"flat", "ctv", "tvc"}
        Tensor layout of the returned ``data`` — see
        :class:`MotionDataset`.
    augmentation : AugmentationPipeline or None
    center_root : bool
        If True (default), subtract each clip's first-frame root
        position after extraction.
    label_fn : callable or None
        ``label_fn(filename_stem) -> int``.
    world_up : str
        Forwarded to :func:`pybvh.read_bvh_file` per clip.  ``"auto"``
        (default) auto-detects; pass ``"+y"`` etc. to override — same
        semantics as :func:`~pybvh_ml.preprocessing.preprocess_directory`.
    lr_mapping : dict or None
        Forwarded to :func:`pybvh.read_bvh_file`.  Explicit left/right
        joint pair mapping for uniform dataset conventions.
    seed : int or None
        See :class:`MotionDataset` for seeding semantics.  Call
        :meth:`set_epoch` at the start of each epoch for reproducible
        per-epoch variety — reaches DataLoader workers (persistent
        ones included) via shared memory; see :class:`MotionDataset`
        for the pickling caveat.
    """

    def __init__(
        self,
        bvh_paths: list[str | Path],
        representation: str = "6d",
        target_length: int | None = None,
        augmentation: AugmentationPipeline | None = None,
        center_root: bool = True,
        label_fn: Callable[[str], int] | None = None,
        seed: int | None = None,
        *,
        world_up: str = "auto",
        lr_mapping: dict[str, str] | None = None,
        temporal: str = "pad",
        layout: str = "flat",
    ) -> None:
        _validate_layout_and_temporal(
            layout, temporal, target_length, "OnTheFlyDataset")
        self.bvh_paths = [Path(p) for p in bvh_paths]
        self.representation = representation
        self.target_length = target_length
        self.temporal = temporal
        self.layout = layout
        self.augmentation = augmentation
        self.center_root = center_root
        self.label_fn = label_fn
        self.world_up = world_up
        self.lr_mapping = lr_mapping
        self.seed = seed
        self._epoch_state = EpochState()

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch for reproducible per-epoch augmentation."""
        self._epoch_state.set(epoch)

    def __len__(self) -> int:
        return len(self.bvh_paths)

    def explain_augmentation(self, idx: int, *,
                             epoch: int | None = None) -> list[dict]:
        """Report what the augmentation did to sample *idx*.

        Re-runs this sample's augmentation on the same
        ``(seed, epoch, idx)`` rng the loader used, so the records
        describe the draw that actually ran rather than a fresh one.
        Their layout is the pipeline's ``return_params`` format:
        ``{"name", "applied", "params"}`` per step.  The source file is
        re-read, so this costs a parse per call.

        Parameters
        ----------
        idx : int
            Sample index; negative indexing works as in ``__getitem__``.
        epoch : int, optional
            Epoch to replay.  Defaults to the dataset's current epoch —
            pass it explicitly to ask about an earlier one.

        Returns
        -------
        list of dict
            One record per configured augmentation step, or ``[]`` when
            the dataset has no augmentation.

        Raises
        ------
        ValueError
            If the dataset was built without a ``seed``.  Unseeded draws
            come from fresh OS entropy and cannot be reconstructed;
            answering with a new draw would describe an augmentation
            that never ran.

        Notes
        -----
        The replay is truthful only while its inputs are unchanged: the
        same pipeline (same steps, probabilities and ranges) over the
        same source file.  Edit the BVH on disk and the records describe
        a run that no longer exists.
        """
        return _replay_augmentation_params(
            self, idx, epoch, "OnTheFlyDataset")

    @property
    def _draws_randomness(self) -> bool:
        """Whether ``__getitem__`` consumes the per-sample rng at all."""
        return self.augmentation is not None or self.temporal == "resample"

    def _clip_arrays(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Sample *idx*'s arrays as the augmentation receives them.

        Re-reads the source file, so ``explain_augmentation`` replays
        against exactly what ``__getitem__`` loads.
        """
        bvh = read_bvh_file(
            self.bvh_paths[idx], world_up=self.world_up,
            lr_mapping=self.lr_mapping)
        root_pos, joint_data = extract_repr(bvh, self.representation)
        if self.center_root and root_pos.shape[0] > 0:
            root_pos = root_pos - root_pos[0:1]
        return root_pos, joint_data

    def __getitem__(self, idx: int) -> dict:
        idx = _normalize_index(idx, len(self.bvh_paths), "OnTheFlyDataset")
        root_pos, joint_data = self._clip_arrays(idx)

        rng = None
        if self._draws_randomness:
            epoch = self._epoch_state._effective(
                "OnTheFlyDataset", self.seed, True)
            rng = rng_for(self.seed, epoch, idx)

        # Draw order matters — see the note in MotionDataset.__getitem__.
        if self.augmentation is not None:
            root_pos, joint_data = self.augmentation(
                root_pos=root_pos, joint_data=joint_data, rng=rng)

        tensor, length = _finalize(
            root_pos, joint_data, layout=self.layout,
            temporal=self.temporal, target_length=self.target_length,
            rng=rng)

        result: dict = {"data": tensor, "length": length}
        if self.label_fn is not None:
            result["label"] = self.label_fn(self.bvh_paths[idx].stem)
        return result
