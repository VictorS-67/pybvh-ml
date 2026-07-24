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
from pybvh_ml.packing import pack_to_flat
from pybvh_ml.preprocessing import extract_repr
from pybvh_ml.sequences import standardize_length
from pybvh_ml.pipeline import AugmentationPipeline


def _compose_rng(
    seed: int | None,
    epoch: int,
    idx: int,
) -> np.random.Generator:
    """Build a per-sample rng for the current (seed, epoch, idx)."""
    if seed is None:
        return np.random.default_rng()
    ss = np.random.SeedSequence([int(seed), int(epoch), int(idx)])
    return np.random.default_rng(ss)


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
    "every epoch will produce identical augmentation per sample. "
    "Call dataset.set_epoch(epoch) at the start of each epoch, or pass "
    "seed=None for fresh OS entropy each call.")


class _EpochState:
    """Shared-memory epoch counter for DataLoader-worker visibility.

    The epoch lives in a ``multiprocessing.Value`` so that
    ``set_epoch()`` in the main process is observed by DataLoader
    workers — including persistent ones (``persistent_workers=True``),
    which are created once and never re-receive the dataset.  Workers
    inherit the shared handle when the DataLoader passes the dataset
    as ``Process`` args, which works under both fork and spawn start
    methods.

    ``-1`` is the never-set sentinel (replaces a separate boolean).
    Deliberately no ``__getstate__``/``__setstate__``: swapping the
    Value for a plain int during pickling would silently break sharing
    under spawn (worker creation uses the same pickle machinery).  The
    cost is that holders cannot be ``copy.deepcopy``-ed or
    ``torch.save``-ed directly — shared ctypes only travel via process
    inheritance.
    """

    def __init__(self) -> None:
        self._epoch = multiprocessing.Value("i", -1)
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

    def effective(self, cls_name: str, seed: int | None,
                  augmentation_active: bool) -> int:
        """Current epoch, warning once if it was never set."""
        with self._epoch.get_lock():
            epoch = self._epoch.value
        if epoch >= 0:
            return epoch
        if augmentation_active and seed is not None and not self._warned:
            warnings.warn(
                _MISSING_SET_EPOCH_MSG.format(cls=cls_name, seed=seed),
                UserWarning, stacklevel=3)
            self._warned = True
        return 0


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
        If given, crop/pad all clips to this length.  The ``length`` reported by ``__getitem__`` is the number of valid frames actually present in the returned tensor — ``min(original_length, target_length)`` — so padded frames are excluded and cropped clips report ``target_length``.
    augmentation : AugmentationPipeline or None
        Applied on-the-fly during ``__getitem__``.
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
    ) -> None:
        self.clips = clips
        self.labels = labels
        self.target_length = target_length
        self.augmentation = augmentation
        self.center_root = center_root
        self.seed = seed
        self._epoch_state = _EpochState()

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch for per-epoch reproducible augmentation.

        Mirrors :meth:`torch.utils.data.distributed.DistributedSampler.set_epoch`;
        reaches DataLoader workers (persistent ones included) via
        shared memory.
        """
        self._epoch_state.set(epoch)

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> dict:
        idx = _normalize_index(idx, len(self.clips), "MotionDataset")
        clip = self.clips[idx]
        root_pos = clip["root_pos"].copy()
        joint_data = clip["joint_data"].copy()

        if self.center_root and root_pos.shape[0] > 0:
            root_pos = root_pos - root_pos[0:1]

        if self.augmentation is not None:
            epoch = self._epoch_state.effective(
                "MotionDataset", self.seed, True)
            rng = _compose_rng(self.seed, epoch, idx)
            root_pos, joint_data = self.augmentation(
                root_pos=root_pos, joint_data=joint_data, rng=rng)

        flat = pack_to_flat(root_pos, joint_data, center_root=False)

        length = flat.shape[0]
        if self.target_length is not None:
            flat = standardize_length(flat, self.target_length, method="pad")
            # Cropping discards trailing frames — report only the valid
            # frames actually present so collate masks stay correct.
            length = min(length, self.target_length)

        tensor = torch.tensor(flat, dtype=torch.float32)

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
        Rotation representation for joint data.
    target_length : int or None
        If given, crop/pad to this length.  The reported ``length`` counts only the valid frames present in the returned tensor (see :class:`MotionDataset`).
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
    ) -> None:
        self.bvh_paths = [Path(p) for p in bvh_paths]
        self.representation = representation
        self.target_length = target_length
        self.augmentation = augmentation
        self.center_root = center_root
        self.label_fn = label_fn
        self.world_up = world_up
        self.lr_mapping = lr_mapping
        self.seed = seed
        self._epoch_state = _EpochState()

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch for reproducible per-epoch augmentation."""
        self._epoch_state.set(epoch)

    def __len__(self) -> int:
        return len(self.bvh_paths)

    def __getitem__(self, idx: int) -> dict:
        idx = _normalize_index(idx, len(self.bvh_paths), "OnTheFlyDataset")
        bvh = read_bvh_file(
            self.bvh_paths[idx], world_up=self.world_up,
            lr_mapping=self.lr_mapping)
        root_pos, joint_data = extract_repr(bvh, self.representation)

        if self.center_root and root_pos.shape[0] > 0:
            root_pos = root_pos - root_pos[0:1]

        if self.augmentation is not None:
            epoch = self._epoch_state.effective(
                "OnTheFlyDataset", self.seed, True)
            rng = _compose_rng(self.seed, epoch, idx)
            root_pos, joint_data = self.augmentation(
                root_pos=root_pos, joint_data=joint_data, rng=rng)

        flat = pack_to_flat(root_pos, joint_data, center_root=False)

        length = flat.shape[0]
        if self.target_length is not None:
            flat = standardize_length(flat, self.target_length, method="pad")
            # Cropping discards trailing frames — report only the valid
            # frames actually present so collate masks stay correct.
            length = min(length, self.target_length)

        tensor = torch.tensor(flat, dtype=torch.float32)

        result: dict = {"data": tensor, "length": length}
        if self.label_fn is not None:
            result["label"] = self.label_fn(self.bvh_paths[idx].stem)
        return result
