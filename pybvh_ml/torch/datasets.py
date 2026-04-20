"""PyTorch Dataset classes for motion capture data."""
from __future__ import annotations

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


_MISSING_SET_EPOCH_MSG = (
    "{cls} was seeded (seed={seed!r}) but set_epoch() was never called; "
    "every epoch will produce identical augmentation per sample. "
    "Call dataset.set_epoch(epoch) at the start of each epoch, or pass "
    "seed=None for fresh OS entropy each call.")


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
        If given, crop/pad all clips to this length.
    augmentation : AugmentationPipeline or None
        Applied on-the-fly during ``__getitem__``.
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
    as :class:`torch.utils.data.distributed.DistributedSampler`.  When
    ``seed`` is set and ``set_epoch`` is never called, every epoch
    sees the same augmentation per sample index (useful for
    debugging, harmful for training dynamics).
    """

    def __init__(
        self,
        clips: list[dict],
        labels: np.ndarray | None = None,
        target_length: int | None = None,
        augmentation: AugmentationPipeline | None = None,
        seed: int | None = None,
    ) -> None:
        self.clips = clips
        self.labels = labels
        self.target_length = target_length
        self.augmentation = augmentation
        self.seed = seed
        self._epoch = 0
        self._epoch_set = False
        self._warned_missing_set_epoch = False

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch for per-epoch reproducible augmentation.

        Mirrors :meth:`torch.utils.data.distributed.DistributedSampler.set_epoch`.
        """
        self._epoch = int(epoch)
        self._epoch_set = True

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> dict:
        clip = self.clips[idx]
        root_pos = clip["root_pos"].copy()
        joint_data = clip["joint_data"].copy()

        if (self.augmentation is not None
                and self.seed is not None
                and not self._epoch_set
                and not self._warned_missing_set_epoch):
            warnings.warn(
                _MISSING_SET_EPOCH_MSG.format(
                    cls="MotionDataset", seed=self.seed),
                UserWarning, stacklevel=2)
            self._warned_missing_set_epoch = True

        if self.augmentation is not None:
            rng = _compose_rng(self.seed, self._epoch, idx)
            root_pos, joint_data = self.augmentation(
                root_pos=root_pos, joint_data=joint_data, rng=rng)

        flat = pack_to_flat(root_pos, joint_data, center_root=False)

        length = flat.shape[0]
        if self.target_length is not None:
            flat = standardize_length(flat, self.target_length, method="pad")

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
    bvh_paths : list of Path
        Paths to BVH files.
    representation : str
        Rotation representation for joint data.
    target_length : int or None
        If given, crop/pad to this length.
    augmentation : AugmentationPipeline or None
    center_root : bool
    label_fn : callable or None
        ``label_fn(filename_stem) -> int``.
    seed : int or None
        See :class:`MotionDataset` for seeding semantics.  Call
        :meth:`set_epoch` at the start of each epoch for reproducible
        per-epoch variety.
    """

    def __init__(
        self,
        bvh_paths: list[Path],
        representation: str = "6d",
        target_length: int | None = None,
        augmentation: AugmentationPipeline | None = None,
        center_root: bool = True,
        label_fn: Callable[[str], int] | None = None,
        seed: int | None = None,
    ) -> None:
        self.bvh_paths = bvh_paths
        self.representation = representation
        self.target_length = target_length
        self.augmentation = augmentation
        self.center_root = center_root
        self.label_fn = label_fn
        self.seed = seed
        self._epoch = 0
        self._epoch_set = False
        self._warned_missing_set_epoch = False

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch for reproducible per-epoch augmentation."""
        self._epoch = int(epoch)
        self._epoch_set = True

    def __len__(self) -> int:
        return len(self.bvh_paths)

    def __getitem__(self, idx: int) -> dict:
        bvh = read_bvh_file(str(self.bvh_paths[idx]))
        root_pos, joint_data = extract_repr(bvh, self.representation)

        if self.center_root and root_pos.shape[0] > 0:
            root_pos = root_pos - root_pos[0:1]

        if (self.augmentation is not None
                and self.seed is not None
                and not self._epoch_set
                and not self._warned_missing_set_epoch):
            warnings.warn(
                _MISSING_SET_EPOCH_MSG.format(
                    cls="OnTheFlyDataset", seed=self.seed),
                UserWarning, stacklevel=2)
            self._warned_missing_set_epoch = True

        if self.augmentation is not None:
            rng = _compose_rng(self.seed, self._epoch, idx)
            root_pos, joint_data = self.augmentation(
                root_pos=root_pos, joint_data=joint_data, rng=rng)

        flat = pack_to_flat(root_pos, joint_data, center_root=False)

        length = flat.shape[0]
        if self.target_length is not None:
            flat = standardize_length(flat, self.target_length, method="pad")

        tensor = torch.tensor(flat, dtype=torch.float32)

        result: dict = {"data": tensor, "length": length}
        if self.label_fn is not None:
            result["label"] = self.label_fn(self.bvh_paths[idx].stem)
        return result
