"""PyTorch Dataset classes for motion capture data."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from pybvh import read_bvh_file
from pybvh_ml.packing import pack_to_flat
from pybvh_ml.sequences import standardize_length
from pybvh_ml.pipeline import AugmentationPipeline


class MotionDataset(Dataset):
    """Dataset that loads preprocessed motion clips.

    Designed to work with the output of
    :func:`pybvh_ml.preprocessing.load_preprocessed`.

    Parameters
    ----------
    clips : list of dict
        Each dict must have ``root_pos`` (F, 3) and ``joint_data``
        (F, J, C).  Optionally ``joint_quats`` (F, J, 4).
    labels : array-like or None
        Per-clip integer labels.
    target_length : int or None
        If given, crop/pad all clips to this length.
    augmentation : AugmentationPipeline or None
        Applied on-the-fly during ``__getitem__``.
    use_quats_for_augmentation : bool
        If True and clips have ``joint_quats``, pass quaternions
        to the augmentation pipeline instead of ``joint_data``.
    seed : int or None
        Base seed for reproducible augmentation.
    """

    def __init__(
        self,
        clips: list[dict],
        labels: np.ndarray | None = None,
        target_length: int | None = None,
        augmentation: AugmentationPipeline | None = None,
        use_quats_for_augmentation: bool = False,
        seed: int | None = None,
    ) -> None:
        self.clips = clips
        self.labels = labels
        self.target_length = target_length
        self.augmentation = augmentation
        self.use_quats_for_augmentation = use_quats_for_augmentation
        self.seed = seed

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> dict:
        clip = self.clips[idx]
        root_pos = clip["root_pos"].copy()
        joint_data = clip["joint_data"].copy()

        # Augmentation
        if self.augmentation is not None:
            rng = None
            if self.seed is not None:
                rng = np.random.default_rng(self.seed + idx)
            if self.use_quats_for_augmentation and "joint_quats" in clip:
                quats = clip["joint_quats"].copy()
                quats, root_pos = self.augmentation(quats, root_pos, rng=rng)
            else:
                joint_data, root_pos = self.augmentation(
                    joint_data, root_pos, rng=rng)

        # Pack to flat
        flat = pack_to_flat(root_pos, joint_data, center_root=False)

        # Standardize length
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

    def __len__(self) -> int:
        return len(self.bvh_paths)

    def __getitem__(self, idx: int) -> dict:
        from pybvh_ml.preprocessing import _extract_repr

        bvh = read_bvh_file(str(self.bvh_paths[idx]))
        root_pos, joint_data = _extract_repr(bvh, self.representation)

        if self.center_root and root_pos.shape[0] > 0:
            root_pos = root_pos - root_pos[0:1]

        if self.augmentation is not None:
            rng = None
            if self.seed is not None:
                rng = np.random.default_rng(self.seed + idx)
            joint_data, root_pos = self.augmentation(
                joint_data, root_pos, rng=rng)

        flat = pack_to_flat(root_pos, joint_data, center_root=False)

        length = flat.shape[0]
        if self.target_length is not None:
            flat = standardize_length(flat, self.target_length, method="pad")

        tensor = torch.tensor(flat, dtype=torch.float32)

        result: dict = {"data": tensor, "length": length}
        if self.label_fn is not None:
            result["label"] = self.label_fn(self.bvh_paths[idx].stem)
        return result
