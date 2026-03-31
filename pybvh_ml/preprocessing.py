"""Batch preprocessing of BVH directories into ML-ready datasets.

Converts a directory of BVH files into on-disk arrays (``npz`` or
``hdf5``) with skeleton metadata and normalization statistics.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import numpy.typing as npt

from pybvh import Bvh, read_bvh_directory, compute_normalization_stats
from pybvh_ml.skeleton import get_skeleton_info


# Maps representation names to Bvh extraction methods
_REPR_EXTRACTORS = {
    "euler": lambda bvh: (bvh.root_pos.copy(), bvh.joint_angles.copy()),
    "quaternion": lambda bvh: (
        bvh.get_frames_as_quaternion()[0],
        bvh.get_frames_as_quaternion()[1]),
    "6d": lambda bvh: (
        bvh.get_frames_as_6d()[0],
        bvh.get_frames_as_6d()[1]),
    "axisangle": lambda bvh: (
        bvh.get_frames_as_axisangle()[0],
        bvh.get_frames_as_axisangle()[1]),
}


def _extract_repr(bvh: Bvh, representation: str) -> tuple[
    npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Extract (root_pos, joint_data) for a given representation."""
    if representation not in _REPR_EXTRACTORS:
        raise ValueError(
            f"Unknown representation '{representation}'. "
            f"Choose from {list(_REPR_EXTRACTORS)}")
    return _REPR_EXTRACTORS[representation](bvh)


def preprocess_directory(
    bvh_dir: str | Path,
    output_path: str | Path,
    representation: str = "6d",
    center_root: bool = True,
    include_quaternions: bool = False,
    label_fn: Callable[[str], int] | None = None,
    file_pattern: str = "*.bvh",
) -> dict:
    """Convert a directory of BVH files to an on-disk dataset.

    Parameters
    ----------
    bvh_dir : path-like
        Directory containing BVH files.
    output_path : path-like
        Output file path.  Extension determines format:
        ``.npz`` (always available) or ``.hdf5`` (requires h5py).
    representation : str
        Rotation representation for joint data.
    center_root : bool
        If True, subtract first frame's root position per clip.
    include_quaternions : bool
        If True, also store pre-computed quaternion arrays per clip
        (useful for runtime speed perturbation / dropout).
    label_fn : callable, optional
        ``label_fn(filename_stem) -> int``.  If provided, stores
        per-clip integer labels.
    file_pattern : str
        Glob pattern for BVH files (default ``"*.bvh"``).

    Returns
    -------
    dict
        Summary with keys: ``num_clips``, ``representation``,
        ``filenames``, ``skeleton_info``.
    """
    bvh_dir = Path(bvh_dir)
    output_path = Path(output_path)

    clips = read_bvh_directory(str(bvh_dir), pattern=file_pattern, sort=True)
    if len(clips) == 0:
        raise ValueError(f"No BVH files found in {bvh_dir} with pattern '{file_pattern}'")

    filenames = sorted(bvh_dir.glob(file_pattern))
    stems = [f.stem for f in filenames]

    # Extract data per clip
    all_root_pos: list[npt.NDArray[np.float64]] = []
    all_joint_data: list[npt.NDArray[np.float64]] = []
    all_joint_quats: list[npt.NDArray[np.float64]] = []

    for bvh in clips:
        root_pos, joint_data = _extract_repr(bvh, representation)
        if center_root and root_pos.shape[0] > 0:
            root_pos = root_pos - root_pos[0:1]
        all_root_pos.append(root_pos)
        all_joint_data.append(joint_data)

        if include_quaternions:
            _, quats, _ = bvh.get_frames_as_quaternion()
            all_joint_quats.append(quats)

    # Skeleton info from first clip
    skel_info = get_skeleton_info(clips[0])

    # Normalization stats
    stats = compute_normalization_stats(clips, representation=representation)

    # Labels
    labels = None
    if label_fn is not None:
        labels = np.array([label_fn(s) for s in stems], dtype=np.int64)

    # Save
    ext = output_path.suffix.lower()
    if ext == ".hdf5" or ext == ".h5":
        _save_hdf5(output_path, all_root_pos, all_joint_data,
                    all_joint_quats, labels, stats, skel_info,
                    representation, stems)
    else:
        _save_npz(output_path, all_root_pos, all_joint_data,
                   all_joint_quats, labels, stats, skel_info,
                   representation, stems)

    return {
        "num_clips": len(clips),
        "representation": representation,
        "filenames": stems,
        "skeleton_info": skel_info,
    }


def _save_npz(
    path: Path,
    root_pos_list: list,
    joint_data_list: list,
    joint_quats_list: list,
    labels: npt.NDArray | None,
    stats: dict,
    skel_info: dict,
    representation: str,
    stems: list[str],
) -> None:
    """Save to .npz format."""
    save_dict: dict[str, object] = {
        "num_clips": np.array(len(root_pos_list)),
        "representation": np.array(representation),
        "filenames": np.array(stems),
        "mean": stats["mean"],
        "std": stats["std"],
        "skeleton_info_json": np.array(json.dumps(skel_info)),
    }
    for i, (rp, jd) in enumerate(zip(root_pos_list, joint_data_list)):
        save_dict[f"clip_{i}_root_pos"] = rp
        save_dict[f"clip_{i}_joint_data"] = jd
    if joint_quats_list:
        for i, jq in enumerate(joint_quats_list):
            save_dict[f"clip_{i}_joint_quats"] = jq
    if labels is not None:
        save_dict["labels"] = labels
    np.savez(path, **save_dict)


def _save_hdf5(
    path: Path,
    root_pos_list: list,
    joint_data_list: list,
    joint_quats_list: list,
    labels: npt.NDArray | None,
    stats: dict,
    skel_info: dict,
    representation: str,
    stems: list[str],
) -> None:
    """Save to HDF5 format."""
    try:
        import h5py
    except ImportError:
        raise ImportError(
            "h5py is required for HDF5 output. Install with: pip install h5py")

    with h5py.File(path, "w") as f:
        f.attrs["num_clips"] = len(root_pos_list)
        f.attrs["representation"] = representation
        f.attrs["skeleton_info_json"] = json.dumps(skel_info)

        f.create_dataset("mean", data=stats["mean"])
        f.create_dataset("std", data=stats["std"])
        f.create_dataset("filenames", data=np.array(stems, dtype="S"))

        if labels is not None:
            f.create_dataset("labels", data=labels)

        for i, (rp, jd) in enumerate(zip(root_pos_list, joint_data_list)):
            grp = f.create_group(f"clip_{i}")
            grp.create_dataset("root_pos", data=rp)
            grp.create_dataset("joint_data", data=jd)
            grp.attrs["filename"] = stems[i]
            if joint_quats_list:
                grp.create_dataset("joint_quats", data=joint_quats_list[i])


def load_preprocessed(path: str | Path) -> dict:
    """Load a preprocessed dataset from disk.

    Parameters
    ----------
    path : path-like
        Path to ``.npz`` or ``.hdf5`` file.

    Returns
    -------
    dict
        Keys: ``clips`` (list of dicts with ``root_pos``,
        ``joint_data``, optionally ``joint_quats``), ``labels``,
        ``mean``, ``std``, ``skeleton_info``, ``representation``,
        ``filenames``.
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".hdf5" or ext == ".h5":
        return _load_hdf5(path)
    else:
        return _load_npz(path)


def _load_npz(path: Path) -> dict:
    """Load from .npz format."""
    data = np.load(path, allow_pickle=False)
    num_clips = int(data["num_clips"])
    representation = str(data["representation"])
    filenames = list(data["filenames"])
    skel_info = json.loads(str(data["skeleton_info_json"]))

    clips = []
    for i in range(num_clips):
        clip: dict[str, npt.NDArray[np.float64]] = {
            "root_pos": data[f"clip_{i}_root_pos"],
            "joint_data": data[f"clip_{i}_joint_data"],
        }
        quat_key = f"clip_{i}_joint_quats"
        if quat_key in data:
            clip["joint_quats"] = data[quat_key]
        clips.append(clip)

    result: dict = {
        "clips": clips,
        "mean": data["mean"],
        "std": data["std"],
        "skeleton_info": skel_info,
        "representation": representation,
        "filenames": filenames,
    }
    if "labels" in data:
        result["labels"] = data["labels"]
    else:
        result["labels"] = None

    return result


def _load_hdf5(path: Path) -> dict:
    """Load from HDF5 format."""
    try:
        import h5py
    except ImportError:
        raise ImportError(
            "h5py is required for HDF5 loading. Install with: pip install h5py")

    with h5py.File(path, "r") as f:
        num_clips = int(f.attrs["num_clips"])
        representation = str(f.attrs["representation"])
        skel_info = json.loads(str(f.attrs["skeleton_info_json"]))

        clips = []
        for i in range(num_clips):
            grp = f[f"clip_{i}"]
            clip: dict[str, npt.NDArray[np.float64]] = {
                "root_pos": grp["root_pos"][()],
                "joint_data": grp["joint_data"][()],
            }
            if "joint_quats" in grp:
                clip["joint_quats"] = grp["joint_quats"][()]
            clips.append(clip)

        result: dict = {
            "clips": clips,
            "mean": f["mean"][()],
            "std": f["std"][()],
            "skeleton_info": skel_info,
            "representation": representation,
            "filenames": [s.decode() if isinstance(s, bytes) else s
                          for s in f["filenames"][()]],
        }
        if "labels" in f:
            result["labels"] = f["labels"][()]
        else:
            result["labels"] = None

    return result
