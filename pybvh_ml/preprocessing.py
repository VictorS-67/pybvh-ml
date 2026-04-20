"""Batch preprocessing of BVH directories into ML-ready datasets.

Converts a directory of BVH files into on-disk arrays (``npz`` or
``hdf5``) with skeleton metadata and normalization statistics.
"""
from __future__ import annotations

import json
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np
import numpy.typing as npt

from pybvh import Bvh, read_bvh_file, compute_normalization_stats
from pybvh import rotations
from pybvh_ml.skeleton import get_skeleton_info


def extract_repr(
    bvh: Bvh,
    representation: str,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Extract ``(root_pos, joint_data)`` for the given representation.

    Thin dispatcher over pybvh's ``to_*`` methods; exposed publicly so
    the PyTorch datasets can reuse the same mapping without reaching
    into a private symbol.

    Parameters
    ----------
    bvh : Bvh
    representation : {"euler", "quaternion", "6d", "axisangle"}

    Returns
    -------
    root_pos : ndarray, shape (F, 3)
    joint_data : ndarray, shape (F, J, C_repr)
    """
    if representation == "euler":
        return bvh.root_pos.copy(), bvh.joint_angles.copy()
    if representation == "quaternion":
        return bvh.to_quaternions()
    if representation == "6d":
        return bvh.to_6d()
    if representation == "axisangle":
        return bvh.to_axisangle()
    raise ValueError(
        f"Unknown representation '{representation}'. "
        f"Choose from ['euler', 'quaternion', '6d', 'axisangle']")


def _extract_primary_and_quats(
    bvh: Bvh,
    representation: str,
    want_quaternions: bool,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64] | None,
]:
    """Extract primary rotation data and (optionally) quaternion secondary.

    When both are needed and both pivot through rotmat (``"6d"`` /
    ``"axisangle"`` primary), shares a single ``to_rotmat()`` call
    and reuses the rotation-matrix array for both derivations.

    Returns
    -------
    root_pos, joint_data, joint_quats_or_None
    """
    if not want_quaternions:
        root_pos, joint_data = extract_repr(bvh, representation)
        return root_pos, joint_data, None

    if representation == "quaternion":
        root_pos, joint_data = bvh.to_quaternions()
        return root_pos, joint_data, joint_data

    if representation == "euler":
        # Euler doesn't compute FK, so call to_quaternions separately.
        root_pos, joint_data = extract_repr(bvh, "euler")
        _, quats = bvh.to_quaternions()
        return root_pos, joint_data, quats

    # 6D and axisangle both pivot through rotmat — share one FK pass.
    root_pos, R = bvh.to_rotmat()
    quats = rotations.rotmat_to_quat(R)
    if representation == "6d":
        joint_data = rotations.rotmat_to_rot6d(R)
    elif representation == "axisangle":
        joint_data = rotations.rotmat_to_axisangle(R)
    else:
        raise ValueError(
            f"Unknown representation '{representation}' for shared "
            f"rotmat extraction")
    return root_pos, joint_data, quats


def _load_one(
    path: Path,
    world_up: str,
    lr_mapping: dict[str, str] | None,
    skip_errors: bool,
) -> Bvh | None:
    """Load one BVH, honoring ``skip_errors``.

    Passes ``warn_on_world_up_disagreement=False`` so pybvh does not emit a
    per-file ``UserWarning`` for rest-vs-animation up-axis disagreement.
    :func:`preprocess_directory` detects the same condition itself post-load
    and emits one aggregated warning for the whole batch (and records it
    under ``uniformity["rest_up"]``).
    """
    try:
        return read_bvh_file(
            path, world_up=world_up, lr_mapping=lr_mapping,
            warn_on_world_up_disagreement=False)
    except Exception as e:
        if not skip_errors:
            raise
        warnings.warn(
            f"preprocess_directory: skipping {path} "
            f"({type(e).__name__}: {e})",
            stacklevel=3)
        return None


def _compute_uniformity(
    clips: list[Bvh], stems: list[str],
) -> dict[str, dict[str, list[str]]]:
    """Group filenames by their world_up, rest-pose forward, and rest-pose up.

    Returned structure::

        {
          "world_up":     {"+z": [stem, ...], "+y": [stem, ...]},
          "rest_forward": {"+y": [stem, ...], "+x": [stem, ...]},
          "rest_up":      {"+z": [stem, ...], "+y": [stem, ...]},
          "rest_anim_mismatch": [stem, ...],  # rest_up != world_up
        }

    ``rest_anim_mismatch`` captures files whose rest-pose up axis
    disagrees with the animation-derived ``world_up`` — the condition
    pybvh warns about per-file at load.  Such files silently corrupt
    training tensors across every rotation representation; pass
    ``target_rest_up`` to reorient them at load.
    """
    world_up: dict[str, list[str]] = {}
    rest_forward: dict[str, list[str]] = {}
    rest_up: dict[str, list[str]] = {}
    rest_anim_mismatch: list[str] = []
    for stem, b in zip(stems, clips):
        anim_up = b.world_up
        r_up = b.rest_up
        world_up.setdefault(anim_up, []).append(stem)
        rest_forward.setdefault(b.rest_forward, []).append(stem)
        rest_up.setdefault(r_up, []).append(stem)
        if r_up != anim_up:
            rest_anim_mismatch.append(stem)
    return {
        "world_up": world_up,
        "rest_forward": rest_forward,
        "rest_up": rest_up,
        "rest_anim_mismatch": rest_anim_mismatch,
    }


def _warn_if_heterogeneous(
    uniformity: dict,
    target_world_up: str | None,
    target_rest_forward: str | None,
    target_rest_up: str | None,
) -> None:
    """Emit one aggregated warning per heterogeneous axis.

    Skips a category when its corresponding ``target_*`` kwarg is set
    (the user has already signaled intent to uniformize).
    """
    def _format(values: dict) -> str:
        """Render '{+z: 900, +y: 100}; first examples per minority: ...'."""
        majority = max(values, key=lambda k: len(values[k]))
        parts = [f"{v!r}: {len(values[v])}" for v in values]
        dist = ", ".join(parts)
        examples = []
        for v, names in values.items():
            if v == majority:
                continue
            examples.append(f"{v!r} e.g. {names[:3]}")
        return f"distribution {{{dist}}}; {'; '.join(examples)}"

    if len(uniformity["world_up"]) > 1 and target_world_up is None:
        warnings.warn(
            "World-up axis is not uniform across the dataset — "
            f"{_format(uniformity['world_up'])}. Pass "
            "target_world_up='<axis>' to harmonize.",
            UserWarning, stacklevel=3)

    if len(uniformity["rest_forward"]) > 1 and target_rest_forward is None:
        warnings.warn(
            "Rest-pose forward direction is not uniform across the "
            f"dataset — {_format(uniformity['rest_forward'])}. Pass "
            "target_rest_forward='<axis>' to harmonize.",
            UserWarning, stacklevel=3)

    if target_rest_up is None:
        # Two independent rest-up categories.  Both are suppressed when the
        # user passes ``target_rest_up``, which reorients every clip's rest
        # pose and resolves both conditions.
        rest_up = uniformity["rest_up"]
        if len(rest_up) > 1:
            warnings.warn(
                "Rest-pose up axis is not uniform across the dataset — "
                f"{_format(rest_up)}. Pass target_rest_up='<axis>' to "
                "harmonize.",
                UserWarning, stacklevel=3)

        mismatch = uniformity["rest_anim_mismatch"]
        if mismatch:
            example = mismatch[:3]
            warnings.warn(
                f"Rest-pose up disagrees with animation-derived world_up "
                f"in {len(mismatch)} file(s) (e.g. {example}). Quaternion / "
                "6D / axis-angle tensors extracted from these files will "
                "not match topology-identical files whose rest pose "
                "agrees. Pass target_rest_up='<axis>' to reorient every "
                "clip's rest pose before extraction.",
                UserWarning, stacklevel=3)


def preprocess_directory(
    bvh_dir: str | Path,
    output_path: str | Path,
    representation: str = "6d",
    center_root: bool = True,
    include_quaternions: bool = False,
    include_velocities: bool = False,
    include_foot_contacts: bool = False,
    label_fn: Callable[[str], int] | None = None,
    filter_fn: Callable[[str], bool] | None = None,
    file_pattern: str = "*.bvh",
    skip_errors: bool = False,
    world_up: str = "auto",
    lr_mapping: dict[str, str] | None = None,
    require_matching_topology: bool = True,
    target_world_up: str | None = None,
    target_rest_forward: str | None = None,
    target_rest_up: str | None = None,
    parallel: bool = False,
    max_workers: int | None = None,
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
        (useful for runtime speed perturbation / dropout).  When
        ``representation="quaternion"`` this flag is redundant and the
        main joint data is used without duplication.
    include_velocities : bool
        If True, compute per-joint linear velocities via
        :meth:`pybvh.Bvh.joint_velocities` (central stencil, edge
        padding — shape ``(F, N, 3)`` aligned with the source frames)
        and store them per clip.  Static features: **not** refreshed
        after augmentation, so use for evaluation / targets, not as
        augmentation-invariant training inputs.
    include_foot_contacts : bool
        If True, compute binary foot-contact labels via
        :meth:`pybvh.Bvh.foot_contacts` (default ``method="combined"``)
        and store them per clip along with the detected foot joint
        names in ``skeleton_info["foot_joints"]``.  Static features,
        same caveat as ``include_velocities``.
    label_fn : callable, optional
        ``label_fn(filename_stem) -> int``.  If provided, stores
        per-clip integer labels.
    filter_fn : callable, optional
        ``filter_fn(filename_stem) -> bool``.  If provided, only
        files for which it returns True are loaded and processed.
        Applied before loading — skipped files are never parsed.
    file_pattern : str
        Glob pattern for BVH files (default ``"*.bvh"``).
    skip_errors : bool
        If True, files that fail to load emit a ``UserWarning`` and
        are skipped rather than propagating the exception.
    world_up : str
        Forwarded to :func:`pybvh.read_bvh_file`.  ``"auto"`` (default)
        auto-detects per file; pass ``"+y"`` etc. to override.
    lr_mapping : dict or None
        Forwarded to :func:`pybvh.read_bvh_file`.  Explicit left/right
        joint pair mapping, useful for uniform dataset conventions.
    require_matching_topology : bool
        If True (default), every clip must share the first clip's
        topology (``joint_names`` + ``euler_orders``).  Mismatched
        clips raise :class:`ValueError`.  Set to False to keep the
        lenient pre-0.3 behavior.
    target_world_up : str or None
        If set (e.g. ``"+y"``), reorient every clip via
        :meth:`pybvh.Bvh.reorient_world_up` so the world vertical axis
        matches.  ``None`` (default) leaves each clip's ``world_up``
        untouched.  Complements the ``world_up`` parsing kwarg:
        ``world_up="auto"`` parses what the file declares;
        ``target_world_up="+y"`` harmonizes across the dataset.
    target_rest_forward : str or None
        If set, reorient every clip via
        :meth:`pybvh.Bvh.reorient_rest_forward` so the rest-pose
        forward direction matches.  ``None`` (default) skips this
        uniformization.  Must not be parallel to the (post-
        ``target_world_up``) up axis.
    target_rest_up : str or None
        If set, reorient every clip via
        :meth:`pybvh.Bvh.reorient_rest_up`.  Typically only needed
        for the rare single-file case where a file's rest-pose up
        disagrees with its animation up.  ``None`` (default) skips.
    parallel : bool
        If True, load BVH files using a :class:`ThreadPoolExecutor`.
        Speeds up large directories; per-file I/O is the bottleneck.
    max_workers : int, optional
        Thread count when ``parallel=True``.  ``None`` defers to
        :class:`ThreadPoolExecutor`'s default.

    Notes
    -----
    **Uniformity warnings.** After loading, this function inspects
    every clip's animation-derived ``world_up``, rest-pose forward
    direction, and rest-pose up axis.  It emits one aggregated
    :class:`UserWarning` per category when files disagree, plus a
    separate aggregated warning when any file's rest-pose up axis
    disagrees with its own animation-derived ``world_up`` (pybvh's
    per-file rest/animation-disagreement warning is suppressed during
    load in favor of this one batch-level message).  Warnings include
    the distribution of values, the first three example filenames per
    minority value, and the exact kwarg that would fix it
    (``target_world_up``, ``target_rest_forward``, ``target_rest_up``).
    When the corresponding ``target_*`` kwarg is explicitly set, that
    category's check is skipped (the target value becomes the
    post-reorient ground truth).

    Returns
    -------
    dict
        Summary with keys: ``num_clips``, ``representation``,
        ``filenames``, ``skeleton_info``, ``uniformity``.
        ``uniformity`` is a dict of the form::

            {
              "world_up":     {value: [stems, ...]},
              "rest_forward": {value: [stems, ...]},
              "rest_up":      {value: [stems, ...]},
              "rest_anim_mismatch": [stems, ...],
            }

        capturing the pre-reorient state of the dataset (useful for
        CI gates that want to fail on heterogeneity).
        ``rest_anim_mismatch`` lists files whose rest-pose up axis
        disagrees with their animation-derived ``world_up`` — the
        condition ``target_rest_up`` repairs.
    """
    bvh_dir = Path(bvh_dir)
    output_path = Path(output_path)

    all_paths = sorted(bvh_dir.glob(file_pattern))
    if filter_fn is not None:
        all_paths = [p for p in all_paths if filter_fn(p.stem)]

    if len(all_paths) == 0:
        raise ValueError(f"No BVH files found in {bvh_dir} with pattern '{file_pattern}'"
                         + (" after filtering" if filter_fn is not None else ""))

    loader = partial(
        _load_one, world_up=world_up, lr_mapping=lr_mapping,
        skip_errors=skip_errors)

    if parallel:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            loaded = list(pool.map(loader, all_paths))
    else:
        loaded = [loader(p) for p in all_paths]

    clips: list[Bvh] = []
    stems: list[str] = []
    for p, b in zip(all_paths, loaded):
        if b is not None:
            clips.append(b)
            stems.append(p.stem)

    if not clips:
        raise ValueError(
            f"No BVH files successfully loaded from {bvh_dir} "
            f"with pattern '{file_pattern}'")

    # ------------------------------------------------------------------
    # Uniformity audit + warnings + optional reorientation.
    # Ordering matters: warnings reflect the *pre-reorient* state so
    # the summary is informative. Reorientation must happen before
    # data extraction so the downstream to_6d / to_quat / etc. see
    # the harmonized angles.
    # ------------------------------------------------------------------
    uniformity = _compute_uniformity(clips, stems)
    _warn_if_heterogeneous(
        uniformity, target_world_up, target_rest_forward, target_rest_up)

    if target_world_up is not None:
        clips = [b.reorient_world_up(target_world_up) or b for b in clips]
    if target_rest_forward is not None:
        clips = [
            b.reorient_rest_forward(target_rest_forward) or b for b in clips]
    if target_rest_up is not None:
        clips = [b.reorient_rest_up(target_rest_up) or b for b in clips]

    if require_matching_topology:
        reference = clips[0]
        for i, b in enumerate(clips[1:], start=1):
            if not reference.matches_topology(b):
                raise ValueError(
                    f"Clip '{stems[i]}' has topology incompatible with "
                    f"'{stems[0]}' (joint_names or euler_orders differ). "
                    f"Pass require_matching_topology=False to accept "
                    f"mismatched clips, or pre-harmonize the dataset "
                    f"with pybvh.harmonize().")

    # Extract data per clip
    all_root_pos: list[npt.NDArray[np.float64]] = []
    all_joint_data: list[npt.NDArray[np.float64]] = []
    all_joint_quats: list[npt.NDArray[np.float64]] = []
    all_velocities: list[npt.NDArray[np.float64]] = []
    all_foot_contacts: list[npt.NDArray[np.float64]] = []

    # Pin foot-joint auto-detection to the first clip so all clips
    # produce contact arrays with the same shape.
    foot_joints: list[str] | None = None
    if include_foot_contacts:
        foot_joints = clips[0].auto_detect_foot_joints()

    for bvh in clips:
        root_pos, joint_data, quats = _extract_primary_and_quats(
            bvh, representation, want_quaternions=include_quaternions)
        if center_root and root_pos.shape[0] > 0:
            root_pos = root_pos - root_pos[0:1]
        all_root_pos.append(root_pos)
        all_joint_data.append(joint_data)

        if include_quaternions:
            assert quats is not None
            all_joint_quats.append(quats)

        if include_velocities:
            all_velocities.append(bvh.joint_velocities())

        if include_foot_contacts:
            all_foot_contacts.append(
                bvh.foot_contacts(foot_joints=foot_joints))

    # Skeleton info from first clip
    skel_info = get_skeleton_info(clips[0])
    if include_foot_contacts:
        skel_info["foot_joints"] = list(foot_joints) if foot_joints else []

    # Normalization stats (computed on the primary representation only;
    # velocities / foot contacts have their own natural scales).
    stats = compute_normalization_stats(clips, representation=representation)

    # Labels
    labels = None
    if label_fn is not None:
        labels = np.array([label_fn(s) for s in stems], dtype=np.int64)

    # Save
    ext = output_path.suffix.lower()
    if ext == ".hdf5" or ext == ".h5":
        _save_hdf5(output_path, all_root_pos, all_joint_data,
                   all_joint_quats, all_velocities, all_foot_contacts,
                   labels, stats, skel_info, representation, stems)
    else:
        _save_npz(output_path, all_root_pos, all_joint_data,
                  all_joint_quats, all_velocities, all_foot_contacts,
                  labels, stats, skel_info, representation, stems)

    return {
        "num_clips": len(clips),
        "representation": representation,
        "filenames": stems,
        "skeleton_info": skel_info,
        "uniformity": uniformity,
    }


def _save_npz(
    path: Path,
    root_pos_list: list,
    joint_data_list: list,
    joint_quats_list: list,
    velocities_list: list,
    foot_contacts_list: list,
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
    if "constant_channels" in stats:
        save_dict["constant_channels"] = stats["constant_channels"]
    for i, (rp, jd) in enumerate(zip(root_pos_list, joint_data_list)):
        save_dict[f"clip_{i}_root_pos"] = rp
        save_dict[f"clip_{i}_joint_data"] = jd
    if joint_quats_list:
        for i, jq in enumerate(joint_quats_list):
            save_dict[f"clip_{i}_joint_quats"] = jq
    if velocities_list:
        for i, v in enumerate(velocities_list):
            save_dict[f"clip_{i}_velocities"] = v
    if foot_contacts_list:
        for i, fc in enumerate(foot_contacts_list):
            save_dict[f"clip_{i}_foot_contacts"] = fc
    if labels is not None:
        save_dict["labels"] = labels
    np.savez(path, **save_dict)


def _save_hdf5(
    path: Path,
    root_pos_list: list,
    joint_data_list: list,
    joint_quats_list: list,
    velocities_list: list,
    foot_contacts_list: list,
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
        if "constant_channels" in stats:
            f.create_dataset("constant_channels", data=stats["constant_channels"])
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
            if velocities_list:
                grp.create_dataset("velocities", data=velocities_list[i])
            if foot_contacts_list:
                grp.create_dataset("foot_contacts", data=foot_contacts_list[i])


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
        ``joint_data``, optionally ``joint_quats`` / ``velocities`` /
        ``foot_contacts``), ``labels``, ``mean``, ``std``,
        ``skeleton_info``, ``representation``, ``filenames``.  Also
        includes ``constant_channels`` when the file was written by
        pybvh-ml >= 0.3 (absent for older files).
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
        for extra in ("joint_quats", "velocities", "foot_contacts"):
            key = f"clip_{i}_{extra}"
            if key in data:
                clip[extra] = data[key]
        clips.append(clip)

    result: dict = {
        "clips": clips,
        "mean": data["mean"],
        "std": data["std"],
        "skeleton_info": skel_info,
        "representation": representation,
        "filenames": filenames,
    }
    if "constant_channels" in data.files:
        result["constant_channels"] = data["constant_channels"]
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
            for extra in ("joint_quats", "velocities", "foot_contacts"):
                if extra in grp:
                    clip[extra] = grp[extra][()]
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
        if "constant_channels" in f:
            result["constant_channels"] = f["constant_channels"][()]
        if "labels" in f:
            result["labels"] = f["labels"][()]
        else:
            result["labels"] = None

    return result
