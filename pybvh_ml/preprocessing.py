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

from pybvh import Bvh, read_bvh_file
from pybvh import harmonize as pybvh_harmonize
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
    representation : {"euler", "quat", "6d", "axisangle"}

    Returns
    -------
    root_pos : ndarray, shape (F, 3)
    joint_data : ndarray, shape (F, J, C_repr)
    """
    if representation == "euler":
        return bvh.root_pos.copy(), bvh.joint_angles.copy()
    if representation == "quat":
        return bvh.to_quat()
    if representation == "6d":
        return bvh.to_6d()
    if representation == "axisangle":
        return bvh.to_axisangle()
    raise ValueError(
        f"Unknown representation '{representation}'. "
        f"Choose from ['euler', 'quat', '6d', 'axisangle']")


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

    if representation == "quat":
        root_pos, joint_data = bvh.to_quat()
        return root_pos, joint_data, joint_data

    if representation == "euler":
        # Euler doesn't compute FK, so call to_quat separately.
        root_pos, joint_data = extract_repr(bvh, "euler")
        _, quats = bvh.to_quat()
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
                f"in {len(mismatch)} file(s) (e.g. {example}). Tensors "
                "extracted from these files (quaternion / 6D / axis-angle "
                "/ rotmat / euler) will live in a different reference "
                "frame than topology-identical files whose rest pose "
                "agrees with their animation-inferred up axis.\n\n"
                "Two recovery paths:\n"
                "  - If the file's rest pose is authoritative (most "
                "common; animation frame 0 may be mid-action and confuse "
                "pybvh's auto-inference): pass world_up='<axis>' to "
                "preprocess_directory to override animation-based "
                "detection at parse time.\n"
                "  - If the animation frame is authoritative: pass "
                "target_rest_up='<axis>' to reorient each clip's rest "
                "pose to match its animation up.",
                UserWarning, stacklevel=3)


_REP_NEEDS_CHANNEL_MATCH = {"euler", "axisangle"}


def _majority_value(distribution: dict[str, list[str]]) -> str | None:
    """Return the key with the most entries in ``distribution`` (ties broken
    by lexical order for determinism), or ``None`` if empty."""
    if not distribution:
        return None
    return max(distribution.keys(), key=lambda k: (len(distribution[k]), k))


def _majority_euler_order(clips: list[Bvh]) -> str | None:
    """Pick the single most common per-joint Euler order across all clips.

    ``pybvh.harmonize(target_euler_order=...)`` takes one order string and
    rewrites every joint to it — so the right default is the order that
    minimizes rewrites: the mode of every joint's order across every clip.
    Ties broken by lexical order for determinism.
    """
    if not clips:
        return None
    counts: dict[str, int] = {}
    for c in clips:
        for o in c.euler_orders:
            counts[o] = counts.get(o, 0) + 1
    return max(counts.keys(), key=lambda k: (counts[k], k))


def _is_already_uniform_euler_order(
    clips: list[Bvh], target: str,
) -> bool:
    """True iff every joint in every clip is already in ``target`` order."""
    return all(all(o == target for o in c.euler_orders) for c in clips)


def _resolve_harmonize_targets(
    clips: list[Bvh],
    uniformity: dict,
    representation: str,
    target_world_up: str | None,
    target_rest_forward: str | None,
    target_rest_up: str | None,
    target_euler_order: str | None,
) -> dict[str, str]:
    """Resolve target signature for ``pybvh.harmonize``: explicit kwargs win,
    audit majority fills in the rest.  Order-sensitive representations
    additionally resolve a target Euler order; rotation-invariant ones
    drop it (mixing orders is harmless in those tensors).
    """
    targets: dict[str, str] = {}
    if target_world_up is not None:
        targets["target_world_up"] = target_world_up
    elif len(uniformity["world_up"]) > 1:
        targets["target_world_up"] = _majority_value(uniformity["world_up"])
    if target_rest_up is not None:
        targets["target_rest_up"] = target_rest_up
    elif uniformity["rest_anim_mismatch"]:
        targets["target_rest_up"] = _majority_value(uniformity["rest_up"])
    if target_rest_forward is not None:
        targets["target_rest_forward"] = target_rest_forward
    elif len(uniformity["rest_forward"]) > 1:
        targets["target_rest_forward"] = _majority_value(
            uniformity["rest_forward"])

    if _channel_layout_depends_on_euler_order(representation):
        if target_euler_order is not None:
            targets["target_euler_order"] = target_euler_order
        else:
            order = _majority_euler_order(clips)
            if order is not None and not _is_already_uniform_euler_order(
                    clips, order):
                targets["target_euler_order"] = order
    return targets


def _normalization_stats_from_arrays(
    root_pos_list: list[npt.NDArray[np.float64]],
    joint_data_list: list[npt.NDArray[np.float64]],
) -> dict[str, npt.NDArray]:
    """Compute global mean/std/constant_channels across already-extracted
    ``(root_pos, joint_data)`` lists.

    Equivalent to ``pybvh.compute_normalization_stats(clips, ...)`` on
    the saved ``(F, 3 + J*C)`` flat layout — but operates on the
    numpy arrays directly so the cross-clip rest-offset check inside
    ``batch_to_numpy`` is bypassed.  Layout: ``[root_pos (3), joint_data
    flattened over (J, C)]`` per frame, concatenated across all clips.
    """
    flats: list[npt.NDArray[np.float64]] = []
    for rp, jd in zip(root_pos_list, joint_data_list):
        F = rp.shape[0]
        flats.append(np.concatenate([rp, jd.reshape(F, -1)], axis=1))
    all_frames = np.concatenate(flats, axis=0)
    mean = all_frames.mean(axis=0)
    std = all_frames.std(axis=0)
    constant_channels = std < 1e-8
    std = std.copy()
    std[constant_channels] = 1.0
    return {"mean": mean, "std": std, "constant_channels": constant_channels}


def _stage_counts(applied_stages: list[dict]) -> dict[str, int]:
    """Aggregate ``HarmonizeReport.applied_stages`` (per-clip list of
    per-stage dicts) into ``{stage_name: clip_count}``."""
    counts: dict[str, int] = {}
    for clip_stages in applied_stages:
        for stage in clip_stages:
            counts[stage] = counts.get(stage, 0) + 1
    return counts


def _run_harmonize(
    clips: list[Bvh],
    stems: list[str],
    uniformity: dict,
    representation: str,
    target_world_up: str | None,
    target_rest_forward: str | None,
    target_rest_up: str | None,
    target_euler_order: str | None,
) -> tuple[list[Bvh], list[str]]:
    """Drive ``pybvh.harmonize`` with resolved targets and surface drops.

    Hierarchy mismatches against the reference clip are dropped by
    ``pybvh.harmonize`` (default ``on_incompatible="drop"``).  Silent
    drops are exactly the failure mode the maintainer report hit — we
    inspect the returned :class:`HarmonizeReport` and raise with the
    dropped filenames + reasons so the user can act.

    Records the resolved targets, per-stage modification counts, and
    the JSON-native report under ``uniformity["harmonized_to"]`` so the
    transformation trail is auditable from the saved dataset metadata.
    """
    import dataclasses

    targets = _resolve_harmonize_targets(
        clips, uniformity, representation,
        target_world_up, target_rest_forward, target_rest_up,
        target_euler_order,
    )
    # Pin the reference clip so pybvh.harmonize gates on the hierarchy
    # graph (names + parent indices) and retargets bone offsets to the
    # first clip.  Without ``reference=`` harmonize is purely reorient,
    # and hierarchy mismatches would only surface later in our own
    # _check_skeleton_compatibility — with worse drop diagnostics.
    harmonized, report = pybvh_harmonize(
        clips, reference=clips[0],
        **targets, return_report=True, verbose=False,
    )
    if report.dropped_indices:
        labels = [
            src if src else f"index={i}"
            for i, src in zip(report.dropped_indices, report.dropped_sources)
        ]
        details = "; ".join(
            f"'{lbl}' ({reason})"
            for lbl, reason in zip(labels, report.drop_reasons)
        )
        raise ValueError(
            f"pybvh.harmonize dropped {len(report.dropped_indices)} clip(s) "
            f"as incompatible with the reference: {details}. "
            f"Hierarchy mismatches cannot be auto-fixed — filter the "
            f"dataset to a single skeleton, or run pybvh.harmonize with "
            f"an explicit reference for retargeting.")

    kept_stems = [stems[i] for i in report.kept_indices]
    uniformity["harmonized_to"] = {
        "targets": targets,
        "stage_counts": _stage_counts(report.applied_stages),
        "report": dataclasses.asdict(report),
    }
    return harmonized, kept_stems


def _channel_layout_depends_on_euler_order(representation: str) -> bool:
    """Whether the saved tensor's channel layout depends on the source Euler order.

    True for ``"euler"`` / ``"axisangle"`` — mixing orders across clips
    yields a tensor whose channels are misaligned per-joint.  False for
    rotation-invariant representations (``"6d"`` / ``"quat"`` /
    ``"rotmat"``), where pybvh's conversion produces an order-agnostic
    layout.
    """
    return representation in _REP_NEEDS_CHANNEL_MATCH


def _clip_label(bvh: Bvh, stem: str) -> str:
    """Prefer the file's ``source_path`` stem (set by pybvh) over the
    caller-supplied stem when both are available — keeps error messages
    consistent with how the user thinks about the files."""
    src = getattr(bvh, "source_path", None)
    if src:
        return Path(src).stem
    return stem


def _check_skeleton_compatibility(
    clips: list[Bvh], stems: list[str], representation: str,
) -> None:
    """Validate that every clip is compatible with the first clip's skeleton.

    Compares the skeleton *graph* (joint names + parent indices) but not
    rest offsets — bone-length variation across actors is intrinsic to
    multi-actor datasets and doesn't affect the angle-based tensors
    pybvh-ml extracts (``joint_data`` is a function of rotations, not
    bone lengths; root translation is centered by default).  Pybvh's
    own ``harmonize(reference=...)`` uses the same loose convention.
    Callers who need bone-length uniformity (e.g. when extracting
    FK-derived features) should pre-run ``pybvh.harmonize(reference=...)``
    via ``harmonize=True``.

    For order-sensitive representations (``"euler"`` / ``"axisangle"``),
    additionally requires channel equality (``matches_channels``).
    Raises :class:`ValueError` on the first divergence, naming both
    clips and pointing at the right recovery.
    """
    reference = clips[0]
    ref_label = _clip_label(reference, stems[0])
    needs_channels = _channel_layout_depends_on_euler_order(representation)
    for i, bvh in enumerate(clips[1:], start=1):
        clip_label = _clip_label(bvh, stems[i])
        if not reference.matches_hierarchy(bvh, match_offsets=False):
            raise ValueError(
                f"Clip '{clip_label}' skeleton graph is incompatible with "
                f"'{ref_label}' (joint names or parent indices differ). "
                f"This is a data problem — clips with different skeletons "
                f"cannot be batched together. Filter the dataset to a "
                f"single skeleton, or use pybvh.harmonize(reference=<ref>) "
                f"if the difference is bone-offset retargetable.")
        if needs_channels and not reference.matches_channels(bvh):
            raise ValueError(
                f"Clip '{clip_label}' has Euler orders incompatible with "
                f"'{ref_label}'. For representation='{representation}' "
                f"the tensor channel layout depends on per-joint Euler "
                f"order, so mixed orders corrupt the batch. Pass "
                f"harmonize=True to unify Euler orders automatically, "
                f"or pick a rotation-invariant representation "
                f"('6d' / 'quat' / 'rotmat') for which "
                f"channel layout is order-agnostic.")


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
    harmonize: bool = False,
    target_world_up: str | None = None,
    target_rest_forward: str | None = None,
    target_rest_up: str | None = None,
    target_euler_order: str | None = None,
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
        ``representation="quat"`` this flag is redundant and the
        main joint data is used without duplication.
    include_velocities : bool
        If True, compute per-joint linear velocities via
        :meth:`pybvh.Bvh.joint_velocities` (central stencil, edge
        padding — shape ``(F, J, 3)`` aligned with ``joint_data`` /
        ``joint_angles``, no end sites) and store them per clip.
        Static features: **not** refreshed after augmentation, so use
        for evaluation / targets, not as augmentation-invariant
        training inputs.
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
    harmonize : bool
        If True, run :func:`pybvh.harmonize` after loading to unify
        clips along every axis the dataset disagrees on.  Targets are
        resolved as: explicit ``target_*`` kwarg wins; otherwise the
        majority value from the uniformity audit fills in.  For
        ``representation in {"euler", "axisangle"}``, an Euler-order
        target is also resolved (majority of ``euler_orders[0]`` across
        clips); rotation-invariant representations skip this stage
        since channel layout is order-agnostic.

        Clips with hierarchy mismatches against the reference are
        dropped by ``pybvh.harmonize``; ``preprocess_directory``
        inspects the returned report and raises :class:`ValueError`
        rather than silently shipping a smaller dataset.  The
        resolved targets and per-stage modification counts land in
        the returned ``uniformity`` dict under
        ``uniformity["harmonized_to"]``.

        Default ``False`` keeps the explicit ``target_*`` kwargs as
        independent uniformization stages (current behavior).
    target_world_up : str or None
        Signed-axis string (``"+y"``, ``"-z"``, ...).  When
        ``harmonize=False`` (default): reorient every clip via
        :meth:`pybvh.Bvh.reorient_world_up`.  When ``harmonize=True``:
        used as the explicit world-up target for
        :func:`pybvh.harmonize`, overriding the audit-majority value.
        ``None`` (default) defers to the dataset majority under
        ``harmonize=True``, or leaves clips untouched otherwise.
    target_rest_forward : str or None
        Same dual semantics as ``target_world_up`` for the rest-pose
        forward direction.  Must not be parallel to the (post-
        ``target_world_up``) up axis.
    target_rest_up : str or None
        Same dual semantics as ``target_world_up`` for the rest-pose
        up axis.  Typically only needed for the rare single-file case
        where a file's rest-pose up disagrees with its animation up.
    target_euler_order : str or None
        Canonical Euler order (``"XYZ"``, ``"ZYX"``, ...) to unify
        joint angles to.  Only honored when ``harmonize=True`` and
        the representation is order-sensitive
        (``"euler"`` / ``"axisangle"``); silently ignored otherwise.
        ``None`` (default) under ``harmonize=True`` picks the majority
        Euler order across clips.
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
              "harmonized_to": {...},   # present only when harmonize=True
            }

        capturing the pre-reorient state of the dataset (useful for
        CI gates that want to fail on heterogeneity).
        ``rest_anim_mismatch`` lists files whose rest-pose up axis
        disagrees with their animation-derived ``world_up`` — the
        condition ``target_rest_up`` repairs.

        When ``harmonize=True``, ``harmonized_to`` carries the
        resolved target signature (``world_up``, ``rest_up``,
        ``rest_forward``, ``euler_order`` — only those that ran),
        ``stage_counts`` (per-stage count of clips modified, from
        pybvh's ``HarmonizeReport.applied_stages``), and the
        serialized ``report`` itself (JSON-native ``dict`` from
        ``dataclasses.asdict``).
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

    if harmonize:
        clips, stems = _run_harmonize(
            clips, stems, uniformity, representation,
            target_world_up, target_rest_forward, target_rest_up,
            target_euler_order,
        )
    else:
        if target_world_up is not None:
            clips = [b.reorient_world_up(target_world_up) or b for b in clips]
        if target_rest_forward is not None:
            clips = [
                b.reorient_rest_forward(target_rest_forward) or b
                for b in clips]
        if target_rest_up is not None:
            clips = [b.reorient_rest_up(target_rest_up) or b for b in clips]

    _check_skeleton_compatibility(clips, stems, representation)

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
    #
    # Computed locally from the already-extracted arrays rather than via
    # pybvh's ``compute_normalization_stats``.  That entry point routes
    # through ``batch_to_numpy``, which insists on rest-offset equality
    # via ``matches_hierarchy(match_offsets=True)``.  Pybvh-ml accepts
    # bone-length variation across actors (the angle tensors don't depend
    # on bone lengths) — see ``_check_skeleton_compatibility``.  Going
    # via the raw arrays sidesteps the redundant compatibility check.
    stats = _normalization_stats_from_arrays(all_root_pos, all_joint_data)

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
