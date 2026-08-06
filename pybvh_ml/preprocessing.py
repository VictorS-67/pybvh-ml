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

from pybvh import Bvh, parse_axis, read_bvh_file
from pybvh import harmonize as pybvh_harmonize
from pybvh import rotations
from pybvh_ml.arrays import POSITION_CENTERINGS
from pybvh_ml.skeleton import get_skeleton_info


_SUPPORTED_REPRESENTATIONS = ("euler", "quat", "6d", "axisangle")
_KNOWN_SUFFIXES = (".npz", ".hdf5", ".h5")

# Which MotionArrays field a stored position stream fills, by space.
_POSITION_STREAM_KEYS = {"joint": "joint_pos", "node": "node_pos"}


def _validate_representation(representation: str) -> None:
    """Reject representations outside pybvh-ml's extraction surface."""
    if representation not in _SUPPORTED_REPRESENTATIONS:
        raise ValueError(
            f"Unknown representation '{representation}'. "
            f"Choose from {list(_SUPPORTED_REPRESENTATIONS)}")


def _validate_position_settings(
    position_space: str,
    position_centering: str,
    center_root: bool,
) -> None:
    """Reject position settings that would write incoherent arrays.

    ``center_root=True`` with ``position_centering="first"`` is the one
    combination refused outright.  ``center_root`` subtracts all three
    components of the first frame's root (pybvh-ml's convention) while
    ``"first"`` is pybvh's ground-plane centering, which subtracts only
    the two non-up ones — so the positions are in a frame offset from
    ``root_pos``, and stay offset by exactly that amount however the
    root is centered.  That is tolerable in a transient container (the
    packers apply the shift to both and preserve the relationship), but
    not in a *written* dataset, where the recorded ``center_root=True``
    would suggest a coherence between the two streams that ground-plane
    centering never established, and nothing downstream could tell.
    """
    if position_space not in _POSITION_STREAM_KEYS:
        raise ValueError(
            f"position_space must be one of "
            f"{list(_POSITION_STREAM_KEYS)}, got {position_space!r}")
    if position_centering not in POSITION_CENTERINGS:
        raise ValueError(
            f"position_centering must be one of "
            f"{list(POSITION_CENTERINGS)}, got {position_centering!r}")
    if center_root and position_centering == "first":
        raise ValueError(
            "center_root=True cannot be combined with "
            "position_centering='first': center_root subtracts all three "
            "components of the first frame's root position (pybvh-ml's "
            "convention) while 'first' is pybvh's ground-plane centering, "
            "which leaves the up axis untouched — the two streams would be "
            "centered differently in that axis. Use "
            "position_centering='world' (the root shift is then applied to "
            "the positions too), or 'skeleton', or center_root=False.")


def _validate_output_suffix(path: Path) -> None:
    """Reject output/input paths with unrecognized dataset extensions.

    ``np.savez`` silently appends ``.npz`` to unknown suffixes, so a
    typo'd extension used to write to a file the caller never named —
    and ``load_preprocessed`` on the original path then failed with
    ``FileNotFoundError``.
    """
    if path.suffix.lower() not in _KNOWN_SUFFIXES:
        raise ValueError(
            f"Unrecognized dataset extension '{path.suffix}' in "
            f"'{path}'. Choose from {list(_KNOWN_SUFFIXES)}")


def extract_repr(
    bvh: Bvh,
    representation: str,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Extract ``(root_pos, joint_rot)`` for the given representation.

    Thin dispatcher over pybvh's ``to_*`` methods; exposed publicly so
    the PyTorch datasets can reuse the same mapping without reaching
    into a private symbol.

    Parameters
    ----------
    bvh : Bvh
    representation : {"euler", "quat", "6d", "axisangle"}
        ``"euler"`` returns ``bvh.joint_angles`` — radians, matching
        pybvh.  ``"rotmat"`` is not part of the extraction surface
        (use :func:`~pybvh_ml.convert_rotations` to derive it
        from any extracted representation).

    Returns
    -------
    root_pos : ndarray, shape (F, 3)
    joint_rot : ndarray, shape (F, J, C_repr)
    """
    _validate_representation(representation)
    if representation == "euler":
        return bvh.root_pos.copy(), bvh.joint_angles.copy()
    if representation == "quat":
        return bvh.to_quat()
    if representation == "6d":
        return bvh.to_6d()
    return bvh.to_axisangle()


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


_REST_UP_UNKNOWN = "unknown"
"""Audit key for a rig whose rest-pose up axis cannot be measured.

``Bvh.rest_up`` is ``None`` for a degenerate rest pose.  The audit is
persisted as JSON, so the key has to be a string — see
:func:`_compute_uniformity`.
"""


def _fps_key(bvh: Bvh) -> str:
    """A clip's frame rate as a stable dict key.

    Frame rates arrive as ``1 / frame_time`` floats, so keying the audit
    on the raw value would split ``120.0`` from ``119.99999999`` into two
    "distinct" rates.  Six significant digits collapses float noise while
    still separating genuinely different rates (30 vs 29.97).
    """
    return f"{1.0 / bvh.frame_time:.6g}"


def _compute_uniformity(
    clips: list[Bvh], stems: list[str],
) -> dict[str, dict[str, list[str]]]:
    """Group filenames by frame rate, world_up, rest-forward, and rest-up.

    Returned structure::

        {
          "fps":          {"120": [stem, ...], "30": [stem, ...]},
          "world_up":     {"+z": [stem, ...], "+y": [stem, ...]},
          "rest_forward": {"+y": [stem, ...], "+x": [stem, ...]},
          "rest_up":      {"+z": [stem, ...], "unknown": [stem, ...]},
          "rest_anim_mismatch": [stem, ...],  # rest_up != world_up
        }

    ``rest_anim_mismatch`` captures files whose rest-pose up axis
    disagrees with the animation-derived ``world_up`` — the condition
    pybvh warns about per-file at load.  Such files silently corrupt
    training tensors across every rotation representation; pass
    ``target_rest_up`` to reorient them at load.  A rig whose rest pose
    is degenerate is *not* listed: ``Bvh.rest_up`` is ``None`` there,
    and an unmeasurable axis cannot disagree with anything — reporting
    it as a mismatch claimed a corruption that was never diagnosed.

    ``rest_up`` files those rigs under the string ``"unknown"`` rather
    than a ``None`` key.  The audit is persisted as JSON, whose object
    keys must be strings: a ``None`` key round-trips back as the string
    ``"null"``, so the saved audit would not equal the returned one.
    Note the deliberate asymmetry with ``skeleton_info["rest_up"]``,
    which stays ``None`` — there the axis is a JSON *value*, and
    ``null`` round-trips to ``None`` correctly.
    """
    fps: dict[str, list[str]] = {}
    world_up: dict[str, list[str]] = {}
    rest_forward: dict[str, list[str]] = {}
    rest_up: dict[str, list[str]] = {}
    rest_anim_mismatch: list[str] = []
    for stem, b in zip(stems, clips):
        anim_up = b.world_up
        r_up = b.rest_up
        fps.setdefault(_fps_key(b), []).append(stem)
        world_up.setdefault(anim_up, []).append(stem)
        rest_forward.setdefault(b.rest_forward, []).append(stem)
        rest_up.setdefault(
            r_up if r_up is not None else _REST_UP_UNKNOWN, []).append(stem)
        if r_up is not None and r_up != anim_up:
            rest_anim_mismatch.append(stem)
    return {
        "fps": fps,
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
    target_fps: float | None = None,
    *,
    harmonize: bool = False,
) -> None:
    """Emit one aggregated warning per heterogeneous axis.

    Skips a category when its corresponding ``target_*`` kwarg is set
    (the user has already signaled intent to uniformize).  With
    ``harmonize=True`` the warnings still fire — the audit summary is
    informative — but the advice changes: harmonize is about to unify
    each axis to its majority value automatically, so the ``target_*``
    kwargs are overrides, not the required fix.
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

    def _advice(kwarg: str, placeholder: str = "'<axis>'") -> str:
        if harmonize:
            return (f"harmonize=True will unify these to the majority "
                    f"value; pass {kwarg}={placeholder} to override.")
        return f"Pass {kwarg}={placeholder} to harmonize."

    if len(uniformity["fps"]) > 1 and target_fps is None:
        warnings.warn(
            "Frame rate is not uniform across the dataset — "
            f"{_format(uniformity['fps'])}. Every frame-indexed feature "
            "(joint_rot, and any velocities / foot contacts) is sampled "
            "at each clip's own rate, so a model sees the same motion at "
            "different speeds. "
            f"{_advice('target_fps', '<hz>')}",
            UserWarning, stacklevel=3)

    if len(uniformity["world_up"]) > 1 and target_world_up is None:
        warnings.warn(
            "World-up axis is not uniform across the dataset — "
            f"{_format(uniformity['world_up'])}. "
            f"{_advice('target_world_up')}",
            UserWarning, stacklevel=3)

    if len(uniformity["rest_forward"]) > 1 and target_rest_forward is None:
        warnings.warn(
            "Rest-pose forward direction is not uniform across the "
            f"dataset — {_format(uniformity['rest_forward'])}. "
            f"{_advice('target_rest_forward')}",
            UserWarning, stacklevel=3)

    if target_rest_up is None:
        # Two independent rest-up categories.  Both are suppressed when the
        # user passes ``target_rest_up``, which reorients every clip's rest
        # pose and resolves both conditions.
        rest_up = uniformity["rest_up"]
        if len(rest_up) > 1:
            warnings.warn(
                "Rest-pose up axis is not uniform across the dataset — "
                f"{_format(rest_up)}. {_advice('target_rest_up')}",
                UserWarning, stacklevel=3)

        mismatch = uniformity["rest_anim_mismatch"]
        if mismatch:
            example = mismatch[:3]
            if harmonize:
                recovery = (
                    "Two recovery paths:\n"
                    "  - If the file's rest pose is authoritative (most "
                    "common; animation frame 0 may be mid-action and "
                    "confuse pybvh's auto-inference): pass "
                    "world_up='<axis>' to preprocess_directory to "
                    "override animation-based detection at parse time.\n"
                    "  - If the animation frame is authoritative: "
                    "harmonize=True will reorient these clips' rest "
                    "poses toward the majority rest-up; pass "
                    "target_rest_up='<axis>' to pick the axis "
                    "explicitly.")
            else:
                recovery = (
                    "Two recovery paths:\n"
                    "  - If the file's rest pose is authoritative (most "
                    "common; animation frame 0 may be mid-action and "
                    "confuse pybvh's auto-inference): pass "
                    "world_up='<axis>' to preprocess_directory to "
                    "override animation-based detection at parse time.\n"
                    "  - If the animation frame is authoritative: pass "
                    "target_rest_up='<axis>' to reorient each clip's "
                    "rest pose to match its animation up.")
            warnings.warn(
                f"Rest-pose up disagrees with animation-derived world_up "
                f"in {len(mismatch)} file(s) (e.g. {example}). Tensors "
                "extracted from these files (quaternion / 6D / axis-angle "
                "/ rotmat / euler) will live in a different reference "
                "frame than topology-identical files whose rest pose "
                "agrees with their animation-inferred up axis.\n\n"
                + recovery,
                UserWarning, stacklevel=3)


_REP_NEEDS_CHANNEL_MATCH = {"euler", "axisangle"}


def _majority_value(distribution: dict[str, list[str]]) -> str | None:
    """Return the key with the most entries in ``distribution`` (ties broken
    by lexical order for determinism), or ``None`` if empty.

    The :data:`_REST_UP_UNKNOWN` key is excluded — it records rigs whose
    rest-pose up axis could not be measured, which is not a value any
    clip can be reoriented *to*.  A distribution of nothing but unknowns
    therefore resolves to ``None`` (no target), not to ``"unknown"``.
    """
    keys = [k for k in distribution if k != _REST_UP_UNKNOWN]
    if not keys:
        return None
    return max(keys, key=lambda k: (len(distribution[k]), k))


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


def _is_parallel(axis_a: str, axis_b: str) -> bool:
    """Whether two signed-axis strings name the same axis, sign ignored.

    ``'+z'`` and ``'-z'`` are parallel; a rest-forward direction that is
    parallel to world up has no ground-plane rotation that reaches it,
    which is what ``pybvh.reorient_rest_forward`` rejects.
    """
    return parse_axis(axis_a).index == parse_axis(axis_b).index


def _effective_world_up(uniformity: dict, target_world_up: str | None) -> str:
    """The up axis every clip will carry once the world-up stage has run.

    An explicit target wins; otherwise it is the audit majority, which
    for a dataset that already agrees is simply that agreed value — so
    this answers correctly whether or not any reorientation happens.
    """
    if target_world_up is not None:
        return target_world_up
    # Bvh.world_up is never None (pybvh falls back rather than returning
    # nothing), so a non-empty dataset always resolves here.
    return _majority_value(uniformity["world_up"])


def _resolve_harmonize_targets(
    clips: list[Bvh],
    uniformity: dict,
    representation: str,
    target_world_up: str | None,
    target_rest_forward: str | None,
    target_rest_up: str | None,
    target_euler_order: str | None,
    target_fps: float | None = None,
) -> dict[str, str | float]:
    """Resolve target signature for ``pybvh.harmonize``: explicit kwargs win,
    audit majority fills in the rest.  Order-sensitive representations
    additionally resolve a target Euler order; rotation-invariant ones
    drop it (mixing orders is harmless in those tensors).
    """
    targets: dict[str, str | float] = {}

    def _fill_from_majority(key: str, distribution: dict) -> None:
        # _majority_value returns None when every clip's value is None
        # (degenerate rigs) — skipping the target beats passing None.
        majority = _majority_value(distribution)
        if majority is not None:
            targets[key] = majority

    if target_fps is not None:
        targets["target_fps"] = float(target_fps)
    elif len(uniformity["fps"]) > 1:
        # Majority is a *rate*, not an axis string — parse the audit key
        # back to the float pybvh.harmonize expects.
        majority_fps = _majority_value(uniformity["fps"])
        if majority_fps is not None:
            targets["target_fps"] = float(majority_fps)

    if target_world_up is not None:
        targets["target_world_up"] = target_world_up
    elif len(uniformity["world_up"]) > 1:
        _fill_from_majority("target_world_up", uniformity["world_up"])
    if target_rest_up is not None:
        targets["target_rest_up"] = target_rest_up
    elif uniformity["rest_anim_mismatch"]:
        _fill_from_majority("target_rest_up", uniformity["rest_up"])
    effective_up = _effective_world_up(uniformity, target_world_up)
    if target_rest_forward is not None:
        if _is_parallel(target_rest_forward, effective_up):
            source = ("target_world_up" if target_world_up is not None
                      else "the dataset majority")
            raise ValueError(
                f"target_rest_forward={target_rest_forward!r} is parallel "
                f"to the world up every clip will have after harmonizing "
                f"({effective_up!r}, from {source}). Rest-forward is "
                f"reached by a rotation in the ground plane, so it must be "
                f"perpendicular to world up — pick a perpendicular axis, "
                f"or change target_world_up.")
        targets["target_rest_forward"] = target_rest_forward
    elif len(uniformity["rest_forward"]) > 1:
        # Resolve each axis's majority independently, then drop
        # rest-forward candidates that are parallel to the up axis the
        # dataset is heading for: the two majorities are computed from
        # different clips and need not co-occur in any single one, and
        # pybvh rejects the parallel pair.  (The alternative — a
        # majority over whole per-clip axis *signatures* — can pick a
        # minority convention on the heaviest axis, and composes badly
        # with an explicit target_world_up.)
        perpendicular = {
            axis: stems for axis, stems in uniformity["rest_forward"].items()
            if not _is_parallel(axis, effective_up)
        }
        if perpendicular:
            _fill_from_majority("target_rest_forward", perpendicular)
        else:
            warnings.warn(
                f"Every rest-forward axis in the dataset "
                f"({sorted(uniformity['rest_forward'])}) is parallel to the "
                f"resolved world up ({effective_up!r}), so rest-forward is "
                f"left unharmonized — clips keep their own facing. Pass "
                f"target_rest_forward='<axis perpendicular to "
                f"{effective_up}>' to unify it anyway.",
                UserWarning, stacklevel=4)

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
    include_root_pos: bool = True,
) -> dict[str, npt.NDArray]:
    """Compute global mean/std/constant_channels across already-extracted
    ``(root_pos, joint_data)`` lists.

    Array-level core shared by the public
    :func:`compute_normalization_stats` (which extracts the arrays from
    Bvh objects first) and :func:`preprocess_directory` (which already
    holds them).  Layout: ``[root_pos (3), joint_data flattened over
    (J, C)]`` per frame, concatenated across all clips; the root_pos
    columns are dropped when ``include_root_pos=False``.
    """
    flats: list[npt.NDArray[np.float64]] = []
    for rp, jd in zip(root_pos_list, joint_data_list):
        F = jd.shape[0]
        flat = jd.reshape(F, -1)
        if include_root_pos:
            flat = np.concatenate([rp, flat], axis=1)
        flats.append(flat)
    all_frames = np.concatenate(flats, axis=0)
    mean = all_frames.mean(axis=0)
    std = all_frames.std(axis=0)
    constant_channels = std < 1e-8
    std = std.copy()
    std[constant_channels] = 1.0
    return {"mean": mean, "std": std, "constant_channels": constant_channels}


def _reject_parallel_rest_forward(
    clips: list[Bvh],
    stems: list[str],
    target_rest_forward: str,
    target_world_up: str | None,
) -> None:
    """Fail, naming the kwargs, on a rest-forward target that cannot be reached.

    ``reorient_rest_forward`` rotates in the ground plane, so a target
    parallel to a clip's world up is unreachable and pybvh rejects it —
    per clip, naming no file and no kwarg.  The offending pair is known
    before any clip is touched, so say so here instead.
    """
    if target_world_up is not None:
        # Every clip carries target_world_up by the time rest-forward runs.
        if _is_parallel(target_rest_forward, target_world_up):
            raise ValueError(
                f"target_rest_forward={target_rest_forward!r} is parallel "
                f"to target_world_up={target_world_up!r}. Rest-forward is "
                f"reached by a rotation in the ground plane, so it must be "
                f"perpendicular to world up.")
        return
    offenders = [
        (s, b.world_up) for s, b in zip(stems, clips)
        if _is_parallel(target_rest_forward, b.world_up)
    ]
    if offenders:
        listed = ", ".join(f"'{s}' (world_up={up!r})"
                           for s, up in offenders[:5])
        raise ValueError(
            f"target_rest_forward={target_rest_forward!r} is parallel to "
            f"the world up of {len(offenders)} clip(s): {listed}. "
            f"Rest-forward is reached by a rotation in the ground plane, "
            f"so it must be perpendicular to world up — pick a "
            f"perpendicular axis, or pass target_world_up=... to move "
            f"those clips first.")


def _reject_degenerate_rest_up_targets(
    clips: list[Bvh], stems: list[str], target_rest_up: str,
) -> None:
    """Fail, with names, before a rest-up target hits a degenerate rig.

    A rest-up target sends every clip through ``reorient_rest_up``,
    which needs a measurable rest-pose up axis to rotate *from*.  A rig
    that has none (``Bvh.rest_up is None``) raises inside pybvh naming
    no file, and under ``harmonize=True`` the target can be filled from
    the dataset majority rather than passed by the caller — so the
    error would arrive for a kwarg they never wrote.

    Raising here rather than skipping those clips is deliberate: the
    target exists to repair a genuine rest/animation disagreement among
    the other clips, and silently leaving some rigs unreoriented would
    ship exactly the mixed reference frames it was meant to fix.
    """
    # TODO(upstream pybvh): batch.harmonize's rest-up stage guard treats
    # rest_up=None as "differs from the target" and calls
    # reorient_rest_up on rigs that have nothing to reorient; it should
    # skip them.  Filed in docs/internal_logs/upstream_feedback.md —
    # once pybvh ships that, this can relax to a warning.
    degenerate = [s for s, b in zip(stems, clips) if b.rest_up is None]
    if not degenerate:
        return
    raise ValueError(
        f"target_rest_up={target_rest_up!r} reorients every clip's rest "
        f"pose, but {len(degenerate)} clip(s) have no measurable rest-pose "
        f"up axis (degenerate rest pose): {degenerate[:5]}. Exclude them "
        f"with filter_fn=..., or drop the rest-up target (with "
        f"harmonize=True it is filled from the dataset majority, so pass "
        f"target_rest_up only when you need it).")


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
    retarget: bool,
    target_fps: float | None = None,
) -> tuple[list[Bvh], list[str]]:
    """Drive ``pybvh.harmonize`` with resolved targets and surface drops.

    By default (``retarget=False``) harmonize runs without a reference
    clip: pure reorientation/resampling, preserving each actor's bone
    lengths.  Hierarchy mismatches then surface in
    :func:`_check_skeleton_compatibility` right after, with recovery
    hints.  With ``retarget=True`` the first clip is pinned as
    ``reference=``, which makes pybvh.harmonize gate on the hierarchy
    graph (names + parent indices) *and* retarget every clip's bone
    offsets to that clip.  Reference-gated drops (default
    ``on_incompatible="drop"``) would otherwise be silent — we inspect
    the returned :class:`HarmonizeReport` and raise with the dropped
    filenames + reasons so the user can act.

    Records the resolved targets, the retarget choice, per-stage
    modification counts, and the JSON-native report under
    ``uniformity["harmonized_to"]``, which the savers persist as
    ``uniformity_json`` — the transformation trail is auditable from
    the saved dataset metadata.
    """
    import dataclasses

    targets = _resolve_harmonize_targets(
        clips, uniformity, representation,
        target_world_up, target_rest_forward, target_rest_up,
        target_euler_order, target_fps,
    )
    if "target_rest_up" in targets:
        # Covers the majority-filled target too, not just an explicit one.
        _reject_degenerate_rest_up_targets(
            clips, stems, targets["target_rest_up"])
    harmonized, report = pybvh_harmonize(
        clips, reference=clips[0] if retarget else None,
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
            f"as incompatible with the reference '{stems[0]}' (pinned by "
            f"retarget=True): {details}. Hierarchy mismatches cannot be "
            f"auto-fixed — filter the dataset to a single skeleton, or run "
            f"pybvh.harmonize with an explicit reference for retargeting.")

    kept_stems = [stems[i] for i in report.kept_indices]
    uniformity["harmonized_to"] = {
        "targets": targets,
        "retarget": retarget,
        "reference": stems[0] if retarget else None,
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
    FK-derived features) should pass ``harmonize=True, retarget=True``.

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
                f"single skeleton, or pass harmonize=True, retarget=True "
                f"(pybvh.harmonize with a pinned reference) if the "
                f"difference is bone-offset retargetable.")
        if needs_channels and not reference.matches_channels(bvh):
            raise ValueError(
                f"Clip '{clip_label}' has Euler orders incompatible with "
                f"'{ref_label}'. For representation='{representation}' "
                f"the tensor channel layout depends on per-joint Euler "
                f"order, so mixed orders corrupt the batch. Pass "
                f"harmonize=True to unify Euler orders automatically, "
                f"or pick a rotation-invariant representation "
                f"('6d' / 'quat') for which channel layout is "
                f"order-agnostic.")


# =========================================================================
# Normalization utilities
# =========================================================================

def compute_normalization_stats(
    bvh_list: list[Bvh],
    representation: str = "euler",
    include_root_pos: bool = True,
    center_root: bool = False,
) -> dict[str, npt.NDArray]:
    """Compute per-channel mean and std across a dataset of BVH objects.

    Extracts every clip in the given representation, concatenates all
    frames, then computes mean and standard deviation per feature
    channel.  Compatible with the ``Mean.npy`` / ``Std.npy`` convention
    used by HumanML3D and MDM.  The channel layout matches
    :func:`pybvh_ml.pack_to_flat` and the arrays saved by
    :func:`preprocess_directory`: ``[root_pos (3), joint_data flattened
    over (J, C)]`` per frame.

    Parameters
    ----------
    bvh_list : list of Bvh
        Dataset of BVH objects.  Clips must share the same skeleton
        graph (joint names + parent indices); bone-length variation
        across actors is accepted, matching the loose compatibility
        convention of :func:`preprocess_directory`.  For
        order-sensitive representations (``'euler'`` / ``'axisangle'``),
        per-joint Euler orders must also match.
    representation : str, optional
        Rotation representation: ``'euler'`` (default), ``'quat'``,
        ``'6d'``, or ``'axisangle'``.
    include_root_pos : bool, optional
        If True (default), include root position in the features.
    center_root : bool, optional
        If True, subtract each clip's first-frame root position before
        computing the stats — reproducing exactly the ``mean`` / ``std``
        that :func:`preprocess_directory` stores under its default
        ``center_root=True`` (whose arrays are centered before the
        stats pass).  Default ``False`` computes stats on raw root
        positions.

    Returns
    -------
    dict
        ``{"mean": ndarray (D,), "std": ndarray (D,),
        "constant_channels": ndarray of bool (D,)}``.

        ``constant_channels[i]`` is True when the raw standard deviation
        for channel ``i`` was below ``1e-8`` and the guard replaced it
        with ``1.0``. Normalized values on these channels are identically
        zero rather than ~N(0, 1) — use this mask to exclude them from
        per-channel diagnostics.

    Raises
    ------
    ValueError
        If ``bvh_list`` is empty, skeletons are incompatible, or the
        representation is unknown.

    Notes
    -----
    Save/load stats with ``np.savez("stats.npz", **stats)`` and
    ``dict(np.load("stats.npz"))``. Bool arrays round-trip cleanly
    through ``.npz``.
    """
    if not bvh_list:
        raise ValueError("bvh_list is empty.")
    _validate_representation(representation)

    labels = [f"bvh_list[{i}]" for i in range(len(bvh_list))]
    _check_skeleton_compatibility(bvh_list, labels, representation)

    root_pos_list: list[npt.NDArray[np.float64]] = []
    joint_data_list: list[npt.NDArray[np.float64]] = []
    for bvh in bvh_list:
        root_pos, joint_data = extract_repr(bvh, representation)
        if center_root and root_pos.shape[0] > 0:
            root_pos = root_pos - root_pos[0:1]
        root_pos_list.append(root_pos)
        joint_data_list.append(joint_data)

    return _normalization_stats_from_arrays(
        root_pos_list, joint_data_list, include_root_pos=include_root_pos)


def normalize_array(
    data: npt.NDArray[np.float64],
    stats: dict[str, npt.NDArray[np.float64]],
) -> npt.NDArray[np.float64]:
    """Apply z-score normalization: ``(data - mean) / std``.

    Parameters
    ----------
    data : ndarray
        Data to normalize. Last dimension must match ``stats["mean"]``.
    stats : dict
        ``{"mean": ndarray (D,), "std": ndarray (D,)}`` from
        :func:`compute_normalization_stats`.

    Returns
    -------
    ndarray
        Normalized data, same shape as input.
    """
    return (data - stats["mean"]) / stats["std"]


def denormalize_array(
    data: npt.NDArray[np.float64],
    stats: dict[str, npt.NDArray[np.float64]],
) -> npt.NDArray[np.float64]:
    """Reverse z-score normalization: ``data * std + mean``.

    Parameters
    ----------
    data : ndarray
        Normalized data to denormalize.
    stats : dict
        ``{"mean": ndarray (D,), "std": ndarray (D,)}`` from
        :func:`compute_normalization_stats`.

    Returns
    -------
    ndarray
        Denormalized data, same shape as input.
    """
    return data * stats["std"] + stats["mean"]


def preprocess_directory(
    bvh_dir: str | Path,
    output_path: str | Path,
    representation: str = "6d",
    center_root: bool = True,
    include_positions: bool = False,
    position_space: str = "joint",
    position_centering: str = "world",
    include_quaternions: bool = False,
    include_velocities: bool = False,
    include_foot_contacts: bool = False,
    foot_joints: list[str] | None = None,
    label_fn: Callable[[str], int] | None = None,
    filter_fn: Callable[[str], bool] | None = None,
    file_pattern: str = "*.bvh",
    skip_errors: bool = False,
    world_up: str = "auto",
    lr_mapping: dict[str, str] | None = None,
    harmonize: bool = False,
    retarget: bool = False,
    target_world_up: str | None = None,
    target_rest_forward: str | None = None,
    target_rest_up: str | None = None,
    target_euler_order: str | None = None,
    target_fps: float | None = None,
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
        The flag is recorded in the saved dataset's metadata and surfaced by :func:`load_preprocessed`, so downstream packing knows the arrays are already centered — pass ``center_root=False`` to the ``pack_to_*`` functions when repacking such clips.  (Re-centering a whole already-centered clip is a harmless no-op; the real hazard is windowed sub-clips, where re-centering zeroes the window's first frame and destroys the clip-relative trajectory.)

        With ``include_positions=True`` and ``position_centering="world"``
        the same shift is applied to every position vertex, keeping the
        two streams in one frame; with ``"skeleton"`` the positions are
        already root-relative and are left alone; with ``"first"`` the
        combination is **rejected** (see ``position_centering``).
    include_positions : bool
        If True, also store per-vertex 3-D positions — the stream
        skeleton action-recognition models consume.  Derived from
        :meth:`pybvh.Bvh.joint_positions` / :meth:`~pybvh.Bvh.node_positions`,
        both backed by pybvh's cached world-frame FK, so requesting them
        alongside a rotation representation costs one array derivation
        rather than a second kinematics pass.

        Unlike ``include_velocities`` and ``include_foot_contacts``,
        these are **not** static features: augmentation transforms them
        with the rest of the clip, and
        :func:`~pybvh_ml.add_joint_rotation_noise` re-derives them by
        forward kinematics.
    position_space : {"joint", "node"}
        Which index space to store.  ``"joint"`` (default) writes
        ``joint_pos``, index-aligned with ``joint_rot`` and with
        ``skeleton_info["edges"]``; ``"node"`` writes ``node_pos``,
        which includes end sites (fingertips, toe tips, head top) and
        pairs with ``node_edges`` / ``node_lr_pairs``.  One flag rather
        than two ``include_*`` booleans, because the two spaces are
        alternatives — ``node_pos`` already contains ``joint_pos``.

        Recorded in ``skeleton_info``, not in the dataset metadata: it
        is a topology fact — which index space, and therefore which
        ``V``, which edge list, which L/R pair list — sitting beside
        ``num_joints`` / ``num_nodes`` / ``edges``, exactly as
        ``foot_joints`` does.
    position_centering : {"world", "skeleton", "first"}
        Which frame the stored positions are in, passed to pybvh's
        ``centered=`` and recorded in the dataset metadata next to
        ``center_root``, whose analogue it is: a statement about the
        values rather than about the topology.

        ``"world"`` (default) keeps positions in the same frame as
        ``root_pos``, so :func:`~pybvh_ml.rotate_vertical` acts
        identically on both and a joint position already contains the
        root trajectory.  ``"skeleton"`` puts the root at the origin in
        every frame — the form most NTU-style pipelines feed a model,
        with the trajectory then carried only by ``root_pos``.
        ``"first"`` is pybvh's ground-plane centering.  The three
        coincide only for a clip whose root never moves.

        Recording it is mandatory for anything this library writes: a
        position array whose frame convention we failed to record is
        exactly the case a caller cannot recover from.  (A
        hand-assembled :class:`~pybvh_ml.MotionArrays` may honestly say
        ``None``; a dataset we wrote may not.)
    include_quaternions : bool
        If True, also store pre-computed quaternion arrays per clip
        (useful for runtime speed perturbation / dropout).  When
        ``representation="quat"`` the main joint data already is the
        quaternion array, so no duplicate is stored on disk —
        :func:`load_preprocessed` aliases ``clip["joint_quats"]`` to
        ``clip["joint_rot"]`` in that case.
    include_velocities : bool
        If True, compute per-joint linear velocities via
        :meth:`pybvh.Bvh.joint_velocities` (central stencil, edge
        padding — shape ``(F, J, 3)`` aligned with ``joint_rot`` /
        ``joint_angles``, no end sites) and store them per clip.
        Static features: **not** refreshed after augmentation, so use
        for evaluation / targets, not as augmentation-invariant
        training inputs.
    include_foot_contacts : bool
        If True, compute binary foot-contact labels via
        :meth:`pybvh.Bvh.foot_contacts` (default ``method="combined"``)
        and store them per clip along with the foot joint names in
        ``skeleton_info["foot_joints"]``.  Static features, same caveat
        as ``include_velocities``.
    foot_joints : list of str, optional
        Explicit foot joint names for contact detection.  ``None``
        (default) auto-detects from the first clip.  Required for
        footless or nonstandard rigs, where auto-detection finds
        nothing and pybvh's detector raises.  Only used with
        ``include_foot_contacts=True``.
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
        target is also resolved (the most common per-joint order across
        all joints of all clips); rotation-invariant representations
        skip this stage since channel layout is order-agnostic.

        Harmonization is pure reorientation/resampling: each actor's
        bone lengths are preserved (bone-length variation across
        actors is intrinsic data, see the skeleton-compatibility
        notes).  Pass ``retarget=True`` to additionally unify bone
        offsets.  Hierarchy mismatches raise :class:`ValueError`
        either way — from the post-harmonize compatibility check by
        default, or from the harmonize report under ``retarget=True``
        — rather than silently shipping a smaller dataset.  The
        resolved targets, the retarget choice, and per-stage
        modification counts land in the returned ``uniformity`` dict
        under ``uniformity["harmonized_to"]`` and are persisted in the
        saved dataset (``uniformity_json``).

        Default ``False`` keeps the explicit ``target_*`` kwargs as
        independent uniformization stages (current behavior).
    retarget : bool
        Only honored with ``harmonize=True``.  If True, pin the first
        clip (alphabetically first stem) as the harmonize reference:
        every other clip's bone offsets are retargeted to it, so the
        whole dataset shares one skeleton geometry — useful when the
        model should not need to be scale-invariant (e.g.
        fixed-topology GCNs).  Bone offsets only — root translations
        keep each clip's original scale (pybvh's ``retarget``
        semantics).  Default ``False`` preserves each actor's own bone
        proportions.
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
    target_fps : float or None
        Frame rate in Hz to resample every clip to, applied **before
        extraction** via :meth:`pybvh.Bvh.resample` (quaternion SLERP
        for rotations, linear for root position).  Resampling first is
        what makes it correct: ``joint_rot``, ``include_velocities``
        and ``include_foot_contacts`` are all derived from the resampled
        clip, so they describe the motion at the target rate.
        Decimating the saved arrays afterwards cannot reproduce this —
        velocities in particular are finite differences whose stencil
        baseline is set by the *original* ``frame_time``.

        Same dual semantics as ``target_world_up``: with
        ``harmonize=False`` (default) each clip is resampled directly;
        with ``harmonize=True`` it becomes the explicit frame-rate
        target for :func:`pybvh.harmonize`, overriding the audit
        majority.  ``None`` (default) defers to the dataset majority
        under ``harmonize=True`` — a mixed-rate dataset is unified to
        its most common rate — and leaves clips untouched otherwise.
    parallel : bool
        If True, load BVH files using a :class:`ThreadPoolExecutor`.
        Speeds up large directories; per-file I/O is the bottleneck.
    max_workers : int, optional
        Thread count when ``parallel=True``.  ``None`` defers to
        :class:`ThreadPoolExecutor`'s default.

    Notes
    -----
    **Uniformity warnings.** After loading, this function inspects
    every clip's frame rate, animation-derived ``world_up``, rest-pose
    forward direction, and rest-pose up axis.  It emits one aggregated
    :class:`UserWarning` per category when files disagree, plus a
    separate aggregated warning when any file's rest-pose up axis
    disagrees with its own animation-derived ``world_up`` (pybvh's
    per-file rest/animation-disagreement warning is suppressed during
    load in favor of this one batch-level message).  Warnings include
    the distribution of values, the first three example filenames per
    minority value, and the exact kwarg that would fix it
    (``target_fps``, ``target_world_up``, ``target_rest_forward``,
    ``target_rest_up``).  When the corresponding ``target_*`` kwarg is
    explicitly set, that category's check is skipped (the target value
    becomes the post-reorient ground truth).

    Returns
    -------
    dict
        Summary with keys: ``num_clips``, ``representation``,
        ``filenames``, ``skeleton_info``, ``uniformity``.
        ``uniformity`` is a dict of the form::

            {
              "fps":          {value: [stems, ...]},
              "world_up":     {value: [stems, ...]},
              "rest_forward": {value: [stems, ...]},
              "rest_up":      {value: [stems, ...]},
              "rest_anim_mismatch": [stems, ...],
              "harmonized_to":   {...},  # only when harmonize=True
              "applied_targets": {...},  # only when harmonize=False
            }

        The four distributions capture the **pre-transform** state of
        the dataset (useful for CI gates that want to fail on
        heterogeneity); what was then *done* to it is the other two
        keys, exactly one of which can be present.
        ``rest_anim_mismatch`` lists files whose rest-pose up axis
        disagrees with their animation-derived ``world_up`` — the
        condition ``target_rest_up`` repairs.  Rigs with an
        unmeasurable rest pose are filed under ``rest_up`` key
        ``"unknown"`` and excluded from ``rest_anim_mismatch``.

        When ``harmonize=True``, ``harmonized_to`` carries the
        resolved target signature (``target_fps``,
        ``target_world_up``, ``target_rest_up``,
        ``target_rest_forward``, ``target_euler_order`` — only those
        that were resolved), the ``retarget`` choice and pinned
        ``reference``, ``stage_counts`` (per-stage count of clips
        modified, from pybvh's ``HarmonizeReport.applied_stages``),
        and the serialized ``report`` itself (JSON-native ``dict``
        from ``dataclasses.asdict``).

        Otherwise ``applied_targets`` records the ``target_*`` kwargs
        this call applied directly, under the same names — absent when
        none was passed.  ``target_euler_order`` never appears: it is
        honored only under ``harmonize=True``, so recording it here
        would claim a transform that did not run.
    """
    bvh_dir = Path(bvh_dir)
    output_path = Path(output_path)
    # Fail before any file is parsed: a bad representation or output
    # extension used to surface only after the full directory load.
    _validate_representation(representation)
    _validate_output_suffix(output_path)
    if include_positions:
        _validate_position_settings(
            position_space, position_centering, center_root)

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
        uniformity, target_world_up, target_rest_forward, target_rest_up,
        target_fps, harmonize=harmonize)

    if harmonize:
        clips, stems = _run_harmonize(
            clips, stems, uniformity, representation,
            target_world_up, target_rest_forward, target_rest_up,
            target_euler_order, retarget, target_fps,
        )
    else:
        if target_rest_forward is not None:
            _reject_parallel_rest_forward(
                clips, stems, target_rest_forward, target_world_up)
        if target_rest_up is not None:
            _reject_degenerate_rest_up_targets(clips, stems, target_rest_up)

        # Stage order mirrors pybvh.harmonize: resample, then world-up
        # (which moves the whole scene), then the rest-pose axes.
        applied_targets: dict[str, str | float] = {}
        if target_fps is not None:
            clips = [b.resample(target_fps) for b in clips]
            applied_targets["target_fps"] = float(target_fps)
        if target_world_up is not None:
            clips = [b.reorient_world_up(target_world_up) or b for b in clips]
            applied_targets["target_world_up"] = target_world_up
        if target_rest_forward is not None:
            clips = [
                b.reorient_rest_forward(target_rest_forward) or b
                for b in clips]
            applied_targets["target_rest_forward"] = target_rest_forward
        if target_rest_up is not None:
            clips = [b.reorient_rest_up(target_rest_up) or b for b in clips]
            applied_targets["target_rest_up"] = target_rest_up
        if applied_targets:
            uniformity["applied_targets"] = applied_targets

    _check_skeleton_compatibility(clips, stems, representation)

    # Extract data per clip
    all_root_pos: list[npt.NDArray[np.float64]] = []
    all_joint_data: list[npt.NDArray[np.float64]] = []
    all_joint_quats: list[npt.NDArray[np.float64]] = []
    all_velocities: list[npt.NDArray[np.float64]] = []
    all_foot_contacts: list[npt.NDArray[np.float64]] = []
    all_positions: list[npt.NDArray[np.float64]] = []

    # Pin foot joints to one list (explicit, or auto-detected from the
    # first clip) so all clips produce contact arrays with the same shape.
    if include_foot_contacts and foot_joints is None:
        foot_joints = clips[0].auto_detect_foot_joints()

    # For representation="quat" the primary joint_data already is the
    # quaternion array — don't extract (or later store) a duplicate.
    want_quats = include_quaternions and representation != "quat"
    for bvh in clips:
        root_pos, joint_data, quats = _extract_primary_and_quats(
            bvh, representation, want_quaternions=want_quats)
        positions = None
        if include_positions:
            positions = (
                bvh.joint_positions(centered=position_centering)
                if position_space == "joint"
                else bvh.node_positions(centered=position_centering))
        if center_root and root_pos.shape[0] > 0:
            shift = root_pos[0:1]
            root_pos = root_pos - shift
            # "skeleton"-centered positions are already root-relative and
            # do not move; "first" is rejected up front.
            if positions is not None and position_centering == "world":
                positions = positions - shift[:, np.newaxis, :]
        all_root_pos.append(root_pos)
        all_joint_data.append(joint_data)
        if positions is not None:
            all_positions.append(positions)

        if want_quats:
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
    if include_positions:
        skel_info["position_space"] = position_space

    # Normalization stats (computed on the primary representation only;
    # velocities / foot contacts have their own natural scales).  Shares
    # the array-level core with the public compute_normalization_stats;
    # going via the already-extracted (and centered) arrays avoids a
    # second extraction pass and a redundant compatibility check.
    stats = _normalization_stats_from_arrays(all_root_pos, all_joint_data)

    # Positions get their own stats block rather than widening mean/std.
    # The existing vector's D = 3 + J*C layout is a public contract that
    # pack_to_flat, describe_features and the HumanML3D Mean.npy /
    # Std.npy convention are all written against; changing D based on a
    # preprocessing flag would make one file format mean two things.
    position_stats = None
    if include_positions:
        position_stats = _normalization_stats_from_arrays(
            all_root_pos, all_positions, include_root_pos=False)

    # Labels
    labels = None
    if label_fn is not None:
        labels = np.array([label_fn(s) for s in stems], dtype=np.int64)

    # Save
    ext = output_path.suffix.lower()
    saver = (_save_hdf5 if ext in (".hdf5", ".h5") else _save_npz)
    saver(output_path, all_root_pos, all_joint_data,
          all_joint_quats, all_velocities, all_foot_contacts,
          labels, stats, skel_info, representation, center_root,
          stems, uniformity,
          positions_list=all_positions,
          position_key=(_POSITION_STREAM_KEYS[position_space]
                        if include_positions else None),
          position_stats=position_stats,
          position_centering=position_centering if include_positions else None)

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
    center_root: bool,
    stems: list[str],
    uniformity: dict | None = None,
    *,
    positions_list: list | None = None,
    position_key: str | None = None,
    position_stats: dict | None = None,
    position_centering: str | None = None,
) -> None:
    """Save to .npz format."""
    save_dict: dict[str, object] = {
        "num_clips": np.array(len(root_pos_list)),
        "representation": np.array(representation),
        "center_root": np.array(center_root),
        "filenames": np.array(stems),
        "mean": stats["mean"],
        "std": stats["std"],
        "skeleton_info_json": np.array(json.dumps(skel_info)),
    }
    if position_centering is not None:
        save_dict["position_centering"] = np.array(position_centering)
    if position_stats is not None:
        save_dict["position_mean"] = position_stats["mean"]
        save_dict["position_std"] = position_stats["std"]
        save_dict["position_constant_channels"] = (
            position_stats["constant_channels"])
    if uniformity is not None:
        save_dict["uniformity_json"] = np.array(json.dumps(uniformity))
    if "constant_channels" in stats:
        save_dict["constant_channels"] = stats["constant_channels"]
    for i, (rp, jd) in enumerate(zip(root_pos_list, joint_data_list)):
        save_dict[f"clip_{i}_root_pos"] = rp
        save_dict[f"clip_{i}_joint_rot"] = jd
    if positions_list:
        for i, pos in enumerate(positions_list):
            save_dict[f"clip_{i}_{position_key}"] = pos
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
    center_root: bool,
    stems: list[str],
    uniformity: dict | None = None,
    *,
    positions_list: list | None = None,
    position_key: str | None = None,
    position_stats: dict | None = None,
    position_centering: str | None = None,
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
        f.attrs["center_root"] = bool(center_root)
        f.attrs["skeleton_info_json"] = json.dumps(skel_info)
        if position_centering is not None:
            f.attrs["position_centering"] = position_centering
        if uniformity is not None:
            f.attrs["uniformity_json"] = json.dumps(uniformity)

        f.create_dataset("mean", data=stats["mean"])
        f.create_dataset("std", data=stats["std"])
        if "constant_channels" in stats:
            f.create_dataset("constant_channels", data=stats["constant_channels"])
        if position_stats is not None:
            f.create_dataset("position_mean", data=position_stats["mean"])
            f.create_dataset("position_std", data=position_stats["std"])
            f.create_dataset("position_constant_channels",
                             data=position_stats["constant_channels"])
        # Variable-length UTF-8 strings — dtype="S" (fixed ASCII bytes)
        # crashes with UnicodeEncodeError on non-ASCII filename stems.
        f.create_dataset(
            "filenames", data=stems,
            dtype=h5py.string_dtype(encoding="utf-8"))

        if labels is not None:
            f.create_dataset("labels", data=labels)

        for i, (rp, jd) in enumerate(zip(root_pos_list, joint_data_list)):
            grp = f.create_group(f"clip_{i}")
            grp.create_dataset("root_pos", data=rp)
            grp.create_dataset("joint_rot", data=jd)
            grp.attrs["filename"] = stems[i]
            if positions_list:
                grp.create_dataset(position_key, data=positions_list[i])
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
        ``joint_rot`` (named ``joint_data`` in datasets written before
        pybvh-ml 0.5.0; both keys load, the new name is what you read),
        optionally ``joint_quats`` / ``velocities`` /
        ``foot_contacts`` / ``joint_pos`` / ``node_pos``), ``labels``,
        ``mean``, ``std``, ``skeleton_info``, ``representation``,
        ``filenames``, ``center_root``, ``uniformity``,
        ``position_centering``, ``position_stats``.  Also includes
        ``constant_channels`` when the file was written by
        pybvh-ml >= 0.3 (absent for older files).

        ``position_centering`` is the frame the stored positions are in
        (``None`` when the dataset carries none, and for every file
        written before pybvh-ml 0.6.0).  It has to be threaded into every
        :class:`~pybvh_ml.MotionArrays` built from these clips — the
        steps that depend on it only ever see the container, not this
        dict.  :meth:`~pybvh_ml.torch.MotionDataset.from_preprocessed`
        does that for you.

        ``position_stats`` is the positions' own
        ``{"mean", "std", "constant_channels"}`` block over the
        ``(F, V*3)`` flattening, or ``None``.  It is deliberately
        separate from ``mean`` / ``std``, whose ``D = 3 + J*C`` layout is
        a public contract.  Ignoring it is a legitimate choice: ST-GCN
        pipelines more commonly root-center or normalize by bone length
        than z-score raw coordinates.

        ``uniformity`` is the axis-uniformity audit recorded at
        preprocessing time: the pre-transform frame-rate / world-up /
        rest-forward / rest-up distributions, plus a record of what was
        applied to them — ``harmonized_to`` (resolved targets,
        ``retarget`` choice, and the full harmonize report) when the
        dataset was built with ``harmonize=True``, or
        ``applied_targets`` (the ``target_*`` kwargs applied directly)
        when it was not.  Files written before pybvh-ml 0.5.0 load it
        as ``None``.

        ``center_root`` is the flag the dataset was preprocessed with (files written before pybvh-ml 0.5.0 don't record it, so it loads as ``None``).  When it is ``True``, the stored ``root_pos`` arrays are already centered — repack them with ``pack_to_*(..., center_root=False)``.

        ``skeleton_info`` always carries every key
        :func:`~pybvh_ml.skeleton.get_skeleton_info` documents, whatever
        version wrote the file: keys an older dataset never recorded
        (``world_up`` / ``rest_forward`` / ``rest_up`` before 0.5.0, the
        node-space block and ``fk_topology`` before 0.6.0) read back as
        ``None`` rather than being absent, so consumers can index them
        directly.  ``position_space`` is the exception, and it follows
        the ``foot_joints`` precedent: it is present only when the
        dataset stores positions, so "not requested" stays
        distinguishable from "requested and empty".
    """
    path = Path(path)
    _validate_output_suffix(path)
    ext = path.suffix.lower()
    if ext == ".hdf5" or ext == ".h5":
        return _load_hdf5(path)
    return _load_npz(path)


# Keys :func:`~pybvh_ml.skeleton.get_skeleton_info` always produces.
# Datasets written by older pybvh-ml versions predate some of them, and
# a key that is sometimes absent forces every consumer into a `.get()`
# dance; loading fills them with None so the shape is version-stable.
_SKELETON_INFO_KEYS = (
    "num_joints", "joint_names", "edges", "euler_orders",
    "lr_pairs", "lr_mapping", "world_up", "rest_forward", "rest_up",
    "num_nodes", "node_names", "node_edges", "node_lr_pairs",
    "end_site_indices", "fk_topology", "mismatched_end_site_pairs",
)

# Per-clip arrays stored beside the mandatory root_pos / joint_rot pair,
# each written only when its preprocessing flag was set.  Listed once
# because both loaders iterate it, and they must stay in step.
#
# The first three are *static* features: unlike joint_rot they are not
# refreshed by augmentation, so they belong to evaluation and targets
# rather than to augmentation-invariant training inputs.  ``joint_pos``
# and ``node_pos`` differ in kind — they are augmentable streams of
# MotionArrays, transformed by every geometric step and re-derived by
# add_joint_rotation_noise — and are listed here only because the
# loaders read them the same way.
_OPTIONAL_CLIP_STREAMS = (
    "joint_quats", "velocities", "foot_contacts", "joint_pos", "node_pos")


def _normalize_skeleton_info(skel_info: dict) -> dict:
    """Fill the ``skeleton_info`` keys an older dataset never recorded.

    Optional-by-design keys (``foot_joints``, ``body_partitions``) are
    left absent: those signal a preprocessing choice, and inventing a
    ``None`` for them would make "not requested" indistinguishable from
    "requested and empty".
    """
    normalized: dict = dict.fromkeys(_SKELETON_INFO_KEYS)
    normalized.update(skel_info)
    return normalized


def _load_npz(path: Path) -> dict:
    """Load from .npz format."""
    data = np.load(path, allow_pickle=False)
    num_clips = int(data["num_clips"])
    representation = str(data["representation"])
    filenames = list(data["filenames"])
    skel_info = _normalize_skeleton_info(
        json.loads(str(data["skeleton_info_json"])))

    clips = []
    for i in range(num_clips):
        rot_key = (f"clip_{i}_joint_rot" if f"clip_{i}_joint_rot" in data
                   else f"clip_{i}_joint_data")
        clip: dict[str, npt.NDArray[np.float64]] = {
            "root_pos": data[f"clip_{i}_root_pos"],
            "joint_rot": data[rot_key],
        }
        for extra in _OPTIONAL_CLIP_STREAMS:
            key = f"clip_{i}_{extra}"
            if key in data:
                clip[extra] = data[key]
        # Quat datasets carry no duplicate joint_quats on disk — the
        # primary joint_rot already is the quaternion array.
        if representation == "quat" and "joint_quats" not in clip:
            clip["joint_quats"] = clip["joint_rot"]
        clips.append(clip)

    result: dict = {
        "clips": clips,
        "mean": data["mean"],
        "std": data["std"],
        "skeleton_info": skel_info,
        "representation": representation,
        "filenames": filenames,
        # Files written before pybvh-ml 0.5.0 don't record the flag.
        "center_root": (bool(data["center_root"])
                        if "center_root" in data.files else None),
        # Files written before pybvh-ml 0.5.0 don't record the audit.
        "uniformity": (json.loads(str(data["uniformity_json"]))
                       if "uniformity_json" in data.files else None),
        # Only datasets written with include_positions=True have one.
        "position_centering": (str(data["position_centering"])
                               if "position_centering" in data.files
                               else None),
        "position_stats": (
            {"mean": data["position_mean"],
             "std": data["position_std"],
             "constant_channels": data["position_constant_channels"]}
            if "position_mean" in data.files else None),
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
        skel_info = _normalize_skeleton_info(
            json.loads(str(f.attrs["skeleton_info_json"])))

        clips = []
        for i in range(num_clips):
            grp = f[f"clip_{i}"]
            rot_key = "joint_rot" if "joint_rot" in grp else "joint_data"
            clip: dict[str, npt.NDArray[np.float64]] = {
                "root_pos": grp["root_pos"][()],
                "joint_rot": grp[rot_key][()],
            }
            for extra in _OPTIONAL_CLIP_STREAMS:
                if extra in grp:
                    clip[extra] = grp[extra][()]
            # Quat datasets carry no duplicate joint_quats on disk — the
            # primary joint_rot already is the quaternion array.
            if representation == "quat" and "joint_quats" not in clip:
                clip["joint_quats"] = clip["joint_rot"]
            clips.append(clip)

        result: dict = {
            "clips": clips,
            "mean": f["mean"][()],
            "std": f["std"][()],
            "skeleton_info": skel_info,
            "representation": representation,
            "filenames": [s.decode() if isinstance(s, bytes) else s
                          for s in f["filenames"][()]],
            # Files written before pybvh-ml 0.5.0 don't record the flag.
            "center_root": (bool(f.attrs["center_root"])
                            if "center_root" in f.attrs else None),
            # Files written before pybvh-ml 0.5.0 don't record the audit.
            "uniformity": (json.loads(str(f.attrs["uniformity_json"]))
                           if "uniformity_json" in f.attrs else None),
            # Only datasets written with include_positions=True have one.
            "position_centering": (str(f.attrs["position_centering"])
                                   if "position_centering" in f.attrs
                                   else None),
            "position_stats": (
                {"mean": f["position_mean"][()],
                 "std": f["position_std"][()],
                 "constant_channels": f["position_constant_channels"][()]}
                if "position_mean" in f else None),
        }
        if "constant_channels" in f:
            result["constant_channels"] = f["constant_channels"][()]
        if "labels" in f:
            result["labels"] = f["labels"][()]
        else:
            result["labels"] = None

    return result
