"""Array-level augmentation for ML pipelines.

Operates on pre-extracted NumPy arrays without Bvh objects.
All functions accept any rotation representation supported by pybvh:
``"quat"``, ``"6d"``, ``"axisangle"``, ``"rotmat"``, or ``"euler"``.
Euler arrays additionally require an ``euler_orders`` kwarg.

Every function takes a :class:`~pybvh_ml.MotionArrays` as its single
positional argument and returns a new one; every other parameter is
keyword-only.  The container is a distinct type, so passing it
positionally cannot be confused with anything else — whereas the loose
``(root_pos, joint_data)`` pair it replaces was two shape-compatible
ndarrays a swapped call would have silently corrupted.  Call with
``rotate_vertical(arrays, angle=..., up_axis=..., representation=...)``.

Angles are in **radians** by default, matching pybvh's convention.  The
functions that take one accept ``degrees=True`` to interpret it in
degrees instead, mirroring pybvh's own opt-in flag.

**Streams.** A sample may carry rotations, positions, or both, and the
governing rule is that *a step must handle every stream the sample
carries, or refuse it* — a pipeline never carries a stream a step left
behind.  Each function declares what it handles with
:func:`handles_streams`, and the declaration is checked before the step
runs.  All four geometric steps (:func:`rotate_vertical`,
:func:`mirror`, :func:`speed_perturbation_arrays`,
:func:`dropout_arrays`) and both noise functions handle every stream;
only the two keypoint-jitter functions decline anything.
``representation`` is consequently ``str | None`` throughout, required
only when the sample carries ``joint_rot``.

**Output dtype.** Every function here computes in ``float64`` — pybvh's dtype, and the only one its conversions are exact in — and returns each stream in the dtype it was given, per stream, so a ``float32`` clip comes back ``float32`` without the arithmetic having been done in single precision. :class:`~pybvh_ml.AugmentationPipeline` does the same across a whole run, which is what keeps a result's dtype independent of which probabilistic steps fired for that sample. Widening is lossless, so the ``float32`` result is exactly the ``float64`` result narrowed. The per-stream rule earns its keep with positions: ``float32`` keypoints beside ``float64`` rotations is the ordinary ST-GCN case.

**Output storage.** A function's result may share storage with its input for a stream it does not touch — :func:`add_root_position_noise` returns the input's ``joint_rot`` by reference under skeleton-centered positions, and the keypoint-jitter functions return the stream they were not pointed at. This is safe rather than sloppy: :class:`~pybvh_ml.MotionArrays` fields are read-only views, so no caller (this package included) can write through the shared buffer, and the alternative — copying every untouched stream — would allocate a full clip per step for nothing. The stronger guarantee, freshly allocated arrays regardless of what ran, belongs to :class:`~pybvh_ml.AugmentationPipeline`, which is the surface a data loader calls; reach for it (or ``np.array(...)``) if you need storage you own.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import numpy.typing as npt

from pybvh import frames_to_node_positions, parse_axis, rotations

from .arrays import (
    STREAM_NAMES,
    MotionArrays,
    require_joint_rot,
    require_position_centering,
)


_POSITION_STREAMS = ("joint_pos", "node_pos")


# =========================================================================
# Stream declarations
# =========================================================================

_DEFAULT_STREAM_SUPPORT = frozenset({"root_pos", "joint_rot"})
"""What a step that declares nothing is assumed to handle.

Exactly the capability of every augmentation written against pybvh-ml
<= 0.5, so no existing pipeline changes behaviour — and the first
positions-carrying sample raises a message naming
:func:`handles_streams` instead of silently dropping the stream.
"""


def handles_streams(*streams: str) -> Callable[[Callable], Callable]:
    """Declare which :class:`~pybvh_ml.MotionArrays` streams a step handles.

    The coherence rule this enforces: **a step must handle every stream
    the sample carries, or refuse it.**  A pipeline never carries a
    stream a step left behind, so a rotation-only step meeting a sample
    with positions raises rather than returning positions that no longer
    match the rotations beside them.

    **"Handles" means the stream is left correct**, by either of two
    routes:

    - **transformed** from its own input — what all four geometric steps
      do (:func:`rotate_vertical`, :func:`mirror`,
      :func:`speed_perturbation_arrays`, :func:`dropout_arrays`); or
    - **re-derived** from another stream — what
      :func:`add_joint_rotation_noise` does, replacing the position
      streams with forward kinematics of the noised rotations rather
      than transforming the incoming ones.

    It does **not** mean the positions stay the exact forward kinematics
    of the rotations beside them, except immediately after a
    re-derivation.  Two divergences are intrinsic to the math, and every
    pipeline in the field has them:

    - **Mirror.** Positions reflect exactly in world space; rotations
      reflect in parent-local space.  The two agree when the rest pose is
      laterally symmetric and diverge on rigs with asymmetric offsets,
      with the error accumulating down the chain.  Each stream stays
      individually correct; the pair stops being FK partners.
    - **Speed perturbation and dropout.** Positions are linearly
      interpolated, rotations slerped — chord versus arc.  They agree at
      the knots and drift between them.

    And a re-derivation **discards whatever stream-specific history the
    positions carried**: on a rig with asymmetric rest offsets,
    ``[mirror, add_joint_rotation_noise]`` ends with FK of
    locally-mirrored rotations, throwing away the world-exact reflection
    the position stream held, while ``[add_joint_rotation_noise,
    mirror]`` keeps it.  Both are defensible and neither is a bug, but a
    user who does not know reads the difference as one.

    Undeclared steps default to ``{"root_pos", "joint_rot"}``.  Decorate
    a custom step once it genuinely transforms the position streams too::

        @handles_streams("root_pos", "joint_rot", "joint_pos")
        def my_step(arrays, *, scale):
            return arrays.replace(joint_pos=arrays.joint_pos * scale)

    Parameters
    ----------
    *streams : str
        Names from :data:`~pybvh_ml.arrays.STREAM_NAMES`.

    Returns
    -------
    callable
        The decorator, which records the declaration on the function and
        returns it unchanged.

    See Also
    --------
    stream_support : Read a step's declaration back.
    """
    unknown = [s for s in streams if s not in STREAM_NAMES]
    if unknown:
        raise ValueError(
            f"handles_streams got unknown stream name(s) {unknown}; "
            f"choose from {list(STREAM_NAMES)}")
    declared = frozenset(streams)

    def decorate(fn: Callable) -> Callable:
        fn.handled_streams = declared  # type: ignore[attr-defined]
        return fn

    return decorate


def _unwrap_step(fn: Callable) -> Callable:
    """Peel :func:`functools.partial` layers off a configured step.

    A step is any callable, and baking kwargs in with ``partial`` is one
    of the two natural ways to write one (the other, a callable
    instance, carries its declaration on the class and needs no
    unwrapping).  Without this a partial-wrapped built-in would read as
    an undeclared custom step and be refused a positions-carrying
    sample.
    """
    seen = fn
    while getattr(seen, "handled_streams", None) is None:
        wrapped = getattr(seen, "func", None)
        if wrapped is None:
            return fn
        seen = wrapped
    return seen


def stream_support(fn: Callable) -> frozenset[str]:
    """The streams *fn* declares it handles.

    ``{"root_pos", "joint_rot"}`` for a step that declares nothing — see
    :func:`handles_streams` for what "handles" means and why that is the
    right default.

    Parameters
    ----------
    fn : callable
        An augmentation step.

    Returns
    -------
    frozenset of str
    """
    declared = getattr(_unwrap_step(fn), "handled_streams", None)
    return _DEFAULT_STREAM_SUPPORT if declared is None else declared


def _step_label(fn: Callable) -> str:
    """Readable name for a step, for the precondition messages."""
    name = getattr(fn, "__name__", None)
    if name is not None:
        return name
    wrapped = getattr(fn, "func", None)
    if wrapped is not None:
        return _step_label(wrapped)
    return type(fn).__name__


_UNHANDLED_STREAM_ADVICE = {
    "joint_rot": (
        "positions are derived from rotations, not the other way round — "
        "recovering rotations from jittered positions is inverse "
        "kinematics, which pybvh-ml does not do. Jitter the rotations with "
        "add_joint_rotation_noise instead (it re-derives the positions by "
        "forward kinematics), or drop joint_rot from the sample"),
    "joint_pos": (
        "use add_joint_position_noise for joint-space keypoint jitter, or "
        "drop joint_pos from the sample"),
    "node_pos": (
        "use add_node_position_noise for node-space keypoint jitter, or "
        "drop node_pos from the sample"),
    "root_pos": (
        "every step in this package handles root_pos; a step that cannot "
        "should not be in a pipeline"),
}


def _check_stream_support(fn: Callable, arrays: MotionArrays) -> None:
    """Refuse a sample carrying a stream *fn* did not declare.

    One subset test.  The pipeline runs it at ``__call__`` entry for
    every configured step, before any of them runs, so a ``p=0.1`` step
    does not raise on one sample in ten; each public function runs it for
    itself so a direct call is checked too.
    """
    supported = stream_support(fn)
    unhandled = sorted(arrays.present_streams - supported)
    if not unhandled:
        return

    label = _step_label(fn)
    declared = getattr(_unwrap_step(fn), "handled_streams", None)
    if declared is None:
        raise ValueError(
            f"augmentation step {label!r} declares no stream support, so it "
            f"defaults to {sorted(_DEFAULT_STREAM_SUPPORT)} — the capability "
            f"of every step written against pybvh-ml <= 0.5 — but this "
            f"sample also carries {unhandled}. Decorate it with "
            f"@handles_streams({', '.join(repr(s) for s in sorted(supported | set(unhandled)))}) "
            f"once it transforms those streams too, or drop them from the "
            f"sample.")
    reasons = "; ".join(
        f"{name}: {_UNHANDLED_STREAM_ADVICE[name]}" for name in unhandled)
    raise ValueError(
        f"augmentation step {label!r} handles {sorted(supported)} but this "
        f"sample carries {unhandled}. {reasons}.")


def _result(
    source: MotionArrays,
    root_pos: npt.ArrayLike,
    joint_rot: npt.ArrayLike | None = None,
    joint_pos: npt.ArrayLike | None = None,
    node_pos: npt.ArrayLike | None = None,
) -> MotionArrays:
    """Build a result carrying *source*'s per-stream dtypes and centering.

    Rotation math runs in ``float64`` — pybvh's dtype, and the only one
    the conversions are exact in — so every function here computes in
    double whatever it was given.  Casting back at the return keeps that
    an implementation detail instead of a visible dtype change, and
    keeps it out of the one place it would be worst: a probabilistic
    pipeline, where the dtype would otherwise depend on which steps
    happened to fire for that sample.

    Each stream follows its own input, since a container may legitimately
    carry ``float32`` positions next to ``float64`` rotations.
    ``np.asarray`` is a no-op when the dtypes already match, so the
    ``float64`` path costs nothing.
    """
    def _cast(value, name):
        if value is None:
            return None
        reference = getattr(source, name)
        dtype = None if reference is None else reference.dtype
        return np.asarray(value, dtype=dtype)

    return MotionArrays(
        root_pos=np.asarray(root_pos, dtype=source.root_pos.dtype),
        joint_rot=_cast(joint_rot, "joint_rot"),
        joint_pos=_cast(joint_pos, "joint_pos"),
        node_pos=_cast(node_pos, "node_pos"),
        # A step never invents or drops a position stream, so the frame
        # the incoming positions were in is still the frame the outgoing
        # ones are in.
        position_centering=(source.position_centering
                            if joint_pos is not None or node_pos is not None
                            else None),
    )


def _validate_sigma(sigma: float, name: str = "sigma") -> None:
    """Reject a negative noise standard deviation."""
    if sigma < 0:
        raise ValueError(f"{name} must be >= 0, got {sigma}")


def _as_radians(angle: float, degrees: bool) -> float:
    """Interpret an angle argument under the ``degrees`` flag."""
    return float(np.radians(angle)) if degrees else float(angle)


def _validate_drop_rate(drop_rate: float) -> None:
    """Reject drop rates outside the documented ``[0, 1)`` range."""
    if not 0.0 <= drop_rate < 1.0:
        raise ValueError(f"drop_rate must be in [0, 1), got {drop_rate}")


def _to_quats(
    joint_data: npt.NDArray[np.float64],
    representation: str,
    euler_orders: list[str] | None,
) -> npt.NDArray[np.float64]:
    """Convert joint data to quaternion space for augmentation math."""
    if representation == "quat":
        return joint_data
    if representation == "euler":
        if euler_orders is None:
            raise ValueError(
                "euler_orders is required when representation='euler'")
        return rotations.convert(
            joint_data, "euler", "quat", order=euler_orders)
    if representation == "rotmat":
        # pybvh-ml carries rotmat flat as (F, J, 9); pybvh's
        # rotations.convert expects (..., 3, 3).  Adapt at the boundary.
        if joint_data.ndim != 3:
            raise ValueError(
                f"representation='rotmat' expects joint data flat as "
                f"(F, J, 9) — the layout convert_arrays documents and "
                f"produces — got shape {joint_data.shape}. Reshape a "
                f"(F, J, 3, 3) array with .reshape(F, J, 9) first; "
                f"augmentation returns the flat layout either way.")
        F, J = joint_data.shape[:2]
        return rotations.convert(
            joint_data.reshape(F, J, 3, 3), "rotmat", "quat")
    return rotations.convert(joint_data, representation, "quat")


def _from_quats(
    quats: npt.NDArray[np.float64],
    representation: str,
    euler_orders: list[str] | None,
) -> npt.NDArray[np.float64]:
    """Convert quaternions back to the original representation."""
    if representation == "quat":
        return quats
    if representation == "euler":
        return rotations.convert(
            quats, "quat", "euler", order=euler_orders)
    if representation == "rotmat":
        R = rotations.convert(quats, "quat", "rotmat")
        F, J = R.shape[:2]
        return R.reshape(F, J, 9)
    return rotations.convert(quats, "quat", representation)


def _build_rotation_quat(
    angle: float,
    up_idx: int,
) -> npt.NDArray[np.float64]:
    """Build a unit quaternion for rotation around a cardinal axis (radians)."""
    half = angle / 2.0
    q = np.array([np.cos(half), 0.0, 0.0, 0.0])
    q[1 + up_idx] = np.sin(half)
    return q


def _build_rotation_matrix(
    angle: float,
    up_idx: int,
) -> npt.NDArray[np.float64]:
    """Build a 3×3 rotation matrix for rotation around a cardinal axis (radians)."""
    return rotations.quat_to_rotmat(_build_rotation_quat(angle, up_idx))


def _mirror_sign_quat(lateral_idx: int) -> npt.NDArray[np.float64]:
    """Sign vector for quaternion reflection across the lateral plane.

    Negates the two imaginary components NOT corresponding to the
    lateral axis.  Derived from ``R' = S @ R @ S``.
    """
    signs = np.ones(4)
    for ax in range(3):
        if ax != lateral_idx:
            signs[1 + ax] = -1.0
    return signs


def _mirror_sign_rot6d(lateral_idx: int) -> npt.NDArray[np.float64]:
    """Sign vector for 6D reflection across the lateral plane.

    For the first two columns of the rotation matrix stored as
    ``[col0(3), col1(3)]``, element ``(i, j)`` of the reflected
    matrix is ``s_i * s_j * R[i, j]`` where ``s[lateral_idx] = -1``.
    """
    s = np.ones(3)
    s[lateral_idx] = -1.0
    return np.array([
        s[0] * s[0], s[1] * s[0], s[2] * s[0],
        s[0] * s[1], s[1] * s[1], s[2] * s[1],
    ])


def _swap_lr_pairs(
    data: npt.NDArray[np.float64],
    lr_joint_pairs: list[tuple[int, int]],
) -> None:
    """Swap L/R vertex data in-place along axis 1.

    Works on any ``(F, V, C)`` array — rotations in any representation,
    or ``(F, V, 3)`` positions in either index space — but it **mutates**,
    so it must be handed a writable working copy and never a container
    field (which is a read-only view).
    """
    for lj, rj in lr_joint_pairs:
        data[:, lj], data[:, rj] = (
            data[:, rj].copy(), data[:, lj].copy())


def _working_positions(
    arrays: MotionArrays,
) -> dict[str, npt.NDArray[np.float64] | None]:
    """Writable ``float64`` copies of whichever position streams are present.

    Copies rather than views because container fields are read-only as of
    0.5.0 and several of the position branches (``_swap_lr_pairs``, the
    dropout in-fill) write in place.
    """
    return {
        name: (None if getattr(arrays, name) is None
               else np.array(getattr(arrays, name), dtype=np.float64))
        for name in _POSITION_STREAMS
    }


def _require_representation(
    arrays: MotionArrays,
    representation: str | None,
    caller: str,
) -> str:
    """Return *representation*, raising when the sample needs one and has none.

    ``representation`` is optional as of 0.6.0 — a positions-only sample
    has no rotation array for the token to describe — but a sample
    carrying ``joint_rot`` still cannot be interpreted without it.
    """
    if representation is None:
        raise ValueError(
            f"{caller} needs a representation to interpret joint_rot "
            f"(one of 'quat', '6d', 'axisangle', 'rotmat', 'euler'). It is "
            f"optional only for a sample that carries no rotations — this "
            f"one has joint_rot of shape {arrays.joint_rot.shape}.")
    return representation


def _rotation_noise_preconditions(
    arrays: MotionArrays,
    fk_topology: object | None,
    world_up: str | None,
) -> None:
    """Everything :func:`add_joint_rotation_noise` needs, checked up front.

    Shared with :class:`~pybvh_ml.AugmentationPipeline`, which runs it at
    ``__call__`` entry before any step fires.  None of these may live
    inside the step body: under ``p < 1`` the raise would be stochastic,
    and a configuration error that surfaces on one sample in ten is a
    configuration error that reaches production.
    """
    if arrays.joint_rot is None:
        raise ValueError(
            "add_joint_rotation_noise has nothing to noise: this sample "
            "carries no joint_rot. The step is meaningless on a "
            "rotation-free clip — use add_joint_position_noise (or "
            "add_node_position_noise) for keypoint jitter, and note that "
            "AugmentationPipeline.standard(representation=None) skips this "
            "step for exactly this reason.")
    if arrays.joint_pos is None and arrays.node_pos is None:
        return
    if fk_topology is None:
        raise ValueError(
            "add_joint_rotation_noise refreshes the position streams by "
            "forward kinematics, so it needs fk_topology= — this sample "
            "carries positions and none was given. Rebuild one from the "
            "dataset with build_fk_topology(skeleton_info), or let "
            "AugmentationPipeline.standard(skeleton_info) wire it.")
    centering = require_position_centering(
        arrays, "add_joint_rotation_noise")
    if centering == "first" and world_up is None:
        raise ValueError(
            "add_joint_rotation_noise needs world_up= to refresh positions "
            "under position_centering='first': ground-plane centering has "
            "to know which coordinate it leaves untouched, and an "
            "FkTopology carries no gravity direction. Pass "
            "skeleton_info['world_up'].")


def _refresh_positions_by_fk(
    *,
    joint_rot: npt.NDArray[np.float64],
    root_pos: npt.NDArray[np.float64],
    representation: str,
    fk_topology: object,
    world_up: str | None,
    position_centering: str,
    want_joint_pos: bool,
    want_node_pos: bool,
) -> dict[str, npt.NDArray[np.float64] | None]:
    """Recompute the requested position streams from *joint_rot*.

    Forward kinematics consumes Euler angles in radians, so a quat / 6d /
    rotmat / axisangle stream converts first — through the array-level
    :func:`pybvh.rotations.convert` rather than the container-level
    :func:`~pybvh_ml.convert_arrays`, which would rebuild a
    ``MotionArrays`` around an array FK wants bare.  The orders used are
    the **topology's**, since those are what its ``joint_angles``
    argument is interpreted under.  Both call sites hand over the
    quaternions they already hold, so the conversion is one hop rather
    than two.

    Passing the container's own ``root_pos`` in means an already-centered
    clip yields positions carrying the identical shift, so ``center_root``
    needs no separate correction here.
    """
    if not want_joint_pos and not want_node_pos:
        return {"joint_pos": None, "node_pos": None}

    if representation == "euler":
        euler = joint_rot
    else:
        euler = rotations.convert(
            (joint_rot.reshape(*joint_rot.shape[:2], 3, 3)
             if representation == "rotmat" else joint_rot),
            representation, "euler", order=list(fk_topology.euler_orders))

    node_positions = np.reshape(
        frames_to_node_positions(
            fk_topology, root_pos, euler,
            centered=position_centering, up=world_up),
        (root_pos.shape[0], -1, 3))

    return {
        "joint_pos": (node_positions[:, _joint_node_indices(fk_topology)]
                      if want_joint_pos else None),
        "node_pos": node_positions if want_node_pos else None,
    }


def _joint_node_indices(fk_topology: object) -> npt.NDArray[np.intp]:
    """Node index of each joint column, so a node array slices to joint space.

    ``joint_idx >= 0`` is the node→joint mask, and inverting it rather
    than masking directly is what makes the slice independent of node
    order: masking assumes ``joint_idx`` counts up in node order, which
    :attr:`pybvh.Bvh.fk_topology` produces but a hand-built topology
    (which pybvh validates but does not reorder) need not.
    """
    joint_idx = np.asarray(fk_topology.joint_idx)
    is_joint = joint_idx >= 0
    indices = np.empty(int(is_joint.sum()), dtype=np.intp)
    indices[joint_idx[is_joint]] = np.flatnonzero(is_joint)
    return indices


def _require_pairs(
    pairs: list[tuple[int, int]] | None,
    param: str,
    stream: str,
) -> None:
    """Reject a missing L/R pair list for a stream that needs one.

    ``None`` is not read as "no pairs": an empty list is the way to say
    that (and reflects every vertex in place, correct only for an
    entirely midline skeleton), so treating an omitted argument the same
    way would silently mirror half a skeleton.
    """
    if pairs is None:
        key = "lr_pairs" if param == "lr_joint_pairs" else "node_lr_pairs"
        raise ValueError(
            f"mirror needs {param}= to know which vertices swap sides, "
            f"because this sample carries {stream}. Pass "
            f"skeleton_info[{key!r}], or an explicit [] if the skeleton is "
            f"entirely midline — omitting it is not the same statement.")


# =========================================================================
# Public augmentation functions
# =========================================================================

@handles_streams(*STREAM_NAMES)
def rotate_vertical(
    arrays: MotionArrays,
    *,
    angle: float,
    up_axis: str,
    representation: str | None = None,
    degrees: bool = False,
    euler_orders: list[str] | None = None,
) -> MotionArrays:
    """Rotate joint arrays around the vertical axis.

    In rotation space only the root joint (index 0) and root position are
    modified; non-root joints are in parent-local space and stay
    unchanged.  Position streams are **all** rotated, since they are
    world coordinates rather than parent-local ones — this is the one
    place where the rotation and position analogs differ structurally.

    Parameters
    ----------
    arrays : MotionArrays
        Any combination of streams.  ``representation`` is required only
        when ``joint_rot`` is present.
    angle : float
        Rotation angle in radians, or degrees when ``degrees=True``.
    degrees : bool, optional
        Interpret ``angle`` in degrees.  Default False (radians), the
        convention throughout pybvh and pybvh-ml; the flag exists so a
        caller whose configs are written in degrees can pass them
        straight through instead of converting at every call site.
    up_axis : str
        Signed axis string: ``'+x'``, ``'-x'``, ``'+y'``, ``'-y'``,
        ``'+z'``, or ``'-z'``.  The sign flips the rotation direction,
        so ``'+y'`` and ``'-y'`` produce opposite yaws for the same
        ``angle``.  Typically ``bvh.world_up``.
    representation : str, optional
        One of ``"quat"``, ``"6d"``, ``"axisangle"``,
        ``"rotmat"``, ``"euler"``.  Required when ``arrays`` carries
        ``joint_rot``; ignored for a positions-only sample.
    euler_orders : list of str, optional
        Per-joint Euler order strings (e.g. ``["ZYX", "ZYX", ...]``).
        Required when ``representation="euler"``, ignored otherwise.

    Returns
    -------
    MotionArrays

    Notes
    -----
    The rotation is about the **world origin**, not about the
    character: ``root_pos`` is rotated as a set of points, so a clip
    whose root sits away from the origin sweeps along an arc rather
    than turning on the spot.  The alternative convention is a pivot at
    the character — typically the first frame's root projected to the
    ground plane — which is turn-in-place.

    The two coincide exactly when the clip's first-frame root is at the
    origin, which is what ``center_root=True`` produces, so the
    packing and Dataset paths (where centering is the default) already
    get turn-in-place from this function.  On uncentered arrays, center
    before rotating and add the offset back if that is what you want.

    The same origin caveat applies to the position streams, and it is
    where a ``position_centering="skeleton"`` clip differs: those
    positions are root-relative, so the rotation is always about the
    character and always turn-in-place, whatever the root trajectory
    does.  The function needs no knowledge of the centering to be
    correct — a rotation about the origin is linear, so it commutes with
    the constant shift that distinguishes the three frames.
    """
    _check_stream_support(rotate_vertical, arrays)
    root_pos = np.array(arrays.root_pos, dtype=np.float64)
    positions = _working_positions(arrays)

    up = parse_axis(up_axis)
    up_idx, up_sign = up.index, up.sign
    signed_angle = _as_radians(angle, degrees) * up_sign
    R_vert = _build_rotation_matrix(signed_angle, up_idx)
    new_root_pos = (R_vert @ root_pos.T).T
    for name, value in positions.items():
        if value is not None:
            positions[name] = value @ R_vert.T

    if arrays.joint_rot is None:
        return _result(arrays, new_root_pos, None, **positions)

    representation = _require_representation(
        arrays, representation, "rotate_vertical")
    joint_data = np.array(arrays.joint_rot, dtype=np.float64)
    if joint_data.shape[1] == 0:
        raise ValueError(
            "rotate_vertical requires at least one joint (joint 0 is "
            "the root whose rotation carries the yaw); got J=0")

    # 6D: rotate the two column vectors of the root rotation matrix directly.
    if representation == "6d":
        new_data = joint_data.copy()
        col0 = joint_data[:, 0, :3]
        col1 = joint_data[:, 0, 3:]
        new_data[:, 0, :3] = (R_vert @ col0.T).T
        new_data[:, 0, 3:] = (R_vert @ col1.T).T
        return _result(arrays, new_root_pos, new_data, **positions)

    # All other representations: work through quaternion space.
    quats = _to_quats(joint_data, representation, euler_orders)
    q_rot = _build_rotation_quat(signed_angle, up_idx)
    new_quats = quats.copy()
    new_quats[:, 0] = rotations.quat_multiply(q_rot, quats[:, 0])
    return _result(
        arrays, new_root_pos,
        _from_quats(new_quats, representation, euler_orders), **positions)


@handles_streams(*STREAM_NAMES)
def mirror(
    arrays: MotionArrays,
    *,
    lr_joint_pairs: list[tuple[int, int]] | None = None,
    lr_node_pairs: list[tuple[int, int]] | None = None,
    lateral_axis: str,
    representation: str | None = None,
    euler_orders: list[str] | None = None,
) -> MotionArrays:
    """Mirror joint arrays left-right.

    Swaps left and right vertex data, negates the lateral component of
    root translation and of every position vertex, and reflects each
    rotation across the sagittal plane.

    Parameters
    ----------
    arrays : MotionArrays
        Any combination of streams.  ``representation`` is required only
        when ``joint_rot`` is present.
    lr_joint_pairs : list of (int, int)
        ``[(left_idx, right_idx), ...]`` in joint-array space, typically
        :func:`pybvh_ml.get_lr_pairs`.  **The reflection is applied to every
        joint; this list only controls which ones additionally swap slots.**
        That is the right behavior for midline joints (hips, spine, neck,
        head), which have no partner and must reflect in place.  It is the
        wrong behavior for a lateral joint whose partner is missing from the
        list: it reflects in place instead of moving to its partner's slot,
        producing a pose where that limb mirrors while the rest of the body
        does not — silently, with no shape or value error.  Completeness of
        this list is the caller's responsibility; an empty list reflects
        every joint in place without any swap, which is correct only for a
        skeleton that is entirely midline.

        Required when the sample carries a joint-space stream
        (``joint_rot`` or ``joint_pos``); ``None`` is refused there
        rather than treated as "no pairs", since the two produce
        different motion and only one of them is ever intended.
    lr_node_pairs : list of (int, int), optional
        The same list in **node** index space — joints *and* their end
        sites — typically ``skeleton_info["node_lr_pairs"]`` (from
        :attr:`pybvh.Bvh.node_lr_pairs`).  Required when the sample
        carries ``node_pos``, and not interchangeable with
        ``lr_joint_pairs``: node indices diverge from joint indices as
        soon as any end site precedes a paired joint in file order, so
        the wrong list silently swaps the wrong vertices.
    lateral_axis : str
        Signed axis string: ``'+x'``, ``'-x'``, ``'+y'``, ``'-y'``,
        ``'+z'``, or ``'-z'``.  The sign is accepted for API symmetry
        with :func:`rotate_vertical` but does not affect the result
        (mirror is sign-invariant).
    representation : str, optional
        One of ``"quat"``, ``"6d"``, ``"axisangle"``,
        ``"rotmat"``, ``"euler"``.  Required when ``arrays`` carries
        ``joint_rot``; ignored for a positions-only sample.
    euler_orders : list of str, optional
        Required when ``representation="euler"``, ignored otherwise.

    Returns
    -------
    MotionArrays

    Notes
    -----
    Mirroring is done in **parent-local rotation space** — reflect each
    rotation, swap the L/R slots — matching :func:`pybvh.transforms.mirror`,
    so mirroring arrays and mirroring the source :class:`~pybvh.Bvh` give the
    same motion.  The alternative is mirroring in world space: run FK, reflect
    the resulting joint positions, and re-solve for local rotations.  The two
    agree exactly when the rest pose is laterally symmetric (every left
    joint's offset is the mirror of its right partner's, and midline offsets
    have no lateral component), which holds for most retargeted rigs.  They
    diverge on rigs with asymmetric offsets: the local-space result is still a
    valid pose, but not the exact reflection of the input, with the error
    accumulating down the chain from the first asymmetric bone.  Neither
    pybvh nor pybvh-ml implements the world-space variant.

    **The position streams take the other route**, because they have no
    choice: a position is a world coordinate, so reflecting it *is* the
    world-space mirror — negate the lateral component, swap the paired
    vertices.  Each stream therefore stays individually correct while the
    pair stops being exact FK partners on an asymmetric rig.  Recompute
    the positions from the mirrored rotations (or run
    :func:`add_joint_rotation_noise`, which re-derives them) if you need
    them to agree.

    ``lateral_axis`` must also be given explicitly here, whereas
    :meth:`pybvh.Bvh.mirror` auto-detects it by averaging left-minus-right
    rest-pose offsets — again, an array-level function has no rest pose to
    measure.  The default ``lateral_axis='+x'`` on
    :meth:`~pybvh_ml.AugmentationPipeline.standard` is a convention, not a
    measurement: it is correct for the usual Y-up / Z-forward rig and wrong
    for a rig whose lateral axis is Z.  When in doubt, mirror one clip on the
    :class:`~pybvh.Bvh` (which measures the axis) and compare against the
    array path before trusting the assumed axis across a dataset.
    """
    _check_stream_support(mirror, arrays)
    new_root_pos = np.array(arrays.root_pos, dtype=np.float64)
    positions = _working_positions(arrays)

    lateral_idx = parse_axis(lateral_axis).index
    new_root_pos[:, lateral_idx] *= -1.0

    pair_lists = {"joint_pos": lr_joint_pairs, "node_pos": lr_node_pairs}
    if arrays.joint_rot is not None:
        _require_pairs(lr_joint_pairs, "lr_joint_pairs", "joint_rot")
    for name, value in positions.items():
        if value is None:
            continue
        pairs = pair_lists[name]
        _require_pairs(
            pairs,
            "lr_joint_pairs" if name == "joint_pos" else "lr_node_pairs",
            name)
        value[:, :, lateral_idx] *= -1.0
        _swap_lr_pairs(value, pairs)

    if arrays.joint_rot is None:
        return _result(arrays, new_root_pos, None, **positions)

    representation = _require_representation(arrays, representation, "mirror")
    new_data = np.array(arrays.joint_rot, dtype=np.float64)

    # 6D and quaternion: swap raw joint data, then apply the analytic
    # sign mask (both layouts are per-joint-uniform, so swap order is
    # irrelevant).
    if representation == "6d":
        _swap_lr_pairs(new_data, lr_joint_pairs)
        new_data *= _mirror_sign_rot6d(lateral_idx)
        return _result(arrays, new_root_pos, new_data, **positions)

    if representation == "quat":
        _swap_lr_pairs(new_data, lr_joint_pairs)
        new_data *= _mirror_sign_quat(lateral_idx)
        return _result(arrays, new_root_pos, new_data, **positions)

    # All other representations: convert to quaternions first, swap in
    # quat space, mask, convert back.  Converting before the swap keeps
    # each joint's raw data interpreted under its own euler_orders
    # entry — swapping raw euler triples would decode a left joint's
    # angles with the right joint's order whenever an L/R pair mixes
    # Euler orders.  Converting back with the destination joint's order
    # is correct: consumers read column j under euler_orders[j].
    quats = _to_quats(new_data, representation, euler_orders)
    _swap_lr_pairs(quats, lr_joint_pairs)
    quats *= _mirror_sign_quat(lateral_idx)
    return _result(
        arrays, new_root_pos,
        _from_quats(quats, representation, euler_orders), **positions)


@handles_streams(*STREAM_NAMES)
def add_joint_rotation_noise(
    arrays: MotionArrays,
    *,
    sigma: float,
    representation: str | None = None,
    degrees: bool = False,
    rng: np.random.Generator | None = None,
    euler_orders: list[str] | None = None,
    fk_topology: object | None = None,
    world_up: str | None = None,
) -> MotionArrays:
    """Add Gaussian rotation noise to every joint.

    For each joint at each frame, generates a small random rotation
    (axis uniformly random on the unit sphere, angle sampled from
    ``N(0, sigma)``) and composes it with the original rotation:
    ``q_noisy = q_noise * q_original``.

    When the sample also carries positions, they are **re-derived by
    forward kinematics** from the noised rotations rather than
    transformed — the one step that handles a stream by re-derivation
    (see :func:`handles_streams`).  That is what keeps the two streams
    FK partners, and it is only possible in this direction: rotation →
    position is FK, position → rotation would be IK.

    Parameters
    ----------
    arrays : MotionArrays
        Must carry ``joint_rot``.  ``root_pos`` passes through
        unchanged — jittering the root trajectory is
        :func:`add_root_position_noise`, a separate function because its
        sigma is a length rather than an angle.
    sigma : float
        Standard deviation of the rotation noise, in radians (or degrees
        when ``degrees=True``).
    representation : str, optional
        One of ``"quat"``, ``"6d"``, ``"axisangle"``,
        ``"rotmat"``, ``"euler"``.  Required here, since this step
        cannot run on a sample without rotations at all.
    degrees : bool, optional
        Interpret ``sigma`` in degrees.  Default False (radians).
    rng : numpy Generator, optional
        ``sigma=0`` is a no-op in value but **still draws** from the
        generator, so a seeded pipeline's draw sequence does not depend
        on the sigma it was configured with.  (:func:`pybvh.transforms.add_rotation_noise`
        short-circuits instead; ours cannot without changing the stream
        of every already-seeded pipeline.)
    euler_orders : list of str, optional
        Required when ``representation="euler"``, ignored otherwise.
    fk_topology : pybvh.FkTopology, optional
        The skeleton as plain arrays, for the position refresh.
        **Required when the sample carries a position stream** and
        ignored otherwise.  Rebuild it once per dataset from
        ``skeleton_info["fk_topology"]`` with
        :func:`~pybvh_ml.build_fk_topology`;
        :meth:`~pybvh_ml.AugmentationPipeline.standard` wires it for you.
    world_up : str, optional
        Signed world-up axis (``bvh.world_up``, persisted as
        ``skeleton_info["world_up"]``).  Read only under
        ``position_centering="first"``, which needs to know which
        coordinate ground-plane centering leaves alone.  pybvh 0.8.2
        removed the ``'+y'`` default precisely because it silently
        mis-centered non-y-up skeletons, and an ``FkTopology`` carries no
        gravity direction, so there is nothing to fall back to.

    Returns
    -------
    MotionArrays

    Notes
    -----
    **Cost of the refresh.** Roughly 0.9 ms per sample at ``F=64`` on a
    31-joint rig (0.64 ms FK plus 0.25 ms 6d→euler), ~1.7 ms at
    ``F=128`` — about a tripling of per-sample augmentation cost when it
    fires.  Dataloader workers absorb it, but it is not free, so it runs
    only when a position stream is actually present.

    The noise model is **isotropic in rotation space**: one random axis
    per joint per frame, with the magnitude drawn from ``N(0, sigma)``.
    The alternative — and what :func:`pybvh.transforms.add_rotation_noise`
    does — is independent Gaussian noise on each Euler channel, which is
    cheaper but anisotropic (its effective magnitude depends on the
    channel order and on how near the pose is to gimbal lock), and is not
    even well defined for a ``6d`` or ``quat`` array.  This function is
    representation-agnostic, so it takes the isotropic route.  The two
    agree in distribution only in the small-angle limit for a joint whose
    rotation is near identity.

    See Also
    --------
    add_root_position_noise : The root-translation counterpart.
    add_joint_position_noise : Keypoint jitter, for a sample with no
        rotations to noise.
    """
    _check_stream_support(add_joint_rotation_noise, arrays)
    _rotation_noise_preconditions(arrays, fk_topology, world_up)
    _validate_sigma(sigma)
    if rng is None:
        rng = np.random.default_rng()

    joint_data = np.asarray(
        require_joint_rot(arrays, "add_joint_rotation_noise"),
        dtype=np.float64)
    representation = _require_representation(
        arrays, representation, "add_joint_rotation_noise")
    sigma_rad = _as_radians(sigma, degrees)

    quats = _to_quats(joint_data, representation, euler_orders)
    F, J, _ = quats.shape

    axis = rng.standard_normal((F, J, 3))
    norm = np.linalg.norm(axis, axis=-1, keepdims=True)
    norm = np.where(norm < 1e-15, 1.0, norm)
    axis = axis / norm

    half_angle = rng.normal(0, sigma_rad, (F, J)) / 2.0
    q_noise = np.empty((F, J, 4), dtype=np.float64)
    q_noise[..., 0] = np.cos(half_angle)
    q_noise[..., 1:] = np.sin(half_angle)[..., np.newaxis] * axis

    noisy_quats = rotations.quat_multiply(q_noise, quats)
    norms = np.linalg.norm(noisy_quats, axis=-1, keepdims=True)
    # q_noise is unit by construction, so a zero norm here means a
    # zero-norm *input* quaternion — not a rotation; match pybvh's
    # quat_to_rotmat contract and fail loudly instead of emitting NaN.
    if np.any(norms == 0.0):
        raise ValueError(
            "joint_rot contains a zero-norm quaternion; the zero "
            "quaternion does not represent a rotation")
    noisy_quats /= norms

    root_pos = np.asarray(arrays.root_pos, dtype=np.float64)
    # The quaternions, not the converted-back result: FK wants Euler
    # angles, and quat → euler is the shorter of the two hops.
    positions = _refresh_positions_by_fk(
        joint_rot=noisy_quats, root_pos=root_pos, representation="quat",
        fk_topology=fk_topology, world_up=world_up,
        position_centering=arrays.position_centering,
        want_joint_pos=arrays.joint_pos is not None,
        want_node_pos=arrays.node_pos is not None)
    return _result(
        arrays, root_pos.copy(),
        _from_quats(noisy_quats, representation, euler_orders), **positions)


@handles_streams(*STREAM_NAMES)
def add_root_position_noise(
    arrays: MotionArrays,
    *,
    sigma: float,
    rng: np.random.Generator | None = None,
) -> MotionArrays:
    """Add Gaussian noise to the root translation.

    Split from :func:`add_joint_rotation_noise` because the two sigmas
    are in different units — radians there, the data's length unit here —
    so a single ``degrees=`` flag could only ever have applied to one of
    them.  pybvh made the same split for the same reason in 0.8.1
    (``add_noise`` → ``add_rotation_noise`` + ``add_position_noise``).

    To reproduce a combined call, chain them with the **same** generator,
    rotation first::

        arrays = add_joint_rotation_noise(arrays, sigma=s, rng=rng,
                                          representation="6d")
        arrays = add_root_position_noise(arrays, sigma=p, rng=rng)

    Parameters
    ----------
    arrays : MotionArrays
        ``joint_rot`` passes through untouched, and may be ``None``.
        Position streams move with the root or not, depending on
        ``position_centering`` — see the Notes.
    sigma : float
        Standard deviation of the noise added to ``root_pos``, in the
        data's positional units.  There is no ``degrees=`` here because
        this is a length, not an angle.
    rng : numpy Generator, optional
        ``sigma=0`` is a no-op and draws nothing, matching
        :func:`pybvh.transforms.add_position_noise` and preserving what
        the fused ``add_joint_noise(sigma_pos=0)`` did.

    Returns
    -------
    MotionArrays

    Raises
    ------
    ValueError
        If the sample carries positions and ``position_centering`` is
        ``None``.  This is one of the three surfaces whose correctness
        depends on the frame, and guessing is worse than refusing.

    Notes
    -----
    **The step is centering-aware, which makes it exactly correct in
    every mode** rather than merely allowed:

    - ``"world"`` / ``"first"`` — a translation of the root translates
      every joint by the same amount, so the identical per-frame offset
      is added to every position vertex.  One broadcast, exact.
    - ``"skeleton"`` — positions are root-relative and genuinely do not
      move.  Exact by construction.

    Declining position streams outright would have been the conservative
    alternative, and it was rejected because it makes root noise
    unusable on skeleton-centered data, where it is trivially correct.
    The failure it prevents is quiet: under ``"world"`` centering,
    jittering ``root_pos`` while leaving the positions alone leaves the
    two streams mutually inconsistent — and under the canonical ST-GCN
    pack ``streams=("joint_pos",)``, where ``root_pos`` is not packed at
    all, it degrades further into an augmentation the model never sees.

    See Also
    --------
    add_joint_rotation_noise : The joint-rotation counterpart.
    add_joint_position_noise : Per-vertex keypoint jitter, which is a
        different augmentation — independent noise per joint rather than
        one offset shared by the whole body.
    """
    _check_stream_support(add_root_position_noise, arrays)
    _validate_sigma(sigma)
    if rng is None:
        rng = np.random.default_rng()

    root_pos = np.asarray(arrays.root_pos, dtype=np.float64)
    carries_positions = (arrays.joint_pos is not None
                         or arrays.node_pos is not None)
    if carries_positions:
        centering = require_position_centering(
            arrays, "add_root_position_noise")

    if sigma <= 0:
        return _result(arrays, root_pos.copy(), arrays.joint_rot,
                       arrays.joint_pos, arrays.node_pos)

    noise = rng.normal(0, sigma, root_pos.shape)
    if not carries_positions or centering == "skeleton":
        return _result(arrays, root_pos + noise, arrays.joint_rot,
                       arrays.joint_pos, arrays.node_pos)

    vertex_noise = noise[:, np.newaxis, :]
    return _result(
        arrays, root_pos + noise, arrays.joint_rot,
        joint_pos=(None if arrays.joint_pos is None
                   else np.asarray(arrays.joint_pos, dtype=np.float64)
                   + vertex_noise),
        node_pos=(None if arrays.node_pos is None
                  else np.asarray(arrays.node_pos, dtype=np.float64)
                  + vertex_noise))


def _add_position_noise(
    arrays: MotionArrays,
    stream: str,
    sigma: float,
    rng: np.random.Generator | None,
    caller: Callable,
) -> MotionArrays:
    """Per-vertex Gaussian jitter on one position stream.

    Shared by :func:`add_joint_position_noise` and
    :func:`add_node_position_noise`, which are the same math on
    different fields.  They stay two functions rather than one with a
    ``streams=`` kwarg because the index space is then visible at the
    call site — the whole point of pybvh's ``joint_`` / ``node_``
    vocabulary — and because the two declare different stream support.
    """
    _check_stream_support(caller, arrays)
    _validate_sigma(sigma)
    if rng is None:
        rng = np.random.default_rng()

    positions = getattr(arrays, stream)
    if positions is None:
        raise ValueError(
            f"{_step_label(caller)} needs {stream}, but this MotionArrays "
            f"carries none ({stream} is None)")

    positions = np.asarray(positions, dtype=np.float64)
    if sigma > 0:
        positions = positions + rng.normal(0, sigma, positions.shape)
    else:
        positions = positions.copy()
    return _result(arrays, np.asarray(arrays.root_pos, dtype=np.float64),
                   arrays.joint_rot, **{stream: positions})


@handles_streams("root_pos", "joint_pos")
def add_joint_position_noise(
    arrays: MotionArrays,
    *,
    sigma: float,
    rng: np.random.Generator | None = None,
) -> MotionArrays:
    """Add independent Gaussian jitter to every joint position.

    The keypoint jitter of the skeleton-action-recognition world: each
    vertex at each frame moves by its own draw, which is what models a
    pose estimator's per-joint error.  Distinct from
    :func:`add_root_position_noise`, where one offset per frame moves the
    whole body rigidly.

    Declines ``joint_rot``-carrying samples, and the refusal is the
    governing asymmetry of this whole surface: positions are derived
    from rotations, so rotation → position is computable (forward
    kinematics) while position → rotation is not (inverse kinematics).
    A jittered position stream cannot be pushed back into the rotations
    beside it, so a sample carrying both is refused rather than left
    incoherent — jitter the rotations with
    :func:`add_joint_rotation_noise` instead, which re-derives the
    positions.

    That refusal also closes the one composition that would be
    destructive: keypoint jitter can never be silently wiped by a later
    FK refresh, because the two steps can never share a pipeline.

    Parameters
    ----------
    arrays : MotionArrays
        Must carry ``joint_pos`` and must not carry ``joint_rot`` or
        ``node_pos``.
    sigma : float
        Standard deviation of the per-vertex noise, in the data's
        positional units.  ``0`` is a no-op and draws nothing, matching
        :func:`add_root_position_noise`.
    rng : numpy Generator, optional

    Returns
    -------
    MotionArrays

    Notes
    -----
    ``position_centering`` is not read: independent per-vertex noise is
    correct in every frame, since the frames differ by a shift that
    commutes with adding noise.

    See Also
    --------
    add_node_position_noise : The node-space counterpart.
    """
    return _add_position_noise(
        arrays, "joint_pos", sigma, rng, add_joint_position_noise)


@handles_streams("root_pos", "node_pos")
def add_node_position_noise(
    arrays: MotionArrays,
    *,
    sigma: float,
    rng: np.random.Generator | None = None,
) -> MotionArrays:
    """Add independent Gaussian jitter to every node position.

    The node-space counterpart of :func:`add_joint_position_noise` —
    same math, applied to the stream that includes end sites
    (fingertips, toe tips, head top).  See that function for why the two
    are separate names rather than one ``streams=`` kwarg, and for why
    both decline ``joint_rot``.

    Parameters
    ----------
    arrays : MotionArrays
        Must carry ``node_pos`` and must not carry ``joint_rot`` or
        ``joint_pos``.
    sigma : float
        Standard deviation of the per-vertex noise, in the data's
        positional units.
    rng : numpy Generator, optional

    Returns
    -------
    MotionArrays

    See Also
    --------
    add_joint_position_noise : The joint-space counterpart.
    """
    return _add_position_noise(
        arrays, "node_pos", sigma, rng, add_node_position_noise)


@handles_streams(*STREAM_NAMES)
def speed_perturbation_arrays(
    arrays: MotionArrays,
    *,
    factor: float,
    representation: str | None = None,
    euler_orders: list[str] | None = None,
) -> MotionArrays:
    """Speed perturbation via time resampling.

    Uses SLERP for rotation interpolation (via quaternion space) and
    linear interpolation for root position and every position vertex.

    Parameters
    ----------
    arrays : MotionArrays
        Any combination of streams; every present one is resampled onto
        the same time stencil, so they cannot come out of step.
    factor : float
        Speed factor.  ``> 1`` = faster (fewer frames),
        ``< 1`` = slower (more frames).
    representation : str, optional
        One of ``"quat"``, ``"6d"``, ``"axisangle"``,
        ``"rotmat"``, ``"euler"``.  Required when ``arrays`` carries
        ``joint_rot``; ignored for a positions-only sample.
    euler_orders : list of str, optional
        Required when ``representation="euler"``, ignored otherwise.

    Returns
    -------
    MotionArrays
        Arrays of ``F'`` frames.
        For ``F >= 2``, ``F' = max(2, round(F / factor))`` (Python
        banker's rounding — ``round(2.5) == 2``).  Inputs with fewer
        than 2 frames have nothing to interpolate between and are
        returned as unchanged copies.

    Notes
    -----
    Rotations are interpolated with ``pybvh.rotations.quat_slerp``
    under its ``shortest=True`` default: the interpolant takes the
    short arc between adjacent frames, so a turn is never read as its
    >180° complement.  The alternative, ``shortest=False``, preserves a
    genuine wind-up or spin that exceeds half a turn; it is not exposed
    here because a per-frame stencil cannot tell one from the other.

    A consequence for ``representation="quat"`` specifically: output
    quaternions may come back as ``-q`` relative to the corresponding
    input, since ``q`` and ``-q`` are the same rotation and the short
    arc picks whichever hemisphere is nearer.  Every other
    representation is unaffected (the sign is not observable in them).
    Compare rotations, not raw components, when diffing output against
    input — or run the input through ``pybvh.rotations.quat_unwrap``
    first, which makes a sequence hemisphere-continuous.

    **Positions are interpolated linearly, rotations slerped** — chord
    versus arc.  The two agree at the knots and drift between them, so
    resampled positions are not the exact forward kinematics of the
    resampled rotations beside them.  This is intrinsic (there is no
    interpolant that is simultaneously the geodesic in rotation space
    and linear in position space), it is what every pipeline in the
    field does, and ``standardize_length(method="resample_linear")``
    documents the same split from the other direction.
    """
    _check_stream_support(speed_perturbation_arrays, arrays)
    if factor <= 0:
        raise ValueError(f"factor must be > 0, got {factor}")

    root_pos = np.asarray(arrays.root_pos, dtype=np.float64)
    positions = _working_positions(arrays)
    joint_data = (None if arrays.joint_rot is None
                  else np.asarray(arrays.joint_rot, dtype=np.float64))

    F = root_pos.shape[0]
    if F < 2:
        return _result(
            arrays, root_pos.copy(),
            None if joint_data is None else joint_data.copy(), **positions)

    F_new = max(2, round(F / factor))
    t_orig = np.linspace(0.0, 1.0, F)
    t_new = np.linspace(0.0, 1.0, F_new)

    new_root_pos = np.empty((F_new, 3), dtype=np.float64)
    for ax in range(3):
        new_root_pos[:, ax] = np.interp(t_new, t_orig, root_pos[:, ax])

    idx_right = np.searchsorted(t_orig, t_new, side='right')
    idx_right = np.clip(idx_right, 1, F - 1)
    idx_left = idx_right - 1

    t_left = t_orig[idx_left]
    t_right = t_orig[idx_right]
    # Adjacent samples of linspace(0, 1, F) with F >= 2, so dt is
    # exactly 1/(F-1) — never the degenerate interval a zero-guard here
    # would be protecting against.
    dt = t_right - t_left
    alpha = (t_new - t_left) / dt

    # Same stencil as the root, vectorized over the vertex axis.
    weight = alpha[:, np.newaxis, np.newaxis]
    for name, value in positions.items():
        if value is not None:
            positions[name] = ((1.0 - weight) * value[idx_left]
                               + weight * value[idx_right])

    if joint_data is None:
        return _result(arrays, new_root_pos, None, **positions)

    representation = _require_representation(
        arrays, representation, "speed_perturbation_arrays")
    quats = _to_quats(joint_data, representation, euler_orders)
    J = quats.shape[1]
    alpha_jt = np.broadcast_to(alpha[:, np.newaxis], (F_new, J))
    new_quats = rotations.quat_slerp(
        quats[idx_left], quats[idx_right], alpha_jt)

    return _result(
        arrays, new_root_pos,
        _from_quats(new_quats, representation, euler_orders), **positions)


@handles_streams(*STREAM_NAMES)
def dropout_arrays(
    arrays: MotionArrays,
    *,
    drop_rate: float,
    representation: str | None = None,
    rng: np.random.Generator | None = None,
    euler_orders: list[str] | None = None,
) -> MotionArrays:
    """Frame dropout with SLERP interpolation.

    Randomly drops frames and fills the gaps with SLERP-interpolated
    rotations (via quaternion space) and linearly interpolated root and
    vertex positions.  First and last frames are always kept.  Shape is
    unchanged — you get the same ``F`` frames, some replaced by
    interpolated values.

    Parameters
    ----------
    arrays : MotionArrays
        Any combination of streams; one keep-mask governs all of them.
    drop_rate : float
        Fraction of frames to drop, in ``[0, 1)``.
    representation : str, optional
        One of ``"quat"``, ``"6d"``, ``"axisangle"``,
        ``"rotmat"``, ``"euler"``.  Required when ``arrays`` carries
        ``joint_rot``; ignored for a positions-only sample.
    rng : numpy Generator, optional
    euler_orders : list of str, optional
        Required when ``representation="euler"``, ignored otherwise.

    Returns
    -------
    MotionArrays

    Notes
    -----
    Frames 0 and ``F-1`` are always kept, so every dropped frame has
    real neighbours on both sides to interpolate between.  Kept frames
    pass through bit-identically; only the dropped ones are rebuilt.

    Dropped frames are rebuilt with ``pybvh.rotations.quat_slerp``
    under its ``shortest=True`` default, so for
    ``representation="quat"`` a rebuilt frame may come back as ``-q``
    relative to the original — the same rotation in the nearer
    hemisphere.  See :func:`speed_perturbation_arrays` for the full
    note; ``pybvh.rotations.quat_unwrap`` makes a sequence
    hemisphere-continuous if you need to compare components directly.

    Rebuilt position frames are interpolated **linearly** between the
    kept neighbours while rotations are slerped — see
    :func:`speed_perturbation_arrays` for that divergence.
    """
    _check_stream_support(dropout_arrays, arrays)
    _validate_drop_rate(drop_rate)
    if rng is None:
        rng = np.random.default_rng()

    root_pos = np.asarray(arrays.root_pos, dtype=np.float64)
    positions = _working_positions(arrays)
    joint_data = (None if arrays.joint_rot is None
                  else np.asarray(arrays.joint_rot, dtype=np.float64))

    F = root_pos.shape[0]
    unchanged = _result(
        arrays, root_pos.copy(),
        None if joint_data is None else joint_data.copy(), **positions)
    if F < 2 or drop_rate == 0:
        return unchanged

    keep_mask = rng.random(F) >= drop_rate
    keep_mask[0] = True
    keep_mask[-1] = True
    kept_indices = np.where(keep_mask)[0]

    dropped = np.where(~keep_mask)[0]
    if len(dropped) == 0:
        return unchanged

    ins = np.searchsorted(kept_indices, dropped, side='right')
    left_idx = kept_indices[np.clip(ins - 1, 0, len(kept_indices) - 1)]
    right_idx = kept_indices[np.clip(ins, 0, len(kept_indices) - 1)]

    # left_idx < dropped < right_idx by construction (frames 0 and F-1
    # are always kept), so dt >= 1 always.
    dt = (right_idx - left_idx).astype(np.float64)
    alpha = (dropped - left_idx).astype(np.float64) / dt

    new_root_pos = root_pos.copy()
    for ax in range(3):
        new_root_pos[dropped, ax] = (
            (1.0 - alpha) * root_pos[left_idx, ax]
            + alpha * root_pos[right_idx, ax])

    weight = alpha[:, np.newaxis, np.newaxis]
    for value in positions.values():
        if value is not None:
            value[dropped] = ((1.0 - weight) * value[left_idx]
                              + weight * value[right_idx])

    if joint_data is None:
        return _result(arrays, new_root_pos, None, **positions)

    representation = _require_representation(
        arrays, representation, "dropout_arrays")
    quats = _to_quats(joint_data, representation, euler_orders)
    J = quats.shape[1]

    q_left = quats[left_idx]
    q_right = quats[right_idx]
    alpha_jt = np.broadcast_to(alpha[:, np.newaxis], (len(dropped), J))

    new_quats = quats.copy()
    new_quats[dropped] = rotations.quat_slerp(q_left, q_right, alpha_jt)

    return _result(
        arrays, new_root_pos,
        _from_quats(new_quats, representation, euler_orders), **positions)


# =========================================================================
# Pipeline-entry precondition dispatch
# =========================================================================

def _rotation_noise_step_check(arrays: MotionArrays, kwargs: dict) -> None:
    """Adapt :func:`_rotation_noise_preconditions` to a step's kwargs dict."""
    _rotation_noise_preconditions(
        arrays, kwargs.get("fk_topology"), kwargs.get("world_up"))


_STEP_PRECONDITIONS = {
    add_joint_rotation_noise: _rotation_noise_step_check,
}
"""Per-step checks that need the configured kwargs, not just the streams.

Kept beside :data:`~pybvh_ml._staged.STAGED_DISPATCH` in spirit: one
table naming the built-ins that need something extra, rather than a
special case buried in the pipeline.
"""


def _check_step_preconditions(
    fn: Callable,
    arrays: MotionArrays,
    kwargs: dict,
) -> None:
    """Everything about a step that is checkable before it runs.

    :class:`~pybvh_ml.AugmentationPipeline` calls this for every
    configured step at ``__call__`` entry, before any of them fires, so
    none of these raises can be stochastic under ``p < 1``.  The entry
    point already holds everything each check needs — the step's
    identity, its configured kwargs, and the sample's streams — which is
    exactly why none of them belongs in a step body.
    """
    _check_stream_support(fn, arrays)
    check = _STEP_PRECONDITIONS.get(_unwrap_step(fn))
    if check is not None:
        check(arrays, kwargs)
