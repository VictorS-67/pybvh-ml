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
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pybvh import parse_axis, rotations

from .arrays import MotionArrays, require_joint_rot


# =========================================================================
# Private helpers
# =========================================================================

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
    """Swap L/R joint data in-place along axis 1."""
    for lj, rj in lr_joint_pairs:
        data[:, lj], data[:, rj] = (
            data[:, rj].copy(), data[:, lj].copy())


# =========================================================================
# Public augmentation functions
# =========================================================================

def rotate_vertical(
    arrays: MotionArrays,
    *,
    angle: float,
    up_axis: str,
    representation: str,
    degrees: bool = False,
    euler_orders: list[str] | None = None,
) -> MotionArrays:
    """Rotate joint arrays around the vertical axis.

    Only the root joint (index 0) and root position are modified;
    non-root joints are in parent-local space and stay unchanged.

    Parameters
    ----------
    arrays : MotionArrays
        Must carry ``joint_rot``.
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
    representation : str
        One of ``"quat"``, ``"6d"``, ``"axisangle"``,
        ``"rotmat"``, ``"euler"``.
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
    """
    joint_data = np.array(
        require_joint_rot(arrays, "rotate_vertical"), dtype=np.float64)
    root_pos = np.array(arrays.root_pos, dtype=np.float64)
    if joint_data.shape[1] == 0:
        raise ValueError(
            "rotate_vertical requires at least one joint (joint 0 is "
            "the root whose rotation carries the yaw); got J=0")

    up = parse_axis(up_axis)
    up_idx, up_sign = up.index, up.sign
    signed_angle = _as_radians(angle, degrees) * up_sign
    R_vert = _build_rotation_matrix(signed_angle, up_idx)
    new_root_pos = (R_vert @ root_pos.T).T

    # 6D: rotate the two column vectors of the root rotation matrix directly.
    if representation == "6d":
        new_data = joint_data.copy()
        col0 = joint_data[:, 0, :3]
        col1 = joint_data[:, 0, 3:]
        new_data[:, 0, :3] = (R_vert @ col0.T).T
        new_data[:, 0, 3:] = (R_vert @ col1.T).T
        return MotionArrays(root_pos=new_root_pos, joint_rot=new_data)

    # All other representations: work through quaternion space.
    quats = _to_quats(joint_data, representation, euler_orders)
    q_rot = _build_rotation_quat(signed_angle, up_idx)
    new_quats = quats.copy()
    new_quats[:, 0] = rotations.quat_multiply(q_rot, quats[:, 0])
    return MotionArrays(
        root_pos=new_root_pos,
        joint_rot=_from_quats(new_quats, representation, euler_orders))


def mirror(
    arrays: MotionArrays,
    *,
    lr_joint_pairs: list[tuple[int, int]],
    lateral_axis: str,
    representation: str,
    euler_orders: list[str] | None = None,
) -> MotionArrays:
    """Mirror joint arrays left-right.

    Swaps left and right joint data, negates the lateral component of
    root translation, and reflects each rotation across the sagittal plane.

    Parameters
    ----------
    arrays : MotionArrays
        Must carry ``joint_rot``.
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
    lateral_axis : str
        Signed axis string: ``'+x'``, ``'-x'``, ``'+y'``, ``'-y'``,
        ``'+z'``, or ``'-z'``.  The sign is accepted for API symmetry
        with :func:`rotate_vertical` but does not affect the result
        (mirror is sign-invariant).
    representation : str
        One of ``"quat"``, ``"6d"``, ``"axisangle"``,
        ``"rotmat"``, ``"euler"``.
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
    new_data = np.array(
        require_joint_rot(arrays, "mirror"), dtype=np.float64)
    new_root_pos = np.array(arrays.root_pos, dtype=np.float64)

    lateral_idx = parse_axis(lateral_axis).index
    new_root_pos[:, lateral_idx] *= -1.0

    # 6D and quaternion: swap raw joint data, then apply the analytic
    # sign mask (both layouts are per-joint-uniform, so swap order is
    # irrelevant).
    if representation == "6d":
        _swap_lr_pairs(new_data, lr_joint_pairs)
        new_data *= _mirror_sign_rot6d(lateral_idx)
        return MotionArrays(root_pos=new_root_pos, joint_rot=new_data)

    if representation == "quat":
        _swap_lr_pairs(new_data, lr_joint_pairs)
        new_data *= _mirror_sign_quat(lateral_idx)
        return MotionArrays(root_pos=new_root_pos, joint_rot=new_data)

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
    return MotionArrays(
        root_pos=new_root_pos,
        joint_rot=_from_quats(quats, representation, euler_orders))


def add_joint_rotation_noise(
    arrays: MotionArrays,
    *,
    sigma: float,
    representation: str,
    degrees: bool = False,
    rng: np.random.Generator | None = None,
    euler_orders: list[str] | None = None,
) -> MotionArrays:
    """Add Gaussian rotation noise to every joint.

    For each joint at each frame, generates a small random rotation
    (axis uniformly random on the unit sphere, angle sampled from
    ``N(0, sigma)``) and composes it with the original rotation:
    ``q_noisy = q_noise * q_original``.

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
    representation : str
        One of ``"quat"``, ``"6d"``, ``"axisangle"``,
        ``"rotmat"``, ``"euler"``.
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

    Returns
    -------
    MotionArrays

    Notes
    -----
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
    """
    _validate_sigma(sigma)
    if rng is None:
        rng = np.random.default_rng()

    joint_data = np.asarray(
        require_joint_rot(arrays, "add_joint_rotation_noise"),
        dtype=np.float64)
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

    return MotionArrays(
        root_pos=arrays.root_pos.copy(),
        joint_rot=_from_quats(noisy_quats, representation, euler_orders))


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

    See Also
    --------
    add_joint_rotation_noise : The joint-rotation counterpart.
    """
    _validate_sigma(sigma)
    if rng is None:
        rng = np.random.default_rng()

    root_pos = np.asarray(arrays.root_pos, dtype=np.float64)
    noise = rng.normal(0, sigma, root_pos.shape) if sigma > 0 else 0.0
    return arrays.replace(root_pos=root_pos + noise)


def speed_perturbation_arrays(
    arrays: MotionArrays,
    *,
    factor: float,
    representation: str,
    euler_orders: list[str] | None = None,
) -> MotionArrays:
    """Speed perturbation via time resampling.

    Uses SLERP for rotation interpolation (via quaternion space) and
    linear interpolation for root position.

    Parameters
    ----------
    arrays : MotionArrays
        Must carry ``joint_rot``.
    factor : float
        Speed factor.  ``> 1`` = faster (fewer frames),
        ``< 1`` = slower (more frames).
    representation : str
        One of ``"quat"``, ``"6d"``, ``"axisangle"``,
        ``"rotmat"``, ``"euler"``.
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
    """
    if factor <= 0:
        raise ValueError(f"factor must be > 0, got {factor}")

    joint_data = np.asarray(
        require_joint_rot(arrays, "speed_perturbation_arrays"),
        dtype=np.float64)
    root_pos = np.asarray(arrays.root_pos, dtype=np.float64)

    F = root_pos.shape[0]
    if F < 2:
        return MotionArrays(root_pos=root_pos.copy(),
                            joint_rot=joint_data.copy())

    F_new = max(2, round(F / factor))
    t_orig = np.linspace(0.0, 1.0, F)
    t_new = np.linspace(0.0, 1.0, F_new)

    new_root_pos = np.empty((F_new, 3), dtype=np.float64)
    for ax in range(3):
        new_root_pos[:, ax] = np.interp(t_new, t_orig, root_pos[:, ax])

    quats = _to_quats(joint_data, representation, euler_orders)
    J = quats.shape[1]

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

    q_left = quats[idx_left]
    q_right = quats[idx_right]
    alpha_jt = np.broadcast_to(alpha[:, np.newaxis], (F_new, J))
    new_quats = rotations.quat_slerp(q_left, q_right, alpha_jt)

    return MotionArrays(
        root_pos=new_root_pos,
        joint_rot=_from_quats(new_quats, representation, euler_orders))


def dropout_arrays(
    arrays: MotionArrays,
    *,
    drop_rate: float,
    representation: str,
    rng: np.random.Generator | None = None,
    euler_orders: list[str] | None = None,
) -> MotionArrays:
    """Frame dropout with SLERP interpolation.

    Randomly drops frames and fills the gaps with SLERP-interpolated
    rotations (via quaternion space) and linearly interpolated root
    positions.  First and last frames are always kept.  Shape is
    unchanged — you get the same ``F`` frames, some replaced by
    interpolated values.

    Parameters
    ----------
    arrays : MotionArrays
        Must carry ``joint_rot``.
    drop_rate : float
        Fraction of frames to drop, in ``[0, 1)``.
    representation : str
        One of ``"quat"``, ``"6d"``, ``"axisangle"``,
        ``"rotmat"``, ``"euler"``.
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
    """
    _validate_drop_rate(drop_rate)
    if rng is None:
        rng = np.random.default_rng()

    joint_data = np.asarray(
        require_joint_rot(arrays, "dropout_arrays"), dtype=np.float64)
    root_pos = np.asarray(arrays.root_pos, dtype=np.float64)

    F = root_pos.shape[0]
    unchanged = MotionArrays(root_pos=root_pos.copy(),
                             joint_rot=joint_data.copy())
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

    quats = _to_quats(joint_data, representation, euler_orders)
    J = quats.shape[1]

    q_left = quats[left_idx]
    q_right = quats[right_idx]
    alpha_jt = np.broadcast_to(alpha[:, np.newaxis], (len(dropped), J))

    new_quats = quats.copy()
    new_quats[dropped] = rotations.quat_slerp(q_left, q_right, alpha_jt)

    return MotionArrays(
        root_pos=new_root_pos,
        joint_rot=_from_quats(new_quats, representation, euler_orders))
