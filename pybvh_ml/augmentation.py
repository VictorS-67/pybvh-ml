"""Array-level augmentation for ML pipelines.

Operates on pre-extracted NumPy arrays without Bvh objects.
All functions accept any rotation representation supported by pybvh:
``"quat"``, ``"6d"``, ``"axisangle"``, ``"rotmat"``, or ``"euler"``.
Euler arrays additionally require an ``euler_orders`` kwarg.

All functions take and return ``(root_pos, joint_data)`` — root position
first, matching pybvh's ``Bvh.from_*`` / ``Bvh.to_*`` convention.  All
parameters are keyword-only: since ``root_pos`` and ``joint_data`` are
shape-compatible ndarrays, accepting them positionally would make a
swapped call silently corrupt data.  Call with
``rotate_vertical(root_pos=..., joint_data=..., angle=..., ...)``.

Angles are in **radians** throughout, matching pybvh's convention.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pybvh import parse_axis, rotations


# =========================================================================
# Private helpers
# =========================================================================

def _validate_noise_sigmas(sigma: float, sigma_pos: float) -> None:
    """Reject negative noise standard deviations."""
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")
    if sigma_pos < 0:
        raise ValueError(f"sigma_pos must be >= 0, got {sigma_pos}")


def _validate_drop_rate(drop_rate: float) -> None:
    """Reject drop rates outside the documented ``[0, 1)`` range."""
    if not 0.0 <= drop_rate < 1.0:
        raise ValueError(f"drop_rate must be in [0, 1), got {drop_rate}")


def _validate_frame_counts(
    root_pos: npt.NDArray[np.float64],
    joint_data: npt.NDArray[np.float64],
) -> None:
    """Reject root_pos / joint_data arrays with different frame counts.

    A mismatch means the caller paired arrays from different clips (or
    different slices of one clip) — downstream math would silently
    interpolate or index the wrong frames.
    """
    rp_shape = np.shape(root_pos)
    jd_shape = np.shape(joint_data)
    if rp_shape[0] != jd_shape[0]:
        raise ValueError(
            f"root_pos and joint_data disagree on frame count: "
            f"root_pos has {rp_shape[0]} frames (shape {rp_shape}), "
            f"joint_data has {jd_shape[0]} (shape {jd_shape})")


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
    *,
    root_pos: npt.NDArray[np.float64],
    joint_data: npt.NDArray[np.float64],
    angle: float,
    up_axis: str,
    representation: str,
    euler_orders: list[str] | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Rotate joint arrays around the vertical axis.

    All arguments are keyword-only.  ``root_pos`` and ``joint_data``
    are shape-compatible ndarrays (both have leading dim ``F``); a
    swapped positional call would silently corrupt, so the API refuses
    positional binding and forces explicit names.

    Only the root joint (index 0) and root position are modified;
    non-root joints are in parent-local space and stay unchanged.

    Parameters
    ----------
    root_pos : ndarray, shape (F, 3)
        Root translation per frame.
    joint_data : ndarray, shape (F, J, C)
        Joint rotation data in ``representation`` format.
    angle : float
        Rotation angle in radians.
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
    new_root_pos : ndarray, shape (F, 3)
    new_joint_data : ndarray, shape (F, J, C)

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
    joint_data = np.array(joint_data, dtype=np.float64)
    root_pos = np.array(root_pos, dtype=np.float64)
    _validate_frame_counts(root_pos, joint_data)
    if joint_data.shape[1] == 0:
        raise ValueError(
            "rotate_vertical requires at least one joint (joint 0 is "
            "the root whose rotation carries the yaw); got J=0")

    up = parse_axis(up_axis)
    up_idx, up_sign = up.index, up.sign
    signed_angle = angle * up_sign
    R_vert = _build_rotation_matrix(signed_angle, up_idx)
    new_root_pos = (R_vert @ root_pos.T).T

    # 6D: rotate the two column vectors of the root rotation matrix directly.
    if representation == "6d":
        new_data = joint_data.copy()
        col0 = joint_data[:, 0, :3]
        col1 = joint_data[:, 0, 3:]
        new_data[:, 0, :3] = (R_vert @ col0.T).T
        new_data[:, 0, 3:] = (R_vert @ col1.T).T
        return new_root_pos, new_data

    # All other representations: work through quaternion space.
    quats = _to_quats(joint_data, representation, euler_orders)
    q_rot = _build_rotation_quat(signed_angle, up_idx)
    new_quats = quats.copy()
    new_quats[:, 0] = rotations.quat_multiply(q_rot, quats[:, 0])
    return new_root_pos, _from_quats(new_quats, representation, euler_orders)


def mirror(
    *,
    root_pos: npt.NDArray[np.float64],
    joint_data: npt.NDArray[np.float64],
    lr_joint_pairs: list[tuple[int, int]],
    lateral_axis: str,
    representation: str,
    euler_orders: list[str] | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Mirror joint arrays left-right.

    Swaps left and right joint data, negates the lateral component of
    root translation, and reflects each rotation across the sagittal plane.

    Parameters
    ----------
    root_pos : ndarray, shape (F, 3)
    joint_data : ndarray, shape (F, J, C)
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
    new_root_pos : ndarray, shape (F, 3)
    new_joint_data : ndarray, shape (F, J, C)

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
    new_data = np.array(joint_data, dtype=np.float64)
    new_root_pos = np.array(root_pos, dtype=np.float64)
    _validate_frame_counts(new_root_pos, new_data)

    lateral_idx = parse_axis(lateral_axis).index
    new_root_pos[:, lateral_idx] *= -1.0

    # 6D and quaternion: swap raw joint data, then apply the analytic
    # sign mask (both layouts are per-joint-uniform, so swap order is
    # irrelevant).
    if representation == "6d":
        _swap_lr_pairs(new_data, lr_joint_pairs)
        new_data *= _mirror_sign_rot6d(lateral_idx)
        return new_root_pos, new_data

    if representation == "quat":
        _swap_lr_pairs(new_data, lr_joint_pairs)
        new_data *= _mirror_sign_quat(lateral_idx)
        return new_root_pos, new_data

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
    return new_root_pos, _from_quats(quats, representation, euler_orders)


def add_joint_noise(
    *,
    root_pos: npt.NDArray[np.float64],
    joint_data: npt.NDArray[np.float64],
    sigma: float,
    representation: str,
    sigma_pos: float = 0.0,
    rng: np.random.Generator | None = None,
    euler_orders: list[str] | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Add Gaussian rotation noise to joint arrays.

    For each joint at each frame, generates a small random rotation
    (axis uniformly random on the unit sphere, angle sampled from
    ``N(0, sigma)`` in radians) and composes it with the original
    rotation: ``q_noisy = q_noise * q_original``.

    Optionally adds Gaussian noise to root positions as well.

    Parameters
    ----------
    root_pos : ndarray, shape (F, 3)
    joint_data : ndarray, shape (F, J, C)
    sigma : float
        Standard deviation of rotation noise in radians.
    representation : str
        One of ``"quat"``, ``"6d"``, ``"axisangle"``,
        ``"rotmat"``, ``"euler"``.
    sigma_pos : float
        Standard deviation of root position noise, in the data's
        positional units (default 0 = none).
    rng : numpy Generator, optional
    euler_orders : list of str, optional
        Required when ``representation="euler"``, ignored otherwise.

    Returns
    -------
    new_root_pos : ndarray, shape (F, 3)
    new_joint_data : ndarray, shape (F, J, C)
    """
    _validate_noise_sigmas(sigma, sigma_pos)
    if rng is None:
        rng = np.random.default_rng()

    joint_data = np.asarray(joint_data, dtype=np.float64)
    root_pos = np.asarray(root_pos, dtype=np.float64)
    _validate_frame_counts(root_pos, joint_data)

    quats = _to_quats(joint_data, representation, euler_orders)
    F, J, _ = quats.shape

    axis = rng.standard_normal((F, J, 3))
    norm = np.linalg.norm(axis, axis=-1, keepdims=True)
    norm = np.where(norm < 1e-15, 1.0, norm)
    axis = axis / norm

    half_angle = rng.normal(0, sigma, (F, J)) / 2.0
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
            "joint_data contains a zero-norm quaternion; the zero "
            "quaternion does not represent a rotation")
    noisy_quats /= norms

    new_root_pos = root_pos.copy()
    if sigma_pos > 0:
        new_root_pos = new_root_pos + rng.normal(0, sigma_pos, root_pos.shape)

    return new_root_pos, _from_quats(noisy_quats, representation, euler_orders)


def speed_perturbation_arrays(
    *,
    root_pos: npt.NDArray[np.float64],
    joint_data: npt.NDArray[np.float64],
    factor: float,
    representation: str,
    euler_orders: list[str] | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Speed perturbation via time resampling.

    Uses SLERP for rotation interpolation (via quaternion space) and
    linear interpolation for root position.

    Parameters
    ----------
    root_pos : ndarray, shape (F, 3)
    joint_data : ndarray, shape (F, J, C)
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
    new_root_pos : ndarray, shape (F', 3)
    new_joint_data : ndarray, shape (F', J, C)
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

    joint_data = np.asarray(joint_data, dtype=np.float64)
    root_pos = np.asarray(root_pos, dtype=np.float64)
    _validate_frame_counts(root_pos, joint_data)

    F = root_pos.shape[0]
    if F < 2:
        return root_pos.copy(), joint_data.copy()

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

    return new_root_pos, _from_quats(new_quats, representation, euler_orders)


def dropout_arrays(
    *,
    root_pos: npt.NDArray[np.float64],
    joint_data: npt.NDArray[np.float64],
    drop_rate: float,
    representation: str,
    rng: np.random.Generator | None = None,
    euler_orders: list[str] | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Frame dropout with SLERP interpolation.

    Randomly drops frames and fills the gaps with SLERP-interpolated
    rotations (via quaternion space) and linearly interpolated root
    positions.  First and last frames are always kept.  Shape is
    unchanged — you get the same ``F`` frames, some replaced by
    interpolated values.

    Parameters
    ----------
    root_pos : ndarray, shape (F, 3)
    joint_data : ndarray, shape (F, J, C)
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
    new_root_pos : ndarray, shape (F, 3)
    new_joint_data : ndarray, shape (F, J, C)

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

    joint_data = np.asarray(joint_data, dtype=np.float64)
    root_pos = np.asarray(root_pos, dtype=np.float64)
    _validate_frame_counts(root_pos, joint_data)

    F = root_pos.shape[0]
    if F < 2 or drop_rate == 0:
        return root_pos.copy(), joint_data.copy()

    keep_mask = rng.random(F) >= drop_rate
    keep_mask[0] = True
    keep_mask[-1] = True
    kept_indices = np.where(keep_mask)[0]

    dropped = np.where(~keep_mask)[0]
    if len(dropped) == 0:
        return root_pos.copy(), joint_data.copy()

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

    return new_root_pos, _from_quats(new_quats, representation, euler_orders)
