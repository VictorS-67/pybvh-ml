"""Array-level augmentation for ML pipelines.

Operates on pre-extracted NumPy arrays without Bvh objects.
All functions accept any rotation representation supported by pybvh:
``"quaternion"``, ``"6d"``, ``"axisangle"``, ``"rotmat"``, or ``"euler"``.
Euler arrays additionally require an ``euler_orders`` kwarg.

All functions take and return ``(root_pos, joint_data)`` — root position
first, matching pybvh's ``Bvh.from_*`` / ``Bvh.to_*`` convention.  All
parameters are keyword-only: since ``root_pos`` and ``joint_data`` are
shape-compatible ndarrays, accepting them positionally would make a
swapped call silently corrupt data.  Call with
``rotate_vertical(root_pos=..., joint_data=..., angle_deg=..., ...)``.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pybvh import rotations
from pybvh.tools import rotX, rotY, rotZ


# =========================================================================
# Private helpers
# =========================================================================

_ROT_FUNCS = {0: rotX, 1: rotY, 2: rotZ}
_AXIS_IDX = {"x": 0, "y": 1, "z": 2}


def _parse_axis(axis: str) -> tuple[int, float]:
    """Parse a signed-axis string into ``(index, sign)``.

    Accepts exactly the six canonical values ``'+x'``, ``'-x'``,
    ``'+y'``, ``'-y'``, ``'+z'``, ``'-z'`` — matching the
    :attr:`pybvh.Bvh.world_up` / :meth:`pybvh.Bvh.forward_at`
    convention.
    """
    if (not isinstance(axis, str)
            or len(axis) != 2
            or axis[0] not in "+-"
            or axis[1] not in "xyz"):
        raise ValueError(
            f"axis must be one of '+x', '-x', '+y', '-y', '+z', '-z'; "
            f"got {axis!r}")
    return _AXIS_IDX[axis[1]], 1.0 if axis[0] == "+" else -1.0


def _to_quats(
    joint_data: npt.NDArray[np.float64],
    representation: str,
    euler_orders: list[str] | None,
) -> npt.NDArray[np.float64]:
    """Convert joint data to quaternion space for augmentation math."""
    if representation == "quaternion":
        return joint_data
    if representation == "euler":
        if euler_orders is None:
            raise ValueError(
                "euler_orders is required when representation='euler'")
        return rotations.convert(
            joint_data, "euler", "quaternion",
            order=euler_orders, degrees=True)
    return rotations.convert(joint_data, representation, "quaternion")


def _from_quats(
    quats: npt.NDArray[np.float64],
    representation: str,
    euler_orders: list[str] | None,
) -> npt.NDArray[np.float64]:
    """Convert quaternions back to the original representation."""
    if representation == "quaternion":
        return quats
    if representation == "euler":
        return rotations.convert(
            quats, "quaternion", "euler",
            order=euler_orders, degrees=True)
    return rotations.convert(quats, "quaternion", representation)


def _quat_multiply(
    q1: npt.NDArray[np.float64],
    q2: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Hamilton product of two quaternion arrays (w, x, y, z)."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=-1)


def _build_rotation_quat(
    angle_deg: float,
    up_idx: int,
) -> npt.NDArray[np.float64]:
    """Build a unit quaternion for rotation around a cardinal axis."""
    half = np.radians(angle_deg) / 2.0
    q = np.array([np.cos(half), 0.0, 0.0, 0.0])
    q[1 + up_idx] = np.sin(half)
    return q


def _build_rotation_matrix(
    angle_deg: float,
    up_idx: int,
) -> npt.NDArray[np.float64]:
    """Build a 3×3 rotation matrix for rotation around a cardinal axis."""
    return _ROT_FUNCS[up_idx](np.radians(angle_deg))


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
    angle_deg: float,
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
    angle_deg : float
        Rotation angle in degrees.
    up_axis : str
        Signed axis string: ``'+x'``, ``'-x'``, ``'+y'``, ``'-y'``,
        ``'+z'``, or ``'-z'``.  The sign flips the rotation direction,
        so ``'+y'`` and ``'-y'`` produce opposite yaws for the same
        ``angle_deg``.  Typically ``bvh.world_up``.
    representation : str
        One of ``"quaternion"``, ``"6d"``, ``"axisangle"``,
        ``"rotmat"``, ``"euler"``.
    euler_orders : list of str, optional
        Per-joint Euler order strings (e.g. ``["ZYX", "ZYX", ...]``).
        Required when ``representation="euler"``, ignored otherwise.

    Returns
    -------
    new_root_pos : ndarray, shape (F, 3)
    new_joint_data : ndarray, shape (F, J, C)
    """
    joint_data = np.array(joint_data, dtype=np.float64)
    root_pos = np.array(root_pos, dtype=np.float64)

    up_idx, up_sign = _parse_axis(up_axis)
    signed_angle = angle_deg * up_sign
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
    new_quats[:, 0] = _quat_multiply(q_rot, quats[:, 0])
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
        ``[(left_idx, right_idx), ...]`` in joint-array space.
    lateral_axis : str
        Signed axis string: ``'+x'``, ``'-x'``, ``'+y'``, ``'-y'``,
        ``'+z'``, or ``'-z'``.  The sign is accepted for API symmetry
        with :func:`rotate_vertical` but does not affect the result
        (mirror is sign-invariant).
    representation : str
        One of ``"quaternion"``, ``"6d"``, ``"axisangle"``,
        ``"rotmat"``, ``"euler"``.
    euler_orders : list of str, optional
        Required when ``representation="euler"``, ignored otherwise.

    Returns
    -------
    new_root_pos : ndarray, shape (F, 3)
    new_joint_data : ndarray, shape (F, J, C)
    """
    new_data = np.array(joint_data, dtype=np.float64)
    new_root_pos = np.array(root_pos, dtype=np.float64)

    lateral_idx, _ = _parse_axis(lateral_axis)
    new_root_pos[:, lateral_idx] *= -1.0
    _swap_lr_pairs(new_data, lr_joint_pairs)

    # 6D and quaternion: apply the analytic sign mask directly.
    if representation == "6d":
        new_data *= _mirror_sign_rot6d(lateral_idx)
        return new_root_pos, new_data

    if representation == "quaternion":
        new_data *= _mirror_sign_quat(lateral_idx)
        return new_root_pos, new_data

    # All other representations: convert the (already swapped) data to
    # quaternions, apply the sign mask, then convert back.
    quats = _to_quats(new_data, representation, euler_orders)
    quats *= _mirror_sign_quat(lateral_idx)
    return new_root_pos, _from_quats(quats, representation, euler_orders)


def add_joint_noise(
    *,
    root_pos: npt.NDArray[np.float64],
    joint_data: npt.NDArray[np.float64],
    sigma_deg: float,
    representation: str,
    sigma_pos: float = 0.0,
    rng: np.random.Generator | None = None,
    euler_orders: list[str] | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Add Gaussian rotation noise to joint arrays.

    For each joint at each frame, generates a small random rotation
    (axis uniformly random on the unit sphere, angle sampled from
    ``N(0, sigma_deg)`` in degrees) and composes it with the original
    rotation: ``q_noisy = q_noise * q_original``.

    Optionally adds Gaussian noise to root positions as well.

    Parameters
    ----------
    root_pos : ndarray, shape (F, 3)
    joint_data : ndarray, shape (F, J, C)
    sigma_deg : float
        Standard deviation of rotation noise in degrees.
    representation : str
        One of ``"quaternion"``, ``"6d"``, ``"axisangle"``,
        ``"rotmat"``, ``"euler"``.
    sigma_pos : float
        Standard deviation of root position noise (default 0 = none).
    rng : numpy Generator, optional
    euler_orders : list of str, optional
        Required when ``representation="euler"``, ignored otherwise.

    Returns
    -------
    new_root_pos : ndarray, shape (F, 3)
    new_joint_data : ndarray, shape (F, J, C)
    """
    if rng is None:
        rng = np.random.default_rng()

    joint_data = np.asarray(joint_data, dtype=np.float64)
    root_pos = np.asarray(root_pos, dtype=np.float64)

    quats = _to_quats(joint_data, representation, euler_orders)
    F, J, _ = quats.shape

    axis = rng.standard_normal((F, J, 3))
    norm = np.linalg.norm(axis, axis=-1, keepdims=True)
    norm = np.where(norm < 1e-15, 1.0, norm)
    axis = axis / norm

    half_angle = np.radians(rng.normal(0, sigma_deg, (F, J))) / 2.0
    q_noise = np.empty((F, J, 4), dtype=np.float64)
    q_noise[..., 0] = np.cos(half_angle)
    q_noise[..., 1:] = np.sin(half_angle)[..., np.newaxis] * axis

    noisy_quats = _quat_multiply(q_noise, quats)
    noisy_quats /= np.linalg.norm(noisy_quats, axis=-1, keepdims=True)

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
        One of ``"quaternion"``, ``"6d"``, ``"axisangle"``,
        ``"rotmat"``, ``"euler"``.
    euler_orders : list of str, optional
        Required when ``representation="euler"``, ignored otherwise.

    Returns
    -------
    new_root_pos : ndarray, shape (F', 3)
    new_joint_data : ndarray, shape (F', J, C)
        ``F' = max(2, round(F / factor))``.
    """
    if factor <= 0:
        raise ValueError(f"factor must be > 0, got {factor}")

    joint_data = np.asarray(joint_data, dtype=np.float64)
    root_pos = np.asarray(root_pos, dtype=np.float64)

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
    dt = t_right - t_left
    dt = np.where(dt < 1e-15, 1.0, dt)
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
        One of ``"quaternion"``, ``"6d"``, ``"axisangle"``,
        ``"rotmat"``, ``"euler"``.
    rng : numpy Generator, optional
    euler_orders : list of str, optional
        Required when ``representation="euler"``, ignored otherwise.

    Returns
    -------
    new_root_pos : ndarray, shape (F, 3)
    new_joint_data : ndarray, shape (F, J, C)
    """
    if rng is None:
        rng = np.random.default_rng()

    joint_data = np.asarray(joint_data, dtype=np.float64)
    root_pos = np.asarray(root_pos, dtype=np.float64)

    F = root_pos.shape[0]
    if F < 2 or drop_rate <= 0:
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

    dt = (right_idx - left_idx).astype(np.float64)
    dt = np.where(dt < 1e-15, 1.0, dt)
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
