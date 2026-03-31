"""Array-level augmentation for ML pipelines.

Operates on pre-extracted NumPy arrays without Bvh objects.
Supports quaternion ``(F, J, 4)`` and 6D ``(F, J, 6)`` representations.

All functions return ``(new_joint_data, new_root_pos)`` — joint data
first, matching pybvh's convention.
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
    """Build a unit quaternion for rotation around an axis."""
    half = np.radians(angle_deg) / 2.0
    q = np.array([np.cos(half), 0.0, 0.0, 0.0])
    q[1 + up_idx] = np.sin(half)
    return q


def _build_rotation_matrix(
    angle_deg: float,
    up_idx: int,
) -> npt.NDArray[np.float64]:
    """Build a 3x3 rotation matrix for rotation around an axis."""
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
# Quaternion-space augmentation
# =========================================================================

def rotate_quaternions_vertical(
    joint_quats: npt.NDArray[np.float64],
    root_pos: npt.NDArray[np.float64],
    angle_deg: float,
    up_idx: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Rotate quaternion arrays around the vertical axis.

    Only the root joint (index 0) and root position are modified;
    non-root joints are in parent-local space and stay unchanged.

    Parameters
    ----------
    joint_quats : ndarray, shape (F, J, 4)
        Quaternion rotations ``(w, x, y, z)`` per joint per frame.
    root_pos : ndarray, shape (F, 3)
        Root translation per frame.
    angle_deg : float
        Rotation angle in degrees.
    up_idx : int
        Vertical axis: 0 = X, 1 = Y, 2 = Z.

    Returns
    -------
    new_joint_quats : ndarray, shape (F, J, 4)
    new_root_pos : ndarray, shape (F, 3)
    """
    joint_quats = np.array(joint_quats, dtype=np.float64)
    root_pos = np.array(root_pos, dtype=np.float64)

    R_vert = _build_rotation_matrix(angle_deg, up_idx)
    q_rot = _build_rotation_quat(angle_deg, up_idx)

    new_quats = joint_quats.copy()
    new_quats[:, 0] = _quat_multiply(q_rot, joint_quats[:, 0])

    new_root_pos = (R_vert @ root_pos.T).T

    return new_quats, new_root_pos


def mirror_quaternions(
    joint_quats: npt.NDArray[np.float64],
    root_pos: npt.NDArray[np.float64],
    lr_joint_pairs: list[tuple[int, int]],
    lateral_idx: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Mirror quaternion arrays left-right.

    Parameters
    ----------
    joint_quats : ndarray, shape (F, J, 4)
        Quaternion rotations ``(w, x, y, z)``.
    root_pos : ndarray, shape (F, 3)
    lr_joint_pairs : list of (int, int)
        ``[(left_idx, right_idx), ...]`` in ``joint_angles`` space.
    lateral_idx : int
        Lateral axis: 0 = X, 1 = Y, 2 = Z.

    Returns
    -------
    new_joint_quats : ndarray, shape (F, J, 4)
    new_root_pos : ndarray, shape (F, 3)
    """
    new_quats = np.array(joint_quats, dtype=np.float64)
    new_root_pos = np.array(root_pos, dtype=np.float64)

    # 1. Negate root lateral component
    new_root_pos[:, lateral_idx] *= -1.0

    # 2. Swap L/R joint pairs
    _swap_lr_pairs(new_quats, lr_joint_pairs)

    # 3. Reflect each quaternion
    signs = _mirror_sign_quat(lateral_idx)
    new_quats *= signs

    return new_quats, new_root_pos


def speed_perturbation_arrays(
    joint_quats: npt.NDArray[np.float64],
    root_pos: npt.NDArray[np.float64],
    factor: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Speed perturbation on pre-extracted quaternion arrays.

    Uses SLERP for rotation interpolation and linear interpolation
    for root position.

    Parameters
    ----------
    joint_quats : ndarray, shape (F, J, 4)
    root_pos : ndarray, shape (F, 3)
    factor : float
        Speed factor.  ``> 1`` = faster (fewer frames),
        ``< 1`` = slower (more frames).

    Returns
    -------
    new_joint_quats : ndarray, shape (F', J, 4)
    new_root_pos : ndarray, shape (F', 3)
        ``F' = max(2, round(F / factor))``.
    """
    if factor <= 0:
        raise ValueError(f"factor must be > 0, got {factor}")

    joint_quats = np.asarray(joint_quats, dtype=np.float64)
    root_pos = np.asarray(root_pos, dtype=np.float64)

    F = root_pos.shape[0]
    if F < 2:
        return joint_quats.copy(), root_pos.copy()

    F_new = max(2, round(F / factor))
    J = joint_quats.shape[1]

    t_orig = np.linspace(0.0, 1.0, F)
    t_new = np.linspace(0.0, 1.0, F_new)

    # Root position: linear interpolation per axis
    new_root_pos = np.empty((F_new, 3), dtype=np.float64)
    for ax in range(3):
        new_root_pos[:, ax] = np.interp(t_new, t_orig, root_pos[:, ax])

    # Joint quaternions: SLERP
    idx_right = np.searchsorted(t_orig, t_new, side='right')
    idx_right = np.clip(idx_right, 1, F - 1)
    idx_left = idx_right - 1

    t_left = t_orig[idx_left]
    t_right = t_orig[idx_right]
    dt = t_right - t_left
    dt = np.where(dt < 1e-15, 1.0, dt)
    alpha = (t_new - t_left) / dt  # (F_new,)

    q_left = joint_quats[idx_left]    # (F_new, J, 4)
    q_right = joint_quats[idx_right]  # (F_new, J, 4)
    alpha_jt = np.broadcast_to(alpha[:, np.newaxis], (F_new, J))

    new_quats = rotations.quat_slerp(q_left, q_right, alpha_jt)

    return new_quats, new_root_pos


def dropout_arrays(
    joint_quats: npt.NDArray[np.float64],
    root_pos: npt.NDArray[np.float64],
    drop_rate: float,
    rng: np.random.Generator | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Frame dropout with SLERP interpolation on quaternion arrays.

    Randomly drops frames and fills gaps with SLERP-interpolated
    quaternions and linearly interpolated root positions.  First and
    last frames are always kept.

    Parameters
    ----------
    joint_quats : ndarray, shape (F, J, 4)
    root_pos : ndarray, shape (F, 3)
    drop_rate : float
        Fraction of frames to drop, in ``[0, 1)``.
    rng : numpy Generator, optional

    Returns
    -------
    new_joint_quats : ndarray, shape (F, J, 4)
    new_root_pos : ndarray, shape (F, 3)
        Same frame count; dropped frames replaced with interpolated
        values.
    """
    if rng is None:
        rng = np.random.default_rng()

    joint_quats = np.asarray(joint_quats, dtype=np.float64)
    root_pos = np.asarray(root_pos, dtype=np.float64)

    F = root_pos.shape[0]
    if F < 2 or drop_rate <= 0:
        return joint_quats.copy(), root_pos.copy()

    # Build keep mask — always keep first and last
    keep_mask = rng.random(F) >= drop_rate
    keep_mask[0] = True
    keep_mask[-1] = True
    kept_indices = np.where(keep_mask)[0]

    new_quats = joint_quats.copy()
    new_root_pos = root_pos.copy()

    dropped = np.where(~keep_mask)[0]
    if len(dropped) == 0:
        return new_quats, new_root_pos

    # For each dropped frame, find surrounding kept frames
    ins = np.searchsorted(kept_indices, dropped, side='right')
    left_idx = kept_indices[np.clip(ins - 1, 0, len(kept_indices) - 1)]
    right_idx = kept_indices[np.clip(ins, 0, len(kept_indices) - 1)]

    dt = (right_idx - left_idx).astype(np.float64)
    dt = np.where(dt < 1e-15, 1.0, dt)
    alpha = (dropped - left_idx).astype(np.float64) / dt  # (num_dropped,)

    # Root position: linear interpolation
    for ax in range(3):
        new_root_pos[dropped, ax] = (
            (1.0 - alpha) * root_pos[left_idx, ax]
            + alpha * root_pos[right_idx, ax])

    # Joint quaternions: SLERP
    J = joint_quats.shape[1]
    q_left = joint_quats[left_idx]    # (num_dropped, J, 4)
    q_right = joint_quats[right_idx]  # (num_dropped, J, 4)
    alpha_jt = np.broadcast_to(
        alpha[:, np.newaxis], (len(dropped), J))
    new_quats[dropped] = rotations.quat_slerp(q_left, q_right, alpha_jt)

    return new_quats, new_root_pos


def add_joint_noise_quaternions(
    joint_quats: npt.NDArray[np.float64],
    root_pos: npt.NDArray[np.float64],
    sigma_deg: float,
    sigma_pos: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Add Gaussian noise to quaternion joint rotations.

    For each joint at each frame, generates a small random rotation
    (axis uniformly random on the unit sphere, angle sampled from
    ``N(0, sigma_deg)`` in degrees) and composes it with the original
    quaternion: ``q_noisy = q_noise * q_original``.

    Optionally adds Gaussian noise to root positions as well.

    Parameters
    ----------
    joint_quats : ndarray, shape (F, J, 4)
        Quaternion rotations ``(w, x, y, z)`` per joint per frame.
    root_pos : ndarray, shape (F, 3)
        Root translation per frame.
    sigma_deg : float
        Standard deviation of rotation noise in degrees.
    sigma_pos : float
        Standard deviation of position noise (default 0 = no noise).
    rng : numpy Generator, optional

    Returns
    -------
    new_joint_quats : ndarray, shape (F, J, 4)
    new_root_pos : ndarray, shape (F, 3)
    """
    if rng is None:
        rng = np.random.default_rng()

    joint_quats = np.asarray(joint_quats, dtype=np.float64)
    root_pos = np.asarray(root_pos, dtype=np.float64)

    F, J, _ = joint_quats.shape

    # 1. Random axis (uniform on unit sphere) per joint per frame
    axis = rng.standard_normal((F, J, 3))
    norm = np.linalg.norm(axis, axis=-1, keepdims=True)
    norm = np.where(norm < 1e-15, 1.0, norm)
    axis = axis / norm

    # 2. Random angle from N(0, sigma_deg), converted to radians
    half_angle = np.radians(rng.normal(0, sigma_deg, (F, J))) / 2.0

    # 3. Build noise quaternion: q = [cos(a/2), sin(a/2) * axis]
    q_noise = np.empty((F, J, 4), dtype=np.float64)
    q_noise[..., 0] = np.cos(half_angle)
    q_noise[..., 1:] = np.sin(half_angle)[..., np.newaxis] * axis

    # 4. Compose: q_noisy = q_noise * q_original
    new_quats = _quat_multiply(q_noise, joint_quats)

    # 5. Re-normalize for numerical safety
    new_quats /= np.linalg.norm(new_quats, axis=-1, keepdims=True)

    # 6. Root position noise
    new_root_pos = root_pos.copy()
    if sigma_pos > 0:
        new_root_pos = new_root_pos + rng.normal(0, sigma_pos, root_pos.shape)

    return new_quats, new_root_pos


# =========================================================================
# 6D-space augmentation
# =========================================================================

def rotate_rot6d_vertical(
    joint_rot6d: npt.NDArray[np.float64],
    root_pos: npt.NDArray[np.float64],
    angle_deg: float,
    up_idx: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Rotate 6D rotation arrays around the vertical axis.

    Only modifies the root joint (index 0) and root position.
    Since ``R_vert`` is orthogonal, the result is a valid rotation
    without needing Gram-Schmidt re-orthogonalization.

    Parameters
    ----------
    joint_rot6d : ndarray, shape (F, J, 6)
        6D rotations (first two columns of rotation matrix).
    root_pos : ndarray, shape (F, 3)
    angle_deg : float
    up_idx : int

    Returns
    -------
    new_joint_rot6d : ndarray, shape (F, J, 6)
    new_root_pos : ndarray, shape (F, 3)
    """
    joint_rot6d = np.array(joint_rot6d, dtype=np.float64)
    root_pos = np.array(root_pos, dtype=np.float64)

    R_vert = _build_rotation_matrix(angle_deg, up_idx)

    new_rot6d = joint_rot6d.copy()
    # Split root 6D into two 3D column vectors, rotate each
    col0 = joint_rot6d[:, 0, :3]  # (F, 3)
    col1 = joint_rot6d[:, 0, 3:]  # (F, 3)
    new_rot6d[:, 0, :3] = (R_vert @ col0.T).T
    new_rot6d[:, 0, 3:] = (R_vert @ col1.T).T

    new_root_pos = (R_vert @ root_pos.T).T

    return new_rot6d, new_root_pos


def mirror_rot6d(
    joint_rot6d: npt.NDArray[np.float64],
    root_pos: npt.NDArray[np.float64],
    lr_joint_pairs: list[tuple[int, int]],
    lateral_idx: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Mirror 6D rotation arrays left-right.

    Uses the identity ``R'[i,j] = s_i * s_j * R[i,j]`` where
    ``s[lateral_idx] = -1`` and ``s`` is ``+1`` elsewhere.
    No rotation matrix conversion needed.

    Parameters
    ----------
    joint_rot6d : ndarray, shape (F, J, 6)
    root_pos : ndarray, shape (F, 3)
    lr_joint_pairs : list of (int, int)
    lateral_idx : int

    Returns
    -------
    new_joint_rot6d : ndarray, shape (F, J, 6)
    new_root_pos : ndarray, shape (F, 3)
    """
    new_rot6d = np.array(joint_rot6d, dtype=np.float64)
    new_root_pos = np.array(root_pos, dtype=np.float64)

    # 1. Negate root lateral component
    new_root_pos[:, lateral_idx] *= -1.0

    # 2. Swap L/R joint pairs
    _swap_lr_pairs(new_rot6d, lr_joint_pairs)

    # 3. Reflect 6D
    signs = _mirror_sign_rot6d(lateral_idx)
    new_rot6d *= signs

    return new_rot6d, new_root_pos
