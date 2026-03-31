"""Tensor layout conversion for ML pipelines.

Converts between pybvh's structured arrays (root_pos, joint_data)
and the tensor layouts that ML models consume: ``(C, T, V)``,
``(T, V, C)``, and flat ``(T, D)``.

Conventions
-----------
- **C** = channels (max of 3 and joint data channels)
- **T** = time / frames
- **V** = vertices / joints (root is vertex 0, joints are 1..J)
- **D** = flat feature dimension (3 + J * C_joint)

Root position (3 channels) is zero-padded to C when C > 3.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt


def _center(
    root_pos: npt.NDArray[np.float64],
    center_root: bool,
) -> npt.NDArray[np.float64]:
    """Optionally subtract first-frame root position."""
    if center_root and root_pos.shape[0] > 0:
        return root_pos - root_pos[0:1]
    return root_pos.copy()


def pack_to_ctv(
    root_pos: npt.NDArray[np.float64],
    joint_data: npt.NDArray[np.float64],
    center_root: bool = True,
) -> npt.NDArray[np.float64]:
    """Pack root position and joint data into ``(C, T, V)`` layout.

    Parameters
    ----------
    root_pos : ndarray, shape (F, 3)
        Root translation per frame.
    joint_data : ndarray, shape (F, J, C_joint)
        Per-joint rotation data (Euler, quaternion, 6D, etc.).
    center_root : bool
        If True, subtract first frame's root position.

    Returns
    -------
    ndarray, shape (C, T, V)
        ``C = max(3, C_joint)``, ``T = F``, ``V = 1 + J``.
        Root is vertex 0, zero-padded to C channels if C_joint > 3.
    """
    root_pos = np.asarray(root_pos, dtype=np.float64)
    joint_data = np.asarray(joint_data, dtype=np.float64)
    rp = _center(root_pos, center_root)

    F = rp.shape[0]
    J = joint_data.shape[1]
    C_joint = joint_data.shape[2]
    C = max(3, C_joint)

    tvc = np.zeros((F, 1 + J, C), dtype=np.float64)
    tvc[:, 0, :3] = rp
    tvc[:, 1:, :C_joint] = joint_data

    return tvc.transpose(2, 0, 1)  # (T, V, C) → (C, T, V)


def pack_to_tvc(
    root_pos: npt.NDArray[np.float64],
    joint_data: npt.NDArray[np.float64],
    center_root: bool = True,
) -> npt.NDArray[np.float64]:
    """Pack root position and joint data into ``(T, V, C)`` layout.

    Parameters
    ----------
    root_pos : ndarray, shape (F, 3)
    joint_data : ndarray, shape (F, J, C_joint)
    center_root : bool

    Returns
    -------
    ndarray, shape (T, V, C)
        ``T = F``, ``V = 1 + J``, ``C = max(3, C_joint)``.
    """
    root_pos = np.asarray(root_pos, dtype=np.float64)
    joint_data = np.asarray(joint_data, dtype=np.float64)
    rp = _center(root_pos, center_root)

    F = rp.shape[0]
    J = joint_data.shape[1]
    C_joint = joint_data.shape[2]
    C = max(3, C_joint)

    tvc = np.zeros((F, 1 + J, C), dtype=np.float64)
    tvc[:, 0, :3] = rp
    tvc[:, 1:, :C_joint] = joint_data

    return tvc


def pack_to_flat(
    root_pos: npt.NDArray[np.float64],
    joint_data: npt.NDArray[np.float64],
    center_root: bool = True,
) -> npt.NDArray[np.float64]:
    """Pack root position and joint data into flat ``(T, D)`` layout.

    Parameters
    ----------
    root_pos : ndarray, shape (F, 3)
    joint_data : ndarray, shape (F, J, C_joint)
    center_root : bool

    Returns
    -------
    ndarray, shape (T, D)
        ``D = 3 + J * C_joint``.  Root position occupies columns
        ``0:3``, joint data occupies ``3:D``.
    """
    root_pos = np.asarray(root_pos, dtype=np.float64)
    joint_data = np.asarray(joint_data, dtype=np.float64)
    rp = _center(root_pos, center_root)

    F = rp.shape[0]
    flat_joints = joint_data.reshape(F, -1)
    return np.concatenate([rp, flat_joints], axis=1)


def unpack_from_ctv(
    data: npt.NDArray[np.float64],
    root_channels: int = 3,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Unpack ``(C, T, V)`` back to root position and joint data.

    Parameters
    ----------
    data : ndarray, shape (C, T, V)
    root_channels : int
        Number of channels used by the root vertex (default 3).

    Returns
    -------
    root_pos : ndarray, shape (T, root_channels)
    joint_data : ndarray, shape (T, V-1, C)
    """
    tvc = np.asarray(data, dtype=np.float64).transpose(1, 2, 0)
    root_pos = tvc[:, 0, :root_channels].copy()
    joint_data = tvc[:, 1:, :].copy()
    return root_pos, joint_data


def unpack_from_tvc(
    data: npt.NDArray[np.float64],
    root_channels: int = 3,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Unpack ``(T, V, C)`` back to root position and joint data.

    Parameters
    ----------
    data : ndarray, shape (T, V, C)
    root_channels : int

    Returns
    -------
    root_pos : ndarray, shape (T, root_channels)
    joint_data : ndarray, shape (T, V-1, C)
    """
    data = np.asarray(data, dtype=np.float64)
    root_pos = data[:, 0, :root_channels].copy()
    joint_data = data[:, 1:, :].copy()
    return root_pos, joint_data


def unpack_from_flat(
    data: npt.NDArray[np.float64],
    root_channels: int = 3,
    joint_channels: int = 3,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Unpack flat ``(T, D)`` back to root position and joint data.

    Parameters
    ----------
    data : ndarray, shape (T, D)
    root_channels : int
        Number of columns for root position (default 3).
    joint_channels : int
        Number of channels per joint (default 3).  Used to reshape
        the remaining columns into ``(T, J, joint_channels)``.

    Returns
    -------
    root_pos : ndarray, shape (T, root_channels)
    joint_data : ndarray, shape (T, J, joint_channels)
    """
    data = np.asarray(data, dtype=np.float64)
    root_pos = data[:, :root_channels].copy()
    flat_joints = data[:, root_channels:]
    J = flat_joints.shape[1] // joint_channels
    joint_data = flat_joints.reshape(data.shape[0], J, joint_channels).copy()
    return root_pos, joint_data
