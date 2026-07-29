"""Tensor layout conversion for ML pipelines.

Converts between :class:`~pybvh_ml.MotionArrays` and the tensor
layouts that ML models consume: ``(C, T, V)``, ``(T, V, C)``, and flat
``(T, D)``.

Conventions
-----------
- **C** = channels (max of 3 and joint data channels)
- **T** = time / frames
- **V** = vertices / joints (root is vertex 0, joints are 1..J)
- **D** = flat feature dimension (3 + J * C_joint)

The root vertex carries 3 position channels; when C > 3 (e.g. quat,
6D, or rotmat joint data), the root vertex's remaining ``C - 3``
channels are zero padding — the position values themselves are
unchanged.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .arrays import MotionArrays, require_joint_rot


def _center(
    root_pos: npt.NDArray[np.float64],
    center_root: bool,
) -> npt.NDArray[np.float64]:
    """Optionally subtract first-frame root position.

    Subtracts all three components — pybvh-ml's root-relative tensor
    convention.  This is NOT pybvh's ``centered="first"``, which is
    ground-plane-only (the up coordinate is left untouched there).
    """
    if center_root and root_pos.shape[0] > 0:
        return root_pos - root_pos[0:1]
    return root_pos.copy()


def pack_to_ctv(
    arrays: MotionArrays,
    center_root: bool = True,
) -> npt.NDArray[np.float64]:
    """Pack root position and joint data into ``(C, T, V)`` layout.

    Parameters
    ----------
    arrays : MotionArrays
        Must carry ``joint_rot``.
    center_root : bool
        If True, subtract first frame's root position.
        This flag is for standalone packing of raw extractions.  Clips from a dataset preprocessed with ``center_root=True`` (see :func:`~pybvh_ml.preprocessing.preprocess_directory` and the ``center_root`` key of :func:`~pybvh_ml.preprocessing.load_preprocessed`) are already centered — pass ``False`` for those.  Re-centering a whole already-centered clip is a harmless no-op, but re-centering a *windowed sub-clip* zeroes the window's first frame and destroys the clip-relative trajectory.

    Returns
    -------
    ndarray, shape (C, T, V)
        ``C = max(3, C_joint)``, ``T = F``, ``V = 1 + J``.
        Root is vertex 0: its position fills channels ``0:3``, and
        when ``C_joint > 3`` its channels ``3:C`` are zero padding.
    """
    root_pos = np.asarray(arrays.root_pos, dtype=np.float64)
    joint_data = np.asarray(
        require_joint_rot(arrays, "pack_to_ctv"), dtype=np.float64)
    rp = _center(root_pos, center_root)

    F = rp.shape[0]
    J = joint_data.shape[1]
    C_joint = joint_data.shape[2]
    C = max(3, C_joint)

    tvc = np.zeros((F, 1 + J, C), dtype=np.float64)
    tvc[:, 0, :3] = rp
    tvc[:, 1:, :C_joint] = joint_data

    # Materialize the transpose: consumers hand this to
    # torch.from_numpy(...).view(...) and C-contiguity assumptions.
    return np.ascontiguousarray(tvc.transpose(2, 0, 1))


def pack_to_tvc(
    arrays: MotionArrays,
    center_root: bool = True,
) -> npt.NDArray[np.float64]:
    """Pack root position and joint data into ``(T, V, C)`` layout.

    Parameters
    ----------
    arrays : MotionArrays
        Must carry ``joint_rot``.
    center_root : bool
        If True, subtract first frame's root position.  Arrays from a preprocessed dataset saved with ``center_root=True`` are already centered — see :func:`pack_to_ctv`.

    Returns
    -------
    ndarray, shape (T, V, C)
        ``T = F``, ``V = 1 + J``, ``C = max(3, C_joint)``.
    """
    root_pos = np.asarray(arrays.root_pos, dtype=np.float64)
    joint_data = np.asarray(
        require_joint_rot(arrays, "pack_to_tvc"), dtype=np.float64)
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
    arrays: MotionArrays,
    center_root: bool = True,
) -> npt.NDArray[np.float64]:
    """Pack root position and joint data into flat ``(T, D)`` layout.

    Parameters
    ----------
    arrays : MotionArrays
        Must carry ``joint_rot``.
    center_root : bool
        If True, subtract first frame's root position.  Arrays from a preprocessed dataset saved with ``center_root=True`` are already centered — see :func:`pack_to_ctv`.

    Returns
    -------
    ndarray, shape (T, D)
        ``D = 3 + J * C_joint``.  Root position occupies columns
        ``0:3``, joint data occupies ``3:D``.
    """
    root_pos = np.asarray(arrays.root_pos, dtype=np.float64)
    joint_data = np.asarray(
        require_joint_rot(arrays, "pack_to_flat"), dtype=np.float64)
    rp = _center(root_pos, center_root)

    F = rp.shape[0]
    flat_joints = joint_data.reshape(F, -1)
    return np.concatenate([rp, flat_joints], axis=1)


def _validate_root_channels(root_channels: int, available: int) -> None:
    """Reject a root width the packed array cannot supply.

    Slicing past the end is silent in NumPy: asking for 7 root channels
    of a 4-channel array returns 4 columns and a ``root_pos`` of the
    wrong width, which then propagates into whatever consumes it.
    """
    if root_channels < 1:
        raise ValueError(
            f"root_channels must be >= 1, got {root_channels}")
    if root_channels > available:
        raise ValueError(
            f"root_channels={root_channels} exceeds the array's "
            f"{available} channel(s); pass the root_channels the array "
            f"was packed with (3 for a position root).")


def unpack_from_ctv(
    data: npt.NDArray[np.float64],
    root_channels: int = 3,
) -> MotionArrays:
    """Unpack ``(C, T, V)`` back to root position and joint data.

    Parameters
    ----------
    data : ndarray, shape (C, T, V)
    root_channels : int
        Number of channels used by the root vertex (default 3).

    Returns
    -------
    MotionArrays

    Raises
    ------
    ValueError
        If ``root_channels`` exceeds the array's channel count.
    """
    tvc = np.asarray(data, dtype=np.float64).transpose(1, 2, 0)
    _validate_root_channels(root_channels, tvc.shape[2])
    root_pos = tvc[:, 0, :root_channels].copy()
    joint_data = tvc[:, 1:, :].copy()
    return MotionArrays(root_pos=root_pos, joint_rot=joint_data)


def unpack_from_tvc(
    data: npt.NDArray[np.float64],
    root_channels: int = 3,
) -> MotionArrays:
    """Unpack ``(T, V, C)`` back to root position and joint data.

    Parameters
    ----------
    data : ndarray, shape (T, V, C)
    root_channels : int

    Returns
    -------
    MotionArrays

    Raises
    ------
    ValueError
        If ``root_channels`` exceeds the array's channel count.
    """
    data = np.asarray(data, dtype=np.float64)
    _validate_root_channels(root_channels, data.shape[2])
    root_pos = data[:, 0, :root_channels].copy()
    joint_data = data[:, 1:, :].copy()
    return MotionArrays(root_pos=root_pos, joint_rot=joint_data)


def unpack_from_flat(
    data: npt.NDArray[np.float64],
    root_channels: int = 3,
    joint_channels: int = 3,
) -> MotionArrays:
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
    MotionArrays
    """
    data = np.asarray(data, dtype=np.float64)
    root_pos = data[:, :root_channels].copy()
    flat_joints = data[:, root_channels:]
    if flat_joints.shape[1] % joint_channels != 0:
        raise ValueError(
            f"Cannot unpack {flat_joints.shape[1]} joint columns "
            f"(D={data.shape[1]} minus root_channels={root_channels}) "
            f"into whole joints of joint_channels={joint_channels}. "
            f"Pass the joint_channels the array was packed with "
            f"(e.g. 4 for quat, 6 for 6d).")
    J = flat_joints.shape[1] // joint_channels
    joint_data = flat_joints.reshape(data.shape[0], J, joint_channels).copy()
    return MotionArrays(root_pos=root_pos, joint_rot=joint_data)
