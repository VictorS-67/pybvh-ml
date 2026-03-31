"""Representation conversion for motion arrays.

Unified dispatch wrapping pybvh's rotation functions for the
common ``(F, J, C)`` array shape.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pybvh import rotations


REPR_CHANNELS: dict[str, int] = {
    "euler": 3,
    "axisangle": 3,
    "quaternion": 4,
    "6d": 6,
    "rotmat": 9,
}


def convert_arrays(
    joint_data: npt.NDArray[np.float64],
    from_repr: str,
    to_repr: str,
    euler_orders: list[str] | None = None,
) -> npt.NDArray[np.float64]:
    """Convert joint rotation arrays between representations.

    Routes through rotation matrices as an intermediate when no
    direct conversion path exists.

    Parameters
    ----------
    joint_data : ndarray, shape (F, J, C_from)
        Input joint rotation data.
    from_repr : str
        Source representation: ``"euler"``, ``"quaternion"``,
        ``"6d"``, ``"axisangle"``, ``"rotmat"``.
    to_repr : str
        Target representation (same options).
    euler_orders : list of str, optional
        Per-joint Euler orders (e.g. ``['ZYX', 'ZYX', ...]``).
        **Required** when *from_repr* or *to_repr* is ``"euler"``.

    Returns
    -------
    ndarray, shape (F, J, C_to)
        Converted joint data.  For ``"rotmat"``, the last dimension
        is 9 (flattened 3x3).
    """
    for name, val in [("from_repr", from_repr), ("to_repr", to_repr)]:
        if val not in REPR_CHANNELS:
            raise ValueError(
                f"Unknown {name} '{val}'. "
                f"Choose from {list(REPR_CHANNELS)}")

    joint_data = np.asarray(joint_data, dtype=np.float64)

    if from_repr == to_repr:
        return joint_data.copy()

    needs_euler = from_repr == "euler" or to_repr == "euler"
    if needs_euler and euler_orders is None:
        raise ValueError(
            "euler_orders is required when converting from/to 'euler'")

    # Convert to rotation matrices first
    rotmats = _to_rotmat(joint_data, from_repr, euler_orders)
    # Convert from rotation matrices to target
    return _from_rotmat(rotmats, to_repr, euler_orders)


def _to_rotmat(
    data: npt.NDArray[np.float64],
    from_repr: str,
    euler_orders: list[str] | None,
) -> npt.NDArray[np.float64]:
    """Convert (F, J, C) to (F, J, 3, 3) rotation matrices."""
    F, J = data.shape[:2]

    if from_repr == "rotmat":
        return data.reshape(F, J, 3, 3)

    if from_repr == "quaternion":
        return rotations.quat_to_rotmat(data)  # (F, J, 3, 3)

    if from_repr == "6d":
        return rotations.rot6d_to_rotmat(data)  # (F, J, 3, 3)

    if from_repr == "axisangle":
        return rotations.axisangle_to_rotmat(data)  # (F, J, 3, 3)

    if from_repr == "euler":
        assert euler_orders is not None
        return _euler_to_rotmat_per_joint(data, euler_orders)

    raise ValueError(f"Unknown from_repr '{from_repr}'")  # pragma: no cover


def _from_rotmat(
    rotmats: npt.NDArray[np.float64],
    to_repr: str,
    euler_orders: list[str] | None,
) -> npt.NDArray[np.float64]:
    """Convert (F, J, 3, 3) rotation matrices to (F, J, C)."""
    F, J = rotmats.shape[:2]

    if to_repr == "rotmat":
        return rotmats.reshape(F, J, 9)

    if to_repr == "quaternion":
        return rotations.rotmat_to_quat(rotmats)  # (F, J, 4)

    if to_repr == "6d":
        return rotations.rotmat_to_rot6d(rotmats)  # (F, J, 6)

    if to_repr == "axisangle":
        return rotations.rotmat_to_axisangle(rotmats)  # (F, J, 3)

    if to_repr == "euler":
        assert euler_orders is not None
        return _rotmat_to_euler_per_joint(rotmats, euler_orders)

    raise ValueError(f"Unknown to_repr '{to_repr}'")  # pragma: no cover


def _euler_to_rotmat_per_joint(
    data: npt.NDArray[np.float64],
    euler_orders: list[str],
) -> npt.NDArray[np.float64]:
    """Convert (F, J, 3) Euler angles to (F, J, 3, 3) with per-joint orders."""
    F, J, _ = data.shape
    result = np.empty((F, J, 3, 3), dtype=np.float64)

    unique_orders = set(euler_orders)
    for order in unique_orders:
        mask = [i for i, o in enumerate(euler_orders) if o == order]
        result[:, mask] = rotations.euler_to_rotmat(
            data[:, mask], order, degrees=True)

    return result


def _rotmat_to_euler_per_joint(
    rotmats: npt.NDArray[np.float64],
    euler_orders: list[str],
) -> npt.NDArray[np.float64]:
    """Convert (F, J, 3, 3) rotation matrices to (F, J, 3) with per-joint orders."""
    F, J = rotmats.shape[:2]
    result = np.empty((F, J, 3), dtype=np.float64)

    unique_orders = set(euler_orders)
    for order in unique_orders:
        mask = [i for i, o in enumerate(euler_orders) if o == order]
        result[:, mask] = rotations.rotmat_to_euler(
            rotmats[:, mask], order, degrees=True)

    return result
