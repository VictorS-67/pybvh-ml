"""Representation conversion for motion arrays.

Thin wrapper over :func:`pybvh.rotations.convert` with the
``euler_orders`` → ``order`` parameter name preserved for
pybvh-ml callers.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pybvh import rotations

from .metadata import REPR_CHANNELS


def convert_arrays(
    joint_data: npt.NDArray[np.float64],
    from_repr: str,
    to_repr: str,
    euler_orders: list[str] | None = None,
) -> npt.NDArray[np.float64]:
    """Convert joint rotation arrays between representations.

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

    # pybvh-ml carries rotmat as (F, J, 9); pybvh's rotations.convert
    # uses (F, J, 3, 3). Adapt at the boundary.
    if from_repr == "rotmat":
        F, J = joint_data.shape[:2]
        joint_data = joint_data.reshape(F, J, 3, 3)

    result = rotations.convert(
        joint_data, from_repr, to_repr,
        order=euler_orders, degrees=True)

    if to_repr == "rotmat":
        F, J = result.shape[:2]
        result = result.reshape(F, J, 9)

    return result
