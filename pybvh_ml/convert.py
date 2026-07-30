"""Representation conversion for motion arrays.

Two levels of the same conversion core, so representation conversion composes with the rest of the package at the level the caller is working at:

- :func:`convert_arrays` takes and returns a :class:`~pybvh_ml.MotionArrays` — the container form, and the one to reach for next to augmentation and packing.
- :func:`convert_rotations` takes and returns a bare ``(F, J, C)`` rotation array, for the case with no root stream to carry (a model's rotation output, a cached quaternion array).

Both are thin wrappers over :func:`pybvh.rotations.convert` with the ``euler_orders`` → ``order`` parameter name preserved for pybvh-ml callers.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pybvh import rotations
from pybvh.rotations import REPRESENTATION_CHANNELS

from .arrays import MotionArrays, require_joint_rot


def convert_arrays(
    arrays: MotionArrays,
    from_repr: str,
    to_repr: str,
    euler_orders: list[str] | None = None,
) -> MotionArrays:
    """Convert a clip's joint rotations between representations.

    The container-level form: :class:`~pybvh_ml.MotionArrays` in, a new one out, like every other array-level function in the package — so a conversion drops into an extract → convert → augment → pack chain without breaking the container open and reassembling it.

    ``root_pos`` is carried through **unchanged** (it is a translation, which no rotation representation applies to), so the result differs from the input in ``joint_rot`` alone.

    Parameters
    ----------
    arrays : MotionArrays
        The clip to convert. Must carry ``joint_rot``. Positional because it is a distinct type; see :class:`~pybvh_ml.MotionArrays`.
    from_repr : str
        Source representation: ``"euler"``, ``"quat"``, ``"6d"``, ``"axisangle"``, ``"rotmat"``. The container does not record which one ``joint_rot`` is in, so the caller declares it here.
    to_repr : str
        Target representation (same options).
    euler_orders : list of str, optional
        Per-joint Euler orders (e.g. ``['ZYX', 'ZYX', ...]``).
        **Required** when *from_repr* or *to_repr* is ``"euler"``.

    Returns
    -------
    MotionArrays
        Freshly converted ``joint_rot``; never aliases the input's, not even when ``from_repr == to_repr``.

    Raises
    ------
    ValueError
        If either representation token is unknown, if ``euler_orders`` is missing for an euler-side conversion, or if *arrays* carries no ``joint_rot``.

    Examples
    --------
    >>> arrays = MotionArrays.from_bvh(bvh, "euler")
    >>> arrays = convert_arrays(arrays, "euler", "6d",
    ...                         euler_orders=bvh.euler_orders)
    >>> arrays.joint_rot.shape
    (120, 31, 6)

    See Also
    --------
    convert_rotations : The same conversion on a bare rotation array.
    """
    if not isinstance(arrays, MotionArrays):
        raise TypeError(
            f"convert_arrays takes a MotionArrays, got "
            f"{type(arrays).__name__}. It took a bare joint_data array "
            f"before 0.5.0: wrap the clip with MotionArrays(root_pos=..., "
            f"joint_rot=...) and read out.joint_rot back, or call "
            f"convert_rotations(joint_rot, ...) if you have rotations "
            f"without a root stream.")
    joint_rot = require_joint_rot(arrays, "convert_arrays")
    return arrays.replace(joint_rot=convert_rotations(
        joint_rot, from_repr, to_repr, euler_orders=euler_orders))


def convert_rotations(
    joint_rot: npt.NDArray[np.float64],
    from_repr: str,
    to_repr: str,
    euler_orders: list[str] | None = None,
) -> npt.NDArray[np.float64]:
    """Convert a bare joint-rotation array between representations.

    The rotation-level form, for data with no root stream attached — a model's rotation output, or a cached quaternion array. With a clip in hand, prefer :func:`convert_arrays`, which keeps the container intact.

    Parameters
    ----------
    joint_rot : ndarray, shape (F, J, C_from)
        Input joint rotation data.  Euler angles are in radians
        (matching ``pybvh.Bvh.joint_angles``).
    from_repr : str
        Source representation: ``"euler"``, ``"quat"``,
        ``"6d"``, ``"axisangle"``, ``"rotmat"``.
    to_repr : str
        Target representation (same options).
    euler_orders : list of str, optional
        Per-joint Euler orders (e.g. ``['ZYX', 'ZYX', ...]``).
        **Required** when *from_repr* or *to_repr* is ``"euler"``.

    Returns
    -------
    ndarray, shape (F, J, C_to)
        Converted joint data, always freshly allocated. Rotation matrices are carried **flat** as ``(F, J, 9)``, pybvh-ml's layout throughout — the alternative, pybvh's ``(F, J, 3, 3)``, is what this function adapts to at the pybvh boundary; reshape with ``.reshape(F, J, 3, 3)`` if a consumer wants the nested form. Computation is in ``float64`` regardless of the input dtype, since that is what pybvh's rotation math returns.

    See Also
    --------
    convert_arrays : The same conversion on a whole clip.
    """
    for name, val in [("from_repr", from_repr), ("to_repr", to_repr)]:
        if val not in REPRESENTATION_CHANNELS:
            raise ValueError(
                f"Unknown {name} '{val}'. "
                f"Choose from {list(REPRESENTATION_CHANNELS)}")

    joint_rot = np.asarray(joint_rot, dtype=np.float64)
    if from_repr == to_repr:
        return joint_rot.copy()

    needs_euler = from_repr == "euler" or to_repr == "euler"
    if needs_euler and euler_orders is None:
        raise ValueError(
            "euler_orders is required when converting from/to 'euler'")

    # pybvh-ml carries rotmat as (F, J, 9); pybvh's rotations.convert
    # uses (F, J, 3, 3). Adapt at the boundary.
    if from_repr == "rotmat":
        F, J = joint_rot.shape[:2]
        joint_rot = joint_rot.reshape(F, J, 3, 3)

    result = rotations.convert(
        joint_rot, from_repr, to_repr, order=euler_orders)

    if to_repr == "rotmat":
        F, J = result.shape[:2]
        result = result.reshape(F, J, 9)

    return result
