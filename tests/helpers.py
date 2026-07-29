"""Small adapters shared by the test modules.

The augmentation surface returns :class:`~pybvh_ml.MotionArrays`; most
assertions here were written against the two arrays it carries, so these
two helpers destructure a result without every test restating it.  They
adapt the *result* only — every call still goes through the real public
signature.
"""
from __future__ import annotations


def as_pair(result):
    """``(root_pos, joint_rot)`` from a ``MotionArrays``."""
    return result.root_pos, result.joint_rot


def as_triple(result):
    """``(root_pos, joint_rot, params)`` from a ``(MotionArrays, params)``."""
    arrays, params = result
    return arrays.root_pos, arrays.joint_rot, params
