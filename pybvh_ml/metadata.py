"""Feature column descriptors for packed arrays.

Enables programmatic access to feature slices without hardcoded
column indices.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from pybvh.rotations import REPRESENTATION_CHANNELS as REPR_CHANNELS


@dataclass
class FeatureDescriptor:
    """Maps feature names to ``(start_col, end_col)`` ranges.

    Attributes
    ----------
    ranges : dict
        ``{feature_name: (start, end)}`` column index ranges.
    total_dim : int
        Total number of feature columns.
    """

    ranges: dict[str, tuple[int, int]] = field(default_factory=dict)
    total_dim: int = 0

    def __getitem__(self, key: str) -> tuple[int, int]:
        return self.ranges[key]

    def slice(self, key: str) -> slice:
        """Return a :class:`slice` for the named feature."""
        start, end = self.ranges[key]
        return slice(start, end)

    def __contains__(self, key: str) -> bool:
        return key in self.ranges


def describe_features(
    num_joints: int,
    representation: str = "6d",
    include_root_pos: bool = True,
) -> FeatureDescriptor:
    """Build a :class:`FeatureDescriptor` for a flat ``(T, D)`` layout.

    Describes the simple ``root_pos + joint_rotations`` layout produced
    by :func:`pybvh_ml.pack_to_flat`.  For a layout that also covers
    velocities and foot contacts (as written by
    ``pybvh.Bvh.to_feature_array``), use
    :meth:`pybvh.Bvh.feature_array_layout` instead — it returns a
    ``{block_name: slice}`` dict covering the full feature array.

    Parameters
    ----------
    num_joints : int
        Number of joints (excluding end sites).
    representation : str
        Rotation representation name.  One of ``"euler"``,
        ``"quaternion"``, ``"6d"``, ``"axisangle"``, ``"rotmat"``.
    include_root_pos : bool
        Whether root position occupies the first 3 columns.

    Returns
    -------
    FeatureDescriptor
    """
    if representation not in REPR_CHANNELS:
        raise ValueError(
            f"Unknown representation '{representation}'. "
            f"Choose from {list(REPR_CHANNELS)}")

    c = REPR_CHANNELS[representation]
    ranges: dict[str, tuple[int, int]] = {}
    col = 0

    if include_root_pos:
        ranges["root_pos"] = (col, col + 3)
        col += 3

    ranges["joint_rotations"] = (col, col + num_joints * c)
    col += num_joints * c

    return FeatureDescriptor(ranges=ranges, total_dim=col)
