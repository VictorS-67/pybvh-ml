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


_STREAM_BLOCKS = {
    "root_pos": "root_pos",
    "joint_rot": "joint_rotations",
    "joint_pos": "joint_positions",
    "node_pos": "node_positions",
}
"""Block name each packed stream appears under.

``joint_rot`` has been ``joint_rotations`` since before positions
existed, so the position blocks follow that spelling rather than the
field names.
"""


def describe_features(
    num_joints: int,
    representation: str = "6d",
    include_root_pos: bool = True,
    *,
    streams: tuple[str, ...] | list[str] | None = None,
    num_nodes: int | None = None,
) -> FeatureDescriptor:
    """Build a :class:`FeatureDescriptor` for a flat ``(T, D)`` layout.

    Describes the layout :func:`pybvh_ml.pack_to_flat` produces for the
    same ``streams``.  For a layout that also covers velocities and foot
    contacts (as written by ``pybvh.Bvh.to_feature_array``), use
    :meth:`pybvh.Bvh.feature_array_layout` instead — it returns a
    ``{block_name: slice}`` dict covering the full feature array.

    Parameters
    ----------
    num_joints : int
        Number of joints (excluding end sites).
    representation : str
        Rotation representation name.  One of ``"euler"``,
        ``"quat"``, ``"6d"``, ``"axisangle"``, ``"rotmat"``.  Read only
        when ``"joint_rot"`` is among the streams.
    include_root_pos : bool
        Whether root position occupies the first 3 columns.  The
        shorthand for the two default stream lists; pass ``streams``
        instead to describe anything else.
    streams : tuple of str, optional
        The ``streams=`` a :func:`~pybvh_ml.pack_to_flat` call was given
        — column order, same vocabulary.  ``None`` (default) derives it
        from ``include_root_pos``, which is the pre-0.6.0 behaviour.
    num_nodes : int, optional
        Node count ``N``, required when ``"node_pos"`` is among the
        streams.  Nodes are joints plus end sites, so it cannot be
        derived from ``num_joints``.

    Returns
    -------
    FeatureDescriptor
        Block names: ``root_pos``, ``joint_rotations``,
        ``joint_positions``, ``node_positions``.

    Examples
    --------
    >>> layout = describe_features(31, streams=("joint_pos", "joint_rot"))
    >>> layout.slice("joint_positions")
    slice(0, 93, None)
    """
    if streams is None:
        streams = ("root_pos", "joint_rot") if include_root_pos else (
            "joint_rot",)
    elif not include_root_pos:
        raise ValueError(
            "describe_features got both streams= and include_root_pos=False, "
            "which contradict each other. include_root_pos is the shorthand "
            "for the two default stream lists — name the streams you want "
            "and leave it alone.")

    unknown = [s for s in streams if s not in _STREAM_BLOCKS]
    if unknown:
        raise ValueError(
            f"describe_features got unknown stream name(s) {unknown}; "
            f"choose from {list(_STREAM_BLOCKS)}")
    if "joint_rot" in streams and representation not in REPR_CHANNELS:
        raise ValueError(
            f"Unknown representation '{representation}'. "
            f"Choose from {list(REPR_CHANNELS)}")
    if "node_pos" in streams and num_nodes is None:
        raise ValueError(
            "describe_features needs num_nodes= to size a 'node_pos' block: "
            "nodes are joints plus end sites, so N cannot be derived from "
            "num_joints. Pass skeleton_info['num_nodes'].")

    widths = {
        "root_pos": 3,
        "joint_rot": num_joints * REPR_CHANNELS.get(representation, 0),
        "joint_pos": num_joints * 3,
        "node_pos": (num_nodes or 0) * 3,
    }

    ranges: dict[str, tuple[int, int]] = {}
    col = 0
    for stream in streams:
        width = widths[stream]
        ranges[_STREAM_BLOCKS[stream]] = (col, col + width)
        col += width

    return FeatureDescriptor(ranges=ranges, total_dim=col)
