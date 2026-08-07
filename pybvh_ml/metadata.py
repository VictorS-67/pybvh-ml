"""Feature column descriptors for packed arrays.

Enables programmatic access to feature slices without hardcoded
column indices.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from pybvh.rotations import REPRESENTATION_CHANNELS as REPR_CHANNELS

from .packing import (
    _index_space,
    _split_root,
    _validate_stream_list,
    channel_count,
)


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
    "joint_vel": "joint_velocities",
    "joint_acc": "joint_accelerations",
    "node_vel": "node_velocities",
    "node_acc": "node_accelerations",
}
"""Block name each packed stream appears under.

``joint_rot`` has been ``joint_rotations`` since before positions
existed, so the position and derived blocks follow that spelling rather
than the field names.
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
        — column order, same vocabulary, including the derived
        ``"joint_vel"`` / ``"joint_acc"`` / ``"node_vel"`` /
        ``"node_acc"``.  ``None`` (default) derives it from
        ``include_root_pos``, which is the pre-0.6.0 behaviour.
    num_nodes : int, optional
        Node count ``N``, required when any node-space stream is among
        them (``"node_pos"``, ``"node_vel"``, ``"node_acc"``).  Nodes
        are joints plus end sites, so it cannot be derived from
        ``num_joints``.

    Returns
    -------
    FeatureDescriptor
        Block names: ``root_pos``, ``joint_rotations``,
        ``joint_positions``, ``node_positions``, ``joint_velocities``,
        ``joint_accelerations``, ``node_velocities``,
        ``node_accelerations``.

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
    # By index space, not by name: 'node_vel' needs N just as 'node_pos'
    # does, and a missed guard here yields a zero-width block rather
    # than an error.
    node_streams = [s for s in streams if _index_space(s) == "node"]
    if node_streams and num_nodes is None:
        raise ValueError(
            f"describe_features needs num_nodes= to size the "
            f"{node_streams} block(s): nodes are joints plus end sites, so "
            f"N cannot be derived from num_joints. Pass "
            f"skeleton_info['num_nodes'].")

    widths = {
        "root_pos": 3,
        "joint_rot": num_joints * REPR_CHANNELS.get(representation, 0),
        "joint_pos": num_joints * 3,
        "node_pos": (num_nodes or 0) * 3,
        "joint_vel": num_joints * 3,
        "joint_acc": num_joints * 3,
        "node_vel": (num_nodes or 0) * 3,
        "node_acc": (num_nodes or 0) * 3,
    }

    ranges: dict[str, tuple[int, int]] = {}
    col = 0
    for stream in streams:
        width = widths[stream]
        ranges[_STREAM_BLOCKS[stream]] = (col, col + width)
        col += width

    return FeatureDescriptor(ranges=ranges, total_dim=col)


def _stream_width(name: str, representation: str) -> int:
    """Channels a per-vertex stream contributes to ``C``.

    Everything on a vertex axis is 3-wide — positions and their
    temporal differences alike — except the rotations, whose width is
    the representation's.
    """
    if name == "joint_rot":
        return REPR_CHANNELS[representation]
    return 3


@dataclass(frozen=True)
class GraphDescriptor:
    """The ``(C, V)`` geometry of a graph-layout pack.

    What :func:`~pybvh_ml.pack_to_ctv` and
    :func:`~pybvh_ml.pack_to_tvc` will produce for a given ``streams``,
    without packing anything.

    Attributes
    ----------
    num_channels : int
        ``C``.
    num_vertices : int
        ``V``.
    packs_root : bool
        Whether vertex 0 is the root.  When True the root's position
        occupies channels ``0:3`` **of vertex 0 only**, and any channels
        beyond that on that vertex are zero padding; the per-vertex
        streams live on vertices ``1:``.
    first_vertex : int
        Index of the first per-vertex-stream vertex — ``1`` when the
        root is packed, ``0`` otherwise.  The offset to add to
        ``skeleton_info["edges"]`` to index packed vertices.
    channel_ranges : dict
        ``{block_name: (start, end)}`` along ``C``, for the per-vertex
        streams in packing order.  ``root_pos`` is deliberately **not**
        a key: it is a vertex, not a channel block, which is the rule
        most often lost when predicting these shapes by hand.
    """

    num_channels: int = 0
    num_vertices: int = 0
    packs_root: bool = False
    first_vertex: int = 0
    channel_ranges: dict[str, tuple[int, int]] = field(default_factory=dict)

    def __getitem__(self, key: str) -> tuple[int, int]:
        return self.channel_ranges[key]

    def slice(self, key: str) -> slice:
        """Return a :class:`slice` along ``C`` for the named block."""
        start, end = self.channel_ranges[key]
        return slice(start, end)

    def __contains__(self, key: str) -> bool:
        return key in self.channel_ranges

    def shape(self, num_frames: int, layout: str = "ctv") -> tuple[int, ...]:
        """The packed shape for *num_frames* frames in *layout*.

        ``"ctv"`` gives ``(C, T, V)``, ``"tvc"`` gives ``(T, V, C)`` —
        the same geometry, transposed.
        """
        if layout == "ctv":
            return (self.num_channels, num_frames, self.num_vertices)
        if layout == "tvc":
            return (num_frames, self.num_vertices, self.num_channels)
        raise ValueError(
            f"GraphDescriptor.shape got layout={layout!r}; the graph "
            f"layouts are 'ctv' and 'tvc'. For the flat layout use "
            f"describe_features, whose total_dim is its D.")


def describe_graph_features(
    num_joints: int,
    representation: str = "6d",
    *,
    streams: tuple[str, ...] | list[str] | None = None,
    num_nodes: int | None = None,
) -> GraphDescriptor:
    """Predict the ``(C, V)`` geometry of a graph-layout pack.

    The counterpart of :func:`describe_features`, which describes the
    flat ``(T, D)`` layout.  Answers "given these streams, what will
    :func:`~pybvh_ml.pack_to_ctv` produce?" without building an array,
    so a config check does not have to restate the packer's vertex and
    channel rules — the restatement being the thing that drifts.

    The rules it saves you reproducing: ``"root_pos"`` adds a **vertex**
    and never a channel block; ``C`` is the per-vertex streams' widths
    floored by the root's 3, so ``("root_pos",)`` alone is still
    ``C = 3``; derived streams are always 3 channels wide; and node
    space cannot share a vertex axis with joint space, which raises here
    exactly as it does in the packer.

    Parameters
    ----------
    num_joints : int
        Number of joints, excluding end sites — ``J``.
    representation : str
        Rotation representation, read only when ``"joint_rot"`` is among
        the streams.  One of ``"euler"``, ``"quat"``, ``"6d"``,
        ``"axisangle"``, ``"rotmat"``.
    streams : tuple of str, optional
        Same vocabulary and order as the packers'.  ``None`` (default)
        means ``("root_pos", "joint_rot")``, the packers' default.
    num_nodes : int, optional
        Node count ``N``, required when any node-space stream is named.

    Returns
    -------
    GraphDescriptor

    Raises
    ------
    ValueError
        For an unknown or repeated stream name, a node-space stream
        beside a joint-space one, an unknown representation, or a
        node-space stream without ``num_nodes`` — the same refusals,
        from the same code, as the packer's.

    Examples
    --------
    >>> desc = describe_graph_features(24, streams=("joint_pos", "joint_vel"))
    >>> desc.num_channels, desc.num_vertices
    (6, 24)
    >>> desc.shape(64)
    (6, 64, 24)
    >>> desc.slice("joint_velocities")
    slice(3, 6, None)

    See Also
    --------
    describe_features : The flat ``(T, D)`` layout's column ranges.
    pybvh_ml.pack_to_ctv : What this predicts.
    """
    if streams is None:
        streams = ("root_pos", "joint_rot")
    _validate_stream_list(streams, "describe_graph_features")
    if "joint_rot" in streams and representation not in REPR_CHANNELS:
        raise ValueError(
            f"Unknown representation '{representation}'. "
            f"Choose from {list(REPR_CHANNELS)}")
    node_streams = [s for s in streams if _index_space(s) == "node"]
    if node_streams and num_nodes is None:
        raise ValueError(
            f"describe_graph_features needs num_nodes= to size the "
            f"{node_streams} vertex axis: nodes are joints plus end "
            f"sites, so N cannot be derived from num_joints. Pass "
            f"skeleton_info['num_nodes'].")

    packs_root, vertex_streams = _split_root(
        streams, "describe_graph_features")

    widths = [_stream_width(name, representation) for name in vertex_streams]
    if not vertex_streams:
        vertex_count = 0
    elif node_streams:
        vertex_count = num_nodes
    else:
        vertex_count = num_joints

    ranges: dict[str, tuple[int, int]] = {}
    channel = 0
    for name, width in zip(vertex_streams, widths):
        ranges[_STREAM_BLOCKS[name]] = (channel, channel + width)
        channel += width

    return GraphDescriptor(
        num_channels=channel_count(packs_root, widths),
        num_vertices=vertex_count + int(packs_root),
        packs_root=packs_root,
        first_vertex=int(packs_root),
        channel_ranges=ranges)
