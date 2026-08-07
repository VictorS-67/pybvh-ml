"""Tensor layout conversion for ML pipelines.

Converts between :class:`~pybvh_ml.MotionArrays` and the tensor
layouts that ML models consume: ``(C, T, V)``, ``(T, V, C)``, and flat
``(T, D)``.

Conventions
-----------
- **C** = channels (the packed streams' channels, concatenated)
- **T** = time / frames
- **V** = vertices (root is vertex 0 when packed, then the per-vertex
  stream's ``J`` joints or ``N`` nodes)
- **D** = flat feature dimension (the same channels, per frame)

``streams=`` names what is packed and in what order — channel order in
the graph layouts, column order in flat.  It defaults to
``("root_pos", "joint_rot")``, which is the layout every pybvh-ml
version has produced.

The root vertex carries 3 position channels; when C > 3 (e.g. quat,
6D, or rotmat joint data), the root vertex's remaining ``C - 3``
channels are zero padding — the position values themselves are
unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .arrays import (
    MotionArrays,
    center_root_streams,
    require_position_centering,
)


DEFAULT_STREAMS: tuple[str, ...] = ("root_pos", "joint_rot")
"""What the packers pack when nothing else is asked for.

The 0.5.0 layout, byte for byte: root as vertex 0, joints as vertices
``1..J``, rotation channels first.
"""

DERIVED_STREAMS: dict[str, tuple[str, int]] = {
    "joint_vel": ("joint_pos", 1),
    "joint_acc": ("joint_pos", 2),
    "node_vel": ("node_pos", 1),
    "node_acc": ("node_pos", 2),
}
"""Packable streams computed from a base stream, as ``(base, order)``.

Temporal finite differences of the position streams: ``joint_vel`` is
the first difference of ``joint_pos``, ``joint_acc`` the second.  They
are *not* :class:`~pybvh_ml.MotionArrays` fields and cannot be
augmented — see :func:`pack_to_ctv` for why they are derived at packing
time instead of carried.
"""

# Per-vertex streams, in the order their vertex counts are read from.
# ``root_pos`` is not here: it contributes a vertex, not a vertex axis.
_VERTEX_STREAMS = ("joint_rot", "joint_pos", "node_pos")

_BASE_STREAMS = ("root_pos", *_VERTEX_STREAMS)

_INDEX_SPACES = {
    "root_pos": "root",
    "joint_rot": "joint",
    "joint_pos": "joint",
    "node_pos": "node",
}

STREAM_VOCABULARY: tuple[str, ...] = (*_BASE_STREAMS, *DERIVED_STREAMS)
"""Every name ``streams=`` accepts, base streams first."""


def _base_of(name: str) -> str:
    """The stream a name is computed from; itself, for a base stream."""
    base, _ = DERIVED_STREAMS.get(name, (name, 0))
    return base


def _index_space(name: str) -> str:
    """``"root"``, ``"joint"`` or ``"node"`` — which vertex axis a name lives on."""
    return _INDEX_SPACES[_base_of(name)]


def _derive(base: npt.NDArray[np.floating], order: int) -> npt.NDArray[np.float64]:
    """The *order*-th temporal difference of *base*, same shape.

    Boundary rule: the first *order* frames are zeroed, because no
    earlier frame exists to difference against.  Prepending instead
    would leave frame ``order - 1`` holding a lower-order difference —
    a velocity sitting in an acceleration channel — so every nonzero
    value here is a genuine *order*-th difference.

    Total for degenerate lengths: an empty clip stays empty, and a clip
    shorter than *order* comes back all zeros, which is the correct
    statement that no frame has *order* predecessors.
    """
    out = np.asarray(base, dtype=np.float64)
    for _ in range(order):
        out = np.diff(out, axis=0, prepend=out[:1])
    out = out.copy()
    out[:order] = 0.0
    return out


def _resolve_streams(
    arrays: MotionArrays,
    streams: tuple[str, ...] | list[str],
    caller: str,
) -> tuple[list[str], bool, int, list[int]]:
    """Validate a ``streams=`` request against the clip it will pack.

    Returns ``(vertex_streams, pack_root, num_vertices, channel_widths)``
    — everything the three layouts need, computed once so they cannot
    disagree about a shape.
    """
    streams = list(streams)
    if not streams:
        raise ValueError(
            f"{caller} was given an empty streams=(); name at least one of "
            f"{list(DEFAULT_STREAMS)} or a position stream.")
    _validate_stream_list(streams, caller)

    missing = [s for s in streams if getattr(arrays, _base_of(s)) is None]
    if missing:
        needed = sorted({_base_of(s) for s in missing})
        raise ValueError(
            f"{caller} was asked to pack {missing}, which needs {needed} — "
            f"this MotionArrays does not carry that (it has "
            f"{sorted(arrays.present_streams)}). "
            f"Extract them with MotionArrays.from_bvh(..., "
            f"include_positions=True), or preprocess with "
            f"include_positions=True.")

    pack_root, vertex_streams = _split_root(streams, caller)

    # Derived streams inherit their base's vertex axis and are always
    # 3 channels wide — they are differences of positions.
    vertex_counts = {getattr(arrays, _base_of(s)).shape[1]
                     for s in vertex_streams}
    num_vertices = (vertex_counts.pop() if vertex_counts else 0) + int(pack_root)
    channel_widths = [3 if s in DERIVED_STREAMS
                      else getattr(arrays, s).shape[2]
                      for s in vertex_streams]
    return vertex_streams, pack_root, num_vertices, channel_widths


def _validate_stream_list(
    streams: list[str] | tuple[str, ...],
    caller: str,
) -> None:
    """Vocabulary and duplicate checks, shared with the descriptors.

    Split out of :func:`_resolve_streams` so that
    :func:`~pybvh_ml.describe_graph_features` enforces the same rules
    from the same code rather than restating them against joint counts
    — the restatement is exactly what a consumer predicting shapes
    downstream ends up maintaining, and what these helpers exist to
    delete.
    """
    streams = list(streams)
    unknown = [s for s in streams if s not in STREAM_VOCABULARY]
    if unknown:
        raise ValueError(
            f"{caller} got unknown stream name(s) {unknown}; choose from "
            f"{list(STREAM_VOCABULARY)}")
    duplicated = [s for s in set(streams) if streams.count(s) > 1]
    if duplicated:
        raise ValueError(
            f"{caller} got repeated stream(s) {sorted(duplicated)} in "
            f"streams={tuple(streams)}; each stream is packed once.")


def _split_root(
    streams: list[str] | tuple[str, ...],
    caller: str,
) -> tuple[bool, list[str]]:
    """Separate the root from the per-vertex streams, enforcing index space.

    The rule a shape-predicting consumer has to reproduce otherwise:
    ``"root_pos"`` contributes a *vertex*, never a channel block, and
    node space cannot share a vertex axis with joint space.
    """
    pack_root = "root_pos" in streams
    vertex_streams = [s for s in streams if s != "root_pos"]
    node_streams = [s for s in vertex_streams if _index_space(s) == "node"]
    if node_streams and len(vertex_streams) > len(node_streams):
        others = [s for s in vertex_streams if _index_space(s) != "node"]
        named = ", ".join(repr(s) for s in node_streams)
        raise ValueError(
            f"{caller} cannot combine {named} with {others} on one "
            f"vertex axis: node space includes end sites, so it has N "
            f"vertices where joint space has J. Pack them separately, or "
            f"slice the node array down to joint space with "
            f"skeleton_info['fk_topology']['joint_idx'] >= 0.")
    return pack_root, vertex_streams


def channel_count(pack_root: bool, channel_widths: list[int]) -> int:
    """``C`` for a graph-layout pack: the widths, floored by the root's 3.

    The floor is the rule that is easy to miss when predicting shapes by
    hand — a pack whose per-vertex streams total fewer than 3 channels
    still needs 3, because vertex 0 carries a position.  With no
    per-vertex stream at all (``streams=("root_pos",)``) it is the only
    thing setting ``C``.
    """
    return max(3 if pack_root else 0, sum(channel_widths))


@dataclass(frozen=True)
class _MaterializedStreams:
    """A resolved, derived, ready-to-pack stream set.

    The intermediate between "a container plus a ``streams=`` request"
    and "a packed tensor".  It exists because derivation and packing
    must be able to happen at *different* times: the Dataset classes
    materialize before temporal standardization (so a difference is
    taken across consecutive frames of the augmented clip, never across
    resampled ones), then pack the result — while a standalone packer
    call does both back to back.

    Carries the resolved layout alongside the arrays so that resolution
    happens exactly once; recomputing it at pack time would be a second
    chance to disagree about a shape.
    """

    streams: tuple[str, ...]
    arrays: dict[str, npt.NDArray[np.float64]]
    position_centering: str | None
    pack_root: bool
    vertex_streams: tuple[str, ...]
    num_vertices: int
    channel_widths: tuple[int, ...]

    def __post_init__(self) -> None:
        counts = {value.shape[0] for value in self.arrays.values()}
        if len(counts) > 1:
            raise ValueError(
                f"_MaterializedStreams got streams of differing frame "
                f"counts: "
                f"{ {name: value.shape[0] for name, value in self.arrays.items()} }. "
                f"Every stream must be standardized by the same index "
                f"vector.")

    @property
    def frame_count(self) -> int:
        return next(iter(self.arrays.values())).shape[0] if self.arrays else 0

    def replace_arrays(
        self,
        arrays: dict[str, npt.NDArray[np.float64]],
    ) -> "_MaterializedStreams":
        """A copy with new arrays and the same layout.

        For temporal standardization, which changes frame counts and
        nothing else.
        """
        return _MaterializedStreams(
            streams=self.streams, arrays=arrays,
            position_centering=self.position_centering,
            pack_root=self.pack_root, vertex_streams=self.vertex_streams,
            num_vertices=self.num_vertices,
            channel_widths=self.channel_widths)


def _materialize_streams(
    arrays: MotionArrays,
    streams: tuple[str, ...] | list[str],
    center_root: bool,
    caller: str,
) -> _MaterializedStreams:
    """Validate a ``streams=`` request, centre, and compute derived streams.

    The single place derivation happens.  Everything downstream only
    *consumes* the result — differencing an already-differenced stream
    would silently turn a velocity into an acceleration.
    """
    vertex_streams, pack_root, num_vertices, widths = _resolve_streams(
        arrays, streams, caller)
    if any(s in DERIVED_STREAMS for s in streams):
        # A difference of "world" positions is world velocity; of
        # "skeleton" positions, velocity relative to the root.  Same
        # shape, different quantity — so an undeclared frame is refused
        # rather than guessed, exactly as center_root refuses it.
        require_position_centering(arrays, caller)
    arrays = _centered_arrays(arrays, center_root, caller)

    materialized: dict[str, npt.NDArray[np.float64]] = {}
    for name in streams:
        if name in DERIVED_STREAMS:
            base, order = DERIVED_STREAMS[name]
            materialized[name] = _derive(getattr(arrays, base), order)
        else:
            materialized[name] = np.asarray(
                getattr(arrays, name), dtype=np.float64)

    return _MaterializedStreams(
        streams=tuple(streams), arrays=materialized,
        position_centering=arrays.position_centering,
        pack_root=pack_root, vertex_streams=tuple(vertex_streams),
        num_vertices=num_vertices, channel_widths=tuple(widths))


def _centered_arrays(
    arrays: MotionArrays,
    center_root: bool,
    caller: str,
) -> MotionArrays:
    """Apply ``center_root`` to every stream it affects.

    Centering only ``root_pos`` would shift vertex 0 and leave the
    position vertices where they were — the same inconsistency
    :func:`~pybvh_ml.add_root_position_noise` guards against, produced
    at pack time instead.  So the shift reaches the positions too under
    ``"world"`` / ``"first"``, leaves them alone under ``"skeleton"``
    (they are already root-relative), and raises when the frame is
    undeclared.  See :func:`~pybvh_ml.arrays.center_root_streams` for
    why ``"first"`` takes the same branch as ``"world"`` despite putting
    the positions in a different frame from ``root_pos``.
    """
    if not center_root or arrays.frame_count == 0:
        return arrays
    if arrays.joint_pos is not None or arrays.node_pos is not None:
        require_position_centering(arrays, caller)
    root_pos, joint_pos, node_pos = center_root_streams(
        np.asarray(arrays.root_pos, dtype=np.float64),
        None if arrays.joint_pos is None
        else np.asarray(arrays.joint_pos, dtype=np.float64),
        None if arrays.node_pos is None
        else np.asarray(arrays.node_pos, dtype=np.float64),
        arrays.position_centering, caller)
    return arrays.replace(
        root_pos=root_pos, joint_pos=joint_pos, node_pos=node_pos)


def _pack_tvc_materialized(
    materialized: _MaterializedStreams,
) -> npt.NDArray[np.float64]:
    """Build the ``(T, V, C)`` block both graph layouts are cut from.

    Consumes only — never derives.  See :func:`_materialize_streams`.
    """
    F = materialized.frame_count
    widths = materialized.channel_widths
    C = channel_count(materialized.pack_root, list(widths))
    tvc = np.zeros((F, materialized.num_vertices, C), dtype=np.float64)
    if materialized.pack_root:
        tvc[:, 0, :3] = materialized.arrays["root_pos"]

    channel = 0
    first_vertex = int(materialized.pack_root)
    for name, width in zip(materialized.vertex_streams, widths):
        tvc[:, first_vertex:, channel:channel + width] = (
            materialized.arrays[name])
        channel += width
    return tvc


def _pack_ctv_materialized(
    materialized: _MaterializedStreams,
) -> npt.NDArray[np.float64]:
    """``(C, T, V)`` from a materialized stream set.  Consumes only."""
    tvc = _pack_tvc_materialized(materialized)
    # Materialize the transpose: consumers hand this to
    # torch.from_numpy(...).view(...) and C-contiguity assumptions.
    return np.ascontiguousarray(tvc.transpose(2, 0, 1))


def _pack_flat_materialized(
    materialized: _MaterializedStreams,
) -> npt.NDArray[np.float64]:
    """``(T, D)`` from a materialized stream set.  Consumes only."""
    F = materialized.frame_count
    blocks = []
    for name in materialized.streams:
        value = materialized.arrays[name]
        blocks.append(value if name == "root_pos" else value.reshape(F, -1))
    return np.concatenate(blocks, axis=1)


def pack_to_ctv(
    arrays: MotionArrays,
    center_root: bool = True,
    *,
    streams: tuple[str, ...] | list[str] = DEFAULT_STREAMS,
) -> npt.NDArray[np.float64]:
    """Pack the named streams into ``(C, T, V)`` layout.

    The ST-GCN layout.  ``streams=("joint_pos",)`` is the canonical
    skeleton-action-recognition input — ``(3, T, J)``, with the joint
    axis indexed directly by ``skeleton_info["edges"]``.

    Parameters
    ----------
    arrays : MotionArrays
        Must carry every stream named in *streams*.
    center_root : bool
        If True, subtract first frame's root position.
        This flag is for standalone packing of raw extractions.  Clips from a dataset preprocessed with ``center_root=True`` (see :func:`~pybvh_ml.preprocessing.preprocess_directory` and the ``center_root`` key of :func:`~pybvh_ml.preprocessing.load_preprocessed`) are already centered — pass ``False`` for those.  Re-centering a whole already-centered clip is a harmless no-op, but re-centering a *windowed sub-clip* zeroes the window's first frame and destroys the clip-relative trajectory.

        It reaches the position vertices too, which is what keeps them
        in the same frame as vertex 0 — see the Notes.
    streams : tuple of str
        What to pack, and in what order.  ``"root_pos"`` adds the root
        as vertex 0 (``V = 1 + J``); omit it and ``V = J``.  The
        per-vertex streams are concatenated along ``C`` in the order
        given.  Default ``("root_pos", "joint_rot")``.

        Besides the four :class:`~pybvh_ml.MotionArrays` streams, the
        **derived** names ``"joint_vel"`` / ``"joint_acc"`` and
        ``"node_vel"`` / ``"node_acc"`` are accepted — temporal
        differences of the matching position stream, computed here
        rather than carried.  Requesting one needs its base stream
        present but does not require packing it: ``streams=("joint_vel",)``
        alone is the 2s-AGCN motion stream.  See the Notes for the unit,
        boundary and ordering conventions they follow.

    Returns
    -------
    ndarray, shape (C, T, V)
        ``T = F``.  ``V`` is the per-vertex stream's vertex count plus
        one when ``"root_pos"`` is packed; ``C`` is the sum of the
        per-vertex streams' channel counts, floored at 3 when the root
        is packed.  Root is vertex 0: its position fills channels
        ``0:3``, and any channels beyond that are zero padding.

    Raises
    ------
    ValueError
        If a named stream is absent from *arrays*; if ``"node_pos"`` is
        combined with a joint-space stream (different ``V``); or if
        ``center_root=True`` meets positions whose
        ``position_centering`` is ``None``.

    Notes
    -----
    Common combinations:

    ===================================== ================= =================
    ``streams``                           shape             note
    ===================================== ================= =================
    ``("root_pos", "joint_rot")``         ``(max(3, C), T, 1+J)`` the default
    ``("joint_pos",)``                    ``(3, T, J)``     ST-GCN input
    ``("node_pos",)``                     ``(3, T, N)``     full skeleton
    ``("joint_pos", "joint_rot")``        ``(3+C, T, J)``   multi-stream
    ``("root_pos", "joint_pos")``         ``(3, T, 1+J)``   vertex 0
                                                            duplicates joint
                                                            0 under ``"world"``
    ``("joint_vel",)``                    ``(3, T, J)``     motion stream
    ``("joint_pos", "joint_vel",
    "joint_acc")``                        ``(9, T, J)``     pos/vel/acc
    ===================================== ================= =================

    **Vertex alignment.** With ``V = J`` (``("joint_pos",)``) the
    ``edges`` / ``lr_pairs`` from ``skeleton_info`` index packed
    vertices one-to-one, and likewise ``node_edges`` / ``node_lr_pairs``
    with ``("node_pos",)``.  Any packing that includes ``"root_pos"``
    shifts every joint by one, so those lists need ``+1`` on both
    indices.

    **Centering.** Under ``position_centering="world"`` or ``"first"``
    the ``center_root`` shift is applied identically to the root and to
    every position vertex; under ``"skeleton"`` the positions are
    already root-relative and are left alone.  Centering only the root
    would move vertex 0 away from a body that stayed put.  It cannot
    change a derived stream at all: ``center_root`` subtracts the *first
    frame's* root, a shift constant in time, which cancels in every
    difference.

    **Derived streams — units.** ``joint_vel`` is a raw per-frame
    difference, ``p[t] - p[t-1]``, with no ``dt``: the convention of
    2s-AGCN, CTR-GCN and PYSKL, and the one that needs no frame rate
    (which :class:`~pybvh_ml.MotionArrays` does not carry).  The
    alternative is physical units — divide by ``frame_time``, or
    equivalently multiply these values by the clip's fps — which no
    parameter here does for you.

    **Derived streams — boundary.** An order-*k* stream has its first
    *k* frames zeroed, so `joint_vel[0]` and ``joint_acc[0:2]`` are 0.
    Prepending the first frame instead would leave ``joint_acc[1]``
    holding ``joint_vel[1]`` — a velocity in an acceleration channel.
    The alternative is central differences, unbiased in time but
    non-causal, and therefore a label leak for an autoregressive model.

    **Derived streams — meaning depends on the centering.** A difference
    of ``"world"`` positions is world-frame velocity including the root
    trajectory; of ``"skeleton"`` positions, velocity relative to the
    root — HumanML3D's ``local_velocity``.  Same shape, different
    quantity, so an undeclared ``position_centering`` is refused rather
    than guessed.

    **Derived streams — derive before any temporal standardization.**
    These are differences of *consecutive* frames.  Padding, cropping
    or index-sampling a clip and only then packing a derived stream
    differences whatever frames survived: a pad boundary produces a
    phantom spike, and ``uniform_temporal_sample`` produces differences
    scaled by a random gap.  Both Dataset classes order this correctly
    for you — they materialize the derived streams before standardizing
    the length.  Doing your own temporal standardization and then
    calling a packer does not, which is why the recommendation is to let
    the Dataset do it.

    A consequence worth knowing before you check it: under
    ``temporal="resample"`` a packed derived stream is **not** the
    difference of the packed positions, and is not meant to be.  It is
    the true per-frame difference *observed at the sampled instants*,
    subsampled alongside the positions rather than recomputed from
    them; the packed positions' own difference is inflated by the mean
    sampling gap (and by its square, one order up).  Under ``"pad"`` and
    ``"crop"`` the two agree on frames ``k..length-1``.

    See Also
    --------
    pack_to_tvc, pack_to_flat : The same streams in the other layouts.
    unpack_from_ctv : The inverse for the default streams (see its Notes
        for why it does not take ``streams=``).
    """
    return _pack_ctv_materialized(_materialize_streams(
        arrays, streams, center_root, "pack_to_ctv"))


def pack_to_tvc(
    arrays: MotionArrays,
    center_root: bool = True,
    *,
    streams: tuple[str, ...] | list[str] = DEFAULT_STREAMS,
) -> npt.NDArray[np.float64]:
    """Pack the named streams into ``(T, V, C)`` layout.

    Parameters
    ----------
    arrays : MotionArrays
        Must carry every stream named in *streams*.
    center_root : bool
        If True, subtract first frame's root position (from the position
        vertices too).  Arrays from a preprocessed dataset saved with ``center_root=True`` are already centered — see :func:`pack_to_ctv`.
    streams : tuple of str
        What to pack, and in what order — the same vocabulary as
        :func:`pack_to_ctv`, including the derived ``"joint_vel"`` /
        ``"joint_acc"`` / ``"node_vel"`` / ``"node_acc"``; its Notes
        carry the unit, boundary and ordering conventions derived
        streams follow.

    Returns
    -------
    ndarray, shape (T, V, C)
        The transpose of what :func:`pack_to_ctv` returns; the same
        shape rules apply.
    """
    return _pack_tvc_materialized(_materialize_streams(
        arrays, streams, center_root, "pack_to_tvc"))


def pack_to_flat(
    arrays: MotionArrays,
    center_root: bool = True,
    *,
    streams: tuple[str, ...] | list[str] = DEFAULT_STREAMS,
) -> npt.NDArray[np.float64]:
    """Pack the named streams into flat ``(T, D)`` layout.

    Parameters
    ----------
    arrays : MotionArrays
        Must carry every stream named in *streams*.
    center_root : bool
        If True, subtract first frame's root position (from the position
        vertices too).  Arrays from a preprocessed dataset saved with ``center_root=True`` are already centered — see :func:`pack_to_ctv`.
    streams : tuple of str
        What to pack, and in what order — here that is **column** order.
        Default ``("root_pos", "joint_rot")``, which is the
        ``[root_pos (3), joint_rot flattened over (J, C)]`` layout
        :func:`~pybvh_ml.describe_features` and the ``mean`` / ``std``
        vectors of a preprocessed dataset are written against.
        The same vocabulary as :func:`pack_to_ctv` applies, including
        the derived ``"joint_vel"`` / ``"joint_acc"`` / ``"node_vel"`` /
        ``"node_acc"`` — see its Notes for the unit, boundary and
        ordering conventions, including why to derive before any
        temporal standardization you run yourself.

    Returns
    -------
    ndarray, shape (T, D)
        ``D`` is 3 for a packed root plus ``V * C`` per per-vertex
        stream.  For the default streams that is ``3 + J * C_rot``,
        with root position in columns ``0:3``.

    Notes
    -----
    The default layout is a public contract — ``pack_to_flat``,
    :func:`~pybvh_ml.describe_features`, the stored normalization
    vectors and HumanML3D's ``Mean.npy`` / ``Std.npy`` all agree on it —
    which is why positions get their own stats block in a preprocessed
    dataset rather than widening ``mean`` / ``std``.  A ``D`` that
    changed with a preprocessing flag would make one file format mean
    two things.
    """
    return _pack_flat_materialized(_materialize_streams(
        arrays, streams, center_root, "pack_to_flat"))


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

    Notes
    -----
    **The unpackers take no ``streams=``**, so they invert only the
    default ``("root_pos", "joint_rot")`` packing: everything past the
    root's channels comes back as ``joint_rot``.  Round-tripping a
    multi-stream pack needs a streams-aware unpacker, which is purely
    additive whenever it lands; the asymmetry is a deliberate deferral
    rather than an oversight.  Until then, unpack a multi-stream tensor
    by slicing the channel axis yourself — the widths are the ones
    :func:`pack_to_ctv` documents.
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
