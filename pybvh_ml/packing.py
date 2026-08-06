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

# Per-vertex streams, in the order their vertex counts are read from.
# ``root_pos`` is not here: it contributes a vertex, not a vertex axis.
_VERTEX_STREAMS = ("joint_rot", "joint_pos", "node_pos")


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
    unknown = [s for s in streams if s not in ("root_pos", *_VERTEX_STREAMS)]
    if unknown:
        raise ValueError(
            f"{caller} got unknown stream name(s) {unknown}; choose from "
            f"['root_pos', 'joint_rot', 'joint_pos', 'node_pos']")
    duplicated = [s for s in set(streams) if streams.count(s) > 1]
    if duplicated:
        raise ValueError(
            f"{caller} got repeated stream(s) {sorted(duplicated)} in "
            f"streams={tuple(streams)}; each stream is packed once.")

    missing = [s for s in streams if getattr(arrays, s) is None]
    if missing:
        raise ValueError(
            f"{caller} was asked to pack {missing}, which this MotionArrays "
            f"does not carry (it has {sorted(arrays.present_streams)}). "
            f"Extract them with MotionArrays.from_bvh(..., "
            f"include_positions=True), or preprocess with "
            f"include_positions=True.")

    pack_root = "root_pos" in streams
    vertex_streams = [s for s in streams if s != "root_pos"]
    if "node_pos" in vertex_streams and len(vertex_streams) > 1:
        others = [s for s in vertex_streams if s != "node_pos"]
        raise ValueError(
            f"{caller} cannot combine 'node_pos' with {others} on one vertex "
            f"axis: node space includes end sites, so it has N vertices "
            f"where joint space has J. Pack them separately, or slice the "
            f"node array down to joint space with "
            f"skeleton_info['fk_topology']['joint_idx'] >= 0.")

    vertex_counts = {getattr(arrays, s).shape[1] for s in vertex_streams}
    num_vertices = (vertex_counts.pop() if vertex_counts else 0) + int(pack_root)
    channel_widths = [getattr(arrays, s).shape[2] for s in vertex_streams]
    return vertex_streams, pack_root, num_vertices, channel_widths


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


def _pack_tvc(
    arrays: MotionArrays,
    streams: tuple[str, ...] | list[str],
    center_root: bool,
    caller: str,
) -> npt.NDArray[np.float64]:
    """Build the ``(T, V, C)`` block both graph layouts are cut from."""
    vertex_streams, pack_root, V, widths = _resolve_streams(
        arrays, streams, caller)
    arrays = _centered_arrays(arrays, center_root, caller)

    F = arrays.frame_count
    C = max(3 if pack_root else 0, sum(widths))
    tvc = np.zeros((F, V, C), dtype=np.float64)
    if pack_root:
        tvc[:, 0, :3] = np.asarray(arrays.root_pos, dtype=np.float64)

    channel = 0
    first_vertex = int(pack_root)
    for name, width in zip(vertex_streams, widths):
        tvc[:, first_vertex:, channel:channel + width] = np.asarray(
            getattr(arrays, name), dtype=np.float64)
        channel += width
    return tvc


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

    ============================== ================= ==========================
    ``streams``                    shape             note
    ============================== ================= ==========================
    ``("root_pos", "joint_rot")``  ``(max(3, C), T, 1+J)`` the default
    ``("joint_pos",)``             ``(3, T, J)``     ST-GCN / CTR-GCN input
    ``("node_pos",)``              ``(3, T, N)``     full visual skeleton
    ``("joint_pos", "joint_rot")`` ``(3+C, T, J)``   multi-stream on ``C``
    ``("root_pos", "joint_pos")``  ``(3, T, 1+J)``   vertex 0 duplicates
                                                     joint 0 under ``"world"``
    ============================== ================= ==========================

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
    would move vertex 0 away from a body that stayed put.

    See Also
    --------
    pack_to_tvc, pack_to_flat : The same streams in the other layouts.
    unpack_from_ctv : The inverse for the default streams (see its Notes
        for why it does not take ``streams=``).
    """
    tvc = _pack_tvc(arrays, streams, center_root, "pack_to_ctv")
    # Materialize the transpose: consumers hand this to
    # torch.from_numpy(...).view(...) and C-contiguity assumptions.
    return np.ascontiguousarray(tvc.transpose(2, 0, 1))


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
        What to pack, and in what order — see :func:`pack_to_ctv`.

    Returns
    -------
    ndarray, shape (T, V, C)
        The transpose of what :func:`pack_to_ctv` returns; the same
        shape rules apply.
    """
    return _pack_tvc(arrays, streams, center_root, "pack_to_tvc")


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
    # Flat needs no vertex/channel geometry — only the same validation.
    _resolve_streams(arrays, streams, "pack_to_flat")
    arrays = _centered_arrays(arrays, center_root, "pack_to_flat")

    F = arrays.frame_count
    blocks = []
    for name in streams:
        value = np.asarray(getattr(arrays, name), dtype=np.float64)
        blocks.append(value if name == "root_pos" else value.reshape(F, -1))
    return np.concatenate(blocks, axis=1)


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
