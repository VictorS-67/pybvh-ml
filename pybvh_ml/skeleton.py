"""Skeleton graph metadata for GCN and graph-based models.

Provides edge lists, left/right pairs in both index spaces, forward-
kinematics topology, and unified skeleton descriptors — the topology
data that GCN and Transformer models consume.  Only uses pybvh's public
API.

**Two index spaces, and they are not interchangeable.** *Joint* space
(``bvh.joint_angles`` order, ``J`` entries) excludes end sites; *node*
space (``bvh.nodes`` order, ``N >= J`` entries) includes them.  Node
indices diverge from joint indices as soon as any end site precedes a
paired joint in file order, so an index list from one space silently
addresses the wrong vertices in the other.  Every key here says which
space it is in, and ``fk_topology["joint_idx"]`` is the map between
them: ``joint_idx[n] >= 0`` marks node ``n`` as a joint and gives its
joint column.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from pybvh import Bvh, FkTopology


def get_edge_list(
    bvh: Bvh,
    include_end_sites: bool = False,
) -> list[tuple[int, int]]:
    """Get skeleton edge list as ``(child_idx, parent_idx)`` tuples.

    Thin re-export of pybvh's edge-list properties.

    Parameters
    ----------
    bvh : Bvh
    include_end_sites : bool
        If False (default), use ``joint_angles`` index space
        (non-end-site joints only) — returns ``bvh.edges``.
        If True, use ``node_index`` space (all nodes including
        end sites) — returns ``bvh.node_edges``.

    Returns
    -------
    list of (int, int)

    Notes
    -----
    **Which packing this indexes directly.** Joint-space edges index the
    vertices of ``pack_to_*(streams=("joint_pos",))``, where ``V = J``,
    one-to-one; node-space edges do the same for
    ``streams=("node_pos",)``, where ``V = N``.  Any packing that
    includes ``"root_pos"`` — the default ``("root_pos", "joint_rot")``
    among them — puts the root at vertex 0 and shifts every joint by
    one, so those edges need ``(child + 1, parent + 1)``.
    """
    return list(bvh.node_edges if include_end_sites else bvh.edges)


def get_lr_pairs(bvh: Bvh) -> list[tuple[int, int]]:
    """Detect left/right joint pairs as index tuples.

    Returns ``list(bvh.lr_pairs)`` — the cached, auto-detected
    index-space pair list from pybvh.  An empty list means no pairs
    were detected on this skeleton.

    Returns
    -------
    list of (int, int)
        ``[(left_idx, right_idx), ...]`` in ``joint_angles`` index
        space.  Empty if no pairs found.
    """
    return list(bvh.lr_pairs) if bvh.lr_pairs else []


def get_node_lr_pairs(bvh: Bvh) -> list[tuple[int, int]]:
    """Detect left/right **node** pairs as index tuples.

    Node-space counterpart of :func:`get_lr_pairs`, covering joints and
    their end sites — what :func:`~pybvh_ml.mirror` needs to swap every
    paired vertex of a ``node_pos`` stream, fingertips and toe tips
    included.

    Returns
    -------
    list of (int, int)
        ``[(left_idx, right_idx), ...]`` in ``bvh.nodes`` index space.
        Empty if no pairs found.

    Notes
    -----
    A joint pair whose two sides carry **different numbers of end sites**
    is returned with its end sites dropped: pybvh has no well-defined tip
    correspondence there, and its property filters rather than raises,
    matching ``lr_pairs``.  That silently produces a half-swapped
    skeleton at mirror time, which is why
    :func:`get_skeleton_info` records the offending pairs under
    ``mismatched_end_site_pairs`` while the ``Bvh`` is still open — see
    :func:`find_mismatched_end_site_pairs`.
    """
    return list(bvh.node_lr_pairs) if bvh.node_lr_pairs else []


def find_mismatched_end_site_pairs(bvh: Bvh) -> list[tuple[int, int]]:
    """L/R joint pairs whose two sides carry different numbers of end sites.

    Detected here, where the :class:`~pybvh.Bvh` is still open, because
    the consequence lands far away: we persist ``node_lr_pairs`` and
    mirror at *train* time, and pybvh's property drops the end sites of
    such a pair rather than raising.  A dropped tip is exactly the
    half-swapped skeleton :func:`pybvh.transforms.mirror` refuses to
    emit — right policy for a property, wrong outcome for stored
    metadata nobody re-checks.

    Parameters
    ----------
    bvh : Bvh

    Returns
    -------
    list of (int, int)
        Offending pairs in **node** index space, empty when every paired
        joint's two sides agree.  A non-empty list means a train-time
        mirror over ``node_pos`` would swap the paired joints but leave
        their end sites on the original side.

    Notes
    -----
    **Both sides of the comparison are node-space, and that is
    load-bearing.**  Mixing in joint-space ``lr_pairs`` does not fail
    loudly: it indexes the end-site counter with the wrong keys and
    returns arbitrary answers — not a uniform false negative, which
    would at least be noticeable, but a wrong pair that looks like a
    result.  The two spaces diverge as soon as any end site precedes a
    paired joint in file order.
    """
    topology = bvh.fk_topology
    parent_idx = topology.parent_idx
    joint_idx = topology.joint_idx
    # Keyed by NODE index — parent_idx values are node indices.
    end_children = Counter(
        int(p) for i, p in enumerate(parent_idx)
        if p >= 0 and joint_idx[i] < 0)
    return [
        (left, right) for left, right in get_node_lr_pairs(bvh)
        if joint_idx[left] >= 0 and joint_idx[right] >= 0
        and end_children[left] != end_children[right]
    ]


def get_fk_topology_dict(bvh: Bvh) -> dict:
    """The four :class:`pybvh.FkTopology` fields, JSON-serializable.

    What makes the train-time FK refresh in
    :func:`~pybvh_ml.add_joint_rotation_noise` possible: store these
    with the dataset, rebuild an ``FkTopology`` once per dataset with
    :func:`build_fk_topology`, and forward kinematics runs from arrays
    alone with the source ``.bvh`` long closed.

    Returns
    -------
    dict
        ``offsets`` ``(N, 3)`` and ``parent_idx`` / ``joint_idx``
        ``(N,)`` as nested lists, plus ``euler_orders`` (length ``J``,
        indexed by **joint column**, not by node).

    Notes
    -----
    Stored as lists rather than arrays because this dict is persisted as
    JSON inside ``skeleton_info``; :func:`build_fk_topology` converts
    back, and pybvh's constructor validates what it gets.
    """
    topology = bvh.fk_topology
    return {
        'offsets': topology.offsets.tolist(),
        'parent_idx': [int(p) for p in topology.parent_idx],
        'joint_idx': [int(j) for j in topology.joint_idx],
        'euler_orders': list(topology.euler_orders),
    }


def build_fk_topology(skeleton_info: dict) -> FkTopology:
    """Rebuild a :class:`pybvh.FkTopology` from stored metadata.

    The train-time counterpart of :func:`get_fk_topology_dict`: call it
    once per dataset (pybvh's constructor validates, which is not free)
    and hand the result to
    :func:`~pybvh_ml.add_joint_rotation_noise` as ``fk_topology=``.
    :meth:`~pybvh_ml.AugmentationPipeline.standard` does exactly this.

    Parameters
    ----------
    skeleton_info : dict
        From :func:`get_skeleton_info` or the ``skeleton_info`` key of
        :func:`~pybvh_ml.load_preprocessed`.  Must carry
        ``fk_topology``.

    Returns
    -------
    pybvh.FkTopology

    Raises
    ------
    ValueError
        If ``skeleton_info`` has no ``fk_topology`` — datasets written
        before pybvh-ml 0.6.0 do not record it, and it cannot be
        reconstructed from the other keys (bone offsets are not stored
        anywhere else).
    """
    topology = skeleton_info.get("fk_topology")
    if not topology:
        raise ValueError(
            "skeleton_info carries no 'fk_topology', so forward kinematics "
            "cannot be run from it. Datasets written before pybvh-ml 0.6.0 "
            "do not record it and it is not recoverable from the other keys "
            "(the bone offsets are stored nowhere else) — re-run "
            "preprocess_directory, or build the topology from an open Bvh "
            "with bvh.fk_topology.")
    return FkTopology(
        offsets=np.asarray(topology["offsets"], dtype=np.float64),
        parent_idx=np.asarray(topology["parent_idx"], dtype=np.int64),
        joint_idx=np.asarray(topology["joint_idx"], dtype=np.int64),
        euler_orders=list(topology["euler_orders"]),
    )


def get_skeleton_info(bvh: Bvh, include_partitions: bool = False) -> dict:
    """Get unified skeleton metadata dict.

    Parameters
    ----------
    bvh : Bvh
    include_partitions : bool
        If True, include heuristic body-part partitions under
        ``body_partitions``.  These are guessed from joint names — see
        :func:`get_body_partitions` for when the guess fails and how to
        tell.

    Returns
    -------
    dict
        Joint-space keys: ``num_joints``, ``joint_names``, ``edges``,
        ``euler_orders``, ``lr_pairs``, ``lr_mapping``.  Node-space
        keys: ``num_nodes``, ``node_names``, ``node_edges``,
        ``node_lr_pairs``, ``end_site_indices``.  Plus ``fk_topology``,
        the axis strings ``world_up`` / ``rest_forward`` / ``rest_up``,
        and ``mismatched_end_site_pairs``.  Optionally
        ``body_partitions``.

        ``lr_mapping`` is the name-keyed dict from ``bvh.lr_mapping``
        (``None`` when no pairs detected).  The three axis strings feed
        runtime augmentation without reopening the source BVH —
        ``world_up`` is the ``up_axis`` for
        :func:`~pybvh_ml.augmentation.rotate_vertical` and
        :meth:`AugmentationPipeline.standard`; ``rest_up`` is ``None``
        for degenerate rigs.

    Notes
    -----
    **Which key pairs with which stream.** A ``joint_pos`` stream (and
    ``joint_rot``) indexes with ``edges`` / ``lr_pairs`` /
    ``joint_names``; a ``node_pos`` stream indexes with ``node_edges`` /
    ``node_lr_pairs`` / ``node_names``.  Mixing them addresses the wrong
    vertices without any shape error.  Note also that the rotation
    layouts put the root at vertex 0, so ``edges`` needs the documented
    off-by-one shift there — ``pack_to_ctv(streams=("joint_pos",))``
    (``V = J``) and ``streams=("node_pos",)`` (``V = N``) are the two
    packings where the edge lists index packed vertices directly.

    ``mismatched_end_site_pairs`` is empty on a well-formed rig; a
    non-empty list means a train-time node-space mirror would leave
    those pairs' end sites unswapped.  See
    :func:`find_mismatched_end_site_pairs`.

    Everything here is JSON-serializable, because
    :func:`~pybvh_ml.preprocess_directory` persists the whole dict.
    """
    topology = bvh.fk_topology
    info = {
        'num_joints': bvh.joint_count,
        'joint_names': list(bvh.joint_names),
        'edges': list(bvh.edges),
        'euler_orders': list(bvh.euler_orders),
        'lr_pairs': get_lr_pairs(bvh),
        'lr_mapping': dict(bvh.lr_mapping) if bvh.lr_mapping else None,
        'num_nodes': len(bvh.nodes),
        'node_names': [node.name for node in bvh.nodes],
        'node_edges': list(bvh.node_edges),
        'node_lr_pairs': get_node_lr_pairs(bvh),
        'end_site_indices': [i for i, j in enumerate(topology.joint_idx)
                             if j < 0],
        'fk_topology': get_fk_topology_dict(bvh),
        'mismatched_end_site_pairs': find_mismatched_end_site_pairs(bvh),
        'world_up': bvh.world_up,
        'rest_forward': bvh.rest_forward,
        'rest_up': bvh.rest_up,
    }
    if include_partitions:
        info['body_partitions'] = get_body_partitions(bvh)
    return info


# =========================================================================
# Body-part partitions
# =========================================================================

_TORSO_KW = {"hips", "spine", "chest", "abdomen", "pelvis", "torso", "back"}
_HEAD_KW = {"head", "neck", "jaw", "eye"}
_ARM_KW = {"arm", "shoulder", "hand", "finger", "thumb", "wrist",
           "elbow", "clavicle", "collar", "forearm"}
_LEG_KW = {"leg", "hip", "knee", "ankle", "foot", "toe", "thigh",
           "shin", "calf", "upleg"}


def _normalize_name(name: str) -> str:
    """Lowercase and strip separators for fuzzy matching."""
    return name.lower().replace("_", "").replace("-", "").replace(" ", "")


def _detect_side(name: str) -> str | None:
    """Detect if a joint name indicates left or right."""
    lower = name.lower()
    if "left" in lower:
        return "left"
    if "right" in lower:
        return "right"
    # L/R prefix: "L" or "R" followed by uppercase
    if len(name) >= 2:
        if name[0] == "L" and name[1].isupper():
            return "left"
        if name[0] == "R" and name[1].isupper():
            return "right"
    return None


def _has_keyword(normalized: str, keywords: set[str]) -> bool:
    """Check if the normalized name contains any keyword."""
    return any(kw in normalized for kw in keywords)


def get_body_partitions(bvh: Bvh) -> dict[str, list[int]]:
    """Heuristic body-part grouping by joint name patterns.

    Groups joints by matching English keywords against joint names
    (``"LeftForeArm"`` → ``left_arm``), with the side read from a
    ``Left``/``Right`` substring or an ``L``/``R`` prefix before an
    uppercase letter.  This is a *guess from naming*, not a fact read from
    the skeleton: pybvh-ml has no anatomical model, and the alternative — a
    partition supplied by whoever knows the rig — is always more reliable
    where it exists.  Rigs named in another language, or with opaque names
    (``"joint12"``, ``"Bip01 L UpperArm"`` variants outside the keyword
    lists), will be grouped wrongly or not at all.

    ``other`` is the "no keyword matched" bucket and is the signal to check
    before trusting the result: on a normal humanoid it should be small
    (typically end-effector helpers and unnamed props).  A large ``other``,
    or an empty ``left_arm`` / ``right_leg`` on a skeleton that visibly has
    those limbs, means the naming convention was not recognized — pass your
    own joint indices instead of this dict.  There is no signal beyond that:
    a joint that matched the *wrong* keyword lands in a named group and is
    indistinguishable from a correct match.

    Parameters
    ----------
    bvh : Bvh

    Returns
    -------
    dict
        Keys: ``torso``, ``head``, ``left_arm``, ``right_arm``,
        ``left_leg``, ``right_leg``, ``other``.
        Values: lists of joint indices in ``joint_angles`` space.
        Every joint appears in exactly one group.

    See Also
    --------
    get_lr_pairs : Left/right pairing, also name-derived, but detected by
        pybvh and overridable per-skeleton via ``Bvh.lr_mapping``.
    """
    partitions: dict[str, list[int]] = {
        "torso": [],
        "head": [],
        "left_arm": [],
        "right_arm": [],
        "left_leg": [],
        "right_leg": [],
        "other": [],
    }

    for idx, name in enumerate(bvh.joint_names):
        normalized = _normalize_name(name)
        side = _detect_side(name)

        # Torso (no side needed)
        if _has_keyword(normalized, _TORSO_KW) and side is None:
            partitions["torso"].append(idx)
        # Head (no side needed)
        elif _has_keyword(normalized, _HEAD_KW) and side is None:
            partitions["head"].append(idx)
        # Arm with side
        elif side is not None and _has_keyword(normalized, _ARM_KW):
            partitions[f"{side}_arm"].append(idx)
        # Leg with side
        elif side is not None and _has_keyword(normalized, _LEG_KW):
            partitions[f"{side}_leg"].append(idx)
        # Torso with side (e.g., "LeftHip" in some skeletons)
        elif _has_keyword(normalized, _TORSO_KW):
            partitions["torso"].append(idx)
        # Head with side (e.g., "LeftEye")
        elif _has_keyword(normalized, _HEAD_KW):
            partitions["head"].append(idx)
        else:
            partitions["other"].append(idx)

    return partitions
