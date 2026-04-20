"""Skeleton graph metadata for GCN and graph-based models.

Provides edge lists, left/right joint pairs, and unified skeleton
descriptors — the topology data that GCN and Transformer models
consume.  Only uses pybvh's public API.
"""
from __future__ import annotations

from pybvh import Bvh


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


def get_skeleton_info(bvh: Bvh, include_partitions: bool = False) -> dict:
    """Get unified skeleton metadata dict.

    Parameters
    ----------
    bvh : Bvh
    include_partitions : bool
        If True, include heuristic body-part partitions.

    Returns
    -------
    dict
        Keys: ``num_joints``, ``joint_names``, ``edges``,
        ``euler_orders``, ``lr_pairs``, ``lr_mapping``.  Optionally
        ``body_partitions``.  ``lr_mapping`` is the name-keyed dict
        from ``bvh.lr_mapping`` (``None`` when no pairs detected).
    """
    info = {
        'num_joints': bvh.joint_count,
        'joint_names': list(bvh.joint_names),
        'edges': list(bvh.edges),
        'euler_orders': list(bvh.euler_orders),
        'lr_pairs': get_lr_pairs(bvh),
        'lr_mapping': dict(bvh.lr_mapping) if bvh.lr_mapping else None,
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

    Groups joints into body parts based on name keywords.  This is
    heuristic and may not be perfect for all skeleton naming
    conventions.

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
