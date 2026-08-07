"""The container every array-level function in pybvh-ml takes and returns.

One clip's motion, as the streams a model consumes.  Augmentation,
packing and the PyTorch datasets all speak :class:`MotionArrays` rather
than loose arrays, so adding a stream later is additive instead of
breaking every call site.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pybvh import Bvh


_UNSET = object()

STREAM_NAMES: tuple[str, ...] = (
    "root_pos", "joint_rot", "joint_pos", "node_pos")
"""Every stream a :class:`MotionArrays` can carry, in field order.

The vocabulary the augmentation stream declarations are written in —
see :func:`~pybvh_ml.handles_streams`.
"""

POSITION_CENTERINGS: tuple[str, ...] = ("world", "skeleton", "first")
"""Legal values of :attr:`MotionArrays.position_centering`.

The three frames :meth:`pybvh.Bvh.node_positions` produces, named
identically so the two never drift.
"""


class MotionArrays:
    """One clip's motion streams: root translation, joint rotations, joint
    or node positions.

    Deliberately **not** a tuple and **not** unpackable.  A tuple's arity
    is part of its contract, and this container grows — the per-joint and
    per-node position streams 0.6.0 added would have turned every
    ``root_pos, joint_rot = ...`` into a "too many values" error, and a
    tuple that silently yielded only its first two fields would drop a
    stream instead.  Attribute access does neither.

    Construction is **keyword-only**, for the reason the augmentation
    functions already refuse positional binding: ``joint_rot`` in
    ``euler`` or ``axisangle`` form is ``(F, J, 3)``, the same shape
    ``joint_pos`` is, so no validator could catch a swapped positional
    call.

    Instances are **frozen**.  Shape and frame-count validation runs
    once, in the constructor, and every array-level function in the
    package relies on it having run — reassigning a field afterwards
    would reintroduce the mismatch with nothing left to catch it.  Use
    :meth:`replace`, which revalidates.

    Frozen covers the arrays too, but only in one direction, and the distinction matters when the source is a cache. The fields are **read-only views**: writing through them (``arrays.root_pos[0] = ...``) raises, so nothing — this package included — can modify a clip through the container. They are views, not copies: the constructor does not duplicate the caller's arrays, so the storage is shared, and mutating the *original* array still changes what the container reads. A container built over a Dataset's cached arrays therefore needs no defensive copy to protect the cache from the pipeline (pipeline outputs never alias their inputs), but it is not insulated from code that writes to that cache directly — pass ``np.array(...)`` copies in if anything does. The alternative, copying in the constructor, was rejected because :meth:`replace` runs once per augmentation step and would copy every clip on every step.

    Consequently a field is not a writable working array: take ``np.array(arrays.joint_rot)`` (or ``.copy()``) when you need one, and prefer ``torch.tensor(...)`` over ``torch.from_numpy(...)``, which warns on read-only input. For a whole container detached from the caller's storage, :func:`copy.deepcopy` is the one operation the read-only views cannot express, and it works — instances rebuild through the constructor, so they are also picklable.

    Parameters
    ----------
    root_pos : ndarray, shape (F, 3)
        Root translation per frame.  Always present — see
        **Keypoint-only clips** below if your source has no root
        trajectory.
    joint_rot : ndarray, shape (F, J, C), optional
        Per-joint rotations in whatever representation the caller
        declares at the call site.  The container does not record which
        representation that is: a rotation array is meaningless without
        the token, and storing an unenforced copy of it here would
        invite it to disagree with the ``representation=`` a step was
        actually given.
    joint_pos : ndarray, shape (F, J, 3), optional
        Per-joint 3-D positions, index-aligned with ``joint_rot`` and
        with :attr:`pybvh.Bvh.joint_angles` — end sites excluded.  The
        stream skeleton action-recognition models (ST-GCN, CTR-GCN,
        PySKL) consume almost exclusively.
    node_pos : ndarray, shape (F, N, 3), optional
        Per-node 3-D positions, joints **and** end sites, aligned with
        :meth:`pybvh.Bvh.node_positions`.  End sites are fingertips, toe
        tips and the head top, so this is the "visual" skeleton an
        NTU-25-style graph wants; it pairs with
        ``get_edge_list(bvh, include_end_sites=True)``.
    position_centering : {"world", "skeleton", "first"}, optional
        Which frame the position streams are expressed in — see the
        Notes.  Must be ``None`` when no position stream is present, and
        may legitimately be ``None`` when one is.

    Attributes
    ----------
    root_pos, joint_rot, joint_pos, node_pos, position_centering
        As above.

    Notes
    -----
    **Keypoint-only clips: "positions-only" means rotation-free, not
    root-free.** ``representation=None`` and a bare ``joint_pos`` is a
    supported clip, but ``root_pos`` stays mandatory, so a source with
    no root trajectory — pose-estimator keypoints, the usual ST-GCN
    arrival path — has to supply something.  ``np.zeros((F, 3))`` is the
    convention, and it is safe **only** if you then keep ``"root_pos"``
    out of ``streams=``.  Two things go quietly wrong otherwise:
    ``center_root=True`` becomes a no-op that looks like it worked (it
    subtracts a zero first frame), and any packing that includes
    ``"root_pos"`` puts a vertex of zeros at index 0, which the model
    reads as a joint and which shifts every real joint one place out of
    step with ``skeleton_info["edges"]``.  So pack
    ``streams=("joint_pos",)`` — which is the canonical ST-GCN layout
    anyway — and the fabricated array never reaches a tensor.  If the
    keypoint set has a pelvis or hip, using it as ``root_pos`` is
    strictly better than zeros: the trajectory becomes real and both
    hazards disappear.

    Why mandatory: ``root_pos`` is the container's frame count and the
    reference point every centering convention is stated against, so
    an optional root would make ``F`` and ``position_centering``
    conditional on which streams happen to be present.  Making it
    genuinely optional is a coherent change and a large one — it is
    scoped for a later release, not an oversight here.

    **Pick one position space, not both.** ``joint_pos`` is a subset of
    ``node_pos`` — ``node_pos[:, joint_idx >= 0]`` is exactly
    ``joint_pos``, with ``joint_idx`` from
    ``skeleton_info["fk_topology"]`` — so carrying both is redundant
    though harmless.  Only ``joint_pos`` can be concatenated with
    ``joint_rot`` on the channel axis; ``node_pos`` has a different
    ``V``, and the packers refuse to combine the two.

    **``position_centering`` is a convention, and it travels with the
    arrays rather than only with the dataset**, because at least one
    step's correctness depends on it.  ``"world"`` leaves positions in
    the same frame as ``root_pos``, so a joint position already contains
    the root trajectory; ``"skeleton"`` puts the root at the origin in
    every frame, the form most NTU-style pipelines feed a model, with
    the trajectory then carried by ``root_pos`` alone; ``"first"`` is
    pybvh's ground-plane centering (the first frame's root subtracted in
    the two axes perpendicular to world up).  The three coincide only
    for a clip whose root never moves.

    **``position_centering=None`` is legal, and fails at use rather than
    at construction.**  The two directions are not symmetric: a
    centering value with no positions to describe is meaningless and
    raises here, but positions with an undeclared frame are a legitimate
    state, because most of the surface does not care.
    :func:`~pybvh_ml.mirror`, :func:`~pybvh_ml.speed_perturbation_arrays`,
    :func:`~pybvh_ml.dropout_arrays`, the keypoint-jitter functions,
    :func:`~pybvh_ml.rotate_vertical` and ``pack_to_*(center_root=False)``
    are all correct without knowing the frame — each is a rigid or
    temporal operation applied identically to both streams.  The
    surfaces that do depend on it — :func:`~pybvh_ml.add_root_position_noise`,
    the FK refresh inside :func:`~pybvh_ml.add_joint_rotation_noise`, and
    ``pack_to_*(center_root=True)`` — raise naming the field.  Requiring
    a declaration at construction would make those raises dead code and,
    worse, invite a guessed ``"world"`` from a caller who does not
    actually know; a confidently wrong convention is worse than an
    honest ``None``.  Anything *this library* writes records it (see
    :func:`~pybvh_ml.preprocess_directory`); ``None`` is for positions
    that came from somewhere else.

    A consequence worth knowing: :meth:`replace` dropping the last
    position stream must clear ``position_centering`` in the same call,
    or the result raises.  No pipeline step drops a stream, so this is
    rare.

    **dtype is preserved, not promoted.** A floating-point input keeps its dtype — ``float32`` in, ``float32`` out — because the container is what a per-sample Dataset holds, and silently doubling a cached clip's memory and bandwidth is not a decision to make on the caller's behalf. Non-floating input (an integer array, a nested list of ints) is promoted to ``float64``, the package's compute dtype, since rotation math on integers is never what was meant. Every stream is converted independently and they may differ — ``float32`` keypoints beside ``float64`` rotations is the normal case for an ST-GCN pipeline, not an exotic one.

    Augmentation preserves it too, without computing in it: every augmentation function and :class:`~pybvh_ml.AugmentationPipeline` runs the math in ``float64`` — pybvh's dtype, and the only one its conversions are exact in — then returns each stream in the dtype it arrived as. So a ``float32`` clip stays ``float32`` end to end through augmentation while the arithmetic is still done in double, and the result never depends on which probabilistic steps happened to fire.

    Where it stops: the packers (:func:`~pybvh_ml.pack_to_ctv` and friends) and ``standardize_length(method="resample_linear")`` produce ``float64`` regardless, so the array handed to a model is ``float64`` either way — the PyTorch datasets then emit ``torch.float32`` tensors. The preservation is about what the container costs to hold and pass around, not a single-precision compute path.

    Raises
    ------
    ValueError
        If ``root_pos`` is not ``(F, 3)``, a rotation or position array
        has the wrong rank or trailing width, two streams disagree on
        frame count, ``joint_pos`` and ``joint_rot`` disagree on ``J``,
        ``node_pos`` has fewer vertices than the joint-space streams, or
        ``position_centering`` is set without a position stream (or is
        not one of :data:`POSITION_CENTERINGS`).

    Examples
    --------
    >>> arrays = MotionArrays(root_pos=root_pos, joint_rot=joint_6d)
    >>> out = rotate_vertical(arrays, angle=np.pi / 4, up_axis="+y",
    ...                       representation="6d")
    >>> out.joint_rot.shape
    (120, 31, 6)

    >>> keypoints = MotionArrays.from_bvh(
    ...     bvh, representation=None, include_positions=True,
    ...     position_centering="skeleton")
    >>> pack_to_ctv(keypoints, streams=("joint_pos",)).shape
    (3, 120, 31)

    See Also
    --------
    MotionArrays.from_bvh : Build one straight from a pybvh ``Bvh``.
    """

    __slots__ = ("root_pos", "joint_rot", "joint_pos", "node_pos",
                 "position_centering")

    def __init__(
        self,
        *,
        root_pos: npt.ArrayLike,
        joint_rot: npt.ArrayLike | None = None,
        joint_pos: npt.ArrayLike | None = None,
        node_pos: npt.ArrayLike | None = None,
        position_centering: str | None = None,
    ) -> None:
        values = {
            "root_pos": _as_readonly_float(root_pos),
            "joint_rot": (None if joint_rot is None
                          else _as_readonly_float(joint_rot)),
            "joint_pos": (None if joint_pos is None
                          else _as_readonly_float(joint_pos)),
            "node_pos": (None if node_pos is None
                         else _as_readonly_float(node_pos)),
        }
        _validate(values, position_centering)
        # Bypass the frozen __setattr__ during construction only.
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "position_centering", position_centering)

    # -- frozen ----------------------------------------------------------

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(
            f"MotionArrays is frozen; cannot set {name!r}. The shape and "
            f"frame-count invariants are checked once at construction and "
            f"the rest of the package relies on them. Use replace() to "
            f"derive a new instance: arrays.replace({name}=...)")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("MotionArrays is frozen; cannot delete fields")

    # -- not a tuple -----------------------------------------------------

    def __iter__(self):
        """Refuse iteration, naming the pre-0.5.0 unpack it comes from.

        Defined only to raise: ``root_pos, joint_data = pipeline(...)`` was the shape every downstream had, and without this it fails as a bare "cannot unpack non-iterable" with nothing pointing at the container's fields.
        """
        raise TypeError(
            "MotionArrays is not iterable, so 'root_pos, joint_rot = ...' "
            "does not unpack it — that was the pre-0.5.0 return shape. Read "
            "the fields instead: out.root_pos / out.joint_rot, and "
            "out.joint_pos / out.node_pos for the position streams added in "
            "0.6.0. The container is deliberately not a tuple because it "
            "grows, and a fixed-arity unpack would either break or silently "
            "drop a stream.")

    # -- construction / derivation ---------------------------------------

    @classmethod
    def from_bvh(
        cls,
        bvh: "Bvh",
        representation: str | None = None,
        *,
        center_root: bool = False,
        include_positions: bool = False,
        position_space: str = "joint",
        position_centering: str = "world",
    ) -> "MotionArrays":
        """Extract one clip's arrays from a pybvh ``Bvh``.

        The producer counterpart to the packers: this is where the
        extract → augment → pack journey starts, so callers never
        hand-assemble the container.

        Parameters
        ----------
        bvh : pybvh.Bvh
        representation : str, optional
            One of ``"quat"``, ``"6d"``, ``"axisangle"``, ``"euler"``.
            ``"rotmat"`` is not an extraction representation — convert
            from another with :func:`~pybvh_ml.convert_arrays`.
            ``None`` extracts no rotation stream, which requires
            ``include_positions=True``: that is the positions-only
            (ST-GCN / CTR-GCN) journey.
        center_root : bool, optional
            Subtract the first frame's root position from every frame.
            All three components, pybvh-ml's root-relative convention —
            **not** pybvh's ``centered="first"``, which zeroes only the
            two ground-plane axes.  Default False, leaving the clip in
            world coordinates.

            Under ``position_centering="world"`` or ``"first"`` the
            identical shift is applied to every position vertex, so the
            two streams stay in one frame; under ``"skeleton"`` the
            positions are already root-relative and are left alone.
        include_positions : bool, optional
            Also extract positions, via :meth:`pybvh.Bvh.joint_positions`
            or :meth:`pybvh.Bvh.node_positions`.  Both are backed by
            pybvh's cached world-frame FK, so requesting positions
            alongside a rotation representation costs one array
            derivation, not a second kinematics pass.
        position_space : {"joint", "node"}, optional
            Which index space the positions live in: ``"joint"``
            (default) fills ``joint_pos`` and index-aligns with
            ``joint_rot``; ``"node"`` fills ``node_pos``, which includes
            end sites and pairs with the node-space edge list.  One flag
            rather than two booleans, because the two spaces are
            alternatives — ``node_pos`` already contains ``joint_pos``.
        position_centering : {"world", "skeleton", "first"}, optional
            Frame the positions are extracted in, passed straight to
            pybvh's ``centered=`` and recorded on the container.  Default
            ``"world"``, which keeps them in the same frame as
            ``root_pos``.  See :class:`MotionArrays`.

        Returns
        -------
        MotionArrays
        """
        from .preprocessing import extract_repr

        if representation is None and not include_positions:
            raise ValueError(
                "MotionArrays.from_bvh was asked for nothing: "
                "representation=None extracts no rotations, so pass "
                "include_positions=True for a positions-only clip, or name "
                "a representation.")

        if representation is None:
            root_pos, joint_rot = bvh.root_pos.copy(), None
        else:
            root_pos, joint_rot = extract_repr(bvh, representation)

        joint_pos = node_pos = None
        if include_positions:
            _validate_position_space(position_space)
            _validate_position_centering(position_centering)
            if position_space == "joint":
                joint_pos = bvh.joint_positions(centered=position_centering)
            else:
                node_pos = bvh.node_positions(centered=position_centering)

        if center_root:
            root_pos, joint_pos, node_pos = center_root_streams(
                root_pos, joint_pos, node_pos,
                None if not include_positions else position_centering,
                "MotionArrays.from_bvh")

        return cls(root_pos=root_pos, joint_rot=joint_rot,
                   joint_pos=joint_pos, node_pos=node_pos,
                   position_centering=(position_centering if include_positions
                                       else None))

    def replace(
        self,
        *,
        root_pos: Any = _UNSET,
        joint_rot: Any = _UNSET,
        joint_pos: Any = _UNSET,
        node_pos: Any = _UNSET,
        position_centering: Any = _UNSET,
    ) -> "MotionArrays":
        """Return a new instance with the given fields replaced.

        The only way to modify a frozen container, and it revalidates —
        so a replacement that breaks an invariant raises here rather
        than surfacing as corrupt data later.  In particular, dropping
        the last position stream (``replace(joint_pos=None)``) must
        clear ``position_centering`` in the same call, since a centering
        with nothing to describe is not a legal state.
        """
        return MotionArrays(
            root_pos=(self.root_pos if root_pos is _UNSET else root_pos),
            joint_rot=(self.joint_rot if joint_rot is _UNSET else joint_rot),
            joint_pos=(self.joint_pos if joint_pos is _UNSET else joint_pos),
            node_pos=(self.node_pos if node_pos is _UNSET else node_pos),
            position_centering=(self.position_centering
                                if position_centering is _UNSET
                                else position_centering),
        )

    # -- introspection ---------------------------------------------------

    @property
    def frame_count(self) -> int:
        """Number of frames, ``F``."""
        return int(self.root_pos.shape[0])

    @property
    def present_streams(self) -> frozenset[str]:
        """Names of the streams this clip actually carries.

        What an augmentation step's declaration is checked against —
        see :func:`~pybvh_ml.handles_streams`.  ``"root_pos"`` is always
        in it.
        """
        return frozenset(
            name for name in STREAM_NAMES if getattr(self, name) is not None)

    def __repr__(self) -> str:
        parts = [f"root_pos={self.root_pos.shape}"]
        for name in ("joint_rot", "joint_pos", "node_pos"):
            value = getattr(self, name)
            if value is not None:
                parts.append(f"{name}={value.shape}")
        if self.position_centering is not None:
            parts.append(f"position_centering={self.position_centering!r}")
        return f"MotionArrays({', '.join(parts)})"

    def __reduce__(self):
        """Rebuild through the constructor, for ``pickle`` and ``copy``.

        Both would otherwise go through ``setattr`` on a blank instance and hit the frozen guard. Rebuilding also means :func:`copy.deepcopy` returns a container whose storage is detached from this one's — the one operation the read-only fields cannot express.
        """
        return (_rebuild, (self.root_pos, self.joint_rot, self.joint_pos,
                           self.node_pos, self.position_centering))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MotionArrays):
            return NotImplemented
        if self.position_centering != other.position_centering:
            return False
        for name in STREAM_NAMES:
            mine, theirs = getattr(self, name), getattr(other, name)
            if (mine is None) != (theirs is None):
                return False
            if mine is not None and not np.array_equal(mine, theirs):
                return False
        return True

    __hash__ = None  # type: ignore[assignment]


def _rebuild(
    root_pos: npt.ArrayLike,
    joint_rot: npt.ArrayLike | None,
    joint_pos: npt.ArrayLike | None,
    node_pos: npt.ArrayLike | None,
    position_centering: str | None,
) -> MotionArrays:
    """Module-level constructor call, for :meth:`MotionArrays.__reduce__`.

    Needed because construction is keyword-only and ``__reduce__`` passes its arguments positionally.
    """
    return MotionArrays(root_pos=root_pos, joint_rot=joint_rot,
                        joint_pos=joint_pos, node_pos=node_pos,
                        position_centering=position_centering)


def _as_readonly_float(value: npt.ArrayLike) -> npt.NDArray[np.floating]:
    """Coerce to a floating array and return a read-only view of it.

    The view is what makes the container's fields unwritable without copying: ``setflags`` on a fresh view leaves the caller's own array object writable, while a write through the field raises.
    """
    arr = np.asarray(value)
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float64)
    readonly = arr.view()
    readonly.setflags(write=False)
    return readonly


def _validate_position_space(position_space: str) -> None:
    """Reject an index space that is neither joint nor node."""
    if position_space not in ("joint", "node"):
        raise ValueError(
            f"position_space must be 'joint' or 'node', got "
            f"{position_space!r}. 'joint' index-aligns with joint_rot; "
            f"'node' includes end sites and pairs with the node-space edge "
            f"list.")


def _validate_position_centering(position_centering: str | None) -> None:
    """Reject a centering token pybvh's ``centered=`` would not accept."""
    if (position_centering is not None
            and position_centering not in POSITION_CENTERINGS):
        raise ValueError(
            f"position_centering must be one of "
            f"{list(POSITION_CENTERINGS)} or None, got "
            f"{position_centering!r}")


def _validate(
    values: dict[str, npt.NDArray[np.floating] | None],
    position_centering: str | None,
) -> None:
    """Shape, frame-count and centering checks, run once per instance."""
    root_pos = values["root_pos"]
    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(
            f"root_pos must have shape (F, 3), got {root_pos.shape}")

    joint_rot = values["joint_rot"]
    if joint_rot is not None and joint_rot.ndim != 3:
        raise ValueError(
            f"joint_rot must have shape (F, J, C), got {joint_rot.shape}. "
            f"rotmat data is carried flat as (F, J, 9) — reshape a "
            f"(F, J, 3, 3) array before building MotionArrays.")

    for name in ("joint_pos", "node_pos"):
        positions = values[name]
        if positions is None:
            continue
        vertex = "J" if name == "joint_pos" else "N"
        if positions.ndim != 3 or positions.shape[2] != 3:
            raise ValueError(
                f"{name} must have shape (F, {vertex}, 3), got "
                f"{positions.shape}")

    frames = root_pos.shape[0]
    for name in ("joint_rot", "joint_pos", "node_pos"):
        stream = values[name]
        if stream is not None and stream.shape[0] != frames:
            raise ValueError(
                f"root_pos and {name} disagree on frame count: root_pos has "
                f"{frames} frames (shape {root_pos.shape}), {name} has "
                f"{stream.shape[0]} (shape {stream.shape})")

    joint_pos = values["joint_pos"]
    if joint_rot is not None and joint_pos is not None:
        if joint_rot.shape[1] != joint_pos.shape[1]:
            raise ValueError(
                f"joint_rot and joint_pos disagree on joint count: "
                f"{joint_rot.shape[1]} vs {joint_pos.shape[1]}. The two are "
                f"index-aligned by definition — a joint_pos row is the "
                f"position of the joint whose rotation is the joint_rot row "
                f"with the same index.")

    node_pos = values["node_pos"]
    if node_pos is not None:
        joint_counts = [s.shape[1] for s in (joint_rot, joint_pos)
                        if s is not None]
        if joint_counts and node_pos.shape[1] < max(joint_counts):
            raise ValueError(
                f"node_pos has {node_pos.shape[1]} vertices but the "
                f"joint-space streams have {max(joint_counts)}. Nodes are a "
                f"superset of joints (joints plus their end sites), so N >= "
                f"J always — this pairs arrays from different skeletons, or "
                f"a joint-space array passed as node_pos.")

    _validate_position_centering(position_centering)
    if position_centering is not None and joint_pos is None and node_pos is None:
        raise ValueError(
            f"position_centering={position_centering!r} was given but this "
            f"MotionArrays carries no position stream to describe. Pass "
            f"joint_pos=... or node_pos=..., or drop the centering. (The "
            f"reverse — positions with position_centering=None — is legal: "
            f"an unknown frame is an honest state, and only the steps that "
            f"depend on it raise.)")


def require_joint_rot(
    arrays: MotionArrays,
    caller: str,
) -> npt.NDArray[np.floating]:
    """Return ``arrays.joint_rot``, raising a named error when absent.

    Shared by every step that cannot do its job without rotations, so the
    message is identical wherever it fires.
    """
    if arrays.joint_rot is None:
        raise ValueError(
            f"{caller} needs joint rotations, but this MotionArrays carries "
            f"none (joint_rot is None)")
    return arrays.joint_rot


def require_position_centering(
    arrays: MotionArrays,
    caller: str,
) -> str:
    """Return ``arrays.position_centering``, raising when it is unknown.

    The shared message for the three surfaces whose correctness depends
    on the frame the positions live in — root-position noise, the FK
    refresh in :func:`~pybvh_ml.add_joint_rotation_noise`, and
    ``pack_to_*(center_root=True)``.  Only reachable when a position
    stream is present; see :class:`MotionArrays` for why ``None`` is
    a legal state everywhere else.
    """
    if arrays.position_centering is None:
        raise ValueError(
            f"{caller} needs to know which frame the position streams are "
            f"in, but position_centering is None. Set it when you build the "
            f"container — MotionArrays(..., position_centering='world') for "
            f"positions in the same frame as root_pos, 'skeleton' for "
            f"root-relative positions, 'first' for pybvh's ground-plane "
            f"centering. Datasets written by preprocess_directory record it; "
            f"load_preprocessed surfaces it as 'position_centering'.")
    return arrays.position_centering


def center_root_streams(
    root_pos: npt.NDArray[np.floating],
    joint_pos: npt.NDArray[np.floating] | None,
    node_pos: npt.NDArray[np.floating] | None,
    position_centering: str | None,
    caller: str,
) -> tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating] | None,
    npt.NDArray[np.floating] | None,
]:
    """Subtract the first frame's root position from every present stream.

    pybvh-ml's root-relative convention: **all three components**, unlike
    pybvh's ``centered="first"``, which zeroes only the two ground-plane
    axes.  The shared implementation behind ``center_root=True`` on
    :meth:`MotionArrays.from_bvh`, the packers and
    :func:`~pybvh_ml.preprocess_directory`, so the three cannot drift.

    What happens to the positions depends on the frame they are in:

    - ``"world"`` — they carry the root trajectory, so the identical
      shift applies to every vertex and the two streams stay in one
      frame.
    - ``"skeleton"`` — they are already root-relative and do not move.
    - ``"first"`` — the identical shift again, and for a subtler reason:
      ground-plane centering already puts the positions in a *different*
      frame from ``root_pos`` (offset by the first frame's root in the
      two non-up axes), so shifting both by the same amount is what
      leaves that relationship unchanged.  Leaving them alone would
      change it.  Note that :func:`~pybvh_ml.preprocess_directory`
      refuses to *write* a dataset with ``center_root=True`` and this
      centering: the recorded flag would suggest a coherence between the
      two streams that ground-plane centering never established.
    - ``None`` with positions present — raises, since guessing would
      either double-shift or leave the streams inconsistent.

    Returns the three streams; the positions are ``None`` where they
    came in ``None``.
    """
    if root_pos.shape[0] == 0:
        return root_pos, joint_pos, node_pos

    shift = root_pos[0:1]
    centered_root = root_pos - shift
    if joint_pos is None and node_pos is None:
        return centered_root, None, None

    if position_centering is None:
        raise ValueError(
            f"{caller} was asked to center the root of a clip carrying "
            f"positions, but position_centering is None, so whether the "
            f"positions move with the root is unknown. Declare it "
            f"('world' / 'first' shift with the root, 'skeleton' does not), "
            f"or center the root before attaching the positions.")

    if position_centering == "skeleton":
        return centered_root, joint_pos, node_pos

    vertex_shift = shift[:, np.newaxis, :]
    return (
        centered_root,
        None if joint_pos is None else joint_pos - vertex_shift,
        None if node_pos is None else node_pos - vertex_shift,
    )
