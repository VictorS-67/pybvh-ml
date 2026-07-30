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


class MotionArrays:
    """One clip's motion streams: root translation and joint rotations.

    Deliberately **not** a tuple and **not** unpackable.  A tuple's arity
    is part of its contract, and this container is expected to grow — the
    per-joint position streams planned for 0.6.0 would turn every
    ``root_pos, joint_rot = ...`` into a "too many values" error, and a
    tuple that silently yielded only its first two fields would drop a
    stream instead.  Attribute access does neither.

    Construction is **keyword-only**, for the reason the augmentation
    functions already refuse positional binding: ``joint_rot`` in
    ``euler`` or ``axisangle`` form is ``(F, J, 3)``, the same shape a
    position stream will be, so no validator could catch a swapped
    positional call.

    Instances are **frozen**.  Frame-count validation runs once, in the
    constructor, and every array-level function in the package relies on
    it having run — reassigning a field afterwards would reintroduce the
    mismatch with nothing left to catch it.  Use :meth:`replace`, which
    revalidates.

    Frozen covers the arrays too, but only in one direction, and the distinction matters when the source is a cache. The fields are **read-only views**: writing through them (``arrays.root_pos[0] = ...``) raises, so nothing — this package included — can modify a clip through the container. They are views, not copies: the constructor does not duplicate the caller's arrays, so the storage is shared, and mutating the *original* array still changes what the container reads. A container built over a Dataset's cached arrays therefore needs no defensive copy to protect the cache from the pipeline (pipeline outputs never alias their inputs), but it is not insulated from code that writes to that cache directly — pass ``np.array(...)`` copies in if anything does. The alternative, copying in the constructor, was rejected because :meth:`replace` runs once per augmentation step and would copy every clip on every step.

    Consequently a field is not a writable working array: take ``np.array(arrays.joint_rot)`` (or ``.copy()``) when you need one, and prefer ``torch.tensor(...)`` over ``torch.from_numpy(...)``, which warns on read-only input. For a whole container detached from the caller's storage, :func:`copy.deepcopy` is the one operation the read-only views cannot express, and it works — instances rebuild through the constructor, so they are also picklable.

    Parameters
    ----------
    root_pos : ndarray, shape (F, 3)
        Root translation per frame.  Always present.
    joint_rot : ndarray, shape (F, J, C), optional
        Per-joint rotations in whatever representation the caller
        declares at the call site.  The container does not record which
        representation that is: a rotation array is meaningless without
        the token, and storing an unenforced copy of it here would
        invite it to disagree with the ``representation=`` a step was
        actually given.

    Attributes
    ----------
    root_pos, joint_rot
        As above.

    Notes
    -----
    **dtype is preserved, not promoted.** A floating-point input keeps its dtype — ``float32`` in, ``float32`` out — because the container is what a per-sample Dataset holds, and silently doubling a cached clip's memory and bandwidth is not a decision to make on the caller's behalf. Non-floating input (an integer array, a nested list of ints) is promoted to ``float64``, the package's compute dtype, since rotation math on integers is never what was meant. ``root_pos`` and ``joint_rot`` are converted independently and may differ.

    Augmentation preserves it too, without computing in it: every augmentation function and :class:`~pybvh_ml.AugmentationPipeline` runs the math in ``float64`` — pybvh's dtype, and the only one its conversions are exact in — then returns each stream in the dtype it arrived as. So a ``float32`` clip stays ``float32`` end to end through augmentation while the arithmetic is still done in double, and the result never depends on which probabilistic steps happened to fire.

    Where it stops: the packers (:func:`~pybvh_ml.pack_to_ctv` and friends) and ``standardize_length(method="resample_linear")`` produce ``float64`` regardless, so the array handed to a model is ``float64`` either way — the PyTorch datasets then emit ``torch.float32`` tensors. The preservation is about what the container costs to hold and pass around, not a single-precision compute path.

    Raises
    ------
    ValueError
        If ``root_pos`` is not ``(F, 3)``, ``joint_rot`` is not 3-D, or
        the two disagree on frame count.  A mismatch means the caller
        paired arrays from different clips (or different slices of one
        clip), which downstream math would silently interpolate or index
        past.

    Examples
    --------
    >>> arrays = MotionArrays(root_pos=root_pos, joint_rot=joint_6d)
    >>> out = rotate_vertical(arrays, angle=np.pi / 4, up_axis="+y",
    ...                       representation="6d")
    >>> out.joint_rot.shape
    (120, 31, 6)

    See Also
    --------
    MotionArrays.from_bvh : Build one straight from a pybvh ``Bvh``.
    """

    __slots__ = ("root_pos", "joint_rot")

    def __init__(
        self,
        *,
        root_pos: npt.ArrayLike,
        joint_rot: npt.ArrayLike | None = None,
    ) -> None:
        root_arr = _as_readonly_float(root_pos)
        rot_arr = None if joint_rot is None else _as_readonly_float(joint_rot)
        _validate(root_arr, rot_arr)
        # Bypass the frozen __setattr__ during construction only.
        object.__setattr__(self, "root_pos", root_arr)
        object.__setattr__(self, "joint_rot", rot_arr)

    # -- frozen ----------------------------------------------------------

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(
            f"MotionArrays is frozen; cannot set {name!r}. The frame-count "
            f"invariant is checked once at construction and the rest of the "
            f"package relies on it. Use replace() to derive a new instance: "
            f"arrays.replace({name}=...)")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("MotionArrays is frozen; cannot delete fields")

    # -- not a tuple -----------------------------------------------------

    def __iter__(self):
        """Refuse iteration, naming the pre-0.5.0 unpack it comes from.

        Defined only to raise: ``root_pos, joint_data = pipeline(...)`` was the shape every downstream had, and without this it fails as a bare "cannot unpack non-iterable" with nothing pointing at the container's two fields.
        """
        raise TypeError(
            "MotionArrays is not iterable, so 'root_pos, joint_rot = ...' "
            "does not unpack it — that was the pre-0.5.0 return shape. Read "
            "the fields instead: out.root_pos / out.joint_rot. The container "
            "is deliberately not a tuple because it is expected to grow (a "
            "joint_pos stream in 0.6.0), and a fixed-arity unpack would "
            "either break or silently drop a stream.")

    # -- construction / derivation ---------------------------------------

    @classmethod
    def from_bvh(
        cls,
        bvh: "Bvh",
        representation: str,
        *,
        center_root: bool = False,
    ) -> "MotionArrays":
        """Extract one clip's arrays from a pybvh ``Bvh``.

        The producer counterpart to the packers: this is where the
        extract → augment → pack journey starts, so callers never
        hand-assemble the container.

        Parameters
        ----------
        bvh : pybvh.Bvh
        representation : str
            One of ``"quat"``, ``"6d"``, ``"axisangle"``, ``"euler"``.
            ``"rotmat"`` is not an extraction representation — convert
            from another with :func:`~pybvh_ml.convert_arrays`.
        center_root : bool, optional
            Subtract the first frame's root position from every frame.
            All three components, pybvh-ml's root-relative convention —
            **not** pybvh's ``centered="first"``, which zeroes only the
            two ground-plane axes.  Default False, leaving the clip in
            world coordinates.

        Returns
        -------
        MotionArrays
        """
        from .preprocessing import extract_repr

        root_pos, joint_rot = extract_repr(bvh, representation)
        if center_root and root_pos.shape[0] > 0:
            root_pos = root_pos - root_pos[0:1]
        return cls(root_pos=root_pos, joint_rot=joint_rot)

    def replace(
        self,
        *,
        root_pos: Any = _UNSET,
        joint_rot: Any = _UNSET,
    ) -> "MotionArrays":
        """Return a new instance with the given fields replaced.

        The only way to modify a frozen container, and it revalidates —
        so a replacement that breaks the frame-count invariant raises
        here rather than surfacing as corrupt data later.
        """
        return MotionArrays(
            root_pos=(self.root_pos if root_pos is _UNSET else root_pos),
            joint_rot=(self.joint_rot if joint_rot is _UNSET else joint_rot),
        )

    # -- introspection ---------------------------------------------------

    @property
    def frame_count(self) -> int:
        """Number of frames, ``F``."""
        return int(self.root_pos.shape[0])

    def __repr__(self) -> str:
        rot = "None" if self.joint_rot is None else str(self.joint_rot.shape)
        return (f"MotionArrays(root_pos={self.root_pos.shape}, "
                f"joint_rot={rot})")

    def __reduce__(self):
        """Rebuild through the constructor, for ``pickle`` and ``copy``.

        Both would otherwise go through ``setattr`` on a blank instance and hit the frozen guard. Rebuilding also means :func:`copy.deepcopy` returns a container whose storage is detached from this one's — the one operation the read-only fields cannot express.
        """
        return (_rebuild, (self.root_pos, self.joint_rot))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MotionArrays):
            return NotImplemented
        if not np.array_equal(self.root_pos, other.root_pos):
            return False
        if (self.joint_rot is None) != (other.joint_rot is None):
            return False
        return (self.joint_rot is None
                or np.array_equal(self.joint_rot, other.joint_rot))

    __hash__ = None  # type: ignore[assignment]


def _rebuild(
    root_pos: npt.ArrayLike,
    joint_rot: npt.ArrayLike | None,
) -> MotionArrays:
    """Module-level constructor call, for :meth:`MotionArrays.__reduce__`.

    Needed because construction is keyword-only and ``__reduce__`` passes its arguments positionally.
    """
    return MotionArrays(root_pos=root_pos, joint_rot=joint_rot)


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


def _validate(
    root_pos: npt.NDArray[np.floating],
    joint_rot: npt.NDArray[np.floating] | None,
) -> None:
    """Shape and frame-count checks, run once per instance."""
    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(
            f"root_pos must have shape (F, 3), got {root_pos.shape}")
    if joint_rot is None:
        return
    if joint_rot.ndim != 3:
        raise ValueError(
            f"joint_rot must have shape (F, J, C), got {joint_rot.shape}. "
            f"rotmat data is carried flat as (F, J, 9) — reshape a "
            f"(F, J, 3, 3) array before building MotionArrays.")
    if root_pos.shape[0] != joint_rot.shape[0]:
        raise ValueError(
            f"root_pos and joint_rot disagree on frame count: root_pos has "
            f"{root_pos.shape[0]} frames (shape {root_pos.shape}), joint_rot "
            f"has {joint_rot.shape[0]} (shape {joint_rot.shape})")


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
