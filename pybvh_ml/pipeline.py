"""Composable augmentation pipeline for ML training.

Designed to be called inside a PyTorch Dataset's ``__getitem__``
or any data loading loop.
"""
from __future__ import annotations

import inspect
from typing import Callable, NamedTuple

import numpy as np
import numpy.typing as npt

from ._staged import STAGED_DISPATCH, _StagingState
from .augmentation import _validate_frame_counts


class AugmentationStep(NamedTuple):
    """One configured pipeline step.

    A :class:`tuple` subclass, so a step still unpacks as
    ``(fn, prob, kwargs)`` and indexes as ``step[2]`` — the named
    fields are what reads at the call site when introspecting a
    pipeline (``pipeline.augmentations[0].kwargs["angle"]``).
    """

    fn: Callable
    prob: float
    kwargs: dict


def _coerce_step(step: object, index: int) -> AugmentationStep:
    """Normalize one constructor entry into an :class:`AugmentationStep`.

    Validates here rather than at call time: a malformed step (wrong
    arity, a probability outside ``[0, 1]``) would otherwise surface as
    an opaque unpacking error deep inside ``__getitem__``, or as a
    step that silently always — or never — fires.

    An already-built :class:`AugmentationStep` goes through the same
    checks rather than being trusted: the named-tuple constructor does
    no validation of its own, so the two ways of spelling a step would
    otherwise disagree about what a valid step is.
    """
    try:
        fn, prob, kwargs = step  # type: ignore[misc]
    except (TypeError, ValueError):
        raise ValueError(
            f"augmentations[{index}] must be a (fn, probability, kwargs) "
            f"triple, got {step!r}") from None
    if not callable(fn):
        raise ValueError(
            f"augmentations[{index}][0] must be callable, got {fn!r}")
    try:
        prob = float(prob)
    except (TypeError, ValueError):
        raise ValueError(
            f"augmentations[{index}][1] must be a probability in [0, 1], "
            f"got {prob!r}") from None
    if not 0.0 <= prob <= 1.0:
        raise ValueError(
            f"augmentations[{index}][1] must be a probability in [0, 1], "
            f"got {prob!r}")
    if not isinstance(kwargs, dict):
        raise ValueError(
            f"augmentations[{index}][2] must be a dict of kwargs, "
            f"got {kwargs!r}")
    return AugmentationStep(fn, prob, kwargs)


def _step_name(fn: Callable) -> str:
    """Readable name for a step function, for records and ``repr``.

    A step is any callable, and the two natural ways to write one with
    baked-in configuration — :func:`functools.partial` and a callable
    instance — carry no ``__name__``.  Unwrap to the underlying
    function where there is one, and fall back to the class name.
    """
    name = getattr(fn, "__name__", None)
    if name is not None:
        return name
    wrapped = getattr(fn, "func", None)      # functools.partial
    if wrapped is not None:
        return _step_name(wrapped)
    return type(fn).__name__


class AugmentationPipeline:
    """Composable sequence of augmentations with per-step probabilities.

    Each augmentation is a tuple of ``(fn, probability, kwargs)`` where
    *fn* has signature ``fn(root_pos, joint_data, **kwargs)`` and
    returns ``(new_root_pos, new_joint_data)`` — root first, matching
    pybvh's ``Bvh.from_*`` / ``Bvh.to_*`` convention.

    Kwargs values may be **callables** of the form ``lambda rng: value``,
    which are resolved at each invocation using the pipeline's rng.
    This enables random parameter sampling per sample (e.g., random
    rotation angles).  Pass ``return_params=True`` to get back what each
    call actually drew — which steps fired and with which sampled
    values (see ``__call__``).

    The pipeline automatically forwards its ``rng`` to augmentation
    functions that accept an ``rng`` parameter (detected via
    signature inspection).  This ensures reproducibility without
    requiring ``"rng": lambda rng: rng`` in kwargs.

    Parameters
    ----------
    augmentations : list of (callable, float, dict)
        Each entry is ``(fn, probability, kwargs)``.
        *probability* is in ``[0, 1]``; the augmentation is applied
        when a uniform draw is below this threshold.  Entries are
        stored as :class:`AugmentationStep` named tuples, so
        ``pipeline.augmentations[i].kwargs`` and the plain
        ``pipeline.augmentations[i][2]`` both work.
    representation : str, optional
        Pipeline-level default for the ``representation`` kwarg.  A
        pipeline is homogeneous in practice, and repeating the token on
        every step is where one step in five ends up disagreeing with
        the rest.  Steps that declare their own ``representation`` keep
        it — the default only fills in for those that don't (and only
        for functions that *name* the parameter; a ``**kwargs``
        catch-all does not count, so custom steps taking neither are
        called with exactly their own kwargs).  Also satisfies the
        ``cache_quats=True`` requirement that *something* declare what
        ``joint_data`` is in.

        Built-in steps must agree on the resulting representation:
        each step's output is the next one's input, so two built-ins
        declaring different tokens with nothing between them to convert
        raises at construction.  A custom step in between lifts the
        restriction, since it may legitimately convert.
    euler_orders : list of str, optional
        Pipeline-level default for the ``euler_orders`` kwarg, with the
        same per-step-override semantics.  Needed only when
        ``representation="euler"``.
    cache_quats : bool, default True
        Share a quaternion cache across pybvh-ml's built-in
        augmentations.  Functions like :func:`add_joint_noise` and
        :func:`speed_perturbation_arrays` always operate in quaternion
        space internally; when a pipeline strings several of them
        together with ``representation="axisangle"`` or ``"euler"``,
        this flag eliminates all but the first and last conversion —
        typically a 2–3× speedup on non-6d pipelines, 1.5× on 6d.
        User-defined augmentations not registered in the internal
        staging table are supported transparently: the cache is
        flushed around them and they receive ``joint_data`` in their
        declared ``representation`` kwarg — or, when they declare
        none, in the pipeline's current declared representation (the
        most recent step carrying a ``representation`` kwarg), exactly
        as on the ``cache_quats=False`` path.  Set to ``False`` for
        historical bit-exact behavior.

    Notes
    -----
    **Composition order matters.** Steps run left-to-right on the
    output of the previous step.  The mathematically interesting
    interactions to be aware of:

    * **Mirror vs. vertical rotation.** ``mirror_*`` reflects the
      lateral axis; ``rotate_*_vertical`` rotates around the up
      axis.  The two commute up to a sign flip on the rotation
      angle (``mirror ∘ rotate(θ)  ==  rotate(-θ) ∘ mirror``).
      Either order is correct, but if you stack them with
      probabilities and expect the resulting distribution to be
      symmetric, keep that sign flip in mind.
    * **Speed perturbation changes F.** ``speed_perturbation_arrays``
      resamples time, so downstream steps receive arrays with a
      different ``F``.  Frame-count-sensitive steps (e.g.
      ``dropout_arrays`` with a fixed keep-mask) should run before
      speed perturbation or be written to tolerate variable
      lengths.
    * **Noise + re-augmentation.** ``add_joint_noise`` perturbs
      rotations in place (via quaternion space); following it with a
      second rotation-space step is fine, but a subsequent
      deterministic check (e.g. equality to the input) will naturally
      fail.

    Examples
    --------
    >>> from pybvh_ml.augmentation import rotate_vertical, mirror
    >>> pipeline = AugmentationPipeline([
    ...     (rotate_vertical, 1.0, {
    ...         "angle": lambda rng: rng.uniform(-np.pi, np.pi),
    ...         "up_axis": bvh.world_up,
    ...     }),
    ...     (mirror, 0.5, {
    ...         "lr_joint_pairs": pairs,
    ...         "lateral_axis": "+x",
    ...     }),
    ... ], representation="6d")
    >>> new_pos, new_rot6d = pipeline(
    ...     root_pos=root_pos, joint_data=joint_rot6d, rng=rng)
    """

    def __init__(
        self,
        augmentations: list[tuple[Callable, float, dict]],
        cache_quats: bool = True,
        *,
        representation: str | None = None,
        euler_orders: list[str] | None = None,
    ) -> None:
        self.augmentations = [
            _coerce_step(step, i) for i, step in enumerate(augmentations)]
        self.cache_quats = cache_quats
        self.representation = representation
        self.euler_orders = euler_orders
        self._reject_conflicting_representations()

    def _reject_conflicting_representations(self) -> None:
        """Reject built-in steps that disagree about the representation.

        Two built-in steps declaring different representations with
        nothing between them to convert is always a bug, and a quiet
        one: the staged path converts through its quat cache and gets a
        sensible answer, while the direct path hands the second step the
        first one's output *reinterpreted* under a different token —
        garbage.  ``cache_quats`` is meant to be an optimization, so a
        configuration where it changes results has to fail instead.

        A custom (unregistered) step between them lifts the check: it
        may legitimately convert representations mid-pipeline, which is
        precisely why the pipeline hands it the declared representation.
        """
        previous: tuple[int, str] | None = None
        for index, step in enumerate(self.augmentations):
            if step.fn not in STAGED_DISPATCH:
                previous = None       # may convert; stop comparing across it
                continue
            declared = step.kwargs.get("representation", self.representation)
            if declared is None:
                continue
            if previous is not None and previous[1] != declared:
                prev_index, prev_repr = previous
                raise ValueError(
                    f"augmentations[{prev_index}] declares "
                    f"representation={prev_repr!r} but augmentations"
                    f"[{index}] declares {declared!r}; a pipeline's "
                    f"built-in steps must agree, since each one's output "
                    f"is the next one's input. Convert explicitly with a "
                    f"custom step between them if that is really the "
                    f"intent, or declare one representation for the "
                    f"pipeline.")
            previous = (index, declared)

    @classmethod
    def standard(
        cls,
        skeleton_info: dict,
        *,
        representation: str = "6d",
        up_axis: str = "+y",
        lateral_axis: str = "+x",
        rotate_angle_range: tuple[float, float] | None = (-np.pi, np.pi),
        mirror_prob: float = 0.5,
        noise_sigma: float | None = np.radians(1.0),
        speed_factor_range: tuple[float, float] | None = (0.8, 1.2),
        cache_quats: bool = True,
    ) -> "AugmentationPipeline":
        """Build the canonical rotate + mirror + noise + speed pipeline.

        Convenience factory that wires the four common augmentation
        steps from a ``skeleton_info`` dict (as returned by
        :func:`pybvh_ml.skeleton.get_skeleton_info` or
        :func:`pybvh_ml.preprocessing.load_preprocessed`) so callers
        don't reassemble the boilerplate for every project.

        Each step is optional: pass ``None`` (or ``0`` for
        ``mirror_prob``) to skip it.  For anything beyond what these
        kwargs expose, build the pipeline directly with the
        ``(fn, prob, kwargs)`` constructor — this factory is the
        opinionated common case, not a wrapper around every knob.

        Parameters
        ----------
        skeleton_info : dict
            Supplies ``lr_pairs`` (required for mirror) and
            ``euler_orders`` (required when
            ``representation="euler"``).
        representation : str
            Rotation representation threaded through every step.
            One of ``"quat"``, ``"6d"``, ``"axisangle"``,
            ``"rotmat"``, ``"euler"``.
        up_axis, lateral_axis : str
            Signed-axis strings (e.g. ``"+y"``, ``"+x"``).  The
            defaults assume a ``+y``-up, ``+x``-lateral skeleton;
            set from ``bvh.world_up`` and the dataset's lateral
            convention otherwise.
        rotate_angle_range : (float, float) or None
            Random yaw range in radians; ``None`` skips rotation.
        mirror_prob : float
            Probability of left/right mirror.  ``0`` skips it.
            Silently skipped when ``skeleton_info["lr_pairs"]`` is
            empty (no pairs detected on this skeleton).
        noise_sigma : float or None
            Per-joint rotation noise standard deviation in radians
            (default one degree); ``None`` skips noise.
        speed_factor_range : (float, float) or None
            Random speed factor range; ``None`` skips speed
            perturbation.  Runs last because it changes ``F``.
        cache_quats : bool
            Passed through to the pipeline constructor.
        """
        # Local import to keep the pipeline module free of a hard
        # dependency cycle with augmentation at import time.
        from .augmentation import (
            add_joint_noise,
            mirror as mirror_fn,
            rotate_vertical,
            speed_perturbation_arrays,
        )

        euler_orders = skeleton_info.get("euler_orders")
        lr_pairs = skeleton_info.get("lr_pairs") or []

        # representation / euler_orders are pipeline-level here rather
        # than repeated on every step — the factory builds exactly the
        # homogeneous pipeline those defaults exist for.
        steps: list[tuple[Callable, float, dict]] = []

        if rotate_angle_range is not None:
            lo, hi = rotate_angle_range
            steps.append((rotate_vertical, 1.0, {
                "angle": lambda rng, lo=lo, hi=hi: rng.uniform(lo, hi),
                "up_axis": up_axis,
            }))

        if mirror_prob > 0 and lr_pairs:
            steps.append((mirror_fn, mirror_prob, {
                "lr_joint_pairs": lr_pairs,
                "lateral_axis": lateral_axis,
            }))

        if noise_sigma is not None:
            steps.append((add_joint_noise, 1.0, {"sigma": noise_sigma}))

        if speed_factor_range is not None:
            lo, hi = speed_factor_range
            steps.append((speed_perturbation_arrays, 1.0, {
                "factor": lambda rng, lo=lo, hi=hi: rng.uniform(lo, hi),
            }))

        return cls(steps, cache_quats=cache_quats,
                   representation=representation, euler_orders=euler_orders)

    def __call__(
        self,
        *,
        root_pos: npt.NDArray[np.float64],
        joint_data: npt.NDArray[np.float64],
        rng: np.random.Generator | None = None,
        return_params: bool = False,
    ) -> tuple:
        """Apply augmentations with their configured probabilities.

        All arguments are keyword-only.  ``root_pos`` and ``joint_data``
        are shape-compatible ndarrays; refusing positional binding
        prevents a silent-corruption swap.

        Parameters
        ----------
        root_pos : ndarray, shape (F, 3)
        joint_data : ndarray
            Joint rotation data (any representation).
        rng : numpy Generator, optional
            Random number generator.  Defaults to a new unseeded one.
        return_params : bool, default False
            Also return what this call drew (see *params* below).
            Purely additive: the random stream is untouched, so a given
            ``rng`` produces identical arrays either way.

        Returns
        -------
        new_root_pos : ndarray
        new_joint_data : ndarray
            Always freshly allocated — the outputs never alias the input arrays, even when no augmentation fires.
        params : list of dict
            Only when ``return_params=True``.  One record per configured
            step, in pipeline order (index-aligned with the
            ``augmentations`` list), each shaped
            ``{"name": str, "applied": bool, "params": dict}``.
            ``applied`` is the outcome of the step's probability draw.
            ``params`` holds the kwargs this call *sampled* — those whose
            spec is a callable — resolved to the values the augmentation
            received.  Static kwargs are pipeline configuration, readable
            from ``augmentations`` (or ``repr``), and ``rng`` is
            machinery rather than a parameter; neither appears here.  A
            step that did not fire reports ``{}``: its callables are
            never invoked, which is what keeps the random stream
            identical.

        Examples
        --------
        >>> new_pos, new_rot, steps = pipeline(
        ...     root_pos=root_pos, joint_data=joint_data, rng=rng,
        ...     return_params=True)
        >>> [(s["name"], s["applied"]) for s in steps]
        [('rotate_vertical', True), ('mirror', False)]
        >>> steps[0]["params"]["angle"]
        1.8721...
        """
        if rng is None:
            rng = np.random.default_rng()
        # Validate at entry on both paths.  The staged ops bypass the
        # public functions' own checks, and the direct path would
        # otherwise only raise if some step happened to fire — making a
        # mismatch a stochastic error under p < 1 steps, and no error at
        # all when nothing fires.
        _validate_frame_counts(root_pos, joint_data)

        call = self._call_staged if self.cache_quats else self._call_direct
        new_root_pos, new_joint_data, params = call(root_pos, joint_data, rng)

        # When no step fires (and, staged, no representation change runs)
        # both paths would hand the inputs straight through; copy on that
        # fall-through so callers can always mutate the outputs safely.
        if new_root_pos is root_pos:
            new_root_pos = root_pos.copy()
        if new_joint_data is joint_data:
            new_joint_data = joint_data.copy()
        if return_params:
            return new_root_pos, new_joint_data, params
        return new_root_pos, new_joint_data

    def _resolve_step(
        self,
        fn: Callable,
        prob: float,
        kwargs: dict,
        rng: np.random.Generator,
    ) -> tuple[bool, dict, dict]:
        """Draw one step's probability and resolve its per-sample kwargs.

        Shared by both call paths so their draw order — and the records
        they report — cannot drift apart.  Callables are resolved only
        when the step fires: resolving them for a skipped step would
        consume rng draws and change every downstream result.

        Returns ``(applied, resolved_kwargs, record)``.
        """
        record = {"name": _step_name(fn), "applied": False, "params": {}}
        if rng.random() >= prob:
            return False, {}, record

        resolved = {k: v(rng) if callable(v) else v for k, v in kwargs.items()}
        record["applied"] = True
        record["params"] = {k: resolved[k] for k, spec in kwargs.items()
                            if callable(spec) and k != "rng"}
        return True, resolved, record

    @staticmethod
    def _forward_rng(fn: Callable, resolved: dict,
                     rng: np.random.Generator) -> None:
        """Give *fn* the pipeline's rng when it takes one and none was set."""
        if "rng" in resolved:
            return
        if "rng" in inspect.signature(fn).parameters:
            resolved["rng"] = rng

    def _apply_defaults(self, fn: Callable, resolved: dict) -> None:
        """Fill the pipeline-level kwargs this step didn't declare itself.

        Only parameters *fn* declares by name are filled, so a custom
        step that takes neither ``representation`` nor ``euler_orders``
        is still called with exactly its own kwargs.  A ``**kwargs``
        catch-all does *not* count as declaring them: it means the
        function tolerates unknown keys, not that it wants these, and
        injecting on that basis would push arguments into every such
        step.  Declare the parameter explicitly to opt in — the same
        rule governs the ``rng`` forwarding in :meth:`_forward_rng`.

        Costs nothing when no pipeline-level default is set (the common
        case), which keeps the per-sample signature inspection off the
        hot path.
        """
        missing = [
            (name, value)
            for name, value in (("representation", self.representation),
                                ("euler_orders", self.euler_orders))
            if value is not None and name not in resolved
        ]
        if not missing:
            return
        params = inspect.signature(fn).parameters
        for name, value in missing:
            if name in params:
                resolved[name] = value

    def _call_direct(
        self,
        root_pos: npt.NDArray[np.float64],
        joint_data: npt.NDArray[np.float64],
        rng: np.random.Generator,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], list[dict]]:
        """Legacy path: each step converts to/from quat independently.

        Used when ``cache_quats=False`` or as a reference for tests
        that want historical bit-exact output.
        """
        records: list[dict] = []
        for fn, prob, kwargs in self.augmentations:
            applied, resolved, record = self._resolve_step(
                fn, prob, kwargs, rng)
            records.append(record)
            if not applied:
                continue
            self._apply_defaults(fn, resolved)
            self._forward_rng(fn, resolved, rng)
            root_pos, joint_data = fn(
                root_pos=root_pos, joint_data=joint_data, **resolved)

        return root_pos, joint_data, records

    def _call_staged(
        self,
        root_pos: npt.NDArray[np.float64],
        joint_data: npt.NDArray[np.float64],
        rng: np.random.Generator,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], list[dict]]:
        """Quat-caching path: share one quaternion view across compatible steps.

        Steps whose function is in :data:`pybvh_ml._staged.STAGED_DISPATCH`
        operate on a shared :class:`_StagingState` that carries a quat
        cache forward.  Unknown functions (e.g. user-defined) fall back
        transparently — the cache is flushed, the function sees a fresh
        ``joint_data`` in its declared representation (or, when it
        declares none, in the pipeline's current declared representation
        — the same array the direct path would pass), and staging
        resumes cold after the call.
        """
        if not self.augmentations:
            return root_pos, joint_data, []

        # Initial representation is whatever the first step declares
        # (a pipeline with steps but no declared representation raises —
        # staging cannot guess what joint_data is).  The representation
        # we report back to the caller at the end comes from the *last*
        # step that carries a "representation" kwarg.
        initial_repr = self._initial_representation()
        euler_orders = self._first_euler_orders()
        state = _StagingState(joint_data, initial_repr, euler_orders)

        final_repr = initial_repr
        records: list[dict] = []

        for fn, prob, kwargs in self.augmentations:
            applied, resolved, record = self._resolve_step(
                fn, prob, kwargs, rng)
            records.append(record)
            if not applied:
                continue
            self._apply_defaults(fn, resolved)

            # Track the representation the user wants at the end.
            step_repr = resolved.get("representation")
            if step_repr is not None:
                final_repr = step_repr

            # Keep euler_orders in sync for state-level conversions.
            if "euler_orders" in resolved and resolved["euler_orders"] is not None:
                state.euler_orders = resolved["euler_orders"]

            staged_fn = STAGED_DISPATCH.get(fn)
            if staged_fn is not None:
                self._forward_rng(staged_fn, resolved, rng)
                root_pos = staged_fn(root_pos, state, **resolved)
            else:
                # Fallback: flush the cache and hand the unknown step
                # joint_data in the pipeline's current declared
                # representation (``final_repr`` — the step's own
                # declaration when present, else the most recent one).
                # This is exactly what the direct path would carry, so
                # cache_quats=True/False agree for custom steps that
                # don't declare a representation.
                state.ensure_repr(final_repr)
                self._forward_rng(fn, resolved, rng)
                root_pos, new_jd = fn(
                    root_pos=root_pos, joint_data=state.jd, **resolved)
                # We don't know what the unknown function did internally;
                # treat the result as opaque data still in final_repr.
                state.set_jd_invalidate_quats(new_jd, final_repr)

        # At the end, ensure joint_data is back in the representation
        # the user expects.
        state.ensure_repr(final_repr)
        return root_pos, state.jd, records

    def _initial_representation(self) -> str:
        """Representation ``joint_data`` arrives in.

        The pipeline-level default when set, otherwise the first step
        that declares one.

        Raises
        ------
        ValueError
            If neither is declared.  The staged path needs to know what
            representation ``joint_data`` is in to manage its quaternion
            cache; guessing silently would corrupt data for non-quat
            inputs.
        """
        if self.representation is not None:
            return self.representation
        for step in self.augmentations:
            v = step.kwargs.get("representation")
            if isinstance(v, str):
                return v
        raise ValueError(
            "No representation declared. The quat-caching path "
            "(cache_quats=True) needs it to know what representation "
            "joint_data is in — pass representation=... to "
            "AugmentationPipeline, declare it on at least one step, or "
            "build the pipeline with cache_quats=False.")

    def _first_euler_orders(self) -> list[str] | None:
        """Pipeline-level ``euler_orders``, else the first step's."""
        if self.euler_orders is not None:
            return self.euler_orders
        for step in self.augmentations:
            v = step.kwargs.get("euler_orders")
            if v is not None and not callable(v):
                return v
        return None

    def __len__(self) -> int:
        return len(self.augmentations)

    def __repr__(self) -> str:
        steps = [
            f"  ({_step_name(fn)}, p={prob}, kwargs={kwargs})"
            for fn, prob, kwargs in self.augmentations
        ]
        defaults = "".join(
            f", {name}={value!r}"
            for name, value in (("representation", self.representation),
                                ("euler_orders", self.euler_orders))
            if value is not None
        )
        return "AugmentationPipeline([\n" + "\n".join(steps) + f"\n]{defaults})"
