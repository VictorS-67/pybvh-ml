"""Composable augmentation pipeline for ML training.

Designed to be called inside a PyTorch Dataset's ``__getitem__``
or any data loading loop.
"""
from __future__ import annotations

import inspect
from typing import Callable, Mapping, NamedTuple

import numpy as np
import numpy.typing as npt

from ._staged import STAGED_DISPATCH, _StagingState
from .arrays import STREAM_NAMES, MotionArrays
# _step_label is the shared "readable name for any callable step" —
# records, repr and the precondition messages must all spell a step the
# same way, so it lives with the checks that fire first.
from .augmentation import _check_step_preconditions, _step_label as _step_name


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


def _finish(
    result: npt.NDArray[np.float64],
    entry: npt.NDArray[np.float64],
    dtype: np.dtype,
) -> npt.NDArray:
    """Return *result* in *dtype*, never sharing storage with the input.

    Both properties in one place because they interact: a cast allocates,
    so it doubles as the defensive copy, and an explicit one is needed
    only when there is nothing to cast *and* a call path handed its own
    input straight back — which is what happens when no step fires, and
    also when a step passes an untouched stream through by reference.

    The test is ``shares_memory`` rather than identity: the pipeline
    wraps its ``float64`` entry arrays in a container, whose fields are
    read-only *views*, so an untouched stream comes back as a different
    object over the very same buffer.
    """
    if result.dtype != dtype:
        return result.astype(dtype)
    return result.copy() if np.shares_memory(result, entry) else result


def _call_step(
    fn: Callable,
    arrays: MotionArrays,
    resolved: dict,
) -> MotionArrays:
    """Invoke one step and check it honoured the step contract.

    A step is ``fn(arrays, **kwargs) -> MotionArrays``.  Steps written
    against pybvh-ml <= 0.4 took ``(root_pos=, joint_data=)`` and
    returned a 2-tuple; both halves of that older contract are detected
    here and reported as a migration, because the natural failure
    otherwise is an ``AttributeError`` on a tuple several frames away
    from the step that caused it.
    """
    try:
        out = fn(arrays, **resolved)
    except TypeError as exc:
        if _looks_like_legacy_step(fn):
            raise TypeError(
                f"augmentation step {_step_name(fn)!r} appears to use the "
                f"pre-0.5.0 signature (root_pos=..., joint_data=...). Steps "
                f"now take a MotionArrays positionally and return one: "
                f"def step(arrays, **kwargs) -> MotionArrays. Read "
                f"arrays.root_pos / arrays.joint_rot inside, and return "
                f"arrays.replace(joint_rot=...)") from exc
        raise
    if not isinstance(out, MotionArrays):
        raise TypeError(
            f"augmentation step {_step_name(fn)!r} returned "
            f"{type(out).__name__}, expected MotionArrays. Steps returning "
            f"a (root_pos, joint_data) tuple are the pre-0.5.0 contract; "
            f"return arrays.replace(joint_rot=...) instead.")
    dropped = sorted(arrays.present_streams - out.present_streams)
    added = sorted(out.present_streams - arrays.present_streams)
    if dropped or added:
        raise ValueError(
            f"augmentation step {_step_name(fn)!r} changed which streams "
            f"the sample carries (dropped {dropped}, added {added}). A step "
            f"transforms the streams it is given; a pipeline never carries "
            f"a stream a step left behind, and never gains one mid-run. "
            f"Return arrays.replace(...) so the untouched streams travel "
            f"with it.")
    return out


def _looks_like_legacy_step(fn: Callable) -> bool:
    """True when *fn* declares the pre-0.5.0 ``root_pos`` / ``joint_data``."""
    try:
        params = inspect.signature(fn).parameters
    except (ValueError, TypeError):
        return False
    return "root_pos" in params or "joint_data" in params


class AugmentationPipeline:
    """Composable sequence of augmentations with per-step probabilities.

    Each augmentation is a tuple of ``(fn, probability, kwargs)`` where
    *fn* has signature ``fn(arrays, **kwargs) -> MotionArrays`` —
    :class:`~pybvh_ml.MotionArrays` in, a new one out.  Custom steps
    read ``arrays.root_pos`` / ``arrays.joint_rot`` and typically return
    ``arrays.replace(joint_rot=...)``.

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
        augmentations.  Functions like :func:`add_joint_rotation_noise` and
        :func:`speed_perturbation_arrays` always operate in quaternion
        space internally; when a pipeline strings several of them
        together with ``representation="axisangle"`` or ``"euler"``,
        this flag eliminates all but the first and last conversion —
        typically a 2–3× speedup on non-6d pipelines, 1.5× on 6d.
        User-defined augmentations not registered in the internal
        staging table are supported transparently: the cache is
        flushed around them and they receive ``joint_rot`` in their
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
    * **Noise + re-augmentation.** ``add_joint_rotation_noise`` perturbs
      rotations in place (via quaternion space); following it with a
      second rotation-space step is fine, but a subsequent
      deterministic check (e.g. equality to the input) will naturally
      fail.
    * **A re-derivation discards the position stream's own history.**
      On a sample carrying positions, ``add_joint_rotation_noise``
      *replaces* them with forward kinematics of the noised rotations
      rather than transforming the incoming ones (see
      :func:`~pybvh_ml.handles_streams`).  So on a rig with asymmetric
      rest offsets, ``[mirror, add_joint_rotation_noise]`` ends with FK
      of locally-mirrored rotations — throwing away the world-exact
      reflection the position stream held — while
      ``[add_joint_rotation_noise, mirror]`` keeps it.  Both are
      defensible and neither is a bug, but the two do not produce the
      same positions.

    **Every step must handle every stream the sample carries.**  The
    check runs once at ``__call__`` entry, for every configured step,
    before any of them fires — a ``p=0.1`` step with the wrong stream
    support or a missing ``fk_topology`` would otherwise raise on one
    sample in ten.  A custom step that declares nothing is assumed to
    handle ``{"root_pos", "joint_rot"}``; decorate it with
    :func:`~pybvh_ml.handles_streams` once it transforms positions too.

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
    >>> out = pipeline(MotionArrays(root_pos=root_pos,
    ...                                 joint_rot=joint_rot6d), rng=rng)
    >>> out.joint_rot.shape
    (120, 31, 6)
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
        representation: str | None = "6d",
        up_axis: str = "+y",
        lateral_axis: str = "+x",
        rotate_angle_range: tuple[float, float] | None = (-np.pi, np.pi),
        mirror_prob: float = 0.5,
        noise_sigma: float | None = np.radians(1.0),
        position_noise_sigma: float | None = None,
        position_space: str | None = None,
        speed_factor_range: tuple[float, float] | None = (0.8, 1.2),
        degrees: bool = False,
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
            Supplies ``lr_pairs`` / ``node_lr_pairs`` (required for
            mirror), ``euler_orders`` (required when
            ``representation="euler"``), and — for a positions-carrying
            dataset — ``fk_topology``, ``world_up`` and
            ``position_space``.
        representation : str or None
            Rotation representation threaded through every step.
            One of ``"quat"``, ``"6d"``, ``"axisangle"``,
            ``"rotmat"``, ``"euler"``.  ``None`` builds a positions-only
            pipeline and **skips the rotation-noise step**, which would
            otherwise be configured by default and refuse every sample:
            noising rotations is meaningless on a clip that has none.
            (A *direct* :func:`~pybvh_ml.add_joint_rotation_noise` call
            on such a sample still raises — a factory declining to
            configure a meaningless step and a function refusing a
            meaningless call are different questions.)
        up_axis, lateral_axis : str
            Signed-axis strings (e.g. ``"+y"``, ``"+x"``).  The
            defaults assume a ``+y``-up, ``+x``-lateral skeleton;
            set from ``bvh.world_up`` and the dataset's lateral
            convention otherwise.
        rotate_angle_range : (float, float) or None
            Random yaw range in radians (degrees when ``degrees=True``);
            ``None`` skips rotation.
        degrees : bool
            Interpret ``rotate_angle_range`` and ``noise_sigma`` in
            degrees.  Default False (radians).  One flag serves both
            because both are angles — which is exactly why root-position
            noise is not a knob on this factory: its sigma is a length,
            and a single flag could not have covered it.  Use
            :func:`~pybvh_ml.add_root_position_noise` as an explicit
            step for that.
        mirror_prob : float
            Probability of left/right mirror.  ``0`` skips it.
            Silently skipped when the pair list this configuration needs
            is empty (``skeleton_info["lr_pairs"]`` for the joint-space
            streams, ``node_lr_pairs`` for ``node_pos``) — no pairs were
            detected on this skeleton.
        noise_sigma : float or None
            Per-joint rotation noise standard deviation in radians
            (default one degree); ``None`` skips noise.  On a dataset
            carrying positions this step also refreshes them by forward
            kinematics, so the factory wires ``fk_topology`` and
            ``world_up`` from *skeleton_info*.
        position_noise_sigma : float or None
            Per-vertex keypoint jitter, in the data's positional units;
            ``None`` (default) skips it.  Joint-space and node-space
            jitter are *different functions* with different stream
            declarations, and the pipeline is built before any sample is
            seen, so the index space is resolved here — from
            ``position_space`` when given, else
            ``skeleton_info["position_space"]``.  Wiring one
            unconditionally would make the pipeline refuse every sample
            of a dataset stored in the other space.
        position_space : {"joint", "node"} or None
            Explicit override for that resolution.  ``None`` (default)
            reads ``skeleton_info["position_space"]``.
        speed_factor_range : (float, float) or None
            Random speed factor range; ``None`` skips speed
            perturbation.  Runs last because it changes ``F``.
        cache_quats : bool
            Passed through to the pipeline constructor.
        """
        # Local import to keep the pipeline module free of a hard
        # dependency cycle with augmentation at import time.
        from .augmentation import (
            add_joint_position_noise,
            add_joint_rotation_noise,
            add_node_position_noise,
            mirror as mirror_fn,
            rotate_vertical,
            speed_perturbation_arrays,
        )
        from .skeleton import build_fk_topology

        euler_orders = skeleton_info.get("euler_orders")
        lr_pairs = skeleton_info.get("lr_pairs") or []
        node_lr_pairs = skeleton_info.get("node_lr_pairs") or []
        space = (position_space if position_space is not None
                 else skeleton_info.get("position_space"))
        if space is not None and space not in ("joint", "node"):
            raise ValueError(
                f"position_space must be 'joint' or 'node', got {space!r} "
                f"(from {'the position_space argument' if position_space else 'skeleton_info'})")

        # representation / euler_orders are pipeline-level here rather
        # than repeated on every step — the factory builds exactly the
        # homogeneous pipeline those defaults exist for.
        steps: list[tuple[Callable, float, dict]] = []

        if rotate_angle_range is not None:
            lo, hi = rotate_angle_range
            steps.append((rotate_vertical, 1.0, {
                "angle": lambda rng, lo=lo, hi=hi: rng.uniform(lo, hi),
                "up_axis": up_axis,
                "degrees": degrees,
            }))

        # Which pair lists this configuration will actually need: the
        # joint-space one for rotations and joint positions, the
        # node-space one for node positions.  A missing list is the
        # documented "no pairs detected" skip, not a hard error.
        required_pairs = []
        if representation is not None or space == "joint":
            required_pairs.append(lr_pairs)
        if space == "node":
            required_pairs.append(node_lr_pairs)
        if mirror_prob > 0 and required_pairs and all(required_pairs):
            steps.append((mirror_fn, mirror_prob, {
                "lr_joint_pairs": lr_pairs,
                "lr_node_pairs": node_lr_pairs or None,
                "lateral_axis": lateral_axis,
            }))

        if noise_sigma is not None and representation is not None:
            noise_kwargs: dict = {"sigma": noise_sigma, "degrees": degrees}
            if skeleton_info.get("fk_topology"):
                # Wired whenever the metadata can supply it, not only
                # when this factory expects positions: the step reads it
                # solely to refresh a position stream, and whether a
                # given *sample* carries one is not knowable here.  The
                # topology is built once, per pipeline, and pybvh
                # validates it in the constructor.
                noise_kwargs["fk_topology"] = build_fk_topology(skeleton_info)
                noise_kwargs["world_up"] = (
                    skeleton_info.get("world_up") or up_axis)
            steps.append((add_joint_rotation_noise, 1.0, noise_kwargs))

        if position_noise_sigma is not None:
            if representation is not None:
                raise ValueError(
                    "position_noise_sigma and representation cannot both be "
                    "set: keypoint jitter declines rotation-carrying samples, "
                    "because a jittered position cannot be pushed back into "
                    "the rotations beside it (that would be inverse "
                    "kinematics). Use noise_sigma for a dataset with "
                    "rotations — it re-derives the positions by forward "
                    "kinematics — or representation=None for a "
                    "positions-only one.")
            if space is None:
                raise ValueError(
                    "position_noise_sigma was given but the index space is "
                    "unknown: joint-space and node-space keypoint jitter are "
                    "different steps, and the pipeline is built before any "
                    "sample is seen. Pass position_space='joint' or 'node', "
                    "or use a skeleton_info that records it (preprocessing "
                    "with include_positions=True does).")
            steps.append((
                add_joint_position_noise if space == "joint"
                else add_node_position_noise,
                1.0, {"sigma": position_noise_sigma}))

        if speed_factor_range is not None:
            lo, hi = speed_factor_range
            steps.append((speed_perturbation_arrays, 1.0, {
                "factor": lambda rng, lo=lo, hi=hi: rng.uniform(lo, hi),
            }))

        return cls(steps, cache_quats=cache_quats,
                   representation=representation, euler_orders=euler_orders)

    def __call__(
        self,
        arrays: MotionArrays,
        *,
        rng: np.random.Generator | None = None,
        return_params: bool = False,
    ) -> MotionArrays | tuple[MotionArrays, list[dict]]:
        """Apply augmentations with their configured probabilities.

        Parameters
        ----------
        arrays : MotionArrays
            The clip to augment.  Positional because it is a distinct
            type — every other argument stays keyword-only.
        rng : numpy Generator, optional
            Random number generator.  Defaults to a new unseeded one.
        return_params : bool, default False
            Also return what this call drew (see *params* below).
            Purely additive: the random stream is untouched, so a given
            ``rng`` produces identical arrays either way.  This is the
            only thing that changes the return arity.

        Returns
        -------
        MotionArrays
            Always freshly allocated — the outputs never share storage with the input arrays, even when no augmentation fires. (The container's fields are read-only either way; take ``np.array(out.joint_rot)`` for a writable working array.) Each stream comes back in the dtype it went in as, with the math done in ``float64`` regardless: the dtype must not depend on which steps this sample's probability draws happened to fire, and both call paths have to agree bit for bit.
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
        >>> out, steps = pipeline(arrays, rng=rng, return_params=True)
        >>> [(s["name"], s["applied"]) for s in steps]
        [('rotate_vertical', True), ('mirror', False)]
        >>> steps[0]["params"]["angle"]
        1.8721...
        """
        if rng is None:
            rng = np.random.default_rng()
        if not isinstance(arrays, MotionArrays):
            raise TypeError(
                f"AugmentationPipeline takes a MotionArrays, got "
                f"{type(arrays).__name__}. The (root_pos=, joint_data=) "
                f"keyword form was replaced in 0.5.0: build the container "
                f"once with MotionArrays(root_pos=..., joint_rot=...) and "
                f"read out.root_pos / out.joint_rot from the result.")

        # Every precondition every configured step has, before any of
        # them runs.  A p=0.1 step whose kwargs are wrong would otherwise
        # raise on one sample in ten, which is a configuration error that
        # reaches production.
        for step in self.augmentations:
            _check_step_preconditions(step.fn, arrays, step.kwargs)

        # Run the whole pipeline in float64 whatever came in, and restore
        # the caller's dtypes once, at the end.  Two things depend on it:
        # the result's dtype must not depend on which steps a probability
        # draw happened to fire, and the two call paths must stay
        # bit-identical — the staged 6d fast path writes rotated columns
        # into a copy of its input, so a float32 clip would otherwise
        # have that step computed in single precision there and in double
        # on the direct path.
        entry = {
            name: (None if getattr(arrays, name) is None
                   else np.asarray(getattr(arrays, name), dtype=np.float64))
            for name in STREAM_NAMES
        }
        work = MotionArrays(
            **entry, position_centering=arrays.position_centering)

        call = self._call_staged if self.cache_quats else self._call_direct
        result, params = call(work, rng)

        out = MotionArrays(
            position_centering=arrays.position_centering,
            **{name: (None if entry[name] is None
                      else _finish(getattr(result, name), entry[name],
                                   getattr(arrays, name).dtype))
               for name in STREAM_NAMES})
        if return_params:
            return out, params
        return out

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
    def _declared_params(fn: Callable) -> Mapping[str, object]:
        """Parameter names *fn* declares, or empty when uninspectable.

        ``inspect.signature`` raises on some C-implemented callables.
        Every caller here uses the result to decide whether to *add* a
        kwarg, so an empty mapping degrades to "pass exactly what the
        step was configured with" — which is the right fallback, and
        better than propagating an error from a convenience feature.
        """
        try:
            return inspect.signature(fn).parameters
        except (ValueError, TypeError):
            return {}

    @classmethod
    def _forward_rng(cls, fn: Callable, resolved: dict,
                     rng: np.random.Generator) -> None:
        """Give *fn* the pipeline's rng when it takes one and none was set."""
        if "rng" in resolved:
            return
        if "rng" in cls._declared_params(fn):
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
        params = self._declared_params(fn)
        for name, value in missing:
            if name in params:
                resolved[name] = value

    def _call_direct(
        self,
        arrays: MotionArrays,
        rng: np.random.Generator,
    ) -> tuple[MotionArrays, list[dict]]:
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
            arrays = _call_step(fn, arrays, resolved)

        return arrays, records

    def _call_staged(
        self,
        arrays: MotionArrays,
        rng: np.random.Generator,
    ) -> tuple[MotionArrays, list[dict]]:
        """Quat-caching path: share one quaternion view across compatible steps.

        Steps whose function is in :data:`pybvh_ml._staged.STAGED_DISPATCH`
        operate on a shared :class:`_StagingState` that carries every
        stream plus a quat cache forward.  Unknown functions (e.g.
        user-defined) fall back transparently — the cache is flushed, the
        function sees a fresh ``joint_rot`` in its declared
        representation (or, when it declares none, in the pipeline's
        current declared representation — the same array the direct path
        would pass), and staging resumes cold after the call.
        """
        if not self.augmentations:
            return arrays, []

        # Initial representation is whatever the first step declares
        # (a rotation-carrying pipeline with steps but no declared
        # representation raises — staging cannot guess what joint_rot
        # is).  The representation we report back to the caller at the
        # end comes from the *last* step that carries a "representation"
        # kwarg.
        initial_repr = self._initial_representation(arrays)
        euler_orders = self._first_euler_orders()
        state = _StagingState(arrays, initial_repr, euler_orders)

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
                staged_fn(state, **resolved)
            else:
                # Fallback: flush the cache and hand the unknown step
                # the streams in the pipeline's current declared
                # representation (``final_repr`` — the step's own
                # declaration when present, else the most recent one).
                # This is exactly what the direct path would carry, so
                # cache_quats=True/False agree for custom steps that
                # don't declare a representation.
                state.ensure_repr(final_repr)
                self._forward_rng(fn, resolved, rng)
                out = _call_step(fn, state.as_arrays(), resolved)
                # We don't know what the unknown function did internally;
                # treat the result as opaque data still in final_repr.
                state.adopt(out, final_repr)

        # At the end, ensure joint_rot is back in the representation
        # the user expects.
        state.ensure_repr(final_repr)
        return state.as_arrays(), records

    def _initial_representation(self, arrays: MotionArrays) -> str | None:
        """Representation ``joint_rot`` arrives in.

        The pipeline-level default when set, otherwise the first step
        that declares one.  ``None`` for a sample carrying no rotations:
        there is nothing for the token to describe, and requiring one
        would make every positions-only pipeline (the ST-GCN case)
        declare a representation it does not have.

        Raises
        ------
        ValueError
            If the sample carries ``joint_rot`` and neither is declared.
            The staged path needs to know what representation it is in to
            manage its quaternion cache; guessing silently would corrupt
            data for non-quat inputs.
        """
        if self.representation is not None:
            return self.representation
        for step in self.augmentations:
            v = step.kwargs.get("representation")
            if isinstance(v, str):
                return v
        if arrays.joint_rot is None:
            return None
        raise ValueError(
            "No representation declared. The quat-caching path "
            "(cache_quats=True) needs it to know what representation "
            "joint_rot is in — pass representation=... to "
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
