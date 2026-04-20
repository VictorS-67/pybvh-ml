"""Composable augmentation pipeline for ML training.

Designed to be called inside a PyTorch Dataset's ``__getitem__``
or any data loading loop.
"""
from __future__ import annotations

import inspect
from typing import Callable

import numpy as np
import numpy.typing as npt

from ._staged import STAGED_DISPATCH, _StagingState


class AugmentationPipeline:
    """Composable sequence of augmentations with per-step probabilities.

    Each augmentation is a tuple of ``(fn, probability, kwargs)`` where
    *fn* has signature ``fn(root_pos, joint_data, **kwargs)`` and
    returns ``(new_root_pos, new_joint_data)`` — root first, matching
    pybvh's ``Bvh.from_*`` / ``Bvh.to_*`` convention.

    Kwargs values may be **callables** of the form ``lambda rng: value``,
    which are resolved at each invocation using the pipeline's rng.
    This enables random parameter sampling per sample (e.g., random
    rotation angles).

    The pipeline automatically forwards its ``rng`` to augmentation
    functions that accept an ``rng`` parameter (detected via
    signature inspection).  This ensures reproducibility without
    requiring ``"rng": lambda rng: rng`` in kwargs.

    Parameters
    ----------
    augmentations : list of (callable, float, dict)
        Each entry is ``(fn, probability, kwargs)``.
        *probability* is in ``[0, 1]``; the augmentation is applied
        when a uniform draw is below this threshold.
    cache_quats : bool, default True
        Share a quaternion cache across pybvh-ml's built-in
        augmentations.  Functions like :func:`add_joint_noise` and
        :func:`speed_perturbation_arrays` always operate in quaternion
        space internally; when a pipeline strings several of them
        together with ``representation="axisangle"`` or ``"euler"``,
        this flag eliminates all but the first and last conversion —
        typically a 2–3× speedup on non-6d pipelines, 1.5× on 6d.
        User-defined augmentations not registered in the internal
        staging table are supported transparently (the cache is
        flushed around them).  Set to ``False`` for historical
        bit-exact behavior.

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
    ...         "angle_deg": lambda rng: rng.uniform(-180, 180),
    ...         "up_axis": bvh.world_up,
    ...         "representation": "6d",
    ...     }),
    ...     (mirror, 0.5, {
    ...         "lr_joint_pairs": pairs,
    ...         "lateral_axis": "+x",
    ...         "representation": "6d",
    ...     }),
    ... ])
    >>> new_pos, new_rot6d = pipeline(
    ...     root_pos=root_pos, joint_data=joint_rot6d, rng=rng)
    """

    def __init__(
        self,
        augmentations: list[tuple[Callable, float, dict]],
        cache_quats: bool = True,
    ) -> None:
        self.augmentations = augmentations
        self.cache_quats = cache_quats

    @classmethod
    def standard(
        cls,
        skeleton_info: dict,
        *,
        representation: str = "6d",
        up_axis: str = "+y",
        lateral_axis: str = "+x",
        rotate_angle_range: tuple[float, float] | None = (-180.0, 180.0),
        mirror_prob: float = 0.5,
        noise_sigma_deg: float | None = 1.0,
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
            One of ``"quaternion"``, ``"6d"``, ``"axisangle"``,
            ``"rotmat"``, ``"euler"``.
        up_axis, lateral_axis : str
            Signed-axis strings (e.g. ``"+y"``, ``"+x"``).  The
            defaults assume a ``+y``-up, ``+x``-lateral skeleton;
            set from ``bvh.world_up`` and the dataset's lateral
            convention otherwise.
        rotate_angle_range : (float, float) or None
            Random yaw range in degrees; ``None`` skips rotation.
        mirror_prob : float
            Probability of left/right mirror.  ``0`` skips it.
            Silently skipped when ``skeleton_info["lr_pairs"]`` is
            empty (no pairs detected on this skeleton).
        noise_sigma_deg : float or None
            Per-joint rotation noise standard deviation in degrees;
            ``None`` skips noise.
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

        steps: list[tuple[Callable, float, dict]] = []

        if rotate_angle_range is not None:
            lo, hi = rotate_angle_range
            steps.append((rotate_vertical, 1.0, {
                "angle_deg": lambda rng, lo=lo, hi=hi: rng.uniform(lo, hi),
                "up_axis": up_axis,
                "representation": representation,
                "euler_orders": euler_orders,
            }))

        if mirror_prob > 0 and lr_pairs:
            steps.append((mirror_fn, mirror_prob, {
                "lr_joint_pairs": lr_pairs,
                "lateral_axis": lateral_axis,
                "representation": representation,
                "euler_orders": euler_orders,
            }))

        if noise_sigma_deg is not None:
            steps.append((add_joint_noise, 1.0, {
                "sigma_deg": noise_sigma_deg,
                "representation": representation,
                "euler_orders": euler_orders,
            }))

        if speed_factor_range is not None:
            lo, hi = speed_factor_range
            steps.append((speed_perturbation_arrays, 1.0, {
                "factor": lambda rng, lo=lo, hi=hi: rng.uniform(lo, hi),
                "representation": representation,
                "euler_orders": euler_orders,
            }))

        return cls(steps, cache_quats=cache_quats)

    def __call__(
        self,
        *,
        root_pos: npt.NDArray[np.float64],
        joint_data: npt.NDArray[np.float64],
        rng: np.random.Generator | None = None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
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

        Returns
        -------
        new_root_pos : ndarray
        new_joint_data : ndarray
        """
        if rng is None:
            rng = np.random.default_rng()

        if self.cache_quats:
            return self._call_staged(root_pos, joint_data, rng)
        return self._call_direct(root_pos, joint_data, rng)

    def _call_direct(
        self,
        root_pos: npt.NDArray[np.float64],
        joint_data: npt.NDArray[np.float64],
        rng: np.random.Generator,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Legacy path: each step converts to/from quat independently.

        Used when ``cache_quats=False`` or as a reference for tests
        that want historical bit-exact output.
        """
        for fn, prob, kwargs in self.augmentations:
            if rng.random() < prob:
                resolved = {
                    k: v(rng) if callable(v) else v
                    for k, v in kwargs.items()
                }
                if "rng" not in resolved:
                    sig = inspect.signature(fn)
                    if "rng" in sig.parameters:
                        resolved["rng"] = rng
                root_pos, joint_data = fn(
                    root_pos=root_pos, joint_data=joint_data, **resolved)

        return root_pos, joint_data

    def _call_staged(
        self,
        root_pos: npt.NDArray[np.float64],
        joint_data: npt.NDArray[np.float64],
        rng: np.random.Generator,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Quat-caching path: share one quaternion view across compatible steps.

        Steps whose function is in :data:`pybvh_ml._staged.STAGED_DISPATCH`
        operate on a shared :class:`_StagingState` that carries a quat
        cache forward.  Unknown functions (e.g. user-defined) fall back
        transparently — the cache is flushed, the function sees a fresh
        ``joint_data`` in its declared representation, and staging
        resumes cold after the call.
        """
        # Initial representation is whatever the first step declares;
        # if no step declares one (unusual), default to "quaternion" so
        # the state is consistent.  The representation we report back to
        # the caller at the end comes from the *last* step that carries
        # a "representation" kwarg.
        initial_repr = self._initial_representation()
        euler_orders = self._first_euler_orders()
        state = _StagingState(joint_data, initial_repr, euler_orders)

        final_repr = initial_repr

        for fn, prob, kwargs in self.augmentations:
            if rng.random() >= prob:
                continue
            resolved = {
                k: v(rng) if callable(v) else v
                for k, v in kwargs.items()
            }

            # Track the representation the user wants at the end.
            step_repr = resolved.get("representation")
            if step_repr is not None:
                final_repr = step_repr

            # Keep euler_orders in sync for state-level conversions.
            if "euler_orders" in resolved and resolved["euler_orders"] is not None:
                state.euler_orders = resolved["euler_orders"]

            staged_fn = STAGED_DISPATCH.get(fn)
            if staged_fn is not None:
                # Forward rng if the staged function accepts it and the user
                # didn't provide one in kwargs.
                if "rng" not in resolved:
                    sig = inspect.signature(staged_fn)
                    if "rng" in sig.parameters:
                        resolved["rng"] = rng
                root_pos = staged_fn(root_pos, state, **resolved)
            else:
                # Fallback: flush the cache, convert jd to the rep this
                # unknown step expects, call it normally, then reset state.
                if step_repr is not None:
                    state.ensure_repr(step_repr)
                if "rng" not in resolved:
                    sig = inspect.signature(fn)
                    if "rng" in sig.parameters:
                        resolved["rng"] = rng
                root_pos, new_jd = fn(
                    root_pos=root_pos, joint_data=state.jd, **resolved)
                # We don't know what the unknown function did internally;
                # treat the result as opaque in `step_repr` (or
                # ``state.current_repr`` if the step didn't declare one).
                state.set_jd_invalidate_quats(
                    new_jd, step_repr or state.current_repr)

        # At the end, ensure joint_data is back in the representation
        # the user expects.
        state.ensure_repr(final_repr)
        return root_pos, state.jd

    def _initial_representation(self) -> str:
        """First step's ``representation`` kwarg, or ``"quaternion"``."""
        for _, _, kwargs in self.augmentations:
            v = kwargs.get("representation")
            if isinstance(v, str):
                return v
        return "quaternion"

    def _first_euler_orders(self) -> list[str] | None:
        """First step's ``euler_orders`` value (if any, and non-callable)."""
        for _, _, kwargs in self.augmentations:
            v = kwargs.get("euler_orders")
            if v is not None and not callable(v):
                return v
        return None

    def __len__(self) -> int:
        return len(self.augmentations)

    def __repr__(self) -> str:
        steps = [
            f"  ({fn.__name__}, p={prob}, kwargs={kwargs})"
            for fn, prob, kwargs in self.augmentations
        ]
        return f"AugmentationPipeline([\n" + "\n".join(steps) + "\n])"
