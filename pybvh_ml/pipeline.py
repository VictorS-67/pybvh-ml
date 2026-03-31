"""Composable augmentation pipeline for ML training.

Designed to be called inside a PyTorch Dataset's ``__getitem__``
or any data loading loop.
"""
from __future__ import annotations

import inspect
from typing import Callable

import numpy as np
import numpy.typing as npt


class AugmentationPipeline:
    """Composable sequence of augmentations with per-step probabilities.

    Each augmentation is a tuple of ``(fn, probability, kwargs)`` where
    *fn* has signature ``fn(joint_data, root_pos, **kwargs)`` and
    returns ``(new_joint_data, new_root_pos)``.

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

    Examples
    --------
    >>> from pybvh_ml.augmentation import rotate_quaternions_vertical, mirror_quaternions
    >>> pipeline = AugmentationPipeline([
    ...     (rotate_quaternions_vertical, 1.0, {
    ...         "angle_deg": lambda rng: rng.uniform(-180, 180),
    ...         "up_idx": 1,
    ...     }),
    ...     (mirror_quaternions, 0.5, {"lr_joint_pairs": pairs, "lateral_idx": 0}),
    ... ])
    >>> new_quats, new_pos = pipeline(joint_quats, root_pos, rng=rng)
    """

    def __init__(
        self,
        augmentations: list[tuple[Callable, float, dict]],
    ) -> None:
        self.augmentations = augmentations

    def __call__(
        self,
        joint_data: npt.NDArray[np.float64],
        root_pos: npt.NDArray[np.float64],
        rng: np.random.Generator | None = None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Apply augmentations with their configured probabilities.

        Parameters
        ----------
        joint_data : ndarray
            Joint rotation data (any representation).
        root_pos : ndarray, shape (F, 3)
        rng : numpy Generator, optional
            Random number generator.  Defaults to a new unseeded one.

        Returns
        -------
        new_joint_data : ndarray
        new_root_pos : ndarray
        """
        if rng is None:
            rng = np.random.default_rng()

        for fn, prob, kwargs in self.augmentations:
            if rng.random() < prob:
                resolved = {
                    k: v(rng) if callable(v) else v
                    for k, v in kwargs.items()
                }
                # Forward rng if the function accepts it and the user
                # didn't already provide it via kwargs.
                if "rng" not in resolved:
                    sig = inspect.signature(fn)
                    if "rng" in sig.parameters:
                        resolved["rng"] = rng
                joint_data, root_pos = fn(joint_data, root_pos, **resolved)

        return joint_data, root_pos

    def __len__(self) -> int:
        return len(self.augmentations)

    def __repr__(self) -> str:
        steps = [
            f"  ({fn.__name__}, p={prob}, kwargs={kwargs})"
            for fn, prob, kwargs in self.augmentations
        ]
        return f"AugmentationPipeline([\n" + "\n".join(steps) + "\n])"
