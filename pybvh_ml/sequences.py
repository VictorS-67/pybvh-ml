"""Sequence length utilities for ML pipelines.

Fixed-length windows and sequence standardization — the universal
pre-processing steps between variable-length motion clips and
fixed-size model inputs.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt


def sliding_window(
    data: npt.NDArray[np.float64],
    window_size: int,
    stride: int = 1,
) -> npt.NDArray[np.float64]:
    """Extract sliding windows from a time-series array.

    Parameters
    ----------
    data : ndarray, shape (T, ...)
        Input array where axis 0 is the time dimension.
    window_size : int
        Number of frames per window.
    stride : int
        Step between consecutive window starts (default 1).

    Returns
    -------
    ndarray, shape (num_windows, window_size, ...)
        ``num_windows = (T - window_size) // stride + 1``.

    Raises
    ------
    ValueError
        If *window_size* exceeds the data length or *stride* < 1.
    """
    data = np.asarray(data)
    T = data.shape[0]

    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if window_size > T:
        raise ValueError(
            f"window_size ({window_size}) exceeds data length ({T})")

    num_windows = (T - window_size) // stride + 1
    shape = (num_windows, window_size) + data.shape[1:]
    strides = (data.strides[0] * stride,) + data.strides
    windowed = np.lib.stride_tricks.as_strided(
        data, shape=shape, strides=strides,
    )
    return windowed.copy()  # contiguous, safe to mutate


def standardize_length(
    data: npt.NDArray[np.float64],
    target_length: int,
    method: str = "pad",
    pad_value: float = 0.0,
) -> npt.NDArray[np.float64]:
    """Standardize array length along axis 0.

    Parameters
    ----------
    data : ndarray, shape (T, ...)
    target_length : int
        Desired number of frames.
    method : {"pad", "crop", "resample"}
        - ``"pad"``: truncate from end if longer, zero-pad at end
          if shorter.
        - ``"crop"``: center-crop if longer, zero-pad at end if
          shorter.
        - ``"resample"``: linearly interpolate to *target_length*
          frames.  Suitable for position data.  **Not recommended
          for rotation data** — use ``pybvh.Bvh.resample()`` with
          SLERP instead.
    pad_value : float
        Value used for padding (default 0.0).  Only used by
        ``"pad"`` and ``"crop"`` methods.

    Returns
    -------
    ndarray, shape (target_length, ...)
    """
    data = np.asarray(data, dtype=np.float64)
    T = data.shape[0]

    if method == "pad":
        if T >= target_length:
            return data[:target_length].copy()
        return _pad(data, target_length, pad_value)

    elif method == "crop":
        if T >= target_length:
            start = (T - target_length) // 2
            return data[start:start + target_length].copy()
        return _pad(data, target_length, pad_value)

    elif method == "resample":
        import warnings
        warnings.warn(
            "standardize_length(method='resample') uses linear interpolation, "
            "which is not suitable for rotation data. For rotation arrays, "
            "use pybvh.Bvh.resample() (SLERP-based) before extracting arrays.",
            stacklevel=2,
        )
        if T == target_length:
            return data.copy()
        old_t = np.linspace(0.0, 1.0, T)
        new_t = np.linspace(0.0, 1.0, target_length)
        flat = data.reshape(T, -1)
        resampled = np.column_stack([
            np.interp(new_t, old_t, flat[:, c])
            for c in range(flat.shape[1])
        ])
        return resampled.reshape((target_length,) + data.shape[1:])

    else:
        raise ValueError(
            f"Unknown method '{method}'. Use 'pad', 'crop', or 'resample'.")


def uniform_temporal_sample(
    num_frames: int,
    clip_length: int,
    mode: str = "train",
    rng: np.random.Generator | None = None,
) -> npt.NDArray[np.intp]:
    """Sample *clip_length* frame indices from a sequence of *num_frames*.

    Divides the sequence into *clip_length* equal segments and picks
    one frame index per segment.  In ``"train"`` mode, picks a random
    offset within each segment (temporal augmentation).  In ``"test"``
    mode, picks a deterministic offset (reproducible evaluation).

    Handles three regimes:

    - ``num_frames < clip_length``: sequential indices with a random
      start (train) or start at 0 (test).  Some indices will be
      ``>= num_frames``; the caller must apply
      ``indices % num_frames`` before indexing into data.
    - ``clip_length <= num_frames < 2 * clip_length``: starts with
      ``[0, ..., clip_length-1]`` and randomly inserts gaps to
      spread indices across the full ``[0, num_frames)`` range.
    - ``num_frames >= 2 * clip_length``: uniform segment-based
      sampling with random (train) or deterministic (test) offsets
      within each segment.

    Parameters
    ----------
    num_frames : int
        Total frames in the source sequence.
    clip_length : int
        Number of frame indices to return.
    mode : {"train", "test"}
        ``"train"`` for random offsets, ``"test"`` for deterministic.
    rng : numpy Generator, optional
        For reproducibility in train mode.  Ignored in test mode.

    Returns
    -------
    ndarray of shape (clip_length,), dtype int
        Frame indices.  May contain values ``>= num_frames`` when
        ``num_frames < clip_length``; apply ``% num_frames`` to use.
    """
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}")
    if clip_length < 1:
        raise ValueError(f"clip_length must be >= 1, got {clip_length}")

    if mode == "train":
        if rng is None:
            rng = np.random.default_rng()
    elif mode == "test":
        rng = np.random.default_rng(0)
    else:
        raise ValueError(f"mode must be 'train' or 'test', got '{mode}'")

    if num_frames < clip_length:
        # Short sequence: sequential indices with random start (train)
        # or start=0 (test).  Caller applies % num_frames for wrapping.
        start = rng.integers(0, num_frames) if mode == "train" else 0
        return np.arange(start, start + clip_length, dtype=np.intp)

    if num_frames < 2 * clip_length:
        # Dense: start with [0..clip_length-1], randomly insert gaps
        # to spread indices across the full [0, num_frames) range.
        n_gaps = num_frames - clip_length
        basic = np.arange(clip_length, dtype=np.intp)
        gap_positions = rng.choice(clip_length + 1, size=n_gaps, replace=False)
        offset = np.zeros(clip_length + 1, dtype=np.intp)
        offset[gap_positions] = 1
        offset = np.cumsum(offset)
        return basic + offset[:clip_length]

    # Uniform segment-based sampling: integer boundaries, discrete offsets
    boundaries = np.array(
        [i * num_frames // clip_length for i in range(clip_length + 1)],
        dtype=np.intp,
    )
    seg_sizes = np.diff(boundaries)
    seg_starts = boundaries[:clip_length]
    offsets = rng.integers(seg_sizes)
    return seg_starts + offsets


def sample_temporal(
    data: npt.NDArray[np.float64],
    clip_length: int,
    num_samples: int = 1,
    mode: str = "train",
    rng: np.random.Generator | None = None,
) -> npt.NDArray[np.float64]:
    """Sample *clip_length* frames from *data* with wraparound.

    Convenience wrapper around :func:`uniform_temporal_sample` that
    applies the sampled indices to an array and supports generating
    multiple independent samples.

    Parameters
    ----------
    data : ndarray, shape (T, ...)
        Input array where axis 0 is the time dimension.
    clip_length : int
        Number of frames to sample.
    num_samples : int
        Number of independent samples to generate (default 1).
    mode : {"train", "test"}
    rng : numpy Generator, optional

    Returns
    -------
    ndarray
        Shape ``(num_samples, clip_length, ...)`` if ``num_samples > 1``,
        or ``(clip_length, ...)`` if ``num_samples == 1``.
    """
    data = np.asarray(data)
    T = data.shape[0]

    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}")

    if rng is None and mode == "train":
        rng = np.random.default_rng()

    samples = []
    for _ in range(num_samples):
        indices = uniform_temporal_sample(T, clip_length, mode=mode, rng=rng)
        indices = indices % T
        samples.append(data[indices])

    if num_samples == 1:
        return samples[0]
    return np.stack(samples, axis=0)


def _pad(
    data: npt.NDArray[np.float64],
    target_length: int,
    pad_value: float,
) -> npt.NDArray[np.float64]:
    """Pad *data* along axis 0 to *target_length*."""
    pad_shape = (target_length - data.shape[0],) + data.shape[1:]
    padding = np.full(pad_shape, pad_value, dtype=np.float64)
    return np.concatenate([data, padding], axis=0)
