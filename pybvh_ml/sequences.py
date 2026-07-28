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
    method : {"pad", "crop", "resample_linear"}
        - ``"pad"``: truncate from end if longer, zero-pad at end
          if shorter.
        - ``"crop"``: center-crop if longer, zero-pad at end if
          shorter.
        - ``"resample_linear"``: linearly interpolate to
          *target_length* frames along axis 0.  Correct for position
          data, velocities, and generic feature arrays.  **Not
          correct for rotation arrays** (Euler / quaternion / 6D /
          axis-angle) — linear interpolation does not preserve
          rotation geometry.  For rotations, resample with
          :meth:`pybvh.Bvh.resample` (SLERP) before extracting
          arrays.  The name makes the limitation visible at the
          call site; no runtime warning is emitted.
    pad_value : float
        Value used for padding (default 0.0).  Only used by
        ``"pad"`` and ``"crop"`` methods.

        Padding is a constant appended at the *end*; the alternatives are
        front-padding and edge-repeat (holding the last frame), neither of
        which is provided.  The three differ for any model that reads the
        padded frames — pair padded arrays with a length or mask, as
        :func:`pybvh_ml.torch.collate_motion_batch` returns, so that they
        never do.

        The default ``0.0`` is a valid feature value but **not a valid
        rotation** in any representation pybvh-ml packs: the zero quaternion
        has no norm, the zero 6D pair has no orthonormalization, and the
        zero rotation matrix is singular.  Zero *is* the identity for
        Euler and axis-angle, so those pad to a rest pose rather than to
        something undefined.  There is no scalar ``pad_value`` that
        expresses the identity for quaternion / 6D / rotation-matrix
        arrays — mask the padded frames instead of trying to pick one.

    Returns
    -------
    ndarray, shape (target_length, ...)
        ``"pad"`` and ``"crop"`` preserve the input dtype — they only
        select and append frames, so a ``float32`` clip stays
        ``float32`` rather than silently doubling in size.
        ``"resample_linear"`` returns ``float64``: it computes new
        values, and the interpolation runs in double precision.
    """
    if target_length < 1:
        raise ValueError(
            f"target_length must be >= 1, got {target_length}")
    data = np.asarray(data)
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

    elif method == "resample_linear":
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
            f"Unknown method '{method}'. "
            f"Use 'pad', 'crop', or 'resample_linear'.")


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
        Offset policy within each segment.  Both modes *draw* their
        offsets from the generator; they differ in which generator they
        default to when ``rng`` is None — fresh entropy for ``"train"``,
        a fixed ``default_rng(0)`` for ``"test"``, which is what makes
        test-mode indices repeatable rather than making them zero.  The
        one exception is the short-clip regime, where test mode starts
        at frame 0.
    rng : numpy Generator, optional
        Drives the sampling in **both** modes.  ``None`` uses fresh
        entropy in train mode and a fixed ``default_rng(0)`` in test
        mode.

    Returns
    -------
    ndarray of shape (clip_length,), dtype int
        Frame indices.  May contain values ``>= num_frames`` when
        ``num_frames < clip_length``; apply ``% num_frames`` to use.

    Notes
    -----
    **``mode="test"`` alone is not a reproducibility guarantee.**  Test
    mode fixes the offset *policy*, not the generator: a supplied
    ``rng`` overrides the fixed default in test mode as much as in
    train mode.  Passing one generator shared with other draws — the
    natural thing to do when the same object is threaded through both
    modes — means it has advanced by the time the next call arrives, so
    repeated reads of the same clip return different frames.  Pass
    ``rng=None``, or a generator freshly seeded per call, whenever
    repeated reads must agree.

    Before 0.5.0 test mode discarded a supplied ``rng`` outright, which
    made a shared generator harmless here and hid the distinction.
    """
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}")
    if clip_length < 1:
        raise ValueError(f"clip_length must be >= 1, got {clip_length}")

    if mode not in ("train", "test"):
        raise ValueError(f"mode must be 'train' or 'test', got '{mode}'")
    if rng is None:
        rng = (np.random.default_rng() if mode == "train"
               else np.random.default_rng(0))

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
        Number of independent samples to generate (default 1).  The
        rng is created once and threaded through every draw, so test
        mode yields ``num_samples`` *distinct* deterministic samples
        (reproducible across calls).
    mode : {"train", "test"}
    rng : numpy Generator, optional
        Drives the sampling in **both** modes; ``None`` uses fresh
        entropy in train mode and a fixed ``default_rng(0)`` in test
        mode.  Supplying a generator that is shared with other draws
        makes even ``mode="test"`` vary between calls — see the Notes
        on :func:`uniform_temporal_sample`.

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

    if rng is None:
        rng = (np.random.default_rng() if mode == "train"
               else np.random.default_rng(0))

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
    """Pad *data* along axis 0 to *target_length*, keeping its dtype."""
    pad_shape = (target_length - data.shape[0],) + data.shape[1:]
    padding = np.full(pad_shape, pad_value, dtype=data.dtype)
    return np.concatenate([data, padding], axis=0)
