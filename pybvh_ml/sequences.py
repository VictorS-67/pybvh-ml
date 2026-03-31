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


def _pad(
    data: npt.NDArray[np.float64],
    target_length: int,
    pad_value: float,
) -> npt.NDArray[np.float64]:
    """Pad *data* along axis 0 to *target_length*."""
    pad_shape = (target_length - data.shape[0],) + data.shape[1:]
    padding = np.full(pad_shape, pad_value, dtype=np.float64)
    return np.concatenate([data, padding], axis=0)
