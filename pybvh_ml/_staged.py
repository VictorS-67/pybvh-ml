"""Internal quaternion-caching dispatch for :class:`AugmentationPipeline`.

When multiple augmentations in a pipeline would each convert to/from
quaternion space independently, this module short-circuits by sharing
a single quaternion cache across steps.  The public augmentation
signatures are unchanged; users still build pipelines exactly as
before, and staging is applied automatically when possible.

Typical speedups on real data (20 files × full-length clips, 4-step
pipeline of rotate + mirror + noise + speed_perturbation):

- 6d:         ~1.5× (fast paths for rotate/mirror already skip quat)
- axisangle:  ~3×   (every quat-internal step was paying a roundtrip)
- euler:      ~3×
- quaternion: ~1×   (nothing to cache — representation IS the cache)

Augmentations not in ``STAGED_DISPATCH`` are still supported; the
pipeline transparently flushes the cache, converts the joint data
back to the representation declared in that step's kwargs, and calls
the function normally.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pybvh import rotations

from .augmentation import (
    _build_rotation_matrix,
    _build_rotation_quat,
    _mirror_sign_quat,
    _mirror_sign_rot6d,
    _parse_axis,
    _quat_multiply,
    _swap_lr_pairs,
    add_joint_noise,
    dropout_arrays,
    mirror,
    rotate_vertical,
    speed_perturbation_arrays,
)


# =========================================================================
# Staging state
# =========================================================================

class _StagingState:
    """Evolving joint-data state shared across pipeline steps.

    Invariant: ``jd`` is a valid representation of the current rotations
    under ``current_repr``.  If ``quats`` is not ``None``, it is also a
    valid quaternion view of the same rotations (and may be the same
    object as ``jd`` when ``current_repr == "quaternion"``).
    """

    def __init__(
        self,
        joint_data: npt.NDArray[np.float64],
        representation: str,
        euler_orders: list[str] | None,
    ) -> None:
        self.jd = joint_data
        self.current_repr = representation
        self.euler_orders = euler_orders
        self.quats: npt.NDArray[np.float64] | None = None

    def materialize_quats(self) -> npt.NDArray[np.float64]:
        """Return the quaternion view, computing it once and caching."""
        if self.quats is not None:
            return self.quats
        if self.current_repr == "quaternion":
            self.quats = self.jd
        elif self.current_repr == "euler":
            self.quats = rotations.convert(
                self.jd, "euler", "quaternion",
                order=self.euler_orders, degrees=True)
        else:
            self.quats = rotations.convert(
                self.jd, self.current_repr, "quaternion")
        return self.quats

    def ensure_repr(self, target_repr: str) -> None:
        """Convert ``jd`` in place to ``target_repr`` (using quat cache)."""
        if self.current_repr == target_repr:
            return
        q = self.materialize_quats()
        if target_repr == "quaternion":
            self.jd = q
        elif target_repr == "euler":
            self.jd = rotations.convert(
                q, "quaternion", "euler",
                order=self.euler_orders, degrees=True)
        else:
            self.jd = rotations.convert(q, "quaternion", target_repr)
        self.current_repr = target_repr

    def set_from_quats(self, new_quats: npt.NDArray[np.float64]) -> None:
        """Commit a quat-space op result.  ``jd`` becomes ``new_quats``."""
        self.quats = new_quats
        self.jd = new_quats
        self.current_repr = "quaternion"

    def set_jd_invalidate_quats(
        self, new_jd: npt.NDArray[np.float64], new_repr: str,
    ) -> None:
        """Commit a non-quat op result; the quat cache is stale."""
        self.jd = new_jd
        self.current_repr = new_repr
        self.quats = None


# =========================================================================
# Per-augmentation staged variants.
#
# Each takes ``(root_pos, state, **resolved_kwargs, rng=None)`` and
# returns the new ``root_pos``.  State mutations happen in-place.
# Parameter order mirrors the public ``(root_pos, joint_data, ...)``
# convention — ``state`` here is the joint-data carrier.
# =========================================================================

def _rotate_vertical_staged(
    root_pos: npt.NDArray[np.float64],
    state: _StagingState,
    angle_deg: float,
    up_axis: str,
    representation: str,
    euler_orders: list[str] | None = None,
    **_: object,
) -> npt.NDArray[np.float64]:
    up_idx, up_sign = _parse_axis(up_axis)
    signed_angle = angle_deg * up_sign
    R_vert = _build_rotation_matrix(signed_angle, up_idx)
    new_rp = (R_vert @ root_pos.T).T

    if representation == "6d":
        # Fast path: rotate the two column vectors of the root rotation
        # matrix directly, no quat conversion.
        state.ensure_repr("6d")
        new_jd = state.jd.copy()
        col0 = new_jd[:, 0, :3]
        col1 = new_jd[:, 0, 3:]
        new_jd[:, 0, :3] = (R_vert @ col0.T).T
        new_jd[:, 0, 3:] = (R_vert @ col1.T).T
        state.set_jd_invalidate_quats(new_jd, "6d")
        return new_rp

    # Quat path — use cache if present.
    q = state.materialize_quats()
    q_rot = _build_rotation_quat(signed_angle, up_idx)
    new_q = q.copy()
    new_q[:, 0] = _quat_multiply(q_rot, q[:, 0])
    state.set_from_quats(new_q)
    return new_rp


def _mirror_staged(
    root_pos: npt.NDArray[np.float64],
    state: _StagingState,
    lr_joint_pairs: list[tuple[int, int]],
    lateral_axis: str,
    representation: str,
    euler_orders: list[str] | None = None,
    **_: object,
) -> npt.NDArray[np.float64]:
    lateral_idx, _sign = _parse_axis(lateral_axis)
    new_rp = root_pos.copy()
    new_rp[:, lateral_idx] *= -1.0

    if representation == "6d":
        state.ensure_repr("6d")
        new_jd = state.jd.copy()
        _swap_lr_pairs(new_jd, lr_joint_pairs)
        new_jd *= _mirror_sign_rot6d(lateral_idx)
        state.set_jd_invalidate_quats(new_jd, "6d")
        return new_rp

    q = state.materialize_quats().copy()
    _swap_lr_pairs(q, lr_joint_pairs)
    q *= _mirror_sign_quat(lateral_idx)
    state.set_from_quats(q)
    return new_rp


def _add_joint_noise_staged(
    root_pos: npt.NDArray[np.float64],
    state: _StagingState,
    sigma_deg: float,
    representation: str,  # kept for signature symmetry; not used in math
    sigma_pos: float = 0.0,
    rng: np.random.Generator | None = None,
    euler_orders: list[str] | None = None,
    **_: object,
) -> npt.NDArray[np.float64]:
    if rng is None:
        rng = np.random.default_rng()

    q = state.materialize_quats()
    F, J, _ = q.shape
    axis = rng.standard_normal((F, J, 3))
    norm = np.linalg.norm(axis, axis=-1, keepdims=True)
    norm = np.where(norm < 1e-15, 1.0, norm)
    axis = axis / norm

    half_angle = np.radians(rng.normal(0, sigma_deg, (F, J))) / 2.0
    q_noise = np.empty((F, J, 4), dtype=np.float64)
    q_noise[..., 0] = np.cos(half_angle)
    q_noise[..., 1:] = np.sin(half_angle)[..., np.newaxis] * axis

    noisy = _quat_multiply(q_noise, q)
    noisy /= np.linalg.norm(noisy, axis=-1, keepdims=True)
    state.set_from_quats(noisy)

    new_rp = root_pos
    if sigma_pos > 0:
        new_rp = root_pos + rng.normal(0, sigma_pos, root_pos.shape)
    return new_rp


def _speed_perturbation_staged(
    root_pos: npt.NDArray[np.float64],
    state: _StagingState,
    factor: float,
    representation: str,
    euler_orders: list[str] | None = None,
    **_: object,
) -> npt.NDArray[np.float64]:
    if factor <= 0:
        raise ValueError(f"factor must be > 0, got {factor}")
    F = root_pos.shape[0]
    if F < 2:
        return root_pos.copy()

    F_new = max(2, round(F / factor))
    t_orig = np.linspace(0.0, 1.0, F)
    t_new = np.linspace(0.0, 1.0, F_new)

    new_rp = np.empty((F_new, 3), dtype=np.float64)
    for ax in range(3):
        new_rp[:, ax] = np.interp(t_new, t_orig, root_pos[:, ax])

    q = state.materialize_quats()
    J = q.shape[1]
    idx_right = np.searchsorted(t_orig, t_new, side='right')
    idx_right = np.clip(idx_right, 1, F - 1)
    idx_left = idx_right - 1
    t_left = t_orig[idx_left]
    t_right = t_orig[idx_right]
    dt = np.where(t_right - t_left < 1e-15, 1.0, t_right - t_left)
    alpha = (t_new - t_left) / dt
    alpha_jt = np.broadcast_to(alpha[:, np.newaxis], (F_new, J))
    new_q = rotations.quat_slerp(q[idx_left], q[idx_right], alpha_jt)
    state.set_from_quats(new_q)
    return new_rp


def _dropout_staged(
    root_pos: npt.NDArray[np.float64],
    state: _StagingState,
    drop_rate: float,
    representation: str,
    rng: np.random.Generator | None = None,
    euler_orders: list[str] | None = None,
    **_: object,
) -> npt.NDArray[np.float64]:
    if rng is None:
        rng = np.random.default_rng()
    F = root_pos.shape[0]
    if F < 2 or drop_rate <= 0:
        return root_pos.copy()

    keep_mask = rng.random(F) >= drop_rate
    keep_mask[0] = True
    keep_mask[-1] = True
    dropped = np.where(~keep_mask)[0]
    if len(dropped) == 0:
        return root_pos.copy()

    kept = np.where(keep_mask)[0]
    ins = np.searchsorted(kept, dropped, side='right')
    left_idx = kept[np.clip(ins - 1, 0, len(kept) - 1)]
    right_idx = kept[np.clip(ins, 0, len(kept) - 1)]
    dt = np.where(right_idx - left_idx < 1e-15,
                  1.0, right_idx - left_idx).astype(np.float64)
    alpha = (dropped - left_idx).astype(np.float64) / dt

    new_rp = root_pos.copy()
    for ax in range(3):
        new_rp[dropped, ax] = (
            (1.0 - alpha) * root_pos[left_idx, ax]
            + alpha * root_pos[right_idx, ax])

    q = state.materialize_quats()
    new_q = q.copy()
    alpha_jt = np.broadcast_to(
        alpha[:, np.newaxis], (len(dropped), q.shape[1]))
    new_q[dropped] = rotations.quat_slerp(
        q[left_idx], q[right_idx], alpha_jt)
    state.set_from_quats(new_q)
    return new_rp


# =========================================================================
# Dispatch registry
# =========================================================================

STAGED_DISPATCH = {
    rotate_vertical: _rotate_vertical_staged,
    mirror: _mirror_staged,
    add_joint_noise: _add_joint_noise_staged,
    speed_perturbation_arrays: _speed_perturbation_staged,
    dropout_arrays: _dropout_staged,
}
