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
- quat:       ~1×   (nothing to cache — representation IS the cache)

Augmentations not in ``STAGED_DISPATCH`` are still supported; the
pipeline transparently flushes the cache, converts the joint data
back to the representation declared in that step's kwargs, and calls
the function normally.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pybvh import parse_axis, rotations

from .arrays import MotionArrays
from .augmentation import (
    _build_rotation_matrix,
    _build_rotation_quat,
    _from_quats,
    _mirror_sign_quat,
    _mirror_sign_rot6d,
    _refresh_positions_by_fk,
    _require_pairs,
    _swap_lr_pairs,
    _to_quats,
    _as_radians,
    _validate_drop_rate,
    _validate_sigma,
    add_joint_position_noise,
    add_joint_rotation_noise,
    add_node_position_noise,
    add_root_position_noise,
    dropout_arrays,
    mirror,
    rotate_vertical,
    speed_perturbation_arrays,
)


_POSITION_STREAMS = ("joint_pos", "node_pos")


# =========================================================================
# Staging state
# =========================================================================

class _StagingState:
    """Every stream of one clip, evolving across pipeline steps.

    Invariant: ``jd`` is a valid representation of the current rotations
    under ``current_repr``.  If ``quats`` is not ``None``, it is also a
    valid quaternion view of the same rotations (and may be the same
    object as ``jd`` when ``current_repr == "quat"``).

    The positions carry no cache — there is only one representation of a
    position — so ``quats`` / ``current_repr`` / ``euler_orders`` are all
    ``None`` on a positions-only pipeline.  They live here rather than
    being threaded through the staged functions because all four
    geometric steps now transform them, and because a single owner is
    where the "speed perturbation changes ``F`` for every stream at
    once" invariant can be asserted.

    Every stream is held in ``float64``: the pipeline computes in double
    whatever it was given and restores the caller's dtypes once, at the
    end.
    """

    __slots__ = ("root_pos", "jd", "joint_pos", "node_pos",
                 "position_centering", "current_repr", "euler_orders",
                 "quats")

    def __init__(
        self,
        arrays: MotionArrays,
        representation: str | None,
        euler_orders: list[str] | None,
    ) -> None:
        self.root_pos = np.asarray(arrays.root_pos, dtype=np.float64)
        self.jd = (None if arrays.joint_rot is None
                   else np.asarray(arrays.joint_rot, dtype=np.float64))
        self.joint_pos = (None if arrays.joint_pos is None
                          else np.asarray(arrays.joint_pos, dtype=np.float64))
        self.node_pos = (None if arrays.node_pos is None
                         else np.asarray(arrays.node_pos, dtype=np.float64))
        self.position_centering = arrays.position_centering
        self.current_repr = representation
        self.euler_orders = euler_orders
        self.quats: npt.NDArray[np.float64] | None = None

    # -- rotation cache --------------------------------------------------

    def materialize_quats(self) -> npt.NDArray[np.float64]:
        """Return the quaternion view, computing it once and caching."""
        if self.quats is None:
            # _to_quats returns jd itself for "quat" (cache aliasing intact).
            self.quats = _to_quats(self.jd, self.current_repr,
                                   self.euler_orders)
        return self.quats

    def ensure_repr(self, target_repr: str | None) -> None:
        """Convert ``jd`` in place to ``target_repr`` (using quat cache)."""
        if self.jd is None or target_repr is None:
            return
        if self.current_repr == target_repr:
            return
        q = self.materialize_quats()
        # _from_quats returns q itself for "quat" (cache aliasing intact).
        self.jd = _from_quats(q, target_repr, self.euler_orders)
        self.current_repr = target_repr

    def set_from_quats(self, new_quats: npt.NDArray[np.float64]) -> None:
        """Commit a quat-space op result.  ``jd`` becomes ``new_quats``."""
        self.quats = new_quats
        self.jd = new_quats
        self.current_repr = "quat"

    def set_jd_invalidate_quats(
        self, new_jd: npt.NDArray[np.float64] | None, new_repr: str | None,
    ) -> None:
        """Commit a non-quat op result; the quat cache is stale."""
        self.jd = new_jd
        self.current_repr = new_repr
        self.quats = None

    # -- position streams ------------------------------------------------

    @property
    def has_positions(self) -> bool:
        return self.joint_pos is not None or self.node_pos is not None

    def positions(self) -> dict[str, npt.NDArray[np.float64] | None]:
        """The present position streams, keyed by field name."""
        return {name: getattr(self, name) for name in _POSITION_STREAMS}

    def set_positions(
        self, values: dict[str, npt.NDArray[np.float64] | None],
    ) -> None:
        """Replace the position streams, keeping "present" unchanged.

        A step may transform a stream but never add or drop one, so a
        value that arrives ``None`` for a stream the clip carries is a
        bug in that step rather than a legitimate drop.
        """
        for name, value in values.items():
            if (getattr(self, name) is None) != (value is None):
                raise AssertionError(
                    f"staged step changed whether {name} is present; steps "
                    f"transform streams, they do not add or drop them")
            setattr(self, name, value)

    def set_frame_count(
        self,
        root_pos: npt.NDArray[np.float64],
        positions: dict[str, npt.NDArray[np.float64] | None],
        joint_rot_frames: int,
    ) -> None:
        """Commit a step that resampled time, checking every stream moved.

        ``speed_perturbation_arrays`` is the one step that changes ``F``,
        and it has to change it for all streams at once — a clip whose
        positions and rotations disagree on frame count is exactly what
        :class:`~pybvh_ml.MotionArrays` validation exists to prevent, and
        the staged path builds no container until the very end, so
        nothing else would catch it.
        """
        frames = root_pos.shape[0]
        for name, value in positions.items():
            if value is not None and value.shape[0] != frames:
                raise AssertionError(
                    f"staged resampling left {name} at {value.shape[0]} "
                    f"frames while root_pos has {frames}")
        if joint_rot_frames not in (-1, frames):
            raise AssertionError(
                f"staged resampling left joint_rot at {joint_rot_frames} "
                f"frames while root_pos has {frames}")
        self.root_pos = root_pos
        self.set_positions(positions)

    # -- container round-trip --------------------------------------------

    def as_arrays(self) -> MotionArrays:
        """A ``float64`` container over the current state.

        What an unregistered custom step receives, and what the pipeline
        returns once the caller's dtypes are restored.
        """
        return MotionArrays(
            root_pos=self.root_pos, joint_rot=self.jd,
            joint_pos=self.joint_pos, node_pos=self.node_pos,
            position_centering=self.position_centering)

    def adopt(self, arrays: MotionArrays, representation: str | None) -> None:
        """Take an unregistered step's result back into the state.

        Its ``joint_rot`` is opaque data still in *representation*, so
        the quat cache is dropped.
        """
        self.root_pos = np.asarray(arrays.root_pos, dtype=np.float64)
        for name in _POSITION_STREAMS:
            value = getattr(arrays, name)
            setattr(self, name, None if value is None
                    else np.asarray(value, dtype=np.float64))
        self.set_jd_invalidate_quats(
            None if arrays.joint_rot is None
            else np.asarray(arrays.joint_rot, dtype=np.float64),
            representation)


# =========================================================================
# Per-augmentation staged variants.
#
# Each takes ``(state, **resolved_kwargs)`` and mutates the state in
# place — the state carries every stream, so there is nothing left to
# return.  (Before 0.6.0 they took and returned ``root_pos`` beside the
# state; four streams is where that stopped paying.)
# =========================================================================

def _rotate_vertical_staged(
    state: _StagingState,
    angle: float,
    up_axis: str,
    representation: str | None = None,
    degrees: bool = False,
    euler_orders: list[str] | None = None,
    **_: object,
) -> None:
    up = parse_axis(up_axis)
    up_idx, up_sign = up.index, up.sign
    signed_angle = _as_radians(angle, degrees) * up_sign
    R_vert = _build_rotation_matrix(signed_angle, up_idx)
    state.root_pos = (R_vert @ state.root_pos.T).T
    state.set_positions({
        name: None if value is None else value @ R_vert.T
        for name, value in state.positions().items()})

    if state.jd is None:
        return
    if state.jd.shape[1] == 0:
        raise ValueError(
            "rotate_vertical requires at least one joint (joint 0 is "
            "the root whose rotation carries the yaw); got J=0")

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
        return

    # Quat path — use cache if present.
    q = state.materialize_quats()
    q_rot = _build_rotation_quat(signed_angle, up_idx)
    new_q = q.copy()
    new_q[:, 0] = rotations.quat_multiply(q_rot, q[:, 0])
    state.set_from_quats(new_q)


def _mirror_staged(
    state: _StagingState,
    lateral_axis: str,
    lr_joint_pairs: list[tuple[int, int]] | None = None,
    lr_node_pairs: list[tuple[int, int]] | None = None,
    representation: str | None = None,
    euler_orders: list[str] | None = None,
    **_: object,
) -> None:
    lateral_idx = parse_axis(lateral_axis).index
    new_rp = state.root_pos.copy()
    new_rp[:, lateral_idx] *= -1.0
    state.root_pos = new_rp

    if state.jd is not None:
        _require_pairs(lr_joint_pairs, "lr_joint_pairs", "joint_rot")
    pair_lists = {"joint_pos": lr_joint_pairs, "node_pos": lr_node_pairs}
    mirrored: dict[str, npt.NDArray[np.float64] | None] = {}
    for name, value in state.positions().items():
        if value is None:
            mirrored[name] = None
            continue
        _require_pairs(
            pair_lists[name],
            "lr_joint_pairs" if name == "joint_pos" else "lr_node_pairs",
            name)
        new_value = value.copy()
        new_value[:, :, lateral_idx] *= -1.0
        _swap_lr_pairs(new_value, pair_lists[name])
        mirrored[name] = new_value
    state.set_positions(mirrored)

    if state.jd is None:
        return

    if representation == "6d":
        state.ensure_repr("6d")
        new_jd = state.jd.copy()
        _swap_lr_pairs(new_jd, lr_joint_pairs)
        new_jd *= _mirror_sign_rot6d(lateral_idx)
        state.set_jd_invalidate_quats(new_jd, "6d")
        return

    q = state.materialize_quats().copy()
    _swap_lr_pairs(q, lr_joint_pairs)
    q *= _mirror_sign_quat(lateral_idx)
    state.set_from_quats(q)


def _add_joint_rotation_noise_staged(
    state: _StagingState,
    sigma: float,
    representation: str | None = None,  # symmetry; not used in the math
    degrees: bool = False,
    rng: np.random.Generator | None = None,
    euler_orders: list[str] | None = None,
    fk_topology: object | None = None,
    world_up: str | None = None,
    **_: object,
) -> None:
    _validate_sigma(sigma)
    if rng is None:
        rng = np.random.default_rng()
    sigma_rad = _as_radians(sigma, degrees)

    q = state.materialize_quats()
    F, J, _ = q.shape
    axis = rng.standard_normal((F, J, 3))
    norm = np.linalg.norm(axis, axis=-1, keepdims=True)
    norm = np.where(norm < 1e-15, 1.0, norm)
    axis = axis / norm

    half_angle = rng.normal(0, sigma_rad, (F, J)) / 2.0
    q_noise = np.empty((F, J, 4), dtype=np.float64)
    q_noise[..., 0] = np.cos(half_angle)
    q_noise[..., 1:] = np.sin(half_angle)[..., np.newaxis] * axis

    noisy = rotations.quat_multiply(q_noise, q)
    norms = np.linalg.norm(noisy, axis=-1, keepdims=True)
    # q_noise is unit by construction — zero norm means a zero-norm
    # input quaternion; fail loudly like the public function.
    if np.any(norms == 0.0):
        raise ValueError(
            "joint_rot contains a zero-norm quaternion; the zero "
            "quaternion does not represent a rotation")
    noisy /= norms
    state.set_from_quats(noisy)

    if state.has_positions:
        # The cache already holds quaternions, so the refresh converts
        # once (quat → euler) rather than round-tripping the declared
        # representation.
        state.set_positions(_refresh_positions_by_fk(
            joint_rot=noisy, root_pos=state.root_pos, representation="quat",
            fk_topology=fk_topology, world_up=world_up,
            position_centering=state.position_centering,
            want_joint_pos=state.joint_pos is not None,
            want_node_pos=state.node_pos is not None))

    # Copy: staged functions must never leave the caller's own root_pos
    # in the state (later in-place edits would reach the caller's array).
    state.root_pos = state.root_pos.copy()


def _add_root_position_noise_staged(
    state: _StagingState,
    sigma: float,
    rng: np.random.Generator | None = None,
    **_: object,
) -> None:
    # Never touches the rotations, so unlike every other staged function
    # this one leaves the quat cache alone — a positional-jitter step no
    # longer forces a materialization it does not need.
    _validate_sigma(sigma)
    if rng is None:
        rng = np.random.default_rng()
    if sigma <= 0:
        state.root_pos = state.root_pos.copy()
        return

    noise = rng.normal(0, sigma, state.root_pos.shape)
    state.root_pos = state.root_pos + noise
    if not state.has_positions:
        return
    if state.position_centering == "skeleton":
        # Root-relative positions do not move with the root.
        return
    vertex_noise = noise[:, np.newaxis, :]
    state.set_positions({
        name: None if value is None else value + vertex_noise
        for name, value in state.positions().items()})


def _add_position_noise_staged(
    state: _StagingState,
    stream: str,
    sigma: float,
    rng: np.random.Generator | None,
) -> None:
    """Shared body of the two staged keypoint-jitter steps."""
    _validate_sigma(sigma)
    if rng is None:
        rng = np.random.default_rng()
    value = getattr(state, stream)
    if sigma > 0:
        value = value + rng.normal(0, sigma, value.shape)
    else:
        value = value.copy()
    setattr(state, stream, value)


def _add_joint_position_noise_staged(
    state: _StagingState,
    sigma: float,
    rng: np.random.Generator | None = None,
    **_: object,
) -> None:
    _add_position_noise_staged(state, "joint_pos", sigma, rng)


def _add_node_position_noise_staged(
    state: _StagingState,
    sigma: float,
    rng: np.random.Generator | None = None,
    **_: object,
) -> None:
    _add_position_noise_staged(state, "node_pos", sigma, rng)


def _speed_perturbation_staged(
    state: _StagingState,
    factor: float,
    representation: str | None = None,
    euler_orders: list[str] | None = None,
    **_: object,
) -> None:
    if factor <= 0:
        raise ValueError(f"factor must be > 0, got {factor}")
    F = state.root_pos.shape[0]
    if F < 2:
        state.root_pos = state.root_pos.copy()
        return

    F_new = max(2, round(F / factor))
    t_orig = np.linspace(0.0, 1.0, F)
    t_new = np.linspace(0.0, 1.0, F_new)

    new_rp = np.empty((F_new, 3), dtype=np.float64)
    for ax in range(3):
        new_rp[:, ax] = np.interp(t_new, t_orig, state.root_pos[:, ax])

    idx_right = np.searchsorted(t_orig, t_new, side='right')
    idx_right = np.clip(idx_right, 1, F - 1)
    idx_left = idx_right - 1
    t_left = t_orig[idx_left]
    t_right = t_orig[idx_right]
    # Adjacent samples of linspace(0, 1, F) with F >= 2, so dt is
    # exactly 1/(F-1) — never the degenerate interval a zero-guard here
    # would be protecting against.
    dt = t_right - t_left
    alpha = (t_new - t_left) / dt

    weight = alpha[:, np.newaxis, np.newaxis]
    resampled = {
        name: (None if value is None
               else (1.0 - weight) * value[idx_left] + weight * value[idx_right])
        for name, value in state.positions().items()
    }

    if state.jd is not None:
        q = state.materialize_quats()
        alpha_jt = np.broadcast_to(alpha[:, np.newaxis], (F_new, q.shape[1]))
        state.set_from_quats(
            rotations.quat_slerp(q[idx_left], q[idx_right], alpha_jt))

    state.set_frame_count(
        new_rp, resampled,
        -1 if state.jd is None else state.jd.shape[0])


def _dropout_staged(
    state: _StagingState,
    drop_rate: float,
    representation: str | None = None,
    rng: np.random.Generator | None = None,
    euler_orders: list[str] | None = None,
    **_: object,
) -> None:
    _validate_drop_rate(drop_rate)
    if rng is None:
        rng = np.random.default_rng()
    F = state.root_pos.shape[0]
    if F < 2 or drop_rate == 0:
        state.root_pos = state.root_pos.copy()
        return

    keep_mask = rng.random(F) >= drop_rate
    keep_mask[0] = True
    keep_mask[-1] = True
    dropped = np.where(~keep_mask)[0]
    if len(dropped) == 0:
        state.root_pos = state.root_pos.copy()
        return

    kept = np.where(keep_mask)[0]
    ins = np.searchsorted(kept, dropped, side='right')
    left_idx = kept[np.clip(ins - 1, 0, len(kept) - 1)]
    right_idx = kept[np.clip(ins, 0, len(kept) - 1)]
    # left_idx < dropped < right_idx by construction (frames 0 and F-1
    # are always kept), so dt >= 1 always.
    dt = (right_idx - left_idx).astype(np.float64)
    alpha = (dropped - left_idx).astype(np.float64) / dt

    new_rp = state.root_pos.copy()
    for ax in range(3):
        new_rp[dropped, ax] = (
            (1.0 - alpha) * state.root_pos[left_idx, ax]
            + alpha * state.root_pos[right_idx, ax])
    state.root_pos = new_rp

    weight = alpha[:, np.newaxis, np.newaxis]
    filled: dict[str, npt.NDArray[np.float64] | None] = {}
    for name, value in state.positions().items():
        if value is None:
            filled[name] = None
            continue
        new_value = value.copy()
        new_value[dropped] = ((1.0 - weight) * value[left_idx]
                              + weight * value[right_idx])
        filled[name] = new_value
    state.set_positions(filled)

    if state.jd is None:
        return
    q = state.materialize_quats()
    new_q = q.copy()
    alpha_jt = np.broadcast_to(
        alpha[:, np.newaxis], (len(dropped), q.shape[1]))
    new_q[dropped] = rotations.quat_slerp(
        q[left_idx], q[right_idx], alpha_jt)
    state.set_from_quats(new_q)


# =========================================================================
# Dispatch registry
# =========================================================================

STAGED_DISPATCH = {
    rotate_vertical: _rotate_vertical_staged,
    mirror: _mirror_staged,
    add_joint_rotation_noise: _add_joint_rotation_noise_staged,
    add_root_position_noise: _add_root_position_noise_staged,
    add_joint_position_noise: _add_joint_position_noise_staged,
    add_node_position_noise: _add_node_position_noise_staged,
    speed_perturbation_arrays: _speed_perturbation_staged,
    dropout_arrays: _dropout_staged,
}
