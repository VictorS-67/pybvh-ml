"""Per-joint and per-node position streams (0.6.0).

Covers the container fields, the stream-coherence rule, the position
branches of the four geometric steps, the FK refresh, storage, packing
and the two Dataset classes.  The rigs written inline here are
*deliberately awkward* skeletons built to separate a correct
implementation from a plausible one — see each fixture's docstring.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from pybvh import frames_to_node_positions, read_bvh_file, rotations

from pybvh_ml import (
    AugmentationPipeline,
    MotionArrays,
    add_joint_position_noise,
    add_joint_rotation_noise,
    add_node_position_noise,
    add_root_position_noise,
    build_fk_topology,
    describe_features,
    dropout_arrays,
    find_mismatched_end_site_pairs,
    get_skeleton_info,
    handles_streams,
    load_preprocessed,
    mirror,
    pack_to_ctv,
    pack_to_flat,
    preprocess_directory,
    rotate_vertical,
    speed_perturbation_arrays,
    stream_support,
)


# =========================================================================
# Purpose-built rigs
# =========================================================================

# L/R partners carrying different numbers of end sites, *and* an end site
# (Head's) placed earlier in file order than the mismatched pair.  That
# ordering is the whole point: without it, node and joint indices coincide
# and a joint-space misreading of the end-site check passes just as well
# as the correct node-space one.  Here they diverge — the correct check
# reports the LeftHand/RightHand pair at node indices (4, 8), while the
# joint-space misreading reports (2, 4), which is LeftArm/RightArm.
END_SITE_MISMATCH_BVH = """HIERARCHY
ROOT Hips
{
  OFFSET 0.0 0.0 0.0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  JOINT Head
  {
    OFFSET 0.0 10.0 0.0
    CHANNELS 3 Zrotation Xrotation Yrotation
    End Site
    {
      OFFSET 0.0 5.0 0.0
    }
  }
  JOINT LeftArm
  {
    OFFSET 5.0 8.0 0.0
    CHANNELS 3 Zrotation Xrotation Yrotation
    JOINT LeftHand
    {
      OFFSET 5.0 0.0 0.0
      CHANNELS 3 Zrotation Xrotation Yrotation
      End Site
      {
        OFFSET 2.0 0.0 0.0
      }
      End Site
      {
        OFFSET 2.0 1.0 0.0
      }
    }
  }
  JOINT RightArm
  {
    OFFSET -5.0 8.0 0.0
    CHANNELS 3 Zrotation Xrotation Yrotation
    JOINT RightHand
    {
      OFFSET -5.0 0.0 0.0
      CHANNELS 3 Zrotation Xrotation Yrotation
      End Site
      {
        OFFSET -2.0 0.0 0.0
      }
    }
  }
}
MOTION
Frames: 3
Frame Time: 0.033333
""" + "\n".join(" ".join(["0.0"] * 21) for _ in range(3)) + "\n"


def _asymmetric_rig(frames: int = 6) -> str:
    """A rig whose left and right rest offsets are not each other's mirror.

    Local-space and world-space mirroring agree exactly on a laterally
    symmetric rest pose, so a symmetric rig cannot tell the two apart —
    which is what the order-dependence test needs to see.
    """
    motion = []
    rng = np.random.default_rng(7)
    for _ in range(frames):
        motion.append(" ".join(f"{v:.4f}" for v in rng.uniform(-20, 20, 18)))
    return """HIERARCHY
ROOT Hips
{
  OFFSET 0.0 0.0 0.0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  JOINT LeftArm
  {
    OFFSET 5.0 8.0 0.0
    CHANNELS 3 Zrotation Xrotation Yrotation
    JOINT LeftHand
    {
      OFFSET 6.0 0.0 0.0
      CHANNELS 3 Zrotation Xrotation Yrotation
      End Site
      {
        OFFSET 2.0 0.0 0.0
      }
    }
  }
  JOINT RightArm
  {
    OFFSET -3.0 8.0 1.5
    CHANNELS 3 Zrotation Xrotation Yrotation
    JOINT RightHand
    {
      OFFSET -4.0 0.5 0.0
      CHANNELS 3 Zrotation Xrotation Yrotation
      End Site
      {
        OFFSET -2.0 0.0 0.0
      }
    }
  }
}
MOTION
Frames: %d
Frame Time: 0.033333
""" % frames + "\n".join(motion) + "\n"


# Two *joints* sharing a name — the only way a name collision can reach
# the edge lists at all (end sites are leaves, so a joint's parent is
# always a joint).  Upstream covers the derivation; ours guards the
# storage path, which is the part only pybvh-ml owns.
NAME_COLLISION_BVH = """HIERARCHY
ROOT Hips
{
  OFFSET 0.0 0.0 0.0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  JOINT LeftArm
  {
    OFFSET 5.0 8.0 0.0
    CHANNELS 3 Zrotation Xrotation Yrotation
    JOINT Tip
    {
      OFFSET 4.0 0.0 0.0
      CHANNELS 3 Zrotation Xrotation Yrotation
      End Site
      {
        OFFSET 1.0 0.0 0.0
      }
    }
  }
  JOINT RightArm
  {
    OFFSET -5.0 8.0 0.0
    CHANNELS 3 Zrotation Xrotation Yrotation
    JOINT Tip
    {
      OFFSET -4.0 0.0 0.0
      CHANNELS 3 Zrotation Xrotation Yrotation
      End Site
      {
        OFFSET -1.0 0.0 0.0
      }
    }
  }
}
MOTION
Frames: 2
Frame Time: 0.033333
""" + "\n".join(" ".join(["0.0"] * 18) for _ in range(2)) + "\n"


@pytest.fixture
def mismatch_rig(tmp_path):
    path = tmp_path / "mismatch.bvh"
    path.write_text(END_SITE_MISMATCH_BVH)
    return read_bvh_file(path)


@pytest.fixture
def asymmetric_rig(tmp_path):
    path = tmp_path / "asymmetric.bvh"
    path.write_text(_asymmetric_rig())
    return read_bvh_file(path)


@pytest.fixture
def collision_rig(tmp_path):
    path = tmp_path / "collision.bvh"
    path.write_text(NAME_COLLISION_BVH)
    return read_bvh_file(path)


def _fk_joint_positions(arrays, topology, representation, centered, up=None):
    """Forward kinematics of *arrays*' rotations, in joint space."""
    euler = rotations.convert(
        arrays.joint_rot, representation, "euler",
        order=list(topology.euler_orders))
    nodes = frames_to_node_positions(
        topology, np.asarray(arrays.root_pos), euler,
        centered=centered, up=up)
    return np.asarray(nodes)[:, np.asarray(topology.joint_idx) >= 0]


# =========================================================================
# Container
# =========================================================================

class TestMotionArraysPositionStreams:

    def test_from_bvh_joint_space(self, bvh_example):
        arrays = MotionArrays.from_bvh(
            bvh_example, "6d", include_positions=True)
        assert arrays.joint_pos.shape == (
            bvh_example.frame_count, bvh_example.joint_count, 3)
        assert arrays.node_pos is None
        assert arrays.position_centering == "world"
        np.testing.assert_allclose(
            arrays.joint_pos, bvh_example.joint_positions())

    def test_from_bvh_node_space(self, bvh_example):
        arrays = MotionArrays.from_bvh(
            bvh_example, None, include_positions=True,
            position_space="node", position_centering="skeleton")
        assert arrays.joint_rot is None
        assert arrays.node_pos.shape[1] == len(bvh_example.nodes)
        np.testing.assert_allclose(
            arrays.node_pos, bvh_example.node_positions(centered="skeleton"))

    def test_from_bvh_needs_something_to_extract(self, bvh_example):
        with pytest.raises(ValueError, match="was asked for nothing"):
            MotionArrays.from_bvh(bvh_example, None)

    def test_present_streams(self, bvh_example):
        arrays = MotionArrays.from_bvh(
            bvh_example, "6d", include_positions=True)
        assert arrays.present_streams == {
            "root_pos", "joint_rot", "joint_pos"}

    def test_centering_without_positions_raises(self, bvh_example):
        with pytest.raises(ValueError, match="no position stream to describe"):
            MotionArrays(root_pos=bvh_example.root_pos,
                         position_centering="world")

    def test_positions_without_centering_are_legal(self, bvh_example):
        """The asymmetry is deliberate: an unknown frame is an honest
        state, and only the steps that depend on it raise."""
        arrays = MotionArrays(
            root_pos=bvh_example.root_pos,
            joint_pos=bvh_example.joint_positions())
        assert arrays.position_centering is None

    def test_joint_count_mismatch_raises(self, bvh_example):
        arrays = MotionArrays.from_bvh(bvh_example, "6d")
        with pytest.raises(ValueError, match="disagree on joint count"):
            arrays.replace(joint_pos=bvh_example.joint_positions()[:, :-1])

    def test_node_count_below_joint_count_raises(self, bvh_example):
        with pytest.raises(ValueError, match="Nodes are a superset"):
            MotionArrays(root_pos=bvh_example.root_pos,
                         joint_rot=bvh_example.to_6d()[1],
                         node_pos=bvh_example.joint_positions()[:, :-2],
                         position_centering="world")

    def test_replace_dropping_last_position_stream_must_clear_centering(
            self, bvh_example):
        arrays = MotionArrays.from_bvh(
            bvh_example, "6d", include_positions=True)
        with pytest.raises(ValueError, match="no position stream to describe"):
            arrays.replace(joint_pos=None)
        assert arrays.replace(
            joint_pos=None, position_centering=None).joint_pos is None

    def test_roundtrips_through_pickle(self, bvh_example):
        import pickle
        arrays = MotionArrays.from_bvh(
            bvh_example, "6d", include_positions=True)
        assert pickle.loads(pickle.dumps(arrays)) == arrays

    def test_center_root_moves_world_positions_with_the_root(
            self, bvh_example):
        raw = MotionArrays.from_bvh(bvh_example, "6d", include_positions=True)
        centered = MotionArrays.from_bvh(
            bvh_example, "6d", include_positions=True, center_root=True)
        shift = raw.root_pos[0]
        np.testing.assert_allclose(
            centered.joint_pos, raw.joint_pos - shift, atol=1e-12)

    def test_center_root_leaves_skeleton_positions_alone(self, bvh_example):
        raw = MotionArrays.from_bvh(
            bvh_example, "6d", include_positions=True,
            position_centering="skeleton")
        centered = MotionArrays.from_bvh(
            bvh_example, "6d", include_positions=True, center_root=True,
            position_centering="skeleton")
        np.testing.assert_array_equal(centered.joint_pos, raw.joint_pos)


# =========================================================================
# Stream declarations and the coherence rule
# =========================================================================

class TestStreamDeclarations:

    def test_geometric_steps_handle_every_stream(self):
        every = set(("root_pos", "joint_rot", "joint_pos", "node_pos"))
        for fn in (rotate_vertical, mirror, speed_perturbation_arrays,
                   dropout_arrays, add_root_position_noise,
                   add_joint_rotation_noise):
            assert stream_support(fn) == every, fn.__name__

    def test_keypoint_jitter_declines_rotations(self):
        assert stream_support(add_joint_position_noise) == {
            "root_pos", "joint_pos"}
        assert stream_support(add_node_position_noise) == {
            "root_pos", "node_pos"}

    def test_undeclared_step_defaults_to_the_pre_0_6_capability(self):
        def custom(arrays):
            return arrays
        assert stream_support(custom) == {"root_pos", "joint_rot"}

    def test_undeclared_step_refuses_positions_naming_the_decorator(
            self, bvh_example):
        def custom(arrays):
            return arrays
        arrays = MotionArrays.from_bvh(
            bvh_example, "6d", include_positions=True)
        pipe = AugmentationPipeline([(custom, 1.0, {})], representation="6d")
        with pytest.raises(ValueError, match="handles_streams"):
            pipe(arrays, rng=np.random.default_rng(0))

    def test_declared_step_accepts_positions(self, bvh_example):
        @handles_streams("root_pos", "joint_rot", "joint_pos")
        def scale_positions(arrays):
            return arrays.replace(joint_pos=arrays.joint_pos * 2.0)

        arrays = MotionArrays.from_bvh(
            bvh_example, "6d", include_positions=True)
        pipe = AugmentationPipeline(
            [(scale_positions, 1.0, {})], representation="6d")
        out = pipe(arrays, rng=np.random.default_rng(0))
        np.testing.assert_allclose(out.joint_pos, arrays.joint_pos * 2.0)

    def test_keypoint_jitter_refusal_names_inverse_kinematics(
            self, bvh_example):
        arrays = MotionArrays.from_bvh(
            bvh_example, "6d", include_positions=True)
        with pytest.raises(ValueError, match="inverse kinematics"):
            add_joint_position_noise(arrays, sigma=0.1)

    def test_unknown_stream_name_rejected_by_the_decorator(self):
        with pytest.raises(ValueError, match="unknown stream name"):
            handles_streams("root_pos", "joint_velocity")

    def test_step_may_not_drop_a_stream(self, bvh_example):
        def dropper(arrays):
            return MotionArrays(root_pos=arrays.root_pos,
                                joint_rot=arrays.joint_rot)

        arrays = MotionArrays.from_bvh(
            bvh_example, "6d", include_positions=True)
        pipe = AugmentationPipeline(
            [(handles_streams("root_pos", "joint_rot", "joint_pos")(dropper),
              1.0, {})], representation="6d")
        with pytest.raises(ValueError, match="changed which streams"):
            pipe(arrays, rng=np.random.default_rng(0))

    @pytest.mark.parametrize("cache_quats", [True, False])
    def test_preconditions_are_checked_before_any_step_runs(
            self, bvh_example, cache_quats):
        """A p=0.05 step with a missing fk_topology must raise on every
        sample, not on one in twenty."""
        arrays = MotionArrays.from_bvh(
            bvh_example, "6d", include_positions=True)
        pipe = AugmentationPipeline(
            [(add_joint_rotation_noise, 0.05, {"sigma": 0.01})],
            representation="6d", cache_quats=cache_quats)
        for seed in range(20):
            with pytest.raises(ValueError, match="fk_topology"):
                pipe(arrays, rng=np.random.default_rng(seed))

    def test_rotation_noise_on_a_rotation_free_sample_raises(
            self, bvh_example):
        arrays = MotionArrays.from_bvh(
            bvh_example, None, include_positions=True)
        with pytest.raises(ValueError, match="nothing to noise"):
            add_joint_rotation_noise(arrays, sigma=0.01)


# =========================================================================
# The four geometric steps, on positions
# =========================================================================

class TestGeometricStepsOnPositions:

    @pytest.fixture
    def arrays(self, bvh_example):
        return MotionArrays.from_bvh(
            bvh_example, "6d", include_positions=True,
            position_centering="skeleton")

    def test_rotate_vertical_rotates_every_vertex(self, arrays, bvh_example):
        angle = 0.7
        out = rotate_vertical(arrays, angle=angle, up_axis=bvh_example.world_up,
                              representation="6d")
        # A rigid rotation preserves every pairwise distance.
        def gram(p):
            return np.einsum("fvc,fwc->fvw", p, p)
        np.testing.assert_allclose(
            gram(out.joint_pos), gram(arrays.joint_pos), atol=1e-9)
        assert not np.allclose(out.joint_pos, arrays.joint_pos)

    def test_mirror_twice_is_the_identity(self, arrays, bvh_example):
        pairs = list(bvh_example.lr_pairs)
        once = mirror(arrays, lr_joint_pairs=pairs, lateral_axis="+x",
                      representation="6d")
        twice = mirror(once, lr_joint_pairs=pairs, lateral_axis="+x",
                       representation="6d")
        np.testing.assert_allclose(twice.joint_pos, arrays.joint_pos,
                                   atol=1e-12)

    def test_speed_perturbation_resamples_every_stream_together(
            self, arrays):
        out = speed_perturbation_arrays(arrays, factor=1.7,
                                        representation="6d")
        assert out.joint_pos.shape[0] == out.root_pos.shape[0]
        assert out.joint_rot.shape[0] == out.root_pos.shape[0]
        assert out.joint_pos.shape[0] != arrays.joint_pos.shape[0]
        # Endpoints are knots of the stencil, so they pass through exactly.
        np.testing.assert_allclose(out.joint_pos[0], arrays.joint_pos[0],
                                   atol=1e-12)
        np.testing.assert_allclose(out.joint_pos[-1], arrays.joint_pos[-1],
                                   atol=1e-12)

    def test_dropout_keeps_shape_and_endpoints(self, arrays):
        out = dropout_arrays(arrays, drop_rate=0.4, representation="6d",
                             rng=np.random.default_rng(0))
        assert out.joint_pos.shape == arrays.joint_pos.shape
        np.testing.assert_allclose(out.joint_pos[0], arrays.joint_pos[0])
        np.testing.assert_allclose(out.joint_pos[-1], arrays.joint_pos[-1])
        assert not np.allclose(out.joint_pos, arrays.joint_pos)

    def test_keypoint_jitter_moves_every_vertex_independently(
            self, bvh_example):
        arrays = MotionArrays.from_bvh(
            bvh_example, None, include_positions=True)
        out = add_joint_position_noise(arrays, sigma=0.5,
                                       rng=np.random.default_rng(0))
        delta = out.joint_pos - arrays.joint_pos
        assert delta.std() > 0.1
        # Not one shared offset per frame: vertices differ within a frame.
        assert delta[0].std(axis=0).min() > 0.0

    def test_position_streams_are_not_written_through(self, arrays):
        """Container fields are read-only views; every position branch has
        to take a writable copy before mutating."""
        before = np.array(arrays.joint_pos)
        for out in (
            mirror(arrays, lr_joint_pairs=[(1, 2)], lateral_axis="+x",
                   representation="6d"),
            dropout_arrays(arrays, drop_rate=0.5, representation="6d",
                           rng=np.random.default_rng(1)),
        ):
            assert out is not arrays
        np.testing.assert_array_equal(arrays.joint_pos, before)


class TestMirrorInBothIndexSpaces:
    """A round trip alone proves nothing here: pybvh's own 0.8.2 notes
    record that ``mirror(mirror(x)) == x`` held while end sites were being
    left unswapped.  The load-bearing assertion is that one mirror maps
    each left vertex's trajectory onto its right partner's."""

    @staticmethod
    def _reflect(positions, lateral_idx=0):
        out = np.array(positions)
        out[:, :, lateral_idx] *= -1.0
        return out

    def test_joint_space_maps_partners_onto_each_other(self, bvh_example):
        arrays = MotionArrays.from_bvh(
            bvh_example, None, include_positions=True)
        pairs = list(bvh_example.lr_pairs)
        assert pairs
        out = mirror(arrays, lr_joint_pairs=pairs, lateral_axis="+x")
        reflected = self._reflect(arrays.joint_pos)
        for left, right in pairs:
            np.testing.assert_allclose(out.joint_pos[:, left],
                                       reflected[:, right], atol=1e-12)
            np.testing.assert_allclose(out.joint_pos[:, right],
                                       reflected[:, left], atol=1e-12)

    def test_node_space_swaps_end_sites_too(self, bvh_example):
        arrays = MotionArrays.from_bvh(
            bvh_example, None, include_positions=True, position_space="node")
        info = get_skeleton_info(bvh_example)
        pairs = info["node_lr_pairs"]
        joint_idx = np.asarray(info["fk_topology"]["joint_idx"])
        end_site_pairs = [(left, right) for left, right in pairs
                          if joint_idx[left] < 0]
        assert end_site_pairs, "rig has no paired end sites to check"

        out = mirror(arrays, lr_node_pairs=pairs, lateral_axis="+x")
        reflected = self._reflect(arrays.node_pos)
        for left, right in end_site_pairs:
            np.testing.assert_allclose(out.node_pos[:, left],
                                       reflected[:, right], atol=1e-12)

    def test_node_stream_without_node_pairs_raises(self, bvh_example):
        arrays = MotionArrays.from_bvh(
            bvh_example, None, include_positions=True, position_space="node")
        with pytest.raises(ValueError, match="lr_node_pairs"):
            mirror(arrays, lateral_axis="+x")

    def test_joint_stream_without_joint_pairs_raises(self, bvh_example):
        arrays = MotionArrays.from_bvh(
            bvh_example, None, include_positions=True)
        with pytest.raises(ValueError, match="lr_joint_pairs"):
            mirror(arrays, lateral_axis="+x")


# =========================================================================
# The FK refresh
# =========================================================================

class TestFkRefresh:

    @pytest.mark.parametrize("cache_quats", [True, False])
    def test_streams_are_fk_partners_after_rotation_noise(
            self, bvh_example, cache_quats):
        info = get_skeleton_info(bvh_example)
        topology = build_fk_topology(info)
        arrays = MotionArrays.from_bvh(
            bvh_example, "6d", include_positions=True)
        pipe = AugmentationPipeline(
            [(add_joint_rotation_noise, 1.0,
              {"sigma": np.radians(3.0), "fk_topology": topology})],
            representation="6d", cache_quats=cache_quats)
        out = pipe(arrays, rng=np.random.default_rng(0))

        assert not np.allclose(out.joint_pos, arrays.joint_pos)
        np.testing.assert_allclose(
            out.joint_pos,
            _fk_joint_positions(out, topology, "6d", "world"),
            atol=1e-9)

    def test_node_space_refresh(self, bvh_example):
        info = get_skeleton_info(bvh_example)
        topology = build_fk_topology(info)
        arrays = MotionArrays.from_bvh(
            bvh_example, "6d", include_positions=True, position_space="node",
            position_centering="skeleton")
        out = add_joint_rotation_noise(
            arrays, sigma=np.radians(2.0), representation="6d",
            fk_topology=topology, rng=np.random.default_rng(0))
        euler = rotations.convert(out.joint_rot, "6d", "euler",
                                  order=list(topology.euler_orders))
        expected = frames_to_node_positions(
            topology, np.asarray(out.root_pos), euler, centered="skeleton")
        np.testing.assert_allclose(out.node_pos, expected, atol=1e-9)

    def test_missing_topology_raises_on_the_first_call(self, bvh_example):
        arrays = MotionArrays.from_bvh(
            bvh_example, "6d", include_positions=True)
        with pytest.raises(ValueError, match="fk_topology"):
            add_joint_rotation_noise(
                arrays, sigma=0.01, representation="6d",
                rng=np.random.default_rng(0))

    def test_first_centering_requires_world_up(self, bvh_example):
        info = get_skeleton_info(bvh_example)
        topology = build_fk_topology(info)
        arrays = MotionArrays.from_bvh(
            bvh_example, "6d", include_positions=True,
            position_centering="first")
        with pytest.raises(ValueError, match="world_up"):
            add_joint_rotation_noise(
                arrays, sigma=0.01, representation="6d",
                fk_topology=topology, rng=np.random.default_rng(0))
        out = add_joint_rotation_noise(
            arrays, sigma=0.01, representation="6d", fk_topology=topology,
            world_up=info["world_up"], rng=np.random.default_rng(0))
        assert out.joint_pos.shape == arrays.joint_pos.shape

    def test_rotation_free_sample_is_untouched_by_the_refresh(
            self, bvh_example):
        """The refresh fires only when a rotation-space step meets a
        position stream — never speculatively."""
        arrays = MotionArrays.from_bvh(
            bvh_example, None, include_positions=True)
        out = add_root_position_noise(arrays, sigma=0.0,
                                      rng=np.random.default_rng(0))
        np.testing.assert_array_equal(out.joint_pos, arrays.joint_pos)


class TestReDerivationOrderDependence:
    """Documents, rather than forbids, that a re-derivation discards the
    position stream's own history.  A regression here would mean a step
    stopped re-deriving, or started to."""

    def test_mirror_then_noise_differs_from_noise_then_mirror(
            self, asymmetric_rig):
        info = get_skeleton_info(asymmetric_rig)
        topology = build_fk_topology(info)
        arrays = MotionArrays.from_bvh(
            asymmetric_rig, "6d", include_positions=True)
        noise = (add_joint_rotation_noise, 1.0,
                 {"sigma": 0.0, "fk_topology": topology})
        mirror_step = (mirror, 1.0, {"lr_joint_pairs": info["lr_pairs"],
                                     "lateral_axis": "+x"})

        first = AugmentationPipeline(
            [mirror_step, noise], representation="6d")(
                arrays, rng=np.random.default_rng(0))
        second = AugmentationPipeline(
            [noise, mirror_step], representation="6d")(
                arrays, rng=np.random.default_rng(0))

        # sigma=0, so the rotations are the same either way...
        np.testing.assert_allclose(first.joint_rot, second.joint_rot,
                                   atol=1e-9)
        # ...but [mirror, noise] ends with FK of locally-mirrored
        # rotations, while [noise, mirror] keeps the world-exact
        # reflection the position stream held.
        assert not np.allclose(first.joint_pos, second.joint_pos, atol=1e-6)
        np.testing.assert_allclose(
            first.joint_pos,
            _fk_joint_positions(first, topology, "6d", "world"), atol=1e-9)


# =========================================================================
# Centering-dependent surfaces
# =========================================================================

class TestPositionCentering:

    def test_root_noise_translates_world_positions(self, bvh_example):
        arrays = MotionArrays.from_bvh(
            bvh_example, "6d", include_positions=True)
        out = add_root_position_noise(arrays, sigma=0.7,
                                      rng=np.random.default_rng(0))
        delta_root = out.root_pos - arrays.root_pos
        delta_joints = out.joint_pos - arrays.joint_pos
        np.testing.assert_allclose(
            delta_joints, np.broadcast_to(delta_root[:, None, :],
                                          delta_joints.shape), atol=1e-12)

    def test_root_noise_leaves_skeleton_positions_alone(self, bvh_example):
        arrays = MotionArrays.from_bvh(
            bvh_example, "6d", include_positions=True,
            position_centering="skeleton")
        out = add_root_position_noise(arrays, sigma=0.7,
                                      rng=np.random.default_rng(0))
        np.testing.assert_array_equal(out.joint_pos, arrays.joint_pos)
        assert not np.allclose(out.root_pos, arrays.root_pos)

    def test_root_noise_reaches_a_joint_pos_only_pack(self, bvh_example):
        """The silent-no-op case: under "world" centering with
        streams=("joint_pos",) the root is not packed at all, so a root
        jitter that failed to move the positions would be invisible to
        the model *and* to any test that inspects root_pos."""
        arrays = MotionArrays.from_bvh(
            bvh_example, "6d", include_positions=True)
        out = add_root_position_noise(arrays, sigma=0.7,
                                      rng=np.random.default_rng(0))
        before = pack_to_ctv(arrays, center_root=False,
                             streams=("joint_pos",))
        after = pack_to_ctv(out, center_root=False, streams=("joint_pos",))
        assert not np.allclose(before, after)

    def test_undeclared_centering_raises_where_it_matters(self, bvh_example):
        arrays = MotionArrays(
            root_pos=bvh_example.root_pos,
            joint_pos=bvh_example.joint_positions())
        with pytest.raises(ValueError, match="position_centering is None"):
            add_root_position_noise(arrays, sigma=0.5,
                                    rng=np.random.default_rng(0))
        with pytest.raises(ValueError, match="position_centering is None"):
            pack_to_ctv(arrays, center_root=True, streams=("joint_pos",))

    def test_undeclared_centering_runs_the_agnostic_pipeline(
            self, bvh_example):
        """The other half of the fail-at-use contract: without this, a
        later "tighten the validation" change could turn None into a
        construction error and nothing would fail."""
        arrays = MotionArrays(
            root_pos=bvh_example.root_pos,
            joint_pos=bvh_example.joint_positions())
        pipe = AugmentationPipeline([
            (mirror, 1.0, {"lr_joint_pairs": list(bvh_example.lr_pairs),
                           "lateral_axis": "+x"}),
            (speed_perturbation_arrays, 1.0, {"factor": 1.1}),
            (dropout_arrays, 1.0, {"drop_rate": 0.2}),
            (add_joint_position_noise, 1.0, {"sigma": 0.05}),
        ])
        out = pipe(arrays, rng=np.random.default_rng(0))
        assert out.position_centering is None
        packed = pack_to_ctv(out, center_root=False, streams=("joint_pos",))
        assert packed.shape[0] == 3


# =========================================================================
# Packing
# =========================================================================

class TestStreamsPacking:

    @pytest.fixture
    def arrays(self, bvh_example):
        return MotionArrays.from_bvh(
            bvh_example, "6d", include_positions=True,
            position_centering="skeleton")

    def test_default_streams_are_byte_identical_to_0_5_0(self, arrays):
        rotations_only = MotionArrays(root_pos=arrays.root_pos,
                                      joint_rot=arrays.joint_rot)
        np.testing.assert_array_equal(
            pack_to_ctv(arrays, center_root=False),
            pack_to_ctv(rotations_only, center_root=False))

    def test_joint_pos_only_is_stgcn_shaped(self, arrays, bvh_example):
        packed = pack_to_ctv(arrays, center_root=False,
                             streams=("joint_pos",))
        assert packed.shape == (3, bvh_example.frame_count,
                                bvh_example.joint_count)

    def test_multi_stream_concatenates_on_c(self, arrays, bvh_example):
        packed = pack_to_ctv(arrays, center_root=False,
                             streams=("joint_pos", "joint_rot"))
        assert packed.shape == (9, bvh_example.frame_count,
                                bvh_example.joint_count)
        np.testing.assert_allclose(
            packed[:3].transpose(1, 2, 0), arrays.joint_pos)

    def test_node_pos_cannot_join_a_joint_space_stream(self, bvh_example):
        arrays = MotionArrays.from_bvh(
            bvh_example, "6d", include_positions=True, position_space="node")
        with pytest.raises(ValueError, match="cannot combine 'node_pos'"):
            pack_to_ctv(arrays, streams=("node_pos", "joint_rot"))

    def test_missing_stream_names_include_positions(self, bvh_example):
        arrays = MotionArrays.from_bvh(bvh_example, "6d")
        with pytest.raises(ValueError, match="include_positions=True"):
            pack_to_ctv(arrays, streams=("joint_pos",))

    def test_flat_column_order_follows_streams(self, arrays, bvh_example):
        flat = pack_to_flat(arrays, center_root=False,
                            streams=("joint_pos", "joint_rot"))
        J = bvh_example.joint_count
        assert flat.shape == (bvh_example.frame_count, J * 3 + J * 6)
        layout = describe_features(J, streams=("joint_pos", "joint_rot"))
        np.testing.assert_allclose(
            flat[:, layout.slice("joint_positions")],
            arrays.joint_pos.reshape(bvh_example.frame_count, -1))
        assert layout.total_dim == flat.shape[1]

    def test_center_root_shifts_position_vertices_too(self, bvh_example):
        arrays = MotionArrays.from_bvh(bvh_example, "6d",
                                       include_positions=True)
        packed = pack_to_ctv(arrays, center_root=True,
                             streams=("root_pos", "joint_pos"))
        shift = np.asarray(arrays.root_pos)[0]
        np.testing.assert_allclose(
            packed[:, :, 1:].transpose(1, 2, 0),
            arrays.joint_pos - shift, atol=1e-12)
        np.testing.assert_allclose(packed[:, 0, 0], 0.0, atol=1e-12)

    def test_describe_features_needs_num_nodes_for_node_blocks(self):
        with pytest.raises(ValueError, match="num_nodes"):
            describe_features(24, streams=("node_pos",))


# =========================================================================
# Skeleton metadata
# =========================================================================

class TestNodeSpaceMetadata:

    def test_end_site_mismatch_reports_the_node_space_pair(
            self, mismatch_rig):
        """The joint-space misreading of this check returns (2, 4) —
        LeftArm/RightArm — where the correct node-space one returns
        (4, 8), LeftHand/RightHand.  Asserting on the *pair* is what
        separates them; asserting merely that the list is non-empty
        cannot."""
        assert find_mismatched_end_site_pairs(mismatch_rig) == [(4, 8)]
        info = get_skeleton_info(mismatch_rig)
        assert info["mismatched_end_site_pairs"] == [(4, 8)]

    def test_upstream_drops_the_mismatched_tips(self, mismatch_rig):
        """Which is why we check at preprocessing time: the stored pair
        list would otherwise half-swap the skeleton at train time."""
        pairs = get_skeleton_info(mismatch_rig)["node_lr_pairs"]
        joint_idx = np.asarray(
            get_skeleton_info(mismatch_rig)["fk_topology"]["joint_idx"])
        assert all(joint_idx[left] >= 0 for left, _ in pairs)

    def test_well_formed_rig_reports_no_mismatch(self, bvh_example):
        assert find_mismatched_end_site_pairs(bvh_example) == []

    def test_fk_topology_roundtrips_through_json(self, bvh_example):
        info = json.loads(json.dumps(get_skeleton_info(bvh_example)))
        topology = build_fk_topology(info)
        np.testing.assert_allclose(
            np.asarray(frames_to_node_positions(
                topology, bvh_example.root_pos, bvh_example.joint_angles)),
            bvh_example.node_positions())

    def test_build_fk_topology_names_the_missing_key(self):
        with pytest.raises(ValueError, match="fk_topology"):
            build_fk_topology({"num_joints": 3})

    def test_joint_slice_of_node_positions_matches(self, bvh_example):
        info = get_skeleton_info(bvh_example)
        joint_idx = np.asarray(info["fk_topology"]["joint_idx"])
        np.testing.assert_allclose(
            bvh_example.node_positions()[:, joint_idx >= 0],
            bvh_example.joint_positions())

    def test_name_collision_metadata_survives_the_round_trip(
            self, collision_rig, tmp_path):
        """Upstream covers the derivation; this guards the storage path,
        which is the part only pybvh-ml owns."""
        assert collision_rig.joint_names.count("Tip") == 2
        live = get_skeleton_info(collision_rig)

        (tmp_path / "clips").mkdir()
        (tmp_path / "clips" / "a.bvh").write_text(NAME_COLLISION_BVH)
        out = tmp_path / "ds.npz"
        preprocess_directory(tmp_path / "clips", out, representation="6d",
                             include_positions=True)
        stored = load_preprocessed(out)["skeleton_info"]

        # json turns the live dict's tuples into lists, so normalize the
        # in-memory side the same way before comparing.
        live = json.loads(json.dumps(live))
        for key in ("edges", "node_edges", "num_nodes", "node_names",
                    "end_site_indices", "lr_pairs", "node_lr_pairs"):
            assert stored[key] == live[key], key
        np.testing.assert_allclose(
            np.asarray(frames_to_node_positions(
                build_fk_topology(stored), collision_rig.root_pos,
                collision_rig.joint_angles)),
            collision_rig.node_positions())


# =========================================================================
# Storage
# =========================================================================

class TestPositionStorage:

    @pytest.fixture
    def clip_dir(self, tmp_path, bvh_dir):
        d = tmp_path / "clips"
        d.mkdir()
        (d / "one.bvh").write_text((bvh_dir / "bvh_test1.bvh").read_text())
        return d

    @pytest.mark.parametrize("ext", [".npz", ".hdf5"])
    @pytest.mark.parametrize("space,key", [("joint", "joint_pos"),
                                           ("node", "node_pos")])
    def test_round_trip(self, clip_dir, tmp_path, ext, space, key):
        if ext == ".hdf5":
            pytest.importorskip("h5py")
        out = tmp_path / f"ds{ext}"
        preprocess_directory(
            clip_dir, out, representation="6d", include_positions=True,
            position_space=space, position_centering="skeleton")
        loaded = load_preprocessed(out)
        clip = loaded["clips"][0]
        assert key in clip
        assert loaded["position_centering"] == "skeleton"
        assert loaded["skeleton_info"]["position_space"] == space
        assert loaded["position_stats"]["mean"].shape == (
            clip[key].shape[1] * 3,)

    def test_no_positions_means_no_metadata(self, clip_dir, tmp_path):
        out = tmp_path / "ds.npz"
        preprocess_directory(clip_dir, out, representation="6d")
        loaded = load_preprocessed(out)
        assert loaded["position_centering"] is None
        assert loaded["position_stats"] is None
        assert "position_space" not in loaded["skeleton_info"]
        assert "joint_pos" not in loaded["clips"][0]

    def test_center_root_with_first_centering_is_rejected(
            self, clip_dir, tmp_path):
        with pytest.raises(ValueError, match="center_root=True cannot"):
            preprocess_directory(
                clip_dir, tmp_path / "ds.npz", include_positions=True,
                center_root=True, position_centering="first")

    def test_center_root_moves_stored_world_positions(
            self, clip_dir, tmp_path):
        raw = tmp_path / "raw.npz"
        centered = tmp_path / "centered.npz"
        preprocess_directory(clip_dir, raw, include_positions=True,
                             center_root=False)
        preprocess_directory(clip_dir, centered, include_positions=True,
                             center_root=True)
        a = load_preprocessed(raw)["clips"][0]
        b = load_preprocessed(centered)["clips"][0]
        np.testing.assert_allclose(
            b["joint_pos"], a["joint_pos"] - a["root_pos"][0], atol=1e-12)

    def test_stats_are_a_separate_block(self, clip_dir, tmp_path):
        """The mean/std layout is a public contract; a D that changed with
        a preprocessing flag would make one file format mean two things."""
        plain = tmp_path / "plain.npz"
        with_pos = tmp_path / "pos.npz"
        preprocess_directory(clip_dir, plain, representation="6d")
        preprocess_directory(clip_dir, with_pos, representation="6d",
                             include_positions=True)
        assert (load_preprocessed(plain)["mean"].shape
                == load_preprocessed(with_pos)["mean"].shape)


# =========================================================================
# End-to-end through the Dataset classes
# =========================================================================

class TestDatasetStreams:

    @pytest.fixture
    def loaded(self, tmp_path, bvh_dir):
        pytest.importorskip("torch")
        d = tmp_path / "clips"
        d.mkdir()
        for name in ("one", "two"):
            (d / f"{name}.bvh").write_text(
                (bvh_dir / "bvh_test1.bvh").read_text())
        out = tmp_path / "ds.npz"
        preprocess_directory(d, out, representation="6d",
                             include_positions=True,
                             position_centering="skeleton")
        return load_preprocessed(out)

    def test_stgcn_shaped_batch(self, loaded):
        from pybvh_ml.torch import MotionDataset
        ds = MotionDataset.from_preprocessed(
            loaded, layout="ctv", streams=("joint_pos",),
            temporal="resample", target_length=64, seed=0)
        J = loaded["skeleton_info"]["num_joints"]
        assert tuple(ds[0]["data"].shape) == (3, 64, J)

    def test_node_space_batch(self, tmp_path, bvh_dir):
        pytest.importorskip("torch")
        from pybvh_ml.torch import OnTheFlyDataset
        ds = OnTheFlyDataset(
            [bvh_dir / "bvh_test1.bvh"], representation=None,
            include_positions=True, position_space="node",
            position_centering="skeleton", layout="ctv",
            streams=("node_pos",), temporal="crop", target_length=32)
        bvh = read_bvh_file(bvh_dir / "bvh_test1.bvh")
        assert tuple(ds[0]["data"].shape) == (3, 32, len(bvh.nodes))

    def test_augmented_positions_stay_fk_partners(self, loaded):
        from pybvh_ml.torch import MotionDataset
        pipe = AugmentationPipeline.standard(loaded["skeleton_info"])
        ds = MotionDataset.from_preprocessed(
            loaded, layout="ctv", streams=("joint_pos",), augmentation=pipe,
            temporal="crop", target_length=32, seed=0)
        assert tuple(ds[0]["data"].shape)[0] == 3

    def test_missing_stream_names_include_positions(self, tmp_path, bvh_dir):
        pytest.importorskip("torch")
        from pybvh_ml.torch import MotionDataset
        d = tmp_path / "clips"
        d.mkdir()
        (d / "one.bvh").write_text((bvh_dir / "bvh_test1.bvh").read_text())
        out = tmp_path / "ds.npz"
        preprocess_directory(d, out, representation="6d")
        with pytest.raises(ValueError, match="include_positions=True"):
            MotionDataset.from_preprocessed(
                load_preprocessed(out), streams=("joint_pos",))

    def test_from_preprocessed_threads_the_centering(self, loaded):
        from pybvh_ml.torch import MotionDataset
        ds = MotionDataset.from_preprocessed(loaded)
        assert ds.position_centering == "skeleton"
        assert ds._clip_arrays(0).position_centering == "skeleton"


# =========================================================================
# dtype
# =========================================================================

class TestPositionDtype:

    def test_float32_positions_beside_float64_rotations(self, bvh_example):
        """The per-stream rule matters more with positions than with
        rotations: float32 keypoints beside float64 rotations is the
        ordinary ST-GCN case."""
        info = get_skeleton_info(bvh_example)
        topology = build_fk_topology(info)
        source = MotionArrays.from_bvh(
            bvh_example, "6d", include_positions=True)
        arrays = MotionArrays(
            root_pos=np.asarray(source.root_pos, dtype=np.float32),
            joint_rot=source.joint_rot,
            joint_pos=np.asarray(source.joint_pos, dtype=np.float32),
            position_centering="world")
        assert arrays.joint_pos.dtype == np.float32

        pipe = AugmentationPipeline.standard(
            info, up_axis=info["world_up"])
        for step in pipe.augmentations:
            if step.fn is add_joint_rotation_noise:
                step.kwargs["fk_topology"] = topology
        out = pipe(arrays, rng=np.random.default_rng(0))
        assert out.root_pos.dtype == np.float32
        assert out.joint_pos.dtype == np.float32
        assert out.joint_rot.dtype == np.float64

    @pytest.mark.parametrize("prob", [0.0, 1.0])
    def test_outputs_never_alias_the_input_positions(self, bvh_example, prob):
        arrays = MotionArrays.from_bvh(
            bvh_example, None, include_positions=True)
        pipe = AugmentationPipeline(
            [(add_joint_position_noise, prob, {"sigma": 0.1})])
        out = pipe(arrays, rng=np.random.default_rng(0))
        assert not np.shares_memory(out.joint_pos, arrays.joint_pos)
