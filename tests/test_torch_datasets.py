"""PyTorch dataset and collate tests.

Split from ``test_pybvh_ml.py`` so the main suite collects without
torch: this module skips entirely when PyTorch is not installed.

Run with: pytest tests/test_torch_datasets.py -v
"""

import pytest
import numpy as np
from pathlib import Path

torch = pytest.importorskip("torch")

from pybvh_ml.torch import (
    EpochState, MotionDataset, OnTheFlyDataset, collate_motion_batch, rng_for,
)
from pybvh_ml.augmentation import rotate_vertical, add_joint_rotation_noise
from pybvh_ml.convert import convert_arrays
from pybvh_ml.pipeline import AugmentationPipeline
from pybvh_ml import MotionArrays
from helpers import as_pair, as_triple


class TestTorchDatasets:
    """Tests for PyTorch Dataset classes and collate function."""

    @pytest.fixture
    def sample_clips(self, bvh_example):
        """Create sample preprocessed clips."""
        root_pos, rot6d = bvh_example.to_6d()
        # Create 3 clips with different lengths
        clips = [
            {"root_pos": root_pos[:30].copy(), "joint_rot": rot6d[:30].copy()},
            {"root_pos": root_pos[:20].copy(), "joint_rot": rot6d[:20].copy()},
            {"root_pos": root_pos[:40].copy(), "joint_rot": rot6d[:40].copy()},
        ]
        return clips

    # --- MotionDataset ---

    def test_motion_dataset_len(self, sample_clips):
        ds = MotionDataset(sample_clips)
        assert len(ds) == 3

    def test_motion_dataset_getitem(self, sample_clips):
        ds = MotionDataset(sample_clips)
        item = ds[0]
        assert "data" in item
        assert "length" in item
        assert isinstance(item["data"], torch.Tensor)

    def test_motion_dataset_dtype(self, sample_clips):
        ds = MotionDataset(sample_clips)
        item = ds[0]
        assert item["data"].dtype == torch.float32

    def test_motion_dataset_with_labels(self, sample_clips):
        labels = np.array([0, 1, 2])
        ds = MotionDataset(sample_clips, labels=labels)
        item = ds[1]
        assert item["label"] == 1

    def test_motion_dataset_target_length(self, sample_clips):
        ds = MotionDataset(sample_clips, target_length=50)
        item = ds[0]
        assert item["data"].shape[0] == 50
        assert item["length"] == 30  # valid frames (clip shorter than target)

    def test_motion_dataset_target_length_crops(self, sample_clips):
        """Regression: a cropped clip reports the frames actually present, not its pre-crop length."""
        ds = MotionDataset(sample_clips, target_length=25)
        item = ds[0]  # 30-frame clip cropped to 25
        assert item["data"].shape[0] == 25
        assert item["length"] == 25

    # --- OnTheFlyDataset ---

    def test_onthefly_len(self, bvh_paths):
        ds = OnTheFlyDataset(bvh_paths)
        assert len(ds) == len(bvh_paths)

    def test_onthefly_getitem(self, bvh_paths):
        ds = OnTheFlyDataset(bvh_paths, representation="6d")
        item = ds[0]
        assert "data" in item
        assert isinstance(item["data"], torch.Tensor)
        assert item["data"].dtype == torch.float32

    def test_onthefly_target_length_reports_valid_frames(self, bvh_paths):
        """Same length semantics as MotionDataset: valid frames in the returned tensor."""
        num_frames = OnTheFlyDataset(bvh_paths, representation="6d")[0]["length"]
        assert num_frames > 10  # sanity: the fixture clip must be croppable

        cropped = OnTheFlyDataset(
            bvh_paths, representation="6d", target_length=10)[0]
        assert cropped["data"].shape[0] == 10
        assert cropped["length"] == 10

        padded = OnTheFlyDataset(
            bvh_paths, representation="6d", target_length=num_frames + 5)[0]
        assert padded["data"].shape[0] == num_frames + 5
        assert padded["length"] == num_frames

    # --- collate_motion_batch ---

    def test_collate_shapes(self, sample_clips):
        ds = MotionDataset(sample_clips, labels=np.array([0, 1, 2]))
        batch = [ds[i] for i in range(3)]
        collated = collate_motion_batch(batch)
        assert collated["data"].shape[0] == 3  # batch size
        assert collated["data"].shape[1] == 40  # max length
        assert collated["lengths"].shape == (3,)
        assert collated["mask"].shape == (3, 40)

    def test_collate_padding(self, sample_clips):
        ds = MotionDataset(sample_clips)
        batch = [ds[i] for i in range(3)]
        collated = collate_motion_batch(batch)
        # Clip 1 has 20 frames — padding should be zero after that
        assert torch.all(collated["data"][1, 20:] == 0)

    def test_collate_mask(self, sample_clips):
        ds = MotionDataset(sample_clips)
        batch = [ds[i] for i in range(3)]
        collated = collate_motion_batch(batch)
        # Clip 0: 30 frames → mask[0, :30] = True, mask[0, 30:] = False
        assert collated["mask"][0, :30].all()
        assert not collated["mask"][0, 30:].any()

    def test_collate_mask_with_cropped_and_padded_clips(self, sample_clips):
        """Regression: the mask must reflect the frames in the tensor, not the pre-standardization clip lengths."""
        ds = MotionDataset(sample_clips, target_length=25)
        batch = [ds[0], ds[1]]  # clip 0: 30→25 cropped; clip 1: 20→25 padded
        collated = collate_motion_batch(batch)
        assert collated["lengths"].tolist() == [25, 20]
        # Cropped clip: every frame in the tensor is valid.
        assert collated["mask"][0].all()
        # Padded clip: True exactly for the valid prefix.
        assert collated["mask"][1, :20].all()
        assert not collated["mask"][1, 20:].any()

    def test_collate_labels(self, sample_clips):
        labels = np.array([5, 3, 7])
        ds = MotionDataset(sample_clips, labels=labels)
        batch = [ds[i] for i in range(3)]
        collated = collate_motion_batch(batch)
        assert "labels" in collated
        assert collated["labels"].tolist() == [5, 3, 7]

    def test_collate_with_dataloader(self, sample_clips):
        from torch.utils.data import DataLoader
        labels = np.array([0, 1, 2])
        ds = MotionDataset(sample_clips, labels=labels)
        loader = DataLoader(ds, batch_size=2, collate_fn=collate_motion_batch)
        batch = next(iter(loader))
        assert batch["data"].shape[0] == 2
        assert "lengths" in batch
        assert "mask" in batch

    # --- set_epoch + seeded augmentation (0.3) ---

    def _make_seeded_dataset(self, sample_clips, seed):
        from pybvh_ml.augmentation import add_joint_rotation_noise
        from pybvh_ml.pipeline import AugmentationPipeline
        # Build a fresh quat-primary dataset so noise aug can run in place.
        from pybvh_ml.convert import convert_arrays
        quat_clips = []
        for c in sample_clips:
            rot6d = c["joint_rot"]
            quats = convert_arrays(rot6d, from_repr="6d", to_repr="quat")
            quat_clips.append({"root_pos": c["root_pos"].copy(),
                               "joint_rot": quats})
        pipeline = AugmentationPipeline([
            (add_joint_rotation_noise, 1.0, {"sigma": np.radians(5.0), "representation": "quat"}),
        ])
        return MotionDataset(
            quat_clips, target_length=30, augmentation=pipeline, seed=seed)

    def test_seed_reproducible_same_epoch(self, sample_clips):
        """Two datasets with the same seed + epoch produce identical tensors."""
        ds1 = self._make_seeded_dataset(sample_clips, seed=42)
        ds2 = self._make_seeded_dataset(sample_clips, seed=42)
        ds1.set_epoch(3)
        ds2.set_epoch(3)
        torch.testing.assert_close(ds1[0]["data"], ds2[0]["data"])

    def test_set_epoch_changes_output(self, sample_clips):
        """Different epochs produce different augmented tensors."""
        ds = self._make_seeded_dataset(sample_clips, seed=42)
        ds.set_epoch(0)
        a = ds[0]["data"].clone()
        ds.set_epoch(1)
        b = ds[0]["data"].clone()
        assert not torch.allclose(a, b)

    def test_seed_none_is_nondeterministic(self, sample_clips):
        """With seed=None, repeated __getitem__ uses fresh entropy."""
        from pybvh_ml.augmentation import add_joint_rotation_noise
        from pybvh_ml.pipeline import AugmentationPipeline
        from pybvh_ml.convert import convert_arrays
        quat_clips = [
            {"root_pos": c["root_pos"].copy(),
             "joint_rot": convert_arrays(
                 c["joint_rot"], from_repr="6d", to_repr="quat")}
            for c in sample_clips
        ]
        pipeline = AugmentationPipeline([
            (add_joint_rotation_noise, 1.0, {"sigma": np.radians(5.0), "representation": "quat"}),
        ])
        ds = MotionDataset(
            quat_clips, target_length=30, augmentation=pipeline, seed=None)
        a = ds[0]["data"].clone()
        b = ds[0]["data"].clone()
        # Vanishingly unlikely to collide.
        assert not torch.allclose(a, b)

    def test_warns_when_seeded_without_set_epoch(self, sample_clips):
        """Silently seeding without set_epoch is a quiet correctness bug."""
        ds = self._make_seeded_dataset(sample_clips, seed=42)
        with pytest.warns(UserWarning, match="set_epoch"):
            ds[0]
        # Warning should only fire once per instance.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ds[1]  # must not raise — already warned

    def test_no_warn_when_set_epoch_called(self, sample_clips):
        ds = self._make_seeded_dataset(sample_clips, seed=42)
        ds.set_epoch(0)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            ds[0]  # must not raise — contract acknowledged

    def test_no_warn_when_seed_none(self, sample_clips):
        """seed=None means fresh entropy per call — no epoch contract needed."""
        from pybvh_ml.augmentation import add_joint_rotation_noise
        from pybvh_ml.pipeline import AugmentationPipeline
        from pybvh_ml.convert import convert_arrays
        quat_clips = [
            {"root_pos": c["root_pos"].copy(),
             "joint_rot": convert_arrays(
                 c["joint_rot"], from_repr="6d", to_repr="quat")}
            for c in sample_clips
        ]
        pipeline = AugmentationPipeline([
            (add_joint_rotation_noise, 1.0, {"sigma": np.radians(5.0), "representation": "quat"}),
        ])
        ds = MotionDataset(
            quat_clips, target_length=30, augmentation=pipeline, seed=None)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            ds[0]

    def test_onthefly_set_epoch(self, bvh_paths):
        """OnTheFlyDataset also honors set_epoch when seeded."""
        from pybvh_ml.augmentation import rotate_vertical
        from pybvh_ml.pipeline import AugmentationPipeline
        pipeline = AugmentationPipeline([
            (rotate_vertical, 1.0, {
                "angle": lambda rng: rng.uniform(-np.pi, np.pi),
                "up_axis": "+y",
                "representation": "quat",
            }),
        ])
        ds = OnTheFlyDataset(
            bvh_paths, representation="quat", target_length=30,
            augmentation=pipeline, seed=7)
        ds.set_epoch(0)
        a = ds[0]["data"].clone()
        ds.set_epoch(5)
        b = ds[0]["data"].clone()
        assert not torch.allclose(a, b)
        ds.set_epoch(0)
        a2 = ds[0]["data"].clone()
        torch.testing.assert_close(a, a2)




class TestSetEpochWorkers:
    """set_epoch must reach DataLoader worker processes.

    Regression: the epoch used to be a plain int attribute — workers
    held a pickled copy, so with persistent_workers=True every epoch
    silently replayed the epoch-0 augmentation (and the
    missing-set_epoch warning was pickled as already-emitted).  The
    epoch now lives in shared memory.
    """

    @pytest.fixture
    def seeded_pair(self, bvh_example):
        """Two identically-constructed seeded datasets (loader + reference)."""
        root_pos, rot6d = bvh_example.to_6d()
        def make():
            clips = [
                {"root_pos": root_pos[:30].copy(),
                 "joint_rot": rot6d[:30].copy()},
                {"root_pos": root_pos[:20].copy(),
                 "joint_rot": rot6d[:20].copy()},
                {"root_pos": root_pos[:40].copy(),
                 "joint_rot": rot6d[:40].copy()},
            ]
            pipeline = AugmentationPipeline([
                (add_joint_rotation_noise, 1.0,
                 {"sigma": np.radians(5.0), "representation": "6d"}),
            ])
            return MotionDataset(clips, target_length=30,
                                 augmentation=pipeline, seed=7)
        return make(), make()

    def test_set_epoch_reaches_persistent_workers(self, seeded_pair):
        from torch.utils.data import DataLoader
        ds, ref = seeded_pair
        loader = DataLoader(
            ds, batch_size=1, shuffle=False, num_workers=2,
            persistent_workers=True, collate_fn=collate_motion_batch)
        per_epoch = []
        for epoch in (0, 1):
            ds.set_epoch(epoch)
            ref.set_epoch(epoch)
            samples = [batch["data"][0] for batch in loader]
            # Bit-equal to the single-process reference: the
            # (seed, epoch, idx) design is worker-count-invariant.
            for i, sample in enumerate(samples):
                torch.testing.assert_close(
                    sample, ref[i]["data"], rtol=0, atol=0)
            per_epoch.append(torch.stack(samples))
        assert not torch.equal(per_epoch[0], per_epoch[1])

    def test_set_epoch_reaches_spawn_workers(self, seeded_pair):
        """Same contract under a spawn-context loader.

        Regression: the shared Value used to come from the *default*
        context (fork on Linux), whose lock is an anonymous unlinked
        semaphore.  It unpickles into a spawn-started worker as a
        dangling handle and segfaults on first acquire, so this exact
        loader died with 'DataLoader worker exited unexpectedly'.
        """
        from torch.utils.data import DataLoader
        ds, ref = seeded_pair
        loader = DataLoader(
            ds, batch_size=1, shuffle=False, num_workers=2,
            persistent_workers=True, multiprocessing_context="spawn",
            collate_fn=collate_motion_batch)
        per_epoch = []
        for epoch in (0, 1):
            ds.set_epoch(epoch)
            ref.set_epoch(epoch)
            samples = [batch["data"][0] for batch in loader]
            for i, sample in enumerate(samples):
                torch.testing.assert_close(
                    sample, ref[i]["data"], rtol=0, atol=0)
            per_epoch.append(torch.stack(samples))
        assert not torch.equal(per_epoch[0], per_epoch[1])

    def test_epoch_lock_is_named_so_it_survives_spawn(self, seeded_pair):
        """The mechanism behind the test above, asserted directly.

        A fork-context semaphore has ``name is None`` (anonymous,
        unlinked at creation); only a named one can be reopened by a
        spawn-started child.
        """
        ds, _ = seeded_pair
        semlock = ds._epoch_state._epoch.get_lock()._semlock
        assert semlock.name is not None

    def test_set_epoch_negative_raises(self, seeded_pair):
        ds, _ = seeded_pair
        with pytest.raises(ValueError, match="epoch must be >= 0"):
            ds.set_epoch(-1)
        with pytest.raises(ValueError, match="epoch must be >= 0"):
            ds.set_epoch(-5)


class TestDatasetErgonomics:
    """Negative indexing, path coercion, passthroughs, honest errors."""

    @pytest.fixture
    def bvh_paths(self):
        bvh_dir = Path(__file__).parent.parent / "bvh_data"
        return sorted(bvh_dir.glob("bvh_test1.bvh"))

    @pytest.fixture
    def clips_6d(self, bvh_example):
        root_pos, rot6d = bvh_example.to_6d()
        return [
            {"root_pos": root_pos[:30].copy(), "joint_rot": rot6d[:30].copy()},
            {"root_pos": root_pos[:20].copy(), "joint_rot": rot6d[:20].copy()},
        ]

    def test_negative_index_matches_positive(self, clips_6d):
        """ds[-1] and ds[len-1] share the (seed, epoch, idx) stream.
        Regression: a raw negative index crashed SeedSequence, but only
        when seeded augmentation was active."""
        pipeline = AugmentationPipeline([
            (add_joint_rotation_noise, 1.0,
             {"sigma": np.radians(5.0), "representation": "6d"}),
        ])
        ds = MotionDataset(clips_6d, target_length=30,
                           augmentation=pipeline, seed=3)
        ds.set_epoch(0)
        torch.testing.assert_close(
            ds[-1]["data"], ds[len(ds) - 1]["data"], rtol=0, atol=0)

    def test_out_of_range_raises_index_error(self, clips_6d):
        ds = MotionDataset(clips_6d)
        with pytest.raises(IndexError):
            ds[len(ds)]
        with pytest.raises(IndexError):
            ds[-len(ds) - 1]

    def test_onthefly_accepts_str_paths(self, bvh_paths):
        """str paths work end to end, including label_fn (which needs
        .stem and used to crash mid-training)."""
        ds = OnTheFlyDataset(
            [str(p) for p in bvh_paths], representation="6d",
            label_fn=lambda stem: len(stem))
        item = ds[0]
        assert item["label"] == len(bvh_paths[0].stem)

    def test_onthefly_reader_kwargs_passthrough(self, bvh_paths, monkeypatch):
        """world_up= / lr_mapping= reach read_bvh_file per clip.

        (world_up doesn't change the extracted rotation arrays — it
        steers FK-derived interpretation — so the passthrough is
        asserted on the reader call itself.)
        """
        import pybvh_ml.torch.datasets as ds_mod
        recorded: dict = {}
        real_reader = ds_mod.read_bvh_file

        def spy(path, **kwargs):
            recorded.update(kwargs)
            return real_reader(path, **kwargs)

        monkeypatch.setattr(ds_mod, "read_bvh_file", spy)
        ds = OnTheFlyDataset(bvh_paths, representation="6d", world_up="+y")
        ds[0]
        assert recorded["world_up"] == "+y"
        assert recorded["lr_mapping"] is None

    def test_motion_dataset_center_root(self, clips_6d):
        """center_root=True zeroes the first-frame root columns of the
        flat tensor for raw (uncentered) clips."""
        assert not np.allclose(clips_6d[0]["root_pos"][0], 0.0), \
            "fixture clip should be uncentered for this test"
        ds = MotionDataset(clips_6d, center_root=True)
        torch.testing.assert_close(
            ds[0]["data"][0, :3], torch.zeros(3), rtol=0, atol=0)
        ds_raw = MotionDataset(clips_6d)
        assert not torch.allclose(ds_raw[0]["data"][0, :3], torch.zeros(3))

    def test_collate_mixed_labels_raises(self, clips_6d):
        items = [
            {"data": torch.zeros(5, 9), "length": 5, "label": 1},
            {"data": torch.zeros(5, 9), "length": 5},
        ]
        with pytest.raises(ValueError, match="some batch items but not all"):
            collate_motion_batch(items)
        # Reverse order too: unlabeled first used to silently drop labels.
        with pytest.raises(ValueError, match="some batch items but not all"):
            collate_motion_batch(items[::-1])


# =============================================================================
# explain_augmentation
# =============================================================================

class TestExplainAugmentation:
    """Replaying a sample's augmentation draw after the fact."""

    @pytest.fixture
    def bvh_paths(self):
        bvh_dir = Path(__file__).parent.parent / "bvh_data"
        return sorted(bvh_dir.glob("bvh_test1.bvh"))

    @pytest.fixture
    def clips_6d(self, bvh_example):
        root_pos, rot6d = bvh_example.to_6d()
        return [
            {"root_pos": root_pos[:30].copy(), "joint_rot": rot6d[:30].copy()},
            {"root_pos": root_pos[:20].copy(), "joint_rot": rot6d[:20].copy()},
        ]

    @staticmethod
    def _pipeline(rotate_prob=0.5):
        """Sampled angle behind a probability + shape-dependent noise.

        The noise step consumes randomness proportional to the clip's
        shape, so a replay that skipped it would misreport later steps.
        """
        return AugmentationPipeline([
            (rotate_vertical, rotate_prob, {
                "angle": lambda rng: rng.uniform(-np.pi, np.pi),
                "up_axis": "+y", "representation": "6d"}),
            (add_joint_rotation_noise, 1.0,
             {"sigma": np.radians(5.0), "representation": "6d"}),
        ])

    def test_replay_reproduces_the_sample_exactly(self, clips_6d):
        """The records describe the draw that really ran.

        Re-running the pipeline on the reported rng must rebuild the
        loader's own tensor bit for bit — if it does, the records
        alongside it are the true ones.
        """
        from pybvh_ml.torch import rng_for
        from pybvh_ml import pack_to_flat

        ds = MotionDataset(clips_6d, augmentation=self._pipeline(), seed=42)
        ds.set_epoch(3)
        loaded = ds[1]["data"]

        params = ds.explain_augmentation(1)
        rp, jd = as_pair(ds._clip_arrays(1))
        rp2, jd2, replay = as_triple(ds.augmentation(MotionArrays(root_pos=rp, joint_rot=jd), rng=rng_for(42, 3, 1), return_params=True))
        torch.testing.assert_close(
            loaded, torch.tensor(pack_to_flat(MotionArrays(root_pos=rp2, joint_rot=jd2), center_root=False),
                                 dtype=torch.float32), rtol=0, atol=0)
        assert params == replay

    def test_unseeded_dataset_refuses_to_answer(self, clips_6d):
        """A fresh draw would look like an answer — it must raise instead."""
        ds = MotionDataset(clips_6d, augmentation=self._pipeline())
        with pytest.raises(ValueError, match="without a seed"):
            ds.explain_augmentation(0)

    def test_without_augmentation_returns_empty(self, clips_6d):
        ds = MotionDataset(clips_6d, seed=1)
        assert ds.explain_augmentation(0) == []

    def test_epoch_defaults_to_current(self, clips_6d):
        ds = MotionDataset(clips_6d, augmentation=self._pipeline(), seed=7)
        ds.set_epoch(2)
        assert ds.explain_augmentation(0) == ds.explain_augmentation(0, epoch=2)

    def test_different_epochs_report_different_draws(self, clips_6d):
        ds = MotionDataset(clips_6d, augmentation=self._pipeline(), seed=7)
        angles = []
        for epoch in range(6):
            record = ds.explain_augmentation(0, epoch=epoch)[0]
            angles.append(record["params"].get("angle"))
        assert len({str(a) for a in angles}) > 1

    def test_negative_index_matches_positive(self, clips_6d):
        ds = MotionDataset(clips_6d, augmentation=self._pipeline(), seed=5)
        assert ds.explain_augmentation(-1) == ds.explain_augmentation(
            len(ds) - 1)

    def test_out_of_range_raises_index_error(self, clips_6d):
        ds = MotionDataset(clips_6d, augmentation=self._pipeline(), seed=5)
        with pytest.raises(IndexError):
            ds.explain_augmentation(99)

    def test_negative_epoch_raises(self, clips_6d):
        ds = MotionDataset(clips_6d, augmentation=self._pipeline(), seed=5)
        with pytest.raises(ValueError, match="epoch must be >= 0"):
            ds.explain_augmentation(0, epoch=-1)

    def test_does_not_consume_the_set_epoch_warning(self, clips_6d):
        """A diagnostic read must not mask the real training warning."""
        ds = MotionDataset(clips_6d, augmentation=self._pipeline(), seed=9)
        ds.explain_augmentation(0)          # before any __getitem__
        with pytest.warns(UserWarning, match="set_epoch"):
            ds[0]

    def test_onthefly_replay_reproduces_the_sample(self, bvh_paths):
        from pybvh_ml.torch import rng_for
        from pybvh_ml import pack_to_flat

        ds = OnTheFlyDataset(bvh_paths, representation="6d",
                             augmentation=self._pipeline(), seed=11)
        ds.set_epoch(1)
        loaded = ds[0]["data"]
        params = ds.explain_augmentation(0)
        rp, jd = as_pair(ds._clip_arrays(0))
        rp2, jd2, replay = as_triple(ds.augmentation(MotionArrays(root_pos=rp, joint_rot=jd), rng=rng_for(11, 1, 0), return_params=True))
        torch.testing.assert_close(
            loaded, torch.tensor(pack_to_flat(MotionArrays(root_pos=rp2, joint_rot=jd2), center_root=False),
                                 dtype=torch.float32), rtol=0, atol=0)
        assert params == replay
        assert [r["name"] for r in params] == [
            "rotate_vertical", "add_joint_rotation_noise"]


class TestSeedingPrimitives:
    """`rng_for` / `EpochState` are public — a Dataset that isn't a
    MotionDataset subclass needs exactly this pair, and copying it into
    downstream projects is what the export exists to prevent."""

    def test_exported_from_torch_namespace(self):
        import pybvh_ml.torch as t
        assert {"EpochState", "rng_for"} <= set(t.__all__)

    def test_rng_for_is_order_and_worker_independent(self):
        # Same triple, same stream, regardless of when it is built.
        a = rng_for(7, 2, 5).standard_normal(4)
        b = rng_for(7, 2, 5).standard_normal(4)
        np.testing.assert_array_equal(a, b)

    @pytest.mark.parametrize("triple", [(7, 2, 6), (7, 3, 5), (8, 2, 5)])
    def test_rng_for_varies_with_every_term(self, triple):
        base = rng_for(7, 2, 5).standard_normal(4)
        other = rng_for(*triple).standard_normal(4)
        assert not np.array_equal(base, other)

    def test_rng_for_unseeded_is_fresh_entropy(self):
        a = rng_for(None, 0, 0).standard_normal(4)
        b = rng_for(None, 0, 0).standard_normal(4)
        assert not np.array_equal(a, b)

    def test_epoch_state_current_defaults_to_zero_without_warning(self):
        import warnings as _warnings
        state = EpochState()
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            assert state.current == 0
        assert caught == []

    def test_epoch_state_set_and_read(self):
        state = EpochState()
        state.set(4)
        assert state.current == 4

    def test_epoch_state_rejects_negative(self):
        with pytest.raises(ValueError, match="epoch must be >= 0"):
            EpochState().set(-1)

    def test_downstream_dataset_pattern(self, bvh_example):
        """The documented usage: hold an EpochState, seed with rng_for.
        Same sample must differ across epochs and repeat within one."""

        state = EpochState()
        state.set(0)
        first = rng_for(3, state.current, 1).standard_normal(4)
        again = rng_for(3, state.current, 1).standard_normal(4)
        state.set(1)
        next_epoch = rng_for(3, state.current, 1).standard_normal(4)
        np.testing.assert_array_equal(first, again)
        assert not np.array_equal(first, next_epoch)


class TestDatasetLayouts:
    """`layout=` reaches the (C, T, V) / (T, V, C) packers the library
    already ships but the Dataset used to skip."""

    @pytest.fixture
    def clips(self, bvh_example):
        root_pos, rot6d = bvh_example.to_6d()
        return [{"root_pos": root_pos[:30].copy(),
                 "joint_rot": rot6d[:30].copy()}]

    def test_flat_is_the_default(self, clips):
        ds = MotionDataset(clips)
        assert ds[0]["data"].dim() == 2

    def test_ctv_layout(self, clips, bvh_example):
        J = bvh_example.joint_count
        ds = MotionDataset(clips, layout="ctv")
        # (C, T, V): C = max(3, 6) = 6, T = 30, V = 1 + J
        assert tuple(ds[0]["data"].shape) == (6, 30, 1 + J)

    def test_tvc_layout(self, clips, bvh_example):
        J = bvh_example.joint_count
        ds = MotionDataset(clips, layout="tvc")
        assert tuple(ds[0]["data"].shape) == (30, 1 + J, 6)

    def test_layout_matches_the_standalone_packer(self, clips):
        from pybvh_ml import pack_to_ctv
        ds = MotionDataset(clips, layout="ctv")
        expected = pack_to_ctv(MotionArrays(root_pos=clips[0]["root_pos"], joint_rot=clips[0]["joint_rot"]), center_root=False)
        torch.testing.assert_close(
            ds[0]["data"], torch.tensor(expected, dtype=torch.float32),
            rtol=0, atol=0)

    def test_unknown_layout_raises(self, clips):
        with pytest.raises(ValueError, match="layout must be one of"):
            MotionDataset(clips, layout="tcv")

    def test_onthefly_layout(self, bvh_paths):
        ds = OnTheFlyDataset(bvh_paths, representation="6d", layout="ctv")
        assert ds[0]["data"].dim() == 3

    def test_graph_layout_is_rejected_by_the_padding_collate(self, clips):
        """(C, T, V) puts channels on axis 0, which is where the collate
        pads — silently masking along channels is exactly the corruption
        the guard exists to prevent."""
        ds = MotionDataset(clips, layout="ctv", target_length=30)
        with pytest.raises(ValueError, match="expects 2-D"):
            collate_motion_batch([ds[0]])

    def test_graph_layout_stacks_with_default_collate(self, clips):
        from torch.utils.data import default_collate
        ds = MotionDataset(clips + clips, layout="ctv", target_length=16,
                           temporal="resample_deterministic")
        batch = default_collate([ds[0], ds[1]])
        assert batch["data"].shape[0] == 2


class TestDatasetTemporalModes:
    """`temporal=` — crop/pad keeps a fixed window, resample keeps the
    whole arc at a fixed frame budget."""

    @pytest.fixture
    def clips(self, bvh_example):
        root_pos, rot6d = bvh_example.to_6d()
        return [
            {"root_pos": root_pos[:40].copy(), "joint_rot": rot6d[:40].copy()},
            {"root_pos": root_pos[:12].copy(), "joint_rot": rot6d[:12].copy()},
        ]

    def test_pad_remains_the_default(self, clips):
        ds = MotionDataset(clips, target_length=20)
        assert ds.temporal == "pad"
        # Long clip truncated from the end, short one zero-padded.
        assert ds[0]["length"] == 20
        assert ds[1]["length"] == 12
        assert torch.all(ds[1]["data"][12:] == 0)

    def test_crop_takes_the_centre(self, clips):
        ds = MotionDataset(clips, target_length=20, temporal="crop")
        item = ds[0]
        assert item["data"].shape[0] == 20
        expected_start = (40 - 20) // 2
        np.testing.assert_allclose(
            item["data"][0, :3].numpy(),
            clips[0]["root_pos"][expected_start], rtol=1e-6)

    @pytest.mark.parametrize("temporal", ["resample",
                                          "resample_deterministic"])
    def test_resample_fills_the_whole_budget(self, clips, temporal):
        """No padding: every frame is real data, so length is always the
        full target — including for a clip shorter than the budget."""
        ds = MotionDataset(clips, target_length=20, temporal=temporal, seed=0)
        for i in range(len(clips)):
            item = ds[i]
            assert item["data"].shape[0] == 20
            assert item["length"] == 20
            assert not torch.all(item["data"][-1] == 0)

    def test_resample_spans_the_clip(self, clips):
        """The arc, not a window: the last sampled frame comes from late
        in a clip that crop/pad would have truncated."""
        ds = MotionDataset(clips, target_length=8,
                           temporal="resample_deterministic")
        last = ds[0]["data"][-1, :3].numpy()
        late = clips[0]["root_pos"][30:]
        assert any(np.allclose(last, f, rtol=1e-6) for f in late)

    def test_resample_deterministic_repeats(self, clips):
        ds = MotionDataset(clips, target_length=16,
                           temporal="resample_deterministic")
        torch.testing.assert_close(ds[0]["data"], ds[0]["data"],
                                   rtol=0, atol=0)
        fresh = MotionDataset(clips, target_length=16,
                              temporal="resample_deterministic")
        torch.testing.assert_close(ds[0]["data"], fresh[0]["data"],
                                   rtol=0, atol=0)

    def test_resample_varies_across_epochs_when_seeded(self, clips):
        ds = MotionDataset(clips, target_length=16, temporal="resample",
                           seed=5)
        ds.set_epoch(0)
        first = ds[0]["data"].clone()
        ds.set_epoch(1)
        assert not torch.equal(first, ds[0]["data"])
        ds.set_epoch(0)
        torch.testing.assert_close(first, ds[0]["data"], rtol=0, atol=0)

    def test_resample_needs_a_target_length(self, clips):
        with pytest.raises(ValueError, match="no length to standardize to"):
            MotionDataset(clips, temporal="resample")

    def test_unknown_temporal_raises(self, clips):
        with pytest.raises(ValueError, match="temporal must be one of"):
            MotionDataset(clips, target_length=10, temporal="resample_linear")

    def test_resample_rejects_an_empty_clip(self, bvh_example):
        root_pos, rot6d = bvh_example.to_6d()
        empty = [{"root_pos": root_pos[:0].copy(),
                  "joint_rot": rot6d[:0].copy()}]
        ds = MotionDataset(empty, target_length=8,
                           temporal="resample_deterministic")
        with pytest.raises(ValueError, match="0 frames"):
            ds[0]

    def test_onthefly_resample(self, bvh_paths):
        ds = OnTheFlyDataset(bvh_paths, representation="6d",
                             target_length=24, temporal="resample",
                             seed=1)
        assert ds[0]["length"] == 24


class TestDatasetRepresentationConversion:
    """`target_repr=` — the Dataset calls convert_arrays so a dataset
    stored in one representation can train in another."""

    @pytest.fixture
    def euler_clips(self, bvh_example):
        return [{"root_pos": bvh_example.root_pos[:20].copy(),
                 "joint_rot": bvh_example.joint_angles[:20].copy()}]

    def test_converts_to_target(self, euler_clips, bvh_example):
        orders = list(bvh_example.euler_orders)
        ds = MotionDataset(euler_clips, source_repr="euler",
                           target_repr="6d", euler_orders=orders)
        expected = convert_arrays(euler_clips[0]["joint_rot"], "euler", "6d",
                                  euler_orders=orders)
        J = bvh_example.joint_count
        assert ds[0]["data"].shape == (20, 3 + J * 6)
        np.testing.assert_allclose(
            ds[0]["data"][:, 3:].numpy().reshape(20, J, 6),
            expected, rtol=1e-5, atol=1e-6)

    def test_target_repr_without_source_repr_raises(self, euler_clips):
        with pytest.raises(ValueError, match="target_repr requires source_repr"):
            MotionDataset(euler_clips, target_repr="6d")

    def test_no_conversion_by_default(self, euler_clips, bvh_example):
        J = bvh_example.joint_count
        ds = MotionDataset(euler_clips)
        assert ds[0]["data"].shape == (20, 3 + J * 3)

    def test_conversion_precedes_augmentation(self, euler_clips, bvh_example):
        """The pipeline sees the target representation, so its declared
        `representation` is target_repr — and explain_augmentation, which
        replays through _clip_arrays, stays truthful."""
        orders = list(bvh_example.euler_orders)
        pipeline = AugmentationPipeline(
            [(rotate_vertical, 1.0, {"angle": lambda rng: rng.uniform(-1, 1),
                                     "up_axis": bvh_example.world_up})],
            representation="6d")
        ds = MotionDataset(euler_clips, source_repr="euler", target_repr="6d",
                           euler_orders=orders, augmentation=pipeline, seed=2)
        _, jd = as_pair(ds._clip_arrays(0))
        assert jd.shape[-1] == 6
        assert ds.explain_augmentation(0)[0]["applied"] is True


class TestFromPreprocessed:
    """The constructor that wires dataset metadata instead of making the
    caller restate it."""

    @pytest.fixture
    def dataset_file(self, tmp_path):
        from pybvh_ml import preprocess_directory
        bvh_dir = Path(__file__).parent.parent / "bvh_data"
        out = tmp_path / "ds.npz"
        preprocess_directory(bvh_dir, out, representation="euler",
                             file_pattern="bvh_test1.bvh",
                             label_fn=lambda stem: 0)
        return out

    def test_wires_clips_labels_and_metadata(self, dataset_file):
        from pybvh_ml import load_preprocessed
        loaded = load_preprocessed(dataset_file)
        ds = MotionDataset.from_preprocessed(loaded)
        assert len(ds) == len(loaded["clips"])
        assert ds.source_repr == "euler"
        assert ds.euler_orders == loaded["skeleton_info"]["euler_orders"]
        assert ds[0]["label"] == 0
        # Stored arrays already reflect the preprocessing center_root.
        assert ds.center_root is False

    def test_target_repr_works_without_restating_the_source(self,
                                                            dataset_file):
        from pybvh_ml import load_preprocessed
        loaded = load_preprocessed(dataset_file)
        ds = MotionDataset.from_preprocessed(
            loaded, target_repr="6d", layout="ctv", temporal="resample",
            target_length=32, seed=0)
        J = loaded["skeleton_info"]["num_joints"]
        assert tuple(ds[0]["data"].shape) == (6, 32, 1 + J)

    def test_explicit_kwargs_override_metadata(self, dataset_file):
        from pybvh_ml import load_preprocessed
        loaded = load_preprocessed(dataset_file)
        ds = MotionDataset.from_preprocessed(loaded, labels=None)
        assert "label" not in ds[0]


class TestTemporalAndAugmentationInteraction:
    """The two stochastic stages share one per-sample generator — the
    ordering and the deterministic-mode carve-out both matter."""

    @pytest.fixture
    def clips(self, bvh_example):
        root_pos, rot6d = bvh_example.to_6d()
        return [{"root_pos": root_pos[:40].copy(),
                 "joint_rot": rot6d[:40].copy()}]

    def _pipeline(self, bvh_example):
        return AugmentationPipeline(
            [(rotate_vertical, 1.0, {"angle": lambda rng: rng.uniform(-1, 1),
                                     "up_axis": bvh_example.world_up})],
            representation="6d")

    def test_deterministic_resample_ignores_the_augmentation_stream(
            self, clips, bvh_example):
        """`resample_deterministic` must pick the same frames every
        epoch even with augmentation active.

        uniform_temporal_sample honors a supplied rng in test mode too,
        so handing it the augmentation-advanced per-sample generator
        would make the frame choice drift with the epoch while still
        calling itself deterministic.  A sigma=0 noise step isolates
        that: it consumes exactly the draws real noise would, but leaves
        the arrays alone — so any epoch-to-epoch difference in the
        output is the frame selection moving, and nothing else.
        """
        pipeline = AugmentationPipeline(
            [(add_joint_rotation_noise, 1.0, {"sigma": 0.0})], representation="6d")
        ds = MotionDataset(clips, target_length=16, seed=3,
                           temporal="resample_deterministic",
                           augmentation=pipeline)
        ds.set_epoch(0)
        first = ds[0]["data"].clone()
        for epoch in (1, 7):
            ds.set_epoch(epoch)
            torch.testing.assert_close(first, ds[0]["data"],
                                       rtol=1e-5, atol=1e-6)

        # And it agrees with the same selection made with no augmentation
        # at all — the augmentation stream never reached it.
        plain = MotionDataset(clips, target_length=16,
                              temporal="resample_deterministic")
        torch.testing.assert_close(first, plain[0]["data"],
                                   rtol=1e-5, atol=1e-6)

    def test_explain_augmentation_stays_exact_under_resample(
            self, clips, bvh_example):
        """Augmentation draws first, resampling continues on the same
        stream — so replaying only the augmentation still lands on the
        identical draw."""
        ds = MotionDataset(clips, target_length=16, seed=8,
                           temporal="resample",
                           augmentation=self._pipeline(bvh_example))
        ds.set_epoch(2)
        params = ds.explain_augmentation(0)
        rp, jd = as_pair(ds._clip_arrays(0))
        _, _, replay = as_triple(ds.augmentation(MotionArrays(root_pos=rp, joint_rot=jd), rng=rng_for(8, 2, 0), return_params=True))
        assert params == replay

    def test_seeded_getitem_is_reproducible_with_both_stages(
            self, clips, bvh_example):
        def build():
            ds = MotionDataset(clips, target_length=16, seed=9,
                               temporal="resample",
                               augmentation=self._pipeline(bvh_example))
            ds.set_epoch(4)
            return ds
        torch.testing.assert_close(build()[0]["data"], build()[0]["data"],
                                   rtol=0, atol=0)
