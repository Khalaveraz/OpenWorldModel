"""Week 1 storage, recovery, policy, and solvability integration checks."""

from __future__ import annotations

from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from predictive_control.__main__ import collect_episodes, diagnose
from predictive_control.baselines import RandomPolicy, ReactivePolicy
from predictive_control.config import load_config
from predictive_control.data import EpisodeStore, episode_seed, pack_episode, validate_episode, write_once
from predictive_control.environment import PhysicsSandbox
import predictive_control.data as data_module


class DataIntegrationTests(unittest.TestCase):
    """Use temporary stores; tests never change the project's datasets."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(ROOT / "configs/stage1.toml")
        cls.examples = []
        env = PhysicsSandbox(cls.config)
        try:
            for index, (task, stratum) in enumerate((("reach", "broad"), ("push", "near"), ("push", "broad"))):
                root = cls.config.dataset["pilot_seed"]
                policy = RandomPolicy(cls.config)
                seed = episode_seed(root, index, 1)
                policy.reset(seed)
                first_force = policy.act()
                while not np.linalg.norm(first_force):
                    first_force = policy.act()
                direction = first_force / np.linalg.norm(first_force)
                options = {"task": task, "stratum": stratum, "layout_index": 0,
                           "agent_position": [0.5, 0.5], "goal_position": [0.8, 0.8]}
                if task == "push":
                    options["object_position"] = (np.array([0.5, 0.5]) + 0.105 * direction).tolist()
                obs, _ = env.reset(seed=episode_seed(root, index), options=options)
                policy.reset(seed)
                transitions, diagnostics = [], [env.diagnostic_state()]
                for _ in range(200):
                    obs, _, done, truncated, _ = env.step(policy.act(obs))
                    transitions.append(env.last_transition)
                    diagnostics.append(env.diagnostic_state())
                    if done or truncated:
                        break
                cls.examples.append(pack_episode(transitions, diagnostics, cls.config, {
                    "split": "pilot", "collection_index": index,
                    "environment_seed": episode_seed(root, index), "policy_seed": seed,
                    "policy_identity": policy.identity,
                    "source": {"test_fixture": "controlled resets for storage checks; not campaign data"}}))
        finally:
            env.close()

    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.store = EpisodeStore(Path(self.temporary.name) / "data")

    def populate(self) -> list[str]:
        return [self.store.append_completed_episode(e) for e in self.examples]

    def test_roundtrip_preserves_final_observation_and_hides_diagnostics(self) -> None:
        identity = self.store.append_completed_episode(self.examples[0])
        loaded = self.store.load_episode(identity)
        self.assertNotIn("diagnostics", loaded)
        for key, value in self.examples[0]["arrays"].items():
            np.testing.assert_array_equal(loaded["arrays"][key], value)
            self.assertFalse(loaded["arrays"][key].flags.writeable)
        self.assertEqual(len(loaded["metadata"]["observation_ids"]), loaded["metadata"]["transitions"] + 1)
        self.assertTrue(loaded["arrays"]["truncated"][-1])
        self.assertEqual(loaded["arrays"]["continuation"][-1], 1)
        self.assertFalse(loaded["arrays"]["reset"][-1])
        with self.assertRaises(FileExistsError):
            self.store.append_completed_episode(self.examples[0])

    def test_terminal_success_survives_serialization(self) -> None:
        config = load_config(ROOT / "configs/stage1.toml", {"dataset.zero_force_probability": 1.0})
        env = PhysicsSandbox(config)
        try:
            env.reset(seed=episode_seed(config.dataset["pilot_seed"], 0), options={"task": "reach", "layout_index": 0,
                                      "agent_position": [0.3, 0.3], "goal_position": [0.3, 0.3]})
            transitions, diagnostics = [], [env.diagnostic_state()]
            for _ in range(3):
                env.step(np.zeros(2, dtype=np.float32))
                transitions.append(env.last_transition)
                diagnostics.append(env.diagnostic_state())
            # A distinct factual controller identity is not admitted to the random store.
            provenance = {"split": "pilot", "collection_index": 0,
                          "environment_seed": episode_seed(config.dataset["pilot_seed"], 0),
                          "policy_seed": episode_seed(config.dataset["pilot_seed"], 0, 1),
                          "policy_identity": "privileged-test", "source": {"test_fixture": "terminal"}}
            with self.assertRaises(ValueError):
                pack_episode(transitions, diagnostics, config, provenance)
            provenance["policy_identity"] = RandomPolicy.identity
            identity = self.store.append_completed_episode(pack_episode(transitions, diagnostics, config, provenance))
            saved = self.store.load_episode(identity)["arrays"]
            self.assertTrue(saved["terminated"][-1])
            self.assertEqual(saved["continuation"][-1], 0)
            self.assertEqual(len(saved["rgb"]), 4)
        finally:
            env.close()

    def test_bad_alignment_unknown_execution_and_imagination_are_rejected(self) -> None:
        for field, value in (("reset", True), ("dt", 0.0), ("executed_action", 9.0), ("continuation", 0.0)):
            with self.subTest(field=field):
                example = deepcopy(self.examples[0])
                example["arrays"][field][-1] = value
                with self.assertRaises(ValueError):
                    self.store.append_completed_episode(example)
        for key, value in (("evidence_kind", "imagined"), ("action_status", ["unknown"] * 200)):
            example = deepcopy(self.examples[0])
            example["metadata"][key] = value
            with self.assertRaises(ValueError):
                self.store.append_completed_episode(example)

    def test_corruption_is_detected_and_cannot_be_sealed(self) -> None:
        identity = self.store.append_completed_episode(self.examples[0])
        with (self.store.episodes / identity / "observed.npz").open("ab") as handle:
            handle.write(b"damage")
        with self.assertRaises(ValueError):
            self.store.load_episode(identity)
        report = self.store.audit("pilot")
        self.assertEqual(report["integrity"], "fail")
        with self.assertRaises(ValueError):
            self.store.seal([identity], "pilot", reviewed_audit=report["audit_hash"])

    def test_reviewed_snapshot_reopens_and_is_immutable(self) -> None:
        identities = self.populate()
        report = self.store.audit("pilot")
        self.assertEqual(report["integrity"], "pass")
        self.assertEqual(report["coverage_status"], "review_required", report["coverage_concerns"])
        for key in ("push/near", "push/broad"):
            self.assertGreater(report["groups"][key]["contact_episodes"], 0)
        with self.assertRaises(ValueError):
            self.store.seal(identities, "pilot", reviewed_audit="stale")
        snapshot = self.store.seal(identities, "pilot", reviewed_audit=report["audit_hash"])
        reopened = EpisodeStore(self.store.root).verify_snapshot("pilot")
        self.assertEqual(snapshot.dataset_hash, reopened.dataset_hash)
        original_hash = data_module.file_hash
        with patch.object(data_module, "file_hash", side_effect=lambda path: (
                "0" * 64 if path == Path(data_module.__file__) else original_hash(path))):
            # Later replay code in data.py must not invalidate the recorded audit.
            self.assertEqual(self.store.verify_snapshot("pilot").dataset_hash, snapshot.dataset_hash)
        with self.assertRaises(ValueError):
            self.store.append_completed_episode(self.examples[0])
        record = self.store.episodes / identities[0] / "record.json"
        record.write_text(record.read_text() + " ", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.store.verify_snapshot("pilot")

    def test_resume_preserves_commits_and_seed_isolation(self) -> None:
        config = load_config(ROOT / "configs/stage1.toml", {"environment.max_episode_steps": 12})
        with redirect_stdout(StringIO()):
            partial = collect_episodes(config, self.store, "pilot", 48, episode_limit=1)
            before = (self.store.episodes / "pilot-00000000" / "record.json").read_bytes()
            # A killed writer's uncommitted directory is never mistaken for an episode.
            (self.store.episodes / ".pending-interrupted").mkdir()
            completed = collect_episodes(config, self.store, "pilot", 48)
            again = collect_episodes(config, self.store, "pilot", 48)
            collect_episodes(config, self.store, "validation", 48)
        self.assertFalse(partial["complete"])
        self.assertTrue(completed["complete"])
        self.assertEqual(again["new_episodes"], 0)
        self.assertEqual(before, (self.store.episodes / "pilot-00000000" / "record.json").read_bytes())
        self.assertEqual(len(self.store.ids("pilot")), 4)
        self.assertEqual(self.store.audit("pilot")["integrity"], "pass")
        other = EpisodeStore(Path(self.temporary.name) / "replica" / "data")
        with redirect_stdout(StringIO()):
            collect_episodes(config, other, "pilot", 48)
        for identity in other.ids("pilot"):
            for key, value in other.load_episode(identity)["arrays"].items():
                np.testing.assert_array_equal(value, self.store.load_episode(identity)["arrays"][key])
        with self.assertRaises(FileExistsError):
            collect_episodes(config, self.store, "pilot", 50)

    def test_seed_blocks_storage_budget_and_path_validation(self) -> None:
        for split in ("pilot", "train", "validation", "test"):
            root = self.config.dataset[f"{split}_seed"]
            self.assertLess(episode_seed(root, 100000, 1), episode_seed(root + 1, 0))
        with self.assertRaises(ValueError):
            self.store.load_episode("../outside")
        tiny = EpisodeStore(Path(self.temporary.name) / "tiny" / "data", max_bytes=1)
        with self.assertRaises(OSError):
            tiny.append_completed_episode(self.examples[0])
        self.assertEqual(tiny.ids("pilot"), [])
        record = Path(self.temporary.name) / "immutable.json"
        write_once(record, {"a": 1})
        write_once(record, {"a": 1})
        with self.assertRaises(FileExistsError):
            write_once(record, {"a": 2})

    def test_random_hold_repeatability_and_reactive_missing_input(self) -> None:
        a, b = RandomPolicy(self.config), RandomPolicy(self.config)
        a.reset(7)
        b.reset(7)
        forces = np.stack([a.act() for _ in range(3000)])
        np.testing.assert_array_equal(forces, np.stack([b.act() for _ in range(3000)]))
        blocks = forces.reshape(-1, 3, 2)
        np.testing.assert_array_equal(blocks[:, 0], blocks[:, 1])
        np.testing.assert_array_equal(blocks[:, 1], blocks[:, 2])
        zero_fraction = np.mean(np.all(blocks[:, 0] == 0, axis=1))
        self.assertGreater(zero_fraction, 0.05)
        self.assertLess(zero_fraction, 0.15)
        episode = self.examples[0]["arrays"]
        observation = {"rgb": episode["rgb"][0], "proprio": episode["proprio"][0],
                       "goal": episode["goal"], "validity": np.array([False, True])}
        np.testing.assert_array_equal(ReactivePolicy().act(observation), [0, 0])

    def test_controlled_solvability(self) -> None:
        result = diagnose(self.config)
        self.assertEqual(result["status"], "pass", result)


class EncoderIntegrationTests(unittest.TestCase):
    """Real pretrained features on isolated, explicitly controlled test fixtures."""

    @classmethod
    def setUpClass(cls):
        import torch
        from predictive_control.encoder import FrozenVisualEncoder, cache_features, fit_normalizer
        torch.set_num_threads(min(4, torch.get_num_threads()))
        cls.temporary = TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.store = EpisodeStore(Path(cls.temporary.name) / "data")
        cls.config = load_config(ROOT / "configs/stage1.toml")
        cls.encoder = FrozenVisualEncoder(cls.config)
        env = PhysicsSandbox(cls.config)
        try:
            for index, (task, stratum) in enumerate((("reach", "broad"), ("push", "near"), ("push", "broad"))):
                root = cls.config.dataset["train_seed"]
                policy = RandomPolicy(cls.config)
                policy_seed = episode_seed(root, index, 1)
                policy.reset(policy_seed)
                force = policy.act()
                while not np.linalg.norm(force):
                    force = policy.act()
                options = {"task": task, "stratum": stratum, "layout_index": 0,
                           "agent_position": [.5, .5], "goal_position": [.8, .8]}
                if task == "push":
                    options["object_position"] = (np.array([.5, .5]) + .105 * force / np.linalg.norm(force)).tolist()
                observation, _ = env.reset(seed=episode_seed(root, index), options=options)
                policy.reset(policy_seed)
                transitions, diagnostics = [], [env.diagnostic_state()]
                for _ in range(200):
                    observation, _, done, truncated, _ = env.step(policy.act(observation))
                    transitions.append(env.last_transition)
                    diagnostics.append(env.diagnostic_state())
                    if done or truncated:
                        break
                cls.store.append_completed_episode(pack_episode(transitions, diagnostics, cls.config,
                    {"split": "train", "collection_index": index, "environment_seed": episode_seed(root, index),
                     "policy_seed": policy_seed, "policy_identity": policy.identity,
                     "source": {"test_fixture": "controlled reset; not campaign data"}}))
        finally:
            env.close()
        report = cls.store.audit("train")
        if report["integrity"] != "pass" or report["coverage_concerns"]:
            raise AssertionError(report["failures"] + report["coverage_concerns"])
        cls.snapshot = cls.store.seal(cls.store.ids("train"), "train", reviewed_audit=report["audit_hash"])
        cls.cache = cache_features(cls.snapshot, cls.encoder, cls.config.augmentation, store=cls.store)
        cls.normalizer = fit_normalizer(cls.snapshot, cache=cls.cache, store=cls.store)

    def test_frozen_spatial_encoder_and_fresh_cache_tolerance(self):
        import torch
        from predictive_control.encoder import load_cached_episode, encoder_diagnostics
        before = {k: v.clone() for k, v in self.encoder.state_dict().items()}
        self.encoder.train(True)
        self.assertFalse(any(m.training for m in self.encoder.modules()))
        self.assertFalse(any(p.requires_grad for p in self.encoder.parameters()))
        cached = load_cached_episode(self.cache, self.snapshot.episode_ids[0], store=self.store)
        fresh = self.encoder.encode(cached["source"]["arrays"]["rgb"][:3])
        self.assertEqual(tuple(fresh.shape), (3, 4608))
        self.assertFalse(fresh.requires_grad)
        np.testing.assert_allclose(cached["arrays"]["clean_visual"][:3].astype(np.float32), fresh.numpy(), rtol=.001, atol=.001)
        self.assertEqual(encoder_diagnostics(self.encoder)["status"], "pass")
        for key, value in self.encoder.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_corruptions_reproducible_independent_and_preserve_clean_geometry(self):
        from predictive_control.encoder import make_corrupted_view
        episode = self.store.load_episode(self.snapshot.episode_ids[0])
        clean = episode["arrays"]["rgb"].copy()
        first = make_corrupted_view(episode, self.config.augmentation, 17)
        second = make_corrupted_view(episode, self.config.augmentation, 17)
        for key in ("rgb", "proprio", "validity", "sample_age", "disk_positions"):
            np.testing.assert_array_equal(first[key], second[key])
        self.assertTrue((~first["validity"][:, 0]).any())
        np.testing.assert_array_equal(clean, episode["arrays"]["rgb"])
        cfg = dict(self.config.augmentation, pixel_noise_std=0., proprio_noise_std_m_per_s=0.,
                   occlusion_start_probability=0., distractor_episode_probability=1.)
        disk = make_corrupted_view(episode, cfg, 17)
        geometry = ~np.all(clean == (245, 245, 245), axis=-1)
        np.testing.assert_array_equal(disk["rgb"][geometry], clean[geometry])
        unrelated = {"metadata": episode["metadata"], "arrays": dict(episode["arrays"],
                       executed_action=-episode["arrays"]["executed_action"], goal=np.array([0, 1, .1, .1]),
                       reward=episode["arrays"]["reward"] + 100)}
        other = make_corrupted_view(unrelated, cfg, 17)
        np.testing.assert_array_equal(disk["disk_positions"], other["disk_positions"])
        blackout = make_corrupted_view(episode, dict(cfg, occlusion_start_probability=1.), 17)
        self.assertFalse(blackout["validity"][:, 0].any())
        self.assertFalse(blackout["rgb"].any())
        self.assertTrue(np.all(np.diff(blackout["sample_age"][:, 0]) > 0))

    def test_normalizer_training_only_frozen_roundtrip_and_clean_targets(self):
        from dataclasses import replace
        from predictive_control.encoder import FeatureNormalizer, fit_normalizer, load_cached_episode
        with self.assertRaises(ValueError):
            fit_normalizer(replace(self.snapshot, split="pilot"), cache=self.cache, store=self.store)
        path = self.cache / "normalizer.json"
        self.normalizer.save(path)
        reopened = FeatureNormalizer.load(path)
        self.assertEqual(reopened.version, self.normalizer.version)
        self.assertFalse(reopened.mean.flags.writeable)
        all_clean = []
        for identity in self.snapshot.episode_ids:
            row = load_cached_episode(self.cache, identity, store=self.store)
            arrays = row["arrays"]
            sensory = np.concatenate((arrays["clean_visual"].astype(np.float32), arrays["clean_proprio"]), -1)
            all_clean.append(sensory)
            self.assertEqual(arrays["corrupted_visual"].shape, arrays["clean_visual"].shape)
            self.assertEqual(row["record"]["source_observation_ids"], row["source"]["metadata"]["observation_ids"])
            np.testing.assert_array_equal(arrays["clean_proprio"], row["source"]["arrays"]["proprio"])
        values = np.concatenate(all_clean).astype(np.float64)
        np.testing.assert_allclose(reopened.mean, values.mean(0), atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(reopened.std, np.maximum(values.std(0), .0001), atol=1e-6, rtol=1e-6)
        masked = reopened.transform(values[:2].astype(np.float32), np.array([[False, True], [True, False]]))
        self.assertEqual(masked.dtype, np.float32)
        self.assertFalse(masked[0, :4608].any())
        self.assertFalse(masked[1, 4608:].any())

    def test_cache_resume_and_corruption_detection(self):
        from predictive_control.encoder import cache_features, load_cached_episode
        path = cache_features(self.snapshot, self.encoder, self.config.augmentation, store=self.store)
        self.assertEqual(path, self.cache)
        identity = self.snapshot.episode_ids[0]
        original = (self.cache / f"{identity}.json").read_bytes()
        try:
            record = json.loads(original)
            record["source_observation_ids"][0] = "wrong-source"
            (self.cache / f"{identity}.json").write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                load_cached_episode(self.cache, identity, store=self.store)
        finally:
            (self.cache / f"{identity}.json").write_bytes(original)


if __name__ == "__main__":
    unittest.main()
