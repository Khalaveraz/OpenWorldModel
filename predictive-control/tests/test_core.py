"""CPU smoke checks for contracts, configuration, and observed physics behavior."""

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import sys
import unittest
import warnings

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from predictive_control.config import config_hash, load_config, validate_config
from predictive_control.contracts import (
    BeliefState, DatasetSnapshot, GateReport, Goal, ModalityEntry, Observation,
    PlanResult, PredictiveDistribution, SequenceBatch, validate_array, validate_units,
)
from predictive_control.environment import PhysicsSandbox


class CoreTests(unittest.TestCase):
    """Exercise the specified invariants without later-week modules or GPU work."""

    def config(self, **overrides):
        return load_config(ROOT / "configs" / "stage1.toml", overrides)

    def environment(self, **overrides):
        values = {"environment.linear_drag_kg_per_s": [0.0],
                  "environment.object_mass_kg": [1.0], "environment.restitution": [0.0]}
        values.update(overrides)
        env = PhysicsSandbox(self.config(**values))
        self.addCleanup(env.close)
        return env

    def test_configuration_is_deeply_immutable_and_hashed(self):
        config = self.config()
        validate_config(config)
        self.assertEqual(config_hash(config), config_hash(self.config()))
        with self.assertRaises(FrozenInstanceError):
            config.schema_version = 2
        with self.assertRaises(TypeError):
            config.environment["physics_dt"] = 1.0
        with self.assertRaises(TypeError):
            config.environment["obstacle_layouts"][1][0][0] = 0.0
        self.assertNotEqual(config_hash(config), config_hash(self.config(**{"dataset.train_seed": 42})))
        for override in ({"environment.unknown": 1}, {"environment.physics_dt": 0.1},
                         {"environment.max_episode_steps": True}, {"augmentation.pixel_noise_std": float("nan")}):
            with self.assertRaises(ValueError):
                self.config(**override)

    def test_validation_helpers_and_observation_snapshot(self):
        rgb = np.zeros((96, 96, 3), dtype=np.uint8)
        entry = ModalityEntry(rgb, rgb.shape, "uint8", "uint8", True, 0.0, "rgb", "v1")
        observation = Observation("o0", "ep", 0, 0.0, True, {"rgb": entry})
        rgb[:] = 255
        self.assertFalse(observation.modalities["rgb"].payload.any())
        with self.assertRaises(ValueError):
            validate_units("cm/s", "m/s", "velocity")
        with self.assertRaises(TypeError):
            validate_array(np.zeros(2, dtype=np.float64), (2,), "float32", "velocity")
        with self.assertRaises(ValueError):
            validate_array(torch.tensor([float("nan")]), (1,), "float32", "tensor")
        with self.assertRaises(ValueError):
            replace(observation, step_id=1)

    def test_other_shared_records_are_runnable(self):
        goal = Goal("push", np.array([0.8, 0.8], dtype=np.float32))
        np.testing.assert_array_equal(goal.as_array()[:2], [0, 1])
        snapshot = DatasetSnapshot(("ep",), "train", "sha256:dataset", {"arena_size_m": 1},
                                   {"kind": "observed"}, {"clean": "v1"}, "aug-v1", "norm-v1", "split-v1")
        with self.assertRaises(TypeError):
            snapshot.environment_configuration["arena_size_m"] = 2
        belief = BeliefState(torch.zeros(1, 512), torch.zeros(1, 64), torch.zeros(1, 64),
                             torch.ones(1, 64), ("o0",), "model-v1", "encoder-v1", "posterior")
        self.assertEqual(belief.stochastic_state.shape, (1, 64))
        with self.assertRaises(ValueError):
            replace(belief, latent_scale=torch.zeros(1, 64))
        distribution = PredictiveDistribution({"reward": torch.zeros(1, 1)},
                                             {"reward": torch.zeros(1, 1)},
                                             {"cost": torch.tensor([[100.0]])}, "fixed-v1")
        self.assertEqual(distribution.binary_logits["cost"].item(), 100)
        proposal = PlanResult(np.zeros(2, dtype=np.float32), 0, 0, 0, 0, 0, True,
                              "ep:o0", "model-v1", "encoder-v1", 7)
        with self.assertRaises(ValueError):
            replace(proposal, evidence_kind="observed")
        report = GateReport({"success": 0.5}, {"control": "random"}, {"success": (0.4, 0.6)},
                            "fail", "config-hash", {"test": "test-hash"}, ("checkpoint",), ("control threshold",))
        self.assertEqual(report.gate_status, "fail")

    def test_sequence_alignment_terminal_masks_and_boundaries(self):
        zeros = lambda *shape: np.zeros(shape, dtype=np.float32)
        batch = SequenceBatch(
            observed_prefix={"features": zeros(1, 1, 4610), "validity": np.ones((1, 1, 2), bool),
                             "sample_ages": zeros(1, 1, 2), "mask": np.ones((1, 1), bool),
                             "actions": zeros(1, 0, 2), "dt": zeros(1, 0)},
            learning_observations=zeros(1, 3, 4610), executed_actions=zeros(1, 2, 2),
            targets={"sensory": zeros(1, 3, 4610), "reward": zeros(1, 2), "cost": zeros(1, 2),
                     "continuation": np.array([[1, 0]], np.float32),
                     "terminated": np.array([[False, True]]), "truncated": np.zeros((1, 2), bool)},
            input_validity=np.ones((1, 3, 2), bool), sample_ages=zeros(1, 3, 2),
            target_validity=np.ones((1, 3, 2), bool), padding_mask=np.ones((1, 2), bool),
            observation_mask=np.ones((1, 3), bool), episode_boundaries=np.array([[True, False, False]]),
            episode_ids=("ep",), view_ids=("clean",), goals=np.array([[1, 0, 0.8, 0.8]], np.float32),
            elapsed_simulation_time=np.full((1, 2), 0.1, np.float32),
        )
        self.assertTrue(batch.padding_mask[0, -1])
        self.assertTrue(batch.targets["terminated"][0, -1])
        with self.assertRaises(ValueError):
            replace(batch, episode_boundaries=np.array([[True, False, True]]))
        with self.assertRaises(ValueError):
            replace(batch, padding_mask=np.array([[False, True]]))

    def test_gymnasium_contract_and_seeded_rollouts(self):
        env = self.environment()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            check_env(env, skip_render_check=True)
        for task in ("reach", "push"):
            results = []
            for _ in range(2):
                observation, _ = env.reset(seed=17, options={"task": task})
                self.assertTrue(env.observation_space.contains(observation))
                self.assertEqual(observation["rgb"].dtype, np.uint8)
                self.assertEqual(observation["rgb"].shape, (96, 96, 3))
                frames = [observation["rgb"]]
                rewards = []
                for _ in range(5):
                    observation, reward, terminated, truncated, info = env.step(np.array([0.2, -0.1], np.float32))
                    frames.append(observation["rgb"])
                    rewards.append(reward)
                    self.assertFalse(terminated or truncated)
                    self.assertNotIn("object_position", info)
                results.append((np.stack(frames), rewards))
            np.testing.assert_array_equal(results[0][0], results[1][0])
            np.testing.assert_allclose(results[0][1], results[1][1], atol=0, rtol=0)

    def test_force_is_applied_for_all_substeps_and_clipping_is_recorded(self):
        env = self.environment()
        env.reset(seed=2, options={"layout_index": 0, "agent_position": [0.3, 0.5], "goal_position": [0.8, 0.8]})
        np.testing.assert_array_equal(env.previous_executed_action, [0, 0])
        env.step(np.array([1.0, 0], np.float32))
        np.testing.assert_allclose(env.diagnostic_state()["agent_velocity"], [0.1, 0], atol=1e-7)
        _, _, _, _, info = env.step(np.array([8, -8], np.float32))
        np.testing.assert_array_equal(info["executed_action"], [2, -2])
        np.testing.assert_array_equal(env.last_transition.action_record.requested_force, [8, -8])
        self.assertEqual(env.last_transition.action_record.status, "clipped")
        self.assertAlmostEqual(env.last_transition.elapsed_simulation_time, 0.1)

    def test_reach_reward_and_time_limit_keep_nonterminal_potential(self):
        env = self.environment(**{"environment.max_episode_steps": 1})
        env.reset(seed=4, options={"layout_index": 0, "agent_position": [0.2, 0.2], "goal_position": [0.8, 0.8]})
        state = env.diagnostic_state()
        phi = -np.linalg.norm(state["agent_position"] - state["goal_position"])
        _, reward, terminated, truncated, _ = env.step(np.zeros(2, np.float32))
        self.assertAlmostEqual(reward, -0.01 + 2 * (0.99 * phi - phi))
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(env.last_transition.continuation, 1)
        self.assertFalse(env.last_transition.next_observation.reset)
        with self.assertRaises(gym.error.ResetNeeded):
            env.step(np.zeros(2, np.float32))

    def test_success_requires_three_decisions_and_zero_terminal_potential(self):
        env = self.environment()
        env.reset(seed=4, options={"layout_index": 0, "agent_position": [0.25, 0.25], "goal_position": [0.25, 0.25]})
        for index in range(3):
            _, reward, terminated, truncated, _ = env.step(np.zeros(2, np.float32))
            self.assertEqual(terminated, index == 2)
            self.assertFalse(truncated)
        self.assertAlmostEqual(reward, 0.99)
        self.assertEqual(env.last_transition.continuation, 0)
        self.assertEqual(env.last_transition.next_observation.step_id, 3)
        unknown = replace(env.last_transition.action_record, status="unknown", executed_force=None,
                          executed_duration=None, execution_timestamp=None, reason="missing acknowledgement")
        self.assertFalse(replace(env.last_transition, action_record=unknown).training_eligible)

    def test_push_reward_contact_and_object_motion(self):
        env = self.environment()
        env.reset(seed=5, options={"task": "push", "stratum": "near", "layout_index": 0,
                                  "agent_position": [0.3, 0.5], "object_position": [0.4, 0.5], "goal_position": [0.8, 0.5]})
        state = env.diagnostic_state()
        phi = -np.linalg.norm(state["object_position"] - state["goal_position"]) - 0.25 * np.linalg.norm(state["agent_position"] - state["object_position"])
        _, reward, _, _, _ = env.step(np.zeros(2, np.float32))
        self.assertAlmostEqual(reward, -0.01 + 2 * (0.99 * phi - phi))
        contacted = False
        for _ in range(8):
            env.step(np.array([2, 0], np.float32))
            contacted |= env.diagnostic_state()["agent_object_contact"]
        self.assertTrue(contacted)
        self.assertGreater(env.diagnostic_state()["object_position"][0], 0.41)

    def test_hazard_is_separate_binary_feedback(self):
        env = self.environment()
        env.reset(seed=6, options={"layout_index": 0, "agent_position": [0.8, 0.12], "goal_position": [0.2, 0.8]})
        _, _, terminated, truncated, info = env.step(np.zeros(2, np.float32))
        self.assertEqual(info["cost"], 1.0)
        self.assertFalse(terminated or truncated)
        self.assertEqual(env.last_transition.cost, 1.0)

    def test_invalid_actions_do_not_advance_the_world(self):
        env = self.environment()
        env.reset(seed=8)
        for action in ([float("nan"), 0], [float("inf"), 0], [0], [True, False], ["1", "2"]):
            with self.assertRaises(ValueError):
                env.step(action)
            self.assertEqual(env.diagnostic_state()["step_id"], 0)
        with self.assertRaises(ValueError):
            env.reset(seed=8, options={"unexpected": True})

    def test_reset_strata_and_returned_arrays_are_independent(self):
        env = self.environment()
        for seed in range(12):
            observation, _ = env.reset(seed=seed, options={"task": "push", "stratum": "near"})
            state = env.diagnostic_state()
            distance = np.linalg.norm(state["agent_position"] - state["object_position"])
            self.assertGreaterEqual(distance, 0.1 - 1e-8)
            self.assertLessEqual(distance, 0.2 + 1e-8)
            observation["rgb"][:] = 0
            self.assertTrue(env.last_observation.modalities["rgb"].payload.any())
            state["agent_position"][:] = 0
            self.assertTrue(env.diagnostic_state()["agent_position"].any())


def rssm_batch(transitions=3):
    """Synthetic shape/mask fixture; never evidence of learned physical skill."""
    n = transitions + 1
    features = torch.randn(1, n, 4610)
    validity = torch.ones(1, n, 2, dtype=torch.bool)
    ages = torch.zeros(1, n, 2)
    actions = torch.zeros(1, transitions, 2)
    boundaries = torch.zeros(1, n, dtype=torch.bool)
    boundaries[:, 0] = True
    return SequenceBatch(
        observed_prefix={"features": features[:, :1], "validity": validity[:, :1], "sample_ages": ages[:, :1],
                         "mask": torch.ones(1, 1, dtype=torch.bool), "actions": actions[:, :0], "dt": torch.zeros(1, 0)},
        learning_observations=features, executed_actions=actions,
        targets={"sensory": features.clone(), "reward": torch.zeros(1, transitions), "cost": torch.zeros(1, transitions),
                 "continuation": torch.ones(1, transitions), "terminated": torch.zeros(1, transitions, dtype=torch.bool),
                 "truncated": torch.zeros(1, transitions, dtype=torch.bool)},
        input_validity=validity, sample_ages=ages, target_validity=validity.clone(),
        padding_mask=torch.ones(1, transitions, dtype=torch.bool), observation_mask=torch.ones(1, n, dtype=torch.bool),
        episode_boundaries=boundaries, episode_ids=("test-episode",), view_ids=("clean",),
        goals=torch.tensor([[1., 0., .8, .8]]), elapsed_simulation_time=torch.full((1, transitions), .1))


class RSSMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from predictive_control.model import RSSM
        torch.set_num_threads(min(4, torch.get_num_threads()))
        torch.manual_seed(5)
        cls.model = RSSM(load_config(ROOT / "configs/stage1.toml"), encoder_version="test-encoder", normalizer_version="test-normalizer")

    def observation(self, validity=(True, True)):
        return {"sensory": torch.ones(1, 4610), "validity": torch.tensor([validity]), "sample_ages": torch.zeros(1, 2),
                "source_ids": ("test-episode:observation:0",), "encoder_version": "test-encoder", "normalizer_version": "test-normalizer"}

    def root(self):
        return self.model.observe(self.model.initial_state(1), self.observation(), torch.zeros(1, 2), torch.zeros(1),
                                  torch.ones(1, dtype=torch.bool), noise=torch.zeros(1, 64)).state

    def test_parameter_count_and_full_sequence_shape(self):
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), 9265928)
        with torch.no_grad():
            outputs = self.model(rssm_batch(32), noise=torch.zeros(1, 33, 64))
        self.assertEqual(outputs["sensory_mean"].shape, (1, 32, 4610))
        self.assertEqual(outputs["reward_mean"].shape, (1, 32))
        self.assertTrue((outputs["posterior_scale"] >= .1).all())
        self.assertTrue((outputs["posterior_scale"] <= 10).all())

    def test_reset_missing_modalities_and_action_sensitivity(self):
        from unittest.mock import patch
        with patch.object(self.model.recurrent, "forward", wraps=self.model.recurrent.forward) as recurrent:
            root = self.root()
            recurrent.assert_not_called()
        torch.testing.assert_close(root.deterministic_state, torch.zeros(1, 512))
        missing = self.observation((False, False))
        with patch.object(self.model.posterior, "forward", wraps=self.model.posterior.forward) as posterior:
            a = self.model.observe(root, missing, torch.zeros(1, 2), torch.full((1,), .1), torch.zeros(1, dtype=torch.bool), noise=torch.zeros(1, 64))
            posterior.assert_not_called()
        self.assertFalse(a.posterior_updated.any())
        torch.testing.assert_close(a.state.latent_mean, a.prior_mean)
        b = self.model.observe(root, missing, torch.ones(1, 2), torch.full((1,), .1), torch.zeros(1, dtype=torch.bool), noise=torch.zeros(1, 64))
        self.assertFalse(torch.equal(a.state.deterministic_state, b.state.deterministic_state))
        partial = self.observation((False, True))
        first = self.model.observe(root, partial, torch.zeros(1, 2), torch.full((1,), .1), torch.zeros(1, dtype=torch.bool), noise=torch.zeros(1, 64))
        partial["sensory"][:, :4608] = 99999
        second = self.model.observe(root, partial, torch.zeros(1, 2), torch.full((1,), .1), torch.zeros(1, dtype=torch.bool), noise=torch.zeros(1, 64))
        torch.testing.assert_close(first.state.stochastic_state, second.state.stochastic_state)
        with self.assertRaises(ValueError):
            self.model.observe(root, partial, torch.full((1, 2), 3.), torch.full((1,), .1), torch.zeros(1, dtype=torch.bool))

    def test_transition_heads_precede_next_observation_and_imagination_is_isolated(self):
        batch = rssm_batch()
        noise = torch.zeros(1, 4, 64)
        with torch.no_grad():
            first = self.model(batch, noise=noise)
            changed = batch.learning_observations.clone()
            changed[:, 1:] += 10
            second = self.model(replace(batch, learning_observations=changed), noise=noise)
        torch.testing.assert_close(first["reward_mean"][:, 0], second["reward_mean"][:, 0], rtol=0, atol=0)
        self.assertFalse(torch.equal(first["sensory_mean"], second["sensory_mean"]))
        root = self.root()
        root_copy = root.deterministic_state.clone()
        weights = {k: v.clone() for k, v in self.model.state_dict().items()}
        rng = torch.random.get_rng_state().clone()
        actions, noise = torch.zeros(1, 2, 3, 2), torch.zeros(1, 2, 3, 64)
        a = self.model.imagine(root, actions, noise, batch.goals)
        b = self.model.imagine(root, actions, noise, torch.tensor([[0., 1., .1, .2]]))
        self.assertEqual(a.evidence_kind, "imagined")
        self.assertEqual(a.states.shape, (1, 2, 3, 576))
        torch.testing.assert_close(a.states, b.states, rtol=0, atol=0)
        self.assertFalse(torch.equal(a.reward_mean, b.reward_mean))
        torch.testing.assert_close(root.deterministic_state, root_copy, rtol=0, atol=0)
        torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
        for key, value in self.model.state_dict().items():
            torch.testing.assert_close(value, weights[key], rtol=0, atol=0)
        with self.assertRaises(ValueError):
            replace(a, evidence_kind="observed")

    def test_likelihood_masks_denominators_terminal_and_ordinary_kl_gradients(self):
        from predictive_control.model import model_loss
        batch = rssm_batch()
        valid = batch.input_validity.clone()
        valid[:, 1] = False
        valid[:, 2, 0] = False
        targets = dict(batch.targets)
        targets["sensory"] = torch.zeros_like(targets["sensory"])
        targets["continuation"] = torch.tensor([[1., 1., 0.]])
        targets["terminated"] = torch.tensor([[False, False, True]])
        batch = replace(batch, input_validity=valid, targets=targets)
        outputs = {k: torch.zeros(1, 3, 4610, requires_grad=True) for k in ("sensory_mean", "sensory_log_variance")}
        outputs.update({k: torch.zeros(1, 3, requires_grad=True) for k in ("reward_mean", "reward_log_variance", "cost_logits", "continuation_logits")})
        outputs.update({k: torch.full((1, 3, 64), 0.0 if k == "prior_mean" else 1.0, requires_grad=True)
                        for k in ("prior_mean", "prior_scale", "posterior_mean", "posterior_scale")})
        result = model_loss(batch, outputs)
        self.assertEqual(int(result["counts"]["vision"]), 4608)
        self.assertEqual(int(result["counts"]["proprio"]), 4)
        self.assertEqual(int(result["counts"]["reward"]), 3)
        self.assertEqual(int(result["counts"]["kl"]), 2)
        self.assertAlmostEqual(float(result["components"]["vision"].detach()), .5 * np.log(2 * np.pi), places=5)
        self.assertAlmostEqual(float(result["components"]["kl"].detach()), .5, places=6)
        result["total"].backward()
        for name in ("prior_mean", "posterior_mean"):
            self.assertGreater(float(outputs[name].grad.abs().sum()), 0)
            self.assertEqual(float(outputs[name].grad[:, 0].abs().sum()), 0)
        absent = replace(batch, input_validity=torch.zeros_like(valid))
        empty = model_loss(absent, outputs)
        for key in ("vision", "proprio", "kl"):
            self.assertEqual(float(empty["components"][key].detach()), 0)
        extreme = dict(outputs, cost_logits=torch.full((1, 3), 10000.), continuation_logits=torch.full((1, 3), -10000.))
        self.assertTrue(torch.isfinite(model_loss(batch, extreme)["total"]))
        pad = torch.tensor([[True, True, False]])
        padded = replace(batch, padding_mask=pad, observation_mask=torch.tensor([[True, True, True, False]]))
        masked = model_loss(padded, outputs)
        self.assertEqual(int(masked["counts"]["vision"]), 0)
        self.assertEqual(int(masked["counts"]["reward"]), 2)

    def test_legacy_configuration_hash_and_representation_mismatch(self):
        from predictive_control.model import RSSM
        config = load_config(ROOT / "configs/stage1.toml")
        self.assertEqual(config_hash(replace(config, model=None)), "0f336224faae3100a7eb0a3329fd52b8eae9d5713b86961b8a9e843d7fae8c45")
        other = RSSM(config, encoder_version="test-encoder", normalizer_version="different")
        with self.assertRaises(ValueError):
            other.predict_heads(self.root(), torch.zeros(1, 2), torch.tensor([[1., 0., .5, .5]]))


if __name__ == "__main__":
    unittest.main()
