"""Gaussian RSSM, explicit imagined trajectories, and masked FP32 likelihoods."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .config import RunConfig, validate_config
from .contracts import BeliefState, PredictiveDistribution, SequenceBatch, validate_array


def _tensor(value: Any, shape: tuple[int | None, ...], name: str, device: torch.device,
            dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Validate public inputs before any state update."""
    if not isinstance(value, torch.Tensor) or value.device != device or value.dtype != dtype:
        raise TypeError(f"{name} must be a {dtype} tensor on {device}")
    validate_array(value, shape, str(dtype).removeprefix("torch."), name)
    return value


@dataclass(frozen=True)
class ObservationUpdate:
    """Observed assimilation plus its differentiable prior and eligibility mask."""

    state: BeliefState
    prior_mean: torch.Tensor
    prior_scale: torch.Tensor
    posterior_updated: torch.Tensor


@dataclass(frozen=True)
class ImaginedTrajectory:
    """Predictions rooted in observed evidence; never an Observation/Transition."""

    states: torch.Tensor
    reward_mean: torch.Tensor
    reward_log_variance: torch.Tensor
    cost_logits: torch.Tensor
    continuation_logits: torch.Tensor
    source_observation_ids: tuple[str, ...]
    model_version: str
    encoder_version: str
    normalizer_version: str
    evidence_kind: str = "imagined"

    def __post_init__(self) -> None:
        if self.evidence_kind != "imagined":
            raise ValueError("imagined trajectories cannot become observed evidence")


class RSSM(nn.Module):
    """Frozen-plan 512/64 Gaussian dynamics; goals enter task outputs only."""

    def __init__(self, config: RunConfig, *, encoder_version: str, normalizer_version: str,
                 model_version: str = "uncheckpointed") -> None:
        super().__init__()
        validate_config(config)
        if config.model is None or not all(isinstance(v, str) and v for v in (encoder_version, normalizer_version, model_version)):
            raise ValueError("RSSM requires model settings and nonempty representation versions")
        self.settings = config.model
        self.encoder_version, self.normalizer_version, self.model_version = encoder_version, normalizer_version, model_version
        self.observation_embedding = nn.Sequential(nn.Linear(4614, 512), nn.SiLU(), nn.LayerNorm(512),
                                                   nn.Linear(512, 256), nn.SiLU(), nn.LayerNorm(256))
        self.transition_projection = nn.Sequential(nn.Linear(67, 256), nn.SiLU(), nn.LayerNorm(256))
        self.recurrent = nn.GRUCell(256, 512)
        self.prior = nn.Sequential(nn.Linear(512, 256), nn.SiLU(), nn.Linear(256, 128))
        self.posterior = nn.Sequential(nn.Linear(768, 256), nn.SiLU(), nn.Linear(256, 128))
        self.sensory_decoder = nn.Sequential(nn.Linear(576, 512), nn.SiLU(), nn.Linear(512, 9220))
        self.task_trunk = nn.Sequential(nn.Linear(582, 256), nn.SiLU())
        self.reward_head = nn.Linear(256, 2)
        self.cost_head = nn.Linear(256, 1)
        self.continuation_head = nn.Linear(256, 1)

    @property
    def device(self) -> torch.device:
        return self.recurrent.weight_hh.device

    def initial_state(self, batch_size: int) -> BeliefState:
        """Create a zero recurrent root; the reset observation is not a transition."""
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        h = torch.zeros(batch_size, 512, device=self.device)
        mean, scale = self._distribution(self.prior(h))
        return self._belief(h, torch.zeros_like(mean), mean, scale, ("uninitialized",) * batch_size, "prior")

    def _belief(self, h, z, mean, scale, ids, kind) -> BeliefState:
        return BeliefState(h, z, mean, scale, tuple(ids), self.model_version, self.encoder_version, kind,
                           f"gaussian-rssm-v1:{self.normalizer_version}")

    def _check_state(self, state: BeliefState) -> int:
        if (state.model_version != self.model_version or state.encoder_version != self.encoder_version
                or state.representation_version != f"gaussian-rssm-v1:{self.normalizer_version}"):
            raise ValueError("belief/model representation mismatch")
        b = len(state.source_observation_ids)
        for name, size in (("deterministic_state", 512), ("stochastic_state", 64), ("latent_mean", 64), ("latent_scale", 64)):
            _tensor(getattr(state, name), (b, size), name, self.device)
        return b

    def _distribution(self, parameters: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, raw_scale = parameters.float().chunk(2, dim=-1)
        return mean, (F.softplus(raw_scale) + self.settings["latent_scale_min"]).clamp(max=self.settings["latent_scale_max"])

    def _advance(self, h, z, action, dt):
        projected = self.transition_projection(torch.cat((z, action / 2.0, dt[:, None] / .1), dim=-1))
        return self.recurrent(projected, h)

    def observe(self, state: BeliefState, features: Mapping[str, Any], previous_action: torch.Tensor,
                dt: torch.Tensor, reset: torch.Tensor, *, noise: torch.Tensor | None = None) -> ObservationUpdate:
        """Advance then assimilate, or assimilate at h=0 when reset is true.

        features contains sensory [B,4610], validity/ages [B,2], source_ids,
        encoder_version and normalizer_version. Explicit noise permits replayable
        diagnostics; absent noise uses the caller's Torch RNG for training.
        """
        b = self._check_state(state)
        sensory = _tensor(features["sensory"], (b, 4610), "sensory", self.device)
        validity = _tensor(features["validity"], (b, 2), "validity", self.device, torch.bool)
        ages = _tensor(features["sample_ages"], (b, 2), "sample_ages", self.device)
        _tensor(previous_action, (b, 2), "previous_action", self.device)
        _tensor(dt, (b,), "dt", self.device)
        _tensor(reset, (b,), "reset", self.device, torch.bool)
        if (ages < 0).any() or (previous_action.abs() > 2).any() or (dt[~reset] <= 0).any() or (dt[reset] != 0).any():
            raise ValueError("invalid ages, force, or elapsed time (reset requires dt=0)")
        if features["encoder_version"] != self.encoder_version or features["normalizer_version"] != self.normalizer_version:
            raise ValueError("input representation mismatch")
        ids = tuple(features["source_ids"])
        if len(ids) != b or not all(isinstance(i, str) and i for i in ids):
            raise ValueError("one observed source ID per batch item is required")
        if noise is not None:
            _tensor(noise, (b, 64), "noise", self.device)
        h = torch.zeros_like(state.deterministic_state)
        if (~reset).any():
            ix = ~reset
            h = h.index_copy(0, ix.nonzero().flatten(), self._advance(state.deterministic_state[ix],
                                    state.stochastic_state[ix], previous_action[ix], dt[ix]))
        prior_mean, prior_scale = self._distribution(self.prior(h))
        mean, scale = prior_mean, prior_scale
        updated = validity.any(dim=-1)
        if updated.any():
            safe = torch.cat((torch.where(validity[:, :1], sensory[:, :4608], 0.0),
                              torch.where(validity[:, 1:], sensory[:, 4608:], 0.0)), -1)
            u = torch.cat((safe, validity.float(), ages.clamp(max=2) / 2), -1)[updated]
            embedding = self.observation_embedding(u)
            qm, qs = self._distribution(self.posterior(torch.cat((h[updated], embedding), -1)))
            indices = updated.nonzero().flatten()
            mean, scale = mean.index_copy(0, indices, qm), scale.index_copy(0, indices, qs)
        z = mean + scale * (torch.randn_like(mean) if noise is None else noise)
        belief = self._belief(h, z, mean, scale, ids, "posterior" if updated.any() else "prior")
        return ObservationUpdate(belief, prior_mean, prior_scale, updated)

    def predict_heads(self, state: BeliefState, action: torch.Tensor, goal: torch.Tensor) -> PredictiveDistribution:
        """Predict this transition's task outcomes from the current belief."""
        b = self._check_state(state)
        _tensor(action, (b, 2), "action", self.device)
        _tensor(goal, (b, 4), "goal", self.device)
        if (action.abs() > 2).any() or not torch.all((goal[:, :2] == 0) | (goal[:, :2] == 1)) or not torch.all(goal[:, :2].sum(-1) == 1) or (goal[:, 2:] < 0).any() or (goal[:, 2:] > 1).any():
            raise ValueError("expected bounded force and task one-hot plus arena goal coordinates")
        trunk = self.task_trunk(torch.cat((state.deterministic_state, state.stochastic_state, action / 2, goal), -1))
        reward = self.reward_head(trunk).float()
        return PredictiveDistribution({"reward": reward[:, 0]}, {"reward": reward[:, 1].clamp(-8, 4)},
                                      {"cost": self.cost_head(trunk).float().squeeze(-1),
                                       "continuation": self.continuation_head(trunk).float().squeeze(-1)}, self.normalizer_version)

    def forward(self, batch: SequenceBatch, *, root: BeliefState | None = None,
                noise: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Unroll a learning window; Week 2 fixtures start at the episode reset.

        Later callers may supply a reconstructed root. This function does not
        implement replay sampling, prefix warm-up, optimization, or checkpoints.
        """
        b, t = batch.executed_actions.shape[:2]
        if noise is not None:
            _tensor(noise, (b, t + 1, 64), "sequence noise", self.device)
        def observed(i: int) -> dict[str, Any]:
            return {"sensory": batch.learning_observations[:, i], "validity": batch.input_validity[:, i],
                    "sample_ages": batch.sample_ages[:, i],
                    "source_ids": tuple(f"{ep}:observation:{int(batch.observed_prefix['mask'][j].sum()) - 1 + i}" for j, ep in enumerate(batch.episode_ids)),
                    "encoder_version": self.encoder_version, "normalizer_version": self.normalizer_version}
        if root is None:
            if not batch.episode_boundaries[:, 0].all() or not (batch.observed_prefix["mask"].sum(-1) == 1).all():
                raise ValueError("a non-reset window needs an explicitly reconstructed root")
            root = self.observe(self.initial_state(b), observed(0), torch.zeros(b, 2, device=self.device),
                                torch.zeros(b, device=self.device), torch.ones(b, device=self.device, dtype=torch.bool),
                                noise=None if noise is None else noise[:, 0]).state
        self._check_state(root)
        outputs: dict[str, list[torch.Tensor]] = {key: [] for key in (
            "sensory_mean", "sensory_log_variance", "reward_mean", "reward_log_variance", "cost_logits",
            "continuation_logits", "prior_mean", "prior_scale", "posterior_mean", "posterior_scale")}
        state = root
        for i in range(t):
            task = self.predict_heads(state, batch.executed_actions[:, i], batch.goals)
            outputs["reward_mean"].append(task.gaussian_means["reward"])
            outputs["reward_log_variance"].append(task.gaussian_log_variances["reward"])
            outputs["cost_logits"].append(task.binary_logits["cost"])
            outputs["continuation_logits"].append(task.binary_logits["continuation"])
            valid_step = batch.padding_mask[:, i]
            update = self.observe(state, observed(i + 1), batch.executed_actions[:, i],
                                  torch.where(valid_step, batch.elapsed_simulation_time[:, i], .1),
                                  torch.zeros(b, dtype=torch.bool, device=self.device),
                                  noise=None if noise is None else noise[:, i + 1])
            current = update.state
            # Padded rows cannot advance the carried state into another episode.
            state = self._belief(*(torch.where(valid_step[:, None], getattr(current, name), getattr(state, name)) for name in
                                  ("deterministic_state", "stochastic_state", "latent_mean", "latent_scale")),
                                 tuple(new if bool(valid_step[j]) else old for j, (new, old) in enumerate(zip(current.source_observation_ids, state.source_observation_ids))),
                                 current.distribution_kind)
            mean, logvar = self.sensory_decoder(torch.cat((state.deterministic_state, state.stochastic_state), -1)).float().chunk(2, -1)
            outputs["sensory_mean"].append(mean)
            outputs["sensory_log_variance"].append(logvar.clamp(-8, 4))
            for key, value in (("prior_mean", update.prior_mean), ("prior_scale", update.prior_scale),
                               ("posterior_mean", current.latent_mean), ("posterior_scale", current.latent_scale)):
                outputs[key].append(value)
        return {key: torch.stack(values, dim=1) for key, values in outputs.items()}

    @torch.no_grad()
    def imagine(self, state: BeliefState, action_sequences: torch.Tensor, noise: torch.Tensor,
                goal: torch.Tensor) -> ImaginedTrajectory:
        """Roll out [B,K,H,2] actions with caller-provided [B,K,H,64] noise.

        No observation assimilation, RNG consumption, weight update, or root
        mutation. The 8,192-step limit is a hard per-call resource bound, not a
        search algorithm. Actions execute for the fixed 0.1-second decision.
        """
        b = self._check_state(state)
        _tensor(action_sequences, (b, None, None, 2), "action_sequences", self.device)
        if "uninitialized" in state.source_observation_ids:
            raise ValueError("imagination requires an assimilated observed root")
        _, k, horizon, _ = action_sequences.shape
        if k < 1 or horizon < 1 or b * k * horizon > 8192 or (action_sequences.abs() > 2).any():
            raise ValueError("invalid imagined actions or per-call rollout budget exceeded")
        _tensor(noise, (b, k, horizon, 64), "imagination noise", self.device)
        _tensor(goal, (b, 4), "goal", self.device)
        ids = tuple(i for i in state.source_observation_ids for _ in range(k))
        h, z, mean, scale = (getattr(state, name).repeat_interleave(k, 0) for name in
                             ("deterministic_state", "stochastic_state", "latent_mean", "latent_scale"))
        expanded_goal = goal.repeat_interleave(k, 0)
        trajectories, rewards, variances, costs, continuations = [], [], [], [], []
        for i in range(horizon):
            action = action_sequences[:, :, i].reshape(b * k, 2)
            belief = self._belief(h, z, mean, scale, ids, "prior")
            task = self.predict_heads(belief, action, expanded_goal)
            rewards.append(task.gaussian_means["reward"].reshape(b, k))
            variances.append(task.gaussian_log_variances["reward"].reshape(b, k))
            costs.append(task.binary_logits["cost"].reshape(b, k))
            continuations.append(task.binary_logits["continuation"].reshape(b, k))
            h = self._advance(h, z, action, torch.full((b * k,), .1, device=self.device))
            mean, scale = self._distribution(self.prior(h))
            z = mean + scale * noise[:, :, i].reshape(b * k, 64)
            trajectories.append(torch.cat((h, z), -1).reshape(b, k, 576))
        return ImaginedTrajectory(torch.stack(trajectories, 2), torch.stack(rewards, 2), torch.stack(variances, 2),
                                  torch.stack(costs, 2), torch.stack(continuations, 2), state.source_observation_ids,
                                  self.model_version, self.encoder_version, self.normalizer_version)


def model_loss(batch: SequenceBatch, outputs: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    """FP32 likelihood means plus ordinary, non-detached mean-coordinate KL.

    Exposes sums/counts so future accumulation can combine denominators correctly.
    Zero eligible coordinates contribute exactly zero, including zero gradient.
    """
    reference = outputs["sensory_mean"]
    b, t = batch.executed_actions.shape[:2]
    sizes = {"sensory_mean": (b, t, 4610), "sensory_log_variance": (b, t, 4610),
             **{k: (b, t) for k in ("reward_mean", "reward_log_variance", "cost_logits", "continuation_logits")},
             **{k: (b, t, 64) for k in ("prior_mean", "prior_scale", "posterior_mean", "posterior_scale")}}
    for key, shape in sizes.items():
        _tensor(outputs[key], shape, key, reference.device)
    def nll(target, mean, logvar):
        ell = logvar.float().clamp(-8, 4)
        return .5 * (math.log(2 * math.pi) + ell + (target.float() - mean.float()).square() * torch.exp(-ell))
    with torch.autocast(device_type=reference.device.type, enabled=False):
        transition = batch.padding_mask
        sensory_mask = batch.input_validity[:, 1:] & batch.target_validity[:, 1:] & transition[..., None]
        sensory = nll(batch.targets["sensory"][:, 1:], reference, outputs["sensory_log_variance"])
        pm, ps, qm, qs = (outputs[k].float() for k in ("prior_mean", "prior_scale", "posterior_mean", "posterior_scale"))
        if (ps <= 0).any() or (qs <= 0).any():
            raise ValueError("latent scales must be positive")
        kl = (torch.log(ps / qs) + (qs.square() + (qm - pm).square()) / (2 * ps.square()) - .5).mean(-1)
        terms = {"vision": (sensory[..., :4608], sensory_mask[..., :1]),
                 "proprio": (sensory[..., 4608:], sensory_mask[..., 1:]),
                 "reward": (nll(batch.targets["reward"], outputs["reward_mean"], outputs["reward_log_variance"]), transition),
                 "cost": (F.binary_cross_entropy_with_logits(outputs["cost_logits"], batch.targets["cost"].float(), reduction="none"), transition),
                 "continuation": (F.binary_cross_entropy_with_logits(outputs["continuation_logits"], batch.targets["continuation"].float(), reduction="none"), transition),
                 "kl": (kl, transition & batch.input_validity[:, 1:].any(-1))}
        sums, counts, components = {}, {}, {}
        for key, (value, mask) in terms.items():
            mask = mask.expand_as(value)
            sums[key] = torch.where(mask, value, 0.0).sum()
            counts[key] = mask.sum()
            components[key] = sums[key] / counts[key].clamp_min(1)
        total = sum(components[k] for k in ("vision", "proprio", "reward", "cost", "continuation")) + .1 * components["kl"]
    return {"total": total, "components": components, "sums": sums, "counts": counts}
