"""Week 1 policies; collection uses only the seeded random policy."""

from __future__ import annotations

from typing import Mapping, Any

import numpy as np

from .config import RunConfig, validate_config


class RandomPolicy:
    """Hold a uniformly sampled force for three decisions, with a zero component."""

    identity = "random-hold-v1"

    def __init__(self, config: RunConfig) -> None:
        validate_config(config)
        self.hold = config.dataset["action_hold_decisions"]
        self.zero_probability = config.dataset["zero_force_probability"]
        self.rng: np.random.Generator | None = None
        self._step = 0
        self._force = np.zeros(2, dtype=np.float32)

    def reset(self, seed: int) -> None:
        """Start an independent action stream for one episode."""
        if type(seed) is not int or seed < 0:
            raise ValueError("policy seed must be a nonnegative integer")
        self.rng = np.random.default_rng(seed)
        self._step = 0
        self._force.fill(0)

    def act(self, observation: Mapping[str, np.ndarray] | None = None) -> np.ndarray:
        """Return a detached bounded force; observations do not affect exploration."""
        if self.rng is None:
            raise RuntimeError("reset the random policy before acting")
        if self._step % self.hold == 0:
            self._force = (np.zeros(2) if self.rng.random() < self.zero_probability
                           else self.rng.uniform(-2, 2, 2)).astype(np.float32)
        self._step += 1
        return self._force.copy()


def _vector(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (2,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain two finite coordinates")
    return result


def _seek(agent: np.ndarray, velocity: np.ndarray, goal: np.ndarray,
          obj: np.ndarray | None, mass: float, drag: float) -> np.ndarray:
    """A bounded, memoryless seek/push heuristic, without obstacle routing."""
    if obj is None:
        desired = 3 * (goal - agent)
    else:
        delta = goal - obj
        distance = float(np.linalg.norm(delta))
        direction = delta / max(distance, 1e-8)
        behind = obj - 0.078 * direction
        desired = 4 * (behind - agent) + 2 * delta
        if distance < 0.025:
            desired = np.zeros(2)
    speed = float(np.linalg.norm(desired))
    desired *= min(1.0, (0.3 if obj is None else 0.075) / max(speed, 1e-8))
    return np.clip(mass * 8 * (desired - velocity) + drag * velocity,
                   -2, 2).astype(np.float32)


class ReactivePolicy:
    """Handcrafted benchmark-color perception plus currently observed velocity.

    This policy never reads diagnostic simulator state. It is a baseline, not
    the collector; missing observations produce the declared zero-force fallback.
    """

    identity = "rgb-centroid-reactive-v1"

    def act(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        """Compute an action from RGB, self velocity, and the declared task goal."""
        rgb = np.asarray(observation["rgb"])
        if rgb.shape != (96, 96, 3) or rgb.dtype != np.uint8:
            raise ValueError("reactive policy expects uint8 RGB[96,96,3]")
        velocity = _vector(observation["proprio"], "proprio")
        goal = np.asarray(observation["goal"])
        if goal.shape != (4,) or not np.isfinite(goal).all() or tuple(goal[:2]) not in ((1, 0), (0, 1)):
            raise ValueError("goal must contain task indicators and target coordinates")
        validity = np.asarray(observation["validity"])
        if validity.shape != (2,) or validity.dtype != np.bool_:
            raise ValueError("validity must be bool[2]")
        if not validity.all():
            return np.zeros(2, dtype=np.float32)

        def centroid(color: tuple[int, int, int]) -> np.ndarray | None:
            y, x = np.nonzero(np.all(rgb == color, axis=-1))
            return None if not len(x) else np.array([x.mean() / 95, 1 - y.mean() / 95])

        agent = centroid((40, 100, 230))
        obj = centroid((220, 60, 60)) if goal[1] else None
        if agent is None or (goal[1] and obj is None):
            return np.zeros(2, dtype=np.float32)
        return _seek(agent, velocity, _vector(goal[2:], "target"), obj, 1.0, 1.0)


class PrivilegedDiagnosticPolicy:
    """State-based solvability diagnostic; prohibited from random collection."""

    identity = "privileged-seek-v1"

    def act(self, state: Mapping[str, Any]) -> np.ndarray:
        """Use explicitly supplied evaluator-only simulator state."""
        agent = _vector(state["agent_position"], "agent_position")
        velocity = _vector(state["agent_velocity"], "agent_velocity")
        target = _vector(state["goal_position"], "goal_position")
        obj = None if state["object_position"] is None else _vector(state["object_position"], "object_position")
        mass, drag = float(state["agent_mass_kg"]), float(state["linear_drag_kg_per_s"])
        if not np.isfinite([mass, drag]).all() or mass <= 0 or drag < 0:
            raise ValueError("invalid diagnostic physical parameters")
        return _seek(agent, velocity, target, obj, mass, drag)
