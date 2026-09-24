"""Strict TOML loading, immutable configuration, and canonical provenance hashes."""

from __future__ import annotations

from dataclasses import dataclass, fields
from hashlib import sha256
import json
import math
from pathlib import Path
from types import MappingProxyType
import tomllib
from typing import Any, Mapping


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise TypeError(f"unsupported configuration value: {type(value).__name__}")


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Recursively immutable configuration; absent model settings preserve Week 1 hashes."""

    schema_version: int
    environment: Mapping[str, Any]
    action: Mapping[str, Any]
    observation: Mapping[str, Any]
    reset: Mapping[str, Any]
    reward: Mapping[str, Any]
    dataset: Mapping[str, Any]
    encoder: Mapping[str, Any]
    augmentation: Mapping[str, Any]
    model: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        for field in fields(self):
            object.__setattr__(self, field.name, _freeze(getattr(self, field.name)))


_KEYS = {
    "environment": "arena_size_m physics_dt substeps max_episode_steps agent_radius_m object_radius_m agent_mass_kg object_mass_kg linear_drag_kg_per_s restitution shape_friction solver_iterations wall_radius_m collision_slop_m obstacle_layouts hazard_rect",
    "action": "low high dtype units invalid_policy fallback",
    "observation": "rgb_shape rgb_dtype rgb_units proprio_shape proprio_dtype proprio_units modalities preprocessing_version time_units",
    "reset": "default_task near_probability near_distance_m max_attempts clearance_m",
    "reward": "discount success_bonus step_penalty shaping_weight push_agent_weight success_radius_m success_speed_m_per_s success_hold_decisions",
    "dataset": "train_transitions validation_transitions test_transitions task_probabilities policy action_hold_decisions zero_force_probability train_seed validation_seed test_seed pilot_seed",
    "encoder": "model weights output_layer feature_shape pooled_shape visual_dimensions frozen evaluation_mode rgb_divisor mean std center_crop cache_dtype normalization_dtype normalization_std_floor",
    "augmentation": "enabled views_per_episode clean_probability occlusion_start_probability occlusion_duration_decisions distractor_episode_probability distractor_radius_px distractor_speed_px_per_decision distractor_direction_hold_decisions distractor_rgb distractor_layer pixel_noise_std proprio_noise_std_m_per_s sample_age_cap_s seed",
}


def load_config(path: str | Path, overrides: Mapping[str, Any] | None = None) -> RunConfig:
    """Load TOML and apply existing dotted-key overrides before validation/freezing."""
    with Path(path).open("rb") as handle:
        raw = tomllib.load(handle)
    for name, value in (overrides or {}).items():
        if not isinstance(name, str):
            raise TypeError("override names must be strings")
        parts = name.split(".")
        node = raw
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                raise ValueError(f"unknown override: {name}")
            node = node[part]
        if parts[-1] not in node or isinstance(node[parts[-1]], dict):
            raise ValueError(f"override must name an existing leaf: {name}")
        node[parts[-1]] = value
    expected = {"schema_version", *_KEYS}
    if set(raw) - {"model"} != expected:
        raise ValueError(f"expected configuration sections {sorted(expected)}")
    config = RunConfig(**raw)
    validate_config(config)
    return config


def validate_config(config: RunConfig) -> None:
    """Validate supported schemas, physical bounds, and frozen Stage 1 constants."""
    if not isinstance(config, RunConfig):
        raise TypeError("config must be RunConfig")
    if type(config.schema_version) is not int or config.schema_version != 1:
        raise ValueError("unsupported configuration schema_version")
    for name, keys in _KEYS.items():
        section = getattr(config, name)
        if not isinstance(section, Mapping) or set(section) != set(keys.split()):
            raise ValueError(f"{name}: missing or unknown configuration keys")

    def number(value: Any, name: str, low: float = 0, high: float = math.inf) -> None:
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{name} must be finite in [{low}, {high}]")

    def integer(value: Any, name: str, low: int = 1, high: int = 2**63 - 1) -> None:
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in [{low}, {high}]")

    def rectangle(rect: Any, name: str) -> None:
        if not isinstance(rect, tuple) or len(rect) != 4:
            raise ValueError(f"{name} must contain x_min, y_min, x_max, y_max")
        for x in rect:
            number(x, name, 0, 1)
        if rect[0] >= rect[2] or rect[1] >= rect[3]:
            raise ValueError(f"{name} must have positive area")

    e, a, o, r, w, d, enc, aug = (
        config.environment, config.action, config.observation, config.reset,
        config.reward, config.dataset, config.encoder, config.augmentation,
    )
    for key, fixed in (("arena_size_m", 1.0), ("physics_dt", 0.01), ("substeps", 10)):
        number(e[key], key)
        if e[key] != fixed:
            raise ValueError(f"Stage 1 requires {key}={fixed}")
    integer(e["substeps"], "substeps")
    integer(e["max_episode_steps"], "max_episode_steps", 1, 200)
    integer(e["solver_iterations"], "solver_iterations")
    for key in ("agent_radius_m", "object_radius_m", "wall_radius_m"):
        number(e[key], key, 1e-6, 0.1)
    number(e["agent_mass_kg"], "agent_mass_kg", 1e-6)
    number(e["shape_friction"], "shape_friction")
    number(e["collision_slop_m"], "collision_slop_m", 0, min(e["agent_radius_m"], e["object_radius_m"]) / 2)
    for key, minimum, maximum in (("object_mass_kg", 1e-6, math.inf), ("linear_drag_kg_per_s", 0, math.inf), ("restitution", 0, 1)):
        if not isinstance(e[key], tuple) or not e[key]:
            raise ValueError(f"{key} must be a nonempty array of choices")
        for value in e[key]:
            number(value, key, minimum, maximum)
    if not isinstance(e["obstacle_layouts"], tuple) or not e["obstacle_layouts"]:
        raise ValueError("obstacle_layouts must contain at least one layout")
    for layout in e["obstacle_layouts"]:
        if not isinstance(layout, tuple):
            raise ValueError("each obstacle layout must be an array")
        for rect in layout:
            rectangle(rect, "obstacle")
    rectangle(e["hazard_rect"], "hazard_rect")
    if a != {"low": (-2.0, -2.0), "high": (2.0, 2.0), "dtype": "float32", "units": "N", "invalid_policy": "clip", "fallback": (0.0, 0.0)}:
        raise ValueError("Stage 1 requires bounded +/-2 N actions and zero-force fallback")
    expected_observation = {
        "rgb_shape": (96, 96, 3), "rgb_dtype": "uint8", "rgb_units": "uint8",
        "proprio_shape": (2,), "proprio_dtype": "float32", "proprio_units": "m/s",
        "modalities": ("rgb", "proprio"), "time_units": "s",
    }
    for key, expected in expected_observation.items():
        if o[key] != expected:
            raise ValueError(f"unsupported observation setting: {key}")
    if not isinstance(o["preprocessing_version"], str) or not o["preprocessing_version"]:
        raise ValueError("preprocessing_version must be nonempty")
    if r["default_task"] not in ("reach", "push"):
        raise ValueError("invalid default task")
    number(r["near_probability"], "near_probability", 0, 1)
    integer(r["max_attempts"], "max_attempts")
    number(r["clearance_m"], "clearance_m", 0, 0.1)
    near = r["near_distance_m"]
    if not isinstance(near, tuple) or len(near) != 2:
        raise ValueError("near_distance_m must contain two bounds")
    for bound in near:
        number(bound, "near_distance_m", 0, 1)
    if not e["agent_radius_m"] + e["object_radius_m"] + r["clearance_m"] <= near[0] < near[1]:
        raise ValueError("near reset bounds permit overlapping bodies")
    fixed_reward = {"discount": 0.99, "success_bonus": 1.0, "step_penalty": -0.01,
                    "shaping_weight": 2.0, "push_agent_weight": 0.25,
                    "success_radius_m": 0.05, "success_speed_m_per_s": 0.1,
                    "success_hold_decisions": 3}
    if w != fixed_reward:
        raise ValueError("reward settings must match frozen Stage 1 semantics")
    integer(w["success_hold_decisions"], "success_hold_decisions")
    for key in ("train_transitions", "validation_transitions", "test_transitions", "action_hold_decisions"):
        integer(d[key], key)
    if d["task_probabilities"] != (0.5, 0.5) or d["policy"] != "random":
        raise ValueError("the baseline collection policy is balanced random exploration")
    number(d["zero_force_probability"], "zero_force_probability", 0, 1)
    seed_keys = ("train_seed", "validation_seed", "test_seed", "pilot_seed")
    for key in seed_keys:
        integer(d[key], key, 0)
    if len({d[key] for key in seed_keys}) != len(seed_keys):
        raise ValueError("split RNG seeds must be distinct")
    fixed_encoder = {"model": "resnet18", "weights": "IMAGENET1K_V1", "output_layer": "layer2",
                     "feature_shape": (128, 12, 12), "pooled_shape": (128, 6, 6), "visual_dimensions": 4608,
                     "frozen": True, "evaluation_mode": True, "rgb_divisor": 255.0,
                     "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225),
                     "center_crop": False, "cache_dtype": "float16", "normalization_dtype": "float32"}
    for key, expected in fixed_encoder.items():
        if enc[key] != expected or (isinstance(expected, bool) and type(enc[key]) is not bool):
            raise ValueError(f"unsupported baseline encoder setting: {key}")
    number(enc["normalization_std_floor"], "normalization_std_floor", 1e-12)
    if aug["enabled"] is not True or aug["views_per_episode"] != 1:
        raise ValueError("Stage 1 requires one enabled corrupted view per episode")
    integer(aug["views_per_episode"], "views_per_episode")
    for key in ("clean_probability", "occlusion_start_probability", "distractor_episode_probability"):
        number(aug[key], key, 0, 1)
    duration = aug["occlusion_duration_decisions"]
    if not isinstance(duration, tuple) or len(duration) != 2:
        raise ValueError("occlusion_duration_decisions requires two integer bounds")
    for value in duration:
        integer(value, "occlusion duration")
    if duration[0] > duration[1]:
        raise ValueError("occlusion duration bounds are reversed")
    for key in ("distractor_radius_px", "distractor_direction_hold_decisions"):
        integer(aug[key], key)
    for key in ("distractor_speed_px_per_decision", "pixel_noise_std", "proprio_noise_std_m_per_s"):
        number(aug[key], key)
    number(aug["sample_age_cap_s"], "sample_age_cap_s", 1e-6)
    integer(aug["seed"], "augmentation seed", 0)
    if len(aug["distractor_rgb"]) != 3:
        raise ValueError("distractor_rgb needs three channels")
    for channel in aug["distractor_rgb"]:
        integer(channel, "distractor color", 0, 255)
    if aug["distractor_layer"] != "behind_task_geometry":
        raise ValueError("distractors must not obscure task geometry")
    if config.model is not None:
        fixed = {
            "deterministic_size": 512, "stochastic_size": 64,
            "observation_hidden": 512, "embedding_size": 256,
            "transition_hidden": 256, "decoder_hidden": 512,
            "task_hidden": 256, "log_variance_min": -8.0,
            "log_variance_max": 4.0, "latent_scale_min": 0.1,
            "latent_scale_max": 10.0, "kl_weight": 0.1,
            "action_scale_n": 2.0, "time_scale_s": 0.1, "age_cap_s": 2.0,
            "sequence_transitions": 32, "learning_rate": 0.0003,
            "weight_decay": 0.0001, "gradient_norm_cap": 10.0,
        }
        if not isinstance(config.model, Mapping) or dict(config.model) != fixed:
            raise ValueError("model settings must match the frozen Gaussian RSSM baseline")
        for key, value in fixed.items():
            if isinstance(value, int):
                integer(config.model[key], key)
            else:
                number(config.model[key], key, -8.0)


def config_hash(config: RunConfig) -> str:
    """Return SHA-256 of the complete canonical resolved configuration."""
    validate_config(config)
    values = {field.name: _plain(getattr(config, field.name)) for field in fields(config)
              if getattr(config, field.name) is not None}
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return sha256(encoded).hexdigest()
