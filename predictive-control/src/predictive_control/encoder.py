"""Frozen spatial features, reproducible corruptions, and observed-only caches."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18

from .config import RunConfig, validate_config
from .contracts import DatasetSnapshot, validate_array
from .data import EpisodeStore, canonical, file_hash, plain, write_once


def digest(value: Any) -> str:
    """Hash a JSON-compatible provenance record."""
    return sha256(canonical(value)).hexdigest()


class FrozenVisualEncoder(nn.Module):
    """ImageNet ResNet18 through layer2; no resizing, cropping, or BN updates."""

    def __init__(self, config: RunConfig, device: str | torch.device = "cpu") -> None:
        super().__init__()
        validate_config(config)
        network = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(network.children())[:6])
        self.register_buffer("mean", torch.tensor(config.encoder["mean"]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(config.encoder["std"]).view(1, 3, 1, 1))
        self.requires_grad_(False)
        self.train(False)
        weight_hash = sha256()
        for name, tensor in self.backbone.state_dict().items():
            weight_hash.update(name.encode())
            weight_hash.update(tensor.contiguous().numpy().tobytes())
        self.provenance = {"schema": "layer2-spatial-v1", "weights_sha256": weight_hash.hexdigest(),
                           "preprocessing": plain(config.encoder), "source_sha256": file_hash(Path(__file__))}
        self.version = digest(self.provenance)
        self.to(device=device)

    def train(self, mode: bool = True) -> FrozenVisualEncoder:
        """Keep the entire encoder in evaluation mode even when its parent trains."""
        super().train(False)
        return self

    @torch.no_grad()
    def encode(self, rgb_batch: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Return unnormalized FP32 [N,4608] features for uint8 [N,96,96,3]."""
        validate_array(rgb_batch, (None, 96, 96, 3), "uint8", "rgb_batch")
        if len(rgb_batch) < 1:
            raise ValueError("encoder batch must be nonempty")
        self.train(False)
        x = torch.as_tensor(np.array(rgb_batch, copy=True) if isinstance(rgb_batch, np.ndarray)
                            else rgb_batch, device=self.mean.device)
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = (x.permute(0, 3, 1, 2).float() / 255.0 - self.mean) / self.std
            x = self.backbone(x)
            if x.shape[1:] != (128, 12, 12):
                raise ValueError("unexpected layer2 geometry")
            return F.adaptive_avg_pool2d(x, (6, 6)).flatten(1).float()

    forward = encode


def make_corrupted_view(episode: Mapping[str, Any], augmentation_config: Mapping[str, Any],
                        seed: int) -> dict[str, Any]:
    """Create one deterministic whole-episode input view; never alter clean targets.

    Separate RNG streams make the disk independent of blackouts, sensor noise,
    actions, rewards, and goals. Blackouts carry invalid zero placeholders, not
    black-image observations. The renderer's flat background identifies pixels
    behind task geometry; changing the renderer requires a new cache version.
    """
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("view seed must be a uint64 integer")
    a, cfg = episode["arrays"], augmentation_config
    n = len(a["rgb"])
    validate_array(a["rgb"], (n, 96, 96, 3), "uint8", "episode RGB")
    validate_array(a["proprio"], (n, 2), "float32", "episode proprioception")
    validate_array(a["validity"], (n, 2), "bool", "episode validity")
    validate_array(a["sample_age"], (n, 2), "float32", "episode sample ages")
    validate_array(a["dt"], (n - 1,), ("float32", "float64"), "episode elapsed time")
    if n < 2 or len(episode["metadata"]["observation_ids"]) != n or (a["sample_age"] < 0).any() or (a["dt"] <= 0).any():
        raise ValueError("corruptions require aligned observations, ages, and positive durations")
    for key in ("occlusion_start_probability", "distractor_episode_probability"):
        if type(cfg[key]) not in (int, float) or not 0 <= cfg[key] <= 1:
            raise ValueError(f"invalid augmentation probability: {key}")
    for key in ("pixel_noise_std", "proprio_noise_std_m_per_s", "distractor_speed_px_per_decision"):
        if type(cfg[key]) not in (int, float) or not np.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError(f"invalid augmentation magnitude: {key}")
    low, high = cfg["occlusion_duration_decisions"]
    if type(low) is not int or type(high) is not int or not 1 <= low <= high:
        raise ValueError("blackout duration must have positive integer bounds")
    if type(cfg["distractor_direction_hold_decisions"]) is not int or cfg["distractor_direction_hold_decisions"] < 1:
        raise ValueError("disk direction hold must be a positive integer")
    if type(cfg["distractor_radius_px"]) is not int or not 1 <= cfg["distractor_radius_px"] < 48 or cfg["distractor_layer"] != "behind_task_geometry":
        raise ValueError("disk must fit the arena and remain behind task geometry")
    rng_blackout, rng_disk, rng_pixel, rng_proprio = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)]
    rgb, proprio = a["rgb"].copy(), a["proprio"].copy()
    validity, ages = a["validity"].copy(), a["sample_age"].copy()
    active_disk = rng_disk.random() < cfg["distractor_episode_probability"]
    radius = cfg["distractor_radius_px"]
    position = rng_disk.uniform(radius, 95 - radius, 2)
    velocity = np.zeros(2)
    remaining = 0
    blackout_age = 0.0
    disk_positions = []
    for i in range(n):
        if i % cfg["distractor_direction_hold_decisions"] == 0:
            angle = rng_disk.uniform(0, 2 * np.pi)
            velocity = cfg["distractor_speed_px_per_decision"] * np.array([np.cos(angle), np.sin(angle)])
        position = np.clip(position + velocity, radius, 95 - radius)
        disk_positions.append(position.copy())
        if active_disk:
            layer = Image.fromarray(rgb[i].copy())
            ImageDraw.Draw(layer).ellipse((* (position - radius), * (position + radius)), fill=tuple(cfg["distractor_rgb"]))
            background = np.all(rgb[i] == (245, 245, 245), axis=-1)
            rgb[i][background] = np.asarray(layer)[background]
        if remaining == 0 and rng_blackout.random() < cfg["occlusion_start_probability"]:
            low, high = cfg["occlusion_duration_decisions"]
            remaining = int(rng_blackout.integers(low, high + 1))
        if remaining:
            validity[i, 0] = False
            blackout_age += float(a["dt"][i - 1]) if i else 0.0
            ages[i, 0] = max(blackout_age, float(ages[i, 0]))
            remaining -= 1
        else:
            blackout_age = float(ages[i, 0])
        noisy = rgb[i].astype(np.float32) / 255.0 + rng_pixel.normal(0, cfg["pixel_noise_std"], rgb[i].shape)
        rgb[i] = np.rint(np.clip(noisy, 0, 1) * 255).astype(np.uint8) if validity[i, 0] else 0
        if validity[i, 1]:
            proprio[i] += rng_proprio.normal(0, cfg["proprio_noise_std_m_per_s"], 2).astype(np.float32)
        else:
            proprio[i] = 0
    return {"rgb": rgb, "proprio": proprio, "validity": validity, "sample_age": ages,
            "source_observation_ids": tuple(episode["metadata"]["observation_ids"]), "seed": seed,
            "disk_active": active_disk, "disk_positions": np.asarray(disk_positions)}


def _snapshot_matches(snapshot: DatasetSnapshot, store: EpisodeStore) -> None:
    if store.verify_snapshot(snapshot.split) != snapshot:
        raise ValueError("snapshot does not match the sealed observed store")


def _write_arrays(path: Path, arrays: Mapping[str, np.ndarray], store: EpisodeStore) -> None:
    """Publish an immutable, non-pickle numeric file within the campaign budget."""
    estimate = sum(a.nbytes for a in arrays.values()) + 65536
    if store.disk_bytes() + estimate + 1024**3 > store.max_bytes or shutil.disk_usage(store.root).free < estimate + 1024**3:
        raise OSError("feature cache would exceed storage budget/headroom")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            if file_hash(Path(temporary)) != file_hash(path):
                raise ValueError("immutable cache content already differs")
        else:
            os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def cache_features(snapshot: DatasetSnapshot, encoder: FrozenVisualEncoder,
                   augmentation_config: Mapping[str, Any], *, store: EpisodeStore,
                   batch_size: int = 64) -> Path:
    """Cache clean features and one corrupted training view, with bounded batches.

    This writes unnormalized coordinates. Fit/load a separate frozen normalizer
    afterward; learners never instantiate the visual backbone or read raw RGB.
    """
    if type(batch_size) is not int or not 1 <= batch_size <= 256:
        raise ValueError("encoding batch_size must be in [1,256]")
    _snapshot_matches(snapshot, store)
    # Stable split/index IDs keep perturbations reproducible across recollection;
    # random episode UUIDs and source hashes must not choose the view's RNG stream.
    seeds = {identity: int(digest({"root": augmentation_config["seed"],
                                   "episode": identity})[:16], 16) for identity in snapshot.episode_ids}
    manifest = {"schema": "spatial-feature-cache-v1", "dataset_hash": snapshot.dataset_hash,
                "split": snapshot.split, "episode_ids": list(snapshot.episode_ids),
                "encoder": encoder.provenance, "encoder_version": encoder.version,
                "augmentation": plain(augmentation_config), "view_seeds": seeds,
                "views": ["clean", "corrupted"] if snapshot.split == "train" else ["clean"],
                "visual_dtype": "float16", "normalization": "separate-frozen-training-record",
                "fp16_tolerance": {"rtol": 0.001, "atol": 0.001}}
    root = store.root / "features" / digest(manifest)
    write_once(root / "manifest.json", manifest)
    for identity in snapshot.episode_ids:
        episode = store.load_episode(identity)
        if (root / f"{identity}.json").exists():
            load_cached_episode(root, identity, store=store)
            continue
        views = {"clean": episode["arrays"]}
        if snapshot.split == "train":
            views["corrupted"] = make_corrupted_view(episode, augmentation_config, seeds[identity])
        arrays = {}
        for name, view in views.items():
            visual = np.zeros((len(view["rgb"]), 4608), dtype=np.float16)
            indices = np.flatnonzero(view["validity"][:, 0])
            for start in range(0, len(indices), batch_size):
                ix = indices[start:start + batch_size]
                fresh = encoder.encode(view["rgb"][ix]).cpu().numpy()
                stored = fresh.astype(np.float16)
                if not np.isfinite(stored).all() or not np.allclose(stored.astype(np.float32), fresh, rtol=.001, atol=.001):
                    raise ValueError("FP16 feature-cache tolerance failed")
                visual[ix] = stored
            arrays.update({f"{name}_visual": visual, f"{name}_proprio": view["proprio"],
                           f"{name}_validity": view["validity"], f"{name}_sample_age": view["sample_age"]})
        path = root / f"{identity}.npz"
        _write_arrays(path, arrays, store)
        record = {"file_sha256": file_hash(path), "source_sha256": file_hash(store.episodes / identity / "record.json"),
                  "source_observation_ids": episode["metadata"]["observation_ids"],
                  "cache_hash": root.name, "seed": seeds[identity], "targets": "clean observed episode"}
        write_once(root / f"{identity}.json", record)
    return root


def load_cached_episode(cache: str | Path, identity: str, *, store: EpisodeStore) -> dict[str, Any]:
    """Verify provenance and return clean/corrupted arrays without privileged state."""
    root = Path(cache)
    manifest = json.loads((root / "manifest.json").read_text())
    if root.name != digest(manifest) or identity not in manifest["episode_ids"]:
        raise ValueError("invalid cache manifest or episode membership")
    record = json.loads((root / f"{identity}.json").read_text())
    source = store.load_episode(identity)
    if (record["cache_hash"] != root.name or record["seed"] != manifest["view_seeds"][identity]
            or record["source_sha256"] != file_hash(store.episodes / identity / "record.json")
            or record["source_observation_ids"] != source["metadata"]["observation_ids"]
            or record["file_sha256"] != file_hash(root / f"{identity}.npz")):
        raise ValueError("cache or observed-source provenance changed")
    with np.load(root / f"{identity}.npz", allow_pickle=False) as handle:
        arrays = {key: handle[key] for key in handle.files}
    n = len(record["source_observation_ids"])
    expected = set()
    for view in manifest["views"]:
        for suffix, shape, dtype in (("visual", (n, 4608), "float16"), ("proprio", (n, 2), "float32"),
                                     ("validity", (n, 2), "bool"), ("sample_age", (n, 2), "float32")):
            key = f"{view}_{suffix}"
            expected.add(key)
            validate_array(arrays[key], shape, dtype, key)
            arrays[key].flags.writeable = False
        if (arrays[f"{view}_sample_age"] < 0).any():
            raise ValueError("negative cached sample age")
    if set(arrays) != expected:
        raise ValueError("unexpected cached arrays")
    return {"arrays": arrays, "source": source, "manifest": manifest, "record": record}


@dataclass(frozen=True)
class FeatureNormalizer:
    """Frozen coordinate statistics with a content-addressed training reference."""

    mean: np.ndarray
    std: np.ndarray
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in ("mean", "std"):
            value = np.array(getattr(self, name), dtype=np.float32, copy=True)
            validate_array(value, (4610,), "float32", name)
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        if (self.std < 1e-4).any() or self.provenance["split"] != "train":
            raise ValueError("normalizer requires training-only statistics and std >= 1e-4")
        from .config import _freeze
        object.__setattr__(self, "provenance", _freeze(dict(self.provenance)))

    def record(self) -> dict[str, Any]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "provenance": plain(self.provenance)}

    @property
    def version(self) -> str:
        return digest(self.record())

    def transform(self, sensory: np.ndarray, validity: np.ndarray) -> np.ndarray:
        """Normalize in FP32; invalid coordinates use zero plus explicit metadata."""
        validate_array(sensory, (None, 4610), ("float16", "float32"), "sensory")
        validate_array(validity, (len(sensory), 2), "bool", "validity")
        result = (sensory.astype(np.float32) - self.mean) / self.std
        result[~validity[:, 0], :4608] = 0
        result[~validity[:, 1], 4608:] = 0
        if not np.isfinite(result).all():
            raise ValueError("nonfinite normalized coordinates")
        return result

    def save(self, path: str | Path) -> None:
        write_once(Path(path), {"normalizer_hash": self.version, **self.record()})

    @classmethod
    def load(cls, path: str | Path) -> FeatureNormalizer:
        record = json.loads(Path(path).read_text())
        result = cls(np.asarray(record["mean"]), np.asarray(record["std"]), record["provenance"])
        if result.version != record["normalizer_hash"]:
            raise ValueError("normalizer checksum mismatch")
        return result


def fit_normalizer(training_snapshot: DatasetSnapshot, *, cache: str | Path,
                   store: EpisodeStore, std_floor: float = 1e-4) -> FeatureNormalizer:
    """Fit population moments on every clean, valid TRAIN observation exactly once."""
    if training_snapshot.split != "train" or std_floor != 1e-4:
        raise ValueError("baseline normalization requires training data and std floor 1e-4")
    _snapshot_matches(training_snapshot, store)
    counts = np.zeros(4610, dtype=np.int64)
    mean, m2 = np.zeros(4610, dtype=np.float64), np.zeros(4610, dtype=np.float64)
    for identity in training_snapshot.episode_ids:
        record = load_cached_episode(cache, identity, store=store)
        manifest, arrays = record["manifest"], record["arrays"]
        if manifest["dataset_hash"] != training_snapshot.dataset_hash or manifest["split"] != "train":
            raise ValueError("normalizer/cache training snapshot mismatch")
        for modality, key, part in ((0, "visual", slice(0, 4608)), (1, "proprio", slice(4608, 4610))):
            values = arrays[f"clean_{key}"][arrays["clean_validity"][:, modality]].astype(np.float64)
            n = len(values)
            if n:
                delta = values.mean(0) - mean[part]
                total = counts[part] + n
                m2[part] += ((values - values.mean(0)) ** 2).sum(0) + delta**2 * counts[part] * n / total
                mean[part] += delta * n / total
                counts[part] = total
    if not counts.all():
        raise ValueError("training snapshot lacks a modality required for normalization")
    return FeatureNormalizer(mean.astype(np.float32), np.maximum(np.sqrt(m2 / counts), std_floor).astype(np.float32),
                             {"split": "train", "dataset_hash": training_snapshot.dataset_hash,
                              "cache_hash": Path(cache).name, "encoder_version": manifest["encoder_version"],
                              "counts": counts.tolist(), "method": "population-moments-clean-fp16-cache-v1",
                              "std_floor": std_floor})


@torch.no_grad()
def encoder_diagnostics(encoder: FrozenVisualEncoder) -> dict[str, Any]:
    """Controlled pixel/feature distinguishability checks, not tracking or control."""
    def scene(first=(28, 48), second=(68, 48), target=(75, 20)):
        image = Image.new("RGB", (96, 96), (245, 245, 245))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 95, 95), outline=(100, 100, 100))
        for (x, y), color in ((target, (40, 190, 70)), (first, (220, 60, 60)), (second, (220, 60, 60))):
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
        return np.asarray(image).copy()
    pairs = {"move_first_identical_object": (scene(), scene(first=(33, 48))),
             "move_second_identical_object": (scene(), scene(second=(73, 48))),
             "contact_vs_gap": (scene(second=(36, 48)), scene(second=(42, 48))),
             "target_location": (scene(), scene(target=(75, 28)))}
    results = {}
    for name, (a, b) in pairs.items():
        features = encoder.encode(np.stack([a, b])).cpu().numpy()
        quantized = features.astype(np.float16).astype(np.float32)
        rounding = float(np.linalg.norm(features - quantized, axis=1).max())
        difference = float(np.linalg.norm(features[0] - features[1]))
        results[name] = {"feature_l2": difference, "rounding_l2": rounding,
                         "pass": bool(difference > max(1e-6, 10 * rounding))}
    return {"status": "pass" if all(r["pass"] for r in results.values()) else "fail",
            "encoder_version": encoder.version, "checks": results,
            "criterion": "feature separation > 10x FP16 rounding error and > 1e-6",
            "scope": "necessary distinguishability only; no claim of decodability, tracking, or generalization"}
