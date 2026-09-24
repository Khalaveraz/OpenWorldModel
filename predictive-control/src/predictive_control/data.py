"""Atomic observed episodes, integrity checks, coverage reports, and snapshots."""

from __future__ import annotations

from dataclasses import fields
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from .config import RunConfig, config_hash
from .contracts import DatasetSnapshot, Transition, validate_array

SPLITS = ("pilot", "train", "validation", "test")
STORE_VERSION = "observed-episodes-v1"
SEED_BLOCK = 1 << 40


def plain(value: Any) -> Any:
    """Convert immutable configuration/record containers to JSON-compatible values."""
    if isinstance(value, Mapping):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def canonical(value: Any) -> bytes:
    """Serialize finite JSON deterministically for hashes and immutable records."""
    return json.dumps(plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def file_hash(path: Path) -> str:
    """Stream a SHA-256 without loading a whole dataset into RAM."""
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_once(path: Path, value: Any) -> None:
    """Atomically publish JSON without overwriting an existing different record."""
    payload = canonical(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"immutable record differs: {path}")
        return
    descriptor, name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # Hard-link publication is atomic and refuses an existing destination.
        os.link(temporary, path)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise
    finally:
        temporary.unlink(missing_ok=True)


def episode_seed(root: int, index: int, stream: int = 0) -> int:
    """Give each configured root a disjoint range; stream 0 is physics, 1 actions."""
    if type(root) is not int or root < 0 or type(index) is not int or not 0 <= index < SEED_BLOCK // 2:
        raise ValueError("invalid seed root or collection index")
    if stream not in (0, 1):
        raise ValueError("stream must be 0 or 1")
    return root * SEED_BLOCK + 2 * index + stream


def pack_episode(transitions: Sequence[Transition], diagnostics: Sequence[Mapping[str, Any]],
                 config: RunConfig, provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Pack a completed factual trajectory; privileged arrays occupy a separate file."""
    transitions = tuple(transitions)
    if not transitions or len(diagnostics) != len(transitions) + 1:
        raise ValueError("need T transitions and T+1 diagnostic states")
    first = transitions[0]
    observations = [first.observation, *(t.next_observation for t in transitions)]
    for index, transition in enumerate(transitions):
        if not isinstance(transition, Transition) or not transition.training_eligible:
            raise ValueError("only validated, known-action observed transitions may be stored")
        if transition.observation.observation_id != observations[index].observation_id:
            raise ValueError("transition observations are not contiguous")
        if transition.observation is not observations[index]:
            for name in ("rgb", "proprio"):
                if not np.array_equal(transition.observation.modalities[name].payload,
                                      observations[index].modalities[name].payload):
                    raise ValueError("reused observation ID has inconsistent contents")
        if transition.goal.task != first.goal.task or not np.array_equal(transition.goal.as_array(), first.goal.as_array()):
            raise ValueError("episode goal changed")
        state = diagnostics[index]
        if state["episode_id"] != observations[index].episode_id or state["step_id"] != index:
            raise ValueError("diagnostics do not align with observations")
    final = diagnostics[-1]
    if final["episode_id"] != observations[-1].episode_id or final["step_id"] != len(transitions):
        raise ValueError("final diagnostic state is misaligned")
    schema = {
        name: {key: getattr(observations[0].modalities[name], key)
               for key in ("shape", "dtype", "units", "sensor_id", "preprocessing_version")}
        for name in ("rgb", "proprio")
    }
    for observation in observations:
        if set(observation.modalities) != {"rgb", "proprio"}:
            raise ValueError("unsupported sensor modality")
        for name, metadata in schema.items():
            if any(getattr(observation.modalities[name], key) != value for key, value in metadata.items()):
                raise ValueError("sensor schema changed within episode")
    if any(o.episode_id != first.observation.episode_id or o.step_id != i for i, o in enumerate(observations)):
        raise ValueError("observations cross an episode boundary")
    actions = [t.action_record for t in transitions]
    arrays = {
        "rgb": np.stack([o.modalities["rgb"].payload for o in observations]),
        "proprio": np.stack([o.modalities["proprio"].payload for o in observations]),
        "validity": np.array([[o.modalities[m].validity for m in ("rgb", "proprio")] for o in observations], dtype=np.bool_),
        "sample_age": np.array([[o.modalities[m].sample_age for m in ("rgb", "proprio")] for o in observations], dtype=np.float32),
        "capture_time": np.array([o.simulation_capture_time for o in observations], dtype=np.float64),
        "receipt_time": np.array([o.receipt_time or 0.0 for o in observations], dtype=np.float64),
        "receipt_valid": np.array([o.receipt_time is not None for o in observations], dtype=np.bool_),
        "reset": np.array([o.reset for o in observations], dtype=np.bool_),
        "requested_action": np.stack([a.requested_force for a in actions]),
        "executed_action": np.stack([a.executed_force for a in actions]),
        "requested_duration": np.array([a.requested_duration for a in actions], dtype=np.float64),
        "executed_duration": np.array([a.executed_duration for a in actions], dtype=np.float64),
        "execution_time": np.array([a.execution_timestamp for a in actions], dtype=np.float64),
        "dt": np.array([t.elapsed_simulation_time for t in transitions], dtype=np.float64),
        "reward": np.array([t.reward for t in transitions], dtype=np.float64),
        "cost": np.array([t.cost for t in transitions], dtype=np.float32),
        "terminated": np.array([t.terminated for t in transitions], dtype=np.bool_),
        "truncated": np.array([t.truncated for t in transitions], dtype=np.bool_),
        "continuation": np.array([t.continuation for t in transitions], dtype=np.float32),
        "goal": first.goal.as_array(),
    }
    diagnostic_arrays = {
        "agent_position": np.array([d["agent_position"] for d in diagnostics], dtype=np.float64),
        "object_position": np.array([d["object_position"] if d["object_position"] is not None else [0, 0]
                                     for d in diagnostics], dtype=np.float64),
        "contact": np.array([d["agent_object_contact"] for d in diagnostics[1:]], dtype=np.bool_),
        "contact_impulse": np.array([d["contact_impulse"] for d in diagnostics[1:]], dtype=np.float64),
    }
    metadata = {
        **plain(provenance), "store_version": STORE_VERSION, "evidence_kind": "observed",
        "episode_id": first.observation.episode_id, "task": first.goal.task,
        "transitions": len(transitions), "configuration_hash": config_hash(config),
        "configuration": {f.name: plain(getattr(config, f.name)) for f in fields(config)},
        "observation_ids": [o.observation_id for o in observations],
        "action_ids": [a.action_id for a in actions], "action_status": [a.status for a in actions],
        "action_reason": [a.reason for a in actions], "modality_schema": plain(schema),
        "units": {"force": "N", "time": "s", "position": "m", "velocity": "m/s"},
        "reset_stratum": diagnostics[0]["reset_stratum"],
        "layout_index": diagnostics[0]["layout_index"],
        "physical_parameters": {k: diagnostics[0][k] for k in (
            "agent_mass_kg", "object_mass_kg", "linear_drag_kg_per_s", "restitution")},
    }
    validate_episode(metadata, arrays, diagnostic_arrays)
    return {"metadata": metadata, "arrays": arrays, "diagnostics": diagnostic_arrays}


def validate_episode(metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray],
                     diagnostics: Mapping[str, np.ndarray] | None = None) -> None:
    """Check the persisted schema, terminal alignment, actions, seeds, and units."""
    n = metadata["transitions"]
    if type(n) is not int or not 1 <= n <= 200:
        raise ValueError("episode length must be 1..200")
    if metadata["store_version"] != STORE_VERSION or metadata["evidence_kind"] != "observed":
        raise ValueError("only observed episodes of the supported version are admissible")
    split, index = metadata["split"], metadata["collection_index"]
    if split not in SPLITS or metadata["policy_identity"] != "random-hold-v1":
        raise ValueError("unsupported split or collection policy")
    root = metadata["configuration"]["dataset"][f"{split}_seed"]
    if metadata["environment_seed"] != episode_seed(root, index) or metadata["policy_seed"] != episode_seed(root, index, 1):
        raise ValueError("seed namespace mismatch")
    if config_hash(RunConfig(**metadata["configuration"])) != metadata["configuration_hash"]:
        raise ValueError("configuration hash mismatch")
    if metadata["units"] != {"force": "N", "time": "s", "position": "m", "velocity": "m/s"}:
        raise ValueError("incorrect physical units")
    schema = {"rgb": ((n + 1, 96, 96, 3), "uint8"), "proprio": ((n + 1, 2), "float32"),
              "validity": ((n + 1, 2), "bool"), "sample_age": ((n + 1, 2), "float32"),
              "capture_time": ((n + 1,), "float64"), "receipt_time": ((n + 1,), "float64"),
              "receipt_valid": ((n + 1,), "bool"), "reset": ((n + 1,), "bool"),
              "requested_action": ((n, 2), "float32"), "executed_action": ((n, 2), "float32"),
              "goal": ((4,), "float32")}
    for key in ("requested_duration", "executed_duration", "execution_time", "dt", "reward"):
        schema[key] = ((n,), "float64")
    for key in ("cost", "continuation"):
        schema[key] = ((n,), "float32")
    for key in ("terminated", "truncated"):
        schema[key] = ((n,), "bool")
    if set(arrays) != set(schema):
        raise ValueError("unexpected or missing observed arrays")
    for name, (shape, dtype) in schema.items():
        validate_array(arrays[name], shape, dtype, name)
    for name, count in (("observation_ids", n + 1), ("action_ids", n)):
        ids = metadata[name]
        if len(ids) != count or len(set(ids)) != count or any(not isinstance(v, str) or not v for v in ids):
            raise ValueError(f"invalid {name}")
    if not arrays["reset"][0] or arrays["reset"][1:].any() or arrays["capture_time"][0] != 0:
        raise ValueError("reset crossed into the episode")
    if (arrays["sample_age"] < 0).any() or (arrays["receipt_time"] < 0).any():
        raise ValueError("negative sensor age or receipt time")
    ends = arrays["terminated"] | arrays["truncated"]
    if ends[:-1].any() or not ends[-1] or (arrays["terminated"] & arrays["truncated"]).any():
        raise ValueError("only completed, uninterrupted episodes may be committed")
    cap = metadata["configuration"]["environment"]["max_episode_steps"]
    if n > cap or (arrays["truncated"][-1] and n != cap):
        raise ValueError("time-limit truncation does not match the environment cap")
    if not np.array_equal(arrays["continuation"], (~arrays["terminated"]).astype(np.float32)):
        raise ValueError("truncation must not suppress continuation")
    if not np.isin(arrays["cost"], [0, 1]).all():
        raise ValueError("cost must be binary")
    expected_dt = (metadata["configuration"]["environment"]["physics_dt"] *
                   metadata["configuration"]["environment"]["substeps"])
    for value in (np.diff(arrays["capture_time"]), arrays["requested_duration"], arrays["executed_duration"], arrays["dt"]):
        if not np.allclose(value, expected_dt, rtol=0, atol=1e-8):
            raise ValueError("action/observation timing mismatch")
    if not np.allclose(arrays["execution_time"], arrays["capture_time"][:-1], rtol=0, atol=1e-8):
        raise ValueError("action timestamps are misaligned")
    if not np.array_equal(np.clip(arrays["requested_action"], -2, 2), arrays["executed_action"]):
        raise ValueError("executed force does not match clipping")
    settings = metadata["configuration"]["dataset"]
    rng = np.random.default_rng(metadata["policy_seed"])
    for i in range(n):
        if i % settings["action_hold_decisions"] == 0:
            force = (np.zeros(2) if rng.random() < settings["zero_force_probability"]
                     else rng.uniform(-2, 2, 2)).astype(np.float32)
        if not np.array_equal(arrays["requested_action"][i], force):
            raise ValueError("recorded actions disagree with the declared random policy/seed")
    if len(metadata["action_status"]) != n or len(metadata["action_reason"]) != n:
        raise ValueError("action metadata lengths differ")
    for i, status in enumerate(metadata["action_status"]):
        expected = "applied" if np.array_equal(arrays["requested_action"][i], arrays["executed_action"][i]) else "clipped"
        if status != expected or (status == "clipped" and not metadata["action_reason"][i]):
            raise ValueError("unknown or inconsistent action execution")
    task = metadata["task"]
    if task not in ("reach", "push") or not np.array_equal(arrays["goal"][:2], [task == "reach", task == "push"]):
        raise ValueError("task goal mismatch")
    if ((arrays["goal"][2:] < 0) | (arrays["goal"][2:] > 1)).any():
        raise ValueError("goal is outside the arena")
    if metadata["reset_stratum"] not in (("broad",) if task == "reach" else ("near", "broad")):
        raise ValueError("invalid reset stratum")
    if diagnostics is not None:
        expected = {"agent_position", "object_position", "contact", "contact_impulse"}
        if set(diagnostics) != expected:
            raise ValueError("invalid diagnostic schema")
        for name in ("agent_position", "object_position"):
            validate_array(diagnostics[name], (n + 1, 2), "float64", name)
        validate_array(diagnostics["contact"], (n,), "bool", "contact")
        validate_array(diagnostics["contact_impulse"], (n,), "float64", "contact_impulse")
        if (diagnostics["contact_impulse"] < 0).any():
            raise ValueError("negative contact impulse")
        if task == "reach" and (diagnostics["contact"].any() or diagnostics["object_position"].any()):
            raise ValueError("reach diagnostics assert an absent object")
        if metadata["reset_stratum"] == "near":
            distance = np.linalg.norm(diagnostics["agent_position"][0] - diagnostics["object_position"][0])
            if not 0.1 - 1e-8 <= distance <= 0.2 + 1e-8:
                raise ValueError("near reset violates its distance stratum")


class EpisodeStore:
    """Single-writer, bounded store with atomic episode and manifest publication.

    Only load_episode(..., include_diagnostics=True) exposes privileged arrays.
    Pending directories from interrupted writes are ignored, never treated as data.
    """

    def __init__(self, root: str | Path, max_bytes: int = 40 * 1024**3) -> None:
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        self.root = Path(root).resolve()
        self.episodes = self.root / "episodes"
        self.manifests = self.root / "manifests"
        self.max_bytes = max_bytes
        for path in (self.episodes, self.manifests):
            path.mkdir(parents=True, exist_ok=True)

    def ids(self, split: str) -> list[str]:
        """List committed episodes in deterministic collection order."""
        if split not in SPLITS:
            raise ValueError("unknown dataset split")
        return sorted(p.name for p in self.episodes.glob(f"{split}-*") if p.is_dir() and not p.is_symlink())

    def _path(self, episode_id: str) -> Path:
        if not isinstance(episode_id, str) or not re.fullmatch(r"(?:pilot|train|validation|test)-[0-9]{8}", episode_id):
            raise ValueError("invalid storage episode ID")
        path = self.episodes / episode_id
        if path.is_symlink():
            raise ValueError("episode paths cannot be symlinks")
        return path

    def disk_bytes(self) -> int:
        """Measure logical campaign bytes; reserve additional headroom before writes."""
        return sum(p.stat().st_size for p in self.root.parent.rglob("*") if p.is_file() and not p.is_symlink())

    def append_completed_episode(self, episode: Mapping[str, Any]) -> str:
        """Commit one complete observed episode; existing episode IDs never change."""
        metadata, arrays, diagnostics = episode["metadata"], episode["arrays"], episode["diagnostics"]
        validate_episode(metadata, arrays, diagnostics)
        split = metadata["split"]
        if (self.manifests / f"{split}.json").exists():
            raise ValueError("cannot append to a sealed split")
        identity = f"{split}-{metadata['collection_index']:08d}"
        destination = self._path(identity)
        if destination.exists():
            raise FileExistsError(f"episode already committed: {identity}")
        estimate = sum(a.nbytes for a in (*arrays.values(), *diagnostics.values())) + len(canonical(metadata)) + 65536
        # ponytail: one directory scan per episode; cache usage if profiling justifies it.
        if self.disk_bytes() + estimate + 1024**3 > self.max_bytes:
            raise OSError("campaign storage budget exhausted (including 1 GiB headroom)")
        if shutil.disk_usage(self.root).free < estimate + 1024**3:
            raise OSError("insufficient free disk space; observed data are never deleted")
        pending = Path(tempfile.mkdtemp(prefix=".pending-", dir=self.episodes))
        for name, values in (("observed", arrays), ("diagnostics", diagnostics)):
            with (pending / f"{name}.npz").open("wb") as handle:
                np.savez_compressed(handle, **values)
                handle.flush()
                os.fsync(handle.fileno())
        record = {"metadata": plain(metadata), "sha256": {
            name: file_hash(pending / f"{name}.npz") for name in ("observed", "diagnostics")}}
        write_once(pending / "record.json", record)
        os.rename(pending, destination)
        return identity

    def load_episode(self, episode_id: str, *, include_diagnostics: bool = False) -> dict[str, Any]:
        """Verify checksums and reopen arrays without pickle or hidden learner inputs."""
        path = self._path(episode_id)
        record = json.loads((path / "record.json").read_text(encoding="utf-8"))
        output: dict[str, Any] = {"metadata": record["metadata"]}
        for name in ("observed", "diagnostics"):
            target = path / f"{name}.npz"
            if target.is_symlink() or file_hash(target) != record["sha256"][name]:
                raise ValueError(f"checksum mismatch: {episode_id}/{name}")
            if name == "diagnostics" and not include_diagnostics:
                continue
            with np.load(target, allow_pickle=False) as archive:
                values = {key: archive[key] for key in archive.files}
            for array in values.values():
                array.flags.writeable = False
            output["arrays" if name == "observed" else name] = values
        validate_episode(output["metadata"], output["arrays"], output.get("diagnostics"))
        expected = f"{output['metadata']['split']}-{output['metadata']['collection_index']:08d}"
        if episode_id != expected:
            raise ValueError("episode storage identity mismatch")
        return output

    def audit(self, split: str) -> dict[str, Any]:
        """Report integrity and physical support; positive coverage still needs review."""
        ids = self.ids(split)
        errors: list[str] = []
        concerns: list[str] = []
        records: dict[str, str] = {}
        groups: dict[str, Any] = {}
        identities: set[str] = set()
        seeds: set[int] = set()
        configuration_hashes: set[str] = set()
        source_hashes: set[str] = set()
        for index, identity in enumerate(ids):
            try:
                episode = self.load_episode(identity, include_diagnostics=True)
                m, a, d = episode["metadata"], episode["arrays"], episode["diagnostics"]
                if m["collection_index"] != index:
                    raise ValueError("collection indices are not contiguous")
                if m["episode_id"] in identities or m["environment_seed"] in seeds or m["policy_seed"] in seeds:
                    raise ValueError("duplicate episode or seed")
                identities.add(m["episode_id"])
                seeds.update((m["environment_seed"], m["policy_seed"]))
                configuration_hashes.add(m["configuration_hash"])
                source_hashes.add(sha256(canonical(m["source"])).hexdigest())
                records[identity] = file_hash(self._path(identity) / "record.json")
                key = f"{m['task']}/{m['reset_stratum']}"
                group = groups.setdefault(key, {"episodes": 0, "transitions": 0, "successes": 0,
                    "contact_steps": 0, "contact_episodes": 0, "returns": [], "rewards": [],
                    "maximum_object_displacement_m": [], "object_path_length_m": [],
                    "zero_action_steps": 0, "action_histogram_4x4": np.zeros((4, 4), dtype=np.int64),
                    "position_cells": set(), "goal_cells": set(), "layouts": {}, "dynamics": {}})
                group["episodes"] += 1
                group["transitions"] += m["transitions"]
                group["successes"] += int(a["terminated"][-1])
                group["contact_steps"] += int(d["contact"].sum())
                group["contact_episodes"] += int(d["contact"].any())
                group["returns"].append(float(a["reward"].sum()))
                group["rewards"].extend(a["reward"].tolist())
                group["zero_action_steps"] += int(np.all(a["executed_action"] == 0, axis=1).sum())
                group["action_histogram_4x4"] += np.histogram2d(*a["executed_action"].T, bins=[np.linspace(-2, 2, 5)] * 2)[0].astype(np.int64)
                positions = d["object_position"] if m["task"] == "push" else d["agent_position"]
                cells = np.clip((positions * 10).astype(int), 0, 9)
                group["position_cells"].update(map(tuple, cells.tolist()))
                group["goal_cells"].add(tuple(np.clip((a["goal"][2:] * 10).astype(int), 0, 9).tolist()))
                group["maximum_object_displacement_m"].append(float(np.linalg.norm(positions - positions[0], axis=1).max()) if m["task"] == "push" else 0.0)
                group["object_path_length_m"].append(float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()) if m["task"] == "push" else 0.0)
                for name, value in (("layouts", str(m["layout_index"])), ("dynamics", canonical(m["physical_parameters"]).decode())):
                    group[name][value] = group[name].get(value, 0) + 1
            except (ValueError, TypeError, KeyError, OSError, EOFError) as error:
                errors.append(f"{identity}: {error}")
        if not ids:
            errors.append("no completed episodes")
        if len(configuration_hashes) > 1:
            errors.append("mixed configurations within one split")
        if len(source_hashes) > 1:
            errors.append("mixed source code or runtimes within one split")
        # Cross-split comparison reads provenance, never learner-accessible diagnostics.
        for other in SPLITS:
            if other == split:
                continue
            for identity in self.ids(other):
                try:
                    m = json.loads((self._path(identity) / "record.json").read_text(encoding="utf-8"))["metadata"]
                    if m["episode_id"] in identities or seeds.intersection((m["environment_seed"], m["policy_seed"])):
                        errors.append(f"cross-split episode/seed overlap with {identity}")
                except (ValueError, KeyError, OSError) as error:
                    errors.append(f"cannot verify split isolation for {identity}: {error}")
        for key in ("reach/broad", "push/near", "push/broad"):
            if key not in groups:
                concerns.append(f"missing reset stratum: {key}")
        for key, group in groups.items():
            for name in ("returns", "rewards", "maximum_object_displacement_m", "object_path_length_m"):
                values = np.asarray(group[name])
                group[name] = {"min": float(values.min()), "median": float(np.median(values)),
                               "max": float(values.max()), "mean": float(values.mean())}
            for name in ("position_cells", "goal_cells"):
                group[name] = sorted([list(cell) for cell in group[name]])
            group["action_histogram_4x4"] = group["action_histogram_4x4"].tolist()
            group["contact_step_fraction"] = group["contact_steps"] / group["transitions"]
            group["success_fraction"] = group["successes"] / group["episodes"]
            if key.startswith("push") and (group["contact_episodes"] == 0 or group["maximum_object_displacement_m"]["max"] <= 1e-8):
                concerns.append(f"{key}: no observed contacts or object motion")
            if np.count_nonzero(group["action_histogram_4x4"]) < 2:
                concerns.append(f"{key}: degenerate action support")
        report = {"audit_version": "coverage-v1", "split": split, "episode_checksums": records,
                  "integrity": "pass" if not errors else "fail", "failure_count": len(errors),
                  "failures": errors, "coverage_concerns": concerns, "groups": groups,
                  "coverage_status": "deficient" if concerns else "review_required",
                  "bins": {"position_and_target_grid": [10, 10], "action_edges_N": [-2, -1, 0, 1, 2]},
                  "interpretation": "Coverage is descriptive; the frozen plan gives no numerical admission threshold. Zero success alone is not failure.",
                  "auditor_source_hash": file_hash(Path(__file__))}
        report["audit_hash"] = sha256(canonical(report)).hexdigest()
        return report

    def seal(self, ids: Sequence[str], split: str, *, reviewed_audit: str) -> DatasetSnapshot:
        """Freeze reviewed membership and checksums; no deficient split can be sealed."""
        if list(ids) != self.ids(split):
            raise ValueError("seal must cover the complete audited split in collection order")
        report = self.audit(split)
        if report["integrity"] != "pass" or report["coverage_status"] == "deficient":
            raise ValueError("dataset integrity/coverage admission failed; inspect the audit")
        if reviewed_audit != report["audit_hash"]:
            raise ValueError("supply the exact hash of the current reviewed audit")
        job_path = self.manifests / f"{split}.collection.json"
        if job_path.exists():
            job = json.loads(job_path.read_text(encoding="utf-8"))
            groups = report["groups"]
            counts = {task: sum(g["transitions"] for k, g in groups.items() if k.startswith(task + "/"))
                      for task in ("reach", "push")}
            if min(counts.values()) < job["requested_transitions"] // 2:
                raise ValueError("collection has not reached its requested per-task target")
            if groups["push/near"]["episodes"] != groups["push/broad"]["episodes"]:
                raise ValueError("collection has not completed its push-stratum pair")
        first = self.load_episode(ids[0])["metadata"]
        snapshot = DatasetSnapshot(
            episode_ids=tuple(ids), split=split, dataset_hash=report["audit_hash"],
            environment_configuration=first["configuration"]["environment"],
            source_provenance={"evidence_kind": "observed", "policy_identity": first["policy_identity"],
                               "configuration_hash": first["configuration_hash"],
                               "episode_checksums": report["episode_checksums"], "reviewed_audit": reviewed_audit},
            cache_versions={"raw": STORE_VERSION}, augmentation_manifest="not_generated",
            normalization_reference="not_fitted", split_version="disjoint-root-blocks-v1")
        write_once(self.manifests / f"{split}.reviewed-{reviewed_audit}.json", report)
        write_once(self.manifests / f"{split}.json", {f.name: plain(getattr(snapshot, f.name)) for f in fields(snapshot)})
        return snapshot

    def verify_snapshot(self, split: str) -> DatasetSnapshot:
        """Reopen a sealed snapshot and verify membership and every stored checksum."""
        if split not in SPLITS:
            raise ValueError("unknown split")
        raw = json.loads((self.manifests / f"{split}.json").read_text(encoding="utf-8"))
        snapshot = DatasetSnapshot(**raw)
        report = self.audit(split)
        if snapshot.split != split or tuple(self.ids(split)) != snapshot.episode_ids:
            raise ValueError("sealed membership changed")
        if report["integrity"] != "pass" or report["episode_checksums"] != dict(snapshot.source_provenance["episode_checksums"]):
            raise ValueError("sealed dataset changed")
        saved = json.loads((self.manifests / f"{split}.reviewed-{snapshot.dataset_hash}.json").read_text(encoding="utf-8"))
        saved_hash = saved.pop("audit_hash")
        if (sha256(canonical(saved)).hexdigest() != saved_hash or saved_hash != snapshot.dataset_hash
                or saved_hash != snapshot.source_provenance["reviewed_audit"]
                or saved["episode_checksums"] != report["episode_checksums"]):
            raise ValueError("reviewed audit provenance changed")
        return snapshot
