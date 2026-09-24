"""Observed-data commands and bounded Week 2 encoder/model diagnostics."""

from __future__ import annotations

import argparse
from importlib.metadata import version
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys
from time import perf_counter
from typing import Any

from .baselines import RandomPolicy, PrivilegedDiagnosticPolicy, ReactivePolicy
from .config import RunConfig, config_hash, load_config
from .data import EpisodeStore, SPLITS, canonical, episode_seed, file_hash, pack_episode, write_once
from .environment import PhysicsSandbox


def source_identity() -> dict[str, Any]:
    """Identify the executed source and libraries without requiring a Git checkout."""
    directory = Path(__file__).parent
    return {"source_sha256": {name: file_hash(directory / name) for name in (
        "__init__.py", "__main__.py", "baselines.py", "config.py", "contracts.py", "data.py", "environment.py",
        "encoder.py", "model.py")},
        "python": platform.python_version(),
        "libraries": {name: version(name) for name in ("torch", "torchvision", "numpy", "gymnasium", "pymunk", "pillow")}}


def collect_episodes(config: RunConfig, store: EpisodeStore, split: str, transitions: int,
                     *, episode_limit: int | None = None) -> dict[str, Any]:
    """Collect complete random-policy episodes, resuming only an identical job.

    Transition counts are nominal per-task minima; natural final observations
    are retained instead of cutting episodes to force an exact array length.
    Push strata alternate so completed collection has equal near/broad episodes.
    """
    if split not in SPLITS or type(transitions) is not int or transitions < 2 or transitions % 2:
        raise ValueError("request an even transition target >=2 and a supported split")
    if episode_limit is not None and (type(episode_limit) is not int or episode_limit < 1):
        raise ValueError("episode_limit must be a positive integer")
    if (store.manifests / f"{split}.json").exists():
        raise ValueError("split is sealed; verify it instead of recollecting")
    job = {"split": split, "configuration_hash": config_hash(config),
           "requested_transitions": transitions, "source": source_identity(),
           "policy_identity": RandomPolicy.identity, "seed_scheme": "root*2**40+index*2+stream",
           "completion_rule": "whole episodes; each task >= target/2; paired push strata"}
    write_once(store.manifests / f"{split}.collection.json", job)
    counts = {"reach": 0, "push": 0}
    push_episodes = 0
    ids = store.ids(split)
    for index, identity in enumerate(ids):
        episode = store.load_episode(identity, include_diagnostics=True)
        metadata = episode["metadata"]
        if metadata["collection_index"] != index or metadata["configuration_hash"] != job["configuration_hash"] or metadata["source"] != job["source"]:
            raise ValueError("cannot resume a different configuration, source, runtime, or discontinuous collection")
        counts[metadata["task"]] += metadata["transitions"]
        if metadata["task"] == "push":
            expected = "near" if push_episodes % 2 == 0 else "broad"
            if metadata["reset_stratum"] != expected:
                raise ValueError("push reset-stratum schedule changed")
            push_episodes += 1
    started = perf_counter()
    initial_count = len(ids)
    env, policy = PhysicsSandbox(config), RandomPolicy(config)
    target = transitions // 2
    root = config.dataset[f"{split}_seed"]

    def completed() -> bool:
        return min(counts.values()) >= target and push_episodes % 2 == 0

    try:
        while not completed():
            if episode_limit is not None and len(ids) - initial_count >= episode_limit:
                break
            index = len(ids)
            need_push = counts["push"] < target or push_episodes % 2 != 0
            task = "reach" if counts["reach"] < target and (index % 2 == 0 or not need_push) else "push"
            stratum = "near" if task == "push" and push_episodes % 2 == 0 else "broad"
            environment_seed, policy_seed = episode_seed(root, index), episode_seed(root, index, 1)
            observation, _ = env.reset(seed=environment_seed, options={"task": task, "stratum": stratum})
            policy.reset(policy_seed)
            recorded, diagnostics = [], [env.diagnostic_state()]
            for _ in range(config.environment["max_episode_steps"]):
                observation, _, terminated, truncated, _ = env.step(policy.act(observation))
                recorded.append(env.last_transition)
                diagnostics.append(env.diagnostic_state())
                if terminated or truncated:
                    break
            provenance = {"split": split, "collection_index": index, "environment_seed": environment_seed,
                          "policy_seed": policy_seed, "policy_identity": policy.identity, "source": job["source"]}
            identity = store.append_completed_episode(pack_episode(recorded, diagnostics, config, provenance))
            # Reopen every new commit immediately; a write is not evidence of a readable dataset.
            stored = store.load_episode(identity)
            counts[task] += stored["metadata"]["transitions"]
            push_episodes += int(task == "push")
            ids.append(identity)
            if len(ids) % 10 == 0:
                print(f"{split}: {len(ids)} episodes, {sum(counts.values())} transitions", flush=True)
    finally:
        env.close()
    return {"split": split, "complete": completed(), "episodes": len(ids), "transitions_by_task": counts,
            "requested_transitions": transitions, "actual_transitions": sum(counts.values()),
            "new_episodes": len(ids) - initial_count, "collection_and_reopen_seconds": perf_counter() - started,
            "logical_campaign_bytes": store.disk_bytes(), "configuration_hash": job["configuration_hash"]}


def diagnose(config: RunConfig) -> dict[str, Any]:
    """Demonstrate controlled, obstacle-free solvability; do not collect this data."""
    results = []
    env = PhysicsSandbox(config)
    try:
        for task in ("reach", "push"):
            for seed in range(3):
                options = {"task": task, "layout_index": 0, "stratum": "broad",
                           "agent_position": [0.25, 0.3], "goal_position": [0.7, 0.3]}
                if task == "push":
                    options["object_position"] = [0.4, 0.3]
                observation, _ = env.reset(seed=seed, options=options)
                reactive = ReactivePolicy().act(observation)
                if not env.action_space.contains(reactive):
                    raise ValueError("reactive baseline produced an invalid action")
                policy = PrivilegedDiagnosticPolicy()
                for decision in range(config.environment["max_episode_steps"]):
                    _, _, terminated, truncated, _ = env.step(policy.act(env.diagnostic_state()))
                    if terminated or truncated:
                        break
                results.append({"task": task, "seed": seed, "success": bool(terminated), "decisions": decision + 1})
    finally:
        env.close()
    return {"diagnostic": "controlled-obstacle-free-solvability", "policy": PrivilegedDiagnosticPolicy.identity,
            "configuration_hash": config_hash(config), "results": results,
            "status": "pass" if all(r["success"] for r in results) else "fail",
            "scope": "Controlled examples only; no generalization or upper-bound claim; never training data."}


def main(argv: list[str] | None = None) -> int:
    """Dispatch implemented commands; failures always return nonzero."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("collect", "audit", "seal", "verify", "diagnose", "encoder-check", "cache", "smoke"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--config", type=Path, default=Path("configs/stage1.toml"))
        command_parser.add_argument("--data-root", type=Path, default=Path("data"))
        if command not in ("diagnose", "encoder-check", "smoke"):
            command_parser.add_argument("--split", choices=SPLITS, required=True)
        if command in ("encoder-check", "cache", "smoke"):
            command_parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
        if command == "cache":
            command_parser.add_argument("--batch-size", type=int, default=64)
        if command == "smoke":
            command_parser.add_argument("--cache", type=Path, required=True)
            command_parser.add_argument("--updates", type=int, default=500)
            command_parser.add_argument("--seed", type=int, default=0)
        if command == "collect":
            command_parser.add_argument("--transitions", type=int)
            command_parser.add_argument("--episode-limit", type=int, help="pause after this many new completed episodes")
        if command == "seal":
            command_parser.add_argument("--reviewed-audit", required=True, help="SHA-256 of the coverage report you inspected")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "encoder-check":
            from .encoder import FrozenVisualEncoder, encoder_diagnostics
            result = encoder_diagnostics(FrozenVisualEncoder(config, args.device))
            print(json.dumps(result, indent=2))
            return 0 if result["status"] == "pass" else 2
        if args.command == "cache":
            from .encoder import FrozenVisualEncoder, cache_features, encoder_diagnostics, fit_normalizer
            store = EpisodeStore(args.data_root)
            snapshot = store.verify_snapshot(args.split)
            encoder = FrozenVisualEncoder(config, args.device)
            checks = encoder_diagnostics(encoder)
            if checks["status"] != "pass":
                print(json.dumps(checks, indent=2))
                return 2
            cache = cache_features(snapshot, encoder, config.augmentation, store=store, batch_size=args.batch_size)
            result = {"status": "pass", "cache": str(cache), "encoder_checks": checks,
                      "split": snapshot.split, "episodes": len(snapshot.episode_ids)}
            if snapshot.split == "train":
                normalizer = fit_normalizer(snapshot, cache=cache, store=store, std_floor=config.encoder["normalization_std_floor"])
                normalizer.save(cache / "normalizer.json")
                result["normalizer"] = str(cache / "normalizer.json")
                result["normalizer_hash"] = normalizer.version
            print(json.dumps(result, indent=2))
            return 0
        if args.command == "smoke":
            result = week2_smoke(config, EpisodeStore(args.data_root), args.cache, args.device, args.updates, args.seed)
            path = args.data_root.parent / "runs" / "week2" / f"smoke-{sha256(canonical(result)).hexdigest()}.json"
            write_once(path, result)
            print(json.dumps({**result, "report_path": str(path)}, indent=2))
            return 0 if result["status"] == "pass" else 2
        if args.command == "diagnose":
            result = diagnose(config)
            print(json.dumps(result, indent=2))
            return 0 if result["status"] == "pass" else 2
        store = EpisodeStore(args.data_root)
        if args.command == "collect":
            target = args.transitions if args.transitions is not None else (
                8000 if args.split == "pilot" else config.dataset[f"{args.split}_transitions"])
            result = collect_episodes(config, store, args.split, target, episode_limit=args.episode_limit)
            print(json.dumps(result, indent=2))
            return 0 if result["complete"] else 3
        if args.command == "audit":
            result = store.audit(args.split)
            path = store.root.parent / "runs" / "week1" / f"{args.split}.audit-{result['audit_hash']}.json"
            write_once(path, result)
            summary = {key: result[key] for key in ("integrity", "coverage_status", "failures", "coverage_concerns", "audit_hash")}
            summary["groups"] = {key: {
                "episodes": group["episodes"], "transitions": group["transitions"],
                "contact_episodes": group["contact_episodes"], "contact_step_fraction": group["contact_step_fraction"],
                "maximum_object_displacement_m": group["maximum_object_displacement_m"],
                "successes": group["successes"], "position_cells_visited": len(group["position_cells"]),
                "goal_cells_visited": len(group["goal_cells"])
            } for key, group in result["groups"].items()}
            summary["report_path"] = str(path)
            summary["uncommitted_directories_ignored"] = len(list(store.episodes.glob(".pending-*")))
            print(json.dumps(summary, indent=2))
            return 0 if result["integrity"] == "pass" and result["coverage_status"] == "review_required" else 2
        if args.command == "seal":
            result = store.seal(store.ids(args.split), args.split, reviewed_audit=args.reviewed_audit)
        else:
            result = store.verify_snapshot(args.split)
        print(canonical({"status": "pass", "split": result.split, "episodes": len(result.episode_ids),
                         "dataset_hash": result.dataset_hash}).decode())
        return 0
    except KeyboardInterrupt:
        print("Interrupted. Completed episodes are retained; rerun the identical collect command to resume.", file=sys.stderr)
        return 130
    except (ValueError, TypeError, KeyError, OSError, EOFError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


def tiny_training_batch(cache: Path, store: EpisodeStore, normalizer: Any, device: str) -> Any:
    """Two reset-root episodes, both whole-view choices, eight learning transitions.

    This fixed diagnostic fixture is not a replay sampler. It never fabricates
    transitions, mixes views within a sequence, or uses evaluator-only state.
    """
    import numpy as np
    import torch
    from .contracts import SequenceBatch
    from .encoder import load_cached_episode
    snapshot = store.verify_snapshot("train")
    chosen = {}
    for identity in snapshot.episode_ids:
        record = load_cached_episode(cache, identity, store=store)
        source, arrays = record["source"], record["arrays"]
        task = source["metadata"]["task"]
        if record["manifest"]["dataset_hash"] != snapshot.dataset_hash or record["manifest"]["encoder_version"] != normalizer.provenance["encoder_version"]:
            raise ValueError("smoke cache/normalizer provenance mismatch")
        if source["metadata"]["transitions"] >= 8 and task not in chosen:
            if (~arrays["corrupted_validity"][1:9, 0]).any():
                chosen[task] = record
        if len(chosen) == 2:
            break
    if len(chosen) != 2:
        raise ValueError("smoke needs a Reach and Push episode with a blackout in their first eight transitions; use a larger training snapshot")
    if normalizer.provenance["dataset_hash"] != snapshot.dataset_hash or normalizer.provenance["cache_hash"] != cache.name:
        raise ValueError("smoke normalizer was fitted on a different training cache")
    rows = []
    for task in ("reach", "push"):
        r = chosen[task]
        arrays, raw = r["arrays"], r["source"]["arrays"]
        def sensory(view):
            return normalizer.transform(np.concatenate((arrays[f"{view}_visual"].astype(np.float32), arrays[f"{view}_proprio"]), -1), arrays[f"{view}_validity"])[:9]
        for view in ("clean", "corrupted"):
            rows.append({"features": sensory(view), "target": sensory("clean"), "validity": arrays[f"{view}_validity"][:9],
                         "ages": arrays[f"{view}_sample_age"][:9], "target_validity": arrays["clean_validity"][:9],
                         "raw": raw, "episode": r["source"]["metadata"]["episode_id"], "view": view})
    def tensor(values, dtype=torch.float32):
        return torch.tensor(np.stack(values), dtype=dtype, device=device)
    features = tensor([r["features"] for r in rows])
    validity = tensor([r["validity"] for r in rows], torch.bool)
    ages = tensor([r["ages"] for r in rows])
    actions = tensor([r["raw"]["executed_action"][:8] for r in rows])
    boundaries = torch.zeros(4, 9, dtype=torch.bool, device=device)
    boundaries[:, 0] = True
    targets = {key: tensor([r["raw"][key][:8] for r in rows], torch.bool if key in ("terminated", "truncated") else torch.float32)
               for key in ("reward", "cost", "continuation", "terminated", "truncated")}
    targets["sensory"] = tensor([r["target"] for r in rows])
    return SequenceBatch(
        observed_prefix={"features": features[:, :1], "validity": validity[:, :1], "sample_ages": ages[:, :1],
                         "mask": torch.ones(4, 1, dtype=torch.bool, device=device), "actions": actions[:, :0],
                         "dt": torch.zeros(4, 0, device=device)},
        learning_observations=features, executed_actions=actions, targets=targets, input_validity=validity,
        sample_ages=ages, target_validity=tensor([r["target_validity"] for r in rows], torch.bool),
        padding_mask=torch.ones(4, 8, dtype=torch.bool, device=device), observation_mask=torch.ones(4, 9, dtype=torch.bool, device=device),
        episode_boundaries=boundaries, episode_ids=tuple(r["episode"] for r in rows), view_ids=tuple(r["view"] for r in rows),
        goals=tensor([r["raw"]["goal"] for r in rows]), elapsed_simulation_time=tensor([r["raw"]["dt"][:8] for r in rows]))


def week2_smoke(config: RunConfig, store: EpisodeStore, cache: Path, device: str,
                updates: int = 500, seed: int = 0) -> dict[str, Any]:
    """Bounded real-model overfit check; no Week 3 trainer or Week 4 control."""
    import torch
    from .encoder import FeatureNormalizer
    from .model import RSSM, model_loss
    if type(updates) is not int or not 50 <= updates <= 500 or type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("smoke requires 50-500 updates and a uint32 seed")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable in this Python environment")
    started = perf_counter()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(seed)
    normalizer = FeatureNormalizer.load(cache / "normalizer.json")
    batch = tiny_training_batch(cache, store, normalizer, device)
    model = RSSM(config, encoder_version=normalizer.provenance["encoder_version"], normalizer_version=normalizer.version).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.model["learning_rate"], weight_decay=config.model["weight_decay"])
    fixed_noise = torch.randn(4, 9, 64, device=device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    @torch.no_grad()
    def measure():
        model.eval()
        outputs = model(batch, noise=fixed_noise)
        loss = model_loss(batch, outputs)
        mask = batch.input_validity[:, 1:] & batch.target_validity[:, 1:]
        mse = (outputs["sensory_mean"] - batch.targets["sensory"][:, 1:]).square()
        vision = mse[..., :4608][mask[..., 0]].mean()
        proprio = mse[..., 4608:][mask[..., 1]].mean()
        return {"total": float(loss["total"]), "vision_mse": float(vision), "proprio_mse": float(proprio),
                "reward_mse": float((outputs["reward_mean"] - batch.targets["reward"]).square().mean()),
                **{k + "_loss": float(v) for k, v in loss["components"].items()}}
    before = measure()
    gradient_min, gradient_max = float("inf"), 0.0
    for update in range(updates):
        if perf_counter() - started > 900:
            raise RuntimeError("Week 2 smoke exceeded its 15-minute budget")
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = model_loss(batch, model(batch))["total"]
        if not torch.isfinite(loss):
            raise ValueError("nonfinite tiny-dataset loss")
        loss.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.model["gradient_norm_cap"], error_if_nonfinite=True))
        if norm == 0:
            raise ValueError("zero model gradient")
        gradient_min, gradient_max = min(gradient_min, norm), max(gradient_max, norm)
        optimizer.step()
        if (update + 1) % 25 == 0:
            print(f"Week 2 overfit: {update + 1}/{updates} updates; loss={float(loss.detach()):.5f}", flush=True)
    after = measure()
    gates = {"loss_improved": after["total"] < before["total"],
             "vision_mse_halved": after["vision_mse"] < .5 * before["vision_mse"],
             "proprio_mse_halved": after["proprio_mse"] < .5 * before["proprio_mse"],
             "reward_mse_improved": after["reward_mse"] < before["reward_mse"],
             "cost_loss_improved": after["cost_loss"] < before["cost_loss"],
             "continuation_loss_improved": after["continuation_loss"] < before["continuation_loss"]}
    return {"status": "pass" if all(gates.values()) else "fail", "scope": "Week 2 tiny-training-set overfit only; not the Stage 1 scientific gate",
            "gates": gates, "before": before, "after": after, "updates": updates, "seed": seed, "device": device,
            "trainable_parameters": sum(p.numel() for p in model.parameters()), "sequences": 4, "transitions_per_sequence": 8,
            "blackout_observations": int((~batch.input_validity[..., 0]).sum()),
            "gradient_norm_min": gradient_min, "gradient_norm_max": gradient_max,
            "wall_seconds": perf_counter() - started, "normalizer_hash": normalizer.version,
            "cache_hash": cache.name, "configuration_hash": config_hash(config), "source": source_identity(),
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved() if device == "cuda" else None}


if __name__ == "__main__":
    raise SystemExit(main())
