#!/usr/bin/env python3
"""Plan and execute the formal OPD ablation registry one GPU-exclusive cell at a time."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TextIO


SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS = {1, 2}
SCIENTIFIC_PROTOCOL = "scientific"
EXECUTION_STATES = {
    "registered",
    "running",
    "training_complete",
    "evaluation_complete",
    "probe_complete",
    "rendered",
    "scientific_result_available",
    "interrupted_resumable",
    "infrastructure_failed",
    "protocol_invalid",
    "blocked_external",
    "blocked_hardware",
}
CONCLUSION_STATES = {
    "not_assessed",
    "replicated",
    "not_replicated_at_seed_42",
    "inconclusive",
}
SAFE_ENV_KEYS = {
    "HOME",
    "USER",
    "LOGNAME",
    "PATH",
    "PYTHONPATH",
    "LD_LIBRARY_PATH",
    "LIBRARY_PATH",
    "CPATH",
    "LANG",
    "LC_ALL",
    "TMPDIR",
    "HF_HOME",
    "HF_HUB_CACHE",
    "TRANSFORMERS_CACHE",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "CUDA_VISIBLE_DEVICES",
}
SOURCE_PATHS = (
    "reproduce_4b/run_opd_4b.sh",
    "reproduce_4b/run_ablations.py",
    "reproduce_4b/evaluate_ablation.py",
    "reproduce_4b/paper_eval_contract.py",
    "reproduce_4b/aggregate_ablations.py",
    "reproduce_4b/ablation_matrix.json",
    "reproduce_4b/prepare_ablation_data.py",
    "reproduce_4b/preflight.py",
    "reproduce_4b/pin_models.py",
    "reproduce_4b/merge_checkpoint.sh",
    "reproduce_4b/generate_eval.py",
    "reproduce_4b/grade_eval.py",
    "reproduce_4b/plot_metrics.py",
    "reproduce_4b/plot_position_entropy.py",
    "reproduce_4b/constraints-2xa100-cu128.txt",
    "verl/verl/utils/opd.py",
    "verl/verl/trainer/ppo/ray_trainer.py",
    "verl/verl/utils/tracking.py",
    "verl/verl/workers/actor/dp_actor.py",
    "verl/verl/workers/fsdp_workers.py",
    "verl/verl/workers/config/rollout.py",
    "verl/verl/trainer/config/rollout/rollout.yaml",
    "verl/verl/trainer/config/ppo_trainer.yaml",
)


@dataclass(frozen=True)
class CellPlan:
    group_id: str
    group_label: str
    cell_id: str
    label: str
    fidelity: str
    paper_location: str
    env: dict[str, str]
    fingerprint: str
    dataset: dict[str, Any]
    protocol: str = ""
    registry_schema_version: int = 1
    scientific: dict[str, Any] = field(default_factory=dict)
    models: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GPUAllocation:
    """One uniform GPU allocation shared by every cell in a runner invocation."""

    count: int
    devices: tuple[str, ...]

    @property
    def selector(self) -> str:
        return ",".join(self.devices)


class GPULocks:
    """A close-compatible bundle of per-device advisory locks."""

    def __init__(self, handles: Sequence[TextIO]) -> None:
        self._handles = list(handles)

    def close(self) -> None:
        while self._handles:
            self._handles.pop().close()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def absolute_path_preserve_symlinks(path: Path) -> Path:
    """Make a path absolute without dereferencing a virtualenv executable."""

    return Path(os.path.abspath(path.expanduser()))


def replace_repo(value: Any, repo_root: Path) -> Any:
    if isinstance(value, str):
        return value.replace("{repo}", str(repo_root))
    if isinstance(value, list):
        return [replace_repo(item, repo_root) for item in value]
    if isinstance(value, dict):
        return {key: replace_repo(item, repo_root) for key, item in value.items()}
    return value


def _reject_duplicate_json_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_registry(path: Path, repo_root: Path) -> dict[str, Any]:
    with path.expanduser().open("r", encoding="utf-8") as handle:
        registry = json.load(handle, object_pairs_hook=_reject_duplicate_json_keys)
    if not isinstance(registry, dict):
        raise ValueError("ablation registry must be a JSON object")
    schema_version = registry.get("schema_version")
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"unsupported registry schema {schema_version!r}; "
            f"expected one of {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
        )
    allowed = {"schema_version", "suite_id", "paper", "protocols", "datasets", "groups"}
    if schema_version >= 2:
        allowed.update({"scientific_spec", "blocked_cells"})
    unknown = set(registry) - allowed
    if unknown:
        raise ValueError(f"unknown top-level registry keys: {sorted(unknown)}")
    if not isinstance(registry.get("suite_id"), str) or not registry["suite_id"]:
        raise ValueError("registry suite_id must be a non-empty string")
    if not isinstance(registry.get("protocols"), dict) or not registry["protocols"]:
        raise ValueError("registry must define protocols")
    if not isinstance(registry.get("groups"), list) or not registry["groups"]:
        raise ValueError("registry must define groups")
    if schema_version >= 2:
        validate_scientific_registry(registry)
    return replace_repo(registry, repo_root)


def validate_scientific_registry(registry: Mapping[str, Any]) -> None:
    """Validate the v2 state model and fail-closed blocker declarations."""

    spec = registry.get("scientific_spec")
    if not isinstance(spec, dict):
        raise ValueError("schema v2 registry must define scientific_spec")
    comparability = spec.get("comparability")
    if not isinstance(comparability, dict) or set(comparability) != {
        "training",
        "evaluation",
        "provenance",
    }:
        raise ValueError("scientific_spec.comparability must define training/evaluation/provenance")
    blocked = registry.get("blocked_cells")
    if not isinstance(blocked, list):
        raise ValueError("schema v2 registry blocked_cells must be a list")
    seen: set[str] = set()
    for item in blocked:
        if not isinstance(item, dict):
            raise ValueError("blocked_cells entries must be objects")
        cell_id = item.get("cell_id")
        if not isinstance(cell_id, str) or not cell_id or cell_id in seen:
            raise ValueError(f"invalid or duplicate blocked cell ID {cell_id!r}")
        seen.add(cell_id)
        if item.get("execution_state") not in {"blocked_external", "blocked_hardware"}:
            raise ValueError(f"blocked cell {cell_id} has an invalid execution_state")
        if item.get("conclusion_state") not in CONCLUSION_STATES:
            raise ValueError(f"blocked cell {cell_id} has an invalid conclusion_state")
        if not isinstance(item.get("reason"), str) or not item["reason"]:
            raise ValueError(f"blocked cell {cell_id} must disclose a reason")


def registered_blockers(registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(item) for item in registry.get("blocked_cells", [])]


def source_tree_hash(repo_root: Path) -> str:
    digest = hashlib.sha256()
    for relative in SOURCE_PATHS:
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"required ablation source is missing: {path}")
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def snapshot_inventory_sha256(snapshot: Path) -> str:
    """Fingerprint an immutable snapshot without rereading multi-GB weight blobs."""

    inventory: list[dict[str, Any]] = []
    for path in sorted(snapshot.rglob("*")):
        if not path.is_file():
            continue
        stat = path.stat()
        item: dict[str, Any] = {
            "path": str(path.relative_to(snapshot)),
            "size": stat.st_size,
        }
        if path.is_symlink():
            item["blob"] = path.resolve().name
        elif stat.st_size <= 16 * 1024 * 1024:
            item["sha256"] = sha256_file(path)
        else:
            item["large_regular_file"] = True
        inventory.append(item)
    if not inventory:
        raise ValueError(f"model snapshot is empty: {snapshot}")
    return canonical_hash(inventory)


def _hub_cache_roots(repo_root: Path) -> list[Path]:
    roots: list[Path] = []
    if os.environ.get("HF_HUB_CACHE"):
        roots.append(Path(os.environ["HF_HUB_CACHE"]).expanduser())
    if os.environ.get("HF_HOME"):
        roots.append(Path(os.environ["HF_HOME"]).expanduser() / "hub")
    roots.extend(
        [
            repo_root.parent / ".root-cache/huggingface/hub",
            Path.home() / ".cache/huggingface/hub",
        ]
    )
    unique: list[Path] = []
    for root in roots:
        absolute = root.resolve()
        if absolute not in unique:
            unique.append(absolute)
    return unique


def _validate_local_snapshot(snapshot: Path) -> None:
    if not snapshot.is_dir() or not (snapshot / "config.json").is_file():
        raise ValueError(f"local model snapshot has no config.json: {snapshot}")
    weights = [
        path
        for pattern in ("*.safetensors", "*.bin")
        for path in snapshot.glob(pattern)
        if path.is_file() and path.stat().st_size > 0
    ]
    if not weights:
        raise ValueError(f"local model snapshot has no complete weights: {snapshot}")
    for index_path in snapshot.glob("*.safetensors.index.json"):
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map") if isinstance(payload, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"invalid safetensors index: {index_path}")
        missing = sorted(
            name for name in set(weight_map.values()) if not (snapshot / str(name)).is_file()
        )
        if missing:
            raise ValueError(f"local model snapshot {snapshot} is missing shards: {missing[:3]}")


def resolve_local_model_snapshot(
    source: str, revision: str, repo_root: Path
) -> dict[str, Any]:
    """Resolve an already-cached immutable model without any Hub API call."""

    local = Path(source).expanduser()
    if local.exists():
        snapshot = local.resolve()
        resolved_revision = revision if len(revision) == 40 else "local"
    else:
        if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
            raise ValueError(
                f"scientific model {source} requires an immutable 40-character revision"
            )
        encoded = "models--" + source.replace("/", "--")
        candidates = [root / encoded / "snapshots" / revision for root in _hub_cache_roots(repo_root)]
        snapshot = next((candidate.resolve() for candidate in candidates if candidate.is_dir()), None)
        if snapshot is None:
            searched = ", ".join(str(path) for path in candidates)
            raise FileNotFoundError(
                f"scientific model {source}@{revision} is not cached locally; searched {searched}"
            )
        resolved_revision = revision
    _validate_local_snapshot(snapshot)
    return {
        "source": source,
        "requested_revision": revision,
        "resolved_revision": resolved_revision,
        "snapshot_path": str(snapshot),
        "snapshot_inventory_sha256": snapshot_inventory_sha256(snapshot),
    }


def resolve_scientific_models(
    env: Mapping[str, str], repo_root: Path
) -> tuple[dict[str, str], dict[str, Any]]:
    resolved_env = dict(env)
    models: dict[str, Any] = {}
    for role, model_key, revision_key in (
        ("student", "STUDENT_MODEL", "STUDENT_REVISION"),
        ("teacher", "TEACHER_MODEL", "TEACHER_REVISION"),
    ):
        source = resolved_env.get(model_key)
        revision = resolved_env.get(revision_key)
        if not source or not revision:
            raise ValueError(f"scientific cell must explicitly define {model_key} and {revision_key}")
        model = resolve_local_model_snapshot(source, revision, repo_root)
        models[role] = model
        resolved_env[model_key] = model["snapshot_path"]
        resolved_env[revision_key] = model["resolved_revision"]
    return resolved_env, models


def scientific_cell_spec(
    registry: Mapping[str, Any], group: Mapping[str, Any], cell: Mapping[str, Any]
) -> dict[str, Any]:
    group_spec = group.get("scientific", {})
    cell_override = cell.get("scientific", {})
    if not isinstance(group_spec, dict) or not isinstance(cell_override, dict):
        raise ValueError(f"cell {cell.get('id')} has malformed scientific metadata")
    merged = {**group_spec, **cell_override}
    suite_spec = registry.get("scientific_spec", {})
    probe_names = merged.get("probes", [])
    probes = suite_spec.get("suite_probes", {})
    merged["evaluation"] = {
        "trend": suite_spec.get("trend_evaluation", {}),
        "trend_steps": merged.get("trend_steps", []),
        "exact": suite_spec.get("exact_evaluation", {}),
        "exact_steps": merged.get("exact_steps", []),
    }
    merged["probes"] = {name: probes[name] for name in probe_names}
    return merged


def validate_datasets(registry: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("dataset validation requires pyarrow") from exc

    by_path: dict[str, dict[str, Any]] = {}
    for dataset_id, raw_spec in registry.get("datasets", {}).items():
        if not isinstance(raw_spec, dict):
            raise ValueError(f"dataset {dataset_id} must be an object")
        path = Path(str(raw_spec.get("path", ""))).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"dataset {dataset_id} is missing: {path}")
        rows = pq.ParquetFile(path).metadata.num_rows
        digest = sha256_file(path)
        if rows != raw_spec.get("rows"):
            raise ValueError(f"dataset {dataset_id} rows={rows}, expected {raw_spec.get('rows')}")
        if digest != raw_spec.get("sha256"):
            raise ValueError(f"dataset {dataset_id} sha256={digest}, expected {raw_spec.get('sha256')}")
        validated = {"id": dataset_id, **raw_spec, "path": str(path)}
        manifest_value = raw_spec.get("selection_manifest")
        if manifest_value is not None:
            manifest_path = Path(str(manifest_value)).expanduser().resolve()
            if not manifest_path.is_file():
                raise FileNotFoundError(
                    f"dataset {dataset_id} selection manifest is missing: {manifest_path}"
                )
            manifest = read_json(manifest_path)
            if manifest is None or manifest.get("seed") != registry.get("scientific_spec", {}).get(
                "primary_seed"
            ):
                raise ValueError(f"dataset {dataset_id} selection manifest has the wrong seed")
            validated["selection_manifest"] = str(manifest_path)
            validated["selection_manifest_sha256"] = sha256_file(manifest_path)
        by_path[str(path)] = validated
    return by_path


def validate_group(group: Mapping[str, Any], schema_version: int = 1) -> None:
    required = {"id", "paper_location", "fidelity", "question", "factor_keys", "constants", "cells"}
    missing = required - set(group)
    if missing:
        raise ValueError(f"group {group.get('id', '?')} is missing {sorted(missing)}")
    factor_keys = group["factor_keys"]
    if not isinstance(factor_keys, list) or not factor_keys or len(set(factor_keys)) != len(factor_keys):
        raise ValueError(f"group {group['id']} factor_keys must be a unique non-empty list")
    if set(factor_keys) & set(group["constants"]):
        raise ValueError(f"group {group['id']} puts a factor key in constants")
    cells = group["cells"]
    paired_blockers = group.get("blocked_comparators", [])
    singleton_allowed = (
        schema_version >= 2
        and isinstance(paired_blockers, list)
        and bool(paired_blockers)
    )
    if not isinstance(cells, list) or not cells or (len(cells) < 2 and not singleton_allowed):
        raise ValueError(
            f"group {group['id']} must contain at least two cells or disclose a blocked comparator"
        )
    seen: set[str] = set()
    for cell in cells:
        cell_id = cell.get("id")
        if not isinstance(cell_id, str) or not cell_id or cell_id in seen:
            raise ValueError(f"group {group['id']} has an invalid or duplicate cell id {cell_id!r}")
        seen.add(cell_id)
        factors = cell.get("factors")
        if not isinstance(factors, dict) or set(factors) != set(factor_keys):
            raise ValueError(
                f"cell {cell_id} factors={sorted(factors or {})}, expected exactly {sorted(factor_keys)}"
            )
    if schema_version >= 2:
        scientific = group.get("scientific")
        if not isinstance(scientific, dict):
            raise ValueError(f"scientific group {group['id']} must define scientific metadata")
        expected = scientific.get("expected_final_step")
        milestones = scientific.get("milestone_steps")
        if not isinstance(expected, int) or isinstance(expected, bool) or expected <= 0:
            raise ValueError(f"scientific group {group['id']} has invalid expected_final_step")
        if (
            not isinstance(milestones, list)
            or not milestones
            or any(not isinstance(step, int) or isinstance(step, bool) for step in milestones)
            or milestones != sorted(set(milestones))
            or milestones[-1] != expected
        ):
            raise ValueError(
                f"scientific group {group['id']} milestones must be unique, sorted, and end at {expected}"
            )


def select_groups(
    registry: Mapping[str, Any],
    requested_groups: set[str],
    requested_cells: set[str],
    include_extensions: bool,
) -> list[dict[str, Any]]:
    all_groups = registry["groups"]
    group_ids = [group.get("id") for group in all_groups]
    cell_ids = [cell.get("id") for group in all_groups for cell in group.get("cells", [])]
    if len(set(group_ids)) != len(group_ids):
        raise ValueError("cell registry contains duplicate group IDs")
    if len(set(cell_ids)) != len(cell_ids):
        raise ValueError("cell registry contains duplicate global cell IDs")
    unknown_groups = requested_groups - set(group_ids)
    unknown_cells = requested_cells - set(cell_ids)
    if unknown_groups:
        raise ValueError(f"unknown group(s): {sorted(unknown_groups)}")
    if unknown_cells:
        raise ValueError(f"unknown cell(s): {sorted(unknown_cells)}")

    selected: list[dict[str, Any]] = []
    for raw_group in all_groups:
        validate_group(raw_group, int(registry.get("schema_version", 1)))
        if requested_groups and raw_group["id"] not in requested_groups:
            continue
        if (
            raw_group.get("disabled_by_default")
            and not include_extensions
            and not requested_groups
            and not requested_cells
        ):
            continue
        cells = [cell for cell in raw_group["cells"] if not requested_cells or cell["id"] in requested_cells]
        if not cells:
            continue
        selected.append({**raw_group, "cells": cells})
    if not selected:
        raise ValueError("selection contains no ablation cells")
    return selected


def build_plans(
    registry: Mapping[str, Any],
    groups: Sequence[Mapping[str, Any]],
    protocol_name: str,
    seed: int,
    dataset_by_path: Mapping[str, dict[str, Any]],
    source_hash: str,
    matrix_sha256: str,
    repo_root: Path | None = None,
) -> list[CellPlan]:
    protocols = registry["protocols"]
    if protocol_name not in protocols:
        raise ValueError(f"unknown protocol {protocol_name!r}; choose from {sorted(protocols)}")
    protocol = protocols[protocol_name]
    if not isinstance(protocol.get("preset"), str) or not isinstance(protocol.get("overrides", {}), dict):
        raise ValueError(f"protocol {protocol_name} is malformed")

    plans: list[CellPlan] = []
    registry_schema = int(registry.get("schema_version", 1))
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[1]
    for group in groups:
        allowed_protocols = group.get("allowed_protocols")
        if allowed_protocols is not None and protocol_name not in allowed_protocols:
            raise ValueError(f"group {group['id']} only allows protocols {allowed_protocols}")
        common = {str(key): str(value) for key, value in protocol.get("overrides", {}).items()}
        common.update({str(key): str(value) for key, value in group["constants"].items()})
        common.update(
            {
                str(key): str(value)
                for key, value in group.get("protocol_overrides", {}).get(protocol_name, {}).items()
            }
        )
        common.update({"PRESET": str(protocol["preset"]), "SEED": str(seed), "ROLLOUT_SEED": str(seed)})

        invariant_reference: dict[str, str] | None = None
        for cell in group["cells"]:
            env = {**common, **{str(key): str(value) for key, value in cell["factors"].items()}}
            invariant = {key: value for key, value in env.items() if key not in group["factor_keys"]}
            if invariant_reference is None:
                invariant_reference = invariant
            elif invariant != invariant_reference:
                raise ValueError(f"group {group['id']} changes an undeclared control variable")

            train_data = str(Path(env["TRAIN_DATA"]).expanduser().resolve())
            env["TRAIN_DATA"] = train_data
            dataset = dataset_by_path.get(train_data)
            if dataset is None:
                raise ValueError(f"cell {cell['id']} uses an unregistered TRAIN_DATA: {train_data}")
            scientific: dict[str, Any] = {}
            models: dict[str, Any] = {}
            if protocol_name == SCIENTIFIC_PROTOCOL:
                if registry_schema < 2:
                    raise ValueError("scientific protocol requires a schema v2 registry")
                scientific = scientific_cell_spec(registry, group, cell)
                expected = scientific.get("expected_final_step")
                if str(expected) != env.get("TOTAL_TRAINING_STEPS"):
                    raise ValueError(
                        f"cell {cell['id']} expected_final_step does not match TOTAL_TRAINING_STEPS"
                    )
                milestone_csv = ",".join(str(step) for step in scientific.get("milestone_steps", []))
                if milestone_csv != env.get("MILESTONE_STEPS"):
                    raise ValueError(
                        f"cell {cell['id']} milestone_steps do not match MILESTONE_STEPS"
                    )
                env, models = resolve_scientific_models(env, repo_root)
            identity = {
                "schema_version": registry_schema,
                "suite_id": registry["suite_id"],
                "protocol": protocol_name,
                "group_id": group["id"],
                "cell_id": cell["id"],
                "env": env,
                "dataset_sha256": dataset["sha256"],
                "source_tree_sha256": source_hash,
                "matrix_sha256": matrix_sha256,
                "models": models,
                "scientific": scientific,
            }
            plans.append(
                CellPlan(
                    group_id=group["id"],
                    group_label=group["question"],
                    cell_id=cell["id"],
                    label=cell["label"],
                    fidelity=group["fidelity"],
                    paper_location=group["paper_location"],
                    env=env,
                    fingerprint=canonical_hash(identity),
                    dataset=dataset,
                    protocol=protocol_name,
                    registry_schema_version=registry_schema,
                    scientific=scientific,
                    models=models,
                )
            )
    return plans


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def write_once_json(path: Path, payload: Mapping[str, Any], label: str) -> bool:
    """Create an immutable JSON contract, accepting only byte-equivalent retries."""

    existing = read_json(path)
    if existing is not None:
        if existing != dict(payload):
            raise RuntimeError(f"existing {label} differs: {path}")
        return False
    atomic_json(path, payload)
    return True


def comparability_for_plan(plan: CellPlan) -> dict[str, str]:
    if plan.protocol == SCIENTIFIC_PROTOCOL:
        return {
            "training": "hardware_adapted_2xa100",
            "evaluation": "paper_evaluation_protocol",
            "provenance": "local_snapshot_and_seed_locked_data",
        }
    return {
        "training": plan.fidelity,
        "evaluation": "not_assessed",
        "provenance": "legacy_registry",
    }


def build_run_contract(
    registry: Mapping[str, Any],
    plan: CellPlan,
    seed: int,
    source_hash: str,
    matrix_sha256: str,
    allocation: GPUAllocation,
) -> dict[str, Any]:
    """Build the deterministic per-cell contract written before the first attempt."""

    if plan.protocol != SCIENTIFIC_PROTOCOL:
        raise ValueError("run contracts are required only for the scientific protocol")
    scientific = plan.scientific
    expected = int(scientific["expected_final_step"])
    milestones = [int(step) for step in scientific["milestone_steps"]]
    return {
        "schema_version": 1,
        "suite_id": registry["suite_id"],
        "protocol": plan.protocol,
        "seed": seed,
        "group_id": plan.group_id,
        "cell_id": plan.cell_id,
        "fingerprint": plan.fingerprint,
        "matrix_sha256": matrix_sha256,
        "scientific_spec": registry.get("scientific_spec", {}),
        "source_tree_sha256": source_hash,
        "paper_location": plan.paper_location,
        "fidelity": plan.fidelity,
        "dataset": plan.dataset,
        "models": plan.models,
        "hardware": {
            "gpu_count": allocation.count,
            "cuda_visible_devices": list(allocation.devices),
            "required_gpu_class": "A100-80GB",
            "minimum_free_gib_per_card": int(plan.env.get("MIN_FREE_GIB", "75")),
        },
        "training": {
            "expected_final_step": expected,
            "milestone_steps": milestones,
            "recovery_checkpoint_frequency": int(plan.env["SAVE_FREQ"]),
            "recovery_checkpoint_retention": int(plan.env["MAX_ACTOR_CKPTS_TO_KEEP"]),
            "environment": dict(sorted(plan.env.items())),
            "phase_boundaries": scientific.get("phase_boundaries", []),
        },
        "evaluation": scientific["evaluation"],
        "probes": scientific.get("probes", {}),
        "position_entropy": {
            "main_steps": scientific.get("position_entropy_main_steps", []),
            "supplement_steps": scientific.get("position_entropy_supplement_steps", []),
        },
        "comparability": comparability_for_plan(plan),
        "conclusion_state": "not_assessed",
    }


def contract_for_cell(
    registry: Mapping[str, Any],
    plan: CellPlan,
    cell_root: Path,
    seed: int,
    source_hash: str,
    matrix_sha256: str,
    allocation: GPUAllocation,
) -> tuple[Path, str]:
    contract = build_run_contract(
        registry, plan, seed, source_hash, matrix_sha256, allocation
    )
    path = cell_root / "run_contract.json"
    write_once_json(path, contract, "run contract")
    return path, sha256_file(path)


def _all_finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    return True


def validate_canonical_metrics(path: Path, expected_final_step: int) -> dict[str, Any]:
    """Require one finite record for every step 1..N, in canonical order."""

    if not path.is_file():
        raise ValueError(f"canonical metrics file is missing: {path}")
    steps: list[int] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"metrics line {line_number} is invalid JSON: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"metrics line {line_number} must be an object")
            step = payload.get("step")
            if not isinstance(step, int) or isinstance(step, bool):
                raise ValueError(f"metrics line {line_number} has an invalid step")
            if not _all_finite(payload):
                raise ValueError(f"metrics step {step} contains NaN or infinity")
            steps.append(step)
    expected = list(range(1, expected_final_step + 1))
    if steps != expected:
        duplicates = sorted({step for step in steps if steps.count(step) > 1})
        missing = sorted(set(expected) - set(steps))
        unexpected = sorted(set(steps) - set(expected))
        raise ValueError(
            "canonical metrics are not exactly continuous 1.."
            f"{expected_final_step}: duplicates={duplicates[:5]}, "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}, last={steps[-1] if steps else None}"
        )
    return {
        "path": str(path.resolve()),
        "rows": len(steps),
        "first_step": 1,
        "last_step": expected_final_step,
        "sha256": sha256_file(path),
    }


def _checkpoint_inventory(path: Path) -> dict[str, Any]:
    files = [item for item in sorted(path.rglob("*")) if item.is_file()]
    if not files:
        raise ValueError(f"milestone checkpoint is empty: {path}")
    config = next((item for item in files if item.name == "config.json"), None)
    weights = [
        item
        for item in files
        if (
            item.suffix in {".pt", ".bin", ".safetensors"}
            and item.stat().st_size > 0
            and ("model" in item.name or item.suffix == ".safetensors")
        )
    ]
    if config is None or not weights:
        raise ValueError(f"milestone checkpoint is not loadable: {path}")
    inventory = [
        {"path": str(item.relative_to(path)), "size": item.stat().st_size}
        for item in files
    ]
    return {
        "path": str(path.resolve()),
        "file_count": len(files),
        "bytes": sum(item["size"] for item in inventory),
        "inventory_sha256": canonical_hash(inventory),
        "loadable": True,
    }


def validate_milestones(cell_root: Path, milestone_steps: Sequence[int]) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for step in milestone_steps:
        candidates = (
            cell_root / "milestones" / f"global_step_{step}",
            cell_root / "checkpoints" / f"global_step_{step}",
        )
        milestone = next((path for path in candidates if path.is_dir()), None)
        if milestone is None:
            raise ValueError(f"required milestone global_step_{step} is missing")
        artifacts.append({"step": step, **_checkpoint_inventory(milestone)})
    return artifacts


def strict_training_validation(
    cell_root: Path, expected_final_step: int, milestone_steps: Sequence[int]
) -> dict[str, Any]:
    metrics = validate_canonical_metrics(cell_root / "metrics.jsonl", expected_final_step)
    milestones = validate_milestones(cell_root, milestone_steps)
    return {"metrics": metrics, "milestones": milestones}


def legacy_state(execution_state: str) -> str:
    if execution_state == "running":
        return "running"
    if execution_state in {
        "training_complete",
        "evaluation_complete",
        "probe_complete",
        "rendered",
        "scientific_result_available",
    }:
        return "completed"
    return "failed"


def attempt_number(cell_root: Path) -> int:
    numbers: list[int] = []
    for path in cell_root.glob("attempt-*"):
        suffix = path.name.removeprefix("attempt-")
        if suffix.isdigit():
            numbers.append(int(suffix))
    return max(numbers, default=0) + 1


def _has_recovery_checkpoint(checkpoint_dir: Path) -> bool:
    """Return whether any committed recovery checkpoint exists under a run."""

    if not checkpoint_dir.is_dir():
        return False
    for path in checkpoint_dir.glob("global_step_*"):
        actor = path / "actor"
        if actor.is_dir() and (actor / "fsdp_config.json").is_file():
            return True
    return False


def last_metric_step(path: Path) -> int | None:
    if not path.is_file():
        return None
    last: int | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            raw = payload.get("step")
            if isinstance(raw, int) and not isinstance(raw, bool):
                last = raw
    return last


def clean_environment(
    plan: CellPlan,
    python_bin: Path,
    run_dir: Path,
    cuda_devices: str | None = None,
) -> dict[str, str]:
    environment = {key: value for key, value in os.environ.items() if key in SAFE_ENV_KEYS}
    environment.update(plan.env)
    if plan.protocol == SCIENTIFIC_PROTOCOL:
        # The offline contract: models were already resolved to absolute
        # snapshot paths at plan build time, so every official task must run
        # without any Hub network access.  Values already pinned in the matrix
        # env are preserved.
        environment.setdefault("HF_HUB_OFFLINE", "1")
        environment.setdefault("TRANSFORMERS_OFFLINE", "1")
        environment.setdefault("HF_DATASETS_OFFLINE", "1")
        environment.setdefault("GPU_TELEMETRY_INTERVAL", "1")
    environment.update(
        {
            "PYTHON_BIN": str(python_bin),
            "EXPERIMENT_NAME": f"ablation-{plan.cell_id}-{plan.fingerprint[:10]}",
            "EXPERIMENT_TAG": plan.cell_id,
            "RUN_DIR": str(run_dir),
            "METRICS_FILE": str(run_dir / "metrics.jsonl"),
            "DRY_RUN": "0",
        }
    )
    if cuda_devices is None:
        cuda_devices = resolve_gpu_allocation(
            [plan], os.environ.get("CUDA_VISIBLE_DEVICES")
        ).selector
    environment["CUDA_VISIBLE_DEVICES"] = cuda_devices
    return environment


def display_command(launcher: Path, env: Mapping[str, str]) -> str:
    scientific = {
        key: value
        for key, value in env.items()
        if key not in SAFE_ENV_KEYS and key not in {"DRY_RUN", "PYTHON_BIN"}
    }
    assignments = [f"{key}={shlex.quote(value)}" for key, value in sorted(scientific.items())]
    return " ".join([*assignments, "bash", shlex.quote(str(launcher))])


def suite_payload(
    registry: Mapping[str, Any],
    protocol: str,
    seed: int,
    source_hash: str,
    matrix_sha256: str,
    plans: Sequence[CellPlan],
) -> dict[str, Any]:
    return {
        "schema_version": int(registry.get("schema_version", 1)),
        "suite_id": registry["suite_id"],
        "protocol": protocol,
        "seed": seed,
        "created_at": utc_now(),
        "source_tree_sha256": source_hash,
        "matrix_sha256": matrix_sha256,
        "cells": [
            {
                "group_id": plan.group_id,
                "cell_id": plan.cell_id,
                "label": plan.label,
                "fidelity": plan.fidelity,
                "paper_location": plan.paper_location,
                "fingerprint": plan.fingerprint,
                "dataset": plan.dataset,
                "environment": plan.env,
                "scientific": plan.scientific,
                "models": plan.models,
                "comparability": comparability_for_plan(plan),
                "execution_state": "registered",
                "conclusion_state": "not_assessed",
            }
            for plan in plans
        ],
        "blocked_cells": registered_blockers(registry),
    }


def merge_suite_manifest(
    existing: Mapping[str, Any], current: Mapping[str, Any]
) -> dict[str, Any]:
    """Append newly selected cells without invalidating prior partial suite runs."""

    identity_keys = (
        "schema_version",
        "suite_id",
        "protocol",
        "seed",
        "source_tree_sha256",
        "matrix_sha256",
    )
    mismatched = [key for key in identity_keys if existing.get(key) != current.get(key)]
    if mismatched:
        raise RuntimeError(
            "existing suite manifest identity differs for " + ", ".join(mismatched)
        )

    def index_cells(
        payload: Mapping[str, Any], name: str
    ) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
        raw_cells = payload.get("cells")
        if not isinstance(raw_cells, list):
            raise RuntimeError(f"{name} suite manifest cells must be a list")
        ordered: list[dict[str, Any]] = []
        indexed: dict[str, dict[str, Any]] = {}
        for raw_cell in raw_cells:
            if not isinstance(raw_cell, dict):
                raise RuntimeError(f"{name} suite manifest contains a non-object cell")
            cell_id = raw_cell.get("cell_id")
            fingerprint = raw_cell.get("fingerprint")
            if not isinstance(cell_id, str) or not cell_id or not isinstance(fingerprint, str):
                raise RuntimeError(f"{name} suite manifest has an invalid cell identity")
            if cell_id in indexed:
                raise RuntimeError(f"{name} suite manifest repeats cell {cell_id}")
            cell = dict(raw_cell)
            ordered.append(cell)
            indexed[cell_id] = cell
        return ordered, indexed

    ordered, indexed = index_cells(existing, "existing")
    incoming, _ = index_cells(current, "current")
    added = False
    for cell in incoming:
        cell_id = cell["cell_id"]
        previous = indexed.get(cell_id)
        if previous is not None:
            if previous.get("fingerprint") != cell.get("fingerprint"):
                raise RuntimeError(f"existing suite cell {cell_id} has a different fingerprint")
            continue
        ordered.append(cell)
        indexed[cell_id] = cell
        added = True

    merged = dict(existing)
    merged["cells"] = ordered
    if current.get("blocked_cells"):
        if existing.get("blocked_cells", current["blocked_cells"]) != current["blocked_cells"]:
            raise RuntimeError("existing suite manifest blocked_cells differ")
        merged["blocked_cells"] = current["blocked_cells"]
    if added:
        merged["updated_at"] = utc_now()
    return merged


def _parse_gpu_count(plan: CellPlan) -> int:
    raw = plan.env.get("N_GPUS", "2")
    if not raw.isdigit() or int(raw) not in {2, 8}:
        raise ValueError(
            f"{plan.cell_id}: N_GPUS must be 2 or 8 for the formal runner, got {raw!r}"
        )
    return int(raw)


def _parse_cuda_devices(cuda_devices: str) -> tuple[str, ...]:
    parts = tuple(part.strip() for part in cuda_devices.split(","))
    if not parts or any(not part.isdigit() for part in parts):
        raise ValueError(
            "CUDA_VISIBLE_DEVICES must contain unique numeric physical GPU indices"
        )
    normalized = tuple(str(int(part)) for part in parts)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"CUDA_VISIBLE_DEVICES contains duplicate GPU indices: {cuda_devices!r}")
    return normalized


def resolve_gpu_allocation(
    plans: Sequence[CellPlan], cuda_devices: str | None
) -> GPUAllocation:
    """Resolve one uniform 2- or 8-GPU selector before any run-side mutation."""

    if not plans:
        raise ValueError("cannot allocate GPUs for an empty plan")
    counts_by_cell = {plan.cell_id: _parse_gpu_count(plan) for plan in plans}
    counts = set(counts_by_cell.values())
    if len(counts) != 1:
        detail = ", ".join(
            f"{cell_id}={count}" for cell_id, count in sorted(counts_by_cell.items())
        )
        raise ValueError(
            "mixed GPU-count plan is not executable in one invocation; "
            f"select the 2-GPU and 8-GPU cells separately ({detail})"
        )
    count = counts.pop()
    devices = (
        tuple(str(index) for index in range(count))
        if cuda_devices is None
        else _parse_cuda_devices(cuda_devices)
    )
    if len(devices) != count:
        raise ValueError(
            f"plan requires exactly {count} visible GPUs, got {len(devices)} in "
            f"CUDA_VISIBLE_DEVICES={cuda_devices!r}"
        )
    return GPUAllocation(count=count, devices=devices)


def acquire_gpu_lock(
    cuda_devices: str, *, lock_dir: Path = Path("/tmp")
) -> GPULocks:
    """Lock each physical GPU so every overlapping selector is mutually exclusive."""

    devices = _parse_cuda_devices(cuda_devices)
    lock_dir.mkdir(parents=True, exist_ok=True)
    handles: list[TextIO] = []
    try:
        for device in sorted(devices, key=int):
            lock_path = lock_dir / f"opd-ablation-gpu-{device}.lock"
            handle = lock_path.open("a+", encoding="utf-8")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                handle.close()
                raise RuntimeError(
                    f"physical GPU {device} from set {cuda_devices} is already locked"
                ) from exc
            handles.append(handle)
            handle.seek(0)
            handle.truncate()
            handle.write(
                json.dumps(
                    {
                        "pid": os.getpid(),
                        "physical_gpu": int(device),
                        "cuda_visible_devices": cuda_devices,
                        "at": utc_now(),
                    }
                )
            )
            handle.flush()
    except Exception:
        GPULocks(handles).close()
        raise
    return GPULocks(handles)


def print_plan(
    plans: Sequence[CellPlan],
    suite_root: Path,
    launcher: Path,
    python_bin: Path,
    blockers: Sequence[Mapping[str, Any]] = (),
) -> None:
    print(f"cells={len(plans)} suite_root={suite_root}")
    for index, plan in enumerate(plans, start=1):
        cell_root = suite_root / plan.cell_id
        status = read_json(cell_root / "status.json")
        state = (
            status.get("execution_state", status.get("state", "unknown"))
            if status
            else "registered"
        )
        preview_dir = cell_root / "attempt-0001"
        env = clean_environment(plan, python_bin, preview_dir)
        allocation = resolve_gpu_allocation(
            [plan], os.environ.get("CUDA_VISIBLE_DEVICES")
        )
        print(f"\n[{index:02d}] {plan.group_id}/{plan.cell_id} [{state}] {plan.label}")
        print(f"     fidelity={plan.fidelity} fingerprint={plan.fingerprint[:16]}")
        print(f"     data={plan.dataset['id']} rows={plan.dataset['rows']} sha256={plan.dataset['sha256'][:16]}")
        print(f"     gpus={allocation.count} CUDA_VISIBLE_DEVICES={allocation.selector}")
        if plan.protocol == SCIENTIFIC_PROTOCOL:
            print(
                f"     expected={plan.scientific['expected_final_step']} "
                f"milestones={plan.scientific['milestone_steps']}"
            )
        print("     " + display_command(launcher, env))
    for blocker in blockers:
        print(
            f"\n[BLOCKED] {blocker['group_id']}/{blocker['cell_id']} "
            f"[{blocker['execution_state']}] {blocker['reason']}"
        )


def run_cells(
    plans: Sequence[CellPlan],
    suite_root: Path,
    launcher: Path,
    python_bin: Path,
    keep_going: bool,
    retry_failed: bool,
    allocation: GPUAllocation,
    resume_interrupted: bool = False,
) -> int:
    if resolve_gpu_allocation(plans, allocation.selector) != allocation:
        raise ValueError("resolved GPU allocation does not match the selected cells")
    lock = acquire_gpu_lock(allocation.selector)
    failures = 0
    try:
        for index, plan in enumerate(plans, start=1):
            scientific = plan.protocol == SCIENTIFIC_PROTOCOL
            cell_root = suite_root / plan.cell_id
            status_path = cell_root / "status.json"
            previous = read_json(status_path)
            if previous and previous.get("fingerprint") != plan.fingerprint:
                raise RuntimeError(
                    f"{plan.cell_id}: existing status fingerprint differs; choose a new suite ID/root"
                )
            if previous and previous.get("state") == "completed":
                print(f"[{index}/{len(plans)}] skip completed {plan.cell_id}")
                continue
            if previous and previous.get("state") == "failed" and not retry_failed:
                if not (scientific and resume_interrupted):
                    raise RuntimeError(
                        f"{plan.cell_id}: previous attempt failed; pass --retry-failed to "
                        "create a fresh attempt or --resume-interrupted to restore from its "
                        "latest recovery checkpoint"
                    )

            attempt = attempt_number(cell_root)
            run_dir = cell_root / f"attempt-{attempt:04d}"
            environment = clean_environment(plan, python_bin, run_dir, allocation.selector)
            resumed = False
            if scientific:
                environment["TRAINING_MATRIX_ROOT"] = str(suite_root)
                if resume_interrupted and previous and previous.get("state") == "failed":
                    previous_run_dir = Path(previous["run_dir"])
                    previous_checkpoints = previous_run_dir / "checkpoints"
                    if previous_checkpoints.is_dir():
                        resumed = True
                        environment["RESUME_MODE"] = "auto"
                        environment["CHECKPOINT_DIR"] = str(previous_checkpoints)
                elif "RESUME_MODE" not in environment:
                    # The scientific matrix pins RESUME_MODE=auto by default;
                    # fall back to disable only when the matrix leaves it unset.
                    environment["RESUME_MODE"] = "disable"
            status = {
                "schema_version": SCHEMA_VERSION,
                "cell_id": plan.cell_id,
                "group_id": plan.group_id,
                "fingerprint": plan.fingerprint,
                "state": "running",
                "attempt": attempt,
                "run_dir": str(run_dir),
                "started_at": utc_now(),
                "resumed": resumed,
            }
            if scientific:
                status["execution_state"] = "running"
                status["conclusion_state"] = "not_assessed"
                contract = read_json(cell_root / "run_contract.json")
                if contract is not None:
                    status["conclusion_state"] = contract.get("conclusion_state", "not_assessed")
            atomic_json(status_path, status)
            print(f"[{index}/{len(plans)}] start {plan.cell_id}, attempt {attempt:04d}")
            started = time.monotonic()
            try:
                result = subprocess.run(["bash", str(launcher)], env=environment, check=False)
                exit_code = result.returncode
            except KeyboardInterrupt:
                exit_code = 130
            elapsed = time.monotonic() - started
            metrics_path = run_dir / "metrics.jsonl"
            final_step = last_metric_step(metrics_path)
            state = "completed" if exit_code == 0 and final_step is not None else "failed"
            status.update(
                {
                    "state": state,
                    "exit_code": exit_code,
                    "ended_at": utc_now(),
                    "elapsed_seconds": elapsed,
                    "metrics_file": str(metrics_path),
                    "last_metric_step": final_step,
                }
            )
            if scientific:
                if exit_code == 0:
                    execution_state = "training_complete"
                elif _has_recovery_checkpoint(run_dir / "checkpoints"):
                    execution_state = "interrupted_resumable"
                else:
                    execution_state = "infrastructure_failed"
                status["execution_state"] = execution_state
            atomic_json(status_path, status)
            print(f"[{index}/{len(plans)}] {state} {plan.cell_id}: exit={exit_code}, step={final_step}")
            if state != "completed":
                failures += 1
                if not keep_going:
                    break
    finally:
        lock.close()
    return 1 if failures else 0


def build_parser(repo_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run", "status"))
    parser.add_argument(
        "--matrix", type=Path, default=repo_root / "reproduce_4b/ablation_matrix.json"
    )
    parser.add_argument("--protocol", default="smoke")
    parser.add_argument("--group", action="append", default=[])
    parser.add_argument("--cell", action="append", default=[])
    parser.add_argument("--include-extensions", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-root", type=Path, default=repo_root / "artifacts/ablations")
    parser.add_argument("--python-bin", type=Path, default=repo_root / ".venv-opd/bin/python")
    parser.add_argument("--yes", action="store_true", help="Required for action=run.")
    parser.add_argument("--acknowledge-multi-day", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument(
        "--resume-interrupted",
        action="store_true",
        help=(
            "Scientific cells only: restore an interrupted attempt from its latest "
            "recovery checkpoint instead of starting a fresh attempt (RESUME_MODE=auto)."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = build_parser(repo_root)
    args = parser.parse_args(argv)
    try:
        matrix_path = args.matrix.expanduser().resolve()
        matrix_sha256 = sha256_file(matrix_path)
        registry = load_registry(matrix_path, repo_root)
        protocol = registry["protocols"].get(args.protocol)
        if protocol is None:
            raise ValueError(f"unknown protocol {args.protocol!r}")
        if args.action == "run" and not args.yes:
            raise ValueError("action=run requires --yes")
        if args.action == "run" and protocol.get("dangerous") and not args.acknowledge_multi_day:
            raise ValueError("paper protocol requires --acknowledge-multi-day")
        groups = select_groups(
            registry,
            set(args.group),
            set(args.cell),
            args.include_extensions,
        )
        datasets = validate_datasets(registry)
        source_hash = source_tree_hash(repo_root)
        plans = build_plans(
            registry,
            groups,
            args.protocol,
            args.seed,
            datasets,
            source_hash,
            matrix_sha256,
        )
        # Do not resolve the venv's ``python`` symlink: CPython discovers the
        # virtual environment from the invoked path, while the resolved base
        # interpreter silently escapes it.
        python_bin = absolute_path_preserve_symlinks(args.python_bin)
        launcher = repo_root / "reproduce_4b/run_opd_4b.sh"
        if not python_bin.is_file():
            raise FileNotFoundError(f"python interpreter does not exist: {python_bin}")
        suite_root = (
            args.run_root.expanduser().resolve()
            / registry["suite_id"]
            / args.protocol
            / f"seed-{args.seed}"
        )
        if args.action == "plan":
            print_plan(plans, suite_root, launcher, python_bin)
            return 0
        if args.action == "status":
            for plan in plans:
                status = read_json(suite_root / plan.cell_id / "status.json")
                state = status.get("state", "unknown") if status else "pending"
                detail = f" step={status.get('last_metric_step')}" if status else ""
                extra = ""
                if status and args.protocol == SCIENTIFIC_PROTOCOL:
                    execution = status.get("execution_state")
                    if execution:
                        extra = f" exec={execution}"
                print(
                    f"{plan.cell_id}\t{state}{detail}{extra}\t{plan.fingerprint[:16]}"
                )
            return 0

        # Resolve the complete selection before suite directories, manifests,
        # status files, lock files, or training subprocesses can be created.
        allocation = resolve_gpu_allocation(plans, os.environ.get("CUDA_VISIBLE_DEVICES"))

        suite_root.mkdir(parents=True, exist_ok=True)
        if args.protocol == SCIENTIFIC_PROTOCOL:
            for plan in plans:
                cell_root = suite_root / plan.cell_id
                cell_root.mkdir(parents=True, exist_ok=True)
                contract_for_cell(
                    registry,
                    plan,
                    cell_root,
                    args.seed,
                    source_hash,
                    matrix_sha256,
                    allocation,
                )
        manifest_path = suite_root / "suite_manifest.json"
        new_manifest = suite_payload(
            registry,
            args.protocol,
            args.seed,
            source_hash,
            matrix_sha256,
            plans,
        )
        existing_manifest = read_json(manifest_path)
        if existing_manifest:
            merged_manifest = merge_suite_manifest(existing_manifest, new_manifest)
            if merged_manifest != existing_manifest:
                atomic_json(manifest_path, merged_manifest)
        else:
            atomic_json(manifest_path, new_manifest)
        return run_cells(
            plans,
            suite_root,
            launcher,
            python_bin,
            args.keep_going,
            args.retry_failed,
            allocation,
            args.resume_interrupted,
        )
    except (FileNotFoundError, json.JSONDecodeError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
