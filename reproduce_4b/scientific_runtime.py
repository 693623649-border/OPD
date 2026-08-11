#!/usr/bin/env python3
"""Crash-safe runtime support for long-horizon OPD experiments.

The training process deliberately writes one immutable metric segment per
attempt.  This module records checkpoint lineage, archives registered
model-only milestones before rolling recovery checkpoints disappear, samples
physical GPU telemetry, and materializes a canonical metric stream after a
resume.  It has no dependency on VERL and is therefore also usable by audits
and unit tests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
ATTEMPT_RECORD = "attempt_runtime.json"
CANONICAL_METRICS = "metrics.jsonl"
METRICS_LINEAGE = "metrics_lineage.json"
CHECKPOINT_RE = re.compile(r"^global_step_([1-9][0-9]*)$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def read_object(path: Path, description: str = "JSON object") -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected {description}")
    return payload


def parse_steps(value: str | Sequence[int] | None) -> tuple[int, ...]:
    """Parse a comma/space separated, strictly positive milestone grid."""

    if value is None or value == "":
        return ()
    if isinstance(value, str):
        pieces = [piece for piece in re.split(r"[\s,]+", value.strip()) if piece]
        if any(not piece.isdigit() for piece in pieces):
            raise ValueError("milestone steps must be comma/space separated integers")
        values = [int(piece) for piece in pieces]
    else:
        values = list(value)
    if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in values):
        raise ValueError("milestone steps must be positive integers")
    if len(set(values)) != len(values):
        raise ValueError("milestone steps must be unique")
    return tuple(sorted(values))


def checkpoint_step(path: Path) -> int:
    match = CHECKPOINT_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"invalid checkpoint directory name: {path}")
    return int(match.group(1))


def _checkpoint_files(path: Path, require_full: bool) -> list[Path]:
    """Return the structurally required files for a complete FSDP checkpoint."""

    actor = path / "actor"
    fsdp_path = actor / "fsdp_config.json"
    if not fsdp_path.is_file():
        raise ValueError(f"checkpoint lacks actor/fsdp_config.json: {path}")
    config = read_object(fsdp_path, "FSDP config object")
    world_size = config.get("world_size")
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size <= 0:
        raise ValueError(f"checkpoint has invalid FSDP world_size: {path}")

    required = [fsdp_path]
    for prefix in ("model",):
        required.extend(
            actor / f"{prefix}_world_size_{world_size}_rank_{rank}.pt"
            for rank in range(world_size)
        )
    if require_full:
        for prefix in ("optim", "extra_state"):
            required.extend(
                actor / f"{prefix}_world_size_{world_size}_rank_{rank}.pt"
                for rank in range(world_size)
            )
        required.append(path / "data.pt")
    missing = [str(item.relative_to(path)) for item in required if not item.is_file()]
    empty = [str(item.relative_to(path)) for item in required if item.is_file() and item.stat().st_size <= 0]
    if missing or empty:
        detail = []
        if missing:
            detail.append("missing=" + ",".join(missing))
        if empty:
            detail.append("empty=" + ",".join(empty))
        raise ValueError(f"incomplete checkpoint {path}: {'; '.join(detail)}")
    return required


def checkpoint_manifest(path: Path, *, require_full: bool, hash_files: bool = True) -> dict[str, Any]:
    required = _checkpoint_files(path, require_full=require_full)
    files: list[dict[str, Any]] = []
    for item in sorted(required):
        entry: dict[str, Any] = {
            "path": str(item.relative_to(path)),
            "size": item.stat().st_size,
        }
        if hash_files:
            entry["sha256"] = sha256_file(item)
        files.append(entry)
    identity = {
        "step": checkpoint_step(path),
        "kind": "full_recovery" if require_full else "model_only",
        "files": files,
    }
    return {**identity, "manifest_sha256": canonical_hash(identity)}


def recovery_checkpoints(checkpoint_dir: Path) -> list[Path]:
    if not checkpoint_dir.is_dir():
        return []
    candidates: list[tuple[int, Path]] = []
    for path in checkpoint_dir.iterdir():
        match = CHECKPOINT_RE.fullmatch(path.name)
        if path.is_dir() and match:
            candidates.append((int(match.group(1)), path))
    return [path for _, path in sorted(candidates)]


def committed_checkpoint_step(checkpoint_dir: Path) -> int | None:
    """Read VERL's commit marker, which is written after every checkpoint."""

    tracker = checkpoint_dir / "latest_checkpointed_iteration.txt"
    if not tracker.is_file():
        return None
    raw = tracker.read_text(encoding="utf-8").strip()
    if not raw.isdigit() or int(raw) <= 0:
        raise ValueError(f"invalid checkpoint tracker: {tracker}")
    return int(raw)


def latest_complete_recovery(checkpoint_dir: Path) -> tuple[Path | None, list[str]]:
    committed = committed_checkpoint_step(checkpoint_dir)
    candidates = recovery_checkpoints(checkpoint_dir)
    if committed is None:
        rejected = [f"{path}: no committed checkpoint tracker" for path in candidates]
        return None, rejected
    selected = checkpoint_dir / f"global_step_{committed}"
    rejected = [
        f"{path}: newer than committed tracker step {committed}"
        for path in candidates
        if checkpoint_step(path) > committed
    ]
    try:
        _checkpoint_files(selected, require_full=True)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        rejected.append(f"{selected}: committed checkpoint is invalid: {exc}")
        return None, rejected
    return selected, rejected


def _milestone_source_files(checkpoint: Path) -> list[Path]:
    _checkpoint_files(checkpoint, require_full=True)
    actor = checkpoint / "actor"
    paths = [actor / "fsdp_config.json", *sorted(actor.glob("model_world_size_*_rank_*.pt"))]
    huggingface = actor / "huggingface"
    if huggingface.is_dir():
        paths.extend(path for path in sorted(huggingface.rglob("*")) if path.is_file())
    actor_hf = checkpoint / "actor_hf"
    if actor_hf.is_dir():
        paths.extend(path for path in sorted(actor_hf.rglob("*")) if path.is_file())
    return paths


def validate_milestone(
    path: Path, expected_step: int | None = None, *, check_loadable: bool = False
) -> dict[str, Any]:
    manifest_path = path / "checkpoint_manifest.json"
    manifest = read_object(manifest_path, "checkpoint milestone manifest")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{manifest_path}: unsupported schema version")
    step = checkpoint_step(path)
    if manifest.get("step") != step or (expected_step is not None and step != expected_step):
        raise ValueError(f"{manifest_path}: milestone step mismatch")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"{manifest_path}: milestone file manifest is empty")
    identity_files: list[dict[str, Any]] = []
    for raw in files:
        if not isinstance(raw, dict) or not isinstance(raw.get("path"), str):
            raise ValueError(f"{manifest_path}: malformed file entry")
        relative = Path(raw["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"{manifest_path}: unsafe file path {relative}")
        target = path / relative
        if not target.is_file():
            raise ValueError(f"{manifest_path}: missing {relative}")
        observed = {"path": raw["path"], "size": target.stat().st_size, "sha256": sha256_file(target)}
        if raw != observed:
            raise ValueError(f"{manifest_path}: hash/size mismatch for {relative}")
        identity_files.append(observed)
    identity = {"step": step, "kind": "model_only", "files": identity_files}
    if manifest.get("manifest_sha256") != canonical_hash(identity):
        raise ValueError(f"{manifest_path}: manifest identity hash mismatch")
    _checkpoint_files(path, require_full=False)
    if check_loadable:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required to load-check milestone shards") from exc
        for shard in sorted((path / "actor").glob("model_world_size_*_rank_*.pt")):
            try:
                torch.load(shard, map_location="meta", weights_only=False)
            except Exception as exc:
                raise ValueError(f"{manifest_path}: model shard is not loadable: {shard.name}") from exc
    return manifest


def archive_milestone(checkpoint: Path, milestone_dir: Path) -> Path:
    """Atomically archive only model state from one complete recovery checkpoint."""

    step = checkpoint_step(checkpoint)
    destination = milestone_dir / checkpoint.name
    if destination.exists():
        validate_milestone(destination, step)
        return destination

    source_files = _milestone_source_files(checkpoint)
    temporary = milestone_dir / f".{checkpoint.name}.tmp-{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        entries: list[dict[str, Any]] = []
        for source in source_files:
            relative = source.relative_to(checkpoint)
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            entries.append(
                {"path": str(relative), "size": target.stat().st_size, "sha256": sha256_file(target)}
            )
        entries.sort(key=lambda item: item["path"])
        identity = {"step": step, "kind": "model_only", "files": entries}
        source_identity = checkpoint_manifest(checkpoint, require_full=True, hash_files=False)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            **identity,
            "manifest_sha256": canonical_hash(identity),
            "archived_at": utc_now(),
            "source_checkpoint": str(checkpoint.resolve()),
            "source_checkpoint_structure_sha256": source_identity["manifest_sha256"],
        }
        atomic_json(temporary / "checkpoint_manifest.json", manifest)
        milestone_dir.mkdir(parents=True, exist_ok=True)
        temporary.replace(destination)
        validate_milestone(destination, step)
        return destination
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def archive_available_milestones(
    checkpoint_dir: Path,
    milestone_dir: Path,
    milestone_steps: Sequence[int],
    skip_steps: Iterable[int] = (),
) -> list[int]:
    requested = set(milestone_steps)
    skipped = set(skip_steps)
    committed = committed_checkpoint_step(checkpoint_dir)
    if committed is None:
        return []
    archived: list[int] = []
    for checkpoint in recovery_checkpoints(checkpoint_dir):
        step = checkpoint_step(checkpoint)
        if step not in requested or step in skipped or step > committed:
            continue
        try:
            _checkpoint_files(checkpoint, require_full=True)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        archive_milestone(checkpoint, milestone_dir)
        archived.append(step)
    return archived


def prune_recovery_checkpoints(
    checkpoint_dir: Path,
    milestone_dir: Path,
    milestone_steps: Sequence[int],
    retention: int,
) -> list[int]:
    if retention <= 0:
        return []
    committed = committed_checkpoint_step(checkpoint_dir)
    if committed is None:
        return []
    complete: list[Path] = []
    for path in recovery_checkpoints(checkpoint_dir):
        if checkpoint_step(path) > committed:
            continue
        try:
            _checkpoint_files(path, require_full=True)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        complete.append(path)
    removable = complete[:-retention]
    removed: list[int] = []
    requested = set(milestone_steps)
    for path in removable:
        step = checkpoint_step(path)
        if step in requested:
            validate_milestone(milestone_dir / path.name, step)
        shutil.rmtree(path)
        removed.append(step)
    return removed


def prepare_attempt(
    run_dir: Path,
    checkpoint_dir: Path,
    milestone_dir: Path,
    metrics_file: Path,
    resume_mode: str,
    expected_final_step: int | None,
    milestone_steps: Sequence[int],
    recovery_retention: int = 2,
) -> dict[str, Any]:
    if resume_mode not in {"disable", "auto"}:
        raise ValueError("resume mode must be disable or auto")
    if expected_final_step is not None and expected_final_step <= 0:
        raise ValueError("expected final step must be positive")
    steps = parse_steps(milestone_steps)
    if expected_final_step is not None and any(step > expected_final_step for step in steps):
        raise ValueError("milestone step exceeds expected final step")
    if recovery_retention < 1:
        raise ValueError("recovery retention must be positive")
    record_path = run_dir / ATTEMPT_RECORD
    if record_path.exists():
        raise FileExistsError(f"attempt runtime record already exists: {record_path}")
    if metrics_file.exists():
        raise FileExistsError(f"metrics segment already exists: {metrics_file}")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    milestone_dir.mkdir(parents=True, exist_ok=True)
    archive_available_milestones(checkpoint_dir, milestone_dir, steps)
    latest, rejected = latest_complete_recovery(checkpoint_dir)
    if resume_mode == "disable" and recovery_checkpoints(checkpoint_dir):
        raise RuntimeError(
            f"resume is disabled but checkpoint directory is not empty: {checkpoint_dir}"
        )
    if resume_mode == "auto" and recovery_checkpoints(checkpoint_dir) and latest is None:
        raise RuntimeError(
            "checkpoint state exists but no structurally valid committed recovery point was found: "
            + "; ".join(rejected)
        )
    resume: dict[str, Any] = {
        "mode": resume_mode,
        "checkpoint_step": 0,
        "checkpoint_path": None,
        "checkpoint_manifest_sha256": None,
        "rejected_newer_checkpoints": rejected,
    }
    if resume_mode == "auto" and latest is not None:
        manifest = checkpoint_manifest(latest, require_full=True, hash_files=True)
        resume.update(
            {
                "checkpoint_step": manifest["step"],
                "checkpoint_path": str(latest.resolve()),
                "checkpoint_manifest_sha256": manifest["manifest_sha256"],
                "checkpoint_files": manifest["files"],
            }
        )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "opd_scientific_attempt",
        "started_at": utc_now(),
        "run_dir": str(run_dir.resolve()),
        "metrics_segment": str(metrics_file.resolve()),
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "milestone_dir": str(milestone_dir.resolve()),
        "expected_final_step": expected_final_step,
        "milestone_steps": list(steps),
        "recovery_retention": recovery_retention,
        "resume": resume,
        "state": "running",
    }
    atomic_json(record_path, payload)
    return payload


def _finite_tree(value: Any, location: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite metric value at {location}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _finite_tree(item, f"{location}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _finite_tree(item, f"{location}.{key}")
        return
    raise ValueError(f"unsupported metric value at {location}: {type(value).__name__}")


def read_metric_segment(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: metric row must be an object")
            step = row.get("step")
            if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
                raise ValueError(f"{path}:{line_number}: metric step must be a positive integer")
            if step in seen:
                raise ValueError(f"{path}:{line_number}: duplicate metric step {step}")
            seen.add(step)
            _finite_tree(row, f"{path}:{line_number}")
            rows.append(row)
    if [row["step"] for row in rows] != sorted(seen):
        raise ValueError(f"{path}: metric steps must be strictly increasing")
    return rows


def finalize_attempt(run_dir: Path, exit_code: int) -> dict[str, Any]:
    path = run_dir / ATTEMPT_RECORD
    payload = read_object(path, "attempt runtime record")
    metrics_path = Path(str(payload["metrics_segment"]))
    rows = read_metric_segment(metrics_path) if metrics_path.is_file() else []
    payload.update(
        {
            "state": "finished" if exit_code == 0 else "interrupted",
            "exit_code": exit_code,
            "ended_at": utc_now(),
            "last_metric_step": rows[-1]["step"] if rows else None,
            "metric_rows": len(rows),
            "metrics_segment_sha256": sha256_file(metrics_path) if metrics_path.is_file() else None,
        }
    )
    atomic_json(path, payload)
    return payload


def attempt_records(cell_root: Path) -> list[tuple[Path, dict[str, Any]]]:
    records: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(cell_root.glob(f"attempt-*/{ATTEMPT_RECORD}")):
        payload = read_object(path, "attempt runtime record")
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"{path}: unsupported attempt runtime schema")
        records.append((path, payload))
    return records


def canonicalize_metrics(cell_root: Path) -> dict[str, Any]:
    records = attempt_records(cell_root)
    if not records:
        raise ValueError(f"no attempt runtime records under {cell_root}")
    canonical: dict[int, dict[str, Any]] = {}
    segments: list[dict[str, Any]] = []
    abandoned: list[dict[str, Any]] = []
    previous_resume = -1
    for index, (record_path, record) in enumerate(records, start=1):
        resume = record.get("resume")
        if not isinstance(resume, dict):
            raise ValueError(f"{record_path}: missing resume lineage")
        resume_step = resume.get("checkpoint_step")
        if isinstance(resume_step, bool) or not isinstance(resume_step, int) or resume_step < 0:
            raise ValueError(f"{record_path}: invalid resume checkpoint step")
        if index == 1 and resume_step != 0:
            raise ValueError(f"{record_path}: first local attempt resumes at step {resume_step}")
        if index > 1 and resume_step < previous_resume:
            raise ValueError(f"{record_path}: resume lineage moves backwards across attempts")
        rolled_back = sorted(step for step in canonical if step > resume_step)
        for step in rolled_back:
            abandoned.append(
                {
                    "step": step,
                    "reason": "superseded_after_checkpoint_rollback",
                    "superseded_by_attempt": record_path.parent.name,
                }
            )
            del canonical[step]

        metrics_path = Path(str(record.get("metrics_segment", "")))
        rows = read_metric_segment(metrics_path) if metrics_path.is_file() else []
        replayed = [row["step"] for row in rows if row["step"] <= resume_step]
        new_rows = [row for row in rows if row["step"] > resume_step]
        if new_rows:
            expected = list(range(resume_step + 1, new_rows[-1]["step"] + 1))
            observed = [row["step"] for row in new_rows]
            if observed != expected:
                raise ValueError(
                    f"{metrics_path}: segment after resume step {resume_step} is not continuous"
                )
        for row in new_rows:
            canonical[row["step"]] = row
        segment_sha = sha256_file(metrics_path) if metrics_path.is_file() else None
        segments.append(
            {
                "attempt": record_path.parent.name,
                "record": str(record_path.resolve()),
                "metrics_segment": str(metrics_path.resolve()),
                "metrics_segment_sha256": segment_sha,
                "resume_checkpoint_step": resume_step,
                "resume_checkpoint_manifest_sha256": resume.get("checkpoint_manifest_sha256"),
                "replayed_rows_ignored": replayed,
                "accepted_steps": [row["step"] for row in new_rows],
            }
        )
        previous_resume = resume_step

    ordered = [canonical[step] for step in sorted(canonical)]
    metrics_text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in ordered)
    canonical_path = cell_root / CANONICAL_METRICS
    atomic_text(canonical_path, metrics_text)
    lineage_identity = {
        "segments": segments,
        "abandoned_rows": abandoned,
        "canonical_steps": [row["step"] for row in ordered],
        "canonical_metrics_sha256": sha256_file(canonical_path),
    }
    lineage = {
        "schema_version": SCHEMA_VERSION,
        "kind": "opd_canonical_metrics_lineage",
        "generated_at": utc_now(),
        **lineage_identity,
        "lineage_sha256": canonical_hash(lineage_identity),
    }
    atomic_json(cell_root / METRICS_LINEAGE, lineage)
    return lineage


def finalize_training(
    cell_root: Path, expected_final_step: int, milestone_steps: Sequence[int]
) -> dict[str, Any]:
    if expected_final_step <= 0:
        raise ValueError("expected final step must be positive")
    steps = parse_steps(milestone_steps)
    lineage = canonicalize_metrics(cell_root)
    observed = lineage["canonical_steps"]
    expected = list(range(1, expected_final_step + 1))
    issues: list[str] = []
    if observed != expected:
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        issues.append(
            f"canonical metric grid mismatch: last={observed[-1] if observed else None}, "
            f"missing={missing[:20]}, extra={extra[:20]}"
        )
    milestone_dir = cell_root / "milestones"
    milestone_manifests: dict[str, str] = {}
    milestone_artifacts: list[dict[str, Any]] = []
    for step in steps:
        try:
            milestone_path = milestone_dir / f"global_step_{step}"
            manifest = validate_milestone(milestone_path, step, check_loadable=True)
            milestone_manifests[str(step)] = str(manifest["manifest_sha256"])
            milestone_artifacts.append(
                {
                    "step": step,
                    "path": str(milestone_path.resolve()),
                    "inventory_sha256": str(manifest["manifest_sha256"]),
                    "loadable": True,
                }
            )
        except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
            issues.append(f"milestone {step} invalid: {exc}")
            milestone_artifacts.append(
                {
                    "step": step,
                    "path": str((milestone_dir / f"global_step_{step}").resolve()),
                    "inventory_sha256": None,
                    "loadable": False,
                }
            )
    canonical_path = cell_root / CANONICAL_METRICS
    complete = not issues
    return {
        "completion": complete,
        "training_complete": complete,
        "expected_final_step": expected_final_step,
        "last_metric_step": observed[-1] if observed else None,
        "canonical_metrics": str(canonical_path.resolve()),
        "canonical_metrics_file": str(canonical_path.resolve()),
        "metrics_sha256": sha256_file(canonical_path),
        "metrics_lineage": str((cell_root / METRICS_LINEAGE).resolve()),
        "milestone_manifests": milestone_manifests,
        "milestone_artifacts": milestone_artifacts,
        "issues": issues,
    }


def _parse_cuda_devices(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    pieces = tuple(piece.strip() for piece in value.split(","))
    if not pieces or any(not piece.isdigit() for piece in pieces):
        raise ValueError("CUDA_VISIBLE_DEVICES must contain numeric physical GPU indices")
    normalized = tuple(str(int(piece)) for piece in pieces)
    if len(set(normalized)) != len(normalized):
        raise ValueError("CUDA_VISIBLE_DEVICES contains duplicate physical GPU indices")
    return normalized


def sample_gpus(cuda_devices: str | None = None) -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise RuntimeError("nvidia-smi is unavailable")
    devices = _parse_cuda_devices(cuda_devices)
    query = (
        "index,uuid,name,memory.total,memory.used,memory.free,"
        "utilization.gpu,temperature.gpu,power.draw"
    )
    command = [executable]
    if devices is not None:
        command.append("--id=" + ",".join(devices))
    command.extend([f"--query-gpu={query}", "--format=csv,noheader,nounits"])
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()}")
    rows: list[dict[str, Any]] = []
    reader = csv.reader(result.stdout.splitlines(), skipinitialspace=True)
    for line_number, fields in enumerate(reader, start=1):
        if len(fields) != 9:
            raise ValueError(f"unexpected nvidia-smi row {line_number}: {fields!r}")
        try:
            rows.append(
                {
                    "physical_index": int(fields[0]),
                    "uuid": fields[1],
                    "name": fields[2],
                    "memory_total_mib": float(fields[3]),
                    "memory_used_mib": float(fields[4]),
                    "memory_free_mib": float(fields[5]),
                    "utilization_gpu_percent": float(fields[6]),
                    "temperature_c": float(fields[7]),
                    "power_draw_w": float(fields[8]),
                }
            )
        except ValueError as exc:
            raise ValueError(f"non-numeric nvidia-smi row {line_number}: {fields!r}") from exc
    if devices is not None and {str(row["physical_index"]) for row in rows} != set(devices):
        raise RuntimeError("nvidia-smi did not return exactly the requested physical GPUs")
    return {"observed_at": utc_now(), "gpus": rows}


def _last_metric_step(path: Path) -> int | None:
    if not path.is_file():
        return None
    rows = read_metric_segment(path)
    return rows[-1]["step"] if rows else None


def _nested_value(payload: Any, keys: Sequence[str]) -> Any:
    if isinstance(payload, dict):
        for key in keys:
            value = payload.get(key)
            if value is not None:
                return value
        for value in payload.values():
            found = _nested_value(value, keys)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _nested_value(value, keys)
            if found is not None:
                return found
    return None


def _matrix_row(cell_root: Path, status: Mapping[str, Any], now: float) -> dict[str, Any]:
    cell_id = str(status.get("cell_id", cell_root.name))
    run_dir_raw = status.get("run_dir")
    run_dir = Path(run_dir_raw) if isinstance(run_dir_raw, str) else None
    candidates = [cell_root / CANONICAL_METRICS]
    metrics_raw = status.get("metrics_file")
    if isinstance(metrics_raw, str):
        candidates.append(Path(metrics_raw))
    if run_dir is not None:
        candidates.append(run_dir / "metrics.jsonl")
    metrics_path = next((path for path in candidates if path.is_file()), None)
    step = _last_metric_step(metrics_path) if metrics_path is not None else None

    contract_path = cell_root / "run_contract.json"
    contract = read_object(contract_path, "run contract") if contract_path.is_file() else {}
    raw_expected = status.get("expected_final_step")
    if raw_expected is None:
        raw_expected = _nested_value(contract, ("expected_final_step", "TOTAL_TRAINING_STEPS"))
    try:
        expected = int(raw_expected) if raw_expected is not None else None
    except (TypeError, ValueError):
        expected = None

    latest_runtime: dict[str, Any] = {}
    segment_step: int | None = None
    runtime_paths = sorted(cell_root.glob(f"attempt-*/{ATTEMPT_RECORD}"))
    if runtime_paths:
        latest_runtime = read_object(runtime_paths[-1], "attempt runtime record")
        segment = Path(str(latest_runtime.get("metrics_segment", "")))
        segment_step = _last_metric_step(segment) if segment.is_file() else None
    resume = latest_runtime.get("resume") if isinstance(latest_runtime.get("resume"), dict) else {}
    resume_step = resume.get("checkpoint_step", 0)
    # During a resumed attempt the previous canonical file can legitimately
    # extend beyond the selected recovery point.  Show the active segment (or
    # the recovery step before its first row), never those abandoned rows.
    if latest_runtime.get("state") == "running":
        step = segment_step if segment_step is not None else int(resume_step or 0)
    elif segment_step is not None and step is None:
        step = segment_step
    eta_seconds: float | None = None
    started_raw = latest_runtime.get("started_at") or status.get("started_at")
    if isinstance(started_raw, str) and step is not None and expected is not None and step < expected:
        try:
            started = datetime.fromisoformat(started_raw).timestamp()
            progressed = step - int(resume_step or 0)
            elapsed = now - started
            if progressed > 0 and elapsed > 0:
                eta_seconds = elapsed / progressed * (expected - step)
        except (TypeError, ValueError):
            pass

    checkpoint_raw = latest_runtime.get("checkpoint_dir") or status.get("checkpoint_dir")
    latest_checkpoint: int | None = None
    if isinstance(checkpoint_raw, str):
        latest, _ = latest_complete_recovery(Path(checkpoint_raw))
        latest_checkpoint = checkpoint_step(latest) if latest is not None else None
    telemetry: dict[str, Any] = {}
    if run_dir is not None and (run_dir / "gpu_telemetry_summary.json").is_file():
        telemetry = read_object(run_dir / "gpu_telemetry_summary.json", "GPU telemetry summary")
    peaks = telemetry.get("peak_memory_used_mib_by_gpu", {})
    current = telemetry.get("latest_memory_used_mib_by_gpu", {})
    progress = step / expected if step is not None and expected else None
    return {
        "cell_id": cell_id,
        "group_id": status.get("group_id"),
        "execution_state": status.get("execution_state", status.get("state", "registered")),
        "conclusion_state": status.get("conclusion_state", "not_assessed"),
        "comparability": status.get("comparability"),
        "attempt": status.get("attempt"),
        "step": step,
        "expected_final_step": expected,
        "progress": progress,
        "eta_seconds": eta_seconds,
        "latest_checkpoint_step": latest_checkpoint,
        "gpu_peak_memory_used_mib": peaks,
        "gpu_latest_memory_used_mib": current,
        "updated_at": status.get("updated_at", status.get("ended_at", status.get("started_at"))),
    }


def render_training_matrix(suite_root: Path) -> dict[str, Any]:
    now = time.time()
    rows: list[dict[str, Any]] = []
    for status_path in sorted(suite_root.glob("*/status.json")):
        status = read_object(status_path, "cell status")
        rows.append(_matrix_row(status_path.parent, status, now))
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "opd_live_training_matrix",
        "generated_at": utc_now(),
        "suite_root": str(suite_root.resolve()),
        "cells": rows,
    }
    atomic_json(suite_root / "training_matrix.json", payload)

    fields = [
        "cell_id",
        "group_id",
        "execution_state",
        "conclusion_state",
        "comparability",
        "attempt",
        "step",
        "expected_final_step",
        "progress",
        "eta_seconds",
        "latest_checkpoint_step",
        "gpu_peak_memory_used_mib",
        "gpu_latest_memory_used_mib",
        "updated_at",
    ]
    csv_path = suite_root / "training_matrix.csv"
    temporary = csv_path.with_name(f".{csv_path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row[key], ensure_ascii=False, sort_keys=True)
                    if isinstance(row.get(key), (dict, list))
                    else row.get(key)
                    for key in fields
                }
            )
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(csv_path)

    def display(value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.3f}"
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False, sort_keys=True).replace("|", "\\|")
        return str(value).replace("|", "\\|")

    headings = (
        "cell",
        "execution",
        "conclusion",
        "step/N",
        "progress",
        "ETA(h)",
        "checkpoint",
        "peak GPU MiB",
    )
    lines = [
        "# Live training matrix",
        "",
        f"Updated: {payload['generated_at']}",
        "",
        "| " + " | ".join(headings) + " |",
        "| " + " | ".join("---" for _ in headings) + " |",
    ]
    for row in rows:
        step_n = f"{display(row['step'])}/{display(row['expected_final_step'])}"
        progress_text = (
            f"{100 * row['progress']:.1f}%" if isinstance(row["progress"], float) else "-"
        )
        eta = row["eta_seconds"] / 3600 if isinstance(row["eta_seconds"], float) else None
        values = (
            row["cell_id"],
            row["execution_state"],
            row["conclusion_state"],
            step_n,
            progress_text,
            eta,
            row["latest_checkpoint_step"],
            row["gpu_peak_memory_used_mib"],
        )
        lines.append("| " + " | ".join(display(value) for value in values) + " |")
    atomic_text(suite_root / "training_matrix.md", "\n".join(lines) + "\n")
    return payload


def monitor_runtime(
    run_dir: Path,
    checkpoint_dir: Path,
    milestone_dir: Path,
    milestone_steps: Sequence[int],
    recovery_retention: int,
    telemetry_interval: float,
    cuda_devices: str | None,
    stop_file: Path,
    training_matrix_root: Path | None = None,
) -> dict[str, Any]:
    if telemetry_interval < 0 or telemetry_interval > 60:
        raise ValueError("GPU telemetry interval must be between 0 and 60 seconds")
    if recovery_retention < 1:
        raise ValueError("recovery retention must be positive")
    poll_interval = telemetry_interval if telemetry_interval > 0 else 1.0
    telemetry_path = run_dir / "gpu_telemetry.jsonl"
    if telemetry_interval > 0 and telemetry_path.exists():
        raise FileExistsError(f"GPU telemetry already exists: {telemetry_path}")
    telemetry_handle = (
        telemetry_path.open("x", encoding="utf-8") if telemetry_interval > 0 else None
    )
    peaks: dict[str, float] = {}
    latest_used: dict[str, float] = {}
    samples = 0
    archived: set[int] = set()
    removed: list[int] = []
    started = time.monotonic()
    monitor_status_path = run_dir / "runtime_monitor.json"
    try:
        while True:
            if telemetry_interval > 0:
                sample = sample_gpus(cuda_devices)
                sample["elapsed_seconds"] = time.monotonic() - started
                assert telemetry_handle is not None
                telemetry_handle.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")
                telemetry_handle.flush()
                samples += 1
                for gpu in sample["gpus"]:
                    key = str(gpu["physical_index"])
                    used = float(gpu["memory_used_mib"])
                    latest_used[key] = used
                    peaks[key] = max(peaks.get(key, 0.0), used)
                atomic_json(
                    run_dir / "gpu_telemetry_summary.json",
                    {
                        "schema_version": SCHEMA_VERSION,
                        "kind": "physical_gpu_telemetry_summary",
                        "updated_at": utc_now(),
                        "sample_count": samples,
                        "interval_seconds": telemetry_interval,
                        "peak_memory_used_mib_by_gpu": peaks,
                        "latest_memory_used_mib_by_gpu": latest_used,
                    },
                )
            archived.update(
                archive_available_milestones(
                    checkpoint_dir, milestone_dir, milestone_steps, skip_steps=archived
                )
            )
            removed.extend(
                prune_recovery_checkpoints(
                    checkpoint_dir,
                    milestone_dir,
                    milestone_steps,
                    recovery_retention,
                )
            )
            if training_matrix_root is not None and training_matrix_root.is_dir():
                render_training_matrix(training_matrix_root)
            atomic_json(
                monitor_status_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "state": "running",
                    "updated_at": utc_now(),
                    "telemetry_samples": samples,
                    "archived_milestones": sorted(archived),
                    "pruned_recovery_steps": sorted(set(removed)),
                },
            )
            if stop_file.exists():
                break
            time.sleep(poll_interval)
        archived.update(
            archive_available_milestones(
                checkpoint_dir, milestone_dir, milestone_steps, skip_steps=archived
            )
        )
        if training_matrix_root is not None and training_matrix_root.is_dir():
            render_training_matrix(training_matrix_root)
        result = {
            "schema_version": SCHEMA_VERSION,
            "state": "completed",
            "ended_at": utc_now(),
            "telemetry_samples": samples,
            "archived_milestones": sorted(archived),
            "pruned_recovery_steps": sorted(set(removed)),
        }
        atomic_json(monitor_status_path, result)
        return result
    except Exception as exc:
        atomic_json(
            monitor_status_path,
            {
                "schema_version": SCHEMA_VERSION,
                "state": "failed",
                "ended_at": utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
                "telemetry_samples": samples,
                "archived_milestones": sorted(archived),
            },
        )
        raise
    finally:
        if telemetry_handle is not None:
            telemetry_handle.close()


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)

    prepare = subparsers.add_parser("prepare-attempt")
    prepare.add_argument("--run-dir", type=Path, required=True)
    prepare.add_argument("--checkpoint-dir", type=Path, required=True)
    prepare.add_argument("--milestone-dir", type=Path, required=True)
    prepare.add_argument("--metrics-file", type=Path, required=True)
    prepare.add_argument("--resume-mode", choices=("disable", "auto"), required=True)
    prepare.add_argument("--expected-final-step", type=_positive_int)
    prepare.add_argument("--milestone-steps", default="")
    prepare.add_argument("--recovery-retention", type=_positive_int, default=2)

    finish_attempt = subparsers.add_parser("finalize-attempt")
    finish_attempt.add_argument("--run-dir", type=Path, required=True)
    finish_attempt.add_argument("--exit-code", type=int, required=True)

    finish_training = subparsers.add_parser("finalize-training")
    finish_training.add_argument("--cell-root", type=Path, required=True)
    finish_training.add_argument("--expected-final-step", type=_positive_int, required=True)
    finish_training.add_argument("--milestone-steps", default="")
    finish_training.add_argument("--allow-incomplete", action="store_true")

    monitor = subparsers.add_parser("monitor")
    monitor.add_argument("--run-dir", type=Path, required=True)
    monitor.add_argument("--checkpoint-dir", type=Path, required=True)
    monitor.add_argument("--milestone-dir", type=Path, required=True)
    monitor.add_argument("--milestone-steps", default="")
    monitor.add_argument("--recovery-retention", type=_positive_int, default=2)
    monitor.add_argument("--telemetry-interval", type=float, default=1.0)
    monitor.add_argument("--cuda-devices")
    monitor.add_argument("--stop-file", type=Path, required=True)
    monitor.add_argument("--training-matrix-root", type=Path)

    matrix = subparsers.add_parser("render-training-matrix")
    matrix.add_argument("--suite-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.action == "prepare-attempt":
            result = prepare_attempt(
                args.run_dir,
                args.checkpoint_dir,
                args.milestone_dir,
                args.metrics_file,
                args.resume_mode,
                args.expected_final_step,
                parse_steps(args.milestone_steps),
                args.recovery_retention,
            )
        elif args.action == "finalize-attempt":
            result = finalize_attempt(args.run_dir, args.exit_code)
        elif args.action == "finalize-training":
            result = finalize_training(
                args.cell_root, args.expected_final_step, parse_steps(args.milestone_steps)
            )
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
            return 0 if result["training_complete"] or args.allow_incomplete else 1
        elif args.action == "monitor":
            result = monitor_runtime(
                args.run_dir,
                args.checkpoint_dir,
                args.milestone_dir,
                parse_steps(args.milestone_steps),
                args.recovery_retention,
                args.telemetry_interval,
                args.cuda_devices,
                args.stop_file,
                args.training_matrix_root,
            )
        else:
            result = render_training_matrix(args.suite_root)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except (FileNotFoundError, json.JSONDecodeError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
