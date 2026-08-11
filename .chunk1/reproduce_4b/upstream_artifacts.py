#!/usr/bin/env python3
"""Validate immutable Table 1/Table 3 upstream stage artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = 1
UPSTREAM_SUITE_IDS = {
    "rethinking-opd-upstream-v1",
    "rethinking-opd-upstream-v2",
    "rethinking-opd-upstream-v3",
}
UPSTREAM_STAGES = ("grpo-teacher", "cold-start-rollout", "cold-start-sft")
PROTOCOL_RANK = {"smoke": 1, "calibration": 2, "paper": 3}


def _read_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _same_path(raw: Any, expected: Path) -> bool:
    return (
        isinstance(raw, str)
        and bool(raw)
        and Path(raw).expanduser().resolve() == expected.expanduser().resolve()
    )


def _fingerprint_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    environment = manifest.get("environment")
    source = manifest.get("source")
    if not isinstance(environment, Mapping):
        raise ValueError("upstream manifest lacks environment")
    if not isinstance(source, Mapping) or not isinstance(source.get("files"), list):
        raise ValueError("upstream manifest lacks source.files")
    seed = environment.get("SEED")
    try:
        numeric_seed = int(seed)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"upstream manifest has invalid environment.SEED={seed!r}") from exc
    return {
        "schema_version": manifest.get("schema_version"),
        "suite_id": manifest.get("suite_id"),
        "stage": manifest.get("stage"),
        "protocol": manifest.get("protocol"),
        "seed": numeric_seed,
        "fidelity": manifest.get("fidelity"),
        "launcher": manifest.get("launcher"),
        "environment": dict(environment),
        "sources": source.get("files"),
        "datasets": manifest.get("data"),
        "models": manifest.get("models"),
    }


def _paper_comparability(
    state: str, protocol: str, manifest: Mapping[str, Any]
) -> tuple[bool, str]:
    if state != "completed":
        return False, f"upstream stage state is {state!r}, expected 'completed'"
    if protocol != "paper":
        return False, f"upstream protocol is {protocol!r}; engineering runs are not paper-comparable"
    if manifest.get("paper_comparable") is not True:
        return False, "upstream manifest does not explicitly assert paper_comparable=true"
    fidelity = str(manifest.get("fidelity", "")).lower()
    if "adaptation" in fidelity or "not a paper result" in fidelity:
        return False, f"upstream fidelity is hardware/reconstruction adapted: {manifest.get('fidelity')}"
    return True, "completed paper upstream stage with explicit immutable comparability assertion"


def validate_upstream_stage(suite_root: Path, stage: str) -> dict[str, Any] | None:
    if stage not in UPSTREAM_STAGES:
        raise ValueError(f"unknown upstream stage {stage!r}")
    suite_root = suite_root.expanduser().resolve()
    status_path = suite_root / stage / "status.json"
    if not status_path.is_file():
        return None
    status = _read_object(status_path)
    if status.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{status_path}: unsupported schema_version")
    if status.get("stage") != stage:
        raise ValueError(f"{status_path}: stage mismatch")
    protocol = status.get("protocol")
    if protocol not in PROTOCOL_RANK:
        raise ValueError(f"{status_path}: unsupported protocol {protocol!r}")
    state = status.get("state")
    if state not in {"running", "completed", "failed"}:
        raise ValueError(f"{status_path}: unsupported state {state!r}")
    fingerprint = status.get("fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ValueError(f"{status_path}: invalid fingerprint")
    manifest_raw = status.get("manifest")
    if not isinstance(manifest_raw, str) or not manifest_raw:
        raise ValueError(f"{status_path}: missing manifest pointer")
    manifest_path = Path(manifest_raw).expanduser().resolve()
    manifest = _read_object(manifest_path)
    suite_id = manifest.get("suite_id")
    if suite_id not in UPSTREAM_SUITE_IDS:
        raise ValueError(f"{manifest_path}: unsupported suite_id {suite_id!r}")
    rooted_suite_ids = [
        part for part in suite_root.parts if part.startswith("rethinking-opd-upstream-v")
    ]
    if rooted_suite_ids and suite_id != rooted_suite_ids[-1]:
        raise ValueError(
            f"{manifest_path}: suite_id={suite_id!r} does not match artifact root "
            f"{rooted_suite_ids[-1]!r}"
        )
    run_raw = status.get("run_dir")
    if not isinstance(run_raw, str) or not run_raw:
        raise ValueError(f"{status_path}: missing run_dir")
    run_dir = Path(run_raw).expanduser().resolve()
    if manifest_path != run_dir / "upstream_manifest.json":
        raise ValueError(f"{status_path}: manifest must be the run-local upstream_manifest.json")

    exact_fields = {
        "schema_version": SCHEMA_VERSION,
        "suite_id": suite_id,
        "stage": stage,
        "protocol": protocol,
        "fingerprint": fingerprint,
        "attempt": status.get("attempt"),
    }
    for key, expected in exact_fields.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"{manifest_path}: {key}={manifest.get(key)!r}, expected {expected!r} from status"
            )
    if not _same_path(manifest.get("run_dir"), run_dir):
        raise ValueError(f"{manifest_path}: run_dir mismatch")
    expected_scope = "Table 1" if stage == "grpo-teacher" else "Table 3"
    if manifest.get("paper_scope") != expected_scope:
        raise ValueError(f"{manifest_path}: paper_scope mismatch")
    recomputed = _canonical_hash(_fingerprint_identity(manifest))
    if recomputed != fingerprint:
        raise ValueError(
            f"{manifest_path}: fingerprint does not match immutable manifest identity"
        )

    if state == "completed":
        if status.get("exit_code") != 0:
            raise ValueError(f"{status_path}: completed stage must have exit_code=0")
        marker_raw = status.get("success_marker")
        if not isinstance(marker_raw, str) or not marker_raw:
            raise ValueError(f"{status_path}: completed stage lacks success_marker")
        marker = Path(marker_raw).expanduser().resolve()
        if not marker.is_file() or run_dir not in marker.parents:
            raise ValueError(f"{status_path}: completed stage success marker is missing or external")

    comparable, reason = _paper_comparability(str(state), str(protocol), manifest)
    environment = manifest.get("environment")
    seed = int(environment["SEED"]) if isinstance(environment, Mapping) else None
    return {
        "upstream_root": str(suite_root),
        "suite_id": manifest.get("suite_id"),
        "stage": stage,
        "protocol": protocol,
        "seed": seed,
        "state": state,
        "attempt": status.get("attempt"),
        "fingerprint": fingerprint,
        "run_dir": str(run_dir),
        "manifest_path": str(manifest_path),
        "status_path": str(status_path),
        "paper_scope": expected_scope,
        "fidelity": manifest.get("fidelity"),
        "paper_comparable": comparable,
        "paper_comparability_reason": reason,
    }


def _expand_roots(raw_roots: Iterable[Path]) -> list[Path]:
    roots: list[Path] = []
    seen: set[Path] = set()
    for raw in raw_roots:
        root = raw.expanduser().resolve()
        direct = any((root / stage / "status.json").is_file() for stage in UPSTREAM_STAGES)
        candidates = [root] if direct else [
            path.parent.parent
            for path in root.rglob("status.json")
            if path.parent.name in UPSTREAM_STAGES
        ] if root.is_dir() else []
        for candidate in sorted(set(candidates)):
            if candidate not in seen:
                roots.append(candidate)
                seen.add(candidate)
    return roots


def collect_upstream_roots(raw_roots: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in _expand_roots(raw_roots):
        for stage in UPSTREAM_STAGES:
            row = validate_upstream_stage(root, stage)
            if row is not None:
                rows.append(row)
    return rows
