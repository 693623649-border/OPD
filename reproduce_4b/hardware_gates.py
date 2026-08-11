#!/usr/bin/env python3
"""Materialize and assess the isolated two-A100 scientific hardware gates.

Gate runs are deliberately kept in a different suite from scientific results.
The three resource profiles are applied to an entire selected gate group, so a
single difficult condition can never receive an undisclosed one-off setting.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from run_ablations import atomic_json, canonical_hash, read_json, sha256_file, utc_now


PROFILES: dict[str, dict[str, str]] = {
    "base": {
        "ACTOR_OPTIMIZER_OFFLOAD": "false",
        "ACTOR_PARAMETER_OFFLOAD": "false",
        "ROLLOUT_GPU_MEMORY_UTILIZATION": "0.40",
    },
    "optimizer_offload": {
        "ACTOR_OPTIMIZER_OFFLOAD": "true",
        "ACTOR_PARAMETER_OFFLOAD": "false",
        "ROLLOUT_GPU_MEMORY_UTILIZATION": "0.30",
    },
    "parameter_offload": {
        "ACTOR_OPTIMIZER_OFFLOAD": "true",
        "ACTOR_PARAMETER_OFFLOAD": "true",
        "ROLLOUT_GPU_MEMORY_UTILIZATION": "0.30",
    },
}
PROFILE_ORDER = tuple(PROFILES)
PREFLIGHT_GPU_RE = re.compile(
    r"\[gpu\]\s+(?P<gpu>\d+):.*?free=(?P<free>[0-9.]+)/(?P<total>[0-9.]+)\s+GiB",
    re.IGNORECASE,
)
FATAL_LOG_PATTERNS = {
    "oom": re.compile(r"CUDA out of memory|OutOfMemoryError", re.IGNORECASE),
    # Word boundaries prevent env-var names such as NCCL_TIMEOUT=600 or
    # TORCH_NCCL_BLOCKING_WAIT from being misread as a collective failure.
    "nccl": re.compile(
        r"\bNCCL\b[^\n]*(?:error|failed|failure|timeout|unhandled)", re.IGNORECASE
    ),
}
TRAINING_COMPLETE_STATES = {
    "training_complete",
    "evaluation_complete",
    "probe_complete",
    "rendered",
    "scientific_result_available",
}


def _load_object(path: Path, label: str) -> dict[str, Any]:
    payload = read_json(path)
    if payload is None:
        raise FileNotFoundError(f"missing {label}: {path}")
    return payload


def _gate_registry(path: Path) -> dict[str, Any]:
    registry = _load_object(path, "gate matrix")
    spec = registry.get("scientific_spec")
    if (
        registry.get("schema_version") != 2
        or not isinstance(spec, dict)
        or spec.get("engineering_gate_only") is not True
    ):
        raise ValueError("hardware gates require a schema-v2 engineering_gate_only matrix")
    return registry


def _profile_suite_id(base: str, profile: str) -> str:
    return base if profile == "base" else f"{base}-{profile.replace('_', '-')}"


def materialize_profile(matrix_path: Path, profile: str, output_path: Path) -> dict[str, Any]:
    """Write one immutable, group-uniform profile matrix."""

    if profile not in PROFILES:
        raise ValueError(f"unknown resource profile {profile!r}")
    source = _gate_registry(matrix_path)
    payload = copy.deepcopy(source)
    payload["suite_id"] = _profile_suite_id(str(source["suite_id"]), profile)
    scientific_spec = payload["scientific_spec"]
    scientific_spec["gate_profile"] = {
        "id": profile,
        "environment": PROFILES[profile],
        "parent_matrix_sha256": sha256_file(matrix_path),
    }
    overrides = payload["protocols"]["scientific"]["overrides"]
    overrides.update(PROFILES[profile])
    identity = {
        "parent_matrix_sha256": sha256_file(matrix_path),
        "profile": profile,
        "suite_id": payload["suite_id"],
        "environment": PROFILES[profile],
    }
    scientific_spec["gate_profile"]["identity_sha256"] = canonical_hash(identity)
    existing = read_json(output_path)
    if existing is not None and existing != payload:
        raise RuntimeError(f"existing derived gate matrix differs: {output_path}")
    if existing is None:
        atomic_json(output_path, payload)
    return payload


def _expected_cells(registry: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    cells: dict[str, dict[str, Any]] = {}
    for group in registry.get("groups", []):
        scientific = group.get("scientific", {})
        for cell in group.get("cells", []):
            cells[str(cell["id"])] = {
                "group_id": str(group["id"]),
                "expected_final_step": int(scientific["expected_final_step"]),
                "milestone_steps": list(scientific["milestone_steps"]),
            }
    declared = registry["scientific_spec"].get("runnable_training_cells")
    if declared != len(cells):
        raise ValueError(
            f"gate matrix declares {declared} cells but registers {len(cells)}"
        )
    return cells


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing canonical metrics: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: metric row is not an object")
            if not _finite_tree(row):
                raise ValueError(f"{path}:{line_number}: non-finite metric")
            rows.append(row)
    return rows


def _finite_tree(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    return True


def _aborted_ratio(row: Mapping[str, Any]) -> float | None:
    data = row.get("data")
    if not isinstance(data, Mapping):
        return None
    raw = data.get("response/aborted_ratio")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    value = float(raw)
    return value if math.isfinite(value) else None


def _run_dirs(cell_root: Path, status: Mapping[str, Any]) -> list[Path]:
    paths = [path for path in sorted(cell_root.glob("attempt-*")) if path.is_dir()]
    raw = status.get("run_dir")
    if isinstance(raw, str):
        current = Path(raw).expanduser().resolve()
        if current.parent != cell_root.resolve():
            raise ValueError(f"{cell_root.name}: status run_dir escapes the cell root")
        if current not in paths:
            paths.append(current)
    return paths


def _preflight_free_gib(run_dirs: Sequence[Path]) -> dict[str, float]:
    latest: dict[str, float] = {}
    for run_dir in run_dirs:
        path = run_dir / "preflight.log"
        if not path.is_file():
            continue
        matches = list(PREFLIGHT_GPU_RE.finditer(path.read_text(encoding="utf-8", errors="replace")))
        if matches:
            latest = {match.group("gpu"): float(match.group("free")) for match in matches}
    return latest


def _fatal_logs(run_dirs: Sequence[Path]) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    for run_dir in run_dirs:
        for name in ("preflight.log", "train.log", "runtime_monitor.log"):
            path = run_dir / name
            if not path.is_file():
                continue
            content = path.read_text(encoding="utf-8", errors="replace")
            for label, pattern in FATAL_LOG_PATTERNS.items():
                match = pattern.search(content)
                if match:
                    findings.append(
                        {
                            "kind": label,
                            "path": str(path.resolve()),
                            "excerpt": match.group(0)[:240],
                        }
                    )
    return findings


def assess_cell(
    cell_root: Path,
    expected: Mapping[str, Any],
    registry: Mapping[str, Any],
    matrix_sha256: str,
) -> dict[str, Any]:
    issues: list[str] = []
    status = _load_object(cell_root / "status.json", "gate status")
    contract = _load_object(cell_root / "run_contract.json", "gate contract")
    if contract.get("suite_id") != registry.get("suite_id"):
        issues.append("contract suite_id mismatch")
    if contract.get("cell_id") != cell_root.name or status.get("cell_id") != cell_root.name:
        issues.append("cell identity mismatch")
    if contract.get("matrix_sha256") != matrix_sha256:
        issues.append("contract matrix SHA is stale")
    if status.get("fingerprint") != contract.get("fingerprint"):
        issues.append("status/contract fingerprint mismatch")
    state = status.get("execution_state", status.get("state"))
    if state not in TRAINING_COMPLETE_STATES and status.get("state") != "completed":
        issues.append(f"execution_state is {state!r}, not training_complete")

    expected_step = int(expected["expected_final_step"])
    metrics = _read_metrics(cell_root / "metrics.jsonl")
    observed_steps = [row.get("step") for row in metrics]
    if observed_steps != list(range(1, expected_step + 1)):
        issues.append(f"canonical steps do not equal 1..{expected_step}")
    ratios = [_aborted_ratio(row) for row in metrics]
    if any(value is None for value in ratios):
        issues.append("response/aborted_ratio is missing or non-finite")
    maximum_aborted = max((value for value in ratios if value is not None), default=None)
    acceptance = registry["scientific_spec"]["gate_acceptance"]
    if maximum_aborted is not None and maximum_aborted > float(
        acceptance["maximum_aborted_ratio"]
    ):
        issues.append(f"aborted ratio {maximum_aborted:.6f} exceeds the gate")

    run_dirs = _run_dirs(cell_root, status)
    preflight = _preflight_free_gib(run_dirs)
    minimum_free = float(acceptance["minimum_free_gib_per_card"])
    if len(preflight) != 2 or any(value < minimum_free for value in preflight.values()):
        issues.append(f"preflight did not prove two GPUs each had >= {minimum_free:g} GiB free")
    fatals = _fatal_logs(run_dirs)
    if fatals:
        issues.append("OOM or NCCL failure found in attempt logs")

    peaks: dict[str, float] = {}
    for run_dir in run_dirs:
        summary = read_json(run_dir / "gpu_telemetry_summary.json")
        if summary is None:
            continue
        raw_peaks = summary.get("peak_memory_used_mib_by_gpu", {})
        if isinstance(raw_peaks, Mapping):
            for gpu, raw in raw_peaks.items():
                if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                    peaks[str(gpu)] = max(peaks.get(str(gpu), 0.0), float(raw))
    limit_mib = float(acceptance["maximum_physical_peak_gib_per_card"]) * 1024
    if len(peaks) != 2 or any(value > limit_mib for value in peaks.values()):
        issues.append(
            "telemetry did not prove two physical GPU peaks <= "
            f"{acceptance['maximum_physical_peak_gib_per_card']} GiB"
        )

    profile = registry["scientific_spec"].get("gate_profile", {})
    expected_env = profile.get("environment", PROFILES["base"])
    contract_env = contract.get("training", {}).get("environment", {})
    if any(contract_env.get(key) != value for key, value in expected_env.items()):
        issues.append("contract does not bind the registered group-uniform resource profile")

    return {
        "cell_id": cell_root.name,
        "group_id": expected["group_id"],
        "gate_state": "passed" if not issues else "failed",
        "issues": issues,
        "execution_state": state,
        "expected_final_step": expected_step,
        "last_step": observed_steps[-1] if observed_steps else None,
        "maximum_aborted_ratio": maximum_aborted,
        "preflight_free_gib_by_gpu": preflight,
        "peak_memory_used_mib_by_gpu": peaks,
        "fatal_log_findings": fatals,
    }


def _next_profile(profile: str) -> str | None:
    index = PROFILE_ORDER.index(profile)
    return PROFILE_ORDER[index + 1] if index + 1 < len(PROFILE_ORDER) else None


def assess_suite(matrix_path: Path, suite_root: Path) -> dict[str, Any]:
    registry = _gate_registry(matrix_path)
    expected = _expected_cells(registry)
    matrix_sha256 = sha256_file(matrix_path)
    observed: list[dict[str, Any]] = []
    for cell_id, spec in expected.items():
        cell_root = suite_root / cell_id
        if not cell_root.is_dir():
            observed.append(
                {
                    "cell_id": cell_id,
                    "group_id": spec["group_id"],
                    "gate_state": "pending",
                    "issues": ["cell has not run"],
                }
            )
            continue
        try:
            observed.append(assess_cell(cell_root, spec, registry, matrix_sha256))
        except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError) as exc:
            observed.append(
                {
                    "cell_id": cell_id,
                    "group_id": spec["group_id"],
                    "gate_state": "failed",
                    "issues": [str(exc)],
                }
            )

    profile_spec = registry["scientific_spec"].get("gate_profile", {})
    profile = str(profile_spec.get("id", "base"))
    groups: list[dict[str, Any]] = []
    for group_id in sorted({item["group_id"] for item in observed}):
        members = [item for item in observed if item["group_id"] == group_id]
        states = {item["gate_state"] for item in members}
        group_state = "passed" if states == {"passed"} else "pending" if "pending" in states else "failed"
        next_profile = _next_profile(profile) if group_state == "failed" else None
        groups.append(
            {
                "group_id": group_id,
                "gate_state": group_state,
                "profile": profile,
                "next_group_uniform_profile": next_profile,
                "terminal_execution_state": (
                    "blocked_hardware"
                    if group_state == "failed" and next_profile is None
                    else None
                ),
            }
        )
    return {
        "schema_version": 1,
        "kind": "rethinking_opd_hardware_gate_assessment",
        "generated_at": utc_now(),
        "suite_id": registry["suite_id"],
        "suite_root": str(suite_root.resolve()),
        "matrix_sha256": matrix_sha256,
        "profile": profile,
        "cells": observed,
        "groups": groups,
        "all_passed": bool(groups) and all(item["gate_state"] == "passed" for item in groups),
        "scientific_evidence": False,
    }


def parser(repo_root: Path) -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="action", required=True)
    materialize = subparsers.add_parser("materialize")
    materialize.add_argument(
        "--matrix", type=Path, default=repo_root / "reproduce_4b/scientific_gate_matrix.json"
    )
    materialize.add_argument("--profile", choices=PROFILE_ORDER, required=True)
    materialize.add_argument("--output", type=Path, required=True)
    assess = subparsers.add_parser("assess")
    assess.add_argument(
        "--matrix", type=Path, default=repo_root / "reproduce_4b/scientific_gate_matrix.json"
    )
    assess.add_argument("--suite-root", type=Path, required=True)
    assess.add_argument("--output", type=Path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    args = parser(repo_root).parse_args(argv)
    try:
        if args.action == "materialize":
            payload = materialize_profile(
                args.matrix.expanduser().resolve(), args.profile, args.output.expanduser().resolve()
            )
            print(json.dumps({"suite_id": payload["suite_id"], "output": str(args.output)}, indent=2))
            return 0
        report = assess_suite(
            args.matrix.expanduser().resolve(), args.suite_root.expanduser().resolve()
        )
        if args.output:
            output = args.output.expanduser().resolve()
            existing = read_json(output)
            if existing is not None and {
                key: value for key, value in existing.items() if key != "generated_at"
            } != {key: value for key, value in report.items() if key != "generated_at"}:
                raise RuntimeError(f"existing gate assessment differs: {output}")
            atomic_json(output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if report["all_passed"] else 1
    except (FileNotFoundError, json.JSONDecodeError, OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"hardware_gates: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
