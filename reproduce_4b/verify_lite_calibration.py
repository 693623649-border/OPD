#!/usr/bin/env python3
"""Gate the high-risk lite Extended pilot on fresh, finite calibrations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import run_ablations


REQUIRED_CELLS = ("fig4-6-deepseek-r1-7b", "fig12-length-7168")
REQUIRED_DATA_KEYS = (
    "opd/metric_schema_version",
    "val-topk/overlap_ratio",
    "actor/grad_norm",
    "perf/max_memory_reserved_gb",
    "response_length/mean",
    "response_length/clip_ratio",
    "response/aborted_ratio",
    "timing_s/step",
    "training/global_step",
)


def _all_numbers_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_all_numbers_finite(item) for item in value)
    if isinstance(value, dict):
        return all(_all_numbers_finite(item) for item in value.values())
    return False


def _metric_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"calibration metrics are missing: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid calibration metric {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"calibration metric {path}:{line_number} is not an object")
            if not _all_numbers_finite(row):
                raise RuntimeError(f"calibration metric {path}:{line_number} contains non-finite data")
            rows.append(row)
    if not rows:
        raise RuntimeError(f"calibration metrics are empty: {path}")
    return rows


def _validate_metric_contract(
    cell_id: str,
    rows: list[dict[str, Any]],
    expected_step: int,
) -> None:
    expected_steps = list(range(1, expected_step + 1))
    steps = [row.get("step") for row in rows]
    if steps != expected_steps:
        raise RuntimeError(
            f"{cell_id} calibration metric steps={steps!r}, expected {expected_steps!r}"
        )
    for step, row in zip(expected_steps, rows, strict=True):
        data = row.get("data")
        if not isinstance(data, dict) or not data:
            raise RuntimeError(f"{cell_id} calibration step {step} has no metric data")
        missing = [key for key in REQUIRED_DATA_KEYS if key not in data]
        if missing:
            raise RuntimeError(
                f"{cell_id} calibration step {step} is missing metrics: {missing}"
            )
        for key in REQUIRED_DATA_KEYS:
            value = data[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise RuntimeError(
                    f"{cell_id} calibration step {step} metric {key} is not finite numeric data"
                )
        if data["training/global_step"] != step:
            raise RuntimeError(
                f"{cell_id} calibration row step {step} disagrees with training/global_step="
                f"{data['training/global_step']!r}"
            )
        if data["opd/metric_schema_version"] != 2:
            raise RuntimeError(
                f"{cell_id} calibration step {step} has unsupported OPD metric schema "
                f"{data['opd/metric_schema_version']!r}"
            )


def verify_required_calibrations(
    matrix_path: Path,
    repo_root: Path,
    run_root: Path,
    seed: int,
) -> list[str]:
    matrix_path = matrix_path.expanduser().resolve()
    run_root = run_root.expanduser().resolve()
    matrix_sha256 = run_ablations.sha256_file(matrix_path)
    registry = run_ablations.load_registry(matrix_path, repo_root)
    datasets = run_ablations.validate_datasets(registry)
    groups = run_ablations.select_groups(
        registry,
        requested_groups=set(),
        requested_cells=set(REQUIRED_CELLS),
        include_extensions=True,
    )
    source_hash = run_ablations.source_tree_hash(repo_root)
    plans = run_ablations.build_plans(
        registry,
        groups,
        "calibration",
        seed,
        datasets,
        source_hash,
        matrix_sha256,
    )
    if {plan.cell_id for plan in plans} != set(REQUIRED_CELLS):
        raise RuntimeError("lite matrix does not resolve the required Extended calibration cells")

    suite_root = run_root / registry["suite_id"] / "calibration" / f"seed-{seed}"
    manifest = run_ablations.read_json(suite_root / "suite_manifest.json")
    if manifest is None:
        raise RuntimeError(f"calibration suite manifest is missing: {suite_root / 'suite_manifest.json'}")
    expected_manifest = {
        "suite_id": registry["suite_id"],
        "protocol": "calibration",
        "seed": seed,
        "matrix_sha256": matrix_sha256,
        "source_tree_sha256": source_hash,
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            raise RuntimeError(
                f"calibration manifest {key}={manifest.get(key)!r}, expected {expected!r}"
            )
    manifest_cells = {
        cell.get("cell_id"): cell
        for cell in manifest.get("cells", [])
        if isinstance(cell, dict) and isinstance(cell.get("cell_id"), str)
    }

    verified: list[str] = []
    for plan in plans:
        manifest_cell = manifest_cells.get(plan.cell_id)
        if manifest_cell is None:
            raise RuntimeError(f"calibration manifest is missing cell: {plan.cell_id}")
        if manifest_cell.get("fingerprint") != plan.fingerprint:
            raise RuntimeError(f"calibration manifest fingerprint is stale: {plan.cell_id}")
        cell_root = suite_root / plan.cell_id
        status_path = cell_root / "status.json"
        status = run_ablations.read_json(status_path)
        if status is None:
            raise RuntimeError(f"required calibration is pending: {plan.cell_id}")
        expected_step = int(plan.env["TOTAL_TRAINING_STEPS"])
        checks = {
            "cell_id": plan.cell_id,
            "fingerprint": plan.fingerprint,
            "state": "completed",
            "exit_code": 0,
            "last_metric_step": expected_step,
        }
        for key, expected in checks.items():
            if status.get(key) != expected:
                raise RuntimeError(
                    f"{plan.cell_id} calibration {key}={status.get(key)!r}, expected {expected!r}"
                )
        metrics_value = status.get("metrics_file")
        if not isinstance(metrics_value, str) or not metrics_value:
            raise RuntimeError(f"{plan.cell_id} calibration has no metrics_file")
        metrics_path = Path(metrics_value).expanduser().resolve()
        try:
            metrics_path.relative_to(cell_root.resolve())
        except ValueError as exc:
            raise RuntimeError(
                f"{plan.cell_id} calibration metrics escape the cell directory: {metrics_path}"
            ) from exc
        rows = _metric_rows(metrics_path)
        _validate_metric_contract(plan.cell_id, rows, expected_step)
        verified.append(plan.cell_id)
    return verified


def build_parser(repo_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        allow_abbrev=False,
    )
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=repo_root / "artifacts/ablations")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = build_parser(repo_root)
    args, _runner_args = parser.parse_known_args(argv)
    try:
        verified = verify_required_calibrations(
            args.matrix,
            repo_root,
            args.run_root,
            args.seed,
        )
    except (FileNotFoundError, json.JSONDecodeError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print("Extended calibration gate passed: " + ", ".join(verified))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
