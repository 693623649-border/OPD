#!/usr/bin/env python3
"""Collect auditable long-form training/evaluation tables for all paper figures."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from paper_eval_contract import validate_paper_evaluation
from upstream_artifacts import collect_upstream_roots


def read_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def scalar_metrics(path: Path) -> Iterable[tuple[int, str, float]]:
    if not path.is_file():
        return
    seen_steps: set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            step = payload.get("step")
            data = payload.get("data")
            if not isinstance(step, int) or isinstance(step, bool) or step <= 0:
                raise ValueError(f"{path}:{line_number}: invalid positive integer step")
            if step in seen_steps:
                raise ValueError(f"{path}:{line_number}: duplicate step {step}")
            if not isinstance(data, Mapping):
                raise ValueError(f"{path}:{line_number}: missing metric data object")
            seen_steps.add(step)
            for metric, raw in data.items():
                if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                    continue
                value = float(raw)
                if not math.isfinite(value):
                    raise ValueError(f"{path}:{line_number}: non-finite {metric}")
                yield step, str(metric), value


def _manifest_cells(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    cells = manifest.get("cells")
    if not isinstance(cells, list):
        raise ValueError("suite manifest has no cells list")
    indexed: dict[str, Mapping[str, Any]] = {}
    for cell in cells:
        if not isinstance(cell, Mapping) or not isinstance(cell.get("cell_id"), str):
            raise ValueError("suite manifest contains invalid cell")
        if cell["cell_id"] in indexed:
            raise ValueError(f"suite manifest repeats {cell['cell_id']}")
        indexed[cell["cell_id"]] = cell
    return indexed


def _training_complete(status: Mapping[str, Any]) -> bool:
    return status.get("state") == "completed" or status.get("execution_state") in {
        "training_complete",
        "evaluation_complete",
        "probe_complete",
        "rendered",
        "scientific_result_available",
    }


def _evaluation_status_for_summary(run_dir: Path, summary_path: Path) -> Path:
    evaluation_root = summary_path.parent.parent
    default_root = (run_dir / "evaluation").resolve()
    if evaluation_root.resolve() == default_root:
        return run_dir / "evaluation_status.json"
    return evaluation_root / "evaluation_status.json"


def collect_suite(suite_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest_path = suite_root / "suite_manifest.json"
    manifest = read_object(manifest_path)
    cells = _manifest_cells(manifest)
    training: list[dict[str, Any]] = []
    evaluation: list[dict[str, Any]] = []
    for cell_id, cell in cells.items():
        status_path = suite_root / cell_id / "status.json"
        if not status_path.is_file():
            continue
        status = read_object(status_path)
        if not _training_complete(status):
            continue
        if status.get("fingerprint") != cell.get("fingerprint"):
            raise ValueError(f"{cell_id}: status/manifest fingerprint mismatch")
        run_dir = Path(str(status.get("run_dir", "")))
        metrics_path = Path(str(status.get("metrics_file", run_dir / "metrics.jsonl")))
        comparability = status.get("comparability", cell.get("comparability"))
        if not isinstance(comparability, Mapping):
            comparability = {}
        common = {
            "suite_id": manifest.get("suite_id"),
            "protocol": manifest.get("protocol"),
            "seed": manifest.get("seed"),
            "source_tree_sha256": manifest.get("source_tree_sha256"),
            "group_id": cell.get("group_id"),
            "cell_id": cell_id,
            "label": cell.get("label"),
            "fidelity": cell.get("fidelity"),
            "fingerprint": cell.get("fingerprint"),
            "attempt": status.get("attempt"),
            "run_dir": str(run_dir),
            "execution_state": status.get("execution_state", status.get("state")),
            "conclusion_state": status.get("conclusion_state", "not_assessed"),
            "training_fidelity": comparability.get("training"),
            "evaluation_fidelity": comparability.get("evaluation"),
            "provenance_fidelity": comparability.get("provenance"),
        }
        for step, metric, value in scalar_metrics(metrics_path):
            training.append({**common, "step": step, "metric": metric, "value": value})
        evaluation_root = run_dir / "evaluation"
        if evaluation_root.is_dir():
            for summary_path in sorted(evaluation_root.glob("**/global_step_*/summary.json")):
                summary = read_object(summary_path)
                step = summary.get("checkpoint_step")
                n = summary.get("n")
                benchmarks = summary.get("benchmarks")
                if not isinstance(step, int) or not isinstance(n, int) or not isinstance(benchmarks, Mapping):
                    raise ValueError(f"invalid checkpoint evaluation summary: {summary_path}")
                evaluation_manifest = summary_path.parent / "target_manifest.json"
                check = validate_paper_evaluation(
                    summary_path,
                    protocol=str(manifest.get("protocol")),
                    manifest_path=evaluation_manifest if evaluation_manifest.is_file() else None,
                    status_path=_evaluation_status_for_summary(run_dir, summary_path),
                )
                key = f"avg@{n}"
                for benchmark, metrics in benchmarks.items():
                    if not isinstance(metrics, Mapping) or not isinstance(metrics.get(key), (int, float)):
                        raise ValueError(f"{summary_path}: missing {key} for {benchmark}")
                    evaluation.append(
                        {
                            **common,
                            "target_kind": "cell_checkpoint",
                            "checkpoint_step": step,
                            "benchmark": benchmark,
                            "n": n,
                            "avg_at_n": float(metrics[key]),
                            "paper_comparable": check.paper_comparable,
                            "evaluation_protocol_comparable": (
                                check.evaluation_protocol_comparable
                            ),
                            "training_comparability": check.training_comparability,
                            "evaluation_id": summary.get("evaluation_id", "default"),
                            "paper_comparability_reason": check.reason,
                            "summary_path": str(summary_path),
                        }
                    )
    return training, evaluation


def collect_model_evaluations(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.is_dir():
        return rows
    for summary_path in sorted(root.glob("*/summary.json")):
        summary = read_object(summary_path)
        manifest_path = summary_path.parent / "target_manifest.json"
        manifest = read_object(manifest_path)
        status_path = summary_path.parent / "status.json"
        check = validate_paper_evaluation(
            summary_path,
            protocol="paper",
            manifest_path=manifest_path,
            status_path=status_path,
        )
        n = summary.get("n")
        benchmarks = summary.get("benchmarks")
        if not isinstance(n, int) or not isinstance(benchmarks, Mapping):
            raise ValueError(f"invalid model evaluation summary: {summary_path}")
        key = f"avg@{n}"
        for benchmark, metrics in benchmarks.items():
            if not isinstance(metrics, Mapping) or not isinstance(metrics.get(key), (int, float)):
                raise ValueError(f"{summary_path}: missing {key} for {benchmark}")
            rows.append(
                {
                    "suite_id": None,
                    "protocol": "paper-eval" if check.paper_comparable else "smoke-eval",
                    "seed": manifest.get("sampling", {}).get("seed"),
                    "source_tree_sha256": None,
                    "group_id": None,
                    "cell_id": manifest.get("target_id"),
                    "label": manifest.get("target_id"),
                    "fidelity": "immutable_model_baseline",
                    "fingerprint": None,
                    "attempt": None,
                    "run_dir": str(summary_path.parent),
                    "target_kind": "model_baseline",
                    "checkpoint_step": 0,
                    "benchmark": benchmark,
                    "n": n,
                    "avg_at_n": float(metrics[key]),
                    "paper_comparable": check.paper_comparable,
                    "paper_comparability_reason": check.reason,
                    "summary_path": str(summary_path),
                }
            )
    return rows


def write_csv_atomic(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser(repo_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", type=Path, action="append", default=[])
    parser.add_argument(
        "--model-eval-root", type=Path, default=repo_root / "artifacts/evaluation/paper-models"
    )
    parser.add_argument(
        "--upstream-root", action="append", type=Path, default=[],
        help="Repeatable upstream suite root (or parent containing protocol/seed suites).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=repo_root / "artifacts/paper_reproduction/tables"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    args = build_parser(repo_root).parse_args(argv)
    training: list[dict[str, Any]] = []
    evaluation: list[dict[str, Any]] = []
    upstream: list[dict[str, Any]] = []
    try:
        for raw_root in args.suite_root:
            train_rows, eval_rows = collect_suite(raw_root.expanduser().resolve())
            training.extend(train_rows)
            evaluation.extend(eval_rows)
        evaluation.extend(collect_model_evaluations(args.model_eval_root.expanduser().resolve()))
        upstream.extend(collect_upstream_roots(args.upstream_root))
        output = args.output_dir.expanduser().resolve()
        training_fields = (
            "suite_id", "protocol", "seed", "source_tree_sha256", "group_id", "cell_id",
            "label", "fidelity", "fingerprint", "attempt", "run_dir", "execution_state",
            "conclusion_state", "training_fidelity", "evaluation_fidelity", "provenance_fidelity",
            "step", "metric", "value",
        )
        evaluation_fields = (
            "suite_id", "protocol", "seed", "source_tree_sha256", "group_id", "cell_id",
            "label", "fidelity", "fingerprint", "attempt", "run_dir", "target_kind",
            "checkpoint_step", "benchmark", "n", "avg_at_n", "evaluation_id",
            "paper_comparable", "evaluation_protocol_comparable", "training_comparability",
            "training_fidelity", "evaluation_fidelity", "provenance_fidelity", "summary_path",
            "paper_comparability_reason",
        )
        upstream_fields = (
            "upstream_root", "suite_id", "stage", "protocol", "seed", "state",
            "attempt", "fingerprint", "run_dir", "manifest_path", "status_path",
            "paper_scope", "fidelity", "paper_comparable", "paper_comparability_reason",
        )
        write_csv_atomic(output / "metrics_long.csv", training, training_fields)
        write_csv_atomic(output / "evaluation_long.csv", evaluation, evaluation_fields)
        write_csv_atomic(output / "upstream_long.csv", upstream, upstream_fields)
        write_json_atomic(output / "metrics_long.json", training)
        write_json_atomic(output / "evaluation_long.json", evaluation)
        write_json_atomic(output / "upstream_long.json", upstream)
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(
        f"training_rows={len(training)} evaluation_rows={len(evaluation)} "
        f"upstream_rows={len(upstream)} output={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
