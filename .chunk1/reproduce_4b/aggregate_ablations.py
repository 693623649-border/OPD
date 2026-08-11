#!/usr/bin/env python3
"""Aggregate formal ablation status, final diagnostics, evaluation scores, and plots."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import plot_metrics


FINAL_METRICS = {
    "overlap_ratio": "val-topk/overlap_ratio",
    "eq7_advantage": "val-topk/adv_intersection",
    "training_reward_proxy": "val-topk/training_adv_intersection",
    "eq8_abs_entropy_gap": "opd/abs_entropy_gap",
    "student_entropy": "actor/entropy",
    "teacher_entropy": "teacher/entropy",
    "student_overlap_mass": "val-topk/student_p_sum_intersection",
    "teacher_overlap_mass": "val-topk/teacher_p_sum_intersection",
    "grad_norm": "actor/grad_norm",
    "mean_response_length": "response_length/mean",
    "max_memory_allocated_gib": "perf/max_memory_allocated_gb",
    "max_memory_reserved_gib": "perf/max_memory_reserved_gb",
}


def finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def last_metrics(path: Path) -> tuple[int | None, dict[str, Any]]:
    last_step: int | None = None
    last_data: dict[str, Any] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            data = payload.get("data", payload)
            if not isinstance(data, Mapping):
                raise ValueError(f"{path}:{line_number}: data must be an object")
            step = payload.get("step")
            if isinstance(step, int) and not isinstance(step, bool):
                last_step = step
                last_data = dict(data)
    return last_step, last_data


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def collect(suite_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    suite = read_json(suite_root / "suite_manifest.json")
    if suite is None:
        raise FileNotFoundError(f"missing suite_manifest.json under {suite_root}")
    rows: list[dict[str, Any]] = []
    for cell in suite.get("cells", []):
        cell_id = cell["cell_id"]
        status = read_json(suite_root / cell_id / "status.json")
        row: dict[str, Any] = {
            "group_id": cell["group_id"],
            "cell_id": cell_id,
            "label": cell["label"],
            "fidelity": cell["fidelity"],
            "fingerprint": cell["fingerprint"],
            "state": status.get("state") if status else "pending",
            "attempt": status.get("attempt") if status else None,
            "last_step": status.get("last_metric_step") if status else None,
            "run_dir": status.get("run_dir") if status else None,
        }
        if status and status.get("run_dir"):
            run_dir = Path(status["run_dir"])
            metrics_path = run_dir / "metrics.jsonl"
            if metrics_path.is_file():
                step, data = last_metrics(metrics_path)
                row["last_step"] = step
                row["metrics_file"] = str(metrics_path)
                row["metric_schema_version"] = data.get("opd/metric_schema_version")
                row["entropy_gap_schema_version"] = data.get("opd/entropy_gap_schema_version")
                for output_name, source_name in FINAL_METRICS.items():
                    row[output_name] = finite(data.get(source_name))
            evaluation_root = run_dir / "evaluation"
            summaries = sorted(evaluation_root.glob("global_step_*/summary.json"))
            if summaries:
                latest = max(summaries, key=lambda path: int(path.parent.name.removeprefix("global_step_")))
                summary = read_json(latest)
                row["evaluation_step"] = summary.get("checkpoint_step") if summary else None
                row["benchmark_macro_mean_avg_at_n"] = (
                    summary.get("benchmark_macro_mean_avg_at_n") if summary else None
                )
                row["evaluation_summary"] = str(latest)
        rows.append(row)
    return suite, rows


def write_outputs(rows: Sequence[Mapping[str, Any]], output_json: Path, output_csv: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(list(rows), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fieldnames = sorted({key for row in rows for key in row})
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_group_plots(rows: Sequence[Mapping[str, Any]], plot_dir: Path) -> list[str]:
    by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("state") == "completed" and row.get("metrics_file"):
            by_group[str(row["group_id"])].append(row)
    outputs: list[str] = []
    for group_id, group_rows in by_group.items():
        if len(group_rows) < 2:
            continue
        runs = [
            plot_metrics.read_file_logger(Path(str(row["metrics_file"])), label=str(row["label"]))
            for row in group_rows
        ]
        plot_metrics.validate_metric_schemas(runs)
        output = plot_dir / f"{group_id}.png"
        plot_metrics.render_figure(runs, output, title=group_id, dpi=180)
        outputs.append(str(output))
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-root", required=True, type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--plot-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        suite_root = args.suite_root.expanduser().resolve()
        _, rows = collect(suite_root)
        output_json = args.output_json or suite_root / "results.json"
        output_csv = args.output_csv or suite_root / "results.csv"
        write_outputs(rows, output_json, output_csv)
        plots = render_group_plots(rows, args.plot_dir or suite_root / "plots")
    except (FileNotFoundError, json.JSONDecodeError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    completed = sum(row.get("state") == "completed" for row in rows)
    print(f"cells={len(rows)} completed={completed} plots={len(plots)}")
    print(f"wrote {output_json} and {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
