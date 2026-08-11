#!/usr/bin/env python3
"""Plot OPD diagnostics from one or more VERL FileLogger JSONL files."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "overlap_ratio": (
        "distillation/overlap_ratio",
        "val-topk/overlap_ratio",
        "overlap_ratio",
    ),
    "overlap_advantage": (
        "distillation/overlap_token_advantage",
        "distillation/overlap_advantage",
        "val-topk/adv_intersection",
        "overlap_token_advantage",
    ),
    "student_entropy": (
        "student/entropy",
        "actor/entropy",
        "distillation/student_entropy",
    ),
    "teacher_entropy": (
        "teacher/entropy",
        "ref/entropy",
        "distillation/teacher_entropy",
    ),
    "abs_entropy_gap": (
        "opd/abs_entropy_gap",
        "distillation/abs_entropy_gap",
    ),
    "student_overlap_mass": (
        "val-topk/student_p_sum_intersection",
        "distillation/student_overlap_mass",
    ),
    "teacher_overlap_mass": (
        "val-topk/teacher_p_sum_intersection",
        "distillation/teacher_overlap_mass",
    ),
    "grad_norm": (
        "actor/grad_norm",
        "train/grad_norm",
        "grad_norm",
    ),
    "response_length": (
        "response_length/mean",
        "train/response_length/mean",
    ),
}


@dataclass(frozen=True)
class Point:
    step: float
    value: float


@dataclass
class RunMetrics:
    label: str
    source: Path
    series: dict[str, list[Point]]
    metric_schema: int | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot OPD diagnostics from VERL FileLogger JSONL logs.")
    parser.add_argument("--input-jsonl", required=True, nargs="+", type=Path, help="One or more VERL JSONL logs.")
    parser.add_argument("--output", required=True, type=Path, help="Output figure (.png, .pdf, or .svg).")
    parser.add_argument("--labels", nargs="+", help="Optional legend labels, one per input log.")
    parser.add_argument("--title", help="Optional figure title.")
    parser.add_argument("--dpi", type=int, default=180, help="Raster output resolution (default: 180).")
    parser.add_argument("--overlap-ratio-key", help="Override the overlap-ratio metric key.")
    parser.add_argument("--overlap-advantage-key", help="Override the overlap-advantage metric key.")
    parser.add_argument("--student-entropy-key", help="Override the student-entropy metric key.")
    parser.add_argument("--teacher-entropy-key", help="Override the teacher-entropy metric key.")
    parser.add_argument("--student-overlap-mass-key", help="Override the student overlap-mass key.")
    parser.add_argument("--teacher-overlap-mass-key", help="Override the teacher overlap-mass key.")
    parser.add_argument("--grad-norm-key", help="Override the gradient-norm metric key.")
    parser.add_argument("--response-length-key", help="Override the mean response-length metric key.")
    parser.add_argument(
        "--allow-mixed-metric-schema",
        action="store_true",
        help="Allow legacy training-reward proxy and schema-v2 Eq. (7) advantage logs on one plot.",
    )
    return parser


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _metric_value(data: Mapping[str, Any], candidates: Sequence[str]) -> float | None:
    for key in candidates:
        value = _finite_number(data.get(key))
        if value is not None:
            return value
    # Some trainers add a namespace such as ``train/`` to logger keys.
    for actual_key, raw_value in data.items():
        if any(str(actual_key).endswith("/" + candidate) for candidate in candidates):
            value = _finite_number(raw_value)
            if value is not None:
                return value
    return None


def read_file_logger(
    path: Path,
    label: str | None = None,
    key_overrides: Mapping[str, str | None] | None = None,
) -> RunMetrics:
    """Read ``{"step": ..., "data": {...}}`` FileLogger records."""

    overrides = key_overrides or {}
    series = {name: [] for name in METRIC_ALIASES}
    metric_schema: int | None = None
    with path.expanduser().open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(payload, Mapping):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            data = payload.get("data", payload)
            if not isinstance(data, Mapping):
                raise ValueError(f"{path}:{line_number}: 'data' must be an object")
            step = _finite_number(payload.get("step", data.get("step")))
            if step is None:
                continue
            raw_schema = _finite_number(data.get("opd/metric_schema_version"))
            if raw_schema is not None:
                row_schema = int(raw_schema)
                if metric_schema is not None and row_schema != metric_schema:
                    raise ValueError(f"{path}: metric schema changes within one log")
                metric_schema = row_schema
            for metric_name, aliases in METRIC_ALIASES.items():
                override = overrides.get(metric_name)
                candidates = (override,) if override else aliases
                value = _metric_value(data, candidates)
                if value is not None:
                    series[metric_name].append(Point(step, value))

    return RunMetrics(label=label or path.stem, source=path, series=series, metric_schema=metric_schema)


def validate_metric_schemas(runs: Sequence[RunMetrics], allow_mixed: bool = False) -> None:
    schemas = {run.metric_schema for run in runs if run.series["overlap_advantage"]}
    if len(schemas) > 1 and not allow_mixed:
        raise ValueError(
            "overlap-advantage logs mix legacy proxy (no schema) with schema-v2 paper Eq. (7); "
            "plot them separately or pass --allow-mixed-metric-schema intentionally"
        )


def entropy_gap(run: RunMetrics) -> list[Point]:
    """Return Eq. (8) when logged, otherwise the legacy scalar proxy."""

    exact = run.series["abs_entropy_gap"]
    if exact:
        return exact

    student = {point.step: point.value for point in run.series["student_entropy"]}
    teacher = {point.step: point.value for point in run.series["teacher_entropy"]}
    return [Point(step, abs(teacher[step] - student[step])) for step in sorted(student.keys() & teacher.keys())]


def _plot_series(axis: Any, runs: Iterable[RunMetrics], metric_name: str) -> bool:
    plotted = False
    for run in runs:
        points = run.series[metric_name]
        if not points:
            continue
        axis.plot([point.step for point in points], [point.value for point in points], label=run.label)
        plotted = True
    return plotted


def render_figure(runs: Sequence[RunMetrics], output: Path, title: str | None = None, dpi: int = 180) -> None:
    """Render the paper's main online-alignment and stability diagnostics."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on runtime image
        raise RuntimeError("Plotting requires matplotlib") from exc

    figure, axes = plt.subplots(2, 3, figsize=(17, 8), constrained_layout=True)
    panels = (
        (axes[0, 0], "overlap_ratio", "Top-k overlap ratio", "ratio"),
        (axes[0, 1], "overlap_advantage", "Overlap-token advantage", "advantage"),
        (axes[1, 1], "grad_norm", "Gradient norm", "norm"),
        (axes[1, 2], "response_length", "Mean response length", "tokens"),
    )
    for axis, metric_name, panel_title, y_label in panels:
        plotted = _plot_series(axis, runs, metric_name)
        axis.set_title(panel_title)
        axis.set_xlabel("step")
        axis.set_ylabel(y_label)
        axis.grid(alpha=0.25)
        if plotted:
            axis.legend()
        else:
            axis.text(0.5, 0.5, "metric not found", ha="center", va="center", transform=axis.transAxes)

    entropy_axis = axes[1, 0]
    entropy_plotted = False
    for run in runs:
        student = run.series["student_entropy"]
        teacher = run.series["teacher_entropy"]
        gap = entropy_gap(run)
        gap_kind = "Eq. (8)" if run.series["abs_entropy_gap"] else "proxy |mean Ht - mean Hs|"
        if student:
            entropy_axis.plot(
                [point.step for point in student],
                [point.value for point in student],
                linestyle=":",
                alpha=0.65,
                label=f"{run.label} student",
            )
            entropy_plotted = True
        if teacher:
            entropy_axis.plot(
                [point.step for point in teacher],
                [point.value for point in teacher],
                linestyle="--",
                alpha=0.65,
                label=f"{run.label} teacher",
            )
            entropy_plotted = True
        if gap:
            entropy_axis.plot(
                [point.step for point in gap],
                [point.value for point in gap],
                linewidth=2.0,
                label=f"{run.label} gap ({gap_kind})",
            )
            entropy_plotted = True
    entropy_axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    entropy_axis.set_title("Student/teacher entropy gap")
    entropy_axis.set_xlabel("step")
    entropy_axis.set_ylabel("entropy")
    entropy_axis.grid(alpha=0.25)
    if entropy_plotted:
        entropy_axis.legend()
    else:
        entropy_axis.text(0.5, 0.5, "metrics not found", ha="center", va="center", transform=entropy_axis.transAxes)

    mass_axis = axes[0, 2]
    mass_plotted = False
    for run in runs:
        for metric_name, role, linestyle in (
            ("student_overlap_mass", "student", "-"),
            ("teacher_overlap_mass", "teacher", "--"),
        ):
            points = run.series[metric_name]
            if points:
                mass_axis.plot(
                    [point.step for point in points],
                    [point.value for point in points],
                    linestyle=linestyle,
                    label=f"{run.label} {role}",
                )
                mass_plotted = True
    mass_axis.set_title("Probability mass on Top-k overlap")
    mass_axis.set_xlabel("step")
    mass_axis.set_ylabel("probability mass")
    mass_axis.grid(alpha=0.25)
    if mass_plotted:
        mass_axis.legend()
    else:
        mass_axis.text(0.5, 0.5, "metrics not found", ha="center", va="center", transform=mass_axis.transAxes)

    if title:
        figure.suptitle(title)
    destination = output.expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=dpi)
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.labels and len(args.labels) != len(args.input_jsonl):
        parser.error("--labels must contain exactly one label per --input-jsonl path")
    overrides = {
        "overlap_ratio": args.overlap_ratio_key,
        "overlap_advantage": args.overlap_advantage_key,
        "student_entropy": args.student_entropy_key,
        "teacher_entropy": args.teacher_entropy_key,
        "student_overlap_mass": args.student_overlap_mass_key,
        "teacher_overlap_mass": args.teacher_overlap_mass_key,
        "grad_norm": args.grad_norm_key,
        "response_length": args.response_length_key,
    }
    labels = args.labels or [path.stem for path in args.input_jsonl]
    try:
        runs = [
            read_file_logger(path, label=label, key_overrides=overrides)
            for path, label in zip(args.input_jsonl, labels)
        ]
        validate_metric_schemas(runs, allow_mixed=args.allow_mixed_metric_schema)
        render_figure(runs, args.output, title=args.title, dpi=args.dpi)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"wrote diagnostics figure to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
