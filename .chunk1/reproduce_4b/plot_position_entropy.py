#!/usr/bin/env python3
"""Render Fig.13/23-style entropy-by-position heatmaps from FileLogger JSONL."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class PositionEntropyRow:
    step: int
    bin_size: int
    student: tuple[float, ...]
    teacher: tuple[float, ...]
    counts: tuple[float, ...]


def _finite_sequence(value: Any, name: str) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON list")
    result = tuple(float(item) for item in value)
    if any(not math.isfinite(item) for item in result):
        raise ValueError(f"{name} contains non-finite values")
    return result


def read_position_entropy(path: Path) -> list[PositionEntropyRow]:
    rows: list[PositionEntropyRow] = []
    with path.expanduser().open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            data = payload.get("data", payload)
            if not isinstance(data, Mapping) or "position-entropy/student_mean_by_bin" not in data:
                continue
            if data.get("opd/position_entropy_schema_version") != 1:
                raise ValueError(f"{path}:{line_number}: unsupported position-entropy schema")
            step = payload.get("step")
            bin_size = data.get("position-entropy/bin_size")
            if not isinstance(step, int) or isinstance(step, bool):
                raise ValueError(f"{path}:{line_number}: step must be an integer")
            if not isinstance(bin_size, int) or isinstance(bin_size, bool) or bin_size <= 0:
                raise ValueError(f"{path}:{line_number}: bin_size must be positive")
            student = _finite_sequence(data["position-entropy/student_mean_by_bin"], "student bins")
            teacher = _finite_sequence(data["position-entropy/teacher_mean_by_bin"], "teacher bins")
            counts = _finite_sequence(data["position-entropy/token_count_by_bin"], "token counts")
            if not student or len(student) != len(teacher) or len(student) != len(counts):
                raise ValueError(f"{path}:{line_number}: position arrays have inconsistent lengths")
            rows.append(PositionEntropyRow(step, bin_size, student, teacher, counts))
    if not rows:
        raise ValueError(f"{path} contains no schema-v1 position entropy records")
    rows.sort(key=lambda row: row.step)
    if len({row.step for row in rows}) != len(rows):
        raise ValueError(f"{path} contains duplicate position-entropy steps")
    if len({row.bin_size for row in rows}) != 1:
        raise ValueError(f"{path} changes position bin size within one run")
    return rows


def render(rows: Sequence[PositionEntropyRow], output: Path, metric: str, title: str | None, dpi: int) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("plotting requires matplotlib and numpy") from exc

    metrics = ("student", "teacher") if metric == "both" else (metric,)
    max_bins = max(len(getattr(row, name)) for row in rows for name in metrics)
    figure, axes = plt.subplots(len(metrics), 1, figsize=(13, 4.5 * len(metrics)), squeeze=False)
    for axis, name in zip(axes[:, 0], metrics):
        matrix = np.full((len(rows), max_bins), np.nan, dtype=float)
        for row_index, row in enumerate(rows):
            values = getattr(row, name)
            valid = [value if row.counts[index] > 0 else np.nan for index, value in enumerate(values)]
            matrix[row_index, : len(valid)] = valid
        image = axis.imshow(matrix, aspect="auto", interpolation="nearest", origin="upper")
        axis.set_yticks(range(len(rows)), [str(row.step) for row in rows])
        bin_size = rows[0].bin_size
        tick_bins = sorted(set([0, max_bins - 1, *range(0, max_bins, max(1, 1024 // bin_size))]))
        axis.set_xticks(tick_bins, [f"{index * bin_size / 1024:g}K" for index in tick_bins])
        axis.set_xlabel("output position")
        axis.set_ylabel("training step")
        axis.set_title(f"{name.capitalize()} entropy by output position")
        figure.colorbar(image, ax=axis, label="entropy")
    if title:
        figure.suptitle(title)
    figure.tight_layout()
    destination = output.expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=dpi)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metric", choices=("student", "teacher", "both"), default="both")
    parser.add_argument("--min-step", type=int)
    parser.add_argument("--max-step", type=int)
    parser.add_argument("--title")
    parser.add_argument("--dpi", type=int, default=180)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        rows = read_position_entropy(args.input_jsonl)
        if args.min_step is not None:
            rows = [row for row in rows if row.step >= args.min_step]
        if args.max_step is not None:
            rows = [row for row in rows if row.step <= args.max_step]
        if not rows:
            raise ValueError("no position entropy rows remain after step filtering")
        render(rows, args.output, args.metric, args.title, args.dpi)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"wrote position-entropy heatmap to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
