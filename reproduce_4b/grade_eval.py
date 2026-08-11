#!/usr/bin/env python3
"""Rule-grade sampled evaluation responses and report avg@N/pass@N.

The authoritative answer extraction and equivalence rules are loaded from
``scripts/val/eval/utils.py`` at grading time.  Keeping that import lazy makes
the CLI and pure metric helpers usable without sympy/pylatexenc installed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import tempfile
from collections import OrderedDict
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable, Mapping, Sequence


DEFAULT_N = 16


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Grade evaluation JSONL with scripts/val/eval/utils.py and report avg@N/pass@N."
    )
    parser.add_argument("--input-jsonl", required=True, type=Path, help="JSONL produced by generate_eval.py.")
    parser.add_argument("--output-json", type=Path, help="Optional summary JSON path; summary is always printed.")
    parser.add_argument(
        "--annotated-jsonl",
        type=Path,
        help="Optional JSONL copy of selected records with correct/formatted fields.",
    )
    parser.add_argument("--n", type=positive_int, default=DEFAULT_N, help="Rollouts per prompt (default: 16).")
    parser.add_argument(
        "--strict-n",
        action="store_true",
        help="Fail unless every prompt has at least N responses; otherwise incomplete prompts are reported.",
    )
    parser.add_argument(
        "--grader-utils",
        type=Path,
        help="Override the repository's scripts/val/eval/utils.py path.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output files intentionally; registered paper evaluations never pass this flag.",
    )
    return parser


def default_grader_utils_path() -> Path:
    return Path(__file__).resolve().parents[1] / "scripts" / "val" / "eval" / "utils.py"


def load_grader_module(path: Path | None = None) -> ModuleType:
    """Load the repository grader without modifying sys.path."""

    module_path = (path or default_grader_utils_path()).expanduser().resolve()
    if not module_path.is_file():
        raise FileNotFoundError(f"grader utility not found: {module_path}")
    spec = importlib.util.spec_from_file_location("opd_eval_utils", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load grader utility: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("grade_answer_verl", "extract_answer"):
        if not callable(getattr(module, name, None)):
            raise AttributeError(f"{module_path} does not define callable {name}")
    return module


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.expanduser().open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: each line must contain a JSON object")
            record["_line_number"] = line_number
            records.append(record)
    if not records:
        raise ValueError(f"{path} contains no JSON records")
    return records


def _record_id(record: Mapping[str, Any]) -> str:
    for key in ("example_id", "row_index", "id"):
        if record.get(key) is not None:
            return str(record[key])
    raise ValueError(f"record on line {record.get('_line_number', '?')} has no example identifier")


def _answer(record: Mapping[str, Any]) -> str:
    for key in ("answer", "ground_truth"):
        if record.get(key) is not None:
            return str(record[key])
    reward_model = record.get("reward_model")
    if isinstance(reward_model, Mapping) and reward_model.get("ground_truth") is not None:
        return str(reward_model["ground_truth"])
    raise ValueError(f"record on line {record.get('_line_number', '?')} has no ground-truth answer")


def _rollout_order(record: Mapping[str, Any], input_position: int) -> tuple[int, int]:
    for key in ("rollout_id", "seed"):
        value = record.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value, input_position
        if isinstance(value, str) and value.isdigit():
            return int(value), input_position
    return input_position, input_position


def group_records(records: Iterable[dict[str, Any]]) -> "OrderedDict[str, list[dict[str, Any]]]":
    grouped: "OrderedDict[str, list[tuple[int, dict[str, Any]]]]" = OrderedDict()
    for position, record in enumerate(records):
        grouped.setdefault(_record_id(record), []).append((position, record))

    ordered: "OrderedDict[str, list[dict[str, Any]]]" = OrderedDict()
    for example_id, positioned in grouped.items():
        positioned.sort(key=lambda item: _rollout_order(item[1], item[0]))
        ordered[example_id] = [record for _, record in positioned]
    return ordered


def evaluate_records(
    records: Iterable[dict[str, Any]],
    n: int,
    grade_fn: Callable[[str, str], bool],
    extract_fn: Callable[[str], Any],
    strict_n: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compute macro avg@N, empirical pass@N, and boxed-answer format rate."""

    grouped = group_records(records)
    incomplete = {example_id: len(items) for example_id, items in grouped.items() if len(items) < n}
    if strict_n:
        wrong_counts = {example_id: len(items) for example_id, items in grouped.items() if len(items) != n}
        if wrong_counts:
            preview = ", ".join(f"{key}={count}" for key, count in list(wrong_counts.items())[:5])
            raise ValueError(
                f"{len(wrong_counts)} prompts do not contain exactly {n} responses ({preview})"
            )
        expected_rollouts = set(range(n))
        malformed_rollouts: list[str] = []
        for example_id, items in grouped.items():
            observed = [item.get("rollout_id") for item in items]
            if any(not isinstance(value, int) or isinstance(value, bool) for value in observed):
                malformed_rollouts.append(example_id)
                continue
            if set(observed) != expected_rollouts or len(set(observed)) != n:
                malformed_rollouts.append(example_id)
        if malformed_rollouts:
            raise ValueError(
                f"{len(malformed_rollouts)} prompts do not contain exactly rollout IDs 0..{n - 1}"
            )

    prompt_scores: list[float] = []
    prompt_passes: list[bool] = []
    format_flags: list[bool] = []
    annotated: list[dict[str, Any]] = []

    for example_id, candidates in grouped.items():
        selected = candidates[:n]
        if not selected:
            continue
        ground_truths = {_answer(record) for record in selected}
        if len(ground_truths) != 1:
            raise ValueError(f"prompt {example_id} has inconsistent ground-truth answers")
        ground_truth = next(iter(ground_truths))
        scores: list[bool] = []
        for record in selected:
            response = str(record.get("response", ""))
            formatted = extract_fn(response) is not None
            correct = bool(grade_fn(response, ground_truth))
            scores.append(correct)
            format_flags.append(formatted)
            clean_record = {key: value for key, value in record.items() if key != "_line_number"}
            annotated.append({**clean_record, "formatted": formatted, "correct": correct})
        prompt_scores.append(sum(scores) / len(scores))
        prompt_passes.append(any(scores))

    if not prompt_scores:
        raise ValueError("no responses were available for scoring")
    avg_at_n = sum(prompt_scores) / len(prompt_scores)
    pass_at_n = sum(prompt_passes) / len(prompt_passes)
    format_rate = sum(format_flags) / len(format_flags)
    total_selected = len(format_flags)
    total_correct = sum(int(record["correct"]) for record in annotated)

    summary: dict[str, Any] = {
        "n": n,
        f"avg@{n}": avg_at_n,
        f"pass@{n}": pass_at_n,
        "format_rate": format_rate,
        "format_error_rate": 1.0 - format_rate,
        "num_prompts": len(prompt_scores),
        "num_complete_prompts": len(prompt_scores) - len(incomplete),
        "num_incomplete_prompts": len(incomplete),
        "num_responses": total_selected,
        "num_correct": total_correct,
        "num_formatted": sum(format_flags),
    }
    # Refuse to emit NaN/Infinity even if a custom grade function behaves oddly.
    if any(not math.isfinite(value) for value in (avg_at_n, pass_at_n, format_rate)):
        raise ValueError("non-finite evaluation metric")
    return summary, annotated


def _commit_text(path: Path, text: str, *, overwrite: bool) -> None:
    destination = path.expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"{destination} already exists; pass --overwrite to replace it")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.tmp-",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output_file:
            output_file.write(text)
            output_file.flush()
            os.fsync(output_file.fileno())
        if overwrite:
            os.replace(temporary, destination)
        else:
            os.link(temporary, destination)
            temporary.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json(path: Path, payload: Mapping[str, Any], *, overwrite: bool = False) -> None:
    _commit_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        overwrite=overwrite,
    )


def write_jsonl(
    path: Path, records: Iterable[Mapping[str, Any]], *, overwrite: bool = False
) -> None:
    text = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    _commit_text(path, text, overwrite=overwrite)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        grader = load_grader_module(args.grader_utils)
        records = read_jsonl(args.input_jsonl)
        summary, annotated = evaluate_records(
            records,
            n=args.n,
            grade_fn=grader.grade_answer_verl,
            extract_fn=grader.extract_answer,
            strict_n=args.strict_n,
        )
    except (AttributeError, FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    input_path = args.input_jsonl.expanduser().resolve()
    grader_path = (args.grader_utils or default_grader_utils_path()).expanduser().resolve()
    summary["input_jsonl"] = str(input_path)
    summary["input_jsonl_sha256"] = sha256_file(input_path)
    summary["grader"] = {
        "path": str(grader_path),
        "sha256": sha256_file(grader_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    try:
        if args.output_json:
            write_json(args.output_json, summary, overwrite=args.overwrite)
        if args.annotated_jsonl:
            write_jsonl(args.annotated_jsonl, annotated, overwrite=args.overwrite)
    except FileExistsError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
