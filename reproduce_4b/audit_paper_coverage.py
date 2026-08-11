#!/usr/bin/env python3
"""Audit structural and observed coverage of every paper figure and table.

The static ledger records what must be produced.  This script joins that ledger
with one or more immutable ablation suite manifests, checkpoint evaluations,
model-only evaluations, and standalone probe summaries.  It never upgrades a
smoke run to scientific evidence: protocol and avg@N provenance remain visible
in every output row.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from paper_eval_contract import validate_paper_evaluation
from upstream_artifacts import PROTOCOL_RANK as UPSTREAM_PROTOCOL_RANK
from upstream_artifacts import collect_upstream_roots


SCHEMA_VERSION = 1
EXPECTED_ENTRY_IDS = tuple(
    [f"figure-{number:02d}" for number in range(1, 24)]
    + [f"table-{number:02d}" for number in range(1, 4)]
)
PROTOCOL_RANK = {"smoke": 1, "calibration": 2, "pilot": 3, "paper": 4}
UPSTREAM_REQUIREMENTS = {
    "table-01": ("grpo-teacher",),
    "table-03": ("cold-start-rollout", "cold-start-sft"),
}
JUST_RL_MIRROR = "hbx/JustRL-DeepSeek-1.5B"
METRIC_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "figure-02": ("val-topk/overlap_ratio",),
    "figure-04": ("val-topk/overlap_ratio",),
    "figure-06": (
        "val-topk/overlap_ratio",
        "val-topk/adv_intersection",
        "opd/abs_entropy_gap",
    ),
    "figure-07": ("val-topk/overlap_ratio", "val-topk/adv_intersection"),
    "figure-08": (
        "val-topk/overlap_ratio",
        "val-topk/adv_intersection",
        "opd/abs_entropy_gap",
    ),
    "figure-09": ("val-topk/overlap_ratio",),
    "figure-10": (
        "val-topk/overlap_ratio",
        "val-topk/student_p_sum_intersection",
        "actor/entropy",
    ),
    "figure-12": ("val-topk/overlap_ratio", "actor/entropy", "actor/grad_norm"),
    "figure-13": (
        "opd/position_entropy_schema_version",
        "position-entropy/student_mean_by_bin",
        "position-entropy/token_count_by_bin",
    ),
    "figure-16": ("val-topk/overlap_ratio", "actor/entropy", "actor/grad_norm"),
    "figure-18": (
        "val-topk/student_p_sum_intersection",
        "val-topk/teacher_p_sum_intersection",
    ),
    "figure-19": (
        "actor/pg_loss",
        "actor/grad_norm",
        "opd/figure19_metric_schema_version",
        "val-extrema/prob_diff_at_max_abs_adv_intersection",
    ),
    "figure-20": (
        "val-topk/overlap_ratio",
        "val-topk/adv_intersection",
        "opd/abs_entropy_gap",
    ),
    "figure-21": (
        "val-topk/student_p_sum_intersection",
        "val-topk/teacher_p_sum_intersection",
    ),
    "figure-23": (
        "opd/position_entropy_schema_version",
        "position-entropy/teacher_mean_by_bin",
        "position-entropy/token_count_by_bin",
    ),
}

SCHEMA_METRICS = {
    "opd/position_entropy_schema_version",
    "opd/figure19_metric_schema_version",
}
ARRAY_METRICS = {
    "position-entropy/student_mean_by_bin",
    "position-entropy/teacher_mean_by_bin",
    "position-entropy/token_count_by_bin",
}


def read_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def matrix_cells(matrix: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if matrix.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("matrix has unsupported schema_version")
    groups = matrix.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("matrix.groups must be a non-empty list")
    output: dict[str, dict[str, Any]] = {}
    for group in groups:
        if not isinstance(group, Mapping) or not isinstance(group.get("id"), str):
            raise ValueError("matrix contains an invalid group")
        constants = group.get("constants")
        factors = group.get("factor_keys")
        cells = group.get("cells")
        if not isinstance(constants, Mapping) or not isinstance(factors, list) or not isinstance(cells, list):
            raise ValueError(f"matrix group {group['id']} is malformed")
        for cell in cells:
            if not isinstance(cell, Mapping) or not isinstance(cell.get("id"), str):
                raise ValueError(f"matrix group {group['id']} contains an invalid cell")
            cell_id = cell["id"]
            if cell_id in output:
                raise ValueError(f"matrix repeats cell ID {cell_id}")
            cell_factors = cell.get("factors")
            if not isinstance(cell_factors, Mapping) or set(cell_factors) != set(factors):
                raise ValueError(f"matrix cell {cell_id} does not define exactly its factor keys")
            output[cell_id] = {
                "group_id": group["id"],
                "group_question": group.get("question"),
                "paper_location": group.get("paper_location"),
                "fidelity": group.get("fidelity"),
                "label": cell.get("label"),
                "disabled_by_default": bool(group.get("disabled_by_default")),
                "allowed_protocols": group.get("allowed_protocols"),
                "environment": {**constants, **cell_factors},
                "protocol_overrides": group.get("protocol_overrides", {}),
                "disclosure": group.get("disclosure"),
            }
    return output


def validate_ledger(
    ledger: Mapping[str, Any], known_cells: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if ledger.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("ledger has unsupported schema_version")
    entries = ledger.get("entries")
    if not isinstance(entries, list):
        raise ValueError("ledger.entries must be a list")
    ids = [entry.get("id") if isinstance(entry, Mapping) else None for entry in entries]
    if tuple(ids) != EXPECTED_ENTRY_IDS:
        missing = sorted(set(EXPECTED_ENTRY_IDS) - set(ids))
        extra = sorted(set(ids) - set(EXPECTED_ENTRY_IDS), key=str)
        raise ValueError(
            "ledger must contain ordered figure-01..figure-23 and table-01..table-03; "
            f"missing={missing}, extra={extra}"
        )
    referenced: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for entry in entries:
        assert isinstance(entry, Mapping)
        entry_id = str(entry["id"])
        producer = entry.get("producer")
        if not isinstance(producer, Mapping):
            raise ValueError(f"{entry_id}.producer must be an object")
        producer_type = producer.get("type")
        cell_ids = producer.get("cell_ids")
        postprocess = producer.get("postprocess")
        if not isinstance(producer_type, str) or not producer_type:
            raise ValueError(f"{entry_id}.producer.type must be non-empty")
        if not isinstance(cell_ids, list) or any(not isinstance(value, str) for value in cell_ids):
            raise ValueError(f"{entry_id}.producer.cell_ids must be a string list")
        if len(cell_ids) != len(set(cell_ids)):
            raise ValueError(f"{entry_id} repeats a producer cell")
        unknown = sorted(set(cell_ids) - set(known_cells))
        if unknown:
            raise ValueError(f"{entry_id} references unknown cells: {unknown}")
        if not isinstance(postprocess, list) or not postprocess or any(
            not isinstance(value, str) or not value for value in postprocess
        ):
            raise ValueError(f"{entry_id}.producer.postprocess must be a non-empty string list")
        for field in ("paper_location", "status", "fidelity"):
            if not isinstance(entry.get(field), str) or not entry[field]:
                raise ValueError(f"{entry_id}.{field} must be non-empty")
        blocker = entry.get("blocker")
        if blocker is not None and (not isinstance(blocker, str) or not blocker):
            raise ValueError(f"{entry_id}.blocker must be null or a non-empty string")
        referenced.update(cell_ids)
        normalized.append(dict(entry))
    unreferenced = sorted(set(known_cells) - referenced)
    if unreferenced:
        raise ValueError(f"matrix cells absent from the paper ledger: {unreferenced}")
    return normalized


def _checkpoint_evaluations(run_dir: Path, protocol: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted((run_dir / "evaluation").glob("global_step_*/summary.json")):
        summary = read_object(summary_path)
        n = summary.get("n")
        benchmarks = summary.get("benchmarks")
        if not isinstance(n, int) or isinstance(n, bool) or not isinstance(benchmarks, Mapping):
            raise ValueError(f"invalid checkpoint evaluation summary: {summary_path}")
        complete = set(benchmarks) == {"AIME24", "AIME25", "AMC23"} and all(
            isinstance(metrics, Mapping) and isinstance(metrics.get(f"avg@{n}"), (int, float))
            for metrics in benchmarks.values()
        )
        evaluation_manifest = summary_path.parent / "target_manifest.json"
        check = validate_paper_evaluation(
            summary_path,
            protocol=protocol,
            manifest_path=evaluation_manifest if evaluation_manifest.is_file() else None,
            status_path=run_dir / "evaluation_status.json",
        )
        rows.append(
            {
                "step": summary.get("checkpoint_step"),
                "n": n,
                "benchmarks": sorted(str(name) for name in benchmarks),
                "complete": complete,
                "paper_comparable": check.paper_comparable,
                "paper_comparability_reason": check.reason,
                "path": str(summary_path),
            }
        )
    return rows


def collect_suites(suite_roots: Iterable[Path]) -> dict[str, list[dict[str, Any]]]:
    observed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for suite_root in suite_roots:
        root = suite_root.expanduser().resolve()
        manifest = read_object(root / "suite_manifest.json")
        protocol = manifest.get("protocol")
        if protocol not in PROTOCOL_RANK:
            raise ValueError(f"{root}: unsupported suite protocol {protocol!r}")
        cells = manifest.get("cells")
        if not isinstance(cells, list):
            raise ValueError(f"{root}: suite manifest lacks cells")
        for cell in cells:
            if not isinstance(cell, Mapping) or not isinstance(cell.get("cell_id"), str):
                raise ValueError(f"{root}: invalid manifest cell")
            cell_id = cell["cell_id"]
            status_path = root / cell_id / "status.json"
            state = "pending"
            status: Mapping[str, Any] = {}
            if status_path.is_file():
                status = read_object(status_path)
                if status.get("fingerprint") != cell.get("fingerprint"):
                    raise ValueError(f"{cell_id}: status/manifest fingerprint mismatch")
                state = str(status.get("state", "unknown"))
            run_dir = Path(str(status.get("run_dir", ""))) if status.get("run_dir") else None
            evaluations = _checkpoint_evaluations(run_dir, str(protocol)) if run_dir else []
            observed[cell_id].append(
                {
                    "suite_root": str(root),
                    "suite_id": manifest.get("suite_id"),
                    "protocol": protocol,
                    "state": state,
                    "last_metric_step": status.get("last_metric_step"),
                    "fingerprint": cell.get("fingerprint"),
                    "run_dir": str(run_dir) if run_dir else None,
                    "metrics_file": status.get("metrics_file"),
                    "environment": dict(cell.get("environment", {})),
                    "evaluations": evaluations,
                }
            )
    return observed


def collect_model_evaluations(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.is_dir():
        return rows
    for manifest_path in sorted(root.glob("*/target_manifest.json")):
        manifest = read_object(manifest_path)
        summary_path = manifest_path.parent / "summary.json"
        status_path = manifest_path.parent / "status.json"
        summary = read_object(summary_path) if summary_path.is_file() else None
        status = read_object(status_path) if status_path.is_file() else None
        check = validate_paper_evaluation(
            summary_path,
            protocol="paper",
            manifest_path=manifest_path,
            status_path=status_path,
        ) if summary else None
        rows.append(
            {
                "target_id": manifest.get("target_id"),
                "model_identity": f"{manifest.get('model')}@{manifest.get('revision')}",
                "state": status.get("state") if status else "missing",
                "paper_comparable": bool(check and check.paper_comparable),
                "paper_comparability_reason": (
                    check.reason if check else "missing completed evaluation summary"
                ),
                "n": summary.get("n") if summary else None,
                "summary_path": str(summary_path) if summary else None,
            }
        )
    return rows


def _best_completed(runs: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    complete = [run for run in runs if run.get("state") == "completed"]
    if not complete:
        return None
    return max(complete, key=lambda run: PROTOCOL_RANK.get(str(run.get("protocol")), 0))


def _metric_value_valid(key: str, value: Any) -> bool:
    if key in SCHEMA_METRICS:
        return type(value) is int and value == 1
    if key in ARRAY_METRICS:
        return (
            isinstance(value, list)
            and bool(value)
            and all(
                isinstance(item, (int, float))
                and not isinstance(item, bool)
                and math.isfinite(float(item))
                for item in value
            )
        )
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _read_metric_rows(run: Mapping[str, Any]) -> tuple[list[dict[str, Any]], str | None]:
    raw_path = run.get("metrics_file")
    if not isinstance(raw_path, str) or not raw_path:
        return [], "completed status does not identify a metrics file"
    path = Path(raw_path).expanduser()
    if not path.is_file():
        return [], f"metrics file is missing: {path}"
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, Mapping):
                    return [], f"{path}:{line_number} is not a JSON object"
                data = payload.get("data", payload)
                if not isinstance(data, Mapping):
                    return [], f"{path}:{line_number} metric data is not an object"
                row = dict(data)
                row["__step__"] = payload.get("step", data.get("step"))
                rows.append(row)
    except (json.JSONDecodeError, OSError) as exc:
        return [], f"cannot read metrics file {path}: {exc}"
    if not rows:
        return [], f"metrics file is empty: {path}"
    return rows, None


def _producer_metric_contract(
    entry_id: str, run: Mapping[str, Any]
) -> tuple[bool, str]:
    required = METRIC_REQUIREMENTS.get(entry_id, ())
    if not required:
        return True, "producer has no registered training-metric contract"
    rows, error = _read_metric_rows(run)
    if error:
        return False, error
    missing = [
        key
        for key in required
        if not any(key in row and _metric_value_valid(key, row[key]) for row in rows)
    ]
    if missing:
        return False, f"missing valid producer metrics: {missing}"
    return True, "all registered producer metrics are present and finite"


def _integer_setting(environment: Mapping[str, Any], key: str) -> int | None:
    value = environment.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _position_entropy_branches(run: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "position_text_condition_complete": False,
        "position_figure_window_complete": False,
        "position_evidence_label": "incomplete",
    }
    rows, error = _read_metric_rows(run)
    if error:
        result["position_evidence_label"] = error
        return result
    environment = run.get("environment")
    if not isinstance(environment, Mapping):
        result["position_evidence_label"] = "suite cell has no environment"
        return result
    expected = {
        "POSITION_ENTROPY_START_STEP": 180,
        "POSITION_ENTROPY_LOG_FREQ": 10,
        "POSITION_ENTROPY_BIN_SIZE": 256,
        "MAX_RESPONSE_LENGTH": 15_360,
    }
    if any(_integer_setting(environment, key) != value for key, value in expected.items()):
        return result
    position_rows = [
        row
        for row in rows
        if _metric_value_valid(
            "opd/position_entropy_schema_version",
            row.get("opd/position_entropy_schema_version"),
        )
        and all(
            _metric_value_valid(key, row.get(key))
            for key in (
                "position-entropy/student_mean_by_bin",
                "position-entropy/teacher_mean_by_bin",
                "position-entropy/token_count_by_bin",
            )
        )
    ]
    expected_bins = 15_360 // 256
    position_rows = [
        row
        for row in position_rows
        if all(
            len(row[key]) == expected_bins
            for key in (
                "position-entropy/student_mean_by_bin",
                "position-entropy/teacher_mean_by_bin",
                "position-entropy/token_count_by_bin",
            )
        )
    ]
    steps = {row.get("__step__") for row in position_rows}
    total_steps = _integer_setting(environment, "TOTAL_TRAINING_STEPS")
    if run.get("protocol") == "paper" and total_steps == 200 and steps == {180, 190, 200}:
        result.update(
            position_text_condition_complete=True,
            position_evidence_label="text-aligned-only",
        )
    elif total_steps == 260 and steps == set(range(180, 261, 10)):
        result.update(
            position_text_condition_complete=True,
            position_figure_window_complete=True,
            position_evidence_label=(
                "text-boundary-plus-undisclosed-figure-window-reconstruction"
            ),
        )
    return result


def _upstream_details(
    entry_id: str, upstream_runs: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    required = UPSTREAM_REQUIREMENTS.get(entry_id, ())
    empty = {
        "upstream_required_stages": list(required),
        "upstream_observed_stages": [],
        "upstream_completed_stages": [],
        "upstream_root": None,
        "upstream_protocol": None,
        "upstream_seed": None,
        "upstream_paper_comparable_count": 0,
        "upstream_comparability_reasons": [],
    }
    if not required:
        return empty
    grouped: dict[tuple[Any, ...], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for run in upstream_runs:
        stage = run.get("stage")
        if stage not in required:
            continue
        key = (
            run.get("upstream_root"),
            run.get("suite_id"),
            run.get("protocol"),
            run.get("seed"),
        )
        if stage in grouped[key]:
            raise ValueError(f"duplicate upstream stage {stage} in coherent suite {key}")
        grouped[key][str(stage)] = run
    if not grouped:
        return empty

    def rank(item: tuple[tuple[Any, ...], Mapping[str, Mapping[str, Any]]]) -> tuple[int, int, int]:
        key, stages = item
        completed = sum(row.get("state") == "completed" for row in stages.values())
        present = len(stages)
        protocol_rank = UPSTREAM_PROTOCOL_RANK.get(str(key[2]), 0)
        return completed, present, protocol_rank

    selected_key, selected = max(grouped.items(), key=rank)
    completed_stages = sorted(
        stage for stage, run in selected.items() if run.get("state") == "completed"
    )
    return {
        "upstream_required_stages": list(required),
        "upstream_observed_stages": sorted(selected),
        "upstream_completed_stages": completed_stages,
        "upstream_root": selected_key[0],
        "upstream_protocol": selected_key[2],
        "upstream_seed": selected_key[3],
        "upstream_paper_comparable_count": sum(
            run.get("paper_comparable") is True for run in selected.values()
        ),
        "upstream_comparability_reasons": sorted(
            f"{stage}: {run.get('paper_comparability_reason', 'missing reason')}"
            for stage, run in selected.items()
            if not run.get("paper_comparable")
        ),
    }


def observed_state(
    entry: Mapping[str, Any],
    suite_runs: Mapping[str, Sequence[Mapping[str, Any]]],
    model_evals: Sequence[Mapping[str, Any]],
    probe_roots: Sequence[Path],
    upstream_runs: Sequence[Mapping[str, Any]] = (),
) -> tuple[str, dict[str, Any]]:
    producer = entry["producer"]
    cell_ids = producer["cell_ids"]
    best = {cell_id: _best_completed(suite_runs.get(cell_id, [])) for cell_id in cell_ids}
    completed = {cell_id: run for cell_id, run in best.items() if run is not None}
    metric_checks = {
        cell_id: _producer_metric_contract(str(entry["id"]), run)
        for cell_id, run in completed.items()
        if run is not None
    }
    producer_validated = {
        cell_id: run
        for cell_id, run in completed.items()
        if metric_checks.get(cell_id, (False, "missing contract"))[0]
    }
    paper_trained = {
        cell_id: run
        for cell_id, run in producer_validated.items()
        if run and run.get("protocol") == "paper"
    }
    paper_evaluated = {
        cell_id: run
        for cell_id, run in producer_validated.items()
        if run
        and run.get("protocol") == "paper"
        and any(evaluation.get("paper_comparable") for evaluation in run["evaluations"])
    }
    models = producer.get("models", [])
    matching_models = [
        row for row in model_evals if row.get("model_identity") in set(models)
    ]
    matched_models = [row for row in matching_models if row.get("state") == "completed"]
    paper_models = [row for row in matched_models if row.get("paper_comparable")]
    probe_names = {
        "figure-11": {"teacher_continuation_summary.json"},
        "figure-14": {"sequence_reward_summary.json"},
    }.get(str(entry["id"]), set())
    probe_files = sorted(
        str(path)
        for root in probe_roots
        if root.is_dir()
        for path in root.rglob("*.json")
        if path.name in probe_names
        or entry["id"].replace("figure-", "fig") in path.name.lower()
    )

    details = {
        "cell_count": len(cell_ids),
        "completed_cell_count": len(completed),
        "completed_cell_ids": sorted(completed),
        "producer_validated_cell_count": len(producer_validated),
        "producer_validated_cell_ids": sorted(producer_validated),
        "producer_metric_reasons": sorted(
            f"{cell_id}: {reason}"
            for cell_id, (valid, reason) in metric_checks.items()
            if not valid
        ),
        "paper_trained_cell_count": len(paper_trained),
        "paper_evaluated_cell_count": len(paper_evaluated),
        "completed_protocols": sorted(
            {str(run["protocol"]) for run in completed.values() if run},
            key=lambda value: PROTOCOL_RANK[value],
        ),
        "model_count": len(models),
        "completed_model_eval_count": len(matched_models),
        "paper_model_eval_count": len(paper_models),
        "evaluation_comparability_reasons": sorted(
            {
                f"{cell_id}@{evaluation.get('step')}: "
                f"{evaluation.get('paper_comparability_reason', 'missing reason')}"
                for cell_id, run in completed.items()
                if run
                for evaluation in run.get("evaluations", [])
                if not evaluation.get("paper_comparable")
            }
        ),
        "model_evaluation_comparability_reasons": sorted(
            {
                f"{row.get('target_id')}: "
                f"{row.get('paper_comparability_reason', 'missing reason')}"
                for row in matching_models
                if not row.get("paper_comparable")
            }
        ),
        "probe_files": probe_files,
        **_upstream_details(str(entry["id"]), upstream_runs),
    }
    if str(entry["id"]) in {"figure-13", "figure-23"} and completed:
        position = _position_entropy_branches(next(iter(completed.values())))
        details.update(position)
    else:
        details.update(
            position_text_condition_complete=False,
            position_figure_window_complete=False,
            position_evidence_label=None,
        )
    declared = str(entry["status"])
    if declared.startswith("blocked"):
        if completed or paper_models or probe_files:
            return "blocked_with_partial_artifacts", details
        return "blocked", details
    upstream_required = details["upstream_required_stages"]
    if upstream_required:
        upstream_observed = details["upstream_observed_stages"]
        upstream_completed = details["upstream_completed_stages"]
        upstream_protocol = details["upstream_protocol"]
        if len(upstream_completed) == len(upstream_required):
            if (
                upstream_protocol == "paper"
                and details["upstream_paper_comparable_count"] == len(upstream_required)
            ):
                return "upstream_paper_complete", details
            return f"upstream_{upstream_protocol}_complete", details
        if upstream_observed:
            return f"upstream_{upstream_protocol}_partial", details
        # A component-model evaluation is not evidence that the Table 1/3
        # producer ran.  Keep its count/reason in the row, but do not use it to
        # upgrade an absent upstream pipeline.
        return "upstream_not_started", details
    if cell_ids:
        if len(paper_evaluated) == len(cell_ids):
            return "paper_evaluated", details
        if len(paper_trained) == len(cell_ids):
            if str(entry["id"]) in {"figure-13", "figure-23"}:
                if details["position_figure_window_complete"]:
                    return "figure_window_reconstruction_complete", details
                if details["position_text_condition_complete"]:
                    return "text_condition_complete_figure_window_missing", details
            return "paper_trained_not_fully_evaluated", details
        if len(producer_validated) == len(cell_ids):
            if str(entry["id"]) in {"figure-13", "figure-23"}:
                if details["position_figure_window_complete"]:
                    return "figure_window_reconstruction_complete", details
                return "position_entropy_training_only", details
            strongest = min(
                (
                    PROTOCOL_RANK[str(run["protocol"])]
                    for run in producer_validated.values()
                    if run
                ),
                default=0,
            )
            label = next((name for name, rank in PROTOCOL_RANK.items() if rank == strongest), "unknown")
            return f"{label}_training_complete", details
        if len(completed) == len(cell_ids):
            return "producer_artifact_incomplete", details
        if completed:
            return "partially_run", details
    if models:
        if len(paper_models) == len(models):
            return "paper_model_evaluation_complete", details
        if matched_models:
            return "smoke_model_evaluation_only", details
    if probe_files:
        return "probe_artifacts_present", details
    if declared in {"implemented", "released_checkpoint_available"}:
        return declared, details
    return "not_started", details


def build_rows(
    entries: Sequence[Mapping[str, Any]],
    known_cells: Mapping[str, Mapping[str, Any]],
    suite_runs: Mapping[str, Sequence[Mapping[str, Any]]],
    model_evals: Sequence[Mapping[str, Any]],
    probe_roots: Sequence[Path],
    upstream_runs: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in entries:
        state, details = observed_state(
            entry, suite_runs, model_evals, probe_roots, upstream_runs
        )
        cells = entry["producer"]["cell_ids"]
        rows.append(
            {
                "entry_id": entry["id"],
                "paper_location": entry["paper_location"],
                "producer_type": entry["producer"]["type"],
                "declared_status": entry["status"],
                "observed_status": state,
                "fidelity": entry["fidelity"],
                "blocker": entry.get("blocker"),
                "cell_ids": cells,
                "disabled_cell_ids": [
                    cell_id for cell_id in cells if known_cells[cell_id]["disabled_by_default"]
                ],
                **details,
            }
        )
    return rows


def markdown_report(rows: Sequence[Mapping[str, Any]], total_cells: int) -> str:
    observed_cells = {
        cell_id
        for row in rows
        for cell_id in row["completed_cell_ids"]
    }
    paper_entries = sum(
        row["observed_status"]
        in {"paper_evaluated", "paper_model_evaluation_complete", "upstream_paper_complete"}
        for row in rows
    )
    # A blocked producer remains a blocked entry even when partial smoke
    # artifacts exist.  Counting only observed_status == "blocked" made the
    # number decrease as unrelated partial evidence arrived.
    blocked = sum(str(row["declared_status"]).startswith("blocked") for row in rows)
    lines = [
        "# 论文完整复现覆盖台账（动态）",
        "",
        f"- 结构覆盖：{len(rows)}/26（23 张图 + 3 张表）",
        f"- 训练轨迹覆盖：{len(observed_cells)}/{total_cells} 至少完成一种协议",
        f"- 论文可比结果条目：{paper_entries}/26",
        f"- 当前硬阻塞条目：{blocked}/26",
        "",
        "| 条目 | 动态状态 | 已跑/所需轨迹 | 论文训练 | avg@16 | 声明保真度 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['entry_id']} | {row['observed_status']} | "
            f"{row['completed_cell_count']}/{row['cell_count']} | "
            f"{row['paper_trained_cell_count']} | {row['paper_evaluated_cell_count']} | "
            f"{str(row['fidelity']).replace('|', '/')} |"
        )
    lines.extend(
        [
            "",
            "> `smoke` 只证明执行、形状和指标链路；只有 paper 协议训练并完成论文规定的 avg@16，才计入论文可比结果。",
            "",
        ]
    )
    rejected = [
        (str(row["entry_id"]), reason)
        for row in rows
        for reason in (
            list(row.get("evaluation_comparability_reasons", []))
            + list(row.get("model_evaluation_comparability_reasons", []))
            + list(row.get("upstream_comparability_reasons", []))
        )
    ]
    if rejected:
        lines.extend(["## 未升级为论文可比的评测证据", ""])
        for entry_id, reason in rejected:
            lines.append(f"- `{entry_id}`: {reason}")
        lines.append("")
    return "\n".join(lines)


def training_matrix_rows(
    known_cells: Mapping[str, Mapping[str, Any]],
    suite_runs: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell_id, cell in known_cells.items():
        environment = cell["environment"]
        paper_overrides = cell.get("protocol_overrides", {}).get("paper", {})
        paper_steps = paper_overrides.get("TOTAL_TRAINING_STEPS")
        best = _best_completed(suite_runs.get(cell_id, []))
        if cell.get("allowed_protocols") == ["paper"]:
            local_budget = "not enabled on 2xA100"
        else:
            local_budget = "pilot=200 steps; calibration=10; smoke=2"
        rows.append(
            {
                "group_id": cell["group_id"],
                "cell_id": cell_id,
                "paper_location": cell.get("paper_location"),
                "label": cell.get("label"),
                "student_model": environment.get("STUDENT_MODEL"),
                "student_revision": environment.get("STUDENT_REVISION"),
                "teacher_model": environment.get("TEACHER_MODEL"),
                "teacher_revision": environment.get("TEACHER_REVISION"),
                "train_data": environment.get("TRAIN_DATA"),
                "response_length": environment.get("MAX_RESPONSE_LENGTH", "7168"),
                "top_k": environment.get("TOP_K"),
                "top_k_strategy": environment.get("TOP_K_STRATEGY"),
                "thinking": environment.get("ENABLE_THINKING"),
                "paper_budget": f"{paper_steps} steps" if paper_steps else "1 epoch",
                "local_2xa100_budget": local_budget,
                "disabled_by_default": bool(cell.get("disabled_by_default")),
                "allowed_protocols": cell.get("allowed_protocols") or [
                    "smoke", "calibration", "pilot", "paper"
                ],
                "best_completed_protocol": best.get("protocol") if best else None,
                "best_completed_step": best.get("last_metric_step") if best else None,
                "fidelity": cell.get("fidelity"),
                "disclosure": cell.get("disclosure"),
                "full_environment": dict(environment),
            }
        )
    return rows


def training_matrix_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    enabled = sum(not row["disabled_by_default"] for row in rows)
    canonical = lambda payload: json.dumps(  # noqa: E731 - compact local identity helper
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    complete_environments = {
        canonical(row["full_environment"])
        for row in rows
    }
    # MIN_FREE_GIB is a preflight admission threshold.  It does not change the
    # model, data, optimization, sampling, or loss, so it must not manufacture
    # an additional scientific condition when the same cell is reused by
    # several paper figures.
    scientific_conditions = {
        canonical(
            {
                key: value
                for key, value in row["full_environment"].items()
                if key != "MIN_FREE_GIB"
            }
        )
        for row in rows
    }
    lines = [
        "# 完整 OPD 训练矩阵",
        "",
        (
            f"共 {len(rows)} 个注册 cell role，对应 {len(complete_environments)} 个完整 launcher env、"
            f"{len(scientific_conditions)} 个去除 MIN_FREE_GIB 后的科学训练条件；"
            f"默认可调度 {enabled} 个，门禁/禁用 {len(rows) - enabled} 个。"
        ),
        "",
        "| 组/Cell | 学生 → 教师 | 数据 | 长度 | 支持 | paper预算 | 本机状态 |",
        "|---|---|---|---:|---|---:|---|",
    ]
    for row in rows:
        student = str(row["student_model"]).split("/")[-1]
        teacher = str(row["teacher_model"]).split("/")[-1]
        data = Path(str(row["train_data"])).name
        support = f"{row['top_k_strategy']}@{row['top_k']}"
        if row["disabled_by_default"]:
            state = "gated"
        elif row["best_completed_protocol"]:
            state = f"{row['best_completed_protocol']} step {row['best_completed_step']}"
        else:
            state = "pending"
        lines.append(
            f"| {row['group_id']}/{row['cell_id']} | {student} → {teacher} | {data} | "
            f"{row['response_length']} | {support} | {row['paper_budget']} | {state} |"
        )
    lines.extend(
        [
            "",
            "> 默认的 2×A100 pilot/calibration/smoke 预算用于本机趋势与工程校验；它们不会被标记为论文数值复现。",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    total_cells: int,
    matrix_rows: Sequence[Mapping[str, Any]],
) -> None:
    serializable = list(rows)
    atomic_text(output_dir / "coverage.json", json.dumps(serializable, ensure_ascii=False, indent=2) + "\n")
    flat_rows = []
    for row in serializable:
        flat_rows.append(
            {
                **row,
                "cell_ids": ";".join(row["cell_ids"]),
                "completed_cell_ids": ";".join(row["completed_cell_ids"]),
                "disabled_cell_ids": ";".join(row["disabled_cell_ids"]),
                "completed_protocols": ";".join(row["completed_protocols"]),
                "evaluation_comparability_reasons": ";".join(
                    row["evaluation_comparability_reasons"]
                ),
                "model_evaluation_comparability_reasons": ";".join(
                    row["model_evaluation_comparability_reasons"]
                ),
                "upstream_required_stages": ";".join(row["upstream_required_stages"]),
                "upstream_observed_stages": ";".join(row["upstream_observed_stages"]),
                "upstream_completed_stages": ";".join(row["upstream_completed_stages"]),
                "upstream_comparability_reasons": ";".join(
                    row["upstream_comparability_reasons"]
                ),
                "probe_files": ";".join(row["probe_files"]),
            }
        )
    fields = sorted({field for row in flat_rows for field in row})
    temporary = output_dir / f".coverage.csv.tmp-{os.getpid()}"
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(flat_rows)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(output_dir / "coverage.csv")
    finally:
        temporary.unlink(missing_ok=True)
    atomic_text(output_dir / "coverage.md", markdown_report(rows, total_cells))
    atomic_text(
        output_dir / "training_matrix.json",
        json.dumps(list(matrix_rows), ensure_ascii=False, indent=2) + "\n",
    )
    matrix_flat = [
        {
            **row,
            "allowed_protocols": ";".join(row["allowed_protocols"]),
        }
        for row in matrix_rows
    ]
    matrix_fields = sorted({field for row in matrix_flat for field in row})
    matrix_temporary = output_dir / f".training_matrix.csv.tmp-{os.getpid()}"
    try:
        with matrix_temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=matrix_fields)
            writer.writeheader()
            writer.writerows(matrix_flat)
            handle.flush()
            os.fsync(handle.fileno())
        matrix_temporary.replace(output_dir / "training_matrix.csv")
    finally:
        matrix_temporary.unlink(missing_ok=True)
    atomic_text(output_dir / "training_matrix.md", training_matrix_markdown(matrix_rows))


def build_parser(repo_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix", type=Path, default=repo_root / "reproduce_4b/paper_full_matrix.json"
    )
    parser.add_argument(
        "--ledger", type=Path, default=repo_root / "reproduce_4b/paper_experiment_ledger.json"
    )
    parser.add_argument("--suite-root", action="append", type=Path, default=[])
    parser.add_argument(
        "--model-eval-root", type=Path, default=repo_root / "artifacts/evaluation/paper-models"
    )
    parser.add_argument(
        "--upstream-root", action="append", type=Path, default=[],
        help="Repeatable upstream suite root (or parent containing protocol/seed suites).",
    )
    parser.add_argument("--probe-root", action="append", type=Path, default=[])
    parser.add_argument(
        "--output-dir", type=Path, default=repo_root / "artifacts/paper_reproduction/coverage"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    args = build_parser(repo_root).parse_args(argv)
    try:
        matrix = read_object(args.matrix.expanduser().resolve())
        ledger = read_object(args.ledger.expanduser().resolve())
        known_cells = matrix_cells(matrix)
        entries = validate_ledger(ledger, known_cells)
        suites = collect_suites(args.suite_root)
        model_evals = collect_model_evaluations(args.model_eval_root.expanduser().resolve())
        upstream_runs = collect_upstream_roots(args.upstream_root)
        rows = build_rows(
            entries, known_cells, suites, model_evals, args.probe_root, upstream_runs
        )
        matrix_rows = training_matrix_rows(known_cells, suites)
        write_outputs(
            args.output_dir.expanduser().resolve(), rows, len(known_cells), matrix_rows
        )
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(
        f"entries={len(rows)} cells={len(known_cells)} "
        f"observed_entries={sum(row['observed_status'] != 'not_started' for row in rows)} "
        f"output={args.output_dir.expanduser().resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
