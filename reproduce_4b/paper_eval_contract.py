#!/usr/bin/env python3
"""Fail-closed validation for paper-comparable three-benchmark evaluations.

An ``avg@16`` label alone is not evidence that the paper workload ran.  This
module validates the immutable evaluation manifest, completion status, summary,
official benchmark cardinalities, and every response's sampling metadata before
allowing downstream coverage/collector tools to call an evaluation comparable.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


PAPER_BENCHMARKS = ("AIME24", "AIME25", "AMC23")
PAPER_BENCHMARK_SPECS: dict[str, dict[str, Any]] = {
    "AIME24": {
        "rows": 30,
        "sha256": "dc3ae15e0236828eb95033a68aab60e19da7ea1afdaad80e26924e5f82d3d96f",
    },
    "AIME25": {
        "rows": 30,
        "sha256": "14ee9a0e8d79cba76c62247bc5ebee1647e023eccf60e31994e279113777dad2",
    },
    "AMC23": {
        "rows": 83,
        "sha256": "0f933978194b73fc091bd667af7d9056d3d745c9d9e3577b5adcad77f8f1ae7c",
    },
}
PAPER_N = 16
PAPER_MAX_TOKENS = 31_744
PAPER_TEMPERATURE = 0.7
PAPER_TOP_P = 0.95


@dataclass(frozen=True)
class PaperEvaluationValidation:
    paper_comparable: bool
    reason: str
    evaluation_protocol_comparable: bool = False
    training_comparability: str | None = None


def _fail(reason: str) -> PaperEvaluationValidation:
    return PaperEvaluationValidation(False, reason)


def _declares_exact_evaluation(payload: Mapping[str, Any]) -> bool:
    return (
        payload.get("paper_comparable") is True
        or payload.get("paper_evaluation_protocol") is True
    )


def _read_object(path: Path, label: str) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, f"missing {label}: {path}"
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (json.JSONDecodeError, OSError) as exc:
        return None, f"invalid {label} {path}: {exc}"
    if not isinstance(payload, dict):
        return None, f"{label} must be a JSON object: {path}"
    return payload, None


def _same_path(raw: Any, expected: Path) -> bool:
    if not isinstance(raw, str) or not raw:
        return False
    return Path(raw).expanduser().resolve() == expected.resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_number(actual: Any, expected: float) -> bool:
    return (
        isinstance(actual, (int, float))
        and not isinstance(actual, bool)
        and math.isfinite(float(actual))
        and math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
    )


def _validate_status(
    status: Mapping[str, Any],
    summary_path: Path,
    *,
    plan_identity: Mapping[str, Any] | None = None,
    planned_summaries: set[Path] | None = None,
    planned_targets: Mapping[str, Path] | None = None,
) -> PaperEvaluationValidation | None:
    if status.get("state") != "completed":
        return _fail(f"evaluation status is {status.get('state')!r}, expected 'completed'")
    if plan_identity is not None:
        if status.get("evaluation_plan") != plan_identity:
            return _fail("evaluation status is linked to a different evaluation plan")
        summaries = status.get("summaries")
        if not isinstance(summaries, list):
            return _fail("checkpoint-grid evaluation status lacks summaries")
        resolved: list[Path] = []
        for raw in summaries:
            if not isinstance(raw, str) or not raw:
                return _fail("checkpoint-grid evaluation status has an invalid summary path")
            resolved.append(Path(raw).expanduser().resolve())
        if len(resolved) != len(set(resolved)):
            return _fail("checkpoint-grid evaluation status repeats a summary")
        if planned_summaries is None or set(resolved) != planned_summaries:
            return _fail("completed evaluation status does not exactly cover the registered checkpoint grid")
        targets = status.get("targets")
        if (
            not isinstance(targets, Mapping)
            or planned_targets is None
            or set(targets) != set(planned_targets)
        ):
            return _fail("completed evaluation status lacks one state entry per registered target")
        for target_id, target in targets.items():
            if (
                not isinstance(target, Mapping)
                or target.get("state") != "completed"
                or not _same_path(target.get("summary"), planned_targets[str(target_id)])
            ):
                return _fail("completed evaluation status contains a non-completed target")
        if summary_path.resolve() not in planned_summaries:
            return _fail("evaluation status does not list this checkpoint summary")
        return None
    if "summary" in status:
        if not _same_path(status.get("summary"), summary_path):
            return _fail("evaluation status does not identify this summary")
        return None
    summaries = status.get("summaries")
    if not isinstance(summaries, list) or not any(_same_path(item, summary_path) for item in summaries):
        return _fail("evaluation status does not list this checkpoint summary")
    return None


def _validate_manifest(
    manifest: Mapping[str, Any], summary: Mapping[str, Any]
) -> PaperEvaluationValidation | None:
    if manifest.get("schema_version") not in (1, 2):
        return _fail("evaluation manifest has an unsupported schema_version")
    kind = manifest.get("kind")
    if kind not in {"immutable_model_baseline", "cell_checkpoint"}:
        return _fail(f"evaluation manifest kind is unsupported: {kind!r}")
    if summary.get("schema_version") != manifest.get("schema_version"):
        return _fail("evaluation summary/manifest schema_version mismatch")
    for key in ("target_id", "model", "tokenizer"):
        if not isinstance(manifest.get(key), str) or not manifest[key]:
            return _fail(f"evaluation manifest lacks a non-empty {key}")
    for key in ("revision", "tokenizer_revision"):
        if key not in manifest or (
            manifest.get(key) is not None and not isinstance(manifest.get(key), str)
        ):
            return _fail(f"evaluation manifest has an invalid or implicit {key}")
    if kind == "immutable_model_baseline":
        model_is_local = Path(str(manifest["model"])).expanduser().is_dir()
        tokenizer_is_local = Path(str(manifest["tokenizer"])).expanduser().is_dir()
        if manifest.get("revision") is None and not model_is_local:
            return _fail("remote baseline model lacks an immutable revision")
        if manifest.get("tokenizer_revision") is None and not tokenizer_is_local:
            return _fail("remote baseline tokenizer lacks an immutable revision")
    if not _declares_exact_evaluation(manifest):
        return _fail(
            "evaluation manifest does not explicitly assert a full paper evaluation protocol"
        )
    if "limit" not in manifest or manifest.get("limit") is not None:
        return _fail(f"evaluation manifest limit is {manifest.get('limit')!r}, expected explicit null")
    sampling = manifest.get("sampling")
    if not isinstance(sampling, Mapping):
        return _fail("evaluation manifest lacks a sampling object")
    expected_sampling: tuple[tuple[str, Any], ...] = (
        ("n", PAPER_N),
        ("max_tokens", PAPER_MAX_TOKENS),
        ("temperature", PAPER_TEMPERATURE),
        ("top_p", PAPER_TOP_P),
        ("thinking", "off"),
    )
    for key, expected in expected_sampling:
        actual = sampling.get(key)
        matches = _exact_number(actual, float(expected)) if isinstance(expected, float) else actual == expected
        if not matches:
            return _fail(f"evaluation manifest sampling.{key}={actual!r}, expected {expected!r}")
    seed = sampling.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        return _fail("evaluation manifest sampling.seed must be an explicit integer")
    for key in ("model", "revision", "tokenizer", "tokenizer_revision"):
        if sampling.get(key) != manifest.get(key):
            return _fail(f"evaluation manifest sampling.{key} does not match manifest {key}")

    benchmarks = manifest.get("benchmarks")
    if not isinstance(benchmarks, Mapping) or set(benchmarks) != set(PAPER_BENCHMARKS):
        return _fail(f"evaluation manifest must contain exactly {PAPER_BENCHMARKS}")
    for benchmark in PAPER_BENCHMARKS:
        observed = benchmarks.get(benchmark)
        expected = PAPER_BENCHMARK_SPECS[benchmark]
        if not isinstance(observed, Mapping):
            return _fail(f"evaluation manifest has malformed {benchmark} metadata")
        if observed.get("rows") != expected["rows"]:
            return _fail(
                f"evaluation manifest {benchmark}.rows={observed.get('rows')!r}, "
                f"expected {expected['rows']}"
            )
        if observed.get("sha256") != expected["sha256"]:
            return _fail(f"evaluation manifest {benchmark} dataset SHA-256 is not the pinned official hash")
        raw_path = observed.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            return _fail(f"evaluation manifest {benchmark} lacks a dataset path")
        dataset_path = Path(raw_path).expanduser()
        if not dataset_path.is_file():
            return _fail(f"evaluation manifest {benchmark} dataset is missing: {dataset_path}")
        try:
            digest = _sha256(dataset_path)
        except OSError as exc:
            return _fail(f"cannot hash {benchmark} dataset {dataset_path}: {exc}")
        if digest != expected["sha256"]:
            return _fail(f"on-disk {benchmark} dataset SHA-256 differs from the pinned official hash")

    if not _declares_exact_evaluation(summary):
        return _fail(
            "evaluation summary does not explicitly assert a full paper evaluation protocol"
        )
    if (
        summary.get("paper_evaluation_protocol")
        != manifest.get("paper_evaluation_protocol")
    ):
        return _fail("evaluation summary/manifest protocol-comparability mismatch")
    for key in ("target_id", "model", "revision"):
        if key not in summary or summary.get(key) != manifest.get(key):
            return _fail(f"evaluation summary/manifest mismatch for {key}")
    if kind == "cell_checkpoint":
        step = manifest.get("checkpoint_step")
        if not isinstance(step, int) or isinstance(step, bool) or step <= 0:
            return _fail("checkpoint target manifest has an invalid checkpoint_step")
        if summary.get("kind") != "cell_checkpoint" or summary.get("checkpoint_step") != step:
            return _fail("checkpoint summary/manifest identity mismatch")
        plan_ref = manifest.get("evaluation_plan")
        if (
            not isinstance(plan_ref, Mapping)
            or summary.get("evaluation_plan_sha256") != plan_ref.get("sha256")
        ):
            return _fail("checkpoint summary/manifest evaluation-plan identity mismatch")
    return None


def _resolve_response_path(summary_path: Path, raw: Any) -> Path | None:
    if not isinstance(raw, str) or not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = summary_path.parent / path
    return path.resolve()


def _resolve_declared_path(anchor: Path, raw: Any) -> Path | None:
    if not isinstance(raw, str) or not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = anchor.parent / path
    return path.resolve()


def _validate_checkpoint_plan(
    summary_path: Path,
    manifest_path: Path,
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, set[Path] | None, PaperEvaluationValidation | None]:
    expected_manifest_path = summary_path.parent / "target_manifest.json"
    if manifest_path.resolve() != expected_manifest_path.resolve():
        return None, None, None, _fail("checkpoint target manifest is not adjacent to its summary")
    plan_ref = manifest.get("evaluation_plan")
    if not isinstance(plan_ref, Mapping):
        return None, None, None, _fail("checkpoint target manifest lacks evaluation_plan identity")
    plan_path = _resolve_declared_path(manifest_path, plan_ref.get("path"))
    expected_plan_path = summary_path.parent.parent / "evaluation_plan.json"
    if plan_path is None or plan_path != expected_plan_path.resolve():
        return None, None, None, _fail("checkpoint target manifest identifies the wrong evaluation plan path")
    plan, error = _read_object(plan_path, "evaluation plan")
    if error:
        return None, None, None, _fail(error)
    assert plan is not None
    try:
        observed_plan_sha = _sha256(plan_path)
    except OSError as exc:
        return None, None, None, _fail(f"cannot hash evaluation plan {plan_path}: {exc}")
    if plan_ref.get("sha256") != observed_plan_sha:
        return None, None, None, _fail("checkpoint target manifest evaluation-plan SHA-256 mismatch")
    plan_identity = {"path": str(plan_path), "sha256": observed_plan_sha}
    if plan.get("schema_version") != 2 or plan.get("kind") != "ablation_checkpoint_grid":
        return None, None, None, _fail("checkpoint evaluation plan has an unsupported schema/kind")
    if plan.get("selection") != "explicit":
        return None, None, None, _fail("paper checkpoint evaluation plan was not explicitly selected")
    if not _declares_exact_evaluation(plan) or plan.get("limit") is not None:
        return None, None, None, _fail("checkpoint evaluation plan is not an explicit full paper workload")
    training = plan.get("training")
    if not isinstance(training, Mapping) or training.get("protocol") not in {"paper", "scientific"}:
        return None, None, None, _fail(
            "checkpoint evaluation plan is not linked to paper/scientific training"
        )
    protocol = str(training["protocol"])
    if protocol == "scientific":
        comparability = training.get("comparability")
        if (
            not isinstance(comparability, Mapping)
            or comparability.get("evaluation") != "paper_evaluation_protocol"
            or plan.get("paper_evaluation_protocol") is not True
        ):
            return None, None, None, _fail(
                "scientific training does not register the exact paper evaluation protocol"
            )
    raw_run_dir = training.get("run_dir")
    cell_id = training.get("cell_id")
    if not isinstance(raw_run_dir, str) or not raw_run_dir:
        return None, None, None, _fail("checkpoint evaluation plan lacks its training run directory")
    if not isinstance(cell_id, str) or not cell_id:
        return None, None, None, _fail("checkpoint evaluation plan lacks its training cell identity")
    training_run_dir = Path(raw_run_dir).expanduser().resolve()
    evaluation_root = plan_path.parent
    evaluation_id = plan.get("evaluation_id", "default")
    if not isinstance(evaluation_id, str) or not evaluation_id:
        return None, None, None, _fail("evaluation plan has an invalid evaluation_id")
    expected_evaluation_root = training_run_dir / "evaluation"
    if evaluation_id != "default":
        expected_evaluation_root /= evaluation_id
    if evaluation_root != expected_evaluation_root.resolve():
        return None, None, None, _fail("evaluation plan is outside its registered training run")
    manifest_evaluation_id = manifest.get("evaluation_id", "default")
    if manifest_evaluation_id != evaluation_id:
        return None, None, None, _fail("checkpoint plan/manifest evaluation_id mismatch")
    if manifest.get("training") != training:
        return None, None, None, _fail("checkpoint target manifest/training plan identity mismatch")
    plan_sampling = plan.get("sampling")
    manifest_sampling = manifest.get("sampling")
    if not isinstance(plan_sampling, Mapping) or not isinstance(manifest_sampling, Mapping):
        return None, None, None, _fail("checkpoint plan/manifest lacks sampling metadata")
    for key in (
        "n",
        "temperature",
        "top_p",
        "max_tokens",
        "seed",
        "thinking",
        "tokenizer",
        "tokenizer_revision",
    ):
        if plan_sampling.get(key) != manifest_sampling.get(key):
            return None, None, None, _fail(f"checkpoint plan/manifest sampling mismatch for {key}")
    if plan.get("benchmarks") != manifest.get("benchmarks"):
        return None, None, None, _fail("checkpoint plan/manifest benchmark identity mismatch")

    steps = plan.get("checkpoint_steps")
    targets = plan.get("targets")
    if (
        not isinstance(steps, list)
        or not steps
        or any(not isinstance(step, int) or isinstance(step, bool) or step <= 0 for step in steps)
        or steps != sorted(set(steps))
    ):
        return None, None, None, _fail("evaluation plan checkpoint grid is not sorted and unique")
    if not isinstance(targets, list) or len(targets) != len(steps):
        return None, None, None, _fail("evaluation plan target count does not match checkpoint grid")

    planned_summaries: set[Path] = set()
    observed_steps: set[int] = set()
    observed_ids: set[str] = set()
    current_matches = 0
    for target in targets:
        if not isinstance(target, Mapping):
            return None, None, None, _fail("evaluation plan contains a malformed target")
        target_id = target.get("target_id")
        step = target.get("checkpoint_step")
        if not isinstance(target_id, str) or not target_id or target_id in observed_ids:
            return None, None, None, _fail("evaluation plan repeats or omits a target_id")
        if not isinstance(step, int) or isinstance(step, bool) or step not in steps or step in observed_steps:
            return None, None, None, _fail("evaluation plan repeats or misstates a checkpoint step")
        observed_ids.add(target_id)
        observed_steps.add(step)
        suffix = "" if evaluation_id == "default" else f"-{evaluation_id}"
        expected_target_id = f"{cell_id}-global-step-{step}{suffix}"
        if target_id != expected_target_id:
            return None, None, None, _fail("evaluation plan target_id does not match cell/checkpoint identity")
        target_root = evaluation_root / f"global_step_{step}"
        target_summary = _resolve_declared_path(plan_path, target.get("summary"))
        target_manifest = _resolve_declared_path(plan_path, target.get("target_manifest"))
        if target_summary != (target_root / "summary.json").resolve():
            return None, None, None, _fail("evaluation plan has a non-canonical summary path")
        if target_manifest != (target_root / "target_manifest.json").resolve():
            return None, None, None, _fail("evaluation plan has a non-canonical target-manifest path")
        source = _resolve_declared_path(plan_path, target.get("source_checkpoint"))
        allowed_sources = {
            (training_run_dir / "checkpoints" / f"global_step_{step}").resolve(),
            (training_run_dir / "milestones" / f"global_step_{step}").resolve(),
            (training_run_dir.parent / "milestones" / f"global_step_{step}").resolve(),
        }
        if source not in allowed_sources:
            return None, None, None, _fail(
                "evaluation plan has a non-canonical source-checkpoint/milestone path"
            )
        planned_summaries.add(target_summary)
        if target_id == manifest.get("target_id"):
            current_matches += 1
            if step != manifest.get("checkpoint_step"):
                return None, None, None, _fail("evaluation plan/current target checkpoint mismatch")
            if target_summary != summary_path.resolve() or target_manifest != manifest_path.resolve():
                return None, None, None, _fail("evaluation plan/current target artifact-path mismatch")
            manifest_source = _resolve_declared_path(manifest_path, manifest.get("source_checkpoint"))
            if source is None or source != manifest_source:
                return None, None, None, _fail("evaluation plan/current target source-checkpoint mismatch")
            expected_model = (training_run_dir / "merged" / f"global_step_{step}").resolve()
            if Path(str(manifest.get("model"))).expanduser().resolve() != expected_model:
                return None, None, None, _fail("checkpoint target manifest has a non-canonical merged-model path")
    if observed_steps != set(steps) or current_matches != 1:
        return None, None, None, _fail("evaluation plan does not identify the current target exactly once")
    return plan, plan_identity, planned_summaries, None


def _validate_responses(
    path: Path,
    benchmark: str,
    expected_prompts: int,
    expected_sampling: Mapping[str, Any],
) -> PaperEvaluationValidation | None:
    if not path.is_file():
        return _fail(f"missing response JSONL for {benchmark}: {path}")
    seen: set[tuple[int, int]] = set()
    example_by_row: dict[int, str] = {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, Mapping):
                    return _fail(f"{path}:{line_number}: response row is not an object")
                row_index = payload.get("row_index")
                rollout_id = payload.get("rollout_id")
                example_id = payload.get("example_id")
                if not isinstance(row_index, int) or isinstance(row_index, bool):
                    return _fail(f"{path}:{line_number}: invalid row_index")
                if not isinstance(rollout_id, int) or isinstance(rollout_id, bool):
                    return _fail(f"{path}:{line_number}: invalid rollout_id")
                if not isinstance(example_id, str) or not example_id:
                    return _fail(f"{path}:{line_number}: invalid example_id")
                if payload.get("data_source") != benchmark:
                    return _fail(f"{path}:{line_number}: data_source does not match {benchmark}")
                if not isinstance(payload.get("response"), str):
                    return _fail(f"{path}:{line_number}: response text is missing")
                key = (row_index, rollout_id)
                if key in seen:
                    return _fail(f"{path}:{line_number}: duplicate row_index/rollout_id {key}")
                seen.add(key)
                previous = example_by_row.setdefault(row_index, example_id)
                if previous != example_id:
                    return _fail(f"{path}:{line_number}: row_index maps to multiple example IDs")
                sampling = payload.get("sampling")
                if not isinstance(sampling, Mapping):
                    return _fail(f"{path}:{line_number}: missing sampling metadata")
                for name in (
                    "n",
                    "max_tokens",
                    "temperature",
                    "top_p",
                    "thinking",
                    "seed",
                    "model",
                    "revision",
                    "tokenizer",
                    "tokenizer_revision",
                ):
                    expected = expected_sampling.get(name)
                    actual = sampling.get(name)
                    matches = _exact_number(actual, float(expected)) if isinstance(expected, float) else actual == expected
                    if not matches:
                        return _fail(
                            f"{path}:{line_number}: sampling.{name}={actual!r}, expected {expected!r}"
                        )
    except (json.JSONDecodeError, OSError) as exc:
        return _fail(f"invalid response JSONL {path}: {exc}")

    expected_rows = set(range(expected_prompts))
    if set(example_by_row) != expected_rows:
        return _fail(f"{benchmark} response row_index set is not exactly 0..{expected_prompts - 1}")
    expected_rollouts = set(range(PAPER_N))
    for row_index in expected_rows:
        observed = {rollout_id for prompt, rollout_id in seen if prompt == row_index}
        if observed != expected_rollouts:
            return _fail(f"{benchmark} row {row_index} does not contain exactly rollout IDs 0..15")
    if len(seen) != expected_prompts * PAPER_N:
        return _fail(
            f"{benchmark} has {len(seen)} responses, expected {expected_prompts * PAPER_N}"
        )
    return None


def validate_paper_evaluation(
    summary_path: Path,
    *,
    protocol: str,
    manifest_path: Path | None,
    status_path: Path | None,
) -> PaperEvaluationValidation:
    """Return a reasoned, fail-closed paper-comparability decision."""

    summary_path = summary_path.expanduser().resolve()
    if protocol not in {"paper", "scientific"}:
        return _fail(
            f"suite/evaluation protocol is {protocol!r}, expected 'paper' or 'scientific'"
        )
    if manifest_path is None:
        return _fail("missing immutable evaluation manifest; summary alone cannot prove the paper sampling contract")
    if status_path is None:
        return _fail("missing evaluation completion status")

    summary, error = _read_object(summary_path, "evaluation summary")
    if error:
        return _fail(error)
    assert summary is not None
    manifest, error = _read_object(manifest_path.expanduser().resolve(), "evaluation manifest")
    if error:
        return _fail(error)
    assert manifest is not None
    status, error = _read_object(status_path.expanduser().resolve(), "evaluation status")
    if error:
        return _fail(error)
    assert status is not None

    failure = _validate_manifest(manifest, summary)
    if failure:
        return failure

    plan_identity: Mapping[str, Any] | None = None
    planned_summaries: set[Path] | None = None
    planned_targets: dict[str, Path] | None = None
    if manifest.get("kind") == "cell_checkpoint":
        plan, plan_identity, planned_summaries, failure = _validate_checkpoint_plan(
            summary_path,
            manifest_path.expanduser().resolve(),
            manifest,
        )
        if failure:
            return failure
        assert plan is not None
        planned_targets = {
            str(target["target_id"]): Path(str(target["summary"])).expanduser().resolve()
            for target in plan["targets"]
        }
    failure = _validate_status(
        status,
        summary_path,
        plan_identity=plan_identity,
        planned_summaries=planned_summaries,
        planned_targets=planned_targets,
    )
    if failure:
        return failure

    if summary.get("n") != PAPER_N:
        return _fail(f"evaluation summary n={summary.get('n')!r}, expected {PAPER_N}")
    benchmarks = summary.get("benchmarks")
    if not isinstance(benchmarks, Mapping) or set(benchmarks) != set(PAPER_BENCHMARKS):
        return _fail(f"evaluation summary must contain exactly {PAPER_BENCHMARKS}")
    benchmark_values: list[float] = []
    expected_sampling = manifest.get("sampling")
    assert isinstance(expected_sampling, Mapping)
    for benchmark in PAPER_BENCHMARKS:
        metrics = benchmarks.get(benchmark)
        expected_prompts = int(PAPER_BENCHMARK_SPECS[benchmark]["rows"])
        if not isinstance(metrics, Mapping):
            return _fail(f"evaluation summary has malformed {benchmark} metrics")
        avg = metrics.get(f"avg@{PAPER_N}")
        if not isinstance(avg, (int, float)) or isinstance(avg, bool) or not math.isfinite(float(avg)):
            return _fail(f"evaluation summary lacks finite {benchmark}/avg@{PAPER_N}")
        benchmark_values.append(float(avg))
        required_counts = {
            "n": PAPER_N,
            "num_prompts": expected_prompts,
            "num_complete_prompts": expected_prompts,
            "num_incomplete_prompts": 0,
            "num_responses": expected_prompts * PAPER_N,
        }
        for key, expected in required_counts.items():
            if metrics.get(key) != expected:
                return _fail(
                    f"evaluation summary {benchmark}.{key}={metrics.get(key)!r}, expected {expected}"
                )
        correct = metrics.get("num_correct")
        expected_responses = expected_prompts * PAPER_N
        if (
            not isinstance(correct, int)
            or isinstance(correct, bool)
            or not 0 <= correct <= expected_responses
        ):
            return _fail(f"evaluation summary {benchmark}.num_correct is invalid")
        if not math.isclose(
            float(avg),
            correct / expected_responses,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            return _fail(f"evaluation summary {benchmark}/avg@16 is inconsistent with num_correct")
        responses = _resolve_response_path(summary_path, metrics.get("input_jsonl"))
        if responses is None:
            return _fail(f"evaluation summary {benchmark} lacks input_jsonl provenance")
        expected_responses_path = summary_path.parent / benchmark / "responses.jsonl"
        if responses != expected_responses_path.resolve():
            return _fail(f"evaluation summary {benchmark} points outside its canonical target directory")
        try:
            responses_sha = _sha256(responses)
        except OSError as exc:
            return _fail(f"cannot hash response JSONL {responses}: {exc}")
        if metrics.get("input_jsonl_sha256") != responses_sha:
            return _fail(f"evaluation summary {benchmark} response SHA-256 provenance mismatch")
        grader = metrics.get("grader")
        expected_grader = Path(__file__).resolve().parents[1] / "scripts/val/eval/utils.py"
        if not isinstance(grader, Mapping):
            return _fail(f"evaluation summary {benchmark} lacks grader provenance")
        grader_path = _resolve_response_path(summary_path, grader.get("path"))
        if grader_path != expected_grader.resolve():
            return _fail(f"evaluation summary {benchmark} uses a different grader path")
        try:
            grader_sha = _sha256(expected_grader)
        except OSError as exc:
            return _fail(f"cannot hash repository grader {expected_grader}: {exc}")
        if grader.get("sha256") != grader_sha:
            return _fail(f"evaluation summary {benchmark} grader SHA-256 mismatch")
        failure = _validate_responses(
            responses,
            benchmark,
            expected_prompts,
            expected_sampling,
        )
        if failure:
            return failure

    macro = summary.get("benchmark_macro_mean_avg_at_n")
    expected_macro = sum(benchmark_values) / len(benchmark_values)
    if (
        not isinstance(macro, (int, float))
        or isinstance(macro, bool)
        or not math.isfinite(float(macro))
        or not math.isclose(float(macro), expected_macro, rel_tol=0.0, abs_tol=1e-12)
    ):
        return _fail("evaluation summary macro mean is inconsistent with the three benchmarks")
    if summary.get("aggregation") != "unweighted mean across AIME24, AIME25, AMC23":
        return _fail("evaluation summary has the wrong benchmark aggregation contract")

    training = manifest.get("training")
    training_label: str | None = None
    if isinstance(training, Mapping):
        comparability = training.get("comparability")
        if isinstance(comparability, Mapping):
            raw_label = comparability.get("training")
            training_label = str(raw_label) if raw_label is not None else None
    return PaperEvaluationValidation(
        paper_comparable=protocol == "paper",
        evaluation_protocol_comparable=True,
        training_comparability=training_label,
        reason=(
            "verified exact paper evaluation protocol, immutable no-limit manifest, official "
            "three-benchmark datasets/cardinalities, complete registered checkpoint grid where "
            "applicable, avg@16 arithmetic, and full response sampling identity; training "
            f"comparability={training_label or protocol}"
        ),
    )
