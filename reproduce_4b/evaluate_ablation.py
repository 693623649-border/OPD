#!/usr/bin/env python3
"""Register, merge, and evaluate an immutable checkpoint grid with avg@N.

Paper-protocol runs require explicit checkpoint steps on first use.  Their
evaluation plan, per-target manifests, generations, metrics, and summaries are
write-once; reruns validate and resume only missing work.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from run_ablations import acquire_gpu_lock, atomic_json, sha256_file, utc_now


BENCHMARKS = ("AIME24", "AIME25", "AMC23")
SCHEMA_VERSION = 2
PAPER_N = 16
PAPER_MAX_TOKENS = 31_744
PAPER_TEMPERATURE = 0.7
PAPER_TOP_P = 0.95
PAPER_AGGREGATION = "unweighted mean across AIME24, AIME25, AMC23"
DEFAULT_EVALUATION_ID = "default"
EVALUATION_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def evaluation_root(run_dir: Path, evaluation_id: str = DEFAULT_EVALUATION_ID) -> Path:
    """Return an immutable namespace for one checkpoint/sampling grid.

    Historical runs used ``run/evaluation`` directly.  Keep that layout for
    the default identity while scientific runs can register independent
    ``trend-n4`` and ``exact-avg16`` grids without overwriting one another.
    """

    if not EVALUATION_ID_RE.fullmatch(evaluation_id):
        raise ValueError(
            "evaluation ID must match [a-z0-9][a-z0-9-]{0,63}"
        )
    root = run_dir / "evaluation"
    return root if evaluation_id == DEFAULT_EVALUATION_ID else root / evaluation_id


def evaluation_status_path(
    run_dir: Path, evaluation_id: str = DEFAULT_EVALUATION_ID
) -> Path:
    if evaluation_id == DEFAULT_EVALUATION_ID:
        return run_dir / "evaluation_status.json"
    return evaluation_root(run_dir, evaluation_id) / "evaluation_status.json"


def read_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def write_once_json(path: Path, payload: Mapping[str, Any], label: str) -> bool:
    """Create an immutable JSON artifact, or validate an identical existing one.

    Returns ``True`` when a new file was committed and ``False`` when an
    identical artifact already existed.  The hard-link commit is exclusive:
    it cannot replace an artifact created by a concurrent or resumed process.
    """

    expected = dict(payload)
    if path.is_file():
        observed = read_object(path, label)
        if observed != expected:
            raise ValueError(f"existing {label} differs; choose a new evaluation identity: {path}")
        return False
    if path.exists():
        raise ValueError(f"{label} path exists but is not a regular file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.tmp-",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(expected, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            observed = read_object(path, label)
            if observed != expected:
                raise ValueError(
                    f"concurrently created {label} differs; choose a new evaluation identity: {path}"
                )
            return False
        return True
    finally:
        if temporary.exists():
            temporary.unlink()


def read_training_identity(run_dir: Path) -> dict[str, Any]:
    """Bind an attempt directory to its immutable suite/cell registration."""

    cell_root = run_dir.parent
    suite_root = cell_root.parent
    status_path = cell_root / "status.json"
    suite_path = suite_root / "suite_manifest.json"
    status = read_object(status_path, "training status")
    suite = read_object(suite_path, "suite manifest")
    legacy_state = status.get("state")
    execution_state = status.get("execution_state")
    if legacy_state != "completed" and execution_state not in {
        "training_complete",
        "evaluation_complete",
        "probe_complete",
        "rendered",
        "scientific_result_available",
    }:
        raise ValueError(
            "training status is not complete: "
            f"state={legacy_state!r}, execution_state={execution_state!r}"
        )
    raw_run_dir = status.get("run_dir")
    if not isinstance(raw_run_dir, str) or Path(raw_run_dir).expanduser().resolve() != run_dir:
        raise ValueError("training status does not identify the requested run directory")
    cells = suite.get("cells")
    if not isinstance(cells, list):
        raise ValueError(f"suite manifest has no cells list: {suite_path}")
    matching = [
        item
        for item in cells
        if isinstance(item, Mapping) and item.get("cell_id") == status.get("cell_id")
    ]
    if len(matching) != 1:
        raise ValueError("training cell is absent or duplicated in the suite manifest")
    cell = matching[0]
    if status.get("fingerprint") != cell.get("fingerprint"):
        raise ValueError("training status/suite cell fingerprint mismatch")
    protocol = suite.get("protocol")
    if not isinstance(protocol, str) or not protocol:
        raise ValueError("suite manifest lacks a protocol")
    comparability = status.get("comparability", cell.get("comparability"))
    contract_payload: Mapping[str, Any] | None = None
    contract_candidates = (
        run_dir / "run_contract.json",
        cell_root / "run_contract.json",
    )
    for contract_path in contract_candidates:
        if not contract_path.is_file():
            continue
        candidate = read_object(contract_path, "scientific run contract")
        if candidate.get("cell_id") not in {None, status.get("cell_id")}:
            raise ValueError("run contract/cell identity mismatch")
        if candidate.get("fingerprint") not in {None, status.get("fingerprint")}:
            raise ValueError("run contract/fingerprint mismatch")
        contract_payload = candidate
        comparability = candidate.get("comparability", comparability)
        break
    identity = {
        "suite_id": suite.get("suite_id"),
        "protocol": protocol,
        "seed": suite.get("seed"),
        "source_tree_sha256": suite.get("source_tree_sha256"),
        "matrix_sha256": suite.get("matrix_sha256"),
        "cell_id": status.get("cell_id"),
        "group_id": status.get("group_id"),
        "fingerprint": status.get("fingerprint"),
        "attempt": status.get("attempt"),
        "run_dir": str(run_dir),
    }
    if isinstance(comparability, Mapping):
        identity["comparability"] = dict(comparability)
    if contract_payload is not None:
        identity["run_contract_sha256"] = sha256_file(contract_path)
    return identity


def discover_checkpoints(run_dir: Path, requested: Sequence[int]) -> list[tuple[int, Path]]:
    discovered: dict[int, Path] = {}
    # Scientific runs keep recovery state at cell scope and archive permanent
    # model-only milestones before rolling checkpoints are pruned.  Prefer the
    # immutable milestones, then fall back to legacy attempt-local checkpoints.
    roots = (
        run_dir / "checkpoints",
        run_dir.parent / "checkpoints",
        run_dir / "milestones",
        run_dir.parent / "milestones",
    )
    for checkpoint_root in roots:
        for path in checkpoint_root.glob("global_step_*"):
            suffix = path.name.removeprefix("global_step_")
            if suffix.isdigit() and (path / "actor").is_dir():
                discovered[int(suffix)] = path
    if requested:
        missing = sorted(set(requested) - set(discovered))
        if missing:
            raise FileNotFoundError(f"checkpoint step(s) not found: {missing}")
        return [(step, discovered[step]) for step in sorted(set(requested))]
    if not discovered:
        raise FileNotFoundError(
            "no global_step_* actor checkpoints/milestones under "
            + ", ".join(str(root) for root in roots)
        )
    latest = max(discovered)
    return [(latest, discovered[latest])]


def read_student_identity(run_dir: Path) -> tuple[str, str | None]:
    path = run_dir / "model_snapshots.json"
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    student = manifest.get("student")
    if not isinstance(student, Mapping):
        raise ValueError(f"{path} has no student model identity")
    source = student.get("source")
    revision = student.get("resolved_revision")
    if not isinstance(source, str) or not source:
        raise ValueError(f"{path} has an invalid student source")
    if revision == "local":
        source = str(student.get("snapshot_path"))
        revision = None
    elif not isinstance(revision, str) or not revision:
        revision = None
    return source, revision


def sampling_contract(
    model: str,
    tokenizer: str,
    tokenizer_revision: str | None,
    n: int,
    max_tokens: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "n": n,
        "temperature": PAPER_TEMPERATURE,
        "top_p": PAPER_TOP_P,
        "max_tokens": max_tokens,
        "seed": seed,
        "thinking": "off",
        "model": model,
        "revision": None,
        "tokenizer": tokenizer,
        "tokenizer_revision": tokenizer_revision,
    }


def benchmark_contract(repo_root: Path) -> dict[str, dict[str, Any]]:
    benchmarks: dict[str, dict[str, Any]] = {}
    for benchmark in BENCHMARKS:
        path = (repo_root / "datasets/test_data" / benchmark / "test.parquet").resolve()
        benchmarks[benchmark] = {
            "path": str(path),
            "rows": parquet_rows(path, None),
            "sha256": sha256_file(path),
        }
    return benchmarks


def is_paper_comparable(protocol: str, n: int, max_tokens: int, limit: int | None) -> bool:
    return (
        protocol == "paper"
        and n == PAPER_N
        and max_tokens == PAPER_MAX_TOKENS
        and limit is None
    )


def is_paper_evaluation_protocol(
    training: Mapping[str, Any],
    n: int,
    max_tokens: int,
    limit: int | None,
) -> bool:
    """Check the evaluation workload independently from training fidelity."""

    exact_workload = n == PAPER_N and max_tokens == PAPER_MAX_TOKENS and limit is None
    if not exact_workload:
        return False
    if training.get("protocol") == "paper":
        return True
    comparability = training.get("comparability")
    return (
        isinstance(comparability, Mapping)
        and comparability.get("evaluation") == "paper_evaluation_protocol"
    )


def checkpoint_target_id(
    training: Mapping[str, Any],
    step: int,
    evaluation_id: str = DEFAULT_EVALUATION_ID,
) -> str:
    cell_id = training.get("cell_id")
    if not isinstance(cell_id, str) or not cell_id:
        raise ValueError("training identity lacks cell_id")
    suffix = "" if evaluation_id == DEFAULT_EVALUATION_ID else f"-{evaluation_id}"
    return f"{cell_id}-global-step-{step}{suffix}"


def build_evaluation_plan(
    repo_root: Path,
    run_dir: Path,
    checkpoints: Sequence[tuple[int, Path]],
    training: Mapping[str, Any],
    tokenizer: str,
    tokenizer_revision: str | None,
    n: int,
    max_tokens: int,
    seed: int,
    limit: int | None,
    *,
    evaluation_id: str = DEFAULT_EVALUATION_ID,
) -> dict[str, Any]:
    root = evaluation_root(run_dir, evaluation_id)
    steps = [step for step, _ in checkpoints]
    if steps != sorted(set(steps)) or not steps:
        raise ValueError("checkpoint grid must be non-empty, sorted, and unique")
    targets = []
    for step, checkpoint in checkpoints:
        target_root = root / f"global_step_{step}"
        targets.append(
            {
                "target_id": checkpoint_target_id(training, step, evaluation_id),
                "checkpoint_step": step,
                "source_checkpoint": str(checkpoint.resolve()),
                "target_manifest": str((target_root / "target_manifest.json").resolve()),
                "summary": str((target_root / "summary.json").resolve()),
            }
        )
    common_sampling = {
        "n": n,
        "temperature": PAPER_TEMPERATURE,
        "top_p": PAPER_TOP_P,
        "max_tokens": max_tokens,
        "seed": seed,
        "thinking": "off",
        "tokenizer": tokenizer,
        "tokenizer_revision": tokenizer_revision,
    }
    protocol = str(training.get("protocol"))
    paper_comparable = is_paper_comparable(protocol, n, max_tokens, limit)
    paper_eval_protocol = is_paper_evaluation_protocol(training, n, max_tokens, limit)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "ablation_checkpoint_grid",
        "training": dict(training),
        "checkpoint_steps": steps,
        "selection": "explicit" if protocol == "paper" else "explicit_or_latest",
        "sampling": common_sampling,
        "limit": limit,
        "paper_comparable": paper_comparable,
        "benchmarks": benchmark_contract(repo_root),
        "targets": targets,
    }
    if evaluation_id != DEFAULT_EVALUATION_ID:
        payload["evaluation_id"] = evaluation_id
    if protocol == "scientific":
        payload["paper_evaluation_protocol"] = paper_eval_protocol
        payload["comparability"] = dict(training.get("comparability", {}))
    return payload


def build_target_manifest(
    run_dir: Path,
    step: int,
    checkpoint: Path,
    training: Mapping[str, Any],
    plan_path: Path,
    plan_sha256: str,
    plan: Mapping[str, Any],
    tokenizer: str,
    tokenizer_revision: str | None,
    *,
    evaluation_id: str = DEFAULT_EVALUATION_ID,
) -> dict[str, Any]:
    merged = (run_dir / "merged" / f"global_step_{step}").resolve()
    target_id = checkpoint_target_id(training, step, evaluation_id)
    sampling_base = plan.get("sampling")
    if not isinstance(sampling_base, Mapping):
        raise ValueError("evaluation plan lacks sampling metadata")
    sampling = sampling_contract(
        str(merged),
        tokenizer,
        tokenizer_revision,
        int(sampling_base["n"]),
        int(sampling_base["max_tokens"]),
        int(sampling_base["seed"]),
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "target_id": target_id,
        "kind": "cell_checkpoint",
        "model": str(merged),
        "revision": None,
        "tokenizer": tokenizer,
        "tokenizer_revision": tokenizer_revision,
        "checkpoint_step": step,
        "source_checkpoint": str(checkpoint.resolve()),
        "training": dict(training),
        "evaluation_plan": {"path": str(plan_path.resolve()), "sha256": plan_sha256},
        "sampling": sampling,
        "limit": plan.get("limit"),
        "paper_comparable": plan.get("paper_comparable") is True,
        "benchmarks": plan.get("benchmarks"),
    }
    if evaluation_id != DEFAULT_EVALUATION_ID:
        payload["evaluation_id"] = evaluation_id
    if training.get("protocol") == "scientific":
        payload["paper_evaluation_protocol"] = (
            plan.get("paper_evaluation_protocol") is True
        )
        payload["comparability"] = dict(plan.get("comparability", {}))
    return payload


def validate_generation(
    path: Path,
    expected_prompts: int,
    n: int,
    expected_sampling: Mapping[str, Any],
) -> None:
    counts: Counter[str] = Counter()
    rollout_ids: dict[str, set[int]] = {}
    rows = 0
    model_values: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("example_id") is None:
                raise ValueError(f"{path}:{line_number}: missing example_id")
            example_id = str(payload["example_id"])
            rollout_id = payload.get("rollout_id")
            sampling = payload.get("sampling")
            if not isinstance(rollout_id, int) or isinstance(rollout_id, bool):
                raise ValueError(f"{path}:{line_number}: rollout_id must be an integer")
            if not isinstance(sampling, Mapping):
                raise ValueError(f"{path}:{line_number}: missing sampling metadata")
            for key, expected in expected_sampling.items():
                actual = sampling.get(key)
                if isinstance(expected, float):
                    if not isinstance(actual, (int, float)) or not math.isclose(float(actual), expected):
                        raise ValueError(f"{path}:{line_number}: sampling {key}={actual!r}, expected {expected}")
                elif actual != expected:
                    raise ValueError(f"{path}:{line_number}: sampling {key}={actual!r}, expected {expected!r}")
            model_values.add(str(sampling.get("model")))
            counts[example_id] += 1
            rollout_ids.setdefault(example_id, set()).add(rollout_id)
            rows += 1
    if len(counts) != expected_prompts or rows != expected_prompts * n:
        raise ValueError(
            f"{path}: rows/prompts={rows}/{len(counts)}, expected {expected_prompts * n}/{expected_prompts}"
        )
    expected_rollouts = set(range(n))
    malformed = [
        key
        for key in counts
        if counts[key] != n or rollout_ids[key] != expected_rollouts
    ]
    if malformed:
        raise ValueError(
            f"{path}: {len(malformed)} prompts lack exactly rollout IDs 0..{n - 1}"
        )
    if len(model_values) != 1:
        raise ValueError(f"{path}: generation mixes model identities: {sorted(model_values)}")


def parquet_rows(path: Path, limit: int | None) -> int:
    import pyarrow.parquet as pq

    rows = pq.ParquetFile(path).metadata.num_rows
    return min(rows, limit) if limit else rows


def aggregate_metrics(metrics_paths: Sequence[Path], n: int) -> dict[str, Any]:
    per_benchmark: dict[str, Any] = {}
    values: list[float] = []
    key = f"avg@{n}"
    for path in metrics_paths:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        value = payload.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"{path} has no finite {key}")
        benchmark = path.parent.name
        if benchmark in per_benchmark:
            raise ValueError(f"duplicate benchmark metrics for {benchmark}")
        per_benchmark[benchmark] = payload
        values.append(float(value))
    if set(per_benchmark) != set(BENCHMARKS):
        raise ValueError(
            f"expected benchmark set {list(BENCHMARKS)}, found {sorted(per_benchmark)}"
        )
    return {
        "n": n,
        "benchmark_macro_mean_avg_at_n": sum(values) / len(values),
        "aggregation": PAPER_AGGREGATION,
        "benchmarks": per_benchmark,
    }


def validate_metrics(
    path: Path,
    responses_path: Path,
    expected_prompts: int,
    n: int,
) -> None:
    payload = read_object(path, "benchmark metrics")
    expected_counts = {
        "n": n,
        "num_prompts": expected_prompts,
        "num_complete_prompts": expected_prompts,
        "num_incomplete_prompts": 0,
        "num_responses": expected_prompts * n,
    }
    for key, expected in expected_counts.items():
        if payload.get(key) != expected:
            raise ValueError(f"{path}: {key}={payload.get(key)!r}, expected {expected}")
    raw_input = payload.get("input_jsonl")
    if not isinstance(raw_input, str) or Path(raw_input).expanduser().resolve() != responses_path.resolve():
        raise ValueError(f"{path}: input_jsonl does not identify {responses_path}")
    if payload.get("input_jsonl_sha256") != sha256_file(responses_path):
        raise ValueError(f"{path}: input_jsonl SHA-256 does not match the registered responses")
    grader = payload.get("grader")
    expected_grader = Path(__file__).resolve().parents[1] / "scripts/val/eval/utils.py"
    if not isinstance(grader, Mapping):
        raise ValueError(f"{path}: missing grader provenance")
    raw_grader_path = grader.get("path")
    if (
        not isinstance(raw_grader_path, str)
        or Path(raw_grader_path).expanduser().resolve() != expected_grader.resolve()
        or grader.get("sha256") != sha256_file(expected_grader)
    ):
        raise ValueError(f"{path}: grader path/SHA-256 does not match the repository grader")
    correct = payload.get("num_correct")
    if not isinstance(correct, int) or isinstance(correct, bool) or not 0 <= correct <= expected_prompts * n:
        raise ValueError(f"{path}: invalid num_correct={correct!r}")
    key = f"avg@{n}"
    avg = payload.get(key)
    expected_avg = correct / (expected_prompts * n)
    if (
        not isinstance(avg, (int, float))
        or isinstance(avg, bool)
        or not math.isfinite(float(avg))
        or not math.isclose(float(avg), expected_avg, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ValueError(f"{path}: {key} is inconsistent with num_correct/num_responses")


def build_checkpoint_summary(
    aggregate: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "target_id": manifest.get("target_id"),
        "kind": "cell_checkpoint",
        "checkpoint_step": manifest.get("checkpoint_step"),
        "model": manifest.get("model"),
        "revision": manifest.get("revision"),
        "paper_comparable": manifest.get("paper_comparable") is True,
        "evaluation_plan_sha256": manifest.get("evaluation_plan", {}).get("sha256"),
        **dict(aggregate),
    }
    if manifest.get("evaluation_id") is not None:
        payload["evaluation_id"] = manifest.get("evaluation_id")
    if manifest.get("training", {}).get("protocol") == "scientific":
        payload["paper_evaluation_protocol"] = (
            manifest.get("paper_evaluation_protocol") is True
        )
        payload["comparability"] = dict(manifest.get("comparability", {}))
    return payload


def write_or_validate_summary(path: Path, payload: Mapping[str, Any]) -> None:
    expected = dict(payload)
    if path.is_file():
        observed = read_object(path, "checkpoint evaluation summary")
        comparable = dict(observed)
        comparable.pop("created_at", None)
        if comparable != expected:
            raise ValueError(f"existing checkpoint summary differs; refusing to overwrite: {path}")
        return
    write_once_json(
        path,
        {**expected, "created_at": utc_now()},
        "checkpoint evaluation summary",
    )


def build_commands(
    repo_root: Path,
    run_dir: Path,
    checkpoints: Sequence[tuple[int, Path]],
    tokenizer: str,
    tokenizer_revision: str | None,
    n: int,
    max_tokens: int,
    seed: int,
    limit: int | None,
    overwrite_incomplete: bool,
    *,
    evaluation_id: str = DEFAULT_EVALUATION_ID,
) -> list[tuple[str, list[str], Path | None]]:
    python = repo_root / ".venv-opd/bin/python"
    commands: list[tuple[str, list[str], Path | None]] = []
    for step, checkpoint in checkpoints:
        merged = run_dir / "merged" / f"global_step_{step}"
        if not ((merged / "config.json").is_file() and any(merged.glob("*.safetensors"))):
            if merged.exists():
                raise ValueError(f"incomplete merged model exists; inspect without overwriting: {merged}")
            commands.append(
                (
                    f"merge-step-{step}",
                    ["bash", str(repo_root / "reproduce_4b/merge_checkpoint.sh"), str(checkpoint), str(merged)],
                    None,
                )
            )
        eval_root = evaluation_root(run_dir, evaluation_id) / f"global_step_{step}"
        for benchmark in BENCHMARKS:
            benchmark_dir = eval_root / benchmark
            input_parquet = repo_root / "datasets/test_data" / benchmark / "test.parquet"
            output_jsonl = benchmark_dir / "responses.jsonl"
            metrics_json = benchmark_dir / "metrics.json"
            expected_prompts = parquet_rows(input_parquet, limit)
            sampling = sampling_contract(
                str(merged.resolve()), tokenizer, tokenizer_revision, n, max_tokens, seed
            )
            generation_complete = False
            if output_jsonl.is_file():
                try:
                    validate_generation(output_jsonl, expected_prompts, n, sampling)
                    generation_complete = True
                except (json.JSONDecodeError, OSError, ValueError):
                    if not overwrite_incomplete:
                        raise
            if not generation_complete:
                generation = [
                    str(python),
                    str(repo_root / "reproduce_4b/generate_eval.py"),
                    "--model",
                    str(merged),
                    "--tokenizer",
                    tokenizer,
                    "--input-parquet",
                    str(input_parquet),
                    "--output-jsonl",
                    str(output_jsonl),
                    "--cuda-visible-devices",
                    os.environ.get("CUDA_VISIBLE_DEVICES", "0,1"),
                    "--tensor-parallel-size",
                    "2",
                    "--n",
                    str(n),
                    "--temperature",
                    "0.7",
                    "--top-p",
                    "0.95",
                    "--max-tokens",
                    str(max_tokens),
                    "--max-model-len",
                    str(max_tokens + 1024),
                    "--thinking",
                    "off",
                    "--seed",
                    str(seed),
                ]
                if tokenizer_revision:
                    generation.extend(["--tokenizer-revision", tokenizer_revision])
                if limit:
                    generation.extend(["--limit", str(limit)])
                if output_jsonl.exists() and overwrite_incomplete:
                    generation.append("--overwrite")
                commands.append((f"generate-step-{step}-{benchmark}", generation, output_jsonl))
            metrics_complete = False
            if metrics_json.is_file() and generation_complete:
                try:
                    validate_metrics(metrics_json, output_jsonl, expected_prompts, n)
                    metrics_complete = True
                except (json.JSONDecodeError, OSError, ValueError):
                    if not overwrite_incomplete:
                        raise
            elif metrics_json.exists() and not overwrite_incomplete:
                raise ValueError(
                    f"metrics exist without a complete registered generation; refusing to overwrite: {metrics_json}"
                )
            if not metrics_complete:
                grading = [
                    str(python),
                    str(repo_root / "reproduce_4b/grade_eval.py"),
                    "--input-jsonl",
                    str(output_jsonl),
                    "--output-json",
                    str(metrics_json),
                    "--n",
                    str(n),
                    "--strict-n",
                ]
                if metrics_json.exists() and overwrite_incomplete:
                    grading.append("--overwrite")
                commands.append((f"grade-step-{step}-{benchmark}", grading, metrics_json))
    return commands


def plan_summary_paths(plan: Mapping[str, Any]) -> list[str]:
    targets = plan.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError("evaluation plan has no targets")
    summaries: list[str] = []
    for target in targets:
        if not isinstance(target, Mapping):
            raise ValueError("evaluation plan contains a malformed target")
        raw = target.get("summary")
        if not isinstance(raw, str) or not raw:
            raise ValueError("evaluation plan target lacks a summary path")
        summaries.append(str(Path(raw).expanduser().resolve()))
    if len(summaries) != len(set(summaries)):
        raise ValueError("evaluation plan repeats a summary path")
    return summaries


def load_or_initialize_status(
    status_path: Path,
    plan_path: Path,
    plan_sha256: str,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    plan_identity = {"path": str(plan_path.resolve()), "sha256": plan_sha256}
    if not status_path.is_file():
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "ablation_checkpoint_grid",
            "evaluation_plan": plan_identity,
            "state": "pending",
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "commands": [],
            "attempts": [],
            "summaries": [],
            "targets": {},
        }
    status = read_object(status_path, "evaluation status")
    if status.get("evaluation_plan") != plan_identity:
        raise ValueError("existing evaluation status is linked to a different evaluation plan")
    for key in ("commands", "attempts", "summaries"):
        if not isinstance(status.get(key), list):
            raise ValueError(f"evaluation status {key} must be a list")
    if not isinstance(status.get("targets"), Mapping):
        raise ValueError("evaluation status targets must be an object")
    allowed_summaries = set(plan_summary_paths(plan))
    observed_summaries = [str(Path(item).expanduser().resolve()) for item in status["summaries"]]
    if len(observed_summaries) != len(set(observed_summaries)):
        raise ValueError("evaluation status repeats a summary path")
    if not set(observed_summaries).issubset(allowed_summaries):
        raise ValueError("evaluation status contains a summary outside the registered grid")
    status["summaries"] = observed_summaries
    return status


def build_parser(repo_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run", "status"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--evaluation-id",
        default=DEFAULT_EVALUATION_ID,
        help="Immutable evaluation namespace (for example trend-n4 or exact-avg16).",
    )
    parser.add_argument("--checkpoint-step", type=positive_int, action="append", default=[])
    parser.add_argument("--n", type=positive_int, default=16)
    parser.add_argument("--max-tokens", type=positive_int, default=31744)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=positive_int)
    parser.add_argument("--overwrite-incomplete", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--acknowledge-full-eval", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = build_parser(repo_root)
    args = parser.parse_args(argv)
    try:
        run_dir = args.run_dir.expanduser().resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"run directory does not exist: {run_dir}")
        training = read_training_identity(run_dir)
        protocol = str(training["protocol"])
        immutable_grid = protocol in {"paper", "scientific"}
        eval_root = evaluation_root(run_dir, args.evaluation_id)
        plan_path = eval_root / "evaluation_plan.json"
        requested_steps = list(args.checkpoint_step)
        if immutable_grid and not requested_steps:
            if plan_path.is_file():
                registered = read_object(plan_path, "evaluation plan").get("checkpoint_steps")
                if (
                    not isinstance(registered, list)
                    or not registered
                    or any(not isinstance(step, int) or isinstance(step, bool) for step in registered)
                ):
                    raise ValueError("existing paper evaluation plan has an invalid checkpoint grid")
                requested_steps = list(registered)
            else:
                raise ValueError(
                    f"{protocol} evaluation requires an explicit --checkpoint-step grid before any evaluation"
                )
        checkpoints = discover_checkpoints(run_dir, requested_steps)
        tokenizer, revision = read_student_identity(run_dir)
        plan = build_evaluation_plan(
            repo_root,
            run_dir,
            checkpoints,
            training,
            tokenizer,
            revision,
            args.n,
            args.max_tokens,
            args.seed,
            args.limit,
            evaluation_id=args.evaluation_id,
        )
        if plan_path.is_file() and read_object(plan_path, "evaluation plan") != plan:
            raise ValueError(
                f"existing evaluation plan differs; checkpoint grid and sampling are write-once: {plan_path}"
            )
        commands = build_commands(
            repo_root,
            run_dir,
            checkpoints,
            tokenizer,
            revision,
            args.n,
            args.max_tokens,
            args.seed,
            args.limit,
            args.overwrite_incomplete,
            evaluation_id=args.evaluation_id,
        )
        if args.action in {"plan", "status"}:
            state = "unregistered"
            status_path = evaluation_status_path(run_dir, args.evaluation_id)
            if status_path.is_file():
                state = str(read_object(status_path, "evaluation status").get("state", "unknown"))
            print(
                f"checkpoints={[step for step, _ in checkpoints]} "
                f"plan={'registered' if plan_path.is_file() else 'unregistered'} "
                f"state={state} pending_commands={len(commands)}"
            )
            for label, command, _ in commands:
                print(label + ": " + " ".join(map(str, command)))
            return 0
        if not args.yes:
            raise ValueError("action=run requires --yes")
        if args.limit is None and not args.acknowledge_full_eval:
            raise ValueError("full avg@N evaluation requires --acknowledge-full-eval")
        if immutable_grid and args.overwrite_incomplete:
            raise ValueError(
                f"{protocol} evaluation artifacts are immutable; --overwrite-incomplete is forbidden"
            )

        if immutable_grid and not plan_path.exists():
            for step, _ in checkpoints:
                target_root = eval_root / f"global_step_{step}"
                preexisting = [
                    path
                    for path in (
                        target_root / "target_manifest.json",
                        target_root / "summary.json",
                        *(target_root / benchmark / "responses.jsonl" for benchmark in BENCHMARKS),
                        *(target_root / benchmark / "metrics.json" for benchmark in BENCHMARKS),
                    )
                    if path.exists()
                ]
                if preexisting:
                    raise ValueError(
                        f"cannot register a {protocol} checkpoint grid after evaluation artifacts exist: "
                        + ", ".join(map(str, preexisting))
                    )

        lock = acquire_gpu_lock(os.environ.get("CUDA_VISIBLE_DEVICES", "0,1"))
        status_path = evaluation_status_path(run_dir, args.evaluation_id)
        status: dict[str, Any] | None = None
        attempt_record: dict[str, Any] | None = None
        try:
            write_once_json(plan_path, plan, "evaluation plan")
            plan_sha256 = sha256_file(plan_path)
            manifests: dict[int, dict[str, Any]] = {}
            for step, checkpoint in checkpoints:
                manifest = build_target_manifest(
                    run_dir,
                    step,
                    checkpoint,
                    training,
                    plan_path,
                    plan_sha256,
                    plan,
                    tokenizer,
                    revision,
                    evaluation_id=args.evaluation_id,
                )
                manifest_path = eval_root / f"global_step_{step}" / "target_manifest.json"
                write_once_json(manifest_path, manifest, "checkpoint target manifest")
                manifests[step] = manifest

            status = load_or_initialize_status(status_path, plan_path, plan_sha256, plan)
            attempt_record = {
                "attempt": len(status["attempts"]) + 1,
                "state": "running",
                "started_at": utc_now(),
                "commands": [],
            }
            status["attempts"].append(attempt_record)
            status.update({"state": "running", "updated_at": utc_now()})
            atomic_json(status_path, status)

            environment = os.environ.copy()
            environment["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
            for label, command, output in commands:
                if output is not None:
                    output.parent.mkdir(parents=True, exist_ok=True)
                result = subprocess.run(command, cwd=repo_root, env=environment, check=False)
                command_result = {"label": label, "exit_code": result.returncode}
                status["commands"].append(command_result)
                attempt_record["commands"].append(command_result)
                status["updated_at"] = utc_now()
                atomic_json(status_path, status)
                if result.returncode != 0:
                    raise RuntimeError(f"evaluation command failed ({label}) with exit {result.returncode}")

            summaries: list[str] = []
            targets: dict[str, Any] = {}
            for step, _ in checkpoints:
                target_root = eval_root / f"global_step_{step}"
                metrics_paths = [
                    target_root / benchmark / "metrics.json"
                    for benchmark in BENCHMARKS
                ]
                for benchmark, metrics_path in zip(BENCHMARKS, metrics_paths):
                    responses_path = target_root / benchmark / "responses.jsonl"
                    validate_generation(
                        responses_path,
                        parquet_rows(
                            repo_root / "datasets/test_data" / benchmark / "test.parquet",
                            args.limit,
                        ),
                        args.n,
                        manifests[step]["sampling"],
                    )
                    validate_metrics(
                        metrics_path,
                        responses_path,
                        parquet_rows(
                            repo_root / "datasets/test_data" / benchmark / "test.parquet",
                            args.limit,
                        ),
                        args.n,
                    )
                aggregate = aggregate_metrics(metrics_paths, args.n)
                summary = build_checkpoint_summary(aggregate, manifests[step])
                summary_path = target_root / "summary.json"
                write_or_validate_summary(summary_path, summary)
                resolved_summary = str(summary_path.resolve())
                summaries.append(resolved_summary)
                targets[str(manifests[step]["target_id"])] = {
                    "state": "completed",
                    "checkpoint_step": step,
                    "summary": resolved_summary,
                }
            if summaries != plan_summary_paths(plan):
                raise RuntimeError("completed summaries do not exactly cover the registered checkpoint grid")
            ended_at = utc_now()
            attempt_record.update({"state": "completed", "ended_at": ended_at})
            status.update(
                {
                    "state": "completed",
                    "updated_at": ended_at,
                    "ended_at": ended_at,
                    "summaries": summaries,
                    "targets": targets,
                }
            )
            atomic_json(status_path, status)
        except Exception:
            if status is not None:
                ended_at = utc_now()
                if attempt_record is not None:
                    attempt_record.update({"state": "failed", "ended_at": ended_at})
                status.update({"state": "failed", "updated_at": ended_at, "ended_at": ended_at})
                atomic_json(status_path, status)
            raise
        finally:
            lock.close()
        return 0
    except (FileNotFoundError, json.JSONDecodeError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
