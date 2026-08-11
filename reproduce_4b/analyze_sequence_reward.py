#!/usr/bin/env python3
"""Reproduce Figure 14 sequence-mean reward analysis from fixed rollouts.

The analysis path is deliberately independent from model inference.  It accepts
one immutable rollout batch and two token-logprob files, validates that all
three files describe exactly the same actions, and computes

    mean_t(log p_teacher(y_t | x, y_<t) - log p_student(y_t | x, y_<t)).

The optional ``score`` command creates a token-logprob file with Transformers.
It is kept separate so that validation and aggregation remain CPU-only and can
be unit-tested without importing a model runtime.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from probe_runtime import (
    atomic_json as runtime_atomic_json,
    canonical_sha256,
    completed_ordinals,
    generate_rollout_chunks_vllm,
    load_dapo_sample,
    merge_chunks,
    write_chunk,
)


SCHEMA_VERSION = 1
FIG14_PROMPT_COUNT = 1_070
FIG14_ROLLOUTS_PER_PROMPT = 4
FIG14_TOTAL_ROLLOUTS = FIG14_PROMPT_COUNT * FIG14_ROLLOUTS_PER_PROMPT
FIG14_SEED = 42
FIG14_MAX_TOKENS = 7_168
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_CONFIDENCE = 0.95
REQUIRED_SAMPLING_KEYS = (
    "model",
    "revision",
    "seed",
    "n",
    "temperature",
    "top_p",
    "max_tokens",
    "thinking",
)


RolloutKey = tuple[str, int]


@dataclass(frozen=True)
class ValidatedRollout:
    batch_id: str
    example_id: str
    rollout_id: int
    prompt_token_ids: tuple[int, ...]
    response_token_ids: tuple[int, ...]
    correct: bool
    sampling: Mapping[str, Any]
    fingerprint: str
    answer: str | None = None
    response_text: str | None = None

    @property
    def key(self) -> RolloutKey:
        return self.example_id, self.rollout_id


@dataclass(frozen=True)
class ValidatedLogprobs:
    batch_id: str
    example_id: str
    rollout_id: int
    role: str
    model: str
    revision: str
    prompt_token_ids: tuple[int, ...]
    response_token_ids: tuple[int, ...]
    token_logprobs: tuple[float, ...]
    rollout_fingerprint: str
    scoring: Mapping[str, Any]

    @property
    def key(self) -> RolloutKey:
        return self.example_id, self.rollout_id


@dataclass(frozen=True)
class SequenceRewardRow:
    batch_id: str
    example_id: str
    rollout_id: int
    correct: bool
    response_length: int
    student_mean_logprob: float
    teacher_mean_logprob: float
    sequence_mean_reward: float


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite")
    return parsed


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _token_ids(value: Any, label: str, *, allow_empty: bool) -> tuple[int, ...]:
    if not isinstance(value, list) or (not allow_empty and not value):
        suffix = "a list" if allow_empty else "a non-empty list"
        raise ValueError(f"{label} must be {suffix} of token IDs")
    output: list[int] = []
    for position, token_id in enumerate(value):
        output.append(_nonnegative_int(token_id, f"{label}[{position}]"))
    return tuple(output)


def validate_sampling_metadata(sampling: Any, expected: Mapping[str, Any], label: str) -> None:
    """Require complete, batch-invariant sampling provenance.

    Exact equality is intentional: silently accepting additional or missing
    sampling fields can mix trajectories produced by distinct protocols.
    """

    if not isinstance(sampling, Mapping):
        raise ValueError(f"{label} must be an object")
    if not isinstance(expected, Mapping) or not expected:
        raise ValueError("expected sampling metadata must be a non-empty object")
    missing = [key for key in REQUIRED_SAMPLING_KEYS if key not in expected]
    if missing:
        raise ValueError(f"expected sampling metadata lacks required keys: {missing}")
    if _canonical_json(sampling) != _canonical_json(expected):
        raise ValueError(f"{label} does not exactly match expected sampling metadata")
    for key in ("model", "revision", "thinking"):
        _nonempty_string(sampling[key], f"{label}.{key}")
    _nonnegative_int(sampling["seed"], f"{label}.seed")
    if _nonnegative_int(sampling["n"], f"{label}.n") == 0:
        raise ValueError(f"{label}.n must be positive")
    if _nonnegative_int(sampling["max_tokens"], f"{label}.max_tokens") == 0:
        raise ValueError(f"{label}.max_tokens must be positive")
    temperature = _finite_number(sampling["temperature"], f"{label}.temperature")
    if temperature < 0.0:
        raise ValueError(f"{label}.temperature must be non-negative")
    top_p = _finite_number(sampling["top_p"], f"{label}.top_p")
    if not 0.0 < top_p <= 1.0:
        raise ValueError(f"{label}.top_p must lie in (0, 1]")


def rollout_fingerprint(record: Mapping[str, Any]) -> str:
    """Hash every field that can change the scored sequence or its label."""

    core = {
        "schema_version": record.get("schema_version"),
        "batch_id": record.get("batch_id"),
        "example_id": record.get("example_id"),
        "rollout_id": record.get("rollout_id"),
        "prompt_token_ids": record.get("prompt_token_ids"),
        "response_token_ids": record.get("response_token_ids"),
        "correct": record.get("correct"),
        "sampling": record.get("sampling"),
    }
    return hashlib.sha256(_canonical_json(core).encode("utf-8")).hexdigest()


def validate_rollouts(
    records: Iterable[Mapping[str, Any]],
    expected_sampling: Mapping[str, Any],
) -> dict[RolloutKey, ValidatedRollout]:
    """Validate one fixed, already-generated and already-graded rollout batch."""

    output: dict[RolloutKey, ValidatedRollout] = {}
    batch_ids: set[str] = set()
    for index, record in enumerate(records, start=1):
        label = f"rollout record {index}"
        if not isinstance(record, Mapping):
            raise ValueError(f"{label} must be an object")
        if record.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"{label} has unsupported schema_version")
        batch_id = _nonempty_string(record.get("batch_id"), f"{label}.batch_id")
        example_id = _nonempty_string(record.get("example_id"), f"{label}.example_id")
        rollout_id = _nonnegative_int(record.get("rollout_id"), f"{label}.rollout_id")
        prompt_ids = _token_ids(record.get("prompt_token_ids"), f"{label}.prompt_token_ids", allow_empty=False)
        response_ids = _token_ids(
            record.get("response_token_ids"), f"{label}.response_token_ids", allow_empty=False
        )
        correct = record.get("correct")
        if not isinstance(correct, bool):
            raise ValueError(f"{label}.correct must be boolean")
        validate_sampling_metadata(record.get("sampling"), expected_sampling, f"{label}.sampling")
        key = (example_id, rollout_id)
        if key in output:
            raise ValueError(f"duplicate rollout ID pair {key!r}")
        batch_ids.add(batch_id)
        output[key] = ValidatedRollout(
            batch_id=batch_id,
            example_id=example_id,
            rollout_id=rollout_id,
            prompt_token_ids=prompt_ids,
            response_token_ids=response_ids,
            correct=correct,
            sampling=dict(record["sampling"]),
            fingerprint=rollout_fingerprint(record),
            answer=str(record["answer"]) if record.get("answer") is not None else None,
            response_text=str(record["response"]) if record.get("response") is not None else None,
        )
    if not output:
        raise ValueError("rollout batch is empty")
    if len(batch_ids) != 1:
        raise ValueError(f"rollouts mix batch IDs: {sorted(batch_ids)}")
    expected_n = int(expected_sampling["n"])
    per_prompt: dict[str, set[int]] = {}
    for example_id, rollout_id in output:
        per_prompt.setdefault(example_id, set()).add(rollout_id)
    malformed = [example_id for example_id, ids in per_prompt.items() if len(ids) != expected_n]
    if malformed:
        raise ValueError(
            f"{len(malformed)} prompts do not have exactly sampling.n={expected_n} unique rollouts"
        )
    return output


def fig14_sampling_contract(model: str, revision: str) -> dict[str, Any]:
    """Return the registered local adaptation for Figure 14 rollout creation."""

    return {
        "model": _nonempty_string(model, "student model"),
        "revision": _nonempty_string(revision, "student revision"),
        "seed": FIG14_SEED,
        "n": FIG14_ROLLOUTS_PER_PROMPT,
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": FIG14_MAX_TOKENS,
        "thinking": "auto",
    }


def validate_fig14_rollout_batch(
    records: Iterable[Mapping[str, Any]], expected_sampling: Mapping[str, Any]
) -> dict[RolloutKey, ValidatedRollout]:
    """Require the fixed 1,070-prompt x four-action Figure 14 batch."""

    expected = fig14_sampling_contract(
        str(expected_sampling.get("model", "")), str(expected_sampling.get("revision", ""))
    )
    validate_sampling_metadata(expected_sampling, expected, "Figure 14 sampling")
    validated = validate_rollouts(records, expected_sampling)
    prompt_ids = {rollout.example_id for rollout in validated.values()}
    if len(prompt_ids) != FIG14_PROMPT_COUNT:
        raise ValueError(
            f"Figure 14 requires exactly {FIG14_PROMPT_COUNT} prompts, found {len(prompt_ids)}"
        )
    if len(validated) != FIG14_TOTAL_ROLLOUTS:
        raise ValueError(
            f"Figure 14 requires exactly {FIG14_TOTAL_ROLLOUTS} actions, found {len(validated)}"
        )
    return validated


def action_batch_fingerprint(rollouts: Iterable[ValidatedRollout]) -> str:
    """Hash the immutable action batch shared by student and both teachers."""

    rows = sorted(rollouts, key=lambda item: item.key)
    if not rows:
        raise ValueError("cannot fingerprint an empty action batch")
    return canonical_sha256(
        [
            {
                "batch_id": row.batch_id,
                "example_id": row.example_id,
                "rollout_id": row.rollout_id,
                "prompt_token_ids": list(row.prompt_token_ids),
                "response_token_ids": list(row.response_token_ids),
                "correct": row.correct,
                "rollout_fingerprint": row.fingerprint,
            }
            for row in rows
        ]
    )


def validate_logprob_records(
    records: Iterable[Mapping[str, Any]],
    *,
    expected_role: str,
    expected_model: str,
    expected_revision: str,
) -> dict[RolloutKey, ValidatedLogprobs]:
    """Validate one model's per-action log probabilities."""

    if expected_role not in {"student", "teacher"}:
        raise ValueError("expected_role must be student or teacher")
    _nonempty_string(expected_model, "expected_model")
    _nonempty_string(expected_revision, "expected_revision")
    output: dict[RolloutKey, ValidatedLogprobs] = {}
    scoring_values: set[str] = set()
    for index, record in enumerate(records, start=1):
        label = f"{expected_role} logprob record {index}"
        if not isinstance(record, Mapping):
            raise ValueError(f"{label} must be an object")
        if record.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"{label} has unsupported schema_version")
        batch_id = _nonempty_string(record.get("batch_id"), f"{label}.batch_id")
        example_id = _nonempty_string(record.get("example_id"), f"{label}.example_id")
        rollout_id = _nonnegative_int(record.get("rollout_id"), f"{label}.rollout_id")
        role = _nonempty_string(record.get("role"), f"{label}.role")
        model = _nonempty_string(record.get("model"), f"{label}.model")
        revision = _nonempty_string(record.get("revision"), f"{label}.revision")
        if role != expected_role:
            raise ValueError(f"{label}.role={role!r}, expected {expected_role!r}")
        if model != expected_model or revision != expected_revision:
            raise ValueError(f"{label} has an unexpected model identity")
        prompt_ids = _token_ids(record.get("prompt_token_ids"), f"{label}.prompt_token_ids", allow_empty=False)
        response_ids = _token_ids(
            record.get("response_token_ids"), f"{label}.response_token_ids", allow_empty=False
        )
        raw_logprobs = record.get("token_logprobs")
        if not isinstance(raw_logprobs, list):
            raise ValueError(f"{label}.token_logprobs must be a list")
        logprobs = tuple(
            _finite_number(value, f"{label}.token_logprobs[{position}]")
            for position, value in enumerate(raw_logprobs)
        )
        if len(logprobs) != len(response_ids):
            raise ValueError(
                f"{label} logprob/token length mismatch: {len(logprobs)} != {len(response_ids)}"
            )
        if any(value > 1e-5 for value in logprobs):
            raise ValueError(f"{label} contains a positive log probability")
        fingerprint = _nonempty_string(
            record.get("rollout_fingerprint"), f"{label}.rollout_fingerprint"
        )
        scoring = record.get("scoring")
        if not isinstance(scoring, Mapping) or scoring.get("method") != "causal-next-token":
            raise ValueError(f"{label}.scoring must declare method=causal-next-token")
        scoring_values.add(_canonical_json(scoring))
        key = (example_id, rollout_id)
        if key in output:
            raise ValueError(f"duplicate {expected_role} logprob ID pair {key!r}")
        output[key] = ValidatedLogprobs(
            batch_id=batch_id,
            example_id=example_id,
            rollout_id=rollout_id,
            role=role,
            model=model,
            revision=revision,
            prompt_token_ids=prompt_ids,
            response_token_ids=response_ids,
            token_logprobs=logprobs,
            rollout_fingerprint=fingerprint,
            scoring=dict(scoring),
        )
    if not output:
        raise ValueError(f"{expected_role} logprob batch is empty")
    if len(scoring_values) != 1:
        raise ValueError(f"{expected_role} logprobs mix scoring protocols")
    return output


def _assert_same_keys(reference: set[RolloutKey], actual: set[RolloutKey], label: str) -> None:
    if reference == actual:
        return
    missing = sorted(reference - actual)[:5]
    extra = sorted(actual - reference)[:5]
    raise ValueError(f"{label} is not the fixed same rollout batch; missing={missing}, extra={extra}")


def compute_sequence_rewards(
    rollouts: Iterable[Mapping[str, Any]],
    student_records: Iterable[Mapping[str, Any]],
    teacher_records: Iterable[Mapping[str, Any]],
    *,
    expected_sampling: Mapping[str, Any],
    student_model: str,
    student_revision: str,
    teacher_model: str,
    teacher_revision: str,
) -> list[SequenceRewardRow]:
    """Validate and compute Figure 14 sequence rewards as a pure function."""

    rollout_map = validate_rollouts(rollouts, expected_sampling)
    student_map = validate_logprob_records(
        student_records,
        expected_role="student",
        expected_model=student_model,
        expected_revision=student_revision,
    )
    teacher_map = validate_logprob_records(
        teacher_records,
        expected_role="teacher",
        expected_model=teacher_model,
        expected_revision=teacher_revision,
    )
    keys = set(rollout_map)
    _assert_same_keys(keys, set(student_map), "student logprobs")
    _assert_same_keys(keys, set(teacher_map), "teacher logprobs")
    rows: list[SequenceRewardRow] = []
    for key in sorted(keys):
        rollout = rollout_map[key]
        student = student_map[key]
        teacher = teacher_map[key]
        for role, scored in (("student", student), ("teacher", teacher)):
            if scored.batch_id != rollout.batch_id:
                raise ValueError(f"{role} batch ID mismatch for {key!r}")
            if scored.rollout_fingerprint != rollout.fingerprint:
                raise ValueError(f"{role} rollout fingerprint mismatch for {key!r}")
            if scored.prompt_token_ids != rollout.prompt_token_ids:
                raise ValueError(f"{role} prompt token IDs mismatch for {key!r}")
            if scored.response_token_ids != rollout.response_token_ids:
                raise ValueError(f"{role} response token IDs mismatch for {key!r}")
        student_mean = sum(student.token_logprobs) / len(student.token_logprobs)
        teacher_mean = sum(teacher.token_logprobs) / len(teacher.token_logprobs)
        token_rewards = [
            teacher_value - student_value
            for student_value, teacher_value in zip(
                student.token_logprobs, teacher.token_logprobs, strict=True
            )
        ]
        mean_reward = sum(token_rewards) / len(token_rewards)
        rows.append(
            SequenceRewardRow(
                batch_id=rollout.batch_id,
                example_id=rollout.example_id,
                rollout_id=rollout.rollout_id,
                correct=rollout.correct,
                response_length=len(rollout.response_token_ids),
                student_mean_logprob=student_mean,
                teacher_mean_logprob=teacher_mean,
                sequence_mean_reward=mean_reward,
            )
        )
    return rows


def tie_safe_auroc(labels: Sequence[bool], scores: Sequence[float]) -> float:
    """Compute AUROC with half credit for score ties (Mann-Whitney form)."""

    if len(labels) != len(scores) or not labels:
        raise ValueError("labels and scores must have the same non-zero length")
    pairs: list[tuple[float, bool]] = []
    for index, (label, score) in enumerate(zip(labels, scores, strict=True)):
        if not isinstance(label, bool):
            raise ValueError(f"labels[{index}] must be boolean")
        pairs.append((_finite_number(score, f"scores[{index}]"), label))
    positives = sum(label for _, label in pairs)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("AUROC requires at least one correct and one incorrect rollout")
    pairs.sort(key=lambda item: item[0])
    favorable = 0.0
    negatives_before = 0
    cursor = 0
    while cursor < len(pairs):
        end = cursor + 1
        while end < len(pairs) and pairs[end][0] == pairs[cursor][0]:
            end += 1
        group = pairs[cursor:end]
        group_positives = sum(label for _, label in group)
        group_negatives = len(group) - group_positives
        favorable += group_positives * (negatives_before + 0.5 * group_negatives)
        negatives_before += group_negatives
        cursor = end
    return favorable / (positives * negatives)


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def clustered_auroc_bootstrap(
    rows: Sequence[SequenceRewardRow],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = FIG14_SEED,
    confidence: float = BOOTSTRAP_CONFIDENCE,
) -> dict[str, Any]:
    """Prompt-cluster bootstrap for the fixed four-rollout action batch."""

    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap confidence must lie in (0, 1)")
    by_prompt: dict[str, list[SequenceRewardRow]] = {}
    for row in rows:
        by_prompt.setdefault(row.example_id, []).append(row)
    prompts = sorted(by_prompt)
    if not prompts:
        raise ValueError("cannot bootstrap an empty reward batch")
    generator = random.Random(seed)
    values: list[float] = []
    # Degenerate resamples are possible when correctness is highly imbalanced.
    # They are rejected, and the attempt cap keeps this path deterministic.
    max_attempts = max(replicates * 20, replicates + 100)
    attempts = 0
    while len(values) < replicates and attempts < max_attempts:
        attempts += 1
        sampled = [prompts[generator.randrange(len(prompts))] for _ in prompts]
        labels: list[bool] = []
        scores: list[float] = []
        for prompt in sampled:
            labels.extend(row.correct for row in by_prompt[prompt])
            scores.extend(row.sequence_mean_reward for row in by_prompt[prompt])
        if all(labels) or not any(labels):
            continue
        values.append(tie_safe_auroc(labels, scores))
    if len(values) != replicates:
        raise ValueError(
            f"only {len(values)}/{replicates} non-degenerate bootstrap replicates were available"
        )
    alpha = (1.0 - confidence) / 2.0
    return {
        "unit": "prompt (four rollout cluster)",
        "seed": seed,
        "replicates": replicates,
        "confidence": confidence,
        "lower": _percentile(values, alpha),
        "upper": _percentile(values, 1.0 - alpha),
    }


def distribution_summary(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty distribution")
    checked = [_finite_number(value, "distribution value") for value in values]
    mean = sum(checked) / len(checked)
    variance = sum((value - mean) ** 2 for value in checked) / len(checked)
    return {
        "n": len(checked),
        "mean": mean,
        "std_population": math.sqrt(variance),
        "min": min(checked),
        "p25": _percentile(checked, 0.25),
        "median": _percentile(checked, 0.5),
        "p75": _percentile(checked, 0.75),
        "max": max(checked),
    }


def summarize_sequence_rewards(
    rows: Sequence[SequenceRewardRow],
    *,
    bootstrap_replicates: int | None = None,
    bootstrap_seed: int = FIG14_SEED,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("no sequence reward rows")
    correct = [row.sequence_mean_reward for row in rows if row.correct]
    incorrect = [row.sequence_mean_reward for row in rows if not row.correct]
    scores = [row.sequence_mean_reward for row in rows]
    labels = [row.correct for row in rows]
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "definition": "mean_t(log_p_teacher(y_t|x,y_<t)-log_p_student(y_t|x,y_<t))",
        "num_rollouts": len(rows),
        "num_prompts": len({row.example_id for row in rows}),
        "batch_id": rows[0].batch_id,
        "correct": distribution_summary(correct),
        "incorrect": distribution_summary(incorrect),
        "all": distribution_summary(scores),
        "auroc_tie_policy": "half-credit (0.5) for equal correct/incorrect scores",
        "auroc": tie_safe_auroc(labels, scores),
    }
    if bootstrap_replicates is not None:
        summary["auroc_bootstrap_ci"] = clustered_auroc_bootstrap(
            rows, replicates=bootstrap_replicates, seed=bootstrap_seed
        )
    return summary


def combine_teacher_summaries(
    summaries: Mapping[str, Mapping[str, Any]],
    *,
    action_batch_sha256: str,
) -> dict[str, Any]:
    """Create the single fail-closed Figure 14 artifact consumed by rendering."""

    if len(summaries) != 2:
        raise ValueError("Figure 14 requires exactly two teacher summaries")
    checked: dict[str, Mapping[str, Any]] = {}
    batch_ids: set[str] = set()
    for teacher, summary in summaries.items():
        _nonempty_string(teacher, "teacher label")
        if summary.get("action_batch_sha256") != action_batch_sha256:
            raise ValueError(f"{teacher} summary does not use the shared action fingerprint")
        if summary.get("num_rollouts") != FIG14_TOTAL_ROLLOUTS:
            raise ValueError(f"{teacher} summary does not contain {FIG14_TOTAL_ROLLOUTS} actions")
        if summary.get("num_prompts") != FIG14_PROMPT_COUNT:
            raise ValueError(f"{teacher} summary does not contain {FIG14_PROMPT_COUNT} prompts")
        auroc = _finite_number(summary.get("auroc"), f"{teacher}.auroc")
        ci = summary.get("auroc_bootstrap_ci")
        if not isinstance(ci, Mapping):
            raise ValueError(f"{teacher} summary lacks AUROC bootstrap CI")
        lower = _finite_number(ci.get("lower"), f"{teacher}.auroc CI lower")
        upper = _finite_number(ci.get("upper"), f"{teacher}.auroc CI upper")
        if not 0.0 <= lower <= auroc <= upper <= 1.0:
            raise ValueError(f"{teacher} has an invalid AUROC confidence interval")
        batch_ids.add(_nonempty_string(summary.get("batch_id"), f"{teacher}.batch_id"))
        checked[teacher] = summary
    if len(batch_ids) != 1:
        raise ValueError("teacher summaries use different fixed rollout batch IDs")
    aurocs = [float(summary["auroc"]) for summary in checked.values()]
    lower_bounds = [float(summary["auroc_bootstrap_ci"]["lower"]) for summary in checked.values()]
    upper_bounds = [float(summary["auroc_bootstrap_ci"]["upper"]) for summary in checked.values()]
    if all(value > 0.5 for value in lower_bounds) and abs(aurocs[0] - aurocs[1]) <= 0.05:
        conclusion = "replicated"
        reason = "both AUROC lower bounds exceed 0.5 and point estimates differ by at most 0.05"
    elif all(value < 0.5 for value in upper_bounds):
        conclusion = "not_replicated_at_seed_42"
        reason = "both AUROC upper bounds are below 0.5"
    else:
        conclusion = "inconclusive"
        reason = "the preregistered dual-teacher AUROC rule is not decisively satisfied"
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": "fig14-dual-teacher-sequence-reward",
        "protocol": "scientific",
        "paper_reference": "Figure 14",
        "batch_id": next(iter(batch_ids)),
        "action_batch_sha256": action_batch_sha256,
        "num_prompts": FIG14_PROMPT_COUNT,
        "num_rollouts": FIG14_TOTAL_ROLLOUTS,
        "teachers": dict(checked),
        "conclusion_state": conclusion,
        "conclusion_reason": reason,
        "paper_reported_auroc": {"JustRL-1.5B": 0.7333, "R1-Distill-7B": 0.7511},
        "comparability": {
            "training": "not_applicable_fixed_rollout_probe",
            "evaluation": "local_explicit_adaptation",
            "provenance": "paper_batch_sampling_undisclosed",
        },
        "paper_comparable": False,
    }


def read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(payload, Mapping):
                raise ValueError(f"{path}:{line_number}: JSONL row must be an object")
            records.append(payload)
    return records


def read_json_object(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain one JSON object")
    return payload


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _write_plot(path: Path, rows: Sequence[SequenceRewardRow]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for PNG output") from exc
    correct = [row.sequence_mean_reward for row in rows if row.correct]
    incorrect = [row.sequence_mean_reward for row in rows if not row.correct]
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    bins = max(5, min(50, int(math.sqrt(len(rows))) + 1))
    axis.hist(
        [correct, incorrect],
        bins=bins,
        alpha=0.62,
        label=[f"Correct (N={len(correct)})", f"Incorrect (N={len(incorrect)})"],
    )
    axis.set_xlabel("Sequence mean reward")
    axis.set_ylabel("Count")
    axis.set_title("Fixed-rollout sequence reward distributions")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_analysis_outputs(
    output_dir: Path,
    rows: Sequence[SequenceRewardRow],
    summary: Mapping[str, Any],
    *,
    overwrite: bool,
    artifact_stem: str = "sequence_reward",
) -> dict[str, Path]:
    if not artifact_stem or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in artifact_stem):
        raise ValueError("artifact_stem must contain only letters, digits, '_' or '-'")
    if artifact_stem == "sequence_reward":
        paths = {
            "json": output_dir / "sequence_reward_summary.json",
            "csv": output_dir / "sequence_rewards.csv",
            "png": output_dir / "sequence_reward_distributions.png",
        }
    else:
        paths = {
            "json": output_dir / f"{artifact_stem}_summary.json",
            "csv": output_dir / f"{artifact_stem}_rows.csv",
            "png": output_dir / f"{artifact_stem}_distributions.png",
        }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing outputs: {existing}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_text(paths["json"], json.dumps(summary, indent=2, sort_keys=True) + "\n")
    fields = list(asdict(rows[0]))
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", prefix=".sequence_rewards.", dir=output_dir, delete=False
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, paths["csv"])
    _write_plot(paths["png"], rows)
    return paths


def score_action_with_kv_chunks(
    model: Any,
    torch_module: Any,
    rollout: ValidatedRollout,
    *,
    device: str,
    chunk_tokens: int,
) -> list[float]:
    """Score one exact action; exposed so chunk equivalence is CPU-testable."""

    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    full_ids = rollout.prompt_token_ids + rollout.response_token_ids
    if len(rollout.prompt_token_ids) < 1 or len(full_ids) < 2:
        raise ValueError(f"rollout {rollout.key!r} is too short for causal scoring")
    context_ids = full_ids[:-1]
    target_ids = full_ids[1:]
    first_response_target = len(rollout.prompt_token_ids) - 1
    past_key_values = None
    collected: list[float] = []
    for start in range(0, len(context_ids), chunk_tokens):
        end = min(start + chunk_tokens, len(context_ids))
        input_ids = torch_module.tensor(
            [context_ids[start:end]], dtype=torch_module.long, device=device
        )
        outputs = model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        target_start = max(start, first_response_target)
        if target_start >= end:
            continue
        local_start = target_start - start
        logits = outputs.logits[0, local_start : end - start].float()
        targets = torch_module.tensor(
            target_ids[target_start:end], dtype=torch_module.long, device=device
        )
        values = logits.log_softmax(dim=-1).gather(-1, targets[:, None]).squeeze(-1)
        collected.extend(float(value) for value in values.cpu().tolist())
    if len(collected) != len(rollout.response_token_ids):
        raise RuntimeError(
            "internal next-token alignment failure: "
            f"{len(collected)} != {len(rollout.response_token_ids)}"
        )
    return collected


def score_rollouts_transformers(
    rollouts: Sequence[ValidatedRollout],
    *,
    role: str,
    model_source: str,
    revision: str,
    device: str,
    dtype: str,
    chunk_tokens: int = 256,
    record_callback: Callable[[int, Mapping[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Score fixed action IDs under one causal LM with a growing KV cache.

    Only ``chunk_tokens`` logits are materialized at a time.  The KV cache is
    retained because every response probability must condition on the exact
    prompt and action prefix; truncating that cache would change Figure 14's
    reward definition.
    """

    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError("score requires torch and transformers") from exc
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype not in dtype_map:
        raise ValueError(f"unsupported dtype: {dtype}")
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    kwargs: dict[str, Any] = {"torch_dtype": dtype_map[dtype]}
    if revision != "local":
        kwargs["revision"] = revision
    model = AutoModelForCausalLM.from_pretrained(model_source, **kwargs).to(device)
    model.eval()
    scoring = {
        "method": "causal-next-token",
        "runtime": "transformers-kv-chunked",
        "dtype": dtype,
        "device": device,
        "chunk_tokens": chunk_tokens,
        "conditioning": "full prompt and exact action prefix",
    }
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for rollout in rollouts:
            collected = score_action_with_kv_chunks(
                model,
                torch,
                rollout,
                device=device,
                chunk_tokens=chunk_tokens,
            )
            record = {
                    "schema_version": SCHEMA_VERSION,
                    "batch_id": rollout.batch_id,
                    "example_id": rollout.example_id,
                    "rollout_id": rollout.rollout_id,
                    "role": role,
                    "model": model_source,
                    "revision": revision,
                    "prompt_token_ids": list(rollout.prompt_token_ids),
                    "response_token_ids": list(rollout.response_token_ids),
                    "token_logprobs": collected,
                    "rollout_fingerprint": rollout.fingerprint,
                    "scoring": scoring,
                }
            records.append(record)
            if record_callback is not None:
                record_callback(len(records) - 1, record)
    return records


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def prepare_fig14_generation(
    *,
    input_parquet: Path,
    output_root: Path,
    student_model: str,
    student_revision: str,
    shard_index: int,
    num_shards: int,
    batch_size: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    dtype: str,
    grader_utils: Path | None,
    overwrite_merge: bool,
) -> dict[str, Any]:
    """Generate one resumable shard of the registered 1,070 x 4 batch."""

    samples, selection = load_dapo_sample(input_parquet, FIG14_PROMPT_COUNT, FIG14_SEED)
    sampling = fig14_sampling_contract(student_model, student_revision)
    contract = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": "fig14-fixed-rollout-generation",
        "paper_reference": "Figure 14",
        "selection": selection,
        "sampling": sampling,
    }
    contract_sha = canonical_sha256(contract)
    contract_path = output_root / "generation_contract.json"
    if contract_path.is_file():
        existing = read_json_object(contract_path)
        if _canonical_json(existing) != _canonical_json(contract):
            raise ValueError("Figure 14 generation contract changed under an existing output root")
    else:
        runtime_atomic_json(contract_path, contract)
    batch_id = f"fig14-seed42-{contract_sha[:16]}"
    progress = generate_rollout_chunks_vllm(
        samples,
        output_root=output_root,
        chunk_kind="fig14-rollouts",
        contract_sha256=contract_sha,
        batch_id=batch_id,
        sampling=sampling,
        shard_index=shard_index,
        num_shards=num_shards,
        batch_size=batch_size,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=dtype,
        grader_utils=grader_utils,
    )
    completed = completed_ordinals(
        output_root,
        "fig14-rollouts",
        range(FIG14_PROMPT_COUNT),
        contract_sha256=contract_sha,
        expected_records=FIG14_ROLLOUTS_PER_PROMPT,
    )
    merged_path = output_root / "fig14_fixed_rollouts.jsonl"
    if len(completed) == FIG14_PROMPT_COUNT:
        merged = merge_chunks(
            output_root,
            "fig14-rollouts",
            FIG14_PROMPT_COUNT,
            contract_sha256=contract_sha,
            expected_records_per_chunk=FIG14_ROLLOUTS_PER_PROMPT,
            output_jsonl=merged_path,
            overwrite=overwrite_merge,
        )
        validated = validate_fig14_rollout_batch(merged, sampling)
        action_sha = action_batch_fingerprint(validated.values())
        runtime_atomic_json(
            output_root / "action_batch_manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                "protocol_id": "fig14-fixed-rollout-generation",
                "batch_id": batch_id,
                "contract_sha256": contract_sha,
                "action_batch_sha256": action_sha,
                "num_prompts": FIG14_PROMPT_COUNT,
                "num_rollouts": FIG14_TOTAL_ROLLOUTS,
                "rollouts_jsonl": str(merged_path.resolve()),
            },
            overwrite=overwrite_merge,
        )
    return {
        **progress,
        "completed_chunks": len(completed),
        "total_chunks": FIG14_PROMPT_COUNT,
        "contract_sha256": contract_sha,
        "batch_id": batch_id,
        "merged_rollouts": str(merged_path.resolve()) if merged_path.is_file() else None,
    }


def score_rollouts_transformers_sharded(
    rollouts: Sequence[ValidatedRollout],
    *,
    output_root: Path,
    output_jsonl: Path,
    role: str,
    model_source: str,
    revision: str,
    device: str,
    dtype: str,
    chunk_tokens: int,
    shard_index: int,
    num_shards: int,
    overwrite_merge: bool,
) -> dict[str, Any]:
    """Resume at rollout granularity and publish scores only when complete."""

    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must lie in [0, num_shards)")
    ordered = sorted(rollouts, key=lambda item: item.key)
    action_sha = action_batch_fingerprint(ordered)
    contract = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": "fig14-kv-chunked-action-scoring",
        "action_batch_sha256": action_sha,
        "role": role,
        "model": model_source,
        "revision": revision,
        "dtype": dtype,
        "chunk_tokens": chunk_tokens,
        "method": "causal-next-token",
    }
    contract_sha = canonical_sha256(contract)
    kind = f"fig14-score-{role}-{canonical_sha256([model_source, revision])[:12]}"
    ordinals = range(len(ordered))
    completed = completed_ordinals(
        output_root,
        kind,
        ordinals,
        contract_sha256=contract_sha,
        expected_records=1,
    )
    pending = [
        (ordinal, rollout)
        for ordinal, rollout in enumerate(ordered)
        if ordinal % num_shards == shard_index and ordinal not in completed
    ]
    # Load the model once, while the scorer still publishes one immutable
    # rollout chunk at a time.  This bounds lost work to the in-flight action.
    generated = 0
    if pending:
        pending_ordinals = [ordinal for ordinal, _ in pending]

        def publish(local_index: int, record: Mapping[str, Any]) -> None:
            nonlocal generated
            ordinal = pending_ordinals[local_index]
            write_chunk(
                output_root,
                kind,
                ordinal,
                contract_sha256=contract_sha,
                records=[record],
            )
            generated += 1

        scored = score_rollouts_transformers(
            [rollout for _, rollout in pending],
            role=role,
            model_source=model_source,
            revision=revision,
            device=device,
            dtype=dtype,
            chunk_tokens=chunk_tokens,
            record_callback=publish,
        )
        if len(scored) != len(pending):
            raise RuntimeError("scorer returned an unexpected number of records")
    completed = completed_ordinals(
        output_root,
        kind,
        ordinals,
        contract_sha256=contract_sha,
        expected_records=1,
    )
    if len(completed) == len(ordered):
        merge_chunks(
            output_root,
            kind,
            len(ordered),
            contract_sha256=contract_sha,
            expected_records_per_chunk=1,
            output_jsonl=output_jsonl,
            overwrite=overwrite_merge,
        )
    return {
        "contract_sha256": contract_sha,
        "action_batch_sha256": action_sha,
        "generated": generated,
        "completed_chunks": len(completed),
        "total_chunks": len(ordered),
        "output_jsonl": str(output_jsonl.resolve()) if output_jsonl.is_file() else None,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    analyze = commands.add_parser("analyze", help="validate fixed scores and write JSON/CSV/PNG")
    analyze.add_argument("--rollouts", type=Path, required=True)
    analyze.add_argument("--student-logprobs", type=Path, required=True)
    analyze.add_argument("--teacher-logprobs", type=Path, required=True)
    analyze.add_argument("--sampling-json", type=Path, required=True)
    analyze.add_argument("--student-model", required=True)
    analyze.add_argument("--student-revision", required=True)
    analyze.add_argument("--teacher-model", required=True)
    analyze.add_argument("--teacher-revision", required=True)
    analyze.add_argument("--output-dir", type=Path, required=True)
    analyze.add_argument("--overwrite", action="store_true")

    dual = commands.add_parser(
        "analyze-dual", help="analyze the same fixed actions under both Figure 14 teachers"
    )
    dual.add_argument("--rollouts", type=Path, required=True)
    dual.add_argument("--student-logprobs", type=Path, required=True)
    dual.add_argument("--sampling-json", type=Path, required=True)
    dual.add_argument("--student-model", required=True)
    dual.add_argument("--student-revision", required=True)
    dual.add_argument("--teacher", action="append", nargs=4, metavar=("LABEL", "MODEL", "REVISION", "LOGPROBS"), required=True)
    dual.add_argument("--bootstrap-replicates", type=_positive_int, default=BOOTSTRAP_REPLICATES)
    dual.add_argument("--output-dir", type=Path, required=True)
    dual.add_argument("--overwrite", action="store_true")

    generate = commands.add_parser(
        "generate", help="generate the fixed 1,070 prompts x four student actions in resumable chunks"
    )
    generate.add_argument("--input-parquet", type=Path, required=True)
    generate.add_argument("--student-model", required=True)
    generate.add_argument("--student-revision", required=True)
    generate.add_argument("--output-root", type=Path, required=True)
    generate.add_argument("--shard-index", type=int, default=0)
    generate.add_argument("--num-shards", type=_positive_int, default=1)
    generate.add_argument("--batch-size", type=_positive_int, default=8)
    generate.add_argument("--tensor-parallel-size", type=_positive_int, default=1)
    generate.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    generate.add_argument("--dtype", choices=("auto", "float16", "bfloat16"), default="bfloat16")
    generate.add_argument("--grader-utils", type=Path)
    generate.add_argument("--overwrite-merge", action="store_true")

    score = commands.add_parser("score", help="score one model on immutable token IDs")
    score.add_argument("--rollouts", type=Path, required=True)
    score.add_argument("--sampling-json", type=Path, required=True)
    score.add_argument("--role", choices=("student", "teacher"), required=True)
    score.add_argument("--model", required=True)
    score.add_argument("--revision", required=True)
    score.add_argument("--output-jsonl", type=Path, required=True)
    score.add_argument("--device", default="cuda:0")
    score.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    score.add_argument("--chunk-tokens", type=_positive_int, default=256)
    score.add_argument("--dry-run", action="store_true")
    score.add_argument("--overwrite", action="store_true")

    sharded = commands.add_parser(
        "score-sharded", help="KV-chunked scoring with atomic per-action resume chunks"
    )
    sharded.add_argument("--rollouts", type=Path, required=True)
    sharded.add_argument("--sampling-json", type=Path, required=True)
    sharded.add_argument("--role", choices=("student", "teacher"), required=True)
    sharded.add_argument("--model", required=True)
    sharded.add_argument("--revision", required=True)
    sharded.add_argument("--output-root", type=Path, required=True)
    sharded.add_argument("--output-jsonl", type=Path, required=True)
    sharded.add_argument("--device", default="cuda:0")
    sharded.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    sharded.add_argument("--chunk-tokens", type=_positive_int, default=256)
    sharded.add_argument("--shard-index", type=int, default=0)
    sharded.add_argument("--num-shards", type=_positive_int, default=1)
    sharded.add_argument("--overwrite-merge", action="store_true")
    return parser


def _jsonl_text(records: Sequence[Mapping[str, Any]]) -> str:
    return "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "generate":
            if args.shard_index < 0 or args.shard_index >= args.num_shards:
                raise ValueError("shard-index must lie in [0, num-shards)")
            report = prepare_fig14_generation(
                input_parquet=args.input_parquet,
                output_root=args.output_root,
                student_model=args.student_model,
                student_revision=args.student_revision,
                shard_index=args.shard_index,
                num_shards=args.num_shards,
                batch_size=args.batch_size,
                tensor_parallel_size=args.tensor_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
                dtype=args.dtype,
                grader_utils=args.grader_utils,
                overwrite_merge=args.overwrite_merge,
            )
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0

        sampling = read_json_object(args.sampling_json)
        rollout_records = read_jsonl(args.rollouts)
        validated = validate_rollouts(rollout_records, sampling)
        if args.command in {"score-sharded", "analyze-dual"}:
            validated = validate_fig14_rollout_batch(rollout_records, sampling)

        if args.command == "score-sharded":
            if args.shard_index < 0 or args.shard_index >= args.num_shards:
                raise ValueError("shard-index must lie in [0, num-shards)")
            report = score_rollouts_transformers_sharded(
                list(validated.values()),
                output_root=args.output_root,
                output_jsonl=args.output_jsonl,
                role=args.role,
                model_source=args.model,
                revision=args.revision,
                device=args.device,
                dtype=args.dtype,
                chunk_tokens=args.chunk_tokens,
                shard_index=args.shard_index,
                num_shards=args.num_shards,
                overwrite_merge=args.overwrite_merge,
            )
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0

        if args.command == "score":
            if args.output_jsonl.exists() and not args.overwrite:
                raise FileExistsError(f"refusing to overwrite {args.output_jsonl}")
            plan = {
                "command": "score",
                "role": args.role,
                "model": args.model,
                "revision": args.revision,
                "device": args.device,
                "dtype": args.dtype,
                "num_rollouts": len(validated),
                "batch_id": next(iter(validated.values())).batch_id,
                "output_jsonl": str(args.output_jsonl),
            }
            if args.dry_run:
                print(json.dumps(plan, indent=2, sort_keys=True))
                return 0
            scored = score_rollouts_transformers(
                list(validated.values()),
                role=args.role,
                model_source=args.model,
                revision=args.revision,
                device=args.device,
                dtype=args.dtype,
                chunk_tokens=args.chunk_tokens,
            )
            _atomic_text(args.output_jsonl, _jsonl_text(scored))
            print(json.dumps(plan, indent=2, sort_keys=True))
            return 0

        if args.command == "analyze-dual":
            if len(args.teacher) != 2:
                raise ValueError("analyze-dual requires exactly two --teacher declarations")
            teacher_labels = [str(spec[0]) for spec in args.teacher]
            if len(set(teacher_labels)) != 2:
                raise ValueError("teacher labels must be unique")
            action_sha = action_batch_fingerprint(validated.values())
            summaries: dict[str, Mapping[str, Any]] = {}
            for label, model, revision, raw_logprobs in args.teacher:
                rows = compute_sequence_rewards(
                    rollout_records,
                    read_jsonl(args.student_logprobs),
                    read_jsonl(Path(raw_logprobs)),
                    expected_sampling=sampling,
                    student_model=args.student_model,
                    student_revision=args.student_revision,
                    teacher_model=model,
                    teacher_revision=revision,
                )
                summary = summarize_sequence_rewards(
                    rows,
                    bootstrap_replicates=args.bootstrap_replicates,
                    bootstrap_seed=FIG14_SEED,
                )
                summary = {
                    **summary,
                    "teacher_label": label,
                    "student_model": {"source": args.student_model, "revision": args.student_revision},
                    "teacher_model": {"source": model, "revision": revision},
                    "sampling": dict(sampling),
                    "action_batch_sha256": action_sha,
                }
                safe_label = "".join(character if character.isalnum() else "-" for character in label).strip("-")
                write_analysis_outputs(
                    args.output_dir,
                    rows,
                    summary,
                    overwrite=args.overwrite,
                    artifact_stem=f"teacher-{safe_label or canonical_sha256(label)[:8]}",
                )
                summaries[label] = summary
            combined = combine_teacher_summaries(summaries, action_batch_sha256=action_sha)
            combined_path = args.output_dir / "sequence_reward_summary.json"
            if combined_path.exists() and not args.overwrite:
                raise FileExistsError(f"refusing to overwrite {combined_path}")
            _atomic_text(combined_path, json.dumps(combined, indent=2, sort_keys=True) + "\n")
            print(json.dumps({"summary": combined, "output": str(combined_path)}, indent=2))
            return 0

        rows = compute_sequence_rewards(
            rollout_records,
            read_jsonl(args.student_logprobs),
            read_jsonl(args.teacher_logprobs),
            expected_sampling=sampling,
            student_model=args.student_model,
            student_revision=args.student_revision,
            teacher_model=args.teacher_model,
            teacher_revision=args.teacher_revision,
        )
        summary = summarize_sequence_rewards(rows)
        summary = {
            **summary,
            "student_model": {"source": args.student_model, "revision": args.student_revision},
            "teacher_model": {"source": args.teacher_model, "revision": args.teacher_revision},
            "sampling": dict(sampling),
        }
        paths = write_analysis_outputs(args.output_dir, rows, summary, overwrite=args.overwrite)
        print(json.dumps({"summary": summary, "outputs": {key: str(value) for key, value in paths.items()}}, indent=2))
        return 0
    except (FileNotFoundError, FileExistsError, json.JSONDecodeError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
