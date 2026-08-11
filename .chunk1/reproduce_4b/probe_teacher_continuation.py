#!/usr/bin/env python3
"""Plan, validate, and summarize the paper's Figure 11(b) continuation probe.

The paper reports the structural protocol (2K DAPO prompts, full student
rollouts, strict length >16K, and 1K/4K/8K/16K prefixes) but does not disclose
the probe's generation seed, temperature, top-p, or exact checkpoint.  This
tool pins those choices as a seed-42 local adaptation and refuses to label
them as paper-reported.  ``run-student`` and ``run-teacher`` execute the GPU
workload through atomic resumable chunks; planning, validation, bootstrap
analysis, and rendering remain CPU-only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from analyze_sequence_reward import (
    SCHEMA_VERSION,
    ValidatedRollout,
    _atomic_text,
    _canonical_json,
    _nonempty_string,
    _nonnegative_int,
    _token_ids,
    read_json_object,
    read_jsonl,
    validate_rollouts,
    validate_sampling_metadata,
)
from probe_runtime import (
    atomic_json as runtime_atomic_json,
    canonical_sha256,
    completed_ordinals,
    generate_rollout_chunks_vllm,
    load_dapo_sample,
    load_grader,
    merge_chunks,
    write_chunk,
)


PAPER_PROMPT_COUNT = 2_000
STRICT_LENGTH_THRESHOLD = 16_384
PREFIX_LENGTHS = (1_024, 4_096, 8_192, 16_384)
PROTOCOL_ID = "fig11b-teacher-continuation"
PROBE_SEED = 42
MAX_TOTAL_RESPONSE_TOKENS = 31_744
MIN_SCIENTIFIC_ROLLOUTS = 30
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_CONFIDENCE = 0.95
ALLOWED_SAMPLING_PROVENANCE = {
    "paper-undisclosed-explicit-local",
    "author-code-verified",
}


def _validate_registered_sampling(sampling: Mapping[str, Any], label: str) -> None:
    expected = {
        "seed": PROBE_SEED,
        "n": 1,
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": MAX_TOTAL_RESPONSE_TOKENS,
        "thinking": "off",
    }
    for key, value in expected.items():
        if sampling.get(key) != value:
            raise ValueError(f"{label}.{key} must be {value!r}")


def validate_protocol(protocol: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate paper-reported invariants and explicit local sampling fields."""

    if not isinstance(protocol, Mapping):
        raise ValueError("protocol must be a JSON object")
    if protocol.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("protocol has unsupported schema_version")
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"protocol_id must be {PROTOCOL_ID!r}")
    if protocol.get("paper_reference") != "Figure 11(b)":
        raise ValueError("paper_reference must be 'Figure 11(b)'")
    if protocol.get("dataset") != "DAPO-Math-17K":
        raise ValueError("Figure 11(b) protocol dataset must be DAPO-Math-17K")
    if protocol.get("prompt_sample_size") != PAPER_PROMPT_COUNT:
        raise ValueError(f"paper protocol requires exactly {PAPER_PROMPT_COUNT} sampled prompts")
    if protocol.get("prompt_sample_seed") != PROBE_SEED:
        raise ValueError(f"prompt_sample_seed must be {PROBE_SEED}")
    if protocol.get("selection") != {"response_tokens_strictly_greater_than": STRICT_LENGTH_THRESHOLD}:
        raise ValueError(f"selection must be strict response length > {STRICT_LENGTH_THRESHOLD}")
    prefixes = protocol.get("prefix_lengths")
    if prefixes != list(PREFIX_LENGTHS):
        raise ValueError(f"prefix_lengths must be exactly {list(PREFIX_LENGTHS)}")
    provenance = protocol.get("sampling_provenance")
    if provenance not in ALLOWED_SAMPLING_PROVENANCE:
        raise ValueError(
            "sampling_provenance must disclose paper-undisclosed-explicit-local "
            "or provide author-code-verified evidence"
        )
    if provenance == "author-code-verified":
        _nonempty_string(protocol.get("sampling_evidence"), "protocol.sampling_evidence")
    elif "sampling_evidence" in protocol:
        raise ValueError("local sampling must not claim author-code sampling evidence")
    student_sampling = protocol.get("student_sampling")
    teacher_sampling = protocol.get("teacher_sampling")
    validate_sampling_metadata(student_sampling, student_sampling, "protocol.student_sampling")
    validate_sampling_metadata(teacher_sampling, teacher_sampling, "protocol.teacher_sampling")
    _validate_registered_sampling(student_sampling, "protocol.student_sampling")
    _validate_registered_sampling(teacher_sampling, "protocol.teacher_sampling")
    if student_sampling["model"] == teacher_sampling["model"] and student_sampling["revision"] == teacher_sampling["revision"]:
        raise ValueError("student and teacher model identities must differ")
    return protocol


def make_protocol_template() -> dict[str, Any]:
    """Return the pinned local-adaptation protocol with disclosures visible."""

    student_sampling = {
        "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "revision": "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562",
        "seed": 42,
        "n": 1,
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 31_744,
        "thinking": "off",
    }
    teacher_sampling = {
        **student_sampling,
        "model": "hbx/JustRL-DeepSeek-1.5B",
        "revision": "0637e4096c789c67f9eecbe8355e0bdeddede1c2",
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "paper_reference": "Figure 11(b)",
        "dataset": "DAPO-Math-17K",
        "prompt_sample_size": PAPER_PROMPT_COUNT,
        "prompt_sample_seed": 42,
        "selection": {"response_tokens_strictly_greater_than": STRICT_LENGTH_THRESHOLD},
        "prefix_lengths": list(PREFIX_LENGTHS),
        "sampling_provenance": "paper-undisclosed-explicit-local",
        "student_sampling": student_sampling,
        "teacher_sampling": teacher_sampling,
        "disclosure": (
            "The paper does not disclose Figure 11(b) sampling/checkpoint details. "
            "These pinned seed-42 sampling fields are a preregistered local explicit adaptation."
        ),
    }


def validate_full_student_batch(
    records: Iterable[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> dict[tuple[str, int], ValidatedRollout]:
    validated_protocol = validate_protocol(protocol)
    rollouts = validate_rollouts(records, validated_protocol["student_sampling"])
    prompt_count = len({rollout.example_id for rollout in rollouts.values()})
    if prompt_count != PAPER_PROMPT_COUNT:
        raise ValueError(
            f"full student batch has {prompt_count} unique prompts, expected exactly {PAPER_PROMPT_COUNT}"
        )
    return rollouts


def select_long_rollouts(
    rollouts: Iterable[ValidatedRollout], threshold: int = STRICT_LENGTH_THRESHOLD
) -> list[ValidatedRollout]:
    """Apply the paper's strict ``length > 16K`` rule (not ``>=``)."""

    if threshold != STRICT_LENGTH_THRESHOLD:
        raise ValueError(f"Figure 11(b) threshold must remain {STRICT_LENGTH_THRESHOLD}")
    selected = [rollout for rollout in rollouts if len(rollout.response_token_ids) > threshold]
    return sorted(selected, key=lambda item: item.key)


def continuation_pair_id(rollout: ValidatedRollout, prefix_length: int) -> str:
    return f"{rollout.batch_id}:{rollout.example_id}:{rollout.rollout_id}:prefix-{prefix_length}"


def build_continuation_plan(
    rollouts: Iterable[ValidatedRollout],
    teacher_sampling: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Create exactly four paired teacher-continuation jobs per selected rollout."""

    selected = select_long_rollouts(rollouts)
    if not selected:
        raise ValueError("no student rollout is strictly longer than 16384 tokens")
    validate_sampling_metadata(teacher_sampling, teacher_sampling, "teacher_sampling")
    _validate_registered_sampling(teacher_sampling, "teacher_sampling")
    plan: list[dict[str, Any]] = []
    for rollout in selected:
        for prefix_length in PREFIX_LENGTHS:
            if len(rollout.response_token_ids) <= prefix_length:
                raise ValueError(
                    f"selected rollout {rollout.key!r} is not long enough for prefix {prefix_length}"
                )
            pair_id = continuation_pair_id(rollout, prefix_length)
            continuation_sampling = {
                **teacher_sampling,
                "max_tokens": MAX_TOTAL_RESPONSE_TOKENS - prefix_length,
            }
            plan.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "protocol_id": PROTOCOL_ID,
                    "pair_id": pair_id,
                    "batch_id": rollout.batch_id,
                    "example_id": rollout.example_id,
                    "rollout_id": rollout.rollout_id,
                    "rollout_fingerprint": rollout.fingerprint,
                    "student_response_length": len(rollout.response_token_ids),
                    "student_correct": rollout.correct,
                    "prefix_length": prefix_length,
                    "prompt_token_ids": list(rollout.prompt_token_ids),
                    "prefix_token_ids": list(rollout.response_token_ids[:prefix_length]),
                    "max_total_response_tokens": MAX_TOTAL_RESPONSE_TOKENS,
                    "max_continuation_tokens": MAX_TOTAL_RESPONSE_TOKENS - prefix_length,
                    "teacher_sampling": continuation_sampling,
                    **({"answer": rollout.answer} if rollout.answer is not None else {}),
                }
            )
    validate_plan(plan, selected, teacher_sampling)
    return plan


def validate_plan(
    plan_records: Iterable[Mapping[str, Any]],
    selected_rollouts: Iterable[ValidatedRollout],
    teacher_sampling: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    """Validate prefix content, the four-way grid, and stable pair IDs."""

    selected = {rollout.key: rollout for rollout in selected_rollouts}
    if not selected:
        raise ValueError("selected rollout set is empty")
    output: dict[str, Mapping[str, Any]] = {}
    observed: Counter[tuple[str, int]] = Counter()
    observed_prefixes: dict[tuple[str, int], set[int]] = {}
    for index, record in enumerate(plan_records, start=1):
        label = f"plan record {index}"
        if not isinstance(record, Mapping):
            raise ValueError(f"{label} must be an object")
        if record.get("schema_version") != SCHEMA_VERSION or record.get("protocol_id") != PROTOCOL_ID:
            raise ValueError(f"{label} has wrong schema/protocol")
        pair_id = _nonempty_string(record.get("pair_id"), f"{label}.pair_id")
        if pair_id in output:
            raise ValueError(f"duplicate continuation pair_id {pair_id!r}")
        example_id = _nonempty_string(record.get("example_id"), f"{label}.example_id")
        rollout_id = _nonnegative_int(record.get("rollout_id"), f"{label}.rollout_id")
        key = (example_id, rollout_id)
        if key not in selected:
            raise ValueError(f"{label} references an unselected rollout {key!r}")
        rollout = selected[key]
        prefix_length = _nonnegative_int(record.get("prefix_length"), f"{label}.prefix_length")
        if prefix_length not in PREFIX_LENGTHS:
            raise ValueError(f"{label} has unsupported prefix length {prefix_length}")
        expected_pair = continuation_pair_id(rollout, prefix_length)
        if pair_id != expected_pair:
            raise ValueError(f"{label} pair_id mismatch: {pair_id!r} != {expected_pair!r}")
        if record.get("batch_id") != rollout.batch_id:
            raise ValueError(f"{label} batch ID mismatch")
        if record.get("rollout_fingerprint") != rollout.fingerprint:
            raise ValueError(f"{label} rollout fingerprint mismatch")
        if record.get("student_response_length") != len(rollout.response_token_ids):
            raise ValueError(f"{label} student response length mismatch")
        if record.get("student_correct") is not rollout.correct:
            raise ValueError(f"{label} student correctness mismatch")
        prompt_ids = _token_ids(record.get("prompt_token_ids"), f"{label}.prompt_token_ids", allow_empty=False)
        prefix_ids = _token_ids(record.get("prefix_token_ids"), f"{label}.prefix_token_ids", allow_empty=False)
        if prompt_ids != rollout.prompt_token_ids:
            raise ValueError(f"{label} prompt token IDs mismatch")
        if len(prefix_ids) != prefix_length:
            raise ValueError(f"{label} prefix token count does not equal prefix_length")
        if prefix_ids != rollout.response_token_ids[:prefix_length]:
            raise ValueError(f"{label} prefix is not the exact student response prefix")
        expected_sampling = {
            **teacher_sampling,
            "max_tokens": MAX_TOTAL_RESPONSE_TOKENS - prefix_length,
        }
        validate_sampling_metadata(
            record.get("teacher_sampling"), expected_sampling, f"{label}.teacher_sampling"
        )
        if record.get("max_total_response_tokens") != MAX_TOTAL_RESPONSE_TOKENS:
            raise ValueError(f"{label} max_total_response_tokens mismatch")
        if record.get("max_continuation_tokens") != MAX_TOTAL_RESPONSE_TOKENS - prefix_length:
            raise ValueError(f"{label} max_continuation_tokens mismatch")
        output[pair_id] = record
        observed[key] += 1
        observed_prefixes.setdefault(key, set()).add(prefix_length)
    if len(output) != len(selected) * len(PREFIX_LENGTHS):
        raise ValueError(
            f"continuation plan has {len(output)} pairs, expected {len(selected) * len(PREFIX_LENGTHS)}"
        )
    malformed = [
        key
        for key in selected
        if observed[key] != len(PREFIX_LENGTHS) or observed_prefixes.get(key) != set(PREFIX_LENGTHS)
    ]
    if malformed:
        raise ValueError(f"{len(malformed)} selected rollouts lack the exact four prefix pairs")
    return output


def validate_continuation_results(
    plan_records: Iterable[Mapping[str, Any]],
    result_records: Iterable[Mapping[str, Any]],
    teacher_sampling: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Require a one-to-one result for every planned continuation pair."""

    validate_sampling_metadata(teacher_sampling, teacher_sampling, "teacher_sampling")
    _validate_registered_sampling(teacher_sampling, "teacher_sampling")

    plan: dict[str, Mapping[str, Any]] = {}
    for index, record in enumerate(plan_records, start=1):
        if not isinstance(record, Mapping):
            raise ValueError(f"plan record {index} must be an object")
        pair_id = _nonempty_string(record.get("pair_id"), f"plan record {index}.pair_id")
        if pair_id in plan:
            raise ValueError(f"duplicate plan pair_id {pair_id!r}")
        plan[pair_id] = record
    if not plan:
        raise ValueError("continuation plan is empty")
    results: dict[str, Mapping[str, Any]] = {}
    for index, record in enumerate(result_records, start=1):
        label = f"continuation result {index}"
        if not isinstance(record, Mapping):
            raise ValueError(f"{label} must be an object")
        if record.get("schema_version") != SCHEMA_VERSION or record.get("protocol_id") != PROTOCOL_ID:
            raise ValueError(f"{label} has wrong schema/protocol")
        pair_id = _nonempty_string(record.get("pair_id"), f"{label}.pair_id")
        if pair_id in results:
            raise ValueError(f"duplicate result pair_id {pair_id!r}")
        if pair_id not in plan:
            raise ValueError(f"{label} has unplanned pair_id {pair_id!r}")
        planned = plan[pair_id]
        for key in ("batch_id", "example_id", "rollout_id", "rollout_fingerprint", "prefix_length"):
            if record.get(key) != planned.get(key):
                raise ValueError(f"{label}.{key} does not match its planned pair")
        correct = record.get("teacher_correct")
        if not isinstance(correct, bool):
            raise ValueError(f"{label}.teacher_correct must be boolean")
        response_ids = _token_ids(
            record.get("teacher_continuation_token_ids"),
            f"{label}.teacher_continuation_token_ids",
            allow_empty=False,
        )
        if not response_ids:
            raise ValueError(f"{label} has an empty continuation")
        validate_sampling_metadata(
            record.get("sampling"), planned.get("teacher_sampling"), f"{label}.sampling"
        )
        budget = planned.get("max_continuation_tokens")
        if not isinstance(budget, int) or len(response_ids) > budget:
            raise ValueError(f"{label} exceeds its registered continuation token budget")
        full_ids = record.get("full_response_token_ids")
        expected_full = tuple(planned["prefix_token_ids"]) + response_ids
        if full_ids is not None and _token_ids(
            full_ids, f"{label}.full_response_token_ids", allow_empty=False
        ) != expected_full:
            raise ValueError(f"{label} full response is not exact prefix plus continuation")
        results[pair_id] = record
    planned_ids = set(plan)
    result_ids = set(results)
    if planned_ids != result_ids:
        missing = sorted(planned_ids - result_ids)[:5]
        extra = sorted(result_ids - planned_ids)[:5]
        raise ValueError(f"continuation pairing failure; missing={missing}, extra={extra}")
    combined: list[dict[str, Any]] = []
    for pair_id in sorted(planned_ids):
        planned = plan[pair_id]
        result = results[pair_id]
        combined.append(
            {
                "pair_id": pair_id,
                "batch_id": planned["batch_id"],
                "example_id": planned["example_id"],
                "rollout_id": planned["rollout_id"],
                "rollout_fingerprint": planned["rollout_fingerprint"],
                "prefix_length": planned["prefix_length"],
                "student_correct": planned["student_correct"],
                "teacher_correct": result["teacher_correct"],
            }
        )
    return combined


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _bootstrap_continuation(
    by_prefix: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    keyed: dict[int, dict[tuple[str, int], Mapping[str, Any]]] = {}
    for prefix, group in by_prefix.items():
        mapping: dict[tuple[str, int], Mapping[str, Any]] = {}
        for row in group:
            key = (str(row["example_id"]), int(row["rollout_id"]))
            if key in mapping:
                raise ValueError(f"duplicate paired rollout {key!r} at prefix {prefix}")
            mapping[key] = row
        keyed[prefix] = mapping
    key_sets = [set(mapping) for mapping in keyed.values()]
    if not key_sets or any(keys != key_sets[0] for keys in key_sets[1:]):
        raise ValueError("prefix groups do not contain the exact same selected rollouts")
    keys = sorted(key_sets[0])
    for key in keys:
        student_values = {bool(keyed[prefix][key]["student_correct"]) for prefix in PREFIX_LENGTHS}
        fingerprints = {keyed[prefix][key].get("rollout_fingerprint") for prefix in PREFIX_LENGTHS}
        if len(student_values) != 1 or len(fingerprints) != 1:
            raise ValueError(f"paired rollout {key!r} changes label or fingerprint across prefixes")
    generator = random.Random(seed)
    distributions = {prefix: [] for prefix in PREFIX_LENGTHS}
    deltas: list[float] = []
    for _ in range(replicates):
        sampled = [keys[generator.randrange(len(keys))] for _ in keys]
        gains: dict[int, float] = {}
        for prefix in PREFIX_LENGTHS:
            values = [
                float(bool(keyed[prefix][key]["teacher_correct"]))
                - float(bool(keyed[prefix][key]["student_correct"]))
                for key in sampled
            ]
            gains[prefix] = sum(values) / len(values)
            distributions[prefix].append(gains[prefix])
        deltas.append(gains[PREFIX_LENGTHS[0]] - gains[PREFIX_LENGTHS[-1]])
    alpha = (1.0 - BOOTSTRAP_CONFIDENCE) / 2.0
    return {
        "unit": "selected student rollout",
        "seed": seed,
        "replicates": replicates,
        "confidence": BOOTSTRAP_CONFIDENCE,
        "prefix_gain_ci": {
            str(prefix): {
                "lower": _percentile(values, alpha),
                "upper": _percentile(values, 1.0 - alpha),
            }
            for prefix, values in distributions.items()
        },
        "gain_1024_minus_16384_ci": {
            "lower": _percentile(deltas, alpha),
            "upper": _percentile(deltas, 1.0 - alpha),
        },
    }


def summarize_paired_results(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = PROBE_SEED,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("no paired continuation results")
    by_prefix: dict[int, list[Mapping[str, Any]]] = {prefix: [] for prefix in PREFIX_LENGTHS}
    for row in rows:
        prefix = row.get("prefix_length")
        if prefix not in by_prefix:
            raise ValueError(f"unexpected prefix length {prefix!r}")
        if not isinstance(row.get("student_correct"), bool) or not isinstance(row.get("teacher_correct"), bool):
            raise ValueError("paired correctness fields must be boolean")
        by_prefix[prefix].append(row)
    counts = {len(group) for group in by_prefix.values()}
    if 0 in counts or len(counts) != 1:
        raise ValueError("prefix groups are not a complete paired grid")
    summaries: list[dict[str, Any]] = []
    for prefix in PREFIX_LENGTHS:
        group = by_prefix[prefix]
        n = len(group)
        student_correct = sum(bool(row["student_correct"]) for row in group)
        teacher_correct = sum(bool(row["teacher_correct"]) for row in group)
        student_accuracy = student_correct / n
        teacher_accuracy = teacher_correct / n
        summaries.append(
            {
                "prefix_length": prefix,
                "num_pairs": n,
                "student_correct": student_correct,
                "teacher_correct": teacher_correct,
                "student_accuracy": student_accuracy,
                "teacher_accuracy": teacher_accuracy,
                "accuracy_gain": teacher_accuracy - student_accuracy,
            }
        )
    bootstrap = _bootstrap_continuation(
        by_prefix, replicates=bootstrap_replicates, seed=bootstrap_seed
    )
    gains = [float(row["accuracy_gain"]) for row in summaries]
    delta_ci = bootstrap["gain_1024_minus_16384_ci"]
    monotonic = all(left >= right for left, right in zip(gains, gains[1:]))
    num_selected = next(iter(counts))
    if num_selected < MIN_SCIENTIFIC_ROLLOUTS:
        conclusion = "inconclusive"
        reason = (
            f"only {num_selected} strict >16K rollouts were selected; "
            f"the preregistered minimum is {MIN_SCIENTIFIC_ROLLOUTS}"
        )
    elif monotonic and float(delta_ci["lower"]) > 0.0:
        conclusion = "replicated"
        reason = "accuracy gains decrease monotonically and the paired 1K-16K CI is above zero"
    elif float(delta_ci["upper"]) <= 0.0:
        conclusion = "not_replicated_at_seed_42"
        reason = "the paired 1K-16K gain difference is non-positive"
    else:
        conclusion = "inconclusive"
        reason = "the monotonicity/paired-CI preregistration is not decisive"
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "selection": f"student response length > {STRICT_LENGTH_THRESHOLD} tokens",
        "num_selected_rollouts": num_selected,
        "prefix_results": summaries,
        "bootstrap": bootstrap,
        "monotonic_nonincreasing_gain": monotonic,
        "conclusion_state": conclusion,
        "conclusion_reason": reason,
        "paper_comparable": False,
    }


def _write_summary_plot(path: Path, summary: Mapping[str, Any]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for PNG output") from exc
    rows = summary["prefix_results"]
    prefixes = [row["prefix_length"] for row in rows]
    gains = [row["accuracy_gain"] for row in rows]
    figure, axis = plt.subplots(figsize=(6.5, 4.2))
    axis.plot(prefixes, gains, marker="o", linewidth=2)
    axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axis.set_xticks(prefixes, labels=["1K", "4K", "8K", "16K"])
    axis.set_xlabel("Student prefix truncation length")
    axis.set_ylabel("Teacher accuracy - student accuracy")
    axis.set_title("Teacher continuation accuracy gain")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_summary_outputs(
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    *,
    overwrite: bool,
) -> dict[str, Path]:
    paths = {
        "json": output_dir / "teacher_continuation_summary.json",
        "csv": output_dir / "teacher_continuation_pairs.csv",
        "png": output_dir / "teacher_continuation_gain.png",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing outputs: {existing}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_text(paths["json"], json.dumps(summary, indent=2, sort_keys=True) + "\n")
    fields = [
        "pair_id",
        "batch_id",
        "example_id",
        "rollout_id",
        "rollout_fingerprint",
        "prefix_length",
        "student_correct",
        "teacher_correct",
    ]
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", prefix=".teacher_continuation.", dir=output_dir, delete=False
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, paths["csv"])
    _write_summary_plot(paths["png"], summary)
    return paths


def _jsonl_text(records: Sequence[Mapping[str, Any]]) -> str:
    return "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)


def run_student_rollout_shard(
    *,
    input_parquet: Path,
    protocol: Mapping[str, Any],
    output_root: Path,
    shard_index: int,
    num_shards: int,
    batch_size: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    dtype: str,
    grader_utils: Path | None,
    overwrite_merge: bool,
) -> dict[str, Any]:
    """Generate one crash-safe shard of the exact 2,000-prompt student batch."""

    protocol = validate_protocol(protocol)
    samples, selection = load_dapo_sample(input_parquet, PAPER_PROMPT_COUNT, PROBE_SEED)
    contract = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "phase": "student-full-rollout",
        "protocol": dict(protocol),
        "selection": selection,
    }
    contract_sha = canonical_sha256(contract)
    contract_path = output_root / "student_generation_contract.json"
    if contract_path.is_file():
        if _canonical_json(read_json_object(contract_path)) != _canonical_json(contract):
            raise ValueError("student generation contract changed under an existing output root")
    else:
        runtime_atomic_json(contract_path, contract)
    batch_id = f"fig11b-seed42-{contract_sha[:16]}"
    progress = generate_rollout_chunks_vllm(
        samples,
        output_root=output_root,
        chunk_kind="fig11b-student-rollouts",
        contract_sha256=contract_sha,
        batch_id=batch_id,
        sampling=protocol["student_sampling"],
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
        "fig11b-student-rollouts",
        range(PAPER_PROMPT_COUNT),
        contract_sha256=contract_sha,
        expected_records=1,
    )
    merged_path = output_root / "student_rollouts.jsonl"
    if len(completed) == PAPER_PROMPT_COUNT:
        if not merged_path.is_file() or overwrite_merge:
            merged = merge_chunks(
                output_root,
                "fig11b-student-rollouts",
                PAPER_PROMPT_COUNT,
                contract_sha256=contract_sha,
                expected_records_per_chunk=1,
                output_jsonl=merged_path,
                overwrite=overwrite_merge,
            )
        else:
            merged = read_jsonl(merged_path)
        validated = validate_full_student_batch(merged, protocol)
        runtime_atomic_json(
            output_root / "student_batch_manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                "protocol_id": PROTOCOL_ID,
                "batch_id": batch_id,
                "contract_sha256": contract_sha,
                "num_prompts": PAPER_PROMPT_COUNT,
                "num_rollouts": len(validated),
                "num_strict_long_rollouts": len(select_long_rollouts(validated.values())),
                "student_rollouts": str(merged_path.resolve()),
            },
            overwrite=overwrite_merge,
        )
    return {
        **progress,
        "completed_chunks": len(completed),
        "total_chunks": PAPER_PROMPT_COUNT,
        "contract_sha256": contract_sha,
        "batch_id": batch_id,
        "merged_rollouts": str(merged_path.resolve()) if merged_path.is_file() else None,
    }


def _generate_teacher_continuation_chunks(
    plan: Sequence[Mapping[str, Any]],
    *,
    output_root: Path,
    contract_sha256: str,
    shard_index: int,
    num_shards: int,
    batch_size: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    dtype: str,
    grader_utils: Path | None,
) -> dict[str, int]:
    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must lie in [0, num_shards)")
    ordinals = range(len(plan))
    completed = completed_ordinals(
        output_root,
        "fig11b-teacher-continuations",
        ordinals,
        contract_sha256=contract_sha256,
        expected_records=1,
    )
    pending = [
        (ordinal, plan[ordinal])
        for ordinal in ordinals
        if ordinal % num_shards == shard_index and ordinal not in completed
    ]
    if not pending:
        return {"assigned": sum(index % num_shards == shard_index for index in ordinals), "generated": 0}
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:  # pragma: no cover - GPU runtime only
        raise RuntimeError("teacher continuation generation requires vLLM") from exc
    base_sampling = plan[0]["teacher_sampling"]
    model_source = str(base_sampling["model"])
    llm_kwargs: dict[str, Any] = {
        "model": model_source,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "dtype": dtype,
        "trust_remote_code": True,
    }
    if not Path(model_source).expanduser().exists() and base_sampling["revision"] != "local":
        llm_kwargs["revision"] = str(base_sampling["revision"])
        llm_kwargs["tokenizer_revision"] = str(base_sampling["revision"])
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    grader = load_grader(grader_utils)
    generated = 0
    # Each prefix has a distinct continuation budget.  Grouping by prefix is
    # required because SamplingParams applies one max_tokens value per call.
    for prefix_length in PREFIX_LENGTHS:
        group = [(ordinal, row) for ordinal, row in pending if row["prefix_length"] == prefix_length]
        for start in range(0, len(group), batch_size):
            current = group[start : start + batch_size]
            prompts = [
                {
                    "prompt_token_ids": list(row["prompt_token_ids"]) + list(row["prefix_token_ids"])
                }
                for _, row in current
            ]
            sampling = current[0][1]["teacher_sampling"]
            params = SamplingParams(
                n=1,
                temperature=float(sampling["temperature"]),
                top_p=float(sampling["top_p"]),
                max_tokens=int(sampling["max_tokens"]),
                seed=int(sampling["seed"]),
            )
            outputs = llm.generate(prompts, params, use_tqdm=False)
            if len(outputs) != len(current):
                raise RuntimeError("vLLM returned a different number of continuation outputs")
            for (ordinal, planned), request_output in zip(current, outputs, strict=True):
                if len(request_output.outputs) != 1:
                    raise RuntimeError("continuation generation requires exactly one candidate")
                candidate = request_output.outputs[0]
                continuation_ids = [int(value) for value in (getattr(candidate, "token_ids", ()) or ())]
                if not continuation_ids:
                    continuation_ids = tokenizer.encode(candidate.text, add_special_tokens=False)
                full_response_ids = list(planned["prefix_token_ids"]) + continuation_ids
                full_response = tokenizer.decode(full_response_ids, skip_special_tokens=True)
                answer = planned.get("answer")
                if not isinstance(answer, str) or not answer:
                    raise ValueError(f"planned pair {planned['pair_id']} lacks a ground-truth answer")
                result = {
                    "schema_version": SCHEMA_VERSION,
                    "protocol_id": PROTOCOL_ID,
                    "pair_id": planned["pair_id"],
                    "batch_id": planned["batch_id"],
                    "example_id": planned["example_id"],
                    "rollout_id": planned["rollout_id"],
                    "rollout_fingerprint": planned["rollout_fingerprint"],
                    "prefix_length": prefix_length,
                    "teacher_continuation_token_ids": continuation_ids,
                    "full_response_token_ids": full_response_ids,
                    "teacher_correct": bool(grader.grade_answer_verl(full_response, answer)),
                    "sampling": dict(sampling),
                }
                write_chunk(
                    output_root,
                    "fig11b-teacher-continuations",
                    ordinal,
                    contract_sha256=contract_sha256,
                    records=[result],
                )
                generated += 1
    return {
        "assigned": sum(index % num_shards == shard_index for index in ordinals),
        "generated": generated,
    }


def run_teacher_continuation_shard(
    *,
    student_rollouts: Path,
    protocol: Mapping[str, Any],
    output_root: Path,
    shard_index: int,
    num_shards: int,
    batch_size: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    dtype: str,
    grader_utils: Path | None,
    bootstrap_replicates: int,
    overwrite_merge: bool,
) -> dict[str, Any]:
    """Continue exact token prefixes and publish the paired Figure 11(b) result."""

    protocol = validate_protocol(protocol)
    raw_rollouts = read_jsonl(student_rollouts)
    all_rollouts = validate_full_student_batch(raw_rollouts, protocol)
    selected = select_long_rollouts(all_rollouts.values())
    if not selected:
        raise ValueError("no student rollout is strictly longer than 16384 tokens")
    plan = build_continuation_plan(selected, protocol["teacher_sampling"])
    contract = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "phase": "teacher-continuation",
        "protocol": dict(protocol),
        "student_batch_sha256": canonical_sha256(raw_rollouts),
        "selected_rollout_fingerprints": [row.fingerprint for row in selected],
        "plan_sha256": canonical_sha256(plan),
    }
    contract_sha = canonical_sha256(contract)
    contract_path = output_root / "teacher_generation_contract.json"
    if contract_path.is_file():
        if _canonical_json(read_json_object(contract_path)) != _canonical_json(contract):
            raise ValueError("teacher generation contract changed under an existing output root")
    else:
        runtime_atomic_json(contract_path, contract)
    plan_path = output_root / "continuation_plan.jsonl"
    if plan_path.is_file():
        validate_plan(read_jsonl(plan_path), selected, protocol["teacher_sampling"])
    else:
        _atomic_text(plan_path, _jsonl_text(plan))
    if len(selected) < MIN_SCIENTIFIC_ROLLOUTS:
        summary = {
            "schema_version": SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "protocol": "scientific",
            "paper_reference": "Figure 11(b)",
            "status": "scientific_result_available",
            "conclusion_state": "inconclusive",
            "conclusion_reason": (
                f"only {len(selected)} strict >16K rollouts were selected from the fixed 2,000 prompts; "
                f"minimum={MIN_SCIENTIFIC_ROLLOUTS}; no resampling was performed"
            ),
            "num_source_prompts": PAPER_PROMPT_COUNT,
            "num_selected_rollouts": len(selected),
            "prefix_results": [],
            "contract_sha256": contract_sha,
            "student_batch_sha256": contract["student_batch_sha256"],
            "paper_comparable": False,
        }
        summary_path = output_root / "teacher_continuation_summary.json"
        if not summary_path.is_file() or overwrite_merge:
            runtime_atomic_json(summary_path, summary, overwrite=overwrite_merge)
        return {
            "conclusion_state": "inconclusive",
            "num_selected_rollouts": len(selected),
            "generated": 0,
            "summary": str(summary_path.resolve()),
        }
    progress = _generate_teacher_continuation_chunks(
        plan,
        output_root=output_root,
        contract_sha256=contract_sha,
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
        "fig11b-teacher-continuations",
        range(len(plan)),
        contract_sha256=contract_sha,
        expected_records=1,
    )
    results_path = output_root / "teacher_continuation_results.jsonl"
    summary_path = output_root / "teacher_continuation_summary.json"
    if len(completed) == len(plan):
        if not results_path.is_file() or overwrite_merge:
            result_records = merge_chunks(
                output_root,
                "fig11b-teacher-continuations",
                len(plan),
                contract_sha256=contract_sha,
                expected_records_per_chunk=1,
                output_jsonl=results_path,
                overwrite=overwrite_merge,
            )
        else:
            result_records = read_jsonl(results_path)
        paired = validate_continuation_results(plan, result_records, protocol["teacher_sampling"])
        summary = summarize_paired_results(
            paired,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=PROBE_SEED,
        )
        summary = {
            **summary,
            "protocol": "scientific",
            "status": "scientific_result_available",
            "contract_sha256": contract_sha,
            "student_batch_sha256": contract["student_batch_sha256"],
            "paper_reported_target_gains": {
                "1024": 0.3659,
                "4096": 0.2709,
                "8192": 0.1522,
                "16384": 0.0237,
            },
            "sampling_provenance": protocol["sampling_provenance"],
            "student_sampling": protocol["student_sampling"],
            "teacher_sampling": protocol["teacher_sampling"],
        }
        write_summary_outputs(output_root, paired, summary, overwrite=overwrite_merge)
    return {
        **progress,
        "num_selected_rollouts": len(selected),
        "completed_chunks": len(completed),
        "total_chunks": len(plan),
        "contract_sha256": contract_sha,
        "summary": str(summary_path.resolve()) if summary_path.is_file() else None,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    template = commands.add_parser("protocol-template", help="write the pinned explicit local-adaptation protocol")
    template.add_argument("--output-json", type=Path)
    template.add_argument("--overwrite", action="store_true")

    plan = commands.add_parser("plan", help="select >16K rollouts and emit the four-prefix workload")
    plan.add_argument("--student-rollouts", type=Path, required=True)
    plan.add_argument("--protocol-json", type=Path, required=True)
    plan.add_argument("--output-jsonl", type=Path, required=True)
    plan.add_argument("--dry-run", action="store_true")
    plan.add_argument("--overwrite", action="store_true")

    validate = commands.add_parser("validate", help="validate a plan against its source rollout batch")
    validate.add_argument("--student-rollouts", type=Path, required=True)
    validate.add_argument("--protocol-json", type=Path, required=True)
    validate.add_argument("--plan-jsonl", type=Path, required=True)

    summarize = commands.add_parser("summarize", help="validate paired results and write JSON/CSV/PNG")
    summarize.add_argument("--student-rollouts", type=Path, required=True)
    summarize.add_argument("--protocol-json", type=Path, required=True)
    summarize.add_argument("--plan-jsonl", type=Path, required=True)
    summarize.add_argument("--results-jsonl", type=Path, required=True)
    summarize.add_argument("--output-dir", type=Path, required=True)
    summarize.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    summarize.add_argument("--overwrite", action="store_true")

    run_student = commands.add_parser(
        "run-student", help="generate the exact seed-42 2,000-prompt student batch with resume chunks"
    )
    run_student.add_argument("--input-parquet", type=Path, required=True)
    run_student.add_argument("--protocol-json", type=Path, required=True)
    run_student.add_argument("--output-root", type=Path, required=True)
    run_student.add_argument("--shard-index", type=int, default=0)
    run_student.add_argument("--num-shards", type=int, default=1)
    run_student.add_argument("--batch-size", type=int, default=8)
    run_student.add_argument("--tensor-parallel-size", type=int, default=1)
    run_student.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    run_student.add_argument("--dtype", choices=("auto", "float16", "bfloat16"), default="bfloat16")
    run_student.add_argument("--grader-utils", type=Path)
    run_student.add_argument("--overwrite-merge", action="store_true")

    run_teacher = commands.add_parser(
        "run-teacher", help="continue all exact 1K/4K/8K/16K prefixes with resume chunks"
    )
    run_teacher.add_argument("--student-rollouts", type=Path, required=True)
    run_teacher.add_argument("--protocol-json", type=Path, required=True)
    run_teacher.add_argument("--output-root", type=Path, required=True)
    run_teacher.add_argument("--shard-index", type=int, default=0)
    run_teacher.add_argument("--num-shards", type=int, default=1)
    run_teacher.add_argument("--batch-size", type=int, default=4)
    run_teacher.add_argument("--tensor-parallel-size", type=int, default=1)
    run_teacher.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    run_teacher.add_argument("--dtype", choices=("auto", "float16", "bfloat16"), default="bfloat16")
    run_teacher.add_argument("--grader-utils", type=Path)
    run_teacher.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    run_teacher.add_argument("--overwrite-merge", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "protocol-template":
            template = make_protocol_template()
            content = json.dumps(template, indent=2, sort_keys=True) + "\n"
            if args.output_json:
                if args.output_json.exists() and not args.overwrite:
                    raise FileExistsError(f"refusing to overwrite {args.output_json}")
                _atomic_text(args.output_json, content)
            else:
                print(content, end="")
            return 0

        if args.command in {"run-student", "run-teacher"}:
            protocol = validate_protocol(read_json_object(args.protocol_json))
            if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
                raise ValueError("shard-index must lie in [0, num-shards)")
            if args.batch_size <= 0 or args.tensor_parallel_size <= 0:
                raise ValueError("batch-size and tensor-parallel-size must be positive")
            if args.command == "run-student":
                report = run_student_rollout_shard(
                    input_parquet=args.input_parquet,
                    protocol=protocol,
                    output_root=args.output_root,
                    shard_index=args.shard_index,
                    num_shards=args.num_shards,
                    batch_size=args.batch_size,
                    tensor_parallel_size=args.tensor_parallel_size,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    dtype=args.dtype,
                    grader_utils=args.grader_utils,
                    overwrite_merge=args.overwrite_merge,
                )
            else:
                if args.bootstrap_replicates <= 0:
                    raise ValueError("bootstrap-replicates must be positive")
                report = run_teacher_continuation_shard(
                    student_rollouts=args.student_rollouts,
                    protocol=protocol,
                    output_root=args.output_root,
                    shard_index=args.shard_index,
                    num_shards=args.num_shards,
                    batch_size=args.batch_size,
                    tensor_parallel_size=args.tensor_parallel_size,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    dtype=args.dtype,
                    grader_utils=args.grader_utils,
                    bootstrap_replicates=args.bootstrap_replicates,
                    overwrite_merge=args.overwrite_merge,
                )
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0

        protocol = validate_protocol(read_json_object(args.protocol_json))
        raw_rollouts = read_jsonl(args.student_rollouts)
        all_rollouts = validate_full_student_batch(raw_rollouts, protocol)
        selected = select_long_rollouts(all_rollouts.values())
        if not selected:
            raise ValueError("no student rollout is strictly longer than 16384 tokens")

        if args.command == "plan":
            plan = build_continuation_plan(selected, protocol["teacher_sampling"])
            report = {
                "protocol_id": PROTOCOL_ID,
                "num_source_prompts": len({row.example_id for row in all_rollouts.values()}),
                "num_source_rollouts": len(all_rollouts),
                "num_selected_rollouts": len(selected),
                "num_continuation_pairs": len(plan),
                "strict_threshold": STRICT_LENGTH_THRESHOLD,
                "prefix_lengths": list(PREFIX_LENGTHS),
                "output_jsonl": str(args.output_jsonl),
                "dry_run": args.dry_run,
            }
            if not args.dry_run:
                if args.output_jsonl.exists() and not args.overwrite:
                    raise FileExistsError(f"refusing to overwrite {args.output_jsonl}")
                _atomic_text(args.output_jsonl, _jsonl_text(plan))
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0

        plan_records = read_jsonl(args.plan_jsonl)
        validate_plan(plan_records, selected, protocol["teacher_sampling"])
        if args.command == "validate":
            print(
                json.dumps(
                    {
                        "status": "valid",
                        "num_selected_rollouts": len(selected),
                        "num_continuation_pairs": len(plan_records),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        paired = validate_continuation_results(
            plan_records,
            read_jsonl(args.results_jsonl),
            protocol["teacher_sampling"],
        )
        if args.bootstrap_replicates <= 0:
            raise ValueError("bootstrap-replicates must be positive")
        summary = summarize_paired_results(
            paired,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=PROBE_SEED,
        )
        summary = {
            **summary,
            "paper_reported_target_gains": {
                "1024": 0.3659,
                "4096": 0.2709,
                "8192": 0.1522,
                "16384": 0.0237,
            },
            "sampling_provenance": protocol["sampling_provenance"],
            "student_sampling": protocol["student_sampling"],
            "teacher_sampling": protocol["teacher_sampling"],
        }
        paths = write_summary_outputs(args.output_dir, paired, summary, overwrite=args.overwrite)
        print(json.dumps({"summary": summary, "outputs": {key: str(value) for key, value in paths.items()}}, indent=2))
        return 0
    except (FileNotFoundError, FileExistsError, json.JSONDecodeError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
