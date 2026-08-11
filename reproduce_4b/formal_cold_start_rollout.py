#!/usr/bin/env python3
"""Auditable, resumable Table 3 teacher rollout without modifying author code.

The paper-disclosed decoding contract is fixed here: one response per prompt,
temperature 0.7, top-p 0.95, top-k -1, and at most 12,288 new tokens.  Each GPU
writes an append-only shard and rejection audit.  A final merge succeeds only
when every deterministically selected prompt has exactly one accepted response.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import random
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
MODEL_ID = "Qwen/Qwen3-4B"
MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
TEMPERATURE = 0.7
TOP_P = 0.95
TOP_K = -1
MAX_TOKENS = 12_288
NUM_ROLLOUTS = 1
MAX_MODEL_LEN = 14_336
MAX_PROMPT_TOKENS = MAX_MODEL_LEN - MAX_TOKENS
PAPER_SUFFIX = r"Please reason step by step, and put your final answer within \boxed{}."


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def parse_gpu_ids(raw: str) -> tuple[int, ...]:
    parts = [part.strip() for part in raw.split(",")]
    if not parts or any(not part or not part.isdigit() for part in parts):
        raise argparse.ArgumentTypeError("GPU IDs must be a comma-separated list of non-negative integers")
    values = tuple(int(part) for part in parts)
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("GPU IDs must be unique")
    return values


def sampling_contract(model: str, revision: str, seed: int) -> dict[str, Any]:
    return {
        "model": model,
        "revision": revision,
        "n": NUM_ROLLOUTS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "max_tokens": MAX_TOKENS,
        "max_model_len": MAX_MODEL_LEN,
        # This is the retry schedule's base, not necessarily the seed used by
        # a particular generation attempt.  Every sampled record separately
        # stores its effective ``generation_seed``.
        "base_seed": seed,
        "thinking": False,
    }


def generation_seed(base_seed: int, zero_based_attempt: int) -> int:
    if zero_based_attempt < 0:
        raise ValueError("zero_based_attempt must be non-negative")
    return base_seed + zero_based_attempt


def take_same_attempt_batch(
    pending: list[tuple[int, Any, int]], batch_size: int
) -> tuple[list[tuple[int, Any, int]], list[tuple[int, Any, int]]]:
    """Take a prefix with one retry level so its shared vLLM seed is truthful."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not pending:
        return [], []
    attempt_level = pending[0][2]
    batch_end = 0
    while (
        batch_end < len(pending)
        and batch_end < batch_size
        and pending[batch_end][2] == attempt_level
    ):
        batch_end += 1
    return pending[:batch_end], pending[batch_end:]


def extract_question(raw: Any) -> str:
    if isinstance(raw, list):
        for message in raw:
            if isinstance(message, Mapping) and message.get("role") == "user":
                return str(message.get("content", "")).strip()
        if raw and isinstance(raw[0], Mapping):
            return str(raw[0].get("content", "")).strip()
    if isinstance(raw, Mapping):
        if "messages" in raw:
            return extract_question(raw["messages"])
        for key in ("question", "problem", "content", "prompt"):
            if key in raw:
                return str(raw[key]).strip()
    return str(raw).strip()


def paper_user_message(raw: Any) -> dict[str, str]:
    question = extract_question(raw)
    # Do not duplicate the exact author instruction when inputs were already
    # preprocessed with the teacher-aligned template.
    if not question.endswith(PAPER_SUFFIX):
        question = f"{question} {PAPER_SUFFIX}".strip()
    return {"role": "user", "content": question}


def detect_repeated_lines(text: str, min_len: int = 20, threshold: int = 5) -> bool:
    lines = [line.strip() for line in text.splitlines() if len(line.strip()) >= min_len]
    return bool(lines and Counter(lines).most_common(1)[0][1] >= threshold)


def detect_ngram_repetition(text: str, n: int = 100, threshold: int = 3) -> bool:
    if len(text) < n * threshold:
        return False
    seen: dict[str, int] = {}
    for index in range(0, len(text) - n + 1, 10):
        chunk = text[index : index + n]
        seen[chunk] = seen.get(chunk, 0) + 1
        if seen[chunk] >= threshold:
            return True
    return False


def detect_consecutive_repeat(text: str, block_size: int = 50, threshold: int = 3) -> bool:
    if len(text) < block_size * threshold:
        return False
    for index in range(len(text) - block_size * threshold + 1):
        block = text[index : index + block_size]
        if all(
            text[index + offset * block_size : index + (offset + 1) * block_size] == block
            for offset in range(1, threshold)
        ):
            return True
    return False


def filter_response(text: str, finish_reason: str | None) -> tuple[bool, str]:
    """Apply the paper-described incomplete/degenerate response filters."""

    if finish_reason == "length":
        return False, "truncated"
    if not text.strip():
        return False, "empty"
    if "\\boxed" not in text:
        return False, "no_boxed"
    if detect_repeated_lines(text):
        return False, "repeated_lines"
    if detect_ngram_repetition(text):
        return False, "ngram_repetition"
    if len(text) > 5000 and detect_consecutive_repeat(text):
        return False, "consecutive_repeat"
    return True, "accepted"


def deterministic_selection(total: int, limit: int, seed: int) -> tuple[int, ...]:
    if total < 0 or limit == 0 or limit < -1:
        raise ValueError("total must be non-negative and limit must be -1 or positive")
    if limit == -1 or limit >= total:
        return tuple(range(total))
    return tuple(sorted(random.Random(seed).sample(range(total), limit)))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            records.append(payload)
    return records


def accepted_by_index(path: Path, expected_sampling: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    accepted: dict[int, dict[str, Any]] = {}
    for record in read_jsonl(path):
        index = record.get("global_index")
        if not isinstance(index, int) or isinstance(index, bool):
            raise ValueError(f"invalid global_index in {path}")
        if index in accepted:
            raise ValueError(f"duplicate accepted global_index={index} in {path}")
        if record.get("sampling") != expected_sampling:
            raise ValueError(f"sampling contract mismatch in resumed shard {path}")
        attempt = record.get("attempt")
        actual_generation_seed = record.get("generation_seed")
        base_seed = expected_sampling.get("base_seed")
        if (
            not isinstance(attempt, int)
            or isinstance(attempt, bool)
            or attempt <= 0
            or not isinstance(base_seed, int)
            or isinstance(base_seed, bool)
            or actual_generation_seed != generation_seed(base_seed, attempt - 1)
        ):
            raise ValueError(f"generation_seed mismatch in resumed shard {path}")
        accepted[index] = record
    return accepted


@dataclass(frozen=True)
class WorkerConfig:
    rank: int
    gpu_id: int
    model: str
    revision: str
    seed: int
    max_attempts: int
    batch_size: int
    output_path: str
    audit_path: str


def rollout_worker(config: WorkerConfig, records: list[tuple[int, Any]]) -> None:
    # Set visibility before importing vLLM/Torch.  Each process owns one GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    from vllm import LLM, SamplingParams

    expected_sampling = sampling_contract(config.model, config.revision, config.seed)
    output_path = Path(config.output_path)
    audit_path = Path(config.audit_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    completed = accepted_by_index(output_path, expected_sampling)
    pending = [(index, raw) for index, raw in records if index not in completed]
    if not pending:
        return

    llm = LLM(
        model=config.model,
        revision=config.revision,
        tokenizer_revision=config.revision,
        tensor_parallel_size=1,
        max_model_len=MAX_MODEL_LEN,
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
    )
    tokenizer = llm.get_tokenizer()

    # Each retry gets a deterministic but distinct seed.  The paper did not
    # disclose its seed/retry cap, so both are retained in the audit metadata.
    pending_attempts = [(index, raw, 0) for index, raw in pending]
    with output_path.open("a", encoding="utf-8") as accepted_file, audit_path.open(
        "a", encoding="utf-8"
    ) as audit_file:
        while pending_attempts:
            # Keep a vLLM call homogeneous in retry number: its shared
            # SamplingParams then has one truthful effective seed.  The queue
            # remains grouped because retries are appended after all pending
            # items of the current attempt level.
            attempt_level = pending_attempts[0][2]
            current, pending_attempts = take_same_attempt_batch(
                pending_attempts, config.batch_size
            )
            formatted: list[str] = []
            generatable: list[tuple[int, Any, int, dict[str, str]]] = []
            for index, raw, attempt in current:
                user_message = paper_user_message(raw)
                prompt = tokenizer.apply_chat_template(
                    [user_message],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
                if len(prompt_tokens) > MAX_PROMPT_TOKENS:
                    audit_file.write(
                        json.dumps(
                            {
                                "schema_version": SCHEMA_VERSION,
                                "at": utc_now(),
                                "global_index": index,
                                "attempt": attempt + 1,
                                "accepted": False,
                                "reason": "prompt_too_long",
                                "generation_seed": None,
                                "prompt_tokens": len(prompt_tokens),
                                "sampling": expected_sampling,
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    continue
                formatted.append(prompt)
                generatable.append((index, raw, attempt, user_message))

            if not generatable:
                audit_file.flush()
                continue
            effective_seed = generation_seed(config.seed, attempt_level)
            params = SamplingParams(
                n=NUM_ROLLOUTS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                max_tokens=MAX_TOKENS,
                seed=effective_seed,
            )
            outputs = llm.generate(formatted, params)
            if len(outputs) != len(generatable):
                raise RuntimeError("vLLM returned a different number of requests than submitted")
            for item, request_output in zip(generatable, outputs, strict=True):
                index, raw, attempt, user_message = item
                candidate = request_output.outputs[0]
                text = candidate.text
                finish_reason = getattr(candidate, "finish_reason", None)
                valid, reason = filter_response(text, finish_reason)
                audit_record = {
                    "schema_version": SCHEMA_VERSION,
                    "at": utc_now(),
                    "global_index": index,
                    "attempt": attempt + 1,
                    "accepted": valid,
                    "reason": reason,
                    "generation_seed": effective_seed,
                    "finish_reason": finish_reason,
                    "response_tokens": len(getattr(candidate, "token_ids", ()) or ()),
                    "sampling": expected_sampling,
                }
                audit_file.write(json.dumps(audit_record, ensure_ascii=False, sort_keys=True) + "\n")
                if valid:
                    accepted_record = {
                        "schema_version": SCHEMA_VERSION,
                        "global_index": index,
                        "rollout_index": 0,
                        "messages": [user_message, {"role": "assistant", "content": text}],
                        "sampling": expected_sampling,
                        "generation_seed": effective_seed,
                        "attempt": attempt + 1,
                        "finish_reason": finish_reason,
                    }
                    accepted_file.write(
                        json.dumps(accepted_record, ensure_ascii=False, sort_keys=True) + "\n"
                    )
                    accepted_file.flush()
                elif attempt + 1 < config.max_attempts and reason != "prompt_too_long":
                    pending_attempts.append((index, raw, attempt + 1))
            audit_file.flush()


def load_prompt_column(path: Path) -> list[Any]:
    import pandas as pd

    frame = pd.read_parquet(path)
    for column in ("prompt", "messages", "question", "problem"):
        if column in frame.columns:
            return frame[column].tolist()
    raise ValueError(f"input parquet has none of prompt/messages/question/problem columns: {path}")


def audit_summary(
    output_dir: Path,
    selected: Sequence[int],
    gpu_count: int,
    expected_sampling: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    accepted: dict[int, dict[str, Any]] = {}
    reasons: Counter[str] = Counter()
    attempts = 0
    for rank in range(gpu_count):
        shard = output_dir / "shards" / f"part-{rank:05d}.jsonl"
        for index, record in accepted_by_index(shard, expected_sampling).items():
            if index in accepted:
                raise ValueError(f"global_index={index} appears in multiple shards")
            accepted[index] = record
        for audit in read_jsonl(output_dir / "audit" / f"part-{rank:05d}.jsonl"):
            reasons[str(audit.get("reason", "unknown"))] += 1
            attempts += 1

    selected_set = set(selected)
    unexpected = set(accepted) - selected_set
    if unexpected:
        raise ValueError(f"shards contain indices outside deterministic selection: {sorted(unexpected)[:5]}")
    missing = sorted(selected_set - set(accepted))
    merged = [accepted[index] for index in sorted(accepted)]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "selected_prompts": len(selected),
        "accepted_prompts": len(accepted),
        "missing_prompts": len(missing),
        "missing_indices_preview": missing[:100],
        "completion_rate": (len(accepted) / len(selected)) if selected else 1.0,
        "generation_attempts": attempts,
        "filter_reason_counts": dict(sorted(reasons.items())),
        "sampling": dict(expected_sampling),
        "acceptance_contract": "exactly one accepted response per selected prompt",
        "filter_contract": "reject truncation, missing boxed answer, empty or degenerate repetition",
    }
    return summary, merged


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--gpu-ids", type=parse_gpu_ids, default=(0, 1))
    parser.add_argument("--limit", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.limit == 0 or args.limit < -1:
            raise ValueError("--limit must be -1 or a positive integer")
        if args.batch_size <= 0 or args.max_attempts <= 0:
            raise ValueError("--batch-size and --max-attempts must be positive")
        if not args.revision or args.revision == "main":
            raise ValueError("--revision must be an immutable model revision, not main")
        sampling = sampling_contract(args.model, args.revision, args.seed)
        plan = {
            "stage": "cold-start-rollout",
            "input_parquet": str(args.input_parquet.expanduser().absolute()),
            "output_dir": str(args.output_dir.expanduser().absolute()),
            "limit": args.limit,
            "gpu_ids": list(args.gpu_ids),
            "sampling": sampling,
            "paper_template": f"{{Question}} {PAPER_SUFFIX}",
            "responses_per_prompt": NUM_ROLLOUTS,
            "max_attempts": args.max_attempts,
            "fidelity_note": "paper decoding parameters; seed/retry cap and 2-GPU sharding are local disclosed assumptions",
        }
        if args.dry_run:
            print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
            return 0

        input_path = args.input_parquet.expanduser().resolve()
        if not input_path.is_file():
            raise FileNotFoundError(f"rollout input parquet is missing: {input_path}")
        output_dir = args.output_dir.expanduser().resolve()
        prompts = load_prompt_column(input_path)
        selected = deterministic_selection(len(prompts), args.limit, args.seed)
        selected_hash = canonical_hash(selected)
        identity = {
            **plan,
            "input_parquet": str(input_path),
            "input_sha256": sha256_file(input_path),
            "input_rows": len(prompts),
            "selected_indices_sha256": selected_hash,
        }
        identity["fingerprint"] = canonical_hash(identity)
        manifest_path = output_dir / "rollout_manifest.json"
        if manifest_path.is_file():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("fingerprint") != identity["fingerprint"]:
                raise RuntimeError("existing rollout directory has a different immutable fingerprint")
        else:
            atomic_json(manifest_path, identity)

        selected_records = [(index, prompts[index]) for index in selected]
        partitions = [selected_records[rank :: len(args.gpu_ids)] for rank in range(len(args.gpu_ids))]
        context = multiprocessing.get_context("spawn")
        processes = []
        for rank, (gpu_id, records) in enumerate(zip(args.gpu_ids, partitions, strict=True)):
            config = WorkerConfig(
                rank=rank,
                gpu_id=gpu_id,
                model=args.model,
                revision=args.revision,
                seed=args.seed,
                max_attempts=args.max_attempts,
                batch_size=args.batch_size,
                output_path=str(output_dir / "shards" / f"part-{rank:05d}.jsonl"),
                audit_path=str(output_dir / "audit" / f"part-{rank:05d}.jsonl"),
            )
            process = context.Process(target=rollout_worker, args=(config, records))
            process.start()
            processes.append(process)
        for process in processes:
            process.join()
        failed = [index for index, process in enumerate(processes) if process.exitcode != 0]
        if failed:
            raise RuntimeError(f"rollout worker(s) failed: {failed}; rerun the same directory to resume shards")

        summary, merged = audit_summary(output_dir, selected, len(args.gpu_ids), sampling)
        atomic_json(output_dir / "filter_audit.json", summary)
        atomic_jsonl(output_dir / "cold_start_sft.jsonl", merged)
        if summary["missing_prompts"]:
            raise RuntimeError(
                f"rollout incomplete: {summary['missing_prompts']} selected prompts have no accepted response; "
                "inspect filter_audit.json and shard audits"
            )
        (output_dir / "_SUCCESS").touch()
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (FileNotFoundError, json.JSONDecodeError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    raise SystemExit(main())
