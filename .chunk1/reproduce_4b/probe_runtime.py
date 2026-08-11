#!/usr/bin/env python3
"""Shared, fail-closed runtime helpers for the long-horizon paper probes.

The helpers in this module deliberately keep GPU imports lazy.  Unit tests and
artifact validation therefore remain CPU-only, while the two probe CLIs can
use the same deterministic DAPO selection and crash-safe chunk store on a GPU
node.  A chunk is one complete prompt (all ``n`` rollouts) or one complete
continuation pair; it is atomically published and is never appended in place.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, content: str, *, overwrite: bool = False) -> None:
    """Publish text atomically; by default an existing artifact is immutable."""

    destination = path.expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite immutable artifact {destination}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.tmp-", dir=destination.parent, text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, destination)
        else:
            os.link(temporary, destination)
            temporary.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path: Path, payload: Mapping[str, Any], *, overwrite: bool = False) -> None:
    atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", overwrite=overwrite)


def atomic_jsonl(
    path: Path, records: Iterable[Mapping[str, Any]], *, overwrite: bool = False
) -> None:
    atomic_text(
        path,
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        overwrite=overwrite,
    )


def deterministic_sample_indices(total: int, count: int, seed: int) -> tuple[int, ...]:
    """Return a stable, sorted sample without replacement."""

    if total < 0 or count <= 0 or count > total:
        raise ValueError(f"cannot sample {count} unique rows from total={total}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("sample seed must be a non-negative integer")
    return tuple(sorted(random.Random(seed).sample(range(total), count)))


def assigned_to_shard(ordinal: int, shard_index: int, num_shards: int) -> bool:
    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must lie in [0, num_shards)")
    return ordinal % num_shards == shard_index


def chunk_path(root: Path, kind: str, ordinal: int) -> Path:
    if ordinal < 0:
        raise ValueError("chunk ordinal must be non-negative")
    return root.expanduser() / "chunks" / kind / f"{ordinal:06d}.json"


def write_chunk(
    root: Path,
    kind: str,
    ordinal: int,
    *,
    contract_sha256: str,
    records: Sequence[Mapping[str, Any]],
) -> Path:
    if not records:
        raise ValueError("a chunk must contain at least one record")
    payload = {
        "schema_version": 1,
        "kind": kind,
        "ordinal": ordinal,
        "contract_sha256": contract_sha256,
        "num_records": len(records),
        "records_sha256": canonical_sha256(list(records)),
        "records": list(records),
    }
    path = chunk_path(root, kind, ordinal)
    atomic_json(path, payload)
    return path


def read_chunk(
    root: Path,
    kind: str,
    ordinal: int,
    *,
    contract_sha256: str,
    expected_records: int | None = None,
) -> list[Mapping[str, Any]]:
    path = chunk_path(root, kind, ordinal)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path}: chunk must be an object")
    if payload.get("schema_version") != 1 or payload.get("kind") != kind or payload.get("ordinal") != ordinal:
        raise ValueError(f"{path}: chunk identity mismatch")
    if payload.get("contract_sha256") != contract_sha256:
        raise ValueError(f"{path}: chunk contract mismatch")
    records = payload.get("records")
    if not isinstance(records, list) or any(not isinstance(row, Mapping) for row in records):
        raise ValueError(f"{path}: records must be an object list")
    if payload.get("num_records") != len(records):
        raise ValueError(f"{path}: num_records mismatch")
    if payload.get("records_sha256") != canonical_sha256(records):
        raise ValueError(f"{path}: records hash mismatch")
    if expected_records is not None and len(records) != expected_records:
        raise ValueError(f"{path}: expected {expected_records} records, found {len(records)}")
    return list(records)


def completed_ordinals(
    root: Path,
    kind: str,
    ordinals: Iterable[int],
    *,
    contract_sha256: str,
    expected_records: int | None = None,
) -> set[int]:
    completed: set[int] = set()
    for ordinal in ordinals:
        path = chunk_path(root, kind, ordinal)
        if not path.exists():
            continue
        # Existing but malformed chunks are a protocol failure, not something
        # that may silently be regenerated under the same contract.
        read_chunk(
            root,
            kind,
            ordinal,
            contract_sha256=contract_sha256,
            expected_records=expected_records,
        )
        completed.add(ordinal)
    return completed


def merge_chunks(
    root: Path,
    kind: str,
    total_chunks: int,
    *,
    contract_sha256: str,
    expected_records_per_chunk: int | None,
    output_jsonl: Path,
    overwrite: bool = False,
) -> list[Mapping[str, Any]]:
    """Fail unless every registered chunk exists, then publish canonical JSONL."""

    if total_chunks <= 0:
        raise ValueError("total_chunks must be positive")
    rows: list[Mapping[str, Any]] = []
    missing: list[int] = []
    for ordinal in range(total_chunks):
        if not chunk_path(root, kind, ordinal).is_file():
            missing.append(ordinal)
            continue
        rows.extend(
            read_chunk(
                root,
                kind,
                ordinal,
                contract_sha256=contract_sha256,
                expected_records=expected_records_per_chunk,
            )
        )
    if missing:
        raise ValueError(
            f"cannot merge incomplete {kind} workload: {len(missing)} chunks missing; first={missing[:8]}"
        )
    atomic_jsonl(output_jsonl, rows, overwrite=overwrite)
    return rows


def load_dapo_sample(path: Path, count: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load an exact deterministic subset while preserving source row IDs."""

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - heavyweight runtime
        raise RuntimeError("DAPO parquet loading requires pandas and pyarrow") from exc
    # Reuse the evaluator's normalization rules so no probe-specific prompt
    # suffix is introduced.
    from generate_eval import _example_id, _ground_truth, _to_builtin, normalize_messages

    source = path.expanduser().resolve()
    frame = pd.read_parquet(source)
    if "prompt" not in frame.columns:
        raise ValueError(f"{source} has no prompt column")
    selected = deterministic_sample_indices(len(frame), count, seed)
    records = frame.iloc[list(selected)].to_dict(orient="records")
    samples: list[dict[str, Any]] = []
    for ordinal, (row_index, row) in enumerate(zip(selected, records, strict=True)):
        samples.append(
            {
                "selection_ordinal": ordinal,
                "row_index": row_index,
                "example_id": str(_example_id(row, row_index)),
                "data_source": _to_builtin(row.get("data_source")),
                "prompt": normalize_messages(row["prompt"]),
                "answer": _ground_truth(row),
            }
        )
    selection = {
        "dataset_path": str(source),
        "dataset_sha256": sha256_file(source),
        "dataset_rows": len(frame),
        "sample_size": count,
        "sample_seed": seed,
        "row_indices": list(selected),
    }
    selection["selection_sha256"] = canonical_sha256(selection)
    return samples, selection


def load_grader(path: Path | None = None) -> Any:
    from grade_eval import load_grader_module

    return load_grader_module(path)


def generate_rollout_chunks_vllm(
    samples: Sequence[Mapping[str, Any]],
    *,
    output_root: Path,
    chunk_kind: str,
    contract_sha256: str,
    batch_id: str,
    sampling: Mapping[str, Any],
    shard_index: int,
    num_shards: int,
    batch_size: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    dtype: str,
    grader_utils: Path | None = None,
) -> dict[str, int]:
    """Generate and grade a deterministic prompt shard into atomic chunks.

    Every sample must carry ``selection_ordinal``, ``example_id``, ``prompt``
    and ``answer``.  The sampling object is intentionally the same strict
    metadata later consumed by :func:`validate_rollouts`.
    """

    if batch_size <= 0 or tensor_parallel_size <= 0:
        raise ValueError("batch_size and tensor_parallel_size must be positive")
    if not 0.0 < gpu_memory_utilization <= 1.0:
        raise ValueError("gpu_memory_utilization must lie in (0, 1]")
    required = {
        "model",
        "revision",
        "seed",
        "n",
        "temperature",
        "top_p",
        "max_tokens",
        "thinking",
    }
    if set(sampling) != required:
        raise ValueError(f"sampling must contain exactly {sorted(required)}")
    n = sampling["n"]
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError("sampling.n must be positive")
    ordinals = [int(sample["selection_ordinal"]) for sample in samples]
    if ordinals != list(range(len(samples))):
        raise ValueError("samples must be in contiguous selection_ordinal order")
    completed = completed_ordinals(
        output_root,
        chunk_kind,
        ordinals,
        contract_sha256=contract_sha256,
        expected_records=n,
    )
    pending = [
        sample
        for sample in samples
        if assigned_to_shard(int(sample["selection_ordinal"]), shard_index, num_shards)
        and int(sample["selection_ordinal"]) not in completed
    ]
    if not pending:
        return {"assigned": sum(assigned_to_shard(i, shard_index, num_shards) for i in ordinals), "generated": 0}

    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:  # pragma: no cover - GPU runtime only
        raise RuntimeError("probe generation requires vLLM") from exc
    from generate_eval import render_prompt, resolve_stop_token_ids

    model_source = str(sampling["model"])
    llm_kwargs: dict[str, Any] = {
        "model": model_source,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "dtype": dtype,
        "trust_remote_code": True,
    }
    if not Path(model_source).expanduser().exists() and sampling["revision"] != "local":
        llm_kwargs["revision"] = str(sampling["revision"])
        llm_kwargs["tokenizer_revision"] = str(sampling["revision"])
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    stop_token_ids = resolve_stop_token_ids(tokenizer)
    sampling_kwargs: dict[str, Any] = {
        "n": n,
        "temperature": float(sampling["temperature"]),
        "top_p": float(sampling["top_p"]),
        "max_tokens": int(sampling["max_tokens"]),
        "seed": int(sampling["seed"]),
    }
    if stop_token_ids:
        sampling_kwargs["stop_token_ids"] = stop_token_ids
    params = SamplingParams(**sampling_kwargs)
    grader = load_grader(grader_utils)
    generated = 0
    for start in range(0, len(pending), batch_size):
        sample_batch = pending[start : start + batch_size]
        rendered = [
            render_prompt(
                tokenizer,
                list(sample["prompt"]),
                thinking=str(sampling["thinking"]),
            )
            for sample in sample_batch
        ]
        outputs = llm.generate(rendered, params, use_tqdm=False)
        if len(outputs) != len(sample_batch):
            raise RuntimeError("vLLM returned a different number of prompt outputs")
        for sample, prompt_text, request_output in zip(sample_batch, rendered, outputs, strict=True):
            if len(request_output.outputs) != n:
                raise RuntimeError(f"vLLM did not return exactly n={n} candidates")
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            records: list[dict[str, Any]] = []
            for rollout_id, candidate in enumerate(request_output.outputs):
                response_ids = list(getattr(candidate, "token_ids", ()) or ())
                if not response_ids:
                    response_ids = tokenizer.encode(candidate.text, add_special_tokens=False)
                response_text = str(candidate.text)
                records.append(
                    {
                        "schema_version": 1,
                        "batch_id": batch_id,
                        "selection_ordinal": int(sample["selection_ordinal"]),
                        "row_index": int(sample["row_index"]),
                        "example_id": str(sample["example_id"]),
                        "rollout_id": rollout_id,
                        "prompt_token_ids": [int(value) for value in prompt_ids],
                        "response_token_ids": [int(value) for value in response_ids],
                        "response": response_text,
                        "answer": str(sample["answer"]),
                        "correct": bool(grader.grade_answer_verl(response_text, str(sample["answer"]))),
                        "finish_reason": (
                            None
                            if getattr(candidate, "finish_reason", None) is None
                            else str(candidate.finish_reason)
                        ),
                        "sampling": dict(sampling),
                    }
                )
            write_chunk(
                output_root,
                chunk_kind,
                int(sample["selection_ordinal"]),
                contract_sha256=contract_sha256,
                records=records,
            )
            generated += len(records)
    return {
        "assigned": sum(assigned_to_shard(i, shard_index, num_shards) for i in ordinals),
        "generated": generated,
    }
