#!/usr/bin/env python3
"""Evaluate an immutable HF/local model with the paper's strict avg@16 protocol.

Unlike ``evaluate_ablation.py``, this entry point is for step-0 students and
teacher baselines that do not live below a VERL checkpoint directory.  It
shares the same generation validator and three-benchmark macro aggregation so
Figure 3 and all baseline/gap-recovery calculations use one auditable format.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluate_ablation import BENCHMARKS, aggregate_metrics, parquet_rows, validate_generation
from run_ablations import acquire_gpu_lock, atomic_json, sha256_file, utc_now


TARGET_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def target_root(output_root: Path, target_id: str) -> Path:
    if not TARGET_RE.fullmatch(target_id):
        raise ValueError("target-id may contain only letters, digits, '.', '_' and '-'")
    return output_root.expanduser().resolve() / target_id


def sampling_contract(model: str, revision: str | None, tokenizer: str, tokenizer_revision: str | None,
                      n: int, max_tokens: int, seed: int) -> dict[str, Any]:
    return {
        "n": n,
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": max_tokens,
        "seed": seed,
        "thinking": "off",
        "model": model,
        "revision": revision,
        "tokenizer": tokenizer,
        "tokenizer_revision": tokenizer_revision,
    }


def build_commands(
    repo_root: Path,
    root: Path,
    model: str,
    revision: str | None,
    tokenizer: str,
    tokenizer_revision: str | None,
    n: int,
    max_tokens: int,
    seed: int,
    limit: int | None,
    overwrite_incomplete: bool,
) -> list[tuple[str, list[str], Path | None]]:
    python = repo_root / ".venv-opd/bin/python"
    commands: list[tuple[str, list[str], Path | None]] = []
    expected_sampling = sampling_contract(
        model, revision, tokenizer, tokenizer_revision, n, max_tokens, seed
    )
    for benchmark in BENCHMARKS:
        benchmark_dir = root / benchmark
        input_parquet = repo_root / "datasets/test_data" / benchmark / "test.parquet"
        output_jsonl = benchmark_dir / "responses.jsonl"
        metrics_json = benchmark_dir / "metrics.json"
        expected_prompts = parquet_rows(input_parquet, limit)
        complete = False
        if output_jsonl.is_file():
            try:
                validate_generation(output_jsonl, expected_prompts, n, expected_sampling)
                complete = True
            except (json.JSONDecodeError, OSError, ValueError):
                if not overwrite_incomplete:
                    raise
        if not complete:
            command = [
                str(python),
                str(repo_root / "reproduce_4b/generate_eval.py"),
                "--model", model,
                "--tokenizer", tokenizer,
                "--input-parquet", str(input_parquet),
                "--output-jsonl", str(output_jsonl),
                "--cuda-visible-devices", os.environ.get("CUDA_VISIBLE_DEVICES", "0,1"),
                "--tensor-parallel-size", "2",
                "--n", str(n),
                "--temperature", "0.7",
                "--top-p", "0.95",
                "--max-tokens", str(max_tokens),
                "--max-model-len", str(max_tokens + 1024),
                "--thinking", "off",
                "--seed", str(seed),
            ]
            if revision:
                command.extend(["--revision", revision])
            if tokenizer_revision:
                command.extend(["--tokenizer-revision", tokenizer_revision])
            if limit:
                command.extend(["--limit", str(limit)])
            if output_jsonl.exists() and overwrite_incomplete:
                command.append("--overwrite")
            commands.append((f"generate-{benchmark}", command, output_jsonl))
        if not metrics_json.is_file() or not complete:
            commands.append(
                (
                    f"grade-{benchmark}",
                    [
                        str(python),
                        str(repo_root / "reproduce_4b/grade_eval.py"),
                        "--input-jsonl", str(output_jsonl),
                        "--output-json", str(metrics_json),
                        "--n", str(n),
                        "--strict-n",
                    ],
                    metrics_json,
                )
            )
    return commands


def build_manifest(
    repo_root: Path,
    target_id: str,
    model: str,
    revision: str | None,
    tokenizer: str,
    tokenizer_revision: str | None,
    n: int,
    max_tokens: int,
    seed: int,
    limit: int | None,
) -> dict[str, Any]:
    datasets = {}
    for benchmark in BENCHMARKS:
        path = repo_root / "datasets/test_data" / benchmark / "test.parquet"
        datasets[benchmark] = {
            "path": str(path),
            "rows": parquet_rows(path, None),
            "sha256": sha256_file(path),
        }
    return {
        "schema_version": 1,
        "target_id": target_id,
        "kind": "immutable_model_baseline",
        "model": model,
        "revision": revision,
        "tokenizer": tokenizer,
        "tokenizer_revision": tokenizer_revision,
        "sampling": sampling_contract(
            model, revision, tokenizer, tokenizer_revision, n, max_tokens, seed
        ),
        "limit": limit,
        "paper_comparable": limit is None and n == 16 and max_tokens == 31_744,
        "benchmarks": datasets,
    }


def validate_existing_manifest(path: Path, expected: Mapping[str, Any]) -> None:
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as handle:
        observed = json.load(handle)
    if observed != expected:
        raise ValueError(f"existing target manifest differs; choose a new target-id: {path}")


def build_parser(repo_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run", "status"))
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--tokenizer")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--output-root", type=Path, default=repo_root / "artifacts/evaluation/paper-models")
    parser.add_argument("--n", type=positive_int, default=16)
    parser.add_argument("--max-tokens", type=positive_int, default=31_744)
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
        tokenizer = args.tokenizer or args.model
        tokenizer_revision = args.tokenizer_revision
        if tokenizer_revision is None and tokenizer == args.model:
            tokenizer_revision = args.revision
        root = target_root(args.output_root, args.target_id)
        manifest = build_manifest(
            repo_root, args.target_id, args.model, args.revision, tokenizer,
            tokenizer_revision, args.n, args.max_tokens, args.seed, args.limit
        )
        manifest_path = root / "target_manifest.json"
        validate_existing_manifest(manifest_path, manifest)
        commands = build_commands(
            repo_root, root, args.model, args.revision, tokenizer, tokenizer_revision,
            args.n, args.max_tokens, args.seed, args.limit, args.overwrite_incomplete
        )
        summary_path = root / "summary.json"
        if args.action in {"plan", "status"}:
            state = "completed" if summary_path.is_file() and not commands else "pending"
            print(f"target={args.target_id} state={state} pending_commands={len(commands)}")
            for label, command, _ in commands:
                print(label + ": " + " ".join(command))
            return 0
        if not args.yes:
            raise ValueError("action=run requires --yes")
        if args.limit is None and not args.acknowledge_full_eval:
            raise ValueError("full avg@N evaluation requires --acknowledge-full-eval")

        lock = acquire_gpu_lock(os.environ.get("CUDA_VISIBLE_DEVICES", "0,1"))
        root.mkdir(parents=True, exist_ok=True)
        atomic_json(manifest_path, manifest)
        status_path = root / "status.json"
        status: dict[str, Any] = {
            "schema_version": 1,
            "target_id": args.target_id,
            "state": "running",
            "started_at": utc_now(),
            "commands": [],
        }
        atomic_json(status_path, status)
        try:
            environment = os.environ.copy()
            environment["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
            for label, command, output in commands:
                if output is not None:
                    output.parent.mkdir(parents=True, exist_ok=True)
                result = subprocess.run(command, cwd=repo_root, env=environment, check=False)
                status["commands"].append({"label": label, "exit_code": result.returncode})
                atomic_json(status_path, status)
                if result.returncode:
                    raise RuntimeError(f"evaluation command failed ({label}) with exit {result.returncode}")
            metrics = [root / benchmark / "metrics.json" for benchmark in BENCHMARKS]
            summary = aggregate_metrics(metrics, args.n)
            summary.update(
                {
                    "schema_version": 1,
                    "target_id": args.target_id,
                    "model": args.model,
                    "revision": args.revision,
                    "paper_comparable": manifest["paper_comparable"],
                    "created_at": utc_now(),
                }
            )
            atomic_json(summary_path, summary)
            status.update({"state": "completed", "ended_at": utc_now(), "summary": str(summary_path)})
            atomic_json(status_path, status)
        except Exception:
            status.update({"state": "failed", "ended_at": utc_now()})
            atomic_json(status_path, status)
            raise
        finally:
            lock.close()
        return 0
    except (FileNotFoundError, json.JSONDecodeError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
