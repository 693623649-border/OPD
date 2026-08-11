#!/usr/bin/env python3
"""Plan, run, and inspect the formal Table 1/Table 3 upstream pipeline.

Stages are intentionally separate because GRPO teacher training and cold-start
data/SFT are independent upstream branches.  Retries always use a fresh attempt
directory; status and provenance files are written atomically.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
SUITE_ID = "rethinking-opd-upstream-v3"
GRPO_RELEASED_DATA_SHA256 = "500bd8c45eca355b98f9ba6f3213194a72bd42c73c5e9569c6fbbb1b51bd0b39"
STAGES = ("grpo-teacher", "cold-start-rollout", "cold-start-sft")
STAGE_PROTOCOLS = {
    "grpo-teacher": frozenset(("smoke", "calibration", "paper")),
    "cold-start-rollout": frozenset(("smoke", "calibration", "paper")),
    "cold-start-sft": frozenset(("smoke", "paper")),
}
MODEL_REVISIONS = {
    "grpo-teacher": (
        {
            "role": "base_model",
            "model_id": "Qwen/Qwen3-4B-Base",
            "revision": "906bfd4b4dc7f14ee4320094d8b41684abff8539",
        },
    ),
    "cold-start-rollout": (
        {
            "role": "teacher",
            "model_id": "Qwen/Qwen3-4B",
            "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
        },
    ),
    "cold-start-sft": (
        {
            "role": "student_initialization",
            "model_id": "Qwen/Qwen3-1.7B-Base",
            "revision": "ea980cb0a6c2ae4b936e82123acc929f1cec04c1",
        },
    ),
}
SOURCE_PATHS = {
    "grpo-teacher": (
        "reproduce_4b/run_upstream.py",
        "reproduce_4b/run_grpo_teacher_4b.sh",
        "grpo.sh",
        "verl/verl/trainer/main_ppo.py",
        "verl/verl/trainer/config/ppo_trainer.yaml",
        "verl/verl/trainer/config/rollout/rollout.yaml",
        "verl/verl/trainer/ppo/ray_trainer.py",
        "verl/verl/trainer/ppo/core_algos.py",
        "verl/verl/workers/fsdp_workers.py",
        "verl/verl/workers/actor/dp_actor.py",
        "verl/verl/utils/tracking.py",
        "verl/verl/utils/reward_score/ttrl_math/__init__.py",
    ),
    "cold-start-rollout": (
        "reproduce_4b/run_upstream.py",
        "reproduce_4b/formal_cold_start_rollout.py",
        "scripts/infer/vllm_rollout.py",
    ),
    "cold-start-sft": (
        "reproduce_4b/run_upstream.py",
        "reproduce_4b/run_cold_start_sft.sh",
        "LlamaFactory/src/llamafactory/extras/env.py",
        "LlamaFactory/src/llamafactory/cli.py",
        "LlamaFactory/src/llamafactory/launcher.py",
        "LlamaFactory/examples/train_full/qwen3_base_full_sft.yaml",
        "LlamaFactory/examples/deepspeed/ds_z2_config.json",
    ),
}
SAFE_ENV_KEYS = {
    "HOME",
    "USER",
    "LOGNAME",
    "PATH",
    "PYTHONPATH",
    "LD_LIBRARY_PATH",
    "LIBRARY_PATH",
    "CPATH",
    "LANG",
    "LC_ALL",
    "TMPDIR",
    "HF_HOME",
    "HF_HUB_CACHE",
    "TRANSFORMERS_CACHE",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "CUDA_VISIBLE_DEVICES",
}


@dataclass(frozen=True)
class StagePlan:
    stage: str
    protocol: str
    fidelity: str
    launcher: tuple[str, ...]
    environment: dict[str, str]
    inputs: tuple[Path, ...]
    success_relative_path: str
    sources: tuple[dict[str, Any], ...]
    datasets: tuple[dict[str, Any], ...]
    models: tuple[dict[str, str], ...]
    fingerprint: str


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


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def attempt_number(stage_root: Path) -> int:
    numbers = []
    for path in stage_root.glob("attempt-*"):
        suffix = path.name.removeprefix("attempt-")
        if suffix.isdigit():
            numbers.append(int(suffix))
    return max(numbers, default=0) + 1


def absolute_path_preserve_symlinks(path: Path) -> Path:
    return Path(os.path.abspath(path.expanduser()))


def source_record(repo_root: Path, relative: str) -> dict[str, Any]:
    path = repo_root / relative
    if not path.is_file():
        raise FileNotFoundError(f"required upstream source is missing: {path}")
    return {"path": relative, "sha256": sha256_file(path), "bytes": path.stat().st_size}


def dataset_record(path: Path) -> dict[str, Any]:
    absolute = Path(os.path.abspath(path.expanduser()))
    record: dict[str, Any] = {"path": str(absolute), "exists": absolute.is_file()}
    if not absolute.is_file():
        return record
    record.update({"sha256": sha256_file(absolute), "bytes": absolute.stat().st_size})
    if absolute.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq

            record["rows"] = pq.ParquetFile(absolute).metadata.num_rows
            record["format"] = "parquet"
        except ImportError:
            record["rows"] = None
            record["format"] = "parquet-uninspected"
    elif absolute.suffix in {".jsonl", ".json"}:
        with absolute.open("rb") as handle:
            record["rows"] = sum(1 for line in handle if line.strip())
        record["format"] = "jsonl"
    else:
        record["rows"] = None
        record["format"] = absolute.suffix.removeprefix(".") or "unknown"
    return record


def git_record(repo_root: Path) -> dict[str, Any]:
    def capture(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else "unavailable"

    status = capture("status", "--short")
    return {
        "commit": capture("rev-parse", "HEAD"),
        "dirty": bool(status and status != "unavailable"),
        "status_sha256": canonical_hash(status),
    }


def completed_rollout_dataset(suite_root: Path) -> Path | None:
    status = read_json(suite_root / "cold-start-rollout" / "status.json")
    if not status or status.get("state") != "completed":
        return None
    run_dir = status.get("run_dir")
    if not isinstance(run_dir, str):
        return None
    candidate = Path(run_dir) / "rollout" / "cold_start_sft.jsonl"
    return candidate if candidate.is_file() else None


def fidelity_label(stage: str, protocol: str) -> str:
    if protocol != "paper":
        return f"engineering-{protocol}; 2xA100 adaptation; not a paper result"
    if stage == "grpo-teacher":
        return "Table 1 disclosed parameters; 2xA100 hardware adaptation vs paper 8xA800-80G"
    if stage == "cold-start-rollout":
        return "paper decoding parameters; local seed/retry/sharding assumptions on 2xA100"
    return "Table 3 disclosed parameters; 2xA100 hardware adaptation, original GPU count undisclosed"


def build_stage_plan(
    *,
    repo_root: Path,
    suite_root: Path,
    stage: str,
    protocol: str,
    seed: int,
    grpo_data: Path,
    rollout_input: Path,
    cold_start_data: Path | None,
    python_bin: Path,
    sft_python_bin: Path,
) -> StagePlan:
    if stage not in STAGES:
        raise ValueError(f"unknown substage {stage!r}")
    if protocol not in STAGE_PROTOCOLS[stage]:
        raise ValueError(f"{stage} supports protocols {sorted(STAGE_PROTOCOLS[stage])}, not {protocol}")

    if stage == "grpo-teacher":
        inputs = (grpo_data,)
        launcher = ("bash", str(repo_root / "reproduce_4b/run_grpo_teacher_4b.sh"))
        environment = {
            "PRESET": protocol,
            "TRAIN_DATA": str(Path(os.path.abspath(grpo_data.expanduser()))),
            "EXPECTED_DATA_ROWS": "17917",
            "EXPECTED_DATA_SHA256": GRPO_RELEASED_DATA_SHA256,
            "DATA_FIDELITY": "author-released-processed; literal prompt suffix uses \\boxed{{}}",
            "PYTHON_BIN": str(absolute_path_preserve_symlinks(python_bin)),
            "N_GPUS": "2",
            "SEED": str(seed),
        }
        success = "grpo/_SUCCESS"
    elif stage == "cold-start-rollout":
        inputs = (rollout_input,)
        limits = {"smoke": "4", "calibration": "128", "paper": "200000"}
        batches = {"smoke": "4", "calibration": "16", "paper": "64"}
        launcher = (
            str(absolute_path_preserve_symlinks(python_bin)),
            str(repo_root / "reproduce_4b/formal_cold_start_rollout.py"),
        )
        environment = {
            "ROLLOUT_INPUT": str(Path(os.path.abspath(rollout_input.expanduser()))),
            "ROLLOUT_LIMIT": limits[protocol],
            "ROLLOUT_BATCH_SIZE": batches[protocol],
            "SEED": str(seed),
            "N_GPUS": "2",
        }
        success = "rollout/_SUCCESS"
    else:
        dependency = cold_start_data or completed_rollout_dataset(suite_root)
        if dependency is None:
            dependency = repo_root / "datasets/OpenThought3-Qwen3-4B.jsonl"
        inputs = (dependency,)
        launcher = ("bash", str(repo_root / "reproduce_4b/run_cold_start_sft.sh"))
        environment = {
            "PRESET": protocol,
            "DATASET_JSONL": str(Path(os.path.abspath(dependency.expanduser()))),
            "SFT_PYTHON_BIN": str(absolute_path_preserve_symlinks(sft_python_bin)),
            "LLAMAFACTORY_ROOT": str(repo_root / "LlamaFactory"),
            "N_GPUS": "2",
            "SEED": str(seed),
        }
        success = "sft/_SUCCESS"

    sources = tuple(source_record(repo_root, relative) for relative in SOURCE_PATHS[stage])
    datasets = tuple(dataset_record(path) for path in inputs)
    models = tuple(dict(record) for record in MODEL_REVISIONS[stage])
    identity = {
        "schema_version": SCHEMA_VERSION,
        "suite_id": SUITE_ID,
        "stage": stage,
        "protocol": protocol,
        "seed": seed,
        "fidelity": fidelity_label(stage, protocol),
        "launcher": launcher,
        "environment": environment,
        "sources": sources,
        "datasets": datasets,
        "models": models,
    }
    return StagePlan(
        stage=stage,
        protocol=protocol,
        fidelity=identity["fidelity"],
        launcher=launcher,
        environment=environment,
        inputs=inputs,
        success_relative_path=success,
        sources=sources,
        datasets=datasets,
        models=models,
        fingerprint=canonical_hash(identity),
    )


def attempt_manifest(
    plan: StagePlan,
    *,
    repo_root: Path,
    run_dir: Path,
    attempt: int,
) -> dict[str, Any]:
    """Build the immutable per-attempt source/data/model provenance record."""

    return {
        "schema_version": SCHEMA_VERSION,
        "suite_id": SUITE_ID,
        "created_at": utc_now(),
        "stage": plan.stage,
        "protocol": plan.protocol,
        "attempt": attempt,
        "run_dir": str(run_dir),
        "fingerprint": plan.fingerprint,
        "fidelity": plan.fidelity,
        "paper_scope": "Table 1" if plan.stage == "grpo-teacher" else "Table 3",
        "source": {
            "repository": git_record(repo_root),
            "files": list(plan.sources),
        },
        "data": list(plan.datasets),
        "models": list(plan.models),
        "environment": dict(plan.environment),
        "launcher": list(plan.launcher),
        "resume_policy": "fresh attempt; no training checkpoint resume",
    }


def selected_stages(requested: Sequence[str], protocol: str) -> tuple[str, ...]:
    if protocol not in {"smoke", "calibration", "paper"}:
        raise ValueError("protocol must be smoke, calibration, or paper")
    if requested:
        unknown = set(requested) - set(STAGES)
        if unknown:
            raise ValueError(f"unknown substage(s): {sorted(unknown)}")
        ordered = tuple(stage for stage in STAGES if stage in set(requested))
        unsupported = [stage for stage in ordered if protocol not in STAGE_PROTOCOLS[stage]]
        if unsupported:
            raise ValueError(f"protocol {protocol} is unsupported for {unsupported}")
        return ordered
    return tuple(stage for stage in STAGES if protocol in STAGE_PROTOCOLS[stage])


def clean_environment(plan: StagePlan, run_dir: Path) -> dict[str, str]:
    environment = {key: value for key, value in os.environ.items() if key in SAFE_ENV_KEYS}
    environment.update(plan.environment)
    environment.update({"RUN_DIR": str(run_dir), "DRY_RUN": "0"})
    environment.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
    if plan.stage == "cold-start-rollout":
        environment["ROLLOUT_OUTPUT"] = str(run_dir / "rollout")
    return environment


def resolved_command(plan: StagePlan, environment: Mapping[str, str]) -> tuple[str, ...]:
    if plan.stage != "cold-start-rollout":
        return plan.launcher
    return (
        *plan.launcher,
        "--input-parquet",
        environment["ROLLOUT_INPUT"],
        "--output-dir",
        environment["ROLLOUT_OUTPUT"],
        "--model",
        MODEL_REVISIONS["cold-start-rollout"][0]["model_id"],
        "--revision",
        MODEL_REVISIONS["cold-start-rollout"][0]["revision"],
        "--gpu-ids",
        environment["CUDA_VISIBLE_DEVICES"],
        "--limit",
        environment["ROLLOUT_LIMIT"],
        "--batch-size",
        environment["ROLLOUT_BATCH_SIZE"],
        "--seed",
        environment["SEED"],
    )


def display_command(plan: StagePlan, run_dir: Path) -> str:
    environment = clean_environment(plan, run_dir)
    scientific = {
        key: value
        for key, value in environment.items()
        if key not in SAFE_ENV_KEYS and key not in {"DRY_RUN"}
    }
    command = resolved_command(plan, environment)
    return " ".join(
        [
            *(f"{key}={shlex.quote(value)}" for key, value in sorted(scientific.items())),
            *(shlex.quote(part) for part in command),
        ]
    )


def validate_run_inputs(plan: StagePlan) -> None:
    for record in plan.datasets:
        if not record.get("exists"):
            raise FileNotFoundError(f"{plan.stage} input dataset is missing: {record['path']}")
        if not isinstance(record.get("sha256"), str):
            raise ValueError(f"{plan.stage} input dataset could not be hashed: {record['path']}")
    if plan.stage == "grpo-teacher":
        rows = plan.datasets[0].get("rows")
        if rows != 17_917:
            raise ValueError(f"formal Table 1 DAPO input must contain 17917 rows, got {rows}")
        digest = plan.datasets[0].get("sha256")
        if digest != GRPO_RELEASED_DATA_SHA256:
            raise ValueError(
                "formal Table 1 requires the author-released processed DAPO file; "
                f"sha256={digest}, expected {GRPO_RELEASED_DATA_SHA256}"
            )
    if plan.stage == "cold-start-rollout" and plan.protocol == "paper":
        rows = plan.datasets[0].get("rows")
        if not isinstance(rows, int) or rows < 200_000:
            raise ValueError(f"paper cold-start rollout requires at least 200000 source prompts, got {rows}")
    if plan.stage == "cold-start-sft" and plan.datasets[0].get("rows") in (None, 0):
        raise ValueError("cold-start SFT dataset must contain at least one JSONL record")


class GPULocks:
    """Collection of per-device locks compatible with the ablation runner."""

    def __init__(self, handles: Sequence[Any]):
        self.handles = tuple(handles)

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.close()


def acquire_gpu_lock(
    cuda_devices: str, *, lock_dir: Path = Path("/tmp")
) -> GPULocks:
    parts = tuple(part.strip() for part in cuda_devices.split(","))
    if len(parts) != 2 or any(not part.isdigit() for part in parts):
        raise ValueError("formal upstream runs require exactly two unique numeric CUDA_VISIBLE_DEVICES")
    normalized = tuple(str(int(part)) for part in parts)
    if len(set(normalized)) != 2:
        raise ValueError("formal upstream runs require exactly two unique numeric CUDA_VISIBLE_DEVICES")
    handles: list[Any] = []
    lock_dir.mkdir(parents=True, exist_ok=True)
    try:
        for device in sorted(normalized, key=int):
            # This filename is deliberately identical to run_ablations.py so
            # a 2-GPU upstream job conflicts with every overlapping 2/8-GPU
            # OPD allocation, not merely with an identical selector string.
            path = lock_dir / f"opd-ablation-gpu-{device}.lock"
            handle = path.open("a+", encoding="utf-8")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                handle.close()
                raise RuntimeError(
                    f"physical GPU {device} from set {cuda_devices} is already locked"
                ) from exc
            handles.append(handle)
            handle.seek(0)
            handle.truncate()
            handle.write(
                json.dumps(
                    {
                        "pid": os.getpid(),
                        "physical_gpu": int(device),
                        "cuda_visible_devices": cuda_devices,
                        "at": utc_now(),
                    }
                )
            )
            handle.flush()
    except Exception:
        GPULocks(handles).close()
        raise
    return GPULocks(handles)


def status_line(stage: str, status: Mapping[str, Any] | None, fingerprint: str) -> str:
    if status is None:
        return f"{stage}\tpending\t{fingerprint[:16]}"
    detail = f" attempt={status.get('attempt')}"
    if status.get("exit_code") is not None:
        detail += f" exit={status.get('exit_code')}"
    return f"{stage}\t{status.get('state', 'unknown')}{detail}\t{fingerprint[:16]}"


def run_stage(plan: StagePlan, repo_root: Path, suite_root: Path, retry_failed: bool) -> int:
    validate_run_inputs(plan)
    stage_root = suite_root / plan.stage
    status_path = stage_root / "status.json"
    previous = read_json(status_path)
    if previous and previous.get("fingerprint") != plan.fingerprint:
        raise RuntimeError(
            f"{plan.stage}: existing status fingerprint differs; use a new --run-root/protocol/seed"
        )
    if previous and previous.get("state") == "completed":
        print(f"skip completed {plan.stage}")
        return 0
    if previous and previous.get("state") == "failed" and not retry_failed:
        raise RuntimeError(f"{plan.stage}: previous attempt failed; pass --retry-failed for a fresh attempt")

    attempt = attempt_number(stage_root)
    run_dir = stage_root / f"attempt-{attempt:04d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = run_dir / "upstream_manifest.json"
    atomic_json(
        manifest_path,
        attempt_manifest(plan, repo_root=repo_root, run_dir=run_dir, attempt=attempt),
    )
    status = {
        "schema_version": SCHEMA_VERSION,
        "stage": plan.stage,
        "protocol": plan.protocol,
        "fingerprint": plan.fingerprint,
        "state": "running",
        "attempt": attempt,
        "run_dir": str(run_dir),
        "manifest": str(manifest_path),
        "started_at": utc_now(),
        "resumed": False,
    }
    atomic_json(status_path, status)
    environment = clean_environment(plan, run_dir)
    command = resolved_command(plan, environment)
    print(f"start {plan.stage}, attempt {attempt:04d}")
    started = time.monotonic()
    try:
        result = subprocess.run(command, cwd=repo_root, env=environment, check=False)
        exit_code = result.returncode
    except KeyboardInterrupt:
        exit_code = 130
    elapsed = time.monotonic() - started
    success_path = run_dir / plan.success_relative_path
    state = "completed" if exit_code == 0 and success_path.is_file() else "failed"
    status.update(
        {
            "state": state,
            "exit_code": exit_code,
            "ended_at": utc_now(),
            "elapsed_seconds": elapsed,
            "success_marker": str(success_path),
        }
    )
    atomic_json(status_path, status)
    print(f"{state} {plan.stage}: exit={exit_code}, marker={success_path.is_file()}")
    return 0 if state == "completed" else 1


def build_parser(repo_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run", "status"))
    parser.add_argument("--protocol", choices=("smoke", "calibration", "paper"), default="smoke")
    parser.add_argument("--substage", action="append", choices=STAGES, default=[])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-root", type=Path, default=repo_root / "artifacts/upstream")
    parser.add_argument(
        "--grpo-data",
        type=Path,
        default=repo_root / "datasets/dapo-math-17k-processed.parquet",
    )
    parser.add_argument(
        "--rollout-input",
        type=Path,
        default=repo_root / "datasets/OpenThoughts3-1.2M-math.parquet",
    )
    parser.add_argument("--cold-start-data", type=Path)
    parser.add_argument("--python-bin", type=Path, default=repo_root / ".venv-opd/bin/python")
    parser.add_argument("--sft-python-bin", type=Path, default=repo_root / ".venv-sft/bin/python")
    parser.add_argument("--yes", action="store_true", help="Required for action=run")
    parser.add_argument("--acknowledge-multi-day", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = build_parser(repo_root)
    args = parser.parse_args(argv)
    try:
        if args.action == "run" and not args.yes:
            raise ValueError("action=run requires --yes")
        if args.action == "run" and args.protocol == "paper" and not args.acknowledge_multi_day:
            raise ValueError("paper protocol requires --acknowledge-multi-day")
        stages = selected_stages(args.substage, args.protocol)
        suite_root = (
            args.run_root.expanduser().resolve()
            / SUITE_ID
            / args.protocol
            / f"seed-{args.seed}"
        )

        def make_plan(stage: str) -> StagePlan:
            return build_stage_plan(
                repo_root=repo_root,
                suite_root=suite_root,
                stage=stage,
                protocol=args.protocol,
                seed=args.seed,
                grpo_data=args.grpo_data,
                rollout_input=args.rollout_input,
                cold_start_data=args.cold_start_data,
                python_bin=args.python_bin,
                sft_python_bin=args.sft_python_bin,
            )

        if args.action in {"plan", "status"}:
            plans = [make_plan(stage) for stage in stages]
            if args.action == "status":
                for plan in plans:
                    status = read_json(suite_root / plan.stage / "status.json")
                    print(status_line(plan.stage, status, plan.fingerprint))
                return 0
            print(f"stages={len(plans)} suite_root={suite_root}")
            for index, plan in enumerate(plans, start=1):
                preview = suite_root / plan.stage / "attempt-0001"
                status = read_json(suite_root / plan.stage / "status.json")
                state = status.get("state", "pending") if status else "pending"
                print(f"\n[{index}] {plan.stage} [{state}]")
                print(f"    fidelity={plan.fidelity}")
                print(f"    fingerprint={plan.fingerprint[:16]}")
                for data in plan.datasets:
                    print(
                        f"    data={data['path']} exists={data['exists']} "
                        f"rows={data.get('rows')} sha256={str(data.get('sha256', 'missing'))[:16]}"
                    )
                for model in plan.models:
                    print(f"    model[{model['role']}]={model['model_id']}@{model['revision']}")
                print("    " + display_command(plan, preview))
            return 0

        lock = acquire_gpu_lock(os.environ.get("CUDA_VISIBLE_DEVICES", "0,1"))
        failures = 0
        try:
            for stage in stages:
                # Rebuild before every stage so SFT can consume the completed
                # rollout attempt produced earlier in this same invocation.
                plan = make_plan(stage)
                result = run_stage(plan, repo_root, suite_root, args.retry_failed)
                failures += int(result != 0)
                if result != 0 and not args.keep_going:
                    break
        finally:
            lock.close()
        return 1 if failures else 0
    except (FileNotFoundError, json.JSONDecodeError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
