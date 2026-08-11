#!/usr/bin/env python3
"""Fail-fast checks for the local OPD launcher."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from pin_models import COMMIT_RE, is_hub_network_error, resolve_hub_revision


REQUIRED_IMPORTS = (
    "torch",
    "transformers",
    "vllm",
    "ray",
    "hydra",
    "pandas",
    "pyarrow",
    "tensordict",
    "flash_attn",
    "math_verify",
    "pylatexenc",
    "verl",
)
REQUIRED_VERSIONS = {
    "torch": "==2.8.0",
    "transformers": ">=4.55.2,<5",
    "vllm": "==0.11.0",
    "ray": ">=2.48,<3",
    "tensordict": ">=0.8,<=0.10,!=0.9.0",
    "flash-attn": "==2.8.1",
}
EXPECTED_VERL_VERSION = "0.7.0.dev"
BOXED_SUFFIX = "Please reason step by step, and put your final answer within \\boxed"
ORIGINAL_PREFIX = "Solve the following math problem step by step."
ORIGINAL_SUFFIX = 'Remember to put your answer on its own line after "Answer:".'
TOP_K_STRATEGIES = ("only_stu", "only_tch", "intersection", "union", "union-intersection")
PADDED_VOCAB_SAFE_STRATEGIES = frozenset(("only_stu", "intersection"))


@dataclass(frozen=True)
class GPUState:
    index: int
    name: str
    total_mib: int
    free_mib: int
    utilization: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate environment, data, tokenizers, and idle GPUs for OPD.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--student", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--teacher", default="lllyx/Qwen3-4B-Base-GRPO")
    parser.add_argument("--student-revision", default="auto")
    parser.add_argument("--teacher-revision", default="auto")
    parser.add_argument("--top-k-strategy", choices=TOP_K_STRATEGIES, default="only_stu")
    parser.add_argument("--train-data", type=Path, default=Path("datasets/dapo-math-17k.parquet"))
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--min-free-gib", type=float, default=70.0)
    parser.add_argument("--skip-gpu", action="store_true")
    parser.add_argument("--skip-model-check", action="store_true")
    parser.add_argument("--skip-imports", action="store_true")
    return parser


def parse_gpu_csv(text: str) -> list[GPUState]:
    states: list[GPUState] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            raise ValueError(f"unexpected nvidia-smi CSV on line {line_number}: {line!r}")
        try:
            states.append(
                GPUState(
                    index=int(parts[0]),
                    name=parts[1],
                    total_mib=int(parts[2]),
                    free_mib=int(parts[3]),
                    utilization=int(parts[4]),
                )
            )
        except ValueError as exc:
            raise ValueError(f"non-numeric nvidia-smi field on line {line_number}: {line!r}") from exc
    return states


def visible_gpu_indices() -> list[int] | None:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return None
    if not raw.strip():
        return []
    indices: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not re.fullmatch(r"\d+", item):
            raise ValueError(
                "CUDA_VISIBLE_DEVICES must contain unique numeric GPU indices; "
                f"cannot safely map selector {item!r}"
            )
        indices.append(int(item))
    if len(set(indices)) != len(indices):
        raise ValueError(f"CUDA_VISIBLE_DEVICES contains duplicate GPU indices: {raw!r}")
    return indices


def query_gpus() -> list[GPUState]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise RuntimeError("nvidia-smi is not available")
    result = subprocess.run(
        [
            executable,
            "--query-gpu=index,name,memory.total,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    states = parse_gpu_csv(result.stdout)
    visible = visible_gpu_indices()
    if visible is not None:
        by_index = {state.index: state for state in states}
        states = [by_index[index] for index in visible if index in by_index]
    return states


def check_gpus(expected: int, min_free_gib: float) -> list[str]:
    errors: list[str] = []
    try:
        states = query_gpus()
    except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as exc:
        return [f"GPU query failed: {exc}"]
    if len(states) != expected:
        errors.append(
            f"need exactly {expected} visible GPUs, found {len(states)}; "
            "set CUDA_VISIBLE_DEVICES to the intended unique numeric indices"
        )
    try:
        import torch

        torch_count = torch.cuda.device_count()
        if not torch.cuda.is_available():
            errors.append("torch.cuda.is_available() is false")
        if torch_count != expected:
            errors.append(f"PyTorch sees {torch_count} CUDA devices, expected exactly {expected}")
    except Exception as exc:
        errors.append(f"PyTorch CUDA visibility check failed: {type(exc).__name__}: {exc}")
    minimum_mib = int(min_free_gib * 1024)
    for state in states:
        print(
            f"[gpu] {state.index}: {state.name}, free={state.free_mib / 1024:.1f}/"
            f"{state.total_mib / 1024:.1f} GiB, util={state.utilization}%"
        )
        if state.free_mib < minimum_mib:
            errors.append(
                f"GPU {state.index} has only {state.free_mib / 1024:.1f} GiB free; "
                f"requires at least {min_free_gib:.1f} GiB"
            )
        if state.utilization >= 20:
            errors.append(f"GPU {state.index} is busy ({state.utilization}% utilization)")
    return errors


def module_version(module: Any) -> str:
    return str(getattr(module, "__version__", getattr(module, "version", "unknown")))


def check_imports(repo_root: Path) -> list[str]:
    errors: list[str] = []
    for name in REQUIRED_IMPORTS:
        try:
            module = importlib.import_module(name)
            print(f"[import] {name}={module_version(module)}")
        except Exception as exc:  # importing CUDA extensions can raise more than ImportError
            errors.append(f"cannot import {name}: {type(exc).__name__}: {exc}")

    try:
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version

        for distribution, specifier in REQUIRED_VERSIONS.items():
            installed = importlib.metadata.version(distribution)
            if Version(installed) not in SpecifierSet(specifier):
                errors.append(f"{distribution}=={installed} does not satisfy {specifier}")
    except Exception as exc:
        errors.append(f"cannot validate dependency versions: {type(exc).__name__}: {exc}")

    try:
        verl = importlib.import_module("verl")
        verl_file = Path(verl.__file__).resolve()
        vendored_root = (repo_root / "verl").resolve()
        if vendored_root not in verl_file.parents:
            errors.append(f"imported verl is not this repository's vendored package: {verl_file}")
        version_file = vendored_root / "verl" / "version" / "version"
        version = version_file.read_text(encoding="utf-8").strip()
        if version != EXPECTED_VERL_VERSION:
            errors.append(f"vendored verl version is {version}, expected {EXPECTED_VERL_VERSION}")
    except Exception as exc:
        errors.append(f"cannot validate vendored verl: {exc}")
    return errors


def _as_python(value: Any) -> Any:
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
        return value.tolist()
    return value


def check_dataset(path: Path) -> list[str]:
    errors: list[str] = []
    if not path.is_file():
        return [f"training parquet does not exist: {path}"]
    try:
        import pandas as pd

        frame = pd.read_parquet(path, columns=["prompt", "reward_model", "data_source"])
    except Exception as exc:
        return [f"cannot read training parquet {path}: {type(exc).__name__}: {exc}"]
    if frame.empty:
        return [f"training parquet is empty: {path}"]
    original_prompts = 0
    teacher_aligned_prompts = 0
    unknown_templates = 0
    missing_ground_truth = 0
    malformed_messages = 0
    missing_data_source = 0
    for prompt, reward_model, data_source in zip(
        frame["prompt"], frame["reward_model"], frame["data_source"]
    ):
        prompt = _as_python(prompt)
        if not isinstance(prompt, list) or not prompt or not isinstance(prompt[-1], dict):
            unknown_templates += 1
        else:
            content = str(prompt[-1].get("content", ""))
            if content.startswith(ORIGINAL_PREFIX) and ORIGINAL_SUFFIX in content:
                original_prompts += 1
            elif BOXED_SUFFIX in content:
                teacher_aligned_prompts += 1
            else:
                unknown_templates += 1
        if not isinstance(prompt, list) or any(
            not isinstance(message, dict)
            or not isinstance(message.get("role"), str)
            or not isinstance(message.get("content"), str)
            for message in prompt
        ):
            malformed_messages += 1
        if (
            not isinstance(reward_model, dict)
            or reward_model.get("ground_truth") is None
            or not str(reward_model.get("ground_truth")).strip()
        ):
            missing_ground_truth += 1
        if not isinstance(data_source, str) or not data_source.strip():
            missing_data_source += 1
    digest = hashlib.sha256()
    with path.open("rb") as parquet_file:
        for block in iter(lambda: parquet_file.read(1024 * 1024), b""):
            digest.update(block)
    print(
        f"[data] {path}: rows={len(frame)}, original-dapo={original_prompts}, "
        f"teacher-aligned={teacher_aligned_prompts}, unknown-template={unknown_templates}, "
        f"sha256={digest.hexdigest()}"
    )
    if unknown_templates:
        errors.append(
            f"{unknown_templates}/{len(frame)} prompts use neither the original DAPO nor teacher-aligned template"
        )
    if missing_ground_truth:
        errors.append(f"{missing_ground_truth}/{len(frame)} rows have no reward_model.ground_truth")
    if malformed_messages:
        errors.append(f"{malformed_messages}/{len(frame)} rows have malformed role/content messages")
    if missing_data_source:
        errors.append(f"{missing_data_source}/{len(frame)} rows have no data_source")
    return errors


def check_source_contract(repo_root: Path) -> list[str]:
    errors: list[str] = []
    source_files = {
        "trainer": repo_root / "verl" / "verl" / "trainer" / "ppo" / "ray_trainer.py",
        "worker": repo_root / "verl" / "verl" / "workers" / "fsdp_workers.py",
        "loss": repo_root / "verl" / "verl" / "trainer" / "ppo" / "core_algos.py",
        "opd": repo_root / "verl" / "verl" / "utils" / "opd.py",
    }
    required_tokens = {
        "trainer": (
            "val-topk/overlap_ratio",
            "compute_distillation_reward",
            "eq7_intersection_advantage",
            "opd/abs_entropy_gap",
            "position-entropy/student_mean_by_bin",
        ),
        "worker": ("student_top_k_ids", "teacher_on_student_log_probs"),
        "loss": ("Handle 3D tensors from top-k sampling",),
        "opd": (
            "build_topk_distillation_tensors",
            "validate_topk_replay_tensors",
            "eq8_absolute_entropy_gap",
            "support_weight_normalization",
        ),
    }
    for label, path in source_files.items():
        if not path.is_file():
            errors.append(f"missing vendored OPD {label} source: {path}")
            continue
        text = path.read_text(encoding="utf-8")
        for token in required_tokens[label]:
            if token not in text:
                errors.append(f"vendored OPD {label} source is missing marker {token!r}")
    return errors


def resolve_pretrained_reference(source: str, revision: str) -> tuple[str, str, bool]:
    """Resolve mutable remote references online before a loader may consult cache."""

    local = Path(source).expanduser()
    if local.exists():
        return str(local.resolve()), "auto", False
    resolved_revision, cache_only = resolve_hub_revision(source, revision)
    if cache_only:
        print(
            f"[models] Hub network unavailable; using exact cached revision "
            f"{source}@{resolved_revision}"
        )
    return source, resolved_revision, cache_only


def load_pretrained_metadata(
    loader: Any,
    source: str,
    revision: str,
    *,
    cache_only: bool = False,
) -> Any:
    """Load metadata, falling back only to the same immutable cached commit."""

    revision_arg = None if revision == "auto" else revision
    kwargs = {"revision": revision_arg, "trust_remote_code": False}
    if cache_only:
        try:
            return loader.from_pretrained(source, **kwargs, local_files_only=True)
        except Exception as cache_exc:
            raise RuntimeError(
                f"Hub is unavailable and fixed revision {source}@{revision} "
                "is not complete in the local cache"
            ) from cache_exc
    try:
        return loader.from_pretrained(source, **kwargs)
    except Exception as exc:
        if not COMMIT_RE.fullmatch(revision) or not is_hub_network_error(exc):
            raise
        print(
            f"[models] Hub network unavailable; retrying exact cached revision "
            f"{source}@{revision}"
        )
        try:
            return loader.from_pretrained(source, **kwargs, local_files_only=True)
        except Exception as cache_exc:
            raise RuntimeError(
                f"Hub is unavailable and fixed revision {source}@{revision} "
                "is not complete in the local cache"
            ) from cache_exc


def check_tokenizers(
    student: str,
    teacher: str,
    student_revision: str = "auto",
    teacher_revision: str = "auto",
    top_k_strategy: str = "only_stu",
) -> list[str]:
    try:
        from transformers import AutoConfig, AutoTokenizer

        student_source, student_revision, student_cache_only = resolve_pretrained_reference(
            student, student_revision
        )
        teacher_source, teacher_revision, teacher_cache_only = resolve_pretrained_reference(
            teacher, teacher_revision
        )
        student_tokenizer = load_pretrained_metadata(
            AutoTokenizer,
            student_source,
            student_revision,
            cache_only=student_cache_only,
        )
        teacher_tokenizer = load_pretrained_metadata(
            AutoTokenizer,
            teacher_source,
            teacher_revision,
            cache_only=teacher_cache_only,
        )
        student_config = load_pretrained_metadata(
            AutoConfig,
            student_source,
            student_revision,
            cache_only=student_cache_only,
        )
        teacher_config = load_pretrained_metadata(
            AutoConfig,
            teacher_source,
            teacher_revision,
            cache_only=teacher_cache_only,
        )
    except Exception as exc:
        return [f"cannot load model metadata/tokenizers: {type(exc).__name__}: {exc}"]

    errors: list[str] = []
    if top_k_strategy not in TOP_K_STRATEGIES:
        return [
            f"unsupported Top-k strategy {top_k_strategy!r}; "
            f"expected one of {', '.join(TOP_K_STRATEGIES)}"
        ]
    student_vocab = student_tokenizer.get_vocab()
    teacher_vocab = teacher_tokenizer.get_vocab()
    mappings_are_dicts = isinstance(student_vocab, dict) and isinstance(teacher_vocab, dict)
    mappings_identical = mappings_are_dicts and student_vocab == teacher_vocab
    if not mappings_are_dicts:
        errors.append("tokenizer get_vocab() must return token->ID dictionaries")
        student_vocab = student_vocab if isinstance(student_vocab, dict) else {}
        teacher_vocab = teacher_vocab if isinstance(teacher_vocab, dict) else {}
    elif not mappings_identical:
        student_tokens = set(student_vocab)
        teacher_tokens = set(teacher_vocab)
        shared_tokens = student_tokens & teacher_tokens
        id_mismatches = sum(
            student_vocab[token] != teacher_vocab[token] for token in shared_tokens
        )
        errors.append(
            "student and teacher token->ID mappings differ "
            f"({len(student_vocab)} vs {len(teacher_vocab)} entries; "
            f"student-only={len(student_tokens - teacher_tokens)}, "
            f"teacher-only={len(teacher_tokens - student_tokens)}, "
            f"shared-ID-mismatches={id_mismatches})"
        )
    # A special-token *default* can differ even when the complete token->id
    # mapping is identical (Qwen3 Base versus Qwen3 Non-thinking does this for
    # EOS).  OPD feeds actor-produced input IDs into the teacher for a forward
    # pass, so shared token IDs are the hard requirement; the teacher does not
    # generate with its EOS default.  Keep this visible without rejecting the
    # paper's pattern-mismatch control.
    special_names = ("bos_token_id", "eos_token_id", "pad_token_id")
    special_mismatches = []
    for name in special_names:
        student_value = getattr(student_tokenizer, name)
        teacher_value = getattr(teacher_tokenizer, name)
        if student_value != teacher_value:
            special_mismatches.append(f"{name}={student_value} vs {teacher_value}")
    if special_mismatches:
        mapping_note = (
            "the full token->ID mapping is identical"
            if mappings_identical
            else "the token->ID mapping mismatch is reported separately"
        )
        print(
            "[models] warning: special-token defaults differ ("
            + ", ".join(special_mismatches)
            + f"); {mapping_note}"
        )
    student_size = getattr(student_config, "vocab_size", None)
    teacher_size = getattr(teacher_config, "vocab_size", None)
    vocab_ids: list[int] = []
    ids_are_valid = mappings_are_dicts
    if mappings_are_dicts:
        if not student_vocab or not teacher_vocab:
            errors.append("student and teacher tokenizer mappings must be non-empty")
            ids_are_valid = False
        for role, vocab in (("student", student_vocab), ("teacher", teacher_vocab)):
            invalid_ids = [
                token_id
                for token_id in vocab.values()
                if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
            ]
            if invalid_ids:
                errors.append(f"{role} tokenizer contains invalid token IDs")
                ids_are_valid = False
        if ids_are_valid:
            vocab_ids = list(student_vocab.values()) + list(teacher_vocab.values())

    maximum_token_id = max(vocab_ids, default=-1)
    configured_sizes_are_valid = True
    for role, configured_size in (("student", student_size), ("teacher", teacher_size)):
        if isinstance(configured_size, bool) or not isinstance(configured_size, int) or configured_size <= 0:
            errors.append(f"{role} model config vocab_size is invalid: {configured_size!r}")
            configured_sizes_are_valid = False
        elif maximum_token_id >= configured_size:
            errors.append(
                f"{role} model config vocab_size={configured_size} cannot represent "
                f"tokenizer max ID {maximum_token_id}"
            )
            configured_sizes_are_valid = False

    if student_size != teacher_size:
        padded_sizes = f"student={student_size}, teacher={teacher_size}"
        padded_difference_is_safe = (
            top_k_strategy in PADDED_VOCAB_SAFE_STRATEGIES
            and mappings_identical
            and ids_are_valid
            and configured_sizes_are_valid
            and (top_k_strategy != "only_stu" or student_size <= teacher_size)
        )
        if padded_difference_is_safe:
            safety_reason = (
                "every student-support ID fits in the teacher output head"
                if top_k_strategy == "only_stu"
                else "the optimization uses only IDs shared by the two Top-k sets"
            )
            print(
                "[models] warning: padded output vocab sizes differ "
                f"({padded_sizes}); allowed for TOP_K_STRATEGY={top_k_strategy} because "
                f"{safety_reason}"
            )
        elif top_k_strategy == "only_stu" and configured_sizes_are_valid and student_size > teacher_size:
            errors.append(
                "padded output vocab sizes differ "
                f"({padded_sizes}); TOP_K_STRATEGY=only_stu is unsafe because a student-support "
                "ID can exceed the teacher output head"
            )
        elif top_k_strategy not in PADDED_VOCAB_SAFE_STRATEGIES:
            errors.append(
                "padded output vocab sizes differ "
                f"({padded_sizes}); TOP_K_STRATEGY={top_k_strategy} consumes teacher support "
                "and is unsafe without explicit filtering of IDs outside the shared tokenizer mapping"
            )
    print(
        "[models] tokenizer mapping entries="
        f"{len(student_vocab)}, max token ID={maximum_token_id}; padded output vocab sizes: "
        f"student={student_size}, teacher={teacher_size}; strategy={top_k_strategy}; "
        f"student={student}, teacher={teacher}"
    )
    print(
        "[models] resolved revisions: student="
        f"{getattr(student_config, '_commit_hash', None) or 'local/unknown'}, teacher="
        f"{getattr(teacher_config, '_commit_hash', None) or 'local/unknown'}"
    )
    return errors


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.expanduser().resolve()
    train_data = args.train_data.expanduser()
    if not train_data.is_absolute():
        train_data = repo_root / train_data

    errors: list[str] = []
    if sys.version_info[:2] != (3, 12):
        errors.append(f"Python 3.12 is required by the pinned FlashAttention wheel; found {sys.version.split()[0]}")
    errors.extend(check_source_contract(repo_root))
    if not args.skip_imports:
        errors.extend(check_imports(repo_root))
    errors.extend(check_dataset(train_data))
    if not args.skip_model_check:
        errors.extend(
            check_tokenizers(
                args.student,
                args.teacher,
                args.student_revision,
                args.teacher_revision,
                args.top_k_strategy,
            )
        )
    if not args.skip_gpu:
        errors.extend(check_gpus(args.gpus, args.min_free_gib))

    if errors:
        print("\nPreflight failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print("\nPreflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
