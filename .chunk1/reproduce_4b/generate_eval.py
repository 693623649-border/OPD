#!/usr/bin/env python3
"""Generate repeated evaluation samples from a VERL-format parquet file.

The parquet ``prompt`` column is already the source of truth.  This script does
not append a reasoning instruction or any other prompt suffix; it only renders
the stored messages with the model tokenizer's chat template.

Heavy dependencies (pandas and vLLM) are intentionally imported only after CLI
parsing so that ``--help`` works in a lightweight environment.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_N = 16
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95
DEFAULT_MAX_TOKENS = 31_744
DEFAULT_STOP_TOKENS = ("<|im_end|>", "<|endoftext|>")


def positive_int(value: str) -> int:
    """argparse converter for strictly positive integers."""

    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def probability(value: str) -> float:
    """argparse converter for probabilities in the interval (0, 1]."""

    parsed = float(value)
    if not 0.0 < parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be in the interval (0, 1]")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate n sampled responses per prompt with vLLM. Existing "
            "parquet prompt messages are used verbatim."
        )
    )
    parser.add_argument("--model", required=True, help="Hugging Face model ID or local checkpoint path.")
    parser.add_argument("--revision", help="Immutable model commit/revision passed to vLLM.")
    parser.add_argument(
        "--tokenizer",
        help=(
            "Optional tokenizer ID/path passed to vLLM. Useful for checkpoints "
            "whose tokenizer_config.json has no chat_template."
        ),
    )
    parser.add_argument("--tokenizer-revision", help="Immutable tokenizer commit/revision passed to vLLM.")
    parser.add_argument("--input-parquet", required=True, type=Path, help="VERL-format evaluation parquet.")
    parser.add_argument("--output-jsonl", required=True, type=Path, help="Destination JSONL file.")
    parser.add_argument("--n", type=positive_int, default=DEFAULT_N, help="Samples per prompt (default: 16).")
    parser.add_argument(
        "--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature (default: 0.7)."
    )
    parser.add_argument("--top-p", type=probability, default=DEFAULT_TOP_P, help="Nucleus top-p (default: 0.95).")
    parser.add_argument(
        "--max-tokens",
        type=positive_int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum generated tokens per response (default: 31744).",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=positive_int,
        default=1,
        help="vLLM tensor parallel size. Use 2 to shard across two visible GPUs (default: 1).",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        help="Optional CUDA_VISIBLE_DEVICES value, for example '0' or '0,1'.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=probability,
        default=0.90,
        help="Fraction of each visible GPU available to vLLM (default: 0.90).",
    )
    parser.add_argument("--dtype", default="auto", help="vLLM dtype (default: auto).")
    parser.add_argument("--max-model-len", type=positive_int, help="Optional vLLM model context-length override.")
    parser.add_argument("--batch-size", type=positive_int, default=64, help="Prompts submitted per vLLM call.")
    parser.add_argument("--limit", type=positive_int, help="Only load the first N prompts (for smoke tests).")
    parser.add_argument("--seed", type=int, help="Optional vLLM sampling seed.")
    parser.add_argument(
        "--thinking",
        choices=("auto", "on", "off"),
        default="off",
        help=(
            "Chat-template thinking mode: off matches the paper evaluation; "
            "auto preserves the tokenizer default (default: off)."
        ),
    )
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow custom model/tokenizer code.")
    parser.add_argument("--enforce-eager", action="store_true", help="Disable CUDA graphs in vLLM.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file.")
    return parser


def _to_builtin(value: Any) -> Any:
    """Convert common parquet/numpy containers to JSON-friendly Python values."""

    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
        value = value.tolist()
    if hasattr(value, "item") and not isinstance(value, (str, bytes, bytearray)):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, Mapping):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    return value


def normalize_messages(prompt: Any) -> list[dict[str, Any]]:
    """Normalize a parquet prompt cell without changing its textual content."""

    prompt = _to_builtin(prompt)
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    if isinstance(prompt, Mapping):
        prompt = [prompt]
    if not isinstance(prompt, Sequence) or isinstance(prompt, (bytes, bytearray)):
        raise ValueError(f"Unsupported prompt value of type {type(prompt).__name__}")

    messages: list[dict[str, Any]] = []
    for index, message in enumerate(prompt):
        if not isinstance(message, Mapping):
            raise ValueError(f"Prompt message {index} is not a mapping")
        normalized = {str(key): _to_builtin(value) for key, value in message.items()}
        if "content" not in normalized:
            raise ValueError(f"Prompt message {index} has no 'content' field")
        normalized.setdefault("role", "user")
        messages.append(normalized)
    if not messages:
        raise ValueError("Prompt message list is empty")
    return messages


def _ground_truth(row: Mapping[str, Any]) -> str:
    reward_model = _to_builtin(row.get("reward_model"))
    if isinstance(reward_model, Mapping) and reward_model.get("ground_truth") is not None:
        return str(reward_model["ground_truth"])
    for key in ("answer", "ground_truth"):
        if row.get(key) is not None:
            return str(_to_builtin(row[key]))
    raise ValueError("Row has neither reward_model.ground_truth nor answer/ground_truth")


def _example_id(row: Mapping[str, Any], row_index: int) -> Any:
    extra_info = _to_builtin(row.get("extra_info"))
    if isinstance(extra_info, Mapping) and extra_info.get("index") is not None:
        return extra_info["index"]
    for key in ("example_id", "id"):
        if row.get(key) is not None:
            return _to_builtin(row[key])
    return row_index


def load_samples(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Load prompt messages and answers from a VERL-format parquet file."""

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - depends on runtime image
        raise RuntimeError("Reading parquet requires pandas and a parquet engine such as pyarrow") from exc

    frame = pd.read_parquet(path)
    if "prompt" not in frame.columns:
        raise ValueError(f"{path} has no 'prompt' column")
    if limit is not None:
        frame = frame.iloc[:limit]

    samples: list[dict[str, Any]] = []
    for row_index, row in enumerate(frame.to_dict(orient="records")):
        samples.append(
            {
                "example_id": _example_id(row, row_index),
                "row_index": row_index,
                "data_source": _to_builtin(row.get("data_source")),
                "prompt": normalize_messages(row["prompt"]),
                "answer": _ground_truth(row),
            }
        )
    return samples


def render_prompt(tokenizer: Any, messages: list[dict[str, Any]], thinking: str = "auto") -> str:
    """Render stored messages while preserving the template default in auto mode."""

    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if thinking != "auto":
        kwargs["enable_thinking"] = thinking == "on"
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError as exc:
        # Older non-Qwen tokenizers may reject the Qwen-specific template kwarg.
        # Qwen3 tokenizers accept it, so their non-thinking behavior remains explicit.
        if "enable_thinking" not in str(exc):
            raise
        kwargs.pop("enable_thinking")
        return tokenizer.apply_chat_template(messages, **kwargs)


def batched(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def resolve_stop_token_ids(tokenizer: Any) -> list[int]:
    """Resolve the two Qwen stop markers used by the paper's evaluator."""

    stop_ids: list[int] = []
    for token in DEFAULT_STOP_TOKENS:
        try:
            encoded = tokenizer.encode(token, add_special_tokens=False)
        except (AttributeError, TypeError, ValueError):
            continue
        if encoded:
            token_id = int(encoded[0])
            if token_id not in stop_ids:
                stop_ids.append(token_id)
    return stop_ids


def _sampling_metadata(args: argparse.Namespace, stop_token_ids: Sequence[int]) -> dict[str, Any]:
    return {
        "n": args.n,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "thinking": args.thinking,
        "stop_token_ids": list(stop_token_ids),
        "model": args.model,
        "revision": args.revision,
        "tokenizer": args.tokenizer,
        "tokenizer_revision": args.tokenizer_revision,
    }


def run_generation(args: argparse.Namespace) -> int:
    """Load vLLM, generate samples, and stream one response per JSONL row."""

    if args.temperature < 0.0:
        raise ValueError("--temperature must be non-negative")
    if args.cuda_visible_devices:
        visible = [item.strip() for item in args.cuda_visible_devices.split(",") if item.strip()]
        if len(visible) < args.tensor_parallel_size:
            raise ValueError("--tensor-parallel-size exceeds the number of --cuda-visible-devices")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(visible)

    # This host's driver cannot load the CUDA kernel produced by FlashInfer's
    # JIT sampler. Keep the portable vLLM PyTorch sampler as the default while
    # still allowing a newer, compatible host to opt back in with value ``1``.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    output_path = args.output_jsonl.expanduser()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} already exists; pass --overwrite to replace it")

    samples = load_samples(args.input_parquet.expanduser(), args.limit)
    if not samples:
        raise ValueError(f"No prompts found in {args.input_parquet}")

    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:  # pragma: no cover - requires GPU runtime
        raise RuntimeError("Generation requires vLLM; install it in the inference environment") from exc

    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": args.dtype,
        "trust_remote_code": args.trust_remote_code,
        "enforce_eager": args.enforce_eager,
    }
    if args.tokenizer:
        llm_kwargs["tokenizer"] = args.tokenizer
    if args.revision:
        llm_kwargs["revision"] = args.revision
    if args.tokenizer_revision:
        llm_kwargs["tokenizer_revision"] = args.tokenizer_revision
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len

    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    stop_token_ids = resolve_stop_token_ids(tokenizer)
    sampling_kwargs: dict[str, Any] = {
        "n": args.n,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }
    if stop_token_ids:
        sampling_kwargs["stop_token_ids"] = stop_token_ids
    if args.seed is not None:
        sampling_kwargs["seed"] = args.seed
    sampling_params = SamplingParams(**sampling_kwargs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Generate into a same-directory temporary file and publish it only after
    # the complete workload succeeds.  A killed vLLM process must never leave
    # a partial JSONL at the registered artifact path, and the default commit
    # is exclusive so a concurrent or resumed evaluator cannot overwrite a
    # completed generation.
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.tmp-",
        text=True,
    )
    temporary_path = Path(temporary_name)
    written = 0
    sampling_metadata = _sampling_metadata(args, stop_token_ids)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output_file:
            for sample_batch in batched(samples, args.batch_size):
                rendered = [
                    render_prompt(tokenizer, sample["prompt"], thinking=args.thinking)
                    for sample in sample_batch
                ]
                batch_outputs = llm.generate(rendered, sampling_params, use_tqdm=False)
                if len(batch_outputs) != len(sample_batch):
                    raise RuntimeError("vLLM returned a different number of request outputs than prompts")
                for sample, request_output in zip(sample_batch, batch_outputs):
                    if len(request_output.outputs) != args.n:
                        raise RuntimeError(
                            "vLLM returned a different number of candidates than the registered n"
                        )
                    for rollout_id, candidate in enumerate(request_output.outputs):
                        record = {
                            **sample,
                            "rollout_id": rollout_id,
                            "response": candidate.text,
                            "finish_reason": (
                                None
                                if getattr(candidate, "finish_reason", None) is None
                                else str(candidate.finish_reason)
                            ),
                            "sampling": sampling_metadata,
                        }
                        output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                        written += 1
                output_file.flush()
                print(f"generated {written}/{len(samples) * args.n} responses", flush=True)
            os.fsync(output_file.fileno())

        if args.overwrite:
            os.replace(temporary_path, output_path)
        else:
            # hard-link creation is atomic and fails if the destination became
            # occupied after the initial existence check.
            os.link(temporary_path, output_path)
            temporary_path.unlink()
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        written = run_generation(args)
    except (FileExistsError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"wrote {written} responses to {args.output_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
