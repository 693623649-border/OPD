from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import generate_eval  # noqa: E402
import grade_eval  # noqa: E402
import plot_metrics  # noqa: E402


class EvalToolsTest(unittest.TestCase):
    def test_generate_defaults_and_max_tokens_override(self) -> None:
        parser = generate_eval.build_parser()
        args = parser.parse_args(
            [
                "--model",
                "lllyx/Qwen3-4B-Base-GRPO",
                "--tokenizer",
                "Qwen/Qwen3-4B-Base",
                "--input-parquet",
                "eval.parquet",
                "--output-jsonl",
                "eval.jsonl",
                "--max-tokens",
                "4096",
            ]
        )
        self.assertEqual(args.n, 16)
        self.assertAlmostEqual(args.temperature, 0.7)
        self.assertAlmostEqual(args.top_p, 0.95)
        self.assertEqual(args.max_tokens, 4096)
        self.assertEqual(args.tensor_parallel_size, 1)
        self.assertEqual(args.thinking, "off")
        self.assertEqual(args.tokenizer, "Qwen/Qwen3-4B-Base")

    def test_prompt_is_not_modified_and_auto_preserves_template_default(self) -> None:
        class FakeTokenizer:
            def __init__(self) -> None:
                self.messages = None
                self.kwargs = None

            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                self.kwargs = kwargs
                return "rendered"

        messages = [{"role": "user", "content": "Already complete prompt.\\n\\boxed{}"}]
        tokenizer = FakeTokenizer()
        self.assertEqual(generate_eval.render_prompt(tokenizer, messages, thinking="auto"), "rendered")
        self.assertEqual(tokenizer.messages, messages)
        self.assertNotIn("enable_thinking", tokenizer.kwargs)
        self.assertEqual(messages[0]["content"], "Already complete prompt.\\n\\boxed{}")

    def test_qwen_non_thinking_eval_is_explicit(self) -> None:
        class FakeTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                if kwargs["enable_thinking"] is not False:
                    raise AssertionError("thinking was not disabled")
                return messages[0]["content"]

        rendered = generate_eval.render_prompt(
            FakeTokenizer(), [{"role": "user", "content": "prompt"}], thinking="off"
        )
        self.assertEqual(rendered, "prompt")

    def test_generation_wires_tokenizer_two_gpu_tp_and_jsonl_without_real_vllm(self) -> None:
        captured = {}

        class FakeTokenizer:
            def encode(self, token, add_special_tokens=False):
                if add_special_tokens:
                    raise AssertionError("stop tokens must be encoded without added tokens")
                return {"<|im_end|>": [151645], "<|endoftext|>": [151643]}[token]

            def apply_chat_template(self, messages, **kwargs):
                captured["template_kwargs"] = kwargs
                return messages[0]["content"]

        class FakeCandidate:
            def __init__(self, text):
                self.text = text
                self.finish_reason = "stop"

        class FakeRequestOutput:
            def __init__(self):
                self.outputs = [FakeCandidate("one"), FakeCandidate("two")]

        class FakeLLM:
            def __init__(self, **kwargs):
                captured["llm_kwargs"] = kwargs
                captured["flashinfer_sampler"] = os.environ.get("VLLM_USE_FLASHINFER_SAMPLER")

            def get_tokenizer(self):
                return FakeTokenizer()

            def generate(self, prompts, sampling_params, use_tqdm=False):
                captured["sampling_kwargs"] = sampling_params.kwargs
                return [FakeRequestOutput() for _ in prompts]

        class FakeSamplingParams:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        fake_vllm = types.ModuleType("vllm")
        fake_vllm.LLM = FakeLLM
        fake_vllm.SamplingParams = FakeSamplingParams
        samples = [
            {
                "example_id": "sample-0",
                "row_index": 0,
                "data_source": "unit",
                "prompt": [{"role": "user", "content": "stored prompt"}],
                "answer": "1",
            }
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "eval.jsonl"
            args = generate_eval.build_parser().parse_args(
                [
                    "--model",
                    "checkpoint",
                    "--tokenizer",
                    "Qwen/Qwen3-4B-Base",
                    "--revision",
                    "model-commit",
                    "--tokenizer-revision",
                    "tokenizer-commit",
                    "--input-parquet",
                    "unused.parquet",
                    "--output-jsonl",
                    str(output),
                    "--n",
                    "2",
                    "--tensor-parallel-size",
                    "2",
                ]
            )
            with mock.patch.object(generate_eval, "load_samples", return_value=samples):
                with mock.patch.dict(os.environ, {}, clear=True), mock.patch.dict(
                    sys.modules, {"vllm": fake_vllm}
                ):
                    self.assertEqual(generate_eval.run_generation(args), 2)
            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(captured["llm_kwargs"]["tensor_parallel_size"], 2)
        self.assertEqual(captured["llm_kwargs"]["tokenizer"], "Qwen/Qwen3-4B-Base")
        self.assertEqual(captured["llm_kwargs"]["revision"], "model-commit")
        self.assertEqual(captured["llm_kwargs"]["tokenizer_revision"], "tokenizer-commit")
        self.assertEqual(captured["flashinfer_sampler"], "0")
        self.assertIs(captured["template_kwargs"]["enable_thinking"], False)
        self.assertEqual(captured["sampling_kwargs"]["n"], 2)
        self.assertEqual(captured["sampling_kwargs"]["stop_token_ids"], [151645, 151643])
        self.assertEqual([record["response"] for record in records], ["one", "two"])
        self.assertEqual(records[0]["prompt"][0]["content"], "stored prompt")
        self.assertEqual(records[0]["sampling"]["stop_token_ids"], [151645, 151643])
        self.assertEqual(records[0]["sampling"]["revision"], "model-commit")

    def test_generation_failure_never_publishes_partial_registered_output(self) -> None:
        class FakeTokenizer:
            def encode(self, token, add_special_tokens=False):
                return []

            def apply_chat_template(self, messages, **kwargs):
                return messages[0]["content"]

        class FakeCandidate:
            text = "only-one"
            finish_reason = "stop"

        class FakeRequestOutput:
            outputs = [FakeCandidate()]

        class FakeLLM:
            def __init__(self, **kwargs):
                pass

            def get_tokenizer(self):
                return FakeTokenizer()

            def generate(self, prompts, sampling_params, use_tqdm=False):
                return [FakeRequestOutput() for _ in prompts]

        class FakeSamplingParams:
            def __init__(self, **kwargs):
                pass

        fake_vllm = types.ModuleType("vllm")
        fake_vllm.LLM = FakeLLM
        fake_vllm.SamplingParams = FakeSamplingParams
        samples = [{
            "example_id": "sample-0",
            "row_index": 0,
            "data_source": "unit",
            "prompt": [{"role": "user", "content": "stored prompt"}],
            "answer": "1",
        }]
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "responses.jsonl"
            args = generate_eval.build_parser().parse_args([
                "--model", "checkpoint",
                "--input-parquet", "unused.parquet",
                "--output-jsonl", str(output),
                "--n", "2",
            ])
            with mock.patch.object(generate_eval, "load_samples", return_value=samples), mock.patch.dict(
                sys.modules, {"vllm": fake_vllm}
            ):
                with self.assertRaisesRegex(RuntimeError, "registered n"):
                    generate_eval.run_generation(args)
            self.assertFalse(output.exists())
            self.assertEqual(list(Path(temp_dir).iterdir()), [])

    def test_grade_metrics_are_macro_avg_pass_and_format(self) -> None:
        records = [
            {"example_id": "a", "rollout_id": 0, "answer": "x", "response": "ok box"},
            {"example_id": "a", "rollout_id": 1, "answer": "x", "response": "bad"},
            {"example_id": "b", "rollout_id": 0, "answer": "x", "response": "ok box"},
            {"example_id": "b", "rollout_id": 1, "answer": "x", "response": "ok box"},
            # The third rollout is intentionally excluded by N=2.
            {"example_id": "b", "rollout_id": 2, "answer": "x", "response": "bad"},
        ]
        summary, annotated = grade_eval.evaluate_records(
            records,
            n=2,
            grade_fn=lambda response, answer: response.startswith("ok") and answer == "x",
            extract_fn=lambda response: "answer" if "box" in response else None,
        )
        self.assertAlmostEqual(summary["avg@2"], 0.75)
        self.assertAlmostEqual(summary["pass@2"], 1.0)
        self.assertAlmostEqual(summary["format_rate"], 0.75)
        self.assertEqual(summary["num_responses"], 4)
        self.assertEqual(len(annotated), 4)

    def test_strict_n_rejects_incomplete_prompt(self) -> None:
        records = [{"example_id": 0, "answer": "1", "response": "\\boxed{1}"}]
        with self.assertRaisesRegex(ValueError, "exactly 2"):
            grade_eval.evaluate_records(records, 2, lambda *_: True, lambda _: "1", strict_n=True)

    def test_strict_n_rejects_extra_or_duplicate_rollout_ids(self) -> None:
        extra = [
            {"example_id": 0, "rollout_id": index, "answer": "1", "response": "x"}
            for index in range(3)
        ]
        with self.assertRaisesRegex(ValueError, "exactly 2 responses"):
            grade_eval.evaluate_records(extra, 2, lambda *_: True, lambda _: "1", strict_n=True)
        duplicate = [
            {"example_id": 0, "rollout_id": 0, "answer": "1", "response": "x"},
            {"example_id": 0, "rollout_id": 0, "answer": "1", "response": "x"},
        ]
        with self.assertRaisesRegex(ValueError, "rollout IDs 0..1"):
            grade_eval.evaluate_records(duplicate, 2, lambda *_: True, lambda _: "1", strict_n=True)

    def test_grade_outputs_are_write_once_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            grade_eval.write_json(path, {"value": 1})
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                grade_eval.write_json(path, {"value": 2})
            self.assertEqual(path.read_bytes(), original)

    @unittest.skipUnless(importlib.util.find_spec("sympy"), "repository grader requires sympy")
    def test_actual_repository_grader_accepts_decimal_ground_truth(self) -> None:
        # The current lightweight environment may omit pylatexenc.  A minimal
        # import shim is sufficient here because these numeric cases never call
        # the LaTeX-to-text branch; scoring still runs through utils.py itself.
        if importlib.util.find_spec("pylatexenc") is None:
            latex2text = types.ModuleType("latex2text")

            class LatexNodes2Text:
                def latex_to_text(self, expression):
                    return expression

            latex2text.LatexNodes2Text = LatexNodes2Text
            pylatexenc = types.ModuleType("pylatexenc")
            pylatexenc.latex2text = latex2text
            modules = {"pylatexenc": pylatexenc, "pylatexenc.latex2text": latex2text}
        else:
            modules = {}
        with mock.patch.dict(sys.modules, modules):
            grader = grade_eval.load_grader_module()
        self.assertTrue(grader.grade_answer_verl("Reasoning. \\boxed{2}", "2"))
        self.assertFalse(grader.grade_answer_verl("Reasoning. \\boxed{3}", "2"))
        # AMC-style parquet answers can be serialized as decimal-looking strings.
        self.assertTrue(grader.grade_answer_verl("Reasoning. \\boxed{142}", "142.0"))

    def test_read_verl_file_logger_and_entropy_gap(self) -> None:
        rows = [
            {
                "step": 1,
                "data": {
                    "distillation/overlap_ratio": 0.25,
                    "distillation/overlap_token_advantage": 0.5,
                    "actor/entropy": 1.5,
                    "teacher/entropy": 1.2,
                    "actor/grad_norm": 2.0,
                },
            },
            {
                "step": 2,
                "data": {
                    "distillation/overlap_ratio": 0.4,
                    "distillation/overlap_token_advantage": -0.2,
                    "actor/entropy": 1.1,
                    "teacher/entropy": 1.6,
                    "actor/grad_norm": 1.8,
                },
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "run.jsonl"
            log_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            run = plot_metrics.read_file_logger(log_path)
        self.assertEqual([point.value for point in run.series["overlap_ratio"]], [0.25, 0.4])
        self.assertEqual([point.value for point in run.series["overlap_advantage"]], [0.5, -0.2])
        gap = [point.value for point in plot_metrics.entropy_gap(run)]
        self.assertAlmostEqual(gap[0], 0.3)
        self.assertAlmostEqual(gap[1], 0.5)
        self.assertEqual([point.value for point in run.series["grad_norm"]], [2.0, 1.8])

    def test_plot_rejects_mixed_legacy_and_eq7_advantage_schemas(self) -> None:
        def run_with_schema(label, schema):
            series = {name: [] for name in plot_metrics.METRIC_ALIASES}
            series["overlap_advantage"] = [plot_metrics.Point(1, -0.1)]
            return plot_metrics.RunMetrics(
                label=label,
                source=Path(f"{label}.jsonl"),
                series=series,
                metric_schema=schema,
            )

        runs = [run_with_schema("legacy", None), run_with_schema("eq7", 2)]
        with self.assertRaisesRegex(ValueError, "mix legacy proxy"):
            plot_metrics.validate_metric_schemas(runs)
        plot_metrics.validate_metric_schemas(runs, allow_mixed=True)

    def test_plot_prefers_logged_eq8_entropy_gap_over_scalar_proxy(self) -> None:
        row = {
            "step": 1,
            "data": {
                "actor/entropy": 1.0,
                "teacher/entropy": 1.0,
                "opd/entropy_gap_schema_version": 1,
                "opd/abs_entropy_gap": 0.75,
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "run.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            run = plot_metrics.read_file_logger(path)
        self.assertEqual([point.value for point in plot_metrics.entropy_gap(run)], [0.75])

    def test_help_does_not_require_gpu_dependencies(self) -> None:
        for script in (
            "generate_eval.py",
            "grade_eval.py",
            "plot_metrics.py",
            "plot_position_entropy.py",
            "prepare_ablation_data.py",
            "run_ablations.py",
            "evaluate_ablation.py",
            "aggregate_ablations.py",
        ):
            with self.subTest(script=script):
                result = subprocess.run(
                    [sys.executable, str(MODULE_DIR / script), "--help"],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout.lower())


if __name__ == "__main__":
    unittest.main()
