from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = MODULE_DIR.parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import formal_cold_start_rollout as formal_rollout  # noqa: E402
import run_ablations  # noqa: E402
import run_upstream  # noqa: E402


class UpstreamLauncherTest(unittest.TestCase):
    def shell_dry(self, script: str, **overrides: str) -> subprocess.CompletedProcess[str]:
        environment = {**os.environ, "DRY_RUN": "1", **overrides}
        return subprocess.run(
            ["bash", str(MODULE_DIR / script)],
            cwd=REPO_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

    def test_grpo_paper_dry_run_preserves_table1_contract(self) -> None:
        result = self.shell_dry("run_grpo_teacher_4b.sh", PRESET="paper")
        self.assertEqual(result.returncode, 0, result.stderr)
        for fragment in (
            "Qwen/Qwen3-4B-Base",
            "906bfd4b4dc7f14ee4320094d8b41684abff8539",
            "datasets/dapo-math-17k-processed.parquet",
            "500bd8c45eca355b98f9ba6f3213194a72bd42c73c5e9569c6fbbb1b51bd0b39",
            r"\boxed{{}}",
            "algorithm.adv_estimator=grpo",
            "data.train_batch_size=64",
            "actor_rollout_ref.actor.ppo_mini_batch_size=64",
            "actor_rollout_ref.rollout.n=8",
            "data.max_prompt_length=1024",
            "data.max_response_length=7168",
            "actor_rollout_ref.actor.optim.lr=1e-6",
            "actor_rollout_ref.actor.loss_agg_mode=token-mean",
            "algorithm.kl_ctrl.kl_coef=0.0",
            "2xA100 hardware adaptation",
            "8xA800-80G",
        ):
            self.assertIn(fragment, result.stdout)
        self.assertNotIn("trainer.total_training_steps=", result.stdout)

    def test_grpo_smoke_is_explicitly_non_paper_and_bounded(self) -> None:
        result = self.shell_dry("run_grpo_teacher_4b.sh", PRESET="smoke")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("not a paper result", result.stdout)
        self.assertIn("data.train_batch_size=2", result.stdout)
        self.assertIn("actor_rollout_ref.rollout.n=2", result.stdout)
        self.assertIn("trainer.total_training_steps=2", result.stdout)

    def test_sft_paper_dry_run_preserves_table3_contract_and_vendored_source(self) -> None:
        result = self.shell_dry("run_cold_start_sft.sh", PRESET="paper")
        self.assertEqual(result.returncode, 0, result.stderr)
        for fragment in (
            "vendored LLaMA-Factory 0.9.5",
            "PYTHONPATH=",
            "python -m llamafactory.cli train",
            "--model_revision ea980cb0a6c2ae4b936e82123acc929f1cec04c1",
            "--finetuning_type full",
            "--template qwen3",
            "--cutoff_len 14336",
            "--per_device_train_batch_size 8",
            "--gradient_accumulation_steps 1",
            "--learning_rate 1.0e-5",
            "--num_train_epochs 1.0",
            "--lr_scheduler_type cosine",
            "--warmup_ratio 0.05",
            "--bf16 true",
        ):
            self.assertIn(fragment, result.stdout)
        self.assertNotIn("--max_steps", result.stdout)

    def test_sft_binds_torchrun_to_the_selected_python_environment(self) -> None:
        source = (MODULE_DIR / "run_cold_start_sft.sh").read_text(encoding="utf-8")
        self.assertIn('LAUNCHER_BIN="${OUTPUT_DIR}/launcher_bin"', source)
        self.assertIn('-m torch.distributed.run "$@"', source)
        self.assertIn('export PATH="${LAUNCHER_BIN}:${PATH}"', source)

    def test_formal_rollout_dry_run_fixes_paper_sampling(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(MODULE_DIR / "formal_cold_start_rollout.py"),
                "--input-parquet",
                "/does/not/need/to/exist.parquet",
                "--output-dir",
                "/tmp/formal-rollout-dry-test",
                "--dry-run",
            ],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["responses_per_prompt"], 1)
        self.assertEqual(
            payload["sampling"],
            formal_rollout.sampling_contract(
                formal_rollout.MODEL_ID, formal_rollout.MODEL_REVISION, 42
            ),
        )
        self.assertEqual(payload["sampling"]["temperature"], 0.7)
        self.assertEqual(payload["sampling"]["top_p"], 0.95)
        self.assertEqual(payload["sampling"]["top_k"], -1)
        self.assertEqual(payload["sampling"]["max_tokens"], 12_288)
        self.assertEqual(payload["sampling"]["base_seed"], 42)
        self.assertNotIn("seed", payload["sampling"])

    def test_formal_filter_and_selection_are_auditable_and_deterministic(self) -> None:
        valid = r"Reasoning. Final answer: \boxed{7}."
        self.assertEqual(formal_rollout.filter_response(valid, "stop"), (True, "accepted"))
        self.assertEqual(formal_rollout.filter_response(valid, "length"), (False, "truncated"))
        self.assertEqual(formal_rollout.filter_response("answer 7", "stop"), (False, "no_boxed"))
        first = formal_rollout.deterministic_selection(100, 8, 42)
        second = formal_rollout.deterministic_selection(100, 8, 42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 8)
        self.assertEqual(formal_rollout.generation_seed(42, 0), 42)
        self.assertEqual(formal_rollout.generation_seed(42, 1), 43)
        current, remaining = formal_rollout.take_same_attempt_batch(
            [(0, "a", 0), (1, "b", 0), (2, "c", 1), (3, "d", 1)], 8
        )
        self.assertEqual([item[0] for item in current], [0, 1])
        self.assertEqual([item[0] for item in remaining], [2, 3])

    def test_resume_validates_effective_retry_seed(self) -> None:
        sampling = formal_rollout.sampling_contract(
            formal_rollout.MODEL_ID, formal_rollout.MODEL_REVISION, 42
        )
        record = {
            "global_index": 7,
            "attempt": 2,
            "generation_seed": 43,
            "sampling": sampling,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "part.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            self.assertEqual(set(formal_rollout.accepted_by_index(path, sampling)), {7})
            record["generation_seed"] = 42
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "generation_seed"):
                formal_rollout.accepted_by_index(path, sampling)


class UpstreamRunnerTest(unittest.TestCase):
    def invoke(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(MODULE_DIR / "run_upstream.py"), *arguments],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

    def test_run_requires_explicit_yes_before_preflight(self) -> None:
        result = self.invoke("run", "--protocol", "smoke", "--substage", "cold-start-rollout")
        self.assertEqual(result.returncode, 2)
        self.assertIn("requires --yes", result.stderr)

    def test_paper_requires_explicit_multi_day_acknowledgement(self) -> None:
        result = self.invoke(
            "run", "--yes", "--protocol", "paper", "--substage", "cold-start-rollout"
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("--acknowledge-multi-day", result.stderr)

    def test_calibration_excludes_sft_by_default_and_rejects_explicit_sft(self) -> None:
        self.assertEqual(
            run_upstream.selected_stages((), "calibration"),
            ("grpo-teacher", "cold-start-rollout"),
        )
        with self.assertRaisesRegex(ValueError, "unsupported"):
            run_upstream.selected_stages(("cold-start-sft",), "calibration")

    def test_atomic_status_round_trip_and_attempt_number(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            status_path = root / "stage" / "status.json"
            run_upstream.atomic_json(status_path, {"state": "running", "attempt": 1})
            self.assertEqual(run_upstream.read_json(status_path)["state"], "running")
            (root / "stage" / "attempt-0001").mkdir()
            (root / "stage" / "attempt-0003").mkdir()
            self.assertEqual(run_upstream.attempt_number(root / "stage"), 4)
            self.assertFalse(any(status_path.parent.glob(".status.json.tmp-*")))

    def test_manifest_contains_source_data_and_pinned_model_revision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "sft.jsonl"
            data.write_text(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "q"},
                            {"role": "assistant", "content": "a"},
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            suite_root = root / "suite"
            plan = run_upstream.build_stage_plan(
                repo_root=REPO_ROOT,
                suite_root=suite_root,
                stage="cold-start-sft",
                protocol="smoke",
                seed=42,
                grpo_data=REPO_ROOT / "datasets/dapo-math-17k-processed.parquet",
                rollout_input=root / "rollout.parquet",
                cold_start_data=data,
                python_bin=REPO_ROOT / ".venv-opd/bin/python",
                sft_python_bin=REPO_ROOT / ".venv-sft/bin/python",
            )
            manifest = run_upstream.attempt_manifest(
                plan, repo_root=REPO_ROOT, run_dir=root / "attempt-0001", attempt=1
            )
        self.assertTrue(manifest["source"]["files"])
        self.assertEqual(manifest["data"][0]["rows"], 1)
        self.assertEqual(
            manifest["models"][0]["revision"],
            "ea980cb0a6c2ae4b936e82123acc929f1cec04c1",
        )
        self.assertEqual(manifest["resume_policy"], "fresh attempt; no training checkpoint resume")

    def test_grpo_plan_pins_author_released_processed_data_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            plan = run_upstream.build_stage_plan(
                repo_root=REPO_ROOT,
                suite_root=Path(directory) / "suite",
                stage="grpo-teacher",
                protocol="paper",
                seed=42,
                grpo_data=REPO_ROOT / "datasets/dapo-math-17k-processed.parquet",
                rollout_input=Path(directory) / "rollout.parquet",
                cold_start_data=None,
                python_bin=REPO_ROOT / ".venv-opd/bin/python",
                sft_python_bin=REPO_ROOT / ".venv-sft/bin/python",
            )
        self.assertEqual(plan.datasets[0]["sha256"], run_upstream.GRPO_RELEASED_DATA_SHA256)
        self.assertIn("dapo-math-17k-processed.parquet", plan.datasets[0]["path"])
        self.assertIn(r"\boxed{{}}", plan.environment["DATA_FIDELITY"])
        run_upstream.validate_run_inputs(plan)

    def test_status_line_is_pure_and_reports_pending_or_attempt(self) -> None:
        fingerprint = "a" * 64
        self.assertEqual(
            run_upstream.status_line("grpo-teacher", None, fingerprint),
            "grpo-teacher\tpending\taaaaaaaaaaaaaaaa",
        )
        line = run_upstream.status_line(
            "grpo-teacher", {"state": "completed", "attempt": 2, "exit_code": 0}, fingerprint
        )
        self.assertIn("completed attempt=2 exit=0", line)

    def test_gpu_locks_conflict_with_overlapping_ablation_allocations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lock_dir = Path(directory)
            upstream = run_upstream.acquire_gpu_lock("101,102", lock_dir=lock_dir)
            try:
                with self.assertRaisesRegex(RuntimeError, "GPU 102|GPU 101"):
                    run_ablations.acquire_gpu_lock("102,103", lock_dir=lock_dir)
                disjoint = run_ablations.acquire_gpu_lock("103,104", lock_dir=lock_dir)
                disjoint.close()
            finally:
                upstream.close()
            released = run_ablations.acquire_gpu_lock("101,102", lock_dir=lock_dir)
            released.close()


if __name__ == "__main__":
    unittest.main()
