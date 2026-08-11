from __future__ import annotations

import os
import json
import importlib.util
import subprocess
import sys
import tempfile
import unittest
import requests
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


MODULE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = MODULE_DIR.parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import preflight  # noqa: E402
import pin_models  # noqa: E402
from verl.utils.opd import (  # noqa: E402
    binned_masked_mean,
    build_topk_distillation_tensors,
    eq7_intersection_advantage,
    eq8_absolute_entropy_gap,
    normalized_reward_weights,
    reward_weights,
    validate_topk_replay_tensors,
)

tracking_spec = importlib.util.spec_from_file_location(
    "opd_tracking", REPO_ROOT / "verl" / "verl" / "utils" / "tracking.py"
)
assert tracking_spec is not None and tracking_spec.loader is not None
tracking_module = importlib.util.module_from_spec(tracking_spec)
tracking_spec.loader.exec_module(tracking_module)
FileLogger = tracking_module.FileLogger


class LauncherTest(unittest.TestCase):
    def run_dry(self, **overrides: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env.update({"DRY_RUN": "1", **overrides})
        return subprocess.run(
            ["bash", str(MODULE_DIR / "run_opd_4b.sh")],
            cwd=REPO_ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )

    @staticmethod
    def run_fake_tokenizer_check(
        *,
        student_vocab: dict[str, int],
        teacher_vocab: dict[str, int],
        student_size: int,
        teacher_size: int,
        strategy: str,
    ) -> tuple[list[str], str]:
        class FakeTokenizer:
            bos_token_id = 151643
            eos_token_id = 151645
            pad_token_id = 151643

            def __init__(self, vocab: dict[str, int]) -> None:
                self.vocab = vocab

            def get_vocab(self) -> dict[str, int]:
                return dict(self.vocab)

        tokenizers = iter((FakeTokenizer(student_vocab), FakeTokenizer(teacher_vocab)))
        configs = iter(
            (
                SimpleNamespace(vocab_size=student_size, _commit_hash="student-rev"),
                SimpleNamespace(vocab_size=teacher_size, _commit_hash="teacher-rev"),
            )
        )
        fake_transformers = SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: next(tokenizers)),
            AutoConfig=SimpleNamespace(from_pretrained=lambda *args, **kwargs: next(configs)),
        )
        output = StringIO()
        with patch.dict(sys.modules, {"transformers": fake_transformers}), patch.object(
            preflight,
            "resolve_pretrained_reference",
            side_effect=lambda source, revision: (source, "f" * 40, False),
        ), redirect_stdout(output):
            errors = preflight.check_tokenizers(
                "student",
                "teacher",
                top_k_strategy=strategy,
            )
        return errors, output.getvalue()

    def test_smoke_paper_pair_dry_run(self) -> None:
        result = self.run_dry(PRESET="smoke", MODEL_PAIR="paper")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Qwen/Qwen3-1.7B-Base", result.stdout)
        self.assertIn("lllyx/Qwen3-4B-Base-GRPO", result.stdout)
        self.assertIn("data.train_batch_size=4", result.stdout)
        self.assertIn("actor_rollout_ref.rollout.n=1", result.stdout)
        self.assertIn("trainer.total_training_steps=2", result.stdout)
        self.assertIn("data.shuffle=false", result.stdout)
        self.assertIn("datasets/dapo-math-17k.parquet", result.stdout)
        self.assertIn("actor_rollout_ref.rollout.seed=42", result.stdout)
        self.assertIn("override_config.attn_implementation=sdpa", result.stdout)
        self.assertIn("actor_rollout_ref.model.use_remove_padding=false", result.stdout)
        self.assertIn("reward_model.model.attn_implementation=sdpa", result.stdout)
        self.assertIn("ea980cb0a6c2ae4b936e82123acc929f1cec04c1", result.stdout)
        self.assertIn("fsdp_config.model_dtype=fp32", result.stdout)
        self.assertIn("reward_model.model.dtype=bfloat16", result.stdout)
        self.assertIn("VLLM_USE_FLASHINFER_SAMPLER=0", result.stdout)
        self.assertNotIn("ray stop", result.stdout)

    def test_paper_preset_retains_reported_core_hyperparameters(self) -> None:
        result = self.run_dry(PRESET="paper", MODEL_PAIR="paper")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("data.train_batch_size=64", result.stdout)
        self.assertIn("actor_rollout_ref.actor.ppo_mini_batch_size=64", result.stdout)
        self.assertIn("actor_rollout_ref.rollout.n=4", result.stdout)
        self.assertIn("data.max_response_length=7168", result.stdout)
        self.assertIn("log_prob_top_k=16", result.stdout)
        self.assertNotIn("trainer.total_training_steps=", result.stdout)
        self.assertIn("trainer.max_actor_ckpt_to_keep=0", result.stdout)
        self.assertIn(
            "actor_rollout_ref.actor.checkpoint.save_contents=\\[\\'model\\'\\]",
            result.stdout,
        )
        self.assertIn("mode=model_only, retention=0 (0=unlimited)", result.stdout)

    def test_unlimited_checkpoint_retention_requires_model_only_mode(self) -> None:
        unsafe = self.run_dry(
            PRESET="paper",
            CHECKPOINT_SAVE_MODE="full",
            MAX_ACTOR_CKPTS_TO_KEEP="0",
        )
        self.assertEqual(unsafe.returncode, 2)
        self.assertIn("requires CHECKPOINT_SAVE_MODE=model_only", unsafe.stderr)
        bounded = self.run_dry(
            PRESET="paper",
            CHECKPOINT_SAVE_MODE="full",
            MAX_ACTOR_CKPTS_TO_KEEP="2",
        )
        self.assertEqual(bounded.returncode, 0, bounded.stderr)

    def test_scientific_resume_milestone_and_telemetry_interfaces(self) -> None:
        result = self.run_dry(
            PRESET="smoke",
            RESUME_MODE="auto",
            CHECKPOINT_SAVE_MODE="full",
            MAX_ACTOR_CKPTS_TO_KEEP="2",
            MILESTONE_STEPS="2",
            GPU_TELEMETRY_INTERVAL="1",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("trainer.resume_mode=auto", result.stdout)
        self.assertIn("resume/milestones  : auto / 2", result.stdout)
        self.assertIn("GPU telemetry      : every 1s", result.stdout)

    def test_resume_and_milestones_require_full_recovery_checkpoints(self) -> None:
        resume = self.run_dry(PRESET="paper", RESUME_MODE="auto")
        self.assertEqual(resume.returncode, 2)
        self.assertIn("requires CHECKPOINT_SAVE_MODE=full", resume.stderr)
        milestone = self.run_dry(PRESET="paper", MILESTONE_STEPS="20")
        self.assertEqual(milestone.returncode, 2)
        self.assertIn("requires full recovery checkpoints", milestone.stderr)

    def test_milestone_grid_must_be_saved_and_bounded(self) -> None:
        unsaved = self.run_dry(
            PRESET="smoke",
            CHECKPOINT_SAVE_MODE="full",
            MAX_ACTOR_CKPTS_TO_KEEP="2",
            MILESTONE_STEPS="1",
        )
        self.assertEqual(unsaved.returncode, 2)
        self.assertIn("not produced by SAVE_FREQ=2", unsaved.stderr)
        beyond = self.run_dry(
            PRESET="smoke",
            CHECKPOINT_SAVE_MODE="full",
            MAX_ACTOR_CKPTS_TO_KEEP="2",
            MILESTONE_STEPS="4",
        )
        self.assertEqual(beyond.returncode, 2)
        self.assertIn("exceeds TOTAL_TRAINING_STEPS=2", beyond.stderr)

    def test_data_and_rollout_seeds_are_explicit_and_overridable(self) -> None:
        result = self.run_dry(SEED="7", ROLLOUT_SEED="9")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("data.seed=7", result.stdout)
        self.assertIn("actor_rollout_ref.rollout.seed=9", result.stdout)

    def test_4b_pair_and_mismatch_thinking_override(self) -> None:
        four_b = self.run_dry(PRESET="pilot", MODEL_PAIR="4b")
        self.assertEqual(four_b.returncode, 0, four_b.stderr)
        self.assertIn("Qwen/Qwen3-4B-Base", four_b.stdout)
        mismatch = self.run_dry(PRESET="smoke", MODEL_PAIR="mismatch")
        self.assertEqual(mismatch.returncode, 0, mismatch.stderr)
        self.assertIn("reward_model.model.path=Qwen/Qwen3-4B", mismatch.stdout)
        self.assertNotIn("apply_chat_template_kwargs.enable_thinking=false", mismatch.stdout)

    def test_paper_ablation_model_pairs_are_pinned(self) -> None:
        success = self.run_dry(PRESET="smoke", MODEL_PAIR="r1_success")
        self.assertEqual(success.returncode, 0, success.stderr)
        self.assertIn("DeepSeek-R1-Distill-Qwen-1.5B", success.stdout)
        self.assertIn("hbx/JustRL-DeepSeek-1.5B", success.stdout)
        self.assertIn("0637e4096c789c67f9eecbe8355e0bdeddede1c2", success.stdout)
        failure = self.run_dry(PRESET="smoke", MODEL_PAIR="r1_failure")
        self.assertEqual(failure.returncode, 0, failure.stderr)
        self.assertIn("DeepSeek-R1-Distill-Qwen-7B", failure.stdout)

    def test_unknown_preset_fails(self) -> None:
        result = self.run_dry(PRESET="unknown")
        self.assertEqual(result.returncode, 2)

    def test_unknown_reward_weight_mode_fails_early(self) -> None:
        result = self.run_dry(REWARD_WEIGHT_MODE="typo")
        self.assertEqual(result.returncode, 2)
        self.assertIn("REWARD_WEIGHT_MODE", result.stderr)

    def test_unknown_support_normalization_fails_early(self) -> None:
        result = self.run_dry(SUPPORT_WEIGHT_NORMALIZATION="typo")
        self.assertEqual(result.returncode, 2)
        self.assertIn("SUPPORT_WEIGHT_NORMALIZATION", result.stderr)

    def test_cli_model_path_override_is_rejected_in_favor_of_pinned_env(self) -> None:
        env = {**os.environ, "DRY_RUN": "1"}
        result = subprocess.run(
            [
                "bash",
                str(MODULE_DIR / "run_opd_4b.sh"),
                "actor_rollout_ref.model.path=mutable/model",
            ],
            cwd=REPO_ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("STUDENT_MODEL/TEACHER_MODEL", result.stderr)

    def test_cli_cannot_bypass_validated_checkpoint_policy(self) -> None:
        env = {**os.environ, "DRY_RUN": "1", "PRESET": "paper"}
        for override in (
            "actor_rollout_ref.actor.checkpoint.save_contents=['model','optimizer']",
            "trainer.max_actor_ckpt_to_keep=2",
            "trainer.remove_previous_ckpt_in_save=true",
            "trainer.save_freq=-1",
            "trainer.resume_mode=auto",
            "++trainer.resume_mode=auto",
        ):
            with self.subTest(override=override):
                result = subprocess.run(
                    ["bash", str(MODULE_DIR / "run_opd_4b.sh"), override],
                    cwd=REPO_ROOT,
                    env=env,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn("validated environment interface", result.stderr)

    def test_disabling_snapshot_pinning_fails_closed(self) -> None:
        result = self.run_dry(PIN_MODEL_SNAPSHOTS="false")
        self.assertEqual(result.returncode, 2)
        self.assertIn("intentionally unsupported", result.stderr)
        self.assertIn("Hub revisions", result.stderr)

    def test_parse_gpu_csv(self) -> None:
        states = preflight.parse_gpu_csv(
            "0, NVIDIA A100-SXM4-80GB, 81920, 70000, 0\n"
            "1, NVIDIA A100-SXM4-80GB, 81920, 71000, 1\n"
        )
        self.assertEqual(len(states), 2)
        self.assertEqual(states[0].free_mib, 70000)
        self.assertEqual(states[1].utilization, 1)

    def test_cuda_visible_devices_is_fail_closed(self) -> None:
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
            self.assertEqual(preflight.visible_gpu_indices(), [])
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,0"}):
            with self.assertRaisesRegex(ValueError, "duplicate"):
                preflight.visible_gpu_indices()
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-deadbeef"}):
            with self.assertRaisesRegex(ValueError, "unique numeric"):
                preflight.visible_gpu_indices()

    def test_source_contract_and_both_dapo_prompt_variants(self) -> None:
        self.assertEqual(preflight.check_source_contract(REPO_ROOT), [])
        self.assertEqual(
            preflight.check_dataset(REPO_ROOT / "datasets" / "dapo-math-17k.parquet"), []
        )
        self.assertEqual(
            preflight.check_dataset(REPO_ROOT / "datasets" / "dapo-math-17k-processed.parquet"), []
        )

    def test_special_token_default_mismatch_is_warning_when_vocab_is_shared(self) -> None:
        class FakeTokenizer:
            bos_token_id = None
            pad_token_id = None

            def __init__(self, eos_token_id: int) -> None:
                self.eos_token_id = eos_token_id

            def get_vocab(self) -> dict[str, int]:
                return {"same": 0, "tokens": 1}

        tokenizers = iter((FakeTokenizer(10), FakeTokenizer(11)))
        configs = iter(
            (
                SimpleNamespace(vocab_size=16, _commit_hash="student-rev"),
                SimpleNamespace(vocab_size=16, _commit_hash="teacher-rev"),
            )
        )
        fake_transformers = SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: next(tokenizers)),
            AutoConfig=SimpleNamespace(from_pretrained=lambda *args, **kwargs: next(configs)),
        )
        output = StringIO()
        with patch.dict(sys.modules, {"transformers": fake_transformers}), patch.object(
            preflight,
            "resolve_pretrained_reference",
            side_effect=lambda source, revision: (source, "f" * 40, False),
        ), redirect_stdout(output):
            errors = preflight.check_tokenizers("student", "teacher")
        self.assertEqual(errors, [])
        self.assertIn("special-token defaults differ", output.getvalue())

    def test_figure5_padded_vocab_is_allowed_for_student_and_intersection_support(self) -> None:
        # Mirrors the measured Figure 5 metadata without allocating a 151K-entry fixture:
        # the two token->ID maps are identical and their real maximum ID is 151664,
        # while the model heads use different padding multiples.
        vocab = {"first": 0, "last-real-token": 151_664}
        for strategy in ("only_stu", "intersection"):
            with self.subTest(strategy=strategy):
                errors, output = self.run_fake_tokenizer_check(
                    student_vocab=vocab,
                    teacher_vocab=vocab,
                    student_size=151_936,
                    teacher_size=152_064,
                    strategy=strategy,
                )
                self.assertEqual(errors, [])
                self.assertIn("padded output vocab sizes differ", output)
                self.assertIn("student=151936, teacher=152064", output)
                self.assertIn("max token ID=151664", output)

    def test_padded_vocab_fails_closed_for_teacher_support_strategies(self) -> None:
        vocab = {"first": 0, "last-real-token": 151_664}
        for strategy in ("only_tch", "union", "union-intersection"):
            with self.subTest(strategy=strategy):
                errors, _ = self.run_fake_tokenizer_check(
                    student_vocab=vocab,
                    teacher_vocab=vocab,
                    student_size=151_936,
                    teacher_size=152_064,
                    strategy=strategy,
                )
                self.assertEqual(len(errors), 1)
                self.assertIn(f"TOP_K_STRATEGY={strategy}", errors[0])
                self.assertIn("unsafe without explicit filtering", errors[0])

    def test_student_support_fails_if_student_padded_head_is_larger(self) -> None:
        errors, _ = self.run_fake_tokenizer_check(
            student_vocab={"first": 0, "last-real-token": 7},
            teacher_vocab={"first": 0, "last-real-token": 7},
            student_size=16,
            teacher_size=8,
            strategy="only_stu",
        )
        self.assertEqual(len(errors), 1)
        self.assertIn("student-support ID can exceed the teacher output head", errors[0])

    def test_tokenizer_mapping_and_max_id_remain_hard_requirements(self) -> None:
        mapping_errors, _ = self.run_fake_tokenizer_check(
            student_vocab={"a": 0, "b": 1},
            teacher_vocab={"a": 0, "b": 2},
            student_size=8,
            teacher_size=8,
            strategy="only_stu",
        )
        self.assertTrue(any("token->ID mappings differ" in error for error in mapping_errors))

        max_id_errors, _ = self.run_fake_tokenizer_check(
            student_vocab={"a": 0, "b": 8},
            teacher_vocab={"a": 0, "b": 8},
            student_size=8,
            teacher_size=16,
            strategy="only_stu",
        )
        self.assertTrue(any("cannot represent tokenizer max ID 8" in error for error in max_id_errors))

    def test_launcher_passes_support_strategy_to_preflight(self) -> None:
        launcher = (MODULE_DIR / "run_opd_4b.sh").read_text(encoding="utf-8")
        self.assertIn('--top-k-strategy "${TOP_K_STRATEGY}"', launcher)
        parsed = preflight.build_parser().parse_args(["--top-k-strategy", "intersection"])
        self.assertEqual(parsed.top_k_strategy, "intersection")

    def test_pin_local_model_is_canonical_and_requires_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model = Path(temp_dir) / "model"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "model.safetensors").touch()
            record = pin_models.pin_one(str(model), "auto")
            self.assertEqual(record["snapshot_path"], str(model.resolve()))
            self.assertEqual(record["resolved_revision"], "local")
            self.assertFalse(record["cache_fallback"])

    def test_pin_rejects_incomplete_sharded_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model = Path(temp_dir) / "model"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "model-00001-of-00002.safetensors").touch()
            (model / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "a": "model-00001-of-00002.safetensors",
                            "b": "model-00002-of-00002.safetensors",
                        }
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "missing 1 indexed"):
                pin_models.pin_one(str(model), "auto")

    @staticmethod
    def _fake_snapshot(root: Path, revision: str) -> Path:
        snapshot = root / revision
        snapshot.mkdir()
        (snapshot / "config.json").write_text("{}", encoding="utf-8")
        (snapshot / "model.safetensors").touch()
        return snapshot

    def test_pin_fixed_revision_falls_back_to_exact_cache_on_network_error(self) -> None:
        revision = "a" * 40
        with tempfile.TemporaryDirectory() as directory:
            snapshot = self._fake_snapshot(Path(directory), revision)
            api = Mock()
            api.model_info.side_effect = requests.exceptions.SSLError("offline")
            download = Mock(return_value=str(snapshot))
            with patch.object(pin_models, "HfApi", return_value=api), patch.object(
                pin_models, "snapshot_download", download
            ):
                record = pin_models.pin_one("owner/model", revision)
        self.assertTrue(record["cache_fallback"])
        download.assert_called_once()
        self.assertTrue(download.call_args.kwargs["local_files_only"])
        self.assertEqual(download.call_args.kwargs["revision"], revision)

    def test_pin_fixed_revision_network_fallback_fails_if_exact_cache_is_missing(self) -> None:
        revision = "e" * 40
        api = Mock()
        api.model_info.side_effect = requests.exceptions.SSLError("offline")
        download = Mock(side_effect=FileNotFoundError("cache miss"))
        with patch.object(pin_models, "HfApi", return_value=api), patch.object(
            pin_models, "snapshot_download", download
        ):
            with self.assertRaisesRegex(RuntimeError, "not complete in the local cache"):
                pin_models.pin_one("owner/model", revision)
        download.assert_called_once()
        self.assertTrue(download.call_args.kwargs["local_files_only"])

    def test_pin_auto_or_http_404_never_falls_back_to_cache(self) -> None:
        download = Mock()
        api = Mock()
        api.model_info.side_effect = requests.exceptions.SSLError("offline")
        with patch.object(pin_models, "HfApi", return_value=api), patch.object(
            pin_models, "snapshot_download", download
        ):
            with self.assertRaises(requests.exceptions.SSLError):
                pin_models.pin_one("owner/model", "auto")
        download.assert_not_called()

        response = SimpleNamespace(status_code=404)
        not_found = requests.exceptions.HTTPError("not found", response=response)
        api.model_info.side_effect = not_found
        with patch.object(pin_models, "HfApi", return_value=api), patch.object(
            pin_models, "snapshot_download", download
        ):
            with self.assertRaises(requests.exceptions.HTTPError):
                pin_models.pin_one("owner/model", "b" * 40)
        download.assert_not_called()

    def test_pin_download_network_failure_retries_only_same_fixed_revision(self) -> None:
        revision = "c" * 40
        with tempfile.TemporaryDirectory() as directory:
            snapshot = self._fake_snapshot(Path(directory), revision)
            api = Mock()
            api.model_info.return_value = SimpleNamespace(sha=revision)
            download = Mock(
                side_effect=[requests.exceptions.SSLError("interrupted"), str(snapshot)]
            )
            with patch.object(pin_models, "HfApi", return_value=api), patch.object(
                pin_models, "snapshot_download", download
            ):
                record = pin_models.pin_one("owner/model", revision)
        self.assertTrue(record["cache_fallback"])
        self.assertEqual(download.call_count, 2)
        self.assertNotIn("local_files_only", download.call_args_list[0].kwargs)
        self.assertTrue(download.call_args_list[1].kwargs["local_files_only"])
        self.assertEqual(
            {call.kwargs["revision"] for call in download.call_args_list}, {revision}
        )

    def test_preflight_fixed_revision_metadata_cache_fallback_is_fail_closed(self) -> None:
        revision = "d" * 40
        expected = object()
        loader = SimpleNamespace(
            from_pretrained=Mock(
                side_effect=[requests.exceptions.SSLError("offline"), expected]
            )
        )
        actual = preflight.load_pretrained_metadata(loader, "owner/model", revision)
        self.assertIs(actual, expected)
        self.assertEqual(loader.from_pretrained.call_count, 2)
        self.assertNotIn("local_files_only", loader.from_pretrained.call_args_list[0].kwargs)
        self.assertTrue(loader.from_pretrained.call_args_list[1].kwargs["local_files_only"])
        self.assertEqual(
            {call.kwargs["revision"] for call in loader.from_pretrained.call_args_list},
            {revision},
        )

        mutable = SimpleNamespace(
            from_pretrained=Mock(side_effect=requests.exceptions.SSLError("offline"))
        )
        with self.assertRaises(requests.exceptions.SSLError):
            preflight.load_pretrained_metadata(mutable, "owner/model", "auto")
        mutable.from_pretrained.assert_called_once()

    def test_preflight_resolves_mutable_remote_revision_before_any_loader_cache(self) -> None:
        with patch.object(
            preflight,
            "resolve_hub_revision",
            side_effect=requests.exceptions.SSLError("offline"),
        ):
            with self.assertRaises(requests.exceptions.SSLError):
                preflight.resolve_pretrained_reference("owner/model", "auto")
        with patch.object(
            preflight,
            "resolve_hub_revision",
            return_value=("f" * 40, True),
        ):
            self.assertEqual(
                preflight.resolve_pretrained_reference("owner/model", "e" * 40),
                ("owner/model", "f" * 40, True),
            )

    def test_merge_dry_run_canonicalizes_relative_paths_before_chdir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            actor = root / "checkpoint" / "actor"
            actor.mkdir(parents=True)
            (actor / "fsdp_config.json").write_text("{}", encoding="utf-8")
            result = subprocess.run(
                ["bash", str(MODULE_DIR / "merge_checkpoint.sh"), "checkpoint", "merged"],
                cwd=root,
                env={**os.environ, "DRY_RUN": "1"},
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(str(actor.resolve()), result.stdout)
            self.assertIn(str((root / "merged").resolve()), result.stdout)

    def test_matrix_reuses_top16_success_reference_instead_of_repeating_it(self) -> None:
        result = subprocess.run(
            ["bash", str(MODULE_DIR / "run_matrix.sh")],
            cwd=REPO_ROOT,
            env={**os.environ, "ACTION": "print", "PRESET": "smoke"},
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count("### "), 6)
        self.assertIn("### success_grpo_teacher", result.stdout)
        self.assertNotIn("### topk_16", result.stdout)

    def test_file_logger_serializes_tensor_metrics_and_flushes(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "metrics.jsonl"
            with patch.dict(os.environ, {"VERL_FILE_LOGGER_PATH": str(target)}):
                logger = FileLogger("project", "experiment")
                logger.log({"scalar": torch.tensor(1.5), "vector": torch.tensor([1, 2])}, step=3)
                record = json.loads(target.read_text(encoding="utf-8"))
                logger.finish()
            self.assertEqual(record["step"], 3)
            self.assertEqual(record["data"]["scalar"], 1.5)
            self.assertEqual(record["data"]["vector"], [1, 2])

    def test_reward_entropy_accepts_noncontiguous_response_logits(self) -> None:
        import torch
        from verl.workers.fsdp_workers import RewardModelWorker

        full_logits = torch.randn(2, 5, 11)
        response_logits = full_logits[:, 1:4, :]
        self.assertFalse(response_logits.is_contiguous())

        actual = RewardModelWorker._compute_entropy_safe(
            object(), response_logits, chunk_size=2
        )
        expected = torch.distributions.Categorical(logits=response_logits).entropy()
        torch.testing.assert_close(actual, expected)

    def test_all_topk_strategies_keep_ids_and_replay_log_probs_aligned(self) -> None:
        import torch

        student_ids = torch.tensor([[[1, 2]]])
        teacher_ids = torch.tensor([[[2, 3]]])
        student_lp = torch.log(torch.tensor([[[0.6, 0.3]]]))
        teacher_on_student_lp = torch.log(torch.tensor([[[0.2, 0.4]]]))
        student_on_teacher_lp = torch.log(torch.tensor([[[0.3, 0.1]]]))
        teacher_lp = torch.log(torch.tensor([[[0.4, 0.5]]]))
        student_in_teacher = torch.tensor([[[False, True]]])
        teacher_in_student = torch.tensor([[[True, False]]])

        common = {
            "reward_weight_mode": "student_p",
            "support_weight_normalization": "selected",
            "student_ids": student_ids,
            "student_log_probs": student_lp,
            "teacher_on_student_log_probs": teacher_on_student_lp,
            "teacher_ids": teacher_ids,
            "teacher_log_probs": teacher_lp,
            "student_on_teacher_log_probs": student_on_teacher_lp,
            "student_in_teacher_mask": student_in_teacher,
            "teacher_in_student_mask": teacher_in_student,
        }
        outputs = {
            strategy: build_topk_distillation_tensors(strategy=strategy, **common)
            for strategy in ("only_stu", "only_tch", "intersection", "union", "union-intersection")
        }
        for strategy, result in outputs.items():
            self.assertTrue(torch.isfinite(result["rm_scores"]).all(), strategy)

        only_teacher = outputs["only_tch"]
        torch.testing.assert_close(only_teacher["union_top_k_ids"], teacher_ids)
        torch.testing.assert_close(only_teacher["union_top_k_log_probs"], student_on_teacher_lp)
        torch.testing.assert_close(
            only_teacher["student_log_probs_on_teacher_ids"], student_on_teacher_lp
        )
        self.assertEqual(
            validate_topk_replay_tensors(only_teacher),
            ("union_top_k_ids", "union_top_k_log_probs"),
        )

        symmetric = outputs["union-intersection"]
        student_union = torch.cat([student_lp, student_on_teacher_lp], dim=-1)
        teacher_union = torch.cat([teacher_on_student_lp, teacher_lp], dim=-1)
        valid = torch.tensor([[[True, False, False, True]]])
        weights = normalized_reward_weights(student_union, teacher_union, valid, "student_p")
        self.assertAlmostEqual(weights.sum().item(), 1.0)
        expected = torch.where(valid, -(student_union - teacher_union) * weights, 0.0)
        torch.testing.assert_close(symmetric["rm_scores"], expected)

        # Empty selected supports must be finite, and intersection must not need
        # the otherwise expensive student-on-teacher forward.
        empty_intersection = build_topk_distillation_tensors(
            strategy="intersection",
            reward_weight_mode="student_p",
            student_ids=student_ids,
            student_log_probs=student_lp,
            teacher_on_student_log_probs=teacher_on_student_lp,
            student_in_teacher_mask=torch.zeros_like(student_in_teacher),
        )
        self.assertTrue(torch.equal(empty_intersection["rm_scores"], torch.zeros_like(student_lp)))
        full_overlap = build_topk_distillation_tensors(
            strategy="union-intersection",
            **{**common, "student_in_teacher_mask": torch.ones_like(student_in_teacher),
               "teacher_in_student_mask": torch.ones_like(teacher_in_student)},
        )
        self.assertTrue(torch.equal(full_overlap["rm_scores"], torch.zeros_like(full_overlap["rm_scores"])))

    def test_nonoverlap_author_raw_mass_and_selected_normalization_are_distinct(self) -> None:
        import torch

        student_ids = torch.tensor([[[1, 2]]])
        teacher_ids = torch.tensor([[[2, 3]]])
        student_lp = torch.log(torch.tensor([[[0.60, 0.30]]]))
        teacher_on_student_lp = torch.log(torch.tensor([[[0.20, 0.40]]]))
        student_on_teacher_lp = torch.log(torch.tensor([[[0.30, 0.10]]]))
        teacher_lp = torch.log(torch.tensor([[[0.40, 0.50]]]))
        student_in_teacher = torch.tensor([[[False, True]]])
        teacher_in_student = torch.tensor([[[True, False]]])
        common = {
            "strategy": "union-intersection",
            "reward_weight_mode": "student_p",
            "student_ids": student_ids,
            "student_log_probs": student_lp,
            "teacher_on_student_log_probs": teacher_on_student_lp,
            "teacher_ids": teacher_ids,
            "teacher_log_probs": teacher_lp,
            "student_on_teacher_log_probs": student_on_teacher_lp,
            "student_in_teacher_mask": student_in_teacher,
            "teacher_in_student_mask": teacher_in_student,
        }
        author = build_topk_distillation_tensors(**common, support_weight_normalization="author")
        selected = build_topk_distillation_tensors(**common, support_weight_normalization="selected")
        valid = torch.tensor([[[True, False, False, True]]])
        student_union = torch.cat([student_lp, student_on_teacher_lp], dim=-1)
        teacher_union = torch.cat([teacher_on_student_lp, teacher_lp], dim=-1)
        raw = reward_weights(student_union, teacher_union, valid, "student_p", normalize=False)
        normalized = normalized_reward_weights(student_union, teacher_union, valid, "student_p")
        self.assertAlmostEqual(raw.sum().item(), 0.7)
        self.assertAlmostEqual(normalized.sum().item(), 1.0)
        torch.testing.assert_close(
            author["rm_scores"],
            torch.where(valid, -(student_union - teacher_union) * raw, 0.0),
        )
        torch.testing.assert_close(
            selected["rm_scores"],
            torch.where(valid, -(student_union - teacher_union) * normalized, 0.0),
        )

    def test_topk_replay_fails_closed_without_matching_old_log_probs(self) -> None:
        import torch

        ids = torch.ones(2, 3, 4, dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "union_top_k_log_probs"):
            validate_topk_replay_tensors({"union_top_k_ids": ids})
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            validate_topk_replay_tensors(
                {"union_top_k_ids": ids, "union_top_k_log_probs": torch.ones(2, 3, 2)}
            )

    def test_eq7_intersection_advantage_renormalizes_both_distributions(self) -> None:
        import torch

        student = torch.log(torch.tensor([[[0.6, 0.3, 0.1]]]))
        teacher = torch.log(torch.tensor([[[0.2, 0.6, 0.2]]]))
        overlap = torch.tensor([[[True, True, False]]])
        actual = eq7_intersection_advantage(student, teacher, overlap, torch.ones(1, 1))
        self.assertIsNotNone(actual)
        student_bar = torch.tensor([2 / 3, 1 / 3])
        teacher_bar = torch.tensor([1 / 4, 3 / 4])
        expected = (student_bar * (teacher_bar.log() - student_bar.log())).mean()
        torch.testing.assert_close(actual, expected)
        self.assertIsNone(eq7_intersection_advantage(student, teacher, overlap, torch.zeros(1, 1)))

    def test_eq8_entropy_gap_is_mean_absolute_state_gap_not_difference_of_means(self) -> None:
        import torch

        student = torch.tensor([[0.0, 2.0]])
        teacher = torch.tensor([[1.0, 1.0]])
        mask = torch.ones_like(student)
        actual = eq8_absolute_entropy_gap(student, teacher, mask)
        self.assertIsNotNone(actual)
        self.assertAlmostEqual(actual.item(), 1.0)
        self.assertAlmostEqual(abs(teacher.mean().item() - student.mean().item()), 0.0)

    def test_binned_position_entropy_keeps_counts_and_masks_empty_bins(self) -> None:
        import torch

        values = torch.tensor([[1.0, 3.0, 5.0, 7.0], [3.0, 5.0, 9.0, 11.0]])
        mask = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]])
        means, counts = binned_masked_mean(values, mask, bin_size=2)
        torch.testing.assert_close(means, torch.tensor([7.0 / 3.0, 0.0]))
        torch.testing.assert_close(counts, torch.tensor([3.0, 0.0]))

    def test_shell_syntax(self) -> None:
        for name in ("run_opd_4b.sh", "setup_env.sh", "merge_checkpoint.sh", "run_matrix.sh"):
            with self.subTest(name=name):
                result = subprocess.run(
                    ["bash", "-n", str(MODULE_DIR / name)], check=False, capture_output=True, text=True
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_launcher_archives_required_untracked_source_for_provenance(self) -> None:
        source = (MODULE_DIR / "run_opd_4b.sh").read_text(encoding="utf-8")
        self.assertIn('${REPO_ROOT}/verl/verl/utils/opd.py', source)
        self.assertIn('${RUN_DIR}/reproduction_code/verl/verl/utils/opd.py', source)
        self.assertNotIn("ls-files --others", source)

    def test_environment_setup_consumes_committed_constraints(self) -> None:
        result = subprocess.run(
            ["bash", str(MODULE_DIR / "setup_env.sh")],
            cwd=REPO_ROOT,
            env={**os.environ, "DRY_RUN": "1"},
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("constraints-2xa100-cu128.txt", result.stdout)
        self.assertIn("pip==26.2", result.stdout)


if __name__ == "__main__":
    unittest.main()
