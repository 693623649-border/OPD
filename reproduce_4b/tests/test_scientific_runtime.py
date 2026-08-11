from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import scientific_runtime as runtime  # noqa: E402


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def write_metrics(path: Path, first: int, last: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps({"step": step, "data": {"loss": 1.0 / step}}) + "\n"
            for step in range(first, last + 1)
        ),
        encoding="utf-8",
    )


def fake_checkpoint(root: Path, step: int, *, full: bool = True) -> Path:
    import torch

    checkpoint = root / f"global_step_{step}"
    actor = checkpoint / "actor"
    actor.mkdir(parents=True)
    write_json(actor / "fsdp_config.json", {"FSDP_version": 1, "world_size": 2})
    for rank in range(2):
        torch.save(
            {"weight": torch.tensor([rank], dtype=torch.float32)},
            actor / f"model_world_size_2_rank_{rank}.pt",
        )
        if full:
            (actor / f"optim_world_size_2_rank_{rank}.pt").write_bytes(f"optim-{rank}".encode())
            (actor / f"extra_state_world_size_2_rank_{rank}.pt").write_bytes(f"extra-{rank}".encode())
    write_json(actor / "huggingface" / "config.json", {"model_type": "fake"})
    if full:
        (checkpoint / "data.pt").write_bytes(b"dataloader")
    return checkpoint


def attempt_record(
    cell_root: Path,
    number: int,
    first: int,
    last: int,
    resume_step: int,
) -> Path:
    run_dir = cell_root / f"attempt-{number:04d}"
    metrics = run_dir / "metrics.jsonl"
    write_metrics(metrics, first, last)
    write_json(
        run_dir / runtime.ATTEMPT_RECORD,
        {
            "schema_version": runtime.SCHEMA_VERSION,
            "metrics_segment": str(metrics.resolve()),
            "resume": {
                "checkpoint_step": resume_step,
                "checkpoint_manifest_sha256": f"checkpoint-{resume_step}",
            },
        },
    )
    return run_dir


class ScientificRuntimeTest(unittest.TestCase):
    def test_parse_steps_is_sorted_unique_and_positive(self) -> None:
        self.assertEqual(runtime.parse_steps("40, 20  60"), (20, 40, 60))
        with self.assertRaisesRegex(ValueError, "unique"):
            runtime.parse_steps("20,20")
        with self.assertRaisesRegex(ValueError, "positive"):
            runtime.parse_steps("0")

    def test_prepare_auto_binds_complete_checkpoint_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoints = root / "checkpoints"
            fake_checkpoint(checkpoints, 20)
            # A newer torn checkpoint must never be selected for resume.
            fake_checkpoint(checkpoints, 40, full=False)
            (checkpoints / "latest_checkpointed_iteration.txt").write_text("20", encoding="utf-8")
            run_dir = root / "attempt-0002"
            run_dir.mkdir()
            record = runtime.prepare_attempt(
                run_dir,
                checkpoints,
                root / "milestones",
                run_dir / "metrics.jsonl",
                "auto",
                60,
                (20, 40, 60),
            )
            self.assertEqual(record["resume"]["checkpoint_step"], 20)
            self.assertEqual(len(record["resume"]["checkpoint_manifest_sha256"]), 64)
            self.assertEqual(len(record["resume"]["rejected_newer_checkpoints"]), 1)

    def test_disable_refuses_a_checkpoint_directory_with_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake_checkpoint(root / "checkpoints", 20)
            run_dir = root / "attempt-0001"
            run_dir.mkdir()
            with self.assertRaisesRegex(RuntimeError, "resume is disabled"):
                runtime.prepare_attempt(
                    run_dir,
                    root / "checkpoints",
                    root / "milestones",
                    run_dir / "metrics.jsonl",
                    "disable",
                    40,
                    (),
                )

    def test_milestone_is_model_only_atomic_and_hash_validated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = fake_checkpoint(root / "checkpoints", 20)
            destination = runtime.archive_milestone(source, root / "milestones")
            manifest = runtime.validate_milestone(destination, 20)
            self.assertEqual(manifest["kind"], "model_only")
            self.assertFalse(any(destination.glob("actor/optim_*.pt")))
            self.assertFalse((destination / "data.pt").exists())
            model = destination / "actor/model_world_size_2_rank_0.pt"
            model.write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "hash/size mismatch"):
                runtime.validate_milestone(destination, 20)

    def test_resume_canonicalization_abandons_rows_after_rollback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cell = Path(directory) / "cell"
            attempt_record(cell, 1, 1, 25, 0)
            attempt_record(cell, 2, 21, 40, 20)
            checkpoints = cell / "recovery"
            runtime.archive_milestone(fake_checkpoint(checkpoints, 20), cell / "milestones")
            runtime.archive_milestone(fake_checkpoint(checkpoints, 40), cell / "milestones")

            result = runtime.finalize_training(cell, 40, (20, 40))
            self.assertTrue(result["completion"], result["issues"])
            self.assertEqual(result["last_metric_step"], 40)
            self.assertEqual(len(result["metrics_sha256"]), 64)
            self.assertTrue(all(item["loadable"] for item in result["milestone_artifacts"]))
            rows = runtime.read_metric_segment(cell / "metrics.jsonl")
            self.assertEqual([row["step"] for row in rows], list(range(1, 41)))
            lineage = runtime.read_object(cell / "metrics_lineage.json")
            self.assertEqual(
                [row["step"] for row in lineage["abandoned_rows"]],
                [21, 22, 23, 24, 25],
            )

    def test_incomplete_and_nonfinite_metrics_cannot_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cell = Path(directory) / "cell"
            attempt_record(cell, 1, 1, 39, 0)
            result = runtime.finalize_training(cell, 40, ())
            self.assertFalse(result["completion"])
            self.assertIn("grid mismatch", result["issues"][0])

            metrics = cell / "attempt-0001/metrics.jsonl"
            metrics.write_text('{"step": 1, "data": {"loss": NaN}}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-finite"):
                runtime.canonicalize_metrics(cell)

    def test_sample_gpus_records_physical_values(self) -> None:
        result = SimpleNamespace(
            returncode=0,
            stdout=(
                "0, GPU-a, NVIDIA A100-SXM4-80GB, 81920, 62000, 19920, 99, 71, 350.5\n"
                "1, GPU-b, NVIDIA A100-SXM4-80GB, 81920, 63000, 18920, 100, 72, 355.5\n"
            ),
            stderr="",
        )
        with patch.object(runtime.shutil, "which", return_value="/usr/bin/nvidia-smi"), patch.object(
            runtime.subprocess, "run", return_value=result
        ) as invoked:
            sample = runtime.sample_gpus("0,1")
        self.assertEqual([gpu["physical_index"] for gpu in sample["gpus"]], [0, 1])
        self.assertEqual(sample["gpus"][1]["memory_used_mib"], 63000.0)
        self.assertIn("--id=0,1", invoked.call_args.args[0])

    def test_monitor_archives_milestone_before_cross_attempt_pruning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoints = root / "checkpoints"
            for step in (20, 40, 60):
                fake_checkpoint(checkpoints, step)
            (checkpoints / "latest_checkpointed_iteration.txt").write_text("60", encoding="utf-8")
            run_dir = root / "attempt-0002"
            run_dir.mkdir()
            stop = run_dir / "stop"
            stop.touch()
            result = runtime.monitor_runtime(
                run_dir,
                checkpoints,
                root / "milestones",
                (20, 60),
                recovery_retention=2,
                telemetry_interval=0,
                cuda_devices=None,
                stop_file=stop,
            )
            self.assertEqual(result["archived_milestones"], [20, 60])
            self.assertEqual(result["pruned_recovery_steps"], [20])
            self.assertFalse((checkpoints / "global_step_20").exists())
            runtime.validate_milestone(root / "milestones/global_step_20", 20)
            runtime.validate_milestone(root / "milestones/global_step_60", 60)

    def test_training_matrix_writes_json_csv_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            suite = Path(directory) / "suite"
            cell = suite / "length-15k"
            run_dir = attempt_record(cell, 1, 1, 3, 0)
            write_json(
                cell / "status.json",
                {
                    "cell_id": "length-15k",
                    "group_id": "length",
                    "execution_state": "running",
                    "conclusion_state": "not_assessed",
                    "run_dir": str(run_dir),
                    "attempt": 1,
                    "expected_final_step": 40,
                },
            )
            payload = runtime.render_training_matrix(suite)
            self.assertEqual(payload["cells"][0]["step"], 3)
            self.assertEqual(payload["cells"][0]["expected_final_step"], 40)
            for name in ("training_matrix.json", "training_matrix.csv", "training_matrix.md"):
                self.assertTrue((suite / name).is_file())
            self.assertIn("3/40", (suite / "training_matrix.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
