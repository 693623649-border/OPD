"""Runner wiring for the offline contract, interrupted resume, and the
three-axis execution state on scientific cells."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import pin_models  # noqa: E402
from run_ablations import (  # noqa: E402
    CellPlan,
    GPUAllocation,
    clean_environment,
    run_cells,
)


def make_plan(cell_id: str = "fig2-compatible-grpo", *, protocol: str = "scientific") -> CellPlan:
    return CellPlan(
        group_id="fig2_teacher_pattern",
        group_label="Fig 2",
        cell_id=cell_id,
        label="cell",
        fidelity="lite_hardware_adapted_not_paper_comparable",
        paper_location="Figure 2",
        env={
            "TOTAL_TRAINING_STEPS": "40",
            "MILESTONE_STEPS": "20,40",
            "STUDENT_MODEL": "/cache/models--s/snapshots/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "TEACHER_MODEL": "/cache/models--t/snapshots/bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "STUDENT_REVISION": "a" * 40,
            "TEACHER_REVISION": "b" * 40,
        },
        fingerprint="f" * 64,
        dataset={},
        protocol=protocol,
        scientific={
            "expected_final_step": 40,
            "milestone_steps": [20, 40],
            "phase_boundaries": [],
            "evaluation": {},
            "probes": {},
        },
        models={},
    )


def fake_checkpoint(root: Path, step: int) -> Path:
    checkpoint = root / f"global_step_{step}" / "actor"
    checkpoint.mkdir(parents=True)
    (checkpoint / "fsdp_config.json").write_text("{}", encoding="utf-8")
    return checkpoint


class OfflineContractTests(unittest.TestCase):
    def test_scientific_env_injects_offline_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = make_plan()
            env = clean_environment(
                plan, Path("/usr/bin/python3"), Path(tmp) / "attempt-0001"
            )
            self.assertEqual(env["HF_HUB_OFFLINE"], "1")
            self.assertEqual(env["TRANSFORMERS_OFFLINE"], "1")
            self.assertEqual(env["HF_DATASETS_OFFLINE"], "1")
            self.assertEqual(env["GPU_TELEMETRY_INTERVAL"], "1")

    def test_scientific_env_preserves_matrix_pinned_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = make_plan()
            plan = CellPlan(**{**plan.__dict__, "env": {**plan.env, "GPU_TELEMETRY_INTERVAL": "5"}})
            env = clean_environment(
                plan, Path("/usr/bin/python3"), Path(tmp) / "attempt-0001"
            )
            self.assertEqual(env["GPU_TELEMETRY_INTERVAL"], "5")

    def test_nonscientific_env_has_no_offline_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = make_plan(protocol="smoke")
            env = clean_environment(
                plan, Path("/usr/bin/python3"), Path(tmp) / "attempt-0001"
            )
            self.assertNotIn("HF_HUB_OFFLINE", env)
            self.assertNotIn("GPU_TELEMETRY_INTERVAL", env)

    def test_offline_resolve_skips_hub_network_call(self) -> None:
        revision = "a" * 40
        with patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}, clear=False):
            with patch("pin_models.HfApi") as hub:
                resolved, cache_only = pin_models.resolve_hub_revision(
                    "some/model", revision
                )
                self.assertEqual(resolved, revision)
                self.assertTrue(cache_only)
                hub.assert_not_called()

    def test_offline_resolve_rejects_non_immutable_revision(self) -> None:
        with patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}, clear=False):
            with self.assertRaisesRegex(RuntimeError, "offline contract"):
                pin_models.resolve_hub_revision("some/model", "auto")


class ResumeWiringTests(unittest.TestCase):
    def _run(
        self,
        plan: CellPlan,
        suite_root: Path,
        *,
        resume_interrupted: bool,
        returncode: int,
        previous: dict | None = None,
        previous_checkpoint: bool = False,
        write_checkpoint_before_failing: bool = False,
    ) -> tuple[int, dict[str, str]]:
        cell_root = suite_root / plan.cell_id
        cell_root.mkdir(parents=True, exist_ok=True)
        if previous is not None:
            previous_run = cell_root / "attempt-0001"
            previous_run.mkdir(parents=True, exist_ok=True)
            (cell_root / "status.json").write_text(
                json.dumps(previous), encoding="utf-8"
            )
            if previous_checkpoint:
                fake_checkpoint(previous_run / "checkpoints", 20)
        captured: dict[str, str] = {}

        def fake_run(command, env=None, check=False, **kwargs):  # noqa: ARG001
            captured.update(env or {})
            # Simulate a crashed training process that managed to commit a
            # recovery checkpoint inside its own run dir before dying.
            if write_checkpoint_before_failing and env is not None:
                fake_checkpoint(Path(env["RUN_DIR"]) / "checkpoints", 20)
            return SimpleNamespace(returncode=returncode)

        class FakeLocks:
            def close(self) -> None:
                pass

        with patch("run_ablations.acquire_gpu_lock", return_value=FakeLocks()):
            with patch("run_ablations.subprocess.run", side_effect=fake_run):
                with patch("run_ablations.last_metric_step", return_value=40):
                    result = run_cells(
                        [plan],
                        suite_root,
                        Path("/bin/true"),
                        Path("/usr/bin/python3"),
                        keep_going=False,
                        retry_failed=False,
                        allocation=GPUAllocation(count=2, devices=("0", "1")),
                        resume_interrupted=resume_interrupted,
                    )
        return result, captured

    def test_resume_interrupted_wires_auto_mode_and_checkpoint_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            suite_root = Path(tmp)
            plan = make_plan()
            previous_run = suite_root / plan.cell_id / "attempt-0001"
            previous = {
                "state": "failed",
                "run_dir": str(previous_run),
                "exit_code": 1,
                "fingerprint": plan.fingerprint,
            }
            result, env = self._run(
                plan,
                suite_root,
                resume_interrupted=True,
                returncode=0,
                previous=previous,
                previous_checkpoint=True,
            )
            self.assertEqual(result, 0)
            self.assertEqual(env["RESUME_MODE"], "auto")
            self.assertEqual(
                env["CHECKPOINT_DIR"], str(previous_run / "checkpoints")
            )
            self.assertEqual(env["TRAINING_MATRIX_ROOT"], str(suite_root))
            status = json.loads(
                (suite_root / plan.cell_id / "status.json").read_text(encoding="utf-8")
            )
            self.assertTrue(status["resumed"])
            self.assertEqual(status["execution_state"], "training_complete")
            self.assertEqual(status["conclusion_state"], "not_assessed")
            self.assertEqual(status["attempt"], 2)

    def test_matrix_pinned_auto_resume_is_preserved_on_fresh_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            suite_root = Path(tmp)
            plan = make_plan()
            plan = CellPlan(
                **{**plan.__dict__, "env": {**plan.env, "RESUME_MODE": "auto"}}
            )
            result, env = self._run(
                plan, suite_root, resume_interrupted=False, returncode=0
            )
            self.assertEqual(result, 0)
            # The scientific matrix pins RESUME_MODE=auto; the runner must not
            # override it for a fresh attempt (auto on an empty checkpoint dir
            # still starts from scratch).
            self.assertEqual(env["RESUME_MODE"], "auto")
            self.assertFalse(
                json.loads(
                    (suite_root / plan.cell_id / "status.json").read_text(
                        encoding="utf-8"
                    )
                )["resumed"]
            )

    def test_fresh_attempt_disables_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            suite_root = Path(tmp)
            plan = make_plan()
            result, env = self._run(
                plan, suite_root, resume_interrupted=True, returncode=0
            )
            self.assertEqual(result, 0)
            self.assertEqual(env["RESUME_MODE"], "disable")
            self.assertNotIn("CHECKPOINT_DIR", env)
            status = json.loads(
                (suite_root / plan.cell_id / "status.json").read_text(encoding="utf-8")
            )
            self.assertFalse(status["resumed"])
            self.assertEqual(status["execution_state"], "training_complete")

    def test_interruption_without_checkpoint_is_infrastructure_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            suite_root = Path(tmp)
            plan = make_plan()
            result, _ = self._run(
                plan, suite_root, resume_interrupted=False, returncode=1
            )
            self.assertEqual(result, 1)
            status = json.loads(
                (suite_root / plan.cell_id / "status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(status["execution_state"], "infrastructure_failed")

    def test_interruption_with_checkpoint_is_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            suite_root = Path(tmp)
            plan = make_plan()
            cell_root = suite_root / plan.cell_id
            cell_root.mkdir(parents=True)
            result, _ = self._run(
                plan,
                suite_root,
                resume_interrupted=False,
                returncode=1,
                write_checkpoint_before_failing=True,
            )
            self.assertEqual(result, 1)
            status = json.loads(
                (cell_root / "status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(status["execution_state"], "interrupted_resumable")

    def test_resume_flag_still_blocks_unrelated_failed_cell(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            suite_root = Path(tmp)
            plan = make_plan(protocol="smoke")
            previous_run = suite_root / plan.cell_id / "attempt-0001"
            previous = {
                "state": "failed",
                "run_dir": str(previous_run),
                "exit_code": 1,
                "fingerprint": plan.fingerprint,
            }
            cell_root = suite_root / plan.cell_id
            cell_root.mkdir(parents=True)
            (cell_root / "status.json").write_text(
                json.dumps(previous), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "resume-interrupted"):
                self._run(
                    plan,
                    suite_root,
                    resume_interrupted=True,
                    returncode=0,
                    previous=previous,
                )


if __name__ == "__main__":
    unittest.main()
