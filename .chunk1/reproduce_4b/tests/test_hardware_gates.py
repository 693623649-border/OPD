from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "reproduce_4b"))

import hardware_gates  # noqa: E402


class HardwareGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_materialize_applies_one_profile_to_the_whole_matrix(self) -> None:
        source = REPO_ROOT / "reproduce_4b/scientific_gate_matrix.json"
        output = self.root / "optimizer.json"
        payload = hardware_gates.materialize_profile(source, "optimizer_offload", output)
        overrides = payload["protocols"]["scientific"]["overrides"]
        self.assertEqual(overrides["ACTOR_OPTIMIZER_OFFLOAD"], "true")
        self.assertEqual(overrides["ACTOR_PARAMETER_OFFLOAD"], "false")
        self.assertEqual(overrides["ROLLOUT_GPU_MEMORY_UTILIZATION"], "0.30")
        self.assertIn("optimizer-offload", payload["suite_id"])
        with self.assertRaises(RuntimeError):
            hardware_gates.materialize_profile(source, "parameter_offload", output)

    def _write_cell(
        self,
        matrix: dict,
        suite: Path,
        cell_id: str,
        expected_step: int,
        *,
        peak_mib: float = 77 * 1024,
        aborted: float = 0.0,
        fatal: str = "",
    ) -> None:
        cell = suite / cell_id
        run_dir = cell / "attempt-0001"
        run_dir.mkdir(parents=True)
        fingerprint = "a" * 64
        matrix_path = self.root / "matrix.json"
        matrix_sha = hardware_gates.sha256_file(matrix_path)
        contract = {
            "suite_id": matrix["suite_id"],
            "cell_id": cell_id,
            "fingerprint": fingerprint,
            "matrix_sha256": matrix_sha,
            "training": {
                "environment": hardware_gates.PROFILES["base"],
            },
        }
        status = {
            "cell_id": cell_id,
            "fingerprint": fingerprint,
            "execution_state": "training_complete",
            "run_dir": str(run_dir.resolve()),
        }
        for name, payload in (("run_contract.json", contract), ("status.json", status)):
            (cell / name).write_text(json.dumps(payload), encoding="utf-8")
        rows = [
            {
                "step": step,
                "data": {"response/aborted_ratio": aborted, "training/global_step": step},
            }
            for step in range(1, expected_step + 1)
        ]
        (cell / "metrics.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        (run_dir / "preflight.log").write_text(
            "[gpu] 0: A100, free=79.0/80.0 GiB, util=0%\n"
            "[gpu] 1: A100, free=79.0/80.0 GiB, util=0%\n",
            encoding="utf-8",
        )
        (run_dir / "train.log").write_text(fatal, encoding="utf-8")
        telemetry = {
            "peak_memory_used_mib_by_gpu": {"0": peak_mib, "1": peak_mib},
        }
        (run_dir / "gpu_telemetry_summary.json").write_text(
            json.dumps(telemetry), encoding="utf-8"
        )

    def _minimal_matrix(self) -> dict:
        matrix = {
            "schema_version": 2,
            "suite_id": "gate-suite",
            "scientific_spec": {
                "engineering_gate_only": True,
                "runnable_training_cells": 1,
                "gate_acceptance": {
                    "minimum_free_gib_per_card": 75,
                    "maximum_physical_peak_gib_per_card": 78,
                    "maximum_aborted_ratio": 0.01,
                },
                "gate_profile": {
                    "id": "base",
                    "environment": hardware_gates.PROFILES["base"],
                },
            },
            "groups": [
                {
                    "id": "long_context_gate",
                    "scientific": {"expected_final_step": 2, "milestone_steps": [2]},
                    "cells": [{"id": "gate-length-15k"}],
                }
            ],
        }
        (self.root / "matrix.json").write_text(json.dumps(matrix), encoding="utf-8")
        return matrix

    def test_assessment_passes_only_full_physical_gate(self) -> None:
        matrix = self._minimal_matrix()
        suite = self.root / "suite"
        self._write_cell(matrix, suite, "gate-length-15k", 2)
        report = hardware_gates.assess_suite(self.root / "matrix.json", suite)
        self.assertTrue(report["all_passed"])
        self.assertFalse(report["scientific_evidence"])

    def test_oom_peak_and_aborted_ratio_fail_with_group_fallback(self) -> None:
        matrix = self._minimal_matrix()
        suite = self.root / "suite"
        self._write_cell(
            matrix,
            suite,
            "gate-length-15k",
            2,
            peak_mib=79 * 1024,
            aborted=0.02,
            fatal="torch.OutOfMemoryError: CUDA out of memory",
        )
        report = hardware_gates.assess_suite(self.root / "matrix.json", suite)
        self.assertFalse(report["all_passed"])
        self.assertEqual(report["groups"][0]["next_group_uniform_profile"], "optimizer_offload")
        issues = " ".join(report["cells"][0]["issues"])
        self.assertIn("aborted ratio", issues)
        self.assertIn("NCCL", issues)

    def test_environment_variable_names_are_not_nccl_fatal_matches(self) -> None:
        matrix = self._minimal_matrix()
        suite = self.root / "suite"
        self._write_cell(
            matrix,
            suite,
            "gate-length-15k",
            2,
            fatal="NCCL_TIMEOUT=600 TORCH_NCCL_BLOCKING_WAIT=1 nccl_timeout=120",
        )
        report = hardware_gates.assess_suite(self.root / "matrix.json", suite)
        self.assertTrue(report["all_passed"])

    def test_real_nccl_failure_still_detected(self) -> None:
        matrix = self._minimal_matrix()
        suite = self.root / "suite"
        self._write_cell(
            matrix,
            suite,
            "gate-length-15k",
            2,
            fatal="[Rank 1] NCCL error: unhandled CudaError",
        )
        report = hardware_gates.assess_suite(self.root / "matrix.json", suite)
        self.assertFalse(report["all_passed"])
        issues = " ".join(report["cells"][0]["issues"])
        self.assertIn("NCCL", issues)


if __name__ == "__main__":
    unittest.main()
