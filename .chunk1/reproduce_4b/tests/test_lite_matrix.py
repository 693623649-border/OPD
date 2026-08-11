from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = MODULE_DIR.parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import evaluate_ablation  # noqa: E402
import run_ablations  # noqa: E402
import verify_lite_calibration  # noqa: E402


class LiteMatrixTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.matrix_path = MODULE_DIR / "lite_matrix.json"
        cls.full_matrix_path = MODULE_DIR / "paper_full_matrix.json"
        cls.registry = run_ablations.load_registry(cls.matrix_path, REPO_ROOT)
        cls.full_registry = run_ablations.load_registry(cls.full_matrix_path, REPO_ROOT)
        cls.datasets = run_ablations.validate_datasets(cls.registry)
        cls.groups = {group["id"]: group for group in cls.registry["groups"]}
        cls.full_groups = {group["id"]: group for group in cls.full_registry["groups"]}

    def plans(self, groups, protocol: str):
        return run_ablations.build_plans(
            self.registry,
            groups,
            protocol,
            seed=42,
            dataset_by_path=self.datasets,
            source_hash="lite-matrix-test-source",
            matrix_sha256=run_ablations.sha256_file(self.matrix_path),
        )

    def test_identity_tiers_and_unique_cells(self) -> None:
        self.assertEqual(self.registry["suite_id"], "rethinking-opd-lite-2xa100-v1")
        self.assertNotEqual(self.registry["suite_id"], self.full_registry["suite_id"])
        for group in self.registry["groups"]:
            run_ablations.validate_group(group)

        default_groups = run_ablations.select_groups(
            self.registry, set(), set(), include_extensions=False
        )
        extended_groups = run_ablations.select_groups(
            self.registry, set(), set(), include_extensions=True
        )
        default_plans = self.plans(default_groups, "pilot")
        extended_plans = self.plans(extended_groups, "pilot")
        self.assertEqual(len(default_plans), 7)
        self.assertEqual(len(extended_plans), 14)
        self.assertEqual(len({plan.cell_id for plan in extended_plans}), 14)
        self.assertEqual(
            {group["id"] for group in self.registry["groups"] if group.get("disabled_by_default")},
            {"fig4_6_deepseek_teachers", "fig8_cold_start", "fig11_13_response_length"},
        )

    def test_source_matrix_is_hash_pinned(self) -> None:
        expected = hashlib.sha256(self.full_matrix_path.read_bytes()).hexdigest()
        self.assertEqual(self.registry["paper"]["source_matrix_sha256"], expected)

    def test_pilot_budget_is_bounded_and_not_paper_comparable(self) -> None:
        groups = run_ablations.select_groups(
            self.registry, set(), set(), include_extensions=True
        )
        plans = self.plans(groups, "pilot")
        self.assertNotIn("paper", self.registry["protocols"])
        for plan in plans:
            self.assertEqual(plan.env["N_GPUS"], "2")
            self.assertEqual(plan.env["TOTAL_TRAINING_STEPS"], "40")
            self.assertEqual(plan.env["N_RESPONSES"], "1")
            self.assertEqual(plan.env["SAVE_FREQ"], "40")
            self.assertEqual(plan.env["MAX_ACTOR_CKPTS_TO_KEEP"], "1")
            self.assertEqual(plan.env["CHECKPOINT_SAVE_MODE"], "model_only")
            self.assertGreaterEqual(int(plan.env["MIN_FREE_GIB"]), 70)
            self.assertNotIn("SKIP_PREFLIGHT", plan.env)
            self.assertNotIn("ALLOW_UNPINNED_MODELS", plan.env)
            self.assertIn("not_paper_comparable", plan.fidelity)
            self.assertNotIn("UNPUBLISHED", " ".join(plan.env.values()))
            self.assertIn(
                plan.env["STUDENT_MODEL"],
                {
                    "Qwen/Qwen3-1.7B-Base",
                    "lllyx/Qwen3-1.7B-SFT",
                    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                },
            )
            self.assertTrue(
                evaluate_ablation.is_paper_comparable(
                    protocol="pilot", n=16, max_tokens=31_744, limit=None
                )
                is False
            )

        length = [plan for plan in plans if plan.group_id == "fig11_13_response_length"]
        ordinary = [plan for plan in plans if plan.group_id != "fig11_13_response_length"]
        self.assertEqual({plan.env["MAX_RESPONSE_LENGTH"] for plan in length}, {"1024", "3072", "7168"})
        self.assertEqual({plan.env["TRAIN_BATCH_SIZE"] for plan in length}, {"2"})
        self.assertEqual({plan.env["MAX_RESPONSE_LENGTH"] for plan in ordinary}, {"2048"})
        self.assertEqual({plan.env["TRAIN_BATCH_SIZE"] for plan in ordinary}, {"4"})

    def test_scientific_identity_matches_the_pinned_full_matrix(self) -> None:
        scientific_keys = {
            "MODEL_PAIR",
            "STUDENT_MODEL",
            "STUDENT_REVISION",
            "TEACHER_MODEL",
            "TEACHER_REVISION",
            "ENABLE_THINKING",
            "TRAIN_DATA",
            "TOP_K",
            "TOP_K_STRATEGY",
            "SUPPORT_WEIGHT_NORMALIZATION",
            "MAX_RESPONSE_LENGTH",
        }
        for group_id, lite_group in self.groups.items():
            full_group = self.full_groups[group_id]
            full_cells = {cell["id"]: cell for cell in full_group["cells"]}
            for lite_cell in lite_group["cells"]:
                full_cell = full_cells[lite_cell["id"]]
                lite_env = {**lite_group["constants"], **lite_cell["factors"]}
                full_env = {**full_group["constants"], **full_cell["factors"]}
                for key in scientific_keys & (set(lite_env) | set(full_env)):
                    self.assertEqual(lite_env.get(key), full_env.get(key), (lite_cell["id"], key))

    def test_wrapper_is_fixed_to_lite_matrix_and_requires_yes(self) -> None:
        wrapper = MODULE_DIR / "run_lite.sh"
        syntax = subprocess.run(["bash", "-n", str(wrapper)], capture_output=True, text=True)
        self.assertEqual(syntax.returncode, 0, syntax.stderr)

        plan = subprocess.run(
            ["bash", str(wrapper), "plan", "core", "pilot"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(plan.returncode, 0, plan.stderr)
        self.assertIn("cells=7", plan.stdout)
        self.assertIn("rethinking-opd-lite-2xa100-v1/pilot", plan.stdout)

        override = subprocess.run(
            [
                "bash",
                str(wrapper),
                "plan",
                "core",
                "pilot",
                "--matrix",
                str(MODULE_DIR / "ablation_matrix.json"),
                "--protocol",
                "smoke",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(override.returncode, 2)
        self.assertIn("is reserved by run_lite.sh", override.stderr)

        core_cell = subprocess.run(
            [
                "bash",
                str(wrapper),
                "plan",
                "core",
                "pilot",
                "--cell",
                "fig7-student-topk",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(core_cell.returncode, 0, core_cell.stderr)
        self.assertIn("cells=1", core_cell.stdout)

        extended_cell = subprocess.run(
            [
                "bash",
                str(wrapper),
                "plan",
                "core",
                "pilot",
                "--cell=fig12-length-7168",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(extended_cell.returncode, 2)
        self.assertIn("belongs to the extended tier", extended_cell.stderr)

        for abbreviated in (
            "--include-e",
            "--cel=fig12-length-7168",
            "--gro=fig11_13_response_length",
        ):
            bypass = subprocess.run(
                ["bash", str(wrapper), "plan", "core", "pilot", abbreviated],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(bypass.returncode, 2, abbreviated)
            self.assertIn("unsupported or abbreviated", bypass.stderr)

        with tempfile.TemporaryDirectory() as directory:
            denied = subprocess.run(
                [
                    "bash",
                    str(wrapper),
                    "run",
                    "core",
                    "smoke",
                    "--run-root",
                    directory,
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(denied.returncode, 0)
            self.assertIn("requires --yes", denied.stderr)
            self.assertFalse(any(Path(directory).iterdir()))

        with tempfile.TemporaryDirectory() as directory:
            gated = subprocess.run(
                [
                    "bash",
                    str(wrapper),
                    "run",
                    "extended",
                    "pilot",
                    "--run-root",
                    directory,
                    "--yes",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(gated.returncode, 2)
            self.assertIn("calibration suite manifest is missing", gated.stderr)
            self.assertFalse(any(Path(directory).iterdir()))

    def test_extended_pilot_requires_fresh_finite_calibrations(self) -> None:
        source_hash = run_ablations.source_tree_hash(REPO_ROOT)
        matrix_sha = run_ablations.sha256_file(self.matrix_path)
        groups = run_ablations.select_groups(
            self.registry,
            set(),
            set(verify_lite_calibration.REQUIRED_CELLS),
            include_extensions=True,
        )
        plans = run_ablations.build_plans(
            self.registry,
            groups,
            "calibration",
            seed=42,
            dataset_by_path=self.datasets,
            source_hash=source_hash,
            matrix_sha256=matrix_sha,
        )

        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory)
            suite_root = (
                run_root
                / self.registry["suite_id"]
                / "calibration"
                / "seed-42"
            )
            manifest = {
                "suite_id": self.registry["suite_id"],
                "protocol": "calibration",
                "seed": 42,
                "matrix_sha256": matrix_sha,
                "source_tree_sha256": source_hash,
                "cells": [
                    {"cell_id": plan.cell_id, "fingerprint": plan.fingerprint}
                    for plan in plans
                ],
            }
            suite_root.mkdir(parents=True)
            (suite_root / "suite_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            def metric_row(step: int) -> dict[str, object]:
                return {
                    "step": step,
                    "data": {
                        "opd/metric_schema_version": 2,
                        "val-topk/overlap_ratio": 0.75,
                        "actor/grad_norm": 2.0,
                        "perf/max_memory_reserved_gb": 55.0,
                        "response_length/mean": 2048.0,
                        "response_length/clip_ratio": 1.0,
                        "response/aborted_ratio": 0.0,
                        "timing_s/step": 20.0,
                        "training/global_step": step,
                    },
                }

            def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
                path.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )

            for plan in plans:
                cell_root = suite_root / plan.cell_id
                run_dir = cell_root / "attempt-0001"
                run_dir.mkdir(parents=True)
                metrics = run_dir / "metrics.jsonl"
                write_rows(metrics, [metric_row(step) for step in range(1, 11)])
                status = {
                    "cell_id": plan.cell_id,
                    "fingerprint": plan.fingerprint,
                    "state": "completed",
                    "exit_code": 0,
                    "last_metric_step": 10,
                    "metrics_file": str(metrics),
                }
                (cell_root / "status.json").write_text(
                    json.dumps(status), encoding="utf-8"
                )

            self.assertEqual(
                set(
                    verify_lite_calibration.verify_required_calibrations(
                        self.matrix_path,
                        REPO_ROOT,
                        run_root,
                        42,
                    )
                ),
                set(verify_lite_calibration.REQUIRED_CELLS),
            )

            bad_metrics = (
                suite_root
                / verify_lite_calibration.REQUIRED_CELLS[0]
                / "attempt-0001"
                / "metrics.jsonl"
            )
            write_rows(bad_metrics, [metric_row(10)])
            with self.assertRaisesRegex(RuntimeError, "metric steps"):
                verify_lite_calibration.verify_required_calibrations(
                    self.matrix_path,
                    REPO_ROOT,
                    run_root,
                    42,
                )

            valid_rows = [metric_row(step) for step in range(1, 11)]
            del valid_rows[4]["data"]["actor/grad_norm"]  # type: ignore[index]
            write_rows(bad_metrics, valid_rows)
            with self.assertRaisesRegex(RuntimeError, "missing metrics"):
                verify_lite_calibration.verify_required_calibrations(
                    self.matrix_path,
                    REPO_ROOT,
                    run_root,
                    42,
                )

            valid_rows = [metric_row(step) for step in range(1, 11)]
            valid_rows[4]["data"]["actor/grad_norm"] = float("nan")  # type: ignore[index]
            write_rows(bad_metrics, valid_rows)
            with self.assertRaisesRegex(RuntimeError, "non-finite"):
                verify_lite_calibration.verify_required_calibrations(
                    self.matrix_path,
                    REPO_ROOT,
                    run_root,
                    42,
                )


if __name__ == "__main__":
    unittest.main()
