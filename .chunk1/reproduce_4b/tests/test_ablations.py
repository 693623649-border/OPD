from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch


MODULE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = MODULE_DIR.parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import plot_position_entropy  # noqa: E402
import prepare_ablation_data  # noqa: E402
import run_ablations  # noqa: E402
import evaluate_ablation  # noqa: E402


class AblationRegistryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = run_ablations.load_registry(MODULE_DIR / "ablation_matrix.json", REPO_ROOT)
        cls.datasets = run_ablations.validate_datasets(cls.registry)

    def plans(self, groups, protocol: str = "smoke"):
        return run_ablations.build_plans(
            self.registry,
            groups,
            protocol,
            42,
            self.datasets,
            source_hash="test-source-hash",
            matrix_sha256="test-matrix-sha",
        )

    @staticmethod
    def suite_manifest(
        cells,
        *,
        protocol: str = "smoke",
        seed: int = 42,
        source: str = "source-tree-a",
        matrix: str = "matrix-a",
    ):
        return {
            "schema_version": 1,
            "suite_id": "opd-paper-ablations",
            "protocol": protocol,
            "seed": seed,
            "created_at": "2026-08-02T00:00:00+00:00",
            "source_tree_sha256": source,
            "matrix_sha256": matrix,
            "cells": cells,
        }

    @staticmethod
    def suite_cell(cell_id: str, fingerprint: str):
        return {
            "group_id": "unit-group",
            "cell_id": cell_id,
            "label": cell_id,
            "fingerprint": fingerprint,
        }

    def test_default_registry_covers_all_paper_ablation_levels(self) -> None:
        groups = run_ablations.select_groups(self.registry, set(), set(), include_extensions=False)
        plans = self.plans(groups)
        self.assertEqual(len(plans), 24)
        by_group = {}
        for plan in plans:
            by_group.setdefault(plan.group_id, []).append(plan)
        self.assertEqual(
            {plan.env["TOP_K"] for plan in by_group["fig15_16_topk"]},
            {"0", "1", "4", "16", "64"},
        )
        self.assertEqual(
            {plan.env["MAX_RESPONSE_LENGTH"] for plan in by_group["fig11_13_response_length"]},
            {"512", "1024", "3072", "7168", "10240", "15360"},
        )
        self.assertEqual(
            {plan.env["TOP_K_STRATEGY"] for plan in by_group["fig7_support"]},
            {"only_stu", "intersection", "union-intersection"},
        )

    def test_group_factors_are_exact_and_undeclared_changes_fail(self) -> None:
        group = next(group for group in self.registry["groups"] if group["id"] == "fig7_support")
        broken = json.loads(json.dumps(group))
        broken["cells"][0]["factors"]["TOP_K"] = "4"
        with self.assertRaisesRegex(ValueError, "expected exactly"):
            run_ablations.validate_group(broken)

    def test_fingerprint_changes_with_scientific_configuration(self) -> None:
        groups = run_ablations.select_groups(
            self.registry, {"fig7_support"}, {"fig7-student-topk"}, include_extensions=False
        )
        smoke = self.plans(groups, "smoke")[0]
        calibration = self.plans(groups, "calibration")[0]
        self.assertNotEqual(smoke.fingerprint, calibration.fingerprint)

    def test_fingerprint_and_manifest_bind_actual_matrix_sha(self) -> None:
        groups = run_ablations.select_groups(
            self.registry, {"fig7_support"}, {"fig7-student-topk"}, include_extensions=False
        )
        first = run_ablations.build_plans(
            self.registry,
            groups,
            "smoke",
            42,
            self.datasets,
            source_hash="same-source",
            matrix_sha256="matrix-a",
        )[0]
        second = run_ablations.build_plans(
            self.registry,
            groups,
            "smoke",
            42,
            self.datasets,
            source_hash="same-source",
            matrix_sha256="matrix-b",
        )[0]
        self.assertNotEqual(first.fingerprint, second.fingerprint)
        manifest = run_ablations.suite_payload(
            self.registry, "smoke", 42, "same-source", "matrix-a", [first]
        )
        self.assertEqual(manifest["matrix_sha256"], "matrix-a")

    def test_gpu_allocation_supports_uniform_two_or_eight_and_rejects_mixed(self) -> None:
        groups = run_ablations.select_groups(
            self.registry, {"fig7_support"}, {"fig7-student-topk"}, include_extensions=False
        )
        two = self.plans(groups)[0]
        eight = replace(two, cell_id="eight", env={**two.env, "N_GPUS": "8"})
        self.assertEqual(
            run_ablations.resolve_gpu_allocation([two], None),
            run_ablations.GPUAllocation(2, ("0", "1")),
        )
        self.assertEqual(
            run_ablations.resolve_gpu_allocation(
                [eight], "7,6,5,4,3,2,1,0"
            ),
            run_ablations.GPUAllocation(8, ("7", "6", "5", "4", "3", "2", "1", "0")),
        )
        with self.assertRaisesRegex(ValueError, "mixed GPU-count plan"):
            run_ablations.resolve_gpu_allocation([two, eight], None)
        with self.assertRaisesRegex(ValueError, "exactly 8"):
            run_ablations.resolve_gpu_allocation([eight], "0,1")

    def test_mixed_gpu_plan_fails_before_lock_write_status_or_subprocess(self) -> None:
        groups = run_ablations.select_groups(
            self.registry, {"fig7_support"}, {"fig7-student-topk"}, include_extensions=False
        )
        two = self.plans(groups)[0]
        eight = replace(two, cell_id="eight", env={**two.env, "N_GPUS": "8"})
        with tempfile.TemporaryDirectory() as directory, patch.object(
            run_ablations, "acquire_gpu_lock"
        ) as acquire, patch.object(run_ablations, "atomic_json") as write, patch.object(
            run_ablations.subprocess, "run"
        ) as launch:
            with self.assertRaisesRegex(ValueError, "mixed GPU-count plan"):
                run_ablations.run_cells(
                    [two, eight],
                    Path(directory),
                    MODULE_DIR / "run_opd_4b.sh",
                    REPO_ROOT / ".venv-opd/bin/python",
                    keep_going=False,
                    retry_failed=False,
                    allocation=run_ablations.GPUAllocation(2, ("0", "1")),
                )
            acquire.assert_not_called()
            write.assert_not_called()
            launch.assert_not_called()
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_main_rejects_mixed_gpu_plan_before_suite_manifest_write(self) -> None:
        groups = run_ablations.select_groups(
            self.registry, {"fig7_support"}, {"fig7-student-topk"}, include_extensions=False
        )
        two = self.plans(groups)[0]
        eight = replace(two, cell_id="eight", env={**two.env, "N_GPUS": "8"})
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory) / "never-created"
            with patch.object(run_ablations, "validate_datasets", return_value={}), patch.object(
                run_ablations, "source_tree_hash", return_value="source"
            ), patch.object(
                run_ablations, "build_plans", return_value=[two, eight]
            ), patch.object(
                run_ablations, "atomic_json"
            ) as write, patch.object(
                run_ablations.subprocess, "run"
            ) as launch:
                with self.assertRaises(SystemExit) as raised:
                    run_ablations.main(
                        [
                            "run",
                            "--matrix",
                            str(MODULE_DIR / "ablation_matrix.json"),
                            "--cell",
                            "fig7-student-topk",
                            "--run-root",
                            str(run_root),
                            "--yes",
                        ]
                    )
            self.assertEqual(raised.exception.code, 2)
            self.assertFalse(run_root.exists())
            write.assert_not_called()
            launch.assert_not_called()

    def test_per_card_locks_block_overlap_and_release_partial_acquisition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lock_dir = Path(directory)
            first = run_ablations.acquire_gpu_lock("0,1", lock_dir=lock_dir)
            try:
                with self.assertRaisesRegex(RuntimeError, "physical GPU 1"):
                    run_ablations.acquire_gpu_lock("1,2", lock_dir=lock_dir)
                disjoint = run_ablations.acquire_gpu_lock("2,3", lock_dir=lock_dir)
                disjoint.close()
            finally:
                first.close()

            held_one = run_ablations.acquire_gpu_lock("1", lock_dir=lock_dir)
            try:
                with self.assertRaisesRegex(RuntimeError, "physical GPU 1"):
                    run_ablations.acquire_gpu_lock("0,1", lock_dir=lock_dir)
                # GPU 0 was acquired first by the failed request and must have
                # been released during rollback.
                zero = run_ablations.acquire_gpu_lock("0", lock_dir=lock_dir)
                zero.close()
            finally:
                held_one.close()

    def test_clean_environment_does_not_inherit_scientific_overrides(self) -> None:
        groups = run_ablations.select_groups(
            self.registry, {"fig7_support"}, {"fig7-student-topk"}, include_extensions=False
        )
        plan = self.plans(groups)[0]
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {"TEACHER_MODEL": "ambient/poison", "TOP_K": "999"},
            clear=False,
        ):
            env = run_ablations.clean_environment(
                plan,
                REPO_ROOT / ".venv-opd/bin/python",
                Path(directory) / "run",
            )
        self.assertNotIn("TEACHER_MODEL", env)
        self.assertEqual(env["TOP_K"], "16")

    def test_python_path_is_made_absolute_without_escaping_virtualenv_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base-python"
            base.touch()
            venv_python = root / "venv-python"
            venv_python.symlink_to(base)
            absolute = run_ablations.absolute_path_preserve_symlinks(venv_python)
            self.assertEqual(absolute, venv_python)
            self.assertNotEqual(absolute, venv_python.resolve())

    def test_strict_paper_template_replaces_literal_double_braces(self) -> None:
        import pandas as pd

        frame = pd.DataFrame(
            {
                "prompt": [[{"role": "user", "content": r"Q Please \boxed{{}}"}]],
                "reward_model": [{"ground_truth": "1"}],
                "data_source": ["x"],
            }
        )
        strict = prepare_ablation_data.strict_paper_prompts(frame)
        content = strict.iloc[0]["prompt"][0]["content"]
        self.assertIn(r"\boxed{}", content)
        self.assertNotIn(r"\boxed{{}}", content)

    def test_position_entropy_reader_validates_schema_and_arrays(self) -> None:
        payload = {
            "step": 180,
            "data": {
                "opd/position_entropy_schema_version": 1,
                "position-entropy/bin_size": 256,
                "position-entropy/student_mean_by_bin": [1.0, 2.0],
                "position-entropy/teacher_mean_by_bin": [1.5, 2.5],
                "position-entropy/token_count_by_bin": [8, 0],
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            rows = plot_position_entropy.read_position_entropy(path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].step, 180)
        self.assertEqual(rows[0].student, (1.0, 2.0))

    def test_generation_validation_rejects_duplicate_rollout_ids(self) -> None:
        sampling = {
            "n": 2,
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 4096,
            "seed": 42,
            "thinking": "off",
            "model": "checkpoint",
        }
        records = [
            {"example_id": "problem-0", "rollout_id": 0, "sampling": sampling},
            {"example_id": "problem-0", "rollout_id": 0, "sampling": sampling},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "responses.jsonl"
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError, "unique rollout IDs|lack exactly rollout IDs"
            ):
                evaluate_ablation.validate_generation(path, 1, 2, sampling)

    def test_generation_validation_rejects_sampling_mismatch(self) -> None:
        expected = {
            "n": 2,
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 4096,
            "seed": 42,
            "thinking": "off",
            "model": "checkpoint",
        }
        records = [
            {"example_id": "problem-0", "rollout_id": 0, "sampling": expected},
            {
                "example_id": "problem-0",
                "rollout_id": 1,
                "sampling": {**expected, "top_p": 0.9},
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "responses.jsonl"
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "sampling top_p"):
                evaluate_ablation.validate_generation(path, 1, 2, expected)

    def test_aggregate_metrics_is_unweighted_three_benchmark_macro_mean(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for benchmark, score, prompts in (
                ("AIME24", 0.0, 1),
                ("AIME25", 0.0, 1_000),
                ("AMC23", 1.0, 1),
            ):
                path = root / benchmark / "metrics.json"
                path.parent.mkdir()
                path.write_text(
                    json.dumps({"avg@16": score, "num_prompts": prompts}),
                    encoding="utf-8",
                )
                paths.append(path)
            summary = evaluate_ablation.aggregate_metrics(paths, 16)

        self.assertAlmostEqual(summary["benchmark_macro_mean_avg_at_n"], 1.0 / 3.0)
        self.assertEqual(set(summary["benchmarks"]), {"AIME24", "AIME25", "AMC23"})
        self.assertEqual(
            summary["aggregation"],
            "unweighted mean across AIME24, AIME25, AMC23",
        )

    def test_aggregate_metrics_rejects_any_other_three_benchmark_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for benchmark in ("AIME24", "AIME25", "MATH500"):
                path = root / benchmark / "metrics.json"
                path.parent.mkdir()
                path.write_text(json.dumps({"avg@16": 0.5}), encoding="utf-8")
                paths.append(path)
            with self.assertRaisesRegex(ValueError, "benchmark"):
                evaluate_ablation.aggregate_metrics(paths, 16)

    def test_evaluation_plan_registers_explicit_sorted_checkpoint_grid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory).resolve()
            checkpoints = [
                (20, run_dir / "checkpoints/global_step_20"),
                (40, run_dir / "checkpoints/global_step_40"),
            ]
            training = {
                "suite_id": "suite",
                "protocol": "paper",
                "seed": 42,
                "source_tree_sha256": "source",
                "cell_id": "cell-a",
                "group_id": "group-a",
                "fingerprint": "fingerprint-a",
                "attempt": 1,
                "run_dir": str(run_dir),
            }
            plan = evaluate_ablation.build_evaluation_plan(
                REPO_ROOT,
                run_dir,
                checkpoints,
                training,
                "org/tokenizer",
                "tokenizer-revision",
                16,
                31_744,
                42,
                None,
            )
        self.assertEqual(plan["selection"], "explicit")
        self.assertEqual(plan["checkpoint_steps"], [20, 40])
        self.assertTrue(plan["paper_comparable"])
        self.assertEqual(
            [target["checkpoint_step"] for target in plan["targets"]], [20, 40]
        )

    def test_write_once_json_is_idempotent_and_rejects_changed_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation_plan.json"
            payload = {"schema_version": 2, "checkpoint_steps": [20, 40]}
            self.assertTrue(evaluate_ablation.write_once_json(path, payload, "plan"))
            original = path.read_bytes()
            self.assertFalse(evaluate_ablation.write_once_json(path, payload, "plan"))
            self.assertEqual(path.read_bytes(), original)
            with self.assertRaisesRegex(ValueError, "differs"):
                evaluate_ablation.write_once_json(
                    path,
                    {"schema_version": 2, "checkpoint_steps": [40]},
                    "plan",
                )
            self.assertEqual(path.read_bytes(), original)

    def test_target_manifest_uses_local_revision_null_and_plan_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory).resolve()
            checkpoint = run_dir / "checkpoints/global_step_20"
            training = {
                "suite_id": "suite",
                "protocol": "paper",
                "seed": 42,
                "source_tree_sha256": "source",
                "cell_id": "cell-a",
                "group_id": "group-a",
                "fingerprint": "fingerprint-a",
                "attempt": 1,
                "run_dir": str(run_dir),
            }
            plan = evaluate_ablation.build_evaluation_plan(
                REPO_ROOT,
                run_dir,
                [(20, checkpoint)],
                training,
                "org/tokenizer",
                "tokenizer-revision",
                16,
                31_744,
                42,
                None,
            )
            plan_path = run_dir / "evaluation/evaluation_plan.json"
            evaluate_ablation.write_once_json(plan_path, plan, "plan")
            plan_hash = run_ablations.sha256_file(plan_path)
            manifest = evaluate_ablation.build_target_manifest(
                run_dir,
                20,
                checkpoint,
                training,
                plan_path,
                plan_hash,
                plan,
                "org/tokenizer",
                "tokenizer-revision",
            )
        self.assertEqual(manifest["kind"], "cell_checkpoint")
        self.assertIsNone(manifest["revision"])
        self.assertIsNone(manifest["sampling"]["revision"])
        self.assertEqual(manifest["tokenizer_revision"], "tokenizer-revision")
        self.assertEqual(manifest["evaluation_plan"]["sha256"], plan_hash)

    def test_evaluation_status_resume_preserves_registered_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            summary = root / "evaluation/global_step_20/summary.json"
            plan = {
                "targets": [{"summary": str(summary)}],
            }
            plan_path = root / "evaluation/evaluation_plan.json"
            evaluate_ablation.write_once_json(plan_path, plan, "plan")
            plan_hash = run_ablations.sha256_file(plan_path)
            status_path = root / "evaluation_status.json"
            status = evaluate_ablation.load_or_initialize_status(
                status_path, plan_path, plan_hash, plan
            )
            status["summaries"] = [str(summary)]
            status["targets"] = {
                "cell-a-global-step-20": {
                    "state": "completed",
                    "summary": str(summary),
                }
            }
            run_ablations.atomic_json(status_path, status)
            resumed = evaluate_ablation.load_or_initialize_status(
                status_path, plan_path, plan_hash, plan
            )
        self.assertEqual(resumed["summaries"], [str(summary)])
        self.assertIn("cell-a-global-step-20", resumed["targets"])

    def test_paper_main_rejects_implicit_latest_before_registering_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            suite_root = Path(directory).resolve() / "suite/paper/seed-42"
            cell_root = suite_root / "cell-a"
            run_dir = cell_root / "attempt-0001"
            run_dir.mkdir(parents=True)
            (suite_root / "suite_manifest.json").write_text(
                json.dumps({
                    "suite_id": "suite",
                    "protocol": "paper",
                    "seed": 42,
                    "source_tree_sha256": "source",
                    "matrix_sha256": "matrix",
                    "cells": [{"cell_id": "cell-a", "fingerprint": "fingerprint"}],
                }),
                encoding="utf-8",
            )
            (cell_root / "status.json").write_text(
                json.dumps({
                    "state": "completed",
                    "cell_id": "cell-a",
                    "group_id": "group-a",
                    "fingerprint": "fingerprint",
                    "attempt": 1,
                    "run_dir": str(run_dir),
                }),
                encoding="utf-8",
            )
            with self.assertRaises(SystemExit) as raised:
                evaluate_ablation.main(["plan", "--run-dir", str(run_dir)])
            self.assertEqual(raised.exception.code, 2)
            self.assertFalse((run_dir / "evaluation").exists())

    def test_suite_manifest_merge_appends_cells_with_matching_identity(self) -> None:
        existing = self.suite_manifest([self.suite_cell("cell-a", "fingerprint-a")])
        current = self.suite_manifest([self.suite_cell("cell-b", "fingerprint-b")])

        merged = run_ablations.merge_suite_manifest(existing, current)

        self.assertEqual([cell["cell_id"] for cell in merged["cells"]], ["cell-a", "cell-b"])
        self.assertEqual(merged["protocol"], "smoke")
        self.assertEqual(merged["seed"], 42)
        self.assertEqual(merged["source_tree_sha256"], "source-tree-a")

    def test_suite_manifest_merge_is_idempotent_for_same_cell_fingerprint(self) -> None:
        cell = self.suite_cell("cell-a", "fingerprint-a")
        existing = self.suite_manifest([cell])
        current = self.suite_manifest([dict(cell)])

        merged = run_ablations.merge_suite_manifest(existing, current)

        self.assertEqual(merged["cells"], [cell])

    def test_suite_manifest_merge_rejects_cell_fingerprint_conflict(self) -> None:
        existing = self.suite_manifest([self.suite_cell("cell-a", "fingerprint-a")])
        current = self.suite_manifest([self.suite_cell("cell-a", "fingerprint-b")])

        with self.assertRaisesRegex(RuntimeError, "fingerprint"):
            run_ablations.merge_suite_manifest(existing, current)

    def test_suite_manifest_merge_rejects_protocol_seed_or_source_change(self) -> None:
        existing = self.suite_manifest([self.suite_cell("cell-a", "fingerprint-a")])
        cases = (
            ("protocol", self.suite_manifest([], protocol="formal")),
            ("seed", self.suite_manifest([], seed=7)),
            ("source_tree_sha256", self.suite_manifest([], source="source-tree-b")),
            ("matrix_sha256", self.suite_manifest([], matrix="matrix-b")),
        )
        for field, current in cases:
            with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, field):
                run_ablations.merge_suite_manifest(existing, current)


if __name__ == "__main__":
    unittest.main()
