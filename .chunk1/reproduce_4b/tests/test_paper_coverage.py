from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import audit_paper_coverage as coverage  # noqa: E402


class PaperCoverageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = MODULE_DIR.parent
        cls.matrix = coverage.read_object(MODULE_DIR / "paper_full_matrix.json")
        cls.ledger = coverage.read_object(MODULE_DIR / "paper_experiment_ledger.json")
        cls.cells = coverage.matrix_cells(cls.matrix)

    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self._metric_index = 0

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def completed_run(
        self,
        *,
        protocol: str,
        evaluations: list[dict[str, object]],
    ) -> dict[str, object]:
        """Return a completed synthetic run that satisfies every metric contract.

        Coverage tests should exercise protocol/evaluation state transitions, not
        accidentally fail the independent producer-artifact validation layer.
        """
        self._metric_index += 1
        metric_path = (
            Path(self._temporary.name) / f"metrics-{self._metric_index:02d}.jsonl"
        )
        bins = [1.0] * 60
        metric_path.write_text(
            json.dumps(
                {
                    "step": 2,
                    "data": {
                        "val-topk/overlap_ratio": 0.5,
                        "val-topk/adv_intersection": 0.1,
                        "opd/abs_entropy_gap": 0.2,
                        "val-topk/student_p_sum_intersection": 0.3,
                        "val-topk/teacher_p_sum_intersection": 0.4,
                        "actor/entropy": 0.6,
                        "actor/grad_norm": 0.7,
                        "actor/pg_loss": 0.8,
                        "opd/position_entropy_schema_version": 1,
                        "opd/figure19_metric_schema_version": 1,
                        "val-extrema/prob_diff_at_max_abs_adv_intersection": 0.9,
                        "position-entropy/student_mean_by_bin": bins,
                        "position-entropy/teacher_mean_by_bin": bins,
                        "position-entropy/token_count_by_bin": bins,
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "protocol": protocol,
            "state": "completed",
            "metrics_file": str(metric_path),
            "environment": {},
            "evaluations": evaluations,
        }

    def test_static_ledger_covers_every_figure_table_and_cell(self) -> None:
        entries = coverage.validate_ledger(self.ledger, self.cells)
        self.assertEqual(len(entries), 26)
        self.assertEqual(len(self.cells), 31)
        referenced = {
            cell_id for entry in entries for cell_id in entry["producer"]["cell_ids"]
        }
        self.assertEqual(referenced, set(self.cells))

    def test_smoke_completion_is_never_paper_comparable(self) -> None:
        entry = next(row for row in self.ledger["entries"] if row["id"] == "figure-07")
        suite_runs = {
            cell_id: [
                self.completed_run(
                    protocol="smoke",
                    evaluations=[{"n": 1, "paper_comparable": False}],
                )
            ]
            for cell_id in entry["producer"]["cell_ids"]
        }
        state, details = coverage.observed_state(entry, suite_runs, [], [])
        self.assertEqual(state, "smoke_training_complete")
        self.assertEqual(details["paper_trained_cell_count"], 0)
        self.assertEqual(details["paper_evaluated_cell_count"], 0)

    def test_training_matrix_has_all_unique_cells_and_explicit_budgets(self) -> None:
        rows = coverage.training_matrix_rows(self.cells, {})
        self.assertEqual(len(rows), 31)
        self.assertEqual(len({row["cell_id"] for row in rows}), 31)
        reverse = next(row for row in rows if row["cell_id"] == "fig5-reverse-r1-1p5b")
        self.assertEqual(reverse["paper_budget"], "600 steps")
        large = next(row for row in rows if row["cell_id"] == "fig20-r1-7b-r1-14b")
        self.assertTrue(large["disabled_by_default"])
        self.assertEqual(large["allowed_protocols"], ["paper"])
        self.assertEqual(large["local_2xa100_budget"], "not enabled on 2xA100")

    def test_hard_blocker_count_uses_declared_entries_not_partial_artifact_state(self) -> None:
        entries = coverage.validate_ledger(self.ledger, self.cells)
        rows = coverage.build_rows(entries, self.cells, {}, [], [])
        report = coverage.markdown_report(rows, len(self.cells))
        self.assertIn("当前硬阻塞条目：3/26", report)

    def test_avg16_checkpoint_evaluation_requires_all_cells(self) -> None:
        entry = next(row for row in self.ledger["entries"] if row["id"] == "figure-02")
        first, second = entry["producer"]["cell_ids"]
        suite_runs = {
            first: [
                self.completed_run(
                    protocol="paper",
                    evaluations=[{"n": 16, "paper_comparable": True}],
                )
            ],
            second: [
                self.completed_run(protocol="paper", evaluations=[])
            ],
        }
        state, details = coverage.observed_state(entry, suite_runs, [], [])
        self.assertEqual(state, "paper_trained_not_fully_evaluated")
        self.assertEqual(details["paper_evaluated_cell_count"], 1)

    def test_smoke_n16_short_summary_without_manifest_cannot_upgrade_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            summary_dir = run_dir / "evaluation/global_step_2"
            summary_dir.mkdir(parents=True)
            (summary_dir / "summary.json").write_text(
                json.dumps({
                    "checkpoint_step": 2,
                    "n": 16,
                    "benchmarks": {
                        "AIME24": {"avg@16": 0.0},
                        "AIME25": {"avg@16": 0.0},
                        "AMC23": {"avg@16": 0.0},
                    },
                }),
                encoding="utf-8",
            )
            evaluations = coverage._checkpoint_evaluations(run_dir, "smoke")
        self.assertEqual(len(evaluations), 1)
        self.assertFalse(evaluations[0]["paper_comparable"])
        self.assertIn("protocol", evaluations[0]["paper_comparability_reason"])

        entry = next(row for row in self.ledger["entries"] if row["id"] == "figure-13")
        cell_id = entry["producer"]["cell_ids"][0]
        state, details = coverage.observed_state(
            entry,
            {
                cell_id: [
                    self.completed_run(protocol="smoke", evaluations=evaluations)
                ]
            },
            [],
            [],
        )
        self.assertEqual(state, "position_entropy_training_only")
        self.assertEqual(details["paper_evaluated_cell_count"], 0)

    def test_suite_status_fingerprint_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {
                "protocol": "smoke",
                "cells": [{"cell_id": "cell-a", "fingerprint": "expected"}],
            }
            (root / "suite_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (root / "cell-a").mkdir()
            (root / "cell-a/status.json").write_text(
                json.dumps({"state": "completed", "fingerprint": "other"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                coverage.collect_suites([root])


if __name__ == "__main__":
    unittest.main()
