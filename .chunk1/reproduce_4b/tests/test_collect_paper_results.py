from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import collect_paper_results  # noqa: E402


class CollectPaperResultsTest(unittest.TestCase):
    def test_scalar_metrics_rejects_duplicate_steps(self) -> None:
        rows = [
            {"step": 1, "data": {"actor/grad_norm": 1.0}},
            {"step": 1, "data": {"actor/grad_norm": 2.0}},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate step"):
                list(collect_paper_results.scalar_metrics(path))

    def test_collect_suite_preserves_provenance_and_all_steps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "cell-a" / "attempt-0001"
            run.mkdir(parents=True)
            metrics = run / "metrics.jsonl"
            metrics.write_text(
                json.dumps({"step": 1, "data": {"actor/grad_norm": 1.5, "array": [1]}}) + "\n"
                + json.dumps({"step": 2, "data": {"actor/grad_norm": 1.0}}) + "\n",
                encoding="utf-8",
            )
            (root / "suite_manifest.json").write_text(
                json.dumps({
                    "suite_id": "suite", "protocol": "pilot", "seed": 42,
                    "source_tree_sha256": "source", "cells": [{
                        "cell_id": "cell-a", "group_id": "group", "label": "A",
                        "fidelity": "proxy", "fingerprint": "fingerprint",
                    }],
                }), encoding="utf-8"
            )
            (root / "cell-a" / "status.json").write_text(
                json.dumps({
                    "state": "completed", "fingerprint": "fingerprint", "attempt": 1,
                    "run_dir": str(run), "metrics_file": str(metrics),
                }), encoding="utf-8"
            )
            training, evaluation = collect_paper_results.collect_suite(root)
        self.assertEqual(len(training), 2)
        self.assertFalse(evaluation)
        self.assertEqual({row["step"] for row in training}, {1, 2})
        self.assertEqual({row["source_tree_sha256"] for row in training}, {"source"})

    def test_n16_short_checkpoint_summary_is_collected_with_false_and_reason(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "cell-a" / "attempt-0001"
            summary_dir = run / "evaluation/global_step_2"
            summary_dir.mkdir(parents=True)
            metrics = run / "metrics.jsonl"
            metrics.write_text(
                json.dumps({"step": 2, "data": {"actor/grad_norm": 1.0}}) + "\n",
                encoding="utf-8",
            )
            (summary_dir / "summary.json").write_text(json.dumps({
                "checkpoint_step": 2,
                "n": 16,
                "benchmarks": {
                    "AIME24": {"avg@16": 0.0},
                    "AIME25": {"avg@16": 0.0},
                    "AMC23": {"avg@16": 0.0},
                },
            }), encoding="utf-8")
            (root / "suite_manifest.json").write_text(json.dumps({
                "suite_id": "suite", "protocol": "smoke", "seed": 42,
                "source_tree_sha256": "source", "cells": [{
                    "cell_id": "cell-a", "group_id": "group", "label": "A",
                    "fidelity": "proxy", "fingerprint": "fingerprint",
                }],
            }), encoding="utf-8")
            (root / "cell-a" / "status.json").write_text(json.dumps({
                "state": "completed", "fingerprint": "fingerprint", "attempt": 1,
                "run_dir": str(run), "metrics_file": str(metrics),
            }), encoding="utf-8")
            _, evaluation = collect_paper_results.collect_suite(root)
        self.assertEqual(len(evaluation), 3)
        self.assertFalse(any(row["paper_comparable"] for row in evaluation))
        self.assertTrue(all("protocol" in row["paper_comparability_reason"] for row in evaluation))


if __name__ == "__main__":
    unittest.main()
