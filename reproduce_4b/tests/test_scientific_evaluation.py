from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import evaluate_ablation as evaluation  # noqa: E402


class ScientificEvaluationNamespaceTest(unittest.TestCase):
    def test_default_layout_remains_backward_compatible(self) -> None:
        run = Path("/tmp/example-run")
        self.assertEqual(evaluation.evaluation_root(run), run / "evaluation")
        self.assertEqual(
            evaluation.evaluation_status_path(run), run / "evaluation_status.json"
        )
        self.assertEqual(
            evaluation.checkpoint_target_id({"cell_id": "cell"}, 20),
            "cell-global-step-20",
        )

    def test_named_grids_have_disjoint_write_once_namespaces(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            trend = evaluation.evaluation_root(run, "trend-n4")
            exact = evaluation.evaluation_root(run, "exact-avg16")
            self.assertNotEqual(trend, exact)
            self.assertEqual(trend, run / "evaluation/trend-n4")
            self.assertEqual(
                evaluation.evaluation_status_path(run, "exact-avg16"),
                run / "evaluation/exact-avg16/evaluation_status.json",
            )
            self.assertEqual(
                evaluation.checkpoint_target_id(
                    {"cell_id": "cell"}, 200, "exact-avg16"
                ),
                "cell-global-step-200-exact-avg16",
            )

    def test_evaluation_id_rejects_paths_and_ambiguous_names(self) -> None:
        for invalid in ("", "../exact", "Exact", "trend_n4", "a" * 65):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                evaluation.evaluation_root(Path("/tmp/run"), invalid)

    def test_scientific_milestone_is_preferred_over_rolling_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cell = Path(directory) / "cell"
            run = cell / "attempt-0001"
            recovery = cell / "checkpoints/global_step_200/actor"
            milestone = cell / "milestones/global_step_200/actor"
            recovery.mkdir(parents=True)
            milestone.mkdir(parents=True)
            discovered = evaluation.discover_checkpoints(run, [200])
        self.assertEqual(discovered, [(200, milestone.parent)])


if __name__ == "__main__":
    unittest.main()
