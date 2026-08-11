from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import run_scientific_evaluations as runner  # noqa: E402


class ScientificEvaluationRunnerTest(unittest.TestCase):
    def _cell(self, root: Path) -> Path:
        cell = root / "cell-a"
        run = cell / "attempt-0001"
        run.mkdir(parents=True)
        # Create fake milestone checkpoints so the step filter finds them.
        for step in (40, 80):
            (run / "checkpoints" / f"global_step_{step}" / "actor").mkdir(
                parents=True
            )
        (cell / "run_contract.json").write_text(
            json.dumps(
                {
                    "protocol": "scientific",
                    "evaluation": {
                        "trend": {
                            "n": 4,
                            "temperature": 0.7,
                            "top_p": 0.95,
                            "max_tokens": 4096,
                            "thinking": "off",
                            "seed": 42,
                        },
                        "trend_steps": [0, 40, 80],
                        "exact": {
                            "n": 16,
                            "temperature": 0.7,
                            "top_p": 0.95,
                            "max_tokens": 31744,
                            "thinking": "off",
                            "seed": 42,
                        },
                        "exact_steps": [80],
                    },
                }
            ),
            encoding="utf-8",
        )
        (cell / "status.json").write_text(
            json.dumps(
                {
                    "state": "completed",
                    "execution_state": "training_complete",
                    "run_dir": str(run.resolve()),
                }
            ),
            encoding="utf-8",
        )
        return cell

    def test_trend_command_uses_registered_positive_grid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cell = self._cell(Path(directory))
            command = runner.command_for_cell(
                MODULE_DIR.parent,
                cell,
                "trend",
                "plan",
                acknowledge_full_eval=False,
            )
        self.assertIn("trend-n4", command)
        self.assertEqual(command.count("--checkpoint-step"), 2)
        self.assertNotIn("0", command[-4:])

    def test_exact_run_requires_explicit_acknowledgement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cell = self._cell(Path(directory))
            with self.assertRaisesRegex(ValueError, "acknowledge"):
                runner.command_for_cell(
                    MODULE_DIR.parent,
                    cell,
                    "exact",
                    "run",
                    acknowledge_full_eval=False,
                )
            command = runner.command_for_cell(
                MODULE_DIR.parent,
                cell,
                "exact",
                "run",
                acknowledge_full_eval=True,
            )
        self.assertIn("exact-avg16", command)
        self.assertIn("--acknowledge-full-eval", command)


if __name__ == "__main__":
    unittest.main()
