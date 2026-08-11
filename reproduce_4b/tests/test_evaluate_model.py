from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = MODULE_DIR.parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import evaluate_model  # noqa: E402


class EvaluateModelTest(unittest.TestCase):
    def test_sampling_contract_is_the_paper_protocol(self) -> None:
        contract = evaluate_model.sampling_contract(
            "org/model", "commit", "org/tokenizer", "tok-commit", 16, 31_744, 42
        )
        self.assertEqual(contract["temperature"], 0.7)
        self.assertEqual(contract["top_p"], 0.95)
        self.assertEqual(contract["thinking"], "off")
        self.assertEqual(contract["n"], 16)
        self.assertEqual(contract["max_tokens"], 31_744)

    def test_target_id_cannot_escape_output_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "target-id"):
                evaluate_model.target_root(Path(directory), "../escape")

    def test_existing_manifest_must_match_exactly(self) -> None:
        expected = {"schema_version": 1, "target_id": "teacher"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps({**expected, "target_id": "other"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs"):
                evaluate_model.validate_existing_manifest(path, expected)

    def test_limit_downgrades_paper_comparability(self) -> None:
        manifest = evaluate_model.build_manifest(
            REPO_ROOT, "smoke", "org/model", "rev", "org/model", "rev", 16, 31_744, 42, 1
        )
        self.assertFalse(manifest["paper_comparable"])
        self.assertEqual(set(manifest["benchmarks"]), {"AIME24", "AIME25", "AMC23"})


if __name__ == "__main__":
    unittest.main()
