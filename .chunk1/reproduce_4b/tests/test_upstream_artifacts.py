from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import audit_paper_coverage as coverage  # noqa: E402
import collect_paper_results  # noqa: E402
from upstream_artifacts import collect_upstream_roots, validate_upstream_stage  # noqa: E402


def canonical_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class UpstreamArtifactTest(unittest.TestCase):
    def _stage(
        self, root: Path, stage: str, *, protocol: str = "smoke", state: str = "completed"
    ) -> tuple[Path, Path]:
        run_dir = root / stage / "attempt-0001"
        run_dir.mkdir(parents=True)
        environment = {"SEED": "42", "PRESET": protocol}
        identity = {
            "schema_version": 1,
            "suite_id": "rethinking-opd-upstream-v1",
            "stage": stage,
            "protocol": protocol,
            "seed": 42,
            "fidelity": f"engineering-{protocol}; 2xA100 adaptation; not a paper result",
            "launcher": ["fixture"],
            "environment": environment,
            "sources": [{"path": "fixture", "sha256": "a", "bytes": 1}],
            "datasets": [{"path": "fixture", "exists": True, "sha256": "b"}],
            "models": [{"role": "fixture", "model_id": "fixture", "revision": "c"}],
        }
        fingerprint = canonical_hash(identity)
        scope = "Table 1" if stage == "grpo-teacher" else "Table 3"
        manifest = {
            "schema_version": 1,
            "suite_id": "rethinking-opd-upstream-v1",
            "stage": stage,
            "protocol": protocol,
            "attempt": 1,
            "run_dir": str(run_dir.resolve()),
            "fingerprint": fingerprint,
            "fidelity": identity["fidelity"],
            "paper_scope": scope,
            "source": {"files": identity["sources"]},
            "data": identity["datasets"],
            "models": identity["models"],
            "environment": environment,
            "launcher": identity["launcher"],
        }
        manifest_path = run_dir / "upstream_manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        marker = run_dir / "output/_SUCCESS"
        marker.parent.mkdir()
        marker.write_text("ok", encoding="utf-8")
        status = {
            "schema_version": 1,
            "stage": stage,
            "protocol": protocol,
            "fingerprint": fingerprint,
            "state": state,
            "attempt": 1,
            "run_dir": str(run_dir.resolve()),
            "manifest": str(manifest_path.resolve()),
        }
        if state == "completed":
            status.update({"exit_code": 0, "success_marker": str(marker.resolve())})
        status_path = root / stage / "status.json"
        status_path.write_text(json.dumps(status), encoding="utf-8")
        return status_path, manifest_path

    def test_completed_smoke_is_valid_but_not_paper_comparable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._stage(root, "grpo-teacher")
            row = validate_upstream_stage(root, "grpo-teacher")
        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row["state"], "completed")
        self.assertFalse(row["paper_comparable"])
        self.assertIn("engineering", row["paper_comparability_reason"])

    def test_fingerprint_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, manifest_path = self._stage(root, "grpo-teacher")
            manifest = json.loads(manifest_path.read_text())
            manifest["environment"]["SEED"] = "7"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                validate_upstream_stage(root, "grpo-teacher")

    def test_status_manifest_protocol_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            status_path, _ = self._stage(root, "grpo-teacher")
            status = json.loads(status_path.read_text())
            status["protocol"] = "paper"
            status_path.write_text(json.dumps(status), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "protocol"):
                validate_upstream_stage(root, "grpo-teacher")

    def test_table1_and_table3_dynamic_states_require_their_stages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._stage(root, "grpo-teacher")
            self._stage(root, "cold-start-rollout")
            runs = collect_upstream_roots([root])
            matrix = coverage.read_object(MODULE_DIR / "paper_full_matrix.json")
            ledger = coverage.read_object(MODULE_DIR / "paper_experiment_ledger.json")
            cells = coverage.matrix_cells(matrix)
            entries = coverage.validate_ledger(ledger, cells)
            table1 = next(row for row in entries if row["id"] == "table-01")
            table3 = next(row for row in entries if row["id"] == "table-03")
            state1, details1 = coverage.observed_state(table1, {}, [], [], runs)
            state3, details3 = coverage.observed_state(table3, {}, [], [], runs)
        self.assertEqual(state1, "upstream_smoke_complete")
        self.assertEqual(details1["upstream_completed_stages"], ["grpo-teacher"])
        self.assertEqual(state3, "upstream_smoke_partial")
        self.assertEqual(details3["upstream_completed_stages"], ["cold-start-rollout"])

    def test_collector_emits_independent_upstream_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._stage(root, "grpo-teacher")
            rows = collect_paper_results.collect_upstream_roots([root])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["paper_scope"], "Table 1")
        self.assertFalse(rows[0]["paper_comparable"])

    def test_component_model_smoke_does_not_upgrade_absent_table3_pipeline(self) -> None:
        matrix = coverage.read_object(MODULE_DIR / "paper_full_matrix.json")
        ledger = coverage.read_object(MODULE_DIR / "paper_experiment_ledger.json")
        cells = coverage.matrix_cells(matrix)
        entries = coverage.validate_ledger(ledger, cells)
        table3 = next(row for row in entries if row["id"] == "table-03")
        model_eval = {
            "target_id": "teacher-smoke",
            "model_identity": "Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c",
            "state": "completed",
            "paper_comparable": False,
            "paper_comparability_reason": "smoke",
        }
        state, details = coverage.observed_state(table3, {}, [model_eval], [], [])
        self.assertEqual(state, "upstream_not_started")
        self.assertEqual(details["completed_model_eval_count"], 1)


if __name__ == "__main__":
    unittest.main()
