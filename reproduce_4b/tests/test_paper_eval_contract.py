from __future__ import annotations

import json
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from paper_eval_contract import (  # noqa: E402
    PAPER_BENCHMARKS,
    PAPER_BENCHMARK_SPECS,
    validate_paper_evaluation,
)


class PaperEvaluationContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = MODULE_DIR.parent

    def _write_contract(self, root: Path, *, limit: int | None, max_tokens: int) -> tuple[Path, Path, Path]:
        summary_path = root / "summary.json"
        manifest_path = root / "target_manifest.json"
        status_path = root / "status.json"
        sampling = {
            "n": 16,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.95,
            "thinking": "off",
            "seed": 42,
            "model": "fixture/model",
            "revision": "fixture-revision",
            "tokenizer": "fixture/tokenizer",
            "tokenizer_revision": "fixture-tokenizer-revision",
        }
        benchmarks = {}
        summary_benchmarks = {}
        for benchmark in PAPER_BENCHMARKS:
            spec = PAPER_BENCHMARK_SPECS[benchmark]
            dataset = self.repo_root / "datasets/test_data" / benchmark / "test.parquet"
            responses = root / benchmark / "responses.jsonl"
            responses.parent.mkdir(parents=True, exist_ok=True)
            with responses.open("w", encoding="utf-8") as handle:
                for row_index in range(spec["rows"]):
                    for rollout_id in range(16):
                        handle.write(json.dumps({
                            "example_id": f"{benchmark}-{row_index}",
                            "row_index": row_index,
                            "data_source": benchmark,
                            "rollout_id": rollout_id,
                            "response": "synthetic contract fixture",
                            "sampling": sampling,
                        }) + "\n")
            benchmarks[benchmark] = {
                "path": str(dataset),
                "rows": spec["rows"],
                "sha256": spec["sha256"],
            }
            summary_benchmarks[benchmark] = {
                "n": 16,
                "avg@16": 0.0,
                "num_prompts": spec["rows"],
                "num_complete_prompts": spec["rows"],
                "num_incomplete_prompts": 0,
                "num_responses": spec["rows"] * 16,
                "num_correct": 0,
                "input_jsonl": str(responses),
                "input_jsonl_sha256": self._sha256(responses),
                "grader": {
                    "path": str(self.repo_root / "scripts/val/eval/utils.py"),
                    "sha256": self._sha256(self.repo_root / "scripts/val/eval/utils.py"),
                },
            }
        manifest = {
            "schema_version": 1,
            "kind": "immutable_model_baseline",
            "paper_comparable": True,
            "limit": limit,
            "sampling": sampling,
            "benchmarks": benchmarks,
            "target_id": "fixture",
            "model": "fixture/model",
            "revision": "fixture-revision",
            "tokenizer": "fixture/tokenizer",
            "tokenizer_revision": "fixture-tokenizer-revision",
        }
        summary = {
            "schema_version": 1,
            "paper_comparable": True,
            "n": 16,
            "benchmarks": summary_benchmarks,
            "benchmark_macro_mean_avg_at_n": 0.0,
            "aggregation": "unweighted mean across AIME24, AIME25, AMC23",
            "target_id": "fixture",
            "model": "fixture/model",
            "revision": "fixture-revision",
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        status_path.write_text(json.dumps({
            "state": "completed", "summary": str(summary_path.resolve())
        }), encoding="utf-8")
        return summary_path, manifest_path, status_path

    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _write_checkpoint_contract(self, root: Path) -> tuple[Path, Path, Path, Path]:
        target_root = root / "evaluation" / "global_step_20"
        summary_path, manifest_path, _ = self._write_contract(
            target_root, limit=None, max_tokens=31_744
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        training = {
            "suite_id": "suite",
            "protocol": "paper",
            "seed": 42,
            "source_tree_sha256": "source",
            "cell_id": "cell-a",
            "group_id": "group-a",
            "fingerprint": "fingerprint-a",
            "attempt": 1,
            "run_dir": str(root),
        }
        source_checkpoint = root / "checkpoints" / "global_step_20"
        plan_path = root / "evaluation" / "evaluation_plan.json"
        plan = {
            "schema_version": 2,
            "kind": "ablation_checkpoint_grid",
            "training": training,
            "checkpoint_steps": [20],
            "selection": "explicit",
            "sampling": {
                key: manifest["sampling"][key]
                for key in (
                    "n", "temperature", "top_p", "max_tokens", "seed", "thinking",
                    "tokenizer", "tokenizer_revision",
                )
            },
            "limit": None,
            "paper_comparable": True,
            "benchmarks": manifest["benchmarks"],
            "targets": [{
                "target_id": "cell-a-global-step-20",
                "checkpoint_step": 20,
                "source_checkpoint": str(source_checkpoint),
                "target_manifest": str(manifest_path.resolve()),
                "summary": str(summary_path.resolve()),
            }],
        }
        plan_path.write_text(json.dumps(plan, sort_keys=True), encoding="utf-8")
        plan_sha = self._sha256(plan_path)
        merged_model = str(root / "merged" / "global_step_20")
        manifest.update({
            "schema_version": 2,
            "kind": "cell_checkpoint",
            "target_id": "cell-a-global-step-20",
            "checkpoint_step": 20,
            "source_checkpoint": str(source_checkpoint),
            "model": merged_model,
            "training": training,
            "evaluation_plan": {"path": str(plan_path.resolve()), "sha256": plan_sha},
        })
        manifest["sampling"]["model"] = merged_model
        summary.update({
            "schema_version": 2,
            "kind": "cell_checkpoint",
            "target_id": "cell-a-global-step-20",
            "checkpoint_step": 20,
            "model": merged_model,
            "evaluation_plan_sha256": plan_sha,
        })
        for benchmark in PAPER_BENCHMARKS:
            responses = target_root / benchmark / "responses.jsonl"
            rows = [json.loads(line) for line in responses.read_text(encoding="utf-8").splitlines()]
            for row in rows:
                row["sampling"]["model"] = merged_model
            responses.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            summary["benchmarks"][benchmark]["input_jsonl_sha256"] = self._sha256(responses)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        status_path = root / "evaluation_status.json"
        status_path.write_text(json.dumps({
            "schema_version": 2,
            "kind": "ablation_checkpoint_grid",
            "evaluation_plan": {"path": str(plan_path.resolve()), "sha256": plan_sha},
            "state": "completed",
            "summaries": [str(summary_path.resolve())],
            "targets": {
                "cell-a-global-step-20": {
                    "state": "completed",
                    "checkpoint_step": 20,
                    "summary": str(summary_path.resolve()),
                }
            },
        }), encoding="utf-8")
        return summary_path, manifest_path, status_path, plan_path

    def test_full_explicit_contract_is_paper_comparable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._write_contract(Path(directory), limit=None, max_tokens=31_744)
            result = validate_paper_evaluation(
                paths[0], protocol="paper", manifest_path=paths[1], status_path=paths[2]
            )
        self.assertTrue(result.paper_comparable, result.reason)

    def test_scientific_training_can_verify_only_the_evaluation_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary_path, manifest_path, status_path, plan_path = (
                self._write_checkpoint_contract(root)
            )
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            plan["training"]["protocol"] = "scientific"
            plan["training"]["comparability"] = {
                "training": "hardware_adapted_2xa100",
                "evaluation": "paper_evaluation_protocol",
                "provenance": "public_exact",
            }
            plan["paper_comparable"] = False
            plan["paper_evaluation_protocol"] = True
            plan_path.write_text(json.dumps(plan, sort_keys=True), encoding="utf-8")
            plan_sha = self._sha256(plan_path)

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["training"] = plan["training"]
            manifest["paper_comparable"] = False
            manifest["paper_evaluation_protocol"] = True
            manifest["comparability"] = plan["training"]["comparability"]
            manifest["evaluation_plan"]["sha256"] = plan_sha
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["paper_comparable"] = False
            summary["paper_evaluation_protocol"] = True
            summary["comparability"] = plan["training"]["comparability"]
            summary["evaluation_plan_sha256"] = plan_sha
            summary_path.write_text(json.dumps(summary), encoding="utf-8")

            status = json.loads(status_path.read_text(encoding="utf-8"))
            status["evaluation_plan"]["sha256"] = plan_sha
            status_path.write_text(json.dumps(status), encoding="utf-8")
            result = validate_paper_evaluation(
                summary_path,
                protocol="scientific",
                manifest_path=manifest_path,
                status_path=status_path,
            )
        self.assertFalse(result.paper_comparable)
        self.assertTrue(result.evaluation_protocol_comparable, result.reason)
        self.assertEqual(result.training_comparability, "hardware_adapted_2xa100")

    def test_n16_limit1_max256_is_never_paper_comparable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._write_contract(Path(directory), limit=1, max_tokens=256)
            result = validate_paper_evaluation(
                paths[0], protocol="paper", manifest_path=paths[1], status_path=paths[2]
            )
        self.assertFalse(result.paper_comparable)
        self.assertIn("limit", result.reason)

    def test_nonpaper_suite_fails_before_avg16_can_upgrade_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._write_contract(Path(directory), limit=None, max_tokens=31_744)
            result = validate_paper_evaluation(
                paths[0], protocol="smoke", manifest_path=paths[1], status_path=paths[2]
            )
        self.assertFalse(result.paper_comparable)
        self.assertIn("protocol", result.reason)

    def test_summary_without_manifest_is_insufficient(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._write_contract(Path(directory), limit=None, max_tokens=31_744)
            result = validate_paper_evaluation(
                paths[0], protocol="paper", manifest_path=None, status_path=paths[2]
            )
        self.assertFalse(result.paper_comparable)
        self.assertIn("manifest", result.reason)

    def test_complete_registered_checkpoint_grid_is_paper_comparable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._write_checkpoint_contract(Path(directory))
            result = validate_paper_evaluation(
                paths[0], protocol="paper", manifest_path=paths[1], status_path=paths[2]
            )
        self.assertTrue(result.paper_comparable, result.reason)

    def test_checkpoint_status_cannot_cherry_pick_registered_grid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary, manifest, status, plan_path = self._write_checkpoint_contract(root)
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            second_root = root / "evaluation" / "global_step_40"
            plan["checkpoint_steps"].append(40)
            plan["targets"].append({
                "target_id": "cell-a-global-step-40",
                "checkpoint_step": 40,
                "source_checkpoint": str(root / "checkpoints/global_step_40"),
                "target_manifest": str(second_root / "target_manifest.json"),
                "summary": str(second_root / "summary.json"),
            })
            plan_path.write_text(json.dumps(plan, sort_keys=True), encoding="utf-8")
            plan_sha = self._sha256(plan_path)
            manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
            manifest_payload["evaluation_plan"]["sha256"] = plan_sha
            manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
            summary_payload = json.loads(summary.read_text(encoding="utf-8"))
            summary_payload["evaluation_plan_sha256"] = plan_sha
            summary.write_text(json.dumps(summary_payload), encoding="utf-8")
            status_payload = json.loads(status.read_text(encoding="utf-8"))
            status_payload["evaluation_plan"]["sha256"] = plan_sha
            status.write_text(json.dumps(status_payload), encoding="utf-8")
            result = validate_paper_evaluation(
                summary, protocol="paper", manifest_path=manifest, status_path=status
            )
        self.assertFalse(result.paper_comparable)
        self.assertIn("grid", result.reason)

    def test_response_full_identity_mismatch_is_rejected(self) -> None:
        for field, wrong in (
            ("model", "wrong/model"),
            ("revision", "wrong-revision"),
            ("tokenizer", "wrong/tokenizer"),
            ("tokenizer_revision", "wrong-tokenizer-revision"),
            ("seed", 7),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                summary, manifest, status = self._write_contract(
                    root, limit=None, max_tokens=31_744
                )
                response = root / "AIME24/responses.jsonl"
                rows = [
                    json.loads(line)
                    for line in response.read_text(encoding="utf-8").splitlines()
                ]
                rows[0]["sampling"][field] = wrong
                response.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )
                summary_payload = json.loads(summary.read_text(encoding="utf-8"))
                summary_payload["benchmarks"]["AIME24"]["input_jsonl_sha256"] = self._sha256(response)
                summary.write_text(json.dumps(summary_payload), encoding="utf-8")
                result = validate_paper_evaluation(
                    summary, protocol="paper", manifest_path=manifest, status_path=status
                )
            self.assertFalse(result.paper_comparable)
            self.assertIn(f"sampling.{field}", result.reason)

    def test_avg_must_match_num_correct(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary, manifest, status = self._write_contract(
                root, limit=None, max_tokens=31_744
            )
            payload = json.loads(summary.read_text(encoding="utf-8"))
            payload["benchmarks"]["AIME24"]["avg@16"] = 0.5
            summary.write_text(json.dumps(payload), encoding="utf-8")
            result = validate_paper_evaluation(
                summary, protocol="paper", manifest_path=manifest, status_path=status
            )
        self.assertFalse(result.paper_comparable)
        self.assertIn("num_correct", result.reason)


if __name__ == "__main__":
    unittest.main()
