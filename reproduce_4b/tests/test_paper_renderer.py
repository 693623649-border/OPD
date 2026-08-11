from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import render_paper_figures as renderer  # noqa: E402


class PaperRendererTest(unittest.TestCase):
    def _write_suite(self, root: Path, cell_ids: list[str], with_n1_eval: bool) -> Path:
        cells = []
        for index, cell_id in enumerate(cell_ids):
            fingerprint = f"fingerprint-{cell_id}"
            cells.append(
                {
                    "cell_id": cell_id,
                    "group_id": "fig7_support",
                    "label": f"Condition {index + 1}",
                    "fidelity": "synthetic smoke fidelity",
                    "fingerprint": fingerprint,
                    "environment": {
                        "PRESET": "smoke",
                        "TOTAL_TRAINING_STEPS": "2",
                    },
                }
            )
            run_dir = root / cell_id / "attempt-0001"
            run_dir.mkdir(parents=True)
            metrics_path = run_dir / "metrics.jsonl"
            rows = [
                {
                    "step": step,
                    "data": {
                        "opd/metric_schema_version": 2,
                        "val-topk/overlap_ratio": 0.60 + 0.02 * index + 0.01 * step,
                        "val-topk/adv_intersection": -0.02 + 0.002 * step,
                        "actor/entropy": 0.70 - 0.01 * step,
                        "teacher/entropy": 0.75,
                        "opd/abs_entropy_gap": 0.05 + 0.01 * step,
                        "val-topk/student_p_sum_intersection": 0.97,
                        "val-topk/teacher_p_sum_intersection": 0.98,
                        "actor/grad_norm": 1.0 + index,
                        "response_length/mean": 128.0 + step,
                    },
                }
                for step in (1, 2)
            ]
            metrics_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            (root / cell_id / "status.json").write_text(
                json.dumps(
                    {
                        "state": "completed",
                        "fingerprint": fingerprint,
                        "attempt": 1,
                        "run_dir": str(run_dir),
                        "metrics_file": str(metrics_path),
                        "last_metric_step": 2,
                    }
                ),
                encoding="utf-8",
            )
            if with_n1_eval:
                eval_dir = run_dir / "evaluation" / "global_step_2"
                eval_dir.mkdir(parents=True)
                benchmark_metrics = {
                    benchmark: {
                        "avg@1": 0.1 * (index + 1),
                        "num_prompts": renderer.EXPECTED_PROMPTS[benchmark],
                    }
                    for benchmark in renderer.BENCHMARKS
                }
                (eval_dir / "summary.json").write_text(
                    json.dumps(
                        {
                            "checkpoint_step": 2,
                            "n": 1,
                            "benchmarks": benchmark_metrics,
                        }
                    ),
                    encoding="utf-8",
                )
        (root / "suite_manifest.json").write_text(
            json.dumps(
                {
                    "suite_id": "synthetic-render-suite",
                    "protocol": "smoke",
                    "seed": 42,
                    "source_tree_sha256": "synthetic-source",
                    "cells": cells,
                }
            ),
            encoding="utf-8",
        )
        return root

    @staticmethod
    def _figure(manifest: dict, figure_id: str) -> dict:
        return next(item for item in manifest["figures"] if item["id"] == figure_id)

    def test_complete_training_group_renders_with_smoke_and_avg1_labels(self) -> None:
        required = [
            "fig7-student-topk",
            "fig7-overlap-topk",
            "fig7-nonoverlap-topk-author",
        ]
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            suite = self._write_suite(base / "suite", required, with_n1_eval=True)
            output = base / "rendered"
            manifest = renderer.render_all(
                ledger_path=MODULE_DIR / "paper_experiment_ledger.json",
                suite_roots=[suite],
                model_eval_roots=[],
                probe_roots=[],
                output_dir=output,
            )
            figure = self._figure(manifest, "figure-07")
            variant = figure["variants"][0]
            self.assertEqual(figure["status"], "rendered")
            self.assertEqual(variant["status"], "rendered")
            self.assertEqual(variant["protocol"], "smoke")
            self.assertEqual(variant["evaluation_n"], 1)
            self.assertFalse(variant["paper_comparable"])
            self.assertIn("protocol=smoke", variant["annotation"])
            self.assertIn("NOT PAPER-COMPARABLE", variant["annotation"])
            image = Path(variant["output"])
            self.assertEqual(image.name, "figure-07_smoke.png")
            self.assertGreater(image.stat().st_size, 0)
            self.assertTrue((output / "render_manifest.json").is_file())
            self.assertEqual(len(manifest["figures"]), 23)

    def test_missing_cell_skips_instead_of_rendering_partial_group(self) -> None:
        partial = ["fig7-student-topk", "fig7-overlap-topk"]
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            suite = self._write_suite(base / "suite", partial, with_n1_eval=False)
            manifest = renderer.render_all(
                ledger_path=MODULE_DIR / "paper_experiment_ledger.json",
                suite_roots=[suite],
                model_eval_roots=[],
                probe_roots=[],
                output_dir=base / "rendered",
            )
            figure = self._figure(manifest, "figure-07")
            variant = figure["variants"][0]
            self.assertEqual(figure["status"], "skipped")
            self.assertEqual(variant["status"], "skipped")
            self.assertTrue(
                any("fig7-nonoverlap-topk-author" in reason for reason in variant["missing_producers"])
            )
            self.assertFalse((base / "rendered" / "figure-07_smoke.png").exists())

    def test_avg1_cannot_be_paper_comparable_even_with_an_explicit_hint(self) -> None:
        artifact = renderer.EvaluationArtifact(
            source=Path("synthetic-summary.json"),
            label="synthetic",
            n=1,
            checkpoint_step=2,
            values={benchmark: 0.5 for benchmark in renderer.BENCHMARKS},
            summary={"paper_comparable": True},
            explicitly_paper_comparable=True,
        )
        self.assertFalse(renderer.evaluation_is_paper_comparable(artifact, "paper"))

    def test_position_entropy_separates_text_condition_from_undisclosed_window(self) -> None:
        entries = renderer.load_ledger(MODULE_DIR / "paper_experiment_ledger.json")
        figure13 = entries["figure-13"]
        self.assertIsNone(renderer.static_fidelity_veto(figure13))
        values = tuple(0.5 for _ in range(60))
        counts = tuple(1.0 for _ in range(60))
        rows = [
            renderer.plot_position_entropy.PositionEntropyRow(
                step, 256, values, values, counts
            )
            for step in (180, 190, 200)
        ]
        environment = {
            "TOTAL_TRAINING_STEPS": "200",
            "POSITION_ENTROPY_START_STEP": "180",
            "POSITION_ENTROPY_LOG_FREQ": "10",
            "POSITION_ENTROPY_BIN_SIZE": "256",
            "MAX_RESPONSE_LENGTH": "15360",
        }
        text_evidence = renderer.position_entropy_evidence(
            figure13, "paper", rows, environment
        )
        self.assertTrue(text_evidence["text_condition_complete"])
        self.assertFalse(text_evidence["figure_window_complete"])
        self.assertFalse(text_evidence["paper_comparable"])
        self.assertFalse(
            renderer.position_entropy_is_paper_comparable(
                figure13, "paper", rows[:-1], environment
            )
        )
        self.assertFalse(
            renderer.position_entropy_is_paper_comparable(
                figure13, "smoke", rows, environment
            )
        )
        wrong_cadence = dict(environment, POSITION_ENTROPY_LOG_FREQ="1")
        self.assertFalse(
            renderer.position_entropy_is_paper_comparable(
                figure13, "paper", rows, wrong_cadence
            )
        )
        window_rows = [
            renderer.plot_position_entropy.PositionEntropyRow(
                step, 256, values, values, counts
            )
            for step in range(180, 261, 10)
        ]
        window_environment = dict(environment, TOTAL_TRAINING_STEPS="260")
        window_evidence = renderer.position_entropy_evidence(
            figure13, "pilot", window_rows, window_environment
        )
        self.assertTrue(window_evidence["text_condition_complete"])
        self.assertTrue(window_evidence["figure_window_complete"])
        self.assertFalse(window_evidence["paper_comparable"])
        self.assertIn("reconstruction", window_evidence["label"])
        self.assertIsNotNone(renderer.static_fidelity_veto(entries["figure-04"]))
        self.assertIsNotNone(renderer.static_fidelity_veto(entries["figure-20"]))

    def test_position_entropy_raw_schema_must_be_exact_integer_one(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            row = {
                "step": 180,
                "data": {
                    # bool compares equal to one in Python, but is not provenance.
                    "opd/position_entropy_schema_version": True,
                    "position-entropy/bin_size": 256,
                    "position-entropy/student_mean_by_bin": [0.5],
                    "position-entropy/teacher_mean_by_bin": [0.6],
                    "position-entropy/token_count_by_bin": [1],
                },
            }
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exact integer"):
                renderer.read_position_entropy_strict(path)

    def test_long_condition_labels_are_compact_and_unique(self) -> None:
        def cell(cell_id: str, label: str) -> renderer.CellArtifact:
            return renderer.CellArtifact(
                cell_id=cell_id,
                label=label,
                fidelity="synthetic",
                fingerprint=f"fingerprint-{cell_id}",
                environment={},
                status=None,
                metrics_path=None,
                run=None,
                complete=False,
                reason="synthetic",
            )

        labels = renderer.compact_condition_labels(
            [
                cell(
                    "fig5-reverse-r1-1p5b",
                    "JustRL-1.5B student to its R1-Distill-1.5B pre-RL checkpoint, 600 steps",
                ),
                cell(
                    "fig5-reverse-r1-7b",
                    "JustRL-1.5B student to R1-Distill-7B, 600 steps",
                ),
            ]
        )
        self.assertEqual(
            labels,
            ["JustRL-1.5B → R1-Distill-1.5B pre-RL", "JustRL-1.5B → R1-Distill-7B"],
        )
        self.assertEqual(len(labels), len(set(labels)))
        self.assertTrue(all(len(label) <= 44 for label in labels))

    def test_figure19_uses_actual_schema_v1_probability_difference_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "step": 1,
                        "data": {
                            "actor/pg_loss": 0.2,
                            "actor/grad_norm": 1.3,
                            "opd/figure19_metric_schema_version": 1,
                            "val-extrema/prob_diff_at_max_abs_adv_intersection": -0.08,
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            series = renderer.read_figure19_series(path)
            self.assertEqual(series["max_abs_adv_probability_difference"][0].value, -0.08)

            path.write_text(
                json.dumps(
                    {
                        "step": 1,
                        "data": {
                            "actor/pg_loss": 0.2,
                            "actor/grad_norm": 1.3,
                            "opd/figure19_metric_schema_version": 1,
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError, "val-extrema/prob_diff_at_max_abs_adv_intersection"
            ):
                renderer.read_figure19_series(path)

            path.write_text(
                "".join(
                    json.dumps(row) + "\n"
                    for row in (
                        {
                            "step": 1,
                            "data": {
                                "actor/pg_loss": 0.2,
                                "actor/grad_norm": 1.3,
                                "opd/figure19_metric_schema_version": 1,
                                "val-extrema/prob_diff_at_max_abs_adv_intersection": -0.08,
                            },
                        },
                        {
                            "step": 2,
                            "data": {
                                "actor/pg_loss": 0.1,
                                "actor/grad_norm": 1.1,
                                "val-extrema/prob_diff_at_max_abs_adv_intersection": -0.04,
                            },
                        },
                    )
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "same-row"):
                renderer.read_figure19_series(path)

            path.write_text(
                "".join(
                    json.dumps(row) + "\n"
                    for row in (
                        {
                            "step": 1,
                            "data": {
                                "actor/pg_loss": 0.2,
                                "actor/grad_norm": 1.3,
                                "opd/figure19_metric_schema_version": 1,
                                "val-extrema/prob_diff_at_max_abs_adv_intersection": -0.08,
                            },
                        },
                        {
                            "step": 2,
                            "data": {
                                "actor/grad_norm": 1.1,
                                "opd/figure19_metric_schema_version": 1,
                                "val-extrema/prob_diff_at_max_abs_adv_intersection": -0.04,
                            },
                        },
                    )
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "identical step coverage"):
                renderer.read_figure19_series(path)


if __name__ == "__main__":
    unittest.main()
