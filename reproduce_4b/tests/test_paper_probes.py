from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import analyze_sequence_reward as sequence_reward  # noqa: E402
import probe_teacher_continuation as continuation  # noqa: E402


def sampling(model: str, revision: str, *, n: int = 1) -> dict:
    return {
        "model": model,
        "revision": revision,
        "seed": 42,
        "n": n,
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 31_744,
        "thinking": "off",
    }


def rollout_record(example_id: str, response: list[int], *, correct: bool = False) -> dict:
    return {
        "schema_version": 1,
        "batch_id": "fixed-batch-a",
        "example_id": example_id,
        "rollout_id": 0,
        "prompt_token_ids": [101, 102],
        "response_token_ids": response,
        "correct": correct,
        "sampling": sampling("student/model", "student-revision"),
    }


def score_record(rollout: dict, role: str, values: list[float]) -> dict:
    model = f"{role}/model"
    revision = f"{role}-revision"
    return {
        "schema_version": 1,
        "batch_id": rollout["batch_id"],
        "example_id": rollout["example_id"],
        "rollout_id": rollout["rollout_id"],
        "role": role,
        "model": model,
        "revision": revision,
        "prompt_token_ids": rollout["prompt_token_ids"],
        "response_token_ids": rollout["response_token_ids"],
        "token_logprobs": values,
        "rollout_fingerprint": sequence_reward.rollout_fingerprint(rollout),
        "scoring": {
            "method": "causal-next-token",
            "runtime": "unit-test",
            "dtype": "float32",
            "device": "cpu",
        },
    }


def validated_rollout(example_id: str, length: int, *, correct: bool = False):
    record = rollout_record(example_id, list(range(length)), correct=correct)
    return sequence_reward.ValidatedRollout(
        batch_id=record["batch_id"],
        example_id=example_id,
        rollout_id=0,
        prompt_token_ids=tuple(record["prompt_token_ids"]),
        response_token_ids=tuple(record["response_token_ids"]),
        correct=correct,
        sampling=record["sampling"],
        fingerprint=sequence_reward.rollout_fingerprint(record),
    )


class SequenceRewardProbeTest(unittest.TestCase):
    def test_tie_safe_auroc_awards_half_credit(self) -> None:
        # Each positive beats the negative at 0 and ties the negative at 1.
        auc = sequence_reward.tie_safe_auroc(
            [True, True, False, False],
            [1.0, 1.0, 1.0, 0.0],
        )
        self.assertAlmostEqual(auc, 0.75)

    def test_sequence_reward_uses_fixed_same_rollout_batch(self) -> None:
        first = rollout_record("a", [1, 2], correct=True)
        second = rollout_record("b", [3, 4], correct=False)
        student = [score_record(first, "student", [-2.0, -3.0])]
        teacher = [
            score_record(first, "teacher", [-1.0, -1.0]),
            score_record(second, "teacher", [-1.0, -1.0]),
        ]
        with self.assertRaisesRegex(ValueError, "fixed same rollout batch"):
            sequence_reward.compute_sequence_rewards(
                [first, second],
                student,
                teacher,
                expected_sampling=sampling("student/model", "student-revision"),
                student_model="student/model",
                student_revision="student-revision",
                teacher_model="teacher/model",
                teacher_revision="teacher-revision",
            )

    def test_sequence_reward_rejects_per_token_length_mismatch(self) -> None:
        record = rollout_record("a", [1, 2], correct=True)
        bad_student = score_record(record, "student", [-1.0])
        teacher = score_record(record, "teacher", [-1.0, -1.0])
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            sequence_reward.compute_sequence_rewards(
                [record],
                [bad_student],
                [teacher],
                expected_sampling=sampling("student/model", "student-revision"),
                student_model="student/model",
                student_revision="student-revision",
                teacher_model="teacher/model",
                teacher_revision="teacher-revision",
            )

    def test_sequence_reward_formula_and_distribution(self) -> None:
        correct = rollout_record("a", [1, 2], correct=True)
        incorrect = rollout_record("b", [3, 4], correct=False)
        rows = sequence_reward.compute_sequence_rewards(
            [correct, incorrect],
            [
                score_record(correct, "student", [-3.0, -2.0]),
                score_record(incorrect, "student", [-1.0, -1.0]),
            ],
            [
                score_record(correct, "teacher", [-1.0, -1.0]),
                score_record(incorrect, "teacher", [-2.0, -2.0]),
            ],
            expected_sampling=sampling("student/model", "student-revision"),
            student_model="student/model",
            student_revision="student-revision",
            teacher_model="teacher/model",
            teacher_revision="teacher-revision",
        )
        self.assertAlmostEqual(rows[0].sequence_mean_reward, 1.5)
        self.assertAlmostEqual(rows[1].sequence_mean_reward, -1.0)
        self.assertAlmostEqual(sequence_reward.summarize_sequence_rewards(rows)["auroc"], 1.0)

    def test_dual_teacher_summary_requires_shared_action_fingerprint(self) -> None:
        fingerprint = "a" * 64
        base = {
            "num_rollouts": sequence_reward.FIG14_TOTAL_ROLLOUTS,
            "num_prompts": sequence_reward.FIG14_PROMPT_COUNT,
            "batch_id": "fixed",
            "action_batch_sha256": fingerprint,
            "auroc": 0.75,
            "auroc_bootstrap_ci": {"lower": 0.70, "upper": 0.80},
            "correct": {"n": 2, "mean": 1.0, "std_population": 0.1},
            "incorrect": {"n": 2, "mean": -1.0, "std_population": 0.1},
        }
        combined = sequence_reward.combine_teacher_summaries(
            {"JustRL": base, "R1-7B": {**base, "auroc": 0.73}},
            action_batch_sha256=fingerprint,
        )
        self.assertEqual(combined["conclusion_state"], "replicated")
        with self.assertRaisesRegex(ValueError, "shared action fingerprint"):
            sequence_reward.combine_teacher_summaries(
                {"JustRL": base, "R1-7B": {**base, "action_batch_sha256": "b" * 64}},
                action_batch_sha256=fingerprint,
            )

    def test_kv_chunk_scorer_matches_one_shot_on_cpu_fixture(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is unavailable")

        class Output:
            def __init__(self, logits, past):
                self.logits = logits
                self.past_key_values = past

        class ToyModel:
            def __call__(self, *, input_ids, past_key_values, use_cache):
                del use_cache
                past = int(past_key_values or 0)
                vocab = 32
                rows = []
                for local, token in enumerate(input_ids[0].tolist()):
                    center = (token + past + local + 1) % vocab
                    rows.append([-abs(index - center) for index in range(vocab)])
                return Output(torch.tensor([rows], dtype=torch.float32), past + input_ids.shape[1])

        rollout = validated_rollout("toy", 7)
        one_shot = sequence_reward.score_action_with_kv_chunks(
            ToyModel(), torch, rollout, device="cpu", chunk_tokens=100
        )
        chunked = sequence_reward.score_action_with_kv_chunks(
            ToyModel(), torch, rollout, device="cpu", chunk_tokens=2
        )
        self.assertEqual(len(chunked), 7)
        for left, right in zip(one_shot, chunked, strict=True):
            self.assertAlmostEqual(left, right, places=6)


class TeacherContinuationProbeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.teacher_sampling = sampling("teacher/model", "teacher-revision")

    def test_strict_length_filter_excludes_exactly_16384(self) -> None:
        boundary = validated_rollout("boundary", continuation.STRICT_LENGTH_THRESHOLD)
        longer = validated_rollout("longer", continuation.STRICT_LENGTH_THRESHOLD + 1)
        selected = continuation.select_long_rollouts([boundary, longer])
        self.assertEqual([row.example_id for row in selected], ["longer"])

    def test_plan_contains_exact_prefix_grid_and_student_prefixes(self) -> None:
        rollout = validated_rollout("long", continuation.STRICT_LENGTH_THRESHOLD + 1, correct=True)
        plan = continuation.build_continuation_plan([rollout], self.teacher_sampling)
        self.assertEqual([row["prefix_length"] for row in plan], list(continuation.PREFIX_LENGTHS))
        for row in plan:
            prefix_length = row["prefix_length"]
            self.assertEqual(len(row["prefix_token_ids"]), prefix_length)
            self.assertEqual(row["prefix_token_ids"], list(rollout.response_token_ids[:prefix_length]))
            self.assertEqual(
                row["max_continuation_tokens"],
                continuation.MAX_TOTAL_RESPONSE_TOKENS - prefix_length,
            )

    def test_plan_rejects_tampered_prefix(self) -> None:
        rollout = validated_rollout("long", continuation.STRICT_LENGTH_THRESHOLD + 1)
        plan = continuation.build_continuation_plan([rollout], self.teacher_sampling)
        plan[0] = {**plan[0], "prefix_token_ids": [999] + plan[0]["prefix_token_ids"][1:]}
        with self.assertRaisesRegex(ValueError, "exact student response prefix"):
            continuation.validate_plan(plan, [rollout], self.teacher_sampling)

    def test_result_pairing_fails_closed_when_one_prefix_is_missing(self) -> None:
        rollout = validated_rollout("long", continuation.STRICT_LENGTH_THRESHOLD + 1)
        plan = continuation.build_continuation_plan([rollout], self.teacher_sampling)
        results = [
            {
                "schema_version": 1,
                "protocol_id": continuation.PROTOCOL_ID,
                "pair_id": row["pair_id"],
                "batch_id": row["batch_id"],
                "example_id": row["example_id"],
                "rollout_id": row["rollout_id"],
                "rollout_fingerprint": row["rollout_fingerprint"],
                "prefix_length": row["prefix_length"],
                "teacher_continuation_token_ids": [7, 8],
                "teacher_correct": False,
                "sampling": row["teacher_sampling"],
            }
            for row in plan[:-1]
        ]
        with self.assertRaisesRegex(ValueError, "pairing failure"):
            continuation.validate_continuation_results(plan, results, self.teacher_sampling)

    def test_underpowered_paired_probe_is_inconclusive_without_resampling(self) -> None:
        rows = []
        for example_id in ("a", "b"):
            for prefix in continuation.PREFIX_LENGTHS:
                rows.append(
                    {
                        "pair_id": f"{example_id}-{prefix}",
                        "batch_id": "fixed",
                        "example_id": example_id,
                        "rollout_id": 0,
                        "rollout_fingerprint": f"fingerprint-{example_id}",
                        "prefix_length": prefix,
                        "student_correct": False,
                        "teacher_correct": prefix < 16_384,
                    }
                )
        summary = continuation.summarize_paired_results(rows, bootstrap_replicates=100)
        self.assertEqual(summary["num_selected_rollouts"], 2)
        self.assertEqual(summary["conclusion_state"], "inconclusive")
        self.assertIn("minimum", summary["conclusion_reason"])

    def test_atomic_chunks_resume_and_fail_closed_on_missing_chunk(self) -> None:
        import probe_runtime

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            probe_runtime.write_chunk(
                root,
                "fixture",
                0,
                contract_sha256="contract-a",
                records=[{"value": 1}],
            )
            self.assertEqual(
                probe_runtime.completed_ordinals(
                    root,
                    "fixture",
                    range(2),
                    contract_sha256="contract-a",
                    expected_records=1,
                ),
                {0},
            )
            with self.assertRaisesRegex(ValueError, "missing"):
                probe_runtime.merge_chunks(
                    root,
                    "fixture",
                    2,
                    contract_sha256="contract-a",
                    expected_records_per_chunk=1,
                    output_jsonl=root / "merged.jsonl",
                )
            with self.assertRaisesRegex(ValueError, "contract mismatch"):
                probe_runtime.completed_ordinals(
                    root,
                    "fixture",
                    range(1),
                    contract_sha256="contract-b",
                    expected_records=1,
                )


if __name__ == "__main__":
    unittest.main()
