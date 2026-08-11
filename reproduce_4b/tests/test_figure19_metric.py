from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from verl.utils.opd import (  # noqa: E402
    prob_diff_at_max_abs_advantage,
    student_side_topk_advantages,
)


class Figure19MetricTest(unittest.TestCase):
    def test_selects_largest_absolute_advantage_and_keeps_probability_sign(self) -> None:
        advantages = torch.tensor([[[0.8, -1.2, 50.0]]])
        overlap = torch.tensor([[[True, True, False]]])
        student = torch.log(torch.tensor([[[0.6, 0.2, 0.1]]]))
        teacher = torch.log(torch.tensor([[[0.1, 0.7, 0.1]]]))

        actual = prob_diff_at_max_abs_advantage(
            advantages,
            student,
            teacher,
            overlap,
            torch.ones(1, 1),
        )

        self.assertIsNotNone(actual)
        # The selected advantage is -1.2, not +0.8 or the out-of-support +50.
        self.assertAlmostEqual(actual.item(), -0.5, places=6)

    def test_averages_over_valid_states_and_handles_empty_support(self) -> None:
        advantages = torch.tensor([[[1.0, -2.0], [9.0, 8.0], [0.1, 0.3]]])
        student = torch.log(torch.tensor([[[0.4, 0.3], [0.9, 0.05], [0.2, 0.7]]]))
        teacher = torch.log(torch.tensor([[[0.1, 0.6], [0.1, 0.8], [0.6, 0.2]]]))
        overlap = torch.tensor([[[True, True], [True, True], [False, True]]])
        response = torch.tensor([[1, 0, 1]])

        actual = prob_diff_at_max_abs_advantage(
            advantages,
            student,
            teacher,
            overlap,
            response,
        )

        self.assertIsNotNone(actual)
        # State 0 selects index 1: 0.3 - 0.6. State 2 selects index 1: 0.7 - 0.2.
        self.assertAlmostEqual(actual.item(), 0.1, places=6)
        self.assertIsNone(
            prob_diff_at_max_abs_advantage(
                advantages,
                student,
                teacher,
                overlap,
                torch.zeros_like(response),
            )
        )
        self.assertIsNone(
            prob_diff_at_max_abs_advantage(
                advantages,
                student,
                teacher,
                torch.zeros_like(overlap),
                response,
            )
        )
        empty = torch.empty(1, 1, 0)
        self.assertIsNone(
            prob_diff_at_max_abs_advantage(
                empty,
                empty,
                empty,
                torch.empty(1, 1, 0, dtype=torch.bool),
                torch.ones(1, 1),
            )
        )

    def test_shape_contract_fails_closed(self) -> None:
        values = torch.zeros(1, 2, 3)
        with self.assertRaisesRegex(ValueError, "same shape"):
            prob_diff_at_max_abs_advantage(
                values,
                values[..., :2],
                values,
                torch.ones_like(values, dtype=torch.bool),
                torch.ones(1, 2),
            )
        with self.assertRaisesRegex(ValueError, "response_mask"):
            prob_diff_at_max_abs_advantage(
                values,
                values,
                values,
                torch.ones_like(values, dtype=torch.bool),
                torch.ones(1, 3),
            )

    def test_union_uses_student_side_even_if_teacher_side_has_larger_extrema(self) -> None:
        union_advantages = torch.tensor([[[0.2, -0.9, 100.0, -200.0]]])
        student_advantages = student_side_topk_advantages(
            union_advantages,
            2,
            union_support=True,
        )
        torch.testing.assert_close(student_advantages, torch.tensor([[[0.2, -0.9]]]))

        actual = prob_diff_at_max_abs_advantage(
            student_advantages,
            torch.log(torch.tensor([[[0.8, 0.1]]])),
            torch.log(torch.tensor([[[0.3, 0.6]]])),
            torch.ones(1, 1, 2, dtype=torch.bool),
            torch.ones(1, 1),
        )
        self.assertIsNotNone(actual)
        self.assertAlmostEqual(actual.item(), -0.5, places=6)
        with self.assertRaisesRegex(ValueError, "support width 4"):
            student_side_topk_advantages(
                union_advantages[..., :2],
                2,
                union_support=True,
            )


if __name__ == "__main__":
    unittest.main()
