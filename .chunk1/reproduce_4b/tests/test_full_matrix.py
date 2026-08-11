from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = MODULE_DIR.parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import run_ablations  # noqa: E402


class FullPaperMatrixTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.matrix_path = MODULE_DIR / "paper_full_matrix.json"
        cls.ledger_path = MODULE_DIR / "paper_experiment_ledger.json"
        cls.registry = run_ablations.load_registry(cls.matrix_path, REPO_ROOT)
        cls.datasets = run_ablations.validate_datasets(cls.registry)
        cls.ledger = json.loads(cls.ledger_path.read_text(encoding="utf-8"))
        cls.groups = {group["id"]: group for group in cls.registry["groups"]}
        cls.cells = {
            cell["id"]: (group, cell)
            for group in cls.registry["groups"]
            for cell in group["cells"]
        }
        cls.entries = {entry["id"]: entry for entry in cls.ledger["entries"]}

    def plans(self, groups, protocol: str):
        return run_ablations.build_plans(
            self.registry,
            groups,
            protocol,
            seed=42,
            dataset_by_path=self.datasets,
            source_hash="full-matrix-test-source",
            matrix_sha256=run_ablations.sha256_file(self.matrix_path),
        )

    def test_registry_is_runner_compatible_and_has_31_unique_role_ids(self) -> None:
        self.assertEqual(self.registry["schema_version"], 1)
        self.assertEqual(self.registry["suite_id"], "rethinking-opd-complete-paper-v3")
        self.assertNotEqual(self.registry["suite_id"], "rethinking-opd-formal-ablations-v1")
        self.assertEqual(len(self.cells), 31)
        self.assertEqual(
            sum(len(group["cells"]) for group in self.registry["groups"]),
            31,
            "global cell IDs must be unique",
        )
        for group in self.registry["groups"]:
            run_ablations.validate_group(group)

        default_groups = run_ablations.select_groups(
            self.registry, set(), set(), include_extensions=False
        )
        default_plans = self.plans(default_groups, "smoke")
        self.assertEqual(len(default_plans), 27)
        self.assertEqual(
            {group["id"] for group in self.registry["groups"] if group.get("disabled_by_default")},
            {"fig4_qwen_new_knowledge", "fig20_cross_model_large"},
        )

    def test_scope_distinguishes_roles_environments_and_scientific_conditions(self) -> None:
        all_groups = run_ablations.select_groups(
            self.registry, set(), set(), include_extensions=True
        )
        plans = self.plans(all_groups, "paper")
        full_environments = {
            json.dumps(plan.env, sort_keys=True, separators=(",", ":")) for plan in plans
        }
        operational_keys = {
            "CHECKPOINT_SAVE_MODE",
            "MAX_ACTOR_CKPTS_TO_KEEP",
            "MIN_FREE_GIB",
            "N_GPUS",
            "POSITION_ENTROPY_BIN_SIZE",
            "POSITION_ENTROPY_LOG_FREQ",
            "POSITION_ENTROPY_START_STEP",
            "SAVE_FREQ",
        }
        scientific_conditions = {
            json.dumps(
                {key: value for key, value in plan.env.items() if key not in operational_keys},
                sort_keys=True,
                separators=(",", ":"),
            )
            for plan in plans
        }
        self.assertEqual((len(plans), len(full_environments), len(scientific_conditions)), (31, 29, 28))
        self.assertIn("Thirty-one registered figure/table roles", self.registry["paper"]["scope"])
        self.assertNotIn("unique OPD training cells", self.registry["paper"]["scope"])

    def test_each_group_changes_only_its_declared_factor_bundle(self) -> None:
        for group in self.registry["groups"]:
            factor_keys = set(group["factor_keys"])
            self.assertFalse(factor_keys & set(group["constants"]), group["id"])
            expanded = []
            for cell in group["cells"]:
                self.assertEqual(set(cell["factors"]), factor_keys, cell["id"])
                expanded.append({**group["constants"], **cell["factors"]})
            reference = expanded[0]
            for candidate in expanded[1:]:
                changed = {key for key in reference if reference[key] != candidate[key]}
                self.assertEqual(changed, factor_keys, group["id"])

    def test_exact_models_revisions_thinking_modes_and_special_budgets(self) -> None:
        public_revisions = []
        for group in self.registry["groups"]:
            for key, value in group["constants"].items():
                if key.endswith("_REVISION") and not value.startswith("UNPUBLISHED"):
                    public_revisions.append(value)
            for cell in group["cells"]:
                for key, value in cell["factors"].items():
                    if key.endswith("_REVISION") and not value.startswith("UNPUBLISHED"):
                        public_revisions.append(value)
        self.assertTrue(public_revisions)
        self.assertTrue(all(re.fullmatch(r"[0-9a-f]{40}", value) for value in public_revisions))

        fig2 = self.groups["fig2_teacher_pattern"]
        thinking = {cell["id"]: cell["factors"]["ENABLE_THINKING"] for cell in fig2["cells"]}
        self.assertEqual(thinking["fig2-compatible-grpo"], "auto")
        self.assertEqual(thinking["fig2-mismatch-nonthinking"], "false")

        qwen_unpublished = self.groups["fig4_qwen_new_knowledge"]
        self.assertTrue(qwen_unpublished["disabled_by_default"])
        self.assertEqual(qwen_unpublished["constants"]["ENABLE_THINKING"], "false")
        qwen_public_cell = run_ablations.select_groups(
            self.registry,
            set(),
            {"fig4-qwen-nonthinking"},
            include_extensions=False,
        )
        self.assertEqual(len(self.plans(qwen_public_cell, "smoke")), 1)

        fig5 = self.groups["fig5_reverse_distillation"]
        paper_plans = self.plans([fig5], "paper")
        smoke_plans = self.plans([fig5], "smoke")
        self.assertEqual({plan.env["TOTAL_TRAINING_STEPS"] for plan in paper_plans}, {"600"})
        self.assertEqual({plan.env["PRESET"] for plan in smoke_plans}, {"smoke"})
        self.assertTrue(all("TOTAL_TRAINING_STEPS" not in plan.env for plan in smoke_plans))
        all_paper_groups = run_ablations.select_groups(
            self.registry, set(), set(), include_extensions=True
        )
        all_paper_plans = self.plans(all_paper_groups, "paper")
        self.assertEqual(
            {plan.env["CHECKPOINT_SAVE_MODE"] for plan in all_paper_plans},
            {"model_only"},
        )
        self.assertEqual(
            {plan.env["MAX_ACTOR_CKPTS_TO_KEEP"] for plan in all_paper_plans},
            {"0"},
        )

        fig20 = self.groups["fig20_cross_model_large"]
        self.assertTrue(fig20["disabled_by_default"])
        self.assertEqual(fig20["allowed_protocols"], ["paper"])
        self.assertEqual(fig20["constants"]["N_GPUS"], "8")
        self.assertEqual(
            fig20["constants"]["STUDENT_MODEL"],
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        )
        self.assertEqual(
            {cell["factors"]["TEACHER_MODEL"] for cell in fig20["cells"]},
            {
                "Skywork/Skywork-OR1-Math-7B",
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            },
        )
        paper_large = self.plans([fig20], "paper")
        self.assertEqual({plan.env["N_GPUS"] for plan in paper_large}, {"8"})
        with self.assertRaisesRegex(ValueError, "only allows protocols"):
            self.plans([fig20], "smoke")

    def test_ledger_covers_figures_1_to_23_and_tables_1_to_3(self) -> None:
        expected = {f"figure-{number:02d}" for number in range(1, 24)} | {
            f"table-{number:02d}" for number in range(1, 4)
        }
        self.assertEqual(set(self.entries), expected)
        self.assertEqual(self.ledger["suite_id"], self.registry["suite_id"])

        referenced_cells = set()
        for entry in self.ledger["entries"]:
            self.assertIn("producer", entry)
            self.assertIn("status", entry)
            self.assertIn("fidelity", entry)
            self.assertIn("blocker", entry)
            producer = entry["producer"]
            self.assertIsInstance(producer.get("type"), str)
            self.assertIsInstance(producer.get("cell_ids"), list)
            self.assertTrue(set(producer["cell_ids"]) <= set(self.cells), entry["id"])
            referenced_cells.update(producer["cell_ids"])
        self.assertEqual(referenced_cells, set(self.cells))

    def test_ledger_expresses_training_reuse_without_duplicate_cells(self) -> None:
        def cells(entry_id: str) -> set[str]:
            return set(self.entries[entry_id]["producer"]["cell_ids"])

        reused_failure = "fig4-6-deepseek-r1-7b"
        self.assertIn(reused_failure, cells("figure-04"))
        self.assertIn(reused_failure, cells("figure-06"))
        self.assertEqual(cells("figure-02"), cells("figure-17"))
        self.assertEqual(cells("figure-08"), cells("figure-21"))
        self.assertEqual(cells("figure-09"), cells("figure-22"))
        self.assertEqual(cells("figure-06"), cells("figure-18"))
        self.assertEqual(cells("figure-06"), cells("figure-19"))
        self.assertEqual(cells("figure-13"), cells("figure-23"))
        self.assertEqual(cells("figure-11"), cells("figure-12"))


if __name__ == "__main__":
    unittest.main()
