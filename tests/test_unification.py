#!/usr/bin/env python3
"""Tests for Stage 4 parameter-unification contracts and score gates."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import unification  # noqa: E402


FULL_OVERRIDES = {
    "prior_std": 1.1,
    "target_norm": "running_return",
    "cov_floor": 0.02,
    "target_norm_update": "ema",
    "target_norm_ema_tau": 0.99,
    "cov_weight_clip": [0.05, 1.0],
    "ess_floor": 0.2,
    "alpha_warmup_updates": 100,
    "old_critic_update": "ema",
    "old_critic_ema_tau": 0.995,
    "bootstrap_mean_mode": "remove_max",
}


def make_plan(overrides_by_env: dict[str, dict] | None = None) -> dict:
    overrides_by_env = overrides_by_env or {
        env: dict(FULL_OVERRIDES) for env in unification.ENVIRONMENTS
    }
    jobs = []
    for env in unification.ENVIRONMENTS:
        overrides = overrides_by_env[env]
        variant = "unified_a" if overrides["prior_std"] <= 1.1 else "unified_b"
        for seed in unification.SEEDS:
            jobs.append(
                {
                    "env": env,
                    "seed": seed,
                    "variant": variant,
                    "overrides": dict(overrides),
                    "hypothesis": "Share non-reward-scale stability parameters.",
                    "evidence": "All per-environment incumbents already pass the target gate.",
                }
            )
    return {
        "status": "continue",
        "target_envs": list(unification.ENVIRONMENTS),
        "reasoning": ["Test a fully specified Stage 4 unification cohort."],
        "jobs": jobs,
    }


def cluster_overrides(group_count: int = 2) -> dict[str, dict]:
    result = {}
    for index, env in enumerate(unification.ENVIRONMENTS):
        overrides = dict(FULL_OVERRIDES)
        overrides["prior_std"] = 1.0 + 0.2 * (index % group_count)
        result[env] = overrides
    return result


def passing_scores(score: float = 100.0) -> list[dict]:
    return [
        {"env": env, "variant": "unified", "complete": True, "top5_mean": score}
        for env in unification.ENVIRONMENTS
    ]


def paper_targets(mean: float = 100.0) -> dict:
    return {
        "environments": {
            env: {"paper_mean": mean} for env in unification.ENVIRONMENTS
        }
    }


def incumbents(score: float = 100.0) -> dict:
    return {
        "incumbents": {
            env: {"top5_mean": score} for env in unification.ENVIRONMENTS
        }
    }


class UnificationPlanTests(unittest.TestCase):
    def assert_rejected(self, plan: dict, message: str, *, mode: str = "global") -> None:
        with self.assertRaisesRegex(unification.UnificationValidationError, message):
            unification.validate_unification_plan(plan, mode=mode)

    def test_default_set_is_all_non_reward_scale_tunables(self) -> None:
        expected = set(unification._plan_validator.ALL_TUNABLE_PARAMETERS) - set(
            unification.REWARD_SCALE_PARAMETERS
        )
        self.assertEqual(set(unification.DEFAULT_UNIFIED_PARAMETERS), expected)
        self.assertTrue(
            set(unification.DEFAULT_UNIFIED_PARAMETERS).isdisjoint(
                unification.REWARD_SCALE_PARAMETERS
            )
        )

    def test_valid_global_plan_covers_exactly_48_jobs(self) -> None:
        approved = unification.validate_unification_plan(make_plan(), mode="global")
        self.assertEqual(len(approved["jobs"]), 48)
        self.assertEqual({job["env"] for job in approved["jobs"]}, set(unification.ENVIRONMENTS))
        self.assertEqual(set(approved["jobs"][0]["overrides"]), set(FULL_OVERRIDES))

    def test_global_plan_rejects_different_shared_overrides(self) -> None:
        overrides = cluster_overrides(2)
        self.assert_rejected(
            make_plan(overrides),
            "one shared override configuration; found 2",
        )

    def test_plan_must_cover_all_twelve_environments(self) -> None:
        plan = make_plan()
        removed = unification.ENVIRONMENTS[-1]
        plan["jobs"] = [job for job in plan["jobs"] if job["env"] != removed]
        plan["target_envs"].remove(removed)
        self.assert_rejected(plan, "must target all 12 environments")

    def test_each_environment_requires_all_four_seeds(self) -> None:
        plan = make_plan()
        plan["jobs"] = [
            job
            for job in plan["jobs"]
            if not (job["env"] == unification.ENVIRONMENTS[0] and job["seed"] == 4)
        ]
        self.assert_rejected(plan, "exactly one job for each seed")

    def test_override_must_completely_include_unified_parameters(self) -> None:
        overrides = {env: dict(FULL_OVERRIDES) for env in unification.ENVIRONMENTS}
        for value in overrides.values():
            value.pop("ess_floor")
        self.assert_rejected(make_plan(overrides), "missing=\\['ess_floor'\\]")

    def test_reward_scale_parameters_cannot_enter_default_unification(self) -> None:
        overrides = {env: dict(FULL_OVERRIDES) for env in unification.ENVIRONMENTS}
        for value in overrides.values():
            value["reward_scaler"] = 0.1
        self.assert_rejected(make_plan(overrides), "extra=\\['reward_scaler'\\]")

    def test_parameter_values_still_use_stage_three_range_contract(self) -> None:
        overrides = {env: dict(FULL_OVERRIDES) for env in unification.ENVIRONMENTS}
        for value in overrides.values():
            value["prior_std"] = 5.0
        self.assert_rejected(make_plan(overrides), "outside")

    def test_valid_cluster_plan_has_exactly_two_shared_configs(self) -> None:
        approved = unification.validate_unification_plan(
            make_plan(cluster_overrides(2)), mode="cluster"
        )
        signatures = {
            unification._canonical_json(job["overrides"])
            for job in approved["jobs"]
        }
        self.assertEqual(len(signatures), 2)

    def test_cluster_plan_rejects_one_or_three_shared_configs(self) -> None:
        self.assert_rejected(
            make_plan(),
            "exactly two shared override configuration; found 1",
            mode="cluster",
        )
        self.assert_rejected(
            make_plan(cluster_overrides(3)),
            "exactly two shared override configuration; found 3",
            mode="cluster",
        )

    def test_custom_subset_must_be_complete_and_cannot_include_reward_scale(self) -> None:
        subset = ("prior_std", "target_norm")
        overrides = {
            env: {"prior_std": 1.1, "target_norm": "running_return"}
            for env in unification.ENVIRONMENTS
        }
        approved = unification.validate_unification_plan(
            make_plan(overrides), mode="global", unified_parameters=subset
        )
        self.assertEqual(set(approved["jobs"][0]["overrides"]), set(subset))
        with self.assertRaisesRegex(
            unification.UnificationValidationError, "environment-specific"
        ):
            unification.validate_unification_plan(
                make_plan(),
                mode="global",
                unified_parameters=("prior_std", "reward_scaler"),
            )


class UnificationEvaluationTests(unittest.TestCase):
    def test_candidate_at_exact_target_and_regression_boundaries_is_accepted(self) -> None:
        report = unification.evaluate_unification_candidate(
            passing_scores(90.0),
            paper_targets(100.0),
            incumbents(90.0 / 0.95),
        )
        self.assertTrue(report["complete_gate"])
        self.assertTrue(report["target_gate"])
        self.assertTrue(report["regression_gate"])
        self.assertTrue(report["accepted"])
        self.assertTrue(report["per_env"]["Humanoid"]["accepted"])

    def test_target_failure_is_reported_per_environment(self) -> None:
        scores = passing_scores(100.0)
        scores[0]["top5_mean"] = 89.99
        report = unification.evaluate_unification_candidate(
            scores, paper_targets(), incumbents(90.0)
        )
        row = report["per_env"][unification.ENVIRONMENTS[0]]
        self.assertFalse(row["target_gate"])
        self.assertTrue(row["regression_gate"])
        self.assertFalse(row["accepted"])
        self.assertFalse(report["accepted"])

    def test_regression_failure_is_reported_per_environment(self) -> None:
        scores = passing_scores(100.0)
        scores[1]["top5_mean"] = 94.99
        report = unification.evaluate_unification_candidate(
            scores, paper_targets(), incumbents(100.0)
        )
        row = report["per_env"][unification.ENVIRONMENTS[1]]
        self.assertTrue(row["target_gate"])
        self.assertFalse(row["regression_gate"])
        self.assertFalse(report["regression_gate"])

    def test_incomplete_score_fails_all_gates_for_that_environment(self) -> None:
        scores = passing_scores()
        scores[2]["complete"] = False
        report = unification.evaluate_unification_candidate(
            scores, paper_targets(), incumbents()
        )
        row = report["per_env"][unification.ENVIRONMENTS[2]]
        self.assertFalse(row["complete_gate"])
        self.assertFalse(row["target_gate"])
        self.assertFalse(row["regression_gate"])
        self.assertIn("candidate_incomplete", row["issues"])

    def test_missing_environment_is_not_silently_accepted(self) -> None:
        report = unification.evaluate_unification_candidate(
            passing_scores()[:-1], paper_targets(), incumbents()
        )
        row = report["per_env"][unification.ENVIRONMENTS[-1]]
        self.assertIn("missing_candidate_result", row["issues"])
        self.assertFalse(report["complete_gate"])
        self.assertFalse(report["accepted"])

    def test_duplicate_and_unexpected_results_fail_structural_gate(self) -> None:
        scores = passing_scores()
        scores.append(dict(scores[0]))
        scores.append({"env": "UnknownEnv", "complete": True, "top5_mean": 100.0})
        report = unification.evaluate_unification_candidate(
            scores, paper_targets(), incumbents()
        )
        self.assertEqual(report["duplicate_environments"], [unification.ENVIRONMENTS[0]])
        self.assertEqual(report["unexpected_environments"], ["UnknownEnv"])
        self.assertFalse(report["accepted"])

    def test_missing_paper_or_incumbent_and_nonfinite_score_fail_closed(self) -> None:
        scores = passing_scores()
        scores[3]["top5_mean"] = math.nan
        targets = paper_targets()["environments"]
        current = incumbents()["incumbents"]
        targets.pop(unification.ENVIRONMENTS[4])
        current.pop(unification.ENVIRONMENTS[5])
        report = unification.evaluate_unification_candidate(scores, targets, current)
        self.assertIn(
            "candidate_score_missing_or_nonfinite",
            report["per_env"][unification.ENVIRONMENTS[3]]["issues"],
        )
        self.assertIn(
            "paper_target_missing_or_invalid",
            report["per_env"][unification.ENVIRONMENTS[4]]["issues"],
        )
        self.assertIn(
            "incumbent_score_missing_or_nonfinite",
            report["per_env"][unification.ENVIRONMENTS[5]]["issues"],
        )
        self.assertFalse(report["accepted"])

    def test_scorer_payload_and_loop_state_wrappers_are_supported(self) -> None:
        report = unification.evaluate_unification_candidate(
            {"schema_version": 1, "results": passing_scores()},
            paper_targets(),
            incumbents(),
        )
        self.assertTrue(report["accepted"])


if __name__ == "__main__":
    unittest.main()
