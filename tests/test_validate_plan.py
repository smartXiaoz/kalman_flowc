#!/usr/bin/env python3
"""Contract tests for the untrusted GPT next-round proposal boundary."""

from __future__ import annotations

import json
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import call_gpt_planner  # noqa: E402
import validate_plan as validator  # noqa: E402


def cohort(
    env: str = "Ant",
    variant: str = "candidate_a",
    overrides: dict | None = None,
    seeds: tuple[int, ...] = validator.SEEDS,
    hypothesis: str = "Reduce critic variance without changing paper parameters.",
    evidence: str = "The current four-seed cohort remains below its paper target.",
) -> list[dict]:
    if overrides is None:
        overrides = {"prior_std": 1.1, "target_norm": "running_return", "cov_floor": 0.02}
    return [
        {
            "env": env,
            "seed": seed,
            "variant": variant,
            "overrides": dict(overrides),
            "hypothesis": hypothesis,
            "evidence": evidence,
        }
        for seed in seeds
    ]


def plan_for(jobs: list[dict], status: str = "continue") -> dict:
    target_envs = list(dict.fromkeys(job["env"] for job in jobs))
    return {
        "status": status,
        "target_envs": target_envs,
        "reasoning": ["Only locally permitted parameters are proposed."],
        "jobs": jobs,
    }


class ValidatePlanTests(unittest.TestCase):
    def assertRejected(self, plan: dict, message: str, **kwargs) -> None:  # noqa: N802
        with self.assertRaisesRegex(validator.PlanValidationError, message):
            validator.validate_plan(plan, stage=kwargs.pop("stage", 1), **kwargs)

    def test_environment_contract_is_exactly_the_paper_twelve(self) -> None:
        self.assertEqual(
            set(validator.ENVIRONMENTS),
            {
                "FrankaCubeStack",
                "Quadcopter",
                "FrankaCabinet",
                "BallBalance",
                "ShadowHandOpenAI_FF",
                "Anymal",
                "Ant",
                "AllegroHand",
                "Humanoid",
                "AllegroKuka",
                "AllegroKukaTwoArmsLSTM",
                "ShadowHand",
            },
        )

    def test_valid_stage_one_four_seed_cohort_is_normalized(self) -> None:
        approved = validator.validate_plan(plan_for(cohort()), stage=1)
        self.assertEqual([job["seed"] for job in approved["jobs"]], [1, 2, 3, 4])
        self.assertEqual(approved["target_envs"], ["Ant"])
        self.assertEqual(approved["jobs"][0]["overrides"]["prior_std"], 1.1)

    def test_invalid_environment_is_rejected(self) -> None:
        jobs = cohort()
        for job in jobs:
            job["env"] = "Cartpole"
        self.assertRejected(plan_for(jobs), "invalid environments")

    def test_seed_must_be_exactly_one_through_four(self) -> None:
        jobs = cohort(seeds=(1, 2, 3, 5))
        self.assertRejected(plan_for(jobs), "seed must be exactly")

    def test_fixed_paper_parameter_cannot_be_overridden(self) -> None:
        jobs = cohort(overrides={"gamma": 0.98})
        self.assertRejected(plan_for(jobs), "fixed/protected")

    def test_core_algorithm_switch_cannot_be_overridden(self) -> None:
        jobs = cohort(overrides={"rollout_mean_mode": "normal"})
        self.assertRejected(plan_for(jobs), "fixed/protected")

    def test_preserved_architecture_cannot_be_overridden(self) -> None:
        jobs = cohort(overrides={"hidden_size": 256})
        self.assertRejected(plan_for(jobs), "fixed/protected")

    def test_dead_parameter_cannot_be_searched(self) -> None:
        jobs = cohort(overrides={"noise_ratio": 0.5})
        self.assertRejected(plan_for(jobs), "dead or unused")

    def test_later_stage_parameter_is_closed_in_stage_one(self) -> None:
        jobs = cohort(overrides={"ess_floor": 0.5})
        self.assertRejected(plan_for(jobs), "not open in stage 1")

    def test_stage_two_opens_normalization_but_not_old_critic(self) -> None:
        approved = validator.validate_plan(
            plan_for(cohort(overrides={"target_norm_update": "ema", "ess_floor": 0.5})),
            stage=2,
        )
        self.assertEqual(approved["jobs"][0]["overrides"]["target_norm_update"], "ema")
        self.assertRejected(
            plan_for(cohort(overrides={"old_critic_update": "ema"})),
            "not open in stage 2",
            stage=2,
        )

    def test_stage_three_opens_old_critic_and_bootstrap(self) -> None:
        approved = validator.validate_plan(
            plan_for(
                cohort(
                    overrides={
                        "old_critic_update": "ema",
                        "old_critic_ema_tau": 0.995,
                        "bootstrap_mean_mode": "remove_max_min",
                    }
                )
            ),
            stage=3,
        )
        self.assertEqual(approved["jobs"][0]["overrides"]["old_critic_update"], "ema")

    def test_behaviorally_inactive_parameters_are_rejected_with_resolved_base(self) -> None:
        base = {
            "configs": {
                "Ant": {
                    "target_norm": "none",
                    "target_norm_update": "cumulative",
                    "target_norm_ema_tau": 0.99,
                    "old_critic_update": "hard",
                    "old_critic_ema_tau": 0.995,
                }
            }
        }
        self.assertRejected(
            plan_for(cohort(overrides={"target_norm_update": "ema"})),
            "target_norm=none",
            stage=2,
            base_configs=base,
        )
        self.assertRejected(
            plan_for(cohort(overrides={"old_critic_ema_tau": 0.9})),
            "inactive unless old_critic_update=ema",
            stage=3,
            base_configs=base,
        )

    def test_active_ema_parameters_are_allowed_when_switches_change_together(self) -> None:
        approved = validator.validate_plan(
            plan_for(
                cohort(
                    overrides={
                        "target_norm": "running_return",
                        "target_norm_update": "ema",
                        "target_norm_ema_tau": 0.97,
                        "old_critic_update": "ema",
                        "old_critic_ema_tau": 0.99,
                    }
                )
            ),
            stage=3,
            base_configs={"configs": {"Ant": {"target_norm": "none", "old_critic_update": "hard"}}},
        )
        overrides = approved["jobs"][0]["overrides"]
        self.assertEqual(overrides["target_norm_update"], "ema")
        self.assertEqual(overrides["old_critic_update"], "ema")

    def test_behavioral_history_signature_ignores_inactive_tau_values(self) -> None:
        first = {
            "cohorts": [
                {
                    "env": "Ant",
                    "resolved_tunables": {
                        "target_norm": "none",
                        "target_norm_update": "cumulative",
                        "target_norm_ema_tau": 0.99,
                        "old_critic_update": "hard",
                        "old_critic_ema_tau": 0.995,
                    },
                }
            ]
        }
        second = {
            "cohorts": [
                {
                    "env": "Ant",
                    "resolved_tunables": {
                        "target_norm": "none",
                        "target_norm_update": "ema",
                        "target_norm_ema_tau": 0.5,
                        "old_critic_update": "hard",
                        "old_critic_ema_tau": 0.1,
                    },
                }
            ]
        }
        self.assertEqual(
            validator.history_signatures(first),
            validator.history_signatures(second),
        )

    def test_parameter_range_and_non_numeric_bool_are_rejected(self) -> None:
        self.assertRejected(
            plan_for(cohort(overrides={"prior_std": 5.0})),
            "outside",
        )
        self.assertRejected(
            plan_for(cohort(overrides={"prior_std": True})),
            "JSON number",
        )

    def test_search_precision_and_grid_are_enforced(self) -> None:
        for key, value in (
            ("old_critic_ema_tau", 0.8001),
            ("prior_std", 0.8984375),
            ("cov_floor", 0.0225),
            ("cov_floor", 0.031),
        ):
            overrides = {key: value}
            if key == "old_critic_ema_tau":
                overrides["old_critic_update"] = "ema"
            self.assertRejected(
                plan_for(cohort(overrides=overrides)),
                "decimal places|0.005 search grid",
                stage=3,
            )

        approved = validator.validate_plan(
            plan_for(cohort(overrides={"cov_floor": 0.035, "prior_std": 0.995})),
            stage=1,
        )
        self.assertEqual(approved["jobs"][0]["overrides"]["cov_floor"], 0.035)

    def test_reward_scaler_is_precision_exempt(self) -> None:
        approved = validator.validate_plan(
            plan_for(cohort(overrides={"reward_scaler": 0.00271828})),
            stage=1,
        )
        self.assertEqual(
            approved["jobs"][0]["overrides"]["reward_scaler"],
            0.00271828,
        )

    def test_legacy_history_is_readable_but_resolved_candidate_must_migrate(self) -> None:
        history = {
            "cohorts": [
                {
                    "env": "Ant",
                    "resolved_tunables": {"prior_std": 0.8984375},
                }
            ]
        }
        self.assertTrue(validator.history_signatures(history))
        base = {"configs": {"Ant": {"prior_std": 0.8984375, "cov_floor": 0.02}}}
        self.assertRejected(
            plan_for(cohort(overrides={"cov_floor": 0.025})),
            "prior_std=.*exceeds 3 decimal places",
            base_configs=base,
        )
        approved = validator.validate_plan(
            plan_for(cohort(overrides={"prior_std": 0.9, "cov_floor": 0.025})),
            stage=1,
            base_configs=base,
        )
        self.assertEqual(approved["jobs"][0]["overrides"]["prior_std"], 0.9)

    def test_diversification_requires_a_real_change_outside_cooled_family(self) -> None:
        base = {
            "configs": {
                "Ant": {
                    "old_critic_update": "ema",
                    "old_critic_ema_tau": 0.8,
                    "cov_floor": 0.02,
                }
            }
        }
        constraints = {
            "diversification": {
                "Ant": {
                    "required": True,
                    "cooldown_families": ["critic_update"],
                }
            }
        }
        self.assertRejected(
            plan_for(cohort(overrides={"old_critic_ema_tau": 0.805})),
            "outside cooled families",
            stage=3,
            base_configs=base,
            search_constraints=constraints,
        )
        self.assertRejected(
            plan_for(
                cohort(
                    overrides={
                        "old_critic_ema_tau": 0.805,
                        "cov_floor": 0.02,
                    }
                )
            ),
            "outside cooled families",
            stage=3,
            base_configs=base,
            search_constraints=constraints,
        )
        approved = validator.validate_plan(
            plan_for(
                cohort(
                    overrides={
                        "old_critic_ema_tau": 0.805,
                        "cov_floor": 0.025,
                    }
                )
            ),
            stage=3,
            base_configs=base,
            search_constraints=constraints,
        )
        self.assertEqual(approved["jobs"][0]["overrides"]["cov_floor"], 0.025)

    def test_cov_weight_clip_must_be_ordered(self) -> None:
        jobs = cohort(overrides={"cov_weight_clip": [0.9, 0.2]})
        self.assertRejected(plan_for(jobs), "lower bound", stage=2)

    def test_cov_weight_clip_allows_an_open_bound(self) -> None:
        approved = validator.validate_plan(
            plan_for(cohort(overrides={"cov_weight_clip": [None, 1.0]})),
            stage=2,
        )
        self.assertEqual(approved["jobs"][0]["overrides"]["cov_weight_clip"], [None, 1.0])

    def test_incomplete_or_duplicate_seed_cohort_is_rejected(self) -> None:
        self.assertRejected(
            plan_for(cohort(seeds=(1, 2, 3))),
            "exactly one job for each seed",
        )
        self.assertRejected(
            plan_for(cohort(seeds=(1, 2, 3, 3))),
            "exactly one job for each seed",
        )

    def test_four_seed_jobs_must_be_identical_apart_from_seed(self) -> None:
        jobs = cohort()
        jobs[-1]["hypothesis"] = "A seed-specific hypothesis is not permitted."
        self.assertRejected(plan_for(jobs), "identical apart from seed")

    def test_at_most_two_variants_per_environment(self) -> None:
        jobs = []
        jobs += cohort(variant="candidate_a", overrides={"prior_std": 1.0})
        jobs += cohort(variant="candidate_b", overrides={"prior_std": 1.1})
        jobs += cohort(variant="candidate_c", overrides={"prior_std": 1.2})
        self.assertRejected(plan_for(jobs), "at most 2")

    def test_same_round_duplicate_config_is_rejected_even_with_new_name(self) -> None:
        jobs = cohort(variant="candidate_a") + cohort(variant="candidate_b")
        self.assertRejected(plan_for(jobs), "same-round configuration")

    def test_round_cannot_exceed_48_jobs(self) -> None:
        jobs = []
        for index, env in enumerate(validator.ENVIRONMENTS[:7]):
            jobs += cohort(env=env, variant="a", overrides={"prior_std": 0.8 + index * 0.01})
            jobs += cohort(env=env, variant="b", overrides={"prior_std": 1.2 + index * 0.01})
        self.assertGreater(len(jobs), 48)
        self.assertRejected(plan_for(jobs), "48-job")

    def test_historical_duplicate_is_rejected(self) -> None:
        overrides = {"prior_std": 1.1, "target_norm": "running_return", "cov_floor": 0.02}
        history = {"unique_cohorts": [{"env": "Ant", "overrides": overrides}]}
        self.assertRejected(
            plan_for(cohort(overrides=overrides)),
            "unique history",
            history=history,
        )

    def test_resolved_history_duplicate_uses_base_config(self) -> None:
        base = {
            "Ant": {
                "prior_std": 1.0,
                "target_norm": "none",
                "cov_floor": 0.0,
                "reward_scaler": 0.002,
            }
        }
        history = {
            "unique_cohorts": [
                {
                    "env": "Ant",
                    "resolved_tunables": {
                        "prior_std": 1.1,
                        "target_norm": "none",
                        "cov_floor": 0.0,
                        "reward_scaler": 0.002,
                    },
                }
            ]
        }
        self.assertRejected(
            plan_for(cohort(overrides={"prior_std": 1.1})),
            "unique history",
            history=history,
            base_configs=base,
        )

    def test_history_duplicate_signature_distinguishes_humanoid_gamma_contract(self) -> None:
        overrides = {"prior_std": 1.0, "target_norm": "none", "cov_floor": 0.0}
        jobs = cohort(env="Humanoid", overrides=overrides)
        old_gamma_history = {
            "cohorts": [
                {
                    "env": "Humanoid",
                    "resolved_tunables": overrides,
                    "fixed_contract": {"gamma": 0.98},
                }
            ]
        }
        approved = validator.validate_plan(
            plan_for(jobs),
            stage=1,
            history=old_gamma_history,
        )
        self.assertEqual(approved["jobs"][0]["env"], "Humanoid")

        current_gamma_history = {
            "cohorts": [
                {
                    "env": "Humanoid",
                    "resolved_tunables": overrides,
                    "fixed_contract": {"gamma": 0.99},
                }
            ]
        }
        self.assertRejected(
            plan_for(jobs),
            "unique history",
            stage=1,
            history=current_gamma_history,
        )

    def test_running_target_update_alias_cannot_evade_history_dedup(self) -> None:
        history = {
            "unique_cohorts": [
                {
                    "env": "Ant",
                    "overrides": {"target_norm_update": "cumulative"},
                }
            ]
        }
        self.assertRejected(
            plan_for(cohort(overrides={"target_norm_update": "running"})),
            "unique history",
            stage=2,
            history=history,
        )

    def test_target_envs_must_match_jobs(self) -> None:
        plan = plan_for(cohort())
        plan["target_envs"] = ["Humanoid"]
        self.assertRejected(plan, "exactly match")

    def test_stop_and_blocked_plans_must_not_schedule_jobs(self) -> None:
        stop = {
            "status": "stop",
            "target_envs": [],
            "reasoning": ["All target and integrity gates are satisfied."],
            "jobs": [],
        }
        self.assertEqual(validator.validate_plan(stop, stage=1)["status"], "stop")
        blocked_with_jobs = plan_for(cohort(), status="blocked")
        self.assertRejected(blocked_with_jobs, "must not schedule jobs")

    def test_schema_is_strict_and_lists_all_tunable_only(self) -> None:
        schema = json.loads((SCRIPTS_DIR / "next_plan.schema.json").read_text())
        self.assertFalse(schema["additionalProperties"])
        override_schema = schema["$defs"]["overrides"]
        self.assertFalse(override_schema["additionalProperties"])
        self.assertEqual(set(override_schema["properties"]), validator.ALL_TUNABLE_PARAMETERS)
        self.assertEqual(set(override_schema["required"]), set(override_schema["properties"]))
        self.assertNotIn("gamma", override_schema["properties"])

    def test_strict_schema_null_placeholders_become_sparse_stage_one_overrides(self) -> None:
        jobs = cohort()
        for job in jobs:
            sparse = dict(job["overrides"])
            job["overrides"] = {
                key: sparse.get(key)
                for key in validator.ALL_TUNABLE_PARAMETERS
            }
        approved = validator.validate_plan(plan_for(jobs), stage=1)
        self.assertEqual(
            approved["jobs"][0]["overrides"],
            {"cov_floor": 0.02, "prior_std": 1.1, "target_norm": "running_return"},
        )

    def test_validator_stage_sets_match_public_parameter_contract(self) -> None:
        contract = json.loads((PROJECT_ROOT / "config" / "parameter_contract.json").read_text())
        self.assertEqual(set(contract["environments"]), set(validator.ENVIRONMENTS))
        self.assertEqual(contract["fixed_parameters"], validator.FIXED_PARAMETERS)
        self.assertEqual(
            set(contract["search_stages"]["stage_1"]),
            validator.STAGE_1_PARAMETERS,
        )
        self.assertEqual(
            set(contract["search_stages"]["stage_2"]),
            validator.STAGE_2_PARAMETERS,
        )
        self.assertEqual(
            set(contract["search_stages"]["stage_3"]),
            validator.STAGE_3_PARAMETERS,
        )
        self.assertTrue(
            set(contract["dead_or_unused_keys"]).issubset(validator.DEAD_OR_UNUSED_PARAMETERS)
        )
        self.assertEqual(contract["search_precision_policy"]["max_decimal_places"], 3)
        self.assertEqual(contract["search_precision_policy"]["required_increment"], 0.005)
        self.assertEqual(
            set(contract["search_precision_policy"]["exempt_parameters"]),
            validator.SEARCH_PRECISION_EXEMPT_PARAMETERS,
        )

    def test_codex_command_is_pinned_read_only_ultra_and_schema_bound(self) -> None:
        output = PROJECT_ROOT / "experiments" / "round_x" / "round_gpt_output.json"
        command = call_gpt_planner.build_codex_command(
            codex_bin="/usr/local/bin/codex",
            output_path=output,
        )
        self.assertIn("gpt-5.6-sol", command)
        self.assertIn("read-only", command)
        self.assertIn('model_reasoning_effort="ultra"', command)
        self.assertIn("--disable", command)
        self.assertEqual(command[command.index("--disable") + 1], "multi_agent")
        self.assertIn("--ephemeral", command)
        self.assertIn("--ignore-user-config", command)
        self.assertIn("--ignore-rules", command)
        self.assertEqual(command[command.index("--output-schema") + 1], str(validator.SCHEMA_PATH))
        self.assertEqual(command[command.index("-o") + 1], str(output))
        self.assertEqual(call_gpt_planner.DEFAULT_MAX_RETRIES, 3)

    def test_stage_four_prompt_declares_strict_global_unification_contract(self) -> None:
        prompt = call_gpt_planner.build_prompt(
            stage=3,
            context_paths=[],
            unification_mode="global",
        )
        self.assertIn("Stage 4 unification mode is `global`", prompt)
        self.assertIn("exactly 1 distinct overrides object", prompt)
        self.assertIn("reward_scaler, acf_coef, vf_coef, and flow_sigma", prompt)
        self.assertIn("three decimal places", prompt)
        self.assertIn("0.005 grid", prompt)
        self.assertIn("diversification", prompt)

    def test_planner_context_is_compacted_under_codex_input_budget(self) -> None:
        with tempfile.TemporaryDirectory(prefix=".planner_context_", dir=PROJECT_ROOT / "tests") as temp:
            root = Path(temp)
            history = {
                "schema_version": 1,
                "unique_config_count": 300,
                "cohorts": [
                    {
                        "env": "Ant",
                        "variant": f"candidate_{index}",
                        "round_id": f"round_{index:03d}",
                        "config_signature": f"sig-{index}",
                        "paper_ratio": 0.8 + (index % 10) / 100.0,
                        "top5_mean": 100.0 + index,
                        "top5_std": 1.0,
                        "fixed_contract": {"gamma": 0.99, "padding": "x" * 500},
                        "resolved_tunables": {"prior_std": 1.1, "cov_floor": 0.02},
                    }
                    for index in range(300)
                ],
            }
            metric_payload = {
                f"unused/{index}": {"last": float(index)}
                for index in range(40)
            }
            metric_payload.update(
                {
                    "rollout/weight_ess_ratio": {"min": 0.95, "last": 0.99},
                    "train/value_loss": {"mean": 1.0},
                }
            )
            diagnostics = {
                "integrity_ok": True,
                "fatal_issues": [],
                "jobs": [
                    {
                        "env": "Ant",
                        "variant": "candidate",
                        "seed": 1,
                        "fatal_issues": [],
                        "stdout_anomalies": [],
                        "tail_metrics": {
                            "metrics": metric_payload
                        },
                    }
                ],
            }
            constraints = {"must_target_exactly": ["Ant"]}
            history_path = root / "planner_history_index.json"
            diagnostics_path = root / "diagnostics.json"
            constraints_path = root / "loop_constraints.json"
            history_path.write_text(json.dumps(history), encoding="utf-8")
            diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")
            constraints_path.write_text(json.dumps(constraints), encoding="utf-8")

            prompt, report = call_gpt_planner.build_prompt_and_report(
                stage=3,
                context_paths=[history_path, diagnostics_path, constraints_path],
            )
            self.assertLessEqual(
                len(prompt), call_gpt_planner.MAX_PLANNER_INPUT_CHARS
            )
            self.assertTrue(report["compacted"])
            self.assertNotIn('"fixed_contract"', prompt)
            self.assertIn("compact_planner_history", prompt)

    def test_input_size_error_is_classified_as_deterministic(self) -> None:
        self.assertTrue(
            call_gpt_planner._is_input_size_error(
                "turn/start failed: Input exceeds the maximum length 1048576"
            )
        )
        self.assertFalse(call_gpt_planner._is_input_size_error("temporary network failure"))

    def test_planner_command_disables_multi_agent_runtime(self) -> None:
        command = call_gpt_planner.build_codex_command(
            codex_bin="/usr/local/bin/codex",
            output_path=PROJECT_ROOT / "experiments" / "round_x" / "round_gpt_output.json",
        )
        assert command[command.index("--disable") + 1] == "multi_agent"
        assert "--ignore-rules" in command

    def test_direct_api_planner_sends_no_tools_and_writes_json(self) -> None:
        class FakeResponse:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(
                    {
                        "id": "probe",
                        "choices": [
                            {
                                "message": {
                                    "content": '{"status":"blocked","target_envs":[],"reasoning":["probe"],"jobs":[]}'
                                }
                            }
                        ],
                    }
                ).encode()

        captured = {}

        def fake_urlopen(request, timeout):
            captured["body"] = json.loads(request.data.decode())
            captured["timeout"] = timeout
            return FakeResponse()

        with tempfile.TemporaryDirectory(prefix=".planner_api_test_", dir=PROJECT_ROOT / "tests") as temp:
            output = Path(temp) / "planner.json"
            with mock.patch.object(call_gpt_planner.urllib.request, "urlopen", side_effect=fake_urlopen), mock.patch.object(
                call_gpt_planner, "_api_key", return_value="test-key"
            ), mock.patch.object(call_gpt_planner, "_api_endpoint", return_value="https://example.test/chat/completions"):
                code, error, metadata = call_gpt_planner.run_api_planner(
                    prompt="Return JSON.", output_path=output, timeout=7.0, attempt=1
                )
            assert code == 0
            assert error == ""
            assert metadata["transport"] == "openai_compatible_chat"
            assert "tools" not in captured["body"]
            assert captured["body"]["reasoning"] == {"effort": "ultra"}
            assert captured["body"]["response_format"]["type"] == "json_schema"
            assert captured["body"]["response_format"]["json_schema"]["strict"] is True
            assert set(captured["body"]["response_format"]["json_schema"]["schema"]["required"]) == {
                "status",
                "target_envs",
                "reasoning",
                "jobs",
            }
            assert json.loads(output.read_text())["status"] == "blocked"

    def test_planner_retries_three_times_then_accepts_without_real_codex(self) -> None:
        with tempfile.TemporaryDirectory(prefix=".planner_test_", dir=PROJECT_ROOT / "tests") as temp:
            round_dir = Path(temp)
            calls = []
            planner_envs = []

            def fake_run(command, **kwargs):
                calls.append(command)
                planner_envs.append(dict(kwargs["env"]))
                output = Path(command[command.index("-o") + 1])
                if len(calls) <= 3:
                    output.write_text(json.dumps({"invalid": True}), encoding="utf-8")
                else:
                    output.write_text(
                        json.dumps(
                            {
                                "status": "blocked",
                                "target_envs": [],
                                "reasoning": ["No authorized result context is available."],
                                "jobs": [],
                            }
                        ),
                        encoding="utf-8",
                    )
                return call_gpt_planner.subprocess.CompletedProcess(command, 0, "", "")

            with mock.patch.object(
                call_gpt_planner.subprocess, "run", side_effect=fake_run
            ), mock.patch.dict(
                call_gpt_planner.os.environ,
                {"LD_LIBRARY_PATH": "/bad/conda/lib", "LD_PRELOAD": "/bad/preload.so"},
            ), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                exit_code = call_gpt_planner.main(
                    [
                        str(round_dir),
                        "--stage",
                        "1",
                        "--codex-bin",
                        sys.executable,
                        "--transport",
                        "cli",
                        "--timeout-sec",
                        "1",
                    ]
                )
            self.assertEqual(exit_code, 0)
            self.assertEqual(len(calls), 4)
            status = json.loads((round_dir / "gpt_planner_status.json").read_text())
            self.assertEqual(status["attempt_count"], 4)
            self.assertEqual(status["status"], "approved")
            self.assertTrue((round_dir / "next_plan.json").is_file())
            for command in calls:
                self.assertIn("read-only", command)
                self.assertIn("gpt-5.6-sol", command)
            for environment in planner_envs:
                self.assertNotIn("LD_LIBRARY_PATH", environment)
                self.assertNotIn("LD_PRELOAD", environment)

    def test_failed_planner_never_fabricates_a_fallback_plan(self) -> None:
        with tempfile.TemporaryDirectory(prefix=".planner_test_", dir=PROJECT_ROOT / "tests") as temp:
            round_dir = Path(temp)

            def fake_run(command, **kwargs):
                output = Path(command[command.index("-o") + 1])
                output.write_text(json.dumps({"invalid": True}), encoding="utf-8")
                return call_gpt_planner.subprocess.CompletedProcess(command, 0, "", "")

            with mock.patch.object(
                call_gpt_planner.subprocess, "run", side_effect=fake_run
            ), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                exit_code = call_gpt_planner.main(
                    [
                        str(round_dir),
                        "--stage",
                        "1",
                        "--codex-bin",
                        sys.executable,
                        "--timeout-sec",
                        "1",
                        "--max-retries",
                        "0",
                    ]
                )
            self.assertEqual(exit_code, 3)
            self.assertFalse((round_dir / "next_plan.json").exists())
            status = json.loads((round_dir / "gpt_planner_status.json").read_text())
            self.assertEqual(status["status"], "failed")

    def test_cli_writes_auditable_contract_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_path = root / "raw.json"
            output_path = root / "next.json"
            record_path = root / "record.json"
            raw_path.write_text(json.dumps(plan_for(cohort())), encoding="utf-8")
            exit_code = validator.main(
                [
                    str(raw_path),
                    "--stage",
                    "1",
                    "--output",
                    str(output_path),
                    "--record",
                    str(record_path),
                ]
            )
            self.assertEqual(exit_code, 0)
            record = json.loads(record_path.read_text())
            self.assertTrue(record["valid"])
            self.assertEqual(record["contract"]["fixed_parameters"]["gamma"], 0.99)
            self.assertIn("prior_std", record["contract"]["parameter_specs"])
            self.assertTrue(output_path.is_file())


if __name__ == "__main__":
    unittest.main()
