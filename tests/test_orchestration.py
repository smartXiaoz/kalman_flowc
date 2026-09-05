from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import common  # noqa: E402
import continuous_tune  # noqa: E402
import launch_round  # noqa: E402
import materialize_round  # noqa: E402
import process_identity  # noqa: E402


def test_round_000_materializes_48_jobs_with_paper_contract(tmp_path: Path) -> None:
    round_dir = materialize_round.materialize(
        PROJECT_ROOT / "config" / "round_000_plan.json",
        None,
        tmp_path,
        "RLS",
        ["0", "1", "2", "3"],
        False,
    )
    manifest = common.read_json(round_dir / "manifest.json")
    assert len(manifest["jobs"]) == 48
    assert manifest["purpose"] == "tuning"
    assert {job["seed"] for job in manifest["jobs"]} == {1, 2, 3, 4}
    assert {job["env"] for job in manifest["jobs"]} == set(common.ENVIRONMENTS)
    for job in manifest["jobs"]:
        config = common.read_json(job["config_path"])
        for key, expected in common.FIXED_PARAMETERS.items():
            assert config[key] == expected
        assert config["gamma"] == 0.99
        expected_budget = 80_000_000 if job["env"] in {"Anymal", "FrankaCabinet"} else 100_000_000
        assert config["total_timesteps"] == expected_budget
        assert job["expected_final_step"] == common.expected_final_step(expected_budget)


def test_delta_plan_inherits_accepted_base_tunables(tmp_path: Path) -> None:
    jobs = [
        {
            "env": "Ant",
            "seed": seed,
            "variant": "delta",
            "overrides": {"reward_scaler": 0.003},
            "hypothesis": "Change only reward scale.",
            "evidence": "Synthetic contract test.",
        }
        for seed in common.SEEDS
    ]
    plan = {
        "status": "continue",
        "target_envs": ["Ant"],
        "reasoning": ["test"],
        "jobs": jobs,
        "round_id": "round_009_delta",
        "stage": 1,
    }
    plan_path = tmp_path / "plan.json"
    common.write_json_atomic(plan_path, plan)
    base_path = tmp_path / "base.json"
    common.write_json_atomic(
        base_path,
        {"configs": {"Ant": {"prior_std": 1.23, "target_norm": "running_return", "cov_floor": 0.03}}},
    )
    round_dir = materialize_round.materialize(
        plan_path,
        None,
        tmp_path / "experiments",
        "RLS",
        ["0", "1", "2", "3"],
        False,
        base_path,
    )
    config = common.read_json(common.read_json(round_dir / "manifest.json")["jobs"][0]["config_path"])
    assert config["prior_std"] == 1.23
    assert config["target_norm"] == "running_return"
    assert config["cov_floor"] == 0.03
    assert config["reward_scaler"] == 0.003


def test_materializer_rejects_out_of_range_values_and_stage_four(tmp_path: Path) -> None:
    base_job = {
        "env": "Ant",
        "variant": "invalid",
        "hypothesis": "Contract test.",
        "evidence": "Contract test.",
    }
    plan = {
        "status": "continue",
        "target_envs": ["Ant"],
        "reasoning": ["test"],
        "round_id": "round_010_invalid",
        "stage": 1,
        "jobs": [
            {**base_job, "seed": seed, "overrides": {"prior_std": 5.0}}
            for seed in common.SEEDS
        ],
    }
    with pytest.raises(ValueError, match="outside"):
        materialize_round.validate_materialization_input(plan)
    plan["stage"] = 4
    with pytest.raises(ValueError, match="stage must be"):
        materialize_round.validate_materialization_input(plan)
    plan["stage"] = 2
    plan["purpose"] = "unification_global"
    with pytest.raises(ValueError, match="must use the complete Stage 3"):
        materialize_round.validate_materialization_input(plan)


def test_materialization_is_transactional_on_mid_write_failure(tmp_path: Path, monkeypatch) -> None:
    original = materialize_round.write_json_atomic
    calls = 0

    def fail_after_first_write(path, payload):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic write failure")
        return original(path, payload)

    monkeypatch.setattr(materialize_round, "write_json_atomic", fail_after_first_write)
    with pytest.raises(OSError, match="synthetic"):
        materialize_round.materialize(
            PROJECT_ROOT / "config" / "round_000_plan.json",
            None,
            tmp_path,
            "RLS",
            ["0", "1", "2", "3"],
            False,
        )
    assert not (tmp_path / "round_000_baseline").exists()
    assert not list(tmp_path.glob(".round_000_baseline.materializing.*"))


def test_dry_run_assigns_six_of_first_wave_to_each_gpu(tmp_path: Path) -> None:
    round_dir = materialize_round.materialize(
        PROJECT_ROOT / "config" / "round_000_plan.json",
        None,
        tmp_path,
        "RLS",
        ["0", "1", "2", "3"],
        False,
    )
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "launch_round.py"),
            str(round_dir),
            "--dry-run",
            "--gpus",
            "0,1,2,3",
            "--slots-per-gpu",
            "6",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    commands = common.read_json(round_dir / "dry_run_commands.json")
    assert len(commands) == 48
    first_wave = commands[:24]
    assert {gpu: sum(row["gpu"] == gpu for row in first_wave) for gpu in "0123"} == {
        "0": 6,
        "1": 6,
        "2": 6,
        "3": 6,
    }
    for row in commands:
        command = row["command"]
        assert "--gpu-id" in command
        assert "CUDA_VISIBLE_DEVICES" not in " ".join(command)
        assert "--reward-log-path" in command


def test_fcs_teardown_requires_explicit_exit_code(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    log = tmp_path / "stdout.log"
    summary.write_text(json.dumps({"global_step": 99_991_552, "completed_budget": True}), encoding="utf-8")
    log.write_text("训练结束\n段错误（核心已转储）\n", encoding="utf-8")
    job = {
        "env": "FrankaCubeStack",
        "summary_json": str(summary),
        "expected_final_step": 99_991_552,
        "stdout_log": str(log),
    }
    unknown_state, _, _ = launch_round.terminal_state(job, None)
    assert unknown_state == "failed_exit_unknown"
    state, message, retry = launch_round.terminal_state(job, 139)
    assert state == "complete_with_teardown_warning"
    assert "fixed budget" in message
    assert retry is False


def test_launcher_rejects_modified_resolved_config(tmp_path: Path) -> None:
    round_dir = materialize_round.materialize(
        PROJECT_ROOT / "config" / "round_000_plan.json",
        None,
        tmp_path,
        "RLS",
        ["0", "1", "2", "3"],
        False,
    )
    manifest = common.read_json(round_dir / "manifest.json")
    job = manifest["jobs"][0]
    config = common.read_json(job["config_path"])
    config["gamma"] = 0.98
    common.write_json_atomic(job["config_path"], config)
    with pytest.raises(RuntimeError, match="hash mismatch"):
        launch_round.verify_integrity(job, manifest)


def test_build_command_uses_configured_conda(monkeypatch) -> None:
    monkeypatch.setenv("FLOWCRITIC_CONDA_EXE", "/opt/conda/bin/conda")
    job = {
        "job_id": "job",
        "config_path": "/tmp/config.json",
        "run_name": "round/job",
        "metrics_jsonl": "/tmp/metrics.jsonl",
        "status_json": "/tmp/status.json",
        "summary_json": "/tmp/summary.json",
        "rewards_path": "/tmp/rewards.txt",
        "job_run_dir": "/tmp/job",
    }
    manifest = {
        "round_id": "round",
        "conda_env": "RLS",
        "script": "/tmp/train.py",
    }

    command = launch_round.build_command(job, manifest, "0")

    assert command[:5] == [
        "/opt/conda/bin/conda",
        "run",
        "--no-capture-output",
        "-n",
        "RLS",
    ]


def test_resume_launches_only_unfinished_job(tmp_path: Path) -> None:
    round_dir = tmp_path / "round_resume"
    runs_dir = round_dir / "runs"
    source_dir = tmp_path / "src"
    source_dir.mkdir(parents=True)
    fake_trainer = source_dir / "fake_trainer.py"
    fake_trainer.write_text(
        "import argparse, json\n"
        "from pathlib import Path\n"
        "p=argparse.ArgumentParser()\n"
        "for name in ('config-path','run-name','gpu-id','research-tag','metrics-jsonl','status-json','summary-json','reward-log-path','early-stop-file'):\n"
        "    p.add_argument('--'+name)\n"
        "p.add_argument('--save-minimal-logs', action='store_true')\n"
        "a=p.parse_args()\n"
        "cfg=json.loads(Path(a.config_path).read_text())\n"
        "step=(int(cfg['total_timesteps'])//16384)*16384\n"
        "Path(a.metrics_jsonl).write_text(json.dumps({'global_step':step})+'\\n')\n"
        "Path(a.reward_log_path).write_text('step,episodic_return\\n'+str(step)+',1.0\\n')\n"
        "payload={'state':'finished','global_step':step,'completed_budget':True}\n"
        "Path(a.status_json).write_text(json.dumps(payload))\n"
        "Path(a.summary_json).write_text(json.dumps(payload))\n",
        encoding="utf-8",
    )

    jobs = []
    state_jobs = {}
    for index, name in enumerate(("finished", "unfinished"), 1):
        job_dir = runs_dir / name
        job_dir.mkdir(parents=True)
        config_path = job_dir / "config.resolved.json"
        config_path.write_text(json.dumps({"total_timesteps": 16_384}), encoding="utf-8")
        job = {
            "job_id": name,
            "env": "Ant",
            "seed": index,
            "variant": "resume",
            "config_path": str(config_path),
            "job_run_dir": str(job_dir),
            "run_name": f"resume/{name}",
            "metrics_jsonl": str(job_dir / "metrics.jsonl"),
            "status_json": str(job_dir / "status.json"),
            "summary_json": str(job_dir / "summary.json"),
            "rewards_path": str(job_dir / "rewards.txt"),
            "stdout_log": str(job_dir / "stdout.log"),
            "expected_final_step": 16_384,
        }
        jobs.append(job)
        state_jobs[name] = {"state": "complete" if name == "finished" else "pending", "attempt": 1 if name == "finished" else 0}

    finished = jobs[0]
    Path(finished["summary_json"]).write_text(
        json.dumps({"global_step": 16_384, "completed_budget": True}), encoding="utf-8"
    )
    Path(finished["stdout_log"]).write_text("sentinel-complete-log\n", encoding="utf-8")
    sentinel_hash = launch_round.log_contains_oom(finished["stdout_log"])
    manifest = {
        "round_id": "round_resume",
        "script": str(fake_trainer),
        "conda_env": "RLS",
        "jobs": jobs,
    }
    common.write_json_atomic(round_dir / "manifest.json", manifest)
    common.write_json_atomic(
        round_dir / "round_state.json",
        {"round_id": "round_resume", "state": "materialized", "jobs": state_jobs},
    )
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "launch_round.py"),
            str(round_dir),
            "--skip-gpu-preflight",
            "--poll-sec",
            "0.1",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    final_state = common.read_json(round_dir / "round_state.json")
    assert final_state["state"] == "complete"
    assert final_state["launcher_identity"] is None
    assert final_state["last_launcher_identity"]["boot_id"] == process_identity.current_boot_id()
    assert final_state["jobs"]["finished"]["attempt"] == 1
    assert final_state["jobs"]["unfinished"]["attempt"] == 1
    assert Path(finished["stdout_log"]).read_text(encoding="utf-8") == "sentinel-complete-log\n"
    assert not (Path(finished["job_run_dir"]) / "attempts").exists()
    assert sentinel_hash is False


def test_current_configs_shape_is_accepted_as_base_map(tmp_path: Path, monkeypatch) -> None:
    destination = tmp_path / "current_configs.json"
    monkeypatch.setattr(continuous_tune, "CURRENT_CONFIGS_PATH", destination)
    state = {
        "incumbents": {
            "Ant": {
                "tunable_config": {"prior_std": 1.2, "target_norm": "running_return"},
                "round_id": "round_000_baseline",
                "variant": "known_best",
                "top5_mean": 9000.0,
                "ratio": 0.96,
            }
        }
    }
    continuous_tune.write_current_configs(state)
    payload = common.read_json(destination)
    assert payload["configs"]["Ant"]["prior_std"] == 1.2
    assert payload["metadata"]["Ant"]["variant"] == "known_best"


def test_materializer_rejects_high_precision_override_and_legacy_resolved_base(
    tmp_path: Path,
) -> None:
    jobs = [
        {
            "env": "Ant",
            "seed": seed,
            "variant": "precision",
            "overrides": {"prior_std": 1.2345},
            "hypothesis": "Synthetic precision rejection.",
            "evidence": "Contract test.",
        }
        for seed in common.SEEDS
    ]
    plan = {
        "status": "continue",
        "target_envs": ["Ant"],
        "reasoning": ["test"],
        "round_id": "round_precision_reject",
        "stage": 1,
        "jobs": jobs,
    }
    with pytest.raises(ValueError, match="exceeds 3 decimal places"):
        materialize_round.validate_materialization_input(plan)

    for job in jobs:
        job["overrides"] = {"cov_floor": 0.025}
    plan_path = tmp_path / "plan.json"
    common.write_json_atomic(plan_path, plan)
    base_path = tmp_path / "legacy.json"
    common.write_json_atomic(
        base_path,
        {"configs": {"Ant": {"prior_std": 0.8984375}}},
    )
    with pytest.raises(ValueError, match="prior_std=.*exceeds 3 decimal places"):
        materialize_round.materialize(
            plan_path,
            None,
            tmp_path / "experiments",
            "RLS",
            ["0", "1", "2", "3"],
            False,
            base_path,
        )


def test_stage_transitions_preserve_cross_stage_stagnation_counts() -> None:
    state = {
        "stage": 1,
        "rounds_completed": 2,
        "env_stagnation_counts": {"Ant": 1},
        "incumbents": {},
    }
    assert continuous_tune.choose_next_stage(state) == 2
    assert state["env_stagnation_counts"] == {"Ant": 1}
    state["env_stagnation_counts"]["Ant"] = 2
    assert continuous_tune.choose_next_stage(state) == 3
    assert state["env_stagnation_counts"] == {"Ant": 2}


def test_target_gate_requires_precision_compliant_incumbents() -> None:
    state = {
        "incumbents": {
            env: {
                "ratio": 1.0,
                "tunable_config": {"prior_std": 1.0},
            }
            for env in common.ENVIRONMENTS
        }
    }
    state["incumbents"]["FrankaCabinet"]["tunable_config"]["prior_std"] = 0.8984375
    assert continuous_tune.all_targets_met(state) is False
    assert "FrankaCabinet" in continuous_tune.underperforming_envs(state)
    state["incumbents"]["FrankaCabinet"]["tunable_config"]["prior_std"] = 0.9
    assert continuous_tune.all_targets_met(state) is True


def test_precision_migration_can_replace_legacy_incumbent_within_five_percent(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        continuous_tune,
        "load_manifest_configs",
        lambda round_dir: {("FrankaCabinet", "rounded"): {"prior_std": 0.9}},
    )
    state = {
        "incumbents": {
            "FrankaCabinet": {
                "top5_mean": 100.0,
                "ratio": 1.0,
                "passed_90": True,
                "tunable_config": {"prior_std": 0.8984375},
            }
        }
    }
    score = {
        "results": [
            {
                "complete": True,
                "env": "FrankaCabinet",
                "variant": "rounded",
                "top5_mean": 96.0,
                "top5_std": 1.0,
                "all_bin_mean": 90.0,
                "all_bin_std": 1.0,
                "paper_mean": 100.0,
                "ratio": 0.96,
                "passed_90": True,
                "seeds": [],
            }
        ]
    }
    _, decisions = continuous_tune.update_incumbents(state, tmp_path, score)
    assert decisions[0]["accepted"] is True
    assert decisions[0]["reason"] == "precision_policy_migration_within_5pct_retention"
    assert state["incumbents"]["FrankaCabinet"]["tunable_config"]["prior_std"] == 0.9


def test_diversification_constraints_cool_repeated_parameter_family(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(continuous_tune, "EXPERIMENTS_DIR", tmp_path)
    history = []
    for index in range(3):
        round_id = f"round_{index:03d}_test"
        round_dir = tmp_path / round_id
        round_dir.mkdir()
        common.write_json_atomic(
            round_dir / "manifest.json",
            {
                "jobs": [
                    {
                        "env": "Ant",
                        "overrides": {"old_critic_ema_tau": 0.8 + index * 0.005},
                    }
                ]
            },
        )
        history.append({"round_id": round_id, "purpose": "tuning"})
    constraints = continuous_tune.diversification_constraints(
        {
            "round_history": history,
            "env_stagnation_counts": {"Ant": 3},
        },
        ["Ant"],
    )
    assert constraints["Ant"]["required"] is True
    assert constraints["Ant"]["cooldown_families"] == ["critic_update"]
    assert constraints["Ant"]["cooldown_exhausted_reset"] is False
    assert constraints["Ant"]["released_families"] == []


def test_diversification_releases_least_used_family_when_all_are_cooled(
    tmp_path: Path, monkeypatch
) -> None:
    """An exhausted cooldown set must leave at least one legal family."""

    monkeypatch.setattr(continuous_tune, "EXPERIMENTS_DIR", tmp_path)
    all_family_keys = {
        "distribution": "prior_std",
        "normalization": "target_norm",
        "weight_stability": "ess_floor",
        "loss_balance": "vf_coef",
        "critic_update": "old_critic_update",
        "bootstrap": "bootstrap_mean_mode",
    }
    history = []
    # Two rounds touch every family; the newest touches distribution once
    # more.  Thus distribution has count 3 and all other families count 2.
    for index, families in enumerate(
        (tuple(all_family_keys), tuple(all_family_keys), ("distribution",))
    ):
        round_id = f"round_{index:03d}_all_families"
        round_dir = tmp_path / round_id
        round_dir.mkdir()
        common.write_json_atomic(
            round_dir / "manifest.json",
            {
                "jobs": [
                    {
                        "env": "Humanoid",
                        "overrides": {
                            all_family_keys[family]: index
                            for family in families
                        },
                    }
                ]
            },
        )
        history.append({"round_id": round_id, "purpose": "tuning"})

    constraints = continuous_tune.diversification_constraints(
        {
            "round_history": history,
            "env_stagnation_counts": {"Humanoid": 3},
        },
        ["Humanoid"],
    )["Humanoid"]

    assert constraints["cooldown_exhausted_reset"] is True
    assert constraints["released_families"] == [
        "bootstrap",
        "critic_update",
        "loss_balance",
        "normalization",
        "weight_stability",
    ]
    assert constraints["cooldown_families_before_reset"] == [
        "bootstrap",
        "critic_update",
        "distribution",
        "loss_balance",
        "normalization",
        "weight_stability",
    ]
    assert constraints["cooldown_families"] == ["distribution"]
    assert constraints["required"] is True
    assert constraints["family_search_counts"]["distribution"] == 3


def test_diversification_fully_resets_when_all_family_counts_tie(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(continuous_tune, "EXPERIMENTS_DIR", tmp_path)
    keys = {
        "distribution": "prior_std",
        "normalization": "target_norm",
        "weight_stability": "ess_floor",
        "loss_balance": "vf_coef",
        "critic_update": "old_critic_update",
        "bootstrap": "bootstrap_mean_mode",
    }
    history = []
    for index in range(3):
        round_id = f"round_{index:03d}_tied_families"
        round_dir = tmp_path / round_id
        round_dir.mkdir()
        common.write_json_atomic(
            round_dir / "manifest.json",
            {
                "jobs": [
                    {
                        "env": "Humanoid",
                        "overrides": {key: index for key in keys.values()},
                    }
                ]
            },
        )
        history.append({"round_id": round_id, "purpose": "tuning"})

    constraints = continuous_tune.diversification_constraints(
        {
            "round_history": history,
            "env_stagnation_counts": {"Humanoid": 3},
        },
        ["Humanoid"],
    )["Humanoid"]

    assert constraints["cooldown_exhausted_reset"] is True
    assert constraints["released_families"] == sorted(keys)
    assert constraints["cooldown_families"] == []
    assert constraints["required"] is False


def test_round_writes_explicit_historical_unique_cohort_comparison(
    tmp_path: Path, monkeypatch
) -> None:
    history_path = tmp_path / "history.json"
    common.write_json_atomic(
        history_path,
        {
            "cohorts": [
                {
                    "env": "Ant",
                    "variant": "old",
                    "round_id": "legacy_round",
                    "top5_mean": 90.0,
                    "top5_std": 2.0,
                    "config_signature": "abc",
                }
            ]
        },
    )
    monkeypatch.setattr(continuous_tune, "HISTORY_PATH", history_path)
    round_dir = tmp_path / "round_001"
    round_dir.mkdir()
    comparison = continuous_tune.build_historical_comparison(
        round_dir,
        {
            "results": [
                {
                    "env": "Ant",
                    "variant": "new",
                    "complete": True,
                    "top5_mean": 100.0,
                }
            ]
        },
    )
    row = comparison[("Ant", "new")]
    assert row["historical_best_top5_mean"] == 90.0
    assert row["delta_vs_historical_best"] == 10.0
    assert row["better_than_historical_best"] is True
    assert (round_dir / "historical_unique_comparison.json").is_file()
    assert "legacy_round" in (round_dir / "historical_unique_comparison.md").read_text()


def test_planner_history_index_keeps_signatures_but_drops_large_reward_audit(
    tmp_path: Path, monkeypatch
) -> None:
    destination = tmp_path / "planner_history_index.json"
    monkeypatch.setattr(continuous_tune, "PLANNER_HISTORY_PATH", destination)
    continuous_tune.write_planner_history_index(
        {
            "cohorts": [
                {
                    "env": "Ant",
                    "variant": "candidate",
                    "round_id": "round_001",
                    "resolved_tunables": {"prior_std": 1.1},
                    "fixed_contract": {"gamma": 0.99},
                    "config_signature": "signature",
                    "top5_mean": 100.0,
                    "reward_sha256": {"1": "x" * 64},
                    "duplicate_observations": [{"large": "y" * 10000}],
                }
            ]
        }
    )
    payload = common.read_json(destination)
    assert payload["unique_config_count"] == 1
    assert payload["cohorts"][0]["config_signature"] == "signature"
    assert "reward_sha256" not in payload["cohorts"][0]
    assert "duplicate_observations" not in payload["cohorts"][0]


def test_continuous_unification_acceptance_updates_all_final_configs(
    tmp_path: Path, monkeypatch
) -> None:
    round_dir = tmp_path / "round_010_unify_global"
    jobs = []
    for env in common.ENVIRONMENTS:
        run_dir = round_dir / "runs" / env
        run_dir.mkdir(parents=True)
        config_path = run_dir / "config.resolved.json"
        common.write_json_atomic(
            config_path,
            {
                "env_id": env,
                "prior_std": 1.1,
                "target_norm": "running_return",
                "cov_floor": 0.02,
            },
        )
        jobs.append(
            {
                "job_id": env,
                "env": env,
                "variant": "unified",
                "config_path": str(config_path),
            }
        )
    common.write_json_atomic(
        round_dir / "manifest.json",
        {"round_id": round_dir.name, "purpose": "unification_global", "jobs": jobs},
    )
    targets_path = tmp_path / "targets.json"
    common.write_json_atomic(
        targets_path,
        {"environments": {env: {"paper_mean": 100.0} for env in common.ENVIRONMENTS}},
    )
    monkeypatch.setattr(continuous_tune, "PAPER_TARGETS_PATH", targets_path)
    monkeypatch.setattr(continuous_tune, "CURRENT_CONFIGS_PATH", tmp_path / "current_configs.json")
    incumbents = {
        env: {
            "top5_mean": 100.0,
            "top5_std": 1.0,
            "ratio": 1.0,
            "tunable_config": {"prior_std": 1.0},
        }
        for env in common.ENVIRONMENTS
    }
    state = {
        "incumbents": {env: dict(value) for env, value in incumbents.items()},
        "unification": {
            "state": "global_materialized",
            "attempts": [],
            "reference_incumbents": incumbents,
        },
    }
    score_payload = {
        "results": [
            {
                "env": env,
                "variant": "unified",
                "complete": True,
                "top5_mean": 100.0,
                "top5_std": 1.0,
                "all_bin_mean": 90.0,
                "all_bin_std": 2.0,
                "paper_mean": 100.0,
                "ratio": 1.0,
                "passed_90": True,
                "seeds": [],
            }
            for env in common.ENVIRONMENTS
        ]
    }
    report = continuous_tune.evaluate_unification_round(
        state, round_dir, score_payload, "global"
    )
    assert report["accepted"] is True
    assert state["unification"]["state"] == "global_accepted"
    assert set(state["incumbents"]) == set(common.ENVIRONMENTS)
    assert all(
        record["tunable_config"]["prior_std"] == 1.1
        for record in state["incumbents"].values()
    )


def test_schedule_global_unification_materializes_a_stage_three_round(
    tmp_path: Path, monkeypatch
) -> None:
    source_round = tmp_path / "round_009_tuning"
    source_round.mkdir()
    experiments = tmp_path / "experiments"
    history = tmp_path / "history.json"
    current = tmp_path / "current.json"
    common.write_json_atomic(history, {"cohorts": []})
    common.write_json_atomic(current, {"configs": {}})
    monkeypatch.setattr(continuous_tune, "EXPERIMENTS_DIR", experiments)
    monkeypatch.setattr(continuous_tune, "HISTORY_PATH", history)
    monkeypatch.setattr(continuous_tune, "PLANNER_HISTORY_PATH", history)
    monkeypatch.setattr(continuous_tune, "CURRENT_CONFIGS_PATH", current)
    monkeypatch.setattr(continuous_tune, "LOOP_STATE_PATH", tmp_path / "loop_state.json")

    shared = {
        "prior_std": 1.1,
        "target_norm": "running_return",
        "cov_floor": 0.02,
        "target_norm_update": "cumulative",
        "target_norm_ema_tau": 0.99,
        "cov_weight_clip": None,
        "ess_floor": 0.0,
        "alpha_warmup_updates": 0,
        "old_critic_update": "hard",
        "old_critic_ema_tau": 0.995,
        "bootstrap_mean_mode": "normal",
    }
    plan = {
        "status": "continue",
        "target_envs": list(common.ENVIRONMENTS),
        "reasoning": ["Synthetic Stage 4 scheduler test."],
        "jobs": [
            {
                "env": env,
                "seed": seed,
                "variant": "global_shared",
                "overrides": dict(shared),
                "hypothesis": "Share stability parameters.",
                "evidence": "Synthetic complete incumbent evidence.",
            }
            for env in common.ENVIRONMENTS
            for seed in common.SEEDS
        ],
    }

    def fake_run(command, *, log_path=None):
        del log_path
        if "call_gpt_planner.py" in " ".join(command):
            common.write_json_atomic(source_round / "next_plan.json", plan)
            return 0
        plan_path = Path(command[command.index("--plan") + 1])
        materialized = common.read_json(plan_path)
        destination = experiments / materialized["round_id"]
        destination.mkdir(parents=True)
        common.write_json_atomic(
            destination / "manifest.json",
            {
                "round_id": materialized["round_id"],
                "stage": materialized["stage"],
                "purpose": materialized["purpose"],
                "jobs": [],
            },
        )
        return 0

    monkeypatch.setattr(continuous_tune, "run_checked", fake_run)
    state = {
        "state": "round_scored",
        "next_round_index": 10,
        "incumbents": {
            env: {
                "top5_mean": 100.0,
                "top5_std": 1.0,
                "ratio": 1.0,
                "tunable_config": {"prior_std": 1.0},
            }
            for env in common.ENVIRONMENTS
        },
    }
    round_dir, error = continuous_tune.schedule_unification_round(
        state, source_round, "global"
    )
    assert error is None
    assert round_dir == experiments / "round_010_unify_global"
    materialization_plan = common.read_json(source_round / "materialization_plan.json")
    assert materialization_plan["stage"] == 3
    assert materialization_plan["purpose"] == "unification_global"
    assert len(materialization_plan["jobs"]) == 48
    assert state["current_round_id"] == "round_010_unify_global"
    assert state["unification"]["state"] == "global_materialized"


def test_failed_optional_unification_retains_successful_environment_configs(
    tmp_path: Path, monkeypatch
) -> None:
    round_dir = tmp_path / "round_011_unify_global"
    round_dir.mkdir()
    monkeypatch.setattr(continuous_tune, "LOOP_STATE_PATH", tmp_path / "loop_state.json")
    monkeypatch.setattr(continuous_tune, "CURRENT_CONFIGS_PATH", tmp_path / "current_configs.json")
    monkeypatch.setattr(continuous_tune, "FINAL_CONFIGS_PATH", tmp_path / "final_configs.json")
    monkeypatch.setattr(continuous_tune, "FINAL_REPORT_PATH", tmp_path / "final_target_report.md")
    monkeypatch.setattr(
        continuous_tune,
        "FINAL_UNIFICATION_REPORT_PATH",
        tmp_path / "final_unification_report.md",
    )
    state = {
        "state": "launching_or_resuming_round",
        "rounds_completed": 10,
        "current_round_id": round_dir.name,
        "incumbents": {
            env: {
                "round_id": "round_009_tuning",
                "variant": "accepted",
                "top5_mean": 100.0,
                "top5_std": 1.0,
                "all_bin_mean": 90.0,
                "paper_mean": 100.0,
                "ratio": 1.0,
                "seed_scores": {str(seed): 100.0 for seed in common.SEEDS},
                "tunable_config": {"prior_std": 1.0},
            }
            for env in common.ENVIRONMENTS
        },
        "unification": {"state": "global_materialized", "attempts": []},
    }
    code = continuous_tune.finalize_failed_unification_attempt(
        state,
        round_dir,
        "unification_global",
        "synthetic OOM in optional unification round",
    )
    assert code == 0
    assert state["state"] == "complete"
    assert state["final_strategy"] == "per_environment"
    assert state["rounds_completed"] == 11
    assert state["unification"]["state"] == "per_environment_retained"
    assert common.read_json(tmp_path / "final_configs.json")["strategy"] == "per_environment"
    assert not (round_dir / "blocked_report.md").exists()


def test_planner_failure_is_persisted_for_retry_without_blocking(
    tmp_path: Path, monkeypatch
) -> None:
    round_dir = tmp_path / "round_retry"
    round_dir.mkdir()
    monkeypatch.setattr(continuous_tune, "LOOP_STATE_PATH", tmp_path / "loop_state.json")
    state = {"state": "waiting_for_gpt_planner", "planner_retry_history": []}
    continuous_tune.wait_for_planner_retry(
        state,
        round_dir,
        "synthetic planner startup failure",
        0.0,
    )
    assert state["state"] == "waiting_to_retry_gpt_planner"
    assert state["planner_retry_count"] == 1
    assert state["last_planner_error"] == "synthetic planner startup failure"
    assert "blocked_reason" not in state
    retry = common.read_json(round_dir / "planner_retry_status.json")
    assert retry["status"] == "waiting_to_retry"
    assert retry["retry_delay_sec"] == 0.0


def test_round_failure_is_persisted_for_recovery_without_blocking(
    tmp_path: Path, monkeypatch
) -> None:
    round_dir = tmp_path / "round_retry"
    round_dir.mkdir()
    loop_state = tmp_path / "loop_state.json"
    monkeypatch.setattr(continuous_tune, "LOOP_STATE_PATH", loop_state)
    monkeypatch.setattr(continuous_tune.time, "sleep", lambda _seconds: None)
    state = {"state": "launching_or_resuming_round", "round_retry_history": []}
    continuous_tune.wait_for_round_retry(
        state,
        round_dir,
        "synthetic short run",
        0.0,
    )
    assert state["state"] == "waiting_to_retry_round"
    assert state["round_retry_pending"] is True
    assert state["round_retry_count"] == 1
    assert "blocked_reason" not in state
    retry = common.read_json(round_dir / "round_retry_status.json")
    assert retry["status"] == "waiting_to_retry"
    assert retry["retry_delay_sec"] == 0.0


@pytest.mark.parametrize(
    ("status", "jobs"),
    (("blocked", []), ("stop", []), ("continue", [])),
)
def test_recovery_never_materializes_non_schedulable_planner_output(
    tmp_path: Path, monkeypatch, status: str, jobs: list
) -> None:
    """A schema-valid blocked/stop result must fall back to GPT retry."""

    round_dir = tmp_path / "round_089_gpt_stage3"
    round_dir.mkdir()
    monkeypatch.setattr(continuous_tune, "LOOP_STATE_PATH", tmp_path / "loop_state.json")
    common.write_json_atomic(
        round_dir / "gpt_planner_status.json",
        {"status": "approved", "finished_at": "2026-09-01T00:00:00Z"},
    )
    common.write_json_atomic(
        round_dir / "next_plan.json",
        {
            "status": status,
            "target_envs": [],
            "reasoning": ["No safe proposal."],
            "jobs": jobs,
        },
    )
    state = {
        "next_round_index": 90,
        "stage": 3,
        "planner_retry_count": 4,
        "last_planner_error": "previous planner issue",
    }

    recovered, created = continuous_tune.recover_materialization_plan(round_dir, state)

    assert recovered is None
    assert created is False
    assert not (round_dir / "materialization_plan.json").exists()
    # The normal planner retry path owns these markers; recovery must not make
    # a blocked answer look like a successful scheduling event.
    assert state["planner_retry_count"] == 4
    assert state["last_planner_error"] == "previous planner issue"


def test_recovery_quarantines_existing_non_schedulable_materialization_plan(
    tmp_path: Path, monkeypatch
) -> None:
    round_dir = tmp_path / "round_089_gpt_stage3"
    round_dir.mkdir()
    monkeypatch.setattr(continuous_tune, "LOOP_STATE_PATH", tmp_path / "loop_state.json")
    rejected = {
        "status": "blocked",
        "target_envs": [],
        "reasoning": ["No safe proposal."],
        "jobs": [],
    }
    common.write_json_atomic(round_dir / "materialization_plan.json", rejected)

    recovered, created = continuous_tune.recover_materialization_plan(
        round_dir, {"next_round_index": 90, "stage": 3}
    )

    assert recovered is None
    assert created is False
    assert not (round_dir / "materialization_plan.json").exists()
    archives = list(round_dir.glob(".materialization_plan.json.rejected.*.json"))
    assert len(archives) == 1
    assert common.read_json(archives[0]) == rejected


def test_recovery_materializes_only_continue_plan_with_jobs(tmp_path: Path) -> None:
    round_dir = tmp_path / "round_089_gpt_stage3"
    round_dir.mkdir()
    common.write_json_atomic(
        round_dir / "gpt_planner_status.json",
        {
            "status": "approved",
            "finished_at": "2026-09-01T00:00:00Z",
            "unification_mode": None,
        },
    )
    common.write_json_atomic(
        round_dir / "next_plan.json",
        {
            "status": "continue",
            "target_envs": ["Ant"],
            "reasoning": ["Test proposal."],
            "jobs": [{"env": "Ant", "seed": 1}],
        },
    )
    state = {
        "next_round_index": 90,
        "stage": 3,
        "planner_retry_count": 4,
        "last_planner_error": "previous planner issue",
        "next_planner_retry_at_epoch": 1.0,
    }

    recovered, created = continuous_tune.recover_materialization_plan(round_dir, state)

    assert recovered == round_dir / "materialization_plan.json"
    assert created is True
    materialized = common.read_json(recovered)
    assert materialized["status"] == "continue"
    assert materialized["jobs"]
    assert materialized["round_id"] == "round_090_gpt_stage3"
    assert materialized["purpose"] == "tuning"
    assert state["planner_retry_count"] == 0
    assert "last_planner_error" not in state
    assert "next_planner_retry_at_epoch" not in state


def test_supervisor_heartbeat_updates_loop_state_only(
    tmp_path: Path, monkeypatch
) -> None:
    loop_state = tmp_path / "loop_state.json"
    monkeypatch.setattr(continuous_tune, "LOOP_STATE_PATH", loop_state)
    state = {
        "state": "launching_or_resuming_round",
        "supervisor_pid": os.getpid(),
        "last_round_error": "synthetic recovered failure",
        "round_retry_count": 2,
    }
    common.write_json_atomic(loop_state, state)
    continuous_tune.persist_supervisor_heartbeat("round_999", 12345)
    updated = common.read_json(loop_state)
    assert updated["heartbeat_round_id"] == "round_999"
    assert updated["heartbeat_launcher_pid"] == 12345
    assert "heartbeat_at" in updated
    assert "last_round_error" not in updated
    assert updated["active_round_retry"]["recovered_error"] == "synthetic recovered failure"


def test_phase_heartbeat_and_clear_are_durable(tmp_path: Path, monkeypatch) -> None:
    loop_state = tmp_path / "loop_state.json"
    monkeypatch.setattr(continuous_tune, "LOOP_STATE_PATH", loop_state)
    state = {"state": "waiting_for_gpt_planner", "supervisor_pid": os.getpid()}
    common.write_json_atomic(loop_state, state)

    continuous_tune.persist_phase_heartbeat("round_055_gpt_stage3", "gpt_planner", 321)
    updated = common.read_json(loop_state)
    assert updated["phase"] == "gpt_planner"
    assert updated["phase_pid"] == 321
    assert updated["heartbeat_round_id"] == "round_055_gpt_stage3"

    continuous_tune.clear_phase_heartbeat("round_055_gpt_stage3", "gpt_planner", 321, 0)
    cleared = common.read_json(loop_state)
    assert "phase" not in cleared
    assert cleared["last_phase"]["return_code"] == 0


def test_main_state_persist_preserves_live_heartbeat_without_resurrecting_cleared_phase(
    tmp_path: Path, monkeypatch
) -> None:
    """A stale main-loop snapshot must not erase or resurrect phase markers."""

    loop_state = tmp_path / "loop_state.json"
    monkeypatch.setattr(continuous_tune, "LOOP_STATE_PATH", loop_state)
    common.write_json_atomic(
        loop_state,
        {
            "state": "waiting_for_gpt_planner",
            "supervisor_pid": os.getpid(),
            "phase": "gpt_planner",
            "phase_pid": 321,
            "phase_heartbeat_at": "2026-09-01T00:00:00Z",
            "heartbeat_at": "2026-09-01T00:00:00Z",
        },
    )

    # This simulates the main loop persisting an older in-memory snapshot while
    # the heartbeat worker has already published its phase.  The stale object
    # intentionally contains old heartbeat keys: this is the failure mode that
    # used to overwrite the durable phase marker.
    continuous_tune.persist_loop_state(
        {
            "state": "round_scored",
            "heartbeat_at": "2026-08-27T00:50:21+08:00",
            "heartbeat_round_id": "round_055_gpt_stage3",
            "heartbeat_launcher_pid": 1434496,
            "phase": "old_phase",
            "phase_pid": 111,
            "phase_heartbeat_at": "2026-08-27T00:50:21+08:00",
        }
    )
    live = common.read_json(loop_state)
    assert live["phase"] == "gpt_planner"
    assert live["phase_pid"] == 321

    continuous_tune.clear_phase_heartbeat("round_ignored", "gpt_planner", 321, 0)
    continuous_tune.persist_loop_state(
        {
            "state": "round_scored",
            "phase": "gpt_planner",
            "phase_pid": 321,
            "phase_heartbeat_at": "2026-09-01T00:00:00Z",
        }
    )
    cleared = common.read_json(loop_state)
    assert "phase" not in cleared
    assert "phase_pid" not in cleared
    assert cleared["last_phase"]["return_code"] == 0


def test_append_supervisor_log_uses_loop_state_directory(tmp_path: Path, monkeypatch) -> None:
    loop_state = tmp_path / "state" / "loop_state.json"
    monkeypatch.setattr(continuous_tune, "LOOP_STATE_PATH", loop_state)

    continuous_tune.append_supervisor_log("synthetic heartbeat")

    log = loop_state.with_name("continuous_tune.supervisor.log")
    assert "synthetic heartbeat" in log.read_text(encoding="utf-8")


def test_launcher_adoption_rejects_stale_boot_identity(
    tmp_path: Path, monkeypatch
) -> None:
    round_dir = tmp_path / "round_999"
    round_dir.mkdir()
    current = {
        "pid": 24680,
        "boot_id": "new-boot",
        "start_time_ticks": 500,
        "cmdline_sha256": "current",
        "argv": [
            sys.executable,
            str(continuous_tune.SCRIPT_DIR / "launch_round.py"),
            str(round_dir),
        ],
    }
    monkeypatch.setattr(process_identity, "capture_process_identity", lambda _pid: current)
    monkeypatch.setattr(continuous_tune, "verify_process_identity", process_identity.verify_process_identity)
    state = {
        "state": "running",
        "launcher_pid": 24680,
        "launcher_identity": {**current, "boot_id": "old-boot"},
    }

    adoptable, reason, _ = continuous_tune.launcher_adoption_status(round_dir, state)

    assert adoptable is False
    assert "boot_id" in reason


def test_legacy_launcher_requires_exact_round_cmdline(
    tmp_path: Path, monkeypatch
) -> None:
    round_dir = tmp_path / "round_999"
    wrong_round = tmp_path / "round_998"
    round_dir.mkdir()
    current = {
        "pid": 24680,
        "boot_id": "boot",
        "start_time_ticks": 500,
        "cmdline_sha256": "current",
        "argv": [
            sys.executable,
            str(continuous_tune.SCRIPT_DIR / "launch_round.py"),
            str(wrong_round),
        ],
    }
    monkeypatch.setattr(process_identity, "capture_process_identity", lambda _pid: current)
    monkeypatch.setattr(continuous_tune, "verify_process_identity", process_identity.verify_process_identity)
    state = {"state": "running", "launcher_pid": 24680}

    adoptable, reason, _ = continuous_tune.launcher_adoption_status(round_dir, state)

    assert adoptable is False
    assert "required paths" in reason


def test_process_identity_matches_current_process() -> None:
    identity = process_identity.capture_process_identity(os.getpid())
    assert identity is not None
    matched, reason, current = process_identity.verify_process_identity(
        os.getpid(), identity
    )
    assert matched is True, reason
    assert current is not None
    assert current["boot_id"] == identity["boot_id"]


def test_new_launcher_path_publishes_heartbeat(
    tmp_path: Path, monkeypatch
) -> None:
    class FakeProcess:
        pid = 24680

        def __init__(self) -> None:
            self.poll_count = 0

        def poll(self) -> int | None:
            self.poll_count += 1
            return None if self.poll_count == 1 else 0

    process = FakeProcess()
    heartbeats: list[tuple[str, int]] = []
    monkeypatch.setattr(continuous_tune.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(continuous_tune.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        continuous_tune,
        "persist_supervisor_heartbeat",
        lambda round_id, launcher_pid: heartbeats.append((round_id, launcher_pid)),
    )

    round_dir = tmp_path / "round_999"
    code = continuous_tune.run_launcher_with_heartbeat(["synthetic-launcher"], round_dir)

    assert code == 0
    assert heartbeats == [("round_999", 24680), ("round_999", 24680)]
    assert "exit_code=0" in (round_dir / "launcher.log").read_text(encoding="utf-8")
