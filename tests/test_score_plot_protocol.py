"""Golden and integrity tests for scripts/score_plot_protocol.py."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "score_plot_protocol.py"
SPEC = importlib.util.spec_from_file_location("score_plot_protocol", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
score = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = score
SPEC.loader.exec_module(score)


def test_golden_top5_tail_and_population_std() -> None:
    """Golden numbers cover top-5, <=5 fallback, all-bin, and seed std."""

    budget = 2_000_000  # ten 200k bins, so the final 10% is exactly bin ten
    final_steps = [1_800_001, 1_820_000, 1_840_000, 1_860_000, 1_880_000, 2_000_000]
    rewards_by_seed = {
        1: [1, 2, 3, 4, 5, 100],       # top five = 22.8; all = 19.1666...
        2: [10, 20],                    # <=5: both metrics = 15
        3: [0, 5, 10, 15, 20, 25],     # top five = 15; all = 12.5
        4: [-10, -5, 0, 5, 10, 15],    # top five = 5; all = 2.5
    }
    expected_top = {1: 22.8, 2: 15.0, 3: 15.0, 4: 5.0}
    expected_all = {1: 115.0 / 6.0, 2: 15.0, 3: 12.5, 4: 2.5}

    seed_rows = []
    for seed, rewards in rewards_by_seed.items():
        bins = score.aggregate_bins(final_steps[: len(rewards)], rewards, budget)
        assert bins["tail_bin_count"] == 1
        assert bins["tail_valid_bin_count"] == 1
        assert bins["top5_bin_tail_mean"] == pytest.approx(expected_top[seed])
        assert bins["all_sample_bin_tail_mean"] == pytest.approx(expected_all[seed])
        seed_rows.append(
            {
                "env": "Ant",
                "variant": "golden",
                "seed": seed,
                "complete": True,
                "completeness": "complete",
                "top5_tail_mean": bins["top5_bin_tail_mean"],
                "all_bin_tail_mean": bins["all_sample_bin_tail_mean"],
            }
        )

    result = score.aggregate_results(seed_rows, {"Ant": {"mean": 10.0, "std": 1.0}})[0]
    expected_mean = sum(expected_top.values()) / 4
    expected_std = math.sqrt(sum((value - expected_mean) ** 2 for value in expected_top.values()) / 4)
    assert result["complete"] is True
    assert result["complete_seed_count"] == 4
    assert result["top5_mean"] == pytest.approx(14.45)
    assert result["top5_std"] == pytest.approx(expected_std)
    assert result["all_bin_mean"] == pytest.approx(sum(expected_all.values()) / 4)
    assert result["ratio"] == pytest.approx(1.445)
    assert result["passed_90"] is True


def test_bin_boundaries_are_open_left_closed_right() -> None:
    result = score.aggregate_bins(
        steps=[0, 1, 200_000, 200_001, 400_000, 400_001],
        rewards=[999, 1, 2, 3, 4, 888],
        budget=400_000,
    )

    assert result["bin_counts"] == [2, 2]
    assert result["top5_bin_means"] == pytest.approx([1.5, 3.5])
    assert result["all_sample_bin_means"] == pytest.approx([1.5, 3.5])
    assert result["rows_at_or_before_zero"] == 1
    assert result["rows_after_budget"] == 1
    # With two bins, int(2 * .9) = 1: only (200k, 400k] is in the tail.
    assert result["top5_bin_tail_mean"] == pytest.approx(3.5)


def _write_job(
    round_dir: Path,
    *,
    job_id: str,
    env: str,
    seed: int,
    reward_step: int,
    metric_step: int,
    rewards: list[float],
    total_timesteps: int,
) -> dict:
    run_dir = round_dir / "runs" / job_id
    run_dir.mkdir(parents=True)
    config = {
        "env_id": env,
        "seed": seed,
        "num_envs": 1024,
        "num_steps": 16,
        "total_timesteps": total_timesteps,
    }
    (run_dir / "config.resolved.json").write_text(json.dumps(config), encoding="utf-8")
    reward_lines = ["step,episodic_return"] + [f"{reward_step},{value}" for value in rewards]
    (run_dir / "rewards.txt").write_text("\n".join(reward_lines) + "\n", encoding="utf-8")
    (run_dir / "metrics.jsonl").write_text(
        json.dumps({"global_step": metric_step, "env_id": env, "seed": seed}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "status.json").write_text(json.dumps({"state": "finished", "global_step": metric_step}), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps({"state": "finished", "exit_code": 0}), encoding="utf-8")
    return {
        "job_id": job_id,
        "job_run_dir": str(run_dir),
        "env": env,
        "variant": "candidate",
        "seed": seed,
        "config_path": str(run_dir / "config.resolved.json"),
        "metrics_jsonl": str(run_dir / "metrics.jsonl"),
        "status_json": str(run_dir / "status.json"),
        "summary_json": str(run_dir / "summary.json"),
        "trainer_reward_path": str(run_dir / "rewards.txt"),
    }


def test_fixed_budget_rejects_short_run_even_with_high_tail_reward(tmp_path: Path) -> None:
    round_dir = tmp_path / "round_short"
    job = _write_job(
        round_dir,
        job_id="001_Ant_candidate_seed1",
        env="Ant",
        seed=1,
        reward_step=99_000_000,
        metric_step=99_000_000,
        rewards=[100_000.0],
        total_timesteps=100_000_000,
    )

    row = score.score_job(round_dir, job)
    assert row["top5_tail_mean"] == pytest.approx(100_000.0)
    assert row["coverage_complete"] is False
    assert row["complete"] is False
    assert row["completeness"] == "short_run"

    cohort = score.aggregate_results([row], {"Ant": {"mean": 1.0, "std": 0.0}})[0]
    assert cohort["complete"] is False
    assert cohort["top5_mean"] is None
    assert cohort["ratio"] is None
    assert cohort["passed_90"] is False


def test_fixed_budget_rejects_missing_durable_exit_code(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.resolved.json").write_text(
        json.dumps(
            {
                "env_id": "Ant",
                "seed": 1,
                "num_envs": 1024,
                "num_steps": 16,
                "total_timesteps": 100_000_000,
            }
        ),
        encoding="utf-8",
    )
    terminal = score.expected_terminal_step(100_000_000)
    (run_dir / "metrics.jsonl").write_text(
        json.dumps({"global_step": terminal}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "rewards.txt").write_text(
        f"step,episodic_return\n{terminal},123.0\n",
        encoding="utf-8",
    )
    row = score.score_job(
        tmp_path,
        {
            "job_id": "run",
            "job_run_dir": str(run_dir),
            "config_path": str(run_dir / "config.resolved.json"),
            "env": "Ant",
            "variant": "candidate",
            "seed": 1,
        },
    )
    assert row["coverage_complete"] is True
    assert row["complete"] is False
    assert row["completeness"] == "missing_exit_code"
    assert "missing_exit_code" in row["warnings"]


@pytest.mark.parametrize("bad_reward", ["NaN", "inf", "not-a-number"])
def test_fixed_budget_rejects_nonfinite_or_malformed_reward_rows(
    tmp_path: Path, bad_reward: str
) -> None:
    round_dir = tmp_path / "round_bad_reward"
    terminal = score.expected_terminal_step(100_000_000)
    job = _write_job(
        round_dir,
        job_id="001_Ant_candidate_seed1",
        env="Ant",
        seed=1,
        reward_step=terminal,
        metric_step=terminal,
        rewards=[123.0],
        total_timesteps=100_000_000,
    )
    rewards_path = Path(job["trainer_reward_path"])
    with rewards_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{terminal},{bad_reward}\n")

    row = score.score_job(round_dir, job)
    assert row["coverage_complete"] is True
    assert row["complete"] is False
    assert row["reward_malformed_rows"] == 1
    if bad_reward in {"NaN", "inf"}:
        assert row["reward_nonfinite_rows"] == 1
        assert row["completeness"] == "nonfinite_rewards"
    else:
        assert row["reward_nonfinite_rows"] == 0
        assert row["completeness"] == "malformed_rewards"


def test_unexplained_nonzero_exit_cannot_be_complete(tmp_path: Path) -> None:
    round_dir = tmp_path / "round_nonzero"
    terminal = score.expected_terminal_step(100_000_000)
    job = _write_job(
        round_dir,
        job_id="001_Ant_candidate_seed1",
        env="Ant",
        seed=1,
        reward_step=terminal,
        metric_step=terminal,
        rewards=[9999.0],
        total_timesteps=100_000_000,
    )
    launcher_status = round_dir / "runs" / job["job_id"] / "launcher_status.json"
    launcher_status.write_text(
        json.dumps({"state": "failed_nonzero", "exit_code": 17}), encoding="utf-8"
    )

    row = score.score_job(round_dir, job)
    assert row["coverage_complete"] is True
    assert row["top5_tail_mean"] == pytest.approx(9999.0)
    assert row["complete"] is False
    assert row["completeness"] == "failed_nonzero"
    assert "nonzero_exit_17" in row["warnings"]


def test_cli_writes_fixed_json_csv_md_and_uses_80m_budget(tmp_path: Path) -> None:
    round_dir = tmp_path / "round_000_baseline"
    terminal = score.expected_terminal_step(80_000_000)
    jobs = []
    for seed in (1, 2, 3, 4):
        jobs.append(
            _write_job(
                round_dir,
                job_id=f"{seed:03d}_Anymal_candidate_seed{seed}",
                env="Anymal",
                seed=seed,
                reward_step=terminal,
                metric_step=terminal,
                rewards=[float(seed * 10)],
                total_timesteps=80_000_000,
            )
        )
    (round_dir / "manifest.json").write_text(
        json.dumps({"round_id": "round_000_baseline", "jobs": jobs}), encoding="utf-8"
    )
    targets_path = tmp_path / "targets.json"
    targets_path.write_text(json.dumps({"targets": {"Anymal": {"mean": 20.0, "std": 2.0}}}), encoding="utf-8")

    assert score.main([str(round_dir), "--targets", str(targets_path), "--output-dir", str(round_dir), "--quiet"]) == 0
    for suffix in ("csv", "json", "md"):
        assert (round_dir / f"score_plot_protocol.{suffix}").is_file()

    payload = json.loads((round_dir / "score_plot_protocol.json").read_text(encoding="utf-8"))
    assert isinstance(payload["results"], list) and len(payload["results"]) == 1
    result = payload["results"][0]
    assert result["env"] == "Anymal"
    assert result["variant"] == "candidate"
    assert result["complete_seed_count"] == 4
    assert result["complete"] is True
    assert result["top5_mean"] == pytest.approx(25.0)
    assert result["top5_std"] == pytest.approx(math.sqrt(125.0))
    assert result["all_bin_mean"] == pytest.approx(25.0)
    assert result["paper_mean"] == pytest.approx(20.0)
    assert result["target_90"] == pytest.approx(18.0)
    assert result["ratio"] == pytest.approx(1.25)
    assert result["passed_90"] is True
    assert [item["seed"] for item in result["seeds"]] == [1, 2, 3, 4]
    assert all(item["budget"] == 80_000_000 for item in result["seeds"])


def test_budget_contract() -> None:
    assert score.budget_for_env("Anymal") == 80_000_000
    assert score.budget_for_env("FrankaCabinet") == 80_000_000
    assert score.budget_for_env("Humanoid") == 100_000_000
    assert score.budget_for_env("Ant") == 100_000_000
