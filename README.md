# FlowCritic Paper-Tuning Release

This repository contains the standalone FlowCritic trainer, its paper-aligned
configuration contract, the fixed-budget scoring protocol, and the resumable
multi-GPU experiment supervisor. It is a reproducibility and tuning harness,
not a claim that every benchmark target has been reached.

## Result Status

The checked-in result snapshot is in [`results/RESULTS.md`](results/RESULTS.md);
the machine-readable scores and incumbent tunables are in
[`results/incumbent_snapshot.json`](results/incumbent_snapshot.json) and
[`results/incumbent_configs.json`](results/incumbent_configs.json). It was
generated from the latest complete cohorts available at the snapshot time
(932 unique complete cohorts; round 117 was the latest completed round; round
118 was still running and is excluded).

Nine of twelve accepted incumbents meet at least 90% of the paper mean. Three
remain below the target:

| Environment | Accepted score | Paper mean | Ratio | Status |
| --- | ---: | ---: | ---: | :---: |
| ShadowHandOpenAI_FF | 336.606 | 376.990 | 89.29% | below target |
| Humanoid | 7213.747 | 8863.610 | 81.39% | below target |
| ShadowHand | 895.648 | 1025.760 | 87.32% | below target |

The accepted AllegroHand score is 945.257 (90.53%), but its historical
`old_critic_ema_tau=0.490625` does not satisfy the current three-decimal search
policy. It is marked as a legacy precision migration in
[`results/incumbent_configs.json`](results/incumbent_configs.json). The latest
round-117 trials were lower than the accepted incumbents and are reported as
trials, not silently promoted results. No all-target success claim is made.

## What Is Included

```text
src/                    Trainer and Flow Matching critic
config/                 Per-environment configs and round-000 plan
paper/                  Machine-readable paper targets and budgets
scripts/                Materializer, scheduler, scorer, validator, planner
tests/                  Deterministic contract and scoring tests
systemd/                Optional user-service template for reboot recovery
results/                Small, curated result snapshot; no raw logs
docs/                   Reproducibility and contribution notes
```

Large run directories, metrics, rewards, PIDs, locks, and live supervisor
state are intentionally excluded by `.gitignore`. Runtime directories are
created automatically when a round is materialized.

## Prerequisites

The trainer uses NVIDIA Isaac Gym Preview 4 and IsaacGymEnvs. Those packages
are distributed separately from this repository and are not vendored here.
You need:

- Linux with a working NVIDIA driver and CUDA-capable GPU(s);
- Python 3.8 in a conda environment (the reference environment is named `RLS`);
- Isaac Gym Preview 4 and the matching `isaacgymenvs` package;
- PyTorch built for the installed CUDA runtime;
- the Python packages listed in [`requirements.txt`](requirements.txt).

The reference result environment used Python 3.8, CUDA 11.8, PyTorch 2.4.1,
NumPy 1.24.4, Gym 0.23.1, Tyro 0.5.10, TorchCFM 1.0.5, and TorchDiffEq 0.2.5.
Exact Isaac Gym/PyTorch compatibility depends on the upstream installation.
Install the matching IsaacGymEnvs directory and expose it before running:

```bash
conda create -n RLS python=3.8
conda activate RLS
python -m pip install -r requirements.txt
export PYTHONPATH="/path/to/IsaacGymEnvs:${PYTHONPATH:-}"
```

`src/flow_ppo_single2.py` imports Isaac Gym before PyTorch to satisfy Isaac
Gym's loader. Preserve that import order. `isaacgym` import warnings about a
missing optional FBX library are known to be non-fatal; a traceback, NaN,
OOM, or unexplained non-zero exit is not.

## Paper Contract

The following values are reapplied during materialization and cannot be
overridden by a planner:

| Parameter | Value |
| --- | ---: |
| `num_envs` | 1024 |
| `num_steps` | 16 |
| `learning_rate` | 0.0005 |
| `gamma` | 0.99 |
| `gae_lambda` | 0.95 |
| `num_minibatches` / `minibatch_size` | 2 / 8192 |
| `update_epochs` | 4 |
| `norm_adv` | true |
| `clip_coef` | 0.2 |
| `max_grad_norm` | 1.0 |
| `k_repeat` | 10 |
| `n_remove` | 1 |
| `diff_step` | 5 |
| `exp_alpha` | 0.1 |
| `velocity_clip_delta` | 0.2 |

`Model_L` is deliberately the current five-hidden-layer implementation with
512-unit hidden layers. The release does not alter it to match the paper's
shorter textual `[512]x4` description. Rollout truncation remains
`rollout_mean_mode=remove_max`; the core Flow Matching implementation is
unchanged. See [`config/parameter_contract.json`](config/parameter_contract.json)
and [`src/source_manifest.json`](src/source_manifest.json) for the complete
contract and provenance notes.

Searchable parameters are opened by stage:

- Stage 1: `prior_std`, `target_norm`, `cov_floor`, `reward_scaler`,
  `acf_coef`, `vf_coef`, `flow_sigma`;
- Stage 2: target-normalization and weight-stability controls;
- Stage 3: old-critic update and bootstrap controls;
- Stage 4: optional shared-parameter compression after all targets pass.

Except for `reward_scaler`, searchable numeric values use at most three decimal
places on the 0.005 grid, with 0.01 steps preferred. Core algorithm switches,
PPO values, seeds, architecture, and rollout truncation are protected.

## Scoring Protocol

Run [`scripts/score_plot_protocol.py`](scripts/score_plot_protocol.py) to
reproduce the plotting statistic exactly:

1. Bin episode returns every 200,000 environment transitions using `(left,right]`
   boundaries.
2. In each bin, keep the five largest returns; when a bin has at most five
   samples, use all samples.
3. Average the final 10% of fixed-budget bins separately for each seed.
4. Average seeds 1, 2, 3, and 4 and report the population standard deviation.

Anymal and FrankaCabinet use an 80M requested-transition budget; all other
environments use 100M. Because the trainer floor-divides by `1024*16`, the
effective terminal steps are 79,986,688 and 99,991,552. A short or incomplete
seed may have a diagnostic value but is excluded from the four-seed target
gate. The scorer also reports an all-sample-bin diagnostic to show whether a
top-five selection effect is driving an apparent improvement.

## Quick Start

First check the source without loading Isaac Gym:

```bash
python -m py_compile src/flow_matching.py src/flow_ppo_single2.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

Run the small Humanoid smoke configuration (one PPO batch; not a paper score):

```bash
python -u src/flow_ppo_single2.py \
  --config-path config/config_smoke_Humanoid.json \
  --gpu-id 0 \
  --run-name smoke/humanoid_seed1 \
  --metrics-jsonl experiments/smoke_humanoid_seed1/metrics.jsonl \
  --status-json experiments/smoke_humanoid_seed1/status.json \
  --summary-json experiments/smoke_humanoid_seed1/summary.json \
  --reward-log-path experiments/smoke_humanoid_seed1/rewards.txt
```

Materialize the complete 12-environment Round-000 baseline without launching
it. This plan is the paper-aligned starting template; it is not a claim that
the historical accepted incumbent configuration for every environment is the
same as this baseline. The accepted historical tunables are listed separately
in `results/incumbent_configs.json`.

```bash
python scripts/materialize_round.py \
  --plan config/round_000_plan.json \
  --round-id round_000_baseline \
  --out-root experiments
python scripts/launch_round.py experiments/round_000_baseline \
  --dry-run --gpus 0,1,2,3 --slots-per-gpu 6
```

The dry run should produce 48 commands, with six jobs assigned to each GPU.
To launch the actual fixed-budget cohort:

```bash
python scripts/launch_round.py experiments/round_000_baseline \
  --gpus 0,1,2,3 --slots-per-gpu 6
python scripts/score_plot_protocol.py experiments/round_000_baseline \
  --targets paper/paper_targets.json --require-complete
```

Never wrap these commands with `CUDA_VISIBLE_DEVICES`; physical GPU routing is
recorded through the trainer's `--gpu-id` argument.

## Continuous Tuning

The supervisor performs:

```text
materialize -> launch on four GPUs -> wait for all seeds -> score ->
diagnostics -> GPT proposal -> local validation -> next round
```

Start it directly after configuring the environment:

```bash
cp .env.example .env.local       # edit locally; do not commit it
set -a; . ./.env.local; set +a
python -u scripts/continuous_tune.py \
  --max-rounds 12 \
  --continue-after-stagnation
```

The scheduler uses GPUs `0,1,2,3`, at most six concurrent jobs per GPU (24
total), seeds `1,2,3,4`, and no early stopping. Launcher, trainer, and
supervisor processes are separate so a supervisor restart can adopt a live
launcher. Failed launcher/scoring/planner phases are recorded and retried;
unfinished results are never presented as completed.

### GPT Planner

The optional planner requests model `gpt-5.6-sol` with reasoning effort
`ultra` and strict JSON-schema output. It can use an OpenAI-compatible API:

```bash
export OPENAI_API_KEY='...'
export FLOWCRITIC_LLM_BASE_URL='https://api.openai.com/v1'
export FLOWCRITIC_PLANNER_TRANSPORT=api
python scripts/call_gpt_planner.py experiments/round_XXX \
  --stage 3 --history state/planner_history_index.json \
  --base-configs state/current_configs.json --force
```

Use a private gateway only through `FLOWCRITIC_LLM_BASE_URL`; credentials are
read from the environment or the user's private Codex configuration and are
never written to this repository. The planner receives compact result context
and can propose metadata only. `scripts/validate_plan.py` rejects fixed
parameter changes, inactive controls, duplicate cohorts, incomplete seed
groups, out-of-range values, and precision violations. A planner failure is
not interpreted as a successful experiment.

### Reboot Recovery With systemd

The service is optional and does not start until explicitly enabled:

```bash
scripts/install_user_service.sh --install-only
# Put OPENAI_API_KEY and any path overrides in:
#   ~/.config/flowcritic-paper-tuning.env
scripts/install_user_service.sh --enable-now
systemctl --user status flowcritic-paper-tuning.service
journalctl --user -u flowcritic-paper-tuning.service -f
```

Set `FLOWCRITIC_CONDA_EXE` and `FLOWCRITIC_PYTHON` when conda is not on `PATH`.
To stop the supervisor without deleting artifacts:

```bash
systemctl --user stop flowcritic-paper-tuning.service
```

Because the unit uses `KillMode=process`, detached launchers/jobs may outlive
a supervisor-only stop. Inspect the recorded launcher PID and round state
before terminating training processes manually. Do not delete a live round's
logs while it is running.

## Validation And Recovery Commands

Validate a planner proposal before materialization:

```bash
python scripts/validate_plan.py PLAN.json --stage 1 \
  --history state/planner_history_index.json \
  --base-configs state/current_configs.json
```

Import historical cohorts from an explicitly supplied archive (the default
source path is intentionally empty/portable):

```bash
python scripts/import_historical_cohorts.py \
  --source-root /path/to/archived/experiments \
  --output state/historical_unique_cohorts.json
```

For a normal restart, rerun the same `continuous_tune.py` command. It adopts a
matching live launcher, resumes only unfinished jobs, and preserves prior
attempts. A changed manifest or source hash fails closed rather than silently
rewriting evidence.

## Output And Provenance

Each materialized round records its manifest, resolved configs, code hashes,
fixed contract, per-job status, score CSV/JSON/Markdown, diagnostics, planner
prompt/output, and acceptance decisions under `experiments/<round>/`. Those
runtime artifacts are intentionally ignored in this release. Keep a small
paper-ready summary under `results/` and archive raw runs separately.

The checked-in result snapshot is not a replacement for raw evidence. Historical
scores require the original hardware, Isaac Gym version, seeds, and reward
logs. The code records SHA-256 values so an archived run can be checked against
the exact source and contract used at launch.

## Limitations

- Isaac Gym and IsaacGymEnvs licensing/distribution are external to this repo.
- The result snapshot contains no checkpoints or raw reward/metric logs.
- Three accepted environments are below the 90% target, and AllegroHand still
  needs a compliant precision revalidation.
- The optional GPT planner requires a reachable compatible endpoint and valid
  credentials; deterministic local validation remains authoritative.
- No software license is asserted by this repository yet. Add the license
  appropriate to the copyright holder before making a public release.

## Citation

Use the paper's bibliographic entry when citing FlowCritic. The target values in
`paper/paper_targets.json` are a machine-readable transcription of the paper's
main performance table; the paper source itself is not bundled in this code
release.

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the compact
reproduction checklist and [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) for
change and artifact rules.
