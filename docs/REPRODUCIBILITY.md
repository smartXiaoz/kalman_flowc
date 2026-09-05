# Reproducibility Notes

## Environment

The trainer depends on NVIDIA Isaac Gym Preview 4 and the matching
IsaacGymEnvs package. Install those packages according to their upstream
instructions, then expose the directory containing `isaacgymenvs` through
`PYTHONPATH`. The reference environment used Python 3.8, CUDA 11.8, PyTorch
2.4.1, NumPy 1.24.4, Gym 0.23.1, and Tyro 0.5.10. Isaac Gym is imported before
PyTorch in `src/flow_ppo_single2.py`; preserve that order.

## Deterministic contract

All paper runs use seeds 1, 2, 3, and 4, four physical GPUs (`0,1,2,3`),
`num_envs=1024`, `num_steps=16`, and the fixed values in
`config/parameter_contract.json`. Humanoid uses `gamma=0.99`, matching the
paper contract. `Model_L` is the current five-hidden-layer, 512-unit
implementation. The release does not silently replace it with the paper's
shorter textual description.

## Scoring

Use `scripts/score_plot_protocol.py`. It bins episode returns in 200,000
environment transitions with `(left,right]` boundaries, keeps the largest five
returns per bin (or all returns when a bin has at most five), averages the final
10% of fixed-budget bins per seed, and reports the mean plus population standard
deviation across the four seeds. Anymal and FrankaCabinet request 80M
transitions; all other environments request 100M. The effective floor-divided
endpoints are 79,986,688 and 99,991,552 respectively. Incomplete seeds are
diagnostic only and never enter the target gate.

## Provenance

`src/flow_matching.py` is unchanged from the upstream source snapshot. The
trainer changes are limited to standalone path resolution and durable logging
metadata; the algorithmic contract is recorded in `src/source_manifest.json`.
The result tables in `results/` are a snapshot, not a claim that every target
has been met. Reproducing historical scores requires the original Isaac Gym
environment, hardware, and raw experiment artifacts, which are not included
in Git.
