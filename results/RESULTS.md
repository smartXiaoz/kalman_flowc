# Release Result Summary

- Score table generated: `2026-09-05T16:18:55+08:00`
- Incumbent/config state source: `2026-09-05T16:14:04+08:00`
- Source workspace: `local tuning workspace (path intentionally omitted)`
- Historical unique complete cohorts: `932`
- Latest completed round: `round_117_gpt_stage3`
- Running round at snapshot: `round_118_gpt_stage3` (`launching_or_resuming_round`)

## Accepted Incumbent Snapshot

| Environment | Score | Paper target | Ratio | Status | Source | Variant |
| --- | ---: | ---: | ---: | :---: | --- | --- |
| FrankaCubeStack | 16.513 | 16.300 | 101.31% | pass | round_000_baseline | known_best |
| Quadcopter | 1392.919 | 1393.560 | 99.95% | pass | round_000_baseline | known_best |
| FrankaCabinet | 3490.891 | 3511.010 | 99.43% | pass | round_020_family_switch | cabinet_p090_cov005 |
| BallBalance | 365.026 | 377.870 | 96.60% | pass | round_020_family_switch | ball_p125_cov025 |
| ShadowHandOpenAI_FF | 336.606 | 376.990 | 89.29% | below target | round_020_family_switch | ff_norm_bootstrap |
| Anymal | 68.859 | 69.630 | 98.89% | pass | round_000_baseline | known_best |
| Ant | 8668.444 | 9345.290 | 92.76% | pass | round_000_baseline | known_best |
| AllegroHand | 945.257 | 1044.120 | 90.53% | pass | round_014_gpt_stage3 | hand_ema0490625 |
| Humanoid | 7213.747 | 8863.610 | 81.39% | below target | round_115_gpt_stage3 | humanoid_post114_covclip020 |
| AllegroKuka | 494.458 | 529.300 | 93.42% | pass | round_027_gpt_stage3 | kuka_acf100_warmup1000_ess0975_boot_remove |
| AllegroKukaTwoArmsLSTM | 548.130 | 339.870 | 161.28% | pass | round_000_baseline | known_best |
| ShadowHand | 895.648 | 1025.760 | 87.32% | below target | round_083_gpt_stage3 | shadowhand_stage3_prior110 |

## Latest Completed Round Per Environment

| Environment | Round | Variant | Score | Ratio |
| --- | --- | --- | ---: | ---: |
| FrankaCubeStack | round_000_baseline | known_best | 16.513 | 101.31% |
| Quadcopter | round_000_baseline | known_best | 1392.919 | 99.95% |
| FrankaCabinet | round_020_family_switch | cabinet_p090_cov005 | 3490.891 | 99.43% |
| BallBalance | round_020_family_switch | ball_p125_cov025 | 365.026 | 96.60% |
| ShadowHandOpenAI_FF | round_117_gpt_stage3 | ff_stage3_tau0805_acf195_targetnone | 298.556 | 79.19% |
| Anymal | round_000_baseline | known_best | 68.859 | 98.89% |
| Ant | round_000_baseline | known_best | 8668.444 | 92.76% |
| AllegroHand | round_117_gpt_stage3 | allegro_stage3_tau0495_flow0085_reward005 | 670.546 | 64.22% |
| Humanoid | round_117_gpt_stage3 | humanoid_stage3_hard_prior105_acf090_targetnone | 5933.697 | 66.94% |
| AllegroKuka | round_027_gpt_stage3 | kuka_acf100_warmup1000_ess0975_boot_remove | 494.458 | 93.42% |
| AllegroKukaTwoArmsLSTM | round_000_baseline | known_best | 548.130 | 161.28% |
| ShadowHand | round_117_gpt_stage3 | shadowhand_stage3_tau0055_vf105 | 505.272 | 49.26% |

## Interpretation

- Scores use fixed 200,000-transition `(left, right]` bins, top-5-per-bin selection, final 10% bins, and population standard deviation across seeds 1–4.
- Ratios below 90% remain explicitly below target; this release does not turn them into success claims.
- The historical AllegroHand incumbent with `old_critic_ema_tau=0.490625` is retained as evidence but is not compliant with the current three-decimal policy.
- Round `round_118_gpt_stage3` was still running at snapshot time and is excluded from the completed-round table.
