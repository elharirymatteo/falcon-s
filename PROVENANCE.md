# Provenance

Fresh history. Code, data and checkpoints were curated from the development repository
`WIG_Plane_RL_Control`, whose **working tree** — not a clean commit — is the source: the snapshot
was taken with uncommitted modifications and untracked files in place, because several of the
shipped protocols (the maneuver harness, the attitude executors, the LQR/MPPI attitude adapters)
lived in that uncommitted state. The snapshot is recorded exactly:

```
source_commit=37c6db86a19810368f2a2503a6290a67d58a236a
source_remote=git@github.com:elharirymatteo/WIG_Plane_RL_Control.git
frozen=2026-09-04
patch_sha256=68c4e3bdc15a108ded34a571de8011062ba98a20762cef87dc56e8577c0ee810
patch_bytes=164270
untracked_entries=43
snapshot_taken=2026-09-05T13:25:02+02:00
```

`patch_sha256` is the SHA-256 of the working-tree diff against `source_commit`; that patch
(164 270 bytes) and the list of the 43 untracked entries it does not carry are archived alongside
the source repository, so the exact tree these results came from can be rebuilt as
`git checkout 37c6db8 && git apply <patch>` plus the untracked files.

Physics fixes that the shipped results depend on (all before the source commit): world-frame
quaternion integration of body rates in the GPU backends; fleet-wide lateral moment sign
correction; LQR gains re-derived on the corrected dynamics; V7 motor constant corrected to reach
its declared trim.

## What differs from the published tables

The shipped tables are re-flown, not copied, and they are not the tables in the paper.
`tests/golden/*.csv` and `tests/golden/tables.tex` hold the published numbers; `results/*.csv` and
`results/tables.tex` hold this repository's own run. The two LaTeX files differ in exactly four
numeric rows, all MPPI. Every learned-policy row (PPO, SAC, TD3) and every LQR row is identical,
as is every ground-effect and robustness row. The four:

| cell | published (`tests/golden`) | shipped (`results`) |
|---|---|---|
| altitude, Airship_V7, MPPI | surv 0.93, acq 0.01 ± 0.01, RMSE 30.7 m, overshoot 41.52 m | surv 0.71, acq 0.03 ± 0.02, RMSE 17.6 m, overshoot 25.02 m |
| altitude, Volantex_Ranger, MPPI | surv 0.18, RMSE 28.1 m, overshoot 28.59 m | surv 0.27, RMSE 26.5 m, overshoot 37.06 m |
| maneuvers, Airship_V7, MPPI circle | 4.8 ± 6.9 (88) | 4.0 ± 5.2 (93) |
| maneuvers, Volantex_Ranger, MPPI | circle 0.5 ± 0.4, figure-8 7.4 ± 13.7 (89), helix 1.6 ± 1.8, s-turn 19.1 ± 14.6 | circle 0.7 ± 0.5, figure-8 8.6 ± 14.1 (86), helix 0.2 ± 0.0, s-turn 7.3 ± 5.6 |

MPPI is a sampling planner, so each of its cells is a *draw* rather than a value: the archive drew
its repeats at one sampler seed and the shipped run draws them at distinct seeds (see "Protocol
changes" below), which moves the mean of a cell and not only its spread. The paper itself reports
the draw-to-draw range for these cells, so the shipped draws sit inside what it claims, and the
finding each row supports is unchanged — MPPI overshoots the commanded altitude change by tens of
metres, and tracks the bank stream to about a degree except on the Volantex figure-of-eight and
s-turn.

Every MPPI cell that the printed tables show moving is in that table; the four *rows* above are
where those cells live. Two further consequences of the same re-draw. The generated maneuver
caption names one more departure than the published one, because a Volantex figure-of-eight draw
left the envelope in the shipped sweep and none did in the archive's. And at full CSV precision
more numbers move than the tables reveal, in cells the tables round to the same digits, while the
seeded `obs_noise` rows of `robustness.csv` move in the third decimal for the reason under
"Protocol changes" — those do round to the published values at the precision the tables print.

The published numbers stay reachable here. `tests/golden/*.csv` are the paper's own tables, and the
archive-mode gate re-flies against them: `tests/test_benchmark_altitude.py`
(`test_full_altitude_table_matches_golden`) runs the altitude protocol with all three MPPI sweeps
on a single seed — `mppi_seeds=(0, 0, 0)`, the archive's mode — and diffs the result against
`tests/golden/altitude.csv`. Its deterministic rows come back bit-identical. Its MPPI rows come
back inside the shipped spread rather than bit-identical: the rollout-cost reduction uses
non-deterministic atomic adds, so even a re-run at the archive's own sampler seed is a fresh draw
(the archive-mode probe read overshoot 33.92 m against the published 28.59 m). Restoring the
archive's seeding recovers the published *cells*; nothing recovers a sampling planner's bits.

## Checkpoint rename map

Old name (development repo, gitignored there) → shipped name.

- `warp_ppo_Airship_V7_dynamic.pt` → `ppo_altitude_Airship_V7_s0.pt`
- `warp_ppo_Airship_V7_dynamic_s1.pt` → `ppo_altitude_Airship_V7_s1.pt`
- `warp_ppo_Airship_V7_dynamic_s2.pt` → `ppo_altitude_Airship_V7_s2.pt`
- `warp_ppo_Volantex_Ranger_dynamic.pt` → `ppo_altitude_Volantex_Ranger_s0.pt`
- `warp_ppo_Volantex_Ranger_dynamic_s1.pt` → `ppo_altitude_Volantex_Ranger_s1.pt`
- `warp_ppo_Volantex_Ranger_dynamic_s2.pt` → `ppo_altitude_Volantex_Ranger_s2.pt`
- `warp_sac_Airship_V7_dynamic.pt` → `sac_altitude_Airship_V7_s0.pt`
- `warp_sac_Airship_V7_dynamic_s1.pt` → `sac_altitude_Airship_V7_s1.pt`
- `warp_sac_Airship_V7_dynamic_s2.pt` → `sac_altitude_Airship_V7_s2.pt`
- `warp_sac_Volantex_Ranger_dynamic.pt` → `sac_altitude_Volantex_Ranger_s0.pt`
- `warp_sac_Volantex_Ranger_dynamic_s1.pt` → `sac_altitude_Volantex_Ranger_s1.pt`
- `warp_sac_Volantex_Ranger_dynamic_s2.pt` → `sac_altitude_Volantex_Ranger_s2.pt`
- `warp_td3_Airship_V7_dynamic.pt` → `td3_altitude_Airship_V7_s0.pt`
- `warp_td3_Airship_V7_dynamic_s1.pt` → `td3_altitude_Airship_V7_s1.pt`
- `warp_td3_Airship_V7_dynamic_s2.pt` → `td3_altitude_Airship_V7_s2.pt`
- `warp_td3_Volantex_Ranger_dynamic.pt` → `td3_altitude_Volantex_Ranger_s0.pt`
- `warp_td3_Volantex_Ranger_dynamic_s1.pt` → `td3_altitude_Volantex_Ranger_s1.pt`
- `warp_td3_Volantex_Ranger_dynamic_s2.pt` → `td3_altitude_Volantex_Ranger_s2.pt`
- `warp_attitude_Airship_V7_dynamic.pt` → `ppo_attitude_Airship_V7_s0.pt`
- `warp_attitude_Airship_V7_dynamic_s1.pt` → `ppo_attitude_Airship_V7_s1.pt`
- `warp_attitude_Airship_V7_dynamic_s2.pt` → `ppo_attitude_Airship_V7_s2.pt`
- `warp_attitude_Volantex_Ranger_dynamic.pt` → `ppo_attitude_Volantex_Ranger_s0.pt`
- `warp_attitude_Volantex_Ranger_dynamic_s1.pt` → `ppo_attitude_Volantex_Ranger_s1.pt`
- `warp_attitude_Volantex_Ranger_dynamic_s2.pt` → `ppo_attitude_Volantex_Ranger_s2.pt`
- `warp_sac_attitude_Airship_V7_dynamic.pt` → `sac_attitude_Airship_V7_s0.pt`
- `warp_sac_attitude_Airship_V7_dynamic_s1.pt` → `sac_attitude_Airship_V7_s1.pt`
- `warp_sac_attitude_Airship_V7_dynamic_s2.pt` → `sac_attitude_Airship_V7_s2.pt`
- `warp_sac_attitude_Volantex_Ranger_dynamic.pt` → `sac_attitude_Volantex_Ranger_s0.pt`
- `warp_sac_attitude_Volantex_Ranger_dynamic_s1.pt` → `sac_attitude_Volantex_Ranger_s1.pt`
- `warp_sac_attitude_Volantex_Ranger_dynamic_s2.pt` → `sac_attitude_Volantex_Ranger_s2.pt`
- `warp_td3_attitude_Airship_V7_dynamic.pt` → `td3_attitude_Airship_V7_s0.pt`
- `warp_td3_attitude_Airship_V7_dynamic_s1.pt` → `td3_attitude_Airship_V7_s1.pt`
- `warp_td3_attitude_Airship_V7_dynamic_s2.pt` → `td3_attitude_Airship_V7_s2.pt`
- `warp_td3_attitude_Volantex_Ranger_dynamic.pt` → `td3_attitude_Volantex_Ranger_s0.pt`
- `warp_td3_attitude_Volantex_Ranger_dynamic_s1.pt` → `td3_attitude_Volantex_Ranger_s1.pt`
- `warp_td3_attitude_Volantex_Ranger_dynamic_s2.pt` → `td3_attitude_Volantex_Ranger_s2.pt`
- `warp_ppo_Airship_A0S_dynamic.pt` → `ppo_altitude_Airship_A0S_s0.pt`
- `warp_ppo_Navion_dynamic.pt` → `ppo_altitude_Navion_s0.pt`
- `warp_ppo_Cirrus_SR22_dynamic.pt` → `ppo_altitude_Cirrus_SR22_s0.pt`

## Protocol changes relative to the archive

The shipped numbers are re-flown, not copied, and six protocol details changed on the way. Each
is a deliberate correction; the reproduction tolerances in `src/falcons/benchmark/__init__.py`
state which archived rows are consequently bounded rather than matched.

- **Altitude, MPPI: distinct sampler seeds.** The sweep is repeated once per entry of
  `mppi_seeds`, default `(0, 1, 2)`. The archive ran its three repeats at the *same* seed (0), so
  the spread it published was only the non-deterministic atomic-add reduction and not the sampler;
  its σ is under-dispersed. `mppi_seeds=(0, 0, 0)` reproduces the archive mode.
- **Robustness: PPO only.** The archive's sweep also carried SAC and SAC+CAPS rows. The paper
  reports neither, and the SAC+CAPS weights are outside the shipped checkpoint set, so they are
  reproduced nowhere here; `results/robustness.csv` has 26 PPO rows against the archive's 78.
- **Sensor noise is seeded per cell.** The archive drew observation noise from the unseeded global
  CUDA generator, making its `obs_noise` rows a single unrepeatable draw. The shipped protocol
  seeds a generator per cell, so a re-run reproduces its own numbers.
- **The render strip draws recorded simulator tracks.** The archive's `render_strip.py` shelled
  out to ffmpeg, pulled the last frame of an mp4 and cropped the PNG, making the figure a
  by-product of an animation nobody kept. `benchmark.figures.maneuver_render` draws the same eight
  tracks with the same bank colouring directly from the flight paths `benchmark.maneuvers.fly`
  records.
- **Every MPPI maneuver metric carries its own spread, and the gate is a 3σ band.** The archive's
  maneuver table reported a standard deviation for the bank RMSE alone, which was enough only
  because that table was produced by replaying a cached set of rollouts. MPPI is a sampling
  planner: *every* metric of one of its cells is a draw, so `benchmark.maneuvers.aggregate` now
  emits a `<metric>_std` beside each averaged column. The reproduction gate compares a sampled row
  within `sigma_k` (= 3) standard deviations rather than one: two independent draws of the same
  statistic sit about √2 σ apart, so a one-σ band would reject an honest re-run roughly half the
  time. A cell that lost draws is gated at six, not three (`sigma_partial_survival_k` = 2), because
  its survivor-conditioned mean is averaged over a different subset of draws each sweep. No band
  exceeds `sigma_max` for its column. Deterministic rows — every learned policy and the LQR — are
  still matched exactly.
- **`throughput.stats()` computes its t-quantile.** The archive baked in 2.262 (the two-sided 95 %
  Student-t critical value at df = 9); it is now evaluated at df = n−1, so the interval stays
  correct if the trial count moves off `R_TRIALS`.

## What was left behind

The committed virtual environment; 150 checkpoints from experiments not in the paper; the
waypoint, bank-heading, flare and constrained-RL tasks; the ground-effect-during-training study;
the X-Plane/Matlab plot overlays; 28 experiment scripts. All remain in the source repository at
the commit above.
