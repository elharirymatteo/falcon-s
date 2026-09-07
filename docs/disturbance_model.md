# Disturbance model — attitude-executor robustness sweep

Methodology and derivation for `src/falcons/benchmark/robustness.py`. Four disturbance families,
each swept mild → medium → severe, applied while an attitude executor tracks the four maneuver
streams (circle / figure-8 / helix / s-turn) in `falcons.envs.attitude.AttitudeEnv(1)`.

**Golden rule:** every reported metric (survival, φ / ḣ / Va RMSE) is computed from the **true
simulator state**, never from the disturbed observation. A disturbance corrupts what the policy
*sees* or *commands*, not what we *score*. Sim step `DT = 0.01 s` (100 Hz control).

---

## 1. Action delay — actuator / command transport lag

The command the actuator executes is the one the policy emitted `k` steps ago (`ActionDelay`).
This is on **top** of the second-order servo + motor lag already inside the plant
(`falcons.sim.warp.Aircraft`); it models the transport dead-time the airframe model does not:
flight-controller compute, the RC/telemetry link, and servo command latency.

| severity | delay | steps @100 Hz |
|----------|-------|---------------|
| mild     | 20 ms | 2 |
| medium   | 50 ms | 5 |
| severe   | 100 ms | 10 |

**Grounding.** Small-UAV loop dead-time is dominated by FC scheduling + link + servo response.
20 ms is a healthy PX4/ArduPilot-class inner loop; 50 ms is a loaded or lower-rate link; 100 ms
is a degraded/long-range link and a slow analog servo — the pessimistic edge that still occurs
in the field. Dead-time is the classical destabiliser for an attitude loop (adds phase lag
∝ ω·τ), so this is the sharpest test of an executor's phase margin.

Buffer note: before the buffer fills, the actuator receives a **neutral** command (zeros), i.e.
the maneuver starts from an un-commanded surface — conservative, no look-ahead.

## 2. Sensor delay — observation latency

The policy acts on the observation from `k` steps ago (`SensorDelay`), modelling EKF/estimator
fusion + sensor pipeline latency. The obs is **cloned** into the buffer — the harness hands out a
view of the single warp obs buffer that is overwritten in place each step, so buffering the view
alone silently makes the delay a no-op (this bug produced sensor-delay ≡ nominal in the first
sweep; fixed).

| severity | delay | steps @100 Hz |
|----------|-------|---------------|
| mild     | 20 ms | 2 |
| medium   | 40 ms | 4 |
| severe   | 80 ms | 8 |

**Grounding.** A well-tuned complementary/EKF attitude estimate on a MEMS IMU lags ~10–30 ms;
40–80 ms represents heavier fusion (vane/airdata blending, lower-rate mag/GPS aiding, filtering
for noise). The paper's nominal assumption is 20 ms. Kept slightly shorter than the action-delay
grid because estimator latency is usually below command-path latency in these stacks.

## 3. Observation noise — sensor floors

Per-channel zero-mean Gaussian noise added to the obs before the policy (`ObsNoise`). Channels
11–14 (previous-action feedback) get **no** noise — they are exact internal state, not sensed.
Physical floors (SI) mapped into the normalised obs space (`obs_noise_sigma`):

| quantity (unit)          | mild | medium | severe |
|--------------------------|------|--------|--------|
| attitude φ,θ (deg)       | 0.5  | 1.0    | 2.0 |
| body rates p,q,r (deg/s) | 1.0  | 2.0    | 5.0 |
| airspeed Va (m/s)        | 0.2  | 0.5    | 1.0 |
| α, β (deg)               | 0.5  | 1.0    | 2.0 |
| climb-rate ż (m/s)       | 0.10 | 0.25   | 0.50 |

**Grounding.** Mild ≈ a good consumer MEMS IMU + pitot after estimator smoothing (sub-degree
attitude, ~1 deg/s rate, 0.2 m/s airspeed); severe ≈ a cheap/vibration-loaded sensor set. The
obs-space σ is the physical floor divided by that channel's normalisation scale (roll/pitch by
π/4, α/β by 0.349 rad = 20°, Va and ż by the per-airframe `max(0.5·Va, 5)`), so a fixed physical
noise is a different fraction of each channel — as in reality.

The noise generator is **seeded per cell** (`noise_seed`), so a re-run of the sweep reproduces its
own numbers. The archive drew from the unseeded global CUDA generator, which is why the
reproduction gate bounds this family (`loose_abs_by["obs_noise"]` in
`src/falcons/benchmark/__init__.py`) instead of demanding equality against the archived table.

## 4. Wind — Dryden turbulence

A full 6-component Dryden gust field (3 linear + 3 angular), the MIL-F-8785C / MIL-HDBK-1797
low-altitude turbulence model already in the sim's `wind_model.turbulence_model`. Severity is set
by a **target longitudinal turbulence intensity** `Iu = σ_u / Va` rather than a raw wind speed, so
the disturbance is a consistent fraction of each airframe's trim airspeed.

| severity | Iu = σ_u / Va |
|----------|---------------|
| mild     | 0.05 |
| medium   | 0.10 |
| severe   | 0.15 |

**Calibration (`calibrate_turb`).** The Dryden filter output σ_u is linear in the wind-at-20-ft
parameter W20, so we measure the constant `k = σ_u / W20` once per airframe (700-step rollout,
statistics after the 150-step filter warm-up), then set `W20 = Iu · Va / k` to hit the target
intensity exactly. This makes the sweep airframe-independent in intensity terms.

**Why not literal MIL wind speeds.** MIL "light/moderate/severe" map to W20 ≈ 15/30/45 kt
(7.7 / 15.4 / 23.1 m/s). For the 15–28 m/s airframe class those σ_u values approach or exceed the
airspeed itself — every controller departs the envelope and the sweep loses all resolution. The
first grid (Iu = 0.10 / 0.20 / 0.30) already crashed every method at medium+. Iu ≤ 0.15 is the
**survivable band where the executors actually separate**, and 0.05–0.15 still spans light →
moderate low-altitude turbulence for this class. This is a resolution choice, stated honestly, not
a claim that these airframes are immune to MIL-severe wind (they are not — see the wind survival
column, which collapses to 0 on Volantex at Iu = 0.15).

---

## Aggregation & scoring

Each (disturbance, severity) **cell** averages over the four maneuvers:
`survival` = fraction of maneuvers completed inside the flight envelope; `phi/hdot/va_rmse` =
mean tracking RMSE over the surviving maneuvers only (a crashed maneuver contributes to survival,
not to RMSE — averaging a "crash RMSE" would be meaningless). One executor (PPO seed 0) and one
gust realisation per cell, so a lone non-monotone survival dip is a draw of the gust field, not a
trend. Outputs: `results/robustness.csv` and `results/figures/fig_robustness_ppo.pdf`
(`falcons benchmark robustness`, then `falcons figures robustness_ppo`).

The sweep is PPO-only here. A disturbance corrupts what the policy sees or commands and never what
is scored, so a cell compares *conditions* rather than *methods*; the archive's sweep also carried
SAC and SAC+CAPS rows, which the paper does not report and whose SAC+CAPS weights are outside the
shipped checkpoint set.

## References

- MIL-F-8785C / MIL-HDBK-1797, *Flying Qualities of Piloted Aircraft* — Dryden turbulence spectra
  and the light/moderate/severe W20 bands.
- Standard small-UAV flight-stack latency figures (PX4 / ArduPilot inner-loop dead-time, EKF2
  estimator lag) for the delay grids.
