# FALCON-S

A simulation framework and controller benchmark for fixed-wing flight close to the ground.

Six-degree-of-freedom rigid-body dynamics with semi-empirical ground effect, actuator dynamics,
sensor noise and Dryden turbulence, across five airframes. Three independent plants — a NumPy
reference, a PyTorch twin and an NVIDIA Warp implementation that steps tens of thousands of
environments in parallel — validated against each other to single-step parity. Reinforcement
learning (PPO, SAC, TD3) and classical control (LQR, MPPI) drive the same command interface through
the same models, so a comparison measures the controller and not the task encoding.

## Requirements

Python 3.10 and an NVIDIA GPU with a CUDA 12.6 driver. The Warp plant is CUDA-only; `falcons check`
and the NumPy reference plant run without one.

## Install

```bash
git clone <this repository> falcon-s && cd falcon-s
bash scripts/setup_venv.sh      # venv → requirements.lock → editable install → falcons check
```

A Docker image is also provided: `docker build -t falcons . && docker run --rm --gpus all falcons`.

## Commands

```bash
falcons check                                     # versions, GPU, load checkpoints, step the plant

# train a policy: 3 algorithms × 2 tasks × 5 airframes × any seed
falcons train --algo ppo --task altitude  --plane Airship_V7 --seed 0
falcons train --algo sac --task attitude  --plane Volantex_Ranger --seed 1
scripts/train_seeds.sh                            # the full 36-run matrix, idempotent

# evaluate one controller on one task, learned or classical
falcons eval --algo ppo  --task altitude --plane Airship_V7 --plot
falcons eval --algo mppi --task attitude --plane Volantex_Ranger --maneuver circle

# sweep a whole protocol across airframes and methods
falcons benchmark altitude                        # acquisition and hold        2 h 11
falcons benchmark maneuvers --refresh             # four named attitude streams    21 min
falcons benchmark ground-effect                   # thrust vs height over span      3 min
falcons benchmark robustness                      # noise, latency, turbulence      9 min
falcons benchmark throughput                      # step rate and cross-plant parity 6 min

falcons tables                                    # results/*.csv → results/tables.tex
falcons figures [name …]                          # → results/figures/
falcons reproduce --out DIR                       # re-fly everything and diff against results/
```

`eval` is the inner loop `benchmark` composes, so a cell printed by `eval` is the cell that lands in
the table. Runtimes are for an RTX 4090.

## Layout

```
src/falcons/
  aircraft/     five airframes: geometry, polynomial aero, LQR gains
  sim/          the three plants — cpu, torch, warp
  envs/         altitude and attitude RL environments
  controllers/  LQR, MPPI, and the checkpoint loader
  train/        PPO, SAC, TD3
  benchmark/    the five protocols, the tables, the figures, reproduce
checkpoints/    39 trained policies
results/        tables, figures and training curves from the shipped run
```

The disturbance models used by the robustness sweep are documented in
[docs/disturbance_model.md](docs/disturbance_model.md).

## Tests

```bash
.venv/bin/pytest -q -m "not slow"     # ~35 s
.venv/bin/pytest -q -m slow           # ~5 h: every protocol re-flown and diffed
```

Three tests need a rollout-cache fixture that is not in the repository; without it the fast tier
skips them. Point `FALCONS_TRACE_FIXTURE` at the cache to run them.

## Reproducing the shipped results

`falcons reproduce` re-flies every protocol from the checkpoints and diffs the outcome against
`results/`. Deterministic rows — every learned policy and the LQR — return bit-identical. MPPI is a
sampling planner, so its cells are draws and are compared within their measured spread.
`PROVENANCE.md` records where the code, data and checkpoints came from and how the shipped tables
relate to the published ones.

## License

MIT — see `LICENSE`.
