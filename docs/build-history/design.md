# falcon-s — clean reproduction repo for the FALCON-S paper

Date: 2026-09-04. Source: `WIG_Plane_RL_Control` @ current `main` (fresh history; the old repo is
the archive). Target: `/home/matteo/Projects/falcon-s`.

## Goal

A new repository containing only what reproduces every table and figure in the paper, with the
code refactored for fewer files, clear names and no redundancy. Reproduction is executable
(`falcons reproduce`) and asserted (`tests/test_reproduce.py`).

Decisions taken during brainstorming:

- **Both tiers.** Checkpoints are the default reproduction path; training from scratch is the
  documented second tier. The 36 paper checkpoints ship in the repo.
- **Fresh git history.** `PROVENANCE.md` records the source commit, the checkpoint rename map and
  the key physics-fix commits (quaternion frame, lateral signs, LQR re-derivation).
- **No LaTeX in the repo.** `falcons tables` writes `results/tables.tex`; the manuscript lives
  elsewhere. Reference PDFs under the old `paper/` are dropped.
- **Approach 2**: installable `falcons` package with a rationalised layout. The three simulator
  backends stay independent code paths — their independence is the paper's parity claim.
- Tests reduced to what guards reproducibility.

## Layout

```
falcon-s/
├── pyproject.toml            name=falcons, python=3.10, deps by name, `falcons` entrypoint
├── requirements.lock         exact working set (torch==2.7.0 cu126, warp-lang==1.8.1, numpy==2.2.6, …)
├── README.md  PROVENANCE.md  LICENSE
├── src/falcons/
│   ├── aircraft/
│   │   ├── config.py         AircraftConfig — one loader (merges utils/config + data/vehicle_config)
│   │   └── data/<plane>/     <plane>.json  <plane>_poly.csv  <plane>_K_LQR.csv   (5 planes)
│   ├── sim/
│   │   ├── cpu/              reference plant, internals untouched: physics/ actuators/ sensors/
│   │   │                     estimators/ aircraft.py
│   │   ├── torch/            altitude.py wind.py
│   │   └── warp/             aircraft.py physics.py aerodynamics.py actuators.py sensors.py
│   │                         wind.py dryden.py termination.py history.py altitude_obs.py altitude_reward.py
│   ├── envs/                 altitude.py attitude.py cpu.py
│   ├── controllers/
│   │   ├── lqr/              control.py attitude.py reference.py   (ruling R8: a package,
│   │   │                     not one module — the attitude executor and the warp-side reference
│   │   │                     twin are separate concerns and stayed separate files)
│   │   ├── mppi/             control.py cost.py kernels.py attitude.py reference.py
│   │   └── policies.py       load_policy(algo, task, plane, seed) — the one checkpoint loader
│   ├── train/                ppo.py sac.py td3.py configs.py tasks.py curves.py
│   ├── benchmark/
│   │   ├── __init__.py       per-table diff tolerances for `reproduce`
│   │   ├── altitude.py       acquisition/hold protocol           → results/altitude.csv
│   │   ├── maneuvers.py      streams + harness + track           → results/maneuvers.csv
│   │   ├── ground_effect.py  trim + closed-loop                  → results/ge_trim.csv, ge_energy.csv
│   │   ├── robustness.py                                         → results/robustness.csv
│   │   ├── throughput.py     parity + step/eval throughput       → results/throughput.csv
│   │   ├── figures.py        the 11 paper figures from CSVs/traces
│   │   └── tables.py         CSVs → results/tables.tex
│   └── cli.py                argparse glue only
├── checkpoints/              36 × {algo}_{task}_{plane}_s{seed}.pt   (~54 MB, plain git)
├── results/                  shipped CSVs, tables.tex, figures/, curves/, environment.json
│   └── traces/               rollout cache, gitignored
├── scripts/                  setup_venv.sh  train_seeds.sh  derive_lqr_gains.py
└── tests/                    test_parity test_checkpoints test_sim test_envs test_reproduce + golden/
```

Kept from the old repo: the 100 first-party files that load at runtime when every paper
entrypoint is imported (verified by import, not by regex). Dropped: `utils/plot_data.py`,
`envs/{waypoint,waypoint_seq,bank_heading}`, `backends/warp/waypoint`, 17 eval scripts,
`trl_paper/{ge_study,results}`, 28 experiment scripts, the committed venv, 150 non-paper
checkpoints, `figures/` (50 MB), `paper/`, agent tooling.

Merged: two checkpoint loaders → `policies.py`; two config managers → `aircraft/config.py`;
`utils/util.py::Trajectory` → the ~40 lines of logging LQR/MPPI use; `eval/backend/{parity,benchmark}`
+ `scripts/bench_sim_throughput` → `benchmark/throughput.py`; five plotting scripts →
`benchmark/figures.py`. About 100 → 55 source files.

Naming: planes are `Airship_V7`-style identifiers defined once in `aircraft/config.py`.
Checkpoints are `{algo}_{task}_{plane}_s{seed}.pt`. Classes lose backend prefixes the module
already carries (`sim.warp.Aircraft`, not `WarpAircraftModel`). Every command is a function first,
CLI second; paths are explicit arguments with package-relative defaults, never module globals.

## CLI

```
falcons check                                   versions, load 36 checkpoints, one CPU-plant step
falcons train  --algo {ppo,sac,td3} --task {altitude,attitude} --plane P --seed N [--steps]
falcons eval   --algo {ppo,sac,td3,lqr,mppi} --task … --plane P [--seed] [--maneuver] [--plot] [--n]
falcons benchmark {altitude,maneuvers,ground-effect,robustness,throughput} [--planes] [--refresh]
                                                maneuvers: --mppi-draws 5 --mppi-seed 0
falcons tables                                  results/*.csv → results/tables.tex
falcons figures [name …]                        → results/figures/
falcons reproduce                               benchmark ×5 → tables → figures → diff vs shipped
```

`eval` is the inner loop `benchmark` composes: one code path from the quick check to the table.
`scripts/train_seeds.sh` is the idempotent 36-run training matrix over `falcons train`.
Old CLIs (`python -m train`, `python -m eval`, `python -m trl_paper.*`, `MPPI_SEED`/`MPPI_REPS`
env vars) do not exist; README maps old → new. The `--backend torch` training path is dropped.

## Regression gate

1. **Freeze goldens** from the old repo before any change: 5 result CSVs, `mppi_variance.json`,
   `throughput.json`, 11 figures → `tests/golden/` (CSV/json, ~1 MB); the 39 MB traces cache stays
   local as a gitignored fixture.
2. **Determinism audit, step 0.** Each benchmark run twice on the old code → the true per-table
   floor. MPPI and wind cells are draws by construction; whether the altitude protocol's random
   initial conditions are seeded is unknown and decides whether that table is "exact" or
   "within a stated tolerance".
3. **Per-step diff.** After each migration step the affected CSV is regenerated on cached traces
   and diffed against golden. Learned + LQR rows exact; the rest within the audited floor. A step
   that fails is not committed.
4. **Final `--refresh` re-fly** with no cache becomes the shipped `results/`; `reproduce` diffs
   against it thereafter. Tolerances live in one table in `benchmark/__init__.py`.

## Tests (~25, fast tier < 30 s)

| file | guards |
|---|---|
| `test_parity.py` | torch↔warp single-step + Dryden parity; cpu↔warp V7 trim |
| `test_checkpoints.py` | all 36 load via `policies.py`, shapes, deterministic action (0 skips) |
| `test_sim.py` | trim for 5 planes, piston thrust, termination kernel |
| `test_envs.py` | attitude obs / reward / propagation / heading invariance |
| `test_reproduce.py` | one plane × one algo × small n on cached traces vs golden; full `reproduce` under `@slow` |

Dropped: flare, descent-spawn, SAC-components tests. Curriculum/schedule tests kept only if
attitude training uses the curriculum (checked during migration).

## Ships vs regenerates

Ships (~62 MB): `checkpoints/`, `results/*.csv`, `tables.tex`, `figures/`, `curves/`,
`environment.json` (GPU, driver, torch/warp versions, commit of the run), `aircraft/data/`.
Regenerates: `results/traces/`. Without it MPPI rows return within σ, not exact — stated in README;
traces optionally published as a release asset.

## Environment

venv first. `pyproject.toml` + `requirements.lock` (torch from the cu126 index) +
`scripts/setup_venv.sh` (venv → lock → `pip install -e .` → `falcons check`). Requirement stated
plainly: NVIDIA GPU with a CUDA 12.6 driver; no CPU-only reproduction of the GPU tables.
Docker is a second pass after the venv reproduces end-to-end: `nvidia/cuda:12.6-cudnn-runtime-ubuntu22.04`
+ python3.10 + the same lock. The old Dockerfile is not a starting point.

## Migration order

| # | step | gate |
|---|---|---|
| 0 | freeze goldens + determinism audit (old repo) | per-table floor known |
| 1 | scaffold: pyproject, lock, setup script, empty package, `check` stub | fresh venv builds |
| 2 | `aircraft/` | configs dict-identical to old loader |
| 3 | `sim/cpu` copied as-is | trim, piston thrust identical |
| 4 | `sim/warp` + `sim/torch` | parity tests; one step from fixed seed bit-identical |
| 5 | `envs/` | obs vector bit-identical; env tests |
| 6 | `controllers/` + renamed checkpoints | 36 load; K identical; cached MPPI replay identical |
| 7 | `benchmark/` altitude → ground-effect → maneuvers → robustness → throughput | each CSV diff vs golden |
| 8 | `tables.py`, `figures.py` | `tables.tex` byte-diff; figures regenerated and compared |
| 9 | `train/` + `cli.py` train/eval | 1000-step smoke all algos × tasks; eval cell == benchmark cell |
| 10 | `reproduce`, `environment.json`, `test_reproduce.py` | fast tier green; slow tier green on cache |
| 11 | final `--refresh` re-fly → `results/`; README, PROVENANCE, LICENSE | `reproduce` green on its own output |
| 12 | Dockerfile (separate pass) | `docker run --gpus all … falcons check` |

Commits by layer: scaffold / sim / controllers / benchmark / results. Cost: several days of
gated work; step 11 alone is hours of GPU. Step 0 can change the plan and is reported before
continuing.
