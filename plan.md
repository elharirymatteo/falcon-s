# Aerodynamics refactor: OpenVSP derivative model replaces the polynomial + empirical GE model

**Status:** planned, no code written yet. Working notes, not a committed artefact — delete when the
refactor lands.

**Target form** (as specified by the user, MATLAB reference):

```matlab
function aero_coefs = aero_coefs_fcn(pos, ang_vel, vel, deflect_angs, aero_angs, ...
                                     b_w, c_bar, coefs, ge, ge_enable)
```

Linearisation about the VSPAERO run point, with an OpenVSP-measured ground-effect surface
interpolated on h/c. Returns `[C_D; C_S; C_L; C_l; C_m; C_n]`.

---

## Before every phase: re-assess

**Do not start a phase by executing it.** Start by checking whether it still makes sense:

1. Re-read this file and the actual state of the tree (`git status`, `git log --oneline -10`) —
   earlier phases may have changed what a later one should do.
2. Check the decisions and assumptions sections below against reality. Data files may have arrived
   (the four missing airframes), a number may have turned out wrong, or the user may have changed
   their mind.
3. State briefly what you believe the phase should now be, call out anything that should change,
   and only then proceed. If a change is material, ask before writing code.

This matters most for Phase 5, whose cost depends on decisions about retraining that were left open.

---

## Progress log

Branch: **`aero/openvsp-derivatives`** (off `main` @ 48973c2).

### Phase 1 — DONE

Refinement made at the re-assessment gate: **Phase 1 is purely additive.** The plan had it
deleting `poly_params_file` and replacing `AerodynamicsParameters`, but `cpu/aircraft.py:47`,
`torch/altitude.py:143` and `AerodynamicsParameters.as_warp_struct()` still read those and the
plants are not touched until Phase 3. So the derivative loader was added *beside* the polynomial
one and the tree stays green after every phase. **Phase 3 must delete:**
`AerodynamicsParameters`, `AerodynamicsParametersStruct`, `poly_path`, and the
`poly_params_file` injection in `config.py`.

Landed:
- CSVs moved to `data/Volantex_Ranger/Volantex_Ranger_{derivatives,ge_derivatives}.csv`
- Volantex JSON precision fix (`area 0.273 -> 0.2733`, `mac 0.157 -> 0.156667`) + its config
  golden, which was the only test to fail on the change
- `config.py`: `derivatives_path`, `ge_derivatives_path`, `has_derivatives`, `read_derivatives`,
  `DUPLICATED_GEOMETRY` + `_reject_drift` (Phase 1b), wired into `load()`
- `params.py`: `CHANNELS` (27, fixed order), `IDX`, `_induced_drag_factor`,
  `DerivativeAeroParameters`
- `tests/test_aircraft.py`: derivative-file presence gated on `has_derivatives`, three drift tests,
  `_stage(with_derivatives=)`
- `tests/test_derivative_aero.py`: new, the loader's invariants

Verified on the real data: increment at the top row is **exactly** zero and
`k_ind[-1] == k_ind_free` **exactly**, so the anchoring gives free air above the table with no
step. `k_ind` runs 0.0320 -> 0.0429 across the sweep, confirming the per-height precompute was
necessary. Drift gate fires on each of span/area/mac; airframes without CSVs still load.

One Phase 5B gate pulled forward: `test_trim_table_matches_golden` is now
`xfail(strict=False, reason=GOLDENS_PENDING)`. The Volantex precision fix moves its trim thrust
~0.07% against a 1e-9 tolerance. **Checked before gating: all 72 diffs are Volantex rows and the
other four airframes reproduce the golden exactly**, so nothing else moved. `GOLDENS_PENDING` in
`tests/test_benchmark_ge.py` is the shared reason string Phase 5B should reuse.

The remaining result goldens are deliberately NOT gated yet — they still pass, and they are the
only regression signal that Phases 2-4 have not disturbed the other four airframes. Gate them in
Phase 5 when the aero swap actually breaks them, not pre-emptively.

Suite state after Phase 1: `pytest -m "not slow"` = **148 passed, 6 skipped, 1 xfailed**.

### Phase 2 — next

## Decisions taken (by the user, 2026-09-10)

| Decision | Answer |
|---|---|
| Backend scope | **All four**: CPU reference plant, warp GPU plant, torch batched plant, MPPI kernels |
| Replace or coexist | **Replace the polynomial outright** |
| Empirical ground effect | **Gone entirely** — not even retained as a comparison curve |
| Argument convention | **Named struct**, not positional vectors |
| GE table low end | **Clamp at the table floor and document it** |
| GE table high end | **Additive increment anchored at the table top** — no high clamp (see Phase 3) |
| Geometry authority | **Aircraft JSON wins; the loader RAISES on any mismatch with the CSV** |
| `cg_offset_vector` | **Deleted everywhere.** No legacy code retained |
| Stall | **No stall model, by design.** Hard `alpha_max_deg` termination only |
| Alpha thresholds | `alpha_max_deg = 19.5` (hard gate) + `alpha_soft_deg = 13.0` (reward penalty) |
| Unused derivatives | **Intentional** — OpenVSP noise that should be zero. Use the given equations as-is |
| `v_c` floor | **0.1 m/s** (as specified; the code being replaced used 1.0) |
| Four airframes without derivative CSVs | **Skip their tests on file absence**; user adds CSVs later, then we review |
| Paper archive result goldens | **Regenerate NOTHING this pass** — gate them; user will retrain first |
| `test_benchmark_ge` theory line | Replace with the **table's own CL/CD ratio** as the reference |

---

## What the patch touches

The coefficient model is duplicated in five places:

| # | Location | What it holds |
|---|---|---|
| 1 | `src/falcons/sim/cpu/physics/aerodynamics.py` | `PolynomialAerodynamics` + empirical `mu_l/mu_d` (reference plant) |
| 2 | `src/falcons/sim/warp/aerodynamics.py:408` + `physics.py:166` | `compute_all_coeffs`, `compute_mu_l/mu_d` (GPU) |
| 3 | `src/falcons/sim/torch/altitude.py:142-145,206-223` | inline poly + inline `mu_l/mu_d` |
| 4 | `src/falcons/controllers/mppi/kernels.py:183` | imports the warp `compute_all_coeffs` |
| 5 | `src/falcons/benchmark/ground_effect.py:44-57` | numpy replica of `mu_l/mu_d` for the theory line |

Plus `tools/jsbsim_validate/gen_jsbsim.py`, which exports the *polynomial* to JSBSim 2-D tables and
has no direct equivalent for a linear derivative set.

---

## Phase 1 — data + loader

- `derivatives.csv` → `src/falcons/aircraft/data/<plane>/<plane>_derivatives.csv`
- `ge_derivatives.csv` → `src/falcons/aircraft/data/<plane>/<plane>_ge_derivatives.csv`

Volantex_Ranger's are ready (currently at repo root). The other four arrive later.

**Policy for the missing four** — `PLANES` keeps all five entries; nothing is deleted. Instead:

- `AircraftConfig` exposes `derivatives_path` / `ge_derivatives_path` unconditionally and raises a
  clear "no derivative data for <plane>; supply <path>" only when a caller actually tries to
  **load** that airframe. Path construction never fails.
- Tests parametrised over `PLANES` get `@pytest.mark.skipif(not path.exists(), reason=...)` keyed on
  file existence — `test_files_present`, `test_params_match_archive`,
  `test_every_shipped_aircraft_is_complete`, and the per-plane benchmark cells.
- This is self-healing: when the CSVs land the skips become runs with no code change, which is the
  point at which the user wants to review what happens.
- Volantex_Ranger is the only airframe expected to pass end to end during Phases 1-4. Treat any
  green result on the other four as suspicious.

- `aircraft/config.py`: add `derivatives_path` / `ge_derivatives_path` alongside `poly_path`; drop
  the `poly_params_file` injection in `load()` (`config.py:68`). Update
  `test_aircraft.py::test_files_present`.
- `aircraft/params.py`: replace `AerodynamicsParameters` with a derivative loader.
  - `derivatives.csv` is long-format `name,value` → dict. Exclude `SM`, `X_np`, `StabilityType`
    as metadata, not coefficients.
  - `ge_derivatives.csv` is keyed by **`h_m` (metres), not h/c** — 11 heights x 62 names. Pivot to
    an `(11, 28)` array ordered by the file's own `h_over_c` row (2.068 -> 20.681 ascending, equal
    to `h_m / FC_Cref_`).
  - **`k_ind` precomputed per height as a 28th channel**, not as one free-air scalar: it varies
    0.0317 -> 0.0426 across the table (26%), so a single scalar is *not* equivalent to the MATLAB,
    which rebuilds `k_ind` from the interpolated `c`. Interpolating it is exactly equivalent and
    still costs zero divides per step. Guard against `CL_Total -> 0` for future airframes.
  - Operating point (`alpha_run` from `FC_AoA_ = 3.9124 deg`, `de_run` from
    `deflect_elevator_deg = -4.4425 deg`) resolved to radians once at load.
  - The `*_h` rows (`CL_h`, `CD_h`, ...) are loaded but not applied — the target form does not use
    them; interpolating `*_Total` over the table already captures dC/dh.

### Phase 1b — the drift check (the JSON is the reference)

The aircraft JSON is authoritative. Any field duplicated between the JSON and `derivatives.csv`
must agree, and the loader **raises** naming the field, the JSON value, and the CSV value, so a
config that has drifted from the geometry which generated the CSV fails loudly instead of flying
silently wrong. Non-dimensionalisation then uses the JSON's values.

Pairs to check: `wing.span` vs `FC_Bref_`, `wing.area` vs `FC_Sref_`, `wing.mac` vs `FC_Cref_`.

> **Volantex_Ranger already drifts and will fail this check on day one.** Fix the JSON first:
> `mac: 0.157 -> 0.156667` and `area: 0.273 -> 0.2733`. Both `tests/golden/configs/Volantex_Ranger.json`
> and `test_params_match_archive` move with it. Do the same audit for the other four airframes when
> their CSVs land.

Exact equality is the rule the user chose (no tolerance), so the JSON must carry the CSV's full
precision. Compare on the parsed float, and put the CSV value in the error message so the fix is
copy-pasteable.

## Phase 2 — the named-argument contract

One field naming shared by all four backends:

```
AeroInputs:  alpha, beta                 (rad)
             elevator, aileron, rudder    (rad)
             p, q, r                      (rad/s)
             h                            (m, = -pos[2])
AeroCoefs:   CD, CS, CL, Cl, Cm, Cn
```

CPU/torch get a `NamedTuple`; warp gets a `wp.struct` with the same field names.

This exists to kill an ordering conflict permanently: the repo uses
**`[elevator, aileron, rudder]`** everywhere (`cpu/physics/aerodynamics.py:80-82`,
`warp/physics.py:167`, the JSON `aero_surfaces` order), while the MATLAB spec reads
`deflect_angs(1)=aileron, (2)=elevator, (3)=rudder`. A positional vector would silently swap
elevator and aileron.

Actuators emit **degrees** (`cpu/actuators.py:186`, and `torch/altitude.py:206` comments the same),
so the deg -> rad conversion happens once at the boundary, never inside the model.

## Phase 3 — the model, four times

The target function verbatim in structure, with `k_ind` interpolated rather than divided per step.

### Ground effect as an additive increment (replaces the spec's high-clamp)

Instead of "interpolate below the table top, free air above":

```
c(h) = c_free + ( c_ge(h/c) - c_ge(h/c_top) )
```

- At and above the table top the increment is identically zero, so **100 m gives exactly the
  free-air set** — no residual ground effect, no clamp, no C0 jump.
- Inside the table the measured *shape* of the GE variation is preserved exactly.
- It cancels the table-top offset (`CMm_Total = -4.4e-4` at the top vs `+5e-7` free air) rather
  than papering over it.
- **Additive, not multiplicative**: a ratio form divides by coefficients that pass through zero
  (`CMm_Total`, `CMl_Total`, `CMn_Total` are all ~0 at the symmetric run point) and blows up.
- Applies to `k_ind` too: interpolate the increment, not the value.
- Low end unchanged: clamp at `h/c = 2.068`, so the increment freezes at its deepest measured
  value below the table floor.
- Accepted cost: this asserts GE is fully dead at the table top (`h/b = 2.0`), understating it by
  <= 0.14% in CL everywhere inside the table. Re-running the tool to `h/b ~ 4-6` would remove even
  that, if it ever matters.

1. **CPU** (`cpu/physics/aerodynamics.py`): delete `PolynomialAerodynamics`,
   `ClassicalAerodynamics`, `AerodynamicsFactory`, `get_ground_effect_factors`. One interp over the
   stacked table instead of 28 scalar `interp1` calls.
2. **Warp** (`warp/aerodynamics.py`): delete `compute_all_coeffs`, `compute_mu_l`, `compute_mu_d`,
   and the `Clp/Cmq/Cnr` block at `:187-189`. Needs a hand-written linear-interp `wp.func` over a
   `wp.array2d` — warp has no array `interp1`; a bounded scan over 11 rows beats a binary search at
   that size. `physics.py` steps 3-5 collapse into one call; `in_ground_effect` -> `ge_enable`.
3. **Torch** (`torch/altitude.py:142-145, 206-223, 239-241`): inline poly, inline `mu_l/mu_d`, and
   the separate damping all go; batched interp via `torch.searchsorted`.
4. **MPPI** (`controllers/mppi/kernels.py:132, 183`): follows warp; drop its `Clp/Cmq/Cnr` params.

> **Main correctness risk: double-counted rate damping.** `CL_q, CMm_q, CMl_p, CMn_r, CS_p, CS_r`
> now live *inside* the returned coefficients. Every separate `Clp/Cmq/Cnr` term must go — including
> the keys in all five aircraft JSONs, `params.py:101-103,137-139,157`, and the `config.py:11-13`
> comment that calls their `0.0` default "neutral rather than borrowed".
> Measured `CMm_q = -24.65` vs the JSON's `Cmq = -12.0`: pitch damping roughly doubles. This is a
> visible dynamics change, not a pure refactor.

### Phase 3b — delete `cg_offset_vector` entirely

No legacy code retained. Its only consumers were the empirical GE height offset and the crash
check. Sites:

- `aircraft/params.py:16-17` (`cg_z_offset`, `cg_offset_vector` in `WingParametersStruct`),
  `:62`, `:70`, `:78`, `:85-86`
- `aircraft/config.py:22` — drop from `REQUIRED`
- All five `aircraft/data/<plane>/<plane>.json` and all five `tests/golden/configs/<plane>.json`
- `cpu/physics/aerodynamics.py:138` — dies with the empirical GE model
- `warp/aerodynamics.py:9-10,21-22,48-49` — dies with `compute_mu_l/mu_d`
- `benchmark/ground_effect.py:69,77,145,154,226,229-240,323,332` — `cg_z`, `geometry()`'s 4-tuple,
  `theory_pct`, `thrust_saving`; `tests/test_benchmark_ge.py:16-19` follows
- `torch/altitude.py:119,216,400`

> **The crash check changes meaning.** Today it is wing height:
> `-pos[2] + cg_offset_vector[2] < 0` (`cpu/aircraft.py:193`, `warp/termination.py:40`,
> `torch/altitude.py:400`, `envs/cpu.py:132`, `envs/altitude.py:350`, `envs/attitude.py:213`,
> `mppi/altitude.py:205`). It becomes CG height: `-pos[2] < 0`. Numerically identical for
> Volantex, Navion and Cirrus (z offsets 0, 0, 0.035) but **not** for Airship_V7, whose offset is
> -0.293 m — its crash altitude shifts by 29 cm. Flag when regenerating V7's goldens.

### Phase 3c — alpha thresholds: two explicit values, no multipliers

`stall_angle_deg` is deleted. Two new `aero_params` keys replace it:

- **`alpha_max_deg = 19.5`** — the single hard termination limit. Today's value in the envs, the
  torch plant and `envs/cpu.py` (`1.5 x 13`), so the 39 checkpoints were trained against it.
- **`alpha_soft_deg = 13.0`** — the saturation point of the reward penalty, preserving today's
  9.1-13 deg band (onset stays `alpha_safety_start x alpha_soft_deg`, default 0.7). Kept separate
  precisely so rebasing the gate does not silently slide the penalty band out to 13.65-19.5 and
  weaken it.

The soft penalty **stays**. It is a training aid, not physics, and it is what actually holds
policies inside the fitted envelope — `mppi/cost.py:189-193` says so outright. Deleting it would
change the reward on top of the physics change and make the regenerated goldens uninterpretable.

Sites to rebase (all multipliers removed):

| Site | Today | Becomes |
|---|---|---|
| `warp/termination.py:43` ← `envs/altitude.py:349`, `envs/attitude.py:212` | `radians(1.5 x stall_deg)` = 19.5 | `alpha_max_deg` |
| `torch/altitude.py:108,401` | `radians(1.5 x stall_deg)` = 19.5 | `alpha_max_deg` |
| `envs/cpu.py:137` | `1.5 x stall_angle_deg` = 19.5 | `alpha_max_deg` |
| `cpu/aircraft.py:199-200` | `2.0 x stall_angle_deg` = **26** | `alpha_max_deg` |
| `warp/termination.py:20` ← `mppi/altitude.py:205` | `2.0 x (2 x stall_angle)` = **52** | `alpha_max_deg` |
| `benchmark/ground_effect.py:124` | `stall_angle` = 13 (trim-validity filter) | `alpha_soft_deg` |
| `warp/altitude_reward.py:75-81`, `torch/altitude.py:308-311`, `envs/altitude.py:581` | `stall_deg` | `alpha_soft_deg` |
| `mppi/cost.py:185-197`, `mppi/attitude.py:96` | `alpha_limit` = `VP.stall_angle` | `alpha_max_deg` |

> **Bug found, fix regardless of this refactor:** `warp/termination.py:20` applies `2.0 x
> stall_angle` while its caller `mppi/altitude.py:205` already passes `2*VP.stall_angle`, so the
> MPPI path terminates at **52 deg** AoA, not 26. Also note `params.py:302,341-342,362` carries a
> second `stall_angle` (radians) alongside the degrees version — both go.

Honesty flag recorded: a derivative set linearised at alpha = 3.9 deg is not credible at 19.5 deg.
19.5 was chosen to preserve trained behaviour, not because the model is valid there.

## Phase 4 — downstream contract

- Return keys become `CD, CS, CL, Cl, Cm, Cn` plus `CL_free, CD_free` and
  `CL_ratio_ige, CD_ratio_ige` as diagnostics. **No `mu_l`/`mu_d` anywhere** — the empirical model
  is gone, so its names go with it.
- `cpu/aircraft.py:175-177` history: `oge_ige` logs the free-air vs in-GE pair from the tables.
- `tools/jsbsim_validate/validate.py:266-267` reads `coeffs["mu_l"]`/`["mu_d"]` → switch to the
  ratio keys.
- `tools/jsbsim_validate/gen_jsbsim.py`: no derivative-set equivalent to its 2-D table export. It
  can emit `<function>` blocks for a linear set, but that rewrites `check_layout`/`emit` — do it as
  a **separate commit after the plants are green**.
- `benchmark/ground_effect.py`: delete `_mu_l`/`_mu_d`. `theory_pct` becomes the table's own
  `CD_Total(interp) / CD_free` ratio. `RATIOS[0]` moves 0.25 -> 0.20 (the table floor,
  `h/c = 2.068`). `test_benchmark_ge.py::test_theory_is_lifting_line_shape` becomes a monotonicity
  + OGE-convergence check against measured data.

## Phase 5 — gate the goldens, regenerate nothing

The user will retrain before any results are re-archived, so **no result golden is regenerated in
this refactor.** Do not "helpfully" refresh one. Three categories, treated differently:

### A. Config snapshots — DO regenerate (they are schema mirrors, not results)

`tests/golden/configs/<plane>.json` mirror the aircraft JSONs, whose schema changes here:
`cg_offset_vector` removed, `stall_angle_deg` -> `alpha_max_deg` + `alpha_soft_deg`,
`poly_params_file` -> the derivative paths, `Clp/Cmq/Cnr` removed. Regenerate alongside the JSON
edits so `test_params_match_archive` stays meaningful. Volantex also takes the `mac`/`area`
precision fix from Phase 1b.

### B. Result goldens — FROZEN, gated with an explicit reason

| File | Test |
|---|---|
| `altitude.csv` | `test_benchmark_altitude.py:22,34` |
| `ge_trim.csv`, `ge_energy.csv` | `test_benchmark_ge.py:27,34` |
| `maneuvers.csv`, `mppi_variance.json` | `test_benchmark_maneuvers.py`, `test_train.py` |
| `robustness.csv` | `test_benchmark_robustness.py:51` |
| `tables.tex` | `test_tables.py` (emitted from the five CSVs) |
| `warp_step.npz` | `test_sim_warp.py:260` (one-step physics regression) |
| `attitude_obs.npy` | attitude observation snapshot |
| `throughput.json` | `test_benchmark_throughput.py` |

Mark each `xfail(strict=False)` or `skip` with the same reason string — something like
`"aero model replaced; goldens pending retrain (plan.md Phase 5)"` — so one grep finds every
frozen gate when the retrain lands. `test_diff.py:21` asserts every `TOLERANCE` entry has a golden
CSV beside it; check it still holds.

**Why not regenerate `ge_trim.csv` even though it needs no policy** (open-loop trim solve,
`benchmark/ground_effect.py::run_trim`): `tables.tex` is emitted from all five CSVs together, so a
fresh `ge_trim` beside four stale policy-derived CSVs yields a table mixing two different plants —
worse than leaving all five untouched. Regenerate the set together, after retraining.

### C. Derived flight inputs — a real problem, see open items

`aircraft/data/<plane>/<plane>_K_LQR.csv` and `<plane>_K_LQR_acquisition.csv` are **flight inputs**,
not goldens: the LQR controller flies them, and `lqr/attitude.py:45` states they "linearize the
plant AROUND trim". Replacing the plant invalidates them, so the LQR path flies gains designed for
the polynomial aero. `tests/golden/lqr_gains/` is only a mirror
(`test_controllers.py::test_lqr_gains_identical_to_archive`).

> **The derivation is not in the repo.** `lqr/attitude.py:62` points at
> `scripts/derive_lqr_gains.py`, but `scripts/` contains only `setup_venv.sh` and
> `train_seeds.sh`. There is no in-repo way to re-derive the gains.

Recommendation: gate the LQR path as known-invalid for now (it is not on the critical path for
Phases 1-4), and treat re-deriving the gains as a separate work item. Upside worth noting: a linear
derivative set gives the state-space A/B matrices analytically, so the missing script is easier to
write against the new model than against the polynomial one — no numerical Jacobian needed.

### Retraining, when the user gets to it

PPO/SAC/TD3 x {altitude, attitude} x planes x 3 seeds = the 39 checkpoints in `checkpoints/`, all
trained under polynomial aero. `scripts/train_seeds.sh` is the entry point.
Environment: `.venv/bin/python` has warp 1.8.1 and CUDA (RTX 4070 Ti SUPER, 16 GB). Plain `python`
has no warp.

---

## Assumptions carried (change these only deliberately)

- **Reference geometry from the aircraft JSON**, which raises on any disagreement with the CSV —
  see Phase 1b. (This reverses an earlier assumption that the CSV won.)
- **Height datum `-pos[2]`**, per the spec's note that `FC_Zcg_ = 0`. With `cg_offset_vector`
  deleted this is now the only height in the codebase.
- **GE floor is real and documented.** The table starts at `h/c = 2.068` (`h/b = 0.20`,
  `h = 0.324 m`). Below that the increment freezes. Takeoff/landing (h -> 0) therefore flies a
  frozen `h/b = 0.20` set. If deep GE matters later, the OpenVSP tool must be re-run at
  `h/c ~ 0.25-2`.
- **No stall model at all, by design**, and no Re/Mach dependence. The set is a linearisation about
  `alpha = 3.91 deg`, `V = 10.875 m/s`; nothing limits lift at high incidence except the hard
  `alpha_max_deg` termination. Envs default to 15 m/s — fine for an incompressible VLM, but say so
  in the docstring.
- **~23 measured derivatives stay unused** (`CMm_Beta`, `CL_p`, `CS_Alpha`, `CMl_q`, `CD_Beta`, ...)
  and this is **intentional**: they are OpenVSP numerical noise that should be zero, not physics
  being discarded. Implement the supplied equations exactly; do not "improve" them by adding terms.
  Say this in the docstring so a later reader does not restore them.
- **`v_c` floor 0.1 m/s** as specified, replacing the 1.0 used by the code being deleted
  (`warp/aerodynamics.py:186`).
- `C_D/C_L/C_m` carry a `*_Total` trim term; `C_S/C_l/C_n` do not. Correct — those are ~0 at the
  symmetric run point (`CS_Total = 2.3e-4`, `CMl_Total = 2.1e-5`, `CMn_Total = -1.1e-4`).

## Open items (not blocking Phases 1-4)

1. **Derivative CSVs for Airship_V7, Airship_A0S, Navion, Cirrus_SR22.** User supplies later; tests
   skip on absence until then. Review the drift check (Phase 1b) against each when they land — the
   JSON geometry will likely need precision fixes as Volantex did.
2. **LQR gains have no in-repo derivation** (Phase 5C). Decide whether to write
   `scripts/derive_lqr_gains.py` against the new model or leave the LQR path gated.
3. **Retraining + result-golden regeneration** (Phase 5B), after the user retrains.
4. **`gen_jsbsim.py` derivative export** (Phase 4) — separate commit once the plants are green.

## Sanity checks already done on the data

- Every field the target form needs exists in `ge_derivatives.csv` (62 names x 11 heights, no gaps).
- Deflection derivatives are **per radian**, not per degree: `CMm_elevator = -1.537` gives
  `dCm = -0.54` at 20 deg (plausible); per-degree would give -30 (absurd). Consistent with
  `de_run = deg2rad(deflect_elevator_deg)`.
- `k_ind = CD_Alpha / (2*CL_Total*CL_Alpha) = 0.0429` free-air. Implied polar curvature
  `2*k_ind*CL_Alpha^2 = 2.69 /rad^2` against `2*CL_Alpha^2/(pi*e*AR) = 2.30 /rad^2` for `e = 0.9`,
  `AR = 9.61` — same ballpark, so the formula is self-consistent.
- Static signs are conventional: `CS_Beta = -0.271`, `CMl_Beta = -0.054` (stable dihedral),
  `CMn_Beta = +0.083` (stable weathercock).
