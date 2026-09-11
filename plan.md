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

### Phase 2 — DONE

Landed `src/falcons/sim/aero_contract.py` (warp-free) + `tests/test_aero_contract.py` (15 tests):
`SURFACES`/`INDEX`, `AeroInputs`, `AeroCoefs`, `deflections_to_radians`, `aero_inputs`,
`check_surface_order`.

Two refinements made at the re-assessment gate:

1. **The warp `wp.struct` mirror is deferred to Phase 3**, where the funcs that take it are
   written. Defining it now needs a second, warp-importing module that Phase 3 would rewrite
   anyway, and it would ship with no consumer and no test. `warp-lang` *is* a hard dependency
   (pyproject), so this is a cleanliness call, not a portability one — the CPU reference plant
   and the torch plant stay warp-free.
2. **`check_surface_order` is not wired into `AircraftConfig.load()`.** Doing so would make
   `aircraft/` import `sim/`, inverting the existing dependency (`sim/cpu/aircraft.py` imports
   `falcons.aircraft.config`). Instead `test_aero_contract.py` checks all five shipped airframes,
   so a reordered JSON fails CI regardless. **Phase 3 must call `check_surface_order` at plant
   construction** in `cpu/aircraft.py`, `warp/aircraft.py` and `torch/altitude.py`.

Verified while re-assessing: all five airframes list `aero_surfaces` as
`elevator, ailerons, rudder` with matching `surface_type`, so there is no pre-existing
mis-indexing to unwind. The guard matches on `surface_type`, not the JSON key, so `ailerons` vs
`aileron` is tolerated but a reorder is not. Note the ordering was previously *implicit in JSON
key order* (`cpu/aircraft.py:110,121` just take `list(...keys())`) and unchecked anywhere.

Suite after Phase 2: **163 passed, 6 skipped, 1 xfailed**.

### Phase 3 — DONE

The model in all four backends, plus 3b (`cg_offset_vector`) and 3c (alpha thresholds).
`tests/test_aero_parity.py` is new and is the thing to trust: it holds warp and torch to the CPU
reference term for term, over five conditions (the run point, deep GE, mid-sweep, exactly the
table top, and below both the V and GE floors), with ground effect on and off. Warp matches at
float32 tolerance, torch at 1e-11.

Refinements made at the re-assessment gate:

1. **No warp `wp.struct` mirror of `AeroInputs`.** Warp kernels have no keyword arguments, so a
   struct cannot give warp the named safety the CPU/torch `NamedTuple` gives -- it only moves the
   positional risk to the struct's construction. `compute_all_coeffs` takes explicit named
   parameters instead, and Phase 2's promise is withdrawn rather than met cosmetically.
2. **`C_L_ige`/`C_D_ige` renamed to `C_*_free`** across warp (`aircraft.py`, `physics.py`,
   `history.py`). They are history-only, never read for physics, and "in ground effect" became a
   lie once GE moved inside the coefficients. The free-air pair costs almost nothing to compute
   (no table lookup) and is the diagnostic the history plots.
3. **`aspect_ratio`/`taper_ratio` kept** in the wing block though nothing reads them now. They
   describe the wing; `cg_offset_vector` was a ground-effect implementation detail. Flagged, not
   deleted. NOTE: Volantex's `aspect_ratio` 9.61 disagrees with `span^2/area` = 9.6026.
4. **`tools/jsbsim_validate` is guarded, not ported.** Both entry points raise `_NOT_PORTED`
   naming plan.md Phase 4. The polynomial CSVs and `AircraftConfig.poly_path` are KEPT: they are
   inert data, and deleting the cross-validation tool's source before its replacement exists
   would be careless. No dead code path remains in any plant.
5. **`cmd_check` picks an airframe that has data** instead of hardcoding Airship_V7, so the smoke
   check does not fail for a reason unrelated to what it checks.

Also landed: `tests/conftest.py` grew `ONLINE_PLANES` / `requires_derivatives()` / `plane_params()`,
and the config goldens were regenerated for Volantex (Phase 5A). `ARCHIVE_EXCLUDES` in
`test_aircraft.py` keeps the 11x27 tables and absolute paths out of the golden while still
comparing the scalars a bad CSV would move.

> **Coverage is materially reduced until the other four CSVs land.** Airship_V7 is the default
> airframe for `test_envs.py`, `test_sim_warp.py`, the MPPI tests and most of `test_controllers.py`,
> and the aero model IS the data, so all of those now skip: **59 skipped**, up from 6. This is the
> direct cost of "replace outright" with one airframe extracted. Nothing is silently passing.

Suite after Phase 3: **128 passed, 59 skipped, 1 xfailed** (`pytest -m "not slow"`).

### Phase 3d — the remaining airframes land — DONE (2026-09-11)

The user supplied derivative sets for Airship_V7, Airship_A0S and Navion, corrected the JSON
geometry to match, and **dropped Cirrus_SR22 from the project**. The polynomial CSVs and the
`K_LQR` gain tables were deleted along the way.

What landed:

- **Cirrus_SR22 removed** from `PLANES` (both copies), `envs/configs.py` (4 tables),
  `benchmark/{tables,figures}.py`, the tests and `tests/golden/configs/`. Its checkpoint
  (`ppo_altitude_Cirrus_SR22_s0.pt`) is left on disk — deleting a trained artefact is the user's
  call, and `test_thirty_nine_checkpoints_ship` still counts 39.
- **CSVs renamed on the way in.** The extractor emits bare `derivatives.csv`/`ge_derivatives.csv`;
  they now carry the airframe prefix. Not decoration: an unprefixed table in the wrong directory
  is invisible, and one already was — Navion arrived carrying Volantex's set, caught only because
  the Phase 1b geometry cross-check raised. The prefix makes that mistake self-evident.
- **`poly_path` / `poly_params_file` deleted** with the CSVs they pointed at.
- **LQR gates, not deletion.** The gains linearised the polynomial plant; they are gone and
  `lqr_gains_path` is now `None` everywhere. `test_lqr_gains_identical_to_archive` is skipped with
  `LQR_PENDING`, and `tests/golden/lqr_gains/` is kept as the record of what the old plant flew.
  The user is refactoring LQR next — see Phase 6.
- **Absent data and broken data are now reported apart** (`conftest._why_offline`). A missing CSV
  is expected mid-extraction; one that fails to load is a defect someone must fix. Skip reasons
  carry the actual exception.
- **`test_every_airframe_is_online` is the one test that FAILS rather than skips.** Everything else
  keyed on derivative data skips, so without it a broken export reads as a quiet green run.
- **A missing operating-point row raises** instead of defaulting to zero. `de_run` offsets every
  elevator command, so a wrong value biases the pitch axis by a constant — in trim, where it is
  least visible. This is what Navion is failing on now.
- **`tests/test_aero_parity.py` runs on every online airframe** (16 → 48 tests) with case heights
  expressed as h/b rather than metres. The rate terms scale by span on p/r and by MAC on q; with
  one airframe a span/MAC mix-up is a constant that every backend agrees on. Volantex and V7
  differ by 3x in span and 4.6x in MAC, which separates them.
- **`test_sim_warp.py`'s termination block was stale** — it still passed `WING_CG_Z` (8 args to a
  7-arg kernel) and `1.5 * STALL_DEG`. It had been skipping since Phase 3 because V7 was offline,
  so the Phase 3 signature change was never exercised. Now reads `alpha_max` from the airframe.
- Two more goldens gated: `warp_step.npz` and `attitude_obs.npy`. `GOLDENS_PENDING` moved from
  `test_benchmark_ge.py` to `conftest.py`, since three files need it and one grep should find all.

Verified: all three online airframes fly the CPU plant to finite state, and ground effect at
h/b = 0.20 gives +7.6%/+7.4%/+6.8% lift and −10.6%/−9.6%/−6.8% drag (V7/A0S/Volantex) — the right
sign and a plausible magnitude on each.

> **Navion is not flyable.** Its re-export is missing the three `deflect_*_deg` rows, so the
> elevator operating point is unknown. Six tests skip and `test_every_airframe_is_online` fails,
> by design. Everything else about it checks out.

**All four run points are trim points**, L/W = 1.0000 exactly. This is now a test
(`test_the_run_point_is_a_trim_point`), gated only on the CSVs existing — not on them loading —
because its job is to diagnose a bad export. Use `FC_Rho_`, not 1.225: Navion's condition is
5000 ft (ρ = 1.05554) at 73.152 m/s and α = −3.02°, and assuming sea level makes a correctly
trimmed set look 16% heavy. The other three are at sea level.

Suite after Phase 3d: **184 passed, 9 skipped, 3 xfailed, 1 failed** (`pytest -m "not slow"`).
The single failure is Navion; six of the nine skips are Navion, one is the LQR gate, two predate
this work (`trace fixture not present`).

### Phase 4 — DONE (2026-09-11)

`tools/jsbsim_validate` ported to the derivative set (user's decision, 2026-09-11) rather than
deleted. The `_NOT_PORTED` guards are gone from both entry points.

The port made the tool **smaller**, which was not the expectation going in. A linear set maps onto
JSBSim arithmetic directly — `<sum>` of `<product>` — so `gen_jsbsim.py` lost `check_layout`,
`coefficient_table`, `table_xml`, the four grid CLI options and the pandas dependency. Generated
aircraft went from ~250 kB of 2-D tables to 16 kB of exact expressions, and check 1 changed meaning
with it: it used to measure bilinear interpolation error (~1.3e-4, irreducible), and now both sides
evaluate the same linear expression, so it should agree to round-off and anything larger is a
transcription bug.

**Rate damping is now exported.** Its absence was a standing caveat — the polynomial had no rate
derivatives and the CPU plant applied none, so the JSBSim aircraft was a reference for the CPU
plant only, explicitly not for warp. `CMl_p`, `CMm_q`, `CMn_r`, `CL_q`, `CD_q`, `CS_p`, `CS_r` are
measured set members that every backend applies, so the export carries them and the caveat is gone.

`validate.py`: `falcons_polynomial` → `falcons_aero`; coefficient names `CY/CMx/CMy/CMz` →
`CS/Cl/Cm/Cn` to match the plant and the emitted XML; check 4 prints the table's own
`CL_ratio_ige`/`CD_ratio_ige` in place of `mu_l`/`mu_d`, with both the cross-simulator ratio and
the within-plant ratio side by side so a gap between them reads as a free-air disagreement rather
than a ground-effect one.

**`tests/test_jsbsim_export.py` is new (18 tests) and needs no JSBSim.** It parses the emitted
`<function>` blocks and evaluates them with an independent expression evaluator against the same
inputs the plant sees, over three airframes and three all-channels-live conditions, agreeing to
1e-12. This matters because JSBSim is not installed here, so the full validation could not be run:
the export is verified, the *flight* comparison is not. One test specifically asserts the
coefficients respond to body rates, which is the caveat that just went away and would otherwise
regress invisibly in any check flown at zero rates.

Stale generated aircraft: Cirrus and Navion deleted (Navion cannot be regenerated until its export
is fixed), the other three regenerated. `validate.py --plane` still defaults to Navion, which fails
with the missing-row message until then.

The tool README's "What the checks found" section is **marked as polynomial-era and unreproduced**
rather than rewritten. Those numbers describe the aeroplane FALCON-S used to be; re-measuring them
needs JSBSim installed and is not something to guess at.

### Phase 4b — all four airframes flying, and two bugs the coverage exposed — DONE (2026-09-11)

Navion re-exported with its `deflect_*_deg` rows, so **every airframe is online**. Suite is fully
green for the first time since the refactor began: **277 passed, 3 skipped, 3 xfailed, 0 failed**
(the skips are the LQR gate and two pre-existing `trace fixture not present`).

Two real bugs surfaced, both found by coverage that only existed once several airframes were live:

**1. The torch backend silently flew the wrong aeroplane.** `_tables()` cached the derivative set
keyed on `id(params)` — a memory address. CPython reuses addresses, so once one airframe's
parameters were freed the next airframe's object could be allocated at the same address and be
handed the first one's tables. Nothing raises; the plant flies, with another aeroplane's
aerodynamics. Demonstrated directly: three airframes in a row returned Volantex's coefficients.
This is what a benchmark sweep over airframes does in one process, and the single-airframe parity
test could not see it — it took the parity matrix growing to four airframes plus a garbage
collection. Now keyed on the two CSV paths, which determine the tables completely.
`test_torch_table_cache_survives_one_airframe_replacing_another` is the guard, and it was confirmed
to fail against the old code.

**2. The CPU plant had quietly acquired a Warp dependency.** `params.py` imports warp at module
scope for the GPU structs, and `cpu/physics/aerodynamics.py` imported `DerivativeAeroParameters`
from it, so the CPU reference plant could no longer be imported without a GPU stack. That broke
`tools/jsbsim_validate`, whose whole value is being an independent check — a reference sharing a
GPU stack with the thing it checks is a weaker reference. Fixed by splitting the warp-free half
into **`falcons/aircraft/derivatives.py`** (`CHANNELS`, `IDX`, `_induced_drag_factor`,
`DerivativeAeroParameters`, with warp imported locally inside `as_warp_struct`). `params.py`
re-exports, so no existing import site changed. Guarded by
`test_the_cpu_path_loads_without_warp`, which blocks warp in a subprocess.

Also: Cirrus and Navion JSBSim aircraft regenerated, `out/` artefacts regenerated, and all
polynomial-era description removed from `tools/jsbsim_validate` — README, docstrings, plot labels
and the tumble narrative, which had become factually wrong (see below).

### Phase 4c — the validation actually run — DONE (2026-09-11)

`.venv-jsbsim` has JSBSim 1.3.1 installed, so the cross-validation was **run for all four
airframes** rather than left unverified. Results are in `tools/jsbsim_validate/README.md`, measured
not claimed:

- **Coefficients agree to ≤ 2.7e-15** — round-off on a float64. The polynomial export could only
  reach ~1.3e-4, because it went through 2-D tables and bilinear interpolation of a curved function
  leaves a residual that trades against file size. A linear set is native JSBSim arithmetic, so
  check 1 now measures the transcription rather than the export's accuracy.
- **Body-frame loads agree to ≤ 4.6e-7 % of span**, with the `beta = 0` and `beta = 10°` blocks
  matching — the standing guard on the wind-to-body rotation.
- **The 6 s rollout is limited by the plant's own time step.** Refining it 20x recovers an order of
  magnitude, and the self-convergence ladder gives order 1.09–1.13, i.e. first order, which is what
  the integrator is.
- **Ground effect is the whole of the remaining difference.** Check 4 prints FALCON-S/JSBSim beside
  FALCON-S-in-GE/FALCON-S-free-air; they agree to every digit printed on all four airframes. Both
  reach exactly 1.000 at the top of the sweep, with no clamp and no step — the anchoring working
  end to end against an independent simulator.

One qualitative change worth recording: **the airframes no longer tumble.** Under the polynomial
an uncontrolled rollout went through several revolutions in six seconds, because that model had no
rate derivatives and the CPU plant applied no damping. The measured set carries `CMm_q`, so the
same initial condition now gives a bounded excursion (α within ±9° at t = 6 s on every airframe).
A large part of the tool's prose was written about that tumble and is now gone.

### Phase 6 — LQR against the derivative set (new, after Phase 4/5)

The user will refactor the LQR controller once the aero work is closed. The gains are deleted, the
path is gated, and the linear derivative set gives the state-space A/B analytically — no numerical
Jacobian — so `scripts/derive_lqr_gains.py` is now a tractable thing to write rather than a
reverse-engineering job. Gates to clear: `LQR_PENDING` in `test_controllers.py`.

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
