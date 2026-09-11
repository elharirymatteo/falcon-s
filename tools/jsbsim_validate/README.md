# Validating FALCON-S against JSBSim

An optional side tool. It writes a JSBSim aircraft carrying a FALCON-S airframe's mass, inertia,
geometry and OpenVSP derivative aerodynamics, then flies both simulators from the same initial
condition with the controls at zero and the throttle shut, and reports where they disagree.

Nothing here is imported by the package, the CLI or the benchmark, and `jsbsim` is not in
`requirements.lock`. Deleting the directory changes nothing.

## Install

Into a fresh venv — the CPU plant needs only numpy, scipy and pandas, so neither torch nor Warp has
to be present:

```bash
python3.10 -m venv .venv-jsbsim
.venv-jsbsim/bin/pip install numpy==2.2.6 scipy==1.15.3 pandas==2.2.3
.venv-jsbsim/bin/pip install -r tools/jsbsim_validate/requirements.txt
.venv-jsbsim/bin/pip install -e . --no-deps          # for `import falcons`
.venv-jsbsim/bin/pip install matplotlib==3.10.3 pillow==11.2.1   # only for --plot
```

That the CPU plant runs without Warp is a property worth keeping, and it is load-bearing here: an
independent check is worth less if it shares a GPU stack with the thing it is checking. It is why
`falcons/aircraft/derivatives.py` exists separately from `params.py` — the derivative set loads
with numpy and pandas alone, and Warp is imported only inside `as_warp_struct`.

## Run

```bash
python tools/jsbsim_validate/gen_jsbsim.py --plane Navion      # -> aircraft/Navion_falcons/
python tools/jsbsim_validate/validate.py   --plane Navion --plot
```

`gen_jsbsim.py` writes the aircraft; `validate.py` flies it. Regenerate whenever an airframe's JSON
or derivative CSVs change — the coefficients are baked in. Both take `--plane` for any of the four
airframes. `validate.py --help` lists the initial condition options: `--altitude`, `--attitude`,
`--speed`, `--alpha`, `--rates`, `--seconds`. A full run is about ten seconds.

`tests/test_jsbsim_export.py` checks the *export* on every commit and needs no JSBSim: it parses
the emitted `<function>` blocks, evaluates them independently, and holds them to the plant at
1e-12. What it cannot check is what JSBSim does with a coefficient once it has one — the axis
conventions, the wind-to-body rotation, the reference lengths. That is check 2 below, and it needs
the real thing.

## What comes out

CSV for every check lands in `out/`. `--plot` adds four figures, one per claim a validation section
has to make, written as PDF for embedding and PNG for looking at. Sized for a two-column paper:
7.0 in across the text block, 3.4 in for a single column, 8 pt type.

| figure | the claim | what is in it |
| --- | --- | --- |
| `fig1_rollout` | the two simulators fly the same aeroplane | altitude, airspeed, alpha and pitch rate across the top; the difference in each underneath on a log axis, at both plant steps. The top row is the evidence that this is a real manoeuvre and not a trim point; the bottom row is the accuracy |
| `fig2_aero` | the aerodynamic model transferred | the coefficients from both sources (line and circles); what is left over; and the relative body-frame load error, explained below |
| `fig3_ground_effect` | ground effect is the one difference that is meant to be there | the ratio the two are measured to differ by against what FALCON-S's own height sweep asks for, and what it does to a release a fifth of a span off the ground |
| `fig4_time_step` | what limits the agreement is FALCON-S's own step | error against the plant's own `dt/64` solution, log-log, with a first-order reference line. JSBSim is not involved, so the slope is about the integrator alone |

### The right-hand panel of `fig2`, in full

It is a static comparison, no integration involved. Both simulators are placed at the *same*
`(alpha, beta, V)` — a sweep of alpha from −12° to 12° at 50 m/s, controls at zero — and asked for
the aerodynamic force and moment they would apply, in body axes. Each point is

    | JSBSim's vector − FALCON-S's vector |  /  | the vector itself |

so a value of 1e-8 means the two agree to eight significant figures on a load of tens of
kilonewtons, and 1e-1 means they disagree by ten per cent of it. Two sweeps are drawn, one at
`beta = 0` and one at `beta = 10°`, for force and for moment separately — four curves.

All four land in the same band, which is why the sideslip series is drawn with open markers —
otherwise it hides the zero-sideslip one underneath. Everything that goes into a load agrees: the
coefficients, the dynamic pressure, the reference area and lengths, the moment reference point, the
axis conventions.

That was not true when this tool was written. The sideslip **force** error stood seven decades above
the rest of the panel while the **moment** error did not move at all, and that asymmetry is what
localised the fault: moments are handed over in body axes and need no rotation, forces are computed
in wind axes and rotated, so only the rotation could be wrong. It was — the wind-to-body matrix
carried the wrong sign of beta. This panel is the guard against it coming back, and the two beta
blocks in check 2's printed output are the same guard without the plotting dependency.

`--diagnostic-plots` additionally writes the every-channel figures (`diag_states_oge`,
`diag_states_ige`): fourteen panels each, for working out where an unexpected number in the printed
table came from. They are how the lateral departure on the rotating earth was found, and they are
not paper material.

Two things to know before reading any of them:

* **`v`, `beta`, `p` and `r` are flat zero on both sides.** Wings level with no lateral forcing,
  neither simulator has anything to excite the lateral dynamics with, and they agree there to 1e-13.
  In the diagnostic figures those panels carry a minimum y range so that machine noise reads as a
  flat line instead of autoscaling to fill the axes and look catastrophic; the panel title always
  gives the real number. This is what the non-rotating planet buys — on the real earth the same
  panels show JSBSim's 1.6e-3 m/s of Coriolis, which then departs into a spin.
* **The rollout is six seconds by default.** With the elevator centred the airframe is not trimmed
  and the run is a real manoeuvre rather than a hold. The derivative set carries pitch damping,
  so alpha wanders rather than diverging: over six seconds from 200 m it stays inside ±9° on every
  airframe. `fig1` plots alpha unwrapped, which keeps the residual free of wrap spikes if a run
  ever crosses ±180° and changes nothing when it does not.

## Statistics printed

Every comparison prints bias, RMS, worst case, and the worst case as a percentage of that channel's
own peak-to-peak over the same window:

* **bias against RMS** separates a constant offset — a mismatched gravity, a units slip — from a
  growing divergence. They are different faults and need different fixes.
* **worst case as a percentage of span** is what makes a number readable without knowing the
  airframe. 0.15 m of altitude error means nothing on its own; 0.25 % of the altitude the
  manoeuvre covers does.

## What transfers, and what deliberately does not

The aerodynamic model is linear in six deltas about one measured operating point, so it maps onto
JSBSim arithmetic directly — `<sum>` of `<product>` terms — with **no tables and no interpolation
anywhere**:

| coefficient | JSBSim axis | terms |
| --- | --- | --- |
| CD | DRAG | `CD_Total + CD_Alpha·da + k_ind·(CL_Alpha·da)² + CD_elevator·de + CD_q·q̂` |
| CS | SIDE | `CS_Beta·β + CS_aileron·δa + CS_rudder·δr + CS_p·p̂ + CS_r·r̂` |
| CL | LIFT | `CL_Total + CL_Alpha·da + CL_elevator·de + CL_q·q̂` |
| Cl | ROLL | `CMl_Beta·β + CMl_aileron·δa + CMl_rudder·δr + CMl_p·p̂ + CMl_r·r̂` |
| Cm | PITCH | `CMm_Total + CMm_Alpha·da + CMm_elevator·de + CMm_q·q̂` |
| Cn | YAW | `CMn_Beta·β + CMn_aileron·δa + CMn_rudder·δr + CMn_p·p̂ + CMn_r·r̂` |

where `da = α − α_run` and `de = δe − δe_run` are offsets from the linearisation point, and
`p̂ = p·b/2V`, `q̂ = q·c̄/2V`, `r̂ = r·b/2V` use JSBSim's own `bi2vel`/`ci2vel`. Coefficients come
from the same `DerivativeAeroParameters` the plant loads, so they cannot drift from the model they
stand for.

Rate damping transfers with everything else: `CMl_p`, `CMm_q`, `CMn_r`, `CL_q`, `CD_q`, `CS_p` and
`CS_r` are members of the measured set and every FALCON-S backend applies them, so the export
carries them and this aircraft is a reference for all four backends.

Left out on purpose:

* **Ground effect.** JSBSim is the out-of-ground-effect reference that FALCON-S's measured height
  sweep is judged against. Giving JSBSim a ground-effect model of its own would defeat the
  comparison, so height enters only through FALCON-S — that is what checks 4 and 5 measure. The XML
  carries the free-air set.
* **Actuator and engine dynamics.** Controls are held at zero and the throttle shut throughout.
  Deflection limits are symmetric on every airframe, so a zero command is exactly zero and a `-1`
  throttle command is exactly zero thrust; both stay there for the whole run.

One place the transcription is deliberately not literal: `bi2vel`/`ci2vel` carry JSBSim's own
low-airspeed guard rather than FALCON-S's 0.1 m/s floor. The two differ only below a few m/s, which
is outside anything this validation flies.

## Making the two comparable

FALCON-S is a flat, non-rotating earth with one constant `g`. JSBSim integrates in an earth-centred
frame with a WGS-84 gravity model, so it carries Coriolis and centrifugal terms FALCON-S has no
equivalent of. **The fix is to take them away from JSBSim rather than to live with them**:
`nonrotating_planet.xml` redefines the planet with `rotation_rate` and `J2` at zero, loaded through
`FGFDMExec::LoadPlanet` before the aircraft. Earth's real GM and radii stay, so airspeed, altitude
and density keep their usual meaning. What that buys:

| | v after 2 s | p after 2 s |
| --- | --- | --- |
| the real rotating earth | −1.59e-03 m/s | 5.05e-03 deg/s |
| `rotation_rate = 0` | 7.80e-14 m/s | 1.22e-13 deg/s |

With no rotation the lateral channels agree to machine precision instead of to a floor.
With `J2 = 0` the WGS-84 and standard gravity models coincide, so `simulation/gravity-model` stops
mattering as well. `--rotating-earth` puts the real earth back, which is the way to measure what
FALCON-S's flat-earth assumption actually costs.

Two smaller things, both handled in `validate.py`:

* JSBSim's gravitation is read at the initial condition and forced on both plants — 9.7977 m/s² at
  200 m against FALCON-S's configured 9.81, which would otherwise integrate to more than a
  centimetre in a couple of seconds. Runs sit at the equator so that local down is the gravity
  direction. What remains is that JSBSim's gravity falls off with altitude while FALCON-S's does
  not: 3.1e-4 m/s² per 100 m of climb, four decades below anything else measured here.
* JSBSim runs at 1 ms against the plant's 10 ms. At 1 ms its trajectory is converged well below
  anything being measured, so its integration error is not part of the result.

## What the checks found

Measured against **JSBSim 1.3.1**, all four airframes, default initial condition (200 m, attitude
[0, −10, 30]°, 50 m/s at α = 2°, controls zero, throttle shut).

**1 and 2, the aerodynamics, agree to machine precision.** Every coefficient matches to
≤ 2.7e-15 absolute — round-off on a float64, not a modelling residual:

| airframe | max \|Δcoefficient\| | max load error, % of span |
| --- | --- | --- |
| Airship_V7 | 2.4e-15 | 3.1e-07 |
| Airship_A0S | 2.7e-15 | 3.3e-07 |
| Volantex_Ranger | 2.7e-15 | 2.8e-07 |
| Navion | 2.2e-15 | 4.6e-07 |

A linear derivative set is native JSBSim arithmetic, so the export carries no approximation of its
own and check 1 measures the transcription rather than the accuracy of a table. There is no knob to
trade against file size and nothing to tune; the whole aircraft is 16 kB.

The `beta = 0` and `beta = 10°` blocks of check 2 agree with each other, which is the standing
guard on the wind-to-body rotation.

**3, the rollout, is limited by FALCON-S's own time step.** Altitude error after six seconds, at
the plant's configured 10 ms and with the step divided by 20:

| airframe | h error @ 10 ms | @ 0.5 ms | α at t = 6 s |
| --- | --- | --- | --- |
| Airship_V7 | 5.8e-02 m | 4.6e-03 m | −2.6° |
| Airship_A0S | 1.7e-01 m | 1.7e-02 m | −3.3° |
| Volantex_Ranger | 2.6e-01 m | 2.4e-02 m | −5.5° |
| Navion | 1.5e-01 m | 1.9e-03 m | 8.8° |

Refining the step recovers roughly an order of magnitude, so what is being measured at 10 ms is the
plant's integrator and not a disagreement with JSBSim. The self-convergence ladder confirms it:
against the plant's own `dt/64` solution the observed order is 1.09–1.13, i.e. first order, which is
what the integrator is.

The α column is worth reading on its own. These runs are untrimmed with the elevator centred, so
nothing holds the aircraft at an attitude; the excursion stays bounded because the measured set
carries `CMm_q` and every backend applies it. An airframe whose α column ran away here would be one
whose pitch damping had gone missing somewhere between the CSV and the plant.

**4, ground effect, is the only difference that is meant to be there — and it is the whole of it.**
The check prints two ratios per height: FALCON-S-in-ground-effect over JSBSim, and
FALCON-S-in-ground-effect over FALCON-S free air. They agree to every digit printed, on all four
airframes, which says the JSBSim aeroplane and FALCON-S's free air are the same aeroplane and every
bit of the remaining difference is the height sweep. At `h/b = 0.20`, the lowest measured band:

| airframe | CL ratio | CD ratio |
| --- | --- | --- |
| Airship_V7 | 1.073 | 0.883 |
| Airship_A0S | 1.069 | 0.892 |
| Volantex_Ranger | 1.065 | 0.917 |
| Navion | 1.118 | 0.878 |

Both ratios reach exactly 1.000 at `h/b = 2.00` — the top of the measured sweep — with no clamp and
no step, which is the additive-increment anchoring working. Below `h/b = 0.20` the increment
freezes rather than extrapolating, which is why the 0.05, 0.10 and 0.20 rows are identical: the
sweep does not go deeper, and holding it is honest where extrapolating a measured table would not
be.

**5, the in-ground-effect rollout, diverges as it should.** Released a fifth of a span up, wings
level, after six seconds Navion is at 72.55 m against JSBSim's 76.28 m. The altitude bias is the
sign that the model is holding the aircraft up; JSBSim, having no ground effect, is not.
