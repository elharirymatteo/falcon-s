# Validating FALCON-S against JSBSim

An optional side tool. It writes a JSBSim aircraft carrying a FALCON-S airframe's mass, inertia,
geometry and OpenVSP derivative aerodynamics, then flies both simulators from the same initial
condition with the controls at zero and the throttle shut, and reports where they disagree.

> **The "What the checks found" section below is from the polynomial-era run and has not been
> reproduced against the derivative model.** The tool was ported when the aerodynamics were
> replaced; the numbers in that section describe the aeroplane FALCON-S used to be. They are kept
> because the *method* findings still stand — the wind-to-body sideslip sign, the rotating-earth
> floor, the plant's first-order time-step error — but every figure quoted there needs re-measuring
> before it is cited. `tests/test_jsbsim_export.py` is what currently holds the export honest, and
> it needs no JSBSim.

Nothing here is imported by the package, the CLI or the benchmark, and `jsbsim` is not in
`requirements.lock`. Deleting the directory changes nothing.

## Install

Into the project venv, or a fresh one — the CPU plant needs only numpy, scipy and pandas, so
neither torch nor Warp has to be present:

```bash
python3.10 -m venv .venv-jsbsim
.venv-jsbsim/bin/pip install numpy==2.2.6 scipy==1.15.3 pandas==2.2.3
.venv-jsbsim/bin/pip install -r tools/jsbsim_validate/requirements.txt
.venv-jsbsim/bin/pip install -e . --no-deps          # for `import falcons`
.venv-jsbsim/bin/pip install matplotlib==3.10.3 pillow==11.2.1   # only for --plot
```

## Run

```bash
python tools/jsbsim_validate/gen_jsbsim.py --plane Navion      # -> aircraft/Navion_falcons/
python tools/jsbsim_validate/validate.py   --plane Navion --plot
```

`gen_jsbsim.py` writes the aircraft; `validate.py` flies it. Regenerate whenever an airframe's
JSON or derivative CSVs change — the coefficients are baked in. Both take `--plane` for any of the
four airframes. `validate.py --help` lists the initial condition options: `--altitude`, `--attitude`,
`--speed`, `--alpha`, `--rates`, `--seconds`. A full run is about ten seconds.

## What comes out

CSV for every check lands in `out/`. `--plot` adds four figures, one per claim a validation
section has to make, written as PDF for embedding and PNG for looking at. Sized for a two-column
paper: 7.0 in across the text block, 3.4 in for a single column, 8 pt type.

| figure | the claim | what is in it |
| --- | --- | --- |
| `fig1_rollout` | the two simulators fly the same aeroplane | altitude, airspeed, alpha and pitch rate across the top; the difference in each underneath on a log axis, at both plant steps. The top row is the evidence that this is a real manoeuvre and not a trim point; the bottom row is the accuracy |
| `fig2_aero` | the aerodynamic model transferred | the coefficients from both sources (line and circles); what is left over, which is interpolation only; and the relative body-frame load error, explained below |
| `fig3_ground_effect` | ground effect is the one difference that is meant to be there | the ratio the two are measured to differ by against what FALCON-S's own height sweep asks for, and what it does to a release a fifth of a span off the ground |
| `fig4_time_step` | what limits the agreement is FALCON-S's own step | error against the plant's own `dt/64` solution, log-log, with a first-order reference line. JSBSim is not involved, so the slope is about the integrator alone |

### The right-hand panel of `fig2`, in full

It is a static comparison, no integration involved. Both simulators are placed at the *same*
`(alpha, beta, V)` — a sweep of alpha from −12° to 12° at 50 m/s, controls at zero — and asked
for the aerodynamic force and moment they would apply, in body axes. Each point is

    | JSBSim's vector − FALCON-S's vector |  /  | the vector itself |

so a value of 1e-8 means the two agree to eight significant figures on a load of tens of
kilonewtons, and 1e-1 means they disagree by ten per cent of it. Two sweeps are drawn, one at
`beta = 0` and one at `beta = 10°`, for force and for moment separately — four curves.

What it shows:

* At `beta = 0`, force and moment both sit at 1e-8, which is round-off. Everything that goes
  into a load agrees: the tables, the dynamic pressure, the reference area and lengths, the
  moment reference point, the axis conventions.
* At `beta = 10°`, the same. All four curves land in the same band, which is why the sideslip
  series is drawn with open markers — otherwise it hides the zero-sideslip one underneath.
* That was not true when this tool was written. The sideslip **force** error stood seven decades
  above the rest of the panel, at 3e-2…3e-1, while the **moment** error did not move at all, and
  that asymmetry is what localised the fault: moments are handed over in body axes and need no
  rotation, forces are computed in wind axes and rotated, so only the rotation could be wrong.
  It was — see the finding below. This panel is now the guard against it coming back.

`--diagnostic-plots` additionally writes the every-channel figures (`diag_states_oge`,
`diag_states_ige`): fourteen panels each, for working out where an unexpected number in the
printed table came from. They are how the lateral departure on the rotating earth was found, and
they are not paper material.

Three things to know before reading any of them:

* **`v`, `beta`, `p` and `r` are flat zero on both sides.** Wings level with no lateral forcing,
  neither simulator has anything to excite the lateral dynamics with, and they agree there to
  1e-13. In the diagnostic figures those panels carry a minimum y range so that machine noise
  reads as a flat line instead of autoscaling to fill the axes and look catastrophic; the panel
  title always gives the real number. This is what the non-rotating planet buys — on the real
  earth the same panels show JSBSim's 1.6e-3 m/s of Coriolis, which then departs into a spin once
  the airframe tumbles.
* **The rollout is six seconds by default**, and `fig1` plots alpha **unwrapped**. With the
  elevator centred these airframes have no pitch trim and no rate damping, so an uncontrolled
  run tumbles whatever initial condition it is given — lowering the airspeed slows the rotation
  but does not stop it. Unwrapped, alpha becomes a monotone climb whose slope is the tumble rate
  (2600° over six seconds, so seven revolutions) instead of a sawtooth through ±180°, and the
  residual underneath is free of wrap spikes. The manoeuvre is aggressive and not a design
  choice; what matters is that both simulators follow it. Over those six seconds the refined run
  holds every channel inside 1 % — `h` to
  2.1e-2 m, `V` to 3.6e-2 m/s, `alpha` to 2.8°, attitude to 3.0° — while at the configured 10 ms
  step the same run reaches 0.99 m and 33° of `alpha`. That gap is the plant's time step, which
  is the point of `fig1` having two curves in its lower row. A tumble amplifies whatever
  difference is already there geometrically, about one e-fold per second here, so the report
  prints where the velocity difference passes 1 % of the initial airspeed: t = 5.56 s refined,
  earlier at the configured step. With `--rotating-earth` it arrives at 2.91 s instead, and past
  it JSBSim rolls and yaws away while FALCON-S stays exactly planar, `p` reaching 295 deg/s
  against 0 by t = 5 s.
* **A red dash-dotted line in the diagnostic figures marks `theta` reaching 90°**, where `phi`
  and `psi` stop being separately defined. The two simulators take opposite branches of the same
  attitude and the two rows differ by 180° while nothing physical happens. The printed table
  stars them; the `attitude` row, taken from the quaternions, is immune.

## Statistics printed

Every comparison prints bias, RMS, worst case, and the worst case as a percentage of that
channel's own peak-to-peak over the same window:

* **bias against RMS** separates a constant offset — a mismatched gravity, a units slip, an
  interpolation chord that always falls on one side of its arc — from a growing divergence. They
  are different faults and need different fixes.
* **max % of span** answers whether a difference matters at the scale of the motion the aeroplane
  actually flew, which an absolute number cannot.
* **the e-folding time** of the velocity difference, fitted from its logarithm, and **the time it
  passes 1 % of the initial airspeed**. The rollouts tumble, so disagreement grows geometrically;
  those two numbers say how far a rollout can be trusted, which no single error figure does.
* **the observed order of accuracy** of the plant's own step, from a ladder of halved steps
  compared against the plant's solution at `dt/64`. No JSBSim in that number at all — it is a
  statement about the integrator by itself.

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
`p̂ = p·b/2V`, `q̂ = q·c̄/2V`, `r̂ = r·b/2V` use JSBSim's own `bi2vel`/`ci2vel`.

Coefficients come from the same `DerivativeAeroParameters` the plant loads, so they cannot drift
from the model they stand for. This got **simpler** with the aero refactor, not harder: the
polynomial needed six 2-D tables spanning the full alpha circle, ~250 kB per airframe, and carried
its own bilinear interpolation error into check 1. The linear set is 16 kB of exact arithmetic, so
check 1 should now agree to round-off and any visible difference is a transcription bug.

Left out on purpose:

* **Ground effect.** JSBSim is the out-of-ground-effect reference that FALCON-S's measured height
  sweep is judged against. Giving JSBSim a ground-effect model of its own would defeat the
  comparison, so height enters only through FALCON-S — that is what check 4 and check 5 measure.
  The XML carries the free-air set.

No longer left out:

* **Rate damping.** `CMl_p`, `CMm_q`, `CMn_r`, `CL_q`, `CD_q`, `CS_p` and `CS_r` are members of the
  measured set and every FALCON-S backend applies them, so the export carries them too. Under the
  polynomial they were omitted because the CPU plant applied none, which made this aircraft a
  reference for the CPU plant only. That caveat is gone — it is now a reference for all four.
* **Actuator and engine dynamics.** Controls are held at zero and the throttle shut throughout.
  Deflection limits are symmetric on every airframe, so a zero command is exactly zero degrees and
  a `-1` throttle command is exactly zero thrust; both stay there for the whole run.

## Making the two comparable

FALCON-S is a flat, non-rotating earth with one constant `g`. JSBSim integrates in an
earth-centred frame with a WGS-84 gravity model, so it carries Coriolis and centrifugal terms
FALCON-S has no equivalent of. **The fix is to take them away from JSBSim rather than to live with
them**: `nonrotating_planet.xml` redefines the planet with `rotation_rate` and `J2` at zero, loaded
through `FGFDMExec::LoadPlanet` before the aircraft. Earth's real GM and radii stay, so airspeed,
altitude and density keep their usual meaning. What that buys:

| | v after 2 s | p after 2 s |
| --- | --- | --- |
| the real rotating earth | −1.59e-03 m/s | 5.05e-03 deg/s |
| `rotation_rate = 0` | 7.80e-14 m/s | 1.22e-13 deg/s |

With no rotation the lateral channels agree to machine precision instead of to a floor, and a full
five-second tumble stays quantitatively comparable rather than departing at three seconds. With
`J2 = 0` the WGS-84 and standard gravity models coincide, so `simulation/gravity-model` stops
mattering as well. `--rotating-earth` puts the real earth back, which is the way to measure what
FALCON-S's flat-earth assumption actually costs.

Two smaller things, both handled in `validate.py`:

* JSBSim's gravitation is read at the initial condition and forced on both plants — 9.7977 m/s²
  at 200 m against FALCON-S's configured 9.81, which would otherwise integrate to more than a
  centimetre in a couple of seconds. Runs sit at the equator so that local down is the gravity
  direction. What remains is that JSBSim's gravity falls off with altitude while FALCON-S's does
  not: 3e-4 m/s² per 100 m of climb, four decades below anything else measured here.
* JSBSim runs at 1 ms against the plant's 10 ms. At 1 ms its trajectory is converged to 5e-7 m/s
  over half a second, so its integration error is not part of what is being measured.

## What the checks found

> Measured against the **polynomial** aero model, before the OpenVSP refactor. Kept for the method
> findings; the figures need re-measuring. See the note at the top.

Numbers below are the Navion, 2 s from 200 m, `V = 50 m/s`, `alpha = 2°`, attitude `0, -10, 30`.

**1 and 2, the aerodynamics, agree exactly.** Table lookup against the polynomial is within
1.3e-4 on every coefficient — pure bilinear interpolation error at the 1° table step, which
`--angle-step` trades against file size. The sweeps run at 0.37° deliberately, since a sample
landing on a table node reads the polynomial back exactly and measures nothing. For `CD` and
`CMy` the bias is nine tenths of the RMS, because a chord across a convex arc falls on the same
side every time; that is the expected signature of interpolation and not a sign of anything
wrong. Body-frame force and moment at matched `(alpha, beta, V)` agree to 4e-4 N in 34 kN and
7e-5 N·m in 14 kN·m, about 1e-8 relative — ten decades under the load being compared. Dynamic
pressure, the reference lengths, the moment reference point and the axis conventions are all
confirmed by that.

**The wind-to-body rotation had the sign of the sideslip terms wrong. Found here, fixed by
"sim: fix the sign of the sideslip terms in the wind-to-body rotation".** At `beta = 10°` the force difference was 4.4e2 N in x and 1.1e3 N in y while the
moments stayed exact. All three plants built the rotation the same way — the CPU one as an
explicit matrix, the Warp and Torch ones as `quat_rpy(0, alpha, beta)` followed by an inverse
rotation, which expands to the same thing — so they agreed with each other and disagreed with
JSBSim together, and cross-plant parity never showed it:

```
was                                           now, and what JSBSim uses
[ ca*cb   ca*sb  -sa ]                        [ ca*cb  -ca*sb  -sa ]
[  -sb     cb     0  ]                        [   sb     cb     0  ]
[ sa*cb   sa*sb   ca ]                        [ sa*cb  -sa*sb   ca ]
```

The old matrix is the new one with `beta` negated — a perfectly valid rotation, of the wrong
angle, which is why nothing ever looked broken. Its first column, the direction the relative wind
blows along and therefore the axis drag acts on, was `[ca*cb, -sb, sa*cb]`, while `alpha` and
`beta` already fix that direction as `[ca*cb, +sb, sa*cb]`. So drag pushed *along* the sideslip
instead of against it: at `beta = 90°` the aeroplane's drag became thrust. The model also
contradicted itself, independently of JSBSim — `beta = asin(v/Va)` and the signs of `CY`, `CMx`
and `CMz` in the polynomials are all standard, so the rotation was the piece out of step, and the
coefficient tables were right as they stood.

Only the forces were affected, and only through `sin(beta)`: `2*Y*cos(alpha)*sin(beta)` in x,
`2*D*sin(beta)` in y, `2*Y*sin(alpha)*sin(beta)` in z. Moments are applied directly in body axes
and were always correct. At `beta = 0` the two matrices are identical, so nothing in symmetric
flight moves: the altitude task, every longitudinal result and ground effect are untouched.
Sideslip is not: attitude tasks that use rudder or aileron, the named manoeuvres, and every
turbulence run, since Dryden's lateral component puts `beta` off zero at every step. The LQR gains
were derived from a linearisation whose `Y_beta` was off by `2*D`, and the checkpoints were
trained against the old lateral aerodynamics.

After the fix the sideslip column of check 2 sits at 1e-5 N against 1e-8 relative, the same as the
zero-sideslip column, and a rolled rollout that develops `beta` up to 5° tracks JSBSim to 0.12° in
`beta` and 0.056 m/s in `v` over three seconds — the lateral model validated for the first time.

**The plant's 10 ms step, not JSBSim, dominates the rollout difference.** At the configured step
the two part by 5.3e-2 m/s at half a second; with the plant at 0.5 ms the same rollout agrees to
3.0e-3 m/s, and every state in the table improves by the same order — `h` from 7.7e-2 to 1.3e-2 m,
`alpha` from 0.21 to 0.046°, `q` from 0.32 to 0.18 deg/s. Against the plant's own solution at
`dt/64`, with JSBSim out of the picture entirely, the observed order of accuracy is

```
      dt [s]   velocity error [m/s]   order
     0.01000              4.874e-02
     0.00500              2.396e-02    1.02
     0.00250              1.159e-02    1.05
     0.00125              5.407e-03    1.10
```

first order, despite the RK45 label — so something is held frozen across the step. Worth knowing
on its own terms, and it is why check 3 flies the rollout twice; `--refine` controls the second
one. With the planet not rotating and the step refined, the plant's own time step is the only
thing left: nothing else in the comparison rises above 3e-3 m/s.

**Ground effect behaves as documented.** Check 4 sweeps height at fixed `alpha` and finds
`CL_falcons / CL_jsbsim` equal to `mu_l` and `CD_falcons / CD_jsbsim` equal to `mu_d · mu_l²`, to
the last digit printed, converging to 1 above `h/b ≈ 1` — so above a span the ground-effect model
is inert and the out-of-ground-effect comparison is genuinely out of ground effect. Check 5 then
flies the same rollout from `h/b = 0.2`, where the two separate immediately: for the Navion,
1.3 m of altitude after 2 s. That divergence is the model working, not a failure.

**The airships tumble hard enough to break the plant.** `Airship_V7` tracks JSBSim to 7e-3 m/s
at half a second and then, around 0.8 s, leaves any physical range — its `Cm` reaches 6.1 against
the Navion's 0.51, on a much smaller inertia, so with no controls and no rate damping the pitch
rate runs away until the plant's own angular-velocity guard trips. The tool stops the rollout and
says why. Compare the airships over shorter windows, or accept that the useful part of the run is
the first half second.

**These airframes have no zero-elevator pitch trim.** The Navion's `Cm` never crosses zero with
the elevator centred — 0.128 at `alpha = 0`, a minimum of 0.059 near 15° — so a run with no
actuator input always pitches up, and with no rate damping it goes round. Full-circle alpha tables
keep that comparable, but any rollout long enough to tumble amplifies small differences
exponentially, so the report gives the difference against time rather than one maximum over the
run. The early rows are the ones that mean something.
