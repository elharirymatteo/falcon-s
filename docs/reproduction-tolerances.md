# Reproduction tolerances

`falcons.benchmark.TOLERANCE` is the single statement of what "reproduces" means, per table:
learned and LQR rows are exact (seeded rollouts are bit-stable); MPPI is a sampling planner whose
cost reduction uses non-deterministic atomics, so its rows must land within the shipped spread;
wind cells are one Dryden draw each. `falcons.benchmark.diff.compare_csv` applies a rule to one
CSV pair, and `falcons.benchmark.reproduce` applies every rule and decides the exit code.

This file records WHY each of those constants is the number it is. Every justification below was
measured on this repository's own tables; none of it is a default carried over from anywhere.
Change a constant here and the paragraph that justifies it stops being true.

## `MAX_SIGMA_EXCURSIONS` (`falcons/benchmark/reproduce.py`)

A mismatch on a row the rule treats as deterministic -- a learned policy, the LQR, a trim sweep --
is DRIFT: the code changed, or the machine did, and the claim is broken. It always fails.

A mismatch on a row matched by `sigma_if` is an EXCURSION: MPPI is a sampling planner whose
rollout-cost reduction uses non-deterministic atomics, so its cells are draws and a band around
them is a probabilistic statement. Requiring zero excursions would assert something false about a
sampling planner. How many to allow is arithmetic on this repository's own tables, and only the
comparisons a sigma band actually decides enter it:

  - the gate visits 1708 (row, column) pairs across the five tables, of which 1638 put a number
    against a number; the rest are string or NaN cells, compared for equality;
  - only 122 of those sit on a `sigma_if` row at all -- the other 1516 are exact, `loose_abs` or
    `rel` comparisons, which cannot excurse;
  - of the 122, a `sigma_floor` decides 55 and the `sigma_max` cap decides 7, leaving 60 whose
    tolerance is the sigma term itself. Those 60 are the whole exposure;
  - a two-sided 3-sigma band on 60 comparisons expects 60 * 0.0027 = 0.16 excursions by chance.

`MAX_SIGMA_EXCURSIONS` is 8: about 50x that expectation, and the headroom is structural rather
than statistical. The 60 comparisons are not 60 independent Gaussian tests. They sit in ten MPPI
cells, and inside a cell phi_rmse_deg, phi_nrmse_pct and phi_theil_u are three renormalisations of
one bank-error residual, so a cell that draws a single bad rollout moves its whole column set at
once. Each band is also drawn around a sigma estimated from three (altitude) or five (maneuvers)
draws, of distributions that are wide and not Gaussian: sigma/mean over these sigma-gated columns
runs to 2.00, and a survivor-conditioned mean averages a different subset of draws each sweep. So
the budget is set in cells rather than in columns -- one MPPI cell may land entirely outside its
bands, two may not -- and the widest cell carries 8 sigma-gated columns.

It is a BACKSTOP for the next person's run, not the thing that makes the current one pass: the
boundary floors and the survivor-conditioned sigma scale already clear all three mismatches the
2026-09-06 run produced, which is why that run's replay reports zero excursions.

## `sigma_k` (`falcons/benchmark/__init__.py`, applied in `diff.compare_csv`)

A sigma row is a sample, and the row it is compared against is another sample of
the same statistic: two independent draws sit ~sqrt(2)*sigma apart on average, so a
1-sigma band would reject an honest re-run about half the time. `sigma_k` widens the
band to k standard deviations; the floor is an absolute minimum, never scaled.

## `sigma_partial_survival_k` (`falcons/benchmark/diff.py`, default 2.0)

A cell that loses draws reports a survivor-conditioned mean: the metric is averaged over
the draws that stayed inside the envelope, and WHICH draws those are changes between
sweeps. Measured on the Volantex s-turn hdot_rmse_ms, the spread between sweeps (0.241
over three sweeps) ran ~2x the within-sweep spread across draws (0.115), so the stored
sigma understates the scale of the quantity actually being compared. Widen it.

## `TOLERANCE["altitude.csv"]`

### `sigma_floor`

sigma_floor exists because the archive's own σ came from three SAME-seed sweeps and is
under-dispersed. The survival / acquired / rmse / energy floors are the atomic-add
nondeterminism measured in Task 7a's draw-0 probe against the archive (rmse 30.99 vs
30.69 m). The overshoot floor is a deliberate widening beyond that, for the reason at its
own entry below.

acquired: the floor is a lower bound only, and sigma dominates it on the
V7 MPPI row (3 * 2 * 0.01886 = 0.113 around 0.0267), so the band does
reach past the emitter's 0.05 print threshold (acq >= 0.05 switches
settling from --- to a number). What keeps a re-draw across that
threshold from reading as a mismatch is skip_cols_if_sigma, not the floor.

settled_frac: bounded 0-1 and quantised by the episode count, and the
Volantex cell sat at 0.0 in every shipped draw (the V7 one at
0.0446 +- 0.0318) -- a statistic pinned at its boundary in every draw has
sigma 0 by construction, which is not evidence that it cannot move. One
settled episode of a ~6-survivor sweep moves it by 0.056.

overshoot is a survivor-only mean over 21–27 of 150
episodes for MPPI on the Volantex; the archive's σ (5.09)
came from same-seed sweeps; 6 m is a third of the 18 m
commanded change against the 28–42 m overshoots the
finding rests on (Task 7a archive-mode gate: 33.92 vs 28.59)

### `skip_cols_if_sigma`

settling is undefined for a method that acquires <5 % of episodes; the
table prints --- there

### `sigma_max`

R26: half the domain of a statistic bounded on 0-1. A band that wide
admits any value the column can take, so it stops being a test.

## `TOLERANCE["maneuvers.csv"]`

### `sigma_cols`

MPPI is a sampling planner: every metric of a cell is a draw, not just the bank RMSE. The
archive's table was cache-replayed, so it never had to state this. Each averaged column now
carries its own <col>_std (benchmark.maneuvers.aggregate) and a sigma row is compared within
it; only phi_rmse_deg needs an explicit mapping, its spread predating the convention.

### `sigma_floor`

floors only where a spread can legitimately be ~0 while the value still
jitters: a normalised, bounded or quantised statistic. Survival is
quantised at 1/n_draws, and a cell that kept every draw in one sweep can
lose one in the next (Volantex figure-8 and s-turn did exactly that
between two sweeps), so its floor is one draw of five.

time_in_band_pct: bounded above at 100 and saturating there -- a statistic
pinned at its boundary in every draw has sigma 0 by construction, which is
not evidence that it cannot move (the Volantex circle cell read 100.0 in
all five shipped draws and 97.7 in the next sweep).

### `sigma_max`

R26: half the domain for the two bounded columns (survival on 0-1,
time-in-band on 0-100); for the bank RMSE, a quarter turn -- past 45 deg
of mean bank error the executor has plainly failed, however wide its draws.

R26, applied last so it caps the floor as well as the band: a tolerance wider
than half a bounded statistic's domain tests nothing; beyond a quarter turn of
bank error the executor has plainly failed, whatever its draw spread.

## `TOLERANCE["robustness.csv"]`

### `loose_abs_by`

obs_noise: the archive drew sensor noise from the unseeded global CUDA
generator; repeat-to-repeat spread 0.018 deg (Task 7d); the shipped
protocol seeds it

A family with a bound of its own (`loose_abs_by`) is judged on that, not on the
shared one: one disturbance's spread says nothing about another's.

## `TOLERANCE["throughput.csv"]`

`rel` 0.5 -- wall-clock: hardware-dependent, order of magnitude. The rows are wall-clock and
therefore hardware-bound, and the shipped numbers are only meaningful on an otherwise idle GPU.
`reproduce.compare` skips this table entirely: its rule exists for a human reading the numbers
rather than for a machine deciding whether the benchmark reproduced.
