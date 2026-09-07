"""The two things that can be checked about the throughput module without a stopwatch.

The timing rows themselves are wall-clock and hardware-bound -- they are gated against
`tests/golden/throughput.json` at rel 0.5 by `TOLERANCE["throughput.csv"]` when the table is
regenerated, not here, because a test that fails when another job is on the GPU is not a test.
What IS checkable here is the interval arithmetic those rows are reported with, and the parity
claim, which is a property of the two plants and independent of how fast they run.
"""
import pytest


def test_stats_ci_matches_t_distribution():
    """`stats` reports mean, SAMPLE std (ddof=1) and a Student-t interval, not a normal one.
    The critical value is read off the t distribution at df = n-1, so the formula is checked
    here at two trial counts -- with the population std (ddof=0) or a 1.96 normal quantile the
    CI comes out ~5 % small at n=10, and much further out at n=5."""
    from scipy.stats import t as t_dist
    from falcons.benchmark.throughput import stats
    for vals in ([1.0] * 9 + [2.0], [1.0] * 4 + [2.0]):
        n = len(vals)
        m, sd, ci = stats(vals)
        assert m == pytest.approx(sum(vals) / n)
        assert ci == pytest.approx(t_dist.ppf(0.975, n - 1) * sd / n ** 0.5, rel=1e-12)
    sd10, ci10 = stats([1.0] * 9 + [2.0])[1:]        # df=9 anchor: the archive's baked constant
    assert ci10 == pytest.approx(2.262 * sd10 / 10 ** 0.5, rel=1e-3)


@pytest.mark.cuda
def test_parity_warp_vs_torch_within_half_metre(tmp_path):
    """The same checkpoint flown on the warp kernels and on the pure-torch twin has to settle to
    the same altitude on every target: the two are independent integrations of one set of
    equations, so a gap here is a plant difference, not noise. Half a metre against settled
    errors of order a metre -- shortened rollout (800 steps, 16 envs/target) to stay ~1 min."""
    from falcons.benchmark.throughput import parity
    d = parity("Volantex_Ranger", n_per_target=16, horizon=800)
    for t in d["targets"]:
        assert abs(d["warp"][t] - d["torch"][t]) < 0.5
