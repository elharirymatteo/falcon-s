"""The CLI and packaging surface.

`falcons check` and the path constants are the smoke test that the installed package imports and
resolves its own directories. Beyond that the CLI is argparse glue, so what has to be tested is
that each subcommand reaches the right runner with the right arguments -- in particular the two
places where the CLI does arithmetic rather than pass-through: the altitude protocol's draw count
(3, not the maneuver default of 5) and its expansion of --mppi-seed/--mppi-draws into the explicit
seed tuple `altitude.run` takes. No GPU, no rollout: the runners are recorders."""
import subprocess
import sys

import falcons.benchmark.altitude as altitude
import falcons.benchmark.maneuvers as maneuvers
from falcons import paths
from falcons.cli import main


def test_paths_resolve():
    assert paths.PKG_DIR.name == "falcons"
    assert (paths.REPO_DIR / "pyproject.toml").exists()


def test_cli_check_runs():
    assert subprocess.run([sys.executable, "-m", "falcons.cli", "check"]).returncode == 0


def _recorder(monkeypatch, module):
    calls = []
    monkeypatch.setattr(module, "run", lambda *a, **kw: calls.append((a, kw)))
    return calls


def test_benchmark_altitude_expands_mppi_seed_into_a_seed_tuple(monkeypatch):
    calls = _recorder(monkeypatch, altitude)
    main(["benchmark", "altitude", "--mppi-seed", "1"])
    (planes, ckpt, results), kw = calls[0]
    assert planes == ["Airship_V7", "Volantex_Ranger"]
    assert kw == {"mppi_seeds": (1, 2, 3)}      # 3 draws: the altitude count, not the maneuver 5


def test_benchmark_altitude_explicit_draws_override_the_altitude_default(monkeypatch):
    calls = _recorder(monkeypatch, altitude)
    main(["benchmark", "altitude", "--mppi-draws", "2", "--mppi-seed", "4"])
    assert calls[0][1] == {"mppi_seeds": (4, 5)}


def test_benchmark_altitude_honours_an_explicit_draw_count_equal_to_the_maneuver_default(monkeypatch):
    """--mppi-draws 5 on the altitude protocol means five sweeps, not the protocol default of
    three: the flag defaults per protocol rather than being sniffed out of the maneuver value."""
    calls = _recorder(monkeypatch, altitude)
    main(["benchmark", "altitude", "--mppi-draws", "5"])
    assert calls[0][1] == {"mppi_seeds": (0, 1, 2, 3, 4)}


def test_benchmark_maneuvers_passes_draws_and_seed_through(monkeypatch):
    calls = _recorder(monkeypatch, maneuvers)
    main(["benchmark", "maneuvers", "--mppi-draws", "2", "--mppi-seed", "4"])
    args, kw = calls[0]
    assert args[0] == ["Airship_V7", "Volantex_Ranger"] and args[3:] == (False, 2, 4) and kw == {}
