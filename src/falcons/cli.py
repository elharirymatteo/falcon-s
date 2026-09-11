"""`falcons` command line: argparse glue only. Every subcommand calls one importable function."""
import argparse
import sys
from pathlib import Path

from falcons.paths import CKPT_DIR, RESULTS_DIR

ALGOS, TASKS = ("ppo", "sac", "td3"), ("altitude", "attitude")
PLANES = ("Airship_V7", "Airship_A0S", "Volantex_Ranger", "Navion")
PROTOCOLS = ("altitude", "maneuvers", "ground-effect", "robustness", "throughput")
ALTITUDE_DRAWS, MANEUVER_DRAWS = 3, 5   # MPPI sampling draws per cell, per protocol


def _dirs(p):
    p.add_argument("--checkpoints", type=Path, default=CKPT_DIR)
    p.add_argument("--results", type=Path, default=RESULTS_DIR)


def cmd_check(a):
    import numpy as np
    import torch
    import warp as wp
    from falcons.controllers.policies import load_policy, ALGOS as AL, TASKS as TK
    from falcons.sim.cpu.aircraft import Aircraft as CpuAircraft
    cuda = torch.cuda.is_available()
    print(f"torch {torch.__version__} cuda={cuda} | warp {wp.__version__}")
    # A GPU is needed to FLY anything, but not to check that the shipped checkpoints deserialize
    # into the actor classes -- so this command stays useful (and honest about it) on a CPU box.
    device = "cuda" if cuda else "cpu"
    n = 0
    for algo in AL:
        for task in TK:
            for plane in ("Airship_V7", "Volantex_Ranger"):
                for s in (0, 1, 2):
                    load_policy(algo, task, plane, s, device=device, ckpt_dir=a.checkpoints); n += 1
    # Fly whichever airframe has its OpenVSP derivative data extracted. The aerodynamic model IS
    # that data, so an airframe without it cannot be built -- and a smoke check that hardcodes one
    # would fail for a reason that has nothing to do with the thing being checked.
    from falcons.aircraft.config import PLANES, AircraftConfig
    flyable = [p for p in PLANES if AircraftConfig(p).has_derivatives]
    if not flyable:
        print(f"{n} checkpoints load on {device}; NO airframe has derivative data, "
              f"so the plant was not stepped.")
        return
    plant = CpuAircraft(flyable[0])
    plant.reset()
    plant.step(np.zeros(plant.get_aero_action_size()), np.zeros(plant.get_motor_action_size()))
    print(f"{n} checkpoints load on {device}; {flyable[0]} cpu plant steps. OK")


def cmd_train(a):
    from falcons.train import tasks, sac, td3
    curves = a.results / "curves"
    if a.algo == "ppo":
        (tasks.train_altitude if a.task == "altitude" else tasks.train_attitude)(
            a.plane, a.steps, a.seed, a.checkpoints, curves)
    else:
        (sac if a.algo == "sac" else td3).train(a.plane, a.task, a.steps, a.seed,
                                                a.checkpoints, curves)


def cmd_eval(a):
    from falcons.benchmark import eval_one
    eval_one(a.algo, a.task, a.plane, a.seed, a.maneuver, a.n, a.plot, a.checkpoints, a.results)


def cmd_benchmark(a):
    from falcons.benchmark import altitude, maneuvers, ground_effect, robustness, throughput
    planes = a.planes or ["Airship_V7", "Volantex_Ranger"]
    # the two protocols draw a different number of MPPI samples per cell, so the flag defaults per
    # protocol rather than to one of them; given explicitly it is always the number that is used
    draws = a.mppi_draws if a.mppi_draws is not None else (
        ALTITUDE_DRAWS if a.protocol == "altitude" else MANEUVER_DRAWS)
    if a.protocol == "altitude":
        altitude.run(planes, a.checkpoints, a.results,
                     mppi_seeds=tuple(range(a.mppi_seed, a.mppi_seed + draws)))
    elif a.protocol == "maneuvers":
        maneuvers.run(planes, a.checkpoints, a.results, a.refresh, draws, a.mppi_seed)
    elif a.protocol == "ground-effect":
        ge = a.planes or list(ground_effect.GE_PLANES)
        ground_effect.run_trim(ge, a.results)
        ground_effect.run_energy(ge, a.checkpoints, a.results)
    elif a.protocol == "robustness":
        robustness.run(planes, a.checkpoints, a.results)
    else:
        throughput.run(planes[0], a.checkpoints, a.results)


def cmd_tables(a):
    from falcons.benchmark.tables import emit
    print(emit(a.results))


def cmd_figures(a):
    from falcons.benchmark.figures import render
    print(*render(a.names or None, a.checkpoints, a.results), sep="\n")


def cmd_reproduce(a):
    from falcons.benchmark.reproduce import run
    return run(a.checkpoints, a.results, a.out)


def main(argv=None):
    p = argparse.ArgumentParser(prog="falcons")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("check", help="verify the install: versions, GPU, checkpoints, one cpu step")
    _dirs(s)
    s = sub.add_parser("train", help="train one (algo, task, plane, seed) checkpoint"); _dirs(s)
    s.add_argument("--algo", choices=ALGOS, required=True); s.add_argument("--task", choices=TASKS, required=True)
    s.add_argument("--plane", choices=PLANES, required=True); s.add_argument("--seed", type=int, default=0)
    s.add_argument("--steps", type=int, help="total env steps (default: the task budget)")
    s = sub.add_parser("eval", help="one benchmark cell, printed"); _dirs(s)
    s.add_argument("--algo", choices=ALGOS + ("lqr", "mppi"), required=True); s.add_argument("--task", choices=TASKS, required=True)
    s.add_argument("--plane", choices=PLANES, required=True)
    s.add_argument("--seed", type=int, default=0,
                   help="checkpoint seed (altitude task; the attitude task aggregates "
                        "every seed checkpoint)")
    s.add_argument("--maneuver", default="circle", choices=("circle", "figure8", "helix", "sturn"))
    s.add_argument("--n", type=int, default=1, help="repetitions (MPPI draws)"); s.add_argument("--plot", action="store_true")
    s = sub.add_parser("benchmark", help="run one protocol across the benchmark airframes"); _dirs(s)
    s.add_argument("protocol", choices=PROTOCOLS); s.add_argument("--planes", nargs="*", choices=PLANES)
    s.add_argument("--refresh", action="store_true", help="re-fly cached maneuver rollouts")
    s.add_argument("--mppi-draws", type=int, default=None,
                   help=f"MPPI sampling draws per cell (default: {ALTITUDE_DRAWS} for altitude, "
                        f"{MANEUVER_DRAWS} for maneuvers)")
    s.add_argument("--mppi-seed", type=int, default=0, help="first MPPI sampling seed")
    s = sub.add_parser("tables", help="results/*.csv -> results/tables.tex"); _dirs(s)
    s = sub.add_parser("figures", help="render the benchmark figures (all, or the ones named)"); _dirs(s)
    s.add_argument("names", nargs="*")
    s = sub.add_parser("reproduce", help="regenerate every table and figure, diff against results/")
    _dirs(s); s.add_argument("--out", type=Path, default=RESULTS_DIR.parent / "reproduce")
    a = p.parse_args(argv)
    return {"check": cmd_check, "train": cmd_train, "eval": cmd_eval, "benchmark": cmd_benchmark,
            "tables": cmd_tables, "figures": cmd_figures, "reproduce": cmd_reproduce}[a.cmd](a) or 0


if __name__ == "__main__":
    sys.exit(main())
