"""Emit the benchmark result tables as LaTeX bodies, straight from the generated CSVs, so a
document that prints them cannot drift from the numbers in the repository.

Reads   <results_dir>/altitude.csv    (acquisition and hold)
        <results_dir>/ge_energy.csv   (ground effect, closed loop)
        <results_dir>/ge_trim.csv     (ground effect, open-loop trim)
        <results_dir>/maneuvers.csv   (attitude-stream tracking)
        <results_dir>/robustness.csv  (disturbance robustness)

Run: emit(results_dir) -> <results_dir>/tables.tex
"""
import csv
import os
from pathlib import Path

from falcons.paths import RESULTS_DIR

# One method order across every table: the learned block, then the classical block.
METHOD_ORDER = ["PPO", "SAC", "TD3", "LQR", "MPPI"]
PRETTY = {"Airship_V7": "Airship V7", "Volantex_Ranger": "Volantex Ranger",
          "Airship_A0S": "Airship A0S", "Navion": "Navion", "Cirrus_SR22": "Cirrus SR22"}


def rows(path):
    if not os.path.exists(path):
        print(f"  missing {path} — skipping its table")
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def num(x, fmt="{:.3f}", dash="---"):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return dash
    if v != v:                       # NaN
        return dash
    if v == float("inf"):
        return "$\\infty$"
    return fmt.format(v)


def rmse_cell(x, sd=None, n_seeds=1):
    """Settled RMSE spans seven decades across these methods, from the LQR's numerical floor to a
    method that never arrives, so a fixed number of decimals is not readable. Below 1 mm the value
    is reported in scientific notation, as the quantity there is a solver residual, not a metre.

    The spread is appended only where more than one sweep was flown. A single-sweep method has no
    measured variance, and printing a zero there would claim one it does not have."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "---"
    if v != v:
        return "---"
    if v < 1e-3:
        m, e = f"{v:.1e}".split("e")
        return f"${m}{{\\times}}10^{{{int(e)}}}$"
    fmt = "{:.3f}" if v < 10 else "{:.1f}"
    cell = fmt.format(v)
    try:
        s = float(sd)
    except (TypeError, ValueError):
        return cell
    if int(float(n_seeds)) > 1 and s == s:
        cell += "\\,$\\pm$\\," + fmt.format(s)
    return cell


def pm(v, sd, n_seeds, fmt="{:.2f}", dash="---"):
    """Mean, with its spread appended only where more than one sweep was flown. A single-sweep
    method has no measured variance, and printing a zero there would claim one it does not have."""
    m = num(v, fmt, dash)
    if m == dash:
        return m
    try:
        s, k = float(sd), int(float(n_seeds))
    except (TypeError, ValueError):
        return m
    return m + (f"\\,$\\pm$\\,{fmt.format(s)}" if k > 1 and s == s else "")


def pct(x):
    """Signed percentage in math mode, so the sign is a minus rather than a hyphen."""
    v = num(x, "{:+.1f}")
    return "---" if v == "---" else f"${v}$\\%"


def block(lines, name):
    """Prepend the rotated airframe cell to a block of already-formatted rows."""
    if not lines:
        return []
    head = f"\\multirow{{{len(lines)}}}{{*}}{{\\rotatebox{{90}}{{{name}}}}}"
    return [head + " " + lines[0]] + lines[1:]


def protocol_table(rs):
    if not rs:
        return ""
    body = []
    for ac in dict.fromkeys(r["aircraft"] for r in rs):
        by = {r["method"]: r for r in rs if r["aircraft"] == ac}
        lines = []
        for m in METHOD_ORDER:
            r = by.get(m)
            if r is None:
                continue
            surv, acq = float(r["survival"]), float(r["acquired"])
            if surv == 0.0:            # never completed: no error metric is meaningful
                lines.append(f"& {m} & {int(float(r['n']))} & 0.00 & 0.00 & --- & --- & --- \\\\")
                continue
            # A settling-time mean over a handful of episodes is not a statistic: when almost
            # nothing acquires, the cell is left empty rather than quoting the one run that did.
            ns = r.get("n_seeds", 1)
            ts = pm(r["settling"], r.get("settling_std"), ns, "{:.1f}") if acq >= 0.05 else "---"
            lines.append(f"& {m} & {int(float(r['n']))} & "
                         f"{pm(r['survival'], r.get('survival_std'), ns)} & "
                         f"{pm(r['acquired'], r.get('acquired_std'), ns)} & "
                         f"{rmse_cell(r['rmse'], r.get('rmse_std'), ns)} & "
                         f"{ts} & {pm(r['overshoot'], r.get('overshoot_std'), ns)} \\\\")
        body += block(lines, PRETTY.get(ac, ac)) + [r"\midrule"]
    body[-1] = r"\bottomrule"
    return "\n".join(
        [r"\begin{table*}[t]",
         r"\caption{Altitude acquisition and hold. Every episode starts "
         r"$\Delta\sim\pm\,\mathcal{U}(12,18)$~m off the commanded altitude, in both directions, "
         r"at commanded altitudes of 20--100~m, and all five controllers track the identical "
         r"rate-limited acquisition reference. Surv. is the fraction of episodes that reach the "
         r"horizon without a crash or a stall, and Acq. the fraction that additionally reach and "
         r"hold the target to within 1~m. Error metrics are means over the surviving episodes, "
         r"and $n$ is the number of episodes in each sweep. The learned rows are means over three "
         r"training seeds, and every metric is quoted as mean $\pm$ standard deviation "
         r"across them; a method flown once carries no spread. "
         r"The LQR is a deterministic law and is swept once; MPPI is a sampling planner and a "
         r"single sweep is one draw, so it is swept three times and averaged the same way as "
         r"the learned methods. The learned policies and MPPI are "
         r"scored on the GPU plant and the LQR on the CPU reference plant, which are validated to "
         r"single-step parity (Section~\ref{sec:validation}) but are not the same code path.}",
         r"\label{tab:altitude-protocol}", r"\centering",
         r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{llrrrrrr}", r"\toprule",
         r"& Method & $n$ & Surv. & Acq. & RMSE (m) & $t_s$ (s) & Over. (m) \\",
         r"\midrule"] + body + [r"\end{tabular}", r"\end{table*}", ""])


def ge_table(rs, trim):
    """Ground-effect table. The thrust columns come from the open-loop trim sweep, which needs no
    policy and therefore covers every band; the closed-loop column is the same quantity measured
    while a policy holds the band, and is left empty where no policy holds it. That is what fills
    the A0S rows below h/b = 2, which the policy-only table had to omit."""
    if not trim:
        return ""
    pol = {(r["aircraft"], r["h_over_b"]): r for r in rs if r["hold_valid"] == "1"}
    shown = [ac for ac in dict.fromkeys(r["aircraft"] for r in rs)
             if any(k[0] == ac for k in pol)]
    body = []
    for ac in shown:
        ts = [r for r in trim if r["aircraft"] == ac]
        if not ts:
            continue
        lines = []
        for r in ts:
            p = pol.get((ac, r["h_over_b"]))
            lines.append(
                f"& {num(r['h_over_b'], '{:.2f}')} & {num(r['altitude'], '{:.2f}')} & "
                f"{num(r['thrust_ge_on'], '{:.2f}')} & {num(r['thrust_ge_off'], '{:.2f}')} & "
                f"{pct(r['delta_thrust_pct'])} & "
                f"{pct(p['delta_thrust_pct']) if p else '---'} & {pct(r['theory_pct'])} \\\\")
        body += block(lines, PRETTY.get(ac, ac)) + [r"\midrule"]
    body[-1] = r"\bottomrule"
    return "\n".join(
        [r"\begin{table}[t]",
         r"\caption{Ground effect and the thrust required to hold altitude. $T_{\mathrm{GE}}$ and "
         r"$T_{\mathrm{no\,GE}}$ are the thrust at a level-flight trim solved with the "
         r"ground-effect correction on and off, and $\Delta T$ their relative change; the solve "
         r"uses no controller, so every band is reported. $\Delta T_\pi$ is the same change "
         r"measured while a policy holds the band, and is empty where no policy holds it. Theory "
         r"is the lifting-line prediction, evaluated at the height the simulator applies the "
         r"correction at. All quantities are paired: only the correction changes between arms.}",
         r"\label{tab:ge-energy}", r"\centering",
         r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{lrrrrrrr}", r"\toprule",
         r"& $h/b$ & $h$ (m) & $T_{\mathrm{GE}}$ (N) & $T_{\mathrm{no\,GE}}$ (N) & "
         r"$\Delta T$ & $\Delta T_\pi$ & Theory \\",
         r"\midrule"] + body + [r"\end{tabular}", r"\end{table}", ""])


def robustness_table(rs):
    """PPO under the four disturbance families, both airframes. One executor only: the families are
    applied to what the policy sees or commands, never to what is scored, so the comparison is
    between conditions rather than between methods, and adding methods would only widen it."""
    if not rs:
        return ""
    rs = [r for r in rs if r["method"] == "PPO"]
    if not rs:
        return ""
    idx = {(r["aircraft"], r["disturbance"], r["severity"]): r for r in rs}
    fams = [("obs_noise", "Sensor noise"), ("sensor_delay", "Sensor delay"),
            ("action_delay", "Action delay"), ("wind", "Turbulence")]

    def cells(dist, sev):
        out = []
        for ac in ("Airship_V7", "Volantex_Ranger"):
            r = idx.get((ac, dist, sev))
            if r is None:
                out += ["---", "---"]
                continue
            out += [num(r["survival"], "{:.2f}"), num(r["phi_rmse"], "{:.1f}")]
        return " & ".join(out)

    body = [f"Nominal & --- & {cells('nominal', '-')} \\\\", r"\midrule"]
    for dist, label in fams:
        for k, sev in enumerate(("mild", "medium", "severe")):
            head = f"\\multirow{{3}}{{*}}{{{label}}}" if k == 0 else ""
            body.append(f"{head} & {sev} & {cells(dist, sev)} \\\\")
        body.append(r"\midrule")
    body[-1] = r"\bottomrule"
    return "\n".join(
        [r"\begin{table}[t]",
         r"\caption{Robustness of the PPO attitude executor. Each cell is the mean over the four "
         r"maneuvers of Table~\ref{tab:maneuver}, under one disturbance family at one severity. "
         r"Surv. is the fraction of maneuvers completed inside the flight envelope and $\phi$ the "
         r"bank RMSE in degrees over those that were, both computed from the true simulator state "
         r"and never from the disturbed observation. Severities are defined in physical units "
         r"(sensor noise as per-channel floors, the delays as 20--100~ms of transport lag, "
         r"turbulence as a Dryden field at $\sigma_u/V_a = 0.05$--$0.15$).}",
         r"\label{tab:robustness}", r"\centering",
         r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{llrrrr}", r"\toprule",
         r"& & \multicolumn{2}{c}{Airship V7} & \multicolumn{2}{c}{Volantex Ranger} \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
         r"& Level & Surv. & $\phi$ ($^\circ$) & Surv. & $\phi$ ($^\circ$) \\",
         r"\midrule"] + body + [r"\end{tabular}", r"\end{table}", ""])


def _departures(rs):
    """The ' : one of three PPO seeds departed on X, ...' clause, read off the table rather than
    written down. A cell whose survival is below 1 lost runs, and the caption has to say so; the
    count of runs is the cell's own n_runs, so this stays true when a re-fly changes which cells
    depart."""
    lost = [r for r in rs if 0.0 < float(r["survival"] or 0) < 1.0]
    if not lost:
        return ". Every run stayed inside the envelope"
    def phrase(r):
        n = int(float(r["n_runs"]))
        k = int(round((1.0 - float(r["survival"])) * n))
        unit = "training seed" if r["method"] in ("PPO", "SAC", "TD3") else "sampling draw"
        return (f"{NUMBER.get(k, k)} of {NUMBER.get(n, n)} {unit}s departed on "
                f"the {PRETTY.get(r['aircraft'], r['aircraft']).replace(' ', '~')} "
                f"{MANEUVER_NAME.get(r['maneuver'], r['maneuver'])}")
    out = [phrase(r) for r in lost]
    return ": " + (", and ".join(out) if len(out) < 3 else ", ".join(out[:-1]) + ", and " + out[-1])


NUMBER = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
MANEUVER_NAME = {"circle": "circle", "figure8": "figure-of-eight", "helix": "helix",
                 "sturn": "s-turn"}


def maneuver_table(rs):
    """Page-width (table*): the cells carry a spread as well as a time-in-band, which no longer
    fits a single IEEE column."""
    if not rs:
        return ""
    shapes = ["circle", "figure8", "helix", "sturn"]
    body = []
    for ac in dict.fromkeys(r["aircraft"] for r in rs):
        lines = []
        for m in METHOD_ORDER:
            cells = []
            for sh in shapes:
                r = next((x for x in rs if x["aircraft"] == ac and x["method"] == m
                          and x["maneuver"] == sh), None)
                if r is None or float(r["survival"]) == 0.0:
                    cells.append("---")
                else:
                    # spread across the runs of the cell (training seeds, or sampling draws for
                    # MPPI); a method flown once has none and carries the bare mean
                    sd = num(r.get("phi_rmse_std"), "{:.1f}", dash="")
                    cells.append(f"{num(r['phi_rmse_deg'], '{:.1f}')}"
                                 + (f"\\,$\\pm$\\,{sd}" if sd else "")
                                 + f" ({num(r['time_in_band_pct'], '{:.0f}')})")
            if all(c == "---" for c in cells):
                continue
            lines.append(f"& {m} & " + " & ".join(cells) + r" \\")
        body += block(lines, PRETTY.get(ac, ac)) + [r"\midrule"]
    body[-1] = r"\bottomrule"
    return "\n".join(
        [r"\begin{table*}[t]",
         r"\caption{Attitude-stream tracking across the four named maneuvers: bank-angle RMSE "
         r"in degrees as mean $\pm$ standard deviation across runs, with the time-in-band "
         r"percentage in brackets. All five methods drive the identical "
         r"$(\phi^\star,\dot h^\star,V_a^\star)$ stream through the same plant. The learned "
         r"methods are flown once per training seed and averaged over three seeds, as in "
         r"Table~\ref{tab:altitude-protocol}. MPPI has no seeds, but a single rollout is one "
         r"draw, so it is flown five times with different sampling seeds. The LQR is a "
         r"deterministic law, is flown once and carries no spread. Each cell is a mean over the "
         r"runs that stayed inside the flight envelope" + _departures(rs) + ".}",
         r"\label{tab:maneuver}", r"\centering",
         r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{llrrrr}", r"\toprule",
         r"& Method & Circle & Figure-8 & Helix & S-turn \\",
         r"\midrule"] + body + [r"\end{tabular}", r"\end{table*}", ""])


def emit(results_dir=RESULTS_DIR):
    """results/*.csv -> results/tables.tex. Generated; never hand-edited.

    Preamble requirements: \\usepackage{booktabs,multirow,graphicx} (rotatebox comes from
    graphicx). The airframe name is set once per block as a rotated \\multirow cell, which keeps
    these tables inside a single IEEE column.

    Every column is read by NAME, so a CSV that carries more columns than a table prints (the
    altitude sweep's per-metric standard deviations, say) emits the same table it always did."""
    d = Path(results_dir)
    body = ["% Generated by falcons.benchmark.tables — do not hand-edit.",
            r"% Requires \usepackage{booktabs,multirow,graphicx} in the preamble.", ""]
    body.append(protocol_table(rows(d / "altitude.csv")))
    body.append(ge_table(rows(d / "ge_energy.csv"), rows(d / "ge_trim.csv")))
    body.append(maneuver_table(rows(d / "maneuvers.csv")))
    body.append(robustness_table(rows(d / "robustness.csv")))
    out = d / "tables.tex"; out.write_text("\n".join(body)); return out
