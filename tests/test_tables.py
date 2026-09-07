"""The tex tables are generated, never hand-edited. The gate is that the emitter reproduces the
archive's own table file from the archive's own CSVs — byte for byte everywhere except the one
caption clause that is now derived from the data instead of written down (see `_departures` in
`falcons.benchmark.tables`), which is checked against the CSVs themselves."""
import csv
import re
import shutil
from pathlib import Path

from falcons.benchmark.tables import emit

GOLD = Path(__file__).parent / "golden"
CSVS = ("altitude", "ge_trim", "ge_energy", "maneuvers", "robustness")
DEPARTURES = re.compile(r"runs that stayed inside the flight envelope.*?\}", re.S)


def body(text):
    """Everything after the generator banner. Only the banner differs: the archive names the
    script that wrote it (three comment lines), this emitter names itself (two)."""
    lines = text.splitlines()
    i = 0
    while i < len(lines) and lines[i].startswith("%"):
        i += 1
    return lines[i:]


def _emit_from_golden(tmp_path):
    for n in CSVS:
        shutil.copy(GOLD / f"{n}.csv", tmp_path / f"{n}.csv")
    return emit(tmp_path).read_text()


def test_every_number_matches_the_archive_table(tmp_path):
    """The table bodies — every figure the paper prints — reproduce exactly."""
    got = DEPARTURES.sub("X", "\n".join(body(_emit_from_golden(tmp_path))))
    want = DEPARTURES.sub("X", "\n".join(body((GOLD / "tables.tex").read_text())))
    assert got == want


def test_the_caption_names_exactly_the_cells_that_lost_runs(tmp_path):
    """The departures clause is read off the table, so it cannot drift from it: every maneuver cell
    with survival below 1 is named, and no other cell is."""
    text = _emit_from_golden(tmp_path)
    clause = DEPARTURES.search(text).group(0)
    rows = list(csv.DictReader(open(GOLD / "maneuvers.csv")))
    lost = [r for r in rows if 0.0 < float(r["survival"] or 0) < 1.0]
    assert lost, "the archive table has cells that lost runs; the fixture must still contain them"
    for r in lost:
        plane = r["aircraft"].replace("_", "~")
        assert plane in clause, (plane, clause)
    kept = {r["aircraft"] for r in rows} - {r["aircraft"] for r in lost}
    for plane in kept:
        assert plane.replace("_", "~") not in clause, (plane, clause)
