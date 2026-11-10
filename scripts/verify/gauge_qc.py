#!/usr/bin/env python3
"""The 24 h gauge totals that failed quality control, kept out of the scores of the runs the database already holds.

mverify flags a gauge total as it is read (mverify.db.qc: outside the physical range, above the totals 12 h either
side of it, unlike the sum of the same hours' amounts, large through hours the station reports as dry, and every
total of a station-year in which a large one failed), and a flagged total is never matched. A run in the
verification database (scripts/verify/verifydb.sh) was matched against the totals that passed on its night; one
that fails afterwards (a whole station-year goes with one false total) is still among its pairs, and the pairs are
kept as they were. So ``mverify db ghcnh --failed-totals`` lists every failed total of the station-years it reads
(failed_totals_<year>.parquet: station_id, valid_time, qc); the lists live in the release verify-gauge-qc, each
night's joined to the one held (``join``: a total that failed once stays out, also on a night its station could not
be read); and the gauge sums of every held run are made again each night from its gauge pairs without the listed
totals (``Failed.left``; mverify_scores.regauge, ens_verify.regauge). A total of a year that has no list was never
checked this way and is left out as well.

    python scripts/verify/gauge_qc.py NEW HELD      # the lists of NEW joined into those of HELD
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

KEY = ["station_id", "valid_time"]


def lists(root: Path) -> dict[int, Path]:
    """The lists in ``root`` by year."""
    return {int(f.stem.rsplit("_", 1)[1]): f for f in sorted(root.glob("failed_totals_*.parquet"))}


def join(new: Path, held: Path) -> None:
    """Each year's list in ``new`` joined into the one in ``held``, which keeps every total either has."""
    held.mkdir(parents=True, exist_ok=True)
    for year, f in lists(new).items():
        if not (old := held / f.name).exists():
            shutil.copyfile(f, old)
            continue
        both = pd.concat([pd.read_parquet(old), pd.read_parquet(f)], ignore_index=True)
        both = both.drop_duplicates(KEY, keep="last")
        both.to_parquet(old, compression="zstd", index=False)
        print(f"{year}: {len(both)} failed gauge totals", flush=True)


def _keys(frame: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_arrays(
        [frame["station_id"].astype(str), frame["valid_time"].astype("datetime64[ns]")], names=KEY
    )


class Failed:
    """The failed totals of the lists in ``root`` and the years they cover."""

    def __init__(self, root: Path):
        found = lists(root)
        if not found:  # never score the gauges uncut
            raise FileNotFoundError(f"no failed_totals_<year>.parquet in {root}")
        self.years = sorted(found)
        self.totals = _keys(pd.concat([pd.read_parquet(f, columns=KEY) for f in found.values()])).unique()
        self.seen = self.out = 0  # the pairs ``left`` has read, and those of them it left out

    def cut(self, pairs: pd.DataFrame) -> pd.DataFrame:
        """``pairs`` (or ensemble cases) without the listed totals and without those of a year that has no list."""
        checked = pairs["valid_time"].dt.year.isin(self.years).to_numpy()
        return pairs[checked & (self.totals.get_indexer(_keys(pairs)) == -1)]

    def left(self, gauges: Path) -> pd.DataFrame | None:
        """What a run's gauge sums are to be made again from, of its gauge pairs in the file ``gauges``: the pairs
        ``cut`` leaves; none where the database does not hold the file; None where it leaves them all (the sums the
        run was archived with stand)."""
        if not gauges.exists():
            return pd.DataFrame()
        pairs = pd.read_parquet(gauges)
        clean = self.cut(pairs)
        self.seen, self.out = self.seen + len(pairs), self.out + len(pairs) - len(clean)
        return None if len(clean) == len(pairs) else clean

    def __str__(self) -> str:
        return f"{self.out} of {self.seen} held pairs left out, their totals failed quality control"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("new", type=Path, help="tonight's lists (mverify db ghcnh --failed-totals)")
    ap.add_argument("held", type=Path, help="the held lists, which take them in")
    a = ap.parse_args()
    join(a.new, a.held)


if __name__ == "__main__":
    main()
