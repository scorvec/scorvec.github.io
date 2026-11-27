#!/usr/bin/env python3
"""The major countries' forecast errors run by run, for the country series of verification.html: per country, model,
variable and lead, the stations, pairs and the sums of the error and its square of every run in the verification
database, from its station sums (stn_<init>.parquet, the single runs; ens-stn_<init>.parquet, the ensemble means).

The countries are the G20's (COUNTRIES); a station's is the first two letters of its GHCNh identifier, the FIPS 10-4
code GHCN files it under. The variables are the station maps' surface layers: 2 m temperature and dew point, sea-level pressure and 10 m wind, which for the single runs (wind10) is the
speed's error and the vector's squared error and for the ensemble means (ws10) the members' mean speed's, both of
them. A run's station sums are on its common sample, so in each run the models of a kind meet the same stations.

The table is kept between nights (``--table``, the workflow's cache): a night reads only the runs it lacks.
verify_figs.py draws it (country_figs).

    python country_series.py --list --table countries.parquet --have have.txt     # the files a night needs
    python country_series.py --table countries.parquet --derived db more          # the runs it lacks, from either
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

KINDS = {"stn": "single", "ens-stn": "means"}  # the station sums' files: the single runs', the ensemble means'
VARS = {"t2m": "t2m", "td2m": "td2m", "msl": "msl", "wind10": "wind10", "ws10": "wind10"}  # as the table names them
KEY = ["kind", "model", "variable", "lead_h", "country", "init"]
SUMS = re.compile(r"(stn|ens-stn)_\d{10}\.parquet")  # a run's station sums, as the database names them
COUNTRIES = {  # the G20's, by their FIPS 10-4 codes
    "AR": "Argentina", "AS": "Australia", "BR": "Brazil", "CA": "Canada", "CH": "China", "FR": "France",
    "GM": "Germany", "ID": "Indonesia", "IN": "India", "IT": "Italy", "JA": "Japan", "KS": "South Korea",
    "MX": "Mexico", "RS": "Russia", "SA": "Saudi Arabia", "SF": "South Africa", "TU": "Turkey",
    "UK": "United Kingdom", "US": "United States",
}  # fmt: skip


def run_rows(path: Path) -> pd.DataFrame:
    """One run's sums per model, variable, lead and country of COUNTRIES (KEY), with the stations they come from."""
    prefix, init = path.stem.rsplit("_", 1)
    cols = ["model", "network", "variable", "station_id", "lead_h", "n", "se", "sse"]
    s = pd.read_parquet(path, columns=cols, filters=[("network", "==", "GHCNH"), ("variable", "in", list(VARS))])
    s = s.astype({"model": str, "variable": str, "station_id": str})
    s = s.assign(
        kind=KINDS[prefix],
        variable=s["variable"].map(VARS),
        country=s["station_id"].str.split(":", n=1).str[-1].str[:2],
        init=pd.to_datetime(init, format="%Y%m%d%H"),
    )
    g = s[s["country"].isin(COUNTRIES)].groupby(KEY)
    return g[["n", "se", "sse"]].sum().join(g.size().rename("stations")).reset_index()


def held(table: pd.DataFrame | None) -> set[str]:
    """The station sums' files the table has read."""
    if table is None:
        return set()
    prefix = {kind: p for p, kind in KINDS.items()}
    runs = table[["kind", "init"]].drop_duplicates()
    return {f"{prefix[k]}_{i:%Y%m%d%H}.parquet" for k, i in zip(runs["kind"].astype(str), runs["init"], strict=True)}


def wanted(have: set[str], table: pd.DataFrame | None) -> list[str]:
    """The database's station sums that the table lacks."""
    done = held(table)
    return sorted(n for n in have if SUMS.fullmatch(n) and n not in done)


def update(table: pd.DataFrame | None, files: list[Path]) -> pd.DataFrame:
    parts = ([table] if table is not None else []) + [run_rows(f) for f in files]
    t = pd.concat(parts, ignore_index=True).drop_duplicates(KEY, keep="last")
    return t.astype(dict.fromkeys(["kind", "model", "variable", "country"], "category")).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", type=Path, required=True, help="the table kept between nights")
    ap.add_argument("--list", action="store_true", help="print the station sums' files the table lacks, one line")
    ap.add_argument("--have", type=Path, help="names of the files the database holds (--list)")
    ap.add_argument("--derived", type=Path, nargs="+", default=[], help="directories holding the files, downloaded")
    a = ap.parse_args()
    table = pd.read_parquet(a.table) if a.table.exists() else None
    if a.list:
        print(*wanted(set(a.have.read_text().split()), table))
        return
    done, files = held(table), {}
    for d in a.derived:
        for f in sorted(d.glob("*stn_*.parquet")):
            if SUMS.fullmatch(f.name) and f.name not in done:
                files.setdefault(f.name, f)  # the same run in two directories is read once
    table = update(table, list(files.values()))
    table.to_parquet(a.table, compression="zstd")
    runs = table.groupby("kind", observed=True)["init"].nunique().to_dict()
    print(f"country series: {len(files)} runs read, {table['country'].nunique()} countries, runs {runs}", flush=True)


if __name__ == "__main__":
    main()
