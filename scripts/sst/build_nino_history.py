#!/usr/bin/env python3
"""Monthly Niño-region history + El Niño event table → JSON for the interactive explorer.

Source: NOAA CPC ERSSTv5 monthly Niño indices (1950–present, 1991–2020 base) —
  https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii
  columns: YR MON NINO1+2 ANOM NINO3 ANOM NINO4 ANOM NINO3.4 ANOM
ERSSTv5 (not OISST) so the record reaches back past 1970; the 1991–2020 base matches
the rest of the El Niño monitor. ONI = Niño-3.4 anomaly, 3-month running mean.

Writes assets/sst/data/nino_history.json: the full monthly series for all four regions
(absolute SST + anomaly) and ONI, plus an El Niño event table (onset year, DJF-peak
month, peak ONI, strength) for every event since 1970. The interactive chart aligns
events by month-offset from their peak (or by ENSO calendar) entirely client-side, so
this file is the single source it needs.

    python build_nino_history.py            # → assets/sst/data/nino_history.json
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if p.name == "scripts") / "lib"))
from webget import get_text  # noqa: E402

HERE = Path(__file__).resolve().parent
SITE_ROOT = Path(os.environ["SST_SITE_ROOT"]).resolve() if os.environ.get("SST_SITE_ROOT") else HERE
OUT = SITE_ROOT / "assets" / "sst" / "data" / "nino_history.json"

SRC = "https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii"
START_YEAR = 1970                       # history floor for the explorer

# El Niño onset years since 1970 (year0 of the DJF event), per the CPC ONI record.
# Strength + exact peak month are computed from the data, not hard-coded.
ELNINO_ONSETS = [1972, 1976, 1977, 1979, 1982, 1986, 1987, 1991, 1994, 1997,
                 2002, 2004, 2006, 2009, 2014, 2015, 2018, 2023]
# Shown checked by default (the strong / most-compared events + current).
DEFAULT_ON = {1982, 1997, 2015, 2023}


def fetch(url: str) -> str:
    return get_text(url, ua="scorvec-enso/1.0")


def parse(txt: str) -> pd.DataFrame:
    rows = []
    for ln in txt.splitlines():
        p = ln.split()
        if len(p) == 10 and p[0].isdigit():
            yr, mon = int(p[0]), int(p[1])
            vals = [float(x) for x in p[2:]]
            rows.append([yr, mon] + vals)
    cols = ["yr", "mon", "n12", "n12a", "n3", "n3a", "n4", "n4a", "n34", "n34a"]
    df = pd.DataFrame(rows, columns=cols)
    df["date"] = pd.to_datetime(dict(year=df.yr, month=df.mon, day=1))
    return df.set_index("date").sort_index()


def strength(peak_oni: float) -> str:
    a = abs(peak_oni)
    if a >= 2.0:
        return "very strong"
    if a >= 1.5:
        return "strong"
    if a >= 1.0:
        return "moderate"
    return "weak"


def build_events(oni: pd.Series) -> list[dict]:
    """For each onset year, find the DJF-season peak (max ONI in Jul[y0]–Mar[y0+1])."""
    events = []
    for y0 in ELNINO_ONSETS:
        win = oni.loc[f"{y0}-07-01":f"{y0+1}-03-31"]
        if win.dropna().empty:
            continue
        pk_date = win.idxmax()
        pk = float(win.max())
        events.append({
            "id": str(y0),
            "label": f"{y0}–{str(y0 + 1)[2:]}",
            "onset": y0,
            "peak": f"{pk_date:%Y-%m}",
            "peakOni": round(pk, 2),
            "strength": strength(pk),
            "default": y0 in DEFAULT_ON,
        })
    return events


def main() -> int:
    print(f"fetching {SRC}")
    df = parse(fetch(SRC))
    df = df.loc[df.index.year >= START_YEAR]
    if df.empty:
        print("no rows parsed", file=sys.stderr)
        return 1

    oni = df["n34a"].rolling(3, center=True, min_periods=2).mean()    # ONI = 3-mo running Niño-3.4 anom

    months = [f"{d:%Y-%m}" for d in df.index]

    def col(name):
        return [round(float(v), 2) if pd.notna(v) else None for v in df[name].values]

    series = {
        "nino12": {"abs": col("n12"), "anom": col("n12a")},
        "nino3":  {"abs": col("n3"),  "anom": col("n3a")},
        "nino4":  {"abs": col("n4"),  "anom": col("n4a")},
        "nino34": {"abs": col("n34"), "anom": col("n34a")},
        "oni":    {"anom": [round(float(v), 2) if pd.notna(v) else None for v in oni.values]},
    }

    events = build_events(oni)
    # Current developing event: align to its onset-year's climatological Dec peak so the
    # lead-up sits against where analogs were before THEIR peaks. The latest data month
    # determines the onset year (an event 'belongs' to the year it peaks/develops into).
    last = df.index[-1]
    cur_onset = last.year if last.month >= 4 else last.year - 1   # ENSO year develops Apr→
    events.append({
        "id": "cur",
        "label": f"{cur_onset}–{str(cur_onset + 1)[2:]} (current)",
        "onset": cur_onset,
        "peak": f"{cur_onset}-12",          # assumed climatological DJF peak (no peak yet)
        "peakOni": None,
        "strength": "developing",
        "default": True,
        "current": True,
    })

    out = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "source": "NOAA CPC ERSSTv5 monthly Niño indices, 1991–2020 base",
        "latest_month": months[-1],
        "metrics": {
            "nino12": "Niño-1+2", "nino3": "Niño-3",
            "nino34": "Niño-3.4", "nino4": "Niño-4", "oni": "ONI (Niño-3.4, 3-mo)",
        },
        "months": months,
        "series": series,
        "events": events,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {OUT}  ({len(months)} months {months[0]}…{months[-1]}, "
          f"{len(events)} events, {OUT.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
