#!/usr/bin/env python3
"""Monthly Niño-region history + El Niño event table → JSON for the interactive explorer.

Sources, and which one wins:
  REGIONAL series — the four Niño boxes from the NOAA ERSSTv6 monthly grid (PSL
  noaa.ersst.v6 sst.mnmean.nc, the file sst.yml keeps current): cosine-weighted
  means of the grid points inside each box, anomalies against the FIXED 1991–2020
  calendar-month mean (nino_boxes). ERSST (not OISST) so the record reaches back
  past 1970; v6 because CPC moved its operational indices to it, and its v5 table
  (ersst5.nino.mth.91-20.ascii) froze on 5 Aug 2026 with content through 2026-06.

  ONI and RONI — CPC's OWN PUBLISHED values (cpc_official_enso.py). Until that
  freeze was noticed, every month after June came from the OISST provisional
  bridge and was presented as ERSST: 2026-07 read +2.10 where CPC had published
  +1.80. The official values are applied across the whole record rather than
  spliced onto the end, so no step appears mid-series.

  Anything CPC has not published yet is the site's own estimate, from the ERSSTv6
  grid or the OISST daily bridge, full 3-month seasons only, and is flagged by
  `provisional_after`.

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
from datetime import datetime, UTC
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SITE_ROOT = Path(os.environ["SST_SITE_ROOT"]).resolve() if os.environ.get("SST_SITE_ROOT") else HERE
OUT = SITE_ROOT / "assets" / "sst" / "data" / "nino_history.json"

START_YEAR = 1970  # history floor for the explorer
# CPC's Niño boxes: (lat south, lat north), (lon west, lon east) in degrees east
NINO_BOXES = {
    "n12": ((-10, 0), (270, 280)),
    "n3": ((-5, 5), (210, 270)),
    "n4": ((-5, 5), (160, 210)),
    "n34": ((-5, 5), (190, 240)),
}

# El Niño onset years since 1970 (year0 of the DJF event), per the CPC ONI record.
# Strength + exact peak month are computed from the data, not hard-coded.
ELNINO_ONSETS = [
    1972,
    1976,
    1977,
    1979,
    1982,
    1986,
    1987,
    1991,
    1994,
    1997,
    2002,
    2004,
    2006,
    2009,
    2014,
    2015,
    2018,
    2023,
]
# Shown checked by default (the strong / most-compared events + current).
DEFAULT_ON = {1982, 1997, 2015, 2023}


ERSST_GRID = "https://downloads.psl.noaa.gov/Datasets/noaa.ersst.v6/sst.mnmean.nc"


def ersst_grid():
    """The ERSSTv6 monthly SST grid (sst.yml keeps scripts/sst/data/ersst_v6_mnmean.nc current; fetched once if absent)."""
    import urllib.request
    import xarray as xr

    cache = HERE / "data" / "ersst_v6_mnmean.nc"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists() or cache.stat().st_size < 1e6:
        print(f"fetching {ERSST_GRID} (~60 MB, cached once)")
        urllib.request.urlretrieve(ERSST_GRID, cache)
    return xr.open_dataset(cache)


def _box_series(sst, box) -> pd.Series:
    """Cosine-weighted mean of the grid points inside a (lat, lon) box, monthly (month-start index)."""
    (la0, la1), (lo0, lo1) = box
    b = sst.sel(lat=slice(la1, la0), lon=slice(lo0, lo1))  # ERSST lat runs N→S
    ser = b.weighted(np.cos(np.deg2rad(b["lat"]))).mean(dim=("lat", "lon"), skipna=True).to_series()
    ser.index = pd.DatetimeIndex(ser.index).to_period("M").to_timestamp()
    return ser


def _anom_9120(ser: pd.Series) -> pd.Series:
    """Anomaly against the fixed 1991–2020 mean of each calendar month."""
    clim = ser[(ser.index.year >= 1991) & (ser.index.year <= 2020)].groupby(lambda t: t.month).mean()
    return ser - clim.reindex(ser.index.month).to_numpy()


def nino_boxes(ds) -> pd.DataFrame:
    """Monthly Niño-region SST (n12 n3 n4 n34, °C) and their anomalies (…a) from the ERSSTv6 grid, the columns of
    CPC's former ERSSTv5 table; months before every box has data are dropped."""
    cols = {}
    for k, box in NINO_BOXES.items():
        s = _box_series(ds["sst"], box)
        cols[k], cols[f"{k}a"] = s, _anom_9120(s)
    return pd.DataFrame(cols).dropna().sort_index()


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
        win = oni.loc[f"{y0}-07-01" : f"{y0 + 1}-03-31"]
        if win.dropna().empty:
            continue
        pk_date = win.idxmax()
        pk = float(win.max())
        events.append(
            {
                "id": str(y0),
                "label": f"{y0}–{str(y0 + 1)[2:]}",
                "onset": y0,
                "peak": f"{pk_date:%Y-%m}",
                "peakOni": round(pk, 2),
                "strength": strength(pk),
                "default": y0 in DEFAULT_ON,
            }
        )
    return events


def tropical_anom(ds, index: pd.DatetimeIndex) -> pd.Series:
    """20°S–20°N ERSSTv6 SST anomaly vs the FIXED 1991–2020 base, monthly: the tropical mean of the fallback RONI,
    from the same grid and base as the Niño boxes, so every year of it stands on one footing (CPC's own historical
    ONI/RONI use sliding 30-year bases, which is exactly the apples-to-oranges problem this avoids)."""
    return _anom_9120(_box_series(ds["sst"], ((-20, 20), (0, 360)))).reindex(index)


def roni_scale() -> dict:
    """Per-calendar-month σ(ONI)/σ(relative) — the same table the live site
    uses (CPC/ECMWF method), so historical and current RONI share one scale."""
    p = HERE / "roni_sigma.json"
    if not p.exists():
        return {}
    tab = json.loads(p.read_text()).get("scale_by_month", {})
    return {int(k): float(v) for k, v in tab.items()}


def main() -> int:
    ds = ersst_grid()
    df = nino_boxes(ds)
    df = df.loc[df.index.year >= START_YEAR]
    if df.empty:
        print("no Niño-box months in the ERSSTv6 grid", file=sys.stderr)
        return 1

    # ONI = 3-mo running Niño-3.4 anom; full seasons only: the season centred on the newest grid month needs the
    # month after it, and a 2-month mean must not stand in for it
    oni = df["n34a"].rolling(3, center=True, min_periods=3).mean()

    # Historical RONI on the same fixed footing: the grid's Niño-3.4 anomaly minus its tropical-mean anomaly,
    # σ-scaled per calendar month, 3-month running mean. A FALLBACK only: the official CPC values below overwrite it
    # wherever CPC has published, which since 1950 is everywhere.
    try:
        trop = tropical_anom(ds, df.index)
        sc = roni_scale()
        rel = (df["n34a"] - trop) * pd.Series([sc.get(t.month, 1.0) for t in df.index], index=df.index)
        roni = rel.rolling(3, center=True, min_periods=3).mean()
    except Exception as e:  # noqa: BLE001
        # The grid is only the FALLBACK side of RONI; CPC's official table below
        # covers every month since 1950, so an empty series still fills from it.
        # (2026-09-26: a PSL 502 here dropped RONI from the history entirely and
        # the forecasts page drew ONI under the RONI label.)
        print(
            f"tropical-mean/RONI fallback unavailable ({repr(e)[:80]}); RONI from CPC's official table only",
            file=sys.stderr,
        )
        roni = pd.Series(np.nan, index=df.index)

    # ── CPC's OWN published values take precedence over the grid-derived pair ──
    # Applied across the WHOLE record, not just the tail: the two differ by base
    # period (official ONI uses 30-year bases that shift every five years, the
    # grid pair a fixed 1991-2020), so splicing only the recent end would put a
    # step in the middle of the series.
    official_months = set()
    try:
        sys.path.insert(0, str(HERE))
        import cpc_official_enso as CPC

        off_oni, off_roni = CPC.fetch_oni(), CPC.fetch_roni()
        official_months = set(off_oni) | set(off_roni)

        def _override(ser, off):
            if ser is None:
                return 0
            n = 0
            for t in ser.index:
                v = off.get(f"{t:%Y-%m}")
                if v is not None:
                    ser.loc[t] = v
                    n += 1
            return n

        n_o, n_r = _override(oni, off_oni), _override(roni, off_roni)
        print(f"  CPC official: {n_o} ONI and {n_r} RONI months from CPC's own tables")
    except Exception as e:  # noqa: BLE001
        off_oni = off_roni = {}
        print(f"  CPC official unavailable ({repr(e)[:70]}); using table-derived ONI/RONI", file=sys.stderr)

    months = [f"{d:%Y-%m}" for d in df.index]

    def col(name):
        return [round(float(v), 2) if pd.notna(v) else None for v in df[name].values]

    series = {
        "nino12": {"abs": col("n12"), "anom": col("n12a")},
        "nino3": {"abs": col("n3"), "anom": col("n3a")},
        "nino4": {"abs": col("n4"), "anom": col("n4a")},
        "nino34": {"abs": col("n34"), "anom": col("n34a")},
        "oni": {"anom": [round(float(v), 2) if pd.notna(v) else None for v in oni.values]},
    }
    series["roni"] = {"anom": [round(float(v), 2) if pd.notna(v) else None for v in roni.values]}

    # Months CPC has published that the grid does not reach yet. The regional
    # series have no official counterpart, so they stay null there.
    for m in sorted(official_months - set(months)):
        if m <= months[-1]:
            continue
        months.append(m)
        for sub in series.values():
            for part in sub:
                sub[part].append(None)
        i = len(months) - 1
        if m in off_oni:
            series["oni"]["anom"][i] = round(float(off_oni[m]), 2)
        if m in off_roni:
            series["roni"]["anom"][i] = round(float(off_roni[m]), 2)
    official_through = max(official_months) if official_months else None
    if not any(v is not None for v in series["roni"]["anom"][-24:]):
        # neither the grid nor CPC's table gave RONI: keep the previous JSON
        # (the workflow step falls back to it) rather than publish without it
        print("no RONI from either source; not writing", file=sys.stderr)
        return 1

    events = build_events(oni)
    # Current developing event: align to its onset-year's climatological Dec peak so the
    # lead-up sits against where analogs were before THEIR peaks. The latest data month
    # determines the onset year (an event 'belongs' to the year it peaks/develops into).
    last = df.index[-1]
    cur_onset = last.year if last.month >= 4 else last.year - 1  # ENSO year develops Apr→
    events.append(
        {
            "id": "cur",
            "label": f"{cur_onset}–{str(cur_onset + 1)[2:]} (current)",
            "onset": cur_onset,
            "peak": f"{cur_onset}-12",  # assumed climatological DJF peak (no peak yet)
            "peakOni": None,
            "strength": "developing",
            "default": True,
            "current": True,
        }
    )

    # ── provisional bridge months from our OISST daily feed ────────────────
    # The ERSST month lags (a month appears in the first week of the next) and the
    # forecast chart then jumps last-ERSST-month → first-forecast-month. Our
    # OISST daily anomalies use the SAME 1991-2020 base (oisst9120 rebase), so
    # completed calendar months missing from ERSST are appended provisionally.
    provisional_from = None
    regional_from = None
    try:
        efeed = json.loads((OUT.parent / "enso_daily.json").read_text())
        daily = efeed["daily"]
        ddates = [d[:7] for d in daily["dates"]]
        import calendar as _cal

        # The bridge starts after the last ERSST month, not after months[-1]: CPC-official ONI/RONI months appended
        # above carry null regional values, and starting after them would leave the four Niño boxes blank there.
        # Those months get their regional values from the bridge; CPC's ONI/RONI stay where CPC published.
        last_table = f"{df.index[-1]:%Y-%m}"
        DMAP = {
            ("nino34", "anom"): "nino34",
            ("nino34", "abs"): "nino34_abs",
            ("nino3", "anom"): "nino3",
            ("nino3", "abs"): "nino3_abs",
            ("nino4", "anom"): "nino4",
            ("nino4", "abs"): "nino4_abs",
            ("nino12", "anom"): "nino12",
            ("nino12", "abs"): "nino12_abs",
        }  # 1+2 and the Nino-3 SST were missing
        mon = efeed.get("monthly", {})
        for m in sorted({x for x in ddates if x > last_table}):
            idx = [i for i, mm in enumerate(ddates) if mm == m]
            y, mo = int(m[:4]), int(m[5:7])
            if len(idx) < _cal.monthrange(y, mo)[1]:
                continue  # incomplete month
            if m in months:
                i = months.index(m)  # an official ONI/RONI month: fill only what is empty
            else:
                months.append(m)
                for sub in series.values():
                    for part in sub:
                        sub[part].append(None)
                i = len(months) - 1
            for k, sub in series.items():
                for part in sub:
                    if sub[part][i] is not None:
                        continue
                    val = None
                    dk = DMAP.get((k, part))
                    if dk and daily.get(dk):
                        vals = [daily[dk][j] for j in idx if daily[dk][j] is not None]
                        if len(vals) == len(idx):  # a whole month of days or nothing
                            val = round(sum(vals) / len(vals), 2)
                    elif k in ("oni", "roni") and part == "anom" and m in mon.get("months", []):
                        val = mon[k][mon["months"].index(m)]
                    sub[part][i] = val
            regional_from = regional_from or last_table
            provisional_from = provisional_from or (official_through or last_table)
    except Exception as _e:  # noqa: BLE001
        print(f"  provisional bridge skipped: {_e}")
    # an ONI/RONI season past CPC's newest is the site's estimate whichever source gave it (the grid's fixed-base
    # fallback, or the bridge): flag it, so no consumer draws it as CPC's
    if official_through and any(
        v is not None for k in ("oni", "roni") for m, v in zip(months, series[k]["anom"]) if m > official_through
    ):
        provisional_from = provisional_from or official_through

    out = {
        "generated": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "provisional_after": provisional_from,  # ONI/RONI: months past this are the site's estimate, not CPC's
        "regional_provisional_after": regional_from,  # Nino boxes: months past this are OISST monthly means
        "source": (
            "ONI/RONI: CPC's published tables (official; RONI is ERSSTv6 and has "
            "been CPC's official ENSO index since Feb 2026); months past provisional_after "
            "are the site's estimate (ERSSTv6 grid on a fixed 1991–2020 base, or OISST v2.1), full "
            "3-month seasons only. Regional Niño series: box means of the NOAA ERSSTv6 "
            "grid, fixed 1991–2020 base; months past regional_provisional_after are OISST v2.1 "
            "monthly means on the same 1991–2020 base."
        ),
        "latest_month": months[-1],
        "metrics": {
            "nino12": "Niño-1+2",
            "nino3": "Niño-3",
            "nino34": "Niño-3.4",
            "nino4": "Niño-4",
            "oni": "ONI (Niño-3.4, 3-mo)",
            "roni": "RONI (relative, σ-scaled, 3-mo — fixed 1991–2020 base)",
        },
        "months": months,
        "series": series,
        "events": events,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, separators=(",", ":")))
    print(
        f"wrote {OUT}  ({len(months)} months {months[0]}…{months[-1]}, "
        f"{len(events)} events, {OUT.stat().st_size / 1024:.0f} KB)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
