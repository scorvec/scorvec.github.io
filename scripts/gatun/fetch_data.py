"""Fetch Lake Gatun levels (ACP) + ONI (NOAA CPC) and build data.js.

Sources (personal/informational use):
  https://evtms-rpts.pancanal.com/eng/h2o/index.html  (history + projection CSV)
  https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt

Outputs gatun/data.js: `const GATUN = {...}` consumed by gatun/index.html.
Run from repo root: python3 scripts/gatun/fetch_data.py   (needs requests + numpy)
"""
import csv
import io
import json
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

import numpy as np
import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
UA = {"User-Agent": "Mozilla/5.0 (personal research)"}

HISTORY = "https://evtms-rpts.pancanal.com/eng/h2o/Download_Gatun_Lake_Water_Level_History.csv"
PROJECTION = "https://evtms-rpts.pancanal.com/eng/h2o/Gatun_Water_Level_Projection.csv"
ONI = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"


def fetch(url, **kw):
    r = requests.get(url, headers=UA, timeout=60, verify=kw.pop("verify", True))
    r.raise_for_status()
    return r.text


def main():
    # ---- history ---------------------------------------------------------
    try:
        hist_text = fetch(HISTORY)
    except requests.exceptions.SSLError:
        hist_text = fetch(HISTORY, verify=False)
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "data" / "gatun_history.csv").write_text(hist_text)

    by_year = defaultdict(lambda: {})   # year -> {doy: level}
    series_d, series_v = [], []
    for row in csv.DictReader(io.StringIO(hist_text)):
        try:
            d = datetime.strptime(row["DATE_LOG"], "%Y-%m-%d").date()
            v = float(row["GATUN_LAKE_LEVEL(FEET)"])
        except (ValueError, KeyError):
            continue
        if not 75.0 <= v <= 90.0:   # ACP uses .00 (and similar) as missing-data sentinels
            continue
        doy = (d - date(d.year, 1, 1)).days  # 0-based, Feb 29 folds naturally
        by_year[d.year][doy] = v
        series_d.append(d.isoformat())
        series_v.append(v)

    years = sorted(by_year)
    # per-year arrays padded with None
    seasonal = {str(y): [by_year[y].get(i) for i in range(366)] for y in years}

    # day-of-year climatology across full period (p10/median/p90)
    clim = {"p10": [], "p50": [], "p90": []}
    for i in range(366):
        vals = [by_year[y][i] for y in years if i in by_year[y]]
        if len(vals) >= 10:
            p10, p50, p90 = np.percentile(vals, [10, 50, 90])
        else:
            p10 = p50 = p90 = None
        clim["p10"].append(None if p10 is None else round(float(p10), 2))
        clim["p50"].append(None if p50 is None else round(float(p50), 2))
        clim["p90"].append(None if p90 is None else round(float(p90), 2))

    # ---- projection ------------------------------------------------------
    proj = {"d": [], "v": []}
    try:
        try:
            ptext = fetch(PROJECTION)
        except requests.exceptions.SSLError:
            ptext = fetch(PROJECTION, verify=False)
        lines = [l for l in ptext.splitlines() if l.count(",") >= 2]
        rdr = csv.DictReader(io.StringIO("\n".join(lines)))
        for row in rdr:
            try:
                d = datetime.strptime(row["projected_date"].strip(), "%m/%d/%Y").date()
                proj["d"].append(d.isoformat())
                proj["v"].append(float(row["projected_gatun_water_level"]))
            except (ValueError, KeyError, AttributeError):
                continue
    except requests.RequestException:
        pass

    # ---- ONI -------------------------------------------------------------
    oni_rows = []
    for line in fetch(ONI).splitlines()[1:]:
        parts = line.split()
        if len(parts) == 4:
            oni_rows.append((parts[0], int(parts[1]), float(parts[3])))
    # DJF anomaly labeled with the Jan/Feb year: the peak that drives that
    # year's dry season (Jan-May) in Panama
    oni_djf = {yr: anom for seas, yr, anom in oni_rows if seas == "DJF"}
    latest_seas, latest_yr, latest_oni = oni_rows[-1]
    # full monthly series (season center month) for episode shading
    SEAS_MONTH = {"DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
                  "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12}
    oni_series = {"d": [f"{yr}-{SEAS_MONTH[s]:02d}-15" for s, yr, a in oni_rows],
                  "a": [a for s, yr, a in oni_rows]}

    # ---- per-year summaries ----------------------------------------------
    year_stats = {}
    for y in years:
        days = by_year[y]
        dry = [days[i] for i in range(0, 151) if i in days]          # Jan-May
        year_stats[str(y)] = {
            "oni_djf": oni_djf.get(y),
            # require decent coverage: a near-empty year (e.g. 2002, almost all
            # missing-data sentinels) must not masquerade as a mild dry season
            "dry_min": round(min(dry), 2) if len(dry) >= 60 else None,
            "min": round(min(days.values()), 2),
            "max": round(max(days.values()), 2),
        }

    # ---- monthly anomaly matrix (stripes) ---------------------------------
    monthly = defaultdict(lambda: defaultdict(list))  # month -> year -> vals
    for dstr, v in zip(series_d, series_v):
        y, m = int(dstr[:4]), int(dstr[5:7])
        monthly[m][y].append(v)
    mclim = {m: np.mean([np.mean(monthly[m][y]) for y in years if monthly[m].get(y)])
             for m in range(1, 13)}
    stripes = [[(round(float(np.mean(monthly[m][y]) - mclim[m]), 2)
                 if monthly[m].get(y) else None)
                for y in years] for m in range(1, 13)]

    data = {
        "updated": series_d[-1],
        "latest": series_v[-1],
        "years": years,
        "seasonal": seasonal,
        "clim": clim,
        "series": {"d": series_d, "v": series_v},
        "proj": proj,
        "year_stats": year_stats,
        "stripes": stripes,
        "oni_latest": {"season": latest_seas, "year": latest_yr, "anom": latest_oni},
        "oni_series": oni_series,
    }
    out = ROOT / "gatun" / "data.js"
    out.write_text("const GATUN = " + json.dumps(data) + ";\n")
    print(f"wrote {out} ({out.stat().st_size // 1024}KB): {years[0]}-{years[-1]}, "
          f"latest {series_d[-1]} = {series_v[-1]} ft, "
          f"ONI {latest_seas} {latest_yr} = {latest_oni:+.2f}, "
          f"projection days: {len(proj['d'])}")


if __name__ == "__main__":
    main()
