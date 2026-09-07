#!/usr/bin/env python3
"""US climate trends by county and month from NOAA nClimDiv (1895 → last month).

Inputs (downloaded to scripts/climate/data, gitignored): the county, division and state
monthly files for average / maximum / minimum temperature, precipitation, heating and cooling
degree days. County values are area means of the 5 km nClimGrid (homogenised station data), so
they are consistent with the division and state files in the same directory.

For every county, month and season (DJF MAM JJA SON, annual) and each variable:
  normal        1991–2020 mean (°F, inches, degree days)
  trend         OLS slope per decade from 1970, 1950 and 1895 to the last complete year
                (precipitation as % of the 1991–2020 normal per decade), with the Theil–Sen slope
                and a Mann–Kendall p-value (Kendall's tau) as the robustness / significance checks
  adjusted      the "trend-adjusted normal": the 1970-start fit evaluated at the current year —
                what normal is now, if the last five decades' trend continues
Outputs (assets/climate/): trends_{var}_{start}.json, normals_{var}.json, regions.json (states ×
months), series/{state fips}_{var}.json (per-county monthly series for the explorer), meta.json.

    python scripts/climate/trends/nclimdiv_trends.py [--no-download]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
SITE = Path(os.environ.get("SST_SITE_ROOT", HERE.parents[2]))
# Everything lands under assets/climate/anim so the frames publisher mirrors it to the orphan
# `frames` branch (replaced wholesale each month — no history growth on main): ~25 MB of
# trend/normal JSON plus ~60 MB of Int16 county series. The page reads it from the frames host.
OUT = SITE / "assets" / "climate" / "anim" / "data"
SERIES = SITE / "assets" / "climate" / "anim" / "series"
BASE = "https://www.ncei.noaa.gov/monitoring-content/data/us/climdiv/monthly/current/"
UA = {"User-Agent": "Mozilla/5.0 (scorvec.com climate trends; contact: site owner)"}
VARS = {"tavg": ("tmpc", "02"), "tmax": ("tmax", "27"), "tmin": ("tmin", "28"), "pcpn": ("pcpn", "01"), "hdd": ("hddc", "25"), "cdd": ("cddc", "26")}
STARTS = (1970, 1950, 1895)
NORMAL = (1991, 2020)
SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11), "ANN": tuple(range(1, 13))}
# NOAA climdiv state code → FIPS state code
NOAA2FIPS = {1: "01", 2: "04", 3: "05", 4: "06", 5: "08", 6: "09", 7: "10", 8: "12", 9: "13", 10: "16", 11: "17", 12: "18", 13: "19", 14: "20", 15: "21",
             16: "22", 17: "23", 18: "24", 19: "25", 20: "26", 21: "27", 22: "28", 23: "29", 24: "30", 25: "31", 26: "32", 27: "33", 28: "34", 29: "35",
             30: "36", 31: "37", 32: "38", 33: "39", 34: "40", 35: "41", 36: "42", 37: "44", 38: "45", 39: "46", 40: "47", 41: "48", 42: "49", 43: "50",
             44: "51", 45: "53", 46: "54", 47: "55", 48: "56", 49: "15", 50: "02"}
STATE_NAME = {"01": "Alabama", "04": "Arizona", "05": "Arkansas", "06": "California", "08": "Colorado", "09": "Connecticut", "10": "Delaware", "12": "Florida", "13": "Georgia",
              "16": "Idaho", "17": "Illinois", "18": "Indiana", "19": "Iowa", "20": "Kansas", "21": "Kentucky", "22": "Louisiana", "23": "Maine", "24": "Maryland", "25": "Massachusetts",
              "26": "Michigan", "27": "Minnesota", "28": "Mississippi", "29": "Missouri", "30": "Montana", "31": "Nebraska", "32": "Nevada", "33": "New Hampshire", "34": "New Jersey",
              "35": "New Mexico", "36": "New York", "37": "North Carolina", "38": "North Dakota", "39": "Ohio", "40": "Oklahoma", "41": "Oregon", "42": "Pennsylvania", "44": "Rhode Island",
              "45": "South Carolina", "46": "South Dakota", "47": "Tennessee", "48": "Texas", "49": "Utah", "50": "Vermont", "51": "Virginia", "53": "Washington", "54": "West Virginia",
              "55": "Wisconsin", "56": "Wyoming", "15": "Hawaii", "02": "Alaska"}
REGION = {"Northeast": ["09", "23", "25", "33", "34", "36", "42", "44", "50"], "Southeast": ["01", "12", "13", "37", "45", "51"], "Ohio Valley": ["17", "18", "21", "29", "39", "47", "54"],
          "Upper Midwest": ["19", "26", "27", "55"], "South": ["05", "20", "22", "28", "40", "48"], "Northern Rockies & Plains": ["30", "31", "38", "46", "56"],
          "Southwest": ["04", "08", "35", "49"], "Northwest": ["16", "41", "53"], "West": ["06", "32"]}


# ── fetch ────────────────────────────────────────────────────────────────────
def latest_files(download: bool) -> dict:
    DATA.mkdir(parents=True, exist_ok=True)
    listing = ""
    if download:
        try:
            with urllib.request.urlopen(urllib.request.Request(BASE, headers=UA), timeout=60) as r:
                listing = r.read().decode("utf-8", "replace")
        except Exception as e:                                          # noqa: BLE001
            print(f"  listing failed ({e}); using cached files", flush=True)
    out = {}
    for var, (stem, _) in VARS.items():
        for scope in ("cy", "dv", "st"):
            name = None
            if listing:
                names = sorted(set(re.findall(rf"climdiv-{stem}{scope}-v1\.0\.0-\d+", listing)))
                name = names[-1] if names else None
            if name and not (DATA / name).exists():
                print(f"  downloading {name}", flush=True)
                urllib.request.urlretrieve(BASE + name, DATA / name) if not UA else _dl(BASE + name, DATA / name)
            if not name:
                cached = sorted(DATA.glob(f"climdiv-{stem}{scope}-v1.0.0-*"))
                name = cached[-1].name if cached else None
            if name:
                out[(var, scope)] = DATA / name
    return out


def _dl(url, dest):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=600) as r, open(dest, "wb") as f:
        f.write(r.read())


def parse(path: Path, county: bool, var: str = "tavg", scope: str = "cy") -> dict:
    """{key: {year: [12 values]}} with key = FIPS county (5) / division (state fips + 2) / state fips.
    Missing sentinels differ by element: −99.90 for temperatures (real monthly means reach −20 °F
    in the northern Plains, so only the sentinel itself is missing), −9.99 for precipitation and
    −9999 for degree days (neither can be negative)."""
    temp = var in ("tavg", "tmax", "tmin")
    out = {}
    with open(path) as f:
        for line in f:
            if county:
                st = int(line[0:2]); cty = line[2:5]; yr = int(line[7:11]); vals = line[11:]
                key = NOAA2FIPS.get(st, None)
                if key is None:
                    continue
                key = key + cty
            elif scope == "st":                                          # state files: 3-digit code, element, year
                st = int(line[0:3]); yr = int(line[6:10]); vals = line[10:]     # code(3) division(1) element(2) year(4)
                key = NOAA2FIPS.get(st, None)
                if key is None:                                          # 101–110 are regions / national
                    continue
            else:                                                        # division files: state 2, division 2
                st = int(line[0:2]); div = line[2:4]; yr = int(line[6:10]); vals = line[10:]
                key = NOAA2FIPS.get(st, None)
                if key is None:
                    continue
                key = key + div
            v = np.array([float(vals[i * 7:(i + 1) * 7]) for i in range(12)])
            if temp:
                v[v <= -99.0] = np.nan
            else:
                v[v < 0] = np.nan
            out.setdefault(key, {})[yr] = v
    return out


# ── statistics ───────────────────────────────────────────────────────────────
def to_matrix(series: dict, y0: int, y1: int) -> np.ndarray:
    m = np.full((y1 - y0 + 1, 12), np.nan)
    for yr, v in series.items():
        if y0 <= yr <= y1:
            m[yr - y0] = v
    return m


def season_values(m: np.ndarray, y0: int, var: str) -> np.ndarray:
    """(years, 17): 12 months then DJF MAM JJA SON ANN. DJF uses Dec of the previous year.
    Temperatures average; precipitation and degree days sum."""
    n = m.shape[0]; out = np.full((n, 17), np.nan); out[:, :12] = m
    agg = np.nanmean if var in ("tavg", "tmax", "tmin", "tdew") else np.nansum
    for k, (name, months) in enumerate(SEASONS.items()):
        for i in range(n):
            if name == "DJF":
                if i == 0:
                    continue
                vals = np.array([m[i - 1, 11], m[i, 0], m[i, 1]])
            else:
                vals = m[i, [mo - 1 for mo in months]]
            if np.isnan(vals).any():
                continue
            out[i, 12 + k] = agg(vals)
    return out


def trend_stats(y: np.ndarray, years: np.ndarray, start: int, last: int):
    sel = (years >= start) & (years <= last) & np.isfinite(y)
    if sel.sum() < max(20, int(0.7 * (last - start + 1))):
        return None
    x = years[sel].astype(float); v = y[sel]
    slope, intercept = np.polyfit(x, v, 1)
    ts = stats.theilslopes(v, x)[0]
    tau, p = stats.kendalltau(x, v)
    return slope * 10, ts * 10, p, intercept + slope * (last + 1)


def load_tdew():
    """PRISM county dewpoint archive (prism_tdmean.py) → ({fips: {year: [12]}}, {state: {year: [12]}}) in °F,
    or None if the archive is absent. State series are cell-count-weighted means of the counties."""
    for d in (OUT, HERE / "seed"):
        b, j = d / "tdmean_county.bin", d / "tdmean_index.json"
        if b.exists() and j.exists():
            break
    else:
        return None
    idx = json.loads(j.read_text()); n = len(idx["fips"]); nm = idx["n_months"]
    arr = np.fromfile(b, dtype="<i2").reshape(n, nm).astype("float64")
    arr[arr == -32768] = np.nan; arr /= 100.0
    y0 = int(idx["first"][:4]); ny = (nm + 11) // 12
    full = np.full((n, ny * 12), np.nan); full[:, :nm] = arr; full = full.reshape(n, ny, 12)
    counts = np.array(idx["counts"], float)
    cy, st = {}, {}
    for i, f in enumerate(idx["fips"]):
        if counts[i] <= 0 or len(f) != 5:
            continue
        cy[f] = {y0 + k: full[i, k] for k in range(ny)}
    for sf in sorted({f[:2] for f in cy}):
        ii = [i for i, f in enumerate(idx["fips"]) if f[:2] == sf and counts[i] > 0]
        w = counts[ii][:, None, None]; v = full[ii]
        num = np.nansum(v * w, axis=0); den = np.sum(np.where(np.isfinite(v), w, 0.0), axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            sv = np.where(den > 0, num / den, np.nan)
        st[sf] = {y0 + k: sv[k] for k in range(ny)}
    print(f"  tdew: {len(cy)} counties from PRISM ({idx.get('first')} → {idx.get('last')})", flush=True)
    return cy, st


def build_units(var: str, normal: float, slope_dec: float):
    """Precipitation trends as % of normal per decade; the rest in native units per decade."""
    if var == "pcpn":
        return (slope_dec / normal * 100.0) if normal and normal > 0.05 else np.nan
    return slope_dec


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--no-download", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    files = latest_files(download=not a.no_download)
    if not files:
        raise SystemExit("no nClimDiv files")
    OUT.mkdir(parents=True, exist_ok=True); SERIES.mkdir(parents=True, exist_ok=True)
    series_index = {}
    meta = {"files": {f"{v}_{s}": p.name for (v, s), p in files.items()}, "starts": list(STARTS), "normal": list(NORMAL),
            "columns": [f"{m:02d}" for m in range(1, 13)] + list(SEASONS)}
    regions = {}
    tdew = load_tdew()
    for var in list(VARS) + (["tdew"] if tdew else []):
        if var == "tdew":
            cy, st_series = tdew
        else:
            cy = parse(files[(var, "cy")], county=True, var=var)
        years_all = sorted({y for s in cy.values() for y in s})
        y0, ylast = years_all[0], years_all[-1]
        # last complete year: the current year is partial (values through last month only)
        sample = next(iter(cy.values()))
        last_full = ylast if np.isfinite(sample.get(ylast, np.full(12, np.nan))).all() else ylast - 1
        years = np.arange(y0, ylast + 1)
        print(f"  {var}: {len(cy)} counties, {y0}–{ylast}, last complete year {last_full} ({time.time() - t0:.0f}s)", flush=True)
        normals, adjusted = {}, {}
        trends = {s: {} for s in STARTS}; pvals = {s: {} for s in STARTS}; sens = {s: {} for s in STARTS}
        per_state = {}
        for fips, ser in cy.items():
            m = to_matrix(ser, y0, ylast); sv = season_values(m, y0, var)
            nsel = (years >= NORMAL[0]) & (years <= NORMAL[1])
            nrm = np.nanmean(sv[nsel], axis=0)
            normals[fips] = [None if not np.isfinite(x) else round(float(x), 2) for x in nrm]
            adj = []
            for s in STARTS:
                tr, pv, tsl = [], [], []
                for c in range(17):
                    r = trend_stats(sv[:, c], years, s, last_full)
                    if r is None:
                        tr.append(None); pv.append(None); tsl.append(None)
                        if s == 1970: adj.append(None)
                        continue
                    slope, ts, p, fit_now = r
                    tr.append(None if not np.isfinite(build_units(var, nrm[c], slope)) else round(float(build_units(var, nrm[c], slope)), 3))
                    tsl.append(None if not np.isfinite(build_units(var, nrm[c], ts)) else round(float(build_units(var, nrm[c], ts)), 3))
                    pv.append(round(float(p), 3) if np.isfinite(p) else None)
                    if s == 1970:
                        adj.append(round(float(fit_now), 2) if np.isfinite(fit_now) else None)
                trends[s][fips] = tr; pvals[s][fips] = pv; sens[s][fips] = tsl
            adjusted[fips] = adj
            per_state.setdefault(fips[:2], {})[fips] = m
        json.dump({"var": var, "normal": normals, "adjusted_1970": adjusted, "columns": meta["columns"]}, open(OUT / f"normals_{var}.json", "w"), separators=(",", ":"))
        for s in STARTS:
            json.dump({"var": var, "start": s, "end": int(last_full), "units": ("% of normal per decade" if var == "pcpn" else ("°F per decade" if var.startswith("t") else "degree days per decade")),
                       "ols": trends[s], "sen": sens[s], "p": pvals[s], "columns": meta["columns"]}, open(OUT / f"trends_{var}_{s}.json", "w"), separators=(",", ":"))
        for st, d in per_state.items():
            # Int16 x100 (°F, inches; degree days x1), little-endian, [county][year][month]; NaN -> -32768
            scale = 1 if var in ("hdd", "cdd") else 100
            fips_list = sorted(d)
            arr = np.stack([d[f] for f in fips_list]) * scale
            arr = np.where(np.isfinite(arr), np.clip(np.round(arr), -32767, 32767), -32768).astype("<i2")
            arr.tofile(SERIES / f"{st}_{var}.bin")
            series_index.setdefault(st, {"fips": fips_list, "years": [int(y0), int(ylast)]})
            series_index[st][f"scale_{var}"] = scale
        # states (from the state files; dewpoint: cell-weighted county mean) for the heatmap
        if var == "tdew" or (var, "st") in files:
            stt = st_series if var == "tdew" else parse(files[(var, "st")], county=False, var=var, scope="st")
            for st, ser in stt.items():
                m = to_matrix(ser, y0, ylast); sv = season_values(m, y0, var)
                nsel = (years >= NORMAL[0]) & (years <= NORMAL[1]); nrm = np.nanmean(sv[nsel], axis=0)
                rec = regions.setdefault(st, {"name": STATE_NAME.get(st, st)})
                for s in STARTS:
                    vals = []
                    for c in range(17):
                        r = trend_stats(sv[:, c], years, s, last_full)
                        vals.append(None if r is None or not np.isfinite(build_units(var, nrm[c], r[0])) else round(float(build_units(var, nrm[c], r[0])), 3))
                    rec[f"{var}_{s}"] = vals
                rec[f"{var}_normal"] = [None if not np.isfinite(x) else round(float(x), 2) for x in nrm]
        meta[f"last_month_{var}"] = f"{ylast}-{int(np.where(np.isfinite(sample.get(ylast, np.full(12, np.nan))))[0].max() + 1):02d}" if ylast in sample else None
        meta[f"last_full_year_{var}"] = int(last_full)
    json.dump({"states": regions, "regions": REGION}, open(OUT / "regions.json", "w"), separators=(",", ":"))
    json.dump(series_index, open(OUT / "series_index.json", "w"), separators=(",", ":"))
    meta["generated"] = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    json.dump(meta, open(OUT / "meta.json", "w"), indent=1)
    print(f"wrote {OUT} in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
