#!/usr/bin/env python3
"""This winter on the "Every winter since 1980" chart (stratosphere.html#shwinters): the observed daily series of the
current northern winter, drawn over the MERRA-2 winters of strat_history.json.

Our MERRA-2 record ends on 2026-06-26 (GES DISC retired its OPeNDAP service), so this winter comes from NASA GMAO's
GEOS FP analyses (the same GEOS system; NCCS OPeNDAP inst3_3d_asm_Np), reduced exactly as the MERRA-2 history was
(fetch_m2_strat_series.py, build_strat_history.py): the 00/06/12/18Z analyses on 1-degree latitudes and every 2.5
degrees of longitude, the zonal mean of each, their daily mean, then
  u10    the zonal-mean wind at 60N, 10 hPa (m/s)
  t10    the cos-weighted 65-90N mean of the zonal-mean temperature at 10 hPa (K)
  z100   the same mean of the height at 100 hPa (m), and z100s, that height standardised against the MERRA-2 fit of the
         history (reference/drip_std.json, build_drip_ref.py: 4 harmonics + trend, 3-harmonic sd, on the common-year
         day; over an October-May winter its mean and sd sit within 0.03 sd of the fit behind the chart's own z100s)
GEOS FP is drawn as it is, with no offset applied; GEOS FP minus MERRA-2 is measured on the 2025/26 winter, every day
both hold (`overlap` -> reference/strat_season_overlap.json, quoted on the page: on 239 days u10 -0.19 m/s, t10
+0.33 K, z100 -8.9 m = -0.09 sd, 1-4 % of the chart's 10-90 % band over the winter). The MERRA-2 side of those days is
reduced once from the local MERRA-2 archive (`m2-overlap` -> reference/strat_season_m2.json), so the overlap itself
runs in Actions (strat-season-ref.yml).

The record keeps every day of the chart's winter axis (strat_history.json "days", 1 October to 31 May) and is never
pruned; each run fills the days of the current winter not yet held, so the first run backfills the season and a day
NCCS missed is caught up on the next.

    python strat_season.py m2-overlap      # -> reference/strat_season_m2.json   (needs the MERRA-2 archive)
    python strat_season.py overlap         # -> reference/strat_season_overlap.json   (strat-season-ref.yml)
    python strat_season.py update          # -> assets/sst/data/strat_season.json   (bdc.yml, daily)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "scripts" / "lib"))
from build_drip_ref import Ref
from upperstrat_data import GEOS_AS, Geos

OUT = REPO / "assets" / "sst" / "data" / "strat_season.json"
HISTORY = REPO / "assets" / "sst" / "data" / "strat_history.json"
M2_SIDE = HERE / "reference" / "strat_season_m2.json"
OVERLAP = HERE / "reference" / "strat_season_overlap.json"
OVERLAP_DAYS = ("2025-10-01", "2026-05-31")  # the last MERRA-2 winter on the chart's axis
COLS = ("u10", "t10", "z100", "z100s")
DIGITS = {"u10": 1, "t10": 1, "z100": 1, "z100s": 2}  # strat_history.json's rounding (z100 to 0.1 m)
CHART = ("u10", "t10", "z100s")  # the overlap statistics with a band on the chart


def log(*a):
    print(*a, flush=True)


class Fp(Geos):
    """The GEOS FP analyses with the history's sampling: 60-90N every degree, every 2.5 degrees of longitude, the 100
    and 10 hPa levels in one strided slab."""

    def __init__(self):
        super().__init__(GEOS_AS)
        j60 = int(np.argmin(np.abs(self.lat - 60.0)))
        self.rows = slice(j60, len(self.lat), 4)  # 0.25 deg grid: 60, 61, ... 90N
        k100, k10 = (int(np.argmin(np.abs(self.lev - L))) for L in (100.0, 10.0))
        self.levs = slice(k100, k10 + 1, k10 - k100)
        self.cols = slice(None, None, 8)  # 0.3125 deg: every 2.5 deg, as MERRA-2's 0.625 deg taken every 4th
        tn = self.times[-1]
        self.end = tn.normalize() if tn.hour >= 18 else tn.normalize() - pd.Timedelta(days=1)  # newest complete day

    def day(self, d: pd.Timestamp):
        """(u10, t10, z100) of day d, or None when an analysis is missing or fails the MERRA-2 fetcher's checks."""
        it = int(np.argmin(np.abs(self.times - d)))
        want = d + pd.to_timedelta([0, 6, 12, 18], "h")
        got = self.times[it : it + 7 : 2]
        if len(got) < 4 or not (got == want).all():
            log(f"  {d:%Y-%m-%d}: not every 00/06/12/18Z analysis is in the archive")
            return None
        try:
            f, lat, _, lev = self.read(("u", "t", "h"), slice(it, it + 7, 2), self.levs, self.rows, self.cols)
        except Exception as e:  # noqa: BLE001
            log(f"  {d:%Y-%m-%d}: {str(e)[:100]}")
            return None
        if not (np.nanmax(np.abs(f["u"])) < 400 and np.nanmin(f["t"]) > 140 and np.nanmax(f["t"]) < 340):
            log(f"  {d:%Y-%m-%d}: values out of range")
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            # the zonal mean of each analysis, then their daily mean: (level, latitude)
            zm = {v: np.nanmean(np.nanmean(f[v], -1), 0) for v in f}
        k10, k100 = (int(np.argmin(np.abs(lev - L))) for L in (10.0, 100.0))
        cap = lat >= 65.0 - 1e-6
        w = np.cos(np.deg2rad(lat[cap]))
        out = (
            float(zm["u"][k10, int(np.argmin(np.abs(lat - 60.0)))]),
            float((zm["t"][k10, cap] * w).sum() / w.sum()),
            float((zm["h"][k100, cap] * w).sum() / w.sum()),
        )
        return out if np.isfinite(out).all() else None


def winter_axis(end: pd.Timestamp) -> pd.DatetimeIndex:
    """The days up to `end` of the chart's winter axis (strat_history.json "days", MM-DD from October) in the winter
    holding `end` (between the axis's last day and the next first day, the winter just ended), after the MERRA-2
    period the chart already draws."""
    h = json.loads(HISTORY.read_text())
    md, m2_end = h["days"], pd.Timestamp(h["period"].split("..")[1])
    y = end.year if f"{end:%m-%d}" >= md[0] else end.year - 1
    days = pd.date_range(pd.Timestamp(f"{y}-{md[0]}"), min(end, pd.Timestamp(f"{y + 1}-{md[-1]}")), freq="D")
    return days[days > m2_end]


def z100s(days, z100) -> np.ndarray:
    return Ref().z(pd.DatetimeIndex(days), 100.0, np.asarray(z100, float))


def update(a) -> int:
    have = json.loads(OUT.read_text())["days"] if OUT.exists() else {}
    fp = Fp()
    days = [d for d in winter_axis(fp.end) if f"{d:%Y-%m-%d}" not in have]
    log(f"GEOS FP this winter: {len(days)} day(s) to fetch, newest complete analysis day {fp.end:%Y-%m-%d}")
    n = 0
    for d in days:
        t0 = time.time()
        r = fp.day(d)
        if r is None:
            continue
        z = float(z100s([d], [r[2]])[0])
        have[f"{d:%Y-%m-%d}"] = [round(v, DIGITS[c]) for c, v in zip(COLS, (*r, z))]
        n += 1
        log(
            f"  {d:%Y-%m-%d}: u10 {r[0]:+.1f} m/s, t10 {r[1]:.1f} K, z100 {r[2]:.0f} m ({z:+.2f} sd), {time.time() - t0:.0f} s"
        )
        if n % 10 == 0:
            write(have)
    if n:
        write(have)
    log(f"GEOS FP this winter: {n} new day(s), {len(have)} held -> {OUT.name}")
    return 0


def write(have: dict):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "source": "NASA GMAO GEOS FP analyses (inst3_3d_asm_Np, NCCS OPeNDAP): daily mean of the 00/06/12/18Z "
                "zonal means, 1 x 2.5 deg sampling as the MERRA-2 history; u10 = u at 60N 10 hPa (m/s), t10 = "
                "cos-weighted 65-90N T at 10 hPa (K), z100 = the same mean of the 100 hPa height (m), z100s = z100 "
                "standardised by the MERRA-2 fit (scripts/strat/reference/drip_std.json); no offset applied",
                "label": "GEOS FP analyses",
                "columns": list(COLS),
                "days": dict(sorted(have.items())),
            },
            separators=(",", ":"),
        )
    )


def m2_overlap(a) -> int:
    """The MERRA-2 side of the overlap from the raw yearly archive: u60 at 10 hPa, cap_T at 10 hPa and cap_z at 100 hPa
    (65-90N) on every day of OVERLAP_DAYS the archive holds, as fetched (no gap filling)."""
    import xarray as xr
    from localdirs import env_dir

    days = pd.date_range(*OVERLAP_DAYS, freq="D")
    src = env_dir("DATA_ARCHIVE", "m2_strat")
    rows = {}
    for y in sorted(set(days.year)):
        with xr.open_dataset(src / f"m2_strat_{y}.nc") as d:
            u = d["u60"].sel(lev=10.0).to_series()
            t = d["cap_T"].sel(lev=10.0).to_series()
            z = d["cap_z"].sel(lev=100.0).to_series()
        for k in u.index.intersection(days):
            v = (float(u[k]), float(t[k]), float(z[k]))
            if np.isfinite(v).all():
                rows[f"{k:%Y-%m-%d}"] = [round(x, 3) for x in v]
    M2_SIDE.write_text(
        json.dumps(
            {
                "source": "MERRA-2 M2I3NPASM daily mean of the 00/06/12/18Z zonal means (fetch_m2_strat_series.py "
                "archive): u60 at 10 hPa (m/s), cap_T at 10 hPa (K) and cap_z at 100 hPa (m), cap 65-90N "
                "cos-weighted; raw, no gap filling",
                "columns": ["u10", "t10", "z100"],
                "days": rows,
            },
            separators=(",", ":"),
        )
    )
    log(f"wrote {M2_SIDE.relative_to(REPO)}: {len(rows)} of {len(days)} days")
    return 0


def stats(diff: pd.Series, spread: float | None) -> dict:
    """GEOS FP minus MERRA-2 on the overlap days: mean with a 95 % interval (sample size cut for the lag-1
    autocorrelation of the daily differences), sd, RMS, the largest difference and the mean by month; `spread` is the
    chart's median 10-90 % band width, against which the mean is also given as a share."""
    x = diff.dropna()
    r1 = float(x.reindex(pd.date_range(x.index[0], x.index[-1])).autocorr(1))
    n_eff = len(x) * (1 - r1) / (1 + r1) if r1 > 0 else len(x)
    m, sd = float(x.mean()), float(x.std(ddof=1))
    half = 1.96 * sd / np.sqrt(n_eff)
    out = {
        "n": len(x),
        "mean": round(m, 3),
        "ci95": [round(m - half, 3), round(m + half, 3)],
        "sd": round(sd, 3),
        "rms": round(float(np.sqrt((x**2).mean())), 3),
        "max_abs": round(float(x.abs().max()), 3),
        "lag1": round(r1, 2),
        "n_eff": round(n_eff, 1),
        "by_month": {f"{k:02d}": round(float(v), 3) for k, v in x.groupby(x.index.month).mean().items()},
    }
    if spread:
        out["chart_band_p10_p90"] = round(spread, 2)
        out["mean_share_of_band"] = round(abs(m) / spread, 3)
    return out


def overlap(a) -> int:
    """GEOS FP on every MERRA-2 overlap day, then the statistics. The GEOS FP values are kept in the output and saved
    every 10 days, so a run cut short (NCCS reads take ~20-35 s a day) is resumed by the next."""
    j = json.loads(M2_SIDE.read_text())
    m2 = pd.DataFrame.from_dict(j["days"], orient="index", columns=j["columns"])
    m2.index = pd.DatetimeIndex(m2.index)
    got = json.loads(OVERLAP.read_text()).get("geosfp", {}) if OVERLAP.exists() else {}
    todo = [d for d in m2.index if f"{d:%Y-%m-%d}" not in got]
    log(f"GEOS FP minus MERRA-2: {len(got)} of {len(m2)} days held, {len(todo)} to fetch")
    fp = Fp() if todo else None
    for k, d in enumerate(todo):
        r = fp.day(d)
        if r is not None:
            got[f"{d:%Y-%m-%d}"] = [round(v, 3) for v in r]
        if k % 10 == 9 and len(got) > 2:
            log(f"  {k + 1}/{len(todo)} {d:%Y-%m-%d}, {len(got)} held")
            write_overlap(m2, got)
    if len(got) < 30:
        raise SystemExit(f"only {len(got)} overlap days from GEOS FP: too few for the statistics")
    res = write_overlap(m2, got)
    for c in COLS:
        s = res[c]
        share = f", {100 * s['mean_share_of_band']:.1f} % of the chart's band" if c in CHART else ""
        log(
            f"  {c:6s} n {s['n']}: mean {s['mean']:+.3f} [{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}], sd {s['sd']:.3f}, "
            f"RMS {s['rms']:.3f}, r {s['r']:.4f}{share}"
        )
    log(f"wrote {OVERLAP.name} ({len(got)} of {len(m2)} days)")
    return 0


def write_overlap(m2: pd.DataFrame, got: dict) -> dict:
    """The statistics on the days held so far, written with the GEOS FP values themselves."""
    g = pd.DataFrame.from_dict(got, orient="index", columns=["u10", "t10", "z100"])
    g.index = pd.DatetimeIndex(g.index)
    g = g.sort_index()
    m2 = m2.loc[g.index].copy()
    g["z100s"], m2["z100s"] = z100s(g.index, g.z100), z100s(m2.index, m2.z100)
    clim = json.loads(HISTORY.read_text())["vars"]
    band = {c: float(np.median(np.subtract(clim[c]["clim"]["p90"], clim[c]["clim"]["p10"]))) for c in CHART}
    res = {c: stats(g[c] - m2[c], band.get(c)) for c in COLS}
    for c in COLS:
        res[c]["r"] = round(float(np.corrcoef(g[c], m2[c])[0, 1]), 4)
    out = {
        "method": "GEOS FP minus MERRA-2 on the same days (both the daily mean of the 00/06/12/18Z zonal means on the "
        "history's 1 x 2.5 deg sampling); 95 % interval of the mean with the sample size cut for the lag-1 "
        "autocorrelation; z100s on the MERRA-2 fit (drip_std.json); chart_band_p10_p90 = median over the "
        "winter axis of the chart's 10-90 % band (strat_history.json)",
        "period": [f"{g.index[0]:%Y-%m-%d}", f"{g.index[-1]:%Y-%m-%d}"],
        "units": {"u10": "m/s", "t10": "K", "z100": "m", "z100s": "sd"},
        "stats": res,
        "geosfp": dict(sorted(got.items())),
    }
    OVERLAP.write_text(json.dumps(out, indent=1))
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["m2-overlap", "overlap", "update"])
    a = ap.parse_args()
    return {"m2-overlap": m2_overlap, "overlap": overlap, "update": update}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
