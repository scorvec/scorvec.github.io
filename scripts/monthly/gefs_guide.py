#!/usr/bin/env python3
"""GEFS guidance for the first half of the target month: the dynamical input to Claude's month-ahead forecast.

GEFS runs 35 days from 00Z every day, so the run on the 15th reaches about the 18th of the target month. What it
says about the first 15 days of the target month is the predictor; how much of that carries to the whole month is
left to the blend (blend.py), which weighs it against the statistical forecast per grid point on the reforecast years.

The predictor in both the hindcast and the live forecast is the same arithmetic: weekly ensemble-mean anomalies
(days 1-7 ... 29-35 of the run, the GEFS page's weeks), each weighted by how many of target days 1-15 it holds.

Hindcast: the GEFSv12 reforecast, 2000-2019 (release gefs-clim-v3, gefs_hindw_<MM>.npz: four members, weekly means,
every Wednesday start of month MM). Each year takes the Wednesday start nearest the 15th of the issue month (the
12th-18th, so on average the live lead). Its weekly anomalies are taken against the other years' starts of the same
month (a linear fit in start date per week and grid point: the seasonal cycle within the month), re-based to
1991-2020 with ERA5 (1991-2020 minus 2001-2020 normals at the valid dates, gefs-ref-v1 era5_shift.npz), and 500 hPa
height has its global mean removed: the GEFS page's anomalies, exactly. Box-averaged onto the outlook's 3° grid and
kept as data/gefs_hc_<target MM>.npz (target years: a January table holds December starts of the year before).

Live: the GEFS page's run archive (gefs_live.py, assets/gefs/anim/runs/gefs_run_<YYYYMMDD>.npz on the frames
branch), the newest run on or before the 15th, 31 members, its anomalies against the same reforecast climatology.

    python scripts/monthly/gefs_guide.py --build [MM,...]     # the reforecast tables (one 290 MB download each)
    python scripts/monthly/gefs_guide.py --live YYYY-MM       # print the live guidance for a target month
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "gefs"))
import gefs_reforecast as GR
import mo_grid as G

DATA = HERE / "data"
HINDW_URL = GR.CLIM_URL.format(name="gefs_hindw_{mm:02d}.npz")
SHIFT_URL = "https://github.com/scorvec/scorvec.github.io/releases/download/gefs-ref-v1/era5_shift.npz"
RUN_URL = "https://raw.githubusercontent.com/scorvec/scorvec.github.io/frames/assets/gefs/anim/runs/gefs_run_{d}.npz"
TAGS = {"z500": "zg500", "t2m": "t2m", "prcp": "pr"}  # outlook variable -> GEFS field
DEMEAN = ("zg500",)  # global mean removed (the GEFS page's rule)
NDAYS = 15  # target days 1..NDAYS
ISSUE_DAY = 15
YEARS = range(2000, 2020)  # reforecast start years
NAME = "GEFS"
COSW = np.cos(np.deg2rad(GR.LAT))[:, None] * np.ones((1, len(GR.LON)))


def issue_month(month: int) -> int:
    return 12 if month == 1 else month - 1


def week_weights(init: dt.date, first: dt.date, ndays: int = NDAYS) -> np.ndarray | None:
    """(5,) weights of the run's weekly means: the share of target days 1..ndays in each week. Day d of a run is
    the UTC day init + d - 1. None when the run does not reach day ndays of the target month."""
    d0 = (first - init).days + 1
    days = np.arange(d0, d0 + ndays)
    if days[0] < 1 or days[-1] > GR.WEEKS[-1][1]:
        return None
    w = np.array([((days >= a) & (days <= b)).sum() for a, b in GR.WEEKS], float)
    return w / ndays


_SHIFT: dict = {}


def _get(url: str, path: Path | None = None) -> bytes:
    if path is not None and path.exists():
        return path.read_bytes()
    with urllib.request.urlopen(url, timeout=300) as r:
        b = r.read()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b)
    return b


def era5_shift(first: dt.date, cache: Path | None = None) -> dict:
    """{tag: (lat, lon)} the ERA5 1991-2020 minus 2001-2020 normal averaged over target days 1..NDAYS (linear in the
    common-year day of year between the 5-day centres, as gefs_live.shift_series)."""
    if not _SHIFT:
        z = np.load(io.BytesIO(_get(SHIFT_URL, cache / "era5_shift.npz" if cache else None)))
        _SHIFT.update({k: z[k] for k in z.files if k == "doy" or (k.startswith("s15_") and k[4:] in TAGS.values())})
    doy = _SHIFT["doy"].astype(float)
    x = np.r_[doy - 365.0, doy, doy + 365.0]
    out = {}
    for tag in TAGS.values():
        f = _SHIFT[f"s15_{tag}"]
        ff = np.concatenate([f, f, f]).astype(np.float64)
        acc = 0.0
        for k in range(NDAYS):
            t = GR.noleap_doy(first + dt.timedelta(days=k))
            j = int(np.searchsorted(x, t))
            w = (t - x[j - 1]) / (x[j] - x[j - 1])
            acc = acc + (1 - w) * ff[j - 1] + w * ff[j]
        out[tag] = acc / NDAYS
    return out


def finish(a: np.ndarray, tag: str, coarse: G.Coarsener) -> np.ndarray:
    """A (lat, lon) anomaly on the GEFS 1.5° grid -> the outlook's (P,) 0-90N 3° cells."""
    if tag in DEMEAN:
        a = a - (a * COSW).sum() / COSW.sum()
    return coarse(a, "nh").ravel().astype(np.float32)


# ------------------------------------------------------------------------------------------------- the hindcast
def build(month: int, cache: Path) -> Path:
    """data/gefs_hc_<MM>.npz for target month `month`: years (target years), init (start dates), and per outlook
    variable an (n, P) table of first-half anomalies."""
    mi = issue_month(month)
    url = HINDW_URL.format(mm=mi)
    with tempfile.TemporaryDirectory(dir=cache) as tmp:
        p = Path(tmp) / "hindw.npz"
        print(f"downloading {url}", flush=True)
        p.write_bytes(_get(url))
        z = np.load(p)
        starts = sorted({k.split("_")[0] for k in z.files if k.endswith("_members")})
        dates = [dt.date(int(s[:4]), int(s[4:6]), int(s[6:])) for s in starts]
        yr = np.array([d.year for d in dates])
        x = np.array([(d - dt.date(d.year, mi, ISSUE_DAY)).days for d in dates], float)  # days from the 15th
        coarse = G.Coarsener(GR.LAT, GR.LON)
        years, inits, rows = [], [], {v: [] for v in TAGS}
        sel = {}
        for y in YEARS:
            cand = [i for i in range(len(starts)) if yr[i] == y]
            if not cand:
                continue
            i = min(cand, key=lambda i: (abs(x[i]), x[i]))
            ty = y + 1 if month == 1 else y
            ww = week_weights(dates[i], dt.date(ty, month, 1))
            if ww is None:
                continue
            sel[y] = (i, ww, ty)
        for v, tag in TAGS.items():
            E = np.stack([GR.unpack(tag, z[f"{s}_w_{tag}"]).mean(0) for s in starts]).astype(np.float32)  # (S,5,...)
            S = E.reshape(len(starts), -1)
            for y, (i, ww, ty) in sel.items():
                keep = yr != y
                X = np.stack([np.ones(keep.sum()), x[keep]], 1)
                coef = np.linalg.solve(X.T @ X, X.T @ S[keep])  # (2, 5*lat*lon): the other years' climatology
                clim = (coef[0] + coef[1] * x[i]).reshape(E.shape[1:])
                anom = np.einsum("w,wij->ij", ww, E[i] - clim)
                sh = era5_shift(dt.date(ty, month, 1), cache)[tag]
                rows[v].append(finish(anom - sh, tag, coarse))
            del E, S
        for i, _, ty in sel.values():
            years.append(ty)
            inits.append(starts[i])
    out = DATA / f"gefs_hc_{month:02d}.npz"
    np.savez_compressed(
        out,
        years=np.array(years),
        init=np.array(inits),
        **{v: np.stack(r) for v, r in rows.items()},
    )
    print(f"{out.name}: {len(years)} years ({years[0]}-{years[-1]}), starts {inits[0]} ... {inits[-1]}", flush=True)
    return out


def hindcast(month: int, v: str) -> tuple[np.ndarray, np.ndarray] | None:
    """(target years, (n, P) anomalies) of the reforecast table, or None."""
    p = DATA / f"gefs_hc_{month:02d}.npz"
    if not p.exists():
        return None
    with np.load(p) as f:
        if v not in f.files:
            return None
        return f["years"], f[v].astype(np.float64)


# ------------------------------------------------------------------------------------------------- the live run
def live(year: int, month: int, issue: dt.date | None = None, cache: Path | None = None) -> dict | None:
    """The newest GEFS run on or before the issue date (default the 15th of the month before), reduced like the
    hindcast: {"init": "YYYY-MM-DD", "members": n, v: (P,)}. None when no run of the last four days is in the
    archive or the run does not reach day NDAYS of the target month."""
    first = dt.date(year, month, 1)
    if issue is None:
        iy = year - 1 if month == 1 else year
        issue = dt.date(iy, issue_month(month), ISSUE_DAY)
    for back in range(4):
        d = issue - dt.timedelta(days=back)
        ww = week_weights(d, first)
        if ww is None:
            continue
        try:
            raw = _get(RUN_URL.format(d=f"{d:%Y%m%d}"), cache / f"gefs_run_{d:%Y%m%d}.npz" if cache else None)
        except Exception:  # noqa: BLE001  not run (or already pruned from the week-long archive)
            continue
        z = np.load(io.BytesIO(raw))
        coarse = G.Coarsener(GR.LAT, GR.LON)
        out = {"init": d.isoformat()}
        for v, tag in TAGS.items():
            a = z[f"A_{tag}"].astype(np.float64)  # (35, lat, lon) daily anomalies vs 1991-2020, z500 demeaned
            wk = np.stack([a[w0 - 1 : w1].mean(0) for w0, w1 in GR.WEEKS])
            out[v] = finish(np.einsum("w,wij->ij", ww, wk), tag, coarse)
        return out
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", nargs="?", const="1,2,3,4,5,6,7,8,9,10,11,12", help="target months to tabulate")
    ap.add_argument("--live", help="YYYY-MM: print the live guidance")
    ap.add_argument("--cache", default=str(Path(tempfile.gettempdir()) / "gefs_guide"))
    a = ap.parse_args()
    cache = Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)
    if a.build:
        for m in map(int, a.build.split(",")):
            build(m, cache)
    if a.live:
        y, m = map(int, a.live.split("-"))
        g = live(y, m, cache=cache)
        if g is None:
            print("no GEFS run in the archive for this issue")
            return 1
        w = np.cos(np.deg2rad(G.LAT_NH))[:, None] * np.ones(len(G.LON))
        print(f"GEFS {g['init']} 00Z, target days 1-{NDAYS} of {y}-{m:02d}")
        for v in TAGS:
            f = g[v].reshape(w.shape)
            print(
                f"  {v}: 20-90N mean {(w * f)[G.LAT_NH >= 20].sum() / w[G.LAT_NH >= 20].sum():+.2f}, "
                f"range {np.nanmin(f):+.1f} .. {np.nanmax(f):+.1f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
