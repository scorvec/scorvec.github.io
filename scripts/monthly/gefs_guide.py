#!/usr/bin/env python3
"""GEFS guidance: the dynamical input to Claude's month-ahead forecast, from the whole 35-day GEFS run.

GEFS runs 35 days from 00Z every day, so the run of the 15th reaches the 18th or 19th of the target month. The run
gives two predictors, and the blend (blend.py) weighs each against the statistical forecast per grid point on the
reforecast years:

  target   the forecast mean over every day of the target month the run reaches (days 1 to ~18)
  lead-in  the forecast mean over the run's days before the target month (the rest of the issue month): what the
           run says the month will start from (snow, soil moisture and sea ice laid down then persist)

Both are the same arithmetic in the hindcast and the live forecast: weekly ensemble-mean anomalies (days 1-7 ...
29-35 of the run, the GEFS page's weeks), each weighted by how many of the window's days it holds.

Hindcast: the GEFSv12 reforecast, 2000-2019 (release gefs-clim-v3, gefs_hindw_<MM>.npz: four members, weekly means,
every Wednesday start of month MM). Each year takes the Wednesday start nearest the 15th of the issue month (the
12th-18th, so on average the live lead, and the target window 15-21 days long, 18 on average). Its weekly anomalies
are taken against the other years' starts of the same month (a linear fit in start date per week and grid point:
the seasonal cycle within the month), re-based to 1991-2020 with ERA5 (1991-2020 minus 2001-2020 normals over each
week's valid dates, gefs-ref-v1 era5_shift.npz), and 500 hPa height has its global mean removed: the GEFS page's
anomalies, exactly. Box-averaged onto the outlook's 3° grid and kept as data/gefs_hc_<target MM>.npz (target years: a
January table holds December starts of the year before).

Live: the GEFS page's run archive (gefs_live.py, assets/gefs/anim/runs/gefs_run_<YYYYMMDD>.npz on the frames
branch), the newest run on or before the 15th, 31 members, its anomalies against the same reforecast climatology.

    python scripts/monthly/gefs_guide.py --build [MM,...]     # the reforecast tables (one 290 MB download each)
    python scripts/monthly/gefs_guide.py --live YYYY-MM       # print the live guidance for a target month
"""

from __future__ import annotations

import argparse
import calendar
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
LAST = GR.WEEKS[-1][1]  # 35
MIN_DAYS = 14  # a run must reach at least this many days of the target month
ISSUE_DAY = 15
YEARS = range(2000, 2020)  # reforecast start years
NAME = "GEFS"
COSW = np.cos(np.deg2rad(GR.LAT))[:, None] * np.ones((1, len(GR.LON)))


def issue_month(month: int) -> int:
    return 12 if month == 1 else month - 1


def windows(init: dt.date, first: dt.date) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Weights (5,) of the run's weekly means for the target window (every day of the target month the run reaches)
    and the lead-in (the run's days before the month), and the target window's length. Day d of a run is the UTC
    day init + d - 1. None when the run reaches fewer than MIN_DAYS of the target month."""
    d0 = (first - init).days + 1  # the run's day of the target month's 1st
    n = min(LAST - d0 + 1, calendar.monthrange(first.year, first.month)[1])
    if d0 < 2 or n < MIN_DAYS:
        return None

    def wts(days):
        return np.array([((days >= a) & (days <= b)).sum() for a, b in GR.WEEKS], float) / len(days)

    return wts(np.arange(d0, d0 + n)), wts(np.arange(1, d0)), n


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


def era5_shift(init: dt.date, cache: Path | None = None) -> dict:
    """{tag: (5, lat, lon)} the ERA5 1991-2020 minus 2001-2020 normal averaged over each week's valid days of a run
    from `init` (linear in the common-year day of year between the 5-day centres, as gefs_live.shift_series)."""
    if not _SHIFT:
        z = np.load(io.BytesIO(_get(SHIFT_URL, cache / "era5_shift.npz" if cache else None)))
        _SHIFT.update({k: z[k] for k in z.files if k == "doy" or (k.startswith("s15_") and k[4:] in TAGS.values())})
    doy = _SHIFT["doy"].astype(float)
    x = np.r_[doy - 365.0, doy, doy + 365.0]
    out = {}
    for tag in TAGS.values():
        f = _SHIFT[f"s15_{tag}"]
        ff = np.concatenate([f, f, f]).astype(np.float64)
        daily = []
        for d in range(LAST):
            t = GR.noleap_doy(init + dt.timedelta(days=d))
            j = int(np.searchsorted(x, t))
            w = (t - x[j - 1]) / (x[j] - x[j - 1])
            daily.append((1 - w) * ff[j - 1] + w * ff[j])
        out[tag] = np.stack([np.mean(daily[a - 1 : b], 0) for a, b in GR.WEEKS])
    return out


def finish(a: np.ndarray, tag: str, coarse: G.Coarsener) -> np.ndarray:
    """(5, lat, lon) weekly anomalies on the GEFS 1.5° grid -> (5, P) on the outlook's 0-90N 3° cells."""
    if tag in DEMEAN:
        a = a - np.einsum("wij,ij->w", a, COSW)[:, None, None] / COSW.sum()
    return coarse(a, "nh").reshape(len(a), -1).astype(np.float32)


def predictors(wk: np.ndarray, wt: np.ndarray, wl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(target, lead-in) means from (5, P) weekly anomalies."""
    return wt @ wk, wl @ wk


# ------------------------------------------------------------------------------------------------- the hindcast
def build(month: int, cache: Path, weekly: Path | None = None) -> Path:
    """data/gefs_hc_<MM>.npz for target month `month`: years (target years), init (start dates), ndays (target
    window length), and per outlook variable v the (n, P) target and v_lead lead-in anomalies. `weekly`: also write
    the per-year weekly anomalies there (for testing other windows)."""
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
        sel = {}
        for y in YEARS:
            cand = [i for i in range(len(starts)) if yr[i] == y]
            if not cand:
                continue
            i = min(cand, key=lambda i: (abs(x[i]), x[i]))
            ty = y + 1 if month == 1 else y
            win = windows(dates[i], dt.date(ty, month, 1))
            if win is not None:
                sel[y] = (i, ty, win)
        wks = {v: [] for v in TAGS}
        for v, tag in TAGS.items():
            E = np.stack([GR.unpack(tag, z[f"{s}_w_{tag}"]).mean(0) for s in starts]).astype(np.float32)  # (S,5,...)
            S = E.reshape(len(starts), -1)
            for y, (i, _, _) in sel.items():
                keep = yr != y
                X = np.stack([np.ones(keep.sum()), x[keep]], 1)
                coef = np.linalg.solve(X.T @ X, X.T @ S[keep])  # (2, 5*lat*lon): the other years' climatology
                clim = (coef[0] + coef[1] * x[i]).reshape(E.shape[1:])
                wks[v].append(finish(E[i] - clim - era5_shift(dates[i], cache)[tag], tag, coarse))
            del E, S
    out_arrs = {
        "years": np.array([ty for _, ty, _ in sel.values()]),
        "init": np.array([starts[i] for i, _, _ in sel.values()]),
        "ndays": np.array([win[2] for _, _, win in sel.values()]),
    }
    for v in TAGS:
        tl = [predictors(wk, win[0], win[1]) for wk, (_, _, win) in zip(wks[v], sel.values(), strict=True)]
        out_arrs[v] = np.stack([t for t, _ in tl])
        out_arrs[f"{v}_lead"] = np.stack([lead for _, lead in tl])
    out = DATA / f"gefs_hc_{month:02d}.npz"
    np.savez_compressed(out, **out_arrs)
    if weekly is not None:
        np.savez_compressed(
            weekly / f"gefs_wk_{month:02d}.npz", **out_arrs, **{f"{v}_wk": np.stack(w) for v, w in wks.items()}
        )
    nd = out_arrs["ndays"]
    print(
        f"{out.name}: {len(nd)} years ({out_arrs['years'][0]}-{out_arrs['years'][-1]}), target days 1-{nd.min()} to "
        f"1-{nd.max()} (mean {nd.mean():.1f})",
        flush=True,
    )
    return out


def hindcast(month: int, v: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """(target years, (n, P) target-window anomalies, (n, P) lead-in anomalies) of the reforecast table, or None."""
    p = DATA / f"gefs_hc_{month:02d}.npz"
    if not p.exists():
        return None
    with np.load(p) as f:
        if v not in f.files:
            return None
        return f["years"], f[v].astype(np.float64), f[f"{v}_lead"].astype(np.float64)


# ------------------------------------------------------------------------------------------------- the live run
def live(year: int, month: int, issue: dt.date | None = None, cache: Path | None = None) -> dict | None:
    """The newest GEFS run on or before the issue date (default the 15th of the month before), reduced like the
    hindcast: {"init": "YYYY-MM-DD", "ndays": n, v: (P,) target window, v_lead: (P,) lead-in}. None when no run of
    the last four days is in the archive."""
    first = dt.date(year, month, 1)
    if issue is None:
        iy = year - 1 if month == 1 else year
        issue = dt.date(iy, issue_month(month), ISSUE_DAY)
    for back in range(4):
        d = issue - dt.timedelta(days=back)
        win = windows(d, first)
        if win is None:
            continue
        try:
            raw = _get(RUN_URL.format(d=f"{d:%Y%m%d}"), cache / f"gefs_run_{d:%Y%m%d}.npz" if cache else None)
        except Exception:  # noqa: BLE001  not run (or already pruned from the week-long archive)
            continue
        z = np.load(io.BytesIO(raw))
        coarse = G.Coarsener(GR.LAT, GR.LON)
        out = {"init": d.isoformat(), "ndays": win[2]}
        for v, tag in TAGS.items():
            a = z[f"A_{tag}"].astype(np.float64)  # (35, lat, lon) daily anomalies vs 1991-2020, z500 demeaned
            wk = np.stack([a[w0 - 1 : w1].mean(0) for w0, w1 in GR.WEEKS])
            out[v], out[f"{v}_lead"] = predictors(finish(wk, tag, coarse), win[0], win[1])
        return out
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", nargs="?", const="1,2,3,4,5,6,7,8,9,10,11,12", help="target months to tabulate")
    ap.add_argument("--weekly", help="also write the per-year weekly anomalies to this directory")
    ap.add_argument("--live", help="YYYY-MM: print the live guidance")
    ap.add_argument("--cache", default=str(Path(tempfile.gettempdir()) / "gefs_guide"))
    a = ap.parse_args()
    cache = Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)
    if a.build:
        for m in map(int, a.build.split(",")):
            build(m, cache, Path(a.weekly) if a.weekly else None)
    if a.live:
        y, m = map(int, a.live.split("-"))
        g = live(y, m, cache=cache)
        if g is None:
            print("no GEFS run in the archive for this issue")
            return 1
        w = np.cos(np.deg2rad(G.LAT_NH))[:, None] * np.ones(len(G.LON))
        w[G.LAT_NH < 20] = 0
        print(f"GEFS {g['init']} 00Z: target days 1-{g['ndays']} of {y}-{m:02d}, lead-in the run's days before")
        for v in TAGS:
            for k in (v, f"{v}_lead"):
                f = g[k].reshape(w.shape)
                print(f"  {k}: 20-90N mean {(w * f).sum() / w.sum():+.2f}, range {f.min():+.1f} .. {f.max():+.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
