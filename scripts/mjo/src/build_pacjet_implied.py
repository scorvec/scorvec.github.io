#!/usr/bin/env python3
"""What the forecast jet implies for North America: regression maps of weekly 2 m temperature, precipitation and
500 hPa height on the two North Pacific jet indices, with the skill they had out of sample.

Predictor  the two jet indices (pacjet_ref.nc PCs, in sigma) averaged over five days.
Target     the 7-day mean anomaly centred on the same day (the forecast's days 9-15 when the predictor is its days
           10-14) and 7 days later (days 16-22, the week after the forecast ends). ERA5 1991-2020, November to March;
           anomalies as in build_pacjet_impacts.py (day-of-year normal, linear trend out).
Skill      each season is predicted from a fit on the other seasons; r_cv is the correlation of those predictions
           with what happened. It is tested by resampling whole seasons (N_BOOT draws; p = the share of draws with
           r_cv <= 0), Benjamini-Hochberg at ALPHA over the map. A point is kept only where that passes AND r_cv is at
           least R_MIN; elsewhere the slopes are NaN and the live figure stays blank.

The slopes are fitted on analysed jet indices. A forecast of them at days 10-14 is an ensemble mean, which is smoother
and smaller than what verifies, so the implied anomaly errs towards zero.

    python src/build_pacjet_implied.py   -> scripts/mjo/data/reference/pacjet_implied.{npz,json}
    render(...) is called by pacjet.py with each run's forecast indices.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from quantmet.stats import benjamini_hochberg
from scipy.ndimage import uniform_filter1d

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "lib")]
import build_pacjet_impacts as BI

NPZ = BI.REF / "pacjet_implied.npz"
OUT = BI.REF / "pacjet_implied.json"
SEASON = (11, 12, 1, 2, 3)
PRED_DAYS = (10, 14)  # the forecast days whose mean jet indices are the predictor
WEEKS = {"wk2": 0, "wk3": 7}  # target week: days from the predictor's centre to the week's centre
ALPHA = 0.10
R_MIN = 0.15
N_BOOT = 1000
PANELS = (("t2m", "wk2"), ("prcp", "wk2"), ("t2m", "wk3"))  # drawn; 500 hPa height is contoured on the t2m panels
T_LEV = np.array([-4, -3, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 3, 4])
P_LEV = np.array([-3, -2, -1.5, -1, -0.5, -0.25, 0.25, 0.5, 1, 1.5, 2, 3])
Z_LEV = [x for x in range(-150, 151, 20) if x]


def skill(x: np.ndarray, y: np.ndarray, season: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(slopes (2, point), r_cv (point), p (point)) of y (day, point) on x (day, 2) with whole seasons left out."""
    X = np.c_[x, np.ones(len(x))]
    seasons = np.unique(season)
    pred = np.zeros_like(y)
    for s in seasons:
        te = season == s
        pred[te] = X[te] @ np.linalg.lstsq(X[~te], y[~te], rcond=None)[0]

    # per-season sums give the correlation of any resample of seasons without touching the days again
    def by_season(q):
        return np.stack([q[season == k].sum(0) for k in seasons])

    days = np.array([(season == k).sum() for k in seasons], float)
    sp, sy, spp, syy, spy = (by_season(q) for q in (pred, y, pred**2, y**2, pred * y))

    def corr(w):
        n = w @ days
        a, b, aa, bb, ab = (w @ q for q in (sp, sy, spp, syy, spy))
        return (ab - a * b / n) / np.sqrt((aa - a * a / n) * (bb - b * b / n))

    r = corr(np.ones(len(seasons)))
    draws = rng.integers(0, len(seasons), (N_BOOT, len(seasons)))
    boot = np.stack([corr(np.bincount(d, minlength=len(seasons)).astype(float)) for d in draws])
    p = (1 + (boot <= 0).sum(0)) / (1 + N_BOOT)
    return np.linalg.lstsq(X, y, rcond=None)[0][:2], r, p


def main() -> int:
    with xr.open_dataset(BI.REF / "pacjet_ref.nc") as ds:
        ref = ds.load()
    rt = pd.DatetimeIndex(ref.time.values)
    n = len(rt)
    x5 = uniform_filter1d(ref.pc.values.astype(float), PRED_DAYS[1] - PRED_DAYS[0] + 1, axis=0, mode="nearest")
    cold = np.isin(rt.month, SEASON)
    season = np.where(rt.month >= 7, rt.year, rt.year - 1)
    rng = np.random.default_rng(0)
    out, doc = {}, {"period": [BI.Y0, BI.Y1], "months": list(SEASON), "alpha_fdr": ALPHA, "r_min": R_MIN, "skill": {}}
    i = np.arange(n)
    for var in ("t2m", "prcp", "z500"):
        t, lat, lon, v = BI.load(var)
        if not (t == rt).all():
            raise SystemExit(f"{var}: the days differ from the jet reference's")
        w7 = uniform_filter1d(BI.anomalies(t, v, cold), 7, axis=0, mode="nearest")
        out[f"lat_{var}"], out[f"lon_{var}"] = lat, lon
        for wk, off in WEEKS.items():
            idx = np.flatnonzero(cold & (i >= 3) & (i + off + 3 < n))
            idx = idx[np.isin(rt[idx + off].month, SEASON)]
            b, r, p = skill(x5[idx], w7[idx + off].reshape(len(idx), -1).astype(float), season[idx], rng)
            keep = benjamini_hochberg(p, ALPHA) & (r >= R_MIN)
            doc["skill"][f"{var}_{wk}"] = {
                "days": len(idx),
                "r_cv_median": round(float(np.median(r)), 2),
                "r_cv_max": round(float(r.max()), 2),
                "share_kept": round(float(keep.mean()), 3),
            }
            print(f"  {var:5s} {wk}: {doc['skill'][f'{var}_{wk}']}", flush=True)
            if var == "z500" or (var, wk) in PANELS:
                out[f"{var}_{wk}"] = np.where(keep, b, np.nan).reshape(2, *v.shape[1:]).astype("float32")
    np.savez_compressed(NPZ, **out)
    OUT.write_text(json.dumps(doc, indent=1))
    print(f"wrote {NPZ} ({NPZ.stat().st_size / 1e3:.0f} kB) and {OUT.name}", flush=True)
    return 0


def render(label: str, init: pd.Timestamp, pcs: np.ndarray, month: int, out: Path) -> dict:
    """Draw what the ensemble-mean jet indices of forecast days PRED_DAYS imply. label: the ensemble's name;
    pcs: (member, day from 0, 2); month: the month the indices are read on, outside SEASON a note is drawn.
    Returns the predictor and the weeks' dates for the page."""
    if month not in SEASON:
        import pacjet_watch as PW

        PW.note(
            out,
            f"What the {label} jet implies — not drawn for {pd.Timestamp(2001, month, 1):%B} (init {init:%Y-%m-%d %HZ})",
            "The link between the jet and the weeks that follow over North America is fitted on ERA5's November to "
            "March days, and only there was it tested out of sample. The maps return with forecasts centred in those "
            "months.",
        )
        return {"in_season": False}
    import matplotlib

    matplotlib.use("Agg")
    import house

    house.apply()
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm

    plt.rcParams.update({"font.size": 10.5, "font.family": "DejaVu Sans"})
    Z = np.load(NPZ)
    x = pcs[:, PRED_DAYS[0] : PRED_DAYS[1] + 1].mean((0, 1))
    mid = init + pd.Timedelta(days=sum(PRED_DAYS) // 2)
    weeks = {k: (mid + pd.Timedelta(days=off - 3), mid + pd.Timedelta(days=off + 3)) for k, off in WEEKS.items()}
    proj, pc_ = ccrs.LambertConformal(central_longitude=-125, standard_parallels=(35, 60)), ccrs.PlateCarree()
    ext = [BI.DOM["lon"][0], BI.DOM["lon"][1], BI.DOM["lat"][0] + 3, BI.DOM["lat"][1] - 3]
    lon, lat = np.linspace(ext[0], ext[1], 60), np.linspace(ext[2], ext[3], 60)
    edge = proj.transform_points(
        pc_, np.r_[lon, lon, np.full(60, ext[0]), np.full(60, ext[1])], np.r_[np.full(60, ext[2]), np.full(60, ext[3]), lat, lat]
    )  # fmt: skip
    mw = 4.3
    mh = mw * np.ptp(edge[:, 1]) / np.ptp(edge[:, 0])
    gap, left, right, top, bot = 0.14, 0.2, 0.2, 1.3, 1.25
    W, H = left + 3 * mw + 2 * gap + right, top + mh + bot
    fig = plt.figure(figsize=(W, H))
    spec = {"t2m": (T_LEV, "RdBu_r", "2 m temperature (°C)"), "prcp": (P_LEV, "BrBG", "precipitation (mm/day)")}
    shade = {}
    for k, (var, wk) in enumerate(PANELS):
        ax = fig.add_axes([(left + k * (mw + gap)) / W, bot / H, mw / W, mh / H], projection=proj)
        ax.set_extent(ext, crs=pc_)
        lev, cmap, name = spec[var]
        f = np.tensordot(x, Z[f"{var}_{wk}"], 1)
        ax.add_feature(cfeature.LAND, facecolor="#f3f1ec", zorder=0)
        shade[var] = ax.contourf(
            Z[f"lon_{var}"], Z[f"lat_{var}"], f, lev, cmap=cmap, norm=BoundaryNorm(lev, 256, extend="both"),
            extend="both", transform=pc_, zorder=1,
        )  # fmt: skip
        if var == "t2m":
            zc = ax.contour(
                Z["lon_z500"], Z["lat_z500"], np.tensordot(x, Z[f"z500_{wk}"], 1), Z_LEV, colors=house.INK,
                linewidths=0.9, transform=pc_, zorder=3,
            )  # fmt: skip
            ax.clabel(zc, zc.levels[::2], fmt="%d", fontsize=7.5, inline_spacing=2)
        ax.coastlines(linewidth=0.55, color="#55524c", zorder=2)
        ax.add_feature(cfeature.BORDERS, linewidth=0.35, edgecolor="#55524c", zorder=2)
        ax.add_feature(cfeature.STATES, linewidth=0.2, edgecolor="#77736c", zorder=2)
        a, b = weeks[wk]
        ax.set_title(f"{name.split(' (')[0].capitalize()}, {a:%-d %b}–{b:%-d %b}", fontsize=11, loc="left", pad=4)
        if not np.isfinite(f).any():
            ax.text(0.5, 0.5, "no skilful signal", transform=ax.transAxes, ha="center", color=house.MUTED, fontsize=10)
    fig.text(left / W, 1 - 0.36 / H, f"What the {label} jet implies", fontsize=15, weight="bold", ha="left")
    state = ", ".join(
        f"{name} {v:+.1f} σ".replace("-", "−") for name, v in zip(("extension index", "poleward shift index"), x)
    )
    fig.text(
        left / W, 1 - 0.66 / H,
        f"Init {init:%Y-%m-%d %HZ}. Ensemble-mean jet, days {PRED_DAYS[0]}–{PRED_DAYS[1]}: {state}",
        fontsize=10.5, color=house.MUTED, ha="left",
    )  # fmt: skip
    for x0, var in ((left, "t2m"), (left + mw + gap, "prcp")):
        cax = fig.add_axes([(x0 + 0.5) / W, 0.78 / H, (mw - 1.0) / W, 0.13 / H])
        cb = fig.colorbar(shade[var], cax=cax, orientation="horizontal", ticks=spec[var][0])
        cb.ax.tick_params(labelsize=8)
        house.spaced(cb.ax.xaxis)
        cb.set_label(spec[var][2], fontsize=8.8)
    note = (
        f"From ERA5 {BI.Y0}–{BI.Y1}, November to March: the week's anomaly that went with a jet like this one. Colour "
        f"only where that link held out of sample (seasons left out, false-discovery rate {ALPHA:.0%}, correlation at "
        f"least {R_MIN}). Lines: 500 hPa height (every 20 m, dashed = lower). The last week lies beyond the forecast."
    )
    fig.text(
        (left + 2 * (mw + gap)) / W, 0.98 / H, textwrap.fill(note, 64), fontsize=8.2, color=house.MUTED, ha="left", va="top"
    )  # fmt: skip
    fig.savefig(out, dpi=105, facecolor="white", pil_kwargs={"quality": 88, "method": 6})
    plt.close(fig)
    return {
        "in_season": True,
        "pc": np.round(x, 2).tolist(),
        "days": list(PRED_DAYS),
        "weeks": {k: [f"{a:%Y-%m-%d}", f"{b:%Y-%m-%d}"] for k, (a, b) in weeks.items()},
    }


if __name__ == "__main__":
    sys.exit(main())
