#!/usr/bin/env python3
"""What each North Pacific jet phase brings - static reference figures (laptop, once; ERA5 from the local store).

For each of the four jet phases (Winters et al. 2019; pacjet_core.phase_of on the season-appropriate ERA5 PCs of
build_pacjet_ref.py), in the months where the phases are defined: composite anomalies over the North Pacific and North
America of 500 hPa height, 2 m temperature and precipitation (ERA5 daily means at 1.5 deg, 1991-2020: z500 and prcp from
~/era5_store/wb2_1p5_daily, t2m from wb2_1p5_daily_global; units checked by value, never by attribute). Anomaly = the
day minus the 1991-2020 day-of-year mean (31-day circular smoothing) minus a linear trend fitted per grid point over the
in-season days, so warming does not masquerade as a phase signal.

Two timings: the SAME days the jet is in the phase, and 3-7 DAYS LATER (the downstream response over North America).

SIGNIFICANCE (site rule). Phase days come in episodes, so the test is on episodes: each run of consecutive phase days is
one value (the mean over its days, or over its days shifted by 3-7), a two-sided one-sample t-test across episodes at
each grid point, and the Benjamini-Hochberg false discovery rate held at 10% over the map (Wilks 2016). Colour only where
it passes; "no significant signal" where nothing does.

    python src/build_pacjet_impacts.py            (laptop) -> data/reference/pacjet_impacts.{npz,json}
    python src/build_pacjet_impacts.py --render   -> assets/sst/pacjet_impacts_{lag0,lag3_7}.webp (pacjet.py calls this
                                                     in Actions: the laptop does not publish rendered figures)
"""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from scipy.ndimage import uniform_filter1d

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
import pacjet_core as PC                                                    # noqa: E402

REF = HERE.parent / "data" / "reference"
SITE = REPO / "assets" / "sst"
STORE = Path.home() / "era5_store"
Y0, Y1 = 1991, 2020
ALPHA = 0.10
DOM = dict(lat=(15.0, 75.0), lon=(160.0, 310.0))                           # 160E-50W
Z_LEV = np.arange(-120, 121, 15)
T_LEV = np.arange(-3.5, 3.51, 0.5)
P_LEV = np.array([-3, -2, -1.5, -1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1, 1.5, 2, 3])
INK, MUTED = "#1a1a1a", "#6f6b64"
PH_COL = {"extension": "#c0392b", "poleward": "#2e86c1", "retraction": "#7d5ba6", "equatorward": "#d68910"}


def load(var):
    layer = "wb2_1p5_daily_global" if var == "t2m" else "wb2_1p5_daily"
    parts, tt = [], []
    for y in range(Y0, Y1 + 1):
        with xr.open_dataset(STORE / layer / var / f"{var}_{y}.nc") as d:
            a = d[var].transpose("time", "latitude", "longitude")
            a = a.assign_coords(longitude=np.round(a.longitude.values % 360, 3), latitude=np.round(a.latitude.values, 3))
            a = a.sortby("latitude").sel(latitude=slice(DOM["lat"][0] - 3, DOM["lat"][1] + 3),
                                         longitude=slice(DOM["lon"][0] - 3, DOM["lon"][1] + 3))
            parts.append(a.values.astype("float32")); tt += list(pd.DatetimeIndex(a.time.values))
            lat, lon = a.latitude.values, a.longitude.values
    v = np.concatenate(parts)
    m = float(np.nanmean(v))
    if var == "t2m":
        assert 240 < m < 310, f"t2m mean {m}: expected K"
        v = v - 273.15
    elif var == "z500":
        assert 5000 < m < 6000, f"z500 mean {m}: expected metres"
    else:
        assert 0.3 < m < 10, f"prcp mean {m}: expected mm/day"
    return pd.DatetimeIndex(tt), lat, lon, v


def anomalies(t, v, season):
    dy = np.minimum(t.dayofyear.values, 365) - 1
    c = np.zeros((365,) + v.shape[1:]); n = np.zeros(365)
    np.add.at(c, dy, v); np.add.at(n, dy, 1)
    c = uniform_filter1d(c / n[:, None, None], 31, axis=0, mode="wrap")
    a = v - c[dy]
    yr = (t.year + (t.month >= 8)).values.astype(float)                    # a cold season is one winter
    for m in (season, ~season):
        x = yr[m] - yr[m].mean()
        b = np.tensordot(x, a[m], 1) / (x ** 2).sum()
        a[m] -= (x[:, None, None] * b[None]).astype("float32")
    return a


def episodes(mask):
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    return np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1)


def fdr(p):
    q = np.sort(p[np.isfinite(p)])
    if q.size == 0:
        return np.zeros_like(p, bool)
    k = np.flatnonzero(q <= ALPHA * np.arange(1, q.size + 1) / q.size)
    return np.isfinite(p) & (p <= (q[k.max()] if k.size else -1.0))


TAGS = (("lag0", (0,), "on the days the jet is in the phase"), ("lag3_7", (3, 4, 5, 6, 7), "3–7 days after the jet is in the phase"))
ROWS = [("z500", "500 hPa height (m)", Z_LEV, "RdBu_r"), ("t2m", "2 m temperature (°C)", T_LEV, "RdBu_r"),
        ("prcp", "precipitation (mm/day)", P_LEV, "BrBG")]
NPZ = REF / "pacjet_impacts.npz"


def main() -> int:
    """Compute the composites and their FDR masks (laptop) -> data/reference/pacjet_impacts.{npz,json}."""
    ref = xr.open_dataset(REF / "pacjet_ref.nc").load()
    rt = pd.DatetimeIndex(ref.time.values)
    ph = pd.Series(ref.phase.values.astype(int), index=rt)
    ins_m = [m for m in range(1, 13) if bool(ref.in_season.sel(month=m))]
    out, stats_out = {}, {"test": "episode-mean t-test, BH FDR 0.10 over the map", "months": ins_m, "phases": {}}
    for var in ("z500", "t2m", "prcp"):
        t, lat, lon, v = load(var)
        a = anomalies(t, v, np.isin(t.month, ins_m))
        out[f"lat_{var}"] = lat; out[f"lon_{var}"] = lon
        phd = ph.reindex(t).fillna(-9).values.astype(int)
        inside = ((lat >= DOM["lat"][0]) & (lat <= DOM["lat"][1]))[:, None] & ((lon >= DOM["lon"][0]) & (lon <= DOM["lon"][1]))[None]
        for tag, lags, _ in TAGS:
            for k, name in enumerate(PC.PHASES):
                mask = (phd == k) & np.isin(t.month, ins_m)
                eps = episodes(mask)
                E = []
                for e in eps:
                    ii = np.unique(np.concatenate([e + L for L in lags]))
                    if ii.max() >= len(t):                             # an episode at the end of the record: no response days
                        continue
                    E.append(a[ii].mean(0))
                E = np.stack(E)
                mean = E.mean(0)
                _, pv = stats.ttest_1samp(E, 0.0, axis=0)
                ok = fdr(np.where(inside, pv, np.nan))
                out[f"{tag}_{name}_{var}"] = np.where(ok, mean, np.nan).astype("float32")
                stats_out["phases"].setdefault(tag, {}).setdefault(name, {})[var] = {
                    "episodes": len(E), "days": int(mask.sum()), "frac_significant": round(float(ok[inside].mean()), 3),
                    "max_abs_significant": (round(float(np.abs(mean[ok]).max()), 2) if ok.any() else None)}
        print(f"  {var} {v.shape}", flush=True)
    np.savez_compressed(NPZ, **out)
    (REF / "pacjet_impacts.json").write_text(json.dumps(stats_out, indent=1))
    for tag, d in stats_out["phases"].items():
        for name, vv in d.items():
            print(f"  {tag} {name}: " + "; ".join(f"{k} {x['episodes']} ep, sig {x['frac_significant']:.0%}, max {x['max_abs_significant']}"
                                                   for k, x in vv.items()), flush=True)
    print(f"wrote {NPZ} ({NPZ.stat().st_size / 1e6:.1f} MB)", flush=True)
    return 0


def render(out_dir=SITE) -> list:
    """Draw the two impact figures from the committed composites (runs in Actions with the live product)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    plt.rcParams.update({"font.size": 10.5, "font.family": "DejaVu Sans"})
    ref = xr.open_dataset(REF / "pacjet_ref.nc").load()
    Z = np.load(NPZ)
    st = json.loads((REF / "pacjet_impacts.json").read_text())
    import pacjet as PJ
    srange = PJ.season_label(ref)
    proj = ccrs.LambertConformal(central_longitude=-125, standard_parallels=(35, 60))
    lo_ = np.linspace(DOM["lon"][0], DOM["lon"][1], 60); la_ = np.linspace(DOM["lat"][0], DOM["lat"][1], 60)
    LO, LA = np.meshgrid(lo_, la_)
    xy = proj.transform_points(ccrs.PlateCarree(), LO.ravel(), LA.ravel())
    asp = (xy[:, 1].max() - xy[:, 1].min()) / (xy[:, 0].max() - xy[:, 0].min())
    written = []
    for tag, lags, tlabel in TAGS:
        ncol = 4
        mw = 3.35; mh = mw * asp
        gap, left, right, top, bot = 0.12, 0.55, 1.3, 1.45, 0.62
        W = left + ncol * mw + (ncol - 1) * gap + right
        H = top + 3 * mh + 2 * 0.22 + bot
        fig = plt.figure(figsize=(W, H))
        for c, name in enumerate(PC.PHASES):
            for r, (var, vlab, lev, cmap) in enumerate(ROWS):
                lat, lon = Z[f"lat_{var}"], Z[f"lon_{var}"]
                shown = Z[f"{tag}_{name}_{var}"]
                info = st["phases"][tag][name][var]
                x0 = (left + c * (mw + gap)) / W; y0 = (bot + (2 - r) * (mh + 0.22)) / H
                ax = fig.add_axes([x0, y0, mw / W, mh / H], projection=proj)
                ax.set_extent([DOM["lon"][0] - 360, DOM["lon"][1] - 360, DOM["lat"][0], DOM["lat"][1]], ccrs.PlateCarree())
                cf = ax.contourf(lon, lat, shown, levels=lev, cmap=cmap, extend="both", transform=ccrs.PlateCarree())
                ax.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.6, edgecolor="#333")
                ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.35, edgecolor="#555")
                ax.add_feature(cfeature.STATES.with_scale("50m"), lw=0.2, edgecolor="#777")
                for sp in ax.spines.values():
                    sp.set_edgecolor(PH_COL[name]); sp.set_linewidth(2.4)
                if r == 0:
                    ax.set_title(f"{PC.PHASE_LABEL[name]}\n{info['days']} days · {info['episodes']} episodes", fontsize=11.5,
                                 fontweight="bold", color=INK)
                    sgn = {"extension": (0, 1), "retraction": (0, -1), "poleward": (1, 1), "equatorward": (1, -1)}[name]
                    patt = sgn[1] * ref.eof_reg.sel(month=1).values[sgn[0]]    # descriptive: the phase's jet pattern (January scale)
                    ax.contour(PC.EOF_LON, PC.EOF_LAT, patt, levels=[-9, -6, -3, 3, 6, 9], colors="#222", linewidths=0.7,
                               transform=ccrs.PlateCarree())
                if c == 0:
                    fig.text((left - 0.15) / W, y0 + mh / H / 2, {"z500": "500 hPa height", "t2m": "temperature", "prcp": "precipitation"}[var],
                             rotation=90, ha="right", va="center", fontsize=11.5, fontweight="bold", color=INK)
                if not np.isfinite(shown).any():
                    ax.text(0.5, 0.5, "no significant signal", transform=ax.transAxes, ha="center", va="center", fontsize=10,
                            color=MUTED, bbox=dict(facecolor="white", lw=0, alpha=0.85))
                if c == ncol - 1:
                    cax = fig.add_axes([(left + ncol * (mw + gap)) / W, y0 + 0.06 * mh / H, 0.14 / W, 0.88 * mh / H])
                    cb = fig.colorbar(cf, cax=cax, ticks=(lev[::2] if var != "prcp" else lev))
                    cb.set_label(vlab, fontsize=9.5); cb.ax.tick_params(labelsize=8.5)
        wrapc = int(W * 13.5)
        fig.text(0.2 / W, 1 - 0.15 / H, f"North Pacific jet phases, {srange}: what each brings, {tlabel}", fontsize=15,
                 fontweight="bold", va="top", color=INK)
        fig.text(0.2 / W, 1 - 0.52 / H, textwrap.fill(
            f"ERA5 1991–2020 composites over the {srange} days in each phase of the 250 hPa jet (Winters et al. 2019 phases: "
            "the cold-season patterns, scaled to each month's spread), anomalies against the day-of-year normal with the season's "
            "linear trend removed. Contours on the top row: the phase's 250 hPa wind pattern (every 3 m/s, negative dashed).", wrapc),
            fontsize=10, va="top", color=MUTED)
        fig.text(0.2 / W, 0.08 / H, textwrap.fill(
            "Colour only where significant: two-sided t-test on episode means (one value per run of consecutive phase days), "
            "false discovery rate held at 10% over the map (Benjamini–Hochberg; Wilks 2016). White = no significant difference "
            "from normal.", wrapc), fontsize=9.5, va="bottom", color=MUTED)
        out = Path(out_dir) / f"pacjet_impacts_{tag}.webp"
        fig.savefig(out, dpi=105, facecolor="white", pil_kwargs={"quality": 88, "method": 6})
        plt.close(fig)
        written.append(out)
    return written


if __name__ == "__main__":
    if "--render" in sys.argv:
        print(render(sys.argv[sys.argv.index("--render") + 1] if len(sys.argv) > sys.argv.index("--render") + 1 else SITE))
    else:
        raise SystemExit(main())
