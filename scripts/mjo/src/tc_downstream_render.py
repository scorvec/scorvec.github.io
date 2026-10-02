#!/usr/bin/env python3
"""Figures for the historical TC-downstream composites (build_tc_downstream.py reference files) - runs in Actions.

    python src/tc_downstream_render.py --out-dir ../../assets/sst
Writes tcdown_{basin}_{view}_{season}.webp (view: z500 / t2m / prcp maps, curves = area-mean lag curves) and
data/tcdown_index.json (what exists + the reference stamp). Skips the render when the stamp matches (static product).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyproj  # noqa: F401  (import before xarray/eccodes: see CLAUDE.md, the eckit libproj clash)
import numpy as np
import xarray as xr

REF = Path(__file__).resolve().parents[1] / "data" / "reference"
BASIN_LAB = {"WP": "western North Pacific", "EP": "eastern North Pacific", "NA": "North Atlantic"}
SEASON_LAB = {"aug_sep": "August–September", "oct_nov": "October–November", "jun_nov": "June–November"}
VAR = {"z500": ("500 hPa height anomaly", "m", "RdBu_r", np.arange(-80, 81, 10)),
       "t2m": ("2 m temperature anomaly", "°C", "RdBu_r", np.arange(-4, 4.01, 0.5)),
       "prcp": ("precipitation anomaly", "mm/day", "BrBG", np.arange(-4, 4.01, 0.5))}
EXTENT = {"WP": (100, 300, 10, 80), "EP": (160, 320, 10, 80), "NA": (250, 400, 15, 80)}
SHOW_LAGS = (0, 4, 8, 12)
CURVE_REG = {"WP": ("wcan_pnw", "west_us", "central_us", "east_us"), "EP": ("wcan_pnw", "west_us", "central_us", "east_us"),
             "NA": ("east_us", "w_europe", "n_europe")}


def stamp() -> str:
    h = hashlib.sha1()
    for f in ("tc_downstream_comp.nc", "tc_downstream.json"):
        h.update((REF / f).read_bytes())
    h.update(Path(__file__).read_bytes())
    return h.hexdigest()[:12]


def maps(ds, js, b, var, sk, out: Path):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm
    title, unit, cmap, lev = VAR[var]
    x0, x1, y0, y1 = EXTENT[b]
    proj = ccrs.PlateCarree(central_longitude=(x0 + x1) / 2)
    info = js["sets"][f"{b}|{sk}"]
    rows = [("all", f"All recurving storms ({info['n_storms']['all']} storms, {info['n_episodes']['all']} independent episodes)")]
    if info["n_episodes"].get("strong") and info["n_episodes"].get("weak"):
        rows.append(("diff", f"Strong minus weak outflow–jet interaction (top vs bottom third: {info['n_episodes']['strong']} vs "
                             f"{info['n_episodes']['weak']} episodes)"))
    H = 1.95 * len(rows) + 1.75
    fig = plt.figure(figsize=(16, H))
    norm = BoundaryNorm(lev, 256, extend="both")
    lat, lon = ds.latitude.values, ds.longitude.values
    lon_c = np.r_[lon, lon[0] + 360]
    cf = None
    for r, (gk, lab) in enumerate(rows):
        vn = f"{var}__{b}__{sk}__{gk}"
        for c, L in enumerate(SHOW_LAGS):
            ax = fig.add_subplot(len(rows), len(SHOW_LAGS), r * len(SHOW_LAGS) + c + 1, projection=proj)
            ax.set_extent((x0, x1, y0, y1), crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.LAND, facecolor="#f2f1ec", zorder=0)
            if vn in ds:
                f = ds[vn].sel(lag=L).values
                if var == "z500" or var == "t2m" or var == "prcp":
                    pass
                fc = np.concatenate([f, f[:, :1]], 1)
                if np.isfinite(fc).any():
                    cf = ax.pcolormesh(lon_c, lat, fc, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), shading="nearest")
                else:
                    ax.text(0.5, 0.5, "nothing significant", transform=ax.transAxes, ha="center", va="center", color="#6f6b64",
                            fontsize=11)
            ax.coastlines(resolution="110m", lw=0.6, color="#333")
            ax.add_feature(cfeature.BORDERS, lw=0.3, edgecolor="#666")
            ax.set_title(f"day +{L}" + (" (recurvature day)" if L == 0 else ""), fontsize=11, loc="left")
            if c == 0:
                ax.text(0, 1.2, lab, transform=ax.transAxes, fontsize=11.5, fontweight="bold", ha="left")
    fig.suptitle(f"After a {BASIN_LAB[b]} tropical cyclone recurves — {title}, {SEASON_LAB[sk]}, ERA5 1991–2020",
                 fontsize=14, fontweight="bold", x=0.02, ha="left", y=0.995)
    fig.subplots_adjust(left=0.02, right=0.98, top=1 - 0.95 / H, bottom=1.2 / H, hspace=0.42, wspace=0.05)
    if cf is not None:
        cax = fig.add_axes([0.3, 0.66 / H, 0.4, 0.14 / H])
        cb = fig.colorbar(cf, cax=cax, orientation="horizontal"); cb.set_label(f"{title} ({unit})")
    fig.text(0.02, 0.06 / H, "Only cells passing a test across independent episodes (storms within 4 days merged) with a 10 % false-"
             "discovery rate are coloured; white = not significant. Anomalies vs a 1991–2020 seasonal cycle + trend. Associations, "
             "not proof of cause.", fontsize=8.6, color="#6f6b64")
    fig.savefig(out, dpi=72, facecolor="white", pil_kwargs={"quality": 82, "method": 6}); plt.close(fig)


def rel_maps(ds, js, b, sk, out: Path):
    """Storm-relative z500: x = degrees east of the recurvature point, no geography (it differs storm to storm)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm
    title, unit, cmap, lev = VAR["z500"]
    info = js["sets"][f"{b}|{sk}"]
    rows = [("all", f"All recurving storms ({info['n_storms']['all']} storms, {info['n_episodes']['all']} episodes)")]
    if info["n_episodes"].get("strong") and info["n_episodes"].get("weak"):
        rows.append(("diff", f"Strong minus weak outflow–jet interaction ({info['n_episodes']['strong']} vs "
                             f"{info['n_episodes']['weak']} episodes)"))
    lat = ds.latitude.values; rl = ds.longitude.values - 180.0
    m = (rl >= -60) & (rl <= 150)
    fig, axs = plt.subplots(len(rows), len(SHOW_LAGS), figsize=(16, 3.3 * len(rows) + 1.4), squeeze=False, sharex=True, sharey=True)
    norm = BoundaryNorm(lev, 256, extend="both"); cf = None
    for r, (gk, lab) in enumerate(rows):
        vn = f"z500rel__{b}__{sk}__{gk}"
        for c, L in enumerate(SHOW_LAGS):
            ax = axs[r, c]
            if vn in ds and np.isfinite(ds[vn].sel(lag=L).values[:, m]).any():
                cf = ax.pcolormesh(rl[m], lat, ds[vn].sel(lag=L).values[:, m], cmap=cmap, norm=norm, shading="nearest")
            else:
                ax.text(0.5, 0.5, "nothing significant", transform=ax.transAxes, ha="center", va="center", color="#6f6b64", fontsize=11)
            ax.plot([0], [info.get("recurve_lat_median", 25)], marker="*", ms=16, color="#111", mec="#fff")
            ax.axvline(0, color="#888", lw=0.6, ls=":")
            ax.set_xlim(-60, 150); ax.set_ylim(10, 80); ax.grid(alpha=0.25)
            ax.set_title(f"day +{L}" + (" (recurvature day)" if L == 0 else ""), fontsize=11, loc="left")
            if c == 0:
                ax.set_ylabel("latitude (°N)")
                ax.text(0, 1.17, lab, transform=ax.transAxes, fontsize=11.5, fontweight="bold")
            if r == len(rows) - 1:
                ax.set_xlabel("degrees east of the recurvature point")
    fig.suptitle(f"Storm-relative view: 500 hPa height anomaly after a {BASIN_LAB[b]} tropical cyclone recurves, {SEASON_LAB[sk]}, "
                 "ERA5 1991–2020", fontsize=14, fontweight="bold", x=0.01, ha="left")
    fig.subplots_adjust(left=0.05, right=0.99, top=0.84 if len(rows) == 1 else 0.88, bottom=0.3 if len(rows) == 1 else 0.2, hspace=0.45, wspace=0.06)
    if cf is not None:
        cax = fig.add_axes([0.3, 0.085 if len(rows) > 1 else 0.11, 0.4, 0.02]); cb = fig.colorbar(cf, cax=cax, orientation="horizontal"); cb.set_label(f"{title} ({unit})")
    fig.text(0.01, 0.008, "Each storm's fields are shifted so it recurves at 0 (★ = median recurvature latitude); the ridge and trough its "
             "outflow builds line up even when storms recurve at different longitudes.\nOnly cells passing a 10 % false-discovery-rate "
             "test across independent episodes are coloured.", fontsize=8.6, color="#6f6b64")
    fig.savefig(out, dpi=72, facecolor="white", pil_kwargs={"quality": 82, "method": 6}); plt.close(fig)


def curves(js, b, sk, out: Path):
    import matplotlib.pyplot as plt
    info = js["sets"][f"{b}|{sk}"]
    regs = CURVE_REG[b]
    fig, axs = plt.subplots(2, len(regs), figsize=(4.1 * len(regs), 7.2), squeeze=False)
    lags = np.array(js["lags"])
    for c, rk in enumerate(regs):
        for r, var in enumerate(("t2m", "prcp")):
            ax = axs[r, c]
            ax.axhline(0, color="#999", lw=0.8)
            for gk, col, lab in (("all", "#c0392b", "all recurving storms"), ("diff", "#1f5fa8", "strong minus weak interaction")):
                cv = info["curves"].get(f"{var}|{rk}|{gk}")
                if not cv:
                    continue
                m, lo, hi, sg = map(np.array, (cv["mean"], cv["lo"], cv["hi"], cv["sig"]))
                ax.fill_between(lags, lo, hi, color=col, alpha=0.15, lw=0)
                ax.plot(lags, m, color=col, lw=1.6, label=lab)
                ax.plot(lags[sg], m[sg], "o", color=col, ms=6)
            if r == 0:
                ax.set_title(js["regions"][rk], fontsize=11.5, loc="left", fontweight="bold")
            ax.set_ylabel(("2 m temperature anomaly (°C)" if var == "t2m" else "precipitation anomaly (mm/day)") if c == 0 else "")
            ax.set_xlabel("days after recurvature" if r == 1 else "")
            ax.grid(alpha=0.3)
            if r == 0 and c == 0:
                ax.legend(fontsize=9, frameon=False, loc="upper left")
    fig.suptitle(f"After a {BASIN_LAB[b]} tropical cyclone recurves — regional means, {SEASON_LAB[sk]}, ERA5 1991–2020 "
                 f"({info['n_storms']['all']} storms, {info['n_episodes']['all']} episodes)", fontsize=13, fontweight="bold",
                 x=0.01, ha="left")
    fig.text(0.01, 0.01, "Shading: 90 % bootstrap interval over episodes. Dots: days that pass a 10 % false-discovery-rate test "
             "run jointly over all regions shown and the 15 lags. Associations, not proof of cause.", fontsize=8.6, color="#6f6b64")
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(out, dpi=72, facecolor="white", pil_kwargs={"quality": 82, "method": 6}); plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--out-dir", default="../../assets/sst"); ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    out = Path(a.out_dir); (out / "data").mkdir(parents=True, exist_ok=True)
    idxf = out / "data" / "tcdown_index.json"
    st = stamp()
    if idxf.exists() and not a.force and json.loads(idxf.read_text()).get("stamp") == st:
        print("  TC downstream figures up to date (reference unchanged)"); return 0
    js = json.loads((REF / "tc_downstream.json").read_text())
    ds = xr.open_dataset(REF / "tc_downstream_comp.nc")
    made = []
    for key, info in js["sets"].items():
        b, sk = key.split("|")
        if not info["n_episodes"].get("all"):
            continue
        for var in VAR:
            f = out / f"tcdown_{b}_{var}_{sk}.webp"; maps(ds, js, b, var, sk, f); made.append(f.name)
        f = out / f"tcdown_{b}_curves_{sk}.webp"; curves(js, b, sk, f); made.append(f.name)
        f = out / f"tcdown_{b}_rel_{sk}.webp"; rel_maps(ds, js, b, sk, f); made.append(f.name)
    idxf.write_text(json.dumps({"stamp": st, "files": made}))
    print(f"  wrote {len(made)} TC downstream figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
