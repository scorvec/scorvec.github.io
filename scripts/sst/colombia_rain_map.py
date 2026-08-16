#!/usr/bin/env python3
"""IMERG rainfall maps over Colombia with IDEAM rain gauges overlaid.

The visual ground-truth: satellite estimate as the field, every reporting
IDEAM gauge as a dot on the SAME color scale — where dots vanish into the
background the satellite is right; where they pop, it isn't. Four panels:

  1. yesterday's IMERG daily total + gauges
  2. 7-day accumulation + gauges (stations reporting >=6 of 7 days)
  3. 30-day accumulation + gauges (>=27 of 30 days)
  4. IMERG-at-gauge vs gauge scatter for the 30-day totals, log-log,
     colored by hydro region, annotated with each region's median
     satellite/gauge ratio — the multiplicative bias factors the
     rain->inflow model will use.

XM region polygons (ideam variant) outlined on every map. Bias factors
also written to colombia_hydro/data/gauge_bias.json.

    python scripts/sst/colombia_rain_map.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.path import Path as MplPath

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import imerg_precip as IP                      # noqa: E402
from ideam_gauges import fetch_range           # noqa: E402

REPO = HERE.parent.parent
REGIONS_GJ = HERE / "colombia_hydro_regions.geojson"
OUT_PNG = REPO / "colombia_hydro" / "rain_vs_gauges.webp"
OUT_JSON = REPO / "colombia_hydro" / "data" / "gauge_bias.json"
ORDER = ["ANTIOQUIA", "CALDAS", "CARIBE", "CENTRO", "ORIENTE", "VALLE"]
RCOL = {"ANTIOQUIA": "#1b7837", "CALDAS": "#762a83", "CARIBE": "#2166ac",
        "CENTRO": "#b2182b", "ORIENTE": "#e08214", "VALLE": "#35978f"}
# Colombia crop (deg E 0-360, deg N)
LON0, LON1, LAT0, LAT1 = 281.0, 294.5, -4.8, 13.2

CMAP = ListedColormap([
    "#f7f7f7", "#c7e9c0", "#74c476", "#238b45", "#41b6c4", "#225ea8",
    "#253494", "#54278f", "#7a0177", "#ae017e"])


def region_paths():
    gj = json.loads(REGIONS_GJ.read_text())
    out = {}
    for ft in gj["features"]:
        name = (ft["properties"].get("region") or ft["properties"].get("name", "")).upper()
        geom = ft["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        rings = [np.array(r[0]) for r in polys]
        out[name] = rings
    return out


def imerg_stack(days: list[datetime]):
    """(nday, lat, lon) cropped daily fields + axes; missing days -> None rows."""
    ml, mt = IP._grid_axes()
    lons = np.sort(IP._LON[ml] % 360)
    lats = np.sort(IP._LAT[mt])
    li = (lons >= LON0) & (lons <= LON1)
    la = (lats >= LAT0) & (lats <= LAT1)
    fields = []
    for d in days:
        g = IP._load(IP.DAILY_CACHE, f"{d:%Y%m%d}")
        fields.append(None if g is None else g[np.ix_(la, li)])
    return fields, lons[li], lats[la]


def main() -> int:
    end = (datetime.now(timezone.utc) - timedelta(days=1)).replace(tzinfo=None)
    days = [end - timedelta(days=k) for k in range(30)][::-1]      # oldest first
    IP.ensure_daily(set(days))
    fields, lons, lats = imerg_stack(days)
    have = [i for i, f in enumerate(fields) if f is not None]
    if not have:
        print("no IMERG dailies cached")
        return 1
    latest_i = have[-1]

    print("fetching gauges …", flush=True)
    gauges = fetch_range(end, 30)                                   # {date: {code:{la,lo,mm}}}

    # The gauge feed has real multi-week holes (verified server-side: zero
    # rows Jul 25-Aug 11 2026). Accumulate over PAIRED days only — the last
    # N days where BOTH a gauge day and an IMERG daily exist — and sum the
    # satellite over exactly those dates, so totals stay comparable.
    feed_days = [d for d in days if gauges.get(d)]
    imerg_ok = {days[i] for i in have}
    paired = [d for d in feed_days if d in imerg_ok]

    def paired_sum(window: list[datetime]):
        need = max(1, int(0.85 * len(window)))
        acc, cnt, meta = {}, {}, {}
        for d in window:
            for code, g in gauges[d].items():
                acc[code] = acc.get(code, 0.0) + g["mm"]
                cnt[code] = cnt.get(code, 0) + 1
                meta[code] = (g["la"], g["lo"])
        gg = {c: {"la": meta[c][0], "lo": meta[c][1], "mm": acc[c]}
              for c in acc if cnt[c] >= need}
        sat = np.nansum([fields[days.index(d)] for d in window], axis=0)
        return gg, sat

    day_g = gauges.get(days[latest_i], {})
    wk_days = paired[-7:]
    mo_days = paired[-30:]
    wk_g, wk = paired_sum(wk_days)
    mo_g, mo = paired_sum(mo_days)

    rp = region_paths()
    paths = {r: [MplPath(np.column_stack([(ring[:, 0] % 360), ring[:, 1]]))
                 for ring in rings] for r, rings in rp.items() if r in ORDER}

    def assign_region(la, lo):
        for r, ps in paths.items():
            if any(p.contains_point((lo % 360, la)) for p in ps):
                return r
        return None

    fig, axes = plt.subplots(1, 4, figsize=(22.5, 7.2))
    panels = [
        (fields[latest_i], day_g, f"IMERG daily — {days[latest_i]:%b %d}",
         [0, 1, 2, 5, 10, 20, 35, 50, 75, 100, 150]),
        (wk, wk_g, f"{len(wk_days)}-day accumulation (paired feed days)",
         [0, 5, 10, 25, 50, 100, 150, 200, 300, 400, 600]),
        (mo, mo_g, f"{len(mo_days)}-day accumulation (paired feed days)",
         [0, 20, 50, 100, 200, 300, 450, 600, 800, 1000, 1500]),
    ]
    for ax, (field, gg, title, lev) in zip(axes[:3], panels):
        norm = BoundaryNorm(lev, CMAP.N)
        pm = ax.pcolormesh(lons, lats, field, cmap=CMAP, norm=norm, shading="nearest")
        for r, rings in rp.items():
            if r not in ORDER:
                continue
            for ring in rings:
                ax.plot(ring[:, 0] % 360, ring[:, 1], color="#222222", lw=0.8)
        if gg:
            gl = np.array([[g["lo"] % 360, g["la"], g["mm"]] for g in gg.values()])
            inside = (gl[:, 0] >= LON0) & (gl[:, 0] <= LON1) & \
                     (gl[:, 1] >= LAT0) & (gl[:, 1] <= LAT1)
            gl = gl[inside]
            ax.scatter(gl[:, 0], gl[:, 1], c=gl[:, 2], cmap=CMAP, norm=norm,
                       s=15, edgecolors="black", linewidths=0.35, zorder=5)
            title += f" · {inside.sum()} gauges"
        ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
        ax.set_xlim(LON0, LON1); ax.set_ylim(LAT0, LAT1)
        ax.set_aspect("equal")
        ax.set_xticks([282, 286, 290, 294])
        ax.set_xticklabels(["78°W", "74°W", "70°W", "66°W"], fontsize=8)
        ax.tick_params(labelsize=8)
        cb = fig.colorbar(pm, ax=ax, orientation="horizontal", fraction=0.05,
                          pad=0.07, aspect=32)
        cb.set_label("mm", fontsize=8)
        cb.ax.tick_params(labelsize=7)

    # ── panel 4: satellite vs gauge, 30-day totals ──────────────────────────
    ax = axes[3]
    bias = {r: [] for r in ORDER}
    pts = {"x": [], "y": [], "c": []}
    for g in mo_g.values():
        la, lo = g["la"], g["lo"] % 360
        if not (LON0 <= lo <= LON1 and LAT0 <= la <= LAT1):
            continue
        i = int(np.argmin(np.abs(lats - la)))
        j = int(np.argmin(np.abs(lons - lo)))
        sat = float(mo[i, j])
        reg = assign_region(la, lo)
        if g["mm"] > 5 and sat > 5:
            pts["x"].append(g["mm"]); pts["y"].append(sat)
            pts["c"].append(RCOL.get(reg, "#999999"))
            if reg:
                bias[reg].append(sat / g["mm"])
    ax.scatter(pts["x"], pts["y"], c=pts["c"], s=13, alpha=0.7,
               edgecolors="none")
    lim = (5, max(max(pts["x"], default=100), max(pts["y"], default=100)) * 1.2)
    ax.plot(lim, lim, color="0.3", lw=1.0, ls="--")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("gauge 30-day total (mm)", fontsize=9)
    ax.set_ylabel("IMERG at gauge (mm)", fontsize=9)
    lines = []
    factors = {}
    for r in ORDER:
        if len(bias[r]) >= 5:
            f = float(np.median(bias[r]))
            factors[r] = round(f, 2)
            star = "" if len(bias[r]) >= 15 else "*"
            lines.append(f"{r[:9]:<9} ×{f:4.2f}{star} (n={len(bias[r])})")
    allb = [b for r in ORDER for b in bias[r]]
    if allb:
        factors["ALL"] = round(float(np.median(allb)), 2)
        lines.append(f"{'ALL':<9} ×{factors['ALL']:4.2f}  (n={len(allb)})")
    ax.text(0.03, 0.97, "median IMERG/gauge (* = few gauges)\n" + "\n".join(lines),
            transform=ax.transAxes, va="top", fontsize=8.5, family="monospace",
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    ax.set_title(f"Satellite vs gauge — {len(mo_days)}-day paired totals", fontsize=11,
                 fontweight="bold", loc="left")
    ax.grid(lw=0.25, alpha=0.5, which="both")
    ax.tick_params(labelsize=8)

    fig.suptitle("IMERG vs IDEAM rain gauges — Colombia · gauges drawn on the SAME "
                 "color scale (dots that vanish = satellite agrees)",
                 fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=115)
    plt.close(fig)
    print(f"wrote {OUT_PNG.relative_to(REPO)}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "window": f"{days[0]:%Y-%m-%d}..{days[-1]:%Y-%m-%d}",
        "median_imerg_over_gauge_paired": factors, "paired_days": len(mo_days),
        "note": ("multiplicative; gauge day = Colombia local calendar day, "
                 "IMERG day = UTC — offset immaterial at 30 days"),
    }, separators=(",", ":")))
    print(f"wrote {OUT_JSON.relative_to(REPO)}: {factors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
