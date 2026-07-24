#!/usr/bin/env python3
"""Per-river rain→inflow correlation — the river-level cut of the validation.

XM meters inflow energy (AporEner) river-by-river, and REGION_SZH carries the
river→subzona provenance, so each river can get rainfall averaged over ITS OWN
catchment (union of its IDEAM subzonas, raw — no disjointness carving at river
level) and its own correlation. Same method as validate_region_rain.py:
3-day trailing blocks (segment-aware, never straddling gaps in the IMERG
record), lag scanned 0–15 d, variants raw / anomaly / burst.

Outputs:
    ~/colombia_hydro/out/river_corr.json
    colombia_hydro/river_corr.webp        (per-region bars, best r per river)

    python scripts/sst/river_corr.py
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import imerg_precip as IP
from build_imerg_clim import eval_clim
from hydro_region_rain import _axes
from validate_region_rain import fetch_aportes, ORDER, COLORS
from build_hydro_regions import REGION_SZH, RAW

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RIVERS_JSON = Path.home() / "colombia_hydro" / "raw" / "xm_listado_rios.json"
OUT = Path.home() / "colombia_hydro" / "out"
OUT_PNG = REPO / "colombia_hydro" / "river_corr.webp"
MIN_DAYS = 45                 # min overlapping 3-day blocks to report an r


def active_rivers() -> list[dict]:
    d = json.load(open(RIVERS_JSON))
    out = []
    for it in d["Items"]:
        for e in it["ListEntities"]:
            v = e["Values"]
            if v.get("Status") == "ACTIVO" and v["HydroRegion"] in ORDER:
                out.append(dict(name=v["Name"].strip().upper(),
                                region=v["HydroRegion"]))
    return out


def river_szh(river: str, region: str) -> list[int]:
    """Subzona codes for one river: exact name-in-comment first, tokens second."""
    notes = REGION_SZH[region]
    exact = [s for s, n in notes.items() if river in n.upper()]
    if exact:
        return exact
    key = river.replace(" CP", "").replace("DESV. ", "").split(" (")[0]
    toks = [t for t in key.split() if len(t) > 3] or [key]
    return [s for s, n in notes.items() if any(t in n.upper() for t in toks)]


def main() -> int:
    rivers = active_rivers()
    lon, lat, coef = _axes()

    # per-river IMERG weights from raw IDEAM subzonas (river catchments may
    # legitimately share subzonas — no carving at this level)
    import geopandas as gpd
    from shapely import contains_xy, prepare
    from shapely.ops import unary_union
    need: dict[str, list[int]] = {}
    for rv in rivers:
        s = river_szh(rv["name"], rv["region"])
        if not s:
            print(f"  !! {rv['name']} ({rv['region']}): no subzona match — skipped")
            continue
        need[rv["name"]] = s
    print(f"loading IDEAM subzonas for {len(need)} rivers …", flush=True)
    gdfz = gpd.read_file(RAW / "ideam_zonificacion.geojson")
    gdfz["cod_szh"] = gdfz["cod_szh"].astype(float).astype(int)
    xx, yy = np.meshgrid(lon, lat)
    coslat = np.cos(np.radians(yy))
    szh_mask: dict[int, np.ndarray] = {}
    for s in sorted({c for v in need.values() for c in v}):
        g = unary_union(gdfz[gdfz["cod_szh"] == s].geometry.values)
        prepare(g)
        szh_mask[s] = contains_xy(g, xx, yy)
    weights = {}
    for name, szhs in need.items():
        m = np.any([szh_mask[s] for s in szhs], axis=0)
        w = np.where(m, coslat, 0.0)
        if w.sum() == 0:                       # catchment smaller than one cell
            print(f"  !! {name}: no IMERG cells inside — skipped")
            continue
        weights[name] = w / w.sum()

    # rain series on every cached daily grid
    days = [datetime.strptime(f.stem, "%Y%m%d").replace(tzinfo=timezone.utc)
            for f in sorted(IP.DAILY_CACHE.glob("*.npy"))]
    dts, grids = [], []
    for d in days:
        g = IP._load(IP.DAILY_CACHE, f"{d:%Y%m%d}")
        if g is not None:
            dts.append(d.replace(tzinfo=None)); grids.append(g)
    stack = np.stack(grids).reshape(len(dts), -1)
    rain = {n: stack @ w.ravel() for n, w in weights.items()}
    clim = {n: np.array([float((eval_clim(coef, d.timetuple().tm_yday) * w).sum())
                         for d in dts]) for n, w in weights.items()}

    # inflows: fetch each contiguous IMERG segment (the record has gaps)
    seg_id = np.zeros(len(dts), int)
    for i in range(1, len(dts)):
        seg_id[i] = seg_id[i - 1] + ((dts[i] - dts[i - 1]).days > 1)
    apor: dict[str, dict[str, float]] = {}
    for sid in np.unique(seg_id):
        ss = [d for d, s in zip(dts, seg_id) if s == sid]
        print(f"fetching XM aportes {ss[0]:%Y-%m-%d} … {ss[-1]:%Y-%m-%d}", flush=True)
        apor.update(fetch_aportes(ss[0], ss[-1]))
    inflow = {}
    for rv in rivers:
        v = np.full(len(dts), np.nan)
        for i, d in enumerate(dts):
            kwh = apor.get(f"{d:%Y-%m-%d}", {}).get(rv["name"])
            if kwh is not None:
                v[i] = kwh / 1e6
        inflow[rv["name"]] = v

    # dump the aligned per-river series so downstream response-model experiments
    # (exp_decay_corr.py etc.) don't re-touch IDEAM polygons or the XM API
    OUT.mkdir(parents=True, exist_ok=True)
    _nan = lambda a: [None if not np.isfinite(x) else round(float(x), 3) for x in a]
    (OUT / "river_series.json").write_text(json.dumps(dict(
        dates=[f"{d:%Y-%m-%d}" for d in dts],
        rivers={rv["name"]: dict(region=rv["region"],
                                 rain=_nan(rain[rv["name"]]),
                                 clim=_nan(clim[rv["name"]]),
                                 inflow=_nan(inflow[rv["name"]]))
                for rv in rivers if rv["name"] in rain})))

    def roll3(x):
        out = np.full(len(x), np.nan)
        for sid in np.unique(seg_id):
            m = seg_id == sid
            xx_ = x[m]
            sm = np.convolve(np.nan_to_num(xx_), np.ones(3), "full")[:m.sum()]
            bad = np.convolve(np.isnan(xx_).astype(float), np.ones(3), "full")[:m.sum()]
            sm[bad > 0] = np.nan
            sm[:2] = np.nan
            out[m] = sm
        return out

    def scan(xs, ys):
        best = (0, 0.0)
        for lag in range(0, 16):
            xr_ = np.roll(xs, lag); xr_[:lag] = np.nan
            m = np.isfinite(xr_) & np.isfinite(ys) & (np.roll(seg_id, lag) == seg_id)
            if m.sum() > 30:
                cc = float(np.corrcoef(xr_[m], ys[m])[0, 1])
                if cc > best[1]:
                    best = (lag, cc)
        return best

    doy = np.array([d.timetuple().tm_yday for d in dts], float)
    w_ = 2 * np.pi * doy / 365.25
    X = np.column_stack([np.ones_like(w_), np.cos(w_), np.sin(w_),
                         np.cos(2 * w_), np.sin(2 * w_)])

    region_tot = {r: 0.0 for r in ORDER}
    for rv in rivers:
        region_tot[rv["region"]] += float(np.nansum(inflow[rv["name"]]))
    rows = []
    for rv in rivers:
        name, reg = rv["name"], rv["region"]
        if name not in rain:
            continue
        ii = inflow[name]
        n = int(np.isfinite(ii).sum())
        share = 100 * float(np.nansum(ii)) / max(region_tot[reg], 1e-9)
        if n < MIN_DAYS:
            rows.append(dict(river=name, region=reg, share_pct=round(share, 1),
                             n_days=n, corr=None))
            continue
        r3, i3 = roll3(rain[name]), roll3(ii)
        cl3 = roll3(clim[name])
        fin = np.isfinite(i3)
        cf, *_ = np.linalg.lstsq(X[fin], i3[fin], rcond=None)
        ia3 = i3 - X @ cf
        variants = {"raw": (r3, i3), "anom": (r3 - cl3, ia3)}
        bT, bB = 0, (0, 0.0)
        for T in range(1, 9):
            lg, cc = scan(roll3(np.maximum(rain[name] - T, 0.0)), ia3)
            if cc > bB[1]:
                bT, bB = T, (lg, cc)
        variants[f"burst>{bT}mm"] = (roll3(np.maximum(rain[name] - bT, 0.0)), ia3)
        res = {nm: scan(x, y) for nm, (x, y) in variants.items()}
        bname = max(res, key=lambda nm: res[nm][1])
        k, c = res[bname]
        rows.append(dict(river=name, region=reg, share_pct=round(share, 1),
                         n_days=n, n_szh=len(need[name]), corr=round(c, 2),
                         lag_days=k, variant=bname,
                         all_variants={nm: dict(lag=l, r=round(r, 3))
                                       for nm, (l, r) in res.items()}))
    rows.sort(key=lambda r: (r["region"], -r["share_pct"]))

    print(f"{'region':<10} {'river':<26} {'share':>6} {'r':>5}  lag  variant")
    for r in rows:
        if r["corr"] is None:
            print(f"{r['region']:<10} {r['river']:<26} {r['share_pct']:>5.1f}%    —  "
                  f"(only {r['n_days']} days)")
        else:
            print(f"{r['region']:<10} {r['river']:<26} {r['share_pct']:>5.1f}% "
                  f"{r['corr']:>5.2f}  {r['lag_days']}d  {r['variant']}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "river_corr.json").write_text(json.dumps(dict(
        window=f"{dts[0]:%Y-%m-%d}..{dts[-1]:%Y-%m-%d}",
        n_grid_days=len(dts), rivers=rows), indent=1))

    fig, axs = plt.subplots(2, 3, figsize=(13.2, 8.6))
    for ax, reg in zip(axs.ravel(), ORDER):
        rr = [r for r in rows if r["region"] == reg and r["corr"] is not None][::-1]
        ys = np.arange(len(rr))
        ax.barh(ys, [r["corr"] for r in rr], color=COLORS[reg],
                alpha=0.85, edgecolor="k", linewidth=0.4)
        for y, r in zip(ys, rr):
            ax.text(max(r["corr"], 0) + 0.015, y,
                    f"{r['share_pct']:.0f}% · {r['lag_days']}d", va="center",
                    fontsize=6, color="0.35")
        ax.set_yticks(ys)
        ax.set_yticklabels([r["river"].title()[:22] for r in rr], fontsize=6.5)
        ax.axvline(0, color="k", lw=0.6)
        ax.set_xlim(min(0, min((r["corr"] for r in rr), default=0)) - 0.05, 1.0)
        ax.set_title(f"{reg}", fontsize=9.5, fontweight="bold", loc="left")
        ax.set_xlabel("best r (3-day rain vs 3-day inflow)", fontsize=7.5)
        ax.tick_params(labelsize=7); ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Per-river skill — rain over each river's own subzonas vs its XM inflow\n"
                 "labels: share of region inflow · best lag (rivers sorted by share)",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT_PNG, dpi=120, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT_PNG.name} + river_corr.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
