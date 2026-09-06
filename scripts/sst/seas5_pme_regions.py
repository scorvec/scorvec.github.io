#!/usr/bin/env python3
"""SEAS5 P − E by region, Brazil and Colombia: the forecast distribution against normal.

For each region, every member's monthly precipitation minus evaporation (mm/day) is
area-averaged over the region's polygon (fractional cell overlap on the 1° grid), and the
51 members' values for each month are drawn as a distribution — this issue against the
previous one — with the hindcast climatology as "normal" (its mean and the year-to-year
±1σ band). Under each panel: the forecast mean, its distance from normal, the change since
the previous issue, and the share of members drier than normal.

Brazil regions are the four electrical subsystems as state groups (Southeast/Centre-West,
South, Northeast, North); Colombia regions come from scripts/sst/colombia_hydro_regions.geojson.
Output: assets/sst/seas5_pme_{br,co}.webp + data/seas5_pme_regions.json.

    python scripts/sst/seas5_pme_regions.py --issue 202609
"""
from __future__ import annotations

import argparse
import calendar
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from seas5_outlook import ASSETS, fc_path, hc_path, previous_issues  # noqa: E402
from seas5_build import load_field, valid_months  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT_JSON = ASSETS / "data" / "seas5_pme_regions.json"

BR_SUBSYSTEMS = {
    "Southeast / Centre-West": ["São Paulo", "Rio de Janeiro", "Minas Gerais", "Espírito Santo", "Goiás", "Mato Grosso",
                                "Mato Grosso do Sul", "Distrito Federal", "Acre", "Rondônia"],
    "South": ["Paraná", "Santa Catarina", "Rio Grande do Sul"],
    "Northeast": ["Piauí", "Ceará", "Rio Grande do Norte", "Paraíba", "Pernambuco", "Alagoas", "Sergipe", "Bahia"],
    "North": ["Amazonas", "Pará", "Amapá", "Roraima", "Tocantins", "Maranhão"],
}
CO_ORDER = ["ANTIOQUIA", "CALDAS", "CARIBE", "CENTRO", "ORIENTE", "VALLE"]


# ── region polygons ──────────────────────────────────────────────────────────
def brazil_polygons() -> dict:
    from cartopy.io import shapereader
    from shapely.ops import unary_union
    p = shapereader.natural_earth(resolution="10m", category="cultural", name="admin_1_states_provinces")
    by_name = {r.attributes["name"]: r.geometry for r in shapereader.Reader(p).records() if r.attributes.get("admin") == "Brazil"}
    out = {}
    for label, states in BR_SUBSYSTEMS.items():
        missing = [s for s in states if s not in by_name]
        if missing:
            print(f"  warning: states not found for {label}: {missing}", flush=True)
        out[label] = unary_union([by_name[s] for s in states if s in by_name])
    return out


def colombia_polygons() -> dict:
    from shapely.geometry import shape
    g = json.loads((HERE / "colombia_hydro_regions.geojson").read_text())
    polys = {ft["properties"]["name"]: shape(ft["geometry"]) for ft in g["features"]}
    return {k: polys[k] for k in CO_ORDER if k in polys}


def cell_weights(poly, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Fractional overlap of each 1° cell (centred on lat/lon) with the polygon, normalised to 1.
    Small regions that enclose no cell centre still get weight from the cells they touch."""
    from shapely.geometry import box
    from shapely.prepared import prep
    minx, miny, maxx, maxy = poly.bounds
    w = np.zeros((lat.size, lon.size))
    dlat = abs(float(lat[1] - lat[0])) if lat.size > 1 else 1.0
    dlon = abs(float(lon[1] - lon[0])) if lon.size > 1 else 1.0
    pp = prep(poly)
    for i, la in enumerate(lat):
        if la + dlat / 2 < miny or la - dlat / 2 > maxy:
            continue
        for j, lo in enumerate(lon):
            if lo + dlon / 2 < minx or lo - dlon / 2 > maxx:
                continue
            cell = box(lo - dlon / 2, la - dlat / 2, lo + dlon / 2, la + dlat / 2)
            if pp.intersects(cell):
                w[i, j] = poly.intersection(cell).area * np.cos(np.deg2rad(la))
    if w.sum() == 0:
        c = poly.centroid
        w[int(np.argmin(np.abs(lat - c.y))), int(np.argmin(np.abs(lon - c.x)))] = 1.0
    return w / w.sum()


# ── model P − E ──────────────────────────────────────────────────────────────
def pme_fields(ym: str):
    """(fc[sample, lead, lat, lon], hc[...], lat, lon) in mm/day, or None."""
    fs, hs, fw, hw = fc_path("sfc", ym), hc_path("sfc", ym[4:]), fc_path("water", ym), hc_path("water", ym[4:])
    if not all(p.exists() for p in (fs, hs, fw, hw)):
        return None
    ftp, lat, lon = load_field(fs, "tprate"); htp, _, _ = load_field(hs, "tprate")
    fe, _, _ = load_field(fw, "e"); he, _, _ = load_field(hw, "e")
    k = 86400.0 * 1000
    return (ftp + fe) * k, (htp + he) * k, lat, lon


def region_series(fields, weights: dict) -> dict:
    """{region: (fc[sample, lead], hc[sample, lead])} area-weighted P − E."""
    fc, hc, lat, lon = fields
    out = {}
    for name, w in weights.items():
        out[name] = (np.tensordot(np.nan_to_num(fc), w, axes=([2, 3], [0, 1])), np.tensordot(np.nan_to_num(hc), w, axes=([2, 3], [0, 1])))
    return out


# ── figure ───────────────────────────────────────────────────────────────────
def render(country: str, label: str, series_now: dict, series_prev: dict | None, ym: str, prev: str, out: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scipy.stats import gaussian_kde

    vm, vmp = valid_months(ym), valid_months(prev)
    regions = list(series_now)
    ncol = len(vm)
    fig, axes = plt.subplots(len(regions), ncol, figsize=(3.05 * ncol + 1.1, 2.55 * len(regions) + 1.9), squeeze=False)
    summary = {}
    for r, name in enumerate(regions):
        fcn, hcn = series_now[name]
        n_yr = 24; per_year = np.nanmean(hcn.reshape(-1, n_yr, hcn.shape[1]), axis=0)          # [year, lead]
        clim, csd = np.nanmean(hcn, axis=0), np.nanstd(per_year, axis=0)
        summary[name] = {}
        for c, mon in enumerate(vm):
            ax = axes[r, c]
            cur = fcn[:, c]
            prv = None
            if series_prev and name in series_prev and mon in vmp:
                prv = series_prev[name][0][:, vmp.index(mon)]
            ax.axvspan(clim[c] - csd[c], clim[c] + csd[c], color="#000", alpha=0.06, zorder=0)
            ax.axvline(clim[c], color="#222", lw=1.0, zorder=4)
            allv = np.concatenate([v for v in (cur, prv) if v is not None])
            lo, hi = allv.min(), allv.max(); pad = 0.35 * (hi - lo + 1e-6)
            x = np.linspace(min(lo, clim[c] - csd[c]) - pad, max(hi, clim[c] + csd[c]) + pad, 240)
            for vals, colr, z in ((prv, "#3f7fbf", 1), (cur, "#b8860b", 2)):
                if vals is None:
                    continue
                kde = gaussian_kde(vals, bw_method=0.55)
                ax.fill_between(x, kde(x), color=colr, alpha=0.22, zorder=z); ax.plot(x, kde(x), color=colr, lw=1.6, zorder=z + 2)
                ax.axvline(vals.mean(), color=colr, lw=1.1, ls="--", zorder=z + 2)
                ax.plot(vals, np.full(vals.shape, -0.012 * kde(x).max() * (1 if z == 2 else 2.2)), "|", color=colr, ms=5, mew=0.8, alpha=0.7, zorder=z + 2, clip_on=False)
            ax.set_yticks([]); ax.spines[["left", "top", "right"]].set_visible(False); ax.tick_params(labelsize=8)
            if r == 0:
                ax.set_title(f"{calendar.month_abbr[int(mon[5:])]} {mon[:4]}", fontsize=11.5, loc="left")
            if c == 0:
                ax.set_ylabel(name, fontsize=9.5, fontweight="bold")
            p_dry = float(np.mean(cur < clim[c]))
            lines = [f"Forecast {cur.mean():+.2f} mm/day ({cur.mean() - clim[c]:+.2f} vs normal)"]
            if prv is not None:
                lines.append(f"Change since {calendar.month_abbr[int(prev[4:])]} issue: {cur.mean() - prv.mean():+.2f}")
            lines.append(f"Drier than normal: {p_dry:.0%} of members")
            ax.text(0.0, -0.22, "\n".join(lines), transform=ax.transAxes, ha="left", va="top", fontsize=7.2, color="#222", linespacing=1.35)
            summary[name][mon] = dict(normal=round(float(clim[c]), 3), normal_sd=round(float(csd[c]), 3), mean=round(float(cur.mean()), 3),
                                      p10=round(float(np.percentile(cur, 10)), 3), p90=round(float(np.percentile(cur, 90)), 3), p_dry=round(p_dry, 3),
                                      prev_mean=(round(float(prv.mean()), 3) if prv is not None else None))
    handles = [Line2D([], [], color="#b8860b", lw=1.6), Line2D([], [], color="#3f7fbf", lw=1.6), Line2D([], [], color="#222", lw=1.0),
               plt.Rectangle((0, 0), 1, 1, color="#000", alpha=0.06)]
    labels = [f"{calendar.month_abbr[int(ym[4:])]} issue (51 members)", f"{calendar.month_abbr[int(prev[4:])]} issue", "normal (hindcast mean)", "normal year-to-year range (±1σ)"]
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 0.012))
    fig.suptitle(f"{label}: SEAS5 forecast of monthly P − E by region (mm/day), {calendar.month_name[int(ym[4:])]} {ym[:4]} issue", x=0.02, y=0.995, ha="left", fontsize=13)
    fig.text(0.02, 1 - 0.42 / fig.get_figheight(), "Each curve is the spread of the 51 members' monthly precipitation minus evaporation, averaged over the region (gold: this issue; blue: last month's). "
             "Negative values are net drying. Black: the model's normal for that month, with its year-to-year range shaded.", fontsize=8.6, color="#444", va="top")
    top = 1 - 0.95 / fig.get_figheight(); bottom = 1.15 / fig.get_figheight()
    fig.subplots_adjust(left=0.06, right=0.99, top=top, bottom=bottom, hspace=0.62, wspace=0.14)
    fig.savefig(out, dpi=105, pil_kwargs={"quality": 84, "method": 6}); plt.close(fig)
    print(f"  wrote {out.name}", flush=True)
    return summary


def build(ym: str) -> None:
    t0 = time.time()
    prev = previous_issues(ym, 1)[0]
    now, before = pme_fields(ym), pme_fields(prev)
    if now is None:
        raise SystemExit("P − E fields for this issue are not on disk")
    lat, lon = now[2], now[3]
    doc = {"generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "issue": ym, "previous": prev, "countries": {}}
    for key, label, polys in (("br", "Brazil", brazil_polygons()), ("co", "Colombia", colombia_polygons())):
        print(f"  {label}: {len(polys)} regions", flush=True)
        weights = {name: cell_weights(poly, lat, lon) for name, poly in polys.items()}
        s_now = region_series(now, weights)
        s_prev = region_series(before, weights) if before is not None else None
        summ = render(key, label, s_now, s_prev, ym, prev, ASSETS / f"seas5_pme_{key}.webp")
        doc["countries"][key] = {"label": label, "file": f"seas5_pme_{key}.webp", "regions": list(polys), "months": summ}
    OUT_JSON.write_text(json.dumps(doc, separators=(",", ":")))
    print(f"wrote {OUT_JSON} in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--issue", required=True)
    build(ap.parse_args().issue)
