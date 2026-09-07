#!/usr/bin/env python3
"""Build for seas5_extremes: threshold-day frequencies as % of the ERA5 normal.

Per region (us: cold days, T min ≤ 0/−10/−20 °C; br: hot days, T max > 35/37/40 °C), per
forecast month, per member: the count of threshold days in each 1° cell from the 6-hourly
extremes chunks, after a per-cell monthly bias shift (SEAS5 hindcast 1993–2016 month mean
minus the ERA5 1993–2016 month mean of the same cell, both on the monthly fields already on
disk / in the local store). Normal = NOAA CPC 1991–2020 daily extremes (0.5°, land,
coarsened to the 1° cells) counted the same way. Outputs:
  assets/sst/data/seas5_extremes.json   population-weighted expected days, normal, % of normal,
                                        member p10/p90, per region × threshold × month
  assets/sst/seas5_xdays_{region}_{thr}_{YYYY_MM}.webp   map: % of normal (hatched where the
                                        normal is under half a day a month)
"""
from __future__ import annotations
import calendar, json, sys, time
from pathlib import Path
import numpy as np
import xarray as xr
sys.path.insert(0, str(Path(__file__).resolve().parent))
from seas5_outlook import ASSETS, DATA, hc_path, previous_issues                  # noqa: E402
from seas5_popT import REGIONS, pop_grid                                            # noqa: E402
from seas5_extremes import xchunk_path, hchunk_path, cpc_path, THRESH, YEARS                    # noqa: E402
from seas5_build import load_field, valid_months                                    # noqa: E402

OUT_JSON = ASSETS / "data" / "seas5_extremes.json"
PCT_LEVELS = [0, 25, 50, 75, 90, 110, 125, 150, 200, 300, 500]
_ERA5_NORMAL: dict = {}


def daily_extremes(region: str, ym: str, k: int):
    """→ (tx[member, day, lat, lon], tn[...], lat, lon) in °C for forecast month k."""
    ds = xr.open_dataset(xchunk_path(region, ym, k), engine="cfgrib", backend_kwargs={"indexpath": ""})
    names = {v: v for v in ds.data_vars}
    tx = ds[[v for v in ds.data_vars if v.startswith("mx")][0]].transpose("number", "step", "latitude", "longitude").values
    tn = ds[[v for v in ds.data_vars if v.startswith("mn")][0]].transpose("number", "step", "latitude", "longitude").values
    lat, lon = ds.latitude.values, ds.longitude.values; ds.close()
    # the CDS serves the 24-hour extremes at daily steps only (one per forecast day)
    return (tx - 273.15).astype(np.float32), (tn - 273.15).astype(np.float32), lat, lon


def cpc_daily(stat: str, region: str, years, lat, lon):
    """CPC daily field on the 1° SEAS5 cells (2×2 box mean of the 0.5° grid) → (vals[time, lat, lon] °C, months[time]), years used."""
    parts, used = [], []
    for y in years:
        p = cpc_path(stat, region, y)
        if not p.exists(): continue
        da = xr.open_dataset(p)[list(xr.open_dataset(p).data_vars)[0]]
        c = da.coarsen(lat=2, lon=2, boundary="trim").mean()
        c = c.sel(lat=xr.DataArray(lat, dims="y"), lon=xr.DataArray(lon, dims="x"), method="nearest")
        parts.append(c); used.append(y)
    if not parts: return None, None, []
    allv = xr.concat(parts, "time")
    return allv.values.astype(np.float32), allv.time.dt.month.values, used


def hindcast_extremes(region: str, ym: str, k: int, stat: str):
    """Hindcast daily Tmin or Tmax for forecast month k → [sample, day, lat, lon] °C (25 members × 24 years)."""
    p = hchunk_path(region, ym[4:], k)
    if not p.exists():
        return None
    ds = xr.open_dataset(p, engine="cfgrib", backend_kwargs={"indexpath": ""})
    v = [x for x in ds.data_vars if x.startswith("mn" if stat == "tn" else "mx")][0]
    da = ds[v]
    dims = [d for d in ("number", "time", "step", "latitude", "longitude") if d in da.dims]
    da = da.transpose(*dims)
    a = da.values.astype(np.float32); ds.close()
    if "time" in dims and "number" in dims:                       # [member, year, day, lat, lon] → [sample, day, lat, lon]
        a = a.reshape(-1, *a.shape[2:])
    return a - 273.15


def quantile_map(fc: np.ndarray, hc: np.ndarray, obs: np.ndarray, nq: int = 41) -> np.ndarray:
    """Per cell: map each forecast value through the hindcast → observed quantile relation.
    fc [member, day, lat, lon]; hc [sample, day, lat, lon]; obs [day', lat, lon] (CPC, same calendar
    month, 1993–2016). Tails beyond the hindcast range carry the end-quantile offset."""
    q = np.linspace(0.01, 0.99, nq)
    hq = np.nanquantile(hc.reshape(-1, *hc.shape[2:]), q, axis=0)         # [q, lat, lon]
    oq = np.nanquantile(obs, q, axis=0)
    out = np.empty_like(fc)
    ny, nx = fc.shape[2], fc.shape[3]
    for iy in range(ny):
        for ix in range(nx):
            h = hq[:, iy, ix]; o = oq[:, iy, ix]
            if not (np.all(np.isfinite(h)) and np.all(np.isfinite(o))):
                out[:, :, iy, ix] = np.nan; continue
            x = fc[:, :, iy, ix]
            out[:, :, iy, ix] = x + np.interp(x, h, o - h)                  # offset at the value's hindcast quantile
    return out


def cpc_normal(region: str, stat: str, lat, lon) -> tuple[dict, list[int]]:
    """Mean monthly count of threshold days per 1° cell over the CPC years on disk → ({thr: [12, lat, lon]}, years)."""
    key = (region, stat)
    if key in _ERA5_NORMAL:
        return _ERA5_NORMAL[key]
    _, _, _, thrs, op = [t for t in THRESH[region] if t[1] == stat][0]
    vals, months, used = cpc_daily(stat, region, YEARS, lat, lon)
    acc = {t: np.full((12, lat.size, lon.size), np.nan) for t in thrs}
    if used:
        for m in range(1, 13):
            sel = months == m; nyr = len(used)
            for t in thrs:
                hit = (vals[sel] <= t) if op == "le" else (vals[sel] >= t)
                acc[t][m - 1] = np.where(np.isfinite(vals[sel]).all(0), hit.sum(0) / nyr, np.nan)
    _ERA5_NORMAL[key] = (acc, used)
    return acc, used


# City labels on the threshold-day maps and the clickable city distributions on the page (user
# 2026-09-07). (name, lat, lon, label side: "r" right of the dot, "l" left).
CITIES = {
    "br": [("Manaus", -3.12, -60.02, "r"), ("Belém", -1.46, -48.50, "r"), ("Fortaleza", -3.72, -38.54, "l"),
           ("Recife", -8.05, -34.88, "l"), ("Salvador", -12.97, -38.51, "l"), ("Brasília", -15.79, -47.88, "r"),
           ("Cuiabá", -15.60, -56.10, "l"), ("Goiânia", -16.68, -49.25, "l"), ("Belo Horizonte", -19.92, -43.94, "r"),
           ("Campo Grande", -20.45, -54.62, "l"), ("Rio de Janeiro", -22.91, -43.17, "r"), ("São Paulo", -23.55, -46.63, "l"),
           ("Curitiba", -25.43, -49.27, "l"), ("Porto Alegre", -30.03, -51.23, "l")],
    "us": [("Seattle", 47.61, -122.33, "r"), ("Portland", 45.52, -122.68, "r"), ("San Francisco", 37.77, -122.42, "r"),
           ("Los Angeles", 34.05, -118.24, "l"), ("Phoenix", 33.45, -112.07, "r"), ("Salt Lake City", 40.76, -111.89, "r"),
           ("Denver", 39.74, -104.99, "r"), ("Dallas", 32.78, -96.80, "r"), ("Houston", 29.76, -95.37, "r"),
           ("Kansas City", 39.10, -94.58, "l"), ("Minneapolis", 44.98, -93.27, "r"), ("St. Louis", 38.63, -90.20, "r"),
           ("Chicago", 41.88, -87.63, "l"), ("Detroit", 42.33, -83.05, "r"), ("Nashville", 36.16, -86.78, "r"),
           ("Atlanta", 33.75, -84.39, "r"), ("Miami", 25.76, -80.19, "r"), ("Washington", 38.90, -77.04, "l", -0.2),
           ("Philadelphia", 39.95, -75.17, "l", 0.35), ("New York", 40.71, -74.01, "r", 0.3), ("Boston", 42.36, -71.06, "r", 0.2),
           ("Toronto", 43.65, -79.38, "l"), ("Montréal", 45.50, -73.57, "r")],
}
MAP_W = 11.0                                            # figure width (in); the axes box is fixed so the page can overlay cities


def map_geometry(region):
    area = REGIONS[region][2]; W = MAP_W; H = W * (area[0] - area[2]) / (area[3] - area[1]) + 2.1
    return area, W, H, [0.03, 0.95 / H, 0.94, (H - 2.1) / H]


def city_fractions(region, lat, lon):
    """Position of a point as fractions of the saved image (x from the left, y from the top)."""
    area, W, H, box = map_geometry(region)
    fx = box[0] + box[2] * (lon - area[1]) / (area[3] - area[1])
    fy = 1.0 - (box[1] + box[3] * (lat - area[2]) / (area[0] - area[2]))
    return round(float(fx), 4), round(float(fy), 4)


_CITY_PX = {}                                            # region -> {name: (x, y)} measured on the drawn axes


def draw_cities(ax, region, pc):
    import matplotlib.patheffects as pe
    halo = [pe.withStroke(linewidth=3.0, foreground="white")]
    if region not in _CITY_PX:                           # measure once from the real transform (cartopy may
        fig = ax.figure; px = {}                         # re-fit the view to the box, so the nominal geometry drifts)
        ax.apply_aspect()                                # the equal-aspect box shrink happens at draw time — apply it now
        for name, la, lo, side in (c[:4] for c in CITIES.get(region, [])):
            x, y = ax.transData.transform((lo, la))
            px[name] = (round(float(x / fig.bbox.width), 4), round(float(1.0 - y / fig.bbox.height), 4))
        _CITY_PX[region] = px
    for city in CITIES.get(region, []):
        name, la, lo, side = city[:4]; dy = city[4] if len(city) > 4 else 0.0      # optional label nudge (deg lat)
        ax.plot(lo, la, "o", ms=5.0, mfc="#111", mec="white", mew=0.9, transform=pc, zorder=6)
        ax.text(lo + (0.55 if side == "r" else -0.55), la + dy, name, fontsize=10.5, fontweight="semibold", color="#111",
                ha="left" if side == "r" else "right", va="center", transform=pc, zorder=6, path_effects=halo)


def pct_map(ratio, lat, lon, normal, region, thr_label, plabel, issue_lbl, out: Path, cold: bool = False):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    import cartopy.crs as ccrs, cartopy.feature as cfeature
    pc = ccrs.PlateCarree(); area, W, H, box = map_geometry(region)
    # cold-day sets: fewer cold days than normal reads warm (red), more reads blue; hot-day sets the reverse
    warm_side = ["#fde4cf", "#f59d68", "#e8703c", "#c8451c", "#8f2a0d"]; cool_side = ["#1f4f8f", "#3672b6", "#8ab6df", "#dbe9f6"]
    cols = (warm_side[::-1] + ["#f4f4f1"] + cool_side[::-1]) if cold else (cool_side + ["#f4f4f1"] + warm_side)
    fig = plt.figure(figsize=(W, H)); ax = fig.add_axes(box, projection=pc)
    ax.set_extent([area[1], area[3], area[2], area[0]], crs=pc)
    ax.add_feature(cfeature.LAND, facecolor="#f4f4f1", zorder=0)
    r = np.where(normal >= 0.5, np.clip(ratio * 100, 0, 499.9), np.nan)
    m = ax.pcolormesh(lon, lat, r, cmap=ListedColormap(cols), norm=BoundaryNorm(PCT_LEVELS, len(cols)), transform=pc, shading="auto", zorder=1)
    thin = np.where(normal < 0.5, 1.0, np.nan)
    ax.contourf(lon, lat, np.ma.masked_invalid(thin), levels=[0.5, 1.5], colors="none", hatches=["////"], transform=pc, zorder=2)
    ax.add_feature(cfeature.OCEAN, facecolor="#ffffff", zorder=2); ax.add_feature(cfeature.LAKES, facecolor="#ffffff", zorder=2)
    ax.coastlines(resolution="50m", linewidth=0.5, color="#333", zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.3, edgecolor="#666", zorder=3)
    ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.2, edgecolor="#999", zorder=3)
    draw_cities(ax, region, pc)
    fig.text(0.03, 1 - 0.14 / H, f"SEAS5 {thr_label} · {plabel} · {issue_lbl}", fontsize=13, fontweight="bold", va="top")
    import textwrap
    fig.text(0.03, 1 - 0.50 / H, "\n".join(textwrap.wrap("Ensemble-mean count of threshold days as % of the CPC 1991–2020 count for the month, on 1° cells; members quantile-mapped per cell from the SEAS5 hindcast onto the CPC record. Hatched: normal under half a day a month.", int(W * 13))),
             fontsize=8.4, color="#444", va="top", linespacing=1.3)
    cax = fig.add_axes([0.25, 0.40 / H, 0.50, 0.14 / H]); cb = fig.colorbar(m, cax=cax, orientation="horizontal", extend="max")
    cb.set_label("% of normal frequency", fontsize=8.5); cb.ax.tick_params(labelsize=7)
    fig.savefig(out, dpi=120, pil_kwargs={"quality": 86, "method": 6}); plt.close(fig)


def build(ym: str) -> dict:
    t0 = time.time()
    doc = {"generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "issue": ym, "regions": {}}
    issue_lbl = f"{calendar.month_name[int(ym[4:])]} {ym[:4]} issue"
    for region, (label, cc, area, unit) in REGIONS.items():
        entry = {"label": label, "sets": {}, "normal_years": None, "cities": {}}
        for name, la, lo, side in (c[:4] for c in CITIES.get(region, [])):
            fx, fy = city_fractions(region, la, lo)
            entry["cities"][name] = {"lat": la, "lon": lo, "x": fx, "y": fy, "side": side, "sets": {}}
        w = None; cache = {}
        for set_key, stat, thr_label, thrs, op in THRESH[region]:
            sub = {"kind": thr_label, "stat": stat, "thresholds": thrs, "months": {}}
            for k in range(1, 7):
                if not xchunk_path(region, ym, k).exists():
                    continue
                if k not in cache:
                    cache.clear(); cache[k] = daily_extremes(region, ym, k)
                tx, tn, lat, lon = cache[k]
                if w is None:
                    w = pop_grid(region, lat, lon); wn = w / w.sum()
                vm = valid_months(ym)[k - 1]; mo = int(vm[5:]); plabel = f"{calendar.month_abbr[mo]} {vm[:4]}"
                raw = tn if stat == "tn" else tx
                hc = hindcast_extremes(region, ym, k, stat)
                ov, om, oyrs = cpc_daily(stat, region, range(1993, 2017), lat, lon)
                if hc is not None and ov is not None and len(oyrs) >= 20:
                    field = quantile_map(raw, hc, ov[om == mo]); corr = "quantile-mapped (hindcast → CPC 1993–2016)"
                    del hc
                else:
                    shift = bias_shift(ym, k, region, lat, lon)
                    field = raw - (shift[None, None] if shift is not None else 0.0); corr = "mean shift" if shift is not None else "uncorrected"
                normal, used = cpc_normal(region, stat, lat, lon); entry["normal_years"] = [min(used), max(used), len(used)] if used else None
                mrec = {}
                ci = [(name, int(np.abs(lat - la).argmin()), int(np.abs(lon - lo).argmin())) for name, la, lo, _ in (c[:4] for c in CITIES.get(region, []))]
                for t in thrs:
                    hit = (field <= t) if op == "le" else (field >= t)
                    cnt = hit.sum(1).astype(np.float32)                                # [member, lat, lon] days
                    for name, i, j in ci:                                                # member counts at the city's cell
                        cs = entry["cities"][name]["sets"].setdefault(set_key, {}).setdefault(vm, {})
                        nv = float(normal[t][mo - 1][i, j]) if used else float("nan")
                        cs[str(t)] = {"m": [int(v) for v in cnt[:, i, j]], "n": (round(nv, 2) if np.isfinite(nv) else None)}
                    pop_days = np.tensordot(cnt, wn, axes=([1, 2], [0, 1]))          # [member]
                    nrm = normal[t][mo - 1] if used else np.full(cnt.shape[1:], np.nan)
                    pop_norm = float(np.nansum(np.where(np.isfinite(nrm), nrm, 0) * wn) / max(np.sum(wn[np.isfinite(nrm)]), 1e-9)) if used else None
                    mean_cnt = cnt.mean(0)
                    out = ASSETS / f"seas5_xdays_{region}_{stat}_{abs(t)}{'m' if t < 0 else ''}_{vm.replace('-', '_')}.webp"
                    if used:
                        ratio = np.where(nrm > 0, mean_cnt / np.maximum(nrm, 1e-6), np.nan)
                        pct_map(ratio, lat, lon, nrm, region, f"{thr_label} {t} °C, % of normal", plabel, issue_lbl, out, cold=(op == "le"))
                    mrec[str(t)] = {"days": round(float(pop_days.mean()), 2), "p10": round(float(np.percentile(pop_days, 10)), 2),
                                    "p90": round(float(np.percentile(pop_days, 90)), 2), "normal": round(pop_norm, 2) if pop_norm is not None else None,
                                    "pct": round(100 * float(pop_days.mean()) / pop_norm, 1) if pop_norm else None,
                                    "map": out.name if used else None, "correction": corr}
                sub["months"][vm] = mrec
                print(f"  {region} {set_key} {vm} [{corr}]: " + ", ".join(f"{t}°C {v['days']:.1f} d (normal {v['normal']}, {v['pct']}%)" for t, v in mrec.items()), flush=True)
            entry["sets"][set_key] = sub
        for name, (fx, fy) in _CITY_PX.get(region, {}).items():
            entry["cities"][name]["x"], entry["cities"][name]["y"] = fx, fy
        doc["regions"][region] = entry
    OUT_JSON.write_text(json.dumps(doc, separators=(",", ":")))
    print(f"wrote {OUT_JSON} in {(time.time() - t0) / 60:.1f} min", flush=True)
    return doc


if __name__ == "__main__":
    import datetime as _dt
    build(sys.argv[1] if len(sys.argv) > 1 else _dt.datetime.utcnow().strftime("%Y%m"))
