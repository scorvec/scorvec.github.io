#!/usr/bin/env python3
"""SEAS5 cold-snap (and heat-wave) RUN risk over a whole season — the convex-risk product
(user 2026-09-07: "p(3 day period or greater of >2 sigma cold anomaly) ... over the full NDJF period
(as cold days could straddle months) ... compare this to normal").

Per member and 1° cell, the daily-mean temperature proxy (Tmax + Tmin)/2 from the quantile-mapped
SEAS5 extremes (seas5_extremes_build) is standardised against the CPC 1991–2020 day-of-year mean and
σ (31-day smoothed), then scanned for runs of ≥ RUN consecutive days at or beyond −SIG σ (US cold) /
+SIG σ (Brazil heat) across the contiguous season window (US: Nov–Feb, Brazil: Dec–Feb) — so a snap
straddling a month boundary counts once. Statistics: P(at least one run), the expected number of
run-days; normal = the same statistics over the CPC record's own seasons (1991/92–2019/20).
Outputs assets/sst/seas5_runs_{region}_{sig}_{prob|pct|days}.webp and data/seas5_runs.json (with
per-city member run-day counts for the page's clickable cities).

Derived tables (seas5_ref.py, 2026-10-04): the quantile mapping reads hc_xdays_{region}_{MM} and
obs_xdays_cpc_{region} (seas5_extremes_build); the CPC day-of-year mean/σ and the observed seasons'
run statistics (the normal) come from obs_runs_cpc_{region}, derived once from the local CPC store —
one set per possible season length (with a 28- or 29-day February), because the observed seasons are
cut to the forecast's day count.

    python seas5_runs.py [--issue 202609]
"""
from __future__ import annotations
import argparse, calendar, json, sys, time
from pathlib import Path
import numpy as np, xarray as xr
sys.path.insert(0, str(Path(__file__).resolve().parent))
from seas5_outlook import ASSETS                                                                  # noqa: E402
from seas5_extremes import xchunk_path, cpc_path, YEARS                                            # noqa: E402
from seas5_extremes_build import daily_extremes, mapped, CITIES, map_geometry, _CITY_PX, draw_cities   # noqa: E402
from seas5_build import valid_months                                                              # noqa: E402
from seas5_popT import REGIONS, region_grid                                                       # noqa: E402
import seas5_ref as R                                                                             # noqa: E402

SIG = [1.5, 2.0, 2.5]
RUN = 3
SEASON = {"us": ("cold", -1, [11, 12, 1, 2], "Nov–Feb", "cold snaps"), "br": ("heat", +1, [12, 1, 2], "Dec–Feb", "heat waves")}
OUT_JSON = ASSETS / "data" / "seas5_runs.json"
PROB_LEVELS = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100.01]
PROB_COLORS = ["#f4f4f1", "#e3ecf5", "#c9dcee", "#a9c8e4", "#86b1d8", "#6197c8", "#437bb3", "#2f629c", "#214b82", "#163665", "#0c224a"]
PCT_LEVELS = [0, 25, 50, 75, 90, 110, 125, 150, 200, 300, 400]
PCT_COLORS_COLD = ["#8f2a0d", "#c8451c", "#e8703c", "#fbc39c", "#f4f4f1", "#dbe9f6", "#b7d2ec", "#8ab6df", "#5c95cd", "#3672b6"]   # more cold runs = blue
PCT_COLORS_HEAT = PCT_COLORS_COLD[::-1]


def cpc_tmean(region, lat, lon):
    """CPC daily-mean proxy (Tmax+Tmin)/2 on the 1° cells with its dates, 1991–2020."""
    parts = []
    for y in YEARS:
        px, pn = cpc_path("tx", region, y), cpc_path("tn", region, y)
        if not (px.exists() and pn.exists()): continue
        a = xr.open_dataset(px); b = xr.open_dataset(pn)
        da = (a[list(a.data_vars)[0]] + b[list(b.data_vars)[0]]) / 2.0
        c = da.coarsen(lat=2, lon=2, boundary="trim").mean().sel(lat=xr.DataArray(lat, dims="y"), lon=xr.DataArray(lon, dims="x"), method="nearest")
        parts.append(c)
    allv = xr.concat(parts, "time")
    return allv.values.astype(np.float32), allv.time.values


def doy_clim(vals, dates):
    """Day-of-year mean and σ (31-day circular smoothing) → (clim[366, lat, lon], sd[366, lat, lon])."""
    doy = ((dates.astype("datetime64[D]") - dates.astype("datetime64[Y]")).astype(int)) + 1
    m = np.full((366,) + vals.shape[1:], np.nan, np.float32); s = m.copy()
    for d in range(1, 367):
        sel = doy == d
        if sel.sum() >= 5:
            m[d - 1] = np.nanmean(vals[sel], 0); s[d - 1] = np.nanstd(vals[sel], 0)
    m[365] = np.where(np.isfinite(m[365]), m[365], m[364]); s[365] = np.where(np.isfinite(s[365]), s[365], s[364])
    k = 15; idx = np.arange(366)
    ms = np.stack([np.nanmean(m[(idx[i] + np.arange(-k, k + 1)) % 366], 0) for i in range(366)]); ss = np.stack([np.nanmean(s[(idx[i] + np.arange(-k, k + 1)) % 366], 0) for i in range(366)])
    return ms, ss


def run_stats(z, sign, thr, run=RUN):
    """z [sample, day, lat, lon] → (nruns, rundays) per sample and cell for runs of ≥ run days beyond thr."""
    hit = (sign * z) >= thr
    n, D = hit.shape[0], hit.shape[1]
    cur = np.zeros((n,) + hit.shape[2:], np.int16); nruns = np.zeros_like(cur); rdays = np.zeros_like(cur)
    for d in range(D):
        h = hit[:, d]
        ended = (~h) & (cur >= run)
        nruns += ended; rdays += np.where(ended, cur, 0)
        cur = np.where(h, cur + 1, 0)
    ended = cur >= run; nruns += ended; rdays += np.where(ended, cur, 0)
    return nruns, rdays


def observed_seasons(oz, odates, smonths, ndays):
    """The CPC record's own seasons on the calendar window, each cut to the forecast's `ndays`
    → [season, day, lat, lon] (seasons shorter than ndays are dropped, as the build always did)."""
    oyears = odates.astype("datetime64[Y]").astype(int) + 1970; omon = odates.astype("datetime64[M]").astype(int) % 12 + 1
    first = smonths[0]; seasons = []
    for y in range(YEARS[0], YEARS[-1]):
        sel = ((oyears == y) & (omon >= first)) | ((oyears == y + 1) & (omon <= smonths[-1]) & (omon < first))
        if sel.sum() >= 80: seasons.append(oz[sel][:ndays] if sel.sum() >= ndays else None)
    seasons = [s for s in seasons if s is not None]
    D = min(s.shape[0] for s in seasons)
    return np.stack([s[:D] for s in seasons])


def season_lengths(smonths):
    """Possible day counts of the season window: February with 28 or 29 days."""
    base = sum(calendar.monthrange(2001, m)[1] for m in smonths)
    return sorted({base, base + (1 if 2 in smonths else 0)})


def derive_obs(region: str) -> None:
    """obs_runs_cpc_{region}: CPC 1991–2020 day-of-year mean/σ on the region's 1° grid and, per possible
    season length and σ threshold, the observed seasons' run counts and run-days per cell."""
    kind, sign, smonths, slabel, what = SEASON[region]
    lat, lon = region_grid(region)
    obs, odates = cpc_tmean(region, lat, lon); clim, sd = doy_clim(obs, odates)
    odoy = ((odates.astype("datetime64[D]") - odates.astype("datetime64[Y]")).astype(int)) + 1
    oz = (obs - clim[odoy - 1]) / np.where(sd[odoy - 1] > 0.5, sd[odoy - 1], np.nan)
    arrs = {"lat": lat, "lon": lon, "clim": clim, "sd": sd}
    for nd in season_lengths(smonths):
        oz_s = observed_seasons(oz, odates, smonths, nd)
        arrs[f"nseasons_{nd}"] = np.array(oz_s.shape[0])
        for T in SIG:
            onr, ord_ = run_stats(oz_s, sign, T)
            key = f"{T:g}".replace(".", "p")
            arrs[f"onr_{nd}_{key}"], arrs[f"ord_{nd}_{key}"] = onr, ord_
    R.save(R.obs_ref(f"runs_cpc_{region}"), meta={"source": "NOAA CPC Global Unified daily Tmax/Tmin (PSL), (Tmax+Tmin)/2, 1991-2020",
                                                   "season": slabel, "run_days": RUN, "sigmas": SIG}, **arrs)


OBS_UNITS = {f"runs_cpc_{r}": (lambda r=r: derive_obs(r)) for r in SEASON}


def draw(field, lat, lon, region, levels, colors, label, title, sub, out, extend="neither", hatch=None):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt, textwrap
    from matplotlib.colors import BoundaryNorm, ListedColormap
    import cartopy.crs as ccrs, cartopy.feature as cfeature
    pc = ccrs.PlateCarree(); area, W, H, box = map_geometry(region)
    fig = plt.figure(figsize=(W, H)); ax = fig.add_axes(box, projection=pc)
    ax.set_extent([area[1], area[3], area[2], area[0]], crs=pc)
    ax.add_feature(cfeature.LAND, facecolor="#f4f4f1", zorder=0)
    m = ax.pcolormesh(lon, lat, field, cmap=ListedColormap(colors), norm=BoundaryNorm(levels, len(colors)), transform=pc, shading="auto", zorder=1)
    if hatch is not None:
        ax.contourf(lon, lat, np.ma.masked_invalid(hatch), levels=[0.5, 1.5], colors="none", hatches=["////"], transform=pc, zorder=2)
    ax.add_feature(cfeature.OCEAN, facecolor="#ffffff", zorder=2); ax.add_feature(cfeature.LAKES, facecolor="#ffffff", zorder=2)
    ax.coastlines(resolution="50m", linewidth=0.5, color="#333", zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.3, edgecolor="#666", zorder=3)
    ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.2, edgecolor="#6f6b64", zorder=3)
    draw_cities(ax, region, pc)
    fig.text(0.03, 1 - 0.14 / H, title, fontsize=13, fontweight="bold", va="top")
    fig.text(0.03, 1 - 0.50 / H, "\n".join(textwrap.wrap(sub, int(W * 13))), fontsize=8.4, color="#444", va="top", linespacing=1.3)
    cax = fig.add_axes([0.25, 0.40 / H, 0.50, 0.14 / H]); cb = fig.colorbar(m, cax=cax, orientation="horizontal", extend=extend)
    cb.set_label(label, fontsize=8.5); cb.ax.tick_params(labelsize=7)
    fig.savefig(out, dpi=120, pil_kwargs={"quality": 86, "method": 6}); plt.close(fig)


def build(ym: str) -> dict:
    t0 = time.time(); vms = valid_months(ym)
    issue_lbl = f"{calendar.month_name[int(ym[4:])]} {ym[:4]} issue"
    doc = {"generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "issue": ym, "run_days": RUN, "sigmas": SIG, "regions": {}}
    for region, (kind, sign, smonths, slabel, what) in SEASON.items():
        ks = [k for k, vm in enumerate(vms, start=1) if int(vm[5:]) in smonths and xchunk_path(region, ym, k).exists()]
        if len(ks) < len(smonths):
            print(f"  {region}: season {slabel} not fully covered by this issue ({len(ks)}/{len(smonths)} months) — skipped", flush=True); continue
        tab = R.load(R.obs_ref(f"runs_cpc_{region}"))
        if tab is None:
            print(f"  {region}: obs_runs_cpc_{region} missing — skipped", flush=True); continue
        fields, lat = [], None; dates = []
        for k in ks:
            tx, tn, lat, lon = daily_extremes(region, ym, k); vm = vms[k - 1]; mo = int(vm[5:])
            parts = {}
            for stat, raw in (("tx", tx), ("tn", tn)):
                parts[stat] = mapped(region, ym, k, stat, raw, lat, lon, mo)
            if parts["tx"] is None or parts["tn"] is None:
                break
            fields.append((parts["tx"] + parts["tn"]) / 2.0)
            nd = fields[-1].shape[1]
            dates += [np.datetime64(f"{vm}-01") + np.timedelta64(i, "D") for i in range(nd)]
        if len(fields) < len(ks):
            print(f"  {region}: quantile tables missing for this start month — skipped", flush=True); continue
        tm = np.concatenate(fields, axis=1); del fields                              # [member, day, lat, lon]
        dates = np.array(dates, dtype="datetime64[D]")
        if not (np.allclose(tab["lat"], lat) and np.allclose(tab["lon"], lon)):
            raise ValueError(f"{region}: chunk grid differs from obs_runs_cpc_{region}")
        clim, sd = tab["clim"], tab["sd"]
        doy = ((dates - dates.astype("datetime64[Y]")).astype(int)) + 1
        z = (tm - clim[doy - 1][None]) / np.where(sd[doy - 1] > 0.5, sd[doy - 1], np.nan)[None]
        # normal: the CPC record's own seasons on the same calendar window, cut to this forecast's day count
        nd = tm.shape[1]
        if f"nseasons_{nd}" not in tab:
            print(f"  {region}: no observed-season table for a {nd}-day window — skipped", flush=True); continue
        nseasons = int(tab[f"nseasons_{nd}"])
        entry = {"label": REGIONS[region][0], "kind": kind, "season": slabel, "months": [vms[k - 1] for k in ks], "normal_seasons": nseasons, "sigmas": {}, "cities": {}}
        ci = [(c[0], int(np.abs(lat - c[1]).argmin()), int(np.abs(lon - c[2]).argmin())) for c in CITIES.get(region, [])]
        for T in SIG:
            nr, rd = run_stats(z, sign, T)
            onr, ord_ = tab[f"onr_{nd}_{f'{T:g}'.replace('.', 'p')}"], tab[f"ord_{nd}_{f'{T:g}'.replace('.', 'p')}"]
            p = (nr >= 1).mean(0); po = (onr >= 1).mean(0); ed = rd.mean(0); eod = ord_.mean(0)
            key = f"{T:g}".replace(".", "p")
            sub = (f"Runs of ≥ {RUN} consecutive days with the daily-mean temperature proxy ((Tmax+Tmin)/2, members quantile-mapped from the SEAS5 hindcast onto CPC) "
                   f"{'at or below −' if sign < 0 else 'at or above +'}{T:g} σ of the CPC 1991–2020 day-of-year climatology, anywhere in {slabel} ({', '.join(entry['months'])}); "
                   f"normal = the same statistic over the {nseasons} CPC seasons. {issue_lbl}.")
            f1 = ASSETS / f"seas5_runs_{region}_{key}_prob.webp"
            draw(100 * p, lat, lon, region, PROB_LEVELS, PROB_COLORS, "probability (%)", f"SEAS5 · chance of a ≥ {RUN}-day {what[:-1]} beyond {T:g} σ · {slabel} {ym[:4]}–{str(int(ym[:4]) + 1)[2:]}", sub, f1)
            r = np.where(po >= 0.05, np.clip(100 * p / np.maximum(po, 1e-6), 0, 9999), np.nan)
            f2 = ASSETS / f"seas5_runs_{region}_{key}_pct.webp"
            draw(r, lat, lon, region, PCT_LEVELS, PCT_COLORS_COLD if sign < 0 else PCT_COLORS_HEAT, "% of the normal chance", f"SEAS5 · chance of a ≥ {RUN}-day {what[:-1]} beyond {T:g} σ, % of normal · {slabel}", sub, f2, extend="max", hatch=np.where(po < 0.05, 1.0, np.nan))
            rr = np.where(eod >= 0.5, np.clip(100 * ed / np.maximum(eod, 1e-6), 0, 9999), np.nan)
            f3 = ASSETS / f"seas5_runs_{region}_{key}_days.webp"
            draw(rr, lat, lon, region, PCT_LEVELS, PCT_COLORS_COLD if sign < 0 else PCT_COLORS_HEAT, "% of the normal run-days", f"SEAS5 · days inside ≥ {RUN}-day runs beyond {T:g} σ, % of normal · {slabel}", sub, f3, extend="max", hatch=np.where(eod < 0.5, 1.0, np.nan))
            entry["sigmas"][f"{T:g}"] = {"prob": f1.name, "pct": f2.name, "days": f3.name, "p_mean": round(float(np.nanmean(p)), 3), "po_mean": round(float(np.nanmean(po)), 3)}
            for name, i, j in ci:
                c = entry["cities"].setdefault(name, {"x": None, "y": None, "sigmas": {}})
                c["sigmas"][f"{T:g}"] = {"nruns": [int(v) for v in nr[:, i, j]], "rundays": [int(v) for v in rd[:, i, j]],
                                          "obs_nruns": [int(v) for v in onr[:, i, j]], "obs_rundays": [int(v) for v in ord_[:, i, j]]}
            print(f"  {region} {T:g}σ: P(run) field mean {100 * np.nanmean(p):.0f}% vs normal {100 * np.nanmean(po):.0f}%; run-days {np.nanmean(ed):.1f} vs {np.nanmean(eod):.1f}", flush=True)
        for name, (fx, fy) in _CITY_PX.get(region, {}).items():
            if name in entry["cities"]: entry["cities"][name]["x"], entry["cities"][name]["y"] = fx, fy
        for c in CITIES.get(region, []):
            entry["cities"][c[0]]["side"] = c[3]
        doc["regions"][region] = entry
        del tm, z
    OUT_JSON.write_text(json.dumps(doc, separators=(",", ":")))
    print(f"wrote {OUT_JSON} in {(time.time() - t0) / 60:.1f} min", flush=True)
    return doc


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--issue", default=None); a = ap.parse_args()
    import datetime as _dt
    build(a.issue or _dt.datetime.utcnow().strftime("%Y%m"))
