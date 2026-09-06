#!/usr/bin/env python3
"""Population-weighted 2 m temperature distributions from SEAS5's 6-hourly members.

For the United States (CONUS) and Brazil: every member's 6-hourly 2 m temperature
on a 1° grid, population-weighted into one national series per member, averaged
to daily means, and pooled month by month into a distribution — 51 members ×
~30 days ≈ 1,500 daily values per month. This issue is drawn against the
previous issue for the months both cover, so the reader sees how the whole
distribution moved, not just the mean: a shift, a widening, a fatter warm tail.

Two rows per country. Top: the model's own temperatures (°F for the US, °C for
Brazil), which carry SEAS5's bias but compare cleanly issue to issue. Bottom:
anomalies against the model's 1993–2016 hindcast monthly mean at each grid point
(the monthly hindcast already pulled for the tercile maps), so drift and bias
are removed and the two issues are on the same footing at their different leads.

Population: geonames cities15000 (places ≥ 15,000 people), feature codes for
populated places and capitals only — PPLX "sections of populated place" are
excluded because they duplicate their parent city — summed into the 1° cells
of the CDS grid. National series = Σ pop·T / Σ pop over the country's cells.

Data: seasonal-original-single-levels, 2m_temperature, leadtime_hour 6..4416
in monthly chunks, cached under scripts/sst/data/seas5/sixh/. ~120 MB per
country per issue.

    python scripts/sst/seas5_popT.py fetch [--issue 202609]   # this issue + previous
    python scripts/sst/seas5_popT.py build [--issue 202609]
    python scripts/sst/seas5_popT.py       # both
"""
from __future__ import annotations

import argparse
import calendar
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from seas5_outlook import ASSETS, DATA, CENTRE, SYSTEM, _client, hc_path, previous_issues  # noqa: E402
from seas5_build import valid_months  # noqa: E402

SIXH = DATA / "sixh"
POP_FILE = Path.home() / "c3s" / "data" / "pop" / "cities15000.txt"
OUT_JSON = ASSETS / "data" / "seas5_popT.json"

REGIONS = {
    # key: (label, country code, CDS area [N, W, S, E], display unit)
    "us": ("United States (CONUS)", "US", [50, -125, 25, -66], "F"),
    "br": ("Brazil", "BR", [6, -74, -34, -34], "C"),
}
PPL_CODES = {"PPL", "PPLA", "PPLA2", "PPLA3", "PPLA4", "PPLA5", "PPLC", "PPLG", "PPLS"}
MONTHS = 6
_HC_CACHE: dict = {}


# ── lead hours per forecast month ────────────────────────────────────────────
def month_hours(ym: str, k: int) -> list[str]:
    """6-hourly leadtime hours covering forecast month k (1-based) of an issue
    starting on the 1st: day d's mean uses the 06, 12, 18 and next-00 UTC steps."""
    y, m = int(ym[:4]), int(ym[4:])
    days_before = 0
    for j in range(1, k):
        mm = (m - 1 + j - 1) % 12 + 1; yy = y + (m - 1 + j - 1) // 12
        days_before += calendar.monthrange(yy, mm)[1]
    mm = (m - 1 + k - 1) % 12 + 1; yy = y + (m - 1 + k - 1) // 12
    ndays = calendar.monthrange(yy, mm)[1]
    h0 = days_before * 24
    return [str(h) for h in range(h0 + 6, h0 + ndays * 24 + 1, 6)]


def chunk_path(region: str, ym: str, k: int) -> Path:
    return SIXH / f"{region}_{ym}_m{k}.grib"


def fetch_chunk(region: str, ym: str, k: int) -> bool:
    dest = chunk_path(region, ym, k)
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    hours = month_hours(ym, k)
    if int(hours[-1]) > 5160:                                     # SEAS5 6-hourly runs to 215 days
        hours = [h for h in hours if int(h) <= 5160]
    req = {"originating_centre": CENTRE, "system": SYSTEM, "variable": ["2m_temperature"],
           "year": [ym[:4]], "month": [ym[4:]], "day": ["01"], "leadtime_hour": hours,
           "area": REGIONS[region][2], "grid": [1.0, 1.0], "data_format": "grib"}
    tmp = dest.with_suffix(".part")
    for attempt in range(3):
        t0 = time.time()
        try:
            print(f"  CDS 6-hourly {region} {ym} month {k} ({len(hours)} steps) …", flush=True)
            _client().retrieve("seasonal-original-single-levels", req, str(tmp))
            if tmp.exists() and tmp.stat().st_size > 0:
                os.replace(tmp, dest)
                print(f"    done {dest.stat().st_size / 1e6:.0f} MB in {(time.time() - t0) / 60:.1f} min", flush=True)
                return True
        except Exception as e:                                    # noqa: BLE001
            msg = str(e).replace("\n", " ")
            if "no data" in msg.lower() or "not found" in msg.lower():
                print(f"    {region} {ym} m{k}: no data on the CDS ({msg[:80]})", flush=True)
                return False
            print(f"    {region} {ym} m{k}: attempt {attempt + 1} failed ({msg[:120]})", flush=True)
            time.sleep(30)
    return False


def fetch(ym: str, regions=None, issues=None) -> dict:
    """Region / issue filters exist so several CDS requests can run in parallel
    (one process per region × issue): a month of 6-hourly members is ~140 MB
    and takes the CDS several minutes, and 24 of them in series is hours."""
    got = {}
    for iss in issues or (ym, previous_issues(ym, 1)[0]):
        for region in regions or list(REGIONS):
            for k in range(1, MONTHS + 1):
                got[(iss, region, k)] = fetch_chunk(region, iss, k)
    return got


# ── population grid ──────────────────────────────────────────────────────────
def pop_grid(region: str, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Population per 1° cell of the region's CDS grid, from geonames cities15000."""
    _, cc, area, _ = REGIONS[region]
    grid = np.zeros((lat.size, lon.size))
    n = 0
    with open(POP_FILE, encoding="utf-8") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 15 or f[8] != cc or f[6] != "P" or f[7] not in PPL_CODES:
                continue
            try:
                la, lo, pop = float(f[4]), float(f[5]), int(f[14])
            except ValueError:
                continue
            if not (area[2] <= la <= area[0] and area[1] <= lo <= area[3]):
                continue
            i = int(np.argmin(np.abs(lat - la))); j = int(np.argmin(np.abs(lon - lo)))
            grid[i, j] += pop; n += 1
    print(f"  {region}: {n} places, {grid.sum() / 1e6:.1f} M people on the grid", flush=True)
    return grid


# ── series ───────────────────────────────────────────────────────────────────
def daily_series(region: str, ym: str, k: int, w: np.ndarray | None = None):
    """→ (daily [member, day] in K, lat, lon, w) for forecast month k."""
    import xarray as xr
    ds = xr.open_dataset(chunk_path(region, ym, k), engine="cfgrib", backend_kwargs={"indexpath": ""})
    t = ds["t2m"].transpose("number", "step", "latitude", "longitude").values.astype(np.float32)
    lat, lon = ds.latitude.values, ds.longitude.values
    ds.close()
    if w is None:
        w = pop_grid(region, lat, lon)
    wn = w / w.sum()
    series = np.tensordot(t, wn, axes=([2, 3], [0, 1]))           # [member, step]
    nstep = series.shape[1] - series.shape[1] % 4
    daily = series[:, :nstep].reshape(series.shape[0], -1, 4).mean(axis=2)
    return daily, lat, lon, w


def hindcast_month_mean(ym: str, region: str, k: int, w: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> float | None:
    """Population-weighted hindcast mean 2 m temperature (K) for forecast month k,
    from the monthly hindcast already pulled for the tercile maps (Americas, 1°)."""
    from seas5_build import load_field
    p = hc_path("sfc", ym[4:])
    if not p.exists():
        return None
    if ym[4:] not in _HC_CACHE:                                    # one 500 MB read per start month, not per call
        hc, hlat, hlon = load_field(p, "t2m")
        _HC_CACHE[ym[4:]] = (np.nanmean(hc, axis=0), hlat, hlon)  # [lead, lat, lon]
    clim_all, hlat, hlon = _HC_CACHE[ym[4:]]
    clim = clim_all[k - 1]                                        # [lat, lon]
    ilat = np.array([int(np.argmin(np.abs(hlat - v))) for v in lat])
    ilon = np.array([int(np.argmin(np.abs(hlon - v))) for v in lon])
    sub = clim[np.ix_(ilat, ilon)]
    wn = w / w.sum()
    return float(np.nansum(sub * wn))


def era5_pop_reference(region: str, w: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> dict | None:
    """Population-weighted ERA5 monthly-mean 2 m temperature (K) for the region, by year and
    month, from the 1940–2025 Americas pull (or the 1991–2025 one if that is all we have).
    → {"years": [...], "months": [...], "t": [...], plus per calendar month: normal30, normal10,
    record_warm (value, year), record_cold (value, year), mean9316}."""
    import xarray as xr
    era = Path(__file__).resolve().parent / "data" / "seas5" / "era5"
    src = next((era / f for f in ("era5_am_t2m_long_1940-2025.grib", "era5_am_sfc_1991-2025.grib") if (era / f).exists()), None)
    if src is None:
        return None
    ds = xr.open_dataset(src, engine="cfgrib", backend_kwargs={"indexpath": "", "filter_by_keys": {"shortName": "2t"}})
    da = ds[list(ds.data_vars)[0]].transpose("time", "latitude", "longitude")
    t = da.time.values
    yrs = np.array([int(str(x)[:4]) for x in t]); mos = np.array([int(str(x)[5:7]) for x in t])
    ilat = np.array([int(np.argmin(np.abs(da.latitude.values - v))) for v in lat])
    ilon = np.array([int(np.argmin(np.abs(da.longitude.values - v))) for v in lon])
    sub = da.values[:, ilat[:, None], ilon[None, :]].astype(np.float64)
    ds.close()
    wn = w / w.sum()
    series = np.tensordot(sub, wn, axes=([1, 2], [0, 1]))           # [time] K
    out = {"years": yrs.tolist(), "months": mos.tolist(), "t": series.tolist(), "by_month": {}}
    for m in range(1, 13):
        sel = mos == m
        y, v = yrs[sel], series[sel]
        n30 = v[(y >= 1991) & (y <= 2020)].mean() if ((y >= 1991) & (y <= 2020)).any() else np.nan
        n10 = v[(y >= 2016) & (y <= 2025)].mean() if ((y >= 2016) & (y <= 2025)).any() else np.nan
        m9316 = v[(y >= 1993) & (y <= 2016)].mean() if ((y >= 1993) & (y <= 2016)).any() else np.nan
        iw, ic = int(np.argmax(v)), int(np.argmin(v))
        out["by_month"][m] = dict(normal30=float(n30), normal10=float(n10), mean9316=float(m9316),
                                  record_warm=(float(v[iw]), int(y[iw])), record_cold=(float(v[ic]), int(y[ic])), n_years=int(len(y)))
    return out


def to_unit(k_arr, unit):
    c = np.asarray(k_arr) - 273.15
    return c * 9 / 5 + 32 if unit == "F" else c


# ── build ────────────────────────────────────────────────────────────────────
def build(ym: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    prev = previous_issues(ym, 1)[0]
    vm_now, vm_prev = valid_months(ym), valid_months(prev)
    summary = {"generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "issue": ym, "previous": prev, "regions": {}}
    ASSETS.mkdir(parents=True, exist_ok=True); (ASSETS / "data").mkdir(parents=True, exist_ok=True)

    for region, (label, cc, area, unit) in REGIONS.items():
        # gather: {calendar month: {issue: (daily_values_unit, monthly_means_unit, clim_unit or None)}}
        data = {}
        w = None
        for iss, vm in ((ym, vm_now), (prev, vm_prev)):
            for k in range(1, MONTHS + 1):
                if not chunk_path(region, iss, k).exists():
                    continue
                daily, lat, lon, w = daily_series(region, iss, k, w)
                clim = hindcast_month_mean(iss, region, k, w, lat, lon)
                data.setdefault(vm[k - 1], {})[iss] = (daily, clim)
        months = [m for m in vm_now if m in data and ym in data[m]]
        if not months:
            print(f"  {region}: nothing to draw", flush=True); continue

        ref = era5_pop_reference(region, w, lat, lon) if w is not None else None
        fig, axes = plt.subplots(2, len(months), figsize=(3.1 * len(months) + 1, 7.6), squeeze=False)
        reg = {"label": label, "unit": unit, "months": {}, "observed_space": ref is not None}
        conv = (9 / 5) if unit == "F" else 1.0
        for col, mon in enumerate(months):
            entry = data[mon]; cm = int(mon[5:])
            rm = ref["by_month"][cm] if ref else None
            for row, mode in enumerate(("abs", "anom")):
                ax = axes[row, col]
                xs_all = []
                for iss, colr, name, z in ((prev, "#3f7fbf", f"{calendar.month_abbr[int(prev[4:])]} issue", 1), (ym, "#b8860b", f"{calendar.month_abbr[int(ym[4:])]} issue", 2)):
                    if iss not in entry:
                        continue
                    daily, clim = entry[iss]
                    if clim is None and ref is not None:
                        continue
                    if ref is not None:
                        # observed space: member − hindcast month mean + ERA5 1993–2016 month mean (all population-weighted)
                        corrected = daily - clim + rm["mean9316"]
                        vals = to_unit(corrected, unit) if mode == "abs" else (corrected - rm["normal30"]) * conv
                    else:
                        if mode == "abs":
                            vals = to_unit(daily, unit)
                        else:
                            if clim is None:
                                continue
                            vals = (daily - clim) * conv
                    flat = vals.ravel()
                    xs_all.append(flat)
                    kde = gaussian_kde(flat)
                    lo, hi = np.percentile(flat, [0.2, 99.8]); pad = 0.1 * (hi - lo)
                    x = np.linspace(lo - pad, hi + pad, 240)
                    ax.fill_between(x, kde(x), color=colr, alpha=0.22, zorder=z)
                    ax.plot(x, kde(x), color=colr, lw=1.6, label=name, zorder=z + 2)
                    ax.axvline(flat.mean(), color=colr, lw=1.1, ls="--", zorder=z + 2)
                    mm = vals.mean(axis=1)                        # monthly mean per member
                    key = f"{mode}_{iss}"
                    reg["months"].setdefault(mon, {})[key] = dict(mean=round(float(flat.mean()), 2), p10=round(float(np.percentile(flat, 10)), 2),
                                                                   p50=round(float(np.percentile(flat, 50)), 2), p90=round(float(np.percentile(flat, 90)), 2),
                                                                   sd=round(float(flat.std()), 2), member_monthly_p10=round(float(np.percentile(mm, 10)), 2),
                                                                   member_monthly_p90=round(float(np.percentile(mm, 90)), 2))
                # observed normals and records as vertical marks (observed space only)
                if ref is not None:
                    if mode == "abs":
                        n30, n10 = to_unit(rm["normal30"], unit), to_unit(rm["normal10"], unit)
                        rw, rc = to_unit(rm["record_warm"][0], unit), to_unit(rm["record_cold"][0], unit)
                    else:
                        n30, n10 = 0.0, (rm["normal10"] - rm["normal30"]) * conv
                        rw, rc = (rm["record_warm"][0] - rm["normal30"]) * conv, (rm["record_cold"][0] - rm["normal30"]) * conv
                    ax.axvline(n30, color="#222", lw=1.0, zorder=4)
                    ax.axvline(n10, color="#222", lw=1.0, ls=(0, (4, 3)), zorder=4)
                    ax.axvline(rw, color="#b0352a", lw=0.9, ls=":", zorder=4)
                    ax.axvline(rc, color="#2a5da8", lw=0.9, ls=":", zorder=4)
                    ylim = ax.get_ylim()[1]
                    ax.text(rw, ylim * 0.98, f"{rm['record_warm'][1]}", color="#b0352a", fontsize=7, ha="center", va="top")
                    ax.text(rc, ylim * 0.98, f"{rm['record_cold'][1]}", color="#2a5da8", fontsize=7, ha="center", va="top")
                    reg["months"].setdefault(mon, {})["reference"] = dict(normal30=round(float(to_unit(rm["normal30"], unit)), 2), normal10=round(float(to_unit(rm["normal10"], unit)), 2),
                                                                        record_warm=[round(float(to_unit(rm["record_warm"][0], unit)), 2), rm["record_warm"][1]],
                                                                        record_cold=[round(float(to_unit(rm["record_cold"][0], unit)), 2), rm["record_cold"][1]], n_years=rm["n_years"])
                if len(xs_all) == 2:
                    d = xs_all[1].mean() - xs_all[0].mean(); ds_ = xs_all[1].std() - xs_all[0].std()
                    ax.text(0.02, 0.95, f"Δmean {d:+.1f}°{unit}\nΔspread {ds_:+.1f}", transform=ax.transAxes, va="top", fontsize=8.5, color="#333")
                if row == 0:
                    ax.set_title(f"{calendar.month_abbr[int(mon[5:])]} {mon[:4]}", fontsize=12, loc="left")
                ax.set_yticks([]); ax.spines[["left", "top", "right"]].set_visible(False)
                ax.tick_params(labelsize=8.5)
                if mode == "anom" and ref is None:
                    ax.axvline(0, color="#666", lw=0.8)
                if col == 0:
                    ax.set_ylabel(("temperature, observed space" if ref else "model temperature") if mode == "abs" else ("anomaly vs 1991–2020" if ref else "anomaly vs hindcast"), fontsize=9.5)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if ref is not None:
            from matplotlib.lines import Line2D
            handles += [Line2D([], [], color="#222", lw=1.0), Line2D([], [], color="#222", lw=1.0, ls=(0, (4, 3))), Line2D([], [], color="#b0352a", lw=0.9, ls=":"), Line2D([], [], color="#2a5da8", lw=0.9, ls=":")]
            labels += ["1991–2020 normal", "2016–2025 normal", "record warm month", "record cold month"]
        axes[0, 0].legend(handles, labels, loc="upper right", fontsize=8, frameon=False)
        fig.suptitle(f"SEAS5 population-weighted 2 m temperature, {label}: daily values, all 51 members, {calendar.month_name[int(ym[4:])]} {ym[:4]} issue vs {calendar.month_name[int(prev[4:])]}",
                     x=0.02, y=0.99, ha="left", fontsize=13)
        if ref is not None:
            nyr = ref["by_month"][int(months[0][5:])]["n_years"]
            note = (f"Each curve pools every member's daily population-weighted mean (≈1,500 values a month; °{unit}), bias-corrected into observed space "
                    f"(member − hindcast month mean + ERA5 1993–2016 month mean). Dashed coloured lines: distribution means. Black: ERA5 population-weighted monthly normals, "
                    f"1991–2020 (solid) and 2016–2025 (dashed); dotted: warmest and coldest month of the ERA5 record ({nyr} years), year labelled. Bottom row: the same, as anomalies from the 1991–2020 normal.")
        else:
            note = (f"Each curve pools every member's daily population-weighted mean (≈1,500 values a month; °{unit}). Dashed lines: distribution means. "
                    "Top row is the model's own temperature (SEAS5 bias included, identical for both issues); bottom row removes each issue's hindcast monthly mean at every grid point.")
        fig.text(0.02, 0.935, note, fontsize=8.8, color="#444", va="top", wrap=True)
        fig.subplots_adjust(left=0.05, right=0.99, top=0.85, bottom=0.07, hspace=0.35, wspace=0.12)
        out = ASSETS / f"seas5_popT_{region}.webp"
        fig.savefig(out, dpi=110, pil_kwargs={"quality": 84, "method": 6}); plt.close(fig)
        reg["file"] = out.name
        summary["regions"][region] = reg
        print(f"  wrote {out.name}", flush=True)
    OUT_JSON.write_text(json.dumps(summary, separators=(",", ":")))
    print(f"wrote {OUT_JSON}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="all", choices=["fetch", "build", "all"])
    ap.add_argument("--issue", default=None)
    ap.add_argument("--region", choices=list(REGIONS), help="fetch only this region (parallel workers)")
    ap.add_argument("--only-issue", help="fetch only this issue YYYYMM (parallel workers)")
    a = ap.parse_args(argv)
    import datetime as _dt
    ym = a.issue or _dt.datetime.utcnow().strftime("%Y%m")
    if a.cmd in ("fetch", "all"):
        got = fetch(ym, [a.region] if a.region else None, [a.only_issue] if a.only_issue else None)
        bad = [k for k, v in got.items() if not v]
        print(f"6-hourly fetch {ym}: {len(got) - len(bad)} ok, {len(bad)} missing", flush=True)
    if a.cmd in ("build", "all"):
        build(ym)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
