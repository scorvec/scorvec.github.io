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

        fig, axes = plt.subplots(2, len(months), figsize=(3.1 * len(months) + 1, 7.6), squeeze=False)
        reg = {"label": label, "unit": unit, "months": {}}
        for col, mon in enumerate(months):
            entry = data[mon]
            for row, mode in enumerate(("abs", "anom")):
                ax = axes[row, col]
                xs_all = []
                for iss, colr, name, z in ((prev, "#3f7fbf", f"{calendar.month_abbr[int(prev[4:])]} issue", 1), (ym, "#b8860b", f"{calendar.month_abbr[int(ym[4:])]} issue", 2)):
                    if iss not in entry:
                        continue
                    daily, clim = entry[iss]
                    if mode == "abs":
                        vals = to_unit(daily, unit)
                    else:
                        if clim is None:
                            continue
                        vals = (daily - clim) * (9 / 5 if unit == "F" else 1.0)
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
                if len(xs_all) == 2:
                    d = xs_all[1].mean() - xs_all[0].mean(); ds_ = xs_all[1].std() - xs_all[0].std()
                    ax.text(0.02, 0.95, f"Δmean {d:+.1f}°{unit if mode == 'abs' else unit}\nΔspread {ds_:+.1f}", transform=ax.transAxes, va="top", fontsize=8.5, color="#333")
                if row == 0:
                    ax.set_title(f"{calendar.month_abbr[int(mon[5:])]} {mon[:4]}", fontsize=12, loc="left")
                ax.set_yticks([]); ax.spines[["left", "top", "right"]].set_visible(False)
                ax.tick_params(labelsize=8.5)
                if mode == "anom":
                    ax.axvline(0, color="#666", lw=0.8)
                if col == 0:
                    ax.set_ylabel("model temperature" if mode == "abs" else "anomaly vs hindcast", fontsize=9.5)
        axes[0, 0].legend(loc="upper right", fontsize=8.5, frameon=False)
        fig.suptitle(f"SEAS5 population-weighted 2 m temperature, {label}: daily values, all 51 members, {calendar.month_name[int(ym[4:])]} {ym[:4]} issue vs {calendar.month_name[int(prev[4:])]}",
                     x=0.02, y=0.99, ha="left", fontsize=13)
        fig.text(0.02, 0.935, f"Each curve pools every member's daily population-weighted mean (≈1,500 values a month; °{unit}). Dashed lines: distribution means. "
                 "Top row is the model's own temperature (SEAS5 bias included, identical for both issues); bottom row removes each issue's hindcast monthly mean at every grid point.",
                 fontsize=8.8, color="#444", va="top", wrap=True)
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
