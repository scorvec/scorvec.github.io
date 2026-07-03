#!/usr/bin/env python3
"""C3S multi-model ENSO seasonal forecast — Niño-3.4 and relative Niño-3.4.

The Copernicus Climate Change Service (C3S) multi-system seasonal ensemble, shown
as a member plume plus one mean trajectory per originating centre, two indices ×
two smoothings:

  Niño-3.4  — the traditional Niño-3.4 SST anomaly. Its 3-month running mean is the
              ONI.
  rNiño-3.4 — the Relative Niño-3.4 anomaly (CPC/ECMWF): the Niño-3.4 anomaly minus
              the 20°S–20°N tropical-mean anomaly, rescaled per calendar month by
              σ(ONI)/σ(relative) so it stays in °C on ONI's thresholds. Its 3-month
              running mean is the RONI. This removes the background tropical warming
              that inflates the traditional index.

Each is drawn monthly (left) and as the 3-month running mean (right) → a 2×2 grid.
Every panel shows the full spread of ALL members from ALL models (shaded plume:
total range + 10–90% + 25–75%), each model's ensemble-mean line, and the bold
multi-model mean. Each model is anomalized against its own 1993–2016 hindcast
(drift/bias removed), putting all centres on one scale.

To keep every centre on the SAME initialization month, the issue is resolved by
quorum: the most recent month for which at least QUORUM models have published (so
the chart never flips to a new month on the back of a single early centre, and
never silently mixes issues).

Two area-subsets are pulled per model: the small Niño-3.4 box at FINE resolution
(1°, 11×51 cells) and the broad 20°S–20°N band at COARSE resolution (5°) for the
tropical mean — so the headline index is properly sampled while the band average,
where 5° is ample, keeps the lagged burst ensembles out of the big-request MARS
queue and within memory. Each model's hindcast climatology (Niño-3.4 and tropical
mean, per start-month, per lead) is cached in c3s_nino34_clim.csv (committed), so
only a never-seen start-month pays the reforecast download. The RONI rescale reuses
roni_sigma.json (the site's OISST-derived σ(ONI)/σ(relative), CPC/ECMWF method).

Data: CDS `seasonal-monthly-single-levels`, sea_surface_temperature, monthly_mean.
Requires ~/.cdsapirc (or CDSAPI_URL/CDSAPI_KEY).

  python c3s_nino34.py                 # latest issue (by quorum), all models
  python c3s_nino34.py --issue 202606  # specific issue
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

HERE = Path(__file__).resolve().parent
SITE_ROOT = Path(os.environ["SST_SITE_ROOT"]).resolve() if os.environ.get("SST_SITE_ROOT") else HERE.parents[1]
ASSETS = SITE_ROOT / "assets" / "sst"
DATA = HERE / "data" / "c3s"
CLIM_PATH = HERE / "c3s_nino34_clim.csv"         # committed cache (per model, start-month, lead) — v2 dual-res
SIGMA_PATH = HERE / "roni_sigma.json"            # RONI σ(ONI)/σ(relative) per calendar month

DATASET = "seasonal-monthly-single-levels"
# TWO area/grid requests per model, so each index is sampled at the right scale:
#   Niño-3.4  — a small box (10°×50°); pulled FINE (1°, 11×51 = 561 cells). At 5°
#               the box collapses to 3 latitude rows (−5/0/+5) × 11 lon = 33 cells,
#               too coarse for the headline index — hence the dedicated fine pull.
#   tropical  — the 20°S–20°N band (for the relative index's tropical mean); pulled
#               COARSE (5°). A 40°-wide band mean is a heavy area-average, so 5° is
#               ample, and it keeps the lagged burst ensembles (NCEP/UKMO/BoM), whose
#               (number × init × step) hypercube would peak ~22 GB RAM at 1° over the
#               whole band, down to ~6 GB. Resolution bias cancels in the
#               forecast−hindcast anomaly (both sampled identically). The two requests
#               share an identical member/init/step structure, so the per-member
#               Niño-3.4 (fine) and tropical mean (coarse) pair element-for-element.
AREA_N34,  GRID_N34  = [5, -170, -5, -120], [1.0, 1.0]     # Niño-3.4 box, fine
AREA_TROP, GRID_TROP = [20, -180, -20, 180], [5.0, 5.0]    # 20°S–20°N band, coarse
LEADS = ["1", "2", "3", "4", "5", "6"]
MAXLEAD = 6
CLIM_YEARS = [str(y) for y in range(1993, 2017)]  # C3S common hindcast period
N34_LAT, N34_LON = (-5, 5), (-170, -120)          # Niño-3.4 in −180..180 longitude
TROP_LAT = (-20, 20)                              # tropical mean, all longitudes
QUORUM = 5                                        # models needed to accept an issue month

# (centre, system, label, colour) — C3S centres with current Niño-3.4 data on the
# CDS. CMCC (35) and JMA have no recent data as of 2026-07 and are omitted; add
# them back here if/when they return.
MODELS = [
    ("ecmwf",        "51",  "ECMWF SEAS5",    "#d62728"),
    ("ukmo",         "610", "UKMO GloSea6",   "#1f77b4"),
    ("meteo_france", "9",   "Météo-France 9", "#2ca02c"),
    ("dwd",          "22",  "DWD GCFS2",      "#9467bd"),
    ("ncep",         "2",   "NCEP CFSv2",     "#ff7f0e"),
    ("eccc",         "4",   "ECCC CanSIPS",   "#8c564b"),
    ("bom",          "2",   "BoM ACCESS-S2",  "#17becf"),
]


def _client():
    import cdsapi
    return cdsapi.Client(timeout=600, quiet=True, progress=False,
                         wait_until_complete=True, retry_max=1)


def _retrieve(centre, system, years, month, dest: Path, area, grid) -> bool:
    """Area-subset SST retrieval → dest. False on MarsNoData / failure (skip)."""
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = {
        "originating_centre": centre, "system": system,
        "variable": "sea_surface_temperature", "product_type": "monthly_mean",
        "year": list(years), "month": month, "leadtime_month": LEADS,
        "area": area, "grid": grid, "data_format": "grib",
    }
    for attempt in range(2):
        try:
            _client().retrieve(DATASET, req, str(dest))
            return dest.exists() and dest.stat().st_size > 0
        except Exception as e:
            msg = str(e).replace("\n", " ")
            if "no data" in msg.lower():
                print(f"    {centre}/{system} {month}: no data on CDS — skipping", file=sys.stderr)
                return False
            print(f"    {centre}/{system} {month}: attempt {attempt+1} failed ({msg[:90]})", file=sys.stderr)
            time.sleep(8)
    return False


def _wmean(da, lat_rng, lon_rng=None):
    """Cosine-latitude-weighted mean over a lat (and optional lon) box, skipna."""
    m = (da["latitude"] >= lat_rng[0]) & (da["latitude"] <= lat_rng[1])
    sub = da.where(m)
    if lon_rng is not None:
        ml = (da["longitude"] >= lon_rng[0]) & (da["longitude"] <= lon_rng[1])
        sub = sub.where(ml)
    w = np.cos(np.deg2rad(da["latitude"]))
    return sub.weighted(w).mean(("latitude", "longitude"), skipna=True)


def _box_series(path: Path, lat_rng, lon_rng):
    """(box_mean_flat, valid_month_flat) for one GRIB — every (member × init × step)
    cell of a cosine-weighted box mean, raveled. Model-agnostic: clean monthly-mean
    files and lagged burst ensembles reduce identically."""
    ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})
    da = ds["sst"] - 273.15
    box = _wmean(da, lat_rng, lon_rng)
    box_b, vt_b = xr.broadcast(box, da["valid_time"])
    vals = np.asarray(box_b.values).ravel()
    vtm = pd.to_datetime(np.asarray(vt_b.values).ravel()).month
    ds.close()
    return vals, vtm


def _members_by_lead(n34_path: Path, trop_path: Path, start_month: int):
    """{lead(1..MAXLEAD) → (n34_arr, trop_arr)} in °C — PAIRED, element-aligned.

    Niño-3.4 comes from the FINE (1°) box file, the tropical mean from the COARSE
    (5°) band file. Both requests share an identical (member × init × step)
    structure, so their raveled series align cell-for-cell (asserted). Pool every
    finite cell by the calendar month of its valid_time; one shared finite mask
    keeps n34/trop paired (needed for the per-member relative index). lead = months
    after start_month (mod-12)."""
    n34_f, vtm = _box_series(n34_path, N34_LAT, N34_LON)
    trop_f, vtm_t = _box_series(trop_path, TROP_LAT, None)
    if n34_f.shape != trop_f.shape or not np.array_equal(vtm, vtm_t):
        raise ValueError(f"N34 (1°) and tropical (5°) member grids misaligned: "
                         f"{n34_f.shape} vs {trop_f.shape}")
    out = {}
    for L in range(1, MAXLEAD + 1):
        want = ((vtm - start_month) % 12) == L
        good = want & np.isfinite(n34_f) & np.isfinite(trop_f)
        if good.any():
            out[L] = (n34_f[good], trop_f[good])
    return out


# ── hindcast climatology (per model, per start-month), cached ─────────────────
def _load_clim() -> pd.DataFrame:
    if CLIM_PATH.exists():
        return pd.read_csv(CLIM_PATH)
    return pd.DataFrame(columns=["model", "month", "lead", "clim_n34", "clim_trop"])


def _clim_rows(df, model, month):
    sub = df[(df.model == model) & (df.month == month)].sort_values("lead")
    if len(sub) < MAXLEAD:
        return None
    n = sub.set_index("lead")["clim_n34"].reindex(range(1, MAXLEAD + 1)).values
    t = sub.set_index("lead")["clim_trop"].reindex(range(1, MAXLEAD + 1)).values
    return (n, t) if np.isfinite(n).all() and np.isfinite(t).all() else None


def _save_clim(model, month, clim_n34, clim_trop):
    df = _load_clim()
    df = df[~((df.model == model) & (df.month == month))]
    rows = pd.DataFrame({"model": model, "month": month, "lead": range(1, MAXLEAD + 1),
                         "clim_n34": np.round(clim_n34, 4), "clim_trop": np.round(clim_trop, 4)})
    df = pd.concat([df, rows], ignore_index=True).sort_values(["model", "month", "lead"])
    tmp = CLIM_PATH.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False)
    os.replace(tmp, CLIM_PATH)


def climatology(model, centre, system, month):
    """(clim_n34[lead], clim_trop[lead]) for a model & start-month, cached."""
    cached = _clim_rows(_load_clim(), model, month)
    if cached is not None:
        return cached
    print(f"  building hindcast climatology: {model} start-month {month:02d} "
          f"({CLIM_YEARS[0]}–{CLIM_YEARS[-1]}) …", flush=True)
    n34_dest = DATA / "hindcast" / f"{centre}_{system}_{month:02d}_n34.grib"   # 1° box
    trop_dest = DATA / "hindcast" / f"{centre}_{system}_{month:02d}.grib"      # 5° band (cached)
    if not _retrieve(centre, system, CLIM_YEARS, f"{month:02d}", n34_dest, AREA_N34, GRID_N34):
        return None
    if not _retrieve(centre, system, CLIM_YEARS, f"{month:02d}", trop_dest, AREA_TROP, GRID_TROP):
        return None
    by_lead = _members_by_lead(n34_dest, trop_dest, month)
    if len(by_lead) < MAXLEAD:
        return None
    cn = np.array([by_lead[L][0].mean() for L in range(1, MAXLEAD + 1)])
    ct = np.array([by_lead[L][1].mean() for L in range(1, MAXLEAD + 1)])
    _save_clim(model, month, cn, ct)
    return cn, ct


# ── RONI per-calendar-month rescale (CPC/ECMWF σ(ONI)/σ(relative)) ─────────────
def _roni_scale_table():
    if not SIGMA_PATH.exists():
        return {}
    tab = json.loads(SIGMA_PATH.read_text()).get("scale_by_month", {})
    return {int(k): float(v) for k, v in tab.items()}


def _month_of_lead(start_month: int, L: int) -> int:
    return ((start_month - 1 + L) % 12) + 1


# ── per-model member trajectories for one issue ───────────────────────────────
def model_members(centre, system, ym):
    """Per-lead member anomalies for one model & issue, both indices (°C):
      {'n34': {lead → array}, 'rnino': {lead → array}}  (rNiño already RONI-scaled)
    None if the model has no data for this issue."""
    month = int(ym[4:])
    # Retrieve the (cheap) forecast FIRST — bail before touching the ~1.4 GB
    # hindcast if this issue month isn't published yet (matters for the quorum
    # probe of a not-yet-released month).
    n34_dest = DATA / "forecast" / f"{centre}_{system}_{ym}_n34.grib"   # 1° box
    trop_dest = DATA / "forecast" / f"{centre}_{system}_{ym}.grib"      # 5° band (cached)
    if not _retrieve(centre, system, [ym[:4]], ym[4:], n34_dest, AREA_N34, GRID_N34):
        return None
    if not _retrieve(centre, system, [ym[:4]], ym[4:], trop_dest, AREA_TROP, GRID_TROP):
        return None
    clim = climatology(centre, centre, system, month)
    if clim is None:
        return None
    cn, ct = clim
    by = _members_by_lead(n34_dest, trop_dest, month)
    if len(by) < MAXLEAD:
        return None
    tab = _roni_scale_table()
    n34_mem, rnino_mem = {}, {}
    for L in range(1, MAXLEAD + 1):
        n34_arr, trop_arr = by[L]
        n34_mem[L] = n34_arr - cn[L - 1]
        rel = (n34_arr - trop_arr) - (cn[L - 1] - ct[L - 1])
        rnino_mem[L] = rel * tab.get(_month_of_lead(month, L), 1.0)
    return {"n34": n34_mem, "rnino": rnino_mem}


def collect(ym):
    """{label → (member_dict, colour)} for every model with data at this issue."""
    out = {}
    for centre, system, label, colour in MODELS:
        r = model_members(centre, system, ym)
        if r is None:
            print(f"  {label:16s} unavailable — skipped", flush=True)
            continue
        out[label] = (r, colour)
        n = np.array([r["n34"][L].mean() for L in range(1, MAXLEAD + 1)])
        rn = np.array([r["rnino"][L].mean() for L in range(1, MAXLEAD + 1)])
        print(f"  {label:16s} {r['n34'][1].size:>3d} mem · "
              f"Niño-3.4 {n[0]:+.2f}→{n[-1]:+.2f}  rNiño-3.4 {rn[0]:+.2f}→{rn[-1]:+.2f}", flush=True)
    return out


def resolve_issue(explicit):
    """(ym, results) — honour --issue, else the newest month meeting QUORUM."""
    if explicit:
        return explicit, collect(explicit)
    now = pd.Timestamp.utcnow()
    best = None
    for back in range(0, 3):
        ym = (now - pd.DateOffset(months=back)).strftime("%Y%m")
        print(f"probing issue {ym} …", flush=True)
        res = collect(ym)
        print(f"  → {len(res)}/{len(MODELS)} models available for {ym}", flush=True)
        if len(res) >= QUORUM:
            return ym, res
        if best is None or len(res) > len(best[1]):
            best = (ym, res)
    if best and best[1]:
        print(f"no issue met quorum {QUORUM}; using best available {best[0]} "
              f"({len(best[1])} models)", flush=True)
        return best
    raise SystemExit("no recent C3S issue with data found on the CDS")


# ── plotting ──────────────────────────────────────────────────────────────────
def _panel(ax, results, key, valid, smooth3, title, ylab, want_labels=False):
    """One panel: member plume + per-model means + multi-model mean.
    Returns (handles, labels) once, for a shared figure legend."""
    ident = lambda a: np.asarray(a, float)
    roll = (lambda a: pd.Series(a).rolling(3, center=True, min_periods=1).mean().values) if smooth3 else ident

    means, pooled = {}, [[] for _ in range(MAXLEAD)]
    for label, (r, colour) in results.items():
        mem = r[key]
        means[label] = (np.array([mem[L].mean() for L in range(1, MAXLEAD + 1)]), colour)
        for i, L in enumerate(range(1, MAXLEAD + 1)):
            pooled[i].append(mem[L])
    pooled = [np.concatenate(p) for p in pooled]

    lo = roll([np.min(p) for p in pooled]); hi = roll([np.max(p) for p in pooled])
    p10 = roll([np.percentile(p, 10) for p in pooled]); p90 = roll([np.percentile(p, 90) for p in pooled])
    p25 = roll([np.percentile(p, 25) for p in pooled]); p75 = roll([np.percentile(p, 75) for p in pooled])
    mmm = roll(np.mean([m for m, _ in means.values()], axis=0))

    h_range = ax.fill_between(valid, lo, hi, color="#5a6b7b", alpha=0.10, lw=0, zorder=1)
    h_10_90 = ax.fill_between(valid, p10, p90, color="#5a6b7b", alpha=0.16, lw=0, zorder=1)
    h_25_75 = ax.fill_between(valid, p25, p75, color="#5a6b7b", alpha=0.24, lw=0, zorder=1)
    handles, labels = [], []
    for label, (m, colour) in means.items():
        (hln,) = ax.plot(valid, roll(m), color=colour, lw=1.3, marker="o", ms=2.8,
                         alpha=0.9, zorder=3)
        handles.append(hln); labels.append(label)
    (h_mmm,) = ax.plot(valid, mmm, color="k", lw=2.8, zorder=5)

    ax.axhline(0, color="0.5", lw=0.8)
    for g in (0.5, 1.0, 1.5, -0.5, -1.0, -1.5):
        ax.axhline(g, color="0.75", lw=0.6, ls=":")
    ax.set_title(title, fontsize=10.5, fontweight="bold", loc="left")
    ax.set_ylabel(ylab)
    ax.grid(True, axis="x", alpha=0.2)
    ax.margins(x=0.02)

    if want_labels:
        handles += [h_mmm, h_25_75, h_10_90, h_range]
        labels += ["Multi-model mean", "All members 25–75%", "10–90%", "Full range"]
        return handles, labels
    return None


def plot(ym: str, results, out: Path):
    if not results:
        raise SystemExit("no models returned data")
    issue = pd.Timestamp(int(ym[:4]), int(ym[4:]), 1)
    valid = [issue + pd.DateOffset(months=L) for L in range(1, MAXLEAD + 1)]

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.7), sharex=True)
    legend = _panel(axes[0, 0], results, "n34", valid, False,
                    "Niño-3.4 anomaly — monthly", "Niño-3.4 (°C)", want_labels=True)
    _panel(axes[0, 1], results, "n34", valid, True,
           "Niño-3.4 anomaly — 3-month running mean  (ONI)", "ONI (°C)")
    _panel(axes[1, 0], results, "rnino", valid, False,
           "Relative Niño-3.4 — monthly", "rNiño-3.4 (°C)")
    _panel(axes[1, 1], results, "rnino", valid, True,
           "Relative Niño-3.4 — 3-month running mean  (RONI)", "RONI (°C)")

    for ax in axes[1, :]:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        for lab in ax.get_xticklabels():
            lab.set_rotation(45); lab.set_ha("right"); lab.set_fontsize(8.5)

    handles, labels = legend
    nmem = sum(results[l][0]["n34"][1].size for l in results)
    fig.suptitle(f"C3S multi-model ENSO forecast — issued {issue:%b %Y}\n"
                 f"{len(results)} systems · {nmem} members · Niño-3.4 and relative Niño-3.4 "
                 f"(monthly + 3-month) · anomaly vs each model's 1993–2016 hindcast",
                 fontsize=12.5, fontweight="bold")
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.055, 1, 0.945))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"saved {out} ({len(results)} models, {nmem} members, issue {ym})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--issue", help="YYYYMM (default: latest available, by quorum)")
    ap.add_argument("--out", default=str(ASSETS / "c3s_nino34.webp"))
    args = ap.parse_args()
    print("C3S multi-model ENSO (Niño-3.4 + rNiño-3.4, monthly + 3-month)")
    ym, results = resolve_issue(args.issue)
    plot(ym, results, Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
