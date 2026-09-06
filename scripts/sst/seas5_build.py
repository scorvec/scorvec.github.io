#!/usr/bin/env python3
"""SEAS5 outlook: indices, tercile maps, polar-cap plumes → JSON + WebP.

Imported by seas5_outlook.py (which owns the CDS retrieval); see its docstring
for what each product is and why it is built the way it is.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*cfgrib.*")
warnings.filterwarnings("ignore", category=RuntimeWarning)

from seas5_outlook import (ASSETS, DATA, HERE, MAXLEAD, PDO_PATTERN, SIGMA_PATH,   # noqa: E402
                           fc_path, hc_path, previous_issues)

OUT_JSON = ASSETS / "data" / "seas5_outlook.json"
G0 = 9.80665

# ── index boxes: (lat0, lat1, lon0, lon1) in −180..180; a lon0 > lon1 box wraps the dateline
BOXES = {
    "nino12": (-10, 0, -90, -80),
    "nino3": (-5, 5, -150, -90),
    "nino34": (-5, 5, -170, -120),
    "nino4": (-5, 5, 160, -150),
    "trop": (-20, 20, -180, 180),
    "natl": (0, 60, -80, 0),
    "glob": (-60, 60, -180, 180),
    "iod_w": (-10, 10, 50, 70),
    "iod_e": (-10, 0, 90, 110),
    "atl3": (-3, 3, -20, 0),
}
INDEX_META = [
    ("nino12", "Niño-1+2", "°C", "0–10°S, 90–80°W: the coastal, east-based index"),
    ("nino3", "Niño-3", "°C", "5°N–5°S, 150–90°W: the eastern basin"),
    ("nino34", "Niño-3.4", "°C", "5°N–5°S, 170–120°W: the ONI region"),
    ("nino4", "Niño-4", "°C", "5°N–5°S, 160°E–150°W: the central-western basin"),
    ("rnino34", "Relative Niño-3.4", "°C", "Niño-3.4 minus the 20°S–20°N tropical mean, RONI-scaled by calendar month"),
    ("tni", "Trans-Niño index", "σ", "standardised Niño-1+2 minus standardised Niño-4: positive means east-loaded"),
    ("pdo", "PDO", "index", "North Pacific EOF projection on the NCEI scale, global mean removed"),
    ("amo", "AMO (relative)", "°C", "North Atlantic 0–60°N minus the 60°S–60°N global mean"),
    ("iod", "Indian Ocean Dipole", "°C", "west (50–70°E) minus east (90–110°E) box"),
    ("atl3", "Atlantic Niño (ATL3)", "°C", "3°S–3°N, 20°W–0°"),
]
ENSO_KEYS = ["nino12", "nino3", "nino34", "nino4", "rnino34", "tni"]
OTHER_KEYS = ["pdo", "amo", "iod", "atl3"]

SEASON_LEADS = [(2, 3, 4), (3, 4, 5), (4, 5, 6)]      # three overlapping seasons after the start month
MONTHS = "JFMAMJJASOND"


# ── GRIB loading ─────────────────────────────────────────────────────────────
def _open(path: Path, **filt) -> xr.Dataset:
    # forecastMonth as the lead axis, not `step`: monthly steps are timedeltas that
    # differ with month length, so a 24-year hindcast opened on `step` fragments into
    # seven partly-empty steps. forecastMonth is 1..6 in every file.
    kw = {"indexpath": "", "time_dims": ("forecastMonth", "time")}
    if filt:
        kw["filter_by_keys"] = dict(filt)
    return xr.open_dataset(path, engine="cfgrib", backend_kwargs=kw)


def _stack_samples(da: xr.DataArray) -> xr.DataArray:
    """(sample, forecastMonth, lat, lon): fold number × time (hindcast years) into one sample axis."""
    extra = [d for d in da.dims if d not in ("forecastMonth", "latitude", "longitude")]
    if not extra:
        da = da.expand_dims("sample")
    else:
        da = da.stack(sample=extra)
    return da.transpose("sample", "forecastMonth", "latitude", "longitude")


def load_field(path: Path, var: str, **filt) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """→ (values[sample, lead, lat, lon] float32, lat, lon) with lead index 0..5."""
    ds = _open(path, **filt)
    if var not in ds.data_vars:                                   # e.g. a single-variable file under another name
        var = list(ds.data_vars)[0]
    da = _stack_samples(ds[var])
    vals = da.values.astype(np.float32)
    lat, lon = da.latitude.values, da.longitude.values
    ds.close()
    if vals.shape[1] != MAXLEAD:
        raise ValueError(f"{path.name}: expected {MAXLEAD} leads, got {vals.shape[1]}")
    return vals, lat, lon


def box_mean(vals: np.ndarray, lat: np.ndarray, lon: np.ndarray, box) -> np.ndarray:
    """Cosine-weighted mean over a box → [sample, lead]; skips NaN (land in SST)."""
    la0, la1, lo0, lo1 = box
    mlat = (lat >= la0) & (lat <= la1)
    mlon = (lon >= lo0) & (lon <= lo1) if lo0 <= lo1 else (lon >= lo0) | (lon <= lo1)
    sub = vals[:, :, mlat][:, :, :, mlon]
    w = np.cos(np.deg2rad(lat[mlat]))[None, None, :, None] * np.ones_like(sub)
    good = np.isfinite(sub)
    num = np.nansum(np.where(good, sub * w, 0.0), axis=(2, 3))
    den = np.sum(np.where(good, w, 0.0), axis=(2, 3))
    return num / den


# ── SST indices ──────────────────────────────────────────────────────────────
def _pdo_index(anom: np.ndarray, lat: np.ndarray, lon: np.ndarray, gmean: np.ndarray) -> np.ndarray:
    """Project [sample, lead, lat, lon] anomalies on the ERSST EOF → PDO [sample, lead]."""
    pat = xr.open_dataset(PDO_PATTERN)
    eof = pat["eof"]
    elat, elon = eof.lat.values, eof.lon.values                  # lon 110..260 (0–360)
    lon360 = np.where(lon < 0, lon + 360, lon)
    order = np.argsort(lon360)
    a = xr.DataArray(anom[:, :, :, order], dims=("s", "l", "lat", "lon"),
                     coords={"lat": lat, "lon": lon360[order]})
    a = a.interp(lat=elat, lon=elon, method="nearest")
    w = np.cos(np.deg2rad(eof.lat))
    den_full = float(((eof ** 2) * w).sum())
    num = (a * eof * w).sum(("lat", "lon"), skipna=True)
    den = ((eof ** 2) * w).where(a.notnull()).sum(("lat", "lon"))
    proj = (num / den.where(den > 0.5 * den_full)).values
    slope, intercept, proj_one = (float(pat.attrs[k]) for k in ("calib_slope", "calib_intercept", "proj_one"))
    return (proj - gmean * proj_one) * slope + intercept


def sst_indices(ym: str) -> dict | None:
    """All SST indices for one issue: {key: {'members': [lead][sample], 'clim_sd': [lead]}}.
    Anomalies are forecast member minus the hindcast mean at the same lead."""
    month = int(ym[4:])
    fcp, hcp = fc_path("sst", ym), hc_path("sst", ym[4:])
    if not (fcp.exists() and hcp.exists()):
        return None
    fc, lat, lon = load_field(fcp, "sst")
    hc, lat2, lon2 = load_field(hcp, "sst")
    assert np.allclose(lat, lat2) and np.allclose(lon, lon2)
    hc_mean = np.nanmean(hc, axis=0)                              # [lead, lat, lon]
    scale = {int(k): float(v) for k, v in json.loads(SIGMA_PATH.read_text())["scale_by_month"].items()}

    raw = {k: (box_mean(fc, lat, lon, b), box_mean(hc, lat, lon, b)) for k, b in BOXES.items()}
    out = {}

    def anom(k):
        f, h = raw[k]
        return f - h.mean(0), h - h.mean(0)

    for k in ("nino12", "nino3", "nino34", "nino4"):
        fa, ha = anom(k)
        out[k] = dict(members=fa, clim_sd=ha.std(0))
    # relative Niño-3.4: (N34 − tropical mean), each vs its own climatology, then the RONI month scale
    f34, h34 = anom("nino34"); ft, ht = anom("trop")
    months = [((month - 1 + L) % 12) + 1 for L in range(MAXLEAD)]
    sc = np.array([scale.get(m, 1.0) for m in months])
    out["rnino34"] = dict(members=(f34 - ft) * sc, clim_sd=((h34 - ht) * sc).std(0))
    # Trans-Niño: standardised by the hindcast spread at each lead
    f12, h12 = anom("nino12"); f4, h4 = anom("nino4")
    z12, z4 = f12 / h12.std(0), f4 / h4.std(0)
    out["tni"] = dict(members=z12 - z4, clim_sd=(h12 / h12.std(0) - h4 / h4.std(0)).std(0))
    # AMO relative, IOD, ATL3
    fna, hna = anom("natl"); fg, hg = anom("glob")
    out["amo"] = dict(members=fna - fg, clim_sd=(hna - hg).std(0))
    fw, hw = anom("iod_w"); fe, he = anom("iod_e")
    out["iod"] = dict(members=fw - fe, clim_sd=(hw - he).std(0))
    fa3, ha3 = anom("atl3")
    out["atl3"] = dict(members=fa3, clim_sd=ha3.std(0))
    # PDO from the gridded anomaly (forecast) and the hindcast years for the climatological spread
    out["pdo"] = dict(members=_pdo_index(fc - hc_mean[None], lat, lon, fg),
                      clim_sd=_pdo_index(hc - hc_mean[None], lat, lon, hg).std(0))
    return out


def _summ(members: np.ndarray) -> dict:
    """members [sample, lead] → quantile summary per lead."""
    q = np.nanpercentile(members, [10, 25, 50, 75, 90], axis=0)
    return dict(mean=np.nanmean(members, 0).round(3).tolist(), p10=q[0].round(3).tolist(), p25=q[1].round(3).tolist(),
                p50=q[2].round(3).tolist(), p75=q[3].round(3).tolist(), p90=q[4].round(3).tolist())


def valid_months(ym: str) -> list[str]:
    y, m = int(ym[:4]), int(ym[4:])
    return [f"{y + (m - 1 + L) // 12}-{((m - 1 + L) % 12) + 1:02d}" for L in range(MAXLEAD)]


# ── terciles over the Americas ───────────────────────────────────────────────
def _season_means(vals: np.ndarray, leads: tuple[int, ...]) -> np.ndarray:
    idx = [L - 1 for L in leads]
    return vals[:, idx].mean(axis=1)                              # [sample, lat, lon]


def tercile_probs(fc: np.ndarray, hc: np.ndarray, leads) -> dict:
    """Model-climatology terciles: thresholds from the 600 hindcast seasonal means
    per grid point, probabilities from the 51 forecast members."""
    f = _season_means(fc, leads); h = _season_means(hc, leads)
    lo, hi = np.nanpercentile(h, [100 / 3, 200 / 3], axis=0)
    below = (f < lo[None]).mean(0); above = (f > hi[None]).mean(0)
    normal = 1.0 - below - above
    return dict(below=below, normal=normal, above=above, ens_anom=f.mean(0) - h.mean(0))


def season_label(ym: str, leads) -> str:
    m0 = int(ym[4:])
    return "".join(MONTHS[(m0 - 1 + L - 1) % 12] for L in leads)


def render_terciles(ym: str, fields: dict, out_dir: Path) -> dict:
    """One figure per variable: three seasons side by side, most-likely tercile shaded
    by its probability. Returns {var: {file, seasons}}."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    bins = [0.40, 0.50, 0.60, 0.70, 0.80, 1.001]
    palettes = {
        "t2m": (["#fde0c8", "#f9b98b", "#f18a4e", "#dc5a23", "#a83a0f"], ["#d6e7f5", "#a9cbe8", "#6fa6d5", "#3d7ebf", "#21569a"]),
        "tp": (["#d8efd2", "#a8dba0", "#6dbf6a", "#3a9a44", "#1d6b2f"], ["#f1e2c7", "#e1c391", "#c9a05a", "#a97b31", "#7b5518"]),
        "z500": (["#fde0c8", "#f9b98b", "#f18a4e", "#dc5a23", "#a83a0f"], ["#d6e7f5", "#a9cbe8", "#6fa6d5", "#3d7ebf", "#21569a"]),
    }
    titles = {"t2m": "2 m temperature", "tp": "Precipitation", "z500": "500 hPa height"}
    proj = ccrs.PlateCarree(central_longitude=-90)
    pc = ccrs.PlateCarree()
    meta = {}
    for var, (fc, hc, lat, lon) in fields.items():
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 7.6), subplot_kw={"projection": proj})
        seasons = []
        for ax, leads in zip(axes, SEASON_LEADS):
            pr = tercile_probs(fc, hc, leads)
            above_c, below_c = palettes[var]
            ax.set_extent([-170, -30, -60, 75], crs=pc)
            ax.add_feature(cfeature.LAND, facecolor="#f4f4f1", zorder=0)
            ax.add_feature(cfeature.OCEAN, facecolor="#ffffff", zorder=0)
            for arr, cols in ((pr["above"], above_c), (pr["below"], below_c)):
                # only where this category is the most likely one AND clears 40 %
                other = np.maximum(pr["normal"], pr["below"] if arr is pr["above"] else pr["above"])
                show = np.where((arr >= 0.40) & (arr >= other), arr, np.nan)
                ax.pcolormesh(lon, lat, show, cmap=ListedColormap(cols), norm=BoundaryNorm(bins, len(cols)),
                              transform=pc, shading="auto", zorder=1)
            if var in ("t2m", "tp"):                            # land products: paint the ocean back over
                ax.add_feature(cfeature.OCEAN, facecolor="#ffffff", zorder=2)
                ax.add_feature(cfeature.LAKES, facecolor="#ffffff", zorder=2)
            ax.coastlines(linewidth=0.5, color="#444", zorder=3)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="#777", zorder=3)
            ax.add_feature(cfeature.STATES, linewidth=0.2, edgecolor="#999", zorder=3)
            lab = season_label(ym, leads)
            vm = valid_months(ym)
            y0s, y1s = vm[leads[0] - 1][:4], vm[leads[-1] - 1][:4]
            yv = y0s if y0s == y1s else f"{y0s}–{y1s[2:]}"
            ax.set_title(f"{lab} {yv}", fontsize=13, loc="left")
            seasons.append(dict(label=lab, leads=list(leads),
                                frac_above=float(np.nanmean(pr["above"] >= 0.40)), frac_below=float(np.nanmean(pr["below"] >= 0.40))))
        # legends: two rows of swatches
        from matplotlib.patches import Patch
        handles = [Patch(color=c, label=f"{int(bins[i]*100)}–{int(min(bins[i+1],1)*100)}%") for i, c in enumerate(palettes[var][0])]
        handles2 = [Patch(color=c, label=f"{int(bins[i]*100)}–{int(min(bins[i+1],1)*100)}%") for i, c in enumerate(palettes[var][1])]
        l1 = fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.06, 0.02), ncol=5, frameon=False,
                        title="Above normal most likely", fontsize=9, title_fontsize=10)
        fig.add_artist(l1)
        fig.legend(handles=handles2, loc="lower right", bbox_to_anchor=(0.94, 0.02), ncol=5, frameon=False,
                   title="Below normal most likely", fontsize=9, title_fontsize=10)
        import calendar
        y0, m0 = ym[:4], int(ym[4:])
        fig.suptitle(f"SEAS5 {titles[var]}: most likely tercile, {calendar.month_name[m0]} {y0} issue (51 members)",
                     x=0.02, y=0.985, ha="left", fontsize=15)
        fig.text(0.02, 0.925, "Terciles from SEAS5's own 1993–2016 hindcast at each grid point (24 years × 25 members), so bias and spread drift are removed before counting.\n"
                 "White: no category reaches 40 %. Near-normal is rarely the most likely tercile in a well-spread ensemble and is not drawn."
                 + ("  With a 1993–2016 base, the warming trend alone tilts temperature toward above normal." if var == "t2m" else ""),
                 fontsize=9.5, color="#444", va="top", linespacing=1.4)
        fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.10, wspace=0.05)
        out = out_dir / f"seas5_terciles_{var}.webp"
        fig.savefig(out, dpi=105, pil_kwargs={"quality": 84, "method": 6}); plt.close(fig)
        meta[var] = dict(file=out.name, seasons=seasons)
        print(f"  wrote {out.name}", flush=True)
    return meta


# ── polar caps ───────────────────────────────────────────────────────────────
def polar_caps(ym: str) -> dict:
    out = {}
    for hemi, kind, box in (("nh", "polar_n", (60, 90, -180, 180)), ("sh", "polar_s", (-90, -60, -180, 180))):
        fcp, hcp = fc_path(kind, ym), hc_path(kind, ym[4:])
        if not (fcp.exists() and hcp.exists()):
            continue
        for lev in (10, 50, 100):
            fc, lat, lon = load_field(fcp, "z", level=lev)
            hc, _, _ = load_field(hcp, "z", level=lev)
            f = box_mean(fc / G0, lat, lon, box); h = box_mean(hc / G0, lat, lon, box)
            a = f - h.mean(0)
            out[f"{hemi}_z{lev}"] = dict(**_summ(a), members=a.round(1).tolist(), clim_sd=h.std(0).round(1).tolist(),
                                          clim_mean=h.mean(0).round(1).tolist(), units="m")
    return out


# ── observed context ─────────────────────────────────────────────────────────
def observed_indices() -> dict:
    """Last 18 observed months of the CPC indices already on the site (ERSSTv5)."""
    p = ASSETS / "data" / "nino_history.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    months = d["months"][-18:]
    out = {"months": months}
    for k in ("nino12", "nino3", "nino34", "nino4", "roni"):
        ser = d["series"].get(k)
        if isinstance(ser, dict):                                  # {"abs": [...], "anom": [...]}
            ser = ser.get("anom", ser.get("abs"))
        if ser:
            out[k] = ser[-18:]
    return out


# ── build ────────────────────────────────────────────────────────────────────
def build(ym: str, n_prev: int = 3) -> None:
    t0 = time.time()
    ASSETS.mkdir(parents=True, exist_ok=True)
    (ASSETS / "data").mkdir(parents=True, exist_ok=True)
    issues = {}
    for iss in [ym] + previous_issues(ym, n_prev):
        print(f"indices {iss} …", flush=True)
        r = sst_indices(iss)
        if r is None:
            print(f"  {iss}: SST forecast or hindcast not on disk — skipped", flush=True)
            continue
        entry = {"valid": valid_months(iss)}
        for k, v in r.items():
            e = _summ(v["members"]); e["clim_sd"] = np.asarray(v["clim_sd"]).round(3).tolist()
            if iss == ym:
                e["members"] = np.asarray(v["members"]).round(3).tolist()
            entry[k] = e
        issues[iss] = entry
    if ym not in issues:
        raise SystemExit(f"no SEAS5 {ym} SST data — run fetch first")

    fields = {}
    if fc_path("sfc", ym).exists() and hc_path("sfc", ym[4:]).exists():
        for var, short, fac in (("t2m", "t2m", 1.0), ("tp", "tprate", 86400.0 * 1000)):
            fc, lat, lon = load_field(fc_path("sfc", ym), short)
            hc, _, _ = load_field(hc_path("sfc", ym[4:]), short)
            fields[var] = (fc * fac, hc * fac, lat, lon)
    if fc_path("z500", ym).exists() and hc_path("z500", ym[4:]).exists():
        fc, lat, lon = load_field(fc_path("z500", ym), "z")
        hc, _, _ = load_field(hc_path("z500", ym[4:]), "z")
        fields["z500"] = (fc / G0, hc / G0, lat, lon)
    terc = render_terciles(ym, fields, ASSETS) if fields else {}

    polar = polar_caps(ym)

    doc = {
        "generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        "system": "ECMWF SEAS5 (C3S originating_centre ecmwf, system 51)",
        "issue": ym, "issues": sorted(issues), "members": 51, "hindcast": "1993–2016, 25 members",
        "index_meta": [dict(key=k, label=l, units=u, note=n) for k, l, u, n in INDEX_META],
        "enso_keys": ENSO_KEYS, "other_keys": OTHER_KEYS,
        "indices": issues,
        "observed": observed_indices(),
        "terciles": terc,
        "polar": polar,
    }
    OUT_JSON.write_text(json.dumps(doc, separators=(",", ":")))
    print(f"wrote {OUT_JSON} ({OUT_JSON.stat().st_size / 1e3:.0f} kB) in {(time.time() - t0) / 60:.1f} min", flush=True)
