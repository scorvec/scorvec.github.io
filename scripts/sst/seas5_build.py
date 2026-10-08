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

from seas5_outlook import (ASSETS, DATA, HERE, MAXLEAD, MONTHLY_LEADS, PDO_PATTERN, SIGMA_PATH,   # noqa: E402
                           fc_path, hc_path, nleads, previous_issues)
import seas5_ref as R                                                                 # noqa: E402

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

# overlapping seasons after the start month; the fourth ends at lead month 7 (FMA for an October start), which only
# issues with 7 leads have (seasons_for)
SEASON_LEADS = [(2, 3, 4), (3, 4, 5), (4, 5, 6), (5, 6, 7)]


def seasons_for(n: int) -> list:
    """The SEASON_LEADS an issue / table with n lead months can make."""
    return [s for s in SEASON_LEADS if max(s) <= n]
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


def load_field(path: Path, var: str, lead7: bool = True, **filt) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """→ (values[sample, lead, lat, lon] float32, lat, lon) with lead index 0..5 from the monthly GRIB, plus lead 7
    (index 6) when the issue's lead-7 supplement exists (forecast/fc_{kind}_{ym}_L7.npz, seas5_lead7.py: the
    sub-daily product reduced to the monthly definition, same area and grid). A lead-7 pseudo-raw hindcast
    (hindcast/hc_{kind}_{MM}_L7.npz, seas5_aws.py) comes back as 7 leads with leads 1–4 empty."""
    path = Path(path)
    if path.suffix == ".npz" and path.stem.endswith("_L7"):
        from seas5_aws import load_pseudo
        return load_pseudo(path, var, filt.get("level"))
    ds = _open(path, **filt)
    asked = var
    if var not in ds.data_vars:                                   # e.g. a single-variable file under another name
        if len(ds.data_vars) != 1:                                # 2026-10-08: never guess in a multi-variable file
            raise KeyError(f"{path.name}: no variable {var!r} (has {list(ds.data_vars)})")
        var = list(ds.data_vars)[0]
    da = _stack_samples(ds[var])
    vals = da.values.astype(np.float32)
    lat, lon = da.latitude.values, da.longitude.values
    ds.close()
    if vals.shape[1] != MONTHLY_LEADS:
        raise ValueError(f"{path.name}: expected {MONTHLY_LEADS} leads, got {vals.shape[1]}")
    sup = path.with_name(path.stem + "_L7.npz")
    ym = path.stem.rsplit("_", 1)[-1]
    if lead7 and path.parent.name == "forecast" and sup.exists() and len(ym) == 6 and nleads(ym) >= 7:
        lev = filt.get("level")
        with np.load(sup) as z:
            names = [k for k in z.files if k not in ("lat", "lon", "__meta__")]
            key = asked if lev is None else f"{asked}_{int(lev)}"
            if key not in names and lev is None and len(names) == 1:
                key = names[0]
            if key in names:
                if not (np.allclose(z["lat"], lat) and np.allclose(((z["lon"] + 180) % 360) - 180, ((lon + 180) % 360) - 180)):
                    raise ValueError(f"{sup.name}: lead-7 grid differs from {path.name}")
                v7 = z[key].astype(np.float32)
                if v7.shape != vals[:, 0].shape:
                    raise ValueError(f"{sup.name}:{key} shape {v7.shape} vs {vals[:, 0].shape}")
                vals = np.concatenate([vals, v7[:, None]], axis=1)
            else:
                print(f"  {sup.name}: no {key} — {path.name} stays at 6 leads", flush=True)
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


def _sst_hc_table(month: str) -> None:
    """hc_build_sst_{MM}: what the SST indices need from the hindcast — the box mean of every sample
    [box, sample, lead], the mean field [lead, lat, lon] (for the PDO anomaly) and every sample's PDO."""
    hc, lat, lon = load_field(hc_path("sst", month), "sst")
    hc_mean = np.nanmean(hc, axis=0)
    hb = np.stack([box_mean(hc, lat, lon, b) for b in BOXES.values()])
    g = hb[list(BOXES).index("glob")]
    pdo = _pdo_index(hc - hc_mean[None], lat, lon, g - g.mean(0))
    R.save(R.hc_ref("build_sst", month), meta=dict(boxes=list(BOXES)), box=hb, hc_mean=hc_mean.astype(np.float32),
           pdo=pdo, lat=lat, lon=lon)


def _save_sst_issue(ym: str, r: dict) -> None:
    """is_build_sst_{ym}: the index members, so a later issue draws this one's plume without its GRIBs."""
    R.save(R.is_ref("build_sst", ym), meta=dict(keys=list(r)),
           **{f"{k}_{f}": np.asarray(v[f], dtype=np.float64) for k, v in r.items() for f in ("members", "clim_sd")})


def sst_indices(ym: str) -> dict | None:
    """All SST indices for one issue: {key: {'members': [lead][sample], 'clim_sd': [lead]}}.
    Anomalies are forecast member minus the hindcast mean at the same lead. The hindcast enters through
    the table hc_build_sst_{MM}; an earlier issue whose forecast GRIB is not on disk comes back from its
    is_build_sst summary."""
    month = int(ym[4:])
    fcp = fc_path("sst", ym)
    if not fcp.exists():
        d = R.load(R.is_ref("build_sst", ym))
        if d is None:
            return None
        return {k: dict(members=d[f"{k}_members"], clim_sd=d[f"{k}_clim_sd"]) for k in d["meta"]["keys"]}
    ref = R.load(R.hc_ref("build_sst", ym[4:]))
    if ref is None:
        print(f"  {ym}: hindcast table hc_build_sst_{ym[4:]} missing", flush=True)
        return None
    fc, lat, lon = load_field(fcp, "sst")
    assert np.allclose(lat, ref["lat"]) and np.allclose(lon, ref["lon"])
    nl = min(fc.shape[1], ref["hc_mean"].shape[0], ref["box"].shape[-1])   # 7 when both the forecast and the table reach lead 7
    if nl < fc.shape[1]:
        print(f"  {ym}: SST indices at {nl} leads (the hindcast table has no lead {fc.shape[1]})", flush=True)
    fc = fc[:, :nl]
    hc_mean = ref["hc_mean"][:nl]                                 # [lead, lat, lon]
    scale = {int(k): float(v) for k, v in json.loads(SIGMA_PATH.read_text())["scale_by_month"].items()}

    names = list(ref["meta"]["boxes"])
    raw = {k: (box_mean(fc, lat, lon, b), ref["box"][names.index(k)][:, :nl]) for k, b in BOXES.items()}
    out = {}

    def anom(k):
        f, h = raw[k]
        return f - h.mean(0), h - h.mean(0)

    for k in ("nino12", "nino3", "nino34", "nino4"):
        fa, ha = anom(k)
        out[k] = dict(members=fa, clim_sd=ha.std(0))
    # relative Niño-3.4: (N34 − tropical mean), each vs its own climatology, then the RONI month scale
    f34, h34 = anom("nino34"); ft, ht = anom("trop")
    months = [((month - 1 + L) % 12) + 1 for L in range(nl)]
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
    # PDO from the gridded anomaly (forecast); the hindcast years' PDO (the climatological spread) is in the table
    out["pdo"] = dict(members=_pdo_index(fc - hc_mean[None], lat, lon, fg),
                      clim_sd=ref["pdo"][:, :nl].std(0))
    _save_sst_issue(ym, out)
    return out


def _summ(members: np.ndarray) -> dict:
    """members [sample, lead] → quantile summary per lead."""
    q = np.nanpercentile(members, [10, 25, 50, 75, 90], axis=0)
    return dict(mean=np.nanmean(members, 0).round(3).tolist(), p10=q[0].round(3).tolist(), p25=q[1].round(3).tolist(),
                p50=q[2].round(3).tolist(), p75=q[3].round(3).tolist(), p90=q[4].round(3).tolist())


def valid_months(ym: str, n: int | None = None) -> list[str]:
    """The calendar months of an issue's leads: nleads(ym) of them (7 from the October 2026 issue), or n."""
    y, m = int(ym[:4]), int(ym[4:])
    return [f"{y + (m - 1 + L) // 12}-{((m - 1 + L) % 12) + 1:02d}" for L in range(n or nleads(ym))]


# ── terciles over the Americas ───────────────────────────────────────────────
def _season_means(vals: np.ndarray, leads: tuple[int, ...]) -> np.ndarray:
    idx = [L - 1 for L in leads]
    return vals[:, idx].mean(axis=1)                              # [sample, lat, lon]


def tercile_probs(fc: np.ndarray, hc: np.ndarray, leads, detrend_ym: str | None = None) -> dict:
    """Model-climatology terciles: thresholds from the 600 hindcast seasonal means
    per grid point, probabilities from the 51 forecast members. With `detrend_ym` the
    hindcast's linear trend (per grid point) is removed and the members are counted against
    it extrapolated to the season's valid year — for temperature and heights, where a
    1993–2016 base alone puts the whole tropics in the upper tercile (seen 2026-09-07)."""
    f = _season_means(fc, leads); h = _season_means(hc, leads)
    if detrend_ym is not None:
        yrs = 1993 + hindcast_years(h.shape[0]); x = (yrs - yrs.mean())[:, None, None]
        hm = np.nanmean(h, axis=0, keepdims=True)
        b = np.nansum(x * (h - hm), axis=0) / float((x[:, 0, 0] ** 2).sum())
        target = int(valid_months(detrend_ym)[leads[len(leads) // 2] - 1][:4]) - yrs.mean()
        h = h - b[None] * x
        f = f - b[None] * target
    lo, hi = np.nanpercentile(h, [100 / 3, 200 / 3], axis=0)
    below = (f < lo[None]).mean(0); above = (f > hi[None]).mean(0)
    normal = 1.0 - below - above
    return dict(below=below, normal=normal, above=above, ens_anom=f.mean(0) - h.mean(0))


def tercile_bounds(hc: np.ndarray, leads, detrend: bool) -> dict:
    """The hindcast half of tercile_probs, kept as a table: per grid point the 1/3 and 2/3 bounds of the
    600 seasonal means and, with `detrend`, the per-point linear-trend slope b (per year)."""
    h = _season_means(hc, leads)
    out = {}
    if detrend:
        yrs = 1993 + hindcast_years(h.shape[0]); x = (yrs - yrs.mean())[:, None, None]
        hm = np.nanmean(h, axis=0, keepdims=True)
        b = np.nansum(x * (h - hm), axis=0) / float((x[:, 0, 0] ** 2).sum())
        h = h - b[None] * x
        out["b"] = b; out["ymean"] = np.float64(yrs.mean())
    out["lo"], out["hi"] = np.nanpercentile(h, [100 / 3, 200 / 3], axis=0)
    return out


def tercile_probs_ref(fc: np.ndarray, tb: dict, leads, ym: str) -> dict:
    """tercile_probs against stored bounds (tercile_bounds): the forecast members are moved by the trend
    extrapolated to the season's valid year when the bounds are detrended."""
    f = _season_means(fc, leads)
    if "b" in tb:
        target = int(valid_months(ym)[leads[len(leads) // 2] - 1][:4]) - float(tb["ymean"])
        f = f - tb["b"][None] * target
    below = (f < tb["lo"][None]).mean(0); above = (f > tb["hi"][None]).mean(0)
    return dict(below=below, normal=1.0 - below - above, above=above)


def season_label(ym: str, leads) -> str:
    m0 = int(ym[4:])
    return "".join(MONTHS[(m0 - 1 + L - 1) % 12] for L in leads)


# Tercile probability bins: six steps to 100 % so a 95 % cell reads darker than an 82 % one
# (user 2026-09-06: "don't make 80–100 % all the same colour").
TERC_BINS = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.001]
TERC_PALETTES = {
    "warm": ["#fde4cf", "#fbc39c", "#f59d68", "#e8703c", "#c8451c", "#8f2a0d"],
    "cool": ["#dbe9f6", "#b7d2ec", "#8ab6df", "#5c95cd", "#3672b6", "#1f4f8f"],
    "wet": ["#dcf0d6", "#b6dfad", "#89c983", "#5aae5c", "#338c3f", "#1b6229"],
    "dry": ["#f3e6cd", "#e6cd9f", "#d3af6f", "#b98d45", "#976c27", "#6b4b14"],
    "sunny": ["#fff4cc", "#ffe59a", "#ffd166", "#fbb53a", "#e59318", "#b86f00"],
    "dull": ["#e3e9ef", "#c4d0dc", "#a1b4c5", "#7d97ad", "#5c7a93", "#405c74"],
}
# Americas extent 170W–30W × 60S–75N is ~1.04 wide:tall on PlateCarree; a figure that
# ignores that leaves dead bands above and below the maps. Size the figure from the panels.
MAP_ASPECT = 135.0 / 140.0                                          # height / width of one panel


def map_layout(width: float, ncols: int, nrows: int, top_in: float, bottom_in: float,
               wspace: float = 0.05, hspace: float = 0.10, side: float = 0.02):
    """Figure height and subplot fractions so that nrows × ncols Americas panels fill the
    figure exactly, with `top_in` inches reserved above the panels (title, subtitle, panel
    titles) and `bottom_in` below (legend / colour bar). Fixed inches, so the text bands
    are the same size whatever the figure height and never overlap the maps."""
    panel_w = width * (1 - 2 * side) / (ncols + (ncols - 1) * wspace)
    panel_h = panel_w * MAP_ASPECT
    rows_h = nrows * panel_h + (nrows - 1) * hspace * panel_h
    h = rows_h + top_in + bottom_in
    return h, dict(left=side, right=1 - side, top=1 - top_in / h, bottom=bottom_in / h, wspace=wspace, hspace=hspace)


def head_text(fig, h, title, sub, title_size=15, sub_size=9.5):
    """Title at the very top, subtitle just under it, both measured in inches from the top."""
    fig.suptitle(title, x=0.02, y=1 - 0.10 / h, ha="left", va="top", fontsize=title_size)
    fig.text(0.02, 1 - 0.42 / h, sub, fontsize=sub_size, color="#444", va="top", linespacing=1.35)


def render_terciles(ym: str, fields: dict, out_dir: Path) -> dict:
    """One map per (variable, season): the most likely tercile against SEAS5's own hindcast, on the
    global 1° fields where they exist (t2m, tp, z500, u850) and the Americas box otherwise
    (user 2026-09-07: "single plot charts only"). Files seas5_terc_{var}_{SEASON_YYYY}.webp."""
    import calendar
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch
    from cartopy.util import add_cyclic_point
    import mapstyle as MS

    bins = TERC_BINS
    palettes = {"t2m": (TERC_PALETTES["warm"], TERC_PALETTES["cool"]), "tp": (TERC_PALETTES["wet"], TERC_PALETTES["dry"]),
                "z500": (TERC_PALETTES["warm"], TERC_PALETTES["cool"]), "u850": (TERC_PALETTES["warm"], TERC_PALETTES["cool"]),
                "sst": (TERC_PALETTES["warm"], TERC_PALETTES["cool"])}
    titles = {"t2m": "2 m temperature", "tp": "Precipitation", "z500": "500 hPa height", "u850": "850 hPa zonal wind", "sst": "Sea surface temperature"}
    sides = {"u850": ("Westerly anomaly most likely", "Easterly anomaly most likely")}
    # global members against the stored hindcast tercile bounds (hc_build_{kind}_{MM}: built from the 600
    # hindcast seasonal means per grid point by tercile_bounds — the raw hindcast is not read here)
    globals_ = {}
    for var, kind, short, fac in GLOBAL_TERC:
        tab = R.load(R.hc_ref(f"build_{kind}", ym[4:]))
        if fc_path(kind, ym).exists() and tab is not None and f"terc_{var}_lo_0" in tab:
            globals_[var] = (kind, short, fac, tab)
        else:
            print(f"  terciles {var}: forecast or table hc_build_{kind}_{ym[4:]} missing — skipped", flush=True)
    y0, m0 = ym[:4], int(ym[4:]); vm = valid_months(ym)
    meta = {}
    for var in globals_:
        kind, short, fac, tab = globals_[var]
        fc, lat, lon = load_field(fc_path(kind, ym), short)
        fc = fc * fac; glob = True
        above_c, below_c = palettes[var]
        seasons = []
        for si, leads in enumerate(SEASON_LEADS):
            if max(leads) > fc.shape[1] or f"terc_{var}_lo_{si}" not in tab:
                print(f"  terciles {var} {season_label(ym, leads)}: lead {max(leads)} not in the forecast or the hindcast table — skipped", flush=True)
                continue
            tb = {k: tab[f"terc_{var}_{k}_{si}"] for k in ("lo", "hi", "b", "ymean") if f"terc_{var}_{k}_{si}" in tab}
            pr = tercile_probs_ref(fc, tb, leads, ym)
            lab = season_label(ym, leads); y0s, y1s = vm[leads[0] - 1][:4], vm[leads[-1] - 1][:4]
            yv = y0s if y0s == y1s else f"{y0s}–{y1s[2:]}"; key = f"{lab}_{y0s}"
            fig, ax, H, pc = MS.open_map(kind="atm") if glob else MS.open_map(kind="atm", extent=[-170, -30, -60, 75], central=-90)
            order = np.argsort(lon); lon_s = lon[order]; lat_a = lat[::-1] if lat[0] > lat[-1] else lat
            def prep(a):
                a = a[:, order]
                if lat[0] > lat[-1]: a = a[::-1]
                if glob:
                    a, lo = add_cyclic_point(a, coord=lon_s); return a, lo
                return a, lon_s
            normal = 1.0 - pr["above"] - pr["below"]
            for arr, other, cols in ((pr["above"], pr["below"], above_c), (pr["below"], pr["above"], below_c)):
                show = np.where((arr >= 0.40) & (arr >= np.maximum(other, normal)), arr, np.nan)
                a, lo = prep(show)
                ax.pcolormesh(lo, lat_a, a, cmap=ListedColormap(cols), norm=BoundaryNorm(bins, len(cols)), transform=pc, shading="auto", zorder=1)
            MS.features(ax, land_only=(var in ("t2m", "tp")))
            top_lab, bot_lab = sides.get(var, ("Above normal most likely", "Below normal most likely"))
            h1 = [Patch(color=c, label=f"{int(bins[k]*100)}–{int(min(bins[k+1],1)*100)}%") for k, c in enumerate(above_c)]
            h2 = [Patch(color=c, label=f"{int(bins[k]*100)}–{int(min(bins[k+1],1)*100)}%") for k, c in enumerate(below_c)]
            l1 = fig.legend(handles=h1, loc="lower left", bbox_to_anchor=(0.04, 0.004), ncol=6, frameon=False, title=top_lab, fontsize=8, title_fontsize=8.5)
            fig.add_artist(l1)
            fig.legend(handles=h2, loc="lower right", bbox_to_anchor=(0.96, 0.004), ncol=6, frameon=False, title=bot_lab, fontsize=8, title_fontsize=8.5)
            MS.heading(fig, H, f"SEAS5 {titles[var]}: most likely tercile · {lab} {yv} · {calendar.month_name[m0]} {y0} issue",
                       "Terciles from SEAS5's own 1993–2016 hindcast at each grid point (24 years × 25 members); the 51 members are counted against them, so bias and spread drift are removed first. "
                       "White: no category reaches 40 %; near-normal is not drawn." + ("  Counted against the hindcast's linear trend extrapolated to the valid year, so the warming trend itself does not tilt the map." if var in ("t2m", "z500") else ""),
                       wrap=190)
            out = out_dir / f"seas5_terc_{var}_{key}.webp"
            MS.save(fig, out, dpi=110)
            seasons.append(dict(key=key, label=f"{lab} {yv}", leads=list(leads), file=out.name, lead7=max(leads) >= 7,
                                frac_above=float(np.nanmean(pr["above"] >= 0.40)), frac_below=float(np.nanmean(pr["below"] >= 0.40))))
        meta[var] = dict(label=titles[var], extent="global" if glob else "americas", seasons=seasons, detrended=var in ("t2m", "z500"))
        print(f"  terciles {var}: {len(seasons)} maps ({'global' if glob else 'americas'})", flush=True)
        del fc
    return meta


# ── global single-map viewer: anomaly and change per month / season ─────────
MAP_SPEC = {
    # var: (label, units, anomaly levels, change levels, cmap, contour step anom, contour step chg)
    "t2m": ("2 m temperature", "°C", [-4, -3, -2, -1.5, -1, -0.5, -0.25, 0.25, 0.5, 1, 1.5, 2, 3, 4],
            [-2, -1.5, -1, -0.5, -0.25, 0.25, 0.5, 1, 1.5, 2], "RdBu_r", None, None),
    "tp": ("Precipitation", "mm/day", [-3, -2, -1.5, -1, -0.5, -0.25, 0.25, 0.5, 1, 1.5, 2, 3],
           [-2, -1.5, -1, -0.5, -0.2, 0.2, 0.5, 1, 1.5, 2], "BrBG", None, None),
    # heights are standardised by the hindcast's interannual σ (per cell, lead, or season): a warm-climate
    # +60 m everywhere saturated a metre scale (user 2026-09-07)
    "z500": ("500 hPa height, standardised", "σ", [-5, -4, -3, -2.5, -2, -1.5, -1, -0.5, -0.25, 0.25, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5],
             [-2, -1.5, -1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1, 1.5, 2], "RdBu_r", None, None),
    # SST: fine steps and labelled isolines (user 2026-09-07: "make it highly detailed")
    # ±3 °C at 0.2 steps saturated on the 2026 El Niño (user 2026-09-07): 0.2 steps to ±3, then 0.5 steps to ±5
    "sst": ("Sea surface temperature", "°C", [-5, -4.5, -4, -3.5] + [round(x, 2) for x in np.arange(-3.0, 3.01, 0.2)] + [3.5, 4, 4.5, 5],
            [-2.5, -2] + [round(x, 2) for x in np.arange(-1.5, 1.51, 0.1)] + [2, 2.5], "RdBu_r", 0.5, 0.25),
    # 850 hPa zonal wind (user 2026-09-07): positive = westerly anomaly; the trade-wind / jet signal of ENSO
    "u850": ("850 hPa zonal wind", "m/s", [-8, -6, -4, -3, -2, -1, -0.5, 0.5, 1, 2, 3, 4, 6, 8],
             [-4, -3, -2, -1, -0.5, 0.5, 1, 2, 3, 4], "PuOr_r", None, None),
    # 850 hPa meridional wind (user 2026-09-08: southerly-flow events, US East Coast): green southerly, purple northerly
    "v850": ("850 hPa meridional wind", "m/s, southerly +", [-6, -4, -3, -2, -1, -0.5, 0.5, 1, 2, 3, 4, 6],
             [-3, -2, -1, -0.5, 0.5, 1, 2, 3], "PRGn", None, None),
    # North America snowfall as cm of snow per month at a 10:1 snow-to-liquid ratio (user 2026-09-07:
    # "instead of mm/day, estimate snowfall using 10:1 ratio")
    # shown as % of the hindcast normal (user: "% of normal might be a good choice"); change in % points
    "sf": ("Snowfall, % of normal", "% of normal", [0, 25, 50, 75, 90, 110, 125, 150, 200, 300, 400],
           [-100, -50, -25, -10, 10, 25, 50, 100], "BrBG", None, None),
    # snow depth (snowpack water equivalent, monthly mean) — absolute anomaly in mm and % of the hindcast
    # normal (user 2026-09-07: "snow depth anomalies ... mainly for northwest/quebec hydro")
    "sd": ("Snowpack water equivalent anomaly", "mm w.e.", [-150, -100, -60, -30, -15, -5, 5, 15, 30, 60, 100, 150],
           [-60, -30, -15, -5, 5, 15, 30, 60], "BrBG", None, None),
    "sdp": ("Snowpack, % of normal", "% of normal", [0, 25, 50, 75, 90, 110, 125, 150, 200, 300, 400],
            [-100, -50, -25, -10, 10, 25, 50, 100], "BrBG", None, None),
}
MAP_CENTRAL = {"sst": -160.0, "sf": -110.0, "sd": -110.0, "sdp": -110.0}   # SST cut at 20°E (Africa); everything else at 40°E; snow is a NA box
MAP_CENTRAL_DEFAULT = -140.0
MAP_EXTENT = {"sf": [-170, -50, 25, 75], "sd": [-170, -50, 25, 75], "sdp": [-170, -50, 25, 75]}      # [W, E, S, N] for regional variables


# (var, kind, GRIB short name, factor to display units) for the global single-map viewer; sf and sd are the
# North America snow kinds (sf's factor depends on the month lengths: _snow_fac). The tercile maps use GLOBAL_TERC.
GLOBAL_VARS = [("t2m", "gl", "t2m", 1.0), ("tp", "gl", "tprate", 86400.0 * 1000), ("sst", "gl", "sst", 1.0),
               ("z500", "gl_z500", "z", 1.0 / G0), ("u850", "gl_u850", "u", 1.0), ("v850", "gl_v850", "v", 1.0),
               ("sf", "na_snow", "mtsfr", None), ("sd", "na_snowdepth", "sd", 1000.0)]
GLOBAL_TERC = [("t2m", "gl", "t2m", 1.0), ("tp", "gl", "tprate", 86400.0 * 1000), ("z500", "gl_z500", "z", 1.0 / G0),
               ("u850", "gl_u850", "u", 1.0)]
DETRENDED_TERC = ("t2m", "z500")
# every lead set a period of this issue, or of the next issue's "previous", can ask for: the six months and
# the three seasons (a month-later start reaches the same calendar months one lead later)
def leadsets_for(n: int) -> list:
    return [(L,) for L in range(1, n + 1)] + [tuple(x) for x in seasons_for(n)]


LEADSETS = leadsets_for(MAXLEAD)


def _ls_key(leads) -> str:
    return "-".join(str(int(x)) for x in leads)


def _snow_fac(ym: str, n: int | None = None) -> np.ndarray:
    """m w.e. s⁻¹ → cm w.e./month → cm of snow at 10:1, per lead (month lengths of this issue's first n valid months)."""
    import calendar as _cal
    days = np.array([_cal.monthrange(int(v[:4]), int(v[5:]))[1] for v in valid_months(ym)[:n]], dtype=np.float32)
    return 86400.0 * 100.0 * 10.0 * days[None, :, None, None]


def _global_hc_table(kind: str, month: str) -> None:
    """hc_build_{kind}_{MM}: per variable of the kind the hindcast mean [lead, lat, lon] in GRIB units,
    the tercile bounds per season (tercile_bounds, display units) and, for 500 hPa height, the per-lead-set
    linear trend of the yearly ensemble means (mean, slope per year, residual σ) that standardises the maps."""
    arrays, meta = {}, {}
    terc_vars = [v for v, *_ in GLOBAL_TERC]
    for var, k, short, fac in GLOBAL_VARS:
        if k != kind:
            continue
        hc, lat, lon = load_field(hc_path(kind, month), short)
        nl = hc.shape[1]                                                  # 6 (CDS monthly) or 7 (lead-7 pseudo-raw)
        meta["season_leads"] = [list(x) for x in seasons_for(nl)]; meta["leadsets"] = [_ls_key(x) for x in leadsets_for(nl)]
        arrays[f"hcm_{var}"] = np.nanmean(hc, axis=0)
        arrays["lat"], arrays["lon"] = lat, lon
        if var in terc_vars:
            hcd = hc * fac
            for si, leads in enumerate(seasons_for(nl)):
                for kk, vv in tercile_bounds(hcd, leads, var in DETRENDED_TERC).items():
                    arrays[f"terc_{var}_{kk}_{si}"] = vv
            del hcd
        if var == "z500" and hc.shape[0] % 24 == 0:
            # samples are member-major, year fastest (_stack_samples): per-year ensemble means give the
            # interannual σ used to standardise heights (user 2026-09-07: "standardize 500 mb heights")
            hc_ym = np.nanmean(hc.reshape(-1, 24, *hc.shape[1:]), axis=0) / G0
            for ls in leadsets_for(nl):
                idx = [L - 1 for L in ls]
                ym_ = hc_ym[:, idx].mean(1); ny = ym_.shape[0]; yr = np.arange(ny) - (ny - 1) / 2
                slope = (yr[:, None, None] * (ym_ - ym_.mean(0))).sum(0) / (yr ** 2).sum()
                arrays[f"ztr_mean_{_ls_key(ls)}"] = ym_.mean(0)
                arrays[f"ztr_slope_{_ls_key(ls)}"] = slope
                arrays[f"ztr_sd_{_ls_key(ls)}"] = np.nanstd(ym_ - slope[None] * yr[:, None, None], axis=0)
            meta["ztr_ny"] = 24
        del hc
    R.save(R.hc_ref(f"build_{kind}", month), meta=meta, **arrays)


def _global_issue_table(ym: str) -> None:
    """is_build_maps_{ym}: this issue's ensemble means [lead, lat, lon] in display units — all the next
    issue's change maps take from it."""
    arrays = {}
    for var, kind, short, fac in GLOBAL_VARS:
        if not fc_path(kind, ym).exists():
            continue
        fc, lat, lon = load_field(fc_path(kind, ym), short)
        f = _snow_fac(ym, fc.shape[1]) if var == "sf" else fac
        arrays[f"fcm_{var}"] = np.nanmean(fc * f, axis=0); arrays[f"lat_{var}"] = lat; arrays[f"lon_{var}"] = lon
        del fc
    if arrays:
        R.save(R.is_ref("build_maps", ym), **arrays)


def load_global(ym: str, current: bool = True) -> dict:
    """{var: (fcm[lead, lat, lon], hc_mean[lead, lat, lon], lat, lon, ztrend|None)} for the single-map viewer.
    The ensemble mean comes from this issue's GRIB (`current`) or, for the previous issue, from its
    is_build_maps summary; the hindcast mean and the height trend from hc_build_{kind}_{MM}."""
    out = {}
    summ = None if current else R.load(R.is_ref("build_maps", ym))
    if not current and summ is None:
        print(f"  global maps: no summary is_build_maps_{ym} — change maps skipped", flush=True)
        return out
    tabs = {}
    for var, kind, short, fac in GLOBAL_VARS:
        if kind not in tabs:
            tabs[kind] = R.load(R.hc_ref(f"build_{kind}", ym[4:]))
        tab = tabs[kind]
        if tab is None or f"hcm_{var}" not in tab:
            print(f"  global {var} {ym}: table hc_build_{kind}_{ym[4:]} missing — skipped", flush=True); continue
        if current:
            if not fc_path(kind, ym).exists():
                continue
            try:
                fc, lat, lon = load_field(fc_path(kind, ym), short)
            except Exception as e:                                          # noqa: BLE001
                print(f"  global {var} {ym}: {str(e)[:100]}", flush=True); continue
            f = _snow_fac(ym, fc.shape[1]) if var == "sf" else fac
            fcm = np.nanmean(fc * f, axis=0); del fc
        else:
            if f"fcm_{var}" not in summ:
                continue
            fcm, lat, lon = summ[f"fcm_{var}"], summ[f"lat_{var}"], summ[f"lon_{var}"]
        nh = tab[f"hcm_{var}"].shape[0]
        hcm = tab[f"hcm_{var}"] * (_snow_fac(ym, nh)[0] if var == "sf" else fac)
        ztr = None
        if var == "z500" and "ztr_ny" in tab["meta"]:
            ztr = {k: tab[k] for k in tab if k.startswith("ztr_")}
        out[var] = (fcm, hcm, lat, lon, ztr)
        if var == "sd":
            out["sdp"] = (fcm, hcm, lat, lon, None)
    if current:
        _global_issue_table(ym)
    return out


def _period_sets(ym: str, prev: str | None):
    """[(key, label, leads_now, leads_prev|None)] — six months then the three seasons; leads_prev is
    None where the previous issue has no counterpart."""
    import calendar
    vm_now = valid_months(ym); vm_prev = valid_months(prev) if prev else []
    out = []
    for L, v in enumerate(vm_now, start=1):
        lp = (vm_prev.index(v) + 1,) if v in vm_prev else None
        out.append((v.replace("-", "_"), f"{calendar.month_abbr[int(v[5:])]} {v[:4]}", (L,), lp))
    for leads in seasons_for(len(vm_now)):
        months = [vm_now[L - 1] for L in leads]
        lp = tuple(vm_prev.index(m) + 1 for m in months) if all(m in vm_prev for m in months) else None
        y0, y1 = months[0][:4], months[-1][:4]
        out.append((f"{season_label(ym, leads)}_{y0}", f"{season_label(ym, leads)} {y0 if y0 == y1 else y0 + '–' + y1[2:]}", leads, lp))
    return out


def _global_map(field, lat, lon, var, levels, cmap, cstep, title, sub, cb_label, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    pc = ccrs.PlateCarree(); proj = ccrs.PlateCarree(central_longitude=MAP_CENTRAL.get(var, MAP_CENTRAL_DEFAULT))
    ext = MAP_EXTENT.get(var)
    lat0, lat1 = (ext[2], ext[3]) if ext else ((-70, 70) if var == "sst" else (-60, 85))
    # Both CRSs are PlateCarree, so put the data in the map's own frame (longitude relative to the central meridian, exact
    # +-180 edge columns) and draw with transform=proj: cartopy then bypasses project_geometry, which in cartopy 0.25 +
    # shapely 2.1 fails at draw time on some masked contour sets ("'GeometryCollection' object is not subscriptable" - the
    # October 2026 SST maps).
    c0 = MAP_CENTRAL.get(var, MAP_CENTRAL_DEFAULT)
    lon_r = (np.asarray(lon, float) - c0 + 180.0) % 360.0 - 180.0
    order = np.argsort(lon_r); lon_s = lon_r[order]; d = field[:, order]
    if lat[0] > lat[-1]:
        lat = lat[::-1]; d = d[::-1]
    gap = lon_s[0] + 360.0 - lon_s[-1]
    w = (180.0 - lon_s[-1]) / gap if gap > 0 else 0.0
    edge = d[:, -1] * (1 - w) + d[:, 0] * w if 0 < w < 1 else (d[:, 0] if w >= 1 else d[:, -1])
    if lon_s[0] > -180.0 + 1e-9:
        d = np.concatenate([edge[:, None], d], axis=1); lon_s = np.concatenate([[-180.0], lon_s])
    if lon_s[-1] < 180.0 - 1e-9:
        d = np.concatenate([d, d[:, :1]], axis=1); lon_s = np.concatenate([lon_s, [180.0]])
    d_c, lon_c = d, lon_s
    lon_span = (ext[1] - ext[0]) if ext else 360.0
    W = 14.0; map_h = W * (lat1 - lat0) / lon_span; top, bot = 0.95, 0.85; H = map_h + top + bot
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0.03, bot / H, 0.94, map_h / H], projection=proj)
    if ext:
        ax.set_extent([ext[0], ext[1], lat0, lat1], crs=pc)
    else:
        ax.set_extent([-180, 180, lat0, lat1], crs=proj)
    ax.add_feature(cfeature.LAND, facecolor="#f1f0eb", zorder=0)
    norm = BoundaryNorm(levels, len(levels) - 1)
    m = ax.contourf(lon_c, lat, np.ma.masked_invalid(d_c), levels=levels, cmap=plt.get_cmap(cmap, len(levels) - 1), norm=norm,
                    extend="both", transform=proj, zorder=1)
    if cstep:
        cl = [x for x in np.arange(-6, 6.01, cstep) if abs(x) > 1e-9]
        cs = ax.contour(lon_c, lat, np.ma.masked_invalid(d_c), levels=cl, colors="#333", linewidths=0.35, transform=proj, zorder=2)
        ax.clabel(cs, fontsize=5.5, fmt=lambda v: f"{v:+.2g}", inline=True, inline_spacing=2)
    if var in ("t2m", "tp", "sf", "sd", "sdp"):
        ax.add_feature(cfeature.OCEAN, facecolor="#ffffff", zorder=2); ax.add_feature(cfeature.LAKES, facecolor="#ffffff", zorder=2)
    ax.coastlines(resolution="50m", linewidth=0.45, color="#222", zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.25, edgecolor="#666", zorder=3)
    if var != "sst":
        ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.15, edgecolor="#6f6b64", zorder=3)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="#6f6b64", alpha=0.5, xlocs=range(-180, 181, 30), ylocs=range(-60, 91, 30), zorder=4)
    gl.top_labels = gl.right_labels = False; gl.xlabel_style = gl.ylabel_style = {"size": 7, "color": "#555"}
    fig.text(0.03, 1 - 0.14 / H, title, fontsize=13.5, fontweight="bold", va="top")
    fig.text(0.03, 1 - 0.50 / H, sub, fontsize=8.6, color="#444", va="top")
    cax = fig.add_axes([0.25, 0.42 / H, 0.50, 0.14 / H])
    cb = fig.colorbar(m, cax=cax, orientation="horizontal", extend="both", spacing="uniform"); cb.set_label(cb_label, fontsize=8.5)
    cb.ax.tick_params(labelsize=7)
    if len(levels) > 16:                                                        # fine steps: label whole degrees only
        cb.set_ticks([x for x in levels if abs(x - round(x)) < 1e-6])
    fig.savefig(out, dpi=125, pil_kwargs={"quality": 86, "method": 6}); plt.close(fig)


def render_global_maps(ym: str, prev: str | None, out_dir: Path, only=None) -> dict:
    """One image per (variable, period, kind): the ensemble-mean anomaly of this issue against its
    own hindcast, and the change against the previous issue (each anomalised against its own
    start-month hindcast). Plate carrée, most of the world. Files: seas5_map_{var}_{anom|chg}_{period}.webp."""
    import calendar
    now = load_global(ym)
    if not now:
        return {}
    before = load_global(prev, current=False) if prev else {}
    issue_lbl = f"{calendar.month_name[int(ym[4:])]} {ym[:4]} issue"
    prev_lbl = f"{calendar.month_name[int(prev[4:])]} issue" if prev else ""
    periods = _period_sets(ym, prev if before else None)
    # every period of this issue; `prev` is None where the previous issue does not reach it (it had 6 leads
    # before October 2026), and the page says so instead of drawing a change map
    meta = {"periods": [dict(key=k, label=l, kind="month" if len(ln) == 1 else "season", leads=list(ln), prev=(list(lp) if lp else None))
                        for k, l, ln, lp in periods], "vars": {}, "previous": prev,
            "previous_last": (valid_months(prev)[-1] if prev else None)}
    for var, tup in now.items():
        if only and var not in only:
            continue
        fcm, hcm, lat, lon, ztr = tup                                          # fcm: ensemble mean [lead, lat, lon]
        label, units, lv_a, lv_c, cmap, cs_a, cs_c = MAP_SPEC[var]
        entry = {"label": label, "units": units, "anom": {}, "chg": {}, "standardised": ztr is not None}
        pb = before.get(var)
        nl = min(fcm.shape[0], hcm.shape[0])                              # 7 when the forecast supplement and the lead-7 table exist
        for key, plabel, ln, lp in periods:
            if max(ln) > nl or (ztr is not None and f"ztr_mean_{_ls_key(ln)}" not in ztr):
                continue
            idx = [L - 1 for L in ln]
            a = fcm[idx].mean(0) - hcm[idx].mean(0)
            if var in ("sf", "sdp"):                                            # % of the hindcast normal, blank where it is tiny
                nrm = hcm[idx].mean(0); floor = 1.0 if var == "sf" else 10.0     # 1 cm/month of snowfall, 10 mm w.e. of snowpack
                a = np.where(nrm >= floor, 100.0 * fcm[idx].mean(0) / np.maximum(nrm, 1e-6), np.nan)
            if ztr is not None:
                # heights carry the warming trend (+40–60 m over the tropics against a 1993–2016 mean, which is
                # many σ where the year-to-year spread is small): the reference is the hindcast's linear trend
                # evaluated at the valid year, the σ its residual spread — the same construction as the normals.
                # Trend mean / slope / σ per lead set come from the table (_global_hc_table).
                k = _ls_key(ln); ny = 24
                vyear = int(valid_months(ym)[idx[len(idx) // 2]][:4]); target = (vyear - 1993) - (ny - 1) / 2
                ref = ztr[f"ztr_mean_{k}"] + ztr[f"ztr_slope_{k}"] * target; sd = ztr[f"ztr_sd_{k}"]
                a = (fcm[idx].mean(0) - ref) / np.where(sd > 0, sd, np.nan)
            out = out_dir / f"seas5_map_{var}_anom_{key}.webp"
            _global_map(a, lat, lon, var, lv_a, cmap, cs_a, f"SEAS5 {label}{'' if var in ('sf', 'sdp', 'sd') else ' anomaly'} · {plabel} · {issue_lbl}",
                        ("Ensemble-mean snowfall (10:1 snow-to-liquid ratio) as % of the 1993–2016 start-month hindcast mean for the same lead (25 members × 24 years), 1° grid; blank where the normal is under 1 cm a month." if var == "sf" else
                         "Ensemble-mean monthly snowpack (snow depth in water equivalent) as % of the 1993–2016 start-month hindcast mean for the same lead (25 members × 24 years), 1° grid; blank where the normal is under 10 mm." if var == "sdp" else
                         "Ensemble mean of 51 members minus the 1993–2016 start-month hindcast mean (25 members × 24 years): monthly-mean snowpack in mm of water equivalent, 1° grid." if var == "sd" else
                         "Ensemble mean of 51 members minus the 1993–2016 start-month hindcast mean (25 members × 24 years), 1° grid." + (" Heights: departure from the hindcast's linear trend at the valid year, in σ of the residual year-to-year spread; the whole tropical belt sits 3–4σ high in this El Niño." if ztr is not None else "")),
                        ("% of the hindcast normal" if var in ("sf", "sdp") else f"anomaly ({units})"), out)
            entry["anom"][key] = dict(file=out.name, mean=float(np.nanmean(a)))
            if pb is not None and lp is not None:
                fcmp, hcp, ztrp = pb[0], pb[1], pb[4]; ip = [L - 1 for L in lp]          # fcmp: previous ensemble mean
                ap = fcmp[ip].mean(0) - hcp[ip].mean(0)
                if var in ("sf", "sdp"):
                    nrmp = hcp[ip].mean(0); floorp = 1.0 if var == "sf" else 10.0
                    ap = np.where(nrmp >= floorp, 100.0 * fcmp[ip].mean(0) / np.maximum(nrmp, 1e-6), np.nan)
                if ztr is not None and ztrp is not None:                         # previous issue: its own trend reference, this σ
                    kp = _ls_key(lp); nyp = 24
                    refp = ztrp[f"ztr_mean_{kp}"] + ztrp[f"ztr_slope_{kp}"] * ((vyear - 1993) - (nyp - 1) / 2)
                    ap = (fcmp[ip].mean(0) - refp) / np.where(sd > 0, sd, np.nan)
                c = a - ap
                out = out_dir / f"seas5_map_{var}_chg_{key}.webp"
                if max(lp) > pb[0].shape[0]:
                    continue
                _global_map(c, lat, lon, var, lv_c, cmap, cs_c, f"SEAS5 {label}: change since the {prev_lbl} · {plabel} · {issue_lbl}",
                            "Each issue's ensemble mean anomalised against its own start-month hindcast, so this is the shift in the forecast, not drift. Periods the earlier issue does not cover are not drawn.",
                            (f"change in % of normal (points), {issue_lbl} minus {prev_lbl}" if var in ("sf", "sdp") else f"change in ensemble-mean anomaly ({units}), {issue_lbl} minus {prev_lbl}"), out)
                entry["chg"][key] = dict(file=out.name, mean=float(np.nanmean(c)))
        meta["vars"][var] = entry
        print(f"  global maps {var}: {len(entry['anom'])} anomaly, {len(entry['chg'])} change", flush=True)
    return meta


# ── polar caps ───────────────────────────────────────────────────────────────
def hindcast_years(n_samples: int, n_years: int = 24):
    """Year index of each hindcast sample: xarray stacks (number, time) with time fastest."""
    return np.tile(np.arange(n_years), n_samples // n_years)


def detrend_pair(f: np.ndarray, h: np.ndarray, ym: str):
    """f [members, lead], h [samples, lead] hindcast (member-major) → (forecast anomaly vs the
    hindcast's linear trend extrapolated to each lead's valid year, hindcast residuals)."""
    vm = valid_months(ym)
    yrs = 1993 + hindcast_years(h.shape[0])
    x = yrs - yrs.mean()
    fa = np.empty_like(f); hr = np.empty_like(h)
    for L in range(h.shape[1]):
        y = h[:, L]; ok = np.isfinite(y)
        b = (x[ok] * (y[ok] - y[ok].mean())).sum() / (x[ok] ** 2).sum(); a0 = y[ok].mean()
        target = int(vm[L][:4]) - yrs.mean()
        fa[:, L] = f[:, L] - (a0 + b * target)
        hr[:, L] = y - (a0 + b * x)
    return fa, hr

POLAR = (("nh", "polar_n", (60, 90, -180, 180)), ("sh", "polar_s", (-90, -60, -180, 180)))


def _polar_series(path: Path, box) -> dict:
    """{level: box-mean height [sample, lead] in metres} from one polar-cap GRIB."""
    out = {}
    for lev in (10, 50, 100):
        v, lat, lon = load_field(path, "z", level=lev)
        out[lev] = box_mean(v / G0, lat, lon, box)
    return out


def _polar_hc_table(month: str) -> None:
    """hc_build_polar_{MM}: the polar-cap mean height of every hindcast sample [sample, lead] per hemisphere
    and level — all detrend_pair needs."""
    arrays = {}
    for hemi, kind, box in POLAR:
        for lev, h in _polar_series(hc_path(kind, month), box).items():
            arrays[f"{hemi}_z{lev}"] = h
    R.save(R.hc_ref("build_polar", month), **arrays)


def polar_caps(ym: str, members: bool = True) -> dict:
    """Polar-cap height plumes. The forecast cap means come from this issue's GRIBs or, for an earlier issue
    whose GRIBs are gone, from its is_build_polar summary (written whenever the GRIBs are read); the
    hindcast from hc_build_polar_{MM}."""
    out = {}
    fcs = {}
    if all(fc_path(kind, ym).exists() for _, kind, _ in POLAR):
        for hemi, kind, box in POLAR:
            for lev, f in _polar_series(fc_path(kind, ym), box).items():
                fcs[f"{hemi}_z{lev}"] = f
        R.save(R.is_ref("build_polar", ym), **fcs)
    else:
        fcs = R.load(R.is_ref("build_polar", ym)) or {}
    hct = R.load(R.hc_ref("build_polar", ym[4:]))
    if hct is None:
        print(f"  polar caps {ym}: table hc_build_polar_{ym[4:]} missing — skipped", flush=True)
        return out
    for hemi, kind, box in POLAR:
        for lev in (10, 50, 100):
            key = f"{hemi}_z{lev}"
            if key not in fcs or key not in hct:
                continue
            f, h = fcs[key], hct[key]
            n = min(f.shape[1], h.shape[1]); f, h = f[:, :n], h[:, :n]
            # DETRENDED: geopotential carries the warming trend (thickness), so an anomaly vs the
            # 1993–2016 mean reads high by construction in 2026. Fit the hindcast's linear trend
            # per lead across its 24 years and take the anomaly against that line extrapolated to
            # the valid year; the hindcast spread is the spread of the residuals.
            a, hr = detrend_pair(f, h, ym)
            e = dict(**_summ(a), clim_sd=hr.std(0).round(1).tolist(), clim_mean=h.mean(0).round(1).tolist(), units="m",
                     valid=valid_months(ym)[:n], detrended=True)
            if members:
                e["members"] = a.round(1).tolist()
            out[key] = e
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


# What lead month 7 is made of (shown on the page under every product that reaches it)
LEAD7_NOTE = ("Lead month 7 is not in the CDS monthly product (leads 1–6). Forecast: the same SEAS5 run's sub-daily fields "
              "(seasonal-original), averaged over the month the way the monthly product is (instantaneous fields: every step "
              "from 00 UTC on the 1st to the last step before the next month; accumulations: the month's total over its "
              "seconds). Hindcast reference: the same 1993–2016 hindcast, 25 members, from the daily copy of the C3S archive "
              "on AWS (instantaneous fields, interpolated to the CDS grid) and the CDS sub-daily hindcast (accumulations, "
              "50 hPa height, 30/50 hPa wind). Skill is lower at lead 7, and the products apply the same calibration and "
              "significance tests to it as to leads 1–6.")


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
            print(f"  {iss}: no SST forecast + hindcast table, and no is_build_sst summary — skipped", flush=True)
            continue
        nl_iss = np.asarray(next(iter(r.values()))["members"]).shape[1]
        entry = {"valid": valid_months(iss)[:nl_iss], "leads": nl_iss}
        for k, v in r.items():
            e = _summ(v["members"]); e["clim_sd"] = np.asarray(v["clim_sd"]).round(3).tolist()
            if iss == ym:
                e["members"] = np.asarray(v["members"]).round(3).tolist()
            entry[k] = e
        issues[iss] = entry
    if ym not in issues:
        raise SystemExit(f"no SEAS5 {ym} SST data — run fetch first")

    terc = render_terciles(ym, {}, ASSETS)

    prev = previous_issues(ym, 1)[0]
    changes = {}                                                    # multi-panel change figures retired 2026-09-07 (single-map viewer below)
    maps = render_global_maps(ym, prev, ASSETS)
    if not maps:
        print(f"  global maps skipped: gl fields for {ym} not on disk", flush=True)

    polar = polar_caps(ym)
    polar_prev = polar_caps(prev, members=False)

    doc = {
        "generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        "system": "ECMWF SEAS5 (C3S originating_centre ecmwf, system 51)",
        "issue": ym, "issues": sorted(issues), "members": 51, "hindcast": "1993–2016, 25 members",
        "leads": nleads(ym), "lead7": LEAD7_NOTE,
        "index_meta": [dict(key=k, label=l, units=u, note=n) for k, l, u, n in INDEX_META],
        "enso_keys": ENSO_KEYS, "other_keys": OTHER_KEYS,
        "indices": issues,
        "observed": observed_indices(),
        "terciles": terc,
        "changes": changes,
        "maps": maps,
        "previous": prev,
        "polar": polar,
        "polar_previous": polar_prev,
    }
    OUT_JSON.write_text(json.dumps(doc, separators=(",", ":")))
    print(f"wrote {OUT_JSON} ({OUT_JSON.stat().st_size / 1e3:.0f} kB) in {(time.time() - t0) / 60:.1f} min", flush=True)


# ── derived tables (seas5_ref.py) ────────────────────────────────────────────
# What the build needs from the raw hindcast, reduced once per start month; and what the next issue needs
# from this one. render_terciles / render_global_maps / sst_indices / polar_caps read only these.
REF_UNITS = {
    "build_sst": dict(raw=["m:sst"], fn=_sst_hc_table),
    **{f"build_{k}": dict(raw=[f"m:{k}"], fn=(lambda month, k=k: _global_hc_table(k, month)))
       for k in ("gl", "gl_z500", "gl_u850", "gl_v850", "na_snow", "na_snowdepth")},
    "build_polar": dict(raw=["m:polar_n", "m:polar_s"], fn=_polar_hc_table),
}
ISSUE_UNITS = {
    # sst_indices / polar_caps write their summaries as a side effect when the forecast GRIBs are on disk
    "build_sst": lambda ym: sst_indices(ym) if fc_path("sst", ym).exists() else None,
    "build_maps": _global_issue_table,
    "build_polar": lambda ym: polar_caps(ym) if fc_path("polar_n", ym).exists() else None,
}
