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
    # global members + FULL hindcast samples per variable (the tercile bounds need the 600 samples)
    globals_ = {}
    for var, kind, short, fac in (("t2m", "gl", "t2m", 1.0), ("tp", "gl", "tprate", 86400.0 * 1000), ("z500", "gl_z500", "z", 1.0 / G0), ("u850", "gl_u850", "u", 1.0)):
        if fc_path(kind, ym).exists() and hc_path(kind, ym[4:]).exists():
            globals_[var] = (kind, short, fac)
    y0, m0 = ym[:4], int(ym[4:]); vm = valid_months(ym)
    meta = {}
    for var in list(dict.fromkeys(list(globals_) + list(fields))):
        if var in globals_:
            kind, short, fac = globals_[var]
            fc, lat, lon = load_field(fc_path(kind, ym), short); hc, _, _ = load_field(hc_path(kind, ym[4:]), short)
            fc, hc = fc * fac, hc * fac; glob = True
        elif var in fields and var in palettes:
            fc, hc, lat, lon = fields[var]; glob = (lon.max() - lon.min()) > 300
        else:
            continue
        above_c, below_c = palettes[var]
        seasons = []
        for leads in SEASON_LEADS:
            pr = tercile_probs(fc, hc, leads, detrend_ym=ym if var in ("t2m", "z500") else None)
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
            seasons.append(dict(key=key, label=f"{lab} {yv}", leads=list(leads), file=out.name,
                                frac_above=float(np.nanmean(pr["above"] >= 0.40)), frac_below=float(np.nanmean(pr["below"] >= 0.40))))
        meta[var] = dict(label=titles[var], extent="global" if glob else "americas", seasons=seasons, detrended=var in ("t2m", "z500"))
        print(f"  terciles {var}: {len(seasons)} maps ({'global' if glob else 'americas'})", flush=True)
        del fc, hc
    return meta


# ── change since the previous issue ──────────────────────────────────────────
def shared_seasons(ym: str, prev: str) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """(leads_now, leads_prev) for each of this issue's seasons the previous issue
    also covers. A month-earlier start reaches the same calendar months one lead
    later, so its final season falls off the end (DJF from a September start is
    leads 4–6; from August it would be 5–7, which does not exist)."""
    vm_now, vm_prev = valid_months(ym), valid_months(prev)
    out = []
    for leads in SEASON_LEADS:
        months = [vm_now[L - 1] for L in leads]
        if all(m in vm_prev for m in months):
            out.append((leads, tuple(vm_prev.index(m) + 1 for m in months)))
    return out


def load_fields(ym: str) -> dict:
    """{var: (fc, hc, lat, lon)} for the tercile / change maps, whatever is on disk."""
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
    return fields


def _change_panel(ax, d, lat, lon, var, levels, cmap, title, pc):
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm
    import cartopy.feature as cfeature
    ax.set_extent([-170, -30, -60, 75], crs=pc)
    ax.add_feature(cfeature.LAND, facecolor="#f4f4f1", zorder=0)
    m = ax.pcolormesh(lon, lat, d, cmap=plt.get_cmap(cmap, len(levels) - 1), norm=BoundaryNorm(levels, len(levels) - 1),
                      transform=pc, shading="auto", zorder=1)
    if var in ("t2m", "tp"):
        ax.add_feature(cfeature.OCEAN, facecolor="#ffffff", zorder=2)
        ax.add_feature(cfeature.LAKES, facecolor="#ffffff", zorder=2)
    ax.coastlines(linewidth=0.5, color="#444", zorder=3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="#777", zorder=3)
    ax.add_feature(cfeature.STATES, linewidth=0.2, edgecolor="#999", zorder=3)
    ax.set_title(title, fontsize=12, loc="left")
    return m


def render_changes(ym: str, prev: str, now: dict, before: dict, out_dir: Path) -> dict:
    """Per variable, two figures: the shared SEASONS side by side, and the shared MONTHS on a
    2 × 3 grid. Both are ensemble-mean anomaly of this issue minus the previous issue, each
    anomaly against its own start-month hindcast."""
    import calendar
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    pairs = shared_seasons(ym, prev)
    vm_now, vm_prev = valid_months(ym), valid_months(prev)
    months = [(L + 1, vm_prev.index(v) + 1, v) for L, v in enumerate(vm_now) if v in vm_prev]   # (lead_now, lead_prev, month)
    if not pairs and not months:
        return {}
    spec = {"t2m": ("2 m temperature", "°C", [-2, -1.5, -1, -0.5, -0.25, 0.25, 0.5, 1, 1.5, 2], "RdBu_r"),
            "tp": ("Precipitation", "mm/day", [-2, -1.5, -1, -0.5, -0.2, 0.2, 0.5, 1, 1.5, 2], "BrBG"),
            "z500": ("500 hPa height", "m", [-40, -30, -20, -10, -5, 5, 10, 20, 30, 40], "RdBu_r")}
    proj, pc = ccrs.PlateCarree(central_longitude=-90), ccrs.PlateCarree()
    sub = ("Each issue's ensemble mean is an anomaly against its own start-month hindcast, so this is the shift in the forecast, not a drift artefact.\n"
           "Only the periods both issues cover are shown; the newest has no counterpart in the earlier issue.")
    meta = {}
    for var in ("t2m", "tp", "z500"):
        if var not in now or var not in before:
            continue
        fc_n, hc_n, lat, lon = now[var]; fc_p, hc_p, _, _ = before[var]
        title, units, levels, cmap = spec[var]
        cb_label = f"change in ensemble-mean anomaly ({units}), {calendar.month_name[int(ym[4:])]} issue minus {calendar.month_name[int(prev[4:])]} issue"
        entry = {"previous": prev}
        # seasonal
        if pairs:
            W = 5.2 * len(pairs) + 1.2
            H, adj = map_layout(W, len(pairs), 1, top_in=1.25, bottom_in=1.15)
            fig, axes = plt.subplots(1, len(pairs), figsize=(W, H), subplot_kw={"projection": proj}, squeeze=False)
            seasons = []
            for ax, (ln, lp) in zip(axes[0], pairs):
                d = (_season_means(fc_n, ln).mean(0) - _season_means(hc_n, ln).mean(0)) - (_season_means(fc_p, lp).mean(0) - _season_means(hc_p, lp).mean(0))
                y0s, y1s = vm_now[ln[0] - 1][:4], vm_now[ln[-1] - 1][:4]
                m = _change_panel(ax, d, lat, lon, var, levels, cmap, f"{season_label(ym, ln)} {y0s if y0s == y1s else y0s + '–' + y1s[2:]}", pc)
                seasons.append(dict(label=season_label(ym, ln), leads=list(ln), mean_change=float(np.nanmean(d))))
            cax = fig.add_axes([0.28, 0.55 / H, 0.44, 0.16 / H])
            cb = fig.colorbar(m, cax=cax, orientation="horizontal", extend="both"); cb.set_label(cb_label, fontsize=10); cb.ax.tick_params(labelsize=9)
            head_text(fig, H, f"SEAS5 {title}: change since the {calendar.month_name[int(prev[4:])]} issue, by season", sub)
            fig.subplots_adjust(**adj)
            out = out_dir / f"seas5_change_{var}.webp"
            fig.savefig(out, dpi=105, pil_kwargs={"quality": 84, "method": 6}); plt.close(fig)
            entry.update(file=out.name, seasons=seasons); print(f"  wrote {out.name}", flush=True)
        # monthly, 2 × 3
        if months:
            ncols, nrows = 3, 2
            W = 15.5
            H, adj = map_layout(W, ncols, nrows, top_in=1.25, bottom_in=1.15, hspace=0.12)
            fig, axes = plt.subplots(nrows, ncols, figsize=(W, H), subplot_kw={"projection": proj})
            mlist = []
            for ax, (ln, lp, v) in zip(axes.ravel(), months):
                d = (fc_n[:, ln - 1].mean(0) - hc_n[:, ln - 1].mean(0)) - (fc_p[:, lp - 1].mean(0) - hc_p[:, lp - 1].mean(0))
                m = _change_panel(ax, d, lat, lon, var, levels, cmap, f"{calendar.month_abbr[int(v[5:])]} {v[:4]}", pc)
                mlist.append(dict(month=v, mean_change=float(np.nanmean(d))))
            for ax in axes.ravel()[len(months):]:
                ax.set_visible(False)
            cax = fig.add_axes([0.28, 0.55 / H, 0.44, 0.16 / H])
            cb = fig.colorbar(m, cax=cax, orientation="horizontal", extend="both"); cb.set_label(cb_label, fontsize=10); cb.ax.tick_params(labelsize=9)
            head_text(fig, H, f"SEAS5 {title}: change since the {calendar.month_name[int(prev[4:])]} issue, by month", sub)
            fig.subplots_adjust(**adj)
            out = out_dir / f"seas5_change_{var}_monthly.webp"
            fig.savefig(out, dpi=105, pil_kwargs={"quality": 84, "method": 6}); plt.close(fig)
            entry.update(file_monthly=out.name, months=mlist); print(f"  wrote {out.name}", flush=True)
        meta[var] = entry
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


def load_global(ym: str) -> dict:
    """{var: (fc[sample, lead, lat, lon], hc_mean[lead, lat, lon], lat, lon)} from the global 1° kinds.
    The hindcast (600 samples × 6 leads × 181 × 360) is reduced to its mean on load."""
    out = {}
    if fc_path("gl", ym).exists() and hc_path("gl", ym[4:]).exists():
        for var, short, fac in (("t2m", "t2m", 1.0), ("tp", "tprate", 86400.0 * 1000), ("sst", "sst", 1.0)):
            try:
                fc, lat, lon = load_field(fc_path("gl", ym), short)
                hc, _, _ = load_field(hc_path("gl", ym[4:]), short)
            except Exception as e:                                          # noqa: BLE001
                print(f"  global {var} {ym}: {str(e)[:100]}", flush=True); continue
            out[var] = (fc * fac, np.nanmean(hc, axis=0) * fac, lat, lon); del hc
    if fc_path("gl_z500", ym).exists() and hc_path("gl_z500", ym[4:]).exists():
        fc, lat, lon = load_field(fc_path("gl_z500", ym), "z")
        hc, _, _ = load_field(hc_path("gl_z500", ym[4:]), "z")
        # samples are member-major, year fastest (_stack_samples): per-year ensemble means give the
        # interannual σ used to standardise heights (user 2026-09-07: "standardize 500 mb heights")
        ny = 24 if hc.shape[0] % 24 == 0 else 1
        hc_ym = np.nanmean(hc.reshape(-1, ny, *hc.shape[1:]), axis=0) / G0 if ny > 1 else None
        out["z500"] = (fc / G0, np.nanmean(hc, axis=0) / G0, lat, lon, hc_ym); del hc
    if fc_path("gl_u850", ym).exists() and hc_path("gl_u850", ym[4:]).exists():
        fc, lat, lon = load_field(fc_path("gl_u850", ym), "u")
        hc, _, _ = load_field(hc_path("gl_u850", ym[4:]), "u")
        out["u850"] = (fc, np.nanmean(hc, axis=0), lat, lon); del hc
    if fc_path("gl_v850", ym).exists() and hc_path("gl_v850", ym[4:]).exists():
        fc, lat, lon = load_field(fc_path("gl_v850", ym), "v")
        hc, _, _ = load_field(hc_path("gl_v850", ym[4:]), "v")
        out["v850"] = (fc, np.nanmean(hc, axis=0), lat, lon); del hc
    if fc_path("na_snow", ym).exists() and hc_path("na_snow", ym[4:]).exists():
        import calendar as _cal
        days = np.array([_cal.monthrange(int(v[:4]), int(v[5:]))[1] for v in valid_months(ym)], dtype=np.float32)
        fac = 86400.0 * 100.0 * 10.0 * days[None, :, None, None]             # m w.e. s⁻¹ → cm w.e./month → cm snow (10:1)
        fc, lat, lon = load_field(fc_path("na_snow", ym), "mtsfr")
        hc, _, _ = load_field(hc_path("na_snow", ym[4:]), "mtsfr")
        out["sf"] = (fc * fac, np.nanmean(hc, axis=0) * fac[0], lat, lon); del hc
    if fc_path("na_snowdepth", ym).exists() and hc_path("na_snowdepth", ym[4:]).exists():
        fc, lat, lon = load_field(fc_path("na_snowdepth", ym), "sd")                   # m of water equivalent
        hc, _, _ = load_field(hc_path("na_snowdepth", ym[4:]), "sd")
        hcm = np.nanmean(hc, axis=0) * 1000.0; del hc
        out["sd"] = (fc * 1000.0, hcm, lat, lon)                                     # mm w.e.
        out["sdp"] = (fc * 1000.0, hcm, lat, lon)
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
    for leads in SEASON_LEADS:
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
    from cartopy.util import add_cyclic_point
    pc = ccrs.PlateCarree(); proj = ccrs.PlateCarree(central_longitude=MAP_CENTRAL.get(var, MAP_CENTRAL_DEFAULT))
    ext = MAP_EXTENT.get(var)
    lat0, lat1 = (ext[2], ext[3]) if ext else ((-70, 70) if var == "sst" else (-60, 85))
    order = np.argsort(lon); lon_s = lon[order]; d = field[:, order]
    if lat[0] > lat[-1]:
        lat = lat[::-1]; d = d[::-1]
    if ext:
        d_c, lon_c = d, lon_s
    else:
        d_c, lon_c = add_cyclic_point(d, coord=lon_s)
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
                    extend="both", transform=pc, zorder=1)
    if cstep:
        cl = [x for x in np.arange(-6, 6.01, cstep) if abs(x) > 1e-9]
        cs = ax.contour(lon_c, lat, np.ma.masked_invalid(d_c), levels=cl, colors="#333", linewidths=0.35, transform=pc, zorder=2)
        ax.clabel(cs, fontsize=5.5, fmt=lambda v: f"{v:+.2g}", inline=True, inline_spacing=2)
    if var in ("t2m", "tp", "sf", "sd", "sdp"):
        ax.add_feature(cfeature.OCEAN, facecolor="#ffffff", zorder=2); ax.add_feature(cfeature.LAKES, facecolor="#ffffff", zorder=2)
    ax.coastlines(resolution="50m", linewidth=0.45, color="#222", zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.25, edgecolor="#666", zorder=3)
    if var != "sst":
        ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.15, edgecolor="#999", zorder=3)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="#888", alpha=0.5, xlocs=range(-180, 181, 30), ylocs=range(-60, 91, 30), zorder=4)
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
    before = load_global(prev) if prev else {}
    issue_lbl = f"{calendar.month_name[int(ym[4:])]} {ym[:4]} issue"
    prev_lbl = f"{calendar.month_name[int(prev[4:])]} issue" if prev else ""
    periods = _period_sets(ym, prev if before else None)
    meta = {"periods": [dict(key=k, label=l, kind="month" if len(ln) == 1 else "season") for k, l, ln, _ in periods], "vars": {}}
    for var, tup in now.items():
        if only and var not in only:
            continue
        fc, hcm, lat, lon = tup[:4]; hc_ym = tup[4] if len(tup) > 4 else None
        label, units, lv_a, lv_c, cmap, cs_a, cs_c = MAP_SPEC[var]
        fcm = np.nanmean(fc, axis=0)                                            # [lead, lat, lon]
        entry = {"label": label, "units": units, "anom": {}, "chg": {}, "standardised": hc_ym is not None}
        pb = before.get(var)
        for key, plabel, ln, lp in periods:
            idx = [L - 1 for L in ln]
            a = fcm[idx].mean(0) - hcm[idx].mean(0)
            if var in ("sf", "sdp"):                                            # % of the hindcast normal, blank where it is tiny
                nrm = hcm[idx].mean(0); floor = 1.0 if var == "sf" else 10.0     # 1 cm/month of snowfall, 10 mm w.e. of snowpack
                a = np.where(nrm >= floor, 100.0 * fcm[idx].mean(0) / np.maximum(nrm, 1e-6), np.nan)
            if hc_ym is not None:
                # heights carry the warming trend (+40–60 m over the tropics against a 1993–2016 mean, which is
                # many σ where the year-to-year spread is small): the reference is the hindcast's linear trend
                # evaluated at the valid year, the σ its residual spread — the same construction as the normals
                ym_ = hc_ym[:, idx].mean(1); ny = ym_.shape[0]; yr = np.arange(ny) - (ny - 1) / 2
                slope = (yr[:, None, None] * (ym_ - ym_.mean(0))).sum(0) / (yr ** 2).sum()
                vyear = int(valid_months(ym)[idx[len(idx) // 2]][:4]); target = (vyear - 1993) - (ny - 1) / 2
                ref = ym_.mean(0) + slope * target; sd = np.nanstd(ym_ - slope[None] * yr[:, None, None], axis=0)
                a = (fcm[idx].mean(0) - ref) / np.where(sd > 0, sd, np.nan)
            out = out_dir / f"seas5_map_{var}_anom_{key}.webp"
            _global_map(a, lat, lon, var, lv_a, cmap, cs_a, f"SEAS5 {label}{'' if var in ('sf', 'sdp', 'sd') else ' anomaly'} · {plabel} · {issue_lbl}",
                        ("Ensemble-mean snowfall (10:1 snow-to-liquid ratio) as % of the 1993–2016 start-month hindcast mean for the same lead (25 members × 24 years), 1° grid; blank where the normal is under 1 cm a month." if var == "sf" else
                         "Ensemble-mean monthly snowpack (snow depth in water equivalent) as % of the 1993–2016 start-month hindcast mean for the same lead (25 members × 24 years), 1° grid; blank where the normal is under 10 mm." if var == "sdp" else
                         "Ensemble mean of 51 members minus the 1993–2016 start-month hindcast mean (25 members × 24 years): monthly-mean snowpack in mm of water equivalent, 1° grid." if var == "sd" else
                         "Ensemble mean of 51 members minus the 1993–2016 start-month hindcast mean (25 members × 24 years), 1° grid." + (" Heights: departure from the hindcast's linear trend at the valid year, in σ of the residual year-to-year spread; the whole tropical belt sits 3–4σ high in this El Niño." if hc_ym is not None else "")),
                        ("% of the hindcast normal" if var in ("sf", "sdp") else f"anomaly ({units})"), out)
            entry["anom"][key] = dict(file=out.name, mean=float(np.nanmean(a)))
            if pb is not None and lp is not None:
                fcp, hcp = pb[0], pb[1]; ip = [L - 1 for L in lp]
                ap = np.nanmean(fcp, axis=0)[ip].mean(0) - hcp[ip].mean(0)
                if var in ("sf", "sdp"):
                    nrmp = hcp[ip].mean(0); floorp = 1.0 if var == "sf" else 10.0
                    ap = np.where(nrmp >= floorp, 100.0 * np.nanmean(fcp, axis=0)[ip].mean(0) / np.maximum(nrmp, 1e-6), np.nan)
                if hc_ym is not None and len(pb) > 4 and pb[4] is not None:     # previous issue: its own trend reference, this σ
                    ymp = pb[4][:, ip].mean(1); nyp = ymp.shape[0]; yrp = np.arange(nyp) - (nyp - 1) / 2
                    slp = (yrp[:, None, None] * (ymp - ymp.mean(0))).sum(0) / (yrp ** 2).sum()
                    refp = ymp.mean(0) + slp * ((vyear - 1993) - (nyp - 1) / 2)
                    ap = (np.nanmean(fcp, axis=0)[ip].mean(0) - refp) / np.where(sd > 0, sd, np.nan)
                c = a - ap
                out = out_dir / f"seas5_map_{var}_chg_{key}.webp"
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

def polar_caps(ym: str, members: bool = True) -> dict:
    out = {}
    for hemi, kind, box in (("nh", "polar_n", (60, 90, -180, 180)), ("sh", "polar_s", (-90, -60, -180, 180))):
        fcp, hcp = fc_path(kind, ym), hc_path(kind, ym[4:])
        if not (fcp.exists() and hcp.exists()):
            continue
        for lev in (10, 50, 100):
            fc, lat, lon = load_field(fcp, "z", level=lev)
            hc, _, _ = load_field(hcp, "z", level=lev)
            f = box_mean(fc / G0, lat, lon, box); h = box_mean(hc / G0, lat, lon, box)
            # DETRENDED: geopotential carries the warming trend (thickness), so an anomaly vs the
            # 1993–2016 mean reads high by construction in 2026. Fit the hindcast's linear trend
            # per lead across its 24 years and take the anomaly against that line extrapolated to
            # the valid year; the hindcast spread is the spread of the residuals.
            a, hr = detrend_pair(f, h, ym)
            e = dict(**_summ(a), clim_sd=hr.std(0).round(1).tolist(), clim_mean=h.mean(0).round(1).tolist(), units="m",
                     valid=valid_months(ym), detrended=True)
            if members:
                e["members"] = a.round(1).tolist()
            out[f"{hemi}_z{lev}"] = e
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

    fields = load_fields(ym)
    terc = render_terciles(ym, fields, ASSETS) if fields else {}

    prev = previous_issues(ym, 1)[0]
    before = load_fields(prev)
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
