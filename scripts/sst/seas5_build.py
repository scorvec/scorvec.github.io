#!/usr/bin/env python3
"""SEAS5 outlook: indices, tercile maps, polar-cap plumes → JSON + WebP.

Imported by seas5_outlook.py (which owns the CDS retrieval); see its docstring
for what each product is and why it is built the way it is.
"""

from __future__ import annotations

import json
import textwrap
import time
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*cfgrib.*")
warnings.filterwarnings("ignore", category=RuntimeWarning)

from seas5_outlook import (
    ASSETS,
    MAXLEAD,
    MONTHLY_LEADS,
    PDO_PATTERN,
    SIGMA_PATH,
    fc_path,
    hc_path,
    nleads,
    previous_issues,
)
import seas5_ref as R
import mapstyle as MS

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
    da = da.expand_dims("sample") if not extra else da.stack(sample=extra)
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
    if var not in ds.data_vars:  # e.g. a single-variable file under another name
        if len(ds.data_vars) != 1:  # never guess in a multi-variable file
            raise KeyError(f"{path.name}: no variable {var!r} (has {list(ds.data_vars)})")
        var = next(iter(ds.data_vars))
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
                if not (
                    np.allclose(z["lat"], lat)
                    and np.allclose(((z["lon"] + 180) % 360) - 180, ((lon + 180) % 360) - 180)
                ):
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
    elat, elon = eof.lat.values, eof.lon.values  # lon 110..260 (0–360)
    lon360 = np.where(lon < 0, lon + 360, lon)
    order = np.argsort(lon360)
    a = xr.DataArray(anom[:, :, :, order], dims=("s", "l", "lat", "lon"), coords={"lat": lat, "lon": lon360[order]})
    a = a.interp(lat=elat, lon=elon, method="nearest")
    w = np.cos(np.deg2rad(eof.lat))
    den_full = float(((eof**2) * w).sum())
    num = (a * eof * w).sum(("lat", "lon"), skipna=True)
    den = ((eof**2) * w).where(a.notnull()).sum(("lat", "lon"))
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
    R.save(
        R.hc_ref("build_sst", month),
        meta=dict(boxes=list(BOXES)),
        box=hb,
        hc_mean=hc_mean.astype(np.float32),
        pdo=pdo,
        lat=lat,
        lon=lon,
    )


def _save_sst_issue(ym: str, r: dict) -> None:
    """is_build_sst_{ym}: the index members, so a later issue draws this one's plume without its GRIBs."""
    R.save(
        R.is_ref("build_sst", ym),
        meta=dict(keys=list(r)),
        **{f"{k}_{f}": np.asarray(v[f], dtype=np.float64) for k, v in r.items() for f in ("members", "clim_sd")},
    )


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
    nl = min(
        fc.shape[1], ref["hc_mean"].shape[0], ref["box"].shape[-1]
    )  # 7 when both the forecast and the table reach lead 7
    if nl < fc.shape[1]:
        print(f"  {ym}: SST indices at {nl} leads (the hindcast table has no lead {fc.shape[1]})", flush=True)
    fc = fc[:, :nl]
    hc_mean = ref["hc_mean"][:nl]  # [lead, lat, lon]
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
    f34, h34 = anom("nino34")
    ft, ht = anom("trop")
    months = [((month - 1 + L) % 12) + 1 for L in range(nl)]
    sc = np.array([scale.get(m, 1.0) for m in months])
    out["rnino34"] = dict(members=(f34 - ft) * sc, clim_sd=((h34 - ht) * sc).std(0))
    # Trans-Niño: standardised by the hindcast spread at each lead
    f12, h12 = anom("nino12")
    f4, h4 = anom("nino4")
    z12, z4 = f12 / h12.std(0), f4 / h4.std(0)
    out["tni"] = dict(members=z12 - z4, clim_sd=(h12 / h12.std(0) - h4 / h4.std(0)).std(0))
    # AMO relative, IOD, ATL3
    fna, hna = anom("natl")
    fg, hg = anom("glob")
    out["amo"] = dict(members=fna - fg, clim_sd=(hna - hg).std(0))
    fw, hw = anom("iod_w")
    fe, he = anom("iod_e")
    out["iod"] = dict(members=fw - fe, clim_sd=(hw - he).std(0))
    fa3, ha3 = anom("atl3")
    out["atl3"] = dict(members=fa3, clim_sd=ha3.std(0))
    # PDO from the gridded anomaly (forecast); the hindcast years' PDO (the climatological spread) is in the table
    out["pdo"] = dict(members=_pdo_index(fc - hc_mean[None], lat, lon, fg), clim_sd=ref["pdo"][:, :nl].std(0))
    _save_sst_issue(ym, out)
    return out


def _summ(members: np.ndarray) -> dict:
    """members [sample, lead] → quantile summary per lead."""
    q = np.nanpercentile(members, [10, 25, 50, 75, 90], axis=0)
    return dict(
        mean=np.nanmean(members, 0).round(3).tolist(),
        p10=q[0].round(3).tolist(),
        p25=q[1].round(3).tolist(),
        p50=q[2].round(3).tolist(),
        p75=q[3].round(3).tolist(),
        p90=q[4].round(3).tolist(),
    )


def valid_months(ym: str, n: int | None = None) -> list[str]:
    """The calendar months of an issue's leads: nleads(ym) of them (7 from the October 2026 issue), or n."""
    y, m = int(ym[:4]), int(ym[4:])
    return [f"{y + (m - 1 + L) // 12}-{((m - 1 + L) % 12) + 1:02d}" for L in range(n or nleads(ym))]


# ── seasonal means of monthly means: each month weighted by its length ──────
def month_weights(months) -> np.ndarray:
    """Weights of calendar months ("YYYY-MM") in the time mean of their monthly means: each month's length in its own
    year (a leap February counts 29 days), normalised to sum to 1."""
    import calendar as _cal

    d = np.array([_cal.monthrange(int(v[:4]), int(v[5:7]))[1] for v in months], dtype=np.float64)
    return d / d.sum()


def lead_weights(ym: str, leads) -> np.ndarray:
    """month_weights of the valid months of `leads` for a run started in ym (an issue, or one hindcast start)."""
    vm = valid_months(ym, max(leads))
    return month_weights([vm[L - 1] for L in leads])


def hindcast_lead_weights(month, leads, n_samples: int, n_years: int = 24) -> np.ndarray:
    """[sample, len(leads)] lead_weights of every hindcast sample, each from its own start year (1993 + year index,
    member-major with the year fastest as _stack_samples orders them)."""
    per_year = np.array([lead_weights(f"{1993 + y}{int(month):02d}", leads) for y in range(n_years)])
    if n_samples % n_years:
        return np.broadcast_to(per_year.mean(0), (n_samples, len(leads)))
    return per_year[hindcast_years(n_samples, n_years)]


def day_weighted(meta: dict | None, leads) -> bool:
    """Whether a derived table's statistics for this lead set are day-weighted means of the monthly values (tables
    record those lead sets in meta["day_weighted"]; one built before the weighting holds plain means, and products
    compare the members with it the same way so the two sides stay consistent)."""
    return [int(x) for x in leads] in [[int(x) for x in p] for p in (meta or {}).get("day_weighted", [])]


# ── terciles over the Americas ───────────────────────────────────────────────
def _season_means(vals: np.ndarray, leads: tuple[int, ...], w=None) -> np.ndarray:
    """Mean over the lead months → [sample, lat, lon]: weighted by `w` ([len(leads)] or [sample, len(leads)], e.g.
    lead_weights) or a plain mean when w is None."""
    idx = [L - 1 for L in leads]
    if w is None:
        return vals[:, idx].mean(axis=1)  # [sample, lat, lon]
    w = np.broadcast_to(np.asarray(w, dtype=np.float64), (vals.shape[0], len(idx)))
    return np.einsum("sl,sl...->s...", w, vals[:, idx]).astype(vals.dtype)


def tercile_probs(fc: np.ndarray, hc: np.ndarray, leads, detrend_ym: str | None = None, ym: str | None = None) -> dict:
    """Model-climatology terciles: thresholds from the 600 hindcast seasonal means
    per grid point, probabilities from the 51 forecast members. With `detrend_ym` the
    hindcast's linear trend (per grid point) is removed and the members are counted against
    it extrapolated to the season's valid year — for temperature and heights, where a
    1993–2016 base alone puts the whole tropics in the upper tercile. With the issue known (`ym` or
    `detrend_ym`) the seasonal means weight each month by its length (lead_weights)."""
    iss = ym or detrend_ym
    f = _season_means(fc, leads, lead_weights(iss, leads) if iss else None)
    h = _season_means(hc, leads, hindcast_lead_weights(iss[4:], leads, hc.shape[0]) if iss else None)
    if detrend_ym is not None:
        yrs = 1993 + hindcast_years(h.shape[0])
        x = (yrs - yrs.mean())[:, None, None]
        hm = np.nanmean(h, axis=0, keepdims=True)
        b = np.nansum(x * (h - hm), axis=0) / float((x[:, 0, 0] ** 2).sum())
        target = int(valid_months(detrend_ym)[leads[len(leads) // 2] - 1][:4]) - yrs.mean()
        h = h - b[None] * x
        f = f - b[None] * target
    lo, hi = np.nanpercentile(h, [100 / 3, 200 / 3], axis=0)
    below = (f < lo[None]).mean(0)
    above = (f > hi[None]).mean(0)
    normal = 1.0 - below - above
    return dict(below=below, normal=normal, above=above, ens_anom=f.mean(0) - h.mean(0))


def tercile_bounds(hc: np.ndarray, leads, detrend: bool, month=None) -> dict:
    """The hindcast half of tercile_probs, kept as a table: per grid point the 1/3 and 2/3 bounds of the
    600 seasonal means and, with `detrend`, the per-point linear-trend slope b (per year). With the start
    `month` the seasonal means weight each month by its length in that sample's year."""
    h = _season_means(hc, leads, hindcast_lead_weights(month, leads, hc.shape[0]) if month is not None else None)
    out = {}
    if detrend:
        yrs = 1993 + hindcast_years(h.shape[0])
        x = (yrs - yrs.mean())[:, None, None]
        hm = np.nanmean(h, axis=0, keepdims=True)
        b = np.nansum(x * (h - hm), axis=0) / float((x[:, 0, 0] ** 2).sum())
        h = h - b[None] * x
        out["b"] = b
        out["ymean"] = np.float64(yrs.mean())
    out["lo"], out["hi"] = np.nanpercentile(h, [100 / 3, 200 / 3], axis=0)
    return out


def tercile_probs_ref(fc: np.ndarray, tb: dict, leads, ym: str, weighted: bool = False) -> dict:
    """tercile_probs against stored bounds (tercile_bounds): the forecast members are moved by the trend
    extrapolated to the season's valid year when the bounds are detrended. `weighted`: the bounds are day-weighted
    seasonal means (day_weighted), so the members are averaged the same way."""
    f = _season_means(fc, leads, lead_weights(ym, leads) if weighted else None)
    if "b" in tb:
        target = int(valid_months(ym)[leads[len(leads) // 2] - 1][:4]) - float(tb["ymean"])
        f = f - tb["b"][None] * target
    below = (f < tb["lo"][None]).mean(0)
    above = (f > tb["hi"][None]).mean(0)
    return dict(below=below, normal=1.0 - below - above, above=above)


def season_label(ym: str, leads) -> str:
    m0 = int(ym[4:])
    return "".join(MONTHS[(m0 - 1 + L - 1) % 12] for L in leads)


# Tercile probability bins: six steps to 100 % so a 95 % cell reads darker than an 82 % one.
TERC_BINS = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.001]
TERC_ALPHA, TERC_FDR_Q = 0.05, 0.10  # the tercile maps' signal test (tercile_signal)
TERC_TEST_NOTE = "and only where the members' split differs significantly from one third each (chi-square, p < 0.05 and FDR q 0.10 over the map)"


def tercile_signal(pr: dict, n_eff) -> np.ndarray:
    """True where the members' split into the three terciles differs from one third each by more than drawing n_eff
    members from the model's own climate would give: Pearson's chi-square on 2 degrees of freedom,
    X² = 3 n_eff Σ_k (p_k − 1/3)², p = exp(−X²/2), with p < TERC_ALPHA and a Benjamini–Hochberg discovery at TERC_FDR_Q
    among the finite cells of the map. n_eff: one system's member count, or for an equal-weight mean of K systems
    with n_k members 1 / Σ_k (1/K)² / n_k, the sample size with the mean's sampling variance (the members are taken
    as independent). At 51 members a 40/33/27 split is p 0.5 and a 53/27/20 split p 0.01: the 40 % colour floor
    alone is not a signal."""
    from quantmet.stats import benjamini_hochberg

    a, b = np.asarray(pr["above"], float), np.asarray(pr["below"], float)
    x2 = 3.0 * np.asarray(n_eff, float) * ((a - 1 / 3) ** 2 + (b - 1 / 3) ** 2 + (1.0 - a - b - 1 / 3) ** 2)
    p = np.where(np.isfinite(x2), np.exp(-x2 / 2.0), np.nan)
    return benjamini_hochberg(p, TERC_FDR_Q) & (p < TERC_ALPHA)


# ── the anomaly and change maps' test: is an ensemble-mean shift more than sampling noise? ──
SHIFT_FDR_Q = 0.10
N_EFF_HC = 24  # a hindcast mean counts its 24 years, not its members: the members of one year share that year's ocean
SHIFT_TEST_NOTE = "Shaded only where significant (Welch t-test, false-discovery rate 0.10 over the map)"


def shift_signal(diff, terms, domain=None) -> np.ndarray:
    """True where `diff` ([lat, lon], a difference of means) is further from zero than sampling noise allows:
    t = diff / √Σv over the independent sampling variances of the means it is made of, `terms` = [(v, dof), ...] (v
    scalar or [lat, lon]; a NaN v makes the cell untestable, a zero one is an absent term), Welch–Satterthwaite degrees
    of freedom (Σv)² / Σ v²/dof, a two-sided p, and a Benjamini–Hochberg discovery at SHIFT_FDR_Q among the finite cells
    inside `domain` (the cells the map shows). Cells with no spread at all are never significant."""
    from scipy import stats
    from quantmet.stats import benjamini_hochberg

    d = np.asarray(diff, dtype=np.float64)
    vs = [np.broadcast_to(np.asarray(v, dtype=np.float64), d.shape) for v, _ in terms]
    se2 = np.sum(vs, axis=0)
    den = np.sum([v * v / np.maximum(np.asarray(k, dtype=np.float64), 1.0) for v, (_, k) in zip(vs, terms)], axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = d / np.sqrt(se2)
        df = se2 * se2 / den
    ok = np.isfinite(t) & np.isfinite(df) & (se2 > 0)
    p = np.full(d.shape, np.nan)
    p[ok] = 2.0 * stats.t.sf(np.abs(t[ok]), df[ok])
    if domain is not None:
        p = np.where(domain, p, np.nan)
    return benjamini_hochberg(p, SHIFT_FDR_Q)


def welch_terms(sd_f, n_f: int, sd_h, n_h: int = N_EFF_HC) -> list:
    """shift_signal terms of a forecast-minus-hindcast mean: n_f members with spread sd_f, and the hindcast mean with
    the single-member spread sd_h over n_h independent years (the CanSIPS maps' convention on the same page)."""
    return [(np.asarray(sd_f, float) ** 2 / n_f, n_f - 1), (np.asarray(sd_h, float) ** 2 / n_h, n_h - 1)]


def trend_read_var(s2, x0: float, n: int = N_EFF_HC):
    """Sampling variance of a straight line fitted to n yearly values with residual variance s2 and read x0 years from
    their mean year: s2 (1/n + x0² / Σ(x - x̄)²). A 24-year trend read 20 years out carries ~10× the variance of the
    24-year mean, which is what testing a height anomaly against the extrapolated trend has to pay."""
    yr = np.arange(n) - (n - 1) / 2.0
    return np.asarray(s2, float) * (1.0 / n + x0 * x0 / float((yr**2).sum()))


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
MAP_ASPECT = 135.0 / 140.0  # height / width of one panel


def map_layout(
    width: float,
    ncols: int,
    nrows: int,
    top_in: float,
    bottom_in: float,
    wspace: float = 0.05,
    hspace: float = 0.10,
    side: float = 0.02,
):
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
    global 1° fields where they exist (t2m, tp, z500, u850) and the Americas box otherwise.
    Files seas5_terc_{var}_{SEASON_YYYY}.webp."""
    import calendar
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from cartopy.util import add_cyclic_point
    import mapstyle as MS

    bins = TERC_BINS
    palettes = {
        "t2m": (TERC_PALETTES["warm"], TERC_PALETTES["cool"]),
        "tp": (TERC_PALETTES["wet"], TERC_PALETTES["dry"]),
        "z500": (TERC_PALETTES["warm"], TERC_PALETTES["cool"]),
        "u850": (TERC_PALETTES["warm"], TERC_PALETTES["cool"]),
        "sst": (TERC_PALETTES["warm"], TERC_PALETTES["cool"]),
    }
    titles = {
        "t2m": "2 m temperature",
        "tp": "Precipitation",
        "z500": "500 hPa height",
        "u850": "850 hPa zonal wind",
        "sst": "Sea surface temperature",
    }
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
    y0, m0 = ym[:4], int(ym[4:])
    vm = valid_months(ym)
    meta = {}
    for var in globals_:
        kind, short, fac, tab = globals_[var]
        fc, lat, lon = load_field(fc_path(kind, ym), short)
        fc = fc * fac
        glob = True
        above_c, below_c = palettes[var]
        seasons = []
        for si, leads in enumerate(SEASON_LEADS):
            if max(leads) > fc.shape[1] or f"terc_{var}_lo_{si}" not in tab:
                print(
                    f"  terciles {var} {season_label(ym, leads)}: lead {max(leads)} not in the forecast or the hindcast table — skipped",
                    flush=True,
                )
                continue
            tb = {k: tab[f"terc_{var}_{k}_{si}"] for k in ("lo", "hi", "b", "ymean") if f"terc_{var}_{k}_{si}" in tab}
            pr = tercile_probs_ref(fc, tb, leads, ym, weighted=day_weighted(tab.get("meta"), leads))
            lab = season_label(ym, leads)
            y0s, y1s = vm[leads[0] - 1][:4], vm[leads[-1] - 1][:4]
            yv = y0s if y0s == y1s else f"{y0s}–{y1s[2:]}"
            key = f"{lab}_{y0s}"
            fig, ax, H, pc = (
                MS.open_map(kind="atm") if glob else MS.open_map(kind="atm", extent=[-170, -30, -60, 75], central=-90)
            )
            order = np.argsort(lon)
            lon_s = lon[order]
            lat_a = lat[::-1] if lat[0] > lat[-1] else lat

            def prep(a, order=order, lat=lat, glob=glob, lon_s=lon_s):
                a = a[:, order]
                if lat[0] > lat[-1]:
                    a = a[::-1]
                if glob:
                    a, lo = add_cyclic_point(a, coord=lon_s)
                    return a, lo
                return a, lon_s

            normal = 1.0 - pr["above"] - pr["below"]
            sig = tercile_signal(pr, fc.shape[0])
            for arr, other, cols in ((pr["above"], pr["below"], above_c), (pr["below"], pr["above"], below_c)):
                show = np.where(sig & (arr >= 0.40) & (arr >= np.maximum(other, normal)), arr, np.nan)
                a, lo = prep(show)
                ax.pcolormesh(
                    lo,
                    lat_a,
                    a,
                    cmap=ListedColormap(cols),
                    norm=BoundaryNorm(bins, len(cols)),
                    transform=pc,
                    shading="auto",
                    zorder=1,
                )
            MS.features(ax, land_only=(var in ("t2m", "tp")))
            top_lab, bot_lab = sides.get(var, ("Above normal most likely", "Below normal most likely"))
            MS.tercile_legends(fig, above_c, below_c, top_lab, bot_lab, bins)
            MS.heading(
                fig,
                H,
                f"SEAS5 {titles[var]}: most likely tercile · {lab} {yv} · {calendar.month_name[m0]} {y0} issue",
                "Terciles from SEAS5's own 1993–2016 hindcast at each grid point (24 years × 25 members); the 51 members are counted against them, so bias and spread drift are removed first. "
                f"Coloured where a category reaches 40 % {TERC_TEST_NOTE}; near-normal is not drawn."
                + (
                    "  Counted against the hindcast's linear trend extrapolated to the valid year, so the warming trend itself does not tilt the map."
                    if var in ("t2m", "z500")
                    else ""
                ),
                wrap=190,
            )
            out = out_dir / f"seas5_terc_{var}_{key}.webp"
            MS.save(fig, out, dpi=110)
            seasons.append(
                dict(
                    key=key,
                    label=f"{lab} {yv}",
                    leads=list(leads),
                    file=out.name,
                    lead7=max(leads) >= 7,
                    frac_above=float(np.nanmean(sig & (pr["above"] >= 0.40))),
                    frac_below=float(np.nanmean(sig & (pr["below"] >= 0.40))),
                )
            )
        meta[var] = dict(
            label=titles[var],
            extent="global" if glob else "americas",
            seasons=seasons,
            detrended=var in ("t2m", "z500"),
        )
        print(f"  terciles {var}: {len(seasons)} maps ({'global' if glob else 'americas'})", flush=True)
        del fc
    return meta


# ── global single-map viewer: anomaly and change per month / season ─────────
MAP_SPEC = {
    # var: (label, units, anomaly levels, change levels, cmap, contour step anom, contour step chg)
    "t2m": (
        "2 m temperature",
        "°C",
        [-4, -3, -2, -1.5, -1, -0.5, -0.25, 0.25, 0.5, 1, 1.5, 2, 3, 4],
        [-2, -1.5, -1, -0.5, -0.25, 0.25, 0.5, 1, 1.5, 2],
        "RdBu_r",
        None,
        None,
    ),
    "tp": (
        "Precipitation",
        "mm/day",
        [-3, -2, -1.5, -1, -0.5, -0.25, 0.25, 0.5, 1, 1.5, 2, 3],
        [-2, -1.5, -1, -0.5, -0.2, 0.2, 0.5, 1, 1.5, 2],
        "BrBG",
        None,
        None,
    ),
    # heights are standardised by the hindcast's interannual σ (per cell, lead, or season): a warm-climate
    # +60 m everywhere saturates a metre scale
    "z500": (
        "500 hPa height, standardised",
        "σ",
        [-5, -4, -3, -2.5, -2, -1.5, -1, -0.5, -0.25, 0.25, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5],
        [-2, -1.5, -1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1, 1.5, 2],
        "RdBu_r",
        None,
        None,
    ),
    # SST: the site's shared SST ladders (mapstyle) and labelled isolines: 0.2 °C classes to ±3, then 0.5 °C to ±5
    # (change maps: 0.1 °C classes), a white neutral class around zero, on the SST ramp every other SST map uses
    "sst": ("Sea surface temperature", "°C", MS.sst_anom_style(5.0)["levels"], MS.SST_CHANGE_LEVELS, "sst", 0.5, 0.25),
    # 850 hPa zonal wind: positive = westerly anomaly; the trade-wind / jet signal of ENSO
    "u850": (
        "850 hPa zonal wind",
        "m/s",
        [-8, -6, -4, -3, -2, -1, -0.5, 0.5, 1, 2, 3, 4, 6, 8],
        [-4, -3, -2, -1, -0.5, 0.5, 1, 2, 3, 4],
        "PuOr_r",
        None,
        None,
    ),
    # 850 hPa meridional wind (southerly-flow events, e.g. on the US East Coast): green southerly, purple northerly
    "v850": (
        "850 hPa meridional wind",
        "m/s, southerly +",
        [-6, -4, -3, -2, -1, -0.5, 0.5, 1, 2, 3, 4, 6],
        [-3, -2, -1, -0.5, 0.5, 1, 2, 3],
        "PRGn",
        None,
        None,
    ),
    # North America snowfall as cm of snow per month at a 10:1 snow-to-liquid ratio, shown as % of the hindcast
    # normal; change in % points
    "sf": (
        "Snowfall, % of normal",
        "% of normal",
        [0, 25, 50, 75, 90, 110, 125, 150, 200, 300, 400],
        [-100, -50, -25, -10, 10, 25, 50, 100],
        "BrBG",
        None,
        None,
    ),
    # snow depth (snowpack water equivalent, monthly mean) — absolute anomaly in mm and % of the hindcast normal
    "sd": (
        "Snowpack water equivalent anomaly",
        "mm w.e.",
        [-150, -100, -60, -30, -15, -5, 5, 15, 30, 60, 100, 150],
        [-60, -30, -15, -5, 5, 15, 30, 60],
        "BrBG",
        None,
        None,
    ),
    "sdp": (
        "Snowpack, % of normal",
        "% of normal",
        [0, 25, 50, 75, 90, 110, 125, 150, 200, 300, 400],
        [-100, -50, -25, -10, 10, 25, 50, 100],
        "BrBG",
        None,
        None,
    ),
}
MAP_CENTRAL = {
    "sst": -160.0,
    "sf": -110.0,
    "sd": -110.0,
    "sdp": -110.0,
}  # SST cut at 20°E (Africa); everything else at 40°E; snow is a NA box
MAP_CENTRAL_DEFAULT = -140.0
MAP_EXTENT = {
    "sf": [-170, -50, 25, 75],
    "sd": [-170, -50, 25, 75],
    "sdp": [-170, -50, 25, 75],
}  # [W, E, S, N] for regional variables


# (var, kind, GRIB short name, factor to display units) for the global single-map viewer; sf and sd are the
# North America snow kinds (sf's factor depends on the month lengths: _snow_fac). The tercile maps use GLOBAL_TERC.
GLOBAL_VARS = [
    ("t2m", "gl", "t2m", 1.0),
    ("tp", "gl", "tprate", 86400.0 * 1000),
    ("sst", "gl", "sst", 1.0),
    ("z500", "gl_z500", "z", 1.0 / G0),
    ("u850", "gl_u850", "u", 1.0),
    ("v850", "gl_v850", "v", 1.0),
    ("sf", "na_snow", "mtsfr", None),
    ("sd", "na_snowdepth", "sd", 1000.0),
]
GLOBAL_TERC = [
    ("t2m", "gl", "t2m", 1.0),
    ("tp", "gl", "tprate", 86400.0 * 1000),
    ("z500", "gl_z500", "z", 1.0 / G0),
    ("u850", "gl_u850", "u", 1.0),
]
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
        nl = hc.shape[1]  # 6 (CDS monthly) or 7 (lead-7 pseudo-raw)
        meta["season_leads"] = [list(x) for x in seasons_for(nl)]
        meta["leadsets"] = [_ls_key(x) for x in leadsets_for(nl)]
        meta["day_weighted"] = [list(x) for x in seasons_for(nl)]  # seasonal statistics: day-weighted means
        arrays[f"hcm_{var}"] = np.nanmean(hc, axis=0)
        arrays["lat"], arrays["lon"] = lat, lon
        if var in terc_vars:
            hcd = hc * fac
            for si, leads in enumerate(seasons_for(nl)):
                for kk, vv in tercile_bounds(hcd, leads, var in DETRENDED_TERC, month=month).items():
                    arrays[f"terc_{var}_{kk}_{si}"] = vv
            del hcd
        if var == "z500" and hc.shape[0] % 24 == 0:
            # samples are member-major, year fastest (_stack_samples): per-year ensemble means give the
            # interannual σ used to standardise heights
            hc_ym = np.nanmean(hc.reshape(-1, 24, *hc.shape[1:]), axis=0) / G0
            for ls in leadsets_for(nl):
                ym_ = _season_means(hc_ym, ls, hindcast_lead_weights(month, ls, 24))
                ny = ym_.shape[0]
                yr = np.arange(ny) - (ny - 1) / 2
                slope = (yr[:, None, None] * (ym_ - ym_.mean(0))).sum(0) / (yr**2).sum()
                arrays[f"ztr_mean_{_ls_key(ls)}"] = ym_.mean(0)
                arrays[f"ztr_slope_{_ls_key(ls)}"] = slope
                arrays[f"ztr_sd_{_ls_key(ls)}"] = np.nanstd(ym_ - slope[None] * yr[:, None, None], axis=0)
            meta["ztr_ny"] = 24
        del hc
    R.save(R.hc_ref(f"build_{kind}", month), meta=meta, **arrays)


# Units of the hindcast spread tables: the display units, except snowfall, whose display factor depends on the
# month's length (_snow_fac) -- its spread is kept in mm of water equivalent per day.
SPREAD_UNITS = {"sf": 86400.0 * 1000.0}


def _spread_fac(var: str, fac) -> float:
    return SPREAD_UNITS.get(var, fac)


def _global_spread_table(kind: str, month: str) -> None:
    """hc_spread_{kind}_{MM}: what the anomaly and change maps' significance test needs from the raw hindcast, per
    variable of the kind (units: _spread_fac):
      sd_{var}_{lead set}  standard deviation of ONE member's lead-set mean (day-weighted in its own year) over the
                           24 years x 25 members -- the spread the hindcast mean is tested with (welch_terms)
      yd_{var}             [year, lead, lat, lon] float16: every year's ensemble mean minus the 24-year mean -- the
                           year-by-year statistics: the residual scatter about a height trend (the reference of the
                           standardised height maps, trend_term) and the year-to-year spread of the difference between
                           two start months' hindcasts (a change map), which is only known year by year
    Lead sets with no data (leads 1-4 of a lead-7 pseudo-raw hindcast) are left out."""
    arrays, meta = {}, {"leadsets": [], "years": 24, "day_weighted": True}
    for var, k, short, fac in GLOBAL_VARS:
        if k != kind:
            continue
        hc, lat, lon = load_field(hc_path(kind, month), short)
        ns, nl = hc.shape[:2]
        if ns % 24:
            print(f"  spread {var} {month}: {ns} samples are not members x 24 years; skipped", flush=True)
            continue
        hc *= np.float32(_spread_fac(var, fac))
        arrays["lat"], arrays["lon"] = lat, lon
        ym_ = np.nanmean(hc.reshape(ns // 24, 24, *hc.shape[1:]), axis=0)  # member-major, year fastest
        arrays[f"yd_{var}"] = (ym_ - np.nanmean(ym_, axis=0, keepdims=True)).astype(np.float16)
        have = [bool(np.isfinite(hc[:, L]).any()) for L in range(nl)]
        for ls in leadsets_for(nl):
            if not all(have[L - 1] for L in ls):
                continue
            h = _season_means(hc, ls, hindcast_lead_weights(month, ls, ns))  # [sample, lat, lon]
            arrays[f"sd_{var}_{_ls_key(ls)}"] = np.nanstd(h, axis=0, ddof=1).astype(np.float32)
            if _ls_key(ls) not in meta["leadsets"]:
                meta["leadsets"].append(_ls_key(ls))
            del h
        del hc, ym_
    if "lat" in arrays:
        R.save(R.hc_ref(f"spread_{kind}", month), meta=meta, **arrays)


def _global_issue_table(ym: str) -> None:
    """is_build_maps_{ym}: this issue's ensemble means [lead, lat, lon] in display units and, per lead set, the
    member count and the members' standard deviation of the lead-set mean (weighted as the maps are) -- what this
    issue's anomaly maps are tested with and what the next issue's change maps take from it."""
    arrays = {}
    for var, kind, short, fac in GLOBAL_VARS:
        if not fc_path(kind, ym).exists():
            continue
        try:
            fc, lat, lon = load_field(fc_path(kind, ym), short)
        except Exception as e:  # noqa: BLE001
            print(f"  global {var} {ym}: {str(e)[:100]}", flush=True)
            continue
        fc *= _snow_fac(ym, fc.shape[1]) if var == "sf" else np.float32(fac)
        arrays[f"fcm_{var}"] = np.nanmean(fc, axis=0)
        arrays[f"n_{var}"] = np.int32(fc.shape[0])
        for ls in leadsets_for(fc.shape[1]):
            w = lead_weights(ym, ls) if var != "sf" else np.full(len(ls), 1.0 / len(ls))
            arrays[f"fsd_{var}_{_ls_key(ls)}"] = np.nanstd(_season_means(fc, ls, w), axis=0, ddof=1).astype(np.float32)
        arrays[f"lat_{var}"] = lat
        arrays[f"lon_{var}"] = lon
        del fc
    if arrays:
        R.save(R.is_ref("build_maps", ym), **arrays)


def load_global(ym: str, current: bool = True) -> dict:
    """{var: dict(fcm, hcm [lead, lat, lon], lat, lon, ztr | None, fsd {lead set: [lat, lon]}, n, spread | None)} for
    the single-map viewer, from the issue's is_build_maps summary (written first from this issue's GRIBs when
    `current`); the hindcast mean and the height trend from hc_build_{kind}_{MM}, the hindcast spread from
    hc_spread_{kind}_{MM}. A summary from before the member spread was kept has no fsd: its maps cannot be tested."""
    out = {}
    if current:
        _global_issue_table(ym)
    summ = R.load(R.is_ref("build_maps", ym))
    if summ is None:
        print(f"  global maps: no summary is_build_maps_{ym} — skipped", flush=True)
        return out
    tabs, spreads = {}, {}
    for var, kind, _short, fac in GLOBAL_VARS:
        if kind not in tabs:
            tabs[kind] = R.load(R.hc_ref(f"build_{kind}", ym[4:]))
            spreads[kind] = R.load(R.hc_ref(f"spread_{kind}", ym[4:]))
        tab = tabs[kind]
        if tab is None or f"hcm_{var}" not in tab:
            print(f"  global {var} {ym}: table hc_build_{kind}_{ym[4:]} missing — skipped", flush=True)
            continue
        if f"fcm_{var}" not in summ:
            continue
        fcm, lat, lon = summ[f"fcm_{var}"], summ[f"lat_{var}"], summ[f"lon_{var}"]
        sp = spreads[kind] if spreads[kind] is not None and f"yd_{var}" in spreads[kind] else None
        if sp is None:
            print(f"  global {var} {ym}: no hindcast spread (hc_spread_{kind}_{ym[4:]}) — maps untestable", flush=True)
        pre = f"fsd_{var}_"
        fsd = {k[len(pre) :]: v for k, v in summ.items() if k.startswith(pre)}
        nh = tab[f"hcm_{var}"].shape[0]
        hcm = tab[f"hcm_{var}"] * (_snow_fac(ym, nh)[0] if var == "sf" else fac)
        ztr = None
        if var == "z500" and "ztr_ny" in tab["meta"]:
            ztr = {k: tab[k] for k in tab if k.startswith("ztr_")}
            ztr["meta"] = tab["meta"]  # which lead sets are day-weighted (day_weighted)
        out[var] = dict(
            fcm=fcm, hcm=hcm, lat=lat, lon=lon, ztr=ztr, fsd=fsd, n=int(summ.get(f"n_{var}", 0)), spread=sp, sv=var
        )
        if var == "sd":
            out["sdp"] = dict(out["sd"], ztr=None)  # the same field as % of normal: sd's spread (sv)
    return out


def _land(lat, lon, sst) -> np.ndarray | None:
    """Land cells of a (lat, lon) grid: where the hindcast SST of the global grid (sst = (field, lat, lon)) has no
    data -- the t2m, precipitation and snow maps show land only, so their test counts land cells only."""
    if sst is None:
        return None
    s, slat, slon = sst
    i = np.abs(np.asarray(slat)[None, :] - np.asarray(lat)[:, None]).argmin(1)
    j = np.abs(((np.asarray(slon)[None, :] - np.asarray(lon)[:, None] + 180.0) % 360.0) - 180.0).argmin(1)
    return ~np.isfinite(s[np.ix_(i, j)])


def trend_term(g: np.ndarray, x0: float) -> tuple:
    """shift_signal term of a straight line fitted to the yearly values g [year, lat, lon] and read x0 years from
    their mean year: the residual variance (n - 2 degrees of freedom) through trend_read_var."""
    n = g.shape[0]
    yr = np.arange(n) - (n - 1) / 2.0
    gm = g.mean(0)
    slope = (yr[:, None, None] * (g - gm)).sum(0) / (yr**2).sum()
    s2 = ((g - gm - slope * yr[:, None, None]) ** 2).sum(0) / (n - 2)
    return trend_read_var(s2, x0, n), n - 2


def _year_devs(e: dict, month: str, leads, conv=None) -> np.ndarray | None:
    """[year, lat, lon]: each hindcast year's ensemble-mean departure over a lead set, in display units -- day-weighted
    in that year, or for snowfall (conv = the issue's days per month) the equal-weight mean of the monthly totals,
    as the maps take it. None when the spread table does not reach the lead set."""
    sp = e["spread"]
    yd = sp[f"yd_{e['sv']}"] if sp is not None else None
    if yd is None or max(leads) > yd.shape[1]:
        return None
    yd = yd.astype(np.float32)
    if conv is not None:
        return _season_means(yd, leads, np.asarray(conv, float) / len(leads))
    return _season_means(yd, leads, hindcast_lead_weights(month, leads, yd.shape[0], yd.shape[0]))


def _hindcast_sd(e: dict, leads, conv=None):
    """The hindcast's single-member spread of a lead-set mean in display units, or None (no spread table / lead set).
    conv: snowfall's days per month (the table holds mm w.e. per day)."""
    sp = e["spread"]
    k = f"sd_{e['sv']}_{_ls_key(leads)}"
    if sp is None or k not in sp:
        return None
    return sp[k] * (float(np.mean(conv)) if conv is not None else 1.0)


def _period_sets(ym: str, prev: str | None):
    """[(key, label, leads_now, leads_prev|None)] — six months then the three seasons; leads_prev is
    None where the previous issue has no counterpart."""
    import calendar

    vm_now = valid_months(ym)
    vm_prev = valid_months(prev) if prev else []
    out = []
    for L, v in enumerate(vm_now, start=1):
        lp = (vm_prev.index(v) + 1,) if v in vm_prev else None
        out.append((v.replace("-", "_"), f"{calendar.month_abbr[int(v[5:])]} {v[:4]}", (L,), lp))
    for leads in seasons_for(len(vm_now)):
        months = [vm_now[L - 1] for L in leads]
        lp = tuple(vm_prev.index(m) + 1 for m in months) if all(m in vm_prev for m in months) else None
        y0, y1 = months[0][:4], months[-1][:4]
        out.append(
            (
                f"{season_label(ym, leads)}_{y0}",
                f"{season_label(ym, leads)} {y0 if y0 == y1 else y0 + '–' + y1[2:]}",
                leads,
                lp,
            )
        )
    return out


def _global_map(field, lat, lon, var, levels, cmap, cstep, title, sub, cb_label, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    pc = ccrs.PlateCarree()
    proj = ccrs.PlateCarree(central_longitude=MAP_CENTRAL.get(var, MAP_CENTRAL_DEFAULT))
    ext = MAP_EXTENT.get(var)
    lat0, lat1 = (ext[2], ext[3]) if ext else ((-70, 70) if var == "sst" else (-60, 85))
    # Both CRSs are PlateCarree, so put the data in the map's own frame (longitude relative to the central meridian, exact
    # +-180 edge columns) and draw with transform=proj: cartopy then bypasses project_geometry, which in cartopy 0.25 +
    # shapely 2.1 fails at draw time on some masked contour sets ("'GeometryCollection' object is not subscriptable").
    c0 = MAP_CENTRAL.get(var, MAP_CENTRAL_DEFAULT)
    lon_r = (np.asarray(lon, float) - c0 + 180.0) % 360.0 - 180.0
    order = np.argsort(lon_r)
    lon_s = lon_r[order]
    d = field[:, order]
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        d = d[::-1]
    gap = lon_s[0] + 360.0 - lon_s[-1]
    w = (180.0 - lon_s[-1]) / gap if gap > 0 else 0.0
    edge = d[:, -1] * (1 - w) + d[:, 0] * w if 0 < w < 1 else (d[:, 0] if w >= 1 else d[:, -1])
    if lon_s[0] > -180.0 + 1e-9:
        d = np.concatenate([edge[:, None], d], axis=1)
        lon_s = np.concatenate([[-180.0], lon_s])
    if lon_s[-1] < 180.0 - 1e-9:
        d = np.concatenate([d, d[:, :1]], axis=1)
        lon_s = np.concatenate([lon_s, [180.0]])
    d_c, lon_c = d, lon_s
    lon_span = (ext[1] - ext[0]) if ext else 360.0
    W = 14.0
    map_h = W * (lat1 - lat0) / lon_span
    top, bot = 0.95, 0.85
    H = map_h + top + bot
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0.03, bot / H, 0.94, map_h / H], projection=proj)
    if ext:
        ax.set_extent([ext[0], ext[1], lat0, lat1], crs=pc)
    else:
        ax.set_extent([-180, 180, lat0, lat1], crs=proj)
    sst_style = None
    if var == "sst":
        # the site's shared SST palette (scripts/sst/mapstyle.py): anomalies on its ladder (white +-0.2 band, 0.2 degC
        # classes to +-3, the same colour for the same value as the OISST loops); the change map keeps its finer
        # 0.1 degC ladder but on the same ramp. The axes background stays white: a blank ocean cell is one with no
        # significant shift (render_global_maps), and land keeps the SST land colour.
        sst_style = MS.sst_style_for(levels)
        ax.add_feature(cfeature.LAND, facecolor=MS.SST_LAND, zorder=0)
        norm, cmap_obj = sst_style["norm"], sst_style["cmap"]
    else:
        ax.add_feature(cfeature.LAND, facecolor="#f1f0eb", zorder=0)
        norm, cmap_obj = BoundaryNorm(levels, len(levels) - 1), plt.get_cmap(cmap, len(levels) - 1)
    m = ax.contourf(
        lon_c,
        lat,
        np.ma.masked_invalid(d_c),
        levels=levels,
        cmap=cmap_obj,
        norm=norm,
        extend="both",
        transform=proj,
        zorder=1,
    )
    if cstep:
        cl = [x for x in np.arange(-6, 6.01, cstep) if abs(x) > 1e-9]
        cs = ax.contour(
            lon_c, lat, np.ma.masked_invalid(d_c), levels=cl, colors="#333", linewidths=0.35, transform=proj, zorder=2
        )
        ax.clabel(cs, fontsize=5.5, fmt=lambda v: f"{v:+.2g}", inline=True, inline_spacing=2)
    if var in ("t2m", "tp", "sf", "sd", "sdp"):
        ax.add_feature(cfeature.OCEAN, facecolor="#ffffff", zorder=2)
        ax.add_feature(cfeature.LAKES, facecolor="#ffffff", zorder=2)
    ax.coastlines(resolution="50m", linewidth=0.45, color="#222", zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.25, edgecolor="#666", zorder=3)
    if var != "sst":
        ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.15, edgecolor="#6f6b64", zorder=3)
    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.3,
        color="#6f6b64",
        alpha=0.5,
        xlocs=range(-180, 181, 30),
        ylocs=range(-60, 91, 30),
        zorder=4,
    )
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 7, "color": "#555"}
    fig.text(0.03, 1 - 0.14 / H, title, fontsize=13.5, fontweight="bold", va="top")
    fig.text(0.03, 1 - 0.50 / H, "\n".join(textwrap.wrap(sub, 215)), fontsize=8.6, color="#444", va="top")
    cax = fig.add_axes([0.25, 0.42 / H, 0.50, 0.14 / H])
    cb = fig.colorbar(m, cax=cax, orientation="horizontal", extend="both", spacing="uniform")
    if sst_style is not None:
        MS.sst_colorbar(cb, sst_style, cb_label, size=8.5, tick_size=7)
        # flat contourf classes: lossless on the frame's own most frequent colours (sharp class edges and labels)
        MS.save_webp_palette(fig, out, dpi=125)
        return
    cb.set_label(cb_label, fontsize=8.5)
    cb.ax.tick_params(labelsize=7)
    if len(levels) > 16:  # fine steps: label whole degrees only
        cb.set_ticks([x for x in levels if abs(x - round(x)) < 1e-6])
    fig.savefig(out, dpi=125, pil_kwargs={"quality": 86, "method": 6})
    plt.close(fig)


LAND_VARS = ("t2m", "tp", "sf", "sd", "sdp")  # drawn over land only (_global_map paints the oceans white)


def _test_sentence(frac: float, change: bool) -> str:
    what = (
        "against both issues' member spreads and the year-to-year spread of the difference between the two start "
        "months' hindcast means"
        if change
        else "of the members against the hindcast's spread over its 24 years"
    )
    return f" Shaded only where the {'change' if change else 'shift'} is significant (Welch t-test {what}, false-discovery rate 0.10): {frac:.0%} of the map."


def render_global_maps(ym: str, prev: str | None, out_dir: Path, only=None) -> dict:
    """One image per (variable, period, kind): the ensemble-mean anomaly of this issue against its own hindcast, and
    the change against the previous issue (each anomalised against its own start-month hindcast), each shaded only
    where it is significant (shift_signal, FDR over the cells the map shows):
      anomaly  the 51 members' spread and the hindcast mean's, one member's spread over 24 independent years
               (welch_terms, conservative); heights against the extrapolated trend, whose error at the valid year is
               the fitted line's own, from its 24 yearly means (trend_term);
      change   both issues' member spreads plus the year-to-year spread of the difference between the two start
               months' yearly hindcast means (hc_spread yd). The two hindcasts share their 24 years, so their
               difference is far better known than either mean: counted as independent, the hindcast terms come to
               4-11 times the forecast terms and almost no change could pass; counted year by year they are 0.1-1.2
               times (2 m temperature, August vs September starts).
    A map whose test inputs are missing (no hc_spread table, a summary without member spread) is not drawn.
    Plate carrée, most of the world. Files: seas5_map_{var}_{anom|chg}_{period}.webp."""
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
    meta = {
        "periods": [
            dict(
                key=k,
                label=l,
                kind="month" if len(ln) == 1 else "season",
                leads=list(ln),
                prev=(list(lp) if lp else None),
            )
            for k, l, ln, lp in periods
        ],
        "vars": {},
        "previous": prev,
        "previous_last": (valid_months(prev)[-1] if prev else None),
        "test": SHIFT_TEST_NOTE,
    }
    sst_ref = (now["sst"]["hcm"][0], now["sst"]["lat"], now["sst"]["lon"]) if "sst" in now else None
    chg_wanted = chg_tested = 0
    for var, e in now.items():
        if only and var not in only:
            continue
        fcm, hcm, lat, lon, ztr = e["fcm"], e["hcm"], e["lat"], e["lon"], e["ztr"]
        label, units, lv_a, lv_c, cmap, cs_a, cs_c = MAP_SPEC[var]
        entry = {"label": label, "units": units, "anom": {}, "chg": {}, "standardised": ztr is not None}
        pb = before.get(var)
        nl = min(fcm.shape[0], hcm.shape[0])  # 7 when the forecast supplement and the lead-7 table exist
        base = np.ones(fcm.shape[1:], bool)
        if var in LAND_VARS:
            land = _land(lat, lon, sst_ref)
            if land is None:
                print(f"  global maps {var}: no SST table for a land mask; tested over every cell", flush=True)
            else:
                base = land
        days = _snow_fac(ym, fcm.shape[0])[0, :, 0, 0] / SPREAD_UNITS["sf"] if var == "sf" else None
        for key, plabel, ln, lp in periods:
            if max(ln) > nl or (ztr is not None and f"ztr_mean_{_ls_key(ln)}" not in ztr):
                continue
            idx = [L - 1 for L in ln]
            # seasons: each month weighted by its length in the valid year (a leap February counts 29 days); snowfall
            # is already a monthly total (cm per month), so its seasonal ratio is of plain sums
            w = lead_weights(ym, ln) if var != "sf" else np.full(len(ln), 1.0 / len(ln))
            conv = days[idx] if days is not None else None

            def mw(x, ii, ww):
                return np.tensordot(ww, x[ii], axes=1)

            fm, hm = mw(fcm, idx, w), mw(hcm, idx, w)
            a = num = fm - hm  # num: the shift the test is about, in display units
            if var in ("sf", "sdp"):  # % of the hindcast normal, blank where it is tiny
                floor = 1.0 if var == "sf" else 10.0  # 1 cm/month of snowfall, 10 mm w.e. of snowpack
                a = np.where(hm >= floor, 100.0 * fm / np.maximum(hm, 1e-6), np.nan)
            if ztr is not None:
                # heights carry the warming trend (+40–60 m over the tropics against a 1993–2016 mean, which is
                # many σ where the year-to-year spread is small): the reference is the hindcast's linear trend
                # evaluated at the valid year, the σ its residual spread — the same construction as the normals.
                # Trend mean / slope / σ per lead set come from the table (_global_hc_table).
                k = _ls_key(ln)
                ny = 24
                vyear = int(valid_months(ym)[idx[len(idx) // 2]][:4])
                target = (vyear - 1993) - (ny - 1) / 2
                ref = ztr[f"ztr_mean_{k}"] + ztr[f"ztr_slope_{k}"] * target
                sd = ztr[f"ztr_sd_{k}"]
                wz = (
                    w if day_weighted(ztr.get("meta"), ln) else np.full(len(ln), 1.0 / len(ln))
                )  # as the table was built
                num = mw(fcm, idx, wz) - ref
                a = num / np.where(sd > 0, sd, np.nan)
            ls = _ls_key(ln)
            sh = _hindcast_sd(e, ln, conv)
            yn = _year_devs(e, ym[4:], ln, conv)
            if sh is None or yn is None or ls not in e["fsd"] or e["n"] < 2:
                print(f"  global maps {var} {key}: no member or hindcast spread for the test — not drawn", flush=True)
                continue
            # the flat hindcast mean: one member's spread over 24 years (conservative); the trend read at the valid
            # year: the fitted line's own error from its 24 yearly means (with one member's spread nothing would pass)
            hterm = trend_term(yn, target) if ztr is not None else (sh**2 / N_EFF_HC, N_EFF_HC - 1)
            dom = base & np.isfinite(a)
            sig = shift_signal(num, [(e["fsd"][ls] ** 2 / e["n"], e["n"] - 1), hterm], dom)
            frac = float(sig[dom].mean()) if dom.any() else 0.0
            out = out_dir / f"seas5_map_{var}_anom_{key}.webp"
            _global_map(
                np.where(sig, a, np.nan),
                lat,
                lon,
                var,
                lv_a,
                cmap,
                cs_a,
                f"SEAS5 {label}{'' if var in ('sf', 'sdp', 'sd') else ' anomaly'} · {plabel} · {issue_lbl}",
                (
                    "Ensemble-mean snowfall (10:1 snow-to-liquid ratio) as % of the 1993–2016 start-month hindcast mean for the same lead (25 members × 24 years), 1° grid; blank where the normal is under 1 cm a month."
                    if var == "sf"
                    else "Ensemble-mean monthly snowpack (snow depth in water equivalent) as % of the 1993–2016 start-month hindcast mean for the same lead (25 members × 24 years), 1° grid; blank where the normal is under 10 mm."
                    if var == "sdp"
                    else "Ensemble mean of 51 members minus the 1993–2016 start-month hindcast mean (25 members × 24 years): monthly-mean snowpack in mm of water equivalent, 1° grid."
                    if var == "sd"
                    else "Ensemble mean of 51 members minus the 1993–2016 start-month hindcast mean (25 members × 24 years), 1° grid."
                    + (
                        " Heights: departure from the hindcast's linear trend at the valid year, in σ of the residual year-to-year spread; the trend's error at the valid year is part of the test."
                        if ztr is not None
                        else ""
                    )
                )
                + _test_sentence(frac, False),
                ("% of the hindcast normal" if var in ("sf", "sdp") else f"anomaly ({units})"),
                out,
            )
            entry["anom"][key] = dict(file=out.name, mean=float(np.nanmean(a)), sig=round(frac, 3))
            if pb is None or lp is None or max(lp) > pb["fcm"].shape[0] or (ztr is not None and pb["ztr"] is None):
                continue
            chg_wanted += 1
            fcmp, hcp, ztrp = pb["fcm"], pb["hcm"], pb["ztr"]
            ip = [L - 1 for L in lp]  # fcmp: previous ensemble mean; the same valid months: the same weights
            fmp, hmp = mw(fcmp, ip, w), mw(hcp, ip, w)
            ap = nump = fmp - hmp
            if var in ("sf", "sdp"):
                floorp = 1.0 if var == "sf" else 10.0
                ap = np.where(hmp >= floorp, 100.0 * fmp / np.maximum(hmp, 1e-6), np.nan)
            if ztr is not None:  # previous issue: its own trend reference, this σ
                kp = _ls_key(lp)
                nyp = 24
                refp = ztrp[f"ztr_mean_{kp}"] + ztrp[f"ztr_slope_{kp}"] * ((vyear - 1993) - (nyp - 1) / 2)
                wzp = w if day_weighted(ztrp.get("meta"), lp) else np.full(len(lp), 1.0 / len(lp))
                nump = mw(fcmp, ip, wzp) - refp
                ap = nump / np.where(sd > 0, sd, np.nan)
            c = a - ap
            lsp = _ls_key(lp)
            yp = _year_devs(pb, prev[4:], lp, conv)
            if yp is None or lsp not in pb["fsd"] or pb["n"] < 2:
                continue
            chg_tested += 1
            vn, vp = e["fsd"][ls] ** 2 / e["n"], pb["fsd"][lsp] ** 2 / pb["n"]
            with np.errstate(divide="ignore", invalid="ignore"):
                if var in (
                    "sf",
                    "sdp",
                ):  # a change of % of normal: linearised in both forecast and normal (delta method)
                    diff, g = c, 100.0 * (fmp / hmp**2 * yp - fm / hm**2 * yn)
                    vn, vp = vn * (100.0 / hm) ** 2, vp * (100.0 / hmp) ** 2
                else:
                    diff, g = num - nump, yn - yp
            # the two trend references, read at the same valid year, or the two flat hindcast means
            hterm = (
                trend_term(g, target) if ztr is not None else (np.var(g, axis=0, ddof=1) / g.shape[0], g.shape[0] - 1)
            )
            dom = base & np.isfinite(c)
            sig = shift_signal(diff, [(vn, e["n"] - 1), (vp, pb["n"] - 1), hterm], dom)
            frac = float(sig[dom].mean()) if dom.any() else 0.0
            out = out_dir / f"seas5_map_{var}_chg_{key}.webp"
            _global_map(
                np.where(sig, c, np.nan),
                lat,
                lon,
                var,
                lv_c,
                cmap,
                cs_c,
                f"SEAS5 {label}: change since the {prev_lbl} · {plabel} · {issue_lbl}",
                "Each issue's ensemble mean anomalised against its own start-month hindcast, so this is the shift in the forecast, not drift. Periods the earlier issue does not cover are not drawn."
                + _test_sentence(frac, True),
                (
                    f"change in % of normal (points), {issue_lbl} minus {prev_lbl}"
                    if var in ("sf", "sdp")
                    else f"change in ensemble-mean anomaly ({units}), {issue_lbl} minus {prev_lbl}"
                ),
                out,
            )
            entry["chg"][key] = dict(file=out.name, mean=float(np.nanmean(c)), sig=round(frac, 3))
        if entry["anom"]:
            meta["vars"][var] = entry
        print(f"  global maps {var}: {len(entry['anom'])} anomaly, {len(entry['chg'])} change", flush=True)
    if chg_wanted and not chg_tested:
        meta["chg_untested"] = (
            f"No change maps for this issue: the {prev_lbl}'s stored forecast or its start month's hindcast carries "
            "no spread, so a change cannot be tested."
        )
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
    fa = np.empty_like(f)
    hr = np.empty_like(h)
    for L in range(h.shape[1]):
        y = h[:, L]
        ok = np.isfinite(y)
        b = (x[ok] * (y[ok] - y[ok].mean())).sum() / (x[ok] ** 2).sum()
        a0 = y[ok].mean()
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
    for hemi, _kind, _box in POLAR:
        for lev in (10, 50, 100):
            key = f"{hemi}_z{lev}"
            if key not in fcs or key not in hct:
                continue
            f, h = fcs[key], hct[key]
            n = min(f.shape[1], h.shape[1])
            f, h = f[:, :n], h[:, :n]
            # DETRENDED: geopotential carries the warming trend (thickness), so an anomaly vs the
            # 1993–2016 mean reads high by construction in 2026. Fit the hindcast's linear trend
            # per lead across its 24 years and take the anomaly against that line extrapolated to
            # the valid year; the hindcast spread is the spread of the residuals.
            a, hr = detrend_pair(f, h, ym)
            e = dict(
                **_summ(a),
                clim_sd=hr.std(0).round(1).tolist(),
                clim_mean=h.mean(0).round(1).tolist(),
                units="m",
                valid=valid_months(ym)[:n],
                detrended=True,
            )
            if members:
                e["members"] = a.round(1).tolist()
            out[key] = e
    return out


# ── observed context ─────────────────────────────────────────────────────────
def observed_indices() -> dict:
    """Last 18 observed months of the Niño indices already on the site (nino_history.json): CPC's ERSSTv5 regional
    table and RONI, continued by the site's OISST v2.1 estimate after the two boundaries carried along, so the page can
    draw the estimate as an estimate."""
    p = ASSETS / "data" / "nino_history.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    months = d["months"][-18:]
    out = {
        "months": months,
        "provisional_after": d.get("provisional_after"),
        "regional_provisional_after": d.get("regional_provisional_after"),
    }
    for k in ("nino12", "nino3", "nino34", "nino4", "roni"):
        ser = d["series"].get(k)
        if isinstance(ser, dict):  # {"abs": [...], "anom": [...]}
            ser = ser.get("anom", ser.get("abs"))
        if ser:
            out[k] = ser[-18:]
    return out


# What lead month 7 is made of (shown on the page under every product that reaches it)
LEAD7_NOTE = (
    "Lead month 7 is not in the CDS monthly product (leads 1–6). Forecast: the same SEAS5 run's sub-daily fields "
    "(seasonal-original), averaged over the month the way the monthly product is (instantaneous fields: every step "
    "from 00 UTC on the 1st to the last step before the next month; accumulations: the month's total over its "
    "seconds). Hindcast reference: the same 1993–2016 hindcast, 25 members, from the daily copy of the C3S archive "
    "on AWS (instantaneous fields, interpolated to the CDS grid) and the CDS sub-daily hindcast (accumulations, "
    "50 hPa height, 30/50 hPa wind). Skill is lower at lead 7; it goes through the same calibration and trend handling as "
    "leads 1–6, and the same tests where a product has one (the teleconnection skill test, the tercile maps' signal "
    "test, the anomaly and change maps' Welch test)."
)


# ── build ────────────────────────────────────────────────────────────────────
def build(ym: str, n_prev: int = 3) -> None:
    t0 = time.time()
    ASSETS.mkdir(parents=True, exist_ok=True)
    (ASSETS / "data").mkdir(parents=True, exist_ok=True)
    issues = {}
    for iss in [ym, *previous_issues(ym, n_prev)]:
        print(f"indices {iss} …", flush=True)
        r = sst_indices(iss)
        if r is None:
            print(f"  {iss}: no SST forecast + hindcast table, and no is_build_sst summary — skipped", flush=True)
            continue
        nl_iss = np.asarray(next(iter(r.values()))["members"]).shape[1]
        entry = {"valid": valid_months(iss)[:nl_iss], "leads": nl_iss}
        for k, v in r.items():
            e = _summ(v["members"])
            e["clim_sd"] = np.asarray(v["clim_sd"]).round(3).tolist()
            if iss == ym:
                e["members"] = np.asarray(v["members"]).round(3).tolist()
            entry[k] = e
        issues[iss] = entry
    if ym not in issues:
        raise SystemExit(f"no SEAS5 {ym} SST data — run fetch first")

    terc = render_terciles(ym, {}, ASSETS)

    prev = previous_issues(ym, 1)[0]
    changes = {}  # change maps come from the single-map viewer below
    maps = render_global_maps(ym, prev, ASSETS)
    if not maps:
        print(f"  global maps skipped: gl fields for {ym} not on disk", flush=True)

    polar = polar_caps(ym)
    polar_prev = polar_caps(prev, members=False)

    doc = {
        "generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        "system": "ECMWF SEAS5 (C3S originating_centre ecmwf, system 51)",
        "issue": ym,
        "issues": sorted(issues),
        "members": 51,
        "hindcast": "1993–2016, 25 members",
        "leads": nleads(ym),
        "lead7": LEAD7_NOTE,
        "index_meta": [dict(key=k, label=l, units=u, note=n) for k, l, u, n in INDEX_META],
        "enso_keys": ENSO_KEYS,
        "other_keys": OTHER_KEYS,
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
    # each kind's map statistics and the spread its maps are tested with, one after the other, so a raw kind is
    # deleted (derive_hindcast --discard-raw) before the next is fetched
    **{
        u: spec
        for k in ("gl", "gl_z500", "gl_u850", "gl_v850", "na_snow", "na_snowdepth")
        for u, spec in (
            (f"build_{k}", dict(raw=[f"m:{k}"], fn=(lambda month, k=k: _global_hc_table(k, month)))),
            (f"spread_{k}", dict(raw=[f"m:{k}"], fn=(lambda month, k=k: _global_spread_table(k, month)))),
        )
    },
    "build_polar": dict(raw=["m:polar_n", "m:polar_s"], fn=_polar_hc_table),
}
ISSUE_UNITS = {
    # sst_indices / polar_caps write their summaries as a side effect when the forecast GRIBs are on disk
    "build_sst": lambda ym: sst_indices(ym) if fc_path("sst", ym).exists() else None,
    "build_maps": _global_issue_table,
    "build_polar": lambda ym: polar_caps(ym) if fc_path("polar_n", ym).exists() else None,
}
