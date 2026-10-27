#!/usr/bin/env python3
"""What a negative South Atlantic subtropical dipole (SASD, SW - NE pole) does to South America, in CMIP6 and ERA5.

Per member (cmip6_satl_extract.py files): monthly anomalies about a centred 21-year running mean of each calendar month
(forced trends never enter), day-weighted seasonal means, then per season the least-squares fit
    field = a + b * SASD + c * Nino34        (both indices standardised within the member)
so b is the response to one standard deviation of the dipole with ENSO held fixed. Three cases: SON and DJF concurrent,
and DJF fields on the SON indices (what follows a dipole seen in spring). The sign is flipped to the NEGATIVE phase.
Members are averaged within a model, models with equal weight. A grid cell is robust when the model means pass a
one-sample t-test at a Benjamini-Hochberg false discovery rate of 10 % (over the cells of that field and case) AND at
least 80 % of the models agree in sign. An observed file in the same format (--obs: indices from ERSST, fields from
a reanalysis, any subset of the fields) is fitted the same way, as the check on the models.

Which way the causality runs (--causal, on coupled runs or the observed file):
  lead-lag   monthly correlation of the South Atlantic High and the regional series with the dipole from 3 months
             before to 3 months after (ENSO removed from both: Nino-3.4 at 0-2 months' lead). Atmosphere driving the
             ocean shows as a peak with the atmosphere LEADING; the Frankignoul estimate of the response to the SST,
             lambda = cov(A(t), T(t-k)) / cov(T(t), T(t-k)) for k = 1, 2, uses only lags at which the atmosphere's own
             noise has decorrelated, so it is what the SST anomaly causes.
  persistent DJF fields on SON SASD holding SON Nino-3.4, the SON South Atlantic High AND the field's own SON value
             fixed: what the ocean carries into summer beyond the atmosphere's own memory.
The atmosphere-only test is the same response fit on `amip` extracts (observed SST prescribed): --cmip6 on those, with
--tag amip.

    python scripts/sst/cmip6_satl_impacts.py --cmip6 extracts/ --out out/ [--tag amip] [--causal]
    python scripts/sst/cmip6_satl_impacts.py --obs obs.npz --out out/ [--causal]
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

CASES = {  # name: (index months, field months, field year offset from the index year)
    "SON": ((9, 10, 11), (9, 10, 11), 0),
    "DJF": ((12, 1, 2), (12, 1, 2), 0),
    "SON to DJF": ((9, 10, 11), (12, 1, 2), 1),
}
LAND = ("pr", "tas")
REGIONS = {  # name: ((lat0, lat1, lon0, lon1) in degrees east 0-360, the fields reported there)
    "La Plata (S Brazil, Uruguay, NE Argentina)": ((-35, -25, 300, 308), LAND),
    "SE Brazil (Sao Paulo, Rio, Minas)": ((-25, -18, 310, 318), LAND),
    "Centre-West / Cerrado": ((-20, -10, 305, 315), LAND),
    "NE Brazil": ((-12, -3, 315, 325), LAND),
    "S Amazon": ((-15, -5, 295, 310), LAND),
    "Patagonia": ((-50, -40, 285, 295), LAND),
    "NE Brazil wind coast (Ceara, Rio Grande do Norte)": ((-7, -2, 320, 325), ("sfcWind", "tas")),
    "NE Brazil wind interior (Bahia, Piaui, Pernambuco)": ((-15, -7, 315, 322), ("sfcWind", "tas")),
    "South Atlantic High": ((-35, -25, 340, 360), ("psl",)),
}
FIELDS = ("pr", "tas", "psl", "sfcWind")
UNITS = {"pr": "mm/day", "tas": "degC", "psl": "hPa", "sfcWind": "m/s"}
RELATIVE = ("sfcWind",)  # also reported as % of the seasonal mean (wind power goes roughly as its cube)
MIN_MODELS = 5
FDR_Q, AGREE = 0.10, 0.8


def month_days(years: np.ndarray, months: np.ndarray, calendar: str) -> np.ndarray:
    if calendar == "360_day":
        return np.full(len(months), 30.0)
    base = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], float)[months - 1]
    if calendar in ("noleap", "365_day"):
        return base
    leap = (years % 4 == 0) & ((years % 100 != 0) | (years % 400 == 0))
    if calendar == "julian":
        leap = years % 4 == 0
    return base + ((months == 2) & leap)


def anomalies(x: np.ndarray, half: int = 10) -> np.ndarray:
    """(years*12, ...) -> anomalies about each calendar month's centred (2*half+1)-year running mean."""
    y = x.reshape(-1, 12, *x.shape[1:]).astype("float64")
    n = y.shape[0]
    c = np.concatenate([np.zeros((1, *y.shape[1:])), np.nancumsum(y, 0)])
    lo = np.clip(np.arange(n) - half, 0, n)
    hi = np.clip(np.arange(n) + half + 1, 0, n)
    mean = (c[hi] - c[lo]) / (hi - lo).reshape(-1, *([1] * (y.ndim - 1)))
    return (y - mean).reshape(x.shape)


def seasonal(x: np.ndarray, years: np.ndarray, months: np.ndarray, days: np.ndarray, sel, offset: int):
    """Day-weighted mean of the months `sel` per season year (December belongs to the next year's DJF)."""
    sy = years + ((months == 12) & (12 in sel) & (1 in sel))
    out, ys = [], []
    for yr in np.unique(sy):
        m = np.isin(months, sel) & (sy == yr)
        if m.sum() != len(sel):
            continue
        w = days[m] / days[m].sum()
        out.append(np.tensordot(w, x[m], axes=(0, 0)))
        ys.append(yr - offset)
    return np.array(out), np.array(ys)


def fit(field: np.ndarray, sasd: np.ndarray, n34: np.ndarray) -> np.ndarray:
    """Response per +1 sd of SASD with Nino-3.4 held fixed (both standardised here), per cell."""
    z = lambda v: (v - v.mean()) / v.std()  # noqa: E731
    X = np.column_stack([np.ones(len(sasd)), z(sasd), z(n34)])
    beta, *_ = np.linalg.lstsq(X, field.reshape(len(sasd), -1), rcond=None)
    return beta[1].reshape(field.shape[1:])


def responses(series: dict, fields: dict, years, months, days) -> dict:
    """{case: {field: map}} for one member or the observations; NEGATIVE phase (sign flipped)."""
    out = {}
    for case, (isel, fsel, off) in CASES.items():
        idx = {}
        for k in ("sasd", "n34"):
            v, yi = seasonal(series[k][:, None], years, months, days, isel, 0)
            idx[k] = dict(zip(yi, v[:, 0]))
        res = {}
        for f, x in fields.items():
            v, yf = seasonal(x, years, months, days, fsel, off)
            keep = [i for i, yr in enumerate(yf) if yr in idx["sasd"]]
            yy = yf[keep]
            res[f] = -fit(v[keep], np.array([idx["sasd"][t] for t in yy]), np.array([idx["n34"][t] for t in yy]))
        out[case] = res
    return out


def member(path: str):
    d = np.load(path)
    ym = np.array([s.split("-") for s in d["months"]], int)
    years, months = ym[:, 0], ym[:, 1]
    days = month_days(years, months, str(d["calendar"]))
    series = {"sasd": anomalies(d["sw"] - d["ne"]), "n34": anomalies(d["n34"])}
    fields = {f: anomalies(d[f]) for f in FIELDS if f in d.files}
    r = responses(series, fields, years, months, days)
    # the seasonal mean of the fields reported relative to it, for the case's field months
    r["clim"] = {
        case: {f: seasonal(d[f], years, months, days, fsel, 0)[0].mean(0) for f in RELATIVE if f in d.files}
        for case, (_isel, fsel, _off) in CASES.items()
    }
    return str(d["source_id"]), r, d["lat"], d["lon"], series, years


def bh(p: np.ndarray, q: float) -> np.ndarray:
    """Benjamini-Hochberg: True where p passes at false discovery rate q."""
    flat = np.where(np.isfinite(p), p, 1.0).ravel()
    order = np.argsort(flat)
    thresh = q * np.arange(1, flat.size + 1) / flat.size
    passed = flat[order] <= thresh
    k = passed.nonzero()[0].max() + 1 if passed.any() else 0
    ok = np.zeros(flat.size, bool)
    ok[order[:k]] = True
    return ok.reshape(p.shape)


def region_mean(m: np.ndarray, lat, lon, box) -> float:
    la0, la1, lo0, lo1 = box
    sel = np.ix_((lat >= la0) & (lat <= la1), (lon >= lo0) & (lon <= lo1))
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones(len(lon))[None, :]
    return float(np.nansum(m[sel] * w[sel]) / np.nansum(w[sel] * np.isfinite(m[sel])))


def cmip6(paths: list[str]):
    from scipy import stats

    by_model = defaultdict(list)
    lat = lon = None
    corr = defaultdict(list)
    for p in paths:
        src, r, lat, lon, series, _years = member(p)
        by_model[src].append(r)
        corr[src].append(float(np.corrcoef(series["sasd"], series["n34"])[0, 1]))
    models = sorted(by_model)
    out = {"models": models, "n_members": int(sum(len(v) for v in by_model.values())), "cases": {}}
    maps = {}
    for case in CASES:
        out["cases"][case] = {}
        for f in FIELDS:
            have = [m for m in models if all(f in r[case] for r in by_model[m])]
            if len(have) < MIN_MODELS:
                continue
            mm = np.array([np.mean([r[case][f] for r in by_model[m]], 0) for m in have])  # (model, lat, lon)
            mean = mm.mean(0)
            t = stats.ttest_1samp(mm, 0.0, axis=0)
            agree = np.mean(np.sign(mm) == np.sign(mean), 0)
            robust = bh(t.pvalue, FDR_Q) & (agree >= AGREE)
            maps[f"{case}|{f}|mean"], maps[f"{case}|{f}|robust"] = mean, robust
            regs = {}
            for name, (box, fs) in REGIONS.items():
                if f not in fs:
                    continue
                per = np.array([region_mean(x, lat, lon, box) for x in mm])
                tr = stats.ttest_1samp(per, 0.0)
                regs[name] = {
                    "mean": round(float(per.mean()), 3),
                    "models_same_sign": round(float(np.mean(np.sign(per) == np.sign(per.mean()))), 2),
                    "p": round(float(tr.pvalue), 4),
                }
                if f in RELATIVE:
                    clim = [
                        np.mean([region_mean(r["clim"][case][f], lat, lon, box) for r in by_model[m]]) for m in have
                    ]
                    regs[name]["pct_of_mean"] = round(float(np.mean(100 * per / np.array(clim))), 2)
            out["cases"][case][f] = {
                "units": UNITS[f],
                "n_models": len(have),
                "robust_cells": round(float(robust.mean()), 3),
                "regions": regs,
            }
    out["sasd_n34_corr_monthly"] = {m: round(float(np.mean(v)), 2) for m, v in corr.items()}
    return out, maps, lat, lon


def figure(maps: dict, lat, lon, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    import house

    house.apply()

    lim = {"pr": 0.6, "tas": 0.4, "psl": 0.8, "sfcWind": 0.3}
    cmap = {"pr": "BrBG", "tas": "RdBu_r", "psl": "PuOr_r", "sfcWind": "PiYG"}
    rows = [f for f in FIELDS if all(f"{c}|{f}|mean" in maps for c in CASES)]
    fig, axs = plt.subplots(
        len(rows),
        3,
        figsize=(12.6, 4.15 * len(rows)),
        subplot_kw={"projection": ccrs.PlateCarree()},
        layout="constrained",
        squeeze=False,
    )
    lon180 = np.where(lon > 180, lon - 360, lon)
    for i, f in enumerate(rows):
        for j, case in enumerate(CASES):
            ax = axs[i, j]
            m, rb = maps[f"{case}|{f}|mean"], maps[f"{case}|{f}|robust"]
            pc = ax.pcolormesh(lon180, lat, m, cmap=cmap[f], vmin=-lim[f], vmax=lim[f], shading="nearest")
            ax.contourf(lon180, lat, (~rb).astype(float), levels=[0.5, 1.5], hatches=["..."], colors="none")
            ax.coastlines(lw=0.6)
            ax.set_extent((-90, 0, -60, 15), ccrs.PlateCarree())
            ax.set_title(f"{case}: {f} ({UNITS[f]})", fontsize=10, loc="left")
        fig.colorbar(pc, ax=axs[i, :].tolist(), shrink=0.85, label=f"per -1 sd SASD ({UNITS[f]})")
    fig.suptitle(title, fontsize=12, x=0.02, ha="left")
    fig.text(
        0.02,
        0.01,
        "Dots: not robust (FDR 10 % on the model t-test, or < 80 % of models agree). ENSO held fixed.",
        fontsize=8.5,
        color="#6f6b64",
    )
    fig.savefig(path, dpi=110)
    plt.close(fig)


LAGS = range(-3, 4)  # months; positive = the atmosphere AFTER the dipole
SAH = REGIONS["South Atlantic High"][0]


def deseason_enso(x: np.ndarray, n34: np.ndarray) -> np.ndarray:
    """Monthly anomaly series with Nino-3.4 at 0, 1 and 2 months' lead regressed out."""
    X = np.column_stack([np.ones(len(x)), n34, np.r_[np.nan, n34[:-1]], np.r_[np.nan, np.nan, n34[:-2]]])
    ok = np.isfinite(X).all(1) & np.isfinite(x)
    beta, *_ = np.linalg.lstsq(X[ok], x[ok], rcond=None)
    out = np.full(len(x), np.nan)
    out[ok] = x[ok] - X[ok] @ beta
    return out


def lagcorr(a: np.ndarray, t: np.ndarray, k: int) -> float:
    """corr(a(t + k), t(t)) over the pairs both finite."""
    x, y = (a[k:], t[: len(t) - k]) if k >= 0 else (a[: len(a) + k], t[-k:])
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def frankignoul(a: np.ndarray, t: np.ndarray, k: int) -> float:
    """Response of a to one sd of t, from the lag-k covariances (t leading)."""
    x, y, z = a[k:], t[: len(t) - k], t[k:]
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    return float(np.cov(x[ok], y[ok])[0, 1] / np.cov(z[ok], y[ok])[0, 1] * np.nanstd(t))


def causal_member(path: str) -> dict:
    """{series: {"lag": {k: r}, "lambda": {k: value}, "persistent": coefficient, "plain": coefficient}} for one member."""
    d = np.load(path)
    ym = np.array([s.split("-") for s in d["months"]], int)
    years, months = ym[:, 0], ym[:, 1]
    days = month_days(years, months, str(d["calendar"]))
    lat, lon = d["lat"], d["lon"]
    n34 = anomalies(d["n34"])
    sasd = anomalies(d["sw"] - d["ne"])
    series = {}
    if "psl" in d.files:
        series["South Atlantic High (psl)"] = (anomalies(d["psl"]), SAH)
    for name, (box, fs) in REGIONS.items():
        for f in fs:
            if f != "psl" and f in d.files:
                series[f"{name} ({f})"] = (None, box, f)
    fields = {f: anomalies(d[f]) for f in FIELDS if f in d.files}
    t = deseason_enso(sasd, n34)
    out = {}
    sah = None
    for key, spec in series.items():
        x = spec[0] if spec[0] is not None else fields[spec[2]]
        box = spec[1]
        reg = np.array([region_mean(m, lat, lon, box) for m in x])
        if key.startswith("South Atlantic High"):
            sah = reg
        a = deseason_enso(reg, n34)
        out[key] = {
            "lag": {k: lagcorr(a, t, k) for k in LAGS},
            "lambda": {k: frankignoul(a, t, k) for k in (1, 2)},
            "_reg": reg,
        }
    # the persistence-controlled spring-to-summer fit, per series
    son = lambda v: seasonal(v[:, None], years, months, days, (9, 10, 11), 0)  # noqa: E731
    djf = lambda v: seasonal(v[:, None], years, months, days, (12, 1, 2), 1)  # noqa: E731
    z = lambda v: (v - v.mean()) / v.std()  # noqa: E731
    s_sasd, ys = son(sasd)
    s_n34, _ = son(n34)
    s_sah, _ = son(sah) if sah is not None else (None, None)
    idx = {yr: i for i, yr in enumerate(ys)}
    for o in out.values():
        reg = o.pop("_reg")
        dj, yd = djf(reg)
        sn, _ = son(reg)
        keep = [(i, idx[yr]) for i, yr in enumerate(yd) if yr in idx]
        y = np.array([dj[i, 0] for i, _ in keep])
        cols = [s_sasd[:, 0], s_n34[:, 0]] + ([s_sah[:, 0]] if s_sah is not None else [])
        base = np.column_stack([np.ones(len(keep))] + [z(np.array([c[j] for _, j in keep])) for c in cols])
        own = z(np.array([sn[j, 0] for _, j in keep]))
        plain, *_ = np.linalg.lstsq(base[:, :3], y, rcond=None)
        full, *_ = np.linalg.lstsq(np.column_stack([base, own]), y, rcond=None)
        o["plain"], o["persistent"] = -float(plain[1]), -float(full[1])  # negative phase
    return out


def causal(paths: list[str]) -> dict:
    from scipy import stats

    by_model = defaultdict(list)
    for p in paths:
        by_model[str(np.load(p)["source_id"])].append(causal_member(p))
    keys = sorted({k for runs in by_model.values() for r in runs for k in r})
    out = {"models": len(by_model), "series": {}}
    for key in keys:
        have = [m for m, runs in by_model.items() if all(key in r for r in runs)]
        if len(have) < MIN_MODELS:
            continue

        def agg(get, key=key, have=have):
            v = np.array([np.mean([get(r[key]) for r in by_model[m]]) for m in have])
            return {
                "mean": round(float(v.mean()), 3),
                "same_sign": round(float(np.mean(np.sign(v) == np.sign(v.mean()))), 2),
                "p": round(float(stats.ttest_1samp(v, 0.0).pvalue), 4),
            }

        out["series"][key] = {
            "n_models": len(have),
            "lag_corr": {str(k): agg(lambda o, k=k: o["lag"][k]) for k in LAGS},
            "frankignoul_per_sd": {str(k): agg(lambda o, k=k: o["lambda"][k]) for k in (1, 2)},
            "son_to_djf_plain": agg(lambda o: o["plain"]),
            "son_to_djf_persistence_held": agg(lambda o: o["persistent"]),
        }
    return out


def causal_obs(path: str) -> dict:
    r = causal_member(path)
    return {
        k: {
            "lag_corr": {str(j): round(v, 3) for j, v in o["lag"].items()},
            "frankignoul_per_sd": {str(j): round(v, 3) for j, v in o["lambda"].items()},
            "son_to_djf_plain": round(o["plain"], 3),
            "son_to_djf_persistence_held": round(o["persistent"], 3),
        }
        for k, o in r.items()
    }


def observed(path: str) -> dict:
    """{case: {field: {region: value}}} for the observed file (one sample: no robustness test)."""
    _src, r, lat, lon, _series, _years = member(path)
    return {
        case: {
            f: {n: round(region_mean(m, lat, lon, b), 3) for n, (b, fs) in REGIONS.items() if f in fs}
            for f, m in fields.items()
        }
        for case, fields in r.items()
        if case in CASES
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cmip6", help="directory of cmip6_satl_extract.py files")
    ap.add_argument("--obs", help="observed file in the extract format")
    ap.add_argument("--out", default="out")
    ap.add_argument("--tag", default="cmip6", help="output name: satl_<tag>.* (amip for the atmosphere-only runs)")
    ap.add_argument("--causal", action="store_true", help="also the lead-lag and persistence tests")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.obs:
        (out / "satl_obs.json").write_text(json.dumps(observed(a.obs), indent=1))
        print((out / "satl_obs.json").read_text())
        if a.causal:
            (out / "satl_obs_causal.json").write_text(json.dumps(causal_obs(a.obs), indent=1))
    if a.cmip6:
        paths = sorted(glob.glob(str(Path(a.cmip6) / "**" / "*.npz"), recursive=True))
        if not paths:
            sys.exit("no extracts")
        summary, maps, lat, lon = cmip6(paths)
        (out / f"satl_{a.tag}.json").write_text(json.dumps(summary, indent=1))
        np.savez_compressed(
            out / f"satl_{a.tag}_maps.npz", lat=lat, lon=lon, **{k.replace("|", "__"): v for k, v in maps.items()}
        )
        d0 = np.load(paths[0])
        exp = str(d0["experiment_id"]) if "experiment_id" in d0.files else "historical"
        span = f"{d0['months'][0][:4]}-{d0['months'][-1][:4]}"
        title = (
            f"Negative South Atlantic dipole (SW - NE), CMIP6 {exp} {span}: {len(summary['models'])} models, "
            f"{summary['n_members']} members"
        )
        figure(maps, lat, lon, out / f"satl_{a.tag}.webp", title)
        print(json.dumps(summary["cases"], indent=1)[:4000])
        if a.causal:
            (out / f"satl_{a.tag}_causal.json").write_text(json.dumps(causal(paths), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
