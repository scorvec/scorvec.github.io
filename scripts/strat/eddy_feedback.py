#!/usr/bin/env python3
"""Is the tropospheric eddy feedback too weak in models, and does it set the surface response to sudden stratospheric
warmings (SSWs)? ERA5 against every CMIP6 historical run, from eddy_extract.py's daily series.

ANOMALIES. Each run against itself, on its own calendar: four annual harmonics and a linear trend (cmip6_coupling).

FEEDBACK MEASURES (northern winter). F = -(1 / (a cos^2 lat)) d([u*v*] cos^2 lat) / d lat is the eddy acceleration
(the quasi-geostrophic horizontal EP-flux divergence as a zonal acceleration), from daily-mean winds.
  efp      the eddy feedback parameter (Hardiman et al. 2022; Saffin et al. 2024): r^2 across winters of DJF-mean
           zonal-mean u with DJF-mean F at each latitude and level, averaged over 25-72.5N and 250 + 500 hPa (the two
           levels every model's day table has); ERA5 also over 200-600 hPa (Saffin et al.) and at 500 hPa (Hardiman)
  g_na     DJF-mean barotropic energy generation rate over the North Atlantic (eddy_extract.Atlantic; negative =
           eddies feeding the large-scale flow), Saffin et al.'s steadier measure; 1e-5 m2 s-3
  b        Lorenz & Hartmann (2001): the jet index z (each run's leading EOF of daily DJF u anomalies, 250 + 500 hPa,
           20-87.5N; + = poleward jet) and its eddy forcing m (F projected on the same pattern); b = cov(z(t),
           m(t + L)) / var(z) over lags L = 7-14 days, per day; tau = z's e-folding time, days
  nao_var  variance of DJF-mean NAO (Stephenson et al. 2006), hPa^2, which Saffin et al. found inflates the EFP

SSW COUPLING.
  Onsets: Charlton & Polvani (2007) at 10 hPa (cmip6_coupling.ssw_onsets); ERA5's from MERRA-2's u(60N, 10 hPa) in
  strat_history.nc (WeatherBench 2 stops at 50 hPa), so ERA5 events start in 1980/81.
  Per event, days 1-30 and 31-60: cap = 65-90N sea-level pressure anomaly (hPa; + = negative NAM), nao, imprint
  (slp_p), z, m, u100 = u(60N, 100 hPa) anomaly.
  gain     across events, cap (days 1-30) regressed on u100 (days 1-30), hPa per m/s (negative = coupling)
  down     every December-March day: the next 30 days' mean cap regressed on today's u100 with today's cap as a second
           predictor (persistence), hPa per m/s; down_sd the same in standard deviations

TESTS.
  1. ERA5 against each model's spread: the model's measure in 1000 bootstrap samples of its member-winters (events for
     gain) of ERA5's size; ERA5's percentile and the one-sided p of ERA5 lying beyond the model on the strong-feedback
     side (higher efp and b, lower g_na, more negative gain and down); Benjamini-Hochberg 10 % across models.
  2. Across models (member means, models with >= 2 members of 64 winters weighted equally): r of each feedback measure
     with gain and down, permutation p (10000), Benjamini-Hochberg 10 % over those pairs; ERA5's place on each line.
     Sensitivity (chosen after test 2): every model, with down_sd as a third coupling measure, BH over those pairs;
     and whether one run's sampling noise links a pair: r of member deviations from their model's mean.
  3. Saffin et al.'s confound: r(efp, nao_var) across models, and across ERA5's bootstrap samples.
  4. Resolution: r of each measure with log10 of each model's nominal resolution (its day ua store's attribute),
     permutation p, BH over the measures (ERA5 is on 1.5 deg, ~165 km).

    python eddy_feedback.py --runs extracts/ --strat reference/strat_history.nc --catalog pangeo-cmip6.csv \
        --out reference/eddy_feedback.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cmip6_coupling as CC
import eddy_extract as EX

A = EX.EARTH_RADIUS
DAY = 86400.0
TROP = (250, 500)
EFP_LAT = (25.0, 72.5)
ERA5_EFP = {"efp_200_600": (200, 250, 300, 400, 500, 600), "efp_500": (500,)}
LAGS = np.arange(7, 15)
DJF, DJFM = (12, 1, 2), (12, 1, 2, 3)
CAP = 65.0
BOOT, PERM = 1000, 10000
FEEDBACK = {"efp": 1, "g_na": -1, "b": 1}  # sign of a stronger feedback
COUPLING = {"gain": -1, "down": -1}  # sign of a stronger downward coupling
RNG = np.random.default_rng(20261108)


# ------------------------------------------------------------------------------------------------------ loading ---
def load(paths: list[Path]) -> dict:
    """One run (or ERA5's spans) in date order."""
    zs = [np.load(p) for p in paths]
    r = {k: np.concatenate([z[k] for z in zs]) for k in zs[0].files if k not in ("calendar", "levels", "lat")}
    o = np.lexsort((r["day"], r["month"], r["year"]))
    r = {k: v[o] for k, v in r.items()}
    r.update(calendar=str(zs[0]["calendar"]), levels=[int(x) for x in zs[0]["levels"]], lat=zs[0]["lat"].astype(float))
    r["winter"] = EX.winter(r["year"].astype(int), r["month"].astype(int))
    return r


def anomalies(r: dict, y: np.ndarray) -> np.ndarray:
    """A daily series of run r (time first) less its four-harmonic, linear-trend fit, column by column."""
    frac, _ = CC.year_fraction(r["year"], r["month"], r["day"], r["calendar"])
    X = CC.design(frac, r["year"].astype(float), 4, True)
    y = np.asarray(y, float)
    Y = y.reshape(len(y), -1)
    out = np.full(Y.shape, np.nan)
    ok = np.isfinite(Y).all(0)
    if ok.any():
        out[:, ok] = Y[:, ok] - X @ np.linalg.lstsq(X, Y[:, ok], rcond=None)[0]
    for c in np.flatnonzero(~ok & (np.isfinite(Y).mean(0) > 0.9)):
        out[:, c] = CC.anomaly(Y[:, c], X)
    return out.reshape(y.shape)


def accel(uv: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """The eddy acceleration F from [u*v*] (..., lat), m/s per day."""
    phi = np.deg2rad(lat)
    c2 = np.cos(phi) ** 2
    return -np.gradient(uv * c2, phi, axis=-1) / (A * c2) * DAY


def seasonal(x: np.ndarray, r: dict, months=DJF, min_days: int = 85) -> np.ndarray:
    """Means over `months` of each winter with at least min_days of them: (winters, ...)."""
    sel = np.isin(r["month"], months)
    ws = [w for w in np.unique(r["winter"][sel]) if (sel & (r["winter"] == w)).sum() >= min_days]
    return np.stack([np.nanmean(x[sel & (r["winter"] == w)], 0) for w in ws])


def area_mean(x: np.ndarray, lat: np.ndarray, lo: float, hi: float = 90.0) -> np.ndarray:
    sel = (lat >= lo) & (lat <= hi)
    w = np.cos(np.deg2rad(lat[sel]))
    return (x[..., sel] * w).sum(-1) / w.sum()


# --------------------------------------------------------------------------------------------- feedback measures ---
def efp(u: np.ndarray, f: np.ndarray, lat: np.ndarray, levels: list, use, rows=None) -> np.ndarray:
    """Mean r^2 over EFP_LAT and the `use` levels of winter means u, f (winters, levels, lat); rows (draws, winters)
    resamples the winters."""
    jl = (lat >= EFP_LAT[0]) & (lat <= EFP_LAT[1])
    kl = [levels.index(L) for L in use]
    U, F = u[:, kl][:, :, jl], f[:, kl][:, :, jl]
    if rows is not None:
        U, F = U[rows], F[rows]
    U, F = U - U.mean(-3, keepdims=True), F - F.mean(-3, keepdims=True)
    rr = (U * F).sum(-3) / np.sqrt((U * U).sum(-3) * (F * F).sum(-3))
    return (rr**2).mean((-2, -1))


def jet_eof(ua: np.ndarray, lat: np.ndarray, sel: np.ndarray):
    """The projection (..., lat) -> (...) onto the leading EOF of daily u anomalies ua (days, lat) on days `sel`,
    signed so that + is a poleward jet."""
    w = np.cos(np.deg2rad(lat))
    X = ua[sel & np.isfinite(ua).all(1)] * np.sqrt(w)
    e = np.linalg.svd(X - X.mean(0), full_matrices=False)[2][0] / np.sqrt(w)
    if e[np.argmin(abs(lat - 60))] < e[np.argmin(abs(lat - 35))]:
        e = -e
    norm = (w * e * e).sum()
    return lambda x: (x * w * e).sum(-1) / norm


def jet(ua: np.ndarray, fa: np.ndarray, lat: np.ndarray, r: dict) -> tuple[np.ndarray, np.ndarray]:
    """The jet index z (m/s) and its eddy forcing m (m/s per day) on the leading EOF of DJF daily u anomalies."""
    proj = jet_eof(ua, lat, np.isin(r["month"], DJF))
    return proj(ua), proj(fa)


def lh_parts(z: np.ndarray, m: np.ndarray, r: dict) -> np.ndarray:
    """Per winter: [days, sum z, sum z^2, then per lag: pairs, sum z, sum m, sum z m] for z on DJF days and m L days
    later in December-March of the same winter, so the feedback can be summed over any set of winters."""
    ws = np.unique(r["winter"])
    out = np.zeros((len(ws), 3 + 4 * len(LAGS)))
    djf, djfm = np.isin(r["month"], DJF), np.isin(r["month"], DJFM)
    n = len(z)
    t = np.flatnonzero(djf & np.isfinite(z))
    for k, w in enumerate(ws):
        tt = t[r["winter"][t] == w]
        out[k, :3] = len(tt), z[tt].sum(), (z[tt] ** 2).sum()
        for li, L in enumerate(LAGS):
            s = tt[tt + L < n]
            s = s[(r["winter"][s + L] == w) & djfm[s + L] & np.isfinite(m[s + L])]
            out[k, 3 + 4 * li : 7 + 4 * li] = len(s), z[s].sum(), m[s + L].sum(), (z[s] * m[s + L]).sum()
    return out


def lh_b(parts: np.ndarray) -> np.ndarray:
    """b from summed parts (..., winters, columns) -> (...)."""
    s = parts.sum(-2)
    var = s[..., 2] / s[..., 0] - (s[..., 1] / s[..., 0]) ** 2
    bs = []
    for li in range(len(LAGS)):
        n, sz, sm, szm = (s[..., 3 + 4 * li + j] for j in range(4))
        bs.append((szm / n - sz / n * sm / n) / var)
    return np.mean(bs, 0)


def tau(z: np.ndarray, r: dict) -> float:
    """e-folding time (days) of z's autocorrelation on DJF days, within winters."""
    djf = np.isin(r["month"], DJF) & np.isfinite(z)
    zz = z - z[djf].mean()
    var, prev, e1 = (zz[djf] ** 2).mean(), 1.0, np.exp(-1)
    for L in range(1, 60):
        s = np.flatnonzero(djf[:-L] & djf[L:] & (r["winter"][:-L] == r["winter"][L:]))
        acf = (zz[s] * zz[s + L]).mean() / var
        if acf < e1:
            return float(L - 1 + (prev - e1) / (prev - acf))
        prev = acf
    return float("nan")


# --------------------------------------------------------------------------------------------------- SSW coupling ---
def onsets(u10: np.ndarray, r: dict) -> list[int]:
    return CC.ssw_onsets(pd.DataFrame({"u10": u10, "month": r["month"], "year": r["year"]}))


def merra2_onsets(path: Path, r: dict) -> list[int]:
    """ERA5 positions of the MERRA-2 onsets (strat_history.nc's daily u(60N, 10 hPa), 1980 on)."""
    import xarray as xr

    with xr.open_dataset(path) as f:
        u10 = f.u10.load()
    t = pd.DatetimeIndex(u10.time.values)
    f = pd.DataFrame({"u10": u10.values.astype(float), "month": t.month, "year": t.year})
    key = {(y, m, d): i for i, (y, m, d) in enumerate(zip(r["year"], r["month"], r["day"]))}
    out = []
    for i in CC.ssw_onsets(f):
        k = (t[i].year, t[i].month, t[i].day)
        if k in key:
            out.append(key[k])
    return out


def event_table(ev: list[int], series: dict, r: dict) -> pd.DataFrame:
    """Each onset's window means (days 1-30 and 31-60, inside its winter)."""
    rows, n = [], len(r["winter"])
    for i in ev:
        row = {"i": i}
        for name, (a, b) in (("d1_30", (1, 30)), ("d31_60", (31, 60))):
            if i + b < n and r["winter"][i + b] == r["winter"][i]:
                row |= {f"{k}_{name}": float(np.nanmean(x[i + a : i + b + 1])) for k, x in series.items()}
        rows.append(row)
    return pd.DataFrame(rows)


def slope(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok] - x[ok].mean(), y[ok] - y[ok].mean()
    return float((x * y).sum() / (x * x).sum()) if ok.sum() > 3 else float("nan")


def down_parts(cap: np.ndarray, u100: np.ndarray, r: dict) -> np.ndarray:
    """Per winter, the cross-products of [1, u100(t), cap(t), next-30-day cap] over December-March days t whose next
    30 days lie in the same winter: (winters, 4, 4), so the regression can be summed over any set of winters."""
    n = len(cap)
    nxt = np.full(n, np.nan)
    c = np.r_[0.0, np.nancumsum(cap)]
    nxt[: n - 30] = (c[31:] - c[1 : n - 29]) / 30.0
    ok = np.isin(r["month"], DJFM) & np.isfinite(cap) & np.isfinite(u100)
    ok[: n - 30] &= r["winter"][30:] == r["winter"][: n - 30]
    ok[n - 30 :] = False
    ok &= np.isfinite(nxt)
    V = np.column_stack([np.ones(n), u100, cap, nxt])
    ws = np.unique(r["winter"])
    return np.stack([V[ok & (r["winter"] == w)].T @ V[ok & (r["winter"] == w)] for w in ws])


def down_coef(parts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(down in hPa per m/s, down_sd) from summed cross-products (..., winters, 4, 4)."""
    S = parts.sum(-3)
    beta = np.linalg.solve(S[..., :3, :3], S[..., :3, 3:4])[..., 0]
    n = S[..., 0, 0]
    sd_u = np.sqrt(S[..., 1, 1] / n - (S[..., 0, 1] / n) ** 2)
    sd_y = np.sqrt(S[..., 3, 3] / n - (S[..., 0, 3] / n) ** 2)
    return beta[..., 1], beta[..., 1] * sd_u / sd_y


# ------------------------------------------------------------------------------------------------------- per run ---
def reduce_run(r: dict, ssw: list[int] | None = None) -> dict:
    """Every measure's ingredients for one run: winter means, per-winter parts, the event table."""
    lat, lev = r["lat"], r["levels"]
    kt = [lev.index(L) for L in TROP]
    ua = anomalies(r, r["u"])
    f = accel(r["uv"].astype(float), lat)
    z, m = jet(ua[:, kt].mean(1), anomalies(r, f[:, kt].mean(1)), lat, r)
    cap = area_mean(anomalies(r, r["slp"]), lat, CAP)
    j60, k100 = int(np.argmin(abs(lat - 60))), lev.index(100)
    u100 = ua[:, k100, j60]
    if ssw is None:
        ssw = onsets(r["u"][:, lev.index(10), j60].astype(float), r)
    series = {
        "cap": cap, "nao": anomalies(r, r["nao"]), "imprint": anomalies(r, r["slp_p"]), "z": z, "m": m, "u100": u100,
    }  # fmt: skip
    out = {
        "u_djf": seasonal(r["u"].astype(float), r),
        "f_djf": seasonal(f, r),
        "g_djf": seasonal(r["g_na"].astype(float), r) * 1e5,
        "nao_djf": seasonal(r["nao"].astype(float), r),
        "lh": lh_parts(z, m, r),
        "tau": tau(z, r),
        "down": down_parts(cap, u100, r),
        "events": event_table(ssw, series, r),
        "winters": len(np.unique(r["winter"])),
        "levels": lev,
        "lat": lat,
    }
    if "uv6" in r:
        out["f6_djf"] = seasonal(accel(r["uv6"].astype(float), lat), r)
    return out


def measures(x: dict, rows=None, ev_rows=None) -> dict:
    """The measures of one sample: x from reduce_run (or several pooled); rows/ev_rows resample winters/events."""
    lat, lev = x["lat"], x["levels"]
    w = (lambda a: a[rows]) if rows is not None else (lambda a: a)
    ev = x["events"] if ev_rows is None else x["events"].iloc[ev_rows]
    down, down_sd = down_coef(w(x["down"]))
    return {
        "efp": efp(x["u_djf"], x["f_djf"], lat, lev, TROP, rows),
        "g_na": w(x["g_djf"]).mean(-1),
        "b": lh_b(w(x["lh"])),
        "nao_var": w(x["nao_djf"]).var(-1),
        "down": down,
        "down_sd": down_sd,
        "gain": slope(ev.u100_d1_30.values, ev.cap_d1_30.values) if "u100_d1_30" in ev else np.nan,
    }


def pool(xs: list[dict]) -> dict:
    """Several runs of one model as one sample (winters and events end to end)."""
    out = {k: np.concatenate([x[k] for x in xs]) for k in ("u_djf", "f_djf", "g_djf", "nao_djf", "lh", "down")}
    out["events"] = pd.concat([x["events"] for x in xs], ignore_index=True)
    out.update(lat=xs[0]["lat"], levels=xs[0]["levels"], winters=sum(x["winters"] for x in xs))
    return out


# ---------------------------------------------------------------------------------------------------------- tests ---
def permutation_r(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    r0 = float(np.corrcoef(x, y)[0, 1])
    rp = np.array([np.corrcoef(x, RNG.permutation(y))[0, 1] for _ in range(PERM)])
    return r0, float((1 + (np.abs(rp) >= abs(r0)).sum()) / (PERM + 1))


def across(models: dict, names: list, coupling: dict, em: dict) -> list[dict]:
    """r across `names` (member means) of each feedback measure with each coupling measure, permutation p,
    Benjamini-Hochberg 10 % over the pairs (counted only with the expected sign); ERA5's place on each line."""
    out, ps = [], []
    for fk in FEEDBACK:
        for ck in coupling:
            x = np.array([models[n]["mean"][fk] for n in names], float)  # None (no value) -> nan
            y = np.array([models[n]["mean"][ck] for n in names], float)
            r0, p = permutation_r(x, y)
            ok = np.isfinite(x) & np.isfinite(y)
            fit = np.polyfit(x[ok], y[ok], 1)
            out.append(
                {"feedback": fk, "coupling": ck, "models": int(ok.sum()), "r": rnd(r0, 3), "p": rnd(p),
                 "expected_sign": FEEDBACK[fk] * coupling[ck], "era5_predicted": rnd(np.polyval(fit, em[fk])),
                 "era5_actual": rnd(em[ck])}
            )  # fmt: skip
            ps.append(p)
    for t, sig in zip(out, CC.fdr(np.array(ps))):
        t["fdr10"] = bool(sig and np.sign(t["r"]) == t["expected_sign"])
    return out


def resolution(catalog: str, names: list) -> dict:
    """{model: nominal resolution, km} from the attributes of each model's newest historical day ua store."""
    import re

    import requests

    cat = EX.catalog(catalog)
    out = {}
    for n in names:
        g = cat[(cat.source_id == n) & (cat.variable_id == "ua")]
        if not len(g):
            continue
        url = g.sort_values("version").zstore.iloc[-1].replace("gs://", "https://storage.googleapis.com/")
        try:
            attrs = requests.get(url.rstrip("/") + "/.zattrs", timeout=30).json()
        except (requests.RequestException, ValueError) as err:
            print(f"  {n}: no resolution ({type(err).__name__})")
            continue
        m = re.match(r"([\d.]+)\s*km", str(attrs.get("nominal_resolution", "")))
        if m:
            out[n] = float(m.group(1))
    return out


def rnd(v, k=4):
    return None if v is None or not np.isfinite(v) else float(round(float(v), k))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="directory of eddy_extract.py's npz files")
    ap.add_argument("--strat", required=True, help="strat_history.nc (MERRA-2 u(60N, 10 hPa) for ERA5's SSW dates)")
    ap.add_argument("--catalog", help="the Pangeo CMIP6 catalogue, for each model's nominal resolution")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files = sorted(p for p in Path(a.runs).glob("*.npz") if ".part" not in p.name)
    era5 = load([p for p in files if p.stem.startswith("ERA5_")])
    E = reduce_run(era5, merra2_onsets(Path(a.strat), era5))
    nw, ne = len(E["u_djf"]), len(E["events"].dropna(subset=["u100_d1_30"]))
    boot = RNG.integers(0, nw, (BOOT, nw))
    em = measures(E)
    eb = measures(E, rows=boot)
    eb["gain"] = np.array([measures(E, ev_rows=RNG.integers(0, ne, ne))["gain"] for _ in range(BOOT)])
    era5_doc = {k: rnd(v) for k, v in em.items()} | {
        "ci90": {k: [rnd(np.nanpercentile(v, 5)), rnd(np.nanpercentile(v, 95))] for k, v in eb.items()},
        "winters_djf": nw, "ssw_events": ne, "tau": rnd(E["tau"], 2),
    }  # fmt: skip
    for k, levs in ERA5_EFP.items():
        era5_doc[k] = rnd(efp(E["u_djf"], E["f_djf"], E["lat"], E["levels"], levs))
    era5_doc["efp_6h_eddies"] = rnd(efp(E["u_djf"], E["f6_djf"], E["lat"], E["levels"], TROP))
    print(f"ERA5: {nw} DJFs, {ne} SSWs: " + ", ".join(f"{k} {v}" for k, v in era5_doc.items() if k != "ci90"))

    by_model: dict[str, list[Path]] = {}
    for p in files:
        if not p.stem.startswith("ERA5_"):
            by_model.setdefault(p.stem.rsplit("_", 1)[0], []).append(p)
    models = {}
    for name, ps in sorted(by_model.items()):
        xs = []
        for p in ps:
            try:
                xs.append(reduce_run(load([p])))
            except Exception as err:  # noqa: BLE001  (a broken run is left out, and said)
                print(f"  {p.stem}: left out ({type(err).__name__}: {err})")
        if not xs:
            continue
        per = [measures(x) for x in xs]
        P = pool(xs)
        npool, nev = len(P["u_djf"]), len(P["events"].dropna(subset=["u100_d1_30"]))
        mb = measures(P, rows=RNG.integers(0, npool, (BOOT, nw)))
        if nev >= ne:
            mb["gain"] = np.array([measures(P, ev_rows=RNG.integers(0, nev, ne))["gain"] for _ in range(BOOT)])
        else:
            mb["gain"] = np.full(BOOT, np.nan)
        doc = {"members": len(xs), "winters": int(npool), "ssw_events": int(nev)}
        doc["ssw_per_decade"] = rnd(10 * len(P["events"]) / npool, 2)
        doc["mean"] = {k: rnd(np.nanmean([q[k] for q in per])) for k in per[0]}
        doc["tau"] = rnd(np.mean([x["tau"] for x in xs]), 2)
        doc["members_values"] = {k: [rnd(q[k]) for q in per] for k in per[0]}
        doc["percentile_of_era5"], doc["p_era5_stronger"] = {}, {}
        for k, sgn in {**FEEDBACK, **COUPLING, "nao_var": 1, "down_sd": -1}.items():
            d, e = np.asarray(mb[k], float), em[k]
            d = d[np.isfinite(d)]
            if len(d) < BOOT // 2 or not np.isfinite(e):
                continue
            doc["percentile_of_era5"][k] = rnd(100 * (d < e).mean(), 1)
            doc["p_era5_stronger"][k] = rnd((1 + (sgn * d >= sgn * e).sum()) / (len(d) + 1))
        models[name] = doc
        print(
            f"  {name}: {len(xs)} runs, {npool} DJFs, {nev} SSWs; "
            + ", ".join(f"{k} {doc['mean'][k]} (ERA5 pct {doc['percentile_of_era5'].get(k)})" for k in ("efp", "g_na", "b", "gain", "down"))
        )  # fmt: skip

    # 1. ERA5 against each model: Benjamini-Hochberg across models, per measure
    test1 = {}
    for k in {**FEEDBACK, **COUPLING, "nao_var": 1, "down_sd": -1}:
        names = [n for n in models if k in models[n]["p_era5_stronger"]]
        p = np.array([models[n]["p_era5_stronger"][k] for n in names])
        sig = CC.fdr(p)
        test1[k] = {
            "models": len(names), "era5_stronger_fdr10": [n for n, s in zip(names, sig) if s],
            "median_percentile": rnd(np.median([models[n]["percentile_of_era5"][k] for n in names]), 1),
        }  # fmt: skip
    # 2. across models (those with >= 2 members), feedback vs coupling; then every model, and the member noise
    keep = [n for n in models if models[n]["members"] >= 2]
    test2 = across(models, keep, COUPLING, em)
    test2_all = across(models, list(models), {**COUPLING, "down_sd": -1}, em)
    noise = []
    for fk in FEEDBACK:
        for ck in (*COUPLING, "down_sd"):
            dx, dy = [], []
            for n in keep:
                x = np.array(models[n]["members_values"][fk], float)
                y = np.array(models[n]["members_values"][ck], float)
                ok = np.isfinite(x) & np.isfinite(y)
                if ok.sum() >= 2:
                    dx += list(x[ok] - x[ok].mean())
                    dy += list(y[ok] - y[ok].mean())
            r0, p = permutation_r(np.array(dx), np.array(dy))
            noise.append({"feedback": fk, "coupling": ck, "members": len(dx), "r": rnd(r0, 3), "p": rnd(p)})
    # 3. Saffin's confound
    x = np.array([models[n]["mean"]["efp"] for n in keep], float)
    y = np.array([models[n]["mean"]["nao_var"] for n in keep], float)
    r0, p = permutation_r(x, y)
    test3 = {
        "r_models": rnd(r0, 3),
        "p_models": rnd(p),
        "r_era5_bootstrap": rnd(np.corrcoef(eb["efp"], eb["nao_var"])[0, 1], 3),
    }

    # 4. resolution
    test4 = []
    if a.catalog:
        res = resolution(a.catalog, list(models))
        for n, km in res.items():
            models[n]["resolution_km"] = km
        names = [n for n in models if n in res]
        x = np.log10([res[n] for n in names])
        for k in (*FEEDBACK, *COUPLING, "nao_var", "down_sd"):
            r0, p = permutation_r(x, np.array([models[n]["mean"][k] for n in names], float))
            test4.append({"measure": k, "models": len(names), "r_log_resolution": rnd(r0, 3), "p": rnd(p)})
        for t, sig in zip(test4, CC.fdr(np.array([t["p"] for t in test4]))):
            t["fdr10"] = bool(sig)
        print("Resolution:", ", ".join(f"{t['measure']} r {t['r_log_resolution']} p {t['p']}" for t in test4))

    doc = {"era5": era5_doc, "models": models, "test_era5_vs_models": test1, "test_across_models": test2,
           "test_across_all_models": test2_all, "test_member_noise": noise, "test_efp_nao": test3,
           "test_resolution": test4, "boot": BOOT, "perm": PERM}  # fmt: skip
    Path(a.out).write_text(json.dumps(doc, indent=1))
    print("\nERA5 against the models (median percentile of ERA5; models where ERA5 is stronger, FDR 10 %):")
    for k, v in test1.items():
        print(f"  {k:8s} {v['median_percentile']}  {len(v['era5_stronger_fdr10'])}/{v['models']}")
    for title, tests in (("Across models (>= 2 members)", test2), ("Across every model", test2_all)):
        print(f"{title}:")
        for t in tests:
            print(f"  {t['feedback']:5s} vs {t['coupling']:7s} r {t['r']} p {t['p']} {'FDR' if t['fdr10'] else ''}")
    print("Member noise:", ", ".join(f"{t['feedback']}/{t['coupling']} {t['r']}" for t in noise))
    print(f"EFP vs NAO variance: {test3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
