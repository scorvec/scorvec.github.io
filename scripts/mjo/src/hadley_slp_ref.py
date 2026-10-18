#!/usr/bin/env python3
"""Drift and skill of the tropical sea-level pressure forecasts (hadley_slp_fc.py), measured on archived runs against
ERA5: the drift the live members are corrected by, and the evidence that gates which lead days the page shows.

ARCHIVE (release hslp-ref-v1, hslp_ref.npz). ERA5's daily means (00, 06, 12 and 18 UTC from ARCO, conservatively onto the
1.5 deg grid of the normals, build_slp_clim.ROWS) of every day since overturn.EPOCH; each model's archived 00Z runs on
overturn's 5-day lattice of starts: the mean of REF_MEMBERS perturbed members' daily means at lead days LEADS (valid
times 00 to 18 UTC of each day: steps 0..354 h). The Google mirror keeps IFS-ENS from March 2024 and AIFS-ENS from July
2025; GEFS comes from NOAA's S3 bucket. Values are stored as int16 hundredths of a hPa about 1010 hPa.

DRIFT. A run's error at lead L, e(L) = its daily mean less ERA5's for the valid day, is regressed per grid point on the
start's common-year day (mean + annual harmonic, DRIFT_TERMS): d(L, doy). Each value is then shrunk towards zero by its
signal-to-noise, d* = d d^2 / (d^2 + var d), var d from the regression's residual variance inflated for the serial
correlation of starts 5 days apart ((1 + r) / (1 - r), r the domain median lag-1 correlation of the residuals at that
lead): a systematic error (a model's own reduction to sea level, a bias under a subtropical high) is taken out, a drift
that is sampling noise is not.

SKILL. The members' mean less d*, against the ERA5 normal, over the domain below build_slp_clim.TERRAIN_M:
MSSS = 1 - sum w (f - o)^2 / sum w o^2, w = cos(lat), f and o the forecast and ERA5 anomalies normalised by the day's sd
(the gate) and, reported alongside, in hPa. The drift is cross-validated: each start is corrected by the drift fitted
without its fold, the calendar month of its start. Intervals and one-sided p-values come from a bootstrap over whole
months of starts (BOOT; neighbouring starts share weather, and a persistent anomaly makes a month of them one sample),
with Benjamini-Hochberg at FDR_Q over every model-lead; a model-lead passes when it survives and its MSSS reaches
MIN_SKILL, and a model shows the lead days from day 0 up to the first that fails. The blend is the three corrected
means weighted by the live member counts (MODELS), on the starts they share.

    python src/hadley_slp_ref.py truth --since 2024-03-01 [--part k/n] --out era5.npz
    python src/hadley_slp_ref.py archive --model ifs --since 2024-03-01 [--part k/n] --out ifs.npz
    python src/hadley_slp_ref.py pack prev/*.npz parts/*.npz          # -> REF
    python src/hadley_slp_ref.py verify --out-dir out                 # -> DRIFT, SKILL and figures in out
"""

from __future__ import annotations

import pyproj  # noqa: F401  (load pyproj's libproj before xarray/eccodes)

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ecmwf"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "era5"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "strat"))
import build_slp_clim as C
import hadley_branches as HB
import overturn as O
from doycal import noleap_doy

REF_DIR = Path(__file__).resolve().parent.parent / "data" / "reference"
REF = REF_DIR / "hslp_ref.npz"  # release hslp-ref-v1
DRIFT = REF_DIR / "hslp_drift.npz"  # release hslp-ref-v1
SKILL = REF_DIR / "hslp_skill.json"  # committed: the page's lead gate
LEADS = tuple(range(15))  # lead days
HOURS = (0, 6, 12, 18)
REF_MEMBERS = 10  # perturbed members per archived run
MODELS = {
    "aifs": {"label": "AIFS-ENS", "store": "aifs-ens", "first": "2025-07-01", "members": 50},
    "ifs": {"label": "IFS-ENS", "store": "ifs", "first": "2024-03-01", "members": 50},
    "gefs": {"label": "GEFS", "store": None, "first": "2024-03-01", "members": 30},
}
DRIFT_TERMS = 3  # mean, cos1, sin1 of the start's common-year day
BOOT = 2000
FDR_Q = 0.10
MIN_SKILL = 0.05
MIN_BLOCKS = 10  # months of starts the bootstrap needs before anything can pass
MIN_STARTS = 30  # archived starts a model needs before its drift is fitted and it is scored
P_OFF, P_SCALE = 1010.0, 100.0  # int16 storage: hundredths of a hPa about 1010 hPa


def steps_of(lead: int) -> tuple:
    return tuple(24 * lead + h for h in HOURS)


def pack16(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round((x - P_OFF) * P_SCALE), -32767, 32767).astype(np.int16)


def unpack16(q: np.ndarray) -> np.ndarray:
    return (q.astype(np.float32) / P_SCALE + P_OFF).astype(np.float32)


# ── sources ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def to_domain(vals: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """A global field (Pa) on any regular grid -> hPa on the normals' rows of the 1.5 deg grid."""
    return (HB.coarse15(vals, lat, lon)[C.ROWS] / 100.0).astype(np.float32)


def decode(msg: bytes) -> tuple[int, int, np.ndarray]:
    """(member number, step h, field on the domain) of one GRIB message."""
    import eccodes

    h = eccodes.codes_new_from_message(msg)
    try:
        g = {k: eccodes.codes_get(h, k) for k in ("Ni", "Nj", "jScansPositively", "endStep")}
        n = eccodes.codes_get(h, "number") if eccodes.codes_is_defined(h, "number") else 0
        la0 = eccodes.codes_get(h, "latitudeOfFirstGridPointInDegrees")
        lo0 = eccodes.codes_get(h, "longitudeOfFirstGridPointInDegrees")
        dj = eccodes.codes_get(h, "jDirectionIncrementInDegrees")
        di = eccodes.codes_get(h, "iDirectionIncrementInDegrees")
        vals = eccodes.codes_get_values(h).reshape(g["Nj"], g["Ni"])
    finally:
        eccodes.codes_release(h)
    lat = la0 + (dj if g["jScansPositively"] else -dj) * np.arange(g["Nj"])
    return int(n), int(g["endStep"]), to_domain(vals, lat, lo0 + di * np.arange(g["Ni"]))


def _messages(path: Path):
    blob = path.read_bytes()
    i = 0
    while (j := blob.find(b"GRIB", i)) >= 0:
        n = int.from_bytes(blob[j + 8 : j + 16], "big")  # GRIB2 section 0: the message's total length
        yield blob[j : j + n]
        i = j + n


def gefs_msl(date: str, member: int, step: int) -> bytes:
    """The PRMSL message of one GEFS member (0 = control) at one step of the 00Z run, byte-ranged from NOAA's S3 bucket
    through the file's .idx. Network only, so it runs in threads."""
    from qbo_duct import _get, gefs_url

    url = gefs_url(date, "00", "gec00" if member == 0 else f"gep{member:02d}", "a", step)
    lines = [x.split(":") for x in _get(url + ".idx").decode().splitlines() if x.strip()]
    for i, p in enumerate(lines):
        if p[3] == "PRMSL" and p[4] == "mean sea level":
            end = int(lines[i + 1][1]) - 1 if i + 1 < len(lines) else -1
            return bytes(_get(url, (int(p[1]), end)))
    raise RuntimeError(f"no PRMSL in {url}")


def fetch_run(model: str, init: pd.Timestamp, members: list[int], workers: int = 16) -> np.ndarray:
    """(member, lead day, lat, lon) daily means (hPa) of the members of the 00Z run at init on the domain, the mean of
    each lead day's four valid times. ECMWF: one file of the members' four steps per lead day from the Google mirror
    (a failure raises); GEFS: member-steps byte-ranged in threads, decoded here, a member-step that fails left NaN."""
    out = np.full((len(members), len(LEADS), C.LAT.size, C.LON.size), np.nan, np.float32)
    pos = {m: i for i, m in enumerate(members)}
    if model == "gefs":
        from concurrent.futures import ThreadPoolExecutor

        acc = np.zeros_like(out)
        cnt = np.zeros(out.shape[:2], int)
        jobs = [(m, L, s) for m in members for L in LEADS for s in steps_of(L)]
        with ThreadPoolExecutor(workers) as ex:
            futs = [(m, L, ex.submit(gefs_msl, f"{init:%Y%m%d}", m, s)) for m, L, s in jobs]
            for m, L, fut in futs:
                try:
                    acc[pos[m], L] += decode(fut.result())[2]
                    cnt[pos[m], L] += 1
                except Exception as e:  # noqa: BLE001
                    print(f"  GEFS {init:%Y-%m-%d} member {m} day {L}: {type(e).__name__} {str(e)[:80]}", flush=True)
        full = cnt == len(HOURS)
        out[full] = acc[full] / len(HOURS)
        return out
    import store as ecmwf

    cyc = ecmwf.Cycle(f"{init:%Y%m%d}", "00")
    for L in LEADS:
        p = ecmwf.ensure(cyc, ecmwf.Spec(MODELS[model]["store"], "pf", "msl", "sfc", (), steps_of(L), max(members)))
        fields = {}
        for msg in _messages(p):
            n, s, f = decode(msg)
            fields[n, s] = f
        for m in members:
            if all((m, s) in fields for s in steps_of(L)):
                out[pos[m], L] = np.mean([fields[m, s] for s in steps_of(L)], axis=0)
        p.unlink(missing_ok=True)
        p.with_suffix(p.suffix + ".json").unlink(missing_ok=True)
    return out


def era5_days(days) -> dict:
    """ERA5's daily means on the domain from ARCO; the first day it has not published (NaN-padded) ends the pass."""
    import era5_store

    ds = xr.open_zarr(era5_store.ARCO, chunks=None, storage_options={"token": "anon"})
    da = ds["mean_sea_level_pressure"]
    lat, lon = da.latitude.values, da.longitude.values
    keep, rows = [], []
    for d in days:
        try:
            v = da.sel(time=[d + pd.Timedelta(hours=h) for h in HOURS]).values
        except KeyError:  # beyond the store's time axis
            v = np.full(1, np.nan)
        if np.isnan(v).any():
            print(f"  ERA5 not published for {d:%Y-%m-%d}: stopping", flush=True)
            break
        keep.append(d)
        rows.append(pack16(to_domain(v.mean(0), lat, lon)))
        if d.day == 1:
            print(f"  ERA5 {d:%Y-%m}", flush=True)
    n = len(keep)
    return {
        "era5_day": np.array(keep, "datetime64[D]"),
        "era5_msl": np.array(rows, np.int16).reshape(n, C.LAT.size, C.LON.size),
    }


def archive(model: str, starts) -> dict:
    """The model's archived runs at the lattice starts: the REF_MEMBERS members' mean daily means, packed."""
    keep, rows = [], []
    for d in starts:
        try:
            x = fetch_run(model, d, list(range(1, REF_MEMBERS + 1)))
        except Exception as e:  # noqa: BLE001
            print(f"  {MODELS[model]['label']} {d:%Y-%m-%d}: {type(e).__name__} {str(e)[:80]}", flush=True)
            continue
        if np.isnan(x).any():
            print(f"  {MODELS[model]['label']} {d:%Y-%m-%d}: incomplete, skipped", flush=True)
            continue
        keep.append(d)
        rows.append(pack16(x.mean(0)))
        print(f"  {MODELS[model]['label']} {d:%Y-%m-%d} archived", flush=True)
    n = len(keep)
    return {
        f"{model}_start": np.array(keep, "datetime64[D]"),
        f"{model}_msl": np.array(rows, np.int16).reshape(n, len(LEADS), C.LAT.size, C.LON.size),
    }


def pack(paths, out: Path = REF) -> dict:
    """Merge truth/archive files (any order, overlaps allowed: the later file's copy of a day is kept) into REF."""
    acc: dict[str, list] = {}
    for p in paths:
        with np.load(p) as f:
            for k in f.files:
                acc.setdefault(k, []).append(f[k])
    doc = {}
    for key_t, key_x in [("era5_day", "era5_msl")] + [(f"{m}_start", f"{m}_msl") for m in MODELS]:
        if key_t not in acc:
            continue
        t, x = np.concatenate(acc[key_t]), np.concatenate(acc[key_x])
        _, last = np.unique(t[::-1], return_index=True)
        keep = t.size - 1 - last  # sorted by time
        doc[key_t], doc[key_x] = t[keep], x[keep]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **doc)
    return doc


# ── the drift ───────────────────────────────────────────────────────────────────────────────────────────────────────
def doy_of(days) -> np.ndarray:
    return noleap_doy(pd.DatetimeIndex(days))


def fit_drift(err: np.ndarray, starts: pd.DatetimeIndex, low: np.ndarray) -> dict:
    """The drift regression of one lead: err (start, point) -> {coef (DRIFT_TERMS, point), cov, s2 (point), kappa}."""
    X = C.harm(doy_of(starts))[:, :DRIFT_TERMS]
    coef = np.linalg.lstsq(X, err, rcond=None)[0]
    r = err - X @ coef
    s2 = (r**2).sum(0) / max(len(starts) - DRIFT_TERMS, 1)
    nxt = np.flatnonzero(np.diff(starts.values).astype("timedelta64[D]").astype(int) == O.LATTICE)
    rho = 0.0
    if nxt.size >= 10:
        num, den = (r[nxt] * r[nxt + 1]).sum(0), (r[nxt] ** 2).sum(0)
        ok = low & (den > 0)
        if ok.any():
            rho = float(np.clip(np.median(num[ok] / den[ok]), 0.0, 0.95))
    return {"coef": coef, "cov": np.linalg.inv(X.T @ X), "s2": s2, "kappa": (1 + rho) / (1 - rho), "rho": rho}


def eval_drift(fit: dict, doy, shrink: bool = True) -> np.ndarray:
    """(n doy, point) drift at common-year days, shrunk by its signal-to-noise unless shrink is False."""
    x = C.harm(doy)[:, :DRIFT_TERMS]
    d = x @ fit["coef"]
    if not shrink:
        return d
    var = np.einsum("ij,jk,ik->i", x, fit["cov"], x)[:, None] * fit["s2"][None] * fit["kappa"]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(d**2 + var > 0, d * d**2 / (d**2 + var), 0.0)


class Drift:
    """The live drift (DRIFT): d*(lead, doy) on the domain for a model, (len(LEADS), lat, lon); None without a fit."""

    def __init__(self, path: Path = DRIFT):
        self.f = dict(np.load(path)) if path.exists() else {}

    def at(self, model: str, doy: float) -> np.ndarray | None:
        if f"{model}_coef" not in self.f:
            return None
        out = []
        for j in range(len(LEADS)):
            fit = {k: self.f[f"{model}_{k}"][j] for k in ("coef", "cov", "s2", "kappa")}
            fit["coef"], fit["s2"] = fit["coef"].reshape(DRIFT_TERMS, -1), fit["s2"].ravel()
            out.append(eval_drift(fit, [doy])[0])
        return np.stack(out).reshape(len(LEADS), C.LAT.size, C.LON.size).astype(np.float32)


# ── verification ────────────────────────────────────────────────────────────────────────────────────────────────────
def bh(p, q=FDR_Q):
    """Benjamini-Hochberg: which of the p-values are rejected at false discovery rate q."""
    p = np.nan_to_num(np.asarray(p, float), nan=1.0)
    order = np.argsort(p)
    ranked = p[order] * p.size / np.arange(1, p.size + 1)
    passed = np.zeros(p.size, bool)
    below = np.flatnonzero(ranked <= q)
    if below.size:
        passed[order[: below.max() + 1]] = True
    return passed


def skill_test(fc, ref, block, nblock, counts):
    """Per-start squared-error sums (start, lead), NaN where not scored -> MSSS = 1 - sum fc / sum ref by lead, its 90 %
    interval and the one-sided p-value of MSSS <= 0, by bootstrap over blocks of starts."""
    fc, ref = np.nan_to_num(fc), np.nan_to_num(ref)
    bf, br = np.zeros((nblock, fc.shape[1])), np.zeros((nblock, fc.shape[1]))
    np.add.at(bf, block, fc)
    np.add.at(br, block, ref)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = 1 - fc.sum(0) / ref.sum(0)
        bs = 1 - (counts @ bf) / (counts @ br)
    p = np.where(np.isfinite(s), np.mean(~(bs > 0), axis=0), 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return s, np.nanpercentile(bs, 5, axis=0), np.nanpercentile(bs, 95, axis=0), p


def verify(out_dir: Path, seed: int = 1) -> dict:
    with np.load(REF) as f:
        ref = {k: f[k] for k in f.files}
    N = C.Normals()
    low = (~N.high).ravel()
    w = (np.cos(np.deg2rad(C.LAT))[:, None] * np.ones(C.LON.size)).ravel() * low
    epos = {pd.Timestamp(d): i for i, d in enumerate(ref["era5_day"])}
    E = unpack16(ref["era5_msl"]).reshape(len(epos), -1)
    norm, sd = {}, {}
    for d in epos:
        norm[d], sd[d] = N.normal(d).ravel(), N.sd(d).ravel()
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    sums, drift_doc, fits, corrected = {}, {}, {}, {}
    for m in MODELS:
        if f"{m}_start" not in ref or len(ref[f"{m}_start"]) < MIN_STARTS:
            print(f"{MODELS[m]['label']}: fewer than {MIN_STARTS} archived starts, not scored")
            continue
        starts = pd.DatetimeIndex(ref[f"{m}_start"])
        X = unpack16(ref[f"{m}_msl"]).reshape(len(starts), len(LEADS), -1)
        fold = np.array([f"{s:%Y-%m}" for s in starts])
        err = np.full(X.shape, np.nan, np.float32)
        for i, s in enumerate(starts):
            for L in LEADS:
                j = epos.get(s + pd.Timedelta(days=L))
                if j is not None:
                    err[i, L] = X[i, L] - E[j]
        cv, fits[m] = np.full_like(err, np.nan), []
        dstat = {k: [] for k in ("rms_hpa", "rms_fit_hpa", "rms_sd", "mean_hpa", "rho", "starts")}
        for L in LEADS:
            ok = np.isfinite(err[:, L, 0])
            fit = fit_drift(err[ok, L], starts[ok], low)
            fits[m].append(fit)
            for f in np.unique(fold[ok]):
                tr, te = ok & (fold != f), ok & (fold == f)
                cv[te, L] = eval_drift(fit_drift(err[tr, L], starts[tr], low), doy_of(starts[te]))
            d, draw = eval_drift(fit, doy_of(starts[ok])), eval_drift(fit, doy_of(starts[ok]), shrink=False)
            sds = np.stack([sd[s + pd.Timedelta(days=L)] for s in starts[ok]])
            wn = w / w.sum()
            dstat["rms_hpa"].append(float(np.sqrt((d**2 @ wn).mean())))
            dstat["rms_fit_hpa"].append(float(np.sqrt((draw**2 @ wn).mean())))
            dstat["rms_sd"].append(float(np.sqrt(((d / sds) ** 2 @ wn).mean())))
            dstat["mean_hpa"].append(float((d @ wn).mean()))
            dstat["rho"].append(round(fit["rho"], 3))
            dstat["starts"].append(int(ok.sum()))
        drift_doc[m] = {k: [round(x, 3) if isinstance(x, float) else x for x in v] for k, v in dstat.items()}
        corrected[m] = (starts, err - cv, err)
    # squared-error sums per start and lead: (start, lead) for the drift-corrected forecast, the raw one, the normal
    for m, (starts, ec, er) in corrected.items():
        sums[m] = _sums(starts, ec, er, epos, E, norm, sd, w)
    common = None
    if len(corrected) == len(MODELS):
        common = sorted(set.intersection(*(set(v[0]) for v in corrected.values())))
        nm = {m: MODELS[m]["members"] for m in MODELS}
        idx = {m: {s: i for i, s in enumerate(v[0])} for m, v in corrected.items()}
        tot = sum(nm.values())
        ec = sum(nm[m] * np.stack([corrected[m][1][idx[m][s]] for s in common]) for m in MODELS) / tot
        er = sum(nm[m] * np.stack([corrected[m][2][idx[m][s]] for s in common]) for m in MODELS) / tot
        sums["blend"] = _sums(pd.DatetimeIndex(common), ec, er, epos, E, norm, sd, w)
    tests = {}
    for m, S in sums.items():
        months = np.array([f"{s:%Y-%m}" for s in S["starts"]])
        names, block = np.unique(months, return_inverse=True)
        counts = np.stack(
            [np.bincount(x, minlength=names.size) for x in rng.integers(0, names.size, (BOOT, names.size))]
        )
        tests[m] = {
            "nblock": names.size,
            "sig": skill_test(S["fc_sig"], S["ref_sig"], block, names.size, counts),
            "hpa": skill_test(S["fc_hpa"], S["ref_hpa"], block, names.size, counts),
            "raw": skill_test(S["raw_sig"], S["ref_sig"], block, names.size, counts),
            "raw_hpa": skill_test(S["raw_hpa"], S["ref_hpa"], block, names.size, counts),
        }
        if names.size < MIN_BLOCKS:
            tests[m]["sig"][3][:] = 1.0
    order = list(tests)
    passed = bh(np.concatenate([tests[m]["sig"][3] for m in order])).reshape(len(order), len(LEADS))
    doc = {
        "members": REF_MEMBERS,
        "live_members": {m: MODELS[m]["members"] for m in MODELS},
        "bootstrap": BOOT,
        "blocks": "calendar months of start dates",
        "cross_validation": "drift refitted without the start's calendar month",
        "fdr_q": FDR_Q,
        "min_skill": MIN_SKILL,
        "lead_days": list(LEADS),
        "terrain_m": C.TERRAIN_M,
        "models": {},
    }
    for k, m in enumerate(order):
        t = tests[m]
        ok = passed[k] & (t["sig"][0] >= MIN_SKILL)
        show = 0
        while show < len(LEADS) and ok[show]:
            show += 1
        S = sums[m]
        doc["models"][m] = {
            "label": "Three ensembles" if m == "blend" else MODELS[m]["label"],
            "starts": len(S["starts"]),
            "first": f"{S['starts'][0]:%Y-%m-%d}",
            "last": f"{S['starts'][-1]:%Y-%m-%d}",
            "months": int(t["nblock"]),
            "scored": [int(x) for x in np.isfinite(S["fc_sig"]).sum(0)],
            "msss": _r3(t["sig"][0]),
            "msss_ci": [_r3(t["sig"][1]), _r3(t["sig"][2])],
            "p": [round(float(x), 4) for x in t["sig"][3]],
            "passed": [bool(x) for x in ok],
            "show_days": show,  # lead days 0 .. show_days - 1
            "msss_hpa": _r3(t["hpa"][0]),
            "msss_hpa_ci": [_r3(t["hpa"][1]), _r3(t["hpa"][2])],
            "msss_no_drift": _r3(t["raw"][0]),
            "msss_hpa_no_drift": _r3(t["raw_hpa"][0]),
            "drift": drift_doc.get(m),
        }
    SKILL.write_text(json.dumps(doc, indent=1) + "\n")
    (out_dir / SKILL.name).write_text(json.dumps(doc, indent=1) + "\n")
    save = {}
    for m, fl in fits.items():
        save[f"{m}_coef"] = np.stack([f["coef"] for f in fl]).reshape(len(LEADS), DRIFT_TERMS, C.LAT.size, C.LON.size)
        save[f"{m}_cov"] = np.stack([f["cov"] for f in fl])
        save[f"{m}_s2"] = np.stack([f["s2"] for f in fl]).reshape(len(LEADS), C.LAT.size, C.LON.size)
        save[f"{m}_kappa"] = np.array([f["kappa"] for f in fl])
    np.savez_compressed(DRIFT, **{k: v.astype(np.float32) for k, v in save.items()})
    report(doc)
    figures(doc, fits, corrected, out_dir)
    return doc


def _sums(starts, ec, er, epos, E, norm, sd, w) -> dict:
    """Squared-error sums (start, lead) of the corrected (ec) and raw (er) errors, normalised and in hPa, and of the
    ERA5 anomaly (the normal's error); NaN where ERA5 has not published the valid day."""
    n = len(starts)
    S = {k: np.full((n, len(LEADS)), np.nan) for k in ("fc_sig", "fc_hpa", "raw_sig", "raw_hpa", "ref_sig", "ref_hpa")}
    for i, s in enumerate(starts):
        for L in LEADS:
            d = s + pd.Timedelta(days=L)
            j = epos.get(d)
            if j is None or not np.isfinite(ec[i, L, 0]):
                continue
            o, inv = E[j] - norm[d], 1.0 / sd[d] ** 2
            S["ref_hpa"][i, L], S["ref_sig"][i, L] = w @ o**2, w @ (o**2 * inv)
            S["fc_hpa"][i, L], S["fc_sig"][i, L] = w @ ec[i, L] ** 2, w @ (ec[i, L] ** 2 * inv)
            S["raw_hpa"][i, L], S["raw_sig"][i, L] = w @ er[i, L] ** 2, w @ (er[i, L] ** 2 * inv)
    S["starts"] = pd.DatetimeIndex(starts)
    return S


def _r3(x):
    return [round(float(v), 3) if np.isfinite(v) else None for v in np.atleast_1d(x)]


def report(doc: dict):
    print(f"MSSS of the normalised anomaly against the ERA5 normal (gate: BH {doc['fdr_q']} and >= {doc['min_skill']})")
    for d in doc["models"].values():
        print(
            f"  {d['label']}: {d['starts']} starts {d['first']}..{d['last']}, {d['months']} months; shows days 0-{d['show_days'] - 1}"
        )
        for L in LEADS:
            dr = d["drift"]
            drift = (
                f"  drift rms {dr['rms_hpa'][L]:.2f} hPa ({dr['rms_sd'][L]:.2f} sd; fit {dr['rms_fit_hpa'][L]:.2f}, r {dr['rho'][L]:.2f})"
                if dr
                else ""
            )
            print(
                f"    day {L:2d}: {d['msss'][L]:+.3f} [{d['msss_ci'][0][L]:+.3f}, {d['msss_ci'][1][L]:+.3f}] p {d['p'][L]:.4f} "
                f"{'PASS' if d['passed'][L] else '-   '}  hPa {d['msss_hpa'][L]:+.3f}  no drift {d['msss_no_drift'][L]:+.3f}{drift}"
            )


def figures(doc: dict, fits: dict, corrected: dict, out_dir: Path):
    """Scratch figures for the check: skill by lead; the shrunk drift at days 0, 5 and 10 (annual mean)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(14, 4.6))
    for d in doc["models"].values():
        x = np.array(LEADS)
        y = np.array(d["msss"], float)
        col = axs[0].plot(x, y, lw=2, label=d["label"])[0].get_color()
        axs[0].fill_between(x, *(np.array(v, float) for v in d["msss_ci"]), color=col, alpha=0.15)
        axs[0].plot(x[np.array(d["passed"])], y[np.array(d["passed"])], "o", color="k", ms=3)
        axs[0].plot(x, np.array(d["msss_no_drift"], float), lw=1, ls=":", color=col)
        if d["drift"]:
            axs[1].plot(x, d["drift"]["rms_hpa"], lw=2, color=col, label=d["label"])
            axs[1].plot(x, d["drift"]["rms_fit_hpa"], lw=1, ls=":", color=col)
    axs[0].axhline(MIN_SKILL, color="#888", lw=0.8)
    axs[0].set_title("MSSS (σ) against the ERA5 normal; dotted: no drift correction; dots pass", loc="left")
    axs[1].set_title("Drift rms over the domain (hPa): shrunk (solid), as fitted (dotted)", loc="left")
    for ax in axs:
        ax.legend(frameon=False)
        ax.grid(alpha=0.3)
        ax.set_xlabel("lead day")
    fig.tight_layout()
    fig.savefig(out_dir / "verify_skill.png", dpi=100)
    plt.close(fig)
    import cartopy.crs as ccrs

    kw = {"projection": ccrs.PlateCarree(central_longitude=180.0)}
    fig, axs = plt.subplots(
        len(fits), 3, figsize=(18, 2.0 * len(fits) + 0.6), squeeze=False, layout="constrained", subplot_kw=kw
    )
    for i, (m, fl) in enumerate(fits.items()):
        for j, L in enumerate((0, 5, 10)):
            d = eval_drift(fl[L], [1.0, 92.0, 183.0, 274.0]).mean(0).reshape(C.LAT.size, C.LON.size)
            ax = axs[i, j]
            im = ax.pcolormesh(C.LON, C.LAT, d, cmap="RdBu_r", vmin=-3, vmax=3, transform=ccrs.PlateCarree())
            ax.coastlines(linewidth=0.6)
            ax.set_title(f"{MODELS[m]['label']} drift, day {L} (hPa, mean of four seasons)", fontsize=9, loc="left")
    fig.colorbar(im, ax=axs, shrink=0.6)
    fig.savefig(out_dir / "drift_maps.png", dpi=90)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("truth", "archive"):
        s = sub.add_parser(name)
        s.add_argument("--since", default=str(O.EPOCH.date()))
        s.add_argument(
            "--until", default=str((pd.Timestamp.now("UTC").tz_localize(None) - pd.Timedelta(days=6)).date())
        )
        s.add_argument("--part", default="0/1")
        s.add_argument("--out", required=True)
        if name == "archive":
            s.add_argument("--model", required=True, choices=sorted(MODELS))
    s = sub.add_parser("pack")
    s.add_argument("parts", nargs="+")
    s = sub.add_parser("verify")
    s.add_argument("--out-dir", default="out")
    a = ap.parse_args()
    if a.cmd in ("truth", "archive"):
        k, n = (int(x) for x in a.part.split("/"))
        if a.cmd == "truth":  # every day the lattice's runs verify on, through their last lead
            until = pd.Timestamp(a.until) + pd.Timedelta(days=LEADS[-1])
            days = pd.date_range(O.lattice(a.since, a.until)[0], until)[k::n]
            print(f"ERA5 on {len(days)} days", flush=True)
            doc = era5_days(days)
        else:
            since = max(pd.Timestamp(a.since), pd.Timestamp(MODELS[a.model]["first"]))
            starts = O.lattice(since, a.until, (k, n))
            print(f"{MODELS[a.model]['label']} on {len(starts)} lattice starts", flush=True)
            doc = archive(a.model, starts)
        np.savez_compressed(a.out, **doc)
        return 0
    if a.cmd == "pack":
        doc = pack(a.parts)
        print({k: v.shape for k, v in doc.items()})
        return 0
    verify(Path(a.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
