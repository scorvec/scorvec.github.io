#!/usr/bin/env python3
"""Analysis of the conservation-and-drag intercomparison: reads ONLY the committed reduced diagnostics
(scripts/amdrag/data/**.npz) and writes scripts/amdrag/data/results.json, which build_page.py and figures.py read.

Conventions: torques in Hadley (1e18 N m, positive = adds westerly AAM to the atmosphere), AAM in 1e25 kg m2 s-1,
pressure in hPa. "Lead d" = forecast day d (hours 24(d-1)..24d for torques, the state at 24d h for AAM).

Statistics (site rule: nothing insignificant presented as a finding): every mean over inits carries a moving-block
bootstrap CI (blocks of 3 consecutive inits, 2,000 draws, percentile 95 %) and a centred-bootstrap two-sided p;
model differences are paired on the same inits; families of tests (leads x regions, leads x hemispheres) are
controlled with Benjamini-Hochberg at q = 0.10. Members are averaged within an init first, so the init is the unit.

    python scripts/amdrag/study.py
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C                                                     # noqa: E402

HU = 1e18
MU = 1e25
DAY = 86400.0
NB = C.NB
SH, NHs = slice(0, NB // 2), slice(NB // 2, NB)
EARTH = 4 * np.pi * C.A ** 2
RNG = np.random.default_rng(20261013)
NBOOT = 2000
BLOCK = 3
Q_FDR = 0.10
# key, label, centre, kind, data dir, own-friction terms
MODELS = [
    ("gfs", "GFS", "NOAA/NCEP", "physics", "gfs", ("Tft", "Tg")),
    ("gefs", "GEFS", "NOAA/NCEP", "physics", "gefs", ("Tft", "Tg")),
    ("aigfs", "AIGFS", "NOAA/NCEP", "AI", "aigfs", ()),
    ("ifs-ens", "IFS-ENS", "ECMWF", "physics", "ifs-ens", ("Tft",)),
    ("aifs-single", "AIFS", "ECMWF", "AI", "aifs-single", ()),
    ("aifs-ens", "AIFS-ENS", "ECMWF", "AI", "aifs-ens", ()),
    ("gdps", "GDPS", "ECCC", "hybrid", "gem/gdps", ()),
    ("geps", "GEPS", "ECCC", "physics", "gem/geps", ()),
]
LABEL = {m[0]: m[1] for m in MODELS} | {"era5": "ERA5"}
KIND = {m[0]: m[3] for m in MODELS}
OWN = {m[0]: m[5] for m in MODELS} | {"era5": ("Tft", "Tg")}
REGION_NAMES = {"ORO": "rough terrain + 500 km downstream", "OCX": "ocean 30-65 deg", "OCO": "other ocean",
                "LND": "other land"}


# ---- loading ------------------------------------------------------------------------------------------------------
class Run:
    """One (model, init, member): global / hemispheric series and the arrays the analysis needs."""

    def __init__(self, z, run, scale_h=1.0):
        g = lambda k: z[f"{run}|{k}"] if f"{run}|{k}" in z.files else None      # noqa: E731
        self.Sd = (g("Ssteps") // 24).astype(int) if g("Ssteps") is not None else np.array([], int)
        self.Ld = (g("Lsteps") // 24).astype(int) if g("Lsteps") is not None else np.array([], int)
        self.S = {}
        for k in ("MO_b", "mass_b", "wv_b", "u10_b", "spd10_rh", "u10_rh"):
            x = g(f"S|{k}")
            if x is not None:
                self.S[k] = x.astype(np.float64)
        self.D = {}
        self.A = {}
        for k in z.files:
            if not k.startswith(run + "|"):
                continue
            rest = k[len(run) + 1:]
            if rest.startswith("D|") or rest.startswith("A|"):
                kk = rest[2:]
                v = z[k].astype(np.float64)
                if kk.startswith("Tm"):
                    v = v * scale_h
                (self.D if rest[0] == "D" else self.A)[kk] = v
        self.L = {}
        for k in ("Mr_b", "MrC11_b", "MrC12_b", "MrC13_b", "MrC14_b", "u850_rh", "u925_rh", "FC11", "FC14",
                  "MrC11_lb", "Mr_lb", "u850_b", "a850_b"):
            x = g(f"L|{k}")
            if x is not None:
                self.L[k] = x.astype(np.float64)

    # helpers returning per-day arrays (NaN where absent), index = day
    def state(self, key, part="G", n=16):
        out = np.full(n, np.nan)
        src = self.S if key in self.S else self.L
        days = self.Sd if key in self.S else self.Ld
        if key not in src:
            return out
        x = src[key]
        v = x.sum(1) if part == "G" else (x[:, SH].sum(1) if part == "S" else x[:, NHs].sum(1))
        for i, d in enumerate(days):
            if d < n:
                out[d] = v[i]
        return out

    def torque(self, key, part="G", n=15):
        out = np.full(n, np.nan)
        if key not in self.D:
            return out
        x = self.D[key]
        v = x.sum(1) if part == "G" else (x[:, SH].sum(1) if part == "S" else x[:, NHs].sum(1))
        m = min(n, len(v))
        out[:m] = v[:m]
        return out


def load_model(key, ddir):
    """{date: {member: Run}} for one model; GEPS files reduced before the decametre fix are rescaled (T_m is linear
    in h)."""
    out, metas = {}, {}
    for f in sorted((C.DATA / ddir).glob("*.npz")):
        z = np.load(f)
        meta = json.loads(str(z["meta"]))
        runs = meta.get("runs") or [key]
        oro = meta.get("oro", {})
        scale = 10.0 if (oro.get("area_mean_m", 230) < 33 and not oro.get("decametres_fixed")) else 1.0
        d = dt.datetime.strptime(f.stem, "%Y%m%d").date()
        out[d] = {}
        for r in runs:
            mem = int(r.split(":")[1]) if ":" in r else 0
            out[d][mem] = Run(z, r, scale)
        metas[d] = meta
    return out, metas


def load_era5():
    """{date: Run} of ERA5 days."""
    out = {}
    for f in sorted((C.DATA / "era5").glob("*.npz")):
        z = np.load(f)
        out[dt.datetime.strptime(f.stem, "%Y%m%d").date()] = Run(z, "era5")
    return out


# ---- statistics ----------------------------------------------------------------------------------------------------
def block_boot(vals: np.ndarray, n=NBOOT, block=BLOCK):
    """Moving-block bootstrap of the mean over inits (vals in init order, NaN dropped).
    -> dict(mean, lo, hi, p, n)."""
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    N = len(v)
    if N < 4:
        return {"mean": float(np.mean(v)) if N else np.nan, "lo": np.nan, "hi": np.nan, "p": np.nan, "n": N}
    b = min(block, N)
    nb = int(np.ceil(N / b))
    starts = RNG.integers(0, N - b + 1, size=(n, nb))
    idx = (starts[:, :, None] + np.arange(b)[None, None, :]).reshape(n, -1)[:, :N]
    bm = v[idx].mean(1)
    m = float(v.mean())
    lo, hi = np.percentile(bm, [2.5, 97.5])
    p = float(min(1.0, (np.abs(bm - m) >= abs(m)).mean() + 1.0 / n))
    return {"mean": m, "lo": float(lo), "hi": float(hi), "p": p, "n": N}


def fdr(ps, q=Q_FDR):
    """Benjamini-Hochberg: boolean list, True = significant."""
    ps = np.asarray([np.nan if p is None else p for p in ps], float)
    ok = np.isfinite(ps)
    out = np.zeros(len(ps), bool)
    if not ok.any():
        return out.tolist()
    pv = ps[ok]
    o = np.argsort(pv)
    m = len(pv)
    thr = q * (np.arange(1, m + 1) / m)
    passed = pv[o] <= thr
    k = np.where(passed)[0].max() + 1 if passed.any() else 0
    sig = np.zeros(m, bool)
    sig[o[:k]] = True
    out[np.where(ok)[0]] = sig
    return out.tolist()


def mark(results: list[dict]):
    """Add 'sig' (BH-FDR within this family) to a list of bootstrap dicts."""
    s = fdr([r.get("p") for r in results])
    for r, x in zip(results, s):
        r["sig"] = bool(x)
    return results


# ---- the budget pieces ---------------------------------------------------------------------------------------------
def total_aam(run: Run, level_set="Mr_b", part="G", n=16):
    """M = wind term (level set) + mass term with the mass-drift part removed: the change of the Omega term caused by
    a change of total mass (2/3 Omega a^2 dm) is not produced by any torque, so it is taken out before the budget."""
    mr = run.state(level_set, part, n)
    mo = run.state("MO_b", part, n)
    ms = run.state("mass_b", part, n)
    corr = (2.0 / 3.0) * C.OMEGA * C.A ** 2 * (ms - ms[0])
    return mr + mo - corr, mr, mo


def own_friction(run: Run, model, part="G", n=15):
    terms = OWN.get(model, ())
    if not terms:
        return None
    tot = np.zeros(n)
    for t in terms:
        x = run.torque(f"{t}_b", part, n)
        tot = tot + x
    return tot


def budget(model, data, n=15):
    """Global budget residuals per (init, member, day): dM/dt - torques, with the model's own friction (where it
    publishes one) and with the common bulk friction. -> per-init arrays (n_init, n) of member means."""
    inits = sorted(data)
    res_own, res_bulk, dmdt, tq_own, tq_bulk = [], [], [], [], []
    for d in inits:
        ro, rb, dm, to, tb = [], [], [], [], []
        for m, r in data[d].items():
            M, _, _ = total_aam(r, "Mr_b", "G", n + 1)
            dM = np.diff(M) / DAY / HU
            tm = r.torque("Tm_b", "G", n) / HU
            fb = r.torque("Tfb_b", "G", n) / HU
            fo = own_friction(r, model, "G", n)
            dm.append(dM)
            rb.append(dM - (tm + fb))
            tb.append(tm + fb)
            if fo is not None:
                ro.append(dM - (tm + fo / HU))
                to.append(tm + fo / HU)
        res_bulk.append(np.nanmean(rb, 0))
        dmdt.append(np.nanmean(dm, 0))
        tq_bulk.append(np.nanmean(tb, 0))
        if ro:
            res_own.append(np.nanmean(ro, 0))
            tq_own.append(np.nanmean(to, 0))
    out = {"inits": inits, "bulk": np.array(res_bulk), "dmdt": np.array(dmdt), "tq_bulk": np.array(tq_bulk)}
    if res_own:
        out["own"] = np.array(res_own)
        out["tq_own"] = np.array(tq_own)
    # pooled (init x member) samples for the explained fraction
    pooled = {"dm": [], "own": [], "bulk": []}
    for d in inits:
        for m, r in data[d].items():
            M, _, _ = total_aam(r, "Mr_b", "G", n + 1)
            dM = np.diff(M) / DAY / HU
            tm = r.torque("Tm_b", "G", n) / HU
            pooled["dm"].append(dM)
            pooled["bulk"].append(dM - tm - r.torque("Tfb_b", "G", n) / HU)
            fo = own_friction(r, model, "G", n)
            if fo is not None:
                pooled["own"].append(dM - tm - fo / HU)
    out["pooled"] = {k: np.array(v) for k, v in pooled.items() if v}
    return out


def explained(dm, res):
    """1 - var(residual) / var(dM/dt) per lead over pooled samples (descriptive)."""
    ef = []
    for j in range(dm.shape[1]):
        a, b = dm[:, j], res[:, j]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 5:
            ef.append(np.nan)
            continue
        ef.append(float(1 - np.var(b[ok]) / np.var(a[ok])))
    return ef


def era5_budget(E):
    days = sorted(E)
    rows = []
    for d in days:
        n1 = d + dt.timedelta(days=1)
        if n1 not in E:
            continue
        r0, r1 = E[d], E[n1]
        M0 = r0.state("Mr_b", "G", 1)[0] + r0.state("MO_b", "G", 1)[0]
        M1 = r1.state("Mr_b", "G", 1)[0] + r1.state("MO_b", "G", 1)[0]
        dm0 = r0.state("mass_b", "G", 1)[0]
        dm1 = r1.state("mass_b", "G", 1)[0]
        dM = (M1 - M0 - (2 / 3) * C.OMEGA * C.A ** 2 * (dm1 - dm0)) / DAY / HU
        tm = r0.torque("Tm_b", "G", 1)[0] / HU
        tf = (r0.torque("Tft_b", "G", 1)[0] + r0.torque("Tg_b", "G", 1)[0]) / HU
        tb = r0.torque("Tfb_b", "G", 1)[0] / HU
        rows.append((d, dM, tm, tf, tb))
    return rows


# ---- references: own analysis and ERA5 -----------------------------------------------------------------------------
def analysis_state(data, date, key, part="G"):
    """The model's own analysis (control member, step 0) of a state variable at `date`, or NaN."""
    if date not in data or 0 not in data[date]:
        return np.nan
    r = data[date][0]
    x = r.state(key, part, 1)[0] if key in r.S or key in r.L else np.nan
    return x


def analysis_rh(data, date, key):
    if date not in data or 0 not in data[date]:
        return None
    r = data[date][0]
    src = r.S if key in r.S else r.L
    if key not in src:
        return None
    days = r.Sd if key in r.S else r.Ld
    i = np.where(days == 0)[0]
    return src[key][i[0]] if len(i) else None


def drift_vs_analysis(data, key="MrC11_b", part="G", leads=range(1, 16)):
    """Per init: member-mean of [forecast(d) - own analysis valid at init + d] for each lead. (n_init, len(leads))."""
    inits = sorted(data)
    out = np.full((len(inits), len(list(leads))), np.nan)
    for i, d in enumerate(inits):
        for j, L in enumerate(leads):
            ref = analysis_state(data, d + dt.timedelta(days=L), key, part)
            if not np.isfinite(ref):
                continue
            vals = [r.state(key, part, 16)[L] for r in data[d].values()]
            vals = [v for v in vals if np.isfinite(v)]
            if vals:
                out[i, j] = np.mean(vals) - ref
    return inits, out


def drift_vs_era5(data, E, key="MrC11_b", part="G", leads=range(1, 16)):
    """[fc(d) - ERA5(init+d)] - [fc(0) - ERA5(init)]: the error growth relative to the initial offset."""
    inits = sorted(data)
    out = np.full((len(inits), len(list(leads))), np.nan)
    for i, d in enumerate(inits):
        if d not in E:
            continue
        e0 = E[d].state(key, part, 1)[0]
        for j, L in enumerate(leads):
            v = d + dt.timedelta(days=L)
            if v not in E:
                continue
            eL = E[v].state(key, part, 1)[0]
            vals = []
            for r in data[d].values():
                s = r.state(key, part, 16)
                if np.isfinite(s[L]) and np.isfinite(s[0]):
                    vals.append((s[L] - eL) - (s[0] - e0))
            if vals:
                out[i, j] = np.mean(vals)
    return inits, out


def torque_bias(data, E, key, part="G", model=None, leads=range(1, 16)):
    """Forecast daily-mean torque on day L minus ERA5's daily-mean torque on the same valid day (Hadley)."""
    inits = sorted(data)
    out = np.full((len(inits), len(list(leads))), np.nan)
    for i, d in enumerate(inits):
        for j, L in enumerate(leads):
            v = d + dt.timedelta(days=L - 1)
            if v not in E:
                continue
            if key == "own":
                ef = E[v].torque("Tft_b", part, 1)[0] + E[v].torque("Tg_b", part, 1)[0]
                vals = [own_friction(r, model, part, 15)[L - 1] for r in data[d].values()
                        if own_friction(r, model, part, 15) is not None]
            else:
                ef = E[v].torque(key, part, 1)[0]
                vals = [r.torque(key, part, 15)[L - 1] for r in data[d].values()]
            vals = [x for x in vals if np.isfinite(x)]
            if vals and np.isfinite(ef):
                out[i, j] = (np.mean(vals) - ef) / HU
    return inits, out


def wind_drift(data, key, leads, ref="own", E=None):
    """(n_init, len(leads), 8) member-mean region drift of a region-mean wind vs the own analysis (or ERA5, minus the
    initial offset)."""
    inits = sorted(data)
    out = np.full((len(inits), len(leads), 8), np.nan)
    for i, d in enumerate(inits):
        for j, L in enumerate(leads):
            v = d + dt.timedelta(days=L)
            if ref == "own":
                a = analysis_rh(data, v, key)
                if a is None:
                    continue
                vals = []
                for r in data[d].values():
                    src = r.S if key in r.S else r.L
                    days = r.Sd if key in r.S else r.Ld
                    k = np.where(days == L)[0]
                    if key in src and len(k):
                        vals.append(src[key][k[0]] - a)
            else:
                if E is None or v not in E or d not in E:
                    continue
                aL, a0 = analysis_rh({v: {0: E[v]}}, v, key), analysis_rh({d: {0: E[d]}}, d, key)
                if aL is None or a0 is None:
                    continue
                vals = []
                for r in data[d].values():
                    src = r.S if key in r.S else r.L
                    days = r.Sd if key in r.S else r.Ld
                    k = np.where(days == L)[0]
                    k0 = np.where(days == 0)[0]
                    if key in src and len(k) and len(k0):
                        vals.append((src[key][k[0]] - aL) - (src[key][k0[0]] - a0))
            if vals:
                out[i, j] = np.nanmean(vals, 0)
    return inits, out


# ---- the study -----------------------------------------------------------------------------------------------------
def summarise(arr, leads):
    return [block_boot(arr[:, j]) | {"lead": int(L)} for j, L in enumerate(leads)]


def main():
    data, metas = {}, {}
    for key, *_rest in MODELS:
        ddir = _rest[3]
        if (C.DATA / ddir).exists():
            data[key], metas[key] = load_model(key, ddir)
    E = load_era5()
    R = {"made": dt.datetime.utcnow().strftime("%Y-%m-%d %H:%MZ"), "models": {}, "era5": {}}
    window = [d for d in E if d >= dt.date(2026, 9, 1)]
    R["era5"]["days"] = [str(min(window)), str(max(window)), len(window)] if window else None
    leads15 = list(range(1, 16))
    for key, label, centre, kind, ddir, own in MODELS:
        if key not in data or not data[key]:
            continue
        D = data[key]
        inits = sorted(D)
        nm = max(len(v) for v in D.values())
        M = {"label": label, "centre": centre, "kind": kind, "own_friction": list(own), "n_inits": len(inits),
             "first": str(inits[0]), "last": str(inits[-1]), "members": nm}
        # mass conservation: global-mean surface pressure drift (total and dry)
        mass, dry = [], []
        for d in inits:
            mm, dd_ = [], []
            for r in D[d].values():
                ms = r.state("mass_b", "G", 16)
                dp = C.G * (ms - ms[0]) / EARTH / 100.0
                mm.append(dp)
                wv = r.state("wv_b", "G", 16)
                if np.isfinite(wv[0]):
                    dd_.append(dp - C.G * (wv - wv[0]) / EARTH / 100.0)
            mass.append(np.nanmean(mm, 0))
            if dd_:
                dry.append(np.nanmean(dd_, 0))
        mass = np.array(mass)
        M["mass_drift_hPa"] = summarise(mass[:, 1:], leads15)
        M["mass_drift_spread_hPa"] = [float(np.nanstd(mass[:, j])) for j in range(1, 16)]
        if dry:
            dry = np.array(dry)
            M["dry_mass_drift_hPa"] = summarise(dry[:, 1:], leads15)
        # budget closure
        B = budget(key, D)
        M["budget_bulk"] = summarise(B["bulk"], leads15)
        M["explained_bulk"] = explained(B["pooled"]["dm"], B["pooled"]["bulk"])
        M["rms_resid_bulk"] = [float(np.sqrt(np.nanmean(B["pooled"]["bulk"][:, j] ** 2))) for j in range(15)]
        if "own" in B:
            M["budget_own"] = summarise(B["own"], leads15)
            M["explained_own"] = explained(B["pooled"]["dm"], B["pooled"]["own"])
            M["rms_resid_own"] = [float(np.sqrt(np.nanmean(B["pooled"]["own"][:, j] ** 2))) for j in range(15)]
        M["sd_dmdt"] = [float(np.nanstd(B["pooled"]["dm"][:, j])) for j in range(15)]
        mark(M["budget_bulk"])
        if "own" in B:
            mark(M["budget_own"])
        mark(M["mass_drift_hPa"])
        if dry is not None and len(dry):
            mark(M["dry_mass_drift_hPa"])
        # AAM drift vs own analysis (C11 and own levels) and vs ERA5, by hemisphere
        M["drift_own"] = {}
        M["drift_era5"] = {}
        for part in ("G", "N", "S"):
            for lev in ("MrC11_b", "Mr_b", "MO_b"):
                _, a = drift_vs_analysis(D, lev, part, leads15)
                M["drift_own"][f"{lev}|{part}"] = summarise(a / MU, leads15)
            for lev in ("MrC11_b", "MO_b"):
                _, b = drift_vs_era5(D, E, lev, part, leads15)
                M["drift_era5"][f"{lev}|{part}"] = summarise(b / MU, leads15)
        fam = []
        for k, v in M["drift_own"].items():
            if k.startswith("MrC11"):
                fam += v
        mark(fam)
        fam = []
        for k, v in M["drift_era5"].items():
            if k.startswith("MrC11"):
                fam += v
        mark(fam)
        for k, v in M["drift_own"].items():
            if not k.startswith("MrC11"):
                mark(v)
        for k, v in M["drift_era5"].items():
            if not k.startswith("MrC11"):
                mark(v)
        # torques against ERA5 on the same valid days
        M["torque_bias"] = {}
        M["torque_bias_agg"] = {}
        agg_fam = []
        for tk in ("Tm_b", "Tm05_b", "Tfb_b") + (("own",) if own else ()):
            for part in ("G", "N", "S"):
                _, a = torque_bias(D, E, tk, part, key, leads15)
                M["torque_bias"][f"{tk}|{part}"] = summarise(a, leads15)
                mark(M["torque_bias"][f"{tk}|{part}"])
                ag = {}
                for lname, (l0, l1) in (("d1", (1, 1)), ("d2_5", (2, 5)), ("d6_10", (6, 10)), ("d11_15", (11, 15))):
                    with np.errstate(all="ignore"):
                        x = np.nanmean(a[:, l0 - 1:l1], 1)
                    ag[lname] = block_boot(x) | {"term": tk, "part": part, "leads": lname}
                    agg_fam.append(ag[lname])
                M["torque_bias_agg"][f"{tk}|{part}"] = ag
        mark(agg_fam)
        # mean torques (descriptive): day 1 and days 6-10, global / NH / SH, plus regions and ranges
        mt = {}
        for tk in ("Tm_b", "Tm05_b", "Tm15_b", "Tfb_b", "Tfb_land_b", "Tft_b", "Tg_b", "Tft_land_b", "Tg_land_b"):
            for part in ("G", "N", "S"):
                arr = np.array([np.nanmean(np.stack([r.torque(tk, part, 15) for r in D[d].values()]), 0)
                                for d in inits]) / HU
                if np.isfinite(arr).any():
                    mt[f"{tk}|{part}"] = {"d1": block_boot(arr[:, 0]), "d6_10": block_boot(np.nanmean(arr[:, 5:10], 1)),
                                          "all": float(np.nanmean(arr))}
        M["torques"] = mt
        rr = {}
        for tk in ("Tm_r", "Tm05_r", "Tm15_r", "Tfb_r"):
            arr = []
            for d in inits:
                x = [r.D[tk][:10] for r in D[d].values() if tk in r.D]
                if x:
                    arr.append(np.nanmean(np.stack(x), 0))
            if arr:
                a = np.stack(arr) / HU                                   # (init, day, range)
                rr[tk] = {"mean": np.nanmean(a, (0, 1)).tolist(), "sd_daily": np.nanstd(a, (0, 1)).tolist(),
                          "boot": [block_boot(np.nanmean(a[:, :, k], 1)) for k in range(a.shape[2])]}
        # fine-scale (native minus 0.5-degree) mountain torque per range, its bootstrap and FDR over ranges
        if "Tm_r" in rr and "Tm05_r" in rr:
            arr = []
            for d in inits:
                x = [r.D["Tm_r"][:10] - r.D["Tm05_r"][:10] for r in D[d].values() if "Tm_r" in r.D]
                if x:
                    arr.append(np.nanmean(np.stack(x), 0))
            a = np.stack(arr) / HU
            rr["fine05"] = mark([block_boot(np.nanmean(a[:, :, k], 1)) for k in range(a.shape[2])])
        M["ranges"] = rr
        rh = {}
        for tk in ("Tfb_rh", "Tm_rh", "Tft_rh", "Tg_rh"):
            arr = []
            for d in inits:
                x = [r.D[tk][:10] for r in D[d].values() if tk in r.D]
                if x:
                    arr.append(np.nanmean(np.stack(x), 0))
            if arr:
                a = np.stack(arr) / HU
                rh[tk] = {"d1": np.nanmean(a[:, 0], 0).tolist(), "d1_10": np.nanmean(a, (0, 1)).tolist()}
        M["regions"] = rh
        # low-level winds vs own analysis: 10 m speed and u, 850 / 925 hPa u, by region, days 5 and 10 (15)
        W = {}
        for wk in ("spd10_rh", "u10_rh", "u850_rh", "u925_rh"):
            for ref in ("own", "era5"):
                leads = [1, 5, 10, 15]
                _, a = wind_drift(D, wk, leads, ref, E)
                res = []
                for j, L in enumerate(leads):
                    for k, rk in enumerate(C.RHKEYS):
                        res.append(block_boot(a[:, j, k]) | {"lead": L, "region": rk})
                mark(res)
                W[f"{wk}|{ref}"] = res
        M["winds"] = W
        # initial (analysis) region winds, for the sign of "stronger / weaker"
        a0 = [analysis_rh(D, d, "spd10_rh") for d in inits]
        a0 = [x for x in a0 if x is not None]
        u0 = [analysis_rh(D, d, "u10_rh") for d in inits]
        u0 = [x for x in u0 if x is not None]
        u8 = [analysis_rh(D, d, "u850_rh") for d in inits]
        u8 = [x for x in u8 if x is not None]
        M["analysis_winds"] = {"spd10": np.nanmean(a0, 0).tolist() if a0 else None,
                               "u10": np.nanmean(u0, 0).tolist() if u0 else None,
                               "u850": np.nanmean(u8, 0).tolist() if u8 else None}
        # vertical sampling: own levels minus the 11 common levels (analysis, day 0)
        dl = [r.state("Mr_b", "G", 1)[0] - r.state("MrC11_b", "G", 1)[0] for d in inits for r in [D[d].get(0)] if r]
        M["levels_minus_C11"] = float(np.nanmean(dl) / MU) if dl else None
        # per-level drift at day 10 vs own analysis (C11 levels x hemisphere)
        lbx = []
        for d in inits:
            v = d + dt.timedelta(days=10)
            if v not in D or 0 not in D[v] or "MrC11_lb" not in D[v][0].L:
                continue
            ref = D[v][0].L["MrC11_lb"][0]
            xs = [r.L["MrC11_lb"][C.LB_STEPS.index(240)] for r in D[d].values()
                  if "MrC11_lb" in r.L and np.isfinite(r.L["MrC11_lb"][C.LB_STEPS.index(240)]).all()]
            if xs:
                lbx.append(np.mean(xs, 0) - ref)
        if lbx:
            a = np.stack(lbx) / MU                                       # (init, 11 levels, 36 bands)
            prof = []
            for k, lev in enumerate(C.C11):
                for part, sl in (("S", SH), ("N", NHs)):
                    prof.append(block_boot(a[:, k, sl].sum(1)) | {"level": lev, "part": part})
            M["level_drift_d10"] = mark(prof)
        # orography scales (static)
        sc = next((m["oro"].get("scales") for m in metas[key].values() if m.get("oro", {}).get("scales")), None)
        M["orography"] = {"scales": sc, "grid": [metas[key][inits[0]]["oro"].get("nlat"),
                                                 metas[key][inits[0]]["oro"].get("nlon")],
                          "scaled_x10": bool(any(m.get("oro", {}).get("area_mean_m", 230) < 33 for m in metas[key].values()))}
        R["models"][key] = M
        print(f"{label}: {len(inits)} inits {inits[0]}..{inits[-1]}, {nm} members", flush=True)
    # ERA5 references
    rows = era5_budget(E)
    if len(rows) > 3:
        dM = np.array([r[1] for r in rows]); tm = np.array([r[2] for r in rows])
        tf = np.array([r[3] for r in rows]); tb = np.array([r[4] for r in rows])
        R["era5"]["budget"] = {"n_days": len(rows), "resid_own": block_boot(dM - tm - tf),
                               "resid_bulk": block_boot(dM - tm - tb),
                               "explained_own": float(1 - np.var(dM - tm - tf) / np.var(dM)),
                               "explained_bulk": float(1 - np.var(dM - tm - tb) / np.var(dM)),
                               "mean_Tm": float(tm.mean()), "mean_Tf_own": float(tf.mean()),
                               "mean_Tfb": float(tb.mean()), "sd_dmdt": float(dM.std())}
    if E:
        d37 = np.array([E[d].state("Mr_b", "G", 1)[0] - E[d].state("MrC11_b", "G", 1)[0] for d in sorted(E)]) / MU
        dd = np.diff(np.array([E[d].state("Mr_b", "G", 1)[0] for d in sorted(E)])) / DAY / HU
        dc = np.diff(np.array([E[d].state("MrC11_b", "G", 1)[0] for d in sorted(E)])) / DAY / HU
        R["era5"]["levels"] = {"mean_37_minus_C11": float(np.nanmean(d37)), "sd": float(np.nanstd(d37)),
                               "sd_daily_tendency_diff_H": float(np.nanstd(dd - dc)),
                               "corr_tendency": float(np.corrcoef(dd[np.isfinite(dd - dc)], dc[np.isfinite(dd - dc)])[0, 1])}
        tmv = {k: np.array([E[d].torque(k, "G", 1)[0] for d in sorted(E)]) / HU for k in ("Tm_b", "Tm05_b", "Tm15_b")}
        R["era5"]["mountain_resolution"] = {k: {"mean": float(np.nanmean(v)), "sd": float(np.nanstd(v))} for k, v in tmv.items()}
        rg = {}
        for tk in ("Tm_r", "Tm05_r", "Tm15_r", "Tfb_r", "Tft_r", "Tg_r"):
            a = np.stack([E[d].D[tk][0] for d in sorted(E) if tk in E[d].D]) / HU
            rg[tk] = {"mean": a.mean(0).tolist(), "sd_daily": a.std(0).tolist()}
        R["era5"]["ranges"] = rg
        win = [d for d in sorted(E) if d >= dt.date(2026, 9, 14)]
        mt = {}
        for tk in ("Tm_b", "Tm05_b", "Tfb_b", "Tft_b", "Tg_b", "Tft_land_b", "Tg_land_b"):
            for part in ("G", "N", "S"):
                v = np.array([E[d].torque(tk, part, 1)[0] for d in win]) / HU
                mt[f"{tk}|{part}"] = block_boot(v)
        R["era5"]["torques"] = mt
        rh = {}
        for tk in ("Tfb_rh", "Tm_rh", "Tft_rh", "Tg_rh"):
            a = np.stack([E[d].D[tk][0] for d in win if tk in E[d].D]) / HU
            rh[tk] = a.mean(0).tolist()
        R["era5"]["regions"] = rh
        import xarray as xr
        o = xr.open_dataarray(C.REF / "era5_orography.nc")
        g = C.Grid(o.latitude.values.astype(float), o.longitude.values.astype(float))
        R["era5"]["orography"] = {"scales": C.orography_stats(g, o.values.astype(float)), "grid": [721, 1440]}
    # day-1 WGNE-comparable land stress (Jan / Jul 2026)
    R["wgne"] = wgne_day1(E)
    # paired model differences of the AAM drift at day 10 (C11, global), on common inits
    R["paired"] = paired(data)
    # within GEM: GDPS (0.15 deg output, and averaged to 0.5) against GEPS (0.5 deg) mountain torque by range, days 1-10,
    # paired on common inits - the resolution / orography part of the 2019 changes
    R["gem_pair"] = gem_pair(data)
    out = C.DATA / "results.json"
    out.write_text(json.dumps(clean(R), indent=0, allow_nan=False))
    print("wrote", out)
    return R


def clean(x):
    """NaN / inf -> None and numpy scalars -> Python, recursively (strict JSON for the page builder)."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    return x


def gem_pair(data):
    if "gdps" not in data or "geps" not in data:
        return None
    common = sorted(set(data["gdps"]) & set(data["geps"]))
    out = {"n": len(common), "native": [], "at05": [], "global": {}}

    def rng(run, key):
        return run.D[key][:10] if key in run.D else None
    for which, gk in (("native", "Tm_r"), ("at05", "Tm05_r")):
        rows = []
        for d in common:
            a = rng(data["gdps"][d][0], gk)
            b = [rng(r, "Tm_r") for r in data["geps"][d].values()]
            b = [x for x in b if x is not None]
            if a is None or not b:
                continue
            rows.append(np.nanmean(a, 0) - np.nanmean(np.nanmean(np.stack(b), 0), 0))
        if rows:
            A = np.stack(rows) / HU
            out[which] = mark([block_boot(A[:, k]) | {"range": C.RKEYS[k]} for k in range(A.shape[1])])
    for which, gk in (("native", "Tm_b"), ("at05", "Tm05_b")):
        x = []
        for d in common:
            a = data["gdps"][d][0].torque(gk, "G", 10)
            b = np.nanmean([r.torque("Tm_b", "G", 10) for r in data["geps"][d].values()], 0)
            x.append(np.nanmean(a - b))
        out["global"][which] = block_boot(np.array(x) / HU)
    return out


def paired(data, lead=10):
    keys = [k for k in LABEL if k in data and k != "era5"]
    drift = {}
    for k in keys:
        inits, a = drift_vs_analysis(data[k], "MrC11_b", "G", [lead])
        drift[k] = {d: a[i, 0] / MU for i, d in enumerate(inits) if np.isfinite(a[i, 0])}
    res = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            common = sorted(set(drift[a]) & set(drift[b]))
            if len(common) < 4:
                continue
            x = np.array([drift[a][d] - drift[b][d] for d in common])
            res.append(block_boot(x) | {"a": a, "b": b})
    return mark(res)


def wgne_day1(E):
    """Zonal-mean surface torque per unit latitude, [tau_x] a^3 cos^2(phi) (N m, the WGNE drag project's Fig. 13
    quantity, sign = stress on the surface, positive in westerlies), first 24 h of each 00Z control forecast."""
    out = {}
    dphi = np.deg2rad(5.0)
    lat = -87.5 + 5 * np.arange(NB)
    for model, ddir in (("gfs", "gfs_d1"), ("gefs", "gefs_d1"), ("ifs-ens", "ifs-ens_d1"), ("era5", "era5")):
        for mon in (1, 7):
            rows = {"pbl_land": [], "sgo": [], "total": [], "pbl_water": []}
            if model == "era5":
                files = [(d, E[d]) for d in sorted(E) if d.year == 2026 and d.month == mon]
                runs = [r for _, r in files]
            else:
                p = C.DATA / ddir
                if not p.exists():
                    continue
                runs = []
                for f in sorted(p.glob(f"2026{mon:02d}*.npz")):
                    z = np.load(f)
                    meta = json.loads(str(z["meta"]))
                    runs.append(Run(z, (meta.get("runs") or [model])[0]))
            for r in runs:
                if "Tft_b" not in r.D:
                    continue
                tl = r.D["Tft_land_b"][0]
                tt = r.D["Tft_b"][0]
                tg = r.D["Tg_b"][0] if "Tg_b" in r.D else np.zeros(NB)
                rows["pbl_land"].append(-tl / (2 * np.pi * dphi))
                rows["pbl_water"].append(-(tt - tl) / (2 * np.pi * dphi))
                rows["sgo"].append(-tg / (2 * np.pi * dphi))
                rows["total"].append(-(tt + tg) / (2 * np.pi * dphi))
            if not rows["pbl_land"]:
                continue
            res = {"n": len(rows["pbl_land"]), "lat": lat.tolist()}
            for k, v in rows.items():
                a = np.stack(v) / 1e19
                m = a.mean(0)
                nh = (lat > 25) & (lat < 65)
                i = int(np.argmax(np.where(nh, m, -np.inf)))
                sh = (lat < -25) & (lat > -65)
                j = int(np.argmax(np.where(sh, m, -np.inf)))
                res[k] = {"zonal": m.tolist(), "nh_peak": float(m[i]), "nh_peak_lat": float(lat[i]),
                          "sd_nh_peak": float(a[:, i].std()), "sh_peak": float(m[j]), "sh_peak_lat": float(lat[j])}
            res["has_sgo"] = model in ("gfs", "gefs", "era5")
            out[f"{model}|{mon}"] = res
    return out


if __name__ == "__main__":
    main()
