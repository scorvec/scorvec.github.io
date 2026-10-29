#!/usr/bin/env python3
"""Teleconnections, days 1-15, for circulation.html#telecon15: the eleven 500 hPa teleconnection indices (telecon15.IDS)
of every member of the newest IFS-ENS, AIFS-ENS, GEFS and GEPS runs, each model's ensemble mean calibrated against its
own archived runs, and the chance of each index's positive and negative phase at the leads where that model has
demonstrated skill. An index CPC does not define in any of the forecast's months gets a figure saying so. No
download: each ensmaps job leaves its members' fields in the Actions cache (Capture, from ensmaps.py), and `run` reads
the newest run of each model there.

  fields       each member's instantaneous 500 hPa height at 00Z of each day of a 00Z run (12Z of a 12Z run), days
               1-15, remapped conservatively from the model's grid onto the patterns' 2.5 deg grid 20-90N
               (latlon.remap) in whole metres, as the verification database keeps its members (z500_acc.run_fields):
               not the maps' daily means, since the hindcast is read at one hour
  indices      telecon15.Ref: the GEFS page's projectors on the anomaly from ERA5 1991-2020 at that hour, in seasonal
               sd of the daily index; the ensemble mean is the mean of the members' indices (the index is linear),
               at a lead where at least MIN_SHARE of the members are complete
  calibration  per model, index and lead, from the hindcast table (telecon15.py hindcast: every 00Z run the 500 hPa
               database holds from z500_acc.SINCE, against ERA5 at 00Z), the runs started within WINDOW days of this
               run's day of the year in any year: the regression obs = a m + b of ERA5's index on the ensemble-mean
               index m, its residual sd s, and the correlation r with its one-sided p on the effective sample
               n (1 - r1 r2) / (1 + r1 r2) (Bretherton et al. 1999; r1, r2 the lag-1 autocorrelations of the forecast
               and observed series over consecutive daily starts), from MIN_RUNS runs - the GEFS page's fit
               (gefs_telecon.py, telecon_geps.fit) on daily starts and single days. A 12Z run takes the 00Z runs' fit
               at the same lead in hours.
  skill        a lead is skilful only when r >= SKILL_MIN, p <= P_MAX and it passes Benjamini-Hochberg at FDR_Q over
               every model x index x lead of the update; elsewhere the figures hatch it, no demonstrated skill, and
               give no probability
  probability  P(index >= +THR) and P(index <= -THR) from the regression's predictive Student t (cal_cdf: n_eff - 2
               degrees of freedom, s widened for the fit's own uncertainty), at the skilful leads
  blend        per index, the most models whose past runs can be tested leave-one-season-out at every lead (subset), from
               their newest common start in the run records: equal or skill weights per lead, chosen out of sample (the
               paired CRPS test; blend_fit), calibrated as a model is and gated on its cross-validated correlation in
               the same false-discovery family (DelSole 2007 on why equal weights are hard to beat)
  observed     ERA5's index at 00Z of the last TAIL_DAYS days, from the same table (ERA5T, about a week behind)
  record       each run's indices, kept as the runs arrive in one record per model (RECORD, the KEEP_DAYS newest days of
               runs; scripts/strat/run_record.py's layout: per index the members' mean and 10th/90th percentiles at each
               valid time, and every member of the newest run), so the run-to-run figure needs nothing but the record
  outputs      per index the members and means (<id>_plume.webp), the phase probabilities (<id>_prob.webp), the last
               SHOW runs' ensemble means against valid date (<id>_runs.webp) and the hindcast skill with each cell's
               sample (<id>_skill.webp) in FIG (the frames store); the numbers in JSON

    python scripts/ensmaps/ensmaps_telecon.py run --fields f/*.npz --ref telecon15_ref.npz --table tbl --site .
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[0] / "lib")]
import telecon15 as T
from latlon import G25N, remap, slot366

MODELS = {"ifs": "IFS-ENS", "aifs": "AIFS-ENS", "gefs": "GEFS", "geps": "GEPS"}  # ensmaps keys, the database's names
SHORT = {"nao": "NAO", "pna": "PNA", "epo": "EPO", "wpo": "WPO", "ea": "EA", "wp": "WP", "epnp": "EP/NP", "eawr": "EA/WR",
         "sca": "SCA", "tnh": "TNH", "pol": "POL"}  # fmt: skip
COLORS = {"ifs": "#b3372f", "aifs": "#7b4fb0", "gefs": "#2c5f8a", "geps": "#2e7d4f"}  # verification.html's centres
WINDOW = 45  # days of the year either side of the run's: the hindcast starts its calibration pools
MIN_RUNS = 30
SKILL_MIN = 0.25  # hindcast correlation below this = no demonstrated skill
P_MAX = 0.05  # ... and so is a correlation not significant at this level (one-sided, on the effective sample)
FDR_Q = 0.10  # ... or one that fails Benjamini-Hochberg at this rate over every model x index x lead of the update
THR = 0.5  # +-0.5 seasonal sd: the positive and negative phases
MIN_SHARE = 0.8  # the members complete at a lead for its ensemble mean
TAIL_DAYS = 21
FIG = "assets/telecon15/fig"
JSON_OUT = "assets/sst/data/telecon15.json"
RECORD = "assets/sst/data/runs/telecon15_{}.json"  # each model's runs, beside the stratosphere products' records
KEEP_DAYS = 10
SHOW = 8  # runs drawn over one another: four days of 00 and 12Z runs
NEG, POS, NEUTRAL = "#3f6f8f", "#b4541f", "#e4e2dc"  # the site's cool and warm tokens, its rule grey
NOT_SIG = "#dcd9d1"  # tested, not skilful (verify_figs.NOT_SIG)
BLEND_INK = "#3d3a35"  # the blend's line: none of the models' colours
NU_MIN = 2.0
QC_RUNS = 3


def log(*a):
    print(*a, file=sys.stderr, flush=True)


# ── the fields, from ensmaps.py ───────────────────────────────────────────────────────────────────────────────────────
class Capture:
    """Each member's 500 hPa height at the steps of telecon15.LEADS, on the patterns' grid. A failure is logged and ends
    the capture, never the maps: put and save catch everything."""

    def __init__(self, n: int):
        self.z = np.full((n, len(T.LEADS), *G25N.shape), np.nan, np.float32)
        self.row = {s: k for k, s in enumerate(T.LEADS)}
        self.error = ""

    def put(self, i: int, step: int, v: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> None:
        k = self.row.get(step)
        if k is None or self.error:
            return
        try:
            self.z[i, k] = remap(v, lat, lon, G25N)
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"
            log(f"::warning::teleconnection fields: {self.error}; none saved for this run")

    def save(self, path: Path, model: str, cycle: str, init: pd.Timestamp) -> bool:
        try:
            if self.error:
                return False
            ok = np.isfinite(self.z).all((2, 3))
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path,
                model=model,
                cycle=cycle,
                init=f"{init:%Y-%m-%dT%H}:00Z",
                lead_h=np.array(T.LEADS, np.int16),
                z=np.where(ok[..., None, None], np.round(np.nan_to_num(self.z)), 0).astype(np.int16),
                ok=ok,
            )
            log(f"  teleconnection fields: {path} ({path.stat().st_size / 1e6:.1f} MB, {int(ok.any(0).sum())} leads)")
            return True
        except Exception as e:  # noqa: BLE001
            log(f"::warning::teleconnection fields not saved: {type(e).__name__}: {e}")
            return False


def load(paths: list[Path]) -> list[dict]:
    """The newest run of each model among the files, in MODELS order."""
    runs = {}
    for p in paths:
        with np.load(p) as z:
            r = {k: z[k] for k in z.files}
        key = str(r["model"])
        r["key"], r["cycle"] = key, str(r["cycle"])
        r["init"] = pd.Timestamp(str(r["init"])).tz_localize(None)
        if key in MODELS and (key not in runs or r["cycle"] > runs[key]["cycle"]):
            runs[key] = r
    return [runs[k] for k in MODELS if k in runs]


def member_indices(ref: T.Ref, r: dict) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """(valid times, members' indices (member, lead, index; NaN where a member is incomplete), the ensemble mean (lead,
    index; NaN at a lead with fewer than MIN_SHARE of the members))."""
    t = pd.DatetimeIndex(r["init"] + pd.to_timedelta(r["lead_h"].astype(int), unit="h"))
    n, nl = r["ok"].shape
    z = r["z"].reshape(n * nl, *G25N.shape)
    tt = pd.DatetimeIndex(np.tile(t.values, n))
    v = ref.index(ref.anomaly(z, tt), tt).reshape(n, nl, -1)
    v[~r["ok"]] = np.nan
    enough = r["ok"].sum(0) >= np.ceil(MIN_SHARE * n)
    mean = np.full(v.shape[1:], np.nan)
    mean[enough] = np.nansum(v[:, enough], 0) / r["ok"].sum(0)[enough, None]
    mean[~np.isfinite(v).any(0)] = np.nan  # an index CPC does not define on the date
    return t, v, mean


# ── calibration (the GEFS and GEPS pages' fit, on daily starts) ───────────────────────────────────────────────────────
def lag1(x: np.ndarray, nxt: np.ndarray) -> float:
    """Correlation of x with its value at the next start (pairs where one exists), 0 when too few
    (telecon_geps.lag1)."""
    ok = nxt >= 0
    if ok.sum() < 10:
        return 0.0
    a, b = x[ok], x[nxt[ok]]
    good = np.isfinite(a) & np.isfinite(b)
    if good.sum() < 10 or np.std(a[good]) == 0 or np.std(b[good]) == 0:
        return 0.0
    return float(np.corrcoef(a[good], b[good])[0, 1])


def fit(f: np.ndarray, o: np.ndarray, nxt: np.ndarray) -> dict | None:
    """The regression o = a f + b of paired values (NaN = no pair; nxt as lag1 takes it), its residual sd s, the
    correlation r and its one-sided p on the effective sample; None below MIN_RUNS pairs (telecon_geps.fit)."""
    from scipy.stats import t as tdist

    ok = np.isfinite(f) & np.isfinite(o)
    if ok.sum() < MIN_RUNS:
        return None
    r1, r2 = lag1(np.where(ok, f, np.nan), nxt), lag1(np.where(ok, o, np.nan), nxt)
    f, o = f[ok], o[ok]
    r = float(np.corrcoef(f, o)[0, 1])
    a, b = np.polyfit(f, o, 1)
    n = int(ok.sum())
    neff = float(np.clip(n * (1 - r1 * r2) / (1 + r1 * r2), 3, n))
    tt = r * np.sqrt(max(neff - 2, 1) / max(1 - r * r, 1e-9))
    return dict(
        n=n,
        n_eff=neff,
        p=float(tdist.sf(tt, max(neff - 2, 1))),
        r=r,
        a=float(a),
        b=float(b),
        s=float(np.std(o - (a * f + b), ddof=2)),
        xbar=float(f.mean()),
        sxx=float(((f - f.mean()) ** 2).sum()),
    )


def cal_cdf(x, c: dict, m):
    """P(observed index <= x) for the ensemble mean m under the regression c: Student t on n_eff - 2 degrees of freedom
    centred on a m + b, scale s sqrt(1 + 1/n_eff + (m - xbar)^2 / (sxx n_eff / n)) - the fit's own uncertainty, counted
    on the effective sample (telecon_geps.cal_cdf)."""
    from scipy.stats import t as tdist

    m = np.asarray(m, np.float64)
    w = np.sqrt(1.0 + 1.0 / c["n_eff"] + (m - c["xbar"]) ** 2 / (c["sxx"] * c["n_eff"] / c["n"]))
    return tdist.cdf((np.asarray(x) - (c["a"] * m + c["b"])) / (c["s"] * w), max(c["n_eff"] - 2.0, 1.0))


def pairs(fc: pd.DataFrame, obs: pd.DataFrame, model: str) -> pd.DataFrame:
    """The model's hindcast rows with ERA5's index on each valid day (obs_<id>), where ERA5 has it."""
    f = fc[fc["model"] == model].copy()
    f["valid"] = f["init"] + pd.to_timedelta(f["lead_h"].astype(int), unit="h")
    o = obs.set_index("valid")[list(T.IDS)].add_prefix("obs_")
    return f.join(o, on="valid", how="inner")


def near(p: pd.DataFrame, t0: pd.Timestamp) -> pd.DataFrame:
    """The rows whose start is within WINDOW days of t0's day of the year, any year."""
    d = (slot366(pd.DatetimeIndex(p["init"])) - slot366(pd.DatetimeIndex([t0]))[0] + 183) % 366 - 183
    return p[np.abs(d) <= WINDOW]


def calibrate(fc: pd.DataFrame | None, obs: pd.DataFrame | None, model: str, t0: pd.Timestamp) -> dict:
    """{(index, lead_h): fit or None} of one model from the runs started within WINDOW days of t0's day of the year."""
    out = dict.fromkeys(((s, lead) for s in T.IDS for lead in T.LEADS), None)
    if fc is None or obs is None:
        return out
    p = pairs(fc, obs, model)
    if p.empty:
        return out
    for lead, g in near(p, t0).sort_values("init").groupby("lead_h"):
        pos = {t: k for k, t in enumerate(g["init"])}
        nxt = np.array([pos.get(t + pd.Timedelta(days=1), -1) for t in g["init"]])
        for s in T.IDS:
            out[s, int(lead)] = fit(g[s].to_numpy(np.float64), g[f"obs_{s}"].to_numpy(np.float64), nxt)
    return out


def raw(fc: pd.DataFrame, obs: pd.DataFrame, model: str, t0: pd.Timestamp) -> dict:
    """{(index, lead_h): (r, n)}: the plain correlation of the model's ensemble-mean index with ERA5's over the same
    starts as calibrate, before any calibration or gate, whatever their number."""
    out = {}
    p = pairs(fc, obs, model)
    for lead, g in near(p, t0).groupby("lead_h") if not p.empty else ():
        for s in T.IDS:
            f, o = g[s].to_numpy(np.float64), g[f"obs_{s}"].to_numpy(np.float64)
            ok = np.isfinite(f) & np.isfinite(o)
            out[s, int(lead)] = (float(np.corrcoef(f[ok], o[ok])[0, 1]) if ok.sum() > 2 else np.nan, int(ok.sum()))
    return out


def gate(cal: dict) -> None:
    """Mark every fitted cell of the update (model x index x lead) skilful or not: r >= SKILL_MIN, p <= P_MAX and
    Benjamini-Hochberg at FDR_Q over all of them."""
    from quantmet.stats import benjamini_hochberg

    cells = [c for per in cal.values() for c in per.values() if c]
    if not cells:
        return
    for c, k in zip(cells, benjamini_hochberg(np.array([c["p"] for c in cells]), FDR_Q), strict=True):
        c["fdr_ok"] = bool(k)
        c["skilful"] = bool(k and c["r"] >= SKILL_MIN and c["p"] <= P_MAX)


def skilful(c: dict | None) -> bool:
    return bool(c and c.get("skilful"))


# ── the blend ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def season(init, t0: pd.Timestamp) -> np.ndarray:
    """The season of each start: the year of the start moved so that t0's day of the year falls mid-year, so the
    starts around t0 never straddle two seasons (around 5 January, November and February fall together)."""
    shift = pd.Timedelta(days=int(slot366(pd.DatetimeIndex([t0]))[0]) - 183)
    return (pd.DatetimeIndex(init) - shift).year.to_numpy()


def crps_t(y: np.ndarray, mu: np.ndarray, scale: np.ndarray, nu: float) -> np.ndarray:
    """The continuous ranked probability score of a Student t predictive distribution (location mu, scale, nu > 1
    degrees of freedom) for the outcomes y (Jordan, Krueger and Lerch 2019, scoringRules' formula)."""
    from scipy.special import beta
    from scipy.stats import t as tdist

    z = (y - mu) / scale
    term = 2.0 * np.sqrt(nu) * beta(0.5, nu - 0.5) / ((nu - 1.0) * beta(0.5, nu / 2.0) ** 2)
    return scale * (z * (2.0 * tdist.cdf(z, nu) - 1.0) + 2.0 * tdist.pdf(z, nu) * (nu + z * z) / (nu - 1.0) - term)


def predictive(c: dict, x) -> tuple:
    """(location, scale, degrees of freedom) of cal_cdf's Student t for the predictor x under the fit c (at least
    NU_MIN degrees of freedom: the CRPS of a t needs more than one)."""
    x = np.asarray(x, np.float64)
    w = np.sqrt(1.0 + 1.0 / c["n_eff"] + (x - c["xbar"]) ** 2 / (c["sxx"] * c["n_eff"] / c["n"]))
    return c["a"] * x + c["b"], c["s"] * w, max(c["n_eff"] - 2.0, NU_MIN)


def paired(d: np.ndarray, nxt: np.ndarray) -> float:
    """One-sided p that the paired differences d (in start order; nxt as lag1 takes it) are below zero on average:
    the t-test on the effective sample n (1 - rho) / (1 + rho), rho their lag-1 autocorrelation."""
    from scipy.stats import t as tdist

    ok = np.isfinite(d)
    n = int(ok.sum())
    if n < MIN_RUNS or np.std(d[ok]) == 0:
        return 1.0
    rho = lag1(np.where(ok, d, np.nan), nxt)
    neff = float(np.clip(n * (1 - rho) / (1 + rho), 3, n))
    return float(tdist.cdf(d[ok].mean() / (d[ok].std(ddof=1) / np.sqrt(neff)), neff - 1))


def wide(fc: pd.DataFrame, obs: pd.DataFrame, sid: str, keys: list[str], t0: pd.Timestamp) -> pd.DataFrame:
    """(start, lead) x model: each model's ensemble-mean index and ERA5's (obs), for the starts within WINDOW days of
    t0's day of the year that ERA5 covers."""
    f = fc[fc["model"].isin([MODELS[k] for k in keys])]
    d = (slot366(pd.DatetimeIndex(f["init"])) - slot366(pd.DatetimeIndex([t0]))[0] + 183) % 366 - 183
    w = f[np.abs(d) <= WINDOW].pivot_table(index=["init", "lead_h"], columns="model", values=sid)
    w = w.rename(columns={MODELS[k]: k for k in keys})
    if w.empty:
        return w
    lead = pd.to_timedelta(w.index.get_level_values("lead_h").astype(int), unit="h")
    w["obs"] = obs.set_index("valid")[sid].reindex(w.index.get_level_values("init") + lead).to_numpy()
    return w[np.isfinite(w["obs"])]


def subset(w: pd.DataFrame, keys: list[str], t0: pd.Timestamp) -> list[str]:
    """The models the blend takes: the most of them (two at least) whose weights can be tested out of sample at every
    lead (blend_fit): a common sample at the last lead in two seasons or more, with MIN_RUNS starts outside each season
    to fit on. The current season is short at the long leads (ERA5 a week behind), so a model with one season besides
    it waits for a second. The sample only shrinks with lead, so the last lead decides; the larger sample breaks a
    tie."""
    import itertools

    last = max(T.LEADS)
    if w.empty or last not in w.index.get_level_values("lead_h"):
        return []
    end = w.xs(last, level="lead_h")
    for n in range(len(keys), 1, -1):
        found = []
        for sub in itertools.combinations([k for k in keys if k in end], n):
            g = end[list(sub)].dropna()
            seasons, count = np.unique(season(g.index, t0), return_counts=True)
            if len(seasons) >= 2 and (len(g) - count).min() >= MIN_RUNS:
                found.append((len(g), list(sub)))
        if found:
            return max(found)[1]
    return []


def blend_fit(w: pd.DataFrame, models: list[str], lead: int, t0: pd.Timestamp) -> dict | None:
    """The blend of the models at one lead: equal weights (the mean of their ensemble-mean indices) or skill weights
    (the regression of ERA5's index on theirs, fitted on this lead and the next ones either side), each calibrated as a
    single model is (fit) and scored out of sample, leave-one-season-out (each season predicted from fits on the others),
    against the other and against each model alone, on the same starts. Skill weights only where their cross-validated
    CRPS beats the equal weights' significantly (paired, one-sided, at P_MAX); the cell's r and p, which the gate reads,
    are the chosen blend's cross-validated ones."""
    if lead not in w.index.get_level_values("lead_h"):
        return None
    g = w.xs(lead, level="lead_h")[[*models, "obs"]].dropna().sort_index()
    sea = season(g.index, t0)
    if len(g) < MIN_RUNS or len(np.unique(sea)) < 2:
        return None
    near = np.isin(w.index.get_level_values("lead_h"), [lead - 24, lead, lead + 24])
    pool = w[near][[*models, "obs"]].dropna()
    psea = season(pool.index.get_level_values("init"), t0)
    pos = {t: k for k, t in enumerate(g.index)}
    nxt = np.array([pos.get(t + pd.Timedelta(days=1), -1) for t in g.index])
    f, o = g[models].to_numpy(np.float64), g["obs"].to_numpy(np.float64)

    def weights(train: np.ndarray) -> np.ndarray:
        p = pool[train]
        a = np.c_[np.ones(len(p)), p[models].to_numpy(np.float64)]
        return np.linalg.lstsq(a, p["obs"].to_numpy(np.float64), rcond=None)[0][1:]

    x = {"ew": f.mean(1), **{k: f[:, j] for j, k in enumerate(models)}}
    mu = {k: np.full(len(g), np.nan) for k in [*x, "sw"]}
    crps = {k: np.full(len(g), np.nan) for k in mu}
    for y in np.unique(sea):
        test, train = sea == y, sea != y
        for k, xk in (x | {"sw": f @ weights(psea != y)}).items():
            c = fit(np.where(train, xk, np.nan), np.where(train, o, np.nan), nxt)
            if c is None:
                return None
            m, sc, nu = predictive(c, xk[test])
            mu[k][test], crps[k][test] = m, crps_t(o[test], m, sc, nu)
    p_sw = paired(crps["sw"] - crps["ew"], nxt)
    kind = "sw" if p_sw <= P_MAX else "ew"
    best = min(models, key=lambda k: crps[k].mean())
    wt = weights(np.ones(len(pool), bool)) if kind == "sw" else np.full(len(models), 1.0 / len(models))
    c, cv = fit(f @ wt, o, nxt), fit(mu[kind], o, nxt)
    if c is None or cv is None:
        return None
    return c | {
        "r_in": c["r"], "r": cv["r"], "p": cv["p"], "kind": kind, "weights": dict(zip(models, wt.tolist(), strict=True)),
        "crps": {k: float(v.mean()) for k, v in crps.items()}, "p_sw": p_sw, "best": best,
        "p_best": paired(crps[kind] - crps[best], nxt), "seasons": [int(y) for y in np.unique(sea)],
    }  # fmt: skip


def blend(fc, obs, ref: T.Ref, runs: list[dict], rec: dict) -> dict:
    """Per index: the blend's models, its fit at each lead (blend_fit), and its forecast from the newest start the run
    records hold for all of them: the calibrated mean, its 10-90 % range and, at the skilful leads, the chance of each
    phase."""
    from scipy.stats import t as tdist

    keys = [r["key"] for r in runs if rec.get(r["key"])]
    t0 = max(r["init"] for r in runs)
    out = {}
    for sid in ref.ids:
        w = wide(fc, obs, sid, keys, t0) if fc is not None and obs is not None else pd.DataFrame()
        models = subset(w, keys, t0)
        e = {"models": models, "fit": {lead: blend_fit(w, models, lead, t0) if models else None for lead in T.LEADS}}
        held = [{x["init"] for x in rec[k] if sid in x} for k in models]
        if held and (common := set.intersection(*held)):
            init = max(common)
            t = pd.DatetimeIndex(pd.Timestamp(init) + pd.to_timedelta(list(T.LEADS), unit="h"))
            f = np.stack([
                pd.Series(_arr(x[sid]["mean"]), index=pd.DatetimeIndex(x["valid"])).reindex(t).to_numpy()
                for x in (next(x for x in rec[k] if x["init"] == init) for k in models)
            ])  # fmt: skip
            mean, lo, hi = (np.full(len(t), np.nan) for _ in range(3))
            prob = {}
            for j, lead in enumerate(T.LEADS):
                c = e["fit"][lead]
                if c is None or not np.isfinite(f[:, j]).all():
                    continue
                xj = float(np.dot([c["weights"][k] for k in models], f[:, j]))
                m, sc, nu = predictive(c, xj)
                mean[j], lo[j], hi[j] = m, m + sc * tdist.ppf(0.1, nu), m + sc * tdist.ppf(0.9, nu)
                prob[lead] = (xj, float(cal_cdf(-THR, c, xj)), float(1.0 - cal_cdf(THR, c, xj)))
            e |= {"init": pd.Timestamp(init), "valid": t, "mean": mean, "lo": lo, "hi": hi, "prob": prob,
                  "reach": np.isfinite(f).all(0)}  # fmt: skip
        out[sid] = e
    return out


# ── figures ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def _plt():
    sys.path.insert(0, str(HERE.parents[0] / "verify"))
    import matplotlib.pyplot as plt
    import verify_figs as VF

    VF.style()
    return plt, VF


def ink_on(face, ink: str) -> str:
    """White or the ink, whichever contrasts more with the cell's colour (WCAG relative luminance)."""
    from matplotlib.colors import to_rgb

    def lum(c) -> float:
        v = np.array(to_rgb(c))
        v = np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)
        return float(v @ [0.2126, 0.7152, 0.0722])

    f = lum(face)
    return "white" if 1.05 / (f + 0.05) > (f + 0.05) / (lum(ink) + 0.05) else ink


def run_label(r: dict) -> str:
    return f"{MODELS[r['key']]} · {r['ok'].shape[0]} members · {r['init']:%-d %b %HZ} run"


def date_ticks(ax, t0, t1, nc: int) -> None:
    """A tick every few days from t0, as many as a panel of an nc-column figure holds (a date label takes about
    0.62 in): a calendar locator restarts at each month and puts 31 Oct beside 1 Nov."""
    import matplotlib.dates as mdates

    step = max(int(np.ceil((pd.Timestamp(t1) - pd.Timestamp(t0)).days * 0.62 / (14.0 / nc))), 1)
    ax.set_xticks(pd.date_range(pd.Timestamp(t0).normalize(), t1, freq=f"{step}D"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%-d %b"))


def spans(t: pd.DatetimeIndex, bad: np.ndarray) -> list[tuple]:
    """The runs of consecutive leads marked bad, as (start, end) half a day either side."""
    out, k = [], 0
    while k < len(t):
        if bad[k]:
            j = k
            while j + 1 < len(t) and bad[j + 1]:
                j += 1
            out.append((t[k] - pd.Timedelta(hours=12), t[j] + pd.Timedelta(hours=12)))
            k = j + 1
        else:
            k += 1
    return out


def blend_label(b: dict) -> str:
    return f"Blend of {', '.join(MODELS[k] for k in b['models'])} · {b['init']:%-d %b %HZ} runs, calibrated"


def grid(n: int) -> tuple[int, int, float]:
    """Rows, columns and the height-to-width ratio of a figure of n panels."""
    return (1, n, 0.36) if n <= 2 else (2, 2, 0.62) if n <= 4 else (2, 3, 0.55)


def plume(sid: str, k: int, runs: list[dict], ind: dict, cal: dict, obs, b: dict, name: str, path: Path):
    """Every member's index (thin) and the ensemble mean, one panel per model, after ERA5's last weeks; the blend's
    calibrated mean and 10-90 % range in a panel of its own; the leads without demonstrated skill hatched."""
    plt, VF = _plt()
    import house
    from matplotlib.patches import Patch

    t_last = max(ind[r["key"]][0][-1] for r in runs)
    t0 = min(r["init"] for r in runs).normalize()
    vals = [ind[r["key"]][1][..., k] for r in runs]
    tail = pd.Series(dtype=float)
    if obs is not None:
        tail = obs[obs["valid"] >= t0 - pd.Timedelta(days=TAIL_DAYS)].set_index("valid")[sid].dropna()
        vals.append(tail.to_numpy())
    t_first = t0 - pd.Timedelta(days=TAIL_DAYS) if len(tail) else t0
    with_blend = bool(b.get("init") is not None and np.isfinite(b["mean"]).any())
    if with_blend:
        vals += [b["lo"], b["hi"]]
    top = max(2.5, 1.08 * _absmax(*vals))
    n = len(runs) + with_blend
    nr, nc, ratio = grid(n)
    fig, axes = plt.subplots(nr, nc, figsize=house.size(15, ratio), sharex=True, sharey=True, constrained_layout=True,
                             squeeze=False)  # fmt: skip
    for ax, r in zip(axes.ravel(), runs, strict=False):
        t, v, mean = ind[r["key"]]
        v, mean = v[..., k], mean[:, k]
        col = COLORS[r["key"]]
        ax.axhspan(-THR, THR, color=NEUTRAL, alpha=0.55, lw=0, zorder=0)
        ax.axhline(0, color=house.EDGE, lw=1.0, zorder=1)
        bad = np.array([not skilful(cal[r["key"]][sid, int(h)]) for h in r["lead_h"]])
        for a0, a1 in spans(t, bad):
            ax.axvspan(a0, a1, facecolor="none", edgecolor=house.EDGE, hatch="//", lw=0, zorder=0)
        ax.plot(t, v.T, color=col, lw=0.7, alpha=0.38, zorder=2)
        ax.plot(t, mean, color=col, lw=3.0, zorder=4)
        if len(tail):
            ax.plot(tail.index, tail.to_numpy(), color=house.INK, lw=2.0, marker="o", ms=3.5, zorder=5)
        ax.axvline(r["init"], color=house.MUTED, lw=1.0, ls=":", zorder=3)
        ax.set_title(run_label(r), color=house.INK)
        ax.grid(True, color=house.GRID, lw=0.8)
        ax.set_ylim(-top, top)
    if with_blend:
        ax = axes.flat[len(runs)]
        ax.axhspan(-THR, THR, color=NEUTRAL, alpha=0.55, lw=0, zorder=0)
        ax.axhline(0, color=house.EDGE, lw=1.0, zorder=1)
        bad = np.array([not skilful(cal["blend"][sid, h]) for h in T.LEADS])
        for a0, a1 in spans(b["valid"], bad):
            ax.axvspan(a0, a1, facecolor="none", edgecolor=house.EDGE, hatch="//", lw=0, zorder=0)
        ax.fill_between(b["valid"], b["lo"], b["hi"], color=BLEND_INK, alpha=0.14, lw=0, zorder=2)
        ax.plot(b["valid"], b["mean"], color=BLEND_INK, lw=3.0, zorder=4)
        if len(tail):
            ax.plot(tail.index, tail.to_numpy(), color=house.INK, lw=2.0, marker="o", ms=3.5, zorder=5)
        ax.axvline(b["init"], color=house.MUTED, lw=1.0, ls=":", zorder=3)
        ax.set_title(blend_label(b), color=house.INK)
        ax.grid(True, color=house.GRID, lw=0.8)
        ax.set_ylim(-top, top)
    for j in range(n, nr * nc):  # the panel above an empty one carries the dates
        axes.flat[j].set_axis_off()
        axes.flat[j - nc].tick_params(labelbottom=True)
    for ax in axes[:, 0]:
        ax.set_ylabel("Seasonal sd")
    date_ticks(axes[0, 0], t_first, t_last, nc)
    axes[0, 0].set_xlim(t_first, t_last + pd.Timedelta(hours=12))
    handles = [
        plt.Line2D([], [], color=house.INK2, lw=0.9, alpha=0.6, label="each member"),
        plt.Line2D([], [], color=house.INK2, lw=3.0, label="ensemble mean"),
        *[Patch(color=BLEND_INK, alpha=0.14, label="blend's calibrated 10–90 %")] * with_blend,
        *[plt.Line2D([], [], color=house.INK, lw=2.0, marker="o", ms=3.5, label="ERA5, 00Z")] * bool(len(tail)),
        Patch(facecolor=NEUTRAL, alpha=0.55, label=f"neutral, within ±{THR:g} sd"),
        Patch(facecolor="white", edgecolor=house.EDGE, hatch="//", label="no demonstrated skill"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncols=min(len(handles), 6), frameon=False)
    fig.suptitle(f"{name}: each member's index every day and the ensemble mean, days 1–15", x=0.01, ha="left")
    return VF.save(fig, path)


def prob(sid: str, k: int, runs: list[dict], ind: dict, probs: dict, cal: dict, b: dict, name: str, path: Path):
    """The calibrated chance of each phase by lead, one row per model and one for the blend, at the skilful leads only
    (a lead the run does not reach, a 12Z ECMWF run's day 15, left blank)."""
    plt, VF = _plt()
    import house
    from matplotlib.patches import Patch

    days = np.arange(1, len(T.LEADS) + 1)
    # (key, the leads its forecast reaches, its probabilities, label); a blend lead without a fit is one without skill
    rows = [(r["key"], np.isfinite(ind[r["key"]][2][:, k]), probs[r["key"]], f"{MODELS[r['key']]}\n{r['init']:%-d %b %HZ}")
            for r in runs]  # fmt: skip
    if b.get("init") is not None and np.isfinite(b["mean"]).any():
        pr = {(sid, h): v[1:] for h, v in b["prob"].items()}
        rows.append(("blend", b["reach"], pr, f"Blend\n{b['init']:%-d %b %HZ}"))
    fig, axes = plt.subplots(len(rows), 1, figsize=house.size(15, 0.13 + 0.115 * len(rows)), sharex=True,
                             constrained_layout=True, squeeze=False)  # fmt: skip
    for ax, (key, reach, pr, label) in zip(axes[:, 0], rows, strict=True):
        for d, h in zip(days, T.LEADS, strict=True):
            c, pp = cal[key][sid, h], pr.get((sid, h))
            if not reach[d - 1]:
                continue
            if not skilful(c) or pp is None:
                ax.bar(d, 1.0, width=0.86, facecolor="none", edgecolor=house.EDGE, hatch="//", lw=0)
                continue
            pn, pq = pp
            for y0, y1, col in ((0.0, pn, NEG), (pn, 1.0 - pq, NEUTRAL), (1.0 - pq, 1.0, POS)):
                ax.bar(d, y1 - y0, bottom=y0, width=0.86, color=col, edgecolor="white", lw=1.5)
            for y, p in ((pn / 2, pn), (1.0 - pq / 2, pq)):
                if p >= 0.12:
                    ax.text(d, y, f"{100 * p:.0f}", ha="center", va="center", color="white", fontweight="semibold")
        ax.set(ylim=(0, 1), yticks=[0, 0.5, 1], yticklabels=["0", "50", "100 %"], xlim=(0.4, len(days) + 0.6))
        ax.set_ylabel(label, rotation=0, ha="right", va="center", labelpad=12)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1, 0].set_xticks(days)
    axes[-1, 0].set_xlabel("Lead (days)")
    handles = [
        Patch(color=POS, label=f"positive, ≥ +{THR:g} sd"),
        Patch(color=NEUTRAL, label="neutral"),
        Patch(color=NEG, label=f"negative, ≤ −{THR:g} sd"),
        Patch(facecolor="white", edgecolor=house.EDGE, hatch="//", label="no demonstrated skill: no probability"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncols=4, frameon=False)
    fig.suptitle(f"{name}: calibrated chance of each phase by lead (%)", x=0.01, ha="left")
    return VF.save(fig, path)


def skill(sid: str, runs: list[dict], cal: dict, b: dict, name: str, t0: pd.Timestamp, path: Path):
    """The hindcast correlation of each model's ensemble-mean index with ERA5's by lead, and each cell's sample (runs):
    coloured where skilful, grey where tested and not, blank below MIN_RUNS runs; the blend's cross-validated
    correlation (SW where skill weights beat equal ones), and beneath it the change of its cross-validated CRPS from the
    best single model's and the skill weights' from the equal weights', in bold where the paired test finds it."""
    plt, VF = _plt()
    import house
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Patch

    cmap = LinearSegmentedColormap.from_list("r", ["#e3eaf3", house.ACCENT])
    days = np.arange(1, len(T.LEADS) + 1)
    rows = [(r["key"], MODELS[r["key"]]) for r in runs] + [("blend", "Blend")] * bool(b["models"])
    tests = [("best", "p_best", "CRPS vs best model"), ("sw", "p_sw", "Skill vs equal weights")] if b["models"] else []
    fig, ax = plt.subplots(figsize=house.size(15, 0.1 + 0.075 * (len(rows) + len(tests))), constrained_layout=True)
    for i, (key, _) in enumerate(rows):
        for d, h in zip(days, T.LEADS, strict=True):
            c = cal[key][sid, h]
            if c is None:
                face, text, ink = "white", "–", house.MUTED
            else:
                x = min(max((c["r"] - SKILL_MIN) / (1 - SKILL_MIN), 0.0), 1.0)
                face = cmap(x) if skilful(c) else NOT_SIG
                ink = ink_on(face, house.INK)
                text = f"{c['r']:.2f}\nn {c['n']}" + (" SW" if c.get("kind") == "sw" else "")
            ax.add_patch(plt.Rectangle((d - 0.5, i - 0.5), 1, 1, facecolor=face, edgecolor="white", lw=2))
            ax.text(d, i, text, ha="center", va="center", color=ink, fontsize=12.5, linespacing=1.25)
    for i, (which, pk, _) in enumerate(tests, start=len(rows)):
        for d, h in zip(days, T.LEADS, strict=True):
            c = cal["blend"][sid, h]
            if c is None:
                continue
            ref = c["crps"][c["best"]] if which == "best" else c["crps"]["ew"]
            new = c["crps"][c["kind"]] if which == "best" else c["crps"]["sw"]
            sig = c[pk] <= P_MAX
            ax.text(d, i, f"{100 * (new / ref - 1):+.0f}%" + (f"\n{MODELS[c['best']]}" if which == "best" else ""),
                    ha="center", va="center", fontsize=11.5 if which == "best" else 12.5, linespacing=1.2,
                    color=house.INK if sig else house.MUTED, fontweight="bold" if sig else "normal")  # fmt: skip
    if tests:
        ax.axhline(len(rows) - 0.5, color=house.EDGE, lw=1.2)
    labels = [lab for _, lab in rows] + [lab for *_, lab in tests]
    ax.set(xlim=(0.5, len(days) + 0.5), ylim=(len(labels) - 0.5, -0.5), xticks=days, xlabel="Lead (days)")
    ax.set_yticks(range(len(labels)), labels)
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    handles = [
        Patch(color=cmap(0.6), label=f"skilful: r ≥ {SKILL_MIN:g}, significant at {P_MAX:g} and in the FDR test"),
        Patch(color=NOT_SIG, label="tested: no demonstrated skill"),
        Patch(facecolor="white", edgecolor=house.EDGE, label=f"–  fewer than {MIN_RUNS} runs"),
        *[Patch(color="none", label=f"CRPS changes: bold where significant at {P_MAX:g}")] * bool(tests),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncols=3 if len(handles) < 4 else 2, frameon=False)
    fig.suptitle(
        f"{name}: hindcast correlation with ERA5 and runs (n) started within {WINDOW} days of {t0:%-d %B}",
        x=0.01,
        ha="left",
    )
    return VF.save(fig, path)


# ── run to run ────────────────────────────────────────────────────────────────────────────────────────────────────────
def _list(a: np.ndarray, nd: int = 2) -> list:
    return [round(float(x), nd) if np.isfinite(x) else None for x in a]


def _pct(m: np.ndarray, q: float) -> np.ndarray:
    """The q-th percentile of the members (member, lead) at each lead, NaN where none has a value."""
    out = np.full(m.shape[1], np.nan)
    ok = np.isfinite(m).any(0)
    out[ok] = np.nanpercentile(m[:, ok], q, 0)
    return out


def _absmax(*a: np.ndarray) -> float:
    v = np.abs(np.concatenate([np.ravel(x) for x in a]))
    v = v[np.isfinite(v)]
    return float(v.max()) if v.size else 0.0


def record(site: Path, ref: T.Ref, r: dict, ind: tuple) -> bool:
    """Add a run to its model's record unless the record holds it: per index, at each valid time where the run has an
    ensemble mean, the mean and the members' 10th and 90th percentiles, and for the newest run every member as well
    (null where incomplete, or where CPC does not define the pattern; an index undefined at every lead left out); runs
    more than KEEP_DAYS older than the newest dropped. True when the record changed."""
    t, v, mean = ind
    p = site / RECORD.format(r["key"])
    doc = json.loads(p.read_text()) if p.exists() else {"product": "telecon15", "model": r["key"], "runs": []}
    init = f"{r['init']:%Y-%m-%dT%H}"
    if any(x["init"] == init for x in doc["runs"]):
        return False
    have = np.isfinite(mean).any(1)
    rec = {"init": init, "members": int(v.shape[0]), "valid": [f"{x:%Y-%m-%dT%H}" for x in t[have]]}
    for k, s in enumerate(ref.ids):
        if not np.isfinite(mean[have, k]).any():
            continue
        m = np.where(np.isfinite(mean[None, have, k]), v[:, have, k], np.nan)
        rec[s] = {"mean": _list(mean[have, k]), "p10": _list(_pct(m, 10)), "p90": _list(_pct(m, 90)),
                  "members": [_list(m[:, j]) for j in range(m.shape[1])]}  # fmt: skip
    runs = sorted([*doc["runs"], rec], key=lambda x: x["init"])
    cut = pd.Timestamp(runs[-1]["init"]) - pd.Timedelta(days=KEEP_DAYS)
    doc["runs"] = [x for x in runs if pd.Timestamp(x["init"]) >= cut]
    for x in doc["runs"][:-1]:  # every member of the newest run only: the older ones keep their mean and 10-90 %
        for v in x.values():
            if isinstance(v, dict):
                v.pop("members", None)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, separators=(",", ":"), allow_nan=False))
    log(f"  run record {p.name}: {len(doc['runs'])} runs, {init} added")
    return True


def recorded(site: Path) -> dict:
    """{model: its recorded runs, oldest first}."""
    out = {}
    for key in MODELS:
        p = site / RECORD.format(key)
        if p.exists():
            out[key] = json.loads(p.read_text())["runs"]
    return out


def tint(colour, white: float) -> tuple:
    """The colour mixed with white (0: itself, 1: white)."""
    from matplotlib.colors import to_rgb

    c = np.array(to_rgb(colour))
    return tuple(c + (1.0 - c) * white)


def _arr(x: list) -> np.ndarray:
    return np.array([np.nan if y is None else y for y in x], np.float64)


def evolution(sid: str, runs: list[dict], rec: dict, cal: dict, name: str, path: Path):
    """The last SHOW runs' ensemble means against valid date, one panel per model: one shade per run start, the oldest
    lightest, the newest darkest with its members' 10-90 % band; the newest run's leads without demonstrated skill
    hatched."""
    plt, VF = _plt()
    import house
    from matplotlib.patches import Patch

    have = {r["key"]: [x for x in rec.get(r["key"], []) if sid in x] for r in runs}  # runs recorded with the index
    keys = [k for k, x in have.items() if x]
    if not keys:
        return note(name, "No run has been recorded with this index yet: the runs appear here as they arrive.", path)
    starts = sorted({x["init"] for k in keys for x in have[k][-SHOW:]})[-SHOW:]  # one shade per start, every panel
    white = {i: 0.85 * (len(starts) - 1 - j) / max(len(starts) - 1, 1) for j, i in enumerate(starts)}  # newest: 0
    nr, nc = (1, len(keys)) if len(keys) <= 2 else (2, 2)
    fig, axes = plt.subplots(nr, nc, figsize=house.size(15, 0.36 if nr == 1 else 0.62), sharex=True, sharey=True,
                             constrained_layout=True, squeeze=False)  # fmt: skip
    top, t_lo, t_hi = 2.5, [], []
    for ax, key in zip(axes.ravel(), keys, strict=False):
        r = next(x for x in runs if x["key"] == key)
        shown = [x for x in have[key] if x["init"] in white]
        col = COLORS[key]
        ax.axhspan(-THR, THR, color=NEUTRAL, alpha=0.55, lw=0, zorder=0)
        ax.axhline(0, color=house.EDGE, lw=1.0, zorder=1)
        t = pd.DatetimeIndex(r["init"] + pd.to_timedelta(r["lead_h"].astype(int), unit="h"))
        for a0, a1 in spans(t, np.array([not skilful(cal[key][sid, int(h)]) for h in r["lead_h"]])):
            ax.axvspan(a0, a1, facecolor="none", edgecolor=house.EDGE, hatch="//", lw=0, zorder=0)
        for j, x in enumerate(shown):
            tv, m = pd.DatetimeIndex(x["valid"]), _arr(x[sid]["mean"])
            new = j == len(shown) - 1
            if new:
                lo, hi = _arr(x[sid]["p10"]), _arr(x[sid]["p90"])
                ax.fill_between(tv, lo, hi, color=col, alpha=0.16, lw=0, zorder=2)
                top = max(top, 1.08 * _absmax(lo, hi))
            ax.plot(tv, m, color=tint(col, white[x["init"]]), lw=3.0 if new else 1.6, zorder=3 + j)
            t_lo.append(tv[0])
            t_hi.append(tv[-1])
        few = len(rec[key]) < SHOW
        since = f" · recorded since {pd.Timestamp(rec[key][0]['init']):%-d %b %HZ}" if few else ""
        ax.set_title(f"{MODELS[key]} · {len(shown)} run{'s' * (len(shown) != 1)}{since}", color=house.INK)
        ax.grid(True, color=house.GRID, lw=0.8)
    for ax in axes.ravel():
        ax.set_ylim(-top, top)
    for j in range(len(keys), nr * nc):
        axes.flat[j].set_axis_off()
        axes.flat[j - nc].tick_params(labelbottom=True)
    for ax in axes[:, 0]:
        ax.set_ylabel("Seasonal sd")
    date_ticks(axes[0, 0], min(t_lo).normalize(), max(t_hi), nc)
    axes[0, 0].set_xlim(min(t_lo) - pd.Timedelta(hours=12), max(t_hi) + pd.Timedelta(hours=12))
    lines = [
        plt.Line2D(
            [],
            [],
            color=tint(house.INK, white[i]),
            lw=3.0 if i == starts[-1] else 1.6,
            label=f"{pd.Timestamp(i):%-d %b %HZ}",
        )
        for i in starts
    ]
    handles = [
        *lines,
        Patch(color=house.INK2, alpha=0.16, label="newest run's members, 10–90 %"),
        Patch(facecolor="white", edgecolor=house.EDGE, hatch="//", label="no demonstrated skill"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncols=min(len(handles), 6), frameon=False)
    fig.suptitle(f"{name}: each run's ensemble mean by valid date, the oldest run lightest, the newest darkest", x=0.01,
                 ha="left")  # fmt: skip
    return VF.save(fig, path)


def note(name: str, text: str, path: Path):
    """A figure that only says why there is nothing to draw (never a stale figure under the same name)."""
    import textwrap

    plt, VF = _plt()
    import house

    fig = plt.figure(figsize=house.size(15, 0.16))
    fig.text(0.015, 0.7, name, fontsize=19, fontweight="semibold", color=house.INK, va="center")
    fig.text(0.015, 0.36, "\n".join(textwrap.wrap(text, 130)), fontsize=15, color=house.INK2, va="center")
    return VF.save(fig, path)


def undefined(k: int, ref: T.Ref, runs: list[dict]) -> str:
    """Why an index has no forecast: the months CPC defines its pattern in, none of them the forecast's."""
    months = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
    t0 = min(r["init"] for r in runs) + pd.Timedelta(hours=min(T.LEADS))
    t1 = max(r["init"] for r in runs) + pd.Timedelta(hours=max(T.LEADS))
    ok = ref.defined[k]
    first = next((i for i in range(12) if ok[i] and not ok[i - 1]), 0)  # the season in order: Dec, Jan, Feb
    have = ", ".join(months[(first + i) % 12] for i in range(12) if ok[(first + i) % 12])
    return (f"CPC defines this pattern only in {have}: none of this forecast's valid days ({t0:%-d %b} to {t1:%-d %b}) "
            "falls in them, so there is no index to draw.")  # fmt: skip


# ── the update ────────────────────────────────────────────────────────────────────────────────────────────────────────
def _r(x, nd=3):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def summary(ref: T.Ref, runs, ind, cal, probs, bl, fc, obs, figs) -> dict:
    sample = {}
    for key, model in MODELS.items():
        f = fc[fc["model"] == model] if fc is not None else None
        if f is not None and len(f):
            sample[key] = {"runs": int(f["init"].nunique()), "first": f"{f['init'].min():%Y-%m-%d}",
                           "last": f"{f['init'].max():%Y-%m-%d}"}  # fmt: skip
    out = {
        "made": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%MZ"),
        "window_days": WINDOW,
        "gate": {"r_min": SKILL_MIN, "p_max": P_MAX, "fdr_q": FDR_Q, "min_runs": MIN_RUNS, "threshold_sd": THR},
        "era5_last": None if obs is None or obs.empty else f"{obs['valid'].max():%Y-%m-%d}",
        "hindcast": sample,
        "runs": {},
        "figures": figs,
    }
    for r in runs:
        _, v, mean = ind[r["key"]]
        e = {"cycle": r["cycle"], "init": f"{r['init']:%Y-%m-%dT%H}:00Z", "members": int(r["ok"].shape[0]),
             "lead_h": [int(h) for h in r["lead_h"]], "indices": {}}  # fmt: skip
        for k, s in enumerate(ref.ids):
            rows = []
            for j, h in enumerate(r["lead_h"]):
                c, pp = cal[r["key"]][s, int(h)], probs[r["key"]].get((s, int(h)))
                m = v[:, j, k][np.isfinite(v[:, j, k])]
                row = {"mean": _r(mean[j, k], 2)} | ({"p10": _r(np.percentile(m, 10), 2),
                                                     "p90": _r(np.percentile(m, 90), 2)} if m.size else {})  # fmt: skip
                if c:
                    row |= {"n": c["n"], "n_eff": _r(c["n_eff"], 1), "r": _r(c["r"], 2), "p": _r(c["p"], 4),
                            "fdr_ok": c["fdr_ok"], "skilful": c["skilful"]}  # fmt: skip
                if skilful(c) and pp is not None:
                    row |= {"p_neg": _r(pp[0]), "p_pos": _r(pp[1]), "cal_mean": _r(c["a"] * mean[j, k] + c["b"], 2)}
                rows.append(row)
            e["indices"][s] = rows
        out["runs"][r["key"]] = e
    out["blend"] = {}
    for s, b in bl.items():
        rows = []
        for j, lead in enumerate(T.LEADS):
            c = cal["blend"][s, lead]
            row = {} if c is None else {
                "n": c["n"], "n_eff": _r(c["n_eff"], 1), "r_cv": _r(c["r"], 2), "p_cv": _r(c["p"], 4),
                "r_in": _r(c["r_in"], 2), "fdr_ok": c["fdr_ok"], "skilful": c["skilful"], "weights": c["kind"],
                "w": {k: _r(v, 3) for k, v in c["weights"].items()}, "seasons": c["seasons"],
                "crps_cv": {k: _r(v, 4) for k, v in c["crps"].items()}, "p_sw": _r(c["p_sw"], 4),
                "best": c["best"], "p_vs_best": _r(c["p_best"], 4),
            }  # fmt: skip
            if b.get("init") is not None and np.isfinite(b["mean"][j]):
                row |= {"mean": _r(b["mean"][j], 2), "p10": _r(b["lo"][j], 2), "p90": _r(b["hi"][j], 2)}
                if skilful(c) and lead in b["prob"]:
                    row |= {"p_neg": _r(b["prob"][lead][1]), "p_pos": _r(b["prob"][lead][2])}
            rows.append(row)
        out["blend"][s] = {"models": b["models"], "init": None if b.get("init") is None else f"{b['init']:%Y-%m-%dT%H}:00Z",
                           "leads": rows}  # fmt: skip
    return out


def cmd_run(a) -> int:
    runs = load(a.fields)
    if not runs:
        log("no teleconnection fields: nothing to draw")
        return 0
    ref = T.Ref(a.ref)
    fc = pd.read_parquet(a.table / T.FC) if a.table and (a.table / T.FC).exists() else None
    obs = pd.read_parquet(a.table / T.OBS) if a.table and (a.table / T.OBS).exists() else None
    if fc is None or obs is None:
        log("::notice::no hindcast table yet: every lead is drawn without demonstrated skill")
    ind = {r["key"]: member_indices(ref, r) for r in runs}
    for r in runs:
        record(a.site, ref, r, ind[r["key"]])
    rec = recorded(a.site)
    cal = {r["key"]: calibrate(fc, obs, MODELS[r["key"]], r["init"]) for r in runs}
    bl = blend(fc, obs, ref, runs, rec)
    cal["blend"] = {(s, lead): bl[s]["fit"][lead] for s in ref.ids for lead in T.LEADS}
    gate(cal)  # every model's cells and the blend's, one family
    probs = {}
    for r in runs:
        mean = ind[r["key"]][2]
        probs[r["key"]] = {
            (s, int(h)): (float(cal_cdf(-THR, c, mean[j, k])), float(1.0 - cal_cdf(THR, c, mean[j, k])))
            for k, s in enumerate(ref.ids)
            for j, h in enumerate(r["lead_h"])
            if skilful(c := cal[r["key"]][s, int(h)]) and np.isfinite(mean[j, k])
        }
    out = a.site / FIG
    out.mkdir(parents=True, exist_ok=True)
    t0 = max(r["init"] for r in runs)
    figs = []
    for k, s in enumerate(ref.ids):
        name = f"{SHORT[s]} ({ref.meta['patterns'][s]['name']})"
        if not any(np.isfinite(ind[r["key"]][1][..., k]).any() for r in runs):
            figs += [
                note(name, undefined(k, ref, runs), out / f"{s}_{v}.webp") for v in ("plume", "prob", "runs", "skill")
            ]
            continue
        figs.append(plume(s, k, runs, ind, cal, obs, bl[s], name, out / f"{s}_plume.webp"))
        figs.append(prob(s, k, runs, ind, probs, cal, bl[s], name, out / f"{s}_prob.webp"))
        figs.append(evolution(s, runs, rec, cal, name, out / f"{s}_runs.webp"))
        figs.append(skill(s, runs, cal, bl[s], name, t0, out / f"{s}_skill.webp"))
    js = a.site / JSON_OUT
    js.parent.mkdir(parents=True, exist_ok=True)
    js.write_text(json.dumps(summary(ref, runs, ind, cal, probs, bl, fc, obs, figs), separators=(",", ":")) + "\n")
    passing = {
        f"{key} {s}": [lead // 24 for lead in T.LEADS if skilful(cal[key][s, lead])]
        for key in [*(r["key"] for r in runs), "blend"]
        for s in ref.ids
    }
    log(f"teleconnections: {', '.join(r['key'] + ' ' + r['cycle'] for r in runs)}; skilful leads (days) {passing}")
    return 0


def cmd_skill(a) -> int:
    """What the gate passes on a date from the hindcast table alone, every model and the blend of all four: the
    skilful leads (days) per index, and the samples. Printed for the hindcast pass's log (telecon15.yml)."""
    fc, obs = pd.read_parquet(a.table / T.FC), pd.read_parquet(a.table / T.OBS)
    t0 = pd.Timestamp(a.date) if a.date else pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    cal = {k: calibrate(fc, obs, m, t0) for k, m in MODELS.items()}
    cal["blend"], used = {}, {}
    for sid in T.IDS:
        w = wide(fc, obs, sid, list(MODELS), t0)
        used[sid] = subset(w, list(MODELS), t0)
        cal["blend"] |= {(sid, lead): blend_fit(w, used[sid], lead, t0) if used[sid] else None for lead in T.LEADS}
    gate(cal)
    plain = {k: raw(fc, obs, m, t0) for k, m in MODELS.items()}
    print(f"hindcast skill for starts within {WINDOW} days of {t0:%-d %B}: at each lead day 1-15 the correlation of "
          "the ensemble-mean index with ERA5's and the number of starts (r/n; the blend's cross-validated), then * "
          f"skilful, x tested and not, - fewer than {MIN_RUNS} starts")  # fmt: skip
    for key, per in cal.items():
        print(f"{MODELS.get(key, 'Blend')}")
        for sid in T.IDS:
            cells = []
            for lead in T.LEADS:
                c = per[sid, lead]
                r, n = (c["r"], c["n"]) if c else plain.get(key, {}).get((sid, lead), (np.nan, 0))
                cells.append(f"{r:+.2f}/{n:<3d}{'*' if skilful(c) else 'x' if c else '-'}")
            extra = f"  ({'+'.join(used[sid])})" if key == "blend" and used[sid] else ""
            print(f"  {SHORT[sid]:6s} {' '.join(cells)}{extra}")
    return 0


def cmd_qc(a) -> int:
    """A few runs' database files checked against ERA5, for IFS-ENS, AIFS-ENS and GEFS on the same starts: the leads
    and members each file holds, the valid time each lead is read at, the members' mean on the patterns' grid against
    the database's ensemble mean, the NH anomaly correlation of the ensemble mean with ERA5 (z500_acc.scores, as the
    verification page scores it) and the index of the forecast against ERA5's. --list prints the files, for the
    newest QC_RUNS starts all three hold with ERA5 on days 1, 5 and 10."""
    sys.path.insert(0, str(HERE.parents[0] / "verify"))
    import z500_acc as Z

    keys = ("ifs", "aifs", "gefs")
    days = (1, 5, 10)
    if a.list:
        have = set(a.have.read_text().split())
        inits = sorted({n.split("_")[1][:10] for n in have if n.startswith("ens-z500m-ifs_")}, reverse=True)
        picked = []
        for init in inits:
            t0 = pd.to_datetime(init, format="%Y%m%d%H")
            want = [Z.file_of(MODELS[k], init, members) for k in keys for members in (False, True)]
            want += [Z.truth_file(t0 + pd.Timedelta(days=d)) for d in days]
            if all(n in have for n in want):
                picked += want + [n for n in (f"acc_{init}.parquet",) if n in have]
            if len(picked) and len({x.split("_")[1][:10] for x in picked if x.startswith("ens-")}) >= QC_RUNS:
                break
        print(" ".join(dict.fromkeys(picked)))
        return 0
    from latlon import G15

    ref = T.Ref(a.ref)
    normal = np.load(a.clim)["z500"]
    for f in sorted(a.derived.glob("ens-z500m-ifs_*.npz")):
        init = f.stem.split("_")[1]
        t0 = pd.to_datetime(init, format="%Y%m%d%H")
        acc = pd.read_parquet(p) if (p := a.derived / f"acc_{init}.parquet").exists() else None
        print(f"start {t0:%Y-%m-%d %HZ}")
        for k in keys:
            m, e = (
                np.load(a.derived / Z.file_of(MODELS[k], init, True)),
                np.load(a.derived / Z.file_of(MODELS[k], init)),
            )
            z = m["z"].astype(np.float64)
            print(f"  {MODELS[k]:8s} members file: leads {m['lead_h'][0]}..{m['lead_h'][-1]} h ({len(m['lead_h'])}, as "
                  f"LEADS: {list(m['lead_h']) == list(T.LEADS)}), {z.shape[1]} members {m['member'].min()}-"
                  f"{m['member'].max()}, NaN {int(np.isnan(z).sum())}, {z.min():.0f}-{z.max():.0f} m; mean file: leads as "
                  f"members' {list(e['lead_h']) == list(m['lead_h'])}, n_members {int(e['n_members'])}")  # fmt: skip
            for d in days:
                j = list(m["lead_h"]).index(24 * d)
                valid = t0 + pd.Timedelta(hours=int(m["lead_h"][j]))
                era = np.load(a.derived / Z.truth_file(valid))["z"]
                sc = Z.scores(e["z"][j], era, normal[slot366(pd.DatetimeIndex([valid]))[0]])["NH"]
                mean25 = z[j].mean(0)
                diff = mean25 - remap(e["z"][j], G15.lat, G15.lon, G25N)
                t = pd.DatetimeIndex([valid, valid])
                fi, oi = ref.index(ref.anomaly(np.stack([mean25, remap(era, G15.lat, G15.lon, G25N)]), t), t)
                db = ""
                if acc is not None:
                    row = acc[(acc["model"] == MODELS[k]) & (acc["lead_h"] == 24 * d) & (acc["region"] == "NH")]
                    db = f", database {float(row['acc'].iloc[0]):.3f}" if len(row) else ""
                idx = " ".join(
                    f"{SHORT[s]} {fi[i]:+.2f}/{oi[i]:+.2f}" for i, s in enumerate(ref.ids) if s in ("nao", "pna", "epo")
                )
                print(f"    day {d:2d} valid {valid:%Y-%m-%d %HZ}: NH ACC {sc['acc']:.3f}{db}; members' mean less the "
                      f"mean file {np.abs(diff).mean():.1f} m; index fc/ERA5 {idx}")  # fmt: skip
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--fields", type=Path, nargs="+", required=True, help="the models' telecon/<model>.npz (Capture)")
    r.add_argument("--ref", type=Path, required=True, help=f"{T.REF_NAME} (release {T.REF_TAG})")
    r.add_argument("--table", type=Path, help="the hindcast table's directory (telecon15.py hindcast)")
    r.add_argument("--site", type=Path, default=HERE.parents[1])
    k = sub.add_parser("skill", help="print what the gate passes on a date from the hindcast table alone")
    k.add_argument("--table", type=Path, required=True)
    k.add_argument("--date", help="YYYY-MM-DD (default today)")
    q = sub.add_parser("qc", help="check a few runs' database files against ERA5")
    q.add_argument("--list", action="store_true", help="print the files to download (--have)")
    q.add_argument("--have", type=Path, help="names of the files the database holds")
    q.add_argument("--derived", type=Path, help="the files, downloaded")
    q.add_argument("--clim", type=Path, help="clim_z500_1p5.npz (release verify-clim-z500-v1)")
    q.add_argument("--ref", type=Path, help=f"{T.REF_NAME}")
    a = ap.parse_args()
    return {"run": cmd_run, "skill": cmd_skill, "qc": cmd_qc}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
