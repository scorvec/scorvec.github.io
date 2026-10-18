#!/usr/bin/env python3
"""Track record of the TC board: ECMWF's tracker on IFS-ENS, the IFS control, AIFS-ENS and AIFS single against best
tracks.

Forecasts (Google Cloud mirror only), every 00 and 12 UTC cycle since FIRST:
  IFS-ENS   .../ifs/0p25/enfo/...-enfo-tf.bufr        50 perturbed members + the IFS control (the type-0 subset; HRES until May 2026)
  AIFS-ENS  .../aifs-ens/0p25/enfo/...-enfo-tf.bufr   50 perturbed members + AIFS single (on the mirror from May 2026)
  `tracks` decodes them into monthly archives (tracks_YYYYMM.npz: the storms with an official id, every member and the
  deterministic run), kept in a GitHub release so that a run decodes only what is new.
Best tracks: IBTrACS v04r01 (NOAA NCEI), the last-three-years file, 6-hourly synoptic positions. A forecast storm is the
best-track storm nearest its observed position at the start, within MATCH_KM.
Scores (`score`), Northern Hemisphere:
  track error    great-circle distance from the best track, by lead, while the best track is not extratropical or a
                 disturbance (IBTrACS nature ET, DS): the deterministic run, the ensemble-mean position (at least half the
                 members carrying the storm) and persistence of the last 12 h of best-track motion, on one homogeneous
                 sample per lead
  recurvature    the board's rule (tc_ens.outcome on each track with the observed history in front) for every member and
                 for the best track: the chance of recurving by day 10 (now or by then) against what happened; storms that
                 had already turned at the start are left out. Brier skill score against the base rate (each storm's
                 reference rate from the other storms), reliability in bins, the watch rule, and the error of the median
                 recurving member's turn
  post-tropical  the share of members whose tracker track ends north of 30N by day 10 (tc_ens.post_tropical_h) against
                 the best track turning extratropical (nature ET) by day 10; storms whose best track gives no nature
                 (provisional records) are left out
Intervals: 95 %, from BOOT resamples of whole storms (forecasts of one storm from consecutive cycles are not independent).
A score is called skilful only when its whole interval is above zero; a sample below MIN_STORMS storms (or MIN_EVENTS
events and non-events) is "no track record yet".
Outputs: scripts/mjo/data/reference/tc_verify.json (numbers, and the board's one-line summaries) and a figure.
    python src/tc_verify.py tracks --start 2024-08 --end 2026-10 --out work/tracks
    python src/tc_verify.py score --tracks work/tracks --ibtracs work/ibtracs.csv \
        --out data/reference/tc_verify.json --figure ../../assets/sst/tc_verify.webp
"""

from __future__ import annotations

import pyproj  # noqa: F401  before eccodes/xarray: eckit's bundled libproj otherwise clashes (double free at exit)

import argparse
import itertools
import datetime as dt
import json
import sys
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import tc_ens as TE
import tclib as TC

FIRST = "2024-08"  # the verification starts with the 2024 Northern-Hemisphere peak season
MODELS = ("ifs", "aifs")
MATCH_KM = 300.0
LEADS = (0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240)
HORIZON_H = 24 * TE.N_DAYS
BOOT = 2000
MIN_STORMS, MIN_EVENTS = 10, 5
BINS = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0001)
IBTRACS = (
    "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/"
    "access/csv/ibtracs.last3years.list.v04r01.csv"
)
NOT_TROPICAL = ("ET", "DS")


# ── tracks: decode the tracker files into monthly archives ──────────────────────────────────────────────────────────
def _cycles(month: str):
    """(YYYYMMDD, HH) of every 00/12 UTC cycle of a month that is at least 9 h old."""
    t = pd.Timestamp(f"{month}-01")
    end = min(t + pd.offsets.MonthBegin(1), pd.Timestamp.now("UTC").tz_localize(None) - pd.Timedelta(hours=9))
    while t < end:
        yield f"{t:%Y%m%d}", f"{t:%H}"
        t += pd.Timedelta(hours=12)


def _decode(job):
    """One cycle and ensemble -> (storm rows, point rows), or None when its tracker file is not on the mirror."""
    model, d8, hh = job
    with tempfile.TemporaryDirectory() as tmp:
        fp = TC.tracker_file(model, d8, hh, Path(tmp), quiet=True)
        if fp is None:
            return job, None
        run_h = TC.run_hours(fp)
        try:
            msgs = TC.decode_all(fp, types=(0, 4))
        except Exception as e:  # noqa: BLE001  an unreadable file counts as missing
            print(f"  {model} {d8} {hh}Z unreadable ({e!r})", flush=True)
            return job, None
    storms, points = [], []
    for m in msgs:
        if not m["named"] or m["obs"] is None or m["obs"][0] <= 0:
            continue
        k = len(storms)
        storms.append((int(d8 + hh), MODELS.index(model), m["id"], m["name"][:24], *m["obs"], run_h))
        for mem, t in m["tracks"].items():
            points.extend(
                (k, mem, int(h), la, lo, p) for h, la, lo, p in zip(t["steps"], t["lat"], t["lon"], t["pmsl"])
            )
    return job, (storms, points)


S_DT = [
    ("cycle", "i8"),
    ("model", "i1"),
    ("sid", "U8"),
    ("name", "U24"),
    ("obs_lat", "f4"),
    ("obs_lon", "f4"),
    ("run_h", "i2"),
]
P_DT = [("storm", "i4"), ("member", "i1"), ("step", "i2"), ("lat", "f4"), ("lon", "f4"), ("pmsl", "f4")]


def tracks(a) -> int:
    """Decode every cycle of the months start..end (YYYY-MM) into out/tracks_YYYYMM.npz: the months whose archive is not
    in out yet, and the last `refresh` months again (their newest cycles were not out, or not old enough, last time).
    The archives written are listed in out/rebuilt.txt."""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    months = list(pd.period_range(a.start, a.end, freq="M").astype(str))
    todo = [
        m
        for i, m in enumerate(months)
        if i >= len(months) - a.refresh or not (out / f"tracks_{m.replace('-', '')}.npz").exists()
    ]
    (out / "rebuilt.txt").write_text("".join(f"tracks_{m.replace('-', '')}.npz\n" for m in todo))
    for month in todo:
        t0 = time.time()
        jobs = [(m, d8, hh) for d8, hh in _cycles(month) for m in MODELS]
        storms, points, missing = [], [], []
        with Pool(a.procs) as pool:
            for job, res in pool.imap(_decode, jobs, chunksize=2):
                if res is None:
                    missing.append((int(job[1] + job[2]), MODELS.index(job[0])))
                    continue
                off = len(storms)
                storms.extend(res[0])
                points.extend((k + off, *rest) for k, *rest in res[1])
        np.savez_compressed(
            out / f"tracks_{month.replace('-', '')}.npz",
            storms=np.array(storms, dtype=S_DT),
            points=np.array(points, dtype=P_DT),
            missing=np.array(missing, dtype=[("cycle", "i8"), ("model", "i1")]),
        )
        n_ok = {
            m: sum(1 for d8, hh in _cycles(month)) - sum(1 for c, mm in missing if mm == MODELS.index(m))
            for m in MODELS
        }
        print(
            f"{month}: {len(storms)} storm forecasts, {len(points)} points; files "
            + ", ".join(f"{TC.MODELS[m]} {n_ok[m]}" for m in MODELS)
            + f" ({time.time() - t0:.0f} s)",
            flush=True,
        )
    return 0


def load_tracks(d: Path):
    """All archives: (storms DataFrame with a row id, {row id: {member: track}}) - member 0 = the deterministic run."""
    S, P, off = [], [], 0
    for f in sorted(d.glob("tracks_*.npz")):
        with np.load(f) as z:
            s, p = pd.DataFrame(z["storms"]), pd.DataFrame(z["points"])
        p["storm"] += off
        off += len(s)
        S.append(s)
        P.append(p)
    storms = pd.concat(S, ignore_index=True)
    pts = pd.concat(P, ignore_index=True).sort_values(["storm", "member", "step"])
    trk = {}
    for (k, mem), g in pts.groupby(["storm", "member"], sort=False):
        trk.setdefault(int(k), {})[int(mem)] = dict(
            steps=g["step"].to_numpy(int),
            lat=g["lat"].to_numpy(float),
            lon=g["lon"].to_numpy(float),
            pmsl=g["pmsl"].to_numpy(float),
            wind=np.full(len(g), np.nan),
        )
    storms["t0"] = pd.to_datetime(storms["cycle"].astype(str), format="%Y%m%d%H")
    return storms, trk


# ── best tracks ──────────────────────────────────────────────────────────────────────────────────────────────────────
def load_ibtracs(path: Path) -> dict:
    """{IBTrACS SID: DataFrame(time, lat, lon, nature, name, provisional)} at the synoptic hours, main and provisional
    tracks only. A provisional track (the current season, from real-time positions) is good for positions but records no
    transition: its nature is not reported, or tropical to the end. Basin "NA" (North Atlantic) is a value, not a missing
    one (keep_default_na=False)."""
    df = pd.read_csv(
        path,
        skiprows=[1],
        usecols=["SID", "NAME", "ISO_TIME", "NATURE", "LAT", "LON", "TRACK_TYPE"],
        keep_default_na=False,
        low_memory=False,
    )
    df = df[~df["TRACK_TYPE"].str.startswith("spur")]
    df["time"] = pd.to_datetime(df["ISO_TIME"])
    df = df[(df["time"].dt.minute == 0) & (df["time"].dt.hour % 6 == 0)]
    df["lat"] = pd.to_numeric(df["LAT"], errors="coerce")
    df["lon"] = pd.to_numeric(df["LON"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])
    df["lon"] = (df["lon"] + 180) % 360 - 180
    df["provisional"] = df.groupby("SID")["TRACK_TYPE"].transform(lambda x: (x == "PROVISIONAL").any())
    return {
        sid: g[["time", "lat", "lon", "NATURE", "NAME", "provisional"]]
        .rename(columns={"NATURE": "nature", "NAME": "name"})
        .reset_index(drop=True)
        for sid, g in df.groupby("SID")
    }


def bt_index(bt: dict) -> dict:
    """time -> [(sid, lat, lon)] for matching."""
    idx = {}
    for sid, g in bt.items():
        for t, la, lo in zip(g["time"], g["lat"], g["lon"]):
            idx.setdefault(t, []).append((sid, la, lo))
    return idx


def bt_window(g: pd.DataFrame, t0, h0=-48, h1=HORIZON_H + 24) -> dict:
    """The best track as a track dict with steps relative to t0 (h0..h1), plus its nature per step."""
    h = ((g["time"] - t0) / pd.Timedelta(hours=1)).to_numpy()
    m = (h >= h0) & (h <= h1)
    return dict(
        steps=h[m].astype(int),
        lat=g["lat"].to_numpy(float)[m],
        lon=g["lon"].to_numpy(float)[m],
        pmsl=np.full(int(m.sum()), np.nan),
        wind=np.full(int(m.sum()), np.nan),
        nature=g["nature"].to_numpy()[m],
    )


# ── statistics ───────────────────────────────────────────────────────────────────────────────────────────────────────
def boot(cases: pd.DataFrame, stat, seed: int = 1) -> tuple[float, float, float]:
    """stat(cases) and its 95 % interval from BOOT resamples of whole storms (column "storm")."""
    est = float(stat(cases))
    codes, uniq = pd.factorize(cases["storm"])
    if len(uniq) < 2:
        return est, np.nan, np.nan
    rows = [np.flatnonzero(codes == k) for k in range(len(uniq))]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(BOOT):
        pick = rng.integers(0, len(uniq), len(uniq))
        v = stat(cases.iloc[np.concatenate([rows[i] for i in pick])])
        if np.isfinite(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan)
    return est, float(lo), float(hi)


def _r(x, n=1):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def loso_rate(cases: pd.DataFrame, col: str) -> np.ndarray:
    """Each case's reference rate: the event frequency over the OTHER storms' cases."""
    tot, n = cases[col].sum(), len(cases)
    by = cases.groupby("storm")[col].agg(["sum", "count"])
    s = cases["storm"].map(by["sum"]).to_numpy()
    c = cases["storm"].map(by["count"]).to_numpy()
    return (tot - s) / np.maximum(n - c, 1)


def prob_scores(cases: pd.DataFrame, p: str, o: str) -> dict:
    """Brier skill score against the leave-one-storm-out base rate, reliability by BINS, sample sizes."""
    c = cases.dropna(subset=[p, o]).copy()
    c["clim"] = loso_rate(c, o)
    n_st, n_ev = c["storm"].nunique(), int(c[o].sum())
    out = dict(cases=len(c), storms=int(n_st), events=n_ev, base_rate=_r(c[o].mean(), 3))
    if n_st < MIN_STORMS or n_ev < MIN_EVENTS or len(c) - n_ev < MIN_EVENTS:
        out["track_record"] = False
        return out

    def bss(x):
        ref = np.mean((x["clim"] - x[o]) ** 2)
        return 1 - np.mean((x[p] - x[o]) ** 2) / ref if ref > 0 else np.nan

    est, lo, hi = boot(c, bss)
    out.update(track_record=True, bss=[_r(est, 3), _r(lo, 3), _r(hi, 3)], skilful=bool(np.isfinite(lo) and lo > 0))
    rel = []
    for b0, b1 in itertools.pairwise(BINS):
        x = c[(c[p] >= b0) & (c[p] < b1)]
        if not len(x):
            continue
        f, olo, ohi = boot(x, lambda y: y[o].mean())
        rel.append(
            dict(
                bin=[b0, min(b1, 1.0)],
                cases=len(x),
                storms=int(x["storm"].nunique()),
                forecast=_r(x[p].mean(), 3),
                observed=[_r(f, 3), _r(olo, 3), _r(ohi, 3)],
            )
        )
    out["reliability"] = rel
    big = [r for r in rel if r["cases"] >= 20 and r["observed"][1] is not None]
    out["reliable"] = bool(big) and all(r["observed"][1] <= r["forecast"] <= r["observed"][2] for r in big)
    return out


def timing(cases: pd.DataFrame, fc: str, ob: str) -> list[dict]:
    """Error (h) of a forecast time against the observed one, by how far ahead the event was."""
    c = cases.dropna(subset=[fc, ob]).copy()
    c["err"] = c[fc] - c[ob]
    out = []
    for a, b in ((0, 48), (48, 96), (96, 144), (144, HORIZON_H + 1)):
        x = c[(c[ob] >= a) & (c[ob] < b)]
        if x["storm"].nunique() < 3:
            out.append(dict(lead_h=[a, b], cases=len(x), storms=int(x["storm"].nunique())))
            continue
        mae = boot(x, lambda y: y["err"].abs().mean())
        bias = boot(x, lambda y: y["err"].mean())
        out.append(
            dict(
                lead_h=[a, b],
                cases=len(x),
                storms=int(x["storm"].nunique()),
                mae_h=[_r(v) for v in mae],
                bias_h=[_r(v) for v in bias],
            )
        )
    return out


# ── score ────────────────────────────────────────────────────────────────────────────────────────────────────────────
def ens_mean(tr: dict, h: int, need: int):
    ps = [q for q in (TC.at(t, h) for k, t in tr.items() if k > 0) if q is not None]
    if len(ps) < need:
        return None
    return float(np.mean([q[0] for q in ps])), TC.lon_median([q[1] for q in ps])


def persistence(bt: dict, h: int):
    a, b = TC.at(bt, -12), TC.at(bt, 0)
    if a is None or b is None:
        return None
    dlo = ((b[1] - a[1] + 180) % 360) - 180
    return b[0] + (b[0] - a[0]) * h / 12, ((b[1] + dlo * h / 12 + 180) % 360) - 180


def member_stats(tr: dict, hist, run_h: int) -> dict:
    """The board's numbers for one ensemble forecast: recurvature chance and turns, post-tropical chance and times."""
    rec, turns, pts = 0, [], []
    for k in range(1, TC.NMEM + 1):
        t = tr.get(k)
        if t is None:
            continue
        o, r = TE.outcome(TE.with_history(t, hist), run_h)
        if o in ("now", "by"):
            rec += 1
            turns.append(r)
        pt = TE.post_tropical_h(t, run_h)
        if pt is not None:
            pts.append(pt)
    return dict(
        k_rec=rec,
        k_pt=len(pts),
        turn_med=float(np.median(turns)) if len(turns) >= TC.NMEM // 4 else np.nan,
        pt_med=float(np.median(pts)) if len(pts) >= TC.NMEM // 4 else np.nan,
        p_lo=TE.wilson(rec, TC.NMEM)[0],
    )


def cases_table(storms: pd.DataFrame, trk: dict, bt: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per forecast storm (cycle, ensemble) matched to a best track: recurvature and post-tropical numbers; and
    one row per (case, lead) with the track errors."""
    idx = bt_index(bt)
    hist_of = {}
    for r in storms.itertuples():
        hist_of.setdefault((r.model, r.sid), {})[r.t0] = (float(r.obs_lat), float(r.obs_lon))
    rows, errs = [], []
    for r in storms.itertuples():
        cand = [(TC._km(r.obs_lat, r.obs_lon, la, lo), sid) for sid, la, lo in idx.get(r.t0, [])]
        cand = [c for c in cand if c[0] <= MATCH_KM]
        if not cand:
            continue
        sid = min(cand)[1]
        g = bt[sid]
        b = bt_window(g, r.t0)
        k0 = np.where(b["steps"] == 0)[0]
        if not len(k0) or b["nature"][k0[0]] in NOT_TROPICAL:
            continue
        tr = trk.get(r.Index, {})
        hs = hist_of[(r.model, r.sid)]
        hist = [
            (-12 * k, *hs[r.t0 - pd.Timedelta(hours=12 * k)])
            for k in range(1, TE.HIST_CYCLES + 1)
            if r.t0 - pd.Timedelta(hours=12 * k) in hs
        ]
        st = member_stats(tr, hist, int(r.run_h))
        det = tr.get(0)
        ob, r_obs = TE.outcome(b, 10**6)
        u = TC.zonal_motion(b)
        turned = (r_obs is not None and r_obs <= 0) or (np.isfinite(u[k0[0]]) and u[k0[0]] >= TC.REC_EAST)
        nat = b["nature"][(b["steps"] > 0) & (b["steps"] <= HORIZON_H)]
        known = not g["provisional"].iloc[0] and np.any((nat != "NR") & (nat != "") & (nat != "MX"))
        et = np.where((b["steps"] > 0) & (b["steps"] <= HORIZON_H) & (b["nature"] == "ET"))[0]
        rows.append(
            dict(
                storm=sid,
                bt_name=g["name"].iloc[0],
                t0=r.t0,
                model=MODELS[r.model],
                sid=r.sid,
                lat0=float(r.obs_lat),
                lon0=float(r.obs_lon),
                p_rec=st["k_rec"] / TC.NMEM,
                k_rec=st["k_rec"],
                p_lo=st["p_lo"],
                turn_fc=st["turn_med"],
                det_rec=None
                if det is None
                else float(TE.outcome(TE.with_history(det, hist), int(r.run_h))[0] in ("now", "by")),
                rec_obs=np.nan if turned else float(ob in ("now", "by")),
                turn_obs=float(r_obs) if (r_obs is not None and not turned and ob in ("now", "by")) else np.nan,
                p_pt=st["k_pt"] / TC.NMEM,
                k_pt=st["k_pt"],
                pt_fc=st["pt_med"],
                pt_obs=np.nan if not known else float(len(et) > 0),
                pt_time=float(b["steps"][et[0]]) if (known and len(et)) else np.nan,
            )
        )
        for L in LEADS:
            if r.run_h < L:
                continue
            kb = np.where(b["steps"] == L)[0]
            if not len(kb) or b["nature"][kb[0]] in NOT_TROPICAL:
                continue
            o_ = (b["lat"][kb[0]], b["lon"][kb[0]])
            d = TC.at(det, L) if det is not None else None
            e = ens_mean(tr, L, TC.NMEM // 2)
            p_ = persistence(b, L)
            errs.append(
                dict(
                    storm=sid,
                    t0=r.t0,
                    model=MODELS[r.model],
                    lead=L,
                    det=np.nan if d is None else TC._km(*o_, *d),
                    ens=np.nan if e is None else TC._km(*o_, *e),
                    pers=np.nan if p_ is None else TC._km(*o_, *p_),
                )
            )
    return pd.DataFrame(rows), pd.DataFrame(errs)


def track_scores(E: pd.DataFrame, model: str) -> dict:
    """Mean track errors by lead on homogeneous samples (deterministic, ensemble mean and persistence all present; at
    lead 0, where persistence is the best track itself, without it), and the skill against persistence after lead 0."""
    out = dict(
        leads=[], cases=[], storms=[], det_km=[], ens_km=[], pers_km=[], ens_minus_det_km=[], det_skill=[], ens_skill=[]
    )
    for L in LEADS:
        x = E[(E["model"] == model) & (E["lead"] == L)].dropna(subset=["det", "ens"] + (["pers"] if L else []))
        if x["storm"].nunique() < MIN_STORMS:
            continue
        out["leads"].append(L)
        out["cases"].append(len(x))
        out["storms"].append(int(x["storm"].nunique()))
        for key, f in (
            ("det_km", lambda y: y["det"].mean()),
            ("ens_km", lambda y: y["ens"].mean()),
            ("pers_km", lambda y: y["pers"].mean()),
            ("ens_minus_det_km", lambda y: (y["ens"] - y["det"]).mean()),
            ("det_skill", lambda y: 1 - y["det"].mean() / y["pers"].mean()),
            ("ens_skill", lambda y: 1 - y["ens"].mean() / y["pers"].mean()),
        ):
            skill = "skill" in key
            out[key].append(
                [None] * 3 if (skill or key == "pers_km") and not L else [_r(v, 3 if skill else 0) for v in boot(x, f)]
            )
    return out


def watch_scores(C: pd.DataFrame) -> dict:
    """The watch rule (lower 90 % bound of the chance of recurving by day 10 >= WATCH_P; the board's second condition,
    the median member's turn by day 10, follows from it) against what happened: the share of watches followed by a
    recurvature, and the share of recurvatures watched."""
    c = C.dropna(subset=["rec_obs"]).copy()
    c["watch"] = (c["p_lo"] >= TE.WATCH_P).astype(float)
    out = dict(watches=int(c["watch"].sum()), storms_watched=int(c.loc[c["watch"] > 0, "storm"].nunique()))
    w = c[c["watch"] > 0]
    if w["storm"].nunique() >= MIN_STORMS:
        out["hit_rate"] = [_r(v, 3) for v in boot(w, lambda y: y["rec_obs"].mean())]
    e = c[c["rec_obs"] > 0]
    if e["storm"].nunique() >= MIN_STORMS:
        out["detected"] = [_r(v, 3) for v in boot(e, lambda y: y["watch"].mean())]
    return out


def pooled(C: pd.DataFrame) -> pd.DataFrame:
    """Cases where both ensembles carry the storm: the pooled chances (100 members)."""
    a = C[C["model"] == "aifs"].set_index(["storm", "t0"])
    i = C[C["model"] == "ifs"].set_index(["storm", "t0"])
    j = a.join(i, how="inner", lsuffix="_a", rsuffix="_i")
    if not len(j):
        return pd.DataFrame(columns=["storm", "p_rec", "rec_obs", "p_pt", "pt_obs", "p_lo"])
    k = j["k_rec_a"] + j["k_rec_i"]
    return pd.DataFrame(
        dict(
            p_rec=k / (2 * TC.NMEM),
            rec_obs=j["rec_obs_i"],
            p_pt=(j["k_pt_a"] + j["k_pt_i"]) / (2 * TC.NMEM),
            pt_obs=j["pt_obs_i"],
            p_lo=[TE.wilson(int(x), 2 * TC.NMEM)[0] for x in k],
        )
    ).reset_index()


def _span(C: pd.DataFrame) -> list[str]:
    return [f"{C['t0'].min():%Y-%m-%d}", f"{C['t0'].max():%Y-%m-%d}"] if len(C) else [None, None]


def _ci(v, pct=True, n=0) -> str:
    f = (lambda x: f"{100 * x:.{n}f} %") if pct else (lambda x: f"{x:.2f}")
    return f"{f(v[0])} (95 % interval {f(v[1])} to {f(v[2])})"


def summary(js: dict) -> dict:
    """The board's one-line statements: skill only where the interval excludes zero; otherwise what the sample shows."""
    out = {}

    def prob_line(what, q, who, since):
        if not q.get("track_record"):
            return (
                f"{who}: no track record yet for the chance of {what} ({q.get('cases', 0)} forecasts of "
                f"{q.get('storms', 0)} storms since {since})."
            )
        s = (
            f"{who}, {q['cases']} forecasts of {q['storms']} storms since {since}: Brier skill score "
            f"{_ci(q['bss'], pct=False)} against the base rate of {100 * q['base_rate']:.0f} %"
        )
        s += " — skilful." if q["skilful"] else " — no skill shown."
        if q.get("reliable"):
            s += " Within sampling error of reliable in every well-filled probability bin."
        return s

    r = js["recurvature"]
    out["recurvature"] = " ".join(
        [
            prob_line(
                f"recurving by day {TE.N_DAYS}", r["pooled"], "AIFS-ENS + IFS-ENS pooled", js["period"]["pooled"][0]
            ),
            prob_line(f"recurving by day {TE.N_DAYS}", r["ifs"], "IFS-ENS alone", js["period"]["ifs"][0]),
        ]
    )
    tm = [x for x in js["recurvature_timing"]["ifs"] if "mae_h" in x]
    if tm:
        out["recurvature"] += (
            " Turn dates (IFS-ENS, the median recurving member): mean absolute error "
            + ", ".join(
                f"{x['mae_h'][0]:.0f} h for turns {x['lead_h'][0] // 24}–{x['lead_h'][1] // 24} days ahead" for x in tm
            )
            + "."
        )
    p = js["post_tropical"]
    out["post_tropical"] = " ".join(
        [
            prob_line(
                f"becoming post-tropical by day {TE.N_DAYS}",
                p["pooled"],
                "AIFS-ENS + IFS-ENS pooled",
                js["period"]["pooled"][0],
            ),
            prob_line(f"becoming post-tropical by day {TE.N_DAYS}", p["ifs"], "IFS-ENS alone", js["period"]["ifs"][0]),
        ]
    )
    late = [x for x in js["post_tropical_timing"]["ifs"] if "bias_h" in x and x["bias_h"][1] is not None]
    if late and all(x["bias_h"][1] > 0 for x in late):  # stated only where every interval is above zero
        out["post_tropical"] += (
            " The median member's last fix comes after the best track's transition: on average "
            + ", ".join(f"{x['bias_h'][0]:.0f} h ({x['bias_h'][1]:.0f} to {x['bias_h'][2]:.0f})" for x in late)
            + f" for transitions 0–2, 2–4, 4–6 and 6–{TE.N_DAYS} days ahead."
            if len(late) == 4
            else " The median member's last fix comes after the best track's transition, by "
            + ", ".join(f"{x['bias_h'][0]:.0f} h" for x in late)
            + " on average."
        )

    def watch_line(w, who, since):
        if "hit_rate" not in w:
            return f"{who}: no track record yet ({w['watches']} watches on {w['storms_watched']} storms since {since})."
        return (
            f"{who} since {since}: {w['watches']} watches on {w['storms_watched']} storms; {_ci(w['hit_rate'])} were "
            f"followed by a recurvature by day {TE.WATCH_DAY}"
            + (f", and {_ci(w['detected'])} of the recurvatures were watched." if "detected" in w else ".")
        )

    out["watch"] = " ".join(
        [
            watch_line(js["watch"]["pooled"], "The watch rule on AIFS-ENS + IFS-ENS", js["period"]["pooled"][0]),
            watch_line(js["watch"]["ifs"], "On IFS-ENS alone", js["period"]["ifs"][0]),
        ]
    )
    t = js["track_error"]["ifs"]
    if 120 in t["leads"]:
        k = t["leads"].index(120)
        d, e, x = t["det_km"][k], t["ens_km"][k], t["ens_minus_det_km"][k]
        sig = x[1] is not None and (x[1] > 0 or x[2] < 0)
        out["tracks"] = (
            f"Day-5 track error since {js['period']['ifs'][0]} ({t['cases'][k]} forecasts of {t['storms'][k]} storms): "
            f"IFS control {d[0]:.0f} km, IFS-ENS mean {e[0]:.0f} km; the difference ({x[0]:+.0f} km, 95 % interval "
            f"{x[1]:+.0f} to {x[2]:+.0f}) is {'significant' if sig else 'not significant'}."
        )
    return out


def score(a) -> int:
    storms, trk = load_tracks(Path(a.tracks))
    bt = load_ibtracs(Path(a.ibtracs))
    print(f"  {len(storms)} forecast storms, {len(bt)} best tracks", flush=True)
    C, E = cases_table(storms, trk, bt)
    print(f"  {len(C)} matched forecasts, {C['storm'].nunique()} storms; {len(E)} track-error pairs", flush=True)
    P = pooled(C)
    js = dict(
        built=f"{pd.Timestamp.now('UTC'):%Y-%m-%dT%H:%MZ}",
        best_track=f"IBTrACS v04r01 last-three-years file ({a.ibtracs_date or 'date not recorded'})",
        period={m: _span(C[C["model"] == m]) for m in MODELS} | {"pooled": _span(C[C["model"] == "aifs"])},
        rules=dict(
            match_km=MATCH_KM,
            horizon_days=TE.N_DAYS,
            recurvature="tclib.recurvature with 48 h of observed history; storms already turned at the start left out",
            post_tropical=f"tracker's last fix north of {TE.PT_LAT:.0f}N by day {TE.N_DAYS} vs IBTrACS nature ET (final best tracks only)",
            intervals=f"95 %, {BOOT} resamples of whole storms",
            min_storms=MIN_STORMS,
            min_events=MIN_EVENTS,
        ),
        track_error={m: track_scores(E, m) for m in MODELS},
        recurvature={m: prob_scores(C[C["model"] == m], "p_rec", "rec_obs") for m in MODELS}
        | {"pooled": prob_scores(P, "p_rec", "rec_obs")},
        recurvature_timing={m: timing(C[C["model"] == m], "turn_fc", "turn_obs") for m in MODELS},
        post_tropical={m: prob_scores(C[C["model"] == m], "p_pt", "pt_obs") for m in MODELS}
        | {"pooled": prob_scores(P, "p_pt", "pt_obs")},
        post_tropical_timing={m: timing(C[C["model"] == m], "pt_fc", "pt_time") for m in MODELS},
        watch={m: watch_scores(C[C["model"] == m]) for m in MODELS} | {"pooled": watch_scores(P)},
    )
    js["summary"] = summary(js)
    for k, v in js["summary"].items():
        print(f"  {k}: {v}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(js, indent=1, default=lambda x: None))
    if a.figure:
        render(js, Path(a.figure))
    return 0


def render(js: dict, out: Path):
    """Track error by lead (IFS and AIFS pairs, each on its own sample) and the reliability of the two chances."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2, figsize=(14.5, 9.4))
    fig.subplots_adjust(left=0.06, right=0.985, top=0.9, bottom=0.13, hspace=0.42, wspace=0.18)
    fig.suptitle(
        "Track record of the TC board — ECMWF tracker on IFS-ENS, IFS control, AIFS-ENS and AIFS single vs IBTrACS best tracks",
        fontsize=13.5,
        fontweight="bold",
        x=0.06,
        ha="left",
    )
    for ax, m, labs in (
        (axs[0][0], "ifs", ("IFS control", "IFS-ENS mean")),
        (axs[0][1], "aifs", ("AIFS single", "AIFS-ENS mean")),
    ):
        t = js["track_error"][m]
        if not t["leads"]:
            ax.text(
                0.5,
                0.5,
                f"{labs[1].split()[0]}: no track record yet",
                ha="center",
                transform=ax.transAxes,
                color="#6f6b64",
            )
            ax.set_axis_off()
            continue
        d = np.array(t["leads"]) / 24
        for key, lab, col in (
            ("det_km", labs[0], "#111"),
            ("ens_km", labs[1], "#c0392b"),
            ("pers_km", "persistence", "#9a9a9a"),
        ):
            v = np.array(t[key], float)
            ax.plot(d, v[:, 0], "-o", color=col, lw=2, ms=4, label=lab)
            ax.fill_between(d, v[:, 1], v[:, 2], color=col, alpha=0.15, lw=0)
        ax.set_ylim(0, None)
        ax.set_xlabel("lead (days)")
        ax.set_ylabel("mean track error (km)")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=9)
        p = js["period"][m]
        ax.set_title(
            f"{labs[1].split()[0]} tracks, {p[0]} to {p[1]}, {max(t['storms'])} storms (95 %: whole storms resampled)",
            fontsize=10,
            loc="left",
        )
    for ax, key, what in ((axs[1][0], "recurvature", "recurving"), (axs[1][1], "post_tropical", "post-tropical")):
        ax.plot([0, 1], [0, 1], color="#999", lw=1, ls="--")
        shown = False
        for m, lab, col in (("ifs", "IFS-ENS", "#c0392b"), ("pooled", "AIFS-ENS + IFS-ENS", "#1f5fa8")):
            q = js[key][m]
            if not q.get("track_record"):
                continue
            shown = True
            f = [r["forecast"] for r in q["reliability"]]
            o = np.array([r["observed"] for r in q["reliability"]], float)
            ax.errorbar(
                f,
                o[:, 0],
                yerr=[o[:, 0] - o[:, 1], o[:, 2] - o[:, 0]],
                fmt="o-",
                color=col,
                capsize=3,
                label=f"{lab}: BSS {q['bss'][0]:.2f} [{q['bss'][1]:.2f}, {q['bss'][2]:.2f}], {q['storms']} storms",
            )
        if not shown:
            ax.text(0.5, 0.5, "no track record yet", ha="center", transform=ax.transAxes, color="#6f6b64")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(f"forecast chance of {what} by day {TE.N_DAYS} (bin mean)")
        ax.set_ylabel("observed frequency (95 %)")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=8.6, loc="upper left")
        ax.set_title(
            f"Reliability: {what} by day {TE.N_DAYS}"
            + (
                " (storms already turned at the start left out)"
                if key == "recurvature"
                else " (best tracks with a recorded nature)"
            ),
            fontsize=10,
            loc="left",
        )
    fig.text(
        0.06,
        0.01,
        "Forecasts: ECMWF tropical-cyclone tracker (open data, CC BY 4.0), 00 and 12 UTC, perturbed members. Best tracks: IBTrACS v04r01 (NOAA NCEI).\n"
        "Track error while the best track is tropical or subtropical; persistence = the last 12 h of best-track motion carried on.\n"
        "Brier skill score against each storm's base rate from the other storms. Post-tropical = the tracker's last fix north "
        f"of {TE.PT_LAT:.0f}°N; observed = IBTrACS nature ET.",
        fontsize=8.2,
        color="#6f6b64",
        va="bottom",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=90, facecolor="white", pil_kwargs={"quality": 85, "method": 6})
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tracks", help="decode the tracker files of months start..end into monthly archives")
    t.add_argument("--start", default=FIRST)
    t.add_argument("--end", default=f"{dt.datetime.now(dt.UTC):%Y-%m}")
    t.add_argument("--out", required=True)
    t.add_argument("--procs", type=int, default=6)
    t.add_argument("--refresh", type=int, default=2, help="rebuild the last N months even when their archive exists")
    s = sub.add_parser("score", help="score the archives against IBTrACS")
    s.add_argument("--tracks", required=True)
    s.add_argument("--ibtracs", required=True)
    s.add_argument("--ibtracs-date", default="")
    s.add_argument("--out", required=True)
    s.add_argument("--figure", default="")
    a = ap.parse_args()
    return {"tracks": tracks, "score": score}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
