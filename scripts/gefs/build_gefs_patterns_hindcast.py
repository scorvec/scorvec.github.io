#!/usr/bin/env python3
"""Calibration file of the GEFS pattern products (laptop, once gefs-clim-v2 is out): gefs_patterns_calib.npz for
release gefs-ref-v1. The GEPS page's telecon_hindcast.py (+ --regimes) ported to the GEFSv12 reforecast (2026-09-26).

Source: the per-member DAILY 2.5 deg reforecast fields of release gefs-clim-v2 (gefs_hind2_MM.npz: every Wednesday
start 2000-2019, members c00 p01-p03, 500 hPa height, sea-level pressure, 100 hPa height). Every start is treated
exactly like a live run (gefs_patterns.member_anoms): member mean minus the lead-matched reforecast climatology at
the start's day of year (c25, linear between the two 5-day centres), minus the ERA5 1991-2020 shift at the valid date
(day d = start + d - 1), projected with the same patterns; the ensemble-mean regime per sector and lead likewise.

The c25 climatology is recomputed here from the same starts with the release's own arithmetic (gefs_daily.cmd_combine:
all member-runs of the starts within +-15 days of each centre, packed to int16) and checked against the released
gefs_clim2_<c>.npz (--check). As on the GEPS page the climatology includes the start being scored (1 of ~350
member-runs per centre), a <1% shrinkage of its anomaly.

Observed: NCEP/NCAR R1 daily means (the reanalysis the patterns are defined on), anomalies against the pattern file's
1991-2020 day-of-year climatology, on each valid date; regimes from the 5-day running mean, classified with the
START's season set (as the live product holds one set); the persistence baseline is the regime observed on the day
before day 1 with its window cut there (days D-2..D, the last day of the live observed strip) - no look-ahead, where
the GEPS side's centred window at the start date saw two days past it.

Outputs (gefs_patterns_calib.npz)
  fit            (index, 366 init doy, 5 weeks, FIT_COLS) float32: per init day of year, the starts within +-45 days,
                 weekly means of the hindcast-mean and observed index: n, r, a, b (obs = a fcst + b), s (residual sd,
                 ddof 2), obs_sd, fc_sd, n_eff (Bretherton et al. 1999: n (1 - r1 r2) / (1 + r1 r2), r1 r2 = lag-1
                 autocorrelations over consecutive weekly starts), p (one-sided t-test of r on n_eff - 2 dof)
  reg_S, reg_fc_<sector>, reg_ob_<sector>, reg_ob0_<sector>   regime labels (0..3, 4 = no regime, -1 missing)
  c1_<tag>       (73 centres, lat, 144) int16: the climatology at lead day 1 for the analysis tail (heights 90N -> 0
                 only, sea-level pressure global)
  meta_json      fit_ids, fit_cols, n_starts, provenance, and the synthetic cluster-test power (--cluster-power M,
                 kept in work/cluster_power.json and merged into every build)

    python build_gefs_patterns_hindcast.py --hind DIR [--check 266] [--cluster-power 31]
    python build_gefs_patterns_hindcast.py --rf2 DIR ...        # rf2_<start>.npz files instead (testing)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", message="Mean of empty slice")                  # weeks of undefined patterns

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gefs_patterns as GP                                           # noqa: E402
import gefs_daily as G                                               # noqa: E402

GEPS = Path.home() / "data_archive" / "geps_subx"
OUT = Path.home() / "data_archive" / "gefs_ref" / "patterns"
FIT_COLS = ["n", "r", "a", "b", "s", "obs_sd", "fc_sd", "n_eff", "p"]
WINDOW_DOY = 45
N_DAYS = GP.DAYS


# ── sources of per-start member fields ────────────────────────────────────────────────────────────────────────
def list_starts(a):
    """[(start 'YYYYMMDD', file, key prefix)] from --hind (gefs_hind2_MM.npz) or --rf2 (rf2_<start>.npz)."""
    out = []
    if a.hind:
        for f in sorted(Path(a.hind).glob("gefs_hind2_*.npz")):
            with np.load(f) as z:
                starts = sorted({k.split("_")[0] for k in z.files if k.endswith("_members")})
            out += [(s, str(f), f"{s}_") for s in starts]
    if a.rf2:
        out += [(f.stem[4:], str(f), "") for f in sorted(Path(a.rf2).glob("rf2_*.npz"))]
    return sorted(out)


def member_sum(item):
    """(start, {tag: member SUM (35, 73, 144)}, n members) - pass 1."""
    s, f, pre = item
    with np.load(f) as z:
        n = len(z[f"{pre}members"])
        return s, {t: G.unpack(t, z[f"{pre}m_{t}"]).sum(0) for t in GP.TAGS}, n


def centres_of(start: str):
    d = GP.doy(GP.parse_date(start))
    return [c for c in GP.CENTRES if min(abs(d - c), 365 - abs(d - c)) <= 15]


def build_clim(starts, workers):
    """c25 at the 73 centres from the starts themselves (gefs_daily.cmd_combine's arithmetic), packed like the
    release -> {tag: (73, 35, 73, 144) float64 unpacked}, member-runs per centre."""
    acc = {t: np.zeros((len(GP.CENTRES), N_DAYS, 73, 144)) for t in GP.TAGS}
    n = np.zeros(len(GP.CENTRES))
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        for i, (s, sums, k) in enumerate(ex.map(member_sum, starts, chunksize=4), 1):
            for c in centres_of(s):
                j = GP.centre_index(c)
                for t in GP.TAGS:
                    acc[t][j] += sums[t]
                n[j] += k
            if i % 100 == 0:
                print(f"  climatology pass: {i}/{len(starts)} starts, {time.time() - t0:.0f} s", flush=True)
    out = {}
    for t in GP.TAGS:
        with np.errstate(invalid="ignore"):
            out[t] = G.unpack(t, G.pack(t, acc[t] / np.maximum(n, 1)[:, None, None, None]))
        out[t][n == 0] = np.nan
    return out, n


_REF = _CLIM = None


def _init(ref_path, clim_file):
    global _REF, _CLIM
    _REF = GP.Ref(ref_path)
    with np.load(clim_file) as z:                                    # into memory once per worker
        _CLIM = {k: z[k] for k in z.files}


def score_start(item):
    """Hindcast-mean indices (16, 35) and regime labels {sector: (35,)} of one start."""
    s, f, pre = item
    ref, C = _REF, _CLIM
    base = GP.parse_date(s)
    days = GP.valid_days(base)
    lo, hi, w = GP.bracket(GP.doy(base))
    fields = {}
    with np.load(f) as z:
        for t in GP.TAGS:
            m = G.unpack(t, z[f"{pre}m_{t}"]).mean(0)
            c = (1 - w) * C[t][GP.centre_index(lo)] + w * C[t][GP.centre_index(hi)]
            sh = ref.shift(t, days)
            fields[t] = ((m - c - (sh if sh is not None else 0.0)) * GP.UNITS[t])[:, None]      # (35, 1, 73, 144)
    idx = GP.indices_from(ref, fields, days, quiet=True)
    F = np.full((len(ref.ids), N_DAYS), np.nan, "float32")
    for i, sid in enumerate(ref.ids):
        if sid in idx:
            F[i] = idx[sid][:, 0]
    season = GP.season_of(base.month)
    labs = {}
    z500 = fields["zg500"][:, 0]
    if np.isfinite(z500).all():
        for sk in GP.SECTORS:
            sec, _, _ = GP.sector_cut(z500, GP.SECTORS[sk])
            lab, _ = ref.classify(GP.running_partial(sec, GP.SMOOTH, axis=0), sk, season)
            labs[sk] = lab.astype("int8")
    return s, F, labs


# ── observations (NCEP/NCAR R1) ───────────────────────────────────────────────────────────────────────────────
def observed(ref, years):
    """{sid: {date: value}} daily observed indices, and {(sector, season): {date: label}} regimes."""
    import xarray as xr
    sys.path.insert(0, str(GEPS))
    import build_telecon_patterns as b                               # the GEPS side's NCEP reader (read-only)
    pat = xr.open_dataset(GEPS / "telecon" / "patterns.nc")
    obs, reg = {}, {}
    for var, level, fld, ck in (("hgt", 500, "z500", "clim_doy_z500"), ("slp", None, "slp", "clim_doy_slp"),
                                ("hgt", 100, "z100", "clim_doy_z100")):
        a = b.daily(var, years, level)
        assert np.allclose(a.lat.values, GP.LAT25) and np.allclose(a.lon.values, GP.LON25)
        t = [dt.date(int(str(x)[:4]), int(str(x)[5:7]), int(str(x)[8:10])) for x in a.time.values]
        dy = np.array([GP.doy(d) for d in t]); mo = np.array([d.month for d in t])
        an = a.values.astype("float64") - pat[ck].values[dy - 1]
        for sid in ref.ids:
            if ref.pat[sid]["field"] == fld:
                obs[sid] = dict(zip(t, ref.index(sid, an, mo, dy)))
        if fld == "z500":
            for sk in GP.SECTORS:
                sec, _, _ = GP.sector_cut(an, GP.SECTORS[sk])
                v = GP.running_partial(sec, GP.SMOOTH, axis=0)
                # the regime a forecaster sees at init: the observed strip's last day, whose 5-day window is cut
                # at that day (days D-2..D, as running_partial ends a series) - no look-ahead past the start
                h = GP.SMOOTH // 2
                tr = np.stack([sec[max(0, i - h):i + 1].mean(0) for i in range(len(sec))])
                for season in GP.SEASONS:
                    lab, _ = ref.classify(v, sk, season)
                    reg[(sk, season)] = dict(zip(t, lab.astype("int8")))
                    lab0, _ = ref.classify(tr, sk, season)
                    reg[(sk, season, "trail")] = dict(zip(t, lab0.astype("int8")))
        print(f"  observed {fld}: {len(t)} days {t[0]} .. {t[-1]}", flush=True)
    return obs, reg


# ── fits ──────────────────────────────────────────────────────────────────────────────────────────────────────
def lag1(x, nxt):
    """Correlation of x with its value at the next weekly start (pairs where one exists)."""
    ok = nxt >= 0
    if ok.sum() < 10:
        return 0.0
    a, b_ = x[ok], x[nxt[ok]]
    good = np.isfinite(a) & np.isfinite(b_)
    if good.sum() < 10 or np.std(a[good]) == 0 or np.std(b_[good]) == 0:
        return 0.0
    return float(np.corrcoef(a[good], b_[good])[0, 1])


def fit_table(F, O, S):
    """(index, 366, 5, FIT_COLS) from paired hindcast-mean / observed daily indices F, O (index, start, day)."""
    from scipy.stats import t as tdist
    dates = [GP.parse_date(s) for s in S]
    sdoy = np.array([GP.doy(d) for d in dates])
    out = np.full((F.shape[0], 366, len(GP.WEEKS), len(FIT_COLS)), np.nan, "float32")
    wk_f = np.stack([np.nanmean(F[:, :, w0 - 1:w1], 2) for w0, w1 in GP.WEEKS], 2)       # (index, start, week)
    wk_o = np.stack([np.nanmean(O[:, :, w0 - 1:w1], 2) for w0, w1 in GP.WEEKS], 2)
    for d in range(1, 367):
        win = np.abs(((sdoy - d + 183) % 366) - 183) <= WINDOW_DOY
        wi = np.where(win)[0]
        pos = {dates[i]: k for k, i in enumerate(wi)}
        nxt = np.array([pos.get(dates[i] + dt.timedelta(days=7), -1) for i in wi])
        for ii in range(F.shape[0]):
            for w in range(len(GP.WEEKS)):
                f, o = wk_f[ii, wi, w], wk_o[ii, wi, w]
                ok = np.isfinite(f) & np.isfinite(o)
                if ok.sum() < 30:
                    continue
                r1, r2 = lag1(np.where(ok, f, np.nan), nxt), lag1(np.where(ok, o, np.nan), nxt)
                f_, o_ = f[ok], o[ok]
                r = float(np.corrcoef(f_, o_)[0, 1])
                a_, b_ = np.polyfit(f_, o_, 1)
                s_ = float(np.std(o_ - (a_ * f_ + b_), ddof=2))
                n = int(ok.sum())
                neff = float(np.clip(n * (1 - r1 * r2) / (1 + r1 * r2), 3, n))
                tt = r * np.sqrt(max(neff - 2, 1) / max(1 - r * r, 1e-9))
                p = float(tdist.sf(tt, max(neff - 2, 1)))
                out[ii, d - 1, w] = [n, r, a_, b_, s_, o_.std(), f_.std(), neff, p]
    return out


# ── cluster-test power (synthetic ensembles) ──────────────────────────────────────────────────────────────────
def _power_trial(args):
    import gefs_clusters as GC
    X, w, sep, seed = args
    rng = np.random.default_rng(seed)
    M = X.shape[0]
    if sep > 0:
        Z = (X * w).reshape(M, -1); D = Z - Z.mean(0)
        u, sv, vt = np.linalg.svd(D, full_matrices=False)
        e1 = vt[0].reshape(X.shape[1:]) / w                          # EOF1 in field units
        sd1 = sv[0] / np.sqrt(M - 1)                                   # sd of the members along it
        half = rng.permutation(M)[: M // 2]
        sgn = np.where(np.isin(np.arange(M), half), 0.5, -0.5)
        X = X + (sgn * sep * sd1)[:, None, None] * e1[None]
    k, lab, significant, tests = GC.cluster_week(X, w, seed=seed)
    return sep, significant, k


def cluster_power(ref, members, trials, workers, mode="gauss"):
    """Detection rate of the cluster test for synthetic ensembles of `members` weekly 500 hPa maps, NA sector: a
    continuum (separation 0) and two equal groups separated by `sep` member sd along the ensemble's EOF1 (the GEPS
    page's calibration recipe). mode "gauss": Gaussian members with the covariance of observed NCEP R1 weekly
    anomalies (Oct-Mar 1991-2020) - the continuum the rotation null assumes; mode "real": members drawn from the
    observed weekly maps of one season window (+-21 d of Jan 15), to see what real non-Gaussianity does to it."""
    import xarray as xr
    sys.path.insert(0, str(GEPS))
    import build_telecon_patterns as b
    pat = xr.open_dataset(GEPS / "telecon" / "patterns.nc")
    a = b.daily("hgt", range(1991, 2021), 500)
    t = [dt.date(int(str(x)[:4]), int(str(x)[5:7]), int(str(x)[8:10])) for x in a.time.values]
    an = a.values.astype("float64") - pat["clim_doy_z500"].values[np.array([GP.doy(d) for d in t]) - 1]
    sec, lat, lon = GP.sector_cut(an, GP.SECTORS["na"])
    nw = len(t) // 7
    wk = sec[: nw * 7].reshape(nw, 7, *sec.shape[1:]).mean(1)
    wmon = np.array([t[i * 7 + 3].month for i in range(nw)])
    w = np.sqrt(np.cos(np.deg2rad(lat)))[:, None] * np.ones((1, lon.size))
    rng = np.random.default_rng(3)
    seps = (0.0, 1.0, 2.0, 3.0, 4.0)
    if mode == "gauss":
        pool = wk[np.isin(wmon, (10, 11, 12, 1, 2, 3))]
        D = (pool - pool.mean(0)).reshape(len(pool), -1)
        draw = lambda: (rng.standard_normal((members, len(pool))) @ D / np.sqrt(len(pool))).reshape(members, *pool.shape[1:])
    else:
        wday = np.array([GP.doy(t[i * 7 + 3]) for i in range(nw)])
        pool = wk[np.abs(((wday - 15 + 183) % 366) - 183) <= 21]
        draw = lambda: pool[rng.choice(len(pool), members, replace=False)]
    jobs = [(draw(), w, s, int(rng.integers(1 << 30))) for s in seps for _ in range(trials)]
    res = {s: [] for s in seps}
    with ProcessPoolExecutor(workers) as ex:
        for s, sig, k in ex.map(_power_trial, jobs):
            res[s].append(sig)
    rate = {s: float(np.mean(v)) for s, v in res.items()}
    print(f"  cluster power ({mode}), {members} members, {trials} trials: "
          + ", ".join(f"{s:.0f} sd {r * 100:.0f}%" for s, r in rate.items()), flush=True)
    xs, ys = np.array(seps[1:]), np.array([rate[s] for s in seps[1:]])
    def at(p):
        return float(np.interp(p, np.maximum.accumulate(ys), xs)) if ys.max() >= p else float("nan")
    return {"fp": rate[0.0], "rates": {f"{s:.0f}": r for s, r in rate.items()}, "sep_60": at(0.6), "sep_90": at(0.9),
            "trials": trials, "mode": mode, "pool": int(len(pool)),
            "source": ("Gaussian, covariance of NCEP R1 weekly 500 hPa anomalies, NA sector, Oct-Mar 1991-2020"
                       if mode == "gauss" else "NCEP R1 weekly 500 hPa anomalies within +-21 d of Jan 15, 1991-2020")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hind", help="directory of gefs_hind2_MM.npz (release gefs-clim-v2)")
    ap.add_argument("--rf2", help="directory of rf2_<start>.npz (testing)")
    ap.add_argument("--ref", default=str(OUT / GP.REF_NAME))
    ap.add_argument("--out", default=str(OUT / GP.CALIB_NAME))
    ap.add_argument("--work", default=str(Path.home() / "data_archive" / "gefs_port_test" / "patterns" / "work"))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--check", type=int, action="append", default=[],
                    help="compare the recomputed climatology with the released gefs_clim2_<centre>.npz")
    ap.add_argument("--cluster-power", type=int, action="append", default=[], help="member count(s) to test")
    ap.add_argument("--power-trials", type=int, default=40)
    ap.add_argument("--power-mode", default="gauss", choices=("gauss", "real"))
    ap.add_argument("--power-only", action="store_true", help="only compute the cluster power (work/cluster_power.json)")
    a = ap.parse_args()
    ref = GP.Ref(Path(a.ref))
    work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pj = work / "cluster_power.json"                                 # computed once, merged into every build
    power = json.loads(pj.read_text()) if pj.exists() else {}
    for m in a.cluster_power:
        key = str(m) if a.power_mode == "gauss" else f"{m}_{a.power_mode}"
        power[key] = cluster_power(ref, m, a.power_trials, a.workers, a.power_mode)
        pj.write_text(json.dumps(power, indent=1))
    if a.power_only:
        print(f"{pj}: cluster power for {sorted(power)} ({time.time() - t0:.0f} s)")
        return 0
    starts = list_starts(a)
    if not starts:
        raise SystemExit("no reforecast starts found")
    print(f"{len(starts)} starts {starts[0][0]} .. {starts[-1][0]}", flush=True)
    clim, n = build_clim(starts, a.workers)
    print(f"  climatology: {int((n > 0).sum())} centres, member-runs per centre {int(n[n > 0].min())}-{int(n.max())} "
          f"({time.time() - t0:.0f} s)", flush=True)
    for c in a.check:
        rel = GP.cached(GP.CLIM2_URL.format(c=c), work / f"gefs_clim2_{c:03d}.npz")
        with np.load(rel) as z:
            print(f"  check centre {c}: member-runs released {int(z['n'])}, recomputed {int(n[GP.centre_index(c)])}",
                  flush=True)
            for t in GP.TAGS:
                mine = clim[t][GP.centre_index(c)]
                if f"c25_{t}" in z.files:
                    d = G.unpack(t, z[f"c25_{t}"]) - mine
                    print(f"    {t}: max |recomputed - released c25| {np.nanmax(np.abs(d)):.3f}", flush=True)
                else:                                                # older release: its 1.5 deg d_ field, interpolated
                    from scipy.interpolate import RegularGridInterpolator
                    g = np.stack(np.meshgrid(GP.LAT25[::-1], GP.LON25, indexing="ij"), -1)
                    lat15, lon15 = np.arange(90.0, -90.1, -1.5), np.arange(0.0, 360.0, 1.5)
                    dd = []
                    for k in range(N_DAYS):
                        f = G.unpack(t, z[f"d_{t}"][k])[::-1]
                        f = np.concatenate([f, f[:, :1]], axis=1)
                        v = RegularGridInterpolator((lat15[::-1], np.r_[lon15, 360.0]), f)(g)[::-1]
                        dd.append(v - mine[k])
                    dd = np.abs(np.array(dd)) * GP.UNITS[t]
                    print(f"    {t}: no c25 in this release file; vs its 1.5 deg d_{t} interpolated: median "
                          f"{np.median(dd):.2f}, p99 {np.percentile(dd, 99):.2f} (m or hPa)", flush=True)
    clim_file = work / "clim25_all.npz"
    np.savez(clim_file, **{t: v.astype("float32") for t, v in clim.items()})
    S, F, LAB = [], [], {sk: [] for sk in GP.SECTORS}
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(Path(a.ref), clim_file)) as ex:
        for i, (s, f, labs) in enumerate(ex.map(score_start, starts, chunksize=4), 1):
            S.append(s); F.append(f)
            for sk in GP.SECTORS:
                LAB[sk].append(labs.get(sk, np.full(N_DAYS, -1, "int8")))
            if i % 100 == 0:
                print(f"  scoring pass: {i}/{len(starts)} ({time.time() - t0:.0f} s)", flush=True)
    F = np.stack(F, 1)                                               # (index, start, day)
    years = sorted({int(s[:4]) for s in S})
    obs, reg = observed(ref, list(range(years[0], years[-1] + 2)))
    O = np.full_like(F, np.nan)
    for ii, sid in enumerate(ref.ids):
        if sid not in obs:
            continue
        for k, s in enumerate(S):
            O[ii, k] = [obs[sid].get(d, np.nan) for d in GP.valid_days(GP.parse_date(s))]
    save = {"fit": fit_table(F, O, S), "reg_S": np.array([int(s) for s in S], "int32")}
    for sk in GP.SECTORS:
        fc = np.stack(LAB[sk]); ob = np.full_like(fc, -1); ob0 = np.full(len(S), -1, "int8")
        for k, s in enumerate(S):
            base = GP.parse_date(s)
            o = reg[(sk, GP.season_of(base.month))]
            ob[k] = [o.get(d, -1) for d in GP.valid_days(base)]
            ob0[k] = reg[(sk, GP.season_of(base.month), "trail")].get(base - dt.timedelta(days=1), -1)
        save[f"reg_fc_{sk}"], save[f"reg_ob_{sk}"], save[f"reg_ob0_{sk}"] = fc, ob, ob0
        okk = (fc >= 0) & (ob >= 0)
        print(f"  regimes {sk}: hit day 1/7/14/21/28 " + " ".join(
            f"{np.mean((fc[:, j] == ob[:, j])[okk[:, j]]):.2f}" for j in (0, 6, 13, 20, 27)), flush=True)
    nh = int((GP.LAT25 >= 0).sum())                                  # heights: the tail only needs the north
    for t in GP.TAGS:
        c1 = G.pack(t, np.nan_to_num(clim[t][:, 0], nan=G.SCALE[t][0]))
        save[f"c1_{t}"] = c1[:, :nh] if t != "mslp" else c1
    meta = {"fit_ids": list(ref.ids), "fit_cols": FIT_COLS, "n_starts": len(S), "first": S[0], "last": S[-1],
            "window_doy": WINDOW_DOY, "valid_convention": "day d valid on start + (d - 1)",
            "observed": "NCEP/NCAR R1 daily, anomalies vs the patterns' 1991-2020 day-of-year climatology",
            "built": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%MZ"), "cluster_power": power}
    save["meta_json"] = np.array(json.dumps(meta))
    np.savez_compressed(a.out, **save)
    np.savez_compressed(work / "hindcast_pairs.npz", F=F, O=O, S=np.array(S), ids=np.array(ref.ids))
    for ii, sid in enumerate(ref.ids):
        ok = np.isfinite(F[ii]) & np.isfinite(O[ii])
        if ok[:, 0].sum() > 30:
            r = [np.corrcoef(F[ii][ok[:, j], j], O[ii][ok[:, j], j])[0, 1] for j in (0, 6, 13, 20, 27)]
            print(f"  {sid:6s} r(day 1/7/14/21/28) " + " ".join(f"{x:+.2f}" for x in r), flush=True)
    print(f"{a.out}  {Path(a.out).stat().st_size / 1e6:.2f} MB  ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
