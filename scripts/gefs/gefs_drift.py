#!/usr/bin/env python3
"""Operational-minus-reforecast drift correction for the GEFS page (2026-09-26; user: "I think that the GEFS has a
strange cool drift at long lead time").

The page removes GEFS's lead-dependent drift with its own 2000-2019 reforecast (gefs_daily / gefs-clim-v2). That is
only right if today's operational runs drift the way the reforecast did. They do not: verified against GEFS's own
analyses (21 August 2026 runs, control), from day 1 to day 35 the operational run cools the NH 30-70 surface by
0.59 K where the reforecast cooled 0.14 K, lowers 500 hPa there by 14 m where the reforecast raised it 6 m, and warms
10 hPa by 0.8 K where the reforecast warmed 3.1 K. Differenced against the reforecast climate, every product therefore
reads spuriously cold at long lead in the northern mid-latitudes and aloft.

The correction, per field, lead and latitude (zonal means - the mismatch is planetary-scale, and a 2-D estimate from a
few dozen runs would be weather noise):

  for each verified init i (all 35 days observed), with A = GEFS's own analyses on the valid day (the 12Z and next
  00Z control analyses, the forecast day's two samples; for precipitation and OLR, which analyses do not carry, the
  day-1 forecast of the run started that day stands in - a stationary day-1 bias cancels in the difference below):
      E_i(L)  = zm[F_i(L)] - zm[A(v_L)]                            operational error vs its analyses
      R_i(L)  = zm[M(L) - M(1)] - zm[O(v_L) - O(v_1)]              the reforecast's drift vs ERA5 (O = ERA5 normals)
      D_i(L)  = [E_i(L) - E_i(1)] - R_i(L)                         the drift the reforecast does not know about
  Delta(L) = mean over the trailing WINDOW of verified inits, smoothed (3 latitudes x 3 days), shrunk toward 0 by
  D^2 / (D^2 + SE^2) with SE from an effective sample size corrected for the overlap of successive runs; Delta(1) = 0.

Every live anomaly then becomes  [F - M(L) - shift] - Delta(L): the reforecast's calibrations (terciles, index
probabilities, MJO skill) were fitted where Delta = 0 by construction, so this also puts the live run back in the frame
they were fitted in.

    python gefs_drift.py backfill --until YYYYMMDD --n 45 --runs DIR     # verify a window of past inits (one-off, Actions)
    python gefs_drift.py estimate --date YYYYMMDD --runs DIR              # verify the newest init, estimate Delta
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gefs_daily as G                                                # noqa: E402
import gefs_reforecast as R                                           # noqa: E402

TAGS = tuple(G.D2)                                                    # the 15 reforecast fields
ACC = ("pr", "olr")
INST = tuple(t for t in TAGS if t not in ACC)
LAG = 36                   # an init is verifiable once its day 35 (init + 34) has an analysis: init <= today - 36
WINDOW = 45                # trailing inits in the estimate
ZN = HERE / "data" / "era5_zonal_normals.npz"
S3 = "https://noaa-gefs-pds.s3.amazonaws.com"


def _url(day: dt.date, hh: str, mem: str, h: int):
    return f"{S3}/gefs.{day:%Y%m%d}/{hh}/atmos/pgrb2ap5/{mem}.t{hh}z.pgrb2a.0p50.f{h:03d}"


def _fetch(jobs):
    with ThreadPoolExecutor(24) as ex:
        return list(ex.map(lambda j: R.get(j[0], (j[1], j[2])), jobs))


def analysis_zm(day: dt.date, hh: str, cache: Path):
    """Zonal means (121,) of the control's f000 analysis of one cycle, instantaneous fields only (cached)."""
    p = cache / "an" / f"an_{day:%Y%m%d}{hh}.npz"
    if p.exists():
        return dict(np.load(p))
    url = _url(day, hh, "gec00", 0)
    idx = R.index(url); jobs, tags = [], []
    for t in INST:
        name, lev = G.D2[t][4], G.D2[t][2]
        hit = [e for e in idx if e[4] == name and e[2] == lev and e[3] == "anl"]
        if not hit:
            raise KeyError(f"{t} not in {url}")
        jobs.append((url, hit[0][0], hit[0][1])); tags.append(t)
    out = {t: G.box(G.to_half(R.decode(b)), 3).mean(1).astype("float32") for t, b in zip(tags, _fetch(jobs))}
    p.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(p, **out)
    return out


def day1_acc_zm(day: dt.date, cache: Path):
    """Day-1 precipitation and OLR zonal means from the 00Z control of `day` (the stand-in truth for fields the
    analyses do not carry)."""
    p = cache / "an" / f"d1_{day:%Y%m%d}.npz"
    if p.exists():
        return dict(np.load(p))
    acc = G.Acc(ACC); jobs, meta = [], []
    for h in (6, 12, 18, 24):
        url = _url(day, "00", "gec00", h); idx = R.index(url)
        for t in ACC:
            name, lev, kind = G.D2[t][4], G.D2[t][2], G.D2[t][3]
            hit = [e for e in idx if e[4] == name and e[2] == lev and e[3] == f"{h - 6}-{h} hour {kind} fcst"]
            jobs.append((url, hit[0][0], hit[0][1])); meta.append(t)
    for t, b in zip(meta, _fetch(jobs)):
        acc.add(t, 1, R.decode(b))
    # Acc wants all 35 days; only day 1 was filled, so read the accumulators directly
    out = {t: (acc.d[t][0] if G.D2[t][3] == "acc" else acc.d[t][0] / max(acc.n[t][0], 1)).mean(1).astype("float32") for t in ACC}
    p.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(p, **out)
    return out


def day_truth(v: dt.date, cache: Path):
    """The valid day's truth: the 12Z analysis of v and the 00Z analysis of v+1 (the forecast day's two samples)."""
    a, b = analysis_zm(v, "12", cache), analysis_zm(v + dt.timedelta(days=1), "00", cache)
    out = {t: 0.5 * (a[t] + b[t]) for t in INST}
    out.update(day1_acc_zm(v, cache))
    return out


def forecast_zm(init: dt.date, cache: Path, runs: Path):
    """(15, 35, 121) zonal means of the run's daily fields: the ensemble mean archived by the live job
    (gefs_fzm_<date>.npz) when there is one, else the control fetched now."""
    p = runs / f"gefs_fzm_{init:%Y%m%d}.npz"
    if p.exists():
        return np.load(p)["zm"]
    q = cache / "fc" / f"fzm_{init:%Y%m%d}.npz"
    if q.exists():
        return np.load(q)["zm"]
    jobs, meta = [], []
    for h in range(6, 841, 6):
        url = _url(init, "00", "gec00", h); idx = R.index(url)
        want = [(t, f"{h} hour fcst") for t in INST] if h % 12 == 0 else []
        want += [(t, f"{h - 6}-{h} hour {G.D2[t][3]} fcst") for t in ACC]
        for t, stext in want:
            name, lev = G.D2[t][4], G.D2[t][2]
            hit = [e for e in idx if e[4] == name and e[2] == lev and e[3] == stext]
            jobs.append((url, hit[0][0], hit[0][1])); meta.append((t, (h + 23) // 24))
    acc = G.Acc(TAGS)
    for (t, d), b in zip(meta, _fetch(jobs)):
        acc.add(t, d, R.decode(b))
    r = acc.finish()
    zm = np.stack([r[f"d_{t}"].mean(2) for t in TAGS]).astype("float32")
    q.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(q, zm=zm)
    return zm


def _normals():
    z = np.load(ZN)
    doy = z["doy"].astype(float)
    return doy, {t: z[f"z_{t}"].astype("float64") for t in TAGS}


def _at_doy(doy, f, d: dt.date):
    t = float(d.timetuple().tm_yday)
    x = np.r_[doy - 365.0, doy, doy + 365.0]; ff = np.concatenate([f, f, f])
    j = int(np.searchsorted(x, t)); w = (t - x[j - 1]) / (x[j] - x[j - 1])
    return (1 - w) * ff[j - 1] + w * ff[j]


def verify(init: dt.date, cache: Path, runs: Path):
    """D_i(L) for one init -> runs/gefs_verify_<init>.npz (15, 35, 121) float32."""
    import gefs_live as L
    p = runs / f"gefs_verify_{init:%Y%m%d}.npz"
    if p.exists():
        return np.load(p)["D"]
    F = forecast_zm(init, cache, runs)
    days = [init + dt.timedelta(days=k) for k in range(35)]
    with ThreadPoolExecutor(6) as ex:
        T = list(ex.map(lambda v: day_truth(v, cache), days))
    C2 = L.clim2_at(init, cache)
    doy, O = _normals()
    D = np.zeros((len(TAGS), 35, len(R.LAT)), "float32")
    for k, t in enumerate(TAGS):
        E = F[k] - np.stack([T[i][t] for i in range(35)])
        M = C2[f"d_{t}"].mean(2)
        Ov = np.stack([_at_doy(doy, O[t], d) for d in days])
        Rf = (M - M[0]) - (Ov - Ov[0])
        D[k] = (E - E[0]) - Rf
    runs.mkdir(parents=True, exist_ok=True); np.savez_compressed(p, D=D, init=np.array(f"{init:%Y%m%d}"))
    return D


def _smooth(a):
    """3-point running mean in latitude and in lead (edges held), then day 1 forced back to 0."""
    from scipy.ndimage import uniform_filter1d
    b = uniform_filter1d(a, 3, axis=-1, mode="nearest")
    b = uniform_filter1d(b, 3, axis=-2, mode="nearest")
    b[..., 0, :] = 0.0
    return b


def estimate(today: dt.date, cache: Path, runs: Path, window=WINDOW, fetch=True):
    """Delta (15, 35, 121) from the trailing `window` verifiable inits; writes runs/gefs_drift_<today>.npz and returns
    it with its diagnostics. Inits whose verification cannot be built (missing analyses) are skipped."""
    last = today - dt.timedelta(days=LAG)
    inits = [last - dt.timedelta(days=k) for k in range(window)]
    Ds, got = [], []
    for i in inits:
        try:
            Ds.append(verify(i, cache, runs) if fetch else np.load(runs / f"gefs_verify_{i:%Y%m%d}.npz")["D"])
            got.append(i)
        except Exception as e:                                       # noqa: BLE001
            print(f"  verify {i}: {str(e)[:80]}", flush=True)
    if len(Ds) < 15:
        raise SystemExit(f"only {len(Ds)} verified inits - no drift estimate")
    D = np.stack(Ds).astype("float64")                               # (n, 15, 35, 121), newest first
    n = len(D)
    m = D.mean(0)
    # effective sample size: successive inits share most of their valid days, so they are not independent
    late = D[:, :, 20:, :].mean(2)                                   # (n, 15, 121)
    a0, a1 = late[:-1] - late[:-1].mean(0), late[1:] - late[1:].mean(0)
    rho = np.clip((a0 * a1).sum(0) / np.sqrt((a0 ** 2).sum(0) * (a1 ** 2).sum(0) + 1e-12), 0.0, 0.95)   # (15, 121)
    neff = n * (1 - rho) / (1 + rho)
    se = D.std(0, ddof=1) / np.sqrt(np.maximum(neff, 2.0))[:, None, :]
    sm, ses = _smooth(m), _smooth(se)
    shrink = sm ** 2 / (sm ** 2 + ses ** 2 + 1e-12)
    delta = (sm * shrink).astype("float32")
    out = runs / f"gefs_drift_{today:%Y%m%d}.npz"
    np.savez_compressed(out, delta=delta, raw=m.astype("float32"), se=se.astype("float32"), neff=neff.astype("float32"),
                        tags=np.array(TAGS), lat=R.LAT, inits=np.array([f"{i:%Y%m%d}" for i in got]))
    print(f"  drift estimate from {n} inits {got[-1]} .. {got[0]} -> {out.name}", flush=True)
    return load(out)


# ── use ─────────────────────────────────────────────────────────────────────────────────────────────────────────
class Drift:
    """Delta by tag: .zonal(tag) -> (35, 121) on the 1.5 deg latitudes; .on(tag, lats) -> (35, len(lats))."""

    def __init__(self, z):
        self.tags = [str(t) for t in z["tags"]]
        self.delta = z["delta"].astype("float64")
        self.lat = z["lat"].astype("float64")
        self.inits = [str(x) for x in z["inits"]]

    def zonal(self, tag):
        return self.delta[self.tags.index(tag)]

    def on(self, tag, lats):
        d = self.zonal(tag); o = np.argsort(self.lat)
        return np.stack([np.interp(np.asarray(lats, float), self.lat[o], row[o]) for row in d])

    def field(self, tag, nlon=len(R.LON)):
        """(35, 121, nlon): the zonal correction broadcast over longitude, for subtracting from 1.5 deg fields."""
        return np.repeat(self.zonal(tag)[:, :, None], nlon, axis=2)

    def weekly(self, tag):
        return np.stack([self.zonal(tag)[a - 1:b].mean(0) for a, b in R.WEEKS])


def load(p: Path | None):
    return Drift(np.load(p)) if p is not None and Path(p).exists() else None


def from_env():
    """The drift estimate named by $GEFS_DRIFT (set by the daily job after gefs_live writes it), or None."""
    import os
    q = os.environ.get("GEFS_DRIFT")
    d = load(Path(q)) if q else None
    print(f"  drift correction: {'from ' + str(len(d.inits)) + ' verified runs' if d else 'none (GEFS_DRIFT unset or missing)'}",
          flush=True)
    return d


def latest(runs: Path, today: dt.date | None = None):
    """The newest drift estimate in the run archive (at most 7 days old), or None."""
    cands = sorted(runs.glob("gefs_drift_*.npz"))
    if not cands:
        return None
    p = cands[-1]
    if today is not None:
        d = dt.datetime.strptime(p.stem.split("_")[-1], "%Y%m%d").date()
        if (today - d).days > 7:
            return None
    return load(p)


def archive_forecast(z, date: str, runs: Path):
    """Keep the run's ensemble-mean daily zonal means, so it is verified 36 days from now without a re-fetch."""
    zm = np.stack([z[f"e_{t}"].mean(2) for t in TAGS]).astype("float32")
    np.savez_compressed(runs / f"gefs_fzm_{date}.npz", zm=zm)


def prune(runs: Path, today: dt.date):
    """verify / fzm files older than the window (plus margin), drift estimates older than a week."""
    for p in runs.glob("gefs_verify_*.npz"):
        if (today - dt.datetime.strptime(p.stem.split("_")[-1], "%Y%m%d").date()).days > LAG + WINDOW + 5:
            p.unlink()
    for p in runs.glob("gefs_fzm_*.npz"):
        if (today - dt.datetime.strptime(p.stem.split("_")[-1], "%Y%m%d").date()).days > LAG + 5:
            p.unlink()
    for p in runs.glob("gefs_drift_*.npz"):
        if (today - dt.datetime.strptime(p.stem.split("_")[-1], "%Y%m%d").date()).days > 7:
            p.unlink()


def figure(Dr: Drift, today: dt.date, out: Path):
    """The correction itself, for the page: lead x latitude for six fields."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = [("t2m", "2 m temperature", "K"), ("zg500", "500 hPa height", "m"), ("mslp", "sea-level pressure", "hPa"),
            ("t100", "100 hPa temperature", "K"), ("t10", "10 hPa temperature", "K"), ("u10", "10 hPa zonal wind", "m/s")]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.86, bottom=0.12, hspace=0.32, wspace=0.12)
    days = np.arange(1, 36)
    for ax, (t, name, unit) in zip(axes.ravel(), show):
        d = Dr.zonal(t) * (0.01 if t == "mslp" else 1.0)
        lim = max(float(np.nanpercentile(np.abs(d), 99)), 1e-3)
        m = ax.pcolormesh(days, Dr.lat, d.T, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="nearest")
        ax.set_title(f"{name} ({unit})", fontsize=10.5, fontweight="bold", loc="left")
        cb = fig.colorbar(m, ax=ax, fraction=0.05, pad=0.02); cb.ax.tick_params(labelsize=7.5)
        ax.set_yticks([-60, -30, 0, 30, 60]); ax.grid(alpha=0.25, lw=0.5); ax.tick_params(labelsize=8)
    for ax in axes[1]:
        ax.set_xlabel("forecast day")
    for ax in axes[:, 0]:
        ax.set_ylabel("latitude")
    fig.suptitle(f"GEFS drift not in its reforecast — the correction applied to every product · {today:%Y-%m-%d}",
                 fontsize=13.5, fontweight="bold", y=0.975)
    fig.text(0.06, 0.915, f"From the {len(Dr.inits)} most recent runs whose 35 days have been observed "
             f"({Dr.inits[-1][:4]}-{Dr.inits[-1][4:6]}-{Dr.inits[-1][6:]} to {Dr.inits[0][:4]}-{Dr.inits[0][4:6]}-{Dr.inits[0][6:]}), "
             "each against GEFS's own analyses.\nZonal mean; blue = the operational model runs colder / lower than its "
             "2000–2019 reforecast did at that lead, so the page adds this back.", fontsize=8.6, color="#5c6b73",
             va="center", linespacing=1.4)
    fig.text(0.06, 0.03, "Operational error growth (forecast minus analysis, relative to day 1) minus the reforecast's drift "
             "against ERA5 normals; smoothed over 3 days and 3 latitudes and shrunk toward zero where the run-to-run spread "
             "makes it uncertain (effective sample size corrected for overlapping runs). Day 1 is zero by construction.",
             fontsize=8, color="#6f6b64", wrap=True)
    p = out / "gefs_drift.webp"
    fig.savefig(p, dpi=105, facecolor="white", pil_kwargs={"quality": 88, "method": 6}); plt.close(fig)
    return p.name


def main() -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("backfill"); b.add_argument("--until", required=True); b.add_argument("--n", type=int, default=WINDOW)
    e = sp.add_parser("estimate"); e.add_argument("--date", required=True)
    for s in (b, e):
        s.add_argument("--runs", required=True); s.add_argument("--cache", default="/tmp/gefs_work/clim")
    a = ap.parse_args()
    runs, cache = Path(a.runs), Path(a.cache)
    if a.cmd == "backfill":
        last = dt.datetime.strptime(a.until, "%Y%m%d").date()
        for k in range(a.n):
            i = last - dt.timedelta(days=k)
            try:
                verify(i, cache, runs); print(f"verified {i}", flush=True)
            except Exception as ex:                                  # noqa: BLE001
                print(f"{i}: {str(ex)[:100]}", flush=True)
        return 0
    today = dt.datetime.strptime(a.date, "%Y%m%d").date()
    estimate(today, cache, runs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
