#!/usr/bin/env python3
"""Weeks 3-5 AR outlook from the GEFS extended run (00Z, 35 days), as anomalies of AR FREQUENCY against GEFS's own
reforecast climatology, shown only where the forecast is distinguishable from that climatology.

AR frequency = the fraction of 6-hourly times in the week with proxy IVT* = K_EXT · PWAT · |V850| ≥ 250 kg m-1 s-1.
K_EXT is FIXED (not refitted per cycle as on the day 1-15 maps) so that the live run and the 2000-2019 reforecast use
the identical definition; its value is the mean of the per-cycle GEFS fits on the control's exact IVT (ar_ens.py).
Weeks: day d = UTC day init + (d-1); week 3 = days 15-21 (steps 336..498 h), week 4 = days 22-28, week 5 = days 29-35
(28 six-hourly samples each), and weeks 3-4 together.

Climatology (release ar-clim-v1, built by `ar_ext.py clim YEAR` in ar-clim.yml): GEFSv12 reforecast 2000-2019,
Wednesday 00Z 35-day starts, control + 4 perturbed members (p01-p04), per start x member x week the count of exceeding
samples on 1-deg cells (each the sum over its four 0.5-deg points, 0..112). For a forecast, every start within ±15
days of the init's day of year (any year) is used: ~85 starts x 5 members.

Significance (site rule: nothing insignificant is shown): per cell, a two-sided Mann-Whitney U test of the 31 member
frequencies against the reforecast member frequencies (H0: the forecast members are draws from the model's
climatological distribution), then Benjamini-Hochberg FDR at alpha 0.10 across the domain (Wilks 2016). Only cells
that pass are shaded. "Significant" means distinguishable from GEFS's own climate, NOT demonstrated skill: week-3+ AR
occurrence skill is low (DeFlorio et al. 2019).

    python ar_ext.py clim 2019 --out out/                      # reforecast counts for one year -> out/arclim_2019.npz
    python ar_ext.py live [--date YYYYMMDD] --work /tmp/arx    # the live outlook -> assets/ar/ext_*.webp + ar_ext.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arlib as A                                                      # noqa: E402

SITE = Path(os.environ.get("SST_SITE_ROOT", HERE.parents[1]))
ASSETS = SITE / "assets" / "ar"
K_EXT = float(os.environ.get("AR_K_EXT", "0.0"))      # set below once measured; 0 = not set (refuse to run)
K_EXT_DEFAULT = 0.90
RF = "https://noaa-gefs-retrospective.s3.amazonaws.com/GEFSv12/reforecast"
RF_MEMBERS = ("c00", "p01", "p02", "p03", "p04")
WEEKS = {"w3": (15, 21), "w4": (22, 28), "w5": (29, 35)}
STEPS = tuple(range(336, 840, 6))                       # days 15..35, 84 samples
# 0.5-deg points of the domain and the 1-deg cells they are summed into (each cell = its SW-anchored 2x2 block)
LAT5 = np.arange(10.0, 70.0, 0.5); LON5 = np.arange(120.0, 330.0, 0.5)
LAT1 = LAT5[::2] + 0.25; LON1 = LON5[::2] + 0.25
REL_TAG = "ar-clim-v1"
REL_URL = "https://github.com/scorvec/scorvec.github.io/releases/download/" + REL_TAG + "/arclim_{y}.npz"
YEARS = range(2000, 2020)


def k_ext():
    return K_EXT or K_EXT_DEFAULT


def week_of(h):
    d = h // 24 + 1
    for w, (a, b) in WEEKS.items():
        if a <= d <= b:
            return w
    return None


def to5(v):
    """Global 0.5-deg field (90 -> -90, 0 -> 359.5) -> the domain points (lat ascending)."""
    nj, ni = v.shape
    lat = np.linspace(90, -90, nj); lon = np.arange(ni) * 360.0 / ni
    i = A.nearest_idx(lat, LAT5); j = A.nearest_idx(lon, LON5)
    return v[np.ix_(i, j)]


def decode_global(b):
    import eccodes as ec
    g = ec.codes_new_from_message(b)
    try:
        nj, ni = ec.codes_get(g, "Nj"), ec.codes_get(g, "Ni")
        v = ec.codes_get_values(g).reshape(nj, ni)
        if ec.codes_get(g, "latitudeOfFirstGridPointInDegrees") < 0:
            v = v[::-1]
    finally:
        ec.codes_release(g)
    return v


def counts(pwat, u, v):
    """(84 steps, lat5, lon5) arrays -> uint8 (3 weeks, lat1, lon1) counts of IVT* >= 250."""
    ex = (k_ext() * pwat * np.hypot(u, v)) >= 250
    out = np.zeros((3, len(LAT1), len(LON1)), "uint8")
    for wi, w in enumerate(WEEKS):
        sel = [k for k, h in enumerate(STEPS) if week_of(h) == w]
        c = ex[sel].sum(0)                                              # (lat5, lon5) 0..28
        out[wi] = c.reshape(len(LAT1), 2, len(LON1), 2).sum((1, 3))
    return out


# ── reforecast climatology ───────────────────────────────────────────────────────────────────────────────────
def rf_member(args):
    start, mem = args
    ymd = start.strftime("%Y%m%d") + "00"
    base = f"{RF}/{start:%Y}/{ymd}/{mem}/Days:10-35"
    want = {}
    for var, fn, lev in (("pwat", f"pwat_eatm_{ymd}_{mem}.grib2", "entire atmosphere (considered as a single layer)"),
                         ("u", f"ugrd_pres_{ymd}_{mem}.grib2", "850 mb"), ("v", f"vgrd_pres_{ymd}_{mem}.grib2", "850 mb")):
        recs = A.idx_records(f"{base}/{fn}")
        if recs is None:
            return start, mem, None
        for s, e, nm, lv, st in recs:
            if lv == lev and st.endswith("hour fcst") and st.split()[0].isdigit() and int(st.split()[0]) in STEPS:
                want[(var, int(st.split()[0]))] = (f"{base}/{fn}", s, e)
    if len(want) != 3 * len(STEPS):
        return start, mem, None
    arr = {"pwat": np.zeros((len(STEPS), len(LAT5), len(LON5)), "float32")}
    arr["u"] = arr["pwat"].copy(); arr["v"] = arr["pwat"].copy()
    si = {h: k for k, h in enumerate(STEPS)}

    def one(item):
        (var, h), (url, s, e) = item
        b = A.http_get(url, (s, e), src="aws-s3")
        if b is None or not b.startswith(b"GRIB"):
            raise RuntimeError(f"bad range {url}")
        arr[var][si[h]] = to5(decode_global(b))

    with ThreadPoolExecutor(12) as ex:
        list(ex.map(one, want.items()))
    return start, mem, counts(arr["pwat"], arr["u"], arr["v"])


def build_clim(year: int, out: Path, procs: int):
    t0 = time.time()
    d = dt.date(year, 1, 1)
    while d.weekday() != 2:                                            # Wednesdays
        d += dt.timedelta(days=1)
    starts = []
    while d.year == year:
        starts.append(d); d += dt.timedelta(days=7)
    jobs = [(s, m) for s in starts for m in RF_MEMBERS]
    res = {}
    with ProcessPoolExecutor(procs) as ex:
        for s, m, c in ex.map(rf_member, jobs):
            res[(s, m)] = c
            if m == RF_MEMBERS[-1]:
                print(f"  {s}: {'ok' if c is not None else 'missing'} ({time.time() - t0:.0f}s)", flush=True)
    ok_starts = [s for s in starts if all(res.get((s, m)) is not None for m in RF_MEMBERS)]
    C = np.stack([np.stack([res[(s, m)] for m in RF_MEMBERS]) for s in ok_starts])   # (start, member, week, lat, lon)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"arclim_{year}.npz", starts=np.array([s.isoformat() for s in ok_starts]), counts=C,
                        members=np.array(RF_MEMBERS), lat=LAT1, lon=LON1, k=np.array(k_ext()), steps=np.array(STEPS))
    print(f"{year}: {len(ok_starts)}/{len(starts)} starts, {C.nbytes / 1e6:.0f} MB raw, {time.time() - t0:.0f}s, "
          f"{A.LEDGER.report()}", flush=True)


# ── live ─────────────────────────────────────────────────────────────────────────────────────────────────────
GEFS = "https://noaa-gefs-pds.s3.amazonaws.com"


def live_member(args):
    date, mem = args
    arr = {"pwat": np.zeros((len(STEPS), len(LAT5), len(LON5)), "float32")}
    arr["u"] = arr["pwat"].copy(); arr["v"] = arr["pwat"].copy()
    jobs = []
    for k, h in enumerate(STEPS):
        url = f"{GEFS}/gefs.{date}/00/atmos/pgrb2ap5/{mem}.t00z.pgrb2a.0p50.f{h:03d}"
        recs = A.idx_records(url)
        if recs is None:
            return mem, None
        for s, e, var, lev, st in recs:
            if var == "PWAT" and lev.startswith("entire atmosphere"):
                jobs.append(("pwat", k, url, s, e))
            elif var in ("UGRD", "VGRD") and lev == "850 mb":
                jobs.append((var[0].lower(), k, url, s, e))
    if len(jobs) != 3 * len(STEPS):
        return mem, None

    def one(j):
        var, k, url, s, e = j
        b = A.http_get(url, (s, e), src="aws-s3")
        if b is None or not b.startswith(b"GRIB"):
            raise RuntimeError(f"bad range {url}")
        arr[var][k] = to5(decode_global(b))

    with ThreadPoolExecutor(12) as ex:
        list(ex.map(one, jobs))
    return mem, counts(arr["pwat"], arr["u"], arr["v"])


def load_clim(init: dt.date, cache: Path, window=15):
    cache.mkdir(parents=True, exist_ok=True)
    samples, ys = [], []
    doy0 = init.timetuple().tm_yday
    for y in YEARS:
        p = cache / f"arclim_{y}.npz"
        if not p.exists():
            b = A.http_get(REL_URL.format(y=y), src="github-release", timeout=300)
            if b is None:
                raise SystemExit(f"climatology {REL_URL.format(y=y)} not found")
            p.write_bytes(b)
        z = np.load(p)
        if abs(float(z["k"]) - k_ext()) > 1e-6:
            raise SystemExit(f"climatology k {float(z['k'])} != live k {k_ext()}: rebuild ar-clim or fix AR_K_EXT")
        for i, s in enumerate(z["starts"]):
            dd = abs(dt.date.fromisoformat(str(s)).timetuple().tm_yday - doy0)
            if min(dd, 365 - dd) <= window:
                samples.append(z["counts"][i]); ys.append(y)
    return np.stack(samples), ys                                         # (start, member, week, lat, lon)


def bh(p, alpha=0.10):
    """Benjamini-Hochberg: boolean mask of discoveries over the finite p-values."""
    flat = p[np.isfinite(p)]
    if flat.size == 0:
        return np.zeros_like(p, bool)
    o = np.sort(flat); m = o.size
    ok = o <= alpha * np.arange(1, m + 1) / m
    thr = o[np.nonzero(ok)[0].max()] if ok.any() else -1.0
    return np.isfinite(p) & (p <= thr)


def ext_figure(out, title, sub, anom, sig, clim_f, n_sig, n_test):
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    import cartopy.crs as ccrs
    pc = ccrs.PlateCarree()
    dom = A.ENS_DOMAIN
    fig, ax, H = A.map_axes(dom)
    lev = [-30, -20, -15, -10, -5, -2, 2, 5, 10, 15, 20, 30]
    cols = ["#8c510a", "#bf812d", "#dfc27d", "#f0dcb0", "#f6efe0", "#ffffff", "#dbe9f6", "#a9cbe6", "#6aa3d2", "#3a76ad",
            "#1b4f8a"]
    cm = ListedColormap(cols); cm.set_under("#5c3305"); cm.set_over("#0b2a55")
    nm = BoundaryNorm(lev, cm.N)
    a = np.where(sig, anom, np.nan)
    pm = ax.pcolormesh(LON1, LAT1, a, cmap=cm, norm=nm, transform=pc, shading="nearest", rasterized=True, zorder=2)
    cs = ax.contour(LON1, LAT1, clim_f, levels=[5, 10, 20, 30], colors="#555", linewidths=[0.5, 0.7, 0.9, 1.1], transform=pc, zorder=4)
    ax.clabel(cs, fmt="%d%%", fontsize=6.5)
    A.map_features(ax)
    A.heading(fig, H, title, sub)
    if n_sig == 0:
        ax.text(0.5, 0.5, "No significant difference from the GEFS reforecast climatology this week",
                transform=ax.transAxes, ha="center", va="center", fontsize=12, color="#333",
                bbox=dict(facecolor="white", edgecolor="#999", boxstyle="round,pad=0.5"), zorder=10)
    cax = fig.add_axes([0.2, 0.36 / H, 0.6, 0.13 / H])
    cb = fig.colorbar(pm, cax=cax, orientation="horizontal", extend="both", ticks=lev)
    cb.ax.tick_params(labelsize=7.5)
    cb.set_label(f"AR-frequency anomaly (percentage points of the week's 6-hourly times) · shaded only where significant "
                 f"({n_sig:,} of {n_test:,} cells, FDR 10 %) · grey contours: climatological frequency", fontsize=7.8)
    fig.savefig(out, dpi=100, facecolor="white", pil_kwargs={"quality": 80, "method": 6})
    plt.close(fig)


def live(date, work: Path, procs: int):
    t0 = time.time()
    sys.path.insert(0, str(HERE.parent / "gefs"))
    import gefs_live                                                    # the GEFS page's resolver (00Z to 840 h)
    date = gefs_live.latest_cycle(date)
    if date is None:
        raise SystemExit("no GEFS extended cycle reaching 840 h in the last week")
    init = dt.date(int(date[:4]), int(date[4:6]), int(date[6:]))
    print(f"== GEFS extended {date} 00Z, weeks 3-5, k = {k_ext()}", flush=True)
    mems = ["gec00"] + [f"gep{i:02d}" for i in range(1, 31)]
    res = {}
    with ProcessPoolExecutor(procs) as ex:
        for m, c in ex.map(live_member, [(date, m) for m in mems]):
            res[m] = c
    got = [m for m in mems if res[m] is not None]
    if len(got) < 25:
        raise SystemExit(f"only {len(got)} complete members")
    F = np.stack([res[m] for m in got]).astype("float32") / 112.0         # (member, week, lat1, lon1)
    t_fetch = time.time() - t0
    Cn, ys = load_clim(init, work / "clim")
    Cf = Cn.reshape(-1, *Cn.shape[2:]).astype("float32") / 112.0         # (start*member, week, lat, lon)
    from scipy.stats import mannwhitneyu
    doc = {"init": init.isoformat() + "T00Z", "members": len(got), "k": k_ext(), "clim_starts": int(Cn.shape[0]),
           "clim_years": sorted(set(ys)), "clim_samples": int(Cf.shape[0]), "weeks": {}}
    views = {"w3": [0], "w4": [1], "w5": [2], "w34": [0, 1]}
    names = {"w3": "Week 3", "w4": "Week 4", "w5": "Week 5", "w34": "Weeks 3–4"}
    for key, wk in views.items():
        f = F[:, wk].mean(1); c = Cf[:, wk].mean(1)                     # member / sample frequency over the window
        a, b = f.reshape(f.shape[0], -1), c.reshape(c.shape[0], -1)
        test = (a.std(0) > 0) | (b.std(0) > 0)
        p = np.full(a.shape[1], np.nan)
        if test.any():
            p[test] = mannwhitneyu(a[:, test], b[:, test], axis=0, alternative="two-sided").pvalue
        sig = bh(p).reshape(f.shape[1:])
        anom = 100.0 * (f.mean(0) - c.mean(0))
        n_sig, n_test = int(sig.sum()), int(test.sum())
        d0 = init + dt.timedelta(days=WEEKS["w3" if key == "w34" else key][0] - 1)
        d1 = init + dt.timedelta(days=(WEEKS["w4"] if key == "w34" else WEEKS[key])[1] - 1)
        ttl = f"AR frequency anomaly, {names[key]} ({d0:%d %b}–{d1:%d %b}) — GEFS {len(got)} members, init {init:%d %b %Y} 00Z"
        sub = (f"Share of 6-hourly times with IVT* = {k_ext():.2f}·PWAT·|V850| ≥ 250, minus GEFSv12 reforecast 2000–2019 at the same lead "
               f"({Cn.shape[0]} starts within ±15 days × 5 members) · Mann–Whitney U, Benjamini–Hochberg FDR 10 %")
        out = ASSETS / f"ext_{key}.webp"
        ext_figure(out, ttl, sub, anom, sig, 100.0 * c.mean(0), n_sig, n_test)
        doc["weeks"][key] = {"label": names[key], "from": d0.isoformat(), "to": d1.isoformat(), "n_sig": n_sig, "n_test": n_test,
                             "max_pos": round(float(np.nanmax(np.where(sig, anom, np.nan))) if n_sig else 0.0, 1),
                             "max_neg": round(float(np.nanmin(np.where(sig, anom, np.nan))) if n_sig else 0.0, 1),
                             "file": out.name}
        print(f"  {key}: {n_sig}/{n_test} significant cells; anomaly range {np.nanmin(anom):.1f}..{np.nanmax(anom):.1f} pp", flush=True)
    import resource
    doc.update(generated=time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), download=A.LEDGER.report(),
               runtime_s={"fetch": round(t_fetch), "total": round(time.time() - t0)},
               peak_rss_MB=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
    (ASSETS / "ar_ext.json").write_text(json.dumps(doc, separators=(",", ":")))
    print("== ext:", json.dumps(doc), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    a1 = sp.add_parser("clim"); a1.add_argument("year", type=int); a1.add_argument("--out", default="out")
    a1.add_argument("--procs", type=int, default=4)
    a2 = sp.add_parser("live"); a2.add_argument("--date"); a2.add_argument("--work", default="/tmp/arx")
    a2.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    if a.cmd == "clim":
        build_clim(a.year, Path(a.out), a.procs)
    else:
        live(a.date, Path(a.work), a.procs)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(rc or 0)
