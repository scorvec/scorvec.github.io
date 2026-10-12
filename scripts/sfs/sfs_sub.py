#!/usr/bin/env python3
"""SFS SUBSEASONAL page (sfs-subseasonal.html): days 1-42 of NOAA's SFS beta2 (31 members, once a month) in the GEFS and
GEPS pages' format (user 2026-10-12: "make an SFS-beta 2 page like what we have for the GEPS/GEFS"; "build it now, let's
see how the forecast evolves over time").

THE GATE. beta2's daily FORECAST stores were defective in both issues it had published when this was written (2026-09,
2026-10): 24 consecutive days written into every other slot of a 47-day axis (state fields on even leads, PRATE/OLR on
odd), so the "forecast" cooled at half the reforecast's rate and its "November" half was late October. The daily
REFORECAST is correct. Every run therefore checks the issue first (check_daily): every lead filled, the 35-65N 2 m
temperature trend across the leads within 0.75-1.33x the reforecast's for the same start month, and the daily mean over
the start month within 0.5 K of the issue's own monthly mean. The result goes into assets/sfs/data/sfs_sub.json; the
products below are only built when the check passes, so the page fills in by itself the first month NOAA's daily store is
sound, and says why it is empty until then.

Anomalies   ensemble mean minus the SFS's OWN 1991-2020 reforecast mean for the same start month and lead (11 members x 30
            years; the reforecast covers the normal period itself, so no re-basing step). 500 hPa height has its daily
            global mean removed, as on the GEFS page. NOAA's published *_clim stores are not used (beta2's October one is
            September's copy).
Weekly      weeks 1-6, days 1-7 ... 36-42 (day 1 = the start date = lead 0).
Terciles    raw member fractions against the terciles of the reforecast's own weekly anomalies (330 per gridpoint and week).
Skill mask  gridpoint anomaly correlation of the reforecast ensemble mean (11 members, leave-one-year-out climatology)
            with ERA5 over 1991-2025 for this start month and week (release sfs-ref-v1, scripts/sfs/sfs_truth.py: 2 m
            temperature global, precipitation Northern Hemisphere only); hatched below 0.25, cross-hatched where there
            is no truth or the week is climatologically dry.
Grid        the GEFS page's 1.5 deg grid; every figure is drawn by gefs_live / gefs_probs with this page's label.

    python scripts/sfs/sfs_sub.py --issue 202610 [--site .] [--procs 4] [--check-only]
    python scripts/sfs/sfs_sub.py --issue 202610 --test-from-reforecast 2025 --years 2016 2020 --members 3   (dev only)
"""
import pyproj  # noqa: F401  (before xarray: eckit's libproj clash)
import argparse
import calendar
import datetime as dt
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "scripts" / "gefs"))
import gefs_reforecast as R  # noqa: E402
import gefs_live as GL       # noqa: E402

SFS_EXP = os.environ.get("SFS_EXP", "beta2")
BASE = f"https://noaa-oar-sfsdev-pds.s3.amazonaws.com/experiments/{SFS_EXP}"
CLIMDIR = HERE / "data" / SFS_EXP
LABEL = f"SFS {SFS_EXP}"
PREFIX = "sfs"
REF_RELEASE = "sfs-ref-v1"
CLIM_Y0, CLIM_Y1 = 1991, 2020
SKILL_MIN = 0.25
DRY_MM = 0.5
WEEKS = [(1, 7), (8, 14), (15, 21), (22, 28), (29, 35), (36, 42)]
# page tag -> (SFS daily variable, factor to the GEFS page's units: K, mm/day, Pa, m, W m-2, m/s)
VARS = {"t2m": ("TMP_2maboveground", 1.0), "pr": ("PRATE_surface", 86400.0), "mslp": ("PRMSL_meansealevel", 1.0),
        "zg500": ("HGT_500mb", 1.0), "olr": ("ULWRF_topofatmosphere", 1.0), "u850": ("UGRD_850mb", 1.0),
        "u200": ("UGRD_200mb", 1.0), "v850": ("VGRD_850mb", 1.0), "v200": ("VGRD_200mb", 1.0)}
TERC = ("t2m", "pr")


def _open(url):
    import fsspec
    import xarray as xr
    return xr.open_zarr(fsspec.get_mapper(url), consolidated=True)


# ── regridding: SFS 1 deg (lat -90..90, lon 0..359) -> the GEFS page's 1.5 deg (lat 90..-90, lon 0..358.5) ─────────
_SI = (R.LAT + 90.0)                                  # fractional source row of every target latitude
_I0 = np.clip(np.floor(_SI).astype(int), 0, 179); _WI = (_SI - _I0)[:, None]
_SJ = R.LON.copy(); _J0 = np.floor(_SJ).astype(int) % 360; _J1 = (_J0 + 1) % 360; _WJ = _SJ - np.floor(_SJ)


def regrid(a):
    """(..., 181, 360) -> (..., 121, 240), bilinear, longitude periodic."""
    a = np.asarray(a, np.float32)
    b = a[..., _I0, :] * (1 - _WI) + a[..., np.minimum(_I0 + 1, 180), :] * _WI
    return b[..., _J0] * (1 - _WJ) + b[..., _J1] * _WJ


def weekly(x, axis=0):
    """Weekly means over WEEKS of a day axis (day 1 = index 0); NaN days are skipped."""
    x = np.moveaxis(np.asarray(x, np.float64), axis, 0)
    with np.errstate(invalid="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out = np.stack([np.nanmean(x[a - 1:b], axis=0) for a, b in WEEKS])
    return np.moveaxis(out, 0, axis)


# ── the gate ─────────────────────────────────────────────────────────────────────────────────────────────────
def _band(a, lat):
    sel = (lat >= 35) & (lat <= 65); w = np.cos(np.deg2rad(lat[sel]))
    return (np.nanmean(a[..., sel, :], axis=-1) * w).sum(-1) / w.sum()


def check_daily(issue: str, month: int) -> dict:
    ds = _open(f"{BASE}/forecast/{issue}/atm_daily.zarr")
    lat = ds["lat"].values
    v = ds["TMP_2maboveground"].isel(member=0).values.astype(np.float64)            # (lead, lat, lon), one chunk
    filled = [int(i) for i in range(v.shape[0]) if np.isfinite(v[i]).any()]
    f = _band(v, lat)
    rf = _open(f"{BASE}/reforecast/{month:02d}/atm_daily.zarr")["TMP_2maboveground"]
    rr = np.mean([_band(rf.isel(member=0, init=i).values.astype(np.float64), lat) for i in range(-5, 0)], axis=0)
    L = np.arange(len(f)); ok = np.isfinite(f)
    sf = float(np.polyfit(L[ok], f[ok], 1)[0]) if ok.sum() > 2 else float("nan")
    sr = float(np.polyfit(L, rr, 1)[0])
    ratio = sf / sr if sr else float("nan")
    m = _open(f"{BASE}/forecast/{issue}/atm_monthly.zarr")["tmp2m"].isel(member=0, lead=0)
    mm = float(_band(m.values.astype(np.float64), m["lat"].values))
    nd = calendar.monthrange(int(issue[:4]), int(issue[4:]))[1]
    dm = float(np.nanmean(f[:nd]))
    res = {"leads": int(v.shape[0]), "filled_leads": len(filled), "trend_forecast_K_per_day": round(sf, 3),
           "trend_reforecast_K_per_day": round(sr, 3), "trend_ratio": round(ratio, 2),
           "start_month_daily_minus_monthly_K": round(dm - mm, 2)}
    res["ok"] = bool(len(filled) == v.shape[0] and 0.75 <= ratio <= 1.33 and abs(dm - mm) < 0.5)
    why = []
    if len(filled) != v.shape[0]:
        why.append(f"only {len(filled)} of {v.shape[0]} forecast days are filled")
    if not 0.75 <= ratio <= 1.33:
        why.append(f"35-65N 2 m temperature changes {sf:+.3f} K a day across the forecast against the reforecast's "
                   f"{sr:+.3f} (ratio {ratio:.2f}): the days are not where the file says they are")
    if abs(dm - mm) >= 0.5:
        why.append(f"the daily fields averaged over the start month differ from the issue's own monthly mean by "
                   f"{dm - mm:+.2f} K")
    res["why"] = why
    return res


# ── reference: climatology, tercile thresholds, skill (one build per start month, cached) ─────────────────────
def ref_truth(month: int, cache: Path) -> dict | None:
    """ERA5 weekly truth for this start month (release sfs-ref-v1), or None."""
    p = cache / f"sfs_truth_{month:02d}.npz"
    if not p.exists():
        cache.mkdir(parents=True, exist_ok=True)
        gh = os.environ.get("GH_BIN", "gh")
        r = subprocess.run([gh, "release", "download", REF_RELEASE, "-p", p.name, "-D", str(cache), "--clobber"],
                           capture_output=True, text=True)
        if r.returncode != 0 or not p.exists():
            print(f"  no {p.name} in {REF_RELEASE} ({r.stderr.strip()[:120]}); skill masks off", flush=True)
            return None
    z = np.load(p)
    return {"years": [int(y) for y in z["years"]], "t2m": z["t2m"].astype(np.float32), "pr": z["pr"].astype(np.float32)}


def build_ref(month: int, years=(CLIM_Y0, CLIM_Y1), members=None, threads=8, truth=None, tag_list=None) -> dict:
    """{d_tag (47, lat, lon) reforecast mean, q33_/q67_tag (6, ...), r_tag (6, ...) skill, wclim_pr (6, ...)}, cached."""
    tag_list = tag_list or list(VARS)
    f = CLIMDIR / f"sub_ref_{month:02d}_{years[0]}_{years[1]}{'' if members is None else f'_m{members}'}.npz"
    if f.exists():
        z = dict(np.load(f))
        if all(f"d_{t}" in z for t in tag_list):
            return z
    ds = _open(f"{BASE}/reforecast/{month:02d}/atm_daily.zarr")
    yrs = [int(str(v)[:4]) for v in ds["init"].values]
    clim_i = [i for i, y in enumerate(yrs) if years[0] <= y <= years[1]]
    all_i = list(range(len(yrs)))
    mem = list(range(ds.sizes["member"] if members is None else min(members, ds.sizes["member"])))
    out = {}
    t0 = time.time()
    for tag in tag_list:
        var, fac = VARS[tag]
        da = ds[var]
        want = all_i if (tag in TERC and truth is not None) else clim_i
        jobs = [(i, m) for i in want for m in mem]

        def get(job, da=da, fac=fac):
            i, m = job
            return i, m, regrid(da.isel(init=i, member=m).values) * fac

        S = np.zeros((47,) + R.LAT.shape + R.LON.shape, np.float64); N = np.zeros_like(S)
        Wk = {} if tag in TERC else None                                   # (init, member) -> weekly (6, lat, lon)
        with ThreadPoolExecutor(threads) as ex:
            for i, m, a in ex.map(get, jobs):
                if i in clim_i:
                    ok = np.isfinite(a); S += np.where(ok, a, 0.0); N += ok
                if Wk is not None:
                    Wk[(i, m)] = weekly(a).astype(np.float32)
        d = (S / np.maximum(N, 1)).astype(np.float32); d[N == 0] = np.nan
        out[f"d_{tag}"] = d
        if Wk is not None:
            wc = weekly(d)                                                  # (6, lat, lon) the climatological week
            A = np.stack([Wk[(i, m)] - wc for i in clim_i for m in mem])   # reforecast member anomalies, 1991-2020
            q = np.nanpercentile(A, [100 / 3, 200 / 3], axis=0)
            out[f"q33_{tag}"], out[f"q67_{tag}"] = q[0].astype(np.float32), q[1].astype(np.float32)
            if tag == "pr":
                out["wclim_pr"] = wc.astype(np.float32)
            if truth is not None:
                # leave-one-year-out ensemble-mean anomaly per start, correlated with ERA5 across the years with truth
                em = {i: np.mean([Wk[(i, m)] for m in mem], axis=0) for i in want}
                tot = np.sum([em[i] for i in clim_i], axis=0)
                ty = truth["years"]
                F, O = [], []
                for i in want:
                    y = yrs[i]
                    if y not in ty:
                        continue
                    o = truth[tag][ty.index(y)]
                    base_c = (tot - em[i]) / (len(clim_i) - 1) if i in clim_i else tot / len(clim_i)
                    F.append(em[i] - base_c); O.append(o)
                F, O = np.array(F), np.array(O)
                ok = np.isfinite(O).all(0) & np.isfinite(F).all(0) & (np.isfinite(O).sum(0) >= 20)
                Fa = F - F.mean(0); Oa = O - np.nanmean(O, 0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    r = (Fa * Oa).sum(0) / np.sqrt((Fa ** 2).sum(0) * (Oa ** 2).sum(0))
                out[f"r_{tag}"] = np.where(ok, r, np.nan).astype(np.float32)
                out[f"n_{tag}"] = np.array(len(F))
        print(f"  ref {tag} {month:02d}: {len(jobs)} reforecast runs, {time.time() - t0:.0f} s", flush=True)
    CLIMDIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, **out)
    return out


# ── the forecast ───────────────────────────────────────────────────────────────────────────────────────────────
def load_forecast(issue: str, threads=8, test_year: int | None = None, month: int | None = None):
    """{tag: (member, 47, lat, lon) float32 on the page grid}, plus the member count. --test-from-reforecast reads one
    reforecast start instead (dev only: exercises the figures while NOAA's daily forecast store is defective)."""
    if test_year is None:
        ds = _open(f"{BASE}/forecast/{issue}/atm_daily.zarr"); sel = {}
        nm = ds.sizes["member"]
    else:
        ds = _open(f"{BASE}/reforecast/{month:02d}/atm_daily.zarr")
        yrs = [int(str(v)[:4]) for v in ds["init"].values]
        sel = {"init": yrs.index(test_year)}; nm = ds.sizes["member"]
    F = {}
    for tag, (var, fac) in VARS.items():
        da = ds[var].isel(**sel) if sel else ds[var]
        with ThreadPoolExecutor(threads) as ex:
            arrs = list(ex.map(lambda m: regrid(da.isel(member=m).values) * fac, range(nm)))
        F[tag] = np.stack(arrs).astype(np.float32)
        print(f"  forecast {tag}: {nm} members", flush=True)
    return F, nm


def anomalies(F, ref):
    A, E = {}, {}
    for tag in VARS:
        e = np.nanmean(F[tag], axis=0)                                   # ensemble mean (47, lat, lon)
        a = e - ref[f"d_{tag}"]
        if GL.FIELDS[tag][3]:                                            # 500 hPa: daily global mean removed
            a = a - np.array([GL.global_mean(np.nan_to_num(x)) for x in a])[:, None, None]
        A[tag], E[tag] = a, e
    return A, E


# ── figures ────────────────────────────────────────────────────────────────────────────────────────────────────
def skill_hatcher(r_week):
    import cartopy.crs as ccrs

    def h(ax, i):
        r = r_week[i]
        for fld, hatch, col in (((np.isfinite(r) & (r < SKILL_MIN)).astype(float), "////", "#2f2c28"),
                                ((~np.isfinite(r)).astype(float), "xx", "#a6a29a")):
            if fld.any():
                cs = ax.contourf(R.LON, R.LAT, fld, levels=[0.5, 1.5], colors="none", hatches=[hatch],
                                 transform=ccrs.PlateCarree(), zorder=3)
                cs.set_edgecolor(col); cs.set_linewidth(0.0)
    return h


def render(issue, base, F, nm, ref, site: Path, procs: int):
    import gefs_probs as GP
    GL.MODEL, GL.PREFIX = LABEL, PREFIX
    GP.MODEL, GP.PREFIX = LABEL, PREFIX
    GP.TERC_NOTE = f"terciles of its own {CLIM_Y0}–{CLIM_Y1} reforecast"
    out = site / "assets" / "sfs" / "sub"; out.mkdir(parents=True, exist_ok=True)
    anim = site / "assets" / "sfs" / "anim" / "sub"; anim.mkdir(parents=True, exist_ok=True)
    for pat in ("sfs_*_weekly*.webp", "sfs_*_prob_*.webp"):          # this script's own figures only (the skill figure is not)
        for old in out.glob(pat):
            old.unlink()
    A, E = anomalies(F, ref)
    days = [base + dt.timedelta(days=d) for d in range(47)]
    titles = [f"Week {k + 1}   {days[a - 1]:%b %d} – {days[b - 1]:%b %d}" for k, (a, b) in enumerate(WEEKS)]
    names = []
    for tag in GL.ANIM_ORDER:
        wk = weekly(A[tag]); fc = weekly(E[tag])
        note = "global mean removed per day" if GL.FIELDS[tag][3] else ""
        cbl = (f"{GL.FIELDS[tag][0]} anomaly vs its {CLIM_Y0}–{CLIM_Y1} reforecast ({GL.FIELDS[tag][1]})"
               + ("   ·   contours: ensemble mean" if tag in GL.CONTOUR else ""))
        for key, dom in GL.DOMAINS.items():
            names.append(GL.map_figure(tag, list(wk), list(fc), base, key, dom, out, "weekly", titles, note, cb_label=cbl))
        if f"r_{tag}" in ref:
            mnote = (note + "   ·   " if note else "") + (f"hatched: reforecast anomaly correlation with ERA5 below {SKILL_MIN} for this week "
                     f"(1991–2025 starts{', Northern Hemisphere only' if tag == 'pr' else ''}); cross-hatched: no ERA5 truth")
            for key, dom in GL.DOMAINS.items():
                names.append(GL.map_figure(tag, list(wk), list(fc), base, key, dom, out, "weekly", titles, mnote, cb_label=cbl,
                                           hatcher=skill_hatcher(ref[f"r_{tag}"]), suffix="_masked"))
    # terciles: raw member fractions against the reforecast's own weekly anomaly terciles
    for tag in TERC:
        Wm = weekly(F[tag], axis=1) - weekly(ref[f"d_{tag}"])[None]       # (member, 6, lat, lon) member anomalies
        q33, q67 = ref[f"q33_{tag}"], ref[f"q67_{tag}"]
        r = ref.get(f"r_{tag}", np.full(q33.shape, np.nan, np.float32))
        dry = (ref["wclim_pr"] < DRY_MM) if tag == "pr" else np.zeros(q33.shape, bool)
        weeks = []
        for k in range(len(WEEKS)):
            w = Wm[:, k]; ok = np.isfinite(q33[k]) & np.isfinite(q67[k]) & np.isfinite(w).all(0)
            raw = np.stack([np.where(ok, (w < q33[k]).mean(0), np.nan), np.where(ok, ((w >= q33[k]) & (w <= q67[k])).mean(0), np.nan),
                            np.where(ok, (w > q67[k]).mean(0), np.nan)])
            weeks.append(dict(k=k + 1, raw=raw, r=r[k], dry=dry[k], d0=days[WEEKS[k][0] - 1], d1=days[WEEKS[k][1] - 1]))
        names += GP.draw(tag, base, weeks, "raw", nm, out)
    nd = GL.daily_loops({t: A[t] for t in GL.ANIM_ORDER}, {f"e_{t}": E[t] for t in GL.ANIM_ORDER}, base, anim, procs,
                        model=LABEL, prefix="sfs_sub", days=list(range(1, 48)))
    return names, nd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--issue", default=dt.datetime.utcnow().strftime("%Y%m"))
    ap.add_argument("--site", default=str(REPO)); ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--cache", default="/tmp/sfs_ref"); ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--test-from-reforecast", type=int, default=None, help="dev: use this reforecast year as the forecast")
    ap.add_argument("--years", type=int, nargs=2, default=[CLIM_Y0, CLIM_Y1]); ap.add_argument("--members", type=int, default=None)
    a = ap.parse_args()
    month = int(a.issue[4:]); base = dt.date(int(a.issue[:4]), month, 1)
    site = Path(a.site); data = site / "assets" / "sfs" / "data"; data.mkdir(parents=True, exist_ok=True)
    status = {"issue": f"{a.issue[:4]}-{a.issue[4:]}", "init": f"{base:%Y-%m-%d} 00Z", "experiment": SFS_EXP,
              "generated": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%MZ")}
    if a.test_from_reforecast is None:
        try:
            status["check"] = check_daily(a.issue, month)
        except Exception as e:                                           # noqa: BLE001
            status["check"] = {"ok": False, "why": [f"the daily stores could not be read ({str(e)[:120]})"]}
        print("  daily-store check:", json.dumps(status["check"]), flush=True)
    else:
        status["check"] = {"ok": True, "why": [], "test_from_reforecast": a.test_from_reforecast}
    status["ok"] = bool(status["check"]["ok"])
    if status["ok"] and not a.check_only:
        truth = ref_truth(month, Path(a.cache))
        ref = build_ref(month, tuple(a.years), a.members, a.threads, truth)
        F, nm = load_forecast(a.issue, a.threads, a.test_from_reforecast, month)
        names, nd = render(a.issue, base, F, nm, ref, site, a.procs)
        days = [base + dt.timedelta(days=d) for d in range(47)]
        status.update({"members": nm, "valid": f"{days[0]:%b %d} – {days[WEEKS[-1][1] - 1]:%b %d}", "figures": names,
                       "frames": nd, "weeks": [[f"{days[a_ - 1]:%Y-%m-%d}", f"{days[b - 1]:%Y-%m-%d}"] for a_, b in WEEKS],
                       "skill_masks": any(k.startswith("r_") for k in ref)})
    (data / "sfs_sub.json").write_text(json.dumps(status, indent=1))
    print(f"  sfs_sub.json: ok={status['ok']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
