#!/usr/bin/env python3
"""Deterministic IVT, days 1-7, four models: ECMWF AIFS single, ECMWF IFS HRES, NCEP GFS, ECCC GDPS.

Every model's IVT is computed here from q, u and v on the pressure levels it publishes (arlib.integrate: 1000-300 hPa,
trapezoidal in p, below-ground levels dropped by surface pressure), 6-hourly from the analysis to +168 h:

    AIFS / IFS  1000 925 850 700 600 500 400 300            (the open data's levels in that range)
    GFS         1000 975 950 925 900 850 ... 350 300         (17 levels, every 25/50 hPa)
    GDPS        1000 985 970 950 925 900 875 850 ... 300     (19 levels)

Per model (one job each in ar-det.yml):
    python ar_det.py model --model gfs --date 20261007 --time 00 --work /tmp/ard
        -> assets/ar/anim/det_<model>_{pac,na}/F00..F28.webp   the IVT loops (two domains)
           assets/ar/anim/det_<model>_pac/coast.json             coastal-transect IVT series + AR-scale ranks per point
           <work>/det_<model>.json                               the manifest part (frames, levels, bytes, runtime)
           <work>/det_<model>_t0.npz                             analysis-time IVT on a 0.5 deg grid (validation)
Then once:
    python ar_det.py combine --work /tmp/ard        -> assets/ar/anim/ar_det_manifest.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arlib as A                                                      # noqa: E402

SITE = Path(os.environ.get("SST_SITE_ROOT", HERE.parents[1]))
ANIM = SITE / "assets" / "ar" / "anim"
MANIFEST = ANIM / "ar_det_manifest.json"
STEPS = tuple(range(0, 169, 6))
MAX_AGE_H = 30                                    # a model missing this cycle keeps its previous run on the page this long


def _frame_label(init, h):
    t = init + pd.Timedelta(hours=h)
    return ("analysis" if h == 0 else f"+{h} h") + f" · {t:%a %d %b %HZ}"


# fields shared with the forked render workers
_R: dict = {}


def _render(job):
    k, dom_key = job
    R = _R
    dom = A.DOMAINS[dom_key]
    m = A.MODELS[R["model"]]
    h = STEPS[k]
    out = R["dirs"][dom_key] / f"F{k:02d}.webp"
    title = f"{m['label']} IVT · init {R['init']:%d %b %Y %HZ} · {_frame_label(R['init'], h)}"
    lv = R["levels"]
    sub = (f"Integrated vapour transport 1000–300 hPa from q, u, v on {len(lv)} levels ({m['res']}); below-ground levels "
           "removed with surface pressure · contours: sea-level pressure, 4 hPa")
    A.ivt_frame(out, R["lat"], R["lon"], R["ivt"][k], R["ivte"][k], R["ivtn"][k], dom, title, sub, msl=R["msl"][k])
    return dom_key, k, out.stat().st_size


def coast_figure(out, m, init, valid, cser, tl, levels):
    """Coastline IVT, latitude x time (the deterministic landfall chart)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    cm, nm = A.ivt_cmap()
    t = pd.DatetimeIndex(valid)
    fig, ax = plt.subplots(figsize=(12, 6.4))
    pm = ax.pcolormesh(t, tl, np.where(cser >= A.IVT_LEV[0], cser, np.nan).T, cmap=cm, norm=nm, shading="nearest")
    cs = ax.contour(t, tl, cser.T, levels=[250, 500, 750, 1000], colors="#111", linewidths=[0.6, 0.9, 1.2, 1.5])
    ax.clabel(cs, fmt="%d", fontsize=7)
    pts = A.coast_points()
    ax.set_yticks([p[0] for p in pts]); ax.set_yticklabels([f"{p[2].split(' (')[0]}  {p[0]:g}°N" for p in pts], fontsize=7.5)
    ax.set_ylim(tl.min(), tl.max())
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d\n%b"))
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
    ax.tick_params(axis="x", labelsize=7.5); ax.grid(True, axis="x", alpha=0.3)
    fig.subplots_adjust(left=0.2, right=0.97, top=0.86, bottom=0.21)
    fig.text(0.02, 0.97, f"Coastline IVT — {m['label']}, init {init:%d %b %Y %HZ}, 6-hourly to day 7", fontsize=12.5,
             fontweight="bold", va="top")
    fig.text(0.02, 0.925, f"IVT 1000–300 hPa ({len(levels)} levels) at the first ocean point west of the coast on each 0.25° latitude, "
             "Baja California to southeast Alaska · contours 250 / 500 / 750 / 1000", fontsize=8.5, color="#444", va="top")
    cax = fig.add_axes([0.3, 0.085, 0.5, 0.025])
    cb = fig.colorbar(pm, cax=cax, orientation="horizontal", extend="max", ticks=A.IVT_LEV); cb.ax.tick_params(labelsize=7.5)
    cb.set_label("IVT (kg m⁻¹ s⁻¹) · AR threshold 250", fontsize=8)
    fig.savefig(out, dpi=100, facecolor="white", pil_kwargs={"quality": 80, "method": 6}); plt.close(fig)


def run_model(model, date, hh, work: Path, procs: int, wait_min: float):
    t0 = time.time()
    init = pd.Timestamp(f"{date}T{hh}:00")
    m = A.MODELS[model]
    print(f"== {m['label']} {date} {hh}Z, {len(STEPS)} steps to +{STEPS[-1]} h", flush=True)
    ivt, ivte, ivtn, msl, lat, lon = [], [], [], [], None, None
    sub8 = []                                     # GFS/GDPS: IVT on the ECMWF 8-level subset, for the level-bias check
    t_end = time.time() + 60 * wait_min
    for h in STEPS:
        while True:
            d = A.read_step(model, date, hh, h)
            if d is not None or time.time() > t_end:
                break
            print(f"  +{h} h not published yet; waiting", flush=True); time.sleep(120)
        if d is None:
            raise SystemExit(f"{model} {date} {hh}Z +{h} h is not published: cycle left partial (nothing written)")
        if d["ps"] is None:
            raise SystemExit(f"{model} +{h} h: no surface pressure")
        # screens: q in kg/kg, winds plausible, surface pressure in Pa
        qx, ux = float(np.nanmax(d["q"])), float(np.nanmax(np.abs(d["u"])))
        psm = float(np.nanmedian(d["ps"]))
        if not (0.005 < qx < 0.05) or not (5 < ux < 150) or not (50000 < psm < 105000):
            raise SystemExit(f"{model} +{h} h fails the range screen: max q {qx:.4f}, max |u| {ux:.0f}, median ps {psm:.0f}")
        mag, e, n = A.ivt_of(d)
        if model in ("gfs", "gdps"):
            sub8.append(A.ivt_of(d, A.ECMWF_LEVELS)[0])
        ivt.append(mag); ivte.append(e); ivtn.append(n); msl.append(d["msl"]); lat, lon = d["lat"], d["lon"]
        if h % 24 == 0:
            print(f"  +{h:3d} h  max IVT {np.nanmax(mag):6.0f}  ({time.time() - t0:.0f}s)", flush=True)
    levels = list(d["levels"])
    t_fetch = time.time() - t0
    ivt = np.stack(ivt); ivte = np.stack(ivte); ivtn = np.stack(ivtn); msl = np.stack(msl)
    valid = [init + pd.Timedelta(hours=h) for h in STEPS]

    # ── level-set bias (GFS, GDPS): full set vs the ECMWF 8 levels, over IVT >= 250 points ──
    lvl_bias = None
    if sub8:
        s8 = np.stack(sub8); ok = ivt >= 250
        lvl_bias = {"n": int(ok.sum()), "median_ratio_8lev_to_full": round(float(np.median(s8[ok] / ivt[ok])), 4),
                    "mean_diff": round(float(np.mean(s8[ok] - ivt[ok])), 1),
                    "p10_ratio": round(float(np.percentile(s8[ok] / ivt[ok], 10)), 4),
                    "p90_ratio": round(float(np.percentile(s8[ok] / ivt[ok], 90)), 4)}
        print(f"  8-level vs {len(levels)}-level IVT where >= 250: {lvl_bias}", flush=True)

    # ── frames ──
    dirs = {k: ANIM / f"det_{model}_{k}" for k in A.DOMAINS}
    for dd in dirs.values():
        dd.mkdir(parents=True, exist_ok=True)
        for f in dd.glob("F*.webp"):
            f.unlink()
    _R.update(model=model, init=init, levels=levels, lat=lat, lon=lon, ivt=ivt, ivte=ivte, ivtn=ivtn, msl=msl, dirs=dirs)
    jobs = [(k, dk) for k in range(len(STEPS)) for dk in A.DOMAINS]
    t1 = time.time()
    import multiprocessing as mp
    A.warm_features()
    with mp.get_context("fork").Pool(procs) as pool:
        res = pool.map(_render, jobs, chunksize=2)
    t_render = time.time() - t1
    frames = {dk: [] for dk in A.DOMAINS}
    for dk, k, _ in sorted(res, key=lambda r: (r[0], r[1])):
        frames[dk].append({"idx": k, "file": f"F{k:02d}.webp", "date": valid[k].strftime("%Y-%m-%d"),
                           "label": _frame_label(init, STEPS[k])})

    # ── coastal transect ──
    tl, tlon = A.transect()
    ii = A.nearest_idx(lat, tl); jj = A.nearest_idx(lon, tlon)
    cser = ivt[:, ii, jj]                                              # (step, transect)
    pts = []
    for la, lo, nm in A.coast_points():
        i = int(A.nearest_idx(lat, la)[0]); j = int(A.nearest_idx(lon, lo)[0])
        s = ivt[:, i, j]
        rk, pk, du, i0 = A.ar_rank(s)
        pts.append({"lat": la, "lon": round(float(lon[j]) - 360.0, 2), "name": nm, "ivt": np.round(s).astype(int).tolist(),
                    "rank": rk, "peak": round(pk), "dur_h": du, "onset": (None if i0 is None else valid[i0].strftime("%Y-%m-%dT%HZ")),
                    "daily": A.daily_ranks(s, valid)})
    coast = {"model": model, "label": m["label"], "init": init.strftime("%Y-%m-%dT%HZ"),
             "valid": [t.strftime("%Y-%m-%dT%HZ") for t in valid], "lat": tl.tolist(),
             "ivt": np.round(cser).astype(int).tolist(), "points": pts, "levels": levels}
    (dirs["pac"] / "coast.json").write_text(json.dumps(coast, separators=(",", ":")))
    coast_figure(SITE / "assets" / "ar" / f"det_coast_{model}.webp", m, init, valid, cser, tl, levels)

    # ── analysis-time IVT on the 0.5-deg points (validation) ──
    la5 = np.arange(10.0, 70.01, 0.5); lo5 = np.arange(130.0, 300.01, 0.5)
    i5 = A.nearest_idx(lat, la5); j5 = A.nearest_idx(lon, lo5)
    np.savez_compressed(work / f"det_{model}_t0.npz", lat=la5, lon=lo5, ivt=ivt[0][np.ix_(i5, j5)],
                        ivte=ivte[0][np.ix_(i5, j5)], ivtn=ivtn[0][np.ix_(i5, j5)],
                        ivt8=(sub8[0][np.ix_(i5, j5)] if sub8 else ivt[0][np.ix_(i5, j5)]), coast=cser)
    nbytes = {dk: A.dir_bytes(dirs[dk]) for dk in A.DOMAINS}
    part = {"model": model, "label": m["label"], "short": m["short"], "init": init.strftime("%Y-%m-%dT%HZ"),
            "levels": levels, "res": m["res"], "src": m["src"], "steps": list(STEPS), "level_bias": lvl_bias,
            "points": [{k: p[k] for k in ("lat", "lon", "name", "rank", "peak", "dur_h", "onset", "daily")} for p in pts],
            "frames": frames, "bytes_frames": nbytes, "download": A.LEDGER.report(),
            "runtime_s": {"fetch": round(t_fetch), "render": round(t_render), "total": round(time.time() - t0)},
            "max_ivt": round(float(np.nanmax(ivt)))}
    (work / f"det_{model}.json").write_text(json.dumps(part, indent=1))
    import resource
    part["peak_rss_MB"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    (work / f"det_{model}.json").write_text(json.dumps(part, indent=1))
    print(f"== {model}: peak RSS {part['peak_rss_MB']} MB, fetch {t_fetch:.0f}s, render {t_render:.0f}s, frames {sum(nbytes.values()) / 1e6:.1f} MB, "
          f"download {A.LEDGER.report()}", flush=True)


def combine(work: Path):
    """Merge the per-model parts into ar_det_manifest.json. A model that did not run this cycle keeps its previous
    entry (its frames stay on the branch) for up to MAX_AGE_H hours, flagged with its own init time."""
    old = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    parts = {p.stem.split("_", 1)[1]: json.loads(p.read_text()) for p in sorted(work.glob("det_*.json"))}
    if not parts:
        raise SystemExit("no model parts to combine")
    newest = max(pd.Timestamp(p["init"].replace("Z", "")) for p in parts.values())
    regions, models = {}, {}
    for model in A.MODELS:
        if model in parts:
            p = parts[model]
            models[model] = {k: p[k] for k in ("label", "short", "init", "levels", "res", "src", "level_bias", "max_ivt", "points")}
            models[model]["stale"] = False
            for dk, fr in p["frames"].items():
                regions[f"det_{model}_{dk}"] = {"label": f"{p['short']} · {A.DOMAINS[dk]['label']}", "frames": fr}
        elif model in old.get("models", {}):
            om = old["models"][model]
            age = (newest - pd.Timestamp(om["init"].replace("Z", ""))) / pd.Timedelta(hours=1)
            if age <= MAX_AGE_H:
                models[model] = dict(om, stale=True)
                for dk in A.DOMAINS:
                    r = old["regions"].get(f"det_{model}_{dk}")
                    if r:
                        regions[f"det_{model}_{dk}"] = r
                print(f"  {model}: not in this cycle; keeping the {om['init']} run ({age:.0f} h older)", flush=True)
            else:
                print(f"  {model}: not in this cycle and the last run is {age:.0f} h old: dropped from the page", flush=True)
    doc = {"ver": int(time.time()), "default": "det_aifs_pac", "cycle": newest.strftime("%Y-%m-%dT%HZ"),
           "generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "models": models, "regions": regions,
           "steps_h": list(STEPS)}
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(doc, separators=(",", ":")))
    print(f"wrote {MANIFEST} ({len(regions)} loops, models {sorted(models)})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    a1 = sp.add_parser("model")
    a1.add_argument("--model", required=True, choices=list(A.MODELS))
    a1.add_argument("--date", required=True); a1.add_argument("--time", required=True)
    a1.add_argument("--work", default="/tmp/ard"); a1.add_argument("--procs", type=int, default=4)
    a1.add_argument("--wait-min", type=float, default=float(os.environ.get("AR_DET_WAIT_MIN", "40")))
    a2 = sp.add_parser("combine")
    a2.add_argument("--work", default="/tmp/ard")
    a = ap.parse_args()
    work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    if a.cmd == "model":
        run_model(a.model, a.date, f"{int(a.time):02d}", work, a.procs, a.wait_min)
    else:
        combine(work)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(rc or 0)                      # cartopy 0.25 exit segfault (see ar_monitor.py)
