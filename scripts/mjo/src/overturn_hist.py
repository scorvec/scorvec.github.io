#!/usr/bin/env python3
"""The analysed Hadley and Walker 7-day means since July 2025: the growing archive behind the analysis loops.

Daily means of the AIFS-ENS control's step-0 analysis (00 and 12 UTC) of the zonal-mean v and the 5S-5N divergent
zonal wind, reduced exactly as the forecasts are (overturn.reduce_file), kept for good in ARCHIVE on the frames branch.
Each day's frame is the trailing 7-day mean ending that day, less the analysis's own offset from ERA5 (overturn.drift
at lead 0 for AIFS-ENS, so the analysis stands on the footing the forecasts are corrected to), against the ERA5
1991-2020 harmonic climatology at the window's midpoint. The colour scale is fixed (VLIM: the 99th percentile of the
analysed anomalies), because frames are drawn once, named by date (D<YYYYMMDD>.webp) and published additively: the
loops grow by a frame a day and an old frame never changes. The forecast loops (overturn_fc.py) lead in with the
last of these frames.

    python src/overturn_hist.py backfill --since 2025-07-01   # archived analyses from the Google mirror
    python src/overturn_hist.py update                         # new days from the rolling cycle histories
    python src/overturn_hist.py render [--rebuild]             # frames for days not drawn yet, manifests, stills
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pyproj  # noqa: F401  (load pyproj's libproj before xarray/eccodes)
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ecmwf"))
import overturn as O
import mmsf
import walker
from doycal import noleap_doy
from mmsf import LEVELS

REPO = Path(__file__).resolve().parents[3]
ANIM = REPO / "assets" / "sst" / "anim"
ARCHIVE = ANIM / "overturn_hist" / "analysis_daily.npz"
FIRST = pd.Timestamp("2025-07-01")  # the AIFS-ENS open data begins here
WIN = 7
VLIM = {"mmsf": 7.5, "walker": 1.0}  # fixed: p99 of |anomaly| over the analysed weeks (2026-10-16)
P_PA = np.asarray(LEVELS, float) * 100.0
P_HPA = np.asarray(LEVELS, float)
PRODUCTS = {
    "mmsf": {
        "dir": "mmsf_hist",
        "still": "mmsf_anom.webp",
        "label": "Meridional mass streamfunction anomaly (7-day mean, Ψ′ vs ERA5 1991-2020)",
    },
    "walker": {
        "dir": "walker_hist",
        "still": "walker_anom.webp",
        "label": "Walker circulation anomaly (7-day mean, Ψ_W′ vs ERA5 1991-2020)",
    },
}


def load() -> dict:
    if not ARCHIVE.exists():
        return {
            "date": np.array([], "datetime64[D]"),
            "vbar": np.zeros((0, len(LEVELS), O.LAT.size), np.float32),
            "ud": np.zeros((0, len(LEVELS), O.NLON_UD), np.float32),
        }
    with np.load(ARCHIVE) as f:
        return {k: f[k] for k in f.files}


def save(doc: dict, add: dict):
    """Merge new days into the archive (a day already archived keeps its values), sorted by date."""
    have = set(doc["date"].astype("datetime64[D]").tolist())
    keep = [i for i, d in enumerate(add["date"].astype("datetime64[D]").tolist()) if d not in have]
    if not keep:
        return doc
    out = {k: np.concatenate([doc[k], add[k][keep]]) for k in ("date", "vbar", "ud")}
    o = np.argsort(out["date"])
    out = {k: v[o] for k, v in out.items()}
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ARCHIVE, **out)
    print(f"  archive: +{len(keep)} days, {out['date'].size} in all ({out['date'][0]} … {out['date'][-1]})", flush=True)
    return out


def backfill(since, until):
    """Archived AIFS-ENS control analyses (step 0 of the 00 and 12 UTC runs) for every day not yet archived."""
    import store as ecmwf

    doc = load()
    have = set(doc["date"].astype("datetime64[D]").tolist())
    days = [d for d in pd.date_range(max(pd.Timestamp(since), FIRST), until) if np.datetime64(d.date()) not in have]
    print(f"backfill: {len(days)} days", flush=True)
    for n, day in enumerate(days):
        vb, ud = [], []
        for hh in ("00", "12"):
            try:
                p = ecmwf.ensure(
                    ecmwf.Cycle(f"{day:%Y%m%d}", hh),
                    ecmwf.Spec("aifs-ens", "cf", ("u", "v"), "pl", tuple(LEVELS), (0,)),
                )
            except Exception as e:  # noqa: BLE001
                print(f"  {day:%Y-%m-%d} {hh}Z: {type(e).__name__} {str(e)[:70]}", flush=True)
                continue
            v, u = O.reduce_file(p, [0], (0,))
            p.unlink(missing_ok=True)
            vb.append(v[0, 0])
            ud.append(u[0, 0])
        if len(vb) == 2:  # a day is both analyses, as the cycle histories' daily mean
            doc = save(
                doc,
                {
                    "date": np.array([day.date()], "datetime64[D]"),
                    "vbar": np.mean(vb, 0)[None],
                    "ud": np.mean(ud, 0)[None],
                },
            )
        if n % 20 == 0:
            print(f"  {n}/{len(days)} days", flush=True)


def update():
    """Days the rolling cycle histories (mmsf.HIST, walker.HIST, kept by mjo.yml) hold at both 00 and 12 UTC."""
    hv, hu = xr.open_dataarray(mmsf.HIST), xr.open_dataarray(walker.HIST)
    out = {"date": [], "vbar": [], "ud": []}
    for day in sorted(set(pd.DatetimeIndex(hv.time.values).normalize())):
        mv = pd.DatetimeIndex(hv.time.values).normalize() == day
        mu = pd.DatetimeIndex(hu.time.values).normalize() == day
        if mv.sum() < 2 or mu.sum() < 2:
            continue
        out["date"].append(np.datetime64(day.date()))
        out["vbar"].append(hv.values[mv].mean(0))
        out["ud"].append(hu.values[mu].mean(0))
    if out["date"]:
        save(load(), {k: np.asarray(v) for k, v in out.items()})


def week(doc: dict, end: pd.Timestamp, ref):
    """(vbar, ud, midpoint doy) of the 7-day window ending `end`, less the analysis offset; None if a day is missing."""
    want = pd.date_range(end - pd.Timedelta(days=WIN - 1), end)
    idx = {d: i for i, d in enumerate(pd.DatetimeIndex(doc["date"]))}
    if any(d not in idx for d in want):
        return None
    rows = [idx[d] for d in want]
    mid = float(noleap_doy(pd.DatetimeIndex([end - pd.Timedelta(days=(WIN - 1) / 2)]))[0])
    dv, du = O.drift(ref, "aifs", mid, "vbar"), O.drift(ref, "aifs", mid, "ud")
    return doc["vbar"][rows].mean(0) - dv[0][0], doc["ud"][rows].mean(0) - du[0][0], mid


def frame_name(d: pd.Timestamp) -> str:
    return f"D{d:%Y%m%d}.webp"


def render(rebuild: bool = False):
    """A frame per archived day with a full window; only days without a drawn frame unless rebuild. Writes the manifests
    and the stills (the latest frame) and returns the dates drawn."""
    doc, ref = load(), O.load_ref()
    if O.drift(ref, "aifs", 1.0, "vbar") is None:
        raise SystemExit("no AIFS-ENS drift reference: the analysis offset is unknown")
    cm = xr.open_dataarray(mmsf.CLIM)
    lon = np.linspace(0.0, 360.0, O.NLON_UD)  # the chi solver's DH2 longitudes (walker.HIST's axis)
    cw = xr.open_dataarray(walker.CLIM).interp(longitude=lon, kwargs={"fill_value": None})
    drawn = {}
    for prod in PRODUCTS:
        mp = ANIM / f"{PRODUCTS[prod]['dir']}_manifest.json"
        drawn[prod] = (
            {}
            if rebuild or not mp.exists()
            else {f["date"]: f for f in json.loads(mp.read_text())["regions"][PRODUCTS[prod]["dir"]]["frames"]}
        )
        (ANIM / PRODUCTS[prod]["dir"]).mkdir(parents=True, exist_ok=True)
    new = 0
    for end in pd.DatetimeIndex(doc["date"]):
        key = f"{end:%Y-%m-%d}"
        if all(key in drawn[p] for p in PRODUCTS):
            continue
        w = week(doc, end, ref)
        if w is None:
            continue
        vb, ud, mid = w
        psi, pc = mmsf.streamfunction(vb, P_PA, O.LAT), mmsf.clim_psi(cm, mid)
        hs = mmsf.hadley_strength(psi, pc, P_HPA, O.LAT, doy=mid)
        mmsf.plot_psi(
            psi - pc,
            P_HPA,
            O.LAT,
            ANIM / "mmsf_hist" / frame_name(end),
            f"Meridional mass streamfunction anomaly Ψ′ — 7-day mean ending {end:%Y-%m-%d}",
            anom=True,
            vlim=VLIM["mmsf"],
            psi_abs=psi,
            texts=mmsf.strength_texts(hs),
            foot="colour = Ψ′ anomaly vs ERA5 1991–2020 (the analysis less its mean offset from ERA5)   ·   "
            "black contours = absolute Ψ (10¹⁰ kg s⁻¹)   ·   arrows = vertical motion (up = ascent; length ∝ strength)",
        )
        drawn["mmsf"][key] = {
            "file": frame_name(end),
            "date": key,
            "label": f"week ending {end:%a %b %d, %Y}",
            "hadley": hs,
        }
        pw, pcw = walker.streamfunction(ud, P_PA), walker.clim_psi(cw, mid)
        pidx, pcl = walker.pac_index(pw, P_HPA, lon), walker.pac_index(pcw, P_HPA, lon)
        walker.plot_psi(
            pw - pcw,
            P_HPA,
            lon,
            ANIM / "walker_hist" / frame_name(end),
            f"Walker circulation anomaly Ψ_W′ — 7-day mean ending {end:%Y-%m-%d}",
            VLIM["walker"],
            pw,
            pidx,
            pcl,
            psi_clim=pcw,
        )
        drawn["walker"][key] = {
            "file": frame_name(end),
            "date": key,
            "label": f"week ending {end:%a %b %d} · Pacific {pidx:+.1f}",
        }
        new += 1
    for prod, cfg in PRODUCTS.items():
        frames = [dict(f, idx=i) for i, (_, f) in enumerate(sorted(drawn[prod].items()))]
        if not frames:
            continue
        doc_m = {
            "ver": int(pd.Timestamp.now("UTC").timestamp()),
            "days": len(frames),
            "regions": {cfg["dir"]: {"label": cfg["label"], "n_frames": len(frames), "frames": frames}},
        }
        if prod == "mmsf":
            doc_m["hadley_latest"] = frames[-1]["hadley"]
        (ANIM / f"{cfg['dir']}_manifest.json").write_text(json.dumps(doc_m))
        latest = ANIM / cfg["dir"] / frames[-1]["file"]
        if latest.exists():  # the still is the newest frame (drawn this run: older ones are on the frames branch only)
            shutil.copyfile(latest, REPO / "assets" / "sst" / cfg["still"])
    print(f"  drew {new} new day(s); loops hold {len(drawn['mmsf'])} weeks", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backfill")
    b.add_argument("--since", default=str(FIRST.date()))
    b.add_argument("--until", default=str((pd.Timestamp.now("UTC").tz_localize(None) - pd.Timedelta(days=1)).date()))
    sub.add_parser("update")
    r = sub.add_parser("render")
    r.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    if a.cmd == "backfill":
        backfill(a.since, a.until)
    elif a.cmd == "update":
        update()
    else:
        render(a.rebuild)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
