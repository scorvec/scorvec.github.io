#!/usr/bin/env python3
"""GEM collector for the conservation-and-drag intercomparison: GDPS and GEPS 00Z inits from the MSC Datamart,
reduced IN THE RUNNER to the compact AAM-budget diagnostics of common.py, oldest init first, before the Datamart's
30-day window drops them. Only the small diagnostic tables are kept (data/gem/<model>/<YYYYMMDD>.npz); no GRIB.

  GDPS  every 00Z init, 0-240 h: p_s, 10 m u/v every 6 h; u on all 33 published levels (1-1015 hPa) and v on the
        14 common levels every 24 h; the model's own orography (GeopotentialHeight_Sfc at 000 h). ~0.7 GB per init.
  GEPS  00Z inits every GEPS_EVERY-th day from WINDOW_START, 0-360 h: PRES_SFC_0, 10 m u/v every 6 h; u and v on its
        12 published levels (10-1000 hPa) every 24 h; HGT_SFC_0 at 000 h. One file holds all 21 members; the control
        and perturbed members 1-4 are reduced (GEPS_MEMBERS). ~2.1 GB per init.
Neither publishes a surface stress, so their friction is the common bulk formula only (common.py).

Politeness: the honest site User-Agent (scripts/lib/webget.py), at most 4 files in flight (one per worker process),
each file fetched once, retries with backoff on 429/5xx only, a byte ledger per UTC day (data/gem/ledger.json) with a
daily cap (AMDRAG_DAY_CAP_GB, default 20) and a per-run cap; `--next` processes ONE init and exits 0, or exits 3 when
nothing is due or a cap is reached (the workflow loops and pushes between inits).

Data: ECCC GDPS and GEPS via the MSC Datamart. Contains information licensed under the Open Government Licence -
Canada (https://eccc-msc.github.io/open-data/licence/readme_en/).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import time
import traceback
import urllib.error
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "lib"))
import common as C                                                     # noqa: E402
import webget                                                          # noqa: E402

BASE = "https://dd.weather.gc.ca"
OUT = C.DATA / "gem"
LEDGER = OUT / "ledger.json"
WINDOW_START = dt.date(2026, 9, 14)
GEPS_EVERY = int(os.environ.get("GEPS_EVERY", "2"))
GEPS_MEMBERS = [int(x) for x in os.environ.get("GEPS_MEMBERS", "0,1,2,3,4").split(",")]
WORKERS = int(os.environ.get("AMDRAG_WORKERS", "4"))
RUN_ID = os.environ.get("GITHUB_RUN_ID", "")
DAY_CAP = float(os.environ.get("AMDRAG_DAY_CAP_GB", "20")) * 1e9
UA = {"User-Agent": webget.DEFAULT_UA}
GDPS_LEVELS = [1, 5, 10, 20, 30, 50, 100, 150, 175, 200, 225, 250, 275, 300, 350, 400, 450, 500, 550, 600, 650, 700,
               750, 800, 850, 875, 900, 925, 950, 970, 985, 1000, 1015]
MODELS = {
    "gdps": dict(
        url=BASE + "/{d}/WXO-DD/model_gdps/15km/00/{L:03d}/{d}T00Z_MSC_GDPS_{f}_LatLon0.15_PT{L:03d}H.grib2",
        sp="Pressure_Sfc", u10="WindU_AGL-10m", v10="WindV_AGL-10m", oro="GeopotentialHeight_Sfc",
        u="WindU_IsbL-{L:04d}", v="WindV_IsbL-{L:04d}", ulev=GDPS_LEVELS, vlev=C.C14, last=240, members=None,
        est=0.75e9),
    "geps": dict(
        url=BASE + "/{d}/WXO-DD/ensemble/geps/grib2/raw/00/{L:03d}/CMC_geps-raw_{f}_latlon0p5x0p5_{d}00_P{L:03d}"
                   "_allmbrs.grib2",
        sp="PRES_SFC_0", u10="UGRD_TGL_10m", v10="VGRD_TGL_10m", oro="HGT_SFC_0",
        u="UGRD_ISBL_{L:04d}", v="VGRD_ISBL_{L:04d}", ulev=C.C12, vlev=C.C12, last=360, members=GEPS_MEMBERS,
        est=2.2e9),
}


# ---------------------------------------------------------------------------------------------------------------
def http_get(u: str, timeout: int = 300) -> bytes | None:
    try:
        return webget.get(u, timeout=timeout, tries=5, backoff=8.0)
    except urllib.error.HTTPError as e:
        if e.code != 404:
            print(f"    HTTP {e.code} {u.rsplit('/', 1)[-1]}", flush=True)
        return None
    except Exception as e:                                             # noqa: BLE001
        print(f"    failed {u.rsplit('/', 1)[-1]}: {str(e)[:80]}", flush=True)
        return None


def http_head(u: str) -> bool:
    try:
        with urllib.request.urlopen(urllib.request.Request(u, method="HEAD", headers=UA), timeout=60) as r:
            return r.status == 200
    except Exception:                                                  # noqa: BLE001
        return False


def datamart_dates() -> list[str]:
    html = webget.get(BASE + "/", timeout=60).decode()
    return sorted(set(re.findall(r'href="(\d{8})/"', html)))


def furl(model: str, d: str, L: int, f: str) -> str:
    return MODELS[model]["url"].format(d=d, L=L, f=f)


def decode(blob: bytes, members=None):
    """GRIB2 bytes -> ({member: 2-D float array, lat ascending}, lat, lon). members None = the single message."""
    import eccodes as ec
    out = {}
    lat = lon = None
    pos = 0
    n = len(blob)
    while pos < n:
        i = blob.find(b"GRIB", pos)
        if i < 0:
            break
        ln = int.from_bytes(blob[i + 8:i + 16], "big")                  # GRIB2 section 0 total length
        h = ec.codes_new_from_message(blob[i:i + ln])
        pos = i + ln
        try:
            if members is not None:
                try:
                    m = int(ec.codes_get(h, "perturbationNumber"))
                except Exception:                                      # noqa: BLE001
                    m = int(ec.codes_get(h, "number"))
                if m not in members:
                    continue
            else:
                m = 0
            Ni, Nj = ec.codes_get(h, "Ni"), ec.codes_get(h, "Nj")
            la0 = ec.codes_get(h, "latitudeOfFirstGridPointInDegrees")
            la1 = ec.codes_get(h, "latitudeOfLastGridPointInDegrees")
            lo0 = ec.codes_get(h, "longitudeOfFirstGridPointInDegrees")
            di = ec.codes_get(h, "iDirectionIncrementInDegrees")
            v = ec.codes_get_values(h).reshape(Nj, Ni).astype(np.float64)
            if ec.codes_get(h, "bitmapPresent"):
                miss = ec.codes_get(h, "missingValue")
                v[v == miss] = np.nan
            la = np.linspace(la0, la1, Nj)
            if la[0] > la[-1]:
                la, v = la[::-1], v[::-1]
            lo = lo0 + di * np.arange(Ni)
            if abs(di * Ni - 360.0 - di) < 1e-3:                       # both ends of the circle: drop the repeat
                lo, v = lo[:-1], v[:, :-1]
            lat, lon = la, lo
            out[m] = v
        finally:
            ec.codes_release(h)
    return out, lat, lon


def month_of(d: str, step: int) -> int:
    t = dt.datetime.strptime(d, "%Y%m%d") + dt.timedelta(hours=step)
    return t.month


# ---- worker side -----------------------------------------------------------------------------------------------
_W = {}


def _winit(model, d, h, hlat, hlon):
    _W.update(model=model, d=d, h=h, hlat=hlat, hlon=hlon, red=None)


def _reducer(lat, lon):
    if _W["red"] is None:
        assert len(lat) == len(_W["hlat"]) and np.allclose(lat, _W["hlat"]) and np.allclose(lon, _W["hlon"]), \
            "orography grid differs from the field grid"
        _W["red"] = C.Reducer(C.Grid(lat, lon), _W["h"])
    return _W["red"]


def work_step(step: int, levels: bool):
    """Fetch + decode + reduce one step for every kept member. -> dict (no exceptions escape)."""
    model, d = _W["model"], _W["d"]
    M = MODELS[model]
    mem = M["members"]
    t0 = time.time()
    res = {"step": step, "bytes": 0, "missing": [], "T": {}, "L": {}, "nan": {}}
    try:
        def get(f):
            b = http_get(furl(model, d, step, f))
            if b is None:
                res["missing"].append(f)
                return None
            res["bytes"] += len(b)
            fld, lat, lon = decode(b, mem)
            if f.startswith(("WindU_IsbL", "WindV_IsbL", "UGRD_ISBL", "VGRD_ISBL")):
                fld = {k: x.astype(np.float32) for k, x in fld.items()}
            return fld, lat, lon
        sp = get(M["sp"]); u10 = get(M["u10"]); v10 = get(M["v10"])
        U = V = None
        if levels:
            U = {L: get(M["u"].format(L=L)) for L in M["ulev"]}
            V = {L: get(M["v"].format(L=L)) for L in M["vlev"]}
        if sp is None or u10 is None or v10 is None:
            return res
        lat, lon = sp[1], sp[2]
        red = _reducer(lat, lon)
        runs = sorted(sp[0]) if mem is not None else [0]
        mon = month_of(d, step)
        for m in runs:
            if m not in u10[0] or m not in v10[0]:
                res["missing"].append(f"member {m} sfc")
                continue
            s_, a_, b_ = sp[0][m], u10[0][m], v10[0][m]
            if np.nanmean(s_) < 2000.0:                                  # hPa, not Pa
                s_ = s_ * 100.0
                res["nan"]["sp_hPa"] = 1
            bad = int(np.isnan(s_).sum() + np.isnan(a_).sum() + np.isnan(b_).sum())
            if bad:
                res["nan"][f"{m}:sfc"] = bad
                continue
            res["T"][m] = red.torque_step(mon, s_, a_, b_, maps=step in C.MAP_STEPS)
            if levels:
                if any(U[L] is None or m not in U[L][0] for L in M["ulev"]):
                    continue
                uu = np.stack([U[L][0][m] for L in M["ulev"]])
                nn = int(np.isnan(uu).sum())
                if nn:                                    # below-ground points left missing: zero weight anyway
                    res["nan"][f"{m}:u"] = nn
                    uu = np.nan_to_num(uu)
                vv = {L: np.nan_to_num(V[L][0][m]) for L in M["vlev"] if V[L] is not None and m in V[L][0]}
                res["L"][m] = red.level_step(M["ulev"], uu, vv, s_, lb=step in C.LB_STEPS,
                                             maps=step in C.MAP_STEPS)
    except Exception:                                                  # noqa: BLE001
        res["error"] = traceback.format_exc()[-600:]
    res["sec"] = time.time() - t0
    return res


# ---- one init ---------------------------------------------------------------------------------------------------
def process(model: str, d: str) -> dict:
    M = MODELS[model]
    t0 = time.time()
    meta = {"model": model, "date": d, "bytes": 0, "missing": [], "nan": {}, "errors": []}
    b = http_get(furl(model, d, 0, M["oro"]))
    if b is None:
        raise RuntimeError(f"{model} {d}: no orography file")
    meta["bytes"] += len(b)
    fld, hlat, hlon = decode(b, M["members"])
    h = fld[min(fld)]
    g = C.Grid(hlat, hlon)
    hm = float((h * g.area[:, None]).sum() / (g.area.sum() * len(hlon)))
    meta["oro"] = {"area_mean_m": hm, "max_m": float(np.nanmax(h)), "min_m": float(np.nanmin(h)),
                   "nlat": len(hlat), "nlon": len(hlon), "lon0": float(hlon[0]), "dlon": float(hlon[1] - hlon[0]),
                   "scales": C.orography_stats(g, h)}
    print(f"  {model} {d}: orography {len(hlat)}x{len(hlon)}, area-mean {hm:.1f} m, max {np.nanmax(h):.0f} m",
          flush=True)
    if not (150 < hm < 330 and np.nanmax(h) > 5000):                 # T_m is linear in h: a unit slip is fixable
        print(f"::warning::{model} {d} orography does not look like metres (mean {hm:.1f}, max {np.nanmax(h):.0f})")
        meta["oro"]["suspect"] = True
    steps = list(range(0, M["last"] + 1, 6))
    T, L = {}, {}
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=_winit, initargs=(model, d, h, hlat, hlon)) as ex:
        futs = [ex.submit(work_step, s, s % 24 == 0) for s in steps]
        for f in futs:
            r = f.result()
            meta["bytes"] += r["bytes"]
            if r["missing"]:
                meta["missing"].append([r["step"], r["missing"][:6]])
            if r["nan"]:
                meta["nan"][str(r["step"])] = r["nan"]
            if "error" in r:
                meta["errors"].append([r["step"], r["error"]])
                print(f"    step {r['step']} ERROR {r['error'][-300:]}", flush=True)
            for m, o in r["T"].items():
                T.setdefault(m, {})[r["step"]] = o
            for m, o in r["L"].items():
                L.setdefault(m, {})[r["step"]] = o
            if r["step"] % 48 == 0:
                print(f"    step {r['step']:3d}: {meta['bytes'] / 1e9:.2f} GB, {time.time() - t0:.0f} s "
                      f"(step {r['sec']:.0f} s)", flush=True)
    pack = {"grid|area_b": g.bands(g.area * len(hlon)), "grid|lat": hlat[[0, -1]], "grid|lon": hlon[[0, -1]]}
    runs = sorted(T)
    for m in runs:
        run = model if M["members"] is None else f"{model}:{m}"
        pack.update(C.pack_run(run, T[m], L.get(m, {})))
    meta["runs"] = [model if M["members"] is None else f"{model}:{m}" for m in runs]
    meta["levels_u"] = M["ulev"]
    meta["levels_v"] = M["vlev"]
    meta["n_T"] = {str(m): len(T[m]) for m in runs}
    meta["n_L"] = {str(m): len(L.get(m, {})) for m in runs}
    meta["seconds"] = round(time.time() - t0, 1)
    meta["made"] = dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%MZ")
    meta["complete"] = (not meta["missing"] and not meta["errors"] and
                        all(len(T[m]) == len(steps) and len(L.get(m, {})) == len(steps[::4]) for m in runs) and
                        len(runs) == (1 if M["members"] is None else len(M["members"])))
    return meta, pack


# ---- queue ------------------------------------------------------------------------------------------------------
def load_ledger():
    if LEDGER.exists():
        return json.loads(LEDGER.read_text())
    return {"bytes_by_day": {}, "inits": {}}


def save_ledger(lg):
    OUT.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(lg, indent=1, sort_keys=True))


def due(lg, dates):
    """(model, date) still to do, oldest first; GDPS before GEPS on the same date."""
    q = []
    for d in dates:
        dd = dt.datetime.strptime(d, "%Y%m%d").date()
        if dd < WINDOW_START:
            continue
        for model in ("gdps", "geps"):
            if model == "geps" and (dd - WINDOW_START).days % GEPS_EVERY:
                continue
            if (OUT / model / f"{d}.npz").exists():
                continue
            rec = lg["inits"].get(f"{model}:{d}", {})
            if rec.get("gave_up") or (RUN_ID and rec.get("failed_run") == RUN_ID):
                continue
            q.append((model, d))
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--next", action="store_true", help="process the oldest due init, then exit")
    ap.add_argument("--run-cap-gb", type=float, default=float(os.environ.get("AMDRAG_RUN_CAP_GB", "20")))
    ap.add_argument("--run-bytes", type=float, default=float(os.environ.get("AMDRAG_RUN_BYTES", "0")),
                    help="bytes already fetched by this workflow run")
    ap.add_argument("--only", default="", help="model:date to process (testing)")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    lg = load_ledger()
    dates = datamart_dates()
    q = due(lg, dates)
    if a.only:
        m, d = a.only.split(":")
        q = [(m, d)]
    if a.list:
        print("Datamart dates", dates[0], "..", dates[-1], f"({len(dates)})")
        for m, d in q:
            print(" due", m, d)
        return 0
    today = dt.datetime.utcnow().strftime("%Y-%m-%d")
    used = lg["bytes_by_day"].get(today, 0)
    for model, d in q:
        M = MODELS[model]
        if used + M["est"] > DAY_CAP:
            print(f"daily cap: {used / 1e9:.1f} GB used today, next init needs ~{M['est'] / 1e9:.1f} GB")
            return 8
        if a.run_bytes + M["est"] > a.run_cap_gb * 1e9:
            print(f"run cap reached ({a.run_bytes / 1e9:.1f} GB)")
            return 7
        # complete yet? (the 00Z run's last step must be on the Datamart)
        if not http_head(furl(model, d, M["last"], M["sp"])):
            print(f"  {model} {d}: last step {M['last']} not published (yet, or already gone) - skip")
            if d == dates[-1]:
                continue
            rec = lg["inits"].setdefault(f"{model}:{d}", {})
            rec["absent"] = rec.get("absent", 0) + 1
            if rec["absent"] >= 3:
                rec["gave_up"] = "last step absent on 3 checks"
            save_ledger(lg)
            continue
        print(f"== {model} {d} (oldest Datamart date {dates[0]})", flush=True)
        key = f"{model}:{d}"
        rec = lg["inits"].setdefault(key, {})
        rec["attempts"] = rec.get("attempts", 0) + 1
        try:
            meta, pack = process(model, d)
        except Exception as e:                                         # noqa: BLE001
            traceback.print_exc()
            rec["last_error"] = str(e)[:300]
            rec["failed_run"] = RUN_ID
            if rec["attempts"] >= 3:
                rec["gave_up"] = "3 failed attempts"
            save_ledger(lg)
            return 4
        lg["bytes_by_day"][today] = lg["bytes_by_day"].get(today, 0) + meta["bytes"]
        rec.update(bytes=meta["bytes"], seconds=meta["seconds"], complete=meta["complete"],
                   missing=len(meta["missing"]), runs=len(meta["runs"]))
        expiring = d == dates[0] or d == dates[min(1, len(dates) - 1)]
        if meta["complete"] or rec["attempts"] >= 2 or expiring:
            (OUT / model).mkdir(parents=True, exist_ok=True)
            np.savez_compressed(OUT / model / f"{d}.npz", meta=json.dumps(meta), **pack)
            sz = (OUT / model / f"{d}.npz").stat().st_size
            rec["kb"] = sz // 1024
            print(f"  saved {model}/{d}.npz ({sz / 1e3:.0f} kB): {meta['bytes'] / 1e9:.2f} GB fetched in "
                  f"{meta['seconds']:.0f} s, complete={meta['complete']}, missing={meta['missing'][:3]}", flush=True)
        else:
            print(f"  {model} {d} incomplete ({meta['missing'][:3]} {meta['errors'][:1]}) - retry next run", flush=True)
        save_ledger(lg)
        print(f"BYTES {meta['bytes']}")
        return 0
    print("nothing due")
    return 3


if __name__ == "__main__":
    sys.exit(main())
