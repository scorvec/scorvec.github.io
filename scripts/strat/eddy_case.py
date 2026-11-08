#!/usr/bin/env python3
"""A forecast case for the eddy-feedback study (eddy_feedback.py, eddy_feedback_gefs.py): one target week over North
America and the Atlantic, run by run and member by member. Did later runs bring stronger eddies, breaking more
cyclonically, pushing the jet south and the NAO negative, and does the same chain hold across each ensemble's members?
IFS-ENS (50 members; its open data has no control at pressure levels), AIFS-ENS (51) and GEFS (31), 00Z runs; once the
week has passed, the IFS analysis (the IFS control at 0 and 6 h from 00 and 12Z: every 6 hours).

FIELDS, 6-hourly from a day before the week to its end, on the polar-cap product's 1.5 deg cells (polarcap.coarse):
u and v at 250 hPa, 500 hPa height, sea-level pressure. ECMWF from the Google Cloud mirror (the 90-18N rows of each
message, store.ensure_rows), GEFS from NOAA's AWS bucket by byte range.

PER MEMBER, over the week; sector = 120W-0, box = 30-70N in it:
  k24        24-hour-difference eddy kinetic energy 1/4 <(du)^2 + (dv)^2> over the box (Wallace et al. 1988's filter:
             synoptic eddies), m2 s-2
  eke        eddy kinetic energy of the departures from the week's mean, 1/2 <u'2 + v'2>, box
  jet_lat    latitude of the sector-mean week-mean 250 hPa wind's maximum, 25-70N; jet_speed its speed
  cwb        cyclonic breaking: -<u'v'> 5-20 deg poleward of the jet (eddies tilted NW-SE carry momentum south there;
             Riviere 2009), sector mean, m2 s-2
  awb        anticyclonic breaking: +<u'v'> 5-20 deg equatorward of the jet (NE-SW tilt)
  shift      the jet shift the eddies drive, deg per day (+ = poleward): -sum F dU/dlat / sum (dU/dlat)^2 over
             25-75N (cos-weighted), F = -(1 / (a cos^2)) d(<u'v'> cos^2) / dlat of the sector means (zonal flux terms
             left out)
  nao, ao    the site's daily indices (polarcap.teleconnections: CPC's patterns, ERA5 1991-2020 normals, seasonal sd),
             week mean
  gbi        Greenland blocking: 500 hPa height anomaly over 60-80N, 80-20W (Hanna et al. 2016), week mean, m

COMBINE: every measure's members by run; the change from the runs made a week or more before the week to the latest
two (Welch t, Benjamini-Hochberg 10 % over measures and models); the chain inside each ensemble: r of the members'
departures from their run's mean, pooled over runs, members permuted within runs (10000 times), BH over pairs and
models; the analysis's percentile in each run. WHERE: the sector-mean 250 hPa wind and -u'v' by latitude, early runs
against the latest two (Welch t at each latitude, BH 10 % over latitudes, both fields and the models), the synoptic
eddy energy by latitude and in latitude bands (ensemble means: the reductions keep its map, not each member's), and
maps of the change (the mean of the ensembles).

    python eddy_case.py member --model ifs --init 2026110100 --start 2026-11-09 --days 7 --out runs/
    python eddy_case.py member --model analysis --start 2026-11-09 --days 7 --out runs/
    python eddy_case.py combine --runs runs/ --start 2026-11-09 --days 7 --out reference/eddy_case/
"""

from __future__ import annotations

import pyproj  # noqa: F401  (load pyproj's libproj before xarray/eccodes)

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gefs_eddy_extract as GX
import polarcap as PC  # (puts scripts/lib and scripts/ecmwf on the path)

A = 6.371e6
DAY = 86400.0
SECTOR = (240.0, 360.0)
BOX = (30.0, 70.0)
GBI = (60.0, 80.0, 280.0, 340.0)
JET = (25.0, 70.0)
BAND = (5.0, 20.0)
SHIFT = (25.0, 75.0)
LAG = 4  # 24 h of 6-hourly steps
K24_BANDS = ((19.5, 30.0), (30.0, 45.0), (45.0, 60.0), (60.0, 75.0))
VIEW = (-150.0, 30.0, 19.5, 85.0)  # the maps' extent, degrees east and north
GEFS_PDS = "https://noaa-gefs-pds.s3.amazonaws.com"
GEFS_MEMBERS = ("gec00", *(f"gep{k:02d}" for k in range(1, 31)))
ECMWF = {"ifs": ("ifs", ("pf",)), "aifs": ("aifs-ens", ("cf", "pf"))}
FIELDS = {"u": ("pl", (250,)), "v": ("pl", (250,)), "gh": ("pl", (500,)), "msl": ("sfc", ())}
PARAM = {"aifs-ens": {"gh": ("z", 1 / 9.80665)}}  # AIFS-ENS publishes geopotential, not height
SCALE = {"msl": 0.01}  # Pa -> hPa
GEFS_KEYS = {"u": ("u", 250), "v": ("v", 250), "gh": ("z", 500), "msl": ("msl", 0)}
MEASURES = ("k24", "eke", "jet_lat", "jet_speed", "cwb", "awb", "shift", "nao", "ao", "gbi")
CHAIN = (
    ("k24", "cwb"), ("cwb", "shift"), ("shift", "jet_lat"), ("jet_lat", "nao"), ("cwb", "nao"), ("k24", "nao"),
    ("awb", "nao"), ("cwb", "gbi"),
)  # fmt: skip
NAMES = {"ifs": "IFS-ENS", "aifs": "AIFS-ENS", "gefs": "GEFS", "analysis": "IFS analysis"}
SOURCES = "ECMWF open data (CC BY 4.0), Google Cloud mirror; NOAA GEFS (AWS Open Data); ERA5 normals"
PERM = 10000
RNG = np.random.default_rng(20261108)
LOCK = threading.Lock()  # numpy reductions stay out of the fetch threads' way (BLAS in threads)


def window(start: pd.Timestamp, days: int) -> pd.DatetimeIndex:
    """6-hourly from a day before the week to its end."""
    return pd.date_range(start - pd.Timedelta(days=1), start + pd.Timedelta(days=days), freq="6h")


def hours(t: pd.Timestamp, base: pd.Timestamp) -> int:
    return int((t - base) / pd.Timedelta(hours=1))


# --------------------------------------------------------------------------------------------------------- fetching ---
def ecmwf(model: str, init: pd.Timestamp, times: pd.DatetimeIndex) -> tuple[dict, list]:
    """{field: (member, time, CLAT, CLON)} and the member numbers of an ECMWF ensemble run."""
    import store as E

    name, kinds = ECMWF[model]
    cyc, steps = E.Cycle(f"{init:%Y%m%d}", f"{init:%H}"), tuple(hours(t, init) for t in times)
    out, members = {}, None
    for f, (lev, levels) in FIELDS.items():
        param, scale = PARAM.get(name, {}).get(f, (f, SCALE.get(f, 1.0)))
        parts, mems = [], []
        for kind in kinds:
            spec = E.Spec(name, kind, param, lev, levels, steps)
            p = E.ensure_rows(cyc, spec, PC.ROWS, reduce=PC._cells, tag="cells")
            a, m, s, _ = E.load_rows(p)
            if s != list(steps):
                raise SystemExit(f"{model} {init:%Y%m%d%H} {f}: steps {s} are not {list(steps)}")
            parts.append(a[:, :, 0] * scale)
            mems += [int(x) for x in m]
        out[f] = np.concatenate(parts)
        if members not in (None, mems):
            raise SystemExit(f"{model} {init:%Y%m%d%H}: members differ between fields")
        members = mems
    return out, members


def gefs(init: pd.Timestamp, times: pd.DatetimeIndex) -> tuple[dict, list]:
    """The same for a GEFS run (control + 30 members, 0.5 deg), four messages per file by byte range."""
    steps = [hours(t, init) for t in times]
    out = {
        f: np.full((len(GEFS_MEMBERS), len(steps), PC.CLAT.size, PC.CLON.size), np.nan, "float32") for f in GEFS_KEYS
    }

    def one(job: tuple[int, int]) -> None:
        i, j = job
        h = f"{init:%H}"
        url = f"{GEFS_PDS}/gefs.{init:%Y%m%d}/{h}/atmos/pgrb2ap5/{GEFS_MEMBERS[i]}.t{h}z.pgrb2a.0p50.f{steps[j]:03d}"
        idx = GX.index(url)
        blobs = {f: GX.get(url, idx[(var, lev, steps[j])]) for f, (var, lev) in GEFS_KEYS.items()}
        with LOCK:
            for f, blob in blobs.items():
                x, lat, lon = GX.decode(blob)
                out[f][i, j] = PC.coarse(x, lat, lon) * SCALE.get(f, 1.0)

    with ThreadPoolExecutor(16) as ex:
        list(ex.map(one, [(i, j) for i in range(len(GEFS_MEMBERS)) for j in range(len(steps))]))
    return out, list(range(len(GEFS_MEMBERS)))


def analysis(times: pd.DatetimeIndex) -> tuple[dict, list]:
    """The IFS analysis every 6 hours: the IFS control at 0 h, and at 6 h for 06 and 18Z, from the 00 and 12Z runs."""
    import store as E

    out = {f: np.full((1, len(times), PC.CLAT.size, PC.CLON.size), np.nan, "float32") for f in FIELDS}
    for base in sorted({t.floor("12h") for t in times}):
        cyc = E.Cycle(f"{base:%Y%m%d}", f"{base:%H}")
        for f, (lev, levels) in FIELDS.items():
            p = E.ensure_rows(cyc, E.Spec("ifs", "fc", f, lev, levels, (0, 6)), PC.ROWS, reduce=PC._cells, tag="cells")
            a, _, s, _ = E.load_rows(p)
            for j, t in enumerate(times):
                if t.floor("12h") == base:
                    out[f][0, j] = a[0, s.index(hours(t, base)), 0] * SCALE.get(f, 1.0)
    return out, [0]


# -------------------------------------------------------------------------------------------------------- measures ---
def box_mean(x: np.ndarray, la: tuple, lo: tuple) -> np.ndarray:
    lat, lon = PC.CLAT, PC.CLON
    m = ((lat >= la[0]) & (lat <= la[1]))[:, None] & ((lon >= lo[0]) & (lon < lo[1]))[None]
    w = np.where(m, np.cos(np.deg2rad(lat))[:, None], 0.0)
    return (x * w).sum((-2, -1)) / w.sum()


def jet(us: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Latitude and speed of the maximum of us (member, lat) in JET, refined by a parabola through three points."""
    j = np.flatnonzero((lat >= JET[0]) & (lat <= JET[1]))
    k = j[np.argmax(us[:, j], 1)]
    r = np.arange(len(us))
    y0, y1, y2 = us[r, k - 1], us[r, k], us[r, k + 1]
    den = y0 - 2 * y1 + y2
    off = np.clip(np.where(den != 0, 0.5 * (y0 - y2) / np.where(den != 0, den, 1), 0.0), -0.5, 0.5)
    return lat[k] + off * (lat[1] - lat[0]), y1 - 0.25 * (y0 - y2) * off


def band(prof: np.ndarray, lat: np.ndarray, centre: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Mean of each member's profile between centre + lo and centre + hi degrees."""
    return np.array([np.interp(np.linspace(c + lo, c + hi, 21), lat, p).mean() for p, c in zip(prof, centre)])


def diagnose(F: dict, times: pd.DatetimeIndex, start: pd.Timestamp) -> dict:
    lat = PC.CLAT
    N = PC.Normals()
    za = F["gh"] - np.stack([N.at("z500", t) for t in times])[None]
    ma = F["msl"] - np.stack([N.at("msl", t) for t in times])[None]
    tele = PC.teleconnections({"z500": za, "msl": ma}, times)
    iw = np.flatnonzero(times >= start)
    u, v = F["u"].astype(np.float64), F["v"].astype(np.float64)
    du, dv = u[:, iw] - u[:, iw - LAG], v[:, iw] - v[:, iw - LAG]
    k24map = 0.25 * (du**2 + dv**2)  # (member, time, lat, lon)
    uw, vw = u[:, iw], v[:, iw]
    U = uw.mean(1)
    up, vp = uw - U[:, None], vw - vw.mean(1)[:, None]
    ekemap, uvmap = 0.5 * (up**2 + vp**2).mean(1), (up * vp).mean(1)
    us, uvs = sector(U), sector(uvmap)
    phi, cw = np.deg2rad(lat), np.cos(np.deg2rad(lat))
    with np.errstate(divide="ignore", invalid="ignore"):
        f = -np.gradient(uvs * cw**2, phi, axis=-1) / (A * cw**2) * DAY  # m/s per day
    f[:, ~np.isfinite(f).all(0)] = np.nan  # the pole
    uy = np.gradient(us, lat, axis=-1)  # m/s per degree
    s = (lat >= SHIFT[0]) & (lat <= SHIFT[1])
    jl, js = jet(us, lat)
    gbi_t = box_mean(za, GBI[:2], GBI[2:])
    sc = {
        "k24": box_mean(k24map, BOX, SECTOR).mean(1),
        "eke": box_mean(ekemap, BOX, SECTOR),
        "jet_lat": jl,
        "jet_speed": js,
        "cwb": -band(uvs, lat, jl, BAND[0], BAND[1]),
        "awb": band(uvs, lat, jl, -BAND[1], -BAND[0]),
        "shift": -(f[:, s] * uy[:, s] * cw[s]).sum(1) / (uy[:, s] ** 2 * cw[s]).sum(1),
        "nao": np.nanmean(tele["nao"][:, iw], 1),
        "ao": np.nanmean(tele["ao"][:, iw], 1),
        "gbi": gbi_t[:, iw].mean(1),
    }
    if not (np.all((js > 15) & (js < 90)) and np.all((sc["k24"] > 0) & (sc["k24"] < 600))):
        raise SystemExit(f"values out of range: jet {js.min():.0f}-{js.max():.0f} m/s, k24 {sc['k24'].max():.0f}")
    if np.nanmax(np.abs(sc["nao"])) > 6:
        raise SystemExit(f"NAO out of range: {np.nanmax(np.abs(sc['nao'])):.1f}")
    return {
        **{f"s_{k}": v.astype("float32") for k, v in sc.items()},
        "ser_k24": box_mean(k24map, BOX, SECTOR).astype("float32"),
        "ser_nao": tele["nao"].astype("float32"),
        "ser_gbi": gbi_t.astype("float32"),
        "prof_u": us.astype("float32"),
        "prof_uv": uvs.astype("float32"),
        "prof_f": f.astype("float32"),
        "map_u": U.mean(0).astype("float32"),
        "map_k24": k24map.mean((0, 1)).astype("float32"),
        "map_uv": uvmap.mean(0).astype("float32"),
        "map_z500": za[:, iw].mean((0, 1)).astype("float32"),
        "map_msl": ma[:, iw].mean((0, 1)).astype("float32"),
    }


def member(a) -> int:
    start = pd.Timestamp(a.start)
    times = window(start, a.days)
    if a.model == "analysis":
        F, mems, init, tag = *analysis(times), None, "analysis"
    else:
        init = pd.Timestamp(f"{a.init[:8]}T{a.init[8:10]}")
        F, mems = gefs(init, times) if a.model == "gefs" else ecmwf(a.model, init, times)
        tag = f"{a.model}_{init:%Y%m%d%H}"
    d = diagnose(F, times, start)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / f"{tag}.npz", model=a.model, init="" if init is None else f"{init:%Y-%m-%dT%H}", members=np.array(mems),
        times=np.array([f"{t:%Y-%m-%dT%H}" for t in times]), start=f"{start:%Y-%m-%d}", lat=PC.CLAT, lon=PC.CLON, **d,
    )  # fmt: skip
    print(f"{tag}: {len(mems)} members; " + ", ".join(f"{k} {float(np.nanmean(d['s_' + k])):.2f}" for k in MEASURES))
    return 0


# --------------------------------------------------------------------------------------------------------- combine ---
def rnd(v, k=3):
    return None if v is None or not np.isfinite(v) else float(round(float(v), k))


def chain_r(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[float, float]:
    """r of x and y (member departures from their run's mean, pooled), permutation p with y shuffled within runs."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, groups = x[ok], y[ok], groups[ok]
    r0 = float(np.corrcoef(x, y)[0, 1])
    ix = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    hits = 0
    for _ in range(PERM):
        yp = y.copy()
        for i in ix:
            yp[i] = y[RNG.permutation(i)]
        hits += abs(np.corrcoef(x, yp)[0, 1]) >= abs(r0)
    return r0, (1 + hits) / (PERM + 1)


def phases(rs: list, start: pd.Timestamp) -> tuple[list, list]:
    """A model's runs made a week or more before the week, and its latest two."""
    return [r for r in rs if (start - pd.Timestamp(str(r["init"]))) >= pd.Timedelta(days=7)], rs[-2:]


def sector(x: np.ndarray) -> np.ndarray:
    """Mean over the sector's longitudes of x (..., lat, lon)."""
    lon = PC.CLON
    return x[..., (lon >= SECTOR[0]) & (lon < SECTOR[1])].mean(-1)


def where_tests(runs: dict, start: pd.Timestamp, stats) -> dict:
    """Each model's early and latest sector-mean profiles; the wind and -u'v' change tested at each latitude."""
    lat, cw = PC.CLAT, np.cos(np.deg2rad(PC.CLAT))
    prof = {}
    for m, rs in runs.items():
        early, late = phases(rs, start)
        if not early or len(rs) == len(early):
            continue
        d = {"lat": [rnd(x, 2) for x in lat]}
        for k, key, sign in (("u", "prof_u", 1.0), ("cwb", "prof_uv", -1.0)):
            x = sign * np.concatenate([r[key] for r in early]).astype(float)
            y = sign * np.concatenate([r[key] for r in late]).astype(float)
            d[f"{k}_early"], d[f"{k}_late"] = [rnd(v) for v in x.mean(0)], [rnd(v) for v in y.mean(0)]
            d[f"{k}_p"] = [rnd(v, 4) for v in stats.ttest_ind(y, x, axis=0, equal_var=False).pvalue]
        ke, kl = (sector(np.mean([r["map_k24"] for r in g], 0)) for g in (early, late))
        d["k24_early"], d["k24_late"] = [rnd(v) for v in ke], [rnd(v) for v in kl]
        d["k24_bands"] = []
        for lo, hi in K24_BANDS:
            b = (lat >= lo) & (lat <= hi)
            d["k24_bands"].append({"band": [lo, hi], "early": rnd(np.average(ke[b], weights=cw[b])),
                                   "late": rnd(np.average(kl[b], weights=cw[b]))})  # fmt: skip
        prof[m] = d
    return prof


def combine(a) -> int:
    from scipy import stats

    import cmip6_coupling as CC

    start = pd.Timestamp(a.start)
    runs = {}
    for p in sorted(Path(a.runs).glob("*.npz")):
        z = np.load(p)
        runs.setdefault(str(z["model"]), []).append({k: z[k] for k in z.files})
    an = runs.pop("analysis", [None])[0]
    doc = {"start": a.start, "days": a.days, "models": {}, "change": [], "chain": [], "sources": SOURCES}
    for m, rs in runs.items():
        rs.sort(key=lambda r: str(r["init"]))
        doc["models"][m] = []
        for r in rs:
            init = pd.Timestamp(str(r["init"]))
            row = {"init": str(r["init"]), "lead_days": rnd((start - init) / pd.Timedelta(days=1), 2),
                   "members": len(r["members"])}  # fmt: skip
            for k in MEASURES:
                x = r[f"s_{k}"].astype(float)
                row[k] = {"mean": rnd(np.nanmean(x)), "p10": rnd(np.nanpercentile(x, 10)),
                          "p90": rnd(np.nanpercentile(x, 90))}  # fmt: skip
                if an is not None:
                    row[k]["analysis_percentile"] = rnd(100 * np.mean(x < float(an[f"s_{k}"][0])), 1)
            doc["models"][m].append(row)
        early, late = phases(rs, start)
        if early and len(rs) > len(early):
            for k in MEASURES:
                x = np.concatenate([r[f"s_{k}"] for r in early]).astype(float)
                y = np.concatenate([r[f"s_{k}"] for r in late]).astype(float)
                t = stats.ttest_ind(y, x, equal_var=False, nan_policy="omit")
                doc["change"].append({"model": m, "measure": k, "early": [str(r["init"]) for r in early],
                                      "late": [str(r["init"]) for r in late], "early_mean": rnd(np.nanmean(x)),
                                      "late_mean": rnd(np.nanmean(y)), "p": rnd(t.pvalue, 4)})  # fmt: skip
        dev = {k: np.concatenate([r[f"s_{k}"] - np.nanmean(r[f"s_{k}"]) for r in rs]).astype(float) for k in MEASURES}
        groups = np.concatenate([np.full(len(r["members"]), i) for i, r in enumerate(rs)])
        for xk, yk in CHAIN:
            r0, p = chain_r(dev[xk], dev[yk], groups)
            doc["chain"].append({"model": m, "x": xk, "y": yk, "members": len(groups), "r": rnd(r0), "p": rnd(p, 4)})
    for key in ("change", "chain"):
        for t, s in zip(doc[key], CC.fdr(np.array([t["p"] for t in doc[key]], float))):
            t["fdr10"] = bool(s)
    doc["where"] = where_tests(runs, start, stats)
    keys = [(m, k) for m in doc["where"] for k in ("u", "cwb")]
    if keys:
        sig = CC.fdr(np.concatenate([np.array(doc["where"][m][f"{k}_p"], float) for m, k in keys]))
        for (m, k), s in zip(keys, np.split(sig, len(keys))):
            doc["where"][m][f"{k}_fdr10"] = [bool(x) for x in s]
    if an is not None:
        doc["analysis"] = {k: rnd(float(an[f"s_{k}"][0])) for k in MEASURES}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eddy_case.json").write_text(json.dumps(doc, indent=1))
    figures(runs, an, doc, start, out)
    if doc["where"]:
        where(runs, doc, start, out)
    for t in doc["change"]:
        print(f"change {t['model']:5s} {t['measure']:9s} {t['early_mean']} -> {t['late_mean']} p {t['p']}"
              f"{' FDR' if t['fdr10'] else ''}")  # fmt: skip
    for t in doc["chain"]:
        print(f"chain  {t['model']:5s} {t['x']:8s} vs {t['y']:8s} r {t['r']} p {t['p']}{' FDR' if t['fdr10'] else ''}")
    return 0


def figures(runs: dict, an, doc: dict, start: pd.Timestamp, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import house

    house.apply()
    colours = {"ifs": house.ACCENT, "aifs": house.WARM, "gefs": house.COOL}
    end = start + pd.Timedelta(days=doc["days"] - 1)
    week = f"{start:%-d}–{end:%-d %b %Y}"
    panels = (
        ("k24", "Synoptic eddy kinetic energy, 250 hPa, 30–70°N 120°W–0 (m² s⁻²)"),
        ("cwb", "Cyclonic tilt poleward of the jet, −u′v′ (m² s⁻²)"),
        ("jet_lat", "Jet latitude, 120°W–0 (°N)"),
        ("nao", "NAO (seasonal sd)"),
    )
    fig, axs = plt.subplots(len(panels), 1, figsize=house.size(9.0, 1.15), sharex=True)
    models = list(runs)
    for ax, (k, title) in zip(axs, panels):
        for mi, m in enumerate(models):
            for r in runs[m]:
                x = pd.Timestamp(str(r["init"])) + pd.Timedelta(hours=5 * (mi - 1))
                y = r[f"s_{k}"].astype(float)
                ax.scatter(np.full(len(y), x), y, s=5, color=colours.get(m, house.INK), alpha=0.35, lw=0)
            means = [(pd.Timestamp(str(r["init"])), np.nanmean(r[f"s_{k}"])) for r in runs[m]]
            ax.plot(*zip(*means), color=colours.get(m, house.INK), lw=1.6, label=NAMES.get(m, m))
        if an is not None:
            ax.axhline(float(an[f"s_{k}"][0]), color=house.INK, lw=1.0, ls="--", label="IFS analysis")
        ax.set_title(title, loc="left", fontsize=10)
    axs[0].legend(loc="upper left", fontsize=8, ncol=4, frameon=False)
    axs[-1].set_xlabel("Run (00Z)")
    fig.suptitle(f"The week of {week} over North America and the Atlantic, run by run (every member)", x=0.01,
                 ha="left", fontsize=12)  # fmt: skip
    house.source(fig, SOURCES)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(out / "eddy_case_runs.png", dpi=130)
    plt.close(fig)

    pairs = (("k24", "cwb"), ("cwb", "nao"), ("jet_lat", "nao"))
    fig, axs = plt.subplots(1, len(pairs), figsize=house.size(10.0, 0.36))
    for ax, (xk, yk) in zip(axs, pairs):
        for m in models:
            x = np.concatenate([r[f"s_{xk}"] - np.nanmean(r[f"s_{xk}"]) for r in runs[m]])
            y = np.concatenate([r[f"s_{yk}"] - np.nanmean(r[f"s_{yk}"]) for r in runs[m]])
            rr = next(t for t in doc["chain"] if t["model"] == m and {t["x"], t["y"]} == {xk, yk})
            ax.scatter(x, y, s=6, color=colours.get(m, house.INK), alpha=0.45, lw=0,
                       label=f"{NAMES.get(m, m)} r {rr['r']}{'' if rr['fdr10'] else ' (ns)'}")  # fmt: skip
        ax.set_xlabel(f"{xk} (member − its run's mean)")
        ax.set_ylabel(f"{yk} (member − its run's mean)")
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle(f"The chain inside the ensembles, week of {week}: every member of every run", x=0.01, ha="left",
                 fontsize=12)  # fmt: skip
    house.source(fig, SOURCES)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(out / "eddy_case_chain.png", dpi=130)
    plt.close(fig)


def where(runs: dict, doc: dict, start: pd.Timestamp, out: Path) -> None:
    """Profiles by latitude (early runs dashed, the latest two solid; dots where the change passes BH) over maps of the
    change, the ensembles averaged with equal weight."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt

    import house

    colours = {"ifs": house.ACCENT, "aifs": house.WARM, "gefs": house.COOL}
    lat, W = PC.CLAT, doc["where"]
    models = list(W)
    fig = plt.figure(figsize=house.size(10.0, 0.74))
    gs = fig.add_gridspec(2, 6, height_ratios=(1.0, 0.62), hspace=0.3, wspace=0.9)
    panels = (
        ("u", "Wind, 250 hPa (m s⁻¹)"),
        ("k24", "Synoptic eddy kinetic energy (m² s⁻²),\nensemble means"),
        ("cwb", "Cyclonic tilt −u′v′ (m² s⁻²)"),
    )
    for i, (k, title) in enumerate(panels):
        ax = fig.add_subplot(gs[0, 2 * i : 2 * i + 2])
        for m in models:
            c = colours.get(m, house.INK)
            late = np.array(W[m][f"{k}_late"], float)
            ax.plot(W[m][f"{k}_early"], lat, color=c, lw=1.2, ls="--")
            ax.plot(late, lat, color=c, lw=1.6, label=NAMES.get(m, m) if k == "k24" else None)
            if f"{k}_fdr10" in W[m]:
                s = np.array(W[m][f"{k}_fdr10"])
                ax.scatter(late[s], lat[s], s=9, color=c, lw=0, zorder=3)
        if k == "cwb":
            ax.axvline(0, color=house.EDGE, lw=0.8)
        ax.set_ylim(lat[0], 80)
        ax.set_ylabel("°N")
        ax.set_title(title, loc="left", fontsize=10)
    key = fig.axes[1]  # the eddy energy panel's lower right stays clear
    key.plot([], [], color=house.INK2, ls="--", lw=1.2, label="runs a week or more ahead")
    key.plot([], [], color=house.INK2, lw=1.6, label="latest two runs")
    key.legend(fontsize=7.5, frameon=False, loc="lower right")

    def mean_map(key: str, which: int, sign: float = 1.0) -> np.ndarray:
        return sign * np.mean([np.mean([r[key] for r in phases(runs[m], start)[which]], 0) for m in models], 0)

    lon = np.where(PC.CLON >= 180, PC.CLON - 360, PC.CLON)
    o = np.argsort(lon)
    lon = lon[o]
    pc = ccrs.PlateCarree()
    maps = (
        ("map_k24", 1.0, np.arange(-90, 91, 15), "change in synoptic eddy kinetic energy (m² s⁻²)",
         "map_u", (30.0, 45.0), "250 hPa wind 30 and 45 m s⁻¹: latest runs solid, early dashed"),
        ("map_uv", -1.0, np.arange(-60, 61, 10), "change in −u′v′ (m² s⁻²): red = more NW–SE (cyclonic) tilt",
         "map_z500", (-150.0, -90.0, -30.0, 30.0, 90.0, 150.0), "latest runs' 500 hPa height anomaly (m)"),
    )  # fmt: skip
    for i, (key, sign, levels, label, ckey, clev, ctitle) in enumerate(maps):
        ax = fig.add_subplot(gs[1, 3 * i : 3 * i + 3], projection=ccrs.PlateCarree(central_longitude=-60))
        ax.set_extent(VIEW, crs=pc)
        d = (mean_map(key, 1, sign) - mean_map(key, 0, sign))[:, o]
        cf = ax.contourf(lon, lat, d, levels=levels, cmap="RdBu_r", extend="both", transform=pc)
        for which, ls in ((1, "-"), (0, "--")) if ckey == "map_u" else ((1, None),):
            cs = ax.contour(lon, lat, mean_map(ckey, which)[:, o], levels=clev, colors=house.INK, linewidths=0.8,
                            linestyles=ls, transform=pc)  # fmt: skip
            if ckey == "map_z500":
                ax.clabel(cs, fmt="%d", fontsize=6.5, inline_spacing=2)
        ax.add_feature(cfeature.COASTLINE.with_scale("110m"), edgecolor=house.INK2, linewidth=0.5)
        ax.set_title(ctitle, loc="left", fontsize=9)
        cb = fig.colorbar(cf, ax=ax, orientation="horizontal", fraction=0.06, pad=0.05, ticks=levels[::2])
        cb.set_label(label, fontsize=8.5)
        cb.ax.tick_params(labelsize=7.5)
    inits = [sorted(str(r["init"])[:10] for m in models for r in phases(runs[m], start)[w]) for w in (0, 1)]
    day = lambda x: f"{pd.Timestamp(x):%-d %b}"  # noqa: E731
    end = start + pd.Timedelta(days=doc["days"] - 1)
    fig.suptitle(f"Where the change sits, {start:%-d}–{end:%-d %b %Y}, 120°W–0: runs of {day(inits[0][0])}–"
                 f"{day(inits[0][-1])} dashed, of {day(inits[1][0])}–{day(inits[1][-1])} solid (dots: the change "
                 f"passes BH 10 %)\nMaps: the later runs minus the earlier, the ensembles averaged",
                 x=0.01, ha="left", fontsize=11)  # fmt: skip
    house.source(fig, SOURCES)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.1)
    fig.savefig(out / "eddy_case_where.png", dpi=130)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("member")
    m.add_argument("--model", required=True, choices=(*ECMWF, "gefs", "analysis"))
    m.add_argument("--init", default="", help="YYYYMMDDHH (not for the analysis)")
    c = sub.add_parser("combine")
    c.add_argument("--runs", required=True)
    for p in (m, c):
        p.add_argument("--start", required=True, help="the week's first day, YYYY-MM-DD")
        p.add_argument("--days", type=int, default=7)
        p.add_argument("--out", required=True)
    a = ap.parse_args()
    return member(a) if a.cmd == "member" else combine(a)


if __name__ == "__main__":
    raise SystemExit(main())
