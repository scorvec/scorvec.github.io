#!/usr/bin/env python3
"""Hadley and Walker forecasts, member by member, from the AIFS-ENS, IFS-ENS and GEFS 00Z runs.

Every member's zonal-mean v and equatorial divergent zonal wind (overturn.reduce_file, the same reductions as the
analysis) at days 0..15, corrected by its model's drift against ERA5 at that lead and season (overturn.drift, the
reference in release overturn-ref-v1), then put through what the analysed loops go through (mmsf.py, walker.py):
TRAILING 7-day means, so the window of an early lead still holds analysed days (less the analysis offset d(0), so
analysis and forecast both stand on ERA5's footing); Psi and Psi_W against the ERA5 1991-2020 harmonic climatology at
the window's midpoint; the cell intensities and edges (build_hadley_clim.metrics, on the climatology's 1.5 deg bands), the
Pacific Walker cell index and the Indo-Pacific ascent longitude (walker.pac_index, walker.ascent_lon).

The loops (assets/sst/anim/{mmsf,walker}_fc_<region>, one region per ensemble and one for all of them pooled) lead in
with the last LEAD_IN analysed weeks (overturn_hist.py's frames, on the same fixed colour scale), then show the members'
mean anomaly ONLY where at least AGREE of the members give it the same sign; elsewhere the ensemble does not say which
way the cell departs, and the frame leaves it blank. A model without a drift reference is left out rather than drawn
raw: its drift would read as an anomaly.

GEFS runs on to day 35: its weeks 3-5 (overturn_rf.WEEKS) are its members' weekly means less GEFS's drift at those weeks,
measured on its 2000-2019 reforecast against ERA5 (overturn_rf.py), appended to the GEFS loops and given to the series
(the "weeks" of its entry in the JSON) only for the weeks whose skill on the reforecast passed (overturn_rf.SKILL). NOAA
publishes a 00Z run's days 16-35 about a day later (the last members by ~03:30 UTC the next day), so the weeks come from
the previous day's run, named as such; a failure there leaves the weeks out and the 15-day loops as they are.

    python src/overturn_fc.py --date 20261016      # -> assets/sst/data/overturn_fc.json + the two loops
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyproj  # noqa: F401  (load pyproj's libproj before xarray/eccodes)
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ecmwf"))
import overturn as O
import overturn_hist as H
import overturn_rf as RFX
import mmsf
import walker
import build_hadley_clim as B
from doycal import noleap_doy
from mmsf import LEVELS

LEADS = list(range(16))  # days
WIN = 7  # trailing window, days (as the analysed frames)
AGREE = 0.8  # shade an anomaly only where this share of the members gives it one sign
LEAD_IN = 14  # analysed weeks the forecast loops open with
KEYS = ("nh_psi", "nh_edge_pole", "sh_psi", "sh_edge_pole")
P_PA = np.asarray(LEVELS, float) * 100.0
P_HPA = np.asarray(LEVELS, float)


def fetch(model: str, init: pd.Timestamp):
    """vbar (member, lead, lev, 721) and ud (member, lead, lev, 173) for every member, one step file at a time."""
    import store as ecmwf

    if model == "gefs":
        return O.gefs_run(f"{init:%Y%m%d}", list(range(O.GEFS_PF + 1)), [24 * L for L in LEADS], workers=16)
    cfg = O.MODELS[model]
    cyc = ecmwf.Cycle(f"{init:%Y%m%d}", "00")
    members = ([0] if cfg["cf"] else []) + list(range(1, 51))
    vb = np.full((len(members), len(LEADS), len(LEVELS), O.LAT.size), np.nan, np.float32)
    ud = np.full((len(members), len(LEADS), len(LEVELS), O.NLON_UD), np.nan, np.float32)
    for j, L in enumerate(LEADS):
        s = (24 * L,)
        for typ in (["cf"] if cfg["cf"] else []) + ["pf"]:
            p = ecmwf.ensure(cyc, ecmwf.Spec(cfg["store"], typ, ("u", "v"), "pl", tuple(LEVELS), s))
            mem = [0] if typ == "cf" else list(range(1, 51))
            v, u = O.reduce_file(p, mem, s)
            rows = [members.index(m) for m in mem]
            vb[rows, j], ud[rows, j] = v[:, 0], u[:, 0]
            p.unlink(missing_ok=True)  # ~0.9 GB a step: never keep them
        print(f"  {cfg['label']} day {L}: {np.isfinite(vb[:, j, 0, 0]).sum()} members", flush=True)
    return vb, ud


def analysed(init: pd.Timestamp, offset_v, offset_u):
    """The 00 UTC analyses of the WIN-1 days before init (the cycle histories), less the analysis offset: (WIN-1, lev,
    lat) and (WIN-1, lev, lon), or None when one is missing (no blended week can be formed). 00 UTC, as the forecast
    steps and the climatologies (build_hadley_clim.py)."""
    days = pd.date_range(init - pd.Timedelta(days=WIN - 1), init - pd.Timedelta(days=1))
    out = []
    for path, off in ((mmsf.HIST, offset_v), (walker.HIST, offset_u)):
        if not Path(path).exists():
            return None
        h = xr.open_dataarray(path)
        t = pd.DatetimeIndex(h.time.values)
        rows = []
        for d in days:
            m = t == d  # the day's 00 UTC cycle
            if not m.any():
                return None
            rows.append(h.values[m][0] - off)
        out.append(np.stack(rows))
    return out


class Sums:
    """Per-lead running sums over members (the loops need the members' mean and sign agreement, never the members)."""

    def __init__(self, nlon: int):
        self.n = 0
        shape_m, shape_w = (len(LEADS), len(LEVELS), B.LAT15.size), (len(LEADS), len(LEVELS), nlon)
        self.m, self.mpos, self.ma = np.zeros(shape_m), np.zeros(shape_m), np.zeros(shape_m)
        self.w, self.wpos, self.wa = np.zeros(shape_w), np.zeros(shape_w), np.zeros(shape_w)

    def add(self, other: Sums):
        for k in ("m", "mpos", "ma", "w", "wpos", "wa"):
            setattr(self, k, getattr(self, k) + getattr(other, k))
        self.n += other.n


def windows(vb, ud, hist_v, hist_u, init, cm, cw, lon):
    """Trailing-WIN-day means of every member's corrected fields -> metrics {key: (member, lead)}, the Sums, and the
    climatology at each window's midpoint."""
    n_m = vb.shape[0]
    res = {k: np.full((n_m, len(LEADS)), np.nan) for k in (*KEYS, "pac", "asc_lon")}
    S = Sums(lon.size)
    mids = [float(noleap_doy(pd.DatetimeIndex([init + pd.Timedelta(days=j - (WIN - 1) / 2)]))[0]) for j in LEADS]
    clim_m = [mmsf.clim_psi(cm, x) for x in mids]
    clim_w = [walker.clim_psi(cw, x) for x in mids]
    for i in range(n_m):
        if not np.isfinite(vb[i]).all():
            continue
        full_v, full_u = np.concatenate([hist_v, vb[i]]), np.concatenate([hist_u, ud[i]])
        S.n += 1
        for j in LEADS:
            wv, wu = full_v[j + 1 : j + 1 + WIN].mean(0), full_u[j + 1 : j + 1 + WIN].mean(0)
            psi = mmsf.streamfunction(B.band_v(wv, O.LAT), P_PA, B.LAT15)  # the climatology's bands
            met = B.sample_metrics(psi, P_HPA, B.LAT15)
            for k in KEYS:
                res[k][i, j] = met[k]
            pw = walker.streamfunction(wu, P_PA)
            res["pac"][i, j] = walker.pac_index(pw, P_HPA, lon)
            res["asc_lon"][i, j] = walker.ascent_lon(pw, P_HPA, lon)
            S.m[j] += psi - clim_m[j]
            S.mpos[j] += psi > clim_m[j]
            S.ma[j] += psi
            S.w[j] += pw - clim_w[j]
            S.wpos[j] += pw > clim_w[j]
            S.wa[j] += pw
    return res, S, mids, clim_m, clim_w


def agreed(mean, pos, n):
    """The members' mean anomaly where at least AGREE of them share its sign, NaN (blank) elsewhere."""
    share = pos / n
    return np.where((share >= AGREE) | (share <= 1 - AGREE), mean, np.nan)


def gefs_weeks(init: pd.Timestamp, cm, cw, lon) -> dict | None:
    """GEFS weeks 3-5 of the 00Z run the day before init (the last whose days 16-35 are all published): every member's
    weekly zonal-mean v (in the climatology's bands) and divergent u less GEFS's reforecast drift, reduced to what the
    loops and the series need (per week the sums of the anomaly, the members above the climatology and the full
    fields; every member's metrics); None without the reference or with no week that passed."""
    if not RFX.SKILL.exists():
        return None
    show = {k: v["show"] for k, v in json.loads(RFX.SKILL.read_text())["targets"].items()}
    run = init - pd.Timedelta(days=1)
    drift = RFX.drift_at(float(noleap_doy(pd.DatetimeIndex([run]))[0]))
    if drift is None or not any(show.values()):
        return None
    vb, ud = O.gefs_run(f"{run:%Y%m%d}", list(range(O.GEFS_PF + 1)), [24 * d for d in RFX.DAYS], workers=16)
    ok = np.isfinite(vb).all(axis=(1, 2, 3)) & np.isfinite(ud).all(axis=(1, 2, 3))
    vb, ud = vb[ok], ud[ok]
    nw = len(RFX.WEEKS)
    out = {
        "n": vb.shape[0],
        "show": show,
        "run": run,
        "end": [run + pd.Timedelta(days=z) for _, z in RFX.WEEKS],
        "metrics": {k: np.full((vb.shape[0], nw), np.nan) for k in (*KEYS, "pac", "asc_lon")},
        **{k: [] for k in ("m", "mpos", "ma", "w", "wpos", "wa", "clim_m", "clim_w")},
    }
    for w, (a, z) in enumerate(RFX.WEEKS):
        days = [d - RFX.DAYS[0] for d in range(a, z + 1)]
        vbw = B.band_v(vb[:, days].mean(1), O.LAT) - drift["vbar"][w][None]
        udw = ud[:, days].mean(1) - drift["ud"][w][None]
        mid = RFX.week_mid(run, w)
        cmw, cww = mmsf.clim_psi(cm, mid), walker.clim_psi(cw, mid)
        psi = np.stack([mmsf.streamfunction(x, P_PA, B.LAT15) for x in vbw])
        pw = np.stack([walker.streamfunction(x, P_PA) for x in udw])
        for i in range(psi.shape[0]):
            met = B.sample_metrics(psi[i], P_HPA, B.LAT15)
            for k in KEYS:
                out["metrics"][k][i, w] = met[k]
            out["metrics"]["pac"][i, w] = walker.pac_index(pw[i], P_HPA, lon)
            out["metrics"]["asc_lon"][i, w] = walker.ascent_lon(pw[i], P_HPA, lon)
        for key, x, c in (("m", psi, cmw), ("w", pw, cww)):
            out[key].append((x - c).sum(0))
            out[key + "pos"].append((x > c).sum(0))
            out[key + "a"].append(x.sum(0))
        out["clim_m"].append(cmw)
        out["clim_w"].append(cww)
    print(
        f"  GEFS weeks 3-5 of the {run:%Y-%m-%d} run: {out['n']} members; shown "
        + ", ".join(f"{k} {v}" for k, v in show.items()),
        flush=True,
    )
    return out


def render(regions: dict, init, mids, clim_m, clim_w, lon, anim_root: Path, manifests: dict, weeks: dict | None = None):
    """The two loops: per region, leads 0..15 of the agreed mean anomaly over the members' mean cells; the GEFS region
    then GEFS's weeks 3-5 that passed (gefs_weeks)."""
    lat = B.LAT15
    vm, vw = H.VLIM["mmsf"], H.VLIM["walker"]  # the analysed weeks' scale: lead-in and forecast read as one loop
    for prod in ("mmsf", "walker"):
        hist_dir = H.PRODUCTS[prod]["dir"]
        hm = anim_root / f"{hist_dir}_manifest.json"
        past = json.loads(hm.read_text())["regions"][hist_dir]["frames"] if hm.exists() else []
        days = {f["date"]: f for f in past}
        lead = [
            days[k] for k in (f"{init - pd.Timedelta(days=n):%Y-%m-%d}" for n in range(LEAD_IN, 0, -1)) if k in days
        ]
        doc = {
            "ver": f"{init:%Y%m%d}00",
            "init": f"{init:%Y-%m-%d}T00:00Z",
            "default": f"{prod}_fc_" + ("blend" if "blend" in regions else next(iter(regions))),
            "selectorLabel": "Forecast",
            "start": len(lead),  # opens on +0 d (sst_anim ?start=manifest)
            "regions": {},
        }
        for rid, S in regions.items():
            d = anim_root / f"{prod}_fc_{rid}"  # the viewer reads <anim>/<region id>/<file>
            d.mkdir(parents=True, exist_ok=True)
            for old in [*d.glob("F*.webp"), *d.glob("W*.webp")]:
                old.unlink()
            name = (
                " + ".join(O.MODELS[m]["label"] for m in regions if m != "blend")
                if rid == "blend"
                else O.MODELS[rid]["label"]
            )
            # the analysed weeks are the archive's own frames, one directory over (the viewer resolves the path)
            frames = [
                {"file": f"../{hist_dir}/{f['file']}", "date": f["date"], "label": f"analysis · {f['label']}"}
                for f in lead
            ]
            for j in LEADS:
                end = init + pd.Timedelta(days=j)
                lab = f"+{j} d · 7-day mean ending {end:%a %b %-d}"
                fp = d / f"F{j:02d}.webp"
                title = f"{name} {init:%Y-%m-%d} 00Z — 7-day mean ending {end:%Y-%m-%d} (+{j} d)"
                agree = f"shown where ≥{AGREE:.0%} of the {S.n} members agree on its sign · members less their model's drift against ERA5"
                if prod == "mmsf":
                    psa = S.ma[j] / S.n
                    mmsf.plot_psi(
                        agreed(S.m[j] / S.n, S.mpos[j], S.n),
                        P_HPA,
                        lat,
                        fp,
                        f"Ψ′ forecast, {title}",
                        anom=True,
                        vlim=vm,
                        psi_abs=psa,
                        # the cells against the climatology on its own 1.5 deg bands (build_hadley_clim)
                        texts=mmsf.strength_texts(mmsf.hadley_strength(psa, clim_m[j], P_HPA, lat, doy=mids[j])),
                        foot=f"colour = members' mean Ψ′ vs ERA5 1991–2020, {agree} · black = members' mean Ψ · arrows = vertical motion",
                    )
                else:
                    pwa = S.wa[j] / S.n
                    pidx = walker.pac_index(pwa, P_HPA, lon)
                    pcl = walker.pac_index(clim_w[j], P_HPA, lon)
                    walker.plot_psi(
                        agreed(S.w[j] / S.n, S.wpos[j], S.n),
                        P_HPA,
                        lon,
                        fp,
                        f"Ψ_W′ forecast, {title}",
                        vw,
                        pwa,
                        pidx,
                        pcl,
                        psi_clim=clim_w[j],
                        foot=f"colour = members' mean Ψ_W′ vs ERA5 1991–2020, {agree}\n"
                        f"black = members' mean cells · green = climatological cells · dashed = reversed · "
                        f"Pacific cell {pidx:+.1f} (clim {pcl:+.1f}) ×10¹⁰ kg/s — lower = weaker Walker",
                    )
                    lab += f" · Pacific {pidx:+.1f}"
                frames.append({"file": fp.name, "date": f"{end:%Y-%m-%d}", "label": lab})
            if rid == "gefs" and weeks:
                frames += week_frames(prod, weeks, name, d, lat, lon, vm if prod == "mmsf" else vw)
            frames = [dict(f, idx=i) for i, f in enumerate(frames)]
            doc["regions"][f"{prod}_fc_{rid}"] = {"label": name, "n_frames": len(frames), "frames": frames}
        manifests[prod].parent.mkdir(parents=True, exist_ok=True)
        manifests[prod].write_text(json.dumps(doc))
        print(f"  {prod}_fc: {len(regions)} regions x {len(LEADS)} frames", flush=True)


def week_frames(prod: str, wk: dict, name: str, d: Path, lat, lon, vlim) -> list:
    """GEFS's weeks 3-5 that passed for this loop (Psi for the Hadley loop, Psi_W for the Walker one), drawn as the
    leads are."""
    frames = []
    n = wk["n"]
    for w in range(wk["show"]["psi" if prod == "mmsf" else "psi_w"]):
        a, z = RFX.WEEKS[w]
        end = wk["end"][w]
        fp = d / f"W{w + 3}.webp"
        title = f"{name} {wk['run']:%Y-%m-%d} 00Z — week {w + 3}, days {a}–{z}, ending {end:%Y-%m-%d}"
        agree = (
            f"shown where ≥{AGREE:.0%} of the {n} members agree on its sign · members less GEFS's drift against ERA5 "
            "at that week, from its 2000–2019 reforecast"
        )
        if prod == "mmsf":
            psa = wk["ma"][w] / n
            mmsf.plot_psi(
                agreed(wk["m"][w] / n, wk["mpos"][w], n),
                P_HPA,
                lat,
                fp,
                f"Ψ′ forecast, {title}",
                anom=True,
                vlim=vlim,
                psi_abs=psa,
                texts=mmsf.strength_texts(
                    mmsf.hadley_strength(psa, wk["clim_m"][w], P_HPA, lat, doy=RFX.week_mid(wk["run"], w))
                ),
                foot=f"colour = members' mean Ψ′ vs ERA5 1991–2020, {agree} · black = members' mean Ψ · arrows = "
                "vertical motion",
            )
            lab = f"week {w + 3} of the {wk['run']:%b %-d} run · ending {end:%a %b %-d}"
        else:
            pwa = wk["wa"][w] / n
            pidx, pcl = walker.pac_index(pwa, P_HPA, lon), walker.pac_index(wk["clim_w"][w], P_HPA, lon)
            walker.plot_psi(
                agreed(wk["w"][w] / n, wk["wpos"][w], n),
                P_HPA,
                lon,
                fp,
                f"Ψ_W′ forecast, {title}",
                vlim,
                pwa,
                pidx,
                pcl,
                psi_clim=wk["clim_w"][w],
                foot=f"colour = members' mean Ψ_W′ vs ERA5 1991–2020, {agree}\n"
                f"black = members' mean cells · green = climatological cells · dashed = reversed · "
                f"Pacific cell {pidx:+.1f} (clim {pcl:+.1f}) ×10¹⁰ kg/s — lower = weaker Walker",
            )
            lab = f"week {w + 3} of the {wk['run']:%b %-d} run · ending {end:%a %b %-d} · Pacific {pidx:+.1f}"
        frames.append({"file": fp.name, "date": f"{end:%Y-%m-%d}", "label": lab})
    return frames


def _json(a):
    """Rounded nested lists with None for NaN (a cell or edge the metric could not resolve)."""
    return [_json(x) for x in a] if np.ndim(a) else (None if not np.isfinite(a) else round(float(a), 2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="00Z run, YYYYMMDD")
    ap.add_argument("--models", default="aifs,ifs,gefs")
    ap.add_argument("--out", default="assets/sst/data/overturn_fc.json")
    ap.add_argument("--anim-root", default="assets/sst/anim")
    a = ap.parse_args()
    init = pd.Timestamp(a.date)
    ref = O.load_ref()
    doy = float(noleap_doy(pd.DatetimeIndex([init]))[0])
    da_v, da_u = O.drift(ref, "aifs", doy, "vbar"), O.drift(ref, "aifs", doy, "ud")
    if da_v is None or da_u is None:
        print("no AIFS-ENS drift: the analysis offset is unknown, nothing is drawn", file=sys.stderr)
        return 1
    hist = analysed(init, da_v[0][0], da_u[0][0])  # the analysis is the AIFS control's step 0
    if hist is None:
        print("the analysis history has a hole in the last week", file=sys.stderr)
        return 1
    lon = xr.open_dataarray(walker.HIST).longitude.values
    cm = xr.open_dataarray(mmsf.CLIM)
    cw = xr.open_dataarray(walker.CLIM).interp(longitude=lon, kwargs={"fill_value": None})
    doc = {
        "init": f"{init:%Y-%m-%d}T00:00Z",
        "valid": [f"{init + pd.Timedelta(days=L):%Y-%m-%d}" for L in LEADS],
        "window_days": WIN,
        "footing": "ERA5: members less their model's drift, analysed days less the analysis offset",
        "models": {},
    }
    regions, mids, clim_m, clim_w = {}, None, None, None
    for model in a.models.split(","):
        dv, du = O.drift(ref, model, doy, "vbar"), O.drift(ref, model, doy, "ud")
        if dv is None or du is None:
            print(f"  {O.MODELS[model]['label']}: no drift reference, left out", flush=True)
            continue
        vb, ud = fetch(model, init)
        vb -= O.drift_at(dv[0], LEADS)[None]
        ud -= O.drift_at(du[0], LEADS)[None]
        res, S, mids, clim_m, clim_w = windows(vb, ud, *hist, init, cm, cw, lon)
        regions[model] = S
        doc["models"][model] = {
            "label": O.MODELS[model]["label"],
            "members": S.n,
            "drift_starts": dv[1],
            **{k: _json(v) for k, v in res.items()},
        }
        med = {k: np.nanmedian(v[:, -1]) for k, v in res.items()}
        print(f"  {O.MODELS[model]['label']} day 15 medians: " + " ".join(f"{k} {v:+.2f}" for k, v in med.items()))
    if not regions:
        return 1
    try:
        weeks = gefs_weeks(init, cm, cw, lon) if "gefs" in regions else None
    except Exception as e:  # noqa: BLE001  (the weeks are an addition: the 15-day loops go out without them)
        print(f"  GEFS weeks 3-5 left out: {type(e).__name__} {str(e)[:120]}", flush=True)
        weeks = None
    if weeks:
        doc["models"]["gefs"]["weeks"] = {
            "run": f"{weeks['run']:%Y-%m-%d}T00:00Z",
            "end": [f"{e:%Y-%m-%d}" for e in weeks["end"]],
            "days": [list(w) for w in RFX.WEEKS],
            **{k: _json(v[:, : weeks["show"][k]]) for k, v in weeks["metrics"].items()},  # only the weeks that passed
        }
    if len(regions) > 1:
        blend = Sums(lon.size)
        for S in regions.values():
            blend.add(S)
        regions = {"blend": blend, **regions}
    anim_root = Path(a.anim_root)
    manifests = {p: anim_root / f"{p}_fc_manifest.json" for p in ("mmsf", "walker")}
    render(regions, init, mids, clim_m, clim_w, lon, anim_root, manifests, weeks)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(doc, separators=(",", ":")))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
