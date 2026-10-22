#!/usr/bin/env python3
"""Sudden-warming outlook, days 1-15: one summary of every ensemble on the stratosphere page, for ssw.html.

    python ssw_summary.py      # reads the published JSON and the committed references -> assets/sst/data/ssw_summary.json

It downloads nothing: every number comes from what the other stratosphere products publish (assets/sst/data) or from
the committed references (reference/). Five questions, each answered per model and then across the ensembles:

  SSW      the share of members with u(60N, 10 hPa) easterly on some day 1-15 (vortex_runs.json, nh_vortex.py), raw
           member counts, not calibrated; the members whose 10-day mean 100 hPa heat flux crosses the SSW warning level
           (heatflux100 ssw_warning); and how often a MERRA-2 sudden warming began in the same 15 calendar days.
           Charlton and Polvani (2007) count November-March onsets only.
  WAVES    the wave-1 and wave-2 shares of the ensemble-mean 100 hPa [v'T'] (45-75N) over days 1-15 and the peak of
           the total against its ERA5 day-of-year sd. A description of the forcing, not a split or displacement
           forecast: in ERA5 the wave-2 share over the 10 days before an onset does not separate MERRA-2's splits from
           its displacements (SPLIT_TEST).
  DESCENT  the CMIP6 coupling test's rule (cmip6_coupling.py wk100_td / wk100_bu) on the polar-cap height of the
           dripping paint (forecast_drip.py, ensemble mean, sd of the day-of-year normal): the first day the 100 hPa
           cap reaches +1 sd, in the analysis tail or the forecast, is top-down when 10 hPa was at +1 sd 3-20 days
           before, bottom-up when not.
  REFLECT  a reflecting configuration is u(60N) weaker at 1 hPa than at 10 hPa (Perlwitz and Harnik 2003; GEFS, the
           one ensemble here with winds to 1 hPa, refractive_gefs.json) with waves coming back down: the zonal-mean
           100 hPa flux below zero, or the Canadian box of the regional reflection index (Messori et al. 2022) below
           -1 sd. Upward flux at least 1 sd above normal into a vortex without that configuration is absorption.
  SURFACE  the members with the 100 hPa cap at +1 sd or more on day 15 (forecast_drip), and the SSW imprint outlook
           (ssw_imprint.py): the chance the surface pattern that follows sudden warmings is positive over the 30 days
           after day 15, against its 50 % base rate, with the skill of the regime it is in (|cap| >= 1 sd or not).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import heatflux100 as HF
import reflection_index as RX

DATA = REPO / "assets" / "sst" / "data"
REF = HERE / "reference"
OUT = DATA / "ssw_summary.json"
NAMES = {"aifs": "AIFS-ENS", "ifs": "IFS-ENS", "hres": "IFS control", "gefs": "GEFS", "geps": "GEPS"}
ENSEMBLES = ("aifs", "ifs", "gefs", "geps")
HEATFLUX = {"aifs": "heatflux100.json", "ifs": "ifs_heatflux100.json", "gefs": "gefs_heatflux100.json",
            "geps": "geps_heatflux100.json"}  # fmt: skip
WINDOW = 15
ONE_SD = 1.0
LAG = (3, 20)  # days before the 100 hPa crossing that a 10 hPa crossing makes the event top-down (cmip6_coupling.py)
WAVE_SHARE = 0.6  # a wavenumber carrying this share of the wave-1 + wave-2 flux names the forcing
ACTIVE_Z = 1.0  # upward flux this many sd above normal on some day is an active wave event
# ERA5 100 hPa wave-2 share of the wave-1 + wave-2 flux over days -10..-1 before the MERRA-2 sudden warmings since
# 1990, splits against displacements (one-sided Mann-Whitney): why WAVES does not forecast the vortex's geometry
SPLIT_TEST = {
    "splits": 6,
    "displacements": 17,
    "split_median": 0.29,
    "displacement_median": 0.23,
    "auc": 0.55,
    "p": 0.38,
}


def _read(name: str):
    try:
        return json.loads((DATA / name).read_text())
    except (OSError, ValueError):
        return None


def _r(x, n=2):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def ssw_part(runs: list) -> dict:
    """{model: {init, members, share, u10_min_mean, u10_min_p10}} from each model's newest run in vortex_runs.json."""
    out = {}
    for run in reversed(runs):
        for m, d in run["models"].items():
            if m in out or m not in ENSEMBLES:
                continue
            u = d["u10"]
            out[m] = {
                "init": run["init"].replace("T", " ") + "Z",
                "members": int(d["members"]),
                "share": _r(d["ssw"]),
                "u10_min_mean": _r(min(u["mean"][1:]), 1),
                "u10_min_p10": _r(min(u["p10"][1:]), 1),
            }
    return out


def base_rate(init: pd.Timestamp) -> dict:
    """The share of MERRA-2 winters with a sudden warming beginning in the 15 calendar days after `init`."""
    ev = json.loads((REF / "strat_history_events.json").read_text())
    hits = 0
    for w in ev["winters"]:
        y = w["y"] if init.month >= 7 else w["y"] + 1  # the winter's July-June year holding init's calendar day
        day = min(init.day, 28) if (init.month, init.day) == (2, 29) and y % 4 else init.day
        d0 = pd.Timestamp(year=y, month=init.month, day=day)
        on = pd.DatetimeIndex(w["ssw"])
        hits += bool(((on > d0) & (on <= d0 + pd.Timedelta(days=WINDOW))).any())
    n = len(ev["winters"])
    return {"winters": n, "with_onset": hits, "share": _r(hits / n, 3), "period": ev["period"]}


def waves_part(hf: dict, clim) -> dict:
    """Per model: wave shares, the peak standardised total flux, the warning meter, the regional boxes."""
    out = {}
    reg = RX.model_series(clim)
    for m, f in HEATFLUX.items():
        d = _read(f)
        if not d:
            continue
        nh = d["hemispheres"]["nh"]
        k1, k2 = (np.clip(np.array(nh[k]["mean"][1:], float), 0, None).sum() for k in ("k1", "k2"))
        v = pd.DatetimeIndex([x[:10] for x in d["valid"]])[1:]
        tot = np.array(nh["tot"]["mean"][1:], float)
        # every model on one scale: the ensemble-mean total against the ERA5 day-of-year normal
        z = (tot - HF.clim_at(clim, "nh", "tot", "mean", v)) / HF.clim_at(clim, "nh", "tot", "sd", v)
        w = nh.get("ssw_warning") or {}
        r = reg.get(m)
        out[m] = {
            "init": pd.Timestamp(d["init"]).strftime("%Y-%m-%d %HZ"),
            "members": d["members"],
            "wave2_share": _r(k2 / (k1 + k2)) if k1 + k2 > 0 else None,
            "z_peak": _r(np.nanmax(z), 1),
            "z_peak_day": int(np.nanargmax(z)) + 1,
            "flux_min": _r(np.nanmin(tot), 1),
            "warning_now": _r(w.get("now"), 1),
            "warning_threshold": w.get("threshold"),
            "warning_members": _r(w.get("members_crossing")),
            "ri_event_share": _r(nh.get("regional", {}).get("event_share")),
            "z_can_min": _r(np.nanmin(r["z_can"][1:]), 1) if r else None,
            "z_sib_max": _r(np.nanmax(r["z_sib"][1:]), 1) if r else None,
        }
    hf.update(out)
    return out


def descent_part() -> dict:
    """Per model (ensemble-mean dripping paint): the first +1 sd crossing at 100 hPa and whether 10 hPa led it."""
    out = {}
    for m in NAMES:
        d = _read(f"{m}_drip.json")
        if not d:
            continue
        lev = d["levels"]
        i10, i100 = lev.index(10.0), lev.index(100.0)
        init = pd.Timestamp(d["init"].replace("Z", ""))
        tail_t = pd.DatetimeIndex(d["tail_days"])
        fc_t = pd.DatetimeIndex([init + pd.Timedelta(hours=h) for h in d["lead_h"]]).normalize()
        tail, fc = np.array(d["tail_sd"], float), np.array(d["forecast_mean_sd"], float)
        s100 = pd.concat([pd.Series(tail[:, i100], tail_t), pd.Series(fc[:, i100], fc_t)])
        s10 = pd.concat([pd.Series(tail[:, i10], tail_t), pd.Series(fc[:, i10], fc_t)])
        s100, s10 = (s[~s.index.duplicated(keep="last")] for s in (s100, s10))
        up = s100[(s100 >= ONE_SD) & (s100.shift(1) < ONE_SD)]
        row = {
            "init": d["init"],
            "members": d["members"],
            "cap100_day15": _r(fc[-1, i100], 1),
            "cap10_day15": _r(fc[-1, i10], 1),
            "frac100_day15": _r(d["frac100_ge1"][-1]) if d.get("frac100_ge1") else None,
            "cross100": None,
            "cross10": None,
            "kind": "none",
        }
        if len(up):
            t = up.index[-1]
            led = s10[(s10.index >= t - pd.Timedelta(days=LAG[1])) & (s10.index <= t - pd.Timedelta(days=LAG[0]))]
            row["cross100"] = f"{t:%Y-%m-%d}"
            row["kind"] = "top-down" if (led >= ONE_SD).any() else "bottom-up"
        at10 = s10[(s10 >= ONE_SD) & (s10.shift(1) < ONE_SD)]
        if len(at10):
            row["cross10"] = f"{at10.index[-1]:%Y-%m-%d}"
        out[m] = row
    return out


def reflect_part(waves: dict) -> dict:
    sh = _read("refractive_gefs.json") or {}
    days = sh.get("u60_1_minus_10_hpa") or {}
    us = _read("upperstrat.json") or {}
    try:
        u = us["models"]["gefs"]["hemis"]["nh"]["u60"]
        last = sorted(u["1"], key=lambda k: int(k[1:]))[-1]
        normal = u["1"][last]["clim_mean"] - u["10"][last]["clim_mean"]
    except (KeyError, IndexError, TypeError):
        normal = None
    shear_min = min(days.values()) if days else None
    zm_down = [m for m, w in waves.items() if w["flux_min"] is not None and w["flux_min"] < 0]
    can_down = [m for m, w in waves.items() if w["z_can_min"] is not None and w["z_can_min"] <= -ONE_SD]
    active = [m for m, w in waves.items() if w["z_peak"] is not None and w["z_peak"] >= ACTIVE_Z]
    ri = [w["ri_event_share"] for w in waves.values() if w["ri_event_share"] is not None]
    if shear_min is not None and shear_min < 0 and (zm_down or can_down):
        verdict = "reflecting"
    elif active:
        verdict = "absorbing"
    else:
        verdict = "quiet"
    return {
        "verdict": verdict,
        "shear_init": sh.get("init"),
        "shear_min": _r(shear_min, 1),
        "shear_max": _r(max(days.values()), 1) if days else None,
        "shear_normal": _r(normal, 1),
        "zonal_mean_downward": zm_down,
        "canada_downward": can_down,
        "active": active,
        "ri_event_share_mean": _r(np.mean(ri)) if ri else None,
    }


def surface_part(descent: dict) -> dict:
    imp = (_read("ssw_imprint.json") or {}).get("models", {})
    with xr.open_dataset(REF / "ssw_imprint.nc") as f:
        skill = json.loads(f.attrs["outlook_skill"])
        tests = json.loads(f.attrs["imprint_tests"])
    out = {"models": {}, "skill": skill, "era5_after_ssw": tests}
    for m, d in imp.items():
        o = d.get("outlook")
        if not o:
            continue
        out["models"][m] = {
            "init": d["init"],
            "p_positive": _r(o["p_positive"]),
            "base_rate": _r(o["base_rate"]),
            "window": o["window"],
            "cap100_day15": _r(o["z100_day15"], 1),
            "frac100_day15": descent.get(m, {}).get("frac100_day15"),
        }
    ps = [v["p_positive"] for v in out["models"].values()]
    caps = [v["cap100_day15"] for v in out["models"].values()]
    out["p_positive_mean"] = _r(np.mean(ps)) if ps else None
    out["cap100_day15_mean"] = _r(np.mean(caps), 1) if caps else None
    out["regime"] = "cap_1sd" if caps and abs(np.mean(caps)) >= ONE_SD else "all_days"
    return out


def references() -> dict:
    """The committed results the page's method section quotes."""
    c = json.loads((REF / "cmip6_coupling_tests.json").read_text())
    keep = {("ssw", o) for o in ("ssi_d1_30", "ssi_d31_60", "tna_d1_30", "teu_d31_60")}
    keep |= {(g, o) for g in ("wk100_bu", "wk100_td", "ssw_absorbing", "ssw_reflecting", "shear_neg")
             for o in ("ssi_d1_30", "ssi_d31_60")}  # fmt: skip
    cmip6 = {f"{t['group']}/{t['outcome']}": {k: t[k] for k in ("mean", "p", "robust")} for t in c["tests"]
             if (t["group"], t["outcome"]) in keep}  # fmt: skip
    rx = json.loads((REF / "reflection_index.json").read_text())
    return {
        "cmip6_models": len(c["model_years"]),
        "cmip6_model_years": int(sum(c["model_years"].values())),
        "cmip6": cmip6,
        "messori_era5": {
            "events": len(rx["events"]),
            "period": rx["period"],
            "epo_after": rx["epo_after"],
            "tna_after": rx["tna_after"],
        },
        "split_test": SPLIT_TEST,
    }


def main() -> int:
    vr = _read("vortex_runs.json")
    if not vr or not vr.get("runs"):
        raise SystemExit("no vortex_runs.json")
    clim = xr.open_dataset(HF.CLIM).load()
    ssw = ssw_part(vr["runs"])
    waves = waves_part({}, clim)
    descent = descent_part()
    reflect = reflect_part(waves)
    surface = surface_part(descent)
    inits = [v["init"] for part in (ssw, waves, descent) for v in part.values()]
    newest = max(pd.Timestamp(i.replace("Z", "")) for i in inits)
    shares = [v["share"] for v in ssw.values() if v["share"] is not None]
    w2 = [w["wave2_share"] for w in waves.values() if w["wave2_share"] is not None]
    w2m = float(np.mean(w2)) if w2 else None
    kinds = [v["kind"] for m, v in descent.items() if m in ENSEMBLES]
    doc = {
        "built": pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%MZ"),
        "newest_init": f"{newest:%Y-%m-%d %HZ}",
        "window": [f"{newest + pd.Timedelta(days=1):%Y-%m-%d}", f"{newest + pd.Timedelta(days=WINDOW):%Y-%m-%d}"],
        "ssw_season": bool(any(m in (11, 12, 1, 2, 3) for m in pd.date_range(newest, periods=WINDOW + 1).month)),
        "names": NAMES,
        "ssw": {
            "models": ssw,
            "share_mean": _r(np.mean(shares)) if shares else None,
            "base_rate": base_rate(newest),
            "warning_record": (_read("heatflux100.json") or {})
            .get("hemispheres", {})
            .get("nh", {})
            .get("ssw_warning", {})
            .get("record"),
        },
        "waves": {
            "models": waves,
            "wave2_share_mean": _r(w2m),
            "kind": None
            if w2m is None
            else "wave-2"
            if w2m >= WAVE_SHARE
            else "wave-1"
            if w2m <= 1 - WAVE_SHARE
            else "mixed",
        },
        "descent": {"models": descent, "kind": max(set(kinds), key=kinds.count) if kinds else None},
        "reflect": reflect,
        "surface": surface,
        "reference": references(),
        "thresholds": {"one_sd": ONE_SD, "lag_days": list(LAG), "wave_share": WAVE_SHARE, "active_z": ACTIVE_Z},
    }
    OUT.write_text(json.dumps(doc, indent=1, default=str))
    print(
        f"wrote {OUT.relative_to(REPO)}: SSW share {doc['ssw']['share_mean']}, {doc['waves']['kind']}, "
        f"{doc['descent']['kind']}, {reflect['verdict']}, P(imprint > 0) {surface['p_positive_mean']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
