#!/usr/bin/env python3
"""How east-based is it RIGHT NOW? Daily OISST projected on the forecast page's E and C modes.

2026-09-27, user: "Quite a discontinuity between the models and observations for how east-based
this event is." The monthly ERSST track behind the E-C chart ends at ERSST's newest (provisional)
month, while the forecast starts in the issue month, so the chart could not show that the models
are already too central in their FIRST month. This step closes the gap every day, in Actions, with
no Copernicus key:

  1. NOAA OISST v2.1 daily anomalies (1991-2020 base, oisst9120) over the Takahashi box, averaged
     into the 2-degree ERSST cells the EOFs live on, and projected on the SAME patterns
     (scripts/sst/reference/enso_flavour_eof.npz, written by c3s_enso_flavour.py on the laptop).
     The last few complete months plus the month so far, with the number of days and how many of
     them are still preliminary.
  2. The first-month check: each model's forecast for the newest observed month (with at least
     MIN_DAYS days) against that month so far, for E, C, the warming centre and the Nino boxes.
  3. A first-month-corrected view: each model's error in that month held constant at every
     lead, and the page's verdict rule re-run on it (c3s_enso_flavour.season_summary). It rests on
     the assumption that the offset persists; the page says so, and only the hindcast test in
     c3s_enso_flavour.py can promote it.

OISST reads the coastal warming stronger than ERSST (August 2026: E +3.7 against +3.2); the page
marks where the observed track changes dataset.

    python scripts/sst/enso_flavour_obs.py      -> assets/sst/data/enso_flavour_obs.json
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oisst9120 as O                                                      # noqa: E402
import c3s_enso_flavour as F                                               # noqa: E402
from enso_models import EXCLUDE_SYSTEMS, EXCLUDE_NOTE                      # noqa: E402

SITE_ROOT = Path(os.environ["SST_SITE_ROOT"]).resolve() if os.environ.get("SST_SITE_ROOT") else HERE.parents[1]
DATA = SITE_ROOT / "assets" / "sst" / "data"
OUT = DATA / "enso_flavour_obs.json"
FL_JSON = DATA / "enso_flavour.json"
DAILY_JSON = DATA / "enso_daily.json"
N_MONTHS = 4          # complete months before the current one
N_WEEKS = 6           # weekly means, the newest ending on the newest day
MIN_DAYS = 10         # a month needs this many days to stand in for the first-month check


def _month_keys(today: pd.Timestamp) -> list[str]:
    cur = today.to_period("M")
    return [str(cur - k) for k in range(N_MONTHS, -1, -1)]


def _daily_anoms(months: list[str]) -> xr.DataArray:
    """Daily anomalies over the box (plus a 1-degree margin), every available day in `months`."""
    lat_s, lon_s = slice(F.EOF_LAT[0] - 1, F.EOF_LAT[1] + 1), slice(F.EOF_LON[0] - 1, F.EOF_LON[1] + 1)
    years = sorted({int(m[:4]) for m in months})
    parts = []
    for y in years:
        p = O.DATA / f"sst.day.mean.{y}.nc"
        if not p.exists():
            try:
                p = O.ensure_mean(y)
            except Exception as e:                                        # noqa: BLE001
                print(f"  OISST {y} unavailable ({str(e)[:80]})", flush=True)
                continue
        with xr.open_dataset(p) as ds:
            da = ds["sst"].sel(lat=lat_s, lon=lon_s).load()
        keep = np.isin(pd.DatetimeIndex(da["time"].values).strftime("%Y-%m"), months)
        if keep.any():
            parts.append(da.isel(time=np.where(keep)[0]))
    if not parts:
        raise SystemExit("no OISST days for the requested months")
    mean = xr.concat(parts, "time")
    clim = O.ltm().sel(lat=lat_s, lon=lon_s)
    idx = O.clim_indices(mean["time"].values)
    c = clim.isel(time=xr.DataArray(idx, dims="time")).load()
    return mean - c.values


def _to_ersst_cells(field: np.ndarray, lat: np.ndarray, lon: np.ndarray, glat, glon) -> np.ndarray:
    """(..., lat, lon) at 1/4 degree -> (..., glat, glon): cos-weighted mean over each 2x2-degree ERSST cell."""
    out = np.full(field.shape[:-2] + (len(glat), len(glon)), np.nan)
    wl = np.cos(np.deg2rad(lat))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i, la in enumerate(glat):
            mi = (lat >= la - 1) & (lat < la + 1)
            sub = field[..., mi, :]
            w = wl[mi][:, None]
            for j, lo in enumerate(glon):
                mj = (lon >= lo - 1) & (lon < lo + 1)
                x = sub[..., mj]
                ww = np.broadcast_to(w, x.shape[-2:])
                ok = np.isfinite(x)
                num = np.nansum(x * ww, axis=(-2, -1))
                den = np.sum(ok * ww, axis=(-2, -1))
                out[..., i, j] = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    return out


def _fill_row(x2: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """An EOF cell OISST has no ocean in (a coastline): nearest ocean cell of the same row."""
    probe = np.isfinite(x2).all(axis=tuple(range(x2.ndim - 2))) if x2.ndim > 2 else np.isfinite(x2)
    for i, j in zip(*np.where(valid & ~probe)):
        ok = np.where(probe[i])[0]
        if ok.size:
            x2[..., i, j] = x2[..., i, ok[np.argmin(np.abs(ok - j))]]
    return x2


def _window(an, sel, lat, lon, grid, eof, final_through) -> dict:
    """Mean anomaly over the days `sel`, projected: E, C, centre, boxes, profile, dates."""
    t = pd.DatetimeIndex(an["time"].values)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)                # land cells: all-NaN by design
        f = np.nanmean(an.values[sel], axis=0)
    f2 = _fill_row(_to_ersst_cells(f, lat, lon, grid["lat"], grid["lon"]), eof["valid"])
    E, C = F.project(f2[None], eof)
    bx = F.box_means(f[None], lat, lon)                                  # the boxes at full resolution
    centre = F.warming_centre(f2[None], grid["lat"], grid["lon"])[0]
    days = t[sel]
    prelim = int((days > pd.Timestamp(final_through)).sum()) if final_through else None
    return dict(days=int(len(sel)), first=str(days[0].date()), last=str(days[-1].date()), preliminary_days=prelim,
                E=round(float(E[0]), 2), C=round(float(C[0]), 2),
                lon=None if not np.isfinite(centre) else round(float(centre), 1),
                **{k: round(float(v[0]), 2) for k, v in bx.items()},
                prof=[None if not np.isfinite(v) else round(float(v), 2)
                      for v in F.eq_profile(f2, grid["lat"], grid["lon"], grid["lon"])])


def observed(today: pd.Timestamp) -> tuple[list[dict], list[dict]]:
    """(months, weeks): the last N_MONTHS complete months and the month so far, and N_WEEKS 7-day
    means ending on the newest day (days after NCEI's final_through are preliminary)."""
    grid, eof = F.load_eof_reference()
    months = _month_keys(today)
    an = _daily_anoms(months)
    lat, lon = an["lat"].values.astype(float), an["lon"].values.astype(float)
    t = pd.DatetimeIndex(an["time"].values)
    final_through = None
    if DAILY_JSON.exists():
        final_through = json.loads(DAILY_JSON.read_text()).get("final_through")
    out = []
    for m in months:
        sel = np.where(t.strftime("%Y-%m") == m)[0]
        if not sel.size:
            continue
        w = _window(an, sel, lat, lon, grid, eof, final_through)
        ndays = pd.Period(m).days_in_month
        out.append(dict(month=m, days_in_month=int(ndays), complete=bool(w["days"] >= ndays), **w))
        print(f"  OISST {m} ({w['days']}/{ndays} d): E {w['E']:+.2f} C {w['C']:+.2f} centre {F._lonlab(w['lon'])} "
              f"N1+2 {w['n12']:+.2f} N4 {w['n4']:+.2f}", flush=True)
    weeks = []
    end = t.max()
    for k in range(N_WEEKS - 1, -1, -1):
        hi = end - pd.Timedelta(days=7 * k)
        lo = hi - pd.Timedelta(days=6)
        sel = np.where((t >= lo) & (t <= hi))[0]
        if len(sel) < 5:
            continue
        w = _window(an, sel, lat, lon, grid, eof, final_through)
        w["mid"] = str((lo + pd.Timedelta(days=3)).date())
        w["preliminary"] = bool(w["preliminary_days"])
        weeks.append(w)
        print(f"  OISST week {w['first']}..{w['last']}: E {w['E']:+.2f} C {w['C']:+.2f} centre {F._lonlab(w['lon'])}"
              + (" (preliminary)" if w["preliminary"] else ""), flush=True)
    return out, weeks


def first_month(fl: dict, obs: list[dict]) -> dict | None:
    """The newest observed month that is also a forecast month and has MIN_DAYS days."""
    vm = fl["valid_months"]
    cands = [o for o in obs if o["month"] in vm and o["days"] >= MIN_DAYS]
    if not cands:
        return None
    o = cands[-1]
    L = vm.index(o["month"])
    rows = {}
    for k, m in fl["models"].items():
        lm = m.get("lead_means")
        if not lm:
            Em, Cm = round(float(np.mean(m["lead"]["E"][L])), 2), round(float(np.mean(m["lead"]["C"][L])), 2)
            lm = {"E": [None] * len(vm), "C": [None] * len(vm)}
            lm["E"][L], lm["C"][L] = Em, Cm
        row = {q: (lm[q][L] if q in lm else None) for q in ("E", "C", "lon", "n12", "n34", "n4")}
        row["dE"] = round(o["E"] - row["E"], 2)
        row["dC"] = round(o["C"] - row["C"], 2)
        rows[k] = row
    return dict(month=o["month"], lead=L + 1, days=o["days"], complete=o["complete"],
                preliminary_days=o["preliminary_days"], obs={q: o[q] for q in ("E", "C", "lon", "n12", "n34", "n4")},
                models=rows)


def weights(fl: dict) -> dict | None:
    """Pooled first-month persistence from the laptop's hindcast test for this start month, if run."""
    ref = F.hindcast_ref_path(int(fl["issue"][5:7]))
    if not ref.exists():
        return None
    hc = json.loads(ref.read_text())
    return {"season": hc.get("damped"), "lead": hc.get("damped_lead"), "years": [hc["years"][0], hc["years"][-1]]}


def corrected(fl: dict, chk: dict) -> tuple[dict, dict]:
    """The verdict rule on members shifted by each model's first-month error (E and C only).

    With the hindcast test in, the error is weighted by its pooled leave-one-year-out persistence
    (per season, and per lead for the monthly strips); without it, it is held at full strength and
    the page says the assumption is untested."""
    w = weights(fl)
    res, offs = {}, {"season": {}, "lead": {}, "method": "hindcast-weighted" if w else "full, untested", "weights": w}
    for k, m in fl["models"].items():
        d = chk["models"][k]
        seas = {}
        offs["season"][k], offs["lead"][k] = {}, []
        for s, v in m["season"].items():
            bE = w["season"][s]["beta_E"] if w else 1.0
            bC = w["season"][s]["beta_C"] if w else 1.0
            oE, oC = round(bE * d["dE"], 3), round(bC * d["dC"], 3)
            offs["season"][k][s] = {"E": oE, "C": oC}
            seas[s] = {"E": np.asarray(v["E"], float) + oE, "C": np.asarray(v["C"], float) + oC,
                       "lon": np.asarray(v["lon"], float)}
            for q in ("n12", "n3", "n34", "n4"):                     # the box means are not corrected
                seas[s][q] = np.full(len(v["E"]), np.nan)
        for L in range(len(fl["valid_months"])):
            if L + 1 < chk["lead"]:
                offs["lead"][k].append({"E": 0.0, "C": 0.0})
                continue
            lw = w["lead"].get(str(L + 2 - chk["lead"])) if w else None      # lead relative to the checked month
            bE = lw["beta_E"] if lw else 1.0
            bC = lw["beta_C"] if lw else 1.0
            offs["lead"][k].append({"E": round(bE * d["dE"], 3), "C": round(bC * d["dC"], 3)})
        res[k] = ({"season": seas}, m["color"])
    out = {}
    for s in fl["seasons"]:
        ev = pd.DataFrame([dict(label=e["label"], kind=e["kind"], E=e[s]["E"], C=e[s]["C"])
                           for e in fl["events"] if e.get(s) and e[s].get("E") is not None])
        S = F.season_summary(res, s, ev)
        if S:
            for q in ("n12", "n3", "n34", "n4"):
                S["mmm"].pop(q, None)
                for mm in S["model_means"].values():
                    mm.pop(q, None)
        out[s] = S
    return out, offs


def keep_models(fl: dict) -> dict:
    """fl with the excluded systems removed (the laptop JSON may predate the exclusion)."""
    fl = dict(fl)
    fl["models"] = {k: v for k, v in fl["models"].items() if k not in EXCLUDE_SYSTEMS}
    return fl


def _events(fl: dict, s: str) -> pd.DataFrame:
    return pd.DataFrame([dict(label=e["label"], kind=e["kind"], E=e[s]["E"], C=e[s]["C"])
                         for e in fl["events"] if e.get(s) and e[s].get("E") is not None])


def _res(fl: dict, shift=None, drop=None) -> dict:
    """season members per model as season_summary wants them; `shift` {model: {season: {E, C}}}."""
    res = {}
    for k, m in fl["models"].items():
        if k == drop:
            continue
        seas = {}
        for s, v in m["season"].items():
            o = (shift or {}).get(k, {}).get(s, {"E": 0.0, "C": 0.0})
            seas[s] = {"E": np.asarray(v["E"], float) + o["E"], "C": np.asarray(v["C"], float) + o["C"],
                       "lon": np.asarray(v["lon"], float)}
            for q in ("n12", "n3", "n34", "n4"):
                seas[s][q] = np.full(len(v["E"]), np.nan)
        res[k] = ({"season": seas}, m["color"])
    return res


def summaries(fl: dict, shift=None, drop=None) -> dict:
    out = {}
    for s in fl["seasons"]:
        S = F.season_summary(_res(fl, shift, drop), s, _events(fl, s))
        if S:
            # the box means come from the laptop build (members carry none): model by model, and their average
            lap = (fl.get("summary") or {}).get(s) or {}
            for k, mm in S["model_means"].items():
                src = (lap.get("model_means") or {}).get(k, {})
                for q in ("n12", "n3", "n34", "n4"):
                    mm[q] = src.get(q)
            for q in ("n12", "n3", "n34", "n4"):
                vals = [mm[q] for mm in S["model_means"].values() if mm.get(q) is not None]
                S["mmm"][q] = round(float(np.mean(vals)), 2) if vals else None
        out[s] = S
    return out


def lomo(fl: dict, shift=None) -> dict:
    """The verdict with each model left out in turn, per season: class, verdict, east share, p."""
    out = {}
    for k in fl["models"]:
        ss = summaries(fl, shift, drop=k)
        out[k] = {s: (dict(verdict=S["verdict"], klass=S["klass"], east=S["frac"]["east"], p=S["ttest"]["p"])
                      if S else None) for s, S in ss.items()}
    return out


def main() -> int:
    today = pd.Timestamp.now("UTC").tz_localize(None).normalize()
    obs, weeks = observed(today)
    doc = {"generated": pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M UTC"),
           "source": "NOAA OISST v2.1 daily, anomalies vs 1991–2020 (oisst9120), averaged into the 2° ERSST cells "
                     "and projected on the same E/C patterns as the forecast page",
           "final_through": json.loads(DAILY_JSON.read_text()).get("final_through") if DAILY_JSON.exists() else None,
           "months": obs, "weeks": weeks,
           "excluded": sorted(EXCLUDE_SYSTEMS), "excluded_note": EXCLUDE_NOTE}
    if FL_JSON.exists():
        fl = keep_models(json.loads(FL_JSON.read_text()))
        doc["issue"] = fl["issue"]
        doc["raw"] = {"summary": F._clean(summaries(fl)), "leave_one_out": F._clean(lomo(fl))}
        chk = first_month(fl, obs)
        if chk:
            doc["first_month"] = chk
            summ, offs = corrected(fl, chk)
            doc["corrected"] = {"offsets": offs, "summary": F._clean(summaries(fl, offs["season"])),
                                "leave_one_out": F._clean(lomo(fl, offs["season"]))}
            ref = F.hindcast_ref_path(int(fl["issue"][5:7]))
            if ref.exists():                   # the hindcast verdict rides along daily; only the reference needs pushing
                doc["hindcast"] = F._clean(F.hindcast_block(json.loads(ref.read_text()), F.peak_season(F.FC_JSON)))
            s = doc["corrected"]["summary"].get("OND") or {}
            print(f"  first month {chk['month']} (lead {chk['lead']}, {chk['days']} d); corrected OND: "
                  f"{s.get('verdict')} {s.get('frac')} p {s.get('ttest', {}).get('p')}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(F._clean(doc), separators=(",", ":"), ensure_ascii=False, allow_nan=False))
    os.replace(tmp, OUT)
    print(f"saved {OUT.name} ({OUT.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
