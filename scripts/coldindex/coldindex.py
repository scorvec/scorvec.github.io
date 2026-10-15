#!/usr/bin/env python3
"""US cold-risk index for weeks 3-4, live (2026-10-15; user: "a nice flashy cold index page on my site and leave it unlinked").

The chance that the US population-weighted 2 m temperature over days 15-28 after the issue day lands in the coldest 15 %
of December-February two-week windows, from the observed state alone (no forecast model). Built and validated in
~/data_archive/coldspell (REPORT.md section 7; coldindex_export.py writes reference/coldindex_ref.json):
  clues      Atlantic block = mean z of the Greenland z500 block, the 1000 hPa polar cap and -NAO (5-day means);
             100 hPa polar-cap height (5-day mean); the last 7 days of US temperature; an SSW in the last 30 days;
             the MJO phase (site's ENSO-cleaned wind RMM, scripts/mjo; no MJO when amplitude < 1)
  weights    likelihood ratios learned in 146 CMIP6 historical runs (28 models), overlap-calibrated there
             (logit p = b0 + b1 (logit 0.15 + sum log LR)); MJO ratios from ERA5 1980-2024, shrunk halfway;
             only bins whose 95 % interval excludes 1 are applied
  inputs     GFS analyses (gfs_daily.py), reduced exactly as the ERA5 training data, minus the GFS-ERA5 offsets of
             winter 2025/26; anomalies with the training operator (3-harmonic climatology + 21-winter mean)
  season     issued Nov 16 - Jan 31 only (the whole target window inside Dec-Feb); other days carry clue readings only
Outputs (assets/coldindex): inputs.csv (rolling daily inputs; missing days are backfilled), current.json, meta.json,
replay.json (1980-2024 hindcast, copied from the reference).

    python scripts/coldindex/coldindex.py --out assets/coldindex
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import gfs_daily as GD                                                   # noqa: E402

REF = json.loads((HERE / "reference" / "coldindex_ref.json").read_text())
BACK = 60                                                                # days of inputs kept fresh (5/7-day means, 30-day SSW look-back)
RETRY = 10                                                               # incomplete days are re-fetched while this young
NAMES = {"block": "Atlantic block", "vortex": "Weak stratospheric vortex", "recent": "Recent US cold", "ssw": "Sudden stratospheric warming",
         "mjo": "MJO phase"}


def harmonics(d: pd.DatetimeIndex) -> np.ndarray:
    w = 2 * np.pi * (d.dayofyear.values - 1) / 365.25
    return np.column_stack([np.ones_like(w)] + [f(j * w) for j in range(1, 4) for f in (np.cos, np.sin)])


def winter_of(d: pd.DatetimeIndex) -> np.ndarray:
    m = d.month.values
    return np.where(m >= 11, d.year.values + 1, np.where(m <= 3, d.year.values, -1))


def anomaly(raw: pd.Series, key: str) -> pd.Series:
    """The training operator: GFS offset removed, then the ERA5 3-harmonic climatology, then the mean anomaly of the
    winters within 10 of this one (the 21-winter neighbourhood, truncated at the end of the record)."""
    s = REF["series"][key]; d = raw.index
    a = raw.values - s["gfs_offset"] - harmonics(d) @ np.array(s["beta"])
    wm = {int(k): v for k, v in s["winter_means"].items()}; ws = np.array(sorted(wm))
    w = winter_of(d); y = np.where(w > 0, w, np.clip(d.year.values, ws[0], ws[-1]))
    tr = np.array([np.mean([wm[z] for z in ws if abs(z - yy) <= 10]) for yy in y])
    return pd.Series(a - tr, index=d)


def trailing(x: pd.Series, k: int) -> pd.Series:
    return x.rolling(k, min_periods=k).mean()


def update_inputs(store: Path, today: dt.date) -> pd.DataFrame:
    old = pd.read_csv(store, parse_dates=["date"]).set_index("date") if store.exists() else pd.DataFrame()
    days = [today - dt.timedelta(days=k) for k in range(BACK, 0, -1)]
    need = []
    for d in days:
        ts = pd.Timestamp(d)
        if ts not in old.index or (old.loc[ts, "n_cycles"] < 4 and (today - d).days <= RETRY):
            need.append(d)
    print(f"inputs: {len(old)} stored days, fetching {len(need)}", flush=True)
    rows = []
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(6) as ex:
        rows = list(ex.map(GD.day, need))
    new = pd.DataFrame(rows)
    if len(new):
        new["date"] = pd.to_datetime(new["date"]); new = new.set_index("date")
        old = new.combine_first(old) if len(old) else new
        old.update(new)
    old = old.sort_index()
    store.parent.mkdir(parents=True, exist_ok=True)
    old.reset_index().to_csv(store, index=False, float_format="%.4f")
    return old


def mjo_daily() -> pd.DataFrame:
    """The site's observed wind-only RMM (scripts/mjo/data/reference/obs_history.nc, AIFS analyses) with the ENSO part
    removed (scripts/mjo/src/enso_rmm.py), one value per day (the latest analysis of that date)."""
    import xarray as xr
    sys.path.insert(0, str(REPO / "scripts" / "mjo" / "src"))
    import enso_rmm as ER
    obs = xr.open_dataset(REPO / "scripts" / "mjo" / "data" / "reference" / "obs_history.nc").load()
    n34 = ER.n34_smoothed()
    if n34 is not None:
        obs = ER.clean_obs(obs, n34, ER.load_c())
    t = pd.to_datetime(obs["time"].values)
    df = pd.DataFrame({"r1": obs["rmm1"].values, "r2": obs["rmm2"].values}, index=t)
    df = df.groupby(df.index.normalize()).last()
    df["amp"] = np.hypot(df.r1, df.r2); df["phase"] = ER.phase_of(df.r1.values, df.r2.values)
    df["cleaned"] = n34 is not None
    return df


def mjo_cat(amp: float, ph: int) -> int:
    if not np.isfinite(amp) or amp < 1:
        return 0
    return 1 if ph in (8, 1) else 2 if ph in (2, 3) else 4 if ph in (6, 7) else 0


def in_season(d: pd.Timestamp) -> bool:
    a, b = d + pd.Timedelta(days=15), d + pd.Timedelta(days=28)
    djf = lambda x: x.month in (12, 1, 2)
    wa = a.year + (1 if a.month == 12 else 0); wb = b.year + (1 if b.month == 12 else 0)
    return djf(a) and djf(b) and wa == wb


def ssw_flag(u: pd.Series) -> tuple[pd.Series, list]:
    """CMIP6 training rule: the daily 60N 10 hPa zonal-mean wind turns easterly between Nov 1 and Feb 10, onsets >= 30 days
    apart; flagged on the onset day and the 29 days after."""
    on, last = [], None
    v = u.values; d = u.index
    for i in range(1, len(v)):
        if not (np.isfinite(v[i]) and np.isfinite(v[i - 1])) or not (v[i] < 0 <= v[i - 1]):
            continue
        if d[i].month not in (11, 12, 1, 2) or (d[i].month == 2 and d[i].day > 10):
            continue
        if last is not None and (d[i] - last).days < 30:
            continue
        on.append(d[i]); last = d[i]
    f = pd.Series(0, index=d)
    for o in on:
        f[(d >= o) & (d < o + pd.Timedelta(days=30))] = 1
    return f, on


def build(inp: pd.DataFrame, mjo: pd.DataFrame | None) -> pd.DataFrame:
    full = inp[inp.n_cycles >= 4].reindex(pd.date_range(inp.index.min(), inp.index.max(), freq="D"))
    A = {k: anomaly(full[k], k) for k in ("t2m_us", "pch1000", "pch100", "gb_raw", "nao_raw")}
    zc = REF["z"]; zz = lambda x, key: (x - zc[key][0]) / zc[key][1]
    F = pd.DataFrame(index=full.index)
    F["z_gb"] = zz(trailing(A["gb_raw"], 5), "greenland_block"); F["z_ao"] = zz(trailing(A["pch1000"], 5), "pch1000")
    F["z_nao"] = zz(trailing(A["nao_raw"], 5), "nao"); F["z_vortex"] = zz(trailing(A["pch100"], 5), "pch100")
    F["z_recent"] = zz(trailing(A["t2m_us"], 7), "pers7"); F["t2m_anom"] = A["t2m_us"]
    F["z_block"] = (F.z_gb + F.z_ao - F.z_nao) / 3
    F["u10"] = full["u10_60n"]; F["ssw"], onsets = ssw_flag(full["u10_60n"])
    F.attrs["ssw_onsets"] = [o.strftime("%Y-%m-%d") for o in onsets]
    bins = REF["bins"]; lr = REF["lr"]; b0, b1 = REF["cal"]["b0"], REF["cal"]["b1"]
    for k, col in (("block", "z_block"), ("vortex", "z_vortex"), ("recent", "z_recent")):
        b = np.digitize(F[col].values, bins[k]).astype(float); b[~np.isfinite(F[col].values)] = np.nan
        F[f"bin_{k}"] = b
    F["bin_ssw"] = np.where(F.u10.notna(), F.ssw, np.nan)
    if mjo is not None:
        M = mjo.reindex(F.index)
        F["mjo_amp"], F["mjo_phase"] = M.amp, M.phase
        F["bin_mjo"] = [mjo_cat(a, int(p)) if np.isfinite(a) else np.nan for a, p in zip(M.amp, M.phase.fillna(0))]
    else:
        F["mjo_amp"] = F["mjo_phase"] = F["bin_mjo"] = np.nan
    logit0 = np.log(0.15 / 0.85)
    circ = sum(np.log(F[f"bin_{k}"].map(lambda v: lr[k].get(str(int(v)), 1.0) if np.isfinite(v) else np.nan)) for k in ("block", "vortex", "recent", "ssw"))
    mj = np.log(F["bin_mjo"].map(lambda v: REF["mjo_lr"].get(str(int(v)), 1.0) if np.isfinite(v) else 1.0))
    F["p"] = 1 / (1 + np.exp(-(b0 + b1 * (logit0 + circ) + REF["mjo_shrink"] * mj)))
    F["in_season"] = [in_season(d) for d in F.index]
    return F


def clue_list(r: pd.Series) -> list:
    b1, sh = REF["cal"]["b1"], REF["mjo_shrink"]; lr = REF["lr"]
    out = []
    for k, zcol in (("block", "z_block"), ("vortex", "z_vortex"), ("recent", "z_recent"), ("ssw", None), ("mjo", None)):
        b = r.get(f"bin_{k}")
        if b is None or not np.isfinite(b):
            out.append({"key": k, "name": NAMES[k], "bin": None, "mult": None}); continue
        L_ = (REF["mjo_lr"] if k == "mjo" else lr[k]).get(str(int(b)), 1.0)
        m = L_ ** (sh if k == "mjo" else b1)
        e = {"key": k, "name": NAMES[k], "bin": int(b), "mult": round(float(m), 3)}
        if zcol:
            e["z"] = round(float(r[zcol]), 2)
        if k == "block":
            e["parts"] = {"greenland": round(float(r.z_gb), 2), "ao_cap": round(float(r.z_ao), 2), "nao": round(float(r.z_nao), 2)}
        if k == "ssw":
            e["u10"] = round(float(r.u10), 1)
        if k == "mjo":
            e["amp"] = round(float(r.mjo_amp), 2) if np.isfinite(r.mjo_amp) else None
            e["phase"] = int(r.mjo_phase) if np.isfinite(r.mjo_amp) and r.mjo_amp >= 1 else None   # no MJO, no phase
        out.append(e)
    return out


def tier(p: float) -> str:
    return "high" if p >= 0.25 else "low" if p <= 0.10 else "normal"


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(REPO / "assets" / "coldindex"))
    ap.add_argument("--today", default=None, help="YYYY-MM-DD (testing)")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    today = dt.date.fromisoformat(a.today) if a.today else dt.datetime.utcnow().date()
    inp = update_inputs(out / "inputs.csv", today)
    try:
        mjo = mjo_daily()
    except Exception as e:                                               # noqa: BLE001 - the index runs without the MJO term
        print(f"MJO unavailable: {e!r}"); mjo = None
    F = build(inp, mjo)
    F = F[F.index < pd.Timestamp(today)]                                  # the state up to yesterday
    ok = F["p"].notna() & F[["bin_block", "bin_vortex", "bin_recent", "bin_ssw"]].notna().all(1)
    if not ok.any():
        raise SystemExit("no complete day yet")
    last = F.index[ok][-1]; r = F.loc[last]
    seas = r.in_season
    yr = last.year if last.month >= 7 else last.year - 1
    cur = {"updated": dt.datetime.utcnow().strftime("%Y-%m-%d %H:%MZ"), "issue": last.strftime("%Y-%m-%d"),
           "inputs_through": inp.index.max().strftime("%Y-%m-%d"), "in_season": bool(seas),
           "season": {"first": f"{yr}-11-16", "last": f"{yr + 1}-01-31"},
           "target": [(last + pd.Timedelta(days=15)).strftime("%Y-%m-%d"), (last + pd.Timedelta(days=28)).strftime("%Y-%m-%d")],
           "p": round(float(r.p) * 100, 1) if seas else None, "tier": tier(float(r.p)) if seas else None,
           "clues": clue_list(r), "mjo_source": "site wind RMM, ENSO removed" if (mjo is not None and bool(mjo.cleaned.iloc[-1])) else ("site wind RMM" if mjo is not None else None),
           "ssw_onsets": F.attrs.get("ssw_onsets", []),
           "history": []}
    since = min(pd.Timestamp(f"{yr}-10-01"), last - pd.Timedelta(days=44))     # the season so far, and at least 45 days for the barcode
    for d, x in F[(F.index >= since) & ok].iterrows():
        cur["history"].append({"d": d.strftime("%Y-%m-%d"), "p": round(float(x.p) * 100, 1) if x.in_season else None,
                               "t": round(float(x.t2m_anom), 2) if np.isfinite(x.t2m_anom) else None,
                               "c": [int(x[f"bin_{k}"]) for k in ("block", "vortex", "recent", "ssw")] + [int(x.bin_mjo) if np.isfinite(x.bin_mjo) else -1],
                               "z": [round(float(x[c]), 2) for c in ("z_block", "z_vortex", "z_recent")]})
    (out / "current.json").write_text(json.dumps(cur, separators=(",", ":")))
    meta = {k: REF[k] for k in ("target", "season", "bins", "lr", "cal", "mjo_lr", "mjo_shrink", "mjo_categories", "ssw", "skill", "training", "lr_ci", "built")}
    meta["gfs_check"] = {k: v["gfs_check"] for k, v in REF["series"].items()}
    (out / "meta.json").write_text(json.dumps(meta, separators=(",", ":")))
    shutil.copyfile(HERE / "reference" / "replay.json", out / "replay.json")
    print(json.dumps({k: cur[k] for k in ("issue", "in_season", "p", "tier", "target")}), json.dumps(cur["clues"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
