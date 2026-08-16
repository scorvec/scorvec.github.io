#!/usr/bin/env python3
"""XM actual generation: national hydro vs total, 2000->present.

POST /hourly MetricId=Gene Entity=Recurso returns per-plant-code hourly
kWh; plant codes classify via ListadoRecursos (Type: HIDRAULICA,
TERMICA, SOLAR, EOLICA, COGENERADOR). Aggregated AT INGEST to daily
national {hydro, total} kWh — the raw hourly is never stored.

Outputs:
  ~/colombia_hydro/raw/generation_daily.json.gz   (cache, incremental)
  colombia_hydro/generation.webp                  (trend + norms + share)
  colombia_hydro/data/generation.json

    python scripts/sst/xm_generation.py [--backfill]
"""
from __future__ import annotations

import gzip
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
API = "https://servapibi.xm.com.co/hourly"
LISTS = "https://servapibi.xm.com.co/lists"
CACHE = Path.home() / "colombia_hydro" / "raw" / "generation_daily.json.gz"
RECURSOS = Path.home() / "colombia_hydro" / "raw" / "xm_listado_recursos.json"
OUT_PNG = REPO / "colombia_hydro" / "generation.webp"
OUT_JSON = REPO / "colombia_hydro" / "data" / "generation.json"
MODEL_JSON = REPO / "colombia_hydro" / "data" / "gen_model.json"
FAN_JSON = REPO / "colombia_hydro" / "data" / "inflow_forecast.json"
APOR_CACHE = Path.home() / "colombia_hydro" / "raw" / "aporener_daily.json.gz"
NINO_JSON = REPO / "assets" / "sst" / "data" / "nino_history.json"
START = "2000-01-01"
WIN = 10
PCTS = [10, 25, 50, 75, 90]


def hydro_codes() -> tuple[set, set]:
    """(hydro codes, all codes). Refreshes the listing if >30 d old."""
    import time as _t
    if not RECURSOS.exists() or _t.time() - RECURSOS.stat().st_mtime > 30 * 86400:
        r = requests.post(LISTS, json={"MetricId": "ListadoRecursos",
                                       "Entity": "Sistema"}, timeout=90)
        ents = [e["Values"] for it in r.json()["Items"] for e in it["ListEntities"]]
        RECURSOS.write_text(json.dumps(ents))
    ents = json.loads(RECURSOS.read_text())
    hyd = {v["Code"] for v in ents if str(v.get("Type", "")).upper() == "HIDRAULICA"}
    return hyd, {v["Code"] for v in ents}


def fetch_range(d0: datetime, d1: datetime, hyd: set) -> dict:
    """{day: {'hydro': kWh, 'total': kWh, 'n_hyd': plants}} for [d0, d1]."""
    out = {}
    cur = d0
    while cur <= d1:
        end = min(cur + timedelta(days=29), d1)
        for attempt in range(3):
            try:
                r = requests.post(API, json={"MetricId": "Gene",
                                             "StartDate": f"{cur:%Y-%m-%d}",
                                             "EndDate": f"{end:%Y-%m-%d}",
                                             "Entity": "Recurso"}, timeout=180)
                r.raise_for_status()
                break
            except Exception as e:                    # noqa: BLE001
                if attempt == 2:
                    raise
                print(f"  retry {cur:%Y-%m} ({repr(e)[:40]})", flush=True)
        for it in r.json().get("Items", []):
            day = it["Date"]
            rec = out.setdefault(day, {"hydro": 0.0, "total": 0.0, "n_hyd": 0})
            for e in it.get("HourlyEntities", []):
                v = e["Values"]
                code = v.get("code")
                tot = sum(float(v[f"Hour{h:02d}"] or 0) for h in range(1, 25)
                          if v.get(f"Hour{h:02d}") not in (None, ""))
                rec["total"] += tot
                if code in hyd:
                    rec["hydro"] += tot
                    rec["n_hyd"] += 1
        print(f"  {cur:%Y-%m-%d}..{end:%Y-%m-%d}: {len(out)} days", flush=True)
        cur = end + timedelta(days=1)
    return out


def load_gen(backfill: bool) -> dict:
    data = {}
    if CACHE.exists():
        with gzip.open(CACHE, "rt") as f:
            data = json.load(f)
    hyd, _ = hydro_codes()
    have = sorted(data)
    d1 = datetime.now() - timedelta(days=1)
    if backfill or not have:
        d0 = datetime.strptime(START, "%Y-%m-%d")
    else:
        d0 = datetime.strptime(have[-1], "%Y-%m-%d") - timedelta(days=2)
    # skip already-complete stretch on backfill
    if backfill and have:
        # fetch only missing days before the first cached + after the last
        first = datetime.strptime(have[0], "%Y-%m-%d")
        if first > d0:
            data.update(fetch_range(d0, first - timedelta(days=1), hyd))
        d0 = datetime.strptime(have[-1], "%Y-%m-%d") - timedelta(days=2)
    if (d1 - d0).days >= 0:
        data.update(fetch_range(d0, d1, hyd))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(CACHE, "wt") as f:
        json.dump(data, f, separators=(",", ":"))
    return data




def trailing_doy_norm(x, doy, half=1095, win=10):
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        lo = max(0, i - half)
        dist = np.minimum(np.abs(doy[lo:i] - doy[i]), 365 - np.abs(doy[lo:i] - doy[i]))
        v = x[lo:i][dist <= win]
        v = v[np.isfinite(v)]
        if len(v) > 15:
            out[i] = np.median(v)
    return out


def fit_gen_model(days, dates, G, doy):
    """Persistence + state model for daily hydro generation.

    G = Gn(doy, trailing 3 yr) + W(weekday) + Ga'; forecast blends
    persistence of Ga' (alpha decays with lead) with the state model
    Ga' ~ inflow7_anom + storage_anom + nino34 fitted on 2003+.
    Saves everything the forecast engine needs to MODEL_JSON."""
    import sys as _s
    _s.path.insert(0, str(HERE))
    from xm_inflow_history import river_region
    from xm_storage import pct_anomaly_series

    Gn = trailing_doy_norm(G, doy)
    wd = np.array([d.astype(datetime).weekday() for d in dates.astype("datetime64[s]")])
    Ga0 = G - Gn
    W = np.zeros(7)
    rec3 = np.arange(len(G)) > len(G) - 1095
    for w in range(7):
        v = Ga0[rec3 & (wd == w)]
        v = v[np.isfinite(v)]
        if len(v) > 20:
            W[w] = np.median(v)
    Ga = Ga0 - W[wd]

    # persistence decay (weekday-adjusted), fit alpha(h) = a0*exp(-h/tau)
    lags = [1, 2, 3, 5, 10, 15, 20, 30]
    rs = []
    for h in lags:
        a, b = Ga[:-h], Ga[h:]
        m = np.isfinite(a) & np.isfinite(b)
        rs.append(float(np.corrcoef(a[m], b[m])[0, 1]))
    lr = np.log(np.clip(rs, 1e-3, None))
    tau = -1.0 / np.polyfit(lags, lr, 1)[0]
    a0 = float(np.exp(np.polyfit(lags, lr, 1)[1]))
    tau = float(np.clip(tau, 5, 60))

    # state model: national inflow anomaly (7-d trailing), storage anom, nino
    with gzip.open(APOR_CACHE, "rt") as f:
        apor = json.load(f)
    r2reg = river_region()
    I = np.array([sum(v for riv, v in apor.get(d, {}).items() if riv in r2reg)
                  for d in days]) / 1e6
    I[I == 0] = np.nan
    In = trailing_doy_norm(I, doy)
    Ia = I - In
    Ia7 = np.convolve(np.where(np.isfinite(Ia), Ia, 0), np.ones(7) / 7,
                      "full")[:len(Ia)]
    sdates, sanom = pct_anomaly_series()
    nat = np.nanmean(np.array([sanom[r] for r in ORDER_S]), axis=0)
    pan = np.full(len(days), np.nan)
    _, ci, si = np.intersect1d(dates, sdates, return_indices=True)
    pan[ci] = nat[si]
    nh = json.loads(NINO_JSON.read_text())
    nmon = np.array(nh["months"], dtype="datetime64[M]")
    nanom = np.array(nh["series"]["nino34"]["anom"], float)
    dmon = dates.astype("datetime64[M]")
    idx = np.minimum(np.searchsorted(nmon, dmon), len(nmon) - 1)
    nino = np.where(nmon[idx] == dmon, nanom[idx], np.nan)
    years = np.array([int(d[:4]) for d in days])
    m = (np.isfinite(Ga) & np.isfinite(Ia7) & np.isfinite(pan)
         & np.isfinite(nino) & (years >= 2003))
    X = np.column_stack([np.ones(m.sum()), Ia7[m], pan[m], nino[m]])
    beta, *_ = np.linalg.lstsq(X, Ga[m], rcond=None)
    fit = X @ beta
    r_in = float(np.corrcoef(fit, Ga[m])[0, 1])
    # LOYO OOS
    yh = np.full(len(days), np.nan)
    for yr in np.unique(years[m]):
        tr = m & (years != yr)
        te = m & (years == yr)
        if te.sum() < 100:
            continue
        bb, *_ = np.linalg.lstsq(np.column_stack(
            [np.ones(tr.sum()), Ia7[tr], pan[tr], nino[tr]]), Ga[tr], rcond=None)
        yh[te] = np.column_stack(
            [np.ones(te.sum()), Ia7[te], pan[te], nino[te]]) @ bb
    mm = np.isfinite(yh) & np.isfinite(Ga)
    r_oos = float(np.corrcoef(yh[mm], Ga[mm])[0, 1])

    # doy vectors for the engine (last value per doy from trailing norms)
    Gn365 = np.full(365, np.nan)
    In365 = np.full(365, np.nan)
    for d_ in range(1, 366):
        ix = np.where(doy == d_)[0]
        for arr, tgt in ((Gn, Gn365), (In, In365)):
            v = arr[ix][np.isfinite(arr[ix])]
            if len(v):
                tgt[d_ - 1] = v[-1]
    for arr in (Gn365, In365):        # fill wrap gaps by interpolation
        bad = ~np.isfinite(arr)
        if bad.any():
            arr[bad] = np.interp(np.where(bad)[0], np.where(~bad)[0], arr[~bad])
    ga_sm = Ga[np.isfinite(Ga)][-5:]
    ia_rec = Ia[np.isfinite(Ia)][-6:]
    MODEL_JSON.write_text(json.dumps({
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "r_in_sample": round(r_in, 3), "r_loyo": round(r_oos, 3),
        "persistence": {"a0": round(a0, 3), "tau_days": round(tau, 1),
                        "lag_r": dict(zip(map(str, lags), np.round(rs, 3)))},
        "coefs": {"c0": round(float(beta[0]), 2),
                  "inflow7_gwh_per_gwh": round(float(beta[1]), 3),
                  "storage_gwh_per_pt": round(float(beta[2]), 3),
                  "nino_gwh_per_degc": round(float(beta[3]), 2)},
        "weekday_offsets_gwh": np.round(W, 2).tolist(),
        "gn365_gwh": np.round(Gn365, 1).tolist(),
        "in365_gwh": np.round(In365, 1).tolist(),
        "ga_now_gwh": round(float(np.mean(ga_sm)), 2),
        "sigma_ga_gwh": round(float(np.nanstd(Ga)), 2),
        "ia_recent_gwh": np.round(ia_rec, 2).tolist(),
        "storage_anom_now": round(float(nat[np.isfinite(nat)][-1]), 2),
        "nino_now": round(float(nanom[np.isfinite(nanom)][-1]), 2),
        "last_day": days[-1],
    }, separators=(",", ":")))
    print(f"gen model: in-sample r={r_in:.3f}, LOYO r={r_oos:.3f}, "
          f"persistence a0={a0:.2f} tau={tau:.0f}d")


ORDER_S = ["ANTIOQUIA", "CALDAS", "CARIBE", "CENTRO", "ORIENTE", "VALLE"]


def main(argv=None) -> int:
    backfill = "--backfill" in (argv or sys.argv[1:])
    data = load_gen(backfill)
    days = sorted(d for d, v in data.items() if v["total"] > 0)
    dates = np.array(days, dtype="datetime64[D]")
    gh = np.array([data[d]["hydro"] for d in days]) / 1e6      # GWh/day
    gt = np.array([data[d]["total"] for d in days]) / 1e6
    share = 100 * gh / gt
    doy = np.array([min(datetime.strptime(d, "%Y-%m-%d").timetuple().tm_yday, 365)
                    for d in days])
    print(f"generation: {days[0]}..{days[-1]} ({len(days)} days) "
          f"hydro now ~{gh[-30:].mean():.0f} GWh/d ({share[-30:].mean():.0f}%)",
          flush=True)

    # doy envelope of hydro gen (recent-decade flavor kept simple: full record)
    env = np.full((365, len(PCTS)), np.nan)
    for d_ in range(1, 366):
        dist = np.minimum(np.abs(doy - d_), 365 - np.abs(doy - d_))
        m = (dist <= WIN) & np.isfinite(gh)
        if m.sum() > 20:
            env[d_ - 1] = np.percentile(gh[m], PCTS)

    fit_gen_model(days, dates, gh, doy)

    fan = None
    if FAN_JSON.exists():
        f_ = json.loads(FAN_JSON.read_text())
        if "generation" in f_ and (np.datetime64(datetime.now(timezone.utc)
                                   .strftime("%Y-%m-%d"))
                                   - np.datetime64(f_["dates"][0])).astype(int) <= 2:
            fan = f_

    t = dates.astype("datetime64[s]").astype(datetime)
    k365 = np.ones(365) / 365
    fig, axes = plt.subplots(3, 1, figsize=(13.0, 11.5))
    ax = axes[0]
    ax.plot(t, gh, color="#9db8d8", lw=0.4, alpha=0.7)
    if len(gh) > 365:
        ax.plot(t[364:], np.convolve(gh, k365, "valid"), color="#1f4e8c", lw=1.8,
                label="365-day mean")
    ax.set_title("National hydro generation, GWh/day — daily + annual mean",
                 fontsize=11, fontweight="bold", loc="left")
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.plot(t, share, color="#d8b89d", lw=0.4, alpha=0.7)
    if len(share) > 365:
        ax.plot(t[364:], np.convolve(share, k365, "valid"), color="#b35806", lw=1.8,
                label="365-day mean")
    ax.set_ylabel("%", fontsize=9)
    ax.set_title("Hydro share of total generation, %", fontsize=11,
                 fontweight="bold", loc="left")
    ax.legend(fontsize=8)
    ax = axes[2]
    x365 = np.arange(1, 366)
    ax.fill_between(x365, env[:, 0], env[:, 4], color="#9db8d8", alpha=0.35,
                    label="p10–p90")
    ax.fill_between(x365, env[:, 1], env[:, 3], color="#5b87c0", alpha=0.35,
                    label="p25–p75")
    ax.plot(x365, env[:, 2], color="#1f4e8c", lw=1.5, label="median")
    rec = dates > dates[-1] - np.timedelta64(200, "D")
    ax.plot(doy[rec], gh[rec], color="#c62828", lw=1.2, label="last 200 days")
    if fan is not None:
        q = fan["generation"]
        fdoy = np.array([min(datetime.strptime(x, "%Y-%m-%d").timetuple().tm_yday,
                             365) for x in fan["dates"]])
        wrap = np.where(np.diff(fdoy) < 0)[0]
        stop = int(wrap[0]) + 1 if len(wrap) else len(fdoy)
        fd = fdoy[:stop]
        ax.fill_between(fd, q["p10"][:stop], q["p90"][:stop], color="#e08214",
                        alpha=0.25, lw=0)
        ax.fill_between(fd, q["p25"][:stop], q["p75"][:stop], color="#e08214",
                        alpha=0.35, lw=0)
        ax.plot(fd, q["p50"][:stop], color="#b35806", lw=1.4, ls="--",
                label="forecast (persistence + inflow/storage states)")
    ax.set_title("Hydro generation vs day-of-year envelope (full record)",
                 fontsize=11, fontweight="bold", loc="left")
    ax.set_xlim(1, 365)
    ax.set_xticks([1, 91, 182, 274, 365])
    ax.set_xticklabels(["Jan", "Apr", "Jul", "Oct", "Jan"])
    ax.legend(fontsize=8, loc="lower left")
    for ax in axes:
        ax.grid(lw=0.25, alpha=0.5)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("GWh/day", fontsize=9)
    axes[2].set_ylabel("GWh/day", fontsize=9)
    fig.suptitle("XM actual generation — hydro output and hydro share, "
                 f"{days[0][:4]}–{days[-1][:4]}", fontsize=12.5,
                 fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(OUT_PNG, dpi=115)
    plt.close(fig)
    print(f"wrote {OUT_PNG.relative_to(REPO)}")

    keep = dates > dates[-1] - np.timedelta64(3 * 365, "D")
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "units": "GWh/day",
        "env_doy": {"pcts": PCTS, "hydro": np.round(np.nan_to_num(env), 1).tolist()},
        "recent": {"dates": [str(x) for x in dates[keep]],
                   "hydro": np.round(gh[keep], 1).tolist(),
                   "total": np.round(gt[keep], 1).tolist()},
    }, separators=(",", ":")))
    print(f"wrote {OUT_JSON.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
