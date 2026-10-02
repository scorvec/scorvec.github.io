#!/usr/bin/env python3
"""How well does the historical PROXY (200 hPa -v_chi . grad(eta), 1.5 deg) track the LIVE outflow index (300-200 hPa
-v_chi . grad(PV), 0.25 deg, tc_jet)? Both are computed from the SAME AIFS-ENS control fields for every storm position in
a few recent cycles (all 12-hourly leads to day 10), so the comparison isolates the definition, not the data.

    python src/tc_downstream_proxy_check.py 20260927 20260929 20261001 20261002
Writes the rank correlation, a linear map live = a + b * proxy, and the live-index value matching the historical
strong-tercile proxy threshold per basin into scripts/mjo/data/reference/tc_downstream.json ("proxy_check").
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ecmwf"))
import build_tc_downstream as B                                           # noqa: E402
import dt_charts as D                                                     # noqa: E402
import store as ecmwf                                                     # noqa: E402
import tc_jet as TJ                                                       # noqa: E402

REF = Path(__file__).resolve().parents[1] / "data" / "reference"


def basin_of(lon360: float) -> str:
    return "WP" if lon360 < 180 else ("EP" if lon360 < 260 else "NA")


def main() -> int:
    rows = []
    g15 = dict(latitude=np.arange(-90, 90.01, 1.5), longitude=np.arange(0, 360, 1.5))
    for date in sys.argv[1:]:
        cyc = ecmwf.Cycle(date, "00")
        fp = TJ.fetch(date, 0, Path(__file__).resolve().parents[2] / "ecmwf" / "cache" / "tc")
        storms = TJ.nh_storms(TJ.decode(fp)) if fp else []
        if not storms:
            continue
        d = D.load(cyc)
        lat, lon = d["t"].latitude.values, d["t"].longitude.values
        p_pa = np.array(D.LEVS, float) * 100
        steps = (d["t"].step / np.timedelta64(1, "h")).values.astype(int)
        raw = {}
        for par in ("u", "v"):
            p = ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", "cf", par, "pl", D.LEVS, D.STEPS))
            da = xr.open_dataset(p, engine="cfgrib", backend_kwargs={"indexpath": ""})[par]
            da = (da.squeeze("number", drop=True) if "number" in da.dims else da).sel(isobaricInhPa=200)
            raw[par] = da.assign_coords(longitude=da.longitude % 360).sortby("longitude")   # open data is -180..180
        for k, h in enumerate(steps):
            here = [(s, TJ.at(s, int(h))) for s in storms]
            here = [(s, pos) for s, pos in here if pos is not None and pos[0] > 0]
            if not here:
                continue
            pv, _ = D.pv_step(d["t"].isel(step=k).values, d["u"].isel(step=k).values, d["v"].isel(step=k).values, p_pa, lat, lon)
            uc, vc = TJ.irrotational(d["u_layer"].isel(step=k), d["v_layer"].isel(step=k), lat, lon)
            live = TJ.pv_advection(uc, vc, pv[[D.LEVS.index(300), D.LEVS.index(250), D.LEVS.index(200)]].mean(0), lat, lon)
            live[lat > 85] = np.nan
            u2 = raw["u"].isel(step=k).sortby("latitude").interp(**g15, kwargs={"fill_value": "extrapolate"}).values
            v2 = raw["v"].isel(step=k).sortby("latitude").interp(**g15, kwargs={"fill_value": "extrapolate"}).values
            prox = B.proxy_field(u2, v2, g15["latitude"], g15["longitude"])
            for s, pos in here:
                rows.append(dict(cycle=date, h=int(h), storm=TJ.label(s), lat=pos[0], lon=pos[1] % 360, basin=basin_of(pos[1] % 360),
                                 live=TJ.disc_index(live, lat, lon, *pos), proxy=TJ.disc_index(prox, g15["latitude"], g15["longitude"], *pos)))
        print(f"  {date}: {len(rows)} storm-times so far", flush=True)
    df = pd.DataFrame(rows); print(df.isna().sum().to_dict(), flush=True); df = df.dropna(subset=["live", "proxy"])
    df.to_csv(REF / "tc_downstream_proxy_check.csv", index=False)
    from scipy import stats
    rho = stats.spearmanr(df.proxy, df.live)
    b, a = np.polyfit(df.proxy, df.live, 1)
    js = json.loads((REF / "tc_downstream.json").read_text()) if (REF / "tc_downstream.json").exists() else {}
    thr = js.get("thresholds", {})
    out = dict(n=int(len(df)), storms=int(df.storm.nunique()), cycles=sys.argv[1:], spearman=round(float(rho.statistic), 3),
               p=float(rho.pvalue), fit_live_eq_a_plus_b_proxy=[round(float(a), 4), round(float(b), 4)],
               live_at_strong_threshold={k: round(float(a + b * v["p67"]), 4) for k, v in thr.items()})
    print(json.dumps(out, indent=1))
    if js:
        js["proxy_check"] = out
        (REF / "tc_downstream.json").write_text(json.dumps(js))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
