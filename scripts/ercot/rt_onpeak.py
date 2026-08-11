#!/usr/bin/env python3
"""ERCOT real-time on-peak daily prices, 2011-present — site product.

Source: ERCOT "Historical RTM Load Zone and Hub Prices" (NP6-785, public MIS),
one xlsx per year of 15-min settlement point prices. Document IDs change as
files update, so the list is resolved live; past years are cached (xlsx +
parsed per-year CSV), only the current year re-downloads on --refresh.

Series: HB_NORTH (the hub the liquid ICE 5x16 contract settles against) as
the headline, HB_HUBAVG for reference. Daily on-peak = mean of HE 7-22.
5x16 flag = weekday excluding the six NERC holidays (Sun->Mon observance).

Outputs: assets/ercot/rt_onpeak.webp (chart, HB_NORTH 5x16)
         assets/ercot/rt_onpeak_daily.csv (all days, both hubs, 5x16 flag)

    python rt_onpeak.py             # parse cache, rebuild outputs
    python rt_onpeak.py --refresh   # re-download current year first
"""
from __future__ import annotations
import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DATA = HERE / "data"
ASSETS = REPO / "assets" / "ercot"
HUBS = ("HB_NORTH", "HB_HUBAVG")
Y0 = 2011
LIST_URL = ("https://www.ercot.com/misapp/servlets/IceDocListJsonWS"
            "?reportTypeId=13061")
DL_URL = "https://www.ercot.com/misdownload/servlets/mirDownload?doclookupId={}"
UA = {"User-Agent": "Mozilla/5.0"}


def doc_ids() -> dict[int, str]:
    r = requests.get(LIST_URL, headers=UA, timeout=60)
    r.raise_for_status()
    out = {}
    for doc in r.json()["ListDocsByRptTypeRes"]["DocumentList"]:
        d = doc["Document"]
        name = d["FriendlyName"]                      # RTMLZHBSPP_2026
        if "_" in name:
            try:
                out[int(name.split("_")[-1])] = d["DocID"]
            except ValueError:
                pass
    return out


def fetch_year(yr: int, docid: str, force: bool = False) -> Path:
    f = DATA / f"rtm_{yr}.xlsx"
    if f.exists() and f.stat().st_size > 1e6 and not force:
        return f
    DATA.mkdir(parents=True, exist_ok=True)
    r = requests.get(DL_URL.format(docid), headers=UA, timeout=300)
    r.raise_for_status()
    f.write_bytes(r.content)
    print(f"{yr}: downloaded {f.stat().st_size/1e6:.0f} MB", flush=True)
    return f


def parse_year(yr: int, force: bool = False) -> pd.DataFrame | None:
    """Hourly on-peak-window prices for HUBS, cached per year."""
    cache = DATA / f"hub_{yr}.csv"
    src = DATA / f"rtm_{yr}.xlsx"
    if cache.exists() and not force:
        return pd.read_csv(cache, parse_dates=["date"])
    if not src.exists():
        return None
    rows = []
    with zipfile.ZipFile(src) as z:
        inner = [n for n in z.namelist() if n.lower().endswith(".xlsx")][0]
        sheets = pd.read_excel(io.BytesIO(z.read(inner)), sheet_name=None)
    for _name, df in sheets.items():
        df.columns = [str(c).strip() for c in df.columns]
        cols = {c.lower(): c for c in df.columns}
        spn = next((cols[k] for k in cols
                    if "settlement point name" in k or k == "settlement point"),
                   None)
        spp = next((cols[k] for k in cols if "price" in k), None)
        dd = next((cols[k] for k in cols if "delivery date" in k), None)
        dh = next((cols[k] for k in cols
                   if "delivery hour" in k or k == "hour ending"), None)
        if not all((spn, spp, dd, dh)):
            continue
        sub = df[df[spn].isin(HUBS)]
        if not len(sub):
            continue
        rows.append(pd.DataFrame({
            "date": pd.to_datetime(sub[dd].astype(str), errors="coerce"),
            "hour": pd.to_numeric(sub[dh], errors="coerce"),
            "hub": sub[spn].values,
            "price": pd.to_numeric(sub[spp], errors="coerce")}))
    if not rows:
        return None
    d = pd.concat(rows).dropna()
    d = d[(d.hour >= 7) & (d.hour <= 22)]
    d = (d.groupby(["date", "hub"])["price"].mean().unstack("hub")
         .rename(columns={"HB_NORTH": "north", "HB_HUBAVG": "hubavg"})
         .reset_index())
    d.to_csv(cache, index=False)
    print(f"{yr}: parsed {len(d)} days", flush=True)
    return d


def nerc_holidays(t0, t1):
    from pandas.tseries.holiday import (AbstractHolidayCalendar, Holiday,
                                        USMemorialDay, USLaborDay,
                                        USThanksgivingDay, sunday_to_monday)

    class NERC(AbstractHolidayCalendar):
        rules = [Holiday("NewYear", month=1, day=1,
                         observance=sunday_to_monday),
                 USMemorialDay,
                 Holiday("July4", month=7, day=4,
                         observance=sunday_to_monday),
                 USLaborDay, USThanksgivingDay,
                 Holiday("Christmas", month=12, day=25,
                         observance=sunday_to_monday)]
    return NERC().holidays(t0, t1)


def build(refresh: bool = False):
    ids = doc_ids()
    cur = max(y for y in ids if y >= Y0)
    for yr in sorted(y for y in ids if Y0 <= y):
        force = refresh and yr == cur
        try:
            fetch_year(yr, ids[yr], force=force)
        except Exception as e:                        # noqa: BLE001
            print(f"{yr}: fetch failed ({str(e)[:80]})", flush=True)
        parse_year(yr, force=force)

    parts = [p for yr in range(Y0, cur + 1)
             if (p := parse_year(yr)) is not None]
    d = pd.concat(parts).sort_values("date").set_index("date")
    hol = nerc_holidays(d.index.min(), d.index.max())
    d["is_5x16"] = (d.index.dayofweek < 5) & ~d.index.isin(hol)
    ASSETS.mkdir(parents=True, exist_ok=True)
    d.round(2).to_csv(ASSETS / "rt_onpeak_daily.csv")

    chart(d)
    print(f"series: {len(d)} days -> {d.index.max():%Y-%m-%d}")


def chart(d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = d.loc[d.is_5x16, "north"].dropna()
    sp = s.clip(lower=1.0)
    fig, ax = plt.subplots(figsize=(14.2, 5.8), constrained_layout=True)
    ax.semilogy(sp.index, sp.values, lw=0.55, color="#2e5d9e", alpha=0.85,
                label="daily on-peak average (HB_NORTH)")
    roll = sp.rolling(30, center=True, min_periods=10).median()
    ax.semilogy(roll.index, roll.values, lw=1.8, color="#c62828",
                label="30-day rolling median")
    events = [("2011-02-02", "Feb 2011\nrolling blackouts"),
              ("2011-08-03", "Aug 2011\nheat"),
              ("2014-01-06", "Jan 2014\npolar vortex"),
              ("2019-08-13", "Aug 2019\nscarcity"),
              ("2021-02-16", "Winter Storm Uri"),
              ("2022-12-23", "Elliott"),
              ("2023-08-17", "Aug 2023\nheat")]
    for ts, lab in events:
        t = pd.Timestamp(ts)
        win = sp.loc[t - pd.Timedelta(days=5): t + pd.Timedelta(days=5)]
        if not len(win):
            continue
        ax.annotate(lab, xy=(win.idxmax(), win.max()), xytext=(0, 14),
                    textcoords="offset points", fontsize=7.5, ha="center",
                    color="0.25",
                    arrowprops=dict(arrowstyle="-", lw=0.6, color="0.45"))
    ax.set_ylim(8, 20000)
    ax.set_ylabel("$/MWh (log scale)", fontsize=10)
    ax.set_title("ERCOT North Hub real-time on-peak daily average — 5×16 "
                 "(HE 7–22, weekdays excl. NERC holidays) · 2011–present",
                 fontsize=13, fontweight="bold", loc="left")
    ax.grid(alpha=0.3, lw=0.4, which="both")
    ax.legend(fontsize=8.5, loc="upper left")
    ax.tick_params(labelsize=9)
    fig.get_layout_engine().set(rect=(0, 0.045, 1, 1))
    fig.text(0.01, 0.008,
             f"source: ERCOT Historical RTM Load Zone and Hub Prices (NP6-785) "
             f"· 15-min HB_NORTH settlement point prices, HE 7–22 mean · "
             f"through {s.index.max():%Y-%m-%d}",
             fontsize=7.3, color="0.4")
    fig.savefig(ASSETS / "rt_onpeak.webp", dpi=125)
    plt.close(fig)
    print(f"saved {ASSETS / 'rt_onpeak.webp'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    build(refresh=args.refresh)


if __name__ == "__main__":
    main()
