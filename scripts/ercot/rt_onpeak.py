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
HUBS = ("HB_NORTH", "HB_WEST", "HB_HOUSTON", "HB_SOUTH", "HB_HUBAVG",
        "LZ_WEST")
REN = {"HB_NORTH": "north", "HB_WEST": "west", "HB_HOUSTON": "houston",
       "HB_SOUTH": "south", "HB_HUBAVG": "hubavg", "LZ_WEST": "lzwest"}
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
         .rename(columns=REN).reset_index())
    d.to_csv(cache, index=False)
    print(f"{yr}: parsed {len(d)} days", flush=True)
    return d


FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
COAL_MMBTU_PER_TONNE = 25.1        # 6,000 kcal/kg seaborne benchmark


def fetch_fuels():
    """Daily Henry Hub spot + monthly seaborne thermal coal (FRED, keyless),
    cached with a 3-day staleness window."""
    import time
    out = {}
    for sid, name in (("DHHNGSP", "gas"), ("PCOALAUUSDM", "coal")):
        f = DATA / f"fred_{sid}.csv"
        if not (f.exists() and time.time() - f.stat().st_mtime < 3 * 86400):
            # FRED stalls python-requests but answers curl instantly
            import subprocess
            rc = subprocess.run(["curl", "-s", "-m", "90", "-o", str(f),
                                 FRED.format(sid)], check=False).returncode
            if (rc != 0 or not f.exists() or f.stat().st_size < 100) \
                    and not f.exists():
                raise RuntimeError(f"FRED {sid} fetch failed (curl rc={rc})")
        df = pd.read_csv(f, na_values=".")
        df.columns = ["date", "v"]
        df["date"] = pd.to_datetime(df["date"])
        out[name] = df.dropna().set_index("date")["v"]
    out["coal"] = out["coal"] / COAL_MMBTU_PER_TONNE      # $/tonne -> $/MMBtu
    return out


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

    # compact JSON feed for the interactive page
    d5 = d.loc[d.is_5x16]
    def ser(v):
        return [None if (x is None or (isinstance(x, float) and np.isnan(x)))
                else round(float(x), 2) for x in v]
    rmean = d5["north"].rolling(30, center=True, min_periods=10).mean()
    feed = {"updated": f"{d.index.max():%Y-%m-%d}",
            "dates": [f"{t:%Y-%m-%d}" for t in d5.index],
            "north": ser(d5["north"]), "mean30": ser(rmean)}
    for k, tag in (("west", "wn"), ("houston", "hn"), ("south", "sn"),
                   ("lzwest", "lzn")):
        if k in d5:
            feed[k] = ser(d5[k])
            sp = d5[k] - d5["north"]
            feed[tag + "_m30"] = ser(sp.rolling(30, center=True,
                                                min_periods=10).mean())
    fuels = fetch_fuels()
    gas = fuels["gas"].reindex(
        pd.date_range(fuels["gas"].index.min(), d.index.max())).ffill(limit=5)
    hr = (d5["north"] / gas.reindex(d5.index)).replace(
        [np.inf, -np.inf], np.nan)
    hr_m30 = hr.rolling(30, center=True, min_periods=10).mean()
    feed["gas"] = ser(gas.reindex(d5.index))
    feed["hr"] = ser(hr)
    feed["hr_m30"] = ser(hr_m30)
    coal = fuels["coal"][fuels["coal"].index >= "2011-01-01"]
    feed["coal_dates"] = [f"{t:%Y-%m-%d}" for t in coal.index]
    feed["coal"] = ser(coal)
    (ASSETS / "rt_onpeak.json").write_text(json.dumps(feed,
                                                      separators=(",", ":")))

    chart(d)
    fuels_chart(d5, gas, hr, hr_m30, coal)
    print(f"series: {len(d)} days -> {d.index.max():%Y-%m-%d}")


def chart(d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = d.loc[d.is_5x16, "north"].dropna()
    sp = s
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(14.2, 9.4),
                                  constrained_layout=True,
                                  height_ratios=[1.35, 1])
    # symlog: linear through zero so the solar-era negative days are honest,
    # logarithmic where the scarcity tail lives
    ax.set_yscale("symlog", linthresh=10, linscale=0.5)
    ax.plot(sp.index, sp.values, lw=0.55, color="#2e5d9e", alpha=0.85,
            label="daily on-peak average (HB_NORTH)")
    rmean = sp.rolling(30, center=True, min_periods=10).mean()
    ax.plot(rmean.index, rmean.values, lw=1.8, color="#c62828",
            label="30-day rolling mean")
    ax.axhline(0, color="0.6", lw=0.6)
    events = [("2011-02-02", "Feb 2011\nrolling blackouts"),
              ("2011-08-03", "Aug 2011\nheat"),
              ("2014-01-06", "Jan 2014\npolar vortex"),
              ("2019-08-13", "Aug 2019\nscarcity"),
              ("2021-02-16", "Winter Storm Uri"),
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
    ax.set_ylim(-15, 20000)
    ax.set_yticks([-10, 0, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000])
    ax.set_yticklabels(["−10", "0", "10", "25", "50", "100", "250", "500",
                        "1,000", "2,500", "5,000", "10,000"])
    ax.minorticks_off()
    ax.set_ylabel("$/MWh (symlog)", fontsize=10)
    ax.set_title("ERCOT North Hub real-time on-peak daily average — 5×16 "
                 "(HE 7–22, weekdays excl. NERC holidays) · 2011–present",
                 fontsize=13, fontweight="bold", loc="left")
    ax.grid(alpha=0.3, lw=0.4, which="both")
    ax.legend(fontsize=8.5, loc="upper left")
    ax.tick_params(labelsize=9)
    # ---- hub spreads vs North: the West wind-congestion story ----
    d5 = d.loc[d.is_5x16]
    ax2.set_yscale("symlog", linthresh=5, linscale=0.5)
    wn = (d5["lzwest"] - d5["north"]).dropna()
    ax2.plot(wn.index, wn.values, lw=0.45, color="#b8860b", alpha=0.4,
             label="LZ_WEST − North daily")
    for k, col, lab in (("lzwest", "#b8860b", "LZ_WEST − North"),
                        ("west", "#8d6e63", "HB_WEST − North"),
                        ("houston", "#00796b", "Houston − North"),
                        ("south", "#6a4fa3", "South − North")):
        spread = (d5[k] - d5["north"]).rolling(30, center=True,
                                               min_periods=10).mean()
        ax2.plot(spread.index, spread.values, lw=1.6, color=col,
                 label=f"{lab} · 30-d mean")
    ax2.axhline(0, color="0.55", lw=0.7)
    ax2.set_ylim(-1500, 1500)
    ax2.set_yticks([-1000, -250, -50, -10, 0, 10, 50, 250, 1000])
    ax2.set_yticklabels(["−1,000", "−250", "−50", "−10", "0", "10", "50",
                         "250", "1,000"])
    ax2.minorticks_off()
    ax2.set_ylabel("spread ($/MWh, symlog)", fontsize=10)
    ax2.set_title("West basis vs North — the load zone carries the story: wind "
                  "congestion, the CREZ collapse, and the Permian-era pocket",
                  fontsize=11.5, fontweight="bold", loc="left")
    ax2.grid(alpha=0.3, lw=0.4, which="major")
    ax2.legend(fontsize=8, loc="upper right", ncols=2)
    ax2.tick_params(labelsize=9)
    fig.get_layout_engine().set(rect=(0, 0.03, 1, 0.97))
    fig.text(0.01, 0.006,
             f"source: ERCOT Historical RTM Load Zone and Hub Prices (NP6-785) "
             f"· 15-min HB_NORTH settlement point prices, HE 7–22 mean · linear below \\$10, log above · "
             f"through {s.index.max():%Y-%m-%d}",
             fontsize=7.3, color="0.4")
    fig.savefig(ASSETS / "rt_onpeak.webp", dpi=125)
    plt.close(fig)
    print(f"saved {ASSETS / 'rt_onpeak.webp'}")


def fuels_chart(d5, gas, hr, hr_m30, coal):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(14.2, 9.0),
                                  constrained_layout=True)
    ax.set_yscale("symlog", linthresh=10, linscale=0.6)
    ax.plot(hr.index, hr.values, lw=0.5, color="#5e35b1", alpha=0.7,
            label="daily implied heat rate (North on-peak ÷ Henry Hub)")
    ax.plot(hr_m30.index, hr_m30.values, lw=1.8, color="#c62828",
            label="30-day rolling mean")
    ax.axhspan(6.5, 10.5, color="0.88", alpha=0.5, zorder=0,
               label="typical CCGT–peaker band")
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_ylim(-12, 4000)
    ax.set_yticks([-10, 0, 5, 10, 15, 25, 50, 100, 250, 1000])
    ax.set_yticklabels(["−10", "0", "5", "10", "15", "25", "50", "100",
                        "250", "1,000"])
    ax.minorticks_off()
    ax.set_ylabel("MMBtu/MWh (symlog)", fontsize=10)
    ax.set_title("Implied market heat rate — fuel-cost moves flatten out; "
                 "scarcity stands alone", fontsize=11.5, fontweight="bold",
                 loc="left")
    ax.grid(alpha=0.3, lw=0.4)
    ax.legend(fontsize=8, loc="upper left")
    ax.tick_params(labelsize=9)

    g = gas[gas.index >= "2011-01-01"].dropna()
    ax2.set_yscale("log")
    ax2.plot(g.index, g.values, lw=0.8, color="#1565c0",
             label="Henry Hub daily spot")
    ax2.plot(coal.index, coal.values, lw=1.6, color="#4e342e",
             drawstyle="steps-post",
             label="seaborne thermal coal (monthly, 6,000 kcal benchmark)")
    ax2.set_ylim(0.8, 30)
    ax2.set_yticks([1, 2, 3, 5, 7, 10, 15, 25])
    ax2.set_yticklabels(["1", "2", "3", "5", "7", "10", "15", "25"])
    ax2.minorticks_off()
    ax2.set_ylabel("$/MMBtu (log)", fontsize=10)
    ax2.set_title("The fuels underneath — gas sets the marginal price; "
                  "2022 is a fuel story, not a scarcity story",
                  fontsize=11.5, fontweight="bold", loc="left")
    ax2.grid(alpha=0.3, lw=0.4, which="major")
    ax2.legend(fontsize=8, loc="upper left")
    ax2.tick_params(labelsize=9)
    fig.get_layout_engine().set(rect=(0, 0.03, 1, 0.97))
    fig.text(0.01, 0.006,
             "heat rate = HB_NORTH 5×16 daily on-peak ÷ Henry Hub daily spot "
             "(FRED DHHNGSP) · coal: IMF seaborne benchmark (FRED PCOALAUUSDM) "
             f"÷ {COAL_MMBTU_PER_TONNE} MMBtu/tonne · not PRB delivered cost",
             fontsize=7.3, color="0.4")
    fig.savefig(ASSETS / "rt_fuels.webp", dpi=125)
    plt.close(fig)
    print(f"saved {ASSETS / 'rt_fuels.webp'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    build(refresh=args.refresh)


if __name__ == "__main__":
    main()
