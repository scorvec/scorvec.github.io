#!/usr/bin/env python3
"""
Crisp 3-page PDF briefing for the Brazil 2026/27 summer forecast.

  p1  Daily Niño-3.4 and relative Niño-3.4: 2026 vs 1997/98, 2015/16,
      2023/24 (OISST daily, base-adjusted to 1991–2020) + the C3S
      multi-model plumes from the site's enso_forecast.json.
  p2  Sep & Oct 2026 C3S 10-system MME maps (t2m + precip) over Brazil.
  p3  1997/98 recall: timeline + Brazil scoreboard + the 20CR monthly
      analog maps.

Output: ~/brazil_summer_briefing.pdf
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg
import cartopy.crs as ccrs
import cartopy.feature as cfeature

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, os.path.expanduser("~/c3s/scripts"))
import f4_lib as F                                           # noqa: E402

SST = HERE.parent / "sst" / "data"
AJ = HERE.parent.parent / "assets" / "sst" / "data"
ANALOGS = {1997: ("#d62728", "1997/98"), 2015: ("#1f77b4", "2015/16"),
           2023: ("#2ca02c", "2023/24")}
INK = "#1a2733"
EXT = (-76, -32, -35, 7)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                     "axes.edgecolor": "#444", "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": "#444",
                     "ytick.color": "#444"})


def box_daily(year0):
    """Daily n34 and rel series Jul(y0)–Apr(y0+1) from cached OISST
    day-anom files (1971–2000 base), corrected to the 1991–2020 base via
    ERSST climatology offsets."""
    parts = []
    for y in (year0, year0 + 1):
        f = SST / f"sst.day.anom.{y}.nc"
        ds = xr.open_dataset(f)
        a = ds["anom"].sortby("lat")
        if "zlev" in a.dims:
            a = a.squeeze("zlev")
        n34 = a.sel(lat=slice(-5, 5), lon=slice(190, 240))
        w = np.cos(np.deg2rad(n34["lat"]))
        n34 = n34.weighted(w).mean(("lat", "lon")).to_series()
        tr = a.sel(lat=slice(-20, 20))
        w = np.cos(np.deg2rad(tr["lat"]))
        tr = tr.weighted(w).mean(("lat", "lon"), skipna=True).to_series()
        parts.append(pd.DataFrame(dict(n34=n34, trop=tr)))
    df = pd.concat(parts)
    df = df.loc[f"{year0}-07-01":f"{year0+1}-04-30"]
    return df["n34"] - OFF_N34, (df["n34"] - OFF_N34) - (df["trop"] - OFF_TR)


def ersst_offsets():
    """Mean 1991–2020 minus 1971–2000 climatology for the two boxes."""
    ds = xr.open_dataset(SST / "ersst_v5_mnmean.nc")
    sst = ds["sst"].sortby("lat")

    def wm(da):
        w = np.cos(np.deg2rad(da["lat"]))
        return da.weighted(w).mean(("lat", "lon"), skipna=True)

    n34 = wm(sst.sel(lat=slice(-5, 5), lon=slice(190, 240))).to_series()
    tr = wm(sst.sel(lat=slice(-20, 20))).to_series()
    off = []
    for s in (n34, tr):
        off.append(float(s.loc["1991":"2020"].mean()
                         - s.loc["1971":"2000"].mean()))
    return off


OFF_N34, OFF_TR = ersst_offsets()


def xpos(ts, year0):
    return (pd.Timestamp(ts) - pd.Timestamp(year0, 7, 1)).days


def page1(pdf):
    dj = json.load(open(AJ / "enso_daily.json"))
    fj = json.load(open(AJ / "enso_forecast.json"))
    dd = dj["daily"]
    dates = pd.to_datetime(dd["dates"])
    cur = pd.DataFrame(dict(n34=dd["nino34"], rel=dd["rel"]),
                       index=dates).loc["2026-07-01":]
    lat = dj["latest"]

    fig, axes = plt.subplots(1, 2, figsize=(11.69, 6.6))
    fig.subplots_adjust(top=0.80, bottom=0.20, left=0.06, right=0.985,
                        wspace=0.16)
    fig.text(0.06, 0.955, "BRAZIL SUMMER 2026/27 — DESK BRIEFING",
             fontsize=15, fontweight="bold")
    fig.text(0.06, 0.915, "1 · ENSO: a 1997-class, east-based event, "
             "still intensifying", fontsize=10.5, color="#bd3a1c",
             fontweight="bold")
    fig.text(0.06, 0.878,
             f"OISST daily {lat['date']}:  Niño-3.4 {lat['nino34']:+.2f}   "
             f"ONI (Jul) {lat['oni']:+.2f}   RONI (Jul) {lat['roni']:+.2f}   "
             f"daily RONI est. {lat['roni_d']:+.2f}   "
             f"Niño-1+2 {lat['nino12']:+.2f}", fontsize=9, color="#444")

    for ax, col, ttl, ylab in (
            (axes[0], "n34", "Niño-3.4 anomaly — daily", "°C vs 1991–2020"),
            (axes[1], "rel", "Relative Niño-3.4 (minus tropical mean) — "
             "the RONI view", "°C vs 1991–2020")):
        for y0, (c, lb) in ANALOGS.items():
            n34s, rels = box_daily(y0)
            s = n34s if col == "n34" else rels
            xs = [xpos(t, y0) for t in s.index]
            ax.plot(xs, s.rolling(7, center=True).mean().values, color=c,
                    lw=1.1, alpha=0.9, label=lb)
        s = cur[col]
        xs = [xpos(t, 2026) for t in s.index]
        ax.plot(xs, s.rolling(7, center=True).mean().values, color="k",
                lw=2.4, label="2026 to date", zorder=5)
        ax.plot(xs[-1], s.rolling(7, center=True, min_periods=1)
                .mean().values[-1], "*", color="k", ms=13, zorder=6)

        key = "n34" if col == "n34" else "rnino"
        for m, mm in fj["models"].items():
            arr = np.array(mm[key])
            xc = [xpos(pd.Timestamp(v + "-15"), 2026)
                  for v in fj["valid_months"]]
            ax.plot(xc, np.median(arr, axis=1), color=mm["color"], lw=1.5,
                    ls="--", label=m)
            ax.fill_between(xc, np.percentile(arr, 10, axis=1),
                            np.percentile(arr, 90, axis=1),
                            color=mm["color"], alpha=0.05, lw=0)
        nd0, nd1 = xpos("2026-11-01", 2026), xpos("2027-04-01", 2026)
        ax.axvspan(nd0, nd1, color="#f5e9c8", alpha=0.55, lw=0, zorder=0)
        ax.text((nd0 + nd1) / 2, -1.62, "NDJFM target", ha="center",
                fontsize=7.5, color="#8a6d00")
        mt = [xpos(pd.Timestamp(2026 if m >= 7 else 2027, m, 1), 2026)
              for m in (7, 8, 9, 10, 11, 12, 1, 2, 3, 4)]
        ax.set_xticks(mt)
        ax.set_xticklabels(list("JASONDJFMA"))
        ax.set_xlim(0, 305)
        ax.set_ylim(-1.8, 4.6)
        ax.axhline(0, color="0.55", lw=0.6)
        ax.grid(lw=0.25, alpha=0.4)
        ax.set_title(ttl, loc="left", fontsize=10)
        ax.set_ylabel(ylab, fontsize=8.5)
    axes[0].legend(fontsize=6.4, ncol=3, frameon=False, loc="lower center",
                   bbox_to_anchor=(0.5, -0.27))
    axes[1].legend(fontsize=6.4, ncol=3, frameon=False, loc="lower center",
                   bbox_to_anchor=(0.5, -0.27))
    fig.text(0.06, 0.012,
             "Analog dailies: OISST v2.1, base-shifted to 1991–2020, 7-day "
             "smoothed. Plumes: C3S Aug-2026 members vs own hindcasts "
             "(median, 10–90%). NDJFM working assumption: RONI +2.75.",
             fontsize=7, color="#666")
    fig.text(0.985, 0.955, "scorvec.com · 24 Aug 2026 · p1/3", ha="right",
             fontsize=7, color="#666")
    pdf.savefig(fig)
    plt.close(fig)


def page2(pdf):
    mme = {("t2m", s): [] for s in (1, 2)} | {("tp", s): [] for s in (1, 2)}
    for mdl in F.models_present("sa_fc"):
        ds = F.load_sa(mdl)
        for v in ("t2m", "tp"):
            for s in (1, 2):
                a = (ds[f"fc_{v}"].isel(step=s).mean("number")
                     - ds[f"hc_{v}"].isel(step=s).mean("sample"))
                mme[(v, s)].append(
                    a.rename(latitude="lat", longitude="lon").sortby("lat"))
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(0.06, 0.955, "2 · SEPTEMBER / OCTOBER — THE PATTERN ARRIVES "
             "EARLY", fontsize=12, fontweight="bold", color="#bd3a1c")
    fig.text(0.06, 0.925,
             "C3S 10-system multi-model mean, Aug 2026 init, anomalies vs "
             "each system's own 1993–2016 hindcast climatology.",
             fontsize=9, color="#444")
    spec = [("t2m", 1, "Sep t2m", 3.0, "RdBu_r", "°C"),
            ("t2m", 2, "Oct t2m", 3.0, "RdBu_r", "°C"),
            ("tp", 1, "Sep precip", 2.0, "BrBG", "mm/day"),
            ("tp", 2, "Oct precip", 2.0, "BrBG", "mm/day")]
    for i, (v, s, ttl, vmax, cmap, unit) in enumerate(spec):
        ax = fig.add_subplot(2, 2, i + 1, projection=ccrs.PlateCarree())
        ax.set_extent(EXT, crs=ccrs.PlateCarree())
        g = mme[(v, s)][0]
        fld = np.nanmean([a.interp(lat=g["lat"], lon=g["lon"]).values
                          for a in mme[(v, s)]], axis=0)
        lv = np.linspace(-vmax, vmax, 21)
        cf = ax.contourf(g["lon"], g["lat"], fld, levels=lv, cmap=cmap,
                         extend="both", transform=ccrs.PlateCarree())
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.5,
                       edgecolor="#333")
        ax.add_feature(cfeature.STATES.with_scale("50m"), lw=0.2,
                       edgecolor="#888")
        ax.coastlines("50m", lw=0.5, color="#333")
        cb = fig.colorbar(cf, ax=ax, fraction=0.035, pad=0.02)
        cb.set_label(unit, fontsize=8)
        cb.ax.tick_params(labelsize=7)
        ax.set_title(ttl, loc="left", fontsize=10, fontweight="bold")
    fig.text(0.06, 0.045,
             "Read: early heat and drying over the N/NE (fire + hydrology "
             "stress), while above-normal rain reaches the Center-South\n"
             "cane belt from September — the crush-tail disruption threat "
             "starts two months before summer.",
             fontsize=8.5, color="#333")
    fig.text(0.985, 0.012, "scorvec.com · 24 Aug 2026 · p2/3", ha="right",
             fontsize=7, color="#666")
    fig.subplots_adjust(top=0.88, bottom=0.09, left=0.05, right=0.97,
                        hspace=0.18, wspace=0.05)
    pdf.savefig(fig)
    plt.close(fig)


def page3(pdf):
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(0.05, 0.955, "3 · 1997/98 — WHAT THE ANALOG ACTUALLY DID",
             fontsize=12, fontweight="bold", color="#bd3a1c")
    txt = (
        "THE EVENT\n"
        "  · Onset Mar–Apr 1997, explosive growth through boreal summer\n"
        "  · Peak Nov–Dec 1997: ONI +2.4, the east-lean record until now\n"
        "    (+2.82) — warm core hugging the Peru coast, like today\n"
        "  · Rapid collapse Apr–May 1998 → strong La Niña by July 1998\n"
        "\n"
        "WHAT BRAZIL GOT\n"
        "  · N / NE:  severe drought; the 1998 NE rainy season failed\n"
        "    outright — crop losses and water rationing in the sertão\n"
        "  · Amazon:  deep drought, record-low rivers; the Roraima fires\n"
        "    of early 1998 burned ~40,000 km² — a national emergency\n"
        "  · South:   the flip side — repeated flooding Oct 1997–Apr 1998\n"
        "    (Iguaçu/Paraná basins), bumper hydro inflows in the S system\n"
        "  · SE:      hot, dry-leaning summer; Jan–Feb 1998 heat waves in\n"
        "    Rio and São Paulo\n"
        "  · Coastal Peru/Ecuador: catastrophic floods — the EP signature\n"
        "\n"
        "WHY IT MATTERS NOW\n"
        "  · 2026 tracks 1997 on both axes (amplitude AND geometry) —\n"
        "    the maps at right are the closest observed template for\n"
        "    NDJFM 2026/27, before adding 29 years of warming (~+0.6 °C)\n"
        "  · Watch the 1998 rhyme into H1-2027: a fast post-peak collapse\n"
        "    toward La Niña is the standard exit from events this large"
    )
    fig.text(0.05, 0.885, txt, fontsize=8.6, va="top", family="DejaVu Sans",
             linespacing=1.45)
    img = mpimg.imread(os.path.expanduser("~/analog_1997_monthly.png"))
    nz = (img[:, :, :3].min(axis=2) < 0.97)
    rows, cols = np.where(nz.any(1))[0], np.where(nz.any(0))[0]
    img = img[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    ax = fig.add_axes((0.44, 0.05, 0.54, 0.85))
    ax.imshow(img)
    ax.axis("off")
    fig.text(0.71, 0.032, "20CRv3 detrended monthly anomalies, Nov 1997 – "
             "Apr 1998 (temp / precip)", ha="center", fontsize=7.5,
             color="#666")
    fig.text(0.985, 0.012, "scorvec.com · 24 Aug 2026 · p3/3", ha="right",
             fontsize=7, color="#666")
    pdf.savefig(fig)
    plt.close(fig)


def main() -> int:
    out = Path.home() / "brazil_summer_briefing.pdf"
    with PdfPages(out) as pdf:
        page1(pdf)
        page2(pdf)
        page3(pdf)
        info = pdf.infodict()
        info["Title"] = "Brazil Summer 2026/27 — Desk Briefing"
        info["Author"] = "Shawn Corvec / scorvec.com"
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
