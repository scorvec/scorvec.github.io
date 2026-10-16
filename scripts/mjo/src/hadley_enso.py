#!/usr/bin/env python3
"""The Hadley cells through El Niño: this season against every moderate-to-strong event since 1963.

An event is a July-March whose RONI (CPC's official ENSO index since February 2026, assets/sst/data/cpc_official.json)
peaks at EVENT_MIN or more; STRONG ones are coloured. For each, every day's cell intensity and poleward edges are read
off ERA5's centred 7-day-mean zonal-mean v - the statistic, the 1.5 deg bands and the data of the climatology
(build_hadley_clim; WeatherBench2 to January 2023, ARCO-ERA5 after) - and drawn beside this season's analysed 7-day means
(overturn_hist's archive, less the analysis offset) and the forecasts (overturn_fc, all ensembles' members pooled).
Every series sits at its window's MIDPOINT, so the centred event windows, the trailing analysed weeks and the forecast
windows line up. Each event is drawn on its own and no composite is formed: the figure places this season among the
events and claims nothing about El Niño's average effect.

    python src/hadley_enso.py list                      # the events and their fetch windows (JSON, for the workflow)
    python src/hadley_enso.py events part*.npz          # -> data/reference/hadley_events.nc
    python src/hadley_enso.py plot --out assets/sst/hadley_enso.webp
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
import build_hadley_clim as B
import overturn as O
import overturn_hist as H
from doycal import noleap_doy

REPO = Path(__file__).resolve().parents[3]
CPC = REPO / "assets" / "sst" / "data" / "cpc_official.json"
OUT = B.REF / "hadley_events.nc"
FC = REPO / "assets" / "sst" / "data" / "overturn_fc.json"  # overturn_fc.py
EVENT_MIN, STRONG = 1.0, 2.0  # RONI peak, July-March
WB2_END = pd.Timestamp("2023-01-06")  # WeatherBench2 ends 2023-01-10 18Z; windows reach 3 days past their day
HALF = B.WIN // 2
KEYS = ("nh_psi", "nh_edge_pole", "sh_psi", "sh_edge_pole")
PANELS = (
    ("nh_psi", "Northern cell intensity", "|Ψ|max  (10¹⁰ kg s⁻¹)"),
    ("nh_edge_pole", "Northern cell poleward edge", "latitude (°N)"),
    ("sh_psi", "Southern cell intensity", "|Ψ|max  (10¹⁰ kg s⁻¹)"),
    ("sh_edge_pole", "Southern cell poleward edge", "latitude (°S)"),
)
STRONG_C = ["#b5482a", "#d08a1e", "#2e7d32", "#1f6fa8", "#7b3fa0", "#c2185b"]


def season_days(y: int) -> pd.DatetimeIndex:
    """1 July of y to 31 March of y+1, without 29 February: 274 days on a common calendar."""
    d = pd.date_range(f"{y}-07-01", f"{y + 1}-03-31")
    return d[~((d.month == 2) & (d.day == 29))]


def events() -> list[dict]:
    roni = json.loads(CPC.read_text())["roni"]
    out = []
    for y in range(1959, pd.Timestamp.now("UTC").year):
        vals = [roni.get(f"{d:%Y-%m}") for d in pd.date_range(f"{y}-07-01", f"{y + 1}-03-01", freq="MS")]
        vals = [v for v in vals if v is not None]
        if len(vals) == 9 and max(vals) >= EVENT_MIN:
            start, end = (
                pd.Timestamp(f"{y}-07-01") - pd.Timedelta(days=HALF),
                pd.Timestamp(f"{y + 1}-03-31") + pd.Timedelta(days=HALF),
            )
            out.append(
                {
                    "year": y,
                    "label": f"{y}/{str(y + 1)[2:]}",
                    "peak": max(vals),
                    "start": f"{start:%Y-%m-%d}",
                    "end": f"{end:%Y-%m-%d}",
                    "source": "wb2" if end <= WB2_END else "arco",
                }
            )
    return out


def build_events(paths) -> xr.Dataset:
    """Every event's daily metrics (event, day of season) from the fetched daily zonal-mean v."""
    acc_d, acc_v = [], []
    for p in paths:
        with np.load(p) as f:
            acc_d.append(f["date"])
            acc_v.append(f["vbar"])
    date, first = np.unique(np.concatenate(acc_d), return_index=True)
    vbar = np.concatenate(acc_v)[first]
    pos = {d: i for i, d in enumerate(pd.DatetimeIndex(date))}
    evs = events()
    data = {k: np.full((len(evs), 274), np.nan) for k in KEYS}
    for e, ev in enumerate(evs):
        for j, d in enumerate(season_days(ev["year"])):
            w = [pos.get(d + pd.Timedelta(days=k)) for k in range(-HALF, HALF + 1)]
            if any(i is None for i in w):
                continue
            m = B.metrics(vbar[w].mean(0), B.LAT15)
            for k in KEYS:
                data[k][e, j] = m[k]
        print(f"  {ev['label']} (RONI {ev['peak']:+.1f}): {np.isfinite(data['nh_psi'][e]).sum()} days", flush=True)
    return xr.Dataset(
        {k: (("event", "day"), v) for k, v in data.items()},
        coords={
            "event": [ev["label"] for ev in evs],
            "peak": ("event", [ev["peak"] for ev in evs]),
            "day": np.arange(274),
        },
        attrs={
            "basis": "metrics of the centred 7-day-mean ERA5 zonal-mean v (build_hadley_clim.metrics), day 0 = 1 July",
            "events": f"July-March RONI peak >= {EVENT_MIN} (CPC official)",
        },
    )


def this_season(today: pd.Timestamp, metrics=lambda w: B.metrics(w[0], O.LAT), keys=KEYS):
    """(midpoint dates, metrics) of the analysed weeks since 1 July, and the pooled forecast (midpoints, {key: members}).
    `metrics` reads one overturn_hist.week; the default is the Hadley cells'."""
    doc, ref = H.load(), O.load_ref()
    y = today.year if today.month >= 7 else today.year - 1
    days = pd.DatetimeIndex(doc["date"])
    rows, mids = [], []
    for end in days[days >= pd.Timestamp(f"{y}-07-01") + pd.Timedelta(days=HALF)]:
        w = H.week(doc, end, ref)
        if w is not None:
            rows.append(metrics(w))
            mids.append(end - pd.Timedelta(days=HALF))
    obs = pd.DataFrame(rows, index=pd.DatetimeIndex(mids))
    fc = json.loads(FC.read_text()) if FC.exists() else None
    if fc and any(k not in d for d in fc["models"].values() for k in keys):
        fc = None  # written before these metrics were
    if fc and len(obs) and pd.Timestamp(fc["init"][:10]) >= obs.index[-1]:
        fmid = pd.DatetimeIndex(fc["valid"]) - pd.Timedelta(days=HALF)
        members = {k: np.concatenate([np.array(d[k], float) for d in fc["models"].values()]) for k in keys}
        return y, obs, (fmid, members, fc["init"][:10])
    return y, obs, None


def plume(v: np.ndarray) -> dict:
    """The 10th, 50th and 90th percentiles over members at each lead where most members give a value."""
    ok = np.isfinite(v).sum(0) >= 0.5 * v.shape[0]
    return {p: np.where(ok, np.nanpercentile(v, p, axis=0), np.nan) for p in (10, 50, 90)}


def limits(ax, events: np.ndarray, *shown):
    """Robust limits from the events - a day whose edge the cell finder misplaces (a cell too weak to bound) must not
    squash the rest - widened to hold this season's analysis and forecast in full."""
    lo, hi = np.nanpercentile(events, [0.5, 99.5])
    s = np.concatenate([np.ravel(v) for v in shown]) if shown else np.array([])
    if np.isfinite(s).any():
        lo, hi = min(lo, np.nanmin(s)), max(hi, np.nanmax(s))
    pad = 0.08 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)


def clim_at(cl: xr.Dataset, key: str, doy, stat: str) -> np.ndarray:
    """A climatological stat at common-year days (doycal.noleap_doy; 29 Feb = 59.5 takes its two neighbours' mean)."""
    lo = cl[key].sel(stat=stat, doy=xr.DataArray(np.floor(doy).astype(int)))
    hi = cl[key].sel(stat=stat, doy=xr.DataArray(np.ceil(doy).astype(int)))
    return (0.5 * (lo + hi)).values


def finish(fig, axes, title: str, foot: str, out: Path):
    """Title, the shared legend and the footnote at their measured heights (no dead space), then save."""
    inv, r = fig.transFigure.inverted(), fig.canvas.get_renderer()
    st = fig.suptitle(title, fontsize=13, fontweight="bold", x=0.005, y=0.995, ha="left", va="top")
    st.set_in_layout(False)  # placed by hand above the legend
    hs, ls = np.ravel(axes)[0].get_legend_handles_labels()
    below = inv.transform(st.get_window_extent(r))[0, 1] - 0.01
    leg = fig.legend(hs, ls, fontsize=8, ncol=4, loc="upper left", bbox_to_anchor=(0.004, below), frameon=False)
    ft = fig.text(0.005, 0.005, foot, fontsize=7.8, color="0.35", ha="left", va="bottom")
    top = inv.transform(leg.get_window_extent(r))[0, 1]
    bottom = inv.transform(ft.get_window_extent(r))[1, 1]
    fig.tight_layout(pad=0.4, rect=(0, bottom + 0.005, 1, top - 0.005))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"wrote {out}")


def plot(out: Path):
    ev = xr.open_dataset(OUT)
    cl = xr.open_dataset(B.OUT)
    y, obs, fc = this_season(pd.Timestamp.now("UTC").tz_localize(None))
    x = season_days(y)
    doy = noleap_doy(x)
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 7.4), sharex=True)
    strong = [e for e in ev.event.values if float(ev.peak.sel(event=e)) >= STRONG]
    for ax, (key, title, ylab) in zip(axes.ravel(), PANELS):
        flip = -1.0 if key.startswith("sh") else 1.0
        ax.fill_between(
            x,
            np.minimum(clim_at(cl, key, doy, "p10") * flip, clim_at(cl, key, doy, "p90") * flip),
            np.maximum(clim_at(cl, key, doy, "p10") * flip, clim_at(cl, key, doy, "p90") * flip),
            color="#e3e9ef",
            lw=0,
            label="ERA5 1991–2020, 10–90 %",
        )
        ax.plot(x, clim_at(cl, key, doy, "p50") * flip, color="#8a5a00", lw=1.3, ls=(0, (5, 2)), label="ERA5 median")
        for e in ev.event.values:
            if e not in strong:
                ax.plot(
                    x,
                    ev[key].sel(event=e).values * flip,
                    color="#9aa3ad",
                    lw=0.7,
                    alpha=0.45,
                    label="moderate El Niño (RONI +1.0 to +2.0)" if e == ev.event.values[0] else None,
                )
        for c, e in zip(STRONG_C, strong):
            ax.plot(
                x,
                ev[key].sel(event=e).values * flip,
                color=c,
                lw=1.4,
                label=f"{e} (RONI {float(ev.peak.sel(event=e)):+.1f})",
            )
        if len(obs):
            ax.plot(obs.index, obs[key].values * flip, color="#111", lw=2.6, label=f"{y}/{str(y + 1)[2:]} analysis")
        shown = [obs[key].values * flip] if len(obs) else []
        if fc:
            fmid, mem, init = fc
            q = plume(mem[key] * flip)
            shown += [q[10], q[90]]
            ax.fill_between(fmid, q[10], q[90], color="#111", alpha=0.12, lw=0)
            ax.plot(
                fmid,
                q[50],
                color="#111",
                lw=1.8,
                ls=(0, (3, 1.5)),
                label=f"forecast {init} (3 ensembles, median and 10–90 %)",
            )
        limits(ax, ev[key].values * flip, *shown)
        ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
        ax.set_ylabel(ylab, fontsize=9)
        ax.grid(alpha=0.25, lw=0.6)
        ax.tick_params(labelsize=8.5)
        if key.endswith("edge_pole"):
            ax.yaxis.set_major_formatter(lambda v_, _: f"{abs(v_):.0f}°")
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
    finish(
        fig,
        axes,
        "Hadley cells through El Niño: this season against every El Niño since 1963 (ERA5)",
        "7-day means at 00 UTC, at their midpoint · events: July–March RONI peak ≥ +1.0 (CPC) · intensity = peak |Ψ| in "
        "the cell, edge = Ψ(300–700 hPa)=0 on the descending flank, all on 1.5° bands\nthis season: analysis less its mean "
        "offset from ERA5; forecasts: members less their model's drift against ERA5 · each event on its own "
        "(no composite) · 1991/92 also carries Pinatubo",
        out,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    e = sub.add_parser("events")
    e.add_argument("parts", nargs="+")
    p = sub.add_parser("plot")
    p.add_argument("--out", default="assets/sst/hadley_enso.webp")
    a = ap.parse_args()
    if a.cmd == "list":
        print(json.dumps(events()))
    elif a.cmd == "events":
        ds = build_events(a.parts)
        ds.to_netcdf(OUT)
        print(f"wrote {OUT}")
    else:
        plot(Path(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
