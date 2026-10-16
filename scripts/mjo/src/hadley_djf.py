#!/usr/bin/env python3
"""The northern Hadley cell each winter against ENSO's strength and flavour: every DJF since 1959/60 in ERA5.

For each winter, the DJF mean of the daily cell metrics of ERA5's centred 7-day-mean zonal-mean v at 00 UTC - the
climatology's statistic (build_hadley_clim.metrics, on its 1.5 deg bands; WeatherBench2 to January 2023, ARCO-ERA5
after) - less the 1991-2020 climatology at each day, against the winter's DJF RONI (CPC's official index) and its DJF E
and C indices (Takahashi et al. 2011: the eastern- and central-Pacific modes, from ERSSTv6 on the site's own EOFs,
c3s_enso_flavour). The cells' edges drift poleward over the decades, so each relation is a partial correlation on the
winter's year (the linear trend removed from both series), tested on the effective sample size of two lag-1
autocorrelated series (Bretherton et al. 1999) and Holm-adjusted across the figure's tests; a fit is drawn only where
the adjusted p is below ALPHA, and a panel without one says so.

    python src/hadley_djf.py plan                                     # the fetch parts (JSON, for hadley-djf.yml)
    python src/hadley_djf.py fetch --windows 1959-11-28:1960-03-03:wb2,... --out part.npz
    python src/hadley_djf.py build part*.npz                          # -> data/reference/hadley_djf.nc
    python src/hadley_djf.py plot --out assets/sst/hadley_djf_enso.webp
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
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
import build_hadley_clim as B
import hadley_enso as E
from doycal import noleap_doy

sys.path.insert(0, str(E.REPO / "scripts" / "sst"))
import c3s_enso_flavour as FL

OUT = B.REF / "hadley_djf.nc"
FIRST = 1959  # DJF 1959/60, WeatherBench2's first winter
KEYS = ("nh_edge_pole", "nh_psi")
HALF = E.HALF
ALPHA = 0.05
VOLCANIC = {1963: "Agung", 1982: "El Chichón", 1991: "Pinatubo"}  # winters after a major tropical eruption
PANELS = (
    ("nh_edge_pole", "Northern cell poleward edge", "DJF anomaly (° latitude; − = equatorward)"),
    ("nh_psi", "Northern cell intensity", "DJF anomaly of |Ψ|max (10¹⁰ kg s⁻¹)"),
)
INDICES = (("roni", "DJF RONI (CPC)"), ("E", "DJF E index (eastern-Pacific mode)"))


def window(y: int) -> tuple[pd.Timestamp, pd.Timestamp, str]:
    """The days winter y (December of y) needs: DJF and the margins its centred windows reach into."""
    t0 = pd.Timestamp(f"{y}-12-01") - pd.Timedelta(days=HALF)
    t1 = pd.Timestamp(f"{y + 1}-03-01") - pd.Timedelta(days=1) + pd.Timedelta(days=HALF)
    return t0, t1, "wb2" if t1 <= E.WB2_END else "arco"


def winters() -> list[int]:
    """Every winter from FIRST whose March has passed by three months (ARCO-ERA5 publishes with a lag)."""
    now = pd.Timestamp.now("UTC").tz_localize(None)
    return [y for y in range(FIRST, now.year) if window(y)[1] + pd.Timedelta(days=90) < now]


def plan() -> list[dict]:
    """WeatherBench2 winters ten to a part; each ARCO winter alone (it reads ~120 MB a day)."""
    wins = [(y, *window(y)) for y in winters()]
    wb2 = [w for w in wins if w[3] == "wb2"]
    groups = [wb2[i : i + 10] for i in range(0, len(wb2), 10)] + [[w] for w in wins if w[3] == "arco"]
    return [
        {"name": f"djf{g[0][0]}", "windows": ",".join(f"{t0:%Y-%m-%d}:{t1:%Y-%m-%d}:{src}" for _, t0, t1, src in g)}
        for g in groups
    ]


def fetch(windows: str) -> dict:
    days, rows = [], []
    for w in windows.split(","):
        t0, t1, src = w.split(":")
        doc = (B.fetch_arco if src == "arco" else B.fetch_wb2)(pd.Timestamp(t0), pd.Timestamp(t1))
        days.append(doc["date"])
        rows.append(doc["vbar"])
    return {"date": np.concatenate(days), "vbar": np.concatenate(rows)}


def build(paths) -> xr.Dataset:
    """Each winter's DJF-mean anomalies of the four metrics, its DJF RONI and its DJF E and C."""
    acc_d, acc_v = [], []
    for p in paths:
        with np.load(p) as f:
            acc_d.append(f["date"])
            acc_v.append(f["vbar"])
    date, first = np.unique(np.concatenate(acc_d), return_index=True)
    vbar = np.concatenate(acc_v)[first]
    pos = {d: i for i, d in enumerate(pd.DatetimeIndex(date))}
    cl = xr.open_dataset(B.OUT)
    ys = winters()
    anom = {k: np.full(len(ys), np.nan) for k in E.KEYS}
    ndays = np.zeros(len(ys), int)
    for n, y in enumerate(ys):
        djf = pd.date_range(f"{y}-12-01", pd.Timestamp(f"{y + 1}-03-01") - pd.Timedelta(days=1))
        rows, days = [], []
        for d in djf:
            w = [pos.get(d + pd.Timedelta(days=k)) for k in range(-HALF, HALF + 1)]
            if all(i is not None for i in w):
                rows.append(B.metrics(vbar[w].mean(0), B.LAT15))
                days.append(d)
        ndays[n] = len(days)
        if not days:
            continue
        doy = noleap_doy(pd.DatetimeIndex(days))
        for k in E.KEYS:
            v = np.array([r[k] for r in rows], float) - E.clim_at(cl, k, doy, "mean")
            if np.isfinite(v).sum() >= 0.8 * djf.size:  # a cell the finder cannot bound on a few days is skipped
                anom[k][n] = np.nanmean(v)
        print(f"  {y}/{str(y + 1)[2:]}: {ndays[n]} days, NH edge {anom['nh_edge_pole'][n]:+.2f}", flush=True)
    roni = json.loads(E.CPC.read_text())["roni"]  # keyed by the season's centre month: DJF = January
    ec = FL.obs_seasons(FL.load_obs(), FL.load_eof_reference()[1])["DJF"]  # indexed by the December's year
    return xr.Dataset(
        {
            **{k: ("winter", v) for k, v in anom.items()},
            "n_days": ("winter", ndays),
            "roni": ("winter", [roni.get(f"{y + 1}-01", np.nan) for y in ys]),
            "E": ("winter", [ec["E"].get(y, np.nan) for y in ys]),
            "C": ("winter", [ec["C"].get(y, np.nan) for y in ys]),
        },
        coords={"winter": ys},
        attrs={
            "basis": "DJF means of the daily metrics of ERA5's centred 7-day-mean zonal-mean v at 00 UTC, less the "
            "1991-2020 climatology (build_hadley_clim) at each day; winter = the December's year",
            "indices": "roni: CPC official DJF RONI; E, C: Takahashi et al. (2011) DJF indices from ERSSTv6 "
            "(c3s_enso_flavour)",
        },
    )


def partial(x: np.ndarray, y: np.ndarray, year: np.ndarray) -> dict:
    """Partial correlation of x and y on the year, its effective sample size and two-sided p, and the fit."""
    ok = np.isfinite(x) & np.isfinite(y)
    X = np.column_stack([np.ones(ok.sum()), year[ok]])

    def res(v):
        return v - X @ np.linalg.lstsq(X, v, rcond=None)[0]

    rx, ry = res(x[ok]), res(y[ok])
    r = float(np.corrcoef(rx, ry)[0, 1])
    a1 = [float(np.corrcoef(s[:-1], s[1:])[0, 1]) for s in (rx, ry)]  # consecutive winters
    n = int(ok.sum())
    n_eff = min(n, n * (1 - a1[0] * a1[1]) / (1 + a1[0] * a1[1]))
    t = r * np.sqrt((n_eff - 3) / (1 - r**2))  # one degree of freedom spent on the year
    return dict(r=r, n=n, n_eff=n_eff, p=float(2 * stats.t.sf(abs(t), n_eff - 3)))


def holm(ps: list[float]) -> list[float]:
    order = np.argsort(ps)
    adj, run = np.empty(len(ps)), 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (len(ps) - i) * ps[j]))
        adj[j] = run
    return list(adj)


def plot(out: Path):
    ds = xr.open_dataset(OUT)
    year = ds.winter.values.astype(float)
    tests = {(k, xk): partial(ds[xk].values, ds[k].values, year) for k, *_ in PANELS for xk, _ in INDICES}
    for t, p in zip(tests.values(), holm([t["p"] for t in tests.values()])):
        t["p_holm"] = p
    fig, axes = plt.subplots(len(PANELS), len(INDICES), figsize=(11.6, 8.4))
    for row, (k, title, ylab) in zip(axes, PANELS):
        X = np.column_stack([np.ones_like(year), year])
        ok = np.isfinite(ds[k].values)
        trend = np.full_like(year, np.nan)
        trend[ok] = X[ok] @ np.linalg.lstsq(X[ok], ds[k].values[ok], rcond=None)[0]
        y = ds[k].values - trend  # the trend removed, as in the test
        for ax, (xk, xlab) in zip(row, INDICES):
            x = ds[xk].values
            first = ax is axes.flat[0]
            volc = np.isin(ds.winter.values, list(VOLCANIC))
            for c, m, lab in (
                ("#b5482a", ds.roni.values >= 0.5, "El Niño winter (DJF RONI ≥ +0.5)"),
                ("#1f6fa8", ds.roni.values <= -0.5, "La Niña winter (≤ −0.5)"),
                ("#8f969c", np.abs(ds.roni.values) < 0.5, "neutral"),
            ):
                kw = dict(s=32, zorder=3, label=lab if first else None)
                ax.scatter(x[m & ~volc], y[m & ~volc], color=c, edgecolor="white", linewidth=0.5, **kw)
                ax.scatter(x[m & volc], y[m & volc], facecolor="none", edgecolor=c, linewidth=1.4, s=34, zorder=3)
            if first:  # the hollow markers' key
                ax.plot(
                    [], [], ls="", marker="o", mfc="none", mec="0.3", ms=6, label="after Agung, El Chichón or Pinatubo"
                )
            x_hi = np.nanpercentile(x, 80)
            for w, xi, yi, ri in zip(ds.winter.values, x, y, ds.roni.values):
                if abs(ri) >= 1.8 and np.isfinite(xi + yi):
                    left = xi > x_hi  # a label at the right edge points inwards
                    ax.annotate(
                        f"{w % 100:02d}/{(w + 1) % 100:02d}",
                        (xi, yi),
                        xytext=(-5 if left else 5, 3),
                        textcoords="offset points",
                        ha="right" if left else "left",
                        fontsize=7.5,
                        color="0.3",
                    )
            t = tests[(k, xk)]
            sig = t["p_holm"] < ALPHA
            if sig:  # the least-squares line of what is drawn
                ok = np.isfinite(x) & np.isfinite(y)
                b1, b0 = np.polyfit(x[ok], y[ok], 1)
                xs = np.array([x[ok].min(), x[ok].max()])
                ax.plot(xs, b0 + b1 * xs, color="#111", lw=1.6)
            stat = f"r = {t['r']:+.2f}, p = {t['p_holm']:.3f} (Holm), n = {t['n']} winters, effective {t['n_eff']:.0f}"
            ax.text(
                0.02,
                0.97,
                ("significant · " if sig else "not significant · ") + stat,
                transform=ax.transAxes,
                fontsize=8.2,
                va="top",
                fontweight="bold" if sig else "normal",
                color="#111" if sig else "#6f6b64",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.5),
                zorder=5,
            )
            ax.axhline(0, color="0.75", lw=0.8)
            ax.axvline(0, color="0.75", lw=0.8)
            ax.set_title(title, fontsize=10.5, fontweight="bold", loc="left")
            ax.set_xlabel(xlab, fontsize=9)
            ax.set_ylabel(ylab, fontsize=9)
            ax.grid(alpha=0.25, lw=0.6)
            ax.tick_params(labelsize=8.5)
    w0, w1 = int(ds.winter.min()), int(ds.winter.max())
    E.finish(
        fig,
        axes,
        f"Northern Hadley cell in winter against ENSO's strength and flavour, DJF {w0}/{str(w0 + 1)[2:]}–{w1}/{str(w1 + 1)[2:]} (ERA5)",
        "each point one winter: DJF mean of the 7-day-mean cells at 00 UTC, less ERA5's 1991–2020 climatology, with the "
        "linear trend over the winters removed · E index: Takahashi et al. (2011), ERSSTv6\npartial correlation on the "
        "year, effective sample size from the lag-1 autocorrelation of both series, Holm-adjusted across the four panels; "
        "a line is drawn only where significant (p < 0.05) · labelled: |RONI| ≥ 1.8",
        out,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    f = sub.add_parser("fetch")
    f.add_argument("--windows", required=True)
    f.add_argument("--out", required=True)
    b = sub.add_parser("build")
    b.add_argument("parts", nargs="+")
    p = sub.add_parser("plot")
    p.add_argument("--out", default="assets/sst/hadley_djf_enso.webp")
    a = ap.parse_args()
    if a.cmd == "plan":
        print(json.dumps(plan()))
    elif a.cmd == "fetch":
        np.savez_compressed(a.out, **fetch(a.windows))
    elif a.cmd == "build":
        ds = build(a.parts)
        ds.to_netcdf(OUT)
        print(f"wrote {OUT}: {ds.winter.size} winters")
    else:
        plot(Path(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
