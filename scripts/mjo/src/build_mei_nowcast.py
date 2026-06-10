#!/usr/bin/env python3
"""Daily nowcast of NOAA PSL's bimonthly MEI.v2.

The official MEI.v2 is the leading combined-EOF PC of five tropical-Pacific fields
(SLP, SST, surface u & v wind, OLR), published only as an overlapping *bimonthly*
series with a ~5-week lag. PSL does not publish the EOF loadings, so we cannot project
onto the official pattern. Instead we REGRESS the published bimonthly MEI.v2 onto a set
of freely-available daily ENSO component indices, then drive that regression daily.

Predictors (raw physical units — OLS is scale-invariant, so coefficients learned on
bimonthly means apply directly to daily trailing means of the same quantity):
  SST  — Nino1+2 / 3 / 4 / 3.4 anomalies   (CPC ERSST5 monthly for the fit;
                                             OISST daily-anom boxes for the live tail)
  SLP  — SOI                               (DailySOI.txt: monthly mean for fit, daily for apply)
  OLR  — central-Pacific OLR anomaly box   (PSL interp-OLR: monthly for fit, daily for apply)
  wind — eq. 850-hPa zonal-wind anomaly box (ERA5 eq-band series for fit, 2026 ARCO for apply)

Fit window 1999-2022 (limited by daily-SOI start and monthly-OLR end); the satellite
era fits MEI.v2 best (R ~ 0.945). Two daily curves are produced: a faithful 30-day
trailing mean (hugs the official value) and a responsive ~14-day mean (leading edge).

    python src/build_mei_nowcast.py --fit --apply 2026
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                   # scripts/mjo
REF = ROOT / "data" / "reference"
MEI_IN = ROOT / "data" / "mei"                       # cached external inputs
CACHE = Path.home() / "mjo" / "era5_cache"
OISST = ROOT.parent / "sst" / "data"                 # repo's live OISST cache (scripts/sst/data),
#                                                      refreshed daily; ~/sst/data was a dead dev mirror
OUT = ROOT.parent.parent / "assets" / "sst" / "mei"  # repo/assets/sst/mei
FIT_JSON = REF / "mei_fit.json"

URLS = {
    "meiv2.data": "https://psl.noaa.gov/enso/mei/data/meiv2.data",
    "nino.ascii": "https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii",
    "olr.mon.mean.nc": "https://downloads.psl.noaa.gov/Datasets/interp_OLR/olr.mon.mean.nc",
}
OLR_DAY = CACHE / "olr.day.mean.nc"                  # ~350 MB, downloaded separately

# Nino SST boxes (lat0,lat1, lon0,lon1) in 0-360
BOXES = {"n12": (-10, 0, 270, 280), "n3": (-5, 5, 210, 270),
         "n4": (-5, 5, 160, 210), "n34": (-5, 5, 190, 240)}
OLR_BOX = (-5, 5, 160, 200)        # central-Pacific convection (PSL interp-OLR; discontinued 2022)
U850_LON = (135, 180)              # west-central Pacific westerly box (eq band already 5S-5N)
# Lean, stable predictor set — one box per MEI component we can source daily and currently:
#   SST = Niño3.4, SLP = SOI, wind = eq-u850.  Leave-one-year-out CV-RMSE (0.296) matches the
#   collinear 6-box model (0.294) but the coefficients are interpretable and don't bounce.
# OLR dropped: PSL interp-OLR ends 2022-12 (no live data) and its fitted coefficient was ~0.
PREDICTORS = ["n34", "soi", "u850"]
FIT_Y0 = 1999                          # fit start (satellite era fits MEI.v2 best; CV ≈ 0.30)
CLIM = (1991, 2020)


# ----------------------------------------------------------------------------- inputs
def _ensure(name: str) -> Path:
    MEI_IN.mkdir(parents=True, exist_ok=True)
    p = MEI_IN / name
    if not p.exists() or p.stat().st_size < 1000:
        print(f"  downloading {name} ...", flush=True)
        urllib.request.urlretrieve(URLS[name], p)
    return p


def load_mei() -> dict:
    """Official MEI.v2 → {(year, season 0..11): value}. seasons DJ,JF,...,ND."""
    out = {}
    for ln in open(_ensure("meiv2.data")):
        p = ln.split()
        if len(p) == 13 and p[0].isdigit() and 1979 <= int(p[0]) <= 2035:
            y = int(p[0])
            for s, v in enumerate(float(x) for x in p[1:]):
                if v > -900:
                    out[(y, s)] = v
    return out


def load_nino_monthly() -> dict:
    """CPC ERSST5 → {(y,m): {box: anom}}. cols: N1+2 a N3 a N4 a N3.4 a (idx 3,5,7,9)."""
    out = {}
    for ln in open(_ensure("nino.ascii")):
        p = ln.split()
        if len(p) >= 10 and p[0].isdigit():
            out[(int(p[0]), int(p[1]))] = {
                "n12": float(p[3]), "n3": float(p[5]), "n4": float(p[7]), "n34": float(p[9])}
    return out


def load_soi_daily() -> pd.Series:
    """DailySOI.txt → daily SOI series (already an index; no deseasonalizing)."""
    rows = []
    for ln in open(REF.parent / "soi" / "DailySOI.txt"):
        p = ln.split()
        if len(p) == 5 and p[0].isdigit():
            d = dt.date(int(p[0]), 1, 1) + dt.timedelta(int(p[1]) - 1)
            rows.append((pd.Timestamp(d), float(p[4])))
    return pd.Series(dict(rows)).sort_index()


def _deseason_daily(s: pd.Series) -> pd.Series:
    """Daily anomaly vs the monthly climatology over CLIM (broadcast by calendar month)."""
    m = s[(s.index.year >= CLIM[0]) & (s.index.year <= CLIM[1])]
    clim = m.groupby(m.index.month).mean()
    return s - s.index.month.map(clim).to_numpy()


def load_olr_box(daily: bool) -> pd.Series:
    """Central-Pacific OLR anomaly. daily=False → monthly (fit), True → daily (apply)."""
    f = OLR_DAY if daily else _ensure("olr.mon.mean.nc")
    if daily and (not f.exists() or f.stat().st_size < 1_000_000):
        raise FileNotFoundError(f"daily OLR not available at {f} (PSL download pending)")
    d = xr.open_dataset(f)
    la0, la1, lo0, lo1 = OLR_BOX
    box = d.olr.sel(lat=slice(la1, la0), lon=slice(lo0, lo1)).mean(("lat", "lon")).to_series()
    return _deseason_daily(box)


def load_u850_box(daily: bool) -> pd.Series:
    """Eq. 850-hPa zonal-wind anomaly box (band already 5S-5N). Monthly clim from the
    historical band series; apply year(s) appended from the 2026 ARCO tail."""
    hist = xr.open_dataset(REF / "eq_u850_bandseries.nc").u850
    box_h = hist.sel(longitude=slice(*U850_LON)).mean("longitude").to_series()
    series = box_h
    tail = REF / "eq_u850_2026_arco.nc"
    if daily and tail.exists():
        box_t = xr.open_dataset(tail).u850.sel(longitude=slice(*U850_LON)).mean("longitude").to_series()
        series = pd.concat([box_h, box_t[~box_t.index.isin(box_h.index)]]).sort_index()
    # monthly climatology over CLIM from the historical band
    m = box_h[(box_h.index.year >= CLIM[0]) & (box_h.index.year <= CLIM[1])]
    clim = m.groupby(m.index.month).mean()
    return (series - series.index.month.map(clim).to_numpy()).dropna()


# ----------------------------------------------------------------- bimonthly fit frame
def _pair(y, s):
    """Months for MEI season s in year y (DJ=Dec(y-1),Jan; ... ND=Nov,Dec)."""
    return [(y - 1, 12), (y, 1)] if s == 0 else [(y, s), (y, s + 1)]


def _monthly(series: pd.Series) -> dict:
    """Daily/sub-monthly series → {(y,m): monthly mean}."""
    mm = series.resample("MS").mean()
    return {(t.year, t.month): float(v) for t, v in mm.items() if np.isfinite(v)}


def fit() -> dict:
    mei = load_mei()
    nino = load_nino_monthly()
    soi_m = _monthly(load_soi_daily())
    u_m = {(t.year, t.month): float(v) for t, v in _monthly_from_anom(load_u850_box(False)).items()}
    src = {"soi": soi_m, "u850": u_m}            # plus nino boxes handled below

    def feat_of(ms):
        f = []
        for p in PREDICTORS:
            if p in BOXES:
                f.append(np.mean([nino[k][p] for k in ms]))
            else:
                f.append(np.mean([src[p][k] for k in ms]))
        return f

    def have(ms):
        return (all(k in nino for k in ms)
                and all(p in BOXES or all(k in src[p] for k in ms) for p in PREDICTORS))

    rows, seasons = [], []
    for (y, s), mv in sorted(mei.items()):
        if y < FIT_Y0:                 # the satellite era fits MEI.v2 best; older data degrades CV
            continue
        ms = _pair(y, s)
        if have(ms):
            rows.append([mv] + feat_of(ms)); seasons.append((y, s))
    a = np.array(rows); Y = a[:, 0]; X = a[:, 1:]
    A = np.column_stack([np.ones(len(Y)), X])
    beta, *_ = np.linalg.lstsq(A, Y, rcond=None)
    pred = A @ beta
    r = float(np.corrcoef(pred, Y)[0, 1])
    rmse = float(np.sqrt(((pred - Y) ** 2).mean()))
    # honest out-of-sample skill: leave-one-year-out CV
    yrs_arr = np.array([y for y, _ in seasons])
    cv_err = []
    for ty in np.unique(yrs_arr):
        tr = yrs_arr != ty; te = ~tr
        At = np.column_stack([np.ones(tr.sum()), X[tr]])
        bt, *_ = np.linalg.lstsq(At, Y[tr], rcond=None)
        cv_err.append((np.column_stack([np.ones(te.sum()), X[te]]) @ bt) - Y[te])
    cv_rmse = float(np.sqrt(np.concatenate(cv_err).__pow__(2).mean()))
    yrs = sorted({y for y, _ in seasons})
    info = {"predictors": PREDICTORS, "intercept": float(beta[0]),
            "coef": {k: float(v) for k, v in zip(PREDICTORS, beta[1:])},
            "n": len(Y), "R": r, "R2": r * r, "RMSE": rmse, "cv_rmse": cv_rmse,
            "fit_years": [yrs[0], yrs[-1]], "clim": list(CLIM)}
    REF.mkdir(parents=True, exist_ok=True)
    FIT_JSON.write_text(json.dumps(info, indent=2))
    print(f"  fit: n={info['n']} ({yrs[0]}-{yrs[-1]})  R={r:.3f}  RMSE={rmse:.3f}  "
          f"CV-RMSE={cv_rmse:.3f}", flush=True)
    print("  coef:", " ".join(f"{k}={v:+.3f}" for k, v in info["coef"].items()), flush=True)
    return info


def _monthly_from_anom(s: pd.Series) -> pd.Series:
    return s.resample("MS").mean()


# --------------------------------------------------------------------------- daily apply
def nino_daily(years: list[int]) -> dict[str, pd.Series]:
    """OISST daily-anom box means for the requested years (cos-lat weighted)."""
    out = {b: [] for b in BOXES}
    for yr in years:
        f = OISST / f"sst.day.anom.{yr}.nc"
        if not f.exists():
            print(f"    (no OISST anom for {yr}; skipping daily SST)", flush=True)
            continue
        d = xr.open_dataset(f)["anom"]
        w = np.cos(np.deg2rad(d.lat))
        for b, (la0, la1, lo0, lo1) in BOXES.items():
            sub = d.sel(lat=slice(la0, la1), lon=slice(lo0, lo1))
            ser = sub.weighted(w.sel(lat=slice(la0, la1))).mean(("lat", "lon")).to_series()
            out[b].append(ser)
    return {b: (pd.concat(v).sort_index() if v else pd.Series(dtype=float)) for b, v in out.items()}


def apply_daily(info: dict, years: list[int]) -> pd.DataFrame:
    """Build the daily predictor matrix, trailing-average, apply the regression."""
    nd = nino_daily(years)
    soi = load_soi_daily()
    u850 = load_u850_box(True)
    cols = {"n12": nd["n12"], "n3": nd["n3"], "n4": nd["n4"], "n34": nd["n34"],
            "soi": soi, "u850": u850}
    idx = pd.date_range(f"{min(years)}-01-01", f"{max(years)}-12-31", freq="D")
    df = pd.DataFrame({k: v.reindex(v.index.union(idx)).interpolate(limit=7).reindex(idx)
                       for k, v in cols.items()})
    df = df.loc[df.index <= df.dropna().index.max()]
    b = info["coef"]
    out = pd.DataFrame(index=df.index)
    for win, lab in ((30, "mei30"), (14, "mei14")):
        tr = df.rolling(win, min_periods=max(7, win // 2)).mean()
        out[lab] = info["intercept"] + sum(b[k] * tr[k] for k in PREDICTORS)
    out["valid"] = df.notna().all(axis=1)
    return out


# --------------------------------------------------------------------------------- plot
def mei_bimonthly_series(mei: dict) -> pd.Series:
    """Official MEI.v2 as a daily-indexed step series at season mid-points."""
    pts = {}
    for (y, s), v in mei.items():
        m0 = _pair(y, s)[0]
        mid = pd.Timestamp(m0[0], m0[1], 15) + pd.Timedelta(days=15)
        pts[mid] = v
    return pd.Series(pts).sort_index()


def plot(info, daily: pd.DataFrame, years: list[int], fname="mei_nowcast.webp", live=True):
    mei = mei_bimonthly_series(load_mei())
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5.2))
    t0 = pd.Timestamp(f"{min(years)}-01-01") - pd.Timedelta(days=120)
    msub = mei[mei.index >= t0]
    ax.step(msub.index, msub.values, where="mid", color="#888", lw=1.6,
            label="official MEI.v2 (bimonthly)", zorder=2)
    ax.scatter(msub.index, msub.values, s=22, color="#888", zorder=3)
    d = daily.dropna(subset=["mei30"], how="all")
    if d["mei30"].notna().sum() == 0:
        raise SystemExit("no valid daily nowcast rows — check predictor availability for "
                         f"{years} (need OISST anom + DailySOI + u850 tail)")
    band = info.get("cv_rmse", info["RMSE"])
    ax.fill_between(d.index, d["mei30"] - band, d["mei30"] + band, color="#d62728",
                    alpha=0.12, lw=0, zorder=1, label=f"±CV-RMSE ({band:.2f})")
    ax.plot(d.index, d["mei30"], color="#d62728", lw=2.0, label="nowcast · 30-day", zorder=4)
    ax.plot(d.index, d["mei14"], color="#ff7f0e", lw=1.2, alpha=0.85,
            label="nowcast · 14-day (fast)", zorder=4)
    last = d.dropna(subset=["mei30"]).iloc[-1]
    ax.scatter([last.name], [last["mei30"]], s=46, color="#d62728", zorder=6)
    ax.annotate(f"{last['mei30']:+.2f}\n{last.name:%b %-d}", (last.name, last["mei30"]),
                textcoords="offset points", xytext=(8, 0), va="center", fontsize=9,
                color="#d62728", fontweight="bold")
    ax.axhline(0, color="0.6", lw=0.7)
    ax.axhspan(0.5, 4, color="#d62728", alpha=0.05); ax.axhspan(-4, -0.5, color="#1f77b4", alpha=0.05)
    ax.set_ylim(min(-2.2, d[["mei30", "mei14"]].min().min() - 0.3),
                max(2.6, d[["mei30", "mei14"]].max().max() + 0.3))
    ax.set_xlim(t0, d.index.max() + pd.Timedelta(days=20))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.set_ylabel("MEI.v2 (std units)")
    head = "MEI.v2 daily nowcast" if live else f"MEI.v2 nowcast vs the {years[0]}–{years[-1]} El Niño"
    ax.set_title(f"{head}  ·  regression on Niño3.4 · SOI · eq-u850  "
                 f"(R={info['R']:.2f}, CV-RMSE {info.get('cv_rmse', info['RMSE']):.2f} vs official)",
                 fontsize=11)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    p = OUT / fname
    fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {p}", flush=True)


ANALOGS = {1982: "#e41a1c", 1997: "#ff7f00", 2015: "#4daf4a", 2023: "#984ea3"}


def render_analogs(info, daily, cur_year):
    """Onset-year overlay: official MEI.v2 for the big El Niño onsets (1982/97/2015/23)
    aligned by month-of-year, with the live nowcast for cur_year on the same axis."""
    mei = load_mei()
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    for Y, c in ANALOGS.items():
        xs, ys = [], []
        for s in range(12):                       # onset year: season s → month s+1
            if (Y, s) in mei:
                xs.append(s + 1); ys.append(mei[(Y, s)])
        for s in range(6):                        # tail into the following year (peak)
            if (Y + 1, s) in mei:
                xs.append(13 + s); ys.append(mei[(Y + 1, s)])
        if xs:
            ax.plot(xs, ys, color=c, lw=1.8, marker="o", ms=3, alpha=0.9,
                    label=f"{Y}–{str(Y + 1)[2:]}  (peak {max(ys):+.1f})")
    # current year — official points + daily nowcast
    xo = [s + 1 for s in range(12) if (cur_year, s) in mei]
    yo = [mei[(cur_year, s)] for s in range(12) if (cur_year, s) in mei]
    ax.scatter(xo, yo, s=42, color="k", zorder=6, label=f"{cur_year} official")
    d = daily.dropna(subset=["mei30"])
    d = d[d.index.year == cur_year]
    xn = [t.month + (t.day - 1) / 30.4 for t in d.index]
    band = info.get("cv_rmse", info["RMSE"])
    ax.fill_between(xn, d["mei30"] - band, d["mei30"] + band, color="k", alpha=0.10, lw=0, zorder=4)
    ax.plot(xn, d["mei30"].values, color="k", lw=2.8, zorder=5, label=f"{cur_year} nowcast")
    ax.axvline(12.5, color="0.6", lw=0.8, ls="--")
    ax.axhline(0, color="0.6", lw=0.7); ax.axhline(0.5, color="#d62728", lw=0.6, ls=":")
    labs = list("JFMAMJJASOND") + list("JFMAMJ")
    ax.set_xticks(range(1, 19)); ax.set_xticklabels(labs)
    ax.set_xlim(0.5, 18.5)
    ax.text(6.5, ax.get_ylim()[1], "onset year", ha="center", va="top", fontsize=8, color="0.5")
    ax.text(15, ax.get_ylim()[1], "year +1", ha="center", va="top", fontsize=8, color="0.5")
    ax.set_ylabel("MEI.v2 (std units)")
    ax.set_title(f"MEI.v2 onset-year analogs — is {cur_year} tracking the big El Niños?", fontsize=11)
    ax.legend(loc="upper left", fontsize=8.5, ncol=2, framealpha=0.9)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    p = OUT / "mei_analogs.webp"; fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {p}", flush=True)


def render_history(info, daily):
    """Full official MEI.v2 record since 1980, El Niño/La Niña shaded, analog onsets marked,
    the live nowcast tail in black."""
    s = mei_bimonthly_series(load_mei())
    s = s[s.index >= pd.Timestamp("1980-01-01")]
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.fill_between(s.index, 0, s.values, where=(s.values >= 0), interpolate=True,
                    color="#d62728", alpha=0.55, lw=0)
    ax.fill_between(s.index, 0, s.values, where=(s.values < 0), interpolate=True,
                    color="#1f77b4", alpha=0.55, lw=0)
    ax.plot(s.index, s.values, color="#333", lw=0.5)
    d = daily.dropna(subset=["mei30"])
    ax.plot(d.index, d["mei30"], color="k", lw=1.4, label="live daily nowcast")
    top = max(2.6, s.max() + 0.2)
    for Y, c in ANALOGS.items():
        ax.axvline(pd.Timestamp(Y, 7, 1), color=c, lw=1.0, ls=":", alpha=0.8)
        ax.text(pd.Timestamp(Y, 7, 1), top, str(Y), ha="center", va="top", fontsize=8,
                color=c, fontweight="bold")
    ax.axhline(0, color="0.5", lw=0.6)
    ax.set_ylim(min(-2.4, s.min() - 0.2), top)
    ax.set_xlim(pd.Timestamp("1980-01-01"), d.index.max() + pd.Timedelta(days=120))
    ax.set_ylabel("MEI.v2 (std units)")
    ax.set_title("Multivariate ENSO Index v2 — full record since 1980 (official bimonthly; "
                 "live nowcast tail in black)", fontsize=11)
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.9)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    p = OUT / "mei_history.webp"; fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {p}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true", help="(re)fit the regression")
    ap.add_argument("--apply", nargs="*", type=int, default=[2026],
                    help="year(s) to produce the daily nowcast for")
    ap.add_argument("--validation", action="store_true",
                    help="also render the 2015–16 out-of-sample validation panel")
    args = ap.parse_args(argv)
    info = fit() if (args.fit or not FIT_JSON.exists()) else json.loads(FIT_JSON.read_text())
    if args.validation:
        plot(info, apply_daily(info, [2015, 2016]), [2015, 2016],
             fname="mei_validation.webp", live=False)
    if args.apply:
        years = sorted(args.apply)
        daily = apply_daily(info, years)
        v = daily.dropna(subset=["mei30"])
        if len(v):
            print(f"  apply {years}: {v.index.min():%Y-%m-%d}→{v.index.max():%Y-%m-%d}; "
                  f"latest 30d={v['mei30'].iloc[-1]:+.2f} 14d={daily['mei14'].dropna().iloc[-1]:+.2f}", flush=True)
        plot(info, daily, years)
        render_analogs(info, daily, max(years))
        render_history(info, daily)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
