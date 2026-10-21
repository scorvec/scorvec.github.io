#!/usr/bin/env python3
"""The SSW surface-imprint index: a day's sea-level pressure projected on the CMIP6 post-SSW pattern.

PATTERN. The CMIP6 "What usually follows" composite (reference/cmip6_follow.nc, build_cmip6_follow.py): the nine
models' sea-level pressure anomaly over days 1-60 after their sudden warmings (set ssw_all, the mean of the d1_30 and
d31_60 windows), the models weighted equally, on its 2.5 deg boxes 20-90N. Only the robust cells carry weight (across
models a t-test passing FDR 10 % and at least 80 % of the models on the sign, strat_history.robust); the rest are zero.

INDEX. I = sum(w P A) / sum(w P^2) over the boxes, w = cos(latitude), A = the day's sea-level pressure anomaly averaged
onto the same boxes (area-weighted overlap, any regular grid): 1 = an imprint as strong as the CMIP6 composite, positive
= SSW-like (pressure high over the Arctic, low across the mid-latitude Atlantic: a negative Arctic Oscillation).
Anomalies are taken from ERA5's 1991-2020 four-harmonic day-of-year climatology (366 slots) with each box's linear trend
removed, the coefficients stored in reference/ssw_imprint.nc (build_ssw_imprint.py) so a forecast uses exactly the
reanalysis' baseline.
"""

from __future__ import annotations

from pathlib import Path

import pyproj  # noqa: F401  (load pyproj's libproj before xarray/eccodes: else a double free at exit)
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REF = HERE / "reference" / "ssw_imprint.nc"
LAT_EDGES = np.arange(20.0, 90.0 + 1e-6, 2.5)  # the CMIP6 composite's boxes
LON_EDGES = np.arange(0.0, 360.0 + 1e-6, 2.5)
HARMONICS = 4
EPOCH = pd.Timestamp("1991-01-01")  # the trend's origin


def _overlap(centres: np.ndarray, width: float, edges: np.ndarray) -> np.ndarray:
    """(boxes, cells) overlap lengths of 1-D cells of `width` centred on `centres` with the boxes between `edges`."""
    lo, hi = centres - width / 2, centres + width / 2
    return np.clip(np.minimum(hi[None], edges[1:, None]) - np.maximum(lo[None], edges[:-1, None]), 0.0, None)


class Boxes:
    """Area-weighted averaging of a regular lat-lon grid onto the 2.5 deg boxes: boxes(field) for (..., lat, lon)."""

    def __init__(self, lat, lon):
        lat, lon = np.asarray(lat, float), np.asarray(lon, float) % 360.0
        dla, dlo = abs(lat[1] - lat[0]), abs(lon[1] - lon[0])
        wla = _overlap(lat, dla, LAT_EDGES) * np.cos(np.deg2rad(lat))[None]
        # a cell straddling 0 E also overlaps the boxes just below 360 E
        wlo = (
            _overlap(lon, dlo, LON_EDGES)
            + _overlap(lon - 360.0, dlo, LON_EDGES)
            + _overlap(lon + 360.0, dlo, LON_EDGES)
        )
        if (wla.sum(1) == 0).any() or (wlo.sum(1) == 0).any():
            raise ValueError("the grid does not cover 20-90N at every longitude")
        self.wla = wla / wla.sum(1, keepdims=True)
        self.wlo = wlo / wlo.sum(1, keepdims=True)

    def __call__(self, field: np.ndarray) -> np.ndarray:
        return np.einsum("il,...lk,jk->...ij", self.wla, np.asarray(field, float), self.wlo, optimize=True)


def design(t: pd.DatetimeIndex) -> np.ndarray:
    """Four annual harmonics on the 366-slot day of year, then the linear trend in years from EPOCH."""
    from quantmet.degdays import doy366

    x = 2 * np.pi * (np.asarray(doy366(t), float) - 1) / 366.0
    cols = [np.ones(len(t))] + [f(k * x) for k in range(1, HARMONICS + 1) for f in (np.cos, np.sin)]
    return np.column_stack([*cols, (t - EPOCH).days.values / 365.25])


class Ref:
    """The stored pattern and baseline: ref.index(times, boxes_hpa) -> the imprint index of each day."""

    def __init__(self, path: Path = REF):
        import xarray as xr

        with xr.open_dataset(path) as f:
            ds = f.load()
        self.ds = ds
        self.pattern = ds.pattern.values  # (lat, lon) hPa, zero outside the robust cells
        self.w = np.cos(np.deg2rad(ds.lat.values))[:, None] * np.ones_like(self.pattern)
        self.coef = ds.baseline.values  # (term, lat, lon)

    def anomaly(self, t: pd.DatetimeIndex, slp_boxes: np.ndarray) -> np.ndarray:
        """(time, lat, lon) sea-level pressure on the boxes (hPa) less the ERA5 baseline of each day."""
        return np.asarray(slp_boxes, float) - np.einsum("tk,kij->tij", design(pd.DatetimeIndex(t)), self.coef)

    def project(self, anom: np.ndarray) -> np.ndarray:
        wp = self.w * self.pattern
        return (np.asarray(anom) * wp).sum((-2, -1)) / (wp * self.pattern).sum()

    def index(self, t, slp_boxes) -> np.ndarray:
        return self.project(self.anomaly(t, slp_boxes))


# ================================================================================================= the live product ===
# A November-April product: a run is drawn when its day 15 falls in the season (the analysis is recorded all year, so
# the tail is there when it starts). Each model's members from their own open data, daily 0-360 h (00 or 12 UTC
# instants), on the boxes:
#   aifs  AIFS-ENS control + 25 perturbed members (ECMWF open data, Google Cloud mirror only), msl
#   ifs   IFS-ENS, all 50 perturbed members (the ensemble files carry no control), msl
#   hres  the IFS control, msl from its own stream (oper)
#   gefs  NOAA GEFS control + 30, PRMSL byte-ranged from NOAA's S3 bucket through the .idx files
#   geps  ECCC GEPS control + 20, the MSC Datamart's PRMSL allmbrs files
# ANALYSIS. The IFS control's step 0 (its 4D-Var analysis) of every 00 and 12 UTC cycle, averaged per day, kept in
# data/ssw_imprint_history.json (backfill seeds it; each run adds its cycle); ERA5's own index stops where the store does.
# RAW. The forecasts are each model's own output, no drift removed. Verified on the controls' runs
# of July-October 2026 against those analyses, every model drifts towards an SSW-like index as the run goes on (about
# +2 for the IFS control and +1.7 for GEFS by day 15), and the figure says the late leads lean that way.
MODELS = {"aifs": "AIFS-ENS", "ifs": "IFS-ENS", "hres": "IFS control", "gefs": "GEFS", "geps": "GEPS"}
STYLE = {"aifs": "#b4541f", "ifs": "#1f4b6e", "hres": "#1f4b6e", "gefs": "#3b88c4", "geps": "#2f7d4f"}  # nh_vortex's
LEADS = tuple(range(0, 361, 24))
AIFS_MEMBERS = 25
REPO = HERE.parents[1]
DATA = REPO / "assets" / "sst" / "data"
OUT_JSON = DATA / "ssw_imprint.json"
HISTORY = DATA / "ssw_imprint_history.json"
RUNS = DATA / "ssw_imprint_runs.json"
FIGURE = REPO / "assets" / "sst" / "ssw_imprint.webp"
OUTLOOK_DAYS = (16, 45)  # the window after day 15 the coupling regression speaks to
SEASON = (11, 12, 1, 2, 3, 4)  # November-April: the product's season and the months the coupling is measured on


def _ecmwf():
    import sys

    sys.path.insert(0, str(HERE))
    import ifs_ens as IE  # pins the store to the Google mirror

    IE._set_env()
    return IE


def _ecmwf_boxes(path: Path, short: str = "msl"):
    """(member, lead, lat, lon) hPa boxes and the lead hours from one store GRIB (members, if any, first)."""
    import xarray as xr

    with xr.open_dataset(
        path, engine="cfgrib", backend_kwargs={"indexpath": "", "filter_by_keys": {"shortName": short}}
    ) as ds:
        da = ds[short]
        da = da.expand_dims("number") if "number" not in da.dims else da
        da = da.expand_dims("step") if "step" not in da.dims else da
        da = da.transpose("number", "step", "latitude", "longitude")
        b = Boxes(da.latitude.values, da.longitude.values)
        leads = (da.step.values / np.timedelta64(1, "h")).astype(int)
        return np.stack([b(da.isel(number=i).values / 100.0) for i in range(da.sizes["number"])]), leads


def slp_ecmwf(model: str, date: str, hh: str, steps=LEADS):
    """(member, lead, lat, lon) hPa boxes of an ECMWF model from the Google mirror, with the lead hours."""
    IE = _ecmwf()
    cyc = IE.E.Cycle(date, hh)
    steps = tuple(int(s) for s in steps)
    if model == "aifs":
        parts = [
            IE.E.Spec("aifs-ens", "cf", "msl", "sfc", (), steps),
            IE.E.Spec("aifs-ens", "pf", "msl", "sfc", (), steps, AIFS_MEMBERS),
        ]
    elif model == "ifs":
        parts = [IE.E.Spec("ifs", "pf", "msl", "sfc", (), steps)]
    else:  # hres, the IFS control
        parts = [IE.E.Spec("ifs", "fc", "msl", "sfc", (), steps)]
    got = [_ecmwf_boxes(IE.ensure(cyc, s)) for s in parts]
    return np.concatenate([g[0] for g in got]), got[0][1]


def slp_gefs(date: str, hh: str, steps=LEADS, workers: int = 32):
    """(member, lead, lat, lon) hPa boxes of GEFS (control first), PRMSL byte-ranged from NOAA's S3 bucket."""
    import sys
    from concurrent.futures import ThreadPoolExecutor

    sys.path.insert(0, str(HERE))
    import strat_ens as SE

    R = SE._gefs_lib()
    mems = SE.GEFS_MEMBERS
    b = Boxes(SE.LAT, SE.LON)

    def one(job):
        mi, li, L = job
        url = SE.gefs_url(date, hh, mems[mi], L)
        step = "anl" if L == 0 else f"{L} hour fcst"
        try:
            hit = [e for e in R.index(url) if e[4] == "PRMSL" and e[2] == "mean sea level" and e[3] == step]
            return mi, li, b(R.decode(R.get(url, (hit[0][0], hit[0][1]))) / 100.0) if hit else None
        except Exception:  # noqa: BLE001  (not on S3: the member-step stays missing)
            return mi, li, None

    out = np.full((len(mems), len(steps), LAT_EDGES.size - 1, LON_EDGES.size - 1), np.nan)
    jobs = [(mi, li, L) for mi in range(len(mems)) for li, L in enumerate(steps)]
    with ThreadPoolExecutor(workers) as ex:
        for mi, li, v in ex.map(one, jobs):
            if v is not None:
                out[mi, li] = v
    return out[np.isfinite(out).all(axis=(1, 2, 3))], np.array(steps)


GEPS_SLP = (
    "https://dd.weather.gc.ca/{d}/WXO-DD/ensemble/geps/grib2/raw/{c}/{L:03d}/"
    "CMC_geps-raw_PRMSL_MSL_0_latlon0p5x0p5_{d}{c}_P{L:03d}_allmbrs.grib2"
)


def slp_geps(date: str, hh: str, steps=LEADS):
    """(member, lead, lat, lon) hPa boxes of GEPS (control first) from the Datamart's PRMSL allmbrs files."""
    import sys

    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(REPO / "scripts" / "geps"))
    import datamart
    import strat_ens as SE
    import strat_maps as sm

    b = Boxes(SE.LAT, SE.LON)
    per = []
    for L in steps:
        f = datamart.download(
            GEPS_SLP.format(d=date, c=hh, L=L), sm.DL / f"geps_{date}{hh}_PRMSL_{L:03d}.grib2", timeout=300
        )
        if f is None:
            raise SystemExit(f"GEPS {date} {hh}Z: PRMSL +{L} h is not on the Datamart")
        msgs = SE._messages(f)
        per.append({m: b(v / 100.0) for m, v in msgs.items()})
    mems = sorted(set.intersection(*(set(p) for p in per)))
    return np.stack([[p[m] for p in per] for m in mems]), np.array(steps)


FETCH = {
    "aifs": lambda d, h: slp_ecmwf("aifs", d, h),
    "ifs": lambda d, h: slp_ecmwf("ifs", d, h),
    "hres": lambda d, h: slp_ecmwf("hres", d, h),
    "gefs": slp_gefs,
    "geps": slp_geps,
}


def _read(path: Path, default):
    import json

    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def _write(path: Path, doc) -> None:
    import json

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(doc, separators=(",", ":")))


def analysis_day(ref: Ref, date: str, hh: str) -> float:
    """The imprint index of the IFS control's analysis (step 0) of one cycle."""
    boxes, _ = slp_ecmwf("hres", date, hh, steps=(0,))
    return float(ref.index(pd.DatetimeIndex([pd.Timestamp(f"{date} {hh}:00")]), boxes[0])[0])


def add_analysis(ref: Ref, date: str, hh: str) -> None:
    """Put one cycle's analysis into the history; a day's value is the mean of its 00 and 12 UTC analyses."""
    h = _read(HISTORY, {"source": "", "cycles": {}})
    h["cycles"][f"{date}{hh}"] = round(analysis_day(ref, date, hh), 3)
    h["source"] = "IFS control step 0 (the analysis), 00 and 12 UTC, Google Cloud mirror; days = the mean of the two"
    _write(HISTORY, h)


def history_days() -> pd.Series:
    h = _read(HISTORY, {}).get("cycles", {})
    if not h:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    s = pd.Series(h)
    s.index = pd.to_datetime(s.index, format="%Y%m%d%H").normalize()
    return s.groupby(level=0).mean().sort_index()


def model_entry(ref: Ref, model: str, date: str, hh: str) -> dict:
    boxes, leads = FETCH[model](date, hh)
    init = pd.Timestamp(f"{date} {hh}:00")
    valid = pd.DatetimeIndex([init + pd.Timedelta(hours=int(L)) for L in leads])
    idx = np.stack([ref.index(valid, m) for m in boxes])  # (member, lead)
    return {
        "name": MODELS[model],
        "init": f"{init:%Y-%m-%d %H}Z",
        "members": int(idx.shape[0]),
        "lead_h": [int(L) for L in leads],
        "valid": [f"{v:%Y-%m-%d}" for v in valid],
        "index": np.round(idx, 2).tolist(),
        "mean": np.round(idx.mean(0), 2).tolist(),
    }


def outlook(ref: Ref, entry: dict, drip: dict | None) -> dict | None:
    """P(the index over days 16-45 > 0) from the ERA5/MERRA-2 coupling regression on this run's day-15 state: the 100 hPa
    cap (the drip's drift-corrected ensemble mean and spread) and the index over days 10-15. Ensembles only; None
    outside November-April, where the coupling was not measured, or when the drip is not on this run's cycle."""
    import json

    from scipy import stats

    if entry["members"] < 5 or not drip or drip.get("init") != entry["init"] or "member_sd" not in drip:
        return None
    d15 = pd.Timestamp(entry["valid"][-1])
    if d15.month not in SEASON:
        return None
    k = [float(x) for x in drip["levels"]].index(100.0)
    j = drip["lead_h"].index(360)
    z, sz = drip["forecast_mean_sd"][j][k], drip["member_sd"][j][k]
    if z is None or sz is None:
        return None
    idx = np.array(entry["index"], float)
    late = [i for i, L in enumerate(entry["lead_h"]) if 240 <= L <= 360]
    i0 = idx[:, late].mean(1)
    cp = json.loads(ref.ds.attrs["coupling"])
    a, b, c = cp["coef"]
    mu = a + b * z + c * float(i0.mean())
    sd = float(np.sqrt(cp["resid_sd"] ** 2 + (b * sz) ** 2 + (c * i0.std()) ** 2))
    start = pd.Timestamp(entry["init"][:10]) + pd.Timedelta(days=OUTLOOK_DAYS[0])
    return {
        "window": [f"{start:%Y-%m-%d}", f"{start + pd.Timedelta(days=OUTLOOK_DAYS[1] - OUTLOOK_DAYS[0]):%Y-%m-%d}"],
        "p_positive": round(float(stats.norm.sf(0.0, mu, sd)), 3),
        "base_rate": cp["base_rate"],
        "mean": round(float(mu), 2),
        "p10": round(float(mu - 1.2816 * sd), 2),
        "p90": round(float(mu + 1.2816 * sd), 2),
        "z100_day15": round(float(z), 2),
        "index_d10_15": round(float(i0.mean()), 2),
    }


# ========================================================================================================== figure ===
TAIL = 45  # days of analysis before the newest run


def _drip(model: str) -> dict | None:
    return _read(DATA / f"{model}_drip.json", None)


def analysis_series(ref: Ref) -> pd.Series:
    """ERA5's index, then the IFS control analyses where the store stops."""
    era5 = pd.Series(ref.ds["index"].values, index=pd.DatetimeIndex(ref.ds.time.values))
    live = history_days()
    live = live[live.index > era5.index[-1]]
    return pd.concat([era5, live]).sort_index() if len(live) else era5


def render(doc: dict, ref: Ref, out: Path = FIGURE) -> None:
    import json

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, MUTED = "#1f2328", "#6f6b64"
    models = [m for m in MODELS if m in doc["models"]]
    inits = {m: pd.Timestamp(doc["models"][m]["init"][:13].replace(" ", "T") + ":00") for m in models}
    t1 = max(inits.values())
    x0, x1 = (t1 - pd.Timedelta(days=TAIL)).normalize(), t1 + pd.Timedelta(days=15)
    drips = {m: _drip(m) for m in models}
    fig = plt.figure(figsize=(14.2, 10.2), dpi=112)
    gs = fig.add_gridspec(
        3, 2, width_ratios=[3.2, 1], left=0.06, right=0.985, top=0.885, bottom=0.135, hspace=0.32, wspace=0.16
    )
    rows = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    for ax in rows[1:]:
        ax.sharex(rows[0])
    side = fig.add_subplot(gs[:2, 1])
    side_note = fig.add_subplot(gs[2, 1])
    side_note.axis("off")

    # rows 1-2: the polar cap at 10 and 100 hPa, GEOS FP analyses then each model's drift-corrected mean
    tail = next((d for d in drips.values() if d), None)
    for ax, lev, title in (
        (rows[0], 10.0, "10 hPa polar cap: the vortex"),
        (rows[1], 100.0, "100 hPa polar cap: the coupling level"),
    ):
        if tail:
            k = [float(x) for x in tail["levels"]].index(lev)
            td = pd.DatetimeIndex(tail["tail_days"])
            ax.plot(td, [r[k] for r in tail["tail_sd"]], color=INK, lw=2.0, label="GEOS FP analysis")
        for m in models:
            d = drips[m]
            if not d or d.get("init") != doc["models"][m]["init"]:
                continue
            k = [float(x) for x in d["levels"]].index(lev)
            t = [inits[m].normalize() + pd.Timedelta(hours=h) for h in d["lead_h"]]
            ax.plot(
                t,
                [r[k] for r in d["forecast_mean_sd"]],
                color=STYLE[m],
                lw=1.8,
                ls="--" if m == "hres" else "-",
                label=MODELS[m],
            )
        ax.axhline(0, color="#9a958c", lw=0.8)
        ax.axhline(1, color="#b3122b", lw=0.9, ls=":")
        ax.set_ylabel("sd (high cap = weak vortex)", fontsize=9)
        ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", color=INK)

    # row 3: the imprint index
    ax = rows[2]
    obs = analysis_series(ref)
    o = obs[(obs.index >= x0 - pd.Timedelta(days=7)) & (obs.index <= t1)]
    sd7 = json.loads(ref.ds.attrs["sd7_by_month"])
    span = pd.date_range(x0, x1.normalize(), freq="D")
    # the monthly values at mid-month, interpolated through the year (cyclically)
    mid = np.array([15.0 + 30.44 * k for k in range(-1, 13)])
    band = np.interp(span.dayofyear.values, mid, [sd7[-1], *sd7, sd7[0]])
    ax.fill_between(span, -band, band, color="#e6e3dc", lw=0, label="ERA5 ±1 sd (7-day means)")
    if len(o):
        ax.plot(o.index, o.values, color=INK, lw=0.7, alpha=0.45)
        ax.plot(
            o.index, o.rolling(7, center=True, min_periods=4).mean(), color=INK, lw=2.0, label="analysis, 7-day mean"
        )
    pooled = []
    for m in models:
        e = doc["models"][m]
        t = pd.DatetimeIndex(e["valid"])
        if m != "hres":
            pooled.append(pd.DataFrame(np.array(e["index"]).T, index=t))
        ax.plot(t, e["mean"], color=STYLE[m], lw=1.8, ls="--" if m == "hres" else "-", label=MODELS[m])
    if pooled:
        P = pd.concat(pooled, axis=1)
        ax.fill_between(
            P.index,
            P.quantile(0.1, axis=1),
            P.quantile(0.9, axis=1),
            color="#5b6b7f",
            alpha=0.16,
            lw=0,
            label="all members 10–90 %",
        )
    ax.axhline(0, color="#9a958c", lw=0.8)
    ax.set_ylabel("index (1 = CMIP6 post-SSW mean)", fontsize=9)
    ax.set_title(
        "Surface: the SSW imprint index (sea-level pressure on the CMIP6 post-SSW pattern)",
        loc="left",
        fontsize=10.5,
        fontweight="bold",
        color=INK,
    )
    for a in rows:
        a.axvline(t1, color="#8a8680", lw=1)
        a.set_xlim(x0, x1)
        a.grid(alpha=0.22)
        a.legend(loc="upper left", fontsize=7.6, ncol=4, framealpha=0.9)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    for a in rows[:2]:
        plt.setp(a.get_xticklabels(), visible=False)
    import matplotlib.dates as mdates

    rows[2].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

    # side: the outlook beyond day 15
    cp = json.loads(ref.ds.attrs["coupling"])
    sk = json.loads(ref.ds.attrs["outlook_skill"])
    ens = [m for m in models if doc["models"][m].get("outlook")]
    side.set_xlim(0, 100)
    side.set_ylim(-0.6, max(len(ens), 1) - 0.4)
    side.invert_yaxis()
    side.axvline(100 * cp["base_rate"], color=INK, lw=1, ls=":")
    for i, m in enumerate(ens):
        ol = doc["models"][m]["outlook"]
        side.barh(i, 100 * ol["p_positive"], color=STYLE[m], alpha=0.8, height=0.55)
        side.text(
            1,
            i + 0.43,
            f"mean {ol['mean']:+.1f} (10–90 % {ol['p10']:+.1f} to {ol['p90']:+.1f})",
            fontsize=7.8,
            color=MUTED,
            va="center",
        )
        side.text(100 * ol["p_positive"] + 1, i, f"{100 * ol['p_positive']:.0f} %", fontsize=9, va="center", color=INK)
    side.set_yticks(range(len(ens)), [MODELS[m] for m in ens], fontsize=9)
    side.set_xlabel("% chance the index is positive", fontsize=9)
    win = doc["models"][ens[0]]["outlook"]["window"] if ens else None
    side.set_title(
        "After the forecast: "
        + (f"{pd.Timestamp(win[0]):%d %b}–{pd.Timestamp(win[1]):%d %b}" if win else "days 16–45"),
        loc="left",
        fontsize=10.5,
        fontweight="bold",
        color=INK,
    )
    note = (
        f"From each run's day 15: the 100 hPa cap and the\nindex over days 10–15, through the ERA5/MERRA-2\nregression "
        f"(Nov–Apr 1980–, {cp['n_winters']} winters).\nDotted: the base rate, {100 * cp['base_rate']:.0f} %.\n"
        f"Brier skill, each winter predicted from the\nothers: {sk['cap_1sd']['bss']:.2f} ({sk['cap_1sd']['ci95'][0]:.2f}–"
        f"{sk['cap_1sd']['ci95'][1]:.2f}) with the cap ≥ 1 sd from\nnormal, {sk['all_days']['bss']:.2f} on every day."
        if ens
        else "No ensemble outlook: the coupling is\nmeasured for November–April."
    )
    side_note.text(0, 1.0, note, transform=side_note.transAxes, fontsize=8, color=MUTED, va="top", linespacing=1.45)
    for s in ("top", "right"):
        side.spines[s].set_visible(False)

    fig.text(
        0.06,
        0.975,
        f"Downward propagation monitor — newest run {t1:%d %b %Y %HZ}",
        fontsize=15,
        fontweight="bold",
        color=INK,
        va="top",
    )
    fig.text(
        0.06,
        0.94,
        "Does the stratosphere's anomaly reach the surface? The polar-cap height at 10 and 100 hPa, standardised as in "
        "the dripping paint,\nthen the surface's SSW imprint index, and what the state at day 15 has meant for the month after.",
        fontsize=9.6,
        color=MUTED,
        va="top",
    )
    tests = json.loads(ref.ds.attrs["imprint_tests"])
    fig.text(
        0.06,
        0.012,
        f"Index: the day's sea-level pressure anomaly (ERA5 1991–2020 baseline) projected on the CMIP6 nine-model mean "
        f"sea-level pressure over days 1–60 after SSWs,\nrobust boxes only; after the {tests['d1_30']['n']} MERRA-2 SSWs "
        f"since 1980 it averaged {tests['d1_30']['mean']:+.1f} over days 1–30 (p {tests['d1_30']['t_p']:.0e}).\n"
        "Forecasts: ensemble means, daily 00/12Z instants, raw model output (no drift removed: on July–October 2026 runs "
        "the models drift towards an SSW-like index, about +2 by day 15).\n"
        "Sources: ECMWF open data (CC BY 4.0, Google Cloud mirror) · NOAA GEFS (AWS Open Data) · ECCC GEPS (MSC Datamart) · "
        "ERA5 (C3S) · MERRA-2 and GEOS FP (NASA GMAO) · CMIP6 (WCRP).",
        fontsize=7.6,
        color=MUTED,
        linespacing=1.45,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=112, facecolor="white", pil_kwargs={"quality": 88, "method": 6})
    plt.close(fig)
    print(f"wrote {out}")


# ============================================================================================================== CLI ===
def in_season(date: str, hh: str) -> bool:
    """A November-April product: a run is drawn when its day 15 falls in the season."""
    return (pd.Timestamp(f"{date} {hh}:00") + pd.Timedelta(hours=LEADS[-1])).month in SEASON


def run(models, date: str, hh: str, draw: bool = True) -> int:
    ref = Ref()
    doc = _read(OUT_JSON, {"models": {}})
    season = in_season(date, hh)
    if not season:
        print(f"  {date} {hh}Z: day 15 is outside November-April; only the analysis is recorded", flush=True)
    for m in models if season else ():
        try:
            doc["models"][m] = model_entry(ref, m, date, hh)
        except (SystemExit, Exception) as e:  # noqa: BLE001  (one model missing must not take the others down)
            print(
                f"::warning::{MODELS[m]} {date} {hh}Z imprint index failed ({type(e).__name__}: {str(e)[:120]}); keeping its previous run"
            )
            continue
        doc["models"][m]["outlook"] = outlook(ref, doc["models"][m], _drip(m))
        e = doc["models"][m]
        print(
            f"  {MODELS[m]} {e['init']}: {e['members']} members, day 15 mean {e['mean'][-1]:+.2f}; outlook {e['outlook']}",
            flush=True,
        )
    if "hres" in models or "ifs" in models:
        try:
            add_analysis(ref, date, hh)
        except Exception as e:  # noqa: BLE001
            print(f"::warning::IFS control analysis {date} {hh}Z not added ({type(e).__name__}: {str(e)[:100]})")
    runs = _read(RUNS, {"runs": {}})
    for m, e in doc["models"].items():
        if e.get("outlook"):
            runs["runs"].setdefault(e["init"], {})[m] = e["outlook"]
    _write(RUNS, runs)
    _write(OUT_JSON, doc)
    if draw and season:
        render(doc, ref)
    return 0


def backfill(start: str, end: str | None = None) -> int:
    """Add every 00 and 12 UTC IFS control analysis from `start` to the history, then compare the overlap with ERA5."""
    ref = Ref()
    end_t = pd.Timestamp(end) if end else pd.Timestamp.now("UTC").tz_localize(None).normalize()
    have = _read(HISTORY, {"cycles": {}})["cycles"]
    for d in pd.date_range(start, end_t, freq="D"):
        for hh in ("00", "12"):
            if f"{d:%Y%m%d}{hh}" in have:
                continue
            try:
                add_analysis(ref, f"{d:%Y%m%d}", hh)
            except Exception as e:  # noqa: BLE001
                print(f"  {d:%Y-%m-%d} {hh}Z: {type(e).__name__}: {str(e)[:80]}", flush=True)
    era5 = pd.Series(ref.ds["index"].values, index=pd.DatetimeIndex(ref.ds.time.values))
    live = history_days()
    both = pd.concat([era5, live], axis=1, keys=["era5", "ifs"]).dropna()
    h = _read(HISTORY, {})
    if len(both) >= 30:
        h["check_vs_era5"] = {
            "days": len(both),
            "r": round(float(both.era5.corr(both.ifs)), 3),
            "mean_diff": round(float((both.ifs - both.era5).mean()), 3),
            "rmse": round(float(np.sqrt(((both.ifs - both.era5) ** 2).mean())), 3),
        }
        print(f"  overlap with ERA5: {h['check_vs_era5']}")
    _write(HISTORY, h)
    return 0


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="each model's members, the analysis of the cycle, the outlook, the figure")
    r.add_argument("--models", default=",".join(MODELS))
    r.add_argument("--date", required=True)
    r.add_argument("--time", required=True)
    r.add_argument("--no-render", action="store_true", help="update the JSON only (several cycles, then one render)")
    b = sub.add_parser("backfill", help="the IFS control analyses from --start into the history")
    b.add_argument("--start", required=True)
    b.add_argument("--end")
    sub.add_parser("render", help="redraw the figure from the published JSON")
    h = sub.add_parser("has", help="exit 0 when the published JSON already holds this model's cycle")
    h.add_argument("--model", required=True)
    h.add_argument("--date", required=True)
    h.add_argument("--time", required=True)
    a = ap.parse_args()
    if a.cmd == "run":
        return run([x for x in a.models.split(",") if x], a.date, f"{int(a.time):02d}", not a.no_render)
    if a.cmd == "backfill":
        return backfill(a.start, a.end)
    if a.cmd == "has":
        init = f"{pd.Timestamp(a.date):%Y-%m-%d} {int(a.time):02d}Z"
        return 0 if _read(OUT_JSON, {"models": {}})["models"].get(a.model, {}).get("init") == init else 1
    render(_read(OUT_JSON, {"models": {}}), Ref())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
