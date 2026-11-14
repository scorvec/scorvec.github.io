#!/usr/bin/env python3
"""What comes before a North Pacific jet extension: ERA5 composites ahead of extension onsets, and how the odds of an
onset change with the state of the tropics, of Asia and of the jet itself.

Onset      the first day of a run of at least MIN_DAYS days in the extension phase (pacjet_ref.nc: PC1 leading, the
           jet index at 1 sigma or more), with no extension day in the QUIET days before it, November to March.
Composites the 200 hPa velocity potential 30S-30N, sea-level pressure and 500 hPa height 20-80N from the Middle East
           to the eastern Pacific, as means over the windows of WINDOWS (days before the onset). Each field is the
           anomaly from its day-of-year normal (31-day smoothed, 1991-2020) less its own mean over the BACK days before
           that day, so the season's standing pattern (ENSO, the trend) is out and what is left is what changed in the
           weeks before. One-sample t-test across onsets at each grid point, Benjamini-Hochberg at ALPHA_MAP over
           the map of each field and window; only significant points are kept, and a window with none north of
           the tropics says so.
           El Nino onsets against the rest: Welch's t-test with the same false-discovery control; the share of the map
           that differs is recorded (the page says so when nothing does).
Odds       P(an onset begins l0..l1 days after a day in the given state) over the base rate, for the lead windows of
           LEADS and the states of STATES (the MJO phase on the cleaned wind-only RMM, the Siberian High, the East Asian
           trough, the jet's own phase, the Himalayan mountain torque, ENSO). Tested by permuting whole seasons of the
           state against the onsets (N_PERM draws, two-sided), Benjamini-Hochberg at ALPHA over every state and lead.
           The two box indices are plain anomalies from the normal, taken on the 2.5 degree grid the ensemble job
           holds its fields on, so that a forecast field gives the same number: their 366-slot normal and
           November-March sigma go into the output with the boxes, and that job (scripts/ensmaps/jet_upstream.py)
           standardises the live values with them.
Episodes   each onset's length in days and its peak PC1, the composite PC1 from EVO[0] to EVO[1] days around the
           onset, and the share of November-March days outside the extension phase that are followed by an onset
           within HORIZON days: the live event tracker's yardsticks.
ENSO       the share of November-March days in the extension phase by ENSO state, and the correlation over seasons of
           that share with the season's ONI: ENSO moves the number of extension days more than the odds of an
           onset.

    python src/build_pacjet_drivers.py            -> scripts/mjo/data/reference/pacjet_drivers.{npz,json}
    python src/build_pacjet_drivers.py --render   -> assets/sst/pacjet_drivers.webp (drawn in Actions by pacjet.py)
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from quantmet.degdays import doy366
from quantmet.stats import benjamini_hochberg
from scipy import stats
from scipy.ndimage import uniform_filter1d

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "lib")]
from latlon import G25N, box_mean, remap
from localdirs import env_dir

REF = HERE.parent / "data" / "reference"
SITE = HERE.parents[2] / "assets" / "sst"
NPZ = REF / "pacjet_drivers.npz"
OUT = REF / "pacjet_drivers.json"
ONI = SITE / "data" / "nino_history.json"
Y0, Y1 = 1991, 2020
SEASON = (11, 12, 1, 2, 3)
MIN_DAYS, QUIET = 3, 7
BACK = 120
WINDOWS = ((-12, -10), (-9, -7), (-6, -4), (-3, -1))
LEADS = ((1, 5), (4, 10), (6, 12), (8, 14))
EVO = (-10, 15)
HORIZON = 13  # the last day of a 16-day forecast (days 0-15) on which an onset can still be seen to last MIN_DAYS
ALPHA_MAP = 0.20  # the composites: false-discovery rate over each map
ALPHA = 0.20  # the odds table
N_PERM = 2000
MAP = dict(lat=(20.0, 80.0), lon=(40.0, 260.0))
STRIP = 30.0
# the two upstream indices: a box mean (cos-latitude weights), for the trough less the mean north of 20N
BOXES = {
    "sib": dict(field="msl", lat=(45.0, 60.0), lon=(60.0, 100.0), less=None),
    "eat": dict(field="z500", lat=(35.0, 50.0), lon=(115.0, 145.0), less=(20.0, 90.0)),
}
FIELDS = {"msl": ("slp", 900.0, 1100.0), "z500": ("z500", 4500.0, 6000.0)}  # store name, plausible mean (hPa, m)


def year_field(layer: str, var: str, year: int) -> xr.DataArray:
    with xr.open_dataset(env_dir("ERA5_STORE", layer, var, f"{var}_{year}.nc")) as ds:
        a = ds[next(iter(ds.data_vars))].transpose("time", "latitude", "longitude").load()
    a = a.assign_coords(longitude=np.round(a.longitude.values % 360, 3), latitude=np.round(a.latitude.values, 3))
    return a.sortby("latitude").sortby("longitude")


def load(field: str, t: pd.DatetimeIndex):
    """(lat, lon, values on MAP at 3 degrees, {box: its daily index}) of a Northern Hemisphere field."""
    var, lo, hi = FIELDS[field]
    maps, idx = [], {k: [] for k, b in BOXES.items() if b["field"] == field}
    for y in range(Y0, Y1 + 1):
        a = year_field("wb2_1p5_daily", var, y)
        if not lo < float(a.mean()) < hi:
            raise SystemExit(f"{var} {y}: mean {float(a.mean()):.1f} is not in {lo}-{hi}")
        g = np.stack([remap(x, a.latitude.values, a.longitude.values, G25N) for x in a.values])
        for k in idx:
            b = BOXES[k]
            idx[k].append(box_mean(g, G25N, b["lat"], b["lon"]) - (box_mean(g, G25N, b["less"]) if b["less"] else 0.0))
        m = a.sel(latitude=slice(*MAP["lat"]), longitude=slice(*MAP["lon"])).isel(
            latitude=slice(None, None, 2), longitude=slice(None, None, 2)
        )
        maps.append(m.values.astype("float32"))
        lat, lon = m.latitude.values, m.longitude.values
    v = np.concatenate(maps)
    if len(v) != len(t):
        raise SystemExit(f"{var}: {len(v)} days, the jet reference has {len(t)}")
    return lat, lon, v, {k: np.concatenate(x) for k, x in idx.items()}


def chi200(t: pd.DatetimeIndex):
    """(lat, lon, values) of the daily 200 hPa velocity potential, 30S-30N over MAP's longitudes, in 1e6 m2 s-1."""
    import walker_chi as wc

    rows = []
    for y in range(Y0, Y1 + 1):
        u, v = (year_field("wb2_1p5_daily_global", x, y) for x in ("u200", "v200"))
        for k in range(u.sizes["time"]):
            chi, lat, lon = wc.chi_strip(u.isel(time=k), v.isel(time=k))
            rows.append(chi)
        print(f"  chi200 {y}", flush=True)
    lat, lon = np.asarray(lat, float), np.asarray(lon, float) % 360
    keep = np.flatnonzero((lon >= MAP["lon"][0]) & (lon <= MAP["lon"][1]))
    keep = keep[np.argsort(lon[keep])]
    rows_ = np.flatnonzero(np.abs(lat) <= STRIP)
    rows_ = rows_[np.argsort(lat[rows_])]
    x = np.stack(rows)[:, rows_][:, :, keep] / 1e6
    if len(x) != len(t):
        raise SystemExit(f"chi200: {len(x)} days, the jet reference has {len(t)}")
    return lat[rows_], lon[keep], x.astype("float32")


def climatology(t: pd.DatetimeIndex, v: np.ndarray) -> np.ndarray:
    """The day-of-year normal of v (366 slots, 31-day smoothed around the year)."""
    dy = doy366(t.values)
    c = np.zeros((366, *v.shape[1:]))
    n = np.zeros(366)
    np.add.at(c, dy, v)
    np.add.at(n, dy, 1)
    return uniform_filter1d(c / n.reshape(-1, *[1] * (v.ndim - 1)), 31, axis=0, mode="wrap")


def normal(t: pd.DatetimeIndex, v: np.ndarray) -> np.ndarray:
    """v less its day-of-year normal."""
    return v - climatology(t, v)[doy366(t.values)]


def less_back(a: np.ndarray) -> np.ndarray:
    """a less its mean over the BACK days before each day (NaN in the first BACK days)."""
    cs = np.concatenate([np.zeros((1, *a.shape[1:])), np.cumsum(a, 0, dtype=np.float64)])
    out = np.full(a.shape, np.nan, np.float32)
    out[BACK:] = a[BACK:] - (cs[BACK:-1] - cs[: -BACK - 1]) / BACK
    return out


def onsets(phase: np.ndarray, t: pd.DatetimeIndex) -> list[tuple[int, int]]:
    """(first day, day after the last) of every extension run that counts as an onset."""
    x = phase == 0
    edge = np.flatnonzero(np.diff(np.concatenate([[0], x.astype(int), [0]])))
    first = BACK - WINDOWS[0][0]  # the earliest window needs its BACK days
    return [
        (a, b)
        for a, b in zip(edge[::2], edge[1::2])
        if b - a >= MIN_DAYS and t[a].month in SEASON and a >= first and not x[a - QUIET : a].any()
    ]


def fdr(p: np.ndarray, alpha: float) -> np.ndarray:
    ok = np.isfinite(p)
    out = np.zeros(p.shape, bool)
    out[ok] = benjamini_hochberg(p[ok], alpha)
    return out


def oni_daily(t: pd.DatetimeIndex) -> np.ndarray:
    h = json.loads(ONI.read_text())
    key = next(k for k in h["series"]["oni"] if k != "abs")
    s = pd.Series(h["series"]["oni"][key], index=pd.to_datetime(h["months"]), dtype=float)
    return s.reindex(t.to_period("M").to_timestamp()).to_numpy()


def mjo(t: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """(phase 1-8, amplitude) of the cleaned wind-only RMM (enso_rmm: ENSO and the IOD's growth removed)."""
    import enso_rmm as E

    with xr.open_dataset(REF / "rmm_era5.nc") as ds:
        r = ds[["rmm1_wind", "rmm2_wind"]].sel(time=t).load()
    dn, dd = E.daily_terms(t, E.ersst_n34(), E.ersst_dmi())
    s = np.outer(dn, E.load_c()) + np.outer(dd, E.load_d())
    x1, x2 = r["rmm1_wind"].values - s[:, 0], r["rmm2_wind"].values - s[:, 1]
    return E.phase_of(x1, x2), np.hypot(x1, x2)


def torque(t: pd.DatetimeIndex) -> np.ndarray:
    """The Himalaya/Tibet mountain torque in each month's sigma, as pacjet.py standardises the forecast's."""
    import build_torque_range_sd as BRS

    tq = BRS.day_anomalies()["HIM"].reindex(t).to_numpy(float)
    sc = json.loads((REF / "pacjet_torque.json").read_text())["torque_scale"]
    mean, sd = (np.array([sc[str(m)][k] for m in t.month]) for k in ("mean", "sd"))
    return (tq - mean) / sd


def rate(event: np.ndarray, state: np.ndarray) -> tuple[float, int]:
    """(share of the state's at-risk days followed by an onset, those days) on season x day grids."""
    m = np.isfinite(event) & (state > 0)
    return (event[m].mean() if m.any() else np.nan), int(m.sum())


def odds(state: dict[str, np.ndarray], onset: np.ndarray, t: pd.DatetimeIndex) -> tuple[list, dict]:
    """Base rate per lead window and, per state, its days, the ratio of its onset rate to the base rate and the
    permutation p, each per lead window."""
    n = len(t)
    season = np.where(t.month >= 7, t.year, t.year - 1)
    day = (t - pd.to_datetime([f"{s}-07-01" for s in season])).days.to_numpy()
    seasons = np.arange(Y0, Y1)
    inside = np.isin(season, seasons)

    def grid(x):
        g = np.full((len(seasons), 366), np.nan)
        g[season[inside] - Y0, day[inside]] = x[inside]
        return g

    rng = np.random.default_rng(0)
    perms = [rng.permutation(len(seasons)) for _ in range(N_PERM)]
    cs = np.concatenate([[0], np.cumsum(onset)])
    i = np.arange(n)
    base, res = [], {k: {"days": [], "ratio": [], "p": []} for k in state}
    for l0, l1 in LEADS:
        hit = (cs[np.clip(i + l1 + 1, 0, n)] - cs[np.clip(i + l0, 0, n)]) > 0
        risk = np.isin(t[np.clip(i + l0, 0, n - 1)].month, SEASON) & (i + l1 < n)
        E = grid(np.where(risk, hit.astype(float), np.nan))
        base.append(float(np.nanmean(E)))
        for k, c in state.items():
            C = grid(np.where(np.isfinite(c), c, np.nan).astype(float))

            obs, days = rate(E, C)
            null = np.array([rate(E, C[p])[0] for p in perms])
            null = null[np.isfinite(null)]
            p = (1 + (np.abs(null - null.mean()) >= abs(obs - null.mean())).sum()) / (1 + len(null))
            res[k]["days"].append(days)
            res[k]["ratio"].append(obs / base[-1])
            res[k]["p"].append(float(p))
    return base, res


def main() -> int:
    import pacjet_core as PC

    with xr.open_dataset(REF / "pacjet_ref.nc") as ds:
        ref = ds.load()
    t = pd.DatetimeIndex(ref.time.values)
    pc = ref.pc.values.astype(float)
    phase = PC.phase_of(pc[:, 0], pc[:, 1])
    ons = onsets(phase, t)
    first = np.array([a for a, _ in ons])
    oni = oni_daily(t)
    state_at = oni[first]
    nino = state_at >= 0.5
    print(f"{len(ons)} onsets: El Nino {nino.sum()}, La Nina {(state_at <= -0.5).sum()}", flush=True)
    season_days = np.isin(t.month, SEASON)

    # ── composites ────────────────────────────────────────────────────────────────────────────────────────────────
    out, index, clim, split = {}, {}, {}, {}
    for name in ("chi", "msl", "z500"):
        if name == "chi":
            lat, lon, v = chi200(t)
        else:
            lat, lon, v, idx = load(name, t)
            clim.update({k: climatology(t, x) for k, x in idx.items()})
            index.update({k: x - clim[k][doy366(t.values)] for k, x in idx.items()})
        a = less_back(normal(t, v))
        out[f"{name}_lat"], out[f"{name}_lon"] = lat, lon
        mean, share = [], []
        for l0, l1 in WINDOWS:
            e = np.stack([a[i + l0 : i + l1 + 1].mean(0) for i in first])
            _, p = stats.ttest_1samp(e, 0.0, axis=0)
            mean.append(np.where(fdr(p, ALPHA_MAP), e.mean(0), np.nan))
            _, p2 = stats.ttest_ind(e[nino], e[~nino], axis=0, equal_var=False)
            share.append(round(float(fdr(p2, ALPHA_MAP).mean()), 3))
        out[name] = np.stack(mean).astype("float32")
        split[name] = share
        print(
            f"  {name}: significant {[f'{np.isfinite(m).mean():.0%}' for m in mean]}, El Nino differs {share}",
            flush=True,
        )
    np.savez_compressed(NPZ, **out)

    # ── odds ──────────────────────────────────────────────────────────────────────────────────────────────────────
    sd = {k: float(x[season_days].std()) for k, x in index.items()}
    z = {k: x / sd[k] for k, x in index.items()}
    mph, amp = mjo(t)
    tq = torque(t)
    state = {f"mjo{a}{b}": np.isin(mph, (a, b)) & (amp >= 1) for a, b in ((8, 1), (2, 3), (4, 5), (6, 7))}
    state |= {
        "mjo_none": amp < 1,
        "sib_hi": z["sib"] >= 1,
        "sib_lo": z["sib"] <= -1,
        "eat_lo": z["eat"] <= -1,
        "eat_hi": z["eat"] >= 1,
        "pc2_lo": pc[:, 1] <= -1,
        "pc2_hi": pc[:, 1] >= 1,
        "pc1_lo": pc[:, 0] <= -1,
        "tq_hi": np.where(np.isfinite(tq), tq >= 1, np.nan),
        "tq_lo": np.where(np.isfinite(tq), tq <= -1, np.nan),
        "el_nino": np.where(np.isfinite(oni), oni >= 0.5, np.nan),
        "la_nina": np.where(np.isfinite(oni), oni <= -0.5, np.nan),
    }
    onset = np.zeros(len(t), bool)
    onset[first] = True
    base, res = odds(state, onset, t)
    sig = fdr(np.array([r["p"] for r in res.values()]), ALPHA).tolist()
    states = {
        k: {"days": r["days"][0], "ratio": [round(x, 2) for x in r["ratio"]], "sig": s}
        for (k, r), s in zip(res.items(), sig)
    }
    for k, r in states.items():
        print(
            f"  {k:9s} {r['days']:5d} d  "
            + "  ".join(f"{x:4.2f}{'*' if s else ' '}" for x, s in zip(r["ratio"], r["sig"]))
        )

    # ── episodes, ENSO ────────────────────────────────────────────────────────────────────────────────────────────
    lags = np.arange(EVO[0], EVO[1] + 1)
    evo = np.stack([pc[i + lags, 0] for i in first if i + EVO[1] < len(t)])
    seasons = np.where(t.month >= 7, t.year, t.year - 1)
    ext = pd.Series((phase == 0)[season_days].astype(float)).groupby(seasons[season_days]).mean()
    soni = pd.Series(oni[season_days]).groupby(seasons[season_days]).mean()
    full = pd.Series(season_days[season_days]).groupby(seasons[season_days]).sum() > 140  # drop the two half seasons
    r, p = stats.pearsonr(ext[full], soni[full])
    cs = np.concatenate([[0], np.cumsum(onset)])
    i = np.flatnonzero(season_days & (phase != 0) & (np.arange(len(t)) + HORIZON < len(t)))
    doc = {
        "period": [Y0, Y1],
        "months": list(SEASON),
        "onset": {"min_days": MIN_DAYS, "quiet_days": QUIET},
        "onsets": {
            "n": len(ons),
            "el_nino": int(nino.sum()),
            "la_nina": int((state_at <= -0.5).sum()),
            "neutral": int((np.abs(state_at) < 0.5).sum()),
        },
        "windows": [list(w) for w in WINDOWS],
        "back_days": BACK,
        "alpha_fdr": ALPHA_MAP,
        "alpha_odds": ALPHA,
        "el_nino_differs": split,
        "leads": [list(x) for x in LEADS],
        "base": [round(b, 4) for b in base],
        "n_perm": N_PERM,
        "states": states,
        "boxes": {k: {**b, "sd": round(sd[k], 3), "normal": np.round(clim[k], 2).tolist()} for k, b in BOXES.items()},
        "episodes": {
            "days": [int(b - a) for a, b in ons],
            "peak": [round(float(pc[a:b, 0].max()), 2) for a, b in ons],
        },
        "horizon": {"days": HORIZON, "p": round(float((cs[i + HORIZON + 1] > cs[i + 1]).mean()), 3)},
        "evolution": {
            "lag": lags.tolist(),
            **{
                k: np.round(np.percentile(evo, q, axis=0), 2).tolist() for k, q in (("lo", 25), ("mid", 50), ("hi", 75))
            },
        },
        "enso_days": {
            "seasons": int(full.sum()),
            "r": round(float(r), 2),
            "p": round(float(p), 4),
            "share": {
                k: round(float((phase == 0)[season_days & m].mean()), 3)
                for k, m in (("el_nino", oni >= 0.5), ("neutral", np.abs(oni) < 0.5), ("la_nina", oni <= -0.5))
            },
        },
    }
    OUT.write_text(json.dumps(doc, indent=1))
    print(f"ENSO and extension days: r {r:.2f} (p {p:.4f}), share {doc['enso_days']['share']}")
    print(f"wrote {NPZ} ({NPZ.stat().st_size / 1e3:.0f} kB) and {OUT.name}", flush=True)
    return 0


def render(out_dir=SITE) -> Path:
    """Draw the composite figure from the committed composites (runs in Actions with the live product)."""
    import matplotlib

    matplotlib.use("Agg")
    import house

    house.apply()
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm

    plt.rcParams.update({"font.size": 10.5, "font.family": "DejaVu Sans"})
    Z = np.load(NPZ)
    st = json.loads(OUT.read_text())
    proj, pc_ = ccrs.PlateCarree(central_longitude=150), ccrs.PlateCarree()
    south, north = -STRIP, MAP["lat"][1]
    mw = 6.1
    mh = mw * (north - south) / (MAP["lon"][1] - MAP["lon"][0])
    left, right, top, bot, gx, gy = 0.5, 0.2, 1.05, 1.5, 0.3, 0.42
    W, H = left + 2 * mw + gx + right, top + 2 * mh + gy + bot
    fig = plt.figure(figsize=(W, H))
    p_lev = np.array([-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6], float)
    c_lev = np.array([-5, -4, -3, -2, -1, -0.5, 0.5, 1, 2, 3, 4, 5], float)
    z_lev = [x for x in range(-90, 91, 15) if x]
    tropics = Z["chi_lat"] <= MAP["lat"][0]
    for k, (l0, l1) in enumerate(st["windows"]):
        r, c = divmod(k, 2)
        ax = fig.add_axes(
            [(left + c * (mw + gx)) / W, (bot + (1 - r) * (mh + gy)) / H, mw / W, mh / H], projection=proj
        )
        ax.set_extent([MAP["lon"][0], MAP["lon"][1], south, north], crs=pc_)
        ax.add_feature(cfeature.LAND, facecolor="#f3f1ec", zorder=0)
        cs_p = ax.contourf(
            Z["msl_lon"], Z["msl_lat"], Z["msl"][k], p_lev, cmap="RdBu_r", norm=BoundaryNorm(p_lev, 256, extend="both"),
            extend="both", transform=pc_, zorder=1,
        )  # fmt: skip
        cs_c = ax.contourf(
            Z["chi_lon"], Z["chi_lat"][tropics], Z["chi"][k][tropics], c_lev, cmap="BrBG_r", norm=BoundaryNorm(c_lev, 256, extend="both"),
            extend="both", transform=pc_, zorder=1,
        )  # fmt: skip
        zc = ax.contour(
            Z["z500_lon"], Z["z500_lat"], Z["z500"][k], z_lev, colors=house.INK, linewidths=0.9, transform=pc_, zorder=3
        )
        ax.clabel(zc, zc.levels[::2], fmt="%d", fontsize=7.5, inline_spacing=2)
        if not (np.isfinite(Z["msl"][k]).any() or np.isfinite(Z["z500"][k]).any()):
            ax.text(
                190, 35, "no significant change north of 20°N", transform=pc_, ha="center", va="center", fontsize=8.6,
                color=house.MUTED, zorder=5,
            )  # fmt: skip
        ax.coastlines(linewidth=0.55, color="#55524c", zorder=2)
        ax.plot([MAP["lon"][0], MAP["lon"][1]], [MAP["lat"][0]] * 2, color=house.EDGE, lw=0.8, transform=pc_, zorder=4)
        ax.set_title(f"{-l0} to {-l1} days before the onset", fontsize=11, loc="left", pad=4)
        gl = ax.gridlines(
            draw_labels=True, linewidth=0.3, color=house.GRID, xlocs=(60, 100, 140, 180, -140, -100), ylocs=(0, 30, 60)
        )
        gl.top_labels = gl.right_labels = False
        gl.left_labels, gl.bottom_labels = c == 0, r == 1
        gl.xlabel_style = gl.ylabel_style = {"size": 8, "color": house.MUTED}
    n = st["onsets"]["n"]
    fig.text(left / W, 1 - 0.36 / H, "Before a North Pacific jet extension", fontsize=15, weight="bold", ha="left")
    fig.text(
        left / W, 1 - 0.66 / H,
        f"ERA5 {st['period'][0]}–{st['period'][1]}, {n} onsets, November to March: what changed from the {st['back_days']} "
        "days before",
        fontsize=10.5, color=house.MUTED, ha="left",
    )  # fmt: skip
    bw = 0.36
    for x0, cs, lev, label in (
        (0.03, cs_c, c_lev, "South of 20°N: 200 hPa velocity potential (10⁶ m² s⁻¹); green = more rising air and rain"),
        (0.53, cs_p, p_lev, "North of 20°N: sea-level pressure (hPa); lines: 500 hPa height (every 15 m, dashed = lower)"),
    ):  # fmt: skip
        cax = fig.add_axes([x0 + (0.47 - bw) / 2, 0.78 / H, bw, 0.13 / H])
        cb = fig.colorbar(cs, cax=cax, orientation="horizontal", ticks=lev)
        cb.ax.tick_params(labelsize=8)
        house.spaced(cb.ax.xaxis)
        fig.text(x0 + 0.235, 1.0 / H, label, fontsize=8.6, ha="center", va="bottom")
    foot = (
        f"Colours and lines only where the mean over the onsets differs from zero (t-test, false-discovery rate "
        f"{st['alpha_fdr']:.0%} over each map). Onset: first of at least {st['onset']['min_days']} days in the extension "
        f"phase after {st['onset']['quiet_days']} days without one."
    )
    fig.text(left / W, 0.12 / H, textwrap.fill(foot, 175), fontsize=8.2, color=house.MUTED, ha="left", va="bottom")
    out = Path(out_dir) / "pacjet_drivers.webp"
    fig.savefig(out, dpi=105, facecolor="white", pil_kwargs={"quality": 88, "method": 6})
    plt.close(fig)
    return out


if __name__ == "__main__":
    if "--render" in sys.argv:
        i = sys.argv.index("--render")
        print(render(sys.argv[i + 1] if len(sys.argv) > i + 1 else SITE))
        sys.exit(0)
    sys.exit(main())
