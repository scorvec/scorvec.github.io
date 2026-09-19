#!/usr/bin/env python3
"""
Oceanic Kelvin wave tracker: equatorial Pacific sea level, the winds that launch the waves, and when the
current waves reach the eastern Pacific.

A westerly wind burst in the western Pacific launches a DOWNWELLING Kelvin wave: a bulge of warm water,
5-15 cm of sea level and 10-30 m of thermocline, that runs east along the equator at ~2.5 m/s (the
basin in about two months) and warms Nino-3 and the South American coast when it arrives. Easterly
surges launch the cold, UPWELLING mirror image.

  sea level   NOAA CoastWatch blended altimetry SLA (daily, 0.5 deg sampling), anomaly against its own
              2017-2025 mean + trend + annual harmonics (build_sla_clim.py); averaged 2S-2N
  wind        10 m zonal wind anomaly, 5S-5N, against the ERA5 1991-2020 climatology the equatorial wind
              Hovmoller uses: NCEI Blended Sea Winds (observed, ~2-3 weeks behind), then the AIFS-ENS
              0-h analyses (kept daily in eq_u10_analysis_history.nc), then the AIFS-ENS ensemble-mean
              forecast to day 15
  filter      eastward waves with phase speeds 1.5-3.5 m/s and periods 15-150 days, over the Pacific
              interior (the basin is not periodic: tapered and zero-padded in longitude)
  tracking    every current crest (|filtered| > 1.5 cm, about one standard deviation of the band) is followed back day by day; its speed is the fit
              of its track, and it is projected to 120W (Nino-3) and 90W (the coast)

    python ocean_kelvin.py --selftest
    python ocean_kelvin.py --date 20260918 --time 12
    python ocean_kelvin.py --backfill-analyses 40 --date 20260918    # rebuild the AIFS 0-h wind history
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import textwrap
import time as _time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
REF = HERE.parent / "data" / "reference"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "scripts" / "sst"))
sys.path.insert(0, str(REPO / "scripts" / "ecmwf"))
from build_sla_clim import fetch as fetch_sla, evaluate as sla_clim_eval       # noqa: E402

SLA_CLIM = REF / "sla_clim.nc"
U_CLIM = REF / "eq_u10_clim.nc"
U_HIST = REF / "eq_u10_analysis_history.nc"
NBS = "https://coastwatch.noaa.gov/erddap/griddap/noaacwBlendedWindsDaily.nc"
M_PER_DEG, DAY = 111320.0, 86400.0
C_MIN, C_MAX, P_MIN, P_MAX = 1.5, 3.5, 15.0, 150.0
FILT_LON = (130.0, 280.0)                     # Pacific interior used by the filter
AMP = 0.015                                   # crest threshold, metres: ~1 sd of the Kelvin band (1.3-1.8 cm in 2026)
DAYS_BACK, DAYS_AHEAD, MAP_DAYS = 180, 60, 60
TARGETS = ((240.0, "Niño-3 (120°W)"), (270.0, "the coast (90°W)"))
U_LAT = 5.0


# ---- the filter ------------------------------------------------------------------------------ #
def ocean_kelvin_filter(x: np.ndarray, dlon: float, pad_t: int = 120, ramp: int = 20) -> np.ndarray:
    """x (time, lon) in a non-periodic basin -> the eastward 1.5-3.5 m/s, 15-150 day part."""
    nt, nx = x.shape
    x = np.where(np.isfinite(x), x, 0.0)
    x = x - x.mean(0, keepdims=True)
    # time: ramp the last row to zero, then zeros; soft start
    r = x[-1][None] * (1 - np.arange(1, ramp + 1) / (ramp + 1))[:, None]
    xp = np.concatenate([x, r, np.zeros((pad_t - ramp, nx))], 0)
    n0 = max(2, int(0.1 * nt))
    xp[:n0] *= (0.5 - 0.5 * np.cos(np.pi * np.arange(n0) / n0))[:, None]
    # longitude: 10% Tukey taper at both coasts, then zero-pad to twice the width
    tw = max(2, int(0.1 * nx))
    taper = np.ones(nx); ramp_x = 0.5 - 0.5 * np.cos(np.pi * np.arange(tw) / tw)
    taper[:tw] = ramp_x; taper[-tw:] = ramp_x[::-1]
    xp = np.concatenate([xp * taper[None], np.zeros((xp.shape[0], nx))], 1)
    NT, NX = xp.shape
    f = np.fft.fftfreq(NT, d=1.0)                          # cycles/day
    s = np.fft.fftfreq(NX, d=dlon)                         # cycles/degree
    F, S = np.meshgrid(f, s, indexing="ij")
    # a crest moves east when f and s have opposite signs (numpy's inverse uses exp(+2 pi i (f t + s x)))
    east = (F * S) < 0
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.abs(F / S) * M_PER_DEG / DAY                # phase speed, m/s
    af = np.abs(F)
    m = east & (c >= C_MIN) & (c <= C_MAX) & (af >= 1 / P_MAX) & (af <= 1 / P_MIN)
    y = np.real(np.fft.ifft2(np.fft.fft2(xp) * m))
    return y[:nt, :nx]


def selftest() -> int:
    dlon = 0.5
    lon = np.arange(0, 150, dlon); t = np.arange(240)[:, None]
    deg_day = 2.5 * DAY / M_PER_DEG                        # 2.5 m/s in degrees/day
    k = 2 * np.pi / 60.0                                   # 60-degree wavelength
    east = np.cos(k * (lon[None, :] - deg_day * t))
    west = np.cos(k * (lon[None, :] + deg_day * t))
    slow = np.cos(k * (lon[None, :] - 0.4 * t))           # eastward but ~0.5 m/s: not a Kelvin wave
    for name, x, keep in (("eastward 2.5 m/s", east, True), ("westward 2.5 m/s", west, False), ("eastward 0.5 m/s", slow, False)):
        y = ocean_kelvin_filter(x, dlon)
        core = (slice(40, -40), slice(40, -40))
        r = np.std(y[core]) / np.std(x[core])
        ok = r > 0.6 if keep else r < 0.15
        print(f"  {name}: kept {r:.2f} of the amplitude -> {'ok' if ok else 'FAIL'}")
        if not ok:
            return 1
    print("selftest passed")
    return 0


# ---- tracking -------------------------------------------------------------------------------- #
def extrema(row: np.ndarray, lon: np.ndarray, sign: int, lo=140.0, hi=265.0) -> list[int]:
    v = sign * row
    idx = []
    for i in range(1, len(row) - 1):
        if lo <= lon[i] <= hi and v[i] > AMP and v[i] >= v[i - 1] and v[i] >= v[i + 1]:
            idx.append(i)
    return idx


def track(filt: np.ndarray, lon: np.ndarray, sign: int, i0: int, back: int = 25):
    """Follow the crest at lon index i0 on the last day back in time; -> (speed m/s, n days, lons)."""
    pos = [lon[i0]]; cur = lon[i0]
    for k in range(1, back + 1):
        row = filt[-1 - k]
        cand = [lon[i] for i in extrema(row, lon, sign, 120.0, 285.0)] if True else []
        # yesterday the crest was 0.6-3.6 deg further WEST (1.5-3.5 m/s is 1.2-2.7 deg/day; allow slack)
        cand = [c for c in cand if 0.3 <= cur - c <= 4.0]
        if not cand:
            break
        cur = max(cand); pos.append(cur)
    if len(pos) < 8:
        return 2.5, len(pos), pos
    t = -np.arange(len(pos)); slope = np.polyfit(t, pos, 1)[0]            # degrees/day
    c = float(np.clip(slope * M_PER_DEG / DAY, C_MIN, C_MAX))
    return c, len(pos), pos


# ---- data ------------------------------------------------------------------------------------ #
def sla_recent(end: pd.Timestamp, days: int) -> xr.DataArray:
    """SLA anomaly (time, lat, lon) in metres for the last `days` days to the newest available."""
    clim = xr.open_dataset(SLA_CLIM)
    s = fetch_sla(f"{end - pd.Timedelta(days=days):%Y-%m-%d}", "last")          # altimetry runs ~1-2 days behind
    if not (np.allclose(s.latitude, clim.latitude) and np.allclose(s.longitude, clim.longitude)):
        s = s.interp(latitude=clim.latitude, longitude=clim.longitude)
    t = pd.DatetimeIndex(s.time.values).normalize()
    a = s.values - sla_clim_eval(clim.coef.values, t)
    return xr.DataArray(a, dims=s.dims, coords={"time": t, "latitude": s.latitude.values, "longitude": s.longitude.values})


def u_clim(doy) -> np.ndarray:
    from build_eq_wind_clim import eval_clim
    return eval_clim(xr.open_dataarray(U_CLIM).values, doy)                # (t, 360) on 0..359


def nbs_band(t0: pd.Timestamp) -> xr.DataArray:
    """NCEI Blended Sea Winds 10 m u, 5S-5N cos-weighted, 1 deg, from t0 to the newest day published."""
    q = (f"{NBS}?u_wind%5B({t0:%Y-%m-%d}T09:00:00Z):1:(last)%5D%5B(10.0)%5D"
         f"%5B(-{U_LAT}):4:({U_LAT})%5D%5B(0.0):4:(359.75)%5D")
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
        urllib.request.urlretrieve(q, f.name)
        d = xr.open_dataset(f.name).load()
    u = d["u_wind"].squeeze("zlev", drop=True)
    w = np.cos(np.deg2rad(u.latitude))
    b = u.weighted(w).mean("latitude")
    b = b.assign_coords(time=pd.DatetimeIndex(b.time.values).normalize()).interp(longitude=np.arange(360.0))
    return b


def aifs_step0_band(date: str, hh: str) -> np.ndarray:
    """AIFS-ENS control 0-h 10 m u, 5S-5N band on the 1-degree grid."""
    import store as ecmwf
    p = ecmwf.ensure(ecmwf.Cycle(date, hh), ecmwf.Spec("aifs-ens", "cf", "10u", "sfc", (), (0,)))
    u = xr.open_dataset(p, engine="cfgrib", backend_kwargs={"indexpath": ""})
    u = u[list(u.data_vars)[0]].squeeze()
    if float(u.longitude.min()) < 0:
        u = u.assign_coords(longitude=u.longitude % 360).sortby("longitude")
    u = u.sortby("latitude").sel(latitude=slice(-U_LAT, U_LAT))
    return u.weighted(np.cos(np.deg2rad(u.latitude))).mean("latitude").interp(longitude=np.arange(360.0)).values


def update_u_history(date: str, hh: str, vals: np.ndarray | None = None) -> xr.DataArray:
    t = pd.Timestamp(f"{date[:4]}-{date[4:6]}-{date[6:]}T{int(hh):02d}:00")
    v = aifs_step0_band(date, hh) if vals is None else vals
    cur = xr.DataArray(v[None].astype("float32"), dims=("time", "longitude"), coords={"time": [t], "longitude": np.arange(360.0)}, name="u10")
    if U_HIST.exists():
        old = xr.open_dataarray(U_HIST).load()
        old = old.sel(time=~old.time.isin([t]))
        cur = xr.concat([old, cur], "time").sortby("time")
    cur = cur.isel(time=slice(-400, None))
    tmp = U_HIST.with_suffix(".tmp.nc"); cur.to_netcdf(tmp); tmp.replace(U_HIST)
    return cur


def u_forecast(date: str, hh: str):
    """AIFS-ENS ensemble-mean 10 m u anomaly along the equator, days 1-15 (the Hovmoller's own path)."""
    import eq_hovmoller as EH
    ens = EH.ensemble_mean_band(EH.download("aifs", date, hh))
    anom, valid = EH.anomalize(ens, pd.Timestamp(f"{date[:4]}-{date[4:6]}-{date[6:]}T{int(hh):02d}:00"))
    return pd.DatetimeIndex(valid).normalize(), anom                        # (n, 360) on EH.LON_GRID


# ---- rendering ------------------------------------------------------------------------------- #
def render_hov(days, lon, sla_eq, filt, wdays, wlon, uanom, u_src_edges, today, waves, out: Path, sla_last: pd.Timestamp):
    fig = plt.figure(figsize=(14.0, 11.6))
    axL = fig.add_axes([0.06, 0.235, 0.52, 0.63]); axR = fig.add_axes([0.66, 0.235, 0.27, 0.63], sharey=axL)
    y = mdates(days); lev = np.arange(-40, 41, 5)
    cf = axL.contourf(lon, y, sla_eq * 100, levels=lev, cmap="RdBu_r", norm=BoundaryNorm(lev, 256, extend="both"), extend="both")
    axL.contour(lon, y, filt * 100, levels=[1.5, 3, 4.5, 6], colors="#1a1a1a", linewidths=[0.7, 1.1, 1.5, 1.9])
    axL.contour(lon, y, filt * 100, levels=[-6, -4.5, -3, -1.5], colors="#1a1a1a", linewidths=[1.9, 1.5, 1.1, 0.7], linestyles="--")
    horizon = mdates([today + pd.Timedelta(days=DAYS_AHEAD)])[0]
    for w in waves:                                         # projections
        deg_day = w["speed"] * DAY / M_PER_DEG
        t_end = (270.0 - w["lon"]) / deg_day
        tt = np.linspace(0, t_end, 20)
        axL.plot(w["lon"] + deg_day * tt, mdates([sla_last])[0] + tt, color="#b3122b" if w["sign"] > 0 else "#1f4e9c", lw=2.2, ls=(0, (2, 2)))
        for tl, name in TARGETS:
            if tl > w["lon"] and w.get(f"eta_{int(tl)}"):
                axL.plot(tl, mdates([pd.Timestamp(w[f"eta_{int(tl)}"])])[0], "o", ms=7, mfc="white", mec="#b3122b" if w["sign"] > 0 else "#1f4e9c", mew=2)
    ul = np.arange(-9, 10, 1.5)
    cu = axR.contourf(wlon, mdates(wdays), uanom, levels=ul, cmap="PuOr_r", norm=BoundaryNorm(ul, 256, extend="both"), extend="both")
    for ax in (axL, axR):
        ax.axhline(mdates([today])[0] + 0.5, color="#1c2430", lw=1.8)
        ax.set_xlim(FILT_LON[0], 280.0)
        ax.set_xticks([140, 160, 180, 200, 220, 240, 260, 280])
        ax.set_xticklabels(["140°E", "160°E", "180°", "160°W", "140°W", "120°W", "100°W", "80°W"], fontsize=8.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axL.set_ylim(horizon, y[0])
    ticks = pd.date_range(days[0].normalize() + pd.offsets.MonthBegin(0), today + pd.Timedelta(days=DAYS_AHEAD), freq="MS")
    axL.set_yticks(mdates(ticks)); axL.set_yticklabels([f"{d:%b %Y}" if d.month == 1 else f"{d:%b}" for d in ticks], fontsize=9)
    plt.setp(axR.get_yticklabels(), visible=False)
    lab = dict(fontsize=8.5, fontweight="bold", color="#1c2430", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    axL.text(278, mdates([today])[0] - 1, "▲ observed", ha="right", va="bottom", **lab)
    axL.text(278, mdates([today])[0] + 2, "▼ projected at each wave's own speed", ha="right", va="top", **lab)
    axR.text(278, mdates([today])[0] + 2, "▼ AIFS-ENS forecast", ha="right", va="top", **lab)
    for e, name in u_src_edges:
        axR.axhline(mdates([e])[0], color="#6f6b64", lw=0.8, ls=":")
        axR.text(131, mdates([e])[0] - 0.8, name, fontsize=7.5, color="#6f6b64", va="bottom")
    axL.set_title("Sea level anomaly, 2°S–2°N (shading, cm); black: Kelvin-band part every 1.5 cm (solid +, dashed −)", loc="left", fontsize=10, fontweight="bold")
    axR.set_title("10 m zonal wind anomaly, 5°S–5°N (m/s)", loc="left", fontsize=10, fontweight="bold")
    c1 = fig.add_axes([0.06, 0.18, 0.52, 0.013]); fig.colorbar(cf, cax=c1, orientation="horizontal").set_label("sea level anomaly (cm)", fontsize=8.5)
    c2 = fig.add_axes([0.66, 0.18, 0.27, 0.013]); fig.colorbar(cu, cax=c2, orientation="horizontal").set_label("westerly (+) / easterly (−) anomaly (m/s)", fontsize=8.5)
    fig.text(0.06, 0.975, "Ocean Kelvin waves in the equatorial Pacific: where they are and when they arrive", fontsize=13.5, fontweight="bold", va="top")
    fig.text(0.06, 0.948, textwrap.fill("A westerly burst (orange, right) launches a downwelling Kelvin wave: a warm bulge of sea level that runs east "
             "at ~2.5 m/s and warms Niño-3 and the coast when it arrives (red streak tilting down to the right, left). Easterly surges launch "
             "the cold mirror image. Dashed lines: each current wave projected at its own measured speed; circles: arrival at 120°W and 90°W.", 175),
             fontsize=8.8, color="#444", va="top", linespacing=1.3)
    rows = [f"{'downwelling (warm)' if w['sign'] > 0 else 'upwelling (cold)':20s} at {lonlab(w['lon']):>6s}  {w['amp_cm']:+5.1f} cm  "
            f"{w['speed']:.1f} m/s ({'tracked ' + str(w['track_days']) + ' d' if w['track_days'] >= 8 else 'default speed'})  "
            + "  ".join(f"→ {name}: {pd.Timestamp(w[f'eta_{int(tl)}']):%b %-d}" for tl, name in TARGETS if w.get(f"eta_{int(tl)}"))
            for w in waves] or ["no Kelvin-band crest above 1.5 cm in the basin today"]
    fig.text(0.06, 0.128, "Waves in the basin on " + f"{sla_last:%b %-d}:", fontsize=9.5, fontweight="bold", va="top")
    fig.text(0.06, 0.108, "\n".join(rows[:4]), fontsize=8.8, family="monospace", va="top", color="#1c2430", linespacing=1.45)
    fig.text(0.06, 0.012, textwrap.fill("Sea level: NOAA CoastWatch blended altimetry, anomaly vs its own 2017–2025 mean, trend and seasonal cycle. "
             "Wind: NCEI Blended Sea Winds, then AIFS-ENS 0-h analyses, then the AIFS-ENS ensemble-mean forecast; anomalies vs ERA5 1991–2020. "
             "Kelvin band: eastward, 1.5–3.5 m/s, 15–150 days.", 200), fontsize=7.6, color="#6f6b64", va="bottom", linespacing=1.3)
    fig.savefig(out, dpi=110, facecolor="white", pil_kwargs={"quality": 86, "method": 6})
    plt.close(fig)


def mdates(ds):
    import matplotlib.dates as md
    return md.date2num(pd.DatetimeIndex(ds).to_pydatetime())


def lonlab(l):
    return f"{l:.0f}°E" if l <= 180 else f"{360 - l:.0f}°W"


def render_map(a2d, lat, lon, day, crests, out: Path):
    import mapstyle as MS
    fig, ax, H, pc = MS.open_map(extent=[120, 290, -15, 15], central=180, width=14.0, top=0.78, bot=0.95)
    lev = np.arange(-40, 41, 5)
    cf = ax.contourf(lon, lat, a2d * 100, levels=lev, cmap="RdBu_r", norm=BoundaryNorm(lev, 256, extend="both"), extend="both", transform=pc, zorder=1)
    MS.features(ax, states=False, gridlines=False)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="#6f6b64", alpha=0.5, xlocs=range(-180, 181, 20), ylocs=[-10, 0, 10], zorder=4)
    gl.top_labels = gl.right_labels = False; gl.xlabel_style = gl.ylabel_style = {"size": 7.5, "color": "#444"}
    for c, sign in crests:
        ax.annotate("K", xy=(c, 0), xycoords=pc._as_mpl_transform(ax), ha="center", va="center", fontsize=10.5, fontweight="bold", color="white",
                    zorder=9, bbox=dict(boxstyle="circle,pad=0.25", fc="#b3122b" if sign > 0 else "#1f4e9c", ec="white", lw=1.2))
    MS.heading(fig, H, f"Equatorial Pacific sea level anomaly · {day:%a %d %b %Y}",
               "NOAA CoastWatch blended altimetry, anomaly against its own 2017–2025 mean, trend and seasonal cycle · "
               "K = Kelvin wave crest on the equator (red: downwelling/warm, blue: upwelling/cold)", title_size=12.5, sub_size=8.2, wrap=190)
    MS.colorbar(fig, H, cf, "sea level anomaly (cm)")
    MS.save(fig, out, dpi=118)


# ---- main ------------------------------------------------------------------------------------ #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date"); ap.add_argument("--time", default="00")
    ap.add_argument("--out", default=str(REPO / "assets/sst/ocean_kelvin.webp"))
    ap.add_argument("--anim-dir", default=str(REPO / "assets/sst/anim/okelvin"))
    ap.add_argument("--manifest", default=str(REPO / "assets/sst/anim/okelvin_manifest.json"))
    ap.add_argument("--json", default=str(REPO / "assets/sst/data/ocean_kelvin.json"))
    ap.add_argument("--selftest", action="store_true"); ap.add_argument("--backfill-analyses", type=int, default=0)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if selftest():
        raise SystemExit("filter self-test failed; not rendering")
    today = pd.Timestamp(f"{a.date[:4]}-{a.date[4:6]}-{a.date[6:]}")
    if a.backfill_analyses:
        for k in range(a.backfill_analyses, -1, -1):
            d = today - pd.Timedelta(days=k)
            for hh in ("00", "12"):
                try:
                    update_u_history(f"{d:%Y%m%d}", hh); print(f"  u10 analysis {d:%Y-%m-%d} {hh}Z", flush=True)
                except Exception as e:                               # noqa: BLE001
                    print(f"  {d:%Y-%m-%d} {hh}Z: {type(e).__name__} {str(e)[:60]}", flush=True)
        return 0
    # ---- sea level
    sla = sla_recent(today, DAYS_BACK)
    sla_last = pd.Timestamp(sla.time.values[-1])
    full = pd.date_range(sla.time.values[0], sla_last, freq="D")
    sla = sla.reindex(time=full).interpolate_na("time", limit=4)
    lat, lon = sla.latitude.values, sla.longitude.values
    eq = sla.where(np.abs(sla.latitude) <= 2.2).mean("latitude").values                       # (t, lon)
    m = (lon >= FILT_LON[0]) & (lon <= FILT_LON[1])
    filt = np.full_like(eq, np.nan)
    sub = eq[:, m]
    good = np.isfinite(sub).mean(0) > 0.8
    subf = np.where(np.isfinite(sub), sub, np.nanmean(sub, 0, keepdims=True))
    filt_sub = ocean_kelvin_filter(subf, float(np.diff(lon[:2])[0]))
    filt_sub[:, ~good] = np.nan
    filt[:, m] = filt_sub
    # ---- waves in the basin now
    waves = []
    fl = np.nan_to_num(filt)
    for sign in (1, -1):
        for i in extrema(fl[-1], lon, sign):
            c, n, _ = track(fl, lon, sign, i)
            w = {"sign": sign, "lon": float(lon[i]), "amp_cm": round(float(fl[-1, i] * 100), 1), "speed": round(c, 2), "track_days": n}
            deg_day = c * DAY / M_PER_DEG
            for tl, _name in TARGETS:
                if tl > lon[i]:
                    w[f"eta_{int(tl)}"] = f"{sla_last + pd.Timedelta(days=float((tl - lon[i]) / deg_day)):%Y-%m-%d}"
            waves.append(w)
    waves.sort(key=lambda w: -abs(w["amp_cm"]))
    print(f"  ocean kelvin: SLA {full[0]:%Y-%m-%d}..{sla_last:%Y-%m-%d}; waves: " +
          "; ".join(f"{'+' if w['sign'] > 0 else '-'}{abs(w['amp_cm'])}cm {lonlab(w['lon'])} {w['speed']}m/s" for w in waves), flush=True)
    # ---- wind: observed (NBS), AIFS analyses, AIFS forecast
    edges = []
    try:
        nb = nbs_band(full[0])
        nb_days = pd.DatetimeIndex(nb.time.values)
        nb_anom = nb.values - u_clim(nb_days.dayofyear.values)
        edges.append((nb_days[-1] + pd.Timedelta(days=1), "↓ AIFS-ENS 0-h analyses"))
    except Exception as e:                                           # noqa: BLE001
        print(f"  NBS winds unavailable ({type(e).__name__}: {str(e)[:60]})", flush=True)
        nb_days, nb_anom = pd.DatetimeIndex([]), np.zeros((0, 360))
    hist = update_u_history(a.date, a.time)
    hd = hist.groupby(hist.time.dt.floor("D")).mean("time").rename({"floor": "time"})
    hdays = pd.DatetimeIndex(hd.time.values)
    keep = hdays > (nb_days[-1] if len(nb_days) else full[0] - pd.Timedelta(days=1))
    h_anom = hd.values[keep] - u_clim(hdays[keep].dayofyear.values)
    fdays, f_anom = u_forecast(a.date, a.time)
    fk = fdays > today
    wdays = pd.DatetimeIndex(list(nb_days) + list(hdays[keep]) + list(fdays[fk]))
    wanom = np.concatenate([nb_anom, h_anom, f_anom[fk]], 0)
    wd = xr.DataArray(wanom, dims=("time", "lon"), coords={"time": wdays, "lon": np.arange(360.0)})
    wd = wd.groupby("time").mean().reindex(time=pd.date_range(wdays.min(), wdays.max(), freq="D")).interpolate_na("time", limit=3)
    wd = wd.sel(time=slice(full[0], None))
    # ---- figures
    days = pd.DatetimeIndex(full)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    render_hov(days, lon, eq, filt, pd.DatetimeIndex(wd.time.values), wd.lon.values, wd.values, edges, today, waves, out, sla_last)
    anim = Path(a.anim_dir); anim.mkdir(parents=True, exist_ok=True)
    for old in anim.glob("F*.webp"):
        old.unlink()
    frames = []
    pick = [i for i in range(len(days)) if days[i] >= sla_last - pd.Timedelta(days=MAP_DAYS) and (sla_last - days[i]).days % 2 == 0]
    for n, i in enumerate(pick):
        cr = [(float(lon[j]), s) for s in (1, -1) for j in extrema(fl[i], lon, s)]
        fp = anim / f"F{n:02d}.webp"
        render_map(sla.values[i], lat, lon, days[i], cr, fp)
        frames.append({"idx": n, "file": fp.name, "date": f"{days[i]:%Y-%m-%d}", "label": f"{days[i]:%a %b %d}"})
    Path(a.manifest).write_text(json.dumps({"ver": int(_time.time()), "regions": {"okelvin": {
        "label": "Equatorial Pacific sea level anomaly (altimetry)", "n_frames": len(frames), "frames": frames}}}))
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps({"sla_through": f"{sla_last:%Y-%m-%d}", "waves": waves,
                                        "wind_sources": [n for _, n in edges]}, separators=(",", ":")))
    print(f"  ocean kelvin: Hovmoller + {len(frames)} map frames", flush=True)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush(); sys.stderr.flush()
    import os
    os._exit(rc or 0)
