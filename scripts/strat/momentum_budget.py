#!/usr/bin/env python3
"""Live zonal-momentum budget of the stratosphere from GEOS FP (2026-09-27; user: "A live budget is something worth
putting together"): what is driving the polar vortex and the tropical upwelling.

TERMS. Transformed-Eulerian-mean zonal momentum equation in pressure coordinates (Andrews, Holton and Leovy 1987):
    du/dt = fhat v*  -  omega* du/dp  +  (1/(a cos phi)) div F  +  GWD  +  ANA  +  residual
    fhat   = f - (1/(a cos phi)) d(u cos phi)/d phi
    v*     = v - d/dp([v'th'] / th_p)                     omega* = omega + (1/(a cos phi)) d/dphi(cos phi [v'th'] / th_p)
    F_phi  = a cos phi (u_p [v'th']/th_p - [u'v'])        F_p    = a cos phi (fhat [v'th']/th_p - [u'omega'])
  "Coriolis" = fhat v*, "vertical advection" = -omega* du/dp (= -w* du/dz in log-pressure height), "resolved waves" = the
  EP-flux divergence (also by zonal wavenumber 1, 2 and 3+, from the same mean state), GWD = GEOS FP's parameterized
  gravity-wave drag (orographic + non-orographic; the collection holds only the total), ANA = the analysis-increment
  tendency (IAU), residual = everything else: moist and turbulent tendencies, numerics and the sampling error of the
  snapshot eddy fluxes. du/dt is observed: [u](next 00Z) - [u](00Z) over one day.
  Because the model's own tendencies are published (DUDTDYN, DUDTGWD, DUDTANA, DUDTMST, DUDTTRB), the budget is checked
  twice: the TEM dynamical terms against the zonal mean of DUDTDYN, and du/dt against the sum of all five tendencies.

DOWNWARD CONTROL. Rearranged, v* = -(G_EP + G_GWD + G_ANA + G_rest - du/dt - omega* u_p ... ) / fhat; integrated from the
  top it gives psi* for each forcing G separately (linear), and the tropical upward mass flux between the turnaround
  latitudes (bdc_dc.py's MERRA-2 day-of-year table) splits into resolved-wave, gravity-wave, analysis-increment, non-steady
  (-du/dt) and "other" (vertical advection + residual) parts. Their sum equals the flux implied by the analysed v* by
  construction; the independent check is w* from the analysed omega*. Mean w* over the upwelling region in log-pressure
  coordinates: w* = F g H / (p 2 pi a^2 (sin phi_N - sin phi_S)), H = 7 km (mm/s).

DATA (NASA NCCS OPeNDAP, sequential reads, every field validated; a parallel OPeNDAP read has silently returned zeros).
  State: assim/inst3_3d_asm_Np u, v, T, omega at every 3-hourly time 00..21Z and the next 00Z, 0.1-200 hPa, every 4th
    grid point (1 x 1.25 deg). Daily means of every term are trapezoid means of the nine snapshots (00Z..24Z), each term
    computed per snapshot. Three-hourly, not six-hourly: the zonal-mean terms swing by +-5-10 m/s/day within a day (tides,
    inertial oscillations near 14 h at 60 deg), and on 2026-09-20 six-hourly sampling missed the model's own dynamical
    tendency by 0.2-0.3 m/s/day at 5 hPa and 1 hPa where three-hourly came within 0.05-0.15. Every 4th longitude is
    enough for the eddy fluxes: full resolution moved no 55-65 deg term by more than 0.3 m/s/day at 1 hPa, 0.1 at 10 hPa.
  Tendencies: assim/tavg3_3d_udt_Nv, all EIGHT 3-hourly windows of the day (exact 00-24Z mean), native levels 5-43
    (pure pressure above 164 hPa) interpolated in log p to the same levels, every 4th latitude; GWD at full longitude
    resolution (spotty orographic drag: at 60S 10 hPa sampling every 1.25 deg read -0.082 against -0.034 m/s/day at full
    resolution on 2026-09-20), the others every 2nd longitude (0.625 deg, within 0.3 % of full resolution). The Nv
    collection runs ~12 h behind real time, the 0.5-deg Cp copy ~21 h (checked 2026-09-27), hence Nv.
  Forecast: fcast/inst3_3d_asm_Np.<YYYYMMDD_00>, 00/06/12/18Z to 240 h. The forecast collections carry NO tendencies, so
    the forecast days show the resolved terms and the forecast du/dt only; nothing stands in for the drag.

USAGE
    python momentum_budget.py tail --days 100        # LAPTOP, once: backfill (~2.5 min a day)
    python momentum_budget.py live                    # daily (Actions): new analysis days + newest forecast + render
    python momentum_budget.py clim                    # LAPTOP: MERRA-2 reference -> reference/mbudget_clim.npz
    python momentum_budget.py render
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bdc_dc as B                                                    # noqa: E402  (shared constants, readers, statistics)

OUT = HERE / "data" / "mbudget"
TAIL = OUT / "tail"                                        # one npz per GEOS FP analysis day (the published cache)
FCST = OUT / "fcst"
REF = HERE / "reference" / "mbudget_clim.npz"

LEVS = B.LEVS
A_E, G, K, P0, OM = B.A_E, B.G, B.K, B.P0, B.OM
H_SCALE = 7000.0
NV_LEV = slice(4, 43)                                      # native levels 5..43: 0.078 -> 226 hPa (brackets 0.1-200)
P_NV = B.P_NATIVE[4:43]
SYN_HOURS = tuple(range(0, 24, 3))                         # 00..21Z; + the next 00Z
UDT_HOURS = tuple(1.5 + 3 * k for k in range(8))            # 01:30 ... 22:30, the exact 00-24Z mean
TRAP_W = np.r_[0.5, np.ones(7), 0.5] / 8                  # trapezoid over 00, 03, ..., 21, 24Z
WAVES = ("k1", "k2", "k3p")

ASSIM_NP, ASSIM_UDT, FCAST_NP = B.ASSIM_NP, B.ASSIM_UDT, B.FCAST_NP
log, Bad, retry, save, load = B.log, B.Bad, B.retry, B.save, B.load


# ----------------------------------------------------------------------------------------------------------------------
# Physics
# ----------------------------------------------------------------------------------------------------------------------
def reduce_waves(f):
    """Zonal means and eddy covariances (total and by zonal wavenumber 1, 2, 3+) on (lev, lat) from (lev, lat, lon)."""
    zm = {k: np.nanmean(f[k], axis=-1) for k in f}
    n = f["u"].shape[-1]
    F = {k: np.fft.rfft(np.nan_to_num(f[k] - zm[k][..., None]), axis=-1) for k in f}
    out = {"ubar": zm["u"], "vbar": zm["v"], "tbar": zm["t"], "wbar": zm["omega"]}
    for name, (a, b) in (("vT", ("v", "t")), ("uv", ("u", "v")), ("uw", ("u", "omega"))):
        c = 2 * np.real(F[a] * np.conj(F[b]))[..., 1:] / n ** 2                # (lev, lat, k >= 1)
        if n % 2 == 0:
            c[..., -1] /= 2                                                     # the Nyquist term is not doubled
        out[name] = c.sum(-1)
        out[f"{name}_k1"], out[f"{name}_k2"], out[f"{name}_k3p"] = c[..., 0], c[..., 1], c[..., 2:].sum(-1)
    return out


def _dphi(x, phi):
    return np.gradient(x, phi, axis=-1)


def tem_terms(lat, r, levs=None):
    """TEM momentum-budget terms (m/s^2) on (lev, lat) from one reduced snapshot r (levels top -> bottom, LEVS unless
    given)."""
    levs = LEVS if levs is None else np.asarray(levs, float)
    phi = np.deg2rad(lat); c = np.cos(phi); cc = np.where(np.abs(c) < 1e-3, 1e-3, c)
    p = levs * 100.0
    pk = (P0 / levs)[:, None] ** K
    th = r["tbar"] * pk
    th_p = np.gradient(th, p, axis=0)
    th_p = np.where(np.abs(th_p) < 1e-7, -1e-7, th_p)
    u = r["ubar"]
    u_p = np.gradient(u, p, axis=0)
    f = 2 * OM * np.sin(phi)
    fhat = f[None, :] - _dphi(u * c[None, :], phi) / (A_E * cc[None, :])
    psi_e = r["vT"] * pk / th_p                                                # [v'th'] / th_p
    vstar = r["vbar"] - np.gradient(psi_e, p, axis=0)
    omstar = r["wbar"] + _dphi(c[None, :] * psi_e, phi) / (A_E * cc[None, :])
    out = {"ubar": u, "tbar": r["tbar"], "fhat": fhat, "vstar": vstar, "omstar": omstar,
           "cor": fhat * vstar, "vad": -omstar * u_p}

    def epd(sfx):
        pe = r[f"vT{sfx}"] * pk / th_p
        Fphi = A_E * c[None, :] * (u_p * pe - r[f"uv{sfx}"])
        Fp = A_E * c[None, :] * (fhat * pe - r[f"uw{sfx}"])
        div = _dphi(Fphi * c[None, :], phi) / (A_E * cc[None, :]) + np.gradient(Fp, p, axis=0)
        return div / (A_E * cc[None, :]), Fphi, Fp
    out["epd"], out["Fphi"], out["Fp"] = epd("")
    for w in WAVES:
        out[f"epd_{w}"] = epd("_" + w)[0]
    return out


def dc_psi(lat, fhat, G_):
    """Downward control: psi* (1e9 kg/s) on (lev, lat) from a forcing G (m/s^2): psi* = -(2 pi a cos phi / g) int_0^p
    G / fhat dp (steady TEM with v* = -G / fhat). Undefined where |fhat| is tiny (the deep tropics)."""
    phi = np.deg2rad(lat); c = np.cos(phi)
    p = LEVS * 100.0
    fh = np.where(np.abs(fhat) < 1e-6, np.nan, fhat)
    q = G_ / fh
    cum = np.concatenate([np.zeros((1, q.shape[1])), np.cumsum(0.5 * (q[1:] + q[:-1]) * np.diff(p)[:, None], axis=0)])
    return -(2 * np.pi * A_E * c[None, :] / G) * cum / 1e9


# ----------------------------------------------------------------------------------------------------------------------
# Reads
# ----------------------------------------------------------------------------------------------------------------------
def read_states(ds, times):
    """inst3_3d_asm_Np at a CONTIGUOUS run of 3-hourly times -> list of reduced dicts (wave-split covariances) on LEVS at
    1 x 1.25 deg. One request per variable for the whole run: per-request overhead dominated single-time reads (12.8 s
    a snapshot against 77 s for a day's nine)."""
    lev = ds.lev.values
    i0 = int(np.argmin(np.abs(lev - LEVS[-1])))
    assert np.allclose(lev[i0:][::-1], LEVS), "unexpected GEOS FP levels"
    tix = pd.DatetimeIndex(ds.time.values)
    k = [tix.get_loc(t) for t in times]
    assert k == list(range(k[0], k[0] + len(k))), "times must be contiguous on the server's axis"
    st = B.STRIDE_FP

    def go():
        sub = ds[["u", "v", "t", "omega"]].isel(time=slice(k[0], k[-1] + 1), lev=slice(i0, None),
                                                 lat=slice(None, None, st), lon=slice(None, None, st)).load()
        out = []
        for n in range(len(times)):
            f = {v: B._clean(sub[v].values[n])[::-1] for v in ("u", "v", "t", "omega")}
            B.check_state(f)
            r = reduce_waves(f); r["lat"] = ds.lat.values[::st]
            out.append(r)
        return out
    return retry(go, f"state {times[0]}..{times[-1]}")


def to_levs(zm):
    x = np.log(P_NV); xi = np.log(LEVS)
    return np.stack([np.interp(xi, x, zm[:, j]) for j in range(zm.shape[1])], axis=1)


UDT_VARS = ("dudtgwd", "dudtana")
AUDIT_VARS = ("dudtdyn", "dudtmst", "dudttrb")


def read_udt_day(ds, day, audit=False):
    """Daily-mean (all eight 3-hourly windows) zonal-mean tendencies on (LEVS, lat), m/s^2: GWD at full longitude
    resolution, ANA every 2nd longitude. DYN is not read: above 100 hPa the moist and turbulent tendencies are exactly zero
    and du/dt = DYN + GWD + ANA closes to 0.01 m/s/day (2026-09-20, every latitude band and level), so DYN = du/dt - GWD -
    ANA. With audit=True DYN, MST and TRB are read as well, to re-check that closure."""
    tix = pd.DatetimeIndex(ds.time.values)
    names = UDT_VARS + (AUDIT_VARS if audit else ())
    tt = [pd.Timestamp(day) + pd.Timedelta(hours=h) for h in UDT_HOURS]
    if tt[-1] not in tix:
        raise Bad(f"udt {tt[-1]} not on the server (axis ends {tix[-1]})")
    k = [tix.get_loc(t) for t in tt]
    assert k == list(range(k[0], k[0] + 8)), "tendency windows not contiguous"
    sl = dict(time=slice(k[0], k[-1] + 1), lev=NV_LEV, lat=slice(None, None, 4))

    def go():
        a = ds[["dudtgwd"]].isel(**sl).load()
        b = ds[list(names[1:])].isel(**sl, lon=slice(None, None, 2)).load()
        out = {}
        for v in names:
            x = B._clean((a if v == "dudtgwd" else b)[v].values)
            for n in range(x.shape[0]):
                if np.isfinite(x[n]).mean() < 0.995 or np.nanmax(np.abs(x[n])) > 5e-2:
                    raise Bad(f"{v} at {tt[n]}: missing or out of range")
                if v in ("dudtgwd", "dudtana", "dudtdyn") and (x[n] == 0).mean() > 0.5:
                    raise Bad(f"{v} at {tt[n]}: mostly zero")
            out[v] = np.nanmean(x, -1)                                          # (8, lev, lat)
        return out
    r = retry(go, f"udt {day}")
    acc = {v: r[v] for v in names}
    out = {v[4:]: to_levs(np.mean(acc[v], 0)) for v in names}
    if audit:
        out = {("srv_" + k if k in ("dyn", "mst", "trb") else k): x for k, x in out.items()}
    return out, ds.lat.values[::4]


# ----------------------------------------------------------------------------------------------------------------------
# One day
# ----------------------------------------------------------------------------------------------------------------------
DAY_FIELDS = ("ubar", "tbar", "fhat", "vstar", "omstar", "cor", "vad", "epd") + tuple(f"epd_{w}" for w in WAVES)


def day_from_snaps(snaps, lat):
    """Trapezoid daily means of the TEM terms over five snapshots (00..24Z) + the observed du/dt."""
    terms = [tem_terms(lat, s) for s in snaps]
    out = {k: np.tensordot(TRAP_W, np.stack([t[k] for t in terms]), axes=1) for k in DAY_FIELDS}
    out["u0"] = snaps[0]["ubar"]
    # the 00Z snapshot alone, as MERRA-2 GMI samples it: measures what once-daily sampling does to the climatology
    for k in ("cor", "vad", "epd"):
        out[f"{k}00"] = terms[0][k]
    out["dudt"] = (snaps[-1]["ubar"] - snaps[0]["ubar"]) / 86400.0
    return out


# what a day record keeps (~180 KB): everything the renderer reads
STORE_FIELDS = ("ubar", "fhat", "omstar", "cor", "vad", "epd", "epd_k1", "epd_k2", "epd_k3p", "dudt", "gwd", "ana", "epd00")


def compact(path):
    """Rewrite a day record with STORE_FIELDS only (and any audit fields)."""
    z = load(path)
    keep = {k: v for k, v in z.items() if k in STORE_FIELDS or k.startswith("srv_") or k in ("lat", "lev", "time")}
    if len(keep) < len(z):
        save(path, keep)


def build_day(dsn, dsu, day, audit=False):
    t0 = time.time()
    times = [pd.Timestamp(day) + pd.Timedelta(hours=h) for h in SYN_HOURS] + [pd.Timestamp(day) + pd.Timedelta(days=1)]
    tix = pd.DatetimeIndex(dsn.time.values)
    miss = [t for t in times if t not in tix]
    if miss:
        raise Bad(f"state {miss[0]} not on the server yet")
    udt, lat_u = read_udt_day(dsu, day, audit)                 # the likelier failure first
    snaps = read_states(dsn, times)
    lat = snaps[0]["lat"]
    assert np.allclose(lat, lat_u)
    rec = day_from_snaps(snaps, lat)
    rec.update(udt)
    rec = {k: v for k, v in rec.items() if k in STORE_FIELDS or k.startswith("srv_")}
    rec.update({"lat": lat, "lev": LEVS, "time": np.array(f"{pd.Timestamp(day):%Y-%m-%d}")})
    save(TAIL / f"{pd.Timestamp(day):%Y%m%d}.npz", rec)
    log(f"  {pd.Timestamp(day):%Y-%m-%d} ok {time.time() - t0:.0f}s")
    return rec


def fetch_tail(days=100, budget_min=None, audit_every=0):
    """Analysis days back from the newest complete one (needs the next day's 00Z state and all eight tendency windows),
    newest first; resumable."""
    t_start = time.time()
    dsn = B.open_ds(ASSIM_NP, True); dsu = B.open_ds(ASSIM_UDT, True)
    last_state = pd.Timestamp(dsn.time.values[-1]); last_udt = pd.Timestamp(dsu.time.values[-1])
    newest = min((last_state - pd.Timedelta(days=1)).normalize(),
                 (last_udt - pd.Timedelta(hours=22.5)).normalize())
    targets = [newest - pd.Timedelta(days=k) for k in range(days)]
    todo = [d for d in targets if not (TAIL / f"{d:%Y%m%d}.npz").exists()]
    log(f"tail: {len(todo)} of {len(targets)} days to fetch (state to {last_state}, tendencies to {last_udt})")
    for i, d in enumerate(todo):
        if budget_min and (time.time() - t_start) / 60 > budget_min:
            log("tail: time budget reached, the rest on the next run"); break
        try:
            build_day(dsn, dsu, d, audit=bool(audit_every) and i % audit_every == 0)
        except Exception as e:                                           # noqa: BLE001
            log(f"  {d:%Y-%m-%d} FAILED: {str(e)[:140]}")


PART = OUT / "part"                                        # per-stream halves of a day, merged when both exist


def _merge_part(d):
    fs, fu = PART / f"state_{d:%Y%m%d}.npz", PART / f"udt_{d:%Y%m%d}.npz"
    if not (fs.exists() and fu.exists()):
        return False
    try:
        rec = {**load(fs), **load(fu)}
    except Exception:                                                    # noqa: BLE001  (the other stream is writing)
        return False
    rec = {k: v for k, v in rec.items() if k in STORE_FIELDS or k.startswith("srv_") or k in ("lat", "lev", "time")}
    save(TAIL / f"{d:%Y%m%d}.npz", rec)
    for f in (fs, fu):
        f.unlink(missing_ok=True)
    log(f"  {d:%Y-%m-%d} merged")
    return True


def fetch_stream(which, days=100, audit_every=0):
    """LAPTOP backfill as two parallel streams (state / tendencies), each sequential on its own OPeNDAP connection;
    whichever finishes a day second merges the two halves into the day record."""
    url = ASSIM_NP if which == "state" else ASSIM_UDT
    ds = B.open_ds(url, True)
    dsn = B.open_ds(ASSIM_NP, True) if which == "udt" else ds
    dsu = B.open_ds(ASSIM_UDT, True) if which == "state" else ds
    last_state = pd.Timestamp(dsn.time.values[-1]); last_udt = pd.Timestamp(dsu.time.values[-1])
    newest = min((last_state - pd.Timedelta(days=1)).normalize(), (last_udt - pd.Timedelta(hours=22.5)).normalize())
    PART.mkdir(parents=True, exist_ok=True)
    targets = [newest - pd.Timedelta(days=k) for k in range(days)]
    for i, d in enumerate(targets):
        if (TAIL / f"{d:%Y%m%d}.npz").exists() or (PART / f"{which}_{d:%Y%m%d}.npz").exists():
            _merge_part(d); continue
        t0 = time.time()
        try:
            if which == "state":
                times = [d + pd.Timedelta(hours=h) for h in SYN_HOURS] + [d + pd.Timedelta(days=1)]
                snaps = read_states(ds, times)
                rec = day_from_snaps(snaps, snaps[0]["lat"])
                rec.update({"lat": snaps[0]["lat"], "lev": LEVS, "time": np.array(f"{d:%Y-%m-%d}")})
            else:
                rec, lat_u = read_udt_day(ds, d, audit=bool(audit_every) and i % audit_every == 0)
                rec["lat_u"] = lat_u
            save(PART / f"{which}_{d:%Y%m%d}.npz", rec)
            log(f"  {which} {d:%Y-%m-%d} {time.time() - t0:.0f}s")
        except Exception as e:                                           # noqa: BLE001
            log(f"  {which} {d:%Y-%m-%d} FAILED: {str(e)[:140]}"); continue
        _merge_part(d)


def fetch_forecast(cyc=None):
    """Newest (or given) 00Z forecast: daily-mean resolved terms for days 0-9 from the 00/06/12/18Z steps."""
    cyc = cyc or B.forecast_cycles()[-1]
    dest = FCST / f"{cyc}.npz"
    if dest.exists():
        log(f"forecast {cyc}: cached"); return dest
    ds = B.open_ds(f"{FCAST_NP}/inst3_3d_asm_Np.{cyc}")
    tix = pd.DatetimeIndex(ds.time.values)
    c0 = pd.Timestamp(cyc[:8])
    days = []
    for k in range(10):
        times = [c0 + pd.Timedelta(days=k, hours=h) for h in SYN_HOURS] + [c0 + pd.Timedelta(days=k + 1)]
        if any(t not in tix for t in times):
            break
        t0 = time.time()
        snaps = read_states(ds, times)
        days.append(day_from_snaps(snaps, snaps[0]["lat"]))
        lat = snaps[0]["lat"]
        log(f"  forecast {cyc} day {k} {time.time() - t0:.0f}s")
    if len(days) < 5:
        raise Bad(f"forecast {cyc}: only {len(days)} complete days")
    save(dest, {**{k: np.stack([d[k] for d in days]) for k in days[0]}, "lat": lat, "lev": LEVS,
                "valid": np.array([f"{c0 + pd.Timedelta(days=k):%Y-%m-%d}" for k in range(len(days))]), "cycle": np.array(cyc)})
    for f in FCST.glob("*.npz"):
        if f != dest and re.fullmatch(r"\d{8}_00\.npz", f.name):
            f.unlink()
    return dest


# ----------------------------------------------------------------------------------------------------------------------
# MERRA-2 reference (LAPTOP: reads the local MERRA-2 GMI and M2TMNPUDT reductions; writes reference/mbudget_clim.npz)
# ----------------------------------------------------------------------------------------------------------------------
UDT_DIR = HERE / "data" / "m2_udt"
UDT_YEARS = range(2005, 2025)            # MERRA-2 assimilates MLS temperature from Aug 2004: the increment's era


def _keep(lev):
    k = np.array([int(np.argmin(np.abs(lev - L))) for L in LEVS])
    assert np.allclose(lev[k], LEVS, rtol=1e-4), "levels do not match"
    return k


def _rec_from(z, i, keep, k_split=True):
    """A reduced record for tem_terms from a MERRA-2 GMI tem file (or an overlap file when i is None)."""
    g = (lambda k: z[k][keep].astype("float64")) if i is None else (lambda k: z[k][i][keep].astype("float64"))
    r = {"ubar": g("ubar"), "vbar": g("vbar"), "tbar": g("tbar"),
         "wbar": g("wbar") if "wbar" in z.files else g("omegabar"),
         "vT": g("vT"), "uv": g("uv"), "uw": g("uw")}
    zero = np.zeros_like(r["uv"])
    for w in WAVES:                                   # GMI saves no wave-split [u'omega']; overlap files no splits at all
        if k_split and "vT_k1" in z.files:
            r[f"vT_{w}"] = g(f"vT_{w}") if w != "k3p" else g("vT_k3") + g("vT_k4p")
            r[f"uv_{w}"] = g(f"uv_{w}") if w != "k3p" else g("uv_k3") + g("uv_k4p")
        else:
            r[f"vT_{w}"] = r[f"uv_{w}"] = zero
        r[f"uw_{w}"] = zero
    return r


def gmi_samples():
    """TEM terms (00Z snapshots) + the day's gravity-wave drag for every MERRA-2 GMI day on file (every 10th day)."""
    rows = {k: [] for k in ("ubar", "tbar", "fhat", "cor", "vad", "epd", "epd_k1", "epd_k2", "epd_k3p", "gwd")}
    T, lat = [], None
    for y in range(1991, 2020):
        tf, gf = B.M2DIR / f"tem_{y}.npz", B.M2DIR / f"gwd_{y}.npz"
        if not (tf.exists() and gf.exists()):
            continue
        t, g = np.load(tf), np.load(gf)
        keep, gkeep = _keep(t["lev"]), _keep(g["lev"])
        gt = {str(x): j for j, x in enumerate(g["time"])}
        for i, tt in enumerate(t["time"]):
            j = gt.get(str(tt))
            if j is None:
                continue
            r = _rec_from(t, i, keep)
            if any(not np.isfinite(r[k]).all() or (r[k] == 0).all(axis=1).any() for k in ("ubar", "tbar", "vT", "uv", "uw")):
                log(f"  MERRA-2 {tt}: rejected (missing or all-zero level)"); continue
            tm = tem_terms(t["lat"], r)
            for k in rows:
                rows[k].append(np.nan_to_num(g["dudtgwd"][j][gkeep].astype("float64")) if k == "gwd" else tm[k])
            T.append(str(tt)); lat = t["lat"]
    return pd.DatetimeIndex(pd.to_datetime(T)), {k: np.array(v) for k, v in rows.items()}, lat


def udt_monthly():
    """MERRA-2 M2TMNPUDT monthly zonal means (ANA, GWD, DYN) on (LEVS, 1-deg lat), years UDT_YEARS."""
    out = {k: [] for k in ("ana", "gwd", "dyn")}; months = []
    for y in UDT_YEARS:
        z = np.load(UDT_DIR / f"udt_{y}.npz")
        keep = _keep(z["lev"])
        for i, m in enumerate(z["time"]):
            for k, v in (("ana", "DUDTANA"), ("gwd", "DUDTGWD"), ("dyn", "DUDTDYN")):
                out[k].append(np.nan_to_num(z[v][i][keep][:, ::2].astype("float64")))
            months.append(str(m))
    return months, {k: np.array(v) for k, v in out.items()}


def up_band(lat, doy, i_lev):
    s, n = B.get_tl()[int(doy) - 1, i_lev]
    return s, n


def upflux_mm(psi_row, lat, doy, lev):
    """Mean log-pressure w* (mm/s) between the MERRA-2 turnaround latitudes from a psi* row (1e9 kg/s)."""
    i = B.IDX_LEVS.index(lev)
    s, n = up_band(lat, doy, i)
    F = B.upflux(psi_row, lat, doy, i) * 1e9
    return F * G * H_SCALE / (lev * 100.0 * 2 * np.pi * A_E ** 2 * (np.sin(np.deg2rad(n)) - np.sin(np.deg2rad(s)))) * 1e3


def direct_w_mm(omstar_row, lat, doy, lev):
    """Area-mean w* = -H omega*/p (mm/s) between the turnaround latitudes, from the analysed omega*."""
    i = B.IDX_LEVS.index(lev)
    s, n = up_band(lat, doy, i)
    w = (lat >= s) & (lat <= n); c = np.cos(np.deg2rad(lat[w]))
    return float(-(omstar_row[w] * c).sum() / c.sum() * H_SCALE / (lev * 100.0) * 1e3)


UP_LEVS = (70, 100)
HEMI = {"nh": (55.0, 65.0), "sh": (-65.0, -55.0)}
VLEVS = (10, 1, 50)


def band(x, lat, lo, hi):
    """cos(lat)-weighted mean over lo..hi on the last axis."""
    w = (lat >= lo - 1e-6) & (lat <= hi + 1e-6); c = np.cos(np.deg2rad(lat[w]))
    return (x[..., w] * c).sum(-1) / c.sum()


def build_clim():
    """MERRA-2 reference for the budget -> reference/mbudget_clim.npz (a few MB) + a JSON summary."""
    t0 = time.time()
    T, S, lat = gmi_samples()
    doy = T.dayofyear.values; X = B.harm(doy)
    XtXi = np.linalg.inv(X.T @ X)
    ref = {"lat": lat, "lev": LEVS, "gmi_years": np.array(sorted(set(T.year))), "gmi_n": len(T), "XtXi": XtXi}
    for k, Y in S.items():
        Yf = Y.reshape(len(T), -1)
        c = np.linalg.lstsq(X, Yf, rcond=None)[0]
        res = Yf - X @ c
        ref[f"{k}_coef"] = c.reshape(X.shape[1], *Y.shape[1:])
        ref[f"{k}_s2"] = ((res ** 2).sum(0) / (len(T) - X.shape[1])).reshape(Y.shape[1:])
        if k in ("ubar", "epd", "gwd", "cor"):
            est, se = B.interannual_var(Y, T, half=15)
            ref[f"{k}_via30"], ref[f"{k}_via30_se"] = est, se
    log(f"GMI samples: {len(T)} days {T.min():%Y}-{T.max():%Y} ({time.time() - t0:.0f}s)")
    # the vortex-strength strip: band-mean u, 3-harmonic normal and seasonal single-day 10/90 % (residual quantiles pooled
    # over +-30 days and all years, smoothed with 2 harmonics, as bdc_dc does for its index)
    days = np.arange(1, 367); Xd = B.harm(days); X2 = B.harm(days, 2)
    for h, (lo, hi) in HEMI.items():
        for L in VLEVS:
            y = band(S["ubar"][:, B.lev_index(L)], lat, lo, hi)
            c = np.linalg.lstsq(X, y, rcond=None)[0]; r = y - X @ c
            ref[f"u_{h}{L}_mean"] = Xd @ c
            for q in (0.1, 0.9):
                qq = np.array([np.quantile(r[np.abs(B.circ_dd(doy, d0)) <= 30], q) for d0 in days])
                ref[f"u_{h}{L}_q{int(q * 100)}"] = Xd @ c + X2 @ np.linalg.lstsq(X2, qq, rcond=None)[0]
    # tropical upwelling parts by downward control, per GMI day (mm/s), for the normals
    up = {f"{nm}{L}": [] for L in UP_LEVS for nm in ("ep", "gw")}
    for n in range(len(T)):
        for nm, Gf in (("ep", S["epd"][n]), ("gw", S["gwd"][n])):
            psi = dc_psi(lat, S["fhat"][n], Gf)
            for L in UP_LEVS:
                up[f"{nm}{L}"].append(upflux_mm(psi[B.lev_index(L)], lat, doy[n], L))
    for k, v in up.items():
        v = np.array(v); c = np.linalg.lstsq(X, v, rcond=None)[0]; r = v - X @ c
        ref[f"up_{k}_coef"] = c; ref[f"up_{k}_s2"] = np.array(float((r ** 2).sum() / (len(v) - X.shape[1])))
        est, se = B.interannual_var(v[:, None], T, half=15)
        ref[f"up_{k}_via30"], ref[f"up_{k}_via30_se"] = est[0], se[0]
    # MERRA-2 monthly tendencies: normals and year-to-year sd of monthly means (~ 30-day means)
    months, U = udt_monthly()
    mo = np.array([int(m[5:7]) for m in months])
    for k, Y in U.items():
        ref[f"m_{k}_mean"] = np.array([Y[mo == m].mean(0) for m in range(1, 13)])
        ref[f"m_{k}_sd"] = np.array([Y[mo == m].std(0, ddof=1) for m in range(1, 13)])
    ref["m_years"] = np.array(list(UDT_YEARS))
    # the increment's upwelling part, per year-month, with the GMI mean state for the month's mid-day
    upa = {L: np.zeros((12, len(UDT_YEARS))) for L in UP_LEVS}
    for i, m in enumerate(months):
        d = pd.Timestamp(f"{m}-15").dayofyear
        fh = np.tensordot(B.harm([d])[0], ref["fhat_coef"], axes=1)
        psi = dc_psi(lat, fh, U["ana"][i])
        for L in UP_LEVS:
            upa[L][int(m[5:7]) - 1, list(UDT_YEARS).index(int(m[:4]))] = upflux_mm(psi[B.lev_index(L)], lat, d, L)
    for L in UP_LEVS:
        ref[f"up_an{L}_mmean"], ref[f"up_an{L}_msd"] = upa[L].mean(1), upa[L].std(1, ddof=1)
    ref.update(seam_offsets(lat, ref))
    save(REF, ref)
    log(f"reference written: {REF} ({REF.stat().st_size / 1e6:.1f} MB, {time.time() - t0:.0f}s)")
    return ref


def seam_offsets(lat, ref):
    """GEOS FP minus MERRA-2 on the 38 same days of 2018-19 (bdc_dc overlap files: 00Z states; GEOS FP GWD and ANA as
    the mean of four tavg3 windows). EPD/cor/vad: 00Z against 00Z. GWD: against the GMI replay's daily GWD. ANA: against
    MERRA-2's monthly increment for the month (the replay saves none). Per point: mean difference, its standard error, and
    for GWD also the no-intercept ratio. Also the upwelling parts at 70/100 hPa."""
    OVL = B.OVL
    udt = {}
    for y in (2018, 2019):
        z = np.load(UDT_DIR / f"udt_{y}.npz"); keep = _keep(z["lev"])
        for i, m in enumerate(z["time"]):
            udt[str(m)] = np.nan_to_num(z["DUDTANA"][i][keep][:, ::2].astype("float64"))
    D = {k: [] for k in ("epd", "cor", "vad", "gwd", "ana")}
    Gm, Gf = [], []
    UP = {f"{nm}{L}": [] for L in UP_LEVS for nm in ("ep", "gw", "an")}
    days = []
    for fm in sorted(OVL.glob("m2_*.npz")):
        day = fm.stem.split("_")[1]
        ff = OVL / f"fp_{day}.npz"; gfile = B.M2DIR / f"gwd_{day[:4]}.npz"
        if not (ff.exists() and gfile.exists()):
            continue
        gz = np.load(gfile); j = [i for i, t in enumerate(gz["time"]) if str(t).replace("-", "") == day]
        if not j:
            continue
        m2, fp = np.load(fm), np.load(ff)
        tm = tem_terms(m2["lat"], _rec_from(m2, None, np.arange(len(LEVS)), False))
        tf = tem_terms(fp["lat"], _rec_from(fp, None, np.arange(len(LEVS)), False))
        g_m2 = np.nan_to_num(gz["dudtgwd"][j[0]][_keep(gz["lev"])].astype("float64"))
        a_m2 = udt[f"{day[:4]}-{day[4:6]}"]
        for k in ("epd", "cor", "vad"):
            D[k].append(tf[k] - tm[k])
        D["gwd"].append(fp["gwd"] - g_m2); D["ana"].append(fp["ana"] - a_m2)
        Gm.append(g_m2); Gf.append(fp["gwd"])
        d = pd.Timestamp(day).dayofyear
        for nm, (Gfp, fhf, Gmm, fhm) in (("ep", (tf["epd"], tf["fhat"], tm["epd"], tm["fhat"])),
                                          ("gw", (fp["gwd"], tf["fhat"], g_m2, tm["fhat"])),
                                          ("an", (fp["ana"], tf["fhat"], a_m2, tm["fhat"]))):
            pf, pm = dc_psi(lat, fhf, Gfp), dc_psi(lat, fhm, Gmm)
            for L in UP_LEVS:
                k = B.lev_index(L)
                UP[f"{nm}{L}"].append(upflux_mm(pf[k], lat, d, L) - upflux_mm(pm[k], lat, d, L))
        days.append(day)
    out = {"seam_n": len(days), "seam_days": np.array(days)}
    for k, v in D.items():
        v = np.array(v)
        out[f"seam_{k}"] = v.mean(0); out[f"seam_{k}_se"] = v.std(0, ddof=1) / np.sqrt(len(v))
    Gm, Gf = np.array(Gm), np.array(Gf)
    den = (Gm ** 2).sum(0)
    out["seam_gwd_ratio"] = r_ = np.where(den > 0, (Gm * Gf).sum(0) / np.where(den > 0, den, 1), 1.0)
    out["seam_gwd_ratio_se"] = np.sqrt(((Gf - r_ * Gm) ** 2).sum(0) / (len(Gm) - 1) / np.where(den > 0, den, np.inf))
    for k, v in UP.items():
        v = np.array(v)
        out[f"seam_up_{k}"] = np.array(v.mean()); out[f"seam_up_{k}_se"] = np.array(v.std(ddof=1) / np.sqrt(len(v)))
    log(f"seam: {len(days)} same days 2018-19")
    return out


# ----------------------------------------------------------------------------------------------------------------------
# Render
# ----------------------------------------------------------------------------------------------------------------------
DAY = 86400.0
TAIL_KEEP = 100                                            # analysis days used and drawn
WIN = 30
INK, MUTED, GRID = B.INK, B.MUTED, B.GRID
NAVY, NORMAL, BAND = B.NAVY, B.NORMAL, B.BAND
# categorical slots 1-5 of the site's validated palette, in order (adjacent-pair CVD and normal-vision checks pass);
# observed du/dt in ink, the residual a neutral grey. Wavenumbers: red / green / violet with the resolved total orange.
C = {"dudt": INK, "cor": "#2a78d6", "epd": "#eb6834", "gwd": "#1baf7a", "vad": "#eda100", "ana": "#e87ba4",
     "resid": "#8a8f96", "k1": "#e34948", "k2": "#008300", "k3p": "#4a3aa7"}
LAB = {"dudt": "observed ∂ū/∂t", "cor": "Coriolis on the residual flow  f̂ v̄*", "epd": "resolved waves (EP-flux divergence)",
       "gwd": "gravity-wave drag", "vad": "vertical advection  −w̄* ∂ū/∂z", "ana": "analysis increment",
       "resid": "residual (numerics, sampling)", "k1": "wave 1", "k2": "wave 2", "k3p": "waves 3+"}
TERMS = ("dudt", "cor", "epd", "gwd", "vad", "ana", "resid")
RESOLVED = ("dudt", "cor", "epd", "vad")                     # what a forecast without tendencies still has


def load_tail():
    files = sorted(f for f in TAIL.glob("*.npz") if re.fullmatch(r"\d{8}\.npz", f.name))[-TAIL_KEEP:]
    if not files:
        raise Bad("no analysis days cached")
    recs = [load(f) for f in files]
    t = pd.DatetimeIndex([pd.Timestamp(str(r["time"])) for r in recs])
    days = pd.date_range(t.min(), t.max(), freq="D")
    keys = STORE_FIELDS
    Z = {}
    for k in keys:
        a = np.full((len(days),) + recs[0]["ubar"].shape, np.nan)
        for r, d in zip(recs, t):
            if k in r:
                a[days.get_loc(d)] = r[k]
        Z[k] = a
    Z["dyn"] = Z["dudt"] - Z["gwd"] - Z["ana"]
    Z["resid"] = Z["dudt"] - (Z["cor"] + Z["vad"] + Z["epd"] + Z["gwd"] + Z["ana"])
    audit = []
    for r in recs:
        if "srv_dyn" in r:
            res = r["dudt"] - (r["srv_dyn"] + r["gwd"] + r["ana"] + r["srv_mst"] + r["srv_trb"])
            audit.append({"day": str(r["time"]), "max_abs_m_s_day": round(float(np.nanmax(np.abs(res[:, 5:-5]))) * DAY, 3),
                          "rms_m_s_day": round(float(np.sqrt(np.nanmean(res[:, 5:-5] ** 2))) * DAY, 4)})
    return days, Z, recs[0]["lat"], audit


def load_forecast():
    fs = sorted(f for f in FCST.glob("*.npz") if re.fullmatch(r"\d{8}_00\.npz", f.name))
    if not fs:
        return None
    z = load(fs[-1])
    return {"days": pd.DatetimeIndex(pd.to_datetime(z["valid"])), "cycle": str(z["cycle"]),
            **{k: z[k] for k in ("dudt", "ubar", "fhat", "omstar", "cor", "vad", "epd", "epd_k1", "epd_k2", "epd_k3p")}}


def smooth(s, mode):
    """Trailing 7-day mean (a centred window would pull forecast days into the analysis line near the seam)."""
    return s if mode == "d1" else s.rolling(7, min_periods=5).mean()


def band_series(Z, days, lat, key, h, L):
    lo, hi = HEMI[h]
    return pd.Series(band(Z[key][:, B.lev_index(L)], lat, lo, hi), index=days)


def closure(Z, days, lat, lo, hi, L):
    """Daily closure at one band and level: du/dt against the sum of the five terms, and the residual (= the model's
    dynamical tendency minus the TEM dynamical terms, since moist and turbulent tendencies are zero above 100 hPa)."""
    k = B.lev_index(L)
    g = {key: band(Z[key][:, k], lat, lo, hi) * DAY for key in ("dudt", "cor", "vad", "epd", "gwd", "ana", "resid", "ubar")}
    ok = np.isfinite(g["dudt"]) & np.isfinite(g["epd"])
    y, x = g["dudt"][ok], (g["cor"] + g["vad"] + g["epd"] + g["gwd"] + g["ana"])[ok]
    sl, ic = np.polyfit(x, y, 1)
    r = g["resid"][ok]; u = g["ubar"][ok] / DAY
    ru = np.polyfit(u, r, 1)[0] if np.std(u) > 0.5 else np.nan
    rms = lambda a: float(np.sqrt(np.mean(np.square(a))))                  # noqa: E731
    return {"n_days": int(ok.sum()), "slope": round(float(sl), 3), "intercept": round(float(ic), 3),
            "r2": round(float(np.corrcoef(x, y)[0, 1] ** 2), 3), "rms_residual": round(rms(r), 3),
            "rms_dudt": round(rms(y), 3), "rms_epd": round(rms(g["epd"][ok]), 3), "rms_cor": round(rms(g["cor"][ok]), 3),
            "mean_residual": round(float(r.mean()), 3), "residual_share_of_epd_rms": round(rms(r) / max(rms(g["epd"][ok]), 1e-9), 2),
            "residual_vs_u_slope_per_day": None if not np.isfinite(ru) else round(float(ru), 4),
            "residual_vs_u_r": round(float(np.corrcoef(u, r)[0, 1]), 2) if np.std(u) > 0.5 else None,
            "mean_abs_gwd": round(float(np.mean(np.abs(g["gwd"][ok]))), 3), "mean_abs_ana": round(float(np.mean(np.abs(g["ana"][ok]))), 3)}


def _style(ax):
    B._style(ax)


def _dates(ax, t0, t1):
    import matplotlib.dates as mdates
    ax.set_xlim(t0, t1)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %-d"))


def _fc_shade(ax, t_last, t1):
    ax.axvspan(t_last + pd.Timedelta(hours=12), t1, color="#f2f3f5", lw=0, zorder=0)
    ax.axvline(t_last + pd.Timedelta(hours=12), color=MUTED, lw=0.8, ls=":")


def _save(fig, path):
    fig.savefig(path.with_suffix(".webp"), format="webp", pil_kwargs={"quality": 90})
    import matplotlib.pyplot as plt
    plt.close(fig)


def seam_note(R, key, lo, hi, L):
    k = B.lev_index(L)
    d, se = band(R[f"seam_{key}"][k], R["lat"], lo, hi) * DAY, band(R[f"seam_{key}_se"][k], R["lat"], lo, hi) * DAY
    return float(d), float(se)


def fig_vortex(days, Z, lat, fc, R, h, L, mode, cl, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lo, hi = HEMI[h]; hs = "N" if h == "nh" else "S"
    blab = f"{min(abs(lo), abs(hi)):.0f}–{max(abs(lo), abs(hi)):.0f}°{hs}"
    t_last = days.max()
    fcd = fc["days"][fc["days"] > t_last] if fc is not None else pd.DatetimeIndex([])
    t0 = days.min(); t1 = (fcd.max() if len(fcd) else t_last) + pd.Timedelta(days=1)
    fig = plt.figure(figsize=(12.4, 10.4), dpi=130)
    POS = ([0.066, 0.81, 0.919, 0.083], [0.066, 0.445, 0.919, 0.30], [0.066, 0.15, 0.919, 0.165])
    k = B.lev_index(L)
    doy_all = pd.date_range(t0, t1, freq="D")
    # vortex strength
    ax = fig.add_axes(POS[0]); _style(ax)
    dd = doy_all.dayofyear.values - 1
    ax.fill_between(doy_all, R[f"u_{h}{L}_q10"][dd], R[f"u_{h}{L}_q90"][dd], color=BAND, lw=0, label="MERRA-2 10–90 %")
    ax.plot(doy_all, R[f"u_{h}{L}_mean"][dd], color=NORMAL, lw=1.3, ls=(0, (5, 3)), label="MERRA-2 normal (1991–2019)")
    ua = band_series(Z, days, lat, "ubar", h, L)
    ax.plot(ua.index, ua.values, color=INK, lw=2.0, label="GEOS FP")
    if len(fcd):
        uf = pd.Series(band(fc["ubar"][:, k], lat, lo, hi), index=fc["days"])
        uf = pd.concat([ua.iloc[-1:], uf[uf.index > t_last]])
        ax.plot(uf.index, uf.values, color=INK, lw=2.0, ls=(0, (2.5, 1.4)))
        _fc_shade(ax, t_last, t1)
    ax.set_ylabel("ū, m s⁻¹", fontsize=9.5, color=INK); _dates(ax, t0, t1)
    ax.legend(fontsize=8.6, frameon=False, ncol=3, loc="lower right", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.2)
    ax.set_title(f"Vortex wind, {blab} at {L} hPa", loc="left", fontsize=11, fontweight="bold", color=INK)
    # the budget
    ax = fig.add_axes(POS[1]); _style(ax)
    last30 = days > t_last - pd.Timedelta(days=WIN)
    n30 = int(np.isfinite(band_series(Z, days, lat, "epd", h, L)[last30]).sum())
    handles = []
    for key in TERMS:
        sa = band_series(Z, days, lat, key, h, L) * DAY
        m30 = float(np.nanmean(sa[last30]))
        lw = 2.6 if key in ("dudt", "epd") else 1.6
        ls = (0, (3, 2)) if key == "resid" else "-"
        ln, = ax.plot(sa.index, smooth(sa, mode).values, color=C[key], lw=lw, ls=ls, zorder=4 if key == "dudt" else 3,
                      label=f"{LAB[key]}  {m30:+.2f}")
        handles.append(ln)
        if key in RESOLVED and len(fcd):
            sf = pd.Series(band(fc[key][:, k], lat, lo, hi) * DAY, index=fc["days"])
            both = smooth(pd.concat([sa, sf[sf.index > t_last]]), mode)
            seg = both[both.index >= t_last]
            ax.plot(seg.index, seg.values, color=C[key], lw=lw, ls=(0, (2.5, 1.4)), zorder=3)
    ax.axhline(0, color="#9aa1a9", lw=0.8)
    if len(fcd):
        _fc_shade(ax, t_last, t1)
        ax.text(t_last + pd.Timedelta(hours=30), 0.025, "forecast: resolved terms only", transform=ax.get_xaxis_transform(),
                fontsize=8.5, color=MUTED, va="bottom")
    _dates(ax, t0, t1)
    ax.set_ylabel("m s⁻¹ per day", fontsize=9.5, color=INK)
    ax.set_title(f"Zonal momentum budget ({'trailing 7-day means' if mode == 's7' else 'daily means'}; legend: mean of the last {n30} days)",
                 loc="left", fontsize=11, fontweight="bold", color=INK)
    ax.legend(handles=handles, fontsize=8.6, frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0.0, -0.07),
              handlelength=2.4, columnspacing=1.6)
    # waves
    ax = fig.add_axes(POS[2]); _style(ax)
    doy = doy_all.dayofyear.values
    clim = np.array([band(np.tensordot(B.harm([d])[0], R["epd_coef"], axes=1)[k], R["lat"], lo, hi) for d in doy]) * DAY
    ax.plot(doy_all, clim, color=NORMAL, lw=1.3, ls=(0, (5, 3)), label="MERRA-2 normal of the total (00Z, 1991–2019)")
    for key in ("epd_k1", "epd_k2", "epd_k3p", "epd"):
        w = key.split("_")[-1] if key != "epd" else "epd"
        sa = band_series(Z, days, lat, key, h, L) * DAY
        m30 = float(np.nanmean(sa[last30]))
        lab = (LAB[w] if w != "epd" else "all resolved waves") + f"  {m30:+.2f}"
        ax.plot(sa.index, smooth(sa, mode).values, color=C[w], lw=2.4 if w == "epd" else 1.5, label=lab)
        if len(fcd):
            sf = pd.Series(band(fc[key][:, k], lat, lo, hi) * DAY, index=fc["days"])
            both = smooth(pd.concat([sa, sf[sf.index > t_last]]), mode)
            seg = both[both.index >= t_last]
            ax.plot(seg.index, seg.values, color=C[w], lw=2.4 if w == "epd" else 1.5, ls=(0, (2.5, 1.4)))
    ax.axhline(0, color="#9aa1a9", lw=0.8)
    if len(fcd):
        _fc_shade(ax, t_last, t1)
    _dates(ax, t0, t1)
    ax.set_ylabel("m s⁻¹ per day", fontsize=9.5, color=INK)
    ax.set_title("Resolved-wave driving by zonal wavenumber (negative = decelerating the westerlies)", loc="left", fontsize=11,
                 fontweight="bold", color=INK)
    ax.legend(fontsize=8.6, frameon=False, ncol=5, loc="upper left", bbox_to_anchor=(0.0, -0.13), handlelength=2.2, columnspacing=1.4)
    cyc = fc["cycle"] if fc is not None else None
    fig.suptitle(f"GEOS FP · Momentum budget of the {'northern' if h == 'nh' else 'southern'} vortex · {blab}, {L} hPa",
                 x=0.066, y=0.975, ha="left", fontsize=15, fontweight="bold", color=INK)
    sub = (f"Transformed Eulerian mean, daily means of 3-hourly analyses to {t_last:%b %-d}"
           + (f", then the {cyc[:4]}-{cyc[4:6]}-{cyc[6:8]} 00Z forecast (dashed, shaded)" if cyc and len(fcd) else ""))
    fig.text(0.066, 0.94, sub, fontsize=10, color=MUTED, va="top")
    c = cl
    foot = (f"Closure over {c['n_days']} days: observed ∂ū/∂t against the sum of the five terms, slope {c['slope']:.2f}, r² {c['r2']:.2f}; "
            f"residual rms {c['rms_residual']:.2f} against {c['rms_epd']:.2f} for the resolved waves (m s⁻¹ per day).\n"
            f"Mean |analysis increment| {c['mean_abs_ana']:.2f}, mean |gravity-wave drag| {c['mean_abs_gwd']:.2f}.\n"
            "GEOS FP publishes no forecast wind tendencies: after the line only the resolved terms and the forecast ∂ū/∂t are drawn, "
            "and nothing stands in for the drag or the increment.")
    fig.text(0.066, 0.01, foot, fontsize=8.6, color=MUTED, va="bottom", linespacing=1.45)
    _save(fig, path)


# -------------------------- tropical upwelling by downward control ---------------------------------------------------
UP_COMP = ("ep", "gw", "an", "ns", "ot")
UP_LAB = {"ep": "resolved waves", "gw": "gravity-wave drag", "an": "analysis increment", "ns": "non-steady (−∂ū/∂t)",
          "ot": "other (vertical advection + residual)", "sum": "sum = w̄* from the analysed v̄*",
          "dir": "w̄* from the analysed ω̄* (independent)"}
UP_C = {"ep": C["epd"], "gw": C["gwd"], "an": C["ana"], "ns": C["vad"], "ot": C["resid"], "sum": INK, "dir": NAVY}


def up_parts(Z, days, lat, fc=None):
    """Daily downward-control parts of mean w* (mm/s) between the turnaround latitudes at 70 and 100 hPa."""
    out = {L: {c: [] for c in UP_COMP + ("sum", "dir")} for L in UP_LEVS}
    for n, d in enumerate(days):
        if not np.isfinite(Z["epd"][n]).any():
            for L in UP_LEVS:
                for c in out[L]:
                    out[L][c].append(np.nan)
            continue
        G_ = {"ep": Z["epd"][n], "gw": Z["gwd"][n], "an": Z["ana"][n], "ns": -Z["dudt"][n], "ot": Z["vad"][n] + Z["resid"][n]}
        psi = {c: dc_psi(lat, Z["fhat"][n], g) for c, g in G_.items()}
        for L in UP_LEVS:
            kk = B.lev_index(L)
            vals = {c: upflux_mm(psi[c][kk], lat, d.dayofyear, L) for c in G_}
            for c, v in vals.items():
                out[L][c].append(v)
            out[L]["sum"].append(sum(vals.values()))
            out[L]["dir"].append(direct_w_mm(Z["omstar"][n][kk], lat, d.dayofyear, L))
    res = {L: pd.DataFrame(out[L], index=days) for L in UP_LEVS}
    fres = None
    if fc is not None:
        fres = {L: pd.DataFrame({"ep": [upflux_mm(dc_psi(lat, fc["fhat"][i], fc["epd"][i])[B.lev_index(L)], lat, d.dayofyear, L)
                                        for i, d in enumerate(fc["days"])]}, index=fc["days"]) for L in UP_LEVS}
    return res, fres


def up_normals(R, doys, L):
    """MERRA-2 normals (mm/s) for a set of days: resolved and GWD parts from the GMI 3-harmonic fits, the increment from
    the monthly M2TMNPUDT fields; each with its variance terms for a WIN-day mean and the 2018-19 seam."""
    out = {}
    Xm = B.harm(doys).mean(0)
    lev_ = float(Xm @ R["XtXi"] @ Xm)
    for c in ("ep", "gw"):
        mu = float(np.mean(B.harm(doys) @ R[f"up_{c}{L}_coef"]))
        var_ia = max(float(R[f"up_{c}{L}_via30"]), 0.0) + float(R[f"up_{c}{L}_via30_se"])
        out[c] = {"normal": mu, "var_ia": var_ia, "var_mu": float(R[f"up_{c}{L}_s2"]) * lev_,
                  "seam": float(R[f"seam_up_{c}{L}"]), "seam_se": float(R[f"seam_up_{c}{L}_se"]), "daily_s2": float(R[f"up_{c}{L}_s2"])}
    mo = pd.DatetimeIndex([pd.Timestamp(2001, 1, 1) + pd.Timedelta(days=int(d) - 1) for d in doys]).month.values - 1
    w = np.bincount(mo, minlength=12) / len(mo)
    mu = float(w @ R[f"up_an{L}_mmean"]); sd2 = float(w @ R[f"up_an{L}_msd"] ** 2)
    out["an"] = {"normal": mu, "var_ia": sd2, "var_mu": sd2 / len(R["m_years"]), "seam": float(R[f"seam_up_an{L}"]),
                 "seam_se": float(R[f"seam_up_an{L}_se"]), "daily_s2": 0.0, "monthly": True}
    return out


def z_test(x_mean, x_daily, times, nrm):
    """Last-WIN-day GEOS FP mean against the seam-adjusted MERRA-2 normal. For the GMI-based parts the variance adds the
    GEOS FP mean's sampling noise (its own day-to-day variance / n, Bartlett-inflated for autocorrelation) to MERRA-2's
    year-to-year variance of 30-day means (estimate clipped at 0 plus one standard error); for the monthly increment the
    spread of MERRA-2's monthly means already is the spread of 30-day means. Plus the normal's and the seam's errors."""
    adj = nrm["normal"] + nrm["seam"]
    if nrm.get("monthly"):
        var_x = 0.0
    else:
        r = x_daily - np.nanmean(x_daily)
        infl = float(B.acf_inflation(np.asarray(r)[:, None], times)[0])
        var_x = float(np.nanvar(x_daily, ddof=1)) / np.isfinite(x_daily).sum() * infl
    var = nrm["var_ia"] + nrm["var_mu"] + nrm["seam_se"] ** 2 + var_x
    z = (x_mean - adj) / np.sqrt(var)
    p = float(B.norm_p(z))
    return {"geosfp": round(float(x_mean), 4), "merra2_normal": round(nrm["normal"], 4), "seam_added": round(nrm["seam"], 4),
            "normal_on_geosfp_scale": round(adj, 4), "anomaly": round(float(x_mean - adj), 4), "sd": round(float(np.sqrt(var)), 4),
            "z": round(float(z), 2), "p": round(p, 3), "significant_5pct": bool(p < 0.05),
            "range_10_90": [round(adj - 1.2816 * np.sqrt(var), 4), round(adj + 1.2816 * np.sqrt(var), 4)]}


def fig_upwelling(up, fup, days, fc, R, stats, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t_last = days.max()
    fcd = fc["days"][fc["days"] > t_last] if fc is not None else pd.DatetimeIndex([])
    t0 = days.min() + pd.Timedelta(days=WIN - 1); t1 = (fcd.max() if len(fcd) else t_last) + pd.Timedelta(days=1)
    fig = plt.figure(figsize=(12.4, 8.9), dpi=130)
    gs = fig.add_gridspec(2, 2, width_ratios=[3.1, 1.25], hspace=0.36, wspace=0.16, left=0.066, right=0.985, top=0.87, bottom=0.22)
    minp = int(0.8 * WIN)
    for row, L in enumerate(UP_LEVS):
        ax = fig.add_subplot(gs[row, 0]); _style(ax)
        df = up[L]
        for c in ("sum", "dir") + UP_COMP:
            run = df[c].rolling(f"{WIN}D", min_periods=minp).mean()
            lw = 2.6 if c in ("sum", "ep") else (1.4 if c == "dir" else 1.6)
            ls = (0, (4, 2)) if c == "dir" else ((0, (3, 2)) if c == "ot" else "-")
            ax.plot(run.index, run.values, color=UP_C[c], lw=lw, ls=ls, label=UP_LAB[c] if row == 0 else None)
        if fup is not None and len(fcd):
            both = pd.concat([df["ep"], fup[L]["ep"][fup[L].index > t_last]])
            run = both.rolling(f"{WIN}D", min_periods=minp).mean()
            seg = run[run.index >= t_last]
            ax.plot(seg.index, seg.values, color=UP_C["ep"], lw=2.6, ls=(0, (2.5, 1.4)))
            _fc_shade(ax, t_last, t1)
        ax.axhline(0, color="#9aa1a9", lw=0.8)
        _dates(ax, t0, t1)
        ax.set_ylabel("mm s⁻¹", fontsize=9.5, color=INK)
        ax.set_title(f"{L} hPa, trailing {WIN}-day means", loc="left", fontsize=11.5, fontweight="bold", color=INK)
        if row == 0 and len(fcd):
            ax.text(t_last + pd.Timedelta(hours=30), 0.975, "forecast:\nresolved only", transform=ax.get_xaxis_transform(),
                    fontsize=8.3, color=MUTED, va="top")
        # the last-30-day split against MERRA-2
        bx = fig.add_subplot(gs[row, 1]); _style(bx); bx.grid(axis="x", visible=False)
        st = stats[f"{L}"]
        names = ["ep", "gw", "an", "ns", "ot", "sum", "dir"]
        vals = [st["last30"][c] for c in names]
        y = np.arange(len(names))[::-1]
        bx.barh(y, vals, color=[UP_C[c] for c in names], height=0.62, edgecolor="white", linewidth=1.0)
        for c, yy in zip(("ep", "gw", "an"), y[:3]):
            t = st["tests"][c]
            bx.plot([t["range_10_90"][0], t["range_10_90"][1]], [yy, yy], color=INK, lw=1.2)
            bx.plot([t["normal_on_geosfp_scale"]], [yy], marker="D", ms=5, color="white", mec=INK, mew=1.2)
        bx.axvline(0, color="#9aa1a9", lw=0.8)
        bx.set_yticks(y); bx.set_yticklabels(["resolved", "grav. wave", "increment", "non-steady", "other", "sum", "from ω̄*"], fontsize=9)
        bx.set_xlabel("mm s⁻¹", fontsize=9, color=INK)
        bx.set_title(f"Last {WIN} days", loc="left", fontsize=11, fontweight="bold", color=INK)
    fig.legend(*fig.axes[0].get_legend_handles_labels(), fontsize=8.8, frameon=False, ncol=4, loc="lower left",
               bbox_to_anchor=(0.06, 0.1), handlelength=2.4, columnspacing=1.6)
    fig.suptitle(f"GEOS FP · What drives the tropical upwelling · downward control, analyses to {t_last:%b %-d}",
                 x=0.066, y=0.975, ha="left", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.066, 0.925, "Mean residual vertical velocity w̄* between the turnaround latitudes, split by the zonal force that "
             "drives it (each part integrated down from the top)", fontsize=10, color=MUTED, va="top")
    t70 = stats["70"]["tests"]
    def sig(t):
        return f"p {t['p']:.2f}, " + ("significant" if t["significant_5pct"] else "not significant")
    foot = ("Right: bars = GEOS FP, last 30 days; ◇ = MERRA-2 normal put on the GEOS FP scale by the 2018–19 same-day difference; "
            "line = its 10–90 % range for a 30-day mean.\n"
            f"70 hPa against the normal: resolved {t70['ep']['anomaly']:+.3f} mm/s ({sig(t70['ep'])}), gravity waves "
            f"{t70['gw']['anomaly']:+.3f} ({sig(t70['gw'])}), increment {t70['an']['anomaly']:+.3f} ({sig(t70['an'])}).\n"
            "Normals: resolved and gravity-wave parts from MERRA-2 GMI (00Z every 10th day, 1991–2019), the increment from MERRA-2 "
            "monthly tendencies 2005–2024; turnaround latitudes fixed by day of year from MERRA-2.")
    fig.text(0.066, 0.012, foot, fontsize=8.5, color=MUTED, va="bottom", linespacing=1.45)
    _save(fig, path)


# -------------------------- sections -----------------------------------------------------------------------------------
SEC_TERMS = (("epd", "Resolved waves (EP-flux divergence)"), ("gwd", "Gravity-wave drag"), ("ana", "Analysis increment"))


def section_stats(Z, days, lat, R):
    """Last-WIN-day mean of each term per (level, latitude) against the seam-adjusted MERRA-2 normal; z-test per point,
    Benjamini-Hochberg false-discovery rate 10 % over 100-1 hPa and 80S-80N, per term."""
    w = np.asarray(days > days.max() - pd.Timedelta(days=WIN))
    doys = days[w].dayofyear.values
    mo = days[w].month.values - 1
    wm = np.bincount(mo, minlength=12) / w.sum()
    dom = ((LEVS >= 1) & (LEVS <= 100))[:, None] & (np.abs(lat) <= 80)[None, :]
    out = {}
    for key, _ in SEC_TERMS:
        X = Z[key][w]; xm = np.nanmean(X, 0)
        # GEOS FP's own sampling noise of a WIN-day mean (day-to-day variance / n, Bartlett-inflated): the floor under
        # MERRA-2's spread, which is ~0 wherever MERRA-2 has no drag or increment at all
        res_all = Z[key] - np.nanmean(Z[key], 0)
        infl_all = B.acf_inflation(np.where(np.isfinite(res_all), res_all, 0.0), days)
        var_own = np.nanvar(X, 0, ddof=1) / np.isfinite(X).sum(0) * infl_all
        if key == "epd":
            H_ = B.harm(doys)
            normal = np.tensordot(H_.mean(0), R["epd_coef"], axes=1)
            Xm = H_.mean(0); lev_ = float(Xm @ R["XtXi"] @ Xm)
            var_mu = R["epd_s2"] * lev_
            var_ia = np.maximum(R["epd_via30"], 0) + R["epd_via30_se"]
            clim_all = np.array([np.tensordot(B.harm([d])[0], R["epd_coef"], axes=1) for d in days.dayofyear.values])
            res = Z[key] - clim_all
            infl = B.acf_inflation(np.where(np.isfinite(res), res, 0.0), days)
            var_x = np.nanvar(X, 0, ddof=1) / np.isfinite(X).sum(0) * infl
            adj = normal + R["seam_epd"]; var_seam = R["seam_epd_se"] ** 2
        else:
            normal = np.tensordot(wm, R[f"m_{key}_mean"], axes=1)
            var_ia = np.tensordot(wm, R[f"m_{key}_sd"] ** 2, axes=1)
            var_mu = var_ia / len(R["m_years"])
            var_ia = np.maximum(var_ia, var_own); var_x = 0.0
            if key == "gwd":                     # GEOS FP's drag is a scaled-down MERRA-2 drag: ratio fitted per point
                adj = normal * R["seam_gwd_ratio"]; var_seam = (normal * R["seam_gwd_ratio_se"]) ** 2
            else:
                adj = normal + R["seam_ana"]; var_seam = R["seam_ana_se"] ** 2
        anom = xm - adj
        z = anom / np.sqrt(var_ia + var_mu + var_seam + var_x)
        p = np.where(dom, B.norm_p(z), np.nan)
        # tested over the whole domain; shaded and counted only where the difference is also >= SEC_MIN (practical size)
        sig = B.bh_fdr(p, 0.10) & (np.abs(anom) * DAY >= SEC_MIN)
        out[key] = {"mean": xm, "normal": normal, "adj": adj, "anom": anom, "sig": sig, "n": int(w.sum())}
    out["window"] = (days[w].min(), days.max())
    return out


SEC_LV = [-8, -4, -2, -1, -0.5, -0.25, 0.25, 0.5, 1, 2, 4, 8]
SEC_CL = [-4, -2, -1, -0.5, 0.5, 1, 2, 4]                  # normal contours (display-smoothed 1.5 deg in latitude)
SEC_MIN = 0.25                                             # m/s/day: smaller significant differences are not shaded


def fig_sections(sec, lat, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from scipy.ndimage import gaussian_filter1d
    k = (LEVS >= 1) & (LEVS <= 100); p = LEVS[k]
    lm = np.abs(lat) <= 80                                   # EP-flux divergence has 1/cos(lat)^2: meaningless at the pole
    la = lat[lm]
    # diverging: blues (westward force) - neutral - oranges (eastward), 11 classes + both extensions
    cols = ["#1b2750", "#26345f", "#3b4f8f", "#5a78b4", "#8ea8d2", "#c3d1e8", "#f7f7f5", "#f3c9a8", "#e59c6c", "#cc6a3a",
            "#9c3f1c", "#6e2710", "#4a1a0b"]
    cmap = ListedColormap(cols[1:-1]); cmap.set_under(cols[0]); cmap.set_over(cols[-1])
    norm = BoundaryNorm(SEC_LV, cmap.N)
    fig = plt.figure(figsize=(13.2, 8.8), dpi=130)
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.07, left=0.058, right=0.985, top=0.86, bottom=0.235)
    yt = [100, 50, 20, 10, 5, 2, 1]
    for j, (key, title) in enumerate(SEC_TERMS):
        s = sec[key]
        for row in (0, 1):
            ax = fig.add_subplot(gs[row, j])
            if row == 0:
                fld = s["mean"][k][:, lm] * DAY
                cf = ax.contourf(la, p, fld, levels=SEC_LV, cmap=cmap, norm=norm, extend="both")
                nrm = gaussian_filter1d(s["adj"][k][:, lm] * DAY, 1.5, axis=1, mode="nearest")
                c1 = ax.contour(la, p, nrm, levels=[x for x in SEC_CL if x > 0], colors=INK, linewidths=0.8)
                c2 = ax.contour(la, p, nrm, levels=[x for x in SEC_CL if x < 0], colors=INK, linewidths=0.8, linestyles="dashed")
                for cs in (c1, c2):
                    ax.clabel(cs, fontsize=7, fmt=lambda v: f"{v:g}", inline_spacing=1)
                ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK)
            else:
                an = np.where(s["sig"][k], s["anom"][k] * DAY, np.nan)[:, lm]
                ax.pcolormesh(la, p, np.ma.masked_invalid(an), cmap=cmap, norm=norm, shading="nearest")
                nsig = int(s["sig"].sum())
                ax.set_title(f"anomaly, significant points only ({nsig})" if nsig else "anomaly: no significant point",
                             loc="left", fontsize=10, color=INK)
            ax.set_yscale("log"); ax.set_ylim(100, 1); ax.set_xlim(-80, 80)
            ax.set_yticks(yt); ax.set_yticklabels([str(v) for v in yt] if j == 0 else [], fontsize=8.5); ax.minorticks_off()
            ax.set_xticks(np.arange(-60, 61, 30))
            ax.set_xticklabels([f"{abs(x)}°{'S' if x < 0 else ('N' if x > 0 else '')}" for x in range(-60, 61, 30)], fontsize=8.5)
            B._style(ax); ax.grid(False)
            if j == 0:
                ax.set_ylabel("hPa", fontsize=9.5, color=INK)
    cax = fig.add_axes([0.28, 0.14, 0.45, 0.022])
    cb = fig.colorbar(cf, cax=cax, orientation="horizontal", ticks=SEC_LV)
    cb.ax.tick_params(labelsize=8.5, colors=INK); cb.outline.set_visible(False)
    cb.set_label("m s⁻¹ per day (negative = westward force: decelerates westerlies)", fontsize=9.5, color=INK)
    d0, d1 = sec["window"]
    fig.suptitle(f"GEOS FP · Where the forces act · {WIN}-day mean {d0:%b %-d} – {d1:%b %-d %Y}", x=0.058, y=0.975, ha="left",
                 fontsize=15, fontweight="bold", color=INK)
    fig.text(0.058, 0.925, "Top: the 30-day mean (shaded) with the MERRA-2 normal for the same days on the GEOS FP scale (contours). "
             "Bottom: the difference, shaded only where significant.", fontsize=10, color=MUTED, va="top")
    foot = ("Test per point: z against MERRA-2's year-to-year spread of 30-day means (never less than GEOS FP's own sampling noise) "
            "plus the errors of the normal and of the 2018–19 adjustment;\n"
            f"false-discovery rate 10 % over 100–1 hPa and 80°S–80°N, and significant differences under {SEC_MIN} m s⁻¹ per day are "
            "not shaded. GEOS FP's drag is compared with MERRA-2's scaled by their measured ratio.\n"
            "Normals: resolved waves from MERRA-2 GMI (00Z every 10th day, 1991–2019); gravity-wave drag and increment from MERRA-2 "
            "monthly tendencies 2005–2024.")
    fig.text(0.058, 0.012, foot, fontsize=8.5, color=MUTED, va="bottom", linespacing=1.45)
    _save(fig, path)


def render(outdir=None):
    outdir = Path(outdir) if outdir else OUT / "figs"
    outdir.mkdir(parents=True, exist_ok=True)
    R = load(REF)
    days, Z, lat, audit = load_tail()
    fc = load_forecast()
    if fc is not None and fc["days"].min() > days.max() + pd.Timedelta(days=1):
        log(f"forecast {fc['cycle']} starts after a gap; drawn anyway")
    js = {"made": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%MZ"), "units": "m/s per day (w*: mm/s)",
          "analysis": [f"{days.min():%Y-%m-%d}", f"{days.max():%Y-%m-%d}"], "analysis_days": int(np.isfinite(Z["epd"][:, 0, 0]).sum()),
          "forecast_cycle": None if fc is None else fc["cycle"], "audit_days": audit,
          "closure": {}, "vortex": {}, "upwelling": {}, "sections": {}}
    last30 = days > days.max() - pd.Timedelta(days=WIN)
    for h, (lo, hi) in HEMI.items():
        for L in VLEVS:
            cl = closure(Z, days, lat, lo, hi, L)
            js["closure"][f"{h}_{L}"] = cl
            for mode in ("s7", "d1"):
                fig_vortex(days, Z, lat, fc, R, h, L, mode, cl, outdir / f"mbudget_vortex_{h}_{L}_{mode}")
            k = B.lev_index(L)
            ser = {key: band(Z[key][:, k], lat, lo, hi) * DAY for key in TERMS + ("epd_k1", "epd_k2", "epd_k3p", "ubar", "epd00")}
            ser["ubar"] = ser["ubar"] / DAY
            v = {"last30": {key: round(float(np.nanmean(ser[key][last30])), 3) for key in ser if key != "epd00"},
                 "sampling_00z_minus_daily_epd": round(float(np.nanmean(ser["epd00"] - ser["epd"])), 3),
                 "sampling_00z_minus_daily_epd_se": round(float(np.nanstd(ser["epd00"] - ser["epd"]) / np.sqrt(np.isfinite(ser["epd"]).sum())), 3),
                 "daily": {f"{d:%Y-%m-%d}": {key: (None if not np.isfinite(ser[key][i]) else round(float(ser[key][i]), 3))
                                             for key in TERMS + ("epd_k1", "epd_k2", "epd_k3p", "ubar")} for i, d in enumerate(days)}}
            if fc is not None:
                v["forecast"] = {f"{d:%Y-%m-%d}": {key: round(float(band(fc[key][i, k], lat, lo, hi) * DAY), 3)
                                                   for key in ("dudt", "cor", "vad", "epd", "epd_k1", "epd_k2", "epd_k3p")}
                                 for i, d in enumerate(fc["days"]) if d > days.max()}
            js["vortex"][f"{h}_{L}"] = v
    for lo, hi, L in ((-20, 20, 10), (-20, 20, 50), (-20, 20, 70)):
        js["closure"][f"tropics_{L}"] = closure(Z, days, lat, lo, hi, L)
    up, fup = up_parts(Z, days, lat, fc)
    ustats = {}
    for L in UP_LEVS:
        df = up[L]; w = df.index > days.max() - pd.Timedelta(days=WIN)
        nrm = up_normals(R, df.index[w].dayofyear.values, L)
        tests = {c: z_test(float(df[c][w].mean()), df[c][w].values, df.index[w], nrm[c]) for c in ("ep", "gw", "an")}
        ustats[f"{L}"] = {"last30": {c: round(float(df[c][w].mean()), 4) for c in df.columns}, "tests": tests,
                          "series_daily": {f"{d:%Y-%m-%d}": {c: (None if not np.isfinite(df.loc[d, c]) else round(float(df.loc[d, c]), 4))
                                                             for c in df.columns} for d in df.index},
                          "corr_sum_vs_direct_daily": round(float(df[["sum", "dir"]].dropna().corr().iloc[0, 1]), 2)}
        if fup is not None:
            ustats[f"{L}"]["forecast_ep"] = {f"{d:%Y-%m-%d}": round(float(fup[L].loc[d, "ep"]), 4) for d in fup[L].index if d > days.max()}
    js["upwelling"] = ustats
    fig_upwelling(up, fup, days, fc, R, ustats, outdir / "mbudget_upwelling")
    sec = section_stats(Z, days, lat, R)
    fig_sections(sec, lat, outdir / "mbudget_sections")
    js["sections"] = {"window": [f"{sec['window'][0]:%Y-%m-%d}", f"{sec['window'][1]:%Y-%m-%d}"],
                      **{key: {"significant_points": int(sec[key]["sig"].sum()),
                               "test": "z per point, BH-FDR 10 %, 100-1 hPa, 80S-80N"} for key, _ in SEC_TERMS}}
    (outdir / "mbudget.json").write_text(json.dumps(_json_safe(js), indent=1, allow_nan=False))
    return js


def _json_safe(o):
    """NaN / inf -> null and numpy scalars -> Python, recursively (a strict JSON file the page can always parse)."""
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def live(budget_min=12):
    t0 = time.time()
    fetch_tail(12, budget_min=budget_min)
    try:
        fetch_forecast()
    except Exception as e:                                               # noqa: BLE001
        log(f"forecast fetch failed ({str(e)[:120]}); rendering with the newest cached cycle")
    js = render()
    log(f"live update done in {(time.time() - t0) / 60:.1f} min")
    return js


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["tail", "day", "forecast", "clim", "render", "live", "seed", "stream-state", "stream-udt"])
    ap.add_argument("--days", type=int, default=100)
    ap.add_argument("--date")
    ap.add_argument("--audit-every", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "tail":
        fetch_tail(a.days, audit_every=a.audit_every)
    elif a.cmd.startswith("stream-"):
        fetch_stream(a.cmd.split("-")[1], a.days, a.audit_every)
    elif a.cmd == "day":
        dsn = B.open_ds(ASSIM_NP, True); dsu = B.open_ds(ASSIM_UDT, True)
        build_day(dsn, dsu, pd.Timestamp(a.date), audit=True)
    elif a.cmd == "forecast":
        fetch_forecast()
    elif a.cmd == "clim":
        build_clim()
    elif a.cmd == "render":
        js = render(a.out)
        print(json.dumps({k: v for k, v in js.items() if k in ("analysis", "forecast_cycle", "closure", "audit_days")}, indent=1))
    elif a.cmd == "live":
        live()
    elif a.cmd == "seed":
        import tarfile
        for f in sorted(TAIL.glob("*.npz")):
            compact(f)
        dest = OUT / "tail_seed.tar.gz"
        with tarfile.open(dest, "w:gz") as tf:
            for f in sorted(f for f in TAIL.glob("*.npz") if re.fullmatch(r"\d{8}\.npz", f.name)):
                tf.add(f, arcname=f"tail/{f.name}")
            for f in sorted(FCST.glob("*_00.npz"))[-1:]:
                tf.add(f, arcname=f"fcst/{f.name}")
        log(f"seed: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
