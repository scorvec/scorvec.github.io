#!/usr/bin/env python3
"""Tropical cyclones and the jet, from the AIFS-ENS control (member 0, the run the dynamic-tropopause charts draw).

Tracks: ECMWF's own tropical-cyclone tracker output for AIFS-ENS (open data, `...-360h-enfo-tf.bufr`, Google mirror
only), 6-hourly; subset with ensembleForecastType 1 is the unperturbed control (member number 51 in the file).

Outflow metric (Archambault, Bosart, Keyser & Cordeira 2013, MWR 141, 2325-2346; Archambault et al. 2015, MWR 143,
1122-1141): negative PV advection by the irrotational wind, -v_chi . grad(PV), in the 300-200 hPa layer. The divergent
outflow of a recurving storm carries low-PV air poleward against the PV gradient at the jet; that is what builds the
downstream ridge, sharpens the jet streak and launches the Rossby wave packet. Here:
  v_chi       from the layer-mean (300/250/200 hPa) wind by a spherical-harmonic inversion of the divergence
              (wind200_vpot.velocity_potential, truncation T106 ~1.7 deg)
  PV          Ertel PV on the pressure grid (dt_charts.pv_step), layer mean of 300/250/200 hPa
  index       area mean of the NEGATIVE part of the advection within 500 km of the storm centre, sign flipped
              (PVU/day; 0 = no outflow-jet interaction). A storm far from the PV gradient scores ~0 however strong it is.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

A_EARTH = 6.371e6
URL = "https://storage.googleapis.com/ecmwf-open-data/{d}/{h:02d}z/aifs-ens/0p25/enfo/{d}{h:02d}0000-360h-enfo-tf.bufr"
MISSING = -1e99
R_KM = 500.0


def fetch(date: str, hour: int, cache: Path) -> Path | None:
    """The AIFS-ENS track BUFR for one cycle, from the Google mirror only (user rule: no other ECMWF mirror)."""
    import requests
    cache.mkdir(parents=True, exist_ok=True)
    fp = cache / f"aifs_ens_tf_{date}{hour:02d}.bufr"
    if fp.exists() and fp.stat().st_size > 1000:
        return fp
    r = requests.get(URL.format(d=date, h=hour), timeout=60)
    if r.status_code != 200:
        print(f"  TC tracks not on the Google mirror ({r.status_code}); maps drawn without storms", flush=True)
        return None
    fp.write_bytes(r.content)
    return fp


def decode(fp: Path) -> list[dict]:
    """Control-member tracks: one dict per storm with steps (h), lat, lon (-180..180), pmsl (hPa), wind (m/s)."""
    import eccodes as ec
    storms = []
    with open(fp, "rb") as f:
        while True:
            h = ec.codes_bufr_new_from_file(f)
            if h is None:
                break
            try:
                ec.codes_set(h, "unpack", 1)
                sid = ec.codes_get(h, "#1#stormIdentifier").strip()
                name = ec.codes_get(h, "#1#longStormName").strip()
                ft = np.atleast_1d(ec.codes_get_array(h, "#1#ensembleForecastType"))
                mem = np.atleast_1d(ec.codes_get_array(h, "#1#ensembleMemberNumber"))
                ctl = np.where(ft == 1)[0]
                if not len(ctl):
                    continue                                     # the control did not carry this storm
                j = int(ctl[0])
                pick = lambda key: (lambda a: float(a[j] if a.size > 1 else a[0]))(np.atleast_1d(ec.codes_get_array(h, key)))
                rep = ec.codes_get(h, "#1#delayedDescriptorReplicationFactor")
                steps, la, lo, pm, ws = [], [], [], [], []
                # t=0: the control's own analysed centre (#2#), else the observed position (#1#) for a named storm
                a_la, a_lo = pick("#2#latitude"), pick("#2#longitude")
                if a_la < MISSING:
                    a_la, a_lo = ec.codes_get(h, "#1#latitude"), ec.codes_get(h, "#1#longitude")
                if a_la > MISSING:
                    steps.append(0); la.append(a_la); lo.append(a_lo)
                    pm.append(pick("#1#pressureReducedToMeanSeaLevel")); ws.append(pick("#1#windSpeedAt10M"))
                for i in range(1, rep + 1):
                    y = pick(f"#{2 * i + 2}#latitude")
                    if y < MISSING:
                        continue
                    steps.append(int(round(pick(f"#{i}#timePeriod")))); la.append(y); lo.append(pick(f"#{2 * i + 2}#longitude"))
                    pm.append(pick(f"#{i + 1}#pressureReducedToMeanSeaLevel")); ws.append(pick(f"#{i + 1}#windSpeedAt10M"))
                if not steps:
                    continue
                pm = [p / 100.0 if p > MISSING else np.nan for p in pm]
                ws = [w if w > MISSING else np.nan for w in ws]
                # genesis storms carry a placeholder id (70-99 + basin letter) and no name
                named = not (sid[:2].isdigit() and int(sid[:2]) >= 70)
                storms.append(dict(id=sid, name=name if named else "", named=named, member=int(mem[j]),
                                   steps=np.array(steps), lat=np.array(la), lon=(np.array(lo) + 180) % 360 - 180,
                                   pmsl=np.array(pm, float), wind=np.array(ws, float)))
            finally:
                ec.codes_release(h)
    return storms


def nh_storms(storms: list[dict], min_wind: float = 17.0) -> list[dict]:
    """Northern-Hemisphere storms that reach tropical-storm strength (10-m wind >= 17 m/s) somewhere in the control."""
    keep = [s for s in storms if (s["lat"] > 0).any() and np.nanmax(np.r_[s["wind"], 0]) >= min_wind]
    # the tracker also lists an observed storm again under a genesis id: drop a genesis track that sits within
    # 150 km of a named one at any shared step
    named = [s for s in keep if s["named"]]
    def dup(g):
        for n in named:
            for h in np.intersect1d(g["steps"], n["steps"]):
                a, b = at(g, int(h)), at(n, int(h))
                if _km(a[0], a[1], b[0], b[1]) < 150:
                    return True
        return False
    return [s for s in keep if s["named"] or not dup(s)]


def _km(la1, lo1, la2, lo2) -> float:
    p1, p2, dl = np.deg2rad(la1), np.deg2rad(la2), np.deg2rad(lo2 - lo1)
    return float(A_EARTH / 1000 * np.arccos(np.clip(np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(dl), -1, 1)))


def label(s: dict) -> str:
    return s["name"].title() if s["named"] else f"new storm ({s['id']})"


def recurvature(s: dict) -> int | None:
    """Lead (h) where the storm's 24-h zonal motion turns from westward (<= -1 m/s) to eastward (>= +2 m/s); None if it
    never turns, or never moved west in the forecast (a storm heading north or east from the start)."""
    st, la = s["steps"].astype(float), s["lat"]
    lo = np.unwrap(np.deg2rad(s["lon"])) * 180 / np.pi
    if len(st) < 5:
        return None
    u = np.full(len(st), np.nan)
    for k in range(len(st)):
        a = np.where(np.abs(st - (st[k] - 12)) < 1e-6)[0]; b = np.where(np.abs(st - (st[k] + 12)) < 1e-6)[0]
        if len(a) and len(b):
            dx = (lo[b[0]] - lo[a[0]]) * 111.2e3 * np.cos(np.deg2rad(la[k]))
            u[k] = dx / (24 * 3600)
    west = np.where(u <= -1.0)[0]
    if not len(west):
        return None
    east = np.where((u >= 2.0) & (np.arange(len(u)) > west[0]))[0]
    if not len(east):
        return None
    # the turn: the last westward point before that eastward motion
    k0 = int(np.where((u <= 0) & (np.arange(len(u)) < east[0]))[0][-1])
    return int(st[k0])


def at(s: dict, h: int):
    """(lat, lon) of the storm at lead h if the control has a position there, else None."""
    k = np.where(s["steps"] == h)[0]
    return (float(s["lat"][k[0]]), float(s["lon"][k[0]])) if len(k) else None


def layer_wind(ds_u, ds_v, levels=(300, 250, 200)):
    """Layer-mean u, v over the given levels (global DataArrays, dims step/isobaricInhPa/latitude/longitude)."""
    return ds_u.sel(isobaricInhPa=list(levels)).mean("isobaricInhPa"), ds_v.sel(isobaricInhPa=list(levels)).mean("isobaricInhPa")


def irrotational(u2d, v2d, lat, lon):
    """(u_chi, v_chi) interpolated onto the target lat/lon grid (lat ascending, lon 0..360)."""
    import xarray as xr
    from wind200_vpot import irrotational_wind, velocity_potential
    chi, dlat, dlon = velocity_potential(u2d, v2d)
    uc, vc = irrotational_wind(chi, dlat, dlon)
    dlon_c = np.asarray(dlon, float)
    if dlon_c[-1] < 360.0 - 1e-6:                                         # close the circle so interpolation reaches 360
        dlon_c = np.r_[dlon_c, dlon_c[0] + 360.0]
        uc = np.concatenate([uc, uc[:, :1]], axis=1); vc = np.concatenate([vc, vc[:, :1]], axis=1)
    keep = np.r_[True, np.diff(dlat) != 0]                                # DH2 lats are unique, but be safe
    dlat, uc, vc = np.asarray(dlat)[keep], uc[keep], vc[keep]
    U = xr.DataArray(uc, dims=("lat", "lon"), coords={"lat": dlat, "lon": dlon_c}).sortby("lat")
    V = xr.DataArray(vc, dims=("lat", "lon"), coords={"lat": dlat, "lon": dlon_c}).sortby("lat")
    tgt = dict(lat=xr.DataArray(lat, dims="y"), lon=xr.DataArray(lon % 360, dims="x"))
    return U.interp(**tgt).values, V.interp(**tgt).values


def pv_advection(uchi, vchi, pv_layer, lat, lon, sigma_deg=1.0):
    """-v_chi . grad(PV) in PVU/day on the (lat, lon) grid; PV in PVU, Gaussian-smoothed by sigma_deg first (the 0.25 deg
    PV gradient is grid-noisy; v_chi is already T106-smooth, so this matches the two scales)."""
    from scipy.ndimage import gaussian_filter
    n = sigma_deg / abs(lat[1] - lat[0])
    pv_layer = gaussian_filter(np.nan_to_num(pv_layer, nan=0.0), sigma=(n, n), mode=("nearest", "wrap"))
    phi = np.deg2rad(lat); lam = np.deg2rad(lon)
    cosphi = np.clip(np.cos(phi), 1e-3, None)[:, None]
    dlam = lam[1] - lam[0]
    dpdx = (np.roll(pv_layer, -1, -1) - np.roll(pv_layer, 1, -1)) / (2 * dlam) / (A_EARTH * cosphi)
    dpdy = np.gradient(pv_layer, phi, axis=0) / A_EARTH
    return -(uchi * dpdx + vchi * dpdy) * 86400.0


def disc_index(adv, lat, lon, clat, clon, r_km=R_KM) -> float:
    """Area mean of max(-adv, 0) within r_km of (clat, clon): the outflow-jet interaction index, PVU/day."""
    j = np.where(np.abs(lat - clat) <= r_km / 111.0 + 0.5)[0]
    if not len(j):
        return np.nan
    la = np.deg2rad(lat[j])[:, None]; lo = np.deg2rad(lon)[None, :]
    c0, l0 = np.deg2rad(clat), np.deg2rad(clon)
    d = A_EARTH / 1000 * np.arccos(np.clip(np.sin(la) * np.sin(c0) + np.cos(la) * np.cos(c0) * np.cos(lo - l0), -1, 1))
    m = d <= r_km
    if not m.any():
        return np.nan
    w = np.broadcast_to(np.cos(la), m.shape)[m]
    return float(np.sum(np.maximum(-adv[j][m], 0) * w) / np.sum(w))


def within(lat, lon, clat, clon, r_km) -> np.ndarray:
    """Boolean (lat, lon) mask of points within r_km of (clat, clon)."""
    la = np.deg2rad(lat)[:, None]; lo = np.deg2rad(lon)[None, :]
    c0, l0 = np.deg2rad(clat), np.deg2rad(clon)
    cosd = np.sin(la) * np.sin(c0) + np.cos(la) * np.cos(c0) * np.cos(lo - l0)
    return cosd >= np.cos(r_km / (A_EARTH / 1000))


COLS = ["#c0392b", "#1f5fa8", "#23874f", "#8e44ad", "#d35400", "#16a085", "#7f8c8d", "#b7950b"]


def render_card(storms, idx, v250, lat, lon, steps, init, out_png: Path, out_json: Path, band=(35.0, 60.0), lon0=100.0):
    """Two panels: the outflow index per storm against lead, and a Hovmöller of 250 hPa v (35-60N) with the storms'
    longitudes and recurvature points, so a wave packet launched by a recurving storm can be followed downstream."""
    import json
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    out_png.parent.mkdir(parents=True, exist_ok=True); out_json.parent.mkdir(parents=True, exist_ok=True)
    days = np.asarray(steps, float) / 24.0
    fig = plt.figure(figsize=(12.6, 11.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.9], hspace=0.30, left=0.07, right=0.97, top=0.93, bottom=0.07)
    ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    fig.suptitle(f"Tropical cyclones and the jet — AIFS-ENS control, init {init:%d %b %Y %HZ}", fontsize=14, fontweight="bold",
                 x=0.07, ha="left")
    rec = {}
    for i, s in enumerate(storms):
        c = COLS[i % len(COLS)]
        hs = sorted(idx.get(s["id"], {}))
        if hs:
            ax1.plot(np.array(hs) / 24, [idx[s["id"]][h] for h in hs], "-o", color=c, lw=2.2, ms=4.5, label=label(s))
        r = recurvature(s); rec[s["id"]] = r
        if r is not None and r in idx.get(s["id"], {}):
            ax1.plot([r / 24], [idx[s["id"]][r]], marker="*", ms=17, color=c, mec="#000", mew=0.8, zorder=5)
    ax1.set_xlim(0, days[-1] if len(days) else 10); ax1.set_ylim(bottom=0)
    ax1.set_xlabel("forecast day"); ax1.set_ylabel("PVU per day")
    ax1.set_title("Outflow meets the jet: negative PV advection by the irrotational wind, 300–200 hPa, within 500 km of the storm"
                  "  (★ recurvature)", fontsize=10.5, loc="left")
    ax1.grid(alpha=0.35)
    if storms:
        ax1.legend(loc="upper left", fontsize=9.5, ncol=min(4, len(storms)), frameon=False)
    else:
        ax1.text(0.5, 0.5, "No tropical storms in the Northern Hemisphere in the control run", transform=ax1.transAxes,
                 ha="center", va="center", fontsize=12, color="#6f6b64")
    # Hovmöller of v250 averaged over the band, longitude starting at lon0
    j = (lat >= band[0]) & (lat <= band[1])
    w = np.cos(np.deg2rad(lat[j]))
    hov = (v250[:, j, :] * w[None, :, None]).sum(1) / w.sum()
    lon_s = (lon - lon0) % 360 + lon0
    o = np.argsort(lon_s)
    cf = ax2.contourf(lon_s[o], days, hov[:, o], levels=np.arange(-32, 33, 4), cmap="RdBu_r", extend="both")
    ax2.set_ylim(days[-1], 0)
    tick = np.arange(lon0, lon0 + 361, 30)
    ax2.set_xticks(tick); ax2.set_xticklabels([f"{int(((t + 180) % 360) - 180)}°" for t in tick])
    ax2.set_xlim(lon0, lon0 + 360)
    for i, s in enumerate(storms):
        c = COLS[i % len(COLS)]
        m = (s["steps"] <= days[-1] * 24) & (s["lat"] > 0)
        if not m.any():
            continue
        ls = (s["lon"][m] - lon0) % 360 + lon0
        tt = s["steps"][m] / 24.0
        brk = np.where(np.abs(np.diff(ls)) > 180)[0] + 1                # crossing the diagram's seam: break the line
        ls_p, tt_p = np.insert(ls, brk, np.nan), np.insert(tt, brk, np.nan)
        ax2.plot(ls_p, tt_p, color=c, lw=3.0)
        ax2.plot(ls_p, tt_p, color="#fff", lw=1.0)
        ax2.annotate(label(s), (ls[0], tt[0]), xytext=(5, -4), textcoords="offset points", color=c, fontsize=9.5,
                     fontweight="bold", va="top", ha="left",
                     path_effects=[pe.Stroke(linewidth=2.4, foreground="#fff"), pe.Normal()])
        r = rec[s["id"]]
        if r is not None and r <= days[-1] * 24:
            k = np.where(s["steps"][m] == r)[0]
            if len(k):
                ax2.plot(ls[k[0]], r / 24, marker="*", ms=19, color=c, mec="#000", mew=0.8, zorder=6)
    ax2.set_ylabel("forecast day"); ax2.set_xlabel("longitude")
    ax2.set_title(f"250 hPa meridional wind, {band[0]:.0f}–{band[1]:.0f}°N mean (m/s): red = southerly, blue = northerly. "
                  "A packet launched by a storm's outflow tilts down and to the right.", fontsize=10.5, loc="left")
    cb = fig.colorbar(cf, ax=ax2, orientation="horizontal", fraction=0.04, pad=0.08)
    cb.set_label("v250 (m/s)")
    fig.text(0.07, 0.012, "Tracks: ECMWF tropical-cyclone tracker on AIFS-ENS (open data, CC BY 4.0), control member. "
             "Metric after Archambault et al. (2013, 2015, Mon. Wea. Rev.).\n"
             "Index = area mean of the negative part of −v_χ·∇PV (sign flipped), irrotational wind from the 300–200 hPa layer (T106), PV smoothed 1°. "
             "★ = the turn from westward to eastward motion.", fontsize=8.4, color="#6f6b64")
    fig.savefig(out_png, dpi=100, facecolor="white", pil_kwargs={"quality": 85, "method": 6}); plt.close(fig)
    js = {"init": f"{init:%Y-%m-%dT%H:%MZ}", "radius_km": R_KM, "units": "PVU/day",
          "storms": [{"id": s["id"], "name": label(s), "named": s["named"], "recurvature_h": rec[s["id"]],
                      "track": [{"h": int(h), "lat": round(float(y), 2), "lon": round(float(x), 2),
                                 "pmsl": None if not np.isfinite(p) else round(float(p), 1),
                                 "wind": None if not np.isfinite(wv) else round(float(wv), 1)}
                                for h, y, x, p, wv in zip(s["steps"], s["lat"], s["lon"], s["pmsl"], s["wind"])],
                      "index": {str(h): round(float(v), 3) for h, v in sorted(idx.get(s["id"], {}).items()) if np.isfinite(v)}}
                     for s in storms]}
    out_json.write_text(json.dumps(js))
    print(f"  wrote {out_png} and {out_json}", flush=True)
