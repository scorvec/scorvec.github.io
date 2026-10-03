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
MIDLAT = 25.0          # a storm is drawn on the packet Hovmöller only once it is this far north
PACKET_SPEED = 25.0    # deg longitude per day: typical Rossby wave-packet (group) speed, the dotted guide


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
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.9], hspace=0.34, left=0.07, right=0.97, top=0.93, bottom=0.095)
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
    # Wave-packet Hovmöller (redesigned 2026-10-02, user: the v250 Hovmöller was "confusing"): the ENVELOPE of the 250 hPa
    # meridional wind (Zimin et al. 2003: zonal wavenumbers 4-15, Hilbert transform along each latitude circle), averaged
    # over the band. A Rossby wave packet is one bright streak instead of alternating red/blue stripes; a storm appears
    # only once it reaches the mid-latitudes (>= MIDLAT N), with a star at recurvature and a dotted guide at a typical
    # packet speed, so a packet the storm launches reads as a streak starting at the star.
    env = packet_envelope(v250, lon)
    j = (lat >= band[0]) & (lat <= band[1])
    w = np.cos(np.deg2rad(lat[j]))
    hov = (env[:, j, :] * w[None, :, None]).sum(1) / w.sum()
    lon_s = (lon - lon0) % 360 + lon0
    o = np.argsort(lon_s)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("pk", ["#ffffff", "#fde9b6", "#fbbf5e", "#f08a2c", "#d4501f", "#9e2117", "#5c0d17"])
    cf = ax2.contourf(lon_s[o], days, hov[:, o], levels=np.arange(0, 41, 4), cmap=cmap, extend="max")
    ax2.set_ylim(days[-1], 0)
    tick = np.arange(lon0, lon0 + 361, 30)
    ax2.set_xticks(tick); ax2.set_xticklabels([f"{int(((t + 180) % 360) - 180)}°" for t in tick])
    ax2.set_xlim(lon0, lon0 + 360)
    shown = 0
    for i, s in enumerate(storms):
        c = COLS[i % len(COLS)]
        m = (s["steps"] <= days[-1] * 24) & (s["lat"] >= MIDLAT)
        if not m.any():
            continue
        shown += 1
        ls = (s["lon"][m] - lon0) % 360 + lon0
        tt = s["steps"][m] / 24.0
        brk = np.where(np.abs(np.diff(ls)) > 180)[0] + 1
        ls_p, tt_p = np.insert(ls, brk, np.nan), np.insert(tt, brk, np.nan)
        ax2.plot(ls_p, tt_p, color="#111", lw=4.2, solid_capstyle="round")
        ax2.plot(ls_p, tt_p, color=c, lw=2.6, solid_capstyle="round")
        ax2.annotate(label(s), (ls[0], tt[0]), xytext=(-6, 0), textcoords="offset points", color=c, fontsize=10,
                     fontweight="bold", va="center", ha="right",
                     path_effects=[pe.Stroke(linewidth=2.6, foreground="#fff"), pe.Normal()])
        r = rec[s["id"]]
        k0 = None
        if r is not None and r <= days[-1] * 24:
            kk = np.where(s["steps"] == r)[0]
            if len(kk):
                k0 = kk[0]
        if k0 is None:                                    # no recurvature in range: guide from where it reaches mid-latitudes
            k0 = np.where(m)[0][0]
        x0 = (s["lon"][k0] - lon0) % 360 + lon0; t0 = s["steps"][k0] / 24.0
        tg = np.linspace(t0, days[-1], 30); xg = x0 + PACKET_SPEED * (tg - t0)
        ok = xg <= lon0 + 360
        ax2.plot(xg[ok], tg[ok], ls=(0, (1.5, 2.5)), color="#222", lw=1.4)
        if r is not None and r <= days[-1] * 24 and len(np.where(s["steps"] == r)[0]):
            ax2.plot(x0, t0, marker="*", ms=19, color=c, mec="#000", mew=0.8, zorder=6)
    if not shown:
        ax2.text(0.5, 0.04, f"No storm reaches {MIDLAT:.0f}°N in the control run within day {days[-1]:.0f}",
                 transform=ax2.transAxes, ha="center", fontsize=11, color="#444",
                 bbox=dict(boxstyle="round", fc="#fff", ec="#bbb"))
    ax2.set_ylabel("forecast day  (time runs down)"); ax2.set_xlabel("longitude  (east is to the right)")
    ax2.set_title(f"Wave packets along the jet: strength of the 250 hPa trough/ridge pattern, {band[0]:.0f}–{band[1]:.0f}°N (m/s)\n"
                  f"Bright streaks = packets of strong troughs and ridges, moving east as they slant down. Coloured lines: storms once north "
                  f"of {MIDLAT:.0f}°N, ★ recurvature,\n"
                  f"dotted: where a packet the storm starts would be at a typical {PACKET_SPEED:.0f}° longitude per day",
                  fontsize=10.2, loc="left")
    cb = fig.colorbar(cf, ax=ax2, orientation="horizontal", fraction=0.04, pad=0.09)
    cb.set_label("wave-packet amplitude: envelope of the 250 hPa north–south wind, zonal wavenumbers 4–15 (m/s)")
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


def packet_envelope(v, lon, kmin=4, kmax=15):
    """Envelope of the meridional wind along each latitude circle (Zimin et al. 2003, MWR 131): keep zonal wavenumbers
    kmin..kmax, build the analytic signal (positive wavenumbers doubled) and take its modulus. v: (..., lon)."""
    F = np.fft.fft(np.nan_to_num(v), axis=-1)
    n = v.shape[-1]
    Z = np.zeros_like(F)
    Z[..., kmin:kmax + 1] = 2.0 * F[..., kmin:kmax + 1]
    return np.abs(np.fft.ifft(Z, axis=-1))


# ── following the storm after the tracker's last fix (2026-10-03) ─────────────────────────────────────────────────────
# ECMWF's tracker stops at (or soon after) extratropical transition, which is exactly when a recurving storm phases with
# a mid-latitude trough and can deepen explosively (Choi-Wan 2026-10-02 00Z: dropped at +120 h near 40N 153E, 979 hPa;
# the control's low is 950 hPa near 57N 170E 24 h later, and 36 of 50 members are below 960 hPa at day 6). follow_low
# continues the track as the surface-pressure minimum, conservatively, so a different low is never picked up.
FOLLOW_REACH_KM = 1800.0     # per 12 h: a transitioning storm's centre can re-form far downstream along the jet
FOLLOW_SLOW_KM = 900.0       # a step longer than this must point roughly along the previous motion ...
FOLLOW_CONE_DEG = 75.0       # ... within this angle
FOLLOW_MIN_DEPTH = 2.0       # hPa below the 400-600 km ring mean: a closed low
FOLLOW_OTHER_KM = 500.0      # a low that existed 12 h earlier within this distance of a candidate (and was not our storm)
                             # makes that candidate a different system
FOLLOW_SAME_KM = 900.0       # a low within this distance of our storm at the previous step is part of the same system (a
                             # phasing storm is often double-centred for a step), so it never blocks a candidate
FOLLOW_MIDLAT = 30.0         # long jumps (re-forming along the jet) only north of this latitude; in the tropics a step is
                             # limited to FOLLOW_SLOW_KM
FOLLOW_DECAY_HPA = 15.0      # once the low has filled this much from its deepest, long jumps stop (a decaying low does not
                             # re-form downstream; a long hop then lands on another system)
FOLLOW_FILLED_HPA = 1010.0   # a low that has filled to this central pressure is no longer followed (over high terrain the
                             # sea-level reduction makes weak "lows" that are not storms)


def _gc_km(lat, lon, clat, clon):
    """Great-circle distance (km) from (clat, clon) to every point of a (lat, lon) grid (1-D axes)."""
    la = np.deg2rad(np.asarray(lat))[:, None]; lo = np.deg2rad(np.asarray(lon))[None, :]
    c0, l0 = np.deg2rad(clat), np.deg2rad(clon)
    cosd = np.sin(la) * np.sin(c0) + np.cos(la) * np.cos(c0) * np.cos(lo - l0)
    return A_EARTH / 1000 * np.arccos(np.clip(cosd, -1, 1))


def _bearing(lat1, lon1, lat2, lon2):
    p1, p2, dl = np.deg2rad(lat1), np.deg2rad(lat2), np.deg2rad(lon2 - lon1)
    return float(np.rad2deg(np.arctan2(np.sin(dl) * np.cos(p2), np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl))))


def closed_lows(f, lat, lon, centre, radius_km, smooth_deg=0.5, sep_deg=2.5):
    """Closed lows in field f (hPa, lat ascending, lon 0..360) within radius_km of centre=(lat, lon):
    list of (lat, lon(-180..180), central pressure, depth). A low = a local minimum of the lightly smoothed field over a
    sep_deg box whose raw central pressure (min within 120 km) is >= FOLLOW_MIN_DEPTH below the 400-600 km ring mean."""
    from scipy.ndimage import gaussian_filter, minimum_filter
    n = abs(lat[1] - lat[0]); lon360 = np.asarray(lon) % 360
    clat, clon = centre
    dl = radius_km / 111.0 + 7.0
    j = np.where(np.abs(lat - clat) <= dl)[0]
    dlon = ((lon360 - (clon % 360) + 180) % 360) - 180
    i = np.where(np.abs(dlon) <= dl / max(np.cos(np.deg2rad(min(abs(clat) + dl, 85))), 0.1))[0]
    if not len(j) or not len(i):
        return []
    sub = np.nan_to_num(f[np.ix_(j, i)], nan=1013.0)
    fs = gaussian_filter(sub, sigma=smooth_deg / n, mode="nearest")
    mn = minimum_filter(fs, size=max(3, int(round(sep_deg / n))) | 1, mode="nearest")
    out = []
    la_s, lo_s = lat[j], lon360[i]
    dist = _gc_km(la_s, lo_s, clat, clon % 360)
    for a, b in np.argwhere((fs == mn) & (dist <= radius_km)):
        if a in (0, len(j) - 1) or b in (0, len(i) - 1):
            continue                                                       # on the window edge: not resolved
        c_la, c_lo = float(la_s[a]), float(lo_s[b])
        dd = _gc_km(la_s, lo_s, c_la, c_lo)
        ring, core = (dd >= 400) & (dd <= 600), dd <= 120
        if not ring.any() or not core.any():
            continue
        pmin = float(sub[core].min()); depth = float(sub[ring].mean() - pmin)
        if depth >= FOLLOW_MIN_DEPTH:
            out.append((c_la, ((c_lo + 180) % 360) - 180, pmin, depth))
    return out


def _step_to(lat0, lon0, dlat, dlon_deg):
    return lat0 + dlat, ((lon0 + dlon_deg + 180) % 360) - 180


def follow_low(msl, lat, lon, steps_h, track, h_max=None):
    """Continue `track` (dict steps/lat/lon[/pmsl]) beyond its last fix as a surface low, every field step.

    msl: (len(steps_h), nlat, nlon) hPa, lat ascending, lon 0..360; steps_h: lead hours (12-hourly). Starts at the
    tracker's last fix that falls on a field step. Each step: the candidates are the closed lows within FOLLOW_REACH_KM
    (scaled by the time step) of the previous position; a candidate further than FOLLOW_SLOW_KM must lie within
    FOLLOW_CONE_DEG of the previous motion (a centre re-forming downstream along the jet, as in a phasing transition, is
    allowed; a jump sideways or backwards is not); a candidate that is the continuation of ANOTHER low already present a
    step earlier (within FOLLOW_OTHER_KM of it, and closer to it than our storm was) is excluded. The survivor nearest to
    the persistence first guess (previous position + previous displacement) is taken; none = the end of the track.
    Returns the CONTINUATION only: dict(steps, lat, lon, pmsl)."""
    steps_h = np.asarray(steps_h, int)
    empty = dict(steps=np.array([], int), lat=np.array([]), lon=np.array([]), pmsl=np.array([]))
    fstep = set(steps_h.tolist())
    on = [int(h) for h in track["steps"] if int(h) in fstep]
    if not on:
        return empty
    h0 = on[-1]
    k0 = int(np.where(track["steps"] == h0)[0][0])
    la, lo = float(track["lat"][k0]), float(track["lon"][k0])
    prev = np.where(track["steps"] <= h0 - 12)[0]
    if len(prev):
        kp = prev[-1]; dt = (h0 - int(track["steps"][kp])) / 12.0
        v_lat = (la - float(track["lat"][kp])) / dt
        v_lon = (((lo - float(track["lon"][kp]) + 180) % 360) - 180) / dt
    else:
        v_lat = v_lon = 0.0
    out = {"steps": [], "lat": [], "lon": [], "pmsl": []}
    h_last = h0
    for k in np.where(steps_h > h0)[0]:
        h = int(steps_h[k])
        if h_max is not None and h > h_max:
            break
        dt12 = (h - h_last) / 12.0
        reach = FOLLOW_REACH_KM * dt12
        cands = closed_lows(msl[k], lat, lon, (la, lo), reach)
        if not cands:
            break
        mot = _bearing(la, lo, *_step_to(la, lo, v_lat, v_lon)) if (abs(v_lat) + abs(v_lon)) > 0.2 else None
        kp_ = np.where(steps_h == h_last)[0]
        before = closed_lows(msl[kp_[0]], lat, lon, (la, lo), reach + FOLLOW_OTHER_KM) if len(kp_) else []
        g_la, g_lo = _step_to(la, lo, v_lat * dt12, v_lon * dt12)
        best, bd = None, np.inf
        for c_la, c_lo, pmin, depth in cands:
            dist = _km(la, lo, c_la, c_lo)
            if pmin >= FOLLOW_FILLED_HPA:
                continue
            if dist > FOLLOW_SLOW_KM * dt12:
                if mot is None or min(la, c_la) < FOLLOW_MIDLAT or (out["pmsl"] and out["pmsl"][-1] - min(out["pmsl"]) >= FOLLOW_DECAY_HPA):
                    continue
                dang = abs(((_bearing(la, lo, c_la, c_lo) - mot + 180) % 360) - 180)
                if dang > FOLLOW_CONE_DEG:
                    continue
            other = False
            for b_la, b_lo, _, _ in before:
                if _km(b_la, b_lo, la, lo) < FOLLOW_SAME_KM:
                    continue                                               # our own storm (or its second centre) a step earlier
                dbc = _km(b_la, b_lo, c_la, c_lo)
                if dbc < FOLLOW_OTHER_KM and dbc < dist:
                    other = True; break
            if other:
                continue
            dg = _km(g_la, g_lo, c_la, c_lo)
            if dg < bd:
                best, bd = (c_la, c_lo, pmin), dg
        if best is None:
            break
        c_la, c_lo, pmin = best
        v_lat = (c_la - la) / dt12
        v_lon = (((c_lo - lo + 180) % 360) - 180) / dt12
        la, lo, h_last = c_la, c_lo, h
        out["steps"].append(h); out["lat"].append(la); out["lon"].append(lo); out["pmsl"].append(round(pmin, 1))
    return {k: np.array(v, int if k == "steps" else float) for k, v in out.items()} if out["steps"] else empty


def extended(track, cont):
    """The tracker's track followed by its continuation, as one track dict (pmsl NaN where the tracker had none)."""
    if cont is None or not len(cont["steps"]):
        return track
    p = track.get("pmsl", np.full(len(track["steps"]), np.nan))
    return dict(steps=np.r_[track["steps"], cont["steps"]].astype(int), lat=np.r_[track["lat"], cont["lat"]],
                lon=np.r_[track["lon"], cont["lon"]], pmsl=np.r_[p, cont["pmsl"]],
                followed=np.r_[np.zeros(len(track["steps"]), bool), np.ones(len(cont["steps"]), bool)])


# ── map helper (2026-10-03, user: "Good lord that map projection") ───────────────────────────────────────────────────
# The TC figures drew 160-200-degree-wide mid-latitude sectors on a flat lat-lon grid squeezed to fit ("auto" aspect),
# which flattened Siberia, Alaska and Canada into slivers. sector_axes() draws a sector the way weather maps do: Lambert
# conformal with standard parallels inside the band, the outline following the sector's own meridians and latitude
# circles (a fan, no empty corners), true aspect.
def sector_projection(lon0, lon1, lat0, lat1):
    import cartopy.crs as ccrs
    clon = (((lon0 + lon1) / 2.0) + 180.0) % 360.0 - 180.0
    span = lat1 - lat0
    return ccrs.LambertConformal(central_longitude=clon, central_latitude=(lat0 + lat1) / 2.0,
                                 standard_parallels=(lat0 + span / 4.0, lat1 - span / 4.0), cutoff=max(lat0 - 5, -10))


def sector_aspect(lon0, lon1, lat0, lat1, n=90):
    """height / width of the sector's outline in the projection (for laying out axes before drawing)."""
    import cartopy.crs as ccrs
    proj = sector_projection(lon0, lon1, lat0, lat1)
    lo = np.r_[np.linspace(lon0, lon1, n), np.linspace(lon1, lon0, n)]
    la = np.r_[np.full(n, lat0), np.full(n, lat1)]
    xy = proj.transform_points(ccrs.PlateCarree(), lo, la)
    return float((xy[:, 1].max() - xy[:, 1].min()) / (xy[:, 0].max() - xy[:, 0].min()))


def sector_axes(fig, rect, lon0, lon1, lat0, lat1, land=True, coast_res="50m", n=90):
    """A GeoAxes at rect=[x, y, w, h] (figure fraction) showing the sector lon0..lon1 (0..360, may exceed 360), lat0..lat1
    as a Lambert fan: true aspect, the outline along the sector's edges, coastlines, borders and light land."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.path as mpath
    proj = sector_projection(lon0, lon1, lat0, lat1)
    ax = fig.add_axes(rect, projection=proj)
    lo = np.r_[np.linspace(lon0, lon1, n), np.linspace(lon1, lon0, n), lon0]
    la = np.r_[np.full(n, lat0), np.full(n, lat1), lat0]
    xy = proj.transform_points(ccrs.PlateCarree(), lo, la)[:, :2]
    ax.set_xlim(xy[:, 0].min(), xy[:, 0].max()); ax.set_ylim(xy[:, 1].min(), xy[:, 1].max())
    ax.set_boundary(mpath.Path(xy), transform=ax.transData)
    if land:
        ax.add_feature(cfeature.LAND, facecolor="#f2f1ec", zorder=0)
    ax.coastlines(coast_res, lw=0.6, color="#333")
    ax.add_feature(cfeature.BORDERS, lw=0.3, edgecolor="#777")
    ax.gridlines(lw=0.3, color="#999", alpha=0.5, xlocs=range(-180, 181, 30), ylocs=range(0, 91, 15))
    return ax
