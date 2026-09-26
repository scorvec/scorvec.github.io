#!/usr/bin/env python3
"""GEFS extended (35-day) outlook: fetch the newest 00Z extended run, take anomalies against GEFS's own reforecast
climatology, and draw the GEFS page's figures in the GEPS page's format (2026-09-26; user: "a GEFS extended page that
mirrors the format of the GEPS page").

Data: NOAA GEFSv12 on S3 (noaa-gefs-pds, anonymous), pgrb2a 0.5 deg, 31 members (gec00 + gep01-30); only the 00Z
cycle runs to 840 h. Every field is byte-ranged through the per-step .idx (the site rule for NOAA models).

Daily values use EXACTLY the reforecast's sampling (gefs_reforecast.py): instantaneous fields = mean of the 12Z and
00Z samples ending each day, precipitation = the four 6-h accumulations summed, OLR = the four 6-h averages meaned;
1.5 deg box means; weekly means over days 1-7 ... 29-35. So forecast and climatology are the same arithmetic, and the
anomaly carries no sampling artefact.

Reference: the GEFSv12 reforecast 2000-2019 (release gefs-clim-v1), read at the INIT day of year and the same week of
lead - the model's drift and mean bias removed together. Anomalies are therefore against GEFS's 2000-2019 climate,
not a 1991-2020 normal; 500 hPa height has its per-week global mean removed (otherwise two decades of warming sit over
every map as a uniform ridge); temperature and precipitation keep theirs, since that is part of the signal.

The vortex panel: 60N 10 hPa zonal wind per member, bias-corrected as raw - model_clim(lead) + MERRA-2(valid date),
both 2000-2019 (data/merra2_u60_10_clim.json).

    python gefs_live.py [--date YYYYMMDD] [--members 31] [--site ~/scorvec.github.io]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gefs_reforecast as R                                           # noqa: E402  (decode, grid, weeks, v1 packing)
import gefs_daily as G                                               # noqa: E402  (v2: the shared daily reduction)

S3 = "https://noaa-gefs-pds.s3.amazonaws.com"
CLIM_URL = "https://github.com/scorvec/scorvec.github.io/releases/download/gefs-clim-v1/gefs_clim_{c:03d}.npz"


def get(url, rng=None):
    return R.get(url, rng)


def step_url(date, mem, h):
    return f"{S3}/gefs.{date}/00/atmos/pgrb2ap5/{mem}.t00z.pgrb2a.0p50.f{h:03d}"


def latest_cycle(explicit=None, weekdays=None) -> str | None:
    """Newest 00Z cycle that has reached 840 h. Daily (user 2026-09-26: "GEFS should run every day once the 00z run
    comes in"), so every run has a partner exactly 7 days earlier for the change maps."""
    if explicit:
        return explicit
    today = dt.datetime.utcnow().date()
    for back in range(0, 8):
        day = today - dt.timedelta(days=back)
        if weekdays and day.weekday() not in weekdays:
            continue
        d = day.strftime("%Y%m%d")
        try:
            get(step_url(d, "gep30", 840) + ".idx")
            return d
        except Exception:                                             # noqa: BLE001
            continue
    return None


def member(date: str, mem: str):
    """One member through gefs_daily.Acc - the reforecast v2's own reduction - or None if incomplete.

    -> dict: w_<tag> weekly (5, lat, lon) for the nine map fields, e_<tag> daily (35, lat, lon) for every field (summed
    into the ensemble mean by fetch), m_<tag> daily 2.5 deg, s vortex scalars (16, 35), band (3, 35, lon)."""
    jobs = []
    for h in range(6, 841, 6):
        url = step_url(date, mem, h)
        try:
            idx = R.index(url)
        except Exception:                                             # noqa: BLE001
            return None
        want = []
        if h % 12 == 0:
            want += [(t, f"{h} hour fcst") for t in G.TABLE if G.TABLE[t][3] == "inst"]
        want += [(t, f"{h - 6}-{h} hour {G.TABLE[t][3]} fcst") for t in G.TABLE if G.TABLE[t][3] != "inst"]
        for t, stext in want:
            name, lev = G.TABLE[t][4], G.TABLE[t][2]
            hit = [e for e in idx if e[4] == name and e[2] == lev and e[3] == stext]
            if not hit:
                return None
            jobs.append((t, (h + 23) // 24, url, hit[0][0], hit[0][1]))
    acc = G.Acc(tuple(G.TABLE))
    with ThreadPoolExecutor(24) as ex:
        for (t, d, *_), b in zip(jobs, ex.map(lambda j: get(j[2], (j[3], j[4])), jobs)):
            acc.add(t, d, R.decode(b))
    r = acc.finish()
    if r is None:
        return None
    out = {f"w_{t}": G.weekly(r[f"d_{t}"]).astype("float32") for t in G.MAPS}
    out.update({f"e_{t}": r[f"d_{t}"].astype("float32") for t in G.TABLE})
    out.update({f"m_{t}": r[f"m_{t}"].astype("float32") for t in G.MEMBER25})
    out["s"] = r["s"].astype("float32")
    out["band"] = np.stack([G.band(r[f"d_{t}"]) for t in G.BANDF]).astype("float32")
    return out


def fetch(date: str, n_members: int, work: Path, procs: int = 1):
    """Every member -> work/gefs2_<date>.npz: per-member weekly maps, 2.5 deg dailies, vortex scalars and MJO band,
    and the ENSEMBLE-MEAN daily fields (e_<tag>) - the member dailies themselves are not kept (2 GB a run)."""
    dest = work / f"gefs2_{date}.npz"
    if dest.exists():
        return dest
    mems = (["gec00"] + [f"gep{i:02d}" for i in range(1, 31)])[:n_members]
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    keep, esum, ms = {}, {}, []
    with ProcessPoolExecutor(procs) as ex:                            # decoding is the CPU cost: one member per core
        for m, r in zip(mems, ex.map(member, [date] * len(mems), mems)):
            print(f"  {m}: {'ok' if r else 'incomplete, skipped'} ({time.time() - t0:.0f} s)", flush=True)
            if not r:
                continue
            ms.append(m)
            for k, v in r.items():
                if k.startswith("e_"):
                    esum[k] = esum.get(k, 0) + v.astype("float64")
                else:
                    keep.setdefault(k, []).append(v)
    if len(ms) < max(1, (n_members + 1) // 2):              # at least half the members asked for
        raise SystemExit(f"only {len(ms)} complete members")
    work.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, members=np.array(ms), scalars=np.array([x[0] for x in G.SCALARS]),
                        **{k: np.stack(v).astype("float32") for k, v in keep.items()},
                        **{k: (v / len(ms)).astype("float32") for k, v in esum.items()})
    return dest


CLIM2_URL = "https://github.com/scorvec/scorvec.github.io/releases/download/gefs-clim-v2/gefs_clim2_{c:03d}.npz"
REF_URL = "https://github.com/scorvec/scorvec.github.io/releases/download/gefs-ref-v1/{name}"
CENTRES = list(range(1, 366, 5))


def _bracket(init: dt.date):
    """The two 5-day day-of-year centres around the init day and the weight of the upper one."""
    doy = init.timetuple().tm_yday
    lo = max(c for c in CENTRES if c <= doy)
    hi = lo + 5 if lo + 5 <= 361 else 1
    return lo, hi, (doy - lo) / 5.0


def _cached(url: str, p: Path) -> Path:
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".part"); tmp.write_bytes(get(url)); tmp.replace(p)
    return p


def clim_at(init: dt.date, cache: Path):
    """v1 WEEKLY reforecast climatology at the init day of year (linear between the two centres around it)."""
    lo, hi, w = _bracket(init)
    out = {}
    for c, wt in ((lo, 1 - w), (hi, w)):
        z = np.load(_cached(CLIM_URL.format(c=c), cache / f"gefs_clim_{c:03d}.npz"))
        for k in [f"w_{t}" for t in R.WEEKLY] + ["u60", "band"]:
            out[k] = out.get(k, 0) + wt * z[k].astype("float64")
    return out


def clim2_at(init: dt.date, cache: Path):
    """v2 DAILY reforecast climatology at the init day of year: d_<tag> (35, 121, 240) for every reforecast field,
    c25_<tag> on the 2.5 deg grid, s (16, 35) vortex scalars. None when the release is not there yet."""
    lo, hi, w = _bracket(init)
    out = {}
    try:
        for c, wt in ((lo, 1 - w), (hi, w)):
            z = np.load(_cached(CLIM2_URL.format(c=c), cache / f"gefs_clim2_{c:03d}.npz"))
            for k in z.files:
                if k.startswith("d_"):
                    v = G.unpack(k[2:], z[k])
                elif k.startswith("c25_"):
                    v = G.unpack(k[4:], z[k])
                elif k == "s":
                    v = z[k].astype("float64")
                else:
                    continue
                out[k] = out.get(k, 0) + wt * v
            out["scalars"] = [str(x) for x in z["scalars"]]
    except Exception as e:                                           # noqa: BLE001
        print(f"  daily climatology v2 unavailable ({str(e)[:80]}) - daily products skipped", flush=True)
        return None
    return out


def ref(name: str, cache: Path, local: Path | None = None) -> Path:
    """A reference file of release gefs-ref-v1 (built once on the laptop), or a local copy if given."""
    if local is not None and (local / name).exists():
        return local / name
    return _cached(REF_URL.format(name=name), cache / "ref" / name)


def shift_series(sh, key: str, days: list[dt.date]):
    """The ERA5 1991-2020 minus 2001-2020 normal at each valid date (linear in day of year, wrapping the year)."""
    doy = sh["doy"].astype(float); f = sh[key]
    x = np.r_[doy - 365.0, doy, doy + 365.0]; ff = np.concatenate([f, f, f])
    out = []
    for d in days:
        t = float(d.timetuple().tm_yday)
        j = int(np.searchsorted(x, t)); w = (t - x[j - 1]) / (x[j] - x[j - 1])
        out.append((1 - w) * ff[j - 1] + w * ff[j])
    return np.stack(out)


def valid_days(base: dt.date):
    """Day d (1..35) is the 12 h + 24 h sample mean: the UTC day init + (d - 1)."""
    return [base + dt.timedelta(days=d) for d in range(35)]


# ── figures (same geometry, projections, colour maps and wording as the GEPS page's render_maps.py) ──────────
FIELDS = {"t2m": ("2 m temperature", "°C", "RdBu_r", False, 1.0),
          "pr": ("Precipitation", "mm/day", "BrBG", False, 1.0),
          "zg500": ("500 hPa height", "m", "RdBu_r", True, 1.0),
          "olr": ("Outgoing longwave", "W m⁻²", "BrBG_r", False, 1.0),
          "u850": ("850 hPa zonal wind", "m/s", "PuOr_r", False, 1.0),
          "u200": ("200 hPa zonal wind", "m/s", "PuOr_r", False, 1.0),
          "v850": ("850 hPa meridional wind", "m/s", "PuOr_r", False, 1.0),
          "v200": ("200 hPa meridional wind", "m/s", "PuOr_r", False, 1.0),
          "mslp": ("Mean sea-level pressure", "hPa", "RdBu_r", False, 0.01)}
# the GEPS page's order (render_maps.ANIM_ORDER); the loops and the page's field list follow it
ANIM_ORDER = ["t2m", "pr", "mslp", "zg500", "olr", "u850", "u200", "v850", "v200"]
MASKABLE = ("t2m", "pr", "zg500", "mslp")                          # fields with a hindcast skill table
DOMAINS = {"na": dict(label="North America", extent=(-168, -52, 12, 72), central=-110, proj="lambert"),
           "sa": dict(label="South America", extent=(-95, -30, -58, 15), central=-62, proj="plate"),
           "glb": dict(label="Global", extent=(-180, 180, -60, 70), central=180, proj="merc", stack=True)}
SCALE_PCT = {"t2m": 99.3, "zg500": 98.0, "mslp": 99.0}
CONTOUR = {"mslp": (np.arange(880.0, 1084.0, 4.0), 0.6), "zg500": (np.arange(4680.0, 6121.0, 60.0), 0.5)}
MUTED = "#6f6b64"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def projection(dom):
    import cartopy.crs as ccrs
    k = dom.get("proj", "plate")
    if k == "lambert":
        return ccrs.LambertConformal(central_longitude=dom["central"], standard_parallels=(33, 45))
    if k == "merc":
        return ccrs.Mercator(central_longitude=dom["central"], min_latitude=dom["extent"][2], max_latitude=dom["extent"][3])
    return ccrs.PlateCarree(central_longitude=dom["central"])


_ASP = {}


def aspect(key, dom):
    import cartopy.crs as ccrs
    plt = _plt()
    if key in _ASP:
        return _ASP[key]
    if dom.get("proj") == "merc":
        f_ = lambda d: np.log(np.tan(np.pi / 4 + np.deg2rad(d) / 2))
        e = dom["extent"]; _ASP[key] = (f_(e[3]) - f_(e[2])) / (2 * np.pi)
    else:
        f = plt.figure(figsize=(4, 4)); a = f.add_subplot(1, 1, 1, projection=projection(dom))
        a.set_extent(dom["extent"], crs=ccrs.PlateCarree()); x0, x1, y0, y1 = a.get_extent(projection(dom))
        _ASP[key] = abs((y1 - y0) / (x1 - x0)); plt.close(f)
    return _ASP[key]


def _grid(stack, asp, n=5):
    plt = _plt()
    if stack:
        panel_w, top_in, bot_in = 11.0, 0.72, 1.15
        map_h = panel_w * asp; fig_w = panel_w + 0.35; fig_h = n * (map_h + 0.30) + top_in + bot_in
        fig = plt.figure(figsize=(fig_w, fig_h))
        rects = [[0.015, 1.0 - (top_in + (i + 1) * (map_h + 0.30)) / fig_h, 0.968, map_h / fig_h] for i in range(n)]
        return fig, rects, [0.36, 0.62 / fig_h, 0.28, 0.13 / fig_h], 0.12 / fig_h, 1.0 - 0.18 / fig_h
    panel_w, top_in, bot_in, gap_y = 5.8, 0.92, 0.98, 0.44
    map_h = panel_w * asp; rows = (n + 1) // 2; fig_w = 2 * panel_w + 0.30
    fig_h = rows * map_h + (rows - 1) * gap_y + top_in + bot_in
    fig = plt.figure(figsize=(fig_w, fig_h)); w_frac = panel_w / fig_w; rects = []
    for i in range(n):
        row, col = divmod(i, 2)
        y = 1.0 - (top_in + (row + 1) * map_h + row * gap_y) / fig_h
        x = (1.0 - w_frac) / 2 if (i == n - 1 and n % 2 == 1) else 0.010 + col * (w_frac + 0.006)
        rects.append([x, y, w_frac * 0.98, map_h / fig_h])
    return fig, rects, [0.36, 0.44 / fig_h, 0.28, 0.15 / fig_h], 0.06 / fig_h, 1.0 - 0.18 / fig_h


def _panel(ax, lat, lon, v, dom, lim, cmap, over=None, clev=None, clw=0.5):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    if dom.get("proj") == "merc":
        ax.set_global(); pr = ax.projection; e = dom["extent"]
        ax.set_ylim(pr.transform_point(0.0, e[2], ccrs.PlateCarree())[1], pr.transform_point(0.0, e[3], ccrs.PlateCarree())[1])
    else:
        ax.set_extent(dom["extent"], crs=ccrs.PlateCarree())
    lo = np.where(lon > 180, lon - 360, lon); o = np.argsort(lo)
    m = ax.pcolormesh(lo[o], lat, v[:, o], cmap=cmap, vmin=-lim, vmax=lim, shading="auto",
                      transform=ccrs.PlateCarree(), rasterized=True)
    if over is not None and clev is not None:
        lo_c = np.r_[lo[o], lo[o][0] + 360.0]; ov = np.concatenate([over[:, o], over[:, o][:, :1]], axis=1)
        cs = ax.contour(lo_c, lat, ov, levels=clev, colors="#1f1f1f", linewidths=clw, transform=ccrs.PlateCarree())
        ax.clabel(cs, cs.levels[::2], inline=True, fontsize=6, fmt="%.0f")
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.5, edgecolor="#3a3a3a")
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.3, edgecolor="#6a6a6a")
    ax.add_feature(cfeature.STATES.with_scale("50m"), lw=0.18, edgecolor="#9a9a9a")
    return m


def scale_for(v, pct=95):
    v = np.abs(v[np.isfinite(v)]); return float(np.nanpercentile(v, pct)) if v.size else 1.0


def domain_rows(dom):
    e = dom["extent"]; return (R.LAT >= e[2] - 3) & (R.LAT <= e[3] + 3)


def map_figure(tag, weeks_anom, weeks_fc, base, key, dom, out: Path, kind="weekly", titles=None, note="", cb_label=None,
               hatcher=None, suffix=""):
    """One weekly figure, the GEPS page's geometry. `hatcher(ax, i)` hatches panel i (the skill-masked variant)."""
    plt = _plt()
    title, unit, cmap, demean, scale = FIELDS[tag]
    rows = domain_rows(dom); lat = R.LAT[rows]
    means = [w[rows] * scale for w in weeks_anom]
    lim = max(scale_for(m, SCALE_PCT.get(tag, 95)) for m in means)
    stack = dom.get("stack", False)
    fig, rects, cax_rect, note_y, title_y = _grid(stack, aspect(key, dom), len(means))
    clev, clw = CONTOUR.get(tag, (None, 0.5))
    for i, (mn, rect) in enumerate(zip(means, rects)):
        ax = fig.add_axes(rect, projection=projection(dom))
        over = (weeks_fc[i][rows] * scale) if (weeks_fc is not None and kind == "weekly") else None
        mesh = _panel(ax, lat, R.LON, mn, dom, lim, cmap, over=over, clev=clev, clw=clw)
        if hatcher is not None:
            hatcher(ax, i)
        ax.set_title(titles[i] if stack else titles[i].replace("   ", "\n", 1), fontsize=10.5, fontweight="bold", pad=4,
                     loc="left" if stack else "center")
    cax = fig.add_axes(cax_rect)
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal", extend="both")
    cb.ax.xaxis.set_label_position("top")
    cb.set_label(cb_label or (f"{title} anomaly vs 1991–2020 ({unit})"
                              + ("   ·   contours: ensemble mean" if tag in CONTOUR else "")), fontsize=10, labelpad=4)
    cb.ax.tick_params(labelsize=8.5, pad=1.5)
    head = "weekly means" if kind == "weekly" else "change vs previous run"
    fig.suptitle(f"{dom['label']} — {title}, {head} · GEFS init {base:%Y-%m-%d}", fontsize=15, fontweight="bold", y=title_y, va="top")
    if note:
        fig.text(0.5, note_y, note, ha="center", va="bottom", fontsize=8.5, color=MUTED)
    p = out / f"gefs_{key}_{tag}_{kind}{suffix}.webp"
    fig.savefig(p, dpi=105, facecolor="white", pil_kwargs={"quality": 88, "method": 6})
    plt.close(fig)
    return p.name


COSW = np.cos(np.deg2rad(R.LAT))[:, None] * np.ones((1, len(R.LON)))


def global_mean(w):
    return float((w * COSW).sum() / COSW.sum())


def anomalies(z, C2, sh, base, Dr=None):
    """Daily ENSEMBLE-MEAN anomalies vs 1991-2020 for the nine map fields: [F - M(init doy, lead)] - shift(valid day),
    M = the GEFS 2000-2019 reforecast climatology, shift = ERA5 1991-2020 minus 2001-2020 (render_maps.reref's
    re-basing; the GEPS page's rule that the model's own drift and bias are removed and only the period moves).
    500 hPa height has its daily global mean removed (two decades of warming would sit on every map as a ridge)."""
    days = valid_days(base)
    A = {}
    for t in ANIM_ORDER:
        a = z[f"e_{t}"].astype("float64") - C2[f"d_{t}"] - shift_series(sh, f"s15_{t}", days)
        if Dr is not None and t in Dr.tags:                             # the operational drift the reforecast lacks
            a = a - Dr.field(t)
        if FIELDS[t][3]:
            a = a - np.array([global_mean(x) for x in a])[:, None, None]
        A[t] = a
    return A


def weekly_of(a):
    return np.stack([a[w0 - 1:w1].mean(0) for w0, w1 in R.WEEKS])


# ── daily animation: one frame per forecast day, the GEPS page's render_maps.daily() geometry ──────────────────
def _frames_job(spec):
    tag, key, dom, a, fc, base_iso, outdir = spec
    plt = _plt()
    base = dt.date.fromisoformat(base_iso)
    title, unit, cmap, demean, scale = FIELDS[tag]
    rows = domain_rows(dom); lat = R.LAT[rows]
    sub = a[:, rows] * scale
    fsub = fc[:, rows] * scale if fc is not None else None
    clev, clw = CONTOUR.get(tag, (None, 0.5))
    lim = scale_for(sub, SCALE_PCT.get(tag, 95))
    asp = aspect(key, dom)
    fw, top_in, bot_in = 7.6, 0.62, 0.66
    map_h = fw * asp; fig_h = map_h + top_in + bot_in
    d = Path(outdir) / f"{key}_{tag}"
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("F*.webp"):
        old.unlink()
    frames = []
    for i in range(sub.shape[0]):
        fig = plt.figure(figsize=(fw, fig_h))
        ax = fig.add_axes([0.0, bot_in / fig_h, 1.0, map_h / fig_h], projection=projection(dom))
        mesh = _panel(ax, lat, R.LON, sub[i], dom, lim, cmap, over=(fsub[i] if fsub is not None else None),
                      clev=clev, clw=clw)
        valid = base + dt.timedelta(days=i)
        ax.set_title(f"{dom['label']} — {title} anomaly\nGEFS init {base:%b %d} · day {i + 1}, valid {valid:%a %d %b}",
                     fontsize=11, fontweight="bold", pad=6)
        cax = fig.add_axes([0.30, 0.34 / fig_h, 0.40, 0.14 / fig_h])
        cb = fig.colorbar(mesh, cax=cax, orientation="horizontal", extend="both")
        cb.ax.xaxis.set_label_position("top")
        cb.set_label(f"{title} anomaly vs 1991–2020 ({unit})" + ("   ·   contours: ensemble mean" if tag in CONTOUR else ""),
                     fontsize=9, labelpad=3)
        cb.ax.tick_params(labelsize=8, pad=1.5)
        f = d / f"F{i:02d}.webp"
        fig.savefig(f, dpi=100, facecolor="white", pil_kwargs={"quality": 86, "method": 6})
        plt.close(fig)
        frames.append({"idx": i, "file": f.name, "date": valid.isoformat(), "label": f"day {i + 1} · {valid:%b %d}"})
    return f"{key}_{tag}", title, frames


def daily_loops(A, z, base, anim: Path, procs: int):
    from concurrent.futures import ProcessPoolExecutor
    specs = [(t, k, dom, A[t].astype("float32"), (z[f"e_{t}"] if t in CONTOUR else None), base.isoformat(), str(anim))
             for k, dom in DOMAINS.items() for t in ANIM_ORDER]
    regions = {k: {} for k in DOMAINS}
    with ProcessPoolExecutor(procs) as ex:
        for rid, title, frames in ex.map(_frames_job, specs):
            regions[rid.split("_")[0]][rid] = {"label": title, "frames": frames}
    for k in DOMAINS:
        ordered = {f"{k}_{t}": regions[k][f"{k}_{t}"] for t in ANIM_ORDER if f"{k}_{t}" in regions[k]}
        (anim / f"gefs_{k}_manifest.json").write_text(json.dumps(
            {"ver": int(time.time()), "selectorLabel": "Field", "regions": ordered, "default": f"{k}_t2m"}))
    return sum(len(r["frames"]) for k in regions for r in regions[k].values())


# ── polar vortex: diagnostics (strat_series.py's four panels) and the daily loop (strat_frames.py) ─────────────
VREF = HERE / "data" / "era5_vortex_ref.json"
EPOCH_FC = 2009.5                                  # centre of the 2000-2019 reforecast
PANELS = [("u60", 10, "u(60{h}) 10 hPa", "m/s", False), ("u60", 100, "u(60{h}) 100 hPa", "m/s", False),
          ("capT", 100, "polar-cap T anomaly, 100 hPa", "K", True), ("capZd", 100, "polar-cap height anomaly, 100 hPa", "m", True)]


def _yearfrac(d: dt.date):
    return d.year + (d.timetuple().tm_yday - 1) / 365.25


def _normal(vref, name, days):
    doy = np.asarray(vref["doy"], float); f = np.asarray(vref["normal_9120"][name])
    x = np.r_[doy - 365.0, doy, doy + 365.0]; ff = np.r_[f, f, f]
    return np.interp([float(d.timetuple().tm_yday) for d in days], x, ff)


def vortex_series(z, C2, base, hemi, prev, tail, out: Path, Dr=None):
    """Four panels per hemisphere, every member drift-corrected against GEFS's own reforecast:
       winds   raw - M(lead) + ERA5 1991-2020 normal(valid)                 (the normal drawn dashed)
       caps    raw - M(lead) - b (t - 2009.5)   = the anomaly against the 1991-2020 normal carried to the valid date by
               ERA5's linear trend where significant (strat_maps.clim_at's epoch rule). Height is the cap minus the
               hemisphere poleward of 20 (strat_series.reduce_field's demeaning).
    -> (figure name, {panel: mean series})."""
    plt = _plt()
    import matplotlib.dates as mdates
    vref = json.loads(VREF.read_text())
    north = hemi == "nh"; H = "N" if north else "S"
    names = [str(x) for x in z["scalars"]]
    days = valid_days(base)
    t = [dt.datetime.combine(d, dt.time(12)) for d in days]
    fig = plt.figure(figsize=(13.4, 8.4))
    L0, R0, T0, B0 = 0.055, 0.988, 0.885, 0.070
    PW = (R0 - L0) / 2; PH = (T0 - B0) / 2
    means = {}
    for pi, (kind, lev, title, unit, as_anom) in enumerate(PANELS):
        name = f"{kind}{lev}{H}"; k = names.index(name)
        raw = z["s"][:, k, :].astype("float64")                           # members x days
        M = np.asarray(C2["s"][C2["scalars"].index(name)]) + scalar_drift(Dr, kind, lev, north)
        normal = _normal(vref, name, days)
        if as_anom:
            b = vref["trend_per_yr"][name]
            corr = raw - M[None] - b * (np.array([_yearfrac(d) for d in days]) - EPOCH_FC)[None]
            rawm = raw.mean(0) - (normal + b * (np.array([_yearfrac(d) for d in days]) - 2005.5))
        else:
            corr = raw - M[None] + normal[None]
            rawm = raw.mean(0)
        means[name] = corr.mean(0)
        col, row = pi % 2, pi // 2
        ax = fig.add_axes([L0 + col * PW + 0.042, T0 - (row + 1) * PH + 0.070, PW * 0.895, PH * 0.735])
        ax.fill_between(t, np.percentile(corr, 10, 0), np.percentile(corr, 90, 0), color="#2c4a72", alpha=0.16, lw=0, label="p10–p90")
        ax.fill_between(t, np.percentile(corr, 25, 0), np.percentile(corr, 75, 0), color="#2c4a72", alpha=0.28, lw=0, label="p25–p75")
        ax.plot(t, corr.mean(0), "#16335c", lw=2.2, label="ensemble mean, bias-corrected")
        ax.plot(t, rawm, color="#16335c", lw=1.1, ls=(0, (1, 1.6)), alpha=0.75, label="raw GEFS mean (model drift left in)")
        tv = tail.get(name) if tail else None
        if tv:
            tt = [dt.datetime.fromisoformat(x) for x in tv["t"]]
            vv = np.array(tv["v"], float)
            if as_anom:
                vref_t = _normal(vref, name, [x.date() for x in tt]) + vref["trend_per_yr"][name] * (np.array([_yearfrac(x.date()) for x in tt]) - 2005.5)
                vv = vv - vref_t
            ax.plot(tt + t[:1], list(vv) + [corr.mean(0)[0]], color="#3f3a33", lw=1.6, label=f"GEFS analysis, last {len(tt)} d")
            ax.plot(tt, vv, "o", color="#3f3a33", ms=3.0)
        if prev and name in prev.get("means", {}):
            pt = [dt.datetime.fromisoformat(x) for x in prev["t"]]
            pm = prev["means"][name]
            keep = [i for i, x in enumerate(pt) if t[0] <= x <= t[-1]]
            if keep:
                ax.plot([pt[i] for i in keep], [pm[i] for i in keep], ls=(0, (4, 2)), color="#c8781e", lw=1.7,
                        label=f"previous run mean, init {dt.date.fromisoformat(prev['date']):%b %d}")
        if as_anom:
            ax.axhline(0, color="#8a8680", lw=0.9)
        else:
            ax.plot(t, normal, ls="--", color="#b4453c", lw=1.5, label="1991–2020 normal (ERA5)")
            ax.axhline(0, color="k", lw=1.6)
        if kind == "u60" and lev == 10:
            frac = float((corr.min(1) < 0).mean())
            lo, hi = ax.get_ylim()
            ax.set_ylim(min(lo, -0.12 * (hi - min(lo, 0.0))), hi)          # room under the zero line for the note
            ax.text(0.99, 0.02, f"{frac:.0%} of members reverse to easterlies within 35 days", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=8.5, color="#a33", style="italic")
        ax.axvline(dt.datetime.combine(base, dt.time(0)), color="#8a8680", lw=0.8, ls=":")
        ax.set_ylabel(unit, fontsize=9.5)
        ax.set_title(title.format(h=H), fontsize=11, fontweight="bold", loc="left", pad=4)
        ax.tick_params(labelsize=8.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d")); ax.xaxis.set_major_locator(mdates.DayLocator(interval=10))
        ax.grid(alpha=0.25, lw=0.5)
        if pi == 0:
            ax.legend(fontsize=7.2, ncol=2, loc="upper left", framealpha=0.9)
    fig.text(0.5, 0.012, "Bias-corrected for the model's drift with lead time: each member minus the GEFSv12 reforecast mean "
             "(2000–2019, same start date and lead) and minus the drift today's GEFS shows beyond it in recently verified runs, "
             "plus the ERA5 1991–2020 normal on the valid date; the anomaly panels take that normal to the valid year along "
             "ERA5's linear trend where it is significant. Dotted: the raw model.",
             ha="center", fontsize=8.3, color="#5c6b73", wrap=True)
    fig.suptitle(f"{'Northern' if north else 'Southern'} polar vortex\nGEFS extended ensemble · init {base:%Y-%m-%d} 00Z",
                 fontsize=13, fontweight="bold", y=0.985, va="top")
    p = out / f"gefs_strat_{hemi}_series.webp"
    fig.savefig(p, dpi=110, facecolor="white", pil_kwargs={"quality": 90, "method": 6}); plt.close(fig)
    return p.name, means


def scalar_drift(Dr, kind, lev, north):
    """The zonal drift correction reduced to a vortex scalar (the same definitions as gefs_daily.scalar), (35,)."""
    if Dr is None:
        return np.zeros(35)
    lat = R.LAT; w = np.cos(np.deg2rad(lat))
    if kind == "u60":
        return Dr.on(f"u{lev}", [60.0 if north else -60.0])[:, 0]
    d = Dr.zonal(f"t{lev}" if kind == "capT" else f"zg{lev}")
    cap = (lat >= 65) if north else (lat <= -65)
    v = (d[:, cap] * w[cap]).sum(1) / w[cap].sum()
    if kind == "capZd":
        hm = (lat >= 20) if north else (lat <= -20)
        v = v - (d[:, hm] * w[hm]).sum(1) / w[hm].sum()
    return v


def analysis_tail(base: dt.date, cache: Path, n=14):
    """The GEFS control's 00Z analyses (f000) of the last n days as vortex scalars - the tail the forecast joins.
    Cached per day; a day the bucket does not have is skipped."""
    names = [s_[0] for s_ in G.SCALARS]
    tail = {nm: {"t": [], "v": []} for nm in names}
    for k in range(n, 0, -1):
        d = base - dt.timedelta(days=k); ds = d.strftime("%Y%m%d")
        cp = cache / "tail" / f"an_{ds}.json"
        if cp.exists():
            vals = json.loads(cp.read_text())
        else:
            url = f"{S3}/gefs.{ds}/00/atmos/pgrb2ap5/gec00.t00z.pgrb2a.0p50.f000"
            try:
                idx = R.index(url)
                vals = {}
                for nm, src, kind, hemi in G.SCALARS:
                    name, lev = G.D2[src][4], G.D2[src][2]
                    hit = [e for e in idx if e[4] == name and e[2] == lev and e[3] == "anl"]
                    if not hit:
                        raise KeyError(src)
                    vals[nm] = G.scalar(G.to_half(R.decode(get(url, (hit[0][0], hit[0][1])))), kind, hemi)
            except Exception as e:                                   # noqa: BLE001
                print(f"  tail {ds}: {str(e)[:60]}", flush=True)
                continue
            cp.parent.mkdir(parents=True, exist_ok=True); cp.write_text(json.dumps(vals))
        for nm in names:
            tail[nm]["t"].append(dt.datetime.combine(d, dt.time(0)).isoformat()); tail[nm]["v"].append(vals[nm])
    return tail


MS2KT = 1.94384
WLEV = [25, 35, 45, 55, 65, 80, 95, 110, 130, 150, 170, 190]          # strat_maps.WLEV (knots)
WCOLS = ["#dbeef8", "#a9d4ec", "#6fb4de", "#3f8fc7", "#61b26b", "#b5d24a", "#f2d03a", "#f29b2c", "#e0562c", "#b0289b", "#7b1d78"]


def _nice(lim, n=24):
    from matplotlib.ticker import MaxNLocator
    lv = MaxNLocator(nbins=n, symmetric=True).tick_values(-lim, lim)
    return lv[lv != 0] if len(lv) > 40 else lv


def _vloop_job(spec):
    """One region (hemisphere x level) of the vortex loop: temperature anomaly | total wind + heights | height anomaly."""
    rid, north, lev, T, Z, Zfull, U, V, base_iso, outdir = spec
    plt = _plt()
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.util import add_cyclic_point
    from matplotlib.colors import BoundaryNorm, ListedColormap
    base = dt.date.fromisoformat(base_iso)
    lat, lon = R.LAT, R.LON
    lat0, lat1 = (20, 90) if north else (-90, -20)
    sub = (lat >= lat0) & (lat <= lat1)
    phi = 60 if north else -60
    capm = (lat >= 65) if north else (lat <= -65); wc = np.cos(np.deg2rad(lat[capm]))
    spd = np.hypot(U, V) * MS2KT
    limT = float(np.nanpercentile(np.abs(T[:, sub]), 99)); limZ = float(np.nanpercentile(np.abs(Z[:, sub]), 99))
    cm = ListedColormap(WCOLS); cm.set_under("#ffffff"); cm.set_over("#4a1050"); nm = BoundaryNorm(WLEV, cm.N)
    proj = ccrs.NorthPolarStereo() if north else ccrs.SouthPolarStereo()
    d = Path(outdir) / rid
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("F*.webp"):
        old.unlink()
    frames = []
    for i in range(T.shape[0]):
        fig = plt.figure(figsize=(3.9 * 3 + 0.2, 5.55))
        span = 1.0 / 3
        for col, (key, lab, unit) in enumerate((("T", "Temperature", "K"), ("W", "Total wind", "kt"), ("Z", "Geopotential height", "m"))):
            ax = fig.add_axes([0.014 + col * span, 0.145, span * 0.945, 0.755], projection=proj)
            ax.set_extent([-180, 180, lat0, lat1], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE.with_scale("110m"), lw=0.4, edgecolor="#555")
            ax.gridlines(lw=0.3, color="#bbb", ylocs=[phi], draw_labels=False)
            gz_, lo_ = add_cyclic_point(Zfull[i], coord=lon)
            if key == "W":
                f_, _ = add_cyclic_point(spd[i], coord=lon); u_, _ = add_cyclic_point(U[i], coord=lon)
                msh = ax.contourf(lo_, lat, f_, levels=WLEV, cmap=cm, norm=nm, extend="both", transform=ccrs.PlateCarree())
                ax.contour(lo_, lat, gz_, levels=10, colors="#1a1a1a", linewidths=0.55, transform=ccrs.PlateCarree())
                ax.contourf(lo_, lat, u_, levels=[-1e9, 0.0], colors="none", hatches=["//"], transform=ccrs.PlateCarree())
                ax.contour(lo_, lat, u_, levels=[0], colors="#c2410c", linewidths=1.1, transform=ccrs.PlateCarree())
                note = f"u({phi if north else -phi}{'N' if north else 'S'}) {float(np.interp(phi, lat[::-1], U[i].mean(1)[::-1])) * MS2KT:+.0f} kt"
                ticks = WLEV[::2]
            else:
                A = T[i] if key == "T" else Z[i]
                f_, _ = add_cyclic_point(A, coord=lon)
                lv = _nice(limT if key == "T" else limZ)
                msh = ax.contourf(lo_, lat, f_, levels=lv, cmap="RdBu_r", extend="both", transform=ccrs.PlateCarree())
                if key == "Z":
                    ax.contour(lo_, lat, gz_, levels=8, colors="#333", linewidths=0.45, transform=ccrs.PlateCarree())
                cav = float((A[capm].mean(1) * wc).sum() / wc.sum())
                note = f"cap {cav:+.1f}" + (" m" if key == "Z" else " K")
                ticks = msh.levels[::4]
            ax.set_title(lab + ("" if key == "W" else " anomaly"), fontsize=11, fontweight="bold", pad=4)
            ax.text(0.5, -0.055, note, transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color="#2c4a72", fontweight="bold")
            cax = fig.add_axes([0.014 + col * span + span * 0.16, 0.072, span * 0.63, 0.020])
            cb = fig.colorbar(msh, cax=cax, orientation="horizontal", extend="both", ticks=ticks)
            cb.set_label(unit, fontsize=8, labelpad=1); cb.ax.tick_params(labelsize=7, pad=1)
        valid = base + dt.timedelta(days=i)
        fig.suptitle(f"{'Northern' if north else 'Southern'} polar vortex — {lev} hPa · GEFS day {i + 1}, valid {valid:%a %d %b}",
                     fontsize=12.5, fontweight="bold", y=0.988, va="top")
        fp = d / f"F{i:02d}.webp"
        fig.savefig(fp, dpi=100, facecolor="white", pil_kwargs={"quality": 86, "method": 6})
        plt.close(fig)
        frames.append({"idx": i, "file": fp.name, "date": valid.isoformat(), "label": f"day {i + 1} · {valid:%b %d}"})
    return rid, frames


def demean_hemi(a, north):
    """Remove the hemispheric (poleward of 20) mean from height anomalies - strat_maps.demean_height: the QBO and the
    tropospheric warming move the whole surface, and that is not the vortex."""
    hm = (R.LAT >= 20) if north else (R.LAT <= -20)
    w = np.cos(np.deg2rad(R.LAT[hm]))
    m = (a[:, hm].mean(2) * w).sum(1) / w.sum()
    return a - m[:, None, None]


def vortex_loop(z, C2, trend, base, anim: Path, procs: int, Dr=None):
    days = valid_days(base)
    dtv = (np.array([_yearfrac(d) for d in days]) - EPOCH_FC)[:, None, None]
    specs = []
    for north in (True, False):
        for lev in (10, 100):
            T = z[f"e_t{lev}"] - C2[f"d_t{lev}"] - trend[f"b_T{lev}"][None] * dtv
            Z = z[f"e_zg{lev}"] - C2[f"d_zg{lev}"] - trend[f"b_Z{lev}"][None] * dtv
            if Dr is not None:
                T = T - Dr.field(f"t{lev}"); Z = Z - Dr.field(f"zg{lev}")
            Z = demean_hemi(Z, north)
            specs.append((f"{'nh' if north else 'sh'}{lev}", north, lev, T.astype("float32"), Z.astype("float32"),
                          z[f"e_zg{lev}"], z[f"e_u{lev}"], z[f"e_v{lev}"], base.isoformat(), str(anim)))
    from concurrent.futures import ProcessPoolExecutor
    regions = {}
    with ProcessPoolExecutor(procs) as ex:
        for rid, frames in ex.map(_vloop_job, specs):
            regions[rid] = {"label": f"{rid[:2].upper()} {rid[2:]} hPa", "frames": frames}
    order = ["nh10", "nh100", "sh10", "sh100"]
    (anim / "gefs_strat_manifest.json").write_text(json.dumps(
        {"ver": int(time.time()), "selectorLabel": "Level", "regions": {r: regions[r] for r in order if r in regions}, "default": "nh10"}))
    return sum(len(r["frames"]) for r in regions.values())


# ── run archive: yesterday's anomalies for the change maps and the previous-run overlays ────────────────────────
def load_prev(runs: Path, date: str, max_gap=7):
    """Newest earlier run within max_gap days (the archive rides on the frames branch, seeded before the run)."""
    base = dt.date(int(date[:4]), int(date[4:6]), int(date[6:]))
    for g in range(1, max_gap + 1):
        d = (base - dt.timedelta(days=g)).strftime("%Y%m%d")
        p = runs / f"gefs_run_{d}.npz"
        if p.exists():
            return g, np.load(p, allow_pickle=False)
    return None, None


def render(npz: Path, date: str, site: Path, cache: Path, procs: int = 4, runs: Path | None = None, ref_dir: Path | None = None):
    """Every GEFS page figure this script owns: weekly maps (+ skill-masked), change vs the previous run, the daily
    animation, the vortex diagnostics and the vortex loop; and the run archive the next run compares with."""
    base = dt.date(int(date[:4]), int(date[4:6]), int(date[6:]))
    out = site / "assets" / "gefs"; out.mkdir(parents=True, exist_ok=True)
    data = out / "data"; data.mkdir(exist_ok=True)
    anim = out / "anim"; anim.mkdir(exist_ok=True)
    runs = runs or (anim / "runs"); runs.mkdir(parents=True, exist_ok=True)
    z = np.load(npz)
    C2 = clim2_at(base, cache)
    if C2 is None:
        raise SystemExit("the daily reforecast climatology (release gefs-clim-v2) is required")
    sh = np.load(ref("era5_shift.npz", cache, ref_dir))
    import gefs_drift as GD
    try:                                  # verifies whatever inits of the window are missing (the first run backfills)
        Dr = GD.estimate(base, cache, runs)
    except (Exception, SystemExit) as e:                             # noqa: BLE001
        print(f"  drift estimate failed ({str(e)[:100]}); using the newest archived one", flush=True)
        Dr = GD.latest(runs, base)
    A = anomalies(z, C2, sh, base, Dr)
    days = valid_days(base)
    titles = [f"Week {i + 1}   {days[a - 1]:%b %d} – {days[b - 1]:%b %d}" for i, (a, b) in enumerate(R.WEEKS)]
    gap, prev = load_prev(runs, date)
    names = []
    try:
        import gefs_skill as SK                                       # the terciles port: skill masks + hatching
    except Exception:                                                # noqa: BLE001
        SK = None
    for tag in ANIM_ORDER:
        wk = weekly_of(A[tag]); fc = weekly_of(z[f"e_{tag}"].astype("float64"))
        note = "global mean removed per day" if FIELDS[tag][3] else ""
        for key, dom in DOMAINS.items():
            names.append(map_figure(tag, wk, fc, base, key, dom, out, "weekly", titles, note))
        if SK is not None and tag in MASKABLE:
            try:
                rdir = ref_dir if (ref_dir and (ref_dir / f"gefs_mapskill_{tag}.npz").exists()) else \
                    ref(f"gefs_mapskill_{tag}.npz", cache).parent
                mk = SK.masks(tag, base, rdir)
                if mk is None:
                    raise FileNotFoundError(f"gefs_mapskill_{tag}.npz")
                import cartopy.crs as ccrs
                hat = lambda ax, i, m=mk: SK.hatch(ax, R.LAT, R.LON, m[0][i], m[1][i], ccrs.PlateCarree())
                mnote = (note + "   ·   " if note else "") + SK.MASK_NOTE[tag]
                for key, dom in DOMAINS.items():
                    names.append(map_figure(tag, wk, fc, base, key, dom, out, "weekly", titles, mnote, hatcher=hat, suffix="_masked"))
            except Exception as e:                                   # noqa: BLE001
                print(f"  {tag}: no skill mask ({str(e)[:80]})", flush=True)
        if prev is not None and f"A_{tag}" in prev.files:
            pa = prev[f"A_{tag}"].astype("float64")
            diffs, ttl = [], []
            for i, (w0, w1) in enumerate(R.WEEKS):
                com = [d for d in range(w0, w1 + 1) if d + gap <= 35]
                diffs.append(np.mean([A[tag][d - 1] - pa[d + gap - 1] for d in com], axis=0))
                ttl.append(titles[i] + ("" if len(com) == 7 else f" ({len(com)} common days)"))
            pdate = base - dt.timedelta(days=gap)
            for key, dom in DOMAINS.items():
                names.append(map_figure(tag, diffs, None, base, key, dom, out, "change", ttl,
                                        f"this run minus the run of {pdate:%Y-%m-%d}, weekly means over the valid days both "
                                        "forecast; red = this run more positive",
                                        f"{FIELDS[tag][0]}: this run minus the previous run ({FIELDS[tag][1]})"))
    n_daily = daily_loops(A, z, base, anim, procs)
    trend = np.load(ref("strat_trend15.npz", cache, ref_dir))
    n_vortex = vortex_loop(z, C2, trend, base, anim, procs, Dr)
    tail = analysis_tail(base, cache)
    pv = None
    if prev is not None and "vortex_means" in prev.files:
        pv = json.loads(str(prev["vortex_means"]))
    vmeans = {}
    for hemi in ("nh", "sh"):
        nm, m = vortex_series(z, C2, base, hemi, pv, tail, out, Dr)
        names.append(nm); vmeans.update({k: v.tolist() for k, v in m.items()})
    # the archive: this run's daily map anomalies (float16) and vortex means, for tomorrow's change maps and overlays
    vjson = json.dumps({"date": base.isoformat(), "t": [dt.datetime.combine(d, dt.time(12)).isoformat() for d in days],
                        "means": vmeans})
    np.savez_compressed(runs / f"gefs_run_{date}.npz", date=np.array(date), vortex_means=np.array(vjson),
                        **{f"A_{t}": A[t].astype("float16") for t in ANIM_ORDER})
    for q in runs.glob("gefs_run_*.npz"):                             # keep a week
        if q.stem.split("_")[-1] < (base - dt.timedelta(days=7)).strftime("%Y%m%d"):
            q.unlink()
    for q in data.glob("gefs_anom_*.npz"):                            # the v1 archive, superseded
        q.unlink()
    GD.archive_forecast(z, date, runs)                                # verified against analyses 36 days from now
    GD.prune(runs, base)
    if Dr is not None:
        names.append(GD.figure(Dr, base, out))
    meta = {"cycle": date, "init": f"{base:%Y-%m-%d} 00Z",
            "valid": f"{days[0]:%b %d} – {days[-1]:%b %d}",
            "members": int(len(z["members"])), "change_vs": (base - dt.timedelta(days=gap)).strftime("%Y%m%d") if gap else None,
            "generated": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%MZ"), "figures": names,
            "frames": {"maps": n_daily, "vortex": n_vortex},
            "drift": {"inits": len(Dr.inits), "first": Dr.inits[-1], "last": Dr.inits[0]} if Dr is not None else None}
    (data / "gefs.json").write_text(json.dumps(meta, indent=1))
    print(f"{len(names)} figures, {n_daily} map frames, {n_vortex} vortex frames; change maps vs "
          f"{meta['change_vs'] or 'none (no earlier run in the archive)'}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date"); ap.add_argument("--members", type=int, default=31); ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--site", default=str(HERE.parents[1])); ap.add_argument("--work", default="/tmp/gefs_work")
    ap.add_argument("--runs", help="run archive dir (default SITE/assets/gefs/anim/runs, seeded from the frames branch)")
    ap.add_argument("--ref", help="local directory holding the gefs-ref-v1 files (default: download the release)")
    ap.add_argument("--force", action="store_true", help="render even if this cycle is already done")
    a = ap.parse_args()
    date = latest_cycle(a.date)
    if not date:
        print("no 00Z GEFS cycle at 840 h in the last week"); return 1
    done = Path(a.site) / "assets" / "gefs" / "data" / "gefs.json"
    if not (a.date or a.force) and done.exists() and json.loads(done.read_text()).get("cycle") == date:
        print(f"{date} already rendered"); return 0
    print(f"GEFS extended {date} 00Z", flush=True)
    work = Path(a.work)
    npz = fetch(date, a.members, work, a.procs)
    render(npz, date, Path(a.site), work / "clim", a.procs, Path(a.runs) if a.runs else None, Path(a.ref) if a.ref else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
