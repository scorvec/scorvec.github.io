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
import gefs_reforecast as R                                           # noqa: E402  (decode, coarsen, grid, packing)

S3 = "https://noaa-gefs-pds.s3.amazonaws.com"
CLIM_URL = "https://github.com/scorvec/scorvec.github.io/releases/download/gefs-clim-v1/gefs_clim_{c:03d}.npz"
LIVE = {"t2m": ("TMP", "2 m above ground"), "zg500": ("HGT", "500 mb"), "mslp": ("PRMSL", "mean sea level"),
        "u200": ("UGRD", "200 mb"), "u850": ("UGRD", "850 mb"), "u10": ("UGRD", "10 mb"),
        "pr": ("APCP", "surface"), "olr": ("ULWRF", "top of atmosphere")}
KIND = {t: R.FIELDS[t][3] for t in LIVE}


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
    """Daily fields of one member at 1.5 deg -> (weekly packed dict, u60 daily, band daily), or None."""
    jobs = []
    for h in range(6, 841, 6):
        url = step_url(date, mem, h)
        try:
            idx = R.index(url)
        except Exception:                                             # noqa: BLE001
            return None
        want = []
        if h % 12 == 0:
            want += [(t, "inst", f"{h} hour fcst") for t in LIVE if KIND[t] == "inst"]
        want += [(t, KIND[t], f"{h - 6}-{h} hour {KIND[t]} fcst") for t in ("pr", "olr")]
        for t, kind, stext in want:
            name, lev = LIVE[t]
            hit = [e for e in idx if e[4] == name and e[2] == lev and e[3] == stext]
            if not hit:
                return None
            jobs.append((t, (h + 23) // 24, url, hit[0][0], hit[0][1]))
    with ThreadPoolExecutor(24) as ex:
        bufs = list(ex.map(lambda j: get(j[2], (j[3], j[4])), jobs))
    daily = {t: np.zeros((35, len(R.LAT), len(R.LON))) for t in LIVE if t != "u10"}
    cnt = {t: np.zeros(35) for t in daily}
    u60 = np.zeros(35); u60n = np.zeros(35)
    for (t, d, *_), b in zip(jobs, bufs):
        v = R.decode(b)
        if t == "u10":
            u60[d - 1] += R.zonal60(v); u60n[d - 1] += 1
            continue
        daily[t][d - 1] += R.coarsen(v); cnt[t][d - 1] += 1
    for t in daily:
        if KIND[t] != "acc":
            daily[t] /= np.maximum(cnt[t], 1)[:, None, None]
    weekly = {t: np.stack([daily[t][a - 1:b].mean(0) for a, b in R.WEEKS]) for t in R.WEEKLY}
    band = np.stack([daily[t][:, R.BAND, :].mean(1) for t in R.BANDF])
    return weekly, u60 / np.maximum(u60n, 1), band


def fetch(date: str, n_members: int, work: Path, procs: int = 1):
    dest = work / f"gefs_{date}.npz"
    if dest.exists():
        return dest
    mems = (["gec00"] + [f"gep{i:02d}" for i in range(1, 31)])[:n_members]
    got = {}
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    with ProcessPoolExecutor(procs) as ex:                            # decoding is the CPU cost: one member per core
        for m, r in zip(mems, ex.map(member, [date] * len(mems), mems)):
            print(f"  {m}: {'ok' if r else 'incomplete, skipped'} ({time.time() - t0:.0f} s)", flush=True)
            if r:
                got[m] = r
    if len(got) < max(1, (n_members + 1) // 2):             # at least half the members asked for
        raise SystemExit(f"only {len(got)} complete members")
    ms = list(got)
    work.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, members=np.array(ms),
                        **{f"w_{t}": np.stack([got[m][0][t] for m in ms]).astype("float32") for t in R.WEEKLY},
                        u60=np.stack([got[m][1] for m in ms]).astype("float32"),
                        band=np.stack([got[m][2] for m in ms]).astype("float32"))
    return dest


def clim_at(init: dt.date, cache: Path):
    """The reforecast climatology at the init day of year, linear between the two 5-day centres around it."""
    doy = init.timetuple().tm_yday
    centres = list(range(1, 366, 5))
    lo = max(c for c in centres if c <= doy) if doy >= 1 else centres[-1]
    hi = lo + 5 if lo + 5 <= 361 else 1
    w = (doy - lo) / 5.0
    out = {}
    for c, wt in ((lo, 1 - w), (hi, w)):
        p = cache / f"gefs_clim_{c:03d}.npz"
        if not p.exists():
            cache.mkdir(parents=True, exist_ok=True)
            p.write_bytes(get(CLIM_URL.format(c=c)))
        z = np.load(p)
        for k in [f"w_{t}" for t in R.WEEKLY] + ["u60", "band"]:
            out[k] = out.get(k, 0) + wt * z[k].astype("float64")
    return out


# ── figures (same geometry, projections, colour maps and wording as the GEPS page's render_maps.py) ──────────
FIELDS = {"t2m": ("2 m temperature", "°C", "RdBu_r", False, 1.0),
          "pr": ("Precipitation", "mm/day", "BrBG", False, 1.0),
          "zg500": ("500 hPa height", "m", "RdBu_r", True, 1.0),
          "olr": ("Outgoing longwave", "W m⁻²", "BrBG_r", False, 1.0),
          "u850": ("850 hPa zonal wind", "m/s", "PuOr_r", False, 1.0),
          "u200": ("200 hPa zonal wind", "m/s", "PuOr_r", False, 1.0),
          "mslp": ("Mean sea-level pressure", "hPa", "RdBu_r", False, 0.01)}
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


def map_figure(tag, weeks_anom, weeks_fc, base, key, dom, out: Path, kind="weekly", titles=None, note="", cb_label=None):
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
        ax.set_title(titles[i] if stack else titles[i].replace("   ", "\n", 1), fontsize=10.5, fontweight="bold", pad=4,
                     loc="left" if stack else "center")
    cax = fig.add_axes(cax_rect)
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal", extend="both")
    cb.ax.xaxis.set_label_position("top")
    cb.set_label(cb_label or (f"{title} anomaly vs GEFS 2000–2019 model climate ({unit})"
                              + ("   ·   contours: ensemble mean" if tag in CONTOUR else "")), fontsize=10, labelpad=4)
    cb.ax.tick_params(labelsize=8.5, pad=1.5)
    head = "weekly means" if kind == "weekly" else "change vs previous run"
    fig.suptitle(f"{dom['label']} — {title}, {head} · GEFS init {base:%Y-%m-%d}", fontsize=15, fontweight="bold", y=title_y, va="top")
    if note:
        fig.text(0.5, note_y, note, ha="center", va="bottom", fontsize=8.5, color=MUTED)
    p = out / f"gefs_{key}_{tag}_{kind}.webp"
    fig.savefig(p, dpi=105, facecolor="white", pil_kwargs={"quality": 88, "method": 6})
    plt.close(fig)
    return p.name


def global_mean(w):
    wt = np.cos(np.deg2rad(R.LAT))[:, None] * np.ones((1, len(R.LON)))
    return float((w * wt).sum() / wt.sum())


def vortex_figure(u_raw, u_clim, obs_clim, base, prev, out: Path):
    """60N 10 hPa zonal wind: every member bias-corrected, the raw mean faint, the MERRA-2 2000-2019 normal."""
    plt = _plt()
    import matplotlib.dates as mdates
    days = [base + dt.timedelta(days=d) for d in range(1, 36)]
    valid_doy = [min(d.timetuple().tm_yday, 366) for d in days]
    obs = np.array([obs_clim[x - 1] for x in valid_doy])
    corr = u_raw - u_clim[None] + obs[None]                                 # members x days
    fig, ax = plt.subplots(figsize=(10.5, 5.2)); fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.17)
    ax.fill_between(days, np.percentile(corr, 10, 0), np.percentile(corr, 90, 0), color="#2c4a72", alpha=0.16, lw=0, label="p10–p90")
    ax.fill_between(days, np.percentile(corr, 25, 0), np.percentile(corr, 75, 0), color="#2c4a72", alpha=0.28, lw=0, label="p25–p75")
    ax.plot(days, corr.mean(0), color="#16335c", lw=2.2, label="ensemble mean, bias-corrected")
    ax.plot(days, u_raw.mean(0), color="#16335c", lw=1.1, ls=(0, (1, 1.6)), alpha=0.75, label="raw GEFS mean (model drift left in)")
    ax.plot(days, obs, ls="--", color="#b4453c", lw=1.5, label="2000–2019 normal (MERRA-2)")
    if prev is not None:
        pt = [dt.date.fromisoformat(t) for t in prev["t"]]
        keep = [i for i, t in enumerate(pt) if days[0] <= t <= days[-1]]
        if keep:
            ax.plot([pt[i] for i in keep], [prev["mean"][i] for i in keep], ls=(0, (4, 2)), color="#c8781e", lw=1.7,
                    label=f"previous run mean, init {prev['date']}")
    ax.axhline(0, color="k", lw=1.6)
    frac = float((corr.min(1) < 0).mean())
    ax.set_title(f"u(60N) 10 hPa   ·   {frac:.0%} of members reverse to easterlies within 35 days", loc="left", fontsize=11, fontweight="bold")
    ax.set_ylabel("m/s"); ax.grid(alpha=0.25, lw=0.5); ax.legend(fontsize=7.6, ncol=2, loc="upper left", framealpha=0.9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d")); ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    fig.suptitle(f"Northern polar vortex · GEFS extended ensemble · init {base:%Y-%m-%d} 00Z", fontsize=13, fontweight="bold", y=0.975)
    fig.text(0.5, 0.02, "Bias-corrected for the model's drift with lead time: raw member minus the GEFSv12 reforecast mean at "
             "the same lead and start date (2000–2019), plus MERRA-2 on the valid date over the same years.",
             ha="center", fontsize=8.3, color=MUTED, wrap=True)
    p = out / "gefs_strat_vortex.webp"
    fig.savefig(p, dpi=110, facecolor="white", pil_kwargs={"quality": 90, "method": 6}); plt.close(fig)
    return p.name, corr.mean(0), days


def _load_run(data: Path, d: str):
    p = data / f"gefs_anom_{d}.npz"
    return np.load(p) if p.exists() else None


def render(npz: Path, date: str, site: Path, cache: Path):
    """Weekly and change maps, the vortex panel, and the run archive the next run compares with.

    The change maps compare with the run exactly SEVEN days earlier, not the previous one: only weekly climatology
    exists, and a 3- or 4-day gap would leave every week half-overlapping. Seven days lines the weeks up exactly -
    this run's week k covers the same valid dates as that run's week k+1 - so weeks 1-4 are compared and week 5 has
    no counterpart. The vortex overlay uses the most recent earlier run of any age, matched on valid dates."""
    base = dt.date(int(date[:4]), int(date[4:6]), int(date[6:]))
    out = site / "assets" / "gefs"; out.mkdir(parents=True, exist_ok=True)
    data = out / "data"; data.mkdir(exist_ok=True)
    z = np.load(npz); C = clim_at(base, cache)
    runs = sorted(q.stem.split("_")[-1] for q in data.glob("gefs_anom_*.npz"))
    older = [r for r in runs if r < date]
    wk = (base - dt.timedelta(days=7)).strftime("%Y%m%d")
    prev_any = _load_run(data, older[-1]) if older else None
    prev_wk = _load_run(data, wk)
    titles = [f"Week {i + 1}   {base + dt.timedelta(days=a):%b %d} – {base + dt.timedelta(days=b):%b %d}" for i, (a, b) in enumerate(R.WEEKS)]
    names, keep = [], {"date": np.array(date)}
    for tag in R.WEEKLY:
        fc = z[f"w_{tag}"].mean(0)                                    # ensemble-mean weekly field
        an = fc - C[f"w_{tag}"]
        note = ""
        if FIELDS[tag][3]:
            an = np.stack([w - global_mean(w) for w in an]); note = "global mean removed per week"
        keep[f"a_{tag}"] = an.astype("float16")
        for key, dom in DOMAINS.items():
            names.append(map_figure(tag, an, fc, base, key, dom, out, "weekly", titles, note))
        if prev_wk is not None and f"a_{tag}" in prev_wk.files:
            pa = prev_wk[f"a_{tag}"].astype("float64")
            diffs = [an[k] - pa[k + 1] for k in range(4)]
            for key, dom in DOMAINS.items():
                names.append(map_figure(tag, diffs, None, base, key, dom, out, "change", titles[:4],
                                        f"the run of {base - dt.timedelta(days=7):%Y-%m-%d} covers these same weeks as its weeks 2–5; "
                                        "red = this run more positive",
                                        f"{FIELDS[tag][0]}: this run minus the run a week earlier ({FIELDS[tag][1]})"))
    obs = json.loads((HERE / "data" / "merra2_u60_10_clim.json").read_text())["doy_1_to_366"]
    pv = None
    if prev_any is not None and "vortex_mean" in prev_any.files:
        d0 = str(prev_any["date"])
        pv = {"date": f"{d0[4:6]}-{d0[6:]}", "t": [str(x) for x in prev_any["vortex_t"]], "mean": prev_any["vortex_mean"].tolist()}
    vname, vmean, vdays = vortex_figure(z["u60"].astype("float64"), C["u60"], obs, base, pv, out)
    names.append(vname)
    keep["vortex_mean"] = vmean.astype("float32"); keep["vortex_t"] = np.array([d.isoformat() for d in vdays])
    np.savez_compressed(data / f"gefs_anom_{date}.npz", **keep)
    for r in runs:                                                    # keep a fortnight of runs
        if r < (base - dt.timedelta(days=15)).strftime("%Y%m%d"):
            (data / f"gefs_anom_{r}.npz").unlink(missing_ok=True)
    meta = {"cycle": date, "init": f"{base:%Y-%m-%d} 00Z",
            "valid": f"{base + dt.timedelta(days=1):%b %d} – {base + dt.timedelta(days=35):%b %d}",
            "members": int(len(z["members"])), "change_vs": wk if prev_wk is not None else None,
            "generated": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%MZ"), "figures": names}
    (data / "gefs.json").write_text(json.dumps(meta, indent=1))
    print(f"{len(names)} figures; change maps vs {wk if prev_wk is not None else 'none (no run 7 days earlier yet)'}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date"); ap.add_argument("--members", type=int, default=31); ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--site", default=str(HERE.parents[1])); ap.add_argument("--work", default="/tmp/gefs_work")
    a = ap.parse_args()
    date = latest_cycle(a.date)
    if not date:
        print("no 00Z GEFS cycle at 840 h in the last week"); return 1
    done = Path(a.site) / "assets" / "gefs" / "data" / "gefs.json"
    if not a.date and done.exists() and json.loads(done.read_text()).get("cycle") == date:
        print(f"{date} already rendered"); return 0
    print(f"GEFS extended {date} 00Z", flush=True)
    work = Path(a.work)
    npz = fetch(date, a.members, work, a.procs)
    render(npz, date, Path(a.site), work / "clim")
    return 0


if __name__ == "__main__":
    sys.exit(main())
