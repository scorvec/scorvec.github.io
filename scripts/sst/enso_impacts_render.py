#!/usr/bin/env python3
"""ENSO impacts on the Americas for the El Nino monitor (enso.html, "Impacts on the Americas"; 2026-09-27, user: "can
the enso regression maps be added to the site (for north and south america variables)?").

Draws the committed reference scripts/sst/reference/enso_impacts_site.{npz,json} (written on the laptop by
cmip6_enso_site.py from the CMIP6 study cmip6_enso_impacts.py) into one map per view, so the stage viewer switches
variable / season / class instead of shrinking a poster. Runs in .github/workflows/enso-impacts.yml on a push of the
reference or this file - never on a schedule.

Significance rule (user, 2026-09-25): a map is shaded ONLY where its test passes; everything else is left blank and the
test is printed on the figure. CMIP6: across-model one-sample t-test, Benjamini-Hochberg FDR 10 % over the map, AND
>= 80 % of the models agreeing on the sign. Observed composites: Welch t-test against neutral seasons, FDR 10 %, drawn
only for classes with >= 8 observed events. Observed regressions: OLS t-test, FDR 10 %. Observed super El Nino
lifecycle: the 3-event observed composite placed among 100,000 3-event composites drawn from the models, FDR 10 %.

    python scripts/sst/enso_impacts_render.py            # -> assets/sst/enso_imp_*.webp + assets/sst/data/enso_impacts.json
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REF = HERE / "reference"
SITE = Path(os.environ.get("SST_SITE_ROOT", HERE.parents[1]))
OUT = SITE / "assets" / "sst"

MUTED = "#6f6b64"
# The PDO views (same-season regressions on the ENSO-free / raw PDO, the phase split and its interaction, the "how much
# is ENSO" table) are WITHDRAWN from the site (2026-09-27, user: "I would remove the PDO maps for now until we have done a
# full causality analysis"): a same-season regression mostly shows the atmosphere driving the SST. The code and reference
# stay; ENSO_IMPACTS_PDO=1 draws them again.
PDO_VIEWS = os.environ.get("ENSO_IMPACTS_PDO") == "1"
SEASONS = ("DJF", "MAM", "JJA", "SON")
SNAME = {"DJF": "December–February", "MAM": "March–May", "JJA": "June–August", "SON": "September–November"}
VNAME = {"tas": "temperature", "pr": "precipitation"}
OBSNAME = {"tas": "ERA5", "pr": "GPCP v2.3"}
CLASS = {"en": "El Niño", "sen": "Strong El Niño", "super": "Super El Niño", "ln": "La Niña", "sln": "Strong La Niña"}
THR = {"en": "≥ +0.5 K", "sen": "≥ +1.5 K", "super": "≥ +2.0 K", "ln": "≤ −0.5 K", "sln": "≤ −1.5 K"}
PHASE = {"JJA0": "JJA before the peak", "SON0": "SON before the peak", "DJF": "DJF peak", "MAM1": "MAM after the peak"}
PHSEA = {"JJA0": "JJA", "SON0": "SON", "DJF": "DJF", "MAM1": "MAM"}
EXT = [-170, -30, -57.5, 75]
LEV = {"comp_tas": [-3, -2, -1.5, -1, -0.6, -0.3, -0.1, 0.1, 0.3, 0.6, 1, 1.5, 2, 3],
       "comp_pr": [-60, -45, -30, -20, -10, -5, 5, 10, 20, 30, 45, 60],
       "reg_tas": [-1.5, -1.0, -0.75, -0.5, -0.3, -0.15, -0.05, 0.05, 0.15, 0.3, 0.5, 0.75, 1.0, 1.5],
       "reg_pr": [-40, -30, -20, -15, -10, -5, -2, 2, 5, 10, 15, 20, 30, 40]}
UNIT = {"comp_tas": "2 m temperature anomaly (K)", "comp_pr": "precipitation (% of normal)",
        "reg_tas": "K per K of Niño-3.4", "reg_pr": "% of normal per K of Niño-3.4"}
CMIP_TEST = ("Shaded only where robust: across-model t-test passes Benjamini–Hochberg FDR 10 % over the map and ≥ 80 % of "
             "the models agree on the sign; blank = not significant.")
MODELS_TXT = "CMIP6 historical 1950–2014"


def load():
    z = np.load(REF / "enso_impacts_site.npz")
    meta = json.loads((REF / "enso_impacts_site.json").read_text())
    return z, meta


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    return plt


def draw_map(path, field, sig, lat, lon, levels, cmap, unit, title, sub, foot, dry=None, extent=None, note_empty=None):
    """One map: the field shaded only where `sig`, blank elsewhere; dry cells (no % of normal) hatched."""
    plt = _plt()
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import textwrap
    from matplotlib.colors import BoundaryNorm, ListedColormap
    ext = extent or EXT
    pc = ccrs.PlateCarree()
    W = 7.6
    AX0, AXW = 0.075, 0.905
    mh = W * AXW * (ext[3] - ext[2]) / (ext[1] - ext[0])
    titl = textwrap.wrap(title, 68); subl = textwrap.wrap(sub, 92); footl = textwrap.wrap(foot, 118)
    top = 0.42 + 0.24 * (len(titl) - 1) + 0.19 * len(subl) + 0.1
    fh = 0.14 * len(footl) + 0.1                       # footnote block, then tick labels + unit label, then the bar
    bar_y = fh + 0.46
    bot = bar_y + 0.12 + 0.3
    H = mh + top + bot
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([AX0, bot / H, AXW, mh / H], projection=pc)
    ax.set_extent(ext, crs=pc)
    ax.add_feature(cfeature.LAND, facecolor="#f1f0eb", zorder=0)
    lo = np.where(lon > 180, lon - 360, lon)
    base = plt.get_cmap(cmap)(np.linspace(0.06, 0.94, 256))
    cm = ListedColormap(base); norm = BoundaryNorm(levels, cm.N, extend="both")
    # 4x bilinear upsampling of the field and of the pass/fail mask, so the edge of the tested region follows the
    # grid-cell boundaries smoothly instead of as a 2.5-degree staircase; a point is shaded where the interpolated mask
    # is >= 0.5, i.e. nearer to passing cells than to failing ones
    from scipy.ndimage import zoom
    ok = sig & np.isfinite(field)
    fz = zoom(np.where(np.isfinite(field), field, 0.0), 4, order=1)
    mz = zoom(ok.astype(float), 4, order=1)
    la = np.linspace(lat[0], lat[-1], fz.shape[0]); lz = np.linspace(lo[0], lo[-1], fz.shape[1])
    show = np.where(mz >= 0.5, fz, np.nan)
    mappable = ax.contourf(lz, la, show, levels=levels, cmap=cm, norm=norm, extend="both", transform=pc, zorder=1)
    if dry is not None:
        dry = ~np.isfinite(field)
    if dry is not None and dry.any():
        dz = zoom(dry.astype(float), 4, order=1)
        ax.contourf(lz, la, dz, levels=[0.5, 1.5], colors="none", hatches=["////"], transform=pc, zorder=2)
    ax.coastlines(resolution="50m", linewidth=0.45, color="#222", zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.25, edgecolor="#666", zorder=3)
    ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.12, edgecolor=MUTED, zorder=3)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color=MUTED, alpha=0.45, xlocs=range(-180, 1, 30),
                      ylocs=range(-60, 91, 30), zorder=4)
    gl.top_labels = gl.right_labels = False; gl.xlabel_style = gl.ylabel_style = {"size": 7, "color": "#555"}
    if not np.any(sig & np.isfinite(field)):
        ax.text(0.5, 0.5, note_empty or "No significant signal", transform=ax.transAxes, ha="center", va="center",
                fontsize=11, color="#333", zorder=6, bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#bbb", lw=0.6))
    fig.text(0.03, 1 - 0.12 / H, "\n".join(titl), fontsize=12.5, fontweight="bold", va="top", linespacing=1.15)
    fig.text(0.03, 1 - (0.44 + 0.24 * (len(titl) - 1)) / H, "\n".join(subl), fontsize=8.6, color="#3d3a36", va="top",
             linespacing=1.3)
    cax = fig.add_axes([0.2, bar_y / H, 0.6, 0.12 / H])
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal", extend="both", spacing="uniform", ticks=levels)
    cb.set_label(unit, fontsize=8.4, labelpad=2); cb.ax.tick_params(labelsize=6.8, pad=1.5)
    cb.ax.set_xticklabels([f"{t:g}" for t in levels])
    fig.text(0.03, fh / H, "\n".join(footl), fontsize=7.2, color=MUTED, va="top", linespacing=1.35)
    fig.savefig(path, dpi=105, facecolor="white", pil_kwargs={"quality": 84, "method": 6})
    plt.close(fig)


def dry_mask(z, s, lat, lon):
    """A flag that this is a precipitation map: draw_map hatches the cells the map itself left unscored."""
    return np.zeros((len(lat), len(lon)), bool)


def main() -> int:
    z, meta = load()
    lat, lon = z["lat"], z["lon"]
    M = meta["maps"]; nmod, nmem = meta["models"], meta["members"]
    OUT.mkdir(parents=True, exist_ok=True)
    made, index = [], {"made": None, "maps": {}}

    def get(key):
        return (z[key + "|f"], z[key + "|s"]) if key + "|f" in z else (None, None)

    def rec(name, key, extra=None):
        e = dict(M.get(key, {})); e.update(extra or {})
        index["maps"][name] = e; made.append(name)

    for v in ("tas", "pr"):
        cmap = "RdBu_r" if v == "tas" else "BrBG"
        for s in SEASONS:
            dry = dry_mask(z, s, lat, lon) if v == "pr" else None
            dry_txt = " Hatched: normal under 0.3 mm/day, not scored." if v == "pr" else ""
            # regression, CMIP6 and observed
            for src in ("cmip6", "obs"):
                key = f"reg|{v}|{s}|{src}"; f, sg = get(key); e = M.get(key, {})
                if f is None:
                    continue
                if src == "cmip6":
                    title = f"CMIP6 · {s} {VNAME[v]} regressed on Niño-3.4"
                    sub = (f"Change per 1 K of the same season's Niño-3.4 index · {e['models']} models, {e['members']} members, "
                           f"{MODELS_TXT} · {100 * e['land_robust']:.0f} % of land robust")
                    foot = (CMIP_TEST + " Each member: seasonal anomalies and the index with the member's own quadratic trend "
                            "removed, OLS slope per grid point" + (", precipitation over the member's own normal" if v == "pr" else "")
                            + "; members averaged within a model, models weighted equally." + dry_txt)
                else:
                    title = f"{OBSNAME[v]} · {s} {VNAME[v]} regressed on Niño-3.4"
                    sub = (f"Change per 1 K of CPC's ONI ({s}) · {e['n']} seasons, {e['span'][0]}–{e['span'][1]} · "
                           f"{100 * e['land_sig']:.0f} % of land significant")
                    foot = ("Shaded only where the regression slope is significant: t-test per grid point, Benjamini–Hochberg "
                            "FDR 10 % over the map; blank = not significant. Quadratic trend removed from the field; "
                            f"{'ERA5 2 m temperature (Copernicus C3S)' if v == 'tas' else 'GPCP v2.3 monthly precipitation (NOAA NCEI)'}."
                            + dry_txt)
                name = f"enso_imp_reg_{v}_{s}_{src}"
                draw_map(OUT / f"{name}.webp", f, sg, lat, lon, LEV[f"reg_{v}"], cmap, UNIT[f"reg_{v}"], title, sub, foot,
                         dry=dry)
                rec(name, key)
            # CMIP6 class composites
            for c in CLASS:
                key = f"x|{v}|{s}|{c}"; f, sg = get(key); e = M.get(key, {})
                name = f"enso_imp_comp_{v}_{s}_{c}"
                title = f"CMIP6 · {s} {VNAME[v]} in {CLASS[c]} seasons"
                if f is None:
                    continue
                sub = (f"{CLASS[c]} ({THR[c]} same-season Niño-3.4) minus neutral seasons · {e['events']:,} events, "
                       f"{e['models']} models, {MODELS_TXT} · {100 * e['land_robust']:.0f} % of land robust")
                foot = (CMIP_TEST + " Member anomalies and index with the member's own quadratic trend removed; class minus "
                        "neutral (|index| < 0.5 K); members averaged within a model (event-weighted), models weighted equally."
                        + dry_txt)
                draw_map(OUT / f"{name}.webp", f, sg, lat, lon, LEV[f"comp_{v}"], cmap, UNIT[f"comp_{v}"], title, sub, foot, dry=dry)
                rec(name, key)
            # observed class composites (tested classes only)
            for c in ("en", "ln"):
                key = f"obs|{v}|{s}|{c}"; f, sg = get(key); e = M.get(key, {})
                name = f"enso_imp_obs_{v}_{s}_{c}"
                if f is None:                        # too few observed seasons: a labelled blank, never an untested map
                    n = e.get("n", 0)
                    draw_map(OUT / f"{name}.webp", np.full((len(lat), len(lon)), np.nan), np.zeros((len(lat), len(lon)), bool),
                             lat, lon, LEV[f"comp_{v}"], cmap, UNIT[f"comp_{v}"],
                             f"{OBSNAME[v]} · {s} {VNAME[v]} in {CLASS[c]} seasons",
                             f"{CLASS[c]} ({THR[c]} ONI) minus neutral seasons · {n} observed seasons",
                             "Classes with fewer than 8 observed seasons are not tested, so nothing is drawn.",
                             note_empty=f"Not tested: only {n} observed seasons")
                    rec(name, key, {"tested": False})
                    continue
                title = f"{OBSNAME[v]} · {s} {VNAME[v]} in {CLASS[c]} seasons"
                sub = (f"{CLASS[c]} ({THR[c]} ONI) minus neutral seasons · {e['n']} observed seasons, "
                       f"{e['span'][0]}–{e['span'][1]}")
                foot = ("Shaded only where significant: Welch t-test of the class against neutral seasons per grid point, "
                        "Benjamini–Hochberg FDR 10 % over the map; blank = not significant. Quadratic trend removed. Strong "
                        "and super classes have fewer than 8 observed seasons and are not tested." + dry_txt)
                draw_map(OUT / f"{name}.webp", f, sg, lat, lon, LEV[f"comp_{v}"], cmap, UNIT[f"comp_{v}"], title, sub, foot, dry=dry)
                rec(name, key)
            # asymmetry
            for c, lab in (("asym", "|index| ≥ 0.5 K"), ("asym_strong", "|index| ≥ 1.5 K")):
                key = f"x|{v}|{s}|{c}"; f, sg = get(key); e = M.get(key, {})
                if f is None:
                    continue
                name = f"enso_imp_asym_{v}_{s}_{c}"
                title = f"CMIP6 · {s} {VNAME[v]}: what does not flip sign"
                sub = (f"El Niño plus La Niña, each rescaled to the same index size ({lab}) · zero for a symmetric response · "
                       f"{e['models']} models, {MODELS_TXT}")
                foot = (CMIP_TEST + " Positive: El Niño's anomaly is larger than La Niña's opposite one (or both push the same way)."
                        + dry_txt)
                draw_map(OUT / f"{name}.webp", f, sg, lat, lon, LEV[f"comp_{v}"], cmap, UNIT[f"comp_{v}"], title, sub, foot, dry=dry)
                rec(name, key)
        # super El Nino lifecycle + beyond linear, observed vs the models
        for ph in PHASE:
            s = PHSEA[ph]
            dry = dry_mask(z, s, lat, lon) if v == "pr" else None
            dry_txt = " Hatched: normal under 0.3 mm/day, not scored." if v == "pr" else ""
            for kind in ("super", "nonlin"):
                key = f"life|{v}|{ph}|{kind}"; f, sg = get(key); e = M.get(key, {})
                if f is None:
                    continue
                name = f"enso_imp_life_{v}_{ph}_{kind}"
                if kind == "super":
                    title = f"CMIP6 · super El Niño, {VNAME[v]}, {PHASE[ph]}"
                    sub = (f"Winters with DJF Niño-3.4 ≥ +2.0 K minus neutral winters, followed through the event · "
                           f"{e['events']:,} events, {e['models']} models, {MODELS_TXT}")
                else:
                    title = f"CMIP6 · super El Niño beyond a scaled-up El Niño, {PHASE[ph]}"
                    sub = (f"Super composite minus the linear regression on the index × the mean super index: what the "
                           f"strongest events add · {VNAME[v]} · {e['events']:,} events, {e['models']} models")
                draw_map(OUT / f"{name}.webp", f, sg, lat, lon, LEV[f"comp_{v}"], cmap, UNIT[f"comp_{v}"], title, sub,
                         CMIP_TEST + dry_txt, dry=dry)
                rec(name, key)
            key = f"ovm|{v}|{ph}|super"; f, sg = get(key); e = M.get(key, {})
            if f is not None:
                name = f"enso_imp_ovm_{v}_{ph}"
                yrs = ", ".join(f"{y - 1}/{str(y)[2:]}" for y in e.get("years", []))
                title = f"{OBSNAME[v]} · observed super El Niño against the models, {PHASE[ph]}"
                sub = (f"The observed composite ({yrs}) where it lies outside the spread of 3-event composites drawn from "
                       f"the models · {e['cells']} grid points pass")
                foot = ("Test: the observed 3-event composite (minus neutral winters) placed among 100,000 3-event composites "
                        "drawn from the CMIP6 members (a model at random, then 3 of its super events); two-sided p, "
                        "Benjamini–Hochberg FDR 10 % over the map; blank = inside the models' range." + dry_txt)
                draw_map(OUT / f"{name}.webp", f, sg, lat, lon, LEV[f"comp_{v}"], cmap, UNIT[f"comp_{v}"], title, sub, foot,
                         dry=dry, note_empty="Inside the models' range everywhere (no point passes)")
                rec(name, key)
    # super El Nino over North America, month by month
    nlat, nlon = z["na_lat"], z["na_lon"]
    for mname, full in (("Nov", "November"), ("Dec", "December"), ("Jan", "January"), ("Feb", "February"), ("Mar", "March")):
        key = f"namon|tas|{mname}|super"; f, sg = get(key); e = M.get(key, {})
        if f is None:
            continue
        name = f"enso_imp_namon_{mname}"
        draw_map(OUT / f"{name}.webp", f, sg, nlat, nlon, LEV["comp_tas"], "RdBu_r", UNIT["comp_tas"],
                 f"CMIP6 · North American temperature in super El Niño winters, {full}",
                 f"DJF Niño-3.4 ≥ +2.0 K minus neutral winters · {e['events']:,} super winters, {e['models']} models, "
                 f"{e['members']} members, {MODELS_TXT} · {100 * e['robust']:.0f} % of the map robust",
                 CMIP_TEST + " Each calendar month's anomaly from the member's own quadratic trend in that month.",
                 extent=[-165, -52, 14, 72])
        rec(name, key)
    table_figs(meta, made, index)
    if (REF / "enso_modes_site.npz").exists():
        render_modes(made, index)
    if (REF / "enso_z500_site.npz").exists():
        render_z500(made, index)
    if (REF / "enso_pdo_causal_site.npz").exists():
        render_pdo_causal(made, index)
    if (REF / "enso_pdo_monthly_site.npz").exists():
        render_pdo_monthly(made, index)
    import datetime as dt
    index["made"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    index["models"], index["members"] = nmod, nmem
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    (OUT / "data" / "enso_impacts.json").write_text(json.dumps(index, separators=(",", ":")))
    print(f"{len(made)} figures -> {OUT}/enso_imp_*.webp")
    return 0


ZLEV = {"reg": [-60, -40, -30, -20, -15, -10, -5, -2, 2, 5, 10, 15, 20, 30, 40, 60],
        "comp": [-150, -100, -75, -50, -35, -20, -10, -5, 5, 10, 20, 35, 50, 75, 100, 150]}


def lonlab(x):
    x = ((x + 180) % 360) - 180
    return f"{abs(x):.0f}°{'W' if x < 0 else 'E'}" if abs(x) not in (0, 180) else f"{abs(x):.0f}°"


def draw_nh(path, field, sig, clim, lat, lon, levels, unit, title, sub, foot, note_empty=None):
    """Flat Northern Hemisphere map (plate carree centred on 180, 0-90N): the field shaded only where `sig`, the seasonal
    climatological 500 hPa height as thin contours for orientation."""
    plt = _plt()
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import textwrap
    from cartopy.util import add_cyclic_point
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from scipy.ndimage import zoom
    pc = ccrs.PlateCarree(); proj = ccrs.PlateCarree(central_longitude=180)
    W = 11.0; AX0, AXW = 0.05, 0.93
    mh = W * AXW * 90 / 360
    titl = textwrap.wrap(title, 100); subl = textwrap.wrap(sub, 135); footl = textwrap.wrap(foot, 170)
    top = 0.42 + 0.24 * (len(titl) - 1) + 0.19 * len(subl) + 0.08
    fh = 0.14 * len(footl) + 0.1; bar_y = fh + 0.46; bot = bar_y + 0.12 + 0.3
    H = mh + top + bot
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([AX0, bot / H, AXW, mh / H], projection=proj)
    ax.set_extent([-180, 180, 0, 90], crs=proj)
    ax.add_feature(cfeature.LAND, facecolor="#f1f0eb", zorder=0)
    base = plt.get_cmap("RdBu_r")(np.linspace(0.06, 0.94, 256)); cm = ListedColormap(base)
    norm = BoundaryNorm(levels, cm.N, extend="both")
    ok = sig & np.isfinite(field)
    f_c, lon_c = add_cyclic_point(np.where(np.isfinite(field), field, 0.0), coord=lon)
    m_c, _ = add_cyclic_point(ok.astype(float), coord=lon)
    fz = zoom(f_c, 3, order=1); mz = zoom(m_c, 3, order=1)
    la = np.linspace(lat[0], lat[-1], fz.shape[0]); lz = np.linspace(lon_c[0], lon_c[-1], fz.shape[1])
    mappable = ax.contourf(lz, la, np.where(mz >= 0.5, fz, np.nan), levels=levels, cmap=cm, norm=norm, extend="both",
                           transform=pc, zorder=1)
    if clim is not None:
        c_c, _ = add_cyclic_point(clim, coord=lon)
        cs = ax.contour(lon_c, lat, c_c, levels=np.arange(5000, 6000, 100), colors="#555", linewidths=0.45, alpha=0.75,
                        transform=pc, zorder=2)
        ax.clabel(cs, levels=[5200, 5500, 5800], fmt="%d", fontsize=6.5, inline=True)
    ax.coastlines(resolution="110m", linewidth=0.45, color="#222", zorder=3)
    gl = ax.gridlines(crs=pc, draw_labels=True, linewidth=0.3, color=MUTED, alpha=0.45, xlocs=range(-180, 181, 60),
                      ylocs=[0, 30, 60, 90], zorder=4)
    gl.top_labels = gl.right_labels = False; gl.xlabel_style = gl.ylabel_style = {"size": 7, "color": "#555"}
    if not ok.any():
        ax.text(0.5, 0.5, note_empty or "No significant signal", transform=ax.transAxes, ha="center", va="center",
                fontsize=11, color="#333", zorder=6, bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#bbb", lw=0.6))
    fig.text(0.03, 1 - 0.12 / H, "\n".join(titl), fontsize=12.5, fontweight="bold", va="top", linespacing=1.15)
    fig.text(0.03, 1 - (0.44 + 0.24 * (len(titl) - 1)) / H, "\n".join(subl), fontsize=8.6, color="#3d3a36", va="top", linespacing=1.3)
    cax = fig.add_axes([0.25, bar_y / H, 0.5, 0.12 / H])
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal", extend="both", spacing="uniform", ticks=levels)
    cb.set_label(unit, fontsize=8.4, labelpad=2); cb.ax.tick_params(labelsize=6.8, pad=1.5)
    cb.ax.set_xticklabels([f"{t:g}" for t in levels])
    fig.text(0.03, fh / H, "\n".join(footl), fontsize=7.2, color=MUTED, va="top", linespacing=1.35)
    fig.savefig(path, dpi=100, facecolor="white", pil_kwargs={"quality": 84, "method": 6})
    plt.close(fig)


CLEV = [-40, -30, -20, -15, -10, -6, -3, -1, 1, 3, 6, 10, 15, 20, 30, 40]
CAUSAL = [("covar_cmip6", "CMIP6: same-winter co-variability (not a PDO effect)",
           "DJF 500 hPa height on the same winter's ENSO-free PDO, Niño-3.4 alongside · 147 members, 16 models"),
          ("covar_era5", "ERA5: same-winter co-variability (not a PDO effect)",
           "DJF 500 hPa height on the same winter's ENSO-free PDO, Niño-3.4 alongside · 1960–2026"),
          ("lag", "CMIP6: forced response, lagged",
           "DJF on the preceding SON's ENSO-free PDO, with SON Aleutian low and SON/DJF Niño-3.4 held fixed"),
          ("frank2", "CMIP6: forced response, Frankignoul lagged covariance",
           "cov(Z in DJF, PDO two months earlier) / cov(PDO, PDO two months earlier), same-month Niño-3.4 removed"),
          ("amip", "AMIP: forced response with observed SST prescribed",
           "DJF on the observed ENSO-free PDO, Niño-3.4, Indian Ocean and warm-pool indices alongside · 1980–2014"),
          ("dcpp_ipsl", "IPSL-CM6A-LR pacemaker: forced response",
           "DCPP-C NexTrop: the Pacific pattern imposed north of the tropics only, (pos − neg) / 2 per sd of the imposed PDO"),
          ("dcpp_hadgem", "HadGEM3-GC31-MM pacemaker: forced response",
           "DCPP-C NexTrop: the Pacific pattern imposed north of the tropics only, (pos − neg) / 2 per sd of the imposed PDO")]
CAUSAL_TEST = {"covar_cmip6": "robust: across-model t-test FDR 10 % and ≥ 80 % sign agreement",
               "covar_era5": "OLS t-test per grid point, FDR 10 %", "lag": "robust: across-model t-test FDR 10 % and ≥ 80 % sign agreement",
               "frank2": "robust: across-model t-test FDR 10 % and ≥ 80 % sign agreement",
               "amip": "robust: across-model t-test FDR 10 % and ≥ 80 % sign agreement",
               "dcpp_ipsl": "Welch t-test of pos vs neg winters per grid point, FDR 10 %",
               "dcpp_hadgem": "Welch t-test of pos vs neg winters per grid point, FDR 10 %"}


def render_pdo_causal(made, index):
    """The PDO's forced effect: same-winter co-variability against four causal estimates (lagged, Frankignoul, AMIP,
    DCPP-C pacemakers), from reference/enso_pdo_causal_site.{npz,json} (cmip6_pdo_causality.py site)."""
    z = np.load(REF / "enso_pdo_causal_site.npz")
    meta = json.loads((REF / "enso_pdo_causal_site.json").read_text())
    lat, lon = z["lat"], z["lon"]; R = meta["rows"]
    zc = np.load(REF / "enso_z500_site.npz") if (REF / "enso_z500_site.npz").exists() else None
    clim = zc["clim_obs|DJF"] if zc is not None and "clim_obs|DJF" in zc.files else None
    for k, title, sub in CAUSAL:
        r = R[k]
        extra = f" · Aleutian box {r['AL_box']:+.1f} gpm per sd" + (f" (95 % CI {r['ci95'][0]:+.1f} to {r['ci95'][1]:+.1f})" if "ci95" in r else "")
        draw_nh(OUT / f"enso_imp_pdocause_{k}.webp", z[f"{k}|f"], z[f"{k}|s"], clim, lat, lon, CLEV, "500 hPa height, gpm per sd of the ENSO-free PDO",
                f"{title} · DJF 500 hPa height", sub + extra,
                f"Shaded only where significant ({CAUSAL_TEST[k]}); blank = not significant. One colour scale for every view so the "
                "amplitudes compare. Contours: ERA5 DJF mean height every 100 m.", note_empty="No significant forced response")
        made.append(f"enso_imp_pdocause_{k}"); index["maps"][f"enso_imp_pdocause_{k}"] = r
    causal_panels(z, meta, clim, made, index)
    causal_table(meta, made, index)


def causal_panels(z, meta, clim, made, index):
    plt = _plt()
    import cartopy.crs as ccrs
    import textwrap
    from cartopy.util import add_cyclic_point
    from matplotlib.colors import BoundaryNorm, ListedColormap
    lat, lon = z["lat"], z["lon"]; R = meta["rows"]
    pc = ccrs.PlateCarree(); proj = ccrs.PlateCarree(central_longitude=180)
    cm = ListedColormap(plt.get_cmap("RdBu_r")(np.linspace(0.06, 0.94, 256))); norm = BoundaryNorm(CLEV, cm.N, extend="both")
    W = 13.0; pw = 6.1; ph = pw * 80 / 360; gap = 0.5
    top = 0.78; bot = 1.0; H = top + 4 * (ph + gap) + bot
    SHORT = {"covar_cmip6": "CMIP6 · co-variability", "covar_era5": "ERA5 · co-variability", "lag": "CMIP6 lagged · forced",
             "frank2": "CMIP6 Frankignoul, 2 months · forced", "amip": "AMIP · forced", "dcpp_ipsl": "IPSL pacemaker · forced",
             "dcpp_hadgem": "HadGEM3 pacemaker · forced"}
    fig = plt.figure(figsize=(W, H))
    for i, (k, title, _) in enumerate(CAUSAL):
        r_, c_ = divmod(i, 2)
        x0 = 0.25 + c_ * (pw + 0.35); y0 = H - top - (r_ + 1) * (ph + gap) + 0.08
        ax = fig.add_axes([x0 / W, y0 / H, pw / W, ph / H], projection=proj)
        ax.set_extent([-180, 180, 10, 90], crs=proj)
        f, s_ = z[f"{k}|f"], z[f"{k}|s"]
        fc, lc = add_cyclic_point(np.where(s_, f, np.nan), coord=lon)
        mappable = ax.contourf(lc, lat, fc, levels=CLEV, cmap=cm, norm=norm, extend="both", transform=pc)
        if clim is not None:
            cc, _ = add_cyclic_point(clim, coord=lon)
            ax.contour(lc, lat, cc, levels=np.arange(5000, 6000, 100), colors="#666", linewidths=0.35, alpha=0.7, transform=pc)
        ax.coastlines(resolution="110m", linewidth=0.4, color="#222")
        r = R[k]
        ax.text(0, 1.03, SHORT[k], transform=ax.transAxes, fontsize=9.6, fontweight="bold", va="bottom")
        ci = f", 95 % CI {r['ci95'][0]:+.1f} to {r['ci95'][1]:+.1f}" if "ci95" in r else ""
        ax.text(1, 1.03, f"Aleutian box {r['AL_box']:+.1f}{ci}", transform=ax.transAxes, fontsize=8.2, color="#3d3a36", ha="right", va="bottom")
        if not s_[lat >= 20].any():
            ax.text(0.5, 0.5, "no significant cell", transform=ax.transAxes, ha="center", va="center", fontsize=9, color="#333",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbb", lw=0.5))
    # the eighth slot: the verdict
    x0 = 0.25 + (pw + 0.35); y0 = H - top - 4 * (ph + gap) + 0.08
    fig.text(x0 / W, (y0 + ph) / H, "\n".join(textwrap.wrap(
        "Verdict: no significant forced response in the lagged or AMIP analyses; the pacemaker runs bound it at about 20 gpm per "
        "sd or less. The familiar PDO maps (top row) are mostly the atmosphere driving the ocean.", 70)),
        fontsize=10, fontweight="bold", va="top", color="#1f3b57", linespacing=1.35)
    fig.text(0.25 / W, 1 - 0.12 / H, "The PDO's forced effect on the winter atmosphere: what survives causality tests", fontsize=14, fontweight="bold", va="top")
    fig.text(0.25 / W, 1 - 0.44 / H, "DJF 500 hPa height per standard deviation of the ENSO-free PDO · top row: same-winter co-variability, "
             "for contrast only · the rest: four estimates of the forced response · shaded only where significant", fontsize=9, color="#3d3a36", va="top")
    cax = fig.add_axes([0.3, 0.62 / H, 0.4, 0.12 / H])
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal", extend="both", spacing="uniform", ticks=CLEV)
    cb.set_label("gpm per sd of the ENSO-free PDO", fontsize=8.5, labelpad=1); cb.ax.tick_params(labelsize=7, pad=1)
    fig.text(0.25 / W, 0.1 / H, "Tests: CMIP6 across-model t-test FDR 10 % and ≥ 80 % sign agreement; ERA5 OLS t-test FDR 10 %; pacemakers "
             "Welch t-test of pos vs neg winters FDR 10 %. Contours: ERA5 DJF mean height every 100 m.", fontsize=7.4, color=MUTED, va="bottom")
    fig.savefig(OUT / "enso_imp_pdocause_all.webp", dpi=95, facecolor="white", pil_kwargs={"quality": 84, "method": 6}); plt.close(fig)
    made.append("enso_imp_pdocause_all"); index["maps"]["enso_imp_pdocause_all"] = {"panels": [k for k, *_ in CAUSAL]}


def causal_table(meta, made, index):
    plt = _plt()
    import textwrap
    R = meta["rows"]; G = meta["granger"]
    pf = lambda p: ("p < 0.001" if p < 0.001 else f"p = {p:.2f}" if p >= 0.01 else f"p = {p:.3f}")
    rows = [("Same-winter co-variability, CMIP6", f"{R['covar_cmip6']['AL_box']:+.1f}", f"{R['covar_cmip6']['agree']} models, {pf(R['covar_cmip6']['p'])}", "the contrast, not an effect"),
            ("Same-winter co-variability, ERA5", f"{R['covar_era5']['AL_box']:+.1f}", f"{pf(R['covar_era5']['p'])}, 1960–2026", "the contrast, not an effect"),
            ("Lagged: SON PDO → DJF, atmosphere held fixed", f"{R['lag']['AL_box']:+.1f}", f"{R['lag']['agree']} models, {pf(R['lag']['p'])}", "not significant"),
            ("Frankignoul lagged covariance, 2 months", f"{R['frank2']['AL_box']:+.1f}", f"{R['frank2']['agree']} models, {pf(R['frank2']['p'])}", "not significant"),
            ("Frankignoul lagged covariance, 3 months", f"{R['frank2']['tau3']:+.1f}", f"{pf(R['frank2']['tau3_p'])}", "not significant"),
            ("AMIP, observed SST prescribed", f"{R['amip']['AL_box']:+.1f}", f"{R['amip']['agree']} models, {pf(R['amip']['p'])}", "not significant"),
            ("Pacemaker IPSL-CM6A-LR", f"{R['dcpp_ipsl']['AL_box']:+.1f}", f"95 % CI {R['dcpp_ipsl']['ci95'][0]:+.1f} to {R['dcpp_ipsl']['ci95'][1]:+.1f}, {pf(R['dcpp_ipsl']['p'])}", "box only; no grid point passes"),
            ("Pacemaker HadGEM3-GC31-MM", f"{R['dcpp_hadgem']['AL_box']:+.1f}", f"95 % CI {R['dcpp_hadgem']['ci95'][0]:+.1f} to {R['dcpp_hadgem']['ci95'][1]:+.1f}, {pf(R['dcpp_hadgem']['p'])}", "not significant")]
    fig = plt.figure(figsize=(10.4, 6.2)); H = 6.2
    fig.text(0.02, 1 - 0.12 / H, "The PDO and the Aleutian low: co-variability against the forced response", fontsize=12.5, fontweight="bold", va="top")
    fig.text(0.02, 1 - 0.44 / H, "DJF 500 hPa height averaged over 30–65°N, 160°E–140°W, gpm per standard deviation of the ENSO-free PDO",
             fontsize=8.8, color="#3d3a36", va="top")
    xs = [0.02, 0.47, 0.57, 0.82]
    y = 1 - 0.85 / H
    for x, h in zip(xs, ["estimate", "gpm / sd", "test", "reading"]):
        fig.text(x, y, h, fontsize=8.6, fontweight="bold", va="top")
    for i, row in enumerate(rows):
        yy = y - (0.34 + 0.3 * i) / H
        for j, (x, t) in enumerate(zip(xs, row)):
            fig.text(x, yy, t, fontsize=8.4, va="top", color="#7a3b12" if i < 2 else "#111", fontweight="bold" if (j == 1 and i < 2) else "normal")
    yg = y - (0.34 + 0.3 * len(rows) + 0.25) / H
    fig.text(0.02, yg, "Which way does the influence run? (Granger tests, 16 CMIP6 models, nested regressions)", fontsize=10, fontweight="bold", va="top")
    a, b = G["p_to_al"], G["al_to_p"]
    lines = [f"PDO → atmosphere: adding the autumn PDO does not improve the winter Aleutian low out of sample (median gain "
             f"{100 * a['cv_gain_median']:+.1f} %, negative in {a['cv_negative']} models); a median {100 * a['members_p05']:.0f} % of members pass "
             f"F-test p < 0.05, which is chance. ERA5: p = {a['era5']['p']:.2f}.",
             f"Atmosphere → PDO: a deeper autumn Aleutian low predicts a more positive winter PDO beyond the PDO's own persistence in "
             f"{b['agree']} models ({pf(b['p'])}); a median {100 * b['members_p05']:.0f} % of members pass p < 0.05 (up to "
             f"{100 * b['members_p05_range'][1]:.0f} %). ERA5, one 67-winter record: p = {b['era5']['p']:.2f}."]
    yy = yg - 0.34 / H
    for ln in lines:
        w = textwrap.wrap(ln, 150)
        fig.text(0.02, yy, "\n".join(w), fontsize=8.4, va="top", linespacing=1.3); yy -= (0.2 * len(w) + 0.12) / H
    fig.text(0.02, 0.02, "\n".join(textwrap.wrap(
        f"Scale: one sd of the ENSO-free PDO is about {abs(meta['K_per_sd']):.2f} K of SST at the pattern's centre (35–45°N, 170°E–150°W) "
        "in the models, so a 20 gpm per sd bound is roughly 45 gpm per K there; atmospheric models have given 10–20 gpm per K for "
        "extratropical SST anomalies (Kushnir et al. 2002). Tests: CMIP6 across-model one-sample t-test; ERA5 OLS t-test; pacemakers "
        "Welch t-test of the Aleutian-box mean, pos against neg winters. Frankignoul lagged covariance at 1 month: "
        f"{R['frank2']['tau1']:+.1f} ({pf(R['frank2']['tau1_p'])}) but inside the atmosphere's own persistence and shaped like the "
        "co-variability, so not a clean estimate.", 165)), fontsize=7.2, color=MUTED, va="bottom", linespacing=1.3)
    fig.savefig(OUT / "enso_imp_pdocause_table.webp", dpi=110, facecolor="white", pil_kwargs={"quality": 86, "method": 6}); plt.close(fig)
    made.append("enso_imp_pdocause_table"); index["maps"]["enso_imp_pdocause_table"] = {"granger": G}


MON3 = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONF = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
PMLEV = {"tas": [-1.0, -0.6, -0.4, -0.3, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 1.0],
         "pr": [-0.5, -0.3, -0.2, -0.15, -0.1, -0.05, -0.02, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]}
PMUNIT = {"tas": "2 m temperature, K per sd", "pr": "precipitation, mm/day per sd", "z500": "500 hPa height, gpm per sd"}
NA_EXT = [-170, -50, 15, 75]
PMETH = {
    "cpl_coast": ("CMIP6 coupled · forced response to the coastal SST",
                  "Month-m field on the coastal NE Pacific SST (30–55°N, 135–120°W; 1 sd ≈ 0.53 K) of the month before, with the "
                  "month-before Pacific–North America z500 (3 PCs), Niño-3.4 and each cell's own previous month held fixed · 147 members, 16 models"),
    "cpl_pdo": ("CMIP6 coupled · forced response to the ENSO-free PDO",
                "Month-m field on the ENSO-free PDO of the month before, with the month-before Pacific–North America z500 (3 PCs), "
                "Niño-3.4 and each cell's own previous month held fixed · 147 members, 16 models"),
    "amip_coast": ("AMIP · forced response to the observed coastal SST",
                   "Observed SST prescribed, so the atmosphere cannot drive it: month-m field on the same month's coastal SST, "
                   "Niño-3.4, Indian Ocean and warm pool alongside · 26 members, 14 models, 1979–2014"),
    "amip_pdo": ("AMIP · forced response to the observed ENSO-free PDO",
                 "Observed SST prescribed: month-m field on the same month's ENSO-free PDO, Niño-3.4, Indian Ocean and warm pool "
                 "alongside · 26 members, 14 models, 1979–2014"),
    "obs": ("ERA5 / GPCP · lagged forced-response estimate",
            "Month-m field on the ENSO-free PDO of the month before, with ERA5's month-before North Pacific z500 (2 PCs) and "
            "Niño-3.4 held fixed · ERA5 1959–2026, GPCP 1979–2026"),
    "covar": ("NOT the PDO's effect · same-month co-variability (contrast)",
              "Month-m field on the SAME month's ENSO-free PDO: mostly the atmosphere driving the ocean, shown only as the contrast · "
              "147 members, 16 models"),
    "ipsl": ("IPSL-CM6A-LR pacemaker · forced response",
             "DCPP-C NexTrop: the Pacific decadal SST pattern imposed north of the tropics only, (pos − neg) / 2 per sd of the "
             "imposed PDO · 10 runs a side"),
    "hadgem": ("HadGEM3-GC31-MM pacemaker · forced response",
               "DCPP-C NexTrop: the Pacific decadal SST pattern imposed north of the tropics only, (pos − neg) / 2 per sd of the "
               "imposed PDO · 12 and 11 runs"),
}
PMTEST = {"cpl_coast": "across-model t-test, FDR 10 % over North American land (20–90°N for height) and ≥ 80 % sign agreement",
          "cpl_pdo": "across-model t-test, FDR 10 % over North American land (20–90°N for height) and ≥ 80 % sign agreement",
          "amip_coast": "across-model t-test, FDR 10 % over North American land and ≥ 80 % sign agreement",
          "amip_pdo": "across-model t-test, FDR 10 % over North American land (20–90°N for height) and ≥ 80 % sign agreement",
          "obs": "OLS t-test per grid point, FDR 10 % over North American land (20–90°N for height)",
          "covar": "across-model t-test, FDR 10 % and ≥ 80 % sign agreement",
          "ipsl": "Welch t-test of pos against neg seasons per grid point, FDR 10 %",
          "hadgem": "Welch t-test of pos against neg seasons per grid point, FDR 10 %"}


def render_pdo_monthly(made, index):
    """PDO forced effects by month over North America (reference/enso_pdo_monthly_site.*, cmip6_pdo_monthly_causal.py site)."""
    z = np.load(REF / "enso_pdo_monthly_site.npz")
    meta = json.loads((REF / "enso_pdo_monthly_site.json").read_text())
    M = meta["maps"]
    latn, lonn, latz, lonz = z["latn"], z["lonn"], z["latz"], z["lonz"]
    zc = np.load(REF / "enso_z500_site.npz") if (REF / "enso_z500_site.npz").exists() else None
    for key in M:
        tag, var, t = key.split("|")
        if f"{key}|f" not in z.files:
            continue
        f, sg = z[f"{key}|f"], z[f"{key}|s"]
        title0, sub0 = PMETH[tag]
        when = MONF[int(t) - 1] if t.isdigit() else t
        name = f"enso_imp_pdomon_{tag}_{var}_{t}"
        vname = {"tas": "2 m temperature", "pr": "precipitation", "z500": "500 hPa height"}[var]
        title = f"{title0} · {when} {vname}"
        empty = "No significant forced response" if tag != "covar" else "No significant co-variability"
        foot = (f"Shaded only where significant ({PMTEST[tag]}); blank = not significant. "
                + ("This is NOT the PDO's effect: a same-month regression mostly shows the atmosphere building the SST anomaly."
                   if tag == "covar" else "Per standard deviation of the index."))
        if var == "z500":
            sea = {1: "DJF", 2: "DJF", 12: "DJF", 3: "MAM", 4: "MAM", 5: "MAM", 6: "JJA", 7: "JJA", 8: "JJA"}.get(int(t) if t.isdigit() else 0, "SON")
            sea = t if not t.isdigit() else sea
            clim = zc[f"clim_obs|{sea}"] if zc is not None and f"clim_obs|{sea}" in zc.files else None
            draw_nh(OUT / f"{name}.webp", f, sg, clim, latz, lonz, CLEV, PMUNIT[var], title, sub0, foot, note_empty=empty)
        else:
            draw_map(OUT / f"{name}.webp", f, sg, latn, lonn, PMLEV[var], "RdBu_r" if var == "tas" else "BrBG", PMUNIT[var],
                     title, sub0, foot, extent=NA_EXT, note_empty=empty)
        made.append(name); index["maps"][name] = M[key]
    pdomonth_summary(meta, made, index)


def pdomonth_summary(meta, made, index):
    """Region x month heat tables of the significant forced temperature response, one per causal method: the coastal
    warming leads."""
    plt = _plt()
    from matplotlib.colors import BoundaryNorm, ListedColormap
    M = meta["maps"]; R = meta["regions"]
    lv = PMLEV["tas"]; cm = ListedColormap(plt.get_cmap("RdBu_r")(np.linspace(0.06, 0.94, 256))); norm = BoundaryNorm(lv, cm.N, extend="both")
    panels = [("cpl_coast", "CMIP6 coupled · coastal SST (per 0.53 K)", MON3, [f"{m:02d}" for m in range(1, 13)]),
              ("amip_coast", "AMIP · observed coastal SST", MON3, [f"{m:02d}" for m in range(1, 13)]),
              ("cpl_pdo", "CMIP6 coupled · ENSO-free PDO", MON3, [f"{m:02d}" for m in range(1, 13)]),
              ("amip_pdo", "AMIP · observed ENSO-free PDO", MON3, [f"{m:02d}" for m in range(1, 13)]),
              ("ipsl", "IPSL pacemaker · per sd of imposed PDO", SEASONS, list(SEASONS)),
              ("hadgem", "HadGEM3 pacemaker · per sd of imposed PDO", SEASONS, list(SEASONS))]
    W, H = 13.0, 11.6
    fig = plt.figure(figsize=(W, H))
    fig.text(0.02, 1 - 0.12 / H, "The North Pacific's forced effect on North American temperature, month by month", fontsize=14,
             fontweight="bold", va="top")
    fig.text(0.02, 1 - 0.44 / H, "Regional 2 m temperature response, K per sd of the index · coloured only where the house test passes; "
             "blank = no significant forced response\nThe one effect every method agrees on: warmer COASTAL air (Alaska's south coast, "
             "the Pacific Northwest and California coasts) when the nearby ocean is warm", fontsize=9, color="#3d3a36", va="top",
             linespacing=1.35)
    rl = [r.replace(" / Southwest", "/SW").replace("Northern Plains / Prairies", "N Plains") for r in R]
    for i, (tag, ttl, cols, keys) in enumerate(panels):
        r_, c_ = divmod(i, 2)
        x0 = 1.45 + c_ * 5.95; w = 4.6 if len(cols) == 12 else 2.2
        y0 = H - 1.05 - (r_ + 1) * 3.35 + 0.35; h = 2.65
        ax = fig.add_axes([x0 / W, y0 / H, w / W, h / H])
        V = np.full((len(R), len(cols)), np.nan)
        for j, k in enumerate(keys):
            e = M.get(f"{tag}|tas|{k}")
            if e:
                for ri, r in enumerate(R):
                    v = e["regions"].get(r)
                    if isinstance(v, (int, float)):
                        V[ri, j] = v
        ax.imshow(np.ones_like(V), cmap=ListedColormap(["#f3f2ee"]), aspect="auto")
        ax.imshow(np.ma.masked_invalid(V), cmap=cm, norm=norm, aspect="auto")
        for ri in range(len(R)):
            for j in range(len(cols)):
                if np.isfinite(V[ri, j]):
                    col = cm(norm(V[ri, j])); lum = 0.299 * col[0] + 0.587 * col[1] + 0.114 * col[2]
                    ax.text(j, ri, f"{V[ri, j]:+.2f}".replace("0.", "."), ha="center", va="center", fontsize=6.4,
                            color="white" if lum < 0.45 else "#111")
        ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, fontsize=7.5)
        ax.set_yticks(range(len(R))); ax.set_yticklabels(rl if c_ == 0 else [], fontsize=7.8)
        ax.tick_params(length=0); [sp.set_visible(False) for sp in ax.spines.values()]
        ax.set_xticks(np.arange(-0.5, len(cols)), minor=True); ax.set_yticks(np.arange(-0.5, len(R)), minor=True)
        ax.grid(which="minor", color="white", lw=1.2); ax.tick_params(which="minor", length=0)
        ax.set_title(ttl, fontsize=9.6, fontweight="bold", loc="left")
        for ri, r in enumerate(R):
            if "coast" in r:
                ax.get_yticklabels()[ri].set_fontweight("bold") if c_ == 0 else None
    fig.text(0.02, 0.012, "Tests: CMIP6 and AMIP across-model t-test, FDR 10 % over the 12 regions, ≥ 80 % of models agreeing; pacemakers "
             "Welch t-test of pos against neg seasons, FDR 10 %. ERA5/GPCP (lagged): no region passes in any month. Coupled estimates hold "
             "the month-before atmosphere and each cell's own previous month fixed; AMIP prescribes the observed SST.", fontsize=7.2,
             color=MUTED, va="bottom", wrap=True)
    fig.savefig(OUT / "enso_imp_pdomon_summary.webp", dpi=100, facecolor="white", pil_kwargs={"quality": 86, "method": 6}); plt.close(fig)
    made.append("enso_imp_pdomon_summary"); index["maps"]["enso_imp_pdomon_summary"] = {"summary": True}


def render_z500(made, index):
    """500 hPa height, flat Northern Hemisphere, for every view that has a temperature and precipitation map, from
    reference/enso_z500_site.{npz,json} (cmip6_z500_impacts.py)."""
    z = np.load(REF / "enso_z500_site.npz")
    meta = json.loads((REF / "enso_z500_site.json").read_text())
    M = meta["maps"]; lat, lon = z["lat"], z["lon"]
    nm_, nmem = meta["models"], meta["members"]
    OBS_T = ("Shaded only where significant: t-test per grid point, Benjamini–Hochberg FDR 10 % over the map (0–90°N); "
             "blank = not significant. ERA5 500 hPa height, quadratic trend removed; contours: ERA5 1959–2026 seasonal mean "
             "height every 100 m.")
    MOD_T = (f"Shaded only where robust: across-model t-test passes Benjamini–Hochberg FDR 10 % over the map (0–90°N) and "
             f"≥ 80 % of the models agree on the sign; blank = not significant. {nm_} CMIP6 models, {nmem} historical members "
             f"(up to 10 per model), 1950–2014, each member's own quadratic trend removed; contours: ERA5 seasonal mean height "
             f"every 100 m.")

    def get(key):
        return (z[key + "|f"], z[key + "|s"]) if key + "|f" in z else (None, None)

    def out(name, key, field, sig, s, lev, unit, title, sub, foot, note=None):
        draw_nh(OUT / f"{name}.webp", field, sig, z[f"clim_obs|{s}"] if f"clim_obs|{s}" in z else None, lat, lon, ZLEV[lev],
                unit, title, sub, foot, note_empty=note)
        index["maps"][name] = dict(M.get(key, {})); made.append(name)

    def blank(name, key, s, lev, title, sub, foot, note):
        out(name, key, np.full((len(lat), len(lon)), np.nan), np.zeros((len(lat), len(lon)), bool), s, lev, "gpm", title, sub, foot, note)

    pos = meta["position"]
    for s in SEASONS:
        ps = pos[s]
        for src in ("cmip6", "obs"):
            key = f"cmip6|reg|{s}" if src == "cmip6" else f"obs|reg|{s}"; f, sg = get(key); e = M.get(key, {})
            if src == "cmip6":
                sub = (f"Height change per 1 K of the same season's Niño-3.4 · {e['models']} models · {100 * e['sig_frac']:.0f} % of "
                       f"20–90°N robust · Aleutian low centre {lonlab(ps['aleutian_low']['cmip6_mean_map'])}, Canadian ridge "
                       f"{lonlab(ps['canadian_ridge']['cmip6_mean_map'])}; pattern r with ERA5 {ps['pattern_r']:+.2f}")
                title = f"CMIP6 · {s} 500 hPa height regressed on Niño-3.4"; foot = MOD_T
            else:
                sub = (f"Height change per 1 K of CPC's ONI · {e['n']} seasons, {e['span'][0]}–{e['span'][1]} · {100 * e['sig_frac']:.0f} % "
                       f"of 20–90°N significant · Aleutian low centre {lonlab(ps['aleutian_low']['obs'])}, Canadian ridge "
                       f"{lonlab(ps['canadian_ridge']['obs'])}")
                title = f"ERA5 · {s} 500 hPa height regressed on Niño-3.4"; foot = OBS_T
            out(f"enso_imp_reg_z500_{s}_{src}", key, f, sg, s, "reg", "gpm per K of Niño-3.4", title, sub, foot)
        for c in CLASS:
            key = f"cmip6|x|{s}|{c}"; f, sg = get(key); e = M.get(key, {})
            if f is None:
                continue
            out(f"enso_imp_comp_z500_{s}_{c}", key, f, sg, s, "comp", "500 hPa height anomaly (gpm)",
                f"CMIP6 · {s} 500 hPa height in {CLASS[c]} seasons",
                f"{CLASS[c]} ({THR[c]} same-season Niño-3.4) minus neutral seasons · {e['events']:,} events, {e['models']} models · "
                f"{100 * e['sig_frac']:.0f} % of 20–90°N robust", MOD_T + " Members event-weighted within a model.")
        for c in ("en", "ln"):
            key = f"obs|x|{s}|{c}"; f, sg = get(key); e = M.get(key, {})
            name = f"enso_imp_obs_z500_{s}_{c}"
            if f is None:
                blank(name, key, s, "comp", f"ERA5 · {s} 500 hPa height in {CLASS[c]} seasons", f"{e.get('n', 0)} observed seasons",
                      "Classes with fewer than 8 observed seasons are not tested, so nothing is drawn.", f"Not tested: only {e.get('n', 0)} observed seasons")
                continue
            out(name, key, f, sg, s, "comp", "500 hPa height anomaly (gpm)", f"ERA5 · {s} 500 hPa height in {CLASS[c]} seasons",
                f"{CLASS[c]} ({THR[c]} ONI) minus neutral seasons · {e['n']} seasons, {e['span'][0]}–{e['span'][1]} · "
                f"{100 * e['sig_frac']:.0f} % of 20–90°N significant",
                "Shaded only where significant: Welch t-test of the class against neutral seasons per grid point, Benjamini–Hochberg "
                "FDR 10 % over the map; blank = not significant. Contours: ERA5 seasonal mean height every 100 m.")
        for c, lab in (("asym", "|index| ≥ 0.5 K"), ("asym_strong", "|index| ≥ 1.5 K")):
            key = f"cmip6|x|{s}|{c}"; f, sg = get(key); e = M.get(key, {})
            if f is None:
                continue
            out(f"enso_imp_asym_z500_{s}_{c}", key, f, sg, s, "comp", "500 hPa height (gpm)",
                f"CMIP6 · {s} 500 hPa height: what does not flip sign",
                f"El Niño plus La Niña, each rescaled to the same index size ({lab}) · zero for a symmetric response · {e['models']} models",
                MOD_T, note="No significant signal")
        for nm in ("E", "C"):
            lab = "east-based (E)" if nm == "E" else "central-Pacific (C)"
            for src in ("cmip6", "obs"):
                key = f"cmip6|{nm}|{s}" if src == "cmip6" else f"obs|{nm}|{s}"; f, sg = get(key); e = M.get(key, {})
                who = "CMIP6" if src == "cmip6" else "ERA5"
                out(f"enso_imp_ec_z500_{s}_{nm}_{src}", key, f, sg, s, "reg", "gpm per standard deviation",
                    f"{who} · {s} 500 hPa height and the {lab} ENSO index",
                    f"Partial regression on Takahashi's E and C together, per 1 sd of {nm} · "
                    + (f"{e['models']} models" if src == "cmip6" else f"{e['n']} seasons, {e['span'][0]}–{e['span'][1]}")
                    + f" · {100 * e['sig_frac']:.0f} % of 20–90°N " + ("robust" if src == "cmip6" else "significant"),
                    MOD_T if src == "cmip6" else OBS_T, note="No significant signal")
        key = f"cmip6|cmp|{s}|epcp"; f, sg = get(key); e = M.get(key, {})
        out(f"enso_imp_ec_z500_{s}_diff", key, f, sg, s, "comp", "500 hPa height (gpm)",
            f"CMIP6 · {s} 500 hPa height: east-based minus central-Pacific El Niño",
            f"Same-strength events: Niño-3.4 {e['x_a']:+.2f} K (east, E > C) vs {e['x_b']:+.2f} K (central) · {e['n_a']:,} vs "
            f"{e['n_b']:,} events, {e['models']} models · {100 * e['sig_frac']:.0f} % of 20–90°N robust",
            MOD_T + " Events matched in Niño-3.4 bins (0.5–1, 1–1.5, 1.5–2, ≥ 2 K).", note="No significant difference")
        for nm in (("free", "raw") if PDO_VIEWS else ()):
            for src in ("cmip6", "obs"):
                key = f"cmip6|{nm}|{s}" if src == "cmip6" else f"obs|{nm}|{s}"; f, sg = get(key); e = M.get(key, {})
                who = "CMIP6" if src == "cmip6" else "ERA5"
                ttl = (f"{who} · {s} 500 hPa height co-varying with the same-season " + ("ENSO-free PDO" if nm == "free" else "PDO index")
                       + " (mostly the atmosphere driving the SST, not the PDO's effect)")
                sub = (("Partial regression on the ENSO-free PDO, Niño-3.4 alongside, per 1 sd" if nm == "free"
                        else "Simple regression on the PDO index, ENSO left in, per 1 sd") + " · "
                       + (f"{e['models']} models" if src == "cmip6" else f"{e['n']} seasons, {e['span'][0]}–{e['span'][1]}")
                       + f" · {100 * e['sig_frac']:.0f} % of 20–90°N " + ("robust" if src == "cmip6" else "significant"))
                out(f"enso_imp_pdo_z500_{s}_{nm}_{src}", key, f, sg, s, "reg", "gpm per standard deviation", ttl, sub,
                    MOD_T if src == "cmip6" else OBS_T, note="No significant signal")
        for nm, lab in ((("en", "El Niño"), ("ln", "La Niña"), ("int", "El Niño")) if PDO_VIEWS else ()):
            key = f"cmip6|cmp|{s}|{'en_pdo' if nm == 'en' else 'ln_pdo' if nm == 'ln' else 'int'}"; f, sg = get(key); e = M.get(key, {})
            ttl = (f"CMIP6 · {s} 500 hPa height: {lab} with a same-winter +PDO minus −PDO (co-variability)" if nm != "int"
                   else f"CMIP6 · {s} 500 hPa height: El Niño × same-winter PDO interaction")
            sub = ((f"{lab} seasons with the same season's ENSO-free PDO ≥ +0.5 sd minus ≤ −0.5 sd, Niño-3.4 matched ({e['x_a']:+.2f} vs "
                    f"{e['x_b']:+.2f} K) · {e['n_a']:,} vs {e['n_b']:,} events" if nm != "int"
                    else "(El Niño, +PDO − −PDO) − (neutral, +PDO − −PDO), per model")
                   + f" · {e['models']} models · {100 * e['sig_frac']:.0f} % of 20–90°N robust")
            out(f"enso_imp_pdoph_z500_{s}_{nm}", key, f, sg, s, "comp", "500 hPa height (gpm)", ttl, sub, MOD_T,
                note="No significant difference")
    for ph in PHASE:
        s = PHSEA[ph]
        for kind in ("super", "nonlin"):
            key = f"cmip6|life|{ph}|{kind}"; f, sg = get(key); e = M.get(key, {})
            if f is None:
                continue
            ttl = (f"CMIP6 · super El Niño, 500 hPa height, {PHASE[ph]}" if kind == "super"
                   else f"CMIP6 · super El Niño beyond a scaled-up El Niño, 500 hPa height, {PHASE[ph]}")
            out(f"enso_imp_life_z500_{ph}_{kind}", key, f, sg, s, "comp", "500 hPa height (gpm)", ttl,
                (f"Winters with DJF Niño-3.4 ≥ +2.0 K minus neutral winters · " if kind == "super" else
                 "Super composite minus the linear regression on the index × the mean super index · ")
                + f"{e['events']:,} events, {e['models']} models · {100 * e['sig_frac']:.0f} % of 20–90°N robust", MOD_T,
                note="No significant signal")
    wave_fig(meta, made, index)


def wave_fig(meta, made, index):
    """Wave-train position: observed vs CMIP6 per season (regression per K) and for the super El Nino composite, and the
    super El Nino Aleutian low against the models' equatorial rain centroid."""
    plt = _plt()
    import textwrap
    pos = meta["position"]; sd = pos["super_DJF"]
    fig = plt.figure(figsize=(12.4, 5.0))
    H = 5.0
    fig.text(0.02, 1 - 0.12 / H, "Where the ENSO wave train sits: CMIP6 against ERA5", fontsize=13, fontweight="bold", va="top")
    fig.text(0.02, 1 - 0.44 / H, "\n".join(textwrap.wrap(
        "500 hPa height regressed on Niño-3.4 per season, and the super El Niño composite; centres are the area-weighted "
        "centroid of the strongest 20 % of the anomaly. Right: each model's super El Niño Aleutian low against where it rains "
        "along the equator.", 150)), fontsize=8.6, color="#3d3a36", va="top", linespacing=1.3)
    hdr = ["", "pattern r", "Aleutian low\nERA5", "Aleutian low, CMIP6\nmedian (10–90 %)", "strength (gpm/K)\nERA5 · CMIP6",
           "Canadian\nridge, ERA5", "Canadian ridge, CMIP6\nmedian (10–90 %)"]
    xs = [0.02, 0.07, 0.13, 0.205, 0.335, 0.44, 0.515]
    y0 = 1 - 1.02 / H
    for x, h in zip(xs, hdr):
        fig.text(x, y0, h, fontsize=7.8, fontweight="bold", va="top", linespacing=1.1)
    rng = lambda d: "–" if not d["models_p10_50_90"] else (f"{lonlab(d['models_p10_50_90'][1])} "
                                                           f"({lonlab(d['models_p10_50_90'][0])}–{lonlab(d['models_p10_50_90'][2])})")
    for i, s_ in enumerate(SEASONS):
        p_ = pos[s_]; y = y0 - (0.52 + 0.32 * i) / H
        al, cr = p_["aleutian_low"], p_["canadian_ridge"]
        vals = [s_, f"{p_['pattern_r']:+.2f}", lonlab(al["obs"]) if al["obs"] else "–", rng(al),
                f"{al['obs_extreme']:+.0f} · {al['cmip6_extreme']:+.0f}", lonlab(cr["obs"]) if cr["obs"] else "–", rng(cr)]
        for x, t in zip(xs, vals):
            fig.text(x, y, t, fontsize=8.3, va="top")
    y = y0 - (0.52 + 0.32 * 4 + 0.12) / H
    vals = ["super\nDJF", f"{sd['pattern_r_obs_vs_cmip6']:+.2f}", lonlab(sd["obs_aleutian_low"]) if sd.get("obs_aleutian_low") else "–",
            f"{lonlab(sd['cmip6_aleutian_low_mean_map'])} (mean map)", "composite", lonlab(sd["obs_canadian_ridge"]) if sd.get("obs_canadian_ridge") else "–",
            f"{lonlab(sd['cmip6_canadian_ridge_mean_map'])} (mean map)"]
    for x, t in zip(xs, vals):
        fig.text(x, y, t, fontsize=8.3, va="top", color="#7a3b12", linespacing=1.1)
    ax = fig.add_axes([0.715, 0.17, 0.265, 0.58])
    rain, alv = np.array(sd["rain_centroid_models"]), np.array(sd["aleutian_low_models"])
    ax.scatter(rain, alv, s=26, c="#7f95ad", edgecolors="white", linewidths=0.5, label="CMIP6 models")
    if sd.get("obs_aleutian_low") is not None:
        ax.scatter([sd["obs_rain_centroid"]], [sd["obs_aleutian_low"]], s=120, marker="*", c="#c0392b", edgecolors="white",
                   linewidths=0.6, label="observed (3 events)", zorder=5)
    k = np.polyfit(rain, alv, 1); xx = np.linspace(rain.min() - 2, max(rain.max(), sd["obs_rain_centroid"]) + 2, 10)
    ax.plot(xx, np.polyval(k, xx), color="#1f4e79", lw=0.9, ls="--")
    ax.set_xlabel("equatorial rain centroid (°E)", fontsize=8.2); ax.set_ylabel("Aleutian low centre (°E)", fontsize=8.2)
    ax.tick_params(labelsize=7.5); ax.grid(lw=0.3, color="#ddd"); ax.legend(fontsize=7.0, frameon=False, loc="upper left")
    sig = "significant" if sd["p_rain_vs_aleutian_low"] < 0.05 else "not significant"
    ax.set_title(f"Super El Niño: r = {sd['r_rain_vs_aleutian_low']:+.2f}, {sd['models']} models\n"
                 f"(p = {sd['p_rain_vs_aleutian_low']:.2f}, {sig})", fontsize=8.4, loc="left")
    fig.text(0.02, 0.025, "\n".join(textwrap.wrap(
        "Aleutian low: minimum, 30–65°N 150°E–130°W; Canadian ridge: maximum, 40–70°N 130–60°W; strength: the extreme of the "
        "map (gpm per K). Pattern r over 20–90°N, cos-weighted, on the unmasked maps. CMIP6: 16 models, 147 members; ERA5 "
        "1959–2026 on CPC's ONI. The positions are descriptive: the ERA5 value is one realisation and the CMIP6 range is the "
        "spread between models, so no test is implied. Super El Niño row: the observed composite is 3 winters (1982/83, "
        "1997/98, 2015/16). Rain centroid: longitude centroid of the positive equatorial (5°S–5°N) DJF rain anomaly, 150°E–90°W; "
        "observed from GPCP. p: two-sided Pearson test across models.", 175)), fontsize=7.0, color=MUTED, va="bottom", linespacing=1.3)
    name = "enso_imp_wave_position"
    fig.savefig(OUT / f"{name}.webp", dpi=110, facecolor="white", pil_kwargs={"quality": 86, "method": 6}); plt.close(fig)
    made.append(name); index["maps"][name] = {"position": pos}


def render_modes(made, index):
    """East-based vs central El Nino (Takahashi E and C), the PDO without ENSO, and El Nino / La Nina by PDO phase, from
    reference/enso_modes_site.{npz,json} (cmip6_enso_modes.py --site)."""
    z = np.load(REF / "enso_modes_site.npz")
    meta = json.loads((REF / "enso_modes_site.json").read_text())
    M = meta["maps"]; lat, lon = z["lat"], z["lon"]

    def get(key):
        return (z[key + "|f"], z[key + "|s"]) if key + "|f" in z else (None, None)

    def rec(name, key, extra=None):
        e = dict(M.get(key, {})); e.update(extra or {}); index["maps"][name] = e; made.append(name)

    per = {"tas": "K per standard deviation", "pr": "% of normal per standard deviation"}
    for v in ("tas", "pr"):
        cmap = "RdBu_r" if v == "tas" else "BrBG"
        dry = np.zeros((len(lat), len(lon)), bool) if v == "pr" else None
        dry_txt = " Hatched: normal under 0.3 mm/day, not scored." if v == "pr" else ""
        for s in SEASONS:
            # E and C, joint (partial) regression
            for nm, lab in (("E", "east-based (E)"), ("C", "central-Pacific (C)")):
                for src in ("cmip6", "obs"):
                    key = f"ec|{v}|{s}|{nm}|{src}"; f, sg = get(key); e = M.get(key, {})
                    if f is None:
                        continue
                    name = f"enso_imp_ec_{v}_{s}_{nm}_{src}"
                    if src == "cmip6":
                        title = f"CMIP6 · {s} {VNAME[v]} and the {lab} ENSO index"
                        sub = (f"Partial regression on Takahashi's E and C together, per 1 sd of {nm} · {e['models']} models, "
                               f"{e['members']} members, 1950–2014 · {100 * e['land_sig']:.0f} % of land robust")
                        foot = (CMIP_TEST + " E and C from each member's own tropical Pacific EOFs (Takahashi et al. 2011), "
                                "fitted jointly so each map is that pattern with the other held fixed." + dry_txt)
                    else:
                        title = f"{OBSNAME[v]} · {s} {VNAME[v]} and the {lab} ENSO index"
                        sub = (f"Partial regression on observed E and C together (ERSST v6), per 1 sd of {nm} · {e['n']} seasons, "
                               f"{e['span'][0]}–{e['span'][1]} · {100 * e['land_sig']:.0f} % of land significant")
                        foot = ("Shaded only where the coefficient is significant: t-test per grid point, Benjamini–Hochberg FDR "
                                "10 % over the map; blank = not significant." + dry_txt)
                    draw_map(OUT / f"{name}.webp", f, sg, lat, lon, LEV[f"reg_{v}"], cmap, per[v], title, sub, foot, dry=dry)
                    rec(name, key)
            # EP minus CP at matched Nino-3.4 strength
            key = f"cmp|{v}|{s}|diff"; f, sg = get(key); e = M.get(key, {})
            name = f"enso_imp_ec_{v}_{s}_diff"
            if f is not None:
                draw_map(OUT / f"{name}.webp", f, sg, lat, lon, LEV[f"comp_{v}"], cmap, UNIT[f"comp_{v}"],
                         f"CMIP6 · {s} {VNAME[v]}: east-based minus central-Pacific El Niño",
                         f"Same-strength events: Niño-3.4 {e['x_a']:+.2f} K (east, E > C) vs {e['x_b']:+.2f} K (central) · "
                         f"{e['n_a']:,} vs {e['n_b']:,} events, {e['models']} models",
                         CMIP_TEST + " Events matched in Niño-3.4 bins (0.5–1, 1–1.5, 1.5–2, ≥ 2 K), each bin weighted by its "
                         "smaller class, so the difference is the type of event, not its size." + dry_txt, dry=dry)
            else:
                draw_map(OUT / f"{name}.webp", np.full((len(lat), len(lon)), np.nan), np.zeros((len(lat), len(lon)), bool),
                         lat, lon, LEV[f"comp_{v}"], cmap, UNIT[f"comp_{v}"],
                         f"CMIP6 · {s} {VNAME[v]}: east-based minus central-Pacific El Niño", "Too few matched events",
                         "Fewer than 5 models have both kinds of event at matched strength.", note_empty="Not tested")
            rec(name, key)
            # PDO: without ENSO (partial on N34 + ENSO-free PDO) and raw
            for nm in (("free", "raw") if PDO_VIEWS else ()):
                for src in ("cmip6", "obs"):
                    key = f"pdo|{v}|{s}|{nm}|{src}"; f, sg = get(key); e = M.get(key, {})
                    if f is None:
                        continue
                    name = f"enso_imp_pdo_{v}_{s}_{nm}_{src}"
                    who = "CMIP6" if src == "cmip6" else OBSNAME[v]
                    if nm == "free":
                        title = f"{who} · {s} {VNAME[v]} co-varying with the same-season ENSO-free PDO"
                        sub = ("Co-variability: mostly the atmosphere driving the sst, not the pdo's effect · partial regression on the ENSO-free PDO, Niño-3.4 alongside, per 1 sd · "
                               + (f"{e['models']} models, {e['members']} members, 1950–2014" if src == "cmip6"
                                  else f"{e['n']} seasons, {e['span'][0]}–{e['span'][1]}")
                               + f" · {100 * e['land_sig']:.0f} % of land " + ("robust" if src == "cmip6" else "significant"))
                    else:
                        title = f"{who} · {s} {VNAME[v]} co-varying with the same-season PDO index"
                        sub = ("Co-variability: mostly the atmosphere driving the sst, not the pdo's effect · simple regression on the PDO index, ENSO left in, per 1 sd · "
                               + (f"{e['models']} models, {e['members']} members, 1950–2014" if src == "cmip6"
                                  else f"NCEI PDO, {e['n']} seasons, {e['span'][0]}–{e['span'][1]}")
                               + f" · {100 * e['land_sig']:.0f} % of land " + ("robust" if src == "cmip6" else "significant"))
                    foot = ((CMIP_TEST if src == "cmip6" else "Shaded only where the coefficient is significant: t-test per grid "
                             "point, Benjamini–Hochberg FDR 10 % over the map; blank = not significant.")
                            + (" ENSO-free PDO = the PDO minus its reddened-ENSO part (Newman et al. 2016): PDO(t) = a·PDO(t−1) + "
                               "b·N34(t), fitted, then driven by Niño-3.4 alone and subtracted." if nm == "free" else "") + dry_txt)
                    draw_map(OUT / f"{name}.webp", f, sg, lat, lon, LEV[f"reg_{v}"], cmap, per[v], title, sub, foot, dry=dry,
                             note_empty="No significant signal")
                    rec(name, key)
            # El Nino / La Nina by ENSO-free PDO phase, matched strength; and the interaction
            for nm, lab, sub0 in () if not PDO_VIEWS else (("en", "El Niño", "Co-variability, not a PDO effect: El Niño seasons with the same season's ENSO-free PDO ≥ +0.5 sd minus ≤ −0.5 sd"),
                                  ("ln", "La Niña", "Co-variability, not a PDO effect: La Niña seasons with the same season's ENSO-free PDO ≥ +0.5 sd minus ≤ −0.5 sd"),
                                  ("int", "El Niño", "El Niño's same-winter PDO difference minus the neutral-winter one (co-variability)")):
                key = f"cmp|{v}|{s}|{nm}"; f, sg = get(key); e = M.get(key, {})
                name = f"enso_imp_pdoph_{v}_{s}_{nm}"
                ttl = (f"CMIP6 · {s} {VNAME[v]}: {lab} with a same-winter +PDO minus −PDO" if nm != "int"
                       else f"CMIP6 · {s} {VNAME[v]}: El Niño × same-winter PDO interaction")
                if f is None:
                    draw_map(OUT / f"{name}.webp", np.full((len(lat), len(lon)), np.nan), np.zeros((len(lat), len(lon)), bool),
                             lat, lon, LEV[f"comp_{v}"], cmap, UNIT[f"comp_{v}"], ttl, sub0, "Fewer than 5 models qualify.",
                             note_empty="Not tested"); rec(name, key); continue
                sub = sub0 + (f" · Niño-3.4 matched ({e['x_a']:+.2f} vs {e['x_b']:+.2f} K) · {e['n_a']:,} vs {e['n_b']:,} events, "
                              f"{e['models']} models" if "n_a" in e else f" · {e['models']} models")
                foot = (CMIP_TEST + " PDO phase from the SAME season's ENSO-free PDO, which that season's atmosphere largely "
                        "builds, so this is co-variability, not the PDO's effect; events matched in |Niño-3.4| bins." + (" Interaction = (El Niño, +PDO − −PDO) − (neutral, +PDO − −PDO), per model."
                                                         if nm == "int" else "") + dry_txt)
                draw_map(OUT / f"{name}.webp", f, sg, lat, lon, LEV[f"comp_{v}"], cmap, UNIT[f"comp_{v}"], ttl, sub, foot, dry=dry,
                         note_empty="No significant difference")
                rec(name, key)
    place_fig(z, meta, made, index)
    if PDO_VIEWS:
        share_fig(meta, made, index)


def place_fig(z, meta, made, index):
    """Where El Nino events sit on Takahashi's E and C: models (density) against the observed events, DJF and the JJA
    before, with 2026."""
    plt = _plt()
    djf, jja = z["djf_events"], z["jja_before"]
    od, oj = meta["obs_djf"], meta["obs_jja"]; pl = meta["placement"]
    fig = plt.figure(figsize=(11.2, 6.3))
    for k, (arr, obs, ttl, key) in enumerate(((djf, od, "December–February: El Niño peaks", "DJF"),
                                              (jja, oj, "June–August before the peak", "JJA"))):
        ax = fig.add_axes([0.06 + k * 0.49, 0.19, 0.41, 0.62])
        from matplotlib.colors import ListedColormap
        light = ListedColormap(plt.get_cmap("Greys")(np.linspace(0.08, 0.45, 256)))
        ax.hexbin(arr[:, 1], arr[:, 0], gridsize=45, extent=(-2, 4, -2, 5.5), cmap=light, mincnt=3, bins="log", linewidths=0)
        sup = (arr[:, 2] >= 2.0) if key == "DJF" else (arr[:, 3] >= 2.0)
        H2, xe, ye = np.histogram2d(arr[sup, 1], arr[sup, 0], bins=[np.linspace(-2, 4, 31), np.linspace(-2, 5.5, 38)])
        from scipy.ndimage import gaussian_filter
        H2 = gaussian_filter(H2, 1.0).T
        lv = np.quantile(H2[H2 > 0], [0.55, 0.8, 0.93]) if (H2 > 0).any() else []
        if len(lv):
            ax.contour(0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1]), H2, levels=np.unique(lv), colors="#c0561a",
                       linewidths=[0.8, 1.1, 1.5][: len(np.unique(lv))])
        ax.plot([], [], color="#c0561a", lw=1.1, label="CMIP6 super El Niños" + (" (DJF ≥ 2 K)" if key == "DJF" else ", the summer before"))
        ax.scatter([], [], marker="h", s=60, c="#bbb", label="all CMIP6 El Niños" + ("" if key == "DJF" else ", the summer before"))
        ax.plot([-2, 4], [-2, 4], color="#6f6b64", lw=0.8, ls="--")
        ax.text(3.9, 3.5, "E = C", color=MUTED, fontsize=8, ha="right", va="top", rotation=0)
        ax.text(-1.85, 5.3, "east-based", color="#333", fontsize=9, va="top"); ax.text(3.9, -1.85, "central-Pacific", color="#333", fontsize=9, ha="right")
        yrs = [y for y in obs if (obs[y]["n34"] >= 0.5 if key == "DJF" else int(y) in (1972, 1982, 1987, 1991, 1994, 1997, 2002, 2004, 2009, 2015, 2023, 2026))]
        hi = {"DJF": {"1983": "1982/83", "1998": "1997/98", "2016": "2015/16", "2010": "2009/10", "2024": "2023/24"},
              "JJA": {"1982": "1982", "1997": "1997", "2015": "2015", "2023": "2023", "2026": "2026"}}[key]
        for y in yrs:
            o = obs[y]
            if y == "2026":
                ax.scatter(o["C"], o["E"], s=170, marker="*", c="#c0392b", edgecolors="white", linewidths=0.8, zorder=6)
                ax.annotate("2026", (o["C"], o["E"]), xytext=(8, 4), textcoords="offset points", fontsize=10, fontweight="bold", color="#c0392b")
            else:
                big = y in hi
                ax.scatter(o["C"], o["E"], s=34 if big else 14, c="#1f4e79" if big else "#7f95ad", edgecolors="white", linewidths=0.5, zorder=5)
                if big:
                    ax.annotate(hi[y], (o["C"], o["E"]), xytext=(6, -3), textcoords="offset points", fontsize=8.6, color="#1f4e79",
                                zorder=7, bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.8))
        ax.set_xlim(-2, 4); ax.set_ylim(-2, 5.5)
        ax.set_xlabel("C, central-Pacific index (sd)", fontsize=9); ax.set_ylabel("E, eastern-Pacific index (sd)", fontsize=9)
        ax.set_title(ttl, fontsize=10.5, fontweight="bold", loc="left")
        ax.tick_params(labelsize=8); ax.grid(lw=0.3, color="#ddd")
        ax.legend(loc="upper right", fontsize=7.6, frameon=True, framealpha=0.9, edgecolor="none")
    fig.text(0.02, 0.975, "Where El Niño events fall on the east-based and central-Pacific indices", fontsize=13, fontweight="bold", va="top")
    fig.text(0.02, 0.928, (f"Grey: {pl['n_djf']:,} CMIP6 El Niño winters (log density) and the summers before them; orange: the "
                           f"{pl['n_super']:,} super El Niños; blue: observed (ERSST v6); star: summer 2026.\n"
                           f"{100 * pl['ep_share_super']:.0f} % of the models' super El Niños are east-based (E > C), "
                           f"{100 * pl['ep_share_strong']:.0f} % of their strong ones and {100 * pl['ep_share_all']:.0f} % of all "
                           "their El Niños."), fontsize=8.8, color="#3d3a36", va="top", linespacing=1.35)
    fig.text(0.02, 0.03, ("Takahashi et al. (2011): E = (PC1 − PC2)/√2 and C = (PC1 + PC2)/√2 from the first two EOFs of tropical "
                          "Pacific SST (10°S–10°N, 140°E–80°W), each dataset its own EOFs, in standard deviations of the monthly index. "
                          "Observed: ERSST v6, linear trend 1950–2025 removed per grid point and extrapolated for 2026. Descriptive: no "
                          "test is implied by the positions."), fontsize=7.2, color=MUTED, va="bottom", wrap=True)
    name = "enso_imp_ec_place"
    fig.savefig(OUT / f"{name}.webp", dpi=110, facecolor="white", pil_kwargs={"quality": 86, "method": 6}); plt.close(fig)
    made.append(name); index["maps"][name] = {"placement": pl}


def share_fig(meta, made, index):
    """How much of the raw PDO's impact is ENSO: per season and variable, robust land share raw vs ENSO-free, their
    pattern correlation and the variance ratio over land."""
    plt = _plt()
    sh = meta["pdo_share"]; q = meta["quality"]; o = meta["obs_ref"]
    rows = [(v, s) for v in ("tas", "pr") for s in SEASONS]
    fig = plt.figure(figsize=(9.6, 1.6 + 0.3 * len(rows) + 0.62))
    H = fig.get_size_inches()[1]
    fig.text(0.02, 1 - 0.14 / H, "How much of what co-varies with the PDO over the Americas is ENSO?", fontsize=12.5, fontweight="bold", va="top")
    fig.text(0.02, 1 - 0.46 / H, "CMIP6 same-season regressions on the raw PDO against the same on the ENSO-free PDO, over land, 16 models "
             "(co-variability: mostly the atmosphere driving the SST, not the PDO's effect)", fontsize=8.6, color="#3d3a36", va="top")
    hdr = ["", "robust land,\nraw PDO", "robust land,\nENSO-free", "robust land,\nwith N34 fitted", "pattern r,\nraw vs free",
           "variance kept\nwithout ENSO", "model range\nof that share"]
    xs = [0.02, 0.2, 0.33, 0.46, 0.6, 0.73, 0.87]
    y0 = 1 - 0.85 / H
    for x, h in zip(xs, hdr):
        fig.text(x, y0, h, fontsize=8, fontweight="bold", va="top", linespacing=1.1)
    for i, (v, s) in enumerate(rows):
        d = sh[f"{v}|{s}"]; y = y0 - (0.55 + 0.3 * i) / H
        vals = [f"{s} {VNAME[v]}", f"{100 * d['raw_land_robust']:.0f} %", f"{100 * d['free_land_robust']:.0f} %",
                f"{100 * d['partial_land_robust']:.0f} %", f"{d['pattern_r']:+.2f}", f"{100 * d['var_share']:.0f} %",
                f"{100 * d['model_var_share'][1]:.0f}–{100 * d['model_var_share'][2]:.0f} %"]
        for x, t in zip(xs, vals):
            fig.text(x, y, t, fontsize=8.6, va="top")
    fig.text(0.02, 0.02,
             (f"ENSO-free PDO = PDO minus its reddened-ENSO part, PDO(t) = a·PDO(t−1) + b·N34(t) (Newman et al. 2016); across the "
              f"429 members a = {q['a'][1]:.2f} ({q['a'][0]:.2f}–{q['a'][2]:.2f}, 10–90 %), and the reddened ENSO explains "
              f"{100 * q['r2'][1]:.0f} % of the monthly PDO variance ({100 * q['r2'][0]:.0f}–{100 * q['r2'][2]:.0f} %); observed "
              f"(NCEI PDO, ERSST v6 Niño-3.4, 1950–2026) a = {o['a']:.2f}, {100 * o['r2']:.0f} %. The lag 0–12 month regression "
              f"of the PDO on Niño-3.4 gives an ENSO-free index correlated {q['r_free_lag'][1]:.2f} with this one. 'Robust' = "
              "across-model t-test FDR 10 % and ≥ 80 % sign agreement. 'Variance kept' = the ENSO-free map's land variance as a "
              "share of the raw map's (per 1 sd of each index): the rest was ENSO."), fontsize=7.1, color=MUTED, va="bottom", wrap=True)
    name = "enso_imp_pdo_share"
    fig.savefig(OUT / f"{name}.webp", dpi=110, facecolor="white", pil_kwargs={"quality": 86, "method": 6}); plt.close(fig)
    made.append(name); index["maps"][name] = {"share": sh}


def table_figs(meta, made, index):
    """Per variable and season: CMIP6 regional composites (value only where robust) and the observed El Nino / La Nina
    composites (value only where their own test passes); plus one list per variable of where the observed composite
    sits outside the models' spread."""
    plt = _plt()
    from matplotlib.colors import BoundaryNorm, ListedColormap
    R = meta["regions"]; RC, RO = meta["regions_cmip6"], meta["regions_obs"]
    unit = {"tas": "K", "pr": "%"}
    for v in ("tas", "pr"):
        lv = LEV[f"comp_{v}"]
        cm = ListedColormap(plt.get_cmap("RdBu_r" if v == "tas" else "BrBG")(np.linspace(0.06, 0.94, 256)))
        norm = BoundaryNorm(lv, cm.N, extend="both")
        for s in SEASONS:
            cols = [("CMIP6\n" + CLASS[c].replace("Strong ", "strong\n").replace("Super ", "super\n"), RC.get(f"{v}|{s}|{c}"), "c")
                    for c in CLASS] + [(f"{OBSNAME[v]}\n{CLASS[c]}", RO.get(f"{v}|{s}|{c}"), "o") for c in ("en", "ln")]
            fig = plt.figure(figsize=(9.6, 8.4))
            ax = fig.add_axes([0.0, 0.1, 1.0, 0.78]); ax.axis("off")
            nr, nc = len(R), len(cols)
            x0, cw, rh = 0.22, (1 - 0.23) / nc, 1.0 / (nr + 1.6)
            for j, (lab, d, kind) in enumerate(cols):
                ax.text(x0 + (j + 0.5) * cw, 1 - 0.8 * rh, lab, ha="center", va="center", fontsize=7.8, fontweight="bold",
                        color="#222" if kind == "c" else "#2c4a72", transform=ax.transAxes)
            for i, r in enumerate(R):
                y = 1 - (i + 1.9) * rh
                ax.text(0.01, y, r, ha="left", va="center", fontsize=8.4, transform=ax.transAxes)
                for j, (lab, d, kind) in enumerate(cols):
                    xc = x0 + (j + 0.5) * cw
                    if d is None:
                        txt, ok, val = "–", False, None
                    elif kind == "c":
                        val = d["value"][i]; ok = d["robust"][i]
                        txt = f"{val:+.1f}" if ok else "n.s."
                    elif not d.get("tested"):
                        txt, ok, val = f"n={d['n']}", False, None
                    else:
                        val = d["value"][i]; ok = d["sig"][i]
                        txt = f"{val:+.1f}" if ok else "n.s."
                    if ok and val is not None:
                        colr = cm(norm(val))
                        ax.add_patch(plt.Rectangle((xc - cw / 2 + 0.003, y - rh / 2 + 0.002), cw - 0.006, rh - 0.004,
                                                   color=colr, transform=ax.transAxes, lw=0))
                        lum = 0.299 * colr[0] + 0.587 * colr[1] + 0.114 * colr[2]
                        ax.text(xc, y, txt, ha="center", va="center", fontsize=8.2, transform=ax.transAxes,
                                color="white" if lum < 0.45 else "#111", fontweight="bold")
                    else:
                        ax.text(xc, y, txt, ha="center", va="center", fontsize=7.6, transform=ax.transAxes, color=MUTED)
            fig.text(0.02, 0.975, f"{s} {VNAME[v]} by region — CMIP6 and observed ({unit[v]}"
                     f"{' of normal' if v == 'pr' else ''}, event class minus neutral)", fontsize=12.2, fontweight="bold", va="top")
            fig.text(0.02, 0.935, f"Land-weighted regional means · {meta['models']} CMIP6 models, {meta['members']} members, "
                     f"1950–2014; observed {OBSNAME[v]} on CPC's ONI", fontsize=8.6, color="#3d3a36", va="top")
            fig.text(0.02, 0.085,
                     "CMIP6 value shown only where robust: across-model t-test, Benjamini–Hochberg FDR 10 % within the row of 20 "
                     "regions, and ≥ 80 % of models agree on the sign. Observed value shown only where a Welch t-test against "
                     "neutral seasons passes FDR 10 % within its column; classes with fewer than 8 observed seasons are not "
                     "tested (n = number of seasons). n.s. = not significant.", fontsize=7.2, color=MUTED, va="top", wrap=True)
            name = f"enso_imp_regions_{v}_{s}"
            fig.savefig(OUT / f"{name}.webp", dpi=110, facecolor="white", pil_kwargs={"quality": 86, "method": 6})
            plt.close(fig)
            made.append(name); index["maps"][name] = {"table": True}
        # observed outside the models, every season and class
        rows = []
        for s in SEASONS:
            for c in CLASS:
                d = RC.get(f"{v}|{s}|{c}")
                if d is None:
                    continue
                for i, r in enumerate(R):
                    if d["obs_outside"][i] and d["obs_value"][i] is not None:
                        rows.append((s, CLASS[c], r, d["obs_value"][i], d["value"][i], d["obs_pctile"][i], d["obs_p"][i],
                                     d["obs_n"][i], d["obs_outside_tablewide"][i]))
        if not rows:                                   # nothing significant: not published (significance rule)
            print(f"  observed-vs-models table for {v}: no region passes - not drawn")
            (OUT / f"enso_imp_ovmtab_{v}.webp").unlink(missing_ok=True)
            continue
        fig = plt.figure(figsize=(9.6, 1.25 + 0.27 * len(rows) + 0.62))
        H = fig.get_size_inches()[1]
        fig.text(0.02, 1 - 0.14 / H, f"Where the observed {VNAME[v]} response sits outside the CMIP6 spread",
                 fontsize=12.2, fontweight="bold", va="top")
        fig.text(0.02, 1 - 0.46 / H, f"{OBSNAME[v]} composites against n-event composites drawn from the models, every season "
                 "and class", fontsize=8.6, color="#3d3a36", va="top")
        hdr = ["season", "class", "region", "observed", "CMIP6 mean", "percentile", "p", "events"]
        xs = [0.02, 0.1, 0.3, 0.55, 0.65, 0.75, 0.86, 0.94]
        y = 1 - 0.95 / H
        for x, h in zip(xs, hdr):
            fig.text(x, y, h, fontsize=8.2, fontweight="bold", va="top")
        if not rows:
            fig.text(0.02, y - 0.3 / H, "No region passes.", fontsize=9, va="top", color="#333")
        for k, (s, cl, r, ov, mv, pc, p, n, strict) in enumerate(rows):
            yy = y - (k + 1) * 0.27 / H
            u = " K" if v == "tas" else " %"
            vals = [s, cl, r, f"{ov:+.1f}{u}", f"{mv:+.1f}{u}", f"{pc:.1f}", f"{p:.4f}", f"{n}"]
            for x, t in zip(xs, vals):
                fig.text(x, yy, t, fontsize=8.2, va="top", fontweight="bold" if strict else "normal")
        fig.text(0.02, 0.02,
                 "Test: the observed n-event composite placed among 20,000 n-event composites drawn from the models (a model at "
                 "random, then n of its events); two-sided p, Benjamini–Hochberg FDR 10 % within each season × class row of 20 "
                 "regions. Rows in bold also pass an FDR over the whole table; " +
                 ("none does." if not any(r[-1] for r in rows) else "the rest do not.") +
                 " 'CMIP6 mean' = the multi-model mean composite." + (" Precipitation in % of normal." if v == "pr" else ""),
                 fontsize=7.2, color=MUTED, va="bottom", wrap=True)
        name = f"enso_imp_ovmtab_{v}"
        fig.savefig(OUT / f"{name}.webp", dpi=110, facecolor="white", pil_kwargs={"quality": 86, "method": 6})
        plt.close(fig)
        made.append(name); index["maps"][name] = {"table": True, "rows": len(rows)}


if __name__ == "__main__":
    raise SystemExit(main())
