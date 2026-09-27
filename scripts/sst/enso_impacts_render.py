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
    import datetime as dt
    index["made"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    index["models"], index["members"] = nmod, nmem
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    (OUT / "data" / "enso_impacts.json").write_text(json.dumps(index, separators=(",", ":")))
    print(f"{len(made)} figures -> {OUT}/enso_imp_*.webp")
    return 0


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
