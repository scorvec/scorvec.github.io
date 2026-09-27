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
    if (REF / "enso_modes_site.npz").exists():
        render_modes(made, index)
    import datetime as dt
    index["made"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    index["models"], index["members"] = nmod, nmem
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    (OUT / "data" / "enso_impacts.json").write_text(json.dumps(index, separators=(",", ":")))
    print(f"{len(made)} figures -> {OUT}/enso_imp_*.webp")
    return 0


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
            for nm in ("free", "raw"):
                for src in ("cmip6", "obs"):
                    key = f"pdo|{v}|{s}|{nm}|{src}"; f, sg = get(key); e = M.get(key, {})
                    if f is None:
                        continue
                    name = f"enso_imp_pdo_{v}_{s}_{nm}_{src}"
                    who = "CMIP6" if src == "cmip6" else OBSNAME[v]
                    if nm == "free":
                        title = f"{who} · {s} {VNAME[v]} and the PDO without ENSO"
                        sub = ("Partial regression on the ENSO-free PDO with same-season Niño-3.4 fitted alongside, per 1 sd · "
                               + (f"{e['models']} models, {e['members']} members, 1950–2014" if src == "cmip6"
                                  else f"{e['n']} seasons, {e['span'][0]}–{e['span'][1]}")
                               + f" · {100 * e['land_sig']:.0f} % of land " + ("robust" if src == "cmip6" else "significant"))
                    else:
                        title = f"{who} · {s} {VNAME[v]} and the raw PDO index"
                        sub = ("Simple regression on the PDO index, ENSO left in, per 1 sd · "
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
            for nm, lab, sub0 in (("en", "El Niño", "El Niño seasons with the ENSO-free PDO ≥ +0.5 sd minus those ≤ −0.5 sd"),
                                  ("ln", "La Niña", "La Niña seasons with the ENSO-free PDO ≥ +0.5 sd minus those ≤ −0.5 sd"),
                                  ("int", "El Niño", "What a +PDO adds to El Niño beyond its own neutral-year effect")):
                key = f"cmp|{v}|{s}|{nm}"; f, sg = get(key); e = M.get(key, {})
                name = f"enso_imp_pdoph_{v}_{s}_{nm}"
                ttl = (f"CMIP6 · {s} {VNAME[v]}: {lab} under +PDO minus −PDO" if nm != "int"
                       else f"CMIP6 · {s} {VNAME[v]}: does a +PDO amplify El Niño?")
                if f is None:
                    draw_map(OUT / f"{name}.webp", np.full((len(lat), len(lon)), np.nan), np.zeros((len(lat), len(lon)), bool),
                             lat, lon, LEV[f"comp_{v}"], cmap, UNIT[f"comp_{v}"], ttl, sub0, "Fewer than 5 models qualify.",
                             note_empty="Not tested"); rec(name, key); continue
                sub = sub0 + (f" · Niño-3.4 matched ({e['x_a']:+.2f} vs {e['x_b']:+.2f} K) · {e['n_a']:,} vs {e['n_b']:,} events, "
                              f"{e['models']} models" if "n_a" in e else f" · {e['models']} models")
                foot = (CMIP_TEST + " PDO phase from the ENSO-free PDO, so El Niño itself does not decide the phase; events "
                        "matched in |Niño-3.4| bins." + (" Interaction = (El Niño, +PDO − −PDO) − (neutral, +PDO − −PDO), per model."
                                                         if nm == "int" else "") + dry_txt)
                draw_map(OUT / f"{name}.webp", f, sg, lat, lon, LEV[f"comp_{v}"], cmap, UNIT[f"comp_{v}"], ttl, sub, foot, dry=dry,
                         note_empty="No significant difference")
                rec(name, key)
    place_fig(z, meta, made, index)
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
    fig.text(0.02, 1 - 0.14 / H, "How much of the PDO's impact over the Americas is ENSO?", fontsize=12.5, fontweight="bold", va="top")
    fig.text(0.02, 1 - 0.46 / H, "CMIP6 regressions on the raw PDO against the same on the ENSO-free PDO, over land, 16 models",
             fontsize=8.6, color="#3d3a36", va="top")
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
