#!/usr/bin/env python3
"""Render the C3S multi-model seasonal anomaly maps, one model at a time plus the multi-model mean, each shaded
only where its anomaly is significant.

Reads c3s_maps.py's tables: per issue the ensemble-mean anomaly (C3S postprocessed members, each against its own
centre's 1993-2016 hindcast) and the members' spread; per start month the hindcast's single-member spread. A cell is
shaded where the ensemble mean is further from the hindcast mean than sampling noise allows -- a Welch t-test of the
members against the hindcast, one member's spread over its 24 years (n_eff = the years, as the CanSIPS maps on the
same page), with a Benjamini-Hochberg false-discovery rate of 0.10 over the map (seas5_build.shift_signal). The
multi-model mean is tested the same way, its variance the sum of every system's terms over K². A system or field
whose hindcast spread is not in yet is not drawn.

Drawn in the shared outlook style (mapstyle.py), so they sit beside the SEAS5 and SFS maps without looking like a
different site. Outputs assets/sst/c3s/c3s_{model}_{field}_{period}.webp and assets/sst/data/c3s_maps.json, which
names every model, field and period the page can offer.

    SST_SITE_ROOT=. python scripts/sst/c3s_maps_render.py --issue 202610
"""

from __future__ import annotations

import argparse
import time
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mapstyle
from c3s_maps import hindcast_spread, issue_anom
from c3s_nino34 import MODELS
from seas5_build import SHIFT_TEST_NOTE, shift_signal, welch_terms

SITE = Path(os.environ.get("SST_SITE_ROOT", HERE.parents[1]))
OUT = SITE / "assets" / "sst" / "c3s"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# field → (label, units, colour map, vmax, map kind)
SPEC = {
    "t2m": ("2 m temperature", "°C", "RdBu_r", 3.0, "atm"),
    "tp": ("Precipitation", "mm/day", "BrBG", 3.0, "atm"),
    "mslp": ("Mean sea-level pressure", "hPa", "RdBu_r", 6.0, "atm"),
    "sst": ("Sea-surface temperature", "°C", "RdBu_r", 2.0, "sst"),
}
# lead groups: month 1 is the start month itself, so the seasons run from lead 2.
# The labels are the calendar months themselves — "lead month 1" tells a reader nothing
# about when the map applies, and the lead number is kept only in the subtitle.
PERIODS = {"m1": [0], "s1": [1, 2, 3], "s2": [4, 5]}


def period_label(idx: list[int], y: int, mo: int) -> str:
    """'Sep 2026', 'Oct–Dec 2026', 'Nov 2026–Jan 2027' for a group of lead indices."""

    def ym(i: int) -> tuple[int, int]:
        m = mo + i
        return y + (m - 1) // 12, (m - 1) % 12 + 1

    (ya, ma), (yb, mb) = ym(idx[0]), ym(idx[-1])
    if idx[0] == idx[-1]:
        return f"{MONTHS[ma - 1]} {ya}"
    if ya == yb:
        return f"{MONTHS[ma - 1]}–{MONTHS[mb - 1]} {ya}"
    return f"{MONTHS[ma - 1]} {ya}–{MONTHS[mb - 1]} {yb}"


def lead_note(idx: list[int]) -> str:
    return f"lead month {idx[0] + 1}" if len(idx) == 1 else f"lead months {idx[0] + 1}–{idx[-1] + 1}"


def month_days(y: int, mo: int, idx: list[int]) -> np.ndarray:
    """Days in the valid month of each lead index (0 = the issue month): the weights of a multi-month mean."""
    import calendar

    return np.array([calendar.monthrange(y + (mo + i - 1) // 12, (mo + i - 1) % 12 + 1)[1] for i in idx], float)


def on_grid(a: np.ndarray, lat, lon, lat2, lon2) -> np.ndarray:
    """a [lat, lon] bilinearly onto (lat2, lon2), wrapping in longitude; NaN stays NaN. Every CDS system is served on
    the same cell-centred 1° grid; this is for a field that is not (the SEAS5 page's pole-inclusive copy of ECMWF)."""
    lat, lon, lat2, lon2 = (np.asarray(v, float) for v in (lat, lon, lat2, lon2))
    if a.shape == (lat2.size, lon2.size) and np.allclose(lat, lat2) and np.allclose(lon, lon2):
        return a
    from scipy.interpolate import RegularGridInterpolator

    lo = np.mod(lon, 360.0)
    o = np.argsort(lo)
    lo, a = lo[o], a[:, o]
    if lat[0] > lat[-1]:
        lat, a = lat[::-1], a[::-1]
    lo_e = np.concatenate([lo[-1:] - 360.0, lo, lo[:1] + 360.0])
    a_e = np.concatenate([a[:, -1:], a, a[:, :1]], axis=1)
    f = RegularGridInterpolator((lat, lo_e), a_e, bounds_error=False, fill_value=None)
    la, lq = np.meshgrid(lat2, np.mod(lon2, 360.0), indexing="ij")
    return f((la, lq)).astype(np.float32)


def draw(v, lat, lon, key: str, title: str, sub: str, out: Path):
    """One map; NaN cells (not significant, or no data) stay blank."""
    label, units, cmap, vmax, kind = SPEC[key]
    fig, ax, H, pc = mapstyle.open_map(kind=kind)
    lv = mapstyle.symmetric_levels(vmax)
    if lon.max() > 180:  # 0-360 -> -180..180 for plotting
        order = np.argsort(((lon + 180) % 360) - 180)
        lon = ((lon + 180) % 360 - 180)[order]
        v = v[:, order]
    # Close the antimeridian: without a cyclic column contourf leaves a white seam down the
    # dateline, which on a Pacific-centred map runs straight through the subject.
    from cartopy.util import add_cyclic_point

    v, lon = add_cyclic_point(v, coord=lon)
    out.parent.mkdir(parents=True, exist_ok=True)
    if key == "sst":
        # the site's shared SST palette and ladder (white +-0.2, 0.2 degC classes; same colours as SEAS5, SFS and the
        # OISST loops) instead of thirds of a degree to +-2. The axes stay white, land in the SST land colour: a blank
        # ocean cell is one without a significant anomaly
        sty = mapstyle.sst_anom_style(5.0)
        m = ax.contourf(
            lon,
            lat,
            np.ma.masked_invalid(v),
            levels=sty["levels"],
            cmap=sty["cmap"],
            norm=sty["norm"],
            extend="both",
            transform=pc,
        )
        mapstyle.features(ax, states=False, land_color=mapstyle.SST_LAND)
        mapstyle.heading(fig, H, title, sub, wrap=205)
        cb = mapstyle.colorbar(fig, H, m, f"{label} anomaly ({units})")
        mapstyle.sst_colorbar(cb, sty, f"{label} anomaly ({units})", size=8.5, tick_size=7)
        mapstyle.save_webp_palette(fig, out, dpi=125)
        return out
    m = ax.contourf(lon, lat, np.ma.masked_invalid(v), levels=lv, cmap=cmap, extend="both", transform=pc)
    mapstyle.features(ax)
    mapstyle.heading(fig, H, title, sub, wrap=205)
    mapstyle.colorbar(fig, H, m, f"{label} anomaly ({units})", levels=lv)
    mapstyle.save(fig, out)
    return out


def test_note(frac: float) -> str:
    return f"{SHIFT_TEST_NOTE}: {frac:.0%} of the map; blank = no significant anomaly"


def mmm_signal(items: list[dict]):
    """Equal-weight mean of the systems' period fields and its test: every system's forecast and hindcast terms, each
    divided by K² where K systems have the cell (welch_terms)."""
    f = np.stack([it["field"] for it in items])
    have = np.isfinite(f)
    k = have.sum(0).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        field = np.where(k > 0, np.nansum(f, axis=0) / k, np.nan)
        k2 = np.where(k > 0, k * k, np.nan)
        terms = []
        for it, h in zip(items, have):
            for v, dof in welch_terms(it["sdf"], it["n"], it["sdh"], it["ny"]):
                terms.append((np.where(h, v, 0.0) / k2, dof))
    dom = np.isfinite(field)
    return field, shift_signal(field, terms, dom), dom


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--issue", default=time.strftime("%Y%m", time.gmtime()))
    ap.add_argument("--fields", default="t2m,tp,mslp,sst")
    a = ap.parse_args()
    issue = a.issue
    y, mo = int(issue[:4]), int(issue[4:6])
    keys = [k for k in a.fields.split(",") if k in SPEC]
    models, made, drawn = [], 0, set()
    stack: dict[tuple[str, str], list] = {}

    for entry in MODELS:
        centre, system, label = entry[0], str(entry[1]), entry[2]
        mid = f"{centre}_{system}"
        have, sig_share = [], {}
        for key in keys:
            fc = issue_anom(centre, system, key, issue)
            if fc is None:
                continue
            hc = hindcast_spread(centre, system, key, issue)
            if hc is None:
                print(f"  {label} {key}: no hindcast spread for start month {issue[4:6]} yet; untested, not drawn")
                continue
            lat, lon = fc["lat"], fc["lon"]
            n, ny = int(fc["meta"]["nmem"]), int(hc["meta"]["years"])
            for pid, want in PERIODS.items():
                idx = [i for i in want if i < fc["mean"].shape[0]]
                if not idx or f"sd_{pid}" not in fc or f"sd_{pid}" not in hc:
                    continue
                pname = f"{period_label(idx, y, mo)} ({lead_note(idx)})"
                field = np.average(
                    fc["mean"][idx], axis=0, weights=month_days(y, mo, idx)
                )  # a time mean: days per month
                sdh = on_grid(hc[f"sd_{pid}"], hc["lat"], hc["lon"], lat, lon)
                dom = np.isfinite(field)
                sig = shift_signal(field, welch_terms(fc[f"sd_{pid}"], n, sdh, ny), dom)
                frac = float(sig[dom].mean()) if dom.any() else 0.0
                sub = (
                    f"{label} · {issue[:4]}-{issue[4:6]} issue · {pname} · ensemble mean of {n} members' anomalies "
                    f"against the model's own 1993–2016 hindcast (C3S postprocessed). {test_note(frac)}."
                )
                out = OUT / f"c3s_{mid}_{key}_{pid}.webp"
                draw(np.where(sig, field, np.nan), lat, lon, key, f"{SPEC[key][0]} anomaly — {label}", sub, out)
                made += 1
                drawn.add(out.name)
                sig_share.setdefault(key, {})[pid] = round(frac, 3)
                stack.setdefault((key, pid), []).append(
                    dict(field=field, sdf=fc[f"sd_{pid}"], n=n, sdh=sdh, ny=ny, lat=lat, lon=lon, idx=idx)
                )
            if key in sig_share:
                have.append(key)
        if have:
            models.append({"id": mid, "label": label, "fields": have, "sig": sig_share})
            print(f"  {label}: {', '.join(have)}", flush=True)

    # multi-model mean: every model that supplied the field on the common grid, equally weighted, tested as one
    mmm_share = {}
    for (key, pid), every in stack.items():
        g0 = every[0]
        items = [
            it
            for it in every
            if it["field"].shape == g0["field"].shape
            and np.allclose(it["lat"], g0["lat"])
            and np.allclose(it["lon"], g0["lon"])
        ]
        if len(items) < 2:
            continue
        field, sig, dom = mmm_signal(items)
        frac = float(sig[dom].mean()) if dom.any() else 0.0
        idx = items[0]["idx"]
        pname = f"{period_label(idx, y, mo)} ({lead_note(idx)})"
        sub = (
            f"Mean of {len(items)} C3S systems, equally weighted · {issue[:4]}-{issue[4:6]} issue · {pname} "
            f"· each model's ensemble-mean anomaly against its own 1993–2016 hindcast; tested with every system's "
            f"member and hindcast spread. {test_note(frac)}."
        )
        out = OUT / f"c3s_mmm_{key}_{pid}.webp"
        draw(
            np.where(sig, field, np.nan),
            items[0]["lat"],
            items[0]["lon"],
            key,
            f"{SPEC[key][0]} anomaly — multi-model mean",
            sub,
            out,
        )
        made += 1
        drawn.add(out.name)
        mmm_share.setdefault(key, {})[pid] = round(frac, 3)
    if mmm_share:
        models.insert(0, {"id": "mmm", "label": "Multi-model mean", "fields": sorted(mmm_share), "sig": mmm_share})

    # an anomaly map this issue could not test leaves the site with the issue it belonged to
    for mid in ["mmm", *(f"{e[0]}_{e[1]}" for e in MODELS)]:
        for key in keys:
            for pid in PERIODS:
                f = OUT / f"c3s_{mid}_{key}_{pid}.webp"
                if f.name not in drawn and f.exists():
                    f.unlink()

    doc = {
        "issue": f"{issue[:4]}-{issue[4:6]}",
        "issue_label": f"{MONTHS[mo - 1]} {y}",
        "source": "C3S seasonal-postprocessed-single-levels members (anomaly vs each model's 1993-2016 hindcast); "
        "hindcast spread from seasonal-monthly-single-levels",
        "test": "Welch t-test of the members against the hindcast's single-member spread (n_eff = hindcast years), "
        "Benjamini-Hochberg FDR q = 0.10 over each map; sig = share of the map's cells shaded",
        "models": models,
        "fields": {k: {"label": SPEC[k][0], "units": SPEC[k][1]} for k in keys},
        "periods": {p: period_label(PERIODS[p], y, mo) for p in PERIODS},
        "period_leads": {p: lead_note(PERIODS[p]) for p in PERIODS},
    }
    (SITE / "assets" / "sst" / "data").mkdir(parents=True, exist_ok=True)
    doc_path = SITE / "assets" / "sst" / "data" / "c3s_maps.json"
    # the tercile block is c3s_terciles_render's: keep it until that step rewrites it (the two run in
    # separate, resumable steps of c3s.yml, and a map re-render must not drop the terciles)
    try:
        old = json.loads(doc_path.read_text())
        if "terciles" in old:
            doc["terciles"] = old["terciles"]
    except Exception:  # noqa: BLE001
        pass
    doc_path.write_text(json.dumps(doc, separators=(",", ":")))
    print(f"wrote {made} maps + c3s_maps.json ({len(models)} models)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
