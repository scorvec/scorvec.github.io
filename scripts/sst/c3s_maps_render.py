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

A change map is this issue's ensemble-mean anomaly minus the previous issue's for the same valid months (one lead
further out there, so the first month and the first season have one, the second season does not). Its test has three
terms: this issue's members, the previous issue's, and the hindcast's -- the year-to-year spread of the difference
between the two start months' hindcast means (c3s_maps.change_spread), since each issue is an anomaly against its own.
The multi-model change is the mean over the systems in both issues. Either multi-model map is drawn once three
quarters of the issue's systems with the field are in it (MMM_QUORUM), so a mean of the first two systems to be tested
never stands in for it.

Drawn in the shared outlook style (mapstyle.py), so they sit beside the SEAS5 and SFS maps without looking like a
different site. Outputs assets/sst/c3s/c3s_{model}_{field}_{period}.webp, c3s_{model}_{field}_chg_{period}.webp and
assets/sst/data/c3s_maps.json, which names every model, field and period the page can offer.

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
from c3s_maps import change_spread, hindcast_spread, issue_anom, prev_issue
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
# the same valid months in the issue before, one lead further out: what a change map compares with, where the six
# leads reach
NEXT = {f"{pid}_next": [i + 1 for i in ix] for pid, ix in PERIODS.items()}
# a change is drawn on half the anomaly's range
CHANGE_SCALE = 0.5
# share of the issue's systems with a field that a multi-model map of it needs
MMM_QUORUM = 0.75


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


def draw(v, lat, lon, key: str, title: str, sub: str, out: Path, change: bool = False):
    """One map of an anomaly, or of its change from the previous issue; NaN cells (not significant, or no data) stay
    blank."""
    label, units, cmap, vmax, kind = SPEC[key]
    fig, ax, H, pc = mapstyle.open_map(kind=kind)
    lv = mapstyle.symmetric_levels(vmax * (CHANGE_SCALE if change else 1.0))
    bar = f"{label} anomaly, change from the previous issue ({units})" if change else f"{label} anomaly ({units})"
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
        # OISST loops) instead of thirds of a degree to +-2, and the shared change ladder for a change. The axes stay
        # white, land in the SST land colour: a blank ocean cell is one without a significant anomaly
        sty = mapstyle.sst_change_style() if change else mapstyle.sst_anom_style(5.0)
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
        cb = mapstyle.colorbar(fig, H, m, bar)
        mapstyle.sst_colorbar(cb, sty, bar, size=8.5, tick_size=7)
        mapstyle.save_webp_palette(fig, out, dpi=125)
        return out
    m = ax.contourf(lon, lat, np.ma.masked_invalid(v), levels=lv, cmap=cmap, extend="both", transform=pc)
    mapstyle.features(ax)
    mapstyle.heading(fig, H, title, sub, wrap=205)
    cb = mapstyle.colorbar(fig, H, m, bar, levels=lv)
    cb.set_ticks(
        lv, labels=[f"{x:g}".replace("-", "−") for x in lv]
    )  # every class edge, so the key reads the same both sides of zero
    mapstyle.save(fig, out)
    return out


def test_note(frac: float, what: str = "anomaly") -> str:
    return f"{SHIFT_TEST_NOTE}: {frac:.0%} of the map; blank = no significant {what}"


def share(sig: np.ndarray, dom: np.ndarray) -> float:
    """Share of the map's cells that are shaded."""
    return float(sig[dom].mean()) if dom.any() else 0.0


def mmm_item(items: list[dict]) -> dict:
    """Equal-weight mean of the systems' period fields with its sampling terms: every system's (`terms`, as
    shift_signal takes them), each divided by K² where K systems have the cell."""
    f = np.stack([it["field"] for it in items])
    have = np.isfinite(f)
    k = have.sum(0).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        field = np.where(k > 0, np.nansum(f, axis=0) / k, np.nan)
        k2 = np.where(k > 0, k * k, np.nan)
        terms = [(np.where(h, v, 0.0) / k2, dof) for it, h in zip(items, have) for v, dof in it["terms"]]
    return dict(field=field, terms=terms, lat=items[0]["lat"], lon=items[0]["lon"], idx=items[0]["idx"])


def same_grid(every: list[dict]) -> list[dict]:
    """The items on the first one's grid (every CDS system is; see on_grid)."""
    g0 = every[0]
    return [
        it
        for it in every
        if it["field"].shape == g0["field"].shape
        and np.allclose(it["lat"], g0["lat"])
        and np.allclose(it["lon"], g0["lon"])
    ]


def change_item(fc: dict, prev: dict | None, chg: dict | None, pid: str, idx: list[int], field, y: int, mo: int):
    """This issue's period field minus the previous issue's for the same valid months, with its three sampling terms
    (this issue's members, the previous issue's, the paired hindcast means); None where any of them is missing."""
    nxt = [i + 1 for i in idx]
    if prev is None or chg is None or idx != PERIODS[pid] or nxt[-1] >= prev["mean"].shape[0]:
        return None
    if f"sd_{pid}_next" not in prev or f"sdd_{pid}" not in chg:
        return None
    lat, lon = fc["lat"], fc["lon"]
    old = np.average(prev["mean"][nxt], axis=0, weights=month_days(y, mo, idx))
    n, n_old, ny = int(fc["meta"]["nmem"]), int(prev["meta"]["nmem"]), int(chg["meta"]["years"])
    sd_old = on_grid(prev[f"sd_{pid}_next"], prev["lat"], prev["lon"], lat, lon)
    sdd = on_grid(chg[f"sdd_{pid}"], chg["lat"], chg["lon"], lat, lon)
    terms = [(np.asarray(sd, float) ** 2 / m, m - 1) for sd, m in ((fc[f"sd_{pid}"], n), (sd_old, n_old), (sdd, ny))]
    return dict(field=field - on_grid(old, prev["lat"], prev["lon"], lat, lon), terms=terms, lat=lat, lon=lon, idx=idx)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--issue", default=time.strftime("%Y%m", time.gmtime()))
    ap.add_argument("--fields", default="t2m,tp,mslp,sst")
    a = ap.parse_args()
    issue = a.issue
    prev = prev_issue(issue)
    y, mo = int(issue[:4]), int(issue[4:6])
    prev_label = f"{MONTHS[int(prev[4:6]) - 1]} {prev[:4]}"
    keys = [k for k in a.fields.split(",") if k in SPEC]
    models, made, drawn = [], 0, set()
    # per (field, period): every system's field and sampling terms, for the multi-model maps; per field: its systems
    stack: dict[tuple[str, str], list] = {}
    cstack: dict[tuple[str, str], list] = {}
    nsys = dict.fromkeys(keys, 0)

    def render(mid: str, key: str, pid: str, it: dict, title: str, sub: str, change: bool) -> float:
        """Test one item and draw it where it passes (sub: the subtitle, {} where the test note goes); -> the share
        of the map shaded."""
        nonlocal made
        dom = np.isfinite(it["field"])
        sig = shift_signal(it["field"], it["terms"], dom)
        frac = share(sig, dom)
        out = OUT / f"c3s_{mid}_{key}_{'chg_' if change else ''}{pid}.webp"
        note = test_note(frac, "change" if change else "anomaly")
        draw(np.where(sig, it["field"], np.nan), it["lat"], it["lon"], key, title, sub.format(note), out, change)
        made += 1
        drawn.add(out.name)
        return round(frac, 3)

    for entry in MODELS:
        centre, system, label = entry[0], str(entry[1]), entry[2]
        mid = f"{centre}_{system}"
        sig_share, chg_share = {}, {}
        for key in keys:
            fc = issue_anom(centre, system, key, issue)
            if fc is None:
                continue
            nsys[key] += 1
            hc = hindcast_spread(centre, system, key, issue)
            if hc is None:
                print(f"  {label} {key}: no hindcast spread for start month {issue[4:6]} yet; anomaly not drawn")
            old, chg = issue_anom(centre, system, key, prev), change_spread(centre, system, key, issue)
            lat, lon = fc["lat"], fc["lon"]
            n = int(fc["meta"]["nmem"])
            for pid, want in PERIODS.items():
                idx = [i for i in want if i < fc["mean"].shape[0]]
                if not idx or f"sd_{pid}" not in fc:
                    continue
                pname = f"{period_label(idx, y, mo)} ({lead_note(idx)})"
                # a time mean: days per month
                field = np.average(fc["mean"][idx], axis=0, weights=month_days(y, mo, idx))
                if hc is not None and f"sd_{pid}" in hc:
                    sdh = on_grid(hc[f"sd_{pid}"], hc["lat"], hc["lon"], lat, lon)
                    terms = welch_terms(fc[f"sd_{pid}"], n, sdh, int(hc["meta"]["years"]))
                    it = dict(field=field, terms=terms, lat=lat, lon=lon, idx=idx)
                    sub = (
                        f"{label} · {issue[:4]}-{issue[4:6]} issue · {pname} · ensemble mean of {n} members' "
                        "anomalies against the model's own 1993–2016 hindcast (C3S postprocessed). {}."
                    )
                    sig_share.setdefault(key, {})[pid] = render(
                        mid, key, pid, it, f"{SPEC[key][0]} anomaly — {label}", sub, False
                    )
                    stack.setdefault((key, pid), []).append(it)
                it = change_item(fc, old, chg, pid, idx, field, y, mo)
                if it is not None:
                    sub = (
                        f"{label} · {issue[:4]}-{issue[4:6]} issue minus {prev[:4]}-{prev[4:6]} issue · {pname} · "
                        f"change in the ensemble-mean anomaly for the same months ({n} members now, "
                        f"{int(old['meta']['nmem'])} before); tested with both issues' member spread and the paired "
                        "hindcast means. {}."
                    )
                    title = f"{SPEC[key][0]} anomaly, change since the {prev_label} issue — {label}"
                    chg_share.setdefault(key, {})[pid] = render(mid, key, pid, it, title, sub, True)
                    cstack.setdefault((key, pid), []).append(it)
        if sig_share or chg_share:
            models.append({"id": mid, "label": label, "fields": list(sig_share), "sig": sig_share, "chg": chg_share})
            print(f"  {label}: {', '.join(sig_share) or 'no anomaly map'}; change: {', '.join(chg_share) or 'none'}")

    # multi-model maps: every model that supplied the field on the common grid, equally weighted, tested as one
    mmm_share, mmm_chg = {}, {}
    for change, stacks, shares in ((False, stack, mmm_share), (True, cstack, mmm_chg)):
        for (key, pid), every in stacks.items():
            items = same_grid(every)
            if len(items) < max(2, int(np.ceil(MMM_QUORUM * nsys[key]))):
                continue
            it = mmm_item(items)
            pname = f"{period_label(it['idx'], y, mo)} ({lead_note(it['idx'])})"
            if change:
                title = f"{SPEC[key][0]} anomaly, change since the {prev_label} issue — multi-model mean"
                sub = (
                    f"Mean of {len(items)} C3S systems, equally weighted · {issue[:4]}-{issue[4:6]} issue minus "
                    f"{prev[:4]}-{prev[4:6]} issue · {pname} · change in each model's ensemble-mean anomaly for the "
                    "same months; tested with every system's member spread in both issues and its paired hindcast "
                    "means. {}."
                )
            else:
                title = f"{SPEC[key][0]} anomaly — multi-model mean"
                sub = (
                    f"Mean of {len(items)} C3S systems, equally weighted · {issue[:4]}-{issue[4:6]} issue · {pname} "
                    "· each model's ensemble-mean anomaly against its own 1993–2016 hindcast; tested with every "
                    "system's member and hindcast spread. {}."
                )
            shares.setdefault(key, {})[pid] = render("mmm", key, pid, it, title, sub, change)
    if mmm_share or mmm_chg:
        models.insert(
            0, {"id": "mmm", "label": "Multi-model mean", "fields": sorted(mmm_share), "sig": mmm_share, "chg": mmm_chg}
        )

    # a map this issue could not test leaves the site with the issue it belonged to
    for mid in ["mmm", *(f"{e[0]}_{e[1]}" for e in MODELS)]:
        for key in keys:
            for pid in PERIODS:
                for f in (OUT / f"c3s_{mid}_{key}_{pid}.webp", OUT / f"c3s_{mid}_{key}_chg_{pid}.webp"):
                    if f.name not in drawn and f.exists():
                        f.unlink()

    doc = {
        "issue": f"{issue[:4]}-{issue[4:6]}",
        "issue_label": f"{MONTHS[mo - 1]} {y}",
        "previous_label": prev_label,
        "source": "C3S seasonal-postprocessed-single-levels members (anomaly vs each model's 1993-2016 hindcast); "
        "hindcast spread and paired hindcast means from seasonal-monthly-single-levels",
        "test": "Welch t-test of the members against the hindcast's single-member spread (n_eff = hindcast years), "
        "Benjamini-Hochberg FDR q = 0.10 over each map; sig = share of the map's cells shaded. chg = the same share "
        "of the change from the previous issue for the same months, tested with both issues' member spread and the "
        "year-to-year spread of the difference between the two start months' hindcast means",
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
