#!/usr/bin/env python3
"""Archived IFS-ENS runs against ERA5: the check of the jet-branch forecasts (jet_branches_fc.py) on a past cold
season, and the groundwork of their skill measurement. Its outputs are workflow artifacts, not site products.

ERA5 for the verifying analyses comes from ARCO-ERA5 (0.25 deg, 37 levels, anonymous GCS), the store WeatherBench 2's
ERA5 was made from: the reference's WeatherBench 2 copy ends in January 2023, before the open-data IFS-ENS archive at
0.25 deg (from 2024-02-28 on the Google Cloud mirror) begins. One ARCO chunk holds one variable's 37 levels at one hour.

  scan     ERA5 at 00 and 12 UTC over a period, identified as the reference identifies it (jet_branches.masks_025)
           -> npz: per-analysis masks, and the superposition events of each region by the reference's thresholds
           (reference/jet_branches_ref.json) and rules (jet_branches.track)
  pick     the scan's largest event in each region -> the 00 UTC runs started LEADS days before its onset
  compare  a run (jet_branches_fc.py run) against the scan at its valid times -> json and a figure:
             maps    the neighbourhood shares against ERA5's neighbourhood occurrence: the Brier score, cos(lat)-weighted
                     over 10-80N, and its skill against the ERA5 1991-2020 normal for the date (the same normal the
                     maps show)
             events  each region's event share by lead against whether ERA5 had an event, with the Brier score of
                     the shares and of the normal rate

    python jet_branches_hindcast.py scan --start 2025-11-15 --end 2026-03-15 --out scan.npz
    python jet_branches_hindcast.py pick --scan scan.npz --leads 3,6 --github-output
    python jet_branches_hindcast.py compare --fc fc.npz --scan scan.npz --clim jet_branches_clim.nc --out-dir out
"""

from __future__ import annotations

import pyproj  # noqa: F401  before eccodes/xarray: eckit's bundled libproj otherwise clashes (double free at exit)

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_jet_branches_ref as B
import jet_branches as JB
import jet_branches_fc as FC

ARCO = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
SHAPE = (3, JB.LAT.size, JB.LON.size)


# ── scan and pick ───────────────────────────────────────────────────────────────────────────────────────────────────
def scan(start, end, out, workers=8):
    times = [d + pd.Timedelta(hours=h) for d in pd.date_range(start, end) for h in B.HOURS]
    times = [t for t in times if JB.season_slot(np.array([t.to_datetime64()]))[0] >= 0]
    t0 = time.time()
    packed = []
    for ts, t, u, v in B.stream(ARCO, times, workers):
        packed.append(np.packbits(JB.masks_025(t, u, v).reshape(3, -1), axis=-1))
        if ts.hour == 0 and ts.day == 1:
            print(f"  {ts:%Y-%m}: {len(packed)} analyses, {time.time() - t0:.0f} s", flush=True)
    masks = np.stack(packed)
    events = scan_events(pd.DatetimeIndex(times), masks)
    print(f"ERA5 {start}..{end}: {len(times)} analyses in {time.time() - t0:.0f} s", flush=True)
    for r, ev in events.items():
        print(f"  {r}: {len(ev)} events")
        for e in sorted(ev, key=lambda e: -e["peak_columns"])[:5]:
            print(f"    {e['onset']} .. {e['end']}: {e['analyses']} analyses, up to {e['peak_columns']} columns")
    np.savez_compressed(out, time=np.array(times, "datetime64[s]"), masks=masks, events=json.dumps(events))


def unpack(packed):
    """(3, packed columns) -> (3, LAT, LON) boolean."""
    return np.unpackbits(packed, axis=-1, count=JB.LAT.size * JB.LON.size).reshape(SHAPE).astype(bool)


def scan_events(times, masks):
    thr = FC.thresholds()
    inside = {r: JB.region_mask(JB.LAT, JB.LON, r) for r in JB.REGIONS}
    found = {r: [JB.groups_of(unpack(p)[2], inside[r]) for p in masks] for r in JB.REGIONS}
    return {
        r: [
            {k: (f"{v:%Y-%m-%d %H:%M}" if isinstance(v, pd.Timestamp) else v) for k, v in e.items()}
            for e in JB.track(times, found[r], thr[r])
        ]
        for r in JB.REGIONS
    }


def load_scan(path):
    with np.load(path) as f:
        return pd.DatetimeIndex(f["time"]), f["masks"], json.loads(str(f["events"]))


def pick(scan_path, leads):
    """The largest event (most superposed columns at one analysis) of each region -> the 00 UTC inits `leads` days
    before the day of its onset, sorted and unique."""
    _, _, events = load_scan(scan_path)
    inits = set()
    for r, ev in events.items():
        if not ev:
            print(f"  {r}: no event in the scan", flush=True)
            continue
        e = max(ev, key=lambda e: (e["peak_columns"], e["analyses"]))
        day = pd.Timestamp(e["onset"]).normalize()
        print(f"  {r}: the largest event, {e['onset']} .. {e['end']}, up to {e['peak_columns']} columns", flush=True)
        inits |= {f"{day - pd.Timedelta(days=d):%Y%m%d}" for d in leads}
    return sorted(inits)


# ── compare ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def brier(p, o, w):
    """cos(lat)-weighted Brier score of probabilities p against 0/1 outcomes o, both (..., LAT, LON)."""
    return float(np.sum(w * (p - o) ** 2) / (np.sum(w) * np.prod(p.shape[:-2])))


def compare(fc_path, scan_path, clim_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    init, steps, masks, cost = FC.load_run(fc_path)
    pr = FC.products(init, steps, masks, clim_path)
    stimes, smasks, _ = load_scan(scan_path)
    at = {t: i for i, t in enumerate(stimes)}
    leads = [i for i, v in enumerate(pr["valid"]) if v in at]
    if not leads:
        raise SystemExit("the scan holds none of the run's valid times")
    w = np.broadcast_to(np.cos(np.deg2rad(JB.LAT))[:, None], (JB.LAT.size, JB.LON.size))
    inside = {r: JB.region_mask(JB.LAT, JB.LON, r) for r in JB.REGIONS}
    rows = []
    for i in leads:
        era = unpack(smasks[at[pr["valid"][i]]])
        occ = JB.near(era).astype(float)
        row = dict(lead_h=int(pr["steps"][i]), valid=f"{pr['valid'][i]:%Y-%m-%d %HZ}", members=pr["members"][i])
        for k, b in enumerate(JB.BRANCHES):
            bs, bc = brier(pr["share"][i, k], occ[k], w), brier(pr["normal"][i, k], occ[k], w)
            row[b] = dict(brier=round(bs, 5), brier_normal=round(bc, 5), bss=round(1 - bs / bc, 3) if bc else None)
        for r in JB.REGIONS:
            hit = any(g[0] >= pr["thresholds"][r] for g in JB.groups_of(era[2], inside[r]))
            p, c = float(pr["events"][r][i]), float(pr["event_normal"][r][i])
            row[r] = dict(share=round(p, 3), normal=round(c, 4), era5_event=bool(hit))
        rows.append(row)
    doc = dict(model=FC.LABEL, init=f"{init:%Y-%m-%d %HZ}", fetch=cost, near_km=JB.NEAR_KM, leads=rows)
    for r in JB.REGIONS:
        o = np.array([x[r]["era5_event"] for x in rows], float)
        p = np.array([x[r]["share"] for x in rows])
        c = np.array([x[r]["normal"] for x in rows])
        doc[f"{r}_events"] = dict(
            era5_analyses_with_event=int(o.sum()),
            brier=round(float(np.mean((p - o) ** 2)), 4),
            brier_normal=round(float(np.mean((c - o) ** 2)), 4),
        )
    tag = f"{init:%Y%m%d}"
    (out_dir / f"compare_{tag}.json").write_text(json.dumps(doc, indent=1) + "\n")
    figure(pr, rows, leads, smasks, at, out_dir / f"compare_{tag}.webp")
    for row in rows:
        print(
            f"  {row['valid']} (+{row['lead_h']:3d} h): BSS polar {row['polar']['bss']}, stj {row['stj']['bss']}, "
            f"sup {row['sup']['bss']}; events "
            + ", ".join(
                f"{r} {row[r]['share']:.2f} (ERA5 {'yes' if row[r]['era5_event'] else 'no'})" for r in JB.REGIONS
            ),
            flush=True,
        )
    return doc


def figure(pr, rows, leads, smasks, at, out):
    """Skill by lead, the regions' event shares against ERA5, and the superposition map where ERA5 had most."""
    import cartopy.crs as ccrs
    from matplotlib.colors import BoundaryNorm, ListedColormap

    plt = FC.mpl()
    W, H = 13.0, 7.4
    fig = plt.figure(figsize=(W, H))
    days = np.array([r["lead_h"] for r in rows]) / 24
    ax = fig.add_axes([0.06, 0.52, 0.27, 0.34])
    for b, col in zip(JB.BRANCHES, ("#2e86c1", "#e8512a", "#4d0f52")):
        ax.plot(days, [r[b]["bss"] for r in rows], color=col, lw=2, marker="o", ms=4, label=FC.TITLE[b].split(" (")[0])
    ax.axhline(0, color=FC.INK, lw=0.8)
    ax.set_title("Brier skill against the ERA5 normal", fontsize=11, loc="left", fontweight="bold")
    ax.set_xlabel("lead (days)")
    ax.legend(fontsize=8.5, frameon=False, loc="lower left")
    ax.grid(color="#e3e3e3", lw=0.6)
    for k, r in enumerate(JB.REGIONS):
        ax = fig.add_axes([0.40 + 0.31 * k, 0.52, 0.27, 0.34])
        ax.bar(days, [100 * x[r]["share"] for x in rows], width=0.4, color="#cc2f2f", label="members with an event")
        ax.plot(days, [100 * x[r]["normal"] for x in rows], color=FC.INK, lw=1.4, ls=(0, (5, 3)), label="ERA5 normal")
        hit = [d for d, x in zip(days, rows) if x[r]["era5_event"]]
        ax.plot(hit, [102] * len(hit), "v", color=FC.INK, ms=6, label="ERA5 had an event")
        ax.set_ylim(0, 108)
        ax.set_title(JB.REGIONS[r]["label"], fontsize=11, loc="left", fontweight="bold")
        ax.set_xlabel("lead (days)")
        ax.grid(axis="y", color="#e3e3e3", lw=0.6)
        if k == 0:
            ax.set_ylabel("%")
            ax.legend(fontsize=8.5, frameon=False, loc="center left")
    # the superposition where ERA5 had the most superposed columns in the run
    best = max(leads, key=lambda i: unpack(smasks[at[pr["valid"][i]]])[2].sum())
    era = JB.near(unpack(smasks[at[pr["valid"][best]]])[2])
    mw = 0.86
    mh = mw * W * (JB.LAT[-1] - JB.LAT[0]) / 360.0 / H
    ax = FC.nh_map(
        fig,
        [0.06, 0.08, mw, mh],
        f"Superposition at day {pr['steps'][best] / 24:g} (valid {pr['valid'][best]:%a %-d %b %HZ}): members within "
        f"{JB.NEAR_KM:g} km (shading), ERA5 within {JB.NEAR_KM:g} km (outline)",
    )
    cmap = ListedColormap(FC.P_COL)
    f, lon = FC.cyclic(100 * pr["share"][best, 2])
    cf = ax.contourf(
        lon, JB.LAT, f, levels=FC.P_LEV, cmap=cmap, norm=BoundaryNorm(FC.P_LEV, cmap.N), transform=ccrs.PlateCarree()
    )
    g, _ = FC.cyclic(era.astype(float))
    ax.contour(lon, JB.LAT, g, levels=[0.5], colors="#0b5394", linewidths=1.6, transform=ccrs.PlateCarree())
    FC.draw_regions(ax)
    cax = fig.add_axes([0.06 + mw + 0.01, 0.08, 0.012, mh])
    fig.colorbar(cf, cax=cax, ticks=FC.P_LEV[::2]).set_label("% of members", fontsize=9)
    fig.text(
        0.06,
        0.975,
        f"Jet branches, archived run against ERA5 — {FC.LABEL} {max(pr['members'])} members, init "
        f"{pr['init']:%Y-%m-%d %HZ}",
        fontsize=13.5,
        fontweight="bold",
        va="top",
    )
    fig.text(0.06, 0.02, "ERA5 from ARCO-ERA5 (Copernicus C3S), identified as the 1991–2020 reference", fontsize=8.5)
    fig.savefig(out, dpi=90, facecolor="white", pil_kwargs={"quality": 85, "method": 6})
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--start", required=True)
    s.add_argument("--end", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--workers", type=int, default=8)
    s = sub.add_parser("pick")
    s.add_argument("--scan", required=True)
    s.add_argument("--leads", default="3,6", help="days before each event's onset")
    s.add_argument("--github-output", action="store_true")
    s = sub.add_parser("compare")
    s.add_argument("--fc", required=True)
    s.add_argument("--scan", required=True)
    s.add_argument("--clim", required=True)
    s.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    if a.cmd == "scan":
        scan(a.start, a.end, a.out, a.workers)
    elif a.cmd == "pick":
        inits = pick(a.scan, [int(x) for x in a.leads.split(",")])
        print(f"inits: {inits}", flush=True)
        if a.github_output and os.environ.get("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
                fh.write(f"inits={json.dumps(inits)}\n")
    else:
        compare(a.fc, a.scan, a.clim, a.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
