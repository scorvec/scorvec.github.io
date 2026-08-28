#!/usr/bin/env python3
"""NHC ATCF model-guidance plotter: one spaghetti panel per active NHC
system/invest (Atlantic, EPAC, CPAC), from the public ATCF a-decks.

For each system in CurrentStorms.json + active invest best-track files, the
latest a-deck cycle's tracks from a curated model set are drawn in standard
guidance colors (OFCL black, GFS red, ECMWF blue, HAFS purple/pink, UKMET
orange, CMC green, consensus grey), with 24-h TAU labels along each track.

Output: assets/tc/invests/invest_<ID>.webp + invests_meta.json (page gallery).

    python invest_models.py --out-dir ../../assets/tc
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

ATCF = "https://ftp.nhc.noaa.gov/atcf"
# (aliases, label, color, lw, ls) — draw order (first = bottom). Each family
# lists its ATCF TECH id variants (interpolated first, then raw): the a-decks
# file the same model under different ids depending on cycle timing.
MODELS = [
    (("NVGI", "NVGM"),                 "NAVGEM",       "#8c564b", 1.1, "-"),
    (("ICNI", "ICON"),                 "ICON",         "#bcbd22", 1.2, "-"),
    (("CMCI", "CMC2", "CMC"),          "CMC",          "#2ca02c", 1.3, "-"),
    (("EGRI", "EGR2", "EGRR", "UKXI", "UKX"), "UKMET", "#ff7f0e", 1.3, "-"),
    (("CTCI", "CTC2", "CTCX"),         "COAMPS-TC",    "#17becf", 1.3, "-"),
    (("HMNI", "HMON"),                 "HMON",         "#7f7f7f", 1.2, "-"),
    (("HFBI", "HFSB"),                 "HAFS-B",       "#e377c2", 1.4, "-"),
    (("HFAI", "HFSA"),                 "HAFS-A",       "#9467bd", 1.4, "-"),
    (("AEMI", "AEM2", "AEMN"),         "GFS ens mean", "#d62728", 1.2, "--"),
    (("AVNI", "AVN2", "AVNO"),         "GFS",          "#d62728", 1.5, "-"),
    (("EMXI", "EMX2", "EMX"),          "ECMWF",        "#1f77b4", 1.5, "-"),
    (("TVCN", "TVCX", "TVCA", "TVCE", "GUNA"), "Consensus", "#555555", 1.8, "--"),
    (("OFCL", "OFCI"),                 "NHC official", "#000000", 2.6, "-"),
]


def active_systems():
    """[{deck 'al95', id 'AL95', name}, ...] from CurrentStorms + invest btk files."""
    out, seen = [], set()
    try:
        r = requests.get("https://www.nhc.noaa.gov/CurrentStorms.json", timeout=15)
        for s in r.json().get("activeStorms", []):
            b = s.get("binNumber", "").upper()          # e.g. "AT3" — use id instead
            sid = s.get("id", "").lower()               # e.g. "al052026"
            m = re.match(r"(al|ep|cp)(\d{2})(\d{4})", sid)
            if m and sid[:4] not in seen:
                seen.add(sid[:4])
                out.append({"deck": f"{m.group(1)}{m.group(2)}{m.group(3)}",
                            "id": f"{m.group(1).upper()}{m.group(2)}",
                            "name": s.get("name", "").title()})
    except Exception as e:                                     # noqa: BLE001
        print(f"  CurrentStorms unavailable ({str(e)[:50]})")
    try:
        listing = requests.get(f"{ATCF}/btk/", timeout=15).text
        for fn in sorted(set(re.findall(r"b((?:al|ep|cp)9\d{5})\.dat", listing))):
            key = fn[:4]
            if key not in seen:
                seen.add(key)
                out.append({"deck": fn, "id": fn[:4].upper(), "name": "Invest"})
    except Exception as e:                                     # noqa: BLE001
        print(f"  btk listing unavailable ({str(e)[:50]})")
    return out


def fetch_adeck(deck: str):
    """Parse the a-deck into {(dtg, tech): [(tau, lat, lon, vmax), ...]}."""
    url = f"{ATCF}/aid_public/a{deck}.dat.gz"
    try:
        r = requests.get(url, timeout=25)
        r.raise_for_status()
        text = gzip.decompress(r.content).decode("ascii", "replace")
    except Exception as e:                                     # noqa: BLE001
        print(f"  a-deck {deck}: {str(e)[:60]}")
        return {}
    rows = {}
    for line in text.splitlines():
        c = [x.strip() for x in line.split(",")]
        if len(c) < 10:
            continue
        dtg, tech, tau = c[2], c[4], c[5]
        try:
            tau = int(tau)
            la, lo = c[6], c[7]
            lat = int(la[:-1]) / 10 * (1 if la.endswith("N") else -1)
            lon = int(lo[:-1]) / 10 * (1 if lo.endswith("E") else -1)
            vmax = int(c[8]) if c[8] and c[8] != "0" else np.nan
        except (ValueError, IndexError):
            continue
        rows.setdefault((dtg, tech), {})[tau] = (lat, lon, vmax)
    return rows


TYPE_NAMES = {"LO": "remnant low", "DB": "disturbance", "WV": "tropical wave",
              "TD": "tropical depression", "TS": "tropical storm",
              "HU": "hurricane", "TY": "typhoon", "SS": "subtropical storm",
              "SD": "subtropical depression", "EX": "extratropical"}


def bdeck_summary(deck: str):
    """History/intensity summary from the best-track deck (designation time,
    current fix, peak intensity, recent motion)."""
    try:
        txt = requests.get(f"{ATCF}/btk/b{deck}.dat", timeout=20).text
    except Exception:                                          # noqa: BLE001
        return {}
    fixes = []
    for line in txt.strip().splitlines():
        c = [x.strip() for x in line.split(",")]
        if len(c) < 11:
            continue
        try:
            la, lo = c[6], c[7]
            fixes.append({
                "dtg": c[2],
                "lat": int(la[:-1]) / 10 * (1 if la.endswith("N") else -1),
                "lon": int(lo[:-1]) / 10 * (1 if lo.endswith("E") else -1),
                "vmax": int(c[8]) if c[8] else 0,
                "mslp": int(c[9]) if c[9] else 0,
                "type": c[10] if len(c) > 10 else ""})
        except (ValueError, IndexError):
            continue
    if not fixes:
        return {}
    fixes = {f["dtg"]: f for f in fixes}                       # dedupe radii lines
    fixes = [fixes[k] for k in sorted(fixes)]
    cur, first = fixes[-1], fixes[0]
    peak = max(fixes, key=lambda f: f["vmax"])
    motion = ""
    if len(fixes) >= 2:
        a, b = fixes[-2], fixes[-1]
        dt = (pd.to_datetime(b["dtg"], format="%Y%m%d%H")
              - pd.to_datetime(a["dtg"], format="%Y%m%d%H")).total_seconds() / 3600
        if dt > 0:
            import math
            dlat = b["lat"] - a["lat"]
            dlon = (b["lon"] - a["lon"]) * math.cos(math.radians(b["lat"]))
            spd = math.hypot(dlat, dlon) * 60 / dt              # nm/h = kt
            brg = (math.degrees(math.atan2(dlon, dlat)) + 360) % 360
            dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S",
                    "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
            motion = f"{dirs[int((brg + 11.25) // 22.5) % 16]} at {spd:.0f} kt"
    hours = (pd.to_datetime(cur["dtg"], format="%Y%m%d%H")
             - pd.to_datetime(first["dtg"], format="%Y%m%d%H")).total_seconds() / 3600
    return {"designated": f"{first['dtg'][:8]} {first['dtg'][8:]}Z",
            "hours_tracked": int(hours),
            "cur_time": f"{cur['dtg'][:8]} {cur['dtg'][8:]}Z",
            "cur_lat": cur["lat"], "cur_lon": cur["lon"],
            "cur_vmax": cur["vmax"], "cur_mslp": cur["mslp"],
            "cur_type": TYPE_NAMES.get(cur["type"], cur["type"] or "—"),
            "peak_vmax": peak["vmax"], "motion": motion}


def plot_system(sysd, adeck, out_dir: Path):
    """One guidance panel for the latest cycle with enough models."""
    dtgs = sorted({d for d, _ in adeck})
    # among the last 4 cycles, pick the one with the best model-family
    # coverage (ties → newest); each family uses its first alias present
    best = (0, None, {})
    for age, dtg in enumerate(reversed(dtgs[-4:])):
        fams = {}
        for aliases, label, *_ in MODELS:
            for tech in aliases:
                if (dtg, tech) in adeck and len(adeck[(dtg, tech)]) >= 3:
                    fams[label] = adeck[(dtg, tech)]
                    break
        score = len(fams) - age * 0.1              # slight preference for newest
        if score > best[0] and len(fams) >= 3:
            best = (score, dtg, fams)
    _, chosen, tracks = best
    if not chosen:
        return None
    # dead-system gate: invest decks linger on the server for weeks — only
    # plot systems with guidance from the last ~30 h
    age_h = (pd.Timestamp.utcnow().tz_localize(None)
             - pd.to_datetime(chosen, format="%Y%m%d%H")).total_seconds() / 3600
    if age_h > 30:
        print(f"  {sysd['id']}: newest guidance {chosen} is {age_h:.0f} h old — skipped (dead)")
        return None
    las = [p[0] for tr in tracks.values() for p in tr.values()]
    l0 = next(iter(tracks.values()))[min(next(iter(tracks.values())))][1]
    wrap = lambda lon: l0 + (((lon - l0 + 180) % 360) - 180)
    los = [wrap(p[1]) for tr in tracks.values() for p in tr.values()]
    w0, e0 = min(los) - 2.5, max(los) + 2.5
    s0, n0 = min(las) - 1.5, max(las) + 1.5
    if e0 - w0 < 14: pad = (14 - (e0 - w0)) / 2; w0 -= pad; e0 += pad
    if n0 - s0 < 10: pad = (10 - (n0 - s0)) / 2; s0 -= pad; n0 += pad
    aspect = (n0 - s0) / (e0 - w0)
    figw = 9.6
    fig = plt.figure(figsize=(figw, figw * aspect * 1.22 + 1.0), facecolor="white")
    ax = fig.add_subplot(1, 1, 1,
                         projection=ccrs.PlateCarree(central_longitude=(w0 + e0) / 2))
    ax.set_extent([w0, e0, s0, n0], crs=ccrs.PlateCarree())
    ax.set_facecolor("#d5ecf5")
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#e8dcb8", zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor="#8a8a7a",
                   linewidth=0.6, zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="#b0b0a0",
                   linewidth=0.35, zorder=3)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="#b9cfd9",
                      linestyle="--", xlocs=range(-180, 181, 5),
                      ylocs=range(-60, 61, 5))
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"color": "#667", "size": 7}
    handles = []
    for aliases, label, color, lw, ls in MODELS:
        tr = tracks.get(label)
        if not tr:
            continue
        taus = sorted(tr)
        la = [tr[t][0] for t in taus]; lo = [wrap(tr[t][1]) for t in taus]
        ln, = ax.plot(lo, la, color=color, lw=lw, ls=ls, alpha=0.95,
                      transform=ccrs.Geodetic(), zorder=6 if label == "NHC official" else 5,
                      label=label)
        handles.append(ln)
        for t in taus:
            if t % 24 == 0 and t > 0:
                ax.annotate(str(t), xy=(wrap(tr[t][1]), tr[t][0]),
                            xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
                            fontsize=5.2, color=color, alpha=0.8, ha="center",
                            zorder=7)
        if 0 in tr:
            ax.plot(wrap(tr[0][1]), tr[0][0], marker="o", ms=5, color=color, mec="k",
                    mew=0.4, transform=ccrs.PlateCarree(), zorder=7)
    ax.legend(handles=handles, loc="best", fontsize=7.2, framealpha=0.92,
              borderpad=0.6)
    nm = sysd["name"] if sysd["name"] not in ("", "Invest") else "Invest"
    ax.set_title(f"{nm} {sysd['id']} — ATCF model guidance · cycle {chosen} · "
                 f"numbers = forecast hour", fontsize=10.5, fontweight="bold")
    fn = f"invest_{sysd['id']}.webp"
    (out_dir / "invests").mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "invests" / fn, dpi=135, bbox_inches="tight",
                facecolor="white", pil_kwargs={"quality": 92, "method": 6})
    plt.close(fig)
    meta = {"id": sysd["id"], "name": nm, "file": f"invests/{fn}", "cycle": chosen}
    meta.update(bdeck_summary(sysd["deck"]))
    return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="../../assets/tc")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    systems = active_systems()
    print(f"== NHC guidance plotter: {len(systems)} active system(s) ==")
    idir = out_dir / "invests"
    if idir.exists():
        for f in idir.glob("invest_*.webp"):
            f.unlink()
    meta = []
    for sysd in systems:
        m = plot_system(sysd, fetch_adeck(sysd["deck"]), out_dir)
        if m:
            meta.append(m)
            print(f"  {m['id']} ({m['name']}): cycle {m['cycle']}")
    (out_dir / "invests_meta.json").write_text(json.dumps(
        {"generated": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%MZ"),
         "systems": meta}))
    print(f"  wrote {len(meta)} guidance panel(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
