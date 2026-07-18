#!/usr/bin/env python3
"""Visitor-location map + site stats from the GoatCounter API.

Renders a world choropleth of unique visitors by country (last 30 days and
all-time side by side) plus a top-pages table, for the stats dashboard page.
Runs daily from .github/workflows/site-stats.yml with GOATCOUNTER_TOKEN
(Settings → API tokens, "read statistics" permission) in the repo secrets.

Outputs:
    assets/site_stats/visitor_map.webp
    assets/site_stats/visitor_stats.json   (totals + top pages, for stats.html)

    GOATCOUNTER_TOKEN=… python scripts/visitor_map.py
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader

HERE = Path(__file__).resolve().parent
OUTDIR = HERE.parent / "assets" / "site_stats"
SITE = "https://scorvec.goatcounter.com"
TOKEN = os.environ.get("GOATCOUNTER_TOKEN", "")


def api(path: str, **params) -> dict:
    r = requests.get(f"{SITE}/api/v0/{path}", params=params,
                     headers={"Authorization": f"Bearer {TOKEN}"}, timeout=45)
    if r.status_code >= 400:
        print(f"  API {path} -> {r.status_code}: {r.text[:300]}", file=sys.stderr)
    r.raise_for_status()
    return r.json()


def _ts(d: str | date) -> str:
    """GoatCounter wants hour-rounded RFC3339 timestamps, not bare dates."""
    return f"{d}T00:00:00Z"


def locations(start: str) -> dict[str, int]:
    """{ISO2: unique-visitor count} since `start` (paged)."""
    out, offset = {}, 0
    while True:
        d = api("stats/locations", start=_ts(start), end=_ts(date.today().isoformat()),
                limit=100, offset=offset)
        for row in d.get("stats", []):
            out[row["id"]] = out.get(row["id"], 0) + int(row["count"])
        if not d.get("more"):
            return out
        offset += 100


def draw_panel(ax, counts: dict[str, int], title: str):
    reader = shpreader.Reader(shpreader.natural_earth(
        resolution="110m", category="cultural", name="admin_0_countries"))
    vmax = max(counts.values()) if counts else 1
    norm = LogNorm(vmin=1, vmax=max(vmax, 10))
    cmap = plt.get_cmap("YlGnBu")
    for rec in reader.records():
        iso = rec.attributes.get("ISO_A2_EH") or rec.attributes.get("ISO_A2")
        n = counts.get(iso, 0)
        face = cmap(norm(n)) if n > 0 else "#f2f0eb"
        ax.add_geometries([rec.geometry], ccrs.PlateCarree(),
                          facecolor=face, edgecolor="#999", linewidth=0.25)
    ax.set_global(); ax.set_frame_on(False)
    total = sum(counts.values())
    ax.set_title(f"{title} — {total:,} visitors · {len(counts)} countries",
                 fontsize=10.5, fontweight="bold", loc="left")
    return plt.cm.ScalarMappable(norm=norm, cmap=cmap)


def main() -> int:
    if not TOKEN:
        print("GOATCOUNTER_TOKEN not set — skipping (add the repo secret to enable).")
        return 0
    start30 = (date.today() - timedelta(days=30)).isoformat()
    loc30 = locations(start30)
    loc_all = locations("2026-07-18")            # site epoch: GoatCounter install date
    pages = api("stats/hits", start=_ts(start30), end=_ts(date.today().isoformat()), limit=10)
    top_pages = [{"path": p["path"], "count": int(p["count"])}
                 for p in pages.get("hits", [])][:10]

    fig = plt.figure(figsize=(12.6, 4.6))
    ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.Robinson())
    ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.Robinson())
    draw_panel(ax1, loc30, "Last 30 days")
    sm = draw_panel(ax2, loc_all, "All time")
    cax = fig.add_axes([0.35, 0.06, 0.3, 0.03])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("unique visitors (log scale)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.suptitle("Where scorvec.com's visitors come from", fontsize=13,
                 fontweight="bold", y=0.99)
    fig.text(0.5, 0.005, "GoatCounter (privacy-friendly, cookie-free) · country-level "
             "IP geolocation · updated daily", ha="center", fontsize=7.5, color="0.4")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTDIR / "visitor_map.webp", dpi=115, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)

    (OUTDIR / "visitor_stats.json").write_text(json.dumps({
        "updated": date.today().isoformat(),
        "total_30d": sum(loc30.values()), "countries_30d": len(loc30),
        "total_all": sum(loc_all.values()), "countries_all": len(loc_all),
        "top_pages_30d": top_pages,
    }, indent=1))
    print(f"wrote visitor_map.webp ({sum(loc30.values())} uniques/30d, "
          f"{len(loc30)} countries) + visitor_stats.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
