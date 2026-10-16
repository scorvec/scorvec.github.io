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
import time
from datetime import date, datetime, timedelta, UTC
from pathlib import Path

import requests
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader

HERE = Path(__file__).resolve().parent
OUTDIR = HERE.parent / "assets" / "site_stats"
SITE = "https://scorvec.goatcounter.com"
TOKEN = os.environ.get("GOATCOUNTER_TOKEN", "")


def api(path: str, **params) -> dict:
    # GoatCounter's hosted service occasionally serves a transient HTML error
    # page (seen as a 404 on a path that works minutes later) — retry before
    # failing the whole stats build.
    last = None
    for attempt in range(4):
        r = requests.get(
            f"{SITE}/api/v0/{path}", params=params, headers={"Authorization": f"Bearer {TOKEN}"}, timeout=45
        )
        if r.status_code < 400:
            return r.json()
        print(f"  API {path} -> {r.status_code} (attempt {attempt + 1}): {r.text[:200]}", file=sys.stderr)
        last = r
        time.sleep(5 * (attempt + 1))
    last.raise_for_status()


def _ts(d: str | date) -> str:
    """GoatCounter wants hour-rounded RFC3339 timestamps, not bare dates."""
    return f"{d}T00:00:00Z"


def _end_ts() -> str:
    """End of the query window: the NEXT hour boundary, so today's traffic
    is included (midnight-today silently drops the current day)."""
    now = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    return (now + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")


def locations(start: str) -> dict[str, int]:
    """{ISO2: unique-visitor count} since `start` (paged)."""
    out, offset = {}, 0
    while True:
        d = api("stats/locations", start=_ts(start), end=_end_ts(), limit=100, offset=offset)
        for row in d.get("stats", []):
            out[row["id"]] = out.get(row["id"], 0) + int(row["count"])
        if not d.get("more"):
            return out
        offset += 100


def draw_panel(ax, counts: dict[str, int], title: str):
    reader = shpreader.Reader(shpreader.natural_earth(resolution="110m", category="cultural", name="admin_0_countries"))
    vmax = max(counts.values()) if counts else 1
    norm = LogNorm(vmin=1, vmax=max(vmax, 10))
    cmap = plt.get_cmap("YlGnBu")
    for rec in reader.records():
        iso = rec.attributes.get("ISO_A2_EH") or rec.attributes.get("ISO_A2")
        n = counts.get(iso, 0)
        face = cmap(norm(n)) if n > 0 else "#f2f0eb"
        ax.add_geometries([rec.geometry], ccrs.PlateCarree(), facecolor=face, edgecolor="#6f6b64", linewidth=0.25)
    ax.set_global()
    ax.set_frame_on(False)
    total = sum(counts.values())
    ax.set_title(
        f"{title} — {total:,} visitors · {len(counts)} countries", fontsize=11.5, fontweight="bold", loc="left"
    )
    return plt.cm.ScalarMappable(norm=norm, cmap=cmap)


def main() -> int:
    if not TOKEN:
        print("GOATCOUNTER_TOKEN not set — skipping (add the repo secret to enable).")
        return 0
    start30 = (date.today() - timedelta(days=30)).isoformat()
    loc30 = locations(start30)
    loc_all = locations("2026-07-18")  # site epoch: GoatCounter install date
    pages = api("stats/hits", start=_ts(start30), end=_end_ts(), limit=10)
    top_pages = [{"path": p["path"], "count": int(p["count"])} for p in pages.get("hits", [])][:10]

    # Fixed axes boxes sized to the Robinson map's ~2:1 aspect: with add_subplot the maps float in the middle of
    # a taller figure and leave a wide empty band under the title
    fig = plt.figure(figsize=(12.6, 4.25))
    ax1 = fig.add_axes([0.005, 0.2, 0.49, 0.66], projection=ccrs.Robinson())
    ax2 = fig.add_axes([0.505, 0.2, 0.49, 0.66], projection=ccrs.Robinson())
    draw_panel(ax1, loc30, "Last 30 days")
    sm = draw_panel(ax2, loc_all, "All time")
    cax = fig.add_axes([0.35, 0.155, 0.3, 0.03])  # above the source line, which sits at y=0.005
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("unique visitors (log scale)", fontsize=9)
    cb.ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))  # 1, 10, 100 rather than 10^0 ...
    cb.ax.tick_params(labelsize=8.5)
    fig.suptitle("Where scorvec.com's visitors come from", fontsize=14, fontweight="bold", y=0.985)
    fig.text(
        0.5,
        0.005,
        "GoatCounter (privacy-friendly, cookie-free) · country-level IP geolocation · updated daily",
        ha="center",
        fontsize=8.5,
        color="#6f6b64",
    )
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTDIR / "visitor_map.webp", dpi=115, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    (OUTDIR / "visitor_stats.json").write_text(
        json.dumps(
            {
                "updated": date.today().isoformat(),
                "total_30d": sum(loc30.values()),
                "countries_30d": len(loc30),
                "total_all": sum(loc_all.values()),
                "countries_all": len(loc_all),
                "top_pages_30d": top_pages,
            },
            indent=1,
        )
    )
    print(f"wrote visitor_map.webp ({sum(loc30.values())} uniques/30d, {len(loc30)} countries) + visitor_stats.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
