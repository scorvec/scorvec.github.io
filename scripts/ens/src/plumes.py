#!/usr/bin/env python3
"""Temperature plumes for US cities — the pooled "mega ensemble" 2 m temperature
distribution over the Day 0–15 forecast, per city.

DEMO scope: pools the members we can pull cheaply — GEFS (Herbie, 31) + GEPS (MSC
datamart all-members file, ~21). IFS/AIFS perturbed members are S3-throttled, so they
are left out for now (the maps use their mean/control instead). Daily leads (00Z),
which capture the synoptic swing + spread but not the full diurnal range — fine for a
first look; densify the step list later for diurnal detail.

    python src/plumes.py --date 20260603 --run 00
"""
from __future__ import annotations
import argparse, os, sys, tempfile, warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from common import LEADS
from fetch import _members_da, _GEPS_BASE

warnings.filterwarnings("ignore")

CITIES = {                       # name → (lat, lon)
    "New York, NY":   (40.71, -74.01),
    "Chicago, IL":    (41.85, -87.65),
    "Denver, CO":     (39.74, -104.99),
    "Los Angeles, CA":(34.05, -118.24),
}
K2F = lambda k: (k - 273.15) * 9 / 5 + 32        # noqa: E731


def _pts(da, lat_name="latitude", lon_name="longitude"):
    """Extract every city point (nearest) from a (…, lat, lon) field → dict name→value(s)."""
    out = {}
    lons = da[lon_name]
    for name, (la, lo) in CITIES.items():
        lo360 = lo % 360 if float(lons.max()) > 180 else lo
        out[name] = da.sel(**{lat_name: la, lon_name: lo360}, method="nearest").values
    return out


def gefs_members(date, run, workers=12):
    """GEFS t2m members at the cities → {city: (n_member, n_lead) °F}."""
    from herbie import Herbie
    members = ["c00"] + [f"p{i:02d}" for i in range(1, 31)]
    cycle = f"{date[:4]}-{date[4:6]}-{date[6:8]} {run}:00"
    # accumulate per (city): list over (member,lead)
    data = {c: np.full((len(members), len(LEADS)), np.nan, "float32") for c in CITIES}

    def one(args):
        mi, m, li, ld = args
        try:
            H = Herbie(cycle, model="gefs", member=m, product="atmos.5", fxx=ld, verbose=False)
            if H.grib is None:
                return None
            ds = H.xarray(":TMP:2 m above ground:", remove_grib=True)
            da = ds[list(ds.data_vars)[0]] if hasattr(ds, "data_vars") else ds
            return mi, li, _pts(da.squeeze(drop=True))
        except Exception:                                    # noqa: BLE001
            return None

    tasks = [(mi, m, li, ld) for mi, m in enumerate(members) for li, ld in enumerate(LEADS)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(one, tasks):
            if r is None:
                continue
            mi, li, pts = r
            for c in CITIES:
                data[c][mi, li] = K2F(float(pts[c]))
    print(f"  GEFS: {len(members)} members × {len(LEADS)} leads", flush=True)
    return data


def geps_members(date, run):
    """GEPS t2m members at the cities → {city: (n_member, n_lead) °F}."""
    import requests
    init = f"{date}{run}"
    data = {c: [] for c in CITIES}                           # per lead: (n_member,) arrays
    counts = []
    for ld in LEADS:
        url = (f"{_GEPS_BASE}/{date}/WXO-DD/ensemble/geps/grib2/raw/{run}/{ld:03d}/"
               f"CMC_geps-raw_TMP_TGL_2m_latlon0p5x0p5_{init}_P{ld:03d}_allmbrs.grib2")
        tgt = tempfile.mktemp(suffix=".grib2")
        try:
            r = requests.get(url, timeout=180); r.raise_for_status()
            open(tgt, "wb").write(r.content)
            da = _members_da(tgt)                            # (number, lat, lon)
            pts = _pts(da)
            for c in CITIES:
                data[c].append(K2F(pts[c].astype("float32")))   # (n_member,)
            counts.append(da.sizes.get("number", 1))
        except Exception as e:                               # noqa: BLE001
            for c in CITIES:
                data[c].append(np.array([np.nan]))
            counts.append(0)
        finally:
            if os.path.exists(tgt):
                os.remove(tgt)
    nmem = max(counts) if counts else 0
    print(f"  GEPS: {nmem} members × {len(LEADS)} leads", flush=True)
    # pad ragged leads to (n_member, n_lead)
    out = {}
    for c in CITIES:
        arr = np.full((nmem, len(LEADS)), np.nan, "float32")
        for li, col in enumerate(data[c]):
            arr[:len(col), li] = col
        out[c] = arr
    return out


def plot(date, run, pooled, out_path):
    init = pd.Timestamp(f"{date}T{run}:00")
    valid = [init + pd.Timedelta(hours=ld) for ld in LEADS]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, city in zip(axes.ravel(), CITIES):
        m = pooled[city]                                     # (n_member, n_lead) °F
        p = {q: np.nanpercentile(m, q, axis=0) for q in (0, 10, 25, 50, 75, 90, 100)}
        ax.fill_between(valid, p[0], p[100], color="#cfe0f5", label="min–max")
        ax.fill_between(valid, p[10], p[90], color="#8fb8e8", label="10–90%")
        ax.fill_between(valid, p[25], p[75], color="#4f86c6", label="25–75%")
        ax.plot(valid, p[50], color="#08306b", lw=2, label="median")
        ax.set_title(f"{city}  ·  {m.shape[0]}-member mega-ensemble", fontsize=10, fontweight="bold")
        ax.set_ylabel("2 m temperature (°F)", fontsize=8)
        ax.grid(alpha=0.3); ax.tick_params(labelsize=7)
        for lab in ax.get_xticklabels():
            lab.set_rotation(30); lab.set_ha("right")
    axes[0, 0].legend(fontsize=7, loc="upper left", framealpha=0.9)
    fig.suptitle(f"2 m temperature plumes — GEFS+GEPS mega-ensemble  ·  init {init:%Y-%m-%d %HZ}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120); plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True); ap.add_argument("--run", default="00")
    ap.add_argument("--out", default="../../assets/ens/plumes_demo.webp")
    a = ap.parse_args()
    print("== temperature plumes (GEFS + GEPS) ==", flush=True)
    g = gefs_members(a.date, a.run)
    p = geps_members(a.date, a.run)
    pooled = {c: np.vstack([g[c], p[c]]) for c in CITIES}    # pool members
    plot(a.date, a.run, pooled, Path(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
