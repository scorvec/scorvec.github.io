#!/usr/bin/env python3
"""2 m temperature forecast plumes for US cities — AIFS-ENS vs ECMWF IFS-ENS, each
model in its own colour so their spread/median can be compared directly.

Members come from the shared ECMWF store: AIFS-ENS reuses the already-stored
`pf_sp-2t` surface batch; IFS-ENS pulls its own `pf 2t` batch (added on first run).
Both are perturbed-member (pf) files, 50 members. Daily 00Z leads (Day 0–15).

TODO: when the 6-hourly 2 m-temp download lands, switch LEADS to the sub-daily steps
for the full diurnal range (and true daily means).

    python src/plumes.py --date 20260604 --run 00
"""
from __future__ import annotations
import argparse, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from common import LEADS
from fetch import _members_da
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ecmwf"))
import store

warnings.filterwarnings("ignore")

CITIES = {                       # name → (lat, lon)
    "New York, NY":   (40.71, -74.01),
    "Chicago, IL":    (41.85, -87.65),
    "Denver, CO":     (39.74, -104.99),
    "Los Angeles, CA":(34.05, -118.24),
}
K2F = lambda k: (k - 273.15) * 9 / 5 + 32        # noqa: E731

MODELS = ["aifs", "ifs"]
MODEL_LABEL = {"aifs": "AIFS-ENS", "ifs": "ECMWF IFS-ENS"}
MODEL_COLOR = {"aifs": "#d1495b", "ifs": "#1f77b4"}     # red-ish / blue


def _member_path(model: str, date: str, run: str) -> Path:
    """Stored pf 2 m-temp file for the model (download IFS's batch on first use)."""
    cyc = store.Cycle(date, run)
    if model == "aifs":
        return store.sfc_path(cyc, "aifs-ens", "pf", "2t")          # the sp-2t batch
    return store.ensure(cyc, store.Spec("ifs", "pf", "2t", "sfc", (), tuple(store.STEPS)))


def members(model: str, date: str, run: str) -> dict:
    """{city: (n_member, n_lead) °F} of 2 m temperature for one model."""
    da = _members_da(str(_member_path(model, date, run)), short="2t")   # (number, step, lat, lon)
    lons = da.longitude
    out = {}
    for name, (la, lo) in CITIES.items():
        lo360 = lo % 360 if float(lons.max()) > 180 else lo
        v = da.sel(latitude=la, longitude=lo360, method="nearest").values   # (number, step)
        out[name] = K2F(np.asarray(v, "float32"))
    n = da.sizes.get("number", 1)
    print(f"  {MODEL_LABEL[model]}: {n} members × {da.sizes.get('step', len(LEADS))} leads", flush=True)
    return out


def plot(date, run, data, out_path):
    """data = {model: {city: (n_member, n_lead) °F}}."""
    init = pd.Timestamp(f"{date}T{run}:00")
    valid = [init + pd.Timedelta(hours=ld) for ld in LEADS]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, city in zip(axes.ravel(), CITIES):
        for model in MODELS:
            if model not in data:
                continue
            m = data[model][city]                                  # (n_member, n_lead)
            col = MODEL_COLOR[model]
            p10, p50, p90 = (np.nanpercentile(m, q, axis=0) for q in (10, 50, 90))
            ax.fill_between(valid, p10, p90, color=col, alpha=0.16, linewidth=0)
            ax.plot(valid, p50, color=col, lw=2.0,
                    label=f"{MODEL_LABEL[model]} ({m.shape[0]} mbrs, 10–90%)")
        ax.set_title(city, fontsize=10, fontweight="bold")
        ax.set_ylabel("2 m temperature (°F)", fontsize=8)
        ax.grid(alpha=0.3); ax.tick_params(labelsize=7)
        for lab in ax.get_xticklabels():
            lab.set_rotation(30); lab.set_ha("right")
    axes[0, 0].legend(fontsize=7.5, loc="upper left", framealpha=0.9)
    fig.suptitle(f"2 m temperature plumes — AIFS-ENS vs ECMWF IFS-ENS  ·  init {init:%Y-%m-%d %HZ}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120); plt.close(fig)
    print(f"  wrote {out_path}")


def build(date: str, run: str, out_path) -> None:
    print("== 2 m temperature plumes (AIFS-ENS + IFS-ENS) ==", flush=True)
    data = {}
    for model in MODELS:
        try:
            data[model] = members(model, date, run)
        except Exception as e:                                     # noqa: BLE001
            print(f"  {MODEL_LABEL[model]}: FAILED ({repr(e)[:80]})", flush=True)
    if data:
        plot(date, run, data, Path(out_path))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True); ap.add_argument("--run", default="00")
    ap.add_argument("--out", default="../../assets/ens/plumes_temps.webp")
    a = ap.parse_args()
    build(a.date, a.run, Path(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
