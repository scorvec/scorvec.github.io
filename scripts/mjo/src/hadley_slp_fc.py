#!/usr/bin/env python3
"""Tropical sea-level pressure forecasts, member by member, from the AIFS-ENS, IFS-ENS and GEFS 00Z runs, against ERA5.

Every perturbed member's daily means at lead days 0..14 (hadley_slp_ref.fetch_run: the mean of the 00, 06, 12 and 18 UTC
valid times) less its model's drift against ERA5 at that lead and season (hadley_slp_ref.Drift, release hslp-ref-v1), then
against the ERA5 normal of the valid day (build_slp_clim.Normals): the anomaly in hPa and in units of the day's sd. A model
without a drift fit is left out rather than drawn raw: its drift would read as an anomaly.

Two manifests, one per units (assets/sst/anim/hslp_fc_manifest.json, normalised, and hslp_fc_hpa_manifest.json), each
with a region per ensemble and "blend" for the three pooled. Every loop opens with the last LEAD_IN analysed days
(hadley_slp.py's frames, on the same fixed scales), then shows the members' mean anomaly ONLY where at least AGREE of
the members give it the same sign, and only the lead days whose skill passed on the archived runs (hadley_slp_ref.SKILL:
MSSS of the normalised anomaly significant after Benjamini-Hochberg and at least hadley_slp_ref.MIN_SKILL); the contours
are the members' mean sea-level pressure.

    python src/hadley_slp_fc.py --date 20261018
"""

from __future__ import annotations

import pyproj  # noqa: F401  (load pyproj's libproj before xarray/eccodes)

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import build_slp_clim as C
import hadley_branch_maps as M
import hadley_slp as S
import hadley_slp_ref as R
from doycal import noleap_doy

AGREE = 0.8  # shade an anomaly only where this share of the members gives it one sign
LEAD_IN = 14  # analysed days the loops open with
MIN_SHARE = 0.5  # a model is drawn with at least this share of its members complete
PREFIX = {"sigma": "hslp_fc", "hpa": "hslp_fc_hpa"}  # manifest and region names by units


def run(model: str, init: pd.Timestamp, N: C.Normals, drift: np.ndarray) -> dict | None:
    """The model's members less the drift, reduced to what the loops need: their number, the sum of their daily means
    (lead, lat, lon) and how many lie above the normal; None when too few members are complete."""
    want = R.MODELS[model]["members"]
    x = R.fetch_run(model, init, list(range(1, want + 1)))
    x = x[np.isfinite(x).all(axis=(1, 2, 3))]
    print(f"  {R.MODELS[model]['label']}: {x.shape[0]} of {want} members complete", flush=True)
    if x.shape[0] < MIN_SHARE * want:
        return None
    x -= drift[None]
    normal = np.stack([N.normal(init + pd.Timedelta(days=L)) for L in R.LEADS])
    return {"n": x.shape[0], "sum": x.sum(0), "pos": (x > normal[None]).sum(0)}


def agreed(mean_anom: np.ndarray, pos: np.ndarray, n: int) -> np.ndarray:
    """The members' mean anomaly where at least AGREE of them share its sign, NaN (blank) elsewhere."""
    share = pos / n
    return np.where((share >= AGREE) | (share <= 1 - AGREE), mean_anom, np.nan)


def lead_in(init: pd.Timestamp, units: str) -> list:
    """The analysed frames of the LEAD_IN days before init, from the analysis loop's manifest (one directory over)."""
    name = S.UNITS[units]["dir"]
    mp = M.ANIM / f"{name}_manifest.json"
    past = json.loads(mp.read_text())["regions"][name]["frames"] if mp.exists() else []
    days = {f["date"]: f for f in past}
    keys = [f"{init - pd.Timedelta(days=n):%Y-%m-%d}" for n in range(LEAD_IN, 0, -1)]
    return [
        {"file": f"../{name}/{days[k]['file']}", "date": k, "label": f"analysis · {days[k]['label']}"}
        for k in keys
        if k in days
    ]


def render(regions: dict, skill: dict, init: pd.Timestamp, N: C.Normals):
    """Per units and region: the lead-in, then the agreed members' mean anomaly at the lead days that passed."""
    names = {rid: R.MODELS[rid]["label"] for rid in regions if rid != "blend"}
    if "blend" in regions:
        names = {"blend": " + ".join(names.values()), **names}
    for units, prefix in PREFIX.items():
        past = lead_in(init, units)
        doc = {
            "ver": f"{init:%Y%m%d}00",
            "init": f"{init:%Y-%m-%d}T00:00Z",
            "default": f"{prefix}_{next(iter(regions))}",
            "selectorLabel": "Forecast",
            "start": len(past),  # opens on +0 d (sst_anim ?start=manifest)
            "regions": {},
        }
        for rid, G in regions.items():
            d = M.ANIM / f"{prefix}_{rid}"  # the viewer reads <anim>/<region id>/<file>
            d.mkdir(parents=True, exist_ok=True)
            for old in d.glob("F*.webp"):
                old.unlink()
            sk = skill[rid]
            whose = "each member less its model's" if rid == "blend" else "members less the model's"
            frames = list(past)
            for L in range(sk["show_days"]):
                valid = init + pd.Timedelta(days=L)
                mean = G["sum"][L] / G["n"]
                fp = d / f"F{L:02d}.webp"
                S.draw(
                    agreed(S.anomalies(mean, N, valid)[units], G["pos"][L], G["n"]),
                    mean,
                    N,
                    units,
                    f"Tropical sea-level pressure — {names[rid]} {init:%Y-%m-%d} 00Z, daily mean {valid:%Y-%m-%d} (+{L} d)",
                    f"Members' mean {S.UNITS[units]['name']} against ERA5 1991–2020, shown where ≥{AGREE:.0%} of the "
                    f"{G['n']} members agree on its sign · contours every {S.CONTOUR:.0f} hPa",
                    f"{whose} drift against ERA5 at that lead and season (archived runs {sk['first'][:7]} to "
                    f"{sk['last'][:7]}) · days 0–{sk['show_days'] - 1} shown, the leads with significant skill on "
                    f"those runs\n{S.normals_note()}",
                    fp,
                )
                frames.append({"file": fp.name, "date": f"{valid:%Y-%m-%d}", "label": f"+{L} d · {valid:%a %b %-d}"})
            frames = [dict(f, idx=i) for i, f in enumerate(frames)]
            doc["regions"][f"{prefix}_{rid}"] = {"label": names[rid], "n_frames": len(frames), "frames": frames}
            print(f"  {prefix}_{rid}: {len(past)} analysed + {sk['show_days']} forecast frames", flush=True)
        (M.ANIM / f"{prefix}_manifest.json").write_text(json.dumps(doc))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="00Z run, YYYYMMDD")
    ap.add_argument("--models", default="aifs,ifs,gefs")
    a = ap.parse_args()
    init = pd.Timestamp(a.date)
    doy = float(noleap_doy(pd.DatetimeIndex([init]))[0])
    skill = json.loads(R.SKILL.read_text())["models"] if R.SKILL.exists() else {}
    D = R.Drift()
    N = C.Normals()
    regions = {}
    for model in a.models.split(","):
        drift = D.at(model, doy)
        if drift is None or not skill.get(model, {}).get("show_days"):
            print(f"  {R.MODELS[model]['label']}: no drift fit or no lead with significant skill, left out", flush=True)
            continue
        try:
            G = run(model, init, N, drift)
        except Exception as e:  # noqa: BLE001
            print(f"  {R.MODELS[model]['label']}: {type(e).__name__} {str(e)[:120]}", flush=True)
            continue
        if G is not None:
            regions[model] = G
    if not regions:
        print("no model drawn", file=sys.stderr)
        return 1
    if len(regions) == len(R.MODELS) and skill.get("blend", {}).get("show_days"):
        regions = {"blend": {k: sum(G[k] for G in regions.values()) for k in ("n", "sum", "pos")}, **regions}
    render(regions, skill, init, N)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
