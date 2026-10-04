#!/usr/bin/env python3
"""Derived reference tables for the SEAS5 page — what lets the build run in Actions.

WHY. The SEAS5 products are anomalies, terciles and quantile maps against SEAS5's own
1993–2016 hindcast (25 members × 24 years per start month) and against observed records
(ERA5 monthly means, CPC daily Tmax/Tmin, the GEPS/GEFS teleconnection patterns). Those raw
inputs are ~23 GB of GRIB per handful of start months plus ~100 GB of local reanalysis, and by
user rule (2026-10-04) they are downloaded ONCE and never again: the laptop never downloads, and
Actions downloads only what is new. So every product reads COMPACT DERIVED TABLES instead of the
raw GRIBs:

  hc_{unit}_{MM}.npz     per hindcast START MONTH: the statistics a product needs from the
                         hindcast (means per lead, tercile bounds per season, per-sample index
                         series, quantile-mapping tables ...). Built once per start month — on the
                         laptop from the local GRIBs where they exist, otherwise in Actions straight
                         after the one-time hindcast fetch, which is then deleted.
  is_{unit}_{YYYYMM}.npz per ISSUE: what next month's build needs from this issue (ensemble means,
                         index members) for the "change since the previous issue" products, so the
                         previous issue's forecast GRIBs are never fetched twice.
  obs_{name}.npz         static observed references (ERA5 normals per calendar month, CPC daily
                         quantiles, teleconnection projectors). Built once on the laptop from the
                         local stores; never fetched in Actions.

All three live flat in one directory (REF, default scripts/sst/data/seas5/ref, override with
SEAS5_REF_DIR) so they map one-to-one onto the assets of the GitHub release `seas5-ref-v1`
(.github/workflows/seas5.yml downloads what the issue needs and uploads what it made).

Each product module declares what it derives:

    REF_UNITS   = {unit: dict(raw=[raw keys], fn=derive(month))}     → hc_{unit}_{MM}.npz
    ISSUE_UNITS = {unit: derive(ym)}                                 → is_{unit}_{ym}.npz
    OBS_UNITS   = {name: derive()}                                   → obs_{name}.npz
    RAW_HC      = {raw key: (paths(month) -> [Path], fetch(month) -> bool)}   (non-monthly raw kinds)

Raw keys "m:{kind}" are the monthly-mean hindcast kinds of seas5_outlook.KINDS.

    python seas5_ref.py hindcast --month 09                # derive what the local GRIBs allow (never fetches)
    python seas5_ref.py hindcast --month 10 --fetch --discard-raw --budget-min 240   # Actions
    python seas5_ref.py issue --issue 202609               # per-issue summaries from local forecasts
    python seas5_ref.py obs                                # observed references from the local stores
    python seas5_ref.py missing --month 10                 # units without a table (exit 1 if any)
    python seas5_ref.py ls
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from seas5_outlook import CLIM_YEARS, DATA, KINDS, _retrieve, hc_path  # noqa: E402

REF = Path(os.environ["SEAS5_REF_DIR"]).resolve() if os.environ.get("SEAS5_REF_DIR") else DATA / "ref"

# product modules, in build order; each may define REF_UNITS / ISSUE_UNITS / OBS_UNITS / RAW_HC
MODULES = ["seas5_build", "seas5_normals", "seas5_tele", "seas5_pme_regions", "seas5_popT",
           "seas5_extremes_build", "seas5_runs", "seas5_wind", "seas5_snow", "seas5_regimes"]


# ── paths and I/O ────────────────────────────────────────────────────────────
def hc_ref(unit: str, month: str) -> Path:
    return REF / f"hc_{unit}_{int(month):02d}.npz"


def is_ref(unit: str, ym: str) -> Path:
    return REF / f"is_{unit}_{ym}.npz"


def obs_ref(name: str) -> Path:
    return REF / f"obs_{name}.npz"


def save(path: Path, meta: dict | None = None, **arrays) -> Path:
    """Atomic compressed npz; `meta` (JSON-able) rides along as the __meta__ entry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + f".part{os.getpid()}.npz")
    if meta is not None:
        arrays["__meta__"] = np.array(json.dumps(meta))
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
    print(f"    ref {path.name}: {path.stat().st_size / 1e6:.2f} MB", flush=True)
    return path


def load(path: Path) -> dict | None:
    """{name: array} (+ 'meta' dict) or None when the table is not there."""
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files if k != "__meta__"}
        if "__meta__" in z.files:
            out["meta"] = json.loads(str(z["__meta__"]))
    return out


def need(path: Path, what: str) -> dict:
    """load() or raise FileNotFoundError naming the missing table (callers skip that product)."""
    d = load(path)
    if d is None:
        raise FileNotFoundError(f"derived table {path.name} missing ({what}); run seas5_ref.py first")
    return d


# ── network guard: the laptop never downloads (user rule 2026-10-04) ─────────
def fetch_allowed() -> bool:
    """Downloads run in GitHub Actions only. SEAS5_OFFLINE=1 forbids them anywhere; outside Actions
    they need an explicit SEAS5_ALLOW_FETCH=1, which nothing on the laptop sets."""
    if os.environ.get("SEAS5_OFFLINE") == "1":
        return False
    return os.environ.get("GITHUB_ACTIONS") == "true" or os.environ.get("SEAS5_ALLOW_FETCH") == "1"


def guard_fetch(what: str) -> None:
    if not fetch_allowed():
        raise RuntimeError(f"download refused ({what}): SEAS5 data is fetched in GitHub Actions only "
                           "(set SEAS5_ALLOW_FETCH=1 to override deliberately)")


# ── registry ─────────────────────────────────────────────────────────────────
def _mod(name: str):
    return importlib.import_module(name)


def _modules() -> list[str]:
    """SEAS5_REF_MODULES=a,b restricts the registry (e.g. while one module is being worked on)."""
    sel = os.environ.get("SEAS5_REF_MODULES")
    return [m for m in MODULES if not sel or m in sel.split(",")]


def units(kind: str) -> dict:
    """kind 'REF_UNITS' | 'ISSUE_UNITS' | 'OBS_UNITS' → {unit: (module name, spec)}."""
    out = {}
    for m in _modules():
        for u, spec in getattr(_mod(m), kind, {}).items():
            if u in out:
                raise ValueError(f"unit {u} declared twice ({out[u][0]}, {m})")
            out[u] = (m, spec)
    return out


def raw_registry() -> dict:
    """raw key → (paths(month), fetch(month))."""
    reg = {}
    for kind in KINDS:
        reg[f"m:{kind}"] = ((lambda month, k=kind: [hc_path(k, f"{int(month):02d}")]),
                            (lambda month, k=kind: _retrieve(k, CLIM_YEARS, f"{int(month):02d}", hc_path(k, f"{int(month):02d}"))))
    for m in _modules():
        reg.update(getattr(_mod(m), "RAW_HC", {}))
    return reg


def raw_present(key: str, month: str, reg: dict) -> bool:
    return all(p.exists() and p.stat().st_size > 0 for p in reg[key][0](month))


def missing(month: str, only=None) -> list[str]:
    return [u for u in units("REF_UNITS") if (not only or u in only) and not hc_ref(u, month).exists()]


def derive_hindcast(month: str, fetch: bool = False, discard_raw: bool = False, budget_min: float | None = None,
                    only=None, force: bool = False) -> list[str]:
    """Build every missing hc_{unit}_{MM}.npz. With `fetch`, raw hindcast kinds that are not on disk are
    pulled (Actions only, sequentially — one CDS key, never parallel); with `discard_raw`, a raw kind is
    deleted as soon as no remaining unit needs it, so a runner's disk holds one start month's raw kind at a
    time. Stops starting new fetches after `budget_min`. Returns the units still missing."""
    t0 = time.time()
    month = f"{int(month):02d}"
    reg = raw_registry()
    todo = [(u, m, spec) for u, (m, spec) in units("REF_UNITS").items()
            if (not only or u in only) and (force or not hc_ref(u, month).exists())]
    left = []
    for i, (u, m, spec) in enumerate(todo):
        raws = spec["raw"]
        absent = [r for r in raws if not raw_present(r, month, reg)]
        if absent and not fetch:
            print(f"  {u} {month}: raw {absent} not on disk — skipped", flush=True); left.append(u); continue
        if absent and budget_min is not None and (time.time() - t0) / 60 > budget_min:
            print(f"  {u} {month}: time budget spent — left for the next run", flush=True); left.append(u); continue
        ok = True
        for r in absent:
            guard_fetch(f"hindcast {r} {month}")
            print(f"  fetching raw {r} for start month {month} …", flush=True)
            try:
                ok &= bool(reg[r][1](month)) and raw_present(r, month, reg)
            except Exception as e:                                      # noqa: BLE001 — try the next unit
                print(f"    raw {r} {month}: fetch FAILED ({str(e)[:200]})", flush=True); ok = False
        if not ok:
            print(f"  {u} {month}: raw fetch failed — skipped", flush=True); left.append(u); continue
        print(f"  deriving {u} {month} ({m}) …", flush=True)
        try:
            spec["fn"](month)
        except Exception as e:                                          # noqa: BLE001 — one unit must not sink the rest
            print(f"  {u} {month}: derive FAILED ({str(e)[:200]})", flush=True); left.append(u)
        if discard_raw:
            later = {r for (_, _, s) in todo[i + 1:] for r in s["raw"]}
            for r in raws:
                if r not in later:
                    for p in reg[r][0](month):
                        if p.exists():
                            p.unlink(); print(f"    discarded raw {p.name}", flush=True)
    return left


def derive_issue(ym: str, only=None) -> None:
    for u, (m, fn) in units("ISSUE_UNITS").items():
        if only and u not in only:
            continue
        print(f"  issue summary {u} {ym} ({m}) …", flush=True)
        try:
            fn(ym)
        except Exception as e:                                          # noqa: BLE001
            print(f"  {u} {ym}: FAILED ({str(e)[:200]})", flush=True)


def derive_obs(only=None) -> None:
    for u, (m, fn) in units("OBS_UNITS").items():
        if only and u not in only:
            continue
        print(f"  observed reference {u} ({m}) …", flush=True)
        fn()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["hindcast", "issue", "obs", "missing", "ls"])
    ap.add_argument("--month"); ap.add_argument("--issue")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--fetch", action="store_true", help="pull absent raw hindcast kinds (Actions only)")
    ap.add_argument("--discard-raw", action="store_true")
    ap.add_argument("--budget-min", type=float, default=None)
    ap.add_argument("--force", action="store_true", help="rebuild tables that exist")
    a = ap.parse_args(argv)
    if a.cmd == "hindcast":
        left = derive_hindcast(a.month, a.fetch, a.discard_raw, a.budget_min, a.only, a.force)
        print(f"hindcast tables {a.month}: {len(left)} missing" + (f" {left}" if left else ""), flush=True)
        return 0
    if a.cmd == "issue":
        derive_issue(a.issue, a.only); return 0
    if a.cmd == "obs":
        derive_obs(a.only); return 0
    if a.cmd == "missing":
        left = missing(a.month, a.only)
        print(" ".join(left)); return 1 if left else 0
    tot = 0
    for p in sorted(REF.glob("*.npz")):
        tot += p.stat().st_size; print(f"{p.stat().st_size / 1e6:9.2f} MB  {p.name}")
    print(f"{tot / 1e6:9.1f} MB  total in {REF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
