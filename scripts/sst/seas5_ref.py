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
from seas5_outlook import CLIM_YEARS, DATA, KINDS, L7_STATE, _retrieve, hc_path  # noqa: E402
import re  # noqa: E402

REF = Path(os.environ["SEAS5_REF_DIR"]).resolve() if os.environ.get("SEAS5_REF_DIR") else DATA / "ref"

# product modules, in build order; each may define REF_UNITS / ISSUE_UNITS / OBS_UNITS / RAW_HC
MODULES = ["seas5_build", "seas5_normals", "seas5_tele", "seas5_pme_regions", "seas5_popT",
           "seas5_extremes_build", "seas5_runs", "seas5_wind", "seas5_snow", "seas5_regimes", "seas5_ssw"]


# ── paths and I/O ────────────────────────────────────────────────────────────
def hc_ref(unit: str, month: str) -> Path:
    return REF / f"hc_{unit}_{int(month):02d}.npz"


def is_ref(unit: str, ym: str) -> Path:
    return REF / f"is_{unit}_{ym}.npz"


def obs_ref(name: str) -> Path:
    return REF / f"obs_{name}.npz"


def _upload_now(month: str, since: float) -> None:
    """2026-10-06: each new hindcast table goes to the reference release as soon as it is built (Actions sets
    SEAS5_REF_UPLOAD_TAG + GH_TOKEN), so a job killed later by its timeout cannot lose finished tables. Best-effort:
    the workflow's own upload step after the tables step uploads anything this missed."""
    tag = os.environ.get("SEAS5_REF_UPLOAD_TAG")
    if not tag or not os.environ.get("GH_TOKEN"):
        return
    import subprocess
    new = [str(p) for p in REF.glob(f"hc_*_{month}.npz") if p.stat().st_mtime >= since - 1]
    if not new:
        return
    r = subprocess.run(["gh", "release", "upload", tag, "--repo", os.environ.get("GITHUB_REPOSITORY", ""), "--clobber", *new],
                       capture_output=True, text=True)
    print(f"    uploaded to release {tag}: {', '.join(Path(x).name for x in new)}" if r.returncode == 0
          else f"    release upload failed (the workflow step retries): {r.stderr.strip()[:200]}", flush=True)


def l7_sibling(path: Path) -> Path:
    """hc_{unit}_{MM}.npz → hc_{unit}_L7_{MM}.npz, the unit's lead-7 extension table."""
    return path.with_name(re.sub(r"_(\d\d)\.npz$", r"_L7_\1.npz", path.name))


def save(path: Path, meta: dict | None = None, **arrays) -> Path:
    """Atomic compressed npz; `meta` (JSON-able) rides along as the __meta__ entry. While a lead-7 table is being
    derived (L7_STATE), an hc_ table goes to its _L7 sibling — the product code is the same, only the hindcast it
    reads (the lead-7 pseudo-raw) and the table it writes differ."""
    if L7_STATE["on"] and path.name.startswith("hc_") and not path.stem.endswith("_L7") and "_L7_" not in path.name:
        path = l7_sibling(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + f".part{os.getpid()}.npz")
    if meta is not None:
        arrays["__meta__"] = np.array(json.dumps(meta))
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
    print(f"    ref {path.name}: {path.stat().st_size / 1e6:.2f} MB", flush=True)
    return path


def _load_one(path: Path) -> dict | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files if k != "__meta__"}
        if "__meta__" in z.files:
            out["meta"] = json.loads(str(z["__meta__"]))
    return out


def load(path: Path) -> dict | None:
    """{name: array} (+ 'meta' dict) or None when the table is not there. A hindcast table whose lead-7 extension
    (hc_{unit}_L7_{MM}.npz) exists comes back MERGED: 7 leads / the fourth season (merge_l7)."""
    out = _load_one(path)
    if out is None or L7_STATE["on"] or not path.name.startswith("hc_") or "_L7" in path.name:
        return out
    ext = _load_one(l7_sibling(path))
    return merge_l7(out, ext, path.name) if ext is not None else out


def merge_l7(base: dict, ext: dict, name: str = "") -> dict:
    """A base hindcast table (leads 1–6 from the CDS monthly hindcast) + its lead-7 extension (the same product code
    run on the lead-7 pseudo-raw hindcast, leads 1–4 empty). Per array:
      * a key only the extension has (the fourth season, the lead-7 lead set, month-7 keys) → taken from it;
      * same shape → the base's (leads 1–6 stay CDS-monthly; the extension's leads 5–6 only serve the new season);
      * one axis 6 → 7 (lead axis) → the base plus the extension's 7th entry;
      * one axis len(base periods) → len(extension periods) (tables with a meta "periods" list) → the extension's
        period order, each period from the base where the base has it.
    meta: lists are unioned (base order first), dicts merged, the rest kept from the base unless new."""
    bm, em = base.get("meta", {}), ext.get("meta", {})
    bp = [tuple(p) for p in bm.get("periods", [])]; ep = [tuple(p) for p in em.get("periods", [])]
    out = dict(base)
    for k, e in ext.items():
        if k == "meta":
            continue
        if k not in base:
            out[k] = e; continue
        b = base[k]
        if not hasattr(b, "shape") or b.shape == e.shape:
            continue
        diff = [i for i in range(b.ndim) if b.ndim == e.ndim and b.shape[i] != e.shape[i]]
        if b.ndim != e.ndim or len(diff) != 1:
            raise ValueError(f"{name}:{k}: cannot merge shapes {b.shape} and {e.shape}")
        ax = diff[0]
        if (b.shape[ax], e.shape[ax]) == (6, 7):
            out[k] = np.concatenate([b, np.take(e, [6], axis=ax)], axis=ax)
        elif bp and ep and b.shape[ax] == len(bp) and e.shape[ax] == len(ep):
            out[k] = np.concatenate([np.take(b, [bp.index(p)], axis=ax) if p in bp else np.take(e, [i], axis=ax)
                                     for i, p in enumerate(ep)], axis=ax)
        else:
            raise ValueError(f"{name}:{k}: axis {ax} {b.shape[ax]} → {e.shape[ax]} is neither leads nor periods")
    meta = dict(bm)
    for k, v in em.items():
        if k not in meta:
            meta[k] = v
        elif isinstance(meta[k], list) and isinstance(v, list):
            meta[k] = list(meta[k]) + [x for x in v if x not in meta[k]]
        elif isinstance(meta[k], dict) and isinstance(v, dict):
            meta[k] = {**meta[k], **v}
    if ep:
        meta["periods"] = [list(p) for p in ep]
    meta["lead7"] = True
    out["meta"] = meta
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


# raw hindcast key → its lead-7 counterpart (seas5_aws pseudo-raw for the monthly kinds; the modules' own k = 7
# chunk for the daily products). A unit whose raw keys all map gets a lead-7 unit {unit}_L7.
def _l7_raw(key: str) -> str | None:
    if key.startswith("m:"):
        import seas5_aws
        return "l7:" + key[2:] if key[2:] in seas5_aws.kinds() else None      # energy: not extended (seas5_lead7)
    return {"x:us": "x7:us", "x:br": "x7:br", "w:us": "w7:us", "sf:na": "sf7:na"}.get(key)


def derive_l7(fn, month: str) -> None:
    """Run a unit's own derivation on the lead-7 pseudo-raw hindcast; its table goes to hc_{unit}_L7_{MM}.npz."""
    L7_STATE["on"] = True
    try:
        fn(month)
    finally:
        L7_STATE["on"] = False


def units(kind: str) -> dict:
    """kind 'REF_UNITS' | 'ISSUE_UNITS' | 'OBS_UNITS' → {unit: (module name, spec)}. REF_UNITS also lists the
    lead-7 units ({unit}_L7, after all base units)."""
    out = {}
    for m in _modules():
        for u, spec in getattr(_mod(m), kind, {}).items():
            if u in out:
                raise ValueError(f"unit {u} declared twice ({out[u][0]}, {m})")
            out[u] = (m, spec)
    if kind == "REF_UNITS" and os.environ.get("SEAS5_LEAD7", "1") != "0":
        l7 = {}
        for u, (m, spec) in out.items():
            raws = [_l7_raw(r) for r in spec["raw"]]
            if not raws or any(r is None for r in raws):
                continue
            if "l7:sfc" in raws and "l7:gl" not in raws:
                raws.insert(raws.index("l7:sfc"), "l7:gl")                   # the sfc pseudo-raw is cut from gl's
            l7[f"{u}_L7"] = (m, dict(raw=raws, fn=(lambda month, f=spec["fn"]: derive_l7(f, month)), l7=True))
        out.update(l7)
    return out


def raw_registry() -> dict:
    """raw key → (paths(month), fetch(month))."""
    reg = {}
    for kind in KINDS:
        reg[f"m:{kind}"] = ((lambda month, k=kind: [hc_path(k, f"{int(month):02d}")]),
                            (lambda month, k=kind: _retrieve(k, CLIM_YEARS, f"{int(month):02d}", hc_path(k, f"{int(month):02d}"))))
        reg[f"l7:{kind}"] = ((lambda month, k=kind: [_pseudo(k, month)]),
                             (lambda month, k=kind: _build_pseudo(k, month)))
    for m in _modules():
        reg.update(getattr(_mod(m), "RAW_HC", {}))
    return reg


def _pseudo(kind: str, month: str) -> Path:
    return DATA / "hindcast" / f"hc_{kind}_{int(month):02d}_L7.npz"


def _build_pseudo(kind: str, month: str) -> bool:
    """The kind's lead-7 pseudo-raw hindcast (seas5_aws: AWS daily + CDS sub-daily); needs icechunk (Python 3.12)."""
    import seas5_aws
    return seas5_aws.build_kind(kind, f"{int(month):02d}") is not None


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
        t_u = time.time()
        try:
            spec["fn"](month)
            _upload_now(month, t_u)
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
