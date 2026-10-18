#!/usr/bin/env python3
"""Skill of the jet-branch forecasts (jet_branches_fc.py) on archived IFS-ENS runs against ERA5: the evidence that
gates the jet-branch view of the circulation page (what the forecast shows: reference/jet_branches_skill.json).

ARCHIVE. Every other 00 UTC IFS-ENS run of the cold seasons from the Google Cloud mirror, perturbed members 1..n,
12-hourly to day 10, identified as the live forecast is (jet_branches_fc.run); ERA5 at 00 and 12 UTC from ARCO,
identified as the reference is (jet_branches_hindcast.scan). Leads valid after 31 March are not verified: the product
drops them. Runs are labelled by the year of their season's January.

STATISTICS of each run (aggregate), for each lead, branch and area (10-80N, the western North Pacific box, North
America), cos(lat)-weighted:
  W[k], O[k]   the weight of the columns where k of the n members have the branch within 300 km, and the part of it
               where ERA5 had it within 300 km: the Brier score of any recalibration of the share k/n follows from them
  normal       the Brier score of the ERA5 1991-2020 normal for the date (what the maps' contours show)
  area         the members' mean identified area and ERA5's: the drift by lead
and for each region the share of members with a superposition event, ERA5's outcome and the normal rate.

TESTS. Brier skill against the normal, BSS = 1 - sum BS / sum BS_normal, by lead day for the maps (10-80N) and by lead
group for the event odds. Intervals and one-sided p-values by bootstrap over whole weeks of start dates (runs of one
week share valid times), BOOT resamples. A recalibration - the training season's reliability, the observed frequency
at each member count made monotone - is fitted on one season and scored on the other, both ways (leave one season
out), and adopted for a product-lead only where its out-of-season Brier score is lower than the raw one with a p-value
that survives Benjamini-Hochberg at FDR_Q over all product-leads; the live product then applies the table fitted on
all seasons. The scores of the adopted forecast (recalibrated or raw) are then tested for BSS > 0, again with
Benjamini-Hochberg at FDR_Q over all product-leads. A product shows the passing leads from its shortest one up to the
first that fails.

    python jet_branches_verify.py plan --seasons 2025,2026 --every 2 --chunk 8 --github-output
    python jet_branches_verify.py runs --inits 20250105,20250107 --members 10 --out-dir parts
    python jet_branches_verify.py verify --runs 'parts/*.npz' --scans 'scans/*.npz' --clim jet_branches_clim.nc \
        --out-dir out [--json ../data/reference/jet_branches_skill.json]
"""

from __future__ import annotations

import pyproj  # noqa: F401  before eccodes/xarray: eckit's bundled libproj otherwise clashes (double free at exit)

import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import jet_branches as JB
import jet_branches_fc as FC
import jet_branches_hindcast as HC

AREAS = ("nh", *JB.REGIONS)  # the maps are scored over 10-80N; the regions are reported alongside
LEAD_DAYS = tuple(range(11))
EVENT_GROUPS = {"0-2": (0, 1, 2), "3-5": (3, 4, 5), "6-10": (6, 7, 8, 9, 10)}  # lead days pooled for the rare events
BOOT = 2000
FDR_Q = 0.10
MIN_EVENTS = 10  # an event lead group is tested only with at least this many ERA5 event analyses
SKILL = FC.REF / "jet_branches_skill.json"


def season_of(ts) -> int:
    ts = pd.Timestamp(ts)
    return ts.year + 1 if ts.month >= 11 else ts.year


# ── plan and runs ───────────────────────────────────────────────────────────────────────────────────────────────────
def season_inits(season, every):
    return [f"{d:%Y%m%d}" for d in pd.date_range(f"{season - 1}-11-01", f"{season}-03-31", freq=f"{every}D")]


def plan(seasons, every, inits, chunk):
    """-> (chunks of inits for the matrix, {season: [first day, last day]} of the ERA5 each season's runs need)."""
    inits = sorted(inits or [i for s in seasons for i in season_inits(s, every)])
    chunks = [inits[i : i + chunk] for i in range(0, len(inits), chunk)]
    windows = {}
    for i in inits:
        t = pd.Timestamp(i)
        s = season_of(t)
        end = min(t + pd.Timedelta(days=10), pd.Timestamp(f"{s}-03-31"))
        w = windows.setdefault(s, [t, end])
        w[0], w[1] = min(w[0], t), max(w[1], end)
    return chunks, {s: [f"{a:%Y-%m-%d}", f"{b:%Y-%m-%d}"] for s, (a, b) in windows.items()}


def runs(inits, members, last_step, out_dir, workers=3):
    """The archived runs, as the forecast fetches them -> out_dir/fc_<init>.npz; a run the mirror lacks is skipped."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in inits:
        t0 = time.time()
        try:
            FC.run(i, last_step, out_dir / f"fc_{i}.npz", workers=workers, members=members)
        except SystemExit as e:
            print(f"::warning::{i}: {e}", flush=True)
            continue
        print(f"{i}: {time.time() - t0:.0f} s", flush=True)


# ── per-run statistics ──────────────────────────────────────────────────────────────────────────────────────────────
_ERA = {}


def _era(scan_paths):
    """{valid time: (raw masks, within-300-km masks)} of the scans, loaded once per process."""
    if not _ERA:
        for p in scan_paths:
            times, packed, _ = HC.load_scan(p)
            for t, x in zip(times, packed):
                m = HC.unpack(x)
                _ERA[t] = (m, JB.near(m))
    return _ERA


def aggregate(job):
    """One run against ERA5 -> its statistics (see the module docstring), leads with all members complete only."""
    fc_path, scan_paths, clim_path, members = job
    era = _era(scan_paths)
    init, steps, masks, _ = FC.load_run(fc_path)
    valid = [init + pd.Timedelta(hours=int(s)) for s in steps]
    keep = [
        i
        for i, v in enumerate(valid)
        if v in era and masks[i].shape[0] == members and JB.season_slot(np.array([v.to_datetime64()]))[0] >= 0
    ]
    if not keep:
        return None
    vt = np.array([valid[i].to_datetime64() for i in keep])
    normal = FC.map_normals(clim_path, vt)
    thr = FC.thresholds()
    ev_normal = FC.event_normals(vt, thr)
    w = np.broadcast_to(np.cos(np.deg2rad(JB.LAT))[:, None], (JB.LAT.size, JB.LON.size))
    inside = {"nh": np.ones_like(w, bool), **{r: JB.region_mask(JB.LAT, JB.LON, r) for r in JB.REGIONS}}
    nl, nb, na, nk = len(keep), len(JB.BRANCHES), len(AREAS), members + 1
    out = dict(
        init=f"{init:%Y%m%d}",
        season=season_of(init),
        lead_h=np.array([int(steps[i]) for i in keep]),
        W=np.zeros((nl, nb, na, nk)),
        O=np.zeros((nl, nb, na, nk)),
        BN=np.zeros((nl, nb, na)),
        area_fc=np.zeros((nl, nb, na)),
        area_era=np.zeros((nl, nb, na)),
        ev_share=np.zeros((nl, len(JB.REGIONS))),
        ev_obs=np.zeros((nl, len(JB.REGIONS))),
        ev_normal=np.zeros((nl, len(JB.REGIONS))),
    )
    for j, i in enumerate(keep):
        m = masks[i]
        k = JB.near(m).sum(axis=0)  # (3, LAT, LON) member counts within 300 km
        raw, near = era[valid[i]]
        for b in range(nb):
            o = near[b]
            for a, area in enumerate(AREAS):
                sel = inside[area]
                ww = w[sel]
                out["W"][j, b, a] = np.bincount(k[b][sel], weights=ww, minlength=nk)
                out["O"][j, b, a] = np.bincount(k[b][sel], weights=ww * o[sel], minlength=nk)
                out["BN"][j, b, a] = np.sum(ww * (normal[j, b][sel] - o[sel]) ** 2)
                out["area_fc"][j, b, a] = np.mean([np.sum(ww * x[b][sel]) for x in m])
                out["area_era"][j, b, a] = np.sum(ww * raw[b][sel])
        share = FC.member_events(m, thr)
        for r, region in enumerate(JB.REGIONS):
            out["ev_share"][j, r] = share[region]
            out["ev_obs"][j, r] = any(g[0] >= thr[region] for g in JB.groups_of(raw[2], inside[region]))
            out["ev_normal"][j, r] = ev_normal[region][j]
    return out


# ── scores ──────────────────────────────────────────────────────────────────────────────────────────────────────────
def brier_levels(W, O, p):
    """Brier sums from level weights W[..., k], observed parts O[..., k] and the forecast p[..., k] at each level."""
    return np.sum(p**2 * W - 2 * p * O + O, axis=-1)


def pav(y, w):
    """Weighted least-squares non-decreasing fit (pool adjacent violators); levels without weight are interpolated."""
    y, w = np.asarray(y, float), np.asarray(w, float)
    idx = np.flatnonzero(w > 0)
    if not idx.size:
        return np.linspace(0, 1, y.size)
    blocks = [[y[i] * w[i], w[i], [i]] for i in idx]
    out = []
    for blk in blocks:
        out.append(blk)
        while len(out) > 1 and out[-2][0] / out[-2][1] > out[-1][0] / out[-1][1]:
            a = out.pop()
            out[-1] = [out[-1][0] + a[0], out[-1][1] + a[1], out[-1][2] + a[2]]
    fit = np.full(y.size, np.nan)
    for s, ww, ii in out:
        fit[ii] = s / ww
    x = np.arange(y.size)
    return np.interp(x, x[idx], fit[idx])


def by_day(lead_h, arr):
    """Per-lead arrays (nlead, ...) of one run -> per lead day (11, ...) sums."""
    out = np.zeros((len(LEAD_DAYS), *arr.shape[1:]))
    for j, h in enumerate(lead_h):
        out[int(h) // 24] += arr[j]
    return out


def collect(stats):
    """Per-run statistics -> arrays over runs with lead days summed: maps (run, day, branch, area[, level]), the
    events' ERA5 outcomes and normal Brier sums (run, day, region)."""
    keys = ("W", "O", "BN", "area_fc", "area_era", "ev_obs", "ev_normal2")
    acc = {k: [] for k in keys}
    for s in stats:
        lh = s["lead_h"]
        acc["W"].append(by_day(lh, s["W"]))
        acc["O"].append(by_day(lh, s["O"]))
        acc["BN"].append(by_day(lh, s["BN"]))
        acc["area_fc"].append(by_day(lh, s["area_fc"]))
        acc["area_era"].append(by_day(lh, s["area_era"]))
        acc["ev_obs"].append(by_day(lh, s["ev_obs"]))
        acc["ev_normal2"].append(by_day(lh, (s["ev_normal"] - s["ev_obs"]) ** 2))  # the normal's event Brier sums
    return {k: np.stack(v) for k, v in acc.items()}


def event_levels(stats, members):
    """Per run, the events' level weights and observed parts by lead day and region: W (run, day, region, k), O."""
    W, O = [], []
    for s in stats:
        k = np.rint(s["ev_share"] * members).astype(int)  # (lead, region)
        w = np.zeros((len(LEAD_DAYS), len(JB.REGIONS), members + 1))
        o = np.zeros_like(w)
        for j, h in enumerate(s["lead_h"]):
            for r in range(len(JB.REGIONS)):
                w[int(h) // 24, r, k[j, r]] += 1
                o[int(h) // 24, r, k[j, r]] += s["ev_obs"][j, r]
        W.append(w)
        O.append(o)
    return np.stack(W), np.stack(O)


def blocks_of(stats):
    """Bootstrap block per run: its season and the week of its start date within the season."""
    lab = []
    for s in stats:
        week = (pd.Timestamp(s["init"]) - pd.Timestamp(year=s["season"] - 1, month=11, day=1)).days // 7
        lab.append(f"{s['season']}-{week:02d}")
    names, idx = np.unique(lab, return_inverse=True)
    return idx, names.size


def boot_counts(nblock, rng):
    """(BOOT, nblock) how often each block is drawn in each resample."""
    draws = rng.integers(0, nblock, size=(BOOT, nblock))
    return np.stack([np.bincount(d, minlength=nblock) for d in draws])


def block_sums(x, block, nblock):
    """(run, ...) -> (block, ...) sums."""
    out = np.zeros((nblock, *x.shape[1:]))
    np.add.at(out, block, x)
    return out


def bh(p, q=FDR_Q):
    """Benjamini-Hochberg: which of the p-values are rejected at false discovery rate q."""
    p = np.nan_to_num(np.asarray(p, float), nan=1.0)
    order = np.argsort(p)
    ranked = p[order] * p.size / np.arange(1, p.size + 1)
    passed = np.zeros(p.size, bool)
    below = np.flatnonzero(ranked <= q)
    if below.size:
        passed[order[: below.max() + 1]] = True
    return passed


def skill_test(fc, ref, block, nblock, counts):
    """Per-run Brier sums of a forecast and a reference, (run, ...) -> skill 1 - sum fc / sum ref, its 90 % interval
    and the one-sided p-value of skill <= 0, by block bootstrap."""
    bf, br = block_sums(fc, block, nblock), block_sums(ref, block, nblock)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = 1 - fc.sum(0) / ref.sum(0)
        bs = 1 - np.tensordot(counts, bf, axes=(1, 0)) / np.tensordot(counts, br, axes=(1, 0))
    p = np.where(np.isfinite(s), np.mean(~(bs > 0), axis=0), 1.0)  # a resample without a reference counts as no skill
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # leads without data: all-NaN resamples
        return s, np.nanpercentile(bs, 5, axis=0), np.nanpercentile(bs, 95, axis=0), p


def calibrated(W, O, season, members, train=None):
    """Recalibrated Brier sums per run, (run, day, ..., k) inputs: each run scored with the table fitted on the runs of
    the other seasons (leave one season out). With train=all, the table fitted on every run (the live product's) is
    returned instead, (day, ..., k)."""
    shape = W.shape[1:]

    def fit(sel):
        Ws, Os = W[sel].sum(0), O[sel].sum(0)
        flat_w, flat_o = Ws.reshape(-1, shape[-1]), Os.reshape(-1, shape[-1])
        with np.errstate(invalid="ignore", divide="ignore"):
            t = [pav(np.where(w > 0, o / w, 0.0), w) for w, o in zip(flat_w, flat_o)]
        return np.array(t).reshape(shape)

    if train == "all":
        return fit(np.ones(W.shape[0], bool))
    out = np.zeros(W.shape[:-1])
    for s in np.unique(season):
        sel = season == s
        out[sel] = brier_levels(W[sel], O[sel], fit(~sel)[None])
    return out


def verify(run_paths, scan_paths, clim_path, out_dir, json_path=None, members=10, workers=4, seed=1):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(p, scan_paths, clim_path, members) for p in sorted(run_paths)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        stats = [s for s in ex.map(aggregate, jobs) if s is not None]
    print(f"{len(stats)} runs scored in {time.time() - t0:.0f} s", flush=True)
    if not stats:
        raise SystemExit("no run could be scored")
    season = np.array([s["season"] for s in stats])
    block, nblock = blocks_of(stats)
    counts = boot_counts(nblock, np.random.default_rng(seed))
    c = collect(stats)
    levels = np.arange(members + 1) / members

    # maps over 10-80N: raw, recalibrated out of season, the normal
    W, O = c["W"][:, :, :, 0], c["O"][:, :, :, 0]  # (run, day, branch, k)
    bs_raw = brier_levels(W, O, levels)
    bs_cal = calibrated(W, O, season, members)
    bs_norm = c["BN"][:, :, :, 0]
    # events by lead group
    eW, eO = event_levels(stats, members)  # (run, day, region, k)
    gW = np.stack([eW[:, list(d)].sum(1) for d in EVENT_GROUPS.values()], axis=1)  # (run, group, region, k)
    gO = np.stack([eO[:, list(d)].sum(1) for d in EVENT_GROUPS.values()], axis=1)
    ev_raw = brier_levels(gW, gO, levels)
    ev_cal = calibrated(gW, gO, season, members)
    ev_norm = np.stack([c["ev_normal2"][:, list(d)].sum(1) for d in EVENT_GROUPS.values()], axis=1)

    # 1. adopt the recalibration where it beats the raw forecast out of season
    imp_m = skill_test(bs_cal, bs_raw, block, nblock, counts)
    imp_e = skill_test(ev_cal, ev_raw, block, nblock, counts)
    adopt_flat = bh(np.concatenate([imp_m[3].ravel(), imp_e[3].ravel()]))
    adopt_m = adopt_flat[: imp_m[3].size].reshape(imp_m[3].shape) & (imp_m[0] > 0)
    adopt_e = adopt_flat[imp_m[3].size :].reshape(imp_e[3].shape) & (imp_e[0] > 0)
    # 2. skill of the adopted forecast against the normal
    fin_m = np.where(adopt_m[None], bs_cal, bs_raw)
    fin_e = np.where(adopt_e[None], ev_cal, ev_raw)
    sk_m = skill_test(fin_m, bs_norm, block, nblock, counts)
    sk_e = skill_test(fin_e, ev_norm, block, nblock, counts)
    n_ev = np.stack([c["ev_obs"][:, list(d)].sum((0, 1)) for d in EVENT_GROUPS.values()])  # (group, region)
    sk_e[3][n_ev < MIN_EVENTS] = 1.0  # too few events to judge
    raw_m = skill_test(bs_raw, bs_norm, block, nblock, counts)
    raw_e = skill_test(ev_raw, ev_norm, block, nblock, counts)
    pass_flat = bh(np.concatenate([sk_m[3].ravel(), sk_e[3].ravel()]))
    pass_m = pass_flat[: sk_m[3].size].reshape(sk_m[3].shape) & (sk_m[0] > 0)
    pass_e = pass_flat[sk_m[3].size :].reshape(sk_e[3].shape) & (sk_e[0] > 0)
    # 3. drift: the members' mean identified area against ERA5's, by lead day, branch and area
    af, ae = c["area_fc"], c["area_era"]
    baf, bae = block_sums(af, block, nblock), block_sums(ae, block, nblock)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = af.sum(0) / ae.sum(0)
        bratio = np.tensordot(counts, baf, axes=(1, 0)) / np.tensordot(counts, bae, axes=(1, 0))
    table_all = calibrated(W, O, season, members, train="all")
    etable_all = calibrated(gW, gO, season, members, train="all")

    def shown(passes):
        """Leads shown: from the first passing one up to the first failure after it."""
        idx = np.flatnonzero(passes)
        if not idx.size:
            return None
        end = idx[0]
        while end + 1 < passes.size and passes[end + 1]:
            end += 1
        return [int(idx[0]), int(end)]

    def r3(x):
        return [round(float(v), 3) if np.isfinite(v) else None for v in np.atleast_1d(x)]

    def pct(x, q):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanpercentile(x, q, axis=0)

    doc = dict(
        model=FC.LABEL,
        members=members,
        seasons=sorted(int(s) for s in np.unique(season)),
        runs=len(stats),
        blocks=f"{nblock} weeks of start dates",
        bootstrap=BOOT,
        fdr_q=FDR_Q,
        near_km=JB.NEAR_KM,
        lead_days=list(LEAD_DAYS),
        maps={},
        events={},
        drift={},
    )
    for b, name in enumerate(JB.BRANCHES):
        doc["maps"][name] = dict(
            bss_raw=r3(raw_m[0][:, b]),
            bss_raw_ci=[r3(raw_m[1][:, b]), r3(raw_m[2][:, b])],
            calibrated=[bool(x) for x in adopt_m[:, b]],
            calibration_gain=r3(imp_m[0][:, b]),
            bss=r3(sk_m[0][:, b]),
            bss_ci=[r3(sk_m[1][:, b]), r3(sk_m[2][:, b])],
            p=[round(float(x), 4) for x in sk_m[3][:, b]],
            passed=[bool(x) for x in pass_m[:, b]],
            show=shown(pass_m[:, b]),
            table=[r3(table_all[d, b]) for d in range(len(LEAD_DAYS))],
        )
        doc["drift"][name] = {
            area: dict(
                ratio=r3(ratio[:, b, a]),
                ci=[r3(pct(bratio[:, :, b, a], 5)), r3(pct(bratio[:, :, b, a], 95))],
            )
            for a, area in enumerate(AREAS)
        }
    for r, name in enumerate(JB.REGIONS):
        doc["events"][name] = dict(
            groups=list(EVENT_GROUPS),
            era5_event_analyses=[int(x) for x in n_ev[:, r]],
            bss_raw=r3(raw_e[0][:, r]),
            bss_raw_ci=[r3(raw_e[1][:, r]), r3(raw_e[2][:, r])],
            calibrated=[bool(x) for x in adopt_e[:, r]],
            bss=r3(sk_e[0][:, r]),
            bss_ci=[r3(sk_e[1][:, r]), r3(sk_e[2][:, r])],
            p=[round(float(x), 4) for x in sk_e[3][:, r]],
            passed=[bool(x) for x in pass_e[:, r]],
            show=shown(pass_e[:, r]),
            table=[r3(etable_all[g, r]) for g in range(len(EVENT_GROUPS))],
        )
    text = json.dumps(doc, indent=1) + "\n"
    (out_dir / "jet_branches_skill.json").write_text(text)
    if json_path:
        Path(json_path).write_text(text)
    np.savez_compressed(
        out_dir / "verify_stats.npz",
        season=season,
        block=block,
        W=c["W"],
        O=c["O"],
        BN=c["BN"],
        area_fc=af,
        area_era=ae,
        eW=eW,
        eO=eO,
    )
    report(doc)
    figures(doc, c, members, out_dir)
    return doc


def _f(x):
    return "   n/a" if x is None else f"{x:+.3f}"


def report(doc):
    print(f"{doc['runs']} runs, {doc['blocks']}, members {doc['members']}, seasons {doc['seasons']}")
    for name, m in doc["maps"].items():
        print(f"  maps {name:5s} show {m['show']}")
        for d in LEAD_DAYS:
            print(
                f"    day {d:2d}: BSS raw {_f(m['bss_raw'][d])} [{_f(m['bss_raw_ci'][0][d])}, {_f(m['bss_raw_ci'][1][d])}]"
                f"  adopted {'cal' if m['calibrated'][d] else 'raw'} {_f(m['bss'][d])} [{_f(m['bss_ci'][0][d])}, "
                f"{_f(m['bss_ci'][1][d])}] p {m['p'][d]:.4f} {'PASS' if m['passed'][d] else '-'}"
            )
        print("    area ratio (10-80N): " + " ".join(_f(x) for x in doc["drift"][name]["nh"]["ratio"]))
    for name, e in doc["events"].items():
        print(f"  events {name} (ERA5 event analyses by group {e['era5_event_analyses']}) show {e['show']}")
        for g, grp in enumerate(e["groups"]):
            print(
                f"    days {grp:5s}: BSS raw {_f(e['bss_raw'][g])} [{_f(e['bss_raw_ci'][0][g])}, {_f(e['bss_raw_ci'][1][g])}]"
                f"  adopted {'cal' if e['calibrated'][g] else 'raw'} {_f(e['bss'][g])} p {e['p'][g]:.4f} "
                f"{'PASS' if e['passed'][g] else '-'}"
            )


def figures(doc, c, members, out_dir):
    """Skill by lead, drift by lead and reliability (scratch figures for the check)."""
    plt = FC.mpl()
    days = np.array(LEAD_DAYS)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    for name, col in zip(JB.BRANCHES, ("#2e86c1", "#e8512a", "#4d0f52")):
        m = {k: (np.array(v, float) if k.startswith("bss") else v) for k, v in doc["maps"][name].items()}
        axs[0].plot(days, m["bss"], color=col, lw=2, label=FC.TITLE[name].split(" (")[0])
        axs[0].fill_between(days, m["bss_ci"][0], m["bss_ci"][1], color=col, alpha=0.15)
        axs[0].plot(days, m["bss_raw"], color=col, lw=1, ls=":")
        axs[0].plot(days[np.array(m["passed"])], m["bss"][np.array(m["passed"])], "o", color=col, ms=5)
        ratio = np.array(doc["drift"][name]["nh"]["ratio"], float)
        axs[1].plot(days, ratio, color=col, lw=2, label=FC.TITLE[name].split(" (")[0])
    axs[0].axhline(0, color=FC.INK, lw=0.8)
    axs[0].set_title("Maps: Brier skill against the ERA5 normal (dots: pass)", loc="left", fontsize=11)
    axs[1].axhline(1, color=FC.INK, lw=0.8)
    axs[1].set_title("Drift: members' identified area / ERA5's (10-80N)", loc="left", fontsize=11)
    for k, name in enumerate(JB.REGIONS):
        e = {k_: (np.array(v, float) if k_.startswith("bss") else v) for k_, v in doc["events"][name].items()}
        x = np.arange(len(e["groups"])) + 0.2 * (k - 0.5)
        axs[2].errorbar(
            x,
            e["bss"],
            yerr=np.nan_to_num(np.clip([e["bss"] - e["bss_ci"][0], e["bss_ci"][1] - e["bss"]], 0, None)),
            fmt="o",
            capsize=3,
            label=JB.REGIONS[name]["label"],
        )
        axs[2].set_xticks(range(len(e["groups"])), [f"days {g}" for g in e["groups"]])
    axs[2].axhline(0, color=FC.INK, lw=0.8)
    axs[2].set_title("Event odds: Brier skill against the normal", loc="left", fontsize=11)
    for ax in axs:
        ax.legend(fontsize=8.5, frameon=False)
        ax.grid(color="#e3e3e3", lw=0.6)
    axs[0].set_xlabel("lead (days)")
    axs[1].set_xlabel("lead (days)")
    fig.suptitle(f"Jet branches: {doc['runs']} archived {doc['model']} runs ({doc['members']} members), seasons "
                 f"{doc['seasons']}", x=0.01, ha="left", fontsize=13, fontweight="bold")  # fmt: skip
    fig.tight_layout()
    fig.savefig(out_dir / "verify_skill.png", dpi=100)
    plt.close(fig)
    # reliability of the raw shares, 10-80N, three lead groups
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    p = np.arange(members + 1) / members
    for ax, name in zip(axs, JB.BRANCHES):
        for grp, col in zip(((0, 1), (4, 5, 6), (8, 9, 10)), ("#2e86c1", "#e8512a", "#4d0f52")):
            Wg = c["W"][:, list(grp), JB.BRANCHES.index(name), 0].sum((0, 1))
            Og = c["O"][:, list(grp), JB.BRANCHES.index(name), 0].sum((0, 1))
            ok = Wg > 0
            ax.plot(p[ok], (Og[ok] / Wg[ok]), "o-", color=col, label=f"days {grp[0]}-{grp[-1]}")
        ax.plot([0, 1], [0, 1], color=FC.INK, lw=0.8)
        ax.set_title(f"Reliability: {FC.TITLE[name].split(' (')[0]}", loc="left", fontsize=11)
        ax.set_xlabel("share of members within 300 km")
        ax.set_ylabel("ERA5 frequency within 300 km")
        ax.legend(fontsize=8.5, frameon=False)
        ax.grid(color="#e3e3e3", lw=0.6)
    fig.tight_layout()
    fig.savefig(out_dir / "verify_reliability.png", dpi=100)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("plan")
    s.add_argument("--seasons", default="2025,2026", help="cold seasons by the year of January")
    s.add_argument("--every", type=int, default=2, help="every Nth day's 00 UTC run")
    s.add_argument("--inits", default="", help="JSON list of runs (YYYYMMDD) instead of the seasons'")
    s.add_argument("--chunk", type=int, default=8)
    s.add_argument("--github-output", action="store_true")
    s = sub.add_parser("runs")
    s.add_argument("--inits", required=True, help="comma-separated YYYYMMDD")
    s.add_argument("--members", type=int, default=10)
    s.add_argument("--last-step", type=int, default=240)
    s.add_argument("--out-dir", required=True)
    s = sub.add_parser("verify")
    s.add_argument("--runs", required=True, help="glob of the runs' npz")
    s.add_argument("--scans", required=True, help="glob of the ERA5 scans' npz")
    s.add_argument("--clim", required=True)
    s.add_argument("--out-dir", required=True)
    s.add_argument("--json", help="also write the skill json here (the committed reference)")
    s.add_argument("--members", type=int, default=10)
    a = ap.parse_args()
    if a.cmd == "plan":
        chunks, windows = plan(
            [int(x) for x in a.seasons.split(",")], a.every, json.loads(a.inits) if a.inits else None, a.chunk
        )
        print(f"{sum(map(len, chunks))} runs in {len(chunks)} chunks; ERA5 windows {windows}", flush=True)
        if a.github_output and os.environ.get("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
                fh.write(f"chunks={json.dumps([','.join(x) for x in chunks])}\n")
                fh.write(
                    f"windows={json.dumps([{'season': s, 'start': w[0], 'end': w[1]} for s, w in windows.items()])}\n"
                )
    elif a.cmd == "runs":
        runs(a.inits.split(","), a.members, a.last_step, a.out_dir)
    else:
        verify(glob.glob(a.runs), glob.glob(a.scans), a.clim, a.out_dir, a.json, a.members)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
