#!/usr/bin/env python3
"""Weather regimes of one ensemble run, computed in its ensmaps job (ensmaps.py) from the 500 hPa members it already
holds: no download of its own. The labels follow the shared regime code exactly (scripts/regimes/regime_core.py: the
four regimes of each sector and season from NCEP R1 1991-2020, every valid day classified with the set of its own
season, "no regime" below a pattern correlation of 0.25, the 5-day running mean along the lead) and the drift rule of
the ECMWF regime producer (scripts/regimes/regimes_ecmwf.py: no model has a reforecast in this job, so each run's
ensemble-mean zonal-mean anomaly is archived and the lead-dependent drift - lead-L forecast less the day-1 forecast of
the run started on its valid day, over past runs - is removed, shrunk n / (n + 10), at the leads with >= 10 verified
runs).

  field     each member's daily mean (06-24 UTC) 500 hPa height averaged onto the 2.5-deg NCEP grid
            (regime_core.box25), less the ERA5 1991-2020 day-of-year normal (scripts/regimes/data/era5_z500_clim.npz)
  drift     the archive of this job's 00Z runs (assets/regimes/data/ensmaps_drift_<model>.json, 60 runs; the IFS-ENS
            and AIFS-ENS archives start from the ECMWF producer's, the same quantity); a 12Z run takes the drift of
            the same lead day. Lead day 16 (GEFS, GEPS) has no verified pairs and is used as it comes.
  labels    regime_core.members_record -> assets/regimes/data/ensmaps_members_<model>.json, the format every regime
            producer writes, so the multi-model view can read it (it carries `cycle`; a 12Z run's day 1 is the day
            after its start)

What the page shows, and the tests that gate it:
  timeline   per sector and valid day, how many members are in each regime (and in none). A share is marked
             significant where it differs from that regime's 1991-2020 frequency within +-15 days of the date
             (regime_core.climatology): the 95 % interval of [share - normal], members resampled with replacement and
             the R1 years resampled (both 1000 times), excludes 0 - the multi-model view's test, per day
  change     per sector, regime and day both runs hold: the share less the previous run's; significant by Fisher's
             exact test on the two runs' member counts with Benjamini-Hochberg at FDR_Q over the sector's table
  scenarios  per span (the periods take the label of their middle day: the 5-day running mean that is the period's
             own mean) and sector, for each regime holding at least SCEN_MIN members: the mean anomaly of those
             members in each field on the 0.5-deg grid, gated as the maps are (a sign test at 5 % on that regime's
             member count; fewer than SCEN_MIN members can never pass one)
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np

import ensmaps as E
import regime_core as RC
import regimes_ecmwf as RE

MIN_MEMBERS = 10  # members complete on every day the labels need (as regimes_ecmwf)
SCEN_MIN = 6  # the fewest members a two-sided 5 % sign test can be passed with (6 of 6: p = 0.031)
BOOT = 1000
FDR_Q = 0.10
SEED = 23


def drift_archive(out_dir: Path, model: str) -> tuple[Path, dict]:
    p = out_dir / f"ensmaps_drift_{model}.json"
    for q in (p, out_dir / f"regimes_drift_{model}.json"):  # a new archive starts from the ECMWF producer's
        try:
            return p, json.loads(q.read_text())
        except Exception:  # noqa: BLE001
            continue
    return p, {"runs": {}}


def labels(mod, cycle: str, init, dates: list, Z25: np.ndarray, ok: np.ndarray, out_dir: Path):
    """-> ({sector: labels (days, members)}, the members' indices, the drift note) or None with too few members."""
    mem = np.flatnonzero(ok.all(1))
    if mem.size < MIN_MEMBERS:
        E.log(f"::warning::{mod.label}: only {mem.size} members complete on every day - no regimes this run")
        return None
    nd = len(dates)
    an = np.full((nd, mem.size, 73, 144), np.nan)
    an[:, :, :37] = Z25[mem].transpose(1, 0, 2, 3)[:, :, :37] - RE.era5_clim(dates)[:, None]
    arch_p, arch = drift_archive(out_dir, mod.key)
    first = dates[0]
    if cycle.endswith("00"):  # the archive holds 00Z runs: day d valid on init + d - 1, as the drift table reads it
        with np.errstate(invalid="ignore"):
            mean = an.mean(1)
        arch["runs"][f"{first:%Y%m%d}"] = {
            sk: np.round(RE.zonal_profile(mean[: RE.NDAYS], sk), 1).tolist() for sk in RC.SECTORS
        }
        arch["runs"] = {k: arch["runs"][k] for k in sorted(arch["runs"])[-RE.DRIFT_KEEP :]}
        arch["note"] = (
            "per 00Z run: ensemble-mean 500 hPa anomaly (m, vs ERA5 1991-2020) averaged over each regime sector's "
            "longitudes, by lead day (rows, day 1 first) and latitude (80N -> 20N, 2.5 deg); ensmaps_regimes.py"
        )
        arch_p.parent.mkdir(parents=True, exist_ok=True)
        arch_p.write_text(json.dumps(arch, separators=(",", ":")))
    # pairs verified before this run's first day only (a 12Z run's first day is the day after its start)
    table = RE.drift_table(arch, first)
    out, applied = {}, {}
    ref = RC.Ref()
    for sk, spec in RC.SECTORS.items():
        sec, _, _ = RC.sector_cut(an, spec)
        dr = table.get(sk)
        if dr is not None:
            drift, n = dr
            use = n >= RE.DRIFT_MIN
            w = np.zeros((nd, drift.shape[1]))
            m = min(nd, drift.shape[0])
            w[:m] = ((n / (n + RE.DRIFT_K))[:, None] * drift * use[:, None])[:m]
            sec = sec - w[:, None, :, None]
            applied[sk] = [int(i + 1) for i in np.flatnonzero(use) if i < nd]
        v = RC.running_partial(sec, RC.SMOOTH, axis=0)
        lab, _ = ref.classify_dated(v.reshape(nd * mem.size, *v.shape[2:]), np.repeat(dates, mem.size), sk)
        out[sk] = lab.reshape(nd, mem.size)
    note = (
        "lead-dependent drift removed (zonal mean by latitude, shrunk n/(n+10)) at the leads with >= 10 verified runs: "
        + "; ".join(f"{sk} days {v}" for sk, v in applied.items() if v)
        if any(applied.values())
        else f"no drift correction yet ({len(arch['runs'])} archived 00Z runs; it needs 10 verified pairs per lead)"
    )
    rec = RC.members_record(
        mod.key,
        mod.label,
        init.date(),
        mem.size,
        dates,
        out,
        notes=f"{mod.source}; daily means 06-24 UTC; anomalies vs ERA5 1991-2020; {note}",
        max_age_days=2,
    )
    rec["cycle"] = cycle
    if cycle.endswith("12"):
        rec["valid_convention"] = "day d valid on the UTC day init + d (a 12Z run's first full day is the next day)"
    rec["drift"] = {"applied": applied, "note": note}
    (out_dir / f"ensmaps_members_{mod.key}.json").write_text(json.dumps(rec))
    E.log(f"  regimes: {mem.size} members; {note}")
    return out, mem, note


def vs_normal(ref, sk: str, fl: np.ndarray, dates: list):
    """Family slots fl (days, members) -> (the R1 normal (F+1, days), significant (F+1, days)): the 95 % interval of
    share - normal, members and R1 years resampled, excludes 0."""
    C, B = RC.climatology(ref, sk, dates, n_boot=BOOT)
    F1 = C.shape[0]
    idx = np.random.default_rng(SEED).integers(0, fl.shape[1], (BOOT, fl.shape[1]))
    xs = fl[:, idx]  # (days, boot, members)
    PB = np.stack([(xs == f).mean(2) for f in range(F1)], axis=1).transpose(2, 1, 0)  # (boot, F+1, days)
    d = PB - B
    lo, hi = np.percentile(d, 2.5, 0), np.percentile(d, 97.5, 0)
    return C, (lo > 0) | (hi < 0)


def vs_previous(counts: np.ndarray, n: int, dates: list, prev: dict | None):
    """This run's member counts (F+1, days) against the previous run's for the dates both hold -> {cycle, dp (F+1,
    days; None where the previous run lacks the date), sig} or None."""
    from quantmet.stats import benjamini_hochberg
    from scipy.stats import fisher_exact

    if not prev:
        return None
    pv = {d: i for i, d in enumerate(prev["valid"])}
    cols = [(j, pv[f"{d:%Y-%m-%d}"]) for j, d in enumerate(dates) if f"{d:%Y-%m-%d}" in pv]
    if not cols:
        return None
    c0, n0 = np.array(prev["count"]), int(prev["n"])
    dp = np.full(counts.shape, np.nan)
    p = np.full(counts.shape, np.nan)
    for j, i in cols:
        for f in range(counts.shape[0]):
            a, b = int(counts[f, j]), int(c0[f, i])
            dp[f, j] = a / n - b / n0
            p[f, j] = fisher_exact([[a, n - a], [b, n0 - b]])[1]
    sig = np.zeros(p.shape, bool)
    fam = np.isfinite(p)
    sig[fam] = benjamini_hochberg(p[fam], alpha=FDR_Q)
    return {
        "dp": [[None if not np.isfinite(x) else round(float(x), 3) for x in row] for row in dp],
        "sig": sig.astype(int).tolist(),
    }


def scenarios(spans, fl_by_sector: dict, mem: np.ndarray, F: dict, high50: np.ndarray, W) -> dict:
    """The member-mean anomaly maps of each regime with at least SCEN_MIN members, per span and sector."""
    out: dict = {}
    for name, k0, k1 in spans:
        kl = k0 if k0 == k1 else min(k0 + 2, k1)  # the period's middle day: its 5-day mean is the period's
        for sk, fl in fl_by_sector.items():
            cards = []
            for fi in range(len(RC.FAMILIES[sk])):
                sel = mem[fl[kl] == fi]
                if sel.size < SCEN_MIN:
                    continue
                card = {"f": fi, "n": {}, "k": {}}
                for f, fd in F.items():
                    if fd.N50 is None:
                        continue
                    s = sel[fd.done[sel, k0 : k1 + 1].all(1)]
                    if s.size < SCEN_MIN:
                        continue
                    norm = fd.N50[k0 : k1 + 1].mean(0)
                    x = fd.S[s, k0 : k1 + 1].mean(1) - norm[None]
                    m, agree = E.agreeing(x, np.arange(s.size))
                    W.put(
                        f"{f}/r/{sk}{fi}/{name}", E.frame(f, m, agree, m + norm, high50 if f == "msl" else None), False
                    )
                    card["n"][f], card["k"][f] = int(s.size), E.sign_k(int(s.size))
                if card["n"]:
                    cards.append(card)
            if cards:
                out.setdefault(name, {"day": kl + 1})[sk] = cards
    return out


def run(mod, cycle: str, init, days: list, Z25: np.ndarray, F: dict, high50: np.ndarray, W, out_dir: Path,
        before: dict | None) -> dict | None:  # fmt: skip
    """The manifest's `regimes` block for this run (labels, timeline, change from `before` - the previous run's
    manifest entry - and scenario maps), or None when too few members are complete."""
    dates = [d.date() for d, _ in days]
    got = labels(mod, cycle, init, dates, Z25, F["z500"].done, out_dir)
    if got is None:
        return None
    labs, mem, note = got
    ref = RC.Ref()
    seasons = tuple(dict.fromkeys(RC.season_of_date(d) for d in dates))
    sectors, fls = {}, {}
    for sk, lab in labs.items():
        fl = RC.family_of_labels(ref, sk, lab, dates)
        fls[sk] = fl
        F1 = len(RC.FAMILIES[sk]) + 1
        counts = np.stack([(fl == f).sum(1) for f in range(F1)])
        C, sig = vs_normal(ref, sk, fl, dates)
        prev = ((before or {}).get("regimes") or {}).get("sectors", {}).get(sk)
        sectors[sk] = {
            "label": RC.SECTORS[sk]["label"],
            # a family with no regime in the seasons of the run's days (the warm Hudson Bay ridge in winter): None
            "names": [
                ref.family_name(sk, f, seasons)
                if f == F1 - 1 or any(RC.FAMILIES[sk][f][s] is not None for s in seasons)
                else None
                for f in range(F1)
            ],
            "colors": ref.family_colors(sk),
            "valid": [f"{d:%Y-%m-%d}" for d in dates],
            "n": int(mem.size),
            "count": counts.astype(int).tolist(),
            "clim": np.round(C, 3).tolist(),
            "sig": sig.astype(int).tolist(),
            "change": vs_previous(counts, int(mem.size), dates, prev),
        }
        if sectors[sk]["change"] is not None:
            sectors[sk]["change"]["cycle"] = before["cycle"]
    t0 = dt.datetime.now()
    scen = scenarios(E.spans(len(days)), fls, mem, F, high50, W)
    ncard = sum(len(c) for v in scen.values() for k, c in v.items() if k != "day")
    E.log(f"  regime scenarios: {ncard} cards in {(dt.datetime.now() - t0).total_seconds():.0f} s")
    return {
        "members": int(mem.size),
        "drift": note,
        "record": f"assets/regimes/data/ensmaps_members_{mod.key}.json",
        "min_members": SCEN_MIN,
        "sectors": sectors,
        "scenarios": scen,
    }
