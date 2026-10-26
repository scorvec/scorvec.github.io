#!/usr/bin/env python3
"""The super-ensemble on ens-anomalies.html: the IFS-ENS, AIFS-ENS, GEFS and GEPS runs from the same 00 or 12 UTC start
combined with EQUAL weight per ensemble, a quarter each however many members it has (the 51 AIFS-ENS members count as
much as the 21 of GEPS). Everything is on the 0.5-deg cells on which every ensemble's run statistics are kept
(ensmaps.save_stats, the Actions cache): nothing is fetched, and a super run follows the last of its four runs within
minutes. It runs from 00 and 12 UTC only: GEPS has no 06/18Z run, and IFS-ENS's reach day 5.

Days: those all four runs hold (15 from 00Z, 14 from 12Z; GEFS and GEPS run a day further), in the spans of one
ensemble's maps (days 1-5, 6-10, 11-15 and each day). An ensemble's span counts when at least ensmaps.MIN_SHARE of its
members are complete on its days, as on its own maps; a span or field one of the four lacks is not drawn.

  anomaly  A = sum_m w_m a_m, a_m the ensemble's mean anomaly over its complete members, each against its own normal
           (2 m temperature at its own surface height). Dotted where the members agree on its sign: the weighted share
           of the members with that sign, sum_m w_m k_m / n_m, at least the gate of a two-sided sign test at 5 % - the
           smallest share whose probability, with every member's sign a fair coin, is below 2.5 % (gate(): exact, the
           four binomials combined). The sea-level pressure anomaly is left out where any ensemble's surface is above
           ensmaps.HIGH_M.
  sigma    500 hPa height and sea-level pressure: A over the 1991-2020 standard deviation of the span's averaging
           period (ensmaps_ref.Spread), as for one ensemble
  mean     sum_m w_m mean_m
  change   from the super run 12, 24, 36 or 48 h earlier over the days both hold: d = sum_m w_m (mean_m - mean0_m), its
           variance sum_m w_m^2 (s_m^2 / n_m + s0_m^2 / n0_m), Welch's test with Satterthwaite's degrees of freedom
           over the eight terms, Benjamini-Hochberg at ensmaps.FDR_Q. Each super run keeps its weighted mean, the
           variance of that mean and the variance's degrees of freedom per span (--save), so a later one needs only its
           own four runs' statistics.

Files: as one ensemble's (ensmaps.py), assets/models/data/ensmaps/super/<cycle>/<field>/{a,s,c,c24,c36,c48}/<span>.webp
on the 0.5-deg grid, except G of a/ and s/: the weighted share of the members agreeing on the sign x SHARE (255 = not
drawn; the spans' "k" is the gate on the same scale). The manifest (assets/models/ensmaps_super.json) names each
ensemble's run and when it was made, and a super run is made again when one of them is (IFS-ENS's 2 m temperature is
replaced from its 0.1-deg files hours after the rest).

    python scripts/ensmaps/ensmaps_super.py plan --manifests assets/models
    python scripts/ensmaps/ensmaps_super.py run --cycle 2026102500 --stats-dir in --prev-dir prev \
        --save prev/2026102500.npz --site . --lists /tmp/lists
"""

from __future__ import annotations

import argparse
import datetime as dt
import functools
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

import ensmaps as E

ENSEMBLES = ("ifs", "aifs", "gefs", "geps")
W = 1.0 / len(ENSEMBLES)  # each ensemble's weight
HOURS = ("00", "12")
SHARE = 250  # the G code of a share of 1
CHANGES = tuple(c for c in E.CHANGES if c[0] % 12 == 0)  # the super runs 12, 24, 36 and 48 h earlier
SUPER = E.Model("super", "Super-ensemble", sum(E.MODELS[m].members for m in ENSEMBLES), 0, dict.fromkeys(E.FIELDS, E.G50),
                "", None, "IFS-ENS, AIFS-ENS, GEFS and GEPS runs from the same start, a quarter each, on 0.5° cells",
                None, dict.fromkeys(HOURS, (E.LAST_STEP, 0.0)))  # fmt: skip
MIN_N = {m: int(np.ceil(E.MIN_SHARE * E.MODELS[m].members)) for m in ENSEMBLES}  # as each ensemble's own maps
log = E.log


@functools.cache
def gate(ns: tuple[int, ...]) -> float:
    """The smallest share t of the members sharing a sign, each ensemble of ns members weighing 1 / len(ns), for which
    a two-sided sign test passes at 5 %: 2 P(T >= t) < 0.05, T = sum_m k_m / n_m / len(ns), every k_m binomial with
    p = 1/2. With one ensemble, t = ensmaps.sign_k(n) / n."""
    from scipy.stats import binom

    vals, probs = np.zeros(1), np.ones(1)
    for n in ns:
        k = np.arange(n + 1)
        vals = (vals[:, None] + k[None] / (n * len(ns))).ravel()
        probs = (probs[:, None] * binom.pmf(k, n, 0.5)[None]).ravel()
    u, inv = np.unique(np.round(vals, 9), return_inverse=True)
    tail = np.cumsum(np.bincount(inv, probs)[::-1])[::-1]  # P(T >= u)
    return float(u[(u > 0.5) & (2 * tail < 0.05)][0])


def load_run(path: Path, init: pd.Timestamp) -> dict | None:
    """One ensemble's run statistics (ensmaps.save_stats): per field the rows of mean, sd, n, and anom, pos on the
    super-ensemble's spans; the keys of both, and the high ground of its sea-level pressure."""
    if not path.exists():
        return None
    with np.load(path) as z:
        if str(z["init"]) != f"{init:%Y%m%d%H}":
            log(f"  {path}: holds run {z['init']}, not {init:%Y%m%d%H}")
            return None
        out = {
            "keys": {str(k): i for i, k in enumerate(z["keys"])},
            "skeys": {str(k): i for i, k in enumerate(z["skeys"])} if "skeys" in z.files else {},
            "high": z["high_msl"].astype(bool) if "high_msl" in z.files else None,
        }
        for f in E.FIELDS:
            if f"mean_{f}" in z.files:
                out[f] = {k: z[f"{k}_{f}"] for k in ("mean", "sd", "n", "anom", "pos") if f"{k}_{f}" in z.files}
    return out


def load_super(path: Path, init: pd.Timestamp) -> dict | None:
    """A super run's statistics (save_super), with the variance's Satterthwaite denominator D = V^2 / df."""
    if not path.exists():
        return None
    with np.load(path) as z:
        if str(z["init"]) != f"{init:%Y%m%d%H}":
            return None
        out = {"keys": {str(k): i for i, k in enumerate(z["keys"])}}
        for f in E.FIELDS:
            if f"M_{f}" in z.files:
                v = z[f"sv_{f}"].astype(np.float32) ** 2
                out[f] = {"M": z[f"M_{f}"], "V": v, "D": v * v / z[f"df_{f}"].astype(np.float32), "n": z[f"n_{f}"]}
    return out


def combine(runs: dict, f: str, k: str) -> tuple | None:
    """(weighted mean M, its variance V, the variance's Satterthwaite denominator D = sum (w^2 s^2 / n)^2 / (n - 1), the
    members of each ensemble) of the span k; None when an ensemble lacks it or has too few complete members."""
    M = V = D = 0.0
    ns = []
    for m, r in runs.items():
        i = r["keys"].get(k)
        if f not in r or i is None or int(r[f]["n"][i]) < MIN_N[m]:
            return None
        n = int(r[f]["n"][i])
        t = W * W * r[f]["sd"][i].astype(np.float32) ** 2 / n
        M, V, D = M + W * r[f]["mean"][i], V + t, D + t * t / (n - 1)
        ns.append(n)
    return M, V, D, tuple(ns)


def anomaly(runs: dict, f: str, k: str) -> tuple | None:
    """(weighted mean anomaly A, the weighted share of the members above normal) of the span k; None when an ensemble
    has no anomaly for it."""
    A = T = 0.0
    for r in runs.values():
        i, j = r["keys"].get(k), r["skeys"].get(k)
        if "anom" not in r.get(f, {}) or i is None or j is None:
            return None
        A = A + W * r[f]["anom"][j].astype(np.float32)
        T = T + W * r[f]["pos"][j] / float(r[f]["n"][i])
    return A, T


def save_super(path: Path, init: pd.Timestamp, days: list, runs: dict) -> None:
    """Per field and span of ensmaps.stat_spans(cuts=False): M (float32), the square root of V and its degrees of
    freedom V^2 / D (float16), the members (all four ensembles); NaN where the span is not combined."""
    sp = E.stat_spans(len(days), cuts=False)
    keys = [E.key(days, k0, k1) for k0, k1 in sp]
    out = {"init": f"{init:%Y%m%d%H}", "keys": np.array(keys)}
    for f in E.FIELDS:
        rows = [combine(runs, f, k) for k in keys]
        if not any(rows):
            continue
        nan = np.full(E.G50.shape, np.nan, np.float32)
        out[f"M_{f}"] = np.stack([r[0] if r else nan for r in rows]).astype(np.float32)
        out[f"sv_{f}"] = np.stack([np.sqrt(r[1]) if r else nan for r in rows]).astype(np.float16)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"df_{f}"] = np.stack([r[1] ** 2 / r[2] if r else nan for r in rows]).astype(np.float16)
        out[f"n_{f}"] = np.array([sum(r[3]) if r else 0 for r in rows])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    log(f"  statistics for the next super runs: {path} ({path.stat().st_size / 1e6:.0f} MB, {len(sp)} spans)")


def welch_p(M1, V1, D1, M0, V0, D0) -> np.ndarray:
    """Two-sided p of the difference of two weighted means (variances V, Satterthwaite denominators D)."""
    from scipy.stats import t as student

    v = V1 + V0
    tested = np.isfinite(M1) & np.isfinite(M0) & (v > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(tested, (M1 - M0) / np.sqrt(v), 0.0)
        df = np.where(tested, v * v / (D1 + D0), 1.0)
    return np.where(tested, 2 * student.sf(np.abs(t), df), np.nan)


def agree_code(A: np.ndarray, T: np.ndarray, t: float) -> np.ndarray:
    """G of an anomaly frame: the share of the members with the sign of A x SHARE, at least the gate's code exactly
    where the share passes it (and below it where not), 255 where A is missing."""
    share = np.where(A > 0, T, np.where(A < 0, 1 - T, 0.0))
    kc = round(t * SHARE)
    with np.errstate(invalid="ignore"):
        g = np.clip(np.rint(share * SHARE), 0, SHARE)
        g = np.where(share >= t - 1e-9, np.maximum(g, kc), np.minimum(g, kc - 1))
    return np.where(np.isfinite(A), g, 255).astype(np.uint8)


def run(a) -> int:
    init = pd.Timestamp(f"{a.cycle[:8]} {a.cycle[8:]}:00")
    days = E.full_days(init, E.LAST_STEP)
    runs = {m: load_run(a.stats_dir / m / f"{a.cycle}.npz", init) for m in ENSEMBLES}
    if missing := [m for m, r in runs.items() if r is None]:
        raise SystemExit(f"{a.cycle}: no run statistics of {', '.join(missing)}")
    if older := [m for m, r in runs.items() if not r["skeys"]]:  # saved before the super-ensemble's spans were kept
        log(
            f"::notice::{a.cycle}: the statistics of {', '.join(older)} hold no anomalies; no super-ensemble of this run"
        )
        return 0
    t_start = time.time()
    site = Path(a.site)
    outdir = site / E.DATA / SUPER.key / a.cycle
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True)
    log(f"super-ensemble {init:%Y-%m-%d %H}Z: {len(days)} days {days[0][0]:%Y-%m-%d}..{days[-1][0]:%Y-%m-%d}")
    highs = [r["high"] for r in runs.values() if r["high"] is not None]
    high = np.any(highs, axis=0) if len(highs) == len(runs) else None
    spread = {f: E.spread_on(f, E.G50) for f in ("z500", "msl")}
    Wr = E.Writer(outdir, a.threads)
    spans_meta: dict = {}
    anom_ok = dict.fromkeys(E.FIELDS, False)
    for name, k0, k1 in E.spans(len(days)):
        k = E.key(days, k0, k1)
        what, when = E.span_text(days, k0, k1)
        e = {"key": name, "first": f"{days[k0][0]:%Y-%m-%d}", "last": f"{days[k1][0]:%Y-%m-%d}",
             "label": f"{what[0].upper()}{what[1:]} · {when}", "n": {}, "k": {}, "members": {}}  # fmt: skip
        for f in E.FIELDS:
            c = combine(runs, f, k)
            if c is None:
                continue
            mean, _, _, ns = c
            an = anomaly(runs, f, k)
            hm = high if f == "msl" else None
            if an is None:
                img = E.frame(f, None, None, mean)
            else:
                A, T = an
                t = gate(ns)
                img = E.frame(f, A, agree_code(A, T, t), mean, hm)
                e["k"][f] = round(t * SHARE)
                anom_ok[f] = True
                if f in spread:
                    sd = spread[f].sd([days[j][0] + pd.Timedelta(hours=12) for j in range(k0, k1 + 1)], k1 - k0 + 1)
                    si = img.copy()
                    si[..., 0] = E.code_signed(A / sd, E.ENC[f]["sigma"])
                    if hm is not None:
                        si[..., 0][hm] = 0
                    Wr.put(f"{f}/s/{name}", si, False)
            Wr.put(f"{f}/a/{name}", img, False)
            e["n"][f] = sum(ns)
            e["members"][f] = dict(zip(ENSEMBLES, ns, strict=True))
        if e["n"]:
            spans_meta[name] = e

    # the change from each earlier super run whose statistics are there
    prev_at, changes = {}, 0
    for h, sub, meta, _ in CHANGES:
        pi = init - pd.Timedelta(hours=h)
        prev = load_super(a.prev_dir / f"{pi:%Y%m%d%H}.npz", pi) if a.prev_dir else None
        if prev is None:
            continue
        prev_at[h] = f"{pi:%Y-%m-%dT%H}:00Z"
        for name, k0, k1 in E.spans(len(days)):
            cut = E.prev_row(days, k0, k1, prev) if name in spans_meta else None
            if cut is None:
                continue
            c1, row = cut
            for f in E.FIELDS:
                c = combine(runs, f, E.key(days, k0, c1))
                if c is None or f not in prev or not prev[f]["n"][row]:
                    continue
                M1, V1, D1, ns = c
                P = prev[f]
                sig = E.significant(welch_p(M1, V1, D1, P["M"][row], P["V"][row], P["D"][row]))
                d = M1 - P["M"][row]
                img = np.dstack(
                    [E.code_signed(d, E.ENC[f]["change"]), sig.astype(np.uint8), np.zeros(d.shape, np.uint8)]
                )
                Wr.put(f"{f}/{sub}/{name}", img, False)
                cw, cwhen = E.span_text(days, k0, c1)
                spans_meta[name].setdefault(meta, {})[f] = {
                    "last": f"{days[c1][0]:%Y-%m-%d}",
                    "label": f"{cw[0].upper()}{cw[1:]} · {cwhen}",
                    "n": sum(ns),
                    "n0": int(P["n"][row]),
                    "sig": round(float(sig.mean()), 4),
                }
                changes += 1
    if a.save:
        save_super(a.save, init, days, runs)
    nbytes = Wr.close()
    log(
        f"super-ensemble {a.cycle}: {nbytes / 1e6:.0f} MB of frames ({changes} change maps), {time.time() - t_start:.0f} s"
    )

    field_of = {"grid": E.G50.desc(), "tiles": None, "change_grid": E.G50.desc(), "units_read": []}
    mans = {m: E.load_manifest(a.manifests / f"ensmaps_{m}.json", E.MODELS[m]) for m in ENSEMBLES}
    of = {m: next((c for c in mans[m].get("cycles", []) if c["cycle"] == a.cycle), {}) for m in ENSEMBLES}
    entry = dict(
        cycle=a.cycle,
        init=f"{init:%Y-%m-%dT%H}:00Z",
        last_step=E.LAST_STEP,
        days=[f"{d:%Y-%m-%d}" for d, _ in days],
        spans=[spans_meta[name] for name, _, _ in E.spans(len(days)) if name in spans_meta],
        fields={
            f: dict(field_of, anomaly=anom_ok[f], sigma=f in spread, unit=E.UNIT[f])
            for f in E.FIELDS
            if any(f in s["n"] for s in spans_meta.values())
        },
        enc=E.ENC,
        high_m=E.HIGH_M,
        fdr_q=E.FDR_Q,
        normals=of["ifs"].get("normals"),
        **{key: prev_at.get(h) for h, _, _, key in E.CHANGES},
        regimes=None,
        share=SHARE,
        weights=dict.fromkeys(ENSEMBLES, W),
        ensembles={m: {"made": of[m].get("made"), "stage": of[m].get("stage")} for m in ENSEMBLES},
        bytes=nbytes,
        wall_s=round(time.time() - t_start),
        made=dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%MZ"),
    )
    E.write_manifest(site, SUPER, entry, a.manifest_in, a.lists)
    return 0


def complete(c: dict) -> bool:
    """A run on an ensemble's page that the super-ensemble can take: whole (not checkpoints of one still landing), to
    its model's last step."""
    return not c.get("through") and c.get("last_step", 0) >= E.LAST_STEP


def cmd_plan(a) -> int:
    """Print the newest 00/12Z cycle whose four runs are on their pages, when the super-ensemble does not have it as
    made from those runs (nothing otherwise; with --cycle: that cycle once its four runs are there)."""
    mans = {m: E.load_manifest(a.manifests / f"ensmaps_{m}.json", E.MODELS[m]) for m in ENSEMBLES}
    sup = E.load_manifest(a.manifests / "ensmaps_super.json", SUPER)
    made = {c["cycle"]: c.get("ensembles", {}) for c in sup.get("cycles", [])}
    runs = {m: {c["cycle"]: c for c in mans[m].get("cycles", []) if complete(c)} for m in ENSEMBLES}
    both = set.intersection(*(set(r) for r in runs.values()))
    cands = sorted((c for c in both if c[8:] in HOURS and (not a.cycle or c == a.cycle)), reverse=True)
    if not cands:
        newest = ", ".join(f"{m} {max(r, default='-')}" for m, r in runs.items())
        log(f"  no {a.cycle or '00/12Z'} cycle has all four runs (the newest of each: {newest})")
        return 0
    c = cands[0]
    if made and max(made) > c and not a.force:
        log(f"  {c}: the page has the newer {max(made)}")
        return 0
    now = {m: {"made": runs[m][c].get("made"), "stage": runs[m][c].get("stage")} for m in ENSEMBLES}
    if made.get(c) == now and not a.force:
        log(f"  {c}: made from these four runs already")
        return 0
    print(c)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--manifests", type=Path, default=E.ROOT / "assets" / "models")
    p.add_argument("--cycle", default="")
    p.add_argument("--force", action="store_true")
    r = sub.add_parser("run")
    r.add_argument("--cycle", required=True)
    r.add_argument("--stats-dir", type=Path, required=True, help="the four runs' statistics, <model>/<cycle>.npz")
    r.add_argument("--prev-dir", type=Path, help="the earlier super runs' statistics, <cycle>.npz (save_super)")
    r.add_argument("--save", type=Path, help="write this super run's statistics here")
    r.add_argument("--site", type=Path, default=E.ROOT)
    r.add_argument("--manifests", type=Path, default=E.ROOT / "assets" / "models", help="the four ensembles' manifests")
    r.add_argument("--manifest-in", type=Path, help="the super-ensemble's manifest on main (default: the site's)")
    r.add_argument("--lists", type=Path, help="write publish.txt, prune.txt and floor.txt here")
    r.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    a = ap.parse_args()
    return cmd_plan(a) if a.cmd == "plan" else run(a)


if __name__ == "__main__":
    raise SystemExit(main())
