#!/usr/bin/env python3
"""The ensembles in the verification database: ECMWF's IFS and AIFS ensembles, NCEP's GEFS and ECCC's GEPS, every
member, against the same observations as the deterministic models (scripts/verify/mverify_scores.py), to day 15.

Downloads only the fields that are scored, for every member (byte ranges from each file's index; a GEPS file holds one
field's members), at 00Z of each day of the forecast:
  * surface: 2 m temperature and dew point (GEFS and GEPS publish 2 m relative humidity, from which mverify derives
    it), mean sea-level pressure and 10 m wind, against every GHCNh station; precipitation every 12 h (GEFS: its 6 h
    buckets) for 24 h totals at the gauges and against Stage IV on the common grid (mverify_scores.rain_pairs)
  * upper air: 500 hPa height, 850 hPa temperature and the wind at 850 and 250 hPa, against radiosondes: the headline
    fields of the WMO lead centre for ensemble verification
with the model terrain and land-sea mask at step 0 (IFS-ENS: from the IFS deterministic run on the same grid; GEFS and
GEPS publish no land-sea mask). On the open-data mirror IFS-ENS is its 50 perturbed members (no control), AIFS-ENS 50
and the control, GEFS 30 and the control, GEPS 20 and the control. ECMWF comes from the Google Cloud mirror only.
Every model is averaged onto the deterministic scores' 0.5 deg grid first (mverify_scores.opened).

Pairs are matched as mverify_scores.matched does (all of them, ``kept`` marking mverify's default cuts), and the
members of each case kept as one row (mverify.ensemble.summarise): ens-sfc-<m>_<init>.parquet (stations and gauges),
ens-raob-<m>_ (radiosondes) and ens-rain-<m>_ (Stage IV cells), m = ifs, aifs, gefs, geps, written to --archive-out for
the monthly releases verify-db-YYYY-MM (scripts/verify/verifydb.sh). One run of one model at a time, so a runner's disk
holds any backfill. ``stats`` then sums each run's cases, on the sample every ensemble of the run has, into
ens-stats_<init>.parquet (``run_stats``), once every ensemble that can still be fetched is in: GEPS only while HPFX
keeps it (about 50 days).

    python scripts/verify/ens_verify.py match --db obs.duckdb --model GEPS --inits 2026091600 2026092000 --work /tmp/e \\
        --have have.txt --archive-out db-new
    python scripts/verify/ens_verify.py stats --derived db --have have.txt --archive-out db-new
"""

from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mverify_scores as D
import numpy as np
import pandas as pd

ENSEMBLES = {"IFS-ENS": "ifs", "AIFS-ENS": "aifs", "GEFS": "gefs", "GEPS": "geps"}
LEADS = range(24, 361, 24)  # 00Z of every day to day 15
RAIN_STEP = {"IFS-ENS": 12, "AIFS-ENS": 12, "GEFS": 6, "GEPS": 12}  # hours between the precipitation fields
RAIN_ENDS = range(24, 361, 12)  # ends of the 24 h totals: the gauges' 00Z and 12Z days and Stage IV's 12Z days
HPFX_DAYS = 48  # GEPS runs older than this are gone from HPFX (it keeps ~50 days)
RANK_BINS = 10  # rank histograms in tenths of the members' range (ensembles of different sizes compare)
STAT_SUMS = ["n", "crps", "crps_oe", "se", "sse", "svar", "sov"]
WINDOWS = (30, 90, 365, None)  # runs of a scorecard window (None: all), as the deterministic ones
LEAD_GROUPS = (("days 1-3", 24, 72), ("days 4-7", 96, 168), ("days 8-15", 192, 360))  # rank and reliability

ECMWF = {
    "sfc": [("2t", "sfc", None), ("2d", "sfc", None), ("msl", "sfc", None), ("10u", "sfc", None), ("10v", "sfc", None)],
    "upper": [
        ("{z}", "pl", "500"),
        ("t", "pl", "850"),
        ("u", "pl", "850"),
        ("v", "pl", "850"),
        ("u", "pl", "250"),
        ("v", "pl", "250"),
    ],
    "tp": [("tp", "sfc", None)],
    "static": [("z", "sfc", None), ("lsm", "sfc", None)],
}
GEFS = {
    "sfc": [
        "TMP:2 m above ground",
        "RH:2 m above ground",
        "PRMSL:mean sea level",
        "UGRD:10 m above ground",
        "VGRD:10 m above ground",
    ],
    "upper": ["HGT:500 mb", "TMP:850 mb", "UGRD:850 mb", "VGRD:850 mb", "UGRD:250 mb", "VGRD:250 mb"],
    "tp": ["APCP:surface"],
    "static": ["HGT:surface"],
}
GEFS_MEMBERS = ["gec00"] + [f"gep{i:02d}" for i in range(1, 31)]
GEFS_URL = "https://noaa-gefs-pds.s3.amazonaws.com/gefs.{d}/{h}/atmos/pgrb2ap5/{m}.t{h}z.pgrb2a.0p50.f{L:03d}"
GEPS = {
    "sfc": ["TMP_TGL_2m", "RH_TGL_2m", "PRMSL_MSL_0", "UGRD_TGL_10m", "VGRD_TGL_10m"],
    "upper": ["HGT_ISBL_0500", "TMP_ISBL_0850", "UGRD_ISBL_0850", "VGRD_ISBL_0850", "UGRD_ISBL_0250", "VGRD_ISBL_0250"],
    "tp": ["APCP_SFC_0"],
    "static": ["HGT_SFC_0"],
}
GEPS_URL = (
    D.DATAMART
    + "/{d}/WXO-DD/ensemble/geps/grib2/raw/{h}/{L:03d}/CMC_geps-raw_{v}_latlon0p5x0p5_{d}{h}_P{L:03d}_allmbrs.grib2"
)


def ecmwf_bytes(url: str, want: set[tuple], types: tuple[str, ...]) -> bytes:
    """The messages of ``want`` (param, levtype, levelist) of the given types (pf, cf, fc) in one open-data file."""
    spans, found = [], set()
    for line in D.get(url + ".index").decode().splitlines():
        d = json.loads(line)
        if (key := (d["param"], d["levtype"], d.get("levelist"))) in want and d["type"] in types:
            spans.append((d["_offset"], d["_offset"] + d["_length"] - 1))
            found.add(key)
    if found != want:
        raise RuntimeError(f"{url}: no {sorted(want - found)} in the index")
    return b"".join(D.get(url + ".grib2", f"{a}-{b}") for a, b in D.merged(spans))


def gefs_member(url: str, want: list[str]) -> bytes:
    """One member's messages of ``want`` (variable:level) from its file, by its .idx."""
    lines = D.get(url + ".idx").decode().splitlines()
    offsets = [int(line.split(":")[1]) for line in lines]
    spans = [
        (offsets[i], offsets[i + 1] - 1) for i, line in enumerate(lines[:-1]) if any(f":{w}:" in line for w in want)
    ]
    if len(spans) < len(want):
        raise RuntimeError(f"{url}: not all of {want} in the .idx")
    return b"".join(D.get(url, f"{a}-{b}") for a, b in D.merged(spans))


def fetch(model: str, init: str, lead: int, kind: str, dest: Path) -> Path:
    """One step's fields of ``kind`` (sfc, upper, tp, static) of every member, into ``dest``."""
    if dest.exists():
        return dest
    day, hh = init[:8], init[8:]
    dest.parent.mkdir(parents=True, exist_ok=True)
    if model in ("IFS-ENS", "AIFS-ENS"):
        z = "gh" if model == "IFS-ENS" else "z"
        want = {(p.format(z=z), lt, lv) for p, lt, lv in ECMWF[kind]}
        base = f"{D.GOOGLE}/{day}/{hh}z/{'ifs' if model == 'IFS-ENS' else 'aifs-ens'}/0p25"
        if kind == "static" and model == "IFS-ENS":  # the deterministic run's terrain and mask, on the same grid
            blob = ecmwf_bytes(f"{base}/oper/{day}{hh}0000-0h-oper-fc", want, ("fc",))
        elif model == "IFS-ENS":
            blob = ecmwf_bytes(f"{base}/enfo/{day}{hh}0000-{lead}h-enfo-ef", want, ("pf",))
        elif kind == "static":
            blob = ecmwf_bytes(f"{base}/enfo/{day}{hh}0000-0h-enfo-cf", want, ("cf",))
        else:
            stem = f"{base}/enfo/{day}{hh}0000-{lead}h-enfo"
            blob = ecmwf_bytes(f"{stem}-cf", want, ("cf",)) + ecmwf_bytes(f"{stem}-pf", want, ("pf",))
    elif model == "GEFS":
        members = GEFS_MEMBERS[:1] if kind == "static" else GEFS_MEMBERS
        urls = [GEFS_URL.format(d=day, h=hh, m=m, L=lead) for m in members]
        with ThreadPoolExecutor(16) as ex:
            blob = b"".join(ex.map(lambda u: gefs_member(u, GEFS[kind]), urls))
    else:  # GEPS: one file per field, all members; HPFX is a shared public service, so a few at a time
        with ThreadPoolExecutor(4) as ex:
            blob = b"".join(ex.map(D.get, [GEPS_URL.format(d=day, h=hh, v=v, L=lead) for v in GEPS[kind]]))
    part = dest.with_suffix(".part")
    part.write_bytes(blob)
    part.rename(dest)
    return dest


def rain_leads(model: str) -> list[int]:
    """The steps whose precipitation makes the 24 h totals (a step-0 accumulation is empty, and absent from some)."""
    return list(range(RAIN_STEP[model], max(LEADS) + 1, RAIN_STEP[model]))


def run_cases(model: str, init: str, work: Path, store) -> dict[str, pd.DataFrame]:
    """One run's cases (mverify.ensemble.summarise): sfc (stations and gauges), raob and rain (Stage IV cells)."""
    from mverify.ensemble import CELL, summarise

    root = work / ENSEMBLES[model] / init
    static = fetch(model, init, 0, "static", root / "static.grib2")
    out: dict[str, list[pd.DataFrame]] = {"sfc": [], "raob": [], "rain": []}
    for lead in LEADS:
        for kind, network in (("sfc", "GHCNH"), ("upper", "RAOB")):
            f = fetch(model, init, lead, kind, root / f"{kind}_{lead:03d}.grib2")
            out["sfc" if kind == "sfc" else "raob"].append(
                summarise(D.matched(D.opened([static, f], model), store, [network]))
            )
            f.unlink()
        print(f"{model} {init} +{lead} h matched", flush=True)
    tp = {lead: fetch(model, init, lead, "tp", root / f"tp_{lead:03d}.grib2") for lead in rain_leads(model)}
    # One 24 h window at a time, from its fields alone. A total from step 0 also gives the first day's window again
    # (0-24 h from the 24 h field), so only the window ending at `end` is kept.
    for end in RAIN_ENDS:
        fc = D.opened([static, *(f for lead, f in tp.items() if end - 24 <= lead <= end)], model)
        gauges = D.matched(fc, store, ["GHCNH"])  # the gauges' 24 h totals (pr24)
        out["sfc"].append(summarise(gauges[gauges["lead_h"] == end]))
        if not (cells := D.rain_pairs(fc, store)).empty:
            cells = cells[cells["lead_h"] == end].assign(variable="tp", sigma_o=np.nan)
            out["rain"].append(summarise(cells, keys=CELL))
    shutil.rmtree(root)
    return {k: D._cat(v) for k, v in out.items()}


def file_of(kind: str, model: str, init: str) -> str:
    return f"ens-{kind}-{ENSEMBLES[model]}_{init}.parquet"


def held(model: str, init: str, have: set[str]) -> bool:
    """Whether the database has the run of the ensemble: its station and radiosonde cases (rain only over Stage IV)."""
    return {file_of(k, model, init) for k in ("sfc", "raob")} <= have


def match(a) -> None:
    from mverify.db.store import ObsStore

    have = set(a.have.read_text().split()) if a.have and a.have.exists() else set()
    a.archive_out.mkdir(parents=True, exist_ok=True)
    with ObsStore.open(a.db, read_only=True) as store:
        for init in D.runs(*a.inits):
            if held(a.model, init, have):
                continue
            for kind, frame in run_cases(a.model, init, a.work, store).items():
                if not frame.empty:
                    frame.to_parquet(a.archive_out / file_of(kind, a.model, init), compression="zstd")
            print(f"{a.model} {init}: archived", flush=True)


# --------------------------------------------------------------------------- per-run sums
def _sums(c: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Sums over cases that add over runs: the count, CRPS (fair, and with observation error), the error of the mean,
    its square, the members' variance and the observation-error variance, and the rank histogram (rank_0 ...), each
    observation's rank among the members as a randomised PIT in tenths, ties split at random."""
    err = c["mean"] - c["obs"]
    pit = (c["below"] + rng.random(len(c)) * (c["ties"] + 1)) / (c["n_members"] + 1)
    d = c.assign(
        n=1,
        se=err,
        sse=err * err,
        svar=c["spread"] ** 2,
        sov=c["sigma_o"] ** 2,
        rank=np.minimum((pit * RANK_BINS).astype(int), RANK_BINS - 1),
    )
    t = d.groupby(D.STAT_BY, sort=False, observed=True)[STAT_SUMS].sum(min_count=1)
    h = d.groupby([*D.STAT_BY, "rank"], sort=False, observed=True).size().unstack("rank", fill_value=0)
    return t.join(h.rename(columns=lambda k: f"rank_{k}")).reset_index()


def _reliability(c: pd.DataFrame) -> pd.DataFrame:
    """Per event threshold (p_ge_<t> columns) and forecast probability in tenths: the cases, their summed probability,
    the events observed and the summed Brier score, for Brier scores (exact) and reliability diagrams."""
    out = []
    for col in [k for k in c.columns if k.startswith("p_ge_")]:
        e = c.dropna(subset=[col])
        if e.empty:
            continue
        t = float(col.removeprefix("p_ge_"))
        o = (e["obs"] >= t).astype(int)
        e = e.assign(threshold=t, bin=(e[col] * 10).round().astype(int), p=e[col], o=o, bs=(e[col] - o) ** 2, n=1)
        g = e.groupby([*D.STAT_BY, "threshold", "bin"], sort=False, observed=True)
        out.append(g[["n", "p", "o", "bs"]].sum().reset_index())
    return D._cat(out)


def run_stats(paths: dict[str, list[Path]], init: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One run's sums (``_sums``, ``_reliability``) per model, network, variable, region and lead, on the kept cases
    that every ensemble of the run has (mverify.verify.common_sample; Stage IV cells by position)."""
    from mverify import scores
    from mverify.ensemble import CELL
    from mverify.verify import common_sample

    rng = np.random.default_rng(int(init))
    stats, rel = [], []
    for kind, files in paths.items():
        c = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        if kind == "rain":
            n = c.groupby([k for k in CELL if k != "model"])["model"].transform("nunique")
            c = c[n == c["model"].nunique()].assign(network="STAGEIV", kept=True)
        else:
            c = common_sample(c[c["kept"]])
        for region in scores.REGIONS:
            if not (rc := scores.assign_regions(c, [region])).empty:
                stats.append(_sums(rc, rng))
                rel.append(_reliability(rc))
    when = pd.to_datetime(init, format="%Y%m%d%H")
    return D._cat(stats).assign(init=when), D._cat(rel).assign(init=when)


def expected(init: str, today: pd.Timestamp) -> list[str]:
    """The ensembles a run's sums wait for: all four while HPFX still has the run's GEPS, else the three others."""
    gone = pd.to_datetime(init, format="%Y%m%d%H") < today - pd.Timedelta(days=HPFX_DAYS)
    return [m for m in ENSEMBLES if not (m == "GEPS" and gone)]


def ready(have: set[str], today: pd.Timestamp) -> dict[str, list[str]]:
    """The runs whose sums are due, with their ensembles: every ensemble that can still come is in, no sums yet."""
    out = {}
    for init in sorted({n.split("_")[1][:10] for n in have if n.startswith("ens-sfc-")}):
        models = [m for m in ENSEMBLES if held(m, init, have)]
        if f"ens-stats_{init}.parquet" not in have and set(expected(init, today)) <= set(models):
            out[init] = models
    return out


def stats(a) -> None:
    have = set(a.have.read_text().split()) if a.have and a.have.exists() else set()
    due = ready(have, pd.Timestamp.now(tz="UTC").tz_localize(None).normalize())
    if a.list:  # the files each due run's sums are made from, for the workflow to download
        for init, models in due.items():
            print(init, *(n for k in ("sfc", "raob", "rain") for m in models if (n := file_of(k, m, init)) in have))
        return
    a.archive_out.mkdir(parents=True, exist_ok=True)
    for init, models in due.items():
        if a.init and init != a.init:
            continue
        paths = {
            k: [a.derived / file_of(k, m, init) for m in models if (a.derived / file_of(k, m, init)).exists()]
            for k in ("sfc", "raob", "rain")
        }
        s, r = run_stats({k: v for k, v in paths.items() if v}, init)
        cats = dict.fromkeys(D.STAT_BY[:4], "category")
        s.astype(cats).to_parquet(a.archive_out / f"ens-stats_{init}.parquet", compression="zstd")
        r.astype(cats).to_parquet(a.archive_out / f"ens-rel_{init}.parquet", compression="zstd")
        print(f"{init}: sums of {', '.join(models)}", flush=True)


# --------------------------------------------------------------------------- windows
def with_wind(s: pd.DataFrame) -> pd.DataFrame:
    """``s`` (sums per run) with the wind at each level from its components: the mean CRPS of u and v, the error and
    spread of the vector (sse and svar added; n the cases of one component)."""
    rows = []
    for var_u in sorted(v for v in s["variable"].astype(str).unique() if v.startswith("u")):
        key = ["model", "network", "region", "lead_h", "init"]
        u = s[s["variable"] == var_u].set_index(key)
        v = s[s["variable"] == "v" + var_u[1:]].set_index(key)
        j = u.join(v, how="inner", lsuffix="_u", rsuffix="_v")
        if j.empty:
            continue
        w = pd.DataFrame({"n": j["n_u"], "variable": "wind" + var_u[1:]}, index=j.index)
        for c in ("crps", "crps_oe"):
            w[c] = (j[f"{c}_u"] + j[f"{c}_v"]) / 2
        for c in ("sse", "svar", "sov"):
            w[c] = j[f"{c}_u"] + j[f"{c}_v"]
        rows.append(w.reset_index())
    return pd.concat([s.astype(dict.fromkeys(D.STAT_BY[:4], str)), *rows], ignore_index=True)


def ens_table(t: pd.DataFrame) -> pd.DataFrame:
    """Scores from summed sums: CRPS (and with observation error), the RMSE of the ensemble mean, the spread (and with
    observation error) and the spread/skill ratios (mverify.scores.ensemble's)."""
    n = t["n"]
    with np.errstate(invalid="ignore", divide="ignore"):
        rmse, spread = np.sqrt(t["sse"] / n), np.sqrt(t["svar"] / n)
        spread_oe = np.sqrt((t["svar"] + t["sov"]) / n)
        return pd.DataFrame(
            {
                "n": n,
                "crps": t["crps"] / n,
                "crps_oe": t["crps_oe"] / n,
                "rmse": rmse,
                "spread": spread,
                "spread_oe": spread_oe,
                "ssr": spread / rmse,
                "ssr_oe": spread_oe / rmse,
            }
        ).reset_index()


def _groups(lead) -> pd.Series:
    return pd.Series(
        np.select([lead.between(a, b) for _, a, b in LEAD_GROUPS], [g for g, _, _ in LEAD_GROUPS], ""), index=lead.index
    )


def ens_card(stats: pd.DataFrame, rel: pd.DataFrame) -> dict | None:
    """The scorecard of the runs in ``stats`` (``run_stats`` rows; mverify.report.scorecard, CRPS tested pair by pair
    with mverify.significance's paired test on the per-run mean CRPS), with each row's rank histograms and, for the
    event thresholds, reliability and Brier scores, per lead group."""
    from mverify import report
    from mverify.report import TRUTH
    from mverify.significance import BY, effective_runs, paired_test

    models, used, left_out = D.window_runs(stats)
    if not used:
        return None
    s = with_wind(stats[stats["model"].isin(models) & stats["init"].isin(used)])
    table = ens_table(s.groupby(D.STAT_BY)[STAT_SUMS].sum(min_count=1))
    runs = s.assign(c=s["crps"] / s["n"]).set_index([*BY, "init", "model"])["c"].unstack("model").sort_index()
    rows = []
    for i, a in enumerate(models):
        for b in models[i + 1 :]:
            for key, g in (runs[a] - runs[b]).dropna().groupby(level=BY):
                d = g.to_numpy()
                found = (len(d), effective_runs(d), paired_test(d))
                rows += [(*key, a, b, *found), (*key, b, a, *found)]
    tests = pd.DataFrame(rows, columns=[*BY, "model", "reference", "runs", "runs_eff", "p"])
    inits = {m: [pd.Timestamp(i).strftime("%Y-%m-%dT%H") for i in used] for m in models}
    card = report.scorecard(table, inits, tests)

    def key(net, var, region):
        return f"{TRUTH.get(net, (net.lower(),))[0]}:{var}:{region}"

    ranks = [c for c in s.columns if c.startswith("rank_")]
    h = s.dropna(subset=ranks, how="all").assign(group=lambda x: _groups(x["lead_h"]))
    h = h[h["group"] != ""].groupby(["network", "variable", "region", "group", "model"])[ranks].sum()
    rank: dict = {}
    for (net, var, region, group, model), r in h.iterrows():
        f = r.to_numpy(float)
        rank.setdefault(key(net, var, region), {}).setdefault(model, {})[group] = list(np.round(f / f.sum(), 4))
    r = rel[rel["model"].isin(models) & rel["init"].isin(used)].astype(dict.fromkeys(D.STAT_BY[:4], str))
    r = r.assign(group=_groups(r["lead_h"]))
    r = (
        r[r["group"] != ""]
        .groupby(["network", "variable", "region", "threshold", "group", "model", "bin"])[["n", "p", "o", "bs"]]
        .sum()
    )
    reliability: dict = {}
    for (net, var, region, thr, group, model), cell in r.groupby(level=[0, 1, 2, 3, 4, 5]):
        g = cell.droplevel([0, 1, 2, 3, 4, 5])
        n, o = g["n"].sum(), g["o"].sum()
        clim = o / n * (1 - o / n)  # the Brier score of the sample's base rate
        reliability.setdefault(key(net, var, region), {}).setdefault(f"{thr:g}", {}).setdefault(model, {})[group] = {
            "bins": [int(b) for b in g.index],
            "n": [int(x) for x in g["n"]],
            "p": list(np.round(g["p"] / g["n"], 4)),
            "o": list(np.round(g["o"] / g["n"], 4)),
            "brier": round(float(g["bs"].sum() / n), 5),
            "bss": round(float(1 - g["bs"].sum() / n / clim), 4) if clim else None,
            "base_rate": round(float(o / n), 4),
        }
    card["rank"], card["reliability"] = rank, reliability
    card["lead_groups"] = [g for g, _, _ in LEAD_GROUPS]
    first, last = (pd.Timestamp(i).strftime("%Y%m%d%H") for i in (used[0], used[-1]))
    card["window"] = {"runs": len(used), "first": first, "last": last, "left_out": left_out}
    return card


def windows(a) -> None:
    """The ensembles' scorecards of the newest 30, 90 and 365 runs and of the whole database once it is longer
    (ens_scores_<runs>.json, ens_scores_all.json), each once the database holds that many runs' sums."""
    from mverify import report

    inits = sorted(f.stem.split("_")[1] for f in a.derived.glob("ens-stats_*.parquet"))
    for runs in WINDOWS:
        if len(inits) < (runs or 366):
            continue
        use = inits[-runs:] if runs else inits
        stats = pd.read_parquet([a.derived / f"ens-stats_{i}.parquet" for i in use])
        rel = pd.read_parquet([a.derived / f"ens-rel_{i}.parquet" for i in use])
        if (card := ens_card(stats, rel)) is None:
            continue
        report.write_json(card, D.OUT_DIR / f"ens_scores_{runs or 'all'}.json")
        w = card["window"]
        print(f"ensembles {runs or 'all'}: {w['runs']} runs {w['first']} to {w['last']}, left out {w['left_out']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("match", help="match runs of one ensemble and summarise their cases")
    m.add_argument("--db", required=True, help="mverify observation database")
    m.add_argument("--model", required=True, choices=list(ENSEMBLES))
    m.add_argument("--inits", nargs=2, required=True, metavar=("FIRST", "LAST"), help="00Z runs, YYYYMMDDHH")
    m.add_argument("--work", type=Path, required=True, help="download directory")
    s = sub.add_parser("stats", help="each run's sums once its ensembles are in")
    s.add_argument("--derived", type=Path, default=Path("."), help="the due runs' ens-sfc/raob/rain files, downloaded")
    s.add_argument("--init", help="only this run (YYYYMMDDHH)")
    s.add_argument("--list", action="store_true", help="print each due run and the files its sums are made from")
    w = sub.add_parser("windows", help="the scorecards of windows of runs, from the runs' sums")
    w.add_argument("--derived", type=Path, required=True, help="the database's ens-stats and ens-rel files, downloaded")
    for p in (m, s):
        p.add_argument("--have", type=Path, help="names of the files the database holds, one per line")
        p.add_argument("--archive-out", type=Path, required=True, help="where the files to add are written")
    a = ap.parse_args()
    {"match": match, "stats": stats, "windows": windows}[a.cmd](a)


if __name__ == "__main__":
    main()
