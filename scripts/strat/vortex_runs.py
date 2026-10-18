#!/usr/bin/env python3
"""How the northern polar-vortex forecast has moved from run to run.

nh_vortex.py hands every run's member series to record(): for each ensemble (AIFS-ENS, IFS-ENS, GEPS) and valid time,
the members' mean and 10th/90th percentiles of u(60N) at 10 hPa and of the 100 hPa 65-90N cap-height anomaly (against
the detrended MERRA-2 median the main figure subtracts), and the members' chance of a sudden warming: the share easterly
at 10 hPa on at least one day to day 15 (WMO). The last KEEP_DAYS days of runs are kept in RUNS. render() draws the last
SHOW runs of each ensemble over one another, the oldest faint and the newest bold with its 10-90 % band, the change of
the days 6-15 mean against the run a day earlier, and every kept run's sudden-warming chance against its start.

    python scripts/strat/vortex_runs.py render          # redraw from the record
    python scripts/strat/nh_vortex.py --cycle 2026101600 --models aifs,ifs --record-only   # add a past run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "assets" / "sst" / "data" / "vortex_runs.json"
OUT = REPO / "assets" / "sst" / "nh_vortex_runs.webp"
MODELS = {"aifs": ("#b4541f", "AIFS-ENS"), "ifs": ("#1f4b6e", "IFS-ENS"), "geps": ("#2f7d4f", "GEPS")}
KEYS = {  # panel title, axis label
    "u10": ("u 10 hPa", "u at 10 hPa, 60°N  (m s⁻¹)"),
    "zcap": ("cap height", "100 hPa 65–90°N height anomaly (m)"),
}
KEEP_DAYS = 30
SHOW = 8  # four days of 00/12Z runs
SSW_DAYS = 15


def summary(df: pd.DataFrame, dec: int) -> dict:
    """The members' mean and 10th/90th percentiles at each valid time; null where no member has a value."""
    v = df.to_numpy(dtype=float)
    have = np.isfinite(v).any(axis=1)

    def stat(fn) -> list:
        s = np.full(len(v), np.nan)
        s[have] = fn(v[have])
        return [round(float(x), dec) if np.isfinite(x) else None for x in s]

    return {
        "mean": stat(lambda a: np.nanmean(a, axis=1)),
        "p10": stat(lambda a: np.nanpercentile(a, 10, axis=1)),
        "p90": stat(lambda a: np.nanpercentile(a, 90, axis=1)),
    }


def ssw_chance(u10: pd.DataFrame, base: pd.Timestamp) -> float | None:
    """Share of members easterly at 10 hPa / 60N on at least one day to day SSW_DAYS, over the members with a value in
    that window (None when none has)."""
    lead = u10[u10.index <= base + pd.Timedelta(days=SSW_DAYS)].dropna(axis=1, how="all")
    return float((lead.to_numpy() < 0).any(axis=0).mean()) if lead.shape[1] else None


def record(base: pd.Timestamp, fc: dict, path: Path = RUNS) -> None:
    """Add (or replace) run `base`: fc = {model: {"u10": frame, "zcap": anomaly frame}}, frames (valid time, member).
    Members and valid times without any value are left out of the record."""
    doc = json.loads(path.read_text()) if path.exists() else {"runs": []}
    run = {"init": f"{base:%Y-%m-%dT%H}", "models": {}}
    for m, d in fc.items():
        u = d.get("u10")
        if u is not None:
            u = u.dropna(axis=1, how="all").dropna(axis=0, how="all")
        if m not in MODELS or u is None or u.shape[1] < 5:
            continue
        rec = {"members": int(u.shape[1]), "valid": [f"{t:%Y-%m-%dT%H}" for t in u.index], "ssw": ssw_chance(u, base)}
        rec["u10"] = summary(u, 1)
        z = d.get("zcap")
        if z is not None:
            z = z.dropna(axis=1, how="all")
        if z is not None and z.shape[1] >= 5:
            rec["zcap"] = summary(z.reindex(u.index), 0)
        run["models"][m] = rec
    if not run["models"]:
        return
    same = [r for r in doc["runs"] if r["init"] == run["init"]]
    if same:  # a run recorded again (a backfill of AIFS-ENS and IFS-ENS) keeps the models it does not bring
        run["models"] = {**same[0]["models"], **run["models"]}
    cut = base - pd.Timedelta(days=KEEP_DAYS)
    runs = [r for r in doc["runs"] if r["init"] != run["init"] and pd.Timestamp(r["init"]) >= cut] + [run]
    doc["runs"] = sorted(runs, key=lambda r: r["init"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, separators=(",", ":"), allow_nan=False))
    print(f"  run record: {len(doc['runs'])} runs, {', '.join(MODELS[m][1] for m in run['models'])}", flush=True)


def _series(rec: dict, key: str, stat: str) -> pd.Series:
    return pd.Series(
        [np.nan if v is None else v for v in rec[key][stat]], index=pd.DatetimeIndex(rec["valid"]), dtype=float
    )


def day_change(runs: list[dict], m: str, key: str) -> str | None:
    """The newest run's days 6-15 mean against the same valid times in the run a day earlier."""
    new = runs[-1]
    t0 = pd.Timestamp(new["init"])
    old = [r for r in runs if pd.Timestamp(r["init"]) == t0 - pd.Timedelta(days=1) and key in r["models"].get(m, {})]
    if not old or key not in new["models"][m]:
        return None
    a, b = _series(new["models"][m], key, "mean"), _series(old[0]["models"][m], key, "mean")
    win = a.index[(a.index >= t0 + pd.Timedelta(days=6)) & (a.index <= t0 + pd.Timedelta(days=15))].intersection(
        b.index
    )
    if len(win) < 3:
        return None
    d = float((a[win] - b[win]).mean())
    unit = "m s⁻¹" if key == "u10" else "m"
    return f"24 h change, days 6–15: {d:+.1f} {unit}"


def render(path: Path = RUNS, out: Path = OUT) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    doc = json.loads(path.read_text())
    runs = doc["runs"]
    models = [m for m in MODELS if any(m in r["models"] for r in runs)]
    fig = plt.figure(figsize=(13.6, 10.4))
    gs = fig.add_gridspec(3, len(models), height_ratios=(1, 1, 0.8), hspace=0.42, wspace=0.16)
    for j, m in enumerate(models):
        col, lab = MODELS[m]
        mine = [r for r in runs if m in r["models"]][-SHOW:]
        for i, key in enumerate(KEYS):
            ax = fig.add_subplot(gs[i, j])
            shown = [r for r in mine if key in r["models"][m]]
            for k, r in enumerate(shown):
                rec, new = r["models"][m], k == len(shown) - 1
                mean = _series(rec, key, "mean")
                if new:
                    ax.fill_between(
                        mean.index, _series(rec, key, "p10"), _series(rec, key, "p90"), color=col, alpha=0.16, lw=0
                    )
                age = (k + 1) / len(shown)
                ax.plot(
                    mean.index,
                    mean.values,
                    color=col if new else "0.35",
                    alpha=1.0 if new else 0.15 + 0.6 * age,
                    lw=2.4 if new else 0.9 + 0.6 * age,
                    zorder=5 if new else 3,
                )
            ax.axhline(0, color="#a33" if key == "u10" else "#8a8680", lw=1.2 if key == "u10" else 0.9, zorder=4)
            ax.set_title(f"{lab} · {KEYS[key][0]}", fontsize=10, fontweight="bold", loc="left")
            ax.set_title(day_change(mine, m, key) or "", fontsize=8, color="0.25", loc="right")
            if j == 0:
                ax.set_ylabel(KEYS[key][1], fontsize=8.6)
            ax.grid(alpha=0.22)
            ax.tick_params(labelsize=8)
            ax.xaxis.set_major_locator(matplotlib.dates.AutoDateLocator(minticks=3, maxticks=6))
            ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%-d %b"))
    ax = fig.add_subplot(gs[2, :])
    for m in models:
        col, lab = MODELS[m]
        mine = [r for r in runs if m in r["models"]]
        t = pd.DatetimeIndex([r["init"] for r in mine])
        pct = [np.nan if r["models"][m]["ssw"] is None else 100 * r["models"][m]["ssw"] for r in mine]
        ax.plot(t, pct, color=col, lw=1.8, marker="o", ms=3.5, label=lab)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("% of members", fontsize=8.6)
    ax.set_title(
        f"Chance of a sudden warming by day {SSW_DAYS}, run by run: members easterly at 10 hPa / 60°N on any day",
        fontsize=10,
        fontweight="bold",
        loc="left",
    )
    ax.grid(alpha=0.22)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, loc="upper left", ncol=3, framealpha=0.9)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%-d %b %HZ"))
    newest = pd.Timestamp(runs[-1]["init"])
    fig.suptitle(
        f"Northern polar vortex: how the forecast has moved, run by run (newest {newest:%Y-%m-%d %HZ})",
        fontsize=13.5,
        fontweight="bold",
        x=0.06,
        ha="left",
        y=0.99,
    )
    fig.text(
        0.06,
        0.952,
        f"the last {SHOW} runs of each ensemble: older runs grey and fainter, the newest bold in the model's colour with "
        "its members' 10–90 % band",
        fontsize=9,
        color="0.25",
    )
    fig.text(
        0.06,
        0.012,
        "Each line is one run's ensemble mean. The 24 h change is the newest run's mean minus the run 24 h before it, "
        "over the newest run's days 6–15.\nHeight anomaly against the MERRA-2 1980– median detrended to the current year, as on "
        "the main vortex figure. AIFS-ENS 25 members + control, IFS-ENS 50 perturbed members (ECMWF open data,\n"
        "Google Cloud mirror), GEPS 20 members + control (ECCC). GEPS has no archive, so its record starts with the "
        "first run kept.",
        fontsize=8,
        color="#6f6b64",
        linespacing=1.45,
    )
    fig.subplots_adjust(left=0.06, right=0.985, top=0.915, bottom=0.1)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, facecolor="white", pil_kwargs={"quality": 88, "method": 6})
    plt.close(fig)
    print(f"wrote {out}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("render",))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    render(out=Path(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
