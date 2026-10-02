#!/usr/bin/env python3
"""Recurvature watch for live storms, stated against the 1991-2020 record (user 2026-10-02, item 3 of the TC follow-ups).

Rule (deliberately simple, because the record does not support more):
  a Northern-Hemisphere storm is on WATCH when at least WATCH_P of the members of EVERY ensemble that tracks it
  (AIFS-ENS and IFS-ENS, from data/tcens.json) recurve, with the median recurvature by day WATCH_DAY.
The outflow-strength index is shown but is NOT part of the rule: in ERA5 1991-2020 the strong-minus-weak interaction
difference (200 hPa proxy, rank correlation 0.79 with the live index on recent AIFS-ENS forecasts) was not significant
anywhere at a 10 % false-discovery rate (build_tc_downstream.py), so a threshold on it would claim what the record does
not show. Each watch line quotes what the record DOES show for that basin and season: the share of the storm-relative
500 hPa map that is significant at days 0/2/4 and any area-mean temperature days that pass the test.

    python src/tc_alert.py --out-dir ../../assets/sst
Writes data/tc_alert.json, adds "alert" to data/tcjet.json and data/tcens.json, and stamps a banner on top of
tcjet.webp and tcens_tracks.webp (once per cycle: the JSON records the stamped init).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REF = Path(__file__).resolve().parents[1] / "data" / "reference"
WATCH_P = 0.5
WATCH_DAY = 10.0
BASIN_OF = {"W": "WP", "E": "EP", "C": "EP", "L": "NA"}
BASIN_LAB = {"WP": "western North Pacific", "EP": "eastern North Pacific", "NA": "North Atlantic"}
SEASON_LAB = {"aug_sep": "Aug–Sep", "oct_nov": "Oct–Nov", "jun_nov": "Jun–Nov"}
REGIONS_OF = {"WP": ("wcan_pnw", "west_us", "central_us", "east_us"), "EP": ("wcan_pnw", "west_us", "central_us", "east_us"),
              "NA": ("east_us", "w_europe", "n_europe")}
MLAB = {"aifs": "AIFS-ENS", "ifs": "IFS-ENS", "aifs-ens": "AIFS-ENS", "ifs-ens": "IFS-ENS"}


def season_of(month: int) -> str:
    return "aug_sep" if month in (8, 9) else "oct_nov" if month in (10, 11) else "jun_nov"


def history_line(ref: dict, basin: str, sk: str) -> tuple[str, dict]:
    info = ref["sets"].get(f"{basin}|{sk}")
    if not info:
        return "no historical composite for this basin and season", {}
    sf = info.get("sig_frac", {})
    rel = sf.get("z500rel|all") or []
    days = {d: round(100 * rel[d], 1) for d in (0, 2, 4, 6) if d < len(rel)}
    parts = []
    if any(v > 0 for v in days.values()):
        last = max(d for d, v in days.items() if v > 0)
        parts.append(f"a significant ridge–trough pattern downstream of the recurvature point through about day {last} "
                     f"({', '.join(f'day {d}: {v}%' for d, v in days.items())} of the storm-relative map)")
    else:
        parts.append("no significant storm-relative 500 hPa pattern")
    temp = []
    for k, cv in info.get("curves", {}).items():
        var, rk, gk = k.split("|")
        if var != "t2m" or gk != "all" or not any(cv["sig"]) or rk not in REGIONS_OF[basin]:
            continue
        lags = [i for i, s in enumerate(cv["sig"]) if s]
        m = np.mean([cv["mean"][i] for i in lags])
        span = f"day {lags[0]}" if len(lags) == 1 else f"days {lags[0]}–{lags[-1]}" if lags[-1] - lags[0] == len(lags) - 1 else \
            "days " + ", ".join(map(str, lags))
        temp.append(f"{ref['regions'][rk]} {m:+.1f} °C on {span}")
    parts.append(("temperature: " + "; ".join(temp)) if temp else "no significant regional temperature signal")
    diff = sf.get("z500rel|diff") or []
    if diff and not any(v > 0 for v in diff):
        parts.append("outflow strength made no significant difference")
    n = info["n_storms"]["all"]
    return f"{BASIN_LAB[basin]} {SEASON_LAB[sk]}, {n} storms 1991–2020: " + "; ".join(parts), \
        dict(n_storms=n, rel_sig_pct=days, temp=temp)


def short_line(basin: str, sk: str, hj: dict) -> str:
    """One banner line: what 1991-2020 shows after storms recurving in this basin and season."""
    if not hj:
        return "no historical composite for this basin and season"
    rel = hj.get("rel_sig_pct", {})
    sigd = [d for d, v in rel.items() if v > 0]
    a = (f"in {hj['n_storms']} {SEASON_LAB[sk]} {BASIN_LAB[basin]} recurvatures (1991–2020) a significant ridge then trough "
         f"ran downstream for ~{max(sigd)} days" if sigd else f"{hj['n_storms']} {SEASON_LAB[sk]} recurvatures (1991–2020) show no "
         "significant downstream 500 hPa pattern")
    t = "; ".join(hj.get("temp", [])) or "no significant regional temperature signal"
    return f"{a}; {t}"


def banner(png: Path, lines: list[str], level: str):
    """Prepend a banner strip to an existing figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    import textwrap
    im = Image.open(png).convert("RGB")
    W = im.width
    wrapped = []                                                   # (text, bold) wrapped to the image width
    for i, ln in enumerate(lines):
        for seg in textwrap.wrap(ln, width=int(W / (10.4 if i == 0 else 7.9))):
            wrapped.append((seg, i == 0))
    lines = [w[0] for w in wrapped]; bold = [w[1] for w in wrapped]
    hpx = 36 + 26 * max(0, len(lines) - 1) + (10 if level == "watch" else 0)
    fig = plt.figure(figsize=(W / 100, hpx / 100), dpi=100)
    bg, fg, ed = ("#fff4d6", "#5a3a00", "#e0a100") if level == "watch" else ("#f2f3f0", "#444", "#cfd3cc")
    fig.patches.append(matplotlib.patches.Rectangle((0, 0), 1, 1, transform=fig.transFigure, facecolor=bg, edgecolor=ed, lw=2))
    y = 1 - 22 / hpx
    for i, ln in enumerate(lines):
        fig.text(0.012, y, ln, fontsize=11.5 if bold[i] else 10, color=fg, fontweight="bold" if bold[i] else "normal", va="center")
        y -= 26 / hpx
    tmp = png.with_suffix(".banner.png")
    fig.savefig(tmp, dpi=100, facecolor=bg); plt.close(fig)
    b = Image.open(tmp).convert("RGB"); tmp.unlink()
    out = Image.new("RGB", (W, b.height + im.height), "white")
    out.paste(b.resize((W, b.height)), (0, 0)); out.paste(im, (0, b.height))
    out.save(png, quality=85, method=6)


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--out-dir", default="../../assets/sst")
    a = ap.parse_args()
    out = Path(a.out_dir)
    ens_f, jet_f = out / "data" / "tcens.json", out / "data" / "tcjet.json"
    if not ens_f.exists():
        print("  no tcens.json: no recurvature watch this cycle"); return 0
    ens = json.loads(ens_f.read_text())
    jet = json.loads(jet_f.read_text()) if jet_f.exists() else {"storms": []}
    ref = json.loads((REF / "tc_downstream.json").read_text())
    init = pd.Timestamp(ens["init"].rstrip("Z"))
    ctl = {s["id"]: s for s in jet.get("storms", [])}
    alerts, others = [], []
    for s in ens["storms"]:
        sid = str(s.get("id") or "")
        if not s["models"]:
            continue
        ps = {m: q["p_recurve"] for m, q in s["models"].items()}
        med = [q["recurve_day_p10_p50_p90"][1] for q in s["models"].values() if q.get("recurve_day_p10_p50_p90")
               and q["recurve_day_p10_p50_p90"][1] is not None]
        mday = float(np.median(med)) if med else None
        on = all(p >= WATCH_P for p in ps.values()) and mday is not None and mday <= WATCH_DAY
        when = init + pd.Timedelta(days=mday or 0)
        # basin = where the CONTROL has the storm nearest the median recurvature time (Nolo 2026 is 15E but recurves near
        # 168E: its downstream is the western Pacific's); the id letter only when the control does not carry the storm
        basin = BASIN_OF.get(sid[-1:].upper())
        tr = ctl.get(sid, {}).get("track") or []
        if tr:
            k = int(np.argmin([abs(t["h"] - 24 * (mday if mday is not None else 0)) for t in tr]))
            lon360 = tr[k]["lon"] % 360
            basin = "WP" if lon360 < 180 else ("EP" if lon360 < 260 else "NA")
        if basin is None:
            continue
        sk = season_of(when.month)
        hist, hj = history_line(ref, basin, sk)
        idx = ctl.get(sid, {}).get("index", {})
        peak = max(idx.values()) if idx else None
        e = dict(id=sid, name=s["name"], basin=basin, season=sk, watch=bool(on), p_recurve=ps,
                 median_recurve_day=mday, recurve_date=when.strftime("%Y-%m-%d") if mday is not None else None,
                 control_peak_outflow_index=peak, history=hist, history_numbers=hj, short=short_line(basin, sk, hj))
        (alerts if on else others).append(e)
    level = "watch" if alerts else "none"
    lines = []
    if alerts:
        lines.append("Recurvature watch (share of AIFS-ENS / IFS-ENS members recurving, median date): " + "; ".join(
            f"{x['name']} {' / '.join(f'{100 * p:.0f}%' for p in x['p_recurve'].values())}, ~{pd.Timestamp(x['recurve_date']):%d %b}"
            for x in alerts))
        seen = []
        for x in alerts:                                          # one record line per basin and season
            if (x["basin"], x["season"]) in seen:
                continue
            seen.append((x["basin"], x["season"]))
            lines.append("Record: " + x["short"])
    else:
        lines.append(f"No recurvature watch: no Northern-Hemisphere storm has ≥ {100 * WATCH_P:.0f}% of members recurving in every "
                     f"ensemble by day {WATCH_DAY:.0f}.")
    js = dict(init=ens["init"], level=level, rule=dict(p_recurve_min=WATCH_P, by_day=WATCH_DAY, every_ensemble=True,
              strength_index_used=False, why="strong-minus-weak interaction not significant in ERA5 1991-2020"),
              watch=alerts, not_on_watch=others, lines=lines)
    (out / "data" / "tc_alert.json").write_text(json.dumps(js))
    for jf, png in ((jet_f, out / "tcjet.webp"), (ens_f, out / "tcens_tracks.webp")):
        if not jf.exists():
            continue
        d = json.loads(jf.read_text())
        stamped = (d.get("alert") or {}).get("stamped_init") == ens["init"]
        d["alert"] = dict(level=level, lines=lines, watch=[x["id"] for x in alerts], stamped_init=ens["init"])
        if png.exists() and not stamped and d.get("init") == ens["init"]:
            banner(png, lines, level)
        jf.write_text(json.dumps(d))
    print("  " + " | ".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
