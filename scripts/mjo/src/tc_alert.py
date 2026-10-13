#!/usr/bin/env python3
"""Recurvature watch for live storms (user 2026-10-02; diagnostic only since 2026-10-03).

Rule: a Northern-Hemisphere storm is on WATCH when at least WATCH_P of the members of EVERY ensemble that tracks it
(AIFS-ENS and IFS-ENS, from data/tcens.json) recurve, with the median recurvature by day WATCH_DAY. North Pacific and
North Atlantic only: a North Indian storm (id letter A/B, or the control west of 100E) is listed, never on watch.
2026-10-03 (user: "keep the TC-jet stuff as diagnostic as possible, leave out the stuff about how the flow could be changed
by the recurving storm"): the banner no longer quotes the 1991-2020 record of what followed past recurvatures; it says
which storms are likely to recurve, in how many members, and when - nothing about consequences.

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

WATCH_P = 0.5
WATCH_DAY = 10.0
BASIN_OF = {"W": "WP", "E": "EP", "C": "EP", "L": "NA", "A": "NI", "B": "NI"}
BASIN_LAB = {"WP": "western North Pacific", "EP": "eastern North Pacific", "NA": "North Atlantic", "NI": "North Indian Ocean"}
NI_LON = 100.0               # west of 100E the control's position is in the North Indian Ocean (Arabian Sea / Bay of Bengal)
MLAB = {"aifs": "AIFS-ENS", "ifs": "IFS-ENS", "aifs-ens": "AIFS-ENS", "ifs-ens": "IFS-ENS"}


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
    init = pd.Timestamp(ens["init"].rstrip("Z"))
    # the control's tracks only from the SAME cycle (2026-10-12 review: a stale file must never be combined with a new one)
    ctl = {s["id"]: s for s in jet.get("storms", [])} if jet.get("init") == ens["init"] else {}
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
        core = sid[4:] if sid.startswith("ifs-") else sid             # an IFS-ENS-only system is "ifs-71W" (tc_ens, 2026-10-12)
        basin = BASIN_OF.get(core.split("-")[0][-1:].upper())         # tc_ens names a 2nd group under one id "72E-2"
        # the control's track through the cluster's own control ids (tc_ens "control_ids"): the control's tracker id can
        # differ from the cluster's (12 Oct 12Z: control 72L is in cluster 73L). An older tcens.json without the field: the
        # id itself, as before
        cids = s["control_ids"] if "control_ids" in s else ([core] if not sid.startswith("ifs-") else [])
        tr = next((ctl[c]["track"] for c in cids if c in ctl and ctl[c].get("track")), [])
        if tr:
            k = int(np.argmin([abs(t["h"] - 24 * (mday if mday is not None else 0)) for t in tr]))
            lon360 = tr[k]["lon"] % 360
            basin = "NI" if lon360 < NI_LON else "WP" if lon360 < 180 else ("EP" if lon360 < 260 else "NA")
        if basin is None:
            continue
        if basin == "NI":
            # handled explicitly: a North Indian storm that "recurves" turns into the subcontinent or Myanmar, not into
            # the mid-latitude westerlies the watch is about; it is listed, never put on watch
            others.append(dict(id=sid, name=s["name"], basin=basin, watch=False, p_recurve=ps, median_recurve_day=mday,
                               recurve_date=when.strftime("%Y-%m-%d") if mday is not None else None,
                               note="North Indian Ocean: not part of the recurvature watch"))
            continue
        e = dict(id=sid, name=s["name"], basin=basin, watch=bool(on), p_recurve=ps,
                 median_recurve_day=mday, recurve_date=when.strftime("%Y-%m-%d") if mday is not None else None)
        (alerts if on else others).append(e)
    level = "watch" if alerts else "none"
    lines = []
    if alerts:
        # each share labelled with its ensemble (2026-10-12 review: a storm in one ensemble showed one % under
        # "AIFS-ENS / IFS-ENS")
        lines.append("Recurvature watch (share of members recurving, median date): " + "; ".join(
            f"{x['name']} " + ", ".join(f"{MLAB.get(m, m)} {100 * p:.0f}%" for m, p in x["p_recurve"].items())
            + f", ~{pd.Timestamp(x['recurve_date']):%d %b}" for x in alerts))
    else:
        lines.append(f"No recurvature watch: no North Pacific or North Atlantic storm has ≥ {100 * WATCH_P:.0f}% of members recurving in every "
                     f"ensemble by day {WATCH_DAY:.0f}.")
    js = dict(init=ens["init"], level=level, rule=dict(p_recurve_min=WATCH_P, by_day=WATCH_DAY, every_ensemble=True),
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
