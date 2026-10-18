#!/usr/bin/env python3
"""Hadley outflow and subtropical descent forecasts, member by member, from the AIFS-ENS, IFS-ENS and GEFS 00Z runs.

Every perturbed member's fields of the 00 UTC state of lead days 0..15 (hadley_branch_ref.run_fields: the analysis's
fields, hadley_branches.fields, on the 1.5 deg grid) less its model's drift against ERA5 at that lead and season
(hadley_branch_ref.Drift, release hbranch-ref-v1), in TRAILING 7-day windows ending on each lead day; the window of an
early lead still holds analysed days, the IFS HRES daily fields less their offset from ERA5 (hadley_branch_maps), so
analysis and forecast stand on ERA5's footing. A model without a drift fit, or with no lead whose skill passed, is left
out rather than drawn raw.

The loops (assets/sst/anim/hbranch_fc_<region>, one region per ensemble and "blend" for the three pooled) open with the
last LEAD_IN analysed weeks (hbranch_hist), then show the members' mean anomaly against ERA5 1991-2020 ONLY where at
least AGREE of the members give it the same sign, and each part only to the last window whose skill passed on the
archived runs (hadley_branch_ref.SKILL): the descent shading, the pressure shading, the 588-dam line and each high its
own. The tracker numbers (FC: each ensemble's median and 10-90 % at the windows that passed) go into the analysed
trackers (hadley_branch_maps.draw_trackers).

    python src/hadley_branch_fc.py run --model ifs --date 20261018 --out runs/ifs.npz   # one ensemble's windows
    python src/hadley_branch_fc.py render --date 20261018 runs/*.npz                     # loops, trackers, FC
"""

from __future__ import annotations

import pyproj  # noqa: F401  (load pyproj's libproj before xarray/eccodes)

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm

sys.path.insert(0, str(Path(__file__).parent))
import hadley_branch_maps as M
import hadley_branch_ref as H
import hadley_branches as HB

AGREE = 0.8  # shade an anomaly only where this share of the members gives it one sign
LEAD_IN = 14  # analysed weeks the loops open with
MIN_SHARE = 0.5  # a model is drawn with at least this share of its members complete
SHADED = ("wmid_s", "msl")  # the shaded anomalies, whose members' signs are counted
PREFIX = "hbranch_fc"


def analysed_before(init: pd.Timestamp) -> np.ndarray:
    """(WIN - 1, field, point): the analysed days before init, the IFS HRES daily fields less their offset from ERA5
    at the day's season; NaN on a day the archive lacks (the windows that hold it are not drawn)."""
    off = M.Offsets()
    date, D = M.load(init - pd.Timedelta(days=H.WIN))
    pos = {d: i for i, d in enumerate(date)}
    out = np.full((H.WIN - 1, len(H.FIELDS), H.ROWS.size * HB.LON.size), np.nan, np.float32)
    for k in range(H.WIN - 1):
        d = init - pd.Timedelta(days=H.WIN - 1 - k)
        if d not in pos:
            print(f"  no analysed fields for {d:%Y-%m-%d}", flush=True)
            continue
        o = off.off2(M._doy(d))
        F = {f: D[f][pos[d]] - o[f] for f in HB.FIELDS}
        F["wmid_s"] = HB.large_scale(F["wmid"], efold=M.W_EFOLD)
        out[k] = np.stack([F[f][H.ROWS] for f in H.FIELDS]).reshape(len(H.FIELDS), -1)
    return out


def run(model: str, init: pd.Timestamp, out: Path) -> bool:
    """One ensemble: its complete members less the drift, reduced (reduce) into `out`; False when it is left out."""
    skill = json.loads(H.SKILL.read_text())["models"].get(model) if H.SKILL.exists() else None
    drift = H.Drift().at(model, M._doy(init))
    if drift is None or not skill or not any(t["show"] for t in skill["targets"].values()):
        print(f"  {H.MODELS[model]['label']}: no drift fit or no lead with significant skill, left out", flush=True)
        return False
    want = H.MODELS[model]["members"]
    x = H.run_fields(model, init, list(range(1, want + 1)), group=1)
    x = x[np.isfinite(x).all(axis=(1, 2, 3, 4))]
    print(f"  {H.MODELS[model]['label']}: {x.shape[0]} of {want} members complete", flush=True)
    if x.shape[0] < MIN_SHARE * want:
        return False
    x -= drift[None]
    doc = reduce(x.reshape(*x.shape[:3], -1), analysed_before(init), init, H.Normals())
    np.savez_compressed(out, model=model, init=np.datetime64(init, "D"), **doc)
    return True


def reduce(x: np.ndarray, before: np.ndarray, init: pd.Timestamp, N: H.Normals) -> dict:
    """What render needs of the members' corrected days (member, lead, field, point): their number, the sum of their
    windows, how many lie above the normal in SHADED, and every member's KEYS (NaN in a window that holds a missing
    analysed day)."""
    w = H.member_windows(x, before)  # (member, window, field, point)
    normal = np.stack([N.window(init + pd.Timedelta(days=j)).reshape(len(H.FIELDS), -1) for j in H.LEADS])
    k = [H.FIELDS.index(f) for f in SHADED]
    shp = (len(H.FIELDS), H.ROWS.size, HB.LON.size)
    idx = np.full((w.shape[0], len(H.LEADS), len(H.KEYS)), np.nan, np.float32)
    for m in range(w.shape[0]):
        for j in H.LEADS:
            if np.isfinite(w[m, j]).all():
                idx[m, j] = H.indices(w[m, j].reshape(shp))
    pos = (w[:, :, k] > normal[None, :, k]).sum(0).astype(np.int16)
    return {"n": w.shape[0], "sum": w.sum(0), "pos": pos, "idx": idx}


def agreed(anom: np.ndarray, pos: np.ndarray, n: int) -> np.ndarray:
    """The members' mean anomaly where at least AGREE of them share its sign, NaN (blank) elsewhere."""
    share = pos / n
    return np.where((share >= AGREE) | (share <= 1 - AGREE), anom, np.nan)


def lead_in(init: pd.Timestamp) -> list:
    """The analysed weeks ending on the LEAD_IN days before init, from the analysed loop's manifest (one directory
    over)."""
    mp = M.ANIM / f"{M.DIR}_manifest.json"
    past = json.loads(mp.read_text())["regions"][M.DIR]["frames"] if mp.exists() else []
    days = {f["date"]: f for f in past}
    keys = [f"{init - pd.Timedelta(days=n):%Y-%m-%d}" for n in range(LEAD_IN, 0, -1)]
    return [
        {"file": f"../{M.DIR}/{days[k]['file']}", "date": k, "label": f"analysis · {days[k]['label']}"}
        for k in keys
        if k in days
    ]


def _panel(fig, rect, cax, field, levels, title, label, shown: bool):
    """hadley_branch_maps' panel, its colour bar standing when the shading is not shown (beyond the leads with skill)."""
    ax = fig.add_axes(rect, projection=M.PROJ)
    norm = BoundaryNorm(levels, plt.get_cmap("RdBu_r").N, extend="both")
    if shown and np.isfinite(field).any():
        ax.contourf(M.XN, HB.LAT, M.native(field), levels=levels, cmap="RdBu_r", norm=norm, extend="both")
    if not shown:
        ax.text(
            0.5,
            0.5,
            "not shown: beyond the leads whose skill is significant",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="#6f6b64",
            zorder=6,
        )
    M.geo(ax, title)
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap="RdBu_r"), cax=fig.add_axes(cax), orientation="vertical")
    cb.set_label(label, fontsize=8)
    cb.ax.tick_params(labelsize=7.5)
    return ax


def _to(show: int) -> str:
    return f"to +{show - 1} d" if show else "not shown"


def draw(mean: np.ndarray, agree: dict, z588_normal: np.ndarray, j: int, show: dict, title: str, foot: str, out: Path):
    """One window, as the analysed weekly map (hadley_branch_maps.draw_map): mean (field, row, lon) the members' mean,
    agree {shaded field: agreed anomaly}, each part drawn while j is below its `show`."""
    full = H.pad(mean)
    ok = {t: j < show.get(t, 0) for t in (*H.MAP_FIELDS, *(f"high_{k}" for k in HB.HIGHS))}
    size, rects, left = M.layout(2)
    fig = plt.figure(figsize=size)
    ax = _panel(
        fig,
        *rects[0],
        HB.PA_S_TO_HPA_DAY * H.globe(agree["wmid_s"]),
        M.LEV["desc"],
        f"Descent · {HB.DESC_LAYER[0]:.0f}–{HB.DESC_LAYER[1]:.0f} hPa vertical motion",
        "anomaly (hPa day⁻¹, + = more descent)",
        ok["wmid_s"],
    )
    if ok["wmid_s"]:
        w = M.native(HB.PA_S_TO_HPA_DAY * full["wmid_s"])
        ax.contour(M.XN, HB.LAT, w, levels=[20.0], colors="#111", linewidths=0.8)
    ax = _panel(
        fig,
        *rects[1],
        H.globe(agree["msl"]),
        M.LEV["msl"],
        "Subtropical ridges · sea-level pressure and the 588-dam line",
        "SLP anomaly (hPa)",
        ok["msl"],
    )
    ax.contour(M.XN, HB.LAT, M.native(z588_normal), levels=[HB.Z588], colors="#111", linewidths=1.0, linestyles="--")
    if ok["z500"]:
        ax.contour(M.XN, HB.LAT, M.native(full["z500"]), levels=[HB.Z588], colors="#111", linewidths=1.6)
    idx = HB.indices(full)
    for k in HB.HIGHS:
        if ok[f"high_{k}"]:
            ax.text(
                idx[f"hlon_{k}"],
                idx[f"hlat_{k}"],
                f"H\n{idx[f'high_{k}']:.0f}",
                transform=M.PC,
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="#111",
            )
    W, Hh = size
    fig.text(left / W, 1 - 0.3 / Hh, title, fontsize=13, fontweight="bold", va="center")
    fig.text(left / W, 0.12 / Hh, foot, fontsize=8.2, color="#555")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100, facecolor="white", pil_kwargs={"quality": 80, "method": 6})
    plt.close(fig)


def render(init: pd.Timestamp, paths: list[str]):
    skill = json.loads(H.SKILL.read_text())["models"]
    regions = {}
    for p in paths:
        with np.load(p) as f:
            regions[str(f["model"])] = {k: f[k] for k in ("n", "sum", "pos", "idx")}
    regions = {m: regions[m] for m in H.MODELS if m in regions}
    if len(regions) == len(H.MODELS) and "blend" in skill:
        blend = {k: sum(G[k] for G in regions.values()) for k in ("n", "sum", "pos")}
        blend["idx"] = np.concatenate([G["idx"] for G in regions.values()])
        regions = {"blend": blend, **regions}
    names = {m: H.MODELS[m]["label"] for m in regions if m != "blend"}
    if "blend" in regions:
        names = {"blend": " + ".join(names.values()), **names}
    N = H.Normals()
    shp = (H.ROWS.size, HB.LON.size)
    past = lead_in(init)
    doc = {
        "ver": f"{init:%Y%m%d}00",
        "init": f"{init:%Y-%m-%d}T00:00Z",
        "default": None,
        "selectorLabel": "Forecast",
        "start": len(past),  # opens on the first forecast window (sst_anim ?start=manifest)
        "regions": {},
    }
    for rid, G in regions.items():
        d = M.ANIM / f"{PREFIX}_{rid}"  # the viewer reads <anim>/<region id>/<file>
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("F*.webp"):
            old.unlink()
        targets = skill[rid]["targets"]
        show = {t: v["show"] for t, v in targets.items()}
        n_win = max(show[f] for f in SHADED)
        if not n_win:
            print(f"  {PREFIX}_{rid}: no shaded field with significant skill, left out", flush=True)
            continue
        whose = "each member less its model's" if rid == "blend" else "members less the model's"
        frames = list(past)
        for j in range(n_win):
            end = init + pd.Timedelta(days=j)
            mean = (G["sum"][j] / G["n"]).reshape(len(H.FIELDS), *shp)
            if not np.isfinite(mean).all():
                print(f"  {PREFIX}_{rid} +{j} d: an analysed day is missing, not drawn", flush=True)
                continue
            normal = N.window(end)
            agree = {
                f: agreed(mean[H.FIELDS.index(f)] - normal[H.FIELDS.index(f)], G["pos"][j, i].reshape(shp), G["n"])
                for i, f in enumerate(SHADED)
            }
            fp = d / f"F{j:02d}.webp"
            draw(
                mean,
                agree,
                M.clim_map(N.cl, "z500", M._doy(end - pd.Timedelta(days=(H.WIN - 1) / 2))),
                j,
                show,
                f"Subtropical descent and ridges — {names[rid]} {init:%Y-%m-%d} 00Z, 7-day mean ending "
                f"{end:%Y-%m-%d} (+{j} d)",
                f"colour = the members' mean anomaly against ERA5 1991–2020 at the window's midpoint, where ≥{AGREE:.0%} "
                f"of the {G['n']} members agree on its sign · {whose} drift against ERA5\ntop: vertical motion strongly "
                f"smoothed (degree {M.W_EFOLD:.0f} damped by 1/e), black line = descent of 20 hPa/day · bottom: solid = "
                "the members' mean 588-dam line, dashed = the normal's, H = their highs (hPa)\nearlier days of a window: "
                "the analyses · each part to the last window with significant skill on the archived runs "
                f"({skill[rid]['first'][:7]} to {skill[rid]['last'][:7]}): descent {_to(show['wmid_s'])}, pressure "
                f"{_to(show['msl'])}, 588-dam line {_to(show['z500'])}",
                fp,
            )
            frames.append(
                {"file": fp.name, "date": f"{end:%Y-%m-%d}", "label": f"+{j} d · week ending {end:%a %b %-d}"}
            )
        frames = [dict(f, idx=i) for i, f in enumerate(frames)]
        doc["regions"][f"{PREFIX}_{rid}"] = {"label": names[rid], "n_frames": len(frames), "frames": frames}
        print(f"  {PREFIX}_{rid}: {len(past)} analysed + {len(frames) - len(past)} forecast windows", flush=True)
    if not doc["regions"]:
        raise SystemExit("no region with a forecast window to draw")
    doc["default"] = next(iter(doc["regions"]))
    (M.ANIM / f"{PREFIX}_manifest.json").write_text(json.dumps(doc))
    tracker_numbers(init, regions, skill)
    off = M.Offsets()
    date, D = M.load(init - pd.Timedelta(days=M.DAYS + H.WIN))
    rows, ends = [], []
    for end in date[date > date.max() - pd.Timedelta(days=M.DAYS)]:
        w = M.week(date, D, end, off)
        if w is not None:
            rows.append(HB.indices(w[0]))
            ends.append(end)
    M.draw_trackers(pd.DataFrame(rows, index=pd.DatetimeIndex(ends)), N.cl, only=M.FIELD_TRACKERS)


def tracker_numbers(init: pd.Timestamp, regions: dict, skill: dict):
    """M.FC: each ensemble's median and 10-90 % of the members' KEYS at the windows whose skill passed."""
    doc = {"init": f"{init:%Y-%m-%d}", "valid": [f"{init + pd.Timedelta(days=j):%Y-%m-%d}" for j in H.LEADS]}
    doc["models"] = {}
    for m, G in regions.items():
        if m == "blend":
            continue
        keys = {}
        for i, k in enumerate(H.KEYS):
            x = G["idx"][:, : skill[m]["targets"][k]["show"], i]
            if np.isfinite(x).any():  # a window holding a missing analysed day stays a gap (null)
                q = np.where(np.isfinite(x).all(0), np.percentile(x, (10, 50, 90), axis=0), np.nan)
                keys[k] = {s: [_num(v) for v in q[a]] for a, s in enumerate(("p10", "p50", "p90"))}
        doc["models"][m] = {"label": H.MODELS[m]["label"], "members": int(G["n"]), "keys": keys}
    M.FC.parent.mkdir(parents=True, exist_ok=True)
    M.FC.write_text(json.dumps(doc) + "\n")


def _num(v: float) -> float | None:
    return round(float(v), 3) if np.isfinite(v) else None


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("run")
    s.add_argument("--model", required=True, choices=sorted(H.MODELS))
    s.add_argument("--date", required=True, help="00Z run, YYYYMMDD")
    s.add_argument("--out", required=True)
    s = sub.add_parser("render")
    s.add_argument("--date", required=True, help="00Z run, YYYYMMDD")
    s.add_argument("runs", nargs="*")
    a = ap.parse_args()
    init = pd.Timestamp(a.date)
    if a.cmd == "run":
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        run(a.model, init, Path(a.out))
        return 0
    if not a.runs:
        print("no ensemble to draw", file=sys.stderr)
        return 1
    render(init, a.runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
