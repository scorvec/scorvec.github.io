#!/usr/bin/env python3
"""The regional reflection index (Messori et al. 2022) for every ensemble: one figure, assets/sst/reflection_index.webp.

RI = z(Siberia) - z(Canada), v*T* at 100 hPa over Siberia (45-75N 140-200E, where upward waves leave the troposphere)
and Canada (45-75N 230-280E, where reflected waves come back down), each standardised by its ERA5 1991-2020 day-of-year
mean and sd (heatflux100.REGIONS / regional_block; the members' boxes in each model's heat-flux JSON). The analysis
tail is ERA5 (reference/vt100_era5.nc) and, after it, the AIFS control's step-0 fields (data/heatflux100_history.json).
The third of Messori's criteria, a reflecting upper stratosphere (u at 60N weaker aloft than at 10 hPa), is read from
GEFS, the one ensemble here with winds to 1 hPa (data/upperstrat.json). What followed the ERA5 events at the surface is
reference/reflection_index.json (build_reflection_index.py).

    python reflection_index.py      # reads the published JSON, draws the figure
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import heatflux100 as HF

DATA = REPO / "assets" / "sst" / "data"
OUT = REPO / "assets" / "sst" / "reflection_index.webp"
REF = HERE / "reference" / "reflection_index.json"
MODELS = {"aifs": ("AIFS-ENS", "heatflux100.json"), "ifs": ("IFS-ENS", "ifs_heatflux100.json"),
          "gefs": ("GEFS", "gefs_heatflux100.json"), "geps": ("GEPS", "geps_heatflux100.json")}  # fmt: skip
STYLE = {"aifs": "#b4541f", "ifs": "#1f4b6e", "gefs": "#3b88c4", "geps": "#2f7d4f"}  # nh_vortex's
TAIL = 45
INK, MUTED = "#1f2328", "#6f6b64"


def _read(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def model_series(clim) -> dict:
    """{model: {init, valid, z_sib, z_can, ri, ri_p10, ri_p90, event_share}} from the models' heat-flux JSON."""
    out = {}
    for m, (_, f) in MODELS.items():
        d = _read(DATA / f)
        reg = (d or {}).get("hemispheres", {}).get("nh", {}).get("regional")
        if not reg:
            continue
        v = pd.DatetimeIndex([x[:10] for x in d["valid"]])
        z = {k: (np.array(reg[k]["mean"]) - HF.clim_at(clim, "nh", k, "mean", v)) / HF.clim_at(clim, "nh", k, "sd", v)
             for k in HF.REGIONS}  # fmt: skip
        ri = reg.get("ri") or {}
        out[m] = {
            "init": pd.Timestamp(d["init"]),
            "valid": v,
            "z_sib": z["sib"],
            "z_can": z["can"],
            # the standardisation is linear, so the members' mean index is the index of the members' mean boxes
            "ri": z["sib"] - z["can"],
            "ri_p10": np.array(ri["p10"]) if ri else None,
            "ri_p90": np.array(ri["p90"]) if ri else None,
            "event_share": reg.get("event_share"),
        }
    return out


def analysis(clim) -> pd.DataFrame:
    """Daily z_sib, z_can and RI: ERA5, then the AIFS control's analyses after it."""
    E = xr.open_dataset(HF.CLIM.parent / "vt100_era5.nc")
    raw = pd.DataFrame({k: E[f"nh_{k}"].mean("hour").to_series() for k in HF.REGIONS})
    hist = _read(DATA / "heatflux100_history.json") or {}
    live = {k: v["nh_reg"] for k, v in hist.items() if "nh_reg" in v}
    if live:
        lv = pd.DataFrame(live).T
        lv.index = pd.to_datetime(lv.index, format="%Y-%m-%d %H")
        lv = lv.astype(float).groupby(lv.index.normalize()).mean()
        raw = pd.concat([raw, lv[lv.index > raw.index[-1]]])
    t = pd.DatetimeIndex(raw.index)
    z = pd.DataFrame({f"z_{k}": (raw[k] - HF.clim_at(clim, "nh", k, "mean", t)) / HF.clim_at(clim, "nh", k, "sd", t)
                      for k in HF.REGIONS}, index=t)  # fmt: skip
    z["ri"] = z.z_sib - z.z_can
    return z.dropna()


def gefs_shear() -> list[str] | None:
    """Messori's third criterion from GEFS: u(60N) at 1 hPa minus 10 hPa on its last day, against the MERRA-2 normal."""
    d = _read(DATA / "upperstrat.json")
    try:
        u = d["models"]["gefs"]["hemis"]["nh"]["u60"]
        last = sorted(u["1"], key=lambda k: int(k[1:]))[-1]
        a, b = u["1"][last], u["10"][last]
    except (TypeError, KeyError, IndexError):
        return None
    sh, norm = a["value"] - b["value"], a["clim_mean"] - b["clim_mean"]
    verdict = "a reflecting layer" if sh < 0 else "no reflecting layer"
    return [
        f"GEFS u(60°N), 1 minus 10 hPa, {pd.Timestamp(a['valid']):%d %b}:",
        f"{sh:+.0f} m/s (normal {norm:+.0f}): {verdict}",
    ]


def render(out: Path = OUT) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    clim = xr.open_dataset(HF.CLIM).load()
    ms = model_series(clim)
    if not ms:
        raise SystemExit("no model carries the regional boxes yet")
    obs = analysis(clim)
    t1 = max(v["init"] for v in ms.values())
    x0, x1 = (t1 - pd.Timedelta(days=TAIL)).normalize(), max(v["valid"][-1] for v in ms.values())
    ref = _read(REF)

    fig = plt.figure(figsize=(14.2, 9.0), dpi=112)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.6, 1.6, 1.15], height_ratios=[1.25, 1], left=0.055, right=0.985,
                          top=0.87, bottom=0.17, hspace=0.34, wspace=0.22)  # fmt: skip
    ax = fig.add_subplot(gs[0, :2])
    axs = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    side = fig.add_subplot(gs[:, 2])
    o = obs[(obs.index >= x0) & (obs.index <= t1)]
    ax.plot(o.index, o.ri, color=INK, lw=2.0, label="analysis (ERA5, then the AIFS control)")
    for m, v in ms.items():
        if v["ri_p10"] is not None:
            ax.fill_between(v["valid"], v["ri_p10"], v["ri_p90"], color=STYLE[m], alpha=0.10, lw=0)
        ax.plot(v["valid"], v["ri"], color=STYLE[m], lw=1.9, label=f"{MODELS[m][0]} ({v['init']:%d %b %HZ})")
    ax.axhline(HF.RI_THRESHOLD, color="#b3122b", lw=1.1, ls="--", label=f"Messori threshold (RI > {HF.RI_THRESHOLD:g})")
    ax.axhline(0, color="#9a958c", lw=0.8)
    ax.set_title(
        "Regional reflection index: z(Siberia) − z(Canada), 100 hPa", loc="left", fontsize=11, fontweight="bold"
    )
    ax.set_ylabel("RI (sd)")
    ax.legend(loc="upper left", fontsize=7.8, ncol=3, framealpha=0.9)
    for a, k, title in ((axs[0], "z_sib", "Siberia (45–75°N, 140–200°E): waves going up"),
                        (axs[1], "z_can", "Canada (45–75°N, 230–280°E): negative = waves coming down")):  # fmt: skip
        a.plot(o.index, o[k], color=INK, lw=1.8)
        for m, v in ms.items():
            a.plot(v["valid"], v[k], color=STYLE[m], lw=1.6)
        a.axhline(0, color="#9a958c", lw=0.8)
        a.set_title(title, loc="left", fontsize=9.8, fontweight="bold")
        a.set_ylabel("standardised v*T* (sd)")
    for a in (ax, *axs):
        a.axvline(t1, color="#8a8680", lw=1)
        a.set_xlim(x0, x1)
        a.grid(alpha=0.22)
        a.xaxis.set_major_locator(mdates.DayLocator(bymonthday=(1, 15)))
        a.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        for s in ("top", "right"):
            a.spines[s].set_visible(False)

    side.axis("off")
    lines = ["Members with an event (RI > 1 for 10+ days,", "Siberia upward and Canada downward):", ""]
    for m, v in ms.items():
        share = v["event_share"]
        lines.append(f"  {MODELS[m][0]:9s} {'—' if share is None else f'{100 * share:.0f} %'}")
    sh = gefs_shear()
    lines += ["", "Messori's third criterion, a reflecting", "upper stratosphere:"]
    lines += [f"  {x}" for x in sh] if sh else ["  (no GEFS upper-air run)"]
    if ref:
        e = {w["days"][0]: w for w in ref["epo_after"]}
        tn = {w["days"][0]: w for w in ref["tna_after"]}
        lines += [
            "",
            f"After the {len(ref['events'])} ERA5 events",
            f"({ref['period'][0][:4]}–{ref['period'][1][:4]}, December–March):",
            f"  EPO, days 1–5: {e[1]['mean']:+.2f} sd (p {e[1]['p']:.2g})",
            f"  EPO, days 6–12: {e[6]['mean']:+.2f} sd (p {e[6]['p']:.2g})",
            f"  N. America 2 m T before onset: {tn[-5]['mean']:+.1f} K",
            f"  and over days 6–12: {tn[6]['mean']:+.1f} K (p {tn[6]['p']:.2g})",
        ]
    side.text(0, 1, "\n".join(lines), transform=side.transAxes, va="top", fontsize=8.6, color=INK, family="DejaVu Sans",
              linespacing=1.5)  # fmt: skip

    fig.text(0.055, 0.975, f"Regional reflection index (Messori et al. 2022) — newest run {t1:%d %b %Y %HZ}", fontsize=15,
             fontweight="bold", color=INK, va="top")  # fmt: skip
    fig.text(0.055, 0.94, "Upward wave flux over Siberia with downward flux over Canada: waves reflected back to the "
             "surface over North America, which has meant a ridge over Alaska (EPO negative) and cold a week later.",
             fontsize=9.6, color=MUTED, va="top")  # fmt: skip
    fig.text(0.055, 0.03, "v*T* (departures from the zonal mean) at 100 hPa, each box standardised by its ERA5 1991–2020 "
             "day-of-year mean and sd. Lines: ensemble means; shading: members 10–90 %, where the run carries them.\n"
             "Messori's events are December–March. Sources: ECMWF open data (CC BY 4.0, Google Cloud mirror) · NOAA GEFS "
             "(AWS Open Data) · ECCC GEPS (MSC Datamart) · ERA5 (C3S).", fontsize=7.8, color=MUTED, linespacing=1.45)  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=112, facecolor="white", pil_kwargs={"quality": 88, "method": 6})
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    render(Path(sys.argv[1]) if len(sys.argv) > 1 else OUT)
