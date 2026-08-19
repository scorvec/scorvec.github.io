#!/usr/bin/env python3
"""Quantile-mapping bias correction for rain, by intensity / space / lead.

Why quantile mapping and not the conditional mean. E[obs|pred] is the
minimum-MSE estimator and it SHRINKS: feeding it a 45 mm IMERG day
returns ~12 mm, and applying it everywhere collapses rainfall variance.
That is the wrong objective when the downstream target is inflow SPIKES,
which need the heavy tail to survive. Quantile mapping instead matches
the whole distribution - it fixes the frequency bias (IMERG produces
30 mm basin means 3.9x too often) while preserving day-to-day rank and
keeping a heavy day heavy.

Two stages, because the two corrections have very different sample sizes:

  stage A   model -> IMERG space, per (model, region, lead band).
            57 archived cycles x 6 regions x 15 leads. Plenty.
  stage B   IMERG -> gauge space, per region.
            2,584 region-days with >= 8 IDEAM gauges. Enough for the
            body of the distribution, thin above 30 mm.

Both are refreshable: rerun as cycles and gauge-days accumulate and the
mappings tighten. Nothing here is hard-coded to a fitting window.

Validation is blocked and out-of-sample - the mapping is fitted on all
year-blocks but the one being scored, so a mapping cannot be graded on
the days that built it.

    python scripts/sst/rain_qmap.py --fit --figure

Outputs  ~/colombia_hydro/data/rain_qmap.json     (the mappings)
         ~/colombia_hydro/site/rain_qmap.webp     (the graphic)
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import imerg_precip as IP                                          # noqa: E402
from hydro_region_rain import region_weights_energy                # noqa: E402

PRIV = Path.home() / "colombia_hydro"
GAUGES = PRIV / "raw" / "gauges"
ARCH = PRIV / "raw" / "fcst_rain"
OUT_JSON = PRIV / "data" / "rain_qmap.json"
OUT_PNG = PRIV / "site" / "rain_qmap.webp"
REGIONS = ["ANTIOQUIA", "CALDAS", "CARIBE", "CENTRO", "ORIENTE", "VALLE"]
BANDS = [(1, 3), (4, 7), (8, 15)]
NQ = 60                       # quantile knots
MIN_N = 150                   # below this a mapping is not fitted
# Stage A samples are ENSEMBLE MEMBERS, so a cell can show n=810 while its
# truth side rests on 17 distinct dates. Members inform the FORECAST
# distribution but carry no extra information about truth, and a 60-knot
# map fitted on ~20 independent days is noise. Stage A is therefore gated
# on DISTINCT VALID DAYS, not row count, and stays unfitted until the
# archive is deep enough. As of 2026-08-19 the whole archive spans 24
# distinct days, so nothing qualifies and the engine keeps its scalar
# factors - the gate opens by itself as cycles accumulate.
MIN_DAYS_A = 120              # distinct valid days required for a stage-A map


def qmap_fit(src, tgt, nq=NQ):
    """Empirical quantile map. Returns knots (src_q, tgt_q).

    Both samples are climatological, not paired - quantile mapping does
    not need matched days, which is what lets stage B use gauge-days the
    forecast archive never covered."""
    qs = np.linspace(0, 100, nq)
    return (np.percentile(src, qs).tolist(), np.percentile(tgt, qs).tolist())


def qmap_apply(x, sq, tq):
    """Apply a mapping. Above the highest knot the correction ratio is
    held constant rather than extrapolated linearly: the top knot is the
    thinnest part of the sample and a linear extension there would let
    one outlier set the slope for every future extreme."""
    sq, tq = np.asarray(sq, float), np.asarray(tq, float)
    x = np.asarray(x, float)
    out = np.interp(x, sq, tq)
    hi = x > sq[-1]
    if hi.any() and sq[-1] > 0:
        out[hi] = x[hi] * (tq[-1] / sq[-1])
    return out


def basin_gauge_imerg(min_gauges=8):
    """[(day, region, gauge_mean, imerg_area)] over the gauge archive."""
    ml, mt = IP._grid_axes()
    lons, lats = np.sort(IP._LON[ml]), np.sort(IP._LAT[mt])
    W = region_weights_energy(lons, lats, REGIONS)
    masks = {r: (np.asarray(W[r]).reshape(len(lats), len(lons)) > 0) for r in W}
    wts = {r: np.asarray(W[r]).reshape(len(lats), len(lons)) for r in W}
    rows = []
    for f in sorted(glob.glob(str(GAUGES / "*.json"))):
        day = Path(f).stem
        st = json.loads(Path(f).read_text())
        npy = IP.DAILY_CACHE / f"{day}.npy"
        if not st or not npy.exists():
            continue
        grid = np.load(npy)
        pts = []
        for v in st.values():
            mm = float(v["mm"])
            if not (0 <= mm <= 450):
                continue
            j = int(np.abs(lons - v["lo"]).argmin())
            i = int(np.abs(lats - v["la"]).argmin())
            pts.append((i, j, mm))
        for r in masks:
            sel = [p for p in pts if masks[r][p[0], p[1]]]
            if len(sel) < min_gauges:
                continue
            rows.append((f"{day[:4]}-{day[4:6]}-{day[6:]}", r,
                         float(np.mean([p[2] for p in sel])),
                         float((grid * wts[r]).sum())))
    return rows


def model_imerg_pairs(member_level=True):
    """Distribution samples for stage A.

    Returns [(valid, model, region, lead, value, imerg_truth)] where
    `value` is an individual ENSEMBLE MEMBER, not the ensemble mean.

    This matters and it is easy to get wrong. colombia_forecast.py applies
    the correction to each member separately, so the mapping must be
    fitted on the distribution it will actually be applied to. The pooled
    member distribution of a well-calibrated ensemble should match the
    climatological distribution of truth - that is what calibration
    means - whereas the ensemble MEAN is narrower than truth by
    construction. A map fitted on means and applied to members would
    squeeze every member toward the mean and silently collapse ensemble
    spread.
    """
    ml, mt = IP._grid_axes()
    lons, lats = np.sort(IP._LON[ml]), np.sort(IP._LAT[mt])
    W = region_weights_energy(lons, lats, REGIONS)
    wts = {r: np.asarray(W[r]).reshape(len(lats), len(lons)) for r in W}
    truth = {}

    def obs(day):
        if day not in truth:
            npy = IP.DAILY_CACHE / f"{day.replace('-','')}.npy"
            truth[day] = (None if not npy.exists()
                          else {r: float((np.load(npy) * wts[r]).sum())
                                for r in wts})
        return truth[day]

    rows = []
    for f in sorted(ARCH.glob("*.json.gz")):
        try:
            d = json.load(gzip.open(f, "rt"))
        except Exception:
            continue
        be = d.get("basins_energy") or {}
        valid = d.get("valid") or []
        n = int(d.get("n_members") or 1)
        mdl = d.get("model")
        for rg, arr in be.items():
            a = np.asarray(arr, float)
            if a.size % n == 0 and a.size // n == len(valid):
                a = a.reshape(n, -1)            # (member, lead)
            elif a.size == len(valid):
                a = a.reshape(1, -1)
            else:
                continue
            if not member_level:
                a = a.mean(0, keepdims=True)
            for L, v in enumerate(valid):
                if L < 1:
                    continue
                o = obs(v)
                if o is None or rg not in o:
                    continue
                for mi in range(a.shape[0]):
                    rows.append((v, mdl, rg, L, float(a[mi, L]), o[rg]))
    return rows


def band_of(lead):
    for i, (a, b) in enumerate(BANDS):
        if a <= lead <= b:
            return i
    return len(BANDS) - 1


def fbi(pred, obs, ts):
    out = []
    for t in ts:
        po = (obs >= t).mean()
        if po * len(obs) < 5:
            out.append(np.nan)
        else:
            out.append((pred >= t).mean() / po)
    return np.array(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--figure", action="store_true")
    a = ap.parse_args(argv)

    print("stage B pairs (gauge vs IMERG) ...", flush=True)
    gb = basin_gauge_imerg()
    print(f"  {len(gb)} region-days")
    print("stage A pairs (model vs IMERG) ...", flush=True)
    ma = model_imerg_pairs()
    print(f"  {len(ma)} model-region-lead rows")

    maps = {"stageB": {}, "stageA": {}, "meta": {
        "n_gauge_days": len(gb), "n_model_rows": len(ma),
        "bands": BANDS, "nq": NQ}}

    # ---- stage B: IMERG -> gauge, per region + pooled ----
    for r in REGIONS + ["POOLED"]:
        sel = [x for x in gb if r == "POOLED" or x[1] == r]
        if len(sel) < MIN_N:
            continue
        s = np.array([x[3] for x in sel])
        t = np.array([x[2] for x in sel])
        sq, tq = qmap_fit(s, t)
        maps["stageB"][r] = {"src_q": sq, "tgt_q": tq, "n": len(sel)}

    # ---- stage A: model -> IMERG, per (model, region, band) ----
    skipped = []
    for mdl in sorted({x[1] for x in ma}):
        for r in REGIONS + ["POOLED"]:
            for bi in range(len(BANDS)):
                sel = [x for x in ma if x[1] == mdl and band_of(x[3]) == bi
                       and (r == "POOLED" or x[2] == r)]
                ndays = len({x[0] for x in sel})
                if len(sel) < MIN_N or ndays < MIN_DAYS_A:
                    skipped.append((f"{mdl}|{r}|{bi}", len(sel), ndays))
                    continue
                s = np.array([x[4] for x in sel])
                t = np.array([x[5] for x in sel])
                sq, tq = qmap_fit(s, t)
                maps["stageA"][f"{mdl}|{r}|{bi}"] = {
                    "src_q": sq, "tgt_q": tq, "n": len(sel), "n_days": ndays}

    if skipped:
        worst = max(d for _, _, d in skipped)
        print(f"  stage A: {len(skipped)} cells UNFITTED - deepest has "
              f"{worst} distinct valid days against a {MIN_DAYS_A} floor.")
        print("  The engine keeps its scalar bias factors until this clears.")

    if a.fit:
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(json.dumps(maps))
        print(f"wrote {len(maps['stageB'])} stage-B and "
              f"{len(maps['stageA'])} stage-A mappings -> {OUT_JSON}")

    # ---- blocked out-of-sample validation ----
    print("\nstage B, leave-one-year-out (IMERG -> gauge, pooled):")
    yrs = sorted({x[0][:4] for x in gb})
    raw_all, cor_all, obs_all = [], [], []
    for y in yrs:
        tr = [x for x in gb if x[0][:4] != y]
        te = [x for x in gb if x[0][:4] == y]
        if len(tr) < MIN_N or len(te) < 30:
            continue
        sq, tq = qmap_fit(np.array([x[3] for x in tr]),
                          np.array([x[2] for x in tr]))
        s = np.array([x[3] for x in te])
        o = np.array([x[2] for x in te])
        c = qmap_apply(s, sq, tq)
        raw_all.append(s); cor_all.append(c); obs_all.append(o)
        print(f"  {y}: n={len(te):4}  mean obs {o.mean():5.2f}  "
              f"raw {s.mean():5.2f}  corrected {c.mean():5.2f}   "
              f"sd obs {o.std():5.2f} raw {s.std():5.2f} cor {c.std():5.2f}")
    R = np.concatenate(raw_all); C = np.concatenate(cor_all)
    O = np.concatenate(obs_all)
    ts = np.array([1, 2, 5, 10, 15, 20, 30])
    print(f"\n  pooled OOS n={len(O)}")
    print(f"    {'thresh':>7} {'FBI raw':>9} {'FBI corr':>9}")
    for t, fr, fc_ in zip(ts, fbi(R, O, ts), fbi(C, O, ts)):
        print(f"    {t:7.0f} {fr:9.2f} {fc_:9.2f}")
    print(f"  RMSE raw {np.sqrt(((R-O)**2).mean()):.3f} -> "
          f"corrected {np.sqrt(((C-O)**2).mean()):.3f}")
    print(f"  correlation preserved: raw {np.corrcoef(R,O)[0,1]:.3f} -> "
          f"corrected {np.corrcoef(C,O)[0,1]:.3f}")
    print(f"  variance ratio vs obs: raw {R.std()/O.std():.2f} -> "
          f"corrected {C.std()/O.std():.2f}   (1.00 = distribution matched)")

    if a.figure:
        figure(gb, ma, maps, R, C, O, ts)
    return 0


def figure(gb, ma, maps, R, C, O, ts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    NAVY = "#13273d"
    fig = plt.figure(figsize=(15.2, 10.4))
    hd = fig.add_axes([0, 0.945, 1, 0.055]); hd.set_axis_off()
    hd.add_patch(plt.Rectangle((0, 0), 1, 1, transform=hd.transAxes,
                               facecolor=NAVY))
    hd.text(0.012, 0.62, "RAINFALL BIAS CORRECTION — QUANTILE MAPPING BY "
            "INTENSITY, REGION AND LEAD", transform=hd.transAxes,
            color="white", fontsize=14, fontweight="bold", va="center")
    hd.text(0.012, 0.2, f"stage B fitted on {maps['meta']['n_gauge_days']} "
            f"region-days vs IDEAM gauges · stage A on "
            f"{maps['meta']['n_model_rows']} model-region-lead rows vs IMERG · "
            "validation is leave-one-year-out", transform=hd.transAxes,
            color="#b9c6d4", fontsize=8.6, va="center")

    # 1. the mapping itself, per region
    ax = fig.add_axes([0.045, 0.56, 0.28, 0.33])
    xx = np.linspace(0, 50, 200)
    for r, c in zip(REGIONS, plt.cm.viridis(np.linspace(0, .85, len(REGIONS)))):
        if r not in maps["stageB"]:
            continue
        m = maps["stageB"][r]
        ax.plot(xx, qmap_apply(xx, m["src_q"], m["tgt_q"]), color=c, lw=1.6,
                label=r.title())
    ax.plot([0, 50], [0, 50], color="#888", ls=":", lw=1.2, label="no change")
    ax.set_xlabel("raw IMERG basin mean (mm/day)", fontsize=9)
    ax.set_ylabel("corrected → gauge space (mm/day)", fontsize=9)
    ax.set_title("A · stage-B map, by region (spatial term)", fontsize=10,
                 fontweight="bold")
    ax.legend(fontsize=6.6, frameon=False, loc="upper left")
    ax.grid(alpha=.25); ax.set_xlim(0, 50); ax.set_ylim(0, 50)

    # 2. FBI before/after
    ax = fig.add_axes([0.375, 0.56, 0.27, 0.33])
    fr, fc_ = fbi(R, O, ts), fbi(C, O, ts)
    ax.plot(ts, fr, "o-", color="#c62828", lw=2, ms=5, label="raw IMERG")
    ax.plot(ts, fc_, "o-", color="#1f7a4d", lw=2, ms=5, label="quantile-mapped")
    ax.axhline(1, color="#333", ls="--", lw=1.2)
    ax.set_xlabel("threshold (mm/day)", fontsize=9)
    ax.set_ylabel("frequency bias index", fontsize=9)
    ax.set_title("B · FBI, out-of-sample", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7.5, frameon=False); ax.grid(alpha=.25)
    ax.set_ylim(0, max(4.2, np.nanmax(fr) * 1.1))

    # 3. Q-Q
    ax = fig.add_axes([0.70, 0.56, 0.27, 0.33])
    qs = np.linspace(1, 99.5, 60)
    ax.plot(np.percentile(O, qs), np.percentile(R, qs), "o", color="#c62828",
            ms=3.4, label="raw IMERG")
    ax.plot(np.percentile(O, qs), np.percentile(C, qs), "o", color="#1f7a4d",
            ms=3.4, label="quantile-mapped")
    hi = max(np.percentile(O, 99.5), np.percentile(R, 99.5))
    ax.plot([0, hi], [0, hi], color="#333", ls="--", lw=1.2)
    ax.set_xlabel("gauge quantile (mm/day)", fontsize=9)
    ax.set_ylabel("estimate quantile (mm/day)", fontsize=9)
    ax.set_title("C · distribution match (Q–Q)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left"); ax.grid(alpha=.25)

    # 4. stage-A maps by lead band
    ax = fig.add_axes([0.045, 0.09, 0.28, 0.33])
    xx = np.linspace(0, 40, 200)
    styles = {"aifs": "-", "ifs": "--"}
    cols = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for mdl in ("aifs", "ifs"):
        for bi, (lo, hi_) in enumerate(BANDS):
            k = f"{mdl}|POOLED|{bi}"
            if k not in maps["stageA"]:
                continue
            m = maps["stageA"][k]
            ax.plot(xx, qmap_apply(xx, m["src_q"], m["tgt_q"]),
                    styles[mdl], color=cols[bi], lw=1.6,
                    label=f"{mdl.upper()} d{lo}-{hi_}")
    ax.plot([0, 40], [0, 40], color="#888", ls=":", lw=1.2)
    ax.set_xlabel("raw model basin mean (mm/day)", fontsize=9)
    ax.set_ylabel("corrected → IMERG space (mm/day)", fontsize=9)
    ax.set_title("D · stage-A map, by lead band (lead term)", fontsize=10,
                 fontweight="bold")
    ax.legend(fontsize=6.4, frameon=False, loc="upper left", ncol=2)
    ax.grid(alpha=.25); ax.set_xlim(0, 40); ax.set_ylim(0, 40)

    # 5. conditional bias by predicted amount, before/after
    ax = fig.add_axes([0.375, 0.09, 0.27, 0.33])
    PB = [0.5, 2, 5, 10, 15, 20, 30, 1e9]
    lbl = ["0.5-2", "2-5", "5-10", "10-15", "15-20", "20-30", ">30"]
    for arr, c, lb in ((R, "#c62828", "raw IMERG"),
                       (C, "#1f7a4d", "quantile-mapped")):
        idx = np.digitize(arr, PB) - 1
        xs, ys = [], []
        for b in range(len(lbl)):
            k = idx == b
            if k.sum() < 10:
                continue
            xs.append(b); ys.append(O[k].mean() / arr[k].mean())
        ax.plot(xs, ys, "o-", color=c, lw=2, ms=5, label=lb)
    ax.axhline(1, color="#333", ls="--", lw=1.2)
    ax.set_xticks(range(len(lbl)))
    ax.set_xticklabels(lbl, fontsize=7, rotation=35)
    ax.set_xlabel("predicted amount bin (mm/day)", fontsize=9)
    ax.set_ylabel("gauge / predicted", fontsize=9)
    ax.set_title("E · conditional bias by predicted amount", fontsize=10,
                 fontweight="bold")
    ax.legend(fontsize=7.5, frameon=False); ax.grid(alpha=.25)

    # 6. variance / correlation summary
    ax = fig.add_axes([0.70, 0.09, 0.27, 0.33]); ax.set_axis_off()
    lines = [
        ("out-of-sample, leave-one-year-out", ""),
        ("", ""),
        ("mean  gauge", f"{O.mean():.2f} mm/day"),
        ("      raw IMERG", f"{R.mean():.2f}"),
        ("      corrected", f"{C.mean():.2f}"),
        ("", ""),
        ("sd    gauge", f"{O.std():.2f}"),
        ("      raw IMERG", f"{R.std():.2f}"),
        ("      corrected", f"{C.std():.2f}"),
        ("", ""),
        ("variance ratio  raw", f"{R.std()/O.std():.2f}"),
        ("                corrected", f"{C.std()/O.std():.2f}"),
        ("", ""),
        ("correlation     raw", f"{np.corrcoef(R,O)[0,1]:.3f}"),
        ("                corrected", f"{np.corrcoef(C,O)[0,1]:.3f}"),
        ("", ""),
        ("RMSE            raw", f"{np.sqrt(((R-O)**2).mean()):.2f}"),
        ("                corrected", f"{np.sqrt(((C-O)**2).mean()):.2f}"),
    ]
    for i, (a_, b_) in enumerate(lines):
        ax.text(0.0, 0.97 - i * 0.055, a_, fontsize=8.6, va="top",
                fontweight="bold" if i == 0 else "normal", family="monospace")
        ax.text(0.72, 0.97 - i * 0.055, b_, fontsize=8.6, va="top",
                family="monospace")
    ax.text(0.0, 0.97 - len(lines) * 0.055 - 0.02,
            "quantile mapping targets the DISTRIBUTION,\n"
            "so rank correlation is unchanged by design;\n"
            "the gain is in frequency and amplitude.",
            fontsize=7.6, va="top", color="#444", style="italic")

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=115, facecolor="white")
    print(f"\nwrote {OUT_PNG}")


if __name__ == "__main__":
    raise SystemExit(main())
