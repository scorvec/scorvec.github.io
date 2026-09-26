#!/usr/bin/env python3
"""Weather-regime probabilities from every GEFS member, days 1-35, Euro-Atlantic and North American sectors - the GEPS
page's regimes_geps.py ported to the GEFS extended page (2026-09-26).

Each member's de-drifted, re-based daily 500 hPa anomaly (gefs_patterns.member_anoms, 2.5 deg) is smoothed with the
5-day running mean the regimes were built on and assigned to the nearest centroid of the season's set (the GEPS page's
build_regimes.py: k-means, k = 4, NCEP/NCAR R1 1991-2020; the centroids are model-independent and ship in
gefs_patterns_ref.npz), or to "no regime" when its pattern correlation with that centroid is below 0.25. The fraction
of members in each class by day is the regime probability; weekly means of it are the table. The set is the one for
the INIT month's season, held for the whole forecast.

Observed tail: GEFS's own analyses (12Z + next 00Z pair per day, gefs_patterns.observed_anoms) on the same projection.
Previous run: the most recent earlier run in the archive (normally yesterday's), boundaries dashed on common days.
Hindcast skill (gefs_patterns_calib.npz, build_gefs_patterns_hindcast.py): the ensemble-mean regime of every
GEFSv12 reforecast start (2000-2019, Wednesdays, 4 members) against the NCEP R1 regime on the valid date, for the starts
within +-45 d of this date, by lead; against persistence of the regime observed the day before day 1 and against always
forecasting the most common regime, with the 95% binomial range of a hit rate no better than that climatology.

    python gefs_regimes.py --npz /tmp/gefs_work/gefs2_20260925.npz --date 20260925 --site . --cache /tmp/gefs_work
    -> assets/gefs/gefs_regimes_{ea,na}_fc.webp, assets/gefs/data/gefs_regimes.json; run archive gefs_regimes_<date>.npz
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gefs_patterns as GP                                           # noqa: E402

K = GP.K
COLORS = ["#b4453c", "#2c5fa8", "#3f9b6a", "#c8781e"]          # regimes 1-4 (GEPS)
NONE_COLOR = "#cfcbc4"
WINDOW_DOY = 45


def member_regimes(ref, z500, sk, season):
    """z500 (35, M, 73, 144) -> (probs (K+1, 35), labels (35, M))."""
    sec, _, _ = GP.sector_cut(z500, GP.SECTORS[sk])
    v = GP.running_partial(sec, GP.SMOOTH, axis=0)
    L, N = v.shape[:2]
    lab, _ = ref.classify(v.reshape(L * N, *v.shape[2:]), sk, season)
    lab = lab.reshape(L, N)
    return np.stack([(lab == k).mean(1) for k in range(K + 1)]), lab


def analysis_regimes(ref, fields, ts, sk, season):
    if not ts or "zg500" not in fields:
        return None
    sec, _, _ = GP.sector_cut(fields["zg500"], GP.SECTORS[sk])
    v = GP.running_partial(sec, GP.SMOOTH, axis=0)
    lab, corr = ref.classify(v, sk, season)
    return list(ts), lab, corr


def weekly(probs):
    """Weekly means of the daily probabilities (K+1, 35); days a previous run does not cover are NaN and skipped."""
    rows = []
    for w0, w1 in GP.WEEKS:
        v = probs[:, w0 - 1:w1]
        ok = np.isfinite(v).all(0)
        if not ok.any():
            continue
        rows.append(dict(w0=w0, w1=w1, ndays=int(ok.sum()),
                         p=[round(float(v[k, ok].mean()), 3) for k in range(K + 1)]))
    return rows


def hindcast_skill(ref, sk, base: dt.date):
    """regimes_geps.hindcast_skill on the calibration file's labels, GEFS valid dates (day d = start + d - 1)."""
    c = ref.calib
    if c is None or f"reg_fc_{sk}" not in c.files:
        return None
    S = [GP.parse_date(str(x)) for x in c["reg_S"]]
    d = np.array([((GP.doy(s) - GP.doy(base) + 183) % 366) - 183 for s in S])
    win = np.abs(d) <= WINDOW_DOY
    fc, ob, ob0 = c[f"reg_fc_{sk}"][win], c[f"reg_ob_{sk}"][win], c[f"reg_ob0_{sk}"][win]
    Sw = [s for s, k in zip(S, win) if k]
    ndays = fc.shape[1]
    same = np.array([[GP.season_of((s0 + dt.timedelta(days=j)).month) == GP.season_of(s0.month) for j in range(ndays)]
                     for s0 in Sw])
    ok = (ob >= 0) & (fc >= 0) & same
    hit = np.array([np.mean((fc[:, i] == ob[:, i])[ok[:, i]]) if ok[:, i].any() else np.nan for i in range(ndays)])
    pers = np.array([np.mean((ob0 == ob[:, i])[ok[:, i] & (ob0 >= 0)]) if ok[:, i].any() else np.nan for i in range(ndays)])
    nlead = ok.sum(0)
    freq = np.bincount(ob[ok], minlength=K + 1) / max(ok.sum(), 1)
    fw = np.bincount(ob0[ob0 >= 0], minlength=K + 1) / max((ob0 >= 0).sum(), 1)
    return dict(day=np.arange(1, ndays + 1), hit=hit, persistence=pers, clim=float(freq.max()), n=int(win.sum()),
                n_lead=nlead, freq_window=fw.tolist())


def plot_sector(ref, sk, season, date, prev_date, base, probs, tail, prev, skill, n_members, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    spec = GP.SECTORS[sk]
    cent, lat, lon = ref.centroids(sk, season)
    names = [ref.regime_name(sk, season, k) for k in range(K)]
    freq = ref.regmeta[f"{sk}_{season}"]["freq"]
    fw = skill.get("freq_window") if skill else None
    fig = plt.figure(figsize=(15.0, 9.6))
    proj = ccrs.LambertConformal(central_longitude=np.mean(spec["lon"]), standard_parallels=(35, 55))
    lim = float(np.nanpercentile(np.abs(cent), 99))
    for k in range(K):
        ax = fig.add_axes([0.03 + k * 0.24, 0.69, 0.22, 0.21], projection=proj)
        ax.set_extent([spec["lon"][0], spec["lon"][1], spec["lat"][1], spec["lat"][0]], ccrs.PlateCarree())
        ax.contourf(lon, lat, cent[k], levels=np.linspace(-lim, lim, 17), cmap="RdBu_r", extend="both",
                    transform=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE.with_scale("110m"), lw=0.5, edgecolor="#333")
        for sp in ax.spines.values():
            sp.set_edgecolor(COLORS[k]); sp.set_linewidth(2.2)
        ax.set_title(f"{k + 1}. {names[k]}\n" + (f"{fw[k] * 100:.0f}% of days at this time of year" if fw else
                                                 f"{freq[k] * 100:.0f}% of days, {season} season"),
                     fontsize=9.2, fontweight="bold", color=COLORS[k])
    ax = fig.add_axes([0.06, 0.30, 0.90, 0.33])
    t = GP.valid_days(base)
    half = dt.timedelta(hours=12)
    order = list(range(K)) + [K]
    cols = COLORS + [NONE_COLOR]
    labels = names + ["no regime"]
    ax.stackplot(t, *[probs[k] * 100 for k in order], colors=cols, labels=labels, alpha=0.9, lw=0)
    if prev is not None:
        pt = GP.valid_days(GP.parse_date(prev_date))
        keep = [i for i, x in enumerate(pt) if t[0] <= x <= t[-1]]
        cum = np.cumsum(np.stack([prev[k] for k in order]), axis=0) * 100
        for j in range(K):
            ax.plot([pt[i] for i in keep], cum[j][keep], ls=(0, (3, 2)), color="#111", lw=0.9, alpha=0.7,
                    label="previous run (boundaries)" if j == 0 else None)
    x0 = dt.datetime.combine(tail[0][0] if tail is not None else t[0], dt.time()) - half
    x1 = dt.datetime.combine(t[-1], dt.time()) + half
    t_init = dt.datetime.combine(base, dt.time()) - half
    ax.set_xlim(x0, x1)
    ax.axvline(t_init, color="#111", lw=1.0)
    ax.text(t_init - dt.timedelta(hours=14), 96, "init", ha="right", va="top", fontsize=8, color="#111")
    ax.set_ylim(0, 100); ax.set_ylabel(f"% of {n_members} members", fontsize=9.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d")); ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.tick_params(labelsize=8.5); ax.grid(alpha=0.25, lw=0.5, axis="y")
    for w0, w1 in GP.WEEKS[1:]:
        ax.axvline(dt.datetime.combine(base + dt.timedelta(days=w0 - 1), dt.time()) - half, color="white", lw=1.2)
    other = [d for d in t if GP.season_of(d.month) != season]
    if other:
        ax.axvspan(dt.datetime.combine(other[0], dt.time()) - half, x1, facecolor="white", alpha=0.45, hatch="////",
                   edgecolor="#8a8680", lw=0, zorder=3)
        ax.text(x1 - dt.timedelta(hours=5), 4, f"{'cold' if season == 'warm' else 'warm'}-season set\n"
                f"applies from {other[0]:%b %d}", fontsize=7.8, color=GP.INK, va="bottom", ha="right", zorder=4,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.0))
    ax.legend(fontsize=7.8, ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.10), frameon=False)
    ax.set_title(f"Regime probability by forecast day · GEFS init {base:%Y-%m-%d} · {season} regime set",
                 fontsize=10.5, fontweight="bold", loc="left", pad=26)
    ax2 = fig.add_axes([0.06, 0.20, 0.90, 0.055])
    if tail is not None:
        tt, lab, _ = tail
        for d, l_ in zip(tt, lab):
            ax2.add_patch(plt.Rectangle((mdates.date2num(dt.datetime.combine(d, dt.time())) - 0.5, 0), 1, 1,
                                        color=cols[l_], lw=0))
        ax2.text(mdates.date2num(dt.datetime.combine(tt[0], dt.time())), 1.08,
                 "observed regime, GEFS analyses (12Z + 00Z, 5-day mean)", fontsize=8, va="bottom", color=GP.INK)
    ax2.set_xlim(mdates.date2num(x0), mdates.date2num(x1))
    ax2.axvline(mdates.date2num(t_init), color="#111", lw=1.0)
    ax2.set_ylim(0, 1); ax2.set_yticks([])
    ax2.xaxis_date(); ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=5)); ax2.tick_params(labelsize=8)
    ax3 = fig.add_axes([0.06, 0.07, 0.36, 0.09])
    if skill is not None:
        d_ = skill["day"][:35]
        nl = np.maximum(skill["n_lead"][:35], 1)
        c = skill["clim"]
        band = 1.96 * np.sqrt(c * (1 - c) / nl)
        ax3.fill_between(d_, (c - band) * 100, (c + band) * 100, color="#b4453c", alpha=0.12, lw=0,
                         label="95% range of a hit rate no better than climatology")
        ax3.plot(d_, skill["hit"][:35] * 100, color="#16335c", lw=2.0, label="ensemble-mean regime correct")
        ax3.plot(d_, skill["persistence"][:35] * 100, color="#8a8680", lw=1.4, ls="--",
                 label="persistence of the regime before day 1")
        ax3.axhline(c * 100, color="#b4453c", lw=1.0, ls=":", label="most common regime (climatology)")
        ax3.set_xlim(1, 35); ax3.set_ylim(0, 100); ax3.set_xlabel("forecast day", fontsize=8.5)
        ax3.set_ylabel("% hit", fontsize=8.5)
        ax3.legend(fontsize=6.6, loc="upper right", ncol=1, framealpha=0.9)
        ax3.set_title(f"Hindcast 2000–2019 (GEFSv12 reforecast, 4-member mean), starts within ±45 d of this date "
                      f"(n={skill['n']})", fontsize=8.6, loc="left")
        ax3.tick_params(labelsize=8); ax3.grid(alpha=0.25, lw=0.5)
    else:
        ax3.text(0.0, 0.5, "hindcast regime skill: not yet scored (build_gefs_patterns_hindcast.py)", fontsize=8.5,
                 color=GP.MUTED)
        ax3.axis("off")

    def flag(r):
        days = [base + dt.timedelta(days=x - 1) for x in range(r["w0"], r["w1"] + 1)]
        return " *" if any(GP.season_of(d.month) != season for d in days) else ""
    wrows = weekly(probs)
    fig.text(0.47, 0.06, "Weekly regime probabilities (% of members):\n" + "\n".join(
        f"week {i + 1}  " + "  ".join(f"{labels[k][:13]:13s}{r['p'][k] * 100:3.0f}%" for k in order) + flag(r)
        for i, r in enumerate(wrows)) + ("\n* week crosses the season boundary: other regime set applies"
                                         if any(flag(r) for r in wrows) else ""),
        fontsize=7.6, family="monospace", va="bottom", color=GP.INK)
    fig.suptitle(f"{spec['label']} weather regimes — GEFS extended ensemble, {n_members} members · init {base:%Y-%m-%d} 00Z"
                 + (f" · dashed: previous run {GP.parse_date(prev_date):%b %d}" if prev_date else "") + GP.TEST_NOTE,
                 fontsize=13, fontweight="bold", y=0.985, va="top")
    fig.text(0.5, 0.004, f"k-means regimes on 5-day-mean 500 hPa anomalies (NCEP/NCAR R1 1991–2020); a member-day with "
             f"pattern correlation below {GP.R_MIN} to every centroid is 'no regime'; the {season} set is used for the whole "
             "forecast\nmembers de-drifted against the lead-matched GEFSv12 reforecast climatology (2000–2019) and re-based to "
             "1991–2020 with the ERA5 2001–2020 → 1991–2020 shift", ha="center", va="bottom", fontsize=7.6, color=GP.MUTED)
    GP.save_webp(fig, out / f"gefs_regimes_{sk}_fc.webp")
    plt.close(fig)
    print(f"  gefs_regimes_{sk}_fc.webp", flush=True)


def main() -> int:
    ap = GP.common_args(argparse.ArgumentParser())
    a = ap.parse_args()
    base, out, data, runs, cache, ref, clim, z = GP.setup(a)
    if clim is None:
        print("no daily climatology (gefs-clim-v2): regimes skipped"); return 0
    season = GP.season_of(base.month)
    n_members = int(len(z["members"]))
    z500 = GP.member_anoms(z, base, ref, {"zg500": clim["zg500"]})["zg500"]
    prev_date, pz = GP.previous_run(runs, "regimes", a.date)
    obs_fields, ts = GP.observed_anoms(base, ref, runs)
    GP.prune_tail(runs, base)
    js = {"cycle": a.date, "prev_cycle": prev_date, "season": season, "r_min": GP.R_MIN,
          "generated": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%MZ"), "n_members": n_members,
          "valid_convention": "day d valid on init + (d - 1)", "test": bool(GP.TEST_NOTE), "sectors": {}}
    keep = {}
    for sk, spec in GP.SECTORS.items():
        probs, lab = member_regimes(ref, z500, sk, season)
        for s in GP.SEASONS:                                         # both sets, so tomorrow's overlay always matches
            keep[f"{sk}_{s}"] = (probs if s == season else member_regimes(ref, z500, sk, s)[0]).astype("float32")
        prev = pz[f"{sk}_{season}"].astype("float64") if pz is not None and f"{sk}_{season}" in pz.files else None
        tail = analysis_regimes(ref, obs_fields, ts, sk, season)
        skill = hindcast_skill(ref, sk, base)
        plot_sector(ref, sk, season, a.date, prev_date if prev is not None else None, base, probs, tail, prev, skill,
                    n_members, out)
        names = [ref.regime_name(sk, season, k) for k in range(K)] + ["no regime"]
        wk = weekly(probs)
        pwk = None
        if prev is not None:
            gap = (base - GP.parse_date(prev_date)).days
            pwk = weekly(np.concatenate([prev[:, gap:], np.full((K + 1, gap), np.nan)], axis=1))
        rm = ref.regmeta[f"{sk}_{season}"]
        rec = {"label": spec["label"], "names": names, "climatology": rm["freq"] + [rm["no_regime"]],
               "daily": {"t": [d.isoformat() for d in GP.valid_days(base)], "p": np.round(probs, 3).tolist()},
               "weeks": wk, "prev_weeks": pwk}
        if tail is not None:
            rec["observed"] = {"t": [x.isoformat() for x in tail[0]], "regime": [int(x) for x in tail[1]]}
        other = [d for d in GP.valid_days(base) if GP.season_of(d.month) != season]
        rec["season_boundary"] = other[0].isoformat() if other else None
        if skill is not None:
            rec["freq_window"] = skill["freq_window"]
            rec["skill"] = {"day": skill["day"][:35].tolist(), "hit": np.round(skill["hit"][:35], 3).tolist(),
                            "persistence": np.round(skill["persistence"][:35], 3).tolist(),
                            "clim": round(skill["clim"], 3), "n_starts": skill["n"]}
        js["sectors"][sk] = rec
        print(f"  {sk} {season}: " + "  ".join(f"wk{r['w0'] // 7 + 1} " + "/".join(f"{p * 100:.0f}" for p in r["p"])
                                               for r in wk), flush=True)
    (data / "gefs_regimes.json").write_text(json.dumps(js, allow_nan=False))
    np.savez_compressed(runs / f"gefs_regimes_{a.date}.npz", **keep)
    GP.prune_runs(runs, "regimes", base)
    return 0


if __name__ == "__main__":
    sys.exit(main())
