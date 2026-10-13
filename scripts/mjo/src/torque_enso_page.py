#!/usr/bin/env python3
"""Write the static study page scorvec.com/torque-enso.html ("El Nino and mountain torque").

Every number is read from the committed study tables (scripts/mjo/data/reference/torque_enso_study/, written by
torque_enso_study.py and torque_budget_2026.py) or from the committed ERA5 history; none is typed by hand.
Figures: studies/torque-enso/*.webp from torque_enso_figs.py --site . ; then scripts/site/apply_chrome.py stamps
the header, menus and footer.

    python scripts/mjo/src/torque_enso_figs.py --site .
    python scripts/mjo/src/torque_enso_page.py --site .
    python scripts/site/apply_chrome.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import torque_enso_study as S                                           # noqa: E402

D = S.REF / "torque_enso_study"
IMG = "/studies/torque-enso"
URL = "https://scorvec.com/torque-enso.html"
GH = "https://github.com/scorvec/scorvec.github.io/blob/main/"
TITLE = "El Niño and angular momentum"
H1 = "Why the atmosphere spins up in El Niño"
DESC = ("Where El Niño's extra atmospheric angular momentum comes from: a standing shift in the Andes mountain torque, "
        "and why ERA5's budget only closes with its analysis increments. ERA5 1959–2026, 41 CMIP6 models, an hourly 2026 budget.")


def num(x, d=1, sign=True):
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace("-", "−")


def dd(lag, before=False):
    n = abs(int(lag))
    return f"{n} day{'' if n == 1 else 's'}" + (" before" if before else "")


LAYER = {"stratosphere (<100 hPa)": "stratosphere (above 100 hPa)", "upper troposphere (100-400 hPa)": "upper troposphere (100–400 hPa)",
         "mid troposphere (400-850 hPa)": "middle troposphere (400–850 hPa)", "lowest layers (>=850 hPa)": "below 850 hPa"}


def ci(lo, hi, d=1):
    return f"[{num(lo, d)}, {num(hi, d)}]"


def pval(p):
    if p < 0.001:
        return "p &lt; 0.001"
    return f"p = {p:.3f}" if p < 0.01 else f"p = {p:.2f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default=".")
    a = ap.parse_args()
    site = Path(a.site)
    R = json.loads((D / "results.json").read_text())
    B = json.loads((D / "budget2026.json").read_text())
    comp = pd.read_csv(D / "q1_compare.csv")
    cm = pd.read_csv(D / "q1_cmip6_summary.csv")
    sh = json.loads((D / "q1_shift_test.json").read_text())
    dec = pd.read_csv(D / "q1_decomposition.csv")
    cc = pd.read_csv(D / "q2_cmip6_contrib_summary.csv").set_index("series")
    ccomp = pd.read_csv(D / "q2_cmip6_composites_summary.csv").set_index("term")
    pt = pd.read_csv(D / "q3_precursor_tests.csv")
    ll = pd.read_csv(D / "q3_leadlag.csv")

    def c1(r, se, mt, cmp_="EN-NEU"):
        q = comp[(comp.range == r) & (comp.season == se) & (comp.metric == mt) & (comp.cmp == cmp_)]
        return q.iloc[0]

    def cmr(r, se, mt):
        q = cm[(cm.range == r) & (cm.season == se) & (cm.metric == mt)]
        return q.iloc[0]
    nsig = {mt: int(comp[(comp.cmp == "EN-NEU") & (comp.metric == mt)].sig.sum())
            for mt in ("n_pos", "n_neg", "amp_pos", "amp_neg", "hp_sd", "evc_pos", "lp_mean")}
    ncm_robust_events = int(cm[cm.metric.isin(["n_pos", "n_neg", "amp_pos", "hp_sd"]) & cm.robust].shape[0])
    nsy = R["q1"]["n_season_years"]
    q2 = R["q2_era5"]
    ce = q2["contributions_lags_m6_m1"]
    tend = q2["tendency_column"]
    n_models = R["q2_cmip6"]["n_models"]; n_members = R["q2_cmip6"]["n_members"]; n_gwd = R["q2_cmip6"]["n_models_with_gwd"]
    n_daily = len(R["q1_cmip6"]["models"])

    # ERA5 history: May Andes ranks and the per-hour absolute means (the 12Z artefact)
    df = S.load_era5()
    a_and, _ = S.harmonic_anom(df["mtn5_AND"])
    mm = a_and.resample("MS").mean()
    may = mm[mm.index.month == 5].loc["1959":"2026"]
    may26 = float(may.loc["2026-05-01"])
    may_rank = int((may > may26).sum() + 1)
    may_top = may.drop(pd.Timestamp("2026-05-01")).sort_values(ascending=False)
    may_first = (may_top.index[0].year, float(may_top.iloc[0]))
    base = df.loc["1991":"2020"]
    hours = {h: {t: float(base[f"{t}_glob_h{h}"].mean()) for t in ("mtn", "fric", "gwd")} for h in ("00", "06", "12", "18")}
    four = {t: float(base[f"{t}_glob"].mean()) for t in ("mtn", "fric", "gwd")}
    hsum = {h: sum(v.values()) for h, v in hours.items()}
    fsum = sum(four.values())

    # ---------------- result-1 numbers
    lpA = c1("AND", "DJF", "lp_mean"); lpH = c1("HIM", "MAM", "lp_mean"); lpR = c1("ROC", "DJF", "lp_mean")
    npA = c1("AND", "DJF", "n_pos"); npR = c1("ROC", "DJF", "n_pos"); npH = c1("HIM", "DJF", "n_pos")
    amA = c1("AND", "DJF", "amp_pos")
    slA = c1("AND", "DJF", "lp_mean", "ONI-slope"); slAJ = c1("AND", "JJA", "lp_mean", "ONI-slope")
    slR = c1("ROC", "DJF", "lp_mean", "ONI-slope"); slRM = c1("ROC", "MAM", "lp_mean", "ONI-slope")
    slH = c1("HIM", "MAM", "lp_mean", "ONI-slope"); slHS = c1("HIM", "SON", "lp_mean", "ONI-slope")
    nslope = comp[(comp.cmp == "ONI-slope") & (comp.metric == "n_pos") & comp.sig]
    nxA = c1("AND", "DJF", "nx_pos"); nxnA = c1("AND", "DJF", "nx_neg")
    sA = sh["AND"]["DJF"]; sR = sh["ROC"]["DJF"]; sH = sh["HIM"]["MAM"]; sG = sh["GRL"]["SON"]
    cA = cmr("AND", "DJF", "lp_mean"); cRM = cmr("ROC", "MAM", "lp_mean"); cRJ = cmr("ROC", "JJA", "lp_mean")
    cHM = cmr("HIM", "MAM", "lp_mean"); cHS = cmr("HIM", "SON", "lp_mean"); cRD = cmr("ROC", "DJF", "lp_mean")
    cnA = cmr("AND", "DJF", "nx_pos"); cnR = cmr("ROC", "DJF", "nx_pos"); cevH = cmr("HIM", "DJF", "n_pos")
    dA = dec[(dec.range == "AND") & (dec.season == "DJF")].iloc[0]
    dR = dec[(dec.range == "ROC") & (dec.season == "DJF")].iloc[0]
    dH = dec[(dec.range == "HIM") & (dec.season == "MAM")].iloc[0]

    def dec_row(lab, d, c):
        return (f"<tr><th scope='row'>{lab}</th><td>{num(d.d_total, 2)}</td><td>{num(d.d_lowpass, 2)}</td>"
                f"<td>{num(d.d_highpass, 2)}</td><td class='ns'>{num(d.d_event_pos, 2)} n.s.</td>"
                f"<td>{num(c['median'], 2)} <span class='sig'>({c.agree:.0%})</span></td></tr>")
    t_dec = ("<div class='tw'><table class='res'><thead><tr><th>El Niño minus neutral (Hadley)</th><th>seasonal mean</th>"
             "<th>standing part (≥30 d)</th><th>synoptic part (&lt;30 d)</th><th>event windows</th>"
             f"<th>CMIP6 standing part</th></tr></thead><tbody>"
             + dec_row("Andes, DJF", dA, cA) + dec_row("Rockies, DJF", dR, cRD) + dec_row("Himalaya/Tibet, MAM", dH, cHM)
             + "</tbody></table></div>")

    # ---------------- result-2 table (ERA5 vs CMIP6 contributions)
    def crow(lab, ke, kc):
        e = ce.get(ke)
        ecell = (f"<td>{num(e['est'], 2)} <span class='sig'>{ci(e['lo'], e['hi'], 2)}</span></td>" if e and e["fdr"]
                 else (f"<td class='ns'>{num(e['est'], 2)} n.s.</td>" if e else "<td></td>"))
        if kc and kc in cc.index:
            c = cc.loc[kc]
            ccell = (f"<td>{num(c['median'], 2)} <span class='sig'>({c.agree:.0%} agree)</span></td>" if c.robust
                     else f"<td class='ns'>{num(c['median'], 2)} not robust</td>")
        else:
            ccell = "<td></td>"
        return f"<tr><th scope='row'>{lab}</th>{ecell}{ccell}</tr>"
    rows2 = [("AAM tendency dM/dt", tend, "dmdt_glob"), ("sum of the torques", "T_glob", "T_mf_glob"),
             ("mountain torque", "mtn_glob", "mtn_glob"), ("friction (ERA5 turbulent stress; CMIP6 tauu)", "fric_glob", "fric_glob"),
             ("gravity-wave drag (CMIP6: 13 models)", "gwd_glob", "gwd_glob"), ("ERA5: dM/dt minus torques", "R_glob", None),
             ("Andes", "mtn5_AND", "mtn_AND"), ("Rockies", "mtn5_ROC", "mtn_ROC"), ("Himalaya/Tibet", "mtn5_HIM", "mtn_HIM"),
             ("Greenland", "mtn5_GRL", "mtn_GRL"), ("Antarctica", "mtn5_ANT", "mtn_ANT"), ("Alps", "mtn5_ALP", "mtn_ALP"),
             ("mountain, 30S–0", "mtn_b3000s", "mtn_b3000s"), ("mountain, 0–30N", "mtn_b0030n", "mtn_b0030n"),
             ("mountain, 30–60N", "mtn_b3060n", "mtn_b3060n"), ("friction, 30S–0", "fric_b3000s", "fric_b3000s"),
             ("friction, 0–30N", "fric_b0030n", "fric_b0030n"), ("friction, 30–60N", "fric_b3060n", "fric_b3060n"),
             ("friction, 60–30S", "fric_b6030s", "fric_b6030s")]
    t_q2 = ("<div class='tw'><table class='res'><thead><tr><th>10²⁵ kg m² s⁻¹ per K of Niño-3.4</th><th>ERA5 1979–2025 [95% CI]</th>"
            f"<th>CMIP6, {n_models} models: median</th></tr></thead><tbody>"
            + "".join(crow(*x) for x in rows2) + "</tbody></table></div>")
    mt, fr, gw = ce["mtn_glob"], ce["fric_glob"], ce["gwd_glob"]
    cmt, cfr, cgw, cdm, cT = cc.loc["mtn_glob"], cc.loc["fric_glob"], cc.loc["gwd_glob"], cc.loc["dmdt_glob"], cc.loc["T_mf_glob"]
    cAND, cROC, cHIM = cc.loc["mtn_AND"], cc.loc["mtn_ROC"], cc.loc["mtn_HIM"]
    mc = q2["monthly_closure"]
    comp_en = q2["composite_EN_minus_NEU"]
    ccd, ccm, ccf, cca = ccomp.loc["dM"], ccomp.loc["mtn_glob"], ccomp.loc["fric_glob"], ccomp.loc["mtn_AND"]

    # ---------------- result-3 numbers
    def llv(r, se, k, lag):
        q = ll[(ll.range == r) & (ll.season == se) & (ll.series == k) & (ll.lag == lag)].iloc[0]
        return q
    def llpeak(r, se, k, lags):
        q = ll[(ll.range == r) & (ll.season == se) & (ll.series == k) & ll.lag.isin(lags)]
        return q.loc[q.r.abs().idxmax()]
    pre_lo = [llpeak(r, se, "M", range(-7, 0)) for r, se in (("HIM", "DJF"), ("ROC", "DJF"), ("AND", "DJF"), ("AND", "JJA"))]
    post_hi = [llpeak(r, se, "M", range(1, 6)) for r, se in (("HIM", "DJF"), ("ROC", "DJF"), ("AND", "DJF"), ("AND", "JJA"))]
    jR = llpeak("ROC", "DJF", "u200jet", range(0, 8)); jA = llpeak("AND", "DJF", "u200jet", range(0, 8))
    jH = llpeak("HIM", "DJF", "u200jet", range(-15, 16)); jHM = llpeak("HIM", "MAM", "u200jet", range(0, 8))
    jbA = llpeak("AND", "DJF", "u200jet", range(-5, 0))
    qc = R["q3_composite_counts"]
    pt_any = bool(pt.sig.any())
    ptH = pt[(pt.range == "HIM") & (pt.season == "MAM") & (pt.metric == "n_prec")].iloc[0]
    ptR = pt[(pt.range == "ROC") & (pt.season == "DJF") & (pt.metric == "hit_rate")].iloc[0]

    # ---------------- 2026
    wt = B["window_test"]; inc = B["increment_per_window_1e25"]; dfh = B["daily_from_hourly"]
    g = {int(x["time"]): x for x in B["hourly_residual_by_end_hour"]}
    t3 = B["three_hourly_window_test"]
    wv = B["window_variants"]
    v12 = wv["12Z daily, motion only, anomaly"]; v12m = wv["12Z daily, motion+mass, anomaly"]
    v4 = wv["4x daily, motion+mass, anomaly"]; vh = wv["hourly daily-means, motion+mass, anomaly vs 4x clim"]
    va1 = wv["1-hourly, motion+mass, absolute"]; va3 = wv["3-hourly, motion+mass, absolute"]
    hw = B["history_window_residual"]; gfz = B["gfz"]; att = B["attribution"]
    cl = B["closure_by_step"]
    zr = (vh["residual"] - hw["mean"]) / hw["sd"]
    oni = S.load_oni()
    oni26 = {m: float(oni.get(pd.Timestamp(f"2026-{m:02d}-01"), np.nan)) for m in (4, 5, 6, 7, 8)}

    def vrow(lab, v):
        return (f"<tr><th scope='row'>{lab}</th><td>{num(v['torque'], 2)}</td><td>{num(v['dM'], 2)}</td>"
                f"<td>{num(v['residual'], 2)}</td><td>{num(v['residual_mean_hadley'], 1)}</td></tr>")
    t_var = ("<div class='tw'><table class='res'><thead><tr><th>15 Apr – 19 Sep 2026</th><th>∫ torques</th><th>ΔM</th>"
             "<th>residual</th><th>residual, mean Hadley</th></tr></thead><tbody>"
             + vrow("12Z once daily, motion term only, anomaly", v12) + vrow("12Z, motion + mass, anomaly", v12m)
             + vrow("00/06/12/18Z, motion + mass, anomaly", v4) + vrow("hourly daily means, motion + mass, anomaly", vh)
             + vrow("3-hourly, motion + mass, absolute", va3) + vrow("hourly, motion + mass, absolute", va1)
             + "</tbody></table></div>")

    def srow(st, lab):
        c = cl[str(st)]
        return (f"<tr><th scope='row'>{lab}</th><td>{c['corr_dMdt_T']:.2f}</td><td>{num(c['mean_R'], 1)}</td>"
                f"<td>{c['sd_R']:.1f}</td><td>{c['sd_Rmotion_only']:.1f}</td></tr>")
    t_step = ("<div class='tw'><table class='res'><thead><tr><th>ERA5 2026, sampled every</th><th>corr(dM/dt, torques)</th>"
              "<th>mean residual</th><th>residual sd</th><th>sd, motion term only</th></tr></thead><tbody>"
              + srow(1, "1 hour") + srow(3, "3 hours") + srow(6, "6 hours") + srow(12, "12 hours") + srow(24, "24 hours")
              + "</tbody></table></div>")
    t_hours = ("<div class='tw'><table class='res'><thead><tr><th>ERA5 1991–2020 mean (Hadley)</th><th>00Z</th><th>06Z</th>"
               "<th>12Z</th><th>18Z</th><th>mean of the four</th></tr></thead><tbody>"
               + "".join(f"<tr><th scope='row'>{lab}</th>" + "".join(f"<td>{num(hours[h][t], 1)}</td>" for h in hours)
                         + f"<td>{num(four[t], 1)}</td></tr>"
                         for t, lab in (("mtn", "mountain"), ("fric", "friction"), ("gwd", "gravity-wave drag")))
               + "<tr class='diff'><th scope='row'>sum (should be ≈ 0)</th>" + "".join(f"<td>{num(hsum[h], 1)}</td>" for h in hours)
               + f"<td>{num(fsum, 1)}</td></tr></tbody></table></div>")
    labs = {"Andes": "Andes", "other ranges": "the other five ranges", "rest of mountain": "mountain outside the six boxes",
            "friction tropics": "friction, 30S–30N", "friction NH ext": "friction, north of 30N",
            "friction SH ext": "friction, south of 30S", "GWD": "gravity-wave drag"}
    t_att = ("<div class='tw'><table class='res'><thead><tr><th>15 Apr – 19 Sep, 10²⁵ kg m² s⁻¹</th><th>2026</th>"
             "<th>trend removed</th><th>1979–2025 sd</th><th>z</th><th>rank of 48</th><th>El Niño onsets, mean</th></tr></thead><tbody>"
             + "".join(f"<tr><th scope='row'>{labs[k]}</th><td>{num(v['v2026'], 1)}</td><td>{num(v['v2026_detrended'], 1)}</td>"
                       f"<td>{v['hist_sd']:.1f}</td><td>{num(v['z'], 1)}</td><td>{v['rank_of_48']}</td>"
                       f"<td>{num(v['en_onset_mean'], 1)}</td></tr>" for k, v in att.items())
             + f"<tr class='diff'><th scope='row'>sum of the torques</th><td>{num(B['attribution_total_torque'], 1)}</td><td></td><td></td><td></td><td></td><td></td></tr>"
             + f"<tr class='diff'><th scope='row'>AAM anomaly change (motion + mass)</th><td>{num(B['attribution_dM_hourly_anomaly'], 1)}</td><td></td><td></td><td></td><td></td><td></td></tr>"
             + "</tbody></table></div>")

    boxes = ("<div class='tw'><table class='res'><thead><tr><th>range</th><th>longitude</th><th>latitude</th></tr></thead><tbody>"
             "<tr><th scope='row'>Himalaya/Tibet</th><td>70–105°E</td><td>25–45°N</td></tr>"
             "<tr><th scope='row'>Rockies</th><td>130–100°W</td><td>30–60°N</td></tr>"
             "<tr><th scope='row'>Andes</th><td>80–65°W</td><td>55°S–10°N</td></tr>"
             "<tr><th scope='row'>Greenland</th><td>60–15°W</td><td>60–85°N</td></tr>"
             "<tr><th scope='row'>Antarctica</th><td>all</td><td>south of 65°S</td></tr>"
             "<tr><th scope='row'>Alps</th><td>0–45°E</td><td>35–50°N</td></tr></tbody></table></div>")


    # ---------------- figure numbering (in order of appearance)
    FIGS = ["gfz", "q2lag", "q2con", "q1grid", "q1thr", "slp", "q3ll", "q3z", "b26cum", "b26att", "b26hr",
            "incprof", "incseason", "incsection", "mapsu", "mapsseason", "mapssplit", "fcdrift"]
    FN = {k: i + 1 for i, k in enumerate(FIGS)}

    def fig(key, f, alt, cap):
        n = FN[key]
        return (f"<figure id='fig{n}'><a href='{IMG}/{f}' class='zoom'><img src='{IMG}/{f}' alt=\"{alt}\" loading='lazy'></a>"
                f"<figcaption><b>Figure {n}.</b> {cap}</figcaption></figure>")

    def ordn(n):
        return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"

    def rank_word(n):
        return {2: "second", 3: "third", 4: "fourth", 5: "fifth"}.get(n, ordn(n))

    def rank_txt(n):
        top_ = 49 - n
        return f"{'second-largest' if top_ == 2 else ordn(top_) + ' largest'} of 48 years" if top_ <= n else f"{ordn(n)} smallest of 48 years"

    # ---------------- increments (torque_increments.py) and the low-level maps
    I = json.loads((D / "increments.json").read_text())
    P = I["profile"]
    g10, g22 = P["10"]["global"], P["22"]["global"]
    w10, w22, m10, m22, T10 = P["10r"]["global"], P["22r"]["global"], P["10m"]["global"], P["22m"]["global"], P["10T"]["global"]
    bl = I["belts"]
    gm = I["global_by_month"]
    gfull = {k: v for k, v in gm.items() if v["n"] >= 20}
    gmin = min(gfull, key=lambda k: gfull[k]["mean"]); gmax = max(gfull, key=lambda k: gfull[k]["mean"])
    u10, u22 = I["u37_J10"], I["u37_J22"]
    L10, L22 = u10["layers"], u22["layers"]
    fsec = I.get("u37_frac_sig_with_data", {})
    ir = I["inwindow_regression"]; it = ir["terms"]
    fd = I["fc_drift"]["by_region"]
    finits = I["fc_drift"]["inits"]
    MP = json.loads((D / "increments_maps.json").read_text())
    sp_ = MP["split"]; area = MP["area"]

    def share(var, s, r):
        v = sp_[f"{var}_{s}_{r}"]
        def pc(x):
            return f"{round(100 * x) + 0:.0f}".replace("-", "−")
        return f"{pc(v['share'])}% [{pc(v['lo'])}, {pc(v['hi'])}]"
    import xarray as xr
    zm = xr.open_dataset(S.REF / "torque_u37maps_2025_2026.nc")
    zlat = zm.lat.values

    def zmean(var, la0, la1):
        m = zm[f"{var}_both_ALL_mean"].values
        sel = (np.abs(zlat) >= la0) & (np.abs(zlat) <= la1)
        w = np.cos(np.deg2rad(zlat))[sel][:, None] * np.ones(m.shape[1])[None, :]
        x = m[sel]
        ok = np.isfinite(x)
        return float((x * w)[ok].sum() / w[ok].sum())

    def zmean_h(var, la0, la1):
        m = zm[f"{var}_both_ALL_mean"].values
        sel = (zlat >= la0) & (zlat <= la1)
        w = np.cos(np.deg2rad(zlat))[sel][:, None] * np.ones(m.shape[1])[None, :]
        x = m[sel]
        ok = np.isfinite(x)
        return float((x * w)[ok].sum() / w[ok].sum())
    u1000_mid, u1000_trop = zmean("u1000", 40, 60), zmean("u1000", 10, 30)
    u10_mid, u10_trop = zmean("u10", 40, 60), zmean("u10", 10, 30)
    u925_mid, u850_nh = zmean("u925", 40, 60), zmean_h("u850", 40, 55)
    u850_trop, u700_trop = zmean("u850", 10, 30), zmean("u700", 10, 30)

    def lead(reg, L):
        return next(x for x in fd[reg] if x["lead"] == L)
    fg1, fg15, fn10, fn15, fs7, fs10, fs15 = (lead("global", 1), lead("global", 15), lead("nh", 10), lead("nh", 15),
                                             lead("sh", 7), lead("sh", 10), lead("sh", 15))
    nh_first = next((x["lead"] for x in fd["nh"] if x["lead"] > 0 and x["sig"] and all(y["sig"] for y in fd["nh"] if y["lead"] >= x["lead"])), None)
    sh_sig = sum(x["sig"] for x in fd["sh"] if x["lead"] >= 3)
    nver = (lead("global", 15)["n"], lead("global", 1)["n"])
    E = lambda v, d=3: f"{num(v['mean'], d)} {ci(v['lo'], v['hi'], d)}"     # noqa: E731
    per_day = g10["mean"] + g22["mean"]
    col37 = (u10["global_mean"] + u22["global_mean"]) / 2
    col2d = (w10["mean"] + w22["mean"]) / 2
    lay_rows = "".join(f"<tr><th scope='row'>{LAYER.get(nm, nm)}</th><td>{num(L10[nm]['mean'], 3)} <span class='sig'>{ci(L10[nm]['lo'], L10[nm]['hi'], 3)}</span></td>"
                       f"<td>{num(L22[nm]['mean'], 3)} <span class='sig'>{ci(L22[nm]['lo'], L22[nm]['hi'], 3)}</span></td></tr>" for nm in L10)
    t_lay = ("<div class='tw'><table class='res'><thead><tr><th>wind-term increment, 10²⁵ kg m² s⁻¹ per window</th>"
             "<th>09 UTC start</th><th>21 UTC start</th></tr></thead><tbody>" + lay_rows
             + f"<tr class='diff'><th scope='row'>all 37 levels</th><td>{num(u10['global_mean'], 3)}</td><td>{num(u22['global_mean'], 3)}</td></tr>"
             + "</tbody></table></div>")
    bel_rows = "".join(f"<tr><th scope='row'>{nm.replace('-', '–')}</th><td>{num(bl['10_' + nm]['mean'], 3)}</td><td>{num(bl['22_' + nm]['mean'], 3)}</td></tr>"
                       for nm in ("90-60S", "60-30S", "30S-0", "0-30N", "30-60N", "60-90N"))
    t_bel = ("<div class='tw'><table class='res'><thead><tr><th>total increment, 10²⁵ kg m² s⁻¹ per window</th><th>09 UTC start</th>"
             "<th>21 UTC start</th></tr></thead><tbody>" + bel_rows
             + f"<tr class='diff'><th scope='row'>global</th><td>{num(g10['mean'], 3)}</td><td>{num(g22['mean'], 3)}</td></tr>"
             + "</tbody></table></div>")
    reg_terms = [("AND", "Andes"), ("ALP", "Alps"), ("ROC", "Rockies"), ("GRL", "Greenland"), ("HIM", "Himalaya/Tibet"),
                 ("ANT", "Antarctica"), ("other_mtn", "other mountain"), ("fric", "friction"), ("gwd", "gravity-wave drag")]
    reg_rows = "".join(
        f"<tr><th scope='row'>{lab}</th>" + (f"<td>{num(100 * it[k]['coef'], 1)}% <span class='sig'>{ci(100 * it[k]['lo'], 100 * it[k]['hi'], 1)}</span></td>"
                                            if it[k].get("fdr") else f"<td class='ns'>{num(100 * it[k]['coef'], 1)}% n.s.</td>")
        + f"<td>{num(it[k]['share_of_mean_residual'], 1)}</td></tr>" for k, lab in reg_terms)
    t_reg = ("<div class='tw'><table class='res'><thead><tr><th>in-window hourly residual regressed on</th><th>coefficient</th>"
             "<th>share of the mean residual (Hadley)</th></tr></thead><tbody>" + reg_rows
             + f"<tr class='diff'><th scope='row'>constant</th><td>{num(it['intercept']['coef'], 1)} Hadley <span class='sig'>{ci(it['intercept']['lo'], it['intercept']['hi'], 1)}</span></td>"
             f"<td>{num(it['intercept']['coef'], 1)}</td></tr></tbody></table></div>")
    regnames = [("ORO", "terrain above 1000 m or rough subgrid orography, and 500 km downstream"), ("OCX", "ocean 30–65°"),
                ("OCO", "other ocean (tropics, subtropics, polar)"), ("LND", "other land")]
    t_split = ("<div class='tw'><table class='res'><thead><tr><th>share of the increment</th><th>area</th><th>below 850 hPa, all days</th>"
               "<th>DJF</th><th>JJA</th><th>whole column</th></tr></thead><tbody>"
               + "".join(f"<tr><th scope='row'>{lab}</th><td>{100 * area[r]:.0f}%</td><td>{share('amL', 'ALL', r)}</td>"
                         f"<td>{share('amL', 'DJF', r)}</td><td>{share('amL', 'JJA', r)}</td><td>{share('amC', 'ALL', r)}</td></tr>"
                         for r, lab in regnames)
               + "</tbody></table></div>")
    # ERA5 AAM regression on Nino-3.4 (peak lag) for the explainer
    lr = pd.read_csv(D / "q2_era5_lagreg.csv")
    aamr = lr[lr.series == "aam_glob"].set_index("lag")
    aam_peak_lag = int(aamr.slope.idxmax()); aam_peak = float(aamr.slope.max())

    def sig_span(s, frame=lr, col="sig"):
        g = frame[frame.series == s].set_index("lag").sort_index()
        L = g.index[g[col]].tolist()
        return (min(L), max(L), int(g.index.min())) if L else None
    span_AND, span_mtn, span_fric = sig_span("mtn5_AND"), sig_span("mtn_glob"), sig_span("fric_glob")
    span_cmtn = sig_span("mtn_glob", pd.read_csv(D / "q2_cmip6_slopes_summary.csv"), "robust")

    WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}

    def span_txt(sp):
        a, b, first = sp
        lead = f"{'at least ' if a == first else ''}{WORDS.get(abs(a), abs(a))} month{'s' if abs(a) != 1 else ''} before"
        return f"from {lead} the Niño-3.4 peak to {WORDS.get(b, b)} month{'s' if b != 1 else ''} after"

    def span_lags(sp):
        a, b, first = sp
        return f"{'−' if a < 0 else '+'}{abs(a)} to {'−' if b < 0 else '+'}{abs(b)} months"
    LODMS = 1e25 / 5.2092e33 * 86400 * 1e3                               # ms of length of day per 1e25 kg m2/s
    # GFZ global AAM by ENSO phase (torque_gfz_enso.py), the explainer's opening figure
    GZ = json.loads((D / "gfz_enso.json").read_text())
    go = GZ["oni"]
    gs = pd.read_csv(D / "gfz_enso_series.csv", parse_dates=["month"]).set_index("month")
    g26 = float(gs.aam3.loc["2026-08-01"]); g26_oni = float(gs.oni.loc["2026-08-01"])
    g26_rank = int((gs.aam3.dropna() > g26).sum() + 1)
    g26_top = gs.aam3.drop(pd.Timestamp("2026-08-01")).dropna().idxmax()
    g26_ratio = g26 / (go["slope_per_K"]["est"] * g26_oni)
    g_off_pre = GZ["gfz_minus_era5_annual"]
    off_pre = sum(g_off_pre[str(y)] for y in (1976, 1977, 1978)) / 3
    off_post = [v for k, v in g_off_pre.items() if int(k) >= 1979]
    off_post = (sum(off_post) / len(off_post))
    off_dev = max(abs(v - off_post) for k, v in g_off_pre.items() if int(k) >= 1979)
    slp_all = comp[(comp.cmp == "ONI-slope") & (comp.metric == "lp_mean") & comp.sig]
    alpS = slp_all[(slp_all.range == "ALP") & (slp_all.season == "SON")]

    body = f"""
<article class="study">
  <p class="crumbs"><a href="/">Home</a> › Studies › El Niño and angular momentum</p>
  <h1>{H1}</h1>
  <p class="dek">In El Niño years the westerly winds strengthen and the day grows a fraction of a millisecond longer: the atmosphere
  spins up and the solid Earth slows. The extra spin can only come from the ground. When surface pressure is higher on the east side
  of a mountain range than on its west side, the air presses westward on the mountains; by Newton&rsquo;s third law the mountains push
  the air eastward, adding westerly angular momentum to the atmosphere and taking the same amount from the solid Earth, whose rotation
  slows very slightly. In El Niño the warm ocean reshapes the pressure pattern around South America: the southeast Pacific high west
  of the Andes weakens and pressure rises over the continent, so pressure stays higher on the east side of the range for months. That
  standing difference keeps the mountain torque positive month after month and brings the extra westerly momentum in; friction,
  through weaker trade winds, works against it. This study measures that chain, and why a reanalysis cannot fully account for it:
  ERA5 four times a day from 1959 to 2026, {n_models} CMIP6 models, and an hour-by-hour budget of the 2026 rise.</p>

  <section class="lede">
  <h2>The short answer</h2>
  <ul>
    <li><b>1. The Andes torque supplies it; friction brakes it.</b> The El Niño rise in the atmosphere's angular momentum starts in the
    ocean and comes in through mountain torque. As the eastern Pacific warms, the southeast Pacific high west of the Andes weakens and
    pressure rises over South America: a pressure difference across the range that lasts for months (the part of the torque that
    lasts 30 days and more), not more weather events. In December–February the Andes torque is {abs(lpA['diff']):.1f} Hadley higher than in
    neutral winters ({pval(lpA['p'])}; in CMIP6 {num(cA['median'], 2)}, all {int(cA.n)} models with daily data of one sign), and the Andes
    torque regressed on Niño-3.4 is significantly positive {span_txt(span_AND)}: a long, steady push. Over the six months before the
    peak the mountain torque supplies {num(mt['est'], 2)} {ci(mt['lo'], mt['hi'], 2)} ×10²⁵ kg m² s⁻¹ per kelvin of Niño-3.4 in ERA5,
    and {num(cmt['median'], 2)} with all {int(cmt.n)} CMIP6 models of one sign. The Andes are the largest single source
    ({num(ce['mtn5_AND']['est'], 2)} in ERA5, {num(cAND['median'], 2)} in CMIP6 with every model agreeing), then the Rockies. Friction is
    not the driver: the weaker trade winds hand the atmosphere less westerly momentum from the spinning Earth, so friction works against
    the rise, taking back about {abs(cfr['median'] / cmt['median']):.0%} of the mountain torque in the models and, in ERA5, more than
    all of it (result 2). The individual east-side highs are no more frequent or stronger: none of 24 range–season tests of their
    number, strength or variance is significant. In May 2026 the Andes torque was {abs(may26):.1f} Hadley above normal, second since 1959
    only to May {may_first[0]}.</li>
    <li><b>2. ERA5's model loses westerly momentum, and the analysis puts it back.</b> At each of ERA5's twice-daily 4D-Var window
    starts the analysis adds westerly angular momentum: {num(g10['mean'], 3)} and {num(g22['mean'], 3)} ×10²⁵ kg m² s⁻¹ at 09 and 21 UTC,
    about {num(per_day / S.HADLEY_DAY_TO_AAM, 0)} Hadley held all day, every day, in every month of 2025–26. ERA5's own torques, sampled
    four times a day, miss the annual budget by {num(fsum, 1)} Hadley. The physics ensemble IFS-ENS drains angular momentum in free
    forecasts (−0.31 ×10²⁵ by day 15) while closing its own torque budget, so its torques remove the momentum; the data-driven AIFS-ENS
    shows no global drain. ERA5 runs IFS cycle 41r2, before ECMWF's 2022 retuning of orographic drag, and the increments correct
    whatever loses the momentum: too much drag is the leading candidate, not the only one.</li>
    <li><b>3. Too much drag, but not where the mountains are.</b> Operational experience with GEM, among other models, is that more drag
    gives better scores at the cost of too-weak low-level flow in places, and ECMWF raised its turbulent orographic form drag over the
    years "to optimise forecast skill" (Kanehama et al., 2022). The increment maps do not point to the mountains, though: of the
    angular momentum the analysis adds below 850 hPa, {share('amL', 'ALL', 'ORO')} lands over or within 500 km downstream of rough
    terrain, which covers {100 * area['ORO']:.0f}% of the globe, and {share('amL', 'ALL', 'OCO')} over the tropical and other oceans,
    where the analysis weakens trade easterlies that the model makes too strong. Near the surface in mid-latitudes the analysis
    <i>weakens</i> the westerlies; it adds westerly momentum above the boundary layer and over the storm-track oceans
    ({share('amC', 'ALL', 'OCX')} of the whole-column increment). The pattern is an oceanic boundary layer that carries too much
    momentum to the surface and an extratropical free troposphere short of westerlies, not excess form drag over the ranges.</li>
    <li><b>Which comes first.</b> Day to day, high mountain torque comes before the rise in global angular momentum and the subtropical jet, not
    after it, and the 500 hPa pattern that sets up an east-side high (a ridge upstream, a trough downstream) is the same in El Niño
    and neutral years.</li>
  </ul>
  </section>

  <h2>1. What angular momentum is, and why it rises in El Niño</h2>
  <p>Angular momentum about the Earth's axis measures how much the atmosphere turns faster than the planet. Most of it is in the
  westerly winds: a parcel moving east at 30 m s⁻¹ in the subtropical jet carries far more than one at the same speed near a pole,
  because it is farther from the axis. A smaller part, the mass term, comes from how the air is spread between latitudes, since the
  rotating Earth carries mass at the equator round faster. Angular momentum is conserved: the atmosphere can gain it only by taking it
  from the solid Earth and oceans, so when the atmosphere spins up the Earth slows. Geodesists see it in the length of the day,
  which tracks the atmosphere's angular momentum and, as Rosen et al. (1984) first showed, rises in El Niño. For this budget,
  10²⁵ kg m² s⁻¹ of extra atmospheric angular momentum lengthens the day by about {LODMS:.2f} ms.</p>
  <p><b>Why El Niño spins the atmosphere up.</b> El Niño warms the whole tropical troposphere, not just the eastern Pacific. ENSO is
  the dominant cause of the year-to-year variability that is symmetric about the equator: tropical warming with bands of cooling in
  the middle latitudes of both hemispheres, from the surface to the tropopause (Seager et al., 2003). By the thermal-wind relation a
  warmer tropics against cooler subtropics means stronger westerly shear on the equatorward side of the subtropical jets, so the
  westerlies in the upper troposphere strengthen there, in both hemispheres. In ERA5 the global angular momentum regressed on Niño-3.4
  peaks {abs(aam_peak_lag)} month{'' if abs(aam_peak_lag) == 1 else 's'} {'after' if aam_peak_lag > 0 else 'before'} the Niño-3.4 anomaly at {num(aam_peak, 2)} ×10²⁵ kg m² s⁻¹ per kelvin, about
  {aam_peak * LODMS:.2f} ms of length of day per kelvin.</p>
  <p><b>From warm ocean to faster winds.</b> The thermal-wind change says where the westerlies strengthen, not where their momentum
  comes from: angular momentum is conserved, so stronger jets with no torque would have to be paid for by easterlies somewhere else.
  The momentum is drawn from the solid Earth, mostly across the Andes, in four steps. (1) The central and eastern Pacific warm.
  (2) The pressure pattern responds: the southeast Pacific high, which normally sits west of the Andes, weakens and pressure rises over
  South America; in northern winter the deeper Aleutian low does the same west of the Rockies (Figure {FN['slp']}). (3) Pressure is then
  higher on the east side of both ranges, not for a few days but for months, so their mountain torque stays positive month after month.
  (4) That torque moves westerly angular momentum from the Earth to the atmosphere. Friction is the brake: the trade winds normally
  hand the atmosphere westerly momentum from the spinning Earth, and in El Niño they weaken, so less comes in. The rise is what the
  mountain torque supplies minus what friction holds back (section 3).</p>
  <p><b>What the record shows.</b> GFZ Potsdam's record of the atmosphere's angular momentum, computed from ECMWF's analyses
  separately from the ERA5 calculation used in the rest of this study, shows the rise plainly (Figure {FN['gfz']}). Since 1979 global
  angular momentum has been above normal in {100 * go['share_EN_above']:.0f}% of El Niño seasons and below normal in
  {100 * go['share_LN_below']:.0f}% of La Niña seasons. On average it is {num(go['EN']['est'], 2)} {ci(go['EN']['lo'], go['EN']['hi'], 2)}
  ×10²⁵ kg m² s⁻¹ in El Niño seasons, {num(go['LN']['est'], 2)} {ci(go['LN']['lo'], go['LN']['hi'], 2)} in La Niña seasons and near
  zero in neutral ones, a swing of about {go['EN_minus_LN']['est'] * LODMS:.2f} ms in the length of the day. It rises by
  {num(go['slope_per_K']['est'], 2)} {ci(go['slope_per_K']['lo'], go['slope_per_K']['hi'], 2)} ×10²⁵ per kelvin of the ONI
  (r = {go['r']['est']:.2f}, {pval(max(go['r']['p'], go['r']['p_bretherton']))}), and the GFZ and ERA5 monthly anomalies correlate
  at {GZ['corr_monthly_anom_with_era5']:.2f}. The ONI does not explain everything: intraseasonal swings, the MJO's among them (Weickmann et
  al., 1997), move angular momentum by a few tenths from month to month even in three-month means. 2026 stands out. In July–September angular momentum was
  {g26:.1f} ×10²⁵ above normal, {'the highest in the record' if g26_rank == 1 else f'second in the record only to early {g26_top.year}' if g26_rank == 2 else rank_word(g26_rank) + ' highest in the record'},
  and {'about twice' if 1.75 <= g26_ratio < 2.25 else f'{g26_ratio:.1f} times'} what an ONI of {g26_oni:.1f} K alone would
  predict (section 6).</p>
  {fig("gfz", "gfz_aam_enso.webp", "Top: monthly 3-month-mean global angular momentum anomalies from GFZ, 1979 to 2026, coloured red in El Niño seasons, blue in La Niña seasons and grey otherwise, with the ONI scaled by the regression overlaid. Bottom: the same anomalies against the ONI, with the regression line, the mean of each phase with its 95% interval, and the 2026 seasons marked", f"Global angular momentum (wind term) from GFZ's effective angular momentum functions, built from ECMWF's analyses: 3-month means of the monthly anomaly from 1991–2020, with the 1979–2025 trend removed, against CPC's official ONI for the same season, {go['span'][0][:4]}–{go['span'][1][:4]} ({go['n_months']} overlapping seasons). Top: the anomaly, coloured by ENSO phase; the black line is the ONI times the fitted slope; the right axis gives the change in the length of the day. Bottom: each season against its ONI. Large dots: phase means with 95% intervals (24-month block bootstrap, 2,000 draws); open circles: 2026.")}
  <p><b>How the momentum gets in.</b> All of it has to come through the surface:</p>
  <p>$$\\frac{{dM}}{{dt}} = T_{{\\text{{mountain}}}} + T_{{\\text{{friction}}}} + T_{{\\text{{gravity-wave drag}}}},\\qquad
  T_{{\\text{{mountain}}}} = -\\iint p_s\\,\\frac{{\\partial h}}{{\\partial\\lambda}}\\,dA = \\iint h\\,\\frac{{\\partial p_s}}{{\\partial \\lambda}}\\,dA .$$</p>
  <p>Torques are in Hadley (10¹⁸ N m); one Hadley held for a day changes $M$ by 0.0086 ×10²⁵ kg m² s⁻¹. The mountain torque on the
  atmosphere is <b>positive when pressure is higher on the east slope of a range than on the west</b>: the air presses westward on the
  mountain, and so on the Earth, and by Newton's third law the Earth pushes the atmosphere eastward, adding westerly momentum.
  Friction works the other way round: surface easterlies, the trades, give the atmosphere westerly momentum from the spinning Earth,
  and surface westerlies give it back, so weaker trades in El Niño mean <i>less</i> westerly momentum coming in. Gravity-wave drag is
  the model's account of the push of unresolved mountains. A cold anticyclone banked against the east side of a range makes a
  positive mountain torque, and those are familiar weather: cold surges down the lee of the Rockies (Colle and Mass, 1995), cold-air
  incursions east of the Andes (Garreaud, 2000) and the East Asian surges that follow the Siberian high. Such events supply most of
  the day-to-day mountain torque and much of the swing in angular momentum over weeks (Iskenderian and Salstein, 1998; Weickmann et
  al., 1997; Lott et al., 2004). See the <a href="/topics/mountain-torque.html">mountain torque</a> and
  <a href="/topics/atmospheric-angular-momentum.html">angular momentum</a> explainers.</p>
  <p>Two things are less settled: whether El Niño's extra momentum arrives through a change in those synoptic events or through the
  time-mean pressure pattern, and how it divides between mountain and friction torque. Reanalysis budgets do not close exactly
  (Huang et al., 1999; Egger et al., 2007), which limits what a reanalysis alone can say; climate models, which have no analysis
  increments, are the natural cross-check.</p>

  <h2>2. Data and method</h2>
  <ul>
    <li><b>ERA5</b> (Hersbach et al., 2020) from the ARCO-ERA5 store on Google Cloud, read in GitHub Actions: surface pressure, the
    turbulent and gravity-wave surface stresses and the vertical integral of eastward mass flux (an exact full-column wind term),
    at 00, 06, 12 and 18 UTC every day from 1 January 1959 to 3 October 2026, 24,748 days, none missing. Four times a day, not
    two: at 00 and 12 UTC the semidiurnal pressure tide has the same phase, so a twice-daily mean keeps its alias. The daily
    value is the mean of the four. The mass term of $M$ comes from surface pressure. For 2026 the same fields were read every hour
    from 1 March. The analysis period is 1979–2025; 1959–2025 is a sensitivity test.</li>
    <li><b>Torques</b> are the site's: friction $-a^3\\cos^2\\varphi\\,\\tau_\\lambda$, gravity-wave drag likewise from ERA5's own
    stress, mountain torque $h\\,\\partial p_s/\\partial\\lambda$ on the 0.25° grid with ERA5's orography, summed over the site's six
    boxes (5° cells; table below), over six latitude bands and over the globe.</li>
  </ul>
  {boxes}
  <ul>
    <li><b>Anomalies</b>: each daily series minus its 1991–2020 seasonal cycle (mean and four harmonics), with the 1979–2025
    linear trend removed (ERA5's torques drift, while angular momentum does not; section 9). A Lanczos filter splits each range's
    torque into a standing part (periods of 30 days and longer, 121-day window; Duchon, 1979) and a synoptic part (the rest).</li>
    <li><b>Events</b>: local peaks of the synoptic part above +1 standard deviation (one per range and season), at least 5 days
    apart; negative events likewise. The same rule applied to the total anomaly gives the "days above threshold" events a
    forecaster sees. About seven events per range per season.</li>
    <li><b>GFZ</b> effective angular momentum functions (ESMGFZ v1.0; Dobslaw et al., 2010), the external check: the wind term,
    3-hourly from ECMWF's analyses, as daily and then monthly means (complete months only), anomalies from 1991–2020 with the
    1979–2025 trend removed, then 3-month means to match the ONI's seasons. The years 1976–78 are left out: the difference
    GFZ minus ERA5 is {off_pre - off_post:.2f} ×10²⁵ larger then than its mean since 1979, against at most {off_dev:.2f} either
    way in any later year, and it is not known which record is wrong. With CPC's relative ONI (RONI) instead of the ONI the slope is
    {num(GZ['roni']['slope_per_K']['est'], 2)} {ci(GZ['roni']['slope_per_K']['lo'], GZ['roni']['slope_per_K']['hi'], 2)}; without the
    trend removal, {num(GZ['oni_no_detrend']['slope_per_K']['est'], 2)} {ci(GZ['oni_no_detrend']['slope_per_K']['lo'], GZ['oni_no_detrend']['slope_per_K']['hi'], 2)}.</li>
    <li><b>ENSO phase</b> from CPC's official ONI for each season (El Niño ≥ +0.5, La Niña ≤ −0.5, otherwise neutral). Season-years
    1979–2025: DJF {nsy['EN']['DJF']} El Niño, {nsy['NEU']['DJF']} neutral, {nsy['LN']['DJF']} La Niña; MAM {nsy['EN']['MAM']},
    {nsy['NEU']['MAM']}, {nsy['LN']['MAM']}; JJA {nsy['EN']['JJA']}, {nsy['NEU']['JJA']}, {nsy['LN']['JJA']}; SON {nsy['EN']['SON']},
    {nsy['NEU']['SON']}, {nsy['LN']['SON']}.</li>
    <li><b>Tests</b>: each season-year is one unit. El Niño minus neutral differences are tested by permuting the phase labels
    (20,000 draws) with bootstrap 95% intervals; regressions on the ONI use a moving-block bootstrap (blocks of three
    season-years). Across the 24 range–season cells of each statistic the false discovery rate is held at 10% (Benjamini and
    Hochberg, 1995; Wilks, 2016). Monthly regressions on the detrended ERSSTv6 Niño-3.4 anomaly at lags of −6 to +12 months use a
    24-month block bootstrap (2,000 draws), with the FDR across lags. Map tests are per grid cell with the FDR over the field.</li>
    <li><b>CMIP6</b> historical runs 1950–2014 from the Pangeo Google Cloud mirror (Eyring et al., 2016): {n_models} models,
    {n_members} members (up to three per model), each on its own grid with its own orography: monthly mountain torque from surface
    pressure (the torque is linear in $p_s$, so the torque of the monthly mean is the monthly-mean torque), surface stress
    <code>tauu</code>, angular momentum from <code>ua</code> masked by surface pressure, and Niño-3.4 from <code>ts</code>.
    {n_gwd} models also publish the orographic gravity-wave-drag tendency <code>utendogw</code> (Gerber and Manzini, 2016) and
    {n_daily} have daily surface pressure, which gives event statistics on each model's own ENSO (ONI-like index of its detrended
    Niño-3.4). A model result is <b>robust</b> when the across-model test passes the FDR and at least 80% of models share the sign.</li>
  </ul>

  <h2>3. How the momentum gets in: the torque budget</h2>
  <p>Summing each budget term's regression on Niño-3.4 over lags −6 to −1 months gives the angular momentum it supplies in the six
  months before the anomaly (Figure {FN['q2con']} and the table). In ERA5 the tendency term supplies {num(ce[tend]['est'], 2)}
  {ci(ce[tend]['lo'], ce[tend]['hi'], 2)} ×10²⁵ kg m² s⁻¹ per kelvin. (The regression of angular momentum itself rises by
  {num(q2['aam_regression_change_lag0_minus_lagm6'], 2)} between those lags; the two differ because one sums monthly tendencies taken
  across month edges and the other differences monthly means.) The mass term's share is negligible (its regression is about +0.02
  per kelvin).</p>
  <p>The mountain torque is the source and friction the brake, and both act for a long time. Regressed on Niño-3.4 at lags from −6
  to +12 months (Figure {FN['q2lag']} shows the global terms), the Andes torque is significantly positive at every lag from {span_lags(span_AND)} (the global
  mountain torque {span_lags(span_mtn)}; robust in CMIP6 from {span_lags(span_cmtn)}), and friction is significantly negative from
  {span_lags(span_fric)}, mostly in the tropics, where the trades weaken (the 30°S–0 and 0–30°N friction rows of the table).</p>
  {fig("q2lag", "q2_lag_regression.webp", "Lagged regressions on Niño-3.4 of the global mountain, friction and gravity-wave-drag torques, their sum and the angular momentum tendency, ERA5 above and the CMIP6 median and interquartile range below", "Regression of each monthly global term on Niño-3.4 (detrended anomalies) against lag. Top: ERA5 1979–2025, filled dots significant (FDR over lags). Bottom: CMIP6 1950–2014, median and interquartile range over models, filled dots robust.")}
  {fig("q2con", "q2_contributions.webp", "Horizontal bars of the angular momentum each term supplies before the Niño-3.4 anomaly, ERA5 bars with confidence intervals and CMIP6 diamonds with interquartile ranges, globally, by range and by latitude band", "Angular momentum supplied in the six months before the Niño-3.4 anomaly, per kelvin. Bars: ERA5 with 95% joint block-bootstrap intervals, pale where not significant after the FDR. Diamonds: CMIP6 median with interquartile range, filled where robust.")}
  {t_q2}
  <p class="tcap">ERA5 cells show the estimate and 95% interval where significant after the FDR, otherwise "n.s.". CMIP6 cells
  show the median and the share of models of its sign where robust.</p>
  <p>The two data sets agree on the mountain torque and its geography: Andes {num(ce['mtn5_AND']['est'], 2)} in ERA5 and
  {num(cAND['median'], 2)} in CMIP6 ({cAND.agree:.0%} of models), Rockies {num(ce['mtn5_ROC']['est'], 2)} and
  {num(cROC['median'], 2)}, Himalaya/Tibet {num(cHIM['median'], 2)} in CMIP6 (not significant in ERA5). They differ on friction.
  In the models friction takes back about {abs(cfr['median'] / cmt['median']):.0%} of the mountain torque, and the remainder,
  {num(cT['median'], 2)}, is about {cT['median'] / cdm['median']:.1f} times the models' tendency, {num(cdm['median'], 2)}. In ERA5 friction
  takes back more than all of it, {num(fr['est'], 2)} against {num(mt['est'], 2)}, so the torques sum to {num(ce['T_glob']['est'], 2)}
  (not significant) while angular momentum rises significantly; the difference sits in ERA5's residual ({num(ce['R_glob']['est'], 2)},
  not significant on its own), mostly the analysis increments (section 7). ERA5's monthly budget otherwise closes reasonably
  (correlation of tendency with summed torques {mc['corr']:.2f}, slope {mc['slope']:.2f}), but its residual is slow (lag-one
  autocorrelation {mc['r_lag1_R']:.2f}), which is what an interannual imbalance looks like.</p>
  <p>Composites of El Niño onsets tell the same story without reaching significance in ERA5's small sample (13 onsets against 14
  neutral years, April–December): the change in angular momentum, the summed torques and the mountain torque differ by
  {num(comp_en['dM']['diff'], 2)}, {num(comp_en['T']['diff'], 2)} and {num(comp_en['mtn_glob']['diff'], 2)} ×10²⁵ kg m² s⁻¹ with
  {pval(comp_en['dM']['p'])}, {pval(comp_en['T']['p'])} and {pval(comp_en['mtn_glob']['p'])}; only gravity-wave drag is significant
  ({num(comp_en['gwd_glob']['diff'], 2)}, {pval(comp_en['gwd_glob']['p'])}) and it is small. In CMIP6 the onset composites are robust:
  angular momentum {num(ccd['median'], 2)}, mountain torque {num(ccm['median'], 2)} and the Andes {num(cca['median'], 2)}; friction
  ({num(ccf['median'], 2)}, {ccf.agree:.0%} agreement) is not robust.</p>

  <h2>4. The Andes: a standing shift, not more events</h2>
  <p>This is the pressure difference of section 1 seen range by range: a slow shift that forms as the Pacific warms and lasts the
  season, so the torque it makes is positive day after day, rather than more or stronger weather events.</p>
  {fig("q1grid", "q1_shift_vs_events.webp", "Grid of six mountain ranges by four seasons showing the El Niño minus neutral difference in the standing torque, the number of positive torque events and their amplitude; only the Andes in DJF and the Himalaya in MAM are significant, and only in the standing part", "El Niño minus neutral seasons, ERA5 1979–2025, for each range and season. Top: the standing (30-day and longer) part of the torque. Middle and bottom: the number and the amplitude of positive synoptic events. Coloured cells are significant at a false discovery rate of 10% (permutation of season-years); grey cells, marked n.s., are not. A black frame marks a change that is robust in the CMIP6 models with daily data (significant across models and at least 80% of models of one sign).")}
  <p>The synoptic events do not change with El Niño (Figure {FN['q1grid']}, middle and bottom). Positive Andes events in DJF average
  {npA['ma']:.2f} a season in El Niño winters and {npA['mb']:.2f} in neutral ones ({num(npA['diff'], 2)} {ci(npA['lo'], npA['hi'], 2)},
  {pval(npA['p'])}); the Rockies differ by {num(npR['diff'], 2)} ({pval(npR['p'])}) and the Himalaya by {num(npH['diff'], 2)}
  ({pval(npH['p'])}, not significant after the FDR). Their amplitude does not change either (Andes DJF {num(amA['diff'], 2)} Hadley,
  {pval(amA['p'])}). Across all 24 cells, {nsig['n_pos']} event-count, {nsig['amp_pos']} amplitude and {nsig['hp_sd']} variance
  differences are significant. Against the ONI as a continuous index, {len(nslope)} of 24 cells show a significant change in event
  count, all small and of mixed sign
  ({'; '.join(f"{S.RNAME[r_.range]} {r_.season} {num(r_['diff'], 2)} per unit ONI" for _, r_ in nslope.iterrows())}).</p>
  <p>The standing part does change (Figure {FN['q1grid']}, top). Against the ONI it is significant in {len(slp_all)} cells, six of them
  large: the Andes in DJF ({num(slA['diff'])} Hadley per unit ONI {ci(slA['lo'], slA['hi'])}) and JJA ({num(slAJ['diff'])}
  {ci(slAJ['lo'], slAJ['hi'])}), the Rockies in DJF ({num(slR['diff'])} {ci(slR['lo'], slR['hi'])}) and MAM ({num(slRM['diff'])}
  {ci(slRM['lo'], slRM['hi'])}), and the Himalaya/Tibet in MAM ({num(slH['diff'])} {ci(slH['lo'], slH['hi'])}) and SON ({num(slHS['diff'])}
  {ci(slHS['lo'], slHS['hi'])}){f"; the seventh is the Alps in SON ({num(alpS['diff'].iloc[0], 2)} {ci(alpS['lo'].iloc[0], alpS['hi'].iloc[0], 2)})" if len(alpS) else ""}.
  Nearly all of the El Niño difference in the seasonal-mean torque is this standing part; the windows around positive events
  contribute nothing significant:</p>
  {t_dec}
  <p class="tcap">The CMIP6 column is the median over the {n_daily} models with daily surface pressure, with the share of models of
  the median's sign; Andes and Himalaya are robust, the Rockies in DJF ({cRD.agree:.0%}) is not. CMIP6 is also robust for the Rockies
  in MAM ({num(cRM['median'], 2)}) and JJA ({num(cRJ['median'], 2)}) and the Himalaya in SON ({num(cHS['median'], 2)}). The ERA5
  Rockies DJF difference ({num(lpR['diff'])}, {pval(lpR['p'])}) does not survive the FDR.</p>
  {fig("q1thr", "q1_threshold_events.webp", "Two grids of ranges by seasons: the El Niño minus neutral change in days-above-threshold torque events, and the part of that change not explained by a pure shift of the distribution", f"Top: events counted on the total anomaly, as a forecaster sees them. Bottom: the pure-shift test, the El Niño count minus the count obtained by shifting each neutral season by the El Niño–neutral difference in the standing part. Colours and frames as in Figure {FN['q1grid']}.")}
  <p><b>Why El Niño winters look eventful.</b> Counted on the total anomaly, Andes DJF has {nxA['ma']:.2f} positive events per
  season in El Niño winters against {nxA['mb']:.2f} in neutral ones ({pval(nxA['p'])}, just short of the FDR) and
  {abs(nxnA['diff']):.1f} fewer negative ones ({pval(nxnA['p'])}). Shifting each neutral winter by the El Niño–neutral difference in the
  standing part predicts {sA['expected_shift']:.2f}; the excess, {num(sA['excess'], 2)} {ci(sA['lo'], sA['hi'], 2)}, is not
  significant ({pval(sA['p'])}), nor is it for the Rockies ({num(sR['excess'], 2)}, {pval(sR['p'])}). In the models the extra
  threshold events are robust (Andes DJF {num(cnA['median'], 2)} a season, {cnA.agree:.0%} of models; Rockies DJF
  {num(cnR['median'], 2)}, {cnR.agree:.0%}) while the synoptic counts are not (the strongest, Himalaya DJF, {num(cevH['median'], 2)}
  with {cevH.agree:.0%} agreement). Two cells go the other way: in El Niño springs the Himalaya/Tibet has fewer events than the
  shift alone gives ({num(sH['excess'], 2)} {ci(sH['lo'], sH['hi'], 2)}, significant after the FDR), as does Greenland in SON
  ({num(sG['excess'], 2)}).</p>
  {fig("slp", "slp_on_oni.webp", "Global maps of seasonal-mean sea-level pressure regressed on the ONI for DJF and MAM, with the Himalaya, Rockies and Andes boxes outlined", "The mechanism. ERA5 1991–2020 seasonal-mean sea-level pressure regressed on the ONI, shaded only where significant (FDR 10%). In DJF the deeper Aleutian low lowers pressure west of the Rockies and the weaker southeast Pacific high lowers it west of the Andes, while pressure rises over South America: higher on the east side of both ranges, a positive torque. In MAM, pressure rises over India and the Philippine Sea.")}

  <h2>5. Which comes first</h2>
  {fig("q3ll", "q3_lead_lag.webp", "Four panels of lagged correlation between each range's synoptic mountain torque and the global angular momentum tendency, the global angular momentum and the zonal-mean subtropical 200 hPa wind", "Lagged correlation of each range's synoptic torque with the global tendency of angular momentum, global angular momentum and the zonal-mean 200 hPa wind at 25–40° in the same hemisphere (all &lt;30-day parts). Dots: significant, with effective sample size after Bretherton et al. (1999) and the FDR over lags.")}
  <p>Figure {FN['q3ll']} correlates each range's synoptic torque, day by day, with what the atmosphere does before and after. High torque
  follows a few days of low global angular momentum (lowest correlation {num(min(p_.r for p_ in pre_lo), 2)} at lags of
  −{min(abs(int(p_.lag)) for p_ in pre_lo)} to −{max(abs(int(p_.lag)) for p_ in pre_lo)} days) and comes before a rise (up to
  {num(max(p_.r for p_ in post_hi), 2)} {dd(post_hi[int(np.argmax([p_.r for p_ in post_hi]))].lag)} later). The zonal-mean subtropical jet
  follows weakly: {num(jR.r, 2)} {dd(jR.lag)} after Rockies torque in DJF and {num(jA.r, 2)} {dd(jA.lag)} after Andes torque in DJF; the
  Himalayan torque in DJF shows nothing in the zonal mean (largest {num(jH.r, 2)}). Before Andes torque the jet is, if anything,
  slightly weaker ({num(jbA.r, 2)}, {dd(jbA.lag, before=True)}). The torque leads; it is not a response to a jet that is already
  stronger.</p>
  {fig("q3z", "q3_z500_precursors.webp", "Maps of 500 hPa height one to three days before positive torque events over the Himalaya, Rockies and Andes in DJF, for all events and for El Niño minus neutral", "The precursor. 500 hPa height (&lt;30-day part) averaged 3 to 1 days before positive events, all events (left) and El Niño minus neutral (right), shaded only where significant (FDR 10% over the map; contours every 10 m show all values).")}
  <p>The precursor is the textbook one for a lee-side surge: a ridge upstream and poleward of the range and a trough downstream,
  over the Urals and northern China for the Himalaya, Alaska and the central United States for the Rockies, and southern Chile
  and the southwest Atlantic for the Andes ({qc['HIM_DJF']['frac_sig_all']:.0%}, {qc['ROC_DJF']['frac_sig_all']:.0%} and
  {qc['AND_DJF']['frac_sig_all']:.0%} of each map significant). El Niño does not change it: no cell of the El Niño minus neutral
  difference is significant for any of the three ranges in DJF ({qc['AND_JJA']['frac_sig_diff']:.0%} of the map for the Andes in
  JJA). Projecting each day's 500 hPa field on the precursor pattern and counting its peaks, El Niño seasons have
  {'a significantly different' if pt_any else 'no significantly different'} number of precursors and no different share that are
  followed by an event within four days (all {len(pt[pt.metric != 'n_events'])} tests not significant after the FDR; the closest are
  {num(ptH['diff'], 2)} precursors a season over the Himalaya in MAM, {pval(ptH['p'])}, and a {num(ptR['diff'], 2)} hit rate over
  the Rockies in DJF, {pval(ptR['p'])}).</p>

  <h2>6. The 2026 event</h2>
  <p>By the CPC ONI, 2026's El Niño grew from {num(oni26[4], 2)} in MAM to {num(oni26[8], 2)} in JAS. In May the Andes torque was
  {num(may26)} Hadley above its seasonal normal, the {'largest' if may_rank == 1 else ('second-largest' if may_rank == 2 else f'{may_rank}th largest')} May
  since 1959 after May {may_first[0]} ({num(may_first[1])}), the start of that year's very strong El Niño. Over 15 April – 19 September
  the Andes torque anomaly added {num(att['Andes']['v2026'])} ×10²⁵ kg m² s⁻¹ (the {rank_txt(att['Andes']['rank_of_48'])},
  z = {num(att['Andes']['z'], 1)}; {num(att['Andes']['v2026_detrended'])} with ERA5's torque trend removed), against
  {num(att['Andes']['en_onset_mean'])} in the average El Niño onset.</p>
  {fig("b26cum", "b2026_cumulative.webp", "Lines of the cumulative 2026 change in angular momentum anomaly and of the cumulative torque anomalies by term since 15 April", "ERA5, from hourly fields: the change in the angular momentum anomaly (motion plus mass term) since 15 April and the cumulative torque anomalies that should explain it. Descriptive.")}
  <p>The budget does not close. The angular momentum anomaly rose by {num(B['attribution_dM_hourly_anomaly'])} ×10²⁵ kg m² s⁻¹ over
  the window while the torque anomalies summed to {num(B['attribution_total_torque'])}: the Andes term ran close to the angular
  momentum curve, mountain torque outside the six boxes added {num(att['rest of mountain']['v2026'])}, but the other five ranges
  ({num(att['other ranges']['v2026'])}) and friction in the tropics ({num(att['friction tropics']['v2026'])}) worked against it
  (Figure {FN['b26cum']}). That residual, {num(vh['residual'])}, is not unusual for ERA5: the same window leaves {num(hw['mean'])} ±
  {hw['sd']:.1f} in 1979–2025, so 2026 sits {num(zr, 1)} standard deviations from the usual. The honest reading is that ERA5 cannot
  attribute the 2026 rise: the Andes signal is real and among the largest on record, but in ERA5 the torques as a whole explain none
  of the rise, and the analysis increments carry it (section 7). In the models, where nothing is added between time steps, friction
  gives back only about {abs(cfr['median'] / cmt['median']):.0%} of a mountain torque of this kind.</p>
  {fig("b26att", "b2026_attribution.webp", "Horizontal bars of each torque term's 2026 integral against the interannual spread of the same window", "Each term's anomaly integrated over 15 April – 19 September 2026 (dark: as observed; pale: with ERA5's 1979–2025 trend removed), against ± one standard deviation of the same window in 1979–2025 (beige).")}
  {t_att}
  <p class="tcap">Anomalies against ERA5's 1991–2020 climatology at four times a day; 2026 daily means from hourly fields. Rank 48 is
  the largest. The trend-removed column is a sensitivity, not a correction: ERA5's tropical friction drifts by
  {num(att['friction tropics']['trend_hadley_per_decade'], 2)} Hadley per decade.</p>
  <p><b>The analysis-increment floor.</b> With every hour of 2026 the budget can be followed through ERA5's twelve-hour 4D-Var
  windows (Figure {FN['b26hr']}). Inside a window the residual is {num(wt['inside_mean'])} ± {wt['inside_sd']:.1f} Hadley each hour. In
  the hour from 09 to 10 UTC it jumps to {num(g[10]['mean'], 0)} ± {g[10]['std']:.0f} and from 21 to 22 UTC to {num(g[22]['mean'], 0)} ±
  {g[22]['std']:.0f}: measured against the in-window mean, the analysis adds {num(inc['10'], 3)} and {num(inc['22'], 3)} ×10²⁵ kg m² s⁻¹
  of angular momentum at each start. A three-hourly product sees the same thing ({num(t3['across_mean'], 0)} ± {t3['across_sd']:.0f}
  Hadley across window starts against {num(t3['inside_mean'], 1)} ± {t3['inside_sd']:.1f} inside). The daily means of the hourly
  residual are {num(dfh['mean_R'])} ± {dfh['sd_R']:.1f} Hadley, with almost no memory from day to day (lag-one correlation
  {dfh['lag1']:.2f}); its spectrum is a comb of lines at whole cycles per day on a low floor, and at periods longer than about two days
  the torques carry nearly all of the variance of the tendency.</p>
  {fig("b26hr", "b2026_hourly.webp", "Bar chart of the hourly budget residual by hour of day with the two window-start bars cut off and labelled, and spectra of the tendency, the summed torques and the residual", "Top: the hourly residual (tendency minus torques) by the hour that ends each interval, mean ± sd over 1 March – 3 October 2026; the two window-start bars are cut off and labelled. Bottom: spectra of the tendency, the summed torques and the residual.")}
  <p><b>GFZ as a check.</b> The GFZ effective angular momentum functions (Dobslaw et al., 2010), built from ECMWF's operational
  analysis every three hours with the pressure tides removed and an inverted-barometer ocean, put the absolute change over the
  window at {num(gfz['dM_window_total'], 2)} ×10²⁵ kg m² s⁻¹ (wind term {num(gfz['dMr_window'], 2)}, mass term
  {num(gfz['dMm_window'], 2)}) against {num(gfz['era5_dM_window_total_absolute'], 2)} from ERA5's daily means of hourly fields; their
  daily tendencies correlate at {gfz['corr_daily_dMdt_vs_era5']:.2f}. The mass terms are not the same quantity: GFZ's includes the
  ocean's inverted-barometer response.</p>

  <h2>7. Why a reanalysis budget does not close: ERA5's analysis increments</h2>
  <p>ERA5's open archive carries no analysis-increment fields, but its hourly fields jump at each 4D-Var window start, so the
  increment can be estimated from the jump. For any band's angular momentum $X$ with torques $T$, let
  $R(h) = X(h) - X(h-1) - T(h)\\,\\Delta t$ be the residual of the hour ending at $h$; the increment at the 09 UTC window start is</p>
  <p>$$J_{{10}} = R(10) - \\tfrac{{1}}{{2}}\\left[R(09) + R(11)\\right],$$</p>
  <p>and likewise $J_{{22}}$ from 20–23 UTC: the window-start hour minus the mean of the hours either side, which removes the
  smooth part of everything that is not an increment (the transport of angular momentum between bands, the tides, the local
  trend). The torques themselves barely change across the jump ({num(T10['mean'], 4)} ×10²⁵ kg m² s⁻¹, negligible). The surface
  fields were read at 08–11 and 20–23 UTC for every day from {I['days'][0]} to {I['days'][1]} ({I['days'][2]} days) and integrated in
  5° bands; the 37-level winds at the same hours give the vertical structure ({u10['n_days']} days), and the same estimator applied to
  the wind itself on a 2.5° grid, every third day, gives the maps. Intervals are 10-day block bootstraps over days; the FDR is held at
  10% across bands, across levels and bands for the section, and across grid cells for the maps. A jump is not purely an increment:
  it is the step between two consecutive analysis trajectories, so it also contains any adjustment of the model in the first hour of
  the new trajectory. And ERA5 is IFS cycle 41r2 (Hersbach et al., 2020): the increments describe that model, before ECMWF's 2022
  revision of orographic drag (Kanehama et al., 2022).</p>
  {fig("incprof", "inc_profile.webp", "Latitude profile of the mean angular momentum increment in 5-degree bands for the 09 and 21 UTC window starts, and its split into wind and mass terms", "Top: the mean increment in each 5° band at the 09 UTC (blue) and 21 UTC (orange) window starts, with 95% intervals; filled dots are significant after the FDR. The legend's global totals are in 10²² kg m² s⁻¹ (48 ×10²² = 0.048 ×10²⁵). Bottom: both window starts together, split into the wind (motion) and mass terms, with intervals and significance as above.")}
  <p><b>How much and where.</b> Summed over the bands, the analysis adds {E(g10)} ×10²⁵ kg m² s⁻¹ of angular momentum at the
  09 UTC start and {E(g22)} at 21 UTC, about {num(per_day, 3)} a day, the equivalent of a steady torque of
  {num(per_day / S.HADLEY_DAY_TO_AAM, 1)} Hadley. (The global series of section 6, with the in-window mean as the baseline, gave
  {num(inc['10'], 3)} and {num(inc['22'], 3)}; the two estimators differ in the baseline, not in substance.) The <b>wind term is the
  same at both starts</b>, {E(w10)} and {E(w22)}; the difference between them is in the <b>mass term</b>, {E(m10)} at 09 UTC
  against {E(m22)} at 21 UTC, a surface-pressure increment. By latitude the analysis adds westerly angular momentum to both
  extratropical belts and removes it from the deep tropics, and every belt is significant:</p>
  {t_bel}
  <p class="tcap">Belts are sums of the 5° bands; all twelve belt means are significant after the FDR. Most of the tropical removal is
  in the mass term (Figure {FN['incprof']}, bottom): a redistribution of surface pressure between the equator and the subtropics, which
  says nothing about drag. The wind term alone is positive in both extratropical belts and slightly negative in the deep tropics.</p>
  {fig("incseason", "inc_season.webp", "Monthly global increment per day and a month by latitude diagram of the increment", "Top: the global increment per day, month by month, with 95% intervals (complete months only). Bottom: the same by latitude in 5° bands; hatched where not significant (t-test with an effective sample size from the lag-one autocorrelation, FDR 10%).")}
  <p><b>Through the year.</b> The global increment is smallest in northern summer ({gmin}: {num(gfull[gmin]['mean'], 3)} ×10²⁵ a day) and
  largest in northern winter and spring ({gmax}: {num(gfull[gmax]['mean'], 3)}); the latitude pattern of extratropical gain and tropical
  loss is there in every month (Figure {FN['incseason']}).</p>
  {fig("incsection", "inc_section.webp", "Latitude by pressure sections of the wind-term angular momentum increment per layer at the two window starts, troposphere and stratosphere on separate scales", "The wind-term increment by latitude and layer at the two window starts: the angular momentum each 5° band and pressure layer gains, in units that sum to the column. Below: 1000–100 hPa on an axis linear in pressure (so in mass); above it, 100–1 hPa on its own scale. Hatched where not significant (10-day block bootstrap, FDR 10% over the cells with data).")}
  <p><b>By height.</b> The vertical sum of the 37-level increment follows the column estimate from the surface fields closely (correlation
  {u10['corr_column_vs_2d']:.3f} across bands and days) but is about {100 * (col37 / col2d - 1):.0f}% larger ({num(col37, 3)} against
  {num(col2d, 3)} per window), probably from summing 37 pressure levels with their lowest layer clipped at the surface rather than
  integrating the model levels exactly. Of the wind-term increment, the largest share goes into the layers below 850 hPa
  ({fsec.get('J10', 0):.0%} and {fsec.get('J22', 0):.0%} of the section's cells with data are significant):</p>
  {t_lay}
  <p>Figure {FN['incsection']} shows where. Between about 35°S and 35°N the analysis adds westerly momentum from the surface to about
  900 hPa and removes it at 850–700 hPa: a vertical dipole across the trade-wind boundary layer, in both window starts, which the
  maps below place over the trade-wind oceans. In the extratropics of both hemispheres it adds westerly momentum through the middle
  troposphere (400–600 hPa, strongest at 30–55°S), and in the northern subtropical jet at 100–200 hPa. Above 100 hPa the
  increments are small: westerly momentum is added at 70–100 hPa from the subtropics across the tropics and removed at 10–30 hPa in
  the northern subtropics.</p>

  <h3>Where the low-level increment goes</h3>
  {fig("mapsu", "inc_maps_u.webp", "World maps of the mean zonal-wind increment at 925 and 850 hPa with terrain contours", "The mean increment of the zonal wind at 925 hPa (top) and 850 hPa (bottom), both window starts together, per day; shaded only where significant (t-test with an effective sample size from the lag-one autocorrelation, FDR 10% over the map). Contours: terrain at 500, 1500 and 3000 m.")}
  {fig("mapsseason", "inc_maps_amL_season.webp", "World maps of the angular momentum increment of the layers at and below 850 hPa in December to February and June to August", "The angular momentum increment of the layers at and below 850 hPa, per 2.5° cell and day, in DJF (top) and JJA (bottom); shaded only where significant (FDR 10%). Terrain contours as in the previous figure.")}
  <p>The maps test the drag hypothesis directly: if ERA5's model lost momentum mainly to excess low-level orographic drag
  (blocking and turbulent orographic form drag), the low-level increments would sit over and just downstream of the Rockies, the Andes,
  Tibet, the East African highlands and the Antarctic margins. They do not. At 925 hPa the analysis adds westerly momentum over most of
  the trade-wind oceans and around Antarctica; at 850 hPa it removes it over the subtropical oceans, the upper half of the dipole in
  Figure {FN['incsection']}. In the zonal mean the 1000 hPa wind is increased by {num(u1000_trop, 2)} m s⁻¹ a day at 10–30° (weaker trade
  easterlies) and decreased by {num(u1000_mid, 2)} m s⁻¹ at 40–60° (weaker surface westerlies); the 10 m wind likewise
  ({num(u10_trop, 2)} and {num(u10_mid, 2)}). Over the mid-latitude oceans the analysis adds westerlies from 925 hPa up
  ({num(u925_mid, 2)} m s⁻¹ a day at 40–60°), while at 850 hPa between 40 and 55°N it removes them ({num(u850_nh, 2)}).</p>
  {fig("mapssplit", "inc_maps_split.webp", "Bar chart of the share of the global low-level angular momentum increment falling in four regions, for all days, DJF and JJA, against each region's share of the Earth's area", "The share of the global increment of angular momentum below 850 hPa that falls in each region, all days and by season, with 95% block-bootstrap intervals over days; the dashed lines mark each region's share of the Earth's area. Regions are defined at 0.25° from ERA5's orography, its subgrid-orography standard deviation and its land-sea mask.")}
  {t_split}
  <p class="tcap">"Rough terrain" is terrain above 1000 m or an ERA5 subgrid-orography standard deviation above 100 m, extended 500 km
  east (downstream); the four regions add up to the globe. With terrain above 1000 m alone the low-level share is
  {share('amL', 'ALL', 'HGT')} ({100 * area['HGT']:.0f}% of the area); with the subgrid criterion alone {share('amL', 'ALL', 'SDO')}
  ({100 * area['SDO']:.0f}%).</p>
  <p>Over and downstream of rough terrain, {100 * area['ORO']:.0f}% of the globe, the analysis puts {share('amL', 'ALL', 'ORO')} of its
  low-level angular momentum, less than an even spread would give; even in northern winter, when blocking drag matters most, the
  share is {share('amL', 'DJF', 'ORO')}, and in JJA it is negative. The tropical, subtropical and polar oceans outside 30–65° take
  {share('amL', 'ALL', 'OCO')}, most of it in the trade-wind boundary layer. In the whole column the storm-track oceans at 30–65°
  dominate ({share('amC', 'ALL', 'OCX')} on {100 * area['OCX']:.0f}% of the area).</p>

  <h3>Possible explanations</h3>
  <p><b>Models are tuned toward more drag.</b> The parameters that set parametrised surface drag "are poorly constrained and often
  tuned to obtain the desired model behaviour, i.e. forecast skill or mean climate" (Sandu et al., 2016, technical memorandum
  version), and models of comparable resolution differ over land by up to 20% in zonal-mean subgrid surface stress, and more in how it
  is split between schemes. At ECMWF the turbulent orographic form drag coefficient "has been increased over time to optimise forecast
  skill": in cycle 40r1 it and the blocking coefficient were raised to offset the loss of skill from reducing turbulent diffusion in
  stable layers, a repartition "purely made to optimise skill, in the absence of any constraints on the drag processes", which left
  the IFS with much more form-drag stress than the Met Office model; ECMWF also reports IFS surface winds that are too weak over
  orography (Kanehama et al., 2022). ERA5's cycle 41r2 came after 40r1 and before the 2022 revision that reduced form drag again.
  Earlier, budget studies of model drift had pushed models towards more low-level and less stratospheric drag (Klinker and
  Sardeshmukh, 1992, as summarised by Brown, 2004), and Brown (2004) found the parametrised orographic torque excessive at low
  resolution between 20 and 50°N in winter. Operational experience with GEM is similar: more drag gave better scores at the cost of
  too-weak 850 hPa flow in places. None of these sources measures what drag does to a model's angular momentum budget; that a model
  tuned this way loses westerly momentum, which the analysis then puts back, is an inference, and ERA5's increments are consistent
  with it.</p>
  <p><b>But not in the form of low-level orographic drag.</b> The maps put the low-level increment over the oceans, not the mountains,
  and near the surface they weaken both the trades and the mid-latitude westerlies: the model's surface winds over the oceans are too
  strong, not too weak. In ERA5 the 850 hPa westerlies of the northern mid-latitudes are, if anything, too strong as well, unlike the
  GEM experience. What the increments describe is an oceanic boundary layer that carries too much momentum down to the surface (or
  surface-wind observations, from scatterometers and buoys, that pull the surface winds down), with too little left above it, and an
  extratropical free troposphere that is short of westerly momentum. Too strong a surface wind over the westerly belts also means too
  much surface stress and so too much friction torque: drag of the boundary-layer kind rather than the orographic kind remains a
  candidate for the lost momentum.</p>
  <p><b>Gravity-wave drag aloft.</b> COORDE (van Niekerk et al., 2020) found, as Kanehama et al. (2022) summarise it, that the IFS,
  like several other models, underestimates orographic gravity-wave drag in the lower stratosphere. On its own that would leave too much
  westerly momentum there and the analysis would have to remove it. Over 25–50°N between 50 and 150 hPa, the latitudes of the Himalaya
  and the Rockies, the increments instead add westerly momentum: {E(u10['ls_25_50N'], 4)} at 09 UTC and {E(u22['ls_25_50N'], 4)} at
  21 UTC, against {E(u10['trop_25_50N'], 4)} and {E(u22['trop_25_50N'], 4)} in the troposphere below 400 hPa. A zonal mean cannot isolate
  the air over and downstream of the ranges, which is where COORDE looked, so this tests the zonal-mean budget, not COORDE's regional
  result.</p>
  <p><b>The steady residual inside the windows.</b> Within a window the hourly residual is {num(ir['mean_residual'], 1)} Hadley.
  Regressing it on each range's mountain torque, the friction and the gravity-wave drag (2026, {ir['n_hours']:,} hours, day-block
  bootstrap) explains {ir['r2']:.0%} of its variance:</p>
  {t_reg}
  <p class="tcap">A coefficient of +1% means the residual behaves as if the model's own torque were 1% larger than the one recomputed
  here from the 0.25° output. The "share" is the coefficient times that term's mean torque.</p>
  <p>The residual is mostly a constant ({num(it['intercept']['coef'], 1)} Hadley), not a scaled copy of any range's torque: the largest
  significant range coefficients are a few per cent and of both signs, and together with friction and gravity-wave drag they account
  for {num(sum(it[k]['share_of_mean_residual'] for k, _ in reg_terms), 1)} Hadley of the mean. A constant offset between the
  model's own mountain torque on its N320 grid and the torque recomputed from 0.25° output is plausible, since the mean mountain
  torque changes by tens of Hadley between 0.25° and 1° (section 9), and so is non-conservation in the semi-Lagrangian dynamics; the
  vertical integrals cannot tell them apart. Nor can they give the latitude structure of this in-window residual: a band's budget also
  contains the transport of angular momentum across its edges, which needs the 3-D wind. Increments and initial tendencies have long
  been read as diagnostics of model error in this way (Klinker and Sardeshmukh, 1992; Rodwell and Palmer, 2007).</p>

  <h3>The forecast side</h3>
  {fig("fcdrift", "fc_drift.webp", "AIFS-ENS ensemble-mean angular momentum minus the verifying analysis by lead day, globally and by hemisphere", f"AIFS-ENS ensemble-mean relative angular momentum (14 levels) minus the same calculation for the verifying analysis, by lead, with 95% weekly-block bootstrap intervals; filled where the interval excludes zero. {finits[2]} inits, of which {nver[0]}–{nver[1]} can be verified at each lead.")}
  <p>A physics model whose torques drain angular momentum should lose it in free forecasts, and IFS-ENS does: in the site's winter
  2025–26 test it loses angular momentum steadily, −0.31 ×10²⁵ kg m² s⁻¹ by day 15, while closing its own torque budget almost exactly
  (<a href="/aifs-aam-budget.html">Do AI weather models conserve angular momentum?</a>): its torques take the momentum out. That is the
  forecast-side counterpart of ERA5's positive increments. The data-driven AIFS-ENS, trained on analyses that already contain the
  increments' correction, shows no global drain in this site's archive ({nver[0]} to {nver[1]} forecasts per lead, inits from 20 July
  to 11 October 2026): {E(fg1)} ×10²⁵ at day 1, then about −0.03 and not significant through day 15 ({E(fg15)}). It redistributes
  angular momentum between the hemispheres instead: the Northern Hemisphere gains ({E(fn15)} at day 15, significant from day
  {nh_first}) and the Southern loses ({E(fs7)} at day 7, significant at {sh_sig} of the 13 leads from day 3; {E(fs15)} at day 15). The
  comparison is not like for like: different seasons (December–February against July–October), different IFS cycles (ERA5 is cycle
  41r2), and an increment is not a free-forecast drift.</p>

  <h2>8. What closes the budget and what does not</h2>
  <p>Two choices matter for closure. Sampling: the summed torques and the tendency correlate at {cl['1']['corr_dMdt_T']:.2f} hour by
  hour, because the increments dominate single hours, and at {cl['24']['corr_dMdt_T']:.2f} in steps of 24 hours. The mass term: without
  it the 24-hour residual has an sd of {cl['24']['sd_Rmotion_only']:.1f} Hadley instead of {cl['24']['sd_R']:.1f}, because the tidal
  pressure waves that drive the mountain torque within a day are balanced by the mass term, not by the winds.</p>
  {t_step}
  <p class="tcap">Each row samples the hourly 2026 fields every 1, 3, 6, 12 or 24 hours and closes the budget over those steps, so the
  24-hour row is not the same estimator as the daily means of the hourly residual in section 6.</p>
  <p>For the 2026 window, sampling more often and adding the mass term make the closure worse rather than better. The once-daily
  12Z wind-term budget (the version first used for 2026) leaves {num(v12['residual'], 2)}; adding the mass term raises the angular
  momentum change by {num(v12m['dM'] - v12['dM'], 2)}, and four- or 24-times-daily torques lower the torque integral by about
  {abs(v4['torque'] - v12['torque']):.1f}. Absolute budgets at one and three hours agree with each other to
  {abs(va1['residual'] - va3['residual']):.1f}, and both miss by about {num(va1['residual_mean_hadley'], 0)} Hadley on average.</p>
  {t_var}
  <p><b>The persistent residual, and the −4.5 Hadley figure.</b> Averaged over 1991–2020 at four times a day, ERA5's torques sum to
  {num(fsum, 1)} Hadley while angular momentum has no trend; hourly in 2026 the residual is {num(cl['1']['mean_R'], 1)} Hadley. Until
  12 October 2026 the site's torque products (<code>build_torque_clim.py</code> and the
  <a href="/topics/mountain-torque.html">mountain torque explainer</a>) quoted an annual-mean non-closure of −4.5 ± 0.7 Hadley, from
  a sample at 12 UTC only. At that hour the semidiurnal tide adds about +4 Hadley to the mountain torque and the friction torque is
  far weaker than at 06 UTC, when the daytime boundary layer over Asia is at its strongest:</p>
  {t_hours}
  <p>The imbalance of ERA5's resolved torques is therefore about two and a half times the figure once quoted. It sits in the mean and
  cancels in anomalies, so the anomaly products were never affected.</p>

  <h2>9. Caveats</h2>
  <ul>
    <li><b>The mean mountain torque is not resolution-converged.</b> Averaging orography and surface pressure together to 1° or
    2.5° moves the global mean from about +1 to −26 and −33 Hadley and the Andes box from −223 to −148, while the anomalies
    correlate at 0.96 or better with the 0.25° values and their ENSO regressions barely change (global +1.00, +1.07 and +0.99 Hadley
    per kelvin). Anomalies from CMIP6's 1–2.5° grids are therefore comparable with ERA5's; absolute values are not. The Andes DJF
    standing shift at 2.5° is +2.1 Hadley ({pval(0.009)}), short of the FDR.</li>
    <li><b>Tide aliasing.</b> Twice-daily sampling at 00 and 12 UTC does not remove the semidiurnal tide; four times a day does,
    and hourly removes the rest (the four-versus-24 offsets are below 0.2 Hadley).</li>
    <li><b>What CMIP6's <code>tauu</code> is.</b> It is the surface (boundary-layer) stress. The Met Office models state that it
    includes turbulent form drag from small hills; parameterised orographic gravity-wave drag is reported separately
    (<code>utendogw</code>, in {n_gwd} models). ERA5's turbulent stress likewise includes turbulent orographic form drag. Because the
    gravity-wave-drag contribution to the El Niño rise is not significant in either data set, the answer does not depend on where it
    is booked.</li>
    <li><b>ERA5's torques drift.</b> The global mountain and friction torques trend by +0.5 and +0.8 Hadley per decade over
    1979–2025 with no matching trend in angular momentum, so they are detrended for the ENSO analysis; the 2026 attribution is
    shown both ways.</li>
    <li><b>ENSO phase</b> is the ONI of each season against fixed thresholds, not CPC's rule of five overlapping seasons, and the
    CMIP6 models use ±0.5 K on their own index whatever the size of their ENSO.</li>
    <li><b>Model independence.</b> Several CMIP6 models share code (four CESM2 variants, NorESM2 and others built on CAM, three MPI
    models), so "{n_models} models" overstates the independent evidence; a one-model-per-centre check was not made.</li>
    <li><b>Samples.</b> Super El Niños (ONI ≥ 2.0; three DJFs) are too few to test and are not used as evidence. The zonal-mean jet
    comes from the local ERA5 store, northern hemisphere 1979–2023 and southern 1991–2020. The torque boxes are the site's and do
    not close the terrain at their edges, so a box's mean includes part of the large-scale pressure gradient over high ground.</li>
  </ul>

  <h2>10. Data and code</h2>
  <ul>
    <li>Daily ERA5 torques and angular momentum, 1959–2026: <a href="{GH}scripts/mjo/data/reference/torque_daily_era5.csv.gz">torque_daily_era5.csv.gz</a>;
    hourly 2026: <a href="{GH}scripts/mjo/data/reference/torque_1h_2026.csv.gz">torque_1h_2026.csv.gz</a>; both written by
    <a href="{GH}scripts/mjo/src/torque_history_era5.py">torque_history_era5.py</a> in the
    <a href="{GH}.github/workflows/torque-history.yml">torque-history</a> workflow.</li>
    <li>CMIP6 monthly table: <a href="{GH}scripts/mjo/data/reference/torque_cmip6_monthly.csv.gz">torque_cmip6_monthly.csv.gz</a>
    from <a href="{GH}scripts/mjo/src/torque_history_cmip6.py">torque_history_cmip6.py</a> in the
    <a href="{GH}.github/workflows/torque-cmip6.yml">torque-cmip6</a> workflow (the daily range torques are a run artifact).</li>
    <li>Analysis: <a href="{GH}scripts/mjo/src/torque_enso_study.py">torque_enso_study.py</a> and
    <a href="{GH}scripts/mjo/src/torque_budget_2026.py">torque_budget_2026.py</a>; GFZ by ENSO phase:
    <a href="{GH}scripts/mjo/src/torque_gfz_enso.py">torque_gfz_enso.py</a>; results:
    <a href="{GH}scripts/mjo/data/reference/torque_enso_study">torque_enso_study/</a>; figures
    <a href="{GH}scripts/mjo/src/torque_enso_figs.py">torque_enso_figs.py</a>; this page
    <a href="{GH}scripts/mjo/src/torque_enso_page.py">torque_enso_page.py</a>.</li>
    <li>Increments: 5° band jumps from <code>--bands5</code>, 37-level jumps
    (<a href="{GH}scripts/mjo/src/torque_increments_u37.py">torque_increments_u37.py</a>,
    <a href="{GH}scripts/mjo/data/reference/torque_u37_jumps_2025_2026.nc">torque_u37_jumps_2025_2026.nc</a>) and the low-level maps
    (<a href="{GH}scripts/mjo/src/torque_increments_maps.py">torque_increments_maps.py</a>,
    <a href="{GH}scripts/mjo/data/reference/torque_u37maps_2025_2026.nc">torque_u37maps_2025_2026.nc</a>), all in the torque-history
    workflow; analysis <a href="{GH}scripts/mjo/src/torque_increments.py">torque_increments.py</a>.</li>
  </ul>

  <h2>References</h2>
  <ol class="refs">
    <li>Benjamini, Y., and Y. Hochberg, 1995: Controlling the false discovery rate: a practical and powerful approach to multiple testing. <i>J. R. Stat. Soc. B</i>, 57, 289–300.</li>
    <li>Bretherton, C. S., M. Widmann, V. P. Dymnikov, J. M. Wallace and I. Bladé, 1999: The effective number of spatial degrees of freedom of a time-varying field. <i>J. Climate</i>, 12, 1990–2009.</li>
    <li>Brown, A. R., 2004: Resolution dependence of orographic torques. ECMWF Technical Memorandum 433 (submitted to <i>Q. J. R. Meteorol. Soc.</i>). <a href="https://www.ecmwf.int/sites/default/files/elibrary/2004/8400-resolution-dependence-orographic-torques.pdf">ECMWF</a></li>
    <li>Colle, B. A., and C. F. Mass, 1995: The structure and evolution of cold surges east of the Rocky Mountains. <i>Mon. Wea. Rev.</i>, 123, 2577–2610.</li>
    <li>Dobslaw, H., R. Dill, A. Grötzsch, A. Brzeziński and M. Thomas, 2010: Seasonal polar motion excitation from numerical models of atmosphere, ocean, and continental hydrosphere. <i>J. Geophys. Res.</i>, 115, B10406. <a href="https://doi.org/10.1029/2009JB007127">doi:10.1029/2009JB007127</a></li>
    <li>Duchon, C. E., 1979: Lanczos filtering in one and two dimensions. <i>J. Appl. Meteor.</i>, 18, 1016–1022.</li>
    <li>Egger, J., K. Weickmann and K.-P. Hoinka, 2007: Angular momentum in the global atmospheric circulation. <i>Rev. Geophys.</i>, 45, RG4007. <a href="https://doi.org/10.1029/2006RG000213">doi:10.1029/2006RG000213</a></li>
    <li>Eyring, V., and co-authors, 2016: Overview of the Coupled Model Intercomparison Project Phase 6 (CMIP6) experimental design and organization. <i>Geosci. Model Dev.</i>, 9, 1937–1958.</li>
    <li>Garreaud, R. D., 2000: Cold air incursions over subtropical South America: mean structure and dynamics. <i>Mon. Wea. Rev.</i>, 128, 2544–2559.</li>
    <li>Gerber, E. P., and E. Manzini, 2016: The Dynamics and Variability Model Intercomparison Project (DynVarMIP) for CMIP6. <i>Geosci. Model Dev.</i>, 9, 3413–3425.</li>
    <li>Hersbach, H., and co-authors, 2020: The ERA5 global reanalysis. <i>Q. J. R. Meteorol. Soc.</i>, 146, 1999–2049. <a href="https://doi.org/10.1002/qj.3803">doi:10.1002/qj.3803</a></li>
    <li>Huang, H.-P., P. D. Sardeshmukh and K. M. Weickmann, 1999: The balance of global angular momentum in a long-term atmospheric data set. <i>J. Geophys. Res.</i>, 104, 2031–2040. <a href="https://doi.org/10.1029/1998JD200068">doi:10.1029/1998JD200068</a></li>
    <li>Iskenderian, H., and D. A. Salstein, 1998: Regional sources of mountain torque variability and high-frequency fluctuations in atmospheric angular momentum. <i>Mon. Wea. Rev.</i>, 126, 1681–1694.</li>
    <li>Kanehama, T., I. Sandu, A. Beljaars, A. van Niekerk, N. Wedi, S. Boussetta, S. Lang, S. Johnson and L. Magnusson, 2022: Evaluation and optimization of orographic drag in the IFS. ECMWF Technical Memorandum 893. <a href="https://www.ecmwf.int/sites/default/files/elibrary/2022/20333-evaluation-and-optimizaton-orographic-drag-ifs.pdf">ECMWF</a></li>
    <li>Klinker, E., and P. D. Sardeshmukh, 1992: The diagnosis of mechanical dissipation in the atmosphere from large-scale balance requirements. <i>J. Atmos. Sci.</i>, 49, 608–627.</li>
    <li>Lott, F., A. W. Robertson and M. Ghil, 2004: Mountain torques and Northern Hemisphere low-frequency variability. Part I: Hemispheric aspects. <i>J. Atmos. Sci.</i>, 61, 1259–1271.</li>
    <li>van Niekerk, A., and T. G. Shepherd, 2016: Sensitivity of resolved and parameterized surface drag to changes in resolution and parameterization. ECMWF workshop presentation. <a href="https://www.ecmwf.int/sites/default/files/elibrary/2016/16619-sensitivity-resolved-and-parametrized-surface-drag-changes-resolution-and-parametrization.pdf">ECMWF</a></li>
    <li>van Niekerk, A., I. Sandu, A. Zadra, E. Bazile, T. Kanehama, M. Köhler, M.-S. Koo, H.-J. Choi, Y. Kuroki, M. D. Toy, S. B. Vosper and V. Yudin, 2020: COnstraining ORographic Drag Effects (COORDE): a model comparison of resolved and parametrized orographic drag. <i>J. Adv. Model. Earth Syst.</i>, 12, e2020MS002160. <a href="https://doi.org/10.1029/2020MS002160">doi:10.1029/2020MS002160</a></li>
    <li>Rodwell, M. J., and T. N. Palmer, 2007: Using numerical weather prediction to assess climate models. <i>Q. J. R. Meteorol. Soc.</i>, 133, 129–146.</li>
    <li>Rosen, R. D., D. A. Salstein, T. M. Eubanks, J. O. Dickey and J. A. Steppe, 1984: An El Niño signal in atmospheric angular momentum and Earth rotation. <i>Science</i>, 225, 411–414.</li>
    <li>Sandu, I., P. Bechtold, A. Beljaars, A. Bozzo, F. Pithan, T. G. Shepherd and A. Zadra, 2016: Impacts of parameterized orographic drag on the Northern Hemisphere winter circulation. <i>J. Adv. Model. Earth Syst.</i>, 8, 196–211. <a href="https://doi.org/10.1002/2015MS000564">doi:10.1002/2015MS000564</a></li>
    <li>Seager, R., N. Harnik, Y. Kushnir, W. Robinson and J. Miller, 2003: Mechanisms of hemispherically symmetric climate variability. <i>J. Climate</i>, 16, 2960–2978.</li>
    <li>Weickmann, K. M., G. N. Kiladis and P. D. Sardeshmukh, 1997: The dynamics of intraseasonal atmospheric angular momentum oscillations. <i>J. Atmos. Sci.</i>, 54, 1445–1461.</li>
    <li>Wilks, D. S., 2016: "The stippling shows statistically significant grid points": how research results are routinely overstated and overinterpreted, and what to do about it. <i>Bull. Amer. Meteor. Soc.</i>, 97, 2263–2273.</li>
  </ol>

  <h2>Data credits</h2>
  <p class="credits">ERA5 (Hersbach et al., 2020), Copernicus Climate Change Service (C3S) / ECMWF, via the ARCO-ERA5 store on
  Google Cloud; contains modified Copernicus Climate Change Service information 2026. Neither ECMWF nor the European Commission is
  responsible for any use of this information. CMIP6 output from the modelling centres via the Pangeo CMIP6 mirror on Google
  Cloud (WCRP CMIP6, CC BY 4.0 where so licensed). ONI and ERSSTv6 Niño-3.4: NOAA CPC and PSL. Effective angular momentum
  functions: GFZ Potsdam (ESMGFZ). About 400 GB of ERA5 and 450 GB of CMIP6 fields were read, in GitHub Actions.</p>
  <p class="byline">By <a href="/about.html">Shawn Corvec</a>, meteorologist. Published 13 October 2026. A static study: it is not
  updated.</p>
</article>
"""
    ld = {"@context": "https://schema.org", "@type": "ScholarlyArticle", "headline": H1, "name": TITLE, "description": DESC,
          "url": URL, "inLanguage": "en", "datePublished": "2026-10-13", "dateModified": "2026-10-13",
          "author": {"@id": "https://scorvec.com/#person"}, "image": [f"https://scorvec.com{IMG}/q1_shift_vs_events.webp"],
          "keywords": "El Niño, ENSO, mountain torque, atmospheric angular momentum, length of day, analysis increments, orographic drag, Andes, Rockies, Himalaya, friction torque, ERA5, CMIP6"}
    aam = site / "aifs-aam-budget.html"
    style = aam.read_text().split("<style>", 1)[1].split("</style>", 1)[0]
    style += "\n  .study code { font-size: 0.9em; }\n"
    lb = aam.read_text().split("</main>", 1)[1].split("<!-- sf:start -->", 1)[0].strip()
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{TITLE} · Shawn Corvec</title>
<meta name="description" content="{DESC}">
<link rel="canonical" href="{URL}">
<link rel="icon" href="/favicon.svg" type="image/svg+xml">
<meta name="author" content="Shawn Corvec">
<meta property="og:type" content="article">
<meta property="og:title" content="{H1}">
<meta property="og:description" content="{DESC}">
<meta property="og:url" content="{URL}">
<meta property="og:image" content="https://scorvec.com{IMG}/q1_shift_vs_events.webp">
<meta property="og:site_name" content="Shawn Corvec">
<meta name="twitter:card" content="summary_large_image">
<script type="application/ld+json">
{json.dumps(ld, indent=1, ensure_ascii=False)}
</script>
<script data-goatcounter="https://scorvec.goatcounter.com/count" async src="//gc.zgo.at/count.js"></script>
<link rel="stylesheet" href="/assets/site.css">
<script src="/assets/site.js" defer></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body,{{delimiters:[{{left:'$$',right:'$$',display:true}},{{left:'$',right:'$',display:false}}],throwOnError:false}});"></script>
<style>{style}</style>
</head>
<body>
<main>
{body}
</main>
{lb}
</body>
</html>
"""
    out = site / "torque-enso.html"
    old = out.read_text() if out.exists() else ""
    if "<!-- sh:start -->" in old:                       # keep the stamped chrome; apply_chrome re-stamps it anyway
        page = page.replace("<body>\n", "<body>\n" + old.split("<body>\n", 1)[1].split("<!-- sh:end -->", 1)[0] + "<!-- sh:end -->\n", 1)
    out.write_text(page)
    print("wrote", out)
    sys.path.insert(0, str(site / "scripts" / "site"))
    import apply_chrome                                                  # noqa: E402  (footer, rail.js, menus)
    apply_chrome.stamp_all(only={"torque-enso.html"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
