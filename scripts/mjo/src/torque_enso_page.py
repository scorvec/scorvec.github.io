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
import torque_enso_study as S

D = S.REF / "torque_enso_study"
IMG = "/studies/torque-enso"
URL = "https://scorvec.com/torque-enso.html"
GH = "https://github.com/scorvec/scorvec.github.io/blob/main/"
TITLE = "El Niño and angular momentum"
H1 = "Why the atmosphere spins up in El Niño"
DESC = (
    "Where El Niño's extra angular momentum comes from: a standing shift in the Andes mountain torque, "
    "and why ERA5's budget needs its analysis increments to close."
)


def num(x, d=1, sign=True):
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    if float(s) == 0:  # no "+0.0" or "−0.0"
        s = f"{0:.{d}f}"
    return s.replace("-", "−")


def day_txt(s, year=True):
    return pd.Timestamp(s).strftime("%-d %B %Y" if year else "%-d %B")


def month_txt(s):
    return pd.Timestamp(s + "-01").strftime("%B %Y")


RSHORT = {
    "HIM": "Himalaya/Tibet",
    "ROC": "Rockies",
    "AND": "Andes",
    "GRL": "Greenland",
    "ANT": "Antarctica",
    "ALP": "Alps",
}


def dd(lag, before=False):
    n = abs(int(lag))
    return f"{n} day{'' if n == 1 else 's'}" + (" before" if before else "")


LAYER = {
    "stratosphere (<100 hPa)": "stratosphere (above 100 hPa)",
    "upper troposphere (100-400 hPa)": "upper troposphere (100–400 hPa)",
    "mid troposphere (400-850 hPa)": "middle troposphere (400–850 hPa)",
    "lowest layers (>=850 hPa)": "below 850 hPa",
}


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

    nsig = {
        mt: int(comp[(comp.cmp == "EN-NEU") & (comp.metric == mt)].sig.sum())
        for mt in ("n_pos", "n_neg", "amp_pos", "amp_neg", "hp_sd", "evc_pos", "lp_mean")
    }
    n_ev = nsig["n_pos"] + nsig["amp_pos"] + nsig["hp_sd"]  # EN-NEU event count, amplitude, variance
    n_ev_txt = "none" if n_ev == 0 else str(n_ev)
    nsy = R["q1"]["n_season_years"]
    q2 = R["q2_era5"]
    ce = q2["contributions_lags_m6_m1"]
    tend = q2["tendency_column"]
    n_models = R["q2_cmip6"]["n_models"]
    n_members = R["q2_cmip6"]["n_members"]
    n_gwd = R["q2_cmip6"]["n_models_with_gwd"]
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
    hours = {
        h: {t: float(base[f"{t}_glob_h{h}"].mean()) for t in ("mtn", "fric", "gwd")} for h in ("00", "06", "12", "18")
    }
    four = {t: float(base[f"{t}_glob"].mean()) for t in ("mtn", "fric", "gwd")}
    hsum = {h: sum(v.values()) for h, v in hours.items()}
    fsum = sum(four.values())

    # ---------------- result-1 numbers
    lpA = c1("AND", "DJF", "lp_mean")
    lpH = c1("HIM", "MAM", "lp_mean")
    lpR = c1("ROC", "DJF", "lp_mean")
    npA = c1("AND", "DJF", "n_pos")
    amA = c1("AND", "DJF", "amp_pos")
    slA = c1("AND", "DJF", "lp_mean", "ONI-slope")
    slAJ = c1("AND", "JJA", "lp_mean", "ONI-slope")
    slR = c1("ROC", "DJF", "lp_mean", "ONI-slope")
    slRM = c1("ROC", "MAM", "lp_mean", "ONI-slope")
    slH = c1("HIM", "MAM", "lp_mean", "ONI-slope")
    slHS = c1("HIM", "SON", "lp_mean", "ONI-slope")
    nslope = comp[(comp.cmp == "ONI-slope") & (comp.metric == "n_pos") & comp.sig]
    nxA = c1("AND", "DJF", "nx_pos")
    sA = sh["AND"]["DJF"]
    sH = sh["HIM"]["MAM"]
    sG = sh["GRL"]["SON"]
    cA = cmr("AND", "DJF", "lp_mean")
    cRM = cmr("ROC", "MAM", "lp_mean")
    cRJ = cmr("ROC", "JJA", "lp_mean")
    cHM = cmr("HIM", "MAM", "lp_mean")
    cHS = cmr("HIM", "SON", "lp_mean")
    cRD = cmr("ROC", "DJF", "lp_mean")
    cnA = cmr("AND", "DJF", "nx_pos")
    cnR = cmr("ROC", "DJF", "nx_pos")
    dA = dec[(dec.range == "AND") & (dec.season == "DJF")].iloc[0]
    dR = dec[(dec.range == "ROC") & (dec.season == "DJF")].iloc[0]
    dH = dec[(dec.range == "HIM") & (dec.season == "MAM")].iloc[0]

    def dec_row(lab, d, c):
        return (
            f"<tr><th scope='row'>{lab}</th><td>{num(d.d_total, 2)}</td><td>{num(d.d_lowpass, 2)}</td>"
            f"<td>{num(d.d_highpass, 2)}</td><td class='ns'>{num(d.d_event_pos, 2)} n.s.</td>"
            f"<td>{num(c['median'], 2)} <span class='sig'>({c.agree:.0%})</span></td></tr>"
        )

    t_dec = (
        "<div class='tw'><table class='res'><thead><tr><th>El Niño minus neutral (Hadley)</th><th>seasonal mean</th>"
        "<th>standing part (≥30 d)</th><th>synoptic part (&lt;30 d)</th><th>event windows</th>"
        "<th>CMIP6 standing part</th></tr></thead><tbody>"
        + dec_row("Andes, DJF", dA, cA)
        + dec_row("Rockies, DJF", dR, cRD)
        + dec_row("Himalaya/Tibet, MAM", dH, cHM)
        + "</tbody></table></div>"
    )

    # ---------------- result-2 table (ERA5 vs CMIP6 contributions)
    def crow(lab, ke, kc):
        e = ce.get(ke)
        ecell = (
            f"<td>{num(e['est'], 2)} <span class='sig'>{ci(e['lo'], e['hi'], 2)}</span></td>"
            if e and e["fdr"]
            else (f"<td class='ns'>{num(e['est'], 2)} n.s.</td>" if e else "<td></td>")
        )
        if kc and kc in cc.index:
            c = cc.loc[kc]
            ccell = (
                f"<td>{num(c['median'], 2)} <span class='sig'>({c.agree:.0%} agree)</span></td>"
                if c.robust
                else f"<td class='ns'>{num(c['median'], 2)} not robust</td>"
            )
        else:
            ccell = "<td></td>"
        return f"<tr><th scope='row'>{lab}</th>{ecell}{ccell}</tr>"

    rows2 = [
        ("AAM tendency dM/dt", tend, "dmdt_glob"),
        ("sum of the torques", "T_glob", "T_mf_glob"),
        ("mountain torque", "mtn_glob", "mtn_glob"),
        ("friction (ERA5 turbulent stress; CMIP6 tauu)", "fric_glob", "fric_glob"),
        ("gravity-wave drag (CMIP6: 13 models)", "gwd_glob", "gwd_glob"),
        ("ERA5: dM/dt minus torques", "R_glob", None),
        ("Andes", "mtn5_AND", "mtn_AND"),
        ("Rockies", "mtn5_ROC", "mtn_ROC"),
        ("Himalaya/Tibet", "mtn5_HIM", "mtn_HIM"),
        ("Greenland", "mtn5_GRL", "mtn_GRL"),
        ("Antarctica", "mtn5_ANT", "mtn_ANT"),
        ("Alps", "mtn5_ALP", "mtn_ALP"),
        ("mountain, 30S–0", "mtn_b3000s", "mtn_b3000s"),
        ("mountain, 0–30N", "mtn_b0030n", "mtn_b0030n"),
        ("mountain, 30–60N", "mtn_b3060n", "mtn_b3060n"),
        ("friction, 30S–0", "fric_b3000s", "fric_b3000s"),
        ("friction, 0–30N", "fric_b0030n", "fric_b0030n"),
        ("friction, 30–60N", "fric_b3060n", "fric_b3060n"),
        ("friction, 30–60S", "fric_b6030s", "fric_b6030s"),
    ]
    t_q2 = (
        "<div class='tw'><table class='res'><thead><tr><th>10²⁵ kg m² s⁻¹ per K of Niño-3.4</th><th>ERA5 1979–2025 [95% CI]</th>"
        f"<th>CMIP6, {n_models} models: median</th></tr></thead><tbody>"
        + "".join(crow(*x) for x in rows2)
        + "</tbody></table></div>"
    )
    mt, fr = ce["mtn_glob"], ce["fric_glob"]
    cmt, cfr, cdm, cT = cc.loc["mtn_glob"], cc.loc["fric_glob"], cc.loc["dmdt_glob"], cc.loc["T_mf_glob"]
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

    pre_lo = [
        llpeak(r, se, "M", range(-7, 0)) for r, se in (("HIM", "DJF"), ("ROC", "DJF"), ("AND", "DJF"), ("AND", "JJA"))
    ]
    post_hi = [
        llpeak(r, se, "M", range(1, 6)) for r, se in (("HIM", "DJF"), ("ROC", "DJF"), ("AND", "DJF"), ("AND", "JJA"))
    ]
    jR = llpeak("ROC", "DJF", "u200jet", range(8))
    jA = llpeak("AND", "DJF", "u200jet", range(8))
    jbA = llpeak("AND", "DJF", "u200jet", range(-5, 0))
    q3cases = ll[
        ll.series.isin(["dMdt", "M", "u200jet"]) & (ll.range + ll.season).isin(["HIMDJF", "ROCDJF", "ANDDJF", "ANDJJA"])
    ]
    neff_lo, neff_hi = (int(round(q3cases.neff.min(), -1)), int(round(q3cases.neff.max(), -1)))
    qc = R["q3_composite_counts"]
    pt_any = bool(pt.sig.any())

    # ---------------- 2026
    wt = B["window_test"]
    inc = B["increment_per_window_1e25"]
    dfh = B["daily_from_hourly"]
    g = {int(x["time"]): x for x in B["hourly_residual_by_end_hour"]}
    t3 = B["three_hourly_window_test"]
    wv = B["window_variants"]
    v12 = wv["12Z daily, motion only, anomaly"]
    v12m = wv["12Z daily, motion+mass, anomaly"]
    v4 = wv["4x daily, motion+mass, anomaly"]
    vh = wv["hourly daily-means, motion+mass, anomaly vs 4x clim"]
    va1 = wv["1-hourly, motion+mass, absolute"]
    va3 = wv["3-hourly, motion+mass, absolute"]
    hw = B["history_window_residual"]
    gfz = B["gfz"]
    att = B["attribution"]
    cl = B["closure_by_step"]
    zr = (vh["residual"] - hw["mean"]) / hw["sd"]
    oni = S.load_oni()
    oni26 = {m: float(oni.get(pd.Timestamp(f"2026-{m:02d}-01"), np.nan)) for m in (4, 5, 6, 7, 8)}

    def vrow(lab, v):
        return (
            f"<tr><th scope='row'>{lab}</th><td>{num(v['torque'], 2)}</td><td>{num(v['dM'], 2)}</td>"
            f"<td>{num(v['residual'], 2)}</td><td>{num(v['residual_mean_hadley'], 1)}</td></tr>"
        )

    t_var = (
        "<div class='tw'><table class='res'><thead><tr><th>15 Apr – 19 Sep 2026</th><th>∫ torques</th><th>ΔM</th>"
        "<th>residual</th><th>residual, mean Hadley</th></tr></thead><tbody>"
        + vrow("12Z once daily, motion term only, anomaly", v12)
        + vrow("12Z, motion + mass, anomaly", v12m)
        + vrow("00/06/12/18Z, motion + mass, anomaly", v4)
        + vrow("hourly daily means, motion + mass, anomaly", vh)
        + vrow("3-hourly, motion + mass, absolute", va3)
        + vrow("hourly, motion + mass, absolute", va1)
        + "</tbody></table></div>"
    )

    def srow(st, lab):
        c = cl[str(st)]
        return (
            f"<tr><th scope='row'>{lab}</th><td>{c['corr_dMdt_T']:.2f}</td><td>{num(c['mean_R'], 1)}</td>"
            f"<td>{c['sd_R']:.1f}</td><td>{c['sd_Rmotion_only']:.1f}</td></tr>"
        )

    t_step = (
        "<div class='tw'><table class='res'><thead><tr><th>ERA5 2026, sampled every</th><th>corr(dM/dt, torques)</th>"
        "<th>mean residual</th><th>residual sd</th><th>sd, motion term only</th></tr></thead><tbody>"
        + srow(1, "1 hour")
        + srow(3, "3 hours")
        + srow(6, "6 hours")
        + srow(12, "12 hours")
        + srow(24, "24 hours")
        + "</tbody></table></div>"
    )
    t_hours = (
        "<div class='tw'><table class='res'><thead><tr><th>ERA5 1991–2020 mean (Hadley)</th><th>00Z</th><th>06Z</th>"
        "<th>12Z</th><th>18Z</th><th>mean of the four</th></tr></thead><tbody>"
        + "".join(
            f"<tr><th scope='row'>{lab}</th>"
            + "".join(f"<td>{num(hours[h][t], 1)}</td>" for h in hours)
            + f"<td>{num(four[t], 1)}</td></tr>"
            for t, lab in (("mtn", "mountain"), ("fric", "friction"), ("gwd", "gravity-wave drag"))
        )
        + "<tr class='diff'><th scope='row'>sum (should be ≈ 0)</th>"
        + "".join(f"<td>{num(hsum[h], 1)}</td>" for h in hours)
        + f"<td>{num(fsum, 1)}</td></tr></tbody></table></div>"
    )
    labs = {
        "Andes": "Andes",
        "other ranges": "the other five ranges",
        "rest of mountain": "mountain outside the six boxes",
        "friction tropics": "friction, 30S–30N",
        "friction NH ext": "friction, north of 30N",
        "friction SH ext": "friction, south of 30S",
        "GWD": "gravity-wave drag",
    }
    t_att = (
        "<div class='tw'><table class='res'><thead><tr><th>15 Apr – 19 Sep, 10²⁵ kg m² s⁻¹</th><th>2026</th>"
        "<th>trend removed</th><th>1979–2025 sd</th><th>z</th><th>rank of 48</th><th>El Niño onsets, mean</th></tr></thead><tbody>"
        + "".join(
            f"<tr><th scope='row'>{labs[k]}</th><td>{num(v['v2026'], 1)}</td><td>{num(v['v2026_detrended'], 1)}</td>"
            f"<td>{v['hist_sd']:.1f}</td><td>{num(v['z'], 1)}</td><td>{v['rank_of_48']}</td>"
            f"<td>{num(v['en_onset_mean'], 1)}</td></tr>"
            for k, v in att.items()
        )
        + f"<tr class='diff'><th scope='row'>sum of the torques</th><td>{num(B['attribution_total_torque'], 1)}</td><td></td><td></td><td></td><td></td><td></td></tr>"
        + f"<tr class='diff'><th scope='row'>AAM anomaly change (motion + mass)</th><td>{num(B['attribution_dM_hourly_anomaly'], 1)}</td><td></td><td></td><td></td><td></td><td></td></tr>"
        + "</tbody></table></div>"
    )

    boxes = (
        "<div class='tw'><table class='res'><thead><tr><th>range</th><th>longitude</th><th>latitude</th></tr></thead><tbody>"
        "<tr><th scope='row'>Himalaya/Tibet</th><td>70–105°E</td><td>25–45°N</td></tr>"
        "<tr><th scope='row'>Rockies</th><td>130–100°W</td><td>30–60°N</td></tr>"
        "<tr><th scope='row'>Andes</th><td>80–65°W</td><td>55°S–10°N</td></tr>"
        "<tr><th scope='row'>Greenland</th><td>60–15°W</td><td>60–85°N</td></tr>"
        "<tr><th scope='row'>Antarctica</th><td>all</td><td>south of 65°S</td></tr>"
        "<tr><th scope='row'>Alps</th><td>0–45°E</td><td>35–50°N</td></tr></tbody></table></div>"
    )

    # ---------------- figure numbering (in order of appearance)
    FIGS = [
        "gfz",
        "q2lag",
        "q2con",
        "q1grid",
        "q1thr",
        "slp",
        "q3ll",
        "q3z",
        "b26cum",
        "b26att",
        "b26hr",
        "incprof",
        "incseason",
        "incsection",
        "mapsu",
        "mapsseason",
        "mapssplit",
        "fcdrift",
    ]
    FN = {k: i + 1 for i, k in enumerate(FIGS)}

    def fig(key, f, alt, cap):
        n = FN[key]
        return (
            f"<figure id='fig{n}'><a href='{IMG}/{f}' class='zoom'><img src='{IMG}/{f}' alt=\"{alt}\" loading='lazy'></a>"
            f"<figcaption><b>Figure {n}.</b> {cap}</figcaption></figure>"
        )

    def ordn(n):
        return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"

    def rank_word(n):
        return {2: "second", 3: "third", 4: "fourth", 5: "fifth"}.get(n, ordn(n))

    def rank_txt(n):
        top_ = 49 - n
        return (
            f"{'second-largest' if top_ == 2 else ordn(top_) + ' largest'} of 48 years"
            if top_ <= n
            else f"{ordn(n)} smallest of 48 years"
        )

    # ---------------- increments (torque_increments.py) and the low-level maps
    I = json.loads((D / "increments.json").read_text())
    P = I["profile"]
    g10, g22 = P["10"]["global"], P["22"]["global"]
    w10, w22, m10, m22, T10 = (
        P["10r"]["global"],
        P["22r"]["global"],
        P["10m"]["global"],
        P["22m"]["global"],
        P["10T"]["global"],
    )
    bl = I["belts"]
    gm = I["global_by_month"]
    gfull = {k: v for k, v in gm.items() if v["n"] >= 20}
    gmin = min(gfull, key=lambda k: gfull[k]["mean"])
    gmax = max(gfull, key=lambda k: gfull[k]["mean"])
    u10, u22 = I["u37_J10"], I["u37_J22"]
    L10, L22 = u10["layers"], u22["layers"]
    fsec = I.get("u37_frac_sig_with_data", {})
    ir = I["inwindow_regression"]
    it = ir["terms"]
    fd = I["fc_drift"]["by_region"]
    finits = I["fc_drift"]["inits"]
    MP = json.loads((D / "increments_maps.json").read_text())
    sp_ = MP["split"]
    area = MP["area"]
    WS = json.loads((D / "inc_wind_sign.json").read_text())  # torque_increments.py --wind-sign
    wsb = WS["background"]

    def ws_cell(r):
        f = r["strengthen"]
        if f is None or r["cells"] == 0:
            return "<td>n.s.</td>"
        word = "strengthens" if f >= 0.67 else ("weakens" if f <= 0.33 else "mixed")
        return f"<td>{word} ({100 * f:.0f}%)</td>"

    ws_rows = "".join(
        f"<tr><th scope='row'>{lv}</th>{ws_cell(WS['zonal'][lv]['trop'])}{ws_cell(WS['zonal'][lv]['extra'])}</tr>"
        for lv in ("1000", "975", "950", "925", "900", "875", "850", "800", "750", "700")
    )
    t_sign = (
        "<div class='tw'><table class='res'><thead><tr><th>hPa</th><th>tropics, 30°S–30°N (easterly trades)</th>"
        "<th>extratropics (mostly westerlies)</th></tr></thead><tbody>" + ws_rows + "</tbody></table></div>"
    )
    WL = WS["layers"]
    wlr = WL["regions"]

    def wl_cell(r):
        word = ("too slow" if r["lo"] > 0 else "too fast") if r["sig"] else "n.s."
        return f"<td>{num(r['mean'], 4)} <span class='sig'>{ci(r['lo'], r['hi'], 4)}</span> {word}</td>"

    wl_names = (
        ("1000-950", "1000–950 hPa (surface layer)"),
        ("925-875", "925–875 hPa (boundary-layer interior)"),
        ("1000-875", "1000–875 hPa (whole boundary layer)"),
        ("850-700", "850–700 hPa (above it)"),
        ("1000-700", "1000–700 hPa (net)"),
    )
    t_layint = (
        "<div class='tw'><table class='res'><thead><tr><th>layer</th><th>tropics, 30°S–30°N</th><th>extratropics, 30–60°</th>"
        "</tr></thead><tbody>"
        + "".join(
            f"<tr><th scope='row'>{lab}</th>{wl_cell(wlr['trop'][k])}{wl_cell(wlr['extra'][k])}</tr>"
            for k, lab in wl_names
        )
        + "</tbody></table></div>"
    )
    trop_off = abs(wlr["trop"]["850-700"]["mean"]) / abs(wlr["trop"]["1000-875"]["mean"])
    # where the column's wind-term increment goes (both window starts, every cell of Figure incsection)
    _z = np.load(D / "inc_section.npz")
    _J = np.nan_to_num(_z["J10_mean"] + _z["J22_mean"])
    _lv, _la = _z["level"], _z["lat"]
    col_ext = _J[:, np.abs(_la) >= 30].sum() / _J.sum()
    ext_free = _J[(_lv >= 100) & (_lv <= 650)][:, np.abs(_la) >= 30].sum() / _J[:, np.abs(_la) >= 30].sum()
    wg = WS.get("grid", {})
    ws_grid = "; ".join(
        f"{lv} hPa {100 * wg[lv]['trop']['strengthen']:.0f}% and {100 * wg[lv]['extra']['strengthen']:.0f}%"
        for lv in ("1000", "925", "850", "700")
        if lv in wg
    )

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

    def lead(reg, L):
        return next(x for x in fd[reg] if x["lead"] == L)

    fg1, fn15, fs7 = lead("global", 1), lead("nh", 15), lead("sh", 7)
    nh_first = next(
        (
            x["lead"]
            for x in fd["nh"]
            if x["lead"] > 0 and x["sig"] and all(y["sig"] for y in fd["nh"] if y["lead"] >= x["lead"])
        ),
        None,
    )
    sh_sig = sum(x["sig"] for x in fd["sh"] if x["lead"] >= 3)
    nver = (lead("global", 15)["n"], lead("global", 1)["n"])

    def E(v, d=3):
        return f"{num(v['mean'], d)} {ci(v['lo'], v['hi'], d)}"

    per_day = g10["mean"] + g22["mean"]
    col37 = (u10["global_mean"] + u22["global_mean"]) / 2
    col2d = (w10["mean"] + w22["mean"]) / 2
    lay_rows = "".join(
        f"<tr><th scope='row'>{LAYER.get(nm, nm)}</th><td>{num(L10[nm]['mean'], 3)} <span class='sig'>{ci(L10[nm]['lo'], L10[nm]['hi'], 3)}</span></td>"
        f"<td>{num(L22[nm]['mean'], 3)} <span class='sig'>{ci(L22[nm]['lo'], L22[nm]['hi'], 3)}</span></td></tr>"
        for nm in L10
    )
    t_lay = (
        "<div class='tw'><table class='res'><thead><tr><th>wind-term increment, 10²⁵ kg m² s⁻¹ per window</th>"
        "<th>09 UTC start</th><th>21 UTC start</th></tr></thead><tbody>"
        + lay_rows
        + f"<tr class='diff'><th scope='row'>all 37 levels</th><td>{num(u10['global_mean'], 3)}</td><td>{num(u22['global_mean'], 3)}</td></tr>"
        + "</tbody></table></div>"
    )
    bel_rows = "".join(
        f"<tr><th scope='row'>{nm.replace('-', '–')}</th><td>{num(bl['10_' + nm]['mean'], 3)}</td><td>{num(bl['22_' + nm]['mean'], 3)}</td></tr>"
        for nm in ("90-60S", "60-30S", "30S-0", "0-30N", "30-60N", "60-90N")
    )
    t_bel = (
        "<div class='tw'><table class='res'><thead><tr><th>total increment, 10²⁵ kg m² s⁻¹ per window</th><th>09 UTC start</th>"
        "<th>21 UTC start</th></tr></thead><tbody>"
        + bel_rows
        + f"<tr class='diff'><th scope='row'>global</th><td>{num(g10['mean'], 3)}</td><td>{num(g22['mean'], 3)}</td></tr>"
        + "</tbody></table></div>"
    )
    reg_terms = [
        ("AND", "Andes"),
        ("ALP", "Alps"),
        ("ROC", "Rockies"),
        ("GRL", "Greenland"),
        ("HIM", "Himalaya/Tibet"),
        ("ANT", "Antarctica"),
        ("other_mtn", "other mountain"),
        ("fric", "friction"),
        ("gwd", "gravity-wave drag"),
    ]
    reg_rows = "".join(
        f"<tr><th scope='row'>{lab}</th>"
        + (
            f"<td>{num(100 * it[k]['coef'], 1)}% <span class='sig'>{ci(100 * it[k]['lo'], 100 * it[k]['hi'], 1)}</span></td>"
            if it[k].get("fdr")
            else f"<td class='ns'>{num(100 * it[k]['coef'], 1)}% n.s.</td>"
        )
        + f"<td>{num(it[k]['share_of_mean_residual'], 1)}</td></tr>"
        for k, lab in reg_terms
    )
    t_reg = (
        "<div class='tw'><table class='res'><thead><tr><th>in-window hourly residual regressed on</th><th>coefficient</th>"
        "<th>share of the mean residual (Hadley)</th></tr></thead><tbody>"
        + reg_rows
        + f"<tr class='diff'><th scope='row'>constant</th><td>{num(it['intercept']['coef'], 1)} Hadley <span class='sig'>{ci(it['intercept']['lo'], it['intercept']['hi'], 1)}</span></td>"
        f"<td>{num(it['intercept']['coef'], 1)}</td></tr></tbody></table></div>"
    )
    regnames = [
        ("ORO", "terrain above 1000 m or rough subgrid orography, and 500 km downstream"),
        ("OCX", "ocean 30–65°"),
        ("OCO", "other ocean (tropics, subtropics, polar)"),
        ("LND", "other land"),
    ]
    t_split = (
        "<div class='tw'><table class='res'><thead><tr><th>share of the increment</th><th>area</th><th>below 850 hPa, all days</th>"
        "<th>DJF</th><th>JJA</th><th>whole column</th></tr></thead><tbody>"
        + "".join(
            f"<tr><th scope='row'>{lab}</th><td>{100 * area[r]:.0f}%</td><td>{share('amL', 'ALL', r)}</td>"
            f"<td>{share('amL', 'DJF', r)}</td><td>{share('amL', 'JJA', r)}</td><td>{share('amC', 'ALL', r)}</td></tr>"
            for r, lab in regnames
        )
        + "</tbody></table></div>"
    )
    # ERA5 AAM regression on Nino-3.4 (peak lag) for the explainer
    lr = pd.read_csv(D / "q2_era5_lagreg.csv")
    aamr = lr[lr.series == "aam_glob"].set_index("lag")
    aam_peak_lag = int(aamr.slope.idxmax())
    aam_peak = float(aamr.slope.max())

    def sig_span(s, frame=lr, col="sig"):
        g = frame[frame.series == s].set_index("lag").sort_index()
        L = g.index[g[col]].tolist()
        return (min(L), max(L), int(g.index.min())) if L else None

    span_AND, span_mtn, span_fric = sig_span("mtn5_AND"), sig_span("mtn_glob"), sig_span("fric_glob")
    span_cmtn = sig_span("mtn_glob", pd.read_csv(D / "q2_cmip6_slopes_summary.csv"), "robust")

    WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}

    def span_txt(sp):
        a, b, first = sp
        lead = (
            f"{'at least ' if a == first else ''}{WORDS.get(abs(a), abs(a))} month{'s' if abs(a) != 1 else ''} ahead of"
        )
        return f"from {lead} Niño-3.4 to {WORDS.get(b, b)} month{'s' if b != 1 else ''} after it"

    def span_lags(sp):
        a, b, _first = sp
        return f"{'−' if a < 0 else '+'}{abs(a)} to {'−' if b < 0 else '+'}{abs(b)} months"

    LODMS = 1e25 / 5.2092e33 * 86400 * 1e3  # ms of length of day per 1e25 kg m2/s
    # GFZ global AAM by ENSO phase (torque_gfz_enso.py), the explainer's opening figure
    GZ = json.loads((D / "gfz_enso.json").read_text())
    go = GZ["oni"]
    gs = pd.read_csv(D / "gfz_enso_series.csv", parse_dates=["month"]).set_index("month")
    g26 = float(gs.aam3.loc["2026-08-01"])
    g26_oni = float(gs.oni.loc["2026-08-01"])
    g26_rank = int((gs.aam3.dropna() > g26).sum() + 1)
    g26_top = gs.aam3.drop(pd.Timestamp("2026-08-01")).dropna().idxmax()
    g26_ratio = g26 / (go["slope_per_K"]["est"] * g26_oni)
    g_off_pre = GZ["gfz_minus_era5_annual"]
    off_pre = sum(g_off_pre[str(y)] for y in (1976, 1977, 1978)) / 3
    off_post = [v for k, v in g_off_pre.items() if int(k) >= 1979]
    off_post = sum(off_post) / len(off_post)
    off_dev = max(abs(v - off_post) for k, v in g_off_pre.items() if int(k) >= 1979)
    slp_all = comp[(comp.cmp == "ONI-slope") & (comp.metric == "lp_mean") & comp.sig]
    alpS = slp_all[(slp_all.range == "ALP") & (slp_all.season == "SON")]

    body = f"""
<article class="study">
  <p class="crumbs"><a href="/">Home</a> › Studies › El Niño and angular momentum</p>
  <p class="ai-note"><b>Made with Claude Code.</b> This study is a just-for-fun project: researched, coded and written entirely by Claude, Anthropic's AI model, working in Claude Code, to show what today's AI models can do with open meteorological data. It has not been peer reviewed.</p>
  <h1>{H1}</h1>
  <p class="dek">In El Niño the westerlies strengthen and the day lengthens by a fraction of a millisecond: the atmosphere takes
  angular momentum from the solid Earth. This study traces how it gets in, mainly through a pressure difference across the Andes that
  lasts for months, and why ERA5's budget closes only with its analysis increments. Data: ERA5 four times a day for 1959–2026,
  {n_models} CMIP6 models and an hourly budget of 2026.</p>

  <section class="lede">
  <h2>The short answer</h2>
  <ul>
    <li><b>1. The Andes torque supplies it; friction brakes it.</b> As the eastern Pacific warms, the southeast Pacific high west of
    the Andes weakens and pressure rises over eastern South America. Pressure stays higher on the east side of the range for months, so the
    mountain torque stays positive. In December–February the Andes torque is {abs(lpA["diff"]):.1f} Hadley higher than in neutral
    winters ({pval(lpA["p"])}; CMIP6 {num(cA["median"], 2)}, all {int(cA.n)} models with daily data agreeing in sign). It is
    significantly positive {span_txt(span_AND)}. Over the six months leading Niño-3.4, mountain torque supplies
    {num(mt["est"], 2)} {ci(mt["lo"], mt["hi"], 2)} ×10²⁵ kg m² s⁻¹ per kelvin of Niño-3.4 in ERA5 and {num(cmt["median"], 2)} in
    CMIP6, with the Andes the largest source. The weaker trades hand the atmosphere less westerly momentum, so friction takes back
    about {abs(cfr["median"] / cmt["median"]):.0%} of the mountain torque in the models and more than all of it in ERA5. The
    individual east-side highs are no more frequent or stronger.</li>
    <li><b>2. ERA5's model loses westerly momentum, and the analysis puts it back.</b> ERA5's torques fall
    {abs(fsum):.1f} Hadley short of the annual budget. At its two daily 4D-Var window starts the analysis adds angular momentum, together the
    equivalent of a steady torque of about {abs(per_day / S.HADLEY_DAY_TO_AAM):.0f} Hadley, in every month of 2025–26. The physics ensemble IFS-ENS loses angular
    momentum in free forecasts through its own torques; the data-driven AIFS-ENS does not. Too much drag in ERA5's model (IFS cycle
    41r2, before the orographic-drag revision ECMWF made operational in cycle 48r1 in June 2023) is the leading candidate, not the only one. About {100 * col_ext:.0f}% of the momentum
    the analysis adds goes into mid-latitude westerlies, {100 * ext_free:.0f}% of that between 650 and 100 hPa; in the tropics the
    gain below 875 hPa and the loss at 850–700 hPa nearly cancel.</li>
    <li><b>3. Too much drag, but not over the mountains.</b> ECMWF's own record (Kanehama et al., 2022) shows the
    IFS tuned toward more drag because it improved scores. Yet only {share("amL", "ALL", "ORO")} of the momentum the analysis adds below
    850 hPa lands over or just downstream of rough terrain, which covers {100 * area["ORO"]:.0f}% of the globe;
    {share("amL", "ALL", "OCO")} lands over the tropical and other oceans. In the tropics the whole trade-wind boundary layer is too fast: the
    analysis slows the trades from the surface to about 875 hPa, more than it speeds them up at 850–700 hPa. In mid-latitudes the
    boundary layer is too slow and the air just above it too fast, and the column balances. Neither pattern sits over mountains, and
    only the mid-latitude one looks like too much drag.</li>
    <li><b>Which comes first.</b> Day to day, mountain torque leads the rise in global angular momentum, as the budget requires,
    and the subtropical jet follows only weakly (r ≈ {max(jR.r, jA.r):.2f}). The 500 hPa pattern that sets up an east-side high (a ridge upstream, a trough downstream) is the same in El Niño and neutral
    years.</li>
  </ul>
  </section>

  <h2>1. What angular momentum is, and why it rises in El Niño</h2>
  <p>Angular momentum about the Earth's axis measures how much faster than the planet the atmosphere turns. Most of it is in the
  westerly winds, weighted toward low latitudes, which are farther from the axis; a smaller mass term depends on how the air is spread
  between latitudes. The atmosphere can gain it only from the solid Earth and oceans, so when the atmosphere spins up the Earth slows
  and the day lengthens. Rosen et al. (1984) showed this in the 1982–83 El Niño; Stefanick (1982) had already tied year-to-year swings in angular momentum and day length to the Southern Oscillation. Here 10²⁵ kg m² s⁻¹ of extra atmospheric
  angular momentum lengthens the day by about {LODMS:.2f} ms.</p>
  <p><b>Why El Niño spins the atmosphere up.</b> El Niño warms the whole tropical troposphere, with bands of cooling in the middle
  latitudes of both hemispheres (Seager et al., 2003). By the thermal-wind relation the stronger temperature gradient strengthens
  the upper westerlies on the equatorward side of both subtropical jets. In ERA5, global angular momentum regressed on Niño-3.4 peaks
  {abs(aam_peak_lag)} month{"" if abs(aam_peak_lag) == 1 else "s"} {"after" if aam_peak_lag > 0 else "before"} the Niño-3.4 anomaly,
  at {num(aam_peak, 2)} ×10²⁵ kg m² s⁻¹ per kelvin ({aam_peak * LODMS:.2f} ms of day length).</p>
  <p><b>From warm ocean to faster winds.</b> The thermal wind says where the westerlies strengthen, not where their momentum comes
  from. Without a torque, stronger jets would have to be paid for by easterlies elsewhere. The momentum comes from the solid Earth,
  mostly across the Andes, in four steps. (1) The central and eastern Pacific warm. (2) The southeast Pacific high west of the Andes
  weakens and pressure rises over eastern South America; in northern winter the deeper Aleutian low does the same west of the Rockies
  (Figure {FN["slp"]}). (3) Pressure stays higher on the east side of both ranges for months. (4) Their mountain torque therefore
  stays positive and moves westerly momentum into the atmosphere. Friction is the brake, because the weaker trades hand the
  atmosphere less westerly momentum. The rise is what the mountain torque supplies minus what friction holds back (section 3).</p>
  <p><b>What the record shows.</b> GFZ Potsdam's record of angular momentum, computed from ECMWF's analyses independently of this
  study's ERA5 calculation, shows the rise plainly (Figure {FN["gfz"]}). Since 1979 global angular momentum has been above normal in
  {100 * go["share_EN_above"]:.0f}% of El Niño seasons and below normal in {100 * go["share_LN_below"]:.0f}% of La Niña seasons. On
  average it is {num(go["EN"]["est"], 2)} {ci(go["EN"]["lo"], go["EN"]["hi"], 2)} ×10²⁵ kg m² s⁻¹ in El Niño and
  {num(go["LN"]["est"], 2)} {ci(go["LN"]["lo"], go["LN"]["hi"], 2)} in La Niña, a swing of about
  {go["EN_minus_LN"]["est"] * LODMS:.2f} ms in day length. It rises by {num(go["slope_per_K"]["est"], 2)}
  {ci(go["slope_per_K"]["lo"], go["slope_per_K"]["hi"], 2)} ×10²⁵ per kelvin of the ONI (r = {go["r"]["est"]:.2f},
  {pval(max(go["r"]["p"], go["r"]["p_bretherton"]))}), and the GFZ and ERA5 monthly anomalies correlate at
  {GZ["corr_monthly_anom_with_era5"]:.2f}. Intraseasonal swings, the MJO among them (Weickmann et al., 1997), still move it by a
  few tenths from month to month. In July–September 2026 it was {g26:.1f} ×10²⁵ above normal,
  {"the highest in the record" if g26_rank == 1 else f"second only to early {g26_top.year}" if g26_rank == 2 else rank_word(g26_rank) + " highest in the record"}
  and {"about twice" if 1.75 <= g26_ratio < 2.25 else f"{g26_ratio:.1f} times"} what an ONI of {g26_oni:.1f} K (provisional) alone predicts
  (section 6).</p>
  {fig("gfz", "gfz_aam_enso.webp", "Top: monthly 3-month-mean global angular momentum anomalies from GFZ, 1979 to 2026, coloured red in El Niño seasons, blue in La Niña seasons and grey otherwise, with the ONI scaled by the regression overlaid. Bottom: the same anomalies against the ONI, with the regression line, the mean of each phase with its 95% interval, and the 2026 seasons marked", f"Global angular momentum (wind term) from GFZ's effective angular momentum functions, built from ECMWF's analyses: 3-month means of the monthly anomaly from 1991–2020, with the 1979–2025 trend removed, against CPC's ONI for the same season, {go['span'][0][:4]}–{go['span'][1][:4]} ({go['n_months']} overlapping seasons). Top: the anomaly, coloured by ENSO phase; the black line is the ONI times the fitted slope; the right axis gives the change in the length of the day. Bottom: each season against its ONI. Large dots: phase means with 95% intervals (24-month block bootstrap, 2,000 draws); open circles: 2026.")}
  <p><b>How the momentum gets in.</b> All of it has to come through the surface:</p>
  <p>$$\\frac{{dM}}{{dt}} = T_{{\\text{{mountain}}}} + T_{{\\text{{friction}}}} + T_{{\\text{{gravity-wave drag}}}},\\qquad
  T_{{\\text{{mountain}}}} = -\\iint p_s\\,\\frac{{\\partial h}}{{\\partial\\lambda}}\\,dA = \\iint h\\,\\frac{{\\partial p_s}}{{\\partial \\lambda}}\\,dA .$$</p>
  <p>Torques are in Hadley (10¹⁸ N m); one Hadley held for a day changes $M$ by 0.0086 ×10²⁵ kg m² s⁻¹. The mountain torque is
  <b>positive when pressure is higher on the east slope of a range than on the west</b>: the air pushes westward on the mountain,
  and the mountain pushes the air eastward. Through friction, surface easterlies (the trades) give the atmosphere westerly momentum
  and surface westerlies take it back. Gravity-wave drag is the model's account of unresolved mountains. Cold highs banked against
  the east side of a range, such as cold surges east of the Rockies (Colle and Mass, 1995), cold-air incursions east of the Andes
  (Garreaud, 2000) and the East Asian surges, supply most of the day-to-day mountain torque and much of the week-to-week swing in
  angular momentum (Iskenderian and Salstein, 1998; Weickmann, 2003; Lott et al., 2004). See the <a href="/topics/mountain-torque.html">mountain torque</a> and
  <a href="/topics/atmospheric-angular-momentum.html">angular momentum</a> explainers.</p>
  <p>Two questions are less settled. Does El Niño's extra momentum come through those weather events or through the time-mean
  pressure pattern? And how does it split between mountain and friction torque? Reanalysis budgets do not close exactly (Huang et al.,
  1999; Egger et al., 2007), so climate models, which have no analysis increments, are the cross-check.</p>

  <h2>2. Data and method</h2>
  <ul>
    <li><b>ERA5</b> (Hersbach et al., 2020) from the ARCO-ERA5 store on Google Cloud (Carver and Merose, 2023), read in GitHub Actions: surface pressure, the
    turbulent and gravity-wave surface stresses and the vertical integral of eastward mass flux (an exact full-column wind term).
    Data are at 00, 06, 12 and 18 UTC from 1 January 1959 to 3 October 2026 (24,748 days, none missing); the last two to three months are ECMWF's preliminary ERA5T release. Four times a day rather than
    two, because at 00 and 12 UTC the semidiurnal pressure tide has the same phase and a twice-daily mean aliases it. The mass term
    of $M$ comes from surface pressure. For 2026 the fields were also read hourly from 1 March. The analysis period is 1979–2025,
    with 1959–2025 as a sensitivity test.</li>
    <li><b>Torques</b> are the site's: friction $-a^3\\cos^2\\varphi\\,\\tau_\\lambda$, gravity-wave drag likewise from ERA5's own
    stress, and mountain torque $h\\,\\partial p_s/\\partial\\lambda$ on the 0.25° grid with ERA5's orography. They are summed over
    the site's six boxes (5° cells; table below), six latitude bands and the globe.</li>
  </ul>
  {boxes}
  <ul>
    <li><b>Anomalies</b>: each daily series minus its 1991–2020 seasonal cycle (mean and four harmonics), with the 1979–2025
    linear trend removed (ERA5's torques drift and angular momentum does not; section 10). A Lanczos filter (121-day window; Duchon,
    1979) splits each range's torque into a standing part (periods of 30 days and longer) and a synoptic part.</li>
    <li><b>Events</b>: peaks of the synoptic part above +1 standard deviation, at least 5 days apart (negative events likewise),
    about seven per range and season. The same rule on the total anomaly gives the "days above threshold" events a forecaster
    sees.</li>
    <li><b>GFZ</b> effective angular momentum functions (ESMGFZ v1.0; Dobslaw et al., 2010), the external check. The 3-hourly wind
    term from ECMWF's analyses is averaged to months (complete months only), taken as anomalies from 1991–2020 with the 1979–2025
    trend removed, then averaged over 3 months to match the ONI's seasons. The years 1976–78 are left out: GFZ minus ERA5 is
    {off_pre - off_post:.2f} ×10²⁵ larger then than its mean since 1979, against at most {off_dev:.2f} in any later year, and which
    record is wrong is unknown. With CPC's relative ONI (RONI; L'Heureux et al., 2024) the slope is
    {num(GZ["roni"]["slope_per_K"]["est"], 2)} {ci(GZ["roni"]["slope_per_K"]["lo"], GZ["roni"]["slope_per_K"]["hi"], 2)}; without the
    trend removal, {num(GZ["oni_no_detrend"]["slope_per_K"]["est"], 2)} {ci(GZ["oni_no_detrend"]["slope_per_K"]["lo"], GZ["oni_no_detrend"]["slope_per_K"]["hi"], 2)}.</li>
    <li><b>ENSO phase</b> from CPC's ONI for each season (El Niño ≥ +0.5, La Niña ≤ −0.5, otherwise neutral; CPC's official index is now the RONI, above). Season-years
    1979–2025: DJF {nsy["EN"]["DJF"]} El Niño, {nsy["NEU"]["DJF"]} neutral, {nsy["LN"]["DJF"]} La Niña; MAM {nsy["EN"]["MAM"]},
    {nsy["NEU"]["MAM"]}, {nsy["LN"]["MAM"]}; JJA {nsy["EN"]["JJA"]}, {nsy["NEU"]["JJA"]}, {nsy["LN"]["JJA"]}; SON {nsy["EN"]["SON"]},
    {nsy["NEU"]["SON"]}, {nsy["LN"]["SON"]}.</li>
    <li><b>Tests</b>: each season-year is one unit. El Niño minus neutral differences are tested by permuting the phase labels
    (20,000 draws), with bootstrap 95% intervals; regressions on the ONI use a moving-block bootstrap (blocks of three season-years; Künsch, 1989).
    The false discovery rate is held at 10% across the 24 range–season cells of each statistic (Benjamini and Hochberg, 1995; Wilks,
    2016). Monthly regressions on the detrended ERSSTv6 (Huang et al., 2025) Niño-3.4 anomaly at lags of −6 to +12 months use a 24-month block bootstrap
    (2,000 draws), with the FDR across lags. Maps are tested per grid cell with the FDR over the field.</li>
    <li><b>CMIP6</b> historical runs 1950–2014 from the Pangeo Google Cloud mirror (Eyring et al., 2016; Abernathey et al., 2021): {n_models} models and
    {n_members} members (up to three per model), each on its own grid with its own orography. The fields are monthly mountain torque
    from surface pressure (the torque is linear in $p_s$, so monthly means suffice), surface stress <code>tauu</code>, angular momentum
    from <code>ua</code> masked by surface pressure, and Niño-3.4 from <code>ts</code>. {n_gwd} models also publish the orographic
    gravity-wave-drag tendency <code>utendogw</code> (Gerber and Manzini, 2016). {n_daily} have daily surface pressure, which gives
    event statistics on each model's own ENSO. A model result is <b>robust</b> when the across-model test passes the FDR and at least
    80% of models share the sign.</li>
    <li><b>Analysis increments</b> (section 7): surface fields at 08–11 and 20–23 UTC every day from {day_txt(I["days"][0])} to
    {day_txt(I["days"][1])} ({I["days"][2]} days), integrated in 5° bands, and the 37-level winds at the same hours
    ({u10["n_days"]} days; on a 2.5° grid every third day for the maps). Intervals are 10-day block bootstraps over days, with the FDR
    at 10% across bands, across levels and bands for the section, and across grid cells for the maps. The vertical sum of the
    37-level increment tracks the column estimate from the surface fields (correlation {u10["corr_column_vs_2d"]:.3f}) but is about
    {100 * (col37 / col2d - 1):.0f}% larger ({num(col37, 3)} against {num(col2d, 3)} ×10²⁵ kg m² s⁻¹ per window), probably because
    37 pressure levels do not integrate the model levels exactly.</li>
  </ul>

  <h2>3. How the momentum gets in: the torque budget</h2>
  <p>Summing each term's regression on Niño-3.4 over lags −6 to −1 months gives the angular momentum it supplies in the six months
  before the anomaly (Figure {FN["q2con"]} and the table). In ERA5 the tendency supplies {num(ce[tend]["est"], 2)}
  {ci(ce[tend]["lo"], ce[tend]["hi"], 2)} ×10²⁵ kg m² s⁻¹ per kelvin. The regression of angular momentum itself rises by
  {num(q2["aam_regression_change_lag0_minus_lagm6"], 2)} between those lags; the two differ because the tendencies are taken across
  month edges. The mass term is negligible (about +0.02 per kelvin).</p>
  <p>Mountain torque is the source and friction the brake, and both act for months (Figure {FN["q2lag"]}). The Andes torque is
  significantly positive at every lag from {span_lags(span_AND)}. The global mountain torque is significant from {span_lags(span_mtn)}
  and robust in CMIP6 from {span_lags(span_cmtn)}. Friction is significantly negative from {span_lags(span_fric)}, mostly in the
  tropics, where the trades weaken.</p>
  {fig("q2lag", "q2_lag_regression.webp", "Lagged regressions on Niño-3.4 of the global mountain, friction and gravity-wave-drag torques, their sum and the angular momentum tendency, ERA5 above and the CMIP6 median and interquartile range below", "Regression of each monthly global term on Niño-3.4 (detrended anomalies) against lag. Top: ERA5 1979–2025, filled dots significant (FDR over lags). Bottom: CMIP6 1950–2014, median and interquartile range over models, filled dots robust.")}
  {fig("q2con", "q2_contributions.webp", "Horizontal bars of the angular momentum each term supplies before the Niño-3.4 anomaly, ERA5 bars with confidence intervals and CMIP6 diamonds with interquartile ranges, globally, by range and by latitude band", "Angular momentum supplied in the six months before the Niño-3.4 anomaly, per kelvin. Bars: ERA5 with 95% joint block-bootstrap intervals, pale where not significant after the FDR. Diamonds: CMIP6 median with interquartile range, filled where robust.")}
  {t_q2}
  <p class="tcap">ERA5 cells show the estimate and 95% interval where significant after the FDR, otherwise "n.s.". CMIP6 cells
  show the median and the share of models of its sign where robust.</p>
  <p>ERA5 and CMIP6 agree on the mountain torque and where it acts: Andes {num(ce["mtn5_AND"]["est"], 2)} in ERA5 and
  {num(cAND["median"], 2)} in CMIP6 ({cAND.agree:.0%} of models), Rockies {num(ce["mtn5_ROC"]["est"], 2)} and
  {num(cROC["median"], 2)}. They differ on friction. In the models friction takes back about
  {abs(cfr["median"] / cmt["median"]):.0%} of the mountain torque, leaving {num(cT["median"], 2)}, about
  {cT["median"] / cdm["median"]:.1f} times the models' tendency of {num(cdm["median"], 2)}. In ERA5 friction takes back more than
  all of it ({num(fr["est"], 2)} against {num(mt["est"], 2)}). ERA5's torques therefore sum to nothing significant while angular
  momentum rises significantly. The gap sits in the budget residual, which is itself not significant; whether ERA5's analysis
  increments (section 7) vary with ENSO was not tested. The monthly budget
  otherwise closes reasonably (correlation of tendency with summed torques {mc["corr"]:.2f}, slope {mc["slope"]:.2f}), but the
  residual is slow (lag-one autocorrelation {mc["r_lag1_R"]:.2f}), as an interannual imbalance would be.</p>
  <p>In ERA5's composites of El Niño onsets (13 onsets against 14 neutral years, April–December) only gravity-wave drag differs
  significantly ({num(comp_en["gwd_glob"]["diff"], 2)} ×10²⁵ kg m² s⁻¹, {pval(comp_en["gwd_glob"]["p"])}); the sample is too small
  for the other terms. In CMIP6 the onset composites are robust: angular momentum {num(ccd["median"], 2)}, mountain torque {num(ccm["median"], 2)} and the Andes
  {num(cca["median"], 2)}. CMIP6 friction ({num(ccf["median"], 2)}, {ccf.agree:.0%} agreement) is not robust.</p>

  <h2>4. The Andes: a standing shift, not more events</h2>
  <p>Range by range, the signal is the pressure difference of section 1: a slow shift that forms as the Pacific warms and lasts the
  season. It is not more or stronger weather events.</p>
  {fig("q1grid", "q1_shift_vs_events.webp", "Grid of six mountain ranges by four seasons showing the El Niño minus neutral difference in the standing torque, the number of positive torque events and their amplitude; only the Andes in DJF and the Himalaya in MAM are significant, and only in the standing part", "El Niño minus neutral seasons, ERA5 1979–2025, for each range and season. Top: the standing (30-day and longer) part of the torque. Middle and bottom: the number and the amplitude of positive synoptic events. Coloured cells are significant at a false discovery rate of 10% (permutation of season-years); grey cells, marked n.s., are not. A black frame marks a change that is robust in the CMIP6 models with daily data (significant across models and at least 80% of models of one sign).")}
  <p>The synoptic events do not change (Figure {FN["q1grid"]}, middle and bottom). Positive Andes events in DJF average
  {npA["ma"]:.2f} a season in El Niño winters and {npA["mb"]:.2f} in neutral ones ({pval(npA["p"])}), and their amplitude does not
  change ({pval(amA["p"])}). Across the 24 range–season cells, {n_ev_txt} of the 72 differences in event count, amplitude and
  variance {"is" if n_ev <= 1 else "are"} significant. Against the ONI, {len(nslope)} of 24 cells show a significant change in
  event count, all small and of mixed sign
  ({"; ".join(f"{RSHORT[r_.range]} {r_.season} {num(r_['diff'], 2)} per unit ONI" for _, r_ in nslope.iterrows())}).</p>
  <p>The standing part does change (Figure {FN["q1grid"]}, top). Against the ONI it is significant in {len(slp_all)} cells, six of
  them large: the Andes in DJF ({num(slA["diff"])} Hadley per unit ONI {ci(slA["lo"], slA["hi"])}) and JJA ({num(slAJ["diff"])}
  {ci(slAJ["lo"], slAJ["hi"])}), the Rockies in DJF ({num(slR["diff"])} {ci(slR["lo"], slR["hi"])}) and MAM ({num(slRM["diff"])}
  {ci(slRM["lo"], slRM["hi"])}), and the Himalaya/Tibet in MAM ({num(slH["diff"])} {ci(slH["lo"], slH["hi"])}) and SON ({num(slHS["diff"])}
  {ci(slHS["lo"], slHS["hi"])}){f"; the seventh is the Alps in SON ({num(alpS['diff'].iloc[0], 2)} {ci(alpS['lo'].iloc[0], alpS['hi'].iloc[0], 2)})" if len(alpS) else ""}.
  Nearly all of the El Niño difference in the seasonal-mean torque is this standing part:</p>
  {t_dec}
  <p class="tcap">The CMIP6 column is the median over the {n_daily} models with daily surface pressure, with the share of models of
  the median's sign; Andes and Himalaya are robust, the Rockies in DJF ({cRD.agree:.0%}) is not. CMIP6 is also robust for the Rockies
  in MAM ({num(cRM["median"], 2)}) and JJA ({num(cRJ["median"], 2)}) and the Himalaya in SON ({num(cHS["median"], 2)}). The ERA5
  Rockies DJF difference ({num(lpR["diff"])}, {pval(lpR["p"])}) does not survive the FDR.</p>
  {fig("q1thr", "q1_threshold_events.webp", "Two grids of ranges by seasons: the El Niño minus neutral change in days-above-threshold torque events, and the part of that change not explained by a pure shift of the distribution", f"Top: events counted on the total anomaly, as a forecaster sees them. Bottom: the pure-shift test, the El Niño count minus the count obtained by shifting each neutral season by the El Niño–neutral difference in the standing part. Colours and frames as in Figure {FN['q1grid']}.")}
  <p><b>Why El Niño winters look eventful.</b> Counted on the total anomaly, Andes DJF has {nxA["ma"]:.2f} positive events a
  season in El Niño winters against {nxA["mb"]:.2f} in neutral ones ({pval(nxA["p"])}, just short of the FDR). Shifting each
  neutral winter by the El Niño–neutral difference in the standing part predicts {sA["expected_shift"]:.2f}, and the excess over
  that, {num(sA["excess"], 2)} {ci(sA["lo"], sA["hi"], 2)}, is not significant ({pval(sA["p"])}); the same holds for the Rockies.
  In the models the extra threshold events are robust (Andes DJF {num(cnA["median"], 2)} a season, {cnA.agree:.0%} of models;
  Rockies DJF {num(cnR["median"], 2)}, {cnR.agree:.0%}) while the synoptic counts are not. Two cells go the other way, with fewer
  events than the shift alone gives: the Himalaya/Tibet in spring ({num(sH["excess"], 2)} {ci(sH["lo"], sH["hi"], 2)}) and
  Greenland in SON ({num(sG["excess"], 2)}), both significant after the FDR.</p>
  {fig("slp", "slp_on_oni.webp", "Global maps of seasonal-mean sea-level pressure regressed on the ONI for DJF and MAM, with the Himalaya, Rockies and Andes boxes outlined", "The mechanism. ERA5 1991–2020 seasonal-mean sea-level pressure regressed on the ONI, shaded only where significant (FDR 10%). In DJF the deeper Aleutian low lowers pressure west of the Rockies and the weaker southeast Pacific high lowers it west of the Andes; over South America the rise is significant only in the east, over Brazil. Pressure ends up higher on the east side of both ranges, a positive torque. In MAM, pressure rises over India and the Philippine Sea. Contours (every 0.5 hPa per unit ONI, dashed where negative) show all values.")}

  <h2>5. Which comes first</h2>
  {fig("q3ll", "q3_lead_lag.webp", "Four panels of lagged correlation between each range's synoptic mountain torque and the global angular momentum tendency, the global angular momentum and the zonal-mean subtropical 200 hPa wind", f"Lagged correlation of each range's synoptic torque with the global tendency of angular momentum, global angular momentum and the zonal-mean 200 hPa wind at 25–40° in the same hemisphere (all &lt;30-day parts). Dots: significant, with effective sample size after Bretherton et al. (1999; here {neff_lo:,} to {neff_hi:,} days) and the FDR over lags.")}
  <p>Day by day (Figure {FN["q3ll"]}), high torque follows a few days of low global angular momentum (lowest correlation
  {num(min(p_.r for p_ in pre_lo), 2)} at lags of −{min(abs(int(p_.lag)) for p_ in pre_lo)} to
  −{max(abs(int(p_.lag)) for p_ in pre_lo)} days) and comes before a rise (up to {num(max(p_.r for p_ in post_hi), 2)},
  {dd(post_hi[int(np.argmax([p_.r for p_ in post_hi]))].lag)} later). The zonal-mean subtropical jet follows weakly, peaking {dd(jR.lag)} after
  Rockies torque (r = {num(jR.r, 2)}) and {dd(jA.lag)} after Andes torque (r = {num(jA.r, 2)}) in DJF, while the Himalayan torque in
  DJF shows almost nothing. Before Andes torque the jet is slightly weaker (r = {num(jbA.r, 2)}, {dd(jbA.lag, before=True)}). That the torque
  leads angular momentum follows from the budget itself; the jet result says the torque is not a response to a jet that is
  already stronger, but the jet's response is small (about {100 * max(jR.r, jA.r) ** 2:.0f}% of its day-to-day variance).</p>
  {fig("q3z", "q3_z500_precursors.webp", "Maps of 500 hPa height one to three days before positive torque events over the Himalaya, Rockies and Andes in DJF, for all events and for El Niño minus neutral", "The precursor. 500 hPa height (&lt;30-day part) averaged 3 to 1 days before positive events, all events (left) and El Niño minus neutral (right), shaded only where significant (FDR 10% over the map; contours every 10 m show all values).")}
  <p>The precursor is the textbook one for a lee-side surge: a ridge upstream and poleward of the range and a trough downstream.
  It sits over the Urals and northern China for the Himalaya, Alaska and the central United States for the Rockies, and southern
  Chile and the southwest Atlantic for the Andes ({qc["HIM_DJF"]["frac_sig_all"]:.0%}, {qc["ROC_DJF"]["frac_sig_all"]:.0%} and
  {qc["AND_DJF"]["frac_sig_all"]:.0%} of each map significant). El Niño does not change it: no cell of the El Niño minus neutral
  difference is significant for any of the three ranges in DJF ({qc["AND_JJA"]["frac_sig_diff"]:.0%} of the map for the Andes in
  JJA). El Niño seasons also have {"a significantly different" if pt_any else "no significantly different"} number of precursors,
  and no different share of them followed by an event within four days ({"some" if pt_any else "none"} of
  {len(pt[pt.metric != "n_events"])} tests significant after the FDR).</p>

  <h2>6. The 2026 event</h2>
  <p>By the CPC ONI, 2026's El Niño grew from {num(oni26[4], 2)} in MAM to {num(oni26[8], 2)} in JAS (provisional: CPC
  revises the latest seasons for about two months). In May the Andes torque was
  {num(may26)} Hadley above normal, the {"largest" if may_rank == 1 else ("second-largest" if may_rank == 2 else f"{may_rank}th largest")} May
  since 1959 after May {may_first[0]} ({num(may_first[1])}), the start of that year's very strong El Niño. Over 15 April – 19 September
  the Andes torque anomaly added {num(att["Andes"]["v2026"])} ×10²⁵ kg m² s⁻¹ (the {rank_txt(att["Andes"]["rank_of_48"])},
  z = {num(att["Andes"]["z"], 1)}; {num(att["Andes"]["v2026_detrended"])} with ERA5's torque trend removed), against
  {num(att["Andes"]["en_onset_mean"])} in the average El Niño onset.</p>
  {fig("b26cum", "b2026_cumulative.webp", "Lines of the cumulative 2026 change in angular momentum anomaly and of the cumulative torque anomalies by term since 15 April", "ERA5, from hourly fields: the change in the angular momentum anomaly (motion plus mass term) since 15 April and the cumulative torque anomalies that should explain it. Descriptive.")}
  <p>The budget does not close. The angular momentum anomaly rose by {num(B["attribution_dM_hourly_anomaly"])} ×10²⁵ kg m² s⁻¹
  while the torque anomalies summed to {num(B["attribution_total_torque"])} (Figure {FN["b26cum"]}). The Andes term tracked angular
  momentum and mountain torque outside the six boxes added {num(att["rest of mountain"]["v2026"])}, but the other five ranges
  ({num(att["other ranges"]["v2026"])}) and tropical friction ({num(att["friction tropics"]["v2026"])}) worked against it. A
  residual of {num(vh["residual"])} is not unusual for ERA5: the same window leaves {num(hw["mean"])} ± {hw["sd"]:.1f} in
  1979–2025, so 2026 is {num(zr, 1)} standard deviations from the usual. ERA5 therefore cannot attribute the 2026 rise. The Andes
  signal is real and among the largest on record, but the torques as a whole explain none of the rise, and the analysis increments
  carry it (section 7).</p>
  {fig("b26att", "b2026_attribution.webp", "Horizontal bars of each torque term's 2026 integral against the interannual spread of the same window", "Each term's anomaly integrated over 15 April – 19 September 2026 (dark: as observed; pale: with ERA5's 1979–2025 trend removed), against ± one standard deviation of the same window in 1979–2025 (beige).")}
  {t_att}
  <p class="tcap">Anomalies against ERA5's 1991–2020 climatology at four times a day; 2026 daily means from hourly fields. Rank 48 is
  the largest. The trend-removed column is a sensitivity, not a correction: ERA5's tropical friction drifts by
  {num(att["friction tropics"]["trend_hadley_per_decade"], 2)} Hadley per decade.</p>
  <p><b>The analysis-increment floor.</b> Hourly fields let the budget be followed through ERA5's twelve-hour 4D-Var windows
  (Figure {FN["b26hr"]}). Inside a window the residual is {num(wt["inside_mean"])} ± {wt["inside_sd"]:.1f} Hadley each hour. In
  the hours from 09 to 10 UTC and 21 to 22 UTC it jumps to {num(g[10]["mean"], 0)} ± {g[10]["std"]:.0f} and
  {num(g[22]["mean"], 0)} ± {g[22]["std"]:.0f}: the analysis adds {num(inc["10"], 3)} and {num(inc["22"], 3)} ×10²⁵ kg m² s⁻¹ at
  each window start. A three-hourly product agrees ({num(t3["across_mean"], 0)} ± {t3["across_sd"]:.0f} Hadley across window starts,
  {num(t3["inside_mean"], 1)} ± {t3["inside_sd"]:.1f} inside). Daily means of the hourly residual are {num(dfh["mean_R"])} ±
  {dfh["sd_R"]:.1f} Hadley with almost no memory from day to day (lag-one correlation {dfh["lag1"]:.2f}). At periods longer than
  about two days the torques carry nearly all of the tendency's variance.</p>
  {fig("b26hr", "b2026_hourly.webp", "Bar chart of the hourly budget residual by hour of day with the two window-start bars cut off and labelled, and spectra of the tendency, the summed torques and the residual", "Top: the hourly residual (tendency minus torques) by the hour that ends each interval, mean ± sd over 1 March – 3 October 2026; the two window-start bars are cut off and labelled. Bottom: spectra of the tendency, the summed torques and the residual.")}
  <p><b>GFZ as a check.</b> GFZ's functions, from ECMWF's operational analysis every three hours with the pressure tides removed and
  an inverted-barometer ocean, put the change over the window at {num(gfz["dM_window_total"], 2)} ×10²⁵ kg m² s⁻¹ (wind term
  {num(gfz["dMr_window"], 2)}, mass term {num(gfz["dMm_window"], 2)}), against {num(gfz["era5_dM_window_total_absolute"], 2)} from
  ERA5's daily means of hourly fields. Their daily tendencies correlate at {gfz["corr_daily_dMdt_vs_era5"]:.2f}. The mass terms differ by construction, since GFZ's
  includes the ocean's inverted-barometer response.</p>

  <h2>7. Why a reanalysis budget does not close: ERA5's analysis increments</h2>
  <p>ERA5's open archive has no analysis-increment fields, but its hourly fields jump at each 4D-Var window start, so the increment
  can be estimated from the jump. With $R(h) = X(h) - X(h-1) - T(h)\\,\\Delta t$ the budget residual of the hour ending at $h$, for a
  band's angular momentum $X$ and torques $T$, the increment at the 09 UTC window start is</p>
  <p>$$J_{{10}} = R(10) - \\tfrac{{1}}{{2}}\\left[R(09) + R(11)\\right],$$</p>
  <p>and likewise $J_{{22}}$ at 21 UTC. Subtracting the hours either side removes whatever changes smoothly (transport between
  bands, the tides, trends); the torques barely change across the jump ({num(T10["mean"], 4)} ×10²⁵ kg m² s⁻¹). A jump also holds
  any model adjustment in the first hour of the new trajectory, and it describes ERA5's model, IFS cycle 41r2 (Hersbach et al.,
  2020), before the orographic-drag revision of IFS cycle 48r1 (operational June 2023; Kanehama et al., 2022). Data and tests: section 2.</p>
  {fig("incprof", "inc_profile.webp", "Latitude profile of the mean angular momentum increment in 5-degree bands for the 09 and 21 UTC window starts, and its split into wind and mass terms", "Top: the mean increment in each 5° band at the 09 UTC (blue) and 21 UTC (orange) window starts, with 95% intervals; filled dots are significant after the FDR. The legend's global totals are in 10²² kg m² s⁻¹ (48 ×10²² = 0.048 ×10²⁵). Bottom: both window starts together, split into the wind (motion) and mass terms, with intervals and significance as above.")}
  <p><b>How much and where.</b> The analysis adds {E(g10)} ×10²⁵ kg m² s⁻¹ at the 09 UTC start and {E(g22)} at 21 UTC, about
  {num(per_day, 3)} a day, equal to a steady torque of {num(per_day / S.HADLEY_DAY_TO_AAM, 1)} Hadley (section 6's in-window
  baseline gives {num(inc["10"], 3)} and {num(inc["22"], 3)}). The <b>wind term is the same at both starts</b>, {E(w10)} and
  {E(w22)}; the difference is a surface-pressure (<b>mass-term</b>) increment, {E(m10)} at 09 UTC against {E(m22)} at 21 UTC. By
  latitude, angular momentum is added in both extratropical belts and removed in the tropics (except 30°S–0 at 09 UTC):</p>
  {t_bel}
  <p class="tcap">Belts are sums of the 5° bands; all twelve belt means are significant after the FDR. Most of the tropical removal is
  in the mass term (Figure {FN["incprof"]}, bottom), a shift of surface pressure between the equator and the subtropics that says
  nothing about drag; the wind term alone is positive in both extratropical belts and slightly negative in the tropics.</p>
  {fig("incseason", "inc_season.webp", "Monthly global increment per day and a month by latitude diagram of the increment", "Top: the global increment per day, month by month, with 95% intervals (complete months only). Bottom: the same by latitude in 5° bands; hatched where not significant (t-test with an effective sample size from the lag-one autocorrelation, FDR 10%).")}
  <p><b>Through the year.</b> The global increment is positive in every month, smallest in northern summer
  ({month_txt(gmin)}: {num(gfull[gmin]["mean"], 3)} ×10²⁵ a day) and largest in northern spring ({month_txt(gmax)}:
  {num(gfull[gmax]["mean"], 3)}), with the extratropical gain and tropical loss in every month (Figure {FN["incseason"]}).</p>
  {fig("incsection", "inc_section.webp", "Latitude by pressure sections of the wind-term angular momentum increment per layer at the two window starts, troposphere and stratosphere on separate scales", "The wind-term increment by latitude and layer at the two window starts: the angular momentum each 5° band and pressure layer gains, in units that sum to the column. Below: 1000–100 hPa on an axis linear in pressure (so in mass); above it, 100–1 hPa on its own scale. Hatched where not significant (10-day block bootstrap, FDR 10% over the cells with data).")}
  <p><b>By height.</b> The largest share of the wind-term increment goes into the layers below 850 hPa:</p>
  {t_lay}
  <p>Between about 35°S and 35°N the analysis adds westerly momentum from the surface to about 900 hPa and removes it at
  850–700 hPa (Figure {FN["incsection"]}; {fsec.get("J10", 0):.0%} and {fsec.get("J22", 0):.0%} of the cells with data are
  significant). The trades are easterly, so that is a weaker trade wind below 900 hPa and a stronger one above; the dipole sits
  over the trade-wind oceans in the maps below. In the extratropics the analysis adds westerly momentum through the middle
  troposphere (400–600 hPa, strongest at 30–55°S) and in the northern subtropical jet at 100–200 hPa. Above 100 hPa the increments
  are small.</p>
  <p><b>Strengthened or weakened?</b> Because angular momentum counts westerlies as positive, its sign alone does not say whether
  the model's wind was too strong or too weak. Too much drag would show as increments that strengthen the wind they act on. The
  table gives, level by level, the share of the significant zonal-mean increment that has the sign of the background wind:</p>
  {t_sign}
  <p class="tcap">Share of the FDR-significant increment in Figure {FN["incsection"]} (5° bands, both window starts, weighted by
  size) that has the sign of the background zonal-mean wind; "strengthens" at 67% or more, "weakens" at 33% or less. Background:
  ERA5 zonal wind at 12 UTC on {wsb["days"]} days of {month_txt(wsb["months"][0])} to {month_txt(wsb["months"][1])}, log-pressure
  interpolated from 1000, 925, 850 and 700 hPa, bands with |u| below 0.5 m s⁻¹ left out. The 2.5° maps give the same pattern
  (share strengthening, tropics and extratropics): {ws_grid}.</p>
  <p><b>Layer by layer.</b> Too little drag leaves a layer's wind too fast as a whole; extra mixing only moves momentum between
  layers. Summed over each layer, with the sign of the background wind (positive: the analysis strengthens it, so the model's wind
  was too slow):</p>
  {t_layint}
  <p class="tcap">Wind-term angular momentum increment per day, both window starts, 10²⁵ kg m² s⁻¹, signed by the background wind;
  95% intervals from a 10-day block bootstrap over {WL["days"]} days. n.s.: the interval includes zero.</p>
  <p>In the tropics the whole trade-wind boundary layer is too fast, and the too-slow layer above it offsets about
  {100 * trop_off:.0f}% of that, so the lower troposphere carries too much easterly momentum. That fits too little drag on the
  trades, or too-strong low-level pressure gradients; the increments cannot tell them apart. In mid-latitudes the 1000 hPa
  westerlies are slightly too strong, but the boundary layer as a whole is too slow and the layer above it too fast, and the column
  below 700 hPa balances: momentum is not mixed down into the boundary layer fast enough, or is taken out of it too fast.</p>

  <h3>Where the low-level increment goes</h3>
  {fig("mapsu", "inc_maps_u.webp", "World maps of the mean zonal-wind increment at 925 and 850 hPa with terrain contours", "The mean increment of the zonal wind at 925 hPa (top) and 850 hPa (bottom), both window starts together, per day; shaded only where significant (t-test with an effective sample size from the lag-one autocorrelation, FDR 10% over the map). Contours: terrain at 500, 1500 and 3000 m.")}
  {fig("mapsseason", "inc_maps_amL_season.webp", "World maps of the angular momentum increment of the layers at and below 850 hPa in December to February and June to August", "The angular momentum increment of the layers at and below 850 hPa, per 2.5° cell and day, in DJF (top) and JJA (bottom); shaded only where significant (FDR 10%). Terrain contours as in the previous figure.")}
  <p>If ERA5's model lost momentum mainly to excess low-level orographic drag (blocking and turbulent orographic form drag), the
  low-level increments would sit over and just downstream of the Rockies, the Andes, Tibet, the East African highlands and the
  Antarctic margins. They do not. At 925 hPa the analysis adds westerly momentum over most of the trade-wind oceans and around
  Antarctica; at 850 hPa it removes it over the subtropical oceans. In the zonal mean the 1000 hPa wind rises by
  {num(u1000_trop, 2)} m s⁻¹ a day at 10–30° (weaker trade easterlies) and falls by {abs(u1000_mid):.2f} m s⁻¹ at 40–60° (weaker
  surface westerlies), and the 10 m wind does the same ({num(u10_trop, 2)} and {num(u10_mid, 2)}). In mid-latitudes the analysis
  adds westerlies at 925 hPa ({num(u925_mid, 2)} m s⁻¹ a day at 40–60°) but removes them at 850 hPa between 40 and 55°N
  ({num(u850_nh, 2)}).</p>
  {fig("mapssplit", "inc_maps_split.webp", "Bar chart of the share of the global low-level angular momentum increment falling in four regions, for all days, DJF and JJA, against each region's share of the Earth's area", "The share of the global increment of angular momentum below 850 hPa that falls in each region, all days and by season, with 95% block-bootstrap intervals over days; the dashed lines mark each region's share of the Earth's area. Regions are defined at 0.25° from ERA5's orography, its subgrid-orography standard deviation and its land-sea mask.")}
  {t_split}
  <p class="tcap">"Rough terrain" is terrain above 1000 m or an ERA5 subgrid-orography standard deviation above 100 m, extended 500 km
  east (downstream); the four regions add up to the globe. With terrain above 1000 m alone the low-level share is
  {share("amL", "ALL", "HGT")} ({100 * area["HGT"]:.0f}% of the area); with the subgrid criterion alone {share("amL", "ALL", "SDO")}
  ({100 * area["SDO"]:.0f}%).</p>
  <p>Rough terrain gets less than an even share of the low-level increment even in northern winter, when blocking drag matters
  most, and a negative share in JJA. Over the whole column the storm-track oceans at 30–65° dominate.</p>

  <h3>Possible explanations</h3>
  <p><b>Models are tuned toward more drag.</b> Surface-drag parameters "are poorly constrained and often tuned to obtain the
  desired model behavior" (Sandu et al., 2016). ECMWF raised its turbulent orographic form drag "to
  optimise forecast skill", notably in cycle 40r1, which left the IFS with much more form-drag stress than the Met Office model; by 2022 its
  surface winds were also too weak over orography (Kanehama et al., 2022). ERA5's cycle 41r2 came between that change and the
  cycle 48r1 revision (operational June 2023) that reduced form drag again. Budget studies of model drift had earlier pushed models toward less stratospheric and more
  low-level drag (Klinker and Sardeshmukh, 1992, as summarised by Brown, 2004). No source measures what such tuning does to a model's angular
  momentum budget, so the link to the lost westerly momentum is an inference, one that ERA5's increments are consistent with.</p>
  <p><b>But not as low-level orographic drag.</b> The increments weaken the surface winds over the oceans, both the trades and the
  westerlies, so the model's ocean surface winds are too strong, not too weak. Its northern mid-latitude 850 hPa westerlies are, if
  anything, too strong as well. Scatterometer and buoy winds pulling the surface winds down would look
  the same at 10 m and 1000 hPa, but would not by themselves speed up the mid-latitude wind at 950–900 hPa. Too-strong surface westerlies also mean too much friction torque, so boundary-layer
  drag, rather than orographic drag, remains a candidate for the lost momentum.</p>
  <p><b>Gravity-wave drag aloft.</b> COORDE (van Niekerk et al., 2020) found the IFS, like several other models, short of
  orographic gravity-wave drag in the lower stratosphere (as summarised by Kanehama et al., 2022), which would leave excess westerly
  momentum there for the analysis to remove. Over 25–50°N at 50–150 hPa the increments instead add westerly momentum,
  {E(u10["ls_25_50N"], 4)} ×10²⁵ kg m² s⁻¹ per window at 09 UTC and {E(u22["ls_25_50N"], 4)} at 21 UTC, against
  {E(u10["trop_25_50N"], 4)} and {E(u22["trop_25_50N"], 4)} from 400 hPa to the surface. A zonal mean cannot isolate the air near
  the ranges, where COORDE looked, so this tests only the zonal-mean budget.</p>
  <p><b>The steady residual inside the windows.</b> Within a window the hourly residual is {num(ir["mean_residual"], 1)} Hadley.
  Regressed on each range's mountain torque, friction and gravity-wave drag (2026, {ir["n_hours"]:,} hours, day-block bootstrap),
  it is mostly a constant, and the terms explain {ir["r2"]:.0%} of its variance:</p>
  {t_reg}
  <p class="tcap">A coefficient of +1% means the residual behaves as if the model's own torque were 1% larger than the one recomputed
  here from the 0.25° output. The "share" is the coefficient times that term's mean torque.</p>
  <p>The significant range coefficients are a few per cent and of both signs; all the terms together account for
  {num(sum(it[k]["share_of_mean_residual"] for k, _ in reg_terms), 1)} Hadley of the mean, against a constant of
  {num(it["intercept"]["coef"], 1)}. The constant could be an offset between the model's own mountain torque on its N320 grid and the
  torque recomputed from 0.25° output (the mean mountain torque changes by tens of Hadley between 0.25° and 1°; section 10), or
  non-conservation in the semi-Lagrangian dynamics; the vertical integrals cannot tell them apart. Increments and initial tendencies
  have long been read as diagnostics of model error in this way (Klinker and Sardeshmukh, 1992; Rodwell and Palmer, 2007).</p>

  <h3>The forecast side</h3>
  {fig("fcdrift", "fc_drift.webp", "AIFS-ENS ensemble-mean angular momentum minus the verifying analysis by lead day, globally and by hemisphere", f"AIFS-ENS ensemble-mean relative angular momentum (14 levels) minus the same calculation for the verifying analysis, by lead, with 95% weekly-block bootstrap intervals; filled where the interval excludes zero. {finits[2]} inits, of which {nver[0]}–{nver[1]} can be verified at each lead.")}
  <p>A physics model whose torques drain angular momentum should lose it in free forecasts, and IFS-ENS does: in the site's winter
  2025–26 test it loses 0.31 ×10²⁵ kg m² s⁻¹ by day 15 while closing its own torque budget almost exactly
  (<a href="/aifs-aam-budget.html">Do AI weather models conserve angular momentum?</a>). That is the forecast-side counterpart of
  ERA5's positive increments. The data-driven AIFS-ENS, trained on analyses that already contain the correction, shows no global
  drain ({nver[0]} to {nver[1]} forecasts per lead, inits {day_txt(finits[0], year=False)} to {day_txt(finits[1])}): {E(fg1)} ×10²⁵
  at day 1 and no significant change after. It shifts angular momentum between the hemispheres instead. The Northern Hemisphere gains
  ({E(fn15)} at day 15, significant from day {nh_first}) and the Southern loses ({E(fs7)} at day 7, significant at {sh_sig} of the
  13 leads from day 3). The comparison is not like for like: the seasons and IFS cycles differ, and an increment is not a
  free-forecast drift.</p>

  <h2>8. What closes the budget and what does not</h2>
  <p>Two choices matter for closure. The first is sampling: the summed torques and the tendency correlate at
  {cl["1"]["corr_dMdt_T"]:.2f} hour by hour, because the increments dominate single hours, and at {cl["24"]["corr_dMdt_T"]:.2f} in
  steps of 24 hours. The second is the mass term: without it the 24-hour residual has an sd of {cl["24"]["sd_Rmotion_only"]:.1f}
  Hadley instead of {cl["24"]["sd_R"]:.1f}, because the tidal pressure waves that drive the mountain torque within a day are
  balanced by the mass term, not by the winds.</p>
  {t_step}
  <p class="tcap">Each row samples the hourly 2026 fields every 1, 3, 6, 12 or 24 hours and closes the budget over those steps, so the
  24-hour row is not the same estimator as the daily means of the hourly residual in section 6.</p>
  <p>For the 2026 window, sampling more often and adding the mass term make the closure worse, not better. The once-daily 12Z
  wind-term budget leaves {num(v12["residual"], 2)} ×10²⁵ kg m² s⁻¹. Adding the mass term raises the angular momentum change by
  {num(v12m["dM"] - v12["dM"], 2)}, and four- and 24-times-daily torques lower the torque integral by
  {abs(v4["torque"] - v12["torque"]):.1f} and {abs(vh["torque"] - v12["torque"]):.1f}. Absolute budgets at one and three hours agree to {abs(va1["residual"] - va3["residual"]):.1f},
  and both miss by about {num(va1["residual_mean_hadley"], 0)} Hadley on average.</p>
  {t_var}
  <p><b>The persistent residual, and the −4.5 Hadley figure.</b> Averaged over 1991–2020 at four times a day, ERA5's torques sum to
  {num(fsum, 1)} Hadley while angular momentum has no trend; hourly in 2026 the residual is {num(cl["1"]["mean_R"], 1)} Hadley. Until
  12 October 2026 the site's torque products (<code>build_torque_clim.py</code> and the
  <a href="/topics/mountain-torque.html">mountain torque explainer</a>) quoted −4.5 ± 0.7 Hadley, from 12 UTC samples alone. At that
  hour the semidiurnal tide adds about +4 Hadley to the mountain torque, and friction is far weaker than at 06 UTC, when the
  daytime boundary layer over Asia is strongest:</p>
  {t_hours}
  <p>ERA5's real imbalance is therefore about two and a half times the figure once quoted. It sits in the mean and cancels in
  anomalies, so the anomaly products were never affected.</p>

  <h2>9. How this fits with earlier work</h2>
  <p><b>Mountain torque as the source.</b> This is the usual picture, drawn mostly from case studies of the largest events. The
  sharp rise in early 1983 came mostly from mountain torque over the American and Eurasian ranges (Ponte and Rosen, 1999).
  Eastern-Pacific El Niños put low pressure west of the Andes, Rockies and Himalaya and lengthen the day more than twice as much as
  central-Pacific ones (de Viron and Dickey, 2014). In most El Niños after 1975 angular momentum rose by roughly 10²⁵ kg m² s⁻¹ per kelvin
  of Niño-3.4 (Huang et al., 2003). This study differs in three ways. It uses every El Niño since 1979 through regressions, not single
  events. It splits each range's torque into a standing part and synoptic events, and finds that only the standing part changes.
  And it checks the result in {n_models} CMIP6 models, which have no analysis increments.</p>
  <p><b>Friction as the brake.</b> Marcus et al. (2011) found the same pairing in the 1976–77 Pacific climate shift: mountain torque
  over the Andes, Rockies and Himalaya was largely offset by friction over the equatorial Pacific. More generally, mountain torque
  creates changes in angular momentum and friction damps them (Egger et al., 2007). Not every event follows the rule. In 2015–16 the
  mountain torque was weaker than in 1982–83 and 1997–98 and friction added to it, yet the day lengthened by a similar amount
  (Lambert et al., 2017). The friction share found here, about {abs(cfr["median"] / cmt["median"]):.0%} of the mountain torque in
  the models, is an average over events, not a rule for each one.</p>
  <p><b>Where this study disagrees: the Himalaya.</b> The case studies count the Himalaya as a source in large eastern-Pacific
  events. Averaged over all events its contribution here is slightly negative: {num(cHIM["median"], 2)} in CMIP6 (robust; not
  significant in ERA5), with a significant negative standing shift in spring ({num(lpH["diff"])} Hadley, {pval(lpH["p"])};
  section 4). The likely reasons are the samples (every event here, the extremes there) and the mix of eastern- and
  central-Pacific events, which this study does not separate.</p>
  <p><b>Reanalysis budgets that do not close.</b> NCEP's torques fall about 10 Hadley short of its observed change (Huang et al.,
  1999), and ERA-20C's total torque has a strong negative bias (Veerman and van Heerwaarden, 2019). ERA5's fall {abs(fsum):.1f}
  Hadley short, the same sign and size. What is new is measuring the analysis increments directly from the jumps at ERA5's 4D-Var
  window starts, together the equivalent of a steady {abs(per_day / S.HADLEY_DAY_TO_AAM):.0f} Hadley torque, and mapping where they fall. Veerman and van Heerwaarden (2019) inferred the
  same correction indirectly in ERA-20C, whose forecasts lose about 9 Hadley that the analyses restore; a literature search found
  no earlier direct estimate, which does not prove none exists.</p>
  <p><b>Surface winds over the oceans.</b> Extra mixing in stable boundary layers, although known to be wrong for those layers, has
  long been used to tune operational models and partly stands in for orographic drag (Sandu et al., 2013). Against ASCAT
  scatterometer winds, ERA5's mean surface westerlies over the mid-latitude oceans are too strong (Belmonte Rivas and Stoffelen,
  2019), which fits the increments weakening them (section 7). That comparison is of the analysis itself, so it shows the correction
  is incomplete rather than measuring the model's own error.</p>

  <h2>10. Caveats</h2>
  <ul>
    <li><b>The mean mountain torque is not resolution-converged</b> (Brown, 2004; van Niekerk et al., 2016). Averaging orography and surface pressure together to 1° or
    2.5° moves the global mean from about +1 to −26 and −33 Hadley, and the Andes box from −223 to −148. The anomalies still
    correlate at 0.96 or better with the 0.25° values, and their ENSO regressions barely change (global +1.00, +1.07 and +0.99
    Hadley per kelvin). Anomalies from CMIP6's 1–2.5° grids are therefore comparable with ERA5's; absolute values are not. The Andes
    DJF standing shift at 2.5° is +2.1 Hadley ({pval(0.009)}), short of the FDR.</li>
    <li><b>Tide aliasing.</b> Sampling at 00 and 12 UTC keeps the semidiurnal tide; four times a day removes it, and hourly removes
    the rest (the four-versus-24 offsets are below 0.2 Hadley).</li>
    <li><b>What CMIP6's <code>tauu</code> is.</b> It is the boundary-layer surface stress; the Met Office models state that it
    includes turbulent form drag from small hills. Orographic gravity-wave drag is reported separately (<code>utendogw</code>, in
    {n_gwd} models). ERA5's turbulent stress likewise includes turbulent orographic form drag. Gravity-wave drag's part in the El Niño
    rise is not significant in either data set, so the answer does not depend on where it is booked.</li>
    <li><b>ERA5's torques drift.</b> The global mountain and friction torques trend by +0.5 and +0.8 Hadley per decade over
    1979–2025 with no matching trend in angular momentum, so they are detrended for the ENSO analysis. The 2026 attribution is shown
    both ways.</li>
    <li><b>ENSO phase</b> is each season's ONI against fixed thresholds, not CPC's rule of five overlapping seasons. The CMIP6 models
    use ±0.5 K on their own index whatever the size of their ENSO.</li>
    <li><b>Model independence.</b> Several CMIP6 models share code (four CESM2 variants, NorESM2 and others built on CAM, three MPI
    models), so "{n_models} models" overstates the independent evidence; a one-model-per-centre check was not made.</li>
    <li><b>Samples.</b> Super El Niños (ONI ≥ 2.0; three DJFs) are too few to test and are not used as evidence. The zonal-mean jet
    comes from 1.5° daily ERA5 fields, northern hemisphere 1979–2023 and southern 1991–2020. The torque boxes are the site's and do
    not close the terrain at their edges, so a box's mean includes part of the large-scale pressure gradient over high ground.</li>
  </ul>

  <h2>11. Data and code</h2>
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
    <li>Abernathey, R. P., and co-authors, 2021: Cloud-native repositories for big scientific data. <i>Comput. Sci. Eng.</i>, 23(2), 26–35. <a href="https://doi.org/10.1109/MCSE.2021.3059437">doi:10.1109/MCSE.2021.3059437</a></li>
    <li>Belmonte Rivas, M., and A. Stoffelen, 2019: Characterizing ERA-Interim and ERA5 surface wind biases using ASCAT. <i>Ocean Sci.</i>, 15, 831–852. <a href="https://doi.org/10.5194/os-15-831-2019">doi:10.5194/os-15-831-2019</a></li>
    <li>Benjamini, Y., and Y. Hochberg, 1995: Controlling the false discovery rate: a practical and powerful approach to multiple testing. <i>J. R. Stat. Soc. B</i>, 57, 289–300. <a href="https://doi.org/10.1111/j.2517-6161.1995.tb02031.x">doi:10.1111/j.2517-6161.1995.tb02031.x</a></li>
    <li>Bretherton, C. S., M. Widmann, V. P. Dymnikov, J. M. Wallace and I. Bladé, 1999: The effective number of spatial degrees of freedom of a time-varying field. <i>J. Climate</i>, 12, 1990–2009. <a href="https://doi.org/10.1175/1520-0442(1999)012%3C1990:TENOSD%3E2.0.CO;2">doi:10.1175/1520-0442(1999)012&lt;1990:TENOSD&gt;2.0.CO;2</a></li>
    <li>Brown, A. R., 2004: Resolution dependence of orographic torques. <i>Q. J. R. Meteorol. Soc.</i>, 130, 3029–3046. <a href="https://doi.org/10.1256/qj.04.21">doi:10.1256/qj.04.21</a></li>
    <li>Carver, R. W., and A. Merose, 2023: ARCO-ERA5: An analysis-ready cloud-optimized reanalysis dataset. <i>22nd Conf. on Artificial Intelligence for Environmental Science</i>, Denver, CO, Amer. Meteor. Soc., 4A.1.</li>
    <li>Colle, B. A., and C. F. Mass, 1995: The structure and evolution of cold surges east of the Rocky Mountains. <i>Mon. Wea. Rev.</i>, 123, 2577–2610. <a href="https://doi.org/10.1175/1520-0493(1995)123%3C2577:TSAEOC%3E2.0.CO;2">doi:10.1175/1520-0493(1995)123&lt;2577:TSAEOC&gt;2.0.CO;2</a></li>
    <li>Dobslaw, H., R. Dill, A. Grötzsch, A. Brzeziński and M. Thomas, 2010: Seasonal polar motion excitation from numerical models of atmosphere, ocean, and continental hydrosphere. <i>J. Geophys. Res.</i>, 115, B10406. <a href="https://doi.org/10.1029/2009JB007127">doi:10.1029/2009JB007127</a></li>
    <li>Duchon, C. E., 1979: Lanczos filtering in one and two dimensions. <i>J. Appl. Meteor.</i>, 18, 1016–1022. <a href="https://doi.org/10.1175/1520-0450(1979)018%3C1016:LFIOAT%3E2.0.CO;2">doi:10.1175/1520-0450(1979)018&lt;1016:LFIOAT&gt;2.0.CO;2</a></li>
    <li>Egger, J., K. Weickmann and K.-P. Hoinka, 2007: Angular momentum in the global atmospheric circulation. <i>Rev. Geophys.</i>, 45, RG4007. <a href="https://doi.org/10.1029/2006RG000213">doi:10.1029/2006RG000213</a></li>
    <li>Eyring, V., and co-authors, 2016: Overview of the Coupled Model Intercomparison Project Phase 6 (CMIP6) experimental design and organization. <i>Geosci. Model Dev.</i>, 9, 1937–1958. <a href="https://doi.org/10.5194/gmd-9-1937-2016">doi:10.5194/gmd-9-1937-2016</a></li>
    <li>Garreaud, R. D., 2000: Cold air incursions over subtropical South America: mean structure and dynamics. <i>Mon. Wea. Rev.</i>, 128, 2544–2559. <a href="https://doi.org/10.1175/1520-0493(2000)128%3C2544:CAIOSS%3E2.0.CO;2">doi:10.1175/1520-0493(2000)128&lt;2544:CAIOSS&gt;2.0.CO;2</a></li>
    <li>Gerber, E. P., and E. Manzini, 2016: The Dynamics and Variability Model Intercomparison Project (DynVarMIP) for CMIP6: assessing the stratosphere–troposphere system. <i>Geosci. Model Dev.</i>, 9, 3413–3425. <a href="https://doi.org/10.5194/gmd-9-3413-2016">doi:10.5194/gmd-9-3413-2016</a></li>
    <li>Hersbach, H., and co-authors, 2020: The ERA5 global reanalysis. <i>Q. J. R. Meteorol. Soc.</i>, 146, 1999–2049. <a href="https://doi.org/10.1002/qj.3803">doi:10.1002/qj.3803</a></li>
    <li>Huang, B., X. Yin, T. Boyer, C. Liu, M. Menne, Y. D. Rao, T. Smith, R. Vose and H.-M. Zhang, 2025: Extended Reconstructed Sea Surface Temperature, Version 6 (ERSSTv6). Part I: An artificial neural network approach. <i>J. Climate</i>, 38, 1105–1121. <a href="https://doi.org/10.1175/JCLI-D-23-0707.1">doi:10.1175/JCLI-D-23-0707.1</a></li>
    <li>Huang, H.-P., P. D. Sardeshmukh and K. M. Weickmann, 1999: The balance of global angular momentum in a long-term atmospheric data set. <i>J. Geophys. Res.</i>, 104, 2031–2040. <a href="https://doi.org/10.1029/1998JD200068">doi:10.1029/1998JD200068</a></li>
    <li>Huang, H.-P., K. M. Weickmann and R. D. Rosen, 2003: Unusual behavior in atmospheric angular momentum during the 1965 and 1972 El Niños. <i>J. Climate</i>, 16, 2526–2539. <a href="https://doi.org/10.1175/1520-0442(2003)016%3C2526:UBIAAM%3E2.0.CO;2">doi:10.1175/1520-0442(2003)016&lt;2526:UBIAAM&gt;2.0.CO;2</a></li>
    <li>Iskenderian, H., and D. A. Salstein, 1998: Regional sources of mountain torque variability and high-frequency fluctuations in atmospheric angular momentum. <i>Mon. Wea. Rev.</i>, 126, 1681–1694. <a href="https://doi.org/10.1175/1520-0493(1998)126%3C1681:RSOMTV%3E2.0.CO;2">doi:10.1175/1520-0493(1998)126&lt;1681:RSOMTV&gt;2.0.CO;2</a></li>
    <li>Kanehama, T., I. Sandu, A. Beljaars, A. van Niekerk, N. Wedi, S. Boussetta, S. Lang, S. Johnson and L. Magnusson, 2022: Evaluation and optimization of orographic drag in the IFS. ECMWF Tech. Memo. 893. <a href="https://doi.org/10.21957/fps6gngqe">doi:10.21957/fps6gngqe</a></li>
    <li>Klinker, E., and P. D. Sardeshmukh, 1992: The diagnosis of mechanical dissipation in the atmosphere from large-scale balance requirements. <i>J. Atmos. Sci.</i>, 49, 608–627. <a href="https://doi.org/10.1175/1520-0469(1992)049%3C0608:TDOMDI%3E2.0.CO;2">doi:10.1175/1520-0469(1992)049&lt;0608:TDOMDI&gt;2.0.CO;2</a></li>
    <li>Künsch, H. R., 1989: The jackknife and the bootstrap for general stationary observations. <i>Ann. Statist.</i>, 17, 1217–1241. <a href="https://doi.org/10.1214/aos/1176347265">doi:10.1214/aos/1176347265</a></li>
    <li>Lambert, S. B., S. L. Marcus and O. de Viron, 2017: Atmospheric torques and Earth's rotation: what drove the millisecond-level length-of-day response to the 2015–2016 El Niño? <i>Earth Syst. Dynam.</i>, 8, 1009–1017. <a href="https://doi.org/10.5194/esd-8-1009-2017">doi:10.5194/esd-8-1009-2017</a></li>
    <li>L'Heureux, M. L., and co-authors, 2024: A relative sea surface temperature index for classifying ENSO events in a changing climate. <i>J. Climate</i>, 37, 1197–1211. <a href="https://doi.org/10.1175/JCLI-D-23-0406.1">doi:10.1175/JCLI-D-23-0406.1</a></li>
    <li>Lott, F., A. W. Robertson and M. Ghil, 2004: Mountain torques and Northern Hemisphere low-frequency variability. Part I: Hemispheric aspects. <i>J. Atmos. Sci.</i>, 61, 1259–1271. <a href="https://doi.org/10.1175/1520-0469(2004)061%3C1259:MTANHL%3E2.0.CO;2">doi:10.1175/1520-0469(2004)061&lt;1259:MTANHL&gt;2.0.CO;2</a></li>
    <li>Marcus, S. L., O. de Viron and J. O. Dickey, 2011: Abrupt atmospheric torque changes and their role in the 1976–1977 climate regime shift. <i>J. Geophys. Res.</i>, 116, D03107. <a href="https://doi.org/10.1029/2010JD015032">doi:10.1029/2010JD015032</a></li>
    <li>van Niekerk, A., T. G. Shepherd, S. B. Vosper and S. Webster, 2016: Sensitivity of resolved and parametrized surface drag to changes in resolution and parametrization. <i>Q. J. R. Meteorol. Soc.</i>, 142, 2300–2313. <a href="https://doi.org/10.1002/qj.2821">doi:10.1002/qj.2821</a></li>
    <li>van Niekerk, A., I. Sandu, A. Zadra, E. Bazile, T. Kanehama, M. Köhler, M.-S. Koo, H.-J. Choi, Y. Kuroki, M. D. Toy, S. B. Vosper and V. Yudin, 2020: COnstraining ORographic Drag Effects (COORDE): a model comparison of resolved and parametrized orographic drag. <i>J. Adv. Model. Earth Syst.</i>, 12, e2020MS002160. <a href="https://doi.org/10.1029/2020MS002160">doi:10.1029/2020MS002160</a></li>
    <li>Ponte, R. M., and R. D. Rosen, 1999: Torques responsible for evolution of atmospheric angular momentum during the 1982–83 El Niño. <i>J. Atmos. Sci.</i>, 56, 3457–3462. <a href="https://doi.org/10.1175/1520-0469(1999)056%3C3457:TRFEOA%3E2.0.CO;2">doi:10.1175/1520-0469(1999)056&lt;3457:TRFEOA&gt;2.0.CO;2</a></li>
    <li>Rodwell, M. J., and T. N. Palmer, 2007: Using numerical weather prediction to assess climate models. <i>Q. J. R. Meteorol. Soc.</i>, 133, 129–146. <a href="https://doi.org/10.1002/qj.23">doi:10.1002/qj.23</a></li>
    <li>Rosen, R. D., D. A. Salstein, T. M. Eubanks, J. O. Dickey and J. A. Steppe, 1984: An El Niño signal in atmospheric angular momentum and Earth rotation. <i>Science</i>, 225, 411–414. <a href="https://doi.org/10.1126/science.225.4660.411">doi:10.1126/science.225.4660.411</a></li>
    <li>Sandu, I., A. Beljaars, P. Bechtold, T. Mauritsen and G. Balsamo, 2013: Why is it so difficult to represent stably stratified conditions in numerical weather prediction (NWP) models? <i>J. Adv. Model. Earth Syst.</i>, 5, 117–133. <a href="https://doi.org/10.1002/jame.20013">doi:10.1002/jame.20013</a></li>
    <li>Sandu, I., P. Bechtold, A. Beljaars, A. Bozzo, F. Pithan, T. G. Shepherd and A. Zadra, 2016: Impacts of parameterized orographic drag on the Northern Hemisphere winter circulation. <i>J. Adv. Model. Earth Syst.</i>, 8, 196–211. <a href="https://doi.org/10.1002/2015MS000564">doi:10.1002/2015MS000564</a></li>
    <li>Seager, R., N. Harnik, Y. Kushnir, W. Robinson and J. Miller, 2003: Mechanisms of hemispherically symmetric climate variability. <i>J. Climate</i>, 16, 2960–2978. <a href="https://doi.org/10.1175/1520-0442(2003)016%3C2960:MOHSCV%3E2.0.CO;2">doi:10.1175/1520-0442(2003)016&lt;2960:MOHSCV&gt;2.0.CO;2</a></li>
    <li>Stefanick, M., 1982: Interannual atmospheric angular momentum variability 1963–1973 and the Southern Oscillation. <i>J. Geophys. Res.</i>, 87, 428–432. <a href="https://doi.org/10.1029/JC087iC01p00428">doi:10.1029/JC087iC01p00428</a></li>
    <li>Veerman, M. A., and C. C. van Heerwaarden, 2019: Trends in and closure of the atmospheric angular momentum budget in the 20th century in ERA-20C. <i>Q. J. R. Meteorol. Soc.</i>, 145, 2990–3003. <a href="https://doi.org/10.1002/qj.3600">doi:10.1002/qj.3600</a></li>
    <li>de Viron, O., and J. O. Dickey, 2014: The two types of El-Niño and their impacts on the length of day. <i>Geophys. Res. Lett.</i>, 41, 3407–3412. <a href="https://doi.org/10.1002/2014GL059948">doi:10.1002/2014GL059948</a></li>
    <li>Weickmann, K. M., 2003: Mountains, the global frictional torque, and the circulation over the Pacific–North American region. <i>Mon. Wea. Rev.</i>, 131, 2608–2622. <a href="https://doi.org/10.1175/1520-0493(2003)131%3C2608:MTGFTA%3E2.0.CO;2">doi:10.1175/1520-0493(2003)131&lt;2608:MTGFTA&gt;2.0.CO;2</a></li>
    <li>Weickmann, K. M., G. N. Kiladis and P. D. Sardeshmukh, 1997: The dynamics of intraseasonal atmospheric angular momentum oscillations. <i>J. Atmos. Sci.</i>, 54, 1445–1461. <a href="https://doi.org/10.1175/1520-0469(1997)054%3C1445:TDOIAA%3E2.0.CO;2">doi:10.1175/1520-0469(1997)054&lt;1445:TDOIAA&gt;2.0.CO;2</a></li>
    <li>Wilks, D. S., 2016: "The stippling shows statistically significant grid points": how research results are routinely overstated and overinterpreted, and what to do about it. <i>Bull. Amer. Meteor. Soc.</i>, 97, 2263–2273. <a href="https://doi.org/10.1175/BAMS-D-15-00267.1">doi:10.1175/BAMS-D-15-00267.1</a></li>
  </ol>

  <h2>Data credits</h2>
  <p class="credits">ERA5 (Hersbach et al., 2020), Copernicus Climate Change Service (C3S) / ECMWF, via the ARCO-ERA5 store on
  Google Cloud (Carver and Merose, 2023), CC BY 4.0; contains modified Copernicus Climate Change Service information 2026. Neither ECMWF nor the European Commission is
  responsible for any use of this information. CMIP6 output (licensed per model as listed by the CMIP6 data citation service, CC BY 4.0 for most) from the Pangeo CMIP6 mirror on Google
  Cloud; models: {", ".join(R["q2_cmip6"]["models"])}. Each model's historical output can be cited through the
  <a href="https://www.wdc-climate.de/ui/project?acronym=CMIP6">CMIP6 data citation service</a>. We acknowledge the World Climate Research Programme,
  which, through its Working Group on Coupled Modelling, coordinated and promoted CMIP6. We thank the climate modeling groups for
  producing and making available their model output, the Earth System Grid Federation (ESGF) for archiving the data and providing
  access, and the multiple funding agencies who support CMIP6 and ESGF. ONI and ERSSTv6 Niño-3.4: NOAA CPC and PSL. Effective angular momentum
  functions: ESMGFZ, GFZ Helmholtz Centre for Geosciences (Dobslaw and Dill 2018), via the IERS Global
  Geophysical Fluids Centre. About 400 GB of ERA5 and 450 GB of CMIP6 fields were read, in GitHub Actions.</p>
  <p class="byline">Written by Claude, Anthropic's AI model, at the direction of <a href="/about.html">Shawn Corvec</a>, meteorologist. Published 13 October 2026. Not updated since.</p>
</article>
"""
    ld = {
        "@context": "https://schema.org",
        "@type": "ScholarlyArticle",
        "headline": H1,
        "name": TITLE,
        "description": DESC,
        "url": URL,
        "inLanguage": "en",
        "datePublished": "2026-10-13",
        "dateModified": "2026-10-13",
        "author": {"@id": "https://scorvec.com/#person"},
        "image": [f"https://scorvec.com{IMG}/q1_shift_vs_events.webp"],
        "keywords": "El Niño, ENSO, mountain torque, atmospheric angular momentum, length of day, analysis increments, orographic drag, Andes, Rockies, Himalaya, friction torque, ERA5, CMIP6",
    }
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
    if "<!-- sh:start -->" in old:  # keep the stamped chrome; apply_chrome re-stamps it anyway
        page = page.replace(
            "<body>\n", "<body>\n" + old.split("<body>\n", 1)[1].split("<!-- sh:end -->", 1)[0] + "<!-- sh:end -->\n", 1
        )
    out.write_text(page)
    print("wrote", out)
    sys.path.insert(0, str(site / "scripts" / "site"))
    import apply_chrome  # (footer, rail.js, menus)

    apply_chrome.stamp_all(only={"torque-enso.html"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
