#!/usr/bin/env python3
"""Assemble the 2026-09-30 SSW precursor study into a PDF: ~/strat_reports/ssw_precursors_2026-09-30.pdf.
Numbers are read from the s1/s1b/s2/s3 outputs; the prose interprets them."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

import sswr_common as C

OUT = Path.home() / "strat_reports" / "ssw_precursors_2026-09-30.pdf"
OUT.parent.mkdir(exist_ok=True)
D = C.DATA
S1 = json.loads((D / "s1_composites.json").read_text())
R1 = json.loads((D / "s1b_r1.json").read_text())
F = json.loads((D / "s2_forward.json").read_text())
S3 = json.loads((D / "s3_structure.json").read_text())

ss = getSampleStyleSheet()
INK = colors.HexColor("#1f2328"); MUT = colors.HexColor("#6f6b64")
B = ParagraphStyle("b", parent=ss["BodyText"], fontName="Helvetica", fontSize=9.3, leading=12.6, textColor=INK, spaceAfter=5)
SM = ParagraphStyle("sm", parent=B, fontSize=7.8, leading=10, textColor=MUT)
H1 = ParagraphStyle("h1", parent=B, fontName="Helvetica-Bold", fontSize=13.5, leading=17, spaceBefore=8, spaceAfter=6)
H2 = ParagraphStyle("h2", parent=B, fontName="Helvetica-Bold", fontSize=10.8, leading=14, spaceBefore=6, spaceAfter=4)
TT = ParagraphStyle("tt", parent=B, fontName="Helvetica-Bold", fontSize=18, leading=22, spaceAfter=4)
BL = ParagraphStyle("bl", parent=B, leftIndent=12, bulletIndent=2)
CELL = ParagraphStyle("cell", parent=B, fontSize=7.6, leading=9.2, spaceAfter=0)
CELLB = ParagraphStyle("cellb", parent=CELL, fontName="Helvetica-Bold")


def P(t, st=B):
    return Paragraph(t, st)


def bullets(items):
    return [Paragraph(t, BL, bulletText="•") for t in items]


def table(rows, widths, head=1, zebra=True):
    data = [[Paragraph(str(c), CELLB if i < head else CELL) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=head)
    st = [("LINEBELOW", (0, head - 1), (-1, head - 1), 0.6, INK), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2), ("LEFTPADDING", (0, 0), (-1, -1), 3)]
    if zebra:
        for i in range(head, len(rows)):
            if (i - head) % 2:
                st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f5f4f1")))
    t.setStyle(TableStyle(st))
    return t


def fig(name, w=17.4):
    from PIL import Image as PI
    p = C.FIGS / name
    im = PI.open(p)
    h = w * im.size[1] / im.size[0]
    return Image(str(p), width=w * cm, height=h * cm)


def pv(p):
    return "&lt; 0.001" if p <= 0.001 else f"{p:.3f}"


def cl(null, key, src=S1["claims"]):
    r = src[null][key]
    s = f"{r['value']:+.2f} (p {pv(r['p'])})"
    return s + (" <b>sig</b>" if r.get("fdr10") and r.get("same_sign", True) and r["p"] < 0.05 else
                (" <b>sig</b>" if r.get("fdr10") and r.get("same_sign", True) else " n.s."))


def flag(r):
    return "<b>sig</b>" if r.get("fdr10") else "n.s."


def main():
    c = S1["claims"]
    st = []
    st.append(P("Is the strong vortex before sudden warmings real?", TT))
    st.append(P("Eligible-date composites, forward prediction on every winter day, and vortex structure, MERRA-2 1980/81–2025/26. "
                "Research report for review, 30 September 2026. Not published on the site.", SM))
    st.append(Spacer(1, 6))
    # ------------------------------------------------------------------ executive summary
    st.append(P("Executive summary", H1))
    u0a, u0b = c["N0"]["u10o strong -48..-44"], c["N0"]["u10o strong -31..-23"]
    u1a, u1b = c["N1"]["u10o strong -48..-44"], c["N1"]["u10o strong -31..-23"]
    ua = {k.split(" ")[-1]: v for k, v in c.items() if k.startswith("u10|N1|anom")}
    ta = {k.split(" ")[-1]: v for k, v in c.items() if k.startswith("Ttr|N1|anom")}
    qa = {k.split(" ")[-1]: v for k, v in c.items() if k.startswith("qbo50|N1|anom")}
    sk = F["skill"]
    st += bullets([
        f"<b>1. The strong vortex 3–7 weeks before a warming does not survive.</b> Compared with dates that have the same "
        f"westerly history (the eligible-date null), the vortex at days −48..−44 and −31..−23 is "
        f"{u1a['value']:+.2f} and {u1b['value']:+.2f} sd above its own level 60–90 days earlier (p {pv(u1a['p'])} and {pv(u1b['p'])}, "
        f"not significant), against {u0a['value']:+.2f} and {u0b['value']:+.2f} in the site's test. In absolute terms it is not strong at "
        f"all ({ua['-48..-44']['value']:+.2f} and {ua['-31..-23']['value']:+.2f} sd, p {pv(ua['-48..-44']['p'])}, {pv(ua['-31..-23']['p'])}). What the "
        f"figure shows is a recovery from an anomalously WEAK early vortex ({ua['-90..-61']['value']:+.2f} sd at days −90..−61, p {pv(ua['-90..-61']['p'])}), "
        "made to look like strength by measuring each event from that weak baseline. Part of the original effect was also the definitional "
        "artefact the question suspected: the eligibility rule alone removes a quarter to a third of it.",
        f"<b>2. The warm tropical lower stratosphere does not survive either.</b> The +{c['N0']['Ttr warm -30..-17']['value']:.2f} sd warm "
        f"spell at days −30..−17 falls to {c['N1']['Ttr warm -30..-17']['value']:+.2f} (p {pv(c['N1']['Ttr warm -30..-17']['p'])}) against "
        f"eligible dates, {c['N3']['Ttr warm -30..-17']['value']:+.2f} (p {pv(c['N3']['Ttr warm -30..-17']['p'])}) when the comparison dates are "
        "also matched on vortex strength, and is not significant after the vortex history is regressed out. In absolute terms the "
        f"tropics are COLD before warmings ({ta['-30..-17']['value']:+.2f} sd at −30..−17, {ta['-90..-61']['value']:+.2f} at −90..−61), "
        f"because warming winters sit in the easterly QBO ({qa['-90..-61']['value']:+.2f} sd, p {pv(qa['-90..-61']['p'])}; Holton–Tan); with "
        "QBO and ENSO regressed out nothing significant remains before the wave pulse. The warming TENDENCY at −43..−39 does pass "
        f"(p {pv(c['N1']['dTtr warming -43..-39']['p'])} eligible, {pv(c['N3']['dTtr warming -43..-39']['p'])} vortex-matched) but the window was "
        "picked from the figure and the same quantity has no forecast value (point 3). The cooling from day −15 is real and in step "
        "with the heat-flux pulse (both significant from day −15).",
        f"<b>3. Nothing predicts an SSW 3–6 weeks ahead.</b> On all {F['n_days']:,} eligible Nov–Mar days, leave-one-winter-out, no "
        f"model beats the seasonal climatology for an onset in 21–42 days (best BSS {max(r['vsCLIM']['bss'] for r in sk['21-42'].values()):+.3f}, "
        "none significant), and vortex strength, tropical temperature, its trend, the heat flux and its 40-day mean add nothing to "
        "QBO + ENSO. The same holds at 15–28 and 43–63 days. Skill exists only for the next two weeks: weak vortex plus strong "
        f"heat flux gives BSS {sk['1-14']['QE+U+HF']['vsCLIM']['bss']:+.2f} (AUC {sk['1-14']['QE+U+HF']['vsCLIM']['auc']:.2f}), significant.",
        "<b>4. The 'spun-up vortex becomes fragile' hypothesis is not supported.</b> Sixteen structure and geometry metrics (edge PV "
        "gradient, jet latitude and width, depth, aspect ratio, centroid, barotropic and QG instability shares, wave-1/2 refractive "
        "index) add no significant skill to strength at 3–6 weeks, alone or as strength × structure; the few significant changes are "
        "LOSSES from over-fitting. Among the 29 strong-vortex spells, the 9 followed by a warming within 63 days do not differ from the "
        "20 that were not on any metric after FDR. Before the 28 events, instability indicators rise only from days −9 to −3, i.e. with the "
        "breakdown, not before it. Splits and displacements differ only at and after onset (aspect ratio, area), which is the "
        "classification itself.",
        "<b>Proposed site change</b> (awaits approval): redraw ssw_timing.webp against eligible dates with FDR, so the vortex and "
        "tropical-temperature panels show no significant signal before day −20, and replace the caption (section 8).",
    ])
    st.append(PageBreak())
    # ------------------------------------------------------------------ methods
    st.append(P("1. Data and methods", H1))
    st.append(P("<b>Events.</b> The site's catalogue: 28 confirmed Charlton–Polvani (CP07) major warmings on MERRA-2 u(60°N, 10 hPa), "
                "1981–2026 (the marginal 2002-02-17 and 2025-11-28 reversals are excluded as on the site, but count as reversals when "
                "deciding eligibility). Types and depths from the catalogue (10 splits, 18 displacements). A second set of 40 CP07 events "
                "on NCEP R1, 1958/59–2025/26, is a robustness check."))
    st.append(P("<b>Series.</b> u(60°N, 10 hPa): MERRA-2 daily means, gap-filled from NCEP R1 plus a local offset (the site's "
                "strat_history.nc); the site figure itself used the unfilled series, which has 11 % of days missing, and both are shown. "
                "Tropical temperature: ERA5 hypsometric 100–50 hPa layer temperature, 0–15°N (local store), its centred 10-day tendency for "
                "composites and its TRAILING 10-day change for prediction. Heat flux: MERRA-2 [v′T′] at 100 hPa, 45–75°N. QBO: CPC 50 hPa "
                "index. ENSO: CPC RONI, the season centred one month earlier (no look-ahead). All series are standardised anomalies against a "
                "4-harmonic daily climatology and a smoothed day-of-year standard deviation."))
    st.append(P("<b>Structure metrics</b> (daily). From MERRA-2 zonal means (40–90°N, 1°, 100–1 hPa; m2_zmu for 20–40°N): the 10 hPa "
                "jet latitude, speed and half-maximum width; a depth ratio u(60°N, 50 hPa)/u(60°N, 10 hPa) and the upper shear "
                "u(1 hPa) − u(10 hPa); the quasi-geostrophic zonal-mean PV gradient q_y = β − (1/a²)∂_φ[(u cos φ)_φ / cos φ] − "
                "(f²/ρ)∂_z(ρ u_z/N²) (Matsuno 1970), giving the edge PV gradient (maximum over 50–85°N at 10 hPa) and the poleward-flank "
                "mean; the share of 40–85°N × 10–1 hPa where β − u_yy &lt; 0 (barotropic instability, Hartmann 1983) and where the full "
                "q_y &lt; 0 (Charney–Stern necessary condition), plus the same on the jet flanks; the stationary-wave refractive index "
                "a²n² for k = 1, 2 over 55–75°N, 50–10 hPa and the share where wave 2 can propagate. From NCEP R1 10 hPa heights: "
                "moment diagnostics (Mitchell et al. 2011; Seviour et al. 2013) of the region below the site's 30,221 m edge — centroid "
                "latitude, aspect ratio (log), area. <b>Approximations stated plainly:</b> no isentropic (Ertel) PV maps are available "
                "locally, so the edge PV gradient is the zonal-mean QG gradient, which smooths a displaced vortex's edge; no "
                "Esler–Matthewman resonance frequency is computed — their controlling parameter is vortex depth relative to the forcing, so the "
                "depth ratio and the wave-2 refractive index are offered as proxies, not as their criterion. Second derivatives are taken "
                "after a 5° latitude smoothing; upper levels (1–5 hPa) are less constrained before MLS assimilation (Aug 2004)."))
    st.append(P("<b>Q1/Q2: composites.</b> Lags −90..+40. 'Base mode' measures every event, and every comparison date, from its own "
                "day −90..−61 mean — exactly the site figure. 'Anomaly mode' uses the plain standardised anomaly. Comparison dates, "
                "one per event per Monte-Carlo draw (3,000 draws): <b>N0</b> the site's null (same calendar day ±10 d, any other winter); "
                "<b>N1 eligible</b>: same calendar day ±15 d in another MERRA-2 winter, Nov–Mar, outside any CP07 refractory period, and "
                "westerly on every one of the L days before it, L = that event's own westerly run (capped at 120 d; 26 of 28 events had "
                "≥ 66 days); <b>N2</b> N1 with no reversal within −90..+40 d; <b>N3</b> N1 matched to the event's vortex anomaly over −50..−20 "
                "(±0.3 sd). Two-sided p per lag, Benjamini–Hochberg FDR 10 % over the 131 lags (Wilks 2016), and window tests for the "
                "figure's specific claims with FDR across those windows."))
    st.append(P(f"<b>Q3/Q4: forward prediction.</b> Every Nov 1–Mar 31 day of 1980/81–2025/26 with westerly u(60°N, 10 hPa) and outside "
                f"a refractory period ({F['n_days']:,} days, {F['n_winters']} winters). Outcome: an onset in 1–14, 15–28, 21–42 (the 3–6 week "
                "window) or 43–63 days. Logistic models fitted leave-one-winter-out; the reference models are the seasonal climatology "
                "(two annual harmonics) and seasonal + QBO + ENSO. Skill on the pooled out-of-sample predictions: Brier skill score and ROC "
                "AUC. Significance: winter-block bootstrap (2,000 resamples of whole winters) of the Brier-score difference, two-sided; "
                "FDR 10 % across every model × window × reference test of each family (physical predictors; structure metrics). The "
                "daily outcomes overlap heavily, so the effective sample is the ~28 events, not the day count; the bootstrap over winters "
                "carries that."))
    # ------------------------------------------------------------------ Q1
    st.append(PageBreak())
    st.append(P("2. Q1 — the strong vortex before warmings", H1))
    st.append(fig("f1_eligible.png"))
    st.append(P("Table 1. The figure's claims under each comparison (window mean, sd; p two-sided; 'sig' = FDR 10 % across the claims "
                "of that null). Base mode unless stated.", SM))
    rows = [["claim", "site test N0", "eligible N1", "eligible + quiet N2", "vortex-matched N3", "NCEP R1, 40 events, N1"]]
    names = [("u10o strong -48..-44", "u10 strong -48..-44", "vortex strong, days −48..−44 (site series)"),
             ("u10o strong -31..-23", "u10 strong -31..-23", "vortex strong, days −31..−23 (site series)"),
             ("u10 strong -48..-44", "u10 strong -48..-44", "same, gap-filled series"),
             ("u10 strong -31..-23", "u10 strong -31..-23", "same, gap-filled series, −31..−23"),
             ("Ttr warm -30..-17", "Ttr warm -30..-17", "tropics warm, −30..−17"),
             ("dTtr warming -43..-39", "dTtr warming -43..-39", "tropics warming tendency, −43..−39"),
             ("dTtr cooling -14..-1", "dTtr cooling -14..-1", "tropics cooling, −14..−1"),
             ("vT100 -10..-1", None, "heat flux high, −10..−1")]
    for k, kr, lab in names:
        rows.append([lab] + [cl(n, k) for n in ("N0", "N1", "N2", "N3")] +
                    [cl("N1|base", kr, R1["claims"]) if kr else "—"])
    st.append(table(rows, [4.6, 2.6, 2.6, 2.6, 2.6, 2.6]))
    st.append(Spacer(1, 6))
    st.append(P("<b>Reading.</b> With the site's own series and null, both vortex windows are nominally significant (p ≈ 0.03), and "
                "some of their lags pass FDR over the 131 lags. Requiring the comparison dates to have had the same "
                "westerly history cuts the effect by about a third and removes significance (p 0.07–0.26). The gap-filled series — the "
                "better data — gives a smaller effect still. The quiet null N2 (no reversal near the comparison date) is the most "
                "favourable and makes the site series pass again (p 0.05 and 0.07, inside the FDR 10 % threshold across the claims); the "
                "longer R1 record passes at −48..−44 only (p 0.03). Note that FDR at 10 % across a family with several p &lt; 0.001 "
                "admits p up to ~0.07. The claim is therefore fragile: it passes in some combinations of series and null and fails in the primary one."))
    st.append(fig("f2_background.png"))
    st.append(P("Table 2. Absolute anomalies (no per-event baseline) against eligible dates (N1), MERRA-2.", SM))
    rows = [["series", "days −90..−61", "−48..−44", "−31..−23", "−30..−17"]]
    for k, lab in (("u10", "u(60°N, 10 hPa)"), ("u10o", "u(60°N, 10 hPa), site series"), ("Ttr", "tropical 100–50 hPa T")):
        g = {kk.split(" ")[-1]: v for kk, v in c.items() if kk.startswith(f"{k}|N1|anom")}
        rows.append([lab] + [f"{g[w]['value']:+.2f} (p {pv(g[w]['p'])})" for w in ("-90..-61", "-48..-44", "-31..-23", "-30..-17")])
    for k, lab in (("qbo50", "QBO 50 hPa"), ("roni", "RONI"), ("Ttr_noqe", "tropical T, QBO + RONI removed")):
        g = {kk.split(" ")[-1]: v for kk, v in c.items() if kk.startswith(f"{k}|N1|anom")}
        rows.append([lab, f"{g['-90..-61']['value']:+.2f} (p {pv(g['-90..-61']['p'])})", "—",
                     f"{g['-48..-23']['value']:+.2f} (p {pv(g['-48..-23']['p'])}) [−48..−23]", f"{g['-30..-17']['value']:+.2f} (p {pv(g['-30..-17']['p'])})"])
    st.append(table(rows, [4.8, 3.1, 3.1, 3.9, 3.1]))
    st.append(Spacer(1, 4))
    st.append(P("<b>Answer to Q1.</b> The vortex before a warming is not anomalously strong. It starts weak (days −90..−61, significant "
                "against eligible dates and FDR-significant at −90..−73 in Figure 2), recovers to about normal by day −60, and stays "
                "near normal until it starts to weaken significantly around day −20 (anomaly mode; day −8 in base mode). The 'strong "
                "vortex' of ssw_timing.webp is the recovery, measured from the weak baseline, plus the eligibility artefact."))
    # ------------------------------------------------------------------ Q2
    st.append(P("3. Q2 — the tropical lower stratosphere", H1))
    tr = S1["ttr_regression"]; tq = S1["ttr_qbo_enso"]
    st.append(P(f"On all Oct–Apr days the tropical layer temperature tracks the vortex: corr(u10(t−L), T(t)) is "
                f"{dict(S1['lagcorr_u_leads_T'])[0]:+.2f} at L = 0, {dict(S1['lagcorr_u_leads_T'])[20]:+.2f} at 20 days and "
                f"{dict(S1['lagcorr_u_leads_T'])[40]:+.2f} at 40 days — a strong vortex means weak wave driving and a slower, warmer "
                f"tropical upwelling branch (Yulaeva et al. 1994; Ueyama &amp; Wallace 2010). A regression on vortex history (0–60 d means) "
                f"and the 20- and 40-day heat flux explains R² {tr['r2']:.2f} of the daily tropical anomaly; QBO + RONI explain R² {tq['r2']:.2f}."))
    st += bullets([
        f"<b>The warm spell (−30..−17) is the quiet period, and not significant.</b> Against eligible dates: "
        f"{c['N1']['Ttr warm -30..-17']['value']:+.2f} (p {pv(c['N1']['Ttr warm -30..-17']['p'])}); vortex-matched: "
        f"{c['N3']['Ttr warm -30..-17']['value']:+.2f} (p {pv(c['N3']['Ttr warm -30..-17']['p'])}); after removing the vortex/heat-flux "
        f"prediction: {c['Ttr_resid N1 -30..-17']['value']:+.2f} (p {pv(c['Ttr_resid N1 -30..-17']['p'])}). R1/ERA5 with 40 events: "
        f"{R1['claims']['N1|base']['Ttr warm -30..-17']['value']:+.2f} (p {pv(R1['claims']['N1|base']['Ttr warm -30..-17']['p'])}). "
        "Matching on the vortex removes two-thirds of it: what remains is what a vortex of that strength does to the tropics.",
        "<b>In absolute terms the tropics are cold, because of the QBO.</b> Warming winters are QBO-easterly at 50 hPa and the easterly "
        "phase is cold at 100–50 hPa; with QBO and RONI regressed out, no lag before day −10 is significant (Figure 2, middle right).",
        f"<b>The warming tendency at −43..−39 passes</b> every null (N1 {cl('N1', 'dTtr warming -43..-39')}, N3 "
        f"{cl('N3', 'dTtr warming -43..-39')}; R1 {cl('N1|base', 'dTtr warming -43..-39', R1['claims'])}). It is a 5-day window chosen "
        "by eye from the figure, so its p-values are optimistic, and as a trailing predictor it carries no information about an onset "
        "in 21–42 days (Section 4). Treat it as a feature of these 28 composites, not a precursor.",
        f"<b>The cooling with the wave pulse passes strongly.</b> Heat flux and tropical cooling become significant together, on day "
        "−15 (FDR), and peak together near day −5. The site caption's 'in step with the wave pulse, not before it' holds for the "
        "cooling; its omission of the warm spell is now moot, since the warm spell does not pass.",
    ])
    # ------------------------------------------------------------------ Q3
    st.append(P("4. Q3 — forward probability on every winter day", H1))
    st.append(fig("f3_curves.png"))
    st.append(P("Figure 3 is the honest version of the question 'does a strong vortex mean a warming is coming?'. For the next two weeks the "
                "answer is strongly no (P falls from 0.27 at −1.5 sd to under 0.02 above +1 sd) — a strong vortex simply cannot reverse that "
                "fast. For 3–6 weeks P is flat at the base rate across the whole strength range (0.07–0.12, intervals overlapping), apart "
                "from the very weak bins where the vortex has usually already broken. The QBO curve is the only one with a visible 3–6 week "
                "gradient (0.24 in the most easterly bin vs 0.06 in the westerly ones), and it rests on 10–13 winters per bin."))
    st.append(fig("f4_skill.png"))
    st.append(P("Table 3. Leave-one-winter-out skill. BSS vs the seasonal climatology and vs seasonal + QBO + ENSO (QE); AUC of the model; "
                "'sig' = FDR 10 % across all 4 × 11 × up-to-3 tests; a significant negative value is an over-fitting loss.", SM))
    rows = [["window", "model", "BSS vs clim", "BSS vs QE", "AUC"]]
    for w in F["windows"]:
        for m in ("QE", "QE+U", "QE+T", "QE+dT", "QE+HF", "QE+HF40", "QE+U+T", "QE+U+dT", "QE+U+HF", "QE+U+T+dT+HF+HF40"):
            r = F["skill"][w][m]
            a = r["vsCLIM"]; b = r.get("vsQE")
            rows.append([w, m, f"{a['bss']:+.3f} (p {pv(a['p'])}) {flag(a)}",
                         f"{b['bss']:+.3f} (p {pv(b['p'])}) {flag(b)}" if b else "—", f"{a['auc']:.2f}"])
    st.append(table(rows, [1.5, 4.3, 4.3, 4.3, 1.4]))
    st.append(Spacer(1, 4))
    co = F["coef"]["21-42"]
    st.append(P(f"<b>Answer to Q3.</b> At 3–6 weeks (21–42 days) the best out-of-sample BSS is "
                f"{max(r['vsCLIM']['bss'] for r in F['skill']['21-42'].values()):+.3f} and nothing is significant: not vortex strength, not the "
                "tropical temperature or its 10-day trend, not the heat flux or its 40-day mean, and not QBO + ENSO either at daily "
                f"resolution. In-sample coefficients of the full model at 21–42 d (per sd): U {co['U']:+.2f}, T {co['T']:+.2f}, dT {co['dT']:+.2f}, "
                f"HF40 {co['HF40']:+.2f}, QBO {co['QBO']:+.2f} — all small. At 43–63 days a strong vortex with cold tropics is the most "
                f"skilful combination (BSS {F['skill']['43-63']['QE+U+T']['vsCLIM']['bss']:+.3f}, p {pv(F['skill']['43-63']['QE+U+T']['vsCLIM']['p'])}), "
                "not significant. Real skill is confined to 1–14 days: weak vortex, strong daily heat flux, and the trailing tropical "
                "cooling that comes with it."))
    # ------------------------------------------------------------------ Q4
    st.append(PageBreak())
    st.append(P("5. Q4 — does structure make a fast vortex fragile?", H1))
    st.append(fig("f5_struct_comp.png", 16.5))
    st.append(P("Before the 28 events no structure metric departs significantly from eligible dates until the breakdown itself: the "
                "edge PV gradient and the poleward-flank gradient fall from about day −15, the vortex shrinks (area) and moves off the "
                "pole (centroid) from about day −15 to −20 — which is the displacement under way — and the instability shares "
                "(q_y &lt; 0 from day −9, β − u_yy &lt; 0 from day −3) rise only with the breakdown. A few scattered single-lag hits earlier (dots or short runs at "
                "−90..−70, upper shear at −83..−26) are what 131 tests per series produce; none forms a coherent pre-conditioning signal. "
                "Hartmann's (1983) barotropic instability appears as part of the breakdown, not as its precursor."))
    st.append(fig("f6_struct_skill.png"))
    rows = [["metric", "21–42 d: + S", "21–42 d: + S + U×S", "1–14 d: + S", "1–14 d: + S + U×S"]]
    from s4_figures import NAMES
    for k in F["struct"]["21-42"]:
        cells = [NAMES[k]]
        for w in ("21-42", "1-14"):
            for t in ("S_vs_U", "UxS_vs_U"):
                r = F["struct"][w][k][t]
                cells.append(f"{r['bss']:+.3f} (p {pv(r['p'])}) {flag(r)}")
        rows.append(cells)
    st.append(P("Table 4. Skill gain of adding structure S (and strength × S) to seasonal + QBO + ENSO + strength, BSS relative to "
                "that model; FDR 10 % over 16 metrics × 4 windows × 3 tests. A significant negative value is a loss.", SM))
    st.append(table(rows, [4.0, 3.4, 3.4, 3.4, 3.4]))
    sp = S3["spells"]["test"]
    rows = [["metric", "followed (n 9)", "not followed (n 20)", "difference", "p", "FDR"]]
    for k, r in sp.items():
        rows.append([NAMES.get(k, k), f"{r['followed']:+.2f}", f"{r['not']:+.2f}", f"{r['diff']:+.2f}", pv(r["p"]), flag(r)])
    st.append(Spacer(1, 6))
    st.append(P("Table 5. The 29 strong-vortex spells (≥ 10 consecutive Nov–Feb days with u(60°N, 10 hPa) ≥ +1 sd): spell-mean "
                "standardised metrics, spells followed by a major warming within 63 days vs not; label-permutation test (10,000).", SM))
    st.append(table(rows, [4.4, 2.6, 2.9, 2.3, 1.8, 1.4]))
    sd = S3["split_disp"]
    nom = [(k, r) for k, r in sd.items() if r["p"] < 0.05]
    st.append(Spacer(1, 6))
    st.append(P("<b>Splits vs displacements</b> (10 vs 18). Of 60 metric × window tests (days −42..−21, −20..−1, 0..+10), only these reach "
                "p &lt; 0.05: " + "; ".join(f"{NAMES.get(k.split('|')[0], k.split('|')[0])} {k.split('|')[1]}: "
                                            f"{r['diff']:+.2f} sd (p {pv(r['p'])}, {flag(r)})" for k, r in nom) +
                ". Nothing in −42..−21 differs; in the last three weeks splits are nominally shallower in the lower stratosphere (lower "
                "u50/u10), more elongated and with a stronger edge PV gradient and heat flux, consistent with Albers &amp; Birner (2014) "
                "and Matthewman &amp; Esler (2011) in direction, but none passes FDR. Only the aspect ratio and area at onset — the "
                "split/displacement classification itself — pass."))
    st.append(fig("f7_cases.png"))
    cp = S3["spells"]["cases_pct"]
    keys = ["u10", "bt_neg_upper", "qg_neg_upper", "qy_pole", "qy_edge", "aspect", "cen_lat", "depth_ratio", "top_shear"]
    rows = [["case (strong spell)", "days"] + [NAMES[k] for k in keys]]
    for name, r in cp.items():
        rows.append([name, r["days"]] + [f"{100 * r[k]:.0f}" for k in keys])
    st.append(P("Table 6. Named cases: mean of each metric over the case's strong-vortex days, as a percentile of all strong-vortex "
                "days (Nov–Feb, U ≥ +1 sd). Descriptive.", SM))
    st.append(table(rows, [3.6, 0.9] + [1.42] * len(keys)))
    st.append(Spacer(1, 4))
    st.append(P("<b>Counter-examples.</b> The record-strong vortices of 2010/11 and 2019/20 were not structurally unusual for strong "
                "vortices: their instability shares sit at the 30th–55th percentile, their edge PV gradient at the 80th–87th (a sharp, "
                "intact edge) and typical centroid latitudes. The strong spells that did end in a split in Jan 2009 and Jan 2013 had "
                "barotropic instability shares at the 91st and 93rd percentile, and 2009 a very weak poleward-flank gradient (3rd) "
                "— the user's mechanism — but the strong spell of 15–31 December 2013 had a spell-mean instability share of +1.5 sd and "
                "was not followed by a warming, and the strong spells before the displacements of 1984, 2000 and 2024 had low to middling ones "
                "(28th–52nd). Across all 29 spells the instability "
                f"share is higher before warmings by {sp['bt_neg_upper']['diff']:+.2f} sd and the poleward-flank PV gradient lower by "
                f"{sp['qy_pole']['diff']:+.2f} sd, in the hypothesised direction, with p {pv(sp['bt_neg_upper']['p'])} and {pv(sp['qy_pole']['p'])} "
                "— not significant. What distinguished the cases is mostly what the vortex met afterwards (a wave pulse) rather than "
                "a measurable fragility before it."))
    st.append(P("<b>Answer to Q4.</b> No: strength × structure does not beat strength alone at 3–6 weeks, structure does not "
                "separate splits from displacements before onset after FDR, and strong vortices that later broke were not "
                "significantly different from those that did not. The two split cases are suggestive and worth keeping in mind for a "
                "larger sample (the SEAS5 hindcast and CMIP6 members), which is the only way to test a 9-vs-20 difference with power."))
    # ------------------------------------------------------------------ literature
    st.append(PageBreak())
    st.append(P("6. Literature, and where these results agree", H1))
    st += bullets([
        "<b>Charlton &amp; Polvani (2007)</b> define the events used here; their composites already show the vortex weakening only in "
        "the last few weeks. <b>Butler et al. (2017)</b>'s compendium gives ~0.6 events per winter in reanalyses — this catalogue has "
        "28 in 46 winters, consistent.",
        "<b>Polvani &amp; Waugh (2004)</b>: anomalous 100 hPa heat flux averaged over ~40 days precedes weak-vortex events. Agrees in "
        "kind: the heat flux is the only robust precursor here, but its useful lead is under two weeks; the 40-day mean adds no "
        "significant skill at any window once the seasonal cycle and QBO/ENSO are in.",
        "<b>Holton &amp; Tan (1980)</b>: easterly QBO, weaker vortex. Agrees: warming winters are QBO-easterly (−0.6 sd, significant), and "
        "that is what makes the tropical lower stratosphere look cold before events. At daily resolution QBO + ENSO alone does not "
        "beat climatology significantly (46 winters).",
        "<b>Yulaeva, Holton &amp; Wallace (1994); Ueyama &amp; Wallace (2010)</b>: extratropical wave driving sets tropical upwelling on "
        "time scales of days. Agrees: tropical cooling starts on the same day as the heat-flux pulse (−15).",
        "<b>McIntyre (1982); Albers &amp; Birner (2014)</b>: preconditioning — vortex geometry focusing wave activity before splits. "
        "Weakly consistent in direction within three weeks of onset; not significant after FDR, and absent at 3–6 weeks.",
        "<b>Esler &amp; Scott (2005); Matthewman &amp; Esler (2011); Esler &amp; Matthewman (2011)</b>: splits as a self-tuning resonance of "
        "the barotropic mode, set by vortex depth. Not tested directly (no resonance frequency computed); the depth-ratio proxy shows "
        "a nominal split/displacement difference only in the last 20 days. 'Self-tuning' implies the vortex is brought to resonance BY "
        "the forcing, which fits the absence of a long-lead structural precursor.",
        "<b>Hartmann (1983); Plumb (1981)</b>: barotropic instability of the polar-night jet and of the distorted vortex. Agrees as a "
        "breakdown mechanism: q_y &lt; 0 and β − u_yy &lt; 0 expand only from days −9 and −3. No evidence that a spun-up vortex becomes "
        "unstable on its own.",
        "<b>Holton &amp; Mass (1976); Kuroda &amp; Kodera (2001)</b>: vacillation cycles and the polar-night jet oscillation, in which a "
        "strengthening vortex precedes its breakdown. Partial: there is a recovery from an early weak state before warmings, and at "
        "43–63 days the strong-vortex coefficient is positive, but neither is significant.",
        "<b>Scott &amp; Polvani (2006)</b>: warmings arise from internal variability even under steady forcing. Consistent with the lack "
        "of predictability beyond two weeks from the vortex's own state.",
    ])
    st.append(P("References are cited from memory for orientation and should be checked before any is quoted on the site: Albers &amp; "
                "Birner 2014 JAS 71:4028; Butler et al. 2017 ESSD 9:63; Charlton &amp; Polvani 2007 J Clim 20:449; Esler &amp; Scott 2005 "
                "JAS 62:3661; Esler &amp; Matthewman 2011 JAS 68:2505; Hartmann 1983 JAS 40:817; Holton &amp; Mass 1976 JAS 33:2218; "
                "Holton &amp; Tan 1980 JAS 37:2200; Kuroda &amp; Kodera 2001 JGR 106:20703; Matsuno 1970 JAS 27:871; Matthewman &amp; "
                "Esler 2011 JAS 68:2481; McIntyre 1982 JMSJ 60:37; Mitchell, Charlton-Perez &amp; Gray 2011 JAS 68:1194; Plumb 1981 JAS "
                "38:2514; Polvani &amp; Waugh 2004 J Clim 17:3548; Scott &amp; Polvani 2006 JAS 63:2758; Seviour, Mitchell &amp; Gray "
                "2013 GRL 40:5268; Ueyama &amp; Wallace 2010 JAS 67:1232; Wilks 2016 BAMS 97:2263; Yulaeva, Holton &amp; Wallace 1994 "
                "JAS 51:169.", SM))
    # ------------------------------------------------------------------ caveats
    st.append(P("7. Caveats", H1))
    st += bullets([
        "<b>Sample.</b> 28 events (40 in R1). A composite of 28 has a standard error of ~0.2 sd for a unit-variance series, so "
        "effects of 0.2–0.3 sd — the size of every disputed signal here — are at the edge of detectability. 'Not significant' means "
        "not demonstrated, not proven absent; the split/displacement and followed/not-followed tests (10 vs 18, 9 vs 20) have little power.",
        "<b>Autocorrelation.</b> Daily composites and daily outcomes are strongly autocorrelated; all tests resample whole event "
        "dates or whole winters. The per-lag FDR treats 131 correlated lags as separate tests, which is conservative for runs.",
        "<b>Post-hoc windows.</b> The windows tested in Table 1 were read off the site figure, so even the passes there are optimistic.",
        "<b>Reanalysis changes.</b> MERRA-2 upper levels change character with MLS assimilation (Aug 2004); 11 % of MERRA-2 days are "
        "missing and filled from R1 (u) or interpolated (structure); R1 heights (2.5°) end 2026-03-17; the tropical layer temperature "
        "is ERA5 and the zonal means MERRA-2. The gap-filled and unfilled u series give composites ~0.1 sd apart at −48..−23.",
        "<b>Proxies.</b> Zonal-mean QG PV gradients and moment diagnostics on geopotential are not isentropic PV; the resonance proxy "
        "is not the Esler–Matthewman criterion. A test on isentropic PV (e.g. MERRA-2 EPV at 850 K) would need a new download.",
        "<b>Eligibility.</b> CP07 also requires the westerlies to return before 30 April; that forward-looking condition cannot be "
        "imposed on comparison dates and was not. Comparison dates may fall shortly before real events in other winters (N2 removes "
        "them; it changes little).",
    ])
    # ------------------------------------------------------------------ recommendation
    st.append(P("8. Recommended change to ssw_timing.webp (awaiting approval)", H1))
    st.append(P("Under the significance rule only these elements pass and may be drawn in colour: the heat-flux rise (significant from "
                "day −15), the vortex decline (from about day −8 measured from the event's baseline, −20 in absolute terms), the polar-cap "
                "warming, and the tropical cooling from day −15 in step with the heat flux. The vortex 'strong' windows at −48..−44 and "
                "−31..−23 and the tropical warm spell at −30..−17 do not pass the eligible-date test and must be grey. The −43..−39 "
                "warming tendency passes but is a post-hoc 5-day window with no predictive value; drawing it in colour would invite the "
                "reading this report rejects, so the recommendation is to apply FDR over lags (as the other history items do) and add one "
                "sentence about it."))
    st.append(P("<b>Proposed rebuild.</b> In build_ssw_precursors.py / the study's p8 script: (1) draw comparison dates from eligible "
                "dates (same westerly history, CP07 refractory rule, Nov–Mar, ±15 d); (2) mask colour by FDR 10 % over lags rather than "
                "p &lt; 0.05 per lag; (3) use the gap-filled u series.", B))
    st.append(P("<b>Proposed caption.</b> “Composite of the 28 major warmings on MERRA-2, 1981–2026, in standard deviations, each event "
                "measured from its own level 60–90 days before and compared with dates that had the same westerly history. Colour where "
                "the difference is significant (false discovery rate 10 % over lags); grey where it is not. The planetary-wave heat flux "
                "rises about two weeks before the central date, the vortex weakens and the tropical lower stratosphere cools in step "
                "with it. Before that, neither the vortex nor the tropics differ significantly from comparable dates: the vortex is not "
                "anomalously strong beforehand (it is weak 2–3 months before, then recovers to normal), and on all winter days neither "
                "its strength nor its structure predicts a warming three to six weeks ahead.”", B))
    st.append(P("Scripts: scripts/strat/research/ (sswr_common, s0_metrics, s1_composites, s1b_r1_check, s2_forward, s3_structure, "
                "s4_figures, s5_report). Intermediate data: ~/research/ssw_structure/data (outside the repo).", SM))

    def on_page(canv, doc):
        canv.saveState(); canv.setFont("Helvetica", 7.5); canv.setFillColor(MUT)
        canv.drawString(1.8 * cm, 1.1 * cm, "SSW precursors and vortex structure — research report, 2026-09-30 (not for publication)")
        canv.drawRightString(A4[0] - 1.8 * cm, 1.1 * cm, str(doc.page)); canv.restoreState()

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.6 * cm, bottomMargin=1.7 * cm,
                            title="SSW precursors and vortex structure", author="Shawn Corvec (analysis by Claude)")
    doc.build(st, onFirstPage=on_page, onLaterPages=on_page)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
