"""The text of model-drag-intercomparison.html (build_page.py calls body()). Every number comes from results.json;
every finding's wording depends on whether it passed the false-discovery-rate control, so the daily rebuild stays
truthful as the sample grows."""
from __future__ import annotations

import numpy as np

from build_page import E, num, sig, td, REG, RANGE, ORDER                # noqa: F401

AI = {"aigfs", "aifs-single", "aifs-ens"}
PHYS = {"gfs", "gefs", "ifs-ens", "geps"}
# WGNE Drag Project Report 1 (Zadra et al. 2013), Figs. 13-14, read off the published curves (+-0.05 x 1e19 N m)
W12 = {"pbl_land": dict(IFS=0.85, GDPS=0.52, lo=0.33, hi=0.55),
       "sgo": dict(IFS=0.2, GDPS=0.6, lo=0.4, hi=1.25),
       "total": dict(IFS=1.85, GDPS=1.9, lo=1.6, hi=2.35)}


def body(R, Mo, keys, L, fig, t_models, made):
    era = R.get("era5", {})
    first = min(Mo[k]["first"] for k in keys)
    last = max(Mo[k]["last"] for k in keys)
    n_tot = sum(Mo[k]["n_inits"] for k in keys)

    def g(k, *path, default=None):
        x = Mo.get(k)
        for p in path:
            if x is None:
                return default
            x = x.get(p) if isinstance(x, dict) else (x[p] if isinstance(x, list) and -len(x) <= p < len(x) else None)
        return x if x is not None else default

    def lead(rows, L_):
        if not rows:
            return None
        for r in rows:
            if r.get("lead") == L_:
                return r
        return None

    def last_lead(k):
        return 10 if k == "gdps" else 15

    # ---------------------------------------------------------------- conservation numbers
    def own(k, L_):
        return lead(g(k, "budget_own"), L_)

    def bulk(k, L_):
        return lead(g(k, "budget_bulk"), L_)

    def mass(k, L_, dry=False):
        return lead(g(k, "dry_mass_drift_hPa" if dry else "mass_drift_hPa"), L_)

    def drift(k, part, L_, ref="own", lev="MrC11_b"):
        src = "drift_own" if ref == "own" else "drift_era5"
        return lead(g(k, src, f"{lev}|{part}"), L_)

    def tbias(k, term, part, agg):
        return g(k, "torque_bias_agg", f"{term}|{part}", agg)

    def wind(k, wk, L_, reg, ref="own"):
        for r in g(k, "winds", f"{wk}|{ref}", default=[]):
            if r["lead"] == L_ and r["region"] == reg:
                return r
        return None

    def mean_t(k, term, part="G", when="d1"):
        return g(k, "torques", f"{term}|{part}", when)

    # ---------------------------------------------------------------- the short answer
    bl = []
    # 1 budget closure with own stress
    own_txt = []
    for k in ("gfs", "gefs", "ifs-ens"):
        if k in Mo and g(k, "budget_own"):
            r1, r10 = own(k, 1), own(k, 10)
            ef = g(k, "explained_own", 0)
            pct = "" if ef is None else (f"{ef * 100:.1f}%" if ef > 0.95 else f"{ef * 100:.0f}%")
            own_txt.append(f"{L[k]} {E(r1, 1)} Hadley on day 1 and {E(r10, 1)} on day 10"
                           + (f" (its torques explain {pct} of the day-1 variance)" if pct else ""))
    if own_txt:
        bl.append("<b>Conservation of angular momentum.</b> With the stress each physics model publishes, the residual of its "
                  "global budget (change in angular momentum minus the torques) is " + "; ".join(t for t in own_txt if t) +
                  ". GFS and GEFS publish the turbulent and the orographic gravity-wave/blocking stress; the open IFS "
                  "data carry only the turbulent part (with turbulent orographic form drag), so an IFS residual includes its "
                  "missing blocking and gravity-wave drag. ERA5 for comparison: " +
                  (E(era.get("budget", {}).get("resid_own"), 1) + " Hadley, the analysis increments" if era.get("budget") else "–") + ".")
    # 2 mass
    mt = []
    for k in keys:
        m = mass(k, 10)
        if m and m.get("mean") is not None:
            mt.append((abs(m["mean"]), k, m))
    mt.sort()
    if mt:
        best = ", ".join(f"{L[k]} {E(m, 3)}" for _, k, m in mt[:3])
        worst = ", ".join(f"{L[k]} {E(m, 3)}" for _, k, m in mt[-2:])
        bl.append(f"<b>Mass.</b> By day 10 the global-mean surface pressure has moved least in {best} hPa and most in {worst} "
                  "hPa. The physics models carry mass fixers; the AI models do not, and their surface pressure is free to drift.")
    # 3 drift vs own analysis day 10
    dr = []
    for k in keys:
        d = drift(k, "G", 10)
        if sig(d):
            dS, dN = drift(k, "S", 10), drift(k, "N", 10)
            where = []
            if sig(dS):
                where.append(f"SH {num(dS['mean'])}")
            if sig(dN):
                where.append(f"NH {num(dN['mean'])}")
            dr.append(f"{L[k]} {num(d['mean'])}" + (f" ({', '.join(where)})" if where else ""))
    nd = [L[k] for k in keys if drift(k, "G", 10) and drift(k, "G", 10).get("mean") is not None and not sig(drift(k, "G", 10))]
    if dr or nd:
        bl.append("<b>Free-forecast drift.</b> Against its own analysis valid the same day, the wind-term angular momentum "
                  "(11 common levels) at day 10 changes significantly in " + ("; ".join(dr) if dr else "none of the models") +
                  " ×10²⁵ kg m² s⁻¹" + (f"; the change is not significant in {', '.join(nd)}" if nd else "") +
                  ". For scale, ERA5's analysis adds about +0.07 ×10²⁵ a day through its increments (the earlier study), and "
                  "IFS-ENS lost −0.31 by day 15 in winter 2025–26.")
    # 4 drag against ERA5
    fb = []
    for k in keys:
        b = tbias(k, "Tfb_b", "G", "d6_10")
        if sig(b):
            fb.append(f"{L[k]} {num(b['mean'], 1)}")
    ob = []
    for k in ("gfs", "gefs", "ifs-ens"):
        b = tbias(k, "own", "G", "d1")
        if sig(b):
            ob.append(f"{L[k]} {num(b['mean'], 1)}")
    if fb or ob:
        bl.append("<b>Surface drag against ERA5.</b> " +
                  (("On days 6–10 the common bulk friction torque (which measures each model's near-surface wind, not its "
                    "drag scheme) differs significantly from ERA5's in " + "; ".join(fb) + " Hadley (negative = more "
                    "westerly drag than ERA5). ") if fb else "") +
                  (("On day 1 the models' own published stresses differ from ERA5's own (turbulent + gravity-wave) by " +
                    "; ".join(ob) + " Hadley.") if ob else ""))
    # 5 winds
    wk = []
    for k in keys:
        r = wind(k, "u850_rh", 10 if k != "gdps" else 10, "OCX_S")
        if sig(r):
            wk.append(f"{L[k]} {num(r['mean'])}")
    if wk:
        bl.append("<b>Low-level winds.</b> Over the Southern Ocean storm track (ocean 30–65°S) the 850 hPa westerlies at day 10 "
                  "differ significantly from the model's own analysis in " + "; ".join(wk) + " m s⁻¹ (negative = weaker "
                  "westerlies).")
    # 6 GEM orography
    if "gdps" in Mo and g("gdps", "orography", "scales"):
        sc = {k: g(k, "orography", "scales", "land", "rms_fine05") for k in keys}
        sc = {k: v for k, v in sc.items() if v}
        rk = sorted(sc, key=lambda k: -sc[k])
        bl.append(f"<b>GEM's 2019 orography filter.</b> On the published grids, GDPS keeps {sc.get('gdps', 0):.0f} m RMS of "
                  f"sub-0.5° relief over land, rank {rk.index('gdps') + 1} of {len(rk)} (IFS {sc.get('ifs-ens', 0):.0f} m, GFS "
                  f"{sc.get('gfs', 0):.0f} m). Its resolved mountain torque is compared range by range in section 7.")
    samp = ", ".join(L[k] + " " + str(Mo[k]["n_inits"]) for k in keys)
    bl.append(f"<b>Sample.</b> {n_tot} model-inits from {first} to {last} 00Z ({samp}). "
              "The Canadian runs are collected daily from a 30-day public archive and their sample grows every day; this page "
              f"is rebuilt daily (last rebuilt {made}).")
    lede = "<ul>" + "".join(f"<li>{x}</li>" for x in bl) + "</ul>"

    # ---------------------------------------------------------------- tables
    def row_budget(k):
        cells = "".join(td(own(k, L_), 1) if g(k, "budget_own") else "<td class='ns'>no stress</td>" for L_ in (1, 5, 10))
        cells += "".join(td(bulk(k, L_), 1) for L_ in (1, 5, 10))
        ef = g(k, "explained_bulk", 0)
        return (f"<tr><th scope='row'>{L[k]}</th>{cells}<td>{'' if ef is None else f'{ef * 100:.0f}%'}</td></tr>")
    t_budget = ("<div class='tw'><table class='res'><thead><tr><th>dM/dt − torques (Hadley)</th><th>own, day 1</th>"
                "<th>own, day 5</th><th>own, day 10</th><th>bulk, day 1</th><th>bulk, day 5</th><th>bulk, day 10</th>"
                "<th>explained, day 1 (bulk)</th></tr></thead><tbody>" + "".join(row_budget(k) for k in keys) +
                ("<tr class='diff'><th scope='row'>ERA5 (analysis days)</th>" + td(era["budget"]["resid_own"], 1) +
                 "<td></td><td></td>" + td(era["budget"]["resid_bulk"], 1) + "<td></td><td></td>"
                 f"<td>{era['budget']['explained_bulk'] * 100:.0f}%</td></tr>" if era.get("budget") else "") +
                "</tbody></table></div>")

    def row_mass(k):
        return (f"<tr><th scope='row'>{L[k]}</th>" + "".join(td(mass(k, L_), 3) for L_ in (1, 5, 10, 15)) +
                "".join(td(mass(k, L_, True), 3) for L_ in (1, 10)) + "</tr>")
    t_mass = ("<div class='tw'><table class='res'><thead><tr><th>global-mean p<sub>s</sub> change (hPa)</th><th>day 1</th>"
              "<th>day 5</th><th>day 10</th><th>day 15</th><th>dry air, day 1</th><th>dry air, day 10</th></tr></thead><tbody>" +
              "".join(row_mass(k) for k in keys) + "</tbody></table></div>")

    def row_drift(k, ref):
        return (f"<tr><th scope='row'>{L[k]}</th>" + "".join(td(drift(k, p, L_, ref)) for p in ("G", "S", "N")
                                                            for L_ in (5, 10)) + "</tr>")
    t_drift = ("<div class='tw'><table class='res'><thead><tr><th>10²⁵ kg m² s⁻¹, vs own analysis</th><th>global d5</th>"
               "<th>global d10</th><th>SH d5</th><th>SH d10</th><th>NH d5</th><th>NH d10</th></tr></thead><tbody>" +
               "".join(row_drift(k, "own") for k in keys) + "</tbody></table></div>")
    t_drifte = ("<div class='tw'><table class='res'><thead><tr><th>10²⁵ kg m² s⁻¹, vs ERA5</th><th>global d5</th>"
                "<th>global d10</th><th>SH d5</th><th>SH d10</th><th>NH d5</th><th>NH d10</th></tr></thead><tbody>" +
                "".join(row_drift(k, "era5") for k in keys) + "</tbody></table></div>")

    def row_tq(k):
        c = ""
        for term in ("Tm_b", "Tfb_b", "Tft_b", "Tg_b"):
            c += td(mean_t(k, term, "G", "d1"), 1)
        return f"<tr><th scope='row'>{L[k]}</th>{c}</tr>"
    et = era.get("torques", {})
    t_tq = ("<div class='tw'><table class='res'><thead><tr><th>day-1 global torque (Hadley)</th><th>mountain</th>"
            "<th>bulk friction</th><th>own turbulent</th><th>own GWD + blocking</th></tr></thead><tbody>" +
            "".join(row_tq(k) for k in keys) +
            (f"<tr class='diff'><th scope='row'>ERA5, same days</th>{td(et.get('Tm_b|G'), 1)}{td(et.get('Tfb_b|G'), 1)}"
             f"{td(et.get('Tft_b|G'), 1)}{td(et.get('Tg_b|G'), 1)}</tr>" if et else "") + "</tbody></table></div>")

    def row_bias(k):
        c = ""
        for term in ("Tm_b", "Tm05_b", "Tfb_b", "own"):
            for agg in ("d1", "d6_10"):
                c += td(tbias(k, term, "G", agg), 1) if tbias(k, term, "G", agg) else "<td></td>"
        return f"<tr><th scope='row'>{L[k]}</th>{c}</tr>"
    t_bias = ("<div class='tw'><table class='res'><thead><tr><th>forecast − ERA5 (Hadley)</th><th>mountain d1</th>"
              "<th>mountain d6–10</th><th>mountain 0.5° d1</th><th>mountain 0.5° d6–10</th><th>bulk friction d1</th>"
              "<th>bulk friction d6–10</th><th>own stress d1</th><th>own stress d6–10</th></tr></thead><tbody>" +
              "".join(row_bias(k) for k in keys) + "</tbody></table></div>")

    # WGNE
    W = R.get("wgne", {})

    def wv(m, mon, part):
        d = W.get(f"{m}|{mon}")
        if not d or (part != "pbl_land" and not d.get("has_sgo")):
            return "<td></td>"
        x = d[part]
        return f"<td>{x['nh_peak']:.2f} <span class='sig'>({x['nh_peak_lat']:.0f}°N, n={d['n']})</span></td>"
    wrows = ""
    for part, nm in (("pbl_land", "boundary-layer stress over land"), ("sgo", "subgrid-orography stress"),
                     ("total", "total subgrid stress, all surfaces")):
        w = W12[part]
        wrows += (f"<tr><th scope='row'>{nm}</th><td>{w['IFS']:.2f}</td><td>{w['GDPS']:.2f}</td>"
                  f"<td>{w['lo']:.2f}–{w['hi']:.2f}</td>" + wv("ifs-ens", 1, part) + wv("gfs", 1, part) +
                  wv("gefs", 1, part) + wv("era5", 1, part) + "</tr>")
    t_wgne = ("<div class='tw'><table class='res'><thead><tr><th>January, NH peak, 10¹⁹ N m</th><th>IFS 2012</th>"
              "<th>GDPS 2012</th><th>2012 high-resolution range</th><th>IFS 2026</th><th>GFS 2026</th><th>GEFS 2026</th>"
              "<th>ERA5 2026</th></tr></thead><tbody>" + wrows + "</tbody></table></div>")

    def row_wind(k, wk):
        return (f"<tr><th scope='row'>{L[k]}</th>" + "".join(td(wind(k, wk, 10, r)) for r in
                                                            ("ORO_N", "ORO_S", "OCX_N", "OCX_S", "OCO_N", "OCO_S", "LND_N", "LND_S")) + "</tr>")
    head_w = "".join(f"<th>{REG[r.split('_')[0]]}, {r.split('_')[1]}H</th>" for r in
                     ("ORO_N", "ORO_S", "OCX_N", "OCX_S", "OCO_N", "OCO_S", "LND_N", "LND_S"))
    t_wind = (f"<div class='tw'><table class='res'><thead><tr><th>850 hPa u, day 10 − own analysis (m s⁻¹)</th>{head_w}"
              "</tr></thead><tbody>" + "".join(row_wind(k, "u850_rh") for k in keys) + "</tbody></table></div>")
    t_spd = (f"<div class='tw'><table class='res'><thead><tr><th>10 m speed, day 10 − own analysis (m s⁻¹)</th>{head_w}"
             "</tr></thead><tbody>" + "".join(row_wind(k, "spd10_rh") for k in keys) + "</tbody></table></div>")

    # mountain resolution table: global mean native / 0.5 / 1.5 and fine-scale part per range
    def row_mtn(k):
        tm = [g(k, "torques", f"{t}|G", "all") for t in ("Tm_b", "Tm05_b", "Tm15_b")]
        fine = g(k, "ranges", "fine05", default=[])
        c = "".join(f"<td>{num(x, 1)}</td>" if x is not None else "<td></td>" for x in tm)
        c += "".join(td(fine[i], 1) if i < len(fine) else "<td></td>" for i in range(6))
        sc = g(k, "orography", "scales", "land", "rms_fine05")
        c += f"<td>{sc:.0f}</td>" if sc else "<td></td>"
        return f"<tr><th scope='row'>{L[k]}</th>{c}</tr>"
    emr = era.get("mountain_resolution", {})
    erow = ""
    if emr:
        er = era.get("ranges", {})
        fine_e = [a - b for a, b in zip(er.get("Tm_r", {}).get("mean", []), er.get("Tm05_r", {}).get("mean", []))]
        erow = ("<tr class='diff'><th scope='row'>ERA5 (analysis days)</th>" +
                "".join(f"<td>{num(emr[t]['mean'], 1)}</td>" for t in ("Tm_b", "Tm05_b", "Tm15_b")) +
                "".join(f"<td>{num(x, 1)}</td>" for x in fine_e) +
                f"<td>{era.get('orography', {}).get('scales', {}).get('land', {}).get('rms_fine05', 0):.0f}</td></tr>")
    t_mtn = ("<div class='tw'><table class='res'><thead><tr><th>mountain torque (Hadley)</th><th>global, own grid</th>"
             "<th>global, 0.5°</th><th>global, 1.5°</th>" + "".join(f"<th>{RANGE[r]}: own − 0.5°</th>" for r in C_RK()) +
             "<th>sub-0.5° relief over land (m RMS)</th></tr></thead><tbody>" + "".join(row_mtn(k) for k in keys) + erow +
             "</tbody></table></div>")

    # ranking table
    def rk_rows():
        out = []
        for k in keys:
            d = drift(k, "G", 10)
            m = mass(k, 10)
            w = wind(k, "u850_rh", 10, "OCX_S")
            out.append((k, d, m, w))
        return out
    rr = rk_rows()

    def rank(vals):
        v = {k: abs(x["mean"]) for k, x in vals if x and x.get("mean") is not None}
        o = sorted(v, key=lambda k: v[k])
        return {k: o.index(k) + 1 for k in o}
    r_d = rank([(k, d) for k, d, _, _ in rr])
    r_m = rank([(k, m) for k, _, m, _ in rr])
    r_w = rank([(k, w) for k, _, _, w in rr])
    t_rank = ("<div class='tw'><table class='res'><thead><tr><th>day 10 (smallest |value| = 1)</th><th>AAM drift vs own analysis</th>"
              "<th>rank</th><th>mass drift (hPa)</th><th>rank</th><th>850 hPa u, ocean 30–65°S</th><th>rank</th></tr></thead><tbody>" +
              "".join(f"<tr><th scope='row'>{L[k]}</th>{td(d)}<td>{r_d.get(k, '')}</td>{td(m, 3)}<td>{r_m.get(k, '')}</td>"
                      f"{td(w)}<td>{r_w.get(k, '')}</td></tr>" for k, d, m, w in rr) + "</tbody></table></div>")
    pairs = [p for p in R.get("paired", []) if p.get("sig")]
    pair_txt = ("; ".join(f"{L.get(p['a'], p['a'])} − {L.get(p['b'], p['b'])} {E(p)}" for p in pairs)
                if pairs else "none of the pairs")

    # per-model prose -------------------------------------------------------------------
    def drift_sentence(k):
        parts = []
        for part, nm in (("G", "globally"), ("S", "in the Southern Hemisphere"), ("N", "in the Northern Hemisphere")):
            d = drift(k, part, 10)
            if sig(d):
                parts.append(f"{nm} {E(d)}")
        if not parts:
            return f"{L[k]}: no significant day-10 change against its own analysis."
        return f"{L[k]}: day 10 " + ", ".join(parts) + "."
    drift_list = "<ul>" + "".join(f"<li>{drift_sentence(k)}</li>" for k in keys) + "</ul>"

    ev = era.get("levels", {})
    gd_rank = ""
    sc_all = {k: g(k, "orography", "scales", "land", "rms_fine05") for k in keys}
    sc_all = {k: v for k, v in sc_all.items() if v}

    def fine_range(k, r):
        f = g(k, "ranges", "fine05", default=[])
        i = C_RK().index(r)
        return f[i] if i < len(f) else None
    gem_lines = []
    for r in C_RK():
        vals = [(k, fine_range(k, r)) for k in keys]
        sigs = [f"{L[k]} {num(v['mean'], 1)}" for k, v in vals if sig(v)]
        if sigs:
            gem_lines.append(f"{RANGE[r]}: " + ", ".join(sigs))
    gem_txt = ("; ".join(gem_lines) + " Hadley") if gem_lines else "no range with a significant difference"

    days_e = era.get("days") or ["–", "–", 0]
    return f"""
<article class="study">
  <p class="crumbs"><a href="/">Home</a> › Studies › {TITLE_()}</p>
  <h1>{H1_()}</h1>
  <p class="dek">Every global forecast model has to put angular momentum into the atmosphere and take it out again through
  the ground: mountains push on the air where the surface pressure differs across them, and the surface, its roughness and
  its unresolved hills drag on the wind. How strongly a model does this, and whether its angular momentum then changes by
  exactly what its torques say, are questions about its drag and its conservation. This study runs eight global systems from
  the United States, ECMWF and Canada through one angular-momentum budget, on each model's own grid and with one arithmetic,
  against ERA5 and against the 2012 WGNE Drag Project. 00Z forecasts from {first} to {last}; the sample grows daily.</p>

  <section class="lede">
  <h2>The short answer</h2>
  {lede}
  </section>

  <h2>1. What the 2012 drag intercomparison found, and what has changed since</h2>
  <p>The WGNE Drag Project (Zadra et al., 2013; Sandu et al., 2016) asked ten models for their subgrid surface stress averaged
  over the first 24 hours of a month of forecasts, January and July 2012. Over the oceans, where all the stress comes from the
  boundary-layer scheme, the models largely agreed. Over land they did not: models of comparable resolution differed by as much
  as 20% in zonal-mean total subgrid stress, and by a factor of two to three in its split between the boundary-layer and the
  subgrid-orography schemes. IFS had the largest boundary-layer stress over mountains and the smallest orographic stress; the
  Met Office, JMA and Bureau of Meteorology models had the opposite split; the Canadian GDPS (then 25 km) and Météo-France's
  ARPEGE lay in between. The project began after the Canadian centre reported at WGNE in 2012 that adjusting its orographic
  blocking and boundary-layer schemes had improved large-scale forecasts, and Sandu et al. (2016) showed that the partition
  itself changes the circulation. Operational experience since has generally pointed the same way: more low-level drag has
  tended to improve medium-range scores.</p>
  <p>All three model families here have changed their drag since 2012:</p>
  <ul>
    <li><b>GEM (ECCC).</b> The July 2019 upgrade (GDPS 7.0.0, 25 → 15 km, 84 levels) replaced most of the physics with a package
    unified with the regional system: a revised moist-TKE boundary layer with a new mixing length and boundary-layer clouds,
    and a new thermal roughness over water (McTaggart-Cowan et al., 2019; ECCC, 2019). It also replaced the orography smoother:
    the old one damped wavelengths up to about 20 grid lengths, the new low-pass filter keeps everything longer than 3 grid
    lengths in the GDPS (Zadra, 2018; Zadra, 2022). Damping less orography moves drag from the subgrid schemes to the resolved
    mountain torque, which is why GEM's mountain torque is one of the questions below (section 7). GEPS runs the same GEM 5
    model at 25 km since June 2024 (GEPS 8.0.0), with stochastically perturbed parameters that include the strength of
    low-level flow blocking, the boundary layer's critical Richardson number and the non-orographic gravity-wave source (ECCC,
    2024). Since 26 May 2026 the GDPS is a hybrid: between about 850 and 250 hPa its winds and temperature at scales longer
    than about 2,500 km are spectrally nudged toward ECCC's AI model GEML (GDPS 10.0.0; ECCC, 2026). GEPS is not nudged.</li>
    <li><b>IFS (ECMWF).</b> Kanehama et al. (2022) rebuilt the subgrid-orography fields to represent scales below the model's
    effective resolution, weakened turbulent orographic form drag and strengthened low-level flow blocking, which raised the
    total parametrized stress in the northern mid-latitudes by 4–8% and made the momentum budget consistent across resolutions.
    Cycle 48r1 (June 2023) introduced improved orographic fields for the drag and the 9 km ensemble.</li>
    <li><b>GFS (NOAA).</b> GFS v16 (March 2021) replaced the boundary layer with a scale-aware TKE-EDMF scheme, added
    non-stationary gravity-wave drag, and went to 127 levels with a model top near 80 km (NCEP EMC, 2021). GEFS is still v12
    (September 2020), at C384 on 64 levels.</li>
  </ul>
  <p>What this study adds to the 2012 comparison: the 2026 versions; three AI models, which have no drag scheme at all and whose
  torques are whatever their pressure and wind fields imply; a free-forecast budget that asks whether each model's drag drains
  or feeds its angular momentum against its own analysis and ERA5, by lead and hemisphere; the resolved mountain torque on
  each model's own grid, the place where GEM's 2019 change should show because GEM publishes no surface stress; and day-1
  stresses in the WGNE form for the models that publish them, in January and July 2026, next to the 2012 values.</p>

  <h2>2. Data and method</h2>
  {t_models}
  <p class="tcap">Inits used so far; "own surface stress" is what the model publishes in its open data. IFS-ENS member 0 is the
  9 km deterministic run, identical in configuration to the ensemble control since cycle 48r1, which the open data publish to
  day 10. GDPS forecasts run to day 10. Raw fields are read once and reduced on the fly; nothing is stored but the diagnostics.</p>
  <ul>
    <li><b>Angular momentum</b> $M = M_r + M_\\Omega$: the wind term
    $M_r = \\frac{{a^3}}{{g}}\\int u\\cos^2\\varphi\\,d\\lambda\\,d\\varphi\\,dp$ between level mid-points, from 0 hPa to the surface
    pressure (layers below ground removed), and the mass term $M_\\Omega = \\frac{{\\Omega a^4}}{{g}}\\int p_s\\cos^3\\varphi\\,d\\lambda\\,
    d\\varphi$. The part of $M_\\Omega$ that changes only because a model's total mass changes ($\\tfrac23\\Omega a^2\\Delta m$) is
    taken out of the budget, since no torque produces it. Comparisons between models use the <b>11 levels every model
    publishes</b> (1000, 925, 850, 700, 500, 400, 300, 250, 200, 100, 50 hPa); each model's own budget uses all its levels.</li>
    <li><b>Torques</b> on the atmosphere, positive when they add westerly momentum: mountain
    $T_m = a^2\\int h\\,\\partial p_s/\\partial\\lambda\\,\\cos\\varphi\\,d\\lambda\\,d\\varphi$ from the model's <i>own</i> orography and
    surface pressure on its own published grid, and again after averaging both to 0.5° and 1.5° (Brown, 2004); friction
    $T_f = -a^3\\int \\tau_x\\cos^2\\varphi\\,d\\lambda\\,d\\varphi$ from (a) one <b>bulk law for every model</b>,
    $\\tau = C_d\\,\\rho\\,|V_{{10}}|\\,u_{{10}}$ with $C_d$ fitted per 5° cell and month to ERA5's turbulent stress 1991–2020
    (the site's torque product), and (b) the model's own published stress where there is one. The bulk law cannot see a drag
    scheme; its differences between models are differences in their 10 m wind. Torques are 6-hourly (so the daily and
    semidiurnal tides cancel) and averaged over each forecast day; angular momentum is taken at the 24-hour steps.</li>
    <li><b>References.</b> Each model's own analysis is its control member's step 0 on the verifying day; ERA5 (ARCO-ERA5,
    {days_e[0]} to {days_e[1]}) goes through the same code on 37 levels, with its own turbulent and gravity-wave stresses.
    Drift against ERA5 is the forecast-minus-ERA5 difference at day d minus the same difference at day 0.</li>
    <li><b>Regions</b> as in the torque study: rough terrain (above 1000 m, or subgrid orography standard deviation above 100 m)
    plus 500 km downstream; ocean at 30–65°; other ocean; other land; each by hemisphere. One ERA5 land-sea mask for all models.</li>
    <li><b>Statistics.</b> Members are averaged within an init, so an init is the unit. Means carry 95% moving-block bootstrap
    intervals (blocks of 3 consecutive inits, 2,000 draws); model differences are paired on the same inits; each family of tests
    (leads × hemispheres, leads × regions, ranges) is controlled with Benjamini–Hochberg at a 10% false-discovery rate. Values
    that fail it are shown as n.s. and are not findings.</li>
  </ul>

  <h2>3. Conservation: does each model's angular momentum follow its torques, and does it keep its mass?</h2>
  {fig("budget", "budget_residual.webp", "Budget residual by forecast day for each model, with the common bulk friction and with the model's own stress, and the explained fraction", "Global budget residual, dM/dt minus the torques, by forecast day: (a) mountain torque plus the common bulk friction for every model, (b) mountain torque plus the model's own published stress (GFS and GEFS: turbulent and gravity-wave/blocking; IFS: turbulent only), (c) share of the daily change explained by mountain plus bulk friction torque. Mean and 95% interval over inits; filled where significant after the FDR. Grey band: ERA5 on the same days, whose residual is its analysis increments.")}
  {t_budget}
  <p class="tcap">"own" uses the model's published stress, "bulk" the common friction law; "explained" is one minus the residual
  variance over the variance of dM/dt on day 1.</p>
  <p>A model with complete published torques should close its budget to within its numerical errors. A residual with the
  model's own stress therefore measures what its open data leave out (IFS's subgrid-orography stress; any non-orographic
  gravity-wave momentum that leaves the top; momentum fixers) plus non-conservation in the dynamics. The bulk residual adds
  the error of the common friction law, the same law for every model, so its <i>differences</i> between models are
  differences in their 10 m winds and pressure. The GDPS residual also contains the nudging toward GEML, which adds or removes
  momentum in the free troposphere without any torque.</p>
  {fig("mass", "mass_drift.webp", "Global-mean surface pressure drift by forecast day for each model, total and dry air", "Global-mean surface-pressure change since the analysis: (a) total mass, (b) dry air where the model publishes column water (GFS/GEFS precipitable water, ECMWF total column water). Mean and 95% interval over inits.")}
  {t_mass}

  <h2>4. Free-forecast drift: draining or feeding angular momentum</h2>
  <p>The torque study found ERA5's analysis putting back westerly momentum that its forecast model loses, mostly below 850
  hPa over the oceans and 83% of it in the Southern Hemisphere, and read that as boundary layers that mix too much momentum
  down. The free-forecast counterpart is a drift of each model's angular momentum away from its own analysis.</p>
  {fig("drift", "aam_drift_own.webp", "Wind-term AAM drift against the model's own analysis by forecast day, globally and by hemisphere", "Wind-term angular momentum (11 common levels), forecast minus the model's own analysis valid the same day, by lead: global, Southern and Northern Hemisphere. Mean and 95% interval over inits; filled where significant after the FDR over leads × hemispheres.")}
  {t_drift}
  {drift_list}
  {fig("drifte", "aam_drift_era5.webp", "Wind-term AAM drift against ERA5 by forecast day", "As Figure 3 but against ERA5 valid the same day, after removing each model's day-0 difference from ERA5. Shorter sample: ERA5 ends about six days before today.")}
  {t_drifte}
  {fig("levels", "level_drift.webp", "Day-10 AAM drift by pressure level and hemisphere", "Where the day-10 drift sits: the angular momentum of each of the 11 common layers, forecast minus own analysis, by hemisphere. Dots: significant after the FDR over levels × hemispheres.")}

  <h2>5. Drag: how strong, against ERA5 and against each other</h2>
  {t_tq}
  <p class="tcap">Day-1 global means over the inits (95% intervals where significant). Mountain torque swings by tens of
  Hadley from day to day with the weather; friction is steadier.</p>
  {fig("torques", "torque_bias.webp", "Global mountain, bulk friction and own-stress torques minus ERA5, day 1 and days 6 to 10", "Global torques minus ERA5's on the same valid days: mountain torque on each model's own grid, the common bulk friction, and the model's own published stress against ERA5's own (turbulent + gravity-wave). Pale: day 1; solid: days 6–10. n.s. = not significant after the FDR.")}
  {t_bias}
  <h3>Day-1 stresses in the WGNE form</h3>
  <p>The 2012 project plotted the zonal-mean stress times $a^3\\cos^2\\varphi$ (N m) averaged over the first 24 hours of each 00Z
  forecast. The same quantity for January and July 2026, from the models that publish their stress, puts the 2026 models on the
  2012 axes:</p>
  {fig("wgne", "wgne_day1.webp", "Zonal-mean day-1 surface stress over land, orographic stress and total, January and July 2026, with the 2012 peak values", "Day-1 zonal-mean subgrid stress in the WGNE form, January (top) and July 2026 (bottom): boundary-layer stress over land, subgrid-orography stress, and the total over all surfaces. Grey box and coloured bars: the 2012 northern peak values read off the WGNE report's Figure 13 (January 2012). IFS publishes no orographic stress.")}
  {t_wgne}
  <p class="tcap">2012 values read off Figure 13 of the WGNE Drag Project report (Zadra et al., 2013), January 2012, first
  24 h, to about ±0.05 ×10¹⁹ N m; "high-resolution range" spans the eight models other than CPTEC's two AGCMs. 2026: mean
  of the 00Z forecasts of the month, northern peak between 25 and 65°N. The 2012 GDPS and IFS are not today's models.</p>

  <h2>6. Low-level winds: do they weaken or strengthen as the forecast runs?</h2>
  <p>In the torque study ERA5's analysis weakened both the trades and the mid-latitude surface westerlies. Here the same
  regions show how each model's low-level winds move away from its own analysis by day 10.</p>
  {fig("winds", "wind_drift.webp", "Day-10 drift of 10 m wind speed and 850 hPa zonal wind by region and model", "Day-10 forecast minus the model's own analysis, averaged over each region: 10 m wind speed (left) and 850 hPa zonal wind (right). Coloured where significant after the FDR over regions × leads.")}
  {t_wind}
  {t_spd}
  <p class="tcap">Trade-wind regions have easterly (negative) u: there a positive change means weaker trades. Ocean 30–65° has
  westerlies: a negative change means weaker westerlies.</p>

  <h2>7. Mountain torque, resolution and GEM's 2019 orography filter</h2>
  <p>GEM publishes no surface stress, so its drag can only be seen through its winds and its resolved mountain torque. A
  filter that keeps more small-scale orography should give a larger resolved torque at fine scales: averaging $p_s$ and $h$ to
  0.5° removes exactly that part. The table gives each model's mean torque on its own grid and at 0.5° and 1.5°, the
  fine-scale part (own grid minus 0.5°) by range, and how much sub-0.5° relief its orography keeps.</p>
  {fig("mountain", "mountain_resolution.webp", "Mean mountain torque by range on each model's grid and at 0.5 and 1.5 degrees", "Mean resolved mountain torque, days 1–10, by range: each model's own published grid (solid), and with surface pressure and orography averaged together to 0.5° (mid) and 1.5° (pale). ERA5 on its analysis days.")}
  {fig("oro", "orography_scales.webp", "Sub-0.5-degree relief kept by each model's orography, by range", "RMS of each model's orography minus its 0.5° area mean, by range and over all land, as published on its output grid.")}
  {t_mtn}
  <p>Significant fine-scale (own grid minus 0.5°) mountain torque, FDR over ranges: {gem_txt}. The published grids are
  interpolations of the models' native grids (0.15° for GDPS, 0.25° for the ECMWF models and AIGFS, 0.5° for GEFS and GEPS; only
  GFS's torque fields are on its own Gaussian physics grid), so a fine-scale difference is the model's orography as seen at
  the published resolution, not its native one. For ERA5 the 37-level wind term is on average
  {num(ev.get('mean_37_minus_C11'), 2)} ×10²⁵ larger than on the 11 common levels, and their daily tendencies differ with an
  sd of {(ev.get('sd_daily_tendency_diff_H') or 0):.1f} Hadley (correlation {(ev.get('corr_tendency') or 0):.3f}): level
  sampling shifts the level of angular momentum much more than its change.</p>

  <h2>8. Ranking</h2>
  {t_rank}
  <p class="tcap">Smallest absolute day-10 value ranks first. A rank is only meaningful where the paired difference is
  significant; of the paired day-10 AAM drift differences on common inits, significant after the FDR: {pair_txt}.</p>

  <h2>9. What is not like for like</h2>
  <ul>
    <li><b>Resolution and grids.</b> 9 to 31 km native grids, published on 0.15° to 0.5° grids by interpolation (GFS's torque fields
    on its own Gaussian grid). Mountain torque is resolution-sensitive by tens of Hadley (Brown, 2004; the 0.5° and 1.5°
    columns); the friction torque is not much.</li>
    <li><b>Vertical sampling.</b> 12 to 41 published levels. The comparison uses the 11 common levels, which change the size of
    angular momentum more than its change (section 7). Below-ground layers are removed by each model's own surface pressure.</li>
    <li><b>Stress definitions.</b> GFS's UFLX is its turbulent stress and U-GWD its orographic gravity-wave and blocking stress;
    IFS's ewss includes turbulent orographic form drag but not blocking or gravity-wave drag; ERA5's turbulent stress includes
    form drag and its gravity-wave stress includes blocking. 2012 WGNE "PBL" and "SGO" follow each model's own split.</li>
    <li><b>The bulk friction law</b> was fitted to ERA5; applied to another model it measures that model's 10 m wind, not its drag.</li>
    <li><b>Sample and season.</b> Autumn 2026 (September–October), with El Niño developing; GEM inits only from 14 September and,
    for GEPS, every second day. Winter results can differ (the earlier IFS test was in December–February).</li>
    <li><b>GDPS is a hybrid since May 2026</b>: its large scales between 850 and 250 hPa follow an AI model, so its drift and
    budget are not GEM's physics alone. GEPS is the cleaner measure of GEM's physics.</li>
    <li><b>Members.</b> GEPS members perturb drag parameters by design; IFS-ENS member 0 stops at day 10; ensembles use control + 4.</li>
  </ul>

  <h2>10. How this fits the earlier studies</h2>
  <p><a href="/aifs-aam-budget.html">Do AI weather models conserve angular momentum?</a> tested AIFS-ENS, AIFS and IFS-ENS in
  winter 2025–26 with the same budget; <a href="/torque-enso.html">Why the atmosphere spins up in El Niño</a> measured ERA5's
  analysis increments and proposed that models' boundary layers mix too much westerly momentum down over the oceans. This study
  extends the free-forecast side of that test to the US and Canadian systems and puts their drag next to the 2012 comparison.</p>

  <h2>References</h2>
  <ol class="refs">
    <li>Beljaars, A. C. M., A. R. Brown and N. Wood, 2004: A new parametrization of turbulent orographic form drag. <i>Q. J. R. Meteorol. Soc.</i>, 130, 1327–1347.</li>
    <li>Benjamini, Y., and Y. Hochberg, 1995: Controlling the false discovery rate. <i>J. R. Stat. Soc. B</i>, 57, 289–300.</li>
    <li>Brown, A. R., 2004: Resolution dependence of orographic torques. <i>Q. J. R. Meteorol. Soc.</i>, 130, 3029–3046.</li>
    <li>ECCC, 2019: Global Deterministic Prediction System (GDPS) version 7.0.0, technical note. Canadian Meteorological Centre. <a href="https://collaboration.cmc.ec.gc.ca/cmc/cmoi/product_guide/docs/tech_notes/technote_gdps-700_e.pdf">PDF</a></li>
    <li>ECCC, 2024: Global Ensemble Prediction System (GEPS) version 8.0.0, technical note. <a href="https://collaboration.cmc.ec.gc.ca/cmc/cmoi/product_guide/docs/tech_notes/technote_geps-800_e.pdf">PDF</a></li>
    <li>ECCC, 2026: Global Deterministic Prediction System (GDPS) version 10.0.0, technical note (spectral nudging toward GEML). <a href="https://collaboration.cmc.ec.gc.ca/cmc/cmoi/product_guide/docs/tech_notes/technote_gdps-1000_e.pdf">PDF</a></li>
    <li>ECMWF, 2023: Implementation of IFS cycle 48r1. <a href="https://confluence.ecmwf.int/display/FCST/Implementation+of+IFS+Cycle+48r1">ECMWF</a></li>
    <li>Egger, J., K. Weickmann and K.-P. Hoinka, 2007: Angular momentum in the global atmospheric circulation. <i>Rev. Geophys.</i>, 45, RG4007.</li>
    <li>Hersbach, H., and co-authors, 2020: The ERA5 global reanalysis. <i>Q. J. R. Meteorol. Soc.</i>, 146, 1999–2049.</li>
    <li>Kanehama, T., I. Sandu, A. Beljaars, A. van Niekerk, N. Wedi, S. Boussetta, S. Lang, S. Johnson and L. Magnusson, 2022: Evaluation and optimization of orographic drag in the IFS. <i>ECMWF Tech. Memo.</i> 893. <a href="https://doi.org/10.21957/fps6gngqe">doi:10.21957/fps6gngqe</a></li>
    <li>McTaggart-Cowan, R., et al., 2019: Modernization of atmospheric physics parameterization in Canadian NWP. <i>J. Adv. Model. Earth Syst.</i>, 11, 3593–3635. <a href="https://doi.org/10.1029/2019MS001781">doi:10.1029/2019MS001781</a></li>
    <li>NCEP EMC, 2021: GFS v16 implementation. <a href="https://www.emc.ncep.noaa.gov/users/impldocs/GFS.v16_Implementation.pdf">PDF</a></li>
    <li>Sandu, I., P. Bechtold, A. Beljaars, A. Bozzo, F. Pithan, T. G. Shepherd and A. Zadra, 2016: Impacts of parameterized orographic drag on the Northern Hemisphere winter circulation. <i>J. Adv. Model. Earth Syst.</i>, 8, 196–211. <a href="https://doi.org/10.1002/2015MS000564">doi:10.1002/2015MS000564</a></li>
    <li>Sandu, I., and co-authors, 2019: Impacts of orography on large-scale atmospheric circulation. <i>npj Clim. Atmos. Sci.</i>, 2, 10.</li>
    <li>van Niekerk, A., and co-authors, 2020: Constraining Orographic Drag Effects (COORDE): a model comparison of resolved and parametrized orographic drag. <i>J. Adv. Model. Earth Syst.</i>, 12, e2020MS002160.</li>
    <li>Wilks, D. S., 2016: "The stippling shows statistically significant grid points". <i>Bull. Amer. Meteor. Soc.</i>, 97, 2263–2273.</li>
    <li>Zadra, A., 2018: Notes on the new low-pass filter for the orography field. Internal report, published with the WGNE Drag Project. <a href="https://collaboration.cmc.ec.gc.ca/science/rpn/drag_project/documents/topo_lowpass_filter.pdf">PDF</a></li>
    <li>Zadra, A., 2022: Challenges representing orographic drag in (Canadian) numerical prediction systems. ECMWF Annual Seminar 2022. <a href="https://events.ecmwf.int/event/300/contributions/3292/attachments/1952/3525/AS2022_Zadra.pdf">PDF</a></li>
    <li>Zadra, A., and contributors, 2013: WGNE Drag Project: an inter-model comparison of surface stresses. Report no. 1. <a href="https://collaboration.cmc.ec.gc.ca/science/rpn/drag_project/documents/wgne_drag_project_report01.pdf">PDF</a>; project pages: <a href="https://collaboration.cmc.ec.gc.ca/science/rpn/drag_project">WGNE Drag Project</a></li>
  </ol>

  <h2>Data credits</h2>
  <p class="credits">GFS, GEFS and AIGFS: NOAA/NCEP, via the NOAA Open Data Dissemination programme on AWS. IFS, AIFS and
  AIFS-ENS: ECMWF open data (CC BY 4.0), Google Cloud mirror. GDPS and GEPS: Environment and Climate Change Canada, MSC Datamart;
  contains information licensed under the Open Government Licence – Canada. ERA5 (Hersbach et al., 2020), Copernicus Climate
  Change Service (C3S) / ECMWF, via the ARCO-ERA5 store on Google Cloud; contains modified Copernicus Climate Change Service
  information 2026. Neither ECMWF nor the European Commission is responsible for any use of this information. All downloads ran in
  GitHub Actions; only the reduced diagnostics are kept (<a href="https://github.com/scorvec/scorvec.github.io/tree/main/scripts/amdrag">scripts/amdrag</a>).</p>
  <p class="byline">By <a href="/about.html">Shawn Corvec</a>, meteorologist. Published 13 October 2026; rebuilt daily as the
  sample grows (last rebuilt {made}).</p>
</article>
"""


def C_RK():
    return ["HIM", "ROC", "AND", "GRL", "ANT", "ALP"]


def TITLE_():
    from build_page import TITLE
    return TITLE


def H1_():
    from build_page import H1
    return H1
