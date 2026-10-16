/* seas5-cards.js — the SEAS5 member-index cards, shared between pages.
   seasonal.html draws the teleconnection card with indexCard(); stratosphere.html draws the SEAS5 stratosphere
   cards (#stratCard, #sswCard, #polar-stratosphere) with indexCard() + stratFigs(), polarCard() and sswCard() from
   the same JSON the seasonal page reads (assets/sst/data/seas5_tele.json, seas5_outlook.json, seas5_ssw.json).
   Every card function takes its element ids as arguments.
   Needs Plotly and /assets/outlook-charts.js (window.OutlookCharts) at draw time; ensure(cb) loads both. */
(function () {
  "use strict";
  var MON = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
  function $(id) { return document.getElementById(id); }
  function OC() { return window.OutlookCharts; }
  function monthDate(ym) { return ym + "-15"; }
  function mmm(ym) { return MON[parseInt(ym.slice(5), 10) - 1] + " " + ym.slice(0, 4); }
  function issueLabel(ym) { return mmm(ym.slice(0, 4) + "-" + ym.slice(4)); }
  function fmt(v, d) { return v == null ? "—" : (v >= 0 ? "+" : "") + v.toFixed(d == null ? 2 : d); }
  function pills(elId, keys, cur, onpick) {
    var el = $(elId); el.innerHTML = "";
    keys.forEach(function (k) {
      var b = document.createElement("button"); b.className = "pill" + (k.key === cur ? " on" : ""); b.textContent = k.label;
      b.addEventListener("click", function () { onpick(k.key); Array.prototype.forEach.call(el.children, function (c) { c.classList.toggle("on", c === b); }); });
      el.appendChild(b);
    });
  }

  // Plotly (cartesian bundle, the one the outlook pages use) and the outlook chart theme, loaded once on demand
  var waiting = null;
  function loadScript(src, cb) { var s = document.createElement("script"); s.src = src; s.onload = cb; s.onerror = cb; document.head.appendChild(s); }
  function ensure(cb) {
    if (window.Plotly && window.OutlookCharts) return cb();
    if (waiting) { waiting.push(cb); return; }
    waiting = [cb];
    var done = function () { var q = waiting; waiting = null; q.forEach(function (f) { f(); }); };
    var oc = function () { if (window.OutlookCharts) done(); else loadScript("/assets/outlook-charts.js", done); };
    if (window.Plotly) oc(); else loadScript("https://cdn.jsdelivr.net/npm/plotly.js-cartesian-dist-min@2.35.2/plotly-cartesian.min.js", oc);
  }

  // generic member-index plume for the teleconnection / stratosphere JSON
  var IDX_MODE = { cal: true };   // calibrated view first where the hindcast regression exists
  function hasCal(e) { return !!(e.cal && e.cal.some(function (c) { return c && c.mu != null && c.s != null && !c.no_skill; })); }
  function idxTraces(T, key) {
    var oc = OC(), e = T.indices[key], pe = T.previous_indices && T.previous_indices[key], x = e.valid.map(monthDate), tr = [];
    var absolute = !!e.absolute, base = absolute ? e.clim_mean : null, hv = { units: e.units, digits: 2, signed: !absolute };
    var calOn = IDX_MODE.cal && hasCal(e) && !absolute;
    tr = tr.concat(oc.climBand(x, e.clim_sd, base, absolute ? "hindcast mean ±1σ" : "hindcast ±1σ"));
    if (absolute) tr.push(oc.hindMeanTrace(x, e.clim_mean, "hindcast mean"));
    if (pe) tr = tr.concat(oc.prevTraces([{ x: pe.valid.map(monthDate), y: pe.mean, name: issueLabel(T.previous) + " issue mean (raw)" }], hv));
    if (calOn) {
      // calibrated: the regression's predictive band per lead from the 24 hindcast years (c.q = its 10/25/75/90 %
      // quantiles), climatology where the skill test fails; a file without c.q has the older Gaussian N(mu, s²)
      var mu = e.cal.map(function (c, i) { return c && c.mu != null && (c.q || !c.no_skill) ? c.mu : e.mean[i]; }), sd = e.cal.map(function (c) { return c && c.s != null ? c.s : null; });
      var q = function (z, k) { return mu.map(function (m, i) { var c = e.cal[i]; return c && c.q ? c.q[k] : sd[i] == null ? null : m + z * sd[i]; }); };
      tr = tr.concat(oc.fanTraces(x, { p10: q(-1.2816, 0), p90: q(1.2816, 3), p25: q(-0.6745, 1), p75: q(0.6745, 2) }));
      tr.push(oc.meanTrace(x, mu, issueLabel(T.issue) + " issue, calibrated to observations", hv));
      tr.push({ x: x, y: e.mean, mode: "lines", line: { color: "#8a4b2a", width: 1.6, dash: "dot" }, name: "raw model mean", hovertemplate: "%{y:+.2f}<extra>raw model mean</extra>" });
    } else {
      tr = tr.concat(oc.fanTraces(x, e));
      if (e.members) tr.push(oc.memberTrace(x, oc.memberRows(e.members, x.length)));
      tr.push(oc.meanTrace(x, e.mean, issueLabel(T.issue) + " issue mean", hv));
    }
    if (e.obs_tail && e.obs_tail.length) tr.push(oc.obsTrace(e.obs_tail.map(function (r) { return monthDate(r[0]); }), e.obs_tail.map(function (r) { return r[1]; }), e.obs_source === "CPC" ? "observed (CPC index)" : "observed (ERA5, same projection)", hv));
    if (e.cpc_tail && e.cpc_tail.length && e.obs_source !== "CPC") tr.push(oc.markerTrace(e.cpc_tail.map(function (r) { return monthDate(r[0]); }), e.cpc_tail.map(function (r) { return r[1]; }), "published index (CPC), its own scale", hv));
    if (key.indexOf("u60") === 0) tr.push({x: [x[0], x[x.length - 1]], y: [0, 0], mode: "lines", line: {color: "#c2185b", width: 1.4, dash: "dash"}, name: "easterly below", hoverinfo: "skip"});
    return tr;
  }
  function idxRange(T, key) {
    var e = T.indices[key];
    var tails = [].concat(e.obs_tail || [], e.cpc_tail || []).map(function (r) { return r[0]; }).sort();
    return OC().monthRange(e.valid, tails);
  }
  function idxTable(T, key) {
    var e = T.indices[key], pe = T.previous_indices && T.previous_indices[key];
    var thr = T.thr || 0.5, hasCal = !!(e.cal && e.cal.some(function (c) { return c && c.p_neg != null; }));
    var head = '<tr><th>Month</th><th>Mean</th><th>P10–P90</th>' + (e.p_easterly ? '<th>P(easterly)</th>' : (e.p_neg05 ? '<th>Members ≤ −' + thr + 'σ / ≥ +' + thr + 'σ</th>' : '<th>P(+)</th><th>P(−)</th>')) + (hasCal ? '<th>Calibrated ≤ −' + thr + 'σ / ≥ +' + thr + 'σ</th>' : '') + (pe ? '<th>' + issueLabel(T.previous) + ' mean</th>' : '') + '<th>Skill r</th></tr>';
    var rows = e.valid.map(function (v, i) {
      if (e.mean[i] == null) return '<tr><td>' + mmm(v) + '</td><td colspan="7" class="flat">not defined by CPC in this month</td></tr>';
      var pv = pe ? pe.valid.indexOf(v) : -1, sk = e.skill && e.skill[i] != null ? e.skill[i].toFixed(2) : "—", c = e.cal ? e.cal[i] : null;
      var skc = e.skill && e.skill[i] != null && c && !c.no_skill ? (e.skill[i] >= 0.5 ? "up" : "") : "flat";
      var pct = function (x) { return x == null ? "—" : Math.round(x * 100) + "%"; };
      return '<tr><td>' + mmm(v) + '</td><td>' + fmt(e.mean[i], 2) + '</td><td>' + fmt(e.p10[i], 2) + ' … ' + fmt(e.p90[i], 2) + '</td>' +
        (e.p_easterly ? '<td>' + pct(e.p_easterly[i]) + '</td>' : (e.p_neg05 ? '<td>' + pct(e.p_neg05[i]) + ' / ' + pct(e.p_pos05[i]) + '</td>' : '<td>' + pct(e.p_pos[i]) + '</td><td>' + pct(e.p_neg[i]) + '</td>')) +
        (hasCal ? '<td class="' + (c && c.no_skill ? "flat" : "") + '">' + (c && c.p_neg != null ? pct(c.p_neg) + ' / ' + pct(c.p_pos) + (c.no_skill ? (c.q ? " (no skill: climatology)" : " (no skill)") : "") : "—") + '</td>' : '') +
        (pe ? '<td>' + (pv >= 0 && pe.mean[pv] != null ? fmt(pe.mean[pv], 2) : "—") + '</td>' : '') + '<td class="' + skc + '">' + sk + '</td></tr>';
    }).join("");
    return head + rows;
  }
  function indexCard(T, group, pillsId, noteId, chartId, tableId, updId, first) {
    var keys = T.groups[group]; if (!keys || !keys.length) { $(chartId).closest(".card").hidden = true; return; }
    var cur = first && keys.indexOf(first) >= 0 ? first : keys[0];
    var modeId = pillsId + "Mode", modeEl = $(modeId);
    if (!modeEl) { modeEl = document.createElement("div"); modeEl.className = "controls"; modeEl.id = modeId; $(pillsId).insertAdjacentElement("afterend", modeEl); }
    function draw(k) {
      var e = T.indices[k];
      $(noteId).textContent = e.note + (T.skill_note ? "  Skill r: " + T.skill_note + "." : "") + (hasCal(e) && IDX_MODE.cal ? "  Calibrated view: the observed index regressed on the hindcast ensemble mean for this month and lead (24 years); mean and spread from that fit" + (e.obs_source === "CPC" ? ", against CPC's published index." : ".") : "");
      OC().react(chartId, idxTraces(T, k), { title: e.label + (hasCal(e) && IDX_MODE.cal && !e.absolute ? " (calibrated)" : ""), yunits: e.units, xrange: idxRange(T, k), includeZero: !e.absolute || /^(u60|qbo)/.test(k) });
      $(tableId).innerHTML = idxTable(T, k);
      modeEl.hidden = !hasCal(e) || !!e.absolute;
      modeEl.innerHTML = modeEl.hidden ? "" : '<span class="ctlsep">View</span><button class="pill' + (IDX_MODE.cal ? " on" : "") + '" data-mode="cal">Calibrated to observations</button><button class="pill' + (!IDX_MODE.cal ? " on" : "") + '" data-mode="raw">Raw model members</button>';
      modeEl.onclick = function (ev) { var b = ev.target.closest("button"); if (!b) return; IDX_MODE.cal = b.dataset.mode === "cal"; draw(k); };
    }
    pills(pillsId, keys.map(function (k) { return {key: k, label: T.indices[k].label}; }), cur, draw);
    draw(cur);
    $(updId).textContent = "built " + T.generated;
  }
  function fig(file, v, alt, cap) { return '<figure><a href="assets/sst/' + file + v + '" target="_blank" rel="noopener"><img src="assets/sst/' + file + v + '" alt="' + alt + '" loading="lazy"></a><figcaption>' + cap + '</figcaption></figure>'; }
  function figs(T) { return { F: T.figures || {}, v: "?v=" + encodeURIComponent(T.generated) }; }
  // the teleconnection figures (seasonal.html) and the stratosphere one (stratosphere.html)
  function teleFigs(T, hostId) {
    var f = figs(T), F = f.F, t = "";
    if (F.tele_probs) t += fig(F.tele_probs, f.v, "SEAS5 teleconnection phase probabilities by month", "<b>Phase probabilities by month.</b> Upper bar: the share of the 51 members at or beyond ±" + (T.thr || 0.5) + "σ. Thin lower bar: the same after hindcast calibration. " + (T.skill_test ? "Hatched cells have no significant hindcast skill (one-sided " + Math.round(T.skill_test.alpha * 100) + " % test over the 24 hindcast years: r below " + T.skill_test.r_crit_n24 + "); their lower bar is climatology." : "Hatched cells have no usable hindcast skill (r below " + (T.skill_min || 0.25) + ")."));
    if (F.tele_series) t += fig(F.tele_series, f.v, "SEAS5 teleconnection indices with the observed ERA5 tail", "<b>Each index against its observed record.</b> Black: ERA5 through the latest month available; red crosses: CPC's published index; bands and navy line: this issue; dashed: last month's. Small r values are the hindcast skill for each month.");
    $(hostId).innerHTML = t;
  }
  function stratFigs(T, hostId) {
    var f = figs(T), F = f.F;
    $(hostId).innerHTML = F.strat_series ? fig(F.strat_series, f.v, "SEAS5 stratosphere and annular-mode indices with the observed ERA5 tail", "<b>Vortex, QBO, NAM by level and 100 hPa waves</b> against ERA5 where an observed record is available (the 10–50 hPa wind panels have no observed tail yet).") : "";
  }

  // SEAS5 polar-cap heights: J = seas5_outlook.json
  function polarCard(J, ids) {
    var st = { hemi: "nh", lev: "100" };
    function draw() {
      var oc = OC(), key = st.hemi + "_z" + st.lev, e = J.polar && J.polar[key]; if (!e) { $(ids.chart).textContent = "no data for " + key; return; }
      var valid = J.indices[J.issue].valid, x = valid.map(monthDate), tr = [], hv = { units: "m", digits: 0, signed: true };
      tr = tr.concat(oc.climBand(x, e.clim_sd, null, "hindcast ±1σ"));
      var pe = J.polar_previous && J.polar_previous[key];
      if (pe) tr = tr.concat(oc.prevTraces([{ x: pe.valid.map(monthDate), y: pe.mean, name: issueLabel(J.previous) + " issue mean" }], hv));
      tr = tr.concat(oc.fanTraces(x, e));
      tr.push(oc.memberTrace(x, oc.memberRows(e.members, x.length)));
      tr.push(oc.meanTrace(x, e.mean, issueLabel(J.issue) + " issue mean", hv));
      var title = (st.hemi === "nh" ? "Northern" : "Southern") + " polar cap, " + st.lev + " hPa height anomaly";
      oc.react(ids.chart, tr, { title: title, yunits: "m", xrange: oc.monthRange(valid, []) });
    }
    var el = $(ids.pills); el.innerHTML = "";
    [{key: "nh", label: "Northern cap"}, {key: "sh", label: "Southern cap"}].forEach(function (k) { var b = document.createElement("button"); b.className = "pill" + (k.key === st.hemi ? " on" : ""); b.textContent = k.label; b.dataset.hemi = k.key; el.appendChild(b); });
    var sep = document.createElement("span"); sep.className = "ctlsep"; sep.textContent = "·"; el.appendChild(sep);
    ["100", "50", "10"].forEach(function (lv) { var b = document.createElement("button"); b.className = "pill" + (lv === st.lev ? " on" : ""); b.textContent = lv + " hPa"; b.dataset.lev = lv; el.appendChild(b); });
    el.addEventListener("click", function (ev) {
      var b = ev.target.closest("button"); if (!b) return;
      if (b.dataset.hemi) { st.hemi = b.dataset.hemi; } else { st.lev = b.dataset.lev; }
      Array.prototype.forEach.call(el.querySelectorAll("button"), function (c) { c.classList.toggle("on", (c.dataset.hemi && c.dataset.hemi === st.hemi) || (c.dataset.lev && c.dataset.lev === st.lev)); });
      draw();
    });
    $(ids.updated).textContent = "built " + J.generated;
    draw();
  }

  /* SEAS5 polar vortex and SSW risk (scripts/sst/seas5_ssw.py). Every claim is
     printed with its test: the risk index is called skilful only when the hindcast BSS 90 % interval excludes zero.
     J = seas5_ssw.json, or null before its first build. */
  function pc(x) { return x == null ? "–" : Math.round(100 * x) + "%"; }
  function f2(x) { return x == null ? "n/a" : x.toFixed(2); }
  function fp(x) { return x == null ? "n/a" : x < 0.001 ? "< 0.001" : x.toFixed(3); }
  // a BSS interval wholly below zero is a significant LOSS against climatology, not "no skill"
  function bssWord(s) { return s.skilful ? "skill beyond climatology" : (s.bss_ci90 && s.bss_ci90[1] != null && s.bss_ci90[1] < 0) ? "worse than climatology" : "no skill beyond climatology"; }
  function sswCard(J, ids) {
    if (!J || !J.windows) {                       // no build yet: say so instead of leaving a dead menu link
      $(ids.verdict).innerHTML = "The first build of this product is still running: the daily 10&nbsp;hPa winds of all 51 October members come from the Copernicus data store, where requests queue for hours. The risk index, its skill test and the charts appear here once that run finishes.";
      return;
    }
    var W = J.windows, A = W["Nov–Mar"], S = A.skill, E = J.enso || {}, v = "?v=" + encodeURIComponent(J.generated);
    var sig = A.p_vs_model_base < 0.05;
    // the lack of skill is stated first: a plain banner ahead of the number whenever the hindcast shows no skill
    var t = S.skilful ? "" : '<div class="chipnote" style="margin:0 0 0.8rem;max-width:none"><b>No track record: read this as the model&rsquo;s view, not a forecast.</b> ' +
      "In SEAS5&rsquo;s own hindcast the share of members with a sudden warming " + (bssWord(S) === "worse than climatology" ? "has done significantly worse than simply quoting the observed SSW rate every year" : "has not told SSW winters from quiet ones") + " (Brier skill score " + f2(S.bss) + ", 90 % CI " + f2(S.bss_ci90[0]) + " to " + f2(S.bss_ci90[1]) + ", " + S.n + " winters)" +
      (A.model_base > A.obs_base_hc_years + 0.1 ? ", and the model puts a major SSW in far more winters than happen (" + pc(A.model_base) + " of its hindcast member-winters, " + pc(A.obs_base_hc_years) + " observed)" : "") +
      '. <span id="sswBannerMore"></span>With no demonstrated skill, the observed rate of ' + pc(A.obs_base_long) + " (MERRA-2, " + A.obs_long_years + ") stays the reference for this winter.</div>";
    t += "<b>" + pc(A.forecast) + " of the 51 members</b> (" + A.forecast_k + ") have a major SSW between November and March, against the model&rsquo;s own base rate of " +
      pc(A.model_base) + " (90 % interval " + pc(A.model_base_ci90[0]) + "–" + pc(A.model_base_ci90[1]) + ") and an observed rate of " + pc(A.obs_base_long) + " (MERRA-2, " + A.obs_long_years + "). " +
      (sig ? "The departure from the model base rate is significant (binomial p " + fp(A.p_vs_model_base) + ")." : "The departure from the model base rate is not significant (binomial p " + fp(A.p_vs_model_base) + "): the members are consistent with the model&rsquo;s normal SSW rate.") + " ";
    t += S.skilful ? "In the hindcast the index has skill beyond climatology: Brier skill score " + f2(S.bss) + " (90 % CI " + f2(S.bss_ci90[0]) + " to " + f2(S.bss_ci90[1]) + ", " + S.n + " winters)."
      : "<b>In the hindcast the index " + (bssWord(S) === "worse than climatology" ? "does significantly worse than climatology" : "carries no skill beyond climatology") + "</b> (Brier skill score " + f2(S.bss) + ", 90 % CI " + f2(S.bss_ci90[0]) + " to " + f2(S.bss_ci90[1]) + "; ROC area " + f2(S.auc) + ", " + S.n + " winters, MERRA-2 as truth)." +
        (A.model_base > A.obs_base_hc_years + 0.1 ? " Part of the loss is calibration: SEAS5 puts a major SSW in " + pc(A.model_base) + " of its hindcast member-winters, against " + pc(A.obs_base_hc_years) + " of the same winters observed." : "");
    if (E.hindcast && E.observed) {
      var h = E.hindcast, o = E.observed;
      t += " ENSO context: in the hindcast, El Niño winters (DJF ONI ≥ 0.5, " + h.n_nino + " of them) average " + pc(h.rate_nino) + " of members with an SSW against " + pc(h.rate_other) + " in the rest (" + (h.p < 0.05 ? "significant, p " + h.p.toFixed(3) : "not significant, p " + h.p.toFixed(2)) + "); observed " + o.years + ": " + pc(o.rate_nino) + " against " + pc(o.rate_other) + " (" + (o.p < 0.05 ? "significant, p " + o.p.toFixed(3) : "not significant, p " + o.p.toFixed(2)) + ").";
    }
    if (J.timing && J.timing.fw_before_march) { var f = J.timing.fw_before_march; t += " Final warming before 1 March: " + pc(f.forecast) + " of members, " + pc(f.hindcast) + " in the hindcast, " + pc(f.observed) + " observed."; }
    $(ids.verdict).innerHTML = t;
    var rows = ["<tr><th>Window</th><th>This issue</th><th>Model base rate</th><th>Observed (MERRA-2)</th><th>Hindcast skill (BSS, 90 % CI)</th></tr>"];
    Object.keys(W).forEach(function (k) {
      var e = W[k], s = e.skill;
      rows.push("<tr><td>" + k + "</td><td>" + pc(e.forecast) + "</td><td>" + pc(e.model_base) + "</td><td>" + pc(e.obs_base_long) + "</td><td class=\"" + (s.skilful ? "up" : "flat") + "\">" +
        (s.skilful ? f2(s.bss) + " (" + f2(s.bss_ci90[0]) + " to " + f2(s.bss_ci90[1]) + ")" : (bssWord(s) === "worse than climatology" ? "worse: " + f2(s.bss) : "n.s.") + " (" + f2(s.bss_ci90[0]) + " to " + f2(s.bss_ci90[1]) + ")") + "</td></tr>");
    });
    if (J.monthly_u10 && J.monthly_u10.length) {
      rows.push("<tr><th>Month</th><th>u60N 10 hPa, mean (P10–P90)</th><th>Model normal (P10–P90)</th><th>Anomaly</th><th>Chance of an easterly monthly mean (normal)</th></tr>");
      J.monthly_u10.forEach(function (e) {
        rows.push("<tr><td>" + mmm(e.month) + "</td><td>" + e.mean.toFixed(1) + " (" + e.p10.toFixed(0) + "–" + e.p90.toFixed(0) + ") m/s</td><td>" + e.normal.toFixed(1) + " (" + e.normal_p10.toFixed(0) + "–" + e.normal_p90.toFixed(0) + ")</td><td>" + (e.z >= 0 ? "+" : "") + e.z.toFixed(1) + "σ</td><td>" + pc(e.p_easterly) + " (" + pc(e.p_easterly_normal) + ")</td></tr>");
      });
    }
    $(ids.table).innerHTML = rows.join("");
    var F = J.figures || {}, caps = { u10: "<b>Daily vortex wind against the model normal.</b> Thin lines: members; band: their 10–90 %; brown: ensemble mean; grey: the hindcast normal; black: GEOS FP analyses so far. Lower panel: anomaly in σ of the hindcast spread.",
      section: "<b>Where the vortex departs from normal</b>, 30–90°N at 10 hPa; inside the black contour the ensemble-mean anomaly is beyond 2 standard errors.",
      u100: "<b>100 hPa, monthly</b>: the lower-stratospheric vortex, members against the 1993–2016 hindcast normal (leads 1–6; no daily or lead-7 hindcast at this level).",
      risk: "<b>Risk, timing and reliability.</b> Bars: share of members with an SSW, against the model and observed base rates with 90 % intervals; histogram: when the first SSW falls; right: hindcast reliability against MERRA-2." };
    $(ids.figs).innerHTML = ["risk", "u10", "section", "u100"].filter(function (k) { return F[k]; }).map(function (k) {
      return '<figure><a href="assets/sst/' + F[k] + v + '" target="_blank" rel="noopener"><img src="assets/sst/' + F[k] + v + '" alt="SEAS5 polar vortex ' + k + '" loading="lazy"></a><figcaption>' + caps[k] + "</figcaption></figure>";
    }).join("");
    $(ids.updated).textContent = "built " + J.generated;
    if (document.getElementById("sswSkill")) getJSON("assets/sst/data/seas5_ssw_skill.json").then(sswSkill);
  }

  /* "Skill by ENSO and by signal strength" (scripts/sst/seas5_ssw_skill.py): six tests fixed before computing,
     Holm-corrected as one family; a result is called significant only when its corrected p is below 0.05. */
  function sswSkill(K) {
    var el = document.getElementById("sswSkill");
    if (!el || !K || !K.tests) return;
    var T = K.tests, sigAny = Object.keys(T).some(function (k) { return T[k].significant; });
    function ci(t) { var c = t.ci95 || t.ci90; return c ? (t.ci95 ? "95 % CI " : "90 % CI ") + c[0] + " to " + c[1] : "–"; }
    var h = '<h3 class="method-head">Skill by ENSO and by signal strength</h3><p class="note">' + K.n + " October starts, " + K.years +
      " (SEAS5 hindcast 1981–2016, 25 members; operational forecasts 2017–2025, members 0–24 of 51 so every winter has 25), against MERRA-2 (" +
      Math.round(100 * K.obs_rate) + " % of winters had a major SSW; SEAS5's own rate " + Math.round(100 * K.model_base) + " %). Six tests were fixed before anything was computed and are " +
      "Holm-corrected as one family. " + (sigAny ? "" : "<b>None is significant: the sample cannot show skill by signal strength or by ENSO amplitude.</b> ");
    var mde = K.mde; if (mde) h += "With " + K.n + " winters, a calibrated SSW forecast is detected with 80 % power only if its Brier skill is about " + mde.bss.toFixed(2) + " (ROC area " + mde.auc.toFixed(2) + ") or better; smaller real skill would go unseen. ";
    var sg = K.groups && K.groups["strong El Niño (ONI ≥ 1.5)"]; if (sg) h += "Strong El Niño winters number " + sg.n + " (" + sg.years.map(function (y) { return y + "/" + String((y + 1) % 100).padStart(2, "0"); }).join(", ") + "): SEAS5 put an SSW in " + Math.round(100 * sg.frac) + " % of members, " + Math.round(sg.obs_rate * sg.n) + " of the " + sg.n + " winters had one — too few winters to test.";
    var more = document.getElementById("sswBannerMore");
    if (more && !sigAny && T.T2 && Array.isArray(T.T2.value)) {
      var w1 = (K.winters || []).filter(function (w) { return w.frac >= 0.999 && !w.obs; })[0];
      more.innerHTML = "Over " + K.n + " October starts (" + K.years + "), winters in the top third of SEAS5&rsquo;s SSW share had one " + Math.round(100 * T.T2.value[0]) + " % of the time, the bottom third " + Math.round(100 * T.T2.value[1]) +
        " %, and no ENSO subset or strong-signal subset shows skill either" + (w1 ? " (" + w1.year + "/" + String((w1.year + 1) % 100).padStart(2, "0") + ": every member had an SSW; none happened)" : "") + ". ";
    }
    if (K.current) h += " This winter's " + Math.round(100 * K.current.frac) + " % (the star) is therefore the model's view, with no demonstrated skill behind it in any of the subsets tested.";
    h += "</p>";
    var rows = ["<tr><th>Pre-registered test</th><th>n</th><th>Result</th><th>Interval</th><th>p</th><th>p (Holm)</th></tr>"];
    ["T1", "T2", "T3", "T4", "T5", "T6"].forEach(function (k) {
      var t = T[k]; if (!t) return;
      var v = Array.isArray(t.value) ? "top " + Math.round(100 * t.value[0]) + " % vs bottom " + Math.round(100 * t.value[1]) + " % (n " + t.n_groups.join(", ") + ")" : (t.value == null ? "–" : t.stat + " " + t.value);
      rows.push("<tr><td>" + k + " " + t.name + "</td><td>" + t.n + "</td><td>" + v + "</td><td>" + ci(t) + "</td><td>" + t.p.toFixed(3) + '</td><td class="' + (t.significant ? "up" : "flat") + '">' + t.p_holm.toFixed(3) + (t.significant ? "" : " n.s.") + "</td></tr>");
    });
    rows.push("<tr><th>ENSO group (DJF ONI)</th><th>n</th><th>Model SSW share</th><th>Observed SSW rate</th><th>BSS (90 % CI)</th><th>ROC area (90 % CI)</th></tr>");
    Object.keys(K.groups || {}).forEach(function (g) {
      var e = K.groups[g];
      rows.push("<tr><td>" + g + "</td><td>" + e.n + "</td><td>" + (e.frac == null ? "–" : Math.round(100 * e.frac) + " %") + "</td><td>" + (e.obs_rate == null ? "–" : Math.round(100 * e.obs_rate) + " %") + "</td><td>" +
        (e.bss == null ? "n too small" : e.bss + " (" + e.bss_ci90.join(" to ") + ")") + "</td><td>" + (e.auc == null ? "–" : e.auc + " (" + e.auc_ci90.join(" to ") + ")") + "</td></tr>");
    });
    var v = "?v=" + encodeURIComponent(K.generated);
    el.innerHTML = h + '<div style="overflow-x:auto"><table class="chg">' + rows.join("") + "</table></div>" +
      '<div class="map-grid"><figure><a href="assets/sst/' + K.figure + v + '" target="_blank" rel="noopener"><img src="assets/sst/' + K.figure + v + '" alt="SEAS5 SSW fraction against observed SSWs, coloured by ENSO" loading="lazy"></a>' +
      "<figcaption><b>Forecast against outcome.</b> Each dot a winter, coloured by DJF ONI; outlined: operational years; star: this winter. Group BSS/ROC values are descriptive (not in the corrected family). " +
      "For the continuous vortex (T3, T4) the literature reports modest seasonal skill in some systems (Scaife et al. 2016, Atmos. Sci. Lett.; Portal et al. 2022, Clim. Dyn.); in SEAS5's October starts it is not significant here.</figcaption></figure></div>";
  }

  function getJSON(url) { return fetch(url + "?v=" + Date.now()).then(function (r) { return r.ok ? r.json() : null; }).catch(function () { return null; }); }

  window.SEAS5Cards = { ensure: ensure, indexCard: indexCard, teleFigs: teleFigs, stratFigs: stratFigs,
                        polarCard: polarCard, sswCard: sswCard, getJSON: getJSON, issueLabel: issueLabel };
})();
