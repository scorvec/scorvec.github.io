/* c3s-vortex-card.js — "December polar vortex, C3S multi-model" on stratosphere.html (2026-10-09).
   Draws assets/sst/data/c3s_vortex.json (scripts/sst/c3s_vortex.py, built in c3s.yml each C3S issue): the multi-model
   mean's December u(60°N, 10 hPa), calibrated on the 1993–2016 hindcast against MERRA-2, as tercile probabilities and a
   forecast distribution, with the hindcast skill printed beside it (the site's rule: every claim with its test).
   Plotly (cartesian bundle) is loaded on first draw. window.C3SVortex.draw(ids) */
(function () {
  "use strict";
  var MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  var COL = { weak: "#b4541f", near: "#8a8680", strong: "#2f6690", clim: "#6f6b64", fc: "#8a4b2a", ink: "#1a1a1a", grid: "#e8e6e0" };
  function $(id) { return document.getElementById(id); }
  function pc(x) { return x == null ? "–" : Math.round(100 * x) + "%"; }
  function f1(x) { return x == null ? "–" : x.toFixed(1); }
  function f2(x) { return x == null ? "–" : x.toFixed(2); }
  function sg(x, d) { if (x == null) return "–"; var s = x.toFixed(d == null ? 1 : d); if (/^-0(\.0+)?$/.test(s)) s = s.slice(1); return (+s > 0 ? "+" : "") + s.replace("-", "−"); }
  function fp(p) { return p == null ? "–" : p < 0.001 ? "< 0.001" : p < 0.1 ? p.toFixed(3) : p.toFixed(2); }
  function ci(c, d) { return c ? "95 % CI " + c[0].toFixed(d || 2) + " to " + c[1].toFixed(d || 2) : ""; }
  function winter(y) { return y + "/" + String((y + 1) % 100).padStart(2, "0"); }
  function issueName(k) { return MON[parseInt(k.slice(5), 10) - 1] + " " + k.slice(0, 4); }

  // Student-t density and quantile (the calibrated forecast is a t distribution with n − 2 degrees of freedom)
  function lgamma(z) {
    var g = 7, c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
      12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
    if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - lgamma(1 - z);
    z -= 1; var x = c[0]; for (var i = 1; i < g + 2; i++) x += c[i] / (z + i);
    var t = z + g + 0.5; return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
  }
  function tpdf(x, df) { return Math.exp(lgamma((df + 1) / 2) - lgamma(df / 2) - 0.5 * Math.log(df * Math.PI) - (df + 1) / 2 * Math.log(1 + x * x / df)); }
  function npdf(x) { return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI); }
  function tq(q, df) { // bisection on the numerically integrated cdf (small df only matters at the 10/90 % points)
    var lo = -12, hi = 12; for (var k = 0; k < 60; k++) { var m = (lo + hi) / 2; if (tcdf(m, df) < q) lo = m; else hi = m; } return (lo + hi) / 2;
  }
  function tcdf(x, df) { var n = 400, a = -40, h = (x - a) / n, s = 0; for (var i = 0; i <= n; i++) { var w = (i === 0 || i === n) ? 1 : (i % 2 ? 4 : 2); s += w * tpdf(a + i * h, df); } return s * h / 3; }

  var plotlyQ = null;
  function plotly(cb) {
    if (window.Plotly) return cb();
    if (plotlyQ) { plotlyQ.push(cb); return; }
    plotlyQ = [cb];
    var s = document.createElement("script");
    s.src = "https://cdn.jsdelivr.net/npm/plotly.js-cartesian-dist-min@2.35.2/plotly-cartesian.min.js";
    s.onload = s.onerror = function () { var q = plotlyQ; plotlyQ = null; q.forEach(function (f) { f(); }); };
    document.head.appendChild(s);
  }
  function layout(el, o) {
    var narrow = el.offsetWidth < 620;
    return Object.assign({
      margin: { l: 56, r: 14, t: narrow ? 64 : 48, b: narrow ? 120 : 96 }, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      font: { family: '"Source Sans 3", Inter, system-ui, sans-serif', size: 13, color: COL.ink },
      legend: { orientation: "h", x: 0, y: narrow ? -0.3 : -0.24, yanchor: "top", font: { size: 12 } },
      hovermode: "closest", hoverlabel: { bgcolor: "#fff", font: { size: 12 } }
    }, o, { title: { text: "<b>" + o.title + "</b>", x: 0.01, xanchor: "left", y: 0.98, yanchor: "top", font: { size: narrow ? 13 : 15 } } });
  }

  // ── chart 1: the December distribution — climate, and the calibrated forecast when the issue is in ──────────────
  function distChart(J, id) {
    var el = $(id), T = J.truth, F = J.forecast, t1 = T.terciles[0], t2 = T.terciles[1];
    var xs = [], lo = Math.min(-5, T.mean - 3.2 * T.sd), hi = T.mean + 3.2 * T.sd;
    if (F) { lo = Math.min(lo, F.mean - 4 * F.scale); hi = Math.max(hi, F.mean + 4 * F.scale); }
    for (var x = lo; x <= hi; x += (hi - lo) / 300) xs.push(x);
    var tr = [];
    var clim = xs.map(function (v) { return npdf((v - T.mean) / T.sd) / T.sd; });
    tr.push({ x: xs, y: clim, mode: "lines", line: { color: COL.clim, width: 2, dash: "dash" }, name: "December climate 1993–2016 (MERRA-2, normal fit)", hoverinfo: "skip" });
    var ymax = Math.max.apply(null, clim);
    if (F) {
      var dens = xs.map(function (v) { return tpdf((v - F.mean) / F.scale, F.df) / F.scale; });
      ymax = Math.max(ymax, Math.max.apply(null, dens));
      [["weak", -1e9, t1], ["near", t1, t2], ["strong", t2, 1e9]].forEach(function (b) {
        var px = [], py = [];
        xs.forEach(function (v, i) { if (v >= b[1] && v <= b[2]) { px.push(v); py.push(dens[i]); } });
        if (!px.length) return;
        tr.push({ x: [px[0]].concat(px, [px[px.length - 1]]), y: [0].concat(py, [0]), mode: "lines", fill: "toself", line: { width: 0 },
                  fillcolor: b[0] === "weak" ? "rgba(180,84,31,0.30)" : b[0] === "strong" ? "rgba(47,102,144,0.30)" : "rgba(138,134,128,0.22)",
                  hoverinfo: "skip", showlegend: false });
      });
      tr.push({ x: xs, y: dens, mode: "lines", line: { color: COL.fc, width: 3.2 }, name: issueName(J.issue) + " issue, calibrated", hoverinfo: "skip" });
    }
    tr.push({ x: T.values.map(function (r) { return r.u; }), y: T.values.map(function () { return 0; }), mode: "markers", name: "observed Decembers 1993–2016",
              marker: { symbol: "line-ns-open", size: 14, color: COL.ink, line: { width: 1.6 } },
              text: T.values.map(function (r) { return "Dec " + r.year + ": " + r.u.toFixed(1) + " m/s"; }), hovertemplate: "%{text}<extra></extra>" });
    var shapes = [t1, t2].map(function (v) { return { type: "line", x0: v, x1: v, y0: 0, y1: 1, yref: "paper", line: { color: "#555", width: 1.2, dash: "dot" } }; });
    var ann = [
      { x: (lo + t1) / 2, y: 1, yref: "paper", text: "weak vortex<br>(lower third)", showarrow: false, font: { size: 12, color: COL.weak }, yanchor: "top" },
      { x: (t1 + t2) / 2, y: 1, yref: "paper", text: "near normal", showarrow: false, font: { size: 12, color: "#5f5b55" }, yanchor: "top" },
      { x: (t2 + hi) / 2, y: 1, yref: "paper", text: "strong vortex<br>(upper third)", showarrow: false, font: { size: 12, color: COL.strong }, yanchor: "top" }
    ];
    if (!F) ann.push({ x: T.mean, y: 0.45, yref: "paper", text: "forecast pending:<br>waiting for the " + issueName(J.issue) + " issue", showarrow: false, font: { size: 13, color: COL.fc }, bgcolor: "rgba(255,255,255,0.85)" });
    Plotly.react(el, tr, layout(el, { title: "December-mean wind, 60°N 10 hPa" + (F ? ": calibrated forecast" : ""),
      xaxis: { title: { text: "m/s (westerly positive)" }, range: [lo, hi], gridcolor: COL.grid, zeroline: true, zerolinecolor: "#c2185b", zerolinewidth: 1.4 },
      yaxis: { title: { text: "probability density" }, range: [-0.04 * ymax, 1.32 * ymax], gridcolor: COL.grid, zeroline: false, showticklabels: false },
      shapes: shapes, annotations: ann }), { displayModeBar: false, responsive: true });
  }

  // ── chart 2: the hindcast — what the calibration is fitted on, 24 Decembers ─────────────────────────────────────
  function hindChart(J, id) {
    var el = $(id), S = J.skill_current, R = S.regression, W = S.winters, F = J.forecast, t1 = J.truth.terciles[0], t2 = J.truth.terciles[1];
    var xmin = Math.min.apply(null, W.map(function (w) { return w.x; })), xmax = Math.max.apply(null, W.map(function (w) { return w.x; }));
    if (F) { xmin = Math.min(xmin, F.mmm_anom); xmax = Math.max(xmax, F.mmm_anom); }
    var pad = 0.12 * (xmax - xmin); xmin -= pad; xmax += pad;
    var df = R.n - 2, q90 = tq(0.9, df), xl = [], ylo = [], yhi = [], ym = [];
    for (var i = 0; i <= 60; i++) {
      var x = xmin + (xmax - xmin) * i / 60, mu = R.a + R.b * x, sc = R.s * Math.sqrt(1 + 1 / R.n + Math.pow(x - R.xbar, 2) / R.sxx);
      xl.push(x); ym.push(mu); ylo.push(mu - q90 * sc); yhi.push(mu + q90 * sc);
    }
    var tr = [
      { x: xl, y: yhi, mode: "lines", line: { width: 0 }, hoverinfo: "skip", showlegend: false },
      { x: xl, y: ylo, mode: "lines", line: { width: 0 }, fill: "tonexty", fillcolor: "rgba(138,75,42,0.13)", name: "80 % predictive range", hoverinfo: "skip" },
      { x: xl, y: ym, mode: "lines", line: { color: COL.fc, width: 2.4 }, name: "regression (the calibration)", hoverinfo: "skip" }
    ];
    var cats = [["weak", 0, COL.weak, "weak December"], ["near", 1, "#7d7972", "near-normal December"], ["strong", 2, COL.strong, "strong December"]];
    cats.forEach(function (c) {
      var w = W.filter(function (r) { return (r.obs < t1 ? 0 : r.obs > t2 ? 2 : 1) === c[1]; });
      tr.push({ x: w.map(function (r) { return r.x; }), y: w.map(function (r) { return r.obs; }), mode: "markers", name: c[3],
                marker: { size: 9, color: c[2], line: { color: "#fff", width: 1 } },
                text: w.map(function (r) { return "Dec " + r.year + "<br>multi-model anomaly " + sg(r.x, 2) + " m/s<br>observed " + r.obs.toFixed(1) + " m/s"; }),
                hovertemplate: "%{text}<extra></extra>" });
    });
    var shapes = [t1, t2].map(function (v) { return { type: "line", x0: 0, x1: 1, xref: "paper", y0: v, y1: v, line: { color: "#555", width: 1, dash: "dot" } }; });
    if (F) {
      var lo80 = F.quantiles.p10, hi80 = F.quantiles.p90;
      shapes.push({ type: "line", x0: F.mmm_anom, x1: F.mmm_anom, y0: 0, y1: 1, yref: "paper", line: { color: COL.fc, width: 2, dash: "dash" } });
      tr.push({ x: [F.mmm_anom], y: [F.mean], mode: "markers", name: issueName(J.issue) + " issue",
                marker: { symbol: "star", size: 16, color: COL.fc, line: { color: "#fff", width: 1 } },
                error_y: { type: "data", symmetric: false, array: [hi80 - F.mean], arrayminus: [F.mean - lo80], color: COL.fc, thickness: 2, width: 6 },
                hovertemplate: "this issue: multi-model anomaly " + sg(F.mmm_anom, 2) + " m/s<br>calibrated December mean " + F.mean.toFixed(1) + " m/s (80 %: " + lo80.toFixed(0) + " to " + hi80.toFixed(0) + ")<extra></extra>" });
    }
    Plotly.react(el, tr, layout(el, { title: (el.offsetWidth < 620 ? "Hindcast: model vs observed Dec" : "Hindcast 1993–2016: model anomaly vs observed December"),
      xaxis: { title: { text: "multi-model December anomaly, " + MON[J.start_month - 1] + " start, m/s" }, range: [xmin, xmax], gridcolor: COL.grid, zeroline: true, zerolinecolor: "#999" },
      yaxis: { title: { text: "observed December mean, m/s" }, gridcolor: COL.grid, zeroline: true, zerolinecolor: "#c2185b" },
      shapes: shapes }), { displayModeBar: false, responsive: true });
  }

  function verdict(J) {
    var S = J.skill_current, P = J.prereg || {}, M = (P.scores || {})["Multi-model mean"], L = S.loo, F = J.forecast, h = "";
    var start = MON[J.start_month - 1];
    if (J.state === "waiting") {
      var n = J.centres_present.length;
      h += '<div class="chipnote" style="margin:0 0 0.8rem;max-width:none"><b>Waiting for the ' + issueName(J.issue) + " C3S issue.</b> Checked " + J.generated + ": " +
        (n ? n + " of the " + J.centres_expected.length + " centres (" + J.centres_present.join(", ") + ") have it on the Copernicus data store" : "no centre has it on the Copernicus data store yet") +
        ". The centres publish around the 10th–13th of the month; the forecast appears here once at least " + J.quorum + " of them are in, and fills in as the rest arrive. The skill below is what it will be judged by.</div>";
    } else if (J.state === "no_skill") {
      h += '<div class="chipnote" style="margin:0 0 0.8rem;max-width:none"><b>No forecast this issue:</b> the centres that have published (' + J.centres_present.join(", ") +
        ") do not have significant hindcast skill as a group (r " + f2(S.r) + ", p " + fp(S.p) + "), so no calibrated probability is shown.</div>";
    } else if (F) {
      h += '<div class="vx-strip">' +
        '<div class="vx-p weak"><div class="l">Weak vortex</div><div class="v">' + pc(F.p_weak) + '</div><div class="s">below ' + J.truth.terciles[0].toFixed(0) + " m/s · climate 33%</div></div>" +
        '<div class="vx-p near"><div class="l">Near normal</div><div class="v">' + pc(F.p_normal) + '</div><div class="s">' + J.truth.terciles[0].toFixed(0) + "–" + J.truth.terciles[1].toFixed(0) + " m/s · climate 33%</div></div>" +
        '<div class="vx-p strong"><div class="l">Strong vortex</div><div class="v">' + pc(F.p_strong) + '</div><div class="s">above ' + J.truth.terciles[1].toFixed(0) + " m/s · climate 33%</div></div>" +
        '<div class="vx-p mid"><div class="l">December mean, calibrated</div><div class="v">' + F.quantiles.p50.toFixed(0) + ' m/s</div><div class="s">80 %: ' + F.quantiles.p10.toFixed(0) + " to " + F.quantiles.p90.toFixed(0) + " · normal " + J.truth.mean.toFixed(0) + "</div></div></div>";
      h += "<p class=\"idx-note\">" + issueName(J.issue) + " issue, " + (J.state === "provisional" ? "<b>provisional</b>: " + J.centres_present.length + " of " + J.centres_expected.length + " centres in (" + J.centres_present.join(", ") + "); it updates as the others publish. " : "all " + J.centres_present.length + " centres. ") +
        "Multi-model mean December anomaly " + sg(F.mmm_anom, 2) + " m/s against the models&rsquo; own normals; the hindcast regression turns that into the distribution above.</p>";
    }
    // skill, stated with its tests
    var pre = "";
    if (M && P.kind === "pre-registered") {
      pre = "<b>Hindcast skill (pre-registered test).</b> From the October start, the multi-model mean&rsquo;s December anomaly correlated with MERRA-2 at <b>r " + f2(M.r) + "</b> (" + ci(M.r_ci95) + "), Holm-corrected p " + fp(M.p_holm) +
        " in a 36-test family fixed before the data were fetched (8 centres + the mean × December, January, February, DJF); December is the only target that passed. January, February and DJF did not, nor did anything at 100 hPa. Mean-square skill " + f2(M.msss) + " (" + ci(M.msss_ci95) + "). ";
    } else if (M) {
      pre = "<b>Hindcast skill.</b> From the " + start + " start, December correlated with MERRA-2 at r " + f2(M.r) + " (" + ci(M.r_ci95) + ") in the descriptive grid of the same study (" + (M.significant ? "passes" : "does not pass") + " the false-discovery-rate control at 10 %, q " + fp(M.q_bh) + "); only the October start was in the pre-registered family. ";
    }
    pre += "With the centres used here (" + J.centres_used.join(", ") + ", current systems) r is " + f2(S.r) + " (" + ci(S.r_ci95) + ", one-sided permutation p " + fp(S.p) + "). " +
      "The raw models are far too timid: their ensemble-mean anomalies span about " + Math.round(100 * S.sd_ratio) + " % of the observed year-to-year spread, so a raw member count would understate every signal. " +
      "That is why the forecast is <b>recalibrated</b>: the observed December is regressed on the multi-model anomaly over the 24 hindcast winters (slope " + f2(S.regression.b) + ", residual " + f1(S.regression.s) + " m/s), and the probabilities come from that fit. " +
      "<b>Reliability, leave one winter out:</b> ranked probability skill " + f2(L.rpss) + " (" + ci(L.rpss_ci95) + ", permutation p " + fp(L.rpss_p) + (L.rpss_significant ? ", significant" : ", not significant") + "); the 80 % range held the observed December in " + L.cover80 + " of " + L.cover80_n +
      " winters (" + (L.cover80_p >= 0.05 ? "consistent with 80 %, binomial p " + f2(L.cover80_p) : "inconsistent with 80 %, binomial p " + fp(L.cover80_p)) + "). With 24 winters the probabilities are coarse: read a 50 % as clearly above 33 %, not as an exact number.";
    h += '<p class="idx-note vx-skill">' + pre + "</p>";
    return h;
  }

  function sysTable(J) {
    var rows = ['<tr><th>Centre (system)</th><th>Members</th><th>December anomaly vs own normal</th><th>Members 10–90 %</th><th>Tested skill, ' + MON[J.start_month - 1] + " start</th></tr>"];
    J.systems.forEach(function (c) {
      var comps = c.components, label = comps.map(function (k) { return k.label.indexOf(c.centre + " ") === 0 ? k.label.slice(c.centre.length + 1) : k.label; }).join(" + "), mem = 0, lo = null, hi = null;
      comps.forEach(function (k) { if (k.members) mem += k.members; });
      if (c.has_issue) {
        var all = []; comps.forEach(function (k) { (k.members_anom || []).forEach(function (v) { all.push(v); }); }); all.sort(function (a, b) { return a - b; });
        var q = function (p) { var i = (all.length - 1) * p, a = Math.floor(i), b = Math.ceil(i); return all[a] + (all[b] - all[a]) * (i - a); };
        lo = q(0.1); hi = q(0.9);
      }
      var t = c.tested, sk;
      if (!t) sk = '<td class="flat">no hindcast at this start</td>';
      else if (t.significant) sk = '<td class="up">r ' + f2(t.r) + (t.p_holm != null ? " (Holm p " + fp(t.p_holm) + ")" : " (q " + fp(t.q_bh) + ")") + "</td>";
      else sk = '<td class="flat">n.s.</td>';
      var note = t && t.system_changed && t.system_changed.length ? '<br><small>tested on the previous system; current ' + (c.current_hindcast ? "hindcast r " + f2(c.current_hindcast.r) + " (p " + fp(c.current_hindcast.p) + ", descriptive)" : "system not tested") + "</small>" : "";
      rows.push("<tr><td>" + c.centre + " <small>(" + label + ")</small>" + note + "</td><td>" + (c.has_issue ? mem : "–") + "</td><td>" + (c.has_issue ? sg(c.anom, 1) + " m/s" : '<span class="flat">not yet published</span>') + "</td><td>" +
        (c.has_issue ? sg(lo, 0) + " to " + sg(hi, 0) : "–") + "</td>" + sk + "</tr>");
    });
    var M = (J.prereg.scores || {})["Multi-model mean"];
    if (J.forecast) rows.push('<tr class="vx-total"><td><b>Multi-model mean</b> <small>(equal weight per centre)</small></td><td>' + J.systems.filter(function (c) { return c.has_issue; }).reduce(function (s, c) { return s + c.components.reduce(function (t, k) { return t + (k.members || 0); }, 0); }, 0) + "</td><td><b>" + sg(J.forecast.mmm_anom, 2) + " m/s</b></td><td>–</td>" +
      (M && M.significant ? '<td class="up">r ' + f2(M.r) + (M.p_holm != null ? " (Holm p " + fp(M.p_holm) + ")" : "") + "</td>" : '<td class="flat">n.s.</td>') + "</tr>");
    return rows.join("");
  }

  function relTable(J) {
    var L = J.skill_current.loo, rows = ["<tr><th>Leave-one-out forecast of</th><th>Forecast probability</th><th>Winters</th><th>Mean forecast</th><th>Observed frequency</th></tr>"];
    [["weak December", L.rel_weak], ["strong December", L.rel_strong]].forEach(function (g) {
      g[1].forEach(function (b, i) {
        rows.push("<tr><td>" + (i === 0 ? g[0] : "") + "</td><td>" + Math.round(100 * b.bin[0]) + "–" + Math.round(100 * b.bin[1]) + " %</td><td>" + b.n + "</td><td>" + (b.mean_p == null ? "–" : pc(b.mean_p)) + "</td><td>" + (b.obs_freq == null ? "–" : pc(b.obs_freq)) + "</td></tr>");
      });
    });
    return rows.join("");
  }

  function draw(ids) {
    fetch("assets/sst/data/c3s_vortex.json?v=" + Date.now()).then(function (r) { return r.ok ? r.json() : null; }).catch(function () { return null; }).then(function (J) {
      if (!J) { $(ids.verdict).innerHTML = "The December vortex outlook has not been built yet: its first build runs with the next C3S job (the 9th–16th of each month)."; return; }
      $(ids.verdict).innerHTML = verdict(J);
      $(ids.table).innerHTML = sysTable(J);
      $(ids.rel).innerHTML = relTable(J);
      $(ids.updated).textContent = "built " + J.generated + " · issue " + issueName(J.issue);
      plotly(function () {
        if (!window.Plotly) { $(ids.dist).textContent = "The charts could not be loaded just now."; return; }
        distChart(J, ids.dist); hindChart(J, ids.hind);
        if (document.fonts && document.fonts.status !== "loaded") document.fonts.ready.then(function () { distChart(J, ids.dist); hindChart(J, ids.hind); });
      });
    });
  }
  window.C3SVortex = { draw: draw };
})();
