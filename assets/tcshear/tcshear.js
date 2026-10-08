// TC shear and environment (circulation.html#tcshear). Mounted by stage.js as the dom block #tcs-view.
// Data: assets/sst/anim/tcshear_manifest.json (main, same origin) -> per-storm JSON + 330 K PV maps on the frames branch
// (sha-pinned through frames_root.js: frameFetch / frameLoad), validation in assets/sst/data/tcshear_validation.json.
// State in the hash: #tcshear/<SID>/<view: fc|val>/<hour>/<measure>.
(function () {
  "use strict";
  var root = document.getElementById("tcs-view");
  if (!root) return;
  var KT = 1 / 0.514444;
  var MAN = null, HC = null, DOC = {}, VAL = null, LIVE = null, ready = false, loading = {};
  var COL = { ctrl: "#1b1d22", aifs: "#2a6fdb", ifs: "#c2410c", gfs: "#15803d", hrrr: "#9333ea", ships: "#d97706", tdr: "#b91c1c", fan: "#7c8da6" };
  var MLAB = { "aifs-ens": "AIFS-ENS control", aifs: "AIFS single", ifs: "IFS HRES", gfs: "GFS", hrrr: "HRRR" };
  var MEAS = [
    ["deep", "200–850 hPa", "kt"], ["low", "850–500 hPa", "kt"], ["up", "500–200 hPa", "kt"], ["gs", "Generalized shear", "kt"],
    ["vi", "Ventilation index", ""], ["tilt", "Vortex tilt", "km"], ["pi", "Potential intensity", "kt"], ["rh600", "600 hPa humidity", "%"],
    ["d2pvu", "Trough distance", "km"]];
  function $(id) { return document.getElementById(id); }
  function esc(s) { return String(s == null ? "" : s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }
  function J(u) { return fetch(u + "?t=" + Math.floor(Date.now() / 300000)).then(function (r) { return r.ok ? r.json() : null; }).catch(function () { return null; }); }
  // the per-storm JSON and the maps live on the frames branch: frames_root.js walks the CDN -> raw -> jsDelivr chain
  // (circulation.html does not load it itself; its loops do, inside their iframes)
  var FR = null;
  function frameRoot() {
    if (window.frameFetch) return Promise.resolve();
    if (!FR) FR = new Promise(function (ok) { var s = document.createElement("script"); s.src = "/assets/frames_root.js"; s.onload = ok; s.onerror = ok; document.head.appendChild(s); });
    return FR;
  }
  function FJ(rel, sha) {
    return frameRoot().then(function () {
      if (!window.frameFetch) return J(rel);
      return window.frameFetch(rel, sha).then(function (r) { return r.json(); });
    }).catch(function () { return null; });
  }
  function t0(doc) { return Date.parse(doc.init.replace(/:00Z$/, ":00:00Z")); }
  function fmtT(ms, long) {
    var d = new Date(ms), D = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"], M = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    return (long ? D[d.getUTCDay()] + " " : "") + d.getUTCDate() + " " + M[d.getUTCMonth()] + " " + String(d.getUTCHours()).padStart(2, "0") + "Z";
  }
  function compass(b) { return ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"][Math.floor(((b % 360) + 371.25) / 22.5) % 16]; }
  function quadrant(rel) {   // rel: angle from downshear, counter-clockwise positive (NH: + = left of shear)
    var a = ((rel + 540) % 360) - 180;
    if (Math.abs(a) <= 22.5) return "downshear"; if (Math.abs(a) >= 157.5) return "upshear";
    if (a > 0) return a < 67.5 ? "downshear-left" : a <= 112.5 ? "left of shear" : "upshear-left";
    return a > -67.5 ? "downshear-right" : a >= -112.5 ? "right of shear" : "upshear-right";
  }
  function relOf(bearing, from) { var down = (from + 180) % 360; return (((down - bearing) + 540) % 360) - 180; }

  // ── state ─────────────────────────────────────────────────────────────────────────────────────────────────
  function state() {
    var h = decodeURIComponent(location.hash.slice(1)).split("/");
    return { prod: h[0], sid: h[1] || null, view: h[2] === "val" ? "val" : "fc", h: h[3] != null && h[3] !== "" ? +h[3] : 0, m: h[4] || "deep" };
  }
  function go(st) {
    var h = "#tcshear/" + (st.sid || "") + "/" + st.view + "/" + st.h + "/" + st.m;
    if (location.hash !== h) history.replaceState(null, "", h);
    render();
  }
  function storms() {
    var a = (MAN && MAN.storms || []).map(function (s) { return Object.assign({ live: true, sha: MAN.frames_sha }, s); });
    (HC && HC.storms || []).forEach(function (s) { a.push(Object.assign({ live: false, sha: HC.frames_sha, hcInit: HC.init }, s)); });
    return a;
  }

  // ── record helpers ────────────────────────────────────────────────────────────────────────────────────────
  function val(r, m) {
    if (!r) return null;
    switch (m) {
      case "deep": case "low": case "up": return r[m] ? r[m][0] * KT : null;
      case "gs": return r.gs != null ? r.gs * KT : null;
      case "pi": return r.pi != null ? r.pi * KT : null;
      case "tilt": return r.tilt && r.sst != null ? r.tilt[0] : null;          // over water only: an inland remnant's tilt means nothing
      default: return r[m] != null ? r[m] : null;
    }
  }
  var FANKEY = { deep: ["deep", KT], low: ["low", KT], up: ["up", KT], gs: ["gs", KT], vi: ["vi", 1], tilt: ["tilt", 1], pi: ["pi", KT], rh600: ["rh600", 1], d2pvu: ["d2pvu", 1] };
  function runs(doc) {
    var out = [], mo = doc.models || {};
    if (mo["aifs-ens"] && mo["aifs-ens"].members && mo["aifs-ens"].members["0"]) out.push(["aifs-ens", mo["aifs-ens"].members["0"], COL.ctrl]);
    ["aifs", "ifs", "gfs", "hrrr"].forEach(function (k) { if (mo[k] && mo[k].run && mo[k].run.length) out.push([k, mo[k].run, COL[k]]); });
    return out;
  }
  function at(recs, h) { for (var i = 0; i < recs.length; i++) if (recs[i].h === h) return recs[i]; return null; }
  function near(recs, h) { var b = null; recs.forEach(function (r) { if (r.h <= h && (!b || r.h > b.h)) b = r; }); return b; }
  function members(doc) { var m = (doc.models["aifs-ens"] || {}).members || {}; return Object.keys(m).filter(function (k) { return k !== "0"; }).map(function (k) { return m[k]; }); }

  // ── SVG primitives ────────────────────────────────────────────────────────────────────────────────────────
  function svgOpen(w, h, label) { return '<svg viewBox="0 0 ' + w + " " + h + '" role="img" aria-label="' + esc(label) + '" xmlns="http://www.w3.org/2000/svg">'; }
  function line(pts, col, w, extra) {
    if (pts.length < 2) return "";
    return '<polyline fill="none" stroke="' + col + '" stroke-width="' + w + '" ' + (extra || "") + ' points="' + pts.map(function (p) { return p[0].toFixed(1) + "," + p[1].toFixed(1); }).join(" ") + '"/>';
  }
  function txt(x, y, s, a, cls) { return '<text x="' + x.toFixed(1) + '" y="' + y.toFixed(1) + '"' + (a ? ' text-anchor="' + a + '"' : "") + ' class="' + (cls || "tl") + '">' + s + "</text>"; }

  // Hodograph: environmental (vortex-removed) wind at every level, kt.
  function hodograph(doc, h) {
    var W = 420, H = 420, cx = W / 2, cy = H / 2, R = 180;
    var lev = doc.levels, rs = runs(doc), ctl = at(rs.length ? rs[0][1] : [], h);
    var vmax = 10;
    function upd(r) { (r && r.env || []).forEach(function (p) { if (p) vmax = Math.max(vmax, Math.hypot(p[0], p[1]) * KT); }); }
    rs.forEach(function (x) { upd(at(x[1], h)); });
    var ringStep = vmax > 60 ? 20 : vmax > 30 ? 10 : 5, top = Math.ceil(vmax / ringStep) * ringStep, sc = R / top;
    function X(u) { return cx + u * KT * sc; } function Y(v) { return cy - v * KT * sc; }
    var s = svgOpen(W, H, "Environmental hodograph, " + doc.name + ", +" + h + " h");
    for (var k = ringStep; k <= top; k += ringStep) s += '<circle cx="' + cx + '" cy="' + cy + '" r="' + (k * sc).toFixed(1) + '" class="ring"/>' + txt(cx + 3, cy - k * sc - 3, k + " kt", null, "tm");
    s += '<line x1="' + (cx - R) + '" y1="' + cy + '" x2="' + (cx + R) + '" y2="' + cy + '" class="axis"/><line x1="' + cx + '" y1="' + (cy - R) + '" x2="' + cx + '" y2="' + (cy + R) + '" class="axis"/>';
    s += txt(cx + R - 2, cy - 5, "u east", "end", "tm") + txt(cx + 5, cy - R + 12, "v north", null, "tm");
    // members: faint
    members(doc).forEach(function (recs) { var r = at(recs, h); if (!r || !r.env) return;
      s += line(r.env.filter(Boolean).map(function (p) { return [X(p[0]), Y(p[1])]; }), "#9aa5b4", 0.7, 'opacity="0.35"'); });
    rs.slice(1).forEach(function (x) { var r = at(x[1], h); if (!r || !r.env) return;
      s += line(r.env.filter(Boolean).map(function (p) { return [X(p[0]), Y(p[1])]; }), x[2], 1.6, 'opacity="0.9"'); });
    if (ctl && ctl.env) {
      // control: segments coloured from the boundary layer (warm) to the upper troposphere (blue)
      var cols = ["#b45309", "#c2410c", "#dc2626", "#be185d", "#9d174d", "#7e22ce", "#5b21b6", "#3730a3", "#1d4ed8", "#0369a1", "#0e7490", "#155e75"];
      for (var i = 1; i < ctl.env.length; i++) { var a = ctl.env[i - 1], b = ctl.env[i]; if (!a || !b) continue;
        s += '<line x1="' + X(a[0]).toFixed(1) + '" y1="' + Y(a[1]).toFixed(1) + '" x2="' + X(b[0]).toFixed(1) + '" y2="' + Y(b[1]).toFixed(1) + '" stroke="' + cols[i] + '" stroke-width="3.2" stroke-linecap="round"/>'; }
      ctl.env.forEach(function (p, i) { if (!p) return; var L = lev[i];
        s += '<circle cx="' + X(p[0]).toFixed(1) + '" cy="' + Y(p[1]).toFixed(1) + '" r="' + ([850, 500, 200].indexOf(L) >= 0 ? 4.5 : 2.6) + '" fill="#fff" stroke="' + cols[i] + '" stroke-width="1.6"/>';
        if ([1000, 850, 700, 500, 300, 200, 100].indexOf(L) >= 0) s += txt(X(p[0]) + 6, Y(p[1]) - 5, String(L), null, "tl lev"); });
      // 200-850 shear vector (850 -> 200), the storm motion from the origin
      var i8 = lev.indexOf(850), i2 = lev.indexOf(200);
      if (ctl.env[i8] && ctl.env[i2]) s += '<line x1="' + X(ctl.env[i8][0]) + '" y1="' + Y(ctl.env[i8][1]) + '" x2="' + X(ctl.env[i2][0]) + '" y2="' + Y(ctl.env[i2][1]) + '" stroke="#d1352b" stroke-width="2" stroke-dasharray="5 3" marker-end="url(#tcsArr)"/>';
      if (ctl.mot) { var ang = ctl.mot[1] * Math.PI / 180, mv = ctl.mot[0];
        s += '<line x1="' + cx + '" y1="' + cy + '" x2="' + X(mv * Math.sin(ang)) + '" y2="' + Y(mv * Math.cos(ang)) + '" stroke="#1b1d22" stroke-width="2" marker-end="url(#tcsArrK)"/>'; }
    }
    s += '<defs><marker id="tcsArr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0L10,5L0,10z" fill="#d1352b"/></marker>' +
         '<marker id="tcsArrK" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0L10,5L0,10z" fill="#1b1d22"/></marker></defs>';
    return s + "</svg>";
  }

  // Tilt in the shear frame: downshear up, left of shear to the left (NH).
  function tiltPlot(doc, h, tm0) {
    var W = 420, H = 420, cx = W / 2, cy = H / 2 + 6, R = 170, rs = runs(doc), lim = 100;
    rs.forEach(function (x) { x[1].forEach(function (r) { if (r.tilt && r.sst != null && r.h <= 120) lim = Math.max(lim, Math.min(200, r.tilt[0])); }); });
    lim = lim > 150 ? 200 : lim > 100 ? 150 : 100;
    var sc = R / lim, sh = doc.basin === "SH" ? -1 : 1;
    function P(km, rel) { var a = rel * Math.PI / 180; km = Math.min(km, lim); return [cx - sh * km * sc * Math.sin(a), cy - km * sc * Math.cos(a)]; }
    var s = svgOpen(W, H, "Vortex tilt relative to the shear, " + doc.name);
    [25, 50, 100, 150, 200].forEach(function (k) { if (k <= lim) s += '<circle cx="' + cx + '" cy="' + cy + '" r="' + (k * sc).toFixed(1) + '" class="ring"/>' + txt(cx + 3, cy - k * sc + 12, k + " km", null, "tm"); });
    s += '<line x1="' + (cx - R) + '" y1="' + cy + '" x2="' + (cx + R) + '" y2="' + cy + '" class="axis"/><line x1="' + cx + '" y1="' + (cy - R) + '" x2="' + cx + '" y2="' + (cy + R) + '" class="axis"/>';
    s += txt(cx, cy - R - 8, "DOWNSHEAR ↑", "middle", "tq") + txt(cx, cy + R + 16, "UPSHEAR", "middle", "tq") +
         txt(cx - R + 2, cy - 6, sh > 0 ? "LEFT" : "RIGHT", null, "tq") + txt(cx + R - 2, cy - 6, sh > 0 ? "RIGHT" : "LEFT", "end", "tq");
    // members at the hour
    members(doc).forEach(function (recs) { var r = at(recs, h); if (!r || !r.tilt || r.tilt_rel == null) return; var p = P(r.tilt[0], r.tilt_rel);
      s += '<circle cx="' + p[0].toFixed(1) + '" cy="' + p[1].toFixed(1) + '" r="2.6" fill="#7c8da6" opacity="0.55"/>'; });
    rs.forEach(function (x, j) {
      var pts = x[1].filter(function (r) { return r.tilt && r.tilt_rel != null && r.sst != null && r.h <= 120; });
      s += line(pts.map(function (r) { return P(r.tilt[0], r.tilt_rel); }), x[2], j ? 1.2 : 2, 'opacity="' + (j ? 0.7 : 0.95) + '"');
      var r = at(x[1], h); if (r && r.tilt && r.tilt_rel != null) { var p = P(r.tilt[0], r.tilt_rel);
        s += '<circle cx="' + p[0].toFixed(1) + '" cy="' + p[1].toFixed(1) + '" r="' + (j ? 4.5 : 6) + '" fill="' + x[2] + '" stroke="#fff" stroke-width="1.5"><title>' + esc(MLAB[x[0]]) + ": " + r.tilt[0] + " km, " + quadrant(r.tilt_rel) + "</title></circle>"; }
    });
    // observed (TDR 2-6 km), rotated by the control's analysed shear (or the nearest forecast hour)
    var ctl = rs.length ? rs[0][1] : [];
    (doc.obs.tdr || []).forEach(function (o) {
      var hh = Math.round((o[0] * 1000 - tm0) / 3600000), r = near(ctl, Math.max(0, hh)) || ctl[0];
      if (!r || !r.deep) return;
      var rel = relOf(o[2], r.deep[1]), p = P(o[1], rel);
      s += '<path d="M' + p[0].toFixed(1) + "," + (p[1] - 6).toFixed(1) + "l6,6l-6,6l-6,-6z" + '" fill="' + COL.tdr + '" stroke="#fff" stroke-width="1.2"><title>TDR ' + fmtT(o[0] * 1000) + ": " + o[1] + " km toward " + o[2] + "° (" + quadrant(rel) + ")</title></path>";
    });
    return s + "</svg>";
  }

  // Time series of one measure along the runs, with the AIFS-ENS fan, SHIPS and the TDR.
  function series(doc, m, h) {
    var W = 860, H = 330, L = 58, Rm = 16, T = 14, B = 34, tm0 = t0(doc), rs = runs(doc), meas = MEAS.filter(function (x) { return x[0] === m; })[0];
    var fan = ((doc.models["aifs-ens"] || {}).fan || {})[FANKEY[m] ? FANKEY[m][0] : m] || {}, fk = FANKEY[m] ? FANKEY[m][1] : 1;
    var xs = [-12, 120];
    var obs = [];
    if (m === "tilt") (doc.obs.tdr || []).forEach(function (o) { obs.push([(o[0] * 1000 - tm0) / 3600000, o[1], o]); });
    var shipsPts = [];
    var sh = doc.obs.ships || {}, key = Object.keys(sh).sort().filter(function (k) { return Date.UTC(2000 + +k.slice(0, 2), +k.slice(2, 4) - 1, +k.slice(4, 6), +k.slice(6, 8)) <= tm0; }).pop();
    if (key && (m === "deep" || m === "pi")) { var S_ = sh[key], off = (S_.t0 * 1000 - tm0) / 3600000;
      (S_.hours || []).forEach(function (hh, i) { var v = m === "deep" ? S_.shear[i] : S_.pot_int[i]; if (v != null) shipsPts.push([hh + off, v]); }); }
    obs = obs.filter(function (o) { return o[0] >= -48 && o[0] <= 120; });
    obs.forEach(function (o) { xs[0] = Math.min(xs[0], Math.floor(o[0] / 6) * 6); });
    var ys = [];
    rs.forEach(function (x) { x[1].forEach(function (r) { var v = val(r, m); if (v != null) ys.push(v); }); });
    Object.keys(fan).forEach(function (k) { ys.push(fan[k][0] * fk, fan[k][4] * fk); });
    obs.forEach(function (o) { ys.push(o[1]); }); shipsPts.forEach(function (p) { ys.push(p[1]); });
    var logy = m === "vi";
    var ylo = logy ? 0.003 : 0, yhi = ys.length ? Math.max.apply(null, ys) : 1;
    if (logy) { yhi = Math.min(1, Math.max(0.3, yhi * 1.2)); ylo = Math.min(0.003, Math.max(0.0005, Math.min.apply(null, ys.filter(function (v) { return v > 0; })) || 0.003)); }
    else yhi = yhi * 1.08 || 1;
    if (m === "rh600") { ylo = 0; yhi = 100; }
    if (m === "tilt") yhi = Math.min(yhi, 160);
    function X(hh) { return L + (hh - xs[0]) / (xs[1] - xs[0]) * (W - L - Rm); }
    function Y(v) { if (logy) { v = Math.min(Math.max(v, ylo), yhi); return T + (1 - (Math.log(v) - Math.log(ylo)) / (Math.log(yhi) - Math.log(ylo))) * (H - T - B); } return T + (1 - (Math.min(v, yhi) - ylo) / (yhi - ylo)) * (H - T - B); }
    var s = svgOpen(W, H, meas[1] + " along the forecast, " + doc.name);
    // grid
    var ticks = [];
    if (logy) [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1].forEach(function (v) { if (v >= ylo && v <= yhi) ticks.push(v); });
    else { var st = niceStep((yhi - ylo) / 5); for (var v = Math.ceil(ylo / st) * st; v <= yhi; v += st) ticks.push(+v.toFixed(6)); }
    ticks.forEach(function (v) { s += '<line x1="' + L + '" x2="' + (W - Rm) + '" y1="' + Y(v).toFixed(1) + '" y2="' + Y(v).toFixed(1) + '" class="grid"/>' + txt(L - 6, Y(v) + 4, fmtNum(v), "end", "tm"); });
    for (var hh = Math.ceil(xs[0] / 24) * 24; hh <= xs[1]; hh += 24) {
      s += '<line x1="' + X(hh).toFixed(1) + '" x2="' + X(hh).toFixed(1) + '" y1="' + T + '" y2="' + (H - B) + '" class="grid"/>' + txt(X(hh), H - B + 15, fmtT(tm0 + hh * 3600000).replace(/ 00Z$/, ""), hh + 12 > xs[1] ? "end" : "middle", "tm"); }
    s += txt(L - 44, T + 4, meas[2], null, "tm");
    s += '<line x1="' + X(0).toFixed(1) + '" x2="' + X(0).toFixed(1) + '" y1="' + T + '" y2="' + (H - B) + '" class="now"/>' + txt(X(0) + 4, H - B - 6, "run start", null, "tm");
    // thresholds (literature)
    if (m === "vi") [[0.03, "0.03: TCs tend to strengthen below (TE12)"], [0.1, "0.1: ~upper bound for strengthening (TE12)"]].forEach(function (q) {
      s += '<line x1="' + L + '" x2="' + (W - Rm) + '" y1="' + Y(q[0]).toFixed(1) + '" y2="' + Y(q[0]).toFixed(1) + '" class="thr"/>' + txt(L + 6, Y(q[0]) - 5, q[1], null, "tl thrl"); });
    if (m === "tilt") [[30, "30 km: weak storms that intensified rapidly stayed below (Fischer et al. 2024)"]].forEach(function (q) {
      s += '<line x1="' + L + '" x2="' + (W - Rm) + '" y1="' + Y(q[0]).toFixed(1) + '" y2="' + Y(q[0]).toFixed(1) + '" class="thr"/>' + txt(L + 6, Y(q[0]) - 5, q[1], null, "tl thrl"); });
    // fan
    var fh = Object.keys(fan).map(Number).sort(function (a, b) { return a - b; });
    if (fh.length > 1) {
      function band(i0, i1, op) { var up = fh.map(function (k) { return [X(k), Y(fan[k][i1] * fk)]; }), dn = fh.slice().reverse().map(function (k) { return [X(k), Y(fan[k][i0] * fk)]; });
        return '<polygon fill="' + COL.fan + '" opacity="' + op + '" points="' + up.concat(dn).map(function (p) { return p[0].toFixed(1) + "," + p[1].toFixed(1); }).join(" ") + '"/>'; }
      s += band(0, 4, 0.18) + band(1, 3, 0.3);
      s += line(fh.map(function (k) { return [X(k), Y(fan[k][2] * fk)]; }), "#4b5d78", 1.4, 'stroke-dasharray="4 3"');
    }
    rs.forEach(function (x, j) {                       // a gap (no value: over land, out of the domain) breaks the line
      var seg = [];
      x[1].forEach(function (r) { var v = val(r, m); if (v == null) { s += line(seg, x[2], j ? 1.6 : 2.4); seg = []; } else seg.push([X(r.h), Y(v)]); });
      s += line(seg, x[2], j ? 1.6 : 2.4);
      if (seg.length === 1) s += '<circle cx="' + seg[0][0].toFixed(1) + '" cy="' + seg[0][1].toFixed(1) + '" r="2.5" fill="' + x[2] + '"/>'; });
    shipsPts.forEach(function (p) { s += '<rect x="' + (X(p[0]) - 3.5).toFixed(1) + '" y="' + (Y(p[1]) - 3.5).toFixed(1) + '" width="7" height="7" fill="' + COL.ships + '"><title>SHIPS ' + key + ": " + p[1] + "</title></rect>"; });
    if (shipsPts.length) s += line(shipsPts.map(function (p) { return [X(p[0]), Y(p[1])]; }), COL.ships, 1, 'stroke-dasharray="2 2"');
    obs.forEach(function (o) { s += '<path d="M' + X(o[0]).toFixed(1) + "," + (Y(o[1]) - 6).toFixed(1) + 'l6,6l-6,6l-6,-6z" fill="' + COL.tdr + '" stroke="#fff"><title>TDR ' + fmtT(o[2][0] * 1000) + ": " + o[1] + " km</title></path>"; });
    s += '<line x1="' + X(h).toFixed(1) + '" x2="' + X(h).toFixed(1) + '" y1="' + T + '" y2="' + (H - B) + '" class="sel"/>';
    return s + "</svg>";
  }
  function niceStep(x) { var p = Math.pow(10, Math.floor(Math.log10(x || 1))), f = x / p; return (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * p; }
  function fmtNum(v) { return v === 0 ? "0" : v >= 100 ? String(Math.round(v)) : v >= 10 ? v.toFixed(0) : v >= 1 ? v.toFixed(1) : v >= 0.01 ? v.toFixed(2) : v.toFixed(3); }

  function legend(doc, m) {
    var rs = runs(doc), out = [];
    out.push('<span><i style="background:' + COL.fan + ';opacity:.5"></i>AIFS-ENS 50 members: 10–90 % and 25–75 %, median dashed</span>');
    rs.forEach(function (x) { out.push('<span><i style="background:' + x[2] + '"></i>' + esc(MLAB[x[0]]) + "</span>"); });
    if (m === "deep" || m === "pi") out.push('<span><i style="background:' + COL.ships + '"></i>SHIPS (GFS), NHC real time</span>');
    if (m === "tilt") out.push('<span><i style="background:' + COL.tdr + '"></i>observed 2–6 km tilt, NOAA tail Doppler radar</span>');
    return out.join("");
  }

  function table(doc, h) {
    var rs = runs(doc), rows = "";
    function f1(x, d) { return x == null ? "—" : (+x).toFixed(d == null ? 0 : d); }
    rs.forEach(function (x) { var r = at(x[1], h); if (!r) return;
      rows += "<tr><th><i style='background:" + x[2] + "'></i>" + esc(MLAB[x[0]]) + "</th><td>" + (r.deep ? f1(r.deep[0] * KT) + " <span class=mu>from " + r.deep[1] + "°</span>" : "—") +
        "</td><td>" + (r.low ? f1(r.low[0] * KT) : "—") + "</td><td>" + (r.up ? f1(r.up[0] * KT) : "—") + "</td><td>" + (r.gs != null ? f1(r.gs * KT) : "—") +
        "</td><td>" + (r.sh_h != null ? (r.sh_h > 0 ? "upper " : "lower ") + f1(Math.abs(r.sh_h), 1) + " / " + f1(r.sh_d) + " hPa" : "—") +
        "</td><td>" + (r.tilt ? f1(r.tilt[0]) + " km <span class=mu>" + quadrant(r.tilt_rel) + "</span>" : "—") +
        "</td><td>" + (r.vi != null ? f1(r.vi, 3) + " <span class=mu>(" + f1(r.vi_env, 3) + ")</span>" : "—") + "</td><td>" + (r.pi != null ? f1(r.pi * KT) : "—") +
        "</td><td>" + f1(r.rh600) + "</td><td>" + (r.d2pvu != null ? f1(r.d2pvu) + " km " + compass(r.b2pvu) : "&gt; 1,700 km") + "</td></tr>"; });
    var mem = members(doc).map(function (recs) { return at(recs, h); }).filter(Boolean);
    if (mem.length >= 10) {
      function q(fn) { var v = mem.map(fn).filter(function (x) { return x != null; }).sort(function (a, b) { return a - b; }); if (v.length < 10) return "—";
        function p(z) { return v[Math.min(v.length - 1, Math.round(z * (v.length - 1)))]; } return [p(0.5), p(0.1), p(0.9)]; }
      function fq(a, d) { return a === "—" ? a : f1(a[0], d) + " <span class=mu>" + f1(a[1], d) + "–" + f1(a[2], d) + "</span>"; }
      rows += "<tr class=ens><th><i style='background:" + COL.fan + "'></i>AIFS-ENS members (" + mem.length + "): median, 10–90 %</th><td>" + fq(q(function (r) { return r.deep ? r.deep[0] * KT : null; })) +
        "</td><td>" + fq(q(function (r) { return r.low ? r.low[0] * KT : null; })) + "</td><td>" + fq(q(function (r) { return r.up ? r.up[0] * KT : null; })) +
        "</td><td>" + fq(q(function (r) { return r.gs != null ? r.gs * KT : null; })) + "</td><td>—</td><td>" + fq(q(function (r) { return r.tilt ? r.tilt[0] : null; })) +
        " km</td><td>" + fq(q(function (r) { return r.vi; }), 3) + "</td><td>" + fq(q(function (r) { return r.pi != null ? r.pi * KT : null; })) + "</td><td>" + fq(q(function (r) { return r.rh600; })) +
        "</td><td>" + fq(q(function (r) { return r.d2pvu; })) + "</td></tr>"; }
    return "<table class='tcs-tab'><thead><tr><th>+" + h + " h</th><th>200–850 kt</th><th>850–500</th><th>500–200</th><th>Generalized</th><th>Shear height / depth</th><th>Tilt 850→500</th><th>VI <span class=mu>(env.)</span></th><th>PI kt</th><th>RH 600 %</th><th>2-PVU air (330 K)</th></tr></thead><tbody>" + rows + "</tbody></table>";
  }

  // Plain-language summary of the analysis (descriptive only)
  function now(doc, live) {
    var rs = runs(doc), c = rs.length ? at(rs[0][1], 0) : null, tm0 = t0(doc);
    if (!c || !c.deep) return "";
    var s = "<b>" + esc(doc.name) + "</b>, AIFS-ENS run of " + fmtT(tm0, true) + ": environmental 200–850 hPa shear <b>" + Math.round(c.deep[0] * KT) + " kt from " + compass(c.deep[1]) +
      "</b> (850–500 hPa " + Math.round(c.low[0] * KT) + " kt, 500–200 hPa " + Math.round(c.up[0] * KT) + " kt; generalized " + Math.round(c.gs * KT) + " kt)";
    if (c.mot) s += ", storm moving toward " + compass(c.mot[1]) + " at " + Math.round(c.mot[0] * KT) + " kt";
    s += ". ";
    var mx = rs[0][1].reduce(function (b, r) { return r.deep && r.h <= 72 && (!b || r.deep[0] > b.deep[0]) ? r : b; }, null);
    if (mx && mx.h > 0) s += "The control's shear peaks at " + Math.round(mx.deep[0] * KT) + " kt at +" + mx.h + " h (" + fmtT(tm0 + mx.h * 3600000, true) + "). ";
    var td = (doc.obs.tdr || []).filter(function (o) { return live || o[0] * 1000 <= tm0 + 3 * 3600000; }).slice(-1)[0];
    if (td) s += "Latest radar tilt (" + fmtT(td[0] * 1000) + "): <b>" + td[1] + " km toward " + compass(td[2]) + "</b>, " + quadrant(relOf(td[2], c.deep[1])) + " of the analysed shear. ";
    if (c.vi != null) s += "Ventilation index " + c.vi.toFixed(3) + " (environmental-temperature core: " + (c.vi_env != null ? c.vi_env.toFixed(3) : "—") + "). ";
    if (c.d2pvu != null) s += "Nearest 2-PVU air on 330 K: " + c.d2pvu.toLocaleString() + " km to the " + compass(c.b2pvu) + ".";
    return s;
  }

  // ── validation view ───────────────────────────────────────────────────────────────────────────────────────
  function fmtStat(st, unit) {
    if (!st || !st.n || st.n < 5) return "<td colspan=4 class=mu>too few cases</td>";
    var sig = st.r_ci && st.r_ci[0] > 0 ? "" : " <span class=mu>(n.s.)</span>";
    return "<td>" + st.n + "</td><td>" + st.bias + " " + unit + " <span class=mu>[" + st.bias_ci[0] + ", " + st.bias_ci[1] + "]</span></td><td>" + st.rmsd + " " + unit + "</td><td>" + st.r + " <span class=mu>[" + st.r_ci[0] + ", " + st.r_ci[1] + "]</span>" + sig + "</td>";
  }
  function scatter(pairs, xl, yl, lim) {
    var W = 360, H = 340, L = 46, B = 38, T = 10, R = 10;
    function X(v) { return L + v / lim * (W - L - R); } function Y(v) { return H - B - v / lim * (H - B - T); }
    var s = svgOpen(W, H, yl + " against " + xl), st = niceStep(lim / 5);
    for (var v = 0; v <= lim + 1e-9; v += st) s += '<line x1="' + L + '" x2="' + (W - R) + '" y1="' + Y(v) + '" y2="' + Y(v) + '" class="grid"/>' + txt(L - 5, Y(v) + 4, fmtNum(v), "end", "tm") +
      '<line y1="' + T + '" y2="' + (H - B) + '" x1="' + X(v) + '" x2="' + X(v) + '" class="grid"/>' + txt(X(v), H - B + 14, fmtNum(v), "middle", "tm");
    s += '<line x1="' + X(0) + '" y1="' + Y(0) + '" x2="' + X(lim) + '" y2="' + Y(lim) + '" class="thr"/>';
    pairs.forEach(function (p) { s += '<circle cx="' + X(Math.min(lim, p[0])).toFixed(1) + '" cy="' + Y(Math.min(lim, p[1])).toFixed(1) + '" r="3" fill="' + (p[2] || "#2a6fdb") + '" opacity="0.6"/>'; });
    s += txt((L + W) / 2, H - 6, xl, "middle", "tm") + '<text transform="translate(12,' + ((T + H - B) / 2) + ') rotate(-90)" text-anchor="middle" class="tm">' + esc(yl) + "</text>";
    return s + "</svg>";
  }
  function validation() {
    if (!VAL) return "<p class=mu>The validation has not been published yet.</p>";
    var out = "", sh = VAL.ships, td = VAL.tdr;
    if (sh && sh.summary) {
      var S_ = sh.summary;
      out += "<h4>Against SHIPS: the same GFS, the same positions (" + sh.rows.length + " real-time SHIPS runs this season)</h4>" +
        "<p>For every SHIPS run NHC publishes the model's input file (ATCF <i>lsdiag</i>). At the run's start it carries the storm position and the GFS shear SHIPS uses. " +
        "The same quantities were computed here from the GFS " + (sh.gfs_run === "anl" ? "analysis of the same cycle" : "6-hour forecast of the previous cycle") +
        " (the input that reproduces SHIPS's annulus shear SHRD, which involves no vortex removal: RMS difference " + sh.gfs_pick_rmsd[sh.gfs_run] + " kt, against " +
        sh.gfs_pick_rmsd[sh.gfs_run === "anl" ? "f06" : "anl"] + " kt for the other run). Ours minus SHIPS, kt; 95 % bootstrap intervals.</p>" +
        "<table class='tcs-tab'><thead><tr><th>Quantity</th><th>n</th><th>Bias</th><th>RMS difference</th><th>Correlation</th></tr></thead><tbody>" +
        [["shrd", "SHRD 200–850 hPa, 200–800 km annulus (no vortex removal)"], ["shrs", "SHRS 850–500 hPa, annulus"], ["shrg", "SHRG generalized shear, annulus"],
         ["shdc", "SHDC 200–850 hPa, vortex removed, 0–500 km"], ["shgc", "SHGC generalized, vortex removed"], ["removal_effect", "Effect of the vortex removal (SHDC − SHRD)"],
         ["shdc_vmax_lt64", "SHDC, storms below hurricane strength"], ["shdc_vmax_ge64", "SHDC, hurricanes"]].map(function (q) {
          return "<tr><th>" + q[1] + "</th>" + fmtStat(S_[q[0]], "kt") + "</tr>"; }).join("") + "</tbody></table>";
      if (S_.sddc_diff_deg_shear_ge10kt && S_.sddc_diff_deg_shear_ge10kt.n) out += "<p>Shear direction (SHDC ≥ 10 kt, n = " + S_.sddc_diff_deg_shear_ge10kt.n + "): median absolute difference " + S_.sddc_diff_deg_shear_ge10kt.median_abs + "°, 90th percentile " + S_.sddc_diff_deg_shear_ge10kt.p90_abs + "°.</p>";
      var lim = 60, pr = sh.rows.filter(function (r) { return r.ours.shdc != null; });
      out += "<div class='tcs-grid'><figure>" + scatter(pr.map(function (r) { return [r.ships.shrd, r.ours.shrd]; }), "SHIPS SHRD (kt)", "ours, annulus (kt)", lim) + "<figcaption>Annulus shear: the data and positions agree.</figcaption></figure>" +
             "<figure>" + scatter(pr.map(function (r) { return [r.ships.shdc, r.ours.shdc, r.vmax >= 64 ? "#c2410c" : "#2a6fdb"]; }), "SHIPS SHDC (kt)", "ours, vortex removed (kt)", lim) + "<figcaption>Vortex-removed shear: the methods differ (orange = hurricanes).</figcaption></figure></div>";
    }
    if (td && td.summary) {
      var ng = ((td.summary.gfs || {}).magnitude || {}).n_groups;
      out += "<h4>Model vortex tilt against the NOAA tail Doppler radar (" + td.rows.length + " radar analyses" + (ng ? " at " + ng + " synoptic times" : "") + ")</h4>" +
        "<p>Each radar analysis with HRD's 2- and 6-km centres against the 850→500 hPa vorticity-centroid tilt of each model's analysis at the nearest synoptic time (within 3 h; a 6-hour forecast where the analysis is not published). Model minus radar, km. " +
        "Several radar analyses share one model time, so the 95 % intervals resample whole synoptic times; with this few independent cases, read the intervals, not the point values. The models resolve large tilts (30 km and more) but put a floor of roughly 10–25 km under small ones, so their small tilts are not evidence of misalignment.</p>" +
        "<table class='tcs-tab'><thead><tr><th>Model</th><th>n</th><th>Bias</th><th>RMS difference</th><th>Correlation</th><th>Direction, radar tilt ≥ 20 km</th><th>Radar ≥ 30 km / model ≥ 30 km / both</th></tr></thead><tbody>" +
        ["gfs", "ifs", "aifs-single"].map(function (k) { var q = td.summary[k] || {}, dr = q.direction_tdr_ge20km || {}, tt = q.tilted_ge30km || {};
          return "<tr><th>" + ({ gfs: "GFS", ifs: "IFS HRES", "aifs-single": "AIFS single" }[k]) + "</th>" + fmtStat(q.magnitude, "km") + "<td>" + (dr.n ? "median |error| " + dr.median_abs + "° (n = " + dr.n + ")" : "—") + "</td><td>" + (tt.n ? tt.n_tdr + " / " + tt.n_model + " / " + tt.both : "—") + "</td></tr>"; }).join("") + "</tbody></table>";
      var pts = [];
      td.rows.forEach(function (r) { ["gfs", "ifs", "aifs-single"].forEach(function (k) { if (r.models[k]) pts.push([r.tilt, r.models[k].tilt, COL[k === "aifs-single" ? "aifs" : k]]); }); });
      out += "<div class='tcs-grid'><figure>" + scatter(pts, "radar 2–6 km tilt (km)", "model 850→500 hPa tilt (km)", 120) + "<figcaption>Blue AIFS single, orange IFS HRES, green GFS.</figcaption></figure></div>";
    }
    if (LIVE && LIVE.rows && LIVE.rows.length) out += "<p class=mu>Live archive: " + LIVE.rows.length + " storm-cycle-model rows of analysis shear against SHIPS and radar tilt kept since the product started.</p>";
    return out;
  }

  // ── render ────────────────────────────────────────────────────────────────────────────────────────────────
  function render() {
    if (!ready || !root.offsetParent) return;
    var st = state(), list = storms();
    var s = list.filter(function (x) { return x.sid === st.sid; })[0] || list[0];
    $("tcs-storms").innerHTML = list.length ? list.map(function (x) {
      return '<button type="button" data-sid="' + esc(x.sid) + '" aria-pressed="' + (s && x.sid === s.sid) + '"' + (x.live ? "" : ' class="hc"') + ">" + esc(x.name) +
        (x.live ? "" : " <span>validation case, " + esc((x.hcInit || "").slice(0, 10)) + "</span>") + "</button>"; }).join("") : "";
    root.querySelectorAll("#tcs-storms button").forEach(function (b) { b.onclick = function () { st.sid = b.getAttribute("data-sid"); st.h = 0; go(st); }; });
    $("tcs-views").innerHTML = [["fc", "Forecast"], ["val", "Validation"]].map(function (v) { return '<button type="button" data-v="' + v[0] + '" aria-pressed="' + (st.view === v[0]) + '">' + v[1] + "</button>"; }).join("");
    root.querySelectorAll("#tcs-views button").forEach(function (b) { b.onclick = function () { st.view = b.getAttribute("data-v"); if (!st.sid && s) st.sid = s.sid; go(st); }; });
    $("tcs-val").hidden = st.view !== "val"; $("tcs-fc").hidden = st.view === "val";
    if (st.view === "val") { $("tcs-val").innerHTML = validation(); $("tcs-now").innerHTML = ""; return; }
    if (!s) { $("tcs-now").innerHTML = "No tropical cyclone is active" + (MAN && MAN.built ? " (checked " + esc(MAN.built.replace("T", " ")) + ")" : "") + ". The forecast panels return with the next storm; the validation stays available."; $("tcs-body").hidden = true; return; }
    $("tcs-body").hidden = false;
    var doc = DOC[s.sid];
    if (!doc) {
      $("tcs-now").innerHTML = "Loading " + esc(s.name) + "…";
      if (!loading[s.sid]) { loading[s.sid] = FJ(s.json, s.sha).then(function (d) { if (d) { DOC[s.sid] = d; render(); } else $("tcs-now").innerHTML = "Could not load the data for " + esc(s.name) + ". Reload to try again."; }); }
      return;
    }
    var tm0 = t0(doc), h = Math.max(0, Math.min(120, st.h - st.h % 6));
    $("tcs-now").innerHTML = now(doc, s.live);
    var sl = $("tcs-h"); sl.value = h;
    sl.oninput = function () { st.h = +sl.value; go(st); };
    $("tcs-hlab").textContent = "+" + h + " h · " + fmtT(tm0 + h * 3600000, true);
    $("tcs-hodo").innerHTML = hodograph(doc, h);
    $("tcs-tilt").innerHTML = tiltPlot(doc, h, tm0);
    $("tcs-meas").innerHTML = MEAS.map(function (q) { return '<button type="button" data-m="' + q[0] + '" aria-pressed="' + (st.m === q[0]) + '">' + q[1] + "</button>"; }).join("");
    root.querySelectorAll("#tcs-meas button").forEach(function (b) { b.onclick = function () { st.m = b.getAttribute("data-m"); go(st); }; });
    $("tcs-ts").innerHTML = series(doc, st.m, h);
    $("tcs-key").innerHTML = legend(doc, st.m);
    $("tcs-tab").innerHTML = table(doc, h);
    // the 330 K PV map of this hour (6-hourly control frames, frames branch)
    var fr = (doc.frames || []).filter(function (f) { return +f.slice(-8, -5) <= h; }).pop();
    var im = $("tcs-map");
    if (fr) { var rel = s.json.replace(/[^/]+$/, "") + fr; im.hidden = false; im.alt = doc.name + ": 330 K PV and shear, " + fr.slice(-8, -5) + " h";
      if (im.getAttribute("data-rel") !== rel) { im.setAttribute("data-rel", rel); if (window.frameLoad) window.frameLoad(im, rel, "", null, null, s.sha); else im.src = rel; } }
    else im.hidden = true;
  }

  Promise.all([J("assets/sst/anim/tcshear_manifest.json"), J("assets/sst/anim/tcshear_hc_manifest.json")]).then(function (r) {
    MAN = r[0]; HC = r[1]; ready = true; render();
    J("assets/sst/data/tcshear_validation.json").then(function (v) { VAL = v; if (state().view === "val") render(); });
    J("assets/sst/data/tcshear_verify.json").then(function (v) { LIVE = v; });
  });
  window.addEventListener("hashchange", render);
  window.addEventListener("popstate", render);
  var shown = false;
  if (window.MutationObserver) new MutationObserver(function () {
    var vis = !!root.offsetParent; if (vis && !shown) render(); shown = vis;
  }).observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ["hidden", "class", "style"] });
})();
