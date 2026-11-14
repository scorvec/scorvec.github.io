// Jet cross-sections (circulation.html#pacjet/branches/xsec). Mounted by stage.js as the dom block #jxs-view: a map of
// the ensemble's 250 hPa wind to pick a transect on - a tap puts it along that longitude, a finger drags that line or,
// from anywhere else, the map, "Draw a line" takes two taps or a drag for any straight line - and under it the section
// along the transect: wind speed, potential temperature, the 2 PVU tropopause, then each jet branch's chance along it
// against the ERA5 normal. Data: assets/sst/anim/jetbr_ifs_xs_manifest.json (same origin;
// jet_branches_fc.write_sections) -> one file per lead on the frames branch, fetched through frames_root.js pinned to
// the manifest's frames_sha: the gzip of uint8 planes on a (latitude south to north, longitude) grid - per pressure
// level the wind speed, the potential temperature above the frame's theta0 of the level and the PV, then each
// branch's chance and its normal - in counts of the manifest's step, each plane stored as its differences along
// longitude, then latitude, modulo 256.
(function () {
  "use strict";
  var root = document.getElementById("jxs-view");
  if (!root) return;
  var MAN = "/assets/sst/anim/jetbr_ifs_xs_manifest.json", COAST = "/assets/models/geo_polar_coarse.json";
  var SPD0 = 20, SPD_STEP = 5;   // m/s: the first colour's floor and each colour's width, as on the jet maps
  var SPD_COL = ["#e8f1fa", "#c6dbef", "#9ecae1", "#6baed6", "#3a8fc9", "#2a9d8f", "#57b86a", "#a6d854", "#f2e34c", "#f9b233",
    "#f37b2d", "#e0442f", "#b8235a", "#7d1d6f", "#4a1060"];   // the last one: everything faster
  var NAMES = { polar: "Polar jet", stj: "Subtropical jet", sup: "Superposition" };
  var COLS = { polar: "#1f5fa8", stj: "#c0392b", sup: "#1c1c1a" };
  var MAP_P = 250;       // hPa: the level of the map's wind
  var MAP_MIN_H = 160;   // px: a narrow screen shows the longitudes that fit at this height, not the whole circle
  var OUTLINE = 0.5;     // the chance outlined on the map
  var KEEP = 6;          // decoded leads held
  var MIN_DEG = 5;       // a drawn line shorter than this is not a transect
  var GRAB = 22;         // px: a finger this near the section's line drags it; from farther off it drags the map
  var ARROW = 40;        // px: the top corners of a map of part of the circle, tapped for the next half of it
  var ISENTROPES = [270, 400, 10];   // K: first, last, spacing of the thin ones
  var DOW = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
  var MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  var $ = function (id) { return document.getElementById(id); };
  var mapCv = $("jxs-map"), keyCv = $("jxs-key"), secCv = $("jxs-sec"), slider = $("jxs-lead");
  var tokens = getComputedStyle(document.documentElement);
  var tok = function (name, or) { return tokens.getPropertyValue(name).trim() || or; };
  var INK = tok("--ink", "#1c1c1a"), MUTED = tok("--muted", "#6f6b64"), RULE = tok("--rule", "#e4e2dc"), EDGE = tok("--edge", "#c9c5bb");
  var RGB = SPD_COL.map(function (h) { return [1, 3, 5].map(function (i) { return parseInt(h.slice(i, i + 2), 16); }); });
  var man = null, coast = [], cache = {}, recent = [], lead = 0, cur = null, centre = 180, tr = null, lining = false;
  var anchor = null, drag = null, probe = null, about = "", queued = false, G = {}, L = 0, LN = [], YF = [];
  var shade = { key: "", cv: document.createElement("canvas") };
  var sect = { key: "", cv: document.createElement("canvas"), im: document.createElement("canvas"), S: null };

  // ── data ──────────────────────────────────────────────────────────────────────────────────────────────────────
  var FR = null;
  function frameRoot() {
    if (window.frameFetch) return Promise.resolve();
    if (!FR) FR = new Promise(function (ok) { var s = document.createElement("script"); s.src = "/assets/frames_root.js"; s.onload = ok; s.onerror = ok; document.head.appendChild(s); });
    return FR;
  }
  function frame(i) {   // lead i's planes, decoded -> Promise<Uint8Array>
    if (!cache[i]) {
      var path = "assets/sst/anim/" + man.dir + "/" + man.frames[i].file, g = man.grid, n = g.nlat * g.nlon;
      cache[i] = frameRoot().then(function () {
        return window.frameFetch ? window.frameFetch(path, man.frames_sha) : fetch("/" + path);
      }).then(function (r) { if (!r.ok) throw new Error(r.status); return r.arrayBuffer(); }).then(function (buf) {
        return new Response(new Blob([buf]).stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer();
      }).then(function (buf) {
        var b = new Uint8Array(buf), o, j;
        if (b.length !== (3 * L + 2 * man.branches.length) * n) throw new Error("size");
        for (o = 0; o < b.length; o += n) {   // the differences back to counts: down the latitudes, then along each
          for (j = g.nlon; j < n; j++) b[o + j] += b[o + j - g.nlon];
          for (j = 1; j < n; j++) if (j % g.nlon) b[o + j] += b[o + j - 1];
        }
        return b;
      });
      cache[i].catch(function () { delete cache[i]; });
    }
    recent = recent.filter(function (k) { return k !== i; }).concat(i);
    while (recent.length > KEEP) delete cache[recent.shift()];
    return cache[i];
  }
  function at(p, lat, lon) {   // plane p's count at a point: bilinear, the longitudes wrap
    var g = man.grid, u = Math.max(0, Math.min(g.nlat - 1, (lat - g.lat0) / g.dlat));
    var v = ((((lon - g.lon0) / g.dlon) % g.nlon) + g.nlon) % g.nlon;
    var j = Math.min(g.nlat - 2, Math.floor(u)), i = Math.floor(v), i1 = (i + 1) % g.nlon, fy = u - j, fx = v - i;
    var o = (p * g.nlat + j) * g.nlon, q = o + g.nlon;
    return (cur[o + i] * (1 - fx) + cur[o + i1] * fx) * (1 - fy) + (cur[q + i] * (1 - fx) + cur[q + i1] * fx) * fy;
  }
  function strongest() {   // the longitude of the strongest wind on the map, 20-60°N
    var g = man.grid, best = -1, lon = centre, la, lo, v;
    for (la = 20; la <= 60; la += 2 * g.dlat) for (lo = G.west; lo < G.west + G.span; lo += g.dlon) {
      v = at(L_MAP(), la, lo);
      if (v > best) { best = v; lon = lo; }
    }
    return lon;
  }
  function L_MAP() { return man.levels.indexOf(MAP_P); }
  function sample() {   // the fields along the transect: n points, each a column of the levels
    var a = tr.a, b = tr.b, g = man.grid, st = man.step, f = man.frames[lead], nb = man.branches.length;
    var n = Math.max(24, Math.ceil(Math.max(Math.abs(b[1] - a[1]) / g.dlat, Math.abs(b[0] - a[0]) / g.dlon)) + 1);
    var S = { n: n, lon: new Float32Array(n), lat: new Float32Array(n), spd: new Float32Array(n * L), th: new Float32Array(n * L),
      pv: new Float32Array(n * L), odds: [], norm: [] };
    var i, k, m, lon, lat;
    for (m = 0; m < nb; m++) { S.odds.push(new Float32Array(n)); S.norm.push(new Float32Array(n)); }
    for (i = 0; i < n; i++) {
      lon = S.lon[i] = a[0] + i / (n - 1) * (b[0] - a[0]);
      lat = S.lat[i] = a[1] + i / (n - 1) * (b[1] - a[1]);
      for (k = 0; k < L; k++) {
        S.spd[i * L + k] = at(k, lat, lon) * st.speed;
        S.th[i * L + k] = at(L + k, lat, lon) * st.theta + f.theta0[k];
        S.pv[i * L + k] = at(2 * L + k, lat, lon) * st.pv;
      }
      for (m = 0; m < nb; m++) {
        S.odds[m][i] = at(3 * L + m, lat, lon) * st.share;
        S.norm[m][i] = at(3 * L + nb + m, lat, lon) * st.share;
      }
    }
    return S;
  }
  function contour(get, nx, ny, lev, seg) {   // get(i, j) on an nx x ny lattice -> seg([i, j], [i, j]) along lev
    var i, j, a, b, c, d, A, B, C, D, p;
    for (j = 0; j < ny - 1; j++) for (i = 0; i < nx - 1; i++) {
      a = get(i, j); b = get(i + 1, j); c = get(i + 1, j + 1); d = get(i, j + 1);
      A = a >= lev; B = b >= lev; C = c >= lev; D = d >= lev;
      if (A === B && B === C && C === D) continue;
      p = [];
      if (A !== B) p.push([i + (lev - a) / (b - a), j]);
      if (B !== C) p.push([i + 1, j + (lev - b) / (c - b)]);
      if (C !== D) p.push([i + (lev - d) / (c - d), j + 1]);
      if (D !== A) p.push([i, j + (lev - a) / (d - a)]);
      seg(p[0], p[1]);
      if (p.length === 4) seg(p[2], p[3]);
    }
  }

  // ── geometry and text ─────────────────────────────────────────────────────────────────────────────────────────
  function layout() {
    var g = man.grid, W = root.clientWidth;
    G.W = W; G.fs = Math.round(0.9 * parseFloat(getComputedStyle(document.documentElement).fontSize));
    G.font = G.fs + "px " + getComputedStyle(root).fontFamily;
    G.lat0 = g.lat0; G.lat1 = g.lat0 + (g.nlat - 1) * g.dlat;
    G.span = Math.min(360, (G.lat1 - G.lat0) * W / MAP_MIN_H);
    G.west = centre - G.span / 2; G.mh = Math.round(W * (G.lat1 - G.lat0) / G.span);
    G.ml = Math.round(2.9 * G.fs); G.pw = W - G.ml - 10; G.mt = Math.round(G.fs / 2) + 3;
    G.ph = Math.round(Math.max(130, Math.min(0.42 * W, 0.36 * window.innerHeight)));
    G.st = G.mt + G.ph + G.fs + 14; G.sh = W < 500 ? 60 : 76; G.H = G.st + G.sh + G.fs + 12;   // the chances' strip: its top, height
  }
  function size(cv, w, h) {   // a canvas drawn at the device's pixel density; returns its 2-D context in CSS pixels
    var r = window.devicePixelRatio || 1;
    cv.style.width = w + "px"; cv.style.height = h + "px";
    if (cv.width !== Math.round(w * r) || cv.height !== Math.round(h * r)) { cv.width = Math.round(w * r); cv.height = Math.round(h * r); }
    var c = cv.getContext("2d"); c.setTransform(r, 0, 0, r, 0, 0); c.clearRect(0, 0, w, h);
    c.font = G.font; c.lineJoin = "round"; c.lineCap = "round"; return c;
  }
  function image(cv, w, h, fill) {   // an off-screen canvas of w x h device pixels, painted by fill(rgba bytes)
    cv.width = w; cv.height = h;
    var c = cv.getContext("2d"), img = c.createImageData(w, h);
    fill(img.data); c.putImageData(img, 0, 0);
  }
  function paint(d, q, v) {   // a wind speed's colour into rgba bytes at q; nothing under the first level
    if (v < SPD0) return;
    var c = RGB[Math.min(RGB.length - 1, Math.floor((v - SPD0) / SPD_STEP))];
    d[q] = c[0]; d[q + 1] = c[1]; d[q + 2] = c[2]; d[q + 3] = 255;
  }
  function text(c, s, x, y, col, align, base) {   // text on a white halo, kept inside the canvas
    var w = c.measureText(s).width, a = align || "left";
    x = a === "center" ? Math.max(w / 2 + 1, Math.min(G.W - w / 2 - 1, x)) : x;
    c.textAlign = a; c.textBaseline = base || "middle";
    c.lineWidth = 3; c.strokeStyle = "rgba(255,255,255,0.88)"; c.strokeText(s, x, y);
    c.fillStyle = col; c.fillText(s, x, y);
  }
  function mx(lon) { return ((((lon - G.west) % 360) + 360) % 360) / G.span * G.W; }
  function my(lat) { return (G.lat1 - lat) / (G.lat1 - G.lat0) * G.mh; }
  function trace(c, pts) {   // a line of [lon, lat] on the map, broken where it leaves one edge for the other
    var px = null;
    pts.forEach(function (p) {
      var x = mx(p[0]), y = my(p[1]);
      if (px === null || Math.abs(x - px) > 180 / G.span * G.W) c.moveTo(x, y); else c.lineTo(x, y);
      px = x;
    });
  }
  function between(a, b, n) { var out = [], i; for (i = 0; i <= n; i++) out.push([a[0] + i / n * (b[0] - a[0]), a[1] + i / n * (b[1] - a[1])]); return out; }
  function ticks(v0, v1, px) {   // round values from v0 to v1, far enough apart for their labels over px pixels
    var need = Math.abs(v1 - v0) * 4.4 * G.fs / px, s = [1, 2, 5, 10, 20, 30, 60, 90].filter(function (x) { return x >= need; })[0] || 180;
    var out = [], v;
    for (v = Math.ceil(Math.min(v0, v1) / s - 1e-9) * s; v <= Math.max(v0, v1) + 1e-9; v += s) out.push(v);
    return out;
  }
  function round(v, n) { var k = Math.pow(10, n || 0); return Math.round(v * k) / k; }
  function flon(v, n) {
    v = round((((v % 360) + 360) % 360), n);
    return v === 0 || v === 360 ? "0°" : v === 180 ? "180°" : v < 180 ? v + "°E" : round(360 - v, n) + "°W";
  }
  function flat(v, n) { return round(v, n) + "°N"; }
  function when(s) { var d = new Date(s); return DOW[d.getUTCDay()] + " " + d.getUTCDate() + " " + MON[d.getUTCMonth()] + " " + ("0" + d.getUTCHours()).slice(-2) + "Z"; }
  function level(fr) {   // a height in the section (0 at the lowest level, 1 at the highest) -> [level below, share of the way up]
    var k = 0;
    while (k < L - 2 && fr > YF[k + 1]) k++;
    return [k, Math.max(0, Math.min(1, (fr - YF[k]) / (YF[k + 1] - YF[k])))];
  }
  function sx(fi) { return G.ml + fi / (sect.S.n - 1) * G.pw; }
  function sy(fk) { var k = Math.min(L - 2, Math.floor(fk)); return G.mt + G.ph * (1 - YF[k] - (fk - k) * (YF[k + 1] - YF[k])); }

  // ── the map ───────────────────────────────────────────────────────────────────────────────────────────────────
  function mapShade(w, h) {   // the map's wind as an image of w x h device pixels
    var key = [lead, G.west, G.span, w, h].join("|");
    if (shade.key === key) return shade.cv;
    var g = man.grid, o = L_MAP() * g.nlat * g.nlon, st = man.step.speed, b = cur;
    var ci = new Int32Array(w), cj = new Int32Array(w), cf = new Float32Array(w), x, y, u, j, f, r0, r1;
    for (x = 0; x < w; x++) {
      u = ((((G.west + (x + 0.5) / w * G.span - g.lon0) / g.dlon) % g.nlon) + g.nlon) % g.nlon;
      ci[x] = Math.floor(u); cj[x] = (ci[x] + 1) % g.nlon; cf[x] = u - ci[x];
    }
    image(shade.cv, w, h, function (d) {
      for (y = 0; y < h; y++) {
        u = Math.max(0, Math.min(g.nlat - 1, (G.lat1 - (y + 0.5) / h * (G.lat1 - G.lat0) - g.lat0) / g.dlat));
        j = Math.min(g.nlat - 2, Math.floor(u)); f = u - j; r0 = o + j * g.nlon; r1 = r0 + g.nlon;
        for (x = 0; x < w; x++) {
          paint(d, (y * w + x) * 4, st * ((b[r0 + ci[x]] * (1 - cf[x]) + b[r0 + cj[x]] * cf[x]) * (1 - f) +
            (b[r1 + ci[x]] * (1 - cf[x]) + b[r1 + cj[x]] * cf[x]) * f));
        }
      }
    });
    shade.key = key;
    return shade.cv;
  }
  function outline(c, m) {   // branch m's OUTLINE chance on the map
    var g = man.grid, o = (3 * L + m) * g.nlat * g.nlon;
    c.beginPath();
    contour(function (i, j) { return cur[o + j * g.nlon + i % g.nlon]; }, g.nlon + 1, g.nlat, OUTLINE / man.step.share, function (p, q) {
      trace(c, [[g.lon0 + p[0] * g.dlon, g.lat0 + p[1] * g.dlat], [g.lon0 + q[0] * g.dlon, g.lat0 + q[1] * g.dlat]]);
    });
    c.strokeStyle = COLS[man.branches[m]]; c.lineWidth = 1.7; c.stroke();
  }
  function drawMap() {
    var c = size(mapCv, G.W, G.mh), f = man.frames[lead], east = G.west + G.span;
    c.drawImage(mapShade(mapCv.width, mapCv.height), 0, 0, G.W, G.mh);
    c.beginPath();
    ticks(G.west, east, G.W).forEach(function (v) { if (v < east) trace(c, [[v, G.lat0], [v, G.lat1]]); });
    [20, 40, 60].forEach(function (v) { c.moveTo(0, my(v)); c.lineTo(G.W, my(v)); });
    c.strokeStyle = "rgba(0,0,0,0.13)"; c.lineWidth = 1; c.stroke();
    c.beginPath(); coast.forEach(function (ln) { trace(c, ln); });
    c.strokeStyle = "#57534c"; c.lineWidth = 0.8; c.stroke();
    man.branches.forEach(function (_, m) { if (f.shown[m]) outline(c, m); });
    ticks(G.west, east, G.W).forEach(function (v) { if (mx(v) > 2 * G.fs && mx(v) < G.W - 2 * G.fs) text(c, flon(v), mx(v), G.mh - 3, MUTED, "center", "bottom"); });
    [20, 40, 60].forEach(function (v) { text(c, flat(v), 4, my(v), MUTED); });
    if (tr) {
      [["#ffffff", 5.5], [INK, 2.2]].forEach(function (s) {
        c.beginPath(); trace(c, between(tr.a, tr.b, 32)); c.strokeStyle = s[0]; c.lineWidth = s[1]; c.stroke();
      });
      if (!tr.mer) [tr.a, tr.b].forEach(dot);
    }
    if (drag && drag.to) { c.beginPath(); trace(c, [drag.at, drag.to]); c.setLineDash([5, 4]); c.strokeStyle = INK; c.lineWidth = 2; c.stroke(); c.setLineDash([]); }
    if (anchor) dot(anchor);
    if (G.span < 360) [0, 1].forEach(function (e) {   // the way west and east: an arrow in each top corner
      var s = ARROW - 10, x = e ? G.W - s - 4 : 4, m = x + s / 2, y = 4 + s / 2, d = e ? 3 : -3;
      c.fillStyle = "rgba(255,255,255,0.92)"; c.fillRect(x, 4, s, s);
      c.strokeStyle = EDGE; c.lineWidth = 1; c.strokeRect(x + 0.5, 4.5, s - 1, s - 1);
      c.beginPath(); c.moveTo(m - d, y - 6); c.lineTo(m + d, y); c.lineTo(m - d, y + 6);
      c.strokeStyle = INK; c.lineWidth = 2; c.stroke();
    });
    c.strokeStyle = EDGE; c.lineWidth = 1; c.strokeRect(0.5, 0.5, G.W - 1, G.mh - 1);
    function dot(p) { c.beginPath(); c.arc(mx(p[0]), my(p[1]), 4, 0, 2 * Math.PI); c.fillStyle = INK; c.fill(); c.strokeStyle = "#ffffff"; c.lineWidth = 1.5; c.stroke(); }
  }
  function drawKey() {   // the wind's colours, for the map and the section
    var c = size(keyCv, G.W, G.fs + 18), label = "Wind (m/s)", x0, w, i;
    c.textAlign = "left"; c.textBaseline = "middle"; c.fillStyle = INK; c.fillText(label, 0, 6);
    x0 = Math.ceil(c.measureText(label).width) + 14; w = Math.floor(Math.min(26, (G.W - x0 - 12) / SPD_COL.length));
    for (i = 0; i < SPD_COL.length; i++) {
      c.fillStyle = SPD_COL[i]; c.fillRect(x0 + i * w, 1, w, 10);
      if (i % 2 === 0) { c.fillStyle = MUTED; c.textAlign = "center"; c.textBaseline = "top"; c.fillText(String(SPD0 + i * SPD_STEP), x0 + i * w, 14); }
    }
    c.strokeStyle = EDGE; c.lineWidth = 1; c.strokeRect(x0 + 0.5, 1.5, SPD_COL.length * w - 1, 9);
  }

  // ── the section ───────────────────────────────────────────────────────────────────────────────────────────────
  function isentrope(c, S, T, col, lw) {
    c.beginPath();
    contour(function (i, k) { return S.th[i * L + k]; }, S.n, L, T, function (p, q) { c.moveTo(sx(p[0]), sy(p[1])); c.lineTo(sx(q[0]), sy(q[1])); });
    c.strokeStyle = col; c.lineWidth = lw; c.stroke();
  }
  function crossing(F, T, i) {   // the height index where column i of the field F passes T, or -1
    for (var k = 0; k < L - 1; k++) {
      var a = F[i * L + k], b = F[i * L + k + 1];
      if ((a <= T && T < b) || (b <= T && T < a)) return k + (T - a) / (b - a);
    }
    return -1;
  }
  function renderSection() {   // everything of the section but the pointer's cross, onto its off-screen canvas
    var S = sect.S, f = man.frames[lead], c = size(sect.cv, G.W, G.H), r = sect.cv.width / G.W;
    var w = Math.round(G.pw * r), h = Math.round(G.ph * r), cal = true, placed = [], x, y, i, k, T, lv, t, fx, v;
    image(sect.im, w, h, function (d) {
      for (y = 0; y < h; y++) {
        lv = level(1 - (y + 0.5) / h); k = lv[0];
        for (x = 0; x < w; x++) {
          t = (x + 0.5) / w * (S.n - 1); i = Math.min(S.n - 2, Math.floor(t)); fx = t - i;
          v = (S.spd[i * L + k] * (1 - fx) + S.spd[(i + 1) * L + k] * fx) * (1 - lv[1]) +
            (S.spd[i * L + k + 1] * (1 - fx) + S.spd[(i + 1) * L + k + 1] * fx) * lv[1];
          paint(d, (y * w + x) * 4, v);
        }
      }
    });
    c.drawImage(sect.im, G.ml, G.mt, G.pw, G.ph);
    c.save(); c.beginPath(); c.rect(G.ml, G.mt, G.pw, G.ph); c.clip();
    for (T = ISENTROPES[0]; T <= ISENTROPES[1]; T += ISENTROPES[2]) isentrope(c, S, T, "rgba(60,56,50,0.5)", 0.8);
    ["polar", "stj"].forEach(function (b) { (man.layers[b] || []).forEach(function (T) { isentrope(c, S, T, COLS[b], 1.8); }); });
    c.beginPath();
    contour(function (i, k) { return S.pv[i * L + k]; }, S.n, L, man.tropopause, function (p, q) { c.moveTo(sx(p[0]), sy(p[1])); c.lineTo(sx(q[0]), sy(q[1])); });
    c.strokeStyle = INK; c.lineWidth = 3; c.stroke();
    function clear(X, Y) { return !placed.some(function (p) { return Math.abs(p[0] - X) < 5 * G.fs && Math.abs(p[1] - Y) < G.fs; }); }
    // the layers' isentropes named: the polar jet's at the right-hand end, the subtropical jet's at the left
    [["polar", -1], ["stj", 1]].forEach(function (e) {
      (man.layers[e[0]] || []).forEach(function (T) {
        var n = Math.round(S.n / 3), j, fk, X, Y;
        for (j = 1; j < n; j++) {
          i = e[1] > 0 ? j : S.n - 1 - j; fk = crossing(S.th, T, i);
          if (fk < 0) continue;
          X = sx(i); Y = sy(fk);
          if (Y < G.mt + G.fs || Y > G.mt + G.ph - G.fs / 2 || !clear(X, Y)) continue;
          placed.push([X, Y]);
          text(c, round(T) + " K", X, Y, COLS[e[0]], e[1] > 0 ? "left" : "right");
          break;
        }
      });
    });
    // the tropopause named under its line, at the first column from the middle where the label is clear of the others
    for (k = 0; k < S.n / 4; k++) {
      i = Math.round(S.n / 2) + (k % 2 ? -1 : 1) * Math.ceil(k / 2); t = crossing(S.pv, man.tropopause, i);
      y = sy(t) + 0.9 * G.fs;
      if (t < 0 || y > G.mt + G.ph - G.fs / 2 || !clear(sx(i), y)) continue;
      text(c, man.tropopause + " PVU", sx(i), y, INK, "center");
      break;
    }
    text(c, "hPa", G.ml + 5, G.mt + 4, MUTED, "left", "top");
    c.restore();
    c.strokeStyle = EDGE; c.lineWidth = 1; c.strokeRect(G.ml + 0.5, G.mt + 0.5, G.pw - 1, G.ph - 1);
    (G.ph < 220 ? [700, 500, 300, 200, 150, 100] : [700, 500, 400, 300, 250, 200, 150, 100]).forEach(function (p) {
      y = G.mt + G.ph * (1 - (LN[0] - Math.log(p)) / (LN[0] - LN[L - 1]));
      c.beginPath(); c.moveTo(G.ml - 4, y); c.lineTo(G.ml, y); c.stroke();
      c.fillStyle = MUTED; c.textAlign = "right"; c.textBaseline = "middle"; c.fillText(String(p), G.ml - 7, y);
    });
    // the chances along the transect, under the section
    man.branches.forEach(function (_, m) { if (f.shown[m] && !f.cal[m]) cal = false; });
    c.fillStyle = INK; c.textAlign = "left"; c.textBaseline = "bottom";
    c.fillText((cal ? "Chance within " : "Members with it within ") + round(man.near_km) + " km, % (dotted: " + (G.pw < 380 ? "" : "ERA5 ") + "normal)", G.ml, G.st - 5);
    [0, 50, 100].forEach(function (p) {
      y = G.st + G.sh * (1 - p / 100);
      c.beginPath(); c.moveTo(G.ml, y); c.lineTo(G.ml + G.pw, y); c.strokeStyle = p ? RULE : EDGE; c.stroke();
      c.fillStyle = MUTED; c.textAlign = "right"; c.textBaseline = "middle"; c.fillText(String(p), G.ml - 7, y);
    });
    man.branches.forEach(function (b, m) {
      if (!f.shown[m]) return;
      [[S.norm[m], 1.5, [2, 4]], [S.odds[m], 2.2, []]].forEach(function (ln) {
        c.beginPath();
        for (i = 0; i < S.n; i++) c[i ? "lineTo" : "moveTo"](sx(i), G.st + G.sh * (1 - Math.min(1, ln[0][i])));
        c.setLineDash(ln[2]); c.strokeStyle = COLS[b]; c.lineWidth = ln[1]; c.stroke();
      });
    });
    c.setLineDash([]);
    // the transect's own axis: latitude, or longitude along a line that runs more east than north
    var a = tr.a, e = tr.b, ax = tr.lon ? 0 : 1;
    ticks(a[ax], e[ax], G.pw).forEach(function (v) {
      x = G.ml + (v - a[ax]) / (e[ax] - a[ax]) * G.pw;
      c.beginPath(); c.moveTo(x, G.st + G.sh); c.lineTo(x, G.st + G.sh + 4); c.moveTo(x, G.mt + G.ph); c.lineTo(x, G.mt + G.ph + 4);
      c.strokeStyle = EDGE; c.stroke();
      text(c, ax ? flat(v) : flon(v), x, G.st + G.sh + 7, MUTED, "center", "top");
    });
  }
  function values(x, y) {   // what the section holds under a point of its canvas, in words
    var S = sect.S, t = Math.max(0, Math.min(1, (x - G.ml) / G.pw)) * (S.n - 1), i = Math.min(S.n - 2, Math.floor(t)), fx = t - i;
    var mix = function (arr, a, b) { return arr[a] * (1 - fx) + arr[b] * fx; };
    var here = flat(mix(S.lat, i, i + 1), 1) + " " + flon(mix(S.lon, i, i + 1), 1), f = man.frames[lead], lv, k, col, out;
    if (y <= G.mt + G.ph) {
      lv = level(1 - (y - G.mt) / G.ph); k = lv[0];
      col = function (arr) { return mix(arr, i * L + k, (i + 1) * L + k) * (1 - lv[1]) + mix(arr, i * L + k + 1, (i + 1) * L + k + 1) * lv[1]; };
      return here + ", " + round(Math.exp(LN[k] + lv[1] * (LN[k + 1] - LN[k])) / 10) * 10 + " hPa: " + round(col(S.spd)) + " m/s, " +
        round(col(S.th)) + " K, " + round(col(S.pv), 1) + " PVU";
    }
    out = man.branches.map(function (b, m) {
      return f.shown[m] ? NAMES[b].toLowerCase() + " " + round(100 * mix(S.odds[m], i, i + 1)) + " % (normal " + round(100 * mix(S.norm[m], i, i + 1)) + " %)" : "";
    }).filter(Boolean);
    return here + ": " + (out.join(", ") || "no branch is drawn at this lead");
  }
  function drawSection() {
    var key = [lead, tr.a, tr.b, G.W, G.ph].join("|"), c;
    if (sect.key !== key) { sect.S = sample(); renderSection(); sect.key = key; }
    c = size(secCv, G.W, G.H);
    c.drawImage(sect.cv, 0, 0, G.W, G.H);
    if (probe) {
      c.beginPath(); c.moveTo(probe[0], G.mt); c.lineTo(probe[0], G.st + G.sh);
      if (probe[1] <= G.mt + G.ph) { c.moveTo(G.ml, probe[1]); c.lineTo(G.ml + G.pw, probe[1]); }
      c.strokeStyle = "rgba(0,0,0,0.55)"; c.lineWidth = 1; c.stroke();
    }
  }

  // ── the page's text and controls ──────────────────────────────────────────────────────────────────────────────
  function say(s) {   // a message in the run line's place
    $("jxs-status").textContent = s; $("jxs-status").hidden = !s; $("jxs-info").hidden = !!s;
  }
  function tell() {   // the line over the section: what it holds under the pointer, or else the run and the transect
    $("jxs-info").textContent = probe ? values(probe[0], probe[1]) : about;
    $("jxs-info").classList.toggle("jxs-on", !!probe);
  }
  function off(s) { cur = probe = null; about = ""; root.classList.add("off"); tell(); say(s); }
  function words() {
    var f = man.frames[lead], miss = [], h = "";
    $("jxs-when").textContent = "Day " + round(f.lead_h / 24, 1) + ", " + when(f.valid);
    about = man.model + " run of " + when(man.init).slice(4) + ", " + f.members + " members. Section along " +
      (tr.mer ? flon(tr.a[0], 1) : flat(tr.a[1], 1) + " " + flon(tr.a[0], 1) + " to " + flat(tr.b[1], 1) + " " + flon(tr.b[0], 1)) + ".";
    man.branches.forEach(function (b, m) {
      var ly = man.layers[b];
      h += '<span class="jxs-k"><i style="border-color:' + COLS[b] + '"></i>' + NAMES[b] + (ly ? " (" + round(ly[0]) + "–" + round(ly[1]) + " K)" : "") + "</span>";
      if (!f.shown[m]) miss.push(NAMES[b].toLowerCase());
    });
    h += '<span class="jxs-miss">Map outlines: a ' + round(100 * OUTLINE) + " % chance." +
      (miss.length ? " No chance drawn at this lead, where the odds did not beat the ERA5 rate on archived runs: " + miss.join(", ") + "." : "") + "</span>";
    $("jxs-leg").innerHTML = h;
    tell();
  }
  function draw() {
    if (!cur) return;
    layout(); drawMap(); drawKey(); drawSection(); words();
  }
  function meridian(lon) { tr = { a: [lon, G.lat0], b: [lon, G.lat1], mer: true, lon: false }; }
  function line(a, b) {   // any straight line, read west to east or south to north, whichever it runs more along
    var lon = Math.abs(b[0] - a[0]) > 2 * Math.abs(b[1] - a[1]), k = lon ? 0 : 1;
    tr = { a: a[k] <= b[k] ? a : b, b: a[k] <= b[k] ? b : a, mer: false, lon: lon };
  }
  function long(a, b) { return Math.max(Math.abs(b[1] - a[1]), Math.abs(b[0] - a[0]) / 2) >= MIN_DEG; }
  function show(i) {
    lead = i; slider.value = i; say("Loading…");
    $("jxs-when").textContent = "Day " + round(man.frames[i].lead_h / 24, 1) + ", " + when(man.frames[i].valid);
    frame(i).then(function (b) {
      if (lead !== i) return;
      cur = b; say(""); root.classList.remove("off");
      if (!tr) { layout(); meridian(strongest()); }
      draw();
    }, function () { if (lead === i) say("This lead could not be loaded."); });
  }
  function where(cv, e) { var b = cv.getBoundingClientRect(); return [e.clientX - b.left, e.clientY - b.top]; }
  function place(e) {   // a pointer on the map -> [longitude, latitude]
    var p = where(mapCv, e);
    return [G.west + p[0] / G.W * G.span, Math.max(G.lat0, Math.min(G.lat1, G.lat1 - p[1] / G.mh * (G.lat1 - G.lat0)))];
  }
  function tap(d) {
    var p = d.at;
    if (G.span < 360 && d.px[1] < ARROW && (d.px[0] < ARROW || d.px[0] > G.W - ARROW)) { centre += (d.px[0] < ARROW ? -1 : 1) * G.span / 2; draw(); return; }
    if (!lining) { meridian(p[0]); draw(); return; }
    if (anchor && long(anchor, p)) { line(anchor, p); anchor = null; say(""); draw(); return; }
    anchor = p; say("Tap the other end of the line."); drawMap();
  }
  function soon() {   // one redraw a frame while a finger moves
    if (queued) return;
    queued = true;
    requestAnimationFrame(function () { queued = false; draw(); });
  }
  mapCv.addEventListener("pointerdown", function (e) {   // a mouse, or Draw a line: a line dragged out; a finger: the section's line, or the map
    if (!cur) return;
    var px = where(mapCv, e), mouse = e.pointerType === "mouse";
    drag = { at: place(e), px: px, x: e.clientX, y: e.clientY, to: null, moved: false, centre: centre,
      mode: mouse || lining ? "line" : tr.mer && Math.abs(px[0] - mx(tr.a[0])) <= GRAB ? "slide" : "pan" };
    if (!mouse && probe) { probe = null; drawSection(); tell(); }
    mapCv.setPointerCapture(e.pointerId);
  });
  mapCv.addEventListener("pointermove", function (e) {
    if (!drag || (!drag.moved && Math.hypot(e.clientX - drag.x, e.clientY - drag.y) < 10)) return;
    drag.moved = true;
    if (drag.mode === "line") { drag.to = place(e); drawMap(); return; }
    if (drag.mode === "slide") meridian(G.west + Math.max(0, Math.min(1, where(mapCv, e)[0] / G.W)) * G.span);
    else centre = drag.centre - (e.clientX - drag.x) / G.W * G.span;
    soon();
  });
  mapCv.addEventListener("pointerup", function () {
    var d = drag;
    drag = null;
    if (!d) return;
    if (d.to && long(d.at, d.to)) { line(d.at, d.to); anchor = null; say(""); draw(); }
    else if (!d.moved) tap(d);
    else draw();
  });
  mapCv.addEventListener("pointercancel", function () { drag = null; if (cur) draw(); });
  mapCv.addEventListener("keydown", function (e) {   // the arrows walk the section a grid column west or east
    var d = e.key === "ArrowLeft" ? -1 : e.key === "ArrowRight" ? 1 : 0, lon;
    if (!d || !cur) return;
    e.preventDefault();
    lon = (tr.a[0] + tr.b[0]) / 2 + d * man.grid.dlon;
    if (mx(lon) > G.W) centre = lon;
    meridian(lon); draw();
  });
  function point(e) {
    if (!cur) return;
    var p = where(secCv, e);
    probe = p[0] >= G.ml && p[0] <= G.ml + G.pw && p[1] >= G.mt && p[1] <= G.st + G.sh ? p : null;
    drawSection(); tell();
  }
  secCv.addEventListener("pointerdown", point);
  secCv.addEventListener("pointermove", point);
  secCv.addEventListener("pointerleave", function (e) { if (e.pointerType === "mouse" && cur) { probe = null; drawSection(); tell(); } });
  slider.addEventListener("input", function () { show(+slider.value); });
  $("jxs-prev").onclick = function () { if (cur && lead > 0) show(lead - 1); };
  $("jxs-next").onclick = function () { if (cur && lead < man.frames.length - 1) show(lead + 1); };
  $("jxs-line").onclick = function () {
    lining = !lining; anchor = null;
    this.setAttribute("aria-pressed", String(lining)); root.classList.toggle("lining", lining);
    say(lining ? "Tap the two ends of a line on the map, or drag from one to the other." : "");
    if (cur) drawMap();
  };

  function start() {
    if (man || !root.offsetParent) return;
    man = {};
    var get = function (u, init) { return fetch(u, init).then(function (r) { if (!r.ok) throw new Error(r.status); return r.json(); }); };
    Promise.all([get(MAN, { cache: "no-cache" }), get(COAST).catch(function () { return null; })]).then(function (rs) {
      man = rs[0];
      if (!man.in_season) return off("Cross-sections run for forecasts started from November to March.");
      if (!man.frames.length) return off("This run's cross-sections are not ready yet.");
      if (typeof DecompressionStream !== "function") return off("This browser is too old to unpack the cross-sections' data.");
      L = man.levels.length; LN = man.levels.map(Math.log);
      YF = LN.map(function (v) { return (LN[0] - v) / (LN[0] - LN[L - 1]); });
      coast = (rs[1] ? rs[1].layers.coast.lines : []).map(function (ln) {
        var q = rs[1].layers.coast.q, x = 0, y = 0, out = [], i;
        for (i = 0; i < ln.length; i += 2) { x += ln[i]; y += ln[i + 1]; out.push([x / q, y / q]); }
        return out;
      }).filter(function (ln) { return ln.some(function (p) { return p[1] > man.grid.lat0; }); });
      slider.max = man.frames.length - 1;
      show(0);
    }, function () { man = null; off("The cross-sections could not be loaded."); });
  }
  window.addEventListener("resize", function () { start(); if (root.offsetParent) draw(); });
  if (document.readyState === "complete") start(); else window.addEventListener("load", start);
})();
