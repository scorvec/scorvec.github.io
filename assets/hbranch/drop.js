// Drop a particle (circulation.html#hadley/turnover/drop). Mounted by stage.js as the dom block #hbd-view: a click between
// 20°S and 20°N picks the nearest seed of hadley_branch_maps.drop(), and that seed's forward path from the chosen
// release and height is drawn on the map, with its pressure and latitude against time below; a click on a path picks
// that particle for the time-height plot (heights from the standard atmosphere: the files hold pressure). Data:
// assets/sst/anim/hbranch_drop_manifest.json (the analysed releases) and hbranch_dropfc_manifest.json (the newest HRES
// forecast, when there is one), both same origin -> int16 files on the frames branch (seed, record, [lat x 100,
// (lon - 180) x 100, p x 10]), fetched through frames_root.js pinned to each manifest's frames_sha.
(function () {
  "use strict";
  var root = document.getElementById("hbd-view");
  if (!root) return;
  var MAN = "/assets/sst/anim/hbranch_drop_manifest.json", MANFC = "/assets/sst/anim/hbranch_dropfc_manifest.json";
  var COLORS = ["#c0392b", "#1f6fb4", "#2a9d3a", "#8e44ad", "#d97706", "#0e9aa7", "#7c4a2d", "#d63384"];
  var MAX = 8, DAYS_PER_S = 5;
  var EXAMPLES = [[0, 150], [5, 195], [-5, 330]];   // first view: the warm pool, the central Pacific, the Atlantic
  var HIT_PX = 8;   // a click this close to a drawn path picks it instead of dropping a particle
  var VIRIDIS = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]];
  var man = null, rels = [], rel = 0, lev = 0, day = null, cache = {}, parts = [], used = 0, playing = null, ready = false, pick = null;
  var $ = function (id) { return document.getElementById(id); };
  var base = $("hbd-base"), map = $("hbd-map"), pch = $("hbd-p"), lch = $("hbd-lat"), slider = $("hbd-day");
  var zbox = $("hbd-zbox"), zch = $("hbd-z");

  // ── data ──────────────────────────────────────────────────────────────────────────────────────────────────────
  var FR = null;
  function frameRoot() {
    if (window.frameFetch) return Promise.resolve();
    if (!FR) FR = new Promise(function (ok) { var s = document.createElement("script"); s.src = "/assets/frames_root.js"; s.onload = ok; s.onerror = ok; document.head.appendChild(s); });
    return FR;
  }
  function file() { var r = rels[rel]; return "assets/sst/anim/" + r.dir + "/" + r.files[String(man.levels[lev])]; }
  function load(path) {
    if (!cache[path]) {
      var sha = rels[rel].sha;
      cache[path] = frameRoot().then(function () {
        return window.frameFetch ? window.frameFetch(path, sha) : fetch("/" + path);
      }).then(function (r) { if (!r.ok) throw new Error(r.status); return r.arrayBuffer(); })
        .then(function (b) { return new Int16Array(b); });
      cache[path].catch(function () { delete cache[path]; });
    }
    return cache[path];
  }
  function records() { return rels[rel].records; }
  function label(r) { return r.init ? "Forecast " + fmtDate(r.date) + " " + r.init.slice(11, 13) + "Z" : fmtDate(r.date); }
  function releases(m) {   // a manifest's releases, each carrying where its files live
    return m ? m.releases.map(function (r) { return { date: r.date, init: r.init, records: r.records, files: r.files, dir: m.dir, sha: m.frames_sha }; }) : [];
  }
  function track(arr, seed) {   // [[lat, lon, p], ...] of one seed, every step_h hours
    var R = records(), out = [];
    for (var k = 0; k < R; k++) {
      var o = (seed * R + k) * 3;
      if (arr[o] === -32768) break;
      out.push([arr[o] / 100, arr[o + 1] / 100 + 180, arr[o + 2] / 10]);
    }
    return out;
  }
  function seedAt(lat, lon) {   // the nearest seed, or -1 outside the seeded band
    var S = man.seeds, dl = S.lat[1] - S.lat[0], dn = S.lon[1] - S.lon[0];
    var i = Math.round((lat - S.lat[0]) / dl), j = Math.round((((lon - S.lon[0]) % 360) + 360) % 360 / dn) % S.lon.length;
    return i < 0 || i >= S.lat.length ? -1 : i * S.lon.length + j;
  }

  // ── geometry ──────────────────────────────────────────────────────────────────────────────────────────────────
  function xn(lon) { return ((((lon - man.basemap.centre + 180) % 360) + 360) % 360) - 180; }
  function px(lat, lon, W, H) {
    var la = man.basemap.lat;
    return [(xn(lon) + 180) / 360 * W, (la[1] - lat) / (la[1] - la[0]) * H];
  }
  function size(cv, w, h) {   // a canvas drawn at the device's pixel density; returns its 2-D context in CSS pixels
    var r = window.devicePixelRatio || 1;
    cv.style.width = w + "px"; cv.style.height = h + "px";
    if (cv.width !== Math.round(w * r) || cv.height !== Math.round(h * r)) { cv.width = Math.round(w * r); cv.height = Math.round(h * r); }
    var c = cv.getContext("2d"); c.setTransform(r, 0, 0, r, 0, 0); c.clearRect(0, 0, w, h); return c;
  }
  function height(p) {   // km in the standard atmosphere: the troposphere's lapse rate, isothermal above 226.32 hPa
    return p >= 226.32 ? 44.3308 * (1 - Math.pow(p / 1013.25, 0.190263)) : 11 + 6.34162 * Math.log(226.32 / p);
  }
  function viridis(f) {
    f = Math.max(0, Math.min(1, f)) * (VIRIDIS.length - 1);
    var i = Math.min(VIRIDIS.length - 2, Math.floor(f)), a = f - i, A = VIRIDIS[i], B = VIRIDIS[i + 1];
    return "rgb(" + [0, 1, 2].map(function (k) { return Math.round(A[k] + a * (B[k] - A[k])); }).join(",") + ")";
  }
  function fmtLat(v) { return Math.abs(v).toFixed(1) + "°" + (v >= 0 ? "N" : "S"); }
  function fmtLon(v) { v = ((v + 540) % 360) - 180; return Math.abs(v).toFixed(1) + "°" + (v >= 0 ? "E" : "W"); }
  function fmtDate(iso, addDays) {
    var d = new Date(Date.parse(iso + "T00:00:00Z") + (addDays || 0) * 864e5), M = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    return d.getUTCDate() + " " + M[d.getUTCMonth()];
  }

  // ── drawing ───────────────────────────────────────────────────────────────────────────────────────────────────
  function upto(t) { var k = Math.round((day == null ? Infinity : day) * 24 / man.step_h); return t.slice(0, Math.min(t.length, k + 1)); }
  function drawMap() {
    var W = base.clientWidth, H = base.clientHeight, c = size(map, W, H);
    parts.slice().sort(function (a, b) { return (a === pick) - (b === pick); }).forEach(function (q) {   // the pick on top
      var t = upto(q.path); if (!t.length) return;
      c.globalAlpha = pick && q !== pick ? 0.35 : 1;
      c.strokeStyle = q.color; c.lineWidth = q === pick ? 3.2 : 2; c.lineJoin = "round"; c.beginPath();
      var prev = null;
      t.forEach(function (r) {
        var p = px(r[0], r[1], W, H);
        if (prev && Math.abs(p[0] - prev[0]) < W / 2) c.lineTo(p[0], p[1]); else c.moveTo(p[0], p[1]);
        prev = p;
      });
      c.stroke();
      c.fillStyle = q.color;
      for (var k = 5 * 24 / man.step_h; k < t.length; k += 5 * 24 / man.step_h) {   // a dot at days 5, 10, 15, ...
        var d = px(t[k][0], t[k][1], W, H); c.beginPath(); c.arc(d[0], d[1], 2.6, 0, 2 * Math.PI); c.fill();
      }
      var s = px(t[0][0], t[0][1], W, H), e = px(t[t.length - 1][0], t[t.length - 1][1], W, H);
      c.lineWidth = 2; c.strokeStyle = "#fff"; c.beginPath(); c.arc(s[0], s[1], 5, 0, 2 * Math.PI); c.fill(); c.stroke();
      c.beginPath(); c.moveTo(e[0], e[1] - 6); c.lineTo(e[0] + 5.5, e[1] + 4); c.lineTo(e[0] - 5.5, e[1] + 4); c.closePath(); c.fill(); c.stroke();
    });
    c.globalAlpha = 1;
  }
  function hit(x, y) {   // the particle whose drawn path passes nearest (x, y) on the map, within HIT_PX
    var W = base.clientWidth, H = base.clientHeight, best = null, bd = HIT_PX;
    parts.forEach(function (q) {
      var t = upto(q.path), prev = null;
      t.forEach(function (r) {
        var p = px(r[0], r[1], W, H);
        if (prev && Math.abs(p[0] - prev[0]) < W / 2) {   // distance to the segment prev-p
          var dx = p[0] - prev[0], dy = p[1] - prev[1], L = dx * dx + dy * dy;
          var u = L ? Math.max(0, Math.min(1, ((x - prev[0]) * dx + (y - prev[1]) * dy) / L)) : 0;
          var d = Math.hypot(x - prev[0] - u * dx, y - prev[1] - u * dy);
          if (d < bd) { bd = d; best = q; }
        }
        prev = p;
      });
    });
    return best;
  }
  function drawHeight() {   // the picked particle's height against time, coloured by its distance from the equator
    if (!zbox) return;
    zbox.hidden = !pick || !pick.path.length;
    if (zbox.hidden) return;
    var w = Math.max(280, root.clientWidth - 8), h = 270, m = { l: 46, r: 58, t: 34, b: 46 }, c = size(zch, w, h);
    var pw = w - m.l - m.r, ph = h - m.t - m.b, D = (records() - 1) * man.step_h / 24, ZMAX = 22;
    var X = function (d) { return m.l + d / D * pw; }, Y = function (z) { return m.t + (1 - z / ZMAX) * ph; };
    var q = pick, t = upto(q.path), s = stats(t), font = getComputedStyle(root).fontFamily;
    c.font = "13px " + font; c.fillStyle = "#1b1d22"; c.textAlign = "left"; c.textBaseline = "alphabetic";
    c.fillText("Time–height of the particle from " + fmtLat(q.path[0][0]) + " " + fmtLon(q.path[0][1]) + ", released " +
      (rels[rel].init ? "the " + label(rels[rel]) : fmtDate(rels[rel].date)) + " at " + Math.round(q.path[0][2]) + " hPa", 4, 15);
    c.font = "12px " + font; c.lineWidth = 1; c.strokeStyle = "#e3e1db"; c.fillStyle = "#6f6b64";
    c.textAlign = "right"; c.textBaseline = "middle";
    for (var z = 0; z <= ZMAX; z += 4) { c.beginPath(); c.moveTo(m.l, Y(z)); c.lineTo(m.l + pw, Y(z)); c.stroke(); c.fillText(z + " km", m.l - 5, Y(z)); }
    c.textAlign = "left";
    [1000, 700, 500, 300, 200, 100, 50].forEach(function (p) { c.fillText(p + " hPa", m.l + pw + 5, Y(height(p))); });
    c.textAlign = "center"; c.textBaseline = "top";
    for (var d = 0; d <= D; d += 5) { c.beginPath(); c.moveTo(X(d), m.t); c.lineTo(X(d), m.t + ph); c.stroke(); c.fillText(d === 0 ? "day 0" : String(d), X(d), m.t + ph + 6); }
    [[s.left, "left 20°"], [s.sank, "down through 500 hPa"]].forEach(function (ev) {
      if (ev[0] == null) return;
      c.save(); c.setLineDash([4, 3]); c.strokeStyle = "#6f6b64"; c.beginPath(); c.moveTo(X(ev[0]), m.t); c.lineTo(X(ev[0]), m.t + ph); c.stroke(); c.restore();
      c.textBaseline = "bottom"; c.fillText(ev[1], X(ev[0]), m.t - 2);
    });
    c.save(); c.beginPath(); c.rect(m.l, m.t, pw, ph); c.clip(); c.lineWidth = 3.2; c.lineCap = "round";
    for (var k = 1; k < t.length; k++) {
      c.strokeStyle = viridis(Math.abs(0.5 * (t[k - 1][0] + t[k][0])) / 45);
      c.beginPath(); c.moveTo(X((k - 1) * man.step_h / 24), Y(height(t[k - 1][2]))); c.lineTo(X(k * man.step_h / 24), Y(height(t[k][2]))); c.stroke();
    }
    c.restore();
    if (day != null) { c.strokeStyle = "#1b1d22"; c.beginPath(); c.moveTo(X(day), m.t); c.lineTo(X(day), m.t + ph); c.stroke(); }
    var kx = m.l + pw - 100, ky = h - 14;   // the colour key under the axis: distance from the equator
    for (var i = 0; i < 100; i++) { c.fillStyle = viridis(i / 99); c.fillRect(kx + i, ky, 1.2, 8); }
    c.fillStyle = "#6f6b64"; c.textBaseline = "middle"; c.font = "11px " + font;
    c.textAlign = "right"; c.fillText("distance from the equator  0°", kx - 4, ky + 4); c.textAlign = "left"; c.fillText("45°", kx + 104, ky + 4);
  }
  function chart(cv, title, f, yOf, ticks, w) {   // f: the record's field; yOf: a value -> 0 (top) .. 1 (bottom)
    var h = 180, m = { l: 44, r: 8, t: 28, b: 26 }, c = size(cv, w, h), pw = w - m.l - m.r, ph = h - m.t - m.b;
    var D = (records() - 1) * man.step_h / 24, X = function (d) { return m.l + d / D * pw; };
    c.font = "12px " + getComputedStyle(root).fontFamily; c.fillStyle = "#1b1d22"; c.textBaseline = "alphabetic"; c.textAlign = "left"; c.fillText(title, 4, 13);
    c.strokeStyle = "#e3e1db"; c.fillStyle = "#6f6b64"; c.lineWidth = 1; c.textAlign = "right"; c.textBaseline = "middle";
    ticks.forEach(function (v) { var y = m.t + yOf(v) * ph; c.beginPath(); c.moveTo(m.l, y); c.lineTo(m.l + pw, y); c.stroke(); c.fillText(String(v), m.l - 5, y); });
    c.textAlign = "center"; c.textBaseline = "top";
    for (var d = 0; d <= D; d += 5) { c.beginPath(); c.moveTo(X(d), m.t); c.lineTo(X(d), m.t + ph); c.stroke(); c.fillText(d === 0 ? "day 0" : String(d), X(d), m.t + ph + 6); }
    c.save(); c.beginPath(); c.rect(m.l, m.t, pw, ph); c.clip();   // a path past the axis range runs off the plot, not over the labels
    parts.forEach(function (q) {
      var t = upto(q.path); c.strokeStyle = q.color; c.lineWidth = 2; c.beginPath();
      t.forEach(function (r, k) { var x = X(k * man.step_h / 24), y = m.t + yOf(r[f]) * ph; if (k) c.lineTo(x, y); else c.moveTo(x, y); });
      c.stroke();
    });
    c.restore();
    if (day != null) { c.strokeStyle = "#1b1d22"; c.beginPath(); c.moveTo(X(day), m.t); c.lineTo(X(day), m.t + ph); c.stroke(); }
  }
  function drawCharts() {
    var two = root.clientWidth > 760, w = Math.max(240, Math.floor((root.clientWidth - (two ? 24 : 0)) / (two ? 2 : 1)));   // the 16 px gap and some slack
    chart(pch, "Pressure (hPa)", 2, function (p) { return Math.log(p / 50) / Math.log(20); }, [50, 100, 200, 300, 500, 700, 1000], w);
    chart(lch, "Latitude (°)", 0, function (v) { return (50 - v) / 100; }, [40, 20, 0, -20, -40], w);
  }
  function stats(t) {
    var out = {}, k;
    for (k = 0; k < t.length; k++) if (Math.abs(t[k][0]) >= 20 && out.left == null) out.left = k * man.step_h / 24;
    for (k = 1; k < t.length; k++) if (t[k][2] >= 500 && t[0][2] < 500 && out.sank == null) out.sank = k * man.step_h / 24;
    out.top = Math.min.apply(null, t.map(function (r) { return r[2]; }));
    return out;
  }
  function list() {
    var host = $("hbd-list"); host.innerHTML = "";
    if (!parts.length) { host.innerHTML = '<p class="hbd-empty">Click the map between 20°S and 20°N to drop a particle; click a path for its time–height plot.</p>'; return; }
    parts.forEach(function (q, i) {
      if (!q.path.length) return;   // its file is still loading
      var t = upto(q.path), e = t[t.length - 1], s = stats(t), d = (t.length - 1) * man.step_h / 24;
      var li = document.createElement("div"); li.className = "hbd-item" + (q === pick ? " on" : "");
      li.innerHTML = '<span class="hbd-dot" style="background:' + q.color + '"></span><button type="button" class="hbd-pick" aria-pressed="' + (q === pick) + '">From ' + fmtLat(q.path[0][0]) + " " + fmtLon(q.path[0][1]) +
        " at " + Math.round(q.path[0][2]) + " hPa: day " + (Math.round(d * 4) / 4) + " at " + fmtLat(e[0]) + " " + fmtLon(e[1]) + ", " + Math.round(e[2]) + " hPa" +
        (s.left != null ? " · left the deep tropics (20°) on day " + Math.round(s.left) : " · still inside 20°") +
        (s.sank != null ? " · down through 500 hPa on day " + Math.round(s.sank) : "") + " · highest " + Math.round(s.top) + " hPa</button>";
      li.querySelector(".hbd-pick").onclick = function () { pick = pick === q ? null : q; draw(); };
      var x = document.createElement("button"); x.type = "button"; x.className = "hbd-x"; x.setAttribute("aria-label", "remove this particle"); x.textContent = "×";
      x.onclick = function () { if (pick === q) pick = null; parts.splice(i, 1); draw(); };
      li.appendChild(x); host.appendChild(li);
    });
  }
  function info() {
    var r = rels[rel], D = (r.records - 1) * man.step_h / 24, hh = r.init ? r.init.slice(11, 13) : "00";
    $("hbd-info").textContent = "Released " + fmtDate(r.date) + " " + hh + " UTC at " + man.levels[lev] + " hPa, followed " + D + " days to " +
      fmtDate(r.date, D + (+hh) / 24) + (r.init ? " on that IFS control forecast (as forecast: the model's drift is not removed)" :
      " (IFS control analyses, less their offset from ERA5)") + " · day " + (day == null ? D : Math.round(day * 4) / 4);
    slider.max = String(D); if (day == null) slider.value = String(D);
  }
  function draw() { if (!ready) return; drawMap(); drawHeight(); drawCharts(); list(); info(); }

  // ── state ─────────────────────────────────────────────────────────────────────────────────────────────────────
  function refresh() {   // the release or the height changed: every particle takes its seed's path from the new file
    var name = file();
    $("hbd-status").textContent = "Loading the paths…";
    return load(name).then(function (arr) {
      if (name !== file()) return;
      parts.forEach(function (q) { q.path = track(arr, q.seed); });
      $("hbd-status").textContent = ""; draw();
    }, function () { $("hbd-status").textContent = "The paths could not be loaded; try again shortly."; });
  }
  function add(seed) {
    if (seed < 0) return;
    if (parts.some(function (q) { return q.seed === seed; })) return;
    if (parts.length >= MAX && parts.shift() === pick) pick = null;
    pick = { seed: seed, color: COLORS[used++ % COLORS.length], path: [] };
    parts.push(pick);
    refresh();
  }
  function pills(host, items, on, pick) {
    host.innerHTML = "";
    items.forEach(function (label, i) {
      var b = document.createElement("button"); b.type = "button"; b.className = "pill" + (i === on ? " on" : ""); b.textContent = label;
      b.setAttribute("aria-pressed", i === on ? "true" : "false"); b.onclick = function () { pick(i); }; host.appendChild(b);
    });
  }
  function controls() {
    pills($("hbd-rel"), rels.map(label), rel, function (i) { rel = i; day = null; stop(); controls(); refresh(); });
    pills($("hbd-lev"), man.levels.map(function (p) { return p + " hPa"; }), lev, function (i) { lev = i; controls(); refresh(); });
  }
  function stop() { if (playing) cancelAnimationFrame(playing); playing = null; $("hbd-play").textContent = "Play"; }
  function play() {
    if (playing) { stop(); return; }
    var D = (records() - 1) * man.step_h / 24, t0 = null, d0 = day == null || day >= D ? 0 : day;
    $("hbd-play").textContent = "Pause";
    function frame(ts) {
      if (t0 == null) t0 = ts;
      day = Math.min(D, d0 + (ts - t0) / 1000 * DAYS_PER_S); slider.value = String(day); draw();
      if (day < D) playing = requestAnimationFrame(frame); else { day = null; stop(); draw(); }
    }
    playing = requestAnimationFrame(frame);
  }

  // ── wiring ────────────────────────────────────────────────────────────────────────────────────────────────────
  map.addEventListener("click", function (ev) {
    if (!man || !man.basemap) return;
    var r = map.getBoundingClientRect(), la = man.basemap.lat, q = hit(ev.clientX - r.left, ev.clientY - r.top);
    if (q) { pick = pick === q ? null : q; $("hbd-status").textContent = ""; draw(); return; }
    var lat = la[1] - (ev.clientY - r.top) / r.height * (la[1] - la[0]);
    var lon = ((ev.clientX - r.left) / r.width * 360 - 180 + man.basemap.centre + 360) % 360;
    var s = seedAt(lat, lon);
    $("hbd-status").textContent = s < 0 ? "Drop the particle between " + fmtLat(man.seeds.lat[0]) + " and " + fmtLat(man.seeds.lat[man.seeds.lat.length - 1]) + "." : "";
    add(s);
  });
  slider.addEventListener("input", function () { stop(); var D = +slider.max, v = +slider.value; day = v >= D ? null : v; draw(); });
  $("hbd-play").onclick = play;
  $("hbd-clear").onclick = function () { stop(); parts = []; pick = null; draw(); };
  base.addEventListener("load", function () { ready = true; draw(); });

  function start() {
    if (man || !root.offsetParent) return;
    man = {};
    var get = function (u) { return fetch(u, { cache: "no-cache" }).then(function (r) { if (!r.ok) throw new Error(r.status); return r.json(); }); };
    Promise.all([get(MAN), get(MANFC).catch(function () { return null; })]).then(function (ms) {
      var m = ms[0]; man = m; rels = releases(m).concat(releases(ms[1])); rel = 0; controls();
      base.src = "/" + m.basemap.src + "?v=" + m.ver;
      EXAMPLES.forEach(function (e) { var s = seedAt(e[0], e[1]); parts.push({ seed: s, color: COLORS[used++ % COLORS.length], path: [] }); });
      pick = parts[0];   // so the time-height plot shows from the start
      refresh();
    }, function () { man = null; $("hbd-status").textContent = "The particle map could not be loaded."; });
  }
  window.addEventListener("resize", function () { start(); if (root.offsetParent) draw(); });
  if (document.readyState === "complete") start(); else window.addEventListener("load", start);
})();
