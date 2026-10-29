/* The wave compass (stratosphere.html#compass), from assets/sst/data/wave_compass_<model>.json (scripts/strat/
   wave_compass.py). The 500 hPa wave-k along 55-65N: the normal standing wave, this run's anomaly and the two added, on
   a polar map the reader can turn the anomaly on; how the two add (in phase); the ERA5 100 hPa heat flux against where
   the anomaly sits; and, at the bottom, a schematic of why a leaning wave carries heat north.
   A coefficient c means a height |c| cos(k lon + arg c) along the band, lon east from 0 E: its ridge is at -arg(c)/k.
   Turning the anomaly east by r degrees multiplies every member's anomaly by exp(-i k r). The expected flux is the ERA5
   fit a + b 2 Re(N A*) + c |A|^2 averaged over the members, which is a + b 2|N||mean A| cos(offset) + c mean|A|^2. */
(function () {
  "use strict";
  var NS = "http://www.w3.org/2000/svg", D2R = Math.PI / 180;
  var FILES = { aifs: "heatflux100.json", ifs: "ifs_heatflux100.json", gefs: "gefs_heatflux100.json", geps: "geps_heatflux100.json" };
  var LABEL = { aifs: "AIFS-ENS", ifs: "IFS-ENS", gefs: "GEFS", geps: "GEPS" };
  var docs = {}, fluxes = {}, coast = null;
  var st = { model: null, k: 1, day: null, rot: 0, timer: null, lean: 40, size: 1.5 };
  var $ = function (id) { return document.getElementById(id); };
  var hour = function () { return new Date().toISOString().slice(0, 13); };
  var getJSON = function (url) { return fetch(url).then(function (r) { return r.ok ? r.json() : null; }).catch(function () { return null; }); };

  // complex numbers as [re, im]
  var cx = {
    add: function (a, b) { return [a[0] + b[0], a[1] + b[1]]; },
    sub: function (a, b) { return [a[0] - b[0], a[1] - b[1]]; },
    turn: function (a, t) { var c = Math.cos(t), s = Math.sin(t); return [a[0] * c - a[1] * s, a[0] * s + a[1] * c]; },
    abs: function (a) { return Math.hypot(a[0], a[1]); },
    arg: function (a) { return Math.atan2(a[1], a[0]); }
  };
  var wrap = function (d, p) { p = p || 360; return ((d % p) + p + p / 2) % p - p / 2; };   // into (-p/2, p/2]
  var ridge = function (c, k) { return wrap(-cx.arg(c) / D2R / k, 360 / k); };              // deg E of the first ridge
  var fmt = function (v, n) { return v.toFixed(n === undefined ? 0 : n); };
  var lonText = function (x) { x = wrap(x); return Math.abs(x) < 0.5 ? "0°" : Math.abs(Math.abs(x) - 180) < 0.5 ? "180°" : fmt(Math.abs(x)) + "°" + (x > 0 ? "E" : "W"); };
  function el(tag, attrs, parent, text) {
    var e = document.createElementNS(NS, tag);
    for (var a in attrs) e.setAttribute(a, attrs[a]);
    if (text !== undefined) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }

  // ── the run's state on the chosen day ────────────────────────────────────────────────────────────────────────────
  function state(doc, k, day, rot) {
    var w = doc.waves[String(k)], N = w.normal[day], t = -k * rot * D2R, mean = [0, 0], a2 = 0, n = w.members[day].length;
    var members = w.members[day].map(function (m) {
      var A = cx.turn(cx.sub(m, N), t); mean = cx.add(mean, A); a2 += A[0] * A[0] + A[1] * A[1]; return cx.add(N, A);
    });
    mean = [mean[0] / n, mean[1] / n]; a2 /= n;
    var off = Math.abs(wrap((cx.arg(mean) - cx.arg(N)) / D2R)), e = w.era5, f = e.fit;
    var expect = function (o) { return Math.max(0, f.a + f.b * 2 * cx.abs(N) * cx.abs(mean) * Math.cos(o * D2R) + f.c * a2); };
    var p = w.p_anomaly[day];
    return { w: w, N: N, A: mean, T: cx.add(N, mean), members: members, off: off, a2: a2, era5: e, expect: expect,
             agree: p !== null && p < 0.05, linked: e.linked, flux: e.linked ? expect(off) : null };
  }

  // ── the polar map ────────────────────────────────────────────────────────────────────────────────────────────────
  var MAP = 440, C = MAP / 2, RIM = 186, EDGE = 20;
  var rad = function (lat) { return (90 - lat) / (90 - EDGE) * RIM; };
  var xy = function (lon, r) { return [C + r * Math.sin(lon * D2R), C + r * Math.cos(lon * D2R)]; };   // 0 E at the bottom, 90 E right
  var SCALE = { 1: 0.22, 2: 0.26 };   // px per metre of height: a ridge pulls the 60N contour this far poleward
  function coastPath() {
    if (!coast) return "";
    var L = coast.layers.coast, q = L.q, d = [];
    L.lines.forEach(function (ln) {
      var x = ln[0], y = ln[1], pen = false;
      for (var i = 0; i < ln.length / 2; i++) {
        if (i) { x += ln[2 * i]; y += ln[2 * i + 1]; }
        var lat = y / q;
        if (lat < EDGE) { pen = false; continue; }
        var p = xy(x / q, rad(lat)); d.push((pen ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1)); pen = true;
      }
    });
    return d.join("");
  }
  function contour(c, k, s) {
    var d = [];
    for (var lon = 0; lon <= 360; lon += 3) {
      var h = cx.abs(c) * Math.cos(k * lon * D2R + cx.arg(c)), p = xy(lon, rad(60) - s * h);
      d.push((lon ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1));
    }
    return d.join("") + "Z";
  }
  function drawMap(S, k) {
    var svg = $("wcMap"); svg.innerHTML = "";
    var s = SCALE[k], amax = 200;
    el("circle", { cx: C, cy: C, r: RIM, class: "wc-disc" }, svg);
    // the anomaly as shading on the 50-70N annulus: warm where it raises the heights, cool where it lowers them
    for (var lon = 0; lon < 360; lon += 5) {
      var h = cx.abs(S.A) * Math.cos(k * (lon + 2.5) * D2R + cx.arg(S.A)), o = Math.min(1, Math.abs(h) / amax);
      var a = xy(lon, rad(70)), b = xy(lon + 5, rad(70)), c = xy(lon + 5, rad(50)), d = xy(lon, rad(50));
      el("path", { d: "M" + a + "A" + rad(70) + " " + rad(70) + " 0 0 0 " + b + "L" + c + "A" + rad(50) + " " + rad(50) + " 0 0 1 " + d + "Z",
                   class: h > 0 ? "wc-warm" : "wc-cool", "fill-opacity": (S.agree ? 0.75 : 0.3) * o }, svg);
    }
    el("path", { d: coastPath(), class: "wc-coast" }, svg);
    [30, 60].forEach(function (la) { el("circle", { cx: C, cy: C, r: rad(la), class: "wc-grat" }, svg); });
    for (var m = 0; m < 360; m += 90) { var e = xy(m, RIM); el("line", { x1: C, y1: C, x2: e[0], y2: e[1], class: "wc-grat" }, svg); }
    [[0, "0°"], [90, "90°E"], [180, "180°"], [-90, "90°W"]].forEach(function (t) {
      var p = xy(t[0], RIM + 13);
      el("text", { x: p[0], y: p[1] + 4, class: "wc-tick", "text-anchor": "middle" }, svg, t[1]);
    });
    // every member's ridge as a tick on the rim, then the normal and this run's 60N contours with their ridge marks
    S.members.forEach(function (M) {
      for (var j = 0; j < k; j++) {
        var lon = ridge(M, k) + j * 360 / k, a = xy(lon, RIM - 9), b = xy(lon, RIM);
        el("line", { x1: a[0], y1: a[1], x2: b[0], y2: b[1], class: "wc-member" }, svg);
      }
    });
    el("path", { d: contour(S.N, k, s), class: "wc-normal" }, svg);
    el("path", { d: contour(S.T, k, s), class: "wc-total" }, svg);
    [[S.N, "wc-mark-normal"], [S.T, "wc-mark-total"]].forEach(function (x) {
      for (var j = 0; j < k; j++) {
        var lon = ridge(x[0], k) + j * 360 / k, p = xy(lon, rad(60) - s * cx.abs(x[0]));
        el("circle", { cx: p[0], cy: p[1], r: 5.5, class: x[1] }, svg);
      }
    });
    el("circle", { cx: C, cy: C, r: RIM, class: "wc-rim" }, svg);
  }
  function dragMap(svg) {
    var last = null;
    var ang = function (ev) {
      var r = svg.getBoundingClientRect(), x = (ev.clientX - r.left) / r.width * MAP - C, y = (ev.clientY - r.top) / r.height * MAP - C;
      return Math.atan2(x, y) / D2R;   // longitude under the pointer (0 E at the bottom)
    };
    svg.addEventListener("pointerdown", function (ev) { last = ang(ev); svg.setPointerCapture(ev.pointerId); svg.classList.add("wc-dragging"); });
    svg.addEventListener("pointermove", function (ev) {
      if (last === null) return;
      var a = ang(ev); st.rot = wrap(st.rot + wrap(a - last)); last = a; redraw();
    });
    var end = function () { last = null; svg.classList.remove("wc-dragging"); };
    svg.addEventListener("pointerup", end); svg.addEventListener("pointercancel", end);
    svg.addEventListener("keydown", function (ev) {
      var step = ev.key === "ArrowLeft" ? -5 : ev.key === "ArrowRight" ? 5 : 0;
      if (step) { ev.preventDefault(); st.rot = wrap(st.rot + step); redraw(); }
    });
  }

  // ── how the two add, in phase (the normal pointing right) ────────────────────────────────────────────────────────
  function drawPhasor(S) {
    var svg = $("wcPhasor"); svg.innerHTML = "";
    // fitted to the box (at most 0.84 px a metre, 250 m across): origin, the normal's tip and this run's tip
    var rel = -cx.arg(S.N), T = cx.turn(S.T, rel), xs = [0, cx.abs(S.N), T[0]], ys = [0, 0, T[1]];
    var x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xs), y0 = Math.min.apply(null, ys), y1 = Math.max.apply(null, ys);
    var sc = Math.min(0.84, 410 / Math.max(1, x1 - x0), 176 / Math.max(1, y1 - y0));
    var o = [15 - x0 * sc + (410 - (x1 - x0) * sc) / 2, 12 + y1 * sc + (176 - (y1 - y0) * sc) / 2];
    var p = function (c) { var t = cx.turn(c, rel); return [o[0] + sc * t[0], o[1] - sc * t[1]]; };
    var defs = el("defs", {}, svg);
    var arrow = function (a, b, cls) {
      var m = el("marker", { id: "wcHead-" + cls, viewBox: "0 0 10 10", refX: 8, refY: 5, markerWidth: 3.4, markerHeight: 3.4, orient: "auto" }, defs);
      el("path", { d: "M0 0L10 5L0 10Z", class: cls + "-head" }, m);
      el("line", { x1: a[0], y1: a[1], x2: b[0], y2: b[1], class: cls, "marker-end": "url(#wcHead-" + cls + ")" }, svg);
    };
    el("circle", { cx: o[0], cy: o[1], r: sc * cx.abs(S.N), class: "wc-grat" }, svg);   // the normal's size, for scale
    var n = p(S.N), t = p(S.T);
    arrow(o, n, "wc-v-normal"); arrow(n, t, "wc-v-anom"); arrow(o, t, "wc-v-total");
    $("wcSum").innerHTML = '<b class="wc-t-normal">normal ' + fmt(cx.abs(S.N)) + ' m</b> + <b class="wc-t-anom">anomaly ' + fmt(cx.abs(S.A)) +
      " m</b>, " + fmt(S.off) + '° apart = <b class="wc-t-total">this run ' + fmt(cx.abs(S.T)) + " m</b>";
  }

  // ── ERA5: the 100 hPa flux against where the anomaly sits ────────────────────────────────────────────────────────
  function drawCurve(S, k) {
    var svg = $("wcCurve"); svg.innerHTML = "";
    var W = 440, H = 270, L = 46, R = 14, T = 14, B = 44, e = S.era5, bins = e.bins.filter(function (b) { return b.n; });
    var top = Math.max.apply(null, bins.map(function (b) { return b.hi90; }).concat([e.mean, S.linked ? S.expect(0) : 0])) * 1.1;
    var X = function (d) { return L + d / 180 * (W - L - R); }, Y = function (v) { return H - B - v / top * (H - T - B); };
    var step = top > 30 ? 10 : top > 12 ? 5 : top > 5 ? 2 : 1;
    for (var v = 0; v <= top; v += step) {
      el("line", { x1: L, x2: W - R, y1: Y(v), y2: Y(v), class: "wc-gl" }, svg);
      el("text", { x: L - 6, y: Y(v) + 4, class: "wc-tick", "text-anchor": "end" }, svg, String(v));
    }
    [[0, "lined up"], [90, "90°"], [180, "opposed"]].forEach(function (t) {
      el("text", { x: X(t[0]), y: H - B + 18, class: "wc-tick", "text-anchor": t[0] === 0 ? "start" : t[0] === 180 ? "end" : "middle" }, svg, t[1]);
    });
    el("text", { x: (L + W - R) / 2, y: H - 6, class: "wc-tick", "text-anchor": "middle" }, svg, "anomaly's offset from the normal ridge (in phase)");
    el("text", { x: 12, y: (T + H - B) / 2, class: "wc-tick", "text-anchor": "middle", transform: "rotate(-90 12 " + (T + H - B) / 2 + ")" }, svg, "K m/s");
    el("line", { x1: L, x2: W - R, y1: Y(e.mean), y2: Y(e.mean), class: "wc-usual" }, svg);
    el("text", { x: W - R, y: Y(e.mean) - 5, class: "wc-tick", "text-anchor": "end" }, svg, "usual for the date " + fmt(e.mean, 1));
    bins.forEach(function (b) {
      var x = X((b.lo + b.hi) / 2);
      el("line", { x1: x, x2: x, y1: Y(b.lo90), y2: Y(b.hi90), class: "wc-bar" }, svg);
      el("circle", { cx: x, cy: Y(b.mean), r: 4.5, class: "wc-bin" }, svg);
    });
    if (!S.linked) return;
    var d = [];
    for (var o = 0; o <= 180; o += 3) d.push((o ? "L" : "M") + X(o).toFixed(1) + " " + Y(S.expect(o)).toFixed(1));
    el("path", { d: d.join(""), class: "wc-fit" }, svg);
    el("circle", { cx: X(S.off), cy: Y(S.flux), r: 6.5, class: "wc-now" }, svg);
  }

  // ── chips and words ──────────────────────────────────────────────────────────────────────────────────────────────
  function chip(l, v, s, cls) { return '<div class="chip"><div class="l">' + l + '</div><div class="v' + (cls ? " " + cls : "") + '">' + v + '</div><div class="s">' + s + "</div></div>"; }
  function modelFlux(doc, k, day) {
    var h = fluxes[st.model], nh = h && h.hemispheres && h.hemispheres.nh;
    if (!nh || String(h.init).slice(0, 13).replace(" ", "T") !== doc.init.slice(0, 13)) return null;   // another run
    var i0 = h.valid.indexOf(doc.valid[day]), lead = doc.waves[String(k)].era5.lead_days[1];
    if (i0 < 0 || i0 + lead >= h.valid.length) return null;
    var s = 0; for (var i = i0; i <= i0 + lead; i++) s += nh["k" + k].mean[i];
    return s / (lead + 1);
  }
  function words(doc, S, k, day) {
    var e = S.era5, lead = e.lead_days[1], v = doc.valid[day].slice(0, 10), rel = cx.abs(S.T) / cx.abs(S.N);
    var where = !S.agree ? "the members disagree where it sits" : S.off < 45 ? "lined up with the normal ridges" : S.off > 135 ? "opposed: on the normal troughs" : "partly lined up";
    var turned = Math.abs(st.rot) > 0.5 ? " (turned " + fmt(Math.abs(st.rot)) + "° " + (st.rot > 0 ? "east" : "west") + ")" : "";
    var mf = Math.abs(st.rot) > 0.5 ? null : modelFlux(doc, k, day);
    var to = "day" + (lead ? "s " + day + "–" + (day + lead) : " " + day);
    $("wcChips").innerHTML =
      chip("Day " + day + " · " + v, (rel >= 1 ? fmt(rel, 1) + "×" : fmt(rel, 2) + "×") + " normal", "wave-" + k + " at 500 hPa, 55–65°N" + turned, rel >= 1.25 ? "wc-up" : rel <= 0.8 ? "wc-down" : "") +
      chip("The anomaly", fmt(S.off) + "° off", where) +
      (S.linked ? chip("ERA5 expects, " + to, fmt(S.flux, 1) + " K m/s", "wave-" + k + " heat flux at 100 hPa; usual " + fmt(e.mean, 1), S.flux >= 1.5 * e.mean ? "wc-up" : S.flux <= 0.67 * e.mean ? "wc-down" : "") : "") +
      (mf !== null ? chip(LABEL[st.model] + "'s own flux, " + to, fmt(mf, 1) + " K m/s", "wave-" + k + " at 100 hPa, the members' mean") : "");
    var lined = e.bins[0].mean, opp = e.bins[e.bins.length - 1].mean;
    $("wcCurveCap").textContent = "ERA5 " + e.years[0] + "–" + e.years[1] + ", days within " + e.window_days + " of " + v.slice(5) +
      ": the wave-" + k + " heat flux at 100 hPa over " + (lead + 1) + " days, by where a 500 hPa anomaly of " + fmt(e.big_m) + " m or more sat (dots, 90% range over winters): " +
      fmt(lined, 1) + " K m/s lined up, " + fmt(opp, 1) + " opposed." +
      (S.linked ? " Line: this run's anomaly on day " + day + " turned round the dial (ERA5 fit, r = " + fmt(e.fit.r, 2) + "); ring: where it sits." : " At this time of year ERA5 shows no significant link, so no expectation is drawn.");
  }

  // ── the lean schematic: air following a meander, coloured by its temperature ─────────────────────────────────────
  var parcels = null, anim = null;
  function leanFlux(lean, size) { return size * size * Math.sin(lean * D2R); }   // the two levels' amplitudes x sin(lean)
  function drawLean() {
    var svg = $("wcLean"); if (!svg) return; svg.innerHTML = "";
    var W = 520, H = 220, y0 = 110, amp = 26 * st.size, k = 2, lean = st.lean * D2R, g = 1.6;
    var X = function (x) { return 20 + x / (2 * Math.PI) * (W - 120); };
    // layer-mean height (500 and 100 hPa, the upper ridge `lean` west) sets the path; the thickness, the temperature
    var zbar = function (p) { return (Math.cos(k * p) + g * Math.cos(k * p + lean)) / (1 + g); };
    var temp = function (p) { return (g * Math.cos(k * p + lean) - Math.cos(k * p)) / (g + 1); };
    var d = [];
    for (var i = 0; i <= 200; i++) { var ph = i / 200 * 2 * Math.PI; d.push((i ? "L" : "M") + X(ph).toFixed(1) + " " + (y0 - amp * zbar(ph)).toFixed(1)); }
    for (i = 0; i < 100; i++) {
      var a = i / 100 * 2 * Math.PI, tv = temp(a + Math.PI / 100);
      el("rect", { x: X(a), y: 20, width: X(2 * Math.PI / 100) - X(0) + 0.6, height: H - 50, class: tv > 0 ? "wc-warm" : "wc-cool", "fill-opacity": Math.min(1, Math.abs(tv) * st.size / 1.6) * 0.5 }, svg);
    }
    el("path", { d: d.join(""), class: "wc-stream" }, svg);
    el("text", { x: 22, y: 16, class: "wc-tick" }, svg, "north ↑");
    el("text", { x: X(Math.PI), y: H - 10, class: "wc-tick", "text-anchor": "middle" }, svg, "east →  (the band round 60°N, unrolled; shading: the layer's temperature)");
    // the heat carried north: the mean of (northward wind × temperature) along the band
    var f = leanFlux(st.lean, st.size), fx = W - 70, base = 160, full = 4 * Math.sin(80 * D2R);
    el("rect", { x: fx, y: 30, width: 34, height: base - 30, class: "wc-gauge" }, svg);
    var hgt = Math.max(-1, Math.min(1, f / full)) * (base - 40);
    el("rect", { x: fx, y: hgt > 0 ? base - hgt - 10 : base - 10, width: 34, height: Math.abs(hgt) + 0.5, class: hgt >= 0 ? "wc-warm" : "wc-cool" }, svg);
    el("text", { x: fx + 17, y: base + 14, class: "wc-tick", "text-anchor": "middle" }, svg, f >= 0 ? "heat north" : "heat south");
    parcels = { X: X, y0: y0, amp: amp, zbar: zbar, temp: temp, g: svg.appendChild(document.createElementNS(NS, "g")) };
    if (!anim && !matchMedia("(prefers-reduced-motion: reduce)").matches) anim = requestAnimationFrame(tick);
    else if (!anim) tick(0, true);
  }
  function tick(t, still) {
    anim = null;
    var P = parcels, svg = $("wcLean");
    if (!P || !svg || !svg.isConnected) return;
    P.g.innerHTML = "";
    for (var i = 0; i < 24; i++) {
      var ph = ((i / 24) * 2 * Math.PI + (still ? 0 : t / 2600)) % (2 * Math.PI), y = P.y0 - P.amp * P.zbar(ph), tv = P.temp(ph);
      var y2 = P.y0 - P.amp * P.zbar(ph + 0.12);
      el("line", { x1: P.X(ph), y1: y, x2: P.X(ph + 0.12), y2: y2, class: "wc-parcel " + (tv > 0 ? "wc-parcel-warm" : "wc-parcel-cool") }, P.g);
      el("circle", { cx: P.X(ph + 0.12), cy: y2, r: 4, class: tv > 0 ? "wc-parcel-warm" : "wc-parcel-cool" }, P.g);
    }
    if (!still) anim = requestAnimationFrame(tick);
  }
  function seedLean(doc, k, day) {   // the run as issued (not turned): the 100 hPa ridge's offset west of the 500 hPa one
    var w = doc.waves[String(k)], up = w.mean_100[day], low = [0, 0], n = w.members[day].length;
    w.members[day].forEach(function (m) { low = [low[0] + m[0] / n, low[1] + m[1] / n]; });
    var lean = wrap((cx.arg(up) - cx.arg(low)) / D2R);
    return { lean: Math.max(-90, Math.min(90, lean)), size: Math.max(0.5, Math.min(3, cx.abs(low) / cx.abs(w.normal[day]))), raw: lean };
  }

  // ── redraw and controls ──────────────────────────────────────────────────────────────────────────────────────────
  function redraw() {
    var doc = docs[st.model]; if (!doc) return;
    var S = state(doc, st.k, st.day, st.rot);
    drawMap(S, st.k); drawPhasor(S); drawCurve(S, st.k); words(doc, S, st.k, st.day);
    $("wcDay").value = st.day; $("wcDate").textContent = doc.valid[st.day].slice(0, 10);
    Array.prototype.forEach.call(document.querySelectorAll("#wcTurn button"), function (b) {
      var on = b.dataset.r === "0" ? Math.abs(st.rot) < 0.5 : Math.abs(Math.abs(st.rot) - 180 / st.k) < 0.5;
      b.classList.toggle("on", on); b.setAttribute("aria-pressed", on ? "true" : "false");
    });
    var sd = seedLean(doc, st.k, st.day);
    $("wcLeanRun").textContent = "This run, day " + st.day + ": the 100 hPa ridge " + fmt(Math.abs(sd.raw)) + "° " + (sd.raw >= 0 ? "west" : "east") + " of the 500 hPa ridge (in phase), the wave " + fmt(sd.size, 1) + "× its normal size";
  }
  function bestDay(doc, k) {
    var best = 1, top = -1;
    for (var d = 0; d <= Math.min(10, doc.valid.length - 1); d++) { var S = state(doc, k, d, 0); if (S.linked && S.flux > top) { top = S.flux; best = d; } }
    return best;
  }
  function bind() {
    if (bind.done) return; bind.done = true;
    dragMap($("wcMap"));
    $("wcDay").addEventListener("input", function () { st.day = +this.value; stop(); redraw(); });
    $("wcPlay").addEventListener("click", function () {
      if (st.timer) return stop();
      var doc = docs[st.model]; if (!doc) return;
      if (st.day >= doc.valid.length - 1) st.day = 0;
      this.textContent = "Pause"; this.setAttribute("aria-pressed", "true");
      st.timer = setInterval(function () { if (st.day >= doc.valid.length - 1) return stop(); st.day++; redraw(); }, 900);
    });
    Array.prototype.forEach.call(document.querySelectorAll("#wcTurn button"), function (b) {
      b.addEventListener("click", function () { st.rot = b.dataset.r === "0" ? 0 : 180 / st.k; redraw(); });
    });
    $("wcLeanSlider").addEventListener("input", function () { st.lean = +this.value; $("wcLeanVal").textContent = leanText(); drawLean(); });
    $("wcSizeSlider").addEventListener("input", function () { st.size = +this.value; $("wcSizeVal").textContent = fmt(st.size, 1) + "×"; drawLean(); });
    $("wcLeanSeed").addEventListener("click", function () {
      var doc = docs[st.model]; if (!doc) return;
      var sd = seedLean(doc, st.k, st.day); st.lean = Math.round(sd.lean); st.size = Math.round(sd.size * 10) / 10;
      $("wcLeanSlider").value = st.lean; $("wcSizeSlider").value = st.size; $("wcLeanVal").textContent = leanText(); $("wcSizeVal").textContent = fmt(st.size, 1) + "×"; drawLean();
    });
  }
  function leanText() { return st.lean === 0 ? "upright" : fmt(Math.abs(st.lean)) + "° " + (st.lean > 0 ? "west" : "east"); }
  function stop() { if (st.timer) clearInterval(st.timer); st.timer = null; var b = $("wcPlay"); if (b) { b.textContent = "Play"; b.setAttribute("aria-pressed", "false"); } }

  function draw(sel) {
    var model = sel.b || "aifs", k = sel.a === "2" ? 2 : 1;
    bind(); stop();
    var fresh = model !== st.model || k !== st.k;
    st.model = model; st.k = k; if (fresh) st.rot = 0;
    var go = function () {
      var doc = docs[model];
      if (!doc) { $("wcChips").innerHTML = ""; $("wcNote").textContent = "The " + LABEL[model] + " compass has not been published yet: it appears with that ensemble's next run."; return; }
      $("wcDay").max = doc.valid.length - 1;
      if (fresh || st.day === null || st.day >= doc.valid.length) st.day = bestDay(doc, k);
      $("wcNote").textContent = doc.label + " run of " + doc.init.slice(0, 10) + " " + doc.init.slice(11, 13) + "Z, " + doc.n_members +
        " members. Drag the map (or use the arrow keys on it) to move the anomaly round the pole: the normal wave stays where the mountains and coasts put it.";
      redraw(); drawLean(); $("wcLeanVal").textContent = leanText(); $("wcSizeVal").textContent = fmt(st.size, 1) + "×";
    };
    var want = [];
    if (!docs[model]) want.push(getJSON("/assets/sst/data/wave_compass_" + model + ".json?v=" + hour()).then(function (d) { docs[model] = d; }));
    if (!(model in fluxes)) want.push(getJSON("/assets/sst/data/" + FILES[model] + "?v=" + hour()).then(function (d) { fluxes[model] = d; }));
    if (!coast) want.push(getJSON("/assets/models/geo_polar_coarse.json").then(function (d) { coast = d || { layers: { coast: { q: 1, lines: [] } } }; }));
    if (want.length) Promise.all(want).then(go); else go();
  }
  window.WaveCompass = { draw: draw };
})();
