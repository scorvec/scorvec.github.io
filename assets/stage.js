/* stage.js — the GEPS-page viewer, shared (2026-09-07). A page defines window.STAGE = {groups, products}
   and includes the .ss-layout markup (rail, crumb, opts-a/b/c, stage, cap, about, hint); this script
   builds the rail, the option rows and the one figure on stage. Products:
     label, fit (false = natural height), a/b/c option groups {label, items:[[v,label],…]} or a
     function of the earlier choices, and ONE of
       img(a,b,c)   -> image url            frame(a,b,c) -> iframe url (sst_anim embed), ratio(a,b,c)
       dom(a,b,c)   -> id of a hidden element to mount (interactive blocks, widgets)
     cap(a,b,c) -> caption text; about -> id of a <template> whose HTML goes under the caption
     (the vetted explanatory text and sources, collapsed); bust:"hourly" appends an hour cache key.
   Selection lives in the hash (#product/a/b/c); ← → step the deepest option row, ↑ ↓ the product. */
(function () {
  "use strict";
  var S = window.STAGE; if (!S) return;
  var P = S.products, GROUPS = S.groups;
  var ORDER = []; GROUPS.forEach(function (g) { g.items.forEach(function (p) { ORDER.push(p[0]); }); });
  var sel = { p: ORDER[0], a: null, b: null, c: null };
  var $ = function (id) { return document.getElementById(id); };
  var mounted = null, mountedHome = null, pinMax = 0;

  function buildRail() {
    var host = $("rail"); host.innerHTML = "";
    // accordion: only the group holding the selected figure is expanded, the others show their
    // title and count (user 2026-09-07: "rearrange them somehow so you don't have to scroll")
    GROUPS.forEach(function (g) {
      var d = document.createElement("div"); d.className = "rail-group"; d.dataset.g = g.label;
      var t = document.createElement("button"); t.type = "button"; t.className = "rail-title";
      t.innerHTML = '<span class="car"></span>' + (g.ico ? '<span class="ico">' + g.ico + '</span>' : "") + '<span class="lbl">' + g.label + '</span><span class="n">' + g.items.length + '</span>';
      t.onclick = function () { d.classList.toggle("open"); }; d.appendChild(t);
      g.items.forEach(function (p) {
        var b = document.createElement("button"); b.type = "button"; b.dataset.p = p[0];
        b.innerHTML = (g.ico ? '<span class="ico">' + g.ico + '</span>' : "") + '<span>' + p[1] + (p[2] ? '<small>' + p[2] + '</small>' : "") + '</span>';
        b.onclick = function () { sel = { p: p[0], a: null, b: null, c: null }; render(); };
        d.appendChild(b);
      });
      host.appendChild(d);
    });
  }
  function buttons(host, group, key) {
    host.innerHTML = "";
    if (!group) { sel[key] = null; return null; }
    if (group.label) { var l = document.createElement("span"); l.className = "seg-label"; l.textContent = group.label; host.appendChild(l); }
    var valid = group.items.some(function (it) { return it[0] === sel[key]; });
    group.items.forEach(function (it, i) {
      var b = document.createElement("button"); b.className = "pill"; b.type = "button"; b.textContent = it[1]; b.dataset.v = it[0];
      if ((valid && sel[key] === it[0]) || (!valid && i === 0)) { b.classList.add("on"); sel[key] = it[0]; }
      b.onclick = function () { sel[key] = it[0]; render(); };
      host.appendChild(b);
    });
    return sel[key];
  }
  // SIZING (reworked 2026-09-09 after "the images are showing up way too small and not fitting
  // properly" on a 1366x768 laptop). The width leads: a figure is drawn at the full stage width
  // unless that would make it taller than tallCap(), and only then is it scaled down. The old rule
  // capped every figure at the first-screen height and let object-fit letterbox the rest, which
  // turned a portrait panel (e.g. the 1312x1025 SST loop) into a 307 px thumbnail in a 975 px stage.
  function stageWidth() {
    var st = $("stage"), cs = getComputedStyle(st);
    return st.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight);
  }
  // height that fits on the first screen: the viewport minus what sits above the figure once the
  // panel is scrolled to the top (site header 74 px + crumb + option rows) and the caption below it
  function stageCap() {
    var st = $("stage"), main = document.querySelector(".ss-main");
    var above = st.getBoundingClientRect().top - main.getBoundingClientRect().top + 74;
    var avail = window.innerHeight - above - Math.max($("cap").offsetHeight, 40) - 30;
    return Math.round(Math.max(360, avail));
  }
  // …and the height a figure may take before it is shrunk at all: a full viewport. A tall panel is
  // then whole on the screen after one scroll — much larger than the old first-screen cap allowed,
  // without ever being taller than the window.
  function tallCap() { return Math.round(Math.max(stageCap(), window.innerHeight * 0.98)); }
  function fitStage() {
    var st = $("stage"), img = st.querySelector("img"), fr = st.querySelector("iframe");
    if (!img && !fr) return;
    var w = stageWidth();
    if (img) {
      // span the stage; only a figure that would then be taller than the budget is narrowed, and it
      // is narrowed by width (so the box always equals the picture — nothing is letterboxed)
      img.style.width = "100%"; img.style.maxHeight = "";
      if (img.naturalWidth && img.naturalHeight) {
        var full = w * img.naturalHeight / img.naturalWidth;      // height at 100% of the stage width
        var lim = w < 700 ? Infinity : tallCap();   // phones: width is the scarce axis — never narrow
        if (full > lim) img.style.width = Math.round(lim * img.naturalWidth / img.naturalHeight) + "px";
      }
    }
    if (fr) {                                     // placeholder box, before the embed posts its size
      var m = /^\s*([\d.]+)\s*\/\s*([\d.]+)/.exec(fr.style.aspectRatio || "");
      if (m) {
        var fullf = w * parseFloat(m[2]) / parseFloat(m[1]);
        fr.style.maxWidth = (w >= 700 && fullf > tallCap()) ? Math.round(tallCap() * parseFloat(m[1]) / parseFloat(m[2])) + "px" : "";
      }
    }
  }
  addEventListener("resize", fitStage);
  function groupOf(p) { for (var i = 0; i < GROUPS.length; i++) for (var j = 0; j < GROUPS[i].items.length; j++) if (GROUPS[i].items[j][0] === p) return [GROUPS[i].label, GROUPS[i].items[j][1]]; return ["", p]; }
  function labelOf(group, v) { if (!group) return null; for (var i = 0; i < group.items.length; i++) if (group.items[i][0] === v) return group.items[i][1]; return null; }
  function hourKey() { return new Date().toISOString().slice(0, 13); }
  function unmount() {
    if (mounted) { mountedHome.parent.insertBefore(mounted, mountedHome.next && mountedHome.parent.contains(mountedHome.next) ? mountedHome.next : null); mounted = null; }
  }
  function render() {
    var p = P[sel.p]; if (!p) { sel.p = ORDER[0]; p = P[sel.p]; }
    Array.prototype.forEach.call(document.querySelectorAll("#rail button[data-p]"), function (b) { b.classList.toggle("on", b.dataset.p === sel.p); });
    Array.prototype.forEach.call(document.querySelectorAll("#rail .rail-group"), function (d) {
      var mine = !!d.querySelector('button[data-p="' + sel.p + '"]'); d.classList.toggle("has", mine); if (mine) d.classList.add("open"); else d.classList.remove("open");
    });
    var ga = typeof p.a === "function" ? p.a() : (p.a || null); buttons($("opts-a"), ga, "a");
    var gb = typeof p.b === "function" ? p.b(sel.a) : (p.b || null); buttons($("opts-b"), gb, "b");
    var gc = typeof p.c === "function" ? p.c(sel.a, sel.b) : (p.c || null); buttons($("opts-c"), gc, "c");
    var st = $("stage"); unmount(); st.innerHTML = ""; pinMax = 0;
    var domId = p.dom && p.dom(sel.a, sel.b, sel.c);   // dom() may return null for an option that is a still
    if (domId) {
      var el = $(domId);
      if (el) { mountedHome = { parent: el.parentNode, next: el.nextSibling }; mounted = el; st.appendChild(el); el.hidden = false;
        if (window.Plotly) Array.prototype.forEach.call(el.querySelectorAll(".js-plotly-plot"), function (g) { try { window.Plotly.Plots.resize(g); } catch (e) {} }); }
    } else if (p.frame && p.frame(sel.a, sel.b, sel.c)) {          // frame() may return null for an option that is a still
      var f = document.createElement("iframe"); f.title = p.label; f.loading = "lazy";
      var fsrc = p.frame(sel.a, sel.b, sel.c); f.src = fsrc + (fsrc.indexOf("?") < 0 ? "?" : "&") + "maxh=" + (stageWidth() < 700 ? 4000 : tallCap());  // the embed caps its picture to the same budget
      f.style.aspectRatio = (p.ratio ? p.ratio(sel.a, sel.b, sel.c) : "1259/700"); st.appendChild(f); fitStage();
    } else {
      var src = p.img(sel.a, sel.b, sel.c); if (p.bust === "hourly") src += (src.indexOf("?") < 0 ? "?" : "&") + "v=" + hourKey();
      var im = document.createElement("img"); im.src = src; im.alt = p.label; st.appendChild(im);
      im.onload = fitStage; im.onclick = function () { var lb = $("lightbox"); lb.querySelector("img").src = im.src; lb.classList.add("on"); };
    }
    $("cap").innerHTML = p.cap ? p.cap(sel.a, sel.b, sel.c) : "";
    var ab = $("about"); ab.innerHTML = "";
    var aboutId = typeof p.about === "function" ? p.about(sel.a, sel.b, sel.c) : p.about;
    if (aboutId && $(aboutId)) {
      var d = document.createElement("details"); d.className = "about";
      d.innerHTML = "<summary>About this figure and its sources</summary>" + $(aboutId).innerHTML; ab.appendChild(d);
      if (window.renderMathInElement) { try { window.renderMathInElement(d, { delimiters: [{ left: "$$", right: "$$", display: true }, { left: "$", right: "$", display: false }] }); } catch (e) {} }
    }
    var g = groupOf(sel.p), parts = [g[0], g[1]];
    [[ga, sel.a], [gb, sel.b], [gc, sel.c]].forEach(function (x) { var l = labelOf(x[0], x[1]); if (l && x[0].items.length > 1) parts.push(l); });
    $("crumb").innerHTML = parts.map(function (t) { return "<b>" + t + "</b>"; }).join("<span>›</span>");
    fitStage();
    var h = "#" + [sel.p, sel.a, sel.b, sel.c].filter(function (x) { return x; }).join("/");
    if (location.hash !== h) history.replaceState(null, "", h);
    window.dispatchEvent(new Event("resize"));
  }
  function lastRow() { var rows = ["c", "b", "a"]; for (var i = 0; i < rows.length; i++) { var bs = document.querySelectorAll("#opts-" + rows[i] + " button"); if (bs.length > 1) return bs; } return null; }
  function step(dir) { var bs = lastRow(); if (!bs) return stepProduct(dir); var i = -1; for (var k = 0; k < bs.length; k++) if (bs[k].classList.contains("on")) i = k; bs[(i + dir + bs.length) % bs.length].click(); }
  function stepProduct(dir) { var i = ORDER.indexOf(sel.p); sel = { p: ORDER[(i + dir + ORDER.length) % ORDER.length], a: null, b: null, c: null }; render(); }
  $("prevBtn").onclick = function () { step(-1); }; $("nextBtn").onclick = function () { step(1); };
  addEventListener("keydown", function (e) {
    if (e.target && /INPUT|SELECT|TEXTAREA/.test(e.target.tagName)) return;
    if (e.key === "ArrowRight") { step(1); e.preventDefault(); } else if (e.key === "ArrowLeft") { step(-1); e.preventDefault(); }
    else if (e.key === "ArrowDown") { stepProduct(1); e.preventDefault(); } else if (e.key === "ArrowUp") { stepProduct(-1); e.preventDefault(); }
    else if (e.key === "Escape") { $("lightbox").classList.remove("on"); }
  });
  $("lightbox").onclick = function () { this.classList.remove("on"); };
  addEventListener("message", function (e) {
    var x = e.data; if (!x || x.type !== "sstAnimHeight") return;
    var f = document.querySelector("#stage iframe");
    if (f && f.contentWindow === e.source) {
      f.style.height = x.h + "px"; f.style.aspectRatio = "auto";
      // pin the iframe to the picture's width only when that is close to the stage width (a portrait
      // loop hugging its figure). A much narrower ask means the embed shrank itself, and honouring it
      // would wrap its region bar and shrink it again — leave the frame full width and let it re-measure.
      var sw = stageWidth();
      // Only ever WIDEN the pin within one render: honouring a smaller ask can wrap the embed's
      // region bar, which shrinks its picture, which posts a smaller width again — a downward
      // ratchet that once left a 1312x1025 loop 307 px wide. An ask under half the stage is that
      // pathology, not a portrait figure, so it is ignored outright.
      if (x.w > pinMax) pinMax = x.w;
      f.style.maxWidth = (sw >= 700 && pinMax > sw * 0.5 && pinMax < sw * 0.98) ? pinMax + "px" : "";
    }
  });
  addEventListener("hashchange", function () { var h = location.hash.replace(/^#/, "").split("/"); if (h[0] && P[h[0]]) { sel = { p: h[0], a: h[1] || null, b: h[2] || null, c: h[3] || null }; render(); } });
  buildRail();
  var h = location.hash.replace(/^#/, "").split("/");
  if (h[0] && P[h[0]]) sel = { p: h[0], a: h[1] || null, b: h[2] || null, c: h[3] || null };
  render();
})();
