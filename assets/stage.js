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
  var mounted = null, mountedHome = null;

  function buildRail() {
    var host = $("rail"); host.innerHTML = "";
    GROUPS.forEach(function (g) {
      var d = document.createElement("div"); d.className = "rail-group";
      var t = document.createElement("div"); t.className = "rail-title"; t.textContent = g.label; d.appendChild(t);
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
  function fitStage() {
    var st = $("stage"), img = st.querySelector("img");
    if (!img) return;
    if (P[sel.p].fit === false) { img.style.maxHeight = ""; img.style.width = "100%"; return; }
    var avail = window.innerHeight - st.getBoundingClientRect().top - $("cap").offsetHeight - 34;
    img.style.maxHeight = Math.max(280, avail) + "px"; img.style.width = "auto";
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
    Array.prototype.forEach.call(document.querySelectorAll("#rail button"), function (b) { b.classList.toggle("on", b.dataset.p === sel.p); });
    var ga = typeof p.a === "function" ? p.a() : (p.a || null); buttons($("opts-a"), ga, "a");
    var gb = typeof p.b === "function" ? p.b(sel.a) : (p.b || null); buttons($("opts-b"), gb, "b");
    var gc = typeof p.c === "function" ? p.c(sel.a, sel.b) : (p.c || null); buttons($("opts-c"), gc, "c");
    var st = $("stage"); unmount(); st.innerHTML = "";
    if (p.dom) {
      var el = $(p.dom(sel.a, sel.b, sel.c));
      if (el) { mountedHome = { parent: el.parentNode, next: el.nextSibling }; mounted = el; st.appendChild(el); el.hidden = false;
        if (window.Plotly) Array.prototype.forEach.call(el.querySelectorAll(".js-plotly-plot"), function (g) { try { window.Plotly.Plots.resize(g); } catch (e) {} }); }
    } else if (p.frame && p.frame(sel.a, sel.b, sel.c)) {          // frame() may return null for an option that is a still
      var f = document.createElement("iframe"); f.src = p.frame(sel.a, sel.b, sel.c); f.title = p.label; f.loading = "lazy";
      f.style.aspectRatio = (p.ratio ? p.ratio(sel.a, sel.b, sel.c) : "1259/700"); st.appendChild(f);
    } else {
      var src = p.img(sel.a, sel.b, sel.c); if (p.bust === "hourly") src += (src.indexOf("?") < 0 ? "?" : "&") + "v=" + hourKey();
      var im = document.createElement("img"); im.src = src; im.alt = p.label; st.appendChild(im);
      im.onload = fitStage; im.onclick = function () { var lb = $("lightbox"); lb.querySelector("img").src = im.src; lb.classList.add("on"); };
    }
    $("cap").innerHTML = p.cap ? p.cap(sel.a, sel.b, sel.c) : "";
    var ab = $("about"); ab.innerHTML = "";
    if (p.about && $(p.about)) {
      var d = document.createElement("details"); d.className = "about";
      d.innerHTML = "<summary>About this figure and its sources</summary>" + $(p.about).innerHTML; ab.appendChild(d);
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
    if (f && f.contentWindow === e.source) { f.style.height = x.h + "px"; f.style.aspectRatio = "auto"; f.style.maxWidth = x.w ? x.w + "px" : ""; }
  });
  addEventListener("hashchange", function () { var h = location.hash.replace(/^#/, "").split("/"); if (h[0] && P[h[0]]) { sel = { p: h[0], a: h[1] || null, b: h[2] || null, c: h[3] || null }; render(); } });
  buildRail();
  var h = location.hash.replace(/^#/, "").split("/");
  if (h[0] && P[h[0]]) sel = { p: h[0], a: h[1] || null, b: h[2] || null, c: h[3] || null };
  render();
})();
