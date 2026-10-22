/* stage.js — the GEPS-page viewer, shared between pages. A page defines window.STAGE = {groups, products}
   and includes the .ss-layout markup (rail, crumb, opts-a/b/c, stage, cap, about, hint); this script
   builds the rail, the option rows and the one figure on stage. Products:
     label, fit (false = natural height), a/b/c option groups {label, items:[[v,label],…]} or a
     function of the earlier choices, and ONE of
       img(a,b,c)   -> image url            frame(a,b,c) -> iframe url (sst_anim embed), ratio(a,b,c)
       dom(a,b,c)   -> id of a hidden element to mount (interactive blocks, widgets)
     cap(a,b,c) -> caption text; about -> id of a <template> whose HTML goes under the caption
     (the vetted explanatory text and sources, collapsed); bust:"hourly" appends an hour cache key.
     STAGE.version() -> the run's key (or null), appended to every other still as ?v= so a new run's figures, which keep
       their file names, never come from a browser cache; the page calls STAGE.refresh() when its key arrives.
     An option row of more than MANY choices is one menu (a <select>) on narrow screens instead of a wall of pills.
     gated:true (dom) -> the block's own script shows/hides it and a hidden one leaves the rail;
     ownHash:true (dom) -> the block keeps its own state after "#id/" in the hash and the stage leaves it; a function of
       (a,b,c) does so for the views it is true for, after "#id/a/" (a product whose parts mount different blocks).
   STAGE.aliases maps old ids to products, or to a product and its options ("product/a"); an old link's own state after
   the id (#oldid/x/y) follows it (#product/a/x/y). STAGE.related = false
   drops the related-plots strip; STAGE.onRender(sel) runs after each render. An optional #stage-stamp element
   shows the stamp the figure on stage reports for itself.
   Selection lives in the hash (#product/a/b/c); with the focus in the viewer ← → step the deepest option row and,
   in the list of plots, ↑ ↓ the product. */
(function () {
  "use strict";
  var S = window.STAGE; if (!S) return;
  var P = S.products, GROUPS = S.groups;
  var AL = S.aliases || {};              // old product ids -> live ones ("id" or "id/a/b"), so old deep links land on a view
  var ORDER = []; GROUPS.forEach(function (g) { g.items.forEach(function (p) { ORDER.push(p[0]); }); });
  var sel = { p: ORDER[0], a: null, b: null, c: null };
  var $ = function (id) { return document.getElementById(id); };
  var mounted = null, mountedHome = null, mountedShown = false, pinMax = 0, wanted = null, deferHash = null;
  addEventListener("load", function () {
    if (deferHash && !location.hash) history.replaceState(null, "", deferHash);
    deferHash = null;
  });

  function buildRail() {
    var host = $("rail"); host.innerHTML = "";
    // accordion: only the group holding the selected figure is expanded, the others show their
    // title and count, so the rail fits without scrolling
    GROUPS.forEach(function (g) {
      var d = document.createElement("div"); d.className = "rail-group"; d.dataset.g = g.label;
      var t = document.createElement("button"); t.type = "button"; t.className = "rail-title"; t.setAttribute("aria-expanded", "false");
      t.innerHTML = '<span class="car" aria-hidden="true"></span>' + (g.ico ? '<span class="ico" aria-hidden="true">' + g.ico + '</span>' : "") + '<span class="lbl">' + g.label + '</span><span class="n">' + g.items.length + '</span>';
      t.onclick = function () { t.setAttribute("aria-expanded", d.classList.toggle("open") ? "true" : "false"); }; d.appendChild(t);
      g.items.forEach(function (p) {
        var b = document.createElement("button"); b.type = "button"; b.dataset.p = p[0];
        b.innerHTML = (g.ico ? '<span class="ico" aria-hidden="true">' + g.ico + '</span>' : "") + '<span>' + p[1] + (p[2] ? '<small>' + p[2] + '</small>' : "") + '</span>';
        b.onclick = function () { choose(p[0]); };
        d.appendChild(b);
      });
      host.appendChild(d);
    });
    buildPicker(host);
  }
  // Pick a product. A new figure is a new history entry (back returns to the previous one); option
  // changes within a figure only replace the entry.
  var pushNext = false;
  function choose(id) { if (!P[id]) return; sel = { p: id, a: null, b: null, c: null }; pushNext = true; render(); }

  // ── Phones and narrow windows. Below 1000 px the rail's wall of pills gives way to one compact bar: previous /
  // "Choose a plot: <current>" / next, which opens a grouped list in a sheet. The figure sits right under it,
  // and a clear horizontal swipe on the figure steps to the previous / next plot. ──
  var pick = null, sheet = null, sheetReturn = null, live = null;
  function narrow() { return window.matchMedia && window.matchMedia("(max-width: 1000px)").matches; }
  function buildPicker(host) {
    pick = document.createElement("div"); pick.className = "ss-pick";
    pick.innerHTML =
      '<button type="button" class="ss-pick-step" data-d="-1" aria-label="Previous plot"><span aria-hidden="true">\u2039</span></button>' +
      '<button type="button" class="ss-pick-btn" aria-haspopup="dialog" aria-expanded="false">' +
        '<span class="ss-pick-k">Choose a plot</span><span class="ss-pick-t"></span><span class="ss-pick-n"></span></button>' +
      '<button type="button" class="ss-pick-step" data-d="1" aria-label="Next plot"><span aria-hidden="true">\u203a</span></button>' +
      '<span class="ss-sr" aria-live="polite"></span>';
    host.insertBefore(pick, host.firstChild);
    live = pick.querySelector(".ss-sr");
    pick.querySelector(".ss-pick-btn").onclick = openSheet;
    Array.prototype.forEach.call(pick.querySelectorAll(".ss-pick-step"), function (b) {
      b.onclick = function () { stepProduct(+b.dataset.d); };
    });
  }
  function availableOrder() { return ORDER.filter(available); }
  function updatePicker(g) {
    if (!pick) return;
    var av = availableOrder(), i = av.indexOf(sel.p);
    pick.querySelector(".ss-pick-t").textContent = g[1];
    pick.querySelector(".ss-pick-n").textContent = (i + 1) + " / " + av.length;
    pick.querySelector(".ss-pick-btn").setAttribute("aria-label", "Choose a plot. Showing " + g[1] + ", " + (i + 1) + " of " + av.length + ", in " + g[0]);
  }
  function buildSheet() {
    sheet = document.createElement("div"); sheet.className = "ss-sheet"; sheet.hidden = true;
    sheet.innerHTML = '<div class="ss-sheet-scrim" data-close></div>' +
      '<div class="ss-sheet-box" role="dialog" aria-modal="true" aria-labelledby="ss-sheet-h">' +
      '<div class="ss-sheet-top"><h2 id="ss-sheet-h">Choose a plot</h2><button type="button" class="ss-sheet-x" data-close>Close</button></div>' +
      '<div class="ss-sheet-list"></div></div>';
    document.body.appendChild(sheet);
    sheet.addEventListener("click", function (e) {
      if (e.target.closest("[data-close]")) { closeSheet(); return; }
      var b = e.target.closest("button[data-p]"); if (b) { closeSheet(); choose(b.dataset.p); }
    });
    sheet.addEventListener("keydown", function (e) {
      if (e.key === "Escape") { e.preventDefault(); closeSheet(); return; }
      if (e.key === "Tab") {                                   // keep focus inside the sheet
        var f = Array.prototype.slice.call(sheet.querySelectorAll("button:not([hidden])"));
        var i = f.indexOf(document.activeElement); e.preventDefault();
        f[(i + (e.shiftKey ? -1 : 1) + f.length) % f.length].focus();
      }
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {      // move through the list
        var items = Array.prototype.slice.call(sheet.querySelectorAll(".ss-sheet-list button"));
        var k = items.indexOf(document.activeElement); if (k < 0) return;
        e.preventDefault(); items[Math.max(0, Math.min(items.length - 1, k + (e.key === "ArrowDown" ? 1 : -1)))].focus();
      }
    });
  }
  // Figure first on narrow screens: the long introduction above the stage (the page-header's .sub, notes, a lede)
  // is cut to a few lines with a "More" button, so the plot is on the first screen. Wide screens are unchanged.
  function clampIntro() {
    var lay = document.querySelector(".ss-layout"); if (!lay) return;
    // the layout's container reorders on narrow screens (stage.css .ss-host): title, then the picker and the
    // figure, then the introduction, the headline chips and the rest
    lay.parentElement.classList.add("ss-host");
    var sel2 = ".page-header > .sub, .page-header > .chipnote, main > header > .sub, main > .lede, main > p.lede, .wrap > .page-header .lede";
    Array.prototype.forEach.call(document.querySelectorAll(sel2), function (el, i) {
      if (!(el.compareDocumentPosition(lay) & Node.DOCUMENT_POSITION_FOLLOWING) || el.classList.contains("ss-clamp")) return;
      el.classList.add("ss-clamp"); if (!el.id) el.id = "ss-intro-" + i;
      var b = document.createElement("button"); b.type = "button"; b.className = "ss-more";
      b.setAttribute("aria-expanded", "false"); b.setAttribute("aria-controls", el.id); b.textContent = "More";
      b.onclick = function () { var o = el.classList.toggle("ss-open"); b.setAttribute("aria-expanded", o ? "true" : "false"); b.textContent = o ? "Less" : "More"; };
      el.parentNode.insertBefore(b, el.nextSibling);
      // an introduction that fits inside the clamp has nothing to open (checked again when its text is filled in)
      var fit = function () { if (!el.classList.contains("ss-open")) b.style.display = el.scrollHeight > el.clientHeight + 1 ? "" : "none"; };
      fit(); addEventListener("resize", fit); if (document.fonts) document.fonts.ready.then(fit);
      new MutationObserver(fit).observe(el, { childList: true, characterData: true, subtree: true });
    });
  }
  // the stage's credits (sources and the figure's date) close its card; on narrow screens, where the introduction and the
  // headline cards follow the figure (stage.css .ss-host), they would sit above those numbers, so there they move to the
  // end of the page, and back under the stage on wide screens
  var credits = null, creditsHome = null, NARROW = matchMedia("(max-width: 1000px)");
  function placeCredits() {
    var lay = document.querySelector(".ss-layout");
    if (!credits) {
      credits = lay && lay.querySelector(".ss-main > .chart-card-footer");
      if (!credits) return;
      creditsHome = { parent: credits.parentNode, next: credits.nextSibling };
    }
    var out = credits.classList.contains("ss-credits-out");
    if (NARROW.matches && !out) { credits.classList.add("ss-credits-out"); lay.parentElement.appendChild(credits); }
    else if (!NARROW.matches && out) { credits.classList.remove("ss-credits-out"); creditsHome.parent.insertBefore(credits, creditsHome.next); }
  }
  function openSheet() {
    if (!sheet) buildSheet();
    var list = sheet.querySelector(".ss-sheet-list"), html = "";
    GROUPS.forEach(function (g) {
      var its = g.items.filter(function (p) { return available(p[0]); }); if (!its.length) return;
      html += '<h3>' + (g.ico ? '<span aria-hidden="true">' + g.ico + '</span> ' : "") + g.label + '</h3>';
      its.forEach(function (p) {
        html += '<button type="button" data-p="' + p[0] + '"' + (p[0] === sel.p ? ' aria-current="true"' : "") + '>' +
          '<span>' + p[1] + '</span>' + (p[2] ? '<small>' + p[2] + '</small>' : "") + '</button>';
      });
    });
    list.innerHTML = html;
    sheetReturn = pick.querySelector(".ss-pick-btn");   // focus goes back to the picker (iOS does not focus a tapped button)
    sheet.hidden = false; document.documentElement.classList.add("ss-lock");
    pick.querySelector(".ss-pick-btn").setAttribute("aria-expanded", "true");
    var cur = list.querySelector('[aria-current="true"]') || list.querySelector("button");
    if (cur) { cur.scrollIntoView({ block: "center" }); cur.focus({ preventScroll: true }); }
  }
  function closeSheet() {
    if (!sheet || sheet.hidden) return;
    sheet.hidden = true; document.documentElement.classList.remove("ss-lock");
    pick.querySelector(".ss-pick-btn").setAttribute("aria-expanded", "false");
    if (sheetReturn && sheetReturn.focus) sheetReturn.focus({ preventScroll: true });
  }
  // A clear horizontal swipe on the figure steps the plot. Vertical scrolling and pinch-zoom are left alone:
  // one finger only, mostly sideways (|dx| > 2|dy|), far enough (60 px) and quick (< 700 ms), and never on
  // something that scrolls sideways or pans by itself (option rows, wide tables, Plotly charts, sliders).
  // Animator sliders: a phone grabs a range thumb from a touch that lands just beside the thin input, so the target
  // is the row, not the input. A touch within a finger's reach of any slider is never a swipe, and a gesture that
  // moved a slider is dropped.
  function nearSlider(root, x, y) {
    var rs = root.querySelectorAll('input[type="range"]');
    for (var i = 0; i < rs.length; i++) {
      var r = rs[i].getBoundingClientRect();
      if (r.width && x > r.left - 24 && x < r.right + 24 && y > r.top - 32 && y < r.bottom + 32) return true;
    }
    return false;
  }
  function swipeable(t, root, x, y) {
    if (nearSlider(root, x, y)) return false;
    for (var el = t; el && el !== root; el = el.parentElement) {
      if (el.matches && el.matches("input, select, textarea, .js-plotly-plot, .controls, [data-noswipe]")) return false;
      if (el.scrollWidth > el.clientWidth + 2) { var ox = getComputedStyle(el).overflowX; if (ox === "auto" || ox === "scroll") return false; }
    }
    return true;
  }
  // Never while the page is pinch-zoomed: a one-finger sideways drag is then a pan around the enlarged figure, not a
  // request for the next plot. Also dropped when the zoom changed during the gesture (a pinch whose fingers landed
  // one after the other).
  function zoom() { return window.visualViewport ? window.visualViewport.scale : 1; }
  function wireSwipe() {
    var st = $("stage"), t0 = null;
    st.addEventListener("touchstart", function (e) {
      var p = e.touches[0];
      t0 = (e.touches.length === 1 && zoom() < 1.02 && swipeable(e.target, st, p.clientX, p.clientY)) ? { x: p.clientX, y: p.clientY, t: Date.now(), z: zoom() } : null;
    }, { passive: true });
    st.addEventListener("input", function (e) { if (e.target && e.target.type === "range") t0 = null; }, true);
    st.addEventListener("touchmove", function (e) { if (e.touches.length > 1) t0 = null; }, { passive: true });
    st.addEventListener("touchend", function (e) {
      if (!t0 || !e.changedTouches.length) return;
      if (e.touches.length || Math.abs(zoom() - t0.z) > 0.01 || zoom() >= 1.02) { t0 = null; return; }
      var dx = e.changedTouches[0].clientX - t0.x, dy = e.changedTouches[0].clientY - t0.y, dt = Date.now() - t0.t; t0 = null;
      if (Math.abs(dx) > 60 && Math.abs(dx) > 2 * Math.abs(dy) && dt < 700) stepProduct(dx < 0 ? 1 : -1);
    }, { passive: true });
  }
  // keep the selected option in view in a row that scrolls sideways, and show which edges have more
  function settleRow(host) {
    var on = host.querySelector(".pill.on");
    if (on && host.scrollWidth > host.clientWidth) host.scrollLeft = Math.max(0, on.offsetLeft - (host.clientWidth - on.offsetWidth) / 2);
    edges(host);
  }
  function edges(host) {
    var more = host.scrollWidth > host.clientWidth + 2;
    host.classList.toggle("more-l", more && host.scrollLeft > 2);
    host.classList.toggle("more-r", more && host.scrollLeft + host.clientWidth < host.scrollWidth - 2);
  }
  var MANY = 6;   // a longer option row is one menu on narrow screens (seventeen pills filled a phone's screen)
  function buttons(host, group, key) {
    host.innerHTML = ""; host.removeAttribute("role"); host.removeAttribute("aria-labelledby");
    if (!group) { sel[key] = null; return null; }
    host.setAttribute("role", "group");                  // a row of pills, one pressed: its label names the group
    if (group.label) { var l = document.createElement("span"); l.className = "seg-label"; l.textContent = group.label; l.id = host.id + "-l"; host.appendChild(l); host.setAttribute("aria-labelledby", l.id); }
    var valid = group.items.some(function (it) { return it[0] === sel[key]; });
    if (NARROW.matches && group.items.length > MANY) {
      var m = document.createElement("select"); m.className = "opt-menu";
      if (group.label) m.setAttribute("aria-labelledby", host.id + "-l");
      group.items.forEach(function (it) { var o = document.createElement("option"); o.value = it[0]; o.textContent = it[1]; m.appendChild(o); });
      m.value = valid ? sel[key] : group.items[0][0]; sel[key] = m.value;
      m.onchange = function () { sel[key] = m.value; render(); };
      host.appendChild(m);
      return sel[key];
    }
    group.items.forEach(function (it, i) {
      var b = document.createElement("button"); b.className = "pill"; b.type = "button"; b.textContent = it[1]; b.dataset.v = it[0];
      var on = (valid && sel[key] === it[0]) || (!valid && i === 0);
      if (on) { b.classList.add("on"); sel[key] = it[0]; }
      b.setAttribute("aria-pressed", on ? "true" : "false");
      b.onclick = function () { sel[key] = it[0]; render(); };
      host.appendChild(b);
    });
    if (!host._edges) { host._edges = true; host.addEventListener("scroll", function () { edges(host); }, { passive: true }); }
    setTimeout(function () { settleRow(host); }, 0);
    return sel[key];
  }
  // SIZING. The width leads: a figure is drawn at the full stage width unless that would make it
  // taller than tallCap(), and only then is it scaled down. Capping every figure at the
  // first-screen height and letting object-fit letterbox the rest would turn a portrait panel
  // (e.g. the 1312x1025 SST loop) into a 307 px thumbnail in a 975 px stage.
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
  // then whole on the screen after one scroll — much larger than a first-screen cap would allow,
  // without ever being taller than the window.
  function tallCap() { return Math.round(Math.max(stageCap(), window.innerHeight * 0.98)); }
  function fitStage() {
    // the stage's own still or embed only: a mounted block's pictures size themselves (stage.css)
    var st = $("stage"), img = st.querySelector(":scope > .ss-zoom > img"), fr = st.querySelector(":scope > iframe");
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
    if (mounted) {
      mountedHome.parent.insertBefore(mounted, mountedHome.next && mountedHome.parent.contains(mountedHome.next) ? mountedHome.next : null);
      // hide again what the mount showed: circulation.html's storm picker and shear dashboard live outside the
      // hidden .dom-holder, so once visited they would stay on the page under every other product
      if (mountedShown) mounted.hidden = true;
      mounted = null; mountedShown = false;
    }
  }
  var PAGE_NAME = ((document.querySelector(".page-header h1") || {}).textContent || "").trim() || document.title;
  var titled = !!location.hash.replace(/^#/, "");
  // gated products: a page whose cards hide themselves when their data is missing
  // (seasonal.html) marks them gated; a hidden one drops out of the rail and is never put on the stage
  function available(id) { var q = P[id]; if (!q || !q.gated || !q.dom) return !!q; var el = $(q.dom()); return !!el && !el.hidden; }
  function firstAvailable() { for (var i = 0; i < ORDER.length; i++) if (available(ORDER[i])) return ORDER[i]; return ORDER[0]; }
  // The control that had the focus is rebuilt by a render (the option pills, the figure); the focus goes to its
  // successor (the pressed pill of the same row, the selected plot in the list, the new figure), so a keyboard user
  // keeps their place and the next arrow key still reaches the viewer.
  function focusSpot() {
    var a = document.activeElement; if (!a || !a.closest) return null;
    if (a.closest("#rail button[data-p], #rail .rail-title")) return "rail";
    var row = a.closest("#opts-a, #opts-b, #opts-c"); if (row) return row.id;
    if (a.closest("#stage > .ss-zoom")) return "zoom";
    return null;
  }
  function refocus(spot) {
    var el = spot === "rail" ? document.querySelector('#rail button[data-p="' + sel.p + '"]')
      : spot === "zoom" ? document.querySelector("#stage > .ss-zoom")
      : spot ? (document.querySelector("#" + spot + " .pill.on") || document.querySelector("#" + spot + " .pill, #" + spot + " .opt-menu")) : null;
    if (el && el.offsetParent !== null) el.focus({ preventScroll: true });
  }
  function render() {
    var spot = focusSpot();
    if (!available(sel.p)) sel = { p: firstAvailable(), a: null, b: null, c: null };
    var p = P[sel.p];
    Array.prototype.forEach.call(document.querySelectorAll("#rail button[data-p]"), function (b) {
      var on = b.dataset.p === sel.p; b.classList.toggle("on", on);
      if (on) b.setAttribute("aria-current", "true"); else b.removeAttribute("aria-current");
    });
    Array.prototype.forEach.call(document.querySelectorAll("#rail .rail-group"), function (d) {
      var mine = !!d.querySelector('button[data-p="' + sel.p + '"]'); d.classList.toggle("has", mine); if (mine) d.classList.add("open"); else d.classList.remove("open");
      var t = d.querySelector(".rail-title"); if (t) t.setAttribute("aria-expanded", mine ? "true" : "false");
    });
    var ga = typeof p.a === "function" ? p.a() : (p.a || null); buttons($("opts-a"), ga, "a");
    var gb = typeof p.b === "function" ? p.b(sel.a) : (p.b || null); buttons($("opts-b"), gb, "b");
    var gc = typeof p.c === "function" ? p.c(sel.a, sel.b) : (p.c || null); buttons($("opts-c"), gc, "c");
    var st = $("stage"); unmount(); st.innerHTML = ""; pinMax = 0;
    // A page's stage footer stamp (#stage-stamp) belongs to the figure on stage: blank until that figure says when it
    // was made (a loop embed posts its manifest's run or build time), never a date that belongs to another product.
    var stamp = $("stage-stamp"); if (stamp) stamp.textContent = "";
    var domId = p.dom && p.dom(sel.a, sel.b, sel.c);   // dom() may return null for an option that is a still
    if (domId) {
      var el = $(domId);
      // a gated block's own script decides whether it has anything to show (its hidden attribute); others are shown
      if (el) { mountedHome = { parent: el.parentNode, next: el.nextSibling }; mounted = el; st.appendChild(el); mountedShown = !p.gated; if (!p.gated) el.hidden = false;
        if (window.Plotly) Array.prototype.forEach.call(el.querySelectorAll(".js-plotly-plot"), function (g) { try { window.Plotly.Plots.resize(g); } catch (e) {} }); }
    } else if (p.frame && p.frame(sel.a, sel.b, sel.c)) {          // frame() may return null for an option that is a still
      var f = document.createElement("iframe"); f.title = p.label; f.loading = "lazy";
      var fsrc = p.frame(sel.a, sel.b, sel.c); f.src = fsrc + (fsrc.indexOf("?") < 0 ? "?" : "&") + "maxh=" + (stageWidth() < 700 ? 4000 : tallCap());  // the embed caps its picture to the same budget
      f.style.aspectRatio = (p.ratio ? p.ratio(sel.a, sel.b, sel.c) : "1259/700"); st.appendChild(f); fitStage();
    } else {
      var src = p.img(sel.a, sel.b, sel.c), key = p.bust === "hourly" ? hourKey() : (S.version ? S.version() : null);
      if (key) src += (src.indexOf("?") < 0 ? "?" : "&") + "v=" + encodeURIComponent(key);
      var im = document.createElement("img"); im.alt = p.label;
      // the still is a button that enlarges it (keyboard and screen readers reach it like any control)
      var zb = document.createElement("button"); zb.type = "button"; zb.className = "ss-zoom"; zb.setAttribute("aria-label", "Enlarge: " + p.label);
      zb.style.cssText = "display:block;width:100%;padding:0;margin:0;border:0;background:none;font:inherit;color:inherit;cursor:zoom-in";
      // a figure whose first build has not been published yet (new product) 404s: say so instead of a broken icon
      im.onerror = function () {
        var m = document.createElement("p"); m.className = "stage-missing";
        m.textContent = "This chart has not been published yet - it appears after the next update of this product (usually within a few hours).";
        m.style.cssText = "padding:2.5rem 1rem;text-align:center;color:#6f6b64;font-size:1.05rem;border:1px dashed #cfcac2;border-radius:8px;margin:0.5rem 0";
        if (zb.parentNode) zb.parentNode.replaceChild(m, zb);
      };
      im.src = src; zb.appendChild(im); st.appendChild(zb);
      im.onload = fitStage; zb.onclick = function () { lbOpen(im.src, im.alt, zb); };
    }
    $("cap").innerHTML = p.cap ? p.cap(sel.a, sel.b, sel.c) : "";
    // a page's own extras that follow the selection (the GEPS/GEFS number tables)
    if (S.onRender) { try { S.onRender({ p: sel.p, a: sel.a, b: sel.b, c: sel.c }); } catch (e) {} }
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
    // Name the tab after the figure so bookmarks, history and shared links say what they are.
    // Not on a hash-less first load: that is how a search crawler sees the page, and it must keep the page's
    // own <title> rather than the first product's.
    if (titled) document.title = g[1] + " · " + PAGE_NAME + " · Shawn Corvec";
    titled = true;
    updatePicker(g);
    if (pushNext && live) live.textContent = g[1];
    fitStage();
    var h = hashOf(sel);
    if (wanted && Date.now() > wanted.until) wanted = null;
    // ownHash: a mounted block that keeps its own state in the hash after the product id and its options
    // (mjo.html's #mi/f/d/m/l/v; circulation.html's #tc/shear/<storm>/...): left alone only while it is that view's
    var own = owns(p, sel) && location.hash.indexOf(h + "/") === 0;
    if (location.hash !== h && !wanted && !own) {                                   // a link still waiting keeps its hash
      if (pushNext) history.pushState(null, "", h);
      // A page opened without a hash gets its hash only once it has loaded: the browser scrolls to the URL's fragment
      // when loading finishes, and a product whose id is also an element id (a mounted card) would pull the page
      // down to the figure, past the title and the picker.
      else if (!location.hash && document.readyState !== "complete") deferHash = h;
      else history.replaceState(null, "", h);
    }
    pushNext = false;
    related();
    refocus(spot);
    window.dispatchEvent(new Event("resize"));
  }

  // The enlarged still: the page's #lightbox, made a modal dialog here. It takes the focus (its Close button), closes
  // on Esc, a click anywhere or Close, keeps Tab inside, and gives the focus back to the figure that opened it.
  var lb = $("lightbox"), lbReturn = null;
  function lbIsOpen() { return !!lb && lb.classList.contains("on"); }
  function lbOpen(src, alt, from) {
    if (!lb) return;
    if (!lb._dialog) {
      lb._dialog = true;
      lb.setAttribute("role", "dialog"); lb.setAttribute("aria-modal", "true"); lb.setAttribute("aria-label", "Enlarged figure");
      var x = document.createElement("button"); x.type = "button"; x.className = "lb-close"; x.textContent = "Close";
      x.style.cssText = "position:absolute;top:12px;right:14px;font-family:inherit;font-weight:600;font-size:1rem;line-height:1;color:#fff;background:rgba(0,0,0,0.55);border:1px solid rgba(255,255,255,0.7);border-radius:6px;padding:0.55rem 0.95rem;cursor:pointer";
      lb.appendChild(x);
      lb.addEventListener("click", lbClose);
      lb.addEventListener("keydown", function (e) {
        if (e.key === "Escape") { e.preventDefault(); lbClose(); }
        else if (e.key === "Tab") { e.preventDefault(); x.focus(); }         // the one control in the dialog
      });
    }
    var pic = lb.querySelector("img"); pic.src = src; pic.alt = alt || "";
    lbReturn = from || null;
    lb.classList.add("on");
    lb.querySelector(".lb-close").focus({ preventScroll: true });
  }
  function lbClose() {
    if (!lbIsOpen()) return;
    lb.classList.remove("on");
    if (lbReturn && document.contains(lbReturn)) lbReturn.focus({ preventScroll: true });
    lbReturn = null;
  }

  // "Related plots": a strip under the figure of products elsewhere on the site
  // that share its variables, regions and models, from the finder's index (assets/site/catalog.json via
  // window.SiteFind). The index is fetched only once the strip scrolls into view.
  var relBox = null, relWanted = false, relCat = null;
  function pagePath() { return location.pathname.replace(/index\.html$/, ""); }
  function related() {
    if (!window.SiteFind || S.related === false) return;
    if (!relBox) {
      relBox = document.createElement("div"); relBox.className = "ss-related"; relBox.hidden = true;
      var anchor = $("about"); anchor.parentNode.insertBefore(relBox, anchor.nextSibling);
      if ("IntersectionObserver" in window) {
        // In view, not 300 px early: its six thumbnails are frames-host requests, never fetched in the background.
        // And still in view a moment later: on page open the strip sits on screen for a few hundred ms, until the
        // figure above it (an embed that posts its height) has grown, which must not count as "seen".
        new IntersectionObserver(function (es, obs) {
          if (!es.some(function (e) { return e.isIntersecting; })) return;
          var tries = 0, check = function () {
            if (relWanted) return;
            var fr = document.querySelector("#stage iframe"), im = document.querySelector("#stage > .ss-zoom img");
            // an embed not sized yet, or a figure still downloading: the box above the strip will still grow
            if (((fr && fr.style.aspectRatio !== "auto") || (im && !im.complete)) && ++tries < 20) { setTimeout(check, 800); return; }
            var r = relBox.getBoundingClientRect();
            if (r.top >= window.innerHeight || r.bottom <= 0) return;  // pushed off screen: wait for the next scroll in
            obs.disconnect(); relWanted = true; fillRelated();
          };
          setTimeout(check, 800);
        }, { rootMargin: "0px" }).observe(relBox);
        relBox.hidden = false; relBox.style.minHeight = "1px";
      } else { relWanted = true; }
    }
    fillRelated();
  }
  function fillRelated() {
    if (!relWanted) return;
    if (!relCat) { window.SiteFind.load().then(function (c) { relCat = c; fillRelated(); }, function () {}); return; }
    // the catalogue entry for this figure: its own id, an old id that aliases to it, or failing both the page itself
    var here = pagePath(), me = null, ids = {};
    ids[here + "#" + sel.p] = 1;
    Object.keys(AL).forEach(function (k) { if (AL[k].split("/")[0] === sel.p) ids[here + "#" + k] = 1; });
    relCat.items.forEach(function (it) { if (it.page === here && ids[it.id]) me = it; });
    if (!me) relCat.items.forEach(function (it) { if (!me && it.id === here) me = it; });
    if (!me) { relBox.hidden = true; return; }
    var set = function (a) { var o = {}; (a || []).forEach(function (x) { o[x] = 1; }); return o; };
    var V = set(me.variables), R = set(me.regions), M = set(me.models);
    var scored = relCat.items.filter(function (it) { return it.page !== here && it.page !== "/catalog.html"; }).map(function (it) {
      var sc = 0;
      (it.variables || []).forEach(function (x) { if (V[x]) sc += 3; });
      (it.regions || []).forEach(function (x) { if (R[x]) sc += 1.5; });
      (it.models || []).forEach(function (x) { if (M[x]) sc += 1; });
      if (it.topic === me.topic) sc += 1.5;
      if (it.horizon && me.horizon) sc += it.horizon === me.horizon ? 2 : -1.5;   // a forecast next to a forecast
      if (it.thumb) sc += 0.5;
      return { it: it, sc: sc };
    }).filter(function (x) { return x.sc >= 5; }).sort(function (a, b) { return b.sc - a.sc; });
    var seenPage = {}, pick = [];                     // at most two from any one page, six in all
    scored.forEach(function (x) { if (pick.length < 6 && (seenPage[x.it.page] || 0) < 2) { seenPage[x.it.page] = (seenPage[x.it.page] || 0) + 1; pick.push(x.it); } });
    if (!pick.length) { relBox.hidden = true; return; }
    var esc = window.SiteFind.esc;
    relBox.hidden = false;
    relBox.innerHTML = '<div class="ss-related-h">Related plots elsewhere on the site</div><div class="ss-related-row">' + pick.map(function (it, i) {
      return '<a class="ss-rel" href="' + esc(it.url) + '"><span class="ss-rel-th" data-i="' + i + '"></span><b>' + esc(it.label) + '</b><small>' + esc(it.page_title) + '</small></a>';
    }).join("") + "</div>";
    pick.forEach(function (it, i) {
      var box = relBox.querySelector('.ss-rel-th[data-i="' + i + '"]');
      if (!it.thumb) { box.classList.add("none"); return; }
      var im = new Image(); im.alt = "";
      im.onload = function () { box.appendChild(im); };
      window.SiteFind.thumbInto(im, it.thumb, function () { box.classList.add("none"); });
    });
  }
  function lastRow() { var rows = ["c", "b", "a"]; for (var i = 0; i < rows.length; i++) { var bs = document.querySelectorAll("#opts-" + rows[i] + " button"); if (bs.length > 1) return bs; } return null; }
  function step(dir) { var bs = lastRow(); if (!bs) return stepProduct(dir); var i = -1; for (var k = 0; k < bs.length; k++) if (bs[k].classList.contains("on")) i = k; bs[(i + dir + bs.length) % bs.length].click(); }
  function stepProduct(dir) {
    var i = ORDER.indexOf(sel.p);
    for (var k = 1; k <= ORDER.length; k++) { var q = ORDER[(i + dir * k + ORDER.length * k) % ORDER.length]; if (available(q)) { choose(q); return; } }
    render();
  }
  function syncRail() {
    Array.prototype.forEach.call(document.querySelectorAll("#rail button[data-p]"), function (b) { b.hidden = !available(b.dataset.p); });
    Array.prototype.forEach.call(document.querySelectorAll("#rail .rail-group"), function (d) {
      var n = d.querySelectorAll("button[data-p]:not([hidden])").length; d.hidden = !n;
      var c = d.querySelector(".rail-title .n"); if (c) c.textContent = n;
    });
  }
  $("prevBtn").onclick = function () { step(-1); }; $("nextBtn").onclick = function () { step(1); };
  // "Larger": hides the rail so the figure takes the whole content width; desktop only (phones already use it all),
  // remembered in this browser
  (function () {
    var nav = $("prevBtn").parentNode, lay = document.querySelector(".ss-layout"); if (!nav || !lay) return;
    var b = document.createElement("button"); b.type = "button"; b.className = "ss-widen";
    function set(on) {
      lay.classList.toggle("ss-wide", on); b.textContent = on ? "Show the menu" : "Larger";
      b.title = on ? "Bring back the list of plots" : "Hide the list of plots and use the full width"; b.setAttribute("aria-pressed", on ? "true" : "false");
      try { localStorage.setItem("ss-wide", on ? "1" : "0"); } catch (e) {}
      fitStage(); setTimeout(fitStage, 60);
    }
    b.onclick = function () { set(!lay.classList.contains("ss-wide")); };
    nav.insertBefore(b, nav.firstChild);
    var saved = null; try { saved = localStorage.getItem("ss-wide"); } catch (e) {}
    if (saved === "1") set(true);
    else { b.textContent = "Larger"; b.title = "Hide the list of plots and use the full width"; b.setAttribute("aria-pressed", "false"); }
  })();
  // Keys work the viewer only where the reader is working it: ← → step the deepest option row while the focus is on
  // the list of plots, the option pills, the prev/next bar or the figure; ↑ ↓ move through the list while the focus is
  // in it, and otherwise scroll the page as usual. Never with a modifier (Alt+← is Back) or from a form control.
  addEventListener("keydown", function (e) {
    if (lbIsOpen()) { if (e.key === "Escape") { e.preventDefault(); lbClose(); } return; }
    if (e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) return;
    if (sheet && !sheet.hidden) return;
    var t = e.target; if (!t || !t.closest || /INPUT|SELECT|TEXTAREA/.test(t.tagName) || t.isContentEditable) return;
    var inList = !!t.closest("#rail") && !t.closest(".ss-pick"), inBar = !!t.closest(".ss-layout .crumb, .ss-layout .opts, #stage > .ss-zoom, .ss-pick");
    if ((e.key === "ArrowRight" || e.key === "ArrowLeft") && (inList || inBar)) { step(e.key === "ArrowRight" ? 1 : -1); e.preventDefault(); }
    else if ((e.key === "ArrowDown" || e.key === "ArrowUp") && inList) { stepProduct(e.key === "ArrowDown" ? 1 : -1); e.preventDefault(); }
  });
  // A stamp more than STALE_K update intervals old says so. The loops with a footer stamp (the circulation and
  // stratosphere pages) are rebuilt each 00/12Z cycle, 12 h, except those listed by manifest: GEOS FP makes one
  // forecast a day.
  var STALE_K = 3, CADENCE_H = { _: 12, "geos_pv_manifest.json": 24 };
  function stampText(x, f) {
    var t = String(x.text || ""), at = +x.at, mf = "";
    try { mf = new URL(f.src, location.href).searchParams.get("manifest") || ""; } catch (e) {}
    var cad = CADENCE_H[mf] || CADENCE_H._, age = Date.now() - at;
    if (!t || !(at > 0) || age <= STALE_K * cad * 3600e3) return t;
    var h = Math.floor(age / 3600e3);
    return t + " · " + (h >= 48 ? Math.floor(h / 24) + " days" : h + " h") + " old; this product normally updates every " + cad + " h";
  }
  addEventListener("message", function (e) {
    var x = e.data; if (!x) return;
    var f = document.querySelector("#stage iframe");
    if (x.type === "sstAnimStamp") {               // the loop on stage names its own run (sst_anim.html postStamp)
      var ss = $("stage-stamp");
      if (ss && f && f.contentWindow === e.source && e.origin === location.origin) ss.textContent = stampText(x, f);
      return;
    }
    // the loop on stage asks for its frame enlarged (sst_anim.html): the stage's own lightbox, unless the page runs
    // lightbox.js, which answers the same message
    if (x.type === "lb-show") {
      if (!window.LB_show && f && f.contentWindow === e.source && e.origin === location.origin && typeof x.src === "string")
        lbOpen(x.src, P[sel.p].label + (typeof x.caption === "string" && x.caption ? ", " + x.caption : ""), f);
      return;
    }
    if (x.type !== "sstAnimHeight") return;
    if (f && f.contentWindow === e.source) {
      f.style.height = x.h + "px"; f.style.aspectRatio = "auto";
      // pin the iframe to the picture's width only when that is close to the stage width (a portrait
      // loop hugging its figure). A much narrower ask means the embed shrank itself, and honouring it
      // would wrap its region bar and shrink it again — leave the frame full width and let it re-measure.
      var sw = stageWidth();
      // Only ever widen the pin within one render: honouring a smaller ask can wrap the embed's
      // region bar, which shrinks its picture, which posts a smaller width again — a downward
      // ratchet that can leave a 1312x1025 loop 307 px wide. An ask under half the stage is that
      // pathology, not a portrait figure, so it is ignored outright.
      if (x.w > pinMax) pinMax = x.w;
      // A landscape figure always spans the stage. The pin exists for a portrait
      // loop that hugs its own picture; honouring it for a wide map would leave a
      // 3.6:1 flux chart at half the stage width on a smaller screen, with the rest of
      // the row empty. Aspect decides, not width alone.
      var wide = x.h > 0 && x.w / x.h > 1.4;
      f.style.maxWidth = (!wide && sw >= 700 && pinMax > sw * 0.5 && pinMax < sw * 0.98) ? pinMax + "px" : "";
    }
  });
  // The hash is positional, #product/a/b/c. An option row a product does not have is written as an empty slot when a
  // later one follows (#pacjet/axis//ifs) and trailing empty slots are dropped. Dropping middle ones too (the writer
  // before 2026-10-16, and the catalogue's variant links) shifted the next value into the wrong row, so #pacjet/axis/ifs
  // reloaded as AIFS. The reader takes both forms: a slot is consumed only by a row the product has (an empty slot
  // where it has none is skipped), so the earlier links still land on their view.
  function hashOf(s) {
    var parts = [s.p, s.a, s.b, s.c].map(function (x) { return x == null ? "" : String(x); });
    while (parts.length > 1 && parts[parts.length - 1] === "") parts.pop();
    return "#" + parts.join("/");
  }
  function row(p, key, n) {           // the product's option group for a / b / c given the earlier choices, or null
    var g = p[key];
    try { return typeof g === "function" ? (key === "a" ? g() : key === "b" ? g(n.a) : g(n.a, n.b)) : (g || null); }
    catch (e) { return null; }
  }
  function fromHash(h) {
    var p = P[h[0]], n = { p: h[0], a: null, b: null, c: null }, k = 1;
    ["a", "b", "c"].forEach(function (key) {
      if (row(p, key, n)) n[key] = h[k++] || null;
      else if (h[k] === "") k++;
    });
    return n;
  }
  function owns(p, s) { return typeof p.ownHash === "function" ? !!p.ownHash(s.a, s.b, s.c) : !!p.ownHash; }
  // an old id is read as its new place; with its own state (#tcshear/<storm>/...) the hash is rewritten once to the new
  // place followed by that state, so the mounted block finds it there
  function hashParts() {
    var h = location.hash.replace(/^#/, "").split("/");
    if (!(h[0] && !P[h[0]] && AL[h[0]])) return h;
    var n = AL[h[0]].split("/");
    if (h.length > 1) { n = n.concat(h.slice(1)); history.replaceState(null, "", "#" + n.join("/")); }
    return n;
  }
  function follow() {
    var h = hashParts(); if (!(h[0] && P[h[0]])) return;
    var n = fromHash(h);
    if (n.p === sel.p && n.a === sel.a && n.b === sel.b && n.c === sel.c) return;
    // a block that keeps its own state in the hash handles its own changes; rebuilding the stage on each of them
    // would remount it twice per click and close an open "About"
    if (n.p === sel.p && P[n.p] && owns(P[n.p], sel) && n.a === sel.a) return;
    sel = n; closeSheet(); render();
  }
  addEventListener("hashchange", follow);
  addEventListener("popstate", follow);
  // a page whose products depend on data it loads after the first paint (the SFS page's data check) redraws with this
  S.refresh = function () { closeSheet(); render(); };
  buildRail();
  wireSwipe();
  clampIntro();
  placeCredits(); NARROW.addEventListener("change", function () { placeCredits(); render(); });   // pills <-> menus
  // watch the gated blocks: a card that shows or hides itself updates the rail, and leaves the stage if it hid
  var gatedEls = ORDER.filter(function (id) { return P[id].gated && P[id].dom; }).map(function (id) { return $(P[id].dom()); }).filter(Boolean);
  if (gatedEls.length && window.MutationObserver) {
    var mo = new MutationObserver(function () {
      syncRail();
      // a deep link to a gated card that was still loading its data when the page opened: show it once it appears
      if (wanted && Date.now() <= wanted.until && available(wanted.p)) { sel = { p: wanted.p, a: wanted.a, b: wanted.b, c: wanted.c }; wanted = null; render(); return; }
      if (!available(sel.p)) render();
    });
    gatedEls.forEach(function (el) { mo.observe(el, { attributes: true, attributeFilter: ["hidden"] }); });
  }
  syncRail();
  var h = hashParts();
  if (h[0] && P[h[0]]) {
    sel = fromHash(h);
    if (!available(h[0])) wanted = { p: h[0], a: sel.a, b: sel.b, c: sel.c, until: Date.now() + 12000 };
  }
  render();
})();
