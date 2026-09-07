/* rail.js — one figure at a time behind a product rail (the GEPS subseasonal page's layout,
   generalised 2026-09-07 at the user's request: "a model for a number of our pages with
   multiple charts, saves you from scrolling").

   Progressive enhancement. Mark the container `<main data-rail>` (optionally
   data-rail="<item selector>"). Items are, by default, every `.deck-panel` inside a
   `.plot-deck` (labelled by the deck's tab buttons, in order) and every standalone
   `.chart-card` (labelled by data-label or its h2). Groups come from the nearest preceding
   `h3.page-section` heading, or an item's data-group, else "Figures". Existing ids on cards
   keep working as deep links (#wave-activity-flux); deck panels get slugs from their labels.
   Iframes keep their lazy data-src behaviour: an item is loaded the first time it is shown.
   Without JavaScript the page is the long scroll it was before. */
(function () {
  "use strict";
  var main = document.querySelector("[data-rail]");
  if (!main) return;
  var sel = main.getAttribute("data-rail") || ".plot-deck .deck-panel, .chart-card:not(.plot-deck)";
  var nodes = Array.prototype.slice.call(main.querySelectorAll(sel));
  if (nodes.length < 3) return;

  function text(el) { return (el ? el.textContent : "").replace(/\s+/g, " ").trim(); }
  function slug(s) { return s.toLowerCase().replace(/&[a-z]+;/g, "").replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 60); }
  function groupOf(el) {
    if (el.dataset.group) return el.dataset.group;
    var deck = el.closest(".plot-deck");
    if (deck && deck.dataset.group) return deck.dataset.group;
    var n = deck || el;
    while (n && n !== main) {
      var p = n.previousElementSibling;
      while (p) {
        if (p.matches("h3.page-section, h2.page-section, .page-section")) return text(p);
        if (p.dataset && p.dataset.group) return p.dataset.group;
        p = p.previousElementSibling;
      }
      n = n.parentElement;
    }
    return "Figures";
  }
  var items = [], used = {};
  nodes.forEach(function (el) {
    var label = el.dataset.label, deck = el.closest(".plot-deck");
    if (!label && deck) {
      var panels = Array.prototype.slice.call(deck.querySelectorAll(".deck-panel"));
      var tabs = deck.querySelectorAll(".deck-tabs button");
      var i = panels.indexOf(el);
      if (tabs[i]) label = text(tabs[i]);
    }
    if (!label) label = text(el.querySelector("h2, h3")) || "Figure";
    var id = el.id || slug(label); while (used[id]) id += "-2"; used[id] = 1;
    items.push({ el: el, label: label, group: groupOf(el), id: id, desc: el.dataset.desc || "",
                 home: { parent: el.parentNode, next: el.nextSibling } });
  });

  // layout
  var layout = document.createElement("div"); layout.className = "rail-layout";
  layout.innerHTML = '<aside class="rail" aria-label="Figures"></aside>' +
    '<section class="rail-main"><div class="rail-crumb"><div class="path"></div>' +
    '<div class="nav"><button type="button" class="rail-prev" title="previous (←)">&larr; prev</button>' +
    '<button type="button" class="rail-next" title="next (→)">next &rarr;</button></div></div>' +
    '<div class="rail-stage"></div>' +
    '<p class="rail-hint"><span class="kbd">&larr;</span> <span class="kbd">&rarr;</span> step through the figures</p></section>';
  var first = nodes[0].closest(".plot-deck") || nodes[0];
  var anchor = first; while (anchor.parentNode !== main) anchor = anchor.parentNode;
  main.insertBefore(layout, anchor);
  var rail = layout.querySelector(".rail"), stage = layout.querySelector(".rail-stage"), path = layout.querySelector(".path");
  // the old scroll is parked, not destroyed: section headings, deck tab bars and the in-page nav go away
  var park = document.createElement("div"); park.className = "rail-park"; park.hidden = true; main.appendChild(park);
  Array.prototype.forEach.call(main.querySelectorAll("h3.page-section, h2.page-section, nav.section-nav"), function (n) { park.appendChild(n); });
  items.forEach(function (it) {                              // the item's outermost block under <main> (a deck for a deck panel)
    var top = it.el; while (top.parentNode && top.parentNode !== main && top.parentNode !== park) top = top.parentNode;
    if (top !== layout && top.parentNode === main) park.appendChild(top);
  });

  // rail
  var groups = [];
  items.forEach(function (it) { if (groups.indexOf(it.group) < 0) groups.push(it.group); });
  groups.forEach(function (g) {
    var d = document.createElement("div"); d.className = "rail-group";
    var t = document.createElement("div"); t.className = "rail-title"; t.textContent = g; d.appendChild(t);
    items.filter(function (it) { return it.group === g; }).forEach(function (it) {
      var b = document.createElement("button"); b.type = "button"; b.dataset.id = it.id;
      b.innerHTML = "<span>" + it.label + "</span>" + (it.desc ? "<small>" + it.desc + "</small>" : "");
      b.addEventListener("click", function () { show(it.id, true); });
      it.btn = b; d.appendChild(b);
    });
    rail.appendChild(d);
  });

  function reveal(el) {
    Array.prototype.forEach.call(el.querySelectorAll("iframe[data-src], img[data-src], img[data-deferred], iframe[data-deferred]"), function (f) {
      var d = f.getAttribute("data-src"); if (d) { f.setAttribute("src", d); f.removeAttribute("data-src"); }
      f.removeAttribute("data-deferred");
    });
  }
  var cur = null;
  // a card hidden by the page's own script (data not available yet) stays out of the rail until it appears
  items.forEach(function (it) {
    if (it.el.classList.contains("deck-panel")) return;
    var sync = function () { if (it !== cur) it.btn.hidden = it.el.hidden; };
    sync();
    if ("MutationObserver" in window) new MutationObserver(sync).observe(it.el, { attributes: true, attributeFilter: ["hidden"] });
  });
  function show(id, push) {
    var it = null; items.forEach(function (x) { if (x.id === id) it = x; }); if (!it) it = items[0];
    // The previous item goes back to the hidden park (its recorded home was its place in <main>
    // BEFORE parking, so returning it there put it back on screen below the stage — seen on
    // seas5.html 2026-09-07 as "stratosphere charts show up below"). Deck panels return to
    // their deck, which is itself parked.
    if (cur && cur !== it) {
      if (cur.el.classList.contains("deck-panel")) { cur.el.hidden = true; if (cur.home.parent.contains(cur.home.next) || !cur.home.next) cur.home.parent.insertBefore(cur.el, cur.home.next); else cur.home.parent.appendChild(cur.el); }
      else park.appendChild(cur.el);
    }
    stage.appendChild(it.el); if (it.el.classList.contains("deck-panel")) it.el.hidden = false; reveal(it.el); cur = it;
    if (window.Plotly) Array.prototype.forEach.call(it.el.querySelectorAll(".js-plotly-plot"), function (g) { try { window.Plotly.Plots.resize(g); } catch (e) {} });
    items.forEach(function (x) { x.btn.classList.toggle("on", x === it); });
    path.innerHTML = '<span class="g">' + it.group + '</span> <span class="sep">&rsaquo;</span> <span class="i">' + it.label + "</span>";
    if (push && history.replaceState) history.replaceState(null, "", "#" + it.id);
    window.dispatchEvent(new Event("resize"));
  }
  function step(k) { var i = items.indexOf(cur); show(items[(i + k + items.length) % items.length].id, true); }
  layout.querySelector(".rail-prev").addEventListener("click", function () { step(-1); });
  layout.querySelector(".rail-next").addEventListener("click", function () { step(1); });
  document.addEventListener("keydown", function (e) {
    if (e.target && /INPUT|SELECT|TEXTAREA/.test(e.target.tagName)) return;
    if (e.key === "ArrowLeft") { step(-1); e.preventDefault(); } else if (e.key === "ArrowRight") { step(1); e.preventDefault(); }
  });
  window.addEventListener("hashchange", function () { var h = location.hash.slice(1); if (h && items.some(function (x) { return x.id === h; })) show(h, false); });
  var h0 = location.hash.slice(1);
  var firstVisible = items.filter(function (x) { return !x.el.hidden || x.el.classList.contains("deck-panel"); })[0] || items[0];
  show(items.some(function (x) { return x.id === h0; }) ? h0 : firstVisible.id, false);
  document.documentElement.classList.add("has-rail");
})();
