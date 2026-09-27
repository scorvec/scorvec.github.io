// Site header behaviour: the mobile menu button and the topic dropdowns, then the finder (below).
// Everything in the header's menus is plain HTML links; this only opens and closes them.
(function () {
  var header = document.querySelector('.sh');
  if (!header) return;

  var toggle = header.querySelector('.sh-toggle');
  var nav = header.querySelector('.sh-nav');
  // Collapse to the compact menu whenever the full one does not fit on one line (2026-09-25). Measured, not a
  // breakpoint: the header scales with each page's own root font size. Phones (<= 860 px) use the CSS rule.
  // First step down (2026-09-27): drop the tagline and shrink the finder to its icon; only then collapse the menu.
  function overflows() {
    var items = header.querySelectorAll('.sh-brand, .sh-find, .sh-list > li');
    var right = 0;
    for (var i = 0; i < items.length; i++) right = Math.max(right, items[i].getBoundingClientRect().right);
    return right > document.documentElement.clientWidth - 4;
  }
  function fitHeader() {
    header.classList.remove('sh--compact', 'sh--tight');
    if (window.innerWidth <= 860 || !nav || !overflows()) return;
    header.classList.add('sh--tight');
    if (!overflows()) return;
    header.classList.remove('sh--tight');
    header.classList.add('sh--compact');
  }
  var fitT; window.addEventListener('resize', function () { clearTimeout(fitT); fitT = setTimeout(fitHeader, 80); });
  fitHeader();
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(fitHeader);
  window.addEventListener('load', fitHeader);
  var wide = function () { return window.matchMedia('(min-width: 861px)').matches && !header.classList.contains('sh--compact'); };
  if (toggle && nav) {
    toggle.addEventListener('click', function () {
      var open = header.classList.toggle('is-open');
      toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
      toggle.lastChild.nodeValue = open ? 'Close' : 'Menu';
    });
  }

  header.querySelectorAll('.sh-has-menu').forEach(function (item) {
    var btn = item.querySelector('.sh-menubtn');
    if (!btn) return;
    function setOpen(open) {
      item.classList.toggle('is-open', open);
      btn.setAttribute('aria-expanded', open ? 'true' : 'false');
    }
    btn.addEventListener('click', function () { setOpen(!item.classList.contains('is-open')); });
    // desktop: open on hover, close when the pointer leaves the item
    item.addEventListener('mouseenter', function () { if (wide()) setOpen(true); });
    item.addEventListener('mouseleave', function () { if (wide()) setOpen(false); });
    // keyboard: close when focus leaves the item
    item.addEventListener('focusout', function (e) { if (!item.contains(e.relatedTarget)) setOpen(false); });
  });

  document.addEventListener('keydown', function (e) {
    if (e.key !== 'Escape') return;
    header.querySelectorAll('.sh-has-menu.is-open').forEach(function (item) {
      item.classList.remove('is-open');
      var b = item.querySelector('.sh-menubtn'); if (b) { b.setAttribute('aria-expanded', 'false'); b.focus(); }
    });
    if (header.classList.contains('is-open') && toggle) toggle.click();
  });
  document.addEventListener('click', function (e) {
    if (header.contains(e.target)) return;
    header.querySelectorAll('.sh-has-menu.is-open').forEach(function (item) {
      item.classList.remove('is-open');
      var b = item.querySelector('.sh-menubtn'); if (b) b.setAttribute('aria-expanded', 'false');
    });
  });
})();

// The finder (2026-09-27, the topic redesign): Ctrl/Cmd-K, "/" or the header's "Find a plot" opens a search
// palette over every plot on the site. The index is assets/site/catalog.json (scripts/site/build_catalog.py, rebuilt
// by catalog.yml whenever a page changes), fetched on first use. ARIA combobox + listbox; a preview pane on wide
// screens, a full-screen sheet on phones. window.SiteFind exposes the loader and the scorer to catalog.html so the
// catalogue page filters with exactly the same matching.
(function () {
  var CAT_URL = '/assets/site/catalog.json';
  var RAW = 'https://raw.githubusercontent.com/scorvec/scorvec.github.io/frames/';
  var MIRROR = 'https://cdn.jsdelivr.net/gh/scorvec/scorvec.github.io@frames/';
  var MAC = /Mac|iPhone|iPad|iPod/.test(navigator.platform || navigator.userAgent || '');
  var catP = null, cat = null;

  function norm(s) {
    return String(s || '').normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase().replace(/[\u2010-\u2015]/g, '-');
  }
  // a few words people type that the pages spell out
  var SYN = {
    t2m: ['temperature'], temp: ['temperature'], precip: ['precipitation', 'rain'], rain: ['rain', 'precipitation'],
    ssw: ['sudden warming', 'ssw'], pv: ['potential vorticity', 'pv'], wwb: ['westerly wind burst', 'wwb'],
    ar: ['atmospheric river'], ivt: ['vapour transport', 'ivt'], vortex: ['vortex'], nino: ['nino', 'enso'],
    z500: ['z500', '500 hpa', 'height'], heights: ['height'], sst: ['sst', 'sea surface'], bdc: ['brewer', 'bdc'],
    tercile: ['tercile'], terciles: ['tercile'], cmip: ['cmip6'], jet: ['jet'], jets: ['jet'], snowband: ['snow band', 'snow-band'],
  };

  function load() {
    if (!catP) {
      catP = fetch(CAT_URL, { cache: 'default' }).then(function (r) {
        if (!r.ok) throw new Error('catalog ' + r.status);
        return r.json();
      }).then(function (c) {
        var tl = {};
        (c.topics || []).forEach(function (t) { tl[t.id] = t.label; });
        c.items.forEach(function (it) {
          it._L = norm(it.label);
          it._G = norm([it.group, it.page_title, it.sub].join(' '));
          it._T = norm([].concat(it.models || [], it.regions || [], it.variables || [], it.horizon || '', tl[it.topic] || '').join(' '));
          it._C = norm(it.cap);
          it._K = norm(it.kw);
          it._V = (it.variants || []).map(function (v) { return norm(v[0]); });
          it._topic = tl[it.topic] || '';
        });
        // the menu pages that are not themselves a single catalogue item (the stage, rail and GEPS/GEFS pages): the
        // palette can offer the page as well as its plots ("snow band" -> the snow-band page first)
        var have = {};
        c.items.forEach(function (it) { have[it.id] = 1; });
        c._pages = (c.pages || []).filter(function (p) { return !have[p.href]; }).map(function (p) {
          var on = c.items.filter(function (x) { return x.page === p.href; });
          var th = on.filter(function (x) { return x.thumb; })[0];
          return { id: p.href, page: p.href, url: p.href, label: p.label, page_title: p.label, group: '', sub: '', cap: p.what,
                   kw: '', thumb: th ? th.thumb : null, variants: [], live: false, models: [], regions: [], variables: [],
                   horizon: '', topic: p.topic, _isPage: true, _meta: p.when + (on.length > 1 ? ' \u00b7 ' + on.length + ' plots' : ''),
                   _L: norm(p.label), _G: norm(p.what), _T: norm(tl[p.topic] || ''), _C: '', _K: '', _V: [], _topic: tl[p.topic] || '' };
        });
        cat = c;
        return c;
      });
      catP.catch(function () { catP = null; });
    }
    return catP;
  }

  function wordScore(f, t) {
    var i = f.indexOf(t);
    if (i < 0) return 0;
    var best = 0;
    while (i >= 0) {
      var start = i === 0 || /[^a-z0-9]/.test(f.charAt(i - 1));
      var end = i + t.length === f.length || /[^a-z0-9]/.test(f.charAt(i + t.length));
      var s = start && end ? 3 : start ? 2 : (t.length >= 3 ? 1 : 0);
      if (s > best) best = s;
      if (best === 3) break;
      i = f.indexOf(t, i + 1);
    }
    return best;
  }
  function subseq(f, t) {                     // letters in order, for dropped-letter typos ("strtosphere")
    if (t.length < 4) return false;
    var j = 0;
    for (var i = 0; i < f.length && j < t.length; i++) if (f.charAt(i) === t.charAt(j)) j++;
    return j === t.length;
  }
  function tokens(q) {
    return norm(q).split(/[\s,;:\u00b7/]+/).filter(Boolean).map(function (t) { return SYN[t] ? [t].concat(SYN[t]) : [t]; });
  }
  // score one item against the query's tokens; every token must match somewhere
  function scoreItem(it, toks, qn) {
    var total = 0, labelAll = true;
    for (var k = 0; k < toks.length; k++) {
      var alts = toks[k], best = 0, inLabel = 0;
      for (var a = 0; a < alts.length; a++) {
        var t = alts[a];
        var l = wordScore(it._L, t) * 10;
        if (l > inLabel) inLabel = l;
        var s = Math.max(l, wordScore(it._G, t) * 5, wordScore(it._T, t) * 5, wordScore(it._C, t) * 2, wordScore(it._K, t));
        for (var v = 0; v < it._V.length && s < 18; v++) s = Math.max(s, wordScore(it._V[v], t) * 6);
        if (s > best) best = s;
      }
      if (!best && subseq(it._L, alts[0])) best = 3;
      if (!best) return null;
      if (inLabel < 20) labelAll = false;
      total += best;
    }
    if (it._L.indexOf(qn) === 0) total += 12;
    else if (labelAll) total += 6;
    if (it._isPage) total += 4;
    // the option the query names, when it names one ("sst absolute" -> SST anomaly maps / Absolute SST)
    var variant = null, vbest = 0;
    for (var v2 = 0; v2 < it._V.length; v2++) {
      var vs = 0;
      for (var k2 = 0; k2 < toks.length; k2++) {
        var m = 0;
        for (var a2 = 0; a2 < toks[k2].length; a2++) m = Math.max(m, wordScore(it._V[v2], toks[k2][a2]));
        if (m && wordScore(it._L, toks[k2][0]) < 3) vs += m;
      }
      if (vs > vbest) { vbest = vs; variant = it.variants[v2]; }
    }
    return { item: it, score: total + vbest, variant: variant };
  }
  function search(q, items) {
    items = items || (cat ? cat.items : []);
    var toks = tokens(q), qn = norm(q).trim();
    if (!toks.length) return items.map(function (it) { return { item: it, score: 0, variant: null }; });
    var out = [];
    for (var i = 0; i < items.length; i++) {
      var r = scoreItem(items[i], toks, qn);
      if (r) { r.order = i; out.push(r); }
    }
    out.sort(function (a, b) { return b.score - a.score || a.order - b.order; });
    return out;
  }
  // highlight the query's words in a label, on the original characters
  function highlight(label, q) {
    var chars = Array.from(label), low = chars.map(function (ch) { var n = norm(ch); return n.length === 1 ? n : ch.toLowerCase(); }).join('');
    var mark = new Array(chars.length).fill(false);
    tokens(q).forEach(function (alts) {
      alts.forEach(function (t) {
        if (t.length < 2) return;
        var i = low.indexOf(t);
        while (i >= 0) { for (var j = i; j < i + t.length && j < mark.length; j++) mark[j] = true; i = low.indexOf(t, i + t.length); }
      });
    });
    var out = '', open = false;
    chars.forEach(function (ch, i) {
      if (mark[i] && !open) { out += '<mark>'; open = true; }
      if (!mark[i] && open) { out += '</mark>'; open = false; }
      out += esc(ch);
    });
    return out + (open ? '</mark>' : '');
  }
  function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; });
  }
  function hrefOf(it, variant) {
    return variant ? it.page + variant[1] : it.url;
  }
  function samePage(page) {
    var here = location.pathname.replace(/index\.html$/, '');
    return here === page.replace(/index\.html$/, '');
  }
  // go to a product: same-page jumps just set the hash on pages whose viewer listens for it
  function go(it, variant, newTab) {
    var url = hrefOf(it, variant);
    remember(it, variant);
    if (newTab) { window.open(url, '_blank', 'noopener'); return; }
    var hash = url.indexOf('#') >= 0 ? url.slice(url.indexOf('#')) : '';
    if (samePage(it.page)) {
      if (it.live && hash) {
        close(false);
        if (location.hash === hash) window.dispatchEvent(new HashChangeEvent('hashchange'));
        else location.hash = hash;
        window.scrollTo({ top: 0, behavior: 'smooth' });
        return;
      }
      location.href = url;
      if (hash) location.reload();
      return;
    }
    location.href = url;
  }
  // Figures. An item's `thumb` and each option view's third element are keys into cat.thumbs, one entry per distinct
  // figure: s = the source (a file on main, or with f=1 a loop frame on the frames branch), t = its ~360 px thumbnail
  // on the frames branch (build_thumbs.py) once one exists. Loaders walk thumbnail -> figure, RAW -> jsDelivr.
  function figRec(key) {
    if (!key) return null;
    if (typeof key === 'object') return key.src ? { s: key.src } : key.frame ? { s: key.frame, f: 1 } : null;   // older index
    return cat && cat.thumbs ? cat.thumbs[key] || null : null;
  }
  function figUrls(rec, full) {
    var first = window.frameHostIsMirror && window.frameHostIsMirror() ? MIRROR : RAW, second = first === RAW ? MIRROR : RAW;
    var out = [];
    var t = rec.t ? 'assets/site/thumbs/' + rec.t + '.webp' : null;
    if (t && !full) out.push(first + t, second + t);
    if (rec.f) out.push(first + rec.s, second + rec.s);
    else out.push('/' + rec.s);
    return out;
  }
  // thumbInto(im, key, onFail, full): the thumbnail (or, with full, the figure itself) into <img> im
  function thumbInto(im, key, onFail, full) {
    var rec = figRec(key);
    if (!rec) { if (onFail) onFail(); return; }
    var tries = figUrls(rec, full), k = 0;
    im.onerror = function () { k++; if (k < tries.length) im.src = tries[k]; else if (onFail) onFail(); };
    im.src = tries[0];
  }
  // the figure a result shows: the matched option's own (null when that option draws in the browser), else the default
  // (null: drawn in the browser; '': a figure not published yet)
  function figOf(r) {
    var k = r.variant ? r.variant[2] : r.item.thumb;
    return k === undefined ? null : k;
  }
  function tileNote(it, key) {
    if (key === '') return 'Not published yet \u2014 opens on the page';
    return it.kind === 'page' || it._isPage ? 'Opens the page' : 'Drawn in the browser \u2014 opens on the page';
  }
  var RECENT = 'siteFindRecent';
  function remember(it, variant) {
    try {
      var r = JSON.parse(localStorage.getItem(RECENT) || '[]').filter(function (x) { return x.id !== it.id; });
      r.unshift({ id: it.id, v: variant ? variant[1] : null });
      localStorage.setItem(RECENT, JSON.stringify(r.slice(0, 6)));
    } catch (e) {}
  }
  function recents() {
    try { return JSON.parse(localStorage.getItem(RECENT) || '[]'); } catch (e) { return []; }
  }

  window.SiteFind = { load: load, search: search, highlight: highlight, thumbInto: thumbInto, figOf: figOf, tileNote: tileNote, norm: norm, esc: esc, remember: remember };

  // -- the palette --
  var root = null, input, list, prev, live, rows = [], active = -1, lastFocus = null, q = '', prevT = 0;
  var header = document.querySelector('.sh');
  var findBtn = header && header.querySelector('.sh-find');
  if (findBtn) {
    var kb = findBtn.querySelector('kbd');
    if (kb) kb.textContent = MAC ? '\u2318K' : 'Ctrl K';
    findBtn.setAttribute('title', 'Find a plot (' + (MAC ? '\u2318K' : 'Ctrl K') + ')');
    findBtn.addEventListener('click', function () { open(); });
    findBtn.addEventListener('pointerenter', function () { load(); });
    findBtn.addEventListener('focus', function () { load(); });
  }

  function build() {
    var dark = header && header.classList.contains('sh--dark');
    root = document.createElement('div');
    root.className = 'fp' + (dark ? ' fp--dark' : '');
    root.hidden = true;
    root.innerHTML =
      '<div class="fp-scrim" data-close></div>' +
      '<div class="fp-box" role="dialog" aria-modal="true" aria-label="Find a plot">' +
        '<div class="fp-top">' +
          '<svg class="fp-ico" viewBox="0 0 20 20" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" aria-hidden="true" focusable="false">' +
            '<circle cx="8.5" cy="8.5" r="5.5" style="fill:none"/><path d="M12.6 12.6 17 17" style="fill:none"/></svg>' +
          '<input class="fp-in" type="text" role="combobox" aria-expanded="true" aria-controls="fp-list" aria-autocomplete="list" ' +
            'autocomplete="off" autocapitalize="off" spellcheck="false" enterkeyhint="go" placeholder="Search every plot: model, variable, region\u2026">' +
          '<button class="fp-x" type="button" data-close><span class="fp-x-t">Cancel</span><kbd>Esc</kbd></button>' +
        '</div>' +
        '<div class="fp-body">' +
          '<ul class="fp-list" id="fp-list" role="listbox" aria-label="Plots"></ul>' +
          '<aside class="fp-prev" aria-hidden="true"></aside>' +
        '</div>' +
        '<div class="fp-foot"><span><kbd>\u2191</kbd><kbd>\u2193</kbd> move</span><span><kbd>\u21b5</kbd> open</span>' +
          '<span class="fp-foot-nt"><kbd>' + (MAC ? '\u2318' : 'Ctrl') + '</kbd><kbd>\u21b5</kbd> new tab</span>' +
          '<a class="fp-all" href="/catalog.html">Browse the catalogue \u2192</a></div>' +
        '<div class="fp-sr" aria-live="polite"></div>' +
      '</div>';
    document.body.appendChild(root);
    input = root.querySelector('.fp-in'); list = root.querySelector('.fp-list');
    prev = root.querySelector('.fp-prev'); live = root.querySelector('.fp-sr');
    root.addEventListener('click', function (e) {
      if (e.target.closest('[data-close]')) { close(true); return; }
      var li = e.target.closest('.fp-o');
      if (li) { var r = rows[+li.dataset.i]; if (r) go(r.item, r.variant, e.metaKey || e.ctrlKey); }
      var pv = e.target.closest('.fp-pv a');
      if (pv) {
        var rr = rows[active]; if (!rr) return;
        e.preventDefault();
        go(rr.item, rr.item.variants[+pv.dataset.v], e.metaKey || e.ctrlKey);
      }
    });
    list.addEventListener('mousemove', function (e) {
      var li = e.target.closest('.fp-o');
      if (li && +li.dataset.i !== active) setActive(+li.dataset.i, false);
    });
    input.addEventListener('input', function () { q = input.value; render(); });
    root.addEventListener('keydown', onKey);
  }

  function onKey(e) {
    if (e.key === 'Escape') { e.preventDefault(); close(true); return; }
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault();
      if (!rows.length) return;
      var n = active + (e.key === 'ArrowDown' ? 1 : -1);
      setActive((n + rows.length) % rows.length, true);
      return;
    }
    if ((e.key === 'PageDown' || e.key === 'PageUp') && rows.length) {
      e.preventDefault();
      setActive(Math.max(0, Math.min(rows.length - 1, active + (e.key === 'PageDown' ? 8 : -8))), true);
      return;
    }
    if (e.key === 'Enter' && e.target === input) {
      e.preventDefault();
      var r = rows[active];
      if (r) go(r.item, r.variant, e.metaKey || e.ctrlKey);
      return;
    }
    if (e.key === 'Tab') {                   // keep focus inside the dialog
      var f = Array.prototype.filter.call(root.querySelectorAll('input, button, a[href]'), function (x) { return x.offsetParent !== null; });
      if (!f.length) return;
      var i = f.indexOf(document.activeElement);
      e.preventDefault();
      f[(i + (e.shiftKey ? -1 : 1) + f.length) % f.length].focus();
    }
  }

  function row(r, i) {
    var it = r.item;
    var meta = r.meta || it._meta ? esc(r.meta || it._meta) : it.group ? esc(it.page_title) + ' \u00b7 ' + esc(it.group)
      : esc([it._topic].concat(it.horizon ? [it.horizon] : []).join(' \u00b7 '));
    return '<li class="fp-o" role="option" id="fp-o-' + i + '" data-i="' + i + '" aria-selected="false">' +
      '<span class="fp-o-t">' + highlight(it.label, q) +
      (r.variant ? '<span class="fp-o-v"> \u203a ' + highlight(r.variant[0], q) + '</span>' : '') + '</span>' +
      '<span class="fp-o-m">' + meta + '</span></li>';
  }
  function head(label) { return '<li class="fp-h" role="presentation">' + esc(label) + '</li>'; }

  function render() {
    if (!cat) {
      list.innerHTML = '<li class="fp-msg" role="presentation">Loading the index\u2026</li>';
      rows = []; active = -1; input.removeAttribute('aria-activedescendant');
      return;
    }
    var html = '', i = 0;
    rows = [];
    if (!q.trim()) {
      var byId = {};
      cat.items.forEach(function (it) { byId[it.id] = it; });
      var rec = recents().map(function (x) {
        var it = byId[x.id]; if (!it) return null;
        var v = x.v ? (it.variants || []).filter(function (w) { return w[1] === x.v; })[0] || null : null;
        return { item: it, variant: v };
      }).filter(Boolean);
      if (rec.length) {
        html += head('Recent');
        rec.forEach(function (r) { rows.push(r); html += row(r, i++); });
      }
      // one row per page, in menu order under each topic: the site map in the palette
      (cat.topics || []).forEach(function (t) {
        var ps = (cat.pages || []).filter(function (p) { return p.topic === t.id; });
        if (!ps.length) return;
        html += head(t.label);
        ps.forEach(function (p) {
          var it = byId[p.href] || (cat._pages || []).filter(function (x) { return x.id === p.href; })[0];
          if (!it) return;
          var r = { item: it, variant: null, meta: it._meta || p.when };
          rows.push(r); html += row(r, i++);
        });
      });
    } else {
      var res = search(q, cat.items.concat(cat._pages || [])).slice(0, 60);
      res.forEach(function (r) { rows.push(r); html += row(r, i++); });
      if (!res.length) html = '<li class="fp-msg" role="presentation">Nothing matches \u201c' + esc(q) + '\u201d. Try a model (GEFS), a variable (z500, snow) or a region.</li>';
    }
    list.innerHTML = html;
    live.textContent = q.trim() ? (rows.length ? rows.length + (rows.length === 60 ? '+' : '') + ' results' : 'No results') : '';
    setActive(rows.length ? 0 : -1, false);
  }

  function setActive(i, scroll) {
    var old = list.querySelector('.fp-o[aria-selected="true"]');
    if (old) old.setAttribute('aria-selected', 'false');
    active = i;
    if (i < 0) { input.removeAttribute('aria-activedescendant'); preview(null); return; }
    var li = document.getElementById('fp-o-' + i);
    if (li) {
      li.setAttribute('aria-selected', 'true');
      input.setAttribute('aria-activedescendant', li.id);
      if (scroll) li.scrollIntoView({ block: 'nearest' });
    }
    clearTimeout(prevT);
    prevT = setTimeout(function () { preview(rows[i]); }, 60);
  }

  // the preview pane: the figure itself (the selected option's own), what it is, and its other views
  function tileHTML(it, note) {
    return '<div class="fp-tile t-' + esc(it.topic || '') + '"><em>' + esc(note) + '</em><span>' + esc(it.label) + '</span>' +
      '<small>' + esc(it.group ? it.page_title : (it._meta || it._topic || '')) + '</small></div>';
  }
  function preview(r) {
    if (!prev || prev.offsetParent === null) return;
    if (!r) { prev.innerHTML = ''; return; }
    var it = r.item, key = figOf(r);
    var tags = [it.horizon].concat((it.models || []).slice(0, 4)).filter(Boolean);
    var vs = (it.variants || []).slice(0, 14);
    prev.innerHTML =
      '<div class="fp-th">' + (key ? '' : tileHTML(it, tileNote(it, key))) + '</div>' +
      '<h3 class="fp-pt">' + esc(it.label) + (r.variant ? ' <span>\u203a ' + esc(r.variant[0]) + '</span>' : '') + '</h3>' +
      '<p class="fp-pm">' + (it.group ? esc(it.page_title) + ' \u00b7 ' + esc(it.group) : esc([it._topic].concat(r.meta || it._meta || it.horizon || []).join(' \u00b7 '))) + '</p>' +
      (tags.length ? '<p class="fp-tags">' + tags.map(function (t) { return '<span>' + esc(t) + '</span>'; }).join('') + '</p>' : '') +
      (it.cap || it.sub ? '<p class="fp-cap">' + esc(it.cap || it.sub) + '</p>' : '') +
      (vs.length > 1 ? '<div class="fp-pv"><h4>' + it.variants.length + ' views</h4>' + vs.map(function (v) {
        var idx = it.variants.indexOf(v);
        return '<a href="' + esc(it.page + v[1]) + '" data-v="' + idx + '"' + (r.variant && r.variant[1] === v[1] ? ' aria-current="true"' : '') + '>' + esc(v[0]) + '</a>';
      }).join('') + (it.variants.length > vs.length ? '<span class="fp-more">+ ' + (it.variants.length - vs.length) + ' more on the page</span>' : '') + '</div>' : '');
    if (!key) return;
    var box = prev.querySelector('.fp-th'), im = new Image(), big = new Image();
    im.alt = ''; big.alt = '';
    var current = function () { return rows[active] === r; };
    im.onload = function () {
      if (!current()) return;
      box.innerHTML = ''; box.appendChild(im);
      // then the full figure, sharp at the pane's size, swapped in once it has arrived
      big.onload = function () { if (current() && im.parentNode === box) box.replaceChild(big, im); };
      thumbInto(big, key, null, true);
    };
    thumbInto(im, key, function () { if (current()) box.innerHTML = tileHTML(it, 'Opens on the page'); });
  }

  function open() {
    if (!root) build();
    if (!root.hidden) { input.focus(); input.select(); return; }
    lastFocus = document.activeElement;
    root.hidden = false;
    document.documentElement.classList.add('fp-lock');
    if (findBtn) findBtn.setAttribute('aria-expanded', 'true');
    input.value = q;
    input.focus(); input.select();
    render();
    load().then(function () { if (!root.hidden) render(); }, function () {
      list.innerHTML = '<li class="fp-msg" role="presentation">The index could not be loaded. <a href="/catalog.html">Open the catalogue</a> or try again.</li>';
    });
  }
  function close(restore) {
    if (!root || root.hidden) return;
    root.hidden = true;
    document.documentElement.classList.remove('fp-lock');
    if (findBtn) findBtn.setAttribute('aria-expanded', 'false');
    if (restore && lastFocus && lastFocus.focus) lastFocus.focus();
  }

  document.addEventListener('keydown', function (e) {
    var k = (e.key || '').toLowerCase();
    if (k === 'k' && (e.metaKey || e.ctrlKey) && !e.altKey && !e.shiftKey) {
      e.preventDefault();
      if (root && !root.hidden) close(true); else open();
      return;
    }
    if (e.key === '/' && !e.metaKey && !e.ctrlKey && !e.altKey) {
      var t = e.target, tag = t && t.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || (t && t.isContentEditable)) return;
      if (root && !root.hidden) return;
      e.preventDefault(); open();
    }
  });
  window.SiteFind.open = open;
})();
