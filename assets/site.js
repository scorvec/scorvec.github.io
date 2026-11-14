// Site header behaviour: the mobile menu button and the topic dropdowns.
// Everything in the header's menus is plain HTML links; this only opens and closes them.
(function () {
  var header = document.querySelector('.sh');
  if (!header) return;

  var toggle = header.querySelector('.sh-toggle');
  var nav = header.querySelector('.sh-nav');
  // Collapse to the compact menu whenever the full one does not fit on one line (2026-09-25). Measured, not a
  // breakpoint: the header scales with each page's own root font size. Phones (<= 860 px) use the CSS rule.
  // First step down (2026-09-27): drop the tagline; then set the menu buttons closer; only then collapse the menu.
  // Against the header's own content box, not just the window (2026-10-12): .sh-in is capped at min(94vw, 2000px) and
  // the page content lines up with it, so a menu that spilled past its padding (the longer "Circulation/tropics" label,
  // 22-49 px at 1440-1920 px) still "fitted" the window while the About menu stuck out past the content edge.
  var inner = header.querySelector('.sh-in');
  var wide = function () { return window.matchMedia('(min-width: 861px)').matches && !header.classList.contains('sh--compact'); };
  function overflows() {
    var items = header.querySelectorAll('.sh-brand, .sh-list > li');
    var right = 0, lim = document.documentElement.clientWidth - 4;
    for (var i = 0; i < items.length; i++) right = Math.max(right, items[i].getBoundingClientRect().right);
    if (inner) lim = Math.min(lim, inner.getBoundingClientRect().right - (parseFloat(getComputedStyle(inner).paddingRight) || 0) + 1);
    return right > lim;
  }
  function fitMenu() {
    header.classList.remove('sh--compact', 'sh--tight', 'sh--tighter');
    if (window.innerWidth <= 860 || !nav || !overflows()) return;
    header.classList.add('sh--tight');
    if (!overflows()) return;
    header.classList.add('sh--tighter');                 // then closer-set menu buttons (down to ~1280 px)
    if (!overflows()) return;
    header.classList.remove('sh--tight', 'sh--tighter');
    header.classList.add('sh--compact');
  }
  // Collapsed (phones and the compact header) every group is listed open and its button is only its title (site.css
  // turns its pointer events off): it leaves the tab order and the accessibility tree and the group carries its name.
  // On a wide header it is the menu's button again.
  function syncHeads() {
    var collapsed = !wide();
    header.querySelectorAll('.sh-has-menu').forEach(function (item) {
      var btn = item.querySelector('.sh-menubtn'), menu = item.querySelector('.sh-menu');
      if (!btn) return;
      if (collapsed) {
        btn.setAttribute('tabindex', '-1'); btn.setAttribute('aria-hidden', 'true'); btn.removeAttribute('aria-expanded');
        if (menu) { menu.setAttribute('role', 'group'); menu.setAttribute('aria-label', btn.textContent.trim()); }
      } else if (btn.hasAttribute('aria-hidden')) {
        btn.removeAttribute('tabindex'); btn.removeAttribute('aria-hidden');
        btn.setAttribute('aria-expanded', item.classList.contains('is-open') ? 'true' : 'false');
        if (menu) { menu.removeAttribute('role'); menu.removeAttribute('aria-label'); }
      }
    });
  }
  function fitHeader() { fitMenu(); syncHeads(); }
  var fitT; window.addEventListener('resize', function () { clearTimeout(fitT); fitT = setTimeout(fitHeader, 80); });
  fitHeader();
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(fitHeader);
  window.addEventListener('load', fitHeader);
  // The homepage's header lies over the hero and would scroll away with it: once the hero has gone by it is the ordinary
  // header pinned to the top of the window (fixed, so nothing below moves), and over the hero the overlay again.
  if (header.classList.contains('sh--overlay')) {
    var hero = document.querySelector('.hero'), pinned = false;
    var pin = function () {
      var on = hero ? hero.getBoundingClientRect().bottom <= header.offsetHeight : window.scrollY > 0;
      if (on === pinned) return;
      pinned = on;
      header.classList.toggle('sh--overlay', !on);
      header.classList.toggle('sh--pinned', on);
      fitHeader();
    };
    window.addEventListener('scroll', pin, { passive: true });
    pin();
  }
  // the stacked menu scrolls inside what is left of the window below the header (2026-10-08 sweep: a fixed
  // calc(100vh - 64px) ran past the bottom of the screen under the taller homepage header and on iOS, whose 100vh
  // is the toolbar-less height, so the last links could not be reached)
  function fitNav() {
    if (!nav) return;
    if (!header.classList.contains('is-open')) { nav.style.maxHeight = ''; return; }
    nav.style.maxHeight = Math.max(160, Math.floor(window.innerHeight - header.getBoundingClientRect().bottom)) + 'px';
  }
  if (toggle && nav) {
    toggle.addEventListener('click', function () {
      var open = header.classList.toggle('is-open');
      toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
      toggle.lastChild.nodeValue = open ? 'Close' : 'Menu';
      fitNav();
    });
    window.addEventListener('resize', fitNav);
  }

  // a dropdown that would cross the window's edge slides back inside it (2026-10-08 sweep: the two-column
  // "Studies and verification" panel opened rightwards from its button and ran ~20 px past the right edge on
  // 1366 and 1920 px screens; the CSS anchors cannot know how far the button sits from the edge)
  function keepOnScreen(menu) {
    if (!menu) return;
    menu.style.transform = '';
    if (!wide()) return;
    var r = menu.getBoundingClientRect(), vw = document.documentElement.clientWidth, dx = 0;
    if (r.right > vw - 8) dx = vw - 8 - r.right;
    if (r.left + dx < 8) dx = 8 - r.left;
    if (dx) menu.style.transform = 'translateX(' + Math.round(dx) + 'px)';
  }
  var groups = header.querySelectorAll('.sh-has-menu');
  groups.forEach(function (item) {
    var btn = item.querySelector('.sh-menubtn');
    if (!btn) return;
    var byHover = false;
    function setOpen(open) {
      if (open) groups.forEach(function (o) {           // one menu at a time (taps on a wide tablet left two open)
        if (o !== item && o.classList.contains('is-open')) {
          o.classList.remove('is-open');
          var b = o.querySelector('.sh-menubtn'); if (b) b.setAttribute('aria-expanded', 'false');
        }
      });
      item.classList.toggle('is-open', open);
      if (wide()) btn.setAttribute('aria-expanded', open ? 'true' : 'false');
      if (!open) byHover = false;
      if (open) keepOnScreen(item.querySelector('.sh-menu'));
    }
    // a click on a menu the pointer has just opened by hovering keeps it open (it used to toggle it shut again: hover
    // opened it, the click closed it, 2026-10-08 sweep); otherwise a click toggles
    btn.addEventListener('click', function () {
      if (byHover && item.classList.contains('is-open')) { byHover = false; return; }
      setOpen(!item.classList.contains('is-open'));
    });
    // desktop: open on hover, close when the pointer leaves the item. A real mouse only: a tap on a touch screen
    // fires emulated mouse events first, which opened the menu and then let the tap's click close it.
    item.addEventListener('pointerenter', function (e) {
      if (e.pointerType !== 'mouse' || !wide() || item.classList.contains('is-open')) return;
      setOpen(true); byHover = true;
    });
    item.addEventListener('pointerleave', function (e) { if (e.pointerType === 'mouse' && wide()) setOpen(false); });
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
  // Close every menu once one is used, so a drop-down never stays over the page on a phone. A link into the page
  // already open (circulation.html#tcshear from circulation.html) only changes the hash, and Back can restore a page
  // from the back/forward cache with its menu still open.
  function closeAll() {
    header.querySelectorAll('.sh-has-menu.is-open').forEach(function (item) {
      item.classList.remove('is-open');
      var b = item.querySelector('.sh-menubtn'); if (b) b.setAttribute('aria-expanded', 'false');
    });
    if (header.classList.contains('is-open')) {
      header.classList.remove('is-open');
      if (toggle) { toggle.setAttribute('aria-expanded', 'false'); toggle.lastChild.nodeValue = 'Menu'; }
    }
  }
  header.addEventListener('click', function (e) { if (e.target.closest && e.target.closest('a[href]')) closeAll(); });
  window.addEventListener('hashchange', closeAll);
  window.addEventListener('pageshow', function (e) { if (e.persisted) closeAll(); });
})();

// The skip link moves focus to <main> without touching the URL: on the stage pages the hash names the product on show,
// and following "#main" would drop it from the address a reader bookmarks or shares.
document.addEventListener('click', function (e) {
  var link = e.target.closest && e.target.closest('a.skip-link');
  var main = link && document.getElementById('main');
  if (!main) return;
  e.preventDefault();
  if (!main.hasAttribute('tabindex')) main.setAttribute('tabindex', '-1');
  main.focus();
});

// Headline-number bands (.chips): as many equal cells a row as fit at 8.5rem or more (or the band's --chip-min, where
// a stylesheet sets one), the rows balanced, so nine numbers are one row of nine on a wide screen and five and four
// on a narrower one, never eight and a lone one; the last cell spans what its row leaves empty. Pages fill their
// bands after loading, so the count is re-read when the cells change.
(function () {
  var bands = document.querySelectorAll('.chips');
  if (!bands.length || !window.ResizeObserver) return;
  function balance(band) {
    var n = band.children.length, w = band.clientWidth;
    if (!n || !w) return;
    var min = parseFloat(getComputedStyle(band).getPropertyValue('--chip-min')) || 8.5;   // rem
    var cell = min * parseFloat(getComputedStyle(document.documentElement).fontSize);
    var rows = Math.ceil(n / Math.max(1, Math.floor(w / cell))), cols = Math.ceil(n / rows), gap = rows * cols - n;
    band.style.setProperty('--cols', String(cols));
    for (var i = 0; i < n; i++) band.children[i].style.gridColumn = i === n - 1 && gap ? 'span ' + (gap + 1) : '';
  }
  var ro = new ResizeObserver(function (entries) { entries.forEach(function (e) { balance(e.target); }); });
  var mo = new MutationObserver(function (recs) { recs.forEach(function (r) { balance(r.target); }); });
  bands.forEach(function (band) { ro.observe(band); mo.observe(band, { childList: true }); balance(band); });
})();

// Plotly charts reach a screen reader as loose tick labels. A chart with no name of its own is announced as one image,
// named by its title, else by the nearest heading above it; the name follows the title through redraws.
(function () {
  if (!window.MutationObserver) return;
  function name(gd) {
    var t = gd.layout && gd.layout.title, text = (t && t.text) || (typeof t === 'string' ? t : '');
    if (!text) {
      var hs = document.querySelectorAll('h2, h3, h4');
      for (var i = 0; i < hs.length && hs[i].compareDocumentPosition(gd) & 4; i++) text = hs[i].textContent;
    }
    gd.setAttribute('role', 'img');
    gd.setAttribute('aria-label', String(text || 'Chart').replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim());
  }
  function adopt(gd) {
    if (!gd.classList.contains('js-plotly-plot') || gd.dataset.named || gd.hasAttribute('aria-label') || gd.hasAttribute('aria-labelledby')) return;
    gd.dataset.named = '1';
    name(gd);
    if (gd.on) gd.on('plotly_afterplot', function () { name(gd); });
  }
  document.querySelectorAll('.js-plotly-plot').forEach(adopt);
  new MutationObserver(function (recs) { recs.forEach(function (r) { adopt(r.target); }); })
    .observe(document.documentElement, { attributes: true, attributeFilter: ['class'], subtree: true });
})();

// Maps that fill the window (the viewers' .mv-map, or any element marked data-fit): --fit-top is the element's top on
// the page and --fit-below the height of what must stay on screen under it (data-fit-below, a selector: the viewers'
// Show row), so a height of calc(100vh - var(--fit-top) - var(--fit-below)) puts the whole map and its controls on
// the first screen, at any window size, with nothing to scroll. --fit-lead (data-fit-with, a selector: a viewer's time
// bar) is what a taller map shares the screen with once the page is scrolled to that element: the sticky header and
// the distance from the element's top to the map's. Re-measured when anything above it changes height.
(function () {
  var els = document.querySelectorAll('.mv-map, [data-fit]');
  if (!els.length) return;
  var header = document.querySelector('.sh');
  function measure() {
    els.forEach(function (el) {
      var top = el.getBoundingClientRect().top + window.scrollY, below = 0;
      var sel = el.getAttribute('data-fit-below');
      if (sel) document.querySelectorAll(sel).forEach(function (b) { if (b.offsetParent) below += b.getBoundingClientRect().height; });
      el.style.setProperty('--fit-top', Math.round(top) + 'px');
      el.style.setProperty('--fit-below', Math.round(below) + 'px');
      var lead = el.getAttribute('data-fit-with'), first = lead && document.querySelector(lead);
      if (first) el.style.setProperty('--fit-lead', Math.round((header ? header.offsetHeight : 0) + top - first.getBoundingClientRect().top - window.scrollY) + 'px');
    });
  }
  measure();
  addEventListener('resize', measure);
  addEventListener('load', measure);
  if (window.ResizeObserver) { var ro = new ResizeObserver(measure); ro.observe(document.body); }
})();
