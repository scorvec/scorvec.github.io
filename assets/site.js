// Site header behaviour: the mobile menu button and the Products dropdown.
// Everything in the header is plain HTML links; this only opens and closes.
(function () {
  var header = document.querySelector('.sh');
  if (!header) return;

  var toggle = header.querySelector('.sh-toggle');
  var nav = header.querySelector('.sh-nav');
  // Collapse to the compact menu whenever the full one does not fit on one line (2026-09-25). Measured, not a
  // breakpoint: the header scales with each page's own root font size. Phones (<= 860 px) use the CSS rule.
  function fitHeader() {
    header.classList.remove('sh--compact');
    if (window.innerWidth <= 860 || !nav) return;
    var items = header.querySelectorAll('.sh-brand, .sh-list > li');
    var right = 0;
    for (var i = 0; i < items.length; i++) right = Math.max(right, items[i].getBoundingClientRect().right);
    if (right > document.documentElement.clientWidth - 4) header.classList.add('sh--compact');
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
