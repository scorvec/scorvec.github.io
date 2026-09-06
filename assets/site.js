// Site header behaviour: the mobile menu button and the Products dropdown.
// Everything in the header is plain HTML links; this only opens and closes.
(function () {
  var header = document.querySelector('.sh');
  if (!header) return;

  var toggle = header.querySelector('.sh-toggle');
  var nav = header.querySelector('.sh-nav');
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
    item.addEventListener('mouseenter', function () { if (window.matchMedia('(min-width: 861px)').matches) setOpen(true); });
    item.addEventListener('mouseleave', function () { if (window.matchMedia('(min-width: 861px)').matches) setOpen(false); });
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
