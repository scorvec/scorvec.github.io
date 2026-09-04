/* Shared heatmap table + colour ramps for the run-over-run pages.
 *
 * Shade varies, not opacity (a linear mix from white spends most of its range
 * within a few percent of white); 0.62 gamma on the magnitude so the mid-range
 * shows real colour while the ends stay where the legend says. */
window.HEAT = (() => {
  const RAMP = {
    warm:  ['#fdf3f0', '#f6d5cd', '#eeb1a3', '#e08a78', '#c95a45', '#a22e20'],
    cold:  ['#eef4fb', '#cfdff2', '#a9c6e6', '#7ea8d5', '#4f84bd', '#25609a'],
    more:  ['#f3f0fa', '#dbd2ef', '#bfaee1', '#9f88cf', '#7d62b8', '#5a3f98'],
    less:  ['#eef7f3', '#c9e6da', '#9fd1bd', '#70b79c', '#3f9878', '#1f7358'],
    green: ['#eef7ef', '#cfe8d2', '#a6d4ab', '#74ba7c', '#3f9a4b', '#1f7530'],
    amber: ['#fdf5ea', '#f7e0bd', '#efc48a', '#e3a457', '#cf822a', '#a6600f'],
    blue:  ['#eef4fb', '#cbdff3', '#9fc3e6', '#6fa3d6', '#3f80bf', '#1f5c98'],
    brown: ['#f8f2ec', '#ead8c4', '#d9b995', '#c39866', '#a8763f', '#835420'],
  };
  const hex = (c) => [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16));
  function mix(stops, t) {
    const p = Math.min(0.9999, Math.max(0, t)) * (stops.length - 1);
    const i = Math.floor(p), f = p - i;
    const a = hex(stops[i]), b = hex(stops[i + 1]);
    return `rgb(${a.map((x, k) => Math.round(x + (b[k] - x) * f)).join(',')})`;
  }
  function color(v, s, limb) {
    if (v === null || v === undefined || !Number.isFinite(v) || Math.abs(v) < 1e-9) return { bg: '', ink: 'var(--ink3)' };
    const t = Math.pow(Math.min(1, Math.abs(v) / s), 0.62);
    return { bg: mix(RAMP[limb(v)], t), ink: t > 0.62 ? '#fff' : 'var(--ink)' };
  }
  function autoScale(vals, nice) {
    const a = vals.filter((v) => v !== null && Number.isFinite(v)).map(Math.abs).sort((x, y) => x - y);
    if (!a.length) return nice[2];
    const q = a[Math.min(a.length - 1, Math.floor(a.length * 0.95))];
    return nice.find((n) => n >= q) || nice[nice.length - 1];
  }
  const NICE = [0.5, 1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 80, 100, 150, 200, 300, 500];
  let tipEl = null;
  function wireTips(host) {
    tipEl = tipEl || document.getElementById('tip');
    if (!tipEl) return;
    host.querySelectorAll('[data-t]').forEach((el) => {
      el.addEventListener('mousemove', (ev) => {
        tipEl.textContent = el.dataset.t; tipEl.style.opacity = 1;
        tipEl.style.left = Math.min(window.innerWidth - tipEl.offsetWidth - 12, ev.clientX + 14) + 'px';
        tipEl.style.top = (ev.clientY + 14) + 'px';
      });
      el.addEventListener('mouseleave', () => { tipEl.style.opacity = 0; });
    });
  }
  /* rows [{key,label,cls}], cols [{key,label,sub}], cell(rk,ck) -> {v, tip, mark, text?}|null,
   * extra [{label, value(rk) -> {text, cls, tip}}], opts {scale, limb, fmt} */
  function table(host, rows, cols, cell, extra, opts) {
    const o = Object.assign({ scale: 'auto', limb: (v) => (v > 0 ? 'more' : 'less'),
                              fmt: (v) => (v > 0 ? '+' : v < 0 ? '−' : '') + Math.abs(v).toFixed(1) }, opts || {});
    const cells = rows.map((r) => cols.map((c) => cell(r.key, c.key)));
    const s = o.scale === 'auto' ? autoScale(cells.flat().map((x) => (x ? x.v : null)), o.nice || NICE) : +o.scale;
    const H = ['<thead><tr><th class="k"></th>'];
    let last = '';
    cols.forEach((c) => {
      const day = c.sub !== undefined ? c.sub : '';
      const grp = day && day !== last; last = day;
      H.push(`<th class="${grp ? 'grp' : ''}"><span class="d">${grp ? day : ''}</span>${c.label}</th>`);
    });
    (extra || []).forEach((e) => H.push(`<th class="x">${e.label}</th>`));
    H.push('</tr></thead><tbody>');
    rows.forEach((r, i) => {
      H.push(`<tr class="${r.cls || ''}"><td class="k">${r.label}</td>`);
      cols.forEach((c, j) => {
        const x = cells[i][j];
        if (!x || x.v === null || x.v === undefined || !Number.isFinite(x.v)) {
          H.push(`<td class="v none"${x && x.tip ? ` data-t="${x.tip}"` : ''}>${x && x.tip ? '–' : ''}</td>`);
          return;
        }
        const col = color(x.v, s, o.limb);
        H.push(`<td class="v" style="background:${col.bg};color:${col.ink}" data-t="${x.tip || ''}">`
          + `${x.text !== undefined ? x.text : o.fmt(x.v)}${x.mark || ''}</td>`);
      });
      (extra || []).forEach((e) => {
        const t = e.value(r.key);
        H.push(`<td class="x${t && t.cls ? ' ' + t.cls : ''}"${t && t.tip ? ` data-t="${t.tip}"` : ''}>${t ? t.text : ''}</td>`);
      });
      H.push('</tr>');
    });
    H.push('</tbody>');
    host.innerHTML = H.join('');
    wireTips(host);
    return s;
  }
  function legend(s, limb, dnLabel, upLabel, fmt) {
    const f = fmt || ((x) => x);
    const sw = (l, t) => `<i style="background:${mix(RAMP[l], t)}"></i>`;
    return `<span class="lg">${[1, .66, .33].map((t) => sw(limb(-1), t)).join('')}`
      + `<b>−${f(s)}</b> ${dnLabel} &nbsp;&nbsp; 0 &nbsp;&nbsp; ${upLabel} <b>+${f(s)}</b>`
      + `${[.33, .66, 1].map((t) => sw(limb(1), t)).join('')}</span>`;
  }
  /* Legend toggles (user, 4 Sep 2026: "make it possible to select on and off
   * the different lines"). A chart tags its legend chips and every SVG element
   * of a series with data-k; clicking a chip hides that key. Hidden keys are
   * remembered per chart id, so a re-render (new data, control change) keeps
   * the reader's selection. Shift-click isolates one series. */
  const hidden = {};
  function applyToggles(host, id) {
    const off = hidden[id] || new Set();
    host.querySelectorAll('[data-k]').forEach((el) => {
      const on = !off.has(el.dataset.k);
      if (el.tagName === 'SPAN') el.classList.toggle('off', !on);
      else el.style.visibility = on ? '' : 'hidden';
    });
  }
  function wireToggles(host, id) {
    hidden[id] = hidden[id] || new Set();
    const chips = host.querySelectorAll('.evolegend span[data-k]');
    chips.forEach((ch) => ch.addEventListener('click', (ev) => {
      const k = ch.dataset.k; const off = hidden[id];
      if (ev.shiftKey) {            // isolate: everything else off, or all back on
        const others = [...chips].map((c) => c.dataset.k).filter((x) => x !== k);
        const alone = others.every((x) => off.has(x)) && !off.has(k);
        if (alone) off.clear(); else { off.clear(); others.forEach((x) => off.add(x)); }
      } else if (off.has(k)) off.delete(k); else off.add(k);
      applyToggles(host, id);
    }));
    applyToggles(host, id);
  }
  return { RAMP, mix, color, table, legend, wireTips, autoScale, NICE, wireToggles, applyToggles, hidden };
})();
