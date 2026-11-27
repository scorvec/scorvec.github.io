/* Line charts drawn in the page as SVG, a row of panels at the host's own width.

   window.vcharts.draw(host, spec) with spec = {
     panels: [{ title, u, v, flip, series, refs, dodge, ends, markers, whiskers, height, tipHead, tipFmt }],
     cols, minWidth, height, legend: [{ name, color, shape, note }], keys: ["a line of key text"], label, data: { href, text },
     share: () => url
   }
   label names the chart in a saved file (else its card's heading). Under every chart a line saves it as a PNG or an
   SVG (the legend, keys and panels with the page's styles written in) or its points as a CSV table, links data, and
   with share copies the link that opens the plot again (and puts it in the address bar).
   u (the independent axis): { type: "linear" | "log" | "logit" | "time", domain, ticks: [number | { u, label }], fmt, label }
   v (the value axis): { domain, floor, ceil, fmt, label }
   series: [{ name, color, shape, note, line, markers, pts: [{ u, v, lo, hi, on }] }]; a null v breaks the line,
   lo and hi draw a whisker (whiskers: false keeps them for the tooltip alone), on === false leaves the marker hollow.
   refs: [{ v, label, soft }] a line across the panel at a value (soft: drawn only where the values reach it), its
   label at the end of the line with the least drawn under it; [{ diag: true }] the line v = u.
   v.floor and v.ceil stop the padding of the value axis at a bound.
   flip puts the independent axis on the vertical (profiles against pressure). */
(function () {
  "use strict";
  const NS = "http://www.w3.org/2000/svg", DAY = 864e5, CHAR = 6.7, TEXT = 12.5;
  const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const r1 = v => Math.round(v * 10) / 10;
  const TRANS = { linear: v => v, time: v => v, log: Math.log10, logit: p => Math.log(p / (1 - p)) };
  let serial = 0;

  // tick values at 1, 2, 2.5 or 5 times a power of ten
  function nice(lo, hi, n) {
    const raw = (hi - lo || Math.abs(hi) || 1) / Math.max(1, n), mag = 10 ** Math.floor(Math.log10(raw));
    const step = [1, 2, 2.5, 5, 10].find(s => s * mag >= raw - mag * 1e-9) * mag, ticks = [];
    for (let k = Math.ceil(lo / step - 1e-9); k * step <= hi + step * 1e-9; k++) ticks.push(k * step);
    return { step, ticks };
  }
  function stepFormat(step) {
    const d = Math.max(0, -Math.floor(Math.log10(step) + 1e-9)), more = (step * 10 ** d) % 1 > 1e-9 ? 1 : 0;
    return v => (Math.abs(v) < step * 1e-9 ? 0 : v).toFixed(d + more);
  }
  // day ticks back from the last date, or month starts with the year at each January
  function timeTicks(a, b, max) {
    const days = (b - a) / DAY, out = [], step = [1, 2, 7, 14].find(s => days / s <= max);
    if (step) {
      for (let t = b; t >= a; t -= step * DAY) {
        const d = new Date(t);
        out.unshift({ u: t, label: d.getUTCDate() + " " + MONTHS[d.getUTCMonth()] });
      }
      return out;
    }
    const every = [1, 2, 3, 6, 12].find(s => days / 30.4 / s <= max) || 12, first = new Date(a), y = first.getUTCFullYear();
    for (let m = first.getUTCMonth(); Date.UTC(y, m, 1) <= b; m++) {
      const t = Date.UTC(y, m, 1), mm = m % 12;
      if (t >= a && mm % every === 0) out.push({ u: t, label: mm === 0 || !out.length ? `${MONTHS[mm]} ${y + Math.floor(m / 12)}` : MONTHS[mm] });
    }
    return out;
  }
  function marker(shape, x, y, r) {
    if (shape === "square") { const s = r * 0.9; return `M${r1(x - s)},${r1(y - s)}h${r1(2 * s)}v${r1(2 * s)}h${r1(-2 * s)}z`; }
    if (shape === "triangle") {
      const s = r * 1.2;
      return `M${r1(x)},${r1(y - s)}L${r1(x + s * 0.95)},${r1(y + s * 0.7)}L${r1(x - s * 0.95)},${r1(y + s * 0.7)}z`;
    }
    if (shape === "diamond") { const s = r * 1.25; return `M${r1(x)},${r1(y - s)}L${r1(x + s)},${r1(y)}L${r1(x)},${r1(y + s)}L${r1(x - s)},${r1(y)}z`; }
    return `M${r1(x - r)},${r1(y)}a${r},${r} 0 1,0 ${2 * r},0a${r},${r} 0 1,0 ${-2 * r},0z`;
  }
  const dot = (s, on, r) => (on === false
    ? `<path d="${marker(s.shape, 0, 0, r)}" style="fill:var(--panel);stroke:${s.color};stroke-width:1.8"/>`
    : `<path d="${marker(s.shape, 0, 0, r)}" style="fill:${s.color};stroke:var(--panel);stroke-width:1"/>`);
  // a marker as a small inline picture, for legends and tooltips
  const swatch = (s, on) => `<svg class="vsw" viewBox="-8 -8 16 16" aria-hidden="true"><path d="M-8,0H8" stroke="${s.color}" stroke-width="2"/>${dot(s, on, 4.5)}</svg>`;
  const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };

  function panel(p, W, H0, host) {
    const wrap = el("div", "vpanel"), flip = !!p.flip, u = p.u || {}, v = p.v || {};
    if (p.title) wrap.append(el("div", "vtitle", p.title));
    const series = p.series.filter(s => s.pts.some(q => q.v != null));
    if (!series.length) { wrap.append(el("p", "vnone", "Nothing scored for this choice yet.")); return wrap; }
    const H = p.height || H0, whiskers = p.whiskers !== false;

    // the value axis: every value, and each interval as far as 60 % of the values' own range beyond them
    const vals = [], ends = [];
    for (const s of series) for (const q of s.pts) if (q.v != null) { vals.push(q.v); if (whiskers && q.lo != null && q.hi != null) ends.push(q.lo, q.hi); }
    for (const r of p.refs || []) if (r.v != null && !r.soft) vals.push(r.v);
    let lo = Math.min(...vals), hi = Math.max(...vals);
    if (hi - lo < 1e-9 * Math.max(Math.abs(lo), Math.abs(hi))) hi = lo;   // equal but for rounding: one value, not a range to cut into ticks
    const reach = (hi - lo || Math.abs(hi) || 1) * 0.6;
    if (ends.length) { lo = Math.max(Math.min(lo, ...ends), lo - reach); hi = Math.min(Math.max(hi, ...ends), hi + reach); }
    if (v.domain) [lo, hi] = v.domain;
    else {
      const pad = (hi - lo || Math.abs(hi) || 1) * 0.07;
      lo -= pad; hi += pad;
      if (v.floor != null) lo = Math.max(lo, v.floor);
      if (v.ceil != null) hi = Math.min(hi, v.ceil);
    }
    const vt = nice(lo, hi, flip ? W / 80 : H / 52), vfmt = v.fmt || stepFormat(vt.step);

    // the independent axis
    const us = [...new Set(series.flatMap(s => s.pts.filter(q => q.v != null).map(q => q.u)))].sort((a, b) => a - b);
    const type = u.type || "linear", tr = TRANS[type], ud = u.domain || [us[0], us[us.length - 1]];
    const room = flip ? H : W;
    let uticks = u.ticks;
    if (!uticks) uticks = type === "time" ? timeTicks(ud[0], ud[1], Math.max(2, Math.floor(room / 74))) : nice(ud[0], ud[1], room / 60).ticks;
    const ufmt = u.fmt || (x => String(x));
    uticks = uticks.map(t => (typeof t === "object" ? t : { u: t, label: ufmt(t) })).filter(t => tr(t.u) >= tr(ud[0]) - 1e-9 && tr(t.u) <= tr(ud[1]) + 1e-9);

    // margins: the widest tick label on the left, the end labels on the right
    const widest = arr => Math.max(0, ...arr.map(s => String(s).length)) * CHAR;
    const left = Math.ceil((flip ? widest(uticks.map(t => t.label)) : widest(vt.ticks.map(vfmt))) + 12);
    const endText = s => s.name + (s.note ? " " + s.note : "");
    const withEnds = !flip && p.ends && W >= 420;
    const right = withEnds ? Math.ceil(widest(series.map(endText)) + 30) : 14;
    const top = 10, bottom = 24 + ((flip ? v.label : u.label) ? 18 : 0), inset = 9;
    const pw = W - left - right, ph = H - top - bottom, span = (flip ? ph : pw) - 2 * inset;
    const a = tr(ud[0]), b = tr(ud[1]);
    const upx = x => (flip ? top : left) + inset + ((tr(x) - a) / (b - a || 1)) * span;
    const vpx = y => (flip ? left + ((y - lo) / (hi - lo)) * pw : top + ph - ((y - lo) / (hi - lo)) * ph);
    const at = (x, y) => (flip ? [vpx(y), upx(x)] : [upx(x), vpx(y)]);

    // side-by-side offsets so whiskers never sit on each other, no wider than the gap between values allows
    let gap = Infinity;
    for (let i = 1; i < us.length; i++) gap = Math.min(gap, Math.abs(upx(us[i]) - upx(us[i - 1])));
    const dodge = p.dodge ? Math.min(p.dodge, gap / (series.length + 0.5)) : 0;
    const off = i => (i - (series.length - 1) / 2) * dodge;
    const pos = (i, x, y) => { const [px, py] = at(x, y); return flip ? [px, py + off(i)] : [px + off(i), py]; };

    const id = "vc" + ++serial, g = [];
    g.push(`<defs><clipPath id="${id}"><rect x="${left}" y="${top}" width="${pw}" height="${ph}"/></clipPath></defs>`);
    // grid and tick labels
    for (const t of vt.ticks) {
      if (flip) {
        const x = r1(vpx(t));
        g.push(`<path class="vgrid" d="M${x},${top}V${top + ph}"/><text x="${x}" y="${top + ph + 16}" text-anchor="middle">${esc(vfmt(t))}</text>`);
      } else {
        const y = r1(vpx(t));
        g.push(`<path class="vgrid" d="M${left},${y}H${left + pw}"/><text x="${left - 7}" y="${y + 4}" text-anchor="end">${esc(vfmt(t))}</text>`);
      }
    }
    let lastEdge = -Infinity;
    for (const t of [...uticks].sort((x, y) => upx(x.u) - upx(y.u))) {
      const c = r1(upx(t.u)), w = String(t.label).length * CHAR;
      if (flip) {
        if (c - TEXT / 2 > lastEdge + 1) {
          g.push(`<path class="vtick" d="M${left - 4},${c}H${left}"/><text x="${left - 7}" y="${c + 4}" text-anchor="end">${esc(t.label)}</text>`);
          lastEdge = c + TEXT / 2;
        }
      } else if (c - w / 2 > lastEdge + 6) {
        g.push(`<path class="vtick" d="M${c},${top + ph}v4"/><text x="${c}" y="${top + ph + 17}" text-anchor="middle">${esc(t.label)}</text>`);
        lastEdge = c + w / 2;
      }
    }
    g.push(`<path class="vaxis" d="${flip ? `M${left},${top}V${top + ph}` : `M${left},${top + ph}H${left + pw}`}"/>`);
    const axisLabel = flip ? v.label : u.label;
    if (axisLabel) g.push(`<text class="vlab" x="${left + pw / 2}" y="${H - 5}" text-anchor="middle">${esc(axisLabel)}</text>`);
    // whiskers under lines under markers; `ink` samples where the lines run, for placing the reference labels
    const marks = [], lines = [], whisk = [], ink = [];
    series.forEach((s, i) => {
      let d = "", pen = null;
      for (const q of s.pts) {
        if (q.v == null) { pen = null; continue; }
        const [x, y] = pos(i, q.u, q.v);
        d += (pen ? "L" : "M") + r1(x) + "," + r1(y);
        const n = pen ? Math.ceil(Math.hypot(x - pen[0], y - pen[1]) / 8) : 1;
        for (let k = 1; k <= n; k++) ink.push(pen ? [pen[0] + ((x - pen[0]) * k) / n, pen[1] + ((y - pen[1]) * k) / n] : [x, y]);
        pen = [x, y];
        if (whiskers && q.lo != null && q.hi != null) {
          const [x0, y0] = pos(i, q.u, q.lo), [x1, y1] = pos(i, q.u, q.hi);
          whisk.push(`<path d="M${r1(x0)},${r1(y0)}L${r1(x1)},${r1(y1)}" stroke="${s.color}"/>`);
        }
        if (s.markers !== false && p.markers !== false) marks.push(`<g transform="translate(${r1(x)},${r1(y)})">${dot(s, q.on, 4.5)}</g>`);
      }
      if (s.line !== false) lines.push(`<path d="${d}" stroke="${s.color}"${s.dash ? ` stroke-dasharray="${s.dash}"` : ""}/>`);
    });
    // reference lines under the data, each label over it at the end of the line with the least drawn under it
    const reft = [];
    for (const r of p.refs || []) {
      if (r.diag) {
        const [x0, y0] = at(ud[0], ud[0]), [x1, y1] = at(ud[1], ud[1]);
        g.push(`<path class="vref" clip-path="url(#${id})" d="M${r1(x0)},${r1(y0)}L${r1(x1)},${r1(y1)}"/>`);
      } else if (r.v >= lo && r.v <= hi) {
        const c = r1(vpx(r.v));
        g.push(`<path class="vref" d="${flip ? `M${c},${top}V${top + ph}` : `M${left},${c}H${left + pw}`}"/>`);
        if (!r.label) continue;
        const w = String(r.label).length * CHAR, spots = [];
        // across: at the left or right end, above or below the line; flipped: right or left of the line, at the top or bottom
        for (const far of [false, true]) for (const second of [false, true]) {
          const x = flip ? (far ? c - 5 : c + 5) : far ? left + pw - 5 : left + 5;
          const y = flip ? (second ? top + ph - 6 : top + 13) : second ? c + 15 : c - 5;
          const x0 = far ? x - w : x, inside = y - 13 >= top - 2 && y + 3 <= top + ph && x0 >= left && x0 + w <= left + pw;
          const under = ink.filter(([px, py]) => px > x0 - 4 && px < x0 + w + 4 && py > y - 16 && py < y + 6).length;
          spots.push({ x, y, anchor: far ? "end" : "start", cost: under + (inside ? 0 : 1e6) });
        }
        const best = spots.reduce((m, q) => (q.cost < m.cost ? q : m));
        reft.push(`<text class="vreft" x="${r1(best.x)}" y="${r1(best.y)}" text-anchor="${best.anchor}">${esc(r.label)}</text>`);
      }
    }
    g.push(`<g class="vwhisk" clip-path="url(#${id})">${whisk.join("")}</g><g class="vline" clip-path="url(#${id})">${lines.join("")}</g><g>${marks.join("")}</g>${reft.join("")}`);
    // the names at the line ends, pushed apart where they would collide
    if (withEnds) {
      const tags = series.map((s, i) => {
        const q = [...s.pts].reverse().find(x => x.v != null), [, y] = pos(i, q.u, q.v);
        return { s, y, text: endText(s) };
      }).sort((x, y) => x.y - y.y);
      for (let i = 1; i < tags.length; i++) tags[i].y = Math.max(tags[i].y, tags[i - 1].y + 16);
      const floor = top + ph - 2;
      for (let i = tags.length - 1; i >= 0; i--) tags[i].y = Math.min(tags[i].y, i === tags.length - 1 ? floor : tags[i + 1].y - 16);
      for (const t of tags) g.push(`<g transform="translate(${left + pw + 12},${r1(t.y)})">${dot(t.s, true, 4)}</g><text class="vend" x="${left + pw + 21}" y="${r1(t.y + 4)}">${esc(t.text)}</text>`);
    }
    g.push(`<g class="vhover"></g><rect class="vhit" x="${left}" y="${top}" width="${pw}" height="${ph}" fill="transparent"/>`);

    const svg = document.createElementNS(NS, "svg");
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.setAttribute("width", W);
    svg.setAttribute("height", H);
    svg.setAttribute("role", "img");
    svg.setAttribute("tabindex", "0");
    svg.setAttribute("aria-label", `${wrap.textContent.trim()}: ${series.map(s => s.name).join(", ")}. Arrow keys read the values.`);
    svg.innerHTML = g.join("");
    wrap.append(svg);

    // the hover layer: the nearest value along the independent axis, every model's number at it
    const hover = svg.querySelector(".vhover"), tip = host.querySelector(".vtip"), places = us.map(upx);
    const head = p.tipHead || (x => (u.label ? u.label + " " : "") + ufmt(x)), tfmt = p.tipFmt || stepFormat(vt.step / 10);
    let cur = -1;
    function show(k) {
      cur = k;
      const x = us[k], c = r1(places[k]), rows = [], rings = [];
      series.forEach((s, i) => {
        const q = s.pts.find(t => t.u === x && t.v != null);
        if (!q) return;
        const [px, py] = pos(i, x, q.v);
        rings.push(`<circle cx="${r1(px)}" cy="${r1(py)}" r="7.5"/>`);
        rows.push(`<tr><td>${swatch(s, q.on)}${esc(s.name)}</td><td>${esc(tfmt(q.v))}</td><td>${q.lo != null && q.hi != null ? esc(tfmt(q.lo)) + " to " + esc(tfmt(q.hi)) : ""}</td></tr>`);
      });
      hover.innerHTML = `<path class="vguide" d="${flip ? `M${left},${c}H${left + pw}` : `M${c},${top}V${top + ph}`}"/>${rings.join("")}`;
      tip.innerHTML = `<div class="vth">${esc(head(x))}</div><table>${rows.join("")}</table>`;
      tip.hidden = false;
      const hb = host.getBoundingClientRect(), sb = svg.getBoundingClientRect(), scale = sb.width / W;
      const ax = sb.left - hb.left + (flip ? left + pw / 2 : c) * scale, ay = sb.top - hb.top + (flip ? c : top) * scale;
      const tw = tip.offsetWidth, th = tip.offsetHeight;
      let tx = flip ? ax - tw / 2 : ax + 14;
      if (!flip && tx + tw > hb.width) tx = ax - 14 - tw;
      tip.style.left = Math.max(0, Math.min(tx, hb.width - tw)) + "px";
      tip.style.top = (flip ? (ay + 16 + th > sb.bottom - hb.top ? ay - 16 - th : ay + 16) : Math.max(0, ay + 4)) + "px";
    }
    function hide() { cur = -1; hover.innerHTML = ""; tip.hidden = true; }
    function nearest(e) {
      const sb = svg.getBoundingClientRect(), c = ((flip ? e.clientY - sb.top : e.clientX - sb.left) * W) / sb.width;
      let best = 0;
      for (let k = 1; k < places.length; k++) if (Math.abs(places[k] - c) < Math.abs(places[best] - c)) best = k;
      if (best !== cur) show(best);
    }
    const hit = svg.querySelector(".vhit");
    hit.addEventListener("pointermove", nearest);
    hit.addEventListener("pointerdown", nearest);
    hit.addEventListener("pointerleave", hide);
    svg.addEventListener("blur", hide);
    svg.addEventListener("keydown", e => {
      const step = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
      if (e.key === "Escape") hide();
      if (!step) return;
      e.preventDefault();
      show(Math.max(0, Math.min(us.length - 1, cur < 0 ? (step > 0 ? 0 : us.length - 1) : cur + step)));
    });
    return wrap;
  }

  function render(host, spec) {
    const W = Math.floor(host.clientWidth), gapPx = 22;
    if (!W || !spec) return;
    const min = spec.minWidth || 300;
    const n = Math.max(1, Math.min(spec.cols || spec.panels.length, Math.floor((W + gapPx) / (min + gapPx))));
    const pw = Math.floor((W - gapPx * (n - 1)) / n);
    const H = Math.round(Math.max(220, Math.min(spec.height || 320, window.innerHeight * 0.72, pw * 1.05)));
    host.textContent = "";
    if (spec.legend || spec.keys) {
      const items = (spec.legend || []).map(s => `<span>${swatch(s, true)}${esc(s.name)}${s.note ? ` <b>${esc(s.note)}</b>` : ""}</span>`);
      const keys = (spec.keys || []).map(k => `<span class="vkey">${k}</span>`);
      host.append(el("div", "vlegend", items.join("") + keys.join("")));
    }
    const tip = el("div", "vtip");
    tip.hidden = true;
    const grid = el("div", "vgridrow");
    grid.style.gridTemplateColumns = `repeat(${n}, minmax(0, 1fr))`;
    host.append(grid, tip);
    for (const p of spec.panels) grid.append(panel(p, pw, H, host));
    // the saving line: the chart as a picture or its numbers as a table, and the file it was drawn from
    const save = el("p", "vdata", "Save this chart: ");
    [["PNG", () => toPng(host, spec)], ["SVG", () => toSvg(host, spec)], ["CSV", () => toCsv(host, spec)]].forEach(([name, f], k) => {
      const b = el("button", "vsave", name);
      b.type = "button";
      b.addEventListener("click", f);
      save.append(...(k ? [" · ", b] : [b]));
    });
    if (spec.data) {
      const a = el("a", null, esc(spec.data.text || "JSON"));
      a.href = spec.data.href;
      save.append(" · ", a);
    }
    if (spec.share) {   // a link that opens this plot again
      const b = el("button", "vsave", "Copy link");
      b.type = "button";
      b.addEventListener("click", () => {
        const url = spec.share(), done = () => { b.textContent = "Link copied"; setTimeout(() => { b.textContent = "Copy link"; }, 2000); };
        history.replaceState(null, "", url);
        (navigator.clipboard ? navigator.clipboard.writeText(url) : Promise.reject()).then(done, done);
      });
      save.append(" · ", b);
    }
    host.append(save);
  }

  // ── saving: one SVG of the legend, the panels and a source line, with the page's styles written into it, so that
  // it reads the same out of the page; a PNG of it at twice the size; the points as a long CSV table
  const STYLE = ["fill", "stroke", "stroke-width", "stroke-dasharray", "stroke-linejoin", "stroke-linecap", "opacity",
    "font-size", "font-weight", "paint-order", "text-anchor"];
  const FONTS = "'Source Sans 3', 'Helvetica Neue', Arial, sans-serif";
  const plain = html => { const d = document.createElement("div"); d.innerHTML = html || ""; return d.textContent.replace(/\s+/g, " ").trim(); };
  const titleOf = (host, spec) => spec.label || host.closest("section")?.querySelector("h2")?.textContent.trim() || "Chart";
  const slugOf = s => s.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "").slice(0, 60) || "chart";
  function download(blob, name) {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = name;
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 4000);
  }
  // a panel's SVG with every element's computed style inline, the hover layer left out
  function inlined(svg) {
    const copy = svg.cloneNode(true), src = [svg, ...svg.querySelectorAll("*")], dst = [copy, ...copy.querySelectorAll("*")];
    src.forEach((e, k) => {
      const cs = getComputedStyle(e);
      dst[k].setAttribute("style", STYLE.map(p => `${p}:${cs.getPropertyValue(p)}`).join(";") + `;font-family:${FONTS}`);
    });
    copy.querySelectorAll(".vhover, .vhit").forEach(e => e.remove());
    return copy;
  }
  function composed(host, spec) {
    const W = Math.floor(host.clientWidth), cs = getComputedStyle(host);
    const ink = cs.getPropertyValue("--ink").trim() || "#1c1c1a", ink2 = cs.getPropertyValue("--ink2").trim() || "#3d3b37";
    const muted = cs.getPropertyValue("--muted").trim() || "#6f6b64", bg = cs.getPropertyValue("--panel").trim() || "#fff";
    const parts = [], pad = 16;
    let y = pad + 18;
    parts.push(`<text x="${pad}" y="${y}" style="font:600 17px ${FONTS};fill:${ink}">${esc(titleOf(host, spec))}</text>`);
    // the legend, wrapped to the width
    let x = pad;
    y += 26;
    for (const s of spec.legend || []) {
      const text = s.name + (s.note ? "  " + s.note : ""), w = 26 + text.length * 7.2;
      if (x + w > W - pad && x > pad) { x = pad; y += 22; }
      parts.push(`<g transform="translate(${x + 8},${y - 4})"><path d="M-8,0H8" stroke="${s.color}" stroke-width="2"/>${dot(s, true, 4.5)}</g>` +
        `<text x="${x + 22}" y="${y}" style="font:13px ${FONTS};fill:${ink2}">${esc(text)}</text>`);
      x += w + 14;
    }
    for (const k of spec.keys || []) {
      y += 20;
      parts.push(`<text x="${pad}" y="${y}" style="font:12px ${FONTS};fill:${muted}">${esc(plain(k))}</text>`);
    }
    const grid = host.querySelector(".vgridrow"), top = y + 14, origin = grid.getBoundingClientRect();
    let bottom = top;
    for (const p of host.querySelectorAll(".vpanel")) {
      const svg = p.querySelector("svg"), title = p.querySelector(".vtitle");
      if (!svg) continue;
      const r = p.getBoundingClientRect(), sr = svg.getBoundingClientRect(), ox = r.left - origin.left + pad, oy = top + sr.top - origin.top;
      if (title) {
        const main = title.cloneNode(true), sub = main.querySelector("small");
        if (sub) sub.remove();
        parts.push(`<text x="${ox}" y="${oy - (sub ? 24 : 8)}" style="font:600 13.5px ${FONTS};fill:${ink}">${esc(main.textContent.trim())}</text>`);
        if (sub) parts.push(`<text x="${ox}" y="${oy - 7}" style="font:12.5px ${FONTS};fill:${ink2}">${esc(sub.textContent.trim())}</text>`);
      }
      const c = inlined(svg), vb = svg.viewBox.baseVal;
      c.setAttribute("x", ox); c.setAttribute("y", oy);
      c.setAttribute("width", sr.width); c.setAttribute("height", sr.width * vb.height / vb.width);
      parts.push(c.outerHTML);
      bottom = Math.max(bottom, oy + sr.width * vb.height / vb.width);
    }
    // the line under the figure that names its observations and runs, wrapped to the width, then where it came from
    let note = host.nextElementSibling;
    while (note && !(note.matches("p.note") && note.textContent.trim())) note = note.nextElementSibling;
    const words = note ? note.textContent.trim().split(/\s+/) : [], lines = [];
    for (const w of words) {
      if (lines.length && (lines[lines.length - 1] + " " + w).length * 6.1 <= W) lines[lines.length - 1] += " " + w;
      else lines.push(w);
    }
    lines.forEach((l, k) => parts.push(`<text x="${pad}" y="${bottom + 22 + 16 * k}" style="font:12px ${FONTS};fill:${ink2}">${esc(l)}</text>`));
    const H = Math.ceil(bottom + 30 + (lines.length ? 10 + 16 * lines.length : 0));
    parts.push(`<text x="${pad}" y="${H - 10}" style="font:11.5px ${FONTS};fill:${muted}">scorvec.com/verification.html · saved ${new Date().toISOString().slice(0, 10)}</text>`);
    const width = W + 2 * pad;
    return { svg: `<svg xmlns="${NS}" viewBox="0 0 ${width} ${H}" width="${width}" height="${H}"><rect width="100%" height="100%" fill="${bg}"/>${parts.join("")}</svg>`, W: width, H };
  }
  function toSvg(host, spec) {
    download(new Blob([composed(host, spec).svg], { type: "image/svg+xml" }), slugOf(titleOf(host, spec)) + ".svg");
  }
  function toPng(host, spec) {
    const c = composed(host, spec), img = new Image(), scale = 2;
    img.onload = () => {
      const cv = document.createElement("canvas");
      cv.width = c.W * scale; cv.height = c.H * scale;
      const g = cv.getContext("2d");
      g.scale(scale, scale);
      g.drawImage(img, 0, 0, c.W, c.H);
      cv.toBlob(b => b && download(b, slugOf(titleOf(host, spec)) + ".png"), "image/png");
    };
    img.src = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(c.svg);
  }
  // every point as a row: the panel, the series, where along the axis, the value, its interval, and whether it is
  // drawn filled (a passed test) where the chart draws that
  function toCsv(host, spec) {
    const q = v => { const s = String(v ?? ""); return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s; };
    const rows = [["panel", "series", "x", "value", "low", "high", "filled"]];
    for (const p of spec.panels) {
      const time = (p.u || {}).type === "time";
      for (const s of p.series) for (const pt of s.pts) {
        if (pt.v == null) continue;
        rows.push([plain(p.title), s.name, time ? new Date(pt.u).toISOString().slice(0, 10) : pt.u, pt.v, pt.lo ?? "", pt.hi ?? "",
          pt.on == null ? "" : pt.on ? "yes" : "no"]);
      }
    }
    download(new Blob([rows.map(r => r.map(q).join(",")).join("\n") + "\n"], { type: "text/csv" }), slugOf(titleOf(host, spec)) + ".csv");
  }

  const state = new WeakMap();
  function draw(host, spec) {
    let st = state.get(host);
    if (!st) {
      st = { width: 0 };
      st.ro = new ResizeObserver(() => {
        const w = Math.floor(host.clientWidth);
        if (w && Math.abs(w - st.width) > 1) { st.width = w; render(host, st.spec); }
      });
      state.set(host, st);
      st.ro.observe(host);
    }
    st.spec = spec;
    st.width = Math.floor(host.clientWidth);
    render(host, spec);
  }
  // a line of text in place of the chart (nothing scored for the choice)
  function clear(host, text) {
    const st = state.get(host);
    if (st) st.spec = null;
    host.textContent = "";
    host.append(el("p", "vnone", esc(text)));
  }
  // a filled and a hollow marker, for a key line
  const pair = (color = "var(--ink2)") => swatch({ color, shape: "circle" }, true) + swatch({ color, shape: "circle" }, false);

  window.vcharts = { draw, clear, pair };
})();
