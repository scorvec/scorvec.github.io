/* Columbia basin precipitation page. Reads pnw_latest.json (divisions,
 * composites, Stage IV ledger, newest issue per model with deltas) and
 * pnw_history.json (period mm / % of normal per issue, same-day dprev). */
const S = { latest: null, hist: null, basin: 'Columbia abv The Dalles', model: 'blend',
            period: 'd1-10', heatwhat: 'pct', nruns: 12 };
const $ = (id) => document.getElementById(id);
const cyc = (c) => `${c.slice(4, 6)}/${c.slice(6, 8)} ${c.slice(8, 10)}Z`;
const cycDay = (c) => `${c.slice(4, 6)}/${c.slice(6, 8)}`;
const MODEL_LABEL = { blend: 'Blend', gfs: 'GFS', gefs: 'GEFS', ecmwf: 'ECMWF IFS', ecmwf_ens: 'ECMWF ENS',
                      aifs: 'AIFS', gdps: 'GDPS', geps: 'GEPS', geps_ext: 'GEPS ext. (Mon/Thu 00Z, 32 d)',
                      wn2: 'WeatherNext-2', wn3: 'WeatherNext-3' };
const MODEL_ORDER = ['blend', 'gefs', 'ecmwf_ens', 'geps', 'gfs', 'ecmwf', 'aifs', 'gdps', 'geps_ext', 'wn2', 'wn3'];
const MODEL_COLOR = { blend: '#0f172a', gfs: '#2b6cb0', gefs: '#6fa8dc', ecmwf: '#b3372a', ecmwf_ens: '#e0745f',
                      aifs: '#2f855a', gdps: '#b8860b', geps: '#d9b45b', geps_ext: '#8a6d1f',
                      wn2: '#8a5cc0', wn3: '#5a3f98' };
const hasBlend = () => !!(S.latest && S.latest.models && S.latest.models.blend);
const PERIOD_LABEL = { 'd1-5': 'days 1–5', 'd6-10': 'days 6–10', 'd11-15': 'days 11–15',
                       'd1-10': 'days 1–10', 'd1-15': 'days 1–15', 'd16-32': 'days 16–32 (GEPS ext.)' };
const limb = (v) => (v > 0 ? 'green' : 'brown');   // wet green, dry brown (user, 3 Sep 2026)
const limbAbs = () => 'green';                        // absolute mm: one-sided ramp
// S.heatwhat: 'pct' (% of normal), 'anom' (mm above/below normal), 'mm' (absolute)
const SHOW = { pct: '% of normal', anom: 'anomaly, mm vs normal', mm: 'absolute mm' };
const NICE_PCT = [5, 10, 15, 20, 30, 50, 80, 100, 150, 200], NICE_MM = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 80, 100];
// a period entry {mm, normal?, pct?} -> the number the page shows under S.heatwhat
function showVal(v) {
  if (!v) return null;
  if (S.heatwhat === 'mm') return v.mm;
  if (v.normal === undefined || v.normal === null) return null;
  return S.heatwhat === 'pct' ? v.pct : v.mm - v.normal;
}
// a same-day change {mm, pct?} under S.heatwhat (a change in anomaly is a change in mm)
function showDelta(d) {
  if (!d) return null;
  if (S.heatwhat === 'pct') return d.pct === undefined ? null : d.pct;
  return d.mm === undefined ? null : d.mm;
}
const showFmt = (v) => (v === null || v === undefined ? '–' : S.heatwhat === 'pct' ? `${v.toFixed(0)}%` : `${v.toFixed(0)} mm`);
const showSgn = (v) => (v === null || v === undefined ? '–' : sgn(v, S.heatwhat === 'pct' ? 0 : 1) + (S.heatwhat === 'pct' ? ' pts' : ' mm'));
const showUnitDelta = () => (S.heatwhat === 'pct' ? ' pts' : ' mm');
const showNice = () => (S.heatwhat === 'pct' ? NICE_PCT : NICE_MM);
const sgn = (v, n = 0) => (v === null || v === undefined ? '–' : (v > 0 ? '+' : v < 0 ? '−' : '') + Math.abs(v).toFixed(n));

// segmented button groups: items [{v, label, color?, cls?}] or plain values
function seg(id, items, value, onPick, labeller) {
  const host = $(id); if (!host) return; host.innerHTML = '';
  for (const it of items) {
    const o = typeof it === 'object' ? it : { v: it, label: labeller ? labeller(it) : it };
    const b = document.createElement('button'); b.type = 'button'; b.dataset.v = o.v;
    b.innerHTML = (o.color ? `<i class="dot" style="background:${o.color}"></i>` : '') + o.label;
    if (o.cls) b.className = o.cls; if (o.title) b.title = o.title;
    b.setAttribute('aria-pressed', String(o.v) === String(value));
    b.addEventListener('click', () => { setSeg(id, o.v); onPick(o.v); });
    host.appendChild(b);
  }
}
function setSeg(id, value) { const host = $(id); if (!host) return; host.querySelectorAll('button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.v) === String(value))); }
function fill(sel, items, value, labeller) {
  sel.innerHTML = '';
  for (const it of items) {
    const o = document.createElement('option');
    if (typeof it === 'object') { o.value = it.value; o.textContent = it.label; if (it.group) o.className = 'grp'; }
    else { o.value = it; o.textContent = labeller ? labeller(it) : it; }
    sel.appendChild(o);
  }
  sel.value = value;
}
const models = () => MODEL_ORDER.filter((m) => S.latest.models[m]).concat(Object.keys(S.latest.models).filter((m) => !MODEL_ORDER.includes(m)));
// A picked archived issue stands in for the newest one of its model. Same
// trick as the tracker's run picker: nothing downstream needs to know.
S.pick = {};
const entry = (m) => S.pick[m] || S.latest.models[m];
const ARCH = window.PNW_ARCH || (DATA_BASE() + 'archive/');
function DATA_BASE() { return window.PNW_DATA || '../data/out/'; }
async function loadArchive(m, cycle) {
  const r = await fetch(`${ARCH}${m}_${cycle}.json.gz?t=${Date.now()}`);
  if (!r.ok) throw new Error('archive not found');
  const buf = await r.arrayBuffer();
  let text;
  try {   // GitHub Pages serves .gz as application/gzip without content-encoding
    const ds = new DecompressionStream('gzip');
    text = await new Response(new Blob([buf]).stream().pipeThrough(ds)).text();
  } catch (e) { text = new TextDecoder().decode(buf); }   // already decompressed by the server
  return JSON.parse(text);
}
function statsOf(members, weights) {
  const nd = members[0].length; const out = { mean: [], p10: [], p90: [] };
  for (let j = 0; j < nd; j++) {
    const v = [], w = [];
    members.forEach((row, i) => { if (row[j] !== null && row[j] !== undefined) { v.push(row[j]); w.push(weights ? weights[i] : 1); } });
    if (!v.length) { out.mean.push(null); out.p10.push(null); out.p90.push(null); continue; }
    const tot = w.reduce((a, b) => a + b, 0);
    out.mean.push(Math.round(100 * v.reduce((a, x, i) => a + x * w[i], 0) / tot) / 100);
    out.p10.push(Math.round(100 * wq(v, w, 0.1)) / 100); out.p90.push(Math.round(100 * wq(v, w, 0.9)) / 100);
  }
  return out;
}
function issueEntry(rec) {
  // divisions + area-weighted composites -> series with stats and members;
  // periods with % of normal, mirroring the server's periods_of
  const series = {};
  for (const [c, v] of Object.entries(rec.div)) series[c] = { members: v.members, weights: v.weights };
  for (const [name, cs0] of Object.entries(S.latest.composites)) {
    const cs = cs0.filter((c) => rec.div[c]); if (!cs.length) continue;
    const area = cs.map((c) => divInfo(c).area); const tot = area.reduce((a, b) => a + b, 0);
    const nm = rec.div[cs[0]].members.length; const nd = rec.dates.length;
    const mem = Array.from({ length: nm }, (_, i) => Array.from({ length: nd }, (_, j) => {
      let t = 0; for (let k = 0; k < cs.length; k++) { const x = rec.div[cs[k]].members[i][j]; if (x === null || x === undefined) return null; t += x * area[k]; } return Math.round(100 * t / tot) / 100; }));
    series[name] = { members: mem, weights: rec.div[cs[0]].weights };
  }
  const out = {};
  for (const [k, v] of Object.entries(series)) { const st = statsOf(v.members, v.weights); if (rec.n_members > 1) { st.members = v.members; if (v.weights) st.weights = v.weights; } out[k] = st; }
  const periods = {};
  for (const [k, st] of Object.entries(out)) {
    const pr = {};
    for (const [p, [a, b]] of Object.entries(PERIOD_DAYS_JS)) {
      let mm = 0, nn = 0, n = 0, ok = true;
      for (let i = a - 1; i < Math.min(b, rec.dates.length); i++) { const v = st.mean[i]; if (v === null) continue; mm += v; n++; const no = normOn(k, rec.dates[i]); if (no === null) ok = false; else nn += no; }
      if (!n) continue;
      const e = { mm: Math.round(mm * 10) / 10, n, want: b - a + 1 };
      if (ok && nn > 0) { e.normal = Math.round(nn * 10) / 10; e.pct = Math.round(100 * mm / nn); }
      pr[p] = e;
    }
    periods[k] = pr;
  }
  return { cycle: rec.cycle, init: rec.init, run_day0: rec.run_day0, dates: rec.dates, n_members: rec.n_members,
           sources: rec.sources, series: out, periods, prev: [], picked: true };
}
async function pickRun(m, cycle) {
  const newest = S.latest.models[m] && S.latest.models[m].cycle;
  if (!cycle || cycle === newest) { delete S.pick[m]; return; }
  try { S.pick[m] = issueEntry(await loadArchive(m, cycle)); }
  catch (e) { delete S.pick[m]; alert && console.warn('archive load failed', e); }
}
function runList(m) {
  const cs = ((S.hist && S.hist.models && S.hist.models[m]) || []).map((x) => x.cycle);
  const newest = S.latest.models[m] && S.latest.models[m].cycle;
  if (newest && !cs.includes(newest)) cs.push(newest);
  return cs.sort().reverse();
}
function fillRuns() {
  const m = S.model; const cs = runList(m).slice(0, 6); const newest = S.latest.models[m] && S.latest.models[m].cycle;
  const cur = S.pick[m] ? S.pick[m].cycle : newest;
  seg('runsel', cs.map((c) => ({ v: c, label: c === newest ? `latest ${cyc(c)}` : cyc(c), cls: c === newest ? '' : 'muted' })), cur,
      async (c) => { await pickRun(S.model, c); render(); });
}
const isComp = (k) => !!S.latest.composites[k];
const divInfo = (code) => S.latest.divisions.find((d) => d.code === code);
const label = (k) => (isComp(k) ? k : (divInfo(k) || { name: k }).name);
function basinList() {
  const out = Object.keys(S.latest.composites).map((k) => ({ value: k, label: k }));
  const byReg = {};
  for (const d of S.latest.divisions) (byReg[d.region] = byReg[d.region] || []).push(d);
  for (const r of Object.keys(byReg).sort()) {
    for (const d of byReg[r].sort((a, b) => a.name.localeCompare(b.name))) {
      out.push({ value: d.code, label: `${r.replace(' Basin', '')} · ${d.name}` });
    }
  }
  return out;
}
function obsFor(k) { const o = S.latest.obs; return isComp(k) ? o.comp[k] : o.div[k]; }
function normFor(k) { return S.latest.obs.normal[k]; }
function normOn(k, date) {
  // daily normal on a forecast date: month's normal / days, from the divisions' table
  const d = new Date(date + 'T00:00:00'); const m = d.getMonth();
  const days = new Date(d.getFullYear(), m + 1, 0).getDate();
  if (isComp(k)) {
    const cs = S.latest.composites[k]; let a = 0, n = 0;
    for (const c of cs) { const di = divInfo(c); if (!di || !di.normals) continue; a += di.area; n += di.area * di.normals[m]; }
    return a ? n / a / days : null;
  }
  const di = divInfo(k); return di && di.normals ? di.normals[m] / days : null;
}

/* ---- plume ----------------------------------------------------------------------------- */
function plume() {
  const host = $('plume'); const k = S.basin;
  // the 32-day GEPS extension joins the plume only when it is the band model,
  // so the 16-day plume is not squeezed into half the width the rest of the time
  const ms = models().filter((m) => entry(m).series[k] && (m !== 'geps_ext' || S.model === 'geps_ext'));
  const odates = S.latest.obs.dates.slice(-30);
  const obs = obsFor(k) ? obsFor(k).slice(-30) : [];
  const fdates = [...new Set(ms.flatMap((m) => entry(m).dates))].sort();
  const dates = [...new Set(odates.concat(fdates))].sort();
  if (!dates.length) { host.innerHTML = '<div class="empty">Nothing archived yet.</div>'; return; }
  const W = Math.max(900, Math.min(1560, host.closest('section').clientWidth - 16));
  const H1 = 300, H2 = 140, GAP = 36, AX = 24, H = H1 + GAP + H2 + AX;
  const M = { l: 48, r: 48, t: 14 };
  const X = (d) => M.l + dates.indexOf(d) * (W - M.l - M.r) / Math.max(1, dates.length - 1);
  const step = (W - M.l - M.r) / Math.max(1, dates.length - 1);
  const ser = ms.map((m) => ({ m, e: entry(m), s: entry(m).series[k] }));
  const vals = obs.filter((v) => v !== null).concat(ser.flatMap((x) => x.s.p90.concat(x.s.mean)).filter((v) => v !== null));
  const y1 = Math.max(2, ...vals) * 1.08;
  const Y = (v) => M.t + (H1 - M.t - 6) * (1 - v / y1);
  const P = [`<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">`];
  for (let i = 0; i <= 4; i++) {
    const v = y1 * i / 4, y = Y(v);
    P.push(`<line x1="${M.l}" x2="${W - M.r}" y1="${y.toFixed(1)}" y2="${y.toFixed(1)}" stroke="#eef2f5"/>`);
    P.push(`<text x="${M.l - 7}" y="${(y + 3).toFixed(1)}" text-anchor="end">${v.toFixed(0)}</text>`);
  }
  P.push(`<text class="ttl" x="${M.l}" y="${M.t - 3}">mm per day (12Z–12Z)</text>`);
  // observed bars
  odates.forEach((d, i) => {
    const v = obs[i]; if (v === null || v === undefined) return;
    P.push(`<rect x="${(X(d) - step * 0.36).toFixed(1)}" y="${Y(v).toFixed(1)}" width="${(step * 0.72).toFixed(1)}" height="${Math.max(0.5, Y(0) - Y(v)).toFixed(1)}" fill="#7ea8d5" data-k="obs"/>`);
  });
  // divider between observed and forecast
  if (fdates.length && odates.length) {
    const xd = (X(odates[odates.length - 1]) + X(fdates[0])) / 2;
    P.push(`<line x1="${xd.toFixed(1)}" x2="${xd.toFixed(1)}" y1="${M.t}" y2="${H1}" stroke="#93a1b5" stroke-dasharray="3 3"/>`);
    P.push(`<text x="${(xd - 5).toFixed(1)}" y="${M.t + 10}" text-anchor="end" fill="#55637a">Stage IV</text>`);
    P.push(`<text x="${(xd + 5).toFixed(1)}" y="${M.t + 10}" fill="#55637a">forecast</text>`);
  }
  // daily normal, dashed
  let nd = '';
  dates.forEach((d) => { const n = normOn(k, d); if (n === null) return; nd += `${nd ? 'L' : 'M'}${X(d).toFixed(1)} ${Y(n).toFixed(1)}`; });
  P.push(`<path d="${nd}" fill="none" stroke="#0f172a" stroke-width="1.2" stroke-dasharray="5 3" opacity=".7" data-k="normal"/>`);
  // band for the chosen model, then EVERY member as a thin trace (user:
  // "for ensembles I would like to see the full plumes with all members")
  const b = ser.find((x) => x.m === S.model);
  if (b) {
    let up = '', dn = '';
    b.e.dates.forEach((d, i) => { if (b.s.p10[i] === null || b.s.p90[i] === null) return;
      up += `${up ? 'L' : 'M'}${X(d).toFixed(1)} ${Y(b.s.p90[i]).toFixed(1)}`; dn = `L${X(d).toFixed(1)} ${Y(b.s.p10[i]).toFixed(1)}` + dn; });
    if (up) P.push(`<path d="${up}${dn}Z" fill="${MODEL_COLOR[b.m] || '#999'}" opacity=".13" data-k="${b.m}"/>`);
    if (b.s.members) {
      const nm = b.s.members.length; const op = nm > 30 ? 0.22 : nm > 15 ? 0.3 : 0.4;
      b.s.members.forEach((row) => {
        let d = ''; let pen = false;
        b.e.dates.forEach((dd, i) => { const v = row[i]; if (v === null || v === undefined) { pen = false; return; }
          d += `${pen ? 'L' : 'M'}${X(dd).toFixed(1)} ${Y(Math.min(v, y1)).toFixed(1)}`; pen = true; });
        P.push(`<path d="${d}" fill="none" stroke="${MODEL_COLOR[b.m] || '#999'}" stroke-width=".9" opacity="${op}" stroke-linejoin="round" data-k="members"/>`);
      });
    }
  }
  ser.forEach((x) => {
    let d = '';
    x.e.dates.forEach((dd, i) => { if (x.s.mean[i] === null) return; d += `${d ? 'L' : 'M'}${X(dd).toFixed(1)} ${Y(x.s.mean[i]).toFixed(1)}`; });
    P.push(`<path d="${d}" fill="none" stroke="${MODEL_COLOR[x.m] || '#999'}" stroke-width="${x.m === S.model ? 3.2 : x.m === 'blend' ? 2.8 : 1.8}" stroke-linejoin="round" stroke-linecap="round" data-k="${x.m}"/>`);
  });
  const mn = hasBlend() ? null : meanSeries(k);
  if (mn && ser.length > 1) {
    let d = ''; mn.dates.forEach((dd, i) => { if (mn.mean[i] === null) return; d += `${d ? 'L' : 'M'}${X(dd).toFixed(1)} ${Y(mn.mean[i]).toFixed(1)}`; });
    P.push(`<path d="${d}" fill="none" stroke="#0f172a" stroke-width="3" stroke-linejoin="round" opacity=".85" data-k="mean"/>`);
  }
  // ---- lower panel: cumulative % of normal through the forecast
  const top2 = H1 + GAP;
  const cum = ser.map((x) => {
    let f = 0, n = 0; const out = [];
    x.e.dates.forEach((d, i) => { const v = x.s.mean[i], nn = normOn(k, d);
      if (v === null || nn === null) { out.push(null); return; } f += v; n += nn; out.push(n > 0 ? 100 * f / n : null); });
    return { m: x.m, dates: x.e.dates, pct: out };
  });
  // observed running % of normal over the last 30 days, for the same eye
  let of = 0, on = 0; const opct = odates.map((d, i) => { const v = obs[i], nn = normFor(k) ? normFor(k).slice(-30)[i] : null;
    if (v === null || v === undefined || nn === null) return null; of += v; on += nn; return on > 0 ? 100 * of / on : null; });
  const pv = cum.flatMap((c) => c.pct).concat(opct).filter((v) => v !== null);
  // capped at 300%: the first day or two of a run can sit at 400%+ of a small
  // September normal and would squash the part of the panel that matters
  const pmax = Math.min(300, Math.max(150, ...pv)) * 1.05;
  const Y2 = (v) => top2 + (H2 - 4) * (1 - Math.min(v, pmax) / pmax);
  P.push(`<text class="ttl" x="${M.l}" y="${top2 - 8}">cumulative % of normal — observed (last 30 days) and each run from its day 1</text>`);
  for (const v of [50, 100, 150, 200, 250, 300].filter((x) => x < pmax)) {
    P.push(`<line x1="${M.l}" x2="${W - M.r}" y1="${Y2(v).toFixed(1)}" y2="${Y2(v).toFixed(1)}" stroke="${v === 100 ? '#93a1b5' : '#eef2f5'}"/>`);
    P.push(`<text x="${M.l - 7}" y="${(Y2(v) + 3).toFixed(1)}" text-anchor="end">${v}%</text>`);
  }
  let od = ''; odates.forEach((d, i) => { if (opct[i] === null) return; od += `${od ? 'L' : 'M'}${X(d).toFixed(1)} ${Y2(opct[i]).toFixed(1)}`; });
  P.push(`<path d="${od}" fill="none" stroke="#4f84bd" stroke-width="2" data-k="obs"/>`);
  const lused = [];
  cum.forEach((c) => { let d = ''; c.dates.forEach((dd, i) => { if (c.pct[i] === null) return; d += `${d ? 'L' : 'M'}${X(dd).toFixed(1)} ${Y2(c.pct[i]).toFixed(1)}`; });
    P.push(`<path d="${d}" fill="none" stroke="${MODEL_COLOR[c.m] || '#999'}" stroke-width="${c.m === S.model ? 2.4 : 1.4}" data-k="${c.m}"/>`);
    const last = c.pct.map((v, i) => [v, i]).filter((x) => x[0] !== null).pop();
    if (last) {
      // end labels pushed apart where runs converge
      let y = Y2(last[0]);
      while (lused.some((u) => Math.abs(u - y) < 10)) y += 10;
      lused.push(y);
      P.push(`<text class="chg" x="${(X(c.dates[last[1]]) + 4).toFixed(1)}" y="${(y + 3).toFixed(1)}" fill="${MODEL_COLOR[c.m] || '#999'}" data-k="${c.m}">${last[0].toFixed(0)}%</text>`);
    }
  });
  dates.forEach((d, i) => {
    const dt = new Date(d + 'T00:00:00');
    if (dates.length > 24 && i % 2) return;
    P.push(`<text x="${X(d).toFixed(1)}" y="${H - 4}" text-anchor="middle" fill="${dt.getDay() % 6 === 0 ? '#b3392b' : '#55637a'}">${dt.getMonth() + 1}/${dt.getDate()}</text>`);
  });
  dates.forEach((d) => {
    const lines = [d]; const oi = odates.indexOf(d);
    if (oi >= 0 && obs[oi] !== null && obs[oi] !== undefined) lines.push(`Stage IV  ${obs[oi].toFixed(1)} mm`);
    const nn = normOn(k, d); if (nn !== null) lines.push(`normal  ${nn.toFixed(1)} mm`);
    if (mn) { const i = mn.dates.indexOf(d); if (i >= 0 && mn.mean[i] !== null) lines.push(`mean of models  ${mn.mean[i].toFixed(1)} mm`); }
    for (const x of ser) { const i = x.e.dates.indexOf(d); if (i >= 0 && x.s.mean[i] !== null) lines.push(`${MODEL_LABEL[x.m] || x.m}  ${x.s.mean[i].toFixed(1)} mm`); }
    P.push(`<rect class="hit" x="${(X(d) - step / 2).toFixed(1)}" y="${M.t}" width="${step.toFixed(1)}" height="${H - M.t - AX}" fill="transparent" data-t="${lines.join('\n')}"/>`);
  });
  P.push('</svg>');
  const chips = ser.map((x) => `<span class="${x.m === S.model ? 'new' : ''}" data-k="${x.m}"><i style="border-color:${MODEL_COLOR[x.m] || '#999'}"></i>${MODEL_LABEL[x.m] || x.m} ${cyc(x.e.cycle)}</span>`).join('')
    + (ser.length > 1 && !hasBlend() ? `<span class="new" data-k="mean"><i style="border-color:#0f172a;border-top-width:3px"></i>mean of models</span>` : '')
    + (b && b.s.members ? `<span data-k="members"><i style="border-color:${MODEL_COLOR[b.m] || '#999'};border-top-width:1px;opacity:.6"></i>${b.s.members.length} members</span>` : '')
    + `<span data-k="obs"><i style="border-color:#7ea8d5;border-top-width:6px"></i>Stage IV</span><span data-k="normal"><i style="border-color:#0f172a;border-top-style:dashed"></i>1991–2020 normal</span>`;
  host.innerHTML = `<div class="evolegend">${chips}</div>` + P.join('');
  HEAT.wireTips(host); HEAT.wireToggles(host, 'plume');
  const di = divInfo(k);
  $('sub1').innerHTML = `${label(k)}${di ? ` · ${di.region} · ${di.area.toLocaleString()} sq mi` : ' · area-weighted union of NWRFC divisions'} · member mean per model, p10–p90 band and members for ${MODEL_LABEL[S.model] || S.model} · <a class="histlink" id="plumehist">cumulative member plumes ▤</a>`;
  $('plumehist').addEventListener('click', () => openHist(k, S.period));
}

/* ---- models mean ------------------------------------------------------------------------------------ */
// Equal-weight mean of the models' member means on the dates they share: the
// single number the page headlines. Not a skill-weighted blend -- nothing here
// has been verified long enough to weight.
function meanPeriod(k, p) {
  const vs = models().map((m) => ((entry(m).periods[k] || {})[p])).filter(Boolean);
  if (!vs.length) return null;
  const mm = vs.reduce((a, v) => a + v.mm, 0) / vs.length;
  const nn = vs.filter((v) => v.normal !== undefined); const normal = nn.length ? nn.reduce((a, v) => a + v.normal, 0) / nn.length : undefined;
  const out = { mm, n: Math.max(...vs.map((v) => v.n)), want: vs[0].want, models: vs.length,
                lo: Math.min(...vs.map((v) => showVal(v))), hi: Math.max(...vs.map((v) => showVal(v))) };
  if (normal !== undefined) { out.normal = normal; out.pct = 100 * mm / normal; }
  return out;
}
function meanDelta(k, p) {
  const ds = models().map((m) => { const pr = entry(m).prev[0]; return pr ? (((pr.delta[k] || {}).periods || {})[p]) : null; }).filter(Boolean);
  if (!ds.length) return null;
  const mm = ds.reduce((a, d) => a + d.mm, 0) / ds.length;
  const pp = ds.filter((d) => d.pct !== undefined);
  return { mm, pct: pp.length ? pp.reduce((a, d) => a + d.pct, 0) / pp.length : undefined, n: ds.length };
}
function meanSeries(k) {
  const ser = models().map((m) => entry(m)).filter((e) => e.series[k]);
  if (!ser.length) return null;
  const dates = [...new Set(ser.flatMap((e) => e.dates))].sort();
  const mean = dates.map((d) => { const v = ser.map((e) => { const i = e.dates.indexOf(d); return i >= 0 ? e.series[k].mean[i] : null; }).filter((x) => x !== null);
    return v.length >= Math.ceil(ser.length / 2) ? v.reduce((a, b) => a + b, 0) / v.length : null; });
  return { dates, mean };
}

function headPeriod(k, p) {
  // the headline: the blend when there is one, else the equal-weight mean
  if (hasBlend()) {
    const v = (entry('blend').periods[k] || {})[p]; if (!v) return null;
    const others = models().filter((m) => m !== 'blend' && m !== 'geps_ext').map((m) => showVal((entry(m).periods[k] || {})[p])).filter((x) => x !== null && x !== undefined);
    const e = Object.assign({}, v, { models: others.length, lo: others.length ? Math.min(...others) : showVal(v), hi: others.length ? Math.max(...others) : showVal(v) });
    return e;
  }
  return meanPeriod(k, p);
}
function headDelta(k, p) {
  if (hasBlend()) { const pr = entry('blend').prev[0]; return pr ? (((pr.delta[k] || {}).periods || {})[p] || null) : null; }
  return meanDelta(k, p);
}
function scorecard() {
  const keys = Object.keys(S.latest.composites);
  const periods = S.latest.periods.filter((p) => p !== 'd11-15' && (p !== 'd16-32' || models().some((m) => Object.values(entry(m).periods).some((pp) => pp['d16-32']))));
  const rows = keys.map((k) => ({ key: k, label: k, cls: k === S.basin ? 'cons' : '' }));
  const cols = periods.map((p) => ({ key: p, label: PERIOD_LABEL[p] }));
  HEAT.table($('score'), rows, cols, (k, p) => {
    const v = headPeriod(k, p); if (!v) return null;
    const main = showVal(v); if (main === null) return null;
    const d = showDelta(headDelta(k, p));
    const fm = (x) => (S.heatwhat === 'anom' ? `${sgn(x, 0)} mm` : showFmt(x));
    // colour by the reading itself (vs normal), the change rides in small type
    const colv = S.heatwhat === 'pct' ? main - 100 : S.heatwhat === 'anom' ? main : main;
    return { v: colv, text: `${fm(main)} <small>${fm(v.lo)}–${fm(v.hi)}${d === null ? '' : ' · ' + showSgn(d)}</small>`,
             mark: v.n < v.want ? '<i class="sh">*</i>' : '',
             tip: `${k} · ${PERIOD_LABEL[p]} · ${hasBlend() ? 'blend' : 'mean'} of ${v.models} models\n${v.mm.toFixed(1)} mm${v.normal !== undefined ? ` / normal ${v.normal.toFixed(1)} = ${v.pct.toFixed(0)}% (${sgn(v.mm - v.normal, 1)} mm)` : ''}\nmodels range ${fm(v.lo)} to ${fm(v.hi)}` + (d === null ? '' : `\nsame-day change vs the previous issue: ${showSgn(d)}`) };
  }, [], { limb: S.heatwhat === 'mm' ? limbAbs : limb, scale: S.heatwhat === 'pct' ? 150 : 'auto', nice: showNice(), fmt: (v) => showFmt(v) });
  $('score').querySelectorAll('td.k').forEach((td, i) => { td.addEventListener('click', () => { S.basin = keys[i]; setBasin(S.basin); render(); });
    td.innerHTML += ` <a class="histlink" title="cumulative member plumes">▤</a>`;
    td.querySelector('.histlink').addEventListener('click', (ev) => { ev.stopPropagation(); openHist(keys[i], S.period); }); });
  const bw = S.latest.blend_weights || null; const bs = hasBlend() ? entry('blend').sources : null;
  $('subsc').textContent = `${hasBlend() ? 'blend' : 'mean'} of the models · ${SHOW[S.heatwhat]} · small type: model range, then the same-day change vs the previous issue`;
  $('scnote').innerHTML = (bs ? `<b>Blend</b> ${cyc(entry('blend').cycle)}: ` + Object.entries(bs).sort((a, b) => b[1].weight - a[1].weight).map(([m, x]) => `${MODEL_LABEL[m] || m} ${(x.weight * 100).toFixed(0)}%${x.cycle !== entry('blend').cycle ? ` (${cyc(x.cycle)})` : ''}`).join(', ')
    + ' — members pooled with weight w/n, so a 21-member ensemble does not outvote a 10-member one; prior weights until Stage IV verification accrues, then pulled halfway toward inverse-MAE. ' : '')
    + 'Composites are area-weighted unions of NWRFC divisions; Columbia above The Dalles is the whole basin above the dam, not the NWRFC mainstem reach. Click a row to open it.';
}

function obstab() {
  const o = S.latest.obs; const keys = Object.keys(S.latest.composites);
  const spans = [7, 14, 30, 60, 90].filter((n) => n <= o.dates.length);
  const rows = keys.map((k) => ({ key: k, label: k, cls: k === S.basin ? 'cons' : '' }));
  const cols = spans.map((n) => ({ key: String(n), label: `last ${n} d` }));
  HEAT.table($('obstab'), rows, cols, (k, n) => {
    const ob = (o.comp[k] || []).slice(-n), nm = (o.normal[k] || []).slice(-n);
    if (!ob.length || ob.some((x) => x === null || x === undefined)) return null;
    const mm = ob.reduce((a, b) => a + b, 0); const normal = nm.some((x) => x === null) ? undefined : nm.reduce((a, b) => a + b, 0);
    const v = { mm, normal, pct: normal ? 100 * mm / normal : undefined };
    const main = showVal(v); if (main === null) return null;
    return { v: S.heatwhat === 'pct' ? main - 100 : main, text: S.heatwhat === 'anom' ? `${sgn(main, 0)} mm` : showFmt(main),
             tip: `${k} · last ${n} days to ${o.dates[o.dates.length - 1]}\nStage IV ${mm.toFixed(1)} mm${normal ? ` / normal ${normal.toFixed(1)} = ${v.pct.toFixed(0)}% (${sgn(mm - normal, 1)} mm)` : ''}` };
  }, [], { limb: S.heatwhat === 'mm' ? limbAbs : limb, scale: S.heatwhat === 'pct' ? 150 : 'auto', nice: showNice(), fmt: (v) => showFmt(v) });
  $('subobs').textContent = `NCEP Stage IV through ${o.dates[o.dates.length - 1]} · ${SHOW[S.heatwhat]}`;
}

/* ---- map ---------------------------------------------------------------------------------------- */
const MAPWHAT = { model: 'model, % of normal', change: 'model, change vs prior issue', obs7: 'Stage IV last 7 d, % of normal',
                  obs14: 'Stage IV last 14 d, % of normal', obs30: 'Stage IV last 30 d, % of normal',
                  obs60: 'Stage IV last 60 d, % of normal', obs90: 'Stage IV last 90 d, % of normal' };
S.mapwhat = 'model';
let GEO = null;
// {mm, normal} -> the map's colour value and label under S.heatwhat
function mapCell(mm, normal, tipHead) {
  const pct = normal > 0 ? 100 * mm / normal : null;
  const tip = `${tipHead}${mm.toFixed(1)} mm` + (normal > 0 ? ` / normal ${normal.toFixed(1)} = ${pct.toFixed(0)}% (${sgn(mm - normal, 1)} mm)` : '');
  if (S.heatwhat === 'mm') return { v: mm, text: `${mm.toFixed(0)}`, tip };
  if (normal === null || normal === undefined || !(normal > 0)) return null;
  return S.heatwhat === 'pct' ? { v: pct - 100, text: `${pct.toFixed(0)}%`, tip } : { v: mm - normal, text: sgn(mm - normal), tip };
}
function mapValue(code) {
  const w = S.mapwhat;
  if (w === 'model') { const e = entry(S.model); const v = e && ((e.periods[code] || {})[S.period]); return v ? mapCell(v.mm, v.normal ?? null, '') : null; }
  if (w === 'change') { const e = entry(S.model); const pr = e && e.prev[0]; if (!pr) return null;
    const d = ((pr.delta[code] || {}).periods || {})[S.period]; const v = showDelta(d); if (v === null) return null;
    return { v, text: sgn(v, S.heatwhat === 'pct' ? 0 : 1), tip: `${sgn(d.mm, 1)} mm${d.pct !== undefined ? `, ${sgn(d.pct)} pts` : ''} vs ${cyc(pr.cycle)}` }; }
  const n = +w.slice(3); const o = S.latest.obs; const ob = (o.div[code] || []).slice(-n), nm = (o.normal[code] || []).slice(-n);
  if (!ob.length || ob.some((x) => x === null)) return null;
  const f = ob.reduce((a, b) => a + b, 0), g = nm.some((x) => x === null) ? null : nm.reduce((a, b) => a + b, 0);
  return mapCell(f, g, `${o.dates.slice(-n)[0]} → ${o.dates[o.dates.length - 1]}: `);
}
// Albers equal-area conic, standard parallels 43 and 50 N, central meridian
// 117 W: the projection the NWRFC and USGS draw the basin in, so the shapes
// look right instead of the squashed plate carrée.
const ALB = (() => {
  const d2r = Math.PI / 180, p1 = 43 * d2r, p2 = 50 * d2r, lam0 = -117 * d2r, phi0 = 47 * d2r;
  const n = (Math.sin(p1) + Math.sin(p2)) / 2, C = Math.cos(p1) ** 2 + 2 * n * Math.sin(p1);
  const rho = (phi) => Math.sqrt(C - 2 * n * Math.sin(phi)) / n, rho0 = rho(phi0);
  return (lon, lat) => { const th = n * (lon * d2r - lam0), r = rho(lat * d2r); return [r * Math.sin(th), rho0 - r * Math.cos(th)]; };
})();
let BASE = null;
function mapPanel() {
  const host = $('map'); if (!GEO) { host.innerHTML = '<div class="empty">Loading geometry…</div>'; return; }
  // frame: the divisions' extent with a margin, in projected units
  // frame hugs the divisions (a third of a degree of margin), so the box is not
  // mostly ocean and Alberta
  const corners = [[-124.9, 40.9], [-109.4, 40.9], [-124.9, 53.2], [-109.4, 53.2], [-117, 40.9], [-117, 53.2]].map((c) => ALB(c[0], c[1]));
  const px0 = Math.min(...corners.map((c) => c[0])), px1 = Math.max(...corners.map((c) => c[0]));
  const py0 = Math.min(...corners.map((c) => c[1])), py1 = Math.max(...corners.map((c) => c[1]));
  const W = 1400, K = (W - 4) / (px1 - px0), H = Math.round((py1 - py0) * K) + 4;   // 2x (user, 3 Sep 2026)
  const PX = (lon, lat) => { const p = ALB(lon, lat); return [2 + (p[0] - px0) * K, 2 + (py1 - p[1]) * K]; };
  const X = (lo) => PX(lo, 47)[0], Y = (la) => PX(-117, la)[1];        // only used for the label centroid fallback
  const pathOf = (geom) => {
    const polys = geom.type === 'Polygon' ? [geom.coordinates] : geom.type === 'MultiPolygon' ? geom.coordinates
      : geom.type === 'LineString' ? [[geom.coordinates]] : geom.type === 'MultiLineString' ? geom.coordinates.map((l) => [l]) : [];
    let d = '';
    for (const poly of polys) for (const ring of poly) d += ring.map((c, i) => { const p = PX(c[0], c[1]); return `${i ? 'L' : 'M'}${p[0].toFixed(1)} ${p[1].toFixed(1)}`; }).join('') + (geom.type.endsWith('Polygon') ? 'Z' : '');
    return d;
  };
  const baseLayer = (name) => (BASE ? BASE.features.filter((f) => f.properties.layer === name).map((f) => pathOf(f.geometry)).join('') : '');
  const vals = GEO.features.map((f) => mapValue(f.properties.code)).filter(Boolean).map((x) => x.v);
  const absMode = S.heatwhat === 'mm' && S.mapwhat !== 'change';
  const scale = absMode ? Math.max(1, ...vals) : S.heatwhat === 'pct' && S.mapwhat !== 'change' ? 150 : HEAT.autoScale(vals, showNice());
  const mlimb = absMode ? limbAbs : limb;
  const P = [`<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" class="basemap">`];
  // ocean, then land, then the coloured divisions, then water and borders on top
  P.push(`<rect x="0" y="0" width="${W}" height="${H}" fill="#cfe0ef"/>`);
  P.push(`<path d="${baseLayer('land')}" fill="#f4f1ea" stroke="none"/>`);
  const labels = [];
  for (const f of GEO.features) {
    const p = f.properties; const x = mapValue(p.code);
    const col = x ? HEAT.color(x.v, scale, mlimb) : { bg: '#e9e6df', ink: 'var(--ink3)' };
    const polys = f.geometry.type === 'Polygon' ? [f.geometry.coordinates] : f.geometry.coordinates;
    let best = null;
    for (const poly of polys) {
      const ring = poly[0]; let sx = 0, sy = 0; ring.forEach((c) => { sx += c[0]; sy += c[1]; });
      if (!best || ring.length > best.n) best = { n: ring.length, x: sx / ring.length, y: sy / ring.length };
    }
    const tip = `${p.name} (${p.code}) · ${p.region} · ${Math.round(p.area).toLocaleString()} sq mi${x ? '\n' + x.tip : '\nno value'}\nclick for the cumulative plumes`;
    P.push(`<path d="${pathOf(f.geometry)}" fill="${col.bg || '#e9e6df'}" class="div ${p.code === S.basin ? 'sel' : ''}" data-code="${p.code}" data-t="${tip}"/>`);
    if (best) { const q = PX(best.x, best.y); labels.push(`<text class="lab" x="${q[0].toFixed(1)}" y="${(q[1] + 3).toFixed(1)}" text-anchor="middle" fill="${x ? 'var(--ink)' : 'var(--ink3)'}">${x ? x.text : '–'}</text>`); }
  }
  P.push(`<path d="${baseLayer('lakes')}" fill="#bcd4ea" stroke="#7fa6cc" stroke-width=".5"/>`);
  P.push(`<path d="${baseLayer('rivers')}" fill="none" stroke="#6f9ccb" stroke-width=".9" opacity=".8"/>`);
  P.push(`<path d="${baseLayer('admin1')}" fill="none" stroke="#5a6b7d" stroke-width=".8" stroke-dasharray="3 2"/>`);
  P.push(`<path d="${baseLayer('admin0')}" fill="none" stroke="#2c3a4a" stroke-width="1.4"/>`);
  P.push(`<path d="${baseLayer('coast')}" fill="none" stroke="#3b5a78" stroke-width=".9"/>`);
  P.push(labels.join('')); P.push('</svg>');
  host.innerHTML = P.join('');
  host.querySelectorAll('path[data-code]').forEach((el) => el.addEventListener('click', () => openHist(el.dataset.code, S.period)));
  host.querySelector('svg').insertAdjacentHTML('beforeend', `<text x="${W - 8}" y="${H - 8}" text-anchor="end" fill="#5a6b7d" style="font-size:12px">Albers equal-area · Natural Earth 10 m · NWRFC water-supply divisions</text>`);
  HEAT.wireTips(host);
  const e = entry(S.model);
  const what = SHOW[S.heatwhat];
  $('submap').textContent = S.mapwhat === 'model' ? `${MODEL_LABEL[S.model] || S.model} ${e ? cyc(e.cycle) : ''} · ${PERIOD_LABEL[S.period]} · ${what} by division` :
    S.mapwhat === 'change' ? `${MODEL_LABEL[S.model] || S.model} · ${PERIOD_LABEL[S.period]} · same-day change vs the previous issue, ${S.heatwhat === 'pct' ? 'percentage points of normal' : 'mm'}` : `NCEP Stage IV · ${MAPWHAT[S.mapwhat].replace('% of normal', what)}`;
  $('submap').textContent += ' · click a division for its cumulative member plumes';
  const sw = (l, t) => `<i style="background:${HEAT.mix(HEAT.RAMP[l], t)}"></i>`;
  $('maplegend').innerHTML = absMode ? `<span class="lg">0 mm ${[.15, .33, .5, .66, .83, 1].map((t) => sw('green', t)).join('')}<b>${scale.toFixed(0)} mm</b></span>`
    : S.mapwhat === 'change' ? HEAT.legend(scale, limb, 'drier', 'wetter', (x) => x + showUnitDelta())
    : S.heatwhat === 'anom' ? HEAT.legend(scale, limb, 'below normal', 'above normal', (x) => x + ' mm')
    : `<span class="lg">${[1, .66, .33].map((t) => sw('brown', t)).join('')}<b>0%</b> &nbsp; 50% &nbsp; <b>100% of normal</b> &nbsp; 150% &nbsp; <b>≥250%</b>${[.33, .66, 1].map((t) => sw('green', t)).join('')}</span>`;
}

/* ---- member histogram pop-up ------------------------------------------------------------------------ */
S.histKey = null; S.histPeriod = null;
// member period totals for one model on one basin, in the page's reading
function memberTotals(m, k, p) {
  const e = entry(m); const st = e.series[k]; if (!st || !st.members) return null;
  const [a, b] = PERIOD_DAYS_JS[p]; const idx = [];
  for (let i = a - 1; i < Math.min(b, e.dates.length); i++) idx.push(i);
  if (!idx.length) return null;
  let normal = 0; for (const i of idx) { const n = normOn(k, e.dates[i]); if (n === null) { normal = null; break; } normal += n; }
  const vals = [], w = [];
  st.members.forEach((row, j) => {
    let t = 0, ok = true; for (const i of idx) { if (row[i] === null || row[i] === undefined) { ok = false; break; } t += row[i]; }
    if (!ok) return;
    const v = S.heatwhat === 'mm' ? t : normal === null || normal <= 0 ? null : S.heatwhat === 'pct' ? 100 * t / normal : t - normal;
    if (v === null) return;
    vals.push(v); w.push(st.weights ? st.weights[j] : 1);
  });
  return vals.length ? { vals, w, normal, n: idx.length, want: b - a + 1 } : null;
}
function detTotal(m, k, p) {
  const v = (entry(m).periods[k] || {})[p]; return v ? showVal(v) : null;
}
const PERIOD_DAYS_JS = { 'd1-5': [1, 5], 'd6-10': [6, 10], 'd11-15': [11, 15], 'd1-10': [1, 10], 'd1-15': [1, 15], 'd16-32': [16, 32] };
function wq(vals, w, q) {
  const o = vals.map((v, i) => [v, w[i]]).sort((x, y) => x[0] - y[0]); const tot = o.reduce((a, x) => a + x[1], 0);
  let c = 0; for (const [v, ww] of o) { c += ww; if (c >= q * tot) return v; } return o[o.length - 1][0];
}
function openHist(k, p) {
  S.histKey = k;
  const modal = $('hist'); modal.hidden = false;
  $('histname').textContent = label(k);
  const di = divInfo(k);
  $('histwhen').textContent = di ? `${di.region} · ${Math.round(di.area).toLocaleString()} sq mi` : 'area-weighted union of NWRFC divisions';
  const ens = models().filter((m) => entry(m).series[k] && entry(m).series[k].members);
  if (!ens.includes(S.histModel)) S.histModel = ens.includes(S.model) ? S.model : ens[0];
  seg('histperiod', ens.map((m) => ({ v: m, label: (MODEL_LABEL[m] || m).replace(' (Mon/Thu 00Z, 32 d)', ' 32 d'), color: MODEL_COLOR[m] || '#999' })), S.histModel, (m) => { S.histModel = m; renderHist(); });
  renderHist();
}
S.histModel = null;
function renderHist() {
  // Cumulative precipitation plumes for one basin: every member of the chosen
  // ensemble accumulated from day 1, the other models' means, the cumulative
  // normal, and the Stage IV run-up on the left. Big, with a legend that
  // toggles, end labels that never collide, and a table of run totals.
  const k = S.histKey, m = S.histModel; const host = $('histbody');
  const e = m && entry(m); const st = e && e.series[k];
  if (!st || !st.members) { host.innerHTML = '<div class="empty">No members for this basin.</div>'; $('histnote').textContent = ''; return; }
  const dates = e.dates; const nd = dates.length;
  const cum = (row) => { let t = 0; return row.map((v) => (v === null || v === undefined || t === null) ? (t = null) : (t += v)); };
  const mem = st.members.map(cum);
  const norm = []; let tn = 0; dates.forEach((d) => { const n = normOn(k, d); tn = n === null || tn === null ? null : tn + n; norm.push(tn); });
  const others = models().filter((x) => x !== m && x !== 'geps_ext' && entry(x).series[k]).map((x) => ({ x, c: cum(entry(x).series[k].mean), dates: entry(x).dates }));
  const o = S.latest.obs; const ob = (obsFor(k) || []).slice(-10), od = o.dates.slice(-10);
  const obsCum = []; let to = 0; ob.forEach((v) => { to = v === null || to === null ? null : to + v; obsCum.push(to); });
  const onorm = []; let tno = 0; od.forEach((d) => { const n = normOn(k, d); tno = n === null || tno === null ? null : tno + n; onorm.push(tno); });
  const W = 1300, H = 560, M = { l: 62, r: 200, t: 34, b: 44 };
  const nl = od.length, nx = nl + nd;
  const X = (i) => M.l + (i / Math.max(1, nx - 1)) * (W - M.l - M.r);
  const ymax = Math.max(5, ...mem.flat().filter((v) => v !== null), ...others.flatMap((z) => z.c).filter((v) => v !== null), ...norm.filter((v) => v !== null), ...obsCum.filter((v) => v !== null)) * 1.06;
  const Y = (v) => M.t + (H - M.t - M.b) * (1 - v / ymax);
  const col = MODEL_COLOR[m] || '#999';
  const P = [`<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" style="width:100%;height:auto">`];
  for (let i = 0; i <= 5; i++) { const v = ymax * i / 5; P.push(`<line x1="${M.l}" x2="${W - M.r}" y1="${Y(v).toFixed(1)}" y2="${Y(v).toFixed(1)}" stroke="#e6ebf0"/><text x="${M.l - 8}" y="${(Y(v) + 4).toFixed(1)}" text-anchor="end">${v.toFixed(0)}</text>`); }
  P.push(`<text x="${M.l - 8}" y="${M.t - 12}" text-anchor="end" fill="#55637a">mm</text>`);
  const xd = nl ? (X(nl - 1) + X(nl)) / 2 : M.l;
  if (nl) {
    P.push(`<rect x="${M.l}" y="${M.t}" width="${(xd - M.l).toFixed(1)}" height="${H - M.t - M.b}" fill="#f3f6f9"/>`);
    P.push(`<line x1="${xd.toFixed(1)}" x2="${xd.toFixed(1)}" y1="${M.t}" y2="${H - M.b}" stroke="#93a1b5" stroke-dasharray="4 3"/>`);
    P.push(`<text class="ttl" x="${(xd - 8).toFixed(1)}" y="${M.t - 12}" text-anchor="end" fill="#55637a">observed, last ${nl} days (Stage IV)</text><text class="ttl" x="${(xd + 8).toFixed(1)}" y="${M.t - 12}" fill="#55637a">forecast, accumulated from day 1 · ${MODEL_LABEL[m] || m} ${cyc(e.cycle)}</text>`);
  }
  let dO = '', dON = ''; obsCum.forEach((v, i) => { if (v !== null) dO += `${dO ? 'L' : 'M'}${X(i).toFixed(1)} ${Y(v).toFixed(1)}`; if (onorm[i] !== null) dON += `${dON ? 'L' : 'M'}${X(i).toFixed(1)} ${Y(onorm[i]).toFixed(1)}`; });
  P.push(`<path d="${dON}" fill="none" stroke="#0f172a" stroke-width="1.4" stroke-dasharray="6 4" opacity=".6" data-k="normal"/>`);
  P.push(`<path d="${dO}" fill="none" stroke="#4f84bd" stroke-width="3" data-k="obs"/>`);
  const op = mem.length > 30 ? 0.22 : mem.length > 15 ? 0.3 : 0.45;
  mem.forEach((row) => { let d = ''; row.forEach((v, i) => { if (v === null) return; d += `${d ? 'L' : 'M'}${X(nl + i).toFixed(1)} ${Y(v).toFixed(1)}`; });
    P.push(`<path d="${d}" fill="none" stroke="${col}" stroke-width="1" opacity="${op}" data-k="members"/>`); });
  let dN = ''; norm.forEach((v, i) => { if (v !== null) dN += `${dN ? 'L' : 'M'}${X(nl + i).toFixed(1)} ${Y(v).toFixed(1)}`; });
  P.push(`<path d="${dN}" fill="none" stroke="#0f172a" stroke-width="2" stroke-dasharray="7 4" data-k="normal"/>`);
  others.forEach((z) => { let d = ''; z.dates.forEach((dd, i) => { const j = dates.indexOf(dd); const v = z.c[i]; if (j < 0 || v === null) return; d += `${d ? 'L' : 'M'}${X(nl + j).toFixed(1)} ${Y(v).toFixed(1)}`; });
    P.push(`<path d="${d}" fill="none" stroke="${MODEL_COLOR[z.x] || '#999'}" stroke-width="${z.x === 'blend' ? 2.6 : 1.6}" opacity=".9" data-k="${z.x}"/>`); });
  const mc = cum(st.mean); let dM = ''; mc.forEach((v, i) => { if (v !== null) dM += `${dM ? 'L' : 'M'}${X(nl + i).toFixed(1)} ${Y(v).toFixed(1)}`; });
  P.push(`<path d="${dM}" fill="none" stroke="${col}" stroke-width="3.6" data-k="mean"/>`);
  // end labels in the right margin, pushed apart where they collide
  const ends = [];
  const lastOf = (arr, ds) => { const i = arr.map((v, j) => [v, j]).filter((x) => x[0] !== null).pop(); return i ? { v: i[0], j: dates.indexOf(ds[i[1]]) } : null; };
  const lm = lastOf(mc, dates); const nlast = norm[lm ? lm.j : nd - 1];
  if (lm) ends.push({ y: Y(lm.v), txt: `${MODEL_LABEL[m] || m} ${lm.v.toFixed(0)} mm${nlast ? ` · ${(100 * lm.v / nlast).toFixed(0)}%` : ''}`, color: col, bold: true, k: 'mean' });
  // a model that stops earlier is labelled against the normal to ITS last day,
  // and says so, otherwise a 10-day total reads as a percentage of 15 days
  others.forEach((z) => { const l = lastOf(z.c, z.dates); if (!l || l.j < 0) return; const nn = norm[l.j]; const shorter = l.j + 1 < nd;
    ends.push({ y: Y(l.v), txt: `${MODEL_LABEL[z.x] || z.x} ${l.v.toFixed(0)}${nn ? ` · ${(100 * l.v / nn).toFixed(0)}%${shorter ? ` of ${l.j + 1} d` : ''}` : ''}`, color: MODEL_COLOR[z.x] || '#999', k: z.x }); });
  if (nlast) ends.push({ y: Y(nlast), txt: `normal ${nlast.toFixed(0)} mm`, color: '#0f172a', k: 'normal' });
  ends.sort((p, q) => p.y - q.y);
  let prev = -1e9; ends.forEach((it) => { it.ly = Math.max(it.y, prev + 15); prev = it.ly; });
  const over = ends.length ? ends[ends.length - 1].ly - (H - M.b) : 0; if (over > 0) ends.forEach((it) => { it.ly -= over; });
  ends.forEach((it) => { const x0 = X(nx - 1) + 4, x1 = W - M.r + 14;
    P.push(`<line x1="${x0.toFixed(1)}" y1="${it.y.toFixed(1)}" x2="${(x1 - 4).toFixed(1)}" y2="${it.ly.toFixed(1)}" stroke="${it.color}" stroke-width=".8" opacity=".6" data-k="${it.k}"/>`);
    P.push(`<text x="${x1.toFixed(1)}" y="${(it.ly + 4).toFixed(1)}" fill="${it.color}" style="font:${it.bold ? 700 : 500} 12px var(--mono)" data-k="${it.k}">${it.txt}</text>`); });
  const allDates = od.concat(dates);
  allDates.forEach((d, i) => { if (allDates.length > 20 && i % 2) return; const dt = new Date(d + 'T00:00:00');
    P.push(`<text x="${X(i).toFixed(1)}" y="${H - 10}" text-anchor="middle" fill="${dt.getDay() % 6 === 0 ? '#b3392b' : '#55637a'}">${dt.getMonth() + 1}/${dt.getDate()}</text>`); });
  // hover columns
  const step = (W - M.l - M.r) / Math.max(1, nx - 1);
  allDates.forEach((d, i) => { const lines = [d]; if (i < nl) { if (obsCum[i] !== null) lines.push(`Stage IV to date  ${obsCum[i].toFixed(1)} mm`); if (onorm[i] !== null) lines.push(`normal to date  ${onorm[i].toFixed(1)} mm`); }
    else { const j = i - nl; if (norm[j] !== null) lines.push(`normal from day 1  ${norm[j].toFixed(1)} mm`); if (mc[j] !== null) lines.push(`${MODEL_LABEL[m] || m} mean  ${mc[j].toFixed(1)} mm`);
      const col_ = mem.map((r) => r[j]).filter((v) => v !== null); if (col_.length) { const w = st.weights || col_.map(() => 1); lines.push(`members p10 ${wq(col_, w.slice(0, col_.length), .1).toFixed(0)} · p50 ${wq(col_, w.slice(0, col_.length), .5).toFixed(0)} · p90 ${wq(col_, w.slice(0, col_.length), .9).toFixed(0)} mm`); }
      others.forEach((z) => { const jj = z.dates.indexOf(dates[j]); if (jj >= 0 && z.c[jj] !== null) lines.push(`${MODEL_LABEL[z.x] || z.x}  ${z.c[jj].toFixed(1)} mm`); }); }
    P.push(`<rect class="hit" x="${(X(i) - step / 2).toFixed(1)}" y="${M.t}" width="${step.toFixed(1)}" height="${H - M.t - M.b}" fill="transparent" data-t="${lines.join('\n')}"/>`); });
  P.push('</svg>');
  const chips = [`<span class="new" data-k="mean"><i style="border-color:${col};border-top-width:3px"></i>${MODEL_LABEL[m] || m} mean</span>`,
    `<span data-k="members"><i style="border-color:${col};border-top-width:1px;opacity:.6"></i>${mem.length} members</span>`]
    .concat(others.map((z) => `<span data-k="${z.x}"><i style="border-color:${MODEL_COLOR[z.x] || '#999'}"></i>${MODEL_LABEL[z.x] || z.x} mean</span>`))
    .concat([`<span data-k="normal"><i style="border-color:#0f172a;border-top-style:dashed"></i>1991–2020 normal</span>`, `<span data-k="obs"><i style="border-color:#4f84bd;border-top-width:3px"></i>Stage IV observed</span>`]).join('');
  // run totals table
  const endv = mem.map((row) => row.filter((v) => v !== null).pop()).filter((v) => v !== undefined && v !== null);
  const w = (st.weights || endv.map(() => 1)).slice(0, endv.length); const tot = w.reduce((a, b) => a + b, 0);
  const q = (qq) => wq(endv, w, qq); const above = nlast ? endv.reduce((a, v, i) => a + (v > nlast ? w[i] : 0), 0) / tot : null;
  const pct = (v) => (nlast ? `${(100 * v / nlast).toFixed(0)}%` : '–');
  const rows = [[`${MODEL_LABEL[m] || m} members`, `p10 ${q(0.1).toFixed(0)} mm (${pct(q(0.1))})`, `median ${q(0.5).toFixed(0)} mm (${pct(q(0.5))})`, `p90 ${q(0.9).toFixed(0)} mm (${pct(q(0.9))})`, above !== null ? `${(above * 100).toFixed(0)}% of members above normal` : '']];
  const tbl = `<table class="heat histtab"><thead><tr><th class="k">${nd}-day total</th><th>low (p10)</th><th>middle (median)</th><th>high (p90)</th><th></th></tr></thead><tbody>${rows.map((r) => `<tr><td class="k">${r[0]}</td>${r.slice(1).map((c) => `<td class="x">${c}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
  host.innerHTML = `<div class="evolegend">${chips}</div>` + P.join('') + tbl;
  HEAT.wireTips(host); HEAT.wireToggles(host, 'histplume');
  $('histnote').textContent = `Left of the divider: the last ${nl} days of Stage IV, accumulated, against the same normal. Right: every ${MODEL_LABEL[m] || m} member accumulated from day 1, its mean (bold), the other models' means, and the cumulative 1991–2020 normal (dashed). End labels give each run's total and its percent of normal. Click a legend label to hide it; hover a day for the numbers.`;
}
function wireHist() {
  $('histclose').addEventListener('click', () => { $('hist').hidden = true; });
  $('histback').addEventListener('click', () => { $('hist').hidden = true; });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') $('hist').hidden = true; });
  $('histopen').addEventListener('click', () => { S.basin = S.histKey; setBasin(S.basin); $('hist').hidden = true; render(); });
}

/* ---- by period ----------------------------------------------------------------------------- */
function ptab() {
  const k = S.basin; const ms = models().filter((m) => entry(m).periods[k]);
  const cols = S.latest.periods.map((p) => ({ key: p, label: PERIOD_LABEL[p] }));
  const rows = (ms.length > 1 && !hasBlend() ? [{ key: '__mean', label: 'Mean of models', cls: 'blend' }] : []).concat(ms.map((m) => ({ key: m, label: MODEL_LABEL[m] || m, cls: m === 'blend' ? 'blend' : '' })));
  HEAT.table($('ptab'), rows, cols, (m, p) => {
    if (m === '__mean') {
      const v = meanPeriod(k, p); if (!v) return null; const main = showVal(v); if (main === null) return null;
      const d = showDelta(meanDelta(k, p));
      const other = S.heatwhat === 'pct' ? `${v.mm.toFixed(0)} mm` : (v.pct !== undefined ? `${v.pct.toFixed(0)}%` : '');
      return { v: d === null ? 0 : d, text: `${S.heatwhat === 'anom' ? sgn(main, 0) + ' mm' : showFmt(main)} <small>${other} ${d === null ? '' : showSgn(d)}</small>`,
               mark: v.n < v.want ? '<i class="sh">*</i>' : '', tip: `mean of ${v.models} models · ${PERIOD_LABEL[p]}\n${v.mm.toFixed(1)} mm${v.normal !== undefined ? ` / normal ${v.normal.toFixed(1)} = ${v.pct.toFixed(0)}%` : ''}` };
    }
    const e = entry(m); const v = (e.periods[k] || {})[p]; if (!v) return null;
    const pr = e.prev[0]; const dd = pr ? (((pr.delta[k] || {}).periods || {})[p] || null) : null;
    const main = showVal(v); if (main === null) return null;
    const d = showDelta(dd);
    const other = S.heatwhat === 'pct' ? `${v.mm.toFixed(0)} mm` : (v.pct !== undefined ? `${v.pct.toFixed(0)}%` : '');
    const txt = `${S.heatwhat === 'anom' ? sgn(main, 0) + ' mm' : showFmt(main)} <small>${other} ${d === null ? '' : showSgn(d)}</small>`;
    return { v: d === null ? 0 : d, text: txt, mark: v.n < v.want ? '<i class="sh">*</i>' : '',
             tip: `${MODEL_LABEL[m] || m} ${cyc(e.cycle)} · ${label(k)} · ${PERIOD_LABEL[p]}\n${v.mm.toFixed(1)} mm` + (v.normal ? ` / normal ${v.normal.toFixed(1)} = ${v.pct.toFixed(0)}%` : '')
               + (pr && dd ? `\nsame-day change vs ${cyc(pr.cycle)}: ${sgn(dd.mm, 1)} mm${dd.pct !== undefined ? `, ${sgn(dd.pct)} pts` : ''}` : '') };
  }, [{ label: 'issue', value: (m) => (m === '__mean' ? { text: '', cls: 'tot' } : { text: cyc(entry(m).cycle) + (S.pick[m] ? ' ▸' : ''), cls: 'tot' }) }], { limb, fmt: (v) => sgn(v), nice: showNice() });
  $('sub2').innerHTML = `${label(k)} · ${SHOW[S.heatwhat]}, and in small type the same-day change vs the model's previous issue`;
}

/* ---- small multiples ---------------------------------------------------------------------- */
function small() {
  const host = $('small'); host.innerHTML = '';
  const ms = models();
  for (const k of Object.keys(S.latest.composites)) {
    const odates = S.latest.obs.dates.slice(-20); const obs = (obsFor(k) || []).slice(-20);
    const ser = ms.filter((m) => entry(m).series[k]).map((m) => ({ m, e: entry(m), s: entry(m).series[k] }));
    const fdates = [...new Set(ser.flatMap((x) => x.e.dates))].sort();
    const dates = [...new Set(odates.concat(fdates))].sort();
    const W = 420, H = 190, M = { l: 32, r: 8, t: 34, b: 18 };
    const X = (d) => M.l + dates.indexOf(d) * (W - M.l - M.r) / Math.max(1, dates.length - 1);
    const step = (W - M.l - M.r) / Math.max(1, dates.length - 1);
    const vals = obs.filter((v) => v !== null).concat(ser.flatMap((x) => x.s.mean).filter((v) => v !== null));
    const y1 = Math.max(2, ...vals) * 1.08; const Y = (v) => M.t + (H - M.t - M.b) * (1 - v / y1);
    const P = [`<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">`];
    P.push(`<text x="${M.l - 4}" y="${(Y(y1 / 2) + 3).toFixed(1)}" text-anchor="end">${(y1 / 2).toFixed(0)}</text>`);
    odates.forEach((d, i) => { const v = obs[i]; if (v === null || v === undefined) return;
      P.push(`<rect x="${(X(d) - step * 0.36).toFixed(1)}" y="${Y(v).toFixed(1)}" width="${(step * 0.72).toFixed(1)}" height="${Math.max(0.5, Y(0) - Y(v)).toFixed(1)}" fill="#7ea8d5"/>`); });
    let nd = ''; dates.forEach((d) => { const n = normOn(k, d); if (n === null) return; nd += `${nd ? 'L' : 'M'}${X(d).toFixed(1)} ${Y(n).toFixed(1)}`; });
    P.push(`<path d="${nd}" fill="none" stroke="#0f172a" stroke-width="1" stroke-dasharray="4 3" opacity=".7"/>`);
    ser.forEach((x) => { let d = ''; x.e.dates.forEach((dd, i) => { if (x.s.mean[i] === null) return; d += `${d ? 'L' : 'M'}${X(dd).toFixed(1)} ${Y(x.s.mean[i]).toFixed(1)}`; });
      P.push(`<path d="${d}" fill="none" stroke="${MODEL_COLOR[x.m] || '#999'}" stroke-width="${x.m === 'blend' ? 2.4 : x.m === S.model ? 2 : 1.2}" stroke-linejoin="round"/>`); });
    // headline: chosen period % of normal per model, mean of models
    const pcts = ser.map((x) => showVal((x.e.periods[k] || {})[S.period])).filter((v) => v !== null && v !== undefined);
    const fm = (v) => (S.heatwhat === 'pct' ? `${Math.round(v)}%` : S.heatwhat === 'anom' ? `${sgn(v, 0)} mm` : `${Math.round(v)} mm`);
    const head = pcts.length ? `${PERIOD_LABEL[S.period]} ${fm(pcts.reduce((a, b) => a + b, 0) / pcts.length)}${S.heatwhat === 'pct' ? ' of normal' : S.heatwhat === 'anom' ? ' vs normal' : ''} (models ${fm(Math.min(...pcts))} to ${fm(Math.max(...pcts))})` : '';
    P.push(`<text class="ttl" x="${M.l}" y="12">${k}</text>`);
    P.push(`<text x="${M.l}" y="25" fill="#55637a">${head}</text>`);
    [dates[0], dates[Math.floor(dates.length / 2)], dates[dates.length - 1]].forEach((d) => { const dt = new Date(d + 'T00:00:00');
      P.push(`<text x="${X(d).toFixed(1)}" y="${H - 3}" text-anchor="middle">${dt.getMonth() + 1}/${dt.getDate()}</text>`); });
    P.push('</svg>');
    const div = document.createElement('div'); div.className = 'smallcard'; div.innerHTML = P.join('');
    div.addEventListener('click', () => { S.basin = k; setBasin(k); render(); window.scrollTo({ top: 0, behavior: 'smooth' }); });
    host.appendChild(div);
  }
  $('sub5').textContent = `last 20 days of Stage IV, normal dashed, each model's newest run · click a card to open it`;
}

/* ---- run over run ---------------------------------------------------------------------------- */
function issuesOf(m) { return (S.hist.models || {})[m] || []; }
function board3() {
  const k = S.basin; const ms = models();
  const cycles = [...new Set(ms.flatMap((m) => issuesOf(m).map((x) => x.cycle)))].sort().slice(-S.nruns);
  const rows = ms.map((m) => ({ key: m, label: MODEL_LABEL[m] || m }));
  const cols = cycles.map((c) => ({ key: c, label: `${c.slice(8, 10)}Z`, sub: cycDay(c) }));
  const ix = S.heatwhat === 'pct' ? 1 : 0;              // history rows: [mm, pct]; an anomaly change is an mm change
  const s = HEAT.table($('h3'), rows, cols, (m, c) => {
    const all = issuesOf(m); const i = all.findIndex((x) => x.cycle === c); if (i < 0) return null;
    const d = ((all[i].dprev || {})[k] || {})[S.period]; const w = ((all[i].w || {})[k] || {})[S.period];
    const v = d ? d[ix] : null;
    return { v: v === undefined ? null : v, mark: w && w[2] ? '<i class="sh">*</i>' : '',
             tip: `${MODEL_LABEL[m] || m} ${cyc(c)} · ${label(k)} · ${PERIOD_LABEL[S.period]}\n${w ? `${w[0].toFixed(1)} mm${w[1] !== null && w[1] !== undefined ? ` = ${w[1].toFixed(0)}% of normal` : ''}` : ''}`
               + (i > 0 && d ? `\nsame-day change vs ${cyc(all[i - 1].cycle)}: ${sgn(d[0], 1)} mm${d[1] !== null && d[1] !== undefined ? `, ${sgn(d[1])} pts` : ''}` : '') };
  }, [{ label: 'latest', value: (m) => { const all = issuesOf(m); if (!all.length) return null; const w = ((all[all.length - 1].w || {})[k] || {})[S.period];
        if (!w) return null; const pv = (entry(m).periods[k] || {})[S.period]; const sv = showVal(pv);
        return { text: sv === null ? `${w[0].toFixed(0)} mm` : S.heatwhat === 'anom' ? `${sgn(sv, 0)} mm` : showFmt(sv), cls: 'tot' }; } }],
    { limb, fmt: (v) => sgn(v, ix ? 0 : 1), nice: showNice() });
  $('sub3').innerHTML = `${label(k)} · ${PERIOD_LABEL[S.period]} · each model's same-day change vs its own previous issue, ${ix ? 'percentage points of normal' : 'mm'} · `
    + HEAT.legend(s, limb, 'drier', 'wetter', (x) => x + showUnitDelta());
}

function board4() {
  const ms = models();
  const keys = Object.keys(S.latest.composites).concat(S.latest.divisions.filter((d) => d.normals).map((d) => d.code));
  const rows = keys.map((k) => ({ key: k, label: label(k), cls: isComp(k) ? 'cons' : '' }));
  const cols = ms.map((m) => ({ key: m, label: MODEL_LABEL[m] || m }));
  const ix = S.heatwhat === 'pct' ? 'pct' : 'mm';
  const s = HEAT.table($('h4'), rows, cols, (k, m) => {
    const e = entry(m); const pr = e.prev[0]; const cur = (e.periods[k] || {})[S.period]; if (!cur) return null;
    const dd = pr ? (((pr.delta[k] || {}).periods || {})[S.period] || {}) : {};
    const v = dd[ix]; if (v === undefined) return { v: null, tip: `${MODEL_LABEL[m] || m}: no earlier issue` };
    return { v, tip: `${MODEL_LABEL[m] || m} ${cyc(e.cycle)} vs ${cyc(pr.cycle)} · ${label(k)} · ${PERIOD_LABEL[S.period]}\nnow ${cur.mm.toFixed(1)} mm${cur.pct !== undefined ? ` = ${cur.pct.toFixed(0)}% (${sgn(cur.mm - cur.normal, 1)} mm)` : ''}; change ${sgn(dd.mm, 1)} mm${dd.pct !== undefined ? `, ${sgn(dd.pct)} pts` : ''}` };
  }, [{ label: 'now (mean of models)', value: (k) => { const p = ms.map((m) => showVal((entry(m).periods[k] || {})[S.period])).filter((v) => v !== null && v !== undefined);
        const mv = p.length ? p.reduce((a, b) => a + b, 0) / p.length : null;
        return mv === null ? null : { text: S.heatwhat === 'anom' ? `${sgn(mv, 0)} mm` : showFmt(mv), cls: 'tot' }; } }],
    { limb, fmt: (v) => sgn(v, ix === 'pct' ? 0 : 1), nice: showNice() });
  $('sub4').innerHTML = `${PERIOD_LABEL[S.period]} · newest issue vs the one before, ${ix === 'pct' ? 'percentage points of normal' : 'mm'}; last column ${SHOW[S.heatwhat]} · ` + HEAT.legend(s, limb, 'drier', 'wetter', (x) => x + showUnitDelta());
}

function verif() {
  const v = S.latest.verif || {}; const ms = models().filter((m) => v[m] && Object.keys(v[m]).length);
  if (!ms.length) { $('verif').innerHTML = '<tbody><tr><td class="empty">No forecast day has a Stage IV observation yet — pairs accrue from tomorrow’s 12Z analysis on.</td></tr></tbody>'; $('sub6').textContent = ''; return; }
  const leads = [...new Set(ms.flatMap((m) => Object.keys(v[m])))].map(Number).sort((a, b) => a - b);
  const rows = ms.map((m) => ({ key: m, label: MODEL_LABEL[m] || m }));
  const cols = leads.map((l) => ({ key: String(l), label: `day ${l}` }));
  HEAT.table($('verif'), rows, cols, (m, l) => { const x = v[m][l]; if (!x) return null;
    const r = x.ratio === null ? null : (x.ratio - 1) * 100;
    return { v: r, text: `${x.ratio === null ? '–' : x.ratio.toFixed(2)} <small>MAE ${x.mae.toFixed(1)} · n ${x.n}</small>`, tip: `${MODEL_LABEL[m] || m} day ${l}: forecast/observed ${x.ratio}, MAE ${x.mae} mm, ${x.n} composite-days` }; },
    [], { limb, scale: 50, fmt: (x) => x.toFixed(0) });
  $('sub6').textContent = 'forecast ÷ Stage IV over the six composites, pooled by lead day; ratio, mean absolute error (mm/day), pairs';
}

function render() {
  scorecard(); obstab(); plume(); mapPanel(); ptab(); small(); board3(); board4(); verif();
  const ms = models(); const o = S.latest.obs.dates;
  const picked = Object.keys(S.pick);
  $('runbadge').textContent = `${ms.length} models · Stage IV through ${o[o.length - 1] || '–'}` + (picked.length ? ` · viewing archived ${picked.map((m) => `${MODEL_LABEL[m] || m} ${cyc(S.pick[m].cycle)}`).join(', ')}` : '');
  document.body.classList.toggle('histrun', picked.length > 0);
  const newest = ms.map((m) => entry(m).cycle).sort().slice(-1)[0];
  const cd = headPeriod('Columbia abv The Dalles', 'd1-10');
  const shortLabel = (m) => (m === 'geps_ext' ? 'the GEPS Mon/Thu extension to 32 days' : MODEL_LABEL[m] || m);
  // The one number worth seeing without reading anything stays out in the
  // open; the descriptive prose lives in a <details> beside it. On a phone
  // those two paragraphs were 518 px of the 1,249 px that stood between the
  // top of the page and the map.
  $('introhead').innerHTML = (cd && cd.pct !== undefined)
    ? `${hasBlend() ? 'Blend' : 'Mean of models'} ${cyc(hasBlend() ? entry('blend').cycle : newest)}: Columbia above The Dalles `
      + `<b>${cd.pct.toFixed(0)}% of normal</b> over days 1–10 · ${cd.mm.toFixed(0)} mm against a normal ${cd.normal.toFixed(0)}`
      + (S.heatwhat === 'pct' ? ` · models ${cd.lo.toFixed(0)}–${cd.hi.toFixed(0)}%` : '')
    : '';
  $('intro').innerHTML = `Precipitation forecasts for the Columbia River basin from ${ms.filter((m) => m !== 'blend').map(shortLabel).join(', ')}${hasBlend() ? ', and a weighted blend' : ''}, `
    + `averaged over the Northwest River Forecast Center's water-supply divisions and read against the 1991–2020 normal and against NCEP Stage IV observed rainfall. `
    + `Every table can be read as percent of normal, as the departure in mm, or as absolute mm.`;
  // What the collapsed control bar says it is set to, so the state is legible
  // without opening it. The MODEL leads and carries its own colour swatch --
  // it is the setting that changes what the map is showing, and in a plain
  // dot-separated list it read as just another word.
  const unitLab = ({ pct: '% of normal', anom: 'mm vs normal', mm: 'mm' })[S.heatwhat];
  $('ctlnow').innerHTML =
    `<i class="dot" style="background:${MODEL_COLOR[S.model] || '#999'}"></i>`
    + `<b>${MODEL_LABEL[S.model] || S.model}</b>`
    + `<span class="ctlrest">`
    + `<span class="sep">·</span>${S.basin.replace('Columbia abv ', 'Columbia ↑ ').replace('Snake abv ', 'Snake ↑ ')}`
    + `<span class="sep">·</span>${PERIOD_SHORT[S.period] || S.period}`
    + `<span class="sep">·</span>${unitLab}`
    + `</span>`;
  $('credits').innerHTML = '<b>Sources and licences.</b> Basins and 1991–2020 mean-areal-precipitation normals: NOAA/NWS Northwest River Forecast Center (public domain). '
    + 'Observed rainfall: NCEP Stage IV multi-sensor precipitation analysis (NOAA, public domain), served by NOMADS and by the Iowa Environmental Mesonet archive. '
    + 'GFS and GEFS: NOAA/NCEP (public domain), from the NOAA Open Data Dissemination buckets on AWS. '
    + 'IFS, AIFS and ENS: ECMWF open data, © ECMWF, licensed under <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>; basin averages are derived by this site and are not an official ECMWF product. '
    + 'GDPS and GEPS: Data Source: Environment and Climate Change Canada (MSC Datamart), used under the ECCC data servers end-use licence; ECCC does not endorse this site. '
    + 'Basemap: Natural Earth (public domain). Model output is shown unadjusted; nothing here is an official forecast. Built by Shawn Corvec.';
  $('foot').innerHTML = 'Divisions and 1991–2020 mean-areal-precipitation normals are the NWRFC water-supply divisions (42, dissolved from the 379 NWRFC forecast basins); '
    + 'the composites are area-weighted unions — the NWRFC mainstem groups are LOCAL reach areas, so "above The Dalles" here is the union of every Columbia, Snake, Middle and Upper Columbia division, not the mainstem row on the NWRFC page. '
    + 'Days are 12Z–12Z, labelled by the ending date, matching the Stage IV 24 h product; a 00Z run\'s first day is its hours 12–36. '
    + 'Percent of normal = period precipitation ÷ the sum of daily normals over the same dates (a month\'s normal spread evenly). '
    + 'Model precipitation is the model\'s own grid, area-averaged on a 0.02° mesh; GEFS members through the 0.5° tail past day 10; ECMWF ENS is a 10-member subset. '
    + 'Observed: NCEP Stage IV 24 h (12Z) division means from the 4 km grid, kept from NOMADS as they publish (about two weeks retained upstream). '
    + '<b>*</b> run stops short of the window. Run-over-run cells are same-calendar-day changes. '
    + `pnw_history.json built ${S.hist.built || ''}.`;
}

const PERIOD_SHORT = { 'd1-5': 'days 1–5', 'd6-10': '6–10', 'd11-15': '11–15', 'd1-10': '1–10', 'd1-15': '1–15', 'd16-32': '16–32 ext.' };
const MAP_SHORT = { model: 'model', change: 'change vs prior', obs7: 'observed 7 d', obs14: '14 d', obs30: '30 d', obs60: '60 d', obs90: '90 d' };
function setBasin(k) {
  setSeg('basin', k);
  const dsel = $('basindiv'); if (dsel) dsel.value = isComp(k) ? '' : k;
}
function controls() {
  const comps = Object.keys(S.latest.composites);
  seg('basin', comps.map((k) => ({ v: k, label: k.replace('Columbia abv ', 'Columbia ↑ ').replace('Snake abv ', 'Snake ↑ ') })), S.basin, (k) => { S.basin = k; setBasin(k); render(); });
  const dsel = $('basindiv'); dsel.innerHTML = '<option value="">…or a division</option>';
  const byReg = {}; for (const d of S.latest.divisions) (byReg[d.region] = byReg[d.region] || []).push(d);
  for (const r of Object.keys(byReg).sort()) { const g = document.createElement('optgroup'); g.label = r;
    for (const d of byReg[r].sort((a, b) => a.name.localeCompare(b.name))) { const o = document.createElement('option'); o.value = d.code; o.textContent = d.name; g.appendChild(o); }
    dsel.appendChild(g); }
  dsel.addEventListener('change', (e) => { if (!e.target.value) return; S.basin = e.target.value; setBasin(S.basin); render(); });
  const ms = models(); if (!ms.includes(S.model)) S.model = ms[0];
  seg('model', ms.map((m) => ({ v: m, label: (MODEL_LABEL[m] || m).replace(' (Mon/Thu 00Z, 32 d)', ''), color: MODEL_COLOR[m] || '#999' })), S.model, (m) => { S.model = m; fillRuns(); render(); });
  seg('period', S.latest.periods, S.period, (p) => { S.period = p; render(); }, (p) => PERIOD_SHORT[p] || p);
  seg('heatwhat', Object.keys(SHOW), S.heatwhat, (k) => { S.heatwhat = k; render(); }, (k) => ({ pct: '% of normal', anom: 'mm vs normal', mm: 'mm' })[k]);
  seg('nruns', ['8', '12', '20', '30'], String(S.nruns), (n) => { S.nruns = +n; render(); });
  seg('mapwhat', Object.keys(MAPWHAT), S.mapwhat, (k) => { S.mapwhat = k; render(); }, (k) => MAP_SHORT[k]);
  fillRuns();
  setBasin(S.basin);
}

const DATA = DATA_BASE();
async function load() {
  S.latest = await (await fetch(DATA + 'pnw_latest.json?t=' + Date.now())).json();
  S.hist = await (await fetch(DATA + 'pnw_history.json?t=' + Date.now())).json();
  if (!GEO) { try { GEO = await (await fetch('pnw_divisions.geojson')).json(); } catch (e) { GEO = null; } }
  if (!BASE) { try { BASE = await (await fetch('pnw_base.geojson')).json(); } catch (e) { BASE = null; } }
}
// The control panel ships open, which is what a desktop wants and what a
// browser with JS off should get. On a phone it is 517 px of buttons above
// the fold, so it starts collapsed there -- the summary bar stays pinned, so
// the controls are one tap away from anywhere on the page rather than a
// scroll back to the top.
const NARROW = '(max-width: 700px)';
function fitControls() {
  const box = document.getElementById('ctlbox');
  if (box) box.open = !window.matchMedia(NARROW).matches;
}

(async () => {
  fitControls();
  window.matchMedia(NARROW).addEventListener('change', fitControls);
  await load(); controls(); wireHist(); render();
  setInterval(async () => {
    try { const l = await (await fetch(DATA + 'pnw_latest.json?t=' + Date.now())).json();
      if (l.built !== S.latest.built) { await load(); render(); } } catch (e) { /* keep */ }
  }, 60000);
})();
