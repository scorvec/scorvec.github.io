/* Columbia basin precipitation page. Reads pnw_latest.json (divisions,
 * composites, Stage IV ledger, newest issue per model with deltas) and
 * pnw_history.json (period mm / % of normal per issue, same-day dprev). */
const S = { latest: null, hist: null, basin: 'Columbia abv The Dalles', model: 'gefs',
            period: 'd1-10', heatwhat: 'pct', nruns: 12 };
const $ = (id) => document.getElementById(id);
const cyc = (c) => `${c.slice(4, 6)}/${c.slice(6, 8)} ${c.slice(8, 10)}Z`;
const cycDay = (c) => `${c.slice(4, 6)}/${c.slice(6, 8)}`;
const MODEL_LABEL = { gfs: 'GFS', gefs: 'GEFS', ecmwf: 'ECMWF IFS', ecmwf_ens: 'ECMWF ENS',
                      aifs: 'AIFS', gdps: 'GDPS', geps: 'GEPS', geps_ext: 'GEPS ext. (Mon/Thu 00Z, 32 d)',
                      wn2: 'WeatherNext-2', wn3: 'WeatherNext-3' };
const MODEL_ORDER = ['gefs', 'ecmwf_ens', 'geps', 'gfs', 'ecmwf', 'aifs', 'gdps', 'geps_ext', 'wn2', 'wn3'];
const MODEL_COLOR = { gfs: '#2b6cb0', gefs: '#6fa8dc', ecmwf: '#b3372a', ecmwf_ens: '#e0745f',
                      aifs: '#2f855a', gdps: '#b8860b', geps: '#d9b45b', geps_ext: '#8a6d1f',
                      wn2: '#8a5cc0', wn3: '#5a3f98' };
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
const entry = (m) => S.latest.models[m];
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
  const ms = models().filter((m) => entry(m).series[k]);
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
    P.push(`<rect x="${(X(d) - step * 0.36).toFixed(1)}" y="${Y(v).toFixed(1)}" width="${(step * 0.72).toFixed(1)}" height="${Math.max(0.5, Y(0) - Y(v)).toFixed(1)}" fill="#7ea8d5"/>`);
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
  P.push(`<path d="${nd}" fill="none" stroke="#0f172a" stroke-width="1.2" stroke-dasharray="5 3" opacity=".7"/>`);
  // band for chosen model
  const b = ser.find((x) => x.m === S.model);
  if (b) {
    let up = '', dn = '';
    b.e.dates.forEach((d, i) => { if (b.s.p10[i] === null || b.s.p90[i] === null) return;
      up += `${up ? 'L' : 'M'}${X(d).toFixed(1)} ${Y(b.s.p90[i]).toFixed(1)}`; dn = `L${X(d).toFixed(1)} ${Y(b.s.p10[i]).toFixed(1)}` + dn; });
    if (up) P.push(`<path d="${up}${dn}Z" fill="${MODEL_COLOR[b.m] || '#999'}" opacity=".16"/>`);
  }
  ser.forEach((x) => {
    let d = '';
    x.e.dates.forEach((dd, i) => { if (x.s.mean[i] === null) return; d += `${d ? 'L' : 'M'}${X(dd).toFixed(1)} ${Y(x.s.mean[i]).toFixed(1)}`; });
    P.push(`<path d="${d}" fill="none" stroke="${MODEL_COLOR[x.m] || '#999'}" stroke-width="${x.m === S.model ? 2.6 : 1.5}" stroke-linejoin="round"/>`);
  });
  const mn = meanSeries(k);
  if (mn && ser.length > 1) {
    let d = ''; mn.dates.forEach((dd, i) => { if (mn.mean[i] === null) return; d += `${d ? 'L' : 'M'}${X(dd).toFixed(1)} ${Y(mn.mean[i]).toFixed(1)}`; });
    P.push(`<path d="${d}" fill="none" stroke="#0f172a" stroke-width="3" stroke-linejoin="round" opacity=".85"/>`);
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
  P.push(`<path d="${od}" fill="none" stroke="#4f84bd" stroke-width="2"/>`);
  const lused = [];
  cum.forEach((c) => { let d = ''; c.dates.forEach((dd, i) => { if (c.pct[i] === null) return; d += `${d ? 'L' : 'M'}${X(dd).toFixed(1)} ${Y2(c.pct[i]).toFixed(1)}`; });
    P.push(`<path d="${d}" fill="none" stroke="${MODEL_COLOR[c.m] || '#999'}" stroke-width="${c.m === S.model ? 2.4 : 1.4}"/>`);
    const last = c.pct.map((v, i) => [v, i]).filter((x) => x[0] !== null).pop();
    if (last) {
      // end labels pushed apart where runs converge
      let y = Y2(last[0]);
      while (lused.some((u) => Math.abs(u - y) < 10)) y += 10;
      lused.push(y);
      P.push(`<text class="chg" x="${(X(c.dates[last[1]]) + 4).toFixed(1)}" y="${(y + 3).toFixed(1)}" fill="${MODEL_COLOR[c.m] || '#999'}">${last[0].toFixed(0)}%</text>`);
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
  const chips = ser.map((x) => `<span class="${x.m === S.model ? 'new' : ''}"><i style="border-color:${MODEL_COLOR[x.m] || '#999'}"></i>${MODEL_LABEL[x.m] || x.m} ${cyc(x.e.cycle)}</span>`).join('')
    + (ser.length > 1 ? `<span class="new"><i style="border-color:#0f172a;border-top-width:3px"></i>mean of models</span>` : '')
    + `<span><i style="border-color:#7ea8d5;border-top-width:6px"></i>Stage IV</span><span><i style="border-color:#0f172a;border-top-style:dashed"></i>1991–2020 normal</span>`;
  host.innerHTML = `<div class="evolegend">${chips}</div>` + P.join('');
  HEAT.wireTips(host);
  const di = divInfo(k);
  $('sub1').textContent = `${label(k)}${di ? ` · ${di.region} · ${di.area.toLocaleString()} sq mi` : ' · area-weighted union of NWRFC divisions'} · member mean per model, p10–p90 band for ${MODEL_LABEL[S.model] || S.model}`;
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

function scorecard() {
  const keys = Object.keys(S.latest.composites);
  const periods = S.latest.periods.filter((p) => p !== 'd11-15' && (p !== 'd16-32' || models().some((m) => Object.values(entry(m).periods).some((pp) => pp['d16-32']))));
  const rows = keys.map((k) => ({ key: k, label: k, cls: k === S.basin ? 'cons' : '' }));
  const cols = periods.map((p) => ({ key: p, label: PERIOD_LABEL[p] }));
  HEAT.table($('score'), rows, cols, (k, p) => {
    const v = meanPeriod(k, p); if (!v) return null;
    const main = showVal(v); if (main === null) return null;
    const d = showDelta(meanDelta(k, p));
    const fm = (x) => (S.heatwhat === 'anom' ? `${sgn(x, 0)} mm` : showFmt(x));
    // colour by the reading itself (vs normal), the change rides in small type
    const colv = S.heatwhat === 'pct' ? main - 100 : S.heatwhat === 'anom' ? main : main;
    return { v: colv, text: `${fm(main)} <small>${fm(v.lo)}–${fm(v.hi)}${d === null ? '' : ' · ' + showSgn(d)}</small>`,
             mark: v.n < v.want ? '<i class="sh">*</i>' : '',
             tip: `${k} · ${PERIOD_LABEL[p]} · mean of ${v.models} models\n${v.mm.toFixed(1)} mm${v.normal !== undefined ? ` / normal ${v.normal.toFixed(1)} = ${v.pct.toFixed(0)}% (${sgn(v.mm - v.normal, 1)} mm)` : ''}\nmodels range ${fm(v.lo)} to ${fm(v.hi)}` + (d === null ? '' : `\nmean same-day change vs previous issues: ${showSgn(d)}`) };
  }, [], { limb: S.heatwhat === 'mm' ? limbAbs : limb, scale: S.heatwhat === 'pct' ? 150 : 'auto', nice: showNice(), fmt: (v) => showFmt(v) });
  $('score').querySelectorAll('td.k').forEach((td, i) => td.addEventListener('click', () => { S.basin = keys[i]; $('basin').value = S.basin; render(); }));
  $('subsc').textContent = `mean of ${models().length} models · ${SHOW[S.heatwhat]} · small type: model range, then the mean same-day change vs the previous issue`;
  $('scnote').textContent = 'Composites are area-weighted unions of NWRFC divisions; Columbia above The Dalles is the whole basin above the dam, not the NWRFC mainstem reach. Click a row to open it.';
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
function mapPanel() {
  const host = $('map'); if (!GEO) { host.innerHTML = '<div class="empty">Loading geometry…</div>'; return; }
  const lon0 = -124.6, lon1 = -109.6, lat0 = 41.0, lat1 = 53.0;
  const K = 62, cosk = Math.cos(47 * Math.PI / 180);
  const W = Math.round((lon1 - lon0) * K * cosk) + 12, H = Math.round((lat1 - lat0) * K) + 12;
  const X = (lo) => 6 + (lo - lon0) * K * cosk, Y = (la) => 6 + (lat1 - la) * K;
  const vals = GEO.features.map((f) => mapValue(f.properties.code)).filter(Boolean).map((x) => x.v);
  const absMode = S.heatwhat === 'mm' && S.mapwhat !== 'change';
  const scale = absMode ? Math.max(1, ...vals) : S.heatwhat === 'pct' && S.mapwhat !== 'change' ? 150 : HEAT.autoScale(vals, showNice());
  const mlimb = absMode ? limbAbs : limb;
  const P = [`<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">`];
  const labels = [];
  for (const f of GEO.features) {
    const p = f.properties; const x = mapValue(p.code);
    const col = x ? HEAT.color(x.v, scale, mlimb) : { bg: '#f1f4f7', ink: 'var(--ink3)' };
    const polys = f.geometry.type === 'Polygon' ? [f.geometry.coordinates] : f.geometry.coordinates;
    let d = ''; let best = null;
    for (const poly of polys) {
      for (const ring of poly) { d += ring.map((c, i) => `${i ? 'L' : 'M'}${X(c[0]).toFixed(1)} ${Y(c[1]).toFixed(1)}`).join('') + 'Z'; }
      const ring = poly[0]; let sx = 0, sy = 0; ring.forEach((c) => { sx += c[0]; sy += c[1]; });
      if (!best || ring.length > best.n) best = { n: ring.length, x: sx / ring.length, y: sy / ring.length };
    }
    const tip = `${p.name} (${p.code}) · ${p.region} · ${Math.round(p.area).toLocaleString()} sq mi${x ? '\n' + x.tip : '\nno value'}`;
    P.push(`<path d="${d}" fill="${col.bg || '#f1f4f7'}" class="${p.code === S.basin ? 'sel' : ''}" data-code="${p.code}" data-t="${tip}"/>`);
    if (best) labels.push(`<text class="lab" x="${X(best.x).toFixed(1)}" y="${(Y(best.y) + 3).toFixed(1)}" text-anchor="middle" fill="${x ? 'var(--ink)' : 'var(--ink3)'}">${x ? x.text : '–'}</text>`);
  }
  P.push(labels.join('')); P.push('</svg>');
  host.innerHTML = P.join('');
  host.querySelectorAll('path[data-code]').forEach((el) => el.addEventListener('click', () => { S.basin = el.dataset.code; $('basin').value = S.basin; render(); }));
  HEAT.wireTips(host);
  const e = entry(S.model);
  const what = SHOW[S.heatwhat];
  $('submap').textContent = S.mapwhat === 'model' ? `${MODEL_LABEL[S.model] || S.model} ${e ? cyc(e.cycle) : ''} · ${PERIOD_LABEL[S.period]} · ${what} by division` :
    S.mapwhat === 'change' ? `${MODEL_LABEL[S.model] || S.model} · ${PERIOD_LABEL[S.period]} · same-day change vs the previous issue, ${S.heatwhat === 'pct' ? 'percentage points of normal' : 'mm'}` : `NCEP Stage IV · ${MAPWHAT[S.mapwhat].replace('% of normal', what)} · click a division to open it`;
  const sw = (l, t) => `<i style="background:${HEAT.mix(HEAT.RAMP[l], t)}"></i>`;
  $('maplegend').innerHTML = absMode ? `<span class="lg">0 mm ${[.15, .33, .5, .66, .83, 1].map((t) => sw('green', t)).join('')}<b>${scale.toFixed(0)} mm</b></span>`
    : S.mapwhat === 'change' ? HEAT.legend(scale, limb, 'drier', 'wetter', (x) => x + showUnitDelta())
    : S.heatwhat === 'anom' ? HEAT.legend(scale, limb, 'below normal', 'above normal', (x) => x + ' mm')
    : `<span class="lg">${[1, .66, .33].map((t) => sw('brown', t)).join('')}<b>0%</b> &nbsp; 50% &nbsp; <b>100% of normal</b> &nbsp; 150% &nbsp; <b>≥250%</b>${[.33, .66, 1].map((t) => sw('green', t)).join('')}</span>`;
}

/* ---- by period ----------------------------------------------------------------------------- */
function ptab() {
  const k = S.basin; const ms = models().filter((m) => entry(m).periods[k]);
  const cols = S.latest.periods.map((p) => ({ key: p, label: PERIOD_LABEL[p] }));
  const rows = (ms.length > 1 ? [{ key: '__mean', label: 'Mean of models', cls: 'blend' }] : []).concat(ms.map((m) => ({ key: m, label: MODEL_LABEL[m] || m })));
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
               + (pr ? `\nsame-day change vs ${cyc(pr.cycle)}: ${sgn(dd.mm, 1)} mm${dd.pct !== undefined ? `, ${sgn(dd.pct)} pts` : ''}` : '') };
  }, [{ label: 'issue', value: (m) => (m === '__mean' ? { text: '', cls: 'tot' } : { text: cyc(entry(m).cycle), cls: 'tot' }) }], { limb, fmt: (v) => sgn(v), nice: showNice() });
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
    const W = 360, H = 164, M = { l: 30, r: 6, t: 32, b: 16 };
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
      P.push(`<path d="${d}" fill="none" stroke="${MODEL_COLOR[x.m] || '#999'}" stroke-width="${x.m === S.model ? 2 : 1.2}"/>`); });
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
    div.addEventListener('click', () => { S.basin = k; $('basin').value = k; render(); window.scrollTo({ top: 0, behavior: 'smooth' }); });
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
  $('runbadge').textContent = `${ms.length} models · Stage IV through ${o[o.length - 1] || '–'}`;
  const newest = ms.map((m) => entry(m).cycle).sort().slice(-1)[0];
  const cd = meanPeriod('Columbia abv The Dalles', 'd1-10');
  $('intro').innerHTML = `Precipitation forecasts for the Columbia River basin from ${ms.map((m) => MODEL_LABEL[m] || m).join(', ')}, `
    + `averaged over the Northwest River Forecast Center's water-supply divisions and read against the 1991–2020 normal and against NCEP Stage IV observed rainfall. `
    + (cd && cd.pct !== undefined ? `Newest issue ${cyc(newest)}: the basin above The Dalles is forecast at <b>${cd.pct.toFixed(0)}% of normal</b> over days 1–10 (${cd.mm.toFixed(0)} mm against a normal ${cd.normal.toFixed(0)}; models ${cd.lo.toFixed(0)}–${cd.hi.toFixed(0)}%). ` : '')
    + `Every table can be read as percent of normal, as the departure in mm, or as absolute mm.`;
  $('credits').innerHTML = 'Sources: NWRFC forecast basins and mean-areal-precipitation normals (NOAA/NWS); NCEP Stage IV multi-sensor precipitation analysis via NOMADS and the Iowa Environmental Mesonet archive; '
    + 'ECMWF open data (IFS, AIFS, ENS) © ECMWF, CC BY 4.0; NOAA GFS and GEFS via NOMADS; Environment and Climate Change Canada GDPS via MSC Datamart. Model output is unadjusted.';
  $('foot').innerHTML = 'Divisions and 1991–2020 mean-areal-precipitation normals are the NWRFC water-supply divisions (42, dissolved from the 379 NWRFC forecast basins); '
    + 'the composites are area-weighted unions — the NWRFC mainstem groups are LOCAL reach areas, so "above The Dalles" here is the union of every Columbia, Snake, Middle and Upper Columbia division, not the mainstem row on the NWRFC page. '
    + 'Days are 12Z–12Z, labelled by the ending date, matching the Stage IV 24 h product; a 00Z run\'s first day is its hours 12–36. '
    + 'Percent of normal = period precipitation ÷ the sum of daily normals over the same dates (a month\'s normal spread evenly). '
    + 'Model precipitation is the model\'s own grid, area-averaged on a 0.02° mesh; GEFS members through the 0.5° tail past day 10; ECMWF ENS is a 10-member subset. '
    + 'Observed: NCEP Stage IV 24 h (12Z) division means from the 4 km grid, kept from NOMADS as they publish (about two weeks retained upstream). '
    + '<b>*</b> run stops short of the window. Run-over-run cells are same-calendar-day changes. '
    + `pnw_history.json built ${S.hist.built || ''}.`;
}

function controls() {
  fill($('basin'), basinList(), S.basin);
  const ms = models(); if (!ms.includes(S.model)) S.model = ms[0];
  fill($('model'), ms, S.model, (m) => MODEL_LABEL[m] || m);
  fill($('period'), S.latest.periods, S.period, (p) => PERIOD_LABEL[p]);
  fill($('heatwhat'), Object.keys(SHOW), S.heatwhat, (k) => SHOW[k]);
  fill($('nruns'), ['8', '12', '20', '30'], String(S.nruns), (n) => `last ${n}`);
  fill($('mapwhat'), Object.keys(MAPWHAT), S.mapwhat, (k) => MAPWHAT[k]);
  const on = (id, f) => $(id).addEventListener('change', (e) => { f(e.target.value); render(); });
  on('basin', (v) => { S.basin = v; }); on('model', (v) => { S.model = v; }); on('period', (v) => { S.period = v; });
  on('heatwhat', (v) => { S.heatwhat = v; }); on('nruns', (v) => { S.nruns = +v; }); on('mapwhat', (v) => { S.mapwhat = v; });
}

const DATA = window.PNW_DATA || '../data/out/';
async function load() {
  S.latest = await (await fetch(DATA + 'pnw_latest.json?t=' + Date.now())).json();
  S.hist = await (await fetch(DATA + 'pnw_history.json?t=' + Date.now())).json();
  if (!GEO) { try { GEO = await (await fetch('pnw_divisions.geojson')).json(); } catch (e) { GEO = null; } }
}
(async () => {
  await load(); controls(); render();
  setInterval(async () => {
    try { const l = await (await fetch(DATA + 'pnw_latest.json?t=' + Date.now())).json();
      if (l.built !== S.latest.built) { await load(); render(); } } catch (e) { /* keep */ }
  }, 60000);
})();
