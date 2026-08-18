/* Skew-T Explorer — all client-side.
   Data: U. Wyoming sounding archive, mirrored to the skewt-data branch by
   .github/workflows/skewt-data.yml (UW sends no CORS headers; raw.github
   serves ACAO *). Physics: SHARPlib via WebAssembly (sharplib.js/.wasm). */
"use strict";

const MISSING = -9999.0;
// ── data-branch fetching with host fallbacks ─────────────────────────────────
// The mirror/climatology/archive live on git branches (kept off main so 6×/day
// force-pushes don't bloat the repo or churn Pages). raw.githubusercontent.com
// is the primary host, but some networks block it — so every branch fetch
// falls back to api.github.com (different domain, always fresh, 60 req/h is
// plenty for the small live files) and then jsDelivr's CDN (very widely
// whitelisted; caches branches up to ~12 h, stale but far better than dead).
const GH_REPO = "scorvec/scorvec.github.io";
async function ghFetch(branch, path, bust, staticish) {
  const q = bust ? "?t=" + encodeURIComponent(bust) : "";
  // Host order matters on networks that block raw.githubusercontent:
  // api.github.com allows only ~60 anonymous requests/hour, so save it for
  // the fresh-critical files (manifest, anomalies, latest soundings).
  // Content that never changes once written (climo, archive zips, dated
  // soundings) passes `staticish` and prefers jsDelivr — its cache
  // staleness is harmless there, and the api quota stays alive for the
  // record map + "Extreme today" panel.
  const raw = [`https://raw.githubusercontent.com/${GH_REPO}/${branch}/${path}${q}`, null];
  const api = [`https://api.github.com/repos/${GH_REPO}/contents/${path}?ref=${branch}`,
     { headers: { Accept: "application/vnd.github.raw" } }];
  const cdn = [`https://cdn.jsdelivr.net/gh/${GH_REPO}@${branch}/${path}`, null];
  const tries = staticish ? [raw, cdn, api] : [raw, api, cdn];
  for (let i = 0; i < tries.length; i++) {
    try {
      const r = await fetch(tries[i][0], tries[i][1] || undefined);
      if (r.ok) return r;
      if (r.status === 404 && i === 0) return r;   // host fine, file truly absent
    } catch (e) { /* blocked / offline — try the next host */ }
  }
  return { ok: false, status: 0 };
}
const CLIMO_BASE = "https://raw.githubusercontent.com/scorvec/scorvec.github.io/skewt-climo/climo/";
let climo = null, climoGid = null;                 // current station climatology
let lastProf = null, lastRes = null, lastMonth = null, lastDoy = null;
let lastValidDt = null;                 // "YYYY-MM-DD HH:MM" of the plotted sounding
let prev12 = null, prev12Key = "";      // the launch ~12 h earlier, for MSE trend
let lastHourZ = 12;
let lastProfFull = null;              // un-thinned profile for the full-res mobile card
let EXPORT_PX = null, BARB_GAP = 22;  // overrides during full-res export
// design aspect (w/h) of each on-screen plot — fitCanvas() letterboxes to
// these and fitLayout() sizes the dialog so the two together fill its width
const PLOT_ASPECT = { skewt: 19 / 21, hodo: 14 / 15 };
const MOBILE_MQ = matchMedia("(max-width: 1020px)");   // mirrors the stacked-layout CSS breakpoint
const hourOf = s => { const m = /[T ](\d{2}):/.exec(String(s)); return m ? +m[1] : 12; };
function doyOf(ymd) {                     // "YYYY-MM-DD" -> 1..365
  const d = new Date(ymd.slice(0, 10) + "T00:00:00Z");
  const j0 = Date.UTC(d.getUTCFullYear(), 0, 1);
  return Math.min(365, Math.floor((d.getTime() - j0) / 864e5) + 1);
}
// climatology is anchored every 5 days; pick the nearest anchor (wrapping)
function climoSlot(idxObj) {
  if (!climo || !climo.doy || lastDoy === null) return -1;
  let best = -1, bd = 1e9;
  climo.doy.forEach((a, i) => {
    let d = Math.abs(a - lastDoy); d = Math.min(d, 365 - d);
    if (d < bd) { bd = d; best = i; }
  });
  return best;
}
const climoCache = new Map();
async function loadClimo(gid) {
  if (!gid) { climo = null; return; }
  if (gid === climoGid) return;
  climoGid = gid; climo = null;
  if (climoCache.has(gid)) { climo = climoCache.get(gid); refreshTables(); return; }
  try {
    const r = await ghFetch("skewt-climo", "climo/" + gid + ".json", null, true);
    const d = r.ok ? await r.json() : null;
    climoCache.set(gid, d);
    if (gid === climoGid) { climo = d; refreshTables(); }
  } catch (e) { climoCache.set(gid, null); }
}
function refreshTables() {
  if (lastProf && lastRes && !modal.hidden) fillTables(lastProf, lastRes);
}
let climoNow = {};
const CLIMO_VARS = [
  ["pwat", "Precipitable water", "mm"], ["850t", "850 hPa temperature", "°C"],
  ["700t", "700 hPa temperature", "°C"], ["500t", "500 hPa temperature", "°C"],
  ["850td", "850 hPa dewpoint", "°C"], ["700td", "700 hPa dewpoint", "°C"],
  ["h500", "500 hPa height", "m"], ["thick", "1000–500 hPa thickness", "m"],
  ["fzl", "Freezing level (AGL)", "m"], ["wbz", "Wet-bulb 0 °C (AGL)", "m"],
  ["kidx", "K-index", ""], ["tott", "Total Totals", ""],
  ["ecape", "ECAPE (MU)", "J/kg"], ["ship", "Significant Hail (SHIP)", ""],
  ["850spd", "850 hPa wind speed", "m/s"], ["250spd", "250 hPa wind speed", "m/s"],
];
const MON = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
const climoModal = document.getElementById("climo-modal");
function openClimo() {
  if (!climo) return;
  const sel = document.getElementById("climo-var");
  sel.innerHTML = CLIMO_VARS.filter(([k]) =>
    (climo.idx || {})[k] && (climo.idx[k].n || []).some(x => x >= 30))
    .map(([k, lab]) => `<option value="${k}">${lab}</option>`).join("");
  document.getElementById("climo-title").textContent =
    `${current.n || current.gid} · climatology`;
  climoModal.hidden = false;
  drawClimo(sel.value);
  drawClimoDist(sel.value);
}
function drawClimo(key) {
  const meta = CLIMO_VARS.find(v => v[0] === key) || [key, key, ""];
  const cv = document.getElementById("climo-canvas"), x = cv.getContext("2d");
  const W = cv.width, H = cv.height, L = 66, R = 22, T = 26, B = 46;
  x.clearRect(0, 0, W, H);
  const A = climo.idx && climo.idx[key];
  if (!A) return;
  const D = climo.doy;

  // usable anchors only (thin windows are dropped by the builder)
  const ok = D.map((_, i) => A.p[i] && A.p[i][0] !== null && (A.n[i] || 0) >= 5);
  const vals = [];
  D.forEach((_, i) => { if (!ok[i]) return;
    vals.push(A.min[i], A.max[i]); });
  if (vals.length < 4) return;
  let lo = Math.min(...vals), hi = Math.max(...vals);
  const pad = (hi - lo) * 0.05 || 1; lo -= pad; hi += pad;

  const px = d => L + (W - L - R) * (d - 1) / 364;
  const py = v => T + (H - T - B) * (1 - (v - lo) / (hi - lo));

  // grid
  x.strokeStyle = "#2a2a40"; x.fillStyle = "#b6b6cc"; x.font = "15px Inter";
  x.textAlign = "right";
  for (let i = 0; i <= 4; i++) {
    const v = lo + (hi - lo) * i / 4, yy = py(v);
    x.beginPath(); x.moveTo(L, yy); x.lineTo(W - R, yy); x.stroke();
    x.fillText(Math.round(v), L - 8, yy + 5);
  }
  x.textAlign = "center"; x.font = "600 14px Inter";
  const MSTART = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335];
  MSTART.forEach((d, m) => x.fillText(MON[m], px(d + 14), H - 16));

  // band between two percentile indices (or min/max arrays)
  const band = (getA, getB, col) => {
    x.fillStyle = col; x.beginPath();
    let started = false;
    D.forEach((d, i) => { if (!ok[i]) return;
      const xx = px(d), yy = py(getA(i));
      started ? x.lineTo(xx, yy) : x.moveTo(xx, yy); started = true; });
    for (let i = D.length - 1; i >= 0; i--) { if (!ok[i]) continue;
      x.lineTo(px(D[i]), py(getB(i))); }
    x.closePath(); x.fill();
  };
  // record envelope (all-time daily max/min), then the percentile bands
  band(i => A.max[i], i => A.min[i], "rgba(140,150,180,0.13)");
  band(i => A.p[i][6], i => A.p[i][2], "rgba(74,122,181,0.26)");   // p10-p90
  band(i => A.p[i][5], i => A.p[i][3], "rgba(74,122,181,0.38)");   // p25-p75

  const line = (get, col, w, dash) => {
    x.strokeStyle = col; x.lineWidth = w; x.setLineDash(dash || []);
    x.beginPath(); let st = false;
    D.forEach((d, i) => { if (!ok[i]) return;
      const xx = px(d), yy = py(get(i));
      st ? x.lineTo(xx, yy) : x.moveTo(xx, yy); st = true; });
    x.stroke(); x.setLineDash([]);
  };
  line(i => A.max[i], "#e0603a", 1.6);                 // record high envelope
  line(i => A.min[i], "#4a7ab5", 1.6);                 // record low envelope
  line(i => A.p[i][4], "#9fc4f5", 2.6);                // median

  // the sounding on display
  const cur = climoNow[key];
  if (isFinite(cur) && lastDoy !== null) {
    x.fillStyle = "#ffd60a"; x.beginPath(); x.arc(px(lastDoy), py(cur), 8, 0, 7); x.fill();
    x.strokeStyle = "#000"; x.lineWidth = 1.2; x.stroke();
  }

  // legend
  x.textAlign = "left"; x.font = "600 12px Inter";
  const key3 = [["#e0603a", "record high"], ["#4a7ab5", "record low"],
                ["#9fc4f5", "median"], ["rgba(74,122,181,0.55)", "10–90 / 25–75 %ile"]];
  let lx = L + 6;
  key3.forEach(([c, lab]) => {
    x.fillStyle = c; x.fillRect(lx, T - 16, 14, 5);
    x.fillStyle = "#a8a8c2"; x.fillText(lab, lx + 19, T - 11);
    lx += 22 + x.measureText(lab).width + 14;
  });
  document.getElementById("climo-sub").textContent =
    `${meta[1]}${meta[2] ? " (" + meta[2] + ")" : ""} · record ${climo.yr0}–${climo.yr1}`;
}

// ---- fog diagnostics (BUFKIT-style radiation-fog screening) ----
// Empirical fog/stratus probabilities: logistic models fit on 87,568 matched
// sounding-METAR pairs (14 co-located stations incl. Buenos Aires, 2006-2025).
// Features standardized with the training means/sds; missing features fall
// back to the training mean (standardized 0). Station-held-out AUC 0.89-0.92.
const FOG_MODEL = {"fog":{"mu":[3.5493,80.5304,4.1064,2.2102,1.8378,114.3392,41.5371,142.164,82.3222,72.5089,1.4016,0.996,443.6591,7.3728,10.2455,-0.0083,0.0013,0.9926],"sd":[3.6834,16.3206,3.9419,5.665,2.6478,154.3423,250.1182,471.2011,16.6885,23.5149,1.5727,7.1132,460.4216,4.2844,8.8232,0.7064,0.7077,0.0854],"w":[-7.1094,-0.043,1.9138,-0.6283,0.0184,-0.535,0.1701,0.017,-0.0878,2.6623,-0.056,0.1444,0.68,-0.043,-0.3321,0.111,-0.161,0.3622,-0.0711]},"strat10":{"mu":[3.5493,80.5304,4.1064,2.2102,1.8378,114.3392,41.5371,142.164,82.3222,72.5089,1.4016,0.996,443.6591,7.3728,10.2455,-0.0083,0.0013,0.9926],"sd":[3.6834,16.3206,3.9419,5.665,2.6478,154.3423,250.1182,471.2011,16.6885,23.5149,1.5727,7.1132,460.4216,4.2844,8.8232,0.7064,0.7077,0.0854],"w":[-6.2643,-2.1899,-2.4963,0.1273,-0.0937,-0.721,-0.4567,0.0536,0.2154,4.2937,-0.3149,-0.181,-0.1266,-2.1899,-0.8317,0.9305,-0.0806,-0.1061,0.0186]},"strat30":{"mu":[3.5493,80.5304,4.1064,2.2102,1.8378,114.3392,41.5371,142.164,82.3222,72.5089,1.4016,0.996,443.6591,7.3728,10.2455,-0.0083,0.0013,0.9926],"sd":[3.6834,16.3206,3.9419,5.665,2.6478,154.3423,250.1182,471.2011,16.6885,23.5149,1.5727,7.1132,460.4216,4.2844,8.8232,0.7064,0.7077,0.0854],"w":[-3.2605,-0.7998,-1.5181,0.1717,-0.0661,-0.3892,-0.5033,0.0778,0.3497,2.6957,0.4889,-0.1052,-0.7021,-0.7998,-0.7401,0.4621,-0.0805,-0.0387,0.0836]}};
const FOG_FEATS = ["dep_sfc","rh_sfc","spd_sfc","ri","inv_dt","inv_top","sat95_dep",
  "rh90_dep","rhmax_lo","rhmax_mid","dq_mix","lapse500","lcl_m","q_sfc","t_sfc",
  "sin_doy","cos_doy","is12z"];
function fogModelFeatures(prof) {
  const esw = tc => 6.112 * Math.exp(17.67 * tc / (tc + 243.5));
  const t0 = prof.T[0] - 273.15, d0 = prof.D[0] - 273.15;
  if (!isFinite(t0) || !isFinite(d0)) return null;
  const f = { dep_sfc: t0 - d0, rh_sfc: 100 * esw(d0) / esw(t0),
    lcl_m: 125 * (t0 - d0), t_sfc: t0,
    q_sfc: 622 * esw(d0) / (prof.P[0] / 100 - esw(d0)) };
  f.spd_sfc = isFinite(prof.U[0]) ? Math.hypot(prof.U[0], prof.V[0]) * KT : null;
  const rb = bulkRi(prof);
  f.ri = rb ? Math.max(-25, Math.min(25, rb.ri)) : null;
  const z0 = prof.H[0];
  let invDt = 0, invTop = 0, tmax = t0;
  let sat95 = 0, rh90 = 0, b95 = false, b90 = false, rhLo = 0, rhMid = 0;
  const qs = [];
  for (let i = 0; i < prof.P.length; i++) {
    const za = prof.H[i] - z0, tC = prof.T[i] - 273.15, dC = prof.D[i] - 273.15;
    if (za <= 1000 && isFinite(tC) && i > 0) {
      if (tC >= tmax - 0.05 && invTop >= 0) {
        tmax = Math.max(tmax, tC); invDt = tmax - t0; invTop = za;
      } else invTop = -1;                      // inversion scan stopped
    }
    if (za > 3000 || !isFinite(tC) || !isFinite(dC)) continue;
    const rh = 100 * esw(dC) / esw(tC);
    if (rh >= 95 && !b95) sat95 = za; else if (rh < 95) b95 = true;
    if (rh >= 90 && !b90) rh90 = za; else if (rh < 90) b90 = true;
    if (za <= 500) rhLo = Math.max(rhLo, rh);
    else if (za <= 1500) rhMid = Math.max(rhMid, rh);
    if (za <= 1500) qs.push([za, 622 * esw(dC) / (prof.P[i] / 100 - esw(dC))]);
  }
  f.inv_dt = invDt; f.inv_top = Math.max(0, invTop);
  f.sat95_dep = sat95; f.rh90_dep = rh90; f.rhmax_lo = rhLo; f.rhmax_mid = rhMid;
  const qlo = qs.filter(q => q[0] <= 300).map(q => q[1]);
  const qhi = qs.filter(q => q[0] >= 600 && q[0] <= 1200).map(q => q[1]);
  f.dq_mix = qlo.length && qhi.length
    ? qlo.reduce((a, b) => a + b) / qlo.length - qhi.reduce((a, b) => a + b) / qhi.length : null;
  let t500 = null;
  for (let i = 1; i < prof.P.length; i++) {
    const za = prof.H[i] - z0;
    if (za >= 500 && isFinite(prof.T[i]) && isFinite(prof.T[i - 1])) {
      const zp = prof.H[i - 1] - z0;
      const fr = (500 - zp) / Math.max(1, za - zp);
      t500 = (prof.T[i - 1] - 273.15) + fr * (prof.T[i] - prof.T[i - 1]); break;
    }
  }
  f.lapse500 = t500 !== null ? (t500 - t0) / 0.5 : null;
  const doy = lastDoy ?? 183;
  f.sin_doy = Math.sin(2 * Math.PI * doy / 365.25);
  f.cos_doy = Math.cos(2 * Math.PI * doy / 365.25);
  f.is12z = lastHourZ === 12 ? 1 : 0;
  return f;
}
function fogModelProbs(prof) {
  const f = fogModelFeatures(prof);
  if (!f) return null;
  const out = {};
  for (const [name, m] of Object.entries(FOG_MODEL)) {
    let z = m.w[0];
    for (let i = 0; i < FOG_FEATS.length; i++) {
      const v = f[FOG_FEATS[i]];
      if (v !== null && isFinite(v)) z += m.w[i + 1] * (v - m.mu[i]) / m.sd[i];
    }
    out[name] = 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, z))));
  }
  return out;
}
function fogRows(prof) {
  const T0 = prof.T[0] - 273.15, D0 = prof.D[0] - 273.15;
  if (!isFinite(T0) || !isFinite(D0))
    return [["Fog panel", "needs temperature and dewpoint — wind-only sounding"]];
  const spd0 = Math.hypot(prof.U[0], prof.V[0]) * KT;
  // FSI needs a real 850 level: at high-elevation stations interpP falls back
  // to the surface value for "850", which would fabricate the index
  const t850 = prof.P[0] > 87000 ? interpP(prof, "T", 85000) : NaN;
  const w8 = windAt(prof, Math.max(100, (interpHagl(prof, 85000) ?? 1500)));
  const v850 = w8 ? Math.hypot(w8[0], w8[1]) * KT : NaN;
  const fsi = (isFinite(t850)) ? 4 * T0 - 2 * ((t850 - 273.15) + D0) + v850 : NaN;
  const cat = !isFinite(fsi) ? "—" : fsi < 31 ? "HIGH risk" : fsi <= 55 ? "moderate" : "low";
  // surface-based inversion: T rising from the surface
  let invTop = null, invDT = 0;
  for (let i = 1; i < prof.P.length; i++) {
    if (!isFinite(prof.T[i]) || !isFinite(prof.T[i - 1])) break;
    if (prof.T[i] >= prof.T[i - 1] - 0.05) { invTop = prof.H[i] - prof.H[0]; invDT = prof.T[i] - prof.T[0]; }
    else break;
    if (prof.H[i] - prof.H[0] > 1500) break;
  }
  const rh500 = layerRH(prof, prof.P[0], Math.max(prof.P[0] - 6000, 50000));
  const rb = bulkRi(prof);
  const riTxt = !rb ? "—"
    : `${rb.ri > 25 ? ">25" : rb.ri < -25 ? "<-25" : rb.ri.toFixed(2)} over lowest ${Math.round(rb.dz)} m — ` +
      (rb.ri > 0.25 ? "stable, turbulence suppressed (surface decoupled)"
       : rb.ri >= 0 ? "weakly stable, mechanical mixing active"
       : "unstable, convective mixing");
  // dew vs frost: below 0 °C deposition happens at the FROST point, which sits
  // above the dew point (ice saturation < water saturation) — invert ice Magnus
  const tf = frostPtC(D0);
  const deposit = D0 > 0.2
    ? `dew (Td above freezing)${T0 - D0 <= 3 ? " — likely tonight with clear skies" : ""}`
    : D0 > -0.2 ? "dew/frost mix — Td right at freezing"
    : `FROST — frost point ${tf.toFixed(1)} °C, only ${Math.max(T0 - tf, 0).toFixed(1)} °C of cooling needed`;
  const mp = fogModelProbs(prof);
  const pct = v => v < 0.01 ? "<1%" : Math.round(v * 100) + "%";
  const probRows = mp ? [
    ["Fog probability", `${pct(mp.fog)} (vis \u2264 1 km at the airport now)`],
    ["Stratus \u2264 1 kft", `${pct(mp.strat10)} \u00b7 \u2264 3 kft: ${pct(mp.strat30)} (ceiling)`],
  ] : [];
  return [
    ...probRows,
    ["FSI", isFinite(fsi) ? `${fsi.toFixed(0)} — ${cat}` : "—"],
    ["Sfc T − Td", `${(T0 - D0).toFixed(1)} °C ${(T0 - D0) <= 2.5 ? "(near saturation)" : ""}`],
    ["Fog point (sfc Td)", `${D0.toFixed(1)} °C — cool ${(T0 - D0).toFixed(1)} °C to saturate`],
    ["If sfc cools to saturation", deposit],
    ["Surface wind", isFinite(spd0) ? `${spd0.toFixed(0)} kt` : "—"],
    ["Bulk Richardson", riTxt],
    ["Sfc inversion", invTop ? `yes — top ${Math.round(invTop)} m, +${invDT.toFixed(1)} °C` : "none detected"],
    ["Mean RH lowest 60 hPa", isFinite(rh500) ? rh500.toFixed(0) + " %" : "—"],
    ["Wet-bulb (sfc)", `${wetbulbC(T0, D0).toFixed(1)} °C`],
  ];
}
// Bulk Richardson number over the lowest resolved layer: buoyant suppression
// vs shear production of turbulence. Ri > ~0.25 means the surface layer is
// dynamically stable/decoupled — radiative cooling and moisture stay trapped.
function bulkRi(prof) {
  const z0 = prof.H[0];
  let i = 1;
  while (i < prof.P.length &&
         (prof.H[i] - z0 < 30 || !isFinite(prof.T[i]) || !isFinite(prof.U[i]))) i++;
  if (i >= prof.P.length || !isFinite(prof.T[0]) || !isFinite(prof.U[0])) return null;
  const thv = k => {
    const dC = prof.D[k] - 273.15;
    const e = isFinite(dC) ? 6.112 * Math.exp(17.67 * dC / (dC + 243.5)) : 0;
    const r = 0.622 * e / (prof.P[k] / 100 - e);
    return prof.T[k] * (1 + 0.61 * r) * Math.pow(100000 / prof.P[k], 0.2854);
  };
  const dz = prof.H[i] - z0;
  const du = prof.U[i] - prof.U[0], dv = prof.V[i] - prof.V[0];
  const s2 = Math.max(du * du + dv * dv, 1e-4);
  return { ri: 9.80665 / (0.5 * (thv(0) + thv(i))) * (thv(i) - thv(0)) * dz / s2, dz };
}
function frostPtC(tdC) {
  const e = 6.112 * Math.exp(17.67 * tdC / (tdC + 243.5));
  return 272.62 * Math.log(e / 6.112) / (22.46 - Math.log(e / 6.112));
}
// PBL-zoom charts: T / Td / frost point vs height AGL (lowest ~2 km, where fog,
// frost and dew are decided) plus an RH strip showing the hydrolapse.
function drawFogCharts(prof) {
  const cv = document.getElementById("fog-canvas");
  const { W, H, ctx } = fitCanvas(cv);
  ctx.fillStyle = TH.panel; ctx.fillRect(0, 0, W, H);
  ctx.font = "10px Inter, sans-serif";
  const z0 = prof.H[0], pts = [];
  for (let i = 0; i < prof.P.length; i++) {
    const z = prof.H[i] - z0;
    if (z > 2300) break;
    if (!isFinite(prof.T[i])) continue;
    const t = prof.T[i] - 273.15, d = isFinite(prof.D[i]) ? prof.D[i] - 273.15 : NaN;
    const ev = isFinite(d) ? 6.112 * Math.exp(17.67 * d / (d + 243.5)) : NaN;
    pts.push({ z, t, d, p: prof.P[i], u: prof.U[i], v: prof.V[i],
      spd: isFinite(prof.U[i]) ? Math.hypot(prof.U[i], prof.V[i]) * KT : NaN,
      f: isFinite(d) ? frostPtC(d) : NaN,
      w: isFinite(ev) ? 622 * ev / (prof.P[i] / 100 - ev) : NaN });
  }
  if (pts.length < 3) {
    ctx.fillStyle = TH.muted; ctx.fillText("not enough low-level data", 12, 20); return;
  }
  const zMax = Math.max(1000, pts[pts.length - 1].z);
  const Mg = { l: 40, t: 20, b: 24 }, wL = Math.round(W * 0.5), gap = 32;
  let xmin = 99, xmax = -99;
  for (const q of pts) {
    xmax = Math.max(xmax, q.t);
    if (isFinite(q.d)) xmin = Math.min(xmin, q.d);
    if (isFinite(q.f) && q.f <= 0.5) xmin = Math.min(xmin, q.f);
  }
  xmin = Math.floor(xmin) - 2; xmax = Math.ceil(xmax) + 2;
  const x = v => Mg.l + (v - xmin) / (xmax - xmin) * (wL - Mg.l - 6);
  const y = z => H - Mg.b - z / zMax * (H - Mg.t - Mg.b);
  let wTop = 0, sMax = 0;
  for (const q of pts) {
    if (isFinite(q.w)) wTop = Math.max(wTop, q.w);
    if (isFinite(q.spd)) sMax = Math.max(sMax, q.spd);
  }
  wTop = Math.max(1, Math.ceil(wTop * 1.15));
  sMax = Math.max(10, Math.ceil(sMax * 1.15 / 5) * 5);
  const rxL = wL + gap, rxR = rxL + (W - wL - 8 - 2 * gap) * 0.5;
  const sxL = rxR + gap, sxR = W - 8;
  const rx = v => rxL + v / wTop * (rxR - rxL);
  const sx = v => sxL + v / sMax * (sxR - sxL);
  // height grid across both panels
  ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (let z = 0; z <= zMax; z += 500) {
    ctx.strokeStyle = TH.gridSub; ctx.beginPath();
    ctx.moveTo(Mg.l, y(z)); ctx.lineTo(wL - 6, y(z));
    ctx.moveTo(rx(0), y(z)); ctx.lineTo(rx(wTop), y(z));
    ctx.moveTo(sx(0), y(z)); ctx.lineTo(sx(sMax), y(z)); ctx.stroke();
    ctx.fillStyle = TH.muted; ctx.fillText(z ? z + " m" : "sfc", Mg.l - 4, y(z));
  }
  // temp ticks + 0C isotherm
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  const step = (xmax - xmin) > 24 ? 10 : 5;
  for (let v = Math.ceil(xmin / step) * step; v <= xmax; v += step) {
    ctx.strokeStyle = TH.gridSub; ctx.beginPath();
    ctx.moveTo(x(v), y(0)); ctx.lineTo(x(v), Mg.t); ctx.stroke();
    ctx.fillStyle = TH.muted; ctx.fillText(v + "\u00b0", x(v), y(0) + 4);
  }
  if (xmin < 0 && xmax > 0) {
    ctx.strokeStyle = TH.isotherm0; ctx.lineWidth = 1.4; ctx.setLineDash([5, 4]);
    ctx.beginPath(); ctx.moveTo(x(0), y(0)); ctx.lineTo(x(0), Mg.t); ctx.stroke();
    ctx.setLineDash([]); ctx.lineWidth = 1;
  }
  // inversion layers shaded
  for (let i = 1; i < pts.length; i++)
    if (pts[i].t > pts[i - 1].t + 0.02) {
      ctx.fillStyle = "rgba(255,159,10,0.10)";
      ctx.fillRect(Mg.l, y(pts[i].z), wL - 6 - Mg.l, y(pts[i - 1].z) - y(pts[i].z));
    }
  const trace = (key, color, dash, cond) => {
    ctx.strokeStyle = color; ctx.lineWidth = 1.8; ctx.setLineDash(dash);
    ctx.beginPath(); let pen = false;
    for (const q of pts) {
      const ok = isFinite(q[key]) && (!cond || cond(q));
      if (!ok) { pen = false; continue; }
      pen ? ctx.lineTo(x(q[key]), y(q.z)) : ctx.moveTo(x(q[key]), y(q.z)); pen = true;
    }
    ctx.stroke(); ctx.setLineDash([]); ctx.lineWidth = 1;
  };
  trace("t", TH.temp, []);
  trace("d", TH.dwpt, []);
  trace("f", "#7dd8ff", [4, 3], q => q.f <= 0.5);   // frost pt only meaningful near/below 0C
  // mixing-ratio strip: constant w with height = well-mixed moisture that will
  // survive daytime mixing; a sharp surface spike is a shallow skin that won't
  ctx.strokeStyle = TH.grid; ctx.strokeRect(rx(0), Mg.t, rx(wTop) - rx(0), y(0) - Mg.t);
  const wSfc = pts.find(q => isFinite(q.w))?.w;
  if (isFinite(wSfc)) {
    ctx.strokeStyle = "rgba(255,159,10,0.75)"; ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(rx(wSfc), y(0)); ctx.lineTo(rx(wSfc), Mg.t); ctx.stroke();
    ctx.setLineDash([]);
  }
  ctx.fillStyle = "rgba(48,209,88,0.14)";
  ctx.beginPath(); ctx.moveTo(rx(0), y(0)); let pen = false;
  for (const q of pts) if (isFinite(q.w)) ctx.lineTo(rx(q.w), y(q.z));
  ctx.lineTo(rx(0), y(Math.min(zMax, pts[pts.length - 1].z))); ctx.closePath(); ctx.fill();
  ctx.strokeStyle = TH.dwpt; ctx.lineWidth = 1.6; ctx.beginPath();
  for (const q of pts) if (isFinite(q.w)) { pen ? ctx.lineTo(rx(q.w), y(q.z)) : ctx.moveTo(rx(q.w), y(q.z)); pen = true; }
  ctx.stroke(); ctx.lineWidth = 1;
  ctx.textAlign = "center"; ctx.fillStyle = TH.muted;
  const wStep = wTop <= 2 ? 0.5 : wTop <= 5 ? 1 : wTop <= 12 ? 2 : 5;
  for (let v = 0; v <= wTop; v += wStep) ctx.fillText(v, rx(v), y(0) + 4);
  ctx.fillText("w g/kg", (rx(0) + rx(wTop)) / 2, 4);
  // wind strip: speed trace + barbs for direction (drainage flow, nocturnal jet)
  ctx.strokeStyle = TH.grid; ctx.strokeRect(sx(0), Mg.t, sx(sMax) - sx(0), y(0) - Mg.t);
  ctx.strokeStyle = TH.barb; ctx.lineWidth = 1.6; ctx.beginPath();
  let spen = false;
  for (const q of pts) if (isFinite(q.spd)) {
    spen ? ctx.lineTo(sx(q.spd), y(q.z)) : ctx.moveTo(sx(q.spd), y(q.z)); spen = true;
  }
  ctx.stroke(); ctx.lineWidth = 1;
  const bp = { P: [], U: [], V: [] }, py = new Map();
  for (const q of pts) if (isFinite(q.u)) {
    bp.P.push(q.p); bp.U.push(q.u); bp.V.push(q.v); py.set(q.p, y(q.z));
  }
  if (bp.P.length) drawBarbs(ctx, bp, sxR - 18, pp => py.get(pp));
  ctx.textAlign = "center"; ctx.fillStyle = TH.muted;
  const sStep = sMax <= 20 ? 5 : sMax <= 50 ? 10 : 20;
  for (let v = 0; v <= sMax; v += sStep) ctx.fillText(v, sx(v), y(0) + 4);
  ctx.fillText("wind kt", (sx(0) + sx(sMax)) / 2, 4);
  // legend
  ctx.textAlign = "left"; ctx.textBaseline = "alphabetic";
  let lx = Mg.l;
  for (const [lab, col] of [["T", TH.temp], ["Td", TH.dwpt], ["frost pt", "#7dd8ff"], ["inversion", "rgba(255,159,10,0.7)"]]) {
    ctx.fillStyle = col; ctx.fillRect(lx, 8, 10, 3);
    ctx.fillStyle = TH.ink; ctx.fillText(lab, lx + 13, 13);
    lx += 13 + ctx.measureText(lab).width + 12;
  }
}
// ---- cloud layer estimate ----
// A level is "cloudy" when RH — with respect to ice below 0 C, where deposition
// saturates first — exceeds a height-dependent threshold (radiosonde humidity
// sensors read dry in cold air, so the bar drops with altitude).
function cloudLayers(prof) {
  const z0 = prof.H[0], levs = [];
  for (let i = 0; i < prof.P.length; i++) {
    const tC = prof.T[i] - 273.15, dC = prof.D[i] - 273.15;
    const z = prof.H[i] - z0;
    if (z > 15000) break;
    if (!isFinite(tC) || !isFinite(dC)) continue;
    const ew = 6.112 * Math.exp(17.67 * dC / (dC + 243.5));
    let rh = 100 * ew / (6.112 * Math.exp(17.67 * tC / (tC + 243.5)));
    if (tC < 0) rh = Math.max(rh, 100 * ew / (6.112 * Math.exp(22.46 * tC / (tC + 272.62))));
    levs.push({ z, tC, rh: Math.min(rh, 105) });
  }
  // calibrated against 49,259 METAR-verified cloud bases at 14 stations
  // (2006-2025): 85/80 beat 90/85 on both POD (0.77) and FAR (0.51)
  const thr = z => z < 2000 ? 85 : 80;
  const raw = [];
  let cur = null;
  for (const L of levs) {
    if (L.rh >= thr(L.z)) {
      if (!cur) cur = { base: L.z, top: L.z, tBase: L.tC, tTop: L.tC, maxRh: 0 };
      cur.top = L.z; cur.tTop = L.tC; cur.maxRh = Math.max(cur.maxRh, L.rh);
    } else if (cur) { raw.push(cur); cur = null; }
  }
  if (cur) raw.push(cur);
  const layers = [];
  for (const l of raw) {
    const prev = layers[layers.length - 1];
    if (prev && l.base - prev.top < 300) {
      prev.top = l.top; prev.tTop = l.tTop; prev.maxRh = Math.max(prev.maxRh, l.maxRh);
    } else layers.push(l);
  }
  return { layers, levs, thr };
}
// shared by the cloud pop-up and the main indices table. bar 94 = CSI
// optimum; conf = fraction of matched layers in that peak-RH bin that
// verified as BKN/OVC ceilings in training
function estCeiling(res) {
  if (!res || res.levs.length < 3) return undefined;
  const pCeil = rh => rh >= 100 ? 77 : rh >= 98 ? 69 : rh >= 96 ? 65 : rh >= 94 ? 62 : 54;
  const ceil = res.layers.find(l => l.maxRh >= 94);
  if (!ceil) return null;
  const lv = res.levs.find(L => L.z >= ceil.base - 1 && L.z <= ceil.top + 1 && L.rh >= 94);
  const zb = lv ? lv.z : ceil.base;
  return { zb, top: ceil.top, obscured: zb < 50, conf: pCeil(ceil.maxRh) };
}
function cloudRows(res) {
  const fz = z => `${Math.round(z)} m / ${(z * 3.28084 / 1000).toFixed(1)} kft`;
  if (res.levs.length < 3)
    return [["Cloud layers", "needs temperature and dewpoint — no moisture profile"]];
  if (!res.layers.length)
    return [["Cloud layers", "none detected — column subsaturated at every level"]];
  const rows = [];
  // ceiling base = where RH first reaches the broken/overcast bar (92%) inside
  // the layer, not the 85% detection edge — surface haze (RH 85-92%) must not
  // read as a 0 ft ceiling. True surface-based saturation reports as obscured.
  const ec = estCeiling(res);
  rows.push(["Est. ceiling", !ec ? "none — no likely ceiling layer"
    : ec.obscured
    ? `obscured — surface-based saturation (fog), top ${Math.round(ec.top)} m / ` +
      `${(ec.top * 3.28084 / 1000).toFixed(1)} kft`
    : `${Math.round(ec.zb)} m / ${Math.round(ec.zb * 3.28084 / 100) * 100} ft AGL ` +
      `(~${ec.conf}% of similar layers verify as ceilings)`]);
  res.layers.forEach((l, i) => {
    const lvl = l.base < 2000 ? "low" : l.base < 6000 ? "mid" : "high";
    const cov = l.maxRh >= 100 ? "likely broken/overcast"
      : l.maxRh >= 94 ? "broken possible" : "few/scattered";
    const lo = Math.min(l.tBase, l.tTop), hi = Math.max(l.tBase, l.tTop);
    const icing = lo < 0 && hi > -20;
    rows.push([`Layer ${i + 1} (${lvl})`,
      `${fz(l.base)} \u2192 ${fz(l.top)} \u00b7 ${Math.max(1, Math.round((l.top - l.base) / 10) * 10)} m thick \u00b7 ` +
      `peak RH ${Math.round(l.maxRh)}% \u00b7 ${cov}` +
      (icing ? " \u00b7 supercooled (icing range)" : "") +
      (lo <= -12 && hi >= -18 ? " \u00b7 spans DGZ (dendritic snow growth)" : "")]);
  });
  return rows;
}
function drawCloudChart(prof, res) {
  const cv = document.getElementById("cloud-canvas");
  const { W, H, ctx } = fitCanvas(cv);
  ctx.fillStyle = TH.panel; ctx.fillRect(0, 0, W, H);
  ctx.font = "10px Inter, sans-serif";
  const { layers, levs, thr } = res;
  if (levs.length < 3) {
    ctx.fillStyle = TH.muted; ctx.fillText("not enough moisture data", 12, 20); return;
  }
  const zTop = Math.min(15000, Math.max(8000, (layers.length ? layers[layers.length - 1].top : 0) + 2000,
    levs[levs.length - 1].z));
  const Mg = { l: 44, t: 20, b: 24 }, colR = Math.round(W * 0.5);
  const y = z => H - Mg.b - z / zTop * (H - Mg.t - Mg.b);
  const rxL = colR + 40, rxR = W - 44;
  const rx = v => rxL + v / 105 * (rxR - rxL);
  ctx.textBaseline = "middle";
  for (let z = 0; z <= zTop; z += 2000) {
    ctx.strokeStyle = TH.gridSub; ctx.beginPath();
    ctx.moveTo(Mg.l, y(z)); ctx.lineTo(colR - 6, y(z));
    ctx.moveTo(rx(0), y(z)); ctx.lineTo(rx(105), y(z)); ctx.stroke();
    ctx.fillStyle = TH.muted; ctx.textAlign = "right";
    ctx.fillText(z ? z / 1000 + " km" : "sfc", Mg.l - 4, y(z));
    ctx.textAlign = "left";
    ctx.fillText(z ? (z * 3.28084 / 1000).toFixed(0) + " kft" : "", rx(105) + 4, y(z));
  }
  // freezing level across the cloud column
  const zf = freezingLvlAgl(prof);
  if (isFinite(zf) && zf > 0 && zf < zTop) {
    ctx.strokeStyle = TH.isotherm0; ctx.setLineDash([5, 4]);
    ctx.beginPath(); ctx.moveTo(Mg.l, y(zf)); ctx.lineTo(colR - 6, y(zf)); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = TH.isotherm0; ctx.textAlign = "left";
    ctx.fillText("0\u00b0C", Mg.l + 2, y(zf) - 8);
  }
  // dendritic growth zone (-12 to -18 C): efficient snow growth where a
  // saturated layer intersects this band
  const dgz = [];
  for (let i = 1; i < levs.length; i++) {
    const a1 = levs[i - 1], b1 = levs[i];
    if (!isFinite(a1.tC) || !isFinite(b1.tC)) continue;
    const zAt = tt => a1.z + (tt - a1.tC) / ((b1.tC - a1.tC) || 1e-9) * (b1.z - a1.z);
    const cutz = [a1.z, b1.z];
    for (const tt of [-12, -18])
      if ((a1.tC - tt) * (b1.tC - tt) < 0) cutz.push(zAt(tt));
    cutz.sort((x1, x2) => x1 - x2);
    for (let k = 1; k < cutz.length; k++) {
      const zm = (cutz[k - 1] + cutz[k]) / 2;
      const tm = a1.tC + (zm - a1.z) / ((b1.z - a1.z) || 1e-9) * (b1.tC - a1.tC);
      if (tm <= -12 && tm >= -18 && cutz[k] > cutz[k - 1]) {
        const prev = dgz[dgz.length - 1];
        if (prev && cutz[k - 1] <= prev[1] + 1) prev[1] = cutz[k];
        else dgz.push([cutz[k - 1], cutz[k]]);
      }
    }
  }
  for (const [zb, zt] of dgz) {
    if (zb > zTop) continue;
    ctx.fillStyle = "rgba(100,210,255,0.09)";
    ctx.fillRect(Mg.l, y(Math.min(zt, zTop)), colR - 6 - Mg.l,
      Math.max(2, y(zb) - y(Math.min(zt, zTop))));
  }
  // cloud boxes, opacity by peak RH, colored by phase regime:
  // liquid (>0 C) white, supercooled (0 to -20 C) cyan, ice (<-20 C) violet
  const reg = tc => tc > 0 ? 0 : tc > -20 ? 1 : 2;
  const regCol = (r, a) => [`rgba(225,228,238,${a})`, `rgba(100,210,255,${a})`, `rgba(191,90,242,${a})`][r];
  const tAtZ = z => {
    for (let i = 1; i < levs.length; i++)
      if (levs[i].z >= z) {
        const f = (z - levs[i - 1].z) / Math.max(1, levs[i].z - levs[i - 1].z);
        return levs[i - 1].tC + f * (levs[i].tC - levs[i - 1].tC);
      }
    return levs[levs.length - 1].tC;
  };
  for (const l of layers) {
    const a = Math.min(0.85, 0.25 + (l.maxRh - 75) / 40);
    const hPx = Math.max(4, y(l.base) - y(l.top));
    // split the box at 0 C / -20 C crossings so each band wears its phase color
    const cuts = [l.base];
    for (let i = 1; i < levs.length; i++) {
      const a1 = levs[i - 1], b1 = levs[i];
      if (b1.z <= l.base || a1.z >= l.top) continue;
      for (const thrT of [0, -20]) {
        if ((a1.tC - thrT) * (b1.tC - thrT) < 0) {
          const zc = a1.z + (thrT - a1.tC) / (b1.tC - a1.tC) * (b1.z - a1.z);
          if (zc > l.base && zc < l.top) cuts.push(zc);
        }
      }
    }
    cuts.push(l.top); cuts.sort((x1, x2) => x1 - x2);
    for (let i = 1; i < cuts.length; i++) {
      const zb = cuts[i - 1], zt = cuts[i];
      ctx.fillStyle = regCol(reg(tAtZ((zb + zt) / 2)), a);
      ctx.fillRect(Mg.l + 10, y(zt), colR - Mg.l - 26, Math.max(2, y(zb) - y(zt)));
    }
    if (hPx >= 15) {
      ctx.fillStyle = "#0b0b12"; ctx.textAlign = "center";
      ctx.fillText(`${(l.base * 3.28084 / 1000).toFixed(1)}\u2013${(l.top * 3.28084 / 1000).toFixed(1)} kft`,
        (Mg.l + colR - 16) / 2, (y(l.top) + y(l.base)) / 2);
    }
  }
  // DGZ edges + label AFTER the cloud boxes — a supercooled deck otherwise
  // hides the band entirely (Goose Bay 2026-07-11 12Z)
  for (const [zb, zt] of dgz) {
    if (zb > zTop) continue;
    const zt2 = Math.min(zt, zTop);
    ctx.strokeStyle = "rgba(100,210,255,0.8)"; ctx.setLineDash([3, 4]);
    ctx.beginPath();
    ctx.moveTo(Mg.l, y(zb)); ctx.lineTo(colR - 6, y(zb));
    ctx.moveTo(Mg.l, y(zt2)); ctx.lineTo(colR - 6, y(zt2)); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "rgba(140,225,255,0.95)"; ctx.textAlign = "left"; ctx.textBaseline = "middle";
    ctx.fillText("DGZ", Mg.l + 3, (y(zb) + y(zt2)) / 2);
    ctx.textBaseline = "alphabetic";
  }
  // phase legend
  ctx.textAlign = "left"; ctx.textBaseline = "middle";
  let lx = Mg.l + 10;
  for (const [lab, r] of [["liquid", 0], ["supercooled", 1], ["ice", 2]]) {
    ctx.fillStyle = regCol(r, 0.8); ctx.fillRect(lx, H - 13, 9, 8);
    ctx.fillStyle = TH.muted; ctx.fillText(lab, lx + 12, H - 9);
    lx += 12 + 7 * lab.length + 12;
  }
  ctx.textBaseline = "alphabetic";
  // RH profile with the cloudy threshold
  ctx.strokeStyle = TH.grid; ctx.strokeRect(rx(0), Mg.t, rx(105) - rx(0), y(0) - Mg.t);
  ctx.strokeStyle = "rgba(255,159,10,0.65)"; ctx.setLineDash([4, 3]); ctx.beginPath();
  let tpen = false;
  for (let z = 0; z <= zTop; z += 100) {
    tpen ? ctx.lineTo(rx(thr(z)), y(z)) : ctx.moveTo(rx(thr(z)), y(z)); tpen = true;
  }
  ctx.stroke(); ctx.setLineDash([]);
  ctx.strokeStyle = TH.dwpt; ctx.lineWidth = 1.6; ctx.beginPath();
  let pen = false;
  for (const L of levs) { pen ? ctx.lineTo(rx(L.rh), y(L.z)) : ctx.moveTo(rx(L.rh), y(L.z)); pen = true; }
  ctx.stroke(); ctx.lineWidth = 1;
  ctx.fillStyle = TH.muted; ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (const v of [0, 50, 80, 100]) ctx.fillText(v, rx(v), y(0) + 4);
  ctx.fillText("RH % (wrt ice below 0\u00b0C)", (rx(0) + rx(105)) / 2, 4);
  ctx.fillText("cloud layers", (Mg.l + colR) / 2, 4);
  ctx.textBaseline = "alphabetic";
}
document.getElementById("cloud-btn").addEventListener("click", () => {
  if (!lastProf) return;
  const res = cloudLayers(lastProf);
  document.getElementById("cloud-table").innerHTML =
    cloudRows(res).map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
  document.getElementById("cloud-modal").hidden = false;
  requestAnimationFrame(() => { try { drawCloudChart(lastProf, res); } catch (e) {} });
});
document.getElementById("cloud-close").addEventListener("click",
  () => document.getElementById("cloud-modal").hidden = true);
document.getElementById("cloud-modal").addEventListener("click", e => {
  if (e.target.id === "cloud-modal") document.getElementById("cloud-modal").hidden = true;
});

document.getElementById("fog-btn").addEventListener("click", () => {
  if (!lastProf) return;
  document.getElementById("fog-table").innerHTML =
    fogRows(lastProf).map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
  document.getElementById("fog-modal").hidden = false;
  requestAnimationFrame(() => { try { drawFogCharts(lastProf); } catch (e) {} });
});
document.getElementById("fog-close").addEventListener("click",
  () => document.getElementById("fog-modal").hidden = true);
document.getElementById("fog-modal").addEventListener("click", e => {
  if (e.target.id === "fog-modal") document.getElementById("fog-modal").hidden = true;
});

document.getElementById("climo-btn").addEventListener("click", openClimo);
document.getElementById("rec-btn").addEventListener("click", openRecords);
document.getElementById("rec-close").addEventListener("click",
  () => document.getElementById("rec-modal").hidden = true);
document.getElementById("rec-modal").addEventListener("click", e => {
  if (e.target.id === "rec-modal") document.getElementById("rec-modal").hidden = true;
});
document.getElementById("climo-var").addEventListener("change", e => {
  drawClimo(e.target.value); drawClimoDist(e.target.value);
});

// ---- distribution on this date, same synoptic hour ----
// The shipped climo pools 00Z and 12Z; for a histogram you compare ONE
// sounding against, mixing hours smears diurnal signal. Extract the same-hour
// +/-10-day sample from the period-of-record text (cached once per station).
function igraKeyVal(get, key) {
  const sl = pa => get(pa);
  const t85 = sl(85000), t70 = sl(70000), t50 = sl(50000);
  switch (key) {
    case "850t": return t85 && t85.t;
    case "850td": return t85 && t85.d;
    case "700t": return t70 && t70.t;
    case "700td": return t70 && t70.d;
    case "500t": return t50 && t50.t;
    case "h500": return t50 && t50.z;
    case "thick": { const b = sl(100000);
      return (t50 && b && t50.z !== null && b.z !== null) ? t50.z - b.z : null; }
    case "kidx": return (t85 && t70 && t50 && [t85.t, t85.d, t70.t, t70.d, t50.t].every(v => v !== null))
      ? t85.t - t50.t + t85.d - (t70.t - t70.d) : null;
    case "tott": return (t85 && t50 && [t85.t, t85.d, t50.t].every(v => v !== null))
      ? t85.t + t85.d - 2 * t50.t : null;
    case "850spd": return t85 && t85.s;
    case "250spd": { const t25 = sl(25000); return t25 && t25.s; }
    default: return null;
  }
}
function climoDistSamples(text, key, doyC, hour) {
  const out = [];
  let keep = false, levs = null;
  const flush = () => {
    if (!levs) return;
    const get = pa => levs.get(pa) || null;
    const v = igraKeyVal(get, key);
    if (v !== null && isFinite(v)) out.push(v);
  };
  for (const line of text.split("\n")) {
    if (line[0] === "#") {
      if (keep) flush();
      keep = false; levs = null;
      const mo = +line.slice(18, 20), dy = +line.slice(21, 23), hr = +line.slice(24, 26);
      if (hr !== hour || !mo) continue;
      let d = Math.abs(doyOf(`2001-${String(mo).padStart(2, "0")}-${String(dy).padStart(2, "0")}`) - doyC);
      d = Math.min(d, 365 - d);
      if (d <= 10) { keep = true; levs = new Map(); }
    } else if (keep) {
      const pa = +line.slice(9, 15);
      if (![100000, 85000, 70000, 50000, 25000].includes(pa)) continue;
      const iv2 = (a, b) => { const v = parseInt(line.slice(a, b)); return (!isFinite(v) || v <= -8888) ? null : v; };
      const tt = iv2(22, 27), dpdp = iv2(34, 39), gph = iv2(16, 21), ws = iv2(46, 51);
      levs.set(pa, { t: tt !== null ? tt / 10 : null,
        d: (tt !== null && dpdp !== null) ? (tt - dpdp) / 10 : null,
        z: gph, s: ws !== null ? ws / 10 : null });
    }
  }
  if (keep) flush();
  return out;
}
async function drawClimoDist(key) {
  const note = document.getElementById("climo-hist-note");
  const cv = document.getElementById("climo-hist"), x = cv.getContext("2d");
  x.clearRect(0, 0, cv.width, cv.height);
  if (!current || !current.gid || lastDoy === null) { note.textContent = ""; return; }
  if (["pwat", "fzl", "ecape", "ship"].includes(key)) {
    note.textContent = "distribution view: not available for this variable"; return;
  }
  if (!igraCache.has(current.gid + ":por")) {
    note.innerHTML = `<button id="climo-dist-load">load this station's full record to see ` +
      `the distribution on this date (tens of MB, cached)</button>`;
    document.getElementById("climo-dist-load").onclick = async () => {
      note.textContent = "downloading period of record…";
      try { await igraText(current.gid, 1900); } catch (e) { note.textContent = "download failed"; return; }
      drawClimoDist(document.getElementById("climo-var").value);
    };
    return;
  }
  const text = igraCache.get(current.gid + ":por");
  const hour = lastHourZ === 0 ? 0 : 12;
  const vals = climoDistSamples(text, key, lastDoy, hour);
  if (vals.length < 30) { note.textContent = `only ${vals.length} same-hour soundings in the window`; return; }
  const cur = (() => {
    if (!lastProf) return NaN;
    const get = pa => {
      const tK = interpP(lastProf, "T", pa), dK = interpP(lastProf, "D", pa);
      const z = interpP(lastProf, "H", pa);
      return { t: isFinite(tK) ? tK - 273.15 : null, d: isFinite(dK) ? dK - 273.15 : null,
               z: isFinite(z) ? z : null };
    };
    const v = igraKeyVal(get, key);
    return v === null ? NaN : v;
  })();
  vals.sort((a, b) => a - b);
  const lo = vals[0], hi = vals[vals.length - 1], span = Math.max(hi - lo, 1e-6);
  const NB = 24, bins = new Array(NB).fill(0);
  for (const v of vals) bins[Math.min(NB - 1, Math.floor((v - lo) / span * NB))]++;
  const W = cv.width, H = cv.height, L = 66, R = 22, T = 16, B = 40;
  const bw = (W - L - R) / NB, peak = Math.max(...bins);
  x.fillStyle = "rgba(100,170,255,0.55)";
  bins.forEach((c, i) => {
    const h = c / peak * (H - T - B);
    x.fillRect(L + i * bw + 1, H - B - h, bw - 2, h);
  });
  x.strokeStyle = "#3a3a5c"; x.strokeRect(L, T, W - L - R, H - T - B);
  x.fillStyle = "#8b8ba3"; x.font = "20px Inter"; x.textAlign = "center";
  for (let i = 0; i <= 4; i++) {
    const v = lo + span * i / 4;
    x.fillText(v.toFixed(Math.abs(span) > 20 ? 0 : 1), L + (W - L - R) * i / 4, H - B + 26);
  }
  if (isFinite(cur)) {
    const cx = L + Math.max(0, Math.min(1, (cur - lo) / span)) * (W - L - R);
    x.strokeStyle = "#ffd60a"; x.lineWidth = 3;
    x.beginPath(); x.moveTo(cx, T); x.lineTo(cx, H - B); x.stroke();
    const below = vals.filter(v => v < cur).length;
    note.textContent = `distribution of ${vals.length} soundings at ${String(hour).padStart(2, "0")}Z ` +
      `within ±10 days of this date · current value (yellow) sits at P${Math.round(100 * below / vals.length)} ` +
      `of the same-hour sample`;
  } else {
    note.textContent = `distribution of ${vals.length} soundings at ${String(hour).padStart(2, "0")}Z within ±10 days of this date`;
  }
}
document.getElementById("climo-close").addEventListener("click", () => climoModal.hidden = true);
climoModal.addEventListener("click", e => { if (e.target === climoModal) climoModal.hidden = true; });
// (Escape handling is centralized in the topmost-overlay keydown handler below)
const CLIMO_PCTS = [1, 5, 10, 25, 50, 75, 90, 95, 99];
const CLIMO_MIN_N = 30;                  // a "record" from 4 soundings is noise
// Only interesting at the top end: a sounding with no convective energy is in
// its normal state, so "record low ECAPE/SHIP" is noise, not news.
const CLIMO_HIGH_ONLY = new Set(["ecape", "ship"]);
// …and a value has to be big enough to MEAN anything. At Barrow the SHIP
// climatology is p10 = p50 = p90 = 0, so a SHIP of 0.01 scores "P99" — a record
// hail day in the Arctic. Percentiles are useless where the distribution is a
// spike at zero, so these indices also have to clear an absolute floor.
const CLIMO_FLOOR = { ecape: 100, ship: 0.5 };
function climoPct(key, v) {                          // -> {pct, rec} or null
  if (!climo || !isFinite(v)) return null;
  const A = climo.idx && climo.idx[key];
  const s = A ? climoSlot(A) : -1;
  if (!A || s < 0 || !A.p[s] || A.p[s][0] === null || (A.n[s] || 0) < CLIMO_MIN_N)
    return null;
  const d = { p: A.p[s], min: A.min[s], max: A.max[s],
              minY: A.minY[s], maxY: A.maxY[s], n: A.n[s] };
  const X = [d.min, ...d.p, d.max], Y = [0, ...CLIMO_PCTS, 100];
  let pct = v <= X[0] ? 0 : v >= X[X.length - 1] ? 100 : 50;
  for (let i = 1; i < X.length; i++) {
    if (v <= X[i]) { const f = (v - X[i - 1]) / ((X[i] - X[i - 1]) || 1);
      pct = Y[i - 1] + f * (Y[i] - Y[i - 1]); break; }
  }
  // A value tied with the bulk of the distribution is not an extreme. ECAPE is
  // zero on nearly every Antarctic sounding, so a zero there is the MEDIAN — yet
  // a naive percentile calls it P0. Require genuine separation from the middle.
  const degenerateLow = v >= d.p[3];      // not actually below the 25th
  const degenerateHigh = v <= d.p[5];     // not actually above the 75th
  if ((pct <= CLIMO_LO && degenerateLow) || (pct >= CLIMO_HI && degenerateHigh))
    return null;
  if (pct <= CLIMO_LO && CLIMO_HIGH_ONLY.has(key)) return null;   // low = a quiet day
  if (CLIMO_FLOOR[key] !== undefined && v < CLIMO_FLOOR[key]) return null;
  // no meaningful upper tail (the whole distribution is a spike): nothing to rank
  if (pct >= CLIMO_HI && !(d.p[6] > d.p[4])) return null;
  // A record carries WHAT it broke: the ±10-day mark by default, escalating to
  // the station's all-time extreme when it clears that too. All-time is exact —
  // every sounding lands inside some anchor window, so the envelope's max over
  // anchors is the true period-of-record extreme.
  let rec = null;
  if (v >= d.max && v > d.p[5]) {
    rec = { t: "high", y: d.maxY, prev: d.max };
    const at = climoAllTime(key, "high");
    if (at && v >= at.v) { rec.tier = "all"; rec.prev = at.v; rec.y = at.y; }
  } else if (v <= d.min && v < d.p[3]) {
    rec = { t: "low", y: d.minY, prev: d.min };
    const at = climoAllTime(key, "low");
    if (at && v <= at.v) { rec.tier = "all"; rec.prev = at.v; rec.y = at.y; }
  }
  return { pct: Math.max(0, Math.min(100, pct)), rec };
}
// station all-time extreme for one index: scan every anchor's envelope
function climoAllTime(key, sense) {
  const A = climo && climo.idx && climo.idx[key];
  if (!A) return null;
  let best = null;
  for (let i = 0; i < A.max.length; i++) {
    const v = sense === "high" ? A.max[i] : A.min[i];
    const y = sense === "high" ? A.maxY[i] : A.minY[i];
    if (v === null || v === undefined) continue;
    if (!best || (sense === "high" ? v > best.v : v < best.v)) best = { v, y };
  }
  return best;
}
// Only the tails are worth flagging: a value sitting at P53 is by definition
// unremarkable, and tagging it just adds noise. Nothing between p10 and p90 is
// colored or labelled; beyond that the tint ramps within the tail itself, so
// P91 is a whisper and P99 is loud.
const CLIMO_LO = 10, CLIMO_HI = 90;
function pctColor(pct) {
  const high = pct >= CLIMO_HI;
  const frac = high ? (pct - CLIMO_HI) / (100 - CLIMO_HI)   // 0 at p90 -> 1 at p100
                    : (CLIMO_LO - pct) / CLIMO_LO;          // 0 at p10 -> 1 at p0
  const a = 0.18 + 0.52 * Math.max(0, Math.min(1, frac));
  const c = high ? [224, 70, 55] : [56, 120, 216];          // red high / blue low
  return `rgba(${c[0]},${c[1]},${c[2]},${a.toFixed(2)})`;
}
// value cell HTML: tinted + labelled ONLY outside p10-p90; ★ + year on a record
function climoCell(key, v, txt) {
  const r = climoPct(key, v);
  if (!r) return txt;
  const notable = r.rec || r.pct >= CLIMO_HI || r.pct <= CLIMO_LO;
  if (!notable) return txt;                                  // unremarkable: plain
  const ord = Math.max(1, Math.min(99, Math.round(r.pct)));
  const tip = r.rec
    ? (r.rec.tier === "all"
        ? `ALL-TIME station record ${r.rec.t} — previous ${r.rec.prev} (${r.rec.y})`
        : `record ${r.rec.t} for this time of year (±10 days) — previous ${r.rec.prev} (${r.rec.y})`)
    : `${ord}th percentile for this time of year`;
  const star = r.rec
    ? ` <span style="color:${r.rec.t === "high" ? "#ff5a3c" : "#5a9bf0"}">${r.rec.tier === "all" ? "★★" : "★"}${String(r.rec.y).slice(2)}</span>`
    : "";
  const sub = r.rec ? "" : `<span class="pctlab">P${ord}</span>`;
  return `<span class="pctcell" style="background:${pctColor(r.pct)}" title="${tip}">${txt}${star}</span>${sub}`;
}

// ---- 🏆 station records drill-down: what stands, and what this sounding broke ----
function openRecords() {
  if (!climo) return;
  const fmtV = v => (v === null || v === undefined || !isFinite(v)) ? "—"
    : (+v).toFixed(Math.abs(v) >= 1000 ? 0 : 1);
  const cell = (r, sense, broken) => {
    if (!r) return "<td>—</td>";
    const bg = broken ? (sense === "high" ? "rgba(224,70,55,0.42)" : "rgba(56,120,216,0.45)") : "";
    return `<td${bg ? ` style="background:${bg}"` : ""}>${fmtV(r.v)}` +
           ` <span class="pctlab">${r.y}</span></td>`;
  };
  const rows = [], broke = [];
  for (const [k, name, unit] of CLIMO_VARS) {
    const A = climo.idx && climo.idx[k];
    if (!A) continue;
    // ECAPE/SHIP lows are quiet days, not records — blank their low columns
    const lowMeaningful = !CLIMO_HIGH_ONLY.has(k);
    const atHi = climoAllTime(k, "high");
    const atLo = lowMeaningful ? climoAllTime(k, "low") : null;
    if (!atHi) continue;
    const s = climoSlot(A);
    const wOk = s >= 0 && A.max[s] !== null && (A.n[s] || 0) >= CLIMO_MIN_N;
    const wHi = wOk ? { v: A.max[s], y: A.maxY[s] } : null;
    const wLo = wOk && lowMeaningful ? { v: A.min[s], y: A.minY[s] } : null;
    const cur = climoNow[k];
    // climoPct applies every honesty guard (degenerate tails, floors, spike
    // distributions) — the modal must not claim a record the tags wouldn't
    const rec = (isFinite(cur) ? (climoPct(k, cur) || {}).rec : null) || null;
    const lab = `${name}${unit ? ` <span class="pctlab">${unit}</span>` : ""}`;
    rows.push(`<tr><td>${lab}</td>` +
      cell(atLo, "low", rec && rec.t === "low" && rec.tier === "all") +
      cell(atHi, "high", rec && rec.t === "high" && rec.tier === "all") +
      cell(wLo, "low", rec && rec.t === "low") +
      cell(wHi, "high", rec && rec.t === "high") +
      `<td>${isFinite(cur) ? climoCell(k, cur, fmtV(cur)) : "—"}</td></tr>`);
    if (rec) broke.push(
      `<b>${name} ${fmtV(cur)}${unit ? " " + unit : ""}</b> — ` +
      (rec.tier === "all" ? "ALL-TIME station record" : "record for this time of year") +
      ` ${rec.t} (previous ${fmtV(rec.prev)}, ${rec.y})`);
  }
  document.getElementById("rec-title").textContent =
    `🏆 ${(current && current.n) || "Station"} records`;
  document.getElementById("rec-table").innerHTML =
    `<tr><th></th><th colspan="2">all-time ${climo.yr0}–${climo.yr1}</th>` +
    `<th colspan="2">this time of year (±10 d)</th><th>this sounding</th></tr>` +
    `<tr><th></th><th>low</th><th>high</th><th>low</th><th>high</th><th></th></tr>` +
    rows.join("");
  document.getElementById("rec-note").innerHTML =
    (broke.length ? `<span style="color:#ff9f95">⚡ This sounding sets: ` +
      broke.join("; ") + `.</span><br>` : "") +
    `Records from this station's ${climo.yr0}–${climo.yr1} radiosonde archive (00Z and 12Z ` +
    `pooled). "This time of year" pools soundings within ±10 days of the plotted date; ` +
    `all-time spans the whole period of record. The year shown is when the mark was set; ` +
    `tying a mark counts as a record. Highlighted cells are marks this sounding takes over.`;
  document.getElementById("rec-modal").hidden = false;
}
const MIRROR = "https://raw.githubusercontent.com/scorvec/scorvec.github.io/skewt-data/";
const SPC = "https://www.spc.noaa.gov/exper/soundings";
const IEM = "https://mesonet.agron.iastate.edu/json/raob.py";
let iemMap = null;                                  // WMO -> ICAO (US/Canada)
fetch("iem_raob.json").then(r => r.ok ? r.json() : {}).then(m => { iemMap = m; })
  .catch(() => { iemMap = {}; });

// Most recent synoptic slots, newest first, stepping every SIX hours. Stepping
// 12 h only ever looked at 00Z/12Z, so a station launching off-hour (06Z/18Z)
// was never found. A source that doesn't publish that hour simply 404s and we
// fall through to the next slot.
function synopticSlots(n = 5) {
  const out = [];
  const now = new Date();
  let d = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate(),
                            Math.floor(now.getUTCHours() / 6) * 6));
  for (let k = 0; k < n; k++) {
    out.push({ y: d.getUTCFullYear(), mo: d.getUTCMonth() + 1, d: d.getUTCDate(),
               hh: d.getUTCHours() });
    d = new Date(d.getTime() - 6 * 3600e3);
  }
  return out;
}
// hours since a manifest timestamp ("YYYY-MM-DD HH:MM", UTC)
function ageHours(dt) {
  if (!dt) return Infinity;
  const ms = Date.parse(dt.replace(" ", "T") + ":00Z");
  return isFinite(ms) ? (Date.now() - ms) / 3600e3 : Infinity;
}
// Blue = reported at the most recent DUE regular slot (00Z or 12Z). A slot
// becomes due 2.5 h after its nominal time (balloons take ~2 h to reach the
// archive); 1.2 h of slack below covers early releases stamped before the
// nominal hour. Hours since that due slot:
function dueAgeH() {
  const slot = 43200e3, grace = 2.5 * 3600e3;
  const now = Date.now();
  let tdue = Math.floor(now / slot) * slot;
  if (now - tdue < grace) tdue -= slot;
  return (now - tdue) / 3600e3;
}
// A launch outside the 00Z/12Z routine is usually a SPECIAL release — an extra
// sounding fired ahead of severe weather or to sample a fast-evolving system —
// so it's worth spotting on the map rather than blending in with the synoptic
// crowd.
function launchHour(dt) {
  const m = /\s(\d{2}):/.exec(dt || "");
  return m ? parseInt(m[1], 10) : null;
}
const isOffHour = dt => { const h = launchHour(dt); return h !== null && h !== 0 && h !== 12; };
const p2 = n => String(n).padStart(2, "0");

// SPC/IEM report wind on only some levels; zero-filling the rest would drag the
// hodograph to the origin, so interpolate U/V across the gaps in log-p.
function fillWinds(o) {
  const n = o.P.length, ok = [];
  for (let i = 0; i < n; i++) if (isFinite(o.U[i]) && isFinite(o.V[i])) ok.push(i);
  if (!ok.length) { for (let i = 0; i < n; i++) { o.U[i] = 0; o.V[i] = 0; } return 0; }
  for (let i = 0; i < n; i++) {
    if (isFinite(o.U[i]) && isFinite(o.V[i])) continue;
    let lo = null, hi = null;
    for (const j of ok) { if (j < i) lo = j; else { hi = j; break; } }
    if (lo === null) { o.U[i] = o.U[ok[0]]; o.V[i] = o.V[ok[0]]; }
    else if (hi === null) { o.U[i] = o.U[lo]; o.V[i] = o.V[lo]; }
    else {
      const f = Math.log(o.P[lo] / o.P[i]) / Math.log(o.P[lo] / o.P[hi]);
      o.U[i] = o.U[lo] + f * (o.U[hi] - o.U[lo]);
      o.V[i] = o.V[lo] + f * (o.V[hi] - o.V[lo]);
    }
  }
  return ok.length;
}

// SPC observed soundings — CORS-open and the fastest public US source (it
// carries 12Z about an hour before IEM does, at slightly coarser resolution)
function spcProfile(text) {
  if (!text || !text.includes("%RAW%")) return null;
  const tm = text.match(/([0-9]{6})\/([0-9]{4})/);
  const raw = text.split("%RAW%")[1].split("%END%")[0];
  const o = { P: [], H: [], T: [], D: [], U: [], V: [] };
  for (const line of raw.trim().split("\n")) {
    const c = line.split(",").map(s => parseFloat(s));
    if (c.length < 6) continue;
    const [pp, hh, tc, dc, wd, ws] = c;
    if (!(pp > 20) || !(tc > -9990)) continue;       // temperature required
    o.P.push(pp * 100);
    o.H.push(hh > -9990 ? hh : NaN);
    o.T.push(tc + 273.15);
    o.D.push(dc > -9990 ? dc + 273.15 : NaN);
    if (ws > -9990 && wd > -9990) {
      const wsms = ws / KT;
      o.U.push(-wsms * Math.sin(wd * Math.PI / 180));
      o.V.push(-wsms * Math.cos(wd * Math.PI / 180));
    } else { o.U.push(NaN); o.V.push(NaN); }   // filled below, never zeroed
  }
  if (o.P.length < 8) return null;
  fillWinds(o);
  const valid = tm ? `20${tm[1].slice(0, 2)}-${tm[1].slice(2, 4)}-${tm[1].slice(4, 6)} ` +
    `${tm[2].slice(0, 2)}:00` : "";
  return { prof: o, valid, src: "SPC real-time" };
}
async function fetchSPC(wmo) {
  const icao = iemMap && iemMap[wmo];
  if (!icao || icao[0] !== "K") return null;         // SPC OBS = US sites
  const id3 = icao.slice(1);
  for (const s of synopticSlots()) {
    const stamp = String(s.y).slice(2) + p2(s.mo) + p2(s.d) + p2(s.hh);
    try {
      const r = await fetch(`${SPC}/${stamp}_OBS/${id3}.txt`);
      if (!r.ok) continue;
      const got = spcProfile(await r.text());
      if (got) return got;
    } catch (e) { /* older slot */ }
  }
  return null;
}

// IEM RAOB json — CORS-open, US + Canada, more levels but ~1 h behind SPC
function iemProfile(j) {
  const pr = j && j.profiles && j.profiles[0];
  if (!pr || !pr.profile) return null;
  const o = { P: [], H: [], T: [], D: [], U: [], V: [] };
  for (const L of pr.profile) {
    if (L.pres == null || L.tmpc == null || L.pres < 20) continue;
    o.P.push(L.pres * 100);
    o.H.push(L.hght == null ? NaN : L.hght);
    o.T.push(L.tmpc + 273.15);
    o.D.push(L.dwpc == null ? NaN : L.dwpc + 273.15);
    if (L.sknt != null && L.drct != null) {
      const ws = L.sknt / KT, wd = L.drct;
      o.U.push(-ws * Math.sin(wd * Math.PI / 180));
      o.V.push(-ws * Math.cos(wd * Math.PI / 180));
    } else { o.U.push(NaN); o.V.push(NaN); }
  }
  if (o.P.length < 8) return null;
  fillWinds(o);
  return { prof: o, valid: (pr.valid || "").slice(0, 16).replace("T", " "),
           src: "IEM real-time" };
}
async function fetchIEM(wmo) {
  const icao = iemMap && iemMap[wmo];
  if (!icao) return null;
  for (const s of synopticSlots()) {
    const ts = `${s.y}${p2(s.mo)}${p2(s.d)}${p2(s.hh)}00`;
    try {
      const r = await fetch(`${IEM}?ts=${ts}&station=${icao}`);
      if (!r.ok) continue;
      const got = iemProfile(await r.json());
      if (got) return got;
    } catch (e) { /* older slot */ }
  }
  return null;
}
const IGRA = "https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/access/";
const UW_ARCHIVE = "https://raw.githubusercontent.com/scorvec/scorvec.github.io/skewt-archive/";
const UW_ARCHIVE_START = "2026-07-10";      // day bundles exist from here on
const dayZipCache = new Map();              // YYYYMMDD -> {filename: Uint8Array} | null
let M = null;                    // wasm module
let entries = {};                // mirror manifest: id -> {n, la, lo, dt, src}
let anomalies = {};              // wmo -> {dt, flags:[{k,lab,v,pct,sense}]} (record watch)
let igraStations = {};           // gid -> station meta (all 2,921 incl. closed)
let byWmo = {};                  // wmo id -> gid
let current = null;              // selected: {gid, id, n, e}
let plotTitle = "", plotCoords = "";
let pinned = null;                 // { prof, title } — overlay for comparison              // drawn on the skew-t canvas itself
let plotNote = "";               // secondary blurb (e.g. wind-only)
let selectedMarker = null;       // highlighted dot on the map
let mode = "latest";
let archHour = 12;
let archDate = new Date(Date.now() - 3 * 864e5).toISOString().slice(0, 10);
const igraCache = new Map();     // gid -> decompressed text

// ---------- wasm ----------
// Version the .wasm URL as well: sharplib.js resolves it itself, so without
// this a cached WASM could pair with fresh JS — mismatched out[] indices and
// silently wrong numbers, which is far worse than a crash. WASM_V is stamped by
// scripts/skewt/stamp_assets.py.
const WASM_V = "66a5af64";
const wasmReady = createSharp({
  locateFile: (f) => f.endsWith(".wasm") ? `${f}?v=${WASM_V}` : f,
}).then(mod => { M = mod; })
  .catch(e => { wasmFailed = String(e).slice(0, 120); });
let wasmFailed = null;

function f32(arr) {
  const p = M._malloc(arr.length * 4);
  M.HEAPF32.set(arr instanceof Float32Array ? arr : new Float32Array(arr), p / 4);
  return p;
}

// ---------- station map ----------
const COARSE = window.matchMedia && matchMedia("(pointer:coarse)").matches;
const RAD = COARSE ? { closed: 5, active: 7, live: 10 } : { closed: 2.5, active: 4, live: 6 };
const map = L.map("map", { worldCopyJump: true, preferCanvas: true }).setView([30, -10], 3);
// mobile nav wraps after fonts/layout settle, resizing the map container —
// Leaflet must re-measure or tiles/markers render for stale geometry
addEventListener("resize", () => map.invalidateSize());
addEventListener("orientationchange", () => setTimeout(() => map.invalidateSize(), 200));
if (window.ResizeObserver) new ResizeObserver(() => map.invalidateSize())
  .observe(document.querySelector(".mapwrap"));
setTimeout(() => map.invalidateSize(), 400);
let redrawTimer = null;
function redrawCharts() {
  if (modal.hidden || !lastProf || !lastRes) return;
  inRender = true;                          // we draw right below — no double draw
  try { fitLayout(); } finally { inRender = false; }
  drawSkewT(lastProf, lastRes); drawHodo(lastProf, lastRes); drawMSE(lastProf);
}
addEventListener("resize", () => { clearTimeout(redrawTimer); redrawTimer = setTimeout(redrawCharts, 120); });
if (window.ResizeObserver) new ResizeObserver(() => {
  clearTimeout(redrawTimer); redrawTimer = setTimeout(redrawCharts, 120);
}).observe(document.querySelector(".main"));
const modal = document.getElementById("modal");
function setStatus(text, busy = false) {
  for (const id of ["status", "mstatus"]) {
    const el = document.getElementById(id);
    if (el) { el.textContent = text; el.classList.toggle("busy", busy); }
  }
}
function openModal() { modal.hidden = false; sizePlots(); }
// give both plots their layout size before anything is drawn (a freshly
// opened card would otherwise show the canvases at their attribute size)
function sizePlots() { for (const id of ["skewt", "hodo"]) plotBox(document.getElementById(id)); }
function closeModal() { modal.hidden = true; }
document.getElementById("close").onclick = closeModal;
modal.addEventListener("click", e => { if (e.target === modal) closeModal(); });
addEventListener("keydown", e => {
  if (e.key !== "Escape") return;
  // close only the TOPMOST visible overlay per keystroke — stacked dialogs
  // (climo/cloud/fog/records/anomaly/png over the sounding) peel one at a time
  for (const id of ["png-view", "cloud-modal", "fog-modal", "rec-modal", "anom-modal"]) {
    const el = document.getElementById(id);
    if (el && !el.hidden) { el.hidden = true; return; }
  }
  if (typeof climoModal !== "undefined" && !climoModal.hidden) { climoModal.hidden = true; return; }
  closeModal();
});
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
  { attribution: "&copy; OpenStreetMap", maxZoom: 10 }).addTo(map);

const closedLayer = L.layerGroup();          // stations with no data since 2024
const ACTIVE_YEAR = new Date().getUTCFullYear() - 1;   // "active" = reported within ~a year

// Long-lived tabs and blocked networks both used to fail SILENTLY: the
// manifest/record-watch load once, so a tab open since yesterday shows
// yesterday's map, and a network that blocks raw.githubusercontent.com (the
// host behind the mirror, climatology AND record watch) just looks broken.
let feedsLoadedAt = 0;
function feedBanner(msg) {
  let el = document.getElementById("feed-banner");
  if (!msg) { if (el) el.remove(); return; }
  if (!el) {
    el = document.createElement("div");
    el.id = "feed-banner";
    el.style.cssText = "position:absolute; top:10px; left:50%; transform:translateX(-50%);" +
      "z-index:950; background:rgba(120,35,35,0.94); color:#ffe0dc; font:600 13px Inter;" +
      "padding:8px 16px; border-radius:9px; cursor:pointer; max-width:min(92%,640px)";
    el.title = "click to reload";
    el.addEventListener("click", () => location.reload());
    document.querySelector(".mapwrap").appendChild(el);
  }
  el.textContent = msg;
}
// a tab that slept through the night: offer a reload instead of stale silence
setInterval(() => {
  if (feedsLoadedAt && Date.now() - feedsLoadedAt > 2.5 * 3600e3)
    feedBanner("This page was loaded " +
      Math.round((Date.now() - feedsLoadedAt) / 3600e3) +
      " h ago — newer soundings exist. Click to refresh.");
}, 5 * 60e3);

Promise.all([
  fetch("stations.json").then(r => r.json()),
  ghFetch("skewt-data", "manifest.json", Date.now()).then(r => r.ok ? r.json() : null).catch(() => null),
  ghFetch("skewt-data", "anomalies.json", Date.now()).then(r => r.ok ? r.json() : {}).catch(() => ({})),
]).then(([stns, man, anom]) => {
  entries = (man && man.entries) || {};
  anomalies = anom || {};
  feedsLoadedAt = Date.now();
  if (!man)
    feedBanner("⚠ The live data feed is unreachable from this network (all hosts " +
      "tried: raw.githubusercontent.com, api.github.com, cdn.jsdelivr.net) — live dots " +
      "and climatology are unavailable; US real-time and archive browsing still work. " +
      "Click to retry.");
  for (const s of stns.stations) {
    igraStations[s.gid] = s;
    if (s.id) byWmo[s.id] = s.gid;
  }
  // A station sits in the mirror manifest for up to 96 h (retention for the
  // per-launch archive), so "in the manifest" is NOT the same as "reported
  // recently" — Glasgow showed as live on a launch 36 h old. Live means it
  // reported at the last regular interval (00Z/12Z).
  const isLive = id => id && entries[id] &&
    ageHours(entries[id].dt) <= dueAgeH() + 1.2;

  // closed stations: archive-only, hidden behind the toggle. Single-year
  // records are one-off pilot-balloon campaigns (63 of them) — map clutter
  // with nothing meaningful behind the dot, so they're scrubbed entirely.
  for (const s of stns.stations) {
    if (s.y1 >= ACTIVE_YEAR || (s.id && entries[s.id])) continue;
    if (s.y1 <= s.y0) continue;
    const m = L.circleMarker([s.la, s.lo], {
      radius: RAD.closed, weight: 0.5, color: "#7a1f1f", fillColor: "#c0392b", fillOpacity: 0.75,
    }).addTo(closedLayer);
    m.bindTooltip(`${s.n} (${s.gid}) · closed ${s.y0}–${s.y1} — click for archive`);
    m.on("click", () => { setMode("archive"); highlight(m); selectStation(s); });
  }
  // active stations that missed the last regular slot: small teal (includes
  // stations still in the manifest whose newest sounding is older than that)
  for (const s of stns.stations) {
    if (s.y1 < ACTIVE_YEAR || isLive(s.id)) continue;
    const stale = s.id && entries[s.id] ? entries[s.id].dt : null;
    const m = L.circleMarker([s.la, s.lo], {
      radius: RAD.active, weight: 1, color: "#1f7a5e", fillColor: "#33c495", fillOpacity: 0.85,
    }).addTo(map);
    m.bindTooltip(stale
      ? `${s.n} (${s.gid}) · last sounding ${stale}Z (${Math.round(ageHours(stale))} h ago) — click for archive`
      : `${s.n} (${s.gid}) · nothing at the last regular interval — click for archive (${s.y0}–${s.y1})`);
    m.on("click", () => { highlight(m); selectStation(s); });
  }
  // stations that reported at the last regular interval: big blue, on top
  for (const [id, s] of Object.entries(entries)) {
    if (!isLive(id)) continue;                    // stale carry-forward: not live
    const ig = igraStations[byWmo[id]];
    if (ig || /dtype|Name:/i.test(s.n || "")) s.n = (ig && ig.n) || id;
    const flag = anomalies[id];
    if (flag) {                                          // record watch: red ring underneath
      L.circleMarker([s.la, s.lo], { radius: RAD.live + 4, weight: 3,
        color: "#ff2d2d", opacity: 0.95, fill: false }).addTo(map);
    }
    const off = isOffHour(s.dt);                         // 06Z / 18Z special release
    const m = L.circleMarker([s.la, s.lo], {
      radius: off ? RAD.live + 1 : RAD.live, weight: 1.5,
      color: off ? "#7a3fa0" : "#1d3a5e",
      fillColor: off ? "#bf5af2" : "#4a7ab5", fillOpacity: 0.95,
    }).addTo(map);
    const arch = ig ? ` · archive ${ig.y0}–${ig.y1}` : "";
    const anomTip = flag ? `<br><b style="color:#ff2d2d">⚡ record watch:</b> ` +
      flag.flags.map(f => f.rec
        ? `${f.lab} ${f.v} — <b>${f.rec.tier === "all" ? "ALL-TIME RECORD" : "RECORD for the date"} ${f.rec.t}</b> (prev ${f.rec.prev}, ${f.rec.y})`
        : `${f.lab} ${f.v} (P${f.pct} ${f.sense})`).join(", ") : "";
    const offTip = off
      ? ` <b style="color:#bf5af2">· off-hour launch (${String(launchHour(s.dt)).padStart(2, "0")}Z)</b>`
      : "";
    m.bindTooltip(`${s.n || id} (${id}) · latest ${s.dt}Z ` +
      `(${Math.round(ageHours(s.dt))} h ago)${offTip}${arch}${anomTip}`);
    m.on("click", () => {
      highlight(m);   // respects the current Latest/Archive mode + chosen date
      selectStation({ gid: byWmo[id], id, n: s.n, e: (igraStations[byWmo[id]] || {}).e || 0 });
    });
  }
  if (!man) setStatus("live mirror unavailable — archive mode still works");
  // Deep links: #72249 opens the latest; #72249/2023-08-16/12 opens that exact
  // archived sounding. Browsing never writes the URL (user preference) — links
  // are minted only by the share button.
  const frag = location.hash.replace("#", "").split("/");
  const want = frag[0];
  if (want && (entries[want] || byWmo[want])) {
    if (frag.length >= 2 && /^\d{4}-\d{2}-\d{2}$/.test(frag[1])) {
      archDate = frag[1];
      archHour = Math.max(0, Math.min(23, parseInt(frag[2], 10) || 0));
      mode = "archive";
      syncControls();
    }
    selectStation({ gid: byWmo[want], id: want,
                    n: (entries[want] || {}).n || (igraStations[byWmo[want]] || {}).n,
                    e: (igraStations[byWmo[want]] || {}).e || 0 });
  }
});

// mint a shareable link for whatever is on screen
function shareLink() {
  if (!current || !current.id) return;
  const base = "https://scorvec.com/skewt/#" + current.id;
  const url = mode === "archive"
    ? `${base}/${archDate}/${String(archHour).padStart(2, "0")}`
    : base;
  navigator.clipboard.writeText(url).then(() => {
    const b = document.getElementById("share-btn");
    if (b) { const was = b.textContent; b.textContent = "✓ copied";
             setTimeout(() => { b.textContent = was; }, 1400); }
  }).catch(() => { prompt("Copy this link:", url); });
}
document.getElementById("share-btn").addEventListener("click", shareLink);

// ---- "extreme today": rank the record watch and make it a destination ----
function buildAnomPanel() {
  const el = document.getElementById("anom-list");
  if (!el) return;
  const ranked = Object.entries(anomalies)
    .map(([wmo, d]) => ({ wmo, d, top: Math.max(...d.flags.map(f => Math.abs(f.pct - 50))) }))
    .sort((a, b) => b.top - a.top).slice(0, 12);
  if (!ranked.length) { el.innerHTML = "<li>nothing unusual right now</li>"; return; }
  el.innerHTML = ranked.map(({ wmo, d }) => {
    const name = (entries[wmo] && entries[wmo].n) || (igraStations[byWmo[wmo]] || {}).n || wmo;
    const f = d.flags[0];
    return `<li data-wmo="${wmo}"><b>${name}</b><br>` +
      d.flags.slice(0, 2).map(g =>
        `<span class="${g.sense === "high" ? "hi" : "lo"}">${g.lab} ${g.v} · ` +
        `${g.rec ? (g.rec.tier === "all" ? "ALL-TIME REC" : "RECORD") : "P" + g.pct}</span>`
      ).join(" &nbsp; ") + `</li>`;
  }).join("");
  el.querySelectorAll("li[data-wmo]").forEach(li => li.addEventListener("click", () => {
    const wmo = li.dataset.wmo;
    // the record modal stacks ABOVE the sounding dialog — close it first,
    // or the sounding opens invisibly underneath
    document.getElementById("anom-modal").hidden = true;
    setMode("latest");
    selectStation({ gid: byWmo[wmo], id: wmo,
      n: (entries[wmo] || {}).n, e: (igraStations[byWmo[wmo]] || {}).e || 0 });
  }));
}
// ---- the record-watch MAP: a static, labeled snapshot of today's extremes ----
let coastSegs = null;                        // cached 110m coastline polylines
async function loadCoast() {
  if (coastSegs) return coastSegs;
  try {
    const r = await fetch("coast110.json");
    coastSegs = (await r.json()).segs;
  } catch (e) { coastSegs = []; }            // map still works, just bare
  return coastSegs;
}

// The record MAP labels stick to the state variables — temps, dews, moisture,
// heights/thickness. ECAPE and friends stay on the dots and the ranked list.
const REC_MAP_KEYS = new Set(["850t", "700t", "500t", "850td", "700td",
                              "pwat", "h500", "thick", "850spd", "250spd"]);
const REC_MAP_CLON = -100;                   // map centered on 100°W
async function drawRecordMap() {
  const cv = document.getElementById("anom-canvas"), ctx = cv.getContext("2d");
  // size the backing store to the dialog's real width so wide monitors get a
  // wide map (canvas CSS is width:100%, so aspect follows these dimensions)
  const panelW = cv.parentElement ? cv.parentElement.clientWidth - 24 : 0;
  const W = Math.max(900, Math.min(2200, Math.round(panelW || 1240)));
  const H = Math.round(Math.min(820, Math.max(560, W * 0.42)));
  // retina-sharp: DPR-scaled backing store, logical-pixel drawing (like fitCanvas)
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  await loadCoast();
  // longitudes as signed offsets from the center meridian (seam at 80°E)
  const offLon = lon => ((lon - REC_MAP_CLON + 540) % 360) - 180;

  // stations to feature: true records labeled; near-records as context dots
  const pts = [];
  for (const [wmo, d] of Object.entries(anomalies)) {
    const e = entries[wmo];
    if (!e || !isFinite(e.la) || !isFinite(e.lo)) continue;
    const name = ((e.n || wmo) + "").split(/[;,]/)[0].trim();
    const recs = d.flags.filter(f => f.rec && REC_MAP_KEYS.has(f.k));
    pts.push({ wmo, la: e.la, lo: offLon(e.lo), name, recs, flags: d.flags, dt: d.dt,
               top: Math.max(...d.flags.map(f => Math.abs(f.pct - 50))) });
  }
  if (!pts.length) {
    ctx.fillStyle = "#f4f5f7"; ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = "#5b6270"; ctx.font = "600 16px Inter"; ctx.textAlign = "center";
    ctx.fillText("nothing unusual on the record watch right now", W / 2, H / 2);
    ctx.textAlign = "left";
    return;
  }
  // age-out: a record stays flagged until its station launches again, so a
  // 4-day-old record from an infrequent launcher (a proving ground, a campaign
  // site) lingers and makes the whole map read as stale. Only label records
  // whose sounding is within 48 h of the newest one; older record-holders
  // stay visible as faded stars (no label).
  const parseDt = s => Date.parse((s || "").replace(" ", "T") + "Z");
  const newestT = Math.max(...pts.map(p => parseDt(p.dt)).filter(isFinite));
  const freshP = p => !isFinite(newestT) || !isFinite(parseDt(p.dt)) ||
                      newestT - parseDt(p.dt) <= 48 * 3600e3;
  const recHolders = pts.filter(p => p.recs.length)
    .sort((a, b) =>
      b.recs.filter(r => r.rec.tier === "all").length -
      a.recs.filter(r => r.rec.tier === "all").length || b.top - a.top);
  const labeled = recHolders.filter(freshP);
  const aged = recHolders.filter(p => !freshP(p));
  // Nothing broke a record today (or anomalies.json predates the rec field):
  // label the highest-percentile extremes instead, so the map is never empty.
  const fallback = !labeled.length;
  const featured = (fallback ? pts.filter(freshP).sort((a, b) => b.top - a.top) : labeled).slice(0, 14);

  // extent: fit the featured stations, pad, enforce a minimum window
  const ext = featured.length ? featured : pts;
  let lo0 = Math.min(...ext.map(p => p.lo)), lo1 = Math.max(...ext.map(p => p.lo));
  let la0 = Math.min(...ext.map(p => p.la)), la1 = Math.max(...ext.map(p => p.la));
  lo0 -= 10; lo1 += 10; la0 -= 7; la1 += 7;
  // keep 100°W on the vertical centerline: symmetric span about offset 0
  lo1 = Math.min(180, Math.max(Math.abs(lo0), Math.abs(lo1), 27.5)); lo0 = -lo1;
  if (la1 - la0 < 30) { const c = (la0 + la1) / 2; la0 = c - 15; la1 = c + 15; }
  la0 = Math.max(-85, la0); la1 = Math.min(85, la1);
  const M = { l: 10, r: 10, t: 46, b: 22 };
  const pw = W - M.l - M.r, ph = H - M.t - M.b;
  // equirectangular, aspect-preserving (1° lat ≈ 1° lon at the extent's center)
  const kLon = Math.cos(((la0 + la1) / 2) * Math.PI / 180);
  const sc = Math.min(pw / ((lo1 - lo0) * kLon), ph / (la1 - la0));
  // fill the canvas: widen the lon window (symmetric about the center meridian)
  // and the lat window to consume any letterbox slack with more map context
  lo1 = Math.min(180, (pw / (kLon * sc)) / 2); lo0 = -lo1;
  { const c = (la0 + la1) / 2, half = (ph / sc) / 2;
    la0 = Math.max(-85, c - half); la1 = Math.min(85, c + half); }
  const ox = M.l + (pw - (lo1 - lo0) * kLon * sc) / 2;
  const oy = M.t + (ph - (la1 - la0) * sc) / 2;
  const X = lon => ox + (lon - lo0) * kLon * sc;
  const Y = lat => oy + (la1 - lat) * sc;

  ctx.fillStyle = "#f4f5f7"; ctx.fillRect(0, 0, W, H);
  ctx.save();
  ctx.beginPath(); ctx.rect(M.l, M.t, pw, ph); ctx.clip();
  ctx.strokeStyle = "#9aa2ad"; ctx.lineWidth = 1;
  for (const seg of coastSegs || []) {
    ctx.beginPath(); let st = false, prev = null;
    for (const [rawLn, lt] of seg) {
      const ln = offLon(rawLn);
      if (prev !== null && Math.abs(ln - prev) > 180) st = false;   // seam crossing
      prev = ln;
      if (ln < lo0 - 5 || ln > lo1 + 5 || lt < la0 - 5 || lt > la1 + 5) { st = false; continue; }
      const x = X(ln), y = Y(lt);
      st ? ctx.lineTo(x, y) : ctx.moveTo(x, y); st = true;
    }
    ctx.stroke();
  }
  // context: every live station as a faint dot, flagged near-records brighter
  ctx.fillStyle = "rgba(90,98,110,0.35)";
  for (const [, s] of Object.entries(entries))
    if (isFinite(s.la) && isFinite(s.lo)) { ctx.beginPath(); ctx.arc(X(offLon(s.lo)), Y(s.la), 1.4, 0, 7); ctx.fill(); }
  for (const p of pts) {
    const hi = p.flags[0].sense === "high";
    ctx.fillStyle = hi ? "rgba(200,68,44,0.75)" : "rgba(47,107,179,0.75)";
    ctx.beginPath(); ctx.arc(X(p.lo), Y(p.la), 2.4, 0, 7); ctx.fill();
  }

  // featured stations: star + collision-avoided label with a leader line
  const placed = [];    // occupied label rects
  const star = (x, y, r, col) => {
    ctx.fillStyle = col; ctx.beginPath();
    for (let i = 0; i < 10; i++) {
      const a = -Math.PI / 2 + i * Math.PI / 5, rr = i % 2 ? r * 0.45 : r;
      ctx[i ? "lineTo" : "moveTo"](x + rr * Math.cos(a), y + rr * Math.sin(a));
    }
    ctx.closePath(); ctx.fill();
  };
  // aged-out record holders (>48 h): faded star, no label — still on the map,
  // but they no longer pin days-old text that makes the watch look frozen
  for (const p of aged) {
    const hi = (p.recs[0] || p.flags[0]).sense !== "low";
    star(X(p.lo), Y(p.la), 4, hi ? "rgba(200,68,44,0.35)" : "rgba(47,107,179,0.35)");
  }
  const lineH = 12;
  const GAP = 8;        // min clearance between label boxes (and around stars)
  // reserve every featured star's spot up front so no label covers a star
  for (const p of featured)
    placed.push({ x: X(p.lo) - 7, y: Y(p.la) - 7, w: 14, h: 14 });
  for (const p of featured) {
    const x = X(p.lo), y = Y(p.la);
    const hi = (p.recs[0] || p.flags[0]).sense !== "low";
    const col = hi ? "#c8442c" : "#2f6bb3";
    star(x, y, 5.5, col);
    const shown = (fallback ? p.flags : p.recs).slice(0, 2);
    // each label carries ITS sounding's launch time — records persist until
    // the station launches again, so without this two views hours apart look
    // identical and the map reads as stale when it isn't
    const hh = p.dt ? p.dt.slice(5, 10).replace("-", "/") + " " + p.dt.slice(11, 13) + "Z" : "";
    const lines = [p.name + (hh ? "  ·  " + hh : "")].concat(shown.map(f => f.rec
      ? `${f.lab} ${f.v} · ${f.rec.tier === "all" ? "ALL-TIME" : "date rec"} (prev ${f.rec.prev} '${String(f.rec.y).slice(2)})`
      : `${f.lab} ${f.v} · P${f.pct} ${f.sense}`));
    ctx.font = "600 10.5px Inter";
    const wpx = Math.max(...lines.map(s => ctx.measureText(s).width)) + 10;
    const hpx = lines.length * lineH + 7;
    // candidate anchor points around the station, nearest first
    let spot = null;
    outer:
    for (const rad of [16, 34, 56, 84, 120]) {
      for (const ang of [0, 180, 315, 225, 45, 135, 270, 90]) {
        const ax = x + rad * Math.cos(ang * Math.PI / 180);
        const ay = y + rad * Math.sin(ang * Math.PI / 180);
        const rx = ang > 90 && ang < 270 ? ax - wpx : ax;   // left side: right-align
        const ry = ay - hpx / 2;
        if (rx < M.l || rx + wpx > W - M.r || ry < M.t || ry + hpx > H - M.b) continue;
        if (placed.some(q => rx < q.x + q.w + GAP && rx + wpx + GAP > q.x &&
                             ry < q.y + q.h + GAP && ry + hpx + GAP > q.y)) continue;
        spot = { x: rx, y: ry, ax, ay };
        break outer;
      }
    }
    if (!spot) continue;                               // too crowded: star only
    placed.push({ x: spot.x, y: spot.y, w: wpx, h: hpx });
    ctx.strokeStyle = "rgba(60,64,76,0.5)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(spot.ax, spot.ay); ctx.stroke();
    ctx.fillStyle = "rgba(255,255,255,0.92)";
    ctx.fillRect(spot.x, spot.y, wpx, hpx);
    ctx.strokeStyle = col; ctx.lineWidth = 0.8; ctx.strokeRect(spot.x, spot.y, wpx, hpx);
    lines.forEach((s, i) => {
      ctx.fillStyle = i ? (s.includes("ALL-TIME") ? "#9a6b00" : "#2a2d35") : col;
      ctx.font = i ? "10px Inter" : "600 10.5px Inter";
      ctx.fillText(s, spot.x + 5, spot.y + 11.5 + i * lineH);
    });
  }
  ctx.restore();

  // title + footer
  const newest = pts.map(p => p.dt).sort().pop() || "";
  const nRec = labeled.length;
  const nAll = labeled.filter(p => p.recs.some(r => r.rec.tier === "all")).length;
  ctx.fillStyle = "#1c1f26"; ctx.font = "700 17px Inter"; ctx.textAlign = "left";
  ctx.fillText(`⚡ Radiosonde record watch — newest sounding ${newest}Z`, 14, 25);
  ctx.fillStyle = "#5b6270"; ctx.font = "11.5px Inter";
  ctx.fillText(fallback
    ? `no station records today — ${pts.length} stations at climatological extremes (P95+)`
    : `${nRec} station(s) set records vs their own sounding archives` +
      (nAll ? ` — ${nAll} ALL-TIME` : "") + ` · ${pts.length} stations flagged P95+` +
      (aged.length ? ` · ${aged.length} older record(s) faded (>48 h)` : "") +
      ` · each label shows its station's latest launch`, 14, 41);
  ctx.font = "11px Inter";
  ctx.fillText("scorvec.com/skewt · records vs each station's full IGRA archive, ±10-day time-of-year window",
    14, H - 7);
}

document.getElementById("anom-toggle").addEventListener("click", async () => {
  document.getElementById("anom-modal").hidden = false;
  // the record watch regenerates 6×/day — refetch when this tab's copy is
  // older than 10 min, so a long-open page never shows yesterday's records
  if (Date.now() - feedsLoadedAt > 10 * 60e3) {
    try {
      const r = await ghFetch("skewt-data", "anomalies.json", Date.now());
      if (r.ok) anomalies = await r.json();
      const m = await ghFetch("skewt-data", "manifest.json", Date.now());
      if (m.ok) {
        const man = await m.json();
        if (man && man.entries) entries = man.entries;   // fresh coords + dts
      }
      feedsLoadedAt = Date.now();
      feedBanner(null);
    } catch (e) { /* keep showing what we have */ }
  }
  buildAnomPanel();
  drawRecordMap();
});
document.getElementById("anom-close").addEventListener("click",
  () => document.getElementById("anom-modal").hidden = true);
document.getElementById("anom-modal").addEventListener("click", e => {
  if (e.target.id === "anom-modal") document.getElementById("anom-modal").hidden = true;
});
document.getElementById("anom-save").addEventListener("click", () => {
  document.getElementById("anom-canvas").toBlob(b => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(b);
    a.download = "record_watch.png";
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 5000);
  });
});

// ---- PNG export: skew-T + hodograph on one branded card ----
function buildExportCanvas(fscale = 1) {
  const sk = document.getElementById("skewt"), ho = document.getElementById("hodo");
  const F = n => Math.round(n * fscale);
  const pad = F(24), foot = F(54);

  // The indices live in HTML tables, so the export re-typesets them from the
  // DOM — whatever is on screen (pairings, percentile tags, records) is exactly
  // what lands in the image.
  const grab = id => [...document.querySelectorAll(`#${id} tr`)]
    .map(tr => [...tr.children].map(td =>
      [...td.childNodes].map(n => n.textContent).join(" ").replace(/\s+/g, " ").trim()));
  const columns = [                       // the card's own 5-column layout
    [["Parcels", grab("pcl-table")], ["Moist static energy", grab("mse-table")]],
    [["Kinematics", grab("kin-table")]],
    [["Composites", grab("kin-table-b")], ["Winter", grab("winter-table")]],
    [["Thermo & moisture", grab("kin-table2")], ["Tropical", grab("tropic-table")]],
    [["Levels", grab("kin-table3")]],
  ];
  const ROW = F(19), TITLE = F(30), GAP = F(10);
  // percentile/record tags ("P95", "\u2605 1988") drawn smaller and colored so
  // "5914 m P95" reads as value + tag, not one number
  const drawValCell = (txt, xr, yb, basePx) => {
    const m = /^(.*?)\s*((?:P\d{1,3})|(?:\u2605\s*\S*))$/.exec(txt);
    if (!m || !m[1]) { x.fillText(txt, xr, yb); return; }
    const prevFont = x.font, prevFill = x.fillStyle;
    x.font = `600 ${F(basePx - 3.5)}px Inter, sans-serif`;
    x.fillStyle = m[2][0] === "\u2605" ? "#ffd60a" : "#7f8ab8";
    const tw = x.measureText(m[2]).width;
    x.fillText(m[2], xr, yb);
    x.font = prevFont; x.fillStyle = prevFill;
    x.fillText(m[1], xr - tw - F(6), yb);
  };
  const colH = col => col.reduce((s, [, rows]) => s + TITLE + rows.length * ROW + GAP, 0);
  const tableH = Math.max(...columns.map(colH)) + 18;

  const W = sk.width + ho.width + pad * 3;
  const H = Math.max(sk.height, ho.height) + pad * 2 + tableH + foot;
  const cv = document.createElement("canvas");
  cv.width = W; cv.height = H;
  const x = cv.getContext("2d");
  x.fillStyle = "#0b0b12"; x.fillRect(0, 0, W, H);
  x.drawImage(sk, pad, pad);
  x.drawImage(ho, sk.width + pad * 2, pad);

  // typeset the index columns
  const tTop = Math.max(sk.height, ho.height) + pad * 2;
  x.strokeStyle = "#23233a";
  x.beginPath(); x.moveTo(pad, tTop - F(8)); x.lineTo(W - pad, tTop - F(8)); x.stroke();
  const weights = [1.35, 1, 1, 1.05, 0.95];
  const totalW = W - pad * 2, wSum = weights.reduce((a, b) => a + b, 0);
  let cx = pad;
  columns.forEach((col, ci) => {
    const cw = totalW * weights[ci] / wSum - 14;
    let cy = tTop + 8;
    for (const [title, rows] of col) {
      x.fillStyle = "#c8c8dc"; x.font = `600 ${F(15)}px Lora, Georgia, serif`;
      x.textAlign = "left";
      x.fillText(title, cx, cy + F(12));
      cy += TITLE;
      for (const cells of rows) {
        if (cells.length > 2) {                       // the wide parcels grid
          const step = cw / cells.length;
          let fpx = 12.5;                             // shrink to the column step
          x.font = `${F(fpx)}px Inter, sans-serif`;
          while (fpx > 8 && Math.max(...cells.map(c => x.measureText(c).width)) > step - F(6)) {
            fpx -= 0.5; x.font = `${F(fpx)}px Inter, sans-serif`;
          }
          cells.forEach((c, i) => {
            x.fillStyle = rows.indexOf(cells) === 0 ? "#8b8ba3" : "#e8e8f0";
            x.textAlign = i === 0 ? "left" : "right";
            if (i === 0) x.fillText(c, cx, cy + F(6));
            else drawValCell(c, cx + step * (i + 1) - 4, cy + F(6), fpx);
          });
        } else {
          let fpx = 12.5;               // shrink to fit: long label + long value
          x.font = `${F(fpx)}px Inter, sans-serif`;
          while (fpx > 8.5 && x.measureText(cells[0] || "").width +
                 x.measureText(cells[1] || "").width + F(10) > cw) {
            fpx -= 0.5; x.font = `${F(fpx)}px Inter, sans-serif`;
          }
          x.fillStyle = "#8b8ba3";
          x.textAlign = "left"; x.fillText(cells[0] || "", cx, cy + F(6));
          x.fillStyle = "#e8e8f0"; x.textAlign = "right";
          drawValCell(cells[1] || "", cx + cw, cy + F(6), fpx);
        }
        cy += ROW;
      }
      cy += GAP;
    }
    cx += totalW * weights[ci] / wSum;
  });

  x.fillStyle = "#64d2ff"; x.font = `700 ${F(26)}px Inter, sans-serif`;
  x.textAlign = "left";
  x.fillText("scorvec.com/skewt", pad, H - F(20));
  x.fillStyle = "#8b8ba3"; x.font = `${F(20)}px Inter, sans-serif`;
  x.textAlign = "right";
  x.fillText(plotTitle, W - pad, H - F(20));
  return cv;
}
// Full-resolution card: re-render the UN-thinned profile onto oversized
// canvases with dense barbs (every reportable wind level), compose, then
// restore the on-screen state — all synchronous, so nothing flickers.
function buildFullCard() {
  const keep = lastProf;
  try {
    EXPORT_PX = { skewt: { w: 1150, h: 1750, fs: 1.5 }, hodo: { w: 950, h: 950, fs: 1.5 },
                  mse: { w: 950, h: 560, fs: 1.5 } };
    BARB_GAP = 10;
    render(lastProfFull || keep);
    return buildExportCanvas(1.5);
  } finally {
    EXPORT_PX = null; BARB_GAP = 22;
    render(keep);
  }
}
function exportPNG() {
  const cv = buildFullCard();
  const a = document.createElement("a");
  a.download = (plotTitle || "sounding").replace(/[^\w.-]+/g, "_") + ".png";
  a.href = cv.toDataURL("image/png");
  a.click();
}
// click/tap the skew-T on ANY device to open the full card in an overlay —
// same image the PNG button saves
document.getElementById("skewt").addEventListener("click", () => {
  if (!lastProf) return;
  document.getElementById("png-view-img").src = buildFullCard().toDataURL("image/png");
  document.getElementById("png-view").hidden = false;
});
document.getElementById("png-view").addEventListener("click", e => {
  if (e.target.id !== "png-view-img") document.getElementById("png-view").hidden = true;
});
document.getElementById("export-btn").addEventListener("click", exportPNG);

// ---- compare: pin the current sounding, overlay it under the next one ----
document.getElementById("pin-btn").addEventListener("click", () => {
  const b = document.getElementById("pin-btn");
  if (pinned) {
    pinned = null;
    b.textContent = "📌 Compare";
    b.title = "pin this sounding, then load another to overlay them";
  } else if (lastProf) {
    pinned = { prof: lastProf, title: plotTitle };
    b.textContent = "📌 Unpin";
    b.title = "comparing against " + pinned.title;
    modal.hidden = true;                    // straight back to the map — no manual close
    setStatus(`📌 pinned ${pinned.title} — click any station (or use the date arrows) to compare`);
  }
  if (lastProf && lastRes) { drawSkewT(lastProf, lastRes); drawHodo(lastProf, lastRes); }
});

// legend + closed-station toggle (Leaflet control)
const legend = L.control({ position: "bottomleft" });
legend.onAdd = () => {
  const div = L.DomUtil.create("div");
  div.className += " maplegend";
  div.innerHTML =
    '<span class="leg-chip" title="legend">ⓘ&nbsp;key</span><div class="leg-body">' +
    '<span style="color:#4a7ab5;font-size:1.05em">●</span> reported at last regular interval (00Z/12Z)<br>' +
    '<span style="color:#bf5af2;font-size:1.05em">●</span> off-hour (06/18Z)<br>' +
    '<span style="color:#33c495;font-size:1.05em">●</span> active &nbsp;&nbsp;' +
    '<span style="color:#c0392b;font-size:1.05em">●</span> closed<br>' +
    '<span style="color:#ff2d2d">◎</span> near record &nbsp;&nbsp;' +
    '<span style="color:#ffd60a">◎</span> selected<br>' +
    '<label style="cursor:pointer"><input type="checkbox" id="show-closed"> show closed stations</label></div>';
  div.addEventListener("click", e => {          // chip toggles on mobile; CSS gates visibility
    if (e.target.id !== "show-closed" && e.target.tagName !== "LABEL")
      div.classList.toggle("expanded");
  });
  L.DomEvent.disableClickPropagation(div);
  return div;
};
legend.addTo(map);
document.getElementById("show-closed").onchange = e =>
  e.target.checked ? closedLayer.addTo(map) : map.removeLayer(closedLayer);

function highlight(marker) {
  if (selectedMarker)                      // restore BOTH weight and color, else
    selectedMarker.setStyle({ weight: selectedMarker._baseW || 1,     // every
                              color: selectedMarker._baseC || "#333" }); // past
  marker._baseW = marker.options.weight;   // selection keeps a yellow outline
  marker._baseC = marker.options.color;
  marker.setStyle({ weight: 3.5, color: "#ffd60a" });
  selectedMarker = marker;
}

function syncControls() {
  document.querySelectorAll('[data-act="latest"]').forEach(b =>
    b.classList.toggle("on", mode === "latest"));
  document.querySelectorAll('[data-act="archive"]').forEach(b =>
    b.classList.toggle("on", mode === "archive"));
  document.querySelectorAll('[data-act="h00"]').forEach(b =>
    b.classList.toggle("on", archHour === 0));
  document.querySelectorAll('[data-act="h12"]').forEach(b =>
    b.classList.toggle("on", archHour === 12));
  document.querySelectorAll(".arch-controls").forEach(el =>
    el.style.display = mode === "archive" ? "inline" : "none");
  document.querySelectorAll(".datectl").forEach(el => { el.value = archDate; });
}

function setMode(m2) { mode = m2; syncControls(); }

function maybeReload() {
  if (!current) return;
  if (modal.hidden && pinned) modal.hidden = false;   // arrows re-open a pinned compare
  if (!modal.hidden) loadSounding();
}

function stepDate(dir) {
  // Step to the NEXT SOUNDING, not the next calendar day: toggle 12Z<->00Z
  // within a day first, and use the station's actual launch-date list (cached
  // after the first archive load) for the day jumps, skipping gap days.
  const dates = (current && current.gid && igraDatesCache.get(current.gid)) || null;
  const dayStep = d => {
    if (dates && dates.length) {
      let i = dates.findIndex(x => x >= archDate);
      if (i < 0) i = dates.length - 1;
      if (dates[i] !== archDate && d < 0) i--;        // between entries going back
      const j = Math.max(0, Math.min(dates.length - 1, i + d));
      return dates[j];
    }
    const dt = new Date(archDate + "T00:00:00Z");
    return new Date(dt.getTime() + d * 864e5).toISOString().slice(0, 10);
  };
  if (dir < 0) {
    if (archHour === 12) archHour = 0;                // 12Z -> 00Z same day
    else { archDate = dayStep(-1); archHour = 12; }   // 00Z -> prev day's 12Z
  } else {
    if (archHour === 0) archHour = 12;                // 00Z -> 12Z same day
    else { archDate = dayStep(1); archHour = 0; }     // 12Z -> next day's 00Z
  }
  if (mode !== "archive") mode = "archive";
  syncControls();
  maybeReload();
}

document.querySelectorAll("[data-act]").forEach(b => {
  b.addEventListener("click", () => {
    const a = b.dataset.act;
    if (a === "latest" || a === "archive") { setMode(a); maybeReload(); }
    else if (a === "h00") { archHour = 0; syncControls(); maybeReload(); }
    else if (a === "h12") { archHour = 12; syncControls(); maybeReload(); }
    else if (a === "dprev") stepDate(-1);
    else if (a === "dnext") stepDate(1);
  });
});
document.querySelectorAll(".datectl").forEach(el => {
  el.addEventListener("change", () => {
    archDate = el.value;
    if (mode !== "archive") mode = "archive";
    syncControls();
    maybeReload();
  });
});
syncControls();

const COUNTRY = { US:"USA", CA:"Canada", MX:"Mexico", BR:"Brazil", AR:"Argentina",
  CH:"China", JA:"Japan", KS:"South Korea", IN:"India", AS:"Australia", NZ:"New Zealand",
  UK:"UK", FR:"France", GM:"Germany", SP:"Spain", PO:"Portugal", IT:"Italy", RS:"Russia",
  NO:"Norway", SW:"Sweden", FI:"Finland", IC:"Iceland", GL:"Greenland", DA:"Denmark",
  NL:"Netherlands", PL:"Poland", AU:"Austria", SZ:"Switzerland", GR:"Greece", TU:"Turkey",
  EG:"Egypt", SF:"South Africa", KE:"Kenya", SA:"Saudi Arabia", IS:"Israel", PK:"Pakistan",
  TH:"Thailand", VM:"Vietnam", ID:"Indonesia", MY:"Malaysia", PH:"Philippines",
  CI:"Chile", PE:"Peru", CO:"Colombia", VE:"Venezuela", EC:"Ecuador", UY:"Uruguay",
  PM:"Panama", CU:"Cuba", FJ:"Fiji" };
function selectStation(s) {
  current = s;
  if (s.gid && s.n) {                       // country from the IGRA FIPS prefix
    const c = COUNTRY[s.gid.slice(0, 2)];
    if (c && !s.n.includes(c))
      s.n = s.n.replace(/[;,]\s*$/, "") + " (" + c + ")";
  }
  openModal();
  loadClimo(s.gid);
  // coordinates: useful for cross-referencing model soundings / satellite
  const ig = s.gid ? igraStations[s.gid] : null;
  const la = s.la ?? (ig && ig.la), lo = s.lo ?? (ig && ig.lo);
  const el = s.e ?? (ig && ig.e);
  plotCoords = (isFinite(la) && isFinite(lo))
    ? `${Math.abs(la).toFixed(2)}°${la >= 0 ? "N" : "S"}  ` +
      `${Math.abs(lo).toFixed(2)}°${lo >= 0 ? "E" : "W"}` +
      (isFinite(el) ? `  ${Math.round(el)} m` : "")
    : "";
  // deep links (#wmoid) are still honored on page load, but clicking no longer rewrites the URL
  document.getElementById("stn-label").textContent = `${s.n || s.gid} · ${s.id || s.gid}`;
  setStatus(mode === "latest"
    ? "fetching latest sounding from the mirror…"
    : "fetching from the NOAA IGRA archive (first load per station can be tens of MB)…", true);
  document.getElementById("uw-link").href = s.id
    ? "https://weather.uwyo.edu/wsgi/sounding?id=" + s.id
    : "https://weather.uwyo.edu/upperair/sounding.shtml";
  loadSounding();
}

// ---------- data fetch ----------
function parseCSV(text) {
  const rows = text.trim().split("\n");
  if (!rows[0] || !rows[0].startsWith("time")) return null;
  const out = { P: [], H: [], T: [], D: [], U: [], V: [] };
  let lastP = 1e9;
  for (let i = 1; i < rows.length; i++) {
    const c = rows[i].split(",");
    if (c.length < 13) continue;
    const p = +c[3] * 100, h = +c[4], t = +c[5] + 273.15, d = +c[6] + 273.15;
    const wd = +c[11], ws = +c[12];
    if (!isFinite(p) || !isFinite(t) || p >= lastP || p < 2000) continue;
    lastP = p;
    out.P.push(p); out.H.push(isFinite(h) ? h : NaN); out.T.push(t);
    out.D.push(isFinite(d) ? d : NaN);
    if (isFinite(wd) && isFinite(ws)) {
      out.U.push(-ws * Math.sin(wd * Math.PI / 180));
      out.V.push(-ws * Math.cos(wd * Math.PI / 180));
    } else { out.U.push(NaN); out.V.push(NaN); }
  }
  return out.P.length >= 10 ? out : null;
}

function thin(prof, target = 350) {
  const N = prof.P.length;
  if (N <= target) return prof;
  const k = Math.ceil(N / target);
  const out = { P: [], H: [], T: [], D: [], U: [], V: [] };
  for (let i = 0; i < N; i++) {
    if (i === 0 || i === N - 1 || i % k === 0)
      for (const key of Object.keys(out)) out[key].push(prof[key][i]);
  }
  return out;
}

async function igraText(gid, year) {
  // the period-of-record file contains everything, so it satisfies any request
  if (igraCache.has(gid + ":por")) return igraCache.get(gid + ":por");
  const recent = year >= new Date().getUTCFullYear() - 1;
  if (recent && igraCache.has(gid + ":y2d")) return igraCache.get(gid + ":y2d");
  const urls = [];
  if (recent) {
    // NCEI names the y2d file by its actual begin year, which tracks how fresh
    // the station's POR file is — usually the current year, sometimes last
    // year. Try both before surrendering to the multi-MB POR download.
    const Y = new Date().getUTCFullYear();
    urls.push(IGRA + `data-y2d/${gid}-data-beg${Y}.txt.zip`);
    urls.push(IGRA + `data-y2d/${gid}-data-beg${Y - 1}.txt.zip`);
  }
  urls.push(IGRA + `data-por/${gid}-data.txt.zip`);
  for (const url of urls) {
    try {
      const r = await fetch(url);
      if (!r.ok) continue;
      const len = +r.headers.get("content-length");
      setStatus(len ? `downloading IGRA archive (${(len / 1048576).toFixed(1)} MB)…`
                    : "downloading IGRA archive…", true);
      const buf = new Uint8Array(await r.arrayBuffer());
      setStatus("decompressing…", true);
      const files = fflate.unzipSync(buf);
      const text = fflate.strFromU8(files[Object.keys(files)[0]]);
      // decompressed POR texts run 50–150+ MB; unbounded caching OOMs mobile
      // tabs after a few archive stations — keep only the most recent two
      while (igraCache.size >= 2) igraCache.delete(igraCache.keys().next().value);
      igraCache.set(url.includes("data-y2d") ? gid + ":y2d" : gid + ":por", text);
      return text;
    } catch (e) { /* try next */ }
  }
  return null;
}

const igraDatesCache = new Map();
function igraDates(text, gid) {
  if (igraDatesCache.has(gid)) return igraDatesCache.get(gid);
  const re = new RegExp("^#" + gid + " ([0-9]{4}) ([0-9]{2}) ([0-9]{2})", "gm");
  const seen = new Set();
  let m;
  while ((m = re.exec(text)) !== null) seen.add(m[1] + "-" + m[2] + "-" + m[3]);
  const arr = [...seen].sort();
  igraDatesCache.set(gid, arr);
  return arr;
}

const iv = (line, a, b) => {
  const v = parseInt(line.slice(a, b), 10);
  return (v === -9999 || v === -8888 || isNaN(v)) ? null : v;
};

function parseIGRA(text, gid, ymd, wantHour, elev) {
  const [Y, Mo, D] = ymd.split("-");
  const re = new RegExp("^#" + gid + " " + Y + " " + Mo + " " + D + " ([0-9]{2})", "gm");
  let best = null, m;
  while ((m = re.exec(text)) !== null) {
    const hh = +m[1] === 99 ? 12 : +m[1];
    if (best === null || Math.abs(hh - wantHour) < Math.abs(best.hh - wantHour))
      best = { idx: m.index, hh };
  }
  if (!best) return null;
  const lines = text.slice(best.idx).split("\n");
  const nlev = parseInt(lines[0].slice(32, 36), 10);
  // IGRA levels can be thermo-only, wind-only, or both — collect separately
  const stdP = h => 101325 * Math.pow(1 - 2.25577e-5 * h, 5.25588);  // ISA, Pa
  const thermo = [], winds = [];
  for (let i = 1; i <= nlev && i < lines.length; i++) {
    const L = lines[i];
    let p = iv(L, 9, 15);
    const gph = iv(L, 16, 21), tt = iv(L, 22, 27), dpdp = iv(L, 34, 39);
    const rh = iv(L, 28, 33);               // pre-~1990 US records carry RH, not DPDP
    const wd = iv(L, 40, 45), ws = iv(L, 46, 51);
    // pilot-balloon levels report by height with pressure missing
    if (p === null && gph !== null && gph > -900) p = stdP(gph);
    if (p === null || p < 2000) continue;
    if (wd !== null && ws !== null && ws >= 0)
      winds.push({ p, h: gph, u: -(ws / 10) * Math.sin(wd * Math.PI / 180),
                              v: -(ws / 10) * Math.cos(wd * Math.PI / 180) });
    if (tt !== null && tt > -8888) {
      // moisture: dewpoint depression when present; otherwise derive Td from RH
      // (1970s US soundings report RH only — requiring DPDP threw away the whole
      // thermo profile and mislabelled real soundings as wind-only pibals)
      let D = NaN;
      if (dpdp !== null) D = tt / 10 + 273.15 - dpdp / 10;
      else if (rh !== null && rh > 0) {
        const tc = tt / 10, RH = Math.min(100, rh / 10);
        const gamma = Math.log(RH / 100) + 17.67 * tc / (tc + 243.5);
        D = 243.5 * gamma / (17.67 - gamma) + 273.15;
      }
      thermo.push({ p, gph, T: tt / 10 + 273.15, D });
    }
  }
  thermo.sort((a, b) => b.p - a.p);
  winds.sort((a, b) => b.p - a.p);
  // "wind-only" means genuinely no usable thermo — NOT "coarse". 1960s marine
  // records (ships, buoy sites) carry 5-7 mandatory-level temperatures and the
  // old <8 cutoff threw them away (Environm Buoy 44005, 1963-09-30 00Z).
  const windOnly = thermo.length < 3;
  if (windOnly && winds.length < 4) return null;
  const windAtP = p => {
    if (!winds.length) return { u: 0, v: 0 };
    if (p >= winds[0].p) return winds[0];
    for (let j = 1; j < winds.length; j++) {
      if (winds[j].p <= p) {
        const f = Math.log(winds[j - 1].p / p) / Math.log(winds[j - 1].p / winds[j].p);
        return { u: winds[j - 1].u + f * (winds[j].u - winds[j - 1].u),
                 v: winds[j - 1].v + f * (winds[j].v - winds[j - 1].v) };
      }
    }
    return winds[winds.length - 1];
  };
  const out = { P: [], H: [], T: [], D: [], U: [], V: [] };
  let lastP = 1e9, lastH = elev, lastT = null;
  if (windOnly) {                                   // pilot-balloon: hodograph only
    let lp = 1e9;
    for (const w of winds) {
      if (w.p >= lp) continue;
      lp = w.p;
      const Hm = (w.h !== null && w.h > -900) ? w.h
        : elev + (287.05 * 250.0 / 9.80665) * Math.log(1e5 / w.p);
      out.P.push(w.p); out.H.push(Hm); out.T.push(NaN); out.D.push(NaN);
      out.U.push(w.u); out.V.push(w.v);
    }
    return out.P.length >= 4
      ? { prof: out, hh: best.hh, nWind: winds.length, windOnly: true } : null;
  }
  for (const lv of thermo) {
    if (lv.p >= lastP) continue;
    let Hm;
    if (lv.gph !== null) Hm = lv.gph;
    else {
      const Tbar = lastT === null ? lv.T : (lv.T + lastT) / 2;
      Hm = lastH + (287.05 * Tbar / 9.80665) * Math.log(lastP / lv.p);
    }
    if (out.P.length === 0 && lv.gph === null) Hm = elev;
    lastP = lv.p; lastH = Hm; lastT = lv.T;
    const w = windAtP(lv.p);
    out.P.push(lv.p); out.H.push(Hm); out.T.push(lv.T); out.D.push(lv.D);
    out.U.push(w.u); out.V.push(w.v);
  }
  // 1960s marine soundings are mandatory-levels-only — 5-7 levels is a real
  // profile, not noise. Require just enough for SHARPlib to work with.
  return out.P.length >= 4 ? { prof: out, hh: best.hh, nWind: winds.length } : null;
}

let loadSeq = 0;
async function loadSounding() {
  if (!current) return;
  const seq = ++loadSeq;                 // stale-response guard
  const stale = () => seq !== loadSeq;
  await wasmReady;
  if (wasmFailed) {
    setStatus(`analysis engine failed to load (${wasmFailed}) — reload the page`);
    return;
  }
  if (stale()) return;
  if (mode === "latest") {
    const s = entries[current.id];
    // real-time first: SPC (fastest US), then IEM (US/Canada, more levels)
    if (current.id && iemMap && iemMap[current.id]) {
      setStatus("fetching real-time sounding…", true);
      let got = await fetchSPC(current.id).catch(() => null)
        || await fetchIEM(current.id).catch(() => null);
      if (stale()) return;
      // SPC/IEM walk back up to 3 slots when the newest is missing on their
      // side — never let that beat a NEWER launch already on the mirror
      // (Chanhassen 2026-07-12 00Z: SPC 404'd it, walked back to 12Z 07-11,
      // and won over the mirror's 00Z).
      if (got && s && s.dt && s.dt > got.valid) got = null;
      if (got) {
        setStatus(`valid ${got.valid}Z · ${got.prof.P.length} levels (${got.src})`);
        plotTitle = `${current.n || ""} ${current.id}  ·  ${got.valid}Z`.trim();
        plotNote = ""; lastMonth = got.valid.slice(5, 7); lastDoy = doyOf(got.valid); lastHourZ = hourOf(got.valid);
        lastValidDt = got.valid;
        render(thin(lastProfFull = got.prof));
        return;
      }
    }
    if (s) {
      setStatus("fetching…", true);
      try {
        const r = await ghFetch("skewt-data", "soundings/" + current.id + ".csv", s.dt);
        if (!r.ok) throw 0;
        const prof = parseCSV(await r.text());
        if (stale()) return;
        if (!prof) throw 0;
        setStatus(`valid ${s.dt}Z · ${prof.P.length} levels (UW BUFR/GTS mirror)`);
        plotTitle = `${current.n || ""} ${current.id}  ·  ${s.dt}Z`.trim();
        plotNote = ""; lastMonth = s.dt.slice(5, 7); lastDoy = doyOf(s.dt); lastHourZ = hourOf(s.dt);
        lastValidDt = s.dt;
        render(thin(lastProfFull = prof));
        return;
      } catch (e) { /* mirror failed — fall through to the archive */ }
    }
    // No live source at all (station absent from SPC/IEM/UW this cycle):
    // "Latest" should still mean something — show the newest ARCHIVED sounding
    // rather than a dead end. The nearest-available fallback below finds it.
    setStatus("no live feed — fetching this station's newest archived sounding…", true);
  }
  // archive mode: recent launches come from the high-res UW mirror when
  // available (BUFR fidelity, ~4-day retention), else NOAA IGRA v2.
  // (Latest mode lands here too when no live source exists — with today's
  // date, so the fallback resolves to the newest launch on record.)
  const ymd = mode === "latest" ? new Date().toISOString().slice(0, 10) : archDate;
  const me = current.id ? entries[current.id] : null;
  const wantDt = `${ymd} ${String(archHour).padStart(2, "0")}:00`;
  if (me && (me.hours || []).includes(wantDt)) {
    setStatus("fetching high-resolution sounding from the mirror…", true);
    try {
      const tag = wantDt.replace(/[-: ]/g, "").slice(0, 10);
      const r = await ghFetch("skewt-data", "soundings/" + current.id + "_" + tag + ".csv", tag);
      if (r.ok) {
        const prof = parseCSV(await r.text());
        if (stale()) return;              // user switched station mid-download
        if (prof) {
          setStatus(`valid ${wantDt}Z · ${prof.P.length} levels (UW BUFR high-res mirror)`);
          plotTitle = `${current.n || ""} ${current.id}  ·  ${wantDt}Z`.trim();
          plotNote = ""; lastMonth = wantDt.slice(5, 7); lastDoy = doyOf(wantDt); lastHourZ = hourOf(wantDt);
          lastValidDt = wantDt;
          render(thin(lastProfFull = prof));
          return;
        }
      }
    } catch (e) { /* fall through to IGRA */ }
  }
  // permanent high-res day bundles (UW BUFR, from the archive branch)
  if (current.id && ymd >= UW_ARCHIVE_START) {
    const dkey = ymd.replaceAll("-", "");
    if (!dayZipCache.has(dkey)) {
      setStatus(`downloading high-res day bundle ${ymd}…`, true);
      try {
        const r = await ghFetch("skewt-archive", "uw-" + dkey + ".zip", null, true);
        dayZipCache.set(dkey, r.ok
          ? fflate.unzipSync(new Uint8Array(await r.arrayBuffer())) : null);
      } catch (e) { /* transient network failure — don't cache, retry next time */ }
    }
    if (stale()) return;                  // user switched station mid-download
    const bundle = dayZipCache.get(dkey);
    if (bundle) {
      const want = `${current.id}_${dkey}${String(archHour).padStart(2, "0")}.csv`;
      const alt = Object.keys(bundle).find(k => k.startsWith(current.id + "_"));
      const pick = bundle[want] ? want : alt;
      if (pick) {
        const prof = parseCSV(fflate.strFromU8(bundle[pick]));
        if (prof) {
          const hh = pick.slice(-6, -4);
          setStatus(`valid ${ymd} ${hh}Z · ${prof.P.length} levels (UW BUFR day archive)`);
          plotTitle = `${current.n || ""} ${current.id}  ·  ${ymd} ${hh}Z`.trim();
          plotNote = ""; lastMonth = ymd.slice(5, 7); lastDoy = doyOf(ymd); lastHourZ = +hh;
          lastValidDt = `${ymd} ${hh}:00`;
          render(thin(lastProfFull = prof));
          return;
        }
      }
    }
  }
  if (!current.gid) { setStatus("station not in the IGRA archive"); return; }
  const text = await igraText(current.gid, +ymd.slice(0, 4));
  if (stale()) return;
  if (!text) { clearPlot("IGRA archive file unavailable for this station"); return; }
  let shown = ymd, fellBack = false;
  let got = parseIGRA(text, current.gid, ymd, archHour, current.e || 0);
  if (!got) {
    // walk outward from the requested date (nearest-first), capped, until a
    // sounding parses. parseIGRA returns wind-only pibal profiles too, so this
    // also lands hodograph-only stations on a usable launch.
    const dates = igraDates(text, current.gid);
    let ci = dates.length - 1;
    while (ci >= 0 && dates[ci] > ymd) ci--;        // nearest index <= requested
    const order = [];
    for (let d = 0; d < dates.length && order.length < 400; d++) {
      if (ci - d >= 0) order.push(ci - d);
      if (ci + 1 + d < dates.length) order.push(ci + 1 + d);
    }
    for (const k of order) {
      got = parseIGRA(text, current.gid, dates[k], archHour, current.e || 0);
      if (got) { shown = dates[k]; fellBack = true; archDate = dates[k]; syncControls(); break; }
    }
  }
  if (!got) {
    clearPlot("this station reported pilot-balloon winds only — no temperature soundings");
    return;
  }
  setStatus(`valid ${shown} ${String(got.hh).padStart(2, "0")}Z · ` +
    (got.windOnly ? `${got.nWind} wind levels — pilot balloon, hodograph only`
                  : `${got.prof.P.length} levels, ${got.nWind} wind levels`) +
    ` (NOAA IGRA v2)` + (fellBack ? ` — nearest available to ${ymd}` : ""));
  plotTitle = `${current.n || ""} ${current.id || current.gid}  ·  ${shown} ` +
    `${String(got.hh).padStart(2, "0")}Z`.trim();
  plotNote = got.windOnly
    ? "⚠ WIND-ONLY DATA (pilot balloon) — no temperature; hodograph valid" : "";
  // shown is date-only — hourOf() would default every IGRA sounding to 12Z and
  // skew the fog model + same-hour climatology for 00Z launches; use the real hour
  lastMonth = shown.slice(5, 7); lastDoy = doyOf(shown); lastHourZ = got.hh;
  lastValidDt = `${shown} ${String(got.hh).padStart(2, "0")}:00`;
  render(thin(lastProfFull = got.prof));
}

// The launch ~12 h before the plotted one, for MSE tendency: destabilization
// shows up as boundary-layer h rising against a static or falling mid-level h*
// well before CAPE says anything. Sources, cheapest first: the mirror's
// per-launch retention (~4 days), then the ALREADY-CACHED IGRA text in archive
// mode — never a fresh multi-MB download just for a trend arrow.
async function queuePrev12() {
  if (!current || !lastValidDt) return;
  const key = (current.id || current.gid || "?") + "|" + lastValidDt;
  if (key === prev12Key) return;                    // have it or fetching it
  prev12Key = key; prev12 = null;
  const t0 = Date.parse(lastValidDt.replace(" ", "T") + ":00Z");
  if (!isFinite(t0)) return;
  let got = null;
  const me = current.id ? entries[current.id] : null;
  if (me && me.hours) {
    let best = null;
    // ideal is 12 h, but a once-daily station (Green Bay flies only 18Z right
    // now) can only offer ~24 h — take it and let the Δ label say so
    for (const hstr of me.hours) {
      const back = (t0 - Date.parse(hstr.replace(" ", "T") + ":00Z")) / 3600e3;
      if (back >= 6 && back <= 30 &&
          (!best || Math.abs(back - 12) < Math.abs(best.back - 12) ||
           (Math.abs(back - 12) === Math.abs(best.back - 12) && back < best.back)))
        best = { hstr, back };
    }
    if (best) {
      try {
        const tag = best.hstr.replace(/[-: ]/g, "").slice(0, 10);
        const r = await ghFetch("skewt-data", "soundings/" + current.id + "_" + tag + ".csv", tag);
        if (r.ok) {
          const p = parseCSV(await r.text());
          if (p) got = { prof: p, dt: best.hstr, backH: Math.round(best.back) };
        }
      } catch (e) { /* the trend is optional — never block the sounding */ }
    }
  }
  // US stations: the mirror can be missing whole slots (UW manifest hiccups),
  // but SPC serves every observed 00Z/12Z sounding directly — hit the exact
  // slot nearest 12 h back, walking the 6-hourly slots inside the window.
  if (!got && current.id && iemMap && iemMap[current.id] &&
      iemMap[current.id][0] === "K") {
    for (const backH of [12, 6, 18, 24]) {
      const d = new Date(t0 - backH * 3600e3);
      if (d.getUTCHours() % 6 !== 0) continue;      // SPC publishes synoptic slots
      const stamp = String(d.getUTCFullYear()).slice(2) + p2(d.getUTCMonth() + 1) +
                    p2(d.getUTCDate()) + p2(d.getUTCHours());
      try {
        const r = await fetch(`${SPC}/${stamp}_OBS/${iemMap[current.id].slice(1)}.txt`);
        if (!r.ok) continue;
        const g = spcProfile(await r.text());
        if (g) { got = { prof: g.prof, dt: g.valid, backH }; break; }
      } catch (e) { /* next slot */ }
    }
  }
  // IEM covers every US/Canada site and a 12-h-old launch is far past its lag
  if (!got && current.id && iemMap && iemMap[current.id]) {
    for (const backH of [12, 6, 18, 24]) {
      const d = new Date(t0 - backH * 3600e3);
      if (d.getUTCHours() % 12 !== 0) continue;     // routine launches only
      const ts = `${d.getUTCFullYear()}${p2(d.getUTCMonth() + 1)}${p2(d.getUTCDate())}${p2(d.getUTCHours())}00`;
      try {
        const r = await fetch(`${IEM}?ts=${ts}&station=${iemMap[current.id]}`);
        if (!r.ok) continue;
        const g = iemProfile(await r.json());
        if (g) { got = { prof: g.prof, dt: g.valid, backH }; break; }
      } catch (e) { /* next slot */ }
    }
  }
  if (!got && current.gid) {
    const text = igraCache.get(current.gid + ":por") || igraCache.get(current.gid + ":y2d");
    if (text) {
      for (const backTry of [12, 24]) {             // 24: once-daily stations
        const prev = new Date(t0 - backTry * 3600e3);
        const ymd = prev.toISOString().slice(0, 10);
        const g = parseIGRA(text, current.gid, ymd, prev.getUTCHours(), current.e || 0);
        if (!g || g.windOnly) continue;
        const backH = (t0 - Date.parse(`${ymd}T${String(g.hh).padStart(2, "0")}:00:00Z`)) / 3600e3;
        if (backH >= 6 && backH <= 30) {
          got = { prof: g.prof, dt: `${ymd} ${String(g.hh).padStart(2, "0")}:00`,
                  backH: Math.round(backH) };
          break;
        }
      }
    }
  }
  if (key !== prev12Key) return;                    // a newer sounding superseded us
  if (got) {
    got.M = mseProfile(got.prof);
    got.forKey = key;
    prev12 = got;
    if (lastProf && lastRes) { fillTables(lastProf, lastRes); drawMSE(lastProf); }
  }
}
// prev12 only if it belongs to the sounding on screen — a station switch renders
// once before the async trend fetch clears, and a stale overlay is a lie
function prev12Live() {
  const key = current ? (current.id || current.gid || "?") + "|" + lastValidDt : "";
  return (prev12 && prev12.forKey === key) ? prev12 : null;
}

// ---------- compute ----------
function compute(prof) {
  const N = prof.P.length;
  const ptrs = ["P", "H", "T", "D", "U", "V"].map(k => f32(prof[k]));
  const nOut = M._out_size();          // ask the WASM; never hardcode (it drifted twice)
  const out = M._malloc(nOut * 4);
  const tr = [M._malloc(N * 4), M._malloc(N * 4), M._malloc(N * 4)];
  const _rc = M._compute_sounding(...ptrs, N, out, tr[0], tr[1], tr[2]);
  const o = Array.from(M.HEAPF32.subarray(out / 4, out / 4 + nOut));
  const traces = tr.map(p => Array.from(M.HEAPF32.subarray(p / 4, p / 4 + N)));
  [...ptrs, out, ...tr].forEach(p => M._free(p));
  if (_rc !== 0) console.warn("compute_sounding rc=", _rc);
  return { rc: _rc, o, sb: traces[0], ml: traces[1], mu: traces[2] };
}

function traceAdiabat(startP, startT, startD, pGrid) {
  const pp = f32(pGrid), oo = M._malloc(pGrid.length * 4);
  M._trace_adiabat(startP, startT, startD, pp, oo, pGrid.length);
  const r = Array.from(M.HEAPF32.subarray(oo / 4, oo / 4 + pGrid.length));
  M._free(pp); M._free(oo);
  return r;
}

// ---------- theme + helpers ----------
const TH = {
  bg: "#0a0a14", panel: "#11111c", grid: "#23233a", gridSub: "#191928",
  ink: "#e8e8f0", muted: "#8b8ba3",
  temp: "#ff453a", dwpt: "#30d158", vtmp: "#ff9f9b",
  parcelMU: "#ffffff", parcelSB: "#ff9f0a", parcelML: "#64d2ff",
  isotherm: "#26263e", isotherm0: "#3d5a8f", dryad: "#3a2f1f", moistad: "#1d3a2a",
  mixr: "#245536", lcl: "#30d158", lfc: "#ffd60a", el: "#bf5af2", eil: "#64d2ff",
  barb: "#cfd0e2",
};
const KT = 1.9438;

function interpHagl(prof, pPa) {          // height AGL (m) at pressure, log-p linear
  if (!isFinite(pPa) || pPa === MISSING) return null;
  const P = prof.P, H = prof.H;
  if (pPa >= P[0]) return 0;
  for (let i = 1; i < P.length; i++) {
    if (P[i] <= pPa) {
      const f = Math.log(P[i - 1] / pPa) / Math.log(P[i - 1] / P[i]);
      return H[i - 1] + f * (H[i] - H[i - 1]) - H[0];
    }
  }
  return null;
}
function interpP(prof, key, pPa) {         // interp any field at pressure (log-p)
  const P = prof.P, A = prof[key];
  if (!isFinite(pPa)) return NaN;
  if (pPa >= P[0]) return (pPa - P[0] < 2000) ? A[0] : NaN;   // underground = "—"''
  for (let i = 1; i < P.length; i++) {
    if (P[i] <= pPa) {
      if (!isFinite(A[i]) || !isFinite(A[i - 1])) return NaN;
      const f = Math.log(P[i - 1] / pPa) / Math.log(P[i - 1] / P[i]);
      return A[i - 1] + f * (A[i] - A[i - 1]);
    }
  }
  return NaN;
}
// Moist static energy: h = cp·T + g·z + Lv·q  (the quantity ECAPE is built on).
// Comparing h against saturation MSE (h*) is the classic conditional-instability
// diagnostic: wherever boundary-layer h exceeds h*, a lifted parcel is buoyant.
const CP = 1005.7, LV = 2.501e6, GRAV = 9.80665;
// Equivalent potential temperature, Bolton (1980) eq. 43 — accurate to ~0.3 K,
// the operational standard (same formulation SPC/SHARPpy use).
function thetaEK(Pa, Tk, Dk) {
  if (![Pa, Tk, Dk].every(isFinite)) return NaN;
  const tdc = Dk - 273.15;
  const e = 6.112 * Math.exp(17.67 * tdc / (tdc + 243.5));         // hPa
  const r = 0.622 * e / Math.max(Pa / 100 - e, 0.1);               // kg/kg
  const TL = 2840 / (3.5 * Math.log(Tk) - Math.log(Math.max(e, 1e-3)) - 4.805) + 55;
  return Tk * Math.pow(100000 / Pa, 0.2854 * (1 - 0.28 * r)) *
         Math.exp((3376 / TL - 2.54) * r * (1 + 0.81 * r));
}
function mseProfile(prof) {
  const n = prof.P.length;
  const h = [], hs = [], z = [];
  const qOf = (Pa, Tk) => {                      // specific humidity at saturation w.r.t. Tk
    const tc = Tk - 273.15;
    const e = 611.2 * Math.exp(17.67 * tc / (tc + 243.5));   // Pa
    const w = 0.622 * e / Math.max(Pa - e, 1);
    return w / (1 + w);
  };
  for (let i = 0; i < n; i++) {
    const P = prof.P[i], T = prof.T[i], D = prof.D[i], H = prof.H[i];
    if (!isFinite(P) || !isFinite(T) || !isFinite(H)) { h.push(NaN); hs.push(NaN); z.push(NaN); continue; }
    const q = isFinite(D) ? qOf(P, D) : NaN;     // actual (from dewpoint)
    const qs = qOf(P, T);                        // saturation (from temperature)
    h.push(isFinite(q) ? (CP * T + GRAV * H + LV * q) / 1000 : NaN);   // kJ/kg
    hs.push((CP * T + GRAV * H + LV * qs) / 1000);
    z.push(H - prof.H[0]);
  }
  // boundary-layer h: mean over the lowest 500 m
  let sum = 0, cnt = 0;
  for (let i = 0; i < n && z[i] <= 500; i++) if (isFinite(h[i])) { sum += h[i]; cnt++; }
  const hbl = cnt ? sum / cnt : NaN;
  // minimum saturation MSE (the mid-level "dry hole" a parcel must survive)
  let hsmin = Infinity, hsminZ = NaN;
  for (let i = 0; i < n; i++) {
    if (isFinite(hs[i]) && z[i] > 1000 && z[i] < 9000 && hs[i] < hsmin) {
      hsmin = hs[i]; hsminZ = z[i];
    }
  }
  if (!isFinite(hsmin)) hsmin = NaN;
  // column-integrated MSE: (1/g)∫ h dp  -> J/m²
  let col = 0, ok = false;
  for (let i = 1; i < n; i++) {
    if (!isFinite(h[i]) || !isFinite(h[i - 1])) continue;
    col += 0.5 * (h[i] + h[i - 1]) * 1000 * (prof.P[i - 1] - prof.P[i]) / GRAV;
    ok = true;
  }
  return { h, hs, z, hbl, hsmin, hsminZ,
           deficit: hsmin - hbl, col: ok ? col : NaN };
}

// SHARPlib has no tropopause routine, so compute both standard definitions.
//  • WMO: lowest level where the lapse rate drops to <=2 K/km AND the mean lapse
//    rate stays <=2 K/km through the next 2 km (guards against false hits).
//  • Cold point: the temperature minimum — the definition that matters in the
//    deep tropics, where it sits well above the WMO level (the TTL).
function tropopause(prof) {
  const P = prof.P, T = prof.T, H = prof.H, n = P.length;
  let wmoZ = NaN, wmoP = NaN;
  for (let i = 1; i < n; i++) {
    if (P[i] > 50000) continue;                 // search above 500 hPa
    if (P[i] < 7000) break;
    if (!isFinite(T[i]) || !isFinite(T[i - 1]) || !isFinite(H[i]) || !isFinite(H[i - 1])) continue;
    const dz = H[i] - H[i - 1];
    if (dz <= 0) continue;
    const lr = -(T[i] - T[i - 1]) / dz * 1000;  // K/km, + = cooling upward
    if (lr > 2) continue;
    let ok = true;                              // sustained through the next 2 km?
    for (let j = i + 1; j < n; j++) {
      if (!isFinite(T[j]) || !isFinite(H[j])) continue;
      if (H[j] - H[i] > 2000) break;
      if (-(T[j] - T[i]) / (H[j] - H[i]) * 1000 > 2) { ok = false; break; }
    }
    if (ok) { wmoZ = H[i] - H[0]; wmoP = P[i]; break; }
  }
  let cpT = Infinity, cpZ = NaN, cpP = NaN;     // cold point
  for (let i = 0; i < n; i++) {
    if (P[i] > 40000 || !isFinite(T[i]) || !isFinite(H[i])) continue;
    if (T[i] < cpT) { cpT = T[i]; cpZ = H[i] - H[0]; cpP = P[i]; }
  }
  return { wmoZ, wmoP, cpT: isFinite(cpT) ? cpT : NaN, cpZ, cpP };
}

// Column relative humidity: CRH = ∫q dp / ∫q_sat dp — the mass-weighted column
// saturation fraction. In the tropics this, not CAPE, is what convective onset
// scales with, so it's the moisture variable that actually matters there.
function columnRH(prof, pTopPa = 10000) {
  const esat = tc => 611.2 * Math.exp(17.67 * tc / (tc + 243.5));   // Pa
  const mix = (Pa, Tk) => { const e = esat(Tk - 273.15); return 0.622 * e / Math.max(Pa - e, 1); };
  let w = 0, ws = 0;
  for (let i = 1; i < prof.P.length; i++) {
    const P0 = prof.P[i - 1], P1 = prof.P[i];
    if (P1 < pTopPa) break;
    if (![prof.T[i], prof.T[i - 1], prof.D[i], prof.D[i - 1]].every(isFinite)) continue;
    const dp = P0 - P1;
    if (dp <= 0) continue;
    w  += 0.5 * (mix(P0, prof.D[i - 1]) + mix(P1, prof.D[i])) * dp;
    ws += 0.5 * (mix(P0, prof.T[i - 1]) + mix(P1, prof.T[i])) * dp;
  }
  return ws > 0 ? 100 * w / ws : NaN;
}

// mean RH through a pressure layer (mid-levels drive entrainment)
function layerRH(prof, pBotPa, pTopPa) {
  const esat = tc => 611.2 * Math.exp(17.67 * tc / (tc + 243.5));
  let s = 0, dpSum = 0;
  for (let i = 1; i < prof.P.length; i++) {
    const P1 = prof.P[i];
    if (P1 > pBotPa || P1 < pTopPa) continue;
    if (!isFinite(prof.T[i]) || !isFinite(prof.D[i])) continue;
    const rh = 100 * esat(prof.D[i] - 273.15) / esat(prof.T[i] - 273.15);
    const dp = prof.P[i - 1] - P1;
    if (dp <= 0) continue;
    s += Math.min(100, rh) * dp; dpSum += dp;
  }
  return dpSum > 0 ? s / dpSum : NaN;
}

// ---- winter diagnostics ----------------------------------------------------
// Precipitation type by the ENERGY-AREA method of Bourgouin (2000) — the
// approach BUFKIT uses. Integrate Rd·(T − 273.15) d(ln p) to get the melting
// energy of any warm nose aloft (positive area, PA) and the refreezing energy
// of the cold layer beneath it (negative area, NA). A snowflake falling through
// enough melting energy becomes a raindrop; whether it then refreezes into an
// ice pellet or survives as freezing rain depends on NA.
const RD = 287.05;
function precipType(prof) {
  const P = prof.P, T = prof.T;
  if (!T.some(isFinite)) return { type: "—", PA: NaN, NA: NaN, sfcC: NaN };
  const sfcC = T[0] - 273.15;
  let PA = 0, NA = 0, warmAloft = false;
  for (let i = P.length - 1; i >= 1; i--) {          // top-down
    if (!isFinite(T[i]) || !isFinite(T[i - 1]) || P[i] < 20000) continue;
    const dlnp = Math.log(P[i - 1] / P[i]);          // > 0
    const tbar = 0.5 * (T[i] + T[i - 1]) - 273.15;   // °C
    const e = RD * tbar * dlnp;                      // J/kg, signed
    if (tbar > 0) { PA += e; warmAloft = true; }
    else if (warmAloft) { NA += -e; }                // cold layer BELOW the nose
  }
  let type;
  if (!warmAloft || PA < 5.6) type = "Snow";
  else if (PA <= 13.2) type = "Snow / rain mix";
  else if (sfcC > 0) type = "Rain";
  else if (NA > 46 + 0.66 * PA) type = "Ice pellets";
  else type = "Freezing rain";
  return { type, PA, NA, sfcC };
}

// Kuchera snow-to-liquid ratio: keyed on the column's warmest temperature,
// because a near-0 °C column gives dense wet snow and a cold one gives fluff.
function kucheraRatio(prof) {
  let tmax = -Infinity;
  for (let i = 0; i < prof.P.length; i++) {
    if (prof.P[i] < 50000) break;                    // surface -> 500 hPa
    if (isFinite(prof.T[i])) tmax = Math.max(tmax, prof.T[i]);
  }
  if (!isFinite(tmax)) return NaN;
  return tmax <= 271.16 ? 12 + 2 * (271.16 - tmax) : 12;
}

function wetbulbC(Tc, Tdc) {               // Stull 2011 approximation (°C)
  const es = tc => 6.112 * Math.exp(17.67 * tc / (tc + 243.5));
  const RH = Math.max(1, Math.min(100, 100 * es(Tdc) / es(Tc)));
  return Tc * Math.atan(0.151977 * Math.sqrt(RH + 8.313659)) + Math.atan(Tc + RH)
    - Math.atan(RH - 1.676331) + 0.00391838 * Math.pow(RH, 1.5) * Math.atan(0.023101 * RH)
    - 4.686035;
}
function wbzAgl(prof) {                    // wet-bulb 0 °C height, m AGL
  const P = prof.P, T = prof.T, D = prof.D, H = prof.H;
  let prevW = null, prevH = null;
  for (let i = 0; i < P.length; i++) {
    if (!isFinite(T[i]) || !isFinite(D[i])) continue;
    const w = wetbulbC(T[i] - 273.15, D[i] - 273.15);
    if (prevW !== null && prevW >= 0 && w < 0) {
      const f = prevW / (prevW - w);
      return prevH + f * (H[i] - prevH) - H[0];
    }
    prevW = w; prevH = H[i];
  }
  return NaN;
}
function freezingLvlAgl(prof) {            // lowest 0 °C crossing, m AGL
  const P = prof.P, T = prof.T, H = prof.H;
  for (let i = 1; i < P.length; i++) {
    if (isFinite(T[i]) && isFinite(T[i - 1]) &&
        (T[i - 1] - 273.15) >= 0 && (T[i] - 273.15) < 0) {
      const f = (273.15 - T[i - 1]) / (T[i] - T[i - 1]);
      return H[i - 1] + f * (H[i] - H[i - 1]) - H[0];
    }
  }
  return NaN;
}
function meanWindAgl(prof, z0, z1) {        // mean wind through a height layer (m/s)
  let su = 0, sv = 0, n = 0;
  const sfc = prof.H[0];
  for (let i = 0; i < prof.P.length; i++) {
    const z = prof.H[i] - sfc;
    if (z < z0 || z > z1 || !isFinite(prof.U[i]) || !isFinite(prof.V[i])) continue;
    su += prof.U[i]; sv += prof.V[i]; n++;
  }
  return n ? [su / n, sv / n] : null;
}
function windAt(prof, hAgl) {             // interp u,v at height AGL
  const H = prof.H, sfc = H[0];
  for (let i = 1; i < H.length; i++) {
    if (H[i] - sfc >= hAgl) {
      const f = (hAgl - (H[i - 1] - sfc)) / (H[i] - H[i - 1]);
      return [prof.U[i - 1] + f * (prof.U[i] - prof.U[i - 1]),
              prof.V[i - 1] + f * (prof.V[i] - prof.V[i - 1])];
    }
  }
  return [prof.U.at(-1), prof.V.at(-1)];
}
const dirOf = (u, v) => ((Math.atan2(-u, -v) * 180 / Math.PI) + 360) % 360;
// mean storm-relative wind (kt) through an AGL layer, Bunkers RM as the motion
function meanSR(prof, o, z0, z1) {
  if (o[22] === MISSING || !isFinite(o[22])) return null;
  const mw = meanWindAgl(prof, z0, z1);
  return mw ? Math.hypot(mw[0] - o[22], mw[1] - o[23]) * KT : null;
}
// BRN shear: 0.5 |V(0-6 km mean) - V(0-500 m mean)|^2  (m^2/s^2)
function brnShear(prof) {
  const a = meanWindAgl(prof, 0, 6000), b = meanWindAgl(prof, 0, 500);
  return a && b ? 0.5 * ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) : null;
}

// ---------- skew-t drawing ----------
// Margins are lean on purpose: the pressure labels right-align against the
// axis and the barb/T-adv columns pack tight, so the profile gets the pixels.
// The full-res export (1150 px, fonts ×1.5) is the binding constraint on l —
// a right-aligned "1000" at 18 px must still start on-canvas.
const SK = { l: 46, r: 70, t: 48, b: 38, pBot: 105000, pTop: 10000, tL: -35, tR: 45 };

// Some feeds report geopotential height as a literal 0 where it's missing
// (Curacao's CSV does this above ~145 hPa), and SHARPlib REQUIRES monotonically
// increasing height — a zeroed level makes it build an inverted height layer and
// throw, which in WASM is an unrecoverable abort. Rebuild any bad height
// hypsometrically from pressure and temperature.
function sanitizeHeights(prof) {
  const Rd = 287.05, g = 9.80665;
  const P = prof.P, T = prof.T, H = prof.H;
  let fixed = 0;
  for (let i = 1; i < P.length; i++) {
    const bad = !isFinite(H[i]) || H[i] <= H[i - 1];
    if (!bad) continue;
    if (isFinite(T[i]) && isFinite(T[i - 1]) && P[i] > 0 && P[i - 1] > P[i]) {
      const tbar = 0.5 * (T[i] + T[i - 1]);
      H[i] = H[i - 1] + Rd * tbar / g * Math.log(P[i - 1] / P[i]);
    } else {
      H[i] = H[i - 1] + 1;                 // last resort: keep it strictly increasing
    }
    fixed++;
  }
  return fixed;
}

// A single missing dewpoint poisons any column integral — SPC's Albuquerque file
// omits it on 4 stratospheric levels and PWAT came out NaN for the whole
// sounding. Heights and winds were already gap-filled; dewpoint was not.
function fillDewpoints(prof) {
  const P = prof.P, T = prof.T, D = prof.D;
  const ok = [];
  for (let i = 0; i < P.length; i++) if (isFinite(D[i])) ok.push(i);
  if (!ok.length) {                                  // no moisture at all: assume dry
    for (let i = 0; i < P.length; i++) D[i] = T[i] - 30;
    prof.dewSynth = true;                            // flag so render() warns on-plot
    return 0;
  }
  let filled = 0;
  for (let i = 0; i < P.length; i++) {
    if (isFinite(D[i])) continue;
    let lo = null, hi = null;
    for (const j of ok) { if (j < i) lo = j; else { hi = j; break; } }
    if (lo !== null && hi !== null) {                // interpolate across the gap
      const f = Math.log(P[lo] / P[i]) / Math.log(P[lo] / P[hi]);
      D[i] = D[lo] + f * (D[hi] - D[lo]);
    } else {                                         // beyond the last observation
      const ref = lo !== null ? lo : hi;             // (usually the dry stratosphere)
      D[i] = Math.min(D[ref], T[i] - 2);
    }
    if (D[i] > T[i]) D[i] = T[i];                    // never supersaturated
    filled++;
  }
  return filled;
}

function fitCanvas(cv) {                    // backing store = panel size × dpr
  if (typeof EXPORT_PX !== "undefined" && EXPORT_PX && EXPORT_PX[cv.id]) {
    const { w, h, fs } = EXPORT_PX[cv.id];  // full-res export: fixed big canvas
    cv.width = w; cv.height = h;
    let ctx = cv.getContext("2d");
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    // the draw code sets fonts in px sized for the ~800px on-screen canvas —
    // scale every font (and gently, line widths) so the big card stays legible
    if (fs && fs !== 1) ctx = new Proxy(ctx, {
      get(tg, k) { const v = tg[k]; return typeof v === "function" ? v.bind(tg) : v; },
      set(tg, k, val) {
        if (k === "font" && typeof val === "string")
          val = val.replace(/(\d+(?:\.\d+)?)px/, (m, n) => (parseFloat(n) * fs).toFixed(1) + "px");
        else if (k === "lineWidth") val = val * Math.sqrt(fs);
        tg[k] = val; return true;
      },
    });
    return { W: w, H: h, ctx };
  }
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  let { w, h } = plotBox(cv);
  const ctx = cv.getContext("2d");
  if (w && h) {
    cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { W: w, H: h, ctx };
  }
  return { W: cv.width, H: cv.height, ctx };
}
// The CSS size a plot should draw at. Desktop: both plots are laid out from
// the space .main offers — the tallest pair of design-aspect rectangles that
// fits its height AND width — and get explicit CSS sizes so their panels hug
// them (auto grid columns, centred as a pair). Mobile: the stacked CSS pins
// each canvas to width 100% at its aspect, so the box itself is right.
function plotBox(cv) {
  const R = PLOT_ASPECT[cv.id], main = cv.closest(".main");
  if (R && main && !MOBILE_MQ.matches) {
    const availH = main.clientHeight - 18, availW = main.clientWidth - 12 - 36;
    if (availH <= 0 || availW <= 0) return { w: 0, h: 0 };   // modal hidden: leave the size alone
    const H = Math.min(availH, availW / (PLOT_ASPECT.skewt + PLOT_ASPECT.hodo));
    const w = Math.floor(H * R), h = Math.floor(H);
    cv.style.width = w + "px"; cv.style.height = h + "px";
    return { w, h };
  }
  cv.style.width = ""; cv.style.height = "";
  let w = cv.clientWidth, h = cv.clientHeight;
  if (R && w && h) { if (w / h > R) w = h * R; else h = w / R; }   // never stretch a plot
  return { w: Math.floor(w), h: Math.floor(h) };
}
function drawSkewT(prof, res) {
  const cv = document.getElementById("skewt");
  const { W, H, ctx } = fitCanvas(cv);
  const o = res.o;
  const pw = W - SK.l - SK.r, ph = H - SK.t - SK.b;
  const yOf = p => SK.t + (1 - Math.log(SK.pBot / p) / Math.log(SK.pBot / SK.pTop)) * ph;
  // The 45° skew shifts a point right by (bottom − y) pixels, so a HOT surface
  // at a HIGH-ELEVATION station — whose surface already sits well up the diagram
  // — can land past the right edge and be clipped (Albuquerque, 1619 m: 37.6 °C
  // at 836 hPa overflowed by 12 px). Widen the temperature axis until the whole
  // profile fits, rather than silently cropping the data.
  // The skew must scale with the plot's ASPECT, not 1 px right per 1 px up.
  // On the old 760×840 canvas those were nearly the same thing; on a tall
  // portrait phone canvas a pixel-for-pixel skew shifts the top of the plot
  // right by MORE than the plot's width — everything aloft displaced out of
  // frame ("cut off above 100 mb / shifted right"). skewR makes an isotherm
  // run corner-to-corner at any canvas shape.
  const skewR = pw / ph;
  let TR = SK.tR;
  for (let i = 0; i < prof.P.length; i++) {
    const tc = prof.T[i] - 273.15;
    if (!isFinite(tc) || !isFinite(prof.P[i])) continue;
    const skew = ((SK.t + ph) - yOf(prof.P[i])) * skewR;  // px the skew pushes it right
    if (skew >= pw - 10) continue;
    const need = SK.tL + (tc - SK.tL) / (1 - skew / pw);
    if (need > TR) TR = need;
  }
  TR = Math.min(75, Math.ceil((TR + 3) / 5) * 5);       // headroom, rounded, sane cap
  const xOf = (tC, y) => SK.l + ((tC - SK.tL) / (TR - SK.tL)) * pw + ((SK.t + ph) - y) * skewR;
  ctx.fillStyle = TH.bg; ctx.fillRect(0, 0, W, H);
  // on-canvas title: station + valid time
  // On a phone the canvas is half the desktop width: the title ran into the
  // watermark and off the edge. Compact mode truncates the title to the space
  // that exists and drops the watermark (it's in the page footer anyway).
  const compact = W < 560;
  ctx.fillStyle = TH.ink; ctx.font = compact ? "700 13px Inter" : "700 16px Inter";
  {
    const maxW = W - SK.l - (compact ? 10 : 200);
    let title = plotTitle;
    if (ctx.measureText(title).width > maxW) {
      // never sacrifice the date/time — ellipsize the STATION NAME instead
      const cut = title.lastIndexOf("·");
      if (cut > 0) {
        const suffix = title.slice(cut);              // "· 2026-07-11 06Z"
        let prefix = title.slice(0, cut);
        while (prefix.length > 3 &&
               ctx.measureText(prefix.trimEnd() + "… " + suffix).width > maxW)
          prefix = prefix.slice(0, -2);
        title = prefix.trimEnd() + "… " + suffix;
      } else {
        while (title.length > 4 && ctx.measureText(title).width > maxW)
          title = title.slice(0, -2);
        title = title.trimEnd() + "…";
      }
    }
    ctx.fillText(title, SK.l, 22);
  }
  if (plotCoords) {
    ctx.fillStyle = TH.muted; ctx.font = compact ? "11px Inter" : "12px Inter";
    ctx.fillText(plotCoords, SK.l, 38);
  }
  if (!compact) {
    ctx.fillStyle = TH.muted; ctx.font = "12px Inter";
    ctx.fillText("SHARPlib · scorvec.com/skewt", W - 190, 22);
  }
  if (plotNote) {
    ctx.fillStyle = "#ffd60a"; ctx.font = "700 15px Inter";
    ctx.fillText(plotNote, SK.l, SK.t + 4);
  }

  ctx.save();
  ctx.beginPath(); ctx.rect(SK.l, SK.t, pw, ph); ctx.clip();

  for (let T = -120; T <= 50; T += 10) {
    const hot = (T === 0 || T === -20);
    ctx.strokeStyle = hot ? TH.isotherm0 : TH.isotherm; ctx.lineWidth = hot ? 1.3 : 1;
    ctx.beginPath();
    ctx.moveTo(xOf(T, SK.t + ph), SK.t + ph); ctx.lineTo(xOf(T, SK.t), SK.t); ctx.stroke();
    if (hot) {
      ctx.fillStyle = TH.isotherm0; ctx.font = "600 12px Inter";
      ctx.fillText(T + "°", xOf(T, yOf(30000)) + 4, yOf(30000));
    }
  }
  const pGrid = [];
  for (let lp = Math.log(105000); lp >= Math.log(SK.pTop); lp -= 0.03) pGrid.push(Math.exp(lp));
  ctx.strokeStyle = TH.dryad; ctx.lineWidth = 1;
  for (let th = 230; th <= 440; th += 10) {
    ctx.beginPath();
    pGrid.forEach((p, i) => {
      const T = th * Math.pow(p / 100000, 0.2854) - 273.15;
      i ? ctx.lineTo(xOf(T, yOf(p)), yOf(p)) : ctx.moveTo(xOf(T, yOf(p)), yOf(p));
    });
    ctx.stroke();
  }
  ctx.strokeStyle = TH.moistad; ctx.lineWidth = 1;
  const pG32 = new Float32Array(pGrid);
  for (let Ts = -24; Ts <= 40; Ts += 8) {
    const tk = traceAdiabat(105000, Ts + 273.15, Ts + 273.15, pG32);
    ctx.beginPath();
    pGrid.forEach((p, i) => {
      i ? ctx.lineTo(xOf(tk[i] - 273.15, yOf(p)), yOf(p))
        : ctx.moveTo(xOf(tk[i] - 273.15, yOf(p)), yOf(p));
    });
    ctx.stroke();
  }
  ctx.strokeStyle = TH.mixr; ctx.setLineDash([2, 4]); ctx.lineWidth = 1;
  for (const w of [1, 2, 3, 5, 8, 12, 20]) {
    ctx.beginPath();
    let started = false;
    for (const p of pGrid) {
      if (p < 55000) break;
      const e = (w * (p / 100)) / (622 + w);
      const Td = (243.5 * Math.log(e / 6.112)) / (17.67 - Math.log(e / 6.112));
      started ? ctx.lineTo(xOf(Td, yOf(p)), yOf(p)) : ctx.moveTo(xOf(Td, yOf(p)), yOf(p));
      started = true;
    }
    ctx.stroke();
  }
  ctx.setLineDash([]);

  // parcel traces: MU white dashed (bold), SB orange dashed
  const drawTrace = (tr, style, lw, dash) => {
    ctx.strokeStyle = style; ctx.lineWidth = lw; ctx.setLineDash(dash);
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < prof.P.length; i++) {
      if (!isFinite(tr[i]) || tr[i] < 100) continue;
      const x = xOf(tr[i] - 273.15, yOf(prof.P[i]));
      started ? ctx.lineTo(x, yOf(prof.P[i])) : ctx.moveTo(x, yOf(prof.P[i]));
      started = true;
    }
    ctx.stroke(); ctx.setLineDash([]);
  };
  drawTrace(res.ml, TH.parcelML, 1.2, [2, 3]);      // mixed-layer
  drawTrace(res.sb, TH.parcelSB, 1.4, [5, 4]);      // surface-based
  drawTrace(res.mu, TH.parcelMU, 1.8, [6, 4]);      // most-unstable (on top)

  // pinned comparison sounding: grey, behind the live profile
  if (pinned && pinned.prof !== prof) {
    const pp = pinned.prof;
    const line = (arr, dash) => {
      ctx.strokeStyle = "#9a9ab0"; ctx.lineWidth = 1.8; ctx.setLineDash(dash);
      ctx.beginPath(); let st = false;
      for (let i = 0; i < pp.P.length; i++) {
        if (!isFinite(arr[i]) || pp.P[i] < SK.pTop || pp.P[i] > SK.pBot) continue;
        const x = xOf(arr[i] - 273.15, yOf(pp.P[i]));
        st ? ctx.lineTo(x, yOf(pp.P[i])) : ctx.moveTo(x, yOf(pp.P[i])); st = true;
      }
      ctx.stroke(); ctx.setLineDash([]);
    };
    line(pp.T, []);
    line(pp.D, [3, 4]);
    ctx.fillStyle = "#9a9ab0"; ctx.font = "600 13px Inter";
    ctx.fillText("vs " + pinned.title, SK.l + 150, 22);
  }

  // environment: virtual temp (thin), dewpoint, temperature
  const vtC = prof.T.map((T, i) => {
    const TdC = prof.D[i] - 273.15;
    const e = 6.112 * Math.exp(17.67 * TdC / (TdC + 243.5));
    const r = 0.622 * e / (prof.P[i] / 100 - e);
    return (T * (1 + 0.61 * r)) - 273.15;
  });
  const drawProf = (valsC, color, lw, dash = []) => {
    ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.setLineDash(dash);
    ctx.beginPath();
    for (let i = 0; i < prof.P.length; i++) {
      const x = xOf(valsC[i], yOf(prof.P[i]));
      i ? ctx.lineTo(x, yOf(prof.P[i])) : ctx.moveTo(x, yOf(prof.P[i]));
    }
    ctx.stroke(); ctx.setLineDash([]);
  };
  if (prof.T.some(v => isFinite(v))) {
    drawProf(vtC, TH.vtmp, 1.1, [2, 3]);
    // frost point: wherever Td is below freezing, deposition happens at ice
    // saturation — a hair warmer than the dewpoint. Drawn segmented (pen up
    // through any above-freezing layers), under the Td trace.
    ctx.strokeStyle = "#7dd8ff"; ctx.lineWidth = 1.2; ctx.setLineDash([4, 4]);
    ctx.beginPath();
    let fpen = false, fBase = null;
    for (let i = 0; i < prof.P.length; i++) {
      const dC = prof.D[i] - 273.15;
      if (!isFinite(dC) || dC > 0.2) { fpen = false; continue; }
      const fy = yOf(prof.P[i]), fx = xOf(frostPtC(dC), fy);
      fpen ? ctx.lineTo(fx, fy) : ctx.moveTo(fx, fy); fpen = true;
      if (!fBase) fBase = [fx, fy];
    }
    ctx.stroke(); ctx.setLineDash([]);
    if (fBase) {
      ctx.fillStyle = "#7dd8ff"; ctx.font = "600 10px Inter";
      ctx.fillText("Tf", fBase[0] + 5, fBase[1] - 3);
    }
    drawProf(prof.D.map(v => v - 273.15), TH.dwpt, 2.8);
    drawProf(prof.T.map(v => v - 273.15), TH.temp, 2.8);
  } else {
    ctx.fillStyle = TH.muted; ctx.font = "600 17px Inter"; ctx.textAlign = "center";
    ctx.fillText("pilot balloon — winds only (see hodograph →)", SK.l + pw / 2, SK.t + ph / 2);
    ctx.textAlign = "left";
  }

  // MU parcel levels, labeled with height AGL

  ctx.font = "700 12px Inter";
  ctx.textAlign = "right";
  const marks = [["LCL", o[12], TH.lcl], ["LFC", o[13], TH.lfc], ["EL", o[14], TH.el]];
  const usedY = [];                     // coincident levels (LCL=LFC) stagger down
  for (const [lab, pP, col] of marks) {
    if (pP === MISSING || !isFinite(pP)) continue;
    const y = yOf(pP), hm = interpHagl(prof, pP);
    ctx.strokeStyle = col; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(SK.l + pw - 26, y); ctx.lineTo(SK.l + pw - 4, y); ctx.stroke();
    let ly = y + 4;
    while (usedY.some(u => Math.abs(u - ly) < 13)) ly += 14;
    usedY.push(ly);
    ctx.fillStyle = col;
    ctx.fillText(`${lab} ${hm === null ? "" : Math.round(hm) + "m"}`, SK.l + pw - 30, ly);
  }
  ctx.textAlign = "left";
  // tropopause (WMO lapse-rate definition): dashed line across the diagram
  const tpM = tropopause(prof);
  if (isFinite(tpM.wmoP) && tpM.wmoP < SK.pBot && tpM.wmoP > SK.pTop) {
    const yT = yOf(tpM.wmoP);
    ctx.strokeStyle = "rgba(191,90,242,0.5)"; ctx.lineWidth = 1.2; ctx.setLineDash([7, 5]);
    ctx.beginPath(); ctx.moveTo(SK.l + 2, yT); ctx.lineTo(SK.l + pw - 2, yT); ctx.stroke();
    ctx.setLineDash([]); ctx.lineWidth = 1;
    ctx.fillStyle = "rgba(214,150,255,0.95)"; ctx.font = "600 11px Inter";
    ctx.fillText(`TROP ${(tpM.wmoZ / 1000).toFixed(1)} km · ${(tpM.wmoZ * 3.28084 / 1000).toFixed(1)} kft · ${Math.round(tpM.wmoP / 100)} hPa`,
      SK.l + 48, yT < SK.t + 20 ? yT + 14 : yT - 5);
    ctx.font = "700 12px Inter";
  }
  if (o[28] !== MISSING && o[29] !== MISSING) {
    const y0 = yOf(o[28]), y1 = yOf(o[29]);
    ctx.strokeStyle = TH.eil; ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(SK.l + 14, y0); ctx.lineTo(SK.l + 5, y0);
    ctx.lineTo(SK.l + 5, y1); ctx.lineTo(SK.l + 14, y1); ctx.stroke();
    ctx.fillStyle = TH.eil; ctx.fillText("EIL", SK.l + 8, (y0 + y1) / 2 + 4);
  }
  ctx.restore();

  // frame + axes
  ctx.strokeStyle = TH.grid; ctx.strokeRect(SK.l, SK.t, pw, ph);
  ctx.fillStyle = TH.muted; ctx.font = "12px Inter";
  for (const pp of (SK.pTop < 10000 ? [70, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
                                     : [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])) {
    const y = yOf(pp * 100);
    ctx.strokeStyle = TH.gridSub; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(SK.l, y); ctx.lineTo(SK.l + pw, y); ctx.stroke();
    ctx.textAlign = "right"; ctx.fillText(pp, SK.l - 6, y + 4); ctx.textAlign = "left";
  }
  // height ticks (km AGL) on the left inside
  ctx.font = "600 12px Inter"; ctx.fillStyle = "#a8a8c2";
  for (const km of [1, 3, 6, 9, 12, 15]) {
    const hTarget = prof.H[0] + km * 1000;
    let pAt = null;
    for (let i = 1; i < prof.P.length; i++) {
      if (prof.H[i] >= hTarget) {
        const f = (hTarget - prof.H[i - 1]) / (prof.H[i] - prof.H[i - 1]);
        pAt = prof.P[i - 1] * Math.pow(prof.P[i] / prof.P[i - 1], f);
        break;
      }
    }
    if (pAt && pAt > SK.pTop) {
      const y = yOf(pAt);
      ctx.fillRect(SK.l, y - 0.5, 8, 1);
      ctx.fillText(km + "km", SK.l + 10, y + 3.5);
    }
  }
  ctx.font = "12px Inter"; ctx.fillStyle = TH.muted;
  for (let T = -30; T <= 40; T += 10)
    ctx.fillText(T, xOf(T, SK.t + ph) - 8, SK.t + ph + 16);
  ctx.fillText("°C", SK.l + pw / 2, H - 5);
  drawBarbs(ctx, prof, W - SK.r + 40, yOf);
  drawTempAdv(prof, ctx, W - SK.r + 4, yOf);

}

// Inferred temperature advection vs height (SPC-style): assume the wind is
// geostrophic, read the thermal wind from its turning with height —
// T_adv = -f/(Rd ln(pb/pt)) * (u_mean*dv - v_mean*du). Veering = warm (NH).
// Skipped within 10 deg of the equator where f is too small for geostrophy.
function drawTempAdv(prof, ctx, x0, yOf) {
  const m = /([\d.]+)°([NS])/.exec(plotCoords || "");
  if (!m) return;
  const lat = +m[1] * (m[2] === "S" ? -1 : 1);
  if (Math.abs(lat) < 10) return;
  const f = 2 * 7.292e-5 * Math.sin(lat * Math.PI / 180), Rd = 287.05;
  const pSfc = prof.P[0], half = 12;   // 14 grazed the barbs in the packed margin
  ctx.font = "600 9px Inter"; ctx.textAlign = "center";
  ctx.fillStyle = TH.muted;
  ctx.fillText("T-adv", x0, SK.t - 6);
  for (let pb = Math.floor(pSfc / 2500) * 2500; pb - 7500 >= 25000; pb -= 7500) {
    const pt = pb - 7500;
    const ub = interpP(prof, "U", pb), vb = interpP(prof, "V", pb);
    const ut = interpP(prof, "U", pt), vt = interpP(prof, "V", pt);
    if (![ub, vb, ut, vt].every(isFinite)) continue;
    const ubar = (ub + ut) / 2, vbar = (vb + vt) / 2;
    const ta = -f / (Rd * Math.log(pb / pt)) *
               (ubar * (vt - vb) - vbar * (ut - ub)) * 3600;   // K/hr
    const wpx = Math.min(1, Math.abs(ta) / 2.5) * half;
    const y0 = yOf(pb), y1 = yOf(pt);
    if (wpx > 0.8) {
      ctx.fillStyle = ta > 0 ? "rgba(255,90,72,0.55)" : "rgba(100,170,255,0.55)";
      ctx.fillRect(x0 - wpx, y1 + 1, wpx * 2, y0 - y1 - 2);
    }
    if (Math.abs(ta) >= 0.5) {
      ctx.fillStyle = ta > 0 ? "#ff9f95" : "#9fc9ff";
      ctx.fillText(ta.toFixed(1), x0, (y0 + y1) / 2 + 3);
    }
  }
}

function drawBarbs(ctx, prof, x0, yOf) {
  let lastY = -1e9;
  for (let i = 0; i < prof.P.length; i++) {
    const y = yOf(prof.P[i]);
    const gap = typeof BARB_GAP !== "undefined" ? BARB_GAP : 22;
    if (Math.abs(y - lastY) < gap || prof.P[i] < SK.pTop) continue;
    lastY = y;
    const u = prof.U[i], v = prof.V[i];
    const kt = Math.hypot(u, v) * KT;
    ctx.save();
    ctx.translate(x0, y); ctx.rotate(Math.atan2(-u, -v));
    ctx.strokeStyle = TH.barb; ctx.lineWidth = 1.1; ctx.fillStyle = TH.barb;
    ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(0, -26); ctx.stroke();
    let rem = Math.round(kt / 5) * 5, yb = -26;
    while (rem >= 50) { ctx.beginPath(); ctx.moveTo(0, yb); ctx.lineTo(9, yb + 3.5);
      ctx.lineTo(0, yb + 7); ctx.closePath(); ctx.fill(); yb += 8; rem -= 50; }
    while (rem >= 10) { ctx.beginPath(); ctx.moveTo(0, yb); ctx.lineTo(10, yb - 4);
      ctx.stroke(); yb += 5; rem -= 10; }
    if (rem >= 5) { ctx.beginPath(); ctx.moveTo(0, yb); ctx.lineTo(5, yb - 2); ctx.stroke(); }
    ctx.restore();
  }
}

// ---------- hodograph ----------
function drawMSE(prof) {
  const cv = document.getElementById("mse");
  if (!cv) return;                       // panel retired from the dialog; table remains
  const { W, H, ctx } = fitCanvas(cv);
  ctx.fillStyle = TH.bg; ctx.fillRect(0, 0, W, H);
  const m = mseProfile(prof);
  const L = 56, R = 12, T = 92, B = 40, zTop = 12000;
  // Scale the axis to the levels we actually DRAW. Using the whole column would
  // let stratospheric h* (~490 kJ/kg — the g·z term) set the range and crush the
  // low-level structure that matters into a few pixels.
  const vals = [];
  for (let i = 0; i < m.z.length; i++) {
    if (!isFinite(m.z[i]) || m.z[i] > zTop) continue;
    if (isFinite(m.h[i])) vals.push(m.h[i]);
    if (isFinite(m.hs[i])) vals.push(m.hs[i]);
  }
  if (isFinite(m.hbl)) vals.push(m.hbl);
  const p12v = prev12Live();
  const pv = p12v ? p12v.M : null;                     // ~12 h earlier, faded
  if (pv) for (let i = 0; i < pv.z.length; i++) {
    if (!isFinite(pv.z[i]) || pv.z[i] > zTop) continue;
    if (isFinite(pv.h[i])) vals.push(pv.h[i]);
    if (isFinite(pv.hs[i])) vals.push(pv.hs[i]);
  }
  if (vals.length < 2) return;
  const pad = Math.max(1.5, (Math.max(...vals) - Math.min(...vals)) * 0.06);
  let lo = Math.min(...vals) - pad, hi = Math.max(...vals) + pad;
  const X = v => L + (W - L - R) * (v - lo) / (hi - lo);
  const Y = z => T + (H - T - B) * (1 - Math.min(z, zTop) / zTop);

  ctx.strokeStyle = "#22223a"; ctx.fillStyle = "#8b8ba3"; ctx.font = "14px Inter";
  ctx.textAlign = "right";
  for (let z = 0; z <= zTop; z += 3000) {                  // height grid
    const y = Y(z);
    ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(W - R, y); ctx.stroke();
    ctx.fillText(z / 1000 + "km", L - 3, y + 3);
  }
  // shade the layer where a boundary-layer parcel is buoyant (h_BL > h*)
  if (isFinite(m.hbl)) {
    ctx.fillStyle = "rgba(48,209,88,0.16)";
    for (let i = 1; i < m.z.length; i++) {
      if (!isFinite(m.hs[i]) || m.z[i] > zTop) continue;
      if (m.hbl > m.hs[i]) {
        const y1 = Y(m.z[i]), y0 = Y(m.z[i - 1]);
        ctx.fillRect(X(m.hs[i]), y1, Math.max(1, X(m.hbl) - X(m.hs[i])), Math.abs(y0 - y1) + 1);
      }
    }
  }
  const line = (arr, col, wd, zs) => {
    const Z = zs || m.z;
    ctx.strokeStyle = col; ctx.lineWidth = wd; ctx.beginPath(); let st = false;
    for (let i = 0; i < arr.length; i++) {
      if (!isFinite(arr[i]) || Z[i] > zTop) continue;
      const x = X(arr[i]), y = Y(Z[i]);
      st ? ctx.lineTo(x, y) : ctx.moveTo(x, y); st = true;
    }
    ctx.stroke();
  };
  if (pv) {                                                // 12 h ago, underneath
    line(pv.hs, "rgba(255,69,58,0.28)", 1.4, pv.z);
    line(pv.h, "rgba(74,155,240,0.30)", 1.4, pv.z);
  }
  line(m.hs, "#ff453a", 2);                                // saturation MSE (h*)
  line(m.h, "#4a9bf0", 2.2);                               // MSE (h)
  if (isFinite(m.hbl)) {                                   // boundary-layer h
    ctx.strokeStyle = "#ffd60a"; ctx.lineWidth = 1.4; ctx.setLineDash([5, 4]);
    ctx.beginPath(); ctx.moveTo(X(m.hbl), T); ctx.lineTo(X(m.hbl), H - B); ctx.stroke();
    ctx.setLineDash([]);
  }
  // legend — spell out what each curve actually is
  ctx.textAlign = "left";
  ctx.font = "600 13px Inter"; ctx.fillStyle = TH.ink;
  ctx.fillText("Moist static energy", 6, 14);
  const key = [
    ["#4a9bf0", "solid", "environmental h"],
    ["#ff453a", "solid", "h* — saturation"],
    ["#ffd60a", "dash",  "boundary-layer h"],
    ["rgba(48,209,88,0.55)", "fill", "buoyant: h(BL) > h*"],
  ];
  if (pv) key.push(["rgba(160,160,190,0.5)", "solid", `faded: ${p12v.backH} h earlier`]);
  ctx.font = "11px Inter";
  key.forEach(([col, kind, lab], i) => {
    const y = 30 + i * 14;
    if (kind === "fill") { ctx.fillStyle = col; ctx.fillRect(6, y - 6, 16, 7); }
    else {
      ctx.strokeStyle = col; ctx.lineWidth = 2.4;
      ctx.setLineDash(kind === "dash" ? [4, 3] : []);
      ctx.beginPath(); ctx.moveTo(6, y - 3); ctx.lineTo(22, y - 3); ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.fillStyle = "#a8a8c2"; ctx.fillText(lab, 27, y);
  });
  // x ticks (the axis now spans only the plotted layer, so these are legible)
  ctx.fillStyle = "#8b8ba3"; ctx.strokeStyle = "#22223a"; ctx.textAlign = "center";
  ctx.font = "14px Inter";
  const span = hi - lo;
  const step = span > 140 ? 50 : span > 90 ? 30 : span > 60 ? 20
    : span > 30 ? 10 : span > 15 ? 5 : 2;
  for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) {
    ctx.beginPath(); ctx.moveTo(X(v), H - B); ctx.lineTo(X(v), H - B + 3); ctx.stroke();
    ctx.fillText(String(Math.round(v)), X(v), H - B + 13);
  }
  ctx.fillText("kJ/kg", (L + W - R) / 2, H - 4);
}

function drawHodo(prof, res) {
  const cv = document.getElementById("hodo");
  const { W, H, ctx } = fitCanvas(cv);
  ctx.fillStyle = TH.bg; ctx.fillRect(0, 0, W, H);
  const o = res.o;
  const agl = prof.H.map(h => h - prof.H[0]);
  let n = agl.findIndex(h => h > 12000); if (n < 0) n = prof.P.length;

  // knots; data-driven viewport including origin and Bunkers vectors
  const us = [], vs = [0];
  for (let i = 0; i < n; i++) { us.push(prof.U[i] * KT); vs.push(prof.V[i] * KT); }
  us.push(0);
  if (o[22] !== MISSING) { us.push(o[22] * KT, o[24] * KT); vs.push(o[23] * KT, o[25] * KT); }
  const uMin = Math.min(...us), uMax = Math.max(...us);
  const vMin = Math.min(...vs), vMax = Math.max(...vs);
  const span = Math.max(uMax - uMin, vMax - vMin, 30) * 1.25;
  const scale = (Math.min(W, H) - 30) / span;
  const cx = W / 2 - ((uMin + uMax) / 2) * scale;
  const cy = H / 2 + ((vMin + vMax) / 2) * scale;
  const X = u => cx + u * scale, Y = v => cy - v * scale;

  // rings + crosshair
  ctx.strokeStyle = "#2c2c48"; ctx.fillStyle = "#9d9dbb"; ctx.font = "14px Inter";
  const rMax = Math.ceil((span * 0.75) / 10) * 10 + 20;
  for (let r = 10; r <= rMax; r += 10) {
    ctx.beginPath(); ctx.arc(X(0), Y(0), r * scale, 0, 7); ctx.stroke();
    ctx.fillText(r, X(r) - 7, Y(0) + 14);
  }
  ctx.strokeStyle = "#34345a"; ctx.lineWidth = 1.2;
  ctx.beginPath(); ctx.moveTo(0, Y(0)); ctx.lineTo(W, Y(0)); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(X(0), 0); ctx.lineTo(X(0), H); ctx.stroke();

  // trace segments by height
  const segs = [[0, 1000, "#ff453a"], [1000, 3000, "#ff9f0a"],
                [3000, 6000, "#30d158"], [6000, 9000, "#ffd60a"], [9000, 12000, "#bf5af2"]];
  // resample on regular height steps (interpolating across level voids) so the
  // trace is continuous regardless of native level spacing
  const topAgl = Math.min(12000, agl[n - 1] ?? 12000);
  const pts = [];
  for (let h = 0; h <= topAgl; h += 125) {
    const [u, v] = windAt(prof, h);
    pts.push([h, X(u * KT), Y(v * KT)]);
  }
  // Effective inflow layer — the air the storm can actually ingest, and thus the
  // stretch of curve that generates the effective SRH. A wide translucent
  // underlay: emphasis on what's already drawn, not another label.
  const eilBot = o[28] !== MISSING ? interpHagl(prof, o[28]) : null;
  const eilTop = o[29] !== MISSING ? interpHagl(prof, o[29]) : null;
  if (eilBot !== null && eilTop !== null && eilTop > eilBot) {
    ctx.strokeStyle = "rgba(100,210,255,0.5)"; ctx.lineWidth = 12;
    ctx.lineCap = "round"; ctx.beginPath();
    let est = false;
    for (const [h, x, y] of pts) {
      if (h < eilBot || h > eilTop) continue;
      est ? ctx.lineTo(x, y) : ctx.moveTo(x, y); est = true;
    }
    ctx.stroke(); ctx.lineCap = "butt";
  }
  for (const [b, tt, col] of segs) {
    if (b > topAgl) break;
    ctx.strokeStyle = col; ctx.lineWidth = 3.4;
    ctx.beginPath();
    let started = false;
    for (const [h, x, y] of pts) {
      if (h < b || h > tt) { if (h > tt) break; continue; }
      started ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      started = true;
    }
    ctx.stroke();
  }

  // pinned comparison: thin grey hodograph under the live markers
  if (pinned && pinned.prof !== prof) {
    const pp = pinned.prof;
    const topP = Math.min(12000, (pp.H[pp.H.length - 1] || 0) - pp.H[0]);
    ctx.strokeStyle = "#9a9ab0"; ctx.lineWidth = 1.6; ctx.setLineDash([4, 4]);
    ctx.beginPath(); let st = false;
    for (let h = 0; h <= topP; h += 250) {
      const w = windAt(pp, h);
      if (!w) continue;
      const x = X(w[0] * KT), y = Y(w[1] * KT);
      st ? ctx.lineTo(x, y) : ctx.moveTo(x, y); st = true;
    }
    ctx.stroke(); ctx.setLineDash([]);
  }

  // storm motion + mean wind markers, SHARPpy-style dir/spd labels
  const markUsed = [];                  // placed label boxes; nudge collisions down
  const mark = (uMs, vMs, lab, col, hollow) => {
    if (uMs === MISSING || !isFinite(uMs)) return;
    const u = uMs * KT, v = vMs * KT;
    ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(X(u), Y(v), 6, 0, 7);
    hollow ? ctx.stroke() : ctx.fill();
    ctx.font = "700 15px Inter";
    const txt = `${Math.round(dirOf(uMs, vMs))}/${Math.round(Math.hypot(u, v))} ${lab}`;
    const tw = ctx.measureText(txt).width;
    let tx = X(u) + 9, ty = Y(v) + 4.5;
    while (markUsed.some(([ux, uy, uw]) => Math.abs(uy - ty) < 17 && tx < ux + uw && ux < tx + tw))
      ty += 18;
    markUsed.push([tx, ty, tw]);
    ctx.fillText(txt, tx, ty);
  };
  mark(o[22], o[23], "RM", "#ff453a", false);
  mark(o[24], o[25], "LM", "#64d2ff", true);
  // 0-6 km mean wind
  let mu6 = 0, mv6 = 0, c6 = 0;
  for (let i = 0; i < n && agl[i] <= 6000; i++) { mu6 += prof.U[i]; mv6 += prof.V[i]; c6++; }
  if (c6) mark(mu6 / c6, mv6 / c6, "MW", "#a8a8c2", true);


  // Effective-layer Bunkers — only when it differs from the fixed-layer vector.
  if (o[47] !== MISSING && o[22] !== MISSING &&
      Math.hypot(o[47] - o[22], o[48] - o[23]) * KT > 3)
    mark(o[47], o[48], "RM eff", "#ff8a7a", true);

  // critical angle: sfc SR inflow vs 0-500 m shear vector (RM)
  if (o[22] !== MISSING) {
    const [u5, v5] = windAt(prof, 500);
    const shU = u5 - prof.U[0], shV = v5 - prof.V[0];
    const srU = o[22] - prof.U[0], srV = o[23] - prof.V[0];
    const dot = shU * srU + shV * srV;
    const mag = Math.hypot(shU, shV) * Math.hypot(srU, srV);
    if (mag > 0.1) {
      const ang = Math.acos(Math.max(-1, Math.min(1, dot / mag))) * 180 / Math.PI;
      ctx.fillStyle = "#ffd60a"; ctx.font = "700 16px Inter";
      ctx.fillText(`Critical angle = ${ang.toFixed(0)}°`, 12, H - 12);
    }
  }
  // height-band legend (labeled colors)
  ctx.font = "600 15px Inter";
  let lx = 12;
  ctx.fillStyle = "#9d9dbb"; ctx.fillText("km AGL:", lx, 22);
  lx += ctx.measureText("km AGL:").width + 10;
  for (const [b2, t2, col] of segs) {
    const lab2 = `${b2 / 1000}–${t2 / 1000}`;
    ctx.fillStyle = col;
    ctx.fillRect(lx, 13, 16, 5);
    ctx.fillText(lab2, lx + 20, 22);
    lx += 20 + ctx.measureText(lab2).width + 14;
  }
  ctx.fillStyle = "#9d9dbb"; ctx.font = "13px Inter";
  ctx.fillText("rings: knots", 12, 40);
  ctx.fillStyle = "rgba(100,210,255,0.85)"; ctx.font = "600 13px Inter";
  ctx.fillText("▬ effective inflow layer (air the storm ingests)", 12, 60);
}

// ---------- tables ----------
const fmt = (v, d = 0, unit = "") =>
  (v === MISSING || v === undefined || !isFinite(v)) ? "—" : v.toFixed(d) + unit;

function fillTables(prof, res) {
  const o = res.o;
  lastProf = prof; lastRes = res;
  const hAt = pP => {
    const h = interpHagl(prof, pP);
    return h === null ? "—" : Math.round(h) + "m";
  };
  const rows = [["PCL", "CAPE", "ECAPE", "CINH", "LCL", "LI", "LFC", "EL"]];
  const ecapeIx = { SFC: 44, ML: 45, MU: 42 };
  const defs = [["SFC", 0, 36], ["ML", 5, 37], ["MU", 10, 38]];
  for (const [nm, i0, li] of defs) {
    const ec = o[ecapeIx[nm]];
    const ecTxt = ec === MISSING ? "—" : fmt(ec);
    rows.push([nm, fmt(o[i0]),
               nm === "MU" ? climoCell("ecape", ec === MISSING ? NaN : ec, ecTxt) : ecTxt,
               fmt(o[i0 + 1]), hAt(o[i0 + 2]), fmt(o[li], 1), hAt(o[i0 + 3]), hAt(o[i0 + 4])]);
  }
  document.getElementById("pcl-table").innerHTML = rows.map((r, i) =>
    `<tr>${r.map(c => i === 0 ? `<th>${c}</th>` : `<td>${c}</td>`).join("")}</tr>`).join("");

  const ktf = (u, v) => (u === MISSING) ? "—" : (Math.hypot(u, v) * KT).toFixed(0) + " kt";
  const vecf = (u, v) => (u === MISSING) ? "—" :
    `${Math.round(dirOf(u, v))}/${Math.round(Math.hypot(u, v) * KT)} kt`;
  const ehi = (cape, srh) => (isFinite(cape) && cape > 0 && srh !== MISSING)
    ? (cape * srh / 160000).toFixed(1) : "—";
  const effIL = (o[28] === MISSING || o[29] === MISSING) ? "—"
    : `${(o[28] / 100).toFixed(0)}–${(o[29] / 100).toFixed(0)} hPa`;
  const pair = (a, b) => `${a} <span class="pctlab">/</span> ${b}`;
  const kinem = [
    ["Shear 0–1 / 0–6 km", pair(ktf(o[20], o[21]), ktf(o[18], o[19]))],
    ["SRH 0–1 / 0–3 km", pair(fmt(o[26]), fmt(o[27])) + " m²/s²"],
    ["Eff. SRH / shear", pair(fmt(o[30]) + " m²/s²",
      o[31] === MISSING ? "—" : (o[31] * KT).toFixed(0) + " kt")],
    ["Eff. inflow", effIL],
    ["SR wind 4–6 / 9–11 km", (() => {
      const a = meanSR(prof, o, 4000, 6000), b = meanSR(prof, o, 9000, 11000);
      return (a === null && b === null) ? "—"
        : `${a === null ? "—" : a.toFixed(0)} / ${b === null ? "—" : b.toFixed(0)} kt`;
    })()],
    // Mean wind through the mixed layer: this is the momentum a well-mixed
    // boundary layer can bring DOWN to the surface, so it's the flow that
    // actually threatens to gust. Highlighted from 35 kt, ramping to deep red.
    ["PBL mean wind", (() => {
      const pp = o[46];
      if (pp === MISSING || !isFinite(pp)) return "—";
      const z = interpHagl(prof, pp);
      if (z === null || z < 50) return "—";        // too shallow to mean anything
      const mw = meanWindAgl(prof, 0, z);
      if (!mw) return "—";
      const kt = Math.hypot(mw[0], mw[1]) * KT;
      const txt = `${Math.round(dirOf(mw[0], mw[1]))}/${kt.toFixed(0)} kt`;
      if (kt < 35) return txt;
      const a = 0.25 + 0.45 * Math.min(1, (kt - 35) / 25);   // 35 kt -> 60 kt
      return `<span class="pctcell" style="background:rgba(224,70,55,${a.toFixed(2)})" ` +
        `title="${kt.toFixed(0)} kt through the mixed layer — can be mixed to the surface">${txt}</span>`;
    })()],
    ["Bunkers RM / LM", pair(vecf(o[22], o[23]), vecf(o[24], o[25]))],
    ["Corfidi up / down", pair(vecf(o[49], o[50]), vecf(o[51], o[52]))],
  ];
  const wmaxE = o[42] > 0 ? Math.sqrt(2 * o[42]) : NaN;   // ECAPE-limited updraft (Peters 2023)
  const wmaxC = o[10] > 0 ? Math.sqrt(2 * o[10]) : NaN;   // undilute CAPE updraft
  // meaningless (and explosive) when there's essentially no CAPE to dilute
  const entEff = (o[42] > 0 && o[10] >= 25) ? (100 * o[42] / o[10]) : NaN;
  const composites = [
    ["EHI 0–1 / 0–3 km", pair(ehi(o[0], o[26]), ehi(o[0], o[27]))],
    ["SCP / STP (eff.)", pair(fmt(o[32], 1), fmt(o[33], 1))],
    ["SHIP", climoCell("ship", o[43], fmt(o[43], 1))],
    ["Max updraft E / C", pair(isFinite(wmaxE) ? wmaxE.toFixed(0) : "—",
      isFinite(wmaxC) ? wmaxC.toFixed(0) + " m/s" : "—")],
    // NOT an "efficiency": SHARPlib returns E_tilde × CAPE, and E_tilde carries a
    // storm-relative kinetic-energy term, so strong SR inflow can push this >100%.
    ["BRN / BRN shear", (() => {
      const bs = brnShear(prof);
      if (bs === null) return "—";
      const cape = o[10] !== MISSING && isFinite(o[10]) ? o[10] : NaN;
      return `${isFinite(cape) && bs > 5 ? Math.round(cape / bs) : "—"} / ${Math.round(bs)} m²/s²`;
    })()],
    ["ECAPE / CAPE", isFinite(entEff) ? entEff.toFixed(0) + " %" : "—"],
  ];
  const M = mseProfile(prof);
  const mseRows = [
    ["h — boundary layer", isFinite(M.hbl) ? M.hbl.toFixed(1) + " kJ/kg" : "—"],
    ["h* minimum", isFinite(M.hsmin)
      ? `${M.hsmin.toFixed(1)} @ ${(M.hsminZ / 1000).toFixed(1)} km` : "—"],
    ["Deficit (h*−h)", isFinite(M.deficit)
      ? (M.deficit >= 0 ? "+" : "") + M.deficit.toFixed(1) + " kJ/kg" : "—"],
    ["Column ∫h dp/g", isFinite(M.col) ? (M.col / 1e9).toFixed(2) + " GJ/m²" : "—"],
  ];
  const p12 = prev12Live();
  if (p12) {
    const sgn = (v, d, u) => !isFinite(v) ? "—" : (v >= 0 ? "+" : "") + v.toFixed(d) + u;
    const tp = `vs the ${p12.dt}Z sounding (${p12.backH} h earlier). Boundary-layer h ` +
      `rising against steady or falling h*min = destabilizing; column ∫h is the reservoir ` +
      `convection drains.`;
    mseRows.push([`Δ${p12.backH}h h_BL / h*min`,
      `<span title="${tp}">${sgn(M.hbl - p12.M.hbl, 1, "")} / ` +
      `${sgn(M.hsmin - p12.M.hsmin, 1, " kJ/kg")}</span>`]);
    mseRows.push([`Δ${p12.backH}h column ∫h`,
      `<span title="${tp}">${sgn((M.col - p12.M.col) / 1e6, 0, " MJ/m²")}</span>`]);
  }
  const thermo = [
    ["DCAPE", fmt(o[39]) + " J/kg"],
    ["0–3 km CAPE", fmt(o[40]) + " J/kg"], ["NCAPE", fmt(o[41], 2)],
    ["PWAT", climoCell("pwat", o[15], (o[15] === MISSING || !isFinite(o[15])) ? "—"
      : `${o[15].toFixed(1)} mm · ${(o[15] / 25.4).toFixed(2)}"`)],
    ["Lapse 0–3 / 3–6 km", pair(fmt(o[16], 1), fmt(o[17], 1) + " K/km")],
    ["RH column / mid-lvl", (() => {
      const c = columnRH(prof), r = layerRH(prof, 70000, 50000);
      return pair(isFinite(c) ? c.toFixed(0) : "—",
                  isFinite(r) ? r.toFixed(0) + " %" : "—");
    })()],
    ["PBL top (mixing depth)", (() => {
      const pp = o[46];
      if (pp === MISSING || !isFinite(pp)) return "—";
      const zz = interpHagl(prof, pp);
      return (zz === null ? "—" : Math.round(zz) + " m AGL") + ` · ${(pp / 100).toFixed(0)} hPa`;
    })()],
  ];

  // --- level values & classic thermodynamic indices (from the profile) ---
  const C = k => { const v = interpP(prof, "T", k * 100); return isFinite(v) ? v - 273.15 : NaN; };
  const Cd = k => { const v = interpP(prof, "D", k * 100); return isFinite(v) ? v - 273.15 : NaN; };
  const t850 = C(850), t700 = C(700), t500 = C(500);
  const d850 = Cd(850), d700 = Cd(700);
  const h500 = interpP(prof, "H", 50000);          // MSL geopotential height (m)
  const h1000 = interpP(prof, "H", 100000);
  const Cw = k => { const u = interpP(prof, "U", k * 100), v = interpP(prof, "V", k * 100);
    return (isFinite(u) && isFinite(v)) ? Math.hypot(u, v) : NaN; };
  const s850 = Cw(850), s250 = Cw(250);
  const kidx = (t850 - t500) + d850 - (t700 - d700);
  const totalT = t850 + d850 - 2 * t500;
  const fzl = freezingLvlAgl(prof);
  const g = (v, u, dp = 1) => isFinite(v) ? v.toFixed(dp) + u : "—";
  thermo.push(["K-index", climoCell("kidx", kidx, g(kidx, "", 0))],
              ["Total Totals", climoCell("tott", totalT, g(totalT, "", 0))]);
  const levels = [
    ["850 hPa T / Td", climoCell("850t", t850, g(t850, "°", 0)) + " / " +
      climoCell("850td", d850, g(d850, " °C", 0))],
    ["700 hPa T / Td", climoCell("700t", t700, g(t700, "°", 0)) + " / " +
      climoCell("700td", d700, g(d700, " °C", 0))],
    ["500 hPa T", climoCell("500t", t500, g(t500, " °C", 0))],
    ["500 hPa height", climoCell("h500", h500, isFinite(h500) ? Math.round(h500) + " m" : "—")],
    ["1000–500 thick.", climoCell("thick", (isFinite(h500) && isFinite(h1000)) ? h500 - h1000 : NaN, (isFinite(h500) && isFinite(h1000)) ? Math.round(h500 - h1000) + " m" : "—")],
    ["Est. ceiling", (() => {
      const ec = estCeiling(cloudLayers(prof));
      if (ec === undefined) return "—";
      if (ec === null) return "none";
      if (ec.obscured) return "obscured (fog)";
      return `${Math.round(ec.zb)} m · ${Math.round(ec.zb * 3.28084 / 100) * 100} ft (~${ec.conf}%)`;
    })()],
    ["850 / 250 wind", climoCell("850spd", s850, isFinite(s850) ? Math.round(s850 * 1.94384) + " kt" : "—") + " / " +
      climoCell("250spd", s250, isFinite(s250) ? Math.round(s250 * 1.94384) + " kt" : "—")],
    ["Freezing level", climoCell("fzl", fzl, isFinite(fzl) ? Math.round(fzl) + " m AGL" : "—")],
    ["Wet-bulb 0 °C", (() => { const w = wbzAgl(prof); return climoCell("wbz", w, isFinite(w) ? Math.round(w) + " m AGL" : "—"); })()],
    ["Tropopause (WMO)", (() => { const tp = tropopause(prof);
      if (!isFinite(tp.wmoZ)) return "—";
      return `${(tp.wmoZ / 1000).toFixed(1)} km · ${(tp.wmoZ * 3.28084 / 1000).toFixed(1)} kft · ${Math.round(tp.wmoP / 100)} hPa`;
    })()],
  ];

  climoNow = { pwat: o[15], "850t": t850, "700t": t700, "500t": t500,
    "850td": d850, "700td": d700,
    h500: h500, thick: (isFinite(h500) && isFinite(h1000)) ? h500 - h1000 : NaN,
    fzl: fzl, kidx: kidx, tott: totalT,
    ecape: o[42] === MISSING ? NaN : o[42], ship: o[43] === MISSING ? NaN : o[43],
    "850spd": s850, "250spd": s250,
    wbz: wbzAgl(prof) };
  document.getElementById("climo-btn").style.display = climo ? "" : "none";
  document.getElementById("rec-btn").style.display = climo ? "" : "none";

  const winter = [
    ["Precip type (Bourgouin)", (() => {
      // Always answer. This says what WOULD fall if it were precipitating, and
      // that's a year-round question — a warm column reads "Rain", not blank.
      const pt = precipType(prof);
      if (!isFinite(pt.PA)) return "—";
      const detail = pt.PA > 0
        ? `melt ${pt.PA.toFixed(0)}${pt.NA > 0 ? " / refreeze " + pt.NA.toFixed(0) : ""} J/kg`
        : "no melting layer";
      return `<b>${pt.type}</b> <span class="pctlab">${detail}</span>`;
    })()],
    ["Dendritic zone", (() => {
      const b = o[53], tp = o[54];
      if (b === MISSING || tp === MISSING || !isFinite(b) || !isFinite(tp)) return "—";
      const zb = interpHagl(prof, b), zt = interpHagl(prof, tp);
      const depth = (zb !== null && zt !== null) ? Math.round(zt - zb) : null;
      const rh = layerRH(prof, b, tp);
      return `${(b / 100).toFixed(0)}–${(tp / 100).toFixed(0)} hPa` +
        (depth !== null ? ` · ${depth} m` : "") +
        (isFinite(rh) ? ` · RH ${rh.toFixed(0)}%` : "");
    })()],
  ];

  // --- tropical: is the column primed for deep, efficient, warm-rain convection ---
  const the0 = thetaEK(prof.P[0], prof.T[0], prof.D[0]);
  let theMin = Infinity, theMinP = NaN;
  for (let i = 0; i < prof.P.length; i++) {
    const p = prof.P[i];
    if (p > 85000 || p < 40000) continue;
    const te = thetaEK(p, prof.T[i], prof.D[i]);
    if (isFinite(te) && te < theMin) { theMin = te; theMinP = p; }
  }
  if (!isFinite(theMin) || theMin === Infinity) { theMin = NaN; }
  const dThe = the0 - theMin;
  const lclZ = interpHagl(prof, o[2]);              // SB parcel LCL
  const wcd = (isFinite(fzl) && lclZ !== null) ? Math.max(0, fzl - lclZ) : NaN;
  const du = interpP(prof, "U", 20000) - interpP(prof, "U", 85000);
  const dv = interpP(prof, "V", 20000) - interpP(prof, "V", 85000);
  const deepShr = (isFinite(du) && isFinite(dv)) ? Math.hypot(du, dv) * KT : NaN;
  const tt = (txt, tip) => `<span title="${tip}">${txt}</span>`;
  const tropical = [
    ["θe sfc / min aloft", (isFinite(the0) && isFinite(theMin))
      ? tt(`${the0.toFixed(0)} / ${theMin.toFixed(0)} K @ ${(theMinP / 100).toFixed(0)} hPa`,
           "Bolton (1980) equivalent potential temperature at the surface, and its minimum " +
           "between 850 and 400 hPa — the driest, lowest-energy air a storm can entrain or " +
           "bring down") : "—"],
    ["Δθe (sfc − min)", isFinite(dThe)
      ? tt(`${dThe.toFixed(0)} K`,
           "convective overturning potential: ≳20 K = strong downdrafts and cold pools " +
           "(squally, self-destructive); ≲10 K = moist-neutral column, the steady rain / " +
           "tropical-cyclone regime") : "—"],
    ["Warm cloud depth", isFinite(wcd)
      ? tt(`${(wcd / 1000).toFixed(1)} km`,
           "LCL to the freezing level: the layer where collision–coalescence makes rain. " +
           "≳4 km = high precipitation efficiency (and a rainfall-flood signal); shallow " +
           "warm layers favor ice-process hail instead") : "—"],
    ["Shear 850–200", isFinite(deepShr)
      ? tt(`${deepShr.toFixed(0)} kt`,
           "deep-layer vector shear, the tropical-cyclone metric: ≲15 kt favors genesis / " +
           "intensification, ≳25 kt ventilates and tilts the core") : "—"],
  ];

  const row = ([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`;
  document.getElementById("kin-table").innerHTML = kinem.map(row).join("");
  document.getElementById("kin-table-b").innerHTML = composites.map(row).join("");
  document.getElementById("kin-table2").innerHTML = thermo.map(row).join("");
  document.getElementById("kin-table3").innerHTML = levels.map(row).join("");
  document.getElementById("mse-table").innerHTML = mseRows.map(row).join("");
  document.getElementById("winter-table").innerHTML = winter.map(row).join("");
  document.getElementById("tropic-table").innerHTML = tropical.map(row).join("");
  queuePrev12();                        // MSE tendency arrives async, re-fills
  fitLayout();
}

// Autoscale the parameter tables into whatever height the plots leave over:
// step the font down (never below 0.55rem) until nothing scrolls, and let a
// big monitor keep the full 0.68rem. Runs after every fill and on resize.
function fitTables() {
  const wrap = document.querySelector(".tables-wrap");
  if (!wrap) return;
  // fits = nothing scrolls vertically AND no table (the 8-column parcels grid
  // is the wide one) overflows its multi-column column
  const fits = () => wrap.clientHeight >= wrap.scrollHeight - 1 &&
    [...wrap.querySelectorAll("table.params")].every(t => t.scrollWidth <= t.parentElement.clientWidth + 1);
  wrap.style.removeProperty("--tbl-fs");             // start from the CSS default
  if (fits()) return;
  for (let fs = 0.66; fs >= 0.549; fs -= 0.03) {
    wrap.style.setProperty("--tbl-fs", fs + "rem");
    if (fits()) return;
  }
}
// Size the sounding card. The index tables get at most ~a third of the body
// (CSS max-height; fitTables shrinks their font to fit, and they scroll only
// if that fails), the plots take the rest; both plots are height-driven at a
// fixed aspect, so the width the card NEEDS is known once the plot height is.
//  1. Tightest card: start wide and only narrow (the tables get taller as the
//     card narrows, so a chase in both directions would oscillate) until the
//     card just fits its two plots — nothing letterboxed, no space wasted.
//  2. If that leaves the tables squeezed into few columns at a tiny font, widen
//     — up to ~1.4x — to the narrowest width where they fit at >= 0.6rem (or
//     at least fit at all). The plots don't change: they keep their height and
//     sit centred with a little room either side, which is fine on a wide
//     monitor and a better trade than 10px type.
let inRender = false;                 // render()/redrawCharts() draw right after fitLayout — no double draw
function fitLayout() {
  const dlg = modal.querySelector(".dialog"), main = modal.querySelector(".main");
  const wrap = modal.querySelector(".tables-wrap");
  if (!dlg || !main || !wrap) return;
  if (MOBILE_MQ.matches || modal.hidden) { dlg.style.width = ""; return; }
  fitLayoutCore(dlg, main, wrap);
  // Anything that re-fills the tables (the async 12-h MSE tendency, a unit
  // toggle) can move the plot boxes; the ResizeObserver only sees .main, so
  // redraw here whenever a plot's box no longer matches its bitmap.
  if (inRender || !lastProf || !lastRes) return;
  const sk = document.getElementById("skewt");
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const { w, h } = plotBox(sk);
  if (Math.round(w * dpr) !== sk.width || Math.round(h * dpr) !== sk.height) {
    drawSkewT(lastProf, lastRes); drawHodo(lastProf, lastRes);
  }
}
function fitLayoutCore(dlg, main, wrap) {
  const chromeW = 12 + 2 * 18 + 28 + 2;   // grid gap + panel padding/border + dialog padding/border
  const ratio = PLOT_ASPECT.skewt + PLOT_ASPECT.hodo;
  const maxW = Math.floor(window.innerWidth * 0.98), minW = Math.min(980, maxW);
  const setW = w => { dlg.style.width = w + "px"; };
  const need = () => Math.round((main.clientHeight - 18) * ratio + chromeW);
  const tblFont = () => parseFloat(wrap.style.getPropertyValue("--tbl-fs")) || 0.68;
  const tblFits = () => wrap.clientHeight >= wrap.scrollHeight - 1;
  // 1. tight
  let cur = maxW; setW(cur);
  for (let i = 0; i < 5; i++) {
    fitTables();
    const w = Math.max(minW, Math.min(need(), cur));
    if (cur - w < 1.5) break;
    setW(w); cur = w;
  }
  fitTables();
  const tight = cur;
  if (tblFits() && tblFont() >= 0.6) return;
  // 2. widen for the tables: the narrowest width where they fit at >= 0.6rem;
  //    failing that the narrowest where they fit at all; failing THAT (short
  //    laptop screens) whichever width leaves the least to scroll
  let best = tight, bestFits = tblFits(), bestOver = wrap.scrollHeight - wrap.clientHeight;
  const lim = Math.min(maxW, Math.round(tight * 1.4));
  for (let w = tight; w < lim; ) {
    w = Math.min(lim, Math.round(w * 1.07));
    setW(w); fitTables();
    const fits = tblFits(), fs = tblFont(), over = wrap.scrollHeight - wrap.clientHeight;
    if (fits && fs >= 0.6) return;                    // narrowest comfortable width
    if (fits ? !bestFits : (!bestFits && over < bestOver - 4)) {
      best = w; bestFits = fits; bestOver = over;
    }
  }
  setW(best); fitTables();
}
document.getElementById("dlg-footer").addEventListener("click",
  e => { if (e.target.tagName !== "A") e.currentTarget.classList.toggle("open"); });

// ---------- render ----------
function clearPlot(msg) {
  setStatus(msg || "no data");
  sizePlots();
  for (const id of ["skewt", "hodo", "mse"]) {
    const cv = document.getElementById(id);
    if (!cv) continue;
    const ctx = cv.getContext("2d");
    ctx.fillStyle = "#0a0a14"; ctx.fillRect(0, 0, cv.width, cv.height);
    ctx.fillStyle = "#8b8ba3"; ctx.font = "600 16px Inter"; ctx.textAlign = "center";
    ctx.fillText(msg || "no data", cv.width / 2, cv.height / 2);
    ctx.textAlign = "left";
  }
  for (const id of ["pcl-table", "kin-table", "kin-table-b", "kin-table2", "kin-table3", "mse-table", "winter-table", "tropic-table"])
    document.getElementById(id).innerHTML = "";
  plotTitle = "";
  lastProf = lastRes = null;      // else a resize redraw resurrects the wiped chart
}

// Kinematics for WIND-ONLY (pilot balloon) soundings: shear, Bunkers and SRH
// are pure wind quantities — no thermodynamics required.
function jsKinematics(prof, o) {
  const w = h => windAt(prof, h);
  const w0 = w(0), w1 = w(1000), w6 = w(6000);
  if (!w0 || !w6) return;
  o[18] = w6[0] - w0[0]; o[19] = w6[1] - w0[1];        // 0-6 km shear
  if (w1) { o[20] = w1[0] - w0[0]; o[21] = w1[1] - w0[1]; }
  const mw = meanWindAgl(prof, 0, 6000);
  if (mw) {
    const sm = Math.hypot(o[18], o[19]);
    if (sm > 0.5) {                                     // Bunkers ±7.5 m/s off the shear
      const dx = 7.5 * o[19] / sm, dy = -7.5 * o[18] / sm;
      o[22] = mw[0] + dx; o[23] = mw[1] + dy;           // right mover
      o[24] = mw[0] - dx; o[25] = mw[1] - dy;           // left mover
      const srh = (top, cu, cv) => {
        let s = 0, prev = null;
        for (let h = 0; h <= top; h += 100) {
          const uv = w(h); if (!uv) continue;
          if (prev) s += (uv[0] - cu) * (prev[1] - cv) - (prev[0] - cu) * (uv[1] - cv);
          prev = uv;
        }
        return s;
      };
      o[26] = srh(1000, o[22], o[23]);                  // SRH 0-1 (RM)
      o[27] = srh(3000, o[22], o[23]);                  // SRH 0-3 (RM)
    }
  }
}

function render(prof) {
  sanitizeHeights(prof);                   // before ANY analysis touches it
  fillDewpoints(prof);                     // a lone NaN would void PWAT, CRH, MSE…
  fillWinds(prof);                         // interpolate wind gaps, never truncate
  const hasT = prof.T.some(v => isFinite(v));
  const res = hasT ? compute(prof) : { o: new Array(64).fill(MISSING),
    sb: [], ml: [], mu: [] };
  if (!hasT) jsKinematics(prof, res.o);    // pibal: winds still give shear/SRH/Bunkers
  if (res.rc === 2) plotNote = "⚠ analysis failed on this profile — charts only";
  if (prof.dewSynth) plotNote = (plotNote ? plotNote + "   " : "") +
    "⚠ NO DEWPOINT DATA — green trace synthesized (T−30°); moisture parameters invalid";
  lastProf = prof; lastRes = res;
  // deep tropical soundings: the tropopause/cold point sits near 100 hPa and
  // was clipped at the old fixed plot top — extend to 70 hPa when the trop is
  // high and the data reaches it (Majuro 2026-07-11 topped at 64 hPa)
  const tpTop = tropopause(prof);
  SK.pTop = (isFinite(tpTop.wmoP) && tpTop.wmoP < 12500 &&
             prof.P[prof.P.length - 1] < 9000) ? 7000 : 10000;
  // tables first: fillTables -> fitLayout settles the card width, so the
  // plots draw once at their final size instead of drawing, then redrawing
  inRender = true;
  try {
    if (hasT) fillTables(prof, res);
    else try { fillTables(prof, res); } catch (e) { /* fall back to clearing */ }
  } finally { inRender = false; }
  drawSkewT(prof, res);
  drawHodo(prof, res);
  drawMSE(prof);
  if (false) for (const id of ["pcl-table", "kin-table", "kin-table-b", "kin-table2", "kin-table3", "mse-table", "winter-table"])
    document.getElementById(id).innerHTML = "";
}
