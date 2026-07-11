/* Skew-T Explorer — all client-side.
   Data: U. Wyoming sounding archive, mirrored to the skewt-data branch by
   .github/workflows/skewt-data.yml (UW sends no CORS headers; raw.github
   serves ACAO *). Physics: SHARPlib via WebAssembly (sharplib.js/.wasm). */
"use strict";

const MISSING = -9999.0;
const CLIMO_BASE = "https://raw.githubusercontent.com/scorvec/scorvec.github.io/skewt-climo/climo/";
let climo = null, climoGid = null;                 // current station climatology
let lastProf = null, lastRes = null, lastMonth = null, lastDoy = null;
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
    const r = await fetch(CLIMO_BASE + gid + ".json");
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
  ["h500", "500 hPa height", "m"], ["thick", "1000–500 hPa thickness", "m"],
  ["fzl", "Freezing level (AGL)", "m"], ["wbz", "Wet-bulb 0 °C (AGL)", "m"],
  ["kidx", "K-index", ""], ["tott", "Total Totals", ""],
  ["ecape", "ECAPE (MU)", "J/kg"], ["ship", "Significant Hail (SHIP)", ""],
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

document.getElementById("climo-btn").addEventListener("click", openClimo);
document.getElementById("climo-var").addEventListener("change", e => drawClimo(e.target.value));
document.getElementById("climo-close").addEventListener("click", () => climoModal.hidden = true);
climoModal.addEventListener("click", e => { if (e.target === climoModal) climoModal.hidden = true; });
document.addEventListener("keydown", e => {
  if (e.key === "Escape" && !climoModal.hidden) climoModal.hidden = true;
});
const CLIMO_PCTS = [1, 5, 10, 25, 50, 75, 90, 95, 99];
const CLIMO_MIN_N = 30;                  // a "record" from 4 soundings is noise
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
  const rec = v >= d.max ? { t: "high", y: d.maxY } : v <= d.min ? { t: "low", y: d.minY } : null;
  return { pct: Math.max(0, Math.min(100, pct)), rec };
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
  const tip = r.rec ? `record ${r.rec.t} ${r.rec.y}` : `${ord}th percentile`;
  const star = r.rec
    ? ` <span style="color:${r.rec.t === "high" ? "#ff5a3c" : "#5a9bf0"}">★${String(r.rec.y).slice(2)}</span>`
    : "";
  const sub = r.rec ? "" : `<span class="pctlab">P${ord}</span>`;
  return `<span class="pctcell" style="background:${pctColor(r.pct)}" title="${tip}">${txt}${star}</span>${sub}`;
}
const MIRROR = "https://raw.githubusercontent.com/scorvec/scorvec.github.io/skewt-data/";
const SPC = "https://www.spc.noaa.gov/exper/soundings";
const IEM = "https://mesonet.agron.iastate.edu/json/raob.py";
let iemMap = null;                                  // WMO -> ICAO (US/Canada)
fetch("iem_raob.json").then(r => r.ok ? r.json() : {}).then(m => { iemMap = m; })
  .catch(() => { iemMap = {}; });

// the three most recent synoptic slots, newest first
function synopticSlots() {
  const out = [], now = Date.now();
  for (let back = 0; back <= 2; back++) {
    const d = new Date(now - back * 12 * 3600e3);
    const hh = d.getUTCHours() >= 12 ? 12 : 0;
    out.push({ y: d.getUTCFullYear(), mo: d.getUTCMonth() + 1, d: d.getUTCDate(), hh });
  }
  return out;
}
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
let plotTitle = "";              // drawn on the skew-t canvas itself
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
const WASM_V = "6ce3fd46";
const wasmReady = createSharp({
  locateFile: (f) => f.endsWith(".wasm") ? `${f}?v=${WASM_V}` : f,
}).then(mod => { M = mod; });

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
function openModal() { modal.hidden = false; }
function closeModal() { modal.hidden = true; }
document.getElementById("close").onclick = closeModal;
modal.addEventListener("click", e => { if (e.target === modal) closeModal(); });
addEventListener("keydown", e => { if (e.key === "Escape") closeModal(); });
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
  { attribution: "&copy; OpenStreetMap", maxZoom: 10 }).addTo(map);

const closedLayer = L.layerGroup();          // stations with no data since 2024
const ACTIVE_YEAR = 2025;

Promise.all([
  fetch("stations.json").then(r => r.json()),
  fetch(MIRROR + "manifest.json?t=" + Date.now()).then(r => r.json()).catch(() => null),
  fetch(MIRROR + "anomalies.json?t=" + Date.now()).then(r => r.ok ? r.json() : {}).catch(() => ({})),
]).then(([stns, man, anom]) => {
  entries = (man && man.entries) || {};
  anomalies = anom || {};
  for (const s of stns.stations) {
    igraStations[s.gid] = s;
    if (s.id) byWmo[s.id] = s.gid;
  }
  // closed stations: archive-only, hidden behind the toggle
  for (const s of stns.stations) {
    if (s.y1 >= ACTIVE_YEAR || (s.id && entries[s.id])) continue;
    const m = L.circleMarker([s.la, s.lo], {
      radius: RAD.closed, weight: 0.5, color: "#7a1f1f", fillColor: "#c0392b", fillOpacity: 0.75,
    }).addTo(closedLayer);
    m.bindTooltip(`${s.n} (${s.gid}) · closed ${s.y0}–${s.y1} — click for archive`);
    m.on("click", () => { setMode("archive"); highlight(m); selectStation(s); });
  }
  // active stations without a launch in the mirror window: small grey
  for (const s of stns.stations) {
    if (s.y1 < ACTIVE_YEAR || (s.id && entries[s.id])) continue;
    const m = L.circleMarker([s.la, s.lo], {
      radius: RAD.active, weight: 1, color: "#1f7a5e", fillColor: "#33c495", fillOpacity: 0.85,
    }).addTo(map);
    m.bindTooltip(`${s.n} (${s.gid}) · no launch in last 36 h — click for archive (${s.y0}–${s.y1})`);
    m.on("click", () => { setMode("archive"); highlight(m); selectStation(s); });
  }
  // stations with a sounding in the last 36 h: big blue, on top
  for (const [id, s] of Object.entries(entries)) {
    const ig = igraStations[byWmo[id]];
    if (ig || /dtype|Name:/i.test(s.n || "")) s.n = (ig && ig.n) || id;
    const flag = anomalies[id];
    if (flag) {                                          // record watch: gold ring underneath
      L.circleMarker([s.la, s.lo], { radius: RAD.live + 4, weight: 3,
        color: "#ff2d2d", opacity: 0.95, fill: false }).addTo(map);
    }
    const m = L.circleMarker([s.la, s.lo], {
      radius: RAD.live, weight: 1.5, color: "#1d3a5e", fillColor: "#4a7ab5", fillOpacity: 0.95,
    }).addTo(map);
    const arch = ig ? ` · archive ${ig.y0}–${ig.y1}` : "";
    const anomTip = flag ? `<br><b style="color:#ff2d2d">⚡ near record:</b> ` +
      flag.flags.map(f => `${f.lab} ${f.v} (${f.sense === "high" ? "P" + f.pct + " high" : "P" + f.pct + " low"})`).join(", ") : "";
    m.bindTooltip(`${s.n || id} (${id}) · latest ${s.dt}Z${arch}${anomTip}`);
    m.on("click", () => {
      highlight(m);   // respects the current Latest/Archive mode + chosen date
      selectStation({ gid: byWmo[id], id, n: s.n, e: (igraStations[byWmo[id]] || {}).e || 0 });
    });
  }
  if (!man) setStatus("live mirror unavailable — archive mode still works");
  const want = location.hash.replace("#", "");
  if (want && entries[want]) {
    selectStation({ gid: byWmo[want], id: want, n: (entries[want] || {}).n,
                    e: (igraStations[byWmo[want]] || {}).e || 0 });
  }
});

// legend + closed-station toggle (Leaflet control)
const legend = L.control({ position: "bottomleft" });
legend.onAdd = () => {
  const div = L.DomUtil.create("div");
  div.className += " maplegend";
  div.innerHTML =
    '<span class="leg-chip" title="legend">ⓘ&nbsp;key</span><div class="leg-body">' +
    '<span style="color:#4a7ab5;font-size:1.05em">●</span> sounding in last 36 h &nbsp; ' +
    '<span style="color:#33c495;font-size:1.05em">●</span> active &nbsp; ' +
    '<span style="color:#c0392b;font-size:1.05em">●</span> closed &nbsp; ' +
    '<span style="color:#ff2d2d">◎</span> near record &nbsp; ' +
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
  if (selectedMarker) selectedMarker.setStyle({ weight: selectedMarker._baseW || 1 });
  marker._baseW = marker.options.weight;
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

function maybeReload() { if (current && !modal.hidden) loadSounding(); }

function stepDate(days) {
  const d = new Date(archDate + "T00:00:00Z");
  archDate = new Date(d.getTime() + days * 864e5).toISOString().slice(0, 10);
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

function selectStation(s) {
  current = s;
  openModal();
  loadClimo(s.gid);
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
    if (![p, h, t, d, wd, ws].every(isFinite) || p >= lastP || p < 2000) continue;
    lastP = p;
    out.P.push(p); out.H.push(h); out.T.push(t); out.D.push(d);
    out.U.push(-ws * Math.sin(wd * Math.PI / 180));
    out.V.push(-ws * Math.cos(wd * Math.PI / 180));
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
    urls.push(IGRA + `data-y2d/${gid}-data-beg${new Date().getUTCFullYear() - 1}.txt.zip`);
  }
  urls.push(IGRA + `data-por/${gid}-data.txt.zip`);
  for (const url of urls) {
    try {
      const r = await fetch(url);
      if (!r.ok) continue;
      const mb = (+r.headers.get("content-length") / 1048576).toFixed(1);
      setStatus(`downloading IGRA archive (${mb} MB)…`, true);
      const buf = new Uint8Array(await r.arrayBuffer());
      setStatus("decompressing…", true);
      const files = fflate.unzipSync(buf);
      const text = fflate.strFromU8(files[Object.keys(files)[0]]);
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
    const wd = iv(L, 40, 45), ws = iv(L, 46, 51);
    // pilot-balloon levels report by height with pressure missing
    if (p === null && gph !== null && gph > -900) p = stdP(gph);
    if (p === null || p < 2000) continue;
    if (wd !== null && ws !== null && ws >= 0)
      winds.push({ p, h: gph, u: -(ws / 10) * Math.sin(wd * Math.PI / 180),
                              v: -(ws / 10) * Math.cos(wd * Math.PI / 180) });
    if (tt !== null && tt > -8888 && dpdp !== null)
      thermo.push({ p, gph, T: tt / 10 + 273.15, D: tt / 10 + 273.15 - dpdp / 10 });
  }
  thermo.sort((a, b) => b.p - a.p);
  winds.sort((a, b) => b.p - a.p);
  const windOnly = thermo.length < 8;
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
  return out.P.length >= 8 ? { prof: out, hh: best.hh, nWind: winds.length } : null;
}

async function loadSounding() {
  if (!current) return;
  await wasmReady;
  if (mode === "latest") {
    const s = entries[current.id];
    // real-time first: SPC (fastest US), then IEM (US/Canada, more levels)
    if (current.id && iemMap && iemMap[current.id]) {
      setStatus("fetching real-time sounding…", true);
      const got = await fetchSPC(current.id).catch(() => null)
        || await fetchIEM(current.id).catch(() => null);
      if (got) {
        setStatus(`valid ${got.valid}Z · ${got.prof.P.length} levels (${got.src})`);
        plotTitle = `${current.n || ""} ${current.id}  ·  ${got.valid}Z`.trim();
        plotNote = ""; lastMonth = got.valid.slice(5, 7);
        render(thin(got.prof));
        return;
      }
    }
    if (!s) {
      clearPlot("station not in the live feed — switch to Archive (IGRA)");
      return;
    }
    setStatus("fetching…", true);
    try {
      const r = await fetch(MIRROR + "soundings/" + current.id + ".csv?t=" + s.dt);
      if (!r.ok) throw 0;
      const prof = parseCSV(await r.text());
      if (!prof) throw 0;
      setStatus(`valid ${s.dt}Z · ${prof.P.length} levels (UW BUFR/GTS mirror)`);
      plotTitle = `${current.n || ""} ${current.id}  ·  ${s.dt}Z`.trim();
      plotNote = ""; lastMonth = s.dt.slice(5, 7); lastDoy = doyOf(s.dt);
      render(thin(prof));
    } catch (e) {
      clearPlot("error: " + (e && e.message ? e.message : "fetch failed"));
    }
    return;
  }
  // archive mode: recent launches come from the high-res UW mirror when
  // available (BUFR fidelity, ~4-day retention), else NOAA IGRA v2
  const ymd = archDate;
  const me = current.id ? entries[current.id] : null;
  const wantDt = `${ymd} ${String(archHour).padStart(2, "0")}:00`;
  if (me && (me.hours || []).includes(wantDt)) {
    setStatus("fetching high-resolution sounding from the mirror…", true);
    try {
      const tag = wantDt.replace(/[-: ]/g, "").slice(0, 10);
      const r = await fetch(MIRROR + "soundings/" + current.id + "_" + tag + ".csv?t=" + tag);
      if (r.ok) {
        const prof = parseCSV(await r.text());
        if (prof) {
          setStatus(`valid ${wantDt}Z · ${prof.P.length} levels (UW BUFR high-res mirror)`);
          plotTitle = `${current.n || ""} ${current.id}  ·  ${wantDt}Z`.trim();
          plotNote = ""; lastMonth = wantDt.slice(5, 7); lastDoy = doyOf(wantDt);
          render(thin(prof));
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
        const r = await fetch(UW_ARCHIVE + "uw-" + dkey + ".zip");
        dayZipCache.set(dkey, r.ok
          ? fflate.unzipSync(new Uint8Array(await r.arrayBuffer())) : null);
      } catch (e) { dayZipCache.set(dkey, null); }
    }
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
          plotNote = ""; lastMonth = ymd.slice(5, 7); lastDoy = doyOf(ymd);
          render(thin(prof));
          return;
        }
      }
    }
  }
  if (!current.gid) { setStatus("station not in the IGRA archive"); return; }
  const text = await igraText(current.gid, +ymd.slice(0, 4));
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
  lastMonth = shown.slice(5, 7); lastDoy = doyOf(shown);
  render(thin(got.prof));
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
  parcelMU: "#ffffff", parcelSB: "#ff9f0a",
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
  if (!isFinite(pPa) || pPa >= P[0]) return A[0];
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

// ---------- skew-t drawing ----------
const SK = { l: 58, r: 90, t: 40, b: 42, pBot: 105000, pTop: 10000, tL: -35, tR: 45 };

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

function fitCanvas(cv) {                    // backing store = panel size × dpr
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = cv.clientWidth, h = cv.clientHeight;
  const ctx = cv.getContext("2d");
  if (w && h) {
    cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { W: w, H: h, ctx };
  }
  return { W: cv.width, H: cv.height, ctx };
}
function drawSkewT(prof, res) {
  const cv = document.getElementById("skewt");
  const { W, H, ctx } = fitCanvas(cv);
  const pw = W - SK.l - SK.r, ph = H - SK.t - SK.b;
  const yOf = p => SK.t + (1 - Math.log(SK.pBot / p) / Math.log(SK.pBot / SK.pTop)) * ph;
  const xOf = (tC, y) => SK.l + ((tC - SK.tL) / (SK.tR - SK.tL)) * pw + ((SK.t + ph) - y);
  ctx.fillStyle = TH.bg; ctx.fillRect(0, 0, W, H);
  // on-canvas title: station + valid time
  ctx.fillStyle = TH.ink; ctx.font = "700 16px Inter";
  ctx.fillText(plotTitle, SK.l, 22);
  ctx.fillStyle = TH.muted; ctx.font = "12px Inter";
  ctx.fillText("SHARPlib · scorvec.com/skewt", W - 190, 22);
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
  for (let lp = Math.log(105000); lp >= Math.log(10000); lp -= 0.03) pGrid.push(Math.exp(lp));
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
  drawTrace(res.sb, TH.parcelSB, 1.4, [5, 4]);
  drawTrace(res.mu, TH.parcelMU, 1.8, [6, 4]);

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
    drawProf(prof.D.map(v => v - 273.15), TH.dwpt, 2.8);
    drawProf(prof.T.map(v => v - 273.15), TH.temp, 2.8);
  } else {
    ctx.fillStyle = TH.muted; ctx.font = "600 17px Inter"; ctx.textAlign = "center";
    ctx.fillText("pilot balloon — winds only (see hodograph →)", SK.l + pw / 2, SK.t + ph / 2);
    ctx.textAlign = "left";
  }

  // MU parcel levels, labeled with height AGL
  const o = res.o;
  ctx.font = "700 12px Inter";
  ctx.textAlign = "right";
  const marks = [["LCL", o[12], TH.lcl], ["LFC", o[13], TH.lfc], ["EL", o[14], TH.el]];
  for (const [lab, pP, col] of marks) {
    if (pP === MISSING || !isFinite(pP)) continue;
    const y = yOf(pP), hm = interpHagl(prof, pP);
    ctx.strokeStyle = col; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(SK.l + pw - 26, y); ctx.lineTo(SK.l + pw - 4, y); ctx.stroke();
    ctx.fillStyle = col;
    ctx.fillText(`${lab} ${hm === null ? "" : Math.round(hm) + "m"}`, SK.l + pw - 30, y + 4);
  }
  ctx.textAlign = "left";
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
  for (let pp = 100; pp <= 1000; pp += 100) {
    const y = yOf(pp * 100);
    ctx.strokeStyle = TH.gridSub; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(SK.l, y); ctx.lineTo(SK.l + pw, y); ctx.stroke();
    ctx.fillText(pp, 14, y + 4);
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
  ctx.fillText("°C", SK.l + pw / 2, H - 6);
  drawBarbs(ctx, prof, W - SK.r + 40, yOf);
}

function drawBarbs(ctx, prof, x0, yOf) {
  let lastY = -1e9;
  for (let i = 0; i < prof.P.length; i++) {
    const y = yOf(prof.P[i]);
    if (Math.abs(y - lastY) < 22 || prof.P[i] < SK.pTop) continue;
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
  const line = (arr, col, wd) => {
    ctx.strokeStyle = col; ctx.lineWidth = wd; ctx.beginPath(); let st = false;
    for (let i = 0; i < arr.length; i++) {
      if (!isFinite(arr[i]) || m.z[i] > zTop) continue;
      const x = X(arr[i]), y = Y(m.z[i]);
      st ? ctx.lineTo(x, y) : ctx.moveTo(x, y); st = true;
    }
    ctx.stroke();
  };
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
    ["#4a9bf0", "solid", "h — air's energy"],
    ["#ff453a", "solid", "h* — saturation"],
    ["#ffd60a", "dash",  "h of the boundary layer"],
    ["rgba(48,209,88,0.55)", "fill", "buoyant: h(BL) > h*"],
  ];
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
  const step = span > 60 ? 20 : span > 30 ? 10 : span > 15 ? 5 : 2;
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

  // storm motion + mean wind markers, SHARPpy-style dir/spd labels
  const mark = (uMs, vMs, lab, col, hollow) => {
    if (uMs === MISSING || !isFinite(uMs)) return;
    const u = uMs * KT, v = vMs * KT;
    ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(X(u), Y(v), 6, 0, 7);
    hollow ? ctx.stroke() : ctx.fill();
    ctx.font = "700 15px Inter";
    ctx.fillText(`${Math.round(dirOf(uMs, vMs))}/${Math.round(Math.hypot(u, v))} ${lab}`,
                 X(u) + 9, Y(v) + 4.5);
  };
  mark(o[22], o[23], "RM", "#ff453a", false);
  mark(o[24], o[25], "LM", "#64d2ff", true);
  // 0-6 km mean wind
  let mu6 = 0, mv6 = 0, c6 = 0;
  for (let i = 0; i < n && agl[i] <= 6000; i++) { mu6 += prof.U[i]; mv6 += prof.V[i]; c6++; }
  if (c6) mark(mu6 / c6, mv6 / c6, "MW", "#a8a8c2", true);

  // Deviant tornado motion: midpoint of Bunkers RM and the 0-500 m mean wind.
  const mw05 = meanWindAgl(prof, 0, 500);
  if (o[22] !== MISSING && mw05)
    mark(0.5 * (o[22] + mw05[0]), 0.5 * (o[23] + mw05[1]), "DTM", "#ff9f0a", false);

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
  ctx.fillStyle = "#9d9dbb"; ctx.fillText("km AGL:", lx, 22); lx += 62;
  for (const [b2, t2, col] of segs) {
    ctx.fillStyle = col;
    ctx.fillRect(lx, 13, 16, 5);
    ctx.fillText(`${b2 / 1000}–${t2 / 1000}`, lx + 20, 22);
    lx += 58;
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
  const thermo = [
    ["DCAPE", fmt(o[39]) + " J/kg"],
    ["0–3 km CAPE", fmt(o[40]) + " J/kg"], ["NCAPE", fmt(o[41], 2)],
    ["PWAT", climoCell("pwat", o[15], o[15] === MISSING ? "—"
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
  const kidx = (t850 - t500) + d850 - (t700 - d700);
  const totalT = t850 + d850 - 2 * t500;
  const fzl = freezingLvlAgl(prof);
  const g = (v, u, dp = 1) => isFinite(v) ? v.toFixed(dp) + u : "—";
  thermo.push(["K-index", climoCell("kidx", kidx, g(kidx, "", 0))],
              ["Total Totals", climoCell("tott", totalT, g(totalT, "", 0))]);
  const levels = [
    ["850 hPa T / Td", climoCell("850t", t850, `${g(t850, "", 0)} / ${g(d850, " °C", 0)}`)],
    ["700 hPa T / Td", climoCell("700t", t700, `${g(t700, "", 0)} / ${g(d700, " °C", 0)}`)],
    ["500 hPa T", climoCell("500t", t500, g(t500, " °C", 0))],
    ["500 hPa height", climoCell("h500", h500, isFinite(h500) ? Math.round(h500) + " m" : "—")],
    ["1000–500 thick.", climoCell("thick", (isFinite(h500) && isFinite(h1000)) ? h500 - h1000 : NaN, (isFinite(h500) && isFinite(h1000)) ? Math.round(h500 - h1000) + " m" : "—")],
    ["Freezing level", climoCell("fzl", fzl, isFinite(fzl) ? Math.round(fzl) + " m AGL" : "—")],
    ["Wet-bulb 0 °C", (() => { const w = wbzAgl(prof); return climoCell("wbz", w, isFinite(w) ? Math.round(w) + " m AGL" : "—"); })()],
    ["Tropopause / cold pt", (() => { const tp = tropopause(prof);
      return pair(isFinite(tp.wmoZ) ? (tp.wmoZ / 1000).toFixed(1) : "—",
        isFinite(tp.cpZ) ? `${(tp.cpZ / 1000).toFixed(1)} km` : "—");
    })()],
  ];

  climoNow = { pwat: o[15], "850t": t850, "700t": t700, "500t": t500,
    h500: h500, thick: (isFinite(h500) && isFinite(h1000)) ? h500 - h1000 : NaN,
    fzl: fzl, kidx: kidx, tott: totalT,
    ecape: o[42] === MISSING ? NaN : o[42], ship: o[43] === MISSING ? NaN : o[43],
    wbz: wbzAgl(prof) };
  document.getElementById("climo-btn").style.display = climo ? "" : "none";

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
    ["Snow ratio / squall", (() => { const k = kucheraRatio(prof);
      return pair(isFinite(k) ? k.toFixed(0) + ":1" : "—", fmt(o[55], 1)); })()],
  ];

  const row = ([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`;
  document.getElementById("kin-table").innerHTML = kinem.map(row).join("");
  document.getElementById("kin-table-b").innerHTML = composites.map(row).join("");
  document.getElementById("kin-table2").innerHTML = thermo.map(row).join("");
  document.getElementById("kin-table3").innerHTML = levels.map(row).join("");
  document.getElementById("mse-table").innerHTML = mseRows.map(row).join("");
  document.getElementById("winter-table").innerHTML = winter.map(row).join("");
}

// ---------- render ----------
function clearPlot(msg) {
  setStatus(msg || "no data");
  for (const id of ["skewt", "hodo", "mse"]) {
    const cv = document.getElementById(id), ctx = cv.getContext("2d");
    ctx.fillStyle = "#0a0a14"; ctx.fillRect(0, 0, cv.width, cv.height);
    ctx.fillStyle = "#8b8ba3"; ctx.font = "600 16px Inter"; ctx.textAlign = "center";
    ctx.fillText(msg || "no data", cv.width / 2, cv.height / 2);
    ctx.textAlign = "left";
  }
  for (const id of ["pcl-table", "kin-table", "kin-table-b", "kin-table2", "kin-table3", "mse-table", "winter-table"])
    document.getElementById(id).innerHTML = "";
  plotTitle = "";
}

function render(prof) {
  sanitizeHeights(prof);                   // before ANY analysis touches it
  const hasT = prof.T.some(v => isFinite(v));
  const res = hasT ? compute(prof) : { o: new Array(64).fill(MISSING),
    sb: [], ml: [], mu: [] };
  if (res.rc === 2) plotNote = "⚠ analysis failed on this profile — charts only";
  lastProf = prof; lastRes = res;
  drawSkewT(prof, res);
  drawHodo(prof, res);
  drawMSE(prof);
  if (hasT) fillTables(prof, res);
  else for (const id of ["pcl-table", "kin-table", "kin-table-b", "kin-table2", "kin-table3", "mse-table", "winter-table"])
    document.getElementById(id).innerHTML = "";
}
