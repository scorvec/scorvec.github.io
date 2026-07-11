/* Skew-T Explorer — all client-side.
   Data: U. Wyoming sounding archive, mirrored to the skewt-data branch by
   .github/workflows/skewt-data.yml (UW sends no CORS headers; raw.github
   serves ACAO *). Physics: SHARPlib via WebAssembly (sharplib.js/.wasm). */
"use strict";

const MISSING = -9999.0;
const CLIMO_BASE = "https://raw.githubusercontent.com/scorvec/scorvec.github.io/skewt-climo/climo/";
let climo = null, climoGid = null;                 // current station climatology
let lastProf = null, lastRes = null, lastMonth = null;
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
    Object.values(climo.months || {}).some(m => m[k]))
    .map(([k, lab]) => `<option value="${k}">${lab}</option>`).join("");
  document.getElementById("climo-title").textContent =
    `${current.n || current.gid} · climatology`;
  climoModal.hidden = false;
  drawClimo(sel.value);
}
function drawClimo(key) {
  const meta = CLIMO_VARS.find(v => v[0] === key) || [key, key, ""];
  const cv = document.getElementById("climo-canvas"), x = cv.getContext("2d");
  const W = cv.width, H = cv.height, L = 54, R = 18, T = 18, B = 34;
  x.clearRect(0, 0, W, H);
  const months = [];
  let lo = Infinity, hi = -Infinity;
  for (let m = 1; m <= 12; m++) {
    const d = (climo.months[String(m).padStart(2, "0")] || {})[key];
    months.push(d || null);
    if (d) { lo = Math.min(lo, d.min); hi = Math.max(hi, d.max); }
  }
  if (!isFinite(lo)) return;
  const pad = (hi - lo) * 0.08 || 1; lo -= pad; hi += pad;
  const px = m => L + (W - L - R) * (m + 0.5) / 12;
  const py = v => T + (H - T - B) * (1 - (v - lo) / (hi - lo));
  // grid + y labels
  x.strokeStyle = "#26263a"; x.fillStyle = "#8b8ba3"; x.font = "11px Inter";
  x.textAlign = "right";
  for (let i = 0; i <= 4; i++) {
    const v = lo + (hi - lo) * i / 4, yy = py(v);
    x.beginPath(); x.moveTo(L, yy); x.lineTo(W - R, yy); x.stroke();
    x.fillText(Math.round(v), L - 6, yy + 4);
  }
  x.textAlign = "center";
  for (let m = 0; m < 12; m++) x.fillText(MON[m], px(m), H - 14);
  const band = (pa, pb, col) => {
    x.fillStyle = col; x.beginPath(); let started = false;
    for (let m = 0; m < 12; m++) { const d = months[m]; if (!d) continue;
      const xx = px(m), yy = py(d.p[pa]);
      started ? x.lineTo(xx, yy) : x.moveTo(xx, yy); started = true; }
    for (let m = 11; m >= 0; m--) { const d = months[m]; if (!d) continue;
      x.lineTo(px(m), py(d.p[pb])); }
    x.closePath(); x.fill();
  };
  band(2, 6, "rgba(74,122,181,0.22)");     // p10-p90
  band(3, 5, "rgba(74,122,181,0.34)");     // p25-p75
  // median line
  x.strokeStyle = "#8fbaf0"; x.lineWidth = 2; x.beginPath(); let st = false;
  for (let m = 0; m < 12; m++) { const d = months[m]; if (!d) continue;
    const xx = px(m), yy = py(d.p[4]); st ? x.lineTo(xx, yy) : x.moveTo(xx, yy); st = true; }
  x.stroke();
  // record markers + years
  x.font = "9px Inter";
  for (let m = 0; m < 12; m++) { const d = months[m]; if (!d) continue;
    x.fillStyle = "#e0603a"; x.fillText("▲", px(m), py(d.max) - 3);
    x.fillStyle = "#4a7ab5"; x.fillText("▼", px(m), py(d.min) + 10);
    x.fillStyle = "#6a6a86"; x.fillText(String(d.maxY).slice(2), px(m), py(d.max) - 12); }
  // current sounding value
  const cur = climoNow[key];
  if (isFinite(cur) && lastMonth) {
    const m = parseInt(lastMonth, 10) - 1;
    x.fillStyle = "#ffd60a"; x.beginPath(); x.arc(px(m), py(cur), 6, 0, 7); x.fill();
    x.strokeStyle = "#000"; x.lineWidth = 1; x.stroke();
  }
  document.getElementById("climo-sub").textContent =
    `${meta[1]} (${meta[2]}) · record ${climo.months["01"] ? "" : ""}` +
    `${Object.values(climo.months)[0].yr0}–${Object.values(climo.months)[0].yr1}`;
}
document.getElementById("climo-btn").addEventListener("click", openClimo);
document.getElementById("climo-var").addEventListener("change", e => drawClimo(e.target.value));
document.getElementById("climo-close").addEventListener("click", () => climoModal.hidden = true);
climoModal.addEventListener("click", e => { if (e.target === climoModal) climoModal.hidden = true; });
document.addEventListener("keydown", e => {
  if (e.key === "Escape" && !climoModal.hidden) climoModal.hidden = true;
});
const CLIMO_PCTS = [1, 5, 10, 25, 50, 75, 90, 95, 99];
function climoPct(key, v) {                          // -> {pct, rec} or null
  if (!climo || !isFinite(v) || lastMonth === null) return null;
  const d = climo.months && climo.months[lastMonth] && climo.months[lastMonth][key];
  if (!d || !d.p) return null;
  const X = [d.min, ...d.p, d.max], Y = [0, ...CLIMO_PCTS, 100];
  let pct = v <= X[0] ? 0 : v >= X[X.length - 1] ? 100 : 50;
  for (let i = 1; i < X.length; i++) {
    if (v <= X[i]) { const f = (v - X[i - 1]) / ((X[i] - X[i - 1]) || 1);
      pct = Y[i - 1] + f * (Y[i] - Y[i - 1]); break; }
  }
  const rec = v >= d.max ? { t: "high", y: d.maxY } : v <= d.min ? { t: "low", y: d.minY } : null;
  return { pct: Math.max(0, Math.min(100, pct)), rec };
}
// below p50 = a graded blue (light near median -> deep near record low);
// above p50 = graded red. Visible even for small departures from the median.
function pctColor(pct) {
  const x = (pct - 50) / 50;                          // -1 .. +1
  const a = 0.16 + 0.54 * Math.min(1, Math.abs(x));   // clear tint even near p50
  const c = x >= 0 ? [224, 70, 55] : [56, 120, 216];  // red high / blue low
  return `rgba(${c[0]},${c[1]},${c[2]},${a.toFixed(2)})`;
}
// value cell HTML: colored background by percentile, ★ + year at the tails
function climoCell(key, v, txt) {
  const r = climoPct(key, v);
  if (!r) return txt;
  const ord = Math.max(1, Math.min(99, Math.round(r.pct)));
  const tip = r.rec ? `record ${r.rec.t} ${r.rec.y}` : `${ord}th percentile`;
  const star = r.rec ? ` <span style="color:${r.rec.t === "high" ? "#ff5a3c" : "#5a9bf0"}">★${String(r.rec.y).slice(2)}</span>` : "";
  const sub = r.rec ? "" : `<span class="pctlab">P${ord}</span>`;
  return `<span class="pctcell" style="background:${pctColor(r.pct)}" title="${tip}">${txt}${star}</span>${sub}`;
}
const MIRROR = "https://raw.githubusercontent.com/scorvec/scorvec.github.io/skewt-data/";
const IGRA = "https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/access/";
const UW_ARCHIVE = "https://raw.githubusercontent.com/scorvec/scorvec.github.io/skewt-archive/";
const UW_ARCHIVE_START = "2026-07-10";      // day bundles exist from here on
const dayZipCache = new Map();              // YYYYMMDD -> {filename: Uint8Array} | null
let M = null;                    // wasm module
let entries = {};                // mirror manifest: id -> {n, la, lo, dt, src}
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
const wasmReady = createSharp().then(mod => { M = mod; });

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
  drawSkewT(lastProf, lastRes); drawHodo(lastProf, lastRes);
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
]).then(([stns, man]) => {
  entries = (man && man.entries) || {};
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
    const m = L.circleMarker([s.la, s.lo], {
      radius: RAD.live, weight: 1.5, color: "#1d3a5e", fillColor: "#4a7ab5", fillOpacity: 0.95,
    }).addTo(map);
    const arch = ig ? ` · archive ${ig.y0}–${ig.y1}` : "";
    m.bindTooltip(`${s.n || id} (${id}) · latest ${s.dt}Z${arch}`);
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
    '<span style="color:#c0392b;font-size:1.05em">●</span> closed<br>' +
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
    if (!s) {
      clearPlot("station not in the UW live feed — switch to Archive (IGRA)");
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
      plotNote = ""; lastMonth = s.dt.slice(5, 7);
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
          plotNote = ""; lastMonth = wantDt.slice(5, 7);
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
          plotNote = ""; lastMonth = ymd.slice(5, 7);
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
  lastMonth = shown.slice(5, 7);
  render(thin(got.prof));
}

// ---------- compute ----------
function compute(prof) {
  const N = prof.P.length;
  const ptrs = ["P", "H", "T", "D", "U", "V"].map(k => f32(prof[k]));
  const out = M._malloc(48 * 4);
  const tr = [M._malloc(N * 4), M._malloc(N * 4), M._malloc(N * 4)];
  M._compute_sounding(...ptrs, N, out, tr[0], tr[1], tr[2]);
  const o = Array.from(M.HEAPF32.subarray(out / 4, out / 4 + 48));
  const traces = tr.map(p => Array.from(M.HEAPF32.subarray(p / 4, p / 4 + N)));
  [...ptrs, out, ...tr].forEach(p => M._free(p));
  return { o, sb: traces[0], ml: traces[1], mu: traces[2] };
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
const SK = { l: 52, r: 84, t: 36, b: 34, pBot: 105000, pTop: 10000, tL: -35, tR: 45 };

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
  ctx.fillStyle = TH.ink; ctx.font = "700 14px Inter";
  ctx.fillText(plotTitle, SK.l, 22);
  ctx.fillStyle = TH.muted; ctx.font = "10px Inter";
  ctx.fillText("SHARPlib · scorvec.com/skewt", W - 190, 22);
  if (plotNote) {
    ctx.fillStyle = "#ffd60a"; ctx.font = "700 12.5px Inter";
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
      ctx.fillStyle = TH.isotherm0; ctx.font = "600 10px Inter";
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
    ctx.fillStyle = TH.muted; ctx.font = "600 15px Inter"; ctx.textAlign = "center";
    ctx.fillText("pilot balloon — winds only (see hodograph →)", SK.l + pw / 2, SK.t + ph / 2);
    ctx.textAlign = "left";
  }

  // MU parcel levels, labeled with height AGL
  const o = res.o;
  ctx.font = "700 11px Inter";
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
  ctx.fillStyle = TH.muted; ctx.font = "11px Inter";
  for (let pp = 100; pp <= 1000; pp += 100) {
    const y = yOf(pp * 100);
    ctx.strokeStyle = TH.gridSub; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(SK.l, y); ctx.lineTo(SK.l + pw, y); ctx.stroke();
    ctx.fillText(pp, 14, y + 4);
  }
  // height ticks (km AGL) on the left inside
  ctx.font = "600 10px Inter"; ctx.fillStyle = "#a8a8c2";
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
  ctx.font = "11px Inter"; ctx.fillStyle = TH.muted;
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
  ctx.strokeStyle = "#2c2c48"; ctx.fillStyle = "#9d9dbb"; ctx.font = "12px Inter";
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
    ctx.font = "700 12.5px Inter";
    ctx.fillText(`${Math.round(dirOf(uMs, vMs))}/${Math.round(Math.hypot(u, v))} ${lab}`,
                 X(u) + 9, Y(v) + 4.5);
  };
  mark(o[22], o[23], "RM", "#ff453a", false);
  mark(o[24], o[25], "LM", "#64d2ff", true);
  // 0-6 km mean wind
  let mu6 = 0, mv6 = 0, c6 = 0;
  for (let i = 0; i < n && agl[i] <= 6000; i++) { mu6 += prof.U[i]; mv6 += prof.V[i]; c6++; }
  if (c6) mark(mu6 / c6, mv6 / c6, "MW", "#a8a8c2", true);

  // critical angle: sfc SR inflow vs 0-500 m shear vector (RM)
  if (o[22] !== MISSING) {
    const [u5, v5] = windAt(prof, 500);
    const shU = u5 - prof.U[0], shV = v5 - prof.V[0];
    const srU = o[22] - prof.U[0], srV = o[23] - prof.V[0];
    const dot = shU * srU + shV * srV;
    const mag = Math.hypot(shU, shV) * Math.hypot(srU, srV);
    if (mag > 0.1) {
      const ang = Math.acos(Math.max(-1, Math.min(1, dot / mag))) * 180 / Math.PI;
      ctx.fillStyle = "#ffd60a"; ctx.font = "700 14px Inter";
      ctx.fillText(`Critical angle = ${ang.toFixed(0)}°`, 12, H - 12);
    }
  }
  // height-band legend (labeled colors)
  ctx.font = "600 12.5px Inter";
  let lx = 12;
  ctx.fillStyle = "#9d9dbb"; ctx.fillText("km AGL:", lx, 22); lx += 62;
  for (const [b2, t2, col] of segs) {
    ctx.fillStyle = col;
    ctx.fillRect(lx, 13, 16, 5);
    ctx.fillText(`${b2 / 1000}–${t2 / 1000}`, lx + 20, 22);
    lx += 58;
  }
  ctx.fillStyle = "#9d9dbb"; ctx.font = "11.5px Inter";
  ctx.fillText("rings: knots", 12, 40);
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
  const rows = [["PCL", "CAPE", "CINH", "LCL", "LI", "LFC", "EL"]];
  const defs = [["SFC", 0, 36], ["ML", 5, 37], ["MU", 10, 38]];
  for (const [nm, i0, li] of defs) {
    rows.push([nm, fmt(o[i0]), fmt(o[i0 + 1]),
               hAt(o[i0 + 2]), fmt(o[li], 1), hAt(o[i0 + 3]), hAt(o[i0 + 4])]);
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
  const kinem = [
    ["0–1 km shear", ktf(o[20], o[21])], ["0–6 km shear", ktf(o[18], o[19])],
    ["SRH 0–1 km", fmt(o[26]) + " m²/s²"], ["SRH 0–3 km", fmt(o[27]) + " m²/s²"],
    ["Eff. SRH", fmt(o[30]) + " m²/s²"],
    ["Eff. shear (EBWD)", o[31] === MISSING ? "—" : (o[31] * KT).toFixed(0) + " kt"],
    ["Eff. inflow", effIL],
    ["Bunkers RM", vecf(o[22], o[23])], ["Bunkers LM", vecf(o[24], o[25])],
  ];
  const composites = [
    ["EHI 0–1 km", ehi(o[0], o[26])], ["EHI 0–3 km", ehi(o[0], o[27])],
    ["SCP", fmt(o[32], 1)], ["STP (eff.)", fmt(o[33], 1)],
    ["SHIP", climoCell("ship", o[43], fmt(o[43], 1))],
  ];
  const thermo = [
    ["ECAPE (MU)", climoCell("ecape", o[42], o[42] === MISSING ? "—" : fmt(o[42]) + " J/kg")],
    ["DCAPE", fmt(o[39]) + " J/kg"],
    ["0–3 km CAPE", fmt(o[40]) + " J/kg"], ["NCAPE", fmt(o[41], 2)],
    ["PWAT", climoCell("pwat", o[15], fmt(o[15], 1) + " mm")],
    ["Lapse 0–3 km", fmt(o[16], 1) + " K/km"], ["Lapse 3–6 km", fmt(o[17], 1) + " K/km"],
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
  ];

  climoNow = { pwat: o[15], "850t": t850, "700t": t700, "500t": t500,
    h500: h500, thick: (isFinite(h500) && isFinite(h1000)) ? h500 - h1000 : NaN,
    fzl: fzl, kidx: kidx, tott: totalT,
    ecape: o[42] === MISSING ? NaN : o[42], ship: o[43] === MISSING ? NaN : o[43],
    wbz: wbzAgl(prof) };
  document.getElementById("climo-btn").style.display = climo ? "" : "none";

  const row = ([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`;
  document.getElementById("kin-table").innerHTML = kinem.map(row).join("");
  document.getElementById("kin-table-b").innerHTML = composites.map(row).join("");
  document.getElementById("kin-table2").innerHTML = thermo.map(row).join("");
  document.getElementById("kin-table3").innerHTML = levels.map(row).join("");
}

// ---------- render ----------
function clearPlot(msg) {
  setStatus(msg || "no data");
  for (const id of ["skewt", "hodo"]) {
    const cv = document.getElementById(id), ctx = cv.getContext("2d");
    ctx.fillStyle = "#0a0a14"; ctx.fillRect(0, 0, cv.width, cv.height);
    ctx.fillStyle = "#8b8ba3"; ctx.font = "600 16px Inter"; ctx.textAlign = "center";
    ctx.fillText(msg || "no data", cv.width / 2, cv.height / 2);
    ctx.textAlign = "left";
  }
  for (const id of ["pcl-table", "kin-table", "kin-table-b", "kin-table2", "kin-table3"])
    document.getElementById(id).innerHTML = "";
  plotTitle = "";
}

function render(prof) {
  const hasT = prof.T.some(v => isFinite(v));
  const res = hasT ? compute(prof) : { o: new Array(48).fill(MISSING),
    sb: [], ml: [], mu: [] };
  lastProf = prof; lastRes = res;
  drawSkewT(prof, res);
  drawHodo(prof, res);
  if (hasT) fillTables(prof, res);
  else for (const id of ["pcl-table", "kin-table", "kin-table-b", "kin-table2", "kin-table3"])
    document.getElementById(id).innerHTML = "";
}
