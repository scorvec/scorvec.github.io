/* Skew-T Explorer — all client-side.
   Data: U. Wyoming sounding archive, mirrored to the skewt-data branch by
   .github/workflows/skewt-data.yml (UW sends no CORS headers; raw.github
   serves ACAO *). Physics: SHARPlib via WebAssembly (sharplib.js/.wasm). */
"use strict";

const MISSING = -9999.0;
const MIRROR = "https://raw.githubusercontent.com/scorvec/scorvec.github.io/skewt-data/";
const IGRA = "https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/access/";
let M = null;                    // wasm module
let entries = {};                // mirror manifest: id -> {n, la, lo, dt, src}
let igraStations = {};           // gid -> station meta (all 2,921 incl. closed)
let byWmo = {};                  // wmo id -> gid
let current = null;              // selected: {gid, id, n, e}
let plotTitle = "";              // drawn on the skew-t canvas itself
let selectedMarker = null;       // highlighted dot on the map
let mode = "latest";
let archHour = 12;
const igraCache = new Map();     // gid -> decompressed text

// ---------- wasm ----------
const wasmReady = createSharp().then(mod => { M = mod; });

function f32(arr) {
  const p = M._malloc(arr.length * 4);
  M.HEAPF32.set(arr instanceof Float32Array ? arr : new Float32Array(arr), p / 4);
  return p;
}

// ---------- station map ----------
const map = L.map("map", { worldCopyJump: true }).setView([25, 0], 2);
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
      radius: 2.5, weight: 0.5, color: "#b3b0a8", fillColor: "#d4d1c9", fillOpacity: 0.6,
    }).addTo(closedLayer);
    m.bindTooltip(`${s.n} (${s.gid}) · closed ${s.y0}–${s.y1} — click for archive`);
    m.on("click", () => { setMode("archive"); highlight(m); selectStation(s); });
  }
  // active stations without a launch in the mirror window: small grey
  for (const s of stns.stations) {
    if (s.y1 < ACTIVE_YEAR || (s.id && entries[s.id])) continue;
    const m = L.circleMarker([s.la, s.lo], {
      radius: 3.5, weight: 1, color: "#8a877f", fillColor: "#b8b5ad", fillOpacity: 0.8,
    }).addTo(map);
    m.bindTooltip(`${s.n} (${s.gid}) · no launch in last 36 h — click for archive (${s.y0}–${s.y1})`);
    m.on("click", () => { setMode("archive"); highlight(m); selectStation(s); });
  }
  // stations with a sounding in the last 36 h: big blue, on top
  for (const [id, s] of Object.entries(entries)) {
    const ig = igraStations[byWmo[id]];
    if (ig || /dtype|Name:/i.test(s.n || "")) s.n = (ig && ig.n) || id;
    const m = L.circleMarker([s.la, s.lo], {
      radius: 6, weight: 1.5, color: "#1d3a5e", fillColor: "#4a7ab5", fillOpacity: 0.95,
    }).addTo(map);
    m.bindTooltip(`${s.n || id} (${id}) · click for latest (${s.dt}Z)`);
    m.on("click", () => {
      setMode("latest");
      highlight(m);
      selectStation({ gid: byWmo[id], id, n: s.n, e: (igraStations[byWmo[id]] || {}).e || 0 });
    });
  }
  if (!man) document.getElementById("status").textContent =
    "live mirror unavailable — archive mode still works";
  const want = location.hash.replace("#", "");
  const id0 = entries[want] ? want : (entries["72520"] ? "72520" : Object.keys(entries)[0]);
  if (id0) selectStation({ gid: byWmo[id0], id: id0, n: (entries[id0] || {}).n,
                           e: (igraStations[byWmo[id0]] || {}).e || 0 });
});

// legend + closed-station toggle (Leaflet control)
const legend = L.control({ position: "topright" });
legend.onAdd = () => {
  const div = L.DomUtil.create("div");
  div.style.cssText = "background:rgba(255,255,255,0.95);padding:8px 10px;border-radius:8px;" +
    "border:1px solid #d8d4cb;font:11px Inter,sans-serif;line-height:1.7";
  div.innerHTML =
    '<span style="color:#4a7ab5">●</span> sounding in last 36 h&nbsp;&nbsp;' +
    '<span style="color:#b8b5ad">●</span> active (archive)<br>' +
    '<label><input type="checkbox" id="show-closed"> show closed stations</label>';
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

function setMode(m2) {
  mode = m2;
  document.getElementById("mode-latest").classList.toggle("on", mode === "latest");
  document.getElementById("mode-archive").classList.toggle("on", mode === "archive");
  document.getElementById("arch-controls").style.display =
    mode === "archive" ? "inline" : "none";
}
document.getElementById("mode-latest").onclick = () => { setMode("latest"); loadSounding(); };
document.getElementById("mode-archive").onclick = () => { setMode("archive"); loadSounding(); };
document.getElementById("h00").onclick = () => { archHour = 0; hourBtns(); loadSounding(); };
document.getElementById("h12").onclick = () => { archHour = 12; hourBtns(); loadSounding(); };
function hourBtns() {
  document.getElementById("h00").classList.toggle("on", archHour === 0);
  document.getElementById("h12").classList.toggle("on", archHour === 12);
}
document.getElementById("date").value = new Date(Date.now() - 3 * 864e5)
  .toISOString().slice(0, 10);
document.getElementById("date").onchange = () => loadSounding();

function selectStation(s) {
  current = s;
  if (s.id) location.hash = s.id;
  document.getElementById("stn-label").textContent = `${s.n || s.gid} · ${s.id || s.gid}`;
  document.getElementById("status").textContent = mode === "latest"
    ? "fetching latest sounding from the mirror…"
    : "fetching from the NOAA IGRA archive (first load per station can be tens of MB)…";
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
  const key = gid + (year >= new Date().getUTCFullYear() - 1 ? ":y2d" : ":por");
  if (igraCache.has(key)) return igraCache.get(key);
  const status = document.getElementById("status");
  const urls = [];
  if (year >= new Date().getUTCFullYear() - 1) {
    urls.push(IGRA + `data-y2d/${gid}-data-beg${new Date().getUTCFullYear() - 1}.txt.zip`);
  }
  urls.push(IGRA + `data-por/${gid}-data.txt.zip`);
  for (const url of urls) {
    try {
      const r = await fetch(url);
      if (!r.ok) continue;
      const mb = (+r.headers.get("content-length") / 1048576).toFixed(1);
      status.textContent = `downloading IGRA archive (${mb} MB)…`;
      const buf = new Uint8Array(await r.arrayBuffer());
      status.textContent = "decompressing…";
      const files = fflate.unzipSync(buf);
      const name = Object.keys(files)[0];
      const text = fflate.strFromU8(files[name]);
      igraCache.set(key, text);
      return text;
    } catch (e) { /* try next */ }
  }
  return null;
}

const iv = (line, a, b) => {
  const v = parseInt(line.slice(a, b), 10);
  return (v === -9999 || v === -8888 || isNaN(v)) ? null : v;
};

function parseIGRA(text, gid, ymd, wantHour, elev) {
  // find headers for the date; pick the closest hour to wantHour
  const [Y, Mo, D] = ymd.split("-");
  const re = new RegExp("^#" + gid + " " + Y + " " + Mo + " " + D + " ([0-9]{2})", "gm");
  let best = null, m;
  while ((m = re.exec(text)) !== null) {
    const hh = +m[1] === 99 ? 12 : +m[1];
    if (best === null || Math.abs(hh - wantHour) < Math.abs(best.hh - wantHour)) {
      best = { idx: m.index, hh };
    }
  }
  if (!best) return null;
  const lines = text.slice(best.idx).split("\n");
  const nlev = parseInt(lines[0].slice(32, 36), 10);
  const out = { P: [], H: [], T: [], D: [], U: [], V: [] };
  let lastP = 1e9, lastH = elev, lastT = null;
  for (let i = 1; i <= nlev && i < lines.length; i++) {
    const L = lines[i];
    const p = iv(L, 9, 15);                                  // Pa
    const gph = iv(L, 16, 21);                               // m
    const tt = iv(L, 22, 27);                                // tenths C
    const dpdp = iv(L, 34, 39);                              // tenths C
    const wd = iv(L, 40, 45), ws = iv(L, 46, 51);            // deg, tenths m/s
    if (p === null || tt === null || dpdp === null || p >= lastP || p < 2000) continue;
    const T = tt / 10 + 273.15;
    let H;
    if (gph !== null) H = gph;
    else {                                                   // hypsometric fill
      const Tbar = lastT === null ? T : (T + lastT) / 2;
      H = lastH + (287.05 * Tbar / 9.80665) * Math.log(lastP / p);
    }
    if (out.P.length === 0 && gph === null) H = elev;
    lastP = p; lastH = H; lastT = T;
    out.P.push(p); out.H.push(H); out.T.push(T);
    out.D.push(T - dpdp / 10);
    const ok = wd !== null && ws !== null;
    out.U.push(ok ? -(ws / 10) * Math.sin(wd * Math.PI / 180) : (out.U.at(-1) ?? 0));
    out.V.push(ok ? -(ws / 10) * Math.cos(wd * Math.PI / 180) : (out.V.at(-1) ?? 0));
  }
  return out.P.length >= 8 ? { prof: out, hh: best.hh } : null;
}

async function loadSounding() {
  if (!current) return;
  const status = document.getElementById("status");
  await wasmReady;
  if (mode === "latest") {
    const s = entries[current.id];
    if (!s) { status.textContent = "no recent launch here — try Archive mode"; return; }
    status.textContent = "fetching…";
    try {
      const r = await fetch(MIRROR + "soundings/" + current.id + ".csv?t=" + s.dt);
      if (!r.ok) throw 0;
      const prof = parseCSV(await r.text());
      if (!prof) throw 0;
      status.textContent = `valid ${s.dt}Z · ${prof.P.length} levels (UW BUFR/GTS mirror)`;
      plotTitle = `${current.n || ""} ${current.id}  ·  ${s.dt}Z`.trim();
      render(thin(prof));
    } catch (e) {
      status.textContent = "error: " + (e && e.message ? e.message : "fetch failed");
    }
    return;
  }
  // archive mode (IGRA v2, straight from NOAA — CORS-open)
  if (!current.gid) { status.textContent = "station not in the IGRA archive"; return; }
  const ymd = document.getElementById("date").value;
  const text = await igraText(current.gid, +ymd.slice(0, 4));
  if (!text) { status.textContent = "IGRA file unavailable"; return; }
  const got = parseIGRA(text, current.gid, ymd, archHour, current.e || 0);
  if (!got) {
    status.textContent = `no sounding on ${ymd} — station record ` +
      `${(igraStations[current.gid] || {}).y0}–${(igraStations[current.gid] || {}).y1}`;
    return;
  }
  status.textContent = `valid ${ymd} ${String(got.hh).padStart(2, "0")}Z · ` +
    `${got.prof.P.length} levels (NOAA IGRA v2)`;
  plotTitle = `${current.n || ""} ${current.id || current.gid}  ·  ${ymd} ` +
    `${String(got.hh).padStart(2, "0")}Z`.trim();
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

function drawSkewT(prof, res) {
  const cv = document.getElementById("skewt"), ctx = cv.getContext("2d");
  const W = cv.width, H = cv.height;
  const pw = W - SK.l - SK.r, ph = H - SK.t - SK.b;
  const yOf = p => SK.t + (1 - Math.log(SK.pBot / p) / Math.log(SK.pBot / SK.pTop)) * ph;
  const xOf = (tC, y) => SK.l + ((tC - SK.tL) / (SK.tR - SK.tL)) * pw + ((SK.t + ph) - y);
  ctx.fillStyle = TH.bg; ctx.fillRect(0, 0, W, H);
  // on-canvas title: station + valid time
  ctx.fillStyle = TH.ink; ctx.font = "700 14px Inter";
  ctx.fillText(plotTitle, SK.l, 22);
  ctx.fillStyle = TH.muted; ctx.font = "10px Inter";
  ctx.fillText("SHARPlib · scorvec.com/skewt", W - 190, 22);

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
  drawProf(vtC, TH.vtmp, 1.1, [2, 3]);
  drawProf(prof.D.map(v => v - 273.15), TH.dwpt, 2.8);
  drawProf(prof.T.map(v => v - 273.15), TH.temp, 2.8);

  // MU parcel levels, labeled with height AGL
  const o = res.o;
  ctx.font = "700 11px Inter";
  const marks = [["LCL", o[12], TH.lcl], ["LFC", o[13], TH.lfc], ["EL", o[14], TH.el]];
  for (const [lab, pP, col] of marks) {
    if (pP === MISSING || !isFinite(pP)) continue;
    const y = yOf(pP), hm = interpHagl(prof, pP);
    ctx.strokeStyle = col; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(SK.l + pw - 58, y); ctx.lineTo(SK.l + pw - 34, y); ctx.stroke();
    ctx.fillStyle = col;
    ctx.fillText(`${lab} ${hm === null ? "" : Math.round(hm) + "m"}`, SK.l + pw - 30, y + 4);
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
  const cv = document.getElementById("hodo"), ctx = cv.getContext("2d");
  const W = cv.width, H = cv.height;
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
  ctx.strokeStyle = TH.gridSub; ctx.fillStyle = TH.muted; ctx.font = "9.5px Inter";
  const rMax = Math.ceil((span * 0.75) / 10) * 10 + 20;
  for (let r = 10; r <= rMax; r += 10) {
    ctx.beginPath(); ctx.arc(X(0), Y(0), r * scale, 0, 7); ctx.stroke();
    ctx.fillText(r, X(r) - 6, Y(0) + 11);
  }
  ctx.strokeStyle = TH.grid;
  ctx.beginPath(); ctx.moveTo(0, Y(0)); ctx.lineTo(W, Y(0)); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(X(0), 0); ctx.lineTo(X(0), H); ctx.stroke();

  // trace segments by height
  const segs = [[0, 1000, "#ff453a"], [1000, 3000, "#ff9f0a"],
                [3000, 6000, "#30d158"], [6000, 9000, "#ffd60a"], [9000, 12000, "#bf5af2"]];
  for (const [b, tt, col] of segs) {
    ctx.strokeStyle = col; ctx.lineWidth = 2.4;
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < n; i++) {
      if (agl[i] < b) continue;
      if (agl[i] > tt) break;
      const x = X(prof.U[i] * KT), y = Y(prof.V[i] * KT);
      started ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      started = true;
    }
    ctx.stroke();
  }

  // storm motion + mean wind markers, SHARPpy-style dir/spd labels
  const mark = (uMs, vMs, lab, col, hollow) => {
    if (uMs === MISSING || !isFinite(uMs)) return;
    const u = uMs * KT, v = vMs * KT;
    ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = 1.6;
    ctx.beginPath(); ctx.arc(X(u), Y(v), 4.5, 0, 7);
    hollow ? ctx.stroke() : ctx.fill();
    ctx.font = "700 10px Inter";
    ctx.fillText(`${Math.round(dirOf(uMs, vMs))}/${Math.round(Math.hypot(u, v))} ${lab}`,
                 X(u) + 7, Y(v) + 3.5);
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
      ctx.fillStyle = "#ffd60a"; ctx.font = "700 11px Inter";
      ctx.fillText(`Critical angle = ${ang.toFixed(0)}°`, 10, H - 24);
    }
  }
  ctx.fillStyle = TH.muted; ctx.font = "9.5px Inter";
  ctx.fillText("kt · 0–1–3–6–9–12 km AGL", 10, H - 8);
}

// ---------- tables ----------
const fmt = (v, d = 0, unit = "") =>
  (v === MISSING || v === undefined || !isFinite(v)) ? "—" : v.toFixed(d) + unit;

function fillTables(prof, res) {
  const o = res.o;
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
  const kin = [
    ["0–3 km CAPE", fmt(o[40]) + " J/kg"], ["NCAPE", fmt(o[41], 2)],
    ["DCAPE", fmt(o[39]) + " J/kg"], ["PWAT", fmt(o[15], 1) + " mm"],
    ["Lapse 0–3 km", fmt(o[16], 1) + " K/km"], ["Lapse 3–6 km", fmt(o[17], 1) + " K/km"],
    ["0–1 km shear", ktf(o[20], o[21])], ["0–6 km shear", ktf(o[18], o[19])],
    ["SRH 0–1 km", fmt(o[26]) + " m²/s²"], ["SRH 0–3 km", fmt(o[27]) + " m²/s²"],
    ["Eff. SRH", fmt(o[30]) + " m²/s²"],
    ["Eff. shear (EBWD)", o[31] === MISSING ? "—" : (o[31] * KT).toFixed(0) + " kt"],
    ["Bunkers RM", vecf(o[22], o[23])], ["Bunkers LM", vecf(o[24], o[25])],
    ["SCP", fmt(o[32], 1)], ["STP (eff.)", fmt(o[33], 1)],
  ];
  document.getElementById("kin-table").innerHTML =
    kin.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
}

// ---------- render ----------
function render(prof) {
  const res = compute(prof);
  drawSkewT(prof, res);
  drawHodo(prof, res);
  fillTables(prof, res);
}
