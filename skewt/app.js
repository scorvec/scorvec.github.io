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

const stationsReady = fetch("stations.json").then(r => r.json()).then(d => {
  for (const s of d.stations) {
    igraStations[s.gid] = s;
    if (s.id) byWmo[s.id] = s.gid;
    const m = L.circleMarker([s.la, s.lo], {
      radius: 3, weight: 1, color: "#9a978f", fillColor: "#c5c2ba",
      fillOpacity: 0.7,
    }).addTo(map);
    m.bindTooltip(`${s.n} (${s.gid}) · archive ${s.y0}–${s.y1}`);
    m.on("click", () => { setMode("archive"); selectStation(s); });
  }
});

stationsReady.then(() =>
  fetch(MIRROR + "manifest.json?t=" + Date.now()).then(r => r.json())
).then(d => {
  entries = d.entries || {};
  for (const [id, s] of Object.entries(entries)) {
    const m = L.circleMarker([s.la, s.lo], {
      radius: 4.5, weight: 1, color: "#2c4a72", fillColor: "#4a7ab5", fillOpacity: 0.85,
    }).addTo(map);
    m.bindTooltip(`${s.n || id} (${id}) · latest ${s.dt}Z`);
    m.on("click", () => {
      setMode("latest");
      selectStation({ gid: byWmo[id], id, n: s.n, e: (igraStations[byWmo[id]] || {}).e || 0 });
    });
  }
  const want = location.hash.replace("#", "");
  const id0 = entries[want] ? want : (entries["72520"] ? "72520" : Object.keys(entries)[0]);
  if (id0) selectStation({ gid: byWmo[id0], id: id0, n: (entries[id0] || {}).n,
                           e: (igraStations[byWmo[id0]] || {}).e || 0 });
}).catch(() => {
  document.getElementById("status").textContent =
    "live mirror unavailable — archive mode still works";
});

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
  const re = new RegExp(`^#${gid} ${Y} ${Mo} ${D} (\\d{2})`, "gm");
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
      render(thin(prof));
    } catch (e) {
      status.textContent = "sounding unavailable for this station";
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
  render(thin(got.prof));
}

// ---------- compute ----------
function compute(prof) {
  const N = prof.P.length;
  const ptrs = ["P", "H", "T", "D", "U", "V"].map(k => f32(prof[k]));
  const out = M._malloc(40 * 4);
  const tr = [M._malloc(N * 4), M._malloc(N * 4), M._malloc(N * 4)];
  M._compute_sounding(...ptrs, N, out, tr[0], tr[1], tr[2]);
  const o = Array.from(M.HEAPF32.subarray(out / 4, out / 4 + 40));
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

// ---------- skew-t drawing ----------
const SK = { l: 46, r: 76, t: 14, b: 30, pBot: 105000, pTop: 10000, tL: -35, tR: 45 };

function drawSkewT(prof, res) {
  const cv = document.getElementById("skewt"), ctx = cv.getContext("2d");
  const W = cv.width, H = cv.height;
  const pw = W - SK.l - SK.r, ph = H - SK.t - SK.b;
  const yOf = p => SK.t + (1 - Math.log(SK.pBot / p) / Math.log(SK.pBot / SK.pTop)) * ph;
  const xOf = (tC, y) => SK.l + ((tC - SK.tL) / (SK.tR - SK.tL)) * pw + ((SK.t + ph) - y);
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, W, H);

  ctx.save();
  ctx.beginPath(); ctx.rect(SK.l, SK.t, pw, ph); ctx.clip();

  // isotherms
  for (let T = -120; T <= 50; T += 10) {
    ctx.strokeStyle = T === 0 ? "#9db8d6" : "#eee4d6"; ctx.lineWidth = T === 0 ? 1.4 : 1;
    ctx.beginPath();
    ctx.moveTo(xOf(T, SK.t + ph), SK.t + ph); ctx.lineTo(xOf(T, SK.t), SK.t); ctx.stroke();
  }
  // dry adiabats
  const pGrid = [];
  for (let lp = Math.log(105000); lp >= Math.log(10000); lp -= 0.03) pGrid.push(Math.exp(lp));
  ctx.strokeStyle = "#e7cfb4"; ctx.lineWidth = 1;
  for (let th = 230; th <= 440; th += 10) {
    ctx.beginPath();
    pGrid.forEach((p, i) => {
      const T = th * Math.pow(p / 100000, 0.2854) - 273.15;
      const x = xOf(T, yOf(p));
      i ? ctx.lineTo(x, yOf(p)) : ctx.moveTo(x, yOf(p));
    });
    ctx.stroke();
  }
  // moist adiabats (SHARPlib lifter — same physics as the parcels)
  ctx.strokeStyle = "#bcd8c2"; ctx.lineWidth = 1;
  const pG32 = new Float32Array(pGrid);
  for (let Ts = -24; Ts <= 40; Ts += 8) {
    const tk = traceAdiabat(105000, Ts + 273.15, Ts + 273.15, pG32);
    ctx.beginPath();
    pGrid.forEach((p, i) => {
      const x = xOf(tk[i] - 273.15, yOf(p));
      i ? ctx.lineTo(x, yOf(p)) : ctx.moveTo(x, yOf(p));
    });
    ctx.stroke();
  }
  // mixing ratio lines
  ctx.strokeStyle = "#9fbf9f"; ctx.setLineDash([2, 4]); ctx.lineWidth = 1;
  for (const w of [1, 2, 3, 5, 8, 12, 20]) {
    ctx.beginPath();
    let started = false;
    for (const p of pGrid) {
      if (p < 55000) break;
      const e = (w * (p / 100)) / (622 + w);
      const Td = (243.5 * Math.log(e / 6.112)) / (17.67 - Math.log(e / 6.112));
      const x = xOf(Td, yOf(p));
      started ? ctx.lineTo(x, yOf(p)) : ctx.moveTo(x, yOf(p));
      started = true;
    }
    ctx.stroke();
  }
  ctx.setLineDash([]);

  // parcel traces (virtual temperature): MU bold, SB secondary
  const drawTrace = (tr, style, dash) => {
    ctx.strokeStyle = style; ctx.lineWidth = 1.8; ctx.setLineDash(dash);
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < prof.P.length; i++) {
      if (!isFinite(tr[i]) || tr[i] < 100) { continue; }
      const x = xOf(tr[i] - 273.15, yOf(prof.P[i]));
      started ? ctx.lineTo(x, yOf(prof.P[i])) : ctx.moveTo(x, yOf(prof.P[i]));
      started = true;
    }
    ctx.stroke(); ctx.setLineDash([]);
  };
  drawTrace(res.sb, "#e67e22", [7, 4]);
  drawTrace(res.mu, "#8e44ad", [3, 3]);

  // environment profiles
  const drawProf = (vals, color) => {
    ctx.strokeStyle = color; ctx.lineWidth = 2.4;
    ctx.beginPath();
    for (let i = 0; i < prof.P.length; i++) {
      const x = xOf(vals[i] - 273.15, yOf(prof.P[i]));
      i ? ctx.lineTo(x, yOf(prof.P[i])) : ctx.moveTo(x, yOf(prof.P[i]));
    }
    ctx.stroke();
  };
  drawProf(prof.D, "#1e8449");
  drawProf(prof.T, "#c0392b");

  // MU parcel level markers
  const o = res.o;
  const marks = [["LCL", o[12]], ["LFC", o[13]], ["EL", o[14]]];
  ctx.font = "600 11px Inter, sans-serif"; ctx.fillStyle = "#8e44ad";
  for (const [lab, p] of marks) {
    if (p === MISSING || !isFinite(p)) continue;
    const y = yOf(p);
    ctx.fillRect(SK.l + pw - 46, y - 0.75, 18, 1.5);
    ctx.fillText(lab, SK.l + pw - 26, y + 3.5);
  }
  // effective inflow layer bracket
  if (o[28] !== MISSING && o[29] !== MISSING) {
    const y0 = yOf(o[28]), y1 = yOf(o[29]);
    ctx.strokeStyle = "#2980b9"; ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(SK.l + 12, y0); ctx.lineTo(SK.l + 4, y0);
    ctx.lineTo(SK.l + 4, y1); ctx.lineTo(SK.l + 12, y1); ctx.stroke();
    ctx.fillStyle = "#2980b9"; ctx.fillText("EIL", SK.l + 6, (y0 + y1) / 2 + 3);
  }
  ctx.restore();

  // axes
  ctx.fillStyle = "#888580"; ctx.font = "11px Inter, sans-serif";
  ctx.strokeStyle = "#d8d4cb"; ctx.lineWidth = 1;
  for (let p = 100; p <= 1000; p += 100) {
    const y = yOf(p * 100);
    ctx.beginPath(); ctx.moveTo(SK.l, y); ctx.lineTo(SK.l + pw, y); ctx.stroke();
    ctx.fillText(p, 10, y + 4);
  }
  for (let T = -30; T <= 40; T += 10) {
    ctx.fillText(T, xOf(T, SK.t + ph) - 8, SK.t + ph + 18);
  }
  ctx.fillText("°C", SK.l + pw / 2, H - 6);

  // wind barbs
  drawBarbs(ctx, prof, W - SK.r + 34, yOf);
}

function drawBarbs(ctx, prof, x0, yOf) {
  let lastY = -1e9;
  for (let i = 0; i < prof.P.length; i++) {
    const y = yOf(prof.P[i]);
    if (Math.abs(y - lastY) < 22) continue;
    lastY = y;
    const u = prof.U[i], v = prof.V[i];
    const kt = Math.hypot(u, v) * 1.9438;
    const ang = Math.atan2(-u, -v);                        // direction FROM
    ctx.save();
    ctx.translate(x0, y); ctx.rotate(ang);
    ctx.strokeStyle = "#1c1c1a"; ctx.lineWidth = 1.1; ctx.fillStyle = "#1c1c1a";
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
  const W = cv.width, H = cv.height, cx = W / 2, cy = H / 2;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, W, H);
  const agl = prof.H.map(h => h - prof.H[0]);
  const iMax = agl.findIndex(h => h > 12000);
  const n = iMax < 0 ? prof.P.length : iMax;
  let vmax = 10;
  for (let i = 0; i < n; i++) vmax = Math.max(vmax, Math.hypot(prof.U[i], prof.V[i]));
  const scale = (Math.min(W, H) / 2 - 18) / vmax;
  ctx.strokeStyle = "#eee4d6"; ctx.fillStyle = "#888580"; ctx.font = "10px Inter";
  for (let r = 10; r <= vmax + 10; r += 10) {
    ctx.beginPath(); ctx.arc(cx, cy, r * scale, 0, 7); ctx.stroke();
    ctx.fillText(r, cx + r * scale - 8, cy - 3);
  }
  const segs = [[0, 1000, "#c0392b"], [1000, 3000, "#e67e22"],
                [3000, 6000, "#1e8449"], [6000, 9000, "#b7950b"], [9000, 12000, "#7d3c98"]];
  for (const [b, t, col] of segs) {
    ctx.strokeStyle = col; ctx.lineWidth = 2.2;
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < n; i++) {
      if (agl[i] < b || agl[i] > t) { if (agl[i] > t) break; continue; }
      const x = cx + prof.U[i] * scale, y = cy - prof.V[i] * scale;
      started ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      started = true;
    }
    ctx.stroke();
  }
  const o = res.o;
  const mark = (u, v, lab, col) => {
    if (u === MISSING) return;
    ctx.fillStyle = col;
    ctx.beginPath(); ctx.arc(cx + u * scale, cy - v * scale, 4, 0, 7); ctx.fill();
    ctx.font = "600 10px Inter"; ctx.fillText(lab, cx + u * scale + 6, cy - v * scale + 3);
  };
  mark(o[22], o[23], "RM", "#c0392b");
  mark(o[24], o[25], "LM", "#2980b9");
  ctx.fillStyle = "#888580"; ctx.font = "10px Inter";
  ctx.fillText("m/s · 0–1–3–6–9–12 km", 8, H - 8);
}

// ---------- tables ----------
const fmt = (v, d = 0, unit = "") =>
  (v === MISSING || !isFinite(v)) ? "—" : v.toFixed(d) + unit;

function fillTables(res) {
  const o = res.o;
  const pcl = (i0) => [o[i0], o[i0 + 1], o[i0 + 2] / 100, o[i0 + 3] / 100, o[i0 + 4] / 100];
  const rows = [["", "SB", "ML", "MU"],
    ["CAPE J/kg", ...[0, 5, 10].map(i => fmt(o[i]))],
    ["CIN J/kg", ...[1, 6, 11].map(i => fmt(o[i]))],
    ["LCL hPa", ...[2, 7, 12].map(i => fmt(o[i] === MISSING ? MISSING : o[i] / 100))],
    ["LFC hPa", ...[3, 8, 13].map(i => fmt(o[i] === MISSING ? MISSING : o[i] / 100))],
    ["EL hPa", ...[4, 9, 14].map(i => fmt(o[i] === MISSING ? MISSING : o[i] / 100))]];
  document.getElementById("pcl-table").innerHTML = rows.map((r, i) =>
    `<tr>${r.map((c, j) => i === 0 ? `<th>${c}</th>` : `<td>${c}</td>`).join("")}</tr>`).join("");

  const ktf = (u, v) => (u === MISSING) ? "—" : (Math.hypot(u, v) * 1.9438).toFixed(0) + " kt";
  const dirf = (u, v) => (u === MISSING) ? "" :
    ((Math.atan2(-u, -v) * 180 / Math.PI + 360) % 360).toFixed(0) + "°/";
  const kin = [
    ["0–1 km shear", ktf(o[20], o[21])],
    ["0–6 km shear", ktf(o[18], o[19])],
    ["SRH 0–1 km", fmt(o[26], 0, " m²/s²")],
    ["SRH 0–3 km", fmt(o[27], 0, " m²/s²")],
    ["Effective SRH", fmt(o[30], 0, " m²/s²")],
    ["Effective shear", o[31] === MISSING ? "—" : (o[31] * 1.9438).toFixed(0) + " kt"],
    ["Bunkers RM", dirf(o[22], o[23]) + ktf(o[22], o[23])],
    ["PWAT", fmt(o[15], 1, " mm")],
    ["Lapse 0–3 km", fmt(o[16], 1, " K/km")],
    ["Lapse 3–6 km", fmt(o[17], 1, " K/km")],
    ["SCP", fmt(o[32], 1)],
    ["STP", fmt(o[33], 1)],
  ];
  document.getElementById("kin-table").innerHTML =
    kin.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
}

// ---------- render ----------
function render(prof) {
  const res = compute(prof);
  drawSkewT(prof, res);
  drawHodo(prof, res);
  fillTables(res);
}
