/* Hurricane reconnaissance plotter (recon.html).
   Data: the recon-data branch, written by .github/workflows/recon.yml (scripts/recon/recon_mirror.py) from the NHC
   reconnaissance archive, which sends no CORS header. manifest.json lists storms and missions; a mission's file
   (missions/<year>/<key>.json) is fetched only when it is opened; storms/<SID>.json = best track + forecast. */
"use strict";
(function () {
  // ── hosts ────────────────────────────────────────────────────────────────
  // Bunny CDN first (2026-10-07: its Edge Rule caches /recon-data/* and /tdr-data/* for 60 s; GitHub's raw host held
  // "fresh" fetches up to 5 min and is blocked on some networks); raw / api / jsDelivr behind it
  const HOSTS = ["cdn", "raw", "jsd", "api"];
  const REPO = "scorvec/scorvec.github.io", BR = "recon-data", TBR = "tdr-data";
  const URLS = {
    cdn: (p, b, br = BR) => [`https://frames.scorvec.com/${br}/${p}`],
    raw: (p, b, br = BR) => [`https://raw.githubusercontent.com/${REPO}/${br}/${p}${b ? "?t=" + encodeURIComponent(b) : ""}`],
    jsd: (p, b, br = BR) => [`https://cdn.jsdelivr.net/gh/${REPO}@${br}/${p}`],
    api: (p, b, br = BR) => [`https://api.github.com/repos/${REPO}/contents/${p}?ref=${br}`, { headers: { Accept: "application/vnd.github.raw" } }],
  };
  // fresh = a file that changes (manifest, an airborne mission): api (always fresh) before jsDelivr (cached ~12 h)
  async function dataFetch(path, bust, fresh, br) {
    let order = HOSTS.slice();
    if (fresh) order = order.filter(h => h !== "jsd").concat(order.includes("jsd") ? ["jsd"] : []);
    for (let i = 0; i < order.length; i++) {
      const [u, opt] = URLS[order[i]](path, bust, br);
      try {
        const r = await fetch(u, opt);
        if (r.ok) return await r.json();
        if (r.status === 404 && (order[i] === "raw" || order[i] === "cdn")) return null;   // host fine, file truly absent
      } catch (e) { /* blocked or offline: next host */ }
    }
    throw new Error("unreachable");
  }

  // ── helpers ──────────────────────────────────────────────────────────────
  const $ = id => document.getElementById(id);
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const p2 = n => String(n).padStart(2, "0");
  const MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const hm = t => { const d = new Date(t * 1000); return `${p2(d.getUTCHours())}:${p2(d.getUTCMinutes())}`; };
  const hms = t => { const d = new Date(t * 1000); return `${hm(t)}:${p2(d.getUTCSeconds())}`; };
  const dayhm = t => { const d = new Date(t * 1000), y = d.getUTCFullYear();
    return `${d.getUTCDate()} ${MON[d.getUTCMonth()]}${y !== new Date().getUTCFullYear() ? " " + y : ""} ${hm(t)}Z`; };
  const ll = (la, lo) => la == null ? "—" : `${Math.abs(la).toFixed(2)}°${la >= 0 ? "N" : "S"} ${Math.abs(lo).toFixed(2)}°${lo >= 0 ? "E" : "W"}`;
  const ago = t => {
    const m = Math.round((Date.now() / 1000 - t) / 60);
    if (m < 1) return "just now";
    if (m < 90) return `${m} min ago`;
    const h = m / 60;
    return h < 48 ? `${h.toFixed(h < 10 ? 1 : 0)} h ago` : `${Math.round(h / 24)} days ago`;
  };
  const esc = s => String(s == null ? "" : s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const dirTxt = (d, s) => s == null ? "—" : `${p2(Math.round(d)).padStart(3, "0")}° ${s} kt`;
  const AIR_MIN = 45;                                         // newest ob younger than this = airborne
  const isAir = m => m && m.last_hdob && (Date.now() / 1000 - m.last_hdob) < AIR_MIN * 60;

  // ── colour scales ────────────────────────────────────────────────────────
  const WIND = [[34, "#4a90d9", "< 34"], [50, "#2fa84f", "34–49"], [64, "#17703a", "50–63"], [83, "#e0b416", "64–82 (Cat 1)"],
                [96, "#f08c1a", "83–95 (Cat 2)"], [113, "#e0442c", "96–112 (Cat 3)"], [137, "#c41e6a", "113–136 (Cat 4)"],
                [1e9, "#7b2fbf", "137+ (Cat 5)"]];
  const SLP = [[940, "#5b1a8e", "< 940"], [960, "#8e2a9c", "940–959"], [980, "#c2378f", "960–979"], [990, "#e2546b", "980–989"],
               [1000, "#f08b3e", "990–999"], [1005, "#e9c13a", "1000–1004"], [1010, "#8cc56b", "1005–1009"], [1e9, "#4a9fd0", "1010+"]];
  const DD = [[1, "#1a7f4b", "< 1"], [2, "#5aa84a", "1–2"], [4, "#b5c642", "2–4"], [7, "#e3a33a", "4–7"], [12, "#c8642b", "7–12"],
              [1e9, "#8a3b1e", "12+"]];
  const TURBO = ["#30123b", "#4662d7", "#36aaf9", "#1ae4b6", "#72fe5e", "#c8ef34", "#faba39", "#f66b19", "#ca2a04", "#7a0403"];
  const VARS = {
    sfmr: { lab: "SFMR surface wind", unit: "kt", bins: WIND },
    ws: { lab: "Flight-level wind (30 s)", unit: "kt", bins: WIND },
    x: { lab: "Extrapolated surface pressure", unit: "hPa", bins: SLP },
    t_: { lab: "Flight-level temperature", unit: "°C", ramp: true },
    dd: { lab: "Dewpoint depression", unit: "°C", bins: DD },
  };
  const binCol = (bins, v) => { for (const b of bins) if (v < b[0]) return b[1]; return bins[bins.length - 1][1]; };
  const catOf = kt => kt == null ? "" : kt < 34 ? "TD" : kt < 64 ? "TS" : kt < 83 ? "Cat 1" : kt < 96 ? "Cat 2" : kt < 113 ? "Cat 3" : kt < 137 ? "Cat 4" : "Cat 5";

  // ── state ────────────────────────────────────────────────────────────────
  const S = { man: null, sid: null, key: null, mis: null, storm: null, v: "sfmr", hoverI: -1, rows: [], col: {}, dropItems: [],
              manAt: 0, misBust: null, fitKey: null };
  try { const v = localStorage.getItem("recon.var"); if (v && VARS[v]) S.v = v; } catch (e) { /* storage blocked */ }

  // ── map ──────────────────────────────────────────────────────────────────
  const dark = () => document.documentElement.dataset.theme === "dark" ||
    (document.documentElement.dataset.theme !== "light" && matchMedia("(prefers-color-scheme: dark)").matches);
  const map = L.map("map", { worldCopyJump: true, preferCanvas: true, zoomSnap: 0.5 }).setView([24, -80], 4);
  const ESRI = "https://services.arcgisonline.com/arcgis/rest/services/Canvas/";
  const base = dark() ? "World_Dark_Gray" : "World_Light_Gray";
  L.tileLayer(ESRI + base + "_Base/MapServer/tile/{z}/{y}/{x}",
    { attribution: "Tiles &copy; Esri &mdash; Esri, HERE, Garmin, &copy; OpenStreetMap contributors", maxZoom: 13 }).addTo(map);
  map.createPane("sat"); map.getPane("sat").style.zIndex = 300; map.getPane("sat").style.pointerEvents = "none";
  map.createPane("labels"); map.getPane("labels").style.zIndex = 350; map.getPane("labels").style.pointerEvents = "none";
  L.tileLayer(ESRI + base + "_Reference/MapServer/tile/{z}/{y}/{x}", { maxZoom: 13, pane: "labels", opacity: 0.9 }).addTo(map);
  const rend = L.canvas({ padding: 0.3, tolerance: 6 });
  const lyr = { btk: L.layerGroup().addTo(map), fixes: L.layerGroup().addTo(map), track: L.layerGroup().addTo(map),
                barbs: L.layerGroup().addTo(map), drops: L.layerGroup().addTo(map), vdm: L.layerGroup().addTo(map),
                hover: L.layerGroup().addTo(map), tdr: L.layerGroup().addTo(map), plane: L.layerGroup().addTo(map) };
  if (window.ResizeObserver) new ResizeObserver(() => map.invalidateSize()).observe($("map"));

  // ── load ─────────────────────────────────────────────────────────────────
  async function loadManifest(first) {
    let man;
    try { man = await dataFetch("manifest.json", Math.floor(Date.now() / 60000), true); }
    catch (e) {
      if (first) setStatus(`<span class="mu">The data mirror could not be reached (tried frames.scorvec.com, raw.githubusercontent.com, jsDelivr and api.github.com). Try again shortly.</span>`);
      return false;
    }
    if (!man) { if (first) setStatus(`<span class="mu">The mirror has not published yet.</span>`); return false; }
    const prevMis = S.man && S.key && S.man.missions[S.key];
    S.man = man; S.manAt = Date.now();
    fillStorms();
    if (first) {
      const h = decodeURIComponent(location.hash.slice(1));
      let key = h && man.missions[h] ? h : null, sid = key ? man.missions[key].storm : (h && man.storms[h] ? h : null);
      if (!sid) sid = defaultStorm();
      pickStorm(sid, key);
    } else {
      fillMissions();
      const cur = S.key && man.missions[S.key];
      if (cur && prevMis && (cur.bytes !== prevMis.bytes || cur.last !== prevMis.last)) await openMission(S.key, true);
      else { status(); if (cur) tdrLoad(S.key, true); }      // TDR graphics arrive ~70 min after each fix, even after landing
    }
    pod();
    alertCheck();
    return true;
  }
  function defaultStorm() {
    const st = Object.entries(S.man.storms);
    const rank = ([sid, s]) => (s.active ? 2e10 : 0) + (s.missions.length ? 1e10 : 0) + (s.last || 0);
    st.sort((a, b) => rank(b) - rank(a));
    return st.length ? st[0][0] : null;
  }
  function stormLabel(sid, s) {
    const n = (s.name && s.name !== sid) ? s.name : "";
    return `${n ? n + " " : ""}(${sid.slice(0, 4)} ${sid.slice(4)})${s.active ? " · active" : ""}${s.pinned ? " · archive" : ""} — ${s.missions.length} mission${s.missions.length === 1 ? "" : "s"}`;
  }
  function fillStorms() {
    const sel = $("selStorm"), st = Object.entries(S.man.storms);
    st.sort((a, b) => (b[1].active - a[1].active) || ((b[1].last || 0) - (a[1].last || 0)));
    const groups = { cur: [], arch: [] };
    for (const [sid, s] of st) (s.year === S.man.season && !s.pinned ? groups.cur : groups.arch).push([sid, s]);
    sel.innerHTML = (groups.cur.length ? `<optgroup label="${S.man.season} season">` + groups.cur.map(([sid, s]) =>
      `<option value="${sid}">${esc(stormLabel(sid, s))}</option>`).join("") + "</optgroup>" : "") +
      (groups.arch.length ? `<optgroup label="Past storms">` + groups.arch.map(([sid, s]) =>
        `<option value="${sid}">${esc(stormLabel(sid, s))}</option>`).join("") + "</optgroup>" : "");
    if (S.sid) sel.value = S.sid;
  }
  function missionLabel(k, m) {
    const span = m.first ? `${dayhm(m.first)}–${hm(m.last)}Z` : "";
    const what = m.n_vdm ? `${m.n_vdm} fix${m.n_vdm > 1 ? "es" : ""}` : (m.how && m.how.startsWith("nearest") ? "surveillance" : "");
    return `${isAir(m) ? "● airborne · " : ""}${m.aircraft} ${m.mission} · ${span}${what ? " · " + what : ""}${m.n_drop ? ` · ${m.n_drop} sondes` : ""}`;
  }
  function fillMissions() {
    const sel = $("selMission"), s = S.man.storms[S.sid];
    const ks = s ? s.missions : [];
    sel.innerHTML = ks.length ? ks.map(k => `<option value="${k}">${esc(missionLabel(k, S.man.missions[k]))}</option>`).join("")
      : `<option value="">No reconnaissance yet</option>`;
    if (S.key && ks.includes(S.key)) sel.value = S.key;
  }
  async function pickStorm(sid, key) {
    S.sid = sid; $("selStorm").value = sid || "";
    fillMissions();
    const s = S.man.storms[sid];
    S.storm = null;
    loadStorm(sid);
    if (!s || !s.missions.length) { clearMission(); status(); return; }
    await openMission(key && s.missions.includes(key) ? key : s.missions[0]);
  }
  async function loadStorm(sid) {
    const s = S.man.storms[sid];
    lyr.btk.clearLayers(); lyr.fixes.clearLayers();
    if (!s) return;
    drawFixTrack(sid);
    if (!s.file) return;
    try {
      const st = await dataFetch(s.file, s.active ? Math.floor(Date.now() / 600000) : s.last, s.active);
      if (S.sid !== sid || !st) return;
      S.storm = st; drawStorm(st, s);
      if (S.mis) { radBuild(); srBuild(); }
      histBuild();
    } catch (e) { /* track is context only */ }
  }
  async function openMission(key, refresh) {
    const m = S.man.missions[key];
    if (!m) return;
    S.key = key; $("selMission").value = key;
    try { history.replaceState(null, "", "#" + encodeURIComponent(key)); } catch (e) { /* sandboxed */ }
    const air = isAir(m);
    if (!refresh) { setStatus(`<span class="mu">Loading ${esc(m.aircraft)} ${esc(m.mission)}…</span>`); }
    let mis;
    try { mis = await dataFetch(m.file, air ? Math.floor(Date.now() / 60000) : m.bytes + "_" + m.last, air); }
    catch (e) { setStatus(`<span class="mu">That mission's file could not be loaded. Try again shortly.</span>`); return; }
    if (S.key !== key || !mis) return;
    S.mis = mis;
    prepRows();
    drawMission(!refresh);
    side(); lists(); drawTS(); status(); drawFixTrack(S.sid);
    analyses();
    tdrLoad(key, refresh);
  }
  function clearMission() {
    S.mis = null; S.key = null; S.rows = [];
    for (const k of ["track", "barbs", "drops", "vdm", "hover", "tdr", "plane"]) lyr[k].clearLayers();
    T.doc = null; $("tdr").hidden = true;
    for (const id of ["rad", "sr", "eye"]) $(id).hidden = true;
    R.legs = []; histBuild();
    $("side").innerHTML = `<h2>Mission</h2><p class="empty">No reconnaissance has been flown into this storm yet. The plan of the day below lists what is tasked.</p>`;
    $("drops").innerHTML = $("vdms").innerHTML = `<p class="empty">—</p>`;
    $("legend").innerHTML = ""; drawTS();
  }

  // ── mission data ─────────────────────────────────────────────────────────
  function prepRows() {
    const h = S.mis.hdob, ix = {};
    h.cols.forEach((c, i) => ix[c] = i);
    S.col = ix;
    S.rows = h.rows;
    // dewpoint depression only where both are valid
    S.dd = S.rows.map(r => (r[ix.t_] != null && r[ix.td] != null) ? Math.max(0, r[ix.t_] - r[ix.td]) : null);
  }
  const val = (i, v) => v === "dd" ? S.dd[i] : S.rows[i][S.col[v]];

  function rampFor(v) {
    const xs = S.rows.map((r, i) => val(i, v)).filter(x => x != null).sort((a, b) => a - b);
    if (!xs.length) return null;
    const lo = xs[Math.floor(xs.length * 0.02)], hi = xs[Math.ceil(xs.length * 0.98) - 1];
    return { lo, hi: hi > lo ? hi : lo + 1 };
  }
  function colourFn(v) {
    const d = VARS[v];
    if (d.bins) return x => binCol(d.bins, x);
    const r = rampFor(v);
    if (!r) return () => "#888";
    return x => TURBO[Math.max(0, Math.min(TURBO.length - 1, Math.floor((x - r.lo) / (r.hi - r.lo) * TURBO.length)))];
  }

  function drawMission(fit) {
    for (const k of ["track", "barbs", "drops", "vdm", "hover"]) lyr[k].clearLayers();
    if (!S.mis) return;
    drawTrack();
    drawDrops();
    drawVDMs();
    drawRecco();
    const pts = S.rows.filter(r => r[S.col.la] != null).map(r => [r[S.col.la], r[S.col.lo]]);
    S.mis.drops.forEach(d => { const p = d.spg || d.spl || d.rel; if (p) pts.push([p.la, p.lo]); });
    if (fit && pts.length && S.fitKey !== S.key) {
      S.fitKey = S.key;
      map.fitBounds(L.latLngBounds(pts).pad(0.08), { maxZoom: 8 });
    }
    legend();
  }
  // the variable actually drawn: a mission with no valid SFMR (high-altitude flights, or every value flagged) is drawn by
  // flight-level wind instead, and the legend says so
  function effVar() {
    S.vNote = "";
    if (S.v !== "sfmr" || !S.rows.length) return S.v;
    if (S.rows.some((r, i) => val(i, "sfmr") != null)) return "sfmr";
    S.vNote = "This mission has no valid SFMR values (not measured at this altitude, or flagged by the aircraft): coloured by flight-level wind.";
    return "ws";
  }
  function drawTrack() {
    lyr.track.clearLayers();
    S.ve = effVar();
    // the pressed button is the variable drawn (SFMR stayed pressed while a flight without SFMR was coloured by FL wind)
    $("segVar").querySelectorAll("button").forEach(x => x.classList.toggle("on", x.dataset.v === S.ve));
    const ix = S.col, rows = S.rows, col = colourFn(S.ve), cas = css("--track-case");
    // casing under the whole track (legible on both basemaps), then one coloured segment per ob pair
    let run = [];
    const flush = () => { if (run.length > 1) L.polyline(run, { renderer: rend, color: cas, weight: 6, interactive: false }).addTo(lyr.track); run = []; };
    for (let i = 0; i < rows.length; i++) {
      const r = rows[i];
      if (r[ix.la] == null) { flush(); continue; }
      if (run.length && rows[i - 1] && r[0] - rows[i - 1][0] > 300) flush();
      run.push([r[ix.la], r[ix.lo]]);
    }
    flush();
    for (let i = 1; i < rows.length; i++) {
      const a = rows[i - 1], b = rows[i];
      if (a[ix.la] == null || b[ix.la] == null || b[0] - a[0] > 300) continue;
      const x = val(i, S.ve);
      const seg = [[a[ix.la], a[ix.lo]], [b[ix.la], b[ix.lo]]];
      if (x == null) L.polyline(seg, { renderer: rend, color: "#9a9a9a", weight: 1.6, opacity: 0.9, interactive: false }).addTo(lyr.track);
      else L.polyline(seg, { renderer: rend, color: col(x), weight: 3.6, interactive: false }).addTo(lyr.track);
    }
    drawBarbs();
    drawPlane();
  }
  // aircraft icon at the newest valid position, pointing along the last few minutes of track. Airborne: live colour
  // with a pulse; a finished flight keeps a muted icon at its last ob (usually the landing field).
  function drawPlane() {
    lyr.plane.clearLayers();
    const ix = S.col, rows = S.rows;
    let i1 = rows.length - 1;
    while (i1 >= 0 && rows[i1][ix.la] == null) i1--;
    if (i1 < 0) return;
    let i0 = i1 - 1;                                   // a point ~2+ min back, so one jittery ob does not set the heading
    while (i0 > 0 && (rows[i0][ix.la] == null || rows[i1][0] - rows[i0][0] < 120)) i0--;
    const la1 = rows[i1][ix.la], lo1 = rows[i1][ix.lo];
    let hdg = 0;
    if (i0 >= 0 && rows[i0][ix.la] != null) {
      const la0 = rows[i0][ix.la], lo0 = rows[i0][ix.lo], r = Math.PI / 180;
      const y = Math.sin((lo1 - lo0) * r) * Math.cos(la1 * r);
      const x = Math.cos(la0 * r) * Math.sin(la1 * r) - Math.sin(la0 * r) * Math.cos(la1 * r) * Math.cos((lo1 - lo0) * r);
      hdg = (Math.atan2(y, x) / r + 360) % 360;
    }
    const m = S.man && S.man.missions[S.key], air = isAir(m);
    const fill = air ? css("--live") : (dark() ? "#c9ccd2" : "#4d544e"), edge = dark() ? "#0d1014" : "#ffffff";
    const svg = `<svg width="30" height="30" viewBox="-15 -15 30 30" style="overflow:visible"><g transform="rotate(${hdg.toFixed(0)})">` +
      `<path d="M0 -12 L1.8 -4 L11 1.5 L11 3.5 L1.8 1.5 L1.4 8 L4.5 10.5 L4.5 12 L0 11 L-4.5 12 L-4.5 10.5 L-1.4 8 L-1.8 1.5 L-11 3.5 L-11 1.5 L-1.8 -4 Z" ` +
      `fill="${fill}" stroke="${edge}" stroke-width="1.4" stroke-linejoin="round"/></g></svg>`;
    const pulse = air ? `<span class="plane-pulse" style="border-color:${fill}"></span>` : "";
    const t = m && m.last ? dayhm(m.last) : "";
    L.marker([la1, lo1], { zIndexOffset: 1000, keyboard: false, title: `${m ? m.aircraft + " " + m.mission : "aircraft"} - ${air ? "latest position" : "last observation"}${t ? " " + t : ""}`,
      icon: L.divIcon({ className: "plane-ic", html: pulse + svg, iconSize: [30, 30], iconAnchor: [15, 15] }) }).addTo(lyr.plane);
  }
  // wind barbs (flight-level wind), thinned to ~48 px apart on screen
  function barbSVG(dir, kt, colr) {
    const s = 34, c = s / 2, len = 15;
    let k = Math.round(kt / 5) * 5, y = -len, parts = [];
    const add = (d) => parts.push(d);
    if (k < 5) return `<svg width="${s}" height="${s}" viewBox="0 0 ${s} ${s}"><circle cx="${c}" cy="${c}" r="3" fill="none" stroke="${colr}" stroke-width="1.5"/></svg>`;
    add(`M0 0 L0 ${-len}`);
    while (k >= 50) { add(`M0 ${y} L8 ${y + 2.5} L0 ${y + 5} Z`); y += 6; k -= 50; }
    while (k >= 10) { add(`M0 ${y} L9 ${y - 3}`); y += 3.2; k -= 10; }
    if (k >= 5) { if (y === -len) y += 3.2; add(`M0 ${y} L5 ${y - 1.7}`); }
    return `<svg width="${s}" height="${s}" viewBox="0 0 ${s} ${s}"><g transform="translate(${c} ${c}) rotate(${dir})" stroke="${colr}" fill="${colr}" stroke-width="1.5" stroke-linecap="round">` +
      `<path d="${parts.join(" ")}"/></g></svg>`;
  }
  function drawBarbs() {
    lyr.barbs.clearLayers();
    if (!S.rows.length) return;
    const ix = S.col, colr = dark() ? "#e6e8ec" : "#1f2620";
    let last = null;
    for (let i = 0; i < S.rows.length; i++) {
      const r = S.rows[i];
      if (r[ix.la] == null || r[ix.ws] == null) continue;
      const p = map.latLngToContainerPoint([r[ix.la], r[ix.lo]]);
      if (last && p.distanceTo(last) < 48) continue;
      last = p;
      L.marker([r[ix.la], r[ix.lo]], { interactive: false, keyboard: false,
        icon: L.divIcon({ className: "barb", html: barbSVG(r[ix.wd], r[ix.ws], colr), iconSize: [34, 34], iconAnchor: [17, 17] }) }).addTo(lyr.barbs);
    }
  }
  // permanent fix labels only once zoomed in enough to separate them
  const lowZ = () => $("map").classList.toggle("lowz", map.getZoom() < 6.5);
  map.on("zoomend", () => { lowZ(); if (S.mis) drawBarbs(); });
  lowZ();

  function dropEst(d) {
    const e = d.sfc_est || {};
    return e.wl150 != null ? e.wl150 : e.mbl != null ? e.mbl : (d.sfc_wind ? d.sfc_wind.ws : null);
  }
  function drawDrops() {
    lyr.drops.clearLayers();
    S.mis.drops.forEach((d, i) => {
      const p = d.spg || d.spl || d.rel;
      if (!p) return;
      const est = dropEst(d);
      const mk = L.circleMarker([p.la, p.lo], { renderer: rend, radius: 7, weight: 2, color: dark() ? "#ffffff" : "#111111",
        fillColor: est != null ? binCol(WIND, est) : "#bbbbbb", fillOpacity: 1 }).addTo(lyr.drops);
      mk.bindTooltip(`Dropsonde ${d.obnum != null ? "OB " + d.obnum : ""} · ${d.spl_t ? hm(d.spl_t) + "Z" : ""} · ${d.sfc_p ? d.sfc_p + " hPa" : "no splash"}`, { direction: "top" });
      mk.bindPopup(() => dropPopup(d, i), { maxWidth: 360 });
      d._mk = mk;
    });
  }
  function dropPopup(d, i) {
    const e = d.sfc_est || {}, sp = d.spg || d.spl;
    const rows = [
      ["Release", d.rel ? `${d.rel_t ? hms(d.rel_t) + "Z" : ""} ${ll(d.rel.la, d.rel.lo)}` : "—"],
      ["Splash", sp ? `${d.spl_t ? hms(d.spl_t) + "Z" : ""} ${ll(sp.la, sp.lo)}${d.spl && !d.spg ? " (estimated)" : ""}` : "—"],
      ["Surface pressure", d.sfc_p ? `<b>${d.sfc_p} hPa</b>` : "no surface level"],
      ["Wind at the surface level", d.sfc_wind ? dirTxt(d.sfc_wind.wd, d.sfc_wind.ws) : "—"],
      ["Mean, lowest 150 m", d.wl150 ? `${dirTxt(d.wl150.wd, d.wl150.ws)} → est. surface <b>${e.wl150} kt</b>` : "—"],
      ["Mean, lowest 500 m", d.mbl ? `${dirTxt(d.mbl.wd, d.mbl.ws)} → est. surface <b>${e.mbl} kt</b>` : "—"],
      ["Layer mean", d.dlm ? `${dirTxt(d.dlm.wd, d.dlm.ws)} (${d.dlm.pbot}–${d.dlm.ptop} hPa)` : "—"],
    ];
    if (d.tags && d.tags.length) rows.push(["Remarks", esc(d.tags.join(", "))]);
    const div = document.createElement("div");
    div.innerHTML = `<b>Dropsonde${d.obnum != null ? " OB " + d.obnum : ""}</b> · ${esc(S.mis.aircraft)} ${esc(S.mis.mission)}` +
      `<table>${rows.map(r => `<tr><td>${r[0]}</td><td>${r[1]}</td></tr>`).join("")}</table>` +
      `<button type="button">Skew-T and hodograph</button>` +
      `<details><summary>Raw message</summary><pre>${esc(d.raw)}</pre></details>`;
    div.querySelector("button").onclick = () => openSkewT(i);
    return div;
  }
  function vdmRows(v) {
    const r = [["Fix", `${hms(v.t)}Z ${ll(v.la, v.lo)}`], ["Minimum pressure", v.mslp ? `<b>${v.mslp} hPa</b>${v.mslp_extrap ? " (extrapolated)" : ""}` : "—"]];
    if (v.lvl) r.push(["Flight level", `${v.lvl} hPa at ${v.lvl_z} m`]);
    if (v.ctr_sfc_wind) r.push(["Sonde surface wind at the centre", dirTxt(v.ctr_sfc_wind.wd, v.ctr_sfc_wind.ws)]);
    if (v.in_sfc) r.push(["Max surface wind, inbound", `${v.in_sfc.ws} kt${v.in_sfc.brg != null ? ` · ${v.in_sfc.brg}° ${v.in_sfc.rng} n mi` : ""}`]);
    if (v.in_fl) r.push(["Max FL wind, inbound", `${dirTxt(v.in_fl.wd, v.in_fl.ws)}${v.in_fl.brg != null ? ` · ${v.in_fl.brg}° ${v.in_fl.rng} n mi` : ""}`]);
    if (v.out_sfc) r.push(["Max surface wind, outbound", `${v.out_sfc.ws} kt${v.out_sfc.brg != null ? ` · ${v.out_sfc.brg}° ${v.out_sfc.rng} n mi` : ""}`]);
    if (v.out_fl) r.push(["Max FL wind, outbound", `${dirTxt(v.out_fl.wd, v.out_fl.ws)}${v.out_fl.brg != null ? ` · ${v.out_fl.brg}° ${v.out_fl.rng} n mi` : ""}`]);
    if (v.eye) r.push(["Eye", esc(v.eye) + (v.eye_shape ? ` · ${esc(v.eye_shape)}` : "")]);
    if (v.t_out && v.t_in) r.push(["Temperature out / in", `${v.t_out.t} / ${v.t_in.t} °C`]);
    if (v.max_fl) r.push(["Max FL wind (remark)", `${v.max_fl.ws} kt · ${v.max_fl.brg}° ${v.max_fl.rng} n mi · ${v.max_fl.hms}Z`]);
    return r;
  }
  function drawVDMs() {
    lyr.vdm.clearLayers();
    for (const v of S.mis.vdms) {
      const mk = L.marker([v.la, v.lo], { icon: L.divIcon({ className: "", iconSize: [22, 22], iconAnchor: [11, 11],
        html: `<svg width="22" height="22" viewBox="0 0 22 22"><circle cx="11" cy="11" r="8" fill="#fff" stroke="#b3261e" stroke-width="2.5"/>` +
          `<path d="M11 3v16M3 11h16" stroke="#b3261e" stroke-width="2"/></svg>` }), zIndexOffset: 500 }).addTo(lyr.vdm);
      mk.bindTooltip(`${v.mslp ? v.mslp + " hPa · " : ""}${hm(v.t)}Z`, { permanent: true, direction: "right", className: "fixlab", offset: [8, 0] });
      mk.bindPopup(`<b>Vortex fix</b> · ${esc(v.aircraft || "")} OB ${v.obnum ?? ""}<table>${vdmRows(v).map(r => `<tr><td>${r[0]}</td><td>${r[1]}</td></tr>`).join("")}</table>` +
        `<details><summary>Raw message</summary><pre>${esc(v.raw)}</pre></details>`, { maxWidth: 380 });
      for (const k of ["in_fl", "out_fl"]) {
        const w = v[k]; if (!w || w.la == null) continue;
        L.circleMarker([w.la, w.lo], { renderer: rend, radius: 4, color: "#b3261e", weight: 1.5, fillColor: "#fff", fillOpacity: 1 })
          .bindTooltip(`Max FL wind ${k === "in_fl" ? "inbound" : "outbound"}: ${w.ws} kt${w.hms ? " at " + w.hms + "Z" : ""}`).addTo(lyr.vdm);
      }
    }
  }
  function drawRecco() {
    for (const r of S.mis.reccos || []) {
      L.circleMarker([r.la, r.lo], { renderer: rend, radius: 3.5, color: "#555", weight: 1, fillColor: "#ddd", fillOpacity: 1 })
        .bindTooltip(`RECCO ${hm(r.t)}Z${r.ws != null ? ` · ${dirTxt(r.wd, r.ws)}` : ""}${r.alt_m ? ` · ${r.alt_m} m` : ""}${r.rmk ? " · " + esc(r.rmk.replace(/^\S+ \S+ \S+\s+OB \d+\s*/, "")) : ""}`)
        .addTo(lyr.vdm);
    }
  }
  // every fix of the storm (all missions, from the manifest) as a track with minimum-pressure labels
  function drawFixTrack(sid) {
    lyr.fixes.clearLayers();
    const s = S.man && S.man.storms[sid]; if (!s) return;
    const fx = stormFixes(sid);
    if (fx.length > 1) L.polyline(fx.map(f => [f.la, f.lo]), { renderer: rend, color: "#b3261e", weight: 1.6, dashArray: "4 4", interactive: false }).addTo(lyr.fixes);
    for (const f of fx) {
      if (f.k === S.key) continue;                               // the open mission draws its own
      L.circleMarker([f.la, f.lo], { renderer: rend, radius: 3.5, color: "#b3261e", weight: 1.5, fillColor: "#fff", fillOpacity: 1 })
        .bindTooltip(`Fix ${dayhm(f.t)}${f.p ? " · " + f.p + " hPa" + (f.x ? " (extrapolated)" : "") : ""} · ${esc(S.man.missions[f.k].aircraft)}`).addTo(lyr.fixes);
    }
  }
  // every vortex fix of a storm, all missions (manifest): {t, la, lo, p (hPa or null), x (1 = extrapolated), k (mission)}
  function stormFixes(sid) {
    const s = S.man && S.man.storms[sid], fx = [];
    if (!s) return fx;
    for (const k of s.missions) for (const f of (S.man.missions[k].fixes || [])) fx.push({ t: f[0], la: f[1], lo: f[2], p: f[3], x: f[4] || 0, k });
    return fx.sort((a, b) => a.t - b.t);
  }
  function drawStorm(st, s) {
    lyr.btk.clearLayers();
    const bt = st.btk || [];
    const lineCol = dark() ? "#c9ccd3" : "#3d433e";
    if (bt.length > 1) L.polyline(bt.map(r => [r[1], r[2]]), { renderer: rend, color: lineCol, weight: 1.5, opacity: 0.8, interactive: false }).addTo(lyr.btk);
    for (const r of bt) {
      L.circleMarker([r[1], r[2]], { renderer: rend, radius: 3, weight: 1, color: lineCol, fillColor: r[3] != null ? binCol(WIND, r[3]) : "#ccc", fillOpacity: 1 })
        .bindTooltip(`Best track ${dayhm(r[0])} · ${r[3] ?? "—"} kt${r[4] ? " · " + r[4] + " hPa" : ""} · ${esc(r[5] || "")}`).addTo(lyr.btk);
    }
    const f = st.fcst;
    if (f && s.active && f.pts && f.pts.length) {
      const pts = f.pts.slice().sort((a, b) => a[0] - b[0]);
      L.polyline(pts.map(p => [p[1], p[2]]), { renderer: rend, color: lineCol, weight: 1.5, dashArray: "6 5", interactive: false }).addTo(lyr.btk);
      for (const p of pts) {
        if (p[0] === 0) continue;
        L.circleMarker([p[1], p[2]], { renderer: rend, radius: 4, weight: 1.5, color: lineCol, fillColor: "#ffffff", fillOpacity: 0.9 })
          .bindTooltip(`NHC forecast ${p[0]} h (${dayhm(f.t0 + p[0] * 3600)}) · ${p[3]} kt ${catOf(p[3])}`).addTo(lyr.btk);
      }
    }
    const a = (S.man.active || []).find(x => x.id === S.sid);
    if (a && a.la != null) {
      L.marker([a.la, a.lo], { interactive: true, icon: L.divIcon({ className: "", iconSize: [26, 26], iconAnchor: [13, 13],
        html: `<svg width="26" height="26" viewBox="-13 -13 26 26"><g fill="none" stroke="${lineCol}" stroke-width="2.2"><circle r="4.5"/>` +
          `<path d="M0 -4.5 C -2 -10, -7 -11, -11 -9 M0 4.5 C 2 10, 7 11, 11 9"/></g></svg>` }) })
        .bindTooltip(`NHC advisory ${a.adv || ""} · ${a.name} · ${a.vmax} kt · ${a.mslp} hPa`).addTo(lyr.btk);
    }
  }

  // ── legend, side panel, lists ────────────────────────────────────────────
  function legend() {
    const v = S.ve || S.v, d = VARS[v];
    let h = (S.vNote ? `<span class="mu" style="flex-basis:100%">${S.vNote}</span>` : "") + `<span><b>${d.lab}</b> <span class="mu">(${d.unit})</span></span>`;
    if (d.bins) h += d.bins.map(b => `<span class="sw"><i style="background:${b[1]}"></i>${b[2]}</span>`).join("");
    else {
      const r = rampFor(v);
      h += r ? `<span class="sw"><i style="width:9em;background:linear-gradient(90deg,${TURBO.join(",")})"></i>${r.lo.toFixed(1)} to ${r.hi.toFixed(1)}</span>` : "";
    }
    h += `<span class="sw"><i style="background:#9a9a9a;height:2px"></i><span class="mu">no valid value (missing or flagged)</span></span>`;
    h += `<span class="sw mu">Dots: dropsondes (est. surface wind) · red crosses: vortex fixes · barbs: flight-level wind</span>`;
    if (T.doc && T.doc.key === S.key && T.doc.centres.some(c => c.c)) h += `<span class="sw mu"><i style="background:${TDRC};height:0.7em;width:0.7em;transform:rotate(45deg)"></i>TDR centres (2 km; line to the 6 km centre)</span>`;
    $("legend").innerHTML = h;
  }
  function stat(label, o, unit, fmt) {
    if (!o) return `<dt>${label}</dt><dd>—</dd>`;
    return `<dt>${label}</dt><dd><b>${fmt ? fmt(o.v) : o.v} ${unit}</b> <small>${hm(o.t)}Z · ${ll(o.la, o.lo)}</small> ` +
      `<a data-go="${o.t}" title="show on the map">map</a></dd>`;
  }
  function side() {
    const m = S.man.missions[S.key], s = S.man.storms[S.sid] || {};
    if (!m) return;
    const air = isAir(m);
    const alt = m.alt_med ? `${m.alt_med.toLocaleString("en-US")} m (${Math.round(m.alt_med * 3.28084 / 100) * 100 >= 18000 ? "FL" + Math.round(m.alt_med * 3.28084 / 1000) * 10 : Math.round(m.alt_med * 3.28084).toLocaleString("en-US") + " ft"})` : "—";
    const nearTxt = m.near != null ? "within 300 n mi of the centre" : "whole flight (no storm track)";
    $("side").innerHTML = `<h2>${esc(m.aircraft)} ${esc(m.mission)}</h2>
      <p class="sub">${esc(m.type || "")}${s.name ? " · " + esc(s.name) : ""} ${S.sid ? "(" + S.sid + ")" : ""}${m.how && m.how !== "mission number" ? `<br>Storm assigned by ${esc(m.how)}` : ""}</p>
      <dl class="stats">
        <dt>Flight</dt><dd>${m.first ? `${dayhm(m.first)} – ${dayhm(m.last)}` : "—"} ${air ? `<b style="color:var(--live)">airborne</b>` : ""}</dd>
        <dt>Typical altitude</dt><dd>${alt}</dd>
        <dt colspan="2" style="grid-column:1/-1;margin-top:0.3rem"><small>Maxima and minima ${nearTxt}</small></dt>
        ${stat("Max SFMR wind", m.max_sfmr, "kt")}
        ${stat("Max flight-level wind", m.max_fl, "kt")}
        ${stat("Peak 10-s FL wind", m.max_peak, "kt")}
        ${stat("Min extrap. pressure", m.min_x, "hPa", v => v.toFixed(1))}
        ${stat("Min pressure (fix)", m.min_vdm, "hPa")}
        <dt>Dropsondes</dt><dd>${m.n_drop}</dd>
        <dt>Vortex fixes</dt><dd>${m.n_vdm}</dd>
        <dt>RECCO obs</dt><dd>${m.n_recco}</dd>
        <dt>HDOB obs</dt><dd>${m.n_hdob.toLocaleString("en-US")} <small>(30 s)</small></dd>
      </dl>
      <p class="qcnote">Quality control: ${m.n_qc ? m.n_qc.toLocaleString("en-US") : 0} of ${m.n_hdob.toLocaleString("en-US")} observations carry an aircraft QC flag;
        ${(m.flagged || 0).toLocaleString("en-US")} flagged values were removed${m.screened ? `, and ${m.screened} more failed a range or position check` : ""}.${qcBreak()}</p>`;
    $("side").querySelectorAll("a[data-go]").forEach(a => a.onclick = () => goTime(+a.dataset.go, true));
  }
  // what the flags said, in words (second digit = meteorological fields, NHC table)
  function qcBreak() {
    const ix = S.col, n = {}, W = { 1: "temperature/dewpoint", 2: "flight-level wind", 3: "SFMR", 4: "temperature/dewpoint and FL wind",
      5: "temperature/dewpoint and SFMR", 6: "FL wind and SFMR", 9: "every meteorological value" };
    for (const r of S.rows) { const q = r[ix.qc]; if (q && q[1] !== "0" && q[1] !== "/") n[q[1]] = (n[q[1]] || 0) + 1; }
    const parts = Object.entries(n).sort((a, b) => b[1] - a[1]).map(([k, c]) => `${c.toLocaleString("en-US")} with ${W[k] || "flag " + k} questionable`);
    return parts.length ? ` By field: ${parts.join("; ")}.` : "";
  }
  function lists() {
    const ds = S.mis.drops;
    $("drops").innerHTML = ds.length ? `<table><thead><tr><th>Splash</th><th>Position</th><th>Sfc P</th><th>Sfc wind</th><th>Est. sfc (WL150 / MBL)</th><th>Note</th></tr></thead><tbody>` +
      ds.map((d, i) => {
        const sp = d.spg || d.spl || d.rel, e = d.sfc_est || {};
        return `<tr class="click" data-i="${i}"><td>${d.spl_t ? hm(d.spl_t) + "Z" : d.rel_t ? hm(d.rel_t) + "Z rel." : "—"}</td><td>${sp ? ll(sp.la, sp.lo) : "—"}</td>` +
          `<td>${d.sfc_p ? d.sfc_p + " hPa" : '<span class="mu">no splash</span>'}</td><td>${d.sfc_wind ? dirTxt(d.sfc_wind.wd, d.sfc_wind.ws) : "—"}</td>` +
          `<td>${e.wl150 ?? "—"} / ${e.mbl ?? "—"} kt</td><td class="mu">${esc((d.tags || []).join(", "))}</td></tr>`;
      }).join("") + "</tbody></table>" : `<p class="empty">No dropsondes on this mission.</p>`;
    $("drops").querySelectorAll("tr[data-i]").forEach(tr => {
      tr.onclick = () => openSkewT(+tr.dataset.i);
      tr.onmouseenter = () => { const d = ds[+tr.dataset.i]; if (d._mk) d._mk.openTooltip(); };
      tr.onmouseleave = () => { const d = ds[+tr.dataset.i]; if (d._mk) d._mk.closeTooltip(); };
    });
    const vs = S.mis.vdms;
    $("vdms").innerHTML = vs.length ? `<table><thead><tr><th>Fix</th><th>Position</th><th>Min P</th><th>Max FL wind in / out</th><th>Eye</th></tr></thead><tbody>` +
      vs.map((v, i) => `<tr class="click" data-i="${i}"><td>${hm(v.t)}Z</td><td>${ll(v.la, v.lo)}</td><td>${v.mslp ? v.mslp + " hPa" : "—"}</td>` +
        `<td>${v.in_fl ? v.in_fl.ws : "—"} / ${v.out_fl ? v.out_fl.ws : "—"} kt</td><td class="mu">${esc(v.eye || "")}${v.eye_shape ? " " + esc(v.eye_shape) : ""}</td></tr>`).join("") +
      "</tbody></table>" : `<p class="empty">No vortex data messages on this mission${S.man.missions[S.key].how && S.man.missions[S.key].how.startsWith("nearest") ? " (a surveillance flight does not fix the centre)" : ""}.</p>`;
    $("vdms").querySelectorAll("tr[data-i]").forEach(tr => tr.onclick = () => {
      const v = vs[+tr.dataset.i]; map.setView([v.la, v.lo], Math.max(map.getZoom(), 7));
      lyr.vdm.eachLayer(l => { if (l.getLatLng && l.getPopup && l.getPopup() && l.getLatLng().lat === v.la && l.getLatLng().lng === v.lo) l.openPopup(); });
    });
  }
  function pod() {
    const p = (S.man.pods || [])[S.man.pods ? S.man.pods.length - 1 : 0];
    $("pod").textContent = p ? p.text : "No plan of the day in the mirror.";
    const box = $("podbox").querySelector("summary");
    box.textContent = p ? `Plan of the day (CARCAH), issued ${dayhm(p.t)}${p.valid ? " · " + p.valid.toLowerCase().replace("valid", "valid") : ""}` : "Plan of the day (CARCAH)";
  }

  // ── status: data age, live polling ───────────────────────────────────────
  function setStatus(h) { $("status").innerHTML = h; }
  function status() {
    if (!S.man) return;
    const m = S.key && S.man.missions[S.key];
    const parts = [];
    if (m) {
      const air = isAir(m);
      if (air) parts.push(`<span class="pill live"><i></i>Airborne</span>`);
      const tl = m.last_hdob || m.last;
      if (tl && Date.now() / 1000 - tl > 2 * 86400) parts.push(`<span>Flown <b>${dayhm(m.first)} – ${dayhm(m.last)}</b></span>`);
      else if (m.last_hdob) parts.push(`<span>Newest observation <b>${dayhm(m.last_hdob)}</b> (${ago(m.last_hdob)})</span>`);
      else if (m.last) parts.push(`<span>Newest message <b>${dayhm(m.last)}</b> (${ago(m.last)})</span>`);
    }
    parts.push(`<span class="mu">Mirror checked ${ago(S.man.updated)} · page refreshes ${anyAir() ? "every 90 s while a mission is airborne" : "every 10 min"}</span>`);
    setStatus(parts.join(""));
    const stale = Date.now() / 1000 - S.man.updated > 3 * 3600 && (S.man.active || []).length;
    $("warn").hidden = !stale;
    if (stale) $("warn").textContent = `The mirror last ran ${ago(S.man.updated)}; newer reconnaissance may exist at the NHC.`;
  }
  const anyAir = () => S.man && Object.values(S.man.missions).some(isAir);
  setInterval(status, 20000);
  (function poll() {
    const wait = anyAir() ? 90e3 : 600e3;
    setTimeout(async () => { if (!document.hidden || AL.on) await loadManifest(false); poll(); }, wait);
  })();
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden && S.man && Date.now() - S.manAt > (anyAir() ? 90e3 : 600e3)) loadManifest(false);
  });

  // ── time series (canvas) ─────────────────────────────────────────────────
  const cv = $("ts"), ctx = cv.getContext("2d");
  let TS = null;
  function drawTS() {
    const dpr = window.devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight;
    if (!W) return;
    cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    $("tskey").innerHTML = `<span><i style="background:${css("--s-fl")}"></i>Flight-level wind (30 s)</span>` +
      `<span><i style="background:${css("--s-pk")}"></i>Peak 10-s FL wind</span><span><i style="background:${css("--s-sfmr")}"></i>SFMR surface wind</span>` +
      `<span><i style="background:${css("--s-slp")}"></i>Extrapolated surface pressure</span><span><i style="background:${css("--s-alt")}"></i>Geopotential height</span>` +
      `<span class="mu">red ticks: vortex fixes · triangles: dropsondes</span>`;
    if (!S.rows.length) { TS = null; ctx.fillStyle = css("--muted"); ctx.font = "14px sans-serif"; ctx.fillText("No flight data.", 12, 24); return; }
    const ix = S.col, rows = S.rows;
    const L0 = 52, R0 = W - 14, t0 = rows[0][0], t1 = rows[rows.length - 1][0] || t0 + 1;
    const X = t => L0 + (t - t0) / Math.max(1, t1 - t0) * (R0 - L0);
    const pan = [{ y0: 6, y1: H * 0.46, keys: ["ws", "wp", "sfmr"], unit: "kt" }, { y0: H * 0.5, y1: H * 0.72, keys: ["x"], unit: "hPa" },
                 { y0: H * 0.76, y1: H - 22, keys: ["z"], unit: "m" }];
    const fg = css("--axis"), grid = css("--grid");
    ctx.font = "11px 'Source Sans 3', sans-serif"; ctx.lineWidth = 1;
    for (const p of pan) {
      let lo = Infinity, hi = -Infinity;
      for (const r of rows) for (const k of p.keys) { const v = r[ix[k]]; if (v != null) { lo = Math.min(lo, v); hi = Math.max(hi, v); } }
      if (!isFinite(lo)) { lo = 0; hi = 1; }
      if (p.unit === "kt") lo = 0;
      if (hi - lo < (p.unit === "hPa" ? 4 : 10)) { const c = (hi + lo) / 2, h = p.unit === "hPa" ? 2 : 5; lo = p.unit === "kt" ? 0 : c - h; hi = c + h; }
      const pad = (hi - lo) * 0.06; hi += pad; if (p.unit !== "kt") lo -= pad;
      p.lo = lo; p.hi = hi; p.Y = v => p.y1 - (v - lo) / (hi - lo) * (p.y1 - p.y0);
      const step = tickStep(hi - lo, p.y1 - p.y0);   // as many labels as the panel height holds (the hPa and m panels were stacked solid)
      ctx.fillStyle = fg; ctx.textAlign = "right"; ctx.textBaseline = "middle";
      for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) {
        const y = p.Y(v); ctx.strokeStyle = grid; ctx.beginPath(); ctx.moveTo(L0, y); ctx.lineTo(R0, y); ctx.stroke();
        ctx.fillText(p.unit === "m" && v >= 1000 ? (v / 1000).toFixed(v % 1000 ? 1 : 0) + "k" : String(Math.round(v * 10) / 10), L0 - 5, y);
      }
      ctx.save(); ctx.translate(12, (p.y0 + p.y1) / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = "center"; ctx.fillText(p.unit, 0, 0); ctx.restore();
    }
    const line = (p, k, colr, w) => {
      ctx.strokeStyle = colr; ctx.lineWidth = w; ctx.beginPath(); let on = false, prevT = null;
      for (const r of rows) {
        const v = r[ix[k]];
        if (v == null || (prevT != null && r[0] - prevT > 300)) { on = false; if (v == null) { prevT = r[0]; continue; } }
        const x = X(r[0]), y = p.Y(v);
        if (on) ctx.lineTo(x, y); else { ctx.moveTo(x, y); on = true; }
        prevT = r[0];
      }
      ctx.stroke();
    };
    line(pan[0], "wp", css("--s-pk"), 1); line(pan[0], "ws", css("--s-fl"), 1.6); line(pan[0], "sfmr", css("--s-sfmr"), 1.8);
    line(pan[1], "x", css("--s-slp"), 1.6); line(pan[2], "z", css("--s-alt"), 1.6);
    // time axis
    ctx.fillStyle = fg; ctx.textAlign = "center"; ctx.textBaseline = "top";
    const span = t1 - t0, st = span > 8 * 3600 ? 3600 : span > 3 * 3600 ? 1800 : span > 3600 ? 900 : 300;
    for (let t = Math.ceil(t0 / st) * st; t <= t1; t += st) {
      const x = X(t); ctx.strokeStyle = grid; ctx.beginPath(); ctx.moveTo(x, 6); ctx.lineTo(x, H - 22); ctx.stroke();
      if (t % (st * (W < 600 ? 2 : 1)) === 0) ctx.fillText(hm(t) + "Z", x, H - 18);
    }
    // markers: fixes, drops
    ctx.fillStyle = "#b3261e";
    for (const v of S.mis.vdms) { const x = X(v.t); ctx.fillRect(x - 1, 6, 2, H - 28); }
    ctx.fillStyle = fg;
    for (const d of S.mis.drops) {
      const t = d.rel_t || d.spl_t; if (!t) continue;
      const x = X(t), y = H - 22; ctx.beginPath(); ctx.moveTo(x, y - 7); ctx.lineTo(x - 4, y); ctx.lineTo(x + 4, y); ctx.closePath(); ctx.fill();
    }
    TS = { X, t0, t1, L0, R0, pan, W, H };
    if (S.hoverI >= 0) crosshair(S.hoverI);
  }
  // tick step for a panel px tall: about one label per 22 px, never closer than 13 px (11 px text)
  function niceUp(x) { const p = Math.pow(10, Math.floor(Math.log10(x))), f = x / p; return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * p; }
  function tickStep(span, px) {
    px = Math.abs(px); let st = niceStep(span / Math.max(3, Math.min(5, Math.floor(px / 22))));
    while (st / span * px < 13) st = niceUp(st * 1.01);
    return st;
  }
  function niceStep(x) { const p = Math.pow(10, Math.floor(Math.log10(x))), f = x / p; return (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * p; }
  function crosshair(i) {
    if (!TS) return;
    const r = S.rows[i], x = TS.X(r[0]);
    ctx.save(); ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(x, 6); ctx.lineTo(x, TS.H - 22); ctx.stroke(); ctx.restore();
  }
  function nearestRow(t) {
    let lo = 0, hi = S.rows.length - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (S.rows[m][0] < t) lo = m; else hi = m; }
    return Math.abs(S.rows[lo][0] - t) <= Math.abs(S.rows[hi][0] - t) ? lo : hi;
  }
  function hoverRow(i, fromMap) {
    S.hoverI = i;
    drawTS();
    if (R.legs.length) { R.hoverI = i; radDraw(); }
    lyr.hover.clearLayers();
    if (i < 0) { $("tsread").textContent = ""; return; }
    const r = S.rows[i], ix = S.col, f = (k, u, d = 0) => r[ix[k]] == null ? "—" : r[ix[k]].toFixed(d) + " " + u;
    $("tsread").innerHTML = `<b>${hms(r[0])}Z</b> ${ll(r[ix.la], r[ix.lo])} · FL wind ${r[ix.ws] != null ? dirTxt(r[ix.wd], r[ix.ws]) : "—"} (peak ${f("wp", "kt")}) · ` +
      `SFMR ${f("sfmr", "kt")}${r[ix.rr] != null ? `, rain ${r[ix.rr]} mm/h` : ""} · extrap. SLP ${f("x", "hPa", 1)} · ${f("p", "hPa", 1)} at ${f("z", "m")} · ` +
      `T ${f("t_", "°C", 1)} Td ${f("td", "°C", 1)}${r[ix.qc] && r[ix.qc] !== "00" ? ` · <span class="mu">QC flags ${r[ix.qc]}</span>` : ""}`;
    if (r[ix.la] != null) {
      L.circleMarker([r[ix.la], r[ix.lo]], { renderer: rend, radius: 7, color: dark() ? "#fff" : "#000", weight: 2.5, fill: false, interactive: false }).addTo(lyr.hover);
      if (!fromMap && !map.getBounds().contains([r[ix.la], r[ix.lo]])) map.panTo([r[ix.la], r[ix.lo]]);
    }
  }
  function goTime(t, pan) {
    if (!S.rows.length) return;
    const i = nearestRow(t); hoverRow(i, !pan);
    const r = S.rows[i]; if (pan && r[S.col.la] != null) map.setView([r[S.col.la], r[S.col.lo]], Math.max(map.getZoom(), 7));
  }
  function tsEvent(e) {
    if (!TS || !S.rows.length) return;
    const b = cv.getBoundingClientRect(), x = e.clientX - b.left;
    if (x < TS.L0 || x > TS.R0) return;
    const t = TS.t0 + (x - TS.L0) / (TS.R0 - TS.L0) * (TS.t1 - TS.t0);
    hoverRow(nearestRow(t), false);
  }
  cv.addEventListener("pointermove", tsEvent);
  cv.addEventListener("pointerdown", tsEvent);
  cv.addEventListener("pointerleave", () => { if (S.hoverI >= 0) { S.hoverI = -1; drawTS(); lyr.hover.clearLayers(); $("tsread").textContent = ""; } });
  // map -> series: the nearest ob within 14 px of the pointer
  map.on("mousemove", e => {
    if (!S.rows.length) return;
    const ix = S.col, p = e.containerPoint;
    let best = -1, bd = 14 * 14;
    for (let i = 0; i < S.rows.length; i++) {
      const r = S.rows[i]; if (r[ix.la] == null) continue;
      const q = map.latLngToContainerPoint([r[ix.la], r[ix.lo]]), d = (q.x - p.x) ** 2 + (q.y - p.y) ** 2;
      if (d < bd) { bd = d; best = i; }
    }
    if (best !== S.hoverI && (best >= 0 || S.hoverI >= 0)) hoverRow(best, true);
  });
  let rz = null;
  addEventListener("resize", () => { clearTimeout(rz); rz = setTimeout(drawTS, 120); });

  // ── dropsonde skew-T: the sounding explorer in an iframe, fed by postMessage ──
  const KT = 1.9438;
  function dropProfile(d) {
    // rows [p, z, T, Td, dir, kt, std] surface -> top; T/Td at wind-only levels interpolated in log p between T levels
    const rows = d.prof.filter(r => r[0] > 0).slice().sort((a, b) => b[0] - a[0]);
    const tl = rows.filter(r => r[2] != null);
    if (tl.length < 2) return null;
    const lp = p => Math.log(p);
    const interp = (p, k) => {
      for (let j = 1; j < tl.length; j++) {
        const a = tl[j - 1], b = tl[j];
        if (p <= a[0] && p >= b[0]) {
          if (a[k] == null || b[k] == null) return null;
          const f = (lp(a[0]) - lp(p)) / (lp(a[0]) - lp(b[0]));
          return a[k] + f * (b[k] - a[k]);
        }
      }
      return null;
    };
    const o = { P: [], H: [], T: [], D: [], U: [], V: [] };
    for (const r of rows) {
      const T = r[2] != null ? r[2] : interp(r[0], 2);
      if (T == null || r[1] == null) continue;
      let Td = r[2] != null ? r[3] : interp(r[0], 3);
      if (Td != null) Td = Math.min(Td, T);
      o.P.push(r[0] * 100); o.H.push(r[1]); o.T.push(T + 273.15); o.D.push(Td == null ? NaN : Td + 273.15);
      if (r[5] != null && r[4] != null) { const ws = r[5] / KT, wd = r[4] * Math.PI / 180; o.U.push(-ws * Math.sin(wd)); o.V.push(-ws * Math.cos(wd)); }
      else { o.U.push(NaN); o.V.push(NaN); }
    }
    return o.P.length >= 3 ? o : null;
  }
  const SND = { ready: false, pend: null };
  function openSkewT(i) {
    const ds = S.mis.drops, items = [];
    let idx = 0;
    ds.forEach((d, j) => {
      const prof = dropProfile(d); if (!prof) return;
      const t = d.spl_t || d.rel_t, iso = t ? new Date(t * 1000).toISOString() : "";
      const sp = d.spg || d.spl || d.rel;
      if (j === i) idx = items.length;
      items.push({ fxx: j, label: `OB ${d.obnum ?? j + 1}`, chip: t ? hm(t) : `#${j + 1}`, valid: iso ? iso.slice(0, 10) + " " + iso.slice(11, 16) : "",
        status: `${S.mis.aircraft} ${S.mis.mission} dropsonde OB ${d.obnum ?? ""} · ${t ? "splash " + hms(t) + "Z" : ""} · ${sp ? ll(sp.la, sp.lo) : ""} · ${prof.P.length} levels`,
        prof });
    });
    if (!items.length) { alert("This dropsonde has too few temperature levels for a skew-T."); return; }
    const d = ds[i], sp = d.spg || d.spl || d.rel;
    $("dock").hidden = false;
    $("dockT").textContent = `Dropsonde OB ${d.obnum ?? ""} · ${S.mis.aircraft} ${S.mis.mission}`;
    $("dockS").textContent = `${sp ? ll(sp.la, sp.lo) : ""}${d.sfc_p ? " · splash " + d.sfc_p + " hPa" : ""}`;
    if (!$("dockF").getAttribute("src")) $("dockF").src = "/skewt/?embed=model";
    send({ type: "mv-sounding", kind: "drop", model: "Dropsonde", stripLabel: "sondes", items, idx,
      title: `${S.mis.aircraft} drop OB ${d.obnum ?? ""}`, coords: sp ? ll(sp.la, sp.lo) : "",
      footer: "<b>Dropsonde</b> (TEMP DROP, NHC reconnaissance archive): mandatory, significant-temperature and 21212 wind levels; heights " +
        "integrated from the splash level and tied to the coded mandatory heights; temperature at wind-only levels interpolated between " +
        "significant levels. Analysis: <a href=\"https://github.com/keltonhalbert/SHARPlib\">SHARPlib</a>, as for the balloon soundings. " +
        "A dropsonde released at low level covers only the boundary layer, so parcel indices are partial." });
  }
  function send(msg) {
    if (SND.ready) { $("dockF").contentWindow.postMessage(msg, location.origin); return; }
    SND.pend = msg;
  }
  addEventListener("message", e => {
    if (e.origin !== location.origin || e.source !== $("dockF").contentWindow) return;
    const d = e.data || {};
    if (d.type === "mv-ready") { SND.ready = true; if (SND.pend) { send(SND.pend); SND.pend = null; } }
    else if (d.type === "mv-close") closeDock();
    else if (d.type === "mv-hour" && S.mis) {
      const dd = S.mis.drops[d.fxx]; const sp = dd && (dd.spg || dd.spl || dd.rel);
      if (dd) { $("dockT").textContent = `Dropsonde OB ${dd.obnum ?? ""} · ${S.mis.aircraft} ${S.mis.mission}`;
                $("dockS").textContent = `${sp ? ll(sp.la, sp.lo) : ""}${dd.sfc_p ? " · splash " + dd.sfc_p + " hPa" : ""}`; }
    }
  });
  function closeDock() { $("dock").hidden = true; }
  $("dockX").onclick = closeDock;
  document.addEventListener("keydown", e => { if (e.key === "Escape" && !$("dock").hidden) { closeDock(); e.preventDefault(); } });

  // ── tail Doppler radar (NOAA AOML HRD real-time graphics, mirrored to the tdr-data branch) ──
  const TDRC = "#0f8b8d";
  const T = { idx: null, idxAt: 0, doc: null, prod: "sw", ti: -1, h: "1", v: {} };
  try { const q = JSON.parse(localStorage.getItem("recon.tdr") || "{}"); if (q.prod) T.prod = q.prod; if (q.h) T.h = q.h; } catch (e) { /* storage blocked */ }
  const tdrSave = () => { try { localStorage.setItem("recon.tdr", JSON.stringify({ prod: T.prod, h: T.h })); } catch (e) { /* blocked */ } };
  const HGTS = ["1", "2", "3", "5", "8", "10"];
  const rng = (a, b) => `${hm(a)}–${hm(b)}Z`;
  const PRODS = {
    sw: { lab: "Swath", times: d => d.analyses.filter(a => Object.keys(a.sw).length).map(a => ({ k: a.t, lab: hm(a.t) })) },
    comp: { lab: "Composite", times: d => d.composites.map(c => ({ k: c.t1, lab: rng(c.t0, c.t1) })),
            vars: [["dbz", "Reflectivity and 2 km wind"], ["nhc", "0.5 km and flight-level wind"]] },
    zoom: { lab: "Inner core", one: d => d.latest.zoom },
    tilt: { lab: "Vortex tilt", one: d => d.latest.tilt },
    rz: { lab: "Azimuthal means", one: d => d.latest.rz, vars: [["vt", "Tangential wind"], ["ur", "Radial wind"], ["dbz", "Reflectivity"]] },
    prof: { lab: "Profiles", times: d => d.analyses.filter(a => a.prof).map(a => ({ k: a.t, lab: hm(a.t) })),
            vars: [["ws", "Wind speed"], ["dbz", "Reflectivity"], ["vr_rel", "Radial wind"], ["PBL_rel", "Boundary layer"]] },
    ctr: { lab: "Centres", one: d => d.latest.ctr || (d.centres.length ? {} : null) },
  };
  const has = (d, k) => PRODS[k].times ? PRODS[k].times(d).length > 0 : !!PRODS[k].one(d);

  async function tdrIndex(force) {
    const air = Object.values(S.man.missions).some(isAir);
    if (T.idx && !force && Date.now() - T.idxAt < (air ? 90e3 : 600e3)) return T.idx;
    try { T.idx = await dataFetch("index.json", Math.floor(Date.now() / 60000), true, TBR) || { missions: {} }; T.idxAt = Date.now(); }
    catch (e) { /* mirror unreachable: keep what we had */ }
    return T.idx;
  }
  async function tdrLoad(key, refresh) {
    const m = S.man.missions[key];
    if (!m) return;
    const idx = await tdrIndex(refresh);
    if (S.key !== key) return;
    satUI(key, idx);
    if (!/^NOAA/.test(m.aircraft || "")) { T.doc = null; lyr.tdr.clearLayers(); $("tdr").hidden = true; return; }
    const e = idx && idx.missions && idx.missions[key];
    if (!e) {
      T.doc = null; lyr.tdr.clearLayers();
      $("tdr").hidden = false;
      $("tdrSub").innerHTML = isAir(m) ? "No tail Doppler radar analyses from this flight in the HRD real-time archive yet; each appears about 70 minutes after its centre fix."
        : "No tail Doppler radar graphics from this flight in the HRD real-time archive.";
      for (const id of ["tdrProd", "tdrTime", "tdrHgt", "tdrVar"]) $(id).innerHTML = "";
      $("tdrTimeL").hidden = $("tdrHgtL").hidden = $("tdrVarL").hidden = true;
      $("tdrProd").parentElement.hidden = true;
      $("tdrFrame").hidden = true; $("tdrCap").innerHTML = ""; $("tdrCtr").innerHTML = "";
      return;
    }
    if (refresh && T.doc && T.doc.key === key && T.docLast === e.last && T.docN === e.n_img) return;
    let d;
    try { d = await dataFetch(e.file, e.last + "_" + e.n_img + "_" + e.n_ctr, isAir(m), TBR); } catch (err) { return; }
    if (S.key !== key || !d) return;
    const first = !T.doc || T.doc.key !== key;
    T.doc = d; T.docLast = e.last; T.docN = e.n_img;
    if (first) T.ti = -1;
    $("tdr").hidden = false; $("tdrProd").parentElement.hidden = false; $("tdrFrame").hidden = false;
    const nC = d.centres.filter(c => c.c).length, nBad = d.centres.length - nC;
    $("tdrSub").innerHTML = `HRD flight <b>${esc(d.id)}</b> (${esc(d.tail || d.aircraft)}), matched to ${esc(m.aircraft)} ${esc(m.mission)} by ${esc(d.how)}. ` +
      `${d.analyses.length} analys${d.analyses.length === 1 ? "is" : "es"}, ${d.composites.length} composite${d.composites.length === 1 ? "" : "s"}` +
      `${d.centres.length ? `, ${nC} TDR centre${nC === 1 ? "" : "s"} on the map${nBad ? ` (${nBad} not drawn: no centre found, or failed the check)` : ""}` : ""}. ` +
      `<a href="${esc(d.src)}">Originals at AOML</a>.` +
      (d.storm ? ` <a href="/circulation.html#tcshear/${esc(d.storm)}/fc/0/tilt">This storm's tilt against the model shear</a>.` : "");
    tdrCentres();
    legend();
    tdrUI();
    if (first && d.centres.some(c => c.c)) { radBuild(); srBuild(); }      // the TDR centres refine the centre track
  }
  function tdrCentres() {
    lyr.tdr.clearLayers();
    const d = T.doc; if (!d) return;
    const cur = tdrTime();
    for (const c of d.centres) {
      if (!c.c) continue;
      const p = c.c["2"] || c.c["05"] || c.c["3"], p6 = c.c["6"];
      if (!p) continue;
      const on = cur && Math.abs(cur - c.t) < 60;
      if (p6) L.polyline([p, p6], { renderer: rend, color: TDRC, weight: on ? 3 : 2, interactive: false }).addTo(lyr.tdr);
      if (p6) L.circleMarker(p6, { renderer: rend, radius: 2.5, color: TDRC, weight: 1.5, fillColor: TDRC, fillOpacity: 1, interactive: false }).addTo(lyr.tdr);
      const sz = on ? 18 : 13;
      const lv = [["05", "0.5"], ["2", "2"], ["3", "3"], ["6", "6"]].filter(([k]) => c.c[k]).map(([k, l]) => `${l} km ${ll(c.c[k][0], c.c[k][1])}`).join("<br>");
      L.marker(p, { zIndexOffset: 400, icon: L.divIcon({ className: "", iconSize: [sz, sz], iconAnchor: [sz / 2, sz / 2],
        html: `<svg width="${sz}" height="${sz}" viewBox="0 0 20 20"><path d="M10 1 L19 10 L10 19 L1 10 Z" fill="${on ? TDRC : "#fff"}" stroke="${TDRC}" stroke-width="2.5"/></svg>` }) })
        .bindTooltip(`<b>TDR centre ${hm(c.t)}Z</b><br>${lv}${c.tilt ? `<br>2–6 km tilt ${c.tilt[0]} km toward ${c.tilt[1]}°` : ""}` +
          `${c.ref ? `<br><span style="color:#5d5a54">${c.ref[1]} km from the ${esc(c.ref[0])}</span>` : ""}<br><i>click: show this analysis</i>`)
        .on("click", () => { const i = PRODS.sw.times(d).findIndex(x => Math.abs(x.k - c.t) < 60); if (i >= 0) { T.prod = "sw"; T.ti = i; tdrUI(); $("tdr").scrollIntoView({ behavior: "smooth", block: "start" }); } })
        .addTo(lyr.tdr);
    }
  }
  function tdrTime() {
    const d = T.doc, P = PRODS[T.prod]; if (!d || !P || !P.times) return null;
    const ts = P.times(d); if (!ts.length) return null;
    const i = T.ti < 0 || T.ti >= ts.length ? ts.length - 1 : T.ti;
    return T.prod === "comp" ? null : ts[i].k;
  }
  function segHTML(items, on) {
    return items.map(([k, lab]) => `<button type="button" data-k="${esc(k)}"${String(k) === String(on) ? ' class="on" aria-pressed="true"' : ' aria-pressed="false"'}>${esc(lab)}</button>`).join("");
  }
  function tdrUI() {
    const d = T.doc; if (!d) return;
    const avail = Object.keys(PRODS).filter(k => has(d, k));
    if (!avail.includes(T.prod)) T.prod = avail[0] || "sw";
    $("tdrProd").innerHTML = segHTML(avail.map(k => [k, PRODS[k].lab]), T.prod);
    const P = PRODS[T.prod];
    // time
    const ts = P.times ? P.times(d) : [];
    if (T.ti >= ts.length) T.ti = -1;
    const ti = T.ti < 0 ? ts.length - 1 : T.ti;
    $("tdrTimeL").hidden = !ts.length;
    $("tdrTimeL").firstChild.textContent = T.prod === "comp" ? "Passes merged " : "Analysis (centre fix) ";
    $("tdrTime").innerHTML = segHTML(ts.map((t, i) => [i, t.lab]), ti);
    // height (swath)
    const a = T.prod === "sw" && ts.length ? d.analyses.find(x => x.t === ts[ti].k) : null;
    const hs = a ? HGTS.filter(h => a.sw[h]) : [];
    if (a && !hs.includes(T.h)) T.h = hs[0];
    $("tdrHgtL").hidden = !a;
    $("tdrHgt").innerHTML = segHTML(hs.map(h => [h, h + " km"]), T.h);
    // field
    let vars = P.vars || [];
    let src = null, cap = "", sq = false;
    if (T.prod === "sw" && a) {
      src = a.sw[T.h];
      cap = `<b>Swath, ${hm(a.t)}Z analysis, ${T.h} km</b>: reflectivity (left) and wind speed in knots (right) from this one pass, ` +
        `earth-relative, with the analysis' centre fix in the corner. ${T.h === "1" ? "1 km is the lowest level shown and the nearest to the surface wind, which is weaker than the wind at 1 km. " : ""}` +
        `Gaps are where the radar saw no scatterers.`;
    } else if (T.prod === "comp" && ts.length) {
      const c = d.composites[ti];
      vars = vars.filter(([k]) => c[k]);
      const v = vars.some(([k]) => k === T.v.comp) ? T.v.comp : (vars[0] || [])[0];
      T.v.comp = v; src = c[v];
      cap = `<b>Composite of the passes ${rng(c.t0, c.t1)}</b>: ${v === "nhc" ? "wind speed at 0.5 km (left) and at 2.5 km, about flight level (right)" : "reflectivity (left) and wind speed with streamlines at 2 km (right)"}, ` +
        `around the mean centre. HRD: composites may not be storm centred, and in storms weaker than hurricanes may not be meaningful; check the swaths first.`;
    } else if (T.prod === "rz") {
      const r = d.latest.rz, v = vars.some(([k]) => k === T.v.rz && r[k]) ? T.v.rz : vars.find(([k]) => r[k])[0];
      vars = vars.filter(([k]) => r[k]); T.v.rz = v; src = r[v];
      cap = `<b>Azimuthal means, passes ${rng(r.t0, r.t1)}</b> (latest composite): ${{ vt: "tangential wind", ur: "radial wind", dbz: "reflectivity" }[v]} against radius and height, ` +
        `2 km by 0.5 km grid, around the composite centre (same caveat as the composites).`;
    } else if (T.prod === "zoom" || T.prod === "tilt") {
      const z = d.latest[T.prod]; src = z.img;
      cap = T.prod === "zoom" ? `<b>Inner 80 km, passes ${rng(z.t0, z.t1)}</b> (latest composite): reflectivity and wind with streamlines.`
        : `<b>Vorticity and vortex tilt, passes ${rng(z.t0, z.t1)}</b> (latest composite), storm-relative flow; the arrow is the tilt of the vortex centre with height (1 km steps).`;
    } else if (T.prod === "prof" && ts.length) {
      const pa = d.analyses.find(x => x.t === ts[ti].k);
      vars = vars.filter(([k]) => pa.prof[k]);
      const v = vars.some(([k]) => k === T.v.prof) ? T.v.prof : vars[0][0];
      T.v.prof = v; src = pa.prof[v];
      cap = `<b>Inbound and outbound profiles, ${hm(pa.t)}Z analysis</b>${pa.azi ? ` (azimuths ${pa.azi[0]}° and ${pa.azi[1]}°)` : ""}: 150 m vertical spacing.`;
    } else if (T.prod === "ctr") {
      src = d.latest.ctr ? d.latest.ctr.img : null; sq = true;
      cap = `<b>TDR centres</b> at 2 km (L) and 6 km (M) for each analysis, from the storm-relative flow (HRD's track map). The same centres are on the map above.`;
    }
    $("tdrVarL").hidden = vars.length < 2;
    $("tdrVar").innerHTML = segHTML(vars, T.v[T.prod]);
    tdrImg(src, sq);
    $("tdrCap").innerHTML = cap + ` <span class="mu">Real time, automated quality control (NOAA AOML HRD).</span>`;
    $("tdrCtr").innerHTML = T.prod === "ctr" ? tdrCtrTable(d) : "";
    tdrCentres();
  }
  function tdrCtrTable(d) {
    if (!d.centres.length) return `<p class="empty">No TDR centre files for this flight.</p>`;
    const f = p => p ? ll(p[0], p[1]) : "—";
    return `<table><thead><tr><th>Analysis</th><th>0.5 km</th><th>2 km</th><th>6 km</th><th>2–6 km tilt</th><th>Check</th></tr></thead><tbody>` +
      d.centres.map(c => c.c ? `<tr><td>${hm(c.t)}Z</td><td>${f(c.c["05"])}</td><td>${f(c.c["2"])}</td><td>${f(c.c["6"])}</td>` +
        `<td>${c.tilt ? `${c.tilt[0]} km to ${c.tilt[1]}°` : "—"}</td><td class="mu">${c.ref ? `${c.ref[1]} km from the ${esc(c.ref[0])}` : "unchecked (no fix or track)"}</td></tr>`
        : `<tr><td>${hm(c.t)}Z</td><td colspan="4" class="mu">not drawn</td><td class="mu">${esc(c.bad)}</td></tr>`).join("") + "</tbody></table>";
  }
  function tdrImg(path, sq) {
    const fr = $("tdrFrame");
    if (!path) { fr.innerHTML = `<p class="ph">No image for this choice.</p>`; return; }
    const v = T.prod === "ctr" && T.doc.latest.ctr ? T.doc.latest.ctr.v : "";
    const urls = [URLS.cdn(path, v, TBR)[0], URLS.raw(path, v, TBR)[0], URLS.jsd(path, null, TBR)[0]];
    let img = fr.querySelector("img"), a = fr.querySelector("a");
    if (!img) { fr.innerHTML = ""; a = document.createElement("a"); a.target = "_blank"; a.rel = "noopener"; a.title = "Open the full-size image";
                img = document.createElement("img"); img.decoding = "async"; a.appendChild(img); fr.appendChild(a); }
    img.onload = () => { a.href = img.src; };
    img.className = sq ? "sq" : "";
    img.alt = $("tdrCap").textContent || "Tail Doppler radar graphic";
    let k = 0;
    img.onerror = () => { if (++k < urls.length) img.src = urls[k]; else fr.innerHTML = `<p class="ph">The image could not be loaded. Try again shortly.</p>`; };
    img.src = urls[0];
  }
  // GOES-East band-13 cut-outs at the centre passes (goes_ir.py), laid under the track on demand
  const SAT = { key: null, t: null, layer: null };
  const SATCOL = [["≤ −80", "#ffffff"], ["−75", "#b428c8"], ["−70", "#c8141e"], ["−65", "#f06e14"], ["−60", "#f5c81e"], ["−50", "#28aa3c"], ["−40", "#286ed2"], ["−30", "#5ac8e6"]];
  function satUI(key, idx) {
    const list = (idx && idx.sat && idx.sat[key]) || [];
    if (SAT.key !== key) { SAT.key = key; SAT.t = null; if (SAT.layer) { map.removeLayer(SAT.layer); SAT.layer = null; } }
    $("satBar").hidden = !list.length;
    if (!list.length) return;
    $("satSel").innerHTML = segHTML([["off", "Off"], ...list.map(x => [String(x[0]), hm(x[0]) + "Z pass"])], SAT.t == null ? "off" : String(SAT.t));
    satInfo(list);
  }
  function satInfo(list) {
    const x = list.find(y => y[0] === SAT.t);
    $("satInfo").innerHTML = x ? `${esc(x[2])} band 13 (10.3 µm), image ${hms(x[1])}Z · ` +
      SATCOL.map(([l, c]) => `<span class="sw"><i style="background:${c}"></i>${l}</span>`).join(" ") + ` <span class="sw"><i style="background:linear-gradient(90deg,#d7d7d7,#282828)"></i>−30 to +30 °C</span> · NOAA GOES-East (AWS open data)`
      : "Cloud-top temperature at each centre pass, from NOAA GOES-East.";
  }
  $("satSel").addEventListener("click", e => {
    const b = e.target.closest("button[data-k]"); if (!b || !T.idx) return;
    const list = (T.idx.sat && T.idx.sat[SAT.key]) || [];
    if (SAT.layer) { map.removeLayer(SAT.layer); SAT.layer = null; }
    SAT.t = b.dataset.k === "off" ? null : +b.dataset.k;
    const x = list.find(y => y[0] === SAT.t);
    if (x) {
      const urls = [URLS.cdn(x[7], null, TBR)[0], URLS.raw(x[7], null, TBR)[0], URLS.jsd(x[7], null, TBR)[0]];
      SAT.layer = L.imageOverlay(urls[0], [[x[3], x[4]], [x[5], x[6]]], { pane: "sat", opacity: 0.85, interactive: false, alt: "GOES infrared" }).addTo(map);
      SAT.layer.once("error", () => SAT.layer && SAT.layer.setUrl(urls[1]));
    }
    $("satSel").querySelectorAll("button").forEach(y => { const on = y === b; y.classList.toggle("on", on); y.setAttribute("aria-pressed", on); });
    satInfo(list);
  });
  const segClick = (id, fn) => $(id).addEventListener("click", e => { const b = e.target.closest("button[data-k]"); if (b) { fn(b.dataset.k); tdrUI(); } });
  segClick("tdrProd", k => { T.prod = k; T.ti = -1; tdrSave(); });
  segClick("tdrTime", k => { T.ti = +k; });
  segClick("tdrHgt", k => { T.h = k; tdrSave(); });
  segClick("tdrVar", k => { T.v[T.prod] = k; });

  // ── centre track and passes (a port of recon_mirror.centre_fn / legs / secondary_max: keep the two in step) ──
  const NMI = 1.852, PASS_R = 20, LEG_MAX_R = 150, TURN_NM = 5, SEC_OUT_NM = 10, SEC_KT = 5, SEC_SPAN_NM = 20, SEC_FRAC = 0.5, SEC_MIN_VMAX = 64;
  const rad = Math.PI / 180;
  function gkm(a, b, c, d) {
    const p1 = a * rad, p2 = c * rad, dp = p2 - p1, dl = (d - b) * rad;
    const h = Math.sin(dp / 2) ** 2 + Math.cos(p1) * Math.cos(p2) * Math.sin(dl / 2) ** 2;
    return 2 * 6371 * Math.asin(Math.min(1, Math.sqrt(h)));
  }
  function btAt(bt, t) {
    if (!bt || !bt.length || t < bt[0][0] - 43200 || t > bt[bt.length - 1][0] + 43200) return null;
    if (t <= bt[0][0]) return [bt[0][1], bt[0][2]];
    if (t >= bt[bt.length - 1][0]) return [bt[bt.length - 1][1], bt[bt.length - 1][2]];
    for (let j = 1; j < bt.length; j++) if (bt[j][0] >= t) {
      const a = bt[j - 1], b = bt[j], f = (t - a[0]) / Math.max(1, b[0] - a[0]);
      return [a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2])];
    }
    return null;
  }
  function btMotion(bt, t) {             // deg/s lat, lon from the best-track segment around t
    if (!bt || bt.length < 2) return null;
    let a = t <= bt[0][0] ? bt[0] : bt[bt.length - 2], b = t <= bt[0][0] ? bt[1] : bt[bt.length - 1];
    for (let j = 1; j < bt.length; j++) if (bt[j - 1][0] <= t && t <= bt[j][0]) { a = bt[j - 1]; b = bt[j]; break; }
    const dt = Math.max(1, b[0] - a[0]);
    return [(b[1] - a[1]) / dt, (b[2] - a[2]) / dt];
  }
  // fixes: [{t, la, lo}] sorted. Linear between fixes <= 6 h apart; else the nearest fix (<= 4 h) moved with the best-track
  // motion; else the best track (+-12 h).
  function centreFn(fixes, bt) {
    return t => {
      for (let j = 1; j < fixes.length; j++) {
        const a = fixes[j - 1], b = fixes[j];
        if (a.t <= t && t <= b.t && b.t - a.t <= 21600) { const w = (t - a.t) / Math.max(1, b.t - a.t); return [a.la + w * (b.la - a.la), a.lo + w * (b.lo - a.lo)]; }
      }
      if (fixes.length) {
        let n = fixes[0];
        for (const f of fixes) if (Math.abs(f.t - t) < Math.abs(n.t - t)) n = f;
        if (Math.abs(n.t - t) <= 14400) {
          const mv = btMotion(bt, t);
          if (mv) return [n.la + mv[0] * (t - n.t), n.lo + mv[1] * (t - n.t)];
          if (Math.abs(n.t - t) <= 1800) return [n.la, n.lo];
        }
      }
      return btAt(bt, t);
    };
  }
  function smoothProfile(pts) {
    if (!pts.length) return [];
    const bins = new Map();
    for (const [r, v] of pts) { const k = Math.floor(r); if (!bins.has(k)) bins.set(k, []); bins.get(k).push(v); }
    const ks = [...bins.keys()], lo = Math.min(...ks), hi = Math.max(...ks), raw = [];
    for (let k = lo; k <= hi; k++) raw.push([k + 0.5, bins.has(k) ? bins.get(k).reduce((a, b) => a + b, 0) / bins.get(k).length : null]);
    const out = [];
    raw.forEach(([r, v], j) => {
      const w = raw.slice(Math.max(0, j - 2), j + 3).map(x => x[1]).filter(x => x != null);
      if (w.length >= 3 && v != null) out.push([r, w.reduce((a, b) => a + b, 0) / w.length]);
    });
    return out;
  }
  function secondaryMax(pts, rmw, vmax) {
    if (vmax < SEC_MIN_VMAX) return null;
    const sp = smoothProfile(pts);
    let best = null;
    for (const [r, v] of sp) {
      if (r < rmw + SEC_OUT_NM || sp.some(([r2, v2]) => Math.abs(r2 - r) <= 5 && v2 > v)) continue;
      const inner = sp.filter(([r2]) => rmw < r2 && r2 < r).map(x => x[1]), outer = sp.filter(([r2]) => r < r2 && r2 <= r + SEC_SPAN_NM).map(x => x[1]);
      if (inner.length < 3 || outer.length < 5) continue;
      const dip = v - Math.min(...inner);
      if (v >= SEC_FRAC * vmax && dip >= SEC_KT && v - Math.min(...outer) >= SEC_KT && (!best || v > best[1])) best = [r, v, dip];
    }
    return best;
  }
  // -> {legs: [{dir, idx: [row indices], tc, rmin, rmax, ws/sfmr/wp: [v, r, t, i], sec}], rr: radius (n mi) per row}
  function computeLegs(rows, ix, cf) {
    const rr = rows.map(r => { if (r[ix.la] == null) return null; const c = cf(r[0]); return c ? gkm(r[ix.la], r[ix.lo], c[0], c[1]) / NMI : null; });
    const good = []; rr.forEach((v, i) => { if (v != null) good.push(i); });
    const used = new Set(), out = [];
    good.forEach((i, gi) => {
      if (rr[i] > PASS_R || used.has(i)) return;
      const win = good.filter(j => Math.abs(rows[j][0] - rows[i][0]) <= 600);
      if (rr[i] > Math.min(...win.map(j => rr[j]))) return;
      win.forEach(j => used.add(j));
      for (const d of [-1, 1]) {
        const seq = [i]; let rmax = rr[i], last = i, k = gi + d;
        while (k >= 0 && k < good.length) {
          const j = good[k];
          if (Math.abs(rows[j][0] - rows[last][0]) > 300 || rr[j] > LEG_MAX_R || rr[j] < rmax - TURN_NM) break;
          rmax = Math.max(rmax, rr[j]); seq.push(j); last = j; k += d;
        }
        if (rmax - rr[i] < 15 || seq.length < 6) continue;
        seq.sort((a, b) => a - b);
        const leg = { dir: d, idx: seq, tc: rows[i][0], rmin: rr[i], rmax };
        for (const col of ["ws", "sfmr", "wp"]) {
          let b = null;
          for (const j of seq) { const v = rows[j][ix[col]]; if (v != null && (!b || v > b[0])) b = [v, rr[j], rows[j][0], j]; }
          if (b) leg[col] = b;
        }
        if (leg.ws) leg.sec = secondaryMax(seq.filter(j => rows[j][ix.ws] != null).map(j => [rr[j], rows[j][ix.ws]]).sort((a, b) => a[0] - b[0]), leg.ws[1], leg.ws[0]);
        out.push(leg);
      }
    });
    out.sort((a, b) => a.tc - b.tc || a.dir - b.dir);
    return { legs: out, rr };
  }
  // centre fixes for the open storm: every vortex fix (all missions) plus this flight's TDR centres (2 km)
  function centreFixes() {
    const fx = stormFixes(S.sid).map(f => ({ t: f.t, la: f.la, lo: f.lo }));
    if (T.doc && T.doc.key === S.key) for (const c of T.doc.centres) { const p = c.c && (c.c["2"] || c.c["05"]); if (p) fx.push({ t: c.t, la: p[0], lo: p[1], tdr: 1 }); }
    return fx.sort((a, b) => a.t - b.t);
  }

  // ── canvas helpers ───────────────────────────────────────────────────────
  function cvSetup(cv) {
    const dpr = window.devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight;
    if (!W) return null;
    cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
    const c = cv.getContext("2d"); c.setTransform(dpr, 0, 0, dpr, 0, 0); c.clearRect(0, 0, W, H);
    c.font = "11px 'Source Sans 3', sans-serif"; c.lineWidth = 1;
    return { c, W, H };
  }
  function axisY(c, Y, lo, hi, x0, x1, unit, fmt) {
    const step = tickStep(hi - lo, Y(hi) - Y(lo));
    c.fillStyle = css("--axis"); c.textAlign = "right"; c.textBaseline = "middle";
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) {
      const y = Y(v); c.strokeStyle = css("--grid"); c.beginPath(); c.moveTo(x0, y); c.lineTo(x1, y); c.stroke();
      c.fillText(fmt ? fmt(v) : String(Math.round(v * 10) / 10), x0 - 5, y);
    }
    if (unit) { const yc = (Y(lo) + Y(hi)) / 2; c.save(); c.translate(11, yc); c.rotate(-Math.PI / 2); c.textAlign = "center"; c.fillText(unit, 0, 0); c.restore(); }
  }
  const LEGC = ["#3b6fd4", "#e0442c", "#2fa84f", "#9b4fd6", "#d98a12", "#1aa3a3", "#c2378f", "#6b7d1f", "#7a5230", "#5b8fd9"];
  const legCol = i => LEGC[i % LEGC.length];

  // ── radial wind profiles ─────────────────────────────────────────────────
  const R = { v: "ws", off: new Set(), legs: [], rr: [], pts: [], hoverI: -1 };
  const RVARS = [["ws", "Flight level (30 s)"], ["sfmr", "SFMR surface"], ["wp", "Peak 10-s flight level"]];
  function radBuild() {
    if (!S.mis || !S.rows.length) { $("rad").hidden = true; R.legs = []; return; }
    const cf = centreFn(centreFixes(), S.storm && S.storm.btk);
    const r = computeLegs(S.rows, S.col, cf);
    R.legs = r.legs; R.rr = r.rr; R.off = new Set();
    $("rad").hidden = false;
    const nT = centreFixes().filter(f => f.tdr).length;
    $("radSub").innerHTML = R.legs.length
      ? `${R.legs.length} legs into or out of the centre on this flight, cut at ${LEG_MAX_R} n mi or where the aircraft turns back. Radius from the centre interpolated in time between the vortex fixes${nT ? " and the TDR centres" : ""} (best-track motion beyond them). Hover to find a point on the map and along the flight.`
      : `No centre passes: the flight never came within ${PASS_R} n mi of the interpolated centre${S.man.missions[S.key].how && S.man.missions[S.key].how.startsWith("nearest") ? " (a surveillance flight)" : ""}.`;
    $("radVar").innerHTML = segHTML(RVARS.filter(([k]) => R.legs.some(g => g[k])), R.v);
    if (!R.legs.some(g => g[R.v])) R.v = "ws";
    $("radVar").innerHTML = segHTML(RVARS.filter(([k]) => R.legs.some(g => g[k])), R.v);
    $("radLegs").innerHTML = R.legs.map((g, i) => `<button type="button" data-i="${i}" aria-pressed="true"><i style="background:${legCol(i)}${g.dir > 0 ? ";background:repeating-linear-gradient(90deg," + legCol(i) + " 0 4px,transparent 4px 7px)" : ""}"></i>${hm(g.tc)}Z ${g.dir < 0 ? "in" : "out"}</button>`).join("");
    $("radTbl").innerHTML = R.legs.length ? `<table><thead><tr><th>Centre pass</th><th>Leg</th><th>To (n mi)</th><th>Max FL wind (RMW)</th><th>Max SFMR (radius)</th><th>Secondary FL maximum</th></tr></thead><tbody>` +
      R.legs.map((g, i) => `<tr class="click" data-i="${i}"><td><i style="display:inline-block;width:0.8em;height:0.8em;border-radius:2px;background:${legCol(i)};margin-right:0.35em"></i>${hm(g.tc)}Z</td><td>${g.dir < 0 ? "inbound" : "outbound"}</td><td>${Math.round(g.rmax)}</td>` +
        `<td>${g.ws ? `${g.ws[0]} kt (${g.ws[1].toFixed(0)} n mi)` : "—"}</td><td>${g.sfmr ? `${g.sfmr[0]} kt (${g.sfmr[1].toFixed(0)} n mi)` : "—"}</td>` +
        `<td${g.sec ? ' class="flag"' : ""}>${g.sec ? `${Math.round(g.sec[1])} kt at ${g.sec[0].toFixed(0)} n mi (dip ${g.sec[2].toFixed(0)} kt inside)` : `<span class="mu">${g.ws && g.ws[0] < SEC_MIN_VMAX ? `not looked for (leg below ${SEC_MIN_VMAX} kt)` : "none"}</span>`}</td></tr>`).join("") + "</tbody></table>" : "";
    $("radTbl").querySelectorAll("tr[data-i]").forEach(tr => tr.onclick = () => { const g = R.legs[+tr.dataset.i]; if (g.ws) goTime(g.ws[2], true); });
    $("radNote").innerHTML = `<b>Radius of maximum wind</b> (circle): where the leg's highest 30-s flight-level wind (or SFMR, when shown) was observed. ` +
      `<b>Secondary wind maximum</b> (triangle, flight-level wind only): on the leg's profile averaged into 1 n mi bins and smoothed over 5 n mi, the highest point within ±5 n mi that lies at least ${SEC_OUT_NM} n mi outside the RMW, ` +
      `is at least ${SEC_KT} kt above the lowest wind between it and the RMW and above the lowest wind in the ${SEC_SPAN_NM} n mi outside it (which must be sampled), and is at least half the leg's maximum ` +
      `(so outer rainbands do not count), on legs whose maximum is ${SEC_MIN_VMAX} kt or more. A flag, not a diagnosis of an eyewall replacement: it depends on where the aircraft flew.`;
    radDraw();
  }
  function radDraw() {
    const cv = $("radCv"); if ($("rad").hidden) return;
    const g0 = cvSetup(cv); if (!g0) return;
    const { c, W, H } = g0, ix = S.col, v = R.v;
    const vis = R.legs.map((g, i) => [g, i]).filter(([g, i]) => !R.off.has(i) && g[v]);
    const L0 = 46, R0 = W - 12, T0 = 8, B0 = H - 26;
    let rmax = 20, vmax = 20;
    for (const [g] of vis) { rmax = Math.max(rmax, g.rmax); vmax = Math.max(vmax, g[v][0]); }
    rmax = Math.min(LEG_MAX_R, Math.ceil(rmax / 10) * 10); vmax = Math.ceil((vmax + 8) / 10) * 10;
    const X = r => L0 + r / rmax * (R0 - L0), Y = w => B0 - w / vmax * (B0 - T0);
    axisY(c, Y, 0, vmax, L0, R0, "kt");
    c.textAlign = "center"; c.textBaseline = "top";
    for (let r = 0; r <= rmax; r += rmax > 80 ? 20 : 10) { c.strokeStyle = css("--grid"); c.beginPath(); c.moveTo(X(r), T0); c.lineTo(X(r), B0); c.stroke(); c.fillStyle = css("--axis"); c.fillText(String(r), X(r), B0 + 4); }
    c.fillText("radius from the centre (n mi)", (L0 + R0) / 2, B0 + 14);
    R.pts = [];
    for (const [g, i] of vis) {
      const col = legCol(i);
      c.strokeStyle = col; c.lineWidth = 1.7; c.setLineDash(g.dir > 0 ? [5, 3] : []);
      c.beginPath(); let on = false;
      const seq = g.idx.slice().sort((a, b) => R.rr[a] - R.rr[b]);
      for (const j of seq) {
        const w = S.rows[j][ix[v]];
        if (w == null) { on = false; continue; }
        const x = X(R.rr[j]), y = Y(w);
        if (on) c.lineTo(x, y); else { c.moveTo(x, y); on = true; }
        R.pts.push([x, y, j, i]);
      }
      c.stroke(); c.setLineDash([]);
      const m = g[v];
      c.fillStyle = col; c.beginPath(); c.arc(X(m[1]), Y(m[0]), 4.5, 0, 2 * Math.PI); c.fill();
      c.strokeStyle = css("--panel"); c.lineWidth = 1.5; c.stroke();
      if (v === "ws" && g.sec) {
        const x = X(g.sec[0]), y = Y(g.sec[1]) - 7;
        c.fillStyle = col; c.beginPath(); c.moveTo(x, y + 6); c.lineTo(x - 6, y - 5); c.lineTo(x + 6, y - 5); c.closePath(); c.fill(); c.stroke();
      }
    }
    if (!vis.length) { c.fillStyle = css("--muted"); c.font = "14px sans-serif"; c.textAlign = "left"; c.fillText(R.legs.length ? "No leg has this wind." : "No legs.", L0 + 10, T0 + 20); }
    const hi = R.hoverI;
    if (hi >= 0 && R.rr[hi] != null && S.rows[hi][ix[v]] != null && vis.some(([g]) => g.idx.includes(hi))) {
      c.strokeStyle = css("--ink"); c.lineWidth = 2; c.beginPath(); c.arc(X(R.rr[hi]), Y(S.rows[hi][ix[v]]), 6, 0, 2 * Math.PI); c.stroke();
    }
    R.geo = { X, Y };
  }
  function radPointer(e) {
    if (!R.pts.length) return;
    const b = $("radCv").getBoundingClientRect(), x = e.clientX - b.left, y = e.clientY - b.top;
    let best = null, bd = 14 * 14;
    for (const p of R.pts) { const d = (p[0] - x) ** 2 + (p[1] - y) ** 2; if (d < bd) { bd = d; best = p; } }
    if (!best) return;
    const g = R.legs[best[3]], j = best[2], r = S.rows[j];
    hoverRow(j, false);
    $("radRead").innerHTML = `<b>${hms(r[0])}Z</b> · ${hm(g.tc)}Z pass, ${g.dir < 0 ? "inbound" : "outbound"} · ${R.rr[j].toFixed(1)} n mi from the centre · ` +
      `FL ${r[S.col.ws] ?? "—"} kt, peak ${r[S.col.wp] ?? "—"} kt, SFMR ${r[S.col.sfmr] ?? "—"} kt`;
  }
  $("radCv").addEventListener("pointermove", radPointer);
  $("radCv").addEventListener("pointerdown", radPointer);
  $("radVar").addEventListener("click", e => { const b = e.target.closest("button[data-k]"); if (!b) return; R.v = b.dataset.k; $("radVar").innerHTML = segHTML(RVARS.filter(([k]) => R.legs.some(g => g[k])), R.v); radDraw(); });
  $("radLegs").addEventListener("click", e => {
    const b = e.target.closest("button[data-i]"); if (!b) return;
    const i = +b.dataset.i; R.off.has(i) ? R.off.delete(i) : R.off.add(i);
    b.setAttribute("aria-pressed", R.off.has(i) ? "false" : "true"); radDraw();
  });

  // ── intensity history (all missions of the storm) ────────────────────────
  const RD_HPA_H = 1.0, RD_MIN_H = 12, RI_KT = 30;
  function histData() {
    const s = S.man.storms[S.sid]; if (!s) return null;
    const fx = stormFixes(S.sid).filter(f => f.p != null && f.p > 850 && f.p < 1030);
    const eye = [], legsW = [];
    for (const k of s.missions) {
      const m = S.man.missions[k];
      for (const d of m.eyedrops || []) eye.push({ t: d[0], p: d[1], ws: d[2], est: d[3], how: d[4], ob: d[5], k });
      for (const g of m.legs || []) {
        if (g[4] != null) legsW.push({ t: g[6], v: g[4], r: g[5], kind: "fl", k, dir: g[0] });
        if (g[7] != null) legsW.push({ t: g[9], v: g[7], r: g[8], kind: "sfmr", k, dir: g[0] });
      }
    }
    // pressure tendency between consecutive measured (not extrapolated) fixes at least 1 h apart
    const meas = fx.filter(f => !f.x), tend = [];
    for (let j = 1, a = meas[0]; j < meas.length; j++) {
      const b = meas[j];
      if (b.t - a.t >= 3600) { tend.push({ t0: a.t, t1: b.t, r: (b.p - a.p) / ((b.t - a.t) / 3600) }); a = b; }
    }
    // rapid deepening: two measured fixes 12-24 h apart falling >= 1 hPa/h on average, and sustained: no fix in between
    // above the start or below the end by more than 2 hPa (a deepen-then-fill cycle does not count)
    const rd = [];
    for (let i = 0; i < meas.length; i++) for (let j = i + 1; j < meas.length; j++) {
      const h = (meas[j].t - meas[i].t) / 3600;
      if (h < RD_MIN_H) continue; if (h > 24) break;
      const mid = meas.slice(i + 1, j);
      if ((meas[i].p - meas[j].p) / h >= RD_HPA_H && mid.every(f => f.p <= meas[i].p + 2 && f.p >= meas[j].p - 2)) rd.push([meas[i].t, meas[j].t, meas[i].p - meas[j].p, h]);
    }
    // NHC rapid intensification: best-track max wind up >= 30 kt in 24 h
    const ri = [], bt = (S.storm && S.storm.btk) || [];
    for (let i = 0; i < bt.length; i++) for (let j = i + 1; j < bt.length; j++) {
      if (bt[j][0] - bt[i][0] > 86400) break;
      if (bt[j][0] - bt[i][0] === 86400 && bt[i][3] != null && bt[j][3] != null && bt[j][3] - bt[i][3] >= RI_KT) ri.push([bt[i][0], bt[j][0], bt[j][3] - bt[i][3]]);
    }
    return { fx, eye, legsW, tend, rd: mergeSpans(rd), ri: mergeSpans(ri), bt };
  }
  function mergeSpans(sp) {
    const s = sp.slice().sort((a, b) => a[0] - b[0]), out = [];
    for (const x of s) { const l = out[out.length - 1]; if (l && x[0] <= l[1]) { l[1] = Math.max(l[1], x[1]); l.n++; l.best = Math.max(l.best, x[2] / (x[3] || 24)); } else out.push(Object.assign([x[0], x[1]], { n: 1, best: x[2] / (x[3] || 24), d: x[2] })); }
    return out;
  }
  const HI = { pts: [] };
  function histBuild() {
    const D = S.sid && histData();
    if (!D || (!D.fx.length && !D.legsW.length && !D.eye.length)) { $("hist").hidden = true; return; }
    $("hist").hidden = false; HI.D = D;
    const s = S.man.storms[S.sid];
    $("histSub").innerHTML = `All ${s.missions.length} missions into ${esc(s.name || S.sid)}: minimum pressure from the vortex fixes (filled: measured by dropsonde; hollow: extrapolated from flight level) and from eye dropsondes (diamonds, splash pressure less 1 hPa per 10 kt of surface wind), ` +
      `the highest flight-level and SFMR wind of each leg, and NHC's best track (grey). Fixes without a valid pressure are left out.`;
    const f = [];
    for (const x of D.rd) f.push(`<span class="flag">Rapid deepening</span> ${dayhm(x[0])} – ${dayhm(x[1])} (measured fixes falling ≥ ${RD_HPA_H} hPa/h over ≥ ${RD_MIN_H} h; steepest ${x.best.toFixed(1)} hPa/h)`);
    for (const x of D.ri) f.push(`<span class="flag">Rapid intensification</span> (best track) ${dayhm(x[0])} – ${dayhm(x[1])}`);
    $("histFlags").innerHTML = f.length ? f.join("<br>") : `No rapid deepening in the measured fixes (≥ ${RD_HPA_H} hPa/h sustained over ≥ ${RD_MIN_H} h) and no rapid intensification in the best track (≥ ${RI_KT} kt in 24 h).`;
    $("histNote").innerHTML = `Rules: <b>rapid deepening</b> = two measured (dropsonde, not extrapolated) fixes ${RD_MIN_H}–24 h apart whose pressure fell by at least ${RD_HPA_H} hPa per hour on average, with no fix in between more than 2 hPa above the first or below the last, so the fall is sustained (shaded red in the pressure panel; overlapping windows merged); ` +
      `<b>rapid intensification</b> = NHC's definition, a ${RI_KT} kt rise of the maximum sustained wind in 24 h, from the best track (the 24-h windows shaded in the wind panel). The bottom panel is the pressure change per hour between consecutive measured fixes at least 1 h apart (red: falling at ≥ ${RD_HPA_H} hPa/h). ` +
      `Eye-sonde estimate: NHC practice of subtracting 1 hPa for every 10 kt of surface wind at the splash, since a sonde rarely lands at the exact centre. The best track after the storm is in progress is NHC's working best track and changes after the season.`;
    histDraw();
  }
  function histDraw() {
    if ($("hist").hidden || !HI.D) return;
    const g0 = cvSetup($("histCv")); if (!g0) return;
    const { c, W, H } = g0, D = HI.D;
    const ts = [...D.fx.map(f => f.t), ...D.eye.map(e => e.t), ...D.legsW.map(l => l.t)];
    if (!ts.length) return;
    const t0 = Math.min(...ts) - 6 * 3600, t1 = Math.max(...ts) + 6 * 3600;
    const L0 = 50, R0 = W - 12, X = t => L0 + (t - t0) / (t1 - t0) * (R0 - L0);
    // the key gets its own rows above the panels (it was drawn over the pressure panel, and left out below 640 px)
    const grey = dark() ? "#9aa0aa" : "#7a7f86", mband = dark() ? "rgba(124,180,255,0.10)" : "rgba(39,75,122,0.08)";
    const key = [["#b3261e", "fix (filled measured, hollow extrapolated)"], [dark() ? "#e6e8ec" : "#1f2620", "eye-sonde estimate"], [css("--s-fl"), "max FL wind per leg"],
                 [css("--s-sfmr"), "max SFMR per leg"], [grey, "best track"], [mband, "this mission", 1]];
    c.font = "11px 'Source Sans 3', sans-serif";
    const kRows = [[]]; { let kx = L0 + 6; for (const k of key) { const w = c.measureText(k[1]).width + 30; if (kx + w > R0 && kRows[kRows.length - 1].length) { kRows.push([]); kx = L0 + 6; } kRows[kRows.length - 1].push([k, kx]); kx += w; } }
    const KH = kRows.length * 16 + 4;
    const P1 = [8 + KH, H * 0.42], P2 = [H * 0.47, H * 0.76], P3 = [H * 0.81, H - 24];
    const bt = D.bt.filter(r => r[0] >= t0 && r[0] <= t1);
    // pressure panel
    const ps = [...D.fx.map(f => f.p), ...D.eye.map(e => e.est ?? e.p), ...bt.map(r => r[4]).filter(x => x)];
    let plo = Math.min(...ps) - 3, phi = Math.max(...ps) + 3; if (phi - plo < 12) { const m = (plo + phi) / 2; plo = m - 6; phi = m + 6; }
    const YP = p => P1[0] + (p - plo) / (phi - plo) * (P1[1] - P1[0]);          // pressure increases downward
    for (const x of D.rd) { c.fillStyle = "rgba(220,50,40,0.13)"; c.fillRect(X(x[0]), P1[0], X(x[1]) - X(x[0]), P1[1] - P1[0]); }
    { const step = niceStep((phi - plo) / 4); c.fillStyle = css("--axis"); c.textAlign = "right"; c.textBaseline = "middle";
      for (let v = Math.ceil(plo / step) * step; v <= phi; v += step) { const y = YP(v); c.strokeStyle = css("--grid"); c.beginPath(); c.moveTo(L0, y); c.lineTo(R0, y); c.stroke(); c.fillText(String(v), L0 - 5, y); }
      c.save(); c.translate(11, (P1[0] + P1[1]) / 2); c.rotate(-Math.PI / 2); c.textAlign = "center"; c.fillText("hPa", 0, 0); c.restore(); }
    // wind panel
    const ws = [...D.legsW.map(l => l.v), ...bt.map(r => r[3]).filter(x => x != null), 40];
    const whi = Math.ceil((Math.max(...ws) + 10) / 10) * 10, YW = w => P2[1] - w / whi * (P2[1] - P2[0]);
    for (const x of D.ri) { c.fillStyle = "rgba(220,50,40,0.13)"; c.fillRect(X(Math.max(t0, x[0])), P2[0], X(Math.min(t1, x[1])) - X(Math.max(t0, x[0])), P2[1] - P2[0]); }
    axisY(c, YW, 0, whi, L0, R0, "");
    c.save(); c.translate(11, (P2[0] + P2[1]) / 2); c.rotate(-Math.PI / 2); c.textAlign = "center"; c.fillStyle = css("--axis"); c.fillText("kt", 0, 0); c.restore();
    // tendency panel
    const rs = D.tend.map(x => x.r), rmax = Math.max(2, ...rs.map(Math.abs)) * 1.1, YT = r => (P3[0] + P3[1]) / 2 - r / rmax * (P3[1] - P3[0]) / 2;
    axisY(c, YT, -Math.ceil(rmax), Math.ceil(rmax), L0, R0, "");
    c.save(); c.translate(11, (P3[0] + P3[1]) / 2); c.rotate(-Math.PI / 2); c.textAlign = "center"; c.fillStyle = css("--axis"); c.fillText("hPa/h", 0, 0); c.restore();
    // time axis
    const span = t1 - t0, st = span > 6 * 86400 ? 86400 : span > 2 * 86400 ? 43200 : span > 86400 ? 21600 : 10800;
    c.textAlign = "center"; c.textBaseline = "top";
    for (let t = Math.ceil(t0 / st) * st; t <= t1; t += st) {
      const x = X(t); c.strokeStyle = css("--grid"); c.beginPath(); c.moveTo(x, P1[0]); c.lineTo(x, H - 24); c.stroke();
      const d = new Date(t * 1000); c.fillStyle = css("--axis");
      if (t % (st * (W < 600 ? 2 : 1)) === 0) c.fillText(`${d.getUTCDate()} ${MON[d.getUTCMonth()]}${d.getUTCHours() ? " " + p2(d.getUTCHours()) + "Z" : ""}`, x, H - 20);
    }
    HI.pts = [];
    const line = (pts, Y, k) => { c.strokeStyle = grey; c.lineWidth = 1.6; c.beginPath(); let on = false; for (const r of pts) { if (r[k] == null) { on = false; continue; } const x = X(r[0]), y = Y(r[k]); on ? c.lineTo(x, y) : c.moveTo(x, y); on = true; } c.stroke(); };
    line(bt, YP, 4); line(bt, YW, 3);
    for (const r of bt) { if (r[4]) HI.pts.push([X(r[0]), YP(r[4]), `Best track ${dayhm(r[0])}: ${r[4]} hPa, ${r[3] ?? "—"} kt`]); }
    // fixes
    for (const f of D.fx) {
      const x = X(f.t), y = YP(f.p);
      c.beginPath(); c.arc(x, y, 4, 0, 2 * Math.PI); c.strokeStyle = "#b3261e"; c.lineWidth = 1.8; c.fillStyle = f.x ? css("--panel") : "#b3261e"; c.fill(); c.stroke();
      HI.pts.push([x, y, `Fix ${dayhm(f.t)}: ${f.p} hPa${f.x ? " (extrapolated)" : ""} · ${esc(S.man.missions[f.k].aircraft)} ${esc(S.man.missions[f.k].mission)}`, f.k, f.t]);
    }
    for (const e of D.eye) {
      const p = e.est ?? e.p, x = X(e.t), y = YP(p);
      c.fillStyle = dark() ? "#e6e8ec" : "#1f2620"; c.beginPath(); c.moveTo(x, y - 5); c.lineTo(x + 5, y); c.lineTo(x, y + 5); c.lineTo(x - 5, y); c.closePath(); c.fill();
      HI.pts.push([x, y, `Eye sonde ${dayhm(e.t)}: splash ${e.p} hPa${e.ws != null ? `, surface wind ${e.ws} kt → est. ${e.est} hPa` : " (no surface wind: no estimate)"} · ${esc(e.how)}`, e.k, e.t]);
    }
    for (const l of D.legsW) {
      const x = X(l.t), y = YW(l.v);
      c.fillStyle = l.kind === "fl" ? css("--s-fl") : css("--s-sfmr"); c.beginPath(); c.arc(x, y, l.kind === "fl" ? 3.2 : 3.6, 0, 2 * Math.PI); c.fill();
      HI.pts.push([x, y, `${l.kind === "fl" ? "Max flight-level wind" : "Max SFMR wind"} ${l.v} kt at ${l.r.toFixed(0)} n mi, ${dayhm(l.t)} (${l.dir < 0 ? "inbound" : "outbound"} leg) · ${esc(S.man.missions[l.k].aircraft)}`, l.k, l.t]);
    }
    c.strokeStyle = css("--axis"); c.beginPath(); c.moveTo(L0, YT(0)); c.lineTo(R0, YT(0)); c.stroke();
    for (const x of D.tend) {
      const xa = X(x.t0), xb = X(x.t1), y = YT(x.r);
      c.fillStyle = x.r <= -RD_HPA_H ? "rgba(200,40,30,0.75)" : (dark() ? "rgba(160,170,185,0.6)" : "rgba(90,100,110,0.5)");
      c.fillRect(xa + 1, Math.min(y, YT(0)), Math.max(1, xb - xa - 2), Math.abs(y - YT(0)));
      HI.pts.push([(xa + xb) / 2, y, `${dayhm(x.t0)} – ${dayhm(x.t1)}: ${x.r > 0 ? "+" : ""}${x.r.toFixed(2)} hPa/h`]);
    }
    // key
    c.textAlign = "left"; c.textBaseline = "top"; c.font = "11px 'Source Sans 3', sans-serif";
    kRows.forEach((row, ri) => { const ky = 6 + ri * 16; for (const [[col, lab, box], kx] of row) {
      c.fillStyle = col; c.fillRect(kx, ky + 1, 10, 10); if (box) { c.strokeStyle = css("--axis"); c.lineWidth = 0.6; c.strokeRect(kx + 0.5, ky + 1.5, 9, 9); }
      c.fillStyle = css("--axis"); c.fillText(lab, kx + 14, ky); } });
    // current mission span
    const m = S.man.missions[S.key];
    if (m && m.first) { c.fillStyle = mband; c.fillRect(X(m.first), P1[0], X(m.last) - X(m.first), H - 24 - P1[0]); }
  }
  function histPointer(e) {
    const b = $("histCv").getBoundingClientRect(), x = e.clientX - b.left, y = e.clientY - b.top;
    let best = null, bd = 16 * 16;
    for (const p of HI.pts) { const d = (p[0] - x) ** 2 + (p[1] - y) ** 2; if (d < bd) { bd = d; best = p; } }
    $("histRead").innerHTML = best ? best[2] : "";
    HI.sel = best;
  }
  $("histCv").addEventListener("pointermove", histPointer);
  $("histCv").addEventListener("click", e => { histPointer(e); const p = HI.sel; if (p && p[3] && p[3] !== S.key) openMission(p[3]).then(() => goTime(p[4], true)); else if (p && p[3]) goTime(p[4], true); });

  // ── storm-relative winds and wind radii ──────────────────────────────────
  const SR = { ori: "north", v: "sfmr" };
  const FLR = [[650, 750, 0.9, "700 hPa"], [800, 900, 0.8, "850 hPa"]];     // Franklin et al. (2003) eyewall reductions (Table 2)
  function flReduce(r) {
    const p = r[S.col.p], w = r[S.col.ws]; if (p == null || w == null) return null;
    for (const [a, b, f] of FLR) if (p >= a && p <= b) return Math.round(w * f);
    return null;
  }
  function srBuild() {
    if (!S.mis || !S.rows.length) { $("sr").hidden = true; return; }
    const cf = centreFn(centreFixes(), S.storm && S.storm.btk), ix = S.col;
    const m = S.man.missions[S.key], tm = (m.first + m.last) / 2;
    const mv = S.storm && btMotion(S.storm.btk, tm);
    let hdg = null, spd = null;
    if (mv) { const c0 = btAt(S.storm.btk, tm); const dy = mv[0] * 3600 * 60, dx = mv[1] * 3600 * 60 * Math.cos((c0 ? c0[0] : 20) * rad); hdg = (Math.atan2(dx, dy) / rad + 360) % 360; spd = Math.hypot(dx, dy); }
    SR.hdg = hdg;
    const pts = [];
    SR.auto = SR.v === "sfmr" && !S.rows.some(r => r[ix.sfmr] != null);     // no valid SFMR on this flight: reduced FL wind
    const use = SR.auto ? "fl" : SR.v;
    for (let i = 0; i < S.rows.length; i++) {
      const r = S.rows[i]; if (r[ix.la] == null) continue;
      const c = cf(r[0]); if (!c) continue;
      const v = use === "sfmr" ? r[ix.sfmr] : flReduce(r); if (v == null) continue;
      const dy = (r[ix.la] - c[0]) * 60, dx = (r[ix.lo] - c[1]) * 60 * Math.cos(c[0] * rad), rr = Math.hypot(dx, dy);
      if (rr > 250) continue;
      pts.push({ i, dx, dy, r: rr, b: (Math.atan2(dx, dy) / rad + 360) % 360, v });
    }
    SR.pts = pts;
    // observed extents by compass quadrant (NHC's convention)
    const Q = ["NE", "SE", "SW", "NW"], qOf = b => Math.floor(b / 90);
    const ext = {};
    for (const thr of [34, 50, 64]) ext[thr] = Q.map((q, k) => {
      const qq = pts.filter(p => qOf(p.b) === k);
      if (!qq.length) return { s: "not sampled" };
      const rs = Math.max(...qq.map(p => p.r)), hit = qq.filter(p => p.v >= thr);
      if (!hit.length) return { s: `none (to ${Math.round(rs)})`, rs };
      const e = Math.max(...hit.map(p => p.r));
      return { e, rs, s: e >= rs - 5 ? `≥ ${Math.round(e)}` : String(Math.round(e)) };
    });
    // NHC radii: the b-deck synoptic time nearest the mission's middle (+-4 h)
    const rad0 = (S.storm && S.storm.radii) || [];
    let nt = null; for (const r of rad0) if (Math.abs(r[0] - tm) <= 14400 && (nt == null || Math.abs(r[0] - tm) < Math.abs(nt - tm))) nt = r[0];
    const nhc = {}; for (const r of rad0) if (r[0] === nt) nhc[r[1]] = r.slice(2, 6);
    SR.nhc = nhc; SR.ext = ext;
    $("sr").hidden = false;
    $("srSub").innerHTML = (SR.auto ? `<b>This flight has no valid SFMR values, so the flight-level wind reduced to the surface is shown.</b> ` : "") +
      `${pts.length.toLocaleString("en-US")} ${use === "sfmr" ? "SFMR surface winds" : "flight-level winds reduced to the surface"} from this flight placed around the centre interpolated to each observation's time` +
      `${hdg != null ? `; storm motion ${Math.round(hdg)}° at ${Math.round(spd)} kt (best track)` : ""}. Dashed arcs: NHC's wind radii${nt ? ` at ${dayhm(nt)} (ATCF best track)` : " (none within 4 h of this flight)"}.`;
    $("srOri").innerHTML = segHTML([["north", "North up"], ["motion", "Motion up"]], SR.ori);
    $("srVar").innerHTML = segHTML([["sfmr", "SFMR"], ["fl", "Flight level reduced"]], SR.auto ? "fl" : SR.v);   // the pressed button is what is drawn
    const cell = (o, n) => `<td>${o.s === "not sampled" ? '<span class="mu">not sampled</span>' : o.s}</td><td class="mu">${n == null ? "—" : n === 0 ? "0" : n}</td>`;
    $("srTbl").innerHTML = `<table><thead><tr><th>Wind</th>${Q.map(q => `<th colspan="2">${q} (obs · NHC)</th>`).join("")}</tr></thead><tbody>` +
      [34, 50, 64].map(thr => `<tr><td>${thr} kt</td>${Q.map((q, k) => cell(ext[thr][k], nhc[thr] ? nhc[thr][k] : null)).join("")}</tr>`).join("") + "</tbody></table>";
    $("srNote").innerHTML = `Extent = the farthest observation in the quadrant (n mi) at or above the threshold. "≥" means the wind was still that strong at the edge of what the aircraft sampled, so the true extent is larger; ` +
      `"none (to R)" means it was sampled out to R n mi without reaching the threshold; "not sampled" means no observation in that quadrant, and no value is estimated. ` +
      `Quadrants are compass quadrants (NE, SE, SW, NW), as in NHC's radii. One flight samples a few radial legs, so an observed extent is a lower bound unless a leg ran along the quadrant's widest wind. ` +
      (use === "fl" ? `<b>Alternative</b>: flight-level wind × 0.9 at 700 hPa and × 0.8 at 850 hPa, the eyewall reductions of Franklin, Black and Valde (2003, <i>Wea. Forecasting</i> 18, 32–44, Table 2); their outer-vortex factors are lower (0.85–0.80 at 700 hPa, 0.80–0.75 at 850 hPa); other levels are left out. ` : "") +
      `SFMR is the measured surface wind and is the primary estimate here.`;
    srDraw();
  }
  function srDraw() {
    if ($("sr").hidden) return;
    const cv = $("srCv"); cv.style.height = cv.clientWidth + "px";
    const g0 = cvSetup(cv); if (!g0) return;
    const { c, W, H } = g0, cx = W / 2, cy = H / 2;
    // the range reaches the farthest observation kept (srBuild keeps 250 n mi; at a fixed 150 the outer legs fell off the canvas)
    const rmaxD = Math.max(50, ...SR.pts.map(p => p.r)), RM = rmaxD > 200 ? 250 : rmaxD > 150 ? 200 : rmaxD > 110 ? 150 : rmaxD > 60 ? 100 : 60, s = (Math.min(W, H) / 2 - 18) / RM;
    const rot = SR.ori === "motion" && SR.hdg != null ? SR.hdg * rad : 0;
    const P = (dx, dy) => { const x = dx * Math.cos(rot) - dy * Math.sin(rot), y = dx * Math.sin(rot) + dy * Math.cos(rot); return [cx + x * s, cy - y * s]; };
    // ring labels on the diagonal farthest from the motion arrow (at 45 deg they sat on a north-east-moving storm's arrow)
    const arrowB = SR.hdg != null ? ((SR.hdg - (rot ? SR.hdg : 0)) % 360 + 360) % 360 : null;
    const labB = arrowB == null ? 45 : [45, 135, 225, 315].sort((a, b) => Math.abs(((b - arrowB + 540) % 360) - 180) - Math.abs(((a - arrowB + 540) % 360) - 180))[0];
    const lsx = Math.sin(labB * rad), lsy = Math.cos(labB * rad);
    c.strokeStyle = css("--grid"); c.fillStyle = css("--axis"); c.textAlign = lsx >= 0 ? "left" : "right"; c.textBaseline = lsy >= 0 ? "bottom" : "top";
    for (const r of [25, 50, 100, 150, 200, 250]) if (r <= RM && (RM < 200 || r !== 25)) { c.beginPath(); c.arc(cx, cy, r * s, 0, 2 * Math.PI); c.stroke(); c.fillText(r + " n mi", cx + r * s * lsx + (lsx >= 0 ? 2 : -2), cy - r * s * lsy); }
    for (let k = 0; k < 4; k++) { const a = k * 90 * rad; const [x, y] = P(Math.sin(a) * RM, Math.cos(a) * RM); c.beginPath(); c.moveTo(cx, cy); c.lineTo(x, y); c.stroke(); }
    // north arrow / motion arrow
    const [nx, ny] = P(0, RM * 0.97); c.fillStyle = css("--axis"); c.textAlign = "center"; c.textBaseline = "middle"; c.fillText("N", nx, ny);
    if (SR.hdg != null) {
      const a = SR.hdg * rad, [mx, my] = P(Math.sin(a) * RM * 0.35, Math.cos(a) * RM * 0.35);
      c.strokeStyle = css("--ink"); c.lineWidth = 2.2; c.beginPath(); c.moveTo(cx, cy); c.lineTo(mx, my); c.stroke();
      const ang = Math.atan2(my - cy, mx - cx); c.beginPath(); c.moveTo(mx, my); c.lineTo(mx - 8 * Math.cos(ang - 0.4), my - 8 * Math.sin(ang - 0.4)); c.lineTo(mx - 8 * Math.cos(ang + 0.4), my - 8 * Math.sin(ang + 0.4)); c.closePath(); c.fillStyle = css("--ink"); c.fill();
      c.lineWidth = 1;
    }
    for (const p of SR.pts) { const [x, y] = P(p.dx, p.dy); c.fillStyle = binCol(WIND, p.v); c.fillRect(x - 1.8, y - 1.8, 3.6, 3.6); }
    // NHC radii: quarter arcs per compass quadrant
    const dash = { 34: [6, 4], 50: [3, 3], 64: [1.5, 2.5] };
    for (const thr of [34, 50, 64]) {
      const q = SR.nhc[thr]; if (!q) continue;
      c.strokeStyle = binCol(WIND, thr); c.lineWidth = 2.2; c.setLineDash(dash[thr]);
      q.forEach((r, k) => {
        if (!r) return;
        const a0 = k * 90, a1 = a0 + 90; c.beginPath();
        for (let a = a0; a <= a1; a += 3) { const [x, y] = P(Math.sin(a * rad) * Math.min(r, RM), Math.cos(a * rad) * Math.min(r, RM)); a === a0 ? c.moveTo(x, y) : c.lineTo(x, y); }
        c.stroke();
      });
    }
    c.setLineDash([]); c.lineWidth = 1;
    c.fillStyle = css("--axis"); c.textAlign = "left"; c.textBaseline = "top";
    c.fillText(SR.ori === "motion" ? "motion up" : "north up", 6, 6);
  }
  $("srOri").addEventListener("click", e => { const b = e.target.closest("button[data-k]"); if (!b) return; SR.ori = b.dataset.k; $("srOri").innerHTML = segHTML([["north", "North up"], ["motion", "Motion up"]], SR.ori); srDraw(); });
  $("srVar").addEventListener("click", e => { const b = e.target.closest("button[data-k]"); if (!b) return; SR.v = b.dataset.k; srBuild(); });

  // ── eye and eyewall dropsondes ───────────────────────────────────────────
  function thetaE(p, T, Td) {         // Bolton (1980), eq. 39 with T_L from eq. 15; p hPa, T/Td degC -> K
    const TK = T + 273.15, DK = Td + 273.15, e = 6.112 * Math.exp(17.67 * Td / (Td + 243.5)), r = 0.622 * e / (p - e);
    const TL = 1 / (1 / (DK - 56) + Math.log(TK / DK) / 800) + 56;
    const thL = TK * Math.pow(1000 / (p - e), 0.2854) * Math.pow(TK / TL, 0.28 * r);
    return thL * Math.exp((3036 / TL - 1.78) * r * (1 + 0.448 * r));
  }
  // eye / centre sondes of the open mission (the rule of recon_mirror.eye_drops; the manifest holds the same for all missions)
  function eyeDropsHere() {
    const cf = centreFn(centreFixes(), S.storm && S.storm.btk), out = [];
    for (const d of S.mis.drops) {
      if (!d.sfc_p) continue;
      const sp = d.spg || d.spl || d.rel, t = d.spl_t || d.rel_t; if (!t) continue;
      let how = ["EYE", "CENTER", "CNTR"].find(x => (d.tags || []).includes(x));
      how = how ? how.toLowerCase() : null;
      if (!how && sp) { const c = cf(t); if (c && gkm(sp.la, sp.lo, c[0], c[1]) / NMI <= 5) how = "within 5 n mi of the centre"; }
      if (!how) continue;
      const ws = d.sfc_wind ? d.sfc_wind.ws : null;
      out.push([t, d.sfc_p, ws, ws != null ? Math.round(d.sfc_p - ws / 10) : null, how, d.obnum]);
    }
    return out;
  }
  function eyeBuild() {
    if (!S.mis) { $("eye").hidden = true; return; }
    const ed = eyeDropsHere();
    const ew = S.mis.drops.map((d, i) => [d, i]).filter(([d]) => (d.tags || []).includes("EYEWALL") && d.prof && d.prof.some(r => r[5] != null && r[1] != null));
    if (!ed.length && !ew.length) { $("eye").hidden = true; return; }
    $("eye").hidden = false;
    $("eyeTbl").innerHTML = ed.length ? `<table><thead><tr><th>Splash</th><th>Sonde</th><th>Splash pressure</th><th>Surface wind</th><th>Central pressure estimate</th><th>Identified by</th></tr></thead><tbody>` +
      ed.map(e => `<tr><td>${hm(e[0])}Z</td><td>OB ${e[5] ?? "—"}</td><td>${e[1]} hPa</td><td>${e[2] != null ? e[2] + " kt" : "—"}</td><td><b>${e[3] != null ? e[3] + " hPa" : "—"}</b></td><td class="mu">${esc(e[4])}</td></tr>`).join("") + "</tbody></table>"
      : `<p class="empty">No eye or centre dropsonde with a splash pressure on this flight.</p>`;
    $("eyeProf").hidden = !ew.length;
    $("eyeKey").innerHTML = ew.map(([d, i], k) => `<button type="button" aria-pressed="true" data-i="${i}"><i style="background:${legCol(k)}"></i>OB ${d.obnum ?? i + 1} ${d.spl_t ? hm(d.spl_t) + "Z" : ""}</button>`).join("");
    $("eyeKey").querySelectorAll("button").forEach(b => b.onclick = () => openSkewT(+b.dataset.i));
    $("eyeNote").innerHTML = `Central pressure estimate from an eye or centre sonde (remark EYE or CENTER, or a splash within 5 n mi of the interpolated centre): the splash pressure less 1 hPa for every 10 kt of surface wind, the rule NHC applies because a sonde in a moving wind field rarely lands at the pressure minimum. ` +
      (ew.length ? `Eyewall sondes are those the operator remarked EYEWALL. θe after Bolton (1980). Click a sonde above for its skew-T.` : "");
    S.eyew = ew; eyeDraw();
  }
  function eyeDraw() {
    if ($("eye").hidden || !S.eyew || !S.eyew.length) return;
    for (const [id, kind] of [["eyeWs", "ws"], ["eyeTe", "te"]]) {
      const g0 = cvSetup($(id)); if (!g0) continue;
      const { c, W, H } = g0, L0 = 46, R0 = W - 10, T0 = 8, B0 = H - 26;
      const series = S.eyew.map(([d]) => d.prof.filter(r => r[1] != null && r[1] >= 0).map(r => [r[1], kind === "ws" ? r[5] : (r[2] != null && r[3] != null && r[0] > 100 ? thetaE(r[0], r[2], r[3]) : null)]).filter(x => x[1] != null).sort((a, b) => a[0] - b[0]));
      const all = series.flat(); if (!all.length) continue;
      const zmax = Math.min(Math.max(...all.map(x => x[0])), 12000), zt = Math.ceil(zmax / 1000) * 1000;
      let vlo = Math.min(...all.map(x => x[1])), vhi = Math.max(...all.map(x => x[1]));
      if (kind === "ws") vlo = 0; vlo = Math.floor(vlo / 5) * 5; vhi = Math.ceil(vhi / 5) * 5 + 5;
      const X = v => L0 + (v - vlo) / (vhi - vlo) * (R0 - L0), Y = z => B0 - z / zt * (B0 - T0);
      axisY(c, Y, 0, zt, L0, R0, "m", v => v >= 1000 ? (v / 1000) + "k" : String(v));
      c.textAlign = "center"; c.textBaseline = "top"; c.fillStyle = css("--axis");
      const st = niceStep((vhi - vlo) / 5);
      for (let v = Math.ceil(vlo / st) * st; v <= vhi; v += st) { c.strokeStyle = css("--grid"); c.beginPath(); c.moveTo(X(v), T0); c.lineTo(X(v), B0); c.stroke(); c.fillText(String(v), X(v), B0 + 4); }
      c.fillText(kind === "ws" ? "wind speed (kt)" : "θe (K)", (L0 + R0) / 2, B0 + 14);
      series.forEach((sr, k) => { c.strokeStyle = legCol(k); c.lineWidth = 1.8; c.beginPath(); sr.forEach(([z, v], j) => { const x = X(v), y = Y(Math.min(z, zt)); j ? c.lineTo(x, y) : c.moveTo(x, y); }); c.stroke(); });
    }
  }

  // ── alerts while the page is open (Notification API, no server push) ────
  const AL = { on: false, last: {} };
  function alertBaseline() { if (!S.man) return; for (const sid of Object.keys(S.man.storms)) { const f = stormFixes(sid); AL.last[sid] = f.length ? f[f.length - 1].t : 0; } }
  function alertCheck() {
    if (!AL.on || !S.man) return;
    for (const [sid, s] of Object.entries(S.man.storms)) {
      if (!s.active && s.year !== S.man.season) continue;
      const fx = stormFixes(sid), prev = AL.last[sid] || 0, nw = fx.filter(f => f.t > prev);
      for (const f of nw) {
        const before = fx.filter(g => g.t < f.t && g.p != null).pop();
        const drop = before && f.p != null ? before.p - f.p : null;
        const title = `${s.name || sid}: new vortex fix`;
        let body = `${f.p != null ? f.p + " hPa" + (f.x ? " (extrapolated)" : "") : "no pressure"} at ${hm(f.t)}Z · ${S.man.missions[f.k].aircraft} ${S.man.missions[f.k].mission}`;
        if (drop != null && drop >= 3) body = `Pressure down ${drop} hPa since ${hm(before.t)}Z. ` + body;
        try { new Notification(drop != null && drop >= 3 ? `${s.name || sid}: pressure falling` : title, { body, tag: `recon-${sid}-${f.t}` }); } catch (e) { /* not allowed */ }
      }
      if (fx.length) AL.last[sid] = Math.max(prev, fx[fx.length - 1].t);
    }
  }
  (function alertInit() {
    const b = $("alertBtn"), n = $("alertNote");
    if (!("Notification" in window)) { b.disabled = true; b.title = "This browser has no notifications"; return; }
    b.onclick = async () => {
      if (AL.on) { AL.on = false; b.setAttribute("aria-pressed", "false"); b.textContent = "Alert me to new fixes"; n.hidden = true; return; }
      let p = Notification.permission;
      if (p === "default") p = await Notification.requestPermission();
      if (p !== "granted") { n.hidden = false; n.textContent = "Notifications are blocked for this site in the browser's settings."; return; }
      AL.on = true; alertBaseline();
      b.setAttribute("aria-pressed", "true"); b.textContent = "Alerts on";
      n.hidden = false; n.textContent = "Only while this tab stays open: a notification for each new vortex fix, and when the pressure falls 3 hPa or more since the previous fix. The page checks every 90 s while a mission is airborne.";
    };
  })();

  function analyses() { radBuild(); histBuild(); srBuild(); eyeBuild(); }
  let rzA = null;
  addEventListener("resize", () => { clearTimeout(rzA); rzA = setTimeout(() => { radDraw(); histDraw(); srDraw(); eyeDraw(); }, 150); });

  // ── controls ─────────────────────────────────────────────────────────────
  $("selStorm").onchange = e => pickStorm(e.target.value, null);
  $("selMission").onchange = e => openMission(e.target.value);
  const seg = $("segVar");
  seg.querySelectorAll("button").forEach(b => b.classList.toggle("on", b.dataset.v === S.v));
  seg.addEventListener("click", e => {
    const b = e.target.closest("button[data-v]"); if (!b) return;
    S.v = b.dataset.v;
    seg.querySelectorAll("button").forEach(x => x.classList.toggle("on", x === b));
    try { localStorage.setItem("recon.var", S.v); } catch (err) { /* storage blocked */ }
    if (S.mis) { drawTrack(); legend(); }
  });

  loadManifest(true);
})();
