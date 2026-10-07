/* Hurricane reconnaissance plotter (recon.html).
   Data: the recon-data branch, written by .github/workflows/recon.yml (scripts/recon/recon_mirror.py) from the NHC
   reconnaissance archive, which sends no CORS header. manifest.json lists storms and missions; a mission's file
   (missions/<year>/<key>.json) is fetched only when it is opened; storms/<SID>.json = best track + forecast. */
"use strict";
(function () {
  // ── hosts ────────────────────────────────────────────────────────────────
  // To serve through the Bunny CDN first (once its Edge Rule caches /recon-data/* for 60 s), put "cdn" first here.
  const HOSTS = ["raw", "jsd", "api"];
  const REPO = "scorvec/scorvec.github.io", BR = "recon-data";
  const URLS = {
    cdn: p => [`https://frames.scorvec.com/${BR}/${p}`],
    raw: (p, b) => [`https://raw.githubusercontent.com/${REPO}/${BR}/${p}${b ? "?t=" + encodeURIComponent(b) : ""}`],
    jsd: p => [`https://cdn.jsdelivr.net/gh/${REPO}@${BR}/${p}`],
    api: p => [`https://api.github.com/repos/${REPO}/contents/${p}?ref=${BR}`, { headers: { Accept: "application/vnd.github.raw" } }],
  };
  // fresh = a file that changes (manifest, an airborne mission): api (always fresh) before jsDelivr (cached ~12 h)
  async function dataFetch(path, bust, fresh) {
    let order = HOSTS.slice();
    if (fresh) order = order.filter(h => h !== "jsd").concat(order.includes("jsd") ? ["jsd"] : []);
    for (let i = 0; i < order.length; i++) {
      const [u, opt] = URLS[order[i]](path, bust);
      try {
        const r = await fetch(u, opt);
        if (r.ok) return await r.json();
        if (r.status === 404 && order[i] === "raw") return null;
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
  const dayhm = t => { const d = new Date(t * 1000); return `${d.getUTCDate()} ${MON[d.getUTCMonth()]} ${hm(t)}Z`; };
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
  map.createPane("labels"); map.getPane("labels").style.zIndex = 350; map.getPane("labels").style.pointerEvents = "none";
  L.tileLayer(ESRI + base + "_Reference/MapServer/tile/{z}/{y}/{x}", { maxZoom: 13, pane: "labels", opacity: 0.9 }).addTo(map);
  const rend = L.canvas({ padding: 0.3, tolerance: 6 });
  const lyr = { btk: L.layerGroup().addTo(map), fixes: L.layerGroup().addTo(map), track: L.layerGroup().addTo(map),
                barbs: L.layerGroup().addTo(map), drops: L.layerGroup().addTo(map), vdm: L.layerGroup().addTo(map),
                hover: L.layerGroup().addTo(map) };
  if (window.ResizeObserver) new ResizeObserver(() => map.invalidateSize()).observe($("map"));

  // ── load ─────────────────────────────────────────────────────────────────
  async function loadManifest(first) {
    let man;
    try { man = await dataFetch("manifest.json", Math.floor(Date.now() / 60000), true); }
    catch (e) {
      if (first) setStatus(`<span class="mu">The data mirror could not be reached (tried raw.githubusercontent.com, jsDelivr and api.github.com). Try again shortly.</span>`);
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
      else status();
    }
    pod();
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
  }
  function clearMission() {
    S.mis = null; S.key = null; S.rows = [];
    for (const k of ["track", "barbs", "drops", "vdm", "hover"]) lyr[k].clearLayers();
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
  map.on("zoomend", () => { if (S.mis) drawBarbs(); });

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
    const fx = [];
    for (const k of s.missions) for (const f of (S.man.missions[k].fixes || [])) fx.push(f.concat([k]));
    fx.sort((a, b) => a[0] - b[0]);
    if (fx.length > 1) L.polyline(fx.map(f => [f[1], f[2]]), { renderer: rend, color: "#b3261e", weight: 1.6, dashArray: "4 4", interactive: false }).addTo(lyr.fixes);
    for (const f of fx) {
      if (f[4] === S.key) continue;                               // the open mission draws its own
      L.circleMarker([f[1], f[2]], { renderer: rend, radius: 3.5, color: "#b3261e", weight: 1.5, fillColor: "#fff", fillOpacity: 1 })
        .bindTooltip(`Fix ${dayhm(f[0])}${f[3] ? " · " + f[3] + " hPa" : ""} · ${esc(S.man.missions[f[4]].aircraft)}`).addTo(lyr.fixes);
    }
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
      if (m.last_hdob) parts.push(`<span>Newest observation <b>${dayhm(m.last_hdob)}</b> (${ago(m.last_hdob)})</span>`);
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
    setTimeout(async () => { if (!document.hidden) await loadManifest(false); poll(); }, wait);
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
      const step = niceStep((hi - lo) / 4);
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
