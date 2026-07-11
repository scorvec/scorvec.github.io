// Execute the ENTIRE render path (drawSkewT, drawHodo, drawMSE, fillTables)
// under node with stubbed canvas/DOM. A parse check cannot catch runtime
// failures — a TDZ ("Cannot access 'o' before initialization") shipped through
// one and blanked the page. This catches that class before push.
//
//   node scripts/skewt/smoke_render.js path/to/sounding.csv
"use strict";
const fs = require("fs");
const path = require("path");
const SKEWT = path.resolve(__dirname, "..", "..", "skewt");

// ---- minimal DOM/canvas stubs -------------------------------------------
const ctxStub = () => new Proxy({}, {
  get: (t, k) => {
    if (k === "measureText") return () => ({ width: 10 });
    if (k === "canvas") return { width: 800, height: 800 };
    return typeof k === "string" && /^(fillStyle|strokeStyle|font|lineWidth|textAlign|lineCap|globalAlpha)$/.test(k)
      ? t[k] : (...a) => undefined;
  },
  set: () => true,
});
const elements = {};
const el = id => elements[id] ||= {
  id, style: {}, hidden: false, innerHTML: "", textContent: "",
  width: 800, height: 800, clientWidth: 800, clientHeight: 800,
  getContext: () => ctxStub(),
  addEventListener: () => {}, classList: { add(){}, remove(){}, toggle(){} },
  querySelectorAll: () => [], appendChild: () => {},
};
global.document = {
  getElementById: el,
  createElement: () => el("_tmp" + Math.random()),
  querySelectorAll: () => [], addEventListener: () => {},
  querySelector: () => el("_q"),
};
global.window = global;
global.location = { hash: "", search: "" };
Object.defineProperty(global, "navigator", { value: { clipboard: { writeText: async () => {} } }, configurable: true });
global.matchMedia = () => ({ matches: false });
global.addEventListener = () => {};
global.fetch = async () => ({ ok: false, json: async () => ({}), text: async () => "" });
global.L = new Proxy({}, { get: () => (...a) => new Proxy({}, {
  get: (t, k) => k === "addTo" ? () => t : (...b) => t }) });
global.ResizeObserver = class { observe(){} };

// ---- load app.js and pull the functions we need ---------------------------
const app = fs.readFileSync(path.join(SKEWT, "app.js"), "utf8");
const grab = n => {
  const i = app.indexOf(`function ${n}(`);
  if (i < 0) throw new Error("missing function " + n);
  let d = 0;
  for (let k = app.indexOf("{", i); k < app.length; k++) {
    if (app[k] === "{") d++;
    if (app[k] === "}") { d--; if (!d) return app.slice(i, k + 1); }
  }
};
const NEED = ["parseCSV", "thin", "sanitizeHeights", "fillDewpoints", "fillWinds", "frostPtC",
  "fitCanvas", "interpHagl", "interpP", "windAt", "meanWindAgl", "wetbulbC", "wbzAgl",
  "freezingLvlAgl", "tropopause", "columnRH", "layerRH", "mseProfile", "precipType",
  "kucheraRatio", "drawBarbs", "climoSlot", "climoPct", "pctColor", "climoCell", "doyOf",
  "drawSkewT", "drawHodo", "drawMSE", "fillTables"];
const preamble = `
const MISSING = -9999, KT = 1.9438, RD = 287.05, CP = 1005.7, LV = 2.501e6, GRAV = 9.80665;
const CLIMO_PCTS = [1,5,10,25,50,75,90,95,99], CLIMO_LO = 10, CLIMO_HI = 90, CLIMO_MIN_N = 30;
const CLIMO_HIGH_ONLY = new Set(["ecape","ship"]);
const CLIMO_FLOOR = { ecape: 100, ship: 0.5 };
const MON = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
const SK = { l: 58, r: 90, t: 48, b: 42, pBot: 105000, pTop: 10000, tL: -35, tR: 45 };
const TH = new Proxy({}, { get: () => "#888" });
let plotTitle = "smoke test", plotCoords = "0N 0E", plotNote = "";
let pinned = null, climo = null, climoNow = {}, lastMonth = "07", lastDoy = 192;
let anomalies = {}, entries = {}, byWmo = {}, igraStations = {};
// traceAdiabat is a WASM bridge (tested separately) — stub it here
const traceAdiabat = () => new Float32Array(0);
const dirOf = (u, v) => ((Math.atan2(-u, -v) * 180 / Math.PI) + 360) % 360;
const fmt = (v, d) => (v === MISSING || !isFinite(v)) ? "—" : (+v).toFixed(d || 0);
`;
const src = preamble + NEED.map(grab).join("\n") +
  "\nreturn { parseCSV, thin, sanitizeHeights, fillDewpoints, drawSkewT, drawHodo, drawMSE, fillTables };";
const fns = new Function("document", "window", src)(global.document, global);

// ---- run it on a real sounding --------------------------------------------
const csv = fs.readFileSync(process.argv[2], "utf8");
const prof = fns.thin(fns.parseCSV(csv));
if (!prof) { console.error("SMOKE: could not parse profile"); process.exit(1); }
fns.sanitizeHeights(prof);
fns.fillDewpoints(prof);
const res = { rc: 0, o: new Array(64).fill(-9999),
  sb: new Array(prof.P.length).fill(-9999),
  ml: new Array(prof.P.length).fill(-9999),
  mu: new Array(prof.P.length).fill(-9999) };
res.o[0] = 500; res.o[22] = 5; res.o[23] = 5; res.o[34] = prof.P[0];

let failed = 0;
for (const f of ["drawSkewT", "drawHodo", "drawMSE", "fillTables"]) {
  try { fns[f](prof, res); console.log(`  ${f}: OK`); }
  catch (e) { console.error(`  ${f}: THREW — ${e.message}`); failed++; }
}
process.exit(failed ? 1 : 0);
