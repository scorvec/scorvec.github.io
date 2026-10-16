// Execute the ENTIRE render path (drawSkewT, drawHodo, fillTables) under node
// with stubbed canvas/DOM. A parse check cannot catch runtime failures (a TDZ
// "Cannot access 'x' before initialization" passes one and blanks the page);
// this catches that class before push.
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
// one-line `const NAME = ...;` declarations, taken from app.js itself so the
// test never runs against a stale copy of the real plot geometry or thresholds
const grabConst = n => {
  const m = new RegExp(`^const ${n} = .*?;`, "m").exec(app);
  if (!m) throw new Error("missing const " + n);
  return m[0];
};
const NEED = ["parseCSV", "thin", "sanitizeHeights", "fillDewpoints", "fillWinds", "frostPtC", "cloudLayers", "estCeiling", "drawTempAdv", "meanSR", "brnShear",
  "fitCanvas", "plotBox", "prev12Live", "interpHagl", "interpP", "windAt", "meanWindAgl", "wetbulbC", "wbzAgl",
  "freezingLvlAgl", "tropopause", "columnRH", "layerRH", "mseProfile", "thetaEK", "precipType",
  "kucheraRatio", "drawBarbs", "climoSlot", "climoTab", "climoAllTime", "synClass", "climoPct", "pctColor",
  "climoCell", "doyOf", "drawSkewT", "drawHodo", "fillTables"];
const CONSTS = ["MISSING", "KT", "RD", "CP", "CLIMO_PCTS", "CLIMO_LO", "CLIMO_MIN_N", "CLIMO_HIGH_ONLY",
  "CLIMO_FLOOR", "MON", "SK", "PLOT_ASPECT", "dirOf", "climoTabs"];
const preamble = CONSTS.map(grabConst).join("\n") + `
const TH = new Proxy({}, { get: () => "#888" });
const MOBILE_MQ = { matches: false };
let plotTitle = "smoke test", plotCoords = "0N 0E", plotNote = "";
let EXPORT_PX = null, BARB_GAP = 22, lastProfFull = null;
let pinned = null, climo = null, climoNow = {}, lastDoy = 192, lastHourZ = 12;
let current = null, prev12 = null, lastValidDt = null;
let anomalies = {}, entries = {}, byWmo = {}, igraStations = {};
// traceAdiabat is a WASM bridge (tested separately); the card layout, the
// async 12-h trend fetch and the model-point chips are DOM/network side
// effects of fillTables — stubbed here
const traceAdiabat = () => new Float32Array(0);
const fitLayout = () => {}, queuePrev12 = () => {}, modelPointChips = () => {};
const fmt = (v, d) => (v === MISSING || !isFinite(v)) ? "—" : (+v).toFixed(d || 0);
`;
const src = preamble + NEED.map(grab).join("\n") +
  "\nreturn { parseCSV, thin, sanitizeHeights, fillDewpoints, drawSkewT, drawHodo, fillTables };";
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
for (const f of ["drawSkewT", "drawHodo", "fillTables"]) {
  try { fns[f](prof, res); console.log(`  ${f}: OK`); }
  catch (e) { console.error(`  ${f}: THREW — ${e.message}`); failed++; }
}
process.exit(failed ? 1 : 0);
