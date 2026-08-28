// Compute ECAPE for each mirrored sounding, using the SAME WebAssembly build the
// browser runs — so a live ECAPE and its climatological percentile come from
// identical physics. Avoids compiling SHARPlib in CI just to get one number.
//
//   node scripts/skewt/ecape_latest.js SOUNDINGS_DIR
//   -> prints "WMO ECAPE" per line (J/kg), skipping anything that won't compute
"use strict";
const fs = require("fs");
const path = require("path");

const MISSING = -9999;
const SKEWT = path.resolve(__dirname, "..", "..", "skewt");

// mirror CSV: 0 time, 3 pressure_hPa, 4 height_m, 5 temp_C, 6 dewpoint_C,
// 11 wind_dir_deg, 12 wind_speed_m/s
function parseCSV(text) {
  const o = { P: [], H: [], T: [], D: [], U: [], V: [] };
  const lines = text.split("\n");
  for (let i = 1; i < lines.length; i++) {
    const c = lines[i].split(",");
    if (c.length < 13) continue;
    const p = +c[3], h = +c[4], t = +c[5], d = +c[6], wd = +c[11], ws = +c[12];
    if (!isFinite(p) || p < 20 || !isFinite(t)) continue;
    o.P.push(p * 100);
    o.H.push(isFinite(h) ? h : NaN);
    o.T.push(t + 273.15);
    o.D.push(isFinite(d) ? d + 273.15 : NaN);
    if (isFinite(wd) && isFinite(ws)) {
      o.U.push(-ws * Math.sin(wd * Math.PI / 180));
      o.V.push(-ws * Math.cos(wd * Math.PI / 180));
    } else { o.U.push(NaN); o.V.push(NaN); }
  }
  return o.P.length >= 10 ? o : null;
}

// SHARPlib requires monotonically increasing height; some feeds report a literal
// 0.0 where the height is missing (see sanitizeHeights in app.js).
function sanitize(o) {
  const Rd = 287.05, g = 9.80665;
  for (let i = 1; i < o.P.length; i++) {
    if (isFinite(o.H[i]) && o.H[i] > o.H[i - 1]) continue;
    o.H[i] = (isFinite(o.T[i]) && isFinite(o.T[i - 1]) && o.P[i - 1] > o.P[i])
      ? o.H[i - 1] + Rd * 0.5 * (o.T[i] + o.T[i - 1]) / g * Math.log(o.P[i - 1] / o.P[i])
      : o.H[i - 1] + 1;
  }
  // winds: interpolate gaps rather than zero-filling (which would wreck the
  // storm-relative terms ECAPE depends on)
  const ok = [];
  for (let i = 0; i < o.U.length; i++) if (isFinite(o.U[i])) ok.push(i);
  if (!ok.length) { o.U.fill(0); o.V.fill(0); return o; }
  for (let i = 0; i < o.U.length; i++) {
    if (isFinite(o.U[i])) continue;
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
  return o;
}

(async () => {
  const dir = process.argv[2];
  if (!dir || !fs.existsSync(dir)) process.exit(0);

  const createSharp = require(path.join(SKEWT, "sharplib.js"));
  const wasmBinary = fs.readFileSync(path.join(SKEWT, "sharplib.wasm"));
  const M = await createSharp({
    instantiateWasm: (imports, cb) => {
      WebAssembly.instantiate(wasmBinary, imports).then(r => cb(r.instance));
      return {};
    },
  });
  const nOut = M._out_size();

  for (const f of fs.readdirSync(dir)) {
    // latest-per-station files only: {wmo}.csv, not {wmo}_{stamp}.csv
    const m = /^(\d+)\.csv$/.exec(f);
    if (!m) continue;
    let prof;
    try { prof = parseCSV(fs.readFileSync(path.join(dir, f), "utf8")); } catch (e) { continue; }
    if (!prof) continue;
    sanitize(prof);

    const N = prof.P.length, F = 4;
    const ptrs = ["P", "H", "T", "D", "U", "V"].map(k => {
      const p = M._malloc(N * F);
      M.HEAPF32.set(new Float32Array(prof[k]), p / F);
      return p;
    });
    const out = M._malloc(nOut * F);
    const a = M._malloc(N * F), b = M._malloc(N * F), c = M._malloc(N * F);
    try {
      const rc = M._compute_sounding(...ptrs, N, out, a, b, c);
      if (rc === 0) {
        const ec = M.HEAPF32[out / F + 42];
        // a valid non-convective sounding has ECAPE of ZERO, not "missing" —
        // the climatology counts those zeros, so the live value must too
        if (isFinite(ec)) console.log(`${m[1]} ${(ec > MISSING ? ec : 0).toFixed(1)}`);
      }
    } catch (e) { /* one bad sounding must not kill the run */ }
    [...ptrs, out, a, b, c].forEach(p => M._free(p));
  }
})();
