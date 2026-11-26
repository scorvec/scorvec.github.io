// Check the REAL skewt/app.js gust estimate against its Python source (gustport.features and the scikit-learn models
// of verify_gust_score.py export) on the export's parity fixture (release skewt-gust-data-v1, gust_fixture.json).
//   node gustport_jstest.js skewt/app.js skewt/gust_model.json gust_fixture.json
const fs = require("fs");
const [appPath, modelPath, casesPath] = process.argv.slice(2);

function extract(src, name) {
  const lines = src.split("\n");
  const fnRe = new RegExp("^function " + name + "\\(");
  const cRe = new RegExp("^const " + name + "\\b");
  for (let i = 0; i < lines.length; i++) {
    if (fnRe.test(lines[i])) {
      let j = i;
      while (lines[j] !== "}") j++;
      return lines.slice(i, j + 1).join("\n");
    }
    if (cRe.test(lines[i])) return lines[i];
  }
  throw new Error("not found: " + name);
}
const app = fs.readFileSync(appPath, "utf8");
const names = ["MISSING", "gustNum", "interpHagl", "meanWindAgl", "npInterp", "gustGrid", "gustBrasseur", "gustFeatures",
  "gustTrees", "gustEstimate"];
const run = new Function(names.map(n => extract(app, n)).join("\n") +
  "\nreturn { gustFeatures, gustEstimate };")();
const M = JSON.parse(fs.readFileSync(modelPath, "utf8")), cases = JSON.parse(fs.readFileSync(casesPath, "utf8"));
let worstF = 0, worstFk = "", worstP = 0, bad = 0;
const nz = v => (v === null ? NaN : v);
for (const c of cases) {
  const f = run.gustFeatures(c.prof, nz(c.pbl_pa), nz(c.dcape), nz(c.sbcape));
  for (const k of M.features) {
    const a = f[k], b = nz(c.feats[k]);
    if (Number.isNaN(a) !== Number.isNaN(b)) { bad++; console.log("missing mismatch", k, a, b); continue; }
    if (Number.isNaN(a)) continue;
    const d = Math.abs(a - b) / Math.max(1, Math.abs(b));
    if (d > worstF) { worstF = d; worstFk = k; }
  }
  const e = run.gustEstimate(f, M), py = c.pred;
  worstP = Math.max(worstP, Math.abs(e.p10 - py[0]), Math.abs(e.p50 - py[1]), Math.abs(e.p90 - py[2]));
}
console.log(`${cases.length} launches: worst feature difference ${worstF.toExponential(2)} (${worstFk}), ` +
  `worst prediction difference ${worstP.toExponential(2)} m/s, ${bad} missing-value mismatches`);
process.exit(bad === 0 && worstF < 1e-9 && worstP < 1e-6 ? 0 : 1);
