// Run the REAL skewt/app.js functions (current + the pre-c53ce78 estCeiling) on test cases so verify_fog.py
// can check its Python port against them.   node fogport_jstest.js app.js app_old.js cases.json out.json
const fs = require("fs");
const [appPath, oldPath, casesPath, outPath] = process.argv.slice(2);

function extract(src, name) {
  const lines = src.split("\n");
  const fnRe = new RegExp("^(async )?function " + name + "\\(");
  const cRe = new RegExp("^(const|let) " + name + "\\b");
  for (let i = 0; i < lines.length; i++) {
    if (fnRe.test(lines[i])) {
      let j = i; while (lines[j] !== "}") j++;
      return lines.slice(i, j + 1).join("\n");
    }
    if (cRe.test(lines[i])) {
      let j = i, depth = 0;
      for (;; j++) {
        for (const ch of lines[j]) { if ("([{".includes(ch)) depth++; else if (")]}".includes(ch)) depth--; }
        if (depth <= 0 && /;\s*(\/\/.*)?$/.test(lines[j])) break;
      }
      return lines.slice(i, j + 1).join("\n");
    }
  }
  throw new Error("not found: " + name);
}
const app = fs.readFileSync(appPath, "utf8"), old = fs.readFileSync(oldPath, "utf8");
const names = ["KT", "iv", "parseIGRA", "thin", "sanitizeHeights", "fillDewpoints", "fillWinds", "cloudLayers",
  "CEIL_NEAR_M", "estCeiling", "FOG_MODEL", "FOG_FEATS", "bulkRi", "fogModelFeatures", "fogModelProbs", "doyOf"];
let code = "let lastDoy = null, lastHourZ = 12;\n";
for (const n of names) code += extract(app, n) + "\n";
code += extract(old, "estCeiling").replace("function estCeiling(", "function estCeilingOld(") + "\n";
code += `
module.exports = function run(cases) {
  const out = [];
  for (const c of cases) {
    let prof;
    if (c.text) {
      const got = parseIGRA(c.text, c.gid, c.ymd, c.hour, c.elev);
      if (!got || got.windOnly) { out.push({ id: c.id, parsed: false }); continue; }
      prof = got.prof;
    } else prof = c.prof;
    const p = thin(prof);
    sanitizeHeights(p); fillDewpoints(p); fillWinds(p);
    const res = cloudLayers(p);
    const ecN = estCeiling(res), ecO = estCeilingOld(res);
    lastDoy = doyOf(c.ymd); lastHourZ = c.hour;
    const pr = fogModelProbs(p);
    out.push({ id: c.id, parsed: true, n: p.P.length, H: p.H, D: p.D,
      layers: res.layers.map(l => [l.base, l.top, l.maxRh]),
      newc: ecN === undefined ? "u" : ecN === null ? "null" : ecN,
      oldc: ecO === undefined ? "u" : ecO === null ? "null" : ecO,
      pfog: pr ? pr.fog : null });
  }
  return out;
};`;
const tmp = outPath + ".harness.js";
fs.writeFileSync(tmp, code);
const run = require(require("path").resolve(tmp));
fs.writeFileSync(outPath, JSON.stringify(run(JSON.parse(fs.readFileSync(casesPath, "utf8")))));
