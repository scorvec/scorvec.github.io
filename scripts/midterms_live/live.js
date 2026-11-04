// Usage: node scripts/midterms_live/live.js [obs.json]   (env: N sims, WATCH=races, JSON=out file)
// Election-night update of the published midterms forecast.
// Re-uses the model's own simulation (midterms/whatif_engine.js + data/whatif.json, same seed and error structure) and
// reweights every simulation by how well it matches the partial counts in obs.json (importance weighting, NYT-needle style).
// Because the draws share national / state / group shocks, a surprise in one race moves every other race.
//   obs.json: [{race:"H:IN-3"|"S:KY"|"G:FL", d:57.0, r:43.0, pct:31.5, skew:0, note:""}, ...,  {race:"S:VA", call:"D"}]
//     d, r   current vote shares; pct = % of expected vote counted
//     skew   how many points more Democratic the counted batch is expected to be than the final (e.g. +8 if it is mostly
//            mail/early ballots in a state where those lean D); default 0
//     skew_sd how unsure that skew is (points, default 0): added in quadrature to the count noise
//     call   an AP call: hard condition (simulations that disagree are dropped)
// Count noise: final margin ~ current margin - skew + t4 error with sd 2 + 28*(1-f)^1.5 (f = share counted) - wide early,
// tight once most of the vote is in; t tails so one odd batch cannot take over the whole forecast.
const fs = require("fs"), path = require("path");
const SITE = process.env.SITE || "/home/user/scorvec.github.io/midterms";
const WhatIf = require(path.join(SITE, "whatif_engine.js"));
const D = JSON.parse(fs.readFileSync(path.join(SITE, "data/whatif.json"), "utf8"));
const OBS = JSON.parse(fs.readFileSync(process.argv[2] || path.join(__dirname, "obs.json"), "utf8"));
const N = +(process.env.N || 20000);
const E = new WhatIf.Engine(D, N);

function countSd(f) { return 2 + 28 * Math.pow(Math.max(0, 1 - f), 1.5); }
function logT4(z) { return -2.5 * Math.log(1 + z * z / 4); }   // Student-t df 4, up to a constant

function simulate() {
  // same arithmetic as Engine.run at the published settings, but keeping every simulation's margins
  const H = D.house, S = D.senate, Gv = D.governor, mv = E.mv, nH = E.nH, nS = E.nS, nG = E.nG, nSt = H.states.length;
  const races = S.races.concat(Gv.races), parts = E.sPart.concat(E.gPart), hs = H.sd, sh = D.shared, gs = sh.group_sd;
  const own = (p) => Math.sqrt(p.shared.reduce((q, x) => q + x * x, 0) * 0 + p.own * p.own);
  const hOwn = E.hPart.map(own), sOwn = parts.map(own);
  const split = (s, m) => { const mm = Math.min(m, 0.95 * s); return { rest: Math.sqrt(s * s - mm * mm), m: mm }; };
  const sRest = races.map((x) => (mv ? split(x.ssh, mv.m).rest : x.ssh) * x.sl), sMove = races.map((x, j) => (mv ? split(x.ssh, mv.m).m : 0) * E.kS[j] * x.sl);
  const hsp = mv ? split(hs.nat, mv.m) : { rest: hs.nat, m: 0 };
  const HM = new Float32Array(N * nH), SM = new Float32Array(N * (nS + nG)), NAT = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const natRest = E.hNat[i] * hsp.rest, natMove = E.hMv[i] * hsp.m, urb = E.hUrb[i] * hs.urban, g = E.hG;
    NAT[i] = natRest + natMove;
    for (let j = 0; j < nH; j++) {
      const s = H.seats[j];
      HM[i * nH + j] = s.fix ? (s.fix === "D" ? 50 : -50) : s.mu + natRest + natMove * E.kH[j] + E.hSt[i * nSt + s.st] * hs.state + urb * s.u
        + g[0][i] * hs.hisp * s.h + g[1][i] * hs.cuban * s.c + g[2][i] * hs.asian * s.a + g[3][i] * hs.wnc * s.w
        + g[4][i] * (hs.htx || 0) * (s.htx || 0) + g[5][i] * (hs.hfl || 0) * (s.hfl || 0) + g[6][i] * (hs.hwest || 0) * (s.hwe || 0)
        + E.hRes[i * nH + j] * hOwn[j];
    }
    const sg = E.sG, wz = E.sW[i] * sh.wnc_sd;
    for (let j = 0; j < nS + nG; j++) {
      const x = races[j];
      SM[i * (nS + nG) + j] = x.mu + E.sNat[i] * sRest[j] + E.hMv[i] * sMove[j] + sg[0][i] * gs.hisp * x.h + sg[1][i] * gs.cuban * x.c + sg[2][i] * gs.asian * x.a + wz * x.z + E.sRes[i * (nS + nG) + j] * sOwn[j];
    }
  }
  return { HM, SM, NAT, races };
}

const H = D.house, S = D.senate, Gv = D.governor, nH = E.nH, nS = E.nS, nG = E.nG, nSG = nS + nG;
const { HM, SM, NAT, races } = simulate();
const idx = {};
H.seats.forEach((s, j) => (idx["H:" + s.seat] = ["H", j]));
races.forEach((x, j) => (idx[(j < nS ? "S:" : "G:") + x.st] = ["S", j]));
const margin = (i, [t, j]) => (t === "H" ? HM[i * nH + j] : SM[i * nSG + j]);

function summarize(obs) {
  const lw = new Float64Array(N), used = [];
  for (const o of obs) {
    const k = idx[o.race]; if (!k) { console.error("unknown race " + o.race); continue; }
    if (o.call) { for (let i = 0; i < N; i++) if ((margin(i, k) > 0) !== (o.call === "D")) lw[i] = -Infinity; }
    if (o.pct == null || !(o.pct > 0)) { if (o.call) used.push(o); continue; }   // a called race keeps its count too, when given
    const f = Math.min(1, o.pct / 100), m = o.d - o.r - (o.skew || 0), sd = Math.hypot(countSd(f), o.skew_sd || 0);
    for (let i = 0; i < N; i++) lw[i] += logT4((margin(i, k) - m) / sd);
    used.push(o);
  }
  let mx = -Infinity; for (const v of lw) if (v > mx) mx = v;
  const w = new Float64Array(N); let sw = 0, sw2 = 0;
  for (let i = 0; i < N; i++) { w[i] = mx === -Infinity ? 0 : Math.exp(lw[i] - mx); sw += w[i]; sw2 += w[i] * w[i]; }
  const ess = sw * sw / sw2;
  const hSeats = new Float64Array(N), sSeats = new Float64Array(N), gSeats = new Float64Array(N);
  let hMaj = 0, sCtl = 0, gMaj = 0, natM = 0;
  const hWin = new Float64Array(nH), sWin = new Float64Array(nSG), hMean = new Float64Array(nH), sMean = new Float64Array(nSG);
  for (let i = 0; i < N; i++) {
    const wi = w[i] / sw; if (!wi) continue;
    let hd = 0; for (let j = 0; j < nH; j++) { const v = HM[i * nH + j]; hMean[j] += wi * v; if (v > 0) { hWin[j] += wi; if (!H.seats[j].ind) hd++; } }
    let sd = S.holdD, si = 0; for (let j = 0; j < nS; j++) { const v = SM[i * nSG + j]; sMean[j] += wi * v; if (v > 0) { sWin[j] += wi; races[j].ind ? si++ : sd++; } }
    let gd = Gv.holdD; for (let j = nS; j < nSG; j++) { const v = SM[i * nSG + j]; sMean[j] += wi * v; if (v > 0) { sWin[j] += wi; gd++; } }
    hSeats[i] = hd; sSeats[i] = sd; gSeats[i] = gd;
    hMaj += wi * (hd >= H.need); sCtl += wi * (sd > 100 - sd - si); gMaj += wi * (gd >= Gv.need); natM += wi * NAT[i];
  }
  const q = (a, p) => { const o = Array.from(a.keys()).filter(i => w[i] > 0).sort((x, y) => a[x] - a[y]); let c = 0; for (const i of o) { c += w[i] / sw; if (c >= p) return a[i]; } return a[o[o.length - 1]]; };
  const wmean = (a) => { let s = 0; for (let i = 0; i < N; i++) s += w[i] / sw * a[i]; return s; };
  return { used, ess, nat: natM, house: { p: hMaj, mean: wmean(hSeats), p10: q(hSeats, .1), p90: q(hSeats, .9) }, senate: { p: sCtl, mean: wmean(sSeats), p10: q(sSeats, .1), p90: q(sSeats, .9) },
           gov: { p: gMaj, mean: wmean(gSeats) }, hWin, sWin, hMean, sMean };
}

const base = summarize([]), now = summarize(OBS);
const pc = (x) => (100 * x).toFixed(1) + "%", sg = (x) => (x >= 0 ? "D+" : "R+") + Math.abs(x).toFixed(1);
console.log(`simulations ${N}, effective after weighting ${now.ess.toFixed(0)}  (observations used: ${now.used.length})`);
console.log(`national environment: ${sg(D.E0 + base.nat)} -> ${sg(D.E0 + now.nat)}  (House-vote margin)`);
console.log(`House   P(D) ${pc(base.house.p)} -> ${pc(now.house.p)}   D seats ${base.house.mean.toFixed(1)} -> ${now.house.mean.toFixed(1)} (80% ${now.house.p10}-${now.house.p90})`);
console.log(`Senate  P(D) ${pc(base.senate.p)} -> ${pc(now.senate.p)}   D seats ${base.senate.mean.toFixed(1)} -> ${now.senate.mean.toFixed(1)} (80% ${now.senate.p10}-${now.senate.p90})`);
console.log(`Govs    D mean ${base.gov.mean.toFixed(1)} -> ${now.gov.mean.toFixed(1)}`);
const watch = new Set(OBS.map(o => o.race).concat((process.env.WATCH || "S:FL,G:FL,S:GA,S:NC,S:ME,S:OH,S:TX,S:IA,S:MI,S:AK,S:SC,H:KY-6,H:IN-1,H:PA-1,H:TX-15").split(",")));
console.log("\nrace        model margin -> now      P(D win)");
for (const r of watch) { const k = idx[r]; if (!k) continue; const [t, j] = k;
  const b = t === "H" ? [base.hMean[j], base.hWin[j]] : [base.sMean[j], base.sWin[j]], n = t === "H" ? [now.hMean[j], now.hWin[j]] : [now.sMean[j], now.sWin[j]];
  const o = OBS.find(x => x.race === r), tag = o ? (o.call ? ` [called ${o.call}]` : ` [${(o.d - o.r >= 0 ? "D+" : "R+") + Math.abs(o.d - o.r).toFixed(1)} at ${o.pct}%]`) : "";
  console.log(`${r.padEnd(10)} ${sg(b[0]).padStart(7)} -> ${sg(n[0]).padStart(7)}   ${pc(b[1]).padStart(6)} -> ${pc(n[1]).padStart(6)}${tag}`); }
if (process.env.OUT) writeOut(process.env.OUT);
function writeOut(dir) {
  // dashboard files: odds.json (history, one row per run, the published forecast first), races.json (key races)
  fs.mkdirSync(dir, { recursive: true });
  const at = process.env.AT || new Date().toISOString().slice(0, 16) + "Z";
  const row = (lab, r) => ({ at: lab, nat: +(D.E0 + r.nat).toFixed(2), house_p: +r.house.p.toFixed(4), senate_p: +r.senate.p.toFixed(4),
    house_mean: +r.house.mean.toFixed(1), house_p10: r.house.p10, house_p90: r.house.p90, senate_mean: +r.senate.mean.toFixed(2),
    senate_p10: r.senate.p10, senate_p90: r.senate.p90, gov_mean: +r.gov.mean.toFixed(1), ess: Math.round(r.ess), n_obs: r.used.length });
  const hp = path.join(__dirname, "history.json");
  let hist = []; try { hist = JSON.parse(fs.readFileSync(hp, "utf8")); } catch (e) {}
  if (!hist.length) hist.push(row("2026-11-03T17:04Z", base));
  hist = hist.filter(h => h.at !== at); hist.push(row(at, now));
  fs.writeFileSync(hp, JSON.stringify(hist, null, 1));
  fs.writeFileSync(path.join(dir, "odds.json"), JSON.stringify(hist));
  const out = [];
  for (const [key, [t, j]] of Object.entries(idx)) {
    const isH = t === "H", x = isH ? H.seats[j] : races[j], kind = key[0] === "H" ? "House" : key[0] === "S" ? "Senate" : "Governor";
    const bp = isH ? base.hWin[j] : base.sWin[j], np = isH ? now.hWin[j] : now.sWin[j];
    const o = OBS.find(q => q.race === key) || null, st = isH ? x.seat.split("-")[0] : x.st;
    const keep = o || kind === "Senate" || st === "FL" ? true : (bp > 0.08 && bp < 0.92) || (np > 0.08 && np < 0.92);
    if (!keep || (isH && x.fix)) continue;
    out.push({ race: key, kind, state: st, label: isH ? x.seat : kind + " " + st + (x.sp ? " (special)" : ""),
      dem: (o && o.dn) || (isH ? "" : x.d || ""), rep: (o && o.rn) || (isH ? "" : x.r || ""),
      model_margin: +(isH ? base.hMean[j] : base.sMean[j]).toFixed(1), now_margin: +(isH ? now.hMean[j] : now.sMean[j]).toFixed(1),
      model_p: +bp.toFixed(4), now_p: +np.toFixed(4),
      d_share: o && o.d != null ? o.d : null, r_share: o && o.d != null ? o.r : null, pct_in: o && o.pct != null ? o.pct : null, call: o && o.call ? o.call : "" });
  }
  out.sort((a, b) => Math.abs(b.now_p - b.model_p) - Math.abs(a.now_p - a.model_p));
  fs.writeFileSync(path.join(dir, "races.json"), JSON.stringify(out));
  console.log(`wrote ${dir}/odds.json (${hist.length} rows) and races.json (${out.length} races)`);
}
if (process.env.JSON) fs.writeFileSync(process.env.JSON, JSON.stringify({ at: new Date().toISOString(), ess: now.ess, nat: D.E0 + now.nat, house: now.house, senate: now.senate, gov: now.gov }, null, 1));
