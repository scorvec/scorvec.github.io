/* What-if engine (2026-10-03): re-runs the forecast's simulation in the browser with the reader's assumptions.
   Data: data/whatif.json (midterms/whatif.py). The draws are generated ONCE from a fixed seed, so moving a slider changes the
   answer only through the assumption, never through new random noise. Error structure as the model's:
     House    margin = mu + nat*s_nat + state[s]*s_state + urban*u + groups + res*own      (model.simulate)
     Senate/  margin = mu + shared*state_miss*el + groups + wnc*z + res*sd                  (senate2026.simulate_senate)
     governor
   The House national draw and the statewide draw are correlated rho (0.3).
   Settings (all default to the published forecast):
     E      national mood (House margin, pts)            shifts House 1:1, Senate x slope, governors x slope
     miss   polls overstate Democrats by this many pts    re-blends every polled race with each poll moved by -miss
     size   multiplier on every error term                0.5 - 2
     corr   0 independent .. 1 model .. 2 all together    moves variance between shared and race-own terms, total kept
     pollw  multiplier on the polls' weight vs the prior  0 (fundamentals only) .. 1 (model) .. 4
     trust  gamma for the pollster track record           weight x exp(-gamma*score), clipped 0.5-2
     picks  {key: 'D'|'R'} races called by the reader     statistics over the simulations that agree with every call */
(function (G) {
  "use strict";
  function rng(seed) { let a = seed >>> 0; return function () { a = (a + 0x6D2B79F5) >>> 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  function normals(r) { let spare = null; return function () { if (spare !== null) { const s = spare; spare = null; return s; } let u, v, q; do { u = 2 * r() - 1; v = 2 * r() - 1; q = u * u + v * v; } while (q >= 1 || q === 0); const f = Math.sqrt(-2 * Math.log(q) / q); spare = v * f; return u * f; }; }
  function tdraw(nrm, df) { const sc = Math.sqrt((df - 2) / df); return function () { let c = 0; for (let k = 0; k < df; k++) { const z = nrm(); c += z * z; } return nrm() / Math.sqrt(c / df) * sc; }; }

  function blend(bl, sys, o) {
    // precision-weighted blend of the fundamentals prior and the race polls (model: Student-t; this reproduces it to ~0.1 pt)
    let sw = 0, sy = 0;
    for (const p of bl.polls) {
      const y = p[0] - o.miss, sd = p[1], disc = p[2] == null ? 1 : p[2], sc = p[3] || 0;
      let w = disc / (sd * sd);
      if (o.trust) w *= Math.min(2, Math.max(0.5, Math.exp(-o.trust * sc)));
      sw += w; sy += w * y;
    }
    if (!(sw > 0)) return bl.prior;
    const avg = sy / sw, pw = o.pollw * 1 / (1 / sw + sys * sys), wp = 1 / (bl.prior_sd * bl.prior_sd);
    return (wp * bl.prior + pw * avg) / (wp + pw);
  }

  function Engine(D, N) {
    this.D = D; this.N = N = N || 5000;
    const r = rng(20261103), nrm = normals(r), t = tdraw(nrm, D.t_df || 5), H = D.house, S = D.senate, Gv = D.governor;
    this.nH = H.seats.length; this.nS = S.races.length; this.nG = Gv ? Gv.races.length : 0;
    const F = (n) => new Float32Array(n);
    // House draws
    this.hNat = F(N); this.hSt = F(N * H.states.length); this.hUrb = F(N); this.hG = [F(N), F(N), F(N), F(N)]; this.hRes = F(N * this.nH);
    // statewide draws (Senate + governors share them, as in the model)
    this.sNat = F(N); this.sG = [F(N), F(N), F(N)]; this.sW = F(N); this.sRes = F(N * (this.nS + this.nG));
    const rho = D.shared.rho, k2 = Math.sqrt(1 - rho * rho);
    for (let i = 0; i < N; i++) {
      this.hNat[i] = t(); this.sNat[i] = rho * this.hNat[i] + k2 * t();
      this.hUrb[i] = t(); for (const a of this.hG) a[i] = t();
      for (const a of this.sG) a[i] = t(); this.sW[i] = t();
    }
    for (let i = 0; i < this.hSt.length; i++) this.hSt[i] = t();
    for (let i = 0; i < this.hRes.length; i++) this.hRes[i] = t();
    for (let i = 0; i < this.sRes.length; i++) this.sRes[i] = t();
    // per-race error components at the model's settings (sd of each piece)
    const hs = H.sd;
    this.hPart = H.seats.map(s => ({ shared: [hs.nat, hs.state, hs.urban * s.u, hs.hisp * s.h, hs.cuban * s.c, hs.asian * s.a, hs.wnc * s.w], own: s.res }));
    const sh = D.shared, gs = sh.group_sd;
    const sw = (x) => ({ shared: [sh.state_miss * x.el, gs.hisp * x.h, gs.cuban * x.c, gs.asian * x.a, sh.wnc_sd * x.z], own: x.sd });
    this.sPart = S.races.map(sw); this.gPart = Gv ? Gv.races.map(sw) : [];
    const frac = (parts) => { let a = 0, n = 0; for (const p of parts) { const v = p.shared.reduce((q, x) => q + x * x, 0); a += v / (v + p.own * p.own); n++; } return a / n; };
    this.c0H = frac(this.hPart); this.c0S = frac(this.sPart.concat(this.gPart));
    // default blends, so a re-blend adds only its change to the published mean
    const def = { miss: 0, pollw: 1, trust: 0 };
    this.hB0 = H.seats.map(s => s.bl ? blend(s.bl, H.sys, def) : null);
    this.sB0 = S.races.map(x => x.bl ? blend(x.bl, S.sys, def) : null);
    this.gB0 = Gv ? Gv.races.map(x => x.bl ? blend(x.bl, Gv.sys, def) : null) : [];
  }

  // shared-vs-own scaling for the "do races move together?" control: u = 0 independent, 1 model, 2 all together
  function scales(u, c0) {
    const c = u <= 1 ? u * c0 : c0 + (u - 1) * (1 - c0), ks = c0 > 0 ? Math.sqrt(c / c0) : 0;
    return ks;
  }
  function ownSd(part, ks, size) {
    const vs = part.shared.reduce((q, x) => q + x * x, 0), v = vs + part.own * part.own;
    return Math.sqrt(Math.max(v - ks * ks * vs, 0)) * size;
  }

  Engine.prototype.run = function (o) {
    const D = this.D, N = this.N, H = D.house, S = D.senate, Gv = D.governor, dE = o.E - D.E0;
    const o2 = { miss: o.miss || 0, pollw: o.pollw == null ? 1 : o.pollw, trust: o.trust || 0 };
    const reblend = o2.miss !== 0 || o2.pollw !== 1 || o2.trust !== 0, size = o.size || 1;
    const ksH = scales(o.corr == null ? 1 : o.corr, this.c0H) * size, ksS = scales(o.corr == null ? 1 : o.corr, this.c0S) * size;
    const picks = o.picks || {};
    // ---- House
    const nH = this.nH, nSt = H.states.length, hMu = new Float32Array(nH), hOwn = new Float32Array(nH);
    for (let j = 0; j < nH; j++) {
      const s = H.seats[j]; let m = s.mu + dE;
      if (reblend && s.bl) m += blend(s.bl, H.sys, o2) - this.hB0[j];
      hMu[j] = m; hOwn[j] = ownSd(this.hPart[j], ksH / size, size);
    }
    // ---- Senate + governors (one statewide block)
    const nS = this.nS, nG = this.nG, races = S.races.concat(Gv ? Gv.races : []), parts = this.sPart.concat(this.gPart);
    const sMu = new Float32Array(nS + nG), sOwn = new Float32Array(nS + nG);
    for (let j = 0; j < nS + nG; j++) {
      const x = races[j], isG = j >= nS, slope = isG ? Gv.slope : S.slope, b0 = isG ? this.gB0[j - nS] : this.sB0[j];
      let m = x.mu + slope * dE;
      if (reblend && x.bl) m += blend(x.bl, isG ? Gv.sys : S.sys, o2) - b0;
      sMu[j] = m; sOwn[j] = ownSd(parts[j], ksS / size, size);
    }
    // ---- simulate, keeping only the simulations that agree with the reader's calls
    const keyH = H.seats.map(s => "H:" + s.seat), keyS = races.map((x, j) => (j < nS ? "S:" : "G:") + x.st);
    const pickH = [], pickS = [];
    keyH.forEach((k, j) => { if (picks[k]) pickH.push([j, picks[k] === "D"]); });
    keyS.forEach((k, j) => { if (picks[k]) pickS.push([j, picks[k] === "D"]); });
    const hs = H.sd, sh = D.shared, gs = sh.group_sd;
    const hWins = new Float64Array(nH), sWins = new Float64Array(nS + nG), houseSeats = [], senSeats = [], govSeats = [];
    let kept = 0, hMaj = 0, sCtl = 0, gMaj = 0, both = 0;
    const hm = new Float32Array(nH), sm = new Float32Array(nS + nG);
    for (let i = 0; i < N; i++) {
      const nat = this.hNat[i] * hs.nat * ksH, urb = this.hUrb[i] * hs.urban * ksH, g = this.hG;
      for (let j = 0; j < nH; j++) {
        const s = H.seats[j];
        if (s.fix) { hm[j] = s.fix === "D" ? 50 : -50; continue; }
        hm[j] = hMu[j] + nat + this.hSt[i * nSt + s.st] * hs.state * ksH + urb * s.u + ksH * (g[0][i] * hs.hisp * s.h + g[1][i] * hs.cuban * s.c + g[2][i] * hs.asian * s.a + g[3][i] * hs.wnc * s.w)
          + this.hRes[i * nH + j] * hOwn[j];
      }
      const shr = this.sNat[i] * sh.state_miss * ksS, sg = this.sG, wz = this.sW[i] * sh.wnc_sd * ksS;
      for (let j = 0; j < nS + nG; j++) {
        const x = races[j];
        sm[j] = sMu[j] + shr * x.el + ksS * (sg[0][i] * gs.hisp * x.h + sg[1][i] * gs.cuban * x.c + sg[2][i] * gs.asian * x.a) + wz * x.z + this.sRes[i * (nS + nG) + j] * sOwn[j];
      }
      let ok = true;
      for (const [j, d] of pickH) if ((hm[j] > 0) !== d) { ok = false; break; }
      if (ok) for (const [j, d] of pickS) if ((sm[j] > 0) !== d) { ok = false; break; }
      if (!ok) continue;
      kept++;
      let hd = 0; for (let j = 0; j < nH; j++) if (hm[j] > 0) { hd++; hWins[j]++; }
      let sd = S.holdD, si = 0; for (let j = 0; j < nS; j++) if (sm[j] > 0) { sWins[j]++; if (races[j].ind) si++; else sd++; }
      let gd = Gv ? Gv.holdD : 0; for (let j = nS; j < nS + nG; j++) if (sm[j] > 0) { sWins[j]++; gd++; }
      const sr = 100 - sd - si, ctl = sd > sr, maj = hd >= H.need;      // Senate: independents abstain, the VP breaks a tie (R)
      hMaj += maj; sCtl += ctl; both += maj && ctl; if (Gv) gMaj += gd >= Gv.need;
      houseSeats.push(hd); senSeats.push(sd); if (Gv) govSeats.push(gd);
    }
    const pct = (a, q) => { if (!a.length) return null; const b = a.slice().sort((x, y) => x - y); return b[Math.min(b.length - 1, Math.floor(q * b.length))]; };
    const mean = (a) => a.length ? a.reduce((p, x) => p + x, 0) / a.length : null;
    return {
      kept, n: N,
      house: { p: kept ? hMaj / kept : null, mean: mean(houseSeats), p10: pct(houseSeats, 0.1), p90: pct(houseSeats, 0.9), seats: houseSeats, win: Array.from(hWins, w => kept ? w / kept : null), mu: hMu },
      senate: { p: kept ? sCtl / kept : null, mean: mean(senSeats), p10: pct(senSeats, 0.1), p90: pct(senSeats, 0.9), seats: senSeats, win: Array.from(sWins.slice(0, nS), w => kept ? w / kept : null), mu: sMu.slice(0, nS) },
      governor: Gv ? { p: kept ? gMaj / kept : null, mean: mean(govSeats), p10: pct(govSeats, 0.1), p90: pct(govSeats, 0.9), seats: govSeats, win: Array.from(sWins.slice(nS), w => kept ? w / kept : null), mu: sMu.slice(nS) } : null,
      both: kept ? both / kept : null
    };
  };
  G.WhatIf = { Engine, blend };
  if (typeof module !== "undefined") module.exports = G.WhatIf;
})(typeof window !== "undefined" ? window : globalThis);
