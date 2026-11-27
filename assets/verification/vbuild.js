/* The reader's own plots on verification.html (views "Plot builder" and "Your own region"), drawn with vcharts.js.

   window.vbuild.builder(ctx, single, ens, figs): a chart composed from the score tables the page has loaded
   (scores_<runs>.json, ens_scores_<runs>.json): any score of any variable and region, by forecast day or by pressure,
   one line a model or one a region, a score itself or its difference from a reference model. A difference carries the
   tables' own paired test, drawn filled where it passes Benjamini-Hochberg at 10 % over the chart; a bias is drawn from
   the night's lead and profile files (verify_figs.py), which hold its interval, and filled where that leaves out zero.

   window.vbuild.region(ctx, manifest): a box drawn on a map, its errors summed from the stations' errors run by run on
   2.5 degree cells (scripts/verify/region_cells.py): the box's bias and RMSE by the date the forecasts verify, or by
   forecast day with each model's RMSE against a reference. The tests are the pipeline's, written again here: a mean
   bias by verify_figs.mean_test (t on the run series, the variance inflated for the lag-1 autocorrelation pooled over
   the models, the runs less one degrees of freedom) and an RMSE difference by mverify.significance.paired_test (t on
   the runs' differences of mean squared error, the runs counted as n (1 - r) / (1 + r) for the differences' own lag-1
   autocorrelation r), each through Benjamini-Hochberg at 10 % over the chart.

   Both give a link that opens the same plot (the query string of verification.html) and save it as vcharts saves. */
(function () {
  "use strict";
  const Q = 0.10;   // the false-discovery rate of every test drawn

  // ── statistics ────────────────────────────────────────────────────────────────────────────────────────────────
  function lgamma(x) {   // Lanczos (Numerical Recipes gammln), x > 0
    const c = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
    let y = x, tmp = x + 5.5, ser = 1.000000000190015;
    tmp -= (x + 0.5) * Math.log(tmp);
    for (const k of c) ser += k / ++y;
    return -tmp + Math.log((2.5066282746310005 * ser) / x);
  }
  function betacf(a, b, x) {   // the continued fraction of the incomplete beta function
    const TINY = 1e-300;
    let c = 1, d = 1 - ((a + b) * x) / (a + 1);
    d = 1 / (Math.abs(d) < TINY ? TINY : d);
    let h = d;
    for (let m = 1; m <= 300; m++) {
      for (const aa of [(m * (b - m) * x) / ((a - 1 + 2 * m) * (a + 2 * m)), (-(a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 1 + 2 * m))]) {
        d = 1 + aa * d; d = 1 / (Math.abs(d) < TINY ? TINY : d);
        c = 1 + aa / c; if (Math.abs(c) < TINY) c = TINY;
        h *= d * c;
      }
      if (Math.abs(d * c - 1) < 3e-14) break;
    }
    return h;
  }
  function ibeta(a, b, x) {   // the regularized incomplete beta function
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    const bt = Math.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * Math.log(x) + b * Math.log(1 - x));
    return x < (a + 1) / (a + b + 2) ? (bt * betacf(a, b, x)) / a : 1 - (bt * betacf(b, a, 1 - x)) / b;
  }
  const tTwo = (t, df) => ibeta(df / 2, 0.5, df / (df + t * t));   // two-sided p of |t| on df degrees of freedom
  function tQuant(level, df) {   // the t quantile of a two-sided interval at `level`
    let lo = 0, hi = 1e3;
    for (let k = 0; k < 80; k++) { const mid = (lo + hi) / 2; if (tTwo(mid, df) > 1 - level) lo = mid; else hi = mid; }
    return (lo + hi) / 2;
  }
  const ok = v => v != null && Number.isFinite(v);
  const mean = xs => xs.reduce((s, v) => s + v, 0) / xs.length;
  const sd = xs => { const m = mean(xs); return Math.sqrt(xs.reduce((s, v) => s + (v - m) ** 2, 0) / (xs.length - 1)); };
  const nEff = (n, rho) => (n * (1 - rho)) / (1 + rho);
  // a series' lag-1 autocorrelation in run order, a gap (null) breaking the pairs it falls in (verify_figs.lag1_rows)
  function lag1(x) {
    const have = x.filter(ok);
    if (!have.length) return NaN;
    const m = mean(have);
    let num = 0, den = 0;
    x.forEach((v, i) => { if (!ok(v)) return; den += (v - m) ** 2; if (i && ok(x[i - 1])) num += (v - m) * (x[i - 1] - m); });
    return den > 0 ? num / den : NaN;
  }
  // verify_figs.mean_test: each series' mean, its 95 % interval and p, from `minN` runs; one rho for them all
  function meanTest(rows, minN) {
    const k = rows.map(r => r.filter(ok).length), on = k.map(n => n >= minN);
    const rs = rows.filter((_, i) => on[i]).map(lag1).filter(Number.isFinite), kk = k.filter((_, i) => on[i]);
    const runs = kk.length ? mean(kk) : 0;
    const rho = runs > 3 && rs.length ? Math.min(0.9, Math.max(0, (mean(rs) + 1 / runs) / (1 - 3 / runs))) : 0;
    return rows.map((r, i) => {
      if (!on[i]) return null;
      const x = r.filter(ok), m = mean(x), se = sd(x) / Math.sqrt(Math.max(nEff(x.length, rho), 1)), h = tQuant(0.95, x.length - 1) * se;
      return { mean: m, lo: m - h, hi: m + h, p: se > 0 ? tTwo(Math.abs(m) / se, x.length - 1) : m ? 0 : 1, n: x.length };
    });
  }
  // mverify.significance.paired_test on the runs' differences d
  function pairedTest(d) {
    d = d.filter(ok);
    const n = d.length;
    if (n < 3) return null;
    const s = sd(d), m = mean(d);
    if (!(s > 0)) return m ? 0 : 1;
    let ne = n;
    if (n >= 4) { const r = Math.min(0.9, Math.max(0, lag1(d) || 0)); ne = nEff(n, r); }
    return tTwo(Math.abs(m) / (s / Math.sqrt(ne)), Math.max(ne - 1, 1));
  }
  // Benjamini-Hochberg: the largest p that passes at Q among those given (-1: none passes)
  function cutOf(ps) {
    const s = ps.filter(ok).sort((a, b) => a - b);
    let c = -1;
    s.forEach((p, i) => { if (p <= (Q * (i + 1)) / s.length) c = p; });
    return c;
  }
  // a running mean over `days` centred, from more than half of them (verify_figs SMOOTH_DAYS)
  function running(x, days) {
    if (days <= 1) return x.slice();
    const h = Math.floor(days / 2), need = h + 1;
    return x.map((_, i) => { const w = x.slice(Math.max(0, i - h), i + h + 1).filter(ok); return w.length >= need ? mean(w) : null; });
  }

  // ── shared: the link to a plot, menus that keep their buttons, a row of on/off buttons ───────────────────────
  const asked = new URLSearchParams(location.search);
  function linkTo(params, view) {
    const q = new URLSearchParams(location.search);
    [...q.keys()].filter(k => /^[br]\./.test(k)).forEach(k => q.delete(k));
    Object.entries(params).forEach(([k, v]) => { if (v != null && v !== "") q.set(k, v); });
    return `${location.origin}${location.pathname}?${q}#${view}`;
  }
  // a menu filled only when its choices change, so its buttons (and the keyboard's place among them) stay
  function menus(fill) {
    return (sel, values, labels) => {
      const now = values.map(v => `${v}\u0001${labels ? labels(v) : v}`).join("\u0002");
      if (sel.dataset.opts !== now) { sel.dataset.opts = now; fill(sel, values, labels, true); }
    };
  }
  const wanted = (sel, key) => { const w = asked.get(key); if (w != null && [...sel.options].some(o => o.value === w)) sel.value = w; };
  // buttons that switch on and off (one always stays on); returns what is on
  function toggles(box, items, on, changed) {
    box.innerHTML = "";
    const keep = new Set(on.filter(v => items.some(([k]) => k === v)));
    if (!keep.size) items.forEach(([k]) => keep.add(k));
    items.forEach(([k, label]) => {
      const b = document.createElement("button");
      b.type = "button"; b.textContent = label; b.setAttribute("aria-pressed", String(keep.has(k)));
      b.onclick = () => {
        if (keep.has(k) && keep.size === 1) return;
        if (keep.has(k)) keep.delete(k); else keep.add(k);
        b.setAttribute("aria-pressed", String(keep.has(k)));
        changed();
      };
      box.append(b);
    });
    return () => items.map(([k]) => k).filter(k => keep.has(k));
  }
  const REGION_COLORS = ["#2c5f8a", "#b3372f", "#2e7d4f", "#7b4fb0", "#8a6a1f", "#4a4744", "#b0487f"];
  const SHAPES = ["circle", "square", "triangle", "diamond"];

  // ── the plot builder ─────────────────────────────────────────────────────────────────────────────────────────
  const SURF = ["t2m", "td2m", "msl", "wind10", "ws10", "u10", "v10", "pr24", "tp"];
  const QNAME = { z: "Height", t: "Temperature", wind: "Vector wind", q: "Specific humidity", u: "West–east wind", v: "South–north wind" };
  const level = v => { const m = /^(z|t|wind|q|u|v)(\d+)$/.exec(v); return m && !SURF.includes(v) ? [m[1], +m[2]] : null; };
  // a kind's scores: [key, the menu's name, the chart's (ref: the reference model), a floor or ceiling of its axis]
  // key "<score>:d" a difference from the reference, ":p" the same in %, ":g" the lead gained on it
  const SCORES = {
    single: [["rmse", "RMSE", () => "RMSE", { floor: 0 }], ["bias", "Bias", () => "Bias", {}], ["mae", "Mean absolute error", () => "Mean absolute error", { floor: 0 }],
      ["corr", "Correlation", () => "Correlation with the observations", { ceil: 1 }],
      ["rmse:p", "RMSE against a reference (%)", r => `RMSE against ${r} (%)`, {}], ["rmse:d", "RMSE less a reference's", r => `RMSE less ${r}'s`, {}],
      ["rmse:g", "Days gained on a reference", r => `Days of lead gained on ${r}, at equal RMSE`, {}]],
    ens: [["crps", "CRPS", () => "CRPS", { floor: 0 }], ["rmse", "RMSE of the mean", () => "RMSE of the ensemble mean", { floor: 0 }],
      ["spread", "Spread", () => "Spread of the members", { floor: 0 }], ["ssr", "Spread over error", () => "Spread over the error of the mean", {}],
      ["crps:p", "CRPS against a reference (%)", r => `CRPS against ${r} (%)`, {}], ["crps:d", "CRPS less a reference's", r => `CRPS less ${r}'s`, {}],
      ["crps:g", "Days gained on a reference", r => `Days of lead gained on ${r}, at equal CRPS`, {}]],
  };
  const UNITLESS = new Set(["corr", "ssr"]);
  const KINDS = { single: "Single runs", ens: "Ensembles" }, REF = { single: "IFS", ens: "IFS-ENS" };
  const REGION_ORDER = ["Global", "NH 20-90N", "Tropics 20S-20N", "SH 20-90S", "North America", "CONUS", "Europe"];
  const brief = g => g.replace(/ \d.*$/, "");

  function builder(ctx, single, ens, figs) {
    const { $, fill, who, ordered, unit, ymd } = ctx, tables = {};
    if (single) tables.single = single;
    if (ens) tables.ens = ens;
    if (!Object.keys(tables).length) return;
    const set = menus(fill);
    const bk = $("pb-kind"), bv = $("pb-var"), bl = $("pb-level"), bs = $("pb-score"), br = $("pb-ref"), bx = $("pb-x"), bd = $("pb-day"),
      bt = $("pb-times"), bn = $("pb-lines"), bg = $("pb-region"), bm = $("pb-model"), pick = $("pb-pick");
    const show = (id, on) => { $(id).hidden = !on; };
    const rows = Object.fromEntries(Object.entries(tables).map(([k, t]) => [k, Object.fromEntries(t.rows.map(r => [`${r.var}|${r.region}`, r]))]));
    const varsOf = k => [...new Set(tables[k].rows.map(r => r.var))];
    const levelsOf = (k, q) => [...new Set(varsOf(k).map(level).filter(x => x && x[0] === q).map(x => x[1]))].sort((x, y) => y - x);
    // the variable menu: the surface variables, then each upper-air quantity (its level is its own menu)
    const menuOf = k => { const vs = varsOf(k); return [...SURF.filter(v => vs.includes(v)), ...Object.keys(QNAME).filter(q => vs.some(v => level(v)?.[0] === q)).map(q => `q:${q}`)]; };
    const nameOf = (k, v) => v.startsWith("q:") ? QNAME[v.slice(2)] : (tables[k].rows.find(r => r.var === v)?.label || v).replace(/^./, c => c.toUpperCase());
    const regionsOf = k => [...new Set(tables[k].rows.map(r => r.region))].sort((x, y) => REGION_ORDER.indexOf(x) - REGION_ORDER.indexOf(y));
    const files = (figs && figs.scores && figs.scores[ctx.W] && figs.scores[ctx.W].data) || {};   // the bias's intervals and tests
    let picks = () => (asked.get("b.p") || "").split("~").filter(Boolean), picksFor = "";

    // the menus for what is chosen so far; returns the choice
    function controls() {
      const k = bk.value, t = tables[k];
      set(bv, menuOf(k), v => nameOf(k, v));
      const q = bv.value.startsWith("q:") ? bv.value.slice(2) : null, levels = q ? levelsOf(k, q) : [];
      set(bx, levels.length > 2 ? ["lead", "level"] : ["lead"], x => x === "lead" ? "Forecast day" : "Pressure");
      const byLevel = bx.value === "level";
      show("pb-x-l", levels.length > 2);
      show("pb-level-l", !!q && !byLevel);
      if (q) {   // the level kept from the last quantity where this one has it, else 500 hPa
        const before = +bl.value;
        set(bl, levels.map(String), p => `${p} hPa`);
        if (!levels.includes(before)) bl.value = String(levels.includes(500) ? 500 : levels[0]);
      }
      const v = q ? `${q}${bl.value}` : bv.value;
      const biasFile = k === "single" && (byLevel ? files.profile?.[q] : files.lead?.[v]);
      const scores = SCORES[k].filter(([sc]) => (sc !== "bias" || biasFile) && !(byLevel && sc.endsWith(":g")));
      set(bs, scores.map(x => x[0]), sc => scores.find(x => x[0] === sc)[1]);
      const [score, how] = bs.value.split(":"), models = ordered(t.models);
      set(br, models);
      show("pb-ref-l", !!how);
      const days = t.leads.filter(h => h > 0 && h % 24 === 0).map(h => String(h / 24));
      set(bd, days, d => `Day ${d}`);
      show("pb-day-l", byLevel);
      set(bt, ["z00", "z12", "all"], x => ({ z00: "Valid 00 UTC", z12: "Valid 12 UTC", all: "Every lead" })[x]);
      show("pb-times-l", !byLevel);
      set(bn, ["models", "regions"], n => n === "models" ? "A line a model" : "A line a region");
      const lines = bn.value, regions = regionsOf(k);
      set(bg, regions);
      set(bm, models);
      show("pb-region-l", lines === "models");
      show("pb-model-l", lines === "regions");
      $("pb-pick-name").textContent = lines === "models" ? "Models" : "Regions";
      const items = lines === "models" ? models.filter(m => !how || m !== br.value).map(m => [m, m]) : regions.map(g => [g, brief(g)]);
      const key = `${k}|${lines}|${items.map(x => x[0]).join()}`;
      if (picksFor !== key) { picksFor = key; picks = toggles(pick, items, picks(), render); }
      return { k, t, q, v, byLevel, score, how, biasFile, lines };
    }
    // one line's points { u, v, lo, hi, p }: a score from the table, a difference with the table's paired test, a bias
    // from the night's lead or profile file (its interval and test)
    async function line(c, model, region) {
      const { k, t, score, how } = c, ref = br.value, keep = h => h > 0 && (bt.value === "all" || h % 24 === (bt.value === "z12" ? 12 : 0));
      if (score === "bias") {
        const d = await ctx.figJson(c.biasFile);
        if (c.byLevel) {
          const at = d?.regions?.[region]?.[bd.value], sc = at?.models?.[model];
          return sc ? at.levels.map((p, i) => ({ u: p, v: sc.bias?.[i] ?? null, lo: sc.blo?.[i] ?? null, hi: sc.bhi?.[i] ?? null, p: sc.bp?.[i] ?? null })) : [];
        }
        const reg = d?.regions?.[region], sc = reg?.models?.[model];
        return sc ? reg.leads.map((h, i) => ({ h, u: h / 24, v: sc.bias?.[i] ?? null, lo: sc.blo?.[i] ?? null, hi: sc.bhi?.[i] ?? null, p: sc.bp?.[i] ?? null })).filter(x => keep(x.h)) : [];
      }
      const val = (row, m, i) => row?.scores?.[m]?.[score]?.[i] ?? null;
      const pOf = (row, i) => row.p?.[`${model}|${ref}`]?.[i] ?? row.p?.[`${ref}|${model}`]?.[i] ?? null;
      const one = (row, i) => {
        const a = val(row, model, i);
        if (a == null) return null;
        if (!how) return { v: a };
        const b = val(row, ref, i);
        if (b == null || model === ref) return null;
        if (how === "g") {   // this lead less the reference's lead with the same error, among its leads of the same hour
          const h = t.leads[i], same = t.leads.map((x, j) => [x, val(row, ref, j)]).filter(([x, e]) => x > 0 && x % 24 === h % 24 && e != null);
          for (let j = 1; j < same.length; j++) {
            const [h0, e0] = same[j - 1], [h1, e1] = same[j];
            if (e1 > e0 && a >= e0 && a <= e1) return { v: (h - h0 - ((a - e0) / (e1 - e0)) * (h1 - h0)) / 24, p: pOf(row, i) };
          }
          return null;   // beyond the reference's own range of errors
        }
        return { v: how === "d" ? a - b : (100 * (a - b)) / b, p: pOf(row, i) };
      };
      if (c.byLevel) {
        const i = t.leads.indexOf(24 * +bd.value);
        return levelsOf(k, c.q).map(p => { const x = one(rows[k][`${c.q}${p}|${region}`], i); return x && { u: p, ...x }; }).filter(Boolean);
      }
      const row = rows[k][`${c.v}|${region}`];
      return row ? t.leads.map((h, i) => { const x = keep(h) && one(row, i); return x && { u: h / 24, ...x }; }).filter(Boolean) : [];
    }
    let ticket = 0;
    async function render() {
      const c = controls(), mine = ++ticket, { k, t, score, how, lines } = c, host = $("pb-fig");
      const sets = lines === "models" ? picks().map(m => [m, bg.value]) : picks().map(g => [bm.value, g]);
      const got = await Promise.all(sets.map(([m, g]) => line(c, m, g)));
      if (mine !== ticket) return;
      // every test the chart shows through one Benjamini-Hochberg cut
      const cut = cutOf(got.flat().map(q => q.p));
      const series = sets.map(([m, g], j) => ({
        ...(lines === "models" ? who(m) : { name: brief(g), color: REGION_COLORS[j % REGION_COLORS.length], shape: SHAPES[j % SHAPES.length] }),
        pts: got[j].map(q => ({ u: q.u, v: q.v, lo: q.lo ?? null, hi: q.hi ?? null, ...("p" in q ? { on: q.p != null && cut >= 0 && q.p <= cut } : {}) })),
      })).filter(sr => sr.pts.some(q => q.v != null));
      if (!series.length) { window.vcharts.clear(host, "Nothing scored for this choice yet."); $("pb-note").textContent = ""; return; }
      const spec = SCORES[k].find(x => x[0] === bs.value), row = t.rows.find(r => r.var === (c.byLevel ? `${c.q}${levelsOf(k, c.q)[0]}` : c.v));
      const u = UNITLESS.has(score) || how === "p" || how === "g" ? "" : ` (${unit(row?.unit || "")})`;
      const what = c.byLevel ? QNAME[c.q].toLowerCase() : (row?.label || c.v);
      const title = `${spec[2](br.value)}${u}: ${what}${c.byLevel ? `, day ${bd.value}` : ""}, ${lines === "models" ? bg.value : `${bm.value} by region`}`;
      const tests = how || score === "bias", refs = tests ? [{ v: 0 }] : score === "ssr" ? [{ v: 1, label: "spread matches error" }] : [];
      const us = [...new Set(series.flatMap(sr => sr.pts.map(q => q.u)))].sort((x, y) => x - y);
      const panel = c.byLevel
        ? { title, flip: true, dodge: 4, v: spec[3], refs, series, u: { type: "log", ticks: us.map(p => ({ u: p, label: String(p) })), fmt: p => `${p} hPa` } }
        : { title, dodge: us.length > 16 ? 0 : 4, markers: us.length <= 16 || !!tests, v: spec[3], refs, series,
            u: { label: "Forecast day", ticks: [0, ...Array.from({ length: Math.ceil(us[us.length - 1]) }, (_, i) => i + 1)], fmt: d => String(+d.toFixed(2)) } };
      const keys = [];
      if (tests) keys.push(`${window.vcharts.pair()}filled where the test passes (false discovery rate 10% over the chart)`);
      if (score === "bias") keys.push("whiskers: 95% intervals");
      if (how === "g") keys.push("above zero: ahead of the reference");
      window.vcharts.draw(host, { label: title, panels: [panel], cols: 1, height: c.byLevel ? 460 : 400, keys, data: false,
        legend: series.map(sr => ({ name: sr.name, color: sr.color, shape: sr.shape })),
        share: () => linkTo({ "b.k": k, "b.v": bv.value, "b.l": bl.value, "b.s": bs.value, "b.r": br.value, "b.x": bx.value, "b.d": bd.value, "b.t": bt.value,
          "b.n": bn.value, "b.g": bg.value, "b.m": bm.value, "b.p": picks().join("~") }, "builder") });
      const inits = Object.values(t.inits).flat().map(x => x.replace(/\D/g, "")).sort();
      $("pb-note").textContent = `${KINDS[k]}, the window's ${Math.max(...Object.values(t.inits).map(x => x.length))} runs from ${ymd(inits[0])} to ${ymd(inits[inits.length - 1])}` +
        `${row?.truth ? `, against ${row.truth}` : ""}.`;
    }
    // the page's own first plot, or a link's: each menu in the order the ones after it depend on
    fill(bk, Object.keys(tables), x => KINDS[x]);
    wanted(bk, "b.k");
    controls();
    bv.value = menuOf(bk.value).includes("t2m") ? "t2m" : bv.value;
    br.value = tables[bk.value].models.includes(REF[bk.value]) ? REF[bk.value] : br.value;
    bg.value = regionsOf(bk.value).includes("Global") ? "Global" : bg.value;
    bd.value = [...bd.options].some(o => o.value === "5") ? "5" : bd.value;
    for (const [sel, key] of [[bv, "b.v"], [bx, "b.x"], [bl, "b.l"], [bs, "b.s"], [br, "b.r"], [bd, "b.d"], [bt, "b.t"], [bn, "b.n"], [bg, "b.g"], [bm, "b.m"]]) {
      wanted(sel, key);
      controls();
    }
    // the other kind's reference when the kind changes
    bk.addEventListener("change", () => { controls(); br.value = tables[bk.value].models.includes(REF[bk.value]) ? REF[bk.value] : br.value; });
    [bk, bv, bl, bs, br, bx, bd, bt, bn, bg, bm].forEach(el => el.addEventListener("change", render));
    $("builder-card").hidden = false;
    render();
  }

  // ── your own region ──────────────────────────────────────────────────────────────────────────────────────────
  const MIN_RUNS = 20;       // runs a model needs in a box's chart
  const MIN_STATIONS = 5;    // stations the box needs on its median run
  const SHARE = 0.95;        // a model with this share of the most runs is compared on the runs they all have
  const PRESETS = [   // [name, south, west, north, east]: whole 2.5 degree cells
    ["Northeastern US", 37.5, -82.5, 47.5, -67.5], ["Southeastern US", 25, -92.5, 37.5, -75], ["US Midwest", 37.5, -97.5, 47.5, -82.5],
    ["US Great Plains", 30, -105, 47.5, -95], ["US Rockies", 35, -115, 47.5, -102.5], ["US West Coast", 32.5, -125, 50, -115],
    ["Canadian Prairies", 47.5, -117.5, 57.5, -95], ["UK and Ireland", 50, -10, 60, 2.5], ["Western Europe", 42.5, -5, 55, 10],
    ["Central Europe", 45, 5, 55, 20], ["The Alps", 42.5, 5, 50, 17.5], ["Scandinavia", 55, 5, 70, 30], ["Iberia", 35, -10, 45, 5],
    ["Japan", 30, 127.5, 45, 145], ["Eastern China", 22.5, 110, 40, 122.5], ["India", 7.5, 67.5, 32.5, 90],
    ["Southeastern Australia", -40, 137.5, -27.5, 155], ["Southern Brazil", -32.5, -57.5, -20, -40], ["Argentina", -40, -67.5, -25, -57.5],
    ["South Africa", -35, 15, -22.5, 32.5]];

  function region(ctx, file) {
    const { $, fill, who, ordered } = ctx, set = menus(fill);
    const rk = $("rg-kind"), rv = $("rg-var"), rs = $("rg-show"), rd = $("rg-day"), rr = $("rg-ref"), rp = $("rg-preset"), rm = $("rg-smooth");
    const show = (id, on) => { $(id).hidden = !on; };
    let M = null, map = null, cellsLayer = null, shade = null, rect = null, drawing = false, started = false;
    const fromLink = (asked.get("r.box") || "").split(",").map(Number);
    let box = fromLink.length === 4 && fromLink.every(Number.isFinite) ? fromLink : PRESETS[0].slice(1);   // [S, W, N, E]
    const dateOf = (init, h) => Date.UTC(+init.slice(0, 4), +init.slice(4, 6) - 1, +init.slice(6, 8), +init.slice(8, 10)) + h * 3600e3;
    const fmtDate = t => new Date(t).toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" });
    const snap = b => { const c = M.cell; return [Math.floor(b[0] / c) * c, Math.floor(b[1] / c) * c, Math.ceil(b[2] / c) * c, Math.ceil(b[3] / c) * c]; };
    const boxText = b => { const ns = a => `${Math.abs(a)}°${a >= 0 ? "N" : "S"}`, ew = a => `${Math.abs(a)}°${a >= 0 ? "E" : "W"}`; return `${ns(b[0])} to ${ns(b[2])}, ${ew(b[1])} to ${ew(b[3])}`; };

    function controls() {
      set(rk, Object.keys(M.kinds), k => k === "single" ? "Single runs" : "Ensemble means");
      const K = M.kinds[rk.value];
      set(rv, Object.keys(K.vars), v => K.vars[v].label);
      set(rs, ["date", "lead"], x => x === "date" ? "By date" : "By forecast day");
      set(rd, M.leads.map(String), h => `Day ${h / 24}`);
      if (!rd.dataset.set) { rd.dataset.set = "1"; rd.value = M.leads.includes(120) ? "120" : String(M.leads[0]); }
      set(rm, ["7", "1"], d => d === "7" ? "Week's running mean" : "Each run");
      const models = ordered(K.models);
      set(rr, models);
      if (rr.dataset.kind !== rk.value) { rr.dataset.kind = rk.value; rr.value = models.find(m => m === "IFS" || m === "IFS-ENS") || models[0]; }
      const byDate = rs.value === "date";
      show("rg-day-l", byDate); show("rg-smooth-l", byDate); show("rg-ref-l", !byDate);
    }
    // the cells in the box and the tile files they are filed in
    function inBox(V) {
      const c = M.cell, nx = M.nx, per = Math.round(M.tile / c), want = new Set();
      for (const [cell] of V.cells) {
        const lat = Math.floor(cell / nx) * c - 90, lon = (cell % nx) * c - 180;
        if (lat >= box[0] && lat + c <= box[2] + 1e-9 && lon >= box[1] && lon + c <= box[3] + 1e-9) want.add(cell);
      }
      const tiles = new Set([...want].map(cell => `${Math.floor(Math.floor(cell / nx) / per)}_${Math.floor((cell % nx) / per)}`));
      return { want, tiles: [...tiles].filter(t => V.tiles.includes(t)) };
    }
    // each model's sums in the box by lead and run: { model: { n, se, sse }[lead][run] }
    async function sums(kind, v) {
      const V = M.kinds[kind].vars[v], K = M.kinds[kind], { want, tiles } = inBox(V), R = K.inits.length, L = M.leads.length;
      const files = await Promise.all(tiles.map(t => ctx.figJson(`region_${kind}_${v}_${t}.json`)));
      if (files.some(f => !f)) return null;
      const out = Object.fromEntries(K.models.map(m => [m, Array.from({ length: L }, () => Array.from({ length: R }, () => ({ n: 0, se: 0, sse: 0 })))]));
      const step = M.step;
      for (const f of files) {
        const C = f.cells.length, ks = f.cells.map((c, k) => want.has(c) ? k : -1).filter(k => k >= 0);
        for (const [m, d] of Object.entries(f.models)) {
          const o = out[m];
          if (!o) continue;
          for (let li = 0; li < L; li++) for (let r = 0; r < R; r++) {
            const base = (li * R + r) * C, s = o[li][r];
            for (const k of ks) {
              const n = f.n[base + k];
              if (!n) continue;
              s.n += n; s.se += n * d.me[base + k] * step; s.sse += n * (d.rm[base + k] * step) ** 2;
            }
          }
        }
      }
      // a run a model does not have at a lead is no run of it (region_cells: present)
      for (const m of K.models) for (let li = 0; li < L; li++) for (let r = 0; r < R; r++) if (V.present[m]?.[li]?.[r] !== "1") out[m][li][r] = null;
      return { out, cells: want.size };
    }
    let ticket = 0;
    async function render() {
      controls();
      const mine = ++ticket, kind = rk.value, v = rv.value, K = M.kinds[kind], V = K.vars[v];
      const host = $("rg-fig");
      $("rg-box").textContent = `The box: ${boxText(box)}.`;
      const got = await sums(kind, v);
      if (mine !== ticket) return;
      if (!got) { window.vcharts.clear(host, "The box's numbers could not be loaded."); return; }
      // the page's window of runs: its newest 30, or all the cells hold (90) for a longer window, laid on the calendar
      // (a day without a run is a gap, as in the country series)
      const R = K.inits.length, first = ctx.W === "10" || ctx.W === "30" ? Math.max(0, R - 30) : 0, runs = K.inits.slice(first);
      const day0 = dateOf(runs[0], 0), dayOf = runs.map(i => Math.round((dateOf(i, 0) - day0) / 864e5)), D = dayOf[dayOf.length - 1] + 1;
      // each model's series at a lead, by the run's day: bias and mean squared error (null: no run, or no pairs in the box)
      const series = li => Object.fromEntries(K.models.map(m => {
        const x = Array(D).fill(null);
        runs.forEach((_, r) => { const s = got.out[m][li][first + r]; if (s && s.n) x[dayOf[r]] = { b: s.se / s.n, q: s.sse / s.n, n: s.n }; });
        return [m, x];
      }));
      const byDate = rs.value === "date", per = M.leads.map((_, li) => series(li)), at = byDate ? M.leads.indexOf(+rd.value) : 0;
      const counts = per[at][K.models[0]].map((_, d) => Math.max(0, ...K.models.map(m => per[at][m][d]?.n || 0))).filter(n => n > 0).sort((x, y) => x - y);
      const stations = counts.length ? counts[Math.floor((counts.length - 1) / 2)] : 0;
      $("rg-note").textContent = "";
      if (stations < MIN_STATIONS) {
        window.vcharts.clear(host, `${stations ? `Only ${stations} station${stations > 1 ? "s" : ""}` : "No stations"} in this box on the median run: draw a larger one (it needs ${MIN_STATIONS}).`);
        return;
      }
      // the models with enough runs; those with nearly all of them are compared on the days they all have, the others on
      // their own (verify_figs.country_series)
      const have = Object.fromEntries(K.models.map(m => [m, per[at][m].filter(Boolean).length])), most = Math.max(...Object.values(have));
      const shown = ordered(K.models.filter(m => have[m] >= MIN_RUNS)), full = shown.filter(m => have[m] >= SHARE * most);
      if (!shown.length) { window.vcharts.clear(host, `Too few runs in this box yet (a model needs ${MIN_RUNS}).`); return; }
      const sample = (li, m) => per[li][m].map((x, d) => x && (!full.includes(m) || full.every(f => per[li][f][d])) ? x : null);
      const u = V.unit, name = `${V.label}${kind === "single" ? "" : " (ensemble means)"}: ${boxText(box)}`;
      const span = `the ${runs.length} 00Z runs from ${fmtDate(day0)} to ${fmtDate(dateOf(runs[runs.length - 1], 0))}`;
      const share = () => linkTo({ "r.box": box.join(","), "r.k": kind, "r.v": v, "r.s": rs.value, "r.d": rd.value, "r.m": rm.value, "r.r": rr.value }, "myregion");
      const rmse = xs => Math.sqrt(mean(xs.map(x => x.q)));
      if (byDate) {
        const h = +rd.value, days = +rm.value, on = shown.map(m => sample(at, m));
        const tests = meanTest(on.map(x => x.map(y => y && y.b)), MIN_RUNS), cut = cutOf(tests.map(x => x && x.p));
        const note = k => {
          const x = on[k].filter(Boolean), mb = tests[k] && cut >= 0 && tests[k].p <= cut ? tests[k].mean : null;
          return `bias ${mb == null ? "–" : (mb > 0 ? "+" : mb < 0 ? "−" : "") + Math.abs(mb).toFixed(2)}, RMSE ${rmse(x).toFixed(2)}${full.includes(shown[k]) ? "" : ` (${x.length} runs)`}`;
        };
        // a line by the date the forecasts verify: each run, or the running mean (of the squared error, then its root)
        const pts = (m, f, root) => running(per[at][m].map(x => x && f(x)), days)
          .map((y, d) => ({ u: day0 + d * 864e5 + h * 3600e3, v: y == null ? null : root ? Math.sqrt(y) : y }));
        const panel = (title, f, root, more) => ({ title, u: { type: "time" }, markers: days <= 1, ends: true, tipHead: fmtDate, ...more,
          series: shown.map(m => ({ ...who(m), pts: pts(m, f, root) })) });
        window.vcharts.draw(host, { label: `${name}, day ${h / 24}`, cols: 1, height: 260, data: false, share,
          legend: shown.map((m, k) => ({ ...who(m), note: note(k) })),
          keys: [`after each name: its means (${u}) over the runs every model has, or over its own (counted); a dash: a mean bias within the noise (false discovery rate 10%)`],
          panels: [panel(`Bias (${u}), ${days > 1 ? `a ${days}-day running mean` : "each run"} by the date the forecasts verify`, x => x.b, false, { refs: [{ v: 0 }] }),
            panel(`RMSE (${u})`, x => x.q, true, { v: { floor: 0 } })] });
        $("rg-note").textContent = `${stations} stations on the median run in ${got.cells} cells; day ${h / 24} forecasts of ${span}.`;
        return;
      }
      // by forecast day: the mean bias with its interval, the RMSE, and the RMSE against the reference on the runs both
      // have, the tests of each panel through one cut
      const ref = rr.value, us = M.leads.map(h => h / 24), axis = { label: "Forecast day", ticks: [0, ...us], fmt: d => String(d) };
      const bias = M.leads.map((_, li) => meanTest(shown.map(m => sample(li, m).map(x => x && x.b)), MIN_RUNS));
      const diff = M.leads.map((_, li) => shown.map(m => {
        const both = per[li][m].map((x, d) => x && per[li][ref]?.[d] ? [x.q, per[li][ref][d].q] : null).filter(Boolean);
        if (m === ref || both.length < MIN_RUNS) return null;
        return { p: pairedTest(both.map(([x, y]) => x - y)), pct: 100 * (Math.sqrt(mean(both.map(z => z[0])) / mean(both.map(z => z[1]))) - 1) };
      }));
      const cutB = cutOf(bias.flat().map(x => x && x.p)), cutD = cutOf(diff.flat().map(x => x && x.p));
      const line = f => shown.map((m, k) => ({ ...who(m), pts: us.map((d, li) => ({ u: d, ...f(m, k, li) })) }));
      const panels = [
        { title: `Bias (${u})`, u: axis, refs: [{ v: 0 }], dodge: 5,
          series: line((m, k, li) => { const x = bias[li][k]; return x ? { v: x.mean, lo: x.lo, hi: x.hi, on: cutB >= 0 && x.p <= cutB } : { v: null }; }) },
        { title: `RMSE (${u})`, u: axis, v: { floor: 0 }, dodge: 5,
          series: line((m, k, li) => { const x = sample(li, m).filter(Boolean); return { v: x.length >= MIN_RUNS ? rmse(x) : null }; }) }];
      if (shown.includes(ref) && shown.length > 1) panels.push({ title: `RMSE against ${ref} (%)`, u: axis, refs: [{ v: 0 }], dodge: 5,
        series: line((m, k, li) => { const x = diff[li][k]; return x ? { v: x.pct, on: x.p != null && cutD >= 0 && x.p <= cutD } : { v: null }; }).filter(sr => sr.name !== ref) });
      window.vcharts.draw(host, { label: `${name}, by forecast day`, panels, height: 330, data: false, share, legend: shown.map(who),
        keys: [`${window.vcharts.pair()}filled where the test passes (false discovery rate 10% over the panel); whiskers: 95% intervals`] });
      $("rg-note").textContent = `${stations} stations on the median run at day 1 in ${got.cells} cells; ${span}.`;
    }

    // the map: where the stations are (a cell's shade: its stations), the box, drawn with the button or shift and a drag
    function paintCells() {
      if (!map) return;
      const V = M.kinds[rk.value].vars[rv.value], c = M.cell, nx = M.nx;
      cellsLayer.clearLayers();
      const top = Math.max(1, ...V.cells.map(x => x[1]));
      for (const [cell, n] of V.cells) {
        if (!n) continue;
        const lat = Math.floor(cell / nx) * c - 90, lon = (cell % nx) * c - 180;
        L.rectangle([[lat, lon], [lat + c, lon + c]], { renderer: shade, stroke: false, fillColor: "#3d3b37",
          fillOpacity: 0.08 + 0.34 * Math.min(1, Math.log1p(n) / Math.log1p(top)), interactive: false }).addTo(cellsLayer);
      }
    }
    function setBox(b, fit) {
      box = snap([Math.max(-90, Math.min(b[0], b[2])), Math.max(-180, Math.min(b[1], b[3])), Math.min(90, Math.max(b[0], b[2])), Math.min(180, Math.max(b[1], b[3]))]);
      if (box[2] - box[0] < M.cell) box[2] = box[0] + M.cell;
      if (box[3] - box[1] < M.cell) box[3] = box[1] + M.cell;
      if (map) {
        const ll = [[box[0], box[1]], [box[2], box[3]]];
        rect ? rect.setBounds(ll) : (rect = L.rectangle(ll, { color: getComputedStyle(document.body).getPropertyValue("--accent").trim() || "#a0522d", weight: 2, fillOpacity: 0.08, interactive: false }).addTo(map));
        if (fit) map.fitBounds(ll, { padding: [30, 30], maxZoom: 5 });
      }
      render();
    }
    function startMap() {
      map = L.map("rg-map", { preferCanvas: true, minZoom: 1, boxZoom: false, scrollWheelZoom: false });   // the wheel scrolls the page past the map
      L.tileLayer("https://services.arcgisonline.com/arcgis/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        { attribution: "Tiles &copy; Esri &mdash; Esri, HERE, Garmin, &copy; OpenStreetMap contributors", maxZoom: 9 }).addTo(map);
      shade = L.canvas({ padding: 0.3 });
      cellsLayer = L.layerGroup().addTo(map);
      paintCells();
      const el = $("rg-map"), btn = $("rg-draw");
      let from = null, ghost = null;
      const latlng = ev => map.mouseEventToLatLng(ev);
      const begin = ev => {
        if (!(drawing || ev.shiftKey) || ev.button > 0) return;
        ev.preventDefault(); ev.stopPropagation();
        map.dragging.disable();
        from = latlng(ev);
        el.setPointerCapture?.(ev.pointerId);
        ghost = L.rectangle([from, from], { color: "#3d3b37", weight: 1.5, dashArray: "4 3", fill: false, interactive: false }).addTo(map);
      };
      const move = ev => { if (from && ghost) ghost.setBounds([from, latlng(ev)]); };
      const end = ev => {
        if (!from) return;
        const to = latlng(ev);
        ghost.remove(); ghost = null;
        map.dragging.enable();
        const b = [from.lat, from.lng, to.lat, to.lng];
        from = null;
        drawing = false; btn.setAttribute("aria-pressed", "false"); el.classList.remove("drawing");
        if (Math.abs(b[2] - b[0]) < 0.5 && Math.abs(b[3] - b[1]) < 0.5) return;   // a tap, not a box
        rp.value = "";
        setBox(b, false);
      };
      el.addEventListener("pointerdown", begin, true);
      el.addEventListener("pointermove", move);
      el.addEventListener("pointerup", end);
      el.addEventListener("pointercancel", () => { if (ghost) ghost.remove(); ghost = null; from = null; map.dragging.enable(); });
      btn.addEventListener("click", () => { drawing = !drawing; btn.setAttribute("aria-pressed", String(drawing)); el.classList.toggle("drawing", drawing); });
      setBox(box, true);
    }
    async function start() {
      M = await ctx.figJson(file);
      if (!M || !Object.keys(M.kinds || {}).length) return;
      fill(rp, ["", ...PRESETS.map(p => p[0])], n => n || "Your own box");
      const named = PRESETS.find(p => p.slice(1).join(",") === box.join(","));
      rp.value = named ? named[0] : "";
      controls();
      for (const [sel, key] of [[rk, "r.k"], [rv, "r.v"], [rs, "r.s"], [rd, "r.d"], [rm, "r.m"], [rr, "r.r"]]) {
        wanted(sel, key);
        controls();
      }
      rp.addEventListener("change", () => { const p = PRESETS.find(x => x[0] === rp.value); if (p) setBox(p.slice(1), true); });
      [rk, rv].forEach(el => el.addEventListener("change", () => { paintCells(); render(); }));
      [rs, rd, rm, rr].forEach(el => el.addEventListener("change", render));
      await ctx.parsed;
      $("region-card").hidden = false;
      return true;
    }
    const ready = start();
    return {
      ready,
      // the view opened: the map is built once its card is on the stage, where it has a size
      show() {
        ready.then(have => {
          if (!have) return;
          if (map) { map.invalidateSize(); return; }
          if (started || !window.L) return;
          started = true;
          requestAnimationFrame(startMap);
        });
      },
    };
  }

  window.vbuild = { builder, region, stats: { meanTest, pairedTest, tTwo, tQuant, cutOf, running } };
})();
