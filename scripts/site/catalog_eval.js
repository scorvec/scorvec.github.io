#!/usr/bin/env node
// Evaluate a page's product spec with the DOM stubbed out, for scripts/site/build_catalog.py.
//
//   node catalog_eval.js page.html   ->  JSON on stdout: {groups:[{label, items:[[id,label,sub]]}], products:{id:{...}}}
//
// Stage pages define window.STAGE = {groups, products}; the GEPS/GEFS pages define top-level `var P` and
// `var GROUPS` ([label, icon, items]). Either way the spec is plain data plus small functions of the option
// choices (a, b, c), so running the page's inline scripts in a sandbox and calling those functions gives the
// real option lists, image URLs and captions - nothing is scraped with regexes. The page's own UI code runs
// against a universal stub (every property is a callable that returns the stub), so it cannot crash the
// evaluation, and any script that still throws is skipped: the spec is always declared before the viewer code.
"use strict";
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const PAGE = path.resolve(process.argv[2]);
const ROOT = path.resolve(__dirname, "..", "..");
const html = fs.readFileSync(PAGE, "utf8");
// Pages that build their spec after fetching status/manifests (snowbands.html) get the repo's own files:
// same-origin relative URLs resolve against the page's directory, root-relative ones against the repo.
function localFetch(u) {
  u = String(u).split("?")[0].split("#")[0];
  if (/^[a-z]+:\/\//i.test(u)) return new Promise(() => {});
  const f = u.startsWith("/") ? path.join(ROOT, u) : path.join(path.dirname(PAGE), u);
  let txt = null; try { txt = fs.readFileSync(f, "utf8"); } catch (e) {}
  return Promise.resolve({ ok: txt !== null, status: txt !== null ? 200 : 404,
    json: () => Promise.resolve(JSON.parse(txt)), text: () => Promise.resolve(txt || "") });
}
const scripts = [];
const re = /<script(\s[^>]*)?>([\s\S]*?)<\/script>/gi;
let m;
while ((m = re.exec(html))) {
  const attrs = m[1] || "";
  if (/\bsrc\s*=/.test(attrs)) continue;
  if (/type\s*=\s*["'](application\/json|application\/ld\+json|text\/template)/i.test(attrs)) continue;
  scripts.push(m[2]);
}

function makeStub() {
  const fn = function () { return stub; };
  const stub = new Proxy(fn, {
    get(t, k) {
      if (k === Symbol.toPrimitive) return () => "";
      if (k === Symbol.iterator) return function* () {};
      if (k === "then") return undefined;                 // never look like a promise
      if (k === "length") return 0;
      if (k === "style" || k === "dataset") return {};
      if (k === "classList") return { add() {}, remove() {}, toggle() { return false; }, contains() { return false; } };
      if (k === "textContent" || k === "innerHTML" || k === "value" || k === "id" || k === "className") return "";
      if (k === "offsetWidth" || k === "clientWidth" || k === "offsetHeight" || k === "clientHeight") return 1000;
      if (k === "getBoundingClientRect") return () => ({ left: 0, top: 0, right: 1000, bottom: 600, width: 1000, height: 600 });
      if (k === "querySelectorAll" || k === "getElementsByTagName" || k === "getElementsByClassName") return () => [];
      return stub;
    },
    set() { return true; },
    apply() { return stub; },
    construct() { return stub; },
  });
  return stub;
}
const stub = makeStub();
const store = { getItem() { return null; }, setItem() {}, removeItem() {} };
const doc = {
  getElementById: () => stub, querySelector: () => stub, querySelectorAll: () => [],
  createElement: () => makeStub(), createTextNode: () => stub, addEventListener() {}, removeEventListener() {},
  body: stub, documentElement: stub, head: stub, readyState: "complete", fonts: { ready: new Promise(() => {}) },
  title: "", cookie: "",
};
const sandbox = {
  document: doc, location: { hash: "", search: "", pathname: "/", href: "https://scorvec.com/" },
  history: { replaceState() {}, pushState() {} }, navigator: { userAgent: "catalog" },
  localStorage: store, sessionStorage: store, console: { log() {}, warn() {}, error() {}, info() {} },
  setTimeout: () => 0, clearTimeout() {}, setInterval: () => 0, clearInterval() {}, requestAnimationFrame: () => 0,
  addEventListener() {}, removeEventListener() {}, dispatchEvent() {},
  matchMedia: () => ({ matches: false, addEventListener() {}, addListener() {} }),
  getComputedStyle: () => new Proxy({}, { get: () => "0px" }),
  fetch: localFetch, Plotly: stub, ResizeObserver: function () { return stub; },
  IntersectionObserver: function () { return stub; }, MutationObserver: function () { return stub; },
  CustomEvent: function () { return stub; }, Image: function () { return makeStub(); },
  innerWidth: 1400, innerHeight: 900, devicePixelRatio: 1, scrollY: 0,
  URLSearchParams, Date, Math, JSON, Promise, Array, Object, String, Number, RegExp, Error, Map, Set,
};
sandbox.window = sandbox; sandbox.self = sandbox; sandbox.globalThis = sandbox;
const ctx = vm.createContext(sandbox);
let errors = 0;
for (const s of scripts) {
  try { vm.runInContext(s, ctx, { timeout: 2000 }); } catch (e) { errors++; }
}

// let promise chains from localFetch settle (specs built after a fetch), then read the spec
(async () => {
for (let i = 0; i < 20; i++) await new Promise((r) => setImmediate(r));
// normalise the two spec shapes
let groups = [], P = null;
if (sandbox.STAGE && sandbox.STAGE.products) {
  P = sandbox.STAGE.products;
  groups = (sandbox.STAGE.groups || []).map((g) => ({ label: g.label, items: g.items }));
} else {
  const src = vm.runInContext("typeof P !== 'undefined' && typeof GROUPS !== 'undefined' ? {P: P, G: GROUPS} : null", ctx);
  if (src) {
    P = src.P;
    groups = src.G.map((g) => ({ label: g[0], items: g[2] }));
  }
}
if (!P) { process.stdout.write(JSON.stringify({ groups: [], products: {}, errors })); process.exit(0); }

const opt = (g, ...args) => { try { const v = typeof g === "function" ? g(...args) : g; return v && v.items ? v : null; } catch (e) { return null; } };
const call = (f, ...args) => { try { return typeof f === "function" ? f(...args) : (f || null); } catch (e) { return null; } };
const out = {};
for (const g of groups) {
  for (const it of g.items) {
    const id = it[0], p = P[id];
    if (!p) continue;
    const A = opt(p.a);
    const a0 = A ? A.items[0][0] : null;
    // second and third axes may depend on earlier choices: record them for EVERY a value (small lists)
    // and the figure each combination shows (img path or sst_anim embed URL), so a search that lands on an option
    // previews that option's own figure, not the default's
    const bBy = {}, cBy = {}, figBy = {};
    const fig = (a, b, c) => {
      const d = p.dom ? call(p.dom, a, b, c) : null;
      if (d) return null;
      const f = p.frame ? call(p.frame, a, b, c) : null;
      if (f) return { frame: String(f) };
      const i = p.img ? call(p.img, a, b, c) : null;
      return i ? { img: String(i) } : null;
    };
    for (const [av] of (A ? A.items : [[null]])) {
      const B = opt(p.b, av);
      if (B) bBy[av] = B;
      for (const [bv] of (B ? B.items : [[null]])) {
        const C = opt(p.c, av, bv);
        if (C) cBy[av + "|" + bv] = C;
        for (const [cv] of (C ? C.items : [[null]])) {
          const g = fig(av, bv, cv);
          if (g) figBy[av + "|" + bv + "|" + cv] = g;
        }
      }
    }
    const B0 = bBy[a0] || null, b0 = B0 ? B0.items[0][0] : null;
    const C0 = cBy[a0 + "|" + b0] || null, c0 = C0 ? C0.items[0][0] : null;
    const kind = p.dom && call(p.dom, a0, b0, c0) ? "dom" : p.frame && call(p.frame, a0, b0, c0) ? "frame" : p.img ? "img" : (p.dom ? "dom" : "none");
    out[id] = {
      label: p.label || it[1], rail: it[1], sub: it[2] || "", group: g.label,
      a: A, bBy, cBy, figBy, def: [a0, b0, c0],
      kind, img: kind === "img" ? call(p.img, a0, b0, c0) : null, frame: kind === "frame" ? call(p.frame, a0, b0, c0) : null,
      cap: call(p.cap, a0, b0, c0) || "", about: call(p.about, a0, b0, c0) || (typeof p.about === "string" ? p.about : null),
    };
  }
}
process.stdout.write(JSON.stringify({ groups, products: out, errors }));
})();
