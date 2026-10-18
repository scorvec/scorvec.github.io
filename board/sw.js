// Service worker for /board/: an authenticated, read-only viewer for a static site kept in a private repository.
//
// Scope /board/. It answers /board/app/<path> from a private GitHub repository through the GitHub API, with the
// read-only token the viewer gave /board/ (kept in this origin's IndexedDB, sent only to api.github.com). Everything
// else under /board/ (the sign-in page, this file) goes to the network untouched. The site inside runs as it would
// from any static web server: relative fetches, <img>, links between its pages and history all just work.
//
// Files are fetched by git blob sha from the branch's tree, so a file that did not change between snapshots is never
// downloaded twice (Cache Storage, keyed by sha); the tree itself is re-read at most once a minute. Blobs the newest
// tree no longer holds are dropped from the cache.
'use strict';
const PREFIX = new URL('./app/', self.registration.scope).pathname;          // "/board/app/"
const BLOBS = 'privboard-blobs-v1';
const TREE_TTL = 60e3;
const MIME = { html: 'text/html; charset=utf-8', js: 'text/javascript; charset=utf-8', css: 'text/css; charset=utf-8',
  json: 'application/json', geojson: 'application/geo+json', svg: 'image/svg+xml', png: 'image/png', webp: 'image/webp',
  jpg: 'image/jpeg', gif: 'image/gif', txt: 'text/plain; charset=utf-8', csv: 'text/csv; charset=utf-8', md: 'text/plain; charset=utf-8' };

self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', e => e.waitUntil(self.clients.claim()));
// Every await on storage or the network is bounded, so a stalled step shows an error instead of a blank page.
function within(p, ms, what) {
  return Promise.race([p, new Promise((_, rej) => setTimeout(() => rej(new Error(`Timed out ${what}.`)), ms))]);
}
self.addEventListener('message', e => {
  const d = e.data || {};
  if (d.type === 'reset') { CFG = undefined; TREE = null; TREE_AT = 0; }
  // the sign-in page asks the worker to open the board's first page before navigating to it, so any failure is
  // reported on that page in words instead of as a blank page
  if (d.type === 'probe' && e.ports && e.ports[0]) {
    const port = e.ports[0];
    e.waitUntil((async () => {
      try {
        CFG = undefined; TREE = null; TREE_AT = 0;
        const c = await within(cfg(), 10e3, 'reading the saved sign-in');
        if (!c || !c.repo || !c.token) throw new Error('The board cannot see the saved sign-in: this browser may be blocking site storage for scorvec.com.');
        const t = await within(tree(c), 30e3, 'listing the board on GitHub');
        if (!t.has('board.json')) throw new Error(`${c.repo} has no board in it (no board.json at its root). Sign out and check the repository name.`);
        const path = d.path || 'web/index.html', ent = t.get(path);
        if (!ent) throw new Error(`${path} is not in ${c.repo}.`);
        const buf = await within(blob(c, ent.sha), 30e3, 'downloading the first page from GitHub');
        port.postMessage({ ok: true, files: t.size, bytes: buf.byteLength });
      } catch (err) { port.postMessage({ ok: false, error: String((err && err.message) || err) }); }
    })());
  }
});

// ---- the viewer's credentials, written by /board/index.html into the same IndexedDB ----
let CFG;            // undefined = not read yet, null = none
function idbGet(key) {
  return new Promise(resolve => {
    const o = indexedDB.open('privboard', 1);
    o.onupgradeneeded = () => o.result.createObjectStore('kv');
    o.onerror = () => resolve(null);
    o.onsuccess = () => {
      try {
        const r = o.result.transaction('kv').objectStore('kv').get(key);
        r.onsuccess = () => resolve(r.result || null); r.onerror = () => resolve(null);
      } catch (e) { resolve(null); }
    };
  });
}
async function cfg() {
  if (CFG === undefined) CFG = await idbGet('cfg');
  return CFG;
}

// ---- the branch's tree: path -> {sha, size} ----
let TREE = null, TREE_AT = 0, TREE_P = null;
async function api(c, path, accept) {
  return fetch(`${c.api || 'https://api.github.com'}/repos/${c.repo}/${path}`,
    { headers: { Authorization: `Bearer ${c.token}`, Accept: accept || 'application/vnd.github+json' }, cache: 'no-store' });
}
async function tree(c) {
  if (TREE && Date.now() - TREE_AT < TREE_TTL) return TREE;
  if (!TREE_P) TREE_P = (async () => {
    try {
      const r = await api(c, `git/trees/${encodeURIComponent(c.ref || 'main')}?recursive=1`);
      if (!r.ok) { if (TREE) return TREE; throw httpError(r); }
      const j = await r.json(), m = new Map();
      for (const t of j.tree || []) if (t.type === 'blob') m.set(t.path, { sha: t.sha, size: t.size });
      const changed = !TREE || TREE.sig !== j.sha;
      TREE = m; TREE.sig = j.sha; TREE_AT = Date.now();
      if (changed) prune(c, m);
      return TREE;
    } finally { TREE_P = null; }
  })();
  return TREE_P;
}
async function prune(c, m) {
  try {
    const keep = new Set([...m.values()].map(v => v.sha)), cache = await caches.open(BLOBS);
    for (const req of await cache.keys()) {
      const sha = req.url.split('/').pop();
      if (!keep.has(sha)) await cache.delete(req);
    }
  } catch (e) { /* best effort */ }
}
function httpError(r) {
  const e = new Error(r.status === 401 ? 'The token was rejected: it is invalid, revoked or expired.'
    : r.status === 403 && r.headers.get('x-ratelimit-remaining') === '0' ? 'GitHub API rate limit reached for this token; it resets within the hour.'
    : r.status === 404 || r.status === 403 ? 'This token cannot read the board repository (Contents: Read on it is needed).'
    : `GitHub answered ${r.status}.`);
  e.status = r.status === 403 && r.headers.get('x-ratelimit-remaining') === '0' ? 429 : r.status; return e;
}

async function blob(c, sha) {
  const cache = await caches.open(BLOBS), key = `https://blob.privboard.invalid/${c.repo}/${sha}`;
  const hit = await cache.match(key);
  if (hit) return hit.arrayBuffer();
  const r = await api(c, `git/blobs/${sha}`, 'application/vnd.github.raw+json');
  if (!r.ok) throw httpError(r);
  const buf = await r.arrayBuffer();
  try { await cache.put(key, new Response(buf)); } catch (e) { /* quota: serve uncached */ }
  return buf;
}

// ---- answering ----
function page(status, title, msg) {
  const body = `<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex"><title>${title}</title>
<style>body{font:16px/1.5 -apple-system,'Segoe UI',Helvetica,Arial,sans-serif;max-width:34rem;margin:4rem auto;padding:0 1rem;color:#29321f;background:#f5f6f3}
a{color:#274b7a}</style><h1 style="font-size:1.3rem">${title}</h1><p>${msg}</p><p><a href="${new URL('./', self.registration.scope).pathname}">Open the sign-in page</a></p>`;
  return new Response(body, { status, headers: { 'Content-Type': MIME.html, 'Cache-Control': 'no-store' } });
}
function json(status, obj) {
  return new Response(JSON.stringify(obj), { status, headers: { 'Content-Type': MIME.json, 'Cache-Control': 'no-store' } });
}
async function serve(req, url) {
  let path = decodeURIComponent(url.pathname.slice(PREFIX.length));
  const nav = req.mode === 'navigate';
  if (!path || path.endsWith('/')) path += 'index.html';
  let c;
  try { c = await within(cfg(), 10e3, 'reading the saved sign-in'); } catch (e) { c = null; }
  if (!c || !c.repo || !c.token) return nav ? page(401, 'Sign in first', 'This board needs a read-only token.') : json(401, { error: 'not signed in' });
  let t;
  try { t = await within(tree(c), 30e3, 'listing the board on GitHub'); }
  catch (e) { return nav ? page(e.status || 502, 'The board could not be opened', e.message) : json(e.status || 502, { error: e.message }); }
  const ent = t.get(path);
  if (!ent) {
    if (path.startsWith('api/')) return json(404, { error: 'static snapshot: this needs the live server' });
    return nav ? page(404, 'Not in this snapshot', `<code>${path.replace(/[<&]/g, '')}</code> is not in the board.`) : new Response('not in the board snapshot', { status: 404 });
  }
  let buf;
  try { buf = await within(blob(c, ent.sha), 120e3, `downloading ${path} from GitHub`); }
  catch (e) { return nav ? page(e.status || 502, 'Could not load this page', e.message) : json(e.status || 502, { error: e.message }); }
  const ext = path.includes('/') && path.startsWith('api/') ? 'json' : (path.split('.').pop() || '').toLowerCase();
  const h = { 'Content-Type': MIME[ext] || 'application/octet-stream', 'Cache-Control': 'no-store', 'X-Robots-Tag': 'noindex' };
  return new Response(buf, { status: 200, headers: h });
}
self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);
  if (url.origin !== self.location.origin || !url.pathname.startsWith(PREFIX) || e.request.method !== 'GET') return;
  e.respondWith(serve(e.request, url));
});
