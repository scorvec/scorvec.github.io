// The one place the frames-branch host is set. Manifests stay on main and are fetched same-origin; only the frame
// requests (and the few data files that live on the `frames` branch) go here. Keys mirror repo paths
// (assets/<product>/anim/<dir>/<frame>.webp), so a host is a prefix and nothing else - see scripts/lib/frames_store.py.
//
// Hosts, in order (the CDN first, because many corporate web filters block GitHub's hosts):
//   CDN    frames.scorvec.com - the site's Bunny pull zone (= scorvec-frames.b-cdn.net), origin
//          raw.githubusercontent.com/scorvec/scorvec.github.io, so CDN/<ref>/<path> serves raw/<ref>/<path>.
//            pinned   CDN/<frames_sha>/<path>  immutable, cached at the edge for a year (the zone's override)
//            branch   CDN/frames/<path>        an Edge Rule holds /frames/* (and /skewt-data/*, /asos5-data/*) for
//                                              60 s at the edge and 60 s in the browser, so a branch path is at most
//                                              ~6 min old (60 s here + raw's own ~5 min), the same as raw today
//          The edge ignores query strings, so CDN URLs carry none. 404s are not cached (cache-control: no-cache).
//   RAW    raw.githubusercontent.com - fresh within ~5 min of a publish; blocked by many corporate web filters
//   MIRROR cdn.jsdelivr.net @<sha>, then @frames (a branch path there can be 12 h old)
//   local  the copy on main, where one still exists
// Everything after the CDN is a silent fallback: a page says nothing about hosts unless the CDN itself has failed
// in this tab. There is deliberately no load-time probe of RAW: on a network that blocks GitHub it would flip the tab
// to jsDelivr and report RAW unreachable while the CDN was serving every frame.
(function () {
  var CDN = "https://frames.scorvec.com/";
  var RAW = "https://raw.githubusercontent.com/scorvec/scorvec.github.io/frames/";
  var MIRROR = "https://cdn.jsdelivr.net/gh/scorvec/scorvec.github.io@frames/";
  var MIRROR_AT = "https://cdn.jsdelivr.net/gh/scorvec/scorvec.github.io@";
  var CDN_STALL_MS = 6000, STALL_MS = 4500, PARK_MS = 30 * 60 * 1000, STRIKES = 3;
  function isSha(s) { return /^[0-9a-f]{7,40}$/.test(s || ""); }
  function isCdn(u) { return u.indexOf(CDN) === 0; }
  function isRaw(u) { return u.indexOf(RAW) === 0; }
  function isMirror(u) { return u.indexOf(MIRROR_AT) === 0; }

  // ── CDN health ──────────────────────────────────────────────────────────────────────────────────────────────
  // A tab parks off the CDN for 30 min only on evidence that the CDN, not the link, is failing: a STRIKE is
  //   (a) a CDN request that failed while another host then served the very same file (a missing file fails on
  //       every host and is not the CDN's fault), or
  //   (b) a CDN request with no answer for CDN_STALL_MS, before the CDN has served anything in this tab, AND a
  //       header-only probe of that URL that also gets no answer within 8 s (slow bandwidth answers a HEAD at once;
  //       a filter that drops the host does not). If a GitHub host served a file while that probe sat unanswered,
  //       the link is fine and the CDN alone is silent: that parks at once (else every frame would wait 6 s).
  // STRIKES of them park the tab. A slow frame, or several frames stalling together on a slow first load, is not a
  // strike: it only races the next host for that one frame. Once the CDN has served, nothing parks it.
  var cdnBad = false, cdnServed = false, strikes = 0, probing = false, probes = 0, otherServed = 0;
  try {
    var cb = +(sessionStorage.getItem("frameCdnBad") || 0);
    if (cb && Date.now() - cb < PARK_MS) cdnBad = true; else if (cb) sessionStorage.removeItem("frameCdnBad");
  } catch (e) {}
  function cdnOk() { return !cdnBad; }
  function strike(n) {
    if (cdnBad || cdnServed) return;
    if ((strikes += n || 1) >= STRIKES) {
      cdnBad = true;
      try { sessionStorage.setItem("frameCdnBad", String(Date.now())); } catch (e) {}
    }
  }
  function served() { cdnServed = true; strikes = 0; }
  function stalled(url) {
    if (cdnBad || cdnServed || probing || probes >= STRIKES + 1 || typeof fetch !== "function") return;
    probing = true; probes++;
    var ctrl = typeof AbortController === "function" ? new AbortController() : null;
    var t = setTimeout(function () { if (ctrl) ctrl.abort(); }, 8000), other0 = otherServed;
    // no-cors: any answer at all (opaque) proves the host is reachable; only silence or a refusal counts
    fetch(url, { method: "HEAD", mode: "no-cors", cache: "no-store", signal: ctrl ? ctrl.signal : undefined })
      .then(function () { clearTimeout(t); probing = false; },
            function () { clearTimeout(t); probing = false; strike(otherServed > other0 ? STRIKES : 1); });
  }
  function servedElsewhere() { otherServed++; }
  window.FRAME_ROOT_CDN = CDN;
  window.frameCdnOk = cdnOk;
  window.frameCdnFailed = strike;        // a caller saw the CDN fail on a file another host then served
  window.frameCdnServed = served;
  window.frameCdnStalled = stalled;      // a caller's CDN request got no answer: probe before counting it
  window.frameOtherServed = servedElsewhere;   // a caller got a file from a GitHub host

  // ── fallback preference among the GitHub hosts (only consulted once the CDN has failed for a file) ─────────────
  // When RAW failed and jsDelivr served the same file, RAW is blocked here: later fallbacks try jsDelivr first.
  // Remembered 30 min, not the whole session, so a passing outage does not pin a tab to the 12 h-stale mirror.
  var onMirror = false;
  try {
    var saved = sessionStorage.getItem("frameHost") || "";
    var at = +(saved.split(":")[1] || 0);
    onMirror = saved.indexOf("mirror") === 0 && at > 0 && Date.now() - at < PARK_MS;
    if (!onMirror && saved) sessionStorage.removeItem("frameHost");
  } catch (e) {}
  function rawBlocked() {
    if (onMirror) return;
    onMirror = true;
    try { sessionStorage.setItem("frameHost", "mirror:" + Date.now()); } catch (e) {}
  }

  // window.FRAME_ROOT: the prefix for a plain `FRAME_ROOT + rel` (unpinned, branch path). Prefer frameUrls/frameLoad/
  // frameFetch, which walk the whole chain.
  window.FRAME_ROOT = cdnBad ? (onMirror ? MIRROR : RAW) : CDN + "frames/";
  window.FRAME_ROOT_RAW = RAW;
  window.FRAME_ROOT_MIRROR = MIRROR;
  // true only when the CDN is parked AND raw is blocked, i.e. the tab really is on jsDelivr
  window.frameHostIsMirror = function () { return cdnBad && onMirror; };

  // frameUrls(rel, sha, query, local): every URL for one file, in the order to try. `query` goes on the GitHub hosts
  // only (the CDN ignores it); `local` is the same-origin copy on main, if any.
  function frameUrls(rel, sha, query, local) {
    var q = query || "", out = [];
    if (!cdnBad) out.push(CDN + (isSha(sha) ? sha : "frames") + "/" + rel);
    var mir = [];
    if (isSha(sha)) mir.push(MIRROR_AT + sha + "/" + rel + q);
    mir.push(MIRROR + rel + q);
    if (onMirror) out = out.concat(mir, [RAW + rel + q]);
    else out = out.concat([RAW + rel + q], mir);
    if (local) out.push(local + q);
    return out;
  }
  window.frameUrls = frameUrls;

  var dead = 0, servedAny = 0, deadSaid = false, mirrorSaid = false, rawOk = false;
  function emit(detail) {
    try { document.dispatchEvent(new CustomEvent("framehost", { detail: detail })); } catch (e) {}
  }
  // after a file has loaded from tries[k]: the bookkeeping shared by frameLoad and frameFetch
  function landed(tries, k, cdnErr, rawErr) {
    var u = tries[k];
    servedAny++;
    if (isCdn(u)) served();
    else { servedElsewhere(); if (cdnErr) strike(); }          // the CDN failed where another host had the file
    if (isMirror(u) && rawErr) rawBlocked();
    if (deadSaid) { deadSaid = false; emit({ recovered: true }); }
    // the slower mirror is carrying this page: say so only when the CDN has failed in this tab (parked)
    if (isMirror(u) && cdnBad && !mirrorSaid) { mirrorSaid = true; emit({ mirror: true, why: "frames.scorvec.com and raw.githubusercontent.com failed" }); }
  }
  function allFailed() {
    // Every host failed FOR THIS FILE. That is only a network verdict if nothing has ever loaded: once a host has
    // served a frame, a failure means that particular file is missing (a half-published loop), and accusing the
    // network over a loop that is visibly playing is worse than saying nothing.
    dead++;
    if (dead >= 2 && servedAny === 0) { if (!deadSaid) { deadSaid = true; emit({ dead: true }); } }
    else if (servedAny) emit({ missing: true, n: dead });
  }

  // frameLoad(im, rel, query, local, onFail, sha): set im.src and walk the chain on error. `rel` is the repo path
  // (assets/sst/anim/<dir>/<file>), `local` the same-origin copy on main (may no longer exist), onFail runs once
  // every host has failed, `sha` the manifest's frames_sha (pins the CDN and jsDelivr to that commit). Consumers keep
  // their own im.onload; a later frameLoad on the same element (or im._frameTok = null) retires this one.
  window.frameLoad = function (im, rel, query, local, onFail, sha) {
    var tries = frameUrls(rel, sha, query, local);
    var k = 0, done = false, tok = {}, racing = false, cdnErr = false, rawErr = false, timer = null;
    im._frameTok = tok;
    var live = function () { return im._frameTok === tok && !done; };
    // A host that neither answers nor refuses (a web filter silently dropping the connection) would hold the frame
    // until the browser's own timeout, a minute or more, and the page would look broken. So a
    // host still silent after its stall time is RACED against the rest of the chain in a hidden Image; whichever
    // arrives first is shown. The CDN is raced after CDN_STALL_MS (and probed - see stalled); RAW only until it has
    // proved reachable in this tab, since after that a slow frame is bandwidth and a race would only halve it.
    function arm() {
      clearTimeout(timer);
      var u = tries[k], ms = isCdn(u) ? CDN_STALL_MS : (isRaw(u) && !rawOk ? STALL_MS : 0);
      if (!ms || racing || k + 1 >= tries.length) return;
      var at = k;
      timer = setTimeout(function () {
        if (!live() || k !== at || racing) return;
        if (isCdn(u)) stalled(u);
        race(at + 1);
      }, ms);
    }
    function race(j) {
      racing = true;
      var r = new Image();
      try { r.fetchPriority = im.fetchPriority || "auto"; } catch (e) {}
      r.onload = function () {
        if (!live() || k >= j) return;
        if (isRaw(tries[k])) rawErr = true;                     // RAW sat silent while the mirror answered
        k = j; go();                                            // already in the cache: instant
      };
      r.onerror = function () {
        if (isRaw(tries[j])) rawErr = true;
        if (!live() || ++j >= tries.length || j <= k) return;
        r.src = tries[j];
      };
      r.src = tries[j];
    }
    function go() { im.src = tries[k]; arm(); }
    im.onerror = function () {
      if (im._frameTok !== tok || done) return;
      if (isCdn(tries[k])) cdnErr = true;                       // a real error from the CDN (a stall is judged by the probe)
      if (isRaw(tries[k])) rawErr = true;
      k++;
      if (k < tries.length) { go(); return; }
      done = true; clearTimeout(timer);
      allFailed();
      if (onFail) onFail(im);
    };
    im.addEventListener("load", function () {
      if (im._frameTok !== tok || done) return;
      done = true; clearTimeout(timer);
      if (isRaw(tries[k])) rawOk = true;
      landed(tries, k, cdnErr, rawErr);
    }, { once: true });
    go();
  };

  // frameFetch(rel, sha, opts) -> Promise<Response>: the same chain for fetch() (JSON/binary data on the frames
  // branch). opts.query (GitHub hosts only), opts.local (copy on main), opts.init (fetch init).
  window.frameFetch = function (rel, sha, opts) {
    opts = opts || {};
    var tries = frameUrls(rel, sha, opts.query, opts.local), k = 0, cdnErr = false, rawErr = false;
    function next() {
      if (k >= tries.length) { allFailed(); return Promise.reject(new Error("unavailable: " + rel)); }
      var u = tries[k++], ms = isCdn(u) ? 8000 : (isRaw(u) && !rawOk ? 7000 : 30000);
      var ctrl = typeof AbortController === "function" ? new AbortController() : null;
      var t = ctrl && setTimeout(function () { ctrl.abort(); }, ms);    // time to HEADERS: latency, not bandwidth
      var init = {}; for (var key in (opts.init || {})) init[key] = opts.init[key];
      if (ctrl) init.signal = ctrl.signal;
      return fetch(u, init).then(function (r) {
        if (t) clearTimeout(t);
        if (!r.ok) {
          if (isCdn(u) && r.status !== 404) cdnErr = true;      // a 404 is the origin's answer, passed through
          if (isRaw(u)) rawErr = true;
          return next();
        }
        if (isRaw(u)) rawOk = true;
        landed(tries, k - 1, cdnErr, rawErr);
        return r;
      }, function (e) {
        if (t) clearTimeout(t);
        if (isCdn(u)) { if (e && e.name === "AbortError") stalled(u); else cdnErr = true; }
        if (isRaw(u)) rawErr = true;
        return next();
      });
    }
    return next();
  };
})();
