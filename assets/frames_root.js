// The ONE place the animation-frame host is set. Manifests stay on main and
// are fetched same-origin; only the <img> requests go here. Keys mirror repo
// paths (assets/<product>/anim/<dir>/<frame>.webp), so this is a prefix and
// nothing else - see scripts/lib/frames_store.py.
//
// Two hosts serve the same `frames` branch:
//   RAW    raw.githubusercontent.com - fresh within ~5 min of a publish
//   MIRROR cdn.jsdelivr.net          - same files, but its edge caches a
//                                      branch path for up to 12 h
// Corporate web filters commonly block raw.githubusercontent.com (a work
// laptop showed "Loading frames..." forever, 2026-09-03) while letting a CDN
// through, so: probe RAW once per tab, switch to MIRROR if it is unreachable,
// and let every <img> walk RAW -> MIRROR -> copy on main before giving up.
(function () {
  var RAW = "https://raw.githubusercontent.com/scorvec/scorvec.github.io/frames/";
  var MIRROR = "https://cdn.jsdelivr.net/gh/scorvec/scorvec.github.io@frames/";
  var onMirror = false, dead = 0, served = 0, deadSaid = false;
  // rawOk: RAW has answered (the probe, or any frame) - slowness after that is bandwidth, not a block.
  // pending: frames still waiting on their first host, re-pointed at once if the tab switches hosts.
  var rawOk = false, stalls = 0, pending = [], STALL_MS = 4500;
  // The switch is remembered for 30 min, not the whole session: a tab parked on
  // the mirror by a passing outage must come back to RAW once it is healthy.
  try {
    var saved = sessionStorage.getItem("frameHost") || "";
    var at = +(saved.split(":")[1] || 0);
    onMirror = saved.indexOf("mirror") === 0 && at > 0 && Date.now() - at < 30 * 60 * 1000;
    if (!onMirror && saved) sessionStorage.removeItem("frameHost");
  } catch (e) {}
  window.FRAME_ROOT = onMirror ? MIRROR : RAW;
  window.FRAME_ROOT_RAW = RAW;
  window.FRAME_ROOT_MIRROR = MIRROR;

  function emit(detail) {
    try { document.dispatchEvent(new CustomEvent("framehost", { detail: detail })); } catch (e) {}
  }
  function useMirror(why) {
    if (onMirror) return;
    onMirror = true;
    window.FRAME_ROOT = MIRROR;
    try { sessionStorage.setItem("frameHost", "mirror:" + Date.now()); } catch (e) {}
    var p = pending; pending = [];
    for (var i = 0; i < p.length; i++) { try { p[i](); } catch (e) {} }
    emit({ mirror: true, why: why });
  }
  window.frameHostIsMirror = function () { return onMirror; };
  // a tab that already switched earlier in the session: tell the page once it
  // has attached its listener (the page scripts run after this one)
  if (onMirror) document.addEventListener("DOMContentLoaded", function () { emit({ mirror: true, why: "session" }); });

  // frameLoad(im, rel, query, local, onFail): set im.src and walk the fallback
  // chain on error. `rel` is the repo path (assets/sst/anim/<dir>/<file>),
  // `local` the same-origin copy on main (may no longer exist), onFail runs
  // once every host has failed. Consumers keep their own im.onload.
  window.frameLoad = function (im, rel, query, local, onFail) {
    var q = query || "";
    var first = onMirror ? MIRROR : RAW, second = onMirror ? RAW : MIRROR;
    var tries = [first + rel + q, second + rel + q];
    if (local) tries.push(local + q);
    var k = 0, done = false, tok = {};
    im._frameTok = tok;                       // a later frameLoad on the same element retires this one
    var live = function () { return im._frameTok === tok && !done; };
    var hop = function () { if (live() && k === 0 && tries[1].indexOf(MIRROR) === 0) { k = 1; im.src = tries[1]; } };
    if (!onMirror) { pending.push(hop); if (pending.length > 400) pending = pending.slice(-200); }   // old entries are long settled
    // A host that neither answers nor refuses (a web filter silently dropping the connection) would hold
    // the frame until the browser's own timeout, a minute or more, before onerror walks the chain - that
    // is the "page looks broken" case (user, 2026-09-26). Until RAW has proved reachable, race the next
    // host after STALL_MS; whichever arrives first is shown. Once RAW has answered, a slow frame is
    // bandwidth and a race would only halve it, so no race.
    setTimeout(function () {
      if (!live() || k !== 0 || rawOk || onMirror) return;
      stalls++;
      var racer = new Image();
      racer.onload = function () {
        if (!live() || k !== 0) return;
        if (tries[1].indexOf(MIRROR) === 0 && stalls >= 2) useMirror("raw stalled");
        k = 1; im.src = tries[1];           // already in the cache: instant
      };
      racer.src = tries[1];
    }, STALL_MS);
    im.onerror = function () {
      if (im._frameTok !== tok) return;
      k++;
      if (k < tries.length) { im.src = tries[k]; return; }
      done = true;
      // Every host failed FOR THIS FRAME. That is only a network verdict if no
      // frame has ever loaded: once a host has served one, a failure means that
      // particular file is missing from the branch (a half-published loop), and
      // saying "this network blocks GitHub" over a loop that is visibly playing
      // is worse than saying nothing (user, 2026-09-11).
      dead++;
      if (dead >= 2 && served === 0) { deadSaid = true; emit({ dead: true }); }
      else if (served) emit({ missing: true, n: dead });
      if (onFail) onFail(im);
    };
    im.addEventListener("load", function () {
      if (im._frameTok !== tok) return;
      done = true;
      served++;
      if (k === 0 && tries[0].indexOf(RAW) === 0) rawOk = true;
      // A host answering after the verdict retracts it. jsDelivr 404s a path it
      // has not warmed yet, so the first frames of a freshly published loop can
      // all fail and the rest arrive seconds later — which is precisely when the
      // page was left accusing the network while the loop played (2026-09-11).
      if (deadSaid) { deadSaid = false; emit({ recovered: true }); }
      // RAW failed but the mirror served the very same file: RAW is blocked
      // here, not missing a frame. Route the rest of the session to the mirror.
      if (k === 1 && !onMirror && tries[1].indexOf(MIRROR) === 0) useMirror("raw failed, mirror served");
    }, { once: true });
    im.src = tries[0];
  };

  // One cheap reachability probe per tab so a blocked network switches before
  // the first loop starts rather than after each frame times out. HEAD, CORS
  // (both hosts send access-control-allow-origin: *), 4 s budget.
  // A 5xx is GitHub's raw host hiccupping (it answers 503 for a moment while the
  // frames branch is being republished), not a blocked network: switching on it
  // parked the whole session on jsDelivr, whose edge holds a branch path for up
  // to 12 h, so freshly republished loops sat on "Loading frames" or showed
  // stale frames (2026-09-18). Retry a 5xx twice; switch only when RAW cannot be
  // reached at all or keeps failing.
  if (!onMirror && typeof fetch === "function" && typeof AbortController === "function") {
    var probe = function (left) {
      var ctrl = new AbortController(), t = setTimeout(function () { ctrl.abort(); }, 4000);
      fetch(RAW + "assets/sst/anim/anomaly/F00.webp?probe=" + Date.now(), { method: "HEAD", signal: ctrl.signal, cache: "no-store" })
        .then(function (r) {
          clearTimeout(t);
          if (r.ok || r.status === 404) { rawOk = true; return; }
          if (r.status >= 500 && left > 0) { setTimeout(function () { probe(left - 1); }, 1500); return; }
          useMirror("probe status " + r.status);
        })
        .catch(function (err) {
          clearTimeout(t);
          // a 4 s silence on a HEAD request is a blocked or dropped host, not a slow one: switch now (the
          // retries below cost ~15 s during which every frame sat on RAW). A fast refusal may be a blip.
          if (err && err.name === "AbortError") { useMirror("probe timed out"); return; }
          if (left > 0) { setTimeout(function () { probe(left - 1); }, 1000); return; }
          useMirror("probe failed");
        });
    };
    probe(2);
  }
})();
