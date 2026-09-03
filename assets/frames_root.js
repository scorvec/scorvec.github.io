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
  var onMirror = false, dead = 0;
  try { onMirror = sessionStorage.getItem("frameHost") === "mirror"; } catch (e) {}
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
    try { sessionStorage.setItem("frameHost", "mirror"); } catch (e) {}
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
    var k = 0;
    im.onerror = function () {
      k++;
      if (k < tries.length) { im.src = tries[k]; return; }
      // every host failed: after two such frames tell the page
      if (++dead === 2) emit({ dead: true });
      if (onFail) onFail(im);
    };
    im.addEventListener("load", function () {
      // RAW failed but the mirror served the very same file: RAW is blocked
      // here, not missing a frame. Route the rest of the session to the mirror.
      if (k === 1 && !onMirror && tries[1].indexOf(MIRROR) === 0) useMirror("raw failed, mirror served");
    });
    im.src = tries[0];
  };

  // One cheap reachability probe per tab so a blocked network switches before
  // the first loop starts rather than after each frame times out. HEAD, CORS
  // (both hosts send access-control-allow-origin: *), 4 s budget.
  if (!onMirror && typeof fetch === "function" && typeof AbortController === "function") {
    var ctrl = new AbortController(), t = setTimeout(function () { ctrl.abort(); }, 4000);
    fetch(RAW + "assets/sst/anim/anomaly/F00.webp", { method: "HEAD", signal: ctrl.signal })
      .then(function (r) { clearTimeout(t); if (!r.ok && r.status !== 404) useMirror("probe status " + r.status); })
      .catch(function () { clearTimeout(t); useMirror("probe failed"); });
  }
})();
