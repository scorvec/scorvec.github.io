// TC storm board: the storms, jet and phase parts of circulation.html#tc. Mounted by stage.js as the dom block #tcb-view
// (the shear part mounts #tcs-view, assets/tcshear/tcshear.js; the record part is a still).
// Data: assets/sst/data/tc_board.json (scripts/mjo/src/tc_ens.py), assets/sst/data/tcphase.json (tc_phase.py) and the
// IFS control dynamic-tropopause loops (assets/sst/anim/tcdt/tcdt_manifest.json, shown through the sst_anim.html embed).
// State in the hash: #tc/<part: storms|jet|phase>/<storm>[/<view: tracks|depth>, storms only].
(function () {
  "use strict";
  var root = document.getElementById("tcb-view");
  if (!root) return;
  var B = null, PH = null, DT = null, ready = false, drawn = "";
  var PARTS = ["storms", "jet", "phase"];
  // a chance or depth is shown only once its track record is skilful (tc_verify.py, pooled odds): until then the board leaves
  // it out, and the Track record part says why
  function skilful(k) { var s = B && B.track_record && B.track_record.skill && B.track_record.skill[k]; return !!s && s.pooled === true; }
  function views() { return [["tracks", "Tracks"]].concat(skilful("depth") ? [["depth", "Depth after transition"]] : []); }
  var ENS = { aifs: "AIFS-ENS", ifs: "IFS-ENS" };
  function $(id) { return document.getElementById(id); }
  function esc(s) { return String(s == null ? "" : s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }
  // a missing file (404) is null; any other failure is ERR, so a failed load never reads as "no storms"
  var ERR = { error: true };
  function good(x) { return !!x && x !== ERR; }
  function J(u) { return fetch(u + "?t=" + Math.floor(Date.now() / 300000)).then(function (r) { return r.ok ? r.json() : r.status === 404 ? null : ERR; }).catch(function () { return ERR; }); }
  function state() {
    var raw = location.hash.slice(1), h;
    try { h = decodeURIComponent(raw).split("/"); } catch (e) { h = raw.split("/"); }
    var v = views().some(function (x) { return x[0] === h[3]; }) ? h[3] : "tracks";
    return { prod: h[0], part: h[1], sid: h[2] || null, view: v };
  }
  function go(st) {
    var h = "#tc/" + st.part + "/" + (st.sid || "") + (st.part === "storms" ? "/" + st.view : "");
    if (location.hash !== h) history.replaceState(null, "", h);
    render();
  }
  function mine() { var st = state(); return st.prod === "tc" && PARTS.indexOf(st.part) >= 0; }
  function pct(x) { return x == null ? "–" : Math.round(100 * x) + " %"; }
  var MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  var DAY = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
  function t0() { return Date.parse(B.init.replace(/Z$/, ":00Z")); }
  function when(h, wd) {                                  // lead hours -> "Tue 20 Oct 06Z" (the hour only when not 00Z)
    if (h == null) return "–";
    var d = new Date(t0() + h * 3600000), hh = d.getUTCHours();
    return (wd ? DAY[d.getUTCDay()] + " " : "") + d.getUTCDate() + " " + MON[d.getUTCMonth()] + (hh ? " " + (hh < 10 ? "0" : "") + hh + "Z" : "");
  }
  function span(q) { return q ? when(q[0]) + "–" + when(q[2]) : "–"; }
  function latlon(p) { return p ? Math.abs(p[0]).toFixed(1) + "°" + (p[0] >= 0 ? "N" : "S") + " " + Math.abs(p[1]).toFixed(1) + "°" + (p[1] >= 0 ? "E" : "W") : ""; }

  function rows(list) {
    return list.map(function (s) {
      var q = s.pooled, carried = Object.keys(s.models).map(function (m) { return ENS[m] + " " + s.models[m].carried + "/" + s.models[m].members; }).join(" · ");
      var dep = q.depth_hpa_p10_p50_p90;
      return '<tr><th scope="row"><button type="button" class="tcb-pick" data-sid="' + esc(s.slug) + '">' + esc(s.name) + "</button>" +
        (s.watch ? ' <span class="tcb-watch">watch</span>' : "") + '<span class="tcb-sub">' + esc(s.basin_label || "") + "</span></th>" +
        "<td>" + esc(carried) + "</td>" +
        "<td><b>" + pct(q.p_recurve) + "</b><span class=tcb-sub>" + (q.p_recurve_90 ? "90 %: " + pct(q.p_recurve_90[0]) + "–" + pct(q.p_recurve_90[1]) : "") + "</span></td>" +
        "<td>" + (q.recurve_h_p10_p50_p90 ? when(q.recurve_h_p10_p50_p90[1], true) + "<span class=tcb-sub>" + span(q.recurve_h_p10_p50_p90) + "</span>" : "–") + "</td>" +
        (skilful("post_tropical") ? "<td><b>" + pct(q.p_post_tropical) + "</b><span class=tcb-sub>" + (q.post_tropical_h_p10_p50_p90 ? when(q.post_tropical_h_p10_p50_p90[1], true) : "") + "</span></td>" : "") +
        (skilful("depth") ? "<td>" + (dep ? Math.round(dep[1]) + " hPa<span class=tcb-sub>" + Math.round(dep[0]) + "–" + Math.round(dep[2]) + " · " + q.below_960 + " below 960</span>" : "–") + "</td>" : "") + "</tr>";
    }).join("");
  }
  // what the track record says about a chance (tc_verify.py via the board's track_record.skill): skilful only when its
  // whole interval is above zero; the pooled odds are the board's, IFS-ENS alone is the longer record
  function verdict(k) {
    var s = B.track_record && B.track_record.skill && B.track_record.skill[k];
    var t = !s ? "no track record yet" : s.pooled === true ? "skilful in its track record"
      : s.pooled === false ? "no skill shown in its track record" : s.ifs === true ? "pooled: no track record yet; skilful for IFS-ENS alone"
      : s.ifs === false ? "pooled: no track record yet; no skill shown for IFS-ENS alone" : "no track record yet";
    return "<span class=tcb-sub>" + t + "</span>";
  }
  // the storms with an official id in one table; the developing systems folded under it (open when one is picked, or
  // when nothing else is on the board), so the picked storm's figure stays near the top
  function table(sel) {
    var head = '<thead><tr><th scope="col">Storm</th><th scope="col">Members carrying it</th>' +
      '<th scope="col">Recurving by day ' + B.rules.horizon_days + verdict("recurvature") + '</th><th scope="col">Turn (median, 10–90 %)</th>' +
      (skilful("post_tropical") ? '<th scope="col">Post-tropical by day ' + B.rules.horizon_days + verdict("post_tropical") + "</th>" : "") +
      (skilful("depth") ? '<th scope="col">Deepest after transition' + verdict("depth") + "</th>" : "") + "</tr></thead>";
    var named = B.storms.filter(function (s) { return s.named; }), gen = B.storms.filter(function (s) { return !s.named; });
    var out = named.length ? '<div class="tcb-scroll"><table class="tcb-tab">' + head + "<tbody>" + rows(named) + "</tbody></table></div>" : "";
    if (gen.length) out += "<details class=tcb-gen" + (!named.length || !sel.named ? " open" : "") + "><summary>Possible new storms (" + gen.length + ")</summary>" +
      '<div class="tcb-scroll"><table class="tcb-tab">' + head + "<tbody>" + rows(gen) + "</tbody></table></div></details>";
    return out;
  }
  function picker(sel) {                                  // the maps and the phase space: a row of storm pills
    return '<div class="tcb-row" role="group" aria-label="Storm">' + B.storms.map(function (s) {
      return '<button type="button" class="tcb-pick-pill" data-sid="' + esc(s.slug) + '" aria-pressed="' + (s.slug === sel.slug) + '">' + esc(s.name) + "</button>"; }).join("") + "</div>";
  }

  function summary(s) {
    var q = s.pooled, c = q.counts, n = q.members, out = [];
    out.push("<b>" + esc(s.name) + "</b>" + (s.basin_label ? " (" + esc(s.basin_label) + ")" : "") + (s.obs ? ", at " + latlon(s.obs) + " at the start" : "") + ". " +
      Object.keys(s.models).map(function (m) { return ENS[m] + ": " + s.models[m].carried + " of " + s.models[m].members + " members carry it"; }).join("; ") + ".");
    out.push("Recurvature by day " + B.rules.horizon_days + " (" + n + " members): recurving now " + pct(c.now / n) + ", recurves later " + pct(c.by / n) +
      (q.recurve_h_p10_p50_p90 ? " (turn " + span(q.recurve_h_p10_p50_p90) + ", median " + when(q.recurve_h_p10_p50_p90[1], true) + ")" : "") +
      ", no recurvature " + pct(c.no / n) + ", unresolved " + pct(c.unresolved / n) + (c.absent ? ", not carried " + pct(c.absent / n) : "") + ". " +
      "Chance of recurving " + pct(q.p_recurve) + (q.p_recurve_90 ? " (90 % interval " + pct(q.p_recurve_90[0]) + "–" + pct(q.p_recurve_90[1]) + ")" : "") +
      (s.watch ? ": on watch." : "."));
    if (skilful("post_tropical")) out.push("Post-tropical (ECMWF's tracker stops north of 30°N) by day " + B.rules.horizon_days + ": " + pct(q.p_post_tropical) + " of members" +
      (q.post_tropical_h_p10_p50_p90 ? ", median " + when(q.post_tropical_h_p10_p50_p90[1], true) + " (10–90 %: " + span(q.post_tropical_h_p10_p50_p90) + ")" : "") + ".");
    if (skilful("depth") && q.depth_hpa_p10_p50_p90) {
      var d = q.depth_hpa_p10_p50_p90;
      out.push("Deepest after it reaches the mid-latitudes: median " + Math.round(d[1]) + " hPa (10–90 %: " + Math.round(d[0]) + "–" + Math.round(d[2]) + " hPa), " +
        q.below_960 + " of " + q.depth_members + " members below 960 hPa" + (q.deepest_at_end ? "; " + q.deepest_at_end + " are deepest at day 10, where the follow stops, and may deepen further" : "") + ".");
    }
    var det = Object.keys(s.deterministic || {}).map(function (k) {
      var x = s.deterministic[k], o = x.outcome === "now" ? "recurving now" : x.outcome === "by" ? "recurves " + when(x.turn_h, true) : x.outcome === "no" ? "no recurvature by day " + B.rules.horizon_days : "recurvature unresolved";
      return k + ": " + o + (skilful("post_tropical") && x.post_tropical_h != null ? ", post-tropical " + when(x.post_tropical_h, true) : "") + (x.min_pmsl != null ? ", lowest " + Math.round(x.min_pmsl) + " hPa on its track" : "");
    });
    if (det.length) out.push(det.join("; ") + ".");
    if (!s.named && s.genesis_h_p10_p50_p90) out.push("Members reach tropical-storm strength (17 m/s) " + span(s.genesis_h_p10_p50_p90) + ", median " + when(s.genesis_h_p10_p50_p90[1], true) + ".");
    return out.map(function (p) { return "<p>" + p + "</p>"; }).join("");
  }

  function figure(s, st) {
    var v = B.built ? "?v=" + encodeURIComponent(B.built) : "";
    if (st.part === "jet") {
      var reg = good(DT) && DT.regions && DT.regions[s.slug];
      if (!reg) return "<p class=tcb-mu>" + (DT === ERR ? "Could not load the map list." : "No dynamic-tropopause loop for this storm this cycle (maps are drawn for the first four storms on the board).") + "</p>";
      if (DT.init !== B.init) return "<p class=tcb-mu>The maps are still from " + esc(DT.init) + "; this cycle's appear after the next update.</p>";
      return '<iframe class="tcb-loop" title="' + esc(s.name) + ' on the dynamic tropopause" loading="lazy" src="sst_anim.html?embed=1&base=assets/sst/anim/tcdt&manifest=tcdt_manifest.json&region=' +
        encodeURIComponent(s.slug) + "&regions=" + encodeURIComponent(s.slug) + '&start=manifest"></iframe>';
    }
    var file, what;
    if (st.part === "phase") {
      var fresh = good(PH) && PH.init === B.init, p = fresh ? (PH.storms || []).filter(function (x) { return x.slug === s.slug; })[0] : null;
      file = p && p.file; what = "phase space (IFS control)";
      if (!file) return "<p class=tcb-mu>" + (fresh ? (PH.note || "The IFS control does not carry this storm long enough for a phase-space path.") : "The phase space for this cycle is not out yet.") + "</p>";
    } else {
      file = st.view === "depth" ? s.depth_file : s.tracks_file; what = st.view === "depth" ? "depth after transition" : "tracks";
      if (!file) return "<p class=tcb-mu>" + (st.view === "depth" ? "Too few members reach the mid-latitudes for a depth chart." : "No figure for this storm.") + "</p>";
    }
    return '<img class="tcb-fig" alt="' + esc(s.name) + ": " + what + '" src="assets/sst/' + esc(file) + v + '">';
  }

  function record() {
    var r = B.track_record;
    if (!r) return "<p class=tcb-mu>No track record yet: the verification against best tracks has not run.</p>";
    var L = r.lines || {};
    return ["recurvature", "watch", "post_tropical", "tracks"].filter(function (k) { return L[k]; }).map(function (k) { return "<p>" + esc(L[k]) + "</p>"; }).join("") +
      "<p class=tcb-mu>Scored " + esc((r.built || "").slice(0, 10)) + " against IBTrACS best tracks; the figures are in the <a href=\"#tc/record\">Track record</a> part.</p>";
  }

  function render() {
    if (!ready || !root.offsetParent || !mine()) return;
    var st = state();
    drawn = location.hash;
    if (!good(B)) {
      $("tcb-head").innerHTML = B === ERR ? "Could not load the storm board (network or server error). Reload to try again." : "The storm board has not been published yet.";
      $("tcb-body").hidden = true; return;
    }
    var gone = Object.keys(B.models).filter(function (m) { return B.models[m].status !== "ok"; });
    // one line above the numbers: the watch, the run, a missing ensemble (the rules are in the About text)
    $("tcb-head").innerHTML = '<p class="tcb-line' + (B.watch.level === "watch" ? " on" : "") + '">' + esc((B.watch.lines || [""])[0]) +
      ' <span class=tcb-mu>· run of ' + esc(B.init.replace("T", " ")) +
      (gone.length ? " · " + gone.map(function (m) { return ENS[m]; }).join(", ") + " unavailable: the odds use the other ensemble alone" : "") + "</span></p>";
    if (!B.storms.length) {
      $("tcb-body").hidden = true;
      $("tcb-head").innerHTML = "<p class=tcb-line>" + (B.status === "missing" ? "Track data are unavailable for this run" : "No tropical storm in the Northern Hemisphere this run") +
        " <span class=tcb-mu>· run of " + esc(B.init.replace("T", " ")) + "</span></p>";
      return;
    }
    $("tcb-body").hidden = false;
    var s = B.storms.filter(function (x) { return x.slug === st.sid; })[0] || B.storms[0];
    st.sid = s.slug;
    var board = st.part === "storms";
    $("tcb-table").innerHTML = board ? table(s) : picker(s);
    root.querySelectorAll(".tcb-pick, .tcb-pick-pill").forEach(function (b) {
      var on = b.getAttribute("data-sid") === s.slug;
      b.setAttribute("aria-pressed", on);
      if (b.closest("tr")) b.closest("tr").classList.toggle("on", on);
      b.onclick = function () { st.sid = b.getAttribute("data-sid"); go(st); };
    });
    $("tcb-sum").innerHTML = board ? summary(s) : "";
    $("tcb-views").hidden = !board;
    $("tcb-views").innerHTML = board && views().length > 1 ? views().map(function (v) { return '<button type="button" data-v="' + v[0] + '" aria-pressed="' + (st.view === v[0]) + '">' + v[1] + "</button>"; }).join("") : "";
    root.querySelectorAll("#tcb-views button").forEach(function (b) { b.onclick = function () { st.view = b.getAttribute("data-v"); go(st); }; });
    $("tcb-fig").innerHTML = figure(s, st);
    $("tcb-rec").hidden = !board;
    $("tcb-record").innerHTML = board ? record() : "";
    if (location.hash !== "#tc/" + st.part + "/" + st.sid + (board ? "/" + st.view : "")) go(st);
  }

  Promise.all([J("assets/sst/data/tc_board.json"), J("assets/sst/data/tcphase.json"), J("assets/sst/anim/tcdt/tcdt_manifest.json")]).then(function (r) {
    B = r[0]; PH = r[1]; DT = r[2]; ready = true; render();
  });
  window.addEventListener("hashchange", render);
  window.addEventListener("popstate", render);
  // stage.js switches products and parts with pushState (no hashchange): draw when the block becomes visible, and when
  // the stage moved it to another part while it stayed visible (storms -> jet remounts it in one go)
  var shown = false;
  if (window.MutationObserver) new MutationObserver(function () {
    var vis = !!root.offsetParent; if (vis && (!shown || location.hash !== drawn)) render(); shown = vis;
  }).observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ["hidden", "class", "style"] });
})();
