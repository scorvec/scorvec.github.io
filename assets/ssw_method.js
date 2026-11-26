/* The SSW outlook's methods page (guides/ssw.html): the test results its text quotes, from the same summary the page
   reads (assets/sst/data/ssw_summary.json, scripts/strat/ssw_summary.py). The checklist and the coupling clock are
   described only while their tests pass, as the page shows them only then. */
(function () {
  "use strict";
  function pct(x) { return x == null ? "–" : Math.round(100 * x) + "\u00A0%"; }
  function sg(x, n) { return x == null ? "–" : (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(n == null ? 1 : n); }
  function p(t) { return t.p < 0.001 ? "p < 0.001" : "p " + t.p.toPrecision(2); }
  function ref(k, t) { document.querySelectorAll('[data-r="' + k + '"]').forEach(function (e) { e.textContent = t; }); }
  function hide(id) { var e = document.getElementById(id); if (e) e.hidden = true; }
  fetch("/assets/sst/data/ssw_summary.json", { cache: "no-cache" }).then(function (r) { if (!r.ok) throw r.status; return r.json(); }).then(function (j) {
    var F = j.surface, X = j.reference;

    var K = j.checklist, KR = K && K.reference;
    var kk = KR ? ["source", "tilt", "flux", "upward", "slowing"].filter(function (k) { return KR.items[k] && KR.items[k].passes; }) : [];
    if (kk.length) {
      var KD = KR.definitions;
      var CN = { source: "waves", tilt: "tilt", flux: "heat flux", upward: "not reflected", slowing: "absorbed" }, O = KR.others;
      ref("cl_tests", kk.map(function (k) { var b = KR.items[k].by_days[String(KD.window)]; return CN[k] + " " + b.hits + " of " + b.n + " against " + pct(b.quiet) + " (" + p(b) + ")"; }).join(", "));
      ref("cl_others", "the 500\u00A0hPa wave-2 size (" + p(O.wave2_size) + ") and tilt (" + p(O.wave2_tilt) + "), and the 10\u00A0hPa wind "
        + O.weak_before.days_before[1] + "–" + O.weak_before.days_before[0] + " days before the onset, a weakened vortex beforehand (" + p(O.weak_before) + ")");
      ref("cl_aloft", pct(O.weaker_aloft.mean) + " of the days against " + pct(O.weaker_aloft.quiet) + ", " + p(O.weaker_aloft));
    } else {
      hide("checklist");
    }

    var C = j.clock;
    if (C && C.reference.passes) {
      var CR = C.reference, CD = CR.definitions;
      ref("ck_n", CR.events.length); ref("ck_thr", "+" + CD.start_threshold);
      ref("ck_start", "ρ " + sg(CR.tests.start.rho, 2) + " with the start day, " + p(CR.tests.start));
      ref("ck_str", "ρ " + sg(CR.tests.strength.rho, 2) + ", " + p(CR.tests.strength));
    } else {
      hide("clock");
    }

    var c = X.cmip6;
    ref("wshare", pct(j.thresholds.wave_share)); ref("active", "+" + j.thresholds.active_z + "\u00A0sd");
    var st = X.split_test; ref("split", st.splits + " splits, " + st.displacements + " displacements, wave-2 share " + st.split_median.toFixed(2) + " against " + st.displacement_median.toFixed(2) + ", area under the curve " + st.auc.toFixed(2) + ", p " + st.p.toFixed(2));
    ref("cm_n", X.cmip6_model_years.toLocaleString("en") + " model-years from " + X.cmip6_models + " CMIP6 models");
    function cm(k) { var t = c[k]; return t ? sg(t.mean, 2) + " (" + p(t) + ")" : "–"; }
    ref("cm_dir", "bottom-up " + cm("wk100_bu/ssi_d1_30") + " over days 1–30 and " + cm("wk100_bu/ssi_d31_60") + " over days 31–60; top-down " + cm("wk100_td/ssi_d1_30") + " and " + cm("wk100_td/ssi_d31_60") + " (imprint index)");
    ref("cm_shear", cm("shear_neg/ssi_d1_30"));
    ref("cm_abs", "absorbing " + cm("ssw_absorbing/ssi_d31_60") + ", reflecting " + cm("ssw_reflecting/ssi_d31_60") + " over days 31–60");
    var mx = X.messori_era5, e1 = mx.epo_after.filter(function (w) { return w.days[0] === 1; })[0], t6 = mx.tna_after.filter(function (w) { return w.days[0] === 6; })[0];
    ref("mx_n", mx.events); ref("mx_epo", e1 ? sg(e1.mean, 2) + "\u00A0sd over days 1–5, p " + e1.p : "–"); ref("mx_tna", t6 ? sg(t6.mean) + "\u00A0K over days 6–12, p " + t6.p : "–");
    var bs = F.skill; ref("bss", bs.all_days.bss.toFixed(2) + " on all days (95\u00A0% range " + bs.all_days.ci95[0].toFixed(2) + "–" + bs.all_days.ci95[1].toFixed(2) + "), "
      + bs.cap_1sd.bss.toFixed(2) + " when the cap is beyond 1\u00A0sd (" + bs.cap_1sd.ci95[0].toFixed(2) + "–" + bs.cap_1sd.ci95[1].toFixed(2) + ")");
    var ei = F.era5_after_ssw; ref("era5_imp", sg(ei.d1_30.mean, 2) + " over days 1–30 (" + ei.d1_30.n + " events, p " + ei.d1_30.t_p.toPrecision(2) + ") and " + sg(ei.d31_60.mean, 2) + " over days 31–60");
    ref("cm_ssw", "sudden warmings were followed by " + cm("ssw/ssi_d1_30") + " over days 1–30, North America " + cm("ssw/tna_d1_30") + "\u00A0K over days 1–30 and northern Eurasia " + cm("ssw/teu_d31_60") + "\u00A0K over days 31–60");
  });
})();
