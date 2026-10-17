"""The 'MJO forecast, all ensembles' block of mjo.html: one phase diagram drawn in the browser from
assets/mjo/rmm_models.json (AIFS-ENS and IFS-ENS, src/rmm_export.py), assets/gefs/data/gefs_rmm_wind.json and
assets/geps/data/geps_rmm_wind.json (the GEFS and GEPS pages' wind-only tracks), with the models, their members, the
horizon and the raw / ENSO-removed index switchable. Called by generate_page.py; returns "" until the first data
file exists, so the page never shows an empty block."""

import json
import sys
from pathlib import Path

ASSETS = Path("assets/mjo")
DATA = ASSETS / "rmm_models.json"
STATUS = ASSETS / "rmm_clean_status.json"
CORR = Path("scripts/mjo/data/reference/enso_rmm_correction.json")


def load_status() -> dict | None:
    """The ENSO-removed forecast's diagnosis (run_rmm.py -> enso_rmm.diagnose), or None before the first one."""
    return json.loads(STATUS.read_text()) if STATUS.exists() and DATA.exists() else None


def _active_change(gr) -> str:
    d = [x["clean_f"]["active"] - x["raw"]["active"] for _, x in gr]
    return "lowers" if all(v < 0 for v in d) else "raises" if all(v > 0 for v in d) else "changes"


def _validation_text() -> str:
    v = json.loads(CORR.read_text()) if CORR.exists() else {}
    val = v.get("validation", {})
    a = val.get("vs_bom_2003_2013_enso_removed", {})
    g = val.get("winters_and_growth_phases", {})
    if not (a.get("raw") and a.get("clean_f")):
        return ""
    r, c = a["raw"], a["clean_f"]
    out = (
        f"Against BoM&rsquo;s ENSO-removed RMM (2003&ndash;2013, {r['n']:,} days) the cleaned wind-only index "
        f"correlates {c['r1']:.2f}/{c['r2']:.2f} (RMM1/RMM2), against {r['r1']:.2f}/{r['r2']:.2f} before, and "
        f"its correlation with the Ni&ntilde;o-3.4 change falls from {r['corr_dn34_r1']:+.2f} to {c['corr_dn34_r1']:+.2f}."
    )
    gr = [(k, x) for k, x in g.items() if "growth" in k]
    if gr:
        out += (
            f" While an El Ni&ntilde;o grows, the correction {_active_change(gr)} the share of &ldquo;active&rdquo; days: "
            + "; ".join(
                f"{k.replace(' (growth)', '')} {x['raw']['active']:.2f} &rarr; {x['clean_f']['active']:.2f}"
                for k, x in gr
            )
            + "."
        )
    return out


SCRIPT = r"""
(function () {
  var box = document.getElementById('mm-box'), cv = document.getElementById('mm-cv'), tip = document.getElementById('mm-tip');
  if (!box || !cv) return;
  var MODELS = [
    ['aifs', 'AIFS-ENS', '#c2410c', '/assets/mjo/rmm_models.json'],
    ['ifs',  'IFS-ENS',  '#1f5fb4', '/assets/mjo/rmm_models.json'],
    ['gefs', 'GEFS',     '#2e7d32', '/assets/gefs/data/gefs_rmm_wind.json'],
    ['geps', 'GEPS',     '#7b3fa0', '/assets/geps/data/geps_rmm_wind.json']
  ];
  var D = { obs: null, dn34: null, c: null, models: {} };
  var S = { on: { aifs: true, ifs: true, gefs: true, geps: true }, members: false, clean: box.dataset.clean === '1', horizon: 15, obsDays: 40 };
  var REGION = [['West. Hem. and Africa', -3.55, 0, -Math.PI / 2], ['Indian Ocean', 0, -3.55, 0], ['Maritime Continent', 3.55, 0, Math.PI / 2], ['Western Pacific', 0, 3.55, 0]];
  var PHASE = { 1: 202.5, 2: 247.5, 3: 292.5, 4: 337.5, 5: 22.5, 6: 67.5, 7: 112.5, 8: 157.5 };
  var pts = [];   // hover targets

  function fmtDate(s) { var d = new Date(s + ':00Z'); return d.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', timeZone: 'UTC' }); }
  function clean1(r, dn) { return S.clean && D.c && dn != null ? r - dn * D.c[0] : r; }
  function clean2(r, dn) { return S.clean && D.c && dn != null ? r - dn * D.c[1] : r; }

  function draw() {
    var W = box.clientWidth; if (!W) return;
    var size = Math.min(W, 760), dpr = window.devicePixelRatio || 1;
    cv.style.width = size + 'px'; cv.style.height = size + 'px';
    cv.width = Math.round(size * dpr); cv.height = Math.round(size * dpr);
    var x = cv.getContext('2d'); x.setTransform(dpr, 0, 0, dpr, 0, 0);
    var pad = 34, R = (size - 2 * pad) / 2, cx = pad + R, cy = pad + R, lim = 4.2, k = R / lim;
    var X = function (v) { return cx + v * k; }, Y = function (v) { return cy - v * k; };
    pts = [];
    x.clearRect(0, 0, size, size);
    x.fillStyle = '#fff'; x.fillRect(0, 0, size, size);
    x.save(); x.beginPath(); x.rect(pad, pad, 2 * R, 2 * R); x.clip();
    // the wheel: unit circle filled, octant lines, rings
    x.fillStyle = '#f2f1ec'; x.beginPath(); x.arc(cx, cy, k, 0, 2 * Math.PI); x.fill();
    x.lineWidth = 1; x.strokeStyle = '#bdb9b0';
    [[0, 1], [1, 0], [1, 1], [1, -1]].forEach(function (d) {
      var n = Math.hypot(d[0], d[1]), ux = d[0] / n, uy = d[1] / n;
      x.beginPath(); x.moveTo(X(ux), Y(uy)); x.lineTo(X(ux * lim * 1.5), Y(uy * lim * 1.5)); x.moveTo(X(-ux), Y(-uy)); x.lineTo(X(-ux * lim * 1.5), Y(-uy * lim * 1.5)); x.stroke();
    });
    x.setLineDash([4, 4]); x.strokeStyle = '#cfcbc2';
    [2, 3, 4].forEach(function (r) { x.beginPath(); x.arc(cx, cy, r * k, 0, 2 * Math.PI); x.stroke(); });
    x.setLineDash([]);
    x.strokeStyle = '#8a857c'; x.lineWidth = 1.2; x.beginPath(); x.arc(cx, cy, k, 0, 2 * Math.PI); x.stroke();
    x.font = '600 13px "Source Sans 3", sans-serif'; x.fillStyle = '#8a857c'; x.textAlign = 'center'; x.textBaseline = 'middle';
    Object.keys(PHASE).forEach(function (p) { var a = PHASE[p] * Math.PI / 180; x.fillText(p, X(3.1 * Math.cos(a)), Y(3.1 * Math.sin(a))); });
    x.font = '400 11px "Source Sans 3", sans-serif'; x.fillStyle = '#6f6b64';
    [1, 2, 3, 4].forEach(function (r) { var a = -Math.PI / 4; x.fillText(r, X(r * Math.cos(a)) + 9, Y(r * Math.sin(a)) + 9); });
    x.font = 'italic 12px "Source Sans 3", sans-serif'; x.fillStyle = '#7a756c';
    REGION.forEach(function (r) { x.save(); x.translate(X(r[1]), Y(r[2])); x.rotate(r[3]); x.fillText(r[0], 0, 0); x.restore(); });
    x.restore();
    x.font = '500 12px "Source Sans 3", sans-serif'; x.fillStyle = '#4a4744'; x.textAlign = 'center'; x.textBaseline = 'middle';
    x.fillText('RMM1', cx, size - 12); x.save(); x.translate(12, cy); x.rotate(-Math.PI / 2); x.fillText('RMM2', 0, 0); x.restore();

    // the observed track: the last obsDays days, older points fainter, a label at each month's first point
    var o = D.obs;
    if (o && o.t.length) {
      var n = o.t.length, cut = Math.max(0, n - S.obsDays * 2), lastMonth = '';
      for (var i = cut + 1; i < n; i++) {
        var f = (i - cut) / (n - cut);
        x.strokeStyle = 'rgba(40,40,40,' + (0.25 + 0.7 * f).toFixed(2) + ')'; x.lineWidth = 1.6;
        x.beginPath(); x.moveTo(X(clean1(o.r1[i - 1], o.dn && o.dn[i - 1])), Y(clean2(o.r2[i - 1], o.dn && o.dn[i - 1])));
        x.lineTo(X(clean1(o.r1[i], o.dn && o.dn[i])), Y(clean2(o.r2[i], o.dn && o.dn[i]))); x.stroke();
      }
      for (var j = cut; j < n; j++) {
        var px = X(clean1(o.r1[j], o.dn && o.dn[j])), py = Y(clean2(o.r2[j], o.dn && o.dn[j])), mo = o.t[j].slice(0, 7);
        pts.push({ x: px, y: py, s: 'Observed ' + fmtDate(o.t[j]), r1: clean1(o.r1[j], o.dn && o.dn[j]), r2: clean2(o.r2[j], o.dn && o.dn[j]) });
        if (o.t[j].slice(11) === '00') { x.fillStyle = 'rgba(40,40,40,' + (0.35 + 0.65 * (j - cut) / (n - cut)).toFixed(2) + ')'; x.beginPath(); x.arc(px, py, 2.2, 0, 2 * Math.PI); x.fill(); }
        if (mo !== lastMonth && j > cut) { lastMonth = mo; label(x, px, py, new Date(o.t[j] + ':00Z').toLocaleDateString('en-GB', { month: 'short', timeZone: 'UTC' }), '#444', -9); }
        if (j === cut) lastMonth = mo;
      }
      var lx = X(clean1(o.r1[n - 1], o.dn && o.dn[n - 1])), ly = Y(clean2(o.r2[n - 1], o.dn && o.dn[n - 1]));
      x.fillStyle = '#111'; x.strokeStyle = '#fff'; x.lineWidth = 1.5; x.beginPath(); x.arc(lx, ly, 5, 0, 2 * Math.PI); x.fill(); x.stroke();
    }

    // the models: members faint under the means; a marker every 5 days on the mean, the end point squared
    MODELS.forEach(function (M) {
      var key = M[0], col = M[2], m = D.models[key]; if (!m || !S.on[key]) return;
      var dn = D.dn34, L = m.lead.length, last = L - 1;
      while (last > 0 && m.lead[last] > S.horizon + 0.01) last--;
      if (S.members) {
        x.strokeStyle = hexa(col, 0.11); x.lineWidth = 0.9;
        for (var mi = 0; mi < m.m1.length; mi++) {
          x.beginPath();
          for (var li = 0; li <= last; li++) { var a1 = m.m1[mi][li], a2 = m.m2[mi][li]; if (a1 == null) continue; var qx = X(clean1(a1, dn)), qy = Y(clean2(a2, dn)); li ? x.lineTo(qx, qy) : x.moveTo(qx, qy); }
          x.stroke();
        }
      }
      x.strokeStyle = col; x.lineWidth = 2.6; x.lineJoin = 'round'; x.beginPath();
      for (var li2 = 0; li2 <= last; li2++) { var bx = X(clean1(m.r1[li2], dn)), by = Y(clean2(m.r2[li2], dn)); li2 ? x.lineTo(bx, by) : x.moveTo(bx, by); }
      x.stroke();
      for (var li3 = 0; li3 <= last; li3++) {
        var ld = m.lead[li3], px2 = X(clean1(m.r1[li3], dn)), py2 = Y(clean2(m.r2[li3], dn));
        pts.push({ x: px2, y: py2, s: m.label + ' day ' + Math.round(ld) + ' · ' + fmtDate(m.t[li3]), r1: clean1(m.r1[li3], dn), r2: clean2(m.r2[li3], dn) });
        var five = Math.abs(ld / 5 - Math.round(ld / 5)) < 0.01 && ld > 0.01;
        if (five || li3 === last) {
          x.fillStyle = col; x.strokeStyle = '#fff'; x.lineWidth = 1.4; x.beginPath();
          if (li3 === last) x.rect(px2 - 4.5, py2 - 4.5, 9, 9); else x.arc(px2, py2, 3.6, 0, 2 * Math.PI);
          x.fill(); x.stroke();
          if (five) label(x, px2, py2, String(Math.round(ld)), col, 10);
        }
      }
    });
  }
  function hexa(h, a) { var n = parseInt(h.slice(1), 16); return 'rgba(' + (n >> 16) + ',' + ((n >> 8) & 255) + ',' + (n & 255) + ',' + a + ')'; }
  function label(x, px, py, s, col, dy) {
    x.font = '600 11px "Source Sans 3", sans-serif'; x.textAlign = 'center'; x.textBaseline = 'middle';
    x.lineWidth = 3; x.strokeStyle = 'rgba(255,255,255,0.9)'; x.strokeText(s, px, py + dy); x.fillStyle = col; x.fillText(s, px, py + dy);
  }

  // the controls
  var legend = document.getElementById('mm-legend');
  function controls() {
    legend.innerHTML = '';
    MODELS.forEach(function (M) {
      var m = D.models[M[0]], b = document.createElement('button'); b.type = 'button'; b.className = 'mm-tog' + (S.on[M[0]] && m ? ' on' : '');
      b.style.setProperty('--c', M[2]);
      b.innerHTML = '<i></i>' + M[1] + (m ? ' <small>' + m.n + ' members · init ' + m.init + '</small>' : ' <small>not published yet</small>');
      if (!m) { b.disabled = true; } else { b.onclick = function () { S.on[M[0]] = !S.on[M[0]]; b.classList.toggle('on', S.on[M[0]]); draw(); }; }
      legend.appendChild(b);
    });
  }
  function seg(id, key, vals) {
    var el = document.getElementById(id); if (!el) return;
    el.querySelectorAll('button').forEach(function (b) {
      var v = b.dataset.v; v = v === '1' ? true : v === '0' ? false : isNaN(+v) ? v : +v;
      b.classList.toggle('on', S[key] === v);
      b.onclick = function () { S[key] = v; el.querySelectorAll('button').forEach(function (q) { q.classList.remove('on'); }); b.classList.add('on'); draw(); note(); };
    });
  }
  function note() {
    var el = document.getElementById('mm-idx'); if (!el) return;
    el.textContent = S.clean
      ? 'ENSO removed: every track minus ΔNiño-3.4 × c (ΔNiño-3.4 ' + (D.dn34 == null ? '—' : (D.dn34 > 0 ? '+' : '') + D.dn34.toFixed(2) + ' °C' + '), the part of the wind index tied to the Niño-3.4 change over the 120-day filter window.')
      : 'Standard index (ENSO not removed): the wind-only RMM with the previous 120-day mean removed, as the Bureau of Meteorology has computed RMM since 2014.';
  }
  cv.addEventListener('pointermove', function (e) {
    var r = cv.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top, best = null, bd = 144;
    for (var i = 0; i < pts.length; i++) { var d = (pts[i].x - mx) * (pts[i].x - mx) + (pts[i].y - my) * (pts[i].y - my); if (d < bd) { bd = d; best = pts[i]; } }
    if (!best) { tip.hidden = true; return; }
    var amp = Math.hypot(best.r1, best.r2), ph = Math.floor((((Math.atan2(best.r2, best.r1) * 180 / Math.PI + 360) % 360 - 180 + 360) % 360) / 45) + 1;
    tip.textContent = best.s + ' · RMM1 ' + best.r1.toFixed(2) + ', RMM2 ' + best.r2.toFixed(2) + ' · amplitude ' + amp.toFixed(2) + (amp >= 1 ? ', phase ' + ph : ' (no MJO)');
    tip.hidden = false; tip.style.left = Math.min(mx + 12, r.width - tip.offsetWidth - 4) + 'px'; tip.style.top = (my + 14) + 'px';
  });
  cv.addEventListener('pointerleave', function () { tip.hidden = true; });

  function got(key, j) {
    if (key === 'aifs') { D.obs = j.obs; D.dn34 = j.dn34; D.c = j.c; Object.keys(j.models || {}).forEach(function (k) { D.models[k] = j.models[k]; }); }
    else if (j && j.r1) D.models[key] = j;
  }
  var urls = {}; MODELS.forEach(function (M) { urls[M[3]] = urls[M[3]] || []; urls[M[3]].push(M[0]); });
  Promise.all(Object.keys(urls).map(function (u) {
    return fetch(u, { cache: 'no-cache' }).then(function (r) { return r.ok ? r.json() : null; }).catch(function () { return null; })
      .then(function (j) { if (j) urls[u].forEach(function (k) { got(k, j); }); });
  })).then(function () {
    if (!D.c) { S.clean = false; var cb = document.querySelector('#mm-index [data-v="1"]'); if (cb) cb.disabled = true; }
    controls(); seg('mm-index', 'clean', null); seg('mm-horizon', 'horizon', null); seg('mm-members', 'members', null); note(); draw();
  });
  if (window.ResizeObserver) new ResizeObserver(function () { draw(); }).observe(box); else addEventListener('resize', draw);
})();
"""

CSS = """
  #mjo-models .mm-ctl { display: flex; flex-wrap: wrap; gap: 0.5rem 1.4rem; align-items: center; margin: 0.6rem 0 0.8rem; font-size: 0.92rem; color: var(--muted); }
  #mjo-models .mm-ctl > span { display: inline-flex; align-items: center; gap: 0.3rem; flex-wrap: wrap; }
  #mjo-models .mm-seg button, #mjo-models .mm-tog { font: 600 0.84rem var(--sans); color: var(--ink2); background: #fff; border: 1.5px solid var(--edge); border-radius: 999px; padding: 0.3rem 0.75rem; cursor: pointer; }
  #mjo-models .mm-seg button.on { color: #fff; background: var(--accent); border-color: var(--accent); }
  #mjo-models .mm-seg button:disabled { color: var(--muted); cursor: default; }
  #mjo-models .mm-legend { display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 0 0 0.6rem; }
  #mjo-models .mm-tog { display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.35rem 0.85rem 0.35rem 0.6rem; text-align: left; }
  #mjo-models .mm-tog i { width: 1.1rem; height: 0.3rem; border-radius: 2px; background: var(--c); opacity: 0.35; flex: none; }
  #mjo-models .mm-tog.on { border-color: var(--c); }
  #mjo-models .mm-tog.on i { opacity: 1; }
  #mjo-models .mm-tog small { display: block; font-weight: 400; font-size: 0.78rem; color: var(--muted); }
  #mjo-models .mm-tog:disabled { color: var(--muted); cursor: default; }
  #mjo-models .mm-box { position: relative; max-width: 760px; margin: 0 auto; }
  #mjo-models canvas { display: block; width: 100%; border: 1px solid var(--rule); border-radius: 6px; touch-action: pan-y; }
  #mjo-models .mm-tip { position: absolute; pointer-events: none; background: rgba(28,28,26,0.92); color: #fff; font-size: 0.82rem; line-height: 1.35; padding: 0.3rem 0.5rem; border-radius: 4px; max-width: 18rem; }
  #mjo-models .mm-idx { color: var(--ink2); font-size: 0.92rem; margin: 0.6rem 0 0; max-width: 90ch; }
"""


def models_section() -> str:
    if not DATA.exists():
        return ""
    data = json.loads(DATA.read_text())
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    status = ""
    try:
        from enso_rmm import ENSO_STRONG_DN34, status_sentence

        st = load_status()
        status = status_sentence(st) if st else ""
        clean_first = bool(st and abs(st["dn34"]) >= ENSO_STRONG_DN34)
    except Exception as e:  # noqa: BLE001
        print(f"models_section: no status line ({type(e).__name__}: {e})", file=sys.stderr)
        clean_first = False
    inits = ", ".join(f"{m['label']} {m['init']}" for m in data.get("models", {}).values())
    return f"""
  <section id="mjo-models">
  <h2>MJO forecasts from the global ensembles</h2>
  {f'<p style="font-size:0.95rem;line-height:1.55;margin:0.4rem 0 0.6rem">{status}</p>' if status else ""}
  <p style="color:var(--muted);font-size:0.95rem;line-height:1.55;margin:0 0 0.6rem">The Wheeler&ndash;Hendon phase diagram with the
  observed track of the last 40 days and the ensemble-mean forecast of each model, one marker every five days and a square at the
  horizon. Members are off by default and faint when on. ECMWF&rsquo;s ensembles reach day 15; GEFS and GEPS reach day 35.
  Latest inits: {inits}.</p>
  <div class="mm-legend" id="mm-legend"></div>
  <div class="mm-ctl">
    <span>Index <span class="mm-seg" id="mm-index"><button type="button" data-v="0">standard</button><button type="button" data-v="1">ENSO removed</button></span></span>
    <span>Horizon <span class="mm-seg" id="mm-horizon"><button type="button" data-v="15">15 days</button><button type="button" data-v="35">35 days</button></span></span>
    <span>Members <span class="mm-seg" id="mm-members"><button type="button" data-v="0">off</button><button type="button" data-v="1">on</button></span></span>
  </div>
  <div class="mm-box" id="mm-box" data-clean="{1 if clean_first else 0}"><canvas id="mm-cv" role="img" aria-label="MJO phase diagram: observed track and the ensembles' forecast tracks"></canvas><div class="mm-tip" id="mm-tip" hidden></div></div>
  <p class="mm-idx" id="mm-idx"></p>
  <details style="margin-top:1rem;font-size:0.92rem"><summary style="cursor:pointer;color:var(--accent)">About this diagram and the ENSO removal</summary>
  <p>Every model is projected the same way: the 15&deg;S&ndash;15&deg;N band means of the 850 and 200&nbsp;hPa zonal wind, anomalies
  against the 1991&ndash;2020 climatology with the previous 120-day mean removed, projected onto the wind parts of the Wheeler &amp;
  Hendon (2004) EOFs and normalised to unit variance. OLR is left out because AIFS-ENS does not output it; the GEFS and GEPS pages
  also carry their full three-field RMM. The GEFS track uses its own analyses for the 120-day filter; the others use the site&rsquo;s
  ERA5 filter map. The observed track is ERA5 up to mid-July 2026 and the AIFS-ENS analysis since. ECMWF&rsquo;s day 0 is the
  analysis; GEFS and GEPS day 1 is the first forecast day, so their tracks start a day in.</p>
  <p>Since 2014 the standard RMM no longer removes ENSO; while an El Ni&ntilde;o grows, the 120-day mean trails the warming and the
  standing convection over the central Pacific projects onto phases 6&ndash;8, which reads as a stalled MJO. The ENSO-removed index
  subtracts &Delta;N34&nbsp;&times;&nbsp;c, where &Delta;N34 is Ni&ntilde;o-3.4 (NOAA OISST, 31-day mean) minus its mean over the
  same 120-day window, and c is the regression of the 120-day-filtered equatorial winds on &Delta;N34, longitude by longitude,
  projected onto the MJO patterns, fitted on ERA5 (ARCO) 2003&ndash;2024 with ERSST&nbsp;v6 Ni&ntilde;o-3.4. {_validation_text()} The
  forecast holds Ni&ntilde;o-3.4 at its latest value, since ENSO changes little in 15 days. The correction is linear, so it removes
  part of a growing El Ni&ntilde;o&rsquo;s projection, not all of it; the status line therefore also quotes NOAA&rsquo;s OLR-based
  real-time OMI (ROMI), which removes the previous 40 days and so is far less sensitive to slow signals. ECMWF&rsquo;s 12Z runs are
  offset-corrected onto the 00Z family as in the successive-runs view.</p>
  <p class="meta" style="margin-top:0.6rem;padding-top:0.6rem">Data: ECMWF AIFS-ENS and IFS-ENS open data, &copy;&nbsp;2026 ECMWF,
  source <a href="https://www.ecmwf.int">www.ecmwf.int</a>, <a href="https://creativecommons.org/licenses/by/4.0/">CC&nbsp;BY&nbsp;4.0</a>,
  modified; ECMWF does not accept any liability for errors or omissions in the data, their availability, or any loss or damage
  arising from their use. NOAA GEFS (AWS open data). ECCC GEPS (MSC Datamart; Data Source: Environment and Climate Change Canada).
  ERA5 (ARCO), contains modified Copernicus Climate Change Service information 2026; NOAA OISST v2.1 and ERSST v6 (NOAA NCEI /
  PSL); ROMI (NOAA PSL / CPC OLR); Bureau of Meteorology RMM for validation.</p>
  </details>
  <script>{SCRIPT}</script>
  </section>
"""
