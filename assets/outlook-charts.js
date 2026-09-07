/* outlook-charts.js — one Plotly theme for the outlook pages' time-series charts
   (seas5.html, sfs.html). Built 2026-09-07 after the user's note that the interactive
   plots "show far too much history (if it even exists) and are way too faint".

   What it fixes, in one place:
   - x-range: at most HISTORY_MONTHS of observed history before the issue month, then
     every forecast month; a chart with no observed series starts at its first forecast
     month. Lead-day charts show the days 16–46 window only.
   - y-range: fitted to the visible data (fans, means, observed, previous issues, members
     when drawn) with 8 % padding — the hindcast ±1σ band never sets the axis.
   - lines: sienna mean at width 4, members at 0.42 alpha / 1.0 px, observed black at 3,
     previous issues in a fixed navy → teal sequence with the dash telling the age.
   - chrome: Inter 14, bold title top-left, legend above the plot, monthly ticks
     ("%b\n%Y"), firm zero line, light grid, unified hover with 2-decimal values, end
     labels on the current mean and on the last observed point.

   Usage (see the pages):
     var OC = window.OutlookCharts;
     var tr = [].concat(OC.climBand(x, sd), OC.prevTraces([...]), OC.fanTraces(x, s),
                        OC.memberTrace(x, OC.memberRows(s.members, x.length)),
                        OC.meanTrace(x, s.mean, "Sep 2026 issue"), OC.obsTrace(ox, oy));
     OC.react("ensoChart", tr, { title: "Niño-3.4 anomaly", yunits: "°C",
                                 xrange: OC.monthRange(valid, obsMonths) });
   Traces may carry private flags: _nofit (excluded from the y-fit), _endLabel (annotate
   the last point). They are stripped before Plotly sees them. */
(function () {
  "use strict";

  var C = {
    ink: "#1a1a1a", ink2: "#4a4744", muted: "#8a8680",
    grid: "#e8e6e0", rule: "#e2e0da", edge: "#c9c5bb", zero: "#666",
    accent: "#8a4b2a", warm: "#b4541f", cool: "#3f6f8f",
    mean: "#8a4b2a",
    fanOuter: "rgba(138,75,42,0.18)", fanInner: "rgba(138,75,42,0.34)",
    member: "rgba(138,75,42,0.42)",
    obs: "#111",
    prev: ["#3b2a8c", "#3f7fbf", "#39b3b0"],           // oldest first
    clim: "rgba(120,120,120,0.18)",
    ref: "#3f6f8f",
    cpc: "#8b1a1a"
  };
  var W = { mean: 4, member: 1.0, obs: 3, prev: 2.6, ref: 2.4, hind: 1.4 };
  var HISTORY_MONTHS = 6, PAD = 0.08;
  var MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

  // ── small utilities ────────────────────────────────────────────────────────────
  function isNum(v) { return typeof v === "number" && isFinite(v); }
  function monthDate(ym) { return ym.slice(0, 7) + "-15"; }            // "2026-09" → mid-month
  function monthStart(ym) { return ym.slice(0, 7) + "-01"; }
  function nextMonthStart(ym) {
    var y = parseInt(ym.slice(0, 4), 10), m = parseInt(ym.slice(5, 7), 10);
    if (m === 12) { y += 1; m = 1; } else m += 1;
    return y + "-" + (m < 10 ? "0" : "") + m + "-01";
  }
  function shiftMonths(ym, k) {                                           // "2026-09", -6 → "2026-03"
    var y = parseInt(ym.slice(0, 4), 10), m = parseInt(ym.slice(5, 7), 10) - 1 + k;
    y += Math.floor(m / 12); m = ((m % 12) + 12) % 12;
    return y + "-" + (m + 1 < 10 ? "0" : "") + (m + 1);
  }
  function mmm(ym) { return MONTHS[parseInt(ym.slice(5, 7), 10) - 1] + " " + ym.slice(0, 4); }
  function fmt(v, d, signed) {
    if (!isNum(v)) return "—";
    var s = v.toFixed(d == null ? 2 : d);
    return signed !== false && v > 0 ? "+" + s : s;
  }
  function xms(x) { return typeof x === "number" ? x : Date.parse(x); }
  function hoverTemplate(units, digits, signed) {
    var d = digits == null ? 2 : digits;
    return "%{y:" + (signed === false ? "" : "+") + "." + d + "f}" + (units ? " " + units : "");
  }
  // Plotly 2.35 ignores the "+" sign flag in "%{y:+.2f}" (the value then renders at full precision), so
  // hover values are pre-formatted into customdata: signed, fixed decimals, units.
  function hoverSpec(y, o) {
    o = o || {};
    if (o.hover) return { hovertemplate: o.hover };
    var d = o.digits == null ? 2 : o.digits, u = o.units ? " " + o.units : "";
    return { customdata: (y || []).map(function (v) { return isNum(v) ? fmt(v, d, o.signed) + u : ""; }), hovertemplate: "%{customdata}" };
  }
  function withHover(t, y, o) { var h = hoverSpec(y, o); for (var k in h) t[k] = h[k]; return t; }
  // members come as [member][lead] or [lead][member]; return [member][lead] for nx leads
  function memberRows(members, nx) {
    if (!members || !members.length) return [];
    var inner = members[0] && members[0].length;
    if (inner === nx && members.length !== nx) return members;             // already [member][lead]
    if (members.length === nx) {                                           // [lead][member] → transpose
      var out = [];
      for (var m = 0; m < inner; m++) out.push(members.map(function (row) { return row[m]; }));
      return out;
    }
    return members;
  }
  function transposeQuantiles(rows, ps) {                                  // per-lead percentiles from [member][lead]
    var nx = rows[0].length, out = {}; ps.forEach(function (p) { out["p" + p] = []; }); out.mean = [];
    for (var L = 0; L < nx; L++) {
      var pool = rows.map(function (r) { return r[L]; }).filter(isNum).sort(function (a, b) { return a - b; });
      if (!pool.length) { ps.forEach(function (p) { out["p" + p].push(null); }); out.mean.push(null); continue; }
      ps.forEach(function (p) {
        var i = (pool.length - 1) * p / 100, lo = Math.floor(i), hi = Math.ceil(i);
        out["p" + p].push(pool[lo] + (pool[hi] - pool[lo]) * (i - lo));
      });
      out.mean.push(pool.reduce(function (a, b) { return a + b; }, 0) / pool.length);
    }
    return out;
  }

  // ── x-range rules ──────────────────────────────────────────────────────────────
  // Monthly charts: [first shown month-01, last forecast month + 1-01]. Observed history
  // is capped at HISTORY_MONTHS months before the first forecast month; without an
  // observed series the axis starts at the first forecast month.
  function monthRange(forecastMonths, obsMonths, history) {
    var h = history == null ? HISTORY_MONTHS : history;
    var f0 = forecastMonths[0].slice(0, 7), f1 = forecastMonths[forecastMonths.length - 1].slice(0, 7);
    var start = f0;
    if (obsMonths && obsMonths.length) {
      var floor = shiftMonths(f0, -h), lastObs = obsMonths[obsMonths.length - 1].slice(0, 7);
      var firstObs = obsMonths[0].slice(0, 7);
      start = firstObs > floor ? firstObs : floor;
      if (lastObs < start) start = lastObs;                                // a short tail still shows its last point
      if (start > f0) start = f0;
    }
    return [monthStart(start), nextMonthStart(f1)];
  }
  // the observed months to draw for a monthly chart (same cap)
  function obsWindow(forecastMonths, obsMonths, history) {
    var r = monthRange(forecastMonths, obsMonths, history), s = r[0].slice(0, 7);
    var out = []; (obsMonths || []).forEach(function (m, i) { if (m.slice(0, 7) >= s) out.push(i); });
    return out;                                                            // indices into obsMonths
  }
  // Lead-day charts: dates of the window, tick every `every`-th point, "Sep 1 / day 16" labels
  function dayAxis(dates, firstDay, every) {
    var n = dates.length, e = every || 2, tv = [], tt = [];
    var stepDays = n > 1 ? Math.max(1, Math.round((Date.parse(dates[1]) - Date.parse(dates[0])) / 86400000)) : 1;
    for (var i = 0; i < n; i += e) {
      var d = new Date(dates[i] + "T00:00:00Z");
      tv.push(dates[i]); tt.push(MONTHS[d.getUTCMonth()] + " " + d.getUTCDate() + "<br>day " + (firstDay + i * stepDays));
    }
    var pad = 0.6 * 86400000 * stepDays;
    return { tickvals: tv, ticktext: tt, stepDays: stepDays,
             range: [new Date(Date.parse(dates[0]) - pad).toISOString().slice(0, 10), new Date(Date.parse(dates[n - 1]) + pad).toISOString().slice(0, 10)] };
  }

  // ── trace builders ─────────────────────────────────────────────────────────────
  function band(x, lo, hi, name, fill, extra) {
    var a = { x: x, y: hi, type: "scatter", mode: "lines", line: { width: 0 }, hoverinfo: "skip", showlegend: false, name: name };
    var b = { x: x, y: lo, type: "scatter", mode: "lines", line: { width: 0 }, fill: "tonexty", fillcolor: fill, hoverinfo: "skip", name: name, showlegend: true };
    if (extra) { for (var k in extra) { a[k] = extra[k]; b[k] = extra[k]; } }
    return [a, b];
  }
  // hindcast / climatology band: sd (± around base) or explicit lo/hi
  function climBand(x, sd, base, name) {
    var b0 = base || sd.map(function () { return 0; });
    var hi = sd.map(function (v, i) { return isNum(v) && isNum(b0[i]) ? b0[i] + v : null; });
    var lo = sd.map(function (v, i) { return isNum(v) && isNum(b0[i]) ? b0[i] - v : null; });
    return band(x, lo, hi, name || "hindcast ±1σ", C.clim, { _nofit: true });
  }
  function climBandLoHi(x, lo, hi, name) { return band(x, lo, hi, name || "hindcast 10–90%", C.clim, { _nofit: true }); }
  // ensemble fans: s = {p10, p25, p75, p90}
  function fanTraces(x, s, opts) {
    var o = opts || {};
    return band(x, s.p10, s.p90, o.outerName || "P10–P90", o.outerFill || C.fanOuter)
      .concat(band(x, s.p25, s.p75, o.innerName || "P25–P75", o.innerFill || C.fanInner));
  }
  // one trace for all members, null-separated (one legend entry, one hover-free layer)
  function memberTrace(x, rows, opts) {
    var o = opts || {}, gx = [], gy = [];
    rows.forEach(function (r) {
      for (var i = 0; i < x.length; i++) { gx.push(x[i]); gy.push(isNum(r[i]) ? r[i] : null); }
      gx.push(x[x.length - 1]); gy.push(null);
    });
    return { x: gx, y: gy, type: "scatter", mode: "lines", line: { color: o.color || C.member, width: o.width || W.member },
             hoverinfo: "skip", name: o.name || ("members (" + rows.length + ")"), showlegend: o.showlegend !== false };
  }
  function meanTrace(x, y, name, opts) {
    var o = opts || {};
    return withHover({ x: x, y: y, type: "scatter", mode: "lines+markers", name: name,
             line: { color: o.color || C.mean, width: o.width || W.mean },
             marker: { size: o.marker || 7, color: o.color || C.mean, line: { color: "#fff", width: 1 } },
             _endLabel: o.endLabel !== false, _labelDigits: o.digits, _labelSigned: o.signed, _labelUnits: o.labelUnits }, y, o);
  }
  function obsTrace(x, y, name, opts) {
    var o = opts || {};
    return withHover({ x: x, y: y, type: "scatter", mode: "lines+markers", name: name || "observed",
             line: { color: o.color || C.obs, width: o.width || W.obs }, marker: { size: o.marker || 5, color: o.color || C.obs },
             _endLabel: o.endLabel !== false, _labelDigits: o.digits, _labelSigned: o.signed, _labelUnits: o.labelUnits }, y, o);
  }
  // previous issues, oldest first: [{x, y, name}] → navy/blue/teal, dot → dash → solid
  function prevTraces(list, opts) {
    var o = opts || {}, n = list.length, dashes = ["dot", "dash", "solid"].slice(Math.max(0, 3 - n));
    return list.map(function (p, k) {
      return withHover({ x: p.x, y: p.y, type: "scatter", mode: "lines", name: p.name,
               line: { color: C.prev[Math.min(k, C.prev.length - 1)], width: W.prev, dash: dashes[Math.min(k, dashes.length - 1)] || "solid" } }, p.y, o);
    });
  }
  // reference line (C3S overlay, hindcast mean, …): cool, dashed
  function refTrace(x, y, name, opts) {
    var o = opts || {};
    return withHover({ x: x, y: y, type: "scatter", mode: o.markers ? "lines+markers" : "lines", name: name,
             line: { color: o.color || C.ref, width: o.width || W.ref, dash: o.dash || "dash" },
             marker: { size: 5, symbol: o.symbol || "diamond", color: o.color || C.ref }, _nofit: !!o.nofit }, y, o);
  }
  function hindMeanTrace(x, y, name) {                                     // grey dashed climatological mean
    return { x: x, y: y, type: "scatter", mode: "lines", name: name || "hindcast mean",
             line: { color: C.muted, width: W.hind, dash: "dash" }, hoverinfo: "skip" };
  }
  function markerTrace(x, y, name, opts) {                                 // e.g. the published CPC index
    var o = opts || {};
    return withHover({ x: x, y: y, type: "scatter", mode: "markers", name: name,
             marker: { symbol: o.symbol || "x", size: o.size || 8, color: o.color || C.cpc, line: { width: 1.5, color: o.color || C.cpc } } }, y, o);
  }
  function hline(y, color, dash, width) {
    return { type: "line", xref: "paper", x0: 0, x1: 1, y0: y, y1: y, layer: "below",
             line: { color: color || "rgba(74,71,68,0.35)", width: width || 1, dash: dash || "dot" } };
  }

  // ── y-fit ─────────────────────────────────────────────────────────────────────
  function yrange(traces, xr, includeZero) {
    var lo = Infinity, hi = -Infinity, x0 = xr ? xms(xr[0]) : -Infinity, x1 = xr ? xms(xr[1]) : Infinity;
    traces.forEach(function (t) {
      if (t._nofit || !t.y) return;
      for (var i = 0; i < t.y.length; i++) {
        var v = t.y[i]; if (!isNum(v)) continue;
        var xv = t.x ? xms(t.x[i]) : null;
        if (xv != null && isFinite(xv) && (xv < x0 || xv > x1)) continue;
        if (v < lo) lo = v; if (v > hi) hi = v;
      }
    });
    if (!isFinite(lo)) return null;
    if (includeZero !== false) { if (lo > 0) lo = 0; if (hi < 0) hi = 0; }
    var span = hi - lo || Math.abs(hi) || 1;
    return [lo - PAD * span, hi + PAD * span];
  }

  // ── layout ────────────────────────────────────────────────────────────────────
  function layout(o) {
    o = o || {};
    var xaxis = {
      gridcolor: C.grid, linecolor: C.rule, tickcolor: C.edge, zeroline: false,
      showspikes: true, spikecolor: "#bbb", spikethickness: 1, spikemode: "across", spikedash: "solid",
      tickfont: { size: 12.5, color: C.ink2 }, range: o.xrange, automargin: true
    };
    if (o.xtype === "day") {                                               // lead-day window
      xaxis.type = "date"; xaxis.tickmode = "array"; xaxis.tickvals = o.tickvals; xaxis.ticktext = o.ticktext;
      xaxis.hoverformat = "%a %b %-d, %Y";
    } else if (o.xtype === "linear") {
      xaxis.type = "linear";
    } else {                                                               // monthly dates
      xaxis.type = "date"; xaxis.tickformat = "%b\n%Y"; xaxis.dtick = "M1"; xaxis.tick0 = "2000-01-15";
      xaxis.hoverformat = "%b %Y";
    }
    var L = {
      margin: { l: 62, r: 70, t: 84, b: 54 }, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "#fff",
      font: { family: "Inter, sans-serif", size: 14, color: C.ink }, showlegend: true,
      title: { text: o.title || "", x: 0, xanchor: "left", xref: "paper", y: 1, yref: "container", yanchor: "top",
               pad: { t: 6 }, font: { family: "Inter, sans-serif", size: 15, weight: 700, color: C.ink } },
      legend: { orientation: "h", x: 0, xanchor: "left", y: 1.02, yanchor: "bottom", font: { size: 12.5, color: C.ink2 },
                itemwidth: 30, traceorder: "normal", bgcolor: "rgba(0,0,0,0)" },
      xaxis: xaxis,
      yaxis: { title: { text: o.yunits || "", standoff: 8, font: { size: 13, color: C.ink2 } }, gridcolor: C.grid, linecolor: C.rule,
               zeroline: true, zerolinecolor: C.zero, zerolinewidth: 1.6, tickfont: { size: 12.5, color: C.ink2 },
               range: o.yrange, fixedrange: false, automargin: true },
      hovermode: "x unified",
      hoverlabel: { bgcolor: "#fff", bordercolor: C.edge, font: { family: "Inter, sans-serif", size: 12.5, color: C.ink }, align: "left" },
      shapes: o.shapes || [], annotations: o.annotations || []
    };
    if (o.height) L.height = o.height;
    return L;
  }

  // end labels on traces flagged _endLabel (last finite point inside the x-range)
  function endLabels(traces, xr) {
    var out = [], x0 = xr ? xms(xr[0]) : -Infinity, x1 = xr ? xms(xr[1]) : Infinity;
    traces.forEach(function (t) {
      if (!t._endLabel || !t.y) return;
      for (var i = t.y.length - 1; i >= 0; i--) {
        var v = t.y[i]; if (!isNum(v)) continue;
        var xv = xms(t.x[i]); if (isFinite(xv) && (xv < x0 || xv > x1)) continue;
        var color = (t.line && t.line.color) || C.ink;
        out.push({ x: t.x[i], y: v, xref: "x", yref: "y", text: fmt(v, t._labelDigits, t._labelSigned) + (t._labelUnits ? " " + t._labelUnits : ""),
                   showarrow: false, xanchor: "left", yanchor: "middle", xshift: 9,
                   font: { family: "Inter, sans-serif", size: 12.5, color: color, weight: 700 },
                   bgcolor: "rgba(255,255,255,0.85)", borderpad: 2 });
        break;
      }
    });
    return out;
  }

  function strip(traces) {
    return traces.map(function (t) {
      var c = {}; for (var k in t) if (k.charAt(0) !== "_") c[k] = t[k]; return c;
    });
  }

  // draw: traces + options → Plotly.react with the fitted y-range and the end labels
  // o: {title, yunits, xrange, xtype, tickvals, ticktext, includeZero, shapes, annotations, height, layout(fn)}
  function react(id, traces, o) {
    o = o || {};
    var yr = o.yrange || yrange(traces, o.xrange, o.includeZero);
    var L = layout({ title: o.title, yunits: o.yunits, xrange: o.xrange, xtype: o.xtype, tickvals: o.tickvals, ticktext: o.ticktext,
                     yrange: yr, shapes: o.shapes, annotations: (o.annotations || []).concat(o.endLabels === false ? [] : endLabels(traces, o.xrange)), height: o.height });
    if (o.layout) o.layout(L);
    return Plotly.react(id, strip(traces), L, { displayModeBar: false, responsive: true });
  }

  window.OutlookCharts = {
    colors: C, widths: W, HISTORY_MONTHS: HISTORY_MONTHS,
    monthDate: monthDate, monthStart: monthStart, shiftMonths: shiftMonths, mmm: mmm, fmt: fmt, isNum: isNum,
    hoverTemplate: hoverTemplate, hoverSpec: hoverSpec, memberRows: memberRows, quantiles: transposeQuantiles,
    monthRange: monthRange, obsWindow: obsWindow, dayAxis: dayAxis,
    band: band, climBand: climBand, climBandLoHi: climBandLoHi, fanTraces: fanTraces, memberTrace: memberTrace,
    meanTrace: meanTrace, obsTrace: obsTrace, prevTraces: prevTraces, refTrace: refTrace, hindMeanTrace: hindMeanTrace,
    markerTrace: markerTrace, hline: hline, yrange: yrange, layout: layout, endLabels: endLabels, react: react
  };
})();
