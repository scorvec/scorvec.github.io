// The national poll charts of the forecast and poll pages: every poll a dot (sized by its sample) about the fitted
// trend and its likely range, drawn at the container's own width so that text and dots stay sharp, with a time range,
// adjusted or published values, and a tooltip per poll. Needs d3 (v7).
(function () {
  "use strict";
  const PN = {
    yougov: "YouGov", "yougov/cbs": "CBS News/YouGov", "yougov/yahoo": "Yahoo News/YouGov", "ft/yougov": "Financial Times/YouGov",
    "cnn/ssrs": "CNN/SSRS", nbc: "NBC News", rmg: "RMG Research", ppp: "Public Policy Polling", harrisx: "HarrisX", tipp: "TIPP",
    siena: "New York Times/Siena", "jl partners": "J.L. Partners", atlasintel: "AtlasIntel", quantus: "Quantus Insights",
    echelon: "Echelon Insights", bullfinch: "The Bullfinch Group", marquette: "Marquette Law School",
    verasight: "Strength in Numbers/Verasight", "big data poll": "Big Data Poll", "morning consult": "Morning Consult",
    "fox news": "Fox News", ipsos: "Ipsos", "ap-norc": "AP-NORC", emerson: "Emerson", quinnipiac: "Quinnipiac",
  };
  const POP = { lv: "likely voters", rv: "registered voters", a: "adults", v: "voters" };
  const RANGES = [["3 months", 91], ["6 months", 182], ["1 year", 365], ["All", 0]];
  const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  const pname = (k) => PN[k] || String(k).replace(/\b([a-z])/g, (c) => c.toUpperCase());
  const day = (d) => new Date(d + "T12:00:00");
  let tip;

  function showTip(e, html) {
    if (!tip) {
      tip = document.createElement("div");
      tip.className = "pc-tip";
      document.body.appendChild(tip);
    }
    tip.innerHTML = html;
    tip.style.display = "block";
    const w = tip.offsetWidth, h = tip.offsetHeight;
    let x = e.clientX + 14, y = e.clientY + 14;
    if (x + w > innerWidth - 8) x = Math.max(8, e.clientX - w - 14);
    if (y + h > innerHeight - 8) y = Math.max(8, e.clientY - h - 14);
    tip.style.left = x + "px";
    tip.style.top = y + "px";
  }
  const hideTip = () => tip && (tip.style.display = "none");

  // o: {kind: "generic" | "approval", label, color, trend [[date, v]], band [[date, lo, hi]], polls [[date, adjusted,
  // published, pollster, pop, n]], t1 [date, v] (today), ci [lo, hi], marks [{label, v}] (published averages),
  // weights {key: {w, influence}} keyed by weightKey}
  function mount(box, o) {
    const fmtn = (v, k) => o.kind === "generic"
      ? (Math.abs(v) < 0.05 ? "Even" : (v > 0 ? "D+" : "R+") + Math.abs(v).toFixed(k))
      : (Math.abs(v) < 0.05 ? "0" : (v > 0 ? "+" : "−") + Math.abs(v).toFixed(k));
    const fmt = (v) => fmtn(v, 1), tick = (v) => fmtn(v, Number.isInteger(v) ? 0 : 1);
    const st = { range: box.clientWidth < 640 ? 91 : 182, adj: true };  // phones open on fewer, larger-spaced polls
    box.classList.add("pc");
    box.innerHTML =
      `<div class="pc-head"><p class="pc-now"></p><div class="pc-ctl" role="group" aria-label="${esc(o.label)} options">` +
      RANGES.map(([l, d]) => `<button type="button" data-r="${d}" aria-pressed="${d === st.range}">${l}</button>`).join("") +
      `<span class="pc-sep"></span><button type="button" data-a="1" aria-pressed="true">Adjusted</button>` +
      `<button type="button" data-a="0" aria-pressed="false">As published</button></div></div>` +
      `<svg class="pc-svg" role="img"></svg><p class="pc-note"></p>`;
    const polls = (o.polls || []).map((p) => ({ d: day(p[0]), date: p[0], adj: p[1], raw: p[2], who: p[3], pop: p[4], n: p[5] }));
    for (const p of polls) p.wt = o.weights ? o.weights[weightKey(p.date, p.who, p.pop, p.raw)] : null;
    const now = o.t1 ? [day(o.t1[0]), o.t1[1]] : null;
    const head = box.querySelector(".pc-now");
    head.innerHTML = now
      ? `${esc(o.label)}: <b style="color:${o.color}">${fmt(now[1])}</b> today` +
        (o.ci ? ` <span class="pc-dim">likely range ${fmt(o.ci[0])} to ${fmt(o.ci[1])}</span>` : "")
      : esc(o.label);
    box.querySelectorAll(".pc-ctl button").forEach((b) =>
      b.addEventListener("click", () => {
        if (b.dataset.r !== undefined) st.range = +b.dataset.r;
        else st.adj = b.dataset.a === "1";
        box.querySelectorAll(".pc-ctl button").forEach((x) =>
          x.setAttribute("aria-pressed", x.dataset.r !== undefined ? +x.dataset.r === st.range : (x.dataset.a === "1") === st.adj));
        draw();
      }));
    let lastW = 0;
    new ResizeObserver(() => {
      if (Math.abs(box.clientWidth - lastW) > 2) draw();
    }).observe(box);

    function draw() {
      const W = Math.max(300, box.clientWidth), narrow = W < 640;
      lastW = box.clientWidth;
      const H = narrow ? 300 : Math.round(Math.min(440, Math.max(320, W * 0.38)));
      const marks = narrow ? [] : (o.marks || []).filter((k) => !/^average$/i.test(k.label));
      const m = { l: 52, r: marks.length ? 150 : 22, t: 14, b: 30 };
      const svg = d3.select(box.querySelector(".pc-svg")).attr("width", W).attr("height", H).attr("viewBox", `0 0 ${W} ${H}`);
      svg.selectAll("*").remove();
      const end = new Date(Math.max(now ? +now[0] : 0, d3.max(polls, (p) => +p.d) || 0, Date.now() - 864e5));
      const first = d3.min([...polls.map((p) => +p.d), ...(o.trend || []).map((p) => +day(p[0]))]);
      const start = new Date(st.range ? +end - st.range * 864e5 : first);
      // the trend to today, today's value appended (the fitted series is stored at intervals)
      let tr = (o.trend || []).map(([d, v]) => [day(d), v]).filter((p) => p[0] >= start - 30 * 864e5);
      let bd = (o.band || []).map(([d, lo, hi]) => [day(d), lo, hi]).filter((p) => p[0] >= start - 30 * 864e5);
      if (now) {
        const at = (a, k) => {
          const i = d3.bisector((p) => p[0]).left(a, now[0]), p0 = a[Math.max(i - 1, 0)], p1 = a[Math.min(i, a.length - 1)];
          return p0[k] + (p1[k] - p0[k]) * ((now[0] - p0[0]) / (p1[0] - p0[0] || 1));
        };
        tr = tr.filter((p) => p[0] < now[0]).concat([now]);
        if (bd.length) bd = bd.filter((p) => p[0] < now[0]).concat([[now[0], at(bd, 1), at(bd, 2)]]);
      }
      const vis = polls.filter((p) => p.d >= start && p.d <= end);
      const val = (p) => (st.adj ? p.adj : p.raw);
      const xs = d3.scaleTime().domain([start, end]).range([m.l, W - m.r]);
      const ext = d3.extent(vis.map(val).concat(bd.flatMap((b) => [b[1], b[2]]), tr.map((p) => p[1]), marks.map((k) => k.v), [0]));
      const ys = d3.scaleLinear().domain(ext).nice().range([H - m.b, m.t]);
      const [lo, hi] = ys.domain();
      svg.append("g").attr("class", "pc-grid").selectAll("line").data(ys.ticks(narrow ? 5 : 7)).join("line")
        .attr("x1", m.l).attr("x2", W - m.r).attr("y1", ys).attr("y2", ys);
      svg.append("line").attr("class", "pc-zero").attr("x1", m.l).attr("x2", W - m.r).attr("y1", ys(0)).attr("y2", ys(0));
      const months = (+end - +start) / (30.4 * 864e5);
      const every = months <= 4 ? d3.timeWeek.every(2) : d3.timeMonth.every(months <= 8 ? 1 : months <= 14 ? 2 : 3);
      const xfmt = months <= 4 ? d3.timeFormat("%b %-d") : (d) => d3.timeFormat(d.getMonth() === 0 ? "%b %Y" : "%b")(d);
      svg.append("g").attr("class", "pc-ax").attr("transform", `translate(0,${H - m.b})`)
        .call(d3.axisBottom(xs).ticks(every).tickFormat(xfmt).tickSizeOuter(0));
      svg.append("g").attr("class", "pc-ax").attr("transform", `translate(${m.l},0)`)
        .call(d3.axisLeft(ys).ticks(narrow ? 5 : 7).tickFormat(tick).tickSizeOuter(0));
      const clip = "pc" + Math.random().toString(36).slice(2, 8);
      svg.append("clipPath").attr("id", clip).append("rect").attr("x", m.l).attr("y", m.t - 6).attr("width", W - m.l - m.r + 6).attr("height", H - m.t - m.b + 6);
      const g = svg.append("g").attr("clip-path", `url(#${clip})`);
      if (bd.length) g.append("path").attr("d", d3.area().x((p) => xs(p[0])).y0((p) => ys(p[1])).y1((p) => ys(p[2])).curve(d3.curveMonotoneX)(bd))
        .attr("fill", o.color).attr("opacity", 0.14);
      const rad = (p) => (narrow ? 0.8 : 1) * (p.n ? 2.6 + 3.4 * Math.sqrt(Math.min(p.n, 3000) / 3000) : 3.2);
      g.append("g").selectAll("circle").data(vis).join("circle")
        .attr("cx", (p) => xs(p.d)).attr("cy", (p) => ys(Math.max(lo, Math.min(hi, val(p))))).attr("r", rad)
        .attr("fill", o.color).attr("fill-opacity", 0.42).attr("stroke", o.color).attr("stroke-opacity", 0.75).attr("stroke-width", 0.8);
      g.append("path").attr("d", d3.line().x((p) => xs(p[0])).y((p) => ys(p[1])).curve(d3.curveMonotoneX)(tr))
        .attr("fill", "none").attr("stroke", o.color).attr("stroke-width", 2.8);
      if (now) svg.append("circle").attr("cx", xs(now[0])).attr("cy", ys(now[1])).attr("r", 5.5).attr("fill", o.color).attr("stroke", "#fff").attr("stroke-width", 1.5);
      if (marks.length) {
        const short = (l) => l.replace("Decision Desk HQ", "DDHQ").replace("FiftyPlusOne", "50+1");
        const ms = marks.map((k) => ({ ...k, y: ys(k.v) })).sort((a, b) => a.y - b.y);
        ms.forEach((k, i) => (k.ly = i ? Math.max(k.y, ms[i - 1].ly + 15) : k.y));
        const over = ms.length ? ms[ms.length - 1].ly - (H - m.b) : 0;
        if (over > 0) ms.forEach((k) => (k.ly -= over));
        ms.forEach((k) => {
          svg.append("path").attr("d", d3.symbol(d3.symbolDiamond, 46)()).attr("transform", `translate(${W - m.r + 10},${k.y})`).attr("fill", "#6f6b64");
          svg.append("line").attr("x1", W - m.r + 16).attr("x2", W - m.r + 24).attr("y1", k.y).attr("y2", k.ly).attr("stroke", "#b9b6ad");
          svg.append("text").attr("class", "pc-avg").attr("x", W - m.r + 27).attr("y", k.ly + 4).text(`${short(k.label)} ${fmt(k.v)}`);
        });
      }
      svg.attr("aria-label", `${o.label}: ${vis.length} polls since ${d3.timeFormat("%B %-d, %Y")(start)}` + (now ? `; the trend today ${fmt(now[1])}` : ""));
      box.querySelector(".pc-note").innerHTML =
        `${vis.length} polls. Each dot is one poll (bigger: a larger sample), ${st.adj ? "adjusted for its pollster's lean" : "as published"}; ` +
        `the line is the trend, shaded its likely range` + ((o.marks || []).length ? (narrow ? `. Published averages: ${(o.marks || []).filter((k) => !/^average$/i.test(k.label)).map((k) => `${esc(k.label)} ${fmt(k.v)}`).join(", ")}.` : "; diamonds are published averages.") : ".");
      // hover: the nearest poll within reach, else the trend at that date
      const guide = svg.append("g").style("display", "none");
      guide.append("line").attr("y1", m.t).attr("y2", H - m.b).attr("stroke", "#b9b6ad");
      const ring = svg.append("circle").attr("fill", "none").attr("stroke", "#1b1d22").attr("stroke-width", 1.6).style("display", "none");
      const pts = vis.map((p) => [xs(p.d), ys(Math.max(lo, Math.min(hi, val(p)))), p]);
      svg.append("rect").attr("x", m.l).attr("y", m.t).attr("width", W - m.l - m.r).attr("height", H - m.t - m.b).attr("fill", "transparent")
        .on("pointermove pointerdown", (e) => {
          const [mx, my] = d3.pointer(e);
          let best = null, bd2 = 18 * 18;
          for (const q of pts) {
            const d2 = (q[0] - mx) ** 2 + (q[1] - my) ** 2;
            if (d2 < bd2) (bd2 = d2), (best = q);
          }
          if (best) {
            const p = best[2];
            guide.style("display", "none");
            ring.style("display", null).attr("cx", best[0]).attr("cy", best[1]).attr("r", rad(p) + 2.5);
            showTip(e, `<b>${esc(pname(p.who))}</b><br>${d3.timeFormat("%b %-d, %Y")(p.d)} (middle of fieldwork) · ${esc(POP[p.pop] || p.pop)}` +
              (p.n ? ` · ${p.n.toLocaleString()}` : "") + `<br>As published ${fmt(p.raw)} · adjusted ${fmt(p.adj)}` +
              (p.wt ? `<br>Weight ${p.wt.w.toFixed(2)} · ${(100 * p.wt.influence).toFixed(1)}% of today's value` : ""));
          } else {
            ring.style("display", "none");
            const d = xs.invert(mx), i = d3.bisector((p) => p[0]).center(tr, d), p = tr[i];
            if (!p) return;
            guide.style("display", null).attr("transform", `translate(${xs(p[0])},0)`);
            showTip(e, `${d3.timeFormat("%b %-d, %Y")(p[0])}: trend <b>${fmt(p[1])}</b>`);
          }
        })
        .on("pointerleave", () => {
          guide.style("display", "none");
          ring.style("display", "none");
          hideTip();
        });
    }
  }

  // a poll's identity across the two files (data/model.json's trend polls, data/polls.json's weighted ones)
  const weightKey = (date, who, pop, raw) => `${date}|${who}|${pop}|${raw}`;
  window.PollChart = { mount, pname, weightKey, POP };
})();
