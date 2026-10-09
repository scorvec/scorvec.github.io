/* radar_glm.js (2026-10-09): GOES GLM lightning flashes for radar.html, read by the browser straight from NOAA's
   open-data buckets on AWS (noaa-goes19 / noaa-goes18, GLM-L2-LCFA, one netCDF-4 file per 20 s per satellite).
   Nothing is built on this site's side.

   A GLM file is 170-650 KB, almost all of it event and group records and metadata; the flashes are a few KB. So the
   page reads only the flash variables, with HTTP range requests:
     learn  (once per satellite per visit): one whole file is parsed with jsfive (a maintained pure-JS HDF5 reader,
            lazy-loaded from jsDelivr), which gives each flash variable's chunk-index (v1 B-tree) address, chunk size,
            filters (shuffle + deflate), type and CF attributes (scale_factor, add_offset, _FillValue, _Unsigned);
            the same file is then decoded again by the range reader below and the two must agree value for value.
     read   (every file): round 1 fetches flash_count, product_time and the five B-tree nodes; round 2 fetches the
            chunks those nodes point to. Every node is checked (signature, type, level, entry count = the chunks
            flash_count needs, chunk offsets), product_time must be the file name's start, every chunk must inflate
            to its exact size, and every value must be in range. About 10 requests and 5-15 KB per file.
   Where the addresses sit depends on the ground-system build that wrote the file, so a satellite keeps a few layout
   VARIANTS (keyed by a signature of the flash variables: types, chunking, filters, CF packing, the nodes' spacing),
   and each variant keeps candidate addresses for flash_count, product_time and the first node, each either
   absolute or counted back from the end of the file (the size comes from the bucket listing):
     2026 builds (shuffle + deflate, 2048/4096-element chunks): everything at fixed absolute offsets;
     2018-mid 2026 (no filters, 256-element chunks): product_time absolute, the B-tree nodes usually 26864 bytes
            before the end (a few files 20720 / 22768 before the end, or at 58930 from the start), flash_count
            79158-ish before the end until early 2024 and at 2939 since.
   The most recently successful candidates are tried first; if they fail, every known candidate is tried at once;
   if none fits, the file is learned (a sparse copy, else the whole file) and its addresses join the candidates.
   A wrong flash_count cannot slip through: the node entry counts must match it, and the element after it in the
   last chunk must be the fill value. The packing attributes (scale_factor, add_offset, _FillValue, _Unsigned) are
   another candidate address, compared byte for byte with the learned file's whenever no file of the variant within
   the hour had them checked (flash_energy's packing changed between 2019 and 2020 with nothing else moving).
   Early 2018 files count the time offsets in milliseconds, and late-2018 files lack _Unsigned on them.
   No DOM here: radar.html draws. Exposed as window.GLM. */
(function (root) {
  "use strict";
  var LIBS = ["https://cdn.jsdelivr.net/npm/jsfive@0.4.2/dist/browser/hdf5.min.js", "https://unpkg.com/jsfive@0.4.2/dist/browser/hdf5.js"];
  var J2000 = Date.UTC(2000, 0, 1, 12);
  // the operational satellite for each slot, newest first (sub-satellite longitude from the files' nominal_satellite_subpoint_lon)
  var SLOTS = {
    East: [{ id: "G19", name: "GOES-19", bucket: "noaa-goes19", sub: -75.2, from: Date.UTC(2025, 3, 7, 15) },
           { id: "G16", name: "GOES-16", bucket: "noaa-goes16", sub: -75.2, from: Date.UTC(2017, 11, 18, 17) }],
    West: [{ id: "G18", name: "GOES-18", bucket: "noaa-goes18", sub: -137.0, from: Date.UTC(2023, 0, 4, 18) },
           { id: "G17", name: "GOES-17", bucket: "noaa-goes17", sub: -137.2, from: Date.UTC(2019, 1, 12, 18) }] };
  // overlap rule: each flash is taken from the satellite whose sub-satellite point is nearer in longitude (both sit on
  // the equator), i.e. a meridian halfway between 75.2 W and 137.0 W. Smaller viewing angle = smaller parallax error.
  var BOUND = (-75.2 + -137.0) / 2;                                      // -106.1
  var FILE_S = 20e3;
  var VARS = ["flash_lat", "flash_lon", "flash_time_offset_of_first_event", "flash_energy", "flash_quality_flag"];
  var MERGE_GAP = 2048;                                                  // ranges closer than this are fetched as one

  var S = { bytes: 0, req: 0, files: 0, full: 0, fullBytes: 0, learned: 0, retries: 0, attrChecks: 0, cap: 40 * 1048576, capped: false, err: null, lib: null, libErr: null,
            learnMs: [], fileMs: [], windows: [] };
  var layouts = {};                                                      // sat id -> {ok, learning, variants: [newest success first], n}
  var MAX_HYP = 8, MAX_VAR = 6, NC_FILL_FLOAT = Math.fround(9.9692099683868690e36);
  var byUrl = {}, files = new Map();                                               // "G19:<s ms>" -> file record
  var wins = new Map();                                                  // "T|N" -> window
  var version = 0, onUpdate = null, updT = 0, projector = null;

  function p2(n) { return (n < 10 ? "0" : "") + n; }
  function p3(n) { return (n < 10 ? "00" : n < 100 ? "0" : "") + n; }
  function doy(d) { return Math.floor((Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate()) - Date.UTC(d.getUTCFullYear(), 0, 1)) / 864e5) + 1; }
  // GLM file-name time: sYYYYDDDHHMMSSt (t = tenths)
  function sStamp(t) { var d = new Date(t); return "" + d.getUTCFullYear() + p3(doy(d)) + p2(d.getUTCHours()) + p2(d.getUTCMinutes()) + p2(d.getUTCSeconds()) + Math.floor(d.getUTCMilliseconds() / 100); }
  function parseS(s) { var m = /^(\d{4})(\d{3})(\d\d)(\d\d)(\d\d)(\d)$/.exec(s); if (!m) return NaN;
    return Date.UTC(+m[1], 0, 1) + (+m[2] - 1) * 864e5 + +m[3] * 36e5 + +m[4] * 6e4 + +m[5] * 1e3 + +m[6] * 100; }
  function satFor(slot, t) { var L = SLOTS[slot]; for (var i = 0; i < L.length; i++) if (t >= L[i].from) return L[i]; return null; }
  function bucketUrl(sat) { return "https://" + sat.bucket + ".s3.amazonaws.com/"; }
  function bump() { version++; if (onUpdate && !updT) updT = setTimeout(function () { updT = 0; onUpdate(); }, 150); }
  function httpErr(r) { var e = new Error("HTTP " + r.status); e.status = r.status; return e; }

  // ── network: byte accounting and a small per-bucket queue (newest file first, urgent windows first) ───────────
  function get(url, range, hi) {
    var o = { mode: "cors", priority: hi ? "high" : "low" };
    if (range) o.headers = { Range: "bytes=" + range[0] + "-" + (range[1] - 1) };
    S.req++;
    return fetch(url, o).then(function (r) {
      if (!r.ok) throw httpErr(r);
      return r.arrayBuffer().then(function (b) {
        S.bytes += b.byteLength; var fo = byUrl[url]; if (fo) fo.bytes = (fo.bytes || 0) + b.byteLength;
        if (range && r.status !== 206) {                                 // the server ignored the range: take the slice
          if (b.byteLength < range[1]) throw new Error("short reply");
          return new Uint8Array(b, range[0], range[1] - range[0]);
        }
        if (range && b.byteLength !== range[1] - range[0]) throw new Error("range reply of " + b.byteLength + " bytes, asked " + (range[1] - range[0]));
        return new Uint8Array(b);
      });
    });
  }
  var queue = [], busy = {}, LANES = 3;
  function enqueue(task) { queue.push(task); pump(); }
  function pump() {
    if (S.bytes >= S.cap) { if (!S.capped) { S.capped = true; bump(); } return; }
    S.capped = false;
    queue = queue.filter(function (q) { if (q.alive()) return true; q.drop(); return false; });
    queue.sort(function (a, b) { return a.prio() - b.prio() || b.t - a.t; });
    for (var i = 0; i < queue.length; i++) {
      var q = queue[i], k = q.sat.bucket, Ls = satLayouts(q.sat);
      if ((busy[k] || 0) >= LANES) continue;
      if (!Ls.ok) {
        if (Ls.failAt && Date.now() - Ls.failAt < 60e3) { queue.splice(i--, 1); q.fail(Ls.err); continue; }
        if (Ls.learning) continue;                                       // the others wait for the layout
        // learn from the smallest file the most urgent window needs (it is downloaded whole)
        var p0 = q.prio(), best = i;
        for (var j = i + 1; j < queue.length; j++) if (queue[j].sat === q.sat && queue[j].prio() === p0 && queue[j].f.size < queue[best].f.size) best = j;
        q = queue[best]; queue.splice(best, 1); if (best === i) i--;
        q.learner = true; Ls.learning = true;
      } else { queue.splice(i--, 1); }
      busy[k] = (busy[k] || 0) + 1;
      q.run().then(null, function () {}).then(function (q2) { return function () { busy[q2.sat.bucket]--; pump(); }; }(q));
      if (S.bytes >= S.cap) break;
    }
  }

  // ── the HDF5 bits the range reader needs ─────────────────────────────────────────────────────────────────────────
  function u64(dv, o) { var lo = dv.getUint32(o, true), hi = dv.getUint32(o + 4, true); return hi === 0xffffffff && lo === 0xffffffff ? -1 : hi * 4294967296 + lo; }
  async function inflate(bytes) {
    var ds = new DecompressionStream("deflate");
    return new Uint8Array(await new Response(new Blob([bytes]).stream().pipeThrough(ds)).arrayBuffer());
  }
  function unshuffle(b, esz) {
    if (esz === 1) return b;
    var n = b.length / esz, o = new Uint8Array(b.length);
    for (var j = 0; j < esz; j++) { var src = j * n; for (var i = 0; i < n; i++) o[i * esz + j] = b[src + i]; }
    return o;
  }
  // v1 B-tree node of raw-data chunks, rank 1 (+ the element dimension): 24-byte header, then per entry a key
  // (chunk size u32, filter mask u32, 2 x u64 offsets) and a child address u64. The writer stores exactly the chunks
  // the data fill (checked on 755 nodes of 151 files, 2018-2026), so the entry count also checks flash_count.
  function btreeNode(bytes, nNeed) {
    var dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    if (bytes.length < 24 || String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3]) !== "TREE") throw new Error("no B-tree node at the learned address");
    if (bytes[4] !== 1 || bytes[5] !== 0) throw new Error("unexpected B-tree node (type " + bytes[4] + ", level " + bytes[5] + ")");
    var n = dv.getUint16(6, true), out = [];
    if (n !== nNeed) throw new Error("B-tree node lists " + n + " chunks, " + nNeed + " needed");
    if (24 + nNeed * 32 > bytes.length) return { more: 24 + n * 32 };
    for (var i = 0; i < nNeed; i++) {
      var k = 24 + i * 32;
      out.push({ size: dv.getUint32(k, true), mask: dv.getUint32(k + 4, true), off: u64(dv, k + 8), addr: u64(dv, k + 24) });
    }
    return { chunks: out };
  }
  function mergeRanges(rs) {
    rs = rs.slice().sort(function (a, b) { return a[0] - b[0]; });
    var out = [];
    rs.forEach(function (r) { var l = out[out.length - 1]; if (l && r[0] - l[1] <= MERGE_GAP) l[1] = Math.max(l[1], r[1]); else out.push([r[0], r[1]]); });
    return out;
  }
  // fetch a set of byte ranges (merged) and return a reader for any address inside them
  function fetchRanges(url, rs, hi) {
    var m = mergeRanges(rs);
    return Promise.all(m.map(function (r) { return get(url, r, hi); })).then(function (parts) {
      return function (a, n) {
        for (var i = 0; i < m.length; i++) if (a >= m[i][0] && a + n <= m[i][1]) return parts[i].subarray(a - m[i][0], a - m[i][0] + n);
        throw new Error("address " + a + " not fetched");
      };
    });
  }
  function readNum(dv, o, kind, esz, le) {
    if (kind === "f") return esz === 4 ? dv.getFloat32(o, le) : dv.getFloat64(o, le);
    if (kind === "u") return esz === 2 ? dv.getUint16(o, le) : esz === 4 ? dv.getUint32(o, le) : dv.getUint8(o);
    return esz === 2 ? dv.getInt16(o, le) : esz === 4 ? dv.getInt32(o, le) : dv.getInt8(o);
  }
  // CF decoding of a raw value: _Unsigned, _FillValue (compared on the stored bits), scale_factor, add_offset, and
  // the unit factor `mul` (milliseconds -> seconds for the time offsets of early 2018 files)
  function cf(v, L, raw) {
    if (L.fill != null && raw === L.fill) return NaN;
    return (L.scale != null ? v * L.scale + (L.offset || 0) : v + (L.offset || 0)) * (L.mul || 1);
  }

  // decode one variable from its chunks; `read(addr, n)` returns stored bytes
  async function decodeVar(L, chunks, read, N) {
    var out = new Float64Array(N);
    for (var c = 0; c < chunks.length; c++) {
      var ch = chunks[c], b = read(ch.addr, ch.size);
      var stages = L.filters.slice();                                    // applied on write in this order; undone in reverse
      for (var f = stages.length - 1; f >= 0; f--) {
        if (ch.mask & (1 << f)) continue;
        if (stages[f] === 1) b = await inflate(b);
        else if (stages[f] === 2) b = unshuffle(b, L.esz);
        else throw new Error("filter " + stages[f] + " not supported");
      }
      if (b.length !== L.chunk * L.esz) throw new Error(L.name + ": chunk inflates to " + b.length + " bytes, expected " + L.chunk * L.esz);
      if (ch.off !== c * L.chunk) throw new Error(L.name + ": chunk offset " + ch.off);
      var dv = new DataView(b.buffer, b.byteOffset, b.byteLength), n = Math.min(L.chunk, N - ch.off);
      // the writer leaves the rest of the last chunk at the fill value: for the float positions (no _FillValue, so
      // netCDF's default 9.96921e36) the element after the count must be fill, which pins flash_count exactly
      if (n < L.chunk && L.kind === "f" && L.esz === 4 && L.fill == null && readNum(dv, n * 4, "f", 4, L.le) !== NC_FILL_FLOAT)
        throw new Error(L.name + ": data continue past flash_count");
      for (var i = 0; i < n; i++) {
        var raw = readNum(dv, i * L.esz, L.kind, L.esz, L.le), v = raw;
        if (L.unsigned && L.kind === "i") v = raw < 0 ? raw + Math.pow(2, 8 * L.esz) : raw;
        out[ch.off + i] = cf(v, L, raw);
      }
    }
    return out;
  }
  function pack(file, N, lat, lon, dt, e, q, ptime) {
    var sub = file.sat && file.sat.sub != null ? file.sat.sub * Math.PI / 180 : null, R = Math.PI / 180, cosMax = Math.cos(84 * R);
    for (var i = 0; i < N; i++) {                                        // plausibility of every value (NaN fails too)
      if (!(lat[i] >= -90 && lat[i] <= 90 && lon[i] >= -180 && lon[i] <= 180)) throw new Error("flash position out of range");
      // within the satellite's disk (the limb is 81.3 degrees of arc from the sub-satellite point)
      if (sub != null && !(Math.cos(lat[i] * R) * Math.cos(lon[i] * R - sub) >= cosMax)) throw new Error("flash outside the satellite's view");
      if (!(dt[i] > -6 && dt[i] < 30)) throw new Error("flash time offset out of range (" + dt[i] + " s)");
      if (!(q[i] === 0 || q[i] === 1 || q[i] === 3 || q[i] === 5)) throw new Error("unknown quality flag " + q[i]);
      if (!(e[i] > 0 && e[i] < 1e-9) && !isNaN(e[i])) throw new Error("flash energy out of range (" + e[i] + " J)");
    }
    var r = { n: N, lat: new Float32Array(lat), lon: new Float32Array(lon), dt: dt, e: new Float32Array(e), q: new Uint8Array(N), t: new Float64Array(N) };
    for (i = 0; i < N; i++) r.q[i] = q[i];
    // the offsets count from time_coverage_start (= product_time, seconds since 2000-01-01 12:00, no leap seconds),
    // which must also be the start time in the file name
    var t0 = J2000 + ptime * 1000;
    if (Math.abs(t0 - file.s) > 60) throw new Error("product_time " + new Date(t0).toISOString() + " is not the file's start");
    for (i = 0; i < N; i++) r.t[i] = t0 + dt[i] * 1000;
    return r;
  }

  // ── layout variants and their candidate addresses ───────────────────────────────────────────────────────────────
  // a candidate ("hypothesis") is {end, v}: the address is v, or size - v when end is set
  function satLayouts(sat) { return layouts[sat.id] || (layouts[sat.id] = { ok: false, n: 0, learning: false, variants: [], sparseMiss: 0 }); }
  function at(h, size, len) { var a = h.end ? size - h.v : h.v; return a >= 0 && a + len <= size ? a : -1; }
  function hypIndex(list, h) { for (var i = 0; i < list.length; i++) if (list[i].end === h.end && list[i].v === h.v) return i; return -1; }
  function promote(list, h) { var i = hypIndex(list, h); if (i > 0) { list.splice(i, 1); list.unshift(h); } }
  function addHyps(list, abs, size) {                                    // the learned file's address, both ways, in front
    [{ end: true, v: size - abs }, { end: false, v: abs }].forEach(function (h) {
      var i = hypIndex(list, h); if (i >= 0) list.splice(i, 1);
      list.unshift(h);
    });
    if (list.length > MAX_HYP) list.length = MAX_HYP;
  }
  // the signature of a layout: everything about the flash variables except where they are (the packing attributes'
  // messages byte for byte, and their spacing)
  function signature(L) {
    var b0 = L.vars.flash_lat.btree;
    return JSON.stringify([VARS.map(function (v) { var x = L.vars[v]; return [v, x.btree - b0, x.chunk, x.kind, x.esz, x.le, x.filters, x.scale, x.offset, x.fill, x.unsigned, x.mul || 1]; }),
      L.attrs.map(function (x) { return [x[0] - L.attr0, Array.prototype.join.call(x[1], ",")]; })]);
  }
  // round 1 for a set of candidates: read them all, keep the combination that validates. Returns null if none does.
  function nodeLen(Lv) { return 24 + 32 * Math.max(2, Math.ceil(1024 / Lv.chunk)); }   // up to 1024 flashes in one read
  async function tryLayout(file, V, H, hi, checkAttrs) {
    var url = bucketUrl(file.sat) + file.key, S = file.size, b0 = V.vars.flash_lat.btree, rs = [];
    function cands(list, len) { var out = [], seen = {}; list.forEach(function (h) { var a = at(h, S, len); if (a >= 0 && !seen[a]) { seen[a] = 1; out.push([h, a]); } }); return out; }
    var cC = cands(H.cnt, 4), cP = cands(H.ptime, 8), cA = checkAttrs ? cands(H.at, V.attrSpan) : [], cB = cands(H.bt, 24).filter(function (x) {
      return VARS.every(function (v) { var a = x[1] + V.vars[v].btree - b0; return a >= 0 && a + 24 <= S; }); });
    if (!cC.length || !cP.length || !cB.length || (checkAttrs && !cA.length)) return null;
    cC.forEach(function (x) { rs.push([x[1], x[1] + 4]); });
    cP.forEach(function (x) { rs.push([x[1], x[1] + 8]); });
    cA.forEach(function (x) { V.attrs.forEach(function (m) { rs.push([x[1] + m[0], x[1] + m[0] + m[1].length]); }); });
    cB.forEach(function (x) { VARS.forEach(function (v) { var a = x[1] + V.vars[v].btree - b0; rs.push([a, Math.min(S, a + nodeLen(V.vars[v]))]); }); });
    var rd = await fetchRanges(url, rs, hi);
    // product_time: seconds since 2000-01-01 12:00 that must be the start in the file name
    var hp = null, ptime;
    for (var i = 0; i < cP.length && !hp; i++) { var t = new DataView(rd(cP[i][1], 8).slice().buffer).getFloat64(0, true); if (Math.abs(J2000 + t * 1000 - file.s) <= 60) { hp = cP[i][0]; ptime = t; } }
    if (!hp) return null;
    // the packing attributes (scale_factor, add_offset, _FillValue, _Unsigned of the flash variables) must be the
    // learned file's byte for byte: the same layout has carried different packings (flash_energy changed in 2019-20)
    var ha = checkAttrs ? null : true;
    for (i = 0; i < cA.length && !ha; i++) {
      if (V.attrs.every(function (m) { var b = rd(cA[i][1] + m[0], m[1].length); for (var j = 0; j < b.length; j++) if (b[j] !== m[1][j]) return false; return true; })) ha = cA[i][0];
    }
    if (!ha) return null;
    // flash_count and the nodes: the node entry counts must be exactly the chunks the count needs, and round 2 then
    // checks the count against the fill after it. A count of 0 has nothing to check it against, so it is not taken
    // from a candidate (the file is learned instead). At most 4 combinations go to round 2.
    var tries = 0;
    for (var c = 0; c < cC.length && tries < 4; c++) {
      var N = new DataView(rd(cC[c][1], 4).slice().buffer).getInt32(0, true);
      if (!(N > 0 && N < 200000)) continue;
      for (var b = 0; b < cB.length && tries < 4; b++) {
        var plan = {}, extra = [], ok = true;
        for (var k = 0; k < VARS.length && ok; k++) {
          var Lv = V.vars[VARS[k]], a = cB[b][1] + Lv.btree - b0, need = Math.ceil(N / Lv.chunk);
          try { var node = btreeNode(rd(a, Math.min(S, a + nodeLen(Lv)) - a), need); if (node.more) extra.push([VARS[k], a, node.more, need]); else plan[VARS[k]] = node.chunks; }
          catch (e) { ok = false; }
        }
        if (!ok) continue;
        tries++;
        try {
          if (extra.length) {                                            // more chunks than one read covers: the full nodes
            var rx = await fetchRanges(url, extra.map(function (x) { return [x[1], x[1] + x[2]]; }), hi);
            extra.forEach(function (x) { plan[x[0]] = btreeNode(rx(x[1], x[2]), x[3]).chunks; });
          }
          var r = await readChunks(file, V, N, plan, ptime, hi);
          return { r: r, hc: cC[c][0], hp: hp, hb: cB[b][0], ha: ha };
        } catch (e) { if (e.status) throw e; }
      }
    }
    return null;
  }
  // round 2: the chunks, decoded and checked
  async function readChunks(file, V, N, plan, ptime, hi) {
    var r2 = [], vals = {};
    VARS.forEach(function (v) { plan[v].forEach(function (c) { if (!(c.addr > 0 && c.size > 0 && c.size < 1e6 && c.addr + c.size <= file.size)) throw new Error("bad chunk record"); r2.push([c.addr, c.addr + c.size]); }); });
    var rd2 = await fetchRanges(bucketUrl(file.sat) + file.key, r2, hi);
    for (var k = 0; k < VARS.length; k++) vals[VARS[k]] = await decodeVar(V.vars[VARS[k]], plan[VARS[k]], rd2, N);
    return pack(file, N, vals.flash_lat, vals.flash_lon, vals.flash_time_offset_of_first_event, vals.flash_energy, vals.flash_quality_flag, ptime);
  }
  function needLearn(msg) { var e = new Error(msg); e.learn = true; return e; }
  // the range reader: the most recently successful candidates first, then every known candidate of every variant.
  // The packing attributes are checked unless a file of this variant within the hour had them checked (packings
  // change with ground-system builds, months apart): about one extra check an hour on the live path.
  var ATTR_TRUST = 3600e3;
  async function readFast(file, Ls, hi) {
    var Vs = Ls.variants.filter(function (v) { return v.fast; }), V = Vs[0], got = null, chk;
    if (!V) throw needLearn("no range-readable layout");
    chk = !(file.s >= V.ver[0] - ATTR_TRUST && file.s <= V.ver[1] + ATTR_TRUST);
    got = await tryLayout(file, V, { cnt: V.cnt.slice(0, 1), ptime: V.ptime.slice(0, 1), bt: V.bt.slice(0, 1), at: V.at.slice(0, 1) }, hi, chk);
    for (var i = 0; !got && i < Vs.length; i++) { V = Vs[i]; chk = true; got = await tryLayout(file, V, V, hi, true); if (got) S.retries++; }
    if (!got) throw needLearn("no known layout fits");
    promote(V.cnt, got.hc); promote(V.ptime, got.hp); promote(V.bt, got.hb); V.uses++;
    if (chk) { promote(V.at, got.ha); V.ver[0] = Math.min(V.ver[0], file.s); V.ver[1] = Math.max(V.ver[1], file.s); S.attrChecks++; }
    var vi = Ls.variants.indexOf(V); if (vi > 0) { Ls.variants.splice(vi, 1); Ls.variants.unshift(V); }
    return got.r;
  }

  // ── learning a layout from one whole file (jsfive) ───────────────────────────────────────────────────────────────
  var libP = null;
  function loadLib() {
    if (S.lib) return Promise.resolve(S.lib);
    if (root.hdf5 && root.hdf5.File) { S.lib = root.hdf5; return Promise.resolve(S.lib); }
    if (libP) return libP;
    libP = new Promise(function (res, rej) {
      var i = 0;
      (function next() {
        if (i >= LIBS.length) { rej(new Error("the HDF5 reader (jsfive) could not be loaded")); return; }
        var s = document.createElement("script"); s.src = LIBS[i++]; s.async = true; s.crossOrigin = "anonymous";
        s.onload = function () { if (root.hdf5 && root.hdf5.File) { S.lib = root.hdf5; res(S.lib); } else next(); };
        s.onerror = next; document.head.appendChild(s);
      })();
    });
    libP.catch(function (e) { S.libErr = e.message; libP = null; });
    return libP;
  }
  // "dense" attribute storage (the netCDF writer uses it for variables with many attributes, and jsfive does not read
  // it): attribute-info message -> v2 B-tree of attribute names (record type 8: heap ID, flags, creation order, hash)
  // -> fractal heap with a direct root block -> attribute messages, each decoded by jsfive's own attribute parser
  function denseAttrs(o, buf) {
    var out = {}, ai = o.find_msg_type(0x15)[0];
    if (!ai) return out;
    var dv = new DataView(buf), m = ai.get("offset_to_message"), fl = dv.getUint8(m + 1), p = m + 2 + (fl & 1 ? 2 : 0);
    var heap = u64(dv, p), bt = u64(dv, p + 8);
    if (heap < 0 || bt < 0) return out;
    var sig = function (a) { return String.fromCharCode(dv.getUint8(a), dv.getUint8(a + 1), dv.getUint8(a + 2), dv.getUint8(a + 3)); };
    if (sig(heap) !== "FRHP" || dv.getUint8(heap + 4) !== 0) throw new Error("unexpected fractal heap");
    var idLen = dv.getUint16(heap + 5, true), filt = dv.getUint16(heap + 7, true), hflags = dv.getUint8(heap + 9);
    if (filt) throw new Error("filtered attribute heap");
    // FRHP v0 with 8-byte offsets and lengths: fixed fields up to the table width at +110
    var q = heap + 110, width = dv.getUint16(q, true), start = u64(dv, q + 2), maxDirect = u64(dv, q + 10), maxBits = dv.getUint16(q + 18, true);
    var root = u64(dv, q + 22), rows = dv.getUint16(q + 30, true), offB = Math.ceil(maxBits / 8);
    // the direct blocks in heap-offset order: [heap offset, size, file address]
    var blocks = [];
    if (root < 0) throw new Error("empty attribute heap");
    if (rows === 0) { if (sig(root) !== "FHDB") throw new Error("no direct block at the heap root"); blocks.push([0, start, root]); }
    else {
      if (sig(root) !== "FHIB") throw new Error("no indirect block at the heap root");
      var maxDirRows = Math.round(Math.log2(maxDirect) - Math.log2(start)) + 2, ho = 0, a = root + 5 + 8 + offB;
      if (rows > maxDirRows) throw new Error("attribute heap has indirect children");
      for (var rr = 0; rr < rows; rr++) {
        var bs = rr < 2 ? start : start * Math.pow(2, rr - 1);
        for (var c = 0; c < width; c++, a += 8, ho += bs) { var ad = u64(dv, a); if (ad >= 0) blocks.push([ho, bs, ad]); }
      }
    }
    if (sig(bt) !== "BTHD" || dv.getUint8(bt + 5) !== 8) throw new Error("unexpected attribute name index");
    var recSize = dv.getUint16(bt + 10, true), depth = dv.getUint16(bt + 12, true), rootNode = u64(dv, bt + 16), nrec = dv.getUint16(bt + 24, true);
    if (depth !== 0 || sig(rootNode) !== "BTLF") throw new Error("attribute name index deeper than one leaf");
    for (var i = 0; i < nrec; i++) {
      var r = rootNode + 6 + i * recSize, idType = (dv.getUint8(r) >> 4) & 3, off = 0, k;
      if (idType !== 0) throw new Error("non-managed attribute object");
      for (k = 0; k < offB; k++) off += dv.getUint8(r + 1 + k) * Math.pow(256, k);
      var blk = blocks.filter(function (B) { return off >= B[0] && off < B[0] + B[1]; })[0];
      if (!blk || sig(blk[2]) !== "FHDB") throw new Error("attribute object outside the heap's blocks");
      var nv = o.unpack_attribute(blk[2] + off - blk[0]);              // object offsets count from the block start (header included)
      out[nv[0]] = nv[1];
    }
    return out;
  }
  function attr(a, k) { var v = a[k]; if (v == null) return null; if (typeof v === "object" && v.length != null && typeof v !== "string") v = v[0]; return v; }
  function contigAddr(o, buf) {
    var msg = o.find_msg_type(8)[0], off = msg.get("offset_to_message"), b = new DataView(buf);
    var ver = b.getUint8(off), cls = b.getUint8(off + 1);
    if (ver !== 3 || cls !== 1) throw new Error("unexpected layout message v" + ver + " class " + cls);
    return u64(b, off + 2);
  }
  // returns [layout, reference flashes decoded by jsfive]
  // jsfive: the layout and CF attributes of the flash variables (reads metadata only)
  // the byte length of an attribute message (v1: name, datatype and dataspace padded to 8; v2/v3: unpadded, v3 with
  // an encoding byte), so its exact bytes can be compared in other files
  function attrMsgLen(dv, off) {
    var ver = dv.getUint8(off), nn = dv.getUint16(off + 2, true), nt = dv.getUint16(off + 4, true), ns = dv.getUint16(off + 6, true);
    var p8 = function (n) { return ver === 1 ? Math.ceil(n / 8) * 8 : n; }, h = ver === 3 ? 9 : 8;
    if (ver < 1 || ver > 3) throw new Error("attribute message v" + ver);
    var t = off + h + p8(nn), s = t + p8(nt), d = s + p8(ns), esz = dv.getUint32(t + 4, true);
    var sver = dv.getUint8(s), rank = dv.getUint8(s + 1), dims = s + (sver === 1 ? 8 : 4), n = 1;
    if (sver === 2 && dv.getUint8(s + 3) === 2) n = 0;                   // null dataspace
    for (var i = 0; i < rank; i++) n *= u64(dv, dims + 8 * i);
    return d - off + n * esz;
  }
  var PACKING = ["scale_factor", "add_offset", "_FillValue", "_Unsigned"];
  function learnLayout(buf, file) {
    var lib = S.lib, f = new lib.File(buf, file.key), L = { vars: {}, attrs: [] }, dv = new DataView(buf);
    VARS.forEach(function (name) {
      var d = f.get(name); if (!d) throw new Error(name + " missing");
      var o = d._dataobjects, orig = o.unpack_attribute;
      // where the packing attributes' messages are (compact ones in the object header, dense ones in the fractal heap)
      o.unpack_attribute = function (off) {
        var nv = orig.call(o, off);
        if (PACKING.indexOf(nv[0]) >= 0 && !L.attrs.some(function (x) { return x[0] === off; })) {
          var len = attrMsgLen(dv, off); L.attrs.push([off, new Uint8Array(buf.slice(off, off + len)), name + ":" + nv[0]]);
        }
        return nv;
      };
      var a = Object.assign({}, d.attrs, denseAttrs(o, buf)), dt = String(d.dtype);
      o.unpack_attribute = orig;
      o._get_chunk_params();
      var m = /^([<>|])([fiu])(\d)$/.exec(dt); if (!m) throw new Error(name + ": dtype " + dt);
      var fp = o.filter_pipeline || [];
      var Lv = { name: name, btree: Number(o._chunk_address), chunk: o.chunks && o.chunks[0], kind: m[2], esz: +m[3], le: m[1] !== ">",
                 filters: fp.map(function (x) { return x.get("filter_id"); }),
                 scale: attr(a, "scale_factor"), offset: attr(a, "add_offset"), fill: attr(a, "_FillValue"), unsigned: String(attr(a, "_Unsigned") || "") === "true" };
      if (!(Lv.chunk > 0)) throw new Error(name + " is not chunked");
      L.vars[name] = Lv;
      if (name === "flash_time_offset_of_first_event") {
        var u = String(attr(a, "units") || ""), mu = /^\s*(seconds|milliseconds) since (\d{4})-(\d\d)-(\d\d)[ T](\d\d):(\d\d):(\d\d(?:\.\d+)?)/.exec(u);
        if (!mu || Math.abs(Date.UTC(+mu[2], +mu[3] - 1, +mu[4], +mu[5], +mu[6]) + parseFloat(mu[7]) * 1000 - file.s) > 60) throw new Error("time units '" + u + "' do not match the file start");
        if (mu[1] === "milliseconds") Lv.mul = 1e-3;                     // early 2018 files (2 ms steps)
        // late-2018 files pack the offsets as unsigned 16-bit (add_offset -5 s, 0.38 ms steps, so -5 to +20 s) but
        // lack the _Unsigned attribute; read as signed, the second half of the file comes out at -17 to -5 s
        if (Lv.kind === "i" && !Lv.unsigned && Lv.offset < 0 && Lv.scale > 0 && Lv.offset + Lv.scale * Math.pow(2, 8 * Lv.esz) > 0) Lv.unsigned = true;
      }
    });
    var fc = f.get("flash_count"), pt = f.get("product_time");
    L.cnt = contigAddr(fc._dataobjects, buf); L.ptime = contigAddr(pt._dataobjects, buf);
    // the packing every era has: energy scaled, time offsets scaled and offset
    var E = L.vars.flash_energy, T = L.vars.flash_time_offset_of_first_event;
    if (E.scale == null || T.scale == null || T.offset == null || L.attrs.length < 4) throw new Error("packing attributes missing");
    L.attrs.sort(function (x, y) { return x[0] - y[0]; });
    L.attr0 = L.attrs[0][0];
    return { f: f, L: L };
  }
  // jsfive: the flash values (the reference the range reader is checked against)
  function refValues(lay, buf, file) {
    var f = lay.f, L = lay.L, raw = {}, fc = f.get("flash_count"), pt = f.get("product_time");
    VARS.forEach(function (name) { var d = f.get(name); raw[name] = d.shape[0] ? d.value : []; });
    var N = Number(fc.value[0] != null ? fc.value[0] : fc.value), conv = {};
    if (raw.flash_lat.length !== N) throw new Error("flash_count " + N + " but " + raw.flash_lat.length + " flashes");
    VARS.forEach(function (name) {
      var Lv = L.vars[name], src = raw[name], o = new Float64Array(N);
      for (var i = 0; i < N; i++) { var r = Number(src[i]), v = r; if (Lv.unsigned && Lv.kind === "i" && r < 0) v = r + Math.pow(2, 8 * Lv.esz); o[i] = cf(v, Lv, r); }
      conv[name] = o;
    });
    var ptv = Number(pt.value[0] != null ? pt.value[0] : pt.value);
    var ref = pack(file, N, conv.flash_lat, conv.flash_lon, conv.flash_time_offset_of_first_event, conv.flash_energy, conv.flash_quality_flag, ptv);
    return ref;
  }
  function learnFrom(buf, file) { var lay = learnLayout(buf, file); return [lay.L, refValues(lay, buf, file)]; }
  function sameFlashes(a, b) {
    if (a.n !== b.n) return false;
    for (var i = 0; i < a.n; i++) {
      if (a.lat[i] !== b.lat[i] || a.lon[i] !== b.lon[i] || a.q[i] !== b.q[i] || Math.abs(a.t[i] - b.t[i]) > 1e-6) return false;
      if (!(a.e[i] === b.e[i] || (isNaN(a.e[i]) && isNaN(b.e[i])))) return false;
    }
    return true;
  }
  // learn from `file`; its flashes are used as that file's result. First from a sparse copy (the metadata and chunk
  // indexes in the first 144 KB, the object-header continuations in the last 16 KB, the flash chunks), and if that
  // does not parse or the two readers disagree, from the whole file.
  var HEAD = 147456, TAIL = 16384;
  var SPARSE_NEEDS = ["flash_energy:scale_factor", "flash_energy:add_offset", "flash_energy:_FillValue", "flash_energy:_Unsigned", "flash_quality_flag:_FillValue",
                      "flash_quality_flag:_Unsigned", "flash_time_offset_of_first_event:scale_factor", "flash_time_offset_of_first_event:add_offset"];
  async function learnSparse(file, hi) {
    var url = bucketUrl(file.sat) + file.key, size = file.size;
    if (!(size > HEAD + TAIL)) throw new Error("small file");
    var parts = await Promise.all([get(url, [0, HEAD], hi), get(url, [size - TAIL, size], hi)]);
    var buf = new ArrayBuffer(size), u = new Uint8Array(buf);
    u.set(parts[0], 0); u.set(parts[1], size - TAIL);
    var lay = learnLayout(buf, file), L = lay.L, N = new DataView(buf).getInt32(L.cnt, true), rs = [];
    // the gap reads as zeros: every packing attribute every era has must have been found, inside the copy
    var tags = L.attrs.map(function (x) { return x[2]; });
    if (!SPARSE_NEEDS.every(function (t) { return tags.indexOf(t) >= 0; }) || !L.attrs.every(function (x) { return x[0] + x[1].length <= HEAD || x[0] >= size - TAIL; }))
      throw new Error("packing attributes outside the sparse copy");
    if (N > 0) VARS.forEach(function (v) {
      var Lv = L.vars[v], need = Math.ceil(N / Lv.chunk);
      if (Lv.btree + 24 + need * 32 > HEAD) throw new Error("chunk index beyond the head");
      btreeNode(u.subarray(Lv.btree, Lv.btree + 24 + need * 32), need).chunks.forEach(function (c) { rs.push([c.addr, c.addr + c.size]); });
    });
    var m = mergeRanges(rs), got = await Promise.all(m.map(function (r) { return get(url, r, hi); }));
    m.forEach(function (r, i) { u.set(got[i], r[0]); });
    return [buf, lay];
  }
  async function learn(file, hi) {
    var t0 = performance.now(), sat = file.sat, cur = satLayouts(sat);
    cur.n++; cur.learning = true;
    var buf, lay, ref, mine, how = "sparse";
    async function check() {
      ref = refValues(lay, buf, file);
      var whole = new Uint8Array(buf), rd = function (a, n) { if (a + n > whole.length) throw new Error("beyond the file"); return whole.subarray(a, a + n); };
      try { mine = await readFastLocal(file, lay.L, rd); } catch (e) { mine = null; console.warn("GLM: range reader failed on " + file.key + ": " + e.message); }
      return !!mine && sameFlashes(mine, ref);
    }
    try {
      await loadLib();
      var same = false;
      // the sparse copy only helps when the chunk indexes sit in the head (2026 builds); after two misses in a row
      // (older builds keep them near the end) go straight to the whole file, until a whole file shows them in the head
      if (cur.sparseMiss < 2) {
        try { var sp = await learnSparse(file, hi); buf = sp[0]; lay = sp[1]; same = await check(); }
        catch (e) { if (e.status) throw e; }
        if (!same) cur.sparseMiss++; else cur.sparseMiss = 0;
      }
      if (!same) {
        how = "whole";
        buf = (await get(bucketUrl(sat) + file.key, null, hi)).buffer;
        S.full++; S.fullBytes += buf.byteLength;
        lay = learnLayout(buf, file); same = await check();
        if (same && headOnly(lay.L, buf)) cur.sparseMiss = 0;
      }
    } catch (e) {
      cur.learning = false;
      if (!e.status && !cur.ok) { cur.failAt = Date.now(); cur.err = e.message || String(e); }   // not the file's fault: hold the others a minute
      throw e;
    }
    // file it under its variant (same signature, same verdict on the range reader), its addresses in front
    var L = lay.L, sig = signature(L), V = null;
    for (var i = 0; i < cur.variants.length; i++) if (cur.variants[i].sig === sig && cur.variants[i].fast === same) V = cur.variants[i];
    if (V) cur.variants.splice(cur.variants.indexOf(V), 1);
    else {
      V = { sig: sig, vars: L.vars, fast: same, cnt: [], ptime: [], bt: [], at: [], uses: 0, learned: 0,
            attrs: L.attrs.map(function (x) { return [x[0] - L.attr0, x[1]]; }) };
      V.attrSpan = Math.max.apply(null, V.attrs.map(function (m) { return m[0] + m[1].length; }));
    }
    cur.variants.unshift(V); if (cur.variants.length > MAX_VAR) cur.variants.length = MAX_VAR;
    addHyps(V.cnt, L.cnt, file.size); addHyps(V.ptime, L.ptime, file.size); addHyps(V.bt, L.vars.flash_lat.btree, file.size); addHyps(V.at, L.attr0, file.size);
    V.learned++; V.learnedFrom = file.key; V.how = how;
    V.ver = V.ver ? [Math.min(V.ver[0], file.s), Math.max(V.ver[1], file.s)] : [file.s, file.s];   // file times with checked packing
    if (!same) console.warn("GLM: range reader disagrees with jsfive on " + file.key + "; whole files for this layout");
    cur.ok = true; cur.learning = false; cur.failAt = 0;
    S.learned++; S.learnMs.push(Math.round(performance.now() - t0));
    return ref;
  }
  function headOnly(L, buf) {
    var N = new DataView(buf).getInt32(L.cnt, true);
    return L.cnt < HEAD && L.ptime < HEAD && VARS.every(function (v) { return L.vars[v].btree + 24 + Math.ceil(N / L.vars[v].chunk) * 32 <= HEAD; });
  }
  // the range reader against bytes already in memory (the self-check, and the tests)
  async function readFastLocal(file, L, rd) {
    var N = new DataView(rd(L.cnt, 4).slice().buffer).getInt32(0, true), ptime = new DataView(rd(L.ptime, 8).slice().buffer).getFloat64(0, true);
    if (N === 0) return pack(file, 0, [], [], [], [], [], ptime);
    var vals = {};
    for (var k = 0; k < VARS.length; k++) {
      var Lv = L.vars[VARS[k]], need = Math.ceil(N / Lv.chunk), node = btreeNode(rd(Lv.btree, 24 + need * 32), need);
      vals[VARS[k]] = await decodeVar(Lv, node.chunks, rd, N);
    }
    return pack(file, N, vals.flash_lat, vals.flash_lon, vals.flash_time_offset_of_first_event, vals.flash_energy, vals.flash_quality_flag, ptime);
  }

  // ── files ────────────────────────────────────────────────────────────────────────────────────────────────────────
  function fileObj(sat, key, size) {
    var m = /_s(\d{14})_/.exec(key), s = m ? parseS(m[1]) : NaN, id = sat.id + ":" + s, f = files.get(id);
    if (!f) { f = { id: id, sat: sat, key: key, size: size, s: s, state: "idle", wins: new Set(), bytes: 0 }; files.set(id, f); byUrl[bucketUrl(sat) + key] = f; }
    return f;
  }
  function retryLater(f) { setTimeout(function () { if (f.state === "error") f.state = "idle"; }, 60e3); }
  function loadFile(f) {
    if (f.state !== "idle") return;
    f.state = "queued";
    var task = { sat: f.sat, f: f, t: f.s, learner: false,
      alive: function () { return f.state === "queued" && Array.from(f.wins).some(function (w) { return w.alive(); }); },
      prio: function () { var p = 9; f.wins.forEach(function (w) { p = Math.min(p, w.prio()); }); return p; },
      drop: function () { if (f.state === "queued") f.state = "idle"; },
      fail: function (err) { f.state = "error"; f.err = err; retryLater(f); bump(); },
      run: function () { return run(f, task); } };
    enqueue(task);
  }
  async function run(f, task) {
    f.state = "loading"; bump();
    var t0 = performance.now(), hi = task.prio() === 0;
    try {
      var Ls = satLayouts(f.sat), r;
      if (task.learner || !Ls.ok) r = await learn(f, hi);
      else if (!Ls.variants.some(function (v) { return v.fast; })) r = await readWhole(f, hi);
      else {
        try { r = await readFast(f, Ls, hi); }
        catch (e) {
          if (e.status) throw e;                                         // HTTP trouble is not a layout change
          console.info("GLM: " + f.key + ": " + e.message + "; learning its layout");
          r = await learn(f, hi);                                        // this file's flashes then come from jsfive
        }
      }
      if (projector) { r.x = new Float32Array(r.n); r.y = new Float32Array(r.n);
        for (var i = 0; i < r.n; i++) { var q = projector(r.lon[i], r.lat[i]); r.x[i] = q[0]; r.y[i] = q[1]; } }
      f.data = r; f.state = "ok"; f.ms = performance.now() - t0; S.files++; S.fileMs.push(Math.round(f.ms));
    } catch (e) {
      f.state = "error"; f.err = e && e.message || String(e); S.err = f.err;
      if (e && e.status === 404) f.state = "missing"; else retryLater(f);
    }
    bump();
  }
  async function readWhole(f, hi) {
    await loadLib();
    var buf = (await get(bucketUrl(f.sat) + f.key, null, hi)).buffer;
    S.full++; S.fullBytes += buf.byteLength;
    return learnFrom(buf, f)[1];
  }

  // ── listings: the S3 keys for one satellite and a time span, from the bucket's own listing (XML) ───────────────
  var lists = new Map();
  function listSpan(sat, t0, t1) {
    var id = sat.id + "|" + t0 + "|" + t1, L = lists.get(id);
    if (L && (L.state === "ok" || L.state === "loading") && !(L.state === "ok" && L.provisional && Date.now() - L.at > 50e3)) return L;
    if (L && L.state === "error" && Date.now() - L.at < 60e3) return L;
    L = { state: "loading", at: Date.now(), files: [] }; lists.set(id, L);
    var days = [], d0 = Math.floor(t0 / 864e5), d1 = Math.floor(t1 / 864e5);
    for (var d = d0; d <= d1; d++) days.push(d);
    L.p = Promise.all(days.map(function (dd) {
      var day = new Date(dd * 864e5), pre = "GLM-L2-LCFA/" + day.getUTCFullYear() + "/" + p3(doy(day)) + "/";
      var a = Math.max(t0, dd * 864e5), b = Math.min(t1, dd * 864e5 + 864e5 - 1), sa = "";
      if (a > dd * 864e5) sa = pre + p2(new Date(a).getUTCHours()) + "/OR_GLM-L2-LCFA_" + sat.id + "_s" + sStamp(a);
      var u = bucketUrl(sat) + "?list-type=2&prefix=" + encodeURIComponent(pre) + (sa ? "&start-after=" + encodeURIComponent(sa) : "") + "&max-keys=" + (Math.ceil((b - a) / FILE_S) + 6);
      return get(u, null, true).then(function (bytes) {
        var x = new TextDecoder().decode(bytes), re = /<Key>([^<]+)<\/Key>[\s\S]*?<Size>(\d+)<\/Size>/g, m, out = [];
        if (!/<ListBucketResult/.test(x)) throw new Error("unexpected listing reply");
        while ((m = re.exec(x))) { var f = fileObj(sat, m[1], +m[2]); if (f.s >= a && f.s <= b) out.push(f); }
        return out;
      });
    })).then(function (parts) {
      L.files = [].concat.apply([], parts); L.state = "ok"; L.at = Date.now();
      var expect = Math.floor((t1 - t0) / FILE_S) + 1;
      L.provisional = L.files.length < expect && Date.now() - t1 < 15 * 60e3;   // the newest files may still be on their way
      bump(); return L;
    }, function (e) { L.state = "error"; L.err = e.message; L.at = Date.now(); bump(); });
    return L;
  }

  // ── windows: the flashes of the N minutes before a radar frame ──────────────────────────────────────────────────
  // files whose flashes can start in [T - N, T): a file [s, s + 20 s) reports flashes that END in it, whose first
  // event can be up to ~5 s earlier, so s runs from T - N - 20 s to T
  function winObj(T, N) {
    var k = T + "|" + N, w = wins.get(k);
    if (!w) {
      w = { key: k, T: T, N: N, t0: T - N * 60e3, wanted: 0, urgent: false, slots: {}, made: performance.now(), done: 0, b0: S.bytes };
      w.alive = function () { return Date.now() - w.wanted < 20 * 60e3; };
      w.prio = function () { return w.urgent && Date.now() - w.wanted < 120e3 ? 0 : 1; };
      wins.set(k, w);
    }
    return w;
  }
  function want(T, N, slots, urgent) {
    var w = winObj(T, N);
    w.wanted = Date.now();
    if (urgent) { wins.forEach(function (o) { o.urgent = false; }); w.urgent = true; }
    slots.forEach(function (slot) {
      var sat = satFor(slot, T), st = w.slots[slot] || (w.slots[slot] = {});
      st.sat = sat;
      if (!sat) { st.state = "none"; return; }
      var L = listSpan(sat, Math.floor((w.t0 - FILE_S) / FILE_S) * FILE_S, T);
      st.list = L;
      if (L.state === "loading") { L.p.then(function () { want(T, N, slots, false); }); return; }
      if (L.state !== "ok") return;
      L.files.forEach(function (f) { f.wins.add(w); if (f.state === "idle") loadFile(f); });
    });
    pump();
    return w;
  }
  // everything loaded so far for a window; the East/West rule applied per flash
  var colCache = null;
  function collect(T, N, slots) {
    var key = T + "|" + N + "|" + slots.join(","), w = wins.get(T + "|" + N);
    if (colCache && colCache.key === key && colCache.v === version) return colCache.r;
    var r = { n: 0, x: [], y: [], lat: [], lon: [], t: [], e: [], sat: [], bad: 0, files: 0, filesOk: 0, filesErr: 0, filesMissing: 0,
              listErr: [], none: [], loading: 0, sats: {}, err: null, bytes: 0 };
    if (!w) { colCache = { key: key, v: version, r: r }; return r; }
    // a slot whose satellite or listing is unavailable hands its side to the other one
    var healthy = {};
    slots.forEach(function (slot) { var st = w.slots[slot]; healthy[slot] = !!(st && st.sat && st.list && st.list.state !== "error"); });
    var t0 = w.t0;
    slots.forEach(function (slot) {
      var st = w.slots[slot];
      if (!st) return;
      if (!st.sat) { r.none.push(slot); return; }
      if (!st.list || st.list.state === "loading") { r.loading++; return; }
      if (st.list.state === "error") { r.listErr.push(st.sat.name + ": " + st.list.err); return; }
      var other = slot === "East" ? "West" : "East", alone = !(other in healthy) || !healthy[other];
      st.list.files.forEach(function (f) {
        r.files++;
        if (f.state === "ok") { r.filesOk++; r.bytes += f.bytes || 0; } else if (f.state === "missing") r.filesMissing++; else if (f.state === "error") { r.filesErr++; r.err = f.err; } else { r.loading++; return; }
        if (f.state !== "ok") return;
        var d = f.data;
        r.sats[f.sat.name] = 1;
        for (var i = 0; i < d.n; i++) {
          var tt = d.t[i]; if (tt < t0 || tt >= T) continue;
          var lo = d.lon[i];
          if (!alone && (slot === "East" ? lo < BOUND : lo >= BOUND)) continue;
          if (d.q[i] !== 0) { r.bad++; continue; }
          r.x.push(d.x ? d.x[i] : 0); r.y.push(d.y ? d.y[i] : 0); r.lat.push(d.lat[i]); r.lon.push(lo); r.t.push(tt); r.e.push(d.e[i]); r.sat.push(f.sat.name);
        }
      });
    });
    r.n = r.t.length;
    if (!r.loading && !w.done && r.files) { w.done = performance.now(); S.windows.push({ T: new Date(T).toISOString().slice(11, 16), N: N, slots: slots.join("+"), ms: Math.round(w.done - w.made), kb: Math.round(r.bytes / 1024), kbPerFile: Math.round(r.bytes / Math.max(1, r.filesOk) / 102.4) / 10, files: r.files, flashes: r.n }); }
    colCache = { key: key, v: version, r: r };
    return r;
  }
  function slotsFor(lonW, lonE) {
    var s = [];
    if (lonE >= BOUND - 1.5) s.push("East");
    if (lonW < BOUND + 1.5) s.push("West");
    return s;
  }

  root.GLM = {
    want: want, collect: collect, slotsFor: slotsFor, satFor: satFor, BOUND: BOUND, SLOTS: SLOTS, S: S,
    get version() { return version; }, set onUpdate(f) { onUpdate = f; }, set projector(f) { projector = f; },
    setCap: function (b) { S.cap = b; S.capped = S.bytes >= b; pump(); bump(); },
    clearQueue: function () { queue.forEach(function (q) { q.drop(); }); queue = []; },
    layouts: layouts, files: files,
    _t: { sStamp: sStamp, parseS: parseS, learnFrom: learnFrom, readFastLocal: readFastLocal, fileObj: fileObj, loadLib: loadLib, btreeNode: btreeNode, mergeRanges: mergeRanges, run: run, denseAttrs: denseAttrs }
  };
})(typeof window !== "undefined" ? window : globalThis);
