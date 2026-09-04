"""Columbia River basin precipitation — the public page's data pipeline.

Self-contained (numpy, shapely, eccodes, ecmwf-opendata, requests) so it runs
in a GitHub Actions job in a few minutes. Precipitation only, Pacific-Northwest
box only, and every model's rain averaged over the NWRFC water-supply
divisions:

    model   source                          steps        cost / run
    gfs     AWS noaa-gfs-bdp-pds, APCP byte-ranged   3 h to 240   81 x 0.5 MB
    gefs    AWS noaa-gefs-pds, APCP byte-ranged      6 h to 240   41 x 31 members x 0.3 MB
    ecmwf   ECMWF open data (Google mirror), tp      3 h/6 h to 240
    aifs    ECMWF open data (Google mirror), tp      6 h to 360
    ecmwf_ens  same, all 50 perturbed members        6 h to 240   41 x 50 x 0.6 MB
    gdps    MSC Datamart Precip-Accum_Sfc            6 h to 240   41 x 3 MB
    geps    MSC Datamart APCP allmbrs (21 members)   6 h to 384, and to 768+ on the
            Monday/Thursday 00Z extended runs (user: "even subseasonal when available")
    obs     NCEP Stage IV 24 h (NOMADS, then the IEM archive)

Day windows are 12Z-to-12Z labelled by the ending date (= the Stage IV 24 h
product); percent of normal = period mm / the sum of the NWRFC 1991-2020
daily normals over the same dates; every run-over-run number is the
same-calendar-day change against the model's previous issue.

The archive is tiny (one gzipped JSON of daily division means per model
cycle, ~2 kB) and lives in the repo under columbia/data/archive; a partial
cycle is never archived. WeatherNext is never fetched here.

    python scripts/columbia/pipeline.py                # newest complete cycle per model
    python scripts/columbia/pipeline.py --models gfs,ecmwf --obs-days 30
"""
from __future__ import annotations
import argparse, calendar, concurrent.futures as cf, datetime as dt, glob, gzip, json, os, random, tempfile, time
import numpy as np, requests, warnings

os.environ.setdefault("TQDM_DISABLE", "1")
warnings.filterwarnings("ignore", category=RuntimeWarning)   # all-NaN member days are expected
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SITE = os.path.join(ROOT, "columbia")
DATA = os.path.join(SITE, "data")
ARCH = os.path.join(DATA, "archive")
OBS = os.path.join(DATA, "obs")
GEO = os.path.join(SITE, "pnw_divisions.geojson")
DIVS = os.path.join(DATA, "divisions.json")
CACHE = os.path.join(ROOT, ".cache", "columbia")

W, E, S, N = -125.5, -109.0, 40.5, 53.5          # the PNW box the models are cut to
UA = {"User-Agent": "scorvec-columbia/1.0"}
PERIODS = ["d1-5", "d6-10", "d11-15", "d1-10", "d1-15", "d16-32"]
PERIOD_DAYS = {"d1-5": (1, 5), "d6-10": (6, 10), "d11-15": (11, 15), "d1-10": (1, 10), "d1-15": (1, 15),
               "d16-32": (16, 32)}          # weeks 3-5: the GEPS Monday/Thursday extension only
MODEL_LABEL = {"gfs": "GFS", "gefs": "GEFS", "ecmwf": "ECMWF IFS", "aifs": "AIFS",
               "ecmwf_ens": "ECMWF ENS", "gdps": "GDPS", "geps": "GEPS", "geps_ext": "GEPS ext.",
               "blend": "Blend"}
# Blend prior weights (sum to 1 over whatever contributed, renormalised per
# date). Ensembles carry the weight; the deterministic runs add placement.
# Once Stage IV verification has >= 20 pairs per model the prior is averaged
# 50/50 with an inverse-MAE weight (see blend_weights).
BLEND_PRIOR = {"ecmwf_ens": 0.28, "gefs": 0.20, "geps": 0.14, "aifs": 0.12,
               "ecmwf": 0.10, "gdps": 0.08, "gfs": 0.08}
BLEND_MIN_ON_TARGET = 0.5        # share of weight that must come from cycles == the target
COMPOSITES = {
    "Columbia abv The Dalles": {"regions": ["Columbia River Main Stem", "Middle Columbia Basin",
                                            "Snake River", "Upper Columbia Basin"], "exclude": ["LCOLUMBIA"]},
    "Columbia abv Grand Coulee": {"regions": ["Upper Columbia Basin"]},
    "Snake abv Ice Harbor": {"regions": ["Snake River"]},
    "Middle Columbia": {"regions": ["Middle Columbia Basin"]},
    "Western Oregon": {"regions": ["Western Oregon"]},
    "Western Washington": {"regions": ["Western Washington"]},
}
ST4 = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/pcpanl/prod/pcpanl.{d}/st4_conus.{d}12.24h.grb2"
ST4_IEM = "https://mesonet.agron.iastate.edu/archive/data/{y}/{m}/{dd}/stage4/ST4.{d}12.24h.grib"
AWS_GFS = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
AWS_GEFS = "https://noaa-gefs-pds.s3.amazonaws.com"
DDMART = "https://dd.weather.gc.ca"


# ---- http ----------------------------------------------------------------------- #
_SESS = requests.Session()


def get(url, tries=3, headers=None, timeout=120):
    for i in range(tries):
        try:
            h = dict(UA); h.update(headers or {})
            r = _SESS.get(url, headers=h, timeout=timeout)
            if r.status_code in (200, 206) and len(r.content) > 200:
                return r.content
        except Exception:
            pass
        if i < tries - 1:
            time.sleep(0.5 * (2 ** i) + random.random() * 0.4)
    return None


def grib(b):
    return b if b and b[:4] == b"GRIB" else None


def ranged(base_url, want: tuple[str, ...], tries=3):
    """One record byte-ranged out of a whole GRIB file on a cloud mirror."""
    txt = get(base_url + ".idx", tries=tries)
    if not txt:
        return None
    offs = []
    for l in txt.decode("utf-8", "ignore").splitlines():
        p = l.split(":")
        if len(p) > 2:
            offs.append((int(p[1]), l))
    for i, (o, l) in enumerate(offs):
        if all(w in l for w in want):
            end = offs[i + 1][0] - 1 if i + 1 < len(offs) else ""
            return grib(get(base_url, tries=tries, headers={"Range": f"bytes={o}-{end}"}))
    return None


# ---- grib decoding --------------------------------------------------------------- #
def messages(buf: bytes) -> list[dict]:
    import eccodes
    out, grids = [], {}
    if not buf or len(buf) < 200:
        return out
    fd, p = tempfile.mkstemp(suffix=".grib2"); os.close(fd)
    try:
        with open(p, "wb") as f:
            f.write(buf)
        with open(p, "rb") as f:
            while True:
                h = eccodes.codes_grib_new_from_file(f)
                if h is None:
                    break
                try:
                    g = {}
                    for k in ("shortName", "typeOfLevel", "stepRange", "gridType", "Ni", "Nj",
                              "perturbationNumber", "units", "endStep", "startStep"):
                        try: g[k] = eccodes.codes_get(h, k)
                        except Exception: g[k] = None
                    ni, nj = int(g["Ni"]), int(g["Nj"])
                    v = np.asarray(eccodes.codes_get_values(h), np.float32)
                    try:
                        miss = eccodes.codes_get(h, "missingValue"); v[v == np.float32(miss)] = np.nan
                    except Exception:
                        pass
                    if v.size != ni * nj:
                        continue
                    gk = (g["gridType"], ni, nj)
                    if gk not in grids:
                        lat = np.asarray(eccodes.codes_get_array(h, "latitudes"), float).reshape(nj, ni)
                        lon = np.asarray(eccodes.codes_get_array(h, "longitudes"), float).reshape(nj, ni)
                        grids[gk] = (lat[:, 0].copy(), lon[0, :].copy()) if g["gridType"] == "regular_ll" else (lat, lon)
                    g["lat"], g["lon"] = grids[gk]
                    g["values"] = v.reshape(nj, ni)
                    out.append(g)
                finally:
                    eccodes.codes_release(h)
    finally:
        try: os.remove(p)
        except OSError: pass
    return out


def span(g):
    try:
        a, b = str(g["stepRange"]).split("-"); return int(b) - int(a)
    except Exception:
        try: return int(g["endStep"]) - int(g["startStep"] or 0)
        except Exception: return None


def box(v, lat, lon):
    """(member, ny, nx) on ascending 1-D axes in -180..180, cut to the PNW box."""
    lon = np.where(lon > 180, lon - 360, lon)
    jj = np.flatnonzero((lon >= W) & (lon <= E)); ii = np.flatnonzero((lat >= S) & (lat <= N))
    if not len(jj) or not len(ii):
        return None
    v = v[..., ii[:, None], jj[None, :]]; lat = lat[ii]; lon = lon[jj]
    if lat[0] > lat[-1]:
        lat = lat[::-1]; v = v[..., ::-1, :]
    o = np.argsort(lon)
    return v[..., o], lat, lon[o]


# ---- models ------------------------------------------------------------------------- #
class Model:
    name = ""; cycles = (0, 12); lag_h = 4.0; steps: list[int] = []; members = ("",)   # 00Z/12Z only
    kind = "bucket"            # 'bucket' (interval, with span) | 'acc' (since init)
    def url(self, init, step, member): ...
    def fetch(self, init, step, member=""):
        return grib(get(self.url(init, step, member)))
    def read(self, buf, step):
        """-> (values (member, ny, nx) in mm, lat, lon, span_h or None)"""
        ms = [g for g in messages(buf) if g["shortName"] in ("tp", "unknown")]
        if not ms:
            return None
        if self.kind == "bucket":
            ms.sort(key=lambda g: span(g) or 99)
            g = ms[0]; sp = span(g)
        else:
            g = ms[0]; sp = None
        v = g["values"][None]
        if g.get("units") == "m":
            v = v * 1000.0
        r = box(v, g["lat"], g["lon"])
        return None if r is None else (r[0], r[1], r[2], sp)
    def available(self, init):
        return bool(self.fetch(init, self.steps[1], self.members[0]))


class GFS(Model):
    name = "gfs"; lag_h = 4.0; steps = list(range(0, 241, 3))
    def fetch(self, init, step, member=""):
        d, h = init.strftime("%Y%m%d"), init.strftime("%H")
        if step == 0:
            return b"ZERO"
        return ranged(f"{AWS_GFS}/gfs.{d}/{h}/atmos/gfs.t{h}z.pgrb2.0p25.f{step:03d}", ("APCP:surface",))
    def available(self, init):
        d, h = init.strftime("%Y%m%d"), init.strftime("%H")
        return bool(get(f"{AWS_GFS}/gfs.{d}/{h}/atmos/gfs.t{h}z.pgrb2.0p25.f240.idx", tries=1))


class GEFS(Model):
    """AWS mirror (user: 'for GEFS use a mirror, AWS has one'): the APCP record
    byte-ranged out of each member file, no NOMADS request at all. All 31
    members at 6 h to 240 is ~350 MB a run; COLUMBIA_GEFS_MEMBERS lowers it."""
    name = "gefs"; lag_h = 5.5; steps = list(range(0, 241, 6))
    @property
    def members(self):
        n = int(os.environ.get("COLUMBIA_GEFS_MEMBERS", 31))      # all 31 (user, 3 Sep 2026)
        return tuple(["c00"] + [f"p{i:02d}" for i in range(1, n)])
    def fetch(self, init, step, member="c00"):
        d, h = init.strftime("%Y%m%d"), init.strftime("%H")
        if step == 0:
            return b"ZERO"
        return ranged(f"{AWS_GEFS}/gefs.{d}/{h}/atmos/pgrb2sp25/ge{member}.t{h}z.pgrb2s.0p25.f{step:03d}",
                      ("APCP:surface",))
    def available(self, init):
        d, h = init.strftime("%Y%m%d"), init.strftime("%H")
        return bool(get(f"{AWS_GEFS}/gefs.{d}/{h}/atmos/pgrb2sp25/gep01.t{h}z.pgrb2s.0p25.f240.idx", tries=1))


class _ECMWF(Model):
    kind = "acc"; stream = "oper"; typ = "fc"; model = "ifs"; lag_h = 7.0
    _clients: dict = {}
    def client(self, source):
        key = (source, self.model)
        if key not in _ECMWF._clients:
            from ecmwf.opendata import Client
            _ECMWF._clients[key] = Client(source=source, model=self.model)
        return _ECMWF._clients[key]
    def one(self, source, **kw):
        fd, p = tempfile.mkstemp(suffix=".grib2"); os.close(fd)
        try:
            self.client(source).retrieve(target=p, **kw)
            with open(p, "rb") as f:
                return grib(f.read())
        except Exception:
            return None
        finally:
            try: os.remove(p)
            except OSError: pass
    def fetch(self, init, step, member=""):
        kw = dict(date=init.strftime("%Y%m%d"), time=init.hour, step=step, stream=self.stream,
                  type=self.typ, param="tp")
        if member:
            kw["number"] = int(member)
        # the Google mirror is 6x faster and does not throttle, but refuses
        # multi-range requests -- one (param, member) per request is fine
        return self.one("google", **kw) or self.one("ecmwf", **kw)
    def available(self, init):
        return bool(self.fetch(init, self.steps[1], self.members[0]))


class IFS(_ECMWF):
    name = "ecmwf"; cycles = (0, 12); steps = list(range(0, 145, 3)) + list(range(150, 241, 6))


class AIFS(_ECMWF):
    name = "aifs"; model = "aifs-single"; steps = list(range(0, 361, 6)); lag_h = 5.0; cycles = (0, 12)


class ENS(_ECMWF):
    name = "ecmwf_ens"; stream = "enfo"; typ = "pf"; cycles = (0, 12); lag_h = 9.0
    steps = list(range(0, 241, 6))
    @property
    def members(self):
        return tuple(str(i) for i in range(1, int(os.environ.get("COLUMBIA_ENS_MEMBERS", 50)) + 1))   # all 50


class GDPS(Model):
    name = "gdps"; kind = "acc"; cycles = (0, 12); lag_h = 6.0; steps = list(range(0, 241, 6))
    def url(self, init, step, member=""):
        d, h = init.strftime("%Y%m%d"), init.strftime("%H")
        return (f"{DDMART}/{d}/WXO-DD/model_gdps/15km/{h}/{step:03d}/"
                f"{d}T{h}Z_MSC_GDPS_Precip-Accum_Sfc_LatLon0.15_PT{step:03d}H.grib2")
    def available(self, init):
        try:
            return requests.head(self.url(init, self.steps[1], ""), headers=UA, timeout=25).status_code == 200
        except Exception:
            return False


class GEPS(Model):
    """ECCC GEPS: one Datamart file per step holds all 21 members (cf + pf),
    accumulated since init. The extended runs (Mon/Thu 00Z) go past 384 h;
    `steps` is decided per cycle by probing the last extended step, so the
    subseasonal weeks ride along whenever they exist and never otherwise."""
    name = "geps"; kind = "acc"; cycles = (0, 12); lag_h = 7.5; members = ("all",)
    steps = list(range(0, 385, 6))
    def url(self, init, step, member=""):
        d, h = init.strftime("%Y%m%d"), init.strftime("%H")
        return (f"{DDMART}/{d}/WXO-DD/ensemble/geps/grib2/raw/{h}/{step:03d}/"
                f"CMC_geps-raw_APCP_SFC_0_latlon0p5x0p5_{d}{h}_P{step:03d}_allmbrs.grib2")
    def _head(self, url):
        try:
            return requests.head(url, headers=UA, timeout=25).status_code == 200
        except Exception:
            return False
    def available(self, init):
        return self._head(self.url(init, self.steps[-1], ""))
    def read(self, buf, step):
        ms = [g for g in messages(buf) if g["shortName"] in ("tp", "unknown")]
        if not ms:
            return None
        ms.sort(key=lambda g: (0 if g.get("perturbationNumber") in (None, 0) else 1, g.get("perturbationNumber") or 0))
        v = np.stack([g["values"] for g in ms])
        r = box(v, ms[0]["lat"], ms[0]["lon"])
        return None if r is None else (r[0], r[1], r[2], None)


class GEPSX(GEPS):
    """The GEPS extended run (Monday and Thursday 00Z, out to 32 days) as its
    own entry, so the subseasonal weeks stay on the page until the next
    extension instead of being displaced by the newer 12Z 16-day run.
    Twelve-hourly past 384 h: a 24 h window only needs the 12Z steps."""
    name = "geps_ext"; cycles = (0,); lag_h = 9.0
    steps = list(range(0, 385, 6)) + list(range(396, 769, 12))
    def available(self, init):
        return init.weekday() in (0, 3) and self._head(self.url(init, self.steps[-1], ""))


MODELS = {c.name: c for c in (GFS, GEFS, IFS, AIFS, ENS, GDPS, GEPS, GEPSX)}


def latest_cycle(m: Model, now=None):
    """Newest cycle whose far end is on the server. Probe, never trust the lag."""
    now = now or dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    cand = []
    day = now.date()
    for back in range(0, 5):
        for h in sorted(m.cycles, reverse=True):
            c = dt.datetime(day.year, day.month, day.day, h)
            if c <= now - dt.timedelta(hours=m.lag_h * 0.5):
                cand.append(c)
        day -= dt.timedelta(days=1)
    for c in sorted(set(cand), reverse=True)[:10]:
        if m.available(c):
            return c
    return None


# ---- divisions and weights ----------------------------------------------------------- #
_DIV = None


def divisions():
    global _DIV
    if _DIV is None:
        _DIV = json.load(open(DIVS))
    return _DIV


def geoms():
    from shapely.geometry import shape
    g = json.load(open(GEO))
    return {f["properties"]["code"]: shape(f["geometry"]) for f in g["features"]}


_W: dict = {}


def weights_1d(lat, lon):
    """(codes, W (ndiv, ny*nx)): each division rasterised on a 0.02 deg mesh,
    mesh points counted into the model cell they fall in."""
    key = (len(lat), len(lon), round(float(lat[0]), 3), round(float(lon[0]), 3))
    if key in _W:
        return _W[key]
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, "w_" + "_".join(str(k) for k in key) + ".npz")
    if os.path.exists(p):
        z = np.load(p, allow_pickle=True); _W[key] = (list(z["codes"]), z["W"]); return _W[key]
    from shapely import contains_xy
    gs = geoms(); cs = []; Wm = np.zeros((len(gs), len(lat) * len(lon)))
    for i, (c, geom) in enumerate(gs.items()):
        cs.append(c)
        x0, y0, x1, y1 = geom.bounds
        X, Y = np.meshgrid(np.arange(x0, x1 + 0.02, 0.02), np.arange(y0, y1 + 0.02, 0.02))
        inside = contains_xy(geom, X.ravel(), Y.ravel())
        if not inside.any():
            continue
        px, py = X.ravel()[inside], Y.ravel()[inside]
        ila = np.clip(np.round((py - lat[0]) / (lat[1] - lat[0])).astype(int), 0, len(lat) - 1)
        ilo = np.clip(np.round((px - lon[0]) / (lon[1] - lon[0])).astype(int), 0, len(lon) - 1)
        w = np.zeros((len(lat), len(lon))); np.add.at(w, (ila, ilo), 1.0)
        Wm[i] = (w / w.sum()).ravel()
    np.savez_compressed(p, codes=np.array(cs), W=Wm)
    _W[key] = (cs, Wm)
    return _W[key]


def weights_2d(lat2, lon2):
    key = ("st4", lat2.shape)
    if key in _W:
        return _W[key]
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, f"w_st4_{lat2.shape[0]}x{lat2.shape[1]}.npz")
    if os.path.exists(p):
        z = np.load(p, allow_pickle=True); _W[key] = (list(z["codes"]), z["W"]); return _W[key]
    from shapely import contains_xy
    lon = np.where(lon2 > 180, lon2 - 360, lon2).ravel(); lat = lat2.ravel()
    idx = np.flatnonzero((lon > -125) & (lon < -109) & (lat > 40.5) & (lat < 53.5))
    gs = geoms(); Wm = np.zeros((len(gs), lat.size)); cs = []
    for i, (c, geom) in enumerate(gs.items()):
        cs.append(c)
        inside = contains_xy(geom, lon[idx], lat[idx])
        if inside.any():
            Wm[i, idx[inside]] = 1.0 / inside.sum()
    np.savez_compressed(p, codes=np.array(cs), W=Wm)
    _W[key] = (cs, Wm)
    return _W[key]


def area():
    return {d["code"]: d["area"] for d in divisions()}


def members_of(name):
    c = COMPOSITES[name]; ex = set(c.get("exclude", []))
    return [d["code"] for d in divisions() if d["region"] in c["regions"] and d["code"] not in ex and d["normals"]]


def normal_mm(key, date):
    if key in COMPOSITES:
        cs = members_of(key); a = area()
        w = np.array([a[c] for c in cs]); w = w / w.sum()
        norms = (w @ np.array([next(d["normals"] for d in divisions() if d["code"] == c) for c in cs], float)).tolist()
    else:
        norms = next((d["normals"] for d in divisions() if d["code"] == key), None)
    if not norms:
        return None
    y, m, _ = (int(x) for x in date.split("-"))
    return norms[m - 1] / calendar.monthrange(y, m)[1]


# ---- a model cycle ------------------------------------------------------------------- #
def windows(init, steps):
    s12 = [s for s in sorted(steps) if (init.hour + s) % 24 == 12]
    return [((init + dt.timedelta(hours=b)).strftime("%Y-%m-%d"), a, b) for a, b in zip(s12, s12[1:]) if b - a == 24]


def apath(model, tag):
    return os.path.join(ARCH, f"{model}_{tag}.json.gz")


def load(model, tag):
    p = apath(model, tag)
    if not os.path.exists(p):
        return None
    with gzip.open(p, "rt") as f:
        return json.load(f)


def cycles(model):
    return sorted(os.path.basename(p)[len(model) + 1:-8] for p in glob.glob(os.path.join(ARCH, f"{model}_??????????.json.gz")))


def run_cycle(model: str, workers=6, min_complete=0.95, cyc=None):
    m = MODELS[model]()
    cyc = cyc or latest_cycle(m)
    if cyc is None:
        print(f"  {model}: no cycle available"); return None
    tag = cyc.strftime("%Y%m%d%H")
    if load(model, tag):
        print(f"  {model} {tag}: already archived"); return None
    steps = m.steps_for(cyc) if hasattr(m, "steps_for") else m.steps
    members = list(m.members)
    one_shot = members == ["all"]                 # every member in one buffer (GEPS)
    jobs = [(s, mem) for s in steps for mem in members]
    t0 = time.time(); got = {}; nbytes = 0; n_bad = 0
    codes = None; vals = {}; spans = {}
    with cf.ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(m.fetch, cyc, s, mem): (s, mem) for s, mem in jobs}
        for f in cf.as_completed(futs):
            s, mem = futs[f]
            try: b = f.result()
            except Exception: b = None
            if not b:
                continue
            if b == b"ZERO":
                vals[(s, mem)] = "zero"; got[(s, mem)] = 1; continue
            nbytes += len(b)
            r = m.read(b, s)
            if r is None:
                n_bad += 1; continue
            v, lat, lon, sp = r
            cs, Wm = weights_1d(lat, lon); codes = cs
            red = np.nan_to_num(v.reshape(v.shape[0], -1)) @ Wm.T        # (member, div)
            if one_shot:
                for k in range(red.shape[0]):
                    vals[(s, k)] = red[k]; spans[(s, k)] = sp
                n_in = red.shape[0]
            else:
                vals[(s, mem)] = red[0]; spans[(s, mem)] = sp
            got[(s, mem)] = 1
    frac = len(got) / len(jobs)
    print(f"  {model} {tag}: {len(got)}/{len(jobs)} steps ({frac:.0%}), {nbytes/1e6:.0f} MB, {time.time()-t0:.0f} s"
          + (f", {n_bad} unreadable" if n_bad else ""))
    if frac < min_complete or codes is None:
        print(f"  {model} {tag}: incomplete, not archived"); return None
    if one_shot:
        members = list(range(n_in))
        for s in steps:
            if vals.get((s, "all")) == "zero":
                for k in members:
                    vals[(s, k)] = "zero"
    nd = len(codes); ix = {s: i for i, s in enumerate(steps)}
    acc = np.full((len(members), len(steps), nd), np.nan)
    for k, mem in enumerate(members):
        if m.kind == "acc":
            for s in steps:
                v = vals.get((s, mem))
                if v is None: continue
                acc[k, ix[s]] = 0.0 if isinstance(v, str) else v
        else:
            acc[k, 0] = 0.0
            for si, s in enumerate(steps[1:], 1):
                v = vals.get((s, mem))
                if v is None or isinstance(v, str): continue
                sp = spans.get((s, mem)) or (s - steps[si - 1]); base = s - sp
                if base in ix and np.isfinite(acc[k, ix[base]]).all():
                    acc[k, si] = acc[k, ix[base]] + v
    wins = windows(cyc, steps)
    if not wins:
        return None
    daily = np.clip(np.stack([acc[:, ix[b]] - acc[:, ix[a]] for _, a, b in wins], 1), 0, None)   # (mem, day, div)
    rec = {"model": model, "cycle": tag, "init": cyc.strftime("%Y-%m-%dT%HZ"), "run_day0": wins[0][0],
           "dates": [w[0] for w in wins], "n_members": len(members), "div": {},
           "fetched": len(got), "jobs": len(jobs), "mb": round(nbytes / 1e6, 1), "secs": round(time.time() - t0)}
    for i, c in enumerate(codes):
        a = daily[:, :, i]
        if np.isfinite(a).any():
            rec["div"][c] = {"members": np.where(np.isfinite(a), a.round(2), None).tolist()}
    os.makedirs(ARCH, exist_ok=True)
    with gzip.open(apath(model, tag), "wt") as f:
        json.dump(rec, f, separators=(",", ":"))
    print(f"  {model} {tag}: archived")
    return rec


# ---- Stage IV ------------------------------------------------------------------------ #
def fetch_obs(date):
    os.makedirs(OBS, exist_ok=True)
    p = os.path.join(OBS, f"{date}.json")
    if os.path.exists(p):
        return json.load(open(p))
    d = date.replace("-", "")
    buf = grib(get(ST4.format(d=d), tries=1))
    if not buf:
        y, mo, dd = date.split("-")
        buf = grib(get(ST4_IEM.format(y=y, m=mo, dd=dd, d=d), tries=2))
    if not buf:
        return None
    g = next((g for g in messages(buf) if g["shortName"] == "tp"), None)
    if g is None:
        return None
    cs, Wm = weights_2d(g["lat"], g["lon"])
    v = np.nan_to_num(g["values"].ravel()); cov = (np.isfinite(g["values"].ravel())[None, :] * Wm).sum(1)
    out = Wm @ v
    rec = {"date": date, "source": "stage4_24h_12z", "div": {c: round(float(out[i]), 2) for i, c in enumerate(cs) if cov[i] > 0.9}}
    json.dump(rec, open(p, "w"))
    return rec


def backfill_obs(days):
    today = dt.datetime.now(dt.timezone.utc).date(); got = []
    for k in range(days, -1, -1):
        d = (today - dt.timedelta(days=k)).strftime("%Y-%m-%d")
        if not os.path.exists(os.path.join(OBS, f"{d}.json")) and fetch_obs(d):
            got.append(d)
    return got


# ---- build the page's JSON ----------------------------------------------------------------- #
def composite_series(div_daily):
    a = area(); out = {}
    for name in COMPOSITES:
        cs = [c for c in members_of(name) if c in div_daily]
        if not cs:
            continue
        w = np.array([a[c] for c in cs]); w = w / w.sum()
        arr = np.stack([np.array(div_daily[c]["members"], float) for c in cs])
        e = {"members": np.tensordot(w, arr, axes=(0, 0)).round(2).tolist()}
        if div_daily[cs[0]].get("weights"):
            e["weights"] = div_daily[cs[0]]["weights"]
        out[name] = e
    return out


def full_series(rec):
    s = dict(rec["div"]); s.update(composite_series(rec["div"])); return s


def wquant(vals, w, q):
    """Weighted quantile of the finite entries (weights renormalised)."""
    m = np.isfinite(vals)
    if not m.any():
        return np.nan
    v, ww = vals[m], w[m]
    o = np.argsort(v); v, ww = v[o], ww[o]
    c = np.cumsum(ww) - 0.5 * ww
    return float(np.interp(q * ww.sum(), c, v))


def stats_of(members, weights=None):
    a = np.array(members, float)
    f = lambda v: [None if not np.isfinite(x) else round(float(x), 2) for x in v]
    if weights is None:
        with np.errstate(all="ignore"):
            return {"mean": f(np.nanmean(a, 0)), "p10": f(np.nanpercentile(a, 10, axis=0)),
                    "p90": f(np.nanpercentile(a, 90, axis=0))}
    w = np.array(weights, float)
    mean, p10, p90 = [], [], []
    for j in range(a.shape[1]):
        col = a[:, j]; m = np.isfinite(col)
        mean.append(float((col[m] * w[m]).sum() / w[m].sum()) if m.any() else np.nan)
        p10.append(wquant(col, w, 0.10)); p90.append(wquant(col, w, 0.90))
    return {"mean": f(mean), "p10": f(p10), "p90": f(p90)}


def wmean(members, weights=None):
    a = np.array(members, float)
    if weights is None:
        with np.errstate(all="ignore"):
            return np.nanmean(a, 0)
    w = np.array(weights, float); out = np.full(a.shape[1], np.nan)
    for j in range(a.shape[1]):
        col = a[:, j]; m = np.isfinite(col)
        if m.any():
            out[j] = (col[m] * w[m]).sum() / w[m].sum()
    return out


def periods_of(series, dates):
    out = {}
    for key, v in series.items():
        mean = wmean(v["members"], v.get("weights")); pr = {}
        for p, (a, b) in PERIOD_DAYS.items():
            vals = [(mean[i], normal_mm(key, dates[i])) for i in range(a - 1, min(b, len(dates))) if np.isfinite(mean[i])]
            if not vals:
                continue
            mm = float(sum(x for x, _ in vals)); nn = [n for _, n in vals if n is not None]
            e = {"mm": round(mm, 1), "n": len(vals), "want": b - a + 1}
            if nn and len(nn) == len(vals) and sum(nn) > 0:
                e["normal"] = round(float(sum(nn)), 1); e["pct"] = round(100.0 * mm / sum(nn), 0)
            pr[p] = e
        out[key] = pr
    return out


def delta(a_s, a_d, b_s, b_d):
    out = {}; ib = {d: i for i, d in enumerate(b_d)}
    for key, va in a_s.items():
        vb = b_s.get(key)
        if not vb:
            continue
        ma = wmean(va["members"], va.get("weights")); mb = wmean(vb["members"], vb.get("weights"))
        dd = [None if ib.get(d) is None or not np.isfinite(ma[i]) or not np.isfinite(mb[ib[d]]) else round(float(ma[i] - mb[ib[d]]), 2)
              for i, d in enumerate(a_d)]
        pr = {}
        for p, (x, y) in PERIOD_DAYS.items():
            vals = [(dd[i], normal_mm(key, a_d[i])) for i in range(x - 1, min(y, len(dd))) if dd[i] is not None]
            if not vals:
                continue
            mm = float(sum(v for v, _ in vals)); nn = [n for _, n in vals if n is not None]
            e = {"mm": round(mm, 1), "n": len(vals)}
            if nn and len(nn) == len(vals) and sum(nn) > 0:
                e["pct"] = round(100.0 * mm / sum(nn), 0)
            pr[p] = e
        out[key] = {"periods": pr, "daily": dd}
    return out


def verification(recs_by_model, obs):
    out = {}; a = area()
    for m, recs in recs_by_model.items():
        acc = {}
        for r in recs:
            s = full_series(r)
            for i, d in enumerate(r["dates"]):
                o = obs.get(d)
                if not o:
                    continue
                for k in COMPOSITES:
                    if k not in s:
                        continue
                    f = float(wmean(s[k]["members"], s[k].get("weights"))[i])
                    cs = [c for c in members_of(k) if c in o]
                    if not cs or not np.isfinite(f):
                        continue
                    w = np.array([a[c] for c in cs]); w /= w.sum()
                    acc.setdefault(i + 1, []).append((f, float(w @ np.array([o[c] for c in cs]))))
        out[m] = {}
        for lead, pairs in sorted(acc.items()):
            f = np.array([p[0] for p in pairs]); o = np.array([p[1] for p in pairs])
            out[m][lead] = {"n": len(pairs), "ratio": round(float(f.sum() / o.sum()), 2) if o.sum() > 0 else None,
                            "mae": round(float(np.mean(np.abs(f - o))), 2)}
    return out


def blend_weights(verif: dict | None) -> dict:
    """Prior weights, pulled halfway toward inverse-MAE once a model has
    >= 20 verified composite-days over leads 1-7."""
    w = dict(BLEND_PRIOR)
    if verif:
        inv = {}
        for m in w:
            v = verif.get(m) or {}
            pairs = [(x["mae"], x["n"]) for l, x in v.items() if int(l) <= 7 and x.get("mae")]
            n = sum(p[1] for p in pairs)
            if n >= 20:
                inv[m] = 1.0 / max(1e-6, sum(a * b for a, b in pairs) / n)
        if len(inv) >= 3:
            tot = sum(inv.values())
            for m in inv:
                w[m] = 0.5 * w[m] + 0.5 * inv[m] / tot * sum(w[k] for k in inv)
    return w


def make_blends(verif=None, max_back_days=12):
    """A blend archive per target cycle: each core model's newest cycle at or
    before the target within 12 h, member-pooled with weight w_m / n_m, and
    only if >= BLEND_MIN_ON_TARGET of the weight comes from cycles that ARE
    the target -- a blend of stale runs relabelled as a new issue would say
    "the models did not move" when they had not run."""
    W = blend_weights(verif)
    by = {m: cycles(m) for m in W}
    targets = sorted({c for m in W for c in by[m]})
    cutoff = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=max_back_days)).strftime("%Y%m%d%H")
    made = []
    for T in targets:
        if T < cutoff or os.path.exists(apath("blend", T)):
            continue
        tT = dt.datetime.strptime(T, "%Y%m%d%H")
        pick = {}
        for m, cs in by.items():
            ok = [c for c in cs if c <= T and (tT - dt.datetime.strptime(c, "%Y%m%d%H")).total_seconds() <= 12 * 3600]
            if ok:
                pick[m] = ok[-1]
        if not pick:
            continue
        wt = sum(W[m] for m in pick); on = sum(W[m] for m, c in pick.items() if c == T)
        if on / wt < BLEND_MIN_ON_TARGET:
            continue
        recs = {m: load(m, c) for m, c in pick.items()}
        recs = {m: r for m, r in recs.items() if r}
        if not recs:
            continue
        dates = sorted({d for r in recs.values() for d in r["dates"]})
        cs_all = sorted({c for r in recs.values() for c in r["div"]})
        div = {}
        mem_w = []
        for m, r in recs.items():
            mem_w += [W[m] / r["n_members"]] * r["n_members"]
        for c in cs_all:
            rows = []
            for m, r in recs.items():
                dm = r["div"].get(c)
                ix = {d: i for i, d in enumerate(r["dates"])}
                for k in range(r["n_members"]):
                    row = []
                    for d in dates:
                        i = ix.get(d)
                        row.append(None if dm is None or i is None else dm["members"][k][i])
                    rows.append(row)
            div[c] = {"members": rows, "weights": [round(x, 5) for x in mem_w]}
        rec = {"model": "blend", "cycle": T, "init": tT.strftime("%Y-%m-%dT%HZ"), "run_day0": dates[0],
               "dates": dates, "n_members": len(mem_w), "div": div,
               "sources": {m: {"cycle": c, "weight": round(W[m] / wt, 3)} for m, c in pick.items()}}
        os.makedirs(ARCH, exist_ok=True)
        with gzip.open(apath("blend", T), "wt") as f:
            json.dump(rec, f, separators=(",", ":"))
        made.append(T)
    if made:
        print(f"  blend: {len(made)} issues ({made[0]} .. {made[-1]}), weights "
              + " ".join(f"{m} {W[m]:.2f}" for m in W))
    return W


def build(keep_prev=4, hist_keep=40, obs_days=120):
    divs = divisions(); a = area()
    obs = {}
    for p in sorted(glob.glob(os.path.join(OBS, "????-??-??.json"))):
        r = json.load(open(p)); obs[r["date"]] = r["div"]
    odates = sorted(obs)[-obs_days:]
    obs_out = {"dates": odates, "div": {}, "comp": {}, "normal": {}}
    for d in divs:
        c = d["code"]
        obs_out["div"][c] = [obs[x].get(c) for x in odates]
        obs_out["normal"][c] = [normal_mm(c, x) for x in odates]
    for name in COMPOSITES:
        cs = members_of(name)
        if not cs:
            continue
        w = np.array([a[c] for c in cs]); w = w / w.sum()
        vals = []
        for x in odates:
            row = [obs[x].get(c) for c in cs]
            vals.append(None if any(v is None for v in row) else round(float(w @ np.array(row)), 2))
        obs_out["comp"][name] = vals; obs_out["normal"][name] = [normal_mm(name, x) for x in odates]
    built = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    latest = {"built": built, "divisions": divs, "composites": {k: members_of(k) for k in COMPOSITES},
              "periods": PERIODS, "obs": obs_out, "models": {}}
    hist = {"built": built, "periods": PERIODS, "models": {}}
    # verification of the raw models first, so the blend weights can use it
    pre = {}
    for m in BLEND_PRIOR:
        rs = [r for r in (load(m, c) for c in cycles(m)[-hist_keep:]) if r]
        if rs:
            pre[m] = rs
    W = make_blends(verification(pre, obs))
    latest["blend_weights"] = W
    recs_by_model = {}
    for m in sorted({os.path.basename(p).rsplit("_", 1)[0] for p in glob.glob(os.path.join(ARCH, "*_??????????.json.gz"))}):
        cs = cycles(m)
        # bound the repo: the page shows the last few issues; drop older archives
        for old in cs[:-hist_keep]:
            try: os.remove(apath(m, old))
            except OSError: pass
        recs = [r for r in (load(m, c) for c in cs[-hist_keep:]) if r]
        if not recs:
            continue
        recs_by_model[m] = recs
        cur = recs[-1]; s_cur = full_series(cur)
        e = {k: cur[k] for k in ("cycle", "init", "run_day0", "dates", "n_members")}
        if cur.get("sources"):
            e["sources"] = cur["sources"]
        e["series"] = {k: stats_of(v["members"], v.get("weights")) for k, v in s_cur.items()}
        # every member, for the plumes (user: "full plumes with all members");
        # composites and divisions alike, one decimal
        if cur["n_members"] > 1:
            for k, v in s_cur.items():
                e["series"][k]["members"] = [[None if x is None or not np.isfinite(x) else round(float(x), 1) for x in row]
                                             for row in np.array(v["members"], float).tolist()]
                if v.get("weights"):
                    e["series"][k]["weights"] = v["weights"]
        e["periods"] = periods_of(s_cur, cur["dates"])
        e["prev"] = [{"cycle": pr["cycle"], "init": pr["init"], "delta": delta(s_cur, cur["dates"], full_series(pr), pr["dates"])}
                     for pr in recs[-1 - keep_prev:-1][::-1]]
        latest["models"][m] = e
        rows = []
        for i, r in enumerate(recs):
            s = full_series(r); per = periods_of(s, r["dates"])
            row = {"cycle": r["cycle"], "init": r["init"], "run_day0": r["run_day0"],
                   "w": {k: {p: [v["mm"], v.get("pct"), 1 if v["n"] < v["want"] else 0] for p, v in pp.items()} for k, pp in per.items()},
                   "dprev": None}
            if i > 0:
                dl = delta(s, r["dates"], full_series(recs[i - 1]), recs[i - 1]["dates"])
                row["dprev"] = {k: {p: [v["mm"], v.get("pct")] for p, v in dd["periods"].items()} for k, dd in dl.items()}
            rows.append(row)
        hist["models"][m] = rows
    latest["verif"] = verification(recs_by_model, obs)
    os.makedirs(DATA, exist_ok=True)
    for name, obj in (("pnw_latest.json", latest), ("pnw_history.json", hist)):
        tmp = os.path.join(DATA, name + ".tmp")
        json.dump(obj, open(tmp, "w"), separators=(",", ":")); os.replace(tmp, os.path.join(DATA, name))
    print(f"  built: {list(latest['models'])}, Stage IV {odates[0] if odates else '-'} .. {odates[-1] if odates else '-'}")


def backfill(models, days, workers):
    """Every cycle of the last `days` days not yet archived, oldest first.
    GDPS/GEPS keep ~a day on Datamart and simply fail their probe further
    back; the S3 and Google mirrors reach back for the rest."""
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    for model in models:
        m = MODELS[model]()
        have = set(cycles(model))
        day = (now - dt.timedelta(days=days)).date()
        while day <= now.date():
            for h in sorted(m.cycles):
                c = dt.datetime(day.year, day.month, day.day, h)
                tag = c.strftime("%Y%m%d%H")
                if tag in have or c > now - dt.timedelta(hours=m.lag_h):
                    continue
                if not m.available(c):
                    continue
                try:
                    run_cycle(model, workers=workers, cyc=c)
                except Exception as e:
                    print(f"  {model} {tag}: {type(e).__name__}: {e}")
            day += dt.timedelta(days=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=os.environ.get("COLUMBIA_MODELS", "gfs,ecmwf,aifs,gdps,geps,geps_ext,gefs,ecmwf_ens"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--obs-days", type=int, default=20)
    ap.add_argument("--no-fetch", action="store_true", help="rebuild the JSON from the archive only")
    ap.add_argument("--backfill", type=int, default=0, help="also archive every cycle of the last N days")
    a = ap.parse_args()
    ms = [x for x in a.models.split(",") if x]
    if not a.no_fetch:
        got = backfill_obs(a.obs_days)
        if got:
            print(f"  stage4: {len(got)} new days ({got[0]} .. {got[-1]})")
        if a.backfill:
            backfill(ms, a.backfill, a.workers)
        for m in ms:
            try:
                run_cycle(m, workers=a.workers)
            except Exception as e:
                print(f"  {m}: {type(e).__name__}: {e}")
    build()


if __name__ == "__main__":
    main()
