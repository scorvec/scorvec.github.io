#!/usr/bin/env python3
"""Drift and skill of the Hadley outflow and descent forecasts (hadley_branch_fc.py), measured on archived runs against
ERA5: the drift the live members are corrected by, and the evidence that gates which leads the page shows.

ARCHIVE (release hbranch-ref-v1, one REF file per stream). ERA5's daily fields (hadley_branches.fields of the mean of the
00, 06, 12 and 18 UTC states from ARCO, as the normals' own days in build_branch_clim.py) on every day the lattice's
windows touch; each model's archived 00Z runs on overturn's 5-day lattice of starts: REF_MEMBERS perturbed members'
fields of the 00 UTC state of each lead day 0..15. One valid time a day keeps the download to a quarter; what separates
it from ERA5's daily mean (the tides in sea-level pressure among it) is systematic and goes into the drift with the rest
of the model's error. FIELDS on the rows ROWS of the 1.5 deg grid, int16 (PACK). The Google mirror keeps IFS-ENS on the
13 levels from January 2025 and AIFS-ENS from July 2025; GEFS comes from NOAA's S3 bucket.

DRIFT. A member's error at lead L, its field less ERA5's daily field of the valid day, averaged over the members, is
regressed per grid point on the start's common-year day (mean + annual harmonic) and shrunk by its signal-to-noise
(hadley_slp_ref.fit_drift and eval_drift: a systematic error is taken out, sampling noise is not).

SKILL. Every member less the drift, its days led in by ERA5's (the live windows' analysed days are the analysis less its
offset from ERA5), in TRAILING 7-day windows ending on lead days 0..15, against ERA5's window: the maps' fields MAP_FIELDS
by MSSS over the map's domain (cos-latitude weighted; sea-level pressure below build_slp_clim.TERRAIN_M), the tracker's
indices KEYS by MSSS of the index, each against the ERA5 normal (the harmonic maps, the indices' mean by day of year,
build_branch_clim.py). The archived runs hold REF_MEMBERS members and the live ones 30 or 50, so each start's squared
error of the members' mean is taken down by (1/REF_MEMBERS - 1/live) times the members' sample variance: the expected
error of the live ensemble's mean, the "fair" score. The drift is cross-validated (refitted without the start's calendar
month); intervals and one-sided p-values come from a bootstrap over whole months of starts, Benjamini-Hochberg at
hadley_slp_ref.FDR_Q over every model, target and lead; a target passes a lead when it survives and its MSSS reaches
hadley_slp_ref.MIN_SKILL, and it is shown from day 0 up to the first lead that fails. The blend pools the three corrected
ensembles, each weighted by its live member count, on the starts they share.

    python src/hadley_branch_ref.py truth --since 2024-03-01 [--part k/n] --out era5_0.npz
    python src/hadley_branch_ref.py archive --model ifs --since 2025-01-10 [--part k/n] --out ifs_0.npz
    python src/hadley_branch_ref.py pack --stream ifs prev/*.npz parts/ifs_*.npz     # -> REF[stream]
    python src/hadley_branch_ref.py verify --out-dir out                               # -> DRIFT, SKILL, figures
"""

from __future__ import annotations

import pyproj  # noqa: F401  (load pyproj's libproj before xarray/eccodes)

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ecmwf"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "strat"))
import build_branch_clim as C
import build_slp_clim as S
import hadley_branch_maps as M
import hadley_branches as HB
import hadley_slp_ref as R
import overturn as O

REF_DIR = Path(__file__).resolve().parent.parent / "data" / "reference"
STREAMS = ("era5", "aifs", "ifs", "gefs")
REF = {s: REF_DIR / f"hbranch_ref_{s}.npz" for s in STREAMS}  # release hbranch-ref-v1
DRIFT = REF_DIR / "hbranch_drift.npz"  # release hbranch-ref-v1
SKILL = REF_DIR / "hbranch_skill.json"  # committed: the page's lead gate
LEADS = tuple(range(16))  # lead days; the window ending on lead day j holds days j-6..j
WIN = M.WIN
REF_MEMBERS = 3  # perturbed members per archived run
MODELS = {
    "aifs": {"label": "AIFS-ENS", "store": "aifs-ens", "z": "z", "first": "2025-07-01", "members": 50},
    "ifs": {"label": "IFS-ENS", "store": "ifs", "z": "gh", "first": "2025-01-10", "members": 50},
    "gefs": {"label": "GEFS", "store": None, "z": None, "first": "2024-03-01", "members": 30},
}
FIELDS = ("vout", "wmid", "wmid_s", "z500", "msl")  # wmid_s: the maps' smoothed vertical motion
PACK = {"vout": (0.0, 0.01), "wmid": (0.0, 1e-4), "wmid_s": (0.0, 1e-4), "z500": (5500.0, 0.1), "msl": (1010.0, 0.01)}
MAP_FIELDS = ("wmid_s", "msl", "z500")  # the descent shading, the pressure shading and highs, the 588-dam line
KEYS = (
    [f"out_{s}" for s in M.ROWS]
    + [f"desc_{s}" for s in M.ROWS]
    + [f"r588_{s}" for s in HB.HIGHS]
    + [f"high_{s}" for s in HB.HIGHS]
)
ROWS = np.flatnonzero(np.abs(HB.LAT) <= 51.0)  # the maps' domain and a row beyond
LAT = HB.LAT[ROWS]
W_LEVELS = tuple(L for L in HB.DT_LEVELS if HB.DESC_LAYER[0] <= L <= HB.DESC_LAYER[1])
_LEV = {L: i for i, L in enumerate(HB.LEVELS)}
_OFF = np.array([PACK[f][0] for f in FIELDS], np.float32)[:, None, None]
_SCALE = np.array([PACK[f][1] for f in FIELDS], np.float32)[:, None, None]
GEFS_WANT = (
    {(v, f"{L} mb") for v in ("UGRD", "VGRD", "TMP") for L in HB.DT_LEVELS}
    | {("VVEL", f"{L} mb") for L in W_LEVELS}
    | {("HGT", "500 mb"), ("PRMSL", "mean sea level")}
)


def pack16(x: np.ndarray) -> np.ndarray:
    """(..., field, row, lon) float -> int16."""
    return np.clip(np.round((x - _OFF) / _SCALE), -32767, 32767).astype(np.int16)


def unpack16(q: np.ndarray) -> np.ndarray:
    return (q.astype(np.float32) * _SCALE + _OFF).astype(np.float32)


# ── sources ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def decode(msg: bytes) -> tuple[str, int, int, int, np.ndarray]:
    """(shortName, level hPa (0 at the surface), member number, step h, field on the 1.5 deg grid) of a GRIB message."""
    import eccodes

    h = eccodes.codes_new_from_message(msg)
    try:
        g = {k: eccodes.codes_get(h, k) for k in ("Ni", "Nj", "jScansPositively", "endStep", "shortName", "level")}
        n = eccodes.codes_get(h, "number") if eccodes.codes_is_defined(h, "number") else 0
        la0 = eccodes.codes_get(h, "latitudeOfFirstGridPointInDegrees")
        lo0 = eccodes.codes_get(h, "longitudeOfFirstGridPointInDegrees")
        dj = eccodes.codes_get(h, "jDirectionIncrementInDegrees")
        di = eccodes.codes_get(h, "iDirectionIncrementInDegrees")
        vals = eccodes.codes_get_values(h).reshape(g["Nj"], g["Ni"])
    finally:
        eccodes.codes_release(h)
    lat = la0 + (dj if g["jScansPositively"] else -dj) * np.arange(g["Nj"])
    f = HB.coarse15(vals, lat, lo0 + di * np.arange(g["Ni"])).astype(np.float32)
    return str(g["shortName"]), int(g["level"]), int(n), int(g["endStep"]), f


def _put(st: dict, short: str, lev: int, f: np.ndarray):
    """File one decoded field into a state: u, v, t and omega on LEVELS (NaN where not fetched), z500 (m), msl (Pa)."""
    if short in ("u", "v", "t", "w") and lev in _LEV:
        st.setdefault(short, np.full((len(HB.LEVELS), HB.LAT.size, HB.LON.size), np.nan, np.float32))[_LEV[lev]] = f
    elif short in ("gh", "z") and lev == 500:
        st["z500"] = f / C.G if short == "z" else f
    elif short in ("msl", "prmsl"):
        st["msl"] = f


def derived(st: dict) -> np.ndarray | None:
    """(len(FIELDS), row, lon) of a state, None if it is incomplete."""
    need = (("u", HB.DT_LEVELS), ("v", HB.DT_LEVELS), ("t", HB.DT_LEVELS), ("w", W_LEVELS))
    if "z500" not in st or "msl" not in st:
        return None
    if any(k not in st or np.isnan(st[k][[_LEV[L] for L in levs]]).any() for k, levs in need):
        return None
    F = HB.fields(st["u"], st["v"], st["w"], st["t"], st["z500"], st["msl"])
    F["wmid_s"] = HB.large_scale(F["wmid"], efold=M.W_EFOLD)
    return np.stack([F[f][ROWS] for f in FIELDS]).astype(np.float32)


def gefs_member_step(date: str, member: int, step: int) -> list[bytes]:
    """The GEFS_WANT messages of one GEFS member (0 = control) at one step of the 00Z run, byte-ranged from NOAA's S3
    bucket through the pgrb2a and pgrb2b .idx files. Network only, so it runs in threads."""
    from qbo_duct import _get, gefs_url

    mem = "gec00" if member == 0 else f"gep{member:02d}"
    out = []
    for kind in ("a", "b"):
        url = gefs_url(date, "00", mem, kind, step)
        lines = [x.split(":") for x in _get(url + ".idx").decode().splitlines() if x.strip()]
        for i, p in enumerate(lines):
            if (p[3], p[4]) in GEFS_WANT:
                end = int(lines[i + 1][1]) - 1 if i + 1 < len(lines) else -1
                out.append(bytes(_get(url, (int(p[1]), end))))
    return out


def states(model: str, init: pd.Timestamp, steps: tuple, members: list[int], workers: int = 16) -> dict:
    """{(member, step h): state} of the 00Z run at init. ECMWF: one file per group of fields from the Google mirror (a
    failure raises); GEFS: member-steps byte-ranged in threads and decoded here (a member-step that fails stays
    incomplete)."""
    out: dict = {}
    if model == "gefs":
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(workers) as ex:
            futs = {(m, s): ex.submit(gefs_member_step, f"{init:%Y%m%d}", m, s) for m in members for s in steps}
            for (m, s), fut in futs.items():
                try:
                    for msg in fut.result():
                        short, lev, _, _, f = decode(msg)
                        _put(out.setdefault((m, s), {}), short, lev, f)
                except Exception as e:  # noqa: BLE001
                    print(f"  GEFS {init:%Y-%m-%d} member {m} +{s} h: {type(e).__name__} {str(e)[:80]}", flush=True)
        return out
    import store as ecmwf

    cfg = MODELS[model]
    cyc = ecmwf.Cycle(f"{init:%Y%m%d}", "00")
    n = max(members) if max(members) < ecmwf.PF_MEMBERS else 0
    specs = (
        ecmwf.Spec(cfg["store"], "pf", ("u", "v", "t"), "pl", HB.DT_LEVELS, steps, n),
        ecmwf.Spec(cfg["store"], "pf", "w", "pl", W_LEVELS, steps, n),
        ecmwf.Spec(cfg["store"], "pf", cfg["z"], "pl", (500,), steps, n),
        ecmwf.Spec(cfg["store"], "pf", "msl", "sfc", (), steps, n),
    )
    want = set(members)
    for spec in specs:
        p = ecmwf.ensure(cyc, spec)
        for msg in M._messages(p.read_bytes()):
            short, lev, m, s, f = decode(msg)
            if m in want:
                _put(out.setdefault((m, s), {}), short, lev, f)
        p.unlink(missing_ok=True)
        p.with_suffix(p.suffix + ".json").unlink(missing_ok=True)
    return out


def run_fields(model: str, init: pd.Timestamp, members: list[int], group: int) -> np.ndarray:
    """(member, lead day, field, row, lon) of the 00Z run at init, from each lead day's 00 UTC state, fetched `group`
    lead days at a time; NaN where a member's state is incomplete."""
    out = np.full((len(members), len(LEADS), len(FIELDS), ROWS.size, HB.LON.size), np.nan, np.float32)
    for g in range(0, len(LEADS), group):
        leads = LEADS[g : g + group]
        got = states(model, init, tuple(24 * L for L in leads), members)
        for j, L in enumerate(leads, start=g):
            done = 0
            for i, m in enumerate(members):
                x = derived(got.pop((m, 24 * L), {}))
                if x is not None:
                    out[i, j], done = x, done + 1
            print(f"  {MODELS[model]['label']} {init:%Y-%m-%d} day {L}: {done} of {len(members)} members", flush=True)
    return out


def era5_days(days) -> dict:
    """ERA5's daily fields from ARCO; the first day it has not published ends the pass."""
    keep, rows = [], []
    for d in days:
        try:
            times = [M.era5_time(d + pd.Timedelta(hours=h)) for h in M.HOURS]
        except KeyError:  # beyond the store's time axis
            times = [None]
        if any(x is None for x in times):
            print(f"  ERA5 not published for {d:%Y-%m-%d}: stopping", flush=True)
            break
        keep.append(d)
        rows.append(pack16(derived(M.day_mean(times))))
        print(f"  ERA5 {d:%Y-%m-%d}", flush=True)
    n = len(keep)
    return {
        "day": np.array(keep, "datetime64[D]"),
        "x": np.array(rows, np.int16).reshape(n, len(FIELDS), ROWS.size, HB.LON.size),
    }


def archive(model: str, starts) -> dict:
    """The model's archived runs at the lattice starts: REF_MEMBERS perturbed members' fields, packed; a start with an
    incomplete member is skipped."""
    keep, rows = [], []
    for d in starts:
        try:
            x = run_fields(model, d, list(range(1, REF_MEMBERS + 1)), group=len(LEADS))
        except Exception as e:  # noqa: BLE001
            print(f"  {MODELS[model]['label']} {d:%Y-%m-%d}: {type(e).__name__} {str(e)[:120]}", flush=True)
            continue
        if np.isnan(x).any():
            print(f"  {MODELS[model]['label']} {d:%Y-%m-%d}: incomplete, skipped", flush=True)
            continue
        keep.append(d)
        rows.append(pack16(x))
        print(f"  {MODELS[model]['label']} {d:%Y-%m-%d} archived", flush=True)
    n = len(keep)
    return {
        "day": np.array(keep, "datetime64[D]"),
        "x": np.array(rows, np.int16).reshape(n, REF_MEMBERS, len(LEADS), len(FIELDS), ROWS.size, HB.LON.size),
    }


def pack(stream: str, paths) -> dict:
    """Merge a stream's files (any order, overlaps allowed: the later file's copy of a day is kept) into REF[stream]."""
    t, x = [], []
    for p in paths:
        with np.load(p) as f:
            t.append(f["day"])
            x.append(f["x"])
    t, x = np.concatenate(t), np.concatenate(x)
    _, last = np.unique(t[::-1], return_index=True)
    keep = t.size - 1 - last  # sorted by day
    doc = {"day": t[keep], "x": x[keep]}
    REF[stream].parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(REF[stream], **doc)
    return doc


# ── windows and normals ─────────────────────────────────────────────────────────────────────────────────────────────
def member_windows(x: np.ndarray, before: np.ndarray) -> np.ndarray:
    """(member, window, field, point): the trailing WIN-day means ending on lead days 0..15 of each member's days
    (member, lead, field, point), led in by the WIN - 1 days before the start (WIN - 1, field, point)."""
    seq = np.concatenate([np.broadcast_to(before, (x.shape[0], *before.shape)), x], axis=1).astype(np.float64)
    c = np.concatenate([np.zeros_like(seq[:, :1]), np.cumsum(seq, axis=1)], axis=1)
    return ((c[:, WIN:] - c[:, :-WIN]) / WIN).astype(np.float32)


def pad(x: np.ndarray) -> dict:
    """(len(FIELDS), row, lon) -> {field: map on the whole 1.5 deg grid}, NaN beyond ROWS (hadley_branches.indices
    reads within them)."""
    out = {}
    for k, f in enumerate(FIELDS):
        g = np.full((HB.LAT.size, HB.LON.size), np.nan, np.float32)
        g[ROWS] = x[k]
        out[f] = g
    return out


def indices(x: np.ndarray) -> np.ndarray:
    """KEYS of a (len(FIELDS), row, lon) set of 7-day-mean fields."""
    d = HB.indices(pad(x))
    return np.array([d[k] for k in KEYS])


class Normals:
    """ERA5 1991-2020 (build_branch_clim.py) on ROWS: the FIELDS' harmonic maps at a day (wmid_s: the smoothed wmid
    normal) and the KEYS' means by day of year."""

    def __init__(self):
        self.cl = xr.open_dataset(C.OUT)
        coef = {f: self.cl["coef"].sel(field=f).values for f in ("vout", "wmid", "z500", "msl")}
        coef["wmid_s"] = np.stack([HB.large_scale(c, efold=M.W_EFOLD) for c in coef["wmid"]])
        self.coef = np.stack([coef[f][:, ROWS] for f in FIELDS]).astype(np.float32)  # (field, harmonic, row, lon)
        self._maps: dict = {}
        self._index: dict = {}

    def maps(self, day: pd.Timestamp) -> np.ndarray:
        """(field, row, lon) normal of a day."""
        doy = M._doy(day)
        if doy not in self._maps:
            self._maps[doy] = np.einsum("h,fhij->fij", M._harm(doy).astype(np.float32), self.coef)
        return self._maps[doy]

    def window(self, end: pd.Timestamp) -> np.ndarray:
        """The normal of the 7-day window ending on `end`, at its midpoint."""
        return self.maps(end - pd.Timedelta(days=(WIN - 1) / 2))

    def index(self, end: pd.Timestamp) -> np.ndarray:
        """KEYS' ERA5 mean for the window ending on `end`."""
        doy = M._doy(end - pd.Timedelta(days=(WIN - 1) / 2))
        if doy not in self._index:
            self._index[doy] = np.array([M.clim_index(self.cl, k, doy, "mean")[0] for k in KEYS])
        return self._index[doy]


def weights() -> dict:
    """{map field: (row, lon) weight} for the maps' scores: cos(latitude) over the map's extent; sea-level pressure
    only below build_slp_clim.TERRAIN_M."""
    with xr.open_dataarray(S.OROG) as o:
        high = HB.coarse15(o.values, o.latitude.values, o.longitude.values)[ROWS] > S.TERRAIN_M
    inside = (np.abs(LAT) <= M.EXTENT[1])[:, None] * np.cos(np.deg2rad(LAT))[:, None] * np.ones(HB.LON.size)
    return {f: (inside * (~high if f == "msl" else 1.0)).astype(np.float32) for f in MAP_FIELDS}


# ── the drift ───────────────────────────────────────────────────────────────────────────────────────────────────────
def doy_of(days) -> np.ndarray:
    return R.doy_of(days)


class Drift:
    """The live drift (DRIFT): d*(lead, field, row, lon) for a model at a start's common-year day; None without a fit."""

    def __init__(self, path: Path = DRIFT):
        self.f = dict(np.load(path)) if path.exists() else {}

    def at(self, model: str, doy: float) -> np.ndarray | None:
        if f"{model}_coef" not in self.f:
            return None
        coef, cov, s2, kappa = (self.f[f"{model}_{k}"] for k in ("coef", "cov", "s2", "kappa"))
        out = np.zeros((len(LEADS), len(FIELDS), ROWS.size * HB.LON.size), np.float32)
        for j in range(len(LEADS)):
            for k in range(len(FIELDS)):
                fit = {"coef": coef[j, k].reshape(R.DRIFT_TERMS, -1), "cov": cov[j, k], "s2": s2[j, k].ravel()}
                fit["kappa"] = kappa[j, k]
                out[j, k] = R.eval_drift(fit, [doy])[0]
        return out.reshape(len(LEADS), len(FIELDS), ROWS.size, HB.LON.size)


# ── verification ────────────────────────────────────────────────────────────────────────────────────────────────────
class Era5:
    """ERA5's daily fields by day, unpacked: (field, point) rows."""

    def __init__(self):
        with np.load(REF["era5"]) as f:
            self.day = pd.DatetimeIndex(f["day"])
            self.x = unpack16(f["x"]).reshape(len(self.day), len(FIELDS), -1)
        self.pos = {d: i for i, d in enumerate(self.day)}

    def days(self, first: pd.Timestamp, n: int) -> np.ndarray:
        """(n, field, point) from `first`, NaN on a day not held."""
        out = np.full((n, len(FIELDS), self.x.shape[-1]), np.nan, np.float32)
        for k in range(n):
            i = self.pos.get(first + pd.Timedelta(days=k))
            if i is not None:
                out[k] = self.x[i]
        return out


def drift_fits(starts: pd.DatetimeIndex, err: np.ndarray, low: np.ndarray) -> tuple[list, np.ndarray]:
    """Per lead and field: the drift fitted on every start, and the cross-validated drift of each start (refitted
    without its calendar month): (fits[lead][field], cv (start, lead, field, point))."""
    fold = np.array([f"{s:%Y-%m}" for s in starts])
    cv = np.full_like(err, np.nan)
    fits = []
    for L in LEADS:
        ok = np.isfinite(err[:, L, 0, 0])
        fits.append([])
        for k in range(len(FIELDS)):
            fits[-1].append(R.fit_drift(err[ok, L, k], starts[ok], low))
            for f in np.unique(fold[ok]):
                tr, te = ok & (fold != f), ok & (fold == f)
                cv[te, L, k] = R.eval_drift(R.fit_drift(err[tr, L, k], starts[tr], low), doy_of(starts[te]))
    return fits, cv


class Run:
    """One model's archived runs for the scoring: members as fields, the cross-validated drift."""

    def __init__(self, model: str, E: Era5, low: np.ndarray):
        with np.load(REF[model]) as f:
            self.starts = pd.DatetimeIndex(f["day"])
            self.q = f["x"]  # (start, member, lead, field, row, lon) int16
        self.model, self.live = model, MODELS[model]["members"]
        err = np.full((len(self.starts), len(LEADS), len(FIELDS), ROWS.size * HB.LON.size), np.nan, np.float32)
        for i, s in enumerate(self.starts):
            err[i] = self.members(i).mean(0) - E.days(s, len(LEADS))
        self.fits, self.cv = drift_fits(self.starts, err, low)
        self.pos = {s: i for i, s in enumerate(self.starts)}

    def members(self, i: int) -> np.ndarray:
        """(member, lead, field, point) of the i-th start."""
        return unpack16(self.q[i]).reshape(REF_MEMBERS, len(LEADS), len(FIELDS), -1)

    def windows(self, s: pd.Timestamp, before: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(member, window, field, point) of the start's corrected members, and of the raw ones."""
        i = self.pos[s]
        x = self.members(i)
        return member_windows(x - self.cv[i][None], before), member_windows(x, before)


def _member_stats(wins: list, wts: list, fair: list) -> tuple[np.ndarray, np.ndarray]:
    """The pooled members' mean (weights wts over the ensembles) and the fair adjustment's variance term
    sum_m w_m^2 (1/REF_MEMBERS - 1/live_m) s_m^2, for (member, ...) arrays of each ensemble."""
    mean = sum(w * x.mean(0) for x, w in zip(wins, wts, strict=True))
    var = sum(w * w * c * x.var(0, ddof=1) for x, w, c in zip(wins, wts, fair, strict=True))
    return mean, var


def _member_indices(x: np.ndarray, shp: tuple) -> np.ndarray:
    """(member, KEYS) of (member, field, point) windows."""
    return np.stack([indices(m.reshape(shp)) for m in x])


def score(runs: list[Run], starts, E: Era5, N: Normals, wts: dict) -> dict:
    """Per start and lead: the squared-error sums of each target (MAP_FIELDS weighted over the map, KEYS) for the
    corrected pooled mean (fair), the raw one, the ERA5 normal and persistence of the analysed window before the
    start."""
    total = sum(r.live for r in runs)
    w_m = [r.live / total for r in runs]
    fair = [1.0 / REF_MEMBERS - 1.0 / r.live for r in runs]
    targets = list(MAP_FIELDS) + list(KEYS)
    kf = [FIELDS.index(f) for f in MAP_FIELDS]
    W = np.stack([wts[f].ravel() for f in MAP_FIELDS])  # (map field, point)
    shp = (len(FIELDS), ROWS.size, HB.LON.size)
    out = {k: np.full((len(starts), len(LEADS), len(targets)), np.nan) for k in ("fc", "raw", "ref", "per")}
    for i, s in enumerate(starts):
        seq = E.days(s - pd.Timedelta(days=WIN), WIN + len(LEADS))  # days -7 .. 15
        before = seq[1:WIN]
        if np.isnan(before).any() or np.isnan(seq[0]).any():
            continue
        truth = member_windows(seq[None, WIN:], before)[0]  # (window, field, point)
        per = seq[:WIN].mean(0)  # the window ending the day before the start
        wc, wr = zip(*(r.windows(s, before) for r in runs), strict=True)
        mc, vc = _member_stats(list(wc), w_m, fair)
        mr = sum(w * x.mean(0) for x, w in zip(wr, w_m, strict=True))
        per_end = s - pd.Timedelta(days=1)
        a0 = per - N.window(per_end).reshape(len(FIELDS), -1)
        ai0 = indices(per.reshape(shp)) - N.index(per_end)
        for j in LEADS:
            if np.isnan(truth[j]).any():
                continue
            end = s + pd.Timedelta(days=j)
            nrm = N.window(end).reshape(len(FIELDS), -1)
            o = truth[j]
            out["fc"][i, j, : len(kf)] = (W * ((mc[j, kf] - o[kf]) ** 2 - vc[j, kf])).sum(1)
            out["raw"][i, j, : len(kf)] = (W * ((mr[j, kf] - o[kf]) ** 2)).sum(1)
            out["ref"][i, j, : len(kf)] = (W * (o[kf] - nrm[kf]) ** 2).sum(1)
            out["per"][i, j, : len(kf)] = (W * ((o[kf] - nrm[kf]) - a0[kf]) ** 2).sum(1)
            oi, mu = indices(o.reshape(shp)), N.index(end)
            fi, vi = _member_stats([_member_indices(x[:, j], shp) for x in wc], w_m, fair)
            ri = sum(w * _member_indices(x[:, j], shp).mean(0) for x, w in zip(wr, w_m, strict=True))
            out["fc"][i, j, len(kf) :] = (fi - oi) ** 2 - vi
            out["raw"][i, j, len(kf) :] = (ri - oi) ** 2
            out["ref"][i, j, len(kf) :] = (oi - mu) ** 2
            out["per"][i, j, len(kf) :] = ((oi - mu) - ai0) ** 2
        if s.day <= 5:
            print(f"  scored {'+'.join(r.model for r in runs)} {s:%Y-%m}", flush=True)
    out["starts"] = pd.DatetimeIndex(starts)
    out["targets"] = targets
    return out


def verify(out_dir: Path, seed: int = 1) -> dict:
    E = Era5()
    N = Normals()
    wts = weights()
    low = np.ones(ROWS.size * HB.LON.size, bool)  # the drift's serial correlation: every point
    rng = np.random.default_rng(seed)
    runs = {}
    for m in MODELS:
        if not REF[m].exists() or len(np.load(REF[m])["day"]) < R.MIN_STARTS:
            print(f"{MODELS[m]['label']}: fewer than {R.MIN_STARTS} archived starts, not scored")
            continue
        runs[m] = Run(m, E, low)
        print(f"{MODELS[m]['label']}: {len(runs[m].starts)} starts, drift fitted", flush=True)
    sums = {m: score([r], r.starts, E, N, wts) for m, r in runs.items()}
    if len(runs) == len(MODELS):
        common = sorted(set.intersection(*(set(r.starts) for r in runs.values())))
        sums["blend"] = score(list(runs.values()), common, E, N, wts)
    tests = {}
    for m, Sm in sums.items():
        months = np.array([f"{s:%Y-%m}" for s in Sm["starts"]])
        names, block = np.unique(months, return_inverse=True)
        counts = np.stack(
            [np.bincount(x, minlength=names.size) for x in rng.integers(0, names.size, (R.BOOT, names.size))]
        )
        tests[m] = {"nblock": names.size}
        for kind in ("fc", "raw", "per"):
            res = [
                R.skill_test(Sm[kind][..., t], Sm["ref"][..., t], block, names.size, counts)
                for t in range(len(Sm["targets"]))
            ]
            tests[m][kind] = [np.stack(x) for x in zip(*res, strict=True)]  # msss, lo, hi, p: (target, lead)
        if names.size < R.MIN_BLOCKS:
            tests[m]["fc"][3][:] = 1.0
    order = list(tests)
    targets = list(MAP_FIELDS) + list(KEYS)
    passed = R.bh(np.concatenate([tests[m]["fc"][3].ravel() for m in order])).reshape(
        len(order), len(targets), len(LEADS)
    )
    doc = {
        "members": REF_MEMBERS,
        "live_members": {m: MODELS[m]["members"] for m in MODELS},
        "score": "MSSS against the ERA5 normal; the members' mean error made fair for the live ensemble size",
        "windows": f"trailing {WIN}-day means ending on lead days 0..{LEADS[-1]}, led in by ERA5 days",
        "bootstrap": R.BOOT,
        "blocks": "calendar months of start dates",
        "cross_validation": "drift refitted without the start's calendar month",
        "fdr_q": R.FDR_Q,
        "min_skill": R.MIN_SKILL,
        "models": {},
    }
    for a, m in enumerate(order):
        t, Sm = tests[m], sums[m]
        doc["models"][m] = {
            "label": "Three ensembles" if m == "blend" else MODELS[m]["label"],
            "starts": len(Sm["starts"]),
            "first": f"{Sm['starts'][0]:%Y-%m-%d}",
            "last": f"{Sm['starts'][-1]:%Y-%m-%d}",
            "months": int(t["nblock"]),
            "targets": {},
        }
        for b, name in enumerate(targets):
            ok = passed[a, b] & (t["fc"][0][b] >= R.MIN_SKILL)
            show = 0
            while show < len(LEADS) and ok[show]:
                show += 1
            doc["models"][m]["targets"][name] = {
                "show": show,  # windows ending on lead days 0 .. show - 1
                "msss": _r3(t["fc"][0][b]),
                "ci": [_r3(t["fc"][1][b]), _r3(t["fc"][2][b])],
                "p": [round(float(x), 4) for x in t["fc"][3][b]],
                "scored": [int(x) for x in np.isfinite(Sm["fc"][..., b]).sum(0)],
                "msss_no_drift": _r3(t["raw"][0][b]),
                "msss_persistence": _r3(t["per"][0][b]),
            }
        if m in runs:
            doc["models"][m]["drift"] = drift_stats(runs[m], wts)
    text = json.dumps(doc, separators=(",", ":")) + "\n"
    SKILL.write_text(text)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / SKILL.name).write_text(text)
    save = {}
    for m, r in runs.items():
        save[f"{m}_coef"] = np.array([[f["coef"] for f in fl] for fl in r.fits])  # (lead, field, term, point)
        save[f"{m}_cov"] = np.array([[f["cov"] for f in fl] for fl in r.fits])
        save[f"{m}_s2"] = np.array([[f["s2"] for f in fl] for fl in r.fits])
        save[f"{m}_kappa"] = np.array([[f["kappa"] for f in fl] for fl in r.fits])
    np.savez_compressed(DRIFT, **{k: v.astype(np.float32) for k, v in save.items()})
    report(doc)
    figures(doc, runs, out_dir)
    return doc


def drift_stats(r: Run, wts: dict) -> dict:
    """The shrunk drift's rms over the map (map fields, natural units) at each lead, as fitted on every start."""
    out = {}
    for f in (*MAP_FIELDS, "vout"):
        k = FIELDS.index(f)
        w = (wts[f] if f in wts else wts["wmid_s"]).ravel()
        w = w / w.sum()
        out[f] = [
            round(float(np.sqrt((R.eval_drift(r.fits[L][k], doy_of(r.starts)) ** 2 @ w).mean())), 4) for L in LEADS
        ]
    return out


def _r3(x):
    return [round(float(v), 3) if np.isfinite(v) else None for v in np.atleast_1d(x)]


def report(doc: dict):
    print(f"MSSS against the ERA5 normal (gate: BH {doc['fdr_q']} and >= {doc['min_skill']}); windows ending day 0..15")
    for d in doc["models"].values():
        print(f"  {d['label']}: {d['starts']} starts {d['first']}..{d['last']}, {d['months']} months")
        for name, t in d["targets"].items():
            ms = " ".join("  .  " if v is None else f"{v:+.2f}" for v in t["msss"])
            nd = " ".join("  .  " if v is None else f"{v:+.2f}" for v in t["msss_no_drift"][::5])
            pe = " ".join("  .  " if v is None else f"{v:+.2f}" for v in t["msss_persistence"][::5])
            print(f"    {name:11s} show {t['show']:2d} | {ms} | no drift d0/5/10/15 {nd} | persistence {pe}")
        if "drift" in d:
            for f, v in d["drift"].items():
                print(f"    drift rms {f}: " + " ".join(f"{x:.3g}" for x in v[::3]))


def figures(doc: dict, runs: dict, out_dir: Path):
    """Scratch figures for the check: the maps' skill by lead per model; the shrunk drift of the maps' fields at days 0,
    7 and 15 (mean of four seasons)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, len(MAP_FIELDS), figsize=(6 * len(MAP_FIELDS), 4.2))
    for d in doc["models"].values():
        for ax, f in zip(axs, MAP_FIELDS, strict=True):
            t = d["targets"][f]
            y = np.array(t["msss"], float)
            col = ax.plot(LEADS, y, lw=2, label=d["label"])[0].get_color()
            ax.fill_between(LEADS, *(np.array(v, float) for v in t["ci"]), color=col, alpha=0.15)
            ax.plot(LEADS, np.array(t["msss_no_drift"], float), lw=1, ls=":", color=col)
            ax.set_title(f"{f}: MSSS of the 7-day window (dotted: no drift)", loc="left", fontsize=10)
    for ax in axs:
        ax.axhline(R.MIN_SKILL, color="#888", lw=0.8)
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
        ax.set_xlabel("window ending on lead day")
    fig.tight_layout()
    fig.savefig(out_dir / "verify_skill.png", dpi=100)
    plt.close(fig)
    scale = {"wmid_s": HB.PA_S_TO_HPA_DAY, "msl": 1.0, "z500": 1.0}
    vmax = {"wmid_s": 10.0, "msl": 3.0, "z500": 30.0}
    for f in MAP_FIELDS:
        k = FIELDS.index(f)
        fig, axs = plt.subplots(len(runs), 3, figsize=(18, 2.2 * len(runs) + 0.6), squeeze=False, layout="constrained")
        for i, (m, r) in enumerate(runs.items()):
            for j, L in enumerate((0, 7, 15)):
                d = R.eval_drift(r.fits[L][k], [1.0, 92.0, 183.0, 274.0]).mean(0).reshape(ROWS.size, -1) * scale[f]
                im = axs[i, j].pcolormesh(HB.LON, LAT, d, cmap="RdBu_r", vmin=-vmax[f], vmax=vmax[f])
                axs[i, j].set_title(f"{MODELS[m]['label']} drift of {f}, day {L}", fontsize=9, loc="left")
        fig.colorbar(im, ax=axs, shrink=0.6)
        fig.savefig(out_dir / f"drift_{f}.png", dpi=80)
        plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("truth", "archive"):
        s = sub.add_parser(name)
        s.add_argument("--since", default=str(O.EPOCH.date()))
        s.add_argument(
            "--until", default=str((pd.Timestamp.now("UTC").tz_localize(None) - pd.Timedelta(days=6)).date())
        )
        s.add_argument("--part", default="0/1")
        s.add_argument("--out", required=True)
        if name == "archive":
            s.add_argument("--model", required=True, choices=sorted(MODELS))
    s = sub.add_parser("pack")
    s.add_argument("--stream", required=True, choices=STREAMS)
    s.add_argument("files", nargs="+")
    s = sub.add_parser("verify")
    s.add_argument("--out-dir", default="out")
    a = ap.parse_args()
    if a.cmd in ("truth", "archive"):
        k, n = (int(x) for x in a.part.split("/"))
        if (
            a.cmd == "truth"
        ):  # every day the lattice's windows touch: the persistence window before a start to its last lead
            first = O.lattice(a.since, a.until)[0] - pd.Timedelta(days=WIN)
            days = pd.date_range(first, pd.Timestamp(a.until) + pd.Timedelta(days=LEADS[-1]))[k::n]
            print(f"ERA5 on {len(days)} days", flush=True)
            doc = era5_days(days)
        else:
            since = max(pd.Timestamp(a.since), pd.Timestamp(MODELS[a.model]["first"]))
            starts = O.lattice(since, a.until, (k, n))
            print(f"{MODELS[a.model]['label']} on {len(starts)} lattice starts", flush=True)
            doc = archive(a.model, starts)
        np.savez_compressed(a.out, **doc)
        return 0
    if a.cmd == "pack":
        doc = pack(a.stream, a.files)
        print({k: v.shape for k, v in doc.items()})
        return 0
    verify(Path(a.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
