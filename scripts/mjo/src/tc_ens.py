#!/usr/bin/env python3
"""Tropical cyclones and the jet, ENSEMBLE view: what the AIFS-ENS and IFS-ENS members say about each storm's
recurvature, and (AIFS-ENS) what recurvature does to the 500 hPa flow downstream.

Tracks: ECMWF's tropical-cyclone tracker output in open data (Google mirror only), one BUFR per model and cycle:
  aifs-ens  .../aifs-ens/0p25/enfo/{d}{hh}0000-360h-enfo-tf.bufr   control (type 1, number 51) + 50 perturbed (type 4)
  ifs-ens   .../ifs/0p25/enfo/{d}{hh}0000-360h-enfo-tf.bufr        50 perturbed (type 4)
  Subsets of type 0 are the deterministic runs (AIFS-single / IFS HRES: identical to their own oper tf files,
  checked 2026-10-02) and are excluded.
Recurvature per member: tc_jet.recurvature (24-h zonal motion from <= -1 to >= +2 m/s).

Step 2 (AIFS-ENS only; IFS-ENS heights not fetched yet): z500 for cf + 50 members, 12-hourly to day 10
(pf ~0.98 GB, cf 20 MB, ~75 s on the laptop). Per storm with >= MIN_GROUP members in each group:
  composite   z500(recurving members) - z500(the rest) at days 5/7/10 (or early vs late recurvers, split at the median
              recurvature time, when nearly all members recurve), Welch t-test per 1-deg cell, Benjamini-Hochberg
              FDR 10 % over all cells of the three maps together; only significant cells are shaded.
  sensitivity regression across members of z500 at days 5/7/10 on the storm's day-3 latitude and longitude
              (ensemble sensitivity, after Torn & Hakim 2008, MWR 136, 663-677, applied in the forecast direction),
              t-test on r, BH FDR 10 % per predictor; only significant cells are shaded.
    python src/tc_ens.py --date 20261002 --time 00 --out-dir ../../assets/sst
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ecmwf"))
import tc_jet as TC                                                     # noqa: E402

URLS = {"aifs": "https://storage.googleapis.com/ecmwf-open-data/{d}/{h:02d}z/aifs-ens/0p25/enfo/{d}{h:02d}0000-360h-enfo-tf.bufr",
        "ifs": "https://storage.googleapis.com/ecmwf-open-data/{d}/{h:02d}z/ifs/0p25/enfo/{d}{h:02d}0000-360h-enfo-tf.bufr"}
MLAB = {"aifs": "AIFS-ENS", "ifs": "IFS-ENS"}
MISSING = -1e99
MIN_GROUP = 8
MIN_SENS = 20                       # members with a day-3 position needed for the sensitivity maps
SENS_H = 72
DAYS = (5, 7, 10)                     # downstream days shown
PRE_DAYS = (0, 2)                      # precursor check: before the storm reaches the jet
STEPS_H = tuple(24 * d for d in PRE_DAYS + DAYS)   # z500 steps loaded; index 0..1 = PRE_DAYS, 2..4 = DAYS
NPRE = len(PRE_DAYS)
ADJ_DAY_IDX = 1                        # the day-2 field the adjusted maps control for
K_PCS = 4                              # leading member-spread patterns of that field used as controls
SECTOR = (100.0, 300.0, 15.0, 75.0)   # lon0, lon1 (0..360), lat0, lat1
MIN_AREA = 0.02                        # a map row is drawn only if >= 2% of it is significant on some day; below that, a note
CACHE = Path(__file__).resolve().parents[2] / "ecmwf" / "cache" / "tc"


def fetch(model: str, date: str, hour: int) -> Path | None:
    import requests
    CACHE.mkdir(parents=True, exist_ok=True)
    fp = CACHE / f"{model}_ens_tf_{date}{hour:02d}.bufr"
    if fp.exists() and fp.stat().st_size > 1000:
        return fp
    r = requests.get(URLS[model].format(d=date, h=hour), timeout=60)
    if r.status_code != 200:
        print(f"  {MLAB[model]} tracks not on the Google mirror ({r.status_code})", flush=True)
        return None
    fp.write_bytes(r.content)
    return fp


def decode_all(fp: Path) -> list[dict]:
    """Every storm message with all ensemble members (types 1 and 4): {id, name, named, tracks: {member: track}}.
    The control is member 0 here (51 in the file); track = dict(steps, lat, lon(-180..180), pmsl hPa, wind m/s)."""
    import eccodes as ec
    out = []
    with open(fp, "rb") as f:
        while True:
            h = ec.codes_bufr_new_from_file(f)
            if h is None:
                break
            try:
                ec.codes_set(h, "unpack", 1)
                sid = ec.codes_get(h, "#1#stormIdentifier").strip()
                name = ec.codes_get(h, "#1#longStormName").strip()
                ft = np.atleast_1d(ec.codes_get_array(h, "#1#ensembleForecastType"))
                mem = np.atleast_1d(ec.codes_get_array(h, "#1#ensembleMemberNumber"))
                n = len(ft)
                def arr(key):
                    a = np.atleast_1d(ec.codes_get_array(h, key)).astype(float)
                    return np.repeat(a, n) if a.size == 1 else a
                rep = ec.codes_get(h, "#1#delayedDescriptorReplicationFactor")
                cols = []                                                # (step, lat, lon, p, w) arrays over subsets
                a_la, a_lo = arr("#2#latitude"), arr("#2#longitude")
                o_la, o_lo = arr("#1#latitude"), arr("#1#longitude")
                a_la = np.where(a_la > MISSING, a_la, o_la); a_lo = np.where(a_lo > MISSING, a_lo, o_lo)
                cols.append((np.zeros(n), a_la, a_lo, arr("#1#pressureReducedToMeanSeaLevel"), arr("#1#windSpeedAt10M")))
                for i in range(1, rep + 1):
                    cols.append((arr(f"#{i}#timePeriod"), arr(f"#{2 * i + 2}#latitude"), arr(f"#{2 * i + 2}#longitude"),
                                 arr(f"#{i + 1}#pressureReducedToMeanSeaLevel"), arr(f"#{i + 1}#windSpeedAt10M")))
                tracks = {}
                for j in range(n):
                    if ft[j] not in (1, 4):
                        continue                                         # type 0 = the deterministic run
                    rows = [(c[0][j], c[1][j], c[2][j], c[3][j], c[4][j]) for c in cols if c[1][j] > MISSING and c[0][j] > MISSING]
                    if not rows:
                        continue
                    st, la, lo, p, w = (np.array(x, float) for x in zip(*rows))
                    m = 0 if ft[j] == 1 else int(mem[j])
                    tracks[m] = dict(steps=st.astype(int), lat=la, lon=(lo + 180) % 360 - 180,
                                     pmsl=np.where(p > MISSING, p / 100.0, np.nan), wind=np.where(w > MISSING, w, np.nan))
                named = not (sid[:2].isdigit() and int(sid[:2]) >= 70)
                if tracks:
                    out.append(dict(id=sid, name=name, named=named, tracks=tracks))
            finally:
                ec.codes_release(h)
    return out


def _dup(tr, other) -> bool:
    for hh in np.intersect1d(tr["steps"], other["steps"]):
        a = TC.at(tr, int(hh)); b = TC.at(other, int(hh))
        if TC._km(a[0], a[1], b[0], b[1]) < 150:
            return True
    return False


def nh_storms(msgs: list[dict], min_frac: float = 0.4, nmem: int = 51) -> list[dict]:
    """Named NH storms, plus genesis clusters carried by >= min_frac of members at tropical-storm strength. A genesis
    member track that duplicates the same member's track of a named storm (within 150 km at a shared step) is dropped."""
    named = [m for m in msgs if m["named"]]
    keep = []
    for m in msgs:
        tr = {k: t for k, t in m["tracks"].items() if (t["lat"] > 0).any() and len(t["steps"]) > 1}
        if not m["named"]:
            tr = {k: t for k, t in tr.items() if not any(k in n["tracks"] and _dup(t, n["tracks"][k]) for n in named)}
            tr = {k: t for k, t in tr.items() if np.nanmax(np.r_[t["wind"], 0]) >= 17.0}
            if len(tr) < min_frac * nmem:
                continue
        if tr:
            keep.append(dict(m, tracks=tr))
    return keep


def stats(s: dict, nmem: int) -> dict:
    """Per model and storm: recurvature lead per member, P(recurve), lead-time spread, P(track ended by day N)."""
    rec = {k: TC.recurvature(t) for k, t in s["tracks"].items()}
    ended = {k: int(t["steps"][-1]) for k, t in s["tracks"].items()}
    lat_end = {k: float(t["lat"][-1]) for k, t in s["tracks"].items()}
    days = np.array([r / 24 for r in rec.values() if r is not None])
    return dict(n_tracked=len(s["tracks"]), n_members=nmem, recurve=rec, ended=ended, lat_end=lat_end,
                p_recurve=round(float(np.sum([r is not None for r in rec.values()])) / nmem, 3),
                rec_day_q=[round(float(x), 2) for x in np.percentile(days, [10, 50, 90])] if len(days) else None)


# ── step 2: downstream z500 ───────────────────────────────────────────────────────────────────────────────────────
def load_z500(date: str, hh: str, steps_h=STEPS_H):
    """{member: (len(steps), lat, lon)} z500 in metres on a 1-deg grid over SECTOR; member 0 = control."""
    import xarray as xr
    import store as ecmwf
    cyc = ecmwf.Cycle(date, hh); S = tuple(range(0, 241, 12))
    out = {}
    for t in ("cf", "pf"):
        p = ecmwf.ensure(cyc, ecmwf.Spec("aifs-ens", t, "z", "pl", (500,), S))
        ds = xr.open_dataset(p, engine="cfgrib", backend_kwargs={"indexpath": ""})
        z = ds["z"]
        z = z.sel(step=[np.timedelta64(h, "h") for h in steps_h])
        z = z.assign_coords(longitude=z.longitude % 360).sortby("longitude").sortby("latitude")
        z = z.sel(latitude=slice(SECTOR[2] - 1, SECTOR[3] + 1), longitude=slice(SECTOR[0] - 1, SECTOR[1] + 1))
        z = z.coarsen(latitude=4, longitude=4, boundary="trim").mean() / 9.80665
        if "number" in z.dims:
            for k, num in enumerate(z.number.values):
                out[int(num)] = z.isel(number=k).values.astype("float32")
        else:
            out[0] = z.values.astype("float32")
        lat, lon = z.latitude.values, z.longitude.values
    return out, lat, lon


def bh(p: np.ndarray, q: float = 0.10) -> np.ndarray:
    """Benjamini-Hochberg: boolean mask of discoveries at FDR q (NaNs never significant)."""
    flat = p.ravel(); ok = np.isfinite(flat)
    sig = np.zeros(flat.shape, bool)
    pv = flat[ok]
    if not pv.size:
        return sig.reshape(p.shape)
    o = np.argsort(pv); m = pv.size
    thr = q * np.arange(1, m + 1) / m
    below = np.where(pv[o] <= thr)[0]
    if below.size:
        cut = pv[o][below[-1]]
        idx = np.where(ok)[0]
        sig[idx[pv <= cut]] = True
    return sig.reshape(p.shape)


def composite(s_aifs: dict, st: dict, Z: dict):
    """Recurving vs not (or early vs late) members: difference, significance mask, group info; None if too few."""
    from scipy import stats as sst
    rec = {k: v for k, v in st["recurve"].items() if k in Z}
    yes = [k for k, v in rec.items() if v is not None]; no = [k for k, v in rec.items() if v is None]
    if len(yes) >= MIN_GROUP and len(no) >= MIN_GROUP:
        a, b, kind = yes, no, ("recurving", "not recurving")
    elif len(yes) >= 2 * MIN_GROUP:
        med = float(np.median([rec[k] for k in yes]))
        a = [k for k in yes if rec[k] <= med]; b = [k for k in yes if rec[k] > med]
        if len(a) < MIN_GROUP or len(b) < MIN_GROUP:
            return None
        kind = (f"early recurvature (by day {med / 24:.1f})", f"later recurvature")
    else:
        return None
    A = np.stack([Z[k] for k in a]); B = np.stack([Z[k] for k in b])
    t, p = sst.ttest_ind(A, B, axis=0, equal_var=False)
    sig = np.concatenate([bh(p[:NPRE], 0.10), bh(p[NPRE:], 0.10)])   # precursor and downstream days tested as separate families
    out = dict(diff=A.mean(0) - B.mean(0), sig=sig, n=(len(a), len(b)), kind=kind, mean=np.concatenate([A, B]).mean(0))
    # 2026-10-03 (user: does this show what the storm does, or the pattern that makes it recurve?): the raw difference
    # mixes both. Adjusted version: the same group difference at the downstream days, after regressing out the leading
    # member-spread patterns of the DAY-2 field (chosen blind to the groups), i.e. what was already there before the
    # storm met the jet. Still an association, but closer to the storm's own effect.
    keys = a + b
    G = np.r_[np.ones(len(a)), np.zeros(len(b))]
    C, ve = _early_pcs(Z, keys)
    out["adj"] = _partial(G, C, np.stack([Z[k][NPRE:] for k in keys]))
    out["adj"]["ve"] = ve
    return out


def _early_pcs(Z: dict, keys, k=K_PCS):
    """Leading principal components (across members) of the day-2 z500 field over SECTOR, cos-lat weighted; and the
    share of the member variance they explain. Computed without the group labels."""
    X = np.stack([Z[q][ADJ_DAY_IDX].ravel() for q in keys]).astype("float64")
    X = X - X.mean(0)
    w = np.sqrt(np.clip(np.cos(np.deg2rad(_LAT)), 0, None)) if _LAT is not None else None
    if w is not None:
        X = X * np.repeat(w, X.shape[1] // len(w))
    X = np.nan_to_num(X)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    k = min(k, len(keys) - 4)
    return U[:, :k] * S[:k], float(np.sum(S[:k] ** 2) / np.sum(S ** 2))


def _partial(x, C, Y):
    """OLS of Y (members, ...) on [1, x, C] at every point: the coefficient of x, its p-value, BH mask (FDR 10%)."""
    from scipy import stats as sst
    n = len(x)
    X = np.column_stack([np.ones(n), x, C])
    shp = Y.shape[1:]; Ym = Y.reshape(n, -1).astype("float64")
    beta, *_ = np.linalg.lstsq(X, Ym, rcond=None)
    res = Ym - X @ beta
    dof = n - X.shape[1]
    s2 = (res ** 2).sum(0) / dof
    cinv = np.linalg.inv(X.T @ X)[1, 1]
    se = np.sqrt(s2 * cinv) + 1e-12
    t = beta[1] / se
    pv = 2 * sst.t.sf(np.abs(t), dof)
    return dict(coef=beta[1].reshape(shp), sig=bh(pv.reshape(shp), 0.10), dof=dof, k=C.shape[1])


_LAT = None


def sensitivity(s: dict, Z: dict):
    """Regression of z500 on the storm's day-3 latitude and longitude across members, FDR-masked."""
    from scipy import stats as sst
    ks, la, lo = [], [], []
    for k, t in s["tracks"].items():
        pos = TC.at(t, SENS_H)
        if pos is not None and k in Z:
            ks.append(k); la.append(pos[0]); lo.append(pos[1])
    if len(ks) < MIN_SENS:
        return None
    Y = np.stack([Z[k] for k in ks])                                     # (n, steps, lat, lon)
    res = {}
    lo_u = np.rad2deg(np.unwrap(np.deg2rad(lo)))
    for nm, x in (("lat", np.array(la)), ("lon", lo_u)):
        xa = x - x.mean()
        if xa.std() < 0.2:                                                # members agree on the position: nothing to regress on
            res[nm] = None
            continue
        Ya = Y - Y.mean(0)
        C, ve = _early_pcs(Z, ks)
        adj = _partial(xa, C, Y[:, NPRE:]); adj["ve"] = ve                # controlled for the day-2 pattern
        slope = np.tensordot(xa, Ya, axes=(0, 0)) / np.sum(xa ** 2)       # m per degree
        r = np.tensordot(xa, Ya, axes=(0, 0)) / (np.sqrt(np.sum(xa ** 2)) * np.sqrt(np.sum(Ya ** 2, 0)) + 1e-9)
        n = len(ks)
        tt = r * np.sqrt((n - 2) / np.clip(1 - r ** 2, 1e-9, None))
        p = 2 * sst.t.sf(np.abs(tt), n - 2)
        res[nm] = dict(slope=slope, r=r, sig=bh(p, 0.10), sd=float(x.std()), adj=adj)
    return dict(n=len(ks), res=res)


# ── figures ───────────────────────────────────────────────────────────────────────────────────────────────────────
def _stormlab(s):
    return s["name"].title() if s["named"] else f"possible new storm ({s['id']})"


def render_tracks(storms: list[dict], st: dict, init, out_png: Path):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    rows = storms[:4]
    nrow = max(1, len(rows))
    H = 4.6 * nrow + 1.6
    fig = plt.figure(figsize=(16.5, H))
    fig.suptitle(f"Tropical-cyclone track ensembles — AIFS-ENS (51) and IFS-ENS (50), init {init:%d %b %Y %HZ}",
                 fontsize=14, fontweight="bold", x=0.02, ha="left", y=1 - 0.25 / H)
    top, bot = 1 - 0.75 / H, 1.1 / H                                       # figure fractions for the storm rows
    rh = (top - bot) / nrow
    cmap = ListedColormap(["#3b0f70", "#6a2c91", "#9a3f8f", "#c9517f", "#ec6a5b", "#f9944a", "#fbc04b", "#f2e86d"])
    bounds = [0, 2, 4, 6, 8, 10, 12, 14, 16]; norm = BoundaryNorm(bounds, cmap.N)
    if not rows:
        fig.text(0.5, 0.5, "No tropical storms in the Northern Hemisphere in either ensemble", ha="center", fontsize=14, color="#6f6b64")
    for i, s in enumerate(rows):
        allt = [t for m in ("aifs", "ifs") for t in s["models"].get(m, {"tracks": {}})["tracks"].values()]
        la = np.concatenate([t["lat"] for t in allt]); lo = np.concatenate([t["lon"] for t in allt])
        lo360 = lo % 360
        clon = float(np.median(lo360))
        rel = (lo360 - clon + 180) % 360 - 180
        ext = [clon + rel.min() - 4, clon + rel.max() + 4, max(-5, la.min() - 4), min(75, la.max() + 5)]
        for c, m in enumerate(("aifs", "ifs")):
            ax = fig.add_axes([0.035 + c * 0.33, top - (i + 1) * rh + 0.04 * rh, 0.30, 0.86 * rh],
                              projection=ccrs.PlateCarree(central_longitude=clon))
            ax.set_extent(ext, crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#efece4"); ax.coastlines("50m", lw=0.6, color="#555")
            ax.gridlines(draw_labels=dict(bottom="x", left="y"), lw=0.3, color="#999", alpha=0.6, x_inline=False, y_inline=False)
            S = s["models"].get(m)
            if not S:
                ax.text(0.5, 0.5, f"not in {MLAB[m]}", transform=ax.transAxes, ha="center", color="#6f6b64")
                continue
            rec = st[s["key"]][m]["recurve"]
            for k, t in S["tracks"].items():
                r = rec.get(k)
                col = "#9a9a9a" if r is None else cmap(norm(r / 24))
                ax.plot(t["lon"], t["lat"], color=col, lw=2.0 if k == 0 else 0.9, alpha=0.95 if k == 0 else 0.75,
                        transform=ccrs.Geodetic(), zorder=4 if r is not None else 3)
                if r is not None:
                    pos = TC.at(t, r)
                    ax.plot(pos[1], pos[0], marker="o", ms=3.2, color=col, mec="#000", mew=0.3, transform=ccrs.PlateCarree(), zorder=5)
            t0 = next(iter(S["tracks"].values()))
            ax.plot(t0["lon"][0], t0["lat"][0], marker="X", ms=11, color="#000", transform=ccrs.PlateCarree(), zorder=6)
            q = st[s["key"]][m]
            ax.set_title(f"{_stormlab(s)} · {MLAB[m]}: {q['n_tracked']} of {q['n_members']} members tracked, "
                         f"{100 * q['p_recurve']:.0f}% recurve", fontsize=10.5, loc="left")
        ax = fig.add_axes([0.71, top - (i + 1) * rh + 0.12 * rh, 0.27, 0.74 * rh])
        width = 0.42
        for c, (m, colr) in enumerate((("aifs", "#1f5fa8"), ("ifs", "#c0392b"))):
            if m not in s["models"]:
                continue
            q = st[s["key"]][m]
            d = np.array([r / 24 for r in q["recurve"].values() if r is not None])
            h_, _ = np.histogram(d, bins=np.arange(0, 16))
            ax.bar(np.arange(15) + 0.5 + (c - 0.5) * width, 100 * h_ / q["n_members"], width=width, color=colr, alpha=0.85,
                   label=f"{MLAB[m]} recurves ({100 * q['p_recurve']:.0f}%)")
            endd = np.array([e / 24 for e in q["ended"].values()])
            frac = [100 * np.sum(endd < N) / q["n_members"] for N in range(16)]
            ax.plot(range(16), frac, color=colr, lw=1.6, ls="--", label=f"{MLAB[m]} track ended (cumulative)")
        ax.set_xlim(0, 15); ax.set_ylim(0, 100); ax.set_xlabel("forecast day"); ax.set_ylabel("% of members")
        ax.grid(alpha=0.3); ax.legend(fontsize=8.2, loc="upper left", frameon=False)
        ax.set_title("Recurvature day (bars) · track ended (lines)", fontsize=10.5, loc="left")
    cax = fig.add_axes([0.08, 0.55 / H, 0.5, 0.18 / H])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", ticks=bounds)
    cb.set_label("member colour = recurvature day (dot = the turn) · grey = no recurvature · thick line = AIFS control · X = start", fontsize=9)
    fig.savefig(out_png, dpi=90, facecolor="white", pil_kwargs={"quality": 85, "method": 6}); plt.close(fig)


def _row_any(sens, pred) -> bool:
    return sens is not None and sens["res"].get(pred) is not None and bool(sens["res"][pred]["sig"].any())


def _sig_area(comp, sens) -> float:
    a = 0.0 if comp is None else float(np.mean(comp["sig"]))
    if sens is not None:
        a += sum(float(np.mean(r["sig"])) for r in sens["res"].values() if r is not None)
    return a


def render_impact(s: dict, comp, sens, Z_lat, Z_lon, init, out_png: Path, note: str | None = None):
    """Rows (2026-10-03, user: "does this chart do as you claim or just show differences between the pattern that leads to a
    recurve and the one that does not"): the raw member-group difference MIXES the steering pattern that makes the storm
    recurve with what the storm then does. So: (1) the same difference BEFORE any interaction (days 0 and 2): what was
    already there; (2) the raw difference at days 5/7/10, labelled as mixed; (3) the difference after regressing out the
    leading member-spread patterns of the day-2 field: closer to the storm's own effect; (4) the position sensitivity,
    controlled the same way. Rows with nothing (or only specks) significant collapse to a one-line note."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    W = 17.0
    ttl = (f"How the flow differs when {_stormlab(s)} recurves — AIFS-ENS 500 hPa height, init {init:%d %b %Y %HZ}" if s is not None
           else f"Tropical cyclones and the downstream flow — AIFS-ENS, init {init:%d %b %Y %HZ}")
    if s is None or (comp is None and sens is None):
        fig = plt.figure(figsize=(W, 3.0))
        fig.suptitle(ttl, fontsize=15, fontweight="bold", x=0.02, ha="left", y=0.95)
        fig.text(0.5, 0.45, note or "No storm has enough members in each group for a test.", ha="center", fontsize=13, color="#6f6b64")
        fig.savefig(out_png, dpi=80, facecolor="white", pil_kwargs={"quality": 85, "method": 6}); plt.close(fig)
        return
    big = lambda sig: max(float(np.mean(x)) for x in sig) >= MIN_AREA
    rr = [t for t in (TC.recurvature(tr) for tr in s["models"]["aifs"]["tracks"].values()) if t is not None]
    rec_med = float(np.median(rr)) / 24 if rr else None
    rows = []   # (heading, days, fields[(value, sig)], contours or None, cmap, levels, cbar label) or (note,)
    if comp is None:
        rows.append(("Recurving vs other members: too few members in one group for a test (needs 8 in each).",))
    else:
        g = f"{comp['kind'][0]} minus {comp['kind'][1]} ({comp['n'][0]} vs {comp['n'][1]} members)"
        lev = np.arange(-160, 161, 20)
        pre = [(comp["diff"][i], comp["sig"][i]) for i in range(NPRE)]
        if big([x[1] for x in pre]):
            rows.append((f"1 · Before the storm reaches the jet (days {PRE_DAYS[0]} and {PRE_DAYS[1]}): {g[0].upper() + g[1:]}. "
                         "Differences here were already in place, so they are a CAUSE of recurvature, not an effect.",
                         PRE_DAYS, pre, [comp["mean"][i] for i in range(NPRE)], "RdBu_r", lev, "height difference (m) · contours: ensemble-mean 500 hPa height"))
        else:
            rows.append((f"1 · Before the storm reaches the jet (days {PRE_DAYS[0]} and {PRE_DAYS[1]}): no significant difference between the groups "
                         "point by point (small early differences can still grow — row 3 accounts for them).",))
        post = [(comp["diff"][NPRE + i], comp["sig"][NPRE + i]) for i in range(len(DAYS))]
        if big([x[1] for x in post]):
            rows.append((f"2 · Raw difference, {g}: cause and effect MIXED (the pattern that steers the storm plus what the storm does)",
                         DAYS, post, [comp["mean"][NPRE + i] for i in range(len(DAYS))], "RdBu_r", lev,
                         "height difference (m) · contours: ensemble-mean 500 hPa height"))
        else:
            rows.append((f"2 · Raw difference, {g}: nothing significant at days 5, 7 or 10.",))
        A = comp["adj"]; adj = [(A["coef"][i], A["sig"][i]) for i in range(len(DAYS))]
        if big([x[1] for x in adj]):
            rows.append((f"3 · The same difference after removing what the day-2 pattern predicts ({A['k']} leading member patterns, "
                         f"{100 * A['ve']:.0f}% of the day-2 spread): closer to the storm's own downstream effect",
                         DAYS, adj, None, "RdBu_r", lev, "height difference not explained by the day-2 pattern (m)"))
        else:
            rows.append((f"3 · After removing what the day-2 pattern predicts ({100 * A['ve']:.0f}% of its member spread): nothing significant "
                         "remains — the raw difference above is largely what was already in place at day 2 (the adjusted test also has less power).",))
    for pred, word in (("lat", "latitude"), ("lon", "longitude")):
        R = None if sens is None else sens["res"].get(pred)
        if sens is None:
            rows.append((f"Storm's day-3 {word}: too few members tracked at day 3 for a test.",)); continue
        if R is None:
            rows.append((f"Storm's day-3 {word}: the members agree on it, nothing to test.",)); continue
        A = R["adj"]; fl = [(A["coef"][i], A["sig"][i]) for i in range(len(DAYS))]
        what = "1° further north" if pred == "lat" else "1° further east"
        if big([x[1] for x in fl]):
            rows.append((f"4 · If the storm is {what} at day 3, with the day-2 pattern held fixed: change in 500 hPa height ({sens['n']} members)",
                         DAYS, fl, None, "PuOr_r", np.arange(-40, 41, 5),
                         f"metres per degree (members' day-3 {word} spreads {R['sd']:.1f}°)"))
        else:
            rows.append((f"Storm's day-3 {word}, day-2 pattern held fixed: no significant link to 500 hPa height at days 5, 7 or 10.",))
    mw = (W - 0.9) / 3.0
    mh = mw * (SECTOR[3] - SECTOR[2]) / (SECTOR[1] - SECTOR[0]) * 1.25
    ROW_ON, ROW_OFF, TOP, BOT = 0.5 + 0.3 + mh + 0.75, 0.42, 1.15, 0.85
    H = TOP + BOT + sum(ROW_ON if len(r) > 1 else ROW_OFF for r in rows)
    fig = plt.figure(figsize=(W, H))
    fx = lambda x: x / W
    fy = lambda y: 1 - y / H
    fig.suptitle(ttl, fontsize=15, fontweight="bold", x=0.02, ha="left", y=fy(0.25), va="top")
    fig.text(0.02, fy(0.58), "The members where the storm recurves differ partly BECAUSE a different pattern steers them, and partly because of "
             "what the storm then does. Row 1 shows the first, row 3 tries to remove it.\nShaded only where significant (Benjamini–Hochberg "
             "false-discovery rate 10%). Thin lines: the AIFS-ENS members' tracks.", fontsize=10.5, color="#444", va="top")
    proj = ccrs.PlateCarree(central_longitude=200)
    y = TOP
    for r in rows:
        if len(r) == 1:
            fig.text(0.02, fy(y + 0.12), "• " + r[0], fontsize=11.5, color="#6f6b64", va="top")
            y += ROW_OFF
            continue
        head, days, fields, conts, cmap, lev, cblab = r
        fig.text(0.02, fy(y + 0.05), head, fontsize=12, fontweight="bold", va="top", wrap=True)
        cf = None
        for c, (d, (val, sg)) in enumerate(zip(days, fields)):
            x0 = 0.3 + c * (mw + 0.15)
            ax = fig.add_axes([fx(x0), fy(y + 0.8 + mh), fx(mw), mh / H], projection=proj)
            ax.set_extent([SECTOR[0], SECTOR[1], SECTOR[2], SECTOR[3]], crs=ccrs.PlateCarree()); ax.set_aspect("auto")
            ax.coastlines("50m", lw=0.6, color="#333"); ax.add_feature(cfeature.BORDERS, lw=0.3, edgecolor="#666")
            cf = ax.contourf(Z_lon, Z_lat, np.where(sg, val, np.nan), levels=lev, cmap=cmap, extend="both", transform=ccrs.PlateCarree())
            if conts is not None:
                ax.contour(Z_lon, Z_lat, conts[c], levels=np.arange(5040, 6001, 60), colors="#222", linewidths=0.6, transform=ccrs.PlateCarree())
            ax.set_title(f"Day {d} · {100 * np.mean(sg):.0f}% of the map significant", fontsize=11, loc="left", pad=4)
            for tr in s["models"]["aifs"]["tracks"].values():
                ax.plot(tr["lon"], tr["lat"], color="#000", lw=0.35, alpha=0.35, transform=ccrs.Geodetic())
        cax = fig.add_axes([fx(W * 0.25), fy(y + 0.8 + mh + 0.32), fx(W * 0.5), 0.12 / H])
        cb = fig.colorbar(cf, cax=cax, orientation="horizontal"); cb.ax.tick_params(labelsize=9.5)
        cb.set_label(cblab, fontsize=10)
        y += ROW_ON
    fig.text(0.02, fy(H - 0.2), "Tracks: ECMWF tropical-cyclone tracker on AIFS-ENS (open data, CC BY 4.0). Heights: AIFS-ENS control + 50 members, "
             "1° means. Rows 1-2: Welch t-test. Rows 3-4: least squares with the leading day-2 member patterns as controls.\n"
             "Sensitivity after Torn & Hakim (2008, MWR). All rows are associations across members; only a run with the storm removed would "
             "isolate its effect." + (f"\nThis storm recurves around day {rec_med:.1f}: by day 2 it may already be shaping the flow, so rows 3-4 can "
             "remove part of its own effect too." if rec_med is not None and rec_med <= 3.5 else ""), fontsize=9, color="#6f6b64", va="bottom")
    fig.savefig(out_png, dpi=80, facecolor="white", pil_kwargs={"quality": 85, "method": 6}); plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True); ap.add_argument("--time", default="00")
    ap.add_argument("--out-dir", default="assets/sst")
    ap.add_argument("--no-z500", action="store_true")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import pandas as pd
    t0 = time.time()
    init = pd.Timestamp(f"{a.date}T{a.time}:00")
    out = Path(a.out_dir); (out / "data").mkdir(parents=True, exist_ok=True)
    nm = {"aifs": 51, "ifs": 50}
    per = {}
    for m in ("aifs", "ifs"):
        try:
            fp = fetch(m, a.date, int(a.time))
            per[m] = nh_storms(decode_all(fp), nmem=nm[m]) if fp else []
        except Exception as e:
            print(f"  {MLAB[m]} tracks failed ({e})", flush=True); per[m] = []
    # merge the two models: named storms by id; genesis clusters stay per model
    storms = {}
    def med_pos(s):
        """median member position at the earliest step most members share (for matching genesis clusters)"""
        hs = np.concatenate([t["steps"] for t in s["tracks"].values()])
        vals, cnt = np.unique(hs, return_counts=True)
        h = int(vals[np.argmax(cnt)])
        ps = [TC.at(t, h) for t in s["tracks"].values() if TC.at(t, h) is not None]
        return h, float(np.median([q[0] for q in ps])), float(np.median([q[1] for q in ps]))
    gen = {m: {s["id"]: s for s in lst if not s["named"]} for m, lst in per.items()}
    shared = set()
    for gid in set(gen.get("aifs", {})) & set(gen.get("ifs", {})):
        ha, la_, lo_ = med_pos(gen["aifs"][gid]); hi, lb, lob = med_pos(gen["ifs"][gid])
        if abs(ha - hi) <= 48 and TC._km(la_, lo_, lb, lob) < 500:
            shared.add(gid)                                              # same tracker id, same place: one system
    for m, lst in per.items():
        for s in lst:
            key = s["id"] if (s["named"] or s["id"] in shared) else f"{m}:{s['id']}"
            storms.setdefault(key, dict(key=key, id=s["id"], name=s["name"], named=s["named"], models={}))["models"][m] = s
    st = {k: {m: stats(s["models"][m], nm[m]) for m in s["models"]} for k, s in storms.items()}
    order = sorted(storms.values(), key=lambda s: (not s["named"], -max(st[s["key"]][m]["p_recurve"] for m in s["models"])))
    print(f"  storms: " + ", ".join(f"{_stormlab(s)} " + "/".join(f"{MLAB[m]} {100 * st[s['key']][m]['p_recurve']:.0f}%" for m in s["models"])
                                    for s in order), flush=True)
    render_tracks(order, st, init, out / "tcens_tracks.webp")
    # step 2
    impact = {}
    pick = None
    tested = []
    if not a.no_z500:
        cand = [s for s in order if "aifs" in s["models"] and s["named"]]
        Z = lat = lon = None
        if cand:
            try:
                Z, lat, lon = load_z500(a.date, a.time)
                globals()["_LAT"] = np.repeat(lat, 1)
            except Exception as e:
                print(f"  z500 unavailable ({e})", flush=True)
        if Z is not None:
            for s in cand:
                comp = composite(s["models"]["aifs"], st[s["key"]]["aifs"], Z)
                sens = sensitivity(s["models"]["aifs"], Z)
                impact[s["key"]] = (comp, sens)
            # every storm is tested on its own; the two with the largest significant area are drawn
            tested = [s for s in cand if impact[s["key"]][0] is not None or impact[s["key"]][1] is not None]
            tested.sort(key=lambda s: -_sig_area(*impact[s["key"]]))
            pick = tested[0] if tested else None
            for n_, fn in enumerate(("tcens_impact.webp", "tcens_impact_2.webp")):
                if n_ < len(tested):
                    render_impact(tested[n_], *impact[tested[n_]["key"]], lat, lon, init, out / fn)
                else:
                    render_impact(None, None, None, lat, lon, init, out / fn,
                                  note=("No other storm" if n_ else "No storm") + " has at least 8 AIFS-ENS members in each group "
                                       "(recurving vs not, or early vs late) or 20 members tracked at day 3, so nothing else is tested this cycle.")
        else:
            for fn in ("tcens_impact.webp", "tcens_impact_2.webp"):
                render_impact(None, None, None, None, None, init, out / fn,
                              note="No named Northern-Hemisphere storm in AIFS-ENS this cycle.")
    js = {"init": f"{init:%Y-%m-%dT%H:%MZ}", "models": {m: {"members": nm[m]} for m in nm}, "impact_storms": [x["key"] for x in tested[:2]],
          "storms": []}
    for s in order:
        e = {"key": s["key"], "id": s["id"], "name": _stormlab(s), "named": s["named"], "models": {}}
        for m in s["models"]:
            q = st[s["key"]][m]
            e["models"][m] = {"tracked": q["n_tracked"], "members": q["n_members"], "p_recurve": q["p_recurve"],
                              "recurve_day_p10_p50_p90": q["rec_day_q"],
                              "p_track_ended_by_day": {str(N): round(float(np.mean([v < N * 24 for v in q["ended"].values()])) * q["n_tracked"] / q["n_members"], 3)
                                                       for N in (3, 5, 7, 10)}}
        if s["key"] in impact:
            comp, sens = impact[s["key"]]
            e["composite"] = None if comp is None else {"groups": comp["kind"], "n": comp["n"],
                                                         "sig_area_pct_by_day": {str(d): round(100 * float(np.mean(comp["sig"][NPRE + i])), 1) for i, d in enumerate(DAYS)},
                                                         "precursor_sig_area_pct_by_day": {str(d): round(100 * float(np.mean(comp["sig"][i])), 1) for i, d in enumerate(PRE_DAYS)},
                                                         "adjusted_sig_area_pct_by_day": {str(d): round(100 * float(np.mean(comp["adj"]["sig"][i])), 1) for i, d in enumerate(DAYS)},
                                                         "adjusted_controls": {"day": PRE_DAYS[ADJ_DAY_IDX], "pcs": comp["adj"]["k"], "var_explained": round(comp["adj"]["ve"], 3)}}
            e["sensitivity"] = None if sens is None else {"n": sens["n"], **{p: None if sens["res"][p] is None else
                                                          {"sd_deg": round(sens["res"][p]["sd"], 2),
                                                           "sig_area_pct_by_day_adjusted": {str(d): round(100 * float(np.mean(sens["res"][p]["adj"]["sig"][i])), 1) for i, d in enumerate(DAYS)}}
                                                          for p in ("lat", "lon")}}
        js["storms"].append(e)
    (out / "data" / "tcens.json").write_text(json.dumps(js))
    print(f"  wrote tcens_tracks.webp, tcens_impact.webp, data/tcens.json in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
