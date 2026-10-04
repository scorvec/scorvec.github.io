#!/usr/bin/env python3
"""Reference files for gefs_mjo.py - LAPTOP, once (rebuild `ref` when the reforecast climatology changes).

  ref       -> gefs_mjo_ref.npz: the WH04 observed climatology, normalisers and EOFs (from the site's
               scripts/mjo/data/reference climatology.nc and eofs.nc - read here with xarray, so the daily job needs
               neither xarray nor those files), and the GEFS reforecast band climatology (3 fields x 35 days x 240
               longitudes per 5-day centre, int16 as gefs_reforecast.pack) from release gefs-clim-v1 `band`, or from
               the 15S-15N rows of v2's daily fields (--clim v2). With both on disk it prints how far they differ.
  hindcast  -> gefs_mjo_hindcast.npz: every GEFSv12 reforecast start 2000-2019 (Wednesdays, 4 members; the per-start
               files ~/data_archive/gefs_rf/rf-YYYY/rf_YYYYMMDD.npz) projected with EXACTLY gefs_mjo.py's arithmetic,
               per member, and BoM's RMM on each valid date. The daily job only rescores these (seconds).

The hindcast, step by step (gefs_mjo.forecast_channels for one start):
  model climatology  per start, the reforecast band climatology at the start's day of year (linear between 5-day
                     centres, as live) but LEAVING OUT every start within 60 days of it - its own forecast and its
                     neighbours', which share its MJO event. The live climatology is all 20 years, which is fine for
                     2026 and would flatter a 2000-2019 score.
  frame shift        + M(day 1; valid doy) - O(valid doy), with the same leave-out
  120-day filter     live, the mean of the 120 GEFS analysis days before the init. The reforecast has no analysis,
                     and the 35-day runs start weekly, so it is the mean of the day-1 channels of the starts in the 120
                     days before (17 of them; at least 15 required) - GEFS's own shortest-range state, the analogue
                     of the live filter and GEPS's choice (mjo_hindcast.py, 4 starts a week there). Weekly sampling of
                     a 120-day mean: noise ~ sigma/sqrt(17) of the non-MJO part; the day-to-day score barely feels a
                     slowly varying term.
  truth              BoM RMM (WH04 method with NCEP winds to 2013, Gottschalck 2010 with ACCESS winds from 2014; BoM
                     also removed the ENSO-SST regression to 2013), on the valid date init + (d - 1).

    python build_gefs_mjo.py ref [--clim v1|v2] [--clim-dir DIR]
    python build_gefs_mjo.py hindcast [--rf DIR [DIR ...]]    # default ~/data_archive/gefs_rf
    python build_gefs_mjo.py plot --doy-now 268       # the skill figure + json into --site, for a look
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gefs_reforecast as R                                           # noqa: E402
import gefs_daily as G                                               # noqa: E402
import gefs_mjo as GM                                                # noqa: E402

MJOREF = HERE.parent / "mjo" / "data" / "reference"
BOM = Path.home() / "data_archive" / "geps_subx" / "telecon" / "bom" / "rmm.74toRealtime.txt"
RF = Path.home() / "data_archive" / "gefs_rf"
OUT = Path.home() / "data_archive" / "gefs_ref" / "mjo"
TAGS = GM.TAGS
CENTRES = list(range(1, 366, 5))
EXCLUDE_DAYS = 60


def band_v1(clim_dir: Path, c: int):
    return np.load(clim_dir / f"gefs_clim_{c:03d}.npz")["band"].astype("float64")


def band_v2(clim_dir: Path, c: int):
    z = np.load(clim_dir / f"gefs_clim2_{c:03d}.npz")
    return np.stack([G.unpack(t, z[f"d_{t}"])[:, R.BAND, :].mean(1) for t in TAGS])


def cmd_ref(a) -> int:
    import xarray as xr
    cl = xr.open_dataset(MJOREF / "climatology.nc")
    e = xr.open_dataset(MJOREF / "eofs.nc")
    lon = cl.longitude.values.astype(float)
    if not (np.allclose(lon, GM.LON25) and np.allclose(e.longitude.values.astype(float), GM.LON25)):
        raise SystemExit("WH04 reference longitudes are not 0..357.5 by 2.5")
    O = np.stack([cl[f"clim_{t}"].transpose("dayofyear", "longitude").values for t in TAGS]).astype("float32")
    if O.shape != (3, 366, 144):
        raise SystemExit(f"unexpected climatology shape {O.shape}")
    clim_dir = Path(a.clim_dir)
    load = band_v1 if a.clim == "v1" else band_v2
    mc = np.stack([load(clim_dir, c) for c in CENTRES])                         # (73, 3, 35, 240)
    other = band_v2 if a.clim == "v1" else band_v1
    try:
        oc = np.stack([other(clim_dir, c) for c in CENTRES])
        d = mc - oc
        for i, t in enumerate(TAGS):
            print(f"  v1 vs v2 {t}: mean {d[:, i].mean():+.3f}, rms days 1-10 {np.sqrt((d[:, i, :10] ** 2).mean()):.3f}, "
                  f"days 11-35 {np.sqrt((d[:, i, 10:] ** 2).mean()):.3f}")
    except FileNotFoundError:
        print(f"  (no {'v2' if a.clim == 'v1' else 'v1'} files in {clim_dir}: no comparison)")
    q = np.stack([R.pack(t, mc[:, i]) for i, t in enumerate(TAGS)], 1)             # (73, 3, 35, 240) int16
    back = np.stack([R.unpack(t, q[:, i]) for i, t in enumerate(TAGS)], 1)
    print(f"  packing error max {np.abs(back - mc).max():.4f}")
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / GM.REF_FILE
    np.savez_compressed(
        p, O=O, std=np.array([cl.attrs[f"std_{t}"] for t in TAGS], "float64"),
        eof_olr=e.eof_olr.sel(mode=[1, 2]).values, eof_u850=e.eof_u850.sel(mode=[1, 2]).values,
        eof_u200=e.eof_u200.sel(mode=[1, 2]).values, pc_std=e.pc_std.sel(mode=[1, 2]).values,
        centres=np.array(CENTRES), mclim=q, tags=np.array(TAGS),
        mclim_source=np.array(f"gefs-clim-{a.clim} band (15S-15N rows, 1.5 deg), int16 as gefs_reforecast.pack"),
        wh04_source=np.array(f"scripts/mjo/data/reference climatology.nc ({cl.attrs.get('base_period', '?')}, "
                             f"{cl.attrs.get('n_harmonics', '?')} harmonics) and eofs.nc"))
    print(f"wrote {p} ({p.stat().st_size / 1e6:.2f} MB), climatology {a.clim}")
    return 0


# ── hindcast ──────────────────────────────────────────────────────────────────────────────────────────────────
def bom_rmm():
    """BoM RMM1/RMM2 by date (missing values skipped)."""
    out = {}
    for ln in BOM.read_text().splitlines():
        p = ln.split()
        if len(p) < 7 or not p[0].isdigit():
            continue
        r1, r2 = float(p[3]), float(p[4])
        if abs(r1) > 900 or abs(r2) > 900:
            continue
        out[dt.date(int(p[0]), int(p[1]), int(p[2]))] = (r1, r2)
    return out


def load_rf(roots):
    """Every rf_YYYYMMDD.npz under any of the roots (the first copy of a start wins)."""
    seen = {}
    for r in roots:
        for f in sorted(Path(r).expanduser().rglob("rf_*.npz")):
            seen.setdefault(f.name, f)
    files = [seen[k] for k in sorted(seen)]
    starts, bands, nm = [], [], []
    for f in files:
        z = np.load(f)
        b = z["band"]                                                          # (m, 3, 35, 240) int16
        x = np.full((4, 3, 35, len(R.LON)), np.nan, "float32")
        for i, t in enumerate(TAGS):
            x[: b.shape[0], i] = R.unpack(t, b[:, i])
        starts.append(dt.date(int(f.stem[3:7]), int(f.stem[7:9]), int(f.stem[9:11])))
        bands.append(x); nm.append(b.shape[0])
    return starts, np.stack(bands), np.array(nm)


def cmd_hindcast(a) -> int:
    t0 = time.time()
    ref = GM.load_ref([str(OUT)])
    starts, B, nm = load_rf(a.rf)
    S = len(starts)
    years = sorted({s.year for s in starts})
    print(f"{S} starts {starts[0]} -> {starts[-1]}, years {years[0]}-{years[-1]} ({len(years)}), members "
          f"{np.bincount(nm).tolist()}  ({time.time() - t0:.0f} s)", flush=True)
    doy = np.array([GM.yday(s) for s in starts])
    tnum = np.array([s.toordinal() for s in starts])
    msum = np.nansum(B, axis=1)                                                 # (S, 3, 35, 240) member sums
    # per-centre sums over the starts within +-15 days of day of year (gefs_reforecast.cmd_combine's window)
    win = []
    Sc = np.zeros((len(CENTRES), 3, 35, len(R.LON))); Nc = np.zeros(len(CENTRES))
    for k, c in enumerate(CENTRES):
        dd = np.abs(doy - c); dd = np.minimum(dd, 365 - dd)
        w = np.where(dd <= 15)[0]
        win.append(w)
        Sc[k] = msum[w].sum(0); Nc[k] = nm[w].sum()
    # check against the published climatology (the ref's mclim), all starts in
    full = np.stack([np.stack([R.unpack(t, ref["mclim"][k, i]) for i, t in enumerate(TAGS)]) for k in range(len(CENTRES))])
    dmax = np.abs(Sc / Nc[:, None, None, None] - full).max()
    print(f"  full-sample band climatology vs the reference file: max |diff| {dmax:.4f} "
          f"({'OK' if dmax < 0.02 else 'MISMATCH'})", flush=True)

    def clim_excl(i, the_doy, lead=None):
        """Band climatology at a day of year, leaving out starts within EXCLUDE_DAYS of start i."""
        out = 0.0
        for c, wt in GM.centre_weights(the_doy):
            k = CENTRES.index(c)
            drop = win[k][np.abs(tnum[win[k]] - tnum[i]) <= EXCLUDE_DAYS]
            n = Nc[k] - nm[drop].sum()
            if lead is None:
                s = Sc[k] - msum[drop].sum(0)
            else:
                s = Sc[k][:, lead] - msum[drop][:, :, lead].sum(0)
            out = out + wt * s / n
        return out

    std = ref["std"][None, :, None, None]
    day1 = np.full((S, 3, len(GM.LON25)), np.nan)                               # ensemble-mean day-1 channel (anomaly units)
    CH = np.full((S, 4, 35, 3, len(GM.LON25)), np.nan, "float32")               # before the filter map, anomaly units
    CH_noshift = np.full((S, 35, 3, len(GM.LON25)), np.nan, "float32")          # ensemble mean, F - M only (diagnostic)
    for i, s0 in enumerate(starts):
        M = clim_excl(i, doy[i])                                                # (3, 35, 240)
        shift = np.empty((3, 35, len(GM.LON25)))
        for k in range(35):
            v = s0 + dt.timedelta(days=k)
            shift[:, k] = GM.to25(clim_excl(i, GM.yday(v), lead=0)) - GM.obs_clim(ref, GM.yday(v))
        an = GM.to25(B[i, :nm[i]].astype("float64") - M[None])                  # (m, 3, 35, 144)
        ch = an + shift[None]
        CH[i, :nm[i]] = np.moveaxis(ch, 2, 1)
        day1[i] = ch[:, :, 0].mean(0)
        CH_noshift[i] = np.moveaxis(an.mean(0), 1, 0)
        if i % 200 == 0:
            print(f"  channels {i}/{S} ({time.time() - t0:.0f} s)", flush=True)
    # 120-day filter from the day-1 channels of the starts in the 120 days before; + the same for F - M alone
    maps = np.full((S, 3, len(GM.LON25)), np.nan); maps_ns = np.full_like(maps, np.nan)
    d1_ns = CH_noshift[:, 0]
    for i in range(S):
        sel = (tnum >= tnum[i] - GM.MAP_DAYS) & (tnum < tnum[i])
        if sel.sum() >= 15:
            maps[i] = day1[sel].mean(0); maps_ns[i] = d1_ns[sel].mean(0)
    fch = (CH - maps[:, None, None, :, :]) / std.reshape(1, 1, 1, 3, 1)
    fm1, fm2 = GM.channels_rmm(fch, ref)                                        # (S, 4, 35)
    ns1, ns2 = GM.channels_rmm((CH_noshift - maps_ns[:, None]) / ref["std"][None, None, :, None], ref)
    obs = bom_rmm()
    o1 = np.full((S, 35), np.nan); o2 = np.full_like(o1, np.nan)
    o1p = np.full_like(o1, np.nan); o2p = np.full_like(o1, np.nan)             # one day later (GEPS's convention)
    for i, s0 in enumerate(starts):
        for k in range(35):
            v = obs.get(s0 + dt.timedelta(days=k)); w = obs.get(s0 + dt.timedelta(days=k + 1))
            if v:
                o1[i, k], o2[i, k] = v
            if w:
                o1p[i, k], o2p[i, k] = w
    a0 = np.array([np.hypot(*obs[s]) if s in obs else np.nan for s in starts])
    with np.errstate(invalid="ignore"):                                      # starts with no filter map are all-NaN
        n_ok = np.isfinite(fm1).sum(1)
        f1 = np.where(n_ok > 0, np.nansum(fm1, 1) / np.maximum(n_ok, 1), np.nan)
        f2 = np.where(n_ok > 0, np.nansum(fm2, 1) / np.maximum(n_ok, 1), np.nan)
    ok = np.isfinite(f1).all(1)
    c_ok = GM._curves(f1[ok], f2[ok], o1[ok], o2[ok])["cor"]
    c_late = GM._curves(f1[ok], f2[ok], o1p[ok], o2p[ok])["cor"]
    c_ns = GM._curves(ns1[ok], ns2[ok], o1[ok], o2[ok])
    c_sh = GM._curves(f1[ok], f2[ok], o1[ok], o2[ok])
    print(f"  valid-date check, COR day 1/2/3: valid = init+(d-1) {c_ok[0]:.3f}/{c_ok[1]:.3f}/{c_ok[2]:.3f};  "
          f"init+d {c_late[0]:.3f}/{c_late[1]:.3f}/{c_late[2]:.3f}", flush=True)
    print("  frame check (COR / RMSE at days 1, 5, 10, 20): with the M(1)-O shift "
          + " ".join(f"{c_sh['cor'][d]:.3f}/{c_sh['rmse'][d]:.3f}" for d in (0, 4, 9, 19)) + ";  F-M only "
          + " ".join(f"{c_ns['cor'][d]:.3f}/{c_ns['rmse'][d]:.3f}" for d in (0, 4, 9, 19)), flush=True)
    p = OUT / GM.HIND_FILE
    np.savez_compressed(p, starts=np.array([s.isoformat() for s in starts]), fm1=fm1.astype("float32"),
                        fm2=fm2.astype("float32"), o1=o1.astype("float32"), o2=o2.astype("float32"),
                        amp0=a0.astype("float32"), n_members=nm,
                        note=np.array(f"GEFSv12 reforecast {years[0]}-{years[-1]}, Wednesday starts, members "
                                      "c00+p01-p03, projected as gefs_mjo.py (leave-60-day-out model climatology, "
                                      "120-day filter from the day-1 channels of the preceding starts); obs = BoM RMM "
                                      "on init + (d - 1)"))
    print(f"wrote {p} ({p.stat().st_size / 1e6:.2f} MB), {int(ok.sum())} scored starts ({time.time() - t0:.0f} s)")
    return 0


def cmd_plot(a) -> int:
    site = Path(a.site)
    GM.skill(OUT / GM.HIND_FILE, a.doy_now, site / "assets" / "gefs", site / "assets" / "gefs" / "data", a.members)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("ref"); r.add_argument("--clim", choices=("v1", "v2"), default="v1")
    r.add_argument("--clim-dir", default=str(Path.home() / "data_archive" / "gefs_port_test" / "mjo" / "cache"))
    h = sp.add_parser("hindcast"); h.add_argument("--rf", nargs="+", default=[str(RF)])
    p = sp.add_parser("plot"); p.add_argument("--doy-now", type=int, required=True)
    p.add_argument("--site", default=str(Path.home() / "data_archive" / "gefs_port_test" / "mjo" / "site"))
    p.add_argument("--members", type=int, default=31)
    a = ap.parse_args()
    return {"ref": cmd_ref, "hindcast": cmd_hindcast, "plot": cmd_plot}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
