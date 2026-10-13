#!/usr/bin/env python3
"""ERA5 daily AAM budget history, 1959 onward - input for the El Nino mountain-torque study
(torque_enso_study.py). Runs in Actions (.github/workflows/torque-history.yml), one job per
chunk of years, each streaming ARCO-ERA5 anonymously and reducing every field on the fly to a
few dozen numbers per day.

Definitions are the site's (build_torque_clim.py, build_torque_basis.py, torque_map_anim.py),
torque ON the atmosphere, eastward = adds westerly AAM:
  friction  T_f = -sum a^3 cos^2(phi) tau_turb dlam dphi   (mean_eastward_turbulent_surface_stress)
  GWD       T_g = -sum a^3 cos^2(phi) tau_gwd  dlam dphi   (mean_eastward_gravity_wave_surface_stress)
  mountain  T_m = +sum a^2 cos(phi) h dp_s/dlam dlam dphi  (periodic in longitude; = -sum p_s dh/dlam)
  AAM (rel) M   = +sum a^3 cos^2(phi) F_u dlam dphi        (F_u = vertical_integral_of_eastward_mass_flux
                                                            = int u dp/g over the full model column)
h is scripts/mjo/data/reference/era5_orography.nc, which holds orography HEIGHT IN METRES despite
its geopotential-style name - it is NOT divided by g.

Sampling: 00, 06, 12 and 18 UTC. 00Z+12Z alone would cancel the diurnal (S1) pressure tide but
not the semidiurnal one (S2 has the same phase at 00Z and 12Z), so its alias would stay in the
daily mean; four synoptic hours remove both. The daily value is the mean of the four.

Regions: global and six latitude bands (rows exactly on a band edge split half to each side, so
the bands add to the global total). Ranges: the site's RANGES boxes. `mtn5_<R>` reproduces
torque_map_anim.box_net5 exactly (5-degree coarsened density summed over block centres in the
box); `mtn_<R>`, `fric_<R>`, `gwd_<R>` are exact 0.25-degree sums over the same 5-degree-aligned
cells. `mtn1_*` and `mtn25_*` recompute the mountain torque after block-averaging BOTH h and p_s
to 1 and 2.5 degrees - the resolution-sensitivity check for the CMIP6 comparison (CMIP6 models
carry their own ~1-2 degree orography).

  AAM mass term   M_m = Omega a^4/g sum p_s cos^3(phi) dlam dphi     (`aamm_*`)

Units: torques in Hadley (1e18 N m); AAM in 1e25 kg m^2 s^-1. `--per-time` writes one row per sampled
time (hourly 2026 for the high-fidelity budget) instead of daily means.

    python scripts/mjo/src/torque_history_era5.py --start 1979-01-01 --end 1980-12-31 --out chunk.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import gcsfs

REF = Path(__file__).resolve().parent.parent / "data" / "reference"
OROG = REF / "era5_orography.nc"
STORE = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
A = 6.371e6
G0 = 9.80665
OMEGA = 7.292115e-5
HU = 1e18
MU = 1e25
DEG = np.pi / 180.0
FIELDS = {"sp": "surface_pressure", "tt": "mean_eastward_turbulent_surface_stress",
          "tg": "mean_eastward_gravity_wave_surface_stress", "mf": "vertical_integral_of_eastward_mass_flux"}
# the site's boxes (torque_map_anim.RANGES), lon0, lon1, lat0, lat1 on 0..360E
RANGES = {"HIM": (70, 105, 25, 45), "ROC": (232, 258, 30, 62), "AND": (282, 296, -56, 12),
          "GRL": (300, 345, 58, 84), "ANT": (0, 360, -90, -66), "ALP": (0, 45, 36, 50)}
BANDS = {"glob": (-90, 90), "b9060s": (-90, -60), "b6030s": (-60, -30), "b3000s": (-30, 0),
         "b0030n": (0, 30), "b3060n": (30, 60), "b6090n": (60, 90)}


def band_weights(lat):
    """(nband, nlat) row weights; a row exactly on an edge counts half to each side."""
    W = np.zeros((len(BANDS), len(lat)))
    for i, (nm, (s, n)) in enumerate(BANDS.items()):
        w = ((lat > s) & (lat < n)).astype(float)
        w[np.isclose(lat, s)] = 0.5 if s > -90 else 1.0
        w[np.isclose(lat, n)] = 0.5 if n < 90 else 1.0
        W[i] = w
    return W


class Grid:
    def __init__(self):
        o = xr.open_dataarray(OROG)
        self.lat = o.latitude.values.astype(np.float64)
        self.lon = o.longitude.values.astype(np.float64)
        self.h = o.values.astype(np.float64)                      # metres (see docstring)
        assert 4000 < self.h.max() < 7000, "orography is not in metres any more"
        self.dl = 0.25 * DEG
        cos = np.cos(np.deg2rad(self.lat))
        self.cellA = (A ** 2) * cos[:, None] * self.dl * self.dl  # m^2
        self.cos2A3 = (A ** 3) * (cos ** 2)[:, None] * self.dl * self.dl
        self.cos3mass = OMEGA * (A ** 4) / G0 * (cos ** 3)[:, None] * self.dl * self.dl   # mass-term AM weight
        self.W = band_weights(self.lat)
        # 5-degree blocks exactly as torque_map_anim (coarsen 20, boundary trim -> drops lat=90 row)
        self.clat = self.lat[:720].reshape(36, 20).mean(1)
        self.clon = self.lon.reshape(72, 20).mean(1)
        self.dA5 = (A ** 2) * np.cos(np.deg2rad(self.clat)) * (5 * DEG) ** 2
        self.m5 = {}
        self.edges = {}
        for r, (lo0, lo1, la0, la1) in RANGES.items():
            ym = (self.clat >= la0) & (self.clat <= la1)
            xm = (self.clon >= lo0) & (self.clon <= lo1)
            self.m5[r] = (ym, xm)
            # the 5-degree cells the site box actually covers -> block-aligned edges
            e = (self.clat[ym].min() - 2.375, self.clat[ym].max() + 2.625,
                 self.clon[xm].min() - 2.375, self.clon[xm].max() + 2.625)
            self.edges[r] = tuple(round(v) for v in e)
        # exact 0.25-degree masks on the block-aligned boxes
        self.box = {}
        for r, (la0, la1, lo0, lo1) in self.edges.items():
            self.box[r] = (np.where((self.lat >= la0) & (self.lat < la1))[0],
                           np.where((self.lon >= lo0) & (self.lon < lo1))[0])
        # coarse grids (h and p_s block-averaged together) for the resolution test
        self.coarse = {}
        for tag, k in (("mtn1", 4), ("mtn25", 10)):
            ny, nx = 720 // k, 1440 // k
            cl = self.lat[:720].reshape(ny, k).mean(1)
            cx = self.lon.reshape(nx, k).mean(1)
            hc = self.h[:720].reshape(ny, k, nx, k).mean((1, 3))
            dlc = k * 0.25 * DEG
            Ac = (A ** 2) * np.cos(np.deg2rad(cl))[:, None] * dlc * dlc
            boxes = {r: (np.where((cl >= la0) & (cl < la1))[0], np.where((cx >= lo0) & (cx < lo1))[0])
                     for r, (la0, la1, lo0, lo1) in self.edges.items()}
            self.coarse[tag] = dict(k=k, ny=ny, nx=nx, hc=hc, dlc=dlc, Ac=Ac, W=band_weights(cl), boxes=boxes)


_ds = None


def _open():
    global _ds
    if _ds is None:
        fs = gcsfs.GCSFileSystem(token="anon")
        _ds = xr.open_zarr(fs.get_mapper(STORE), chunks=None, decode_timedelta=False)
    return _ds


def one(t: pd.Timestamp, G: Grid, t0: pd.Timestamp):
    """All diagnostics for one hourly ERA5 time, or (t, None, reason)."""
    ds = _open()
    i = int((t - t0) / pd.Timedelta(hours=1))
    f = {}
    try:
        for k, v in FIELDS.items():
            f[k] = np.asarray(ds[v].isel(time=i).values, dtype=np.float64)[::-1, :]   # ARCO lat descending
    except Exception as e:                                                             # noqa: BLE001
        return t, None, f"{type(e).__name__}: {e}"[:160]
    nfill = 0
    for k, a in f.items():
        bad = ~np.isfinite(a)
        if bad.any():
            # a handful of NaN cells occur in ARCO's vertical-integral fields (seen 1959-2019, < 0.05%);
            # fill them with their latitude row's mean rather than lose the whole time
            if bad.mean() > 5e-4:
                return t, None, f"{k} has {bad.mean():.4f} NaN"
            rowm = np.nanmean(np.where(bad, np.nan, a), axis=1)
            a[bad] = np.broadcast_to(rowm[:, None], a.shape)[bad]
            nfill += int(bad.sum())
    sp = f["sp"]
    if not (40000 < sp.min() and sp.max() < 110000):
        return t, None, f"sp out of range {sp.min():.0f}..{sp.max():.0f}"
    out = {"nfill": nfill}
    dpd = (np.roll(sp, -1, axis=1) - np.roll(sp, 1, axis=1)) / (2 * G.dl)
    dens = {"mtn": G.h * dpd * G.cellA / HU, "fric": -f["tt"] * G.cos2A3 / HU,
            "gwd": -f["tg"] * G.cos2A3 / HU, "aam": f["mf"] * G.cos2A3 / MU,
            "aamm": sp * G.cos3mass / MU}
    for term, d in dens.items():
        z = d.sum(1)
        for b, v in zip(BANDS, G.W @ z):
            out[f"{term}_{b}"] = float(v)
        if term not in ("aam", "aamm"):
            for r, (yi, xi) in G.box.items():
                out[f"{term}_{r}"] = float(d[np.ix_(yi, xi)].sum())
    # site-exact 5-degree box sums (torque_map_anim.box_net5)
    m5 = (G.h * dpd)[:720].reshape(36, 20, 72, 20).mean((1, 3)) * G.dA5[:, None] / HU
    for r, (ym, xm) in G.m5.items():
        out[f"mtn5_{r}"] = float(m5[ym][:, xm].sum())
    # coarse-resolution mountain torque (h and p_s averaged together)
    for tag, c in G.coarse.items():
        k = c["k"]
        pc = sp[:720].reshape(c["ny"], k, c["nx"], k).mean((1, 3))
        dc = (np.roll(pc, -1, axis=1) - np.roll(pc, 1, axis=1)) / (2 * c["dlc"])
        d = c["hc"] * dc * c["Ac"] / HU
        z = d.sum(1)
        for b, v in zip(BANDS, c["W"] @ z):
            out[f"{tag}_{b}"] = float(v)
        for r, (yi, xi) in c["boxes"].items():
            out[f"{tag}_{r}"] = float(d[np.ix_(yi, xi)].sum())
    return t, out, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--hours", default="0,6,12,18")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-time", action="store_true",
                    help="one row per sampled time instead of daily means (the 2026 high-fidelity budget)")
    a = ap.parse_args()

    G = Grid()
    print("block-aligned range edges (lat0, lat1, lon0, lon1):", G.edges, flush=True)
    ds = _open()
    t0 = pd.Timestamp(ds.time.values[0])
    # the time axis is pre-allocated (1900..2050); the valid span is in the attributes
    tl = pd.Timestamp(ds.attrs.get("valid_time_stop_era5t") or ds.attrs["valid_time_stop"]) + pd.Timedelta(hours=23)
    tv0 = pd.Timestamp(ds.attrs.get("valid_time_start", "1940-01-01"))
    assert pd.Timestamp(ds.time.values[1]) - t0 == pd.Timedelta(hours=1)
    assert np.array_equal(ds.latitude.values[::-1].astype(np.float64), G.lat), "ARCO grid changed"
    assert np.array_equal(ds.longitude.values.astype(np.float64), G.lon), "ARCO grid changed"
    hours = [int(x) for x in a.hours.split(",")]
    end = min(pd.Timestamp(a.end), tl.normalize())
    days = pd.date_range(max(pd.Timestamp(a.start), tv0), end, freq="D")
    times = [d + pd.Timedelta(hours=h) for d in days for h in hours]
    print(f"{len(days)} days x {len(hours)} hours = {len(times)} times ({a.start}..{end:%Y-%m-%d}); "
          f"store valid {tv0:%Y-%m-%d}..{tl}", flush=True)

    res, bad, tstart = {}, [], time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for n, (t, row, err) in enumerate(ex.map(lambda tt: one(tt, G, t0), times)):
            if row is None:
                bad.append((t, err))
                print(f"  skip {t}: {err}", flush=True)
            else:
                res[t] = row
            if n % 400 == 0:
                el = time.time() - tstart
                print(f"  {n}/{len(times)} {t:%Y-%m-%d %HZ}  {el / 60:.1f} min, "
                      f"{el / max(n, 1):.2f} s/time", flush=True)

    if a.per_time:
        df = pd.DataFrame([{"time": t.strftime("%Y-%m-%dT%H:%M")} | res[t] for t in times if t in res])
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(a.out, index=False, float_format="%.9g")
        print(f"wrote {a.out}: {len(df)} of {len(times)} times, {len(bad)} bad, "
              f"{(time.time() - tstart) / 60:.1f} min", flush=True)
        return 0 if len(bad) < 0.02 * len(times) else 1

    rows = []
    for d in days:
        got = [res[d + pd.Timedelta(hours=h)] for h in hours if (d + pd.Timedelta(hours=h)) in res]
        r = {"date": d.strftime("%Y-%m-%d"), "nh": len(got)}
        if got:
            keys = got[0].keys()
            if len(got) == len(hours):
                r.update({k: float(np.mean([g[k] for g in got])) for k in keys})
            for h in hours:
                g = res.get(d + pd.Timedelta(hours=h))
                for term in ("mtn", "fric", "gwd", "aam", "aamm"):
                    r[f"{term}_glob_h{h:02d}"] = g[f"{term}_glob"] if g else np.nan
        rows.append(r)
    df = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False, float_format="%.9g")
    print(f"wrote {a.out}: {len(df)} days, {int((df.nh == len(hours)).sum())} complete, {len(bad)} bad times, "
          f"{(time.time() - tstart) / 60:.1f} min", flush=True)
    return 0 if len(bad) < 0.02 * len(times) else 1


if __name__ == "__main__":
    sys.exit(main())
