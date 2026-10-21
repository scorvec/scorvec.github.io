#!/usr/bin/env python3
"""Daily stratosphere-troposphere coupling series from CMIP6, for testing wave reflection and downward propagation on a
large sample (cmip6_coupling.py). Runs in Actions (.github/workflows/strat-coupling-cmip6.yml), one job per model,
reading the public Pangeo CMIP6 Google Cloud mirror anonymously, a zarr chunk at a time; nothing raw is kept.

Per day of every run, from the day table (ua, va, ta, zg on plev8; psl; tas), each model on its own grid:
  u10            zonal-mean u at 60N, 10 hPa (linear in latitude): the SSW criterion
  u50            the same at 50 hPa (the shear below the 10 hPa level)
  z10, z100      polar-cap (65-90N, area-weighted) geopotential height, m
  hf100_k1, hf100_k2, hf100_tot, hf50_k1, hf50_k2, hf50_tot
                 zonal-mean eddy heat flux [v'T'] over 45-75N (area-weighted), waves 1, 2 and 1-12, K m s-1
  hf100_sib, hf100_can
                 v*T* (departures from the zonal mean) over Siberia 45-75N 140-200E and Canada 45-75N 230-280E,
                 the two halves of Messori et al.'s (2022) regional reflection index
  slp_p, slp_loo the day's sea-level pressure (hPa) on ssw_imprint's 2.5 deg boxes projected on the CMIP6 post-SSW
                 pattern, raw (the baseline comes off later: the projection is linear, so fitting the seasonal cycle
                 to the projected series is the same as fitting it box by box first); slp_loo on the pattern made
                 without this model, for a model that helped make it
  tas_na, tas_eu 2 m temperature, North America 30-60N 240-290E and northern Eurasia 45-65N 30-120E, K
with year, month, day of the model's own calendar (stored with it).

    python cmip6_coupling_extract.py --catalog pangeo-cmip6.csv --model MRI-ESM2-0 --picontrol-years 300 \
        --historical-members 3 --out out/
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ssw_imprint as SI

DAY_VARS = ("ua", "va", "ta", "zg", "psl", "tas")
BOX = {
    "sib": (45.0, 75.0, 140.0, 200.0),
    "can": (45.0, 75.0, 230.0, 280.0),
    "na": (30.0, 60.0, 240.0, 290.0),
    "eu": (45.0, 65.0, 30.0, 120.0),
}
KMAX = 12
SERIES = (
    "u10", "u50", "z10", "z100",
    "hf100_k1", "hf100_k2", "hf100_tot", "hf50_k1", "hf50_k2", "hf50_tot",
    "hf100_sib", "hf100_can", "slp_p", "slp_loo", "tas_na", "tas_eu",
)  # fmt: skip


def catalog(path: str) -> pd.DataFrame:
    c = pd.read_csv(path)
    return c[(c.table_id == "day") & c.variable_id.isin(DAY_VARS)]


def runs_for(cat: pd.DataFrame, model: str, members: int) -> list[tuple[str, str, dict]]:
    """[(experiment, member, {var: zstore})] with every day variable on one grid: piControl first, then up to
    `members` historical members (r1, r2, ... in order)."""
    c = cat[cat.source_id == model]
    out = []
    for exp in ("piControl", "historical"):
        e = c[c.experiment_id == exp]
        mems = []
        for (mem, grid), g in e.groupby(["member_id", "grid_label"]):
            have = g.sort_values("version").groupby("variable_id").zstore.last()
            if set(DAY_VARS) <= set(have.index):
                mems.append((mem, grid, have.to_dict()))
        mems.sort(key=lambda x: (x[1] != "gn", [int(p) for p in re.findall(r"\d+", x[0])]))
        seen = set()
        for mem, _, stores in mems:
            if mem in seen:
                continue
            seen.add(mem)
            out.append((exp, mem, stores))
            if exp == "piControl" or len([o for o in out if o[0] == exp]) >= members:
                break
    return out


def open_var(zstore: str, var: str):
    import gcsfs

    fs = gcsfs.GCSFileSystem(token="anon")
    ds = xr.open_zarr(fs.get_mapper(zstore), consolidated=True, decode_times=xr.coders.CFDatetimeCoder(use_cftime=True))
    da = ds[var]
    ren = {d: n for d, n in (("latitude", "lat"), ("longitude", "lon")) if d in da.dims}
    da = da.rename(ren)
    if "lat" not in da.dims or "lon" not in da.dims:
        raise ValueError(f"{var}: not a regular lat/lon grid ({da.dims})")
    return da


class Grid:
    """Weights and selections on a model's native grid (latitudes ascending after sorting, longitudes 0..360)."""

    def __init__(self, lat, lon, pattern, loo):
        self.lat, self.lon = np.asarray(lat, float), np.asarray(lon, float) % 360.0
        self.cos = np.cos(np.deg2rad(self.lat))
        j = np.searchsorted(self.lat, 60.0)
        self.i60 = (j - 1, j, (60.0 - self.lat[j - 1]) / (self.lat[j] - self.lat[j - 1]))
        self.cap = self.lat >= 65.0
        self.band = (self.lat >= 45.0) & (self.lat <= 75.0)
        self.box = {
            k: ((self.lat >= a) & (self.lat <= b), (self.lon >= c) & (self.lon <= d)) for k, (a, b, c, d) in BOX.items()
        }
        self.boxes = SI.Boxes(self.lat, self.lon)
        w = np.cos(np.deg2rad(SI.Ref().ds.lat.values))[:, None]
        self.proj = [w * p / (w * p * p).sum() for p in (pattern, loo)]

    def u60(self, u):  # (t, lat, lon) -> (t,)
        a, b, f = self.i60
        zm = u.mean(-1)
        return zm[:, a] * (1 - f) + zm[:, b] * f

    def cap_mean(self, z):
        zm = z[:, self.cap].mean(-1)
        w = self.cos[self.cap]
        return (zm * w).sum(1) / w.sum()

    def heatflux(self, v, t):
        """(t, lat, lon) v and T -> {k1, k2, tot} zonal-mean [v'T'] over 45-75N, and the regional v*T* boxes."""
        vb, tb = v[:, self.band].astype("float64"), t[:, self.band].astype("float64")
        n = v.shape[-1]
        V, T = np.fft.rfft(vb, axis=-1), np.fft.rfft(tb, axis=-1)
        cr = 2.0 * (V * np.conj(T)).real / n**2  # (t, lat, k): each wavenumber's share of the zonal mean
        w = self.cos[self.band] / self.cos[self.band].sum()
        out = {f"k{k}": (cr[..., k] * w).sum(1) for k in (1, 2)}
        out["tot"] = (cr[..., 1 : KMAX + 1].sum(-1) * w).sum(1)
        prod = (v - v.mean(-1, keepdims=True)) * (t - t.mean(-1, keepdims=True))
        for k in ("sib", "can"):
            la, lo = self.box[k]
            ww = self.cos[la]
            out[k] = (prod[:, la][:, :, lo].mean(-1) * ww).sum(1) / ww.sum()
        return out

    def box_mean(self, x, k):
        la, lo = self.box[k]
        ww = self.cos[la]
        return (x[:, la][:, :, lo].mean(-1) * ww).sum(1) / ww.sum()


def chunk_reduce(das: dict, grid: Grid, i0: int, i1: int) -> dict:
    """The day series for time indices i0..i1: each variable read once (the levels it needs together, float32, so a
    chunk of a plev8 store is decompressed once), reduced, and let go before the next."""

    def read(v, levels=None):
        da = das[v].isel(time=slice(i0, i1))
        if levels:
            p = da.plev.values
            scale = 100.0 if p.max() > 2000 else 1.0  # Pa, as CMIP6 has it; hPa just in case
            da = da.isel(plev=[int(np.argmin(np.abs(p - L * scale))) for L in levels])
        a = da.sortby("lat").values.astype("float32")
        return np.where(np.abs(a) > 1e10, np.nan, a)

    out = {}
    u = read("ua", (10, 50))
    out["u10"], out["u50"] = grid.u60(u[:, 0]), grid.u60(u[:, 1])
    z = read("zg", (10, 100))
    out["z10"], out["z100"] = grid.cap_mean(z[:, 0]), grid.cap_mean(z[:, 1])
    del u, z
    v, t = read("va", (100, 50)), read("ta", (100, 50))
    for i, L in enumerate((100, 50)):
        hf = grid.heatflux(v[:, i], t[:, i])
        for k in ("k1", "k2", "tot"):
            out[f"hf{L}_{k}"] = hf[k]
        if L == 100:
            out["hf100_sib"], out["hf100_can"] = hf["sib"], hf["can"]
    del v, t
    b = grid.boxes(read("psl") / 100.0)
    out["slp_p"], out["slp_loo"] = ((b * w).sum((-2, -1)) for w in grid.proj)
    t2 = read("tas")
    out["tas_na"], out["tas_eu"] = grid.box_mean(t2, "na"), grid.box_mean(t2, "eu")
    return out


def run(model: str, exp: str, member: str, stores: dict, years: int | None, out: Path, workers: int) -> dict:
    dest = out / f"{model}_{exp}_{member}.npz"
    if dest.exists():
        return {"run": dest.stem, "status": "have"}
    t0 = time.time()
    das = {v: open_var(stores[v], v) for v in DAY_VARS}
    # the times every variable shares, by calendar date
    keys = {}
    for v, da in das.items():
        keys[v] = {(t.year, t.month, t.day): i for i, t in enumerate(da.time.values)}
    common = sorted(set.intersection(*(set(k) for k in keys.values())))
    if years:
        y0 = common[0][0]
        common = [c for c in common if c[0] < y0 + years]
    if len(common) < 3650:
        return {"run": dest.stem, "status": f"only {len(common)} common days"}
    # each variable must hold the common days as one contiguous block (true of CMIP6 day stores)
    for v in DAY_VARS:
        idx = np.array([keys[v][c] for c in common])
        if not (np.diff(idx) == 1).all():
            return {"run": dest.stem, "status": f"{v}: the common days are not contiguous"}
        das[v] = das[v].isel(time=slice(int(idx[0]), int(idx[-1]) + 1))
    lat = np.sort(das["psl"].lat.values)
    ref = SI.Ref()
    loo = ref.ds.pattern_loo.sel(model=model).values if model in ref.ds.model.values else ref.pattern
    grid = Grid(lat, das["psl"].lon.values, ref.pattern, loo)
    for v in ("ua", "va", "ta", "zg"):
        if not np.allclose(np.sort(das[v].lat.values), lat) or len(das[v].lon) != len(das["psl"].lon):
            return {"run": dest.stem, "status": f"{v} is on another grid than psl"}
    n = len(common)
    ct = int((das["va"].encoding.get("chunks") or (365,))[0])
    starts = list(range(0, n, ct))
    res = {k: np.full(n, np.nan, "float32") for k in SERIES}

    def one(s):
        for attempt in range(5):
            try:
                return s, chunk_reduce(das, grid, s, min(s + ct, n))
            except Exception as e:  # noqa: BLE001  (a GCS hiccup: retry the chunk)
                err = e
                time.sleep(5 * (attempt + 1))
        raise err

    with ThreadPoolExecutor(workers) as ex:
        for done, (s, r) in enumerate(ex.map(one, starts), 1):
            for k, v in r.items():
                res[k][s : s + len(v)] = v
            if done % 20 == 0:
                print(f"    {dest.stem}: {done}/{len(starts)} chunks, {time.time() - t0:.0f} s", flush=True)
    cal = str(das["psl"].time.values[0].calendar)
    ymd = np.array(common, "int16")
    np.savez_compressed(
        dest.with_suffix(".part.npz"), year=ymd[:, 0], month=ymd[:, 1], day=ymd[:, 2], calendar=cal, **res
    )
    dest.with_suffix(".part.npz").replace(dest)
    return {
        "run": dest.stem,
        "status": "ok",
        "days": n,
        "calendar": cal,
        "first": f"{common[0]}",
        "last": f"{common[-1]}",
        "seconds": round(time.time() - t0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--picontrol-years", type=int, default=300)
    ap.add_argument("--historical-members", type=int, default=3)
    ap.add_argument("--workers", type=int, default=3, help="chunks in flight (each ~0.5 GB at a 1-deg grid)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = runs_for(catalog(a.catalog), a.model, a.historical_members)
    print(f"{a.model}: {len(runs)} runs with every day variable: {[f'{e} {m}' for e, m, _ in runs]}", flush=True)
    log = []
    for exp, mem, stores in runs:
        try:
            r = run(a.model, exp, mem, stores, a.picontrol_years if exp == "piControl" else None, out, a.workers)
        except Exception as e:  # noqa: BLE001  (one bad store must not cost the model's other runs)
            r = {"run": f"{a.model}_{exp}_{mem}", "status": f"failed: {type(e).__name__}: {str(e)[:200]}"}
        print(f"  {r}", flush=True)
        log.append(r)
    (out / f"{a.model}_log.json").write_text(json.dumps(log, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
