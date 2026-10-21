#!/usr/bin/env python3
"""Daily stratosphere-troposphere coupling series from CMIP6, for testing wave reflection and downward propagation on a
large sample (cmip6_coupling.py). Runs in Actions (.github/workflows/strat-coupling-cmip6.yml), one job per run,
reading the public Pangeo CMIP6 Google Cloud mirror anonymously, a zarr chunk at a time; nothing raw is kept.

WHAT THE MIRROR HAS (catalogue of 2026-09-25): gridded daily temperature (day ta) for three models only, so the vortex
and the wave driving come from the DynVarMIP zonal means, which some twenty models publish:
  EdayZ ua    zonal-mean u on plev39 (required)  -> u60 at 2, 10, 50 and 100 hPa (linear in latitude)
  EdayZ epfz  the vertical Eliassen-Palm flux (where published) -> fz100, fz50: its 45-75N area mean, m3 s-2; it is
              f a cos(phi) [v'theta'] / [theta]_p, so its sign is the heat flux's: negative = net downward waves
  day psl     (required) -> slp_p, slp_loo: the day's sea-level pressure (hPa) on ssw_imprint's 2.5 deg boxes projected
              on the CMIP6 post-SSW pattern, raw (the seasonal cycle comes off later: the projection is linear, so fitting
              it to the projected series is the same as fitting it box by box first); slp_loo on the pattern made
              without this model, for the nine that made it
  day tas     (required) -> tas_na, tas_eu: North America 30-60N 240-290E and northern Eurasia 45-65N 30-120E, K
  day va, ta  (with --regional, where both exist) -> hf100_sib, hf100_can: v*T* (departures from the zonal mean) at
              100 hPa over Siberia 45-75N 140-200E and Canada 45-75N 230-280E (Messori et al. 2022), and hf100_k1, the
              zonal-mean wave-1 [v'T'] over 45-75N
Each with year, month and day of the run's own calendar, which is stored with it.

    python cmip6_coupling_extract.py plan --catalog pangeo-cmip6.csv --models CanESM5,UKESM1-0-LL --members 3
    python cmip6_coupling_extract.py run --catalog pangeo-cmip6.csv --model CanESM5 --experiment historical \
        --member r1i1p2f1 --out out/
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

TABLE = {"ua": "EdayZ", "epfz": "EdayZ", "psl": "day", "tas": "day", "va": "day", "ta": "day"}
REQUIRED = ("ua", "psl", "tas")
U_LEVELS = (2, 10, 50, 100)
BOX = {
    "sib": (45.0, 75.0, 140.0, 200.0),
    "can": (45.0, 75.0, 230.0, 280.0),
    "na": (30.0, 60.0, 240.0, 290.0),
    "eu": (45.0, 65.0, 30.0, 120.0),
}
SERIES = (
    "u2", "u10", "u50", "u100", "fz100", "fz50", "slp_p", "slp_loo", "tas_na", "tas_eu",
    "hf100_sib", "hf100_can", "hf100_k1",
)  # fmt: skip


# ----------------------------------------------------------------------------------------------------- catalogue ---
def catalog(path: str) -> pd.DataFrame:
    c = pd.read_csv(path)
    return c[c.variable_id.isin(TABLE) & c.experiment_id.isin(("piControl", "historical"))]


def stores(cat: pd.DataFrame, model: str, exp: str, member: str, regional: bool) -> dict:
    """{var: zstore} for one run, the newest version of each (a native grid first); {} without a required variable."""
    c = cat[(cat.source_id == model) & (cat.experiment_id == exp) & (cat.member_id == member)]
    out = {}
    for v, tab in TABLE.items():
        if v in ("va", "ta") and not regional:
            continue
        g = c[(c.table_id == tab) & (c.variable_id == v)]
        if len(g):
            g = g.assign(native=g.grid_label.str.startswith("gn")).sort_values(["native", "version"])
            out[v] = g.zstore.iloc[-1]
    if not all(v in out for v in REQUIRED):
        return {}
    if not {"va", "ta"} <= set(out):
        out.pop("va", None)
        out.pop("ta", None)
    return out


def plan(cat: pd.DataFrame, models, members: int, regional: bool) -> list[dict]:
    """Every run to extract: each model's piControl and its first `members` historical members (r1, r2, ...)."""
    out = []
    for m in models:
        for exp, n in (("piControl", 1), ("historical", members)):
            mems = sorted(
                cat[(cat.source_id == m) & (cat.experiment_id == exp)].member_id.unique(),
                key=lambda x: [int(p) for p in re.findall(r"\d+", x)],
            )
            got = 0
            for mem in mems:
                if got >= n:
                    break
                if stores(cat, m, exp, mem, regional):
                    out.append({"model": m, "experiment": exp, "member": mem})
                    got += 1
    return out


# -------------------------------------------------------------------------------------------------------- reduce ---
def open_var(zstore: str, var: str):
    import gcsfs

    fs = gcsfs.GCSFileSystem(token="anon")
    ds = xr.open_zarr(fs.get_mapper(zstore), consolidated=True, decode_times=xr.coders.CFDatetimeCoder(use_cftime=True))
    da = ds[var]
    da = da.rename({d: n for d, n in (("latitude", "lat"), ("longitude", "lon")) if d in da.dims})
    if "lon" in da.dims and da.sizes["lon"] == 1:
        da = da.isel(lon=0, drop=True)
    return da.sortby("lat")


def _plev_index(p, hpa):
    scale = 100.0 if np.nanmax(p) > 2000 else 1.0  # Pa, as CMIP6 has it; hPa just in case
    k = int(np.argmin(np.abs(p - hpa * scale)))
    return k if abs(p[k] - hpa * scale) <= 0.05 * hpa * scale else None


def _area_mean(x, lat, lo, hi):
    sel = (lat >= lo) & (lat <= hi)
    w = np.cos(np.deg2rad(lat[sel]))
    return (x[..., sel] * w).sum(-1) / w.sum()


def _interp60(zm, lat):
    j = int(np.searchsorted(lat, 60.0))
    f = (60.0 - lat[j - 1]) / (lat[j] - lat[j - 1])
    return zm[..., j - 1] * (1 - f) + zm[..., j] * f


class Fields:
    """The 2-D reductions on one day-table grid (latitudes ascending)."""

    def __init__(self, lat, lon, model):
        self.lat, self.lon = np.asarray(lat, float), np.asarray(lon, float) % 360.0
        self.cos = np.cos(np.deg2rad(self.lat))
        self.box = {
            k: ((self.lat >= a) & (self.lat <= b), (self.lon >= c) & (self.lon <= d)) for k, (a, b, c, d) in BOX.items()
        }
        self.boxes = SI.Boxes(self.lat, self.lon)
        ref = SI.Ref()
        loo = ref.ds.pattern_loo.sel(model=model).values if model in ref.ds.model.values else ref.pattern
        w = np.cos(np.deg2rad(ref.ds.lat.values))[:, None]
        self.proj = [w * p / (w * p * p).sum() for p in (ref.pattern, loo)]

    def box_mean(self, x, k):
        la, lo = self.box[k]
        ww = self.cos[la]
        return (x[:, la][:, :, lo].mean(-1) * ww).sum(1) / ww.sum()


def reduce_chunk(var: str, a: np.ndarray, da, F: Fields, t100=None) -> dict:
    """One chunk (time first) of one variable -> its series; for "regional", a is v and t100 is T at 100 hPa."""
    if var == "ua":
        p, lat = da.plev.values, da.lat.values
        out = {}
        for L in U_LEVELS:
            k = _plev_index(p, L)
            out[f"u{L}"] = _interp60(a[:, k], lat) if k is not None else np.full(len(a), np.nan)
        return out
    if var == "epfz":
        p, lat = da.plev.values, da.lat.values
        out = {}
        for L in (100, 50):
            k = _plev_index(p, L)
            out[f"fz{L}"] = _area_mean(a[:, k], lat, 45.0, 75.0) if k is not None else np.full(len(a), np.nan)
        return out
    if var == "psl":
        b = F.boxes(a / 100.0)
        return dict(zip(("slp_p", "slp_loo"), ((b * w).sum((-2, -1)) for w in F.proj)))
    if var == "tas":
        return {"tas_na": F.box_mean(a, "na"), "tas_eu": F.box_mean(a, "eu")}
    v, t = a.astype("float64"), t100.astype("float64")
    prod = (v - v.mean(-1, keepdims=True)) * (t - t.mean(-1, keepdims=True))
    out = {}
    for k in ("sib", "can"):
        la, lo = F.box[k]
        ww = F.cos[la]
        out[f"hf100_{k}"] = (prod[:, la][:, :, lo].mean(-1) * ww).sum(1) / ww.sum()
    band = (F.lat >= 45.0) & (F.lat <= 75.0)
    n = v.shape[-1]
    V, T = np.fft.rfft(v[:, band], axis=-1), np.fft.rfft(t[:, band], axis=-1)
    w = F.cos[band] / F.cos[band].sum()
    out["hf100_k1"] = ((2.0 * (V[..., 1] * np.conj(T[..., 1])).real / n**2) * w).sum(1)
    return out


def _read(da, idx, level=None):
    """The store's time steps `idx` (a contiguous run is read as a slice, anything else index by index)."""
    x = da.isel(time=slice(int(idx[0]), int(idx[-1]) + 1) if (np.diff(idx) == 1).all() else idx)
    if level is not None:
        x = x.isel(plev=_plev_index(da.plev.values, level))
    a = x.values.astype("float32")
    return np.where(np.abs(a) > 1e10, np.nan, a)


def run(cat, model, exp, member, years, out: Path, workers: int, regional: bool) -> dict:
    dest = out / f"{model}_{exp}_{member}.npz"
    st = stores(cat, model, exp, member, regional)
    if not st:
        return {"run": dest.stem, "status": "missing a required variable"}
    t0 = time.time()
    das = {v: open_var(z, v) for v, z in st.items()}
    keys = {v: {(t.year, t.month, t.day): i for i, t in enumerate(da.time.values)} for v, da in das.items()}
    common = sorted(set.intersection(*(set(keys[v]) for v in REQUIRED)))
    if years:
        common = [c for c in common if c[0] < common[0][0] + years]
    if len(common) < 3650:
        return {"run": dest.stem, "status": f"only {len(common)} common days"}
    # Each variable's time step for every common day. A store can hold a day twice or skip one (CESM2-WACCM psl), so
    # the days are indexed, not sliced; the days some required variable lacks are dropped, and counted.
    span = {c for c in keys["psl"] if common[0] <= c <= common[-1]}
    dropped = len(span) - len(common)
    if dropped > 0.01 * len(span):
        return {"run": dest.stem, "status": f"{dropped} of {len(span)} days missing from some required variable"}
    idx = {}
    for v in list(das):
        ix = np.array([keys[v].get(c, -1) for c in common])
        if (ix < 0).any():
            das.pop(v)  # an optional variable that does not cover the run is left out
            continue
        idx[v] = ix
    if "va" not in das or "ta" not in das:
        das.pop("va", None)
        das.pop("ta", None)
    n = len(common)
    F = Fields(das["psl"].lat.values, das["psl"].lon.values, model)
    res = {k: np.full(n, np.nan, "float32") for k in SERIES}
    for job in [v for v in ("ua", "epfz", "psl", "tas") if v in das] + (["regional"] if "va" in das else []):
        lead = "va" if job == "regional" else job
        ct = int((das[lead].encoding.get("chunks") or (365,))[0])

        def one(s, job=job, lead=lead, ct=ct):
            e = min(s + ct, n)
            for attempt in range(5):
                try:
                    if job == "regional":
                        v = _read(das["va"], idx["va"][s:e], 100)
                        t = _read(das["ta"], idx["ta"][s:e], 100)
                        return s, reduce_chunk(job, v, das["va"], F, t)
                    return s, reduce_chunk(job, _read(das[lead], idx[lead][s:e]), das[lead], F)
                except Exception as err_:  # noqa: BLE001  (a GCS hiccup: retry the chunk)
                    err = err_
                    time.sleep(5 * (attempt + 1))
            raise err

        with ThreadPoolExecutor(workers) as ex:
            for s, r in ex.map(one, range(0, n, ct)):
                for k, x in r.items():
                    res[k][s : s + len(x)] = x
        print(f"    {dest.stem}: {job} done, {time.time() - t0:.0f} s", flush=True)
    cal = str(das["psl"].time.values[0].calendar)
    ymd = np.array(common, "int16")
    part = dest.with_suffix(".part.npz")
    np.savez_compressed(part, year=ymd[:, 0], month=ymd[:, 1], day=ymd[:, 2], calendar=cal, **res)
    part.replace(dest)
    return {
        "run": dest.stem,
        "status": "ok",
        "days": n,
        "calendar": cal,
        "first": "-".join(map(str, common[0])),
        "last": "-".join(map(str, common[-1])),
        "variables": sorted(das),
        "days_dropped": dropped,
        "seconds": round(time.time() - t0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="the runs to extract, as a JSON list (for the workflow's matrix)")
    p.add_argument("--catalog", required=True)
    p.add_argument("--models", required=True)
    p.add_argument("--members", type=int, default=3)
    p.add_argument("--regional", action="store_true")
    r = sub.add_parser("run", help="extract one run")
    r.add_argument("--catalog", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--experiment", required=True)
    r.add_argument("--member", required=True)
    r.add_argument("--picontrol-years", type=int, default=500)
    r.add_argument("--regional", action="store_true")
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--out", required=True)
    a = ap.parse_args()
    cat = catalog(a.catalog)
    if a.cmd == "plan":
        print(json.dumps(plan(cat, [m.strip() for m in a.models.split(",") if m.strip()], a.members, a.regional)))
        return 0
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    years = a.picontrol_years if a.experiment == "piControl" else None
    try:
        res = run(cat, a.model, a.experiment, a.member, years, out, a.workers, a.regional)
    except Exception as e:  # noqa: BLE001
        res = {"run": f"{a.model}_{a.experiment}_{a.member}", "status": f"failed: {type(e).__name__}: {str(e)[:300]}"}
    print(res, flush=True)
    (out / f"{res['run']}_log.json").write_text(json.dumps(res))
    return 0 if res["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
