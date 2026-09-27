#!/usr/bin/env python3
"""Data for the PDO causality tests (2026-09-27; user: "Is there a way you could test the causality?"). LAPTOP job,
resumable, nothing raw kept. Two sources:

  amip   Pangeo CMIP6 amip (observed SST and sea ice prescribed, 1979-2014) for the 16 ENSO-study models where it exists
         (14 of them), up to 3 members each: monthly zg 500 hPa (0-90N, 2.5 deg, area-overlap), tas and pr (the Americas
         grid of cmip6_enso_extract, bilinear), psl (0-90N).
  dcpp   ESGF CMIP6 DCPP component C (Boer et al. 2016, GMD 9, 3751): dcppC-ipv-NexTrop-pos/-neg (the Interdecadal Pacific
         Variability SST pattern imposed ONLY north of the tropics, everything else free) and dcppC-ipv-pos/-neg (the full
         pattern) for the three models that published atmospheric fields (CNRM-CM6-1, HadGEM3-GC31-MM, IPSL-CM6A-LR):
         monthly zg 500 hPa (OPeNDAP slice of the one level), psl, tas, pr (compressed HTTP download, reduced, deleted),
         and ts for 3 members per run (the imposed SST pattern, measured rather than assumed).

Output: data/cmip6_enso/forced/{amip,dcpp}/<name>.npz

    python scripts/sst/cmip6_forced_pdo_extract.py amip [--workers 2] [--per-model 3]
    python scripts/sst/cmip6_forced_pdo_extract.py dcpp [--workers 2]
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE))
import cmip6_z500_extract as ZX                                             # noqa: E402  NH 2.5-deg area-overlap ops
import cmip6_pacific_ts_extract as PX                                       # noqa: E402  Pacific ocean-box ops

OUT = HERE / "data" / "cmip6_enso" / "forced"
CAT = HERE.parent / "strat" / "data" / "cmip6" / "pangeo_cmip6.csv"
LAT_A = np.arange(-60, 75.01, 2.5)
LON_A = np.arange(190, 330.01, 2.5)
ESGF = "https://esgf.ceda.ac.uk/esg-search/search"
DCPP_EXPS = ("dcppC-ipv-NexTrop-pos", "dcppC-ipv-NexTrop-neg", "dcppC-ipv-pos", "dcppC-ipv-neg")
DCPP_MODELS = ("IPSL-CM6A-LR", "HadGEM3-GC31-MM", "CNRM-CM6-1")         # CEDA-served first; CNRM only on its own node
UA = {"User-Agent": "scorvec-research/1.0"}


def months_of(da):
    return [f"{t.year:04d}-{t.month:02d}" for t in pd.to_datetime([str(t)[:10] for t in da.time.values])]


def norm(da):
    if "lon" in da.coords:
        da = da.assign_coords(lon=da.lon % 360).sortby("lon")
    return da.sortby("lat")


def nh_box(da):
    """(time, lat, lon) native -> 0-90N 2.5 deg area-overlap means."""
    A, B = ZX.ops(da.lat.values.astype(float), da.lon.values.astype(float))
    return np.einsum("il,tlm,jm->tij", A, da.values.astype("float32"), B, optimize=True).astype("float32")


def americas(da, scale=1.0):
    x = da.interp(lat=LAT_A, lon=LON_A).values.astype("float32") * scale
    return x


def pacific_ts(da):
    ocean, M, okbox, gw = PX.grid_ops(da.lat.values.astype(float), da.lon.values.astype(float))
    X = da.values.reshape(da.sizes["time"], -1).astype("float32")
    valid = np.isfinite(X) & (X >= PX.ICE); Xf = np.where(valid, X, 0.0)
    num = (M @ Xf.T).T; den = (M @ valid.T.astype("float32")).T
    box = np.where((den > 0) & okbox[None, :], num / np.maximum(den, 1e-12), np.nan)
    gm = (Xf @ gw) / np.maximum(valid @ gw, 1e-12)
    return box.reshape(-1, len(PX.LAT), len(PX.LON)).astype("float32"), gm.astype("float32")


def z500_of(da):
    plev = da.plev.values
    tgt = 50000.0 if plev.max() > 2000 else 500.0
    k = int(np.argmin(np.abs(plev - tgt)))
    if abs(plev[k] - tgt) > 1:
        raise ValueError(f"no 500 hPa: {plev}")
    return da.isel(plev=k)


# ------------------------------------------------------------------------------------------------ amip (Pangeo)
def amip_one(job):
    src, member, stores = job
    dest = OUT / "amip" / f"{src}_{member}.npz"
    if dest.exists():
        return f"{src} {member}: have"
    t0 = time.time()
    try:
        out = {}
        for var in ("zg", "psl", "tas", "pr"):
            if var not in stores or not isinstance(stores[var], str):
                continue
            ds = xr.open_zarr(stores[var], storage_options=PX.SO, consolidated=True)
            v = ds[var].sel(time=slice("1979-01-01", "2014-12-31"))
            if var == "zg":
                v = z500_of(v)
            v = norm(v).transpose("time", "lat", "lon")
            out["months"] = np.array(months_of(v))
            if var in ("zg", "psl"):
                out[var] = nh_box(v)
            else:
                out[var] = americas(v.load(), 86400.0 if var == "pr" else 1.0)
        np.savez_compressed(dest, **out, source_id=src, member_id=member)
        return f"{src} {member}: {time.time() - t0:.0f}s"
    except Exception as e:                                                  # noqa: BLE001
        return f"{src} {member}: FAILED {type(e).__name__}: {str(e)[:160]}"


def amip(a):
    (OUT / "amip").mkdir(parents=True, exist_ok=True)
    models = sorted({Path(f).stem.rsplit("_", 1)[0] for f in (HERE / "data" / "cmip6_enso").glob("*.nc") if ".part" not in f.name})
    cat = pd.read_csv(CAT)
    c = cat[(cat.table_id == "Amon") & (cat.experiment_id == "amip") & cat.variable_id.isin(["zg", "psl", "tas", "pr"])]
    c = c.sort_values("version").groupby(["source_id", "member_id", "variable_id"]).zstore.last().unstack("variable_id")
    jobs = []
    for m in models:
        if m not in c.index.get_level_values(0):
            continue
        sub = c.loc[m].dropna(subset=["zg", "tas", "pr"])
        for mem in sorted(sub.index, key=lambda s: (len(s), s))[: a.per_model]:
            jobs.append((m, mem, sub.loc[mem].to_dict()))
    print(f"amip: {len(jobs)} members, {len({j[0] for j in jobs})} models", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(amip_one, jobs)):
            print(f"[{i + 1}/{len(jobs)}] {r}", flush=True)
    print("done", flush=True)


# ------------------------------------------------------------------------------------------------ DCPP-C (ESGF)
def esgf_files(model, exp, var):
    """{member: [(title, http_url, opendap_url)]} sorted by title (time order); CEDA index, one data node per dataset."""
    q = (f"{ESGF}?project=CMIP6&source_id={model}&experiment_id={exp}&table_id=Amon&variable_id={var}&type=File&"
         "latest=true&limit=1000&format=application%2Fsolr%2Bjson&fields=title,url,data_node,dataset_id")
    docs = json.load(urllib.request.urlopen(urllib.request.Request(q, headers=UA), timeout=180))["response"]["docs"]
    by = {}
    for d in docs:
        mem = d["dataset_id"].split(".")[5]
        urls = {u.split("|")[2]: u.split("|")[0] for u in d["url"]}
        if urls.get("OPENDAP", "").endswith(".html"):                      # ESGF lists the DAP form page, not the endpoint
            urls["OPENDAP"] = urls["OPENDAP"][:-5]
        by.setdefault(mem, {}).setdefault(d["data_node"], {})[d["title"]] = (urls.get("HTTPServer"), urls.get("OPENDAP"))
    out = {}
    for mem, nodes in by.items():
        node = "esgf.ceda.ac.uk" if "esgf.ceda.ac.uk" in nodes else sorted(nodes)[0]
        out[mem] = [(t, *nodes[node][t]) for t in sorted(nodes[node])]
    return out


def fetch_http(url, dest):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=600) as r, open(dest, "wb") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        return
                    f.write(b)
        except Exception:                                                   # noqa: BLE001
            if attempt == 2:
                raise
            time.sleep(15)


def dcpp_one(job):
    model, exp, mem, files = job
    dest = OUT / "dcpp" / f"{model}_{exp}_{mem}.npz"
    if dest.exists():
        return f"{model} {exp} {mem}: have"
    t0 = time.time()
    try:
        out = {}
        for var, flist in files.items():
            parts, months = [], []
            for title, http, dap in flist:
                if var == "zg":
                    try:                                            # OPeNDAP: just the 500 hPa slice
                        ds = xr.open_dataset(dap, engine="netcdf4", decode_times=True)
                        v = norm(z500_of(ds["zg"])).transpose("time", "lat", "lon").load(); ds.close()
                    except Exception:                               # noqa: BLE001  some nodes' DAP fails: whole file
                        with tempfile.TemporaryDirectory() as td:
                            p = Path(td) / title
                            fetch_http(http, p)
                            ds = xr.open_dataset(p)
                            v = norm(z500_of(ds["zg"])).transpose("time", "lat", "lon").load(); ds.close()
                    parts.append(nh_box(v)); months += months_of(v)
                else:
                    with tempfile.TemporaryDirectory() as td:
                        p = Path(td) / title
                        fetch_http(http, p)
                        ds = xr.open_dataset(p)
                        v = norm(ds[var]).transpose("time", "lat", "lon").load(); ds.close()
                    months += months_of(v)
                    if var == "psl":
                        parts.append(nh_box(v))
                    elif var == "ts":
                        b, g = pacific_ts(v); parts.append(b); out.setdefault("ts_gm_parts", []).append(g)
                    else:
                        parts.append(americas(v, 86400.0 if var == "pr" else 1.0))
            out[var] = np.concatenate(parts); out[f"{var}_months"] = np.array(months)
        if "ts_gm_parts" in out:
            out["ts_gm"] = np.concatenate(out.pop("ts_gm_parts"))
        np.savez_compressed(dest, **out, source_id=model, experiment_id=exp, member_id=mem)
        return f"{model} {exp} {mem}: {time.time() - t0:.0f}s"
    except Exception as e:                                                  # noqa: BLE001
        return f"{model} {exp} {mem}: FAILED {type(e).__name__}: {str(e)[:200]}"


def dcpp(a):
    (OUT / "dcpp").mkdir(parents=True, exist_ok=True)
    jobs = []
    for model in DCPP_MODELS:                               # (jobs are re-ordered below: every NexTrop run before any full ipv run)
        if model == "CNRM-CM6-1":                           # esg1.umr-cnrm.fr timed out on every request on 2026-09-27
            try:
                urllib.request.urlopen(urllib.request.Request("http://esg1.umr-cnrm.fr/thredds/catalog.html", headers=UA), timeout=30)
            except Exception as e:                          # noqa: BLE001
                print(f"  {model}: data node unreachable ({type(e).__name__}) - skipped", flush=True); continue
        for exp in DCPP_EXPS:
            per = {var: esgf_files(model, exp, var) for var in ("zg", "psl", "tas", "pr", "ts")}
            mems = sorted(set(per["zg"]) & set(per["tas"]) & set(per["pr"]) & set(per["psl"]), key=lambda s: (len(s), s))
            mems = mems[: a.max_members]                    # 12 members x ~10 winters per arm is ample for a pos-neg difference
            for i, mem in enumerate(mems):
                files = {v: per[v][mem] for v in ("zg", "psl", "tas", "pr")}
                if i < 3 and mem in per["ts"]:
                    files["ts"] = per["ts"][mem]
                jobs.append((model, exp, mem, files))
            print(f"  {model} {exp}: {len(mems)} members", flush=True)
    jobs.sort(key=lambda j: (0 if "NexTrop" in j[1] else 1, j[0] != "IPSL-CM6A-LR"))
    print(f"dcpp: {len(jobs)} member-runs", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(dcpp_one, jobs)):
            print(f"[{i + 1}/{len(jobs)}] {r}", flush=True)
    print("done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["amip", "dcpp"])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--per-model", type=int, default=3)
    ap.add_argument("--max-members", type=int, default=12)
    a = ap.parse_args()
    {"amip": amip, "dcpp": dcpp}[a.mode](a)


if __name__ == "__main__":
    main()
