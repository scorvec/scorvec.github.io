#!/usr/bin/env python3
"""CMIP6 historical AAM budget terms per model - input for the El Nino mountain-torque study.

Runs in Actions (.github/workflows/torque-cmip6.yml), one job per model, reading the public
Pangeo CMIP6 Google Cloud mirror anonymously. 1950-2014, up to --members members.

Monthly (Amon/fx), on each model's own grid:
  mountain  T_m = +sum a^2 h dp_s/dlam dlam (sin phi_n - sin phi_s)     ps + fx orog
            (LINEAR in p_s with h fixed, so the torque of the monthly-mean ps IS the monthly-mean torque)
  surface   T_f = -sum a^3 cos(phi) tau_u dlam (sin phi_n - sin phi_s)  tauu (its own grid)
  AAM       M   = +sum a^3/g cos(phi) [sum_k u_k dp_k] dlam (sin phi_n - sin phi_s), dp_k clipped to ps
  Nino-3.4  area-mean ts over 5S-5N, 170-120W (all ocean, so ts = SST)
  GWD       T_g = +2 pi a^3/g sum cos(phi) [int utendogw dp] (sin phi_n - sin phi_s)
            from EdayZ utendogw (daily zonal-mean orographic GWD tendency) where a model has it
Daily (CFday ps, one member): the mountain torque per range and band, for event statistics.

Regions are the ERA5 study's: global + six latitude bands, and the site's RANGES boxes in their
block-aligned form (the 5-degree cells torque_map_anim.box_net5 sums), selected by cell centre.
Units: torques Hadley (1e18 N m), AAM 1e25 kg m^2 s^-1, Nino-3.4 K.

    python scripts/mjo/src/torque_history_cmip6.py --catalog pangeo-cmip6.csv --model CESM2 --out out/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import gcsfs

A = 6.371e6
G = 9.80665
HU = 1e18
MU = 1e25
Y0, Y1 = 1950, 2014
# block-aligned site boxes: (lat0, lat1, lon0, lon1), half-open, by cell centre
EDGES = {"HIM": (25, 45, 70, 105), "ROC": (30, 60, 230, 260), "AND": (-55, 10, 280, 295),
         "GRL": (60, 85, 300, 345), "ANT": (-90, -65, 0, 360), "ALP": (35, 50, 0, 45)}
BANDS = {"glob": (-90, 90), "b9060s": (-90, -60), "b6030s": (-60, -30), "b3000s": (-30, 0),
         "b0030n": (0, 30), "b3060n": (30, 60), "b6090n": (60, 90)}
FS = gcsfs.GCSFileSystem(token="anon")


def band_weights(lat):
    W = np.zeros((len(BANDS), len(lat)))
    for i, (s, n) in enumerate(BANDS.values()):
        w = ((lat > s) & (lat < n)).astype(float)
        w[np.isclose(lat, s)] = 0.5 if s > -90 else 1.0
        w[np.isclose(lat, n)] = 0.5 if n < 90 else 1.0
        W[i] = w
    return W


def open_store(zstore):
    ds = xr.open_zarr(FS.get_mapper(zstore), consolidated=True,
                      decode_times=xr.coders.CFDatetimeCoder(use_cftime=True), decode_timedelta=False)
    return ds


def std_grid(da):
    """Rename to lat/lon, longitudes to 0..360 ascending, latitudes ascending."""
    ren = {}
    for d in da.dims:
        if d in ("latitude", "y") and "lat" not in da.dims:
            ren[d] = "lat"
        if d in ("longitude", "x") and "lon" not in da.dims:
            ren[d] = "lon"
    da = da.rename(ren)
    if "lat" not in da.dims or "lon" not in da.dims:
        raise ValueError(f"not a regular lat/lon grid: dims {da.dims}")
    da = da.assign_coords(lon=(da.lon.values % 360.0)).sortby("lon").sortby("lat")
    return da


class Geom:
    def __init__(self, lat, lon):
        self.lat = np.asarray(lat, float); self.lon = np.asarray(lon, float)
        d = np.diff(self.lon)
        if not np.allclose(d, d[0], rtol=1e-3) or abs(d[0] * len(self.lon) - 360) > 1e-3 * 360:
            raise ValueError("longitude spacing is not uniform/global")
        self.dlam = np.deg2rad(d[0])
        e = np.empty(len(self.lat) + 1)
        e[1:-1] = 0.5 * (self.lat[1:] + self.lat[:-1]); e[0] = -90.0; e[-1] = 90.0
        self.dsin = np.sin(np.deg2rad(e[1:])) - np.sin(np.deg2rad(e[:-1]))      # = cos(phi) dphi
        self.cos = np.cos(np.deg2rad(self.lat))
        self.W = band_weights(self.lat)
        self.box = {r: (np.where((self.lat >= la0) & (self.lat < la1))[0],
                        np.where((self.lon >= lo0) & (self.lon < lo1))[0])
                    for r, (la0, la1, lo0, lo1) in EDGES.items()}

    def regions(self, dens, prefix, out, ranges=True):
        """dens (..., lat, lon) already multiplied by cell weight -> band and range sums."""
        z = dens.sum(-1)
        bands = z @ self.W.T
        for i, b in enumerate(BANDS):
            out[f"{prefix}_{b}"] = bands[..., i]
        if ranges:
            for r, (yi, xi) in self.box.items():
                out[f"{prefix}_{r}"] = dens[..., yi, :][..., xi].sum((-2, -1))
        return out


def mountain(ps, h, g: Geom):
    """ps (t, lat, lon) Pa, h (lat, lon) m -> per-cell torque density x area, Hadley."""
    dpd = (np.roll(ps, -1, axis=-1) - np.roll(ps, 1, axis=-1)) / (2 * g.dlam)
    return h * dpd * (A ** 2) * g.dsin[:, None] * g.dlam / HU


def months_mask(t):
    yr = np.array([x.year for x in t]); mo = np.array([x.month for x in t])
    return yr, mo, (yr >= Y0) & (yr <= Y1)


def pick(cat, model):
    h = cat[(cat.source_id == model) & (cat.experiment_id == "historical")]

    def stores(tab, var):
        x = h[(h.table_id == tab) & (h.variable_id == var)]
        return {m: grp.sort_values("version").zstore.iloc[-1] for m, grp in x.groupby("member_id")}, x

    amon = {v: stores("Amon", v)[0] for v in ("ps", "tauu", "ua", "ts")}
    common = set.intersection(*(set(d) for d in amon.values()))

    def mkey(m):
        import re
        r, i, p, f = (int(x) for x in re.match(r"r(\d+)i(\d+)p(\d+)f(\d+)", m).groups())
        return (i, p, f, r)
    mems = sorted(common, key=mkey)
    psg = h[(h.table_id == "Amon") & (h.variable_id == "ps")].grid_label.iloc[0] if len(mems) else None
    o = cat[(cat.source_id == model) & (cat.table_id == "fx") & (cat.variable_id == "orog")]
    o = o.assign(pref=(o.experiment_id != "historical").astype(int) + (o.grid_label != psg).astype(int) * 2)
    orog = o.sort_values(["pref", "member_id"]).zstore.tolist()
    cfd, _ = stores("CFday", "ps")
    ogw, _ = stores("EdayZ", "utendogw")
    return dict(members=mems, amon=amon, orog=orog, cfday=cfd, ogw=ogw)


def load_orog(paths, lat, lon):
    for p in paths:
        try:
            o = std_grid(open_store(p)["orog"])
            if o.ndim > 2:
                o = o.isel({d: 0 for d in o.dims if d not in ("lat", "lon")})
            same = o.sizes["lat"] == len(lat) and o.sizes["lon"] == len(lon) and \
                np.allclose(o.lat.values, lat, atol=0.05) and np.allclose(o.lon.values, lon, atol=0.05)
            if same:
                return np.nan_to_num(o.values.astype(float)), p, "same grid"
            oi = o.interp(lat=lat, lon=lon, method="linear", kwargs={"fill_value": None})
            return np.nan_to_num(oi.values.astype(float)), p, "INTERPOLATED to the ps grid"
        except Exception as e:                                                 # noqa: BLE001
            print(f"  orog {p}: {e}", flush=True)
    raise RuntimeError("no usable orog")


def monthly_member(sel, mem, hgt_paths, log):
    t0 = time.time()
    out = {}
    ps = std_grid(open_store(sel["amon"]["ps"][mem])["ps"])
    yr, mo, m = months_mask(ps.time.values)
    ps = ps.isel(time=np.where(m)[0])
    yr, mo = yr[m], mo[m]
    gp = Geom(ps.lat.values, ps.lon.values)
    h, hp, how = load_orog(hgt_paths, gp.lat, gp.lon)
    log.update(orog=hp, orog_grid=how, ps_grid=f"{len(gp.lat)}x{len(gp.lon)}")
    P = ps.values.astype(float)
    if np.nanmax(P) < 2000:                                    # hPa stored -> Pa
        P = P * 100.0
    gp.regions(mountain(P, h, gp), "mtn", out)
    # surface stress, own grid
    tu = std_grid(open_store(sel["amon"]["tauu"][mem])["tauu"])
    yt, mt_, mk = months_mask(tu.time.values)
    tu = tu.isel(time=np.where(mk)[0])
    assert np.array_equal(yt[mk], yr) and np.array_equal(mt_[mk], mo), "tauu months differ from ps"
    gt = Geom(tu.lat.values, tu.lon.values)
    T = np.nan_to_num(tu.values.astype(float))
    gt.regions(-T * (A ** 3) * (gt.cos * gt.dsin)[:, None] * gt.dlam / HU, "fric", out)
    log["tauu_comment"] = str(tu.attrs.get("comment", ""))[:200]
    # Nino-3.4 from ts
    ts = std_grid(open_store(sel["amon"]["ts"][mem])["ts"])
    yx, mx, mkx = months_mask(ts.time.values)
    ts = ts.isel(time=np.where(mkx)[0])
    assert np.array_equal(yx[mkx], yr), "ts months differ from ps"
    b = ts.sel(lat=slice(-5, 5), lon=slice(190, 240))
    w = np.cos(np.deg2rad(b.lat.values))[:, None] * np.ones(b.sizes["lon"])[None, :]
    out["nino34"] = (b.values * w).sum((1, 2)) / w.sum()
    # AAM from ua on plev, ps-masked
    ua = std_grid(open_store(sel["amon"]["ua"][mem])["ua"])
    pdim = [d for d in ua.dims if d not in ("time", "lat", "lon")][0]
    ua = ua.sortby(pdim)
    yu, mu_, mku = months_mask(ua.time.values)
    ua = ua.isel(time=np.where(mku)[0])
    assert np.array_equal(yu[mku], yr), "ua months differ from ps"
    p = ua[pdim].values.astype(float)
    if p.max() < 2000:
        p = p * 100.0
    gu = Geom(ua.lat.values, ua.lon.values)
    if len(gu.lat) == len(gp.lat) and np.allclose(gu.lat, gp.lat) and np.allclose(gu.lon, gp.lon):
        Pu = P
    else:                                                      # staggered (UM) grids: ps to the u points
        Pu = ps.copy(data=P).interp(lat=gu.lat, lon=gu.lon, method="linear",
                                    kwargs={"fill_value": None}).values
    edges = np.empty(len(p) + 1)
    edges[1:-1] = 0.5 * (p[1:] + p[:-1]); edges[0] = 0.0; edges[-1] = p[-1] + 0.5 * (p[-1] - p[-2])
    aam = np.empty((len(yr), len(gu.lat), len(gu.lon)))
    wlat = (A ** 3 / G) * (gu.cos * gu.dsin)[:, None] * gu.dlam / MU
    for i0 in range(0, len(yr), 24):
        u = np.nan_to_num(ua.isel(time=slice(i0, i0 + 24)).values.astype(float))   # (t, lev, lat, lon)
        sp = Pu[i0:i0 + 24][:, None]
        lo = np.minimum(edges[:-1][None, :, None, None], sp)
        hi = np.minimum(edges[1:][None, :, None, None], sp)
        aam[i0:i0 + 24] = (u * np.clip(hi - lo, 0, None)).sum(1) * wlat
    gu.regions(aam, "aam", out, ranges=False)
    df = pd.DataFrame({"year": yr, "month": mo} | {k: np.asarray(v) for k, v in out.items()})
    print(f"  {mem}: monthly done ({time.time() - t0:.0f}s)", flush=True)
    return df


def gwd_member(path):
    """EdayZ utendogw (time, plev, lat) daily zonal means -> monthly GWD torque per band."""
    da = open_store(path)["utendogw"]
    ren = {d: "lat" for d in da.dims if d in ("latitude",)}
    da = da.rename(ren).sortby("lat")
    pdim = [d for d in da.dims if d not in ("time", "lat")][0]
    da = da.sortby(pdim)
    yr, mo, m = months_mask(da.time.values)
    da = da.isel(time=np.where(m)[0])
    p = da[pdim].values.astype(float)
    if p.max() < 2000:
        p = p * 100.0
    edges = np.empty(len(p) + 1)
    edges[1:-1] = 0.5 * (p[1:] + p[:-1]); edges[0] = 0.0; edges[-1] = p[-1] + 0.5 * (p[-1] - p[-2])
    dp = np.diff(edges)
    lat = da.lat.values.astype(float)
    e = np.empty(len(lat) + 1); e[1:-1] = 0.5 * (lat[1:] + lat[:-1]); e[0] = -90; e[-1] = 90
    dsin = np.sin(np.deg2rad(e[1:])) - np.sin(np.deg2rad(e[:-1]))
    W = band_weights(lat)
    vals = []
    for i0 in range(0, da.sizes["time"], 3650):
        x = np.nan_to_num(da.isel(time=slice(i0, i0 + 3650)).values.astype(float))   # (t, lev, lat)
        col = (x * dp[None, :, None]).sum(1)                                         # int u_t dp
        tq = 2 * np.pi * (A ** 3 / G) * col * (np.cos(np.deg2rad(lat)) * dsin)[None, :] / HU
        vals.append(tq @ W.T)
    v = np.concatenate(vals)
    df = pd.DataFrame({"year": yr[m], "month": mo[m]} | {f"gwd_{b}": v[:, i] for i, b in enumerate(BANDS)})
    return df.groupby(["year", "month"], as_index=False).mean()


def daily_member(path, hgt_paths, log):
    ps = std_grid(open_store(path)["ps"])
    t = ps.time.values
    yr = np.array([x.year for x in t])
    keep = np.where((yr >= Y0) & (yr <= Y1))[0]
    g = Geom(ps.lat.values, ps.lon.values)
    h, hp, how = load_orog(hgt_paths, g.lat, g.lon)
    log.update(daily_orog=hp, daily_orog_grid=how, daily_grid=f"{len(g.lat)}x{len(g.lon)}",
               calendar=str(getattr(t[0], "calendar", "")))
    parts = []
    for i0 in range(0, len(keep), 730):
        idx = keep[i0:i0 + 730]
        P = ps.isel(time=idx).values.astype(float)
        out = {}
        g.regions(mountain(P, h, g), "mtn", out)
        tt = t[idx]
        parts.append(pd.DataFrame({"date": [f"{x.year:04d}-{x.month:02d}-{x.day:02d}" for x in tt]}
                                  | {k: np.asarray(v) for k, v in out.items()}))
    df = pd.concat(parts, ignore_index=True)
    return df.groupby("date", as_index=False).mean()          # 1 value/day even if the store is sub-daily


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--members", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-daily", action="store_true")
    a = ap.parse_args()
    cat = pd.read_csv(a.catalog)
    sel = pick(cat, a.model)
    od = Path(a.out); od.mkdir(parents=True, exist_ok=True)
    log = {"model": a.model, "members_available": len(sel["members"]), "members": [],
           "cfday_members": sorted(sel["cfday"]), "ogw_members": sorted(sel["ogw"])}
    if not sel["members"] or not sel["orog"]:
        log["error"] = "missing Amon ps/tauu/ua/ts member or fx orog"
        (od / f"{a.model}_log.json").write_text(json.dumps(log, indent=1))
        print(log); return 1
    frames = []
    for mem in sel["members"][:a.members]:
        try:
            df = monthly_member(sel, mem, sel["orog"], log)
        except Exception as e:                                                 # noqa: BLE001
            print(f"  {mem}: FAILED {type(e).__name__}: {e}", flush=True)
            log.setdefault("failed", []).append(f"{mem}: {type(e).__name__}: {e}"[:300]); continue
        if mem in sel["ogw"]:
            try:
                gw = gwd_member(sel["ogw"][mem])
                df = df.merge(gw, on=["year", "month"], how="left")
                log.setdefault("gwd_members", []).append(mem)
            except Exception as e:                                             # noqa: BLE001
                log.setdefault("failed", []).append(f"gwd {mem}: {e}"[:300])
        df.insert(0, "member", mem); df.insert(0, "model", a.model)
        frames.append(df); log["members"].append(mem)
    if frames:
        pd.concat(frames).to_csv(od / f"{a.model}_monthly.csv", index=False, float_format="%.6g")
    if sel["cfday"] and not a.no_daily:
        prefer = [m for m in sel["members"] if m in sel["cfday"]] + sorted(sel["cfday"])
        mem = prefer[0]
        try:
            t0 = time.time()
            dd = daily_member(sel["cfday"][mem], sel["orog"], log)
            dd.insert(0, "member", mem); dd.insert(0, "model", a.model)
            dd.to_csv(od / f"{a.model}_daily.csv", index=False, float_format="%.6g")
            log["daily_member"] = mem; log["daily_days"] = len(dd)
            log["daily_span"] = [dd.date.iloc[0], dd.date.iloc[-1]]
            print(f"  daily {mem}: {len(dd)} days ({time.time() - t0:.0f}s)", flush=True)
            # the ENSO index for the daily member, if it was not one of the monthly members
            if mem not in log["members"] and mem in sel["amon"]["ts"]:
                ts = std_grid(open_store(sel["amon"]["ts"][mem])["ts"])
                yr, mo, mk = months_mask(ts.time.values)
                b = ts.isel(time=np.where(mk)[0]).sel(lat=slice(-5, 5), lon=slice(190, 240))
                w = np.cos(np.deg2rad(b.lat.values))[:, None] * np.ones(b.sizes["lon"])[None, :]
                pd.DataFrame({"model": a.model, "member": mem, "year": yr[mk], "month": mo[mk],
                              "nino34": (b.values * w).sum((1, 2)) / w.sum()}).to_csv(
                    od / f"{a.model}_daily_nino34.csv", index=False, float_format="%.6g")
        except Exception as e:                                                 # noqa: BLE001
            print(f"  daily {mem}: FAILED {type(e).__name__}: {e}", flush=True)
            log.setdefault("failed", []).append(f"daily {mem}: {type(e).__name__}: {e}"[:300])
    (od / f"{a.model}_log.json").write_text(json.dumps(log, indent=1, default=str))
    print(json.dumps(log, indent=1, default=str))
    return 0 if frames else 1


if __name__ == "__main__":
    sys.exit(main())
