#!/usr/bin/env python3
"""The PNA, EPO and WPO indices of the days 1-15 teleconnection forecasts (ensmaps_telecon.py), defined once: the GEFS
and GEPS pages' projectors applied to the 500 hPa height at one synoptic hour, and the table of every archived run's
index against ERA5's that calibrates them.

  ref       The reference, built once in Actions (telecon15.yml; release telecon15-ref-v1, telecon15_ref.npz):
            - the projectors of release gefs-ref-v1 (gefs_patterns_ref.npz; method and validation in
              scripts/geps/build/build_telecon_patterns.py): the PNA the month-specific least-squares projector of the
              regression on CPC's monthly indices, applied to 500 hPa height anomalies over std_month (the sd of the
              calendar month's monthly anomalies) weighted by sqrt(cos lat); EPO and WPO PSL's box differences in metres.
              The weight is folded into the projector, which is cut to the 2.5 deg rows 20-90N (latlon.G25N), with
              std_month on the same rows: the build stops if any weight south of 20N would be lost.
            - sigma, the seasonal sd of each daily index (1991-2020, within 15 days of the day of year), and the months
              CPC defines it in, on the 366-slot calendar (latlon.slot366), 29 February the mean of 28 February and
              1 March as the GEFS page reads it (gefs_reforecast.at_key).
            - the normal: ERA5 1991-2020 of release verify-clim-z500-v1 (daily means, 1.5 deg) remapped conservatively
              onto the same grid, plus the 500 hPa tide at 00 and 12 UTC, each hour's departure from the mean of the
              four synoptic hours in WeatherBench 2's ERA5 climatology by hour of day (1990-2019, 1.5 deg), fitted
              with a mean and three annual harmonics: a 00Z or 12Z field is compared with the normal of its own hour.
            The build checks the cut projectors against gefs_patterns.Ref.index on random global fields and prints the
            tide's part of each index.
  index     (field - normal of its hour and day) [/ std_month] . projector / sigma, in seasonal sd of the daily index;
            NaN in a month CPC does not define a pattern (every month for these three).
  hindcast  The table the calibration reads (the Actions cache, telecon15.yml): for every archived 00Z run of the
            500 hPa database from z500_acc.SINCE (scripts/verify/verifydb.sh), the ensemble-mean index at each lead,
            the mean of the members' indices (the index is linear in the field, so it is the index of the members'
            mean field, ens-z500m-<m>_<init>.npz), in fc.parquet; ERA5's index at 00Z of each day (era5-z500_, from
            the database's 1.5 deg grid onto 2.5 deg) in obs.parquet. A pass reads only the runs and days it lacks.

    python telecon15.py ref --patterns gefs_patterns_ref.npz --clim clim_z500_1p5.npz --out telecon15_ref.npz
    python telecon15.py hindcast --list --table tbl --have have.txt              # the database files the table lacks
    python telecon15.py hindcast --table tbl --derived db --ref telecon15_ref.npz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "lib"))
from latlon import G15, G25N, remap, slot366

REF_TAG = "telecon15-ref-v1"
REF_NAME = "telecon15_ref.npz"
IDS = ("pna", "epo", "wpo")
HOURS = (0, 12)  # the synoptic hours a forecast is read at: a 00Z run's 00Z, a 12Z run's 12Z
LEADS = tuple(range(24, 361, 24))  # hours: 00Z of each day to day 15, the database's ensemble leads (ens_verify.LEADS)
WB2_CLIM = (
    "gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_240x121_equiangular_with_poles_conservative.zarr"
)
TIDE_HARMONICS = 3
G = 9.80665
FC, OBS = "fc.parquet", "obs.parquet"


# ── the reference ─────────────────────────────────────────────────────────────────────────────────────────────────────
class Ref:
    """telecon15_ref.npz: proj (index, month, row, col; sqrt(cos lat) folded in), std_month (month, row, col),
    standardize (index), sigma (index, slot), defined (index, month), normal (slot, hour, row, col), meta."""

    def __init__(self, path: Path):
        with np.load(path) as z:
            self.ids = [str(s) for s in z["ids"]]
            self.P = z["proj"].astype(np.float64)
            self.std = z["std_month"].astype(np.float64)
            self.standardize = z["standardize"].astype(bool)
            self.sigma = z["sigma"].astype(np.float64)
            self.defined = z["defined"].astype(bool)
            self.normal = z["normal"]
            self.meta = json.loads(str(z["meta"]))

    def anomaly(self, z: np.ndarray, times: pd.DatetimeIndex) -> np.ndarray:
        """(n, row, col) 500 hPa heights (m) at `times`, each 00 or 12 UTC, less the normal of that hour and day."""
        if not np.isin(times.hour, HOURS).all():
            raise ValueError(f"the normal has the hours {HOURS} only, not {sorted(set(times.hour))}")
        h = np.searchsorted(HOURS, times.hour)
        return np.asarray(z, np.float64) - self.normal[slot366(times), h]

    def index(self, anom: np.ndarray, times: pd.DatetimeIndex) -> np.ndarray:
        """(n, index) in seasonal sd of the daily index, of anomalies (n, row, col) valid at `times`."""
        mo, s = times.month.to_numpy() - 1, slot366(times)
        out = np.empty((len(times), len(self.ids)))
        for k in range(len(self.ids)):
            a = anom / self.std[mo] if self.standardize[k] else anom
            out[:, k] = np.einsum("nij,nij->n", a, self.P[k][mo]) / self.sigma[k, s]
            out[~self.defined[k, mo], k] = np.nan
        return out


def tide(path: str = WB2_CLIM) -> np.ndarray:
    """(366, 2, 121, 240) on the 1.5 deg grid, 90N first, from 0E: ERA5's 500 hPa height at 00 and 12 UTC less the mean
    of the four synoptic hours, by slot of the 366-day calendar, as a mean and TIDE_HARMONICS annual harmonics.
    WeatherBench 2 keys its climatology by the raw day of the year; slot s is read as its day s + 1 (a day apart after
    February in common years, nothing in a curve this smooth)."""
    import xarray as xr

    z = xr.open_zarr(path, chunks=None, storage_options={"token": "anon"})["geopotential"].sel(level=500)
    z = z.transpose("dayofyear", "hour", "latitude", "longitude")
    if list(z["hour"].values) != [0, 6, 12, 18] or list(z["dayofyear"].values) != list(range(1, 367)):
        raise RuntimeError("WeatherBench 2's climatology is not by day of year and the four synoptic hours")
    lat = z["latitude"].values
    if not (np.allclose(np.sort(lat)[::-1], G15.lat) and np.allclose(z["longitude"].values, G15.lon)):
        raise RuntimeError("WeatherBench 2's climatology is not on the verification grid")
    v = z.values.astype(np.float64) / G
    v = v[:, :, ::-1] if lat[0] < lat[-1] else v
    d = v[:, [0, 2]] - v.mean(1, keepdims=True)  # 00 and 12 UTC
    x = 2 * np.pi * np.arange(366)[:, None] * np.arange(1, TIDE_HARMONICS + 1)[None] / 366.0
    B = np.hstack([np.ones((366, 1)), np.cos(x), np.sin(x)])
    coef, *_ = np.linalg.lstsq(B, d.reshape(366, -1), rcond=None)
    return (B @ coef).reshape(d.shape)


def build_ref(a) -> None:
    sys.path.insert(0, str(HERE.parents[0] / "gefs"))
    import gefs_patterns as GP
    import gefs_reforecast as GR

    gp = GP.Ref(a.patterns)
    rows = slice(0, G25N.lat.size)
    if not (np.allclose(GP.LAT25[rows], G25N.lat) and np.allclose(GP.LON25, G25N.lon)):
        raise SystemExit("the patterns' grid is not latlon.G25N's 20-90N")
    proj, std, sigma, defined = [], gp.std[:, rows], [], []
    days = pd.date_range("2024-01-01", "2024-12-31", freq="D")  # a leap year: every slot once, in order
    if not np.array_equal(slot366(days), np.arange(366)):
        raise SystemExit("slot366 does not number a leap year's days 0..365")
    for sid in IDS:
        P = gp.P[sid] * gp.w[None]
        if np.abs(P[:, rows.stop :]).max() > 0:
            raise SystemExit(f"{sid}: the projector has weight south of 20N")
        if gp.pat[sid]["standardize"] and not (std > 0).all():
            raise SystemExit(f"{sid}: std_month is not positive on every row 20-90N")
        if gp.pat[sid].get("demean"):
            raise SystemExit(f"{sid}: a demeaned pattern needs the whole grid")
        proj.append(P[:, rows])
        sigma.append(GR.at_key(gp.sigma_doy[sid], GR.noleap_doy(days.values.astype("datetime64[D]"))))
        defined.append([m in gp.defined[sid] for m in range(1, 13)])
    ref = dict(
        ids=np.array(IDS),
        proj=np.array(proj, np.float32),
        std_month=std.astype(np.float32),
        standardize=np.array([bool(gp.pat[s]["standardize"]) for s in IDS]),
        sigma=np.array(sigma, np.float64),
        defined=np.array(defined),
    )
    clim = np.load(a.clim)
    if clim["z500"].shape != (366, *G15.shape):
        raise SystemExit(f"{a.clim}: not the 366 slots of the 1.5 deg normal")
    daily = np.stack([remap(c, G15.lat, G15.lon, G25N) for c in clim["z500"]]).astype(np.float64)
    td = tide()
    td25 = np.stack([[remap(t, G15.lat, G15.lon, G25N) for t in s] for s in td])
    ref["normal"] = (daily[:, None] + td25).astype(np.float32)
    sha = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (a.patterns, a.clim)}
    meta = {
        "patterns": {s: gp.pat[s] for s in IDS},
        "hours": list(HOURS),
        "sources": {
            "patterns": f"gefs-ref-v1/{a.patterns.name} sha256 {sha[a.patterns.name]}",
            "normal": f"verify-clim-z500-v1/{a.clim.name} sha256 {sha[a.clim.name]}",
            "tide": f"{WB2_CLIM}, hours 00 and 12 less the four hours' mean, mean + {TIDE_HARMONICS} harmonics",
        },
    }
    ref["meta"] = np.array(json.dumps(meta))
    np.savez_compressed(a.out, **ref)
    check_ref(Ref(a.out), gp, GR, td25)
    print(f"reference: {a.out} ({a.out.stat().st_size / 1e6:.1f} MB)", flush=True)


def check_ref(r: Ref, gp, GR, td25: np.ndarray) -> None:
    """The cut projectors give gefs_patterns.Ref.index's numbers on random global fields, every day of a common and a
    leap year; the tide's size and its part of each index, month by month."""
    rng = np.random.default_rng(1)
    t = pd.date_range("2023-01-01", "2024-12-31", freq="D")
    anom = rng.normal(0.0, 80.0, (len(t), *gp.std.shape[1:]))
    mine = r.index(anom[:, : G25N.lat.size], t)
    keys = GR.noleap_doy(t.values.astype("datetime64[D]"))
    ref = np.stack([gp.index(s, anom, t.month, keys) for s in r.ids], 1)
    err = float(np.abs(mine - ref).max())
    print(f"check: the cut projectors against gefs_patterns.Ref.index, largest difference {err:.2e} sd", flush=True)
    if err > 1e-6:
        raise SystemExit("the cut projectors do not reproduce the GEFS page's indices")
    days = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    for h, hour in enumerate(HOURS):
        part = r.index(td25[slot366(days), h], days)
        by = pd.DataFrame(part, index=days, columns=r.ids).groupby(days.month).agg(lambda x: np.abs(x).max())
        print(f"check: the {hour:02d} UTC tide, largest {np.abs(td25[:, h]).max():.1f} m 20-90N; its part of each "
              f"index, largest |sd| in each month:\n{by.round(3)}", flush=True)  # fmt: skip


# ── the hindcast table ────────────────────────────────────────────────────────────────────────────────────────────────
def _read(table: Path) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    fc, obs = table / FC, table / OBS
    return (pd.read_parquet(fc) if fc.exists() else None, pd.read_parquet(obs) if obs.exists() else None)


def wanted(have: set[str], fc: pd.DataFrame | None, obs: pd.DataFrame | None) -> list[str]:
    """The members' files of the database's runs (from each model's SINCE) and the ERA5 days the table lacks."""
    import z500_acc as Z

    done = set() if fc is None else set(zip(fc["model"], fc["init"].dt.strftime("%Y%m%d%H"), strict=True))
    days = set() if obs is None else set(obs["valid"].dt.strftime("%Y%m%d%H"))
    code = {c: m for m, c in Z.E.ENSEMBLES.items()}
    out = []
    for n in sorted(have):
        if m := re.fullmatch(r"ens-z500m-(\w+)_(\d{10})\.npz", n):
            model = code.get(m[1])
            if model and m[2] >= Z.SINCE[model] and (model, m[2]) not in done:
                out.append(n)
        elif (m := re.fullmatch(r"era5-z500_(\d{10})\.npz", n)) and m[1] not in days:
            out.append(n)
    return out


def run_rows(r: Ref, path: Path, model: str) -> pd.DataFrame:
    """One run's ensemble-mean index at each lead (the index of its members' mean field)."""
    with np.load(path) as d:
        leads, z, lat, lon = d["lead_h"], d["z"], d["lat"], d["lon"]
    if not (np.allclose(lat, G25N.lat) and np.allclose(lon, G25N.lon)):
        raise RuntimeError(f"{path.name}: not on the 2.5 deg grid 20-90N")
    init = pd.to_datetime(path.stem.split("_")[1], format="%Y%m%d%H")
    t = pd.DatetimeIndex(init + pd.to_timedelta(leads, unit="h"))
    v = r.index(r.anomaly(z.astype(np.float64).mean(1), t), t)
    return pd.DataFrame(
        {"model": model, "init": init, "lead_h": leads.astype(np.int16), "n_members": np.int16(z.shape[1])}
        | {s: v[:, k].astype(np.float32) for k, s in enumerate(r.ids)}
    )


def era5_rows(r: Ref, paths: list[Path]) -> pd.DataFrame:
    """ERA5's index at 00Z of each day, its 1.5 deg field remapped onto the patterns' grid."""
    if not paths:
        return pd.DataFrame()
    t = pd.DatetimeIndex([pd.to_datetime(p.stem.split("_")[1], format="%Y%m%d%H") for p in paths])
    z = np.stack([remap(np.load(p)["z"], G15.lat, G15.lon, G25N) for p in paths])
    v = r.index(r.anomaly(z, t), t)
    return pd.DataFrame({"valid": t} | {s: v[:, k].astype(np.float32) for k, s in enumerate(r.ids)})


def hindcast(a) -> None:
    table = a.table
    fc, obs = _read(table)
    if a.list:
        print("\n".join(wanted(set(a.have.read_text().split()), fc, obs)))
        return
    import z500_acc as Z

    if tuple(Z.E.LEADS) != LEADS:
        raise SystemExit(f"the database's leads {list(Z.E.LEADS)} are not telecon15.LEADS")
    r = Ref(a.ref)
    code = {c: m for m, c in Z.E.ENSEMBLES.items()}
    runs = [
        run_rows(r, p, code[p.stem.split("_")[0].removeprefix("ens-z500m-")])
        for p in sorted(a.derived.glob("ens-z500m-*_*.npz"))
    ]
    days = era5_rows(r, sorted(a.derived.glob("era5-z500_*.npz")))
    table.mkdir(parents=True, exist_ok=True)
    if runs:
        fc = pd.concat([x for x in (fc, *runs) if x is not None], ignore_index=True)
        fc = fc.drop_duplicates(["model", "init", "lead_h"], keep="last").sort_values(["model", "init", "lead_h"])
        fc.reset_index(drop=True).to_parquet(table / FC, compression="zstd")
    if not days.empty:
        obs = pd.concat([x for x in (obs, days) if x is not None], ignore_index=True)
        obs = obs.drop_duplicates("valid", keep="last").sort_values("valid").reset_index(drop=True)
        obs.to_parquet(table / OBS, compression="zstd")
    held = {m: int(g["init"].nunique()) for m, g in fc.groupby("model")} if fc is not None else {}
    print(f"hindcast: {len(runs)} runs and {len(days)} ERA5 days added; the table holds runs {held}, "
          f"{0 if obs is None else len(obs)} ERA5 days", flush=True)  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("ref", help="build the reference (once)")
    b.add_argument("--patterns", type=Path, required=True, help="gefs_patterns_ref.npz (release gefs-ref-v1)")
    b.add_argument("--clim", type=Path, required=True, help="clim_z500_1p5.npz (release verify-clim-z500-v1)")
    b.add_argument("--out", type=Path, required=True)
    h = sub.add_parser("hindcast", help="add the database's new runs and ERA5 days to the table")
    h.add_argument("--table", type=Path, required=True, help="the table's directory (fc.parquet, obs.parquet)")
    h.add_argument("--list", action="store_true", help="print the database files the table lacks, one per line")
    h.add_argument("--have", type=Path, help="names of the files the database holds (--list)")
    h.add_argument("--derived", type=Path, help="the files, downloaded")
    h.add_argument("--ref", type=Path, help=f"{REF_NAME} (release {REF_TAG})")
    a = ap.parse_args()
    if a.cmd == "ref":
        build_ref(a)
    else:
        sys.path.insert(0, str(HERE.parents[0] / "verify"))
        hindcast(a)


if __name__ == "__main__":
    main()
