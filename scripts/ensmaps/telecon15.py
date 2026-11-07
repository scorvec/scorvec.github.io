#!/usr/bin/env python3
"""The teleconnection indices of the days 1-15 forecasts (ensmaps_telecon.py), defined once: twelve patterns of the GEFS
and GEPS pages (IDS: the Arctic Oscillation from sea-level pressure; CPC's NAO, PNA, EA, WP, EP/NP, EA/WR, SCA, TNH and
POL and PSL's EPO and WPO from 500 hPa height), their projectors applied to the field at one synoptic hour, and the
table of every archived run's indices against ERA5's that calibrates them.

  ref       The reference, built once in Actions (telecon15.yml; release REF_TAG, telecon15_ref.npz):
            - the projectors of release gefs-ref-v1 (gefs_patterns_ref.npz; method and validation in
              scripts/geps/build/build_telecon_patterns.py): CPC's patterns the month-specific least-squares projector
              of the regression on CPC's monthly indices (the nine together), applied to 500 hPa height anomalies over
              std_month (the sd of the calendar month's monthly anomalies) weighted by sqrt(cos lat); EPO and WPO PSL's
              box differences in metres; the AO the leading EOF of sea-level pressure anomalies (hPa) north of 20N
              over 1991-2020, weighted by sqrt(cos lat).
              The weight is folded into the projector, which is cut to the 2.5 deg rows 20-90N (latlon.G25N), with
              std_month on the same rows: the build stops if any weight south of 20N would be lost.
            - sigma, the seasonal sd of each daily index (1991-2020, within 15 days of the day of year), and the months
              CPC defines it in, on the 366-slot calendar (latlon.slot366), 29 February the mean of 28 February and
              1 March as the GEFS page reads it (gefs_reforecast.at_key).
            - the 500 hPa normal: ERA5 1991-2020 at 00Z of release verify-clim-z500-v1 (1.5 deg, z500_acc.clim: the
              normal of the hindcast's ERA5 and of the 500 hPa anomaly correlations) remapped conservatively onto the
              same grid; at 12Z the same plus the tide between the two hours, 12Z less 00Z in WeatherBench 2's ERA5
              climatology by hour of day (1990-2019, 1.5 deg) fitted with a mean and three annual harmonics, so a 12Z
              run's field is compared with a normal of its own hour.
            - the sea-level pressure normal: ERA5 1991-2020 at 00Z and at 12Z (z500_acc.py clim --field msl, the same
              construction at each hour, built with the reference and kept in its release), remapped the same way.
            The build checks the cut projectors against gefs_patterns.Ref.index on random global fields and prints the
            tide's part of each 500 hPa index.
  index     (field - normal of its hour and day) [/ std_month] . projector / sigma, in seasonal sd of the daily index;
            NaN in a month CPC does not define a pattern (EP/NP: not in December; TNH: December to February only).
  hindcast  The table the calibration reads (the Actions cache, telecon15.yml), a pair of files for each field (FC,
            OBS): for every archived 00Z run of the database (scripts/verify/verifydb.sh) the ensemble-mean index at
            each lead, from z500_acc.SINCE the 500 hPa ones as the mean of the members' indices (the index is linear
            in the field, so it is the index of the members' mean field, ens-z500m-<m>_<init>.npz) and from
            z500_acc.MSL_SINCE the AO from the ensemble-mean sea-level pressure (ens-msl-<m>_); ERA5's indices at 00Z
            of each day (era5-z500_, from the database's 1.5 deg grid onto 2.5 deg; era5-msl_, on 2.5 deg). A pass
            reads only the runs and days it lacks; ``tables`` puts the fields' indices side by side.

    python telecon15.py ref --patterns gefs_patterns_ref.npz --clim clim_z500_1p5.npz --clim-msl clim_msl_1p5.npz \
        --out telecon15_ref.npz
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

REF_TAG = "telecon15-ref-v3"  # v1: the PNA, EPO and WPO only; v2: the eleven 500 hPa patterns
REF_NAME = "telecon15_ref.npz"
IDS = (
    "nao",
    "ao",
    "pna",
    "epo",
    "wpo",
    "ea",
    "wp",
    "epnp",
    "eawr",
    "sca",
    "tnh",
    "pol",
)  # gefs_patterns.ORDER's 500 hPa ones and the AO
# what each index projects: 500 hPa height (m) or sea-level pressure (hPa)
FIELD = dict.fromkeys(IDS, "z500") | {"ao": "msl"}
PATTERN_FIELD = {"z500": "z500", "msl": "slp"}  # gefs_patterns' names of the fields
HOURS = (0, 12)  # the synoptic hours a forecast is read at: a 00Z run's 00Z, a 12Z run's 12Z
CLIM_HOUR = 0  # the hour of verify-clim-z500-v1's normal
LEADS = tuple(range(24, 361, 24))  # hours: 00Z of each day to day 15, the database's ensemble leads (ens_verify.LEADS)
WB2_CLIM = (
    "gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_240x121_equiangular_with_poles_conservative.zarr"
)
TIDE_HARMONICS = 3
G = 9.80665
FC = {"z500": "fc.parquet", "msl": "fc_msl.parquet"}  # the hindcast table's files, by field
OBS = {"z500": "obs.parquet", "msl": "obs_msl.parquet"}


# ── the reference ─────────────────────────────────────────────────────────────────────────────────────────────────────
class Ref:
    """telecon15_ref.npz: ids and the field each projects (field), proj (index, month, row, col; sqrt(cos lat) folded
    in), std_month (month, row, col), standardize (index), sigma (index, slot), defined (index, month), the normals
    normal (500 hPa height) and normal_msl (slot, hour, row, col), meta."""

    def __init__(self, path: Path):
        with np.load(path) as z:
            self.ids = [str(s) for s in z["ids"]]
            self.field = [str(f) for f in z["field"]]
            self.P = z["proj"].astype(np.float64)
            self.std = z["std_month"].astype(np.float64)
            self.standardize = z["standardize"].astype(bool)
            self.sigma = z["sigma"].astype(np.float64)
            self.defined = z["defined"].astype(bool)
            self.normal = {"z500": z["normal"], "msl": z["normal_msl"]}
            self.meta = json.loads(str(z["meta"]))

    def anomaly(self, x: np.ndarray, times: pd.DatetimeIndex, field: str = "z500") -> np.ndarray:
        """(n, row, col) fields (500 hPa height in m, or sea-level pressure in hPa) at `times`, each 00 or 12 UTC, less
        the normal of that hour and day."""
        if not np.isin(times.hour, HOURS).all():
            raise ValueError(f"the normal has the hours {HOURS} only, not {sorted(set(times.hour))}")
        h = np.searchsorted(HOURS, times.hour)
        return np.asarray(x, np.float64) - self.normal[field][slot366(times), h]

    def index(self, anom: dict[str, np.ndarray], times: pd.DatetimeIndex) -> np.ndarray:
        """(n, index) in seasonal sd of the daily index, of anomalies {field: (n, row, col)} valid at `times`; NaN for
        an index whose field is not given."""
        mo, s = times.month.to_numpy() - 1, slot366(times)
        out = np.full((len(times), len(self.ids)), np.nan)
        for k, f in enumerate(self.field):
            if f not in anom:
                continue
            a = anom[f] / self.std[mo] if self.standardize[k] else anom[f]
            with np.errstate(invalid="ignore", divide="ignore"):  # sigma is NaN where CPC defines no pattern
                out[:, k] = np.einsum("nij,nij->n", a, self.P[k][mo]) / self.sigma[k, s]
            out[~self.defined[k, mo], k] = np.nan
        return out


def tide(path: str = WB2_CLIM) -> np.ndarray:
    """(366, HOURS, 121, 240) on the 1.5 deg grid, 90N first, from 0E: ERA5's 500 hPa height at each of HOURS less that
    at CLIM_HOUR, by slot of the 366-day calendar, as a mean and TIDE_HARMONICS annual harmonics. WeatherBench 2 keys its
    climatology by the raw day of the year; slot s is read as its day s + 1 (a day apart after February in common
    years, nothing in a curve this smooth)."""
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
    hour = list(z["hour"].values)
    d = v[:, [hour.index(h) for h in HOURS]] - v[:, [hour.index(CLIM_HOUR)]]
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
        if gp.pat[sid]["field"] != PATTERN_FIELD[FIELD[sid]]:
            raise SystemExit(f"{sid}: the pattern projects {gp.pat[sid]['field']}, not {FIELD[sid]}")
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
        field=np.array([FIELD[s] for s in IDS]),
        proj=np.array(proj, np.float32),
        std_month=std.astype(np.float32),
        standardize=np.array([bool(gp.pat[s]["standardize"]) for s in IDS]),
        sigma=np.array(sigma, np.float64),
        defined=np.array(defined),
    )
    clim = np.load(a.clim)
    if clim["z500"].shape != (366, *G15.shape):
        raise SystemExit(f"{a.clim}: not the 366 slots of the 1.5 deg normal")
    normal = np.stack([remap(c, G15.lat, G15.lon, G25N) for c in clim["z500"]]).astype(np.float64)
    td25 = np.stack([[remap(t, G15.lat, G15.lon, G25N) for t in s] for s in tide()])
    ref["normal"] = (normal[:, None] + td25).astype(np.float32)
    msl = np.load(a.clim_msl)
    if msl["msl"].shape != (366, len(HOURS), *G15.shape) or list(msl["hours"]) != list(HOURS):
        raise SystemExit(f"{a.clim_msl}: not the 366 slots of the 1.5 deg normal at {HOURS} UTC")
    if not 1005 < np.median(msl["msl"]) < 1020:
        raise SystemExit(f"{a.clim_msl}: median {np.median(msl['msl']):.0f}, not sea-level pressure in hPa")
    ref["normal_msl"] = np.stack([[remap(c, G15.lat, G15.lon, G25N) for c in s] for s in msl["msl"]]).astype(np.float32)
    sha = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (a.patterns, a.clim, a.clim_msl)}
    meta = {
        "patterns": {s: gp.pat[s] for s in IDS},
        "hours": list(HOURS),
        "sources": {
            "patterns": f"gefs-ref-v1/{a.patterns.name} sha256 {sha[a.patterns.name]}",
            "normal": f"verify-clim-z500-v1/{a.clim.name} sha256 {sha[a.clim.name]}",
            "tide": f"{WB2_CLIM}, each hour less {CLIM_HOUR:02d} UTC, mean + {TIDE_HARMONICS} annual harmonics",
            "normal_msl": f"{REF_TAG}/{a.clim_msl.name} sha256 {sha[a.clim_msl.name]} (z500_acc.py clim --field msl)",
        },
    }
    ref["meta"] = np.array(json.dumps(meta))
    np.savez_compressed(a.out, **ref)
    check_ref(Ref(a.out), gp, GR, td25)
    print(f"reference: {a.out} ({a.out.stat().st_size / 1e6:.1f} MB)", flush=True)


def check_ref(r: Ref, gp, GR, td25: np.ndarray) -> None:
    """The cut projectors give gefs_patterns.Ref.index's numbers on random global fields, every day of a common and a
    leap year; the tide's size and its part of each 500 hPa index, month by month; the sea-level pressure normal's
    range."""
    rng = np.random.default_rng(1)
    t = pd.date_range("2023-01-01", "2024-12-31", freq="D")
    anom = rng.normal(0.0, 80.0, (len(t), *gp.std.shape[1:]))
    mine = r.index(dict.fromkeys(PATTERN_FIELD, anom[:, : G25N.lat.size]), t)
    keys = GR.noleap_doy(t.values.astype("datetime64[D]"))
    ref = np.stack([gp.index(s, anom, t.month, keys) for s in r.ids], 1)
    same = np.isfinite(mine) == np.isfinite(ref)  # undefined (NaN) in the same months
    err = float(np.nanmax(np.abs(mine - ref)))
    print(f"check: the cut projectors against gefs_patterns.Ref.index, largest difference {err:.2e} sd, "
          f"{int((~np.isfinite(ref)).sum())} undefined index-days alike: {same.all()}", flush=True)  # fmt: skip
    if not (same.all() and err <= 1e-6):
        raise SystemExit("the cut projectors do not reproduce the GEFS page's indices")
    days = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    for h, hour in enumerate(HOURS):
        if hour == CLIM_HOUR:
            continue
        part = r.index({"z500": td25[slot366(days), h]}, days)
        z500 = [s for s, f in zip(r.ids, r.field, strict=True) if f == "z500"]
        by = pd.DataFrame(part, index=days, columns=r.ids)[z500].groupby(days.month).agg(lambda x: np.abs(x).max())
        print(f"check: {hour:02d} UTC less {CLIM_HOUR:02d} UTC, largest {np.abs(td25[:, h]).max():.1f} m 20-90N; its "
              f"part of each index, largest |sd| in each month:\n{by.round(3)}", flush=True)  # fmt: skip
    n = r.normal["msl"]
    print(f"check: the sea-level pressure normal 20-90N, {n.min():.1f} to {n.max():.1f} hPa; 12 less 00 UTC at most "
          f"{np.abs(n[:, 1] - n[:, 0]).max():.1f} hPa", flush=True)  # fmt: skip


# ── the hindcast table ────────────────────────────────────────────────────────────────────────────────────────────────
# each field's database files: a run's (the members on the patterns' grid; the ensemble mean) and ERA5's
FILES = {
    "z500": (r"ens-z500m-(\w+)_(\d{10})\.npz", r"era5-z500_(\d{10})\.npz"),
    "msl": (r"ens-msl-(\w+)_(\d{10})\.npz", r"era5-msl_(\d{10})\.npz"),
}


def _read(table: Path, field: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    fc, obs = table / FC[field], table / OBS[field]
    return (pd.read_parquet(fc) if fc.exists() else None, pd.read_parquet(obs) if obs.exists() else None)


def tables(table: Path | None) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """(fc, obs) with every index side by side, NaN where a run or day lacks its field (the AO before MSL_SINCE); None
    before the table holds 500 hPa indices."""
    fc, obs = _read(table, "z500") if table else (None, None)
    if fc is None or obs is None:
        return None, None
    mf, mo = _read(table, "msl")
    if mf is not None:
        fc = fc.merge(mf, on=["model", "init", "lead_h"], how="left")
    if mo is not None:
        obs = obs.merge(mo, on="valid", how="left")
    return tuple(x.assign(**{s: np.nan for s in IDS if s not in x}) for x in (fc, obs))


def wanted(have: set[str], table: Path) -> list[str]:
    """The database's files of each field the table lacks: the runs from each model's first (z500_acc.SINCE, MSL_SINCE)
    and ERA5's days."""
    import z500_acc as Z

    code = {c: m for m, c in Z.E.ENSEMBLES.items()}
    out = []
    for field, (run, era) in FILES.items():
        fc, obs = _read(table, field)
        since = Z.SINCE if field == "z500" else Z.MSL_SINCE
        done = set() if fc is None else set(zip(fc["model"], fc["init"].dt.strftime("%Y%m%d%H"), strict=True))
        days = set() if obs is None else set(obs["valid"].dt.strftime("%Y%m%d%H"))
        for n in sorted(have):
            if m := re.fullmatch(run, n):
                model = code.get(m[1])
                if model in since and m[2] >= since[model] and (model, m[2]) not in done:
                    out.append(n)
            elif (m := re.fullmatch(era, n)) and m[1] not in days:
                out.append(n)
    return out


def _on_grid(d, path: Path) -> None:
    if not (np.allclose(d["lat"], G25N.lat) and np.allclose(d["lon"], G25N.lon)):
        raise RuntimeError(f"{path.name}: not on the 2.5 deg grid 20-90N")


def run_rows(r: Ref, path: Path, model: str, field: str) -> pd.DataFrame:
    """One run's ensemble-mean indices of one field at each lead: of 500 hPa height the index of its members' mean
    field, of sea-level pressure that of the ensemble-mean field."""
    with np.load(path) as d:
        _on_grid(d, path)
        leads = d["lead_h"]
        x = d["z"].astype(np.float64).mean(1) if field == "z500" else d["msl"].astype(np.float64)
        n = {"n_members": np.int16(d["z"].shape[1])} if field == "z500" else {}
    init = pd.to_datetime(path.stem.split("_")[1], format="%Y%m%d%H")
    t = pd.DatetimeIndex(init + pd.to_timedelta(leads, unit="h"))
    v = r.index({field: r.anomaly(x, t, field)}, t)
    return pd.DataFrame(
        {"model": model, "init": init, "lead_h": leads.astype(np.int16)} | n
        | {s: v[:, k].astype(np.float32) for k, s in enumerate(r.ids) if r.field[k] == field}
    )  # fmt: skip


def era5_rows(r: Ref, paths: list[Path], field: str) -> pd.DataFrame:
    """ERA5's indices of one field at 00Z of each day: its 500 hPa height remapped from the database's 1.5 deg grid onto
    the patterns' grid, its sea-level pressure there already."""
    if not paths:
        return pd.DataFrame()
    t = pd.DatetimeIndex([pd.to_datetime(p.stem.split("_")[1], format="%Y%m%d%H") for p in paths])
    if field == "z500":
        x = np.stack([remap(np.load(p)["z"], G15.lat, G15.lon, G25N) for p in paths])
    else:
        x = np.stack([np.load(p)["msl"] for p in paths])
    v = r.index({field: r.anomaly(x, t, field)}, t)
    return pd.DataFrame(
        {"valid": t} | {s: v[:, k].astype(np.float32) for k, s in enumerate(r.ids) if r.field[k] == field}
    )


def hindcast(a) -> None:
    table = a.table
    if a.list:
        print("\n".join(wanted(set(a.have.read_text().split()), table)))
        return
    import z500_acc as Z

    if tuple(Z.E.LEADS) != LEADS:
        raise SystemExit(f"the database's leads {list(Z.E.LEADS)} are not telecon15.LEADS")
    r = Ref(a.ref)
    code = {c: m for m, c in Z.E.ENSEMBLES.items()}
    table.mkdir(parents=True, exist_ok=True)
    for field, (run, era) in FILES.items():
        fc, obs = _read(table, field)
        names = sorted(p.name for p in a.derived.glob("*_*.npz"))
        runs = [
            run_rows(r, a.derived / n, code[m[1]], field) for n in names if (m := re.fullmatch(run, n)) and m[1] in code
        ]
        days = era5_rows(r, [a.derived / n for n in names if re.fullmatch(era, n)], field)
        if runs:
            fc = pd.concat([x for x in (fc, *runs) if x is not None], ignore_index=True)
            fc = fc.drop_duplicates(["model", "init", "lead_h"], keep="last").sort_values(["model", "init", "lead_h"])
            fc.reset_index(drop=True).to_parquet(table / FC[field], compression="zstd")
        if not days.empty:
            obs = pd.concat([x for x in (obs, days) if x is not None], ignore_index=True)
            obs = obs.drop_duplicates("valid", keep="last").sort_values("valid").reset_index(drop=True)
            obs.to_parquet(table / OBS[field], compression="zstd")
        held = {m: int(g["init"].nunique()) for m, g in fc.groupby("model")} if fc is not None else {}
        print(f"hindcast {field}: {len(runs)} runs and {len(days)} ERA5 days added; the table holds runs {held}, "
              f"{0 if obs is None else len(obs)} ERA5 days", flush=True)  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("ref", help="build the reference (once)")
    b.add_argument("--patterns", type=Path, required=True, help="gefs_patterns_ref.npz (release gefs-ref-v1)")
    b.add_argument("--clim", type=Path, required=True, help="clim_z500_1p5.npz (release verify-clim-z500-v1)")
    b.add_argument("--clim-msl", type=Path, required=True, help="clim_msl_1p5.npz (z500_acc.py clim --field msl)")
    b.add_argument("--out", type=Path, required=True)
    h = sub.add_parser("hindcast", help="add the database's new runs and ERA5 days to the table")
    h.add_argument("--table", type=Path, required=True, help="the table's directory (FC, OBS)")
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
