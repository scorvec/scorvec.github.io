#!/usr/bin/env python3
"""Each station's recent forecasts and observations, for the station maps of verification.html: a click on a station
draws what it observed against every model's forecast for the same hours, at the map's lead.

From the verification database's surface pairs (sfc_<init>.parquet, the kept ones), the newest SERIES_DAYS days at
the station maps' leads (mverify_scores.MAP_LEADS): 2 m temperature and dew point, sea-level pressure and 10 m wind
speed (from the u/v pairs), the surface layers of the maps, one row per station, variable, valid time and lead, with
the observation and each model's forecast. The table is kept between nights (``--series``, the workflow's cache): a night reads only the runs it
lacks, then drops what has left the window. It is written as one JSON file per variable and tile of TILE degrees
(<out>/<variable>/<tile>.json, the frames store), so a click loads only its neighbourhood.

    python station_series.py --list --series series.parquet --have have.txt     # the sfc_ files a night needs
    python station_series.py --derived db --series series.parquet --out assets/verification/stn
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import mverify_scores as D
import numpy as np
import pandas as pd

SERIES_DAYS = 45
LEADS = tuple(D.MAP_LEADS)
VARS = {"t2m": "°C", "td2m": "°C", "msl": "hPa", "wind10": "m/s"}
TILE = 10  # degrees
KEY = ["station_id", "variable", "valid_time", "lead_h"]


def tile_of(lat: float, lon: float) -> str:
    """The tile a station is filed under: the corner of its TILE-degree square (verification.html finds it so too)."""
    return f"{int(np.floor(lat / TILE) * TILE)}_{int(np.floor(lon / TILE) * TILE)}"


def run_rows(path: Path) -> pd.DataFrame:
    """One run's kept pairs at LEADS: (station, variable, valid time, lead) -> its position, the observation and each
    model's forecast."""
    cols = ["model", "lead_h", "valid_time", "station_id", "lat", "lon", "variable", "fcst", "obs"]
    p = pd.read_parquet(path, columns=cols, filters=[("lead_h", "in", list(LEADS)), ("kept", "==", True)])
    p = p.astype({"model": str, "variable": str, "station_id": str})
    k = ["model", "lead_h", "valid_time", "station_id"]
    u, v = p[p["variable"] == "u10"], p[p["variable"] == "v10"]
    w = u.merge(v[[*k, "fcst", "obs"]], on=k, suffixes=("_u", "_v"))
    w = w.assign(variable="wind10", fcst=np.hypot(w["fcst_u"], w["fcst_v"]), obs=np.hypot(w["obs_u"], w["obs_v"]))[cols]
    p = pd.concat([p[p["variable"].isin([*VARS])], w], ignore_index=True)
    wide = p.pivot_table(index=KEY, columns="model", values="fcst", observed=True)
    return wide.join(p.groupby(KEY)[["obs", "lat", "lon"]].first()).reset_index()


def runs_of(series: pd.DataFrame) -> set[str]:
    return set((series["valid_time"] - pd.to_timedelta(series["lead_h"], unit="h")).dt.strftime("%Y%m%d%H"))


def wanted(have: set[str], series: pd.DataFrame | None, today: pd.Timestamp) -> list[str]:
    """The database's runs inside the window that the table lacks."""
    first = today - pd.Timedelta(days=SERIES_DAYS + max(LEADS) // 24)
    done = runs_of(series) if series is not None else set()
    runs = sorted(n[4:14] for n in have if n.startswith("sfc_"))
    return [f"sfc_{r}.parquet" for r in runs if pd.to_datetime(r, format="%Y%m%d%H") >= first and r not in done]


def update(series: pd.DataFrame | None, files: list[Path], today: pd.Timestamp) -> pd.DataFrame:
    parts = ([series] if series is not None else []) + [run_rows(f) for f in files]
    s = pd.concat(parts, ignore_index=True).drop_duplicates(KEY, keep="last")
    return s[s["valid_time"] >= today - pd.Timedelta(days=SERIES_DAYS)].reset_index(drop=True)


def tiles(series: pd.DataFrame, out: Path) -> int:
    """One JSON per variable and tile: each station's valid times (hours after the file's t0), observations and each
    model's forecast at each lead, on the same times (null where a run is missing)."""
    if out.exists():
        shutil.rmtree(out)
    models = sorted(c for c in series.columns if c not in [*KEY, "obs", "lat", "lon"])
    t0 = series["valid_time"].min().floor("D")
    n = 0
    for var, unit in VARS.items():
        s = series[series["variable"] == var]
        pos = s.groupby("station_id")[["lat", "lon"]].first()
        stations: dict[str, dict] = {}
        for sid, g in s.groupby("station_id"):
            times = np.sort(g["valid_time"].unique())
            obs = g.groupby("valid_time")["obs"].first().reindex(times)
            fc = {}
            for lead, gl in g.groupby("lead_h"):
                on = gl.set_index("valid_time").reindex(times)
                fc[str(lead)] = {m: _round(on[m]) for m in models if on[m].notna().any()}
            la, lo = pos.loc[sid]
            stations.setdefault(tile_of(la, lo), {})[sid] = {
                "lat": round(float(la), 3),
                "lon": round(float(lo), 3),
                "t": ((pd.DatetimeIndex(times) - t0) / pd.Timedelta("1h")).astype(int).tolist(),
                "o": _round(obs),
                "f": fc,
            }
        for tile, st in stations.items():
            f = out / var / f"{tile}.json"
            f.parent.mkdir(parents=True, exist_ok=True)
            body = {"variable": var, "unit": unit, "t0": f"{t0:%Y-%m-%dT%H}Z", "leads": list(LEADS), "stations": st}
            f.write_text(json.dumps(body, separators=(",", ":")))
            n += 1
    return n


def _round(s: pd.Series) -> list:
    return [None if not np.isfinite(v) else round(float(v), 1) for v in s.to_numpy(dtype=float)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series", type=Path, required=True, help="the table kept between nights")
    ap.add_argument("--list", action="store_true", help="print the sfc_ files the table lacks, one line")
    ap.add_argument("--have", type=Path, help="names of the files the database holds (--list)")
    ap.add_argument("--derived", type=Path, help="the sfc_ files, downloaded")
    ap.add_argument("--out", type=Path, help="the tiles' directory")
    a = ap.parse_args()
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    series = pd.read_parquet(a.series) if a.series.exists() else None
    if a.list:
        print(*wanted(set(a.have.read_text().split()), series, today))
        return
    series = update(series, sorted(a.derived.glob("sfc_*.parquet")), today)
    series.to_parquet(a.series, compression="zstd")
    print(f"station series: {series['station_id'].nunique()} stations, {len(runs_of(series))} runs, "
          f"{tiles(series, a.out)} tiles", flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
