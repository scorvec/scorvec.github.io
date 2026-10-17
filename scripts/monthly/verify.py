#!/usr/bin/env python3
"""Score every issued month-ahead outlook whose target month is complete in the ERA5 archive.

    python scripts/monthly/verify.py

For each assets/monthly/data/outlook_<YYYYMM>.npz with a complete target month, per variable over 20-90N
(area-weighted): the anomaly correlation (uncentred) and RMSE of the outlook, the same for the trend component
alone (the no-skill-beyond-warming reference), the share of the area where the outlook had the sign right, and the
ranked probability skill score of its tercile probabilities against climatology (1/3 each).

Benchmark: the C3S multi-system seasonal forecast issued in the same month (published around the 13th, two days
before this outlook), at its lead month 2, which is the same target month. Its ensemble-mean anomalies per system
(2 m temperature and precipitation; C3S has no 500 hPa height in its anomaly set) are the `is_anom_*` tables that
c3s.yml keeps in the `c3s-ref-v1` release; they are averaged with equal weight, moved from the C3S 1993-2016 base to
1991-2020 with ERA5, regridded onto the outlook's 3° cells and cached as bench_c3s_<YYYYMM>.npz. The benchmark is a
yardstick only: nothing from it enters the outlook.

Writes assets/monthly/data/verification.json (one row per target month, newest first) and
assets/monthly/verify_<YYYYMM>.webp (outlook beside what happened).
"""

from __future__ import annotations

import io
import json
import sys
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mo_grid as G
import mo_method as M
import render as R

DATA = R.DATA
REPO = HERE.parents[1]
RELEASE = "https://github.com/scorvec/scorvec.github.io/releases/download/c3s-ref-v1"
C3S_FIELDS = {"t2m": "t2m", "prcp": "tp"}  # outlook variable -> C3S field key
C3S_SYSTEMS = ("ecmwf_51", "ukmo_610", "meteo_france_9", "dwd_22", "ncep_2", "eccc_4", "eccc_5", "bom_2")


def weights() -> np.ndarray:
    w = np.cos(np.deg2rad(G.LAT_NH))[:, None] * np.ones(len(G.LON))
    w[G.LAT_NH < 20] = 0
    return w.ravel()


def scores(f: np.ndarray, o: np.ndarray, w: np.ndarray) -> dict:
    accv = float((w * f * o).sum() / np.sqrt((w * f * f).sum() * (w * o * o).sum() + 1e-30))
    rmse = float(np.sqrt((w * (f - o) ** 2).sum() / w.sum()))
    sign = float((w * (np.sign(f) == np.sign(o))).sum() / w.sum())
    return {"acc": round(accv, 3), "rmse": round(rmse, 3), "sign": round(sign, 3)}


def rpss(pb: np.ndarray, pa: np.ndarray, o: np.ndarray, lo: np.ndarray, hi: np.ndarray, w: np.ndarray) -> float:
    cat = np.greater_equal(o, lo).astype(int) + np.greater(o, hi)  # 0 below, 1 near, 2 above normal
    rps = (pb - (cat == 0)) ** 2 + ((1 - pa) - (cat <= 1)) ** 2
    ref = (1 / 3 - (cat == 0)) ** 2 + (2 / 3 - (cat <= 1)) ** 2
    return round(float(1 - (w * rps).sum() / (w * ref).sum()), 3)


def month_anoms(A: M.Archive, v: str, m: int) -> np.ndarray:
    years = A.years
    x = M.month_series(A.cal[v], years, m, int(years[0]))[:, M.nh_rows(v)].reshape(len(years), -1)
    return M.clim_anom(x, years)


def observed(A: M.Archive, y: int, m: int, variables) -> dict | None:
    if y > A.years[-1]:
        return None
    out = {}
    for v in variables:
        a = month_anoms(A, v, m)[y - A.years[0]]
        if not np.isfinite(a).all():
            return None
        out[v] = a
    return out


def c3s_systems() -> list[str]:
    p = REPO / "assets" / "sst" / "data" / "c3s_maps.json"
    try:
        ids = [m["id"] for m in json.loads(p.read_text())["models"] if m["id"] != "mmm"]
        return ids or list(C3S_SYSTEMS)
    except Exception:  # noqa: BLE001  the registry file is optional here
        return list(C3S_SYSTEMS)


def c3s_benchmark(A: M.Archive, y: int, m: int) -> dict | None:
    """The C3S multi-system mean anomaly for target (y, m) at lead month 2, on the outlook grid and 1991-2020 base,
    cached in assets/monthly/data/bench_c3s_<YYYYMM>.npz. None while no system's table is in the release yet."""
    tag = f"{y}{m:02d}"
    cache = DATA / f"bench_c3s_{tag}.npz"
    if cache.exists():
        with np.load(cache) as f:
            return {k: f[k] for k in f.files}
    iy, im = (y, m - 1) if m > 1 else (y - 1, 12)
    issue = f"{iy}{im:02d}"
    out = {}
    for v, key in C3S_FIELDS.items():
        maps = []
        for sysid in c3s_systems():
            url = f"{RELEASE}/is_anom_{sysid}_{key}_{issue}.npz"
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    z = np.load(io.BytesIO(r.read()))
                    mean, lat, lon = z["mean"], z["lat"], z["lon"]
            except Exception:  # noqa: BLE001  a system that has not published (or been reduced) yet
                continue
            co = G.Coarsener(lat, lon)
            maps.append(co(mean[1].astype(np.float64), "nh"))  # lead month 2 = the target month
        if not maps:
            continue
        mm = np.mean(maps, 0).ravel()
        # C3S anomalies are against 1993-2016; ERA5 moves them to 1991-2020
        x = M.month_series(A.cal[v], A.years, m, int(A.years[0]))[:, M.nh_rows(v)].reshape(len(A.years), -1)
        b9316 = np.nanmean(x[(A.years >= 1993) & (A.years <= 2016)], 0)
        b9120 = np.nanmean(x[(A.years >= M.CLIM[0]) & (A.years <= M.CLIM[1])], 0)
        out[v] = (mm + b9316 - b9120).astype(np.float32)
        out[f"{v}_n"] = np.array(len(maps))
    if not out:
        return None
    if all(int(out.get(f"{v}_n", 0)) >= 4 for v in C3S_FIELDS):  # keep once most systems are in
        np.savez_compressed(cache, **out)
    return out


def figure(label: str, o: dict, obs: dict, sc: dict, variables, path: Path) -> None:
    nv = len(variables)
    fig = plt.figure(figsize=(12.4, 5.4 * nv + 0.6))
    h = 1.0 / nv
    for r, v in enumerate(variables):
        for c, (fld, name) in enumerate(((o[f"{v}_fc"], "Forecast"), (obs[v], "ERA5"))):
            ax = R.polar_ax(fig, [0.02 + c * 0.45, 1 - (r + 1) * h + 0.06 * h, 0.42, 0.78 * h])
            R.shade(ax, fld, v)
            t = f"{name}: {R.TITLE[v]}"
            if c == 0:
                t += f"\nACC {sc[v]['acc']:+.2f} (trend alone {sc[v]['acc_trend']:+.2f})"
                if "rpss" in sc[v]:
                    t += f", RPSS {sc[v]['rpss']:+.2f}"
                if "c3s_acc" in sc[v]:
                    t += f", C3S ACC {sc[v]['c3s_acc']:+.2f}"
            ax.set_title(t, fontsize=9.5, fontweight="bold", pad=4)
        cax = fig.add_axes([0.92, 1 - (r + 1) * h + 0.15 * h, 0.012, 0.6 * h])
        sm = plt.cm.ScalarMappable(cmap=R.CMAP[v], norm=matplotlib.colors.BoundaryNorm(R.LEVELS[v], 256, extend="both"))
        cb = fig.colorbar(sm, cax=cax, extend="both", ticks=R.LEVELS[v][::2])
        cb.ax.tick_params(labelsize=7)
        cb.set_label(R.UNIT[v], fontsize=8)
    fig.suptitle(f"{label}: the forecast against what happened", x=0.02, ha="left", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.004, R.CREDIT, ha="center", fontsize=7.5, color=R.MUTED)
    fig.savefig(path, dpi=80, facecolor="white", pil_kwargs={"quality": 82, "method": 6})
    plt.close(fig)


def main() -> int:
    A = M.load_archive()
    w = weights()
    rows = []
    for p in sorted(DATA.glob("outlook_[0-9]*.npz")):
        tag = p.stem.split("_")[1]
        y, m = int(tag[:4]), int(tag[4:])
        with np.load(p) as f:
            o = {k: f[k] for k in f.files}
        variables = [v for v in M.TARGETS if f"{v}_fc" in o]
        obs = observed(A, y, m, variables)
        if obs is None:
            continue
        summ = json.loads((DATA / f"outlook_{tag}.json").read_text())
        row = {"target": f"{y}-{m:02d}", "label": summ["target_label"], "issued": summ["issued"]}
        bench = c3s_benchmark(A, y, m)
        for v in variables:
            s = scores(o[f"{v}_fc"], obs[v], w)
            tr = scores(o[f"{v}_comp"][M.COMPONENTS.index("TR")], obs[v], w)
            s["acc_trend"], s["rmse_trend"] = tr["acc"], tr["rmse"]
            if f"{v}_p_below" in o and f"{v}_tlo" in o:
                s["rpss"] = rpss(o[f"{v}_p_below"], o[f"{v}_p_above"], obs[v], o[f"{v}_tlo"], o[f"{v}_thi"], w)
            if bench is not None and v in bench:
                b = scores(bench[v], obs[v], w)
                s["c3s_acc"], s["c3s_rmse"], s["c3s_n"] = b["acc"], b["rmse"], int(bench[f"{v}_n"])
            row[v] = s
        fp = R.ASSETS / f"verify_{tag}.webp"
        if not fp.exists():  # drawn once; the scores are recomputed every run (ERA5T is replaced by ERA5 later)
            figure(summ["target_label"], o, obs, row, variables, fp)
        rows.append(row)
        print(f"verified {tag}: " + ", ".join(f"{v} ACC {row[v]['acc']:+.2f}" for v in variables), flush=True)
    rows.sort(key=lambda r: r["target"], reverse=True)
    (DATA / "verification.json").write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
