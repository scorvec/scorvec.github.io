#!/usr/bin/env python3
"""Score every issued month-ahead outlook whose target month is complete in the ERA5 archive.

    python scripts/monthly/verify.py

For each assets/monthly/data/outlook_<YYYYMM>.npz with a complete target month: the anomaly correlation (uncentred,
20-90N, area-weighted) and RMSE of the outlook, the same for the trend component alone (the no-skill-beyond-warming
reference), and the share of the area where the outlook had the sign right. Writes assets/monthly/data/
verification.json (one row per target month, newest first) and assets/monthly/verify_<YYYYMM>.webp (outlook beside
what happened).
"""

from __future__ import annotations

import json
import sys
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


def weights() -> np.ndarray:
    w = np.cos(np.deg2rad(G.LAT_NH))[:, None] * np.ones(len(G.LON))
    w[G.LAT_NH < 20] = 0
    return w.ravel()


def scores(f: np.ndarray, o: np.ndarray, w: np.ndarray) -> dict:
    accv = float((w * f * o).sum() / np.sqrt((w * f * f).sum() * (w * o * o).sum() + 1e-30))
    rmse = float(np.sqrt((w * (f - o) ** 2).sum() / w.sum()))
    sign = float((w * (np.sign(f) == np.sign(o))).sum() / w.sum())
    return {"acc": round(accv, 3), "rmse": round(rmse, 3), "sign": round(sign, 3)}


def observed(A: M.Archive, y: int, m: int) -> dict | None:
    years = A.years
    if y > years[-1]:
        return None
    out = {}
    for v in M.TARGETS:
        x = M.month_series(A.cal[v], years, m, int(years[0])).reshape(len(years), -1)
        a = M.clim_anom(x, years)[y - years[0]]
        if not np.isfinite(a).all():
            return None
        out[v] = a
    return out


def figure(tag: str, label: str, o: dict, obs: dict, sc: dict, path: Path) -> None:
    fig = plt.figure(figsize=(12.4, 11.2))
    for r, v in enumerate(M.TARGETS):
        for c, (fld, name) in enumerate(((o[f"{v}_fc"], "Forecast"), (obs[v], "ERA5"))):
            ax = R.polar_ax(fig, [0.02 + c * 0.49, 0.53 - r * 0.47, 0.46, 0.40])
            R.shade(ax, fld, v)
            t = f"{name}: {R.TITLE[v]}"
            if c == 0:
                t += f"\nACC {sc[v]['acc']:+.2f} (trend alone {sc[v]['acc_trend']:+.2f}), sign right {100 * sc[v]['sign']:.0f}%"
            ax.set_title(t, fontsize=10, fontweight="bold", pad=4)
        cax = fig.add_axes([0.93, 0.56 - r * 0.47, 0.012, 0.34])
        sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=matplotlib.colors.BoundaryNorm(R.LEVELS[v], 256, extend="both"))
        cb = fig.colorbar(sm, cax=cax, extend="both", ticks=R.LEVELS[v][::2])
        cb.ax.tick_params(labelsize=7)
        cb.set_label(R.UNIT[v], fontsize=8)
    fig.text(0.02, 0.975, f"{label}: the forecast against what happened", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.012, R.CREDIT, ha="center", fontsize=7.5, color=R.MUTED)
    fig.savefig(path, dpi=85, facecolor="white", pil_kwargs={"quality": 82, "method": 6})
    plt.close(fig)


def main() -> int:
    A = M.load_archive()
    w = weights()
    rows = []
    for p in sorted(DATA.glob("outlook_[0-9]*.npz")):
        tag = p.stem.split("_")[1]
        y, m = int(tag[:4]), int(tag[4:])
        obs = observed(A, y, m)
        if obs is None:
            continue
        with np.load(p) as f:
            o = {k: f[k] for k in f.files}
        summ = json.loads((DATA / f"outlook_{tag}.json").read_text())
        row = {"target": f"{y}-{m:02d}", "label": summ["target_label"], "issued": summ["issued"]}
        for v in M.TARGETS:
            s = scores(o[f"{v}_fc"], obs[v], w)
            tr = scores(o[f"{v}_comp"][M.COMPONENTS.index("TR")], obs[v], w)
            s["acc_trend"], s["rmse_trend"] = tr["acc"], tr["rmse"]
            row[v] = s
        fp = R.ASSETS / f"verify_{tag}.webp"
        if not fp.exists():  # drawn once; the scores are recomputed every run (ERA5T is replaced by ERA5 later)
            figure(tag, summ["target_label"], o, obs, row, fp)
        rows.append(row)
        print(f"verified {tag}: " + ", ".join(f"{v} ACC {row[v]['acc']:+.2f}" for v in M.TARGETS), flush=True)
    rows.sort(key=lambda r: r["target"], reverse=True)
    (DATA / "verification.json").write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
