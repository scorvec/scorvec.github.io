"""
Plot RMM phase-space diagram and amplitude time series for an AIFS ensemble.

Usage:
    python src/plot.py --rmm data/aifs/rmm_20240601_00z.nc \
                       --obs-rmm data/era5/obs_rmm.nc \
                       --out plots/rmm_20240601_00z.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import xarray as xr


# Phase number centres (degrees, counterclockwise from +RMM1 axis).
# Phase 1 sits in the left sector (West. Hem. & Africa), going CCW to 8.
PHASE_ANGLES = {
    1: 202.5,
    2: 247.5,
    3: 292.5,
    4: 337.5,
    5:  22.5,
    6:  67.5,
    7: 112.5,
    8: 157.5,
}

# Region labels: (text, x, y, rotation)
# Placed along the four edges of the diagram.
REGION_LABELS = [
    ("West. Hem.\nand Africa",  -4.6,  0.0,  90),
    ("Indian Ocean",             0.0, -4.6,   0),
    ("Maritime\nContinent",      4.6,  0.0, -90),
    ("Western Pacific",          0.0,  4.6,   0),
]


def draw_phase_wheel(ax: plt.Axes, radius: float = 4.0,
                     amp_rings=(1, 2, 3)) -> None:
    """Draw the 8-phase octant lines, labels, and amplitude reference circles.

    Concentric dashed rings at ``amp_rings`` let amplitude be read directly off
    the diagram as radial distance from the origin (replacing a separate
    amplitude time-series panel).
    """
    ax.set_aspect("equal")

    # Horizontal and vertical axes
    ax.axhline(0, color="0.5", lw=0.7, zorder=1)
    ax.axvline(0, color="0.5", lw=0.7, zorder=1)

    # Diagonal sector boundaries (full width, dashed)
    for angle in (45, 135):
        rad = np.deg2rad(angle)
        ax.plot([-radius, radius], [-radius * np.tan(rad), radius * np.tan(rad)],
                color="0.6", lw=0.6, ls="--", zorder=1)

    # Phase-number labels inside sectors
    for phase, centre_angle in PHASE_ANGLES.items():
        rad = np.deg2rad(centre_angle)
        r_label = radius * 0.78
        ax.text(r_label * np.cos(rad), r_label * np.sin(rad),
                str(phase), ha="center", va="center",
                fontsize=10, color="0.35", fontweight="bold", zorder=2)

    # Region labels along axes edges
    for text, x, y, rot in REGION_LABELS:
        ax.text(x, y, text, ha="center", va="center",
                fontsize=8, color="0.4", style="italic",
                rotation=rot, rotation_mode="anchor")

    # Amplitude reference circles (radial distance = RMM amplitude)
    theta = np.linspace(0, 2 * np.pi, 200)
    for r in amp_rings:
        ax.plot(r * np.cos(theta), r * np.sin(theta),
                color="0.45", ls="--", lw=0.8, zorder=1, alpha=0.6)
        # label each ring in the lower-right (usually the emptiest sector)
        ax.text(r * np.cos(np.deg2rad(-45)), r * np.sin(np.deg2rad(-45)),
                str(r), ha="center", va="center", fontsize=8, color="0.4",
                zorder=2, bbox=dict(boxstyle="round,pad=0.05", fc="white",
                                    ec="none", alpha=0.7))

    lim = radius + 0.3
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("RMM1", fontsize=11)
    ax.set_ylabel("RMM2", fontsize=11)


def plot_rmm(
    rmm: xr.Dataset,
    obs: xr.Dataset | None = None,
    truth: xr.Dataset | None = None,
    out_path: Path | None = None,
) -> plt.Figure:
    date  = rmm.attrs.get("init_date", "")
    rtime = rmm.attrs.get("init_time", "00")
    init  = pd.Timestamp(f"{date}T{rtime}:00") if date else None

    # Observed track day-offsets relative to init (negative → before init).
    has_obs = obs is not None and obs.sizes.get("time", 0) > 0
    if has_obs:
        obs_off = (pd.to_datetime(obs["time"].values) - init).days
        obs_src = obs.attrs.get("source", "analysis")

    # Archived AIFS-analysis 'truth' history (zero-lag, self-consistent).
    has_truth = truth is not None and truth.sizes.get("time", 0) > 0
    if has_truth:
        truth_off = (pd.to_datetime(truth["time"].values) - init).days

    rmm1 = rmm["rmm1"].values   # (member, lead_day)
    rmm2 = rmm["rmm2"].values
    lead_days = rmm["lead_day"].values.astype(float)
    n_leads = len(lead_days)

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    ax.set_title(
        f"AIFS Ensemble RMM  —  Init: {date[:4]}-{date[4:6]}-{date[6:8]} {rtime}Z",
        fontsize=13, fontweight="bold",
    )

    # ── Phase-space diagram (amplitude = radial distance) ────────────────────
    draw_phase_wheel(ax)

    # Colour by lead time
    cmap = plt.get_cmap("plasma_r")
    norm = mcolors.Normalize(vmin=0, vmax=n_leads - 1)

    for m in range(rmm1.shape[0]):
        for i in range(n_leads - 1):
            color = cmap(norm(i))
            ax.plot(rmm1[m, i:i+2], rmm2[m, i:i+2],
                    color=color, alpha=0.25, lw=0.7, zorder=2)

    # Ensemble mean
    mean_rmm1 = rmm1.mean(axis=0)
    mean_rmm2 = rmm2.mean(axis=0)
    for i in range(n_leads - 1):
        color = cmap(norm(i))
        ax.plot(mean_rmm1[i:i+2], mean_rmm2[i:i+2],
                color=color, lw=2.2, zorder=4)

    # Spread ellipses every 5 days
    from matplotlib.patches import Ellipse
    for i in range(0, n_leads, 5):
        std1 = rmm1[:, i].std()
        std2 = rmm2[:, i].std()
        ell = Ellipse(
            xy=(mean_rmm1[i], mean_rmm2[i]),
            width=2 * std1, height=2 * std2,
            edgecolor=cmap(norm(i)), facecolor="none",
            alpha=0.5, lw=1, zorder=3,
        )
        ax.add_patch(ell)

    # Observed tail (NCEP wind-only analysis, leading into init)
    if has_obs:
        ax.plot(obs["rmm1"].values, obs["rmm2"].values,
                "k-", lw=1.8, zorder=5, label=f"Observed ({obs_src}, wind-only)")
        ax.plot(obs["rmm1"].values[-1], obs["rmm2"].values[-1],
                "ko", ms=6, zorder=6)

    # AIFS-analysis archive ('truth') — accumulates one point per run
    if has_truth:
        ax.plot(truth["rmm1"].values, truth["rmm2"].values,
                "-s", color="darkorange", ms=4, lw=1.2, zorder=6,
                label="AIFS analysis (archive)")

    # Mark day-0 and last day
    ax.scatter(mean_rmm1[0], mean_rmm2[0], c="green", s=60, zorder=7, label="Day 0 (mean)")
    ax.scatter(mean_rmm1[-1], mean_rmm2[-1], c="red", s=60, zorder=7, label=f"Day {int(lead_days[-1])} (mean)")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=int(lead_days[-1])))
    plt.colorbar(sm, ax=ax, label="Forecast lead day", fraction=0.046, pad=0.02)
    ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {out_path}")

    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rmm", required=True)
    parser.add_argument("--obs-rmm", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    rmm = xr.open_dataset(args.rmm)
    obs = xr.open_dataset(args.obs_rmm) if args.obs_rmm else None

    out_path = Path(args.out) if args.out else None
    plot_rmm(rmm, obs, out_path=out_path)

    if out_path is None:
        plt.show()


if __name__ == "__main__":
    main()
