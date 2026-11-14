#!/usr/bin/env python3
"""The North Pacific jet's extension watch (circulation.html#pacjet): the ensemble's extension episodes against ERA5's,
and the state of the things that changed the odds of an onset in ERA5. Called by pacjet.py with the members' jet
indices; the yardsticks are those of build_pacjet_drivers.py (data/reference/pacjet_drivers.json), so an onset, a
state and a lead window mean here what they meant in the reanalysis.

render     assets/sst/pacjet_{model}_event.webp and the numbers behind it. An onset is the first day of a run of at
           least `min_days` days in the extension phase with no extension day in the `quiet_days` before it (the last
           analyses stand in for the days before day 0); a member's onset must begin by the `horizon` day, the last on
           which a run can still be seen to last. A run that reaches the last forecast day is counted at its length so
           far. When most members are in the extension phase on day 0 the episode is under way, and the figure follows
           each member to the day it leaves the phase.
checklist  one row per ingredient: its value now, the state that puts it in (None between the thresholds) and the lead
           windows at which that state changed the odds of an onset significantly in ERA5. An ingredient whose input
           is missing or stale is left out. ENSO is the background, not a trigger: its row carries the share of
           each kind of winter's days spent in the extension phase, its verdict comes from the seasons' relation, and
           its lead windows are shown as one mean ratio (the state does not change within them). A forecast
           ingredient also gives its largest departure of the next AHEAD days ("ahead"), when that is another state
           and one that changed the odds. Nothing is multiplied together: the ingredients are not independent, and
           the ensemble's own share of onsets against ERA5's base rate is the forecast.

Inputs already in the repository, none fetched: the cleaned wind-only RMM (assets/mjo/rmm_models.json), the Siberian
High and the East Asian trough of the newest ensemble runs (assets/sst/data/pacjet_upstream.json, written by
scripts/ensmaps/jet_upstream.py), the Himalayan mountain torque forecast and the ONI.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "lib")]
import pacjet_core as PC

REF = HERE.parent / "data" / "reference"
F_DRIVERS, F_SKILL, F_TORQ = REF / "pacjet_drivers.json", REF / "pacjet_skill.json", REF / "pacjet_torque.json"
DATA = REPO / "assets" / "sst" / "data"
F_RMM, F_UPSTREAM, F_ONI = (
    REPO / "assets" / "mjo" / "rmm_models.json",
    DATA / "pacjet_upstream.json",
    DATA / "nino_history.json",
)
ENSEMBLE = {"aifs": "AIFS-ENS", "ifs": "IFS-ENS", "gefs": "GEFS", "geps": "GEPS"}
TAIL = 20  # analysed days drawn before day 0
R_USEFUL = 0.6  # the forecast index's correlation with the analysis below which a lead is marked
MIN_MEMBERS = 5  # onsets needed before the members' episodes are compared with ERA5's
FRESH = pd.Timedelta(hours=36)  # how far from this run an ensemble's or the torque's own run may be
MJO_FRESH = pd.Timedelta(days=3)
AHEAD = pd.Timedelta(days=7)  # how far past an ingredient's first value its forecast extreme is looked for
THRESHOLD = 1.0  # sigma: a state begins here, as in the reference
ENSO = 0.5
ALPHA = 0.05  # the seasons' ENSO relation sets the ENSO row's verdict when its p is below this


def reference() -> dict:
    return json.loads(F_DRIVERS.read_text())


def in_season(month: int) -> bool:
    """The watch runs for forecasts read on a November-March basis, the months of the reference's onsets."""
    return month in reference()["months"]


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import house
    import matplotlib.pyplot as plt

    house.apply()
    plt.rcParams.update({"font.size": 10.5, "font.family": "DejaVu Sans"})
    return plt, house


def _day(t: pd.Timestamp) -> str:
    return f"{t.day} {t:%b}"


def note(out: Path, title: str, text: str) -> None:
    """A figure that says in a sentence or two why a view is not drawn."""
    plt, house = _mpl()
    fig = plt.figure(figsize=(13.0, 1.9))
    fig.text(0.04, 0.82, title, fontsize=15, fontweight="bold", color=house.INK, va="top")
    fig.text(0.04, 0.5, textwrap.fill(text, 150), fontsize=11.5, color=house.INK, va="top", linespacing=1.4)
    fig.savefig(out, dpi=100, facecolor="white", pil_kwargs={"quality": 88, "method": 6})
    plt.close(fig)


def _runs(x: np.ndarray) -> list[tuple[int, int]]:
    """(first, after the last) of each run of True."""
    edge = np.flatnonzero(np.diff(np.concatenate([[0], x.astype(int), [0]])))
    return list(zip(edge[::2].tolist(), edge[1::2].tolist()))


def episodes(pcs: np.ndarray, before: np.ndarray, D: dict) -> dict:
    """The members' extension episodes. pcs: (member, day from 0, 2); before: (quiet_days, 2), the analysed indices of
    the days up to the one before day 0, NaN where there is no analysis (counted as outside the phase).
    -> state "under_way" or "watch", and per member the episode's first day (NaN without one; negative when it began
    before day 0), its length in days, whether it is still going on the last day, and its peak PC1."""
    quiet, need, last = D["onset"]["quiet_days"], D["onset"]["min_days"], D["horizon"]["days"]
    ext = PC.phase_of(pcs[..., 0], pcs[..., 1]) == 0
    was = PC.phase_of(before[:, 0], before[:, 1]) == 0
    n, nday = ext.shape
    first, length, peak = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    going = np.zeros(n, bool)
    under_way = ext[:, 0].mean() >= 0.5
    since = next((k for k in range(quiet) if not was[quiet - 1 - k]), quiet) if under_way else 0
    for m in range(n):
        full = np.concatenate([was, ext[m]])
        if under_way:
            hit = next(((a, b) for a, b in _runs(full) if a <= quiet < b), None)
        else:
            hit = next(
                (
                    (a, b)
                    for a, b in _runs(full)
                    if quiet <= a <= quiet + last and b - a >= need and not full[a - quiet : a].any()
                ),
                None,
            )
        if hit is None:
            continue
        a, b = max(hit[0], quiet - since), hit[1]
        first[m], length[m], going[m] = a - quiet, b - a, b == quiet + nday
        peak[m] = np.nanmax(np.concatenate([before[a:quiet, 0], pcs[m, max(a - quiet, 0) : b - quiet, 0]]))
    return {
        "state": "under_way" if under_way else "watch",
        "ext": ext,
        "was": was,
        "first": first,
        "length": length,
        "going": going,
        "peak": peak,
    }


def _weak_from(model: str) -> int | None:
    """The first lead (days) at which the measured correlation of the forecast PC1 with the analysis is below
    R_USEFUL; None when it is not measured or stays above."""
    if not F_SKILL.exists():
        return None
    sk = json.loads(F_SKILL.read_text())
    mm = sk.get("models", {}).get(model, {})
    blk = mm.get("season") or mm.get("all")
    if not blk:
        return None
    return next((int(lead) for lead, r in zip(sk["leads"], blk["pc1"]["r"]) if r < R_USEFUL), None)


def _quartiles(x) -> list[float]:
    return [round(float(q), 2) for q in np.percentile(np.asarray(x, float), [25, 50, 75])]


def render(model, label, colour, init, pcs, month, tail, out) -> dict:
    """The extension tracker's figure and numbers. tail: (days, (n, 2)) of the analyses before day 0 on this basis."""
    D = reference()
    mname = pd.Timestamp(2001, month, 1).strftime("%B")
    if month not in D["months"]:
        note(
            out,
            f"North Pacific jet extension watch — not run in {mname} ({label} init {init:%Y-%m-%d %HZ})",
            "The watch compares the forecast with ERA5's extension onsets of November to March, the months in which "
            "the jet's extension pattern is at its strongest. It runs for forecasts centred in those months; the jet "
            "maps and the jet-axis view carry the forecast in the meantime.",
        )
        return {"in_season": False}
    plt, house = _mpl()
    quiet, horizon = D["onset"]["quiet_days"], D["horizon"]["days"]
    base = 100 * D["horizon"]["p"]
    day0 = init.normalize()
    past = pd.Series(list(tail[1]), index=tail[0]).reindex(pd.date_range(day0 - pd.Timedelta(days=TAIL), periods=TAIL))
    past = np.array([p if isinstance(p, np.ndarray) else np.full(2, np.nan) for p in past]).reshape(TAIL, 2)
    E = episodes(pcs, past[-quiet:], D)
    n, nday = E["ext"].shape
    days = np.arange(nday)
    has = np.isfinite(E["first"])
    under_way = E["state"] == "under_way"
    ext_share = 100 * E["ext"].mean(0)
    era_len, era_peak = D["episodes"]["days"], D["episodes"]["peak"]
    doc = {
        "in_season": True,
        "state": E["state"],
        "members": n,
        "analysed_days": int(np.isfinite(past[-quiet:, 0]).sum()),
        "base": round(base, 1),
        "horizon": horizon,
        "ext_share": np.round(ext_share).astype(int).tolist(),
        "era5": {"episodes": len(era_len), "length": _quartiles(era_len), "peak": _quartiles(era_peak)},
    }
    when = lambda d: (init + pd.Timedelta(days=round(d))).strftime("%Y-%m-%d")  # noqa: E731
    if has.any():
        doc["length"] = {"median": float(np.median(E["length"][has])), "going": int(E["going"][has].sum())}
        doc["peak"] = round(float(np.median(E["peak"][has])), 2)
    if under_way:
        left = np.array([next((d for d in days if not x[d]), nday) for x in E["ext"]])
        start = int(np.nanmin(E["first"]))
        doc |= {"since": when(start), "going": int((left == nday).sum())}
        if np.median(left) < nday:
            doc["end"] = when(np.median(left) - 1)
        head = (
            f"In the extension phase since {_day(init + pd.Timedelta(days=start))}. "
            + (
                f"Half the members have left it by {_day(init + pd.Timedelta(days=int(np.median(left))))}"
                if np.median(left) < nday
                else f"More than half the members are still in it on day {nday - 1}"
            )
            + f" ({doc['going']} of {n} still in it on day {nday - 1}). "
            f"ERA5's {len(era_len)} episodes lasted {doc['era5']['length'][1]:.0f} days at the median."
        )
    else:
        cum = np.array([100 * (E["first"][has] <= d).sum() / n for d in days])
        doc["onset"] = {
            "members": int(has.sum()),
            "share": round(100 * has.mean(), 1),
            "by_day": np.round(cum).astype(int).tolist(),
        }
        if has.any():
            doc["onset"]["median"] = when(np.median(E["first"][has]))
        by = _day(init + pd.Timedelta(days=horizon))
        head = (
            (
                f"{int(has.sum())} of {n} members begin an extension by {by}"
                + (f" (median onset {_day(pd.Timestamp(doc['onset']['median']))})" if has.any() else "")
                if has.any()
                else f"No member begins an extension by {by}"
            )
            + f"; in ERA5 one begins within {horizon} days of {base:.0f} % of November–March days outside the phase. "
            f"An onset: at least {D['onset']['min_days']} days in the extension phase after {quiet} days without one."
        )

    W, H = 13.4, 7.3
    fig = plt.figure(figsize=(W, H))
    x0, x1 = -TAIL - 0.6, nday - 0.4
    ticks = np.arange(-TAIL, nday, 5)

    def dates(ax, labels=True):
        ax.set_xlim(x0, x1)
        ax.set_xticks(ticks)
        ax.set_xticklabels([_day(init + pd.Timedelta(days=int(d))) for d in ticks] if labels else [], fontsize=9)
        ax.axvline(0, color="#777", lw=0.8, zorder=1)
        ax.tick_params(labelsize=9)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)

    # the index
    ax = fig.add_axes([0.055, 0.42, 0.585, 0.415])
    weak = _weak_from(model)
    if weak is not None:
        ax.axvspan(
            weak - 0.5,
            x1,
            color="#000",
            alpha=0.045,
            lw=0,
            label=f"day {weak} on: the index's measured correlation is below {R_USEFUL}",
        )
    ax.axhline(0, color="#777", lw=0.7, zorder=1)
    ax.axhline(THRESHOLD, color="#999", lw=0.8, ls=(0, (2, 2)), zorder=1)
    if not under_way and has.sum() >= MIN_MEMBERS:
        lag = np.array(D["evolution"]["lag"]) + np.median(E["first"][has])
        keep = lag <= x1
        ax.fill_between(
            lag[keep],
            np.array(D["evolution"]["lo"])[keep],
            np.array(D["evolution"]["hi"])[keep],
            color=house.WARM,
            alpha=0.16,
            lw=0,
        )
        ax.plot(
            lag[keep],
            np.array(D["evolution"]["mid"])[keep],
            color=house.WARM,
            lw=1.6,
            ls=(0, (5, 2)),
            label=f"ERA5's {len(era_len)} onsets set at the members' median onset (median, middle half)",
        )
    ax.plot(days, pcs[..., 0].T, color=colour, lw=0.5, alpha=0.22, zorder=2)
    ax.fill_between(days, *np.percentile(pcs[..., 0], [10, 90], axis=0), color=colour, alpha=0.16, lw=0, zorder=2)
    ax.plot(days, pcs[..., 0].mean(0), color=colour, lw=2.4, zorder=4, label=f"{label}: members, 10–90 % and the mean")
    ax.plot(np.arange(-TAIL, 0), past[:, 0], color=house.INK, lw=1.8, zorder=4, label="analysis")
    ax.plot([-1, 0], [past[-1, 0], pcs[:, 0, 0].mean()], color=house.INK, lw=1.8, zorder=4)
    dates(ax, labels=False)
    ax.set_ylabel("σ", fontsize=10)
    ax.set_title(
        "Extension index (PC1): above the dashed line the jet is extended",
        fontsize=11,
        loc="left",
        fontweight="bold",
        color=house.INK,
    )
    ax.legend(loc="upper left", fontsize=8.6, frameon=False, handlelength=2.2)
    house.legend_room(ax)

    # the phase by day
    axb = fig.add_axes([0.055, 0.09, 0.585, 0.24])
    axb.bar(
        np.arange(-TAIL, 0)[-len(past) :],
        100 * (PC.phase_of(past[:, 0], past[:, 1]) == 0),
        width=0.82,
        color="#8a8780",
        label="analysis in the extension phase",
    )
    axb.bar(days, ext_share, width=0.82, color=colour, alpha=0.45, label="members in the extension phase")
    if not under_way:
        axb.step(days, cum, where="post", color=house.INK, lw=1.8, label="members whose extension has begun")
        axb.plot(
            [0, horizon],
            [base, base],
            color=house.WARM,
            lw=1.4,
            ls=(0, (5, 2)),
            label=f"ERA5: an onset within {horizon} days ({base:.0f} %)",
        )
    dates(axb)
    axb.set_ylim(0, 100)
    axb.set_ylabel("% of members", fontsize=10)
    axb.set_title("In the extension phase, day by day", fontsize=11, loc="left", fontweight="bold", color=house.INK)
    axb.legend(loc="upper left", fontsize=8.6, frameon=False, handlelength=2.2)
    house.legend_room(axb)

    # the episodes against ERA5's
    enough = has.sum() >= MIN_MEMBERS
    for rect, era, mine, bins, title, unit in (
        (
            [0.71, 0.555, 0.275, 0.28],
            era_len,
            E["length"][has],
            np.arange(2.5, 19.5),
            "How long: days in the extension phase",
            "d",
        ),
        (
            [0.71, 0.09, 0.275, 0.28],
            era_peak,
            E["peak"][has],
            np.arange(0.8, 3.81, 0.2),
            "How strong: the episode's peak index",
            "σ",
        ),
    ):
        axh = fig.add_axes(rect)
        top = bins[-1] - (bins[1] - bins[0]) / 2
        h = np.histogram(np.minimum(era, top), bins)[0]
        axh.bar(
            bins[:-1],
            100 * h / len(era),
            width=np.diff(bins),
            align="edge",
            color="#d4d1c9",
            edgecolor="white",
            lw=0.5,
            label=f"ERA5, {len(era)} episodes (median {np.median(era):.3g} {unit})",
        )
        if enough:
            h = np.histogram(np.minimum(mine, top), bins)[0]
            axh.stairs(
                100 * h / len(mine),
                bins,
                color=colour,
                lw=2,
                label=f"{label}, {len(mine)} members (median {np.median(mine):.3g} {unit})",
            )
        else:
            axh.plot([], [], color=colour, lw=2, label=f"{label}: too few members with an episode")
        axh.set_ylabel("% of episodes", fontsize=10)
        axh.tick_params(labelsize=9)
        for s_ in ("top", "right"):
            axh.spines[s_].set_visible(False)
        axh.set_title(title, fontsize=11, loc="left", fontweight="bold", color=house.INK)
        axh.legend(loc="upper right", fontsize=8.6, frameon=False, handlelength=1.6)
        house.legend_room(axh)

    fig.text(
        0.055,
        1 - 0.14 / H,
        f"North Pacific jet extension watch — {label} {n} members, init {init:%Y-%m-%d %HZ}",
        fontsize=14,
        fontweight="bold",
        va="top",
        color=house.INK,
    )
    fig.text(0.055, 1 - 0.47 / H, textwrap.fill(head, 168), fontsize=9.8, va="top", color=house.MUTED, linespacing=1.3)
    going = int(E["going"][has].sum())
    fig.text(
        0.055,
        0.1 / H,
        f"ERA5 1991–2020, November to March; the index in σ of the {mname} ± 1 month spread."
        + (
            f" {going} members' episodes are still going on day {nday - 1} and are counted at their length so far."
            if going
            else ""
        ),
        fontsize=8.6,
        va="bottom",
        color=house.MUTED,
    )
    house.clear(fig, 100)
    fig.savefig(out, dpi=100, facecolor="white", pil_kwargs={"quality": 88, "method": 6})
    plt.close(fig)
    return doc


def _sigma(v: float) -> str:
    return f"{v:+.1f} σ".replace("-", "−")


def _effects(D: dict, state: str | None) -> list[dict]:
    """The lead windows at which a state changed the onset odds significantly."""
    s = D["states"].get(state, {}) if state else {}
    return [{"lead": lead, "ratio": r} for lead, r, sig in zip(D["leads"], s.get("ratio", []), s.get("sig", [])) if sig]


def _ahead(D: dict, valid: list, values: list, hi: str, lo: str, now: str | None) -> dict:
    """The forecast's largest departure within AHEAD of its first value, when that is a state other than the first
    value's and one that changed the odds: {"ahead": ...} for the row, else {}."""
    later = [(t, v) for t, v in zip(valid[1:], values[1:]) if v is not None and t - valid[0] <= AHEAD]
    if not later:
        return {}
    t, v = max(later, key=lambda x: abs(x[1]))
    state = _banded(v, hi, lo)
    effects = _effects(D, state)
    if state == now or not effects:
        return {}
    return {"ahead": {"value": _sigma(v), "valid": f"{t:%Y-%m-%d}", "state": state, "effects": effects}}


def _row(D: dict, key: str, name: str, now: str, asof, state: str | None, **more) -> dict:
    """A checklist row: the lead windows at which the state changed the odds significantly, and what they add up to."""
    effects = _effects(D, state)
    up, down = any(e["ratio"] > 1 for e in effects), any(e["ratio"] < 1 for e in effects)
    verdict = "mixed" if up and down else "favours" if up else "against" if down else "none"
    return {
        "key": key,
        "name": name,
        "now": now,
        "asof": asof if isinstance(asof, str) else f"{asof:%Y-%m-%d}",
        "state": state,
        "effects": effects,
        "verdict": verdict,
    } | more


def _banded(v: float, hi: str | None, lo: str | None) -> str | None:
    return hi if v >= THRESHOLD else lo if v <= -THRESHOLD else None


def _mjo(D: dict, init: pd.Timestamp) -> dict | None:
    if not F_RMM.exists():
        return None
    r = json.loads(F_RMM.read_text())
    o = r["obs"]
    t = pd.to_datetime(o["t"], format="%Y-%m-%dT%H")
    k = next((i for i in range(len(t) - 1, -1, -1) if o["r1"][i] is not None and o["r2"][i] is not None), None)
    if k is None or abs(init - t[k]) > MJO_FRESH:
        return None
    x1, x2 = (o[f"r{j + 1}"][k] - (o["dn"][k] or 0.0) * r["c"][j] - (o["dd"][k] or 0.0) * r["d"][j] for j in range(2))
    amp = float(np.hypot(x1, x2))
    if amp < 1:
        return _row(D, "mjo", "MJO", f"no MJO (amplitude {amp:.1f})", t[k], "mjo_none")
    phase = int(((np.degrees(np.arctan2(x2, x1)) % 360 - 180) % 360) // 45) + 1
    pair = next(p for p in ((8, 1), (2, 3), (4, 5), (6, 7)) if phase in p)
    return _row(D, "mjo", "MJO", f"phase {phase}, amplitude {amp:.1f}", t[k], f"mjo{pair[0]}{pair[1]}")


def _upstream(D: dict, model: str, init: pd.Timestamp) -> list[dict]:
    if not F_UPSTREAM.exists():
        return []
    runs = json.loads(F_UPSTREAM.read_text())["models"]
    age = {k: abs(pd.Timestamp(r["init"].rstrip("Z")) - init) for k, r in runs.items()}
    order = sorted((k for k in runs if age[k] <= FRESH), key=lambda k: (k != model, age[k]))
    rows = []
    for key, name, hi, lo, words in (
        ("sib", "Siberian High", "sib_hi", "sib_lo", ("stronger than normal", "weaker than normal")),
        ("eat", "East Asian trough", "eat_hi", "eat_lo", ("shallower than normal", "deeper than normal")),
    ):
        for k in order:
            r = runs[k]
            t0 = pd.Timestamp(r["init"].rstrip("Z"))
            valid = [t0 + pd.Timedelta(days=i + 1) for i in range(len(r.get(key, [])))]
            i = next((i for i, (t, v) in enumerate(zip(valid, r.get(key, []))) if t >= init and v is not None), None)
            if i is None or valid[i] - init > FRESH:
                continue
            v = r[key][i]
            state = _banded(v, hi, lo)
            word = words[0] if v >= THRESHOLD else words[1] if v <= -THRESHOLD else "near normal"
            more = _ahead(D, valid[i:], r[key][i:], hi, lo, state)
            rows.append(
                _row(D, key, name, f"{_sigma(v)}, {word}", valid[i], state, source=f"{ENSEMBLE[k]} mean", **more)
            )
            break
    return rows


def _torque(D: dict, init: pd.Timestamp, month: int, torque_json) -> dict | None:
    p = Path(torque_json)
    if not (p.exists() and F_TORQ.exists()):
        return None
    tq = json.loads(p.read_text())
    if abs(pd.Timestamp(tq["init"].rstrip("Z")) - init) > FRESH:
        return None
    sc = json.loads(F_TORQ.read_text())["torque_scale"][str(month)]
    z = [(float(x) - sc["mean"]) / sc["sd"] for x in tq["ranges"]["Himalaya/Tibet"]]
    valid = [pd.Timestamp(t) for t in tq["valid"]]
    state = _banded(z[0], "tq_hi", "tq_lo")
    return _row(
        D,
        "torque",
        "Himalayan mountain torque",
        _sigma(z[0]),
        valid[0],
        state,
        **({"source": f"{tq['model']} mean"} if tq.get("model") else {}),
        **_ahead(D, valid, z, "tq_hi", "tq_lo", state),
    )


def _enso(D: dict) -> dict | None:
    if not F_ONI.exists():
        return None
    h = json.loads(F_ONI.read_text())
    k = next((i for i in range(len(h["months"]) - 1, -1, -1) if h["series"]["oni"]["anom"][i] is not None), None)
    if k is None:
        return None
    v = h["series"]["oni"]["anom"][k]
    state, word = ("el_nino", "El Niño") if v >= ENSO else ("la_nina", "La Niña") if v <= -ENSO else (None, "neutral")
    e = D["enso_days"]
    row = _row(D, "enso", "ENSO", f"{word} (ONI {v:+.1f})".replace("-", "−"), h["months"][k], state)
    if state and e["p"] < ALPHA:
        row["verdict"] = "favours" if (state == "el_nino") == (e["r"] > 0) else "against"
    return row | {"days": {s: round(100 * x) for s, x in e["share"].items()}}


def checklist(model: str, init: pd.Timestamp, pcs: np.ndarray, month: int, torque_json) -> list[dict]:
    """The ingredients' rows, in the order the page lists them; [] out of season."""
    D = reference()
    if month not in D["months"]:
        return []
    pc1, pc2 = pcs[:, 0].mean(0)
    extended = bool(PC.phase_of(pc1, pc2) == 0)
    rows = [
        _mjo(D, init),
        *_upstream(D, model, init),
        _row(
            D,
            "pc1",
            "Jet length",
            "in the extension phase"
            if extended
            else f"{_sigma(pc1)}, {'retracted' if pc1 <= -THRESHOLD else 'near normal' if pc1 < THRESHOLD else 'long'}",
            init,
            None if extended else _banded(pc1, None, "pc1_lo"),
        ),
        _row(
            D,
            "pc2",
            "Jet latitude",
            f"{_sigma(pc2)}, {'toward the pole' if pc2 >= THRESHOLD else 'toward the equator' if pc2 <= -THRESHOLD else 'near normal'}",
            init,
            _banded(pc2, "pc2_hi", "pc2_lo"),
        ),
        _torque(D, init, month, torque_json),
        _enso(D),
    ]
    return [r for r in rows if r]
