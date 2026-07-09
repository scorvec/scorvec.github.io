"""Forecast models.

House: polls-first structural model. Weighted generic-ballot average ->
simulated true national margin (systematic bias + drift) -> seats via a
seats-votes curve -> P(Dem control) by Monte Carlo.

Senate: race-by-race simulation. Per-race P(Dem win) comes from Polymarket
prices (prior for unlisted safe seats); a shared national shock on the logit
scale correlates outcomes across races.

Ensemble: blends the structural models with the direct Kalshi/Polymarket
control markets for the headline probabilities.
"""
from datetime import date, datetime

import numpy as np

import config


# ---------------------------------------------------------------------------
def poll_average(polls, as_of=None):
    """Recency- and size-weighted generic ballot average with population
    adjustment. Returns (avg_margin, effective_n, details)."""
    cfg = config.HOUSE
    as_of = as_of or date.today()
    rows = []
    for p in polls:
        if not p.get("end_date"):
            continue
        end = datetime.strptime(p["end_date"], "%Y-%m-%d").date()
        age = (as_of - end).days
        if age < 0 or age > cfg["poll_window_days"]:
            continue
        if p.get("internal"):
            continue
        n = p.get("sample_size") or 800
        w = (0.5 ** (age / cfg["recency_halflife_days"])) * np.sqrt(min(n, 3000) / 1000)
        adj = cfg["pop_adjust"].get(p.get("population"), cfg["pop_adjust"][None])
        rows.append((p["margin"] + adj, w, p))
    if not rows:
        return None, 0, []
    margins = np.array([r[0] for r in rows])
    weights = np.array([r[1] for r in rows])
    avg = float(np.average(margins, weights=weights))
    eff_n = float(weights.sum() ** 2 / (weights ** 2).sum())
    return avg, eff_n, rows


def house_model(polls, rng):
    cfg = config.HOUSE
    avg, eff_n, rows = poll_average(polls)
    if avg is None:
        return None
    n = cfg["n_sims"]
    margin_true = (avg
                   - rng.normal(cfg["bias_mean"], cfg["bias_sd"], n)
                   + rng.normal(0, cfg["drift_sd"], n))
    dem_seats = (218
                 + cfg["seats_per_point"] * (margin_true - cfg["even_split_margin"])
                 + rng.normal(0, cfg["seat_noise_sd"], n))
    dem_seats = np.clip(np.round(dem_seats), 0, 435)
    return {
        "poll_average_margin": round(avg, 2),
        "n_polls_used": len(rows),
        "effective_n": round(eff_n, 1),
        "p_dem_control": float((dem_seats >= 218).mean()),
        "dem_seats_mean": float(dem_seats.mean()),
        "dem_seats_p10": float(np.percentile(dem_seats, 10)),
        "dem_seats_p90": float(np.percentile(dem_seats, 90)),
        "seat_histogram": _hist(dem_seats, 180, 280),
    }


def _hist(seats, lo, hi):
    bins = {}
    for s in seats:
        k = int(min(max(s, lo), hi))
        bins[k] = bins.get(k, 0) + 1
    total = len(seats)
    return {str(k): round(v / total, 5) for k, v in sorted(bins.items())}


# ---------------------------------------------------------------------------
def senate_model(races, rng):
    cfg = config.SENATE
    states = sorted(races)
    p = np.array([min(max(races[s]["p_dem"], cfg["prob_floor"]),
                      1 - cfg["prob_floor"]) for s in states])
    logits = np.log(p / (1 - p))
    n = cfg["n_sims"]
    shock = rng.normal(0, 1, (n, 1)) * cfg["national_shock_beta"]
    p_sim = 1 / (1 + np.exp(-(logits[None, :] + shock)))
    wins = rng.random((n, len(states))) < p_sim
    dem_seats = config.DEM_SEATS_NOT_UP + wins.sum(axis=1)
    flip_rates = wins.mean(axis=0)
    return {
        "p_dem_control": float((dem_seats >= config.DEM_SENATE_MAJORITY).mean()),
        "dem_seats_mean": float(dem_seats.mean()),
        "dem_seats_p10": float(np.percentile(dem_seats, 10)),
        "dem_seats_p90": float(np.percentile(dem_seats, 90)),
        "seat_histogram": _hist(dem_seats.astype(float), 40, 60),
        "races": {s: {"p_dem_sim": round(float(fr), 3),
                      "p_dem_market": round(races[s]["p_dem"], 3),
                      "incumbent_party": races[s]["incumbent_party"],
                      "prior_used": races[s].get("prior_used", False)}
                  for s, fr in zip(states, flip_rates)},
    }


# ---------------------------------------------------------------------------
def ensemble(p_model, p_kalshi, p_poly):
    w = dict(config.ENSEMBLE)
    parts = {"model": p_model, "kalshi": p_kalshi, "polymarket": p_poly}
    avail = {k: v for k, v in parts.items() if v is not None}
    tot = sum(w[k] for k in avail)
    return sum(w[k] * v for k, v in avail.items()) / tot if avail else None


def run(bundle, seed=2026):
    rng = np.random.default_rng(seed)
    polls = bundle.get("polls", {}).get("polls", [])
    markets = bundle.get("markets", {})

    house = house_model(polls, rng)
    senate = senate_model(markets.get("senate_races", {}), rng) \
        if markets.get("senate_races") else None

    k_house = (markets.get("kalshi", {}).get("house_dem") or {}).get("p_dem")
    k_sen = (markets.get("kalshi", {}).get("senate_dem") or {}).get("p_dem")
    p_house = (markets.get("polymarket", {}).get("house_dem") or {}).get("p_dem")
    p_sen = (markets.get("polymarket", {}).get("senate_dem") or {}).get("p_dem")

    return {
        "as_of": bundle.get("fetched_at"),
        "house": {
            "model": house,
            "markets": {"kalshi_p_dem": k_house, "polymarket_p_dem": p_house},
            "ensemble_p_dem_control": ensemble(
                house["p_dem_control"] if house else None, k_house, p_house),
        },
        "senate": {
            "model": senate,
            "markets": {"kalshi_p_dem": k_sen, "polymarket_p_dem": p_sen},
            "ensemble_p_dem_control": ensemble(
                senate["p_dem_control"] if senate else None, k_sen, p_sen),
        },
        "social_indicator": bundle.get("social"),
    }
