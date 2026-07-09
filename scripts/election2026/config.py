"""Configuration: Senate race table and model parameters.

All model parameters are calibration knobs, documented inline. They encode
historical relationships (generic ballot -> House seats, poll bias) and are
the first thing to refine with proper backtesting.
"""

# ---------------------------------------------------------------------------
# 2026 Senate races: 33 Class 2 seats + OH and FL specials (35 total).
# "inc" = party currently holding the seat (defending party).
# Polymarket slug pattern: "<state>-senate-election-winner"; races without a
# listed market fall back to an incumbent-party prior.
# ---------------------------------------------------------------------------
SENATE_RACES = {
    "Alabama": "R", "Alaska": "R", "Arkansas": "R", "Colorado": "D",
    "Delaware": "D", "Florida": "R", "Georgia": "D", "Idaho": "R",
    "Illinois": "D", "Iowa": "R", "Kansas": "R", "Kentucky": "R",
    "Louisiana": "R", "Maine": "R", "Massachusetts": "D", "Michigan": "D",
    "Minnesota": "D", "Mississippi": "R", "Montana": "R", "Nebraska": "R",
    "New Hampshire": "D", "New Jersey": "D", "New Mexico": "D",
    "North Carolina": "R", "Ohio": "R", "Oklahoma": "R", "Oregon": "D",
    "Rhode Island": "D", "South Carolina": "R", "South Dakota": "R",
    "Tennessee": "R", "Texas": "R", "Virginia": "D", "West Virginia": "R",
    "Wyoming": "R",
}

# Seats NOT up in 2026 (senators seated through 2027+): 34 D-caucus, 31 R.
# Current chamber: 53 R / 47 D-caucus; 13 D and 22 R seats are up above.
DEM_SEATS_NOT_UP = 34
# VP (Vance, R) breaks ties, so Democrats need 51 for control.
DEM_SENATE_MAJORITY = 51

# Fallback P(Dem win) by defending party when no market exists (safe seats).
SAFE_SEAT_PRIOR = {"D": 0.95, "R": 0.05}

# Party lookup for candidate-level markets whose titles carry no (D)/(R) tag.
# Only needed where Polymarket lists candidates untagged (currently Alaska).
CANDIDATE_PARTY = {
    "mary peltola": "D",
    "dan sullivan": "R",
}

# ---------------------------------------------------------------------------
# House model parameters
# ---------------------------------------------------------------------------
HOUSE = {
    # Poll averaging
    "poll_window_days": 45,        # only polls ending within this window
    "recency_halflife_days": 18,   # exponential down-weighting of older polls
    # Population adjustments added to the Dem margin (LV is the benchmark;
    # RV and adult samples historically run ~1-1.5 pts more Democratic).
    "pop_adjust": {"lv": 0.0, "rv": -1.0, "a": -1.5, None: -1.0},

    # Undecided handling ("option 3"): a poll leaving many voters unallocated
    # is less informative, so its weight shrinks by (two-party share /
    # typical)^2 with a floor. Undecideds are assumed to split evenly in the
    # mean; the window's average undecided share above a baseline adds
    # forecast variance instead of being allocated to either party.
    "two_party_typical": 86.0,
    "undecided_weight_floor": 0.35,
    "undecided_baseline_pct": 8.0,
    "undecided_extra_sd_per_pt": 0.12,

    # Systematic polling bias: generic-ballot averages have overstated the
    # eventual Dem House margin in recent cycles (2020 badly, 2022 mildly).
    # Modeled as N(mean, sd) subtracted from the poll average.
    "bias_mean": 1.0,
    "bias_sd": 2.5,

    # Random drift between now and election day (~4 months out in July).
    "drift_sd": 3.0,

    # Seats-votes curve: dem_seats = 218 + slope * (margin - even_split_margin)
    # even_split_margin: national Dem popular-vote margin at which the chamber
    # splits ~even. Set to +1.5 reflecting a modest net-GOP tilt after the
    # 2025-26 mid-decade redistricting wars (TX/NC/MO/OH offset partly by CA).
    # This is the single most uncertain knob in the model.
    "even_split_margin": 1.5,
    "seats_per_point": 5.0,
    "seat_noise_sd": 9.0,          # residual seats-votes translation error

    "n_sims": 50_000,
}

SENATE = {
    # Correlated simulation of per-race win probabilities: a shared national
    # shock is applied on the logit scale to every race, so a good/bad night
    # for one party moves all races together.
    "national_shock_beta": 0.7,
    # Clamp market prices away from 0/1 before logit transform.
    "prob_floor": 0.015,
    "n_sims": 50_000,
}

# Ensemble weights for the headline P(control): our structural model vs the
# direct control markets. The Senate structural model is itself built from
# per-race market prices, so its independent contribution is the correlation
# structure, not the level.
ENSEMBLE = {"model": 0.50, "kalshi": 0.25, "polymarket": 0.25}

# ---------------------------------------------------------------------------
# Social (Bluesky) — measured and reported as an experimental indicator only;
# it does NOT enter the forecast. Volume + crude lexicon sentiment.
# ---------------------------------------------------------------------------
BLUESKY_QUERIES = ["midterms", "2026 election", "democrats", "republicans"]

# Silver Bulletin: headline average scraped as a cross-check (display only),
# and the public poll-level CSV snapshotted for future backtesting.
SILVER_BULLETIN = {
    "page": "https://www.natesilver.net/p/generic-ballot-average-2026-nate-silver-bulletin-congress-polls",
    "csv": "https://docs.google.com/spreadsheets/d/e/2PACX-1vRsvXNCZ0ubJr8D_yNcU5q6C0_HBa35K7oDK03KpO7Ca43UwdXaIdvVLWoXEmHHph0EREz5430Hm5yZ/pub?output=csv",
}

SENTIMENT_POS = {
    "win", "winning", "won", "great", "good", "strong", "hope", "hopeful",
    "energized", "momentum", "landslide", "flip", "surge", "excited",
    "popular", "love", "best", "record", "turnout", "inspiring", "lead",
    "leading", "ahead",
}
SENTIMENT_NEG = {
    "lose", "losing", "lost", "bad", "worst", "weak", "corrupt", "broke",
    "scandal", "fail", "failing", "failed", "disaster", "angry", "hate",
    "unpopular", "behind", "collapse", "panic", "doomed", "fraud", "rigged",
    "afraid", "freak",
}
