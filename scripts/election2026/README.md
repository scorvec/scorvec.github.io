# 2026 Midterm Election Model (v0.1)

Forecasts control of the US House and Senate by blending **traditional data**
(poll-level generic-ballot data) with **alternative data** (prediction market
prices, social media signals). All sources are free, public, unauthenticated APIs.

## Quick start

```bash
# from the repo root (needs python3 with requests + numpy)
python3 scripts/election2026/run_forecast.py     # fetch fresh data + run models
python3 scripts/election2026/build_dashboard.py  # rebuild election-2026/index.html
```

Every run snapshots raw data to `data/election2026/raw/<timestamp>/` (polls, markets, social)
so the model can later be backtested against history, and writes
`data/election2026/forecast.json`. The published page is `election-2026/index.html`.

## Data sources

| Source | Type | What we take |
|---|---|---|
| [VoteHub](https://api.votehub.com) | traditional | poll-level generic-ballot data (pollster, dates, sample, population) |
| [Kalshi](https://kalshi.com) | alternative | House/Senate control market prices (`CONTROLH-2026`, `CONTROLS-2026`) |
| [Polymarket](https://polymarket.com) | alternative | control markets + per-race Senate candidate prices (35 races) |
| [Bluesky](https://bsky.app) | alternative | post velocity + lexicon sentiment (display-only, not in forecast) |

## Model

- **House** (`model.py:house_model`): recency/size-weighted generic-ballot
  average with LV adjustment → Monte Carlo over systematic polling bias
  (recent cycles overstated Dems) and 4 months of drift → seats via a
  seats–votes curve → P(Dem ≥ 218).
- **Senate** (`model.py:senate_model`): per-race P(Dem) from Polymarket
  candidate prices (party-tagged or via `config.CANDIDATE_PARTY`), safe-seat
  priors where no market exists (currently only Alabama); correlated Monte
  Carlo with a shared national shock on the logit scale → P(Dem ≥ 51).
- **Ensemble**: 50% structural model + 25% Kalshi + 25% Polymarket for the
  headline probabilities.

## Known limitations (v0.1)

- **Calibration knobs in `config.py` are informed guesses, not backtested**
  — especially `even_split_margin` (the seats–votes tilt after the 2025–26
  mid-decade redistricting) and the bias/drift variances.
- The Senate structural model inherits its *levels* from market prices; its
  independent contribution is only the correlation structure.
- Nebraska: an independent candidacy (caucus unknown) isn't modeled — we use
  the Democratic candidate price only.
- Social sentiment is a crude word-count lexicon over 100 posts/query on one
  (left-leaning) platform; it is deliberately excluded from the forecast.
- No pollster house-effect or quality weights yet; internal/partisan polls are
  dropped, nothing more.

## Roadmap ideas

1. Nightly scheduled run + trend charts of the forecast itself over time.
2. Backtest the House knobs on 2010–2022 cycles (historical generic-ballot data).
3. Pollster house effects + quality weighting.
4. District-level House model (Cook/Sabato ratings + district markets).
5. Historical market-price time series (both APIs offer candlesticks) to study
   polls-vs-markets lead/lag.
6. LLM-based sentiment/stance classification instead of the lexicon; add more
   platforms for balance.
