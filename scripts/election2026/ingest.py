"""Data ingestion: VoteHub polls, Kalshi + Polymarket prediction markets,
Bluesky social posts. Every fetch is snapshotted to data/raw/<UTC-timestamp>/
so historical runs can be replayed and the model backtested later.

All sources are free, public, unauthenticated APIs.
"""
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

import config

ROOT = Path(__file__).resolve().parents[2]
UA = {"User-Agent": "election2026-research-model/0.1 (personal research)"}
BROWSER_UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                            "AppleWebKit/537.36"}

VOTEHUB = "https://api.votehub.com/polls"
KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
GAMMA = "https://gamma-api.polymarket.com"
BSKY = "https://api.bsky.app/xrpc/app.bsky.feed.searchPosts"


def _get(url, params=None, timeout=30):
    r = requests.get(url, params=params, headers=UA, timeout=timeout)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Traditional: poll-level generic ballot data (VoteHub)
# ---------------------------------------------------------------------------
def fetch_polls():
    raw = _get(VOTEHUB, {"poll_type": "generic-ballot", "page_size": 2000})
    polls = []
    for p in raw:
        if p.get("subject") != "2026":
            continue
        dem = rep = None
        for a in p.get("answers", []):
            if a["choice"].lower().startswith("dem"):
                dem = a["pct"]
            elif a["choice"].lower().startswith("rep"):
                rep = a["pct"]
        if dem is None or rep is None:
            continue
        polls.append({
            "pollster": p.get("pollster"),
            "end_date": p.get("end_date"),
            "sample_size": p.get("sample_size"),
            "population": p.get("population"),
            "internal": p.get("internal", False),
            "partisan": p.get("partisan"),
            "dem": dem, "rep": rep, "margin": round(dem - rep, 2),
        })
    polls.sort(key=lambda p: p["end_date"] or "", reverse=True)
    out = {"source": "votehub", "polls": polls}
    # Silver Bulletin headline average as a cross-check (display only)
    try:
        r = requests.get(config.SILVER_BULLETIN["page"], headers=BROWSER_UA,
                         timeout=30)
        m = re.search(r"currently sitting at ([DR])\s*\+([\d.]+)", r.text)
        if m:
            out["silver_bulletin"] = {
                "margin": (1 if m.group(1) == "D" else -1) * float(m.group(2))}
    except requests.RequestException:
        pass
    return out


# ---------------------------------------------------------------------------
# Alternative: prediction markets (Kalshi + Polymarket)
# ---------------------------------------------------------------------------
def _kalshi_control(event_ticker):
    """Return P(Dem control) from a Kalshi control event (mid price)."""
    data = _get(f"{KALSHI}/markets", {"event_ticker": event_ticker})
    for m in data.get("markets", []):
        if "democrat" in (m.get("yes_sub_title") or "").lower():
            bid = float(m.get("yes_bid_dollars") or 0)
            ask = float(m.get("yes_ask_dollars") or 0)
            mid = (bid + ask) / 2 if bid and ask else float(m.get("last_price_dollars") or 0)
            return {"p_dem": mid, "ticker": m["ticker"],
                    "volume": float(m.get("volume_fp") or 0)}
    return None


def _poly_event(slug):
    data = _get(f"{GAMMA}/events", {"slug": slug})
    return data[0] if data else None


def _yes_price(m):
    prices = json.loads(m["outcomePrices"])
    outcomes = json.loads(m.get("outcomes") or '["Yes","No"]')
    for o, pr in zip(outcomes, prices):
        if o.lower() == "yes":
            return float(pr)
    return None


def _poly_party_prob(event, party_word):
    """P(party wins) from a Polymarket event.

    Handles both market shapes: party-level markets ("Democrat"/"Republican")
    and candidate-level markets ("Roy Cooper (D)"), summing candidate Yes
    prices by party tag, with config.CANDIDATE_PARTY for untagged names.
    """
    tag = "(d)" if party_word == "democrat" else "(r)"
    party_letter = "D" if party_word == "democrat" else "R"
    party_p, cand_sum, cand_found = None, 0.0, False
    for m in event.get("markets", []):
        if not m.get("outcomePrices"):
            continue
        title = (m.get("groupItemTitle") or m.get("question") or "").strip()
        tl = title.lower()
        yes = _yes_price(m)
        if yes is None:
            continue
        if party_word in tl and tag not in tl:
            party_p = yes
        elif tag in tl or config.CANDIDATE_PARTY.get(tl) == party_letter:
            cand_sum += yes
            cand_found = True
    if party_p is not None:
        return party_p
    return min(cand_sum, 1.0) if cand_found else None


def fetch_markets():
    out = {"kalshi": {}, "polymarket": {}, "senate_races": {}}

    out["kalshi"]["house_dem"] = _kalshi_control("CONTROLH-2026")
    out["kalshi"]["senate_dem"] = _kalshi_control("CONTROLS-2026")

    for chamber, slug in [("house", "which-party-will-win-the-house-in-2026"),
                          ("senate", "which-party-will-win-the-senate-in-2026")]:
        ev = _poly_event(slug)
        if ev:
            out["polymarket"][f"{chamber}_dem"] = {
                "p_dem": _poly_party_prob(ev, "democrat"),
                "volume": ev.get("volume"), "liquidity": ev.get("liquidity"),
            }

    # Per-race Senate markets
    for state, inc in config.SENATE_RACES.items():
        slug = state.lower().replace(" ", "-") + "-senate-election-winner"
        entry = {"incumbent_party": inc, "market": None, "p_dem": None}
        try:
            ev = _poly_event(slug)
        except requests.RequestException:
            ev = None
        if ev:
            p_dem = _poly_party_prob(ev, "democrat")
            if p_dem is not None:
                entry["market"] = {"slug": slug, "volume": ev.get("volume"),
                                   "liquidity": ev.get("liquidity")}
                entry["p_dem"] = p_dem
        if entry["p_dem"] is None:
            entry["p_dem"] = config.SAFE_SEAT_PRIOR[inc]
            entry["prior_used"] = True
        out["senate_races"][state] = entry
        time.sleep(0.15)  # be polite to the API
    return out


# ---------------------------------------------------------------------------
# Alternative: social media (Bluesky, free public search)
# ---------------------------------------------------------------------------
def _score(text):
    words = {w.strip(".,!?;:'\"()").lower() for w in text.split()}
    return (len(words & config.SENTIMENT_POS), len(words & config.SENTIMENT_NEG))


def fetch_social():
    out = {"source": "bluesky", "queries": {}}
    for q in config.BLUESKY_QUERIES:
        posts = []
        try:
            data = _get(BSKY, {"q": q, "limit": 100, "sort": "latest"})
            posts = data.get("posts", [])
        except requests.RequestException:
            pass
        texts, newest, oldest = [], None, None
        pos = neg = 0
        for p in posts:
            rec = p.get("record", {})
            t = rec.get("text", "")
            created = rec.get("createdAt", "")
            texts.append(t)
            newest = max(newest or created, created)
            oldest = min(oldest or created, created)
            s = _score(t)
            pos += s[0]
            neg += s[1]
        span_hours = None
        if newest and oldest and newest != oldest:
            fmt = "%Y-%m-%dT%H:%M:%S"
            try:
                dt_n = datetime.fromisoformat(newest.replace("Z", "+00:00"))
                dt_o = datetime.fromisoformat(oldest.replace("Z", "+00:00"))
                span_hours = max((dt_n - dt_o).total_seconds() / 3600, 0.01)
            except ValueError:
                pass
        out["queries"][q] = {
            "n_posts": len(posts),
            "posts_per_hour": round(len(posts) / span_hours, 1) if span_hours else None,
            "pos_hits": pos, "neg_hits": neg,
            "net_sentiment": round((pos - neg) / max(len(posts), 1), 3),
            "sample_texts": texts[:5],
        }
        time.sleep(0.2)
    return out


# ---------------------------------------------------------------------------
def snapshot():
    """Fetch everything, save raw snapshot, return the bundle."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    outdir = ROOT / "data" / "election2026" / "raw" / ts
    outdir.mkdir(parents=True, exist_ok=True)

    bundle = {"fetched_at": ts}
    for name, fn in [("polls", fetch_polls), ("markets", fetch_markets),
                     ("social", fetch_social)]:
        try:
            bundle[name] = fn()
        except Exception as e:  # keep going if one source is down
            bundle[name] = {"error": f"{type(e).__name__}: {e}"}
        (outdir / f"{name}.json").write_text(json.dumps(bundle[name], indent=2))
    # Silver Bulletin poll-level CSV (public sheet): snapshot for backtesting
    try:
        r = requests.get(config.SILVER_BULLETIN["csv"], headers=BROWSER_UA,
                         timeout=30)
        if r.ok and r.text.startswith("subgroup,"):
            (outdir / "silver_polls.csv").write_text(r.text)
    except requests.RequestException:
        pass
    return bundle


if __name__ == "__main__":
    b = snapshot()
    print(json.dumps({k: (v if k == "fetched_at" else "ok" if "error" not in v else v["error"])
                      for k, v in b.items()}, indent=2))
