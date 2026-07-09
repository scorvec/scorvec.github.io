"""Run the full pipeline: ingest -> model -> output/forecast.json + summary."""
import json
from pathlib import Path

import ingest
import model

ROOT = Path(__file__).resolve().parents[2]


def main():
    print("Fetching data (VoteHub, Kalshi, Polymarket, Bluesky)...")
    bundle = ingest.snapshot()
    for k in ("polls", "markets", "social"):
        status = bundle[k].get("error", "ok") if isinstance(bundle[k], dict) else "ok"
        print(f"  {k}: {status}")

    forecast = model.run(bundle)
    out = ROOT / "data" / "election2026"
    out.mkdir(exist_ok=True)
    (out / "forecast.json").write_text(json.dumps(forecast, indent=2))
    print(f"\nWrote {out / 'forecast.json'}\n")

    h, s = forecast["house"], forecast["senate"]
    print("=" * 62)
    print(f"2026 MIDTERM FORECAST  (as of {forecast['as_of']})")
    print("=" * 62)
    if h["model"]:
        m = h["model"]
        print(f"HOUSE   generic ballot avg: D{m['poll_average_margin']:+.1f} "
              f"({m['n_polls_used']} polls)")
        sb = forecast["house"].get("silver_bulletin_margin")
        print(f"        avg undecided {m['avg_undecided_pct']}% "
              f"(adds {m['undecided_extra_sd']} sd)"
              + (f" | Silver Bulletin cross-check: D{sb:+.1f}" if sb else ""))
        print(f"        model P(Dem): {m['p_dem_control']:.0%}   "
              f"seats: {m['dem_seats_mean']:.0f} "
              f"[{m['dem_seats_p10']:.0f}-{m['dem_seats_p90']:.0f}]")
    print(f"        markets P(Dem): kalshi {h['markets']['kalshi_p_dem']}, "
          f"polymarket {h['markets']['polymarket_p_dem']}")
    print(f"        ENSEMBLE P(Dem House): {h['ensemble_p_dem_control']:.0%}")
    print("-" * 62)
    if s["model"]:
        m = s["model"]
        print(f"SENATE  model P(Dem): {m['p_dem_control']:.0%}   "
              f"seats: {m['dem_seats_mean']:.1f} "
              f"[{m['dem_seats_p10']:.0f}-{m['dem_seats_p90']:.0f}]")
        comp = sorted(m["races"].items(), key=lambda kv: abs(kv[1]["p_dem_market"] - 0.5))
        print("        closest races (market P(Dem)):")
        for st, r in comp[:8]:
            tag = " [prior]" if r["prior_used"] else ""
            print(f"          {st:<16} {r['p_dem_market']:.0%}  "
                  f"(def: {r['incumbent_party']}){tag}")
    print(f"        markets P(Dem): kalshi {s['markets']['kalshi_p_dem']}, "
          f"polymarket {s['markets']['polymarket_p_dem']}")
    print(f"        ENSEMBLE P(Dem Senate): {s['ensemble_p_dem_control']:.0%}")
    print("-" * 62)
    soc = forecast.get("social_indicator") or {}
    if "queries" in soc:
        print("SOCIAL (Bluesky, experimental — not in forecast):")
        for q, v in soc["queries"].items():
            print(f"          '{q}': {v['n_posts']} posts, "
                  f"{v['posts_per_hour']}/hr, net sentiment {v['net_sentiment']:+.2f}")


if __name__ == "__main__":
    main()
