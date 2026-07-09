"""Build output/dashboard.html: compute chart-ready data from the latest
forecast + raw snapshot and inject it into src/dashboard_template.html."""
import json
from datetime import date, timedelta
from pathlib import Path

import model

ROOT = Path(__file__).resolve().parents[2]


def latest_snapshot():
    runs = sorted((ROOT / "data" / "election2026" / "raw").iterdir())
    return runs[-1]


def main():
    snap = latest_snapshot()
    forecast = json.loads((ROOT / "data" / "election2026" / "forecast.json").read_text())
    polls = json.loads((snap / "polls.json").read_text())["polls"]

    # Poll scatter: everything since Oct 2025
    scatter = [{"d": p["end_date"], "m": p["margin"], "pollster": p["pollster"],
                "n": p["sample_size"], "pop": p["population"]}
               for p in polls if (p["end_date"] or "") >= "2025-10-01"
               and not p.get("internal")]

    # Weekly weighted-average trend (same estimator as the model)
    trend = []
    d = date(2025, 11, 15)
    today = date.today()
    while d <= today:
        avg, _, rows = model.poll_average(polls, as_of=d)
        if avg is not None and len(rows) >= 5:
            trend.append({"d": d.isoformat(), "m": round(avg, 2)})
        d += timedelta(days=7)

    def hist_bins(h, width):
        agg = {}
        for k, v in h.items():
            b = int(k) // width * width
            agg[b] = agg.get(b, 0) + v
        return [{"x": k, "p": round(v, 5)} for k, v in sorted(agg.items())]

    hm, sm = forecast["house"]["model"], forecast["senate"]["model"]
    races = [{"state": s, "p": r["p_dem_market"], "def": r["incumbent_party"],
              "prior": r["prior_used"]}
             for s, r in sm["races"].items()]
    races.sort(key=lambda r: r["p"])

    data = {
        "as_of": forecast["as_of"],
        "house": {
            "ensemble": forecast["house"]["ensemble_p_dem_control"],
            "model": hm["p_dem_control"],
            "kalshi": forecast["house"]["markets"]["kalshi_p_dem"],
            "poly": forecast["house"]["markets"]["polymarket_p_dem"],
            "seats_mean": hm["dem_seats_mean"], "p10": hm["dem_seats_p10"],
            "p90": hm["dem_seats_p90"],
            "poll_avg": hm["poll_average_margin"], "n_polls": hm["n_polls_used"],
            "hist": hist_bins(hm["seat_histogram"], 4), "majority": 218,
        },
        "senate": {
            "ensemble": forecast["senate"]["ensemble_p_dem_control"],
            "model": sm["p_dem_control"],
            "kalshi": forecast["senate"]["markets"]["kalshi_p_dem"],
            "poly": forecast["senate"]["markets"]["polymarket_p_dem"],
            "seats_mean": sm["dem_seats_mean"], "p10": sm["dem_seats_p10"],
            "p90": sm["dem_seats_p90"],
            "hist": hist_bins(sm["seat_histogram"], 1), "majority": 51,
            "races": races,
        },
        "scatter": scatter,
        "trend": trend,
        "social": {q: {k: v[k] for k in
                       ("n_posts", "posts_per_hour", "net_sentiment")}
                   for q, v in json.loads((snap / "social.json").read_text())
                   .get("queries", {}).items()},
    }

    tpl = (ROOT / "scripts" / "election2026" / "dashboard_template.html").read_text()
    html = tpl.replace("/*__DATA__*/null", json.dumps(data))
    out = ROOT / "election-2026" / "index.html"
    out.write_text(html)
    print(f"wrote {out} ({len(html)//1024}KB), trend points: {len(trend)}, "
          f"scatter polls: {len(scatter)}")


if __name__ == "__main__":
    main()
