"""One-off analysis of public generic-ballot crosstabs (YouGov weekly tracker).

Needs pandas + openpyxl (not part of the daily pipeline). Downloads live in
data/election2026/crosstabs/. Prints a report and writes crosstabs_summary.json
used by the write-up; nothing here feeds the forecast directly.

YouGov tracker: one sheet per subgroup, columns = weekly waves 2017->present,
rows = Dem / Rep / Other / Not sure / would-not-vote / bases.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
XLSX = ROOT / "data" / "election2026" / "crosstabs" / "yougov_tracker.xlsx"

ROWS = {"The Democratic Party candidate": "dem",
        "The Republican Party candidate": "rep",
        "Other": "other", "Not sure": "notsure",
        "I would not vote": "novote"}


def load_sheet(xl, name):
    df = xl.parse(name, header=None)
    dates = pd.to_datetime(df.iloc[0, 1:], errors="coerce")
    out = {}
    for i in range(1, len(df)):
        key = ROWS.get(str(df.iloc[i, 0]).strip())
        if key:
            out[key] = pd.to_numeric(df.iloc[i, 1:], errors="coerce").values
    return dates.values, out


def wave_stats(series, idx):
    d, r, ns = series["dem"][idx], series["rep"][idx], series["notsure"][idx]
    return {"dem": round(float(d) * 100, 1), "rep": round(float(r) * 100, 1),
            "margin": round(float(d - r) * 100, 1),
            "notsure": round(float(ns) * 100, 1)}


def nearest(dates, when):
    target = pd.Timestamp(when)
    return int(pd.Series(dates).sub(target).abs().idxmin())


def main():
    xl = pd.ExcelFile(XLSX)
    report = {"latest_wave": None, "subgroups": {}, "cycle_2022": {},
              "cycle_2026_trend": {}}

    for name in xl.sheet_names:
        dates, s = load_sheet(xl, name)
        last = len(dates) - 1
        report["subgroups"][name] = wave_stats(s, last)
        if name == "US Registered Voters":
            report["latest_wave"] = str(pd.Timestamp(dates[last]).date())
            # 2022 analog: how Not-sure and margin moved July -> election eve
            for label, when in [("jul_2022", "2022-07-08"),
                                ("sep_2022", "2022-09-08"),
                                ("eve_2022", "2022-11-07"),
                                ("jan_2026", "2026-01-08"),
                                ("apr_2026", "2026-04-08")]:
                i = nearest(dates, when)
                tgt = report["cycle_2022"] if "2022" in label else report["cycle_2026_trend"]
                tgt[label] = {"wave": str(pd.Timestamp(dates[i]).date()),
                              **wave_stats(s, i)}
        if name == "Independent":
            for label, when in [("jul_2022", "2022-07-08"),
                                ("eve_2022", "2022-11-07")]:
                i = nearest(dates, when)
                report["cycle_2022"][f"ind_{label}"] = {
                    "wave": str(pd.Timestamp(dates[i]).date()), **wave_stats(s, i)}

    out = XLSX.parent / "crosstabs_summary.json"
    out.write_text(json.dumps(report, indent=2))

    print(f"YouGov tracker, latest wave {report['latest_wave']}\n")
    print(f"{'subgroup':<20}{'Dem':>6}{'Rep':>6}{'margin':>8}{'not sure':>10}")
    order = sorted(report["subgroups"], key=lambda k: -report["subgroups"][k]["notsure"])
    for name in order:
        v = report["subgroups"][name]
        print(f"{name:<20}{v['dem']:>6}{v['rep']:>6}{v['margin']:>+8.1f}{v['notsure']:>9.1f}%")
    print("\n2022 cycle analog (US RVs):")
    for k, v in report["cycle_2022"].items():
        print(f"  {k:<14} {v['wave']}  D{v['margin']:+.1f}  not sure {v['notsure']}%")
    print("\n2026 cycle so far (US RVs):")
    for k, v in report["cycle_2026_trend"].items():
        print(f"  {k:<14} {v['wave']}  D{v['margin']:+.1f}  not sure {v['notsure']}%")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
