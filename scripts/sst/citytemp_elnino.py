#!/usr/bin/env python3
"""Hourly-temperature timeseries for 8 US cities across the four super-El-Niño years
(plus the current event), each over a May 1 → following-May 1 window aligned by
day-of-window so the developing/peaking/decaying El Niño years overlay directly.

Data: IEM ASOS hourly air temperature (tmpf), one primary station per metro, reusing
the IEM request idiom from kiribati_history.py. Raw pulls are cached per station-window
(1982-era data is pulled once). One-off review figure — not wired to a page yet.

    python scripts/sst/citytemp_elnino.py            # -> citytemp_elnino.png
"""
from __future__ import annotations

import io
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent                          # repo root
CACHE = HERE / "citytemp_cache"
OUT = ROOT / "assets" / "sst" / "citytemp" / "citytemp_elnino.webp"   # base; figures go alongside

# Primary IEM ASOS station per metro (all have hourly back to the 1970s/80s) + its IEM
# network (for the daily-summary endpoint used to build the detrended climatology).
CITIES = [
    ("Houston",      "IAH", "TX_ASOS"),
    ("Dallas",       "DFW", "TX_ASOS"),
    ("Phoenix",      "PHX", "AZ_ASOS"),
    ("Washington DC","DCA", "VA_ASOS"),
    ("New York City","LGA", "NY_ASOS"),   # LaGuardia (Central Park 'NYC' lacks 1982/1991 hourly)
    ("Boston",       "BOS", "MA_ASOS"),
    ("Portland OR",  "PDX", "OR_ASOS"),
    ("Seattle",      "SEA", "WA_ASOS"),
]

# onset year → (label, color). May 1 of onset year onward; current event drawn bold on top.
EVENTS = {
    1982: ("1982", "#1f77b4"),   # blue
    1991: ("1991", "#2ca02c"),   # green
    1997: ("1997", "#8c564b"),   # brown
    2015: ("2015", "#9467bd"),   # purple
    2026: ("2026 (current)", "#d62728"),   # red, bold, on top (partial: May → present)
}

MAXDAY = 245          # May 1 (day 0) … Dec 31 (day 244); restrict the window here

IEM = ("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"
       "station={stn}&data=tmpf&"
       "year1={y1}&month1=5&day1=1&year2={y2}&month2=5&day2=1&"
       "tz=Etc/UTC&format=onlycomma&missing=M")


def fetch(stn: str, onset: int, force: bool = False) -> pd.DataFrame:
    """Hourly tmpf for one station over [May 1 onset, May 1 onset+1). Cached on disk
    (historical onsets never change); `force` bypasses the cache for the current year so
    its line keeps extending. Fails soft (rate-limit backoff / network error → empty)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    fp = CACHE / f"{stn}_{onset}.csv"
    if not force and fp.exists() and fp.stat().st_size > 0:
        df = pd.read_csv(fp)
    else:
        url = IEM.format(stn=stn, y1=onset, y2=onset + 1)
        txt = None
        for attempt in range(5):
            try:
                txt = urllib.request.urlopen(url, timeout=180).read().decode()
            except Exception as e:                                  # noqa: BLE001
                print(f"    {stn} {onset}: fetch error {repr(e)[:60]}; retry", flush=True)
                time.sleep(15); continue
            if "Too many requests" in txt:
                print(f"    {stn} {onset}: rate-limited, backing off", flush=True)
                time.sleep(25); txt = None; continue
            break
        if not txt:
            return pd.DataFrame(columns=["valid", "tmpf"])
        df = pd.read_csv(io.StringIO(txt))
        df.to_csv(fp, index=False)
        time.sleep(5)                                                # be gentle to IEM
    if "tmpf" not in df.columns:
        return pd.DataFrame(columns=["valid", "tmpf"])
    df["valid"] = pd.to_datetime(df["valid"], errors="coerce")
    df["tmpf"] = pd.to_numeric(df["tmpf"], errors="coerce")
    return df.dropna(subset=["valid", "tmpf"])


def day_of_window(valid: pd.Series, onset: int) -> np.ndarray:
    """Fractional days since May 1 00Z of the onset year (aligns all events on 0..365)."""
    start = pd.Timestamp(f"{onset}-05-01", tz="UTC")
    return (valid.dt.tz_localize("UTC") - start).dt.total_seconds().to_numpy() / 86400.0


HIST = [1982, 1991, 1997, 2015]      # one comparison panel per historical super-Niño
CURRENT = 2026                        # overlaid (red) in every panel as the reference


def daily_stats(x, t) -> pd.DataFrame:
    """Hourly (day-of-window, °F) → per-day min/max/mean over May 1 … Dec 31."""
    g = pd.DataFrame({"d": np.floor(x).astype(int), "t": t})
    g = g[(g.d >= 0) & (g.d <= MAXDAY)].groupby("d")["t"]
    return pd.DataFrame({"lo": g.min(), "hi": g.max(), "mean": g.mean()})


def _draw(ax, d: pd.DataFrame, color: str, off: float, lw: float, alpha: float, z: int):
    """Daily high–low range as a vertical bar per day (+ a daily-mean line on top)."""
    if d.empty:
        return
    ax.vlines(d.index + off, d["lo"], d["hi"], color=color, lw=lw, alpha=alpha, zorder=z)
    ax.plot(d.index + off, d["mean"], color=color, lw=1.3, alpha=min(1.0, alpha + 0.2),
            zorder=z + 1, solid_capstyle="round")


def render_city(city: str, stn: str, daily_by_onset: dict, out: Path):
    """2×2 figure: each panel pits one historical super-Niño against current 2026, showing
    the hourly-derived daily high–low range (bars) + daily-mean line for each."""
    fig, axes = plt.subplots(2, 2, figsize=(17, 11), sharex=True, sharey=True)
    axes = axes.ravel()
    month_starts = pd.date_range("2001-05-01", "2002-01-01", freq="MS")
    ticks = [(d - pd.Timestamp("2001-05-01")).days for d in month_starts]
    labels = [d.strftime("%b 1") for d in month_starts]
    cur = daily_by_onset.get(CURRENT, pd.DataFrame())
    cur_col = EVENTS[CURRENT][1]

    for ax, H in zip(axes, HIST):
        hcol = EVENTS[H][1]
        _draw(ax, daily_by_onset.get(H, pd.DataFrame()), hcol, off=-0.18, lw=2.0, alpha=0.45, z=3)
        _draw(ax, cur, cur_col, off=0.18, lw=2.0, alpha=0.7, z=6)
        ax.set_xlim(0, MAXDAY)
        ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=10)
        ax.set_xticks(range(0, MAXDAY, 10), minor=True)
        ax.grid(True, which="major", alpha=0.22)
        ax.grid(True, which="minor", axis="x", alpha=0.08)
        ax.set_ylabel("Temperature (°F)", fontsize=11)
        ax.set_title(f"{H}  vs  2026", fontsize=13, loc="left", fontweight="bold")
        handles = [plt.Line2D([], [], color=hcol, lw=3, label=str(H)),
                   plt.Line2D([], [], color=cur_col, lw=3, label="2026 (current)")]
        ax.legend(handles=handles, loc="lower center", ncol=2, fontsize=10,
                  framealpha=0.85, borderpad=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    fig.suptitle(f"{city}  ({stn}) — daily temperature range vs each super-El-Niño year\n"
                 "bars = daily high–low (from hourly obs) · line = daily mean · May 1 → Dec 31",
                 fontsize=15, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"saved {out}", flush=True)


def slug(city: str) -> str:
    return city.lower().replace(" ", "_")


# ─── Detrended-anomaly product (removes the seasonal cycle AND the secular warming
#     trend, so each event is measured against its own era's normal) ──────────────
IEM_DAILY = ("https://mesonet.agron.iastate.edu/cgi-bin/request/daily.py?"
             "network={net}&stations={stn}&year1=1973&month1=1&day1=1&"
             "year2={yr}&month2=12&day2=31&var=max_temp_f&var=min_temp_f&na=blank&format=comma")
TREND_REF = 2000.0          # year the detrended anomaly is referenced to (offset only)


def fetch_daily(stn: str, net: str) -> pd.DataFrame:
    """Full daily max/min record (1973→present) for one station. Always re-fetched (one
    request) so the current-year tail extends; the climatology/trend barely move."""
    CACHE.mkdir(parents=True, exist_ok=True)
    fp = CACHE / f"daily_{stn}.csv"
    cur_yr = pd.Timestamp.now().year
    url = IEM_DAILY.format(net=net, stn=stn, yr=cur_yr)
    txt = None
    for _ in range(5):
        try:
            txt = urllib.request.urlopen(url, timeout=300).read().decode()
        except Exception as e:                                      # noqa: BLE001
            print(f"    daily {stn}: fetch error {repr(e)[:60]}; retry", flush=True)
            time.sleep(15); continue
        if "Too many requests" in txt:
            time.sleep(25); txt = None; continue
        break
    if not txt:
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(txt))
    df.to_csv(fp, index=False); time.sleep(5)
    if "max_temp_f" not in df.columns:
        return pd.DataFrame()
    out = pd.DataFrame({"max": pd.to_numeric(df["max_temp_f"], errors="coerce").to_numpy(),
                        "min": pd.to_numeric(df["min_temp_f"], errors="coerce").to_numpy()},
                       index=pd.to_datetime(df["day"], errors="coerce")).dropna().sort_index()
    return out[~out.index.duplicated()]


def _detrend(s: pd.Series) -> pd.Series:
    """Daily series → detrended anomaly: subtract a smoothed day-of-year climatology and
    the long-term linear trend (referenced to TREND_REF)."""
    s = s.dropna()
    dm = s.groupby(s.index.dayofyear).mean()                       # raw doy normal (1..366)
    ext = pd.concat([dm, dm, dm])                                  # wrap for ±-day smoothing
    sm = ext.rolling(15, center=True, min_periods=1).mean().iloc[len(dm):2 * len(dm)]
    sm.index = dm.index
    resid = s.to_numpy() - s.index.dayofyear.map(sm).to_numpy()    # de-seasonalised
    dy = s.index.year + (s.index.dayofyear - 1) / 365.25
    b = np.polyfit(dy, resid, 1)[0]                                # °F per year
    return pd.Series(resid - b * (dy - TREND_REF), index=s.index)


def detrended_anom(daily: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=daily.index)
    out["max_anom"] = _detrend(daily["max"])
    out["min_anom"] = _detrend(daily["min"])
    return out.dropna()


def event_anom(anom: pd.DataFrame, onset: int) -> pd.DataFrame:
    start = pd.Timestamp(f"{onset}-05-01")
    sub = anom[(anom.index >= start) & (anom.index <= pd.Timestamp(f"{onset}-12-31"))]
    if sub.empty:
        return pd.DataFrame(columns=["lo", "hi", "mean"])
    day = (sub.index - start).days
    return pd.DataFrame({"lo": sub["min_anom"].to_numpy(), "hi": sub["max_anom"].to_numpy(),
                         "mean": (sub["min_anom"].to_numpy() + sub["max_anom"].to_numpy()) / 2},
                        index=day)


SMOOTH = 7        # days; rolling window for the mean-anomaly line


def _draw_anom(ax, d, color, bar_alpha, line_lw, line_alpha, z):
    """Faint daily high–low anomaly bars (spread context) + a bold SMOOTH-day rolling
    mean-anomaly line (the ENSO-scale tendency)."""
    if d.empty:
        return
    ax.vlines(d.index, d["lo"], d["hi"], color=color, lw=1.0, alpha=bar_alpha, zorder=z)
    m = d["mean"].rolling(SMOOTH, center=True, min_periods=2).mean()
    ax.plot(d.index, m.to_numpy(), color=color, lw=line_lw, alpha=line_alpha,
            zorder=z + 2, solid_capstyle="round")


def render_anomaly_city(city: str, stn: str, anom_by_onset: dict, out: Path):
    """2×2 anomaly figure: each panel = one historical super-Niño vs 2026, detrended
    (warming removed). Faint bars = daily high–low anomaly; bold line = 7-day mean anomaly."""
    fig, axes = plt.subplots(2, 2, figsize=(17, 11), sharex=True, sharey=True)
    axes = axes.ravel()
    month_starts = pd.date_range("2001-05-01", "2002-01-01", freq="MS")
    ticks = [(d - pd.Timestamp("2001-05-01")).days for d in month_starts]
    labels = [d.strftime("%b 1") for d in month_starts]
    cur, cur_col = anom_by_onset.get(CURRENT, pd.DataFrame()), EVENTS[CURRENT][1]

    for ax, H in zip(axes, HIST):
        hcol = EVENTS[H][1]
        ax.axhline(0, color="#444", lw=1.2, zorder=2)
        _draw_anom(ax, anom_by_onset.get(H, pd.DataFrame()), hcol,
                   bar_alpha=0.14, line_lw=2.4, line_alpha=0.9, z=3)
        _draw_anom(ax, cur, cur_col, bar_alpha=0.22, line_lw=3.2, line_alpha=1.0, z=6)
        ax.set_xlim(0, MAXDAY)
        ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=10)
        ax.set_xticks(range(0, MAXDAY, 10), minor=True)
        ax.grid(True, which="major", alpha=0.22)
        ax.grid(True, which="minor", axis="x", alpha=0.08)
        ax.set_ylabel("Temp anomaly (°F)", fontsize=11)
        ax.set_title(f"{H}  vs  2026", fontsize=13, loc="left", fontweight="bold")
        handles = [plt.Line2D([], [], color=hcol, lw=3, label=str(H)),
                   plt.Line2D([], [], color=cur_col, lw=3, label="2026 (current)")]
        ax.legend(handles=handles, loc="lower center", ncol=2, fontsize=10,
                  framealpha=0.85, borderpad=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    yr = pd.Timestamp.now().year
    fig.suptitle(f"{city}  ({stn}) — detrended temperature anomaly vs each super-El-Niño year\n"
                 f"departure from the trend-adjusted seasonal normal (1973–{yr}; warming removed) · "
                 f"bold line = {SMOOTH}-day mean anomaly · faint bars = daily high–low · May 1 → Dec 31",
                 fontsize=14, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"saved {out}", flush=True)


def main() -> int:
    for city, stn, net in CITIES:
        # absolute figure (hourly-derived daily high–low range)
        daily_by_onset = {}
        for onset in EVENTS:
            df = fetch(stn, onset, force=(onset == CURRENT))
            if df.empty:
                print(f"  {city} ({stn}) {onset}: no hourly data", flush=True); continue
            x = day_of_window(df["valid"], onset)
            daily_by_onset[onset] = daily_stats(x, df["tmpf"].to_numpy())
        render_city(city, stn, daily_by_onset, OUT.with_name(f"citytemp_{slug(city)}.webp"))

        # detrended-anomaly figure (one full daily record per station covers climo + all events)
        daily = fetch_daily(stn, net)
        if daily.empty:
            print(f"  {city} ({stn}): no daily record — skipping anomaly", flush=True); continue
        anom = detrended_anom(daily)
        anom_by_onset = {o: event_anom(anom, o) for o in EVENTS}
        render_anomaly_city(city, stn, anom_by_onset, OUT.with_name(f"citytemp_anom_{slug(city)}.webp"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
