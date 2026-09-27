"""The 'ENSO removed' block of mjo.html (2026-09-27): an animator of the ENSO-removed wind-only RMM runs
(assets/mjo/rmmclean_*z.png, written by run_rmm.py via src/enso_rmm.py), its status line and the method notes.
Called by generate_page.py; returns "" until the first cleaned plot exists, so the page never shows an empty block."""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ASSETS = Path("assets/mjo")
MANIFEST = ASSETS / "rmmclean_manifest.json"
STATUS = ASSETS / "rmm_clean_status.json"
CORR = Path("scripts/mjo/data/reference/enso_rmm_correction.json")
RE = re.compile(r"rmmclean_(\d{8})_(\d{2})z\.png$")


def _items():
    out = []
    for p in ASSETS.glob("rmmclean_*z.png"):
        m = RE.search(p.name)
        if m:
            out.append((m.group(1), m.group(2), p))
    out.sort(key=lambda x: (x[0], x[1]))
    return out


def _label(d, h):
    return f"{d[:4]}-{d[4:6]}-{d[6:8]} {h}Z"


def clean_section() -> str:
    items = _items()
    if not items:
        return ""
    frames = [{"idx": i, "file": p.name, "date": f"{d[:4]}-{d[4:6]}-{d[6:8]}", "label": _label(d, h)}
              for i, (d, h, p) in enumerate(items)]
    # region id must be the folder name: sst_anim.html loads {base}/{region}/{file} = assets/mjo/<file>
    MANIFEST.write_text(json.dumps({"ver": int(datetime.now(timezone.utc).timestamp()), "days": len(frames),
                                    "regions": {"mjo": {"label": "ENSO-removed wind-only RMM — successive runs",
                                                             "n_frames": len(frames), "frames": frames}}}))
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    try:
        from enso_rmm import status_sentence
        status = status_sentence(json.loads(STATUS.read_text())) if STATUS.exists() else ""
    except Exception:                                                                 # noqa: BLE001
        status = ""
    v = json.loads(CORR.read_text()) if CORR.exists() else {}
    val = v.get("validation", {})
    a = val.get("vs_bom_2003_2013_enso_removed", {})
    g = val.get("winters_and_growth_phases", {})
    vtxt = ""
    if a.get("raw") and a.get("clean_f"):
        r, c = a["raw"], a["clean_f"]
        vtxt = (f"Against BoM&rsquo;s ENSO-removed RMM (2003&ndash;2013, {r['n']:,} days) the cleaned wind-only index "
                f"correlates {c['r1']:.2f}/{c['r2']:.2f} (RMM1/RMM2), against {r['r1']:.2f}/{r['r2']:.2f} before, and "
                f"its correlation with the Ni&ntilde;o-3.4 change falls from {r['corr_dn34_r1']:+.2f} to "
                f"{c['corr_dn34_r1']:+.2f}.")
        gr = [(k, x) for k, x in g.items() if "growth" in k]
        if gr:
            vtxt += " While an El Ni&ntilde;o grows, it lowers the share of &ldquo;active&rdquo; days: " + "; ".join(
                f"{k.replace(' (growth)', '')} {x['raw']['active']:.2f} &rarr; {x['clean_f']['active']:.2f}" for k, x in gr) + "."
    d, h, _ = items[-1]
    return f"""
  <h2>MJO with the El Ni&ntilde;o signal removed</h2>
  <p class="lede lede--wide">Since 2014 the standard RMM no longer removes ENSO; in a strong El Ni&ntilde;o its standing
  convection projects onto phases 7&ndash;8 and masquerades as a stalled MJO. This is the same AIFS-ENS forecast
  (wind-only), observed track and IFS overlay with the part tied to the Ni&ntilde;o-3.4 change removed. Latest init:
  <strong>{_label(d, h)}</strong>.</p>
  {f'<p class="lede lede--wide">{status}</p>' if status else ''}
  <iframe class="anim-embed" src="sst_anim.html?embed=1&amp;base=assets&amp;manifest=mjo/rmmclean_manifest.json&amp;region=mjo"
    title="ENSO-removed wind-only RMM — successive runs animation" loading="lazy"></iframe>
  <details style="margin-top:1rem;max-width:80ch;font-size:0.92rem"><summary style="cursor:pointer;color:var(--accent)">How the El Ni&ntilde;o signal is removed</summary>
  <p>Wheeler &amp; Hendon (2004) removed two things before projecting onto the MJO patterns: the part of each day&rsquo;s
  anomaly linearly related to ENSO, then the mean of the previous 120 days. The Bureau of Meteorology did both until the
  end of 2013; since 2014 it removes only the 120-day mean, as this page&rsquo;s raw index does. While an El Ni&ntilde;o is
  growing, the 120-day mean trails the warming, so the leftover projects onto the MJO phases.</p>
  <p>Here the leftover is taken out directly: the wind-only RMM minus &Delta;N34&nbsp;&times;&nbsp;c, where &Delta;N34 is
  Ni&ntilde;o-3.4 (NOAA OISST, 31-day mean) minus its mean over the same 120-day window, and c is the regression of the
  120-day-filtered 850 and 200&nbsp;hPa equatorial winds on &Delta;N34, longitude by longitude, projected onto the MJO
  patterns. It is fitted on ERA5 (ARCO) 2003&ndash;2024 with ERSST&nbsp;v6 Ni&ntilde;o-3.4 (BoM used its own SST index,
  SST1). {vtxt} The forecast holds Ni&ntilde;o-3.4 at its latest value, since ENSO changes little in 15 days.</p>
  <p>The correction is linear, so it removes part of a growing El Ni&ntilde;o&rsquo;s projection, not all of it. The
  status line therefore also quotes NOAA&rsquo;s OLR-based real-time OMI (ROMI), which removes the previous 40 days and
  so is far less sensitive to slow signals: an independent check on whether a real MJO is present.
  Wind-only (no precipitation channel), and no 12Z cycle offset correction, unlike the raw diagram above.</p>
  <p class="meta" style="margin-top:0.6rem;padding-top:0.6rem">Data: ECMWF AIFS-ENS and IFS-ENS open data (CC&nbsp;BY&nbsp;4.0);
  ERA5 (Copernicus C3S, ARCO); NOAA OISST v2.1 and ERSST v6 (NOAA NCEI / PSL); ROMI (NOAA PSL / CPC OLR);
  Bureau of Meteorology RMM for validation.</p>
"""
