"""Which C3S systems the ENSO forecasts page leaves out — the ONE list.

Read by c3s_nino34.py (plume, enso_forecast.json, the static figure), c3s_evolution.py (issue by
issue), c3s_enso_flavour.py (east/central forecast and the hindcast test), enso_flavour_obs.py (the
daily OISST step) and, through enso_site.render_all's __ENSO_EXCLUDE__ token, the page's own JS.
Changing the set is a one-line edit; the laptop products follow at their next build, the page and
the daily step at the next sst.yml run.

Labels are the display labels in c3s_nino34.MODELS / MERGE (a merged system by its merged label).
"""
EXCLUDE_SYSTEMS = {"DWD GCFS2", "BoM ACCESS-S2"}   # site owner's decision 2026-09-27
EXCLUDE_NOTE = "excluded at the site owner's discretion: poor ENSO skill and drift"


def keep(label: str) -> bool:
    return label not in EXCLUDE_SYSTEMS
