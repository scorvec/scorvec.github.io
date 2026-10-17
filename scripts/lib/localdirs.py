"""Folders of the offline builders' local inputs, each named by an environment variable with no default.

    ERA5_STORE    the ERA5 store kept by scripts/era5/ (WeatherBench 2 daily layers, ARCO-ERA5 0.25 deg fields)
    DATA_ARCHIVE  archived model data and reference work folders (GEPS and GEFS hindcasts, CMIP6 extracts, ...)
    MJO_CACHE     the MJO and torque builders' caches (reduced ERA5 samples, ESMGFZ angular-momentum series)

A builder resolves its folders at start and stops with a clear message when a variable is unset:

    sys.path.insert(0, str(REPO / "scripts" / "lib"))
    from localdirs import env_dir
    STORE = env_dir("ERA5_STORE", "wb2_1p5_daily")

A module that also runs in Actions calls env_dir where the folder is used, never at import, and a workflow whose run
reaches that code sets the variable itself (ERA5_STORE: ${{ runner.temp }}/era5).
"""

from __future__ import annotations

import os
from pathlib import Path

WHAT = {
    "ERA5_STORE": "the ERA5 store (scripts/era5/)",
    "DATA_ARCHIVE": "the archive of model data and reference work folders",
    "MJO_CACHE": "the MJO and torque builders' cache folder",
}


def env_dir(var: str, *parts: str) -> Path:
    """$var joined with `parts`; exits with a clear message when the variable is unset (there is no default)."""
    d = os.environ.get(var, "")
    if not d:
        raise SystemExit(f"{var} is not set: point it at {WHAT.get(var, 'the folder it names')}")
    return Path(d).expanduser().joinpath(*parts)
