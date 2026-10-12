"""Shared settings for the SFS beta2 vs CFSv2 hindcast study (2026-10-12). Laptop study; the large intermediate files
(raw GRIB/netCDF, *.npz) go to $SFS_CFS_WORK, never into the repo."""
import os
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import MONTHS, WINDOWS, WNAMES, ELAT, ELON, weekly, regrid  # noqa: F401,E402

OUT = os.environ.get("SFS_CFS_WORK", "/tmp/sfs_cfs") + "/"
OLD = str(HERE.parent)          # the GEFS/GEPS study (sfs_fetch.one, common.py)
# CFSv2 on IRI SubX: hindcast 1999-2016 (4 cycles/day x 1 member), real-time from 2017-10-15 (4 cycles x 4 members)
YEARS = list(range(1999, 2017)) + list(range(2018, 2026))
