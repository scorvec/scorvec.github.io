SFS beta2 vs GEFSv12 vs GEPS8 weekly hindcast skill (2026-10-12, user: "Can you do any measurements on how its
subseasonal forecasts perform vs the GEPS/GEFS using hindcasts?").

38 starts (Sep 1 / Oct 1, 2001-2019), 4-member means, each system against its own leave-one-year-out hindcast
climatology, ERA5 truth from the local store (NH), year-block bootstrap 95% CIs. Result table:
hindcast_skill_sepoct_2001_2019.csv (var, window, region, metric, system, value, lo, hi, sig). Laptop scripts
(they read ~/era5_store and the local GEPS hindcast archive): sfs_fetch.py, gefs_fetch.py, geps_era5.py, score.py.
The SFS page draws its hindcast-skill figure from the CSV (scripts/sfs/sfs_skillfig.py).
