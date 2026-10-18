SFS beta2 vs GEFSv12 vs GEPS8 weekly hindcast skill.

38 starts (Sep 1 / Oct 1, 2001-2019), 4-member means, each system against its own leave-one-year-out hindcast
climatology, ERA5 truth from the ERA5 daily store (NH), year-block bootstrap 95% CIs. Result table:
hindcast_skill_sepoct_2001_2019.csv (var, window, region, metric, system, value, lo, hi, sig). Offline scripts
(they read the ERA5 daily store and a GEPS hindcast archive): sfs_fetch.py, gefs_fetch.py, geps_era5.py, score.py.
The SFS page draws its hindcast-skill figure from the CSV (scripts/sfs/sfs_skillfig.py).
