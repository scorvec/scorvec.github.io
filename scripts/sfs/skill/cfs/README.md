SFS beta2 vs CFSv2 hindcast skill (2026-10-12, user: "Maybe the better comparison is a comparison with the CFS? Do we
also have an indication of seasonal performance?"). Laptop study; truth from ~/era5_store, Nino-3.4 from
scripts/sst/reference/oisst_box_monthly.json. Intermediate files go to $SFS_CFS_WORK (default /tmp/sfs_cfs).

Part A, weekly (partA_sfs_vs_cfs_weekly.csv): Sep 1 / Oct 1 starts, 1999-2016 + 2018-2025. SFS 4-member means
(all 15 4-of-6 subsets); CFSv2 the 00/06/12/18Z cycles of the 1st, member 1 each (IRI SubX hindcast + real-time;
z500 2019-2025 from NOAA's operational CFS on S3). LOO anomalies, year-block bootstrap (5000).
  -> CFSv2 significantly better in weeks 1-2 (t2m, z500, precipitation); weeks 3-5 no significant difference.

Part B, seasonal (partB_sfs_vs_cfs_seasonal.csv): 1991-2025, SFS 11 members on the 1st vs CFSv2 from the IRI NMME
archive (24 lagged members; scored as random 11-member subsets, full 24 also listed), each against its own LOO
climatology; 1999-split and LOO-detrended variants included.
  -> SFS better for 2 m temperature from Sep starts and global/NH from Oct starts (about half of it trend), DJF
     precipitation from Oct starts, and Nino-3.4 Nov-Feb from Oct starts (RMSE ~0.2 C lower); CFSv2 better into
     spring (Nino-3.4 Apr-May from Oct starts, with CPC's 1999 split). CONUS winter t2m from Oct starts: no difference.

Scripts: c2.py (settings), sfs_fetch2.py / sfs_monthly.py (SFS reforecast), cfs_fetch.py (SubX + S3), cfs_nmme.py
(NMME), era5_truth2.py / era5_seasonal.py (truth), score_cfs.py (Part A), score_seasonal.py (Part B).
