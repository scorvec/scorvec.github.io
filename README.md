# UW sounding day archive (skewt-archive branch)

Append-only vault for the Skew-T Explorer (scorvec.com/skewt/): one
`uw-YYYYMMDD.zip` per complete UTC day, each holding that day's mirrored
University of Wyoming BUFR/GTS soundings as thinned CSVs
(`{wmo}_{YYYYMMDDHH}.csv`). Written once daily by skewt-data.yml; never
rewritten, so pushes stay incremental. Fetched by the browser via
raw.githubusercontent.com (CORS-open) and unzipped client-side.
