# Sounding explorer

Code behind [scorvec.com/skewt](https://scorvec.com/skewt/): pick any radiosonde station and get a skew-T,
hodograph, parcel and shear analysis, and where each value ranks in that station's own record.

The page has no backend. The physics is [SHARPlib](https://github.com/keltonhalbert/SHARPlib) compiled to
WebAssembly and run in the browser; scheduled jobs only mirror soundings and precompute climatologies.
What the panels show and how each number is computed: [methodology.html](https://scorvec.com/skewt/methodology.html).

## Files

```
skewt/
  index.html, app.js      the page: sources, parsers, WASM bridge, drawing, climatology panels
  sharplib.{js,wasm}      SHARPlib + the wrapper, built by Emscripten
  stations.json           IGRA station list; iem_raob.json maps WMO to ICAO ids
  methodology.html        the method page
  gaps.html, gaps.json    stations that stopped reporting

scripts/skewt/
  skewt_wasm.cpp          the wrapper: one compute_sounding() for the browser and the climatology
  build_wasm.sh           rebuild sharplib.{js,wasm} and restamp the page's asset versions
  mirror_soundings.py     recent launches from IEM, ECCC Datamart and the WMO WIS 2.0 cache
  archive_days.py         complete UTC days bundled as day-YYYYMMDD.zip
  build_climo.py          per-station percentiles and records from the IGRA period of record
  climo_cape.cpp          native build of the same wrapper, used by build_climo.py
  flag_anomalies.py       the map's record watch: latest sounding against its climatology
  sounding_gaps.py        builds gaps.json
```

## Data

The app tries sources from freshest to oldest and names the one in use: SPC observed (US), IEM RAOB (US and
Canada), the mirror (global, hourly), day bundles, then NOAA IGRA v2 (global, back to the 1930s).

| Branch | Holds | Written by |
|---|---|---|
| `skewt-data` | last 96 h of soundings, the manifest, `anomalies.json` | `skewt-data.yml`, every run replaces it |
| `skewt-archive` | `day-YYYYMMDD.zip` day bundles, append-only | `skewt-data.yml` |
| `skewt-climo` | `climo/{station}.json`, percentiles and records at 73 day-of-year anchors, 00Z and 12Z apart | `skewt-climo.yml` (dispatch) |

## Use

```bash
python -m http.server 8000                 # from the repository root, then open /skewt/
scripts/skewt/build_wasm.sh                # after changing skewt_wasm.cpp; needs Emscripten (em++)
python scripts/skewt/mirror_soundings.py OUT_DIR PREVIOUS_DIR   # one mirror pass, as skewt-data.yml runs it
```

Rebuild the climatology (`skewt-climo.yml`) whenever `skewt_wasm.cpp` changes a ranked quantity, so live values
and their percentiles come from the same code.

## Rules the climatology follows

- A sounding with no CAPE counts as zero ECAPE, not missing, so percentiles describe all days.
- No filled-in data: a sounding is used only with a real surface dew point, a top at or above 400 hPa and dew
  point on at least 60 % of levels below it.
- A value is ranked only against launches at its own synoptic hour, and flagged only with 30 or more samples.

## Credits

Not for operational use. Physics by SHARPlib (K. Halbert, NWS SPC). Data: NOAA/NWS SPC; the Iowa Environmental
Mesonet, Iowa State University; Environment and Climate Change Canada (MSC Datamart, ECCC Data Servers End-use
Licence); the WMO WIS 2.0 Global Cache (WMO Unified Data Policy, national services credited per sounding);
NOAA NCEI Integrated Global Radiosonde Archive v2. Built by Claude (Anthropic), directed by Shawn Corvec.
