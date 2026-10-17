# scorvec.com

Source of [scorvec.com](https://scorvec.com), a weather and climate diagnostics site: real-time products for the
mid-latitudes, the tropics and the stratosphere, sub-seasonal and seasonal outlooks, and the studies behind them.
Every product is computed from open data by the pipelines in this repository and published as a static site on
GitHub Pages.

## What is on the site

| Theme | Products (examples) |
|---|---|
| Mid-latitude weather | short-range model viewer (HRRR, RRFS, RDPS), dynamic tropopause, North Pacific jet, atmospheric rivers, wave-activity flux |
| Tropics | ENSO monitor and forecasts (ERSSTv6, official ONI/RONI), MJO, equatorial waves and Kelvin waves, Walker circulation by basin, tropical-cyclone shear and environment |
| Stratosphere | polar vortex and E–P flux (AIFS-ENS, IFS-ENS, GEFS), 100 hPa heat flux, QBO, Brewer–Dobson circulation, sudden-warming history |
| Weeks to seasons | GEPS and GEFS days 1–35, SEAS5 and the C3S multi-system seasonal maps, NOAA SFS, regimes and teleconnections, Claude's statistical month-ahead forecast |
| Studies and verification | AIFS verification, angular-momentum budgets of AI and physics models, El Niño and angular momentum |
| Tools | searchable plot catalogue, sounding explorer, five-minute airport observations, US climate trends, US credit conditions |

Methods, data sources and citations are documented on each page; [topics/](topics/) holds the explainers.

## How it is built

```
open data ──► GitHub Actions workflows ──► scripts/<product>/ (Python) ──► figures + JSON ──► GitHub Pages
                     (cron + dispatch)                                       │
                                                                             └► animation frames ──► object store (CDN)
```

- **Workflows** (`.github/workflows/`): one per product family, on cron schedules, most with a concurrency group so
  a product never builds twice at once. Each run fetches its inputs, computes the diagnostics, and commits the outputs
  with a snapshot-and-resync step (outputs are copied aside, the checkout is reset to the current `main`, and the
  outputs are restored before committing), so concurrent runs never overwrite each other's results.
- **Pipelines** (`scripts/<product>/`): Python 3.11/3.12 with xarray, numpy/scipy, matplotlib/cartopy and eccodes.
  Shared numerical methods come from [quantmet](https://github.com/scorvec/quantmet-tools), pinned to a release tag.
- **Animation frames** go to an S3-compatible object store served from `frames.scorvec.com`
  (`scripts/lib/frames_store.py`); large state and archives live on dedicated data branches (`frames`,
  `verify-data`, `skewt-data`, `skewt-archive`, …) rather than on `main`.
- **Pages**: hand-written HTML with shared chrome stamped in by `scripts/site/apply_chrome.py` (menus, header and
  footer from one product table). The multi-product "stage" pages are generated from partials in
  `scripts/sst/pages/` by `scripts/sst/enso_site.py`.

## Data

Inputs are open datasets. Real-time inputs are fetched by the workflows; static reference climatologies are built
once from the same archives. Sources include ECMWF open data (IFS, AIFS; Google Cloud mirror), NOAA
(GFS/GEFS/HRRR/RRFS on AWS, CPC, PSL, NCEI), ECCC (GDPS/GEPS/RDPS), Copernicus C3S/CDS (ERA5, SEAS5 and the
C3S seasonal systems), NASA (MERRA-2, GEOS FP, IMERG) and others. Each page credits its sources and their licences.

## Statistical conventions

Anomalies are against 1991–2020 unless a page says otherwise. Fields and composites are shown only where they pass
significance testing (Benjamini–Hochberg false-discovery-rate control for maps); results that do not are not
published.

## Repository layout

```
*.html               site pages
assets/              site JS/CSS, generated figures and data (written by the workflows)
scripts/<product>/   the pipeline of each product (sst, mjo, strat, geps, gefs, sfs, models, skewt, …)
scripts/lib/         shared helpers (frame store, locking, publishing)
scripts/site/        site chrome, catalogue and topic-page generators
scripts/watch/       upstream-change watches (GFS v17 upgrade notices; opens an issue when something is new)
topics/  studies/    explainers and study pages
skewt/   qbo/        self-contained apps
.github/workflows/   the schedules
```

## Running a pipeline

Each product's workflow is the reference for how it runs: it lists the Python version, the requirements file
(`scripts/*/requirements*.txt` or inline `pip install`), the environment variables and the exact commands. To run
one outside Actions, install the same requirements and run the same commands from the repository root.

## Security

Report security problems privately; see [SECURITY.md](SECURITY.md).

## License

Code: [MIT](LICENSE). Site text and figures © Shawn Corvec unless noted; third-party data remain under their
providers' terms, as credited on each page.
