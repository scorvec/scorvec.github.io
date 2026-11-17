# scorvec.com

Source of [scorvec.com](https://scorvec.com): weather and climate diagnostics computed from open data in GitHub
Actions and served as a static site on GitHub Pages.

| Theme | Products |
|---|---|
| Forecasts | short-range model viewer (HRRR, RRFS, RDPS), South America and stratosphere model maps, world ensemble maps, jets, atmospheric rivers, warm conveyor belts |
| Tropics | ENSO monitor and forecasts, MJO, equatorial waves, Walker and Hadley circulations |
| Stratosphere | polar vortex, E–P flux, 100 hPa heat flux, sudden-warming outlook, QBO, Brewer–Dobson circulation |
| Weeks to seasons | GEPS and GEFS days 1–35, SEAS5 and the C3S seasonal systems, NOAA SFS, regimes, a month-ahead forecast |
| Studies and tools | model verification, angular-momentum budgets, sounding explorer, airport observations, US climate trends |

Each page links to a guide under [guides/](guides/) with its method, data sources and caveats.

## How it works

```
open data ─► workflow (cron or dispatch) ─► scripts/<product>/ ─► figures + JSON in assets/ ─► pages.yml ─► GitHub Pages
```

- **One workflow per product family** in `.github/workflows/`. A run fetches its inputs, computes the product and
  commits the outputs to `main`.
- **`pages.yml`** assembles the deployed site with `scripts/site/build_site.py` on every push and twice an hour.
- **Animation frames** are served from `frames.scorvec.com`; large state lives on data branches (`frames`,
  `verify-data`, `skewt-data`, …), not on `main`.
- **Shared numerical methods** come from [quantmet](https://github.com/scorvec/quantmet-tools), pinned to a release.

## Run a pipeline

The product's workflow is the reference: it names the Python version, the requirements, the environment variables
and the commands. To run one elsewhere, repeat its steps from the repository root, for example:

```bash
export PIP_CONSTRAINT=$PWD/scripts/constraints.txt SST_SITE_ROOT=$PWD
pip install -r scripts/sst/requirements-sst.txt
python scripts/sst/sst-roni.py            # the ENSO monitor, as sst.yml runs it
```

## Edit the pages

```bash
python scripts/site/apply_chrome.py       # restamp header, menus, footer and asset hashes on every page
python scripts/site/guides.py             # rebuild guides/ from scripts/site/guides/*.html
python scripts/site/build_site.py --check # assemble _site/ and fail on a missing file
ruff check . && ruff format --check . && python scripts/site/style_check.py
```

Pages are hand-written HTML. A new page is one entry in `PRODUCTS` and `PAGES` in `apply_chrome.py`. The multi-view
pages (`enso.html`, `circulation.html`, `stratosphere.html`, …) are generated from `scripts/sst/pages/*.html` by
`scripts/sst/enso_site.py`: edit the partial, not the built page.

## Layout

```
*.html               pages
assets/              site JS and CSS; figures and data written by the workflows
guides/  topics/  studies/   method guides, explainers, study pages
scripts/<product>/   one pipeline per product (sst, mjo, strat, geps, gefs, sfs, models, ensmaps, …)
scripts/lib/         shared helpers (house figure style, frame store, publishing)
scripts/site/        chrome, guides, site build, style check
skewt/  qbo/         self-contained apps
.github/workflows/   schedules
```

## Conventions

- Anomalies are against 1991–2020 unless a page says otherwise.
- Maps and composites show only what passes significance testing (Benjamini–Hochberg for fields).
- ECMWF data comes from the Google Cloud mirror, NOAA models from AWS, ECCC from its Datamart.

## Licence

Code: [MIT](LICENSE). Site text and figures © Shawn Corvec unless noted; third-party data stay under their
providers' terms, credited on each page. Security reports: [SECURITY.md](SECURITY.md).
