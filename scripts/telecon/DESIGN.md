# NH Teleconnection Engine — design

Agreed 2026-07-26. A conditioned-composite + analog forecasting product:
given today's ENSO x MJO x GWO(NH) state, show the canonical day-10-35
NH pattern, the WAF wave path it takes, whether the wave train is already
en route, and whether AIFS-ENS week 2 is following or fighting it.

## Decisions

- **NH focus** (user, 2026-07-26): GWO phase from NH relative AAM + NH dM/dt
  (hemispheric GWO diverges; NH is where the torque->jet story lives).
  Composites over 20-90N. Torque conditioning on Himalaya + Rockies
  percentiles; Andes diagnostic only. Winter-half-year emphasis in masks.
- **Julia** for the compositor and per-cycle renderer (Makie), following
  scripts/julia conventions.
- **Reuse existing data**:
  - ~/era5_store (74 GB, u 14-lev + sp, 0.25 deg, 1991-2026, ~5-daily):
    state series (NH AAM, torque) for 1991-present with zero download.
  - WB2 1.5 deg daily (like build_u850_bandseries.py) for: outcome fields
    Z500 / T2m / precip / u200 / v200 (1959-present, ~10-15 GB), plus daily
    u/sp at 1.5 deg to give the GWO tendency a daily cadence pre-fill.
  - ARCO for the recent tail (WB2 ends ~2023).
  - nino_history.json (1970-), CPC ONI (1950-), BoM RMM (1974-).

## State keys (daily)

| key | source | bins |
|---|---|---|
| MJO | BoM RMM (own EOFs as check) | phase 1-8, amp <1 / >=1 |
| ENSO | ONI + RONI, Nino3 vs Nino4 | strength (strong/mod/weak/neutral, both signs), EP/CP flavor |
| GWO (NH) | NH AAM anom + centred dM/dt | phase 1-8 (M vs dM/dt plane), amp |
| Torque | Himalaya / Rockies mountain torque, 5-day mean | percentile terciles per region |

Fallback hierarchy when a joint bin is thin (<~40 samples):
MJO x ENSO x GWO -> MJO x ENSO -> MJO only; the used level is always labeled.

## Outcome library

Keyed NetCDF: (bin, lag 0-35 d, lat 20-90N, lon) for Z500 anom, T2m anom,
precip anom, WAF (fx, fy at 200 hPa, Takaya-Nakamura, same formulation as
scripts/mjo WAF product) + n-samples + bootstrap significance mask.

## Per-cycle renderer (run_local.sh, after the AAM/torque stage)

Inputs all exist per cycle: RMM point, Nino3.4/RONI, NH GWO phase from the
AAM archive, today's torque, today's WAF field. Store addition: z@500 as
**em type only** (a few MB/cycle — NOT the old 0.5 GB pf pull).
Outputs: composite panel webp + JSON (pattern correlation model-vs-composite,
WAF-en-route correlation, active torque-event flag) -> new page section.

## Analog ranker (phase 4)

Yearly state vectors (Nino3.4 trajectory since March, RONI, subsurface heat,
seasonal NH AAM) -> weighted top-k years -> composites from the same library.

## Phases

1. State series: NH AAM + torque from ~/era5_store (1991-), extended back and
   made daily via WB2 1.5 deg; RMM + ONI/RONI ingest.   <- current
2. Julia compositor -> keyed library.
3. Renderer + page + pipeline hook (+ em z500 in the ECMWF store registry).
4. Analog ranker.
