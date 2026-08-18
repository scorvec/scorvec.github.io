# Global Real-time & Archived Sounding Explorer

**[scorvec.com/skewt](https://scorvec.com/skewt/)** — click any radiosonde station on Earth and get a
full SHARPpy-grade sounding analysis: skew-T, hodograph, parcel theory, kinematics, severe-weather
composites, and where every value ranks against that station's own climatology.

Everything runs **client-side**. There is no backend, no API server, and no compute bill: the physics
is NWS SPC's [SHARPlib](https://github.com/keltonhalbert/SHARPlib) compiled to WebAssembly and executed
in your browser tab. The only server-side work is a set of scheduled jobs that mirror data and
precompute climatologies.

---

## 1. Data sources

A sounding can come from four places. The app walks them in order of *freshness*, and the status line
always names the one you're looking at, so you're never guessing.

| Source | Coverage | Latency | Resolution | CORS |
|---|---|---|---|---|
| **SPC observed** (`spc.noaa.gov/exper/soundings`) | US | Fastest — ~1 h post-synoptic | ~110–200 levels | ✅ |
| **IEM RAOB** (`mesonet.agron.iastate.edu`) | US + Canada | ~1 h behind SPC | ~180–250 levels | ✅ |
| **UW mirror** (`weather.uwyo.edu`, mirrored) | Global | 6×/day cron | Full BUFR, thinned to 260 | ❌ → mirrored |
| **NOAA IGRA v2** (`ncei.noaa.gov`) | Global, 1905–present | ~1–2 day lag | Mandatory + significant levels | ✅ |

### The physics floor on "real-time"

A radiosonde is *launched* at 00Z/12Z, but the balloon takes **~1.5–2 hours** to ascend and transmit.
No source on Earth has a complete 12Z sounding before ~13:30 UTC. "Real-time" for soundings means
*as soon as the atmosphere finishes reporting itself* — not at 12:00 sharp.

Within that floor, **SPC is the fastest public source**, which is why it's tried first for US stations
even though it carries slightly fewer levels than IEM. Once IEM catches up it offers a
higher-resolution profile of the same launch.

### Why the University of Wyoming is mirrored, not fetched

UW sends no `Access-Control-Allow-Origin` header, so a browser cannot fetch it directly. A GitHub
Action pulls ~600 stations 6×/day and force-pushes them to a `skewt-data` branch, which
`raw.githubusercontent.com` serves CORS-open. Complete UTC days are additionally bundled into
`uw-YYYYMMDD.zip` on an append-only `skewt-archive` branch — a permanent, growing high-resolution
archive (~16 MB/day) that the browser fetches and unzips client-side with `fflate`.

### Archive mode resolution order

`per-launch mirror (last 4 days)` → `UW day bundle` → `NOAA IGRA v2`

So recent dates get full BUFR fidelity, and anything older falls back to IGRA's mandatory and
significant levels — which reaches back to the 1930s–40s for many stations (Fort Worth's record
begins in **September 1937**).

---

## 2. The analysis engine

[SHARPlib](https://github.com/keltonhalbert/SHARPlib) (Kelton Halbert, NWS Storm Prediction Center
lineage) is compiled with Emscripten into `sharplib.wasm` (~250 KB). The same `compute_sounding()`
entry point is used by **both** the browser and the offline climatology builder, so a live value and
its historical percentile are computed by *identical code* — never two implementations that might
drift apart.

**Computed per sounding**

| Group | Values |
|---|---|
| Parcels | SB/ML/MU — CAPE, **ECAPE**, CINH, LCL, LI, LFC, EL |
| Kinematics | shear (0–1, 0–6 km, effective), SRH (0–1, 0–3 km, effective), effective inflow layer, Bunkers RM/LM (fixed **and** effective-layer), Corfidi up/downshear, critical angle, DTM |
| Composites | EHI, SCP, STP, SHIP, max updraft (ECAPE vs undilute), ECAPE/CAPE |
| Thermo & moisture | DCAPE, 0–3 km CAPE, NCAPE, PWAT (mm and inches), lapse rates, **column RH (CRH)**, mid-level RH, PBL top, K-index, Total Totals |
| Moist static energy | boundary-layer *h*, minimum *h\**, MSE deficit, column ∫h dp/g |
| Levels | 850/700/500 hPa T/Td, 500 hPa height, 1000–500 thickness, freezing level, wet-bulb zero, **tropopause** (WMO + cold point) |
| Winter | **dendritic growth zone** (with RH), Kuchera snow ratio, snow squall parameter, **precip type** (Bourgouin) |

---

## 3. ECAPE — entraining CAPE

**This is the headline parameter, and it's worth understanding why.**

### The problem with classic CAPE

Textbook CAPE lifts a parcel **undilute** — as if it were sealed in a bag, never mixing with the air
it rises through. Real updrafts are not sealed. They entrain drier environmental air across their
edges, which evaporates cloud water, cools the parcel, and destroys buoyancy. This is why a sounding
showing 3000 J/kg of CAPE routinely fails to produce a 77 m/s updraft: most of that buoyancy is
never realized.

Undilute CAPE therefore systematically **overstates** the energy actually available to a storm, and
it overstates it *unevenly* — the error depends on the wind profile and the storm's size, so two
soundings with identical CAPE can behave completely differently.

### What ECAPE does

**ECAPE** (Peters et al., 2023, *J. Atmos. Sci.* — "An analytic formula for entraining CAPE") derives
the entrainment rate from theory rather than a tuned fudge factor. The key insight is that an
updraft's dilution depends on its **width**, and its width is set by the **storm-relative wind and
shear** — so ECAPE is a function of the *thermodynamic* profile **and** the *kinematic* profile
together. Stronger storm-relative inflow and deeper shear support a wider, more protected updraft
core, which entrains proportionally less and preserves more buoyancy.

This is why SHARPlib's signature takes winds and a moist-static-energy profile, not just temperature
and dewpoint:

```cpp
float entrainment_cape(const float pressure[], const float height[],
                       const float temperature[], const float mse_arr[],
                       const float u_wind[], const float v_wind[],
                       const std::ptrdiff_t N, Parcel* pcl);
```

ECAPE is **always ≤ CAPE**, and the gap between them is physically meaningful — it is the buoyancy
the atmosphere promises but entrainment takes away.

### The flavors

ECAPE is defined **per lifted parcel**, exactly like CAPE, so there are three of them. The Parcels
table shows CAPE and ECAPE side by side for each:

| Parcel | Meaning |
|---|---|
| **SB-ECAPE** | Surface-based — a parcel lifted from the surface |
| **ML-ECAPE** | Mixed-layer — the lowest 100 hPa mixed, the most representative for most convection |
| **MU-ECAPE** | Most-unstable — the parcel with the greatest buoyancy, used for elevated convection |

### Derived quantities (Composites column)

| Quantity | Formula | What it tells you |
|---|---|---|
| **Max updraft (ECAPE)** | √(2·ECAPE) | The realistic entrainment-limited updraft speed |
| **Max updraft (CAPE)** | √(2·CAPE) | The classic undilute speed — shown for contrast |
| **Entrainment efficiency** | ECAPE / CAPE | Fraction of buoyancy surviving mixing |

Both updraft speeds are theoretical ceilings that ignore water loading and the perturbation pressure
gradient, so real updrafts fall short of even the ECAPE value. The point is not the absolute number
but the **ratio**: an entrainment efficiency of 55% means nearly half the advertised CAPE is fiction.

### Caveats

- ECAPE **requires wind data**. A sounding with a broken wind profile yields no meaningful ECAPE.
- ECAPE is **zero when CAPE is zero** — this is a real value, not a missing one (see §4).
- Corfidi MCS vectors (`mcs_motion_corfidi`) are computed and shown in the Kinematics table, but kept
  off the hodograph — extra arrows cost more legibility than they buy.

### Moist static energy — the quantity underneath it all

ECAPE is built on **moist static energy**, so the explorer reports it directly (the *Moist static
energy* table under the plots):

**h = cp·T + g·z + Lv·q**

MSE is *conserved* under both dry and moist adiabatic ascent — a parcel carries its h upward
unchanged, whether or not it's condensing. That makes it the natural currency for convection.

Two profiles matter:

- **h** — the actual moist static energy of the air at each level.
- **h\*** — the **saturation** MSE: what h *would* be if that level were saturated. It depends
  only on temperature and pressure, so it's a property of the environment, not its moisture.

A parcel lifted from the boundary layer conserves its h. It is **buoyant wherever its h exceeds the
environment's h\*** — the layer where BL h > h\* is precisely the layer that generates CAPE, and
the gap between them *is* the instability. (An h / h\* profile panel used to sit beside the
hodograph; it was retired in favour of larger skew-T and hodograph plots — `drawMSE()` in app.js
still draws it if a `<canvas id="mse">` is present.)

The scalars in the table:

| Value | Meaning |
|---|---|
| **MSE (0–500 m)** | Boundary-layer h — the energy a surface parcel carries aloft |
| **Min saturation MSE** | The lowest h\* aloft (usually mid-levels) — the "dry hole" a parcel must survive, with its height |
| **MSE deficit (h\*−h)** | Negative ⇒ the BL parcel out-energizes the mid-levels ⇒ conditionally unstable |
| **Column MSE** | The vertical integral (1/g)∫h dp, in GJ/m² — the column's total moist energy, the state variable of tropical MSE-budget theory |

A worked example (Fort Worth, 2026-07-11 12Z): BL h = 344.1 kJ/kg against a minimum h\* of
337.6 kJ/kg at 4.3 km — a deficit of **−6.5 kJ/kg**, i.e. the boundary layer carries more energy than
the mid-troposphere can hold at saturation. That gap is the sounding's CAPE, expressed in energy
terms.

---

## 4. Climatology — how the percentiles are built

Every station's **entire period of record** is downloaded from IGRA (up to ~90 MB per station, back to
the 1930s), every historical sounding is analyzed, and the results are reduced to **monthly percentile
breakpoints** (1/5/10/25/50/75/90/95/99) plus **record extremes with their year**. The result is one
small JSON per station on the `skewt-climo` branch, which the browser fetches on demand.

Indices ranked: PWAT, 850/700/500 hPa temperature, 500 hPa height, 1000–500 thickness, freezing level,
K-index, Total Totals, **ECAPE**, **SHIP**.

### Reading the colour shading

A value is marked **only when it is genuinely unusual**. Anything between the 10th and 90th percentile
is left plain — a "P53" badge says nothing, and tinting every row would drown the cases that matter.

| What you see | Meaning |
|---|---|
| plain text | 10th–90th percentile: unremarkable for this station on this day of the year |
| **red tint** + `P93` | above the 90th percentile |
| **blue tint** + `P4` | below the 10th percentile |
| **★ + year** | an outright record, with the year it was set |

The tint ramps *within the tail* (P91 is a whisper, P99 is loud), so intensity tracks how extreme the
value really is rather than merely its distance from the median. Hovering gives the exact percentile.
Nothing is flagged unless the index has **≥30 samples** in the window — a "record" from four soundings
is noise.

### The climatology panel

The **📊 Climatology** button charts the full annual cycle of any ranked variable:

| Element | Meaning |
|---|---|
| inner band | 25th–75th percentile (the middle half of all soundings) |
| outer band | 10th–90th percentile |
| pale line | median |
| red envelope | the **record high** for each day of the year |
| blue envelope | the **record low** |
| gold dot | the sounding on display, on its own day of the year |

The gap between the bands and the envelopes is itself the story: where they hug, the station is
well-behaved; where the envelope flares far above the 90th percentile, the station is capable of rare
and violent excursions.

### Record watch (the map)

A **red ring** around a live station means its latest sounding is beyond the 5th/95th percentile for
some index; the hover tooltip names which. (Gold rings mean *selected* — a different thing entirely.)

### Two methodological decisions that matter

**(a) A non-convective sounding has ECAPE of *zero*, not "missing."**
It is tempting to record ECAPE only when a parcel has positive CAPE. That is wrong: it makes the
climatology *conditional on convective days*, so the "10th percentile ECAPE" describes only the days
convection happened. Resistencia, Argentina had 51 such samples in January against 2,007 for PWAT —
the percentiles were meaningless. Valid soundings with no CAPE now correctly contribute zeros, which
is what makes a statement like "this is a 95th-percentile ECAPE day" mean anything.

**(b) Never fabricate missing data — gate on it instead.**
Radiosonde reporting is far from uniform. Much of South America reports temperature on ~99% of levels
but **dewpoint on only ~39%**, with as few as 12 levels per sounding. Filling those gaps with
plausible-looking guesses (a constant dewpoint depression, evenly spaced heights) silently produces
confident, wrong numbers — it manufactured absurd 0.5 mm precipitable-water values and destroyed CAPE
across an entire continent.

The builder now **refuses to invent data**. A sounding is used only if it has a real surface dewpoint,
reaches at least 400 hPa, and reports dewpoint on ≥60% of levels below 400 hPa. Genuine gaps *within*
a qualifying profile are interpolated in log-pressure (heights hypsometrically), never guessed from a
rule of thumb. Soundings that don't qualify are skipped rather than fudged.

The cost is a smaller sample; the benefit is that the sample is real. Each index stores its own `n`,
and the record watch ignores any index with fewer than 30 samples in that month.

---

## 5. Architecture

```
skewt/
  index.html      page shell, dark theme, map-first layout
  app.js          everything: sources, parsers, WASM bridge, skew-T + hodograph, climo UI
  sharplib.{js,wasm}   SHARPlib compiled by Emscripten
  stations.json   2,704 IGRA stations (pruned to those with real data files)
  iem_raob.json   WMO → ICAO map (290 US/Canada sites)

scripts/skewt/
  skewt_wasm.cpp    C++ wrapper — the single source of truth for the physics
  climo_cape.cpp    native build of the SAME wrapper, for the climatology
  build_climo.py    per-station climatology builder
  flag_anomalies.py record-watch: compares each latest sounding to its climatology
  mirror_soundings.py  UW mirror

branches (data, served CORS-open via raw.githubusercontent.com)
  skewt-data     latest soundings + manifest + anomalies.json   (force-pushed, 6×/day)
  skewt-archive  uw-YYYYMMDD.zip day bundles                     (append-only, grows forever)
  skewt-climo    climo/{gid}.json per-station climatology        (rebuilt on methodology change)
```

## 6. Known limitations

- **Not for operational use.** This is a personal research tool.
- Climatology is built for **active stations** (~915). Long-closed stations have no percentiles.
- CAPE-derived climatologies (ECAPE, SHIP) rest on fewer samples than level diagnostics, because they
  require complete, quality-gated soundings. Check `n` before leaning on a percentile.
- IGRA's historical vertical resolution is coarser than modern BUFR, so a 1960 hodograph is
  genuinely less detailed than today's — not a rendering artifact.
- SPC coverage is US-only; Canada falls back to IEM; the rest of the world to the UW mirror.

---

*Built by Claude (Anthropic), directed by Shawn Corvec. Physics by SHARPlib (K. Halbert, NWS SPC).
Data from NOAA/NWS SPC, Iowa State's IEM, the University of Wyoming, and NOAA NCEI IGRA v2.*
