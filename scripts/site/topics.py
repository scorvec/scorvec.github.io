#!/usr/bin/env python3
"""Topic explainers: one crawlable page per diagnostic under /topics/, so that searches for specialist terms such as
"E-P flux divergence" or "mountain torque" find the site.

Why these pages exist: the live products sit on stage pages (circulation.html, enso.html) whose explanations live in
<template> blocks and whose products are addressed by URL fragments (#epflux). Search engines index neither, so to
Google the site never mentions an E-P flux. Each topic page here carries the definition, the equations, how the live
product computes it and its current figure or loop, under its own URL, title and description.

The figures are the live products' own files (/assets/sst/*.webp, the sst_anim loops), so the pages stay current
without being rebuilt. Rebuild only when the text changes:

    python scripts/site/topics.py            # writes topics/*.html, stamps the site chrome, updates sitemap.xml
"""

from __future__ import annotations

import datetime as dt
import html
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "topics"
SITE = "https://scorvec.com"
PERSON = {"@id": f"{SITE}/#person"}


def loop(manifest: str, region: str, title: str, how: str = "Use the controls to step through the forecast.") -> dict:
    return {
        "kind": "loop",
        "src": f"/sst_anim.html?embed=1&manifest={manifest}&region={region}",
        "title": title,
        "how": how,
    }


def img(path: str, alt: str, cap: str = "") -> dict:
    return {"kind": "img", "src": path, "alt": alt, "cap": cap}


# Each topic: slug, the term (h1), alternate names people search for, title/description, the body (HTML with TeX in
# $...$ / $$...$$), the live figures (and `still`: a loop-only page's product as a still, for link previews), where the
# interactive product lives, related topics and references.
TOPICS = [
    dict(
        slug="ep-flux",
        term="Eliassen–Palm flux and E–P flux divergence",
        short="E–P flux",
        alt=[
            "E-P flux",
            "EP flux",
            "Eliassen-Palm flux",
            "E-P flux divergence",
            "EP flux divergence",
            "wave driving",
            "transformed Eulerian mean",
        ],
        title="E–P flux and its divergence, with a live forecast",
        desc="What the Eliassen–Palm (E–P) flux and its divergence measure, how to read an "
        "E–P cross-section, and a live 15-day GEFS forecast of the wave driving up to 1 hPa.",
        body=r"""
<p>The <b>Eliassen–Palm (E–P) flux</b> is a vector in the latitude–height plane that shows where atmospheric waves
carry their activity, and its <b>divergence</b> is the force those waves exert on the zonal-mean westerlies. It is the
standard diagnostic of wave–mean-flow interaction: of how baroclinic eddies maintain the jet stream, and of how
planetary waves rising out of the troposphere decelerate the stratospheric polar vortex before a sudden warming.</p>

<h2>The equations</h2>
<p>In the transformed Eulerian mean (TEM) form of the zonal-mean momentum equation, the eddies enter only through the
divergence of the E–P flux $\mathbf{F}$:</p>
<p>$$\frac{\partial \bar u}{\partial t} - f\,\bar v^{*} = \frac{\nabla\cdot\mathbf{F}}{\rho_0\,a\cos\phi} + \bar X$$</p>
<p>$$\nabla\cdot\mathbf{F} = \frac{1}{a\cos\phi}\frac{\partial}{\partial\phi}\!\left(F^{(\phi)}\cos\phi\right) + \frac{\partial F^{(z)}}{\partial z}$$</p>
<p>where $z=-H\ln(p/p_0)$ is log-pressure height with scale height $H$, $\rho_0=\rho_s e^{-z/H}$ the basic-state density,
$\bar v^{*}$ the residual meridional circulation and $\bar X$ any other force, such as gravity-wave drag.
In the quasi-geostrophic approximation the two components are</p>
<p>$$F^{(\phi)} = -\rho_0\,a\cos\phi\;\overline{u'v'}$$</p>
<p>$$F^{(z)} = \rho_0\,a\cos\phi\;f\,\frac{\overline{v'\theta'}}{\partial\bar\theta/\partial z}$$</p>
<p>The meridional component is the eddy momentum flux with its sign reversed; the vertical component is proportional
to the <a href="/topics/eddy-heat-flux.html">eddy heat flux</a>. Waves that tilt westward with height carry heat
poleward and therefore carry E–P flux upward.</p>

<h2>How to read it</h2>
<ul>
<li><b>The arrows</b> point where wave activity is going, along the waves' group velocity. Upward arrows at the
tropopause in winter are planetary waves entering the stratosphere; equatorward arrows in the upper troposphere are
baroclinic waves propagating toward their critical lines in the subtropics.</li>
<li><b>Convergence</b> ($\nabla\cdot\mathbf{F}<0$) is an easterly force: the waves decelerate the westerlies there.
Sustained convergence in the polar stratosphere is what weakens, and in a sudden warming reverses, the polar vortex.
The same wave drag, integrated, drives the Brewer–Dobson circulation (Haynes et al., 1991).</li>
<li><b>Divergence</b> ($\nabla\cdot\mathbf{F}>0$) is a westerly force.</li>
<li>Only the longest planetary waves (zonal wavenumbers 1–3) can propagate into the winter stratospheric westerlies
(Charney and Drazin, 1961), which is why stratospheric E–P diagnostics are usually restricted to them.</li>
</ul>

<h2>The live forecast</h2>
<p>The loop below is the E–P flux for zonal wavenumbers 1–3 and its divergence, computed from every member of the
NOAA GEFS ensemble (the control and 30 perturbed members, 31 pressure levels from 1000 up to 1 hPa) and stepped through
the 15-day forecast. The flux is quadratic in the wave amplitude, so it is computed for each member and then averaged;
the flux of the ensemble-mean fields would fade with lead time as the members' wave phases drift apart. The
quasi-geostrophic form is used with the zonal-mean static stability and vertical derivatives in ln p; levels below
the ground on more than 10% of a latitude circle are masked (grey); the 1–2 hPa layer,
where the derivative is one-sided, is hatched, and latitudes poleward of 82° are masked. The arrows use the Edmon et al.
(1980) display scaling for a log-pressure axis, multiplied by 1000 hPa/p so the stratosphere is legible: directions are
exact, but lengths grow with height, so compare them along a level. It is recomputed every 00 and 12 UTC cycle.</p>
""",
        figures=[
            loop(
                "epflux_gefs_manifest.json",
                "epflux_gefs",
                "E–P flux and wave driving, wavenumbers 1–3, GEFS 15-day forecast",
            )
        ],
        live=("/stratosphere.html#epflux", "Open the E–P flux product on the stratosphere page"),
        related=["eddy-heat-flux", "sudden-stratospheric-warming", "wave-activity-flux", "mountain-torque"],
        refs=[
            "Eliassen, A., and E. Palm, 1961: On the transfer of energy in stationary mountain waves. <i>Geofysiske Publikasjoner</i>, 22(3), 1–23.",
            "Charney, J. G., and P. G. Drazin, 1961: Propagation of planetary-scale disturbances from the lower into the upper atmosphere. <i>J. Geophys. Res.</i>, 66, 83–109.",
            "Andrews, D. G., and M. E. McIntyre, 1976: Planetary waves in horizontal and vertical shear: the generalized Eliassen–Palm relation and the mean zonal acceleration. <i>J. Atmos. Sci.</i>, 33, 2031–2048.",
            "Edmon, H. J., B. J. Hoskins, and M. E. McIntyre, 1980: Eliassen–Palm cross sections for the troposphere. <i>J. Atmos. Sci.</i>, 37, 2600–2616.",
            "Andrews, D. G., J. R. Holton, and C. B. Leovy, 1987: <i>Middle Atmosphere Dynamics</i>. Academic Press.",
            "Haynes, P. H., M. E. McIntyre, T. G. Shepherd, C. J. Marks, and K. P. Shine, 1991: On the “downward control” of extratropical diabatic circulations by eddy-induced mean zonal forces. <i>J. Atmos. Sci.</i>, 48, 651–678.",
        ],
    ),
    dict(
        slug="mountain-torque",
        term="Mountain torque and friction torque",
        short="Mountain torque",
        alt=[
            "mountain torque",
            "friction torque",
            "frictional torque",
            "surface torque",
            "torque budget",
            "atmospheric angular momentum budget",
            "gravity wave drag torque",
        ],
        title="Mountain and friction torque, with a live forecast",
        desc="How mountain and friction torques trade angular momentum between the Earth "
        "and the atmosphere, how to read torque maps, and live IFS-ENS forecasts by range.",
        body=r"""
<p>The atmosphere's angular momentum about the Earth's axis changes only through <b>torques</b> exerted at the
surface. Three act: <b>friction torque</b>, the drag of the surface on the wind; <b>mountain torque</b>, the pressure
difference across mountain ranges pushing on the topography; and the <b>gravity-wave drag</b> torque from terrain too
small for a model to resolve. Whatever angular momentum the atmosphere gains, the solid Earth loses, which is why
atmospheric angular momentum tracks the measured length of day.</p>

<h2>The equations</h2>
<p>$$\frac{dM}{dt} = T_F + T_M + T_{GW}$$</p>
<p>$$T_M = -a^2\!\iint p_s\,\frac{\partial h}{\partial\lambda}\,\cos\phi\;d\lambda\,d\phi
= a^2\!\iint h\,\frac{\partial p_s}{\partial\lambda}\,\cos\phi\;d\lambda\,d\phi$$</p>
<p>$$T_F = a^3\!\iint \tau_\lambda\cos^2\!\phi\;d\lambda\,d\phi$$</p>
<p>Here $M$ is the atmosphere's angular momentum, $p_s$ the surface pressure, $h$ the surface height, $a$ the Earth's
radius and $\tau_\lambda$ the eastward surface stress acting on the atmosphere. The two forms of $T_M$ are equal by
integration by parts around each latitude circle. Torques are quoted in Hadleys: 1 Hadley = 10<sup>18</sup> N m.</p>

<h2>How to read it</h2>
<ul>
<li><b>Mountain torque</b>: high pressure on the west side of a range and low pressure on its east side pushes the
mountain eastward, so the mountain pushes the atmosphere westward. The atmosphere loses westerly momentum; this is a
braking event. The reverse pattern (low to the west, high to the east) adds westerly momentum. Because the pressure
pattern moves with the weather, mountain torque changes within days, and the Rockies, the Tibetan Plateau and the
Andes carry most of it.</li>
<li><b>Friction torque</b> follows the surface winds: the easterly trade winds gain westerly momentum from the
surface, and the mid-latitude westerlies lose it. It changes more slowly, over weeks, as the wind belts shift.</li>
<li>Large mountain-torque events tend to come before changes in the jet downstream and in the global angular
momentum. On this site's own test (ERA5, Hersbach et al., 2020; December–February 1991–2020), a positive Himalayan torque
anomaly is followed 3 to 12 days later by a more extended North Pacific jet, most strongly after about six days (correlation
about 0.2, significant at a 10% false discovery rate).</li>
</ul>

<h2>The live forecast</h2>
<p>The friction and mountain torques are computed from the 50 members of ECMWF's physics-based IFS ensemble for days
1–15, each day's mean, as anomalies from the ERA5 1991–2020 climatology for the same time of year. Friction is the
model's own surface stress, which its open data carry, and its climatology is the same quantity in ERA5, ERA5's
own turbulent surface stress. Anomalies are used on purpose: the absolute budget cannot be
closed. ERA5's own terms sum to −11 ± 0.6 Hadleys in the annual mean (sampled four times a day, 1991–2020) where the answer must be
zero; the reanalysis adds angular momentum at every assimilation cycle, and the resolved
mountain torque changes with grid spacing. Both errors sit in the mean and cancel in an anomaly. The open forecast
data carry no gravity-wave stress, so that term stays in the residual between the net torque and the actual change in
angular momentum.</p>
<p>The forecast is IFS-ENS's rather than the AI ensemble's because its budget closes. A separate study tests whether a
forecast's own torques account for its change in angular momentum:
<a href="/aifs-aam-budget.html">does an AI weather model conserve angular momentum?</a> In winter 2025–26 (lead by lead, without correction for the 15 leads tested) the AIFS
ensemble's own torques explained 72% of its day-to-day change in angular momentum on days 1–10 and 11% on days 11–15;
the IFS ensemble's explained 89% at every lead.</p>
""",
        figures=[
            loop("torque_manifest.json", "torque", "Friction and mountain torque-density anomalies, IFS-ENS days 1–15"),
            img(
                "/assets/sst/torque_timeseries.webp",
                "Global and hemispheric friction and mountain torque anomalies against the change in atmospheric angular momentum, IFS-ENS forecast",
                "The torques integrated over the globe and each hemisphere, against the actual change in angular momentum. Shading: "
                "±1σ and ±2σ of the net anomaly in 1991–2020. The gap is the gravity-wave-drag torque, which the open data "
                "do not carry, and, hemisphere by hemisphere, transport across the equator.",
            ),
            img(
                "/assets/sst/torque_ranges.webp",
                "Mountain torque by mountain range: Rockies, Andes, Tibetan Plateau and others, IFS-ENS forecast",
                "Which mountain ranges are doing the work.",
            ),
        ],
        live=("/circulation.html#torque", "Open the surface torque product in the circulation viewer"),
        related=["atmospheric-angular-momentum", "wave-activity-flux", "ep-flux"],
        refs=[
            "Weickmann, K. M., and P. D. Sardeshmukh, 1994: The atmospheric angular momentum cycle associated with a Madden–Julian oscillation. <i>J. Atmos. Sci.</i>, 51, 3194–3208.",
            "Lott, F., A. W. Robertson, and M. Ghil, 2004: Mountain torques and Northern Hemisphere low-frequency variability. Part I: Hemispheric aspects. <i>J. Atmos. Sci.</i>, 61, 1259–1271.",
            "Egger, J., K. Weickmann, and K.-P. Hoinka, 2007: Angular momentum in the global atmospheric circulation. <i>Rev. Geophys.</i>, 45, RG4007.",
            "Hersbach, H., and co-authors, 2020: The ERA5 global reanalysis. <i>Q. J. R. Meteorol. Soc.</i>, 146, 1999–2049.",
        ],
    ),
    dict(
        slug="atmospheric-angular-momentum",
        term="Atmospheric angular momentum (AAM)",
        short="Atmospheric angular momentum",
        alt=[
            "AAM",
            "relative AAM",
            "global atmospheric angular momentum",
            "Global Wind Oscillation",
            "GWO",
            "length of day",
            "AAM tendency",
        ],
        title="Atmospheric angular momentum (AAM), live forecast",
        desc="What atmospheric angular momentum measures, why it tracks El Niño and the "
        "length of day, the GWO phase space, and a live IFS-ENS forecast by hemisphere.",
        body=r"""
<p><b>Atmospheric angular momentum (AAM)</b> is the atmosphere's total spin about the Earth's axis. Its <i>relative</i>
part, the one carried by the winds, is</p>
<p>$$M_r = \frac{a^3}{g}\iiint u\,\cos^2\!\phi\;d\lambda\,d\phi\,dp$$</p>
<p>the mass-weighted westerly wind, weighted again by the distance from the axis. The <i>mass</i> term,
$\Omega a^4 g^{-1}\iint p_s\cos^3\!\phi\,d\lambda\,d\phi$, is the angular momentum the atmosphere carries by
rotating with the Earth, and shifts only as air mass moves between latitudes.</p>

<h2>What moves it</h2>
<ul>
<li><b>The subtropical jets dominate the total.</b> The $\cos^2\phi$ weight is largest at the equator, but the deep
tropics hold weak or easterly winds, so they add little to the total. The variability is carried mostly by zonal-wind
changes aloft in the tropics and subtropics, where the weight is large; the shallow trade-wind layer holds little mass,
so the trades act through the friction torque rather than as a store of momentum.</li>
<li><b>El Niño raises AAM.</b> It strengthens the subtropical jets and weakens the trades, so the atmosphere holds
more westerly momentum. Conservation then requires the solid Earth to slow down: AAM anomalies appear as changes of a
fraction of a millisecond in the length of day, and about 6×10<sup>25</sup> kg m<sup>2</sup> s<sup>−1</sup> of AAM
corresponds to one millisecond.</li>
<li><b>It can change only through surface torques</b>, the <a href="/topics/mountain-torque.html">mountain and
friction torques</a>. Mountain torque works within days; friction over weeks.</li>
<li><b>The Global Wind Oscillation</b> (Weickmann and Berry, 2009) follows the global AAM anomaly against its
tendency in a two-dimensional phase space, much as the MJO index follows tropical convection. Orbits through it pass
through a recurring sequence of jet and torque patterns, although, unlike the MJO, the GWO behaves as red noise rather than a
regular oscillation.</li>
</ul>

<h2>The live forecast</h2>
<p>Relative AAM is integrated for the globe and each hemisphere from the winds of the 50 members of ECMWF's IFS
ensemble on 14 pressure levels from 10 to 1000 hPa, which is effectively the whole column. It is shown on the ERA5 (Hersbach et al., 2020) 1991–2020 annual cycle
together with the 15-day forecast. GFZ's ESMGFZ series (Dobslaw and Dill, 2018), computed from ECMWF analyses for the Earth-rotation community,
is drawn alongside as an independent check. The phase plot follows each hemisphere through AAM anomaly against
tendency: the hemispheric version of the Global Wind Oscillation.</p>
<p>The forecast is the physics-based IFS ensemble's because its angular momentum changes for the right reasons, which
the AI ensemble's does not after about day 10:
<a href="/aifs-aam-budget.html">does an AI weather model conserve angular momentum?</a></p>
""",
        figures=[
            img(
                "/assets/sst/aam.webp",
                "Global and hemispheric relative atmospheric angular momentum: observed and IFS-ENS 15-day forecast, absolute and anomaly",
                "Relative AAM for the globe and each hemisphere on its 1991–2020 annual cycle, with the 15-day forecast. The "
                "observed line is ERA5 at 12 UTC every third day; it ends 5–14 days before the forecast starts because "
                "ERA5 is published late.",
            ),
            img(
                "/assets/sst/aam_phase.webp",
                "Atmospheric angular momentum anomaly against its tendency: hemispheric Global Wind Oscillation phase space with the IFS-ENS forecast",
                "Each hemisphere's path through AAM anomaly and tendency, the last 75 days and the forecast. The headline "
                "(days 12–15 against days 0–2) includes the model's own drift, which is not removed.",
            ),
        ],
        live=("/circulation.html#aam", "Open the AAM product in the circulation viewer"),
        related=["mountain-torque", "hadley-cell", "wave-activity-flux"],
        refs=[
            "Rosen, R. D., and D. A. Salstein, 1983: Variations in atmospheric angular momentum on global and regional scales and the length of day. <i>J. Geophys. Res.</i>, 88, 5451–5470.",
            "Weickmann, K. M., and P. D. Sardeshmukh, 1994: The atmospheric angular momentum cycle associated with a Madden–Julian oscillation. <i>J. Atmos. Sci.</i>, 51, 3194–3208.",
            "Egger, J., K. Weickmann, and K.-P. Hoinka, 2007: Angular momentum in the global atmospheric circulation. <i>Rev. Geophys.</i>, 45, RG4007.",
            "Weickmann, K., and E. Berry, 2009: The tropical Madden–Julian oscillation and the global wind oscillation. <i>Mon. Wea. Rev.</i>, 137, 1601–1614.",
            "Dobslaw, H., and R. Dill, 2018: Predicting Earth orientation changes from global forecasts of atmosphere–hydrosphere dynamics. <i>Adv. Space Res.</i>, 61, 1047–1054.",
            "Hersbach, H., and co-authors, 2020: The ERA5 global reanalysis. <i>Q. J. R. Meteorol. Soc.</i>, 146, 1999–2049.",
        ],
    ),
    dict(
        slug="wave-activity-flux",
        term="Wave activity flux (Takaya–Nakamura)",
        short="Wave activity flux",
        alt=[
            "Takaya-Nakamura flux",
            "Takaya-Nakamura wave activity flux",
            "TN01",
            "Plumb flux",
            "Rossby wave packet",
            "Rossby wave propagation",
            "WAF",
        ],
        title="Takaya–Nakamura wave activity flux, live forecast",
        desc="What the Takaya–Nakamura wave activity flux shows about Rossby wave packets, "
        "downstream development and blocking, and a live AIFS-ENS forecast at 250 hPa.",
        body=r"""
<p>The <b>wave activity flux</b> of Takaya and Nakamura (2001) is a vector that points along the group velocity of
quasi-stationary Rossby waves: where wave energy is travelling, rather than where the ridges and troughs are. Its
divergence marks a wave source, such as tropical convection or flow over a mountain range, and its convergence marks
where the waves are depositing activity and the downstream flow will amplify days later. That makes it a practical tool
for downstream development, blocking onset and the teleconnections that radiate from the tropics, such as the El
Niño–PNA pathway.</p>

<h2>The equation</h2>
<p>For a basic state with horizontal wind $\mathbf{U}=(U,V)$ and a streamfunction anomaly $\psi'$, the horizontal
flux on a pressure surface is</p>
<p>$$\mathbf{W}=\frac{p\cos\phi}{2\,|\mathbf{U}|}\begin{pmatrix}U(\psi_x'^2-\psi'\psi_{xx}')+V(\psi_x'\psi_y'-\psi'\psi_{xy}')\\[2pt]U(\psi_x'\psi_y'-\psi'\psi_{xy}')+V(\psi_y'^2-\psi'\psi_{yy}')\end{pmatrix}$$</p>
<p>Here $p$ is pressure divided by 1000 hPa (0.25 at 250 hPa). It generalizes Plumb's (1985) flux for stationary waves on a zonal flow to a zonally varying basic state and to
slowly migrating waves. It is <i>phase-independent</i>: it does not oscillate between ridge and trough, so packets
can be followed without averaging over a wavelength.</p>

<h2>How to read it</h2>
<ul>
<li>Arrows arcing along the jet are a wave train; follow them to see where a disturbance will be felt next.</li>
<li>Where the arrows converge and slow down, the wave amplifies: that is where ridges build and blocks form.</li>
<li>A packet arcing from the subtropical Pacific toward North America is often tropical heating reaching the
extratropics; the live map starts at 20° latitude.</li>
</ul>

<h2>The live forecast</h2>
<p>The flux is computed at 250 hPa, the level of the jet core and the stationary-wave maximum, for each of 25 AIFS
ensemble members at every day to day 15, and then averaged. Because the flux is quadratic in $\psi'$, the flux of the
ensemble mean would fade with lead time as the members decorrelate, while the mean of the members' fluxes keeps the
packets that are actually forecast. Each member's $\psi'$ is smoothed with a 5-day running mean along the forecast,
because the theory is for quasi-stationary waves. The basic state is the ERA5 (Hersbach et al., 2020) 1991–2020 climatology plus the 30-day
mean anomaly of the AIFS analyses. The flux is masked equatorward of 20° and where the basic-state wind is below
3 m/s, since the theory needs a westerly waveguide, and shading is hatched where fewer than 60 % of members agree on the
sign of the divergence.</p>
""",
        still="/assets/sst/waf.webp",
        figures=[
            loop(
                "waf_manifest.json",
                "waf",
                "Takaya–Nakamura wave activity flux at 250 hPa, AIFS-ENS member mean, days 0–15",
            ),
        ],
        live=("/circulation.html#waf", "Open the wave activity flux in the circulation viewer"),
        related=["ep-flux", "mountain-torque", "dynamic-tropopause"],
        refs=[
            "Hoskins, B. J., and D. J. Karoly, 1981: The steady linear response of a spherical atmosphere to thermal and orographic forcing. <i>J. Atmos. Sci.</i>, 38, 1179–1196.",
            "Plumb, R. A., 1985: On the three-dimensional propagation of stationary waves. <i>J. Atmos. Sci.</i>, 42, 217–229.",
            "Takaya, K., and H. Nakamura, 2001: A formulation of a phase-independent wave-activity flux for stationary and migratory quasigeostrophic eddies on a zonally varying basic flow. <i>J. Atmos. Sci.</i>, 58, 608–627.",
            "Hersbach, H., and co-authors, 2020: The ERA5 global reanalysis. <i>Q. J. R. Meteorol. Soc.</i>, 146, 1999–2049.",
        ],
    ),
    dict(
        slug="eddy-heat-flux",
        term="100 hPa eddy heat flux",
        short="Eddy heat flux",
        alt=[
            "eddy heat flux",
            "meridional eddy heat flux",
            "v'T'",
            "vT 100 hPa",
            "upward wave activity",
            "vertical E-P flux",
            "wave-1 heat flux",
        ],
        title="100 hPa eddy heat flux, with a live forecast",
        desc="Why the 100 hPa poleward eddy heat flux measures planetary waves entering the "
        "stratosphere, how its 40-day mean leads the vortex, and a live AIFS-ENS forecast.",
        body=r"""
<p>The zonal-mean poleward <b>eddy heat flux</b> $[v'T']$ at 100 hPa is the most direct measure of how much
planetary-wave activity is leaving the troposphere for the stratosphere. It is proportional to the vertical component
of the <a href="/topics/ep-flux.html">Eliassen–Palm flux</a>, $F^{(z)}\propto f\,\overline{v'\theta'}/\bar\theta_z$,
so a large positive value means waves are propagating upward into the polar vortex.</p>

<h2>Why amplitude is not enough</h2>
<p>A large planetary wave is not necessarily moving upward. A wave that is vertically stacked carries almost no heat
flux; one whose ridges and troughs tilt westward with height carries a lot. Maps of wave amplitude show where the wave
sits, and the heat flux shows whether it is going up.</p>

<h2>How it leads the vortex</h2>
<ul>
<li>The vortex responds to the accumulated flux, not to single days. Polvani and Waugh (2004) showed that extremes of
the 40-day mean 100 hPa heat flux precede extreme stratospheric events: weeks above normal weaken the vortex, and
weeks below normal let it strengthen.</li>
<li>Averaged from mid-January to the end of February, the 100 hPa heat flux at 45–75°N predicts the Arctic lower
stratosphere's temperature in early March (Newman et al., 2001), and with it how much ozone survives into spring.</li>
<li>In early autumn radiative cooling still dominates, so a burst of heat flux slows the vortex's seasonal spin-up
rather than reversing it.</li>
</ul>

<h2>The live forecast</h2>
<p>The flux is averaged over 45–75° in each hemisphere (positive is poleward in both), computed for the AIFS control
and 25 perturbed members and then averaged: like every quadratic diagnostic, the flux of the ensemble-mean fields
would fade with lead time. Zonal wavenumbers 1–72 are kept, and the climatology is the same flux from ERA5
(Hersbach et al., 2020), 1991–2020, taken like the forecasts from single fields. ERA5 arrives about five days late, so the
analysed part of each panel is the AIFS control's initial state, ECMWF's operational analysis. Each model's own mean error
against ERA5, measured on a year or more of its archived runs by season and lead, is removed where it is statistically
significant: AIFS-ENS, for one, runs about a quarter low in the northern winter by its second week (95 % interval
8–34 %). The panels show the daily flux, its wave-1 and wave-2 parts, the trailing 40-day mean standardized, the
flux by latitude through the forecast, and an SSW warning meter: every member's trailing 10-day mean against
27.5 K m/s, the level that best separates the 30 days before the 21 sudden warmings on record since
1998 from quiet winter windows (20 of them reached it, a median 13 days
ahead; on such a day the chance of a warming within 30 days is 35 % against 12 % on
any winter day). The stratosphere page also carries an IFS-ENS version and GEPS and GEFS versions to day 35.</p>
""",
        figures=[
            img(
                "/assets/sst/heatflux100_nh.webp",
                "100 hPa zonal-mean eddy heat flux 45–75°N: AIFS-ENS members against the 1991–2020 climatology, wave-1 and wave-2 parts, 40-day mean",
                "Northern Hemisphere, 45–75°N. The Southern Hemisphere version is on the stratosphere page.",
            ),
        ],
        live=("/stratosphere.html#heatflux", "Open the eddy heat flux on the stratosphere page"),
        related=["ep-flux", "sudden-stratospheric-warming"],
        refs=[
            "Newman, P. A., E. R. Nash, and J. E. Rosenfield, 2001: What controls the temperature of the Arctic stratosphere during the spring? <i>J. Geophys. Res.</i>, 106, 19999–20010.",
            "Polvani, L. M., and D. W. Waugh, 2004: Upward wave activity flux as a precursor to extreme stratospheric events and subsequent anomalous surface weather regimes. <i>J. Climate</i>, 17, 3548–3554.",
            "Andrews, D. G., J. R. Holton, and C. B. Leovy, 1987: <i>Middle Atmosphere Dynamics</i>. Academic Press.",
            "Hersbach, H., and co-authors, 2020: The ERA5 global reanalysis. <i>Q. J. R. Meteorol. Soc.</i>, 146, 1999–2049.",
        ],
    ),
    dict(
        slug="sudden-stratospheric-warming",
        term="Sudden stratospheric warmings and the polar vortex",
        short="Sudden stratospheric warmings",
        alt=[
            "SSW",
            "sudden stratospheric warming",
            "major SSW",
            "polar vortex",
            "polar vortex split",
            "vortex displacement",
            "Charlton-Polvani",
            "stratosphere-troposphere coupling",
            "dripping paint",
        ],
        title="Sudden stratospheric warming: every SSW since 1980",
        desc="What a sudden stratospheric warming is, splits and displacements, how SSWs "
        "reach the surface, every major SSW since 1980, and a live polar vortex forecast.",
        body=r"""
<p>A <b>sudden stratospheric warming (SSW)</b> is the collapse of the winter polar vortex: planetary waves rising from
the troposphere decelerate the stratospheric westerlies until they reverse, and the polar stratosphere warms by tens of
degrees within days. The usual definition (Charlton and Polvani, 2007) is the first day in November–March on which the
zonal-mean zonal wind at 60°N and 10 hPa turns easterly. A second event in the same winter must be separated by 20
consecutive westerly days, and a reversal after which the wind does not return westerly for 10 consecutive days before
30 April is the final warming, not an SSW.</p>

<h2>Splits and displacements</h2>
<p>Seen from above, the vortex either shifts off the pole (a <b>displacement</b>, driven mostly by zonal wavenumber 1)
or breaks into two separate lows (a <b>split</b>, driven by wavenumber 2). Splits tend to be more abrupt, and some studies
find them more strongly coupled to the surface (Mitchell et al., 2013), although Charlton and Polvani (2007) found the
tropospheric response largely insensitive to the type.</p>

<h2>How they reach the surface</h2>
<p>Baldwin and Dunkerton (2001) showed that large stratospheric anomalies work their way down over a few weeks and then
hold the surface in the same state for one to two months. Drawn against time and height the composite looks like paint
dripping down a wall. After a warming that reaches the lower stratosphere, the Arctic Oscillation and North Atlantic
Oscillation tend to turn negative, bringing cold air into northern Eurasia and eastern North America.</p>

<h2>Every warming since 1980</h2>
<p>This site's catalogue, built from NASA's MERRA-2 reanalysis (Gelaro et al., 2017), finds 28 major warmings in 24 of the 46 winters since
1980, 10 of them splits. The events are split into deep and shallow by how strong the anomaly is at 100–150 hPa over the
following month. After the 14 deep events the Arctic Oscillation averaged 1.42 below the same dates in other years over
days 1–30, and 0.69 below over days 31–60. Both differences are significant (t-test, false-discovery rate 10 %; Benjamini and Hochberg, 1995; Wilks, 2016).
After the 14 shallow events there was no significant change.</p>

<h2>The live forecast</h2>
<p>The vortex chart tracks the zonal-mean wind at 60°N at 10 and 100 hPa and the polar-cap height at 100 hPa against
the MERRA-2 percentile bands in three ensembles, ECMWF's AIFS-ENS and IFS-ENS and ECCC's GEPS (each as its members'
10–90 % range and mean), with the deterministic GDPS and NASA GEOS FP runs dashed. The fraction of members reversing
the wind (a forecast major warming) is called out.</p>
""",
        figures=[
            img(
                "/assets/sst/nh_vortex.webp",
                "Northern polar vortex forecast: zonal-mean wind at 60°N at 10 and 100 hPa and 100 hPa polar-cap height, AIFS-ENS, IFS-ENS and GEPS ranges against MERRA-2 percentiles",
                "The live vortex forecast: three ensembles and two deterministic runs against the MERRA-2 1980–2026 range.",
            ),
            img(
                "/assets/sst/strat_hist_timeline.webp",
                "Timeline of every major sudden stratospheric warming and strong-vortex event since 1980 from MERRA-2, with ENSO and the QBO",
                "Every major warming and strong-vortex event since 1980, winter by winter.",
            ),
            img(
                "/assets/sst/strat_hist_drip_ssw_deep.webp",
                "Dripping-paint composite of polar-cap height from the surface to 1 hPa around deep sudden stratospheric warmings, with the Arctic Oscillation",
                "The dripping-paint composite for the deep events: colour only where significant.",
            ),
        ],
        live=(
            "/stratosphere.html#vortex",
            "Open the polar vortex forecast and the SSW history on the stratosphere page",
        ),
        related=["eddy-heat-flux", "ep-flux", "wave-activity-flux"],
        refs=[
            "Baldwin, M. P., and T. J. Dunkerton, 2001: Stratospheric harbingers of anomalous weather regimes. <i>Science</i>, 294, 581–584.",
            "Charlton, A. J., and L. M. Polvani, 2007: A new look at stratospheric sudden warmings. Part I: Climatology and modeling benchmarks. <i>J. Climate</i>, 20, 449–469.",
            "Butler, A. H., J. P. Sjoberg, D. J. Seidel, and K. H. Rosenlof, 2017: A sudden stratospheric warming compendium. <i>Earth Syst. Sci. Data</i>, 9, 63–76.",
            "Baldwin, M. P., et al., 2021: Sudden stratospheric warmings. <i>Rev. Geophys.</i>, 59, e2020RG000708.",
            "Mitchell, D. M., L. J. Gray, J. Anstey, M. P. Baldwin, and A. J. Charlton-Perez, 2013: The influence of stratospheric vortex displacements and splits on surface climate. <i>J. Climate</i>, 26, 2668–2682.",
            "Gelaro, R., and co-authors, 2017: The Modern-Era Retrospective Analysis for Research and Applications, Version 2 (MERRA-2). <i>J. Climate</i>, 30, 5419–5454.",
            "Benjamini, Y., and Y. Hochberg, 1995: Controlling the false discovery rate: a practical and powerful approach to multiple testing. <i>J. R. Stat. Soc. B</i>, 57, 289–300.",
            "Wilks, D. S., 2016: “The stippling shows statistically significant grid points”: how research results are routinely overstated and overinterpreted, and what to do about it. <i>Bull. Amer. Meteor. Soc.</i>, 97, 2263–2273.",
        ],
    ),
    dict(
        slug="hadley-cell",
        term="Hadley cell and the mean meridional mass streamfunction",
        short="Hadley cell",
        alt=[
            "Hadley circulation",
            "mean meridional streamfunction",
            "meridional mass streamfunction",
            "MMSF",
            "Ferrel cell",
            "Hadley cell strength",
            "Hadley cell edge",
        ],
        title="Hadley cell and mass streamfunction, live analysis",
        desc="How the mean meridional mass streamfunction describes the Hadley, Ferrel and polar cells, how their "
        "edges are measured, and a live ECMWF analysis.",
        body=r"""
<p>The <b>Hadley cell</b> is the tropical overturning circulation: air rises near the equator, flows poleward aloft,
sinks in the subtropics and returns toward the equator in the trade winds. It and the weaker Ferrel and polar cells
are described by the zonal-mean <b>meridional mass streamfunction</b></p>
<p>$$\Psi(\phi,p)=\frac{2\pi a\cos\phi}{g}\int_0^{p}[v]\,dp'$$</p>
<p>where $[v]$ is the zonal-mean meridional wind. The mass flowing northward between two pressure levels is the
difference of $\Psi$ between them, and the streamlines are its contours. With this convention $\Psi>0$ is a clockwise
cell when latitude runs north to the right and pressure increases downward, which is the Northern Hemisphere Hadley
cell.</p>

<h2>Strength and edges</h2>
<ul>
<li><b>Strength</b> is usually the maximum of $|\Psi|$ near 500 hPa, of order 10<sup>11</sup> kg s<sup>−1</sup> in
the winter hemisphere's cell.</li>
<li><b>The edge</b> is the latitude where $\Psi$ at 500 hPa crosses zero on the poleward side of the cell. It marks
the subtropical dry zones, and its poleward drift in a warming climate is one of the most studied circulation trends (Seidel et al., 2008).</li>
<li>The cells follow the sun (Dima and Wallace, 2003): the winter hemisphere's cell dominates and extends across the equator, while the summer
cell nearly vanishes.</li>
<li>El Niño strengthens the Hadley circulation (Oort and Yienger, 1996) and pulls its edges equatorward (Lu et al., 2008;
Nguyen et al., 2013).</li>
</ul>

<h2>The live analysis</h2>
<p>$\Psi$ is computed from ECMWF's operational analysis, as carried at step 0 of the AIFS-ENS control in the open
data, and shown as the anomaly from the ERA5 (Hersbach et al., 2020) 1991–2020 climatology for the
same day of year, with the absolute cells contoured and vertical motion drawn as arrows. The analysis is first corrected
by its mean offset from ERA5 at the same season (in autumn it reads the northern cell about 10% stronger than ERA5
does), so the anomaly is against the reference it is measured on. Each frame is a 7-day mean of the 00 UTC analyses, one a day since July 2025
and kept for good, so the evolution of the cells is legible through the day-to-day noise; the circulation viewer also
carries the ensemble forecasts of the same quantity to day 15.</p>
""",
        still="/assets/sst/mmsf_anom.webp",
        figures=[
            loop(
                "mmsf_hist_manifest.json",
                "mmsf_hist",
                "Meridional mass streamfunction anomaly, 7-day mean, ECMWF analysis against ERA5 1991–2020",
                "Use the controls to step through the 7-day means since July 2025, one a day.",
            ),
        ],
        live=("/circulation.html#hadley", "Open the Hadley cell product in the circulation viewer"),
        related=["atmospheric-angular-momentum", "ep-flux"],
        refs=[
            "Held, I. M., and A. Y. Hou, 1980: Nonlinear axially symmetric circulations in a nearly inviscid atmosphere. <i>J. Atmos. Sci.</i>, 37, 515–533.",
            "Oort, A. H., and J. J. Yienger, 1996: Observed interannual variability in the Hadley circulation and its connection to ENSO. <i>J. Climate</i>, 9, 2751–2767.",
            "Dima, I. M., and J. M. Wallace, 2003: On the seasonality of the Hadley cell. <i>J. Atmos. Sci.</i>, 60, 1522–1527.",
            "Lu, J., G. Chen, and D. M. W. Frierson, 2008: Response of the zonal mean atmospheric circulation to El Niño versus global warming. <i>J. Climate</i>, 21, 5835–5851.",
            "Nguyen, H., A. Evans, C. Lucas, I. Smith, and B. Timbal, 2013: The Hadley circulation in reanalyses: climatology, variability, and change. <i>J. Climate</i>, 26, 3357–3376.",
            "Seidel, D. J., Q. Fu, W. J. Randel, and T. J. Reichler, 2008: Widening of the tropical belt in a changing climate. <i>Nature Geosci.</i>, 1, 21–24.",
            "Hersbach, H., and co-authors, 2020: The ERA5 global reanalysis. <i>Q. J. R. Meteorol. Soc.</i>, 146, 1999–2049.",
        ],
    ),
    dict(
        slug="dynamic-tropopause",
        term="Dynamic tropopause and potential vorticity (2 PVU)",
        short="Dynamic tropopause",
        alt=[
            "dynamic tropopause map",
            "DT map",
            "theta on 2 PVU",
            "potential temperature on the dynamic tropopause",
            "potential vorticity",
            "PV",
            "isentropic potential vorticity",
            "PV thinking",
            "Rossby wave breaking",
        ],
        title="Dynamic tropopause (2 PVU) maps, live forecast",
        desc="What the dynamic tropopause is, how to read θ on the 2-PVU surface and "
        "potential vorticity on 330 and 350 K, and live IFS control forecast maps to day 10.",
        body=r"""
<p>The <b>dynamic tropopause</b> is the surface of constant <b>potential vorticity (PV)</b> that separates
stratospheric air (high PV) from tropospheric air (low PV), conventionally the 2-PVU surface, where
1 PVU = 10<sup>−6</sup> K m<sup>2</sup> kg<sup>−1</sup> s<sup>−1</sup>. In isentropic coordinates Ertel's PV is</p>
<p>$$P = -g\,(\zeta_\theta + f)\,\frac{\partial\theta}{\partial p}$$</p>
<p>It is conserved following the flow in the absence of heating and friction and, given a balance condition, the
whole balanced flow can be recovered from it. A map of the dynamic tropopause therefore shows the upper-level
circulation in a single field.</p>

<h2>How to read a dynamic tropopause map</h2>
<ul>
<li><b>Potential temperature on 2 PVU</b> is the usual variable. Low values (cold colours) are the stratosphere
reaching down: troughs and cut-off lows. High values (warm colours) are tropical air lifting the tropopause: ridges.</li>
<li>A tight gradient of θ on the tropopause is the jet stream.</li>
<li>A tongue of low θ that thins and wraps up is a Rossby wave breaking; cyclonic and anticyclonic breaking lead to
different downstream weather.</li>
<li>PV on the 330 K and 350 K isentropes shows the same structure on surfaces near the winter jet and near the summer
and subtropical jet.</li>
</ul>

<h2>The live forecast</h2>
<p>Ertel PV is computed in pressure coordinates, keeping the tilting terms that the isentropic form above folds into
$\zeta_\theta$, from the temperature and wind of the IFS control (ECMWF's unperturbed 9 km run, called HRES until
May 2026) on nine levels from 700 to 100 hPa, every 6 hours to day 10. A single run is used on purpose: potential vorticity does not survive ensemble averaging,
because the mean of fifty differently placed folds is a smear. The 2-PVU surface is found by searching upward from
700 hPa, and isentropic PV by interpolation in θ. Where PV is still below 2 PVU at 100 hPa (the deep tropics) the
maps show the 100 hPa values. Besides the hemispheric map there are five regional views; the South America one is
on the −2-PVU surface, because PV is negative in the Southern Hemisphere.</p>
""",
        figures=[
            loop(
                "dt_manifest.json",
                "dt",
                "Potential temperature on the 2-PVU dynamic tropopause, IFS control, 6-hourly to day 10",
            ),
        ],
        live=("/circulation.html#dt", "Open the dynamic tropopause maps in the circulation viewer"),
        related=["wave-activity-flux", "ep-flux"],
        refs=[
            "Hoskins, B. J., M. E. McIntyre, and A. W. Robertson, 1985: On the use and significance of isentropic potential vorticity maps. <i>Q. J. R. Meteorol. Soc.</i>, 111, 877–946.",
            "Morgan, M. C., and J. W. Nielsen-Gammon, 1998: Using tropopause maps to diagnose midlatitude weather systems. <i>Mon. Wea. Rev.</i>, 126, 2555–2579.",
        ],
    ),
]
BY_SLUG = {t["slug"]: t for t in TOPICS}

STYLE = """
<style>
  body { background: var(--c-bg, #f5f6f3); color: var(--c-ink, #29321f); margin: 0; }
  .topic { font-size: clamp(1rem, 0.92rem + 0.22vw, 1.14rem); max-width: clamp(46rem, 32rem + 22vw, 60rem); margin: 0 auto; padding: 2rem clamp(1rem, 4vw, 3rem) 3.5rem; }   /* the column widens with the window: ~810 px up to a 1120 px window, ~880 at 1440 px, ~985 at 1920, 60rem max; 18-20 px text, ~90-100 characters a line */
  .ai-note { font-size: 0.9em; line-height: 1.5; margin: 0 0 1.3rem; padding: 0.65rem 0.95rem; background: var(--c-panel, #eef1ea); border-left: 3px solid var(--c-accent, #274b7a); border-radius: 0 4px 4px 0; }
  .topic figcaption, .figs figcaption { font-size: clamp(0.88rem, 0.84rem + 0.15vw, 0.95rem); }   /* a step below the text (site.css pins rem) */
  .topic .crumbs { font-size: 0.92em; color: var(--c-muted, #5c6b73); margin-bottom: 1.2rem; }
  .topic .crumbs a { color: inherit; }
  .topic h1 { text-align: left; margin: 0 0 0.4rem; }
  .topic .aka { color: var(--c-muted, #5c6b73); font-size: 0.95em; margin: 0 0 1.6rem; }
  .topic h2 { margin: 2.2rem 0 0.7rem; text-align: left; }
  .topic p, .topic li { line-height: 1.6; }
  .topic ul { padding-left: 1.2rem; }
  .topic li { margin: 0.45rem 0; }
  .topic a { color: var(--c-accent, #274b7a); text-underline-offset: 0.2em; }
  .topic .katex-display { overflow-x: auto; overflow-y: hidden; padding: 0.2rem 0; }
  .figs { max-width: clamp(62rem, 48rem + 22vw, 76rem); margin: 1.6rem auto 0; padding: 0 clamp(1rem, 4vw, 3rem); }   /* wider than the text, and widening with it: the live charts are dense */
  .figs figure { margin: 0 0 2rem; }
  .figs img { display: block; width: auto; max-width: 100%; height: auto; margin: 0 auto; border: 1px solid var(--c-rule, #dfe3dc); background: #fff; }   /* never wider than the file */
  .figs iframe { display: block; width: 100%; aspect-ratio: 16 / 10; border: 1px solid var(--c-rule, #dfe3dc); background: #fff; }
  .figs figcaption { color: var(--c-muted, #5c6b73); margin-top: 0.5rem; }
  .live { display: inline-block; margin: 0.6rem 0 0; font-weight: 600; }
  .related { border-top: 1px solid var(--c-rule, #dfe3dc); margin-top: 2.6rem; padding-top: 1.2rem; }
  .related ul { list-style: none; padding: 0; display: flex; flex-wrap: wrap; gap: 0.5rem 1.6rem; }
  .refs { font-size: 0.95em; }
  .refs li { margin: 0.4rem 0; }
  .byline { color: var(--c-muted, #5c6b73); font-size: 0.92em; margin-top: 2.2rem; }
  .hub-list { list-style: none; padding: 0; }
  .hub-list li { border-top: 1px solid var(--c-rule, #dfe3dc); padding: 1rem 0; margin: 0; }
  .hub-list a { font-family: var(--f-serif); font-size: 1.3em; text-decoration: none; }
  .hub-list a:hover { text-decoration: underline; }
  .hub-list p { margin: 0.3rem 0 0; color: var(--c-muted, #5c6b73); }
</style>"""

KATEX = """<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body,{delimiters:[{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}],throwOnError:false});"></script>"""

# the loop player posts its content height; size each iframe to it (no letterboxing), as stage.js does
RESIZE = """<script>
addEventListener("message", function (e) {
  var x = e.data; if (!x || x.type !== "sstAnimHeight") return;
  document.querySelectorAll(".figs iframe").forEach(function (f) {
    if (f.contentWindow === e.source) { f.style.height = x.h + "px"; f.style.aspectRatio = "auto"; }
  });
});
// The loop player loads once it is on screen, so no frames load in the background. Native
// loading="lazy" starts an iframe 1250-2500 px before it is visible, i.e. on page open for most readers.
(function () {
  var q = document.querySelectorAll(".figs iframe[data-src]");
  function load(f) { var d = f.getAttribute("data-src"); if (d) { f.setAttribute("src", d); f.removeAttribute("data-src"); } }
  if (!("IntersectionObserver" in window)) { Array.prototype.forEach.call(q, load); return; }
  var io = new IntersectionObserver(function (es) {
    es.forEach(function (e) { if (e.isIntersecting) { io.unobserve(e.target); load(e.target); } });
  }, { rootMargin: "0px" });
  Array.prototype.forEach.call(q, function (f) { io.observe(f); });
})();
</script>"""


def share_image(t: dict | None) -> str:
    """The page's sharing image: its first still figure, else the still of its loop's product, else the site photo."""
    figs = [f["src"] for f in (t or {}).get("figures", []) if f["kind"] == "img"]
    return SITE + (figs[0] if figs else (t or {}).get("still", "/hero-supercell-1280.webp"))


def head(title: str, desc: str, url: str, ld: dict, math: bool, image: str) -> str:
    e = html.escape
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{e(title)} · Shawn Corvec</title>
<meta name="description" content="{e(desc)}">
<link rel="canonical" href="{url}">
<link rel="icon" href="/favicon.svg" type="image/svg+xml">
<meta name="author" content="Shawn Corvec">
<meta property="og:type" content="article">
<meta property="og:title" content="{e(title)}">
<meta property="og:description" content="{e(desc)}">
<meta property="og:url" content="{url}">
<meta property="og:image" content="{image}">
<meta property="og:site_name" content="Shawn Corvec">
<meta name="twitter:card" content="summary_large_image">
<script type="application/ld+json">
{json.dumps(ld, indent=1, ensure_ascii=False)}
</script>
<link rel="stylesheet" href="/assets/site.css">
<script src="/assets/site.js" defer></script>
{KATEX if math else ""}{STYLE}
</head>
<body>
"""


def figure_html(f: dict) -> str:
    if f["kind"] == "loop":
        return (
            f'<figure><iframe data-src="{html.escape(f["src"])}" title="{html.escape(f["title"])}"></iframe>'
            f"<figcaption>{html.escape(f['title'])}. {html.escape(f['how'])}</figcaption></figure>"
        )
    cap = f"<figcaption>{html.escape(f['cap'])}</figcaption>" if f.get("cap") else ""
    return f'<figure><img src="{html.escape(f["src"])}" alt="{html.escape(f["alt"])}" loading="lazy">{cap}</figure>'


def topic_page(t: dict, today: str) -> str:
    url = f"{SITE}/topics/{t['slug']}.html"
    ld = {
        "@context": "https://schema.org",
        "@graph": [
            {
                "@type": "TechArticle",
                "@id": f"{url}#article",
                "headline": t["term"],
                "name": t["title"],
                "description": t["desc"],
                "url": url,
                "mainEntityOfPage": url,
                "inLanguage": "en",
                "author": PERSON,
                "publisher": PERSON,
                "dateModified": today,
                "about": {"@type": "DefinedTerm", "name": t["short"], "alternateName": t["alt"]},
                "keywords": ", ".join([t["short"]] + t["alt"]),
                "image": [f"{SITE}{f['src']}" for f in t["figures"] if f["kind"] == "img"] or share_image(t),
                "citation": [re.sub(r"<[^>]+>", "", r) for r in t["refs"]],
            },
            {
                "@type": "BreadcrumbList",
                "itemListElement": [
                    {"@type": "ListItem", "position": 1, "name": "Home", "item": f"{SITE}/"},
                    {"@type": "ListItem", "position": 2, "name": "Explainers", "item": f"{SITE}/topics/"},
                    {"@type": "ListItem", "position": 3, "name": t["short"], "item": url},
                ],
            },
        ],
    }
    aka = "Also called: " + ", ".join(html.escape(a) for a in t["alt"][:6])
    related = "".join(
        f'<li><a href="/topics/{s}.html">{html.escape(BY_SLUG[s]["short"])}</a></li>'
        for s in t["related"]
        if s in BY_SLUG
    )
    refs = "".join(f"<li>{r}</li>" for r in t["refs"])
    return (
        head(t["title"], t["desc"], url, ld, math=True, image=share_image(t))
        + f"""<main>
<article class="topic">
  <p class="crumbs"><a href="/">Home</a> › <a href="/topics/">Explainers</a> › {html.escape(t["short"])}</p>
  <p class="ai-note"><b>Made with Claude Code.</b> This explainer is a just-for-fun project: researched, coded and written entirely by Claude, Anthropic's AI model, working in Claude Code, to show what today's AI models can do with open meteorological data. It has not been peer reviewed.</p>
  <h1>{html.escape(t["term"])}</h1>
  <p class="aka">{aka}</p>
{t["body"].strip()}
  <a class="live" href="{t["live"][0]}">{html.escape(t["live"][1])}</a>
</article>
<div class="figs">
{chr(10).join(figure_html(f) for f in t["figures"])}
</div>
<div class="topic" style="padding-top:0">
  <section class="related"><h2>Related</h2><ul>{related}<li><a href="/topics/">All explainers</a></li></ul></section>
  <section><h2>References and further reading</h2><ol class="refs">{refs}</ol></section>
{data_credits(t)}  <p class="byline">Written by Claude, Anthropic's AI model, at the direction of <a href="/">Shawn Corvec</a>, meteorologist. The figures on this page are live and update with each forecast cycle; the text was last revised {today}.</p>
</div>
</main>
{RESIZE}
</body>
</html>
"""
    )


DATA_CREDITS = [
    (
        "IFS",  # AIFS, AIFS-ENS, IFS-ENS, IFS HRES
        "ECMWF open data: © 2026 European Centre for Medium-Range Weather Forecasts (ECMWF), source "
        '<a href="https://www.ecmwf.int">www.ecmwf.int</a>, CC BY 4.0, modified; ECMWF does not accept any liability for errors or '
        "omissions in the data, their availability, or any loss or damage arising from their use",
    ),
    (
        "ERA5",
        "ERA5: Copernicus Climate Change Service (C3S) / ECMWF, CC BY 4.0; contains modified Copernicus Climate Change Service "
        "information 2026; neither the European Commission nor ECMWF is responsible for any use of this information",
    ),
    ("GEFS", "NOAA GEFS, NOAA Open Data Dissemination (AWS)"),
    ("NCEP/NCAR", "NCEP/NCAR reanalysis: NOAA PSL, Boulder, Colorado, USA"),
    ("MERRA-2", "MERRA-2: NASA GMAO"),
    ("GFZ", "ESMGFZ effective angular momentum functions: GFZ Potsdam"),
    (
        "GEPS",
        "ECCC GEPS and GDPS: Data Source: Environment and Climate Change Canada (MSC Datamart), under the "
        '<a href="https://eccc-msc.github.io/open-data/licence/readme_en/">ECCC Data Servers End-use Licence</a>',
    ),
    ("GEOS FP", "GEOS FP: NASA GMAO"),
]


def data_credits(t):
    """'Data:' line naming the providers of every dataset the page uses (by the names its text and figures mention)."""
    txt = t["body"] + " ".join(str(f) for f in t["figures"])
    used = [c for k, c in DATA_CREDITS if k in txt]
    return f'  <p class="byline">Data: {"; ".join(used)}.</p>\n' if used else ""


def hub_page(today: str) -> str:
    url = f"{SITE}/topics/"
    title = "Atmospheric dynamics explainers and live forecasts"
    desc = (
        "E–P flux, mountain torque, angular momentum, wave activity flux, eddy heat flux, SSWs, the Hadley cell and "
        "the dynamic tropopause, explained with live data."
    )
    ld = {
        "@context": "https://schema.org",
        "@type": "CollectionPage",
        "url": url,
        "name": title,
        "description": desc,
        "author": PERSON,
        "mainEntity": {
            "@type": "ItemList",
            "itemListElement": [
                {"@type": "ListItem", "position": i + 1, "url": f"{SITE}/topics/{t['slug']}.html", "name": t["term"]}
                for i, t in enumerate(TOPICS)
            ],
        },
    }
    items = "".join(
        f'<li><a href="/topics/{t["slug"]}.html">{html.escape(t["term"])}</a><p>{html.escape(t["desc"])}</p></li>'
        for t in TOPICS
    )
    return (
        head(title, desc, url, ld, math=False, image=share_image(None))
        + f"""<main>
<article class="topic">
  <p class="crumbs"><a href="/">Home</a> › Explainers</p>
  <p class="ai-note"><b>Made with Claude Code.</b> This explainer is a just-for-fun project: researched, coded and written entirely by Claude, Anthropic's AI model, working in Claude Code, to show what today's AI models can do with open meteorological data. It has not been peer reviewed.</p>
  <h1>Atmospheric dynamics explainers, with live forecasts</h1>
  <p>The diagnostics behind this site's circulation and stratosphere products, each explained with its equations and
  shown with its current forecast. They are written for meteorologists and students who want to know what a chart is
  measuring before trusting it.</p>
  <ul class="hub-list">{items}</ul>
  <p class="byline">Written by Claude, Anthropic's AI model, at the direction of <a href="/">Shawn Corvec</a>, meteorologist. Last revised {today}.</p>
</article>
</main>
</body>
</html>
"""
    )


def update_sitemap(paths: list[str], today: str) -> None:
    p = REPO / "sitemap.xml"
    s = p.read_text()
    for loc in paths:
        full = f"{SITE}/{loc}"
        if f"<loc>{full}</loc>" in s:
            s = re.sub(rf"(<loc>{re.escape(full)}</loc>\s*<lastmod>)[^<]*", rf"\g<1>{today}", s)
        else:
            s = s.replace(
                "</urlset>",
                f"  <url>\n    <loc>{full}</loc>\n    <lastmod>{today}</lastmod>\n    <priority>0.6</priority>\n  </url>\n</urlset>",
            )
    p.write_text(s)


def main() -> int:
    today = dt.date.today().isoformat()
    OUT.mkdir(exist_ok=True)
    written = []
    for t in TOPICS:
        (OUT / f"{t['slug']}.html").write_text(topic_page(t, today))
        written.append(f"topics/{t['slug']}.html")
    (OUT / "index.html").write_text(hub_page(today))
    written.append("topics/index.html")
    sys.path.insert(0, str(REPO / "scripts" / "site"))
    import apply_chrome as A  # (PAGES picks up topics/*.html)

    A.stamp_all(only=set(written), quiet=True)
    update_sitemap(["topics/"] + [f"topics/{t['slug']}.html" for t in TOPICS], today)
    print(f"wrote {len(written)} pages to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
