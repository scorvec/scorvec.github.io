#!/usr/bin/env python3
"""Topic explainers: one crawlable page per diagnostic under /topics/ (2026-09-26; user: "I'd like my site to show up
for esoteric search terms like 'E-P flux divergence' and mountain torque").

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


def loop(manifest: str, region: str, title: str) -> dict:
    return {"kind": "loop", "src": f"/sst_anim.html?embed=1&manifest={manifest}&region={region}", "title": title}


def img(path: str, alt: str, cap: str = "") -> dict:
    return {"kind": "img", "src": path, "alt": alt, "cap": cap}


# Each topic: slug, the term (h1), alternate names people search for, title/description, the body (HTML with TeX in
# $...$ / $$...$$), the live figures, where the interactive product lives, related topics and references.
TOPICS = [
    dict(
        slug="ep-flux",
        term="Eliassen–Palm flux and E–P flux divergence",
        short="E–P flux",
        alt=["E-P flux", "EP flux", "Eliassen-Palm flux", "E-P flux divergence", "EP flux divergence", "wave driving",
             "transformed Eulerian mean"],
        title="E–P flux and E–P flux divergence: explainer and live forecast",
        desc="What the Eliassen–Palm (E–P) flux and its divergence measure, how to read an E–P cross-section, and a live "
             "15-day AIFS-ENS forecast of E–P flux and wave driving of the zonal-mean flow.",
        body=r"""
<p>The <b>Eliassen–Palm (E–P) flux</b> is a vector in the latitude–height plane that shows where atmospheric waves
carry their activity, and its <b>divergence</b> is the force those waves exert on the zonal-mean westerlies. It is the
standard diagnostic of wave–mean-flow interaction: of how baroclinic eddies maintain the jet stream, and of how
planetary waves rising out of the troposphere decelerate the stratospheric polar vortex before a sudden warming.</p>

<h2>The equations</h2>
<p>In the transformed Eulerian mean (TEM) form of the zonal-mean momentum equation, the eddies enter only through the
divergence of the E–P flux $\mathbf{F}$:</p>
<p>$$\frac{\partial \bar u}{\partial t} - f\,\bar v^{*} = \frac{\nabla\cdot\mathbf{F}}{\rho_0\,a\cos\phi} + \bar X,
\qquad \nabla\cdot\mathbf{F} = \frac{1}{a\cos\phi}\frac{\partial}{\partial\phi}\!\left(F^{(\phi)}\cos\phi\right) + \frac{\partial F^{(z)}}{\partial z}$$</p>
<p>where $\bar v^{*}$ is the residual meridional circulation and $\bar X$ is any other force, such as gravity-wave drag.
In the quasi-geostrophic approximation the two components are</p>
<p>$$F^{(\phi)} = -\rho_0\,a\cos\phi\;\overline{u'v'}, \qquad F^{(z)} = \rho_0\,a\cos\phi\;f\,\frac{\overline{v'\theta'}}{\partial\bar\theta/\partial z}$$</p>
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
The same wave drag, integrated, drives the Brewer–Dobson circulation.</li>
<li><b>Divergence</b> ($\nabla\cdot\mathbf{F}>0$) is a westerly force.</li>
<li>Only the longest planetary waves (zonal wavenumbers 1–3) can propagate into the winter stratospheric westerlies
(Charney and Drazin, 1961), which is why stratospheric E–P diagnostics are usually restricted to them.</li>
</ul>

<h2>The live forecast</h2>
<p>The loop below is the E–P flux for zonal wavenumbers 1–3 and its divergence, computed from every member of the
ECMWF AIFS ensemble (the control and 25 perturbed members, 14 pressure levels) and stepped through the 15-day forecast.
The flux is quadratic in the wave amplitude, so it is computed for each member and then averaged; the flux of the
ensemble-mean fields would fade with lead time as the members' wave phases drift apart. The quasi-geostrophic form is
used with a global-mean static stability, and latitudes poleward of 82° are masked. It is recomputed every 00 and
12 UTC cycle.</p>
""",
        figures=[loop("epflux_manifest.json", "epflux", "E–P flux and wave driving, wavenumbers 1–3, AIFS-ENS 15-day forecast")],
        live=("/stratosphere.html#epflux", "Open the E–P flux product on the stratosphere page"),
        related=["eddy-heat-flux", "sudden-stratospheric-warming", "wave-activity-flux", "mountain-torque"],
        refs=[
            "Eliassen, A., and E. Palm, 1961: On the transfer of energy in stationary mountain waves. <i>Geofysiske Publikasjoner</i>, 22(3), 1–23.",
            "Charney, J. G., and P. G. Drazin, 1961: Propagation of planetary-scale disturbances from the lower into the upper atmosphere. <i>J. Geophys. Res.</i>, 66, 83–109.",
            "Andrews, D. G., and M. E. McIntyre, 1976: Planetary waves in horizontal and vertical shear: the generalized Eliassen–Palm relation and the mean zonal acceleration. <i>J. Atmos. Sci.</i>, 33, 2031–2048.",
            "Edmon, H. J., B. J. Hoskins, and M. E. McIntyre, 1980: Eliassen–Palm cross sections for the troposphere. <i>J. Atmos. Sci.</i>, 37, 2600–2616.",
            "Andrews, D. G., J. R. Holton, and C. B. Leovy, 1987: <i>Middle Atmosphere Dynamics</i>. Academic Press.",
        ],
    ),
    dict(
        slug="mountain-torque",
        term="Mountain torque and friction torque",
        short="Mountain torque",
        alt=["mountain torque", "friction torque", "frictional torque", "surface torque", "torque budget",
             "atmospheric angular momentum budget", "gravity wave drag torque"],
        title="Mountain torque and friction torque: the atmosphere's angular momentum budget, with a live forecast",
        desc="How mountain and friction torques exchange angular momentum between the solid Earth and the atmosphere, "
             "the equations, how to read torque maps, and live AIFS-ENS forecasts of both torques by mountain range.",
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
momentum. On this site's own test (ERA5, 1991–2020), strong Himalayan torque days are followed about five days later
by a stronger jet exit over the North Pacific.</li>
</ul>

<h2>The live forecast</h2>
<p>The friction and mountain torques are computed from the ECMWF AIFS ensemble for days 0–15 as anomalies from the
ERA5 1991–2020 climatology for the same time of year. Anomalies are used on purpose: the absolute budget cannot be
closed. ERA5's own terms sum to −4.5 ± 0.7 Hadleys in the annual mean where the answer must be zero, and the resolved
mountain torque changes with grid spacing. Both errors sit in the mean and cancel in an anomaly. The open forecast
data carry no gravity-wave stress, so that term stays in the residual between the net torque and the actual change in
angular momentum.</p>
""",
        figures=[
            loop("torque_manifest.json", "torque", "Friction and mountain torque-density anomalies, AIFS-ENS days 0–15"),
            img("/assets/sst/torque_timeseries.webp", "Global and hemispheric friction and mountain torque anomalies against the change in atmospheric angular momentum, AIFS-ENS forecast",
                "The torques integrated over the globe and each hemisphere, against the actual change in angular momentum."),
            img("/assets/sst/torque_ranges.webp", "Mountain torque by mountain range: Rockies, Andes, Tibetan Plateau and others, AIFS-ENS forecast",
                "Which mountain ranges are doing the work."),
        ],
        live=("/circulation.html#torque", "Open the surface torque product in the circulation viewer"),
        related=["atmospheric-angular-momentum", "wave-activity-flux", "ep-flux"],
        refs=[
            "Weickmann, K. M., and P. D. Sardeshmukh, 1994: The atmospheric angular momentum cycle associated with a Madden–Julian oscillation. <i>J. Atmos. Sci.</i>, 51, 3194–3208.",
            "Lott, F., A. W. Robertson, and M. Ghil, 2004: Mountain torques and Northern Hemisphere low-frequency variability. Part I: Hemispheric aspects. <i>J. Atmos. Sci.</i>, 61, 1259–1271.",
            "Egger, J., K. Weickmann, and K.-P. Hoinka, 2007: Angular momentum in the global atmospheric circulation. <i>Rev. Geophys.</i>, 45, RG4007.",
        ],
    ),
    dict(
        slug="atmospheric-angular-momentum",
        term="Atmospheric angular momentum (AAM)",
        short="Atmospheric angular momentum",
        alt=["AAM", "relative AAM", "global atmospheric angular momentum", "Global Wind Oscillation", "GWO",
             "length of day", "AAM tendency"],
        title="Atmospheric angular momentum (AAM): explainer and live forecast",
        desc="What atmospheric angular momentum measures, why it tracks El Niño and the length of day, the Global Wind "
             "Oscillation phase space, and a live AIFS-ENS forecast of relative AAM for the globe and each hemisphere.",
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
tropics hold weak or easterly winds, so they add little to the total while dominating its variability: a small change
in the trades acts over a huge area at the largest weight.</li>
<li><b>El Niño raises AAM.</b> It strengthens the subtropical jets and weakens the trades, so the atmosphere holds
more westerly momentum. Conservation then requires the solid Earth to slow down: AAM anomalies appear as changes of a
fraction of a millisecond in the length of day, and about 6×10<sup>25</sup> kg m<sup>2</sup> s<sup>−1</sup> of AAM
corresponds to one millisecond.</li>
<li><b>It can change only through surface torques</b>, the <a href="/topics/mountain-torque.html">mountain and
friction torques</a>. Mountain torque works within days; friction over weeks.</li>
<li><b>The Global Wind Oscillation</b> (Weickmann and Berry, 2009) follows the global AAM anomaly against its
tendency in a two-dimensional phase space, much as the MJO index follows tropical convection. Orbits through it pass
through the same sequence of jet and torque patterns in each subseasonal cycle.</li>
</ul>

<h2>The live forecast</h2>
<p>Relative AAM is integrated for the globe and each hemisphere from the ECMWF AIFS ensemble's winds on 14 pressure
levels from 10 to 1000 hPa, which is effectively the whole column. It is shown on the ERA5 1991–2020 annual cycle
together with the 15-day forecast. GFZ's ESMGFZ series, computed from ECMWF analyses for the Earth-rotation community,
is drawn alongside as an independent check. The phase plot follows each hemisphere through AAM anomaly against
tendency: the hemispheric version of the Global Wind Oscillation.</p>
""",
        figures=[
            img("/assets/sst/aam.webp", "Global and hemispheric relative atmospheric angular momentum: observed and AIFS-ENS 15-day forecast, absolute and anomaly",
                "Relative AAM for the globe and each hemisphere on its 1991–2020 annual cycle, with the 15-day forecast."),
            img("/assets/sst/aam_phase.webp", "Atmospheric angular momentum anomaly against its tendency: hemispheric Global Wind Oscillation phase space with the AIFS-ENS forecast",
                "Each hemisphere's path through AAM anomaly and tendency, the last 75 days and the forecast."),
        ],
        live=("/circulation.html#aam", "Open the AAM product in the circulation viewer"),
        related=["mountain-torque", "hadley-cell", "wave-activity-flux"],
        refs=[
            "Rosen, R. D., and D. A. Salstein, 1983: Variations in atmospheric angular momentum on global and regional scales and the length of day. <i>J. Geophys. Res.</i>, 88, 5451–5470.",
            "Weickmann, K. M., and P. D. Sardeshmukh, 1994: The atmospheric angular momentum cycle associated with a Madden–Julian oscillation. <i>J. Atmos. Sci.</i>, 51, 3194–3208.",
            "Egger, J., K. Weickmann, and K.-P. Hoinka, 2007: Angular momentum in the global atmospheric circulation. <i>Rev. Geophys.</i>, 45, RG4007.",
            "Weickmann, K., and E. Berry, 2009: The tropical Madden–Julian oscillation and the global wind oscillation. <i>Mon. Wea. Rev.</i>, 137, 1601–1614.",
        ],
    ),
    dict(
        slug="wave-activity-flux",
        term="Wave activity flux (Takaya–Nakamura)",
        short="Wave activity flux",
        alt=["Takaya-Nakamura flux", "Takaya-Nakamura wave activity flux", "TN01", "Plumb flux", "Rossby wave packet",
             "Rossby wave propagation", "WAF"],
        title="Takaya–Nakamura wave activity flux: explainer and live 250 hPa forecast",
        desc="What the Takaya–Nakamura wave activity flux shows about Rossby wave packets, downstream development and "
             "blocking, the equation, and a live member-averaged AIFS-ENS forecast at 250 hPa.",
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
<p>It generalizes Plumb's (1985) flux for stationary waves on a zonal flow to a zonally varying basic state and to
slowly migrating waves. It is <i>phase-independent</i>: it does not oscillate between ridge and trough, so packets
can be followed without averaging over a wavelength.</p>

<h2>How to read it</h2>
<ul>
<li>Arrows arcing along the jet are a wave train; follow them to see where a disturbance will be felt next.</li>
<li>Where the arrows converge and slow down, the wave amplifies: that is where ridges build and blocks form.</li>
<li>A packet leaving the tropical Pacific toward North America is the tropical heating reaching the extratropics.</li>
</ul>

<h2>The live forecast</h2>
<p>The flux is computed at 250 hPa, the level of the jet core and the stationary-wave maximum, for each of 25 AIFS
ensemble members at every day to day 15, and then averaged. Because the flux is quadratic in $\psi'$, the flux of the
ensemble mean would fade with lead time as the members decorrelate, while the mean of the members' fluxes keeps the
packets that are actually forecast. Each member's $\psi'$ is smoothed with a 5-day running mean along the forecast,
because the theory is for quasi-stationary waves. The basic state is the ERA5 1991–2020 climatology plus the 30-day
mean anomaly of the AIFS analyses, and shading is hatched where fewer than 60 % of members agree on the sign of the
divergence.</p>
""",
        figures=[
            loop("waf_manifest.json", "waf", "Takaya–Nakamura wave activity flux at 250 hPa, AIFS-ENS member mean, days 0–15"),
        ],
        live=("/circulation.html#waf", "Open the wave activity flux in the circulation viewer"),
        related=["ep-flux", "mountain-torque", "dynamic-tropopause"],
        refs=[
            "Hoskins, B. J., and D. J. Karoly, 1981: The steady linear response of a spherical atmosphere to thermal and orographic forcing. <i>J. Atmos. Sci.</i>, 38, 1179–1196.",
            "Plumb, R. A., 1985: On the three-dimensional propagation of stationary waves. <i>J. Atmos. Sci.</i>, 42, 217–229.",
            "Takaya, K., and H. Nakamura, 2001: A formulation of a phase-independent wave-activity flux for stationary and migratory quasigeostrophic eddies on a zonally varying basic flow. <i>J. Atmos. Sci.</i>, 58, 608–627.",
        ],
    ),
    dict(
        slug="eddy-heat-flux",
        term="100 hPa eddy heat flux",
        short="Eddy heat flux",
        alt=["eddy heat flux", "meridional eddy heat flux", "v'T'", "vT 100 hPa", "upward wave activity",
             "vertical E-P flux", "wave-1 heat flux"],
        title="100 hPa eddy heat flux: the upward wave activity into the stratosphere, with a live forecast",
        desc="Why the zonal-mean poleward eddy heat flux at 100 hPa measures planetary waves entering the stratosphere, "
             "how its 40-day mean leads the polar vortex, and a live member-by-member AIFS-ENS forecast for both hemispheres.",
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
<li>Averaged over January and February, the 45–75°N heat flux predicts the Arctic stratosphere's temperature in March
(Newman et al., 2001), and with it how much ozone survives into spring.</li>
<li>In early autumn radiative cooling still dominates, so a burst of heat flux slows the vortex's seasonal spin-up
rather than reversing it.</li>
</ul>

<h2>The live forecast</h2>
<p>The flux is averaged over 45–75° in each hemisphere (positive is poleward in both), computed for the AIFS control
and 25 perturbed members and then averaged: like every quadratic diagnostic, the flux of the ensemble-mean fields
would fade with lead time. Zonal wavenumbers 1–72 are kept so that the forecast resolves the same eddies as the
1991–2020 NCEP/NCAR reanalysis climatology it is compared with. The panels show the daily flux, its wave-1 and wave-2
parts, the trailing 40-day mean standardized, and the flux by latitude through the forecast.</p>
""",
        figures=[
            img("/assets/sst/heatflux100_nh.webp", "100 hPa zonal-mean eddy heat flux 45–75°N: AIFS-ENS members against the 1991–2020 climatology, wave-1 and wave-2 parts, 40-day mean",
                "Northern Hemisphere, 45–75°N. The Southern Hemisphere version is on the stratosphere page."),
        ],
        live=("/stratosphere.html#heatflux", "Open the eddy heat flux on the stratosphere page"),
        related=["ep-flux", "sudden-stratospheric-warming"],
        refs=[
            "Newman, P. A., E. R. Nash, and J. E. Rosenfield, 2001: What controls the temperature of the Arctic stratosphere during the spring? <i>J. Geophys. Res.</i>, 106, 19999–20010.",
            "Polvani, L. M., and D. W. Waugh, 2004: Upward wave activity flux as a precursor to extreme stratospheric events and subsequent anomalous surface weather regimes. <i>J. Climate</i>, 17, 3548–3554.",
            "Andrews, D. G., J. R. Holton, and C. B. Leovy, 1987: <i>Middle Atmosphere Dynamics</i>. Academic Press.",
        ],
    ),
    dict(
        slug="sudden-stratospheric-warming",
        term="Sudden stratospheric warmings and the polar vortex",
        short="Sudden stratospheric warmings",
        alt=["SSW", "sudden stratospheric warming", "major SSW", "polar vortex", "polar vortex split",
             "vortex displacement", "Charlton-Polvani", "stratosphere-troposphere coupling", "dripping paint"],
        title="Sudden stratospheric warmings (SSW): definition, every event since 1980, and a live polar vortex forecast",
        desc="What a sudden stratospheric warming is, split and displacement events, how SSWs reach the surface, a "
             "catalogue of every major SSW since 1980 from MERRA-2, and a live AIFS-ENS polar vortex forecast.",
        body=r"""
<p>A <b>sudden stratospheric warming (SSW)</b> is the collapse of the winter polar vortex: planetary waves rising from
the troposphere decelerate the stratospheric westerlies until they reverse, and the polar stratosphere warms by tens of
degrees within days. The usual definition (Charlton and Polvani, 2007) is the first day in November–March on which the
zonal-mean zonal wind at 60°N and 10 hPa turns easterly. A second event in the same winter must be separated by 20
consecutive westerly days, and a reversal that never recovers before spring is the final warming, not an SSW.</p>

<h2>Splits and displacements</h2>
<p>Seen from above, the vortex either shifts off the pole (a <b>displacement</b>, driven mostly by zonal wavenumber 1)
or breaks into two separate lows (a <b>split</b>, driven by wavenumber 2). Splits tend to be more abrupt and to reach
deeper into the lower stratosphere.</p>

<h2>How they reach the surface</h2>
<p>Baldwin and Dunkerton (2001) showed that large stratospheric anomalies work their way down over a few weeks and then
hold the surface in the same state for one to two months. Drawn against time and height the composite looks like paint
dripping down a wall. After a warming that reaches the lower stratosphere, the Arctic Oscillation and North Atlantic
Oscillation tend to turn negative, bringing cold air into northern Eurasia and eastern North America.</p>

<h2>Every warming since 1980</h2>
<p>This site's catalogue, built from NASA's MERRA-2 reanalysis, finds 28 major warmings in 24 of the 46 winters since
1980, 10 of them splits. The events are split into deep and shallow by how strong the anomaly is at 100–150 hPa over the
following month. After the 14 deep events the Arctic Oscillation averaged 1.42 below the same dates in other years over
days 1–30, and 0.69 below over days 31–60. Both differences are significant (t-test, false-discovery rate 10 %).
After the 14 shallow events there was no significant change.</p>

<h2>The live forecast</h2>
<p>The vortex chart tracks the zonal-mean wind at 60°N at 10 and 100 hPa and the polar-cap height at 100 hPa for every
AIFS ensemble member against the MERRA-2 percentile bands, so that the fraction of members reversing the wind (a
forecast major warming) can be read directly.</p>
""",
        figures=[
            img("/assets/sst/nh_vortex.webp", "Northern polar vortex forecast: zonal-mean wind at 60°N at 10 and 100 hPa and 100 hPa polar-cap height, AIFS-ENS members against MERRA-2 percentiles",
                "The live vortex forecast: every AIFS-ENS member against the MERRA-2 1980–2026 range."),
            img("/assets/sst/strat_hist_timeline.webp", "Timeline of every major sudden stratospheric warming and strong-vortex event since 1980 from MERRA-2, with ENSO and the QBO",
                "Every major warming and strong-vortex event since 1980, winter by winter."),
            img("/assets/sst/strat_hist_drip_ssw_deep.webp", "Dripping-paint composite of polar-cap height from the surface to 1 hPa around deep sudden stratospheric warmings, with the Arctic Oscillation",
                "The dripping-paint composite for the deep events: colour only where significant."),
        ],
        live=("/stratosphere.html#vortex", "Open the polar vortex forecast and the SSW history on the stratosphere page"),
        related=["eddy-heat-flux", "ep-flux", "wave-activity-flux"],
        refs=[
            "Baldwin, M. P., and T. J. Dunkerton, 2001: Stratospheric harbingers of anomalous weather regimes. <i>Science</i>, 294, 581–584.",
            "Charlton, A. J., and L. M. Polvani, 2007: A new look at stratospheric sudden warmings. Part I: Climatology and modeling benchmarks. <i>J. Climate</i>, 20, 449–469.",
            "Butler, A. H., J. P. Sjoberg, D. J. Seidel, and K. H. Rosenlof, 2017: A sudden stratospheric warming compendium. <i>Earth Syst. Sci. Data</i>, 9, 63–76.",
            "Baldwin, M. P., et al., 2021: Sudden stratospheric warmings. <i>Rev. Geophys.</i>, 59, e2020RG000708.",
        ],
    ),
    dict(
        slug="hadley-cell",
        term="Hadley cell and the mean meridional mass streamfunction",
        short="Hadley cell",
        alt=["Hadley circulation", "mean meridional streamfunction", "meridional mass streamfunction", "MMSF",
             "Ferrel cell", "Hadley cell strength", "Hadley cell edge"],
        title="Hadley cell and the mean meridional mass streamfunction: explainer and live analysis",
        desc="How the zonal-mean meridional mass streamfunction describes the Hadley, Ferrel and polar cells, how cell "
             "strength and edges are measured, and a live AIFS analysis of the overturning against ERA5 1991–2020.",
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
the subtropical dry zones, and its poleward drift in a warming climate is one of the clearest circulation trends.</li>
<li>The cells follow the sun: the winter hemisphere's cell dominates and extends across the equator, while the summer
cell nearly vanishes.</li>
<li>El Niño strengthens the Hadley circulation and pulls its edges equatorward (Oort and Yienger, 1996).</li>
</ul>

<h2>The live analysis</h2>
<p>$\Psi$ is computed from the ECMWF AIFS analysis and shown as the anomaly from the ERA5 1991–2020 climatology for the
same day of year, with the absolute cells contoured and vertical motion drawn as arrows. Each frame is a 7-day running
mean, so the evolution of the cells is legible through the day-to-day noise.</p>
""",
        figures=[
            loop("mmsf_manifest.json", "mmsf", "Meridional mass streamfunction anomaly, 7-day mean, AIFS analysis against ERA5 1991–2020"),
        ],
        live=("/circulation.html#hadley", "Open the Hadley cell product in the circulation viewer"),
        related=["atmospheric-angular-momentum", "ep-flux"],
        refs=[
            "Held, I. M., and A. Y. Hou, 1980: Nonlinear axially symmetric circulations in a nearly inviscid atmosphere. <i>J. Atmos. Sci.</i>, 37, 515–533.",
            "Oort, A. H., and J. J. Yienger, 1996: Observed interannual variability in the Hadley circulation and its connection to ENSO. <i>J. Climate</i>, 9, 2751–2767.",
            "Dima, I. M., and J. M. Wallace, 2003: On the seasonality of the Hadley cell. <i>J. Atmos. Sci.</i>, 60, 1522–1527.",
        ],
    ),
    dict(
        slug="dynamic-tropopause",
        term="Dynamic tropopause and potential vorticity (2 PVU)",
        short="Dynamic tropopause",
        alt=["dynamic tropopause map", "DT map", "theta on 2 PVU", "potential temperature on the dynamic tropopause",
             "potential vorticity", "PV", "isentropic potential vorticity", "PV thinking", "Rossby wave breaking"],
        title="Dynamic tropopause (2 PVU) maps: potential vorticity explainer and live forecast",
        desc="What the dynamic tropopause is, how to read potential temperature on the 2-PVU surface, isentropic potential "
             "vorticity on 330 and 350 K, and live AIFS forecast maps to day 10.",
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
<p>Ertel PV is computed from the temperature and wind of AIFS ensemble member 0 on nine levels from 700 to 100 hPa,
every 12 hours to day 10. A single member is used on purpose: potential vorticity does not survive ensemble averaging,
because the mean of fifty differently placed folds is a smear. The 2-PVU surface is found by searching upward from
700 hPa, and isentropic PV by interpolation in θ.</p>
""",
        figures=[
            loop("dt_manifest.json", "dt", "Potential temperature on the 2-PVU dynamic tropopause, AIFS member 0, 12-hourly to day 10"),
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
  .topic { max-width: 46rem; margin: 0 auto; padding: 2rem 1.25rem 3.5rem; }
  .topic .crumbs { font-size: 0.92rem; color: var(--c-muted, #5c6b73); margin-bottom: 1.2rem; }
  .topic .crumbs a { color: inherit; }
  .topic h1 { text-align: left; margin: 0 0 0.4rem; }
  .topic .aka { color: var(--c-muted, #5c6b73); font-size: 0.95rem; margin: 0 0 1.6rem; }
  .topic h2 { margin: 2.2rem 0 0.7rem; text-align: left; }
  .topic p, .topic li { line-height: 1.6; }
  .topic ul { padding-left: 1.2rem; }
  .topic li { margin: 0.45rem 0; }
  .topic a { color: var(--c-accent, #274b7a); text-underline-offset: 0.2em; }
  .topic .katex-display { overflow-x: auto; overflow-y: hidden; padding: 0.2rem 0; }
  .figs { max-width: 64rem; margin: 1.6rem auto 0; padding: 0 1.25rem; }
  .figs figure { margin: 0 0 2rem; }
  .figs img { display: block; width: 100%; height: auto; border: 1px solid var(--c-rule, #dfe3dc); background: #fff; }
  .figs iframe { display: block; width: 100%; aspect-ratio: 16 / 10; border: 1px solid var(--c-rule, #dfe3dc); background: #fff; }
  .figs figcaption { color: var(--c-muted, #5c6b73); margin-top: 0.5rem; }
  .live { display: inline-block; margin: 0.6rem 0 0; font-weight: 600; }
  .related { border-top: 1px solid var(--c-rule, #dfe3dc); margin-top: 2.6rem; padding-top: 1.2rem; }
  .related ul { list-style: none; padding: 0; display: flex; flex-wrap: wrap; gap: 0.5rem 1.6rem; }
  .refs { font-size: 0.95rem; }
  .refs li { margin: 0.4rem 0; }
  .byline { color: var(--c-muted, #5c6b73); font-size: 0.92rem; margin-top: 2.2rem; }
  .hub-list { list-style: none; padding: 0; }
  .hub-list li { border-top: 1px solid var(--c-rule, #dfe3dc); padding: 1rem 0; margin: 0; }
  .hub-list a { font-family: var(--f-serif); font-size: 1.3rem; text-decoration: none; }
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
</script>"""


def head(title: str, desc: str, url: str, ld: dict, math: bool) -> str:
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
<meta property="og:image" content="{SITE}/hero-supercell-1280.webp">
<meta property="og:site_name" content="Shawn Corvec">
<meta name="twitter:card" content="summary_large_image">
<script type="application/ld+json">
{json.dumps(ld, indent=1, ensure_ascii=False)}
</script>
<script data-goatcounter="https://scorvec.goatcounter.com/count" async src="//gc.zgo.at/count.js"></script>
<link rel="stylesheet" href="/assets/site.css">
<script src="/assets/site.js" defer></script>
{KATEX if math else ""}{STYLE}
</head>
<body>
"""


def figure_html(f: dict) -> str:
    if f["kind"] == "loop":
        return (f'<figure><iframe src="{f["src"]}" title="{html.escape(f["title"])}" loading="lazy"></iframe>'
                f'<figcaption>{html.escape(f["title"])}. Use the controls to step through the forecast.</figcaption></figure>')
    cap = f'<figcaption>{html.escape(f["cap"])}</figcaption>' if f.get("cap") else ""
    return f'<figure><img src="{f["src"]}" alt="{html.escape(f["alt"])}" loading="lazy">{cap}</figure>'


def topic_page(t: dict, today: str) -> str:
    url = f"{SITE}/topics/{t['slug']}.html"
    ld = {"@context": "https://schema.org", "@graph": [
        {"@type": "TechArticle", "@id": f"{url}#article", "headline": t["term"], "name": t["title"],
         "description": t["desc"], "url": url, "mainEntityOfPage": url, "inLanguage": "en",
         "author": PERSON, "publisher": PERSON, "dateModified": today,
         "about": {"@type": "DefinedTerm", "name": t["short"], "alternateName": t["alt"]},
         "keywords": ", ".join([t["short"]] + t["alt"]),
         "image": [f"{SITE}{f['src']}" for f in t["figures"] if f["kind"] == "img"] or f"{SITE}/hero-supercell-1280.webp",
         "citation": [re.sub(r"<[^>]+>", "", r) for r in t["refs"]]},
        {"@type": "BreadcrumbList", "itemListElement": [
            {"@type": "ListItem", "position": 1, "name": "Home", "item": f"{SITE}/"},
            {"@type": "ListItem", "position": 2, "name": "Explainers", "item": f"{SITE}/topics/"},
            {"@type": "ListItem", "position": 3, "name": t["short"], "item": url}]}]}
    aka = "Also called: " + ", ".join(html.escape(a) for a in t["alt"][:6])
    related = "".join(f'<li><a href="/topics/{s}.html">{html.escape(BY_SLUG[s]["short"])}</a></li>' for s in t["related"] if s in BY_SLUG)
    refs = "".join(f"<li>{r}</li>" for r in t["refs"])
    return (head(t["title"], t["desc"], url, ld, math=True)
            + f"""<main>
<article class="topic">
  <p class="crumbs"><a href="/">Home</a> › <a href="/topics/">Explainers</a> › {html.escape(t["short"])}</p>
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
  <section><h2>References</h2><ol class="refs">{refs}</ol></section>
  <p class="byline">By <a href="/">Shawn Corvec</a>, meteorologist. The figures on this page are live and update with each forecast cycle; the text was last revised {today}.</p>
</div>
</main>
{RESIZE}
</body>
</html>
""")


def hub_page(today: str) -> str:
    url = f"{SITE}/topics/"
    title = "Atmospheric dynamics explainers, with live forecasts"
    desc = ("Plain explanations of E–P flux, mountain torque, atmospheric angular momentum, wave activity flux, eddy heat "
            "flux, sudden stratospheric warmings, the Hadley cell and the dynamic tropopause, each with a live AIFS-ENS forecast.")
    ld = {"@context": "https://schema.org", "@type": "CollectionPage", "url": url, "name": title, "description": desc,
          "author": PERSON, "mainEntity": {"@type": "ItemList", "itemListElement": [
              {"@type": "ListItem", "position": i + 1, "url": f"{SITE}/topics/{t['slug']}.html", "name": t["term"]}
              for i, t in enumerate(TOPICS)]}}
    items = "".join(f'<li><a href="/topics/{t["slug"]}.html">{html.escape(t["term"])}</a><p>{html.escape(t["desc"])}</p></li>' for t in TOPICS)
    return (head(title, desc, url, ld, math=False)
            + f"""<main>
<article class="topic">
  <p class="crumbs"><a href="/">Home</a> › Explainers</p>
  <h1>{title}</h1>
  <p>The diagnostics behind this site's circulation and stratosphere products, each explained with its equations and
  shown with its current forecast. They are written for meteorologists and students who want to know what a chart is
  measuring before trusting it.</p>
  <ul class="hub-list">{items}</ul>
  <p class="byline">By <a href="/">Shawn Corvec</a>, meteorologist. Last revised {today}.</p>
</article>
</main>
</body>
</html>
""")


def update_sitemap(paths: list[str], today: str) -> None:
    p = REPO / "sitemap.xml"; s = p.read_text()
    for loc in paths:
        full = f"{SITE}/{loc}"
        if f"<loc>{full}</loc>" in s:
            s = re.sub(rf"(<loc>{re.escape(full)}</loc>\s*<lastmod>)[^<]*", rf"\g<1>{today}", s)
        else:
            s = s.replace("</urlset>", f"  <url>\n    <loc>{full}</loc>\n    <lastmod>{today}</lastmod>\n    <priority>0.6</priority>\n  </url>\n</urlset>")
    p.write_text(s)


def main() -> int:
    today = dt.date.today().isoformat()
    OUT.mkdir(exist_ok=True)
    written = []
    for t in TOPICS:
        (OUT / f"{t['slug']}.html").write_text(topic_page(t, today)); written.append(f"topics/{t['slug']}.html")
    (OUT / "index.html").write_text(hub_page(today)); written.append("topics/index.html")
    sys.path.insert(0, str(REPO / "scripts" / "site"))
    import apply_chrome as A                                            # noqa: E402  (PAGES picks up topics/*.html)
    A.stamp_all(only=set(written), quiet=True)
    update_sitemap(["topics/"] + [f"topics/{t['slug']}.html" for t in TOPICS], today)
    print(f"wrote {len(written)} pages to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
