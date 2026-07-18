# Julia/Makie TC storm-panel renderer — draws an ensemble storm from the
# tracker's tracks.json (assets/tc/tracks.json), Weathernerds-style.
# First Julia experiment for the site (2026-07-18); candidate base for a
# future Makie port of the TC animation rendering.
#   julia tc_storm_plot.jl
using Pkg
Pkg.activate(@__DIR__; io=devnull)          # env = this directory (Project.toml)
for pkg in ("CairoMakie", "GeoMakie", "JSON3")
    if Base.find_package(pkg) === nothing
        @info "installing $pkg (first run precompiles — takes a while)"
        Pkg.add(pkg)
    end
end
using CairoMakie, GeoMakie, JSON3

t0 = time()
data = JSON3.read(read("/Users/shawn/scorvec.github.io/assets/tc/tracks.json"))
storm = data.storms[argmax([s.support for s in data.storms])]
@info "storm $(storm.id) — support $(round(Int, storm.support*100))% · $(length(storm.tracks)) member tracks"

# Weathernerds wind bins (kt) on 10 m winds (m/s → kt)
function wind_color(w_ms)
    kt = w_ms * 1.94384
    kt >= 80 ? colorant"#e01e9d" : kt >= 70 ? colorant"#e2231e" :
    kt >= 60 ? colorant"#f08a1e" : kt >= 50 ? colorant"#e6d82e" :
    kt >= 40 ? colorant"#2fb52f" : kt >= 30 ? colorant"#2a52dd" :
    kt >= 20 ? colorant"#25c8c8" : colorant"#b8b8b8"
end
using Colors

using Statistics
lats = Float64[]; lons = Float64[]
for tr in storm.tracks, fx in tr.fix
    push!(lats, fx[2]); push!(lons, fx[3])
end
lo1, lo2 = quantile(lons, 0.05) - 4, quantile(lons, 0.95) + 4
la1, la2 = quantile(lats, 0.05) - 3, quantile(lats, 0.95) + 3
limits = (lo1, lo2, la1, la2)

fig = Figure(size = (1400, 1100), backgroundcolor = :white, fontsize = 18)
ga = GeoAxis(fig[1, 1];
    dest = "+proj=merc",
    limits = (limits[1:2], limits[3:4]),
    title = "$(storm.id) — $(round(Int, storm.support*100))% support" *
            (isempty(storm.invest) ? "" : " · ≈ $(storm.invest)") *
            " · init $(data.init) · rendered with Julia + Makie",
    titlesize = 20, xgridstyle = :dash, ygridstyle = :dash,
    ytickformat = vs -> [string(round(Int, v)) * "°" for v in vs],
    xtickformat = vs -> [string(round(Int, v)) * "°" for v in vs],
    xgridcolor = (:steelblue, 0.3), ygridcolor = (:steelblue, 0.3))

# land fill + coastlines over an ocean-colored backdrop
translate!(poly!(ga, Rect2f(limits[1], limits[3], limits[2]-limits[1], limits[4]-limits[3]);
                 color = "#d5ecf5"), 0, 0, -10)
poly!(ga, GeoMakie.land(); color = "#e8dcb8")
lines!(ga, GeoMakie.coastlines(); color = "#8a8a7a", linewidth = 0.7)

# member tracks, per-segment wind colors
for tr in storm.tracks
    fx = tr.fix
    for i in 1:length(fx)-1
        lines!(ga, [fx[i][3], fx[i+1][3]], [fx[i][2], fx[i+1][2]];
               color = (wind_color(fx[i+1][6]), 0.75), linewidth = 1.6)
    end
end

# ensemble-mean track (per-hour mean position where ≥30% of members report)
hours = sort(unique(vcat([[fx[1] for fx in tr.fix] for tr in storm.tracks]...)))
mlat = Float64[]; mlon = Float64[]; mh = Int[]
for h in hours
    ps = [(fx[2], fx[3]) for tr in storm.tracks for fx in tr.fix if fx[1] == h]
    if length(ps) >= 0.3 * length(storm.tracks)
        push!(mlat, mean(first.(ps))); push!(mlon, mean(last.(ps))); push!(mh, h)
    end
end
lines!(ga, mlon, mlat; color = :black, linewidth = 5)
for (h, lo, la) in zip(mh, mlon, mlat)
    h % 48 == 0 && text!(ga, lo, la + 0.55; text = "d$(h ÷ 24)", fontsize = 13,
                         color = (:black, 0.6), align = (:center, :bottom))
end

out = "/Users/shawn/scorvec.github.io/assets/tc/julia_demo.png"
save(out, fig)
@info "saved $out in $(round(time() - t0, digits=1)) s (excl. package precompile)"
