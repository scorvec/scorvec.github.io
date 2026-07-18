# TC animation frame renderer (Makie port of tc_tracker._render_frame).
# One warm process renders every frame for every region × model:
#   julia tc_render.jl <render_spec.json>
# The spec is written by tc_tracker.py; frames land as F%02d.png in
# <out_dir>/<rid>/ (Python converts to webp + writes manifests afterwards).
using Pkg
Pkg.activate(@__DIR__; io=devnull)
using CairoMakie, GeoMakie, JSON3, NPZ, ColorSchemes, Colors, Statistics, Dates
CairoMakie.activate!()

const spec = JSON3.read(read(ARGS[1]))
const init = DateTime(spec.init)
const NDAYS = spec.ndays
const LM = npzread(String(spec.land_mask))
const MLAT = LM["lat"];  const MLON = LM["lon"];  const LAND = LM["land"]

t_start = time()

is_ocean(la, lo) = begin
    i = round(Int, (la - MLAT[1]) / 0.5) + 1
    j = mod(round(Int, (mod(lo + 180, 360) - 180 - MLON[1]) / 0.5), length(MLON)) + 1
    (1 <= i <= length(MLAT)) ? !Bool(LAND[i, j]) : false
end
seg_water(a, b) = begin
    midlon = a[3] + (mod(b[3] - a[3] + 180, 360) - 180) / 2
    is_ocean(a[2], a[3]) && is_ocean(b[2], b[3]) &&
        is_ocean((a[2] + b[2]) / 2, mod(midlon + 180, 360) - 180)
end

function wn_color(w_ms)
    kt = w_ms * 1.94384
    kt >= 80 ? colorant"#e01e9d" : kt >= 70 ? colorant"#e2231e" :
    kt >= 60 ? colorant"#f08a1e" : kt >= 50 ? colorant"#e6d82e" :
    kt >= 40 ? colorant"#2fb52f" : kt >= 30 ? colorant"#2a52dd" :
    kt >= 20 ? colorant"#25c8c8" : colorant"#b8b8b8"
end

# strike probability on the 0.5° mask grid (fraction of ALL n members)
function strike_prob(tracks, n_members, hmax)
    hit = Dict{String,BitMatrix}()
    for tr in tracks
        fx = [f for f in tr.fix if f[1] <= hmax]
        isempty(fx) && continue
        m = get!(hit, String(tr.member)) do
            falses(length(MLAT), length(MLON))
        end
        pts = Tuple{Float64,Float64}[]
        for i in 1:length(fx)-1
            a, b = fx[i], fx[i+1]
            dlon = mod(b[3] - a[3] + 180, 360) - 180
            for f in (0.0, 0.25, 0.5, 0.75)
                push!(pts, (a[2] + f * (b[2] - a[2]),
                            mod(a[3] + f * dlon + 180, 360) - 180))
            end
        end
        push!(pts, (fx[end][2], fx[end][3]))
        for (la, lo) in pts
            dla = 200.0 / 111.0
            dlo = 200.0 / max(111.0 * cosd(la), 20.0)
            i1 = max(1, floor(Int, (la - dla - MLAT[1]) / 0.5) + 1)
            i2 = min(length(MLAT), ceil(Int, (la + dla - MLAT[1]) / 0.5) + 1)
            for i in i1:i2, j in 1:length(MLON)
                dl = abs(mod(MLON[j] - lo + 180, 360) - 180)
                dl > dlo && continue
                # haversine
                dphi = deg2rad(MLAT[i] - la); dlam = deg2rad(dl)
                aa = sin(dphi/2)^2 + cosd(la) * cosd(MLAT[i]) * sin(dlam/2)^2
                if 2 * 6371.0 * asin(min(sqrt(aa), 1.0)) <= 200.0
                    m[i, j] = true
                end
            end
        end
    end
    prob = zeros(length(MLAT), length(MLON))
    for m in values(hit)
        prob .+= m
    end
    prob ./ max(n_members, 1)
end

# seam-safe coastline segments shifted to a central longitude
function shifted_coastlines(clon)
    xs = Float64[]; ys = Float64[]
    for geom in GeoMakie.coastlines()
        pts = GeoMakie.GeoInterface.coordinates(geom)
        for p in pts
            lo = mod(p[1] - clon + 180, 360) - 180
            if !isempty(xs) && abs(lo - xs[end]) > 180
                push!(xs, NaN); push!(ys, NaN)
            end
            push!(xs, lo); push!(ys, p[2])
        end
        push!(xs, NaN); push!(ys, NaN)
    end
    xs, ys
end

const YLORRD = cgrad(ColorSchemes.YlOrRd.colors)

function render_region_model(reg, mm)
    rid = String(mm.rid)
    outdir = joinpath(String(spec.out_dir), rid)
    mkpath(outdir)
    w, e, s, n = reg.w, reg.e, reg.s, reg.n
    clon = reg.key == "globe" ? -160.0 : 0.0
    # basins live inside ±180 — only the Pacific-centred globe re-maps lons
    shift(lo) = reg.key == "globe" ? mod(lo - clon + 180, 360) - 180 : Float64(lo)
    wsh, esh = Float64(w), Float64(e)
    if reg.key == "globe"
        wsh, esh = -180.0, 180.0
    end
    # mask-grid columns for this window (globe: circshift to clon)
    jshift = round(Int, clon / 0.5)
    landg = circshift(LAND, (0, -jshift))
    cl_x, cl_y = shifted_coastlines(clon)
    mdl = mm.model == "aifs" ? "AIFS-ENS" : "IFS-ENS"
    for k in 0:NDAYS-1
        prob = strike_prob(mm.tracks, mm.n, k * 24)
        probm = [((prob[i, j] > 0.05) && !Bool(LAND[i, j])) ? prob[i, j] : NaN
                 for i in 1:length(MLAT), j in 1:length(MLON)]
        probsh = circshift(probm, (0, -jshift))
        valid = init + Day(k)
        fig = Figure(size = (round(Int, reg.figw * 100), round(Int, reg.figh * 100)),
                     backgroundcolor = :white, fontsize = 13)
        ax = Axis(fig[1, 1];
            limits = ((wsh, esh), (Float64(s), Float64(n))),
            title = "$(reg.label) — TC-strength lows · $mdl ($(mm.n) members) · " *
                    "init $(Dates.format(init, "yyyy-mm-dd")) 00Z\n" *
                    "day $k (valid $(Dates.format(valid, "e u d"))) · shading = strike " *
                    "probability through day $k · ● = day-$k member positions",
            titlesize = 14, titlefont = :bold,
            xgridstyle = :dash, ygridstyle = :dash,
            xgridcolor = (:steelblue, 0.35), ygridcolor = (:steelblue, 0.35),
            backgroundcolor = "#d5ecf5",
            xtickformat = vs -> ["$(round(Int, mod(v + clon + 180, 360) - 180))°" for v in vs],
            ytickformat = vs -> ["$(round(Int, v))°" for v in vs])
        # land: mask image (0.5° — indistinguishable at these scales); the
        # circshift means column j of the shifted data sits at shifted lon MLON[j]
        heatmap!(ax, MLON, MLAT,
                 [Bool(landg[i, j]) ? 1.0 : NaN for j in 1:length(MLON), i in 1:length(MLAT)];
                 colormap = [colorant"#e8dcb8", colorant"#e8dcb8"], colorrange = (0, 1))
        lines!(ax, cl_x, cl_y; color = "#8a8a7a", linewidth = 0.8)
        # strike probability
        heatmap!(ax, MLON, MLAT,
                 permutedims(probsh); colormap = YLORRD, colorrange = (0.0, 0.9),
                 alpha = 0.75)
        # trails + day-k dots
        for tr in mm.tracks
            fx = [f for f in tr.fix if f[1] <= k * 24]
            length(fx) < 2 && continue
            xs = Float64[]; ys = Float64[]
            for i in 1:length(fx)-1
                if seg_water(fx[i], fx[i+1])
                    push!(xs, shift(fx[i][3]));   push!(ys, fx[i][2])
                    push!(xs, shift(fx[i+1][3])); push!(ys, fx[i+1][2])
                    push!(xs, NaN); push!(ys, NaN)
                end
            end
            !isempty(xs) && lines!(ax, xs, ys; color = ("#3a6ea8", 0.3), linewidth = 0.6)
            for f in fx
                if f[1] == k * 24 && is_ocean(f[2], f[3])
                    scatter!(ax, [shift(f[3])], [f[2]]; markersize = reg.key == "globe" ? 5 : 7,
                             color = wn_color(f[6]), strokecolor = :black, strokewidth = 0.4)
                end
            end
        end
        # genesis markers + labels (per-model support, born storms only)
        placed = Tuple{Float64,Float64}[]
        sy, sx = reg.key == "globe" ? (4.5, 16.0) : (3.2, 8.0)
        for cl in mm.clusters
            cl.g[1] > k * 24 && continue
            gx, gy = shift(cl.g[3]), cl.g[2]
            scatter!(ax, [gx], [gy]; marker = :diamond,
                     markersize = reg.key == "globe" ? 7 : 10,
                     color = "#222222", strokecolor = :white, strokewidth = 0.7)
            (w + 2 <= cl.g[3] <= e - 2 && s + 1.5 <= gy <= n - 1.5) || continue
            any(abs(gy - p[1]) < sy && abs(gx - p[2]) < sx for p in placed) && continue
            push!(placed, (gy, gx))
            dy = gy > n - 4 ? -2.4 : 1.6
            text!(ax, gx, gy + dy; text = "$(cl.id)  $(round(Int, cl.sup * 100))%",
                  fontsize = reg.key == "globe" ? 10 : 13, font = :bold,
                  color = "#1a1a2e", align = (:center, :center))
        end
        # wind-bin legend
        bins = [("<20", "#b8b8b8"), ("20+", "#25c8c8"), ("30+", "#2a52dd"),
                ("40+", "#2fb52f"), ("50+", "#e6d82e"), ("60+", "#f08a1e"),
                ("70+", "#e2231e"), ("80+ kt", "#e01e9d")]
        elems = [MarkerElement(marker = :circle, markersize = 9, color = c,
                               strokecolor = :black, strokewidth = 0.4) for (_, c) in bins]
        push!(elems, MarkerElement(marker = :diamond, markersize = 9, color = "#222222"))
        axislegend(ax, elems, [first.(bins); "genesis"], "10 m wind (kt)";
                   position = :rt, nbanks = 3, labelsize = 9, titlesize = 10,
                   framevisible = true, backgroundcolor = (:white, 0.85))
        Colorbar(fig[2, 1], colormap = YLORRD, colorrange = (0, 0.9),
                 vertical = false, flipaxis = false, height = 10,
                 label = "fraction of all $(mm.n) members with a track within 200 km",
                 labelsize = 11, ticklabelsize = 9)
        rowsize!(fig.layout, 2, Auto(0.06))
        save(joinpath(outdir, "F" * lpad(k, 2, '0') * ".png"), fig)
    end
    println("  $rid: $(NDAYS) frames")
end

for reg in spec.regions, mm in reg.models
    render_region_model(reg, mm)
end
println("JULIA RENDER DONE in $(round(time() - t_start, digits=1)) s")
