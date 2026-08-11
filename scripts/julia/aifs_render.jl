# AIFS single vs AIFS-ENS control — frame rasterizer.
# Specs + warped regular-grid fields staged by scripts/verify/aifs_compare_anim.py
# (fields nearest-neighbour-warped onto projected grids in Python; Natural Earth
# polylines pre-projected). Style-parity port of the matplotlib renderer, ~10x
# faster; run several instances with (part, nparts) striding for parallelism.
#
#   julia --project=. aifs_render.jl spec.json [part nparts]
using CairoMakie
using JSON3
using NPZ
using Colors

# ---- banded fill: value field -> band-index image with a categorical cmap ----
function band_index(vals, bounds; over::Bool = false)
    nb = length(bounds) - 1
    idx = fill(NaN32, size(vals))
    @inbounds for i in eachindex(vals)
        v = vals[i]
        isnan(v) && continue
        if v < bounds[1]
            idx[i] = NaN32
        elseif v >= bounds[end]
            idx[i] = over ? Float32(nb) : Float32(nb - 1)
        else
            idx[i] = Float32(searchsortedlast(bounds, v) - 1)
        end
    end
    return idx
end

function band_image!(ax, xlim, ylim, vals, cols; bounds, over = nothing)
    allc = copy(cols)
    over !== nothing && push!(allc, parse(Colorant, over))
    idx = band_index(vals, bounds; over = over !== nothing)
    image!(ax, (xlim[1], xlim[2]), (ylim[1], ylim[2]), idx;
           colormap = cgrad(allc; categorical = true),
           colorrange = (-0.5, length(allc) - 0.5),
           nan_color = :transparent, interpolate = false)
end

# equal-width categorical colorbar with the band edges as tick labels
function band_colorbar!(pos, cols, bounds; label, over = nothing)
    allc = [parse(Colorant, c) for c in cols]
    over !== nothing && push!(allc, parse(Colorant, over))
    n = length(allc)
    ticks = (collect(0.0:(length(bounds) - 1)),
             [b == floor(b) ? string(Int(b)) : string(b) for b in Float64.(bounds)])
    Colorbar(pos; colormap = cgrad(allc; categorical = true),
             limits = (0.0, Float64(n)), ticks = ticks, vertical = false,
             flipaxis = false, label = label, labelsize = 12,
             ticklabelsize = 10, height = 13)
end

function draw_overlays!(ax, ov)
    haskey(ov, "grid_x") &&
        lines!(ax, ov["grid_x"], ov["grid_y"], color = (:gray55, 0.9), linewidth = 0.4)
    haskey(ov, "lakes_x") &&
        lines!(ax, ov["lakes_x"], ov["lakes_y"], color = "#4d4d4d", linewidth = 0.6)
    haskey(ov, "states_x") &&
        lines!(ax, ov["states_x"], ov["states_y"], color = "#4d4d4d", linewidth = 0.6)
    haskey(ov, "borders_x") &&
        lines!(ax, ov["borders_x"], ov["borders_y"], color = "#262626", linewidth = 0.85)
    lines!(ax, ov["coast_x"], ov["coast_y"], color = "#0d0d0d", linewidth = 1.15)
end

# Makie's label pipeline crashes on a level set that yields no contour lines,
# so only levels the field actually crosses may be drawn with labels=true.
function crossed(z, lv)
    lo = false; hi = false
    @inbounds for v in z
        isnan(v) && continue
        v < lv ? (lo = true) : (hi = true)
        lo && hi && return true
    end
    return false
end

function labeled_contour!(ax, xs, ys, z; levels, color, linewidth,
                          linestyle = :solid, labels = Float64[], labelsize = 8)
    lvs = [Float64(l) for l in levels if crossed(z, Float64(l))]
    isempty(lvs) && return
    labs = [Float64(l) for l in labels if l in lvs]
    plain = setdiff(lvs, labs)
    isempty(plain) || contour!(ax, xs, ys, z; levels = plain, color = color,
                               linewidth = linewidth, linestyle = linestyle,
                               labels = false)
    isempty(labs) || contour!(ax, xs, ys, z; levels = labs, color = color,
                              linewidth = linewidth, linestyle = linestyle,
                              labels = true, labelsize = labelsize,
                              labelcolor = color,
                              labelformatter = v -> string(round(Int, v)))
end

function two_panel_figure(loop, suptitle)
    x0, x1 = Float64.(loop.xlim); y0, y1 = Float64.(loop.ylim)
    asp = (x1 - x0) / (y1 - y0)
    H = 520.0
    W = 2 * asp * H + 30
    fig = Figure(size = (round(Int, W), round(Int, H + 96)), backgroundcolor = :white)
    Label(fig[0, 1:2], suptitle, fontsize = 15, font = :bold, padding = (0, 0, 2, 2))
    axes = [Axis(fig[1, j], title = t, titlealign = :left, titlesize = 14,
                 titlefont = :bold, backgroundcolor = :white)
            for (j, t) in ((1, "AIFS single"), (2, "AIFS-ENS control"))]
    for ax in axes
        hidedecorations!(ax)
        limits!(ax, x0, x1, y0, y1)
        ax.aspect = DataAspect()
    end
    rowgap!(fig.layout, 6); colgap!(fig.layout, 10)
    return fig, axes
end

function render_compare(loop, fr, ov, staging)
    z = npzread(joinpath(staging, String(fr.npz)))
    fig, axes = two_panel_figure(loop, String(fr.title))
    xs = range(Float64(loop.xlim[1]), Float64(loop.xlim[2]), size(z["s_msl"], 2))
    ys = range(Float64(loop.ylim[1]), Float64(loop.ylim[2]), size(z["s_msl"], 1))
    cols = [parse(Colorant, c) for c in String.(loop.p_colors)]
    bounds = Float64.(loop.p_levels)
    for (ax, pre) in zip(axes, ("s_", "e_"))
        pr, msl, thk = z[pre * "pr"], z[pre * "msl"], z[pre * "thk"]
        band_image!(ax, loop.xlim, loop.ylim, permutedims(pr), cols;
                    bounds = bounds, over = "#8e1616")
        t = permutedims(thk)
        labeled_contour!(ax, xs, ys, t; levels = 410:6:534, color = "#1565c0",
                         linewidth = 0.7, linestyle = :dash, labels = 414:12:534)
        labeled_contour!(ax, xs, ys, t; levels = 546:6:618, color = "#c62828",
                         linewidth = 0.7, linestyle = :dash, labels = 546:12:618)
        labeled_contour!(ax, xs, ys, t; levels = [540.0], color = "#1565c0",
                         linewidth = 1.7, linestyle = :dash, labels = [540.0],
                         labelsize = 9)
        labeled_contour!(ax, xs, ys, permutedims(msl); levels = 940:4:1060,
                         color = :black, linewidth = 0.8, labels = 944:8:1060)
        draw_overlays!(ax, ov)
    end
    band_colorbar!(fig[2, 1:2], String.(loop.p_colors), bounds;
                   label = "6-h precipitation (mm)", over = "#8e1616")
    save(joinpath(staging, String(fr.id) * ".png"), fig)
end

function render_z500(loop, fr, ov, staging)
    z = npzread(joinpath(staging, String(fr.npz)))
    fig, axes = two_panel_figure(loop, String(fr.title))
    xs = range(Float64(loop.xlim[1]), Float64(loop.xlim[2]), size(z["s_z5"], 2))
    ys = range(Float64(loop.ylim[1]), Float64(loop.ylim[2]), size(z["s_z5"], 1))
    bounds = collect(486.0:6.0:600.0)
    nb = length(bounds) - 1
    cols = [RGB(get(cgrad(:turbo), (k - 0.5) / nb)) for k in 1:nb]
    for (ax, pre) in zip(axes, ("s_", "e_"))
        z5 = permutedims(z[pre * "z5"])
        band_image!(ax, loop.xlim, loop.ylim, z5, cols;
                    bounds = bounds, over = "#7a0403")
        labeled_contour!(ax, xs, ys, z5; levels = bounds[1:2:end], color = :black,
                         linewidth = 0.55, labels = bounds[1:4:end], labelsize = 7)
        draw_overlays!(ax, ov)
    end
    band_colorbar!(fig[2, 1:2], ["#" * hex(c) for c in cols], bounds;
                   label = "500 hPa height (dam)", over = "#7a0403")
    save(joinpath(staging, String(fr.id) * ".png"), fig)
end

function main()
    t0 = time()
    spec = JSON3.read(read(ARGS[1], String))
    part = length(ARGS) >= 3 ? parse(Int, ARGS[2]) : 1
    nparts = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1
    staging = String(spec.staging)
    work = [(loop, fr) for loop in spec.loops for fr in loop.frames]
    mine = work[part:nparts:end]
    ovs = Dict{String, Any}()
    n = 0
    for (loop, fr) in mine
        ov = get!(ovs, String(loop.overlays)) do
            npzread(joinpath(staging, String(loop.overlays)))
        end
        try
            if loop.kind == "compare"
                render_compare(loop, fr, ov, staging)
            else
                render_z500(loop, fr, ov, staging)
            end
            n += 1
        catch e
            println("FRAME FAILED: ", fr.id, " : ",
                    sprint(showerror, e)[1:min(end, 300)])
        end
    end
    println("part $part/$nparts: $n/$(length(mine)) frames in ",
            round(time() - t0, digits = 1), "s")
end

main()
