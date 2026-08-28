# Synoptic frame rasterizer: curvilinear quad mesh + Natural Earth polylines
# + colorbar + title, from specs staged by scripts/synoptic/julia_bridge.py.
# Style-parity port of render_maps.render_map's matplotlib output.
using CairoMakie
using JSON3
using NPZ
using Colors

function banded_cmap(spec)
    cols = [parse(Colorant, c) for c in spec.cmap.colors]
    bounds = Float64.(spec.cmap.bounds)
    return cols, bounds
end

function render_frame(spec, staging)
    z = npzread(joinpath(staging, String(spec.frame_id) * ".npz"))
    vals = z["values"]
    x0, x1, y0, y1 = Float64.(spec.extent)
    figsize = (round(Int, spec.figsize[1] * 100), round(Int, spec.figsize[2] * 100))
    fig = Figure(size = figsize, backgroundcolor = :white)
    ax = Axis(fig[1, 1], backgroundcolor = :white,
              title = String(spec.title), titlealign = :left, titlesize = 15,
              titlegap = 8)
    hidedecorations!(ax); hidespines!(ax)
    limits!(ax, x0, x1, y0, y1)
    ax.aspect = DataAspect()

    local plotobj
    if spec.cmap.kind == "banded"
        cols, bounds = banded_cmap(spec)
        cm = cgrad(cols; categorical = true)
        # map values to band index for a categorical colormap
        nb = length(bounds) - 1
        idx = fill(NaN32, size(vals))
        @inbounds for i in eachindex(vals)
            v = vals[i]
            isnan(v) && continue
            if v < bounds[1]
                idx[i] = haskey(spec.cmap, :under) ? 0.0f0 : NaN32
            elseif v >= bounds[end]
                idx[i] = haskey(spec.cmap, :over) ? Float32(nb + 1) : Float32(nb)
            else
                k = searchsortedlast(bounds, v)
                idx[i] = Float32(k)
            end
        end
        allc = copy(cols)
        haskey(spec.cmap, :under) && pushfirst!(allc, parse(Colorant, spec.cmap.under))
        haskey(spec.cmap, :over) && push!(allc, parse(Colorant, spec.cmap.over))
        cmx = cgrad(allc; categorical = true)
        lo = haskey(spec.cmap, :under) ? -0.5 : 0.5
        hi = haskey(spec.cmap, :over) ? length(allc) - 0.5 + lo : length(allc) + lo
        plotobj = image!(ax, (x0, x1), (y0, y1), permutedims(idx);
                         colormap = cmx,
                         colorrange = (lo + 0.0, lo + length(allc)),
                         nan_color = :transparent, interpolate = false)
    else
        cols = [parse(Colorant, c) for c in spec.cmap.colors]
        plotobj = image!(ax, (x0, x1), (y0, y1), permutedims(Float32.(vals));
                         colormap = cgrad(cols),
                         colorrange = (spec.cmap.vmin, spec.cmap.vmax),
                         nan_color = :transparent, interpolate = false)
    end

    ov = npzread(String(spec.overlays))
    lines!(ax, ov["states_x"], ov["states_y"], color = "#222222", linewidth = 0.5)
    lines!(ax, ov["borders_x"], ov["borders_y"], color = "#000000", linewidth = 0.7)
    lines!(ax, ov["coast_x"], ov["coast_y"], color = "#000000", linewidth = 0.7)

    if spec.cmap.kind == "banded"
        cols, bounds = banded_cmap(spec)
        cb = Colorbar(fig[1, 2], colormap = cgrad(cols; categorical = true),
                      limits = (bounds[1], bounds[end]),
                      label = String(spec.cbar_label), labelsize = 13,
                      ticks = spec.cbar_ticks === nothing ?
                              Makie.automatic : Float64.(spec.cbar_ticks),
                      ticklabelsize = 11)
    else
        Colorbar(fig[1, 2], plotobj, label = String(spec.cbar_label),
                 labelsize = 13, ticklabelsize = 11)
    end
    colgap!(fig.layout, 6)
    save(joinpath(staging, String(spec.frame_id) * ".png"), fig)
end

function main()
    t0 = time()
    spec = JSON3.read(read(ARGS[1], String))
    staging = String(spec.staging)
    n = 0
    for fr in spec.frames
        try
            render_frame(fr, staging)
            n += 1
        catch e
            println("FRAME FAILED: ", fr.frame_id, " : ", sprint(showerror, e)[1:min(end,200)])
        end
    end
    println("rendered $n/$(length(spec.frames)) frames")
    println("JULIA RENDER DONE in $(round(time() - t0, digits=1))s")
end

main()
