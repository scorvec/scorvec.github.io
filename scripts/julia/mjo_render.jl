# Generic MJO-suite frame renderer (walker/jets/WAF/MMSF cross-sections and
# maps). Spec-driven: python serializes per-frame layers (filled contours,
# line contours, arrows, text labels) + axis config; this rasterizes them.
# Pattern and JSON conventions follow tc_render.jl / synoptic_render.jl.
#
# spec.json: { staging, frames: [ { frame_id, out_png, figsize: [w,h],
#   title, footer, xlabel, ylabel, ylog: bool, yreversed: bool,
#   xlim: [a,b], ylim: [a,b], xticks: [...], xticklabels: [...],
#   yticks: [...],
#   fill: { npz: key, x: key, y: key, levels: [...], colors: [hex...],
#           cbar_label },
#   contours: [ { npz: key, levels: [...], color, width, dash: bool } ],
#   arrows: { x: key, y: key, u: key, v: key, scale },     # optional
#   texts: [ { x, y, s, size, color } ] } ] }
# Array data per frame in <frame_id>.npz keyed as referenced above.
using CairoMakie
using JSON3
using NPZ
using Colors

function render_frame(spec, staging)
    z = npzread(joinpath(staging, String(spec.frame_id) * ".npz"))
    w, h = spec.figsize
    fig = Figure(size = (round(Int, w * 100), round(Int, h * 100)),
                 backgroundcolor = :white)
    ax = Axis(fig[1, 1],
              title = String(spec.title), titlealign = :left, titlesize = 14,
              xlabel = String(spec.xlabel), ylabel = String(spec.ylabel),
              yscale = spec.ylog ? log10 : identity,
              xticklabelsize = 11, yticklabelsize = 11)
    if spec.xticks !== nothing
        ax.xticks = (Float64.(spec.xticks), String.(spec.xticklabels))
    end
    if spec.yticks !== nothing
        yt = Float64.(spec.yticks)
        ax.yticks = (yt, string.(round.(Int, yt)))
    end

    f = spec.fill
    X = Float64.(z[String(f.x)]); Y = Float64.(z[String(f.y)])
    Z = Float64.(z[String(f.npz)])
    levels = Float64.(f.levels)
    cols = [parse(Colorant, c) for c in f.colors]
    cf = contourf!(ax, X, Y, permutedims(Z); levels = levels,
                   colormap = cgrad(cols; categorical = true),
                   extendlow = :auto, extendhigh = :auto)

    for c in spec.contours
        Zc = Float64.(z[String(c.npz)])
        contour!(ax, X, Y, permutedims(Zc);
                 levels = Float64.(c.levels),
                 color = parse(Colorant, String(c.color)),
                 linewidth = Float64(c.width),
                 linestyle = Bool(c.dash) ? :dash : :solid)
    end

    if spec.arrows !== nothing
        a = spec.arrows
        ax_, ay_ = Float64.(z[String(a.x)]), Float64.(z[String(a.y)])
        au, av = Float64.(z[String(a.u)]), Float64.(z[String(a.v)])
        arrows!(ax, ax_, ay_, au .* Float64(a.scale), av .* Float64(a.scale);
                color = (:black, 0.75), arrowsize = 6, linewidth = 0.8)
    end

    for t in spec.texts
        text!(ax, Float64(t.x), Float64(t.y); text = String(t.s),
              fontsize = Float64(t.size), color = String(t.color),
              align = (:center, :top))
    end

    xlims!(ax, Float64.(spec.xlim)...)
    ylims!(ax, Float64.(spec.ylim)...)   # pass (1000,100) for a reversed p-axis
    Colorbar(fig[1, 2], cf, label = String(f.cbar_label), labelsize = 11,
             ticklabelsize = 10)
    Label(fig[2, 1:2], String(spec.footer), fontsize = 9, color = :gray40,
          tellwidth = false)
    rowgap!(fig.layout, 4); colgap!(fig.layout, 6)
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
            println("FRAME FAILED: ", fr.frame_id, " : ",
                    sprint(showerror, e)[1:min(end, 250)])
        end
    end
    println("rendered $n/$(length(spec.frames)) frames")
    println("JULIA RENDER DONE in $(round(time() - t0, digits = 1))s")
end

main()
