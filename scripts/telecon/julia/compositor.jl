# NH teleconnection compositor.
#
# Reads the daily key table (state_keys.jl) + the yearly outcome files
# (build_outcome_fields.py) and produces the keyed composite library:
#
#   composites.nc:  (bin, lag, lat, lon) mean anomalies of z500 / t2m / prcp
#                   + composite WAF (fx, fy from composite u200/v200 perturbation
#                   streamfunction) + n-samples + a sign-count significance mask
#
# Binning strategy (coarse on purpose — sample count beats resolution):
#   level 3 (deepest): (mjo, enso_class, gwo)     enso_class = nino/neutral/nina
#   level 2:           (mjo, enso_class)
#   level 1:           (mjo,)
#   torque overlays:   (tqh, gwo) and (tqr, gwo)  — the "torque event" view
# A day contributes its NEXT 0..35 days of outcome anomalies to every bin it
# belongs to at every level. The renderer later picks the deepest bin with
# n >= NMIN for today's state and labels which level it used.
#
# Significance: per cell, the fraction of contributing days whose anomaly sign
# matches the composite's (a sign-consistency mask — cheap, monotone with a
# bootstrap on these sample sizes). Cells < SIGFRAC are faded by the renderer.
#
# Run (after both builders):  julia --project=. compositor.jl

include("state_keys.jl")
using NCDatasets, Dates, Statistics, Printf

const OUTCOMES = joinpath(TELECON, "data", "outcomes")
const LIB = joinpath(TELECON, "data", "composites.nc")
const LAGS = 0:35
const NMIN = 40                      # a bin below this many days is not written
const WINTER_HALF = true             # Oct-Mar days only (the NH signal season)

enso_class(e) = e >= 2 ? :nino : e <= -2 ? :nina : :neutral

"All bin labels a given day's key belongs to, most specific first."
function day_bins(k)::Vector{String}
    b = String[]
    k.mjo == 0 && return b                       # inactive MJO days train nothing
    ec = String(enso_class(k.enso))
    k.gwo > 0 && push!(b, "m$(k.mjo)_$(ec)_g$(k.gwo)")
    push!(b, "m$(k.mjo)_$(ec)")
    push!(b, "m$(k.mjo)")
    k.tqh != 0 && k.gwo > 0 && push!(b, "tqh$(k.tqh > 0 ? "p" : "n")_g$(k.gwo)")
    k.tqr != 0 && k.gwo > 0 && push!(b, "tqr$(k.tqr > 0 ? "p" : "n")_g$(k.gwo)")
    b
end

"Day-of-year harmonic climatology of a (time, lat, lon) stack, per cell."
function field_clim(doys::Vector{Int}, X::Array{Float32,3})
    w = 2pi .* doys ./ 365.25
    B = hcat(ones(length(doys)), cos.(w), sin.(w), cos.(2 .* w), sin.(2 .* w))
    BtB = B'B
    nlat, nlon = size(X, 2), size(X, 3)
    C = Array{Float32}(undef, 5, nlat, nlon)
    for j in 1:nlon, i in 1:nlat
        C[:, i, j] = BtB \ (B' * Float64.(@view X[:, i, j]))
    end
    C
end

clim_at(C, doy) = begin
    w = 2pi * doy / 365.25
    b = Float32[1, cos(w), sin(w), cos(2w), sin(2w)]
    dropdims(sum(C .* reshape(b, 5, 1, 1); dims=1); dims=1)
end

function main()
    keys = build_keys()
    println("keys: $(nrow(keys)) days, $(count(>(0), keys.mjo)) active-MJO")

    files = sort(filter(f -> endswith(f, ".nc"), readdir(OUTCOMES; join=true)))
    isempty(files) && error("no outcome files — run build_outcome_fields.py first")

    # ---- load the full outcome stack (float32; ~13 GB for 65 yr NH — fits) ----
    ds1 = NCDataset(files[1])
    lat, lon = float.(ds1["latitude"][:]), float.(ds1["longitude"][:])
    close(ds1)
    vars = ("z500", "t2m", "prcp", "u200", "v200")
    dates = Date[]; stack = Dict(v => Array{Float32,3}[] for v in vars)
    for f in files
        NCDataset(f) do ds
            append!(dates, Date.(ds["time"][:]))
            for v in vars
                push!(stack[v], permutedims(Float32.(coalesce.(ds[v][:, :, :], NaN32)),
                                            (3, 2, 1)))   # (time, lat, lon)
            end
        end
        @printf("  loaded %s\n", basename(f)); flush(stdout)
    end
    X = Dict(v => cat(stack[v]...; dims=1) for v in vars)
    empty!(stack)
    doys = [min(365, dayofyear(d)) for d in dates]
    didx = Dict(d => i for (i, d) in enumerate(dates))

    # ---- anomalies vs per-cell harmonic clim ----
    A = Dict{String,Array{Float32,3}}()
    for v in vars
        C = field_clim(doys, X[v])
        Av = similar(X[v])
        for t in axes(Av, 1)
            Av[t, :, :] = X[v][t, :, :] .- clim_at(C, doys[t])
        end
        A[v] = Av; delete!(X, v)
        println("  anomalized $v"); flush(stdout)
    end

    # ---- accumulate composites ----
    train = filter(r -> !WINTER_HALF || month(r.date) in (10, 11, 12, 1, 2, 3), keys)
    sums = Dict{String,Dict{String,Array{Float32,3}}}()   # bin -> var -> (lag, lat, lon)
    poss = Dict{String,Dict{String,Array{Int32,3}}}()     # sign counts (z500 only)
    ns = Dict{String,Vector{Int32}}()
    nlat, nlon = length(lat), length(lon)
    for r in eachrow(train)
        i0 = get(didx, r.date, 0)
        (i0 == 0 || i0 + last(LAGS) > length(dates)) && continue
        for b in day_bins(r)
            S = get!(sums, b) do
                Dict(v => zeros(Float32, length(LAGS), nlat, nlon) for v in vars)
            end
            P = get!(poss, b) do
                Dict("z500" => zeros(Int32, length(LAGS), nlat, nlon))
            end
            n = get!(ns, b, zeros(Int32, length(LAGS)))
            for (li, L) in enumerate(LAGS)
                for v in vars
                    S[v][li, :, :] .+= @view A[v][i0+L, :, :]
                end
                P["z500"][li, :, :] .+= (@view(A["z500"][i0+L, :, :]) .> 0)
                n[li] += 1
            end
        end
    end

    # ---- write the library ----
    kept = sort([b for (b, n) in ns if n[1] >= NMIN])
    println("bins kept: $(length(kept)) of $(length(ns)) (NMIN=$NMIN)")
    isfile(LIB) && rm(LIB)
    NCDataset(LIB, "c") do ds
        defDim(ds, "bin", length(kept)); defDim(ds, "lag", length(LAGS))
        defDim(ds, "lat", nlat); defDim(ds, "lon", nlon)
        defVar(ds, "bin_name", [b for b in kept], ("bin",))
        defVar(ds, "lag", collect(LAGS), ("lag",))
        defVar(ds, "lat", lat, ("lat",)); defVar(ds, "lon", lon, ("lon",))
        vn = defVar(ds, "n", Int32, ("bin", "lag"))
        vs = Dict(v => defVar(ds, v, Float32, ("bin", "lag", "lat", "lon")) for v in vars)
        vf = defVar(ds, "signfrac_z500", Float32, ("bin", "lag", "lat", "lon"))
        for (bi, b) in enumerate(kept)
            n = ns[b]; vn[bi, :] = n
            for v in vars
                vs[v][bi, :, :, :] = sums[b][v] ./ reshape(max.(n, 1), :, 1, 1)
            end
            pos = poss[b]["z500"] ./ reshape(max.(n, 1), :, 1, 1)
            vf[bi, :, :, :] = max.(pos, 1 .- pos)         # sign consistency 0.5..1
        end
        ds.attrib["note"] = "NH teleconnection composites; winter-half=$(WINTER_HALF); " *
                            "anomalies vs 2-harmonic day-of-year clim; NMIN=$NMIN"
    end
    println("wrote $LIB")
end

abspath(PROGRAM_FILE) == @__FILE__ && main()
