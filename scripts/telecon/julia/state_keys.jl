# State-key assignment for the NH teleconnection compositor.
#
# Turns the raw daily series (state_series.nc + RMM + ONI/Nino) into one
# categorical key per day. Every key dimension is deliberately COARSE — the
# whole design trades resolution for sample count, and the compositor's
# fallback hierarchy (joint -> pair -> MJO-only) handles the thin bins.
#
#   mjo    :: 0 = inactive (amp < 1), 1..8 = active RMM phase
#   enso   :: -3..3  (strength x sign: 0 neutral, ±1 weak, ±2 moderate, ±3 strong)
#   flavor :: :ep | :cp | :none      (Nino3 vs Nino4 anomaly, active ENSO only)
#   gwo    :: 0 = weak (amp < 1 sd), 1..8 = NH GWO octant
#   tqh/tqr:: -1 | 0 | +1            (Himalaya / Rockies 5-day torque tercile)

using NCDatasets, Dates, Statistics, DataFrames

const TELECON = normpath(joinpath(@__DIR__, ".."))

"Fit mean + annual + semiannual harmonics; return the day-of-year climatology."
function harmonic_clim(doy::Vector{Int}, x::Vector{Float64})
    w = 2pi .* doy ./ 365.25
    B = hcat(ones(length(doy)), cos.(w), sin.(w), cos.(2 .* w), sin.(2 .* w))
    c = B \ x
    d = 2pi .* (1:366) ./ 365.25
    hcat(ones(366), cos.(d), sin.(d), cos.(2 .* d), sin.(2 .* d)) * c
end

anom_vs_clim(doy, x) = x .- harmonic_clim(doy, x)[doy]

"Centred difference tendency (per day), NaN-padded at the ends."
function tendency(x::Vector{Float64}; step::Int=2)
    t = fill(NaN, length(x))
    for i in (step+1):(length(x)-step)
        t[i] = (x[i+step] - x[i-step]) / (2step)
    end
    t
end

"GWO octant from standardized (M', dM/dt): phase 1..8 anticlockwise, 0 if weak."
function gwo_phase(m::Float64, dm::Float64)
    amp = hypot(m, dm)
    amp < 1.0 && return 0
    # octants rotated so phase 1 = low AAM / falling, matching the M vs dM/dt
    # plane convention of the site's aam_phase plot
    a = mod(atan(dm, m) + 2pi, 2pi)
    1 + Int(floor(mod(a + pi/8, 2pi) / (pi/4))) % 8
end

"Load state_series.nc -> DataFrame(date, aam_nh, tq_himalaya, tq_rockies)."
function load_state_series(path = joinpath(TELECON, "data", "state_series.nc"))
    NCDataset(path) do ds
        DataFrame(date = Date.(ds["time"][:]),
                  aam_nh = float.(ds["aam_nh"][:]),
                  tqh = float.(ds["tq_himalaya"][:]),
                  tqr = float.(ds["tq_rockies"][:]))
    end
end

"Load the IRI RMM tsv pair -> DataFrame(date, phase, amp)."
function load_rmm(dir = joinpath(TELECON, "data"))
    jd2date(j) = Date(1858, 11, 17) + Day(floor(Int, j - 2400000.5))
    read2(f) = [(jd2date(parse(Float64, split(l)[1])), parse(Float64, split(l)[2]))
                for l in Iterators.drop(eachline(joinpath(dir, f)), 2)
                if length(split(l)) == 2]
    ph = Dict(read2("rmm_phase.tsv")); am = Dict(read2("rmm_amplitude.tsv"))
    d = sort(collect(keys(ph)))
    DataFrame(date = d, phase = [ph[k] for k in d],
              amp = [get(am, k, NaN) for k in d])
end

"Monthly ENSO keys from ONI (strength) + Nino3/Nino4 anomalies (flavor)."
function load_enso(oni = joinpath(TELECON, "data", "oni.txt"),
                   nino = joinpath(TELECON, "..", "sst", "data"))
    seas = Dict("DJF"=>1,"JFM"=>2,"FMA"=>3,"MAM"=>4,"AMJ"=>5,"MJJ"=>6,
                "JJA"=>7,"JAS"=>8,"ASO"=>9,"SON"=>10,"OND"=>11,"NDJ"=>12)
    out = Dict{Tuple{Int,Int},Float64}()
    for l in Iterators.drop(eachline(oni), 1)
        p = split(l); length(p) == 4 || continue
        out[(parse(Int, p[2]), seas[p[1]])] = parse(Float64, p[4])
    end
    out                                       # (year, centre-month) -> ONI
end

enso_strength(o) = isnan(o) ? 0 : sign(o) * (abs(o) >= 1.5 ? 3 : abs(o) >= 1.0 ? 2 : abs(o) >= 0.5 ? 1 : 0) |> Int

"""
Assemble the daily key table over the state-series span. Torque terciles and
GWO standardization are computed against the full record (day-of-year aware
for torque; all-days sd for the GWO plane, matching common GWO practice).
"""
function build_keys()
    ss = load_state_series()
    doy = [min(365, dayofyear(d)) for d in ss.date]

    m = anom_vs_clim(doy, ss.aam_nh)
    dm = tendency(m)
    ms, dms = std(skipmissing(filter(!isnan, m))), std(filter(!isnan, dm))
    gwo = [isnan(x) || isnan(y) ? 0 : gwo_phase(x / ms, y / dms) for (x, y) in zip(m, dm)]

    key_tq(x, lo, hi) = x <= lo ? -1 : x >= hi ? 1 : 0
    tq5(v) = [mean(v[max(1, i-4):i]) for i in eachindex(v)]      # trailing 5-day mean
    tqh5, tqr5 = tq5(anom_vs_clim(doy, ss.tqh)), tq5(anom_vs_clim(doy, ss.tqr))
    hq = quantile(filter(!isnan, tqh5), [1/3, 2/3])
    rq = quantile(filter(!isnan, tqr5), [1/3, 2/3])

    rmm = leftjoin(ss[:, [:date]], load_rmm(), on = :date)
    oni = load_enso()

    df = DataFrame(date = ss.date,
        mjo = [ismissing(a) || isnan(a) || a < 1 ? 0 : Int(p)
               for (a, p) in zip(rmm.amp, rmm.phase)],
        enso = [enso_strength(get(oni, (year(d), month(d)), NaN)) for d in ss.date],
        gwo = gwo,
        tqh = key_tq.(tqh5, hq[1], hq[2]),
        tqr = key_tq.(tqr5, rq[1], rq[2]),
        aam_anom = m, aam_tend = dm)
    df
end
