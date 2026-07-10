// SHARPlib → WASM wrapper for the client-side Skew-T explorer.
// Inputs are parallel float arrays (SI units: Pa, m MSL, K, m/s), surface first,
// strictly decreasing pressure. Outputs: a 40-float parameter block plus three
// parcel virtual-temperature traces (SB / 100-hPa ML / MU) for drawing.
//
// Native test:  clang++ -std=c++17 -O2 -DTEST_MAIN -Iinclude skewt_wasm.cpp \
//                 src/SHARPlib/*.cpp src/SHARPlib/params/*.cpp -o /tmp/skewt_test
// WASM build:   see build_wasm.sh
#include <cmath>
#include <cstddef>
#include <vector>

#include <SHARPlib/constants.h>
#include <SHARPlib/interp.h>
#include <SHARPlib/layer.h>
#include <SHARPlib/parcel.h>
#include <SHARPlib/params/convective.h>
#include <SHARPlib/thermo.h>
#include <SHARPlib/winds.h>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#define KEEP EMSCRIPTEN_KEEPALIVE
#else
#define KEEP
#endif

using namespace sharp;

// out[] layout (all floats; MISSING = -9999):
//  0 sb_cape   1 sb_cinh   2 sb_lcl_p   3 sb_lfc_p   4 sb_el_p
//  5 ml_cape   6 ml_cinh   7 ml_lcl_p   8 ml_lfc_p   9 ml_el_p
// 10 mu_cape  11 mu_cinh  12 mu_lcl_p  13 mu_lfc_p  14 mu_el_p
// 15 pwat_mm  16 lr_0_3km 17 lr_3_6km
// 18 shr6_u   19 shr6_v   20 shr1_u    21 shr1_v
// 22 bunkR_u  23 bunkR_v  24 bunkL_u   25 bunkL_v
// 26 srh1_R   27 srh3_R   28 eff_bot_p 29 eff_top_p
// 30 eff_srh_R 31 eff_shear_mag 32 scp 33 stp
// 34 mu_lpl_p 35 sb_lcl_hght_agl

extern "C" {

KEEP int compute_sounding(const float* pres, const float* hght,
                          const float* tmpk, const float* dwpk,
                          const float* uwin, const float* vwin, const int N,
                          float* out, float* sb_vt, float* ml_vt, float* mu_vt) {
    if (N < 5) return 1;
    std::vector<float> mixr(N), vtmp(N), thta(N), buoy(N), scratch(N);
    for (int i = 0; i < N; ++i) {
        mixr[i] = mixratio(pres[i], dwpk[i]);
        vtmp[i] = virtual_temperature(tmpk[i], mixr[i]);
        thta[i] = theta(pres[i], tmpk[i], THETA_REF_PRESSURE);
    }
    lifter_wobus lifter;

    // --- parcels -----------------------------------------------------------
    Parcel sb = Parcel::surface_parcel(pres[0], tmpk[0], dwpk[0]);
    sb.lift_parcel(lifter, pres, sb_vt, N);
    buoyancy(sb_vt, vtmp.data(), buoy.data(), N);
    sb.cape_cinh(pres, hght, buoy.data(), N);

    PressureLayer mix_lyr(pres[0], pres[0] - 10000.0f);
    Parcel ml = Parcel::mixed_layer_parcel(mix_lyr, pres, hght, thta.data(),
                                           mixr.data(), N);
    ml.lift_parcel(lifter, pres, ml_vt, N);
    buoyancy(ml_vt, vtmp.data(), buoy.data(), N);
    ml.cape_cinh(pres, hght, buoy.data(), N);

    // effective inflow layer + its most-unstable parcel (SPC convention)
    Parcel mu;
    PressureLayer eff = effective_inflow_layer(
        lifter, pres, hght, tmpk, dwpk, vtmp.data(), scratch.data(),
        buoy.data(), N, 100.0f, -250.0f, &mu);
    // re-lift the winning MU parcel so its trace is in mu_vt
    mu.lift_parcel(lifter, pres, mu_vt, N);
    buoyancy(mu_vt, vtmp.data(), buoy.data(), N);
    mu.cape_cinh(pres, hght, buoy.data(), N);

    // --- thermo ------------------------------------------------------------
    const float pwat = precipitable_water(PressureLayer(pres[0], pres[N - 1]),
                                          pres, mixr.data(), N);
    const float lr03 = lapse_rate(HeightLayer(0.0f, 3000.0f), hght, tmpk, N);
    const float lr36 = lapse_rate(HeightLayer(3000.0f, 6000.0f), hght, tmpk, N);

    // --- kinematics (MSL height layers; coord = MSL heights) ---------------
    const float sfc = hght[0];
    WindComponents shr6 = wind_shear(HeightLayer(sfc, sfc + 6000.0f), hght,
                                     uwin, vwin, N);
    WindComponents shr1 = wind_shear(HeightLayer(sfc, sfc + 1000.0f), hght,
                                     uwin, vwin, N);
    WindComponents bunkR = storm_motion_bunkers(pres, hght, uwin, vwin, N,
                                                HeightLayer(0.0f, 6000.0f),
                                                HeightLayer(0.0f, 6000.0f),
                                                false, false);
    WindComponents bunkL = storm_motion_bunkers(pres, hght, uwin, vwin, N,
                                                HeightLayer(0.0f, 6000.0f),
                                                HeightLayer(0.0f, 6000.0f),
                                                true, false);
    const float srh1 = helicity(HeightLayer(sfc, sfc + 1000.0f), bunkR, hght,
                                uwin, vwin, N);
    const float srh3 = helicity(HeightLayer(sfc, sfc + 3000.0f), bunkR, hght,
                                uwin, vwin, N);

    float eff_srh = MISSING, eff_shear_mag = MISSING;
    float scp = MISSING, stp = MISSING;
    if (eff.bottom != MISSING && mu.eql_pressure != MISSING) {
        eff_srh = helicity(eff, bunkR, pres, uwin, vwin, N);
        // effective bulk shear: EIL bottom → 50% of the MU parcel EL height
        const float eil_bot_hght = interp_pressure(eff.bottom, pres, hght, N);
        const float el_hght = interp_pressure(mu.eql_pressure, pres, hght, N);
        const float half_hght = eil_bot_hght + 0.5f * (el_hght - eil_bot_hght);
        WindComponents eshr = wind_shear(HeightLayer(eil_bot_hght, half_hght),
                                         hght, uwin, vwin, N);
        if (eshr.u != MISSING) {
            eff_shear_mag = std::sqrt(eshr.u * eshr.u + eshr.v * eshr.v);
            scp = supercell_composite_parameter(mu.cape, eff_srh, eff_shear_mag);
            const float sb_lcl_agl =
                interp_pressure(sb.lcl_pressure, pres, hght, N) - sfc;
            stp = significant_tornado_parameter(ml, sb_lcl_agl, eff_srh,
                                                eff_shear_mag);
        }
    }

    const float sb_lcl_agl = (sb.lcl_pressure != MISSING)
        ? interp_pressure(sb.lcl_pressure, pres, hght, N) - sfc : MISSING;

    const float o[] = {
        sb.cape, sb.cinh, sb.lcl_pressure, sb.lfc_pressure, sb.eql_pressure,
        ml.cape, ml.cinh, ml.lcl_pressure, ml.lfc_pressure, ml.eql_pressure,
        mu.cape, mu.cinh, mu.lcl_pressure, mu.lfc_pressure, mu.eql_pressure,
        pwat, lr03, lr36,
        shr6.u, shr6.v, shr1.u, shr1.v,
        bunkR.u, bunkR.v, bunkL.u, bunkL.v,
        srh1, srh3, eff.bottom, eff.top,
        eff_srh, eff_shear_mag, scp, stp,
        mu.pres, sb_lcl_agl,
    };
    for (size_t i = 0; i < sizeof(o) / sizeof(float); ++i) out[i] = o[i];
    return 0;
}

// moist + dry adiabat tracer for background lines: lift a parcel from
// (start_pres, start_tmpk) and fill tmpk at each pressure in pres[].
KEEP void trace_adiabat(const float start_pres, const float start_tmpk,
                        const float start_dwpk, const float* pres,
                        float* out_tmpk, const int N) {
    lifter_wobus lifter;
    float plcl, tlcl;
    drylift(start_pres, start_tmpk, start_dwpk, plcl, tlcl);
    for (int i = 0; i < N; ++i) {
        if (pres[i] >= plcl) {
            out_tmpk[i] = theta(start_pres, start_tmpk, pres[i]);
        } else {
            out_tmpk[i] = lifter(plcl, tlcl, pres[i]);
        }
    }
}
}  // extern "C"

#ifdef TEST_MAIN
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
int main(int argc, char** argv) {
    // read the UW TEXT:CSV format (header + rows)
    std::ifstream f(argv[1]);
    std::string line;
    std::getline(f, line);  // header
    std::vector<float> P, H, T, D, U, V;
    float last_p = 1e9f;
    while (std::getline(f, line)) {
        std::stringstream ss(line);
        std::string tok;
        std::vector<std::string> c;
        while (std::getline(ss, tok, ',')) c.push_back(tok);
        if (c.size() < 13) continue;
        try {
            float p = std::stof(c[3]) * 100.0f;         // hPa → Pa
            float h = std::stof(c[4]);
            float t = std::stof(c[5]) + 273.15f;        // C → K
            float d = std::stof(c[6]) + 273.15f;
            float wd = std::stof(c[11]), ws = std::stof(c[12]);
            if (p >= last_p || p < 2000.0f) continue;
            last_p = p;
            P.push_back(p); H.push_back(h); T.push_back(t); D.push_back(d);
            U.push_back(-ws * std::sin(wd * 3.14159265f / 180.0f));
            V.push_back(-ws * std::cos(wd * 3.14159265f / 180.0f));
        } catch (...) { continue; }
    }
    const int N = (int)P.size();
    std::vector<float> out(40), sbv(N), mlv(N), muv(N);
    int rc = compute_sounding(P.data(), H.data(), T.data(), D.data(), U.data(),
                              V.data(), N, out.data(), sbv.data(), mlv.data(),
                              muv.data());
    printf("rc=%d N=%d\n", rc, N);
    printf("SB: CAPE %.0f CINH %.0f LCL %.0f hPa LFC %.0f EL %.0f\n",
           out[0], out[1], out[2] / 100, out[3] / 100, out[4] / 100);
    printf("ML: CAPE %.0f CINH %.0f LCL %.0f\n", out[5], out[6], out[7] / 100);
    printf("MU: CAPE %.0f CINH %.0f (LPL %.0f hPa)\n", out[10], out[11], out[34] / 100);
    printf("PWAT %.1f mm  LR 0-3 %.1f  3-6 %.1f K/km\n", out[15], out[16], out[17]);
    printf("shear6 %.1f m/s  SRH1 %.0f SRH3 %.0f  effSRH %.0f effShr %.1f\n",
           std::hypot(out[18], out[19]), out[26], out[27], out[30], out[31]);
    printf("SCP %.1f STP %.1f  bunkersR (%.1f,%.1f)\n", out[32], out[33], out[22], out[23]);
    return 0;
}
#endif
