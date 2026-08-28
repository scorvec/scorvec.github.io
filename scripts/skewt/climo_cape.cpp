// Emit "MM ecape ship" per sounding for one IGRA period-of-record file, using
// the SAME compute_sounding() as the WASM build so climo matches the live app.
//   climo_cape GID < station.txt   (or)   climo_cape GID station.txt
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <vector>

extern "C" int compute_sounding(const float* pres, const float* hght,
                                const float* tmpk, const float* dwpk,
                                const float* uwin, const float* vwin, int N,
                                float* out, float* a, float* b, float* c);

static float fld(const std::string& L, int a, int b) {
    if ((int)L.size() < b) return NAN;
    std::string s = L.substr(a, b - a);
    try { int v = std::stoi(s); return v <= -8888 ? NAN : (float)v; }
    catch (...) { return NAN; }
}

int main(int argc, char** argv) {
    if (argc < 2) return 1;
    std::string gid = argv[1];
    // IGRA frequently reports the surface level with NO geopotential height
    // (-9999) — 1137 of 1144 July soundings at Santa Maria, Brazil. The surface
    // height is simply the station elevation, so take it as an argument rather
    // than discarding 99% of a station's record.
    const float elev = (argc >= 3) ? std::atof(argv[2]) : 0.0f;
    std::istream* in = &std::cin;
    std::ifstream f;
    if (argc >= 4) { f.open(argv[3]); in = &f; }

    std::string line;
    std::vector<std::string> block;
    int year = 0, month = 0, day = 0, nlev = 0;
    auto flush = [&]() {
        if (block.empty() || month < 1) return;
        std::vector<float> P, H, T, D, U, V;
        for (auto& L : block) {
            if ((int)L.size() < 52) continue;
            float p = fld(L, 9, 15), gph = fld(L, 16, 21), tt = fld(L, 22, 27),
                  dpdp = fld(L, 34, 39), wd = fld(L, 40, 45), ws = fld(L, 46, 51),
                  rh = fld(L, 28, 33);   // pre-~1990 US: RH instead of DPDP
            if (std::isnan(p) || p < 2000) continue;
            if (std::isnan(tt)) continue;                 // need thermo for CAPE
            if (!P.empty() && p >= P.back()) continue;    // monotonic pressure
            P.push_back(p);
            H.push_back(std::isnan(gph) ? NAN : gph);
            T.push_back(tt / 10 + 273.15f);
            float Dv = NAN;
            if (!std::isnan(dpdp)) Dv = tt / 10 - dpdp / 10 + 273.15f;
            else if (!std::isnan(rh) && rh > 0) {    // Td from RH (inverse Magnus)
                const float tc = tt / 10.0f;
                const float g = std::log(std::min(100.0f, rh / 10.0f) / 100.0f)
                              + 17.67f * tc / (tc + 243.5f);
                Dv = 243.5f * g / (17.67f - g) + 273.15f;
            }
            D.push_back(Dv);
            float u = NAN, v = NAN;
            if (!std::isnan(wd) && !std::isnan(ws)) {
                u = -(ws / 10) * std::sin(wd * M_PI / 180);
                v = -(ws / 10) * std::cos(wd * M_PI / 180);
            }
            U.push_back(u); V.push_back(v);
        }
        int N = P.size();
        if (N < 10) return;

        // --- quality gate: never fabricate data ------------------------------
        // Many stations (much of South America) report dewpoint on only a
        // fraction of levels. The old code invented D = T-30 and heights in
        // arbitrary +100 m steps, which produced garbage CAPE. Instead we
        // require real moisture through the lower/mid troposphere and
        // interpolate only across genuine gaps.
        int ndew = 0, ndew_low = 0, nlow = 0;
        for (int i = 0; i < N; ++i) {
            if (!std::isnan(D[i])) ++ndew;
            if (P[i] >= 40000.0f) {                  // at/below 400 hPa
                ++nlow;
                if (!std::isnan(D[i])) ++ndew_low;
            }
        }
        const float ptop = P[N - 1];
        if (std::isnan(D[0]) || std::isnan(T[0])) return;      // need a real sfc
        if (ptop > 40000.0f) return;                            // must reach 400 hPa
        if (ndew < 6 || nlow < 4 || ndew_low < 0.6f * nlow) return;  // real moisture

        // interpolate dewpoint across gaps in log-p (never invent T-30)
        for (int i = 0; i < N; ++i) {
            if (!std::isnan(D[i])) continue;
            int lo = -1, hi = -1;
            for (int j = i - 1; j >= 0; --j) if (!std::isnan(D[j])) { lo = j; break; }
            for (int j = i + 1; j < N; ++j)  if (!std::isnan(D[j])) { hi = j; break; }
            if (lo >= 0 && hi >= 0) {
                const float f = std::log(P[lo] / P[i]) / std::log(P[lo] / P[hi]);
                D[i] = D[lo] + f * (D[hi] - D[lo]);
            } else if (lo >= 0) {
                D[i] = std::min(D[lo], T[i] - 2.0f);   // dry aloft, but physical
            } else return;
        }
        // surface height = station elevation when IGRA omits it
        if (std::isnan(H[0])) H[0] = elev;
        // heights: hypsometric from P and T (not arbitrary 100 m steps)
        for (int i = 1; i < N; ++i) {
            if (!std::isnan(H[i])) continue;
            const float tbar = 0.5f * (T[i] + T[i - 1]);
            H[i] = H[i - 1] + 287.05f * tbar / 9.80665f * std::log(P[i - 1] / P[i]);
        }
        // winds: interpolate across gaps (carry at the ends), never zero-fill
        {
            int first = -1, last = -1;
            for (int i = 0; i < N; ++i) if (!std::isnan(U[i])) { if (first < 0) first = i; last = i; }
            if (first < 0) { for (int i = 0; i < N; ++i) { U[i] = 0; V[i] = 0; } }
            else {
                for (int i = 0; i < N; ++i) {
                    if (!std::isnan(U[i])) continue;
                    if (i < first) { U[i] = U[first]; V[i] = V[first]; continue; }
                    if (i > last)  { U[i] = U[last];  V[i] = V[last];  continue; }
                    int lo = -1, hi = -1;
                    for (int j = i - 1; j >= 0; --j) if (!std::isnan(U[j])) { lo = j; break; }
                    for (int j = i + 1; j < N; ++j)  if (!std::isnan(U[j])) { hi = j; break; }
                    const float f = std::log(P[lo] / P[i]) / std::log(P[lo] / P[hi]);
                    U[i] = U[lo] + f * (U[hi] - U[lo]);
                    V[i] = V[lo] + f * (V[hi] - V[lo]);
                }
            }
        }

        std::vector<float> out(64), a(N), b(N), c(N);
        try {
            if (compute_sounding(P.data(), H.data(), T.data(), D.data(),
                                 U.data(), V.data(), N, out.data(),
                                 a.data(), b.data(), c.data()) != 0) return;
        } catch (...) { return; }        // never let one bad sounding kill the run
        // A valid non-convective sounding has ECAPE/SHIP of ZERO, not "missing".
        // Skipping them made the climatology conditional on convective days
        // (n=51 vs 2007 for PWAT) and the percentiles meaningless.
        const float mucape = out[10];
        if (!(mucape > -8888.0f)) return;
        const float ec = (out[42] > -8888.0f) ? out[42] : 0.0f;
        const float sh = (out[43] > -8888.0f) ? out[43] : 0.0f;
        printf("%d %02d %02d %.1f %.3f\n", year, month, day, ec, sh);
    };

    while (std::getline(*in, line)) {
        if (line.size() > gid.size() + 1 && line[0] == '#' &&
            line.compare(1, gid.size(), gid) == 0) {
            flush();
            block.clear();
            try {
                year = std::stoi(line.substr(13, 4));
                month = std::stoi(line.substr(18, 2));
                day = std::stoi(line.substr(21, 2));
                nlev = std::stoi(line.substr(32, 4));
            } catch (...) { year = month = day = 0; }
            (void)nlev;
        } else if (!line.empty() && line[0] != '#') {
            block.push_back(line);
        }
    }
    flush();
    return 0;
}
