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
    std::istream* in = &std::cin;
    std::ifstream f;
    if (argc >= 3) { f.open(argv[2]); in = &f; }

    std::string line;
    std::vector<std::string> block;
    int year = 0, month = 0, nlev = 0;
    auto flush = [&]() {
        if (block.empty() || month < 1) return;
        std::vector<float> P, H, T, D, U, V;
        for (auto& L : block) {
            if ((int)L.size() < 52) continue;
            float p = fld(L, 9, 15), gph = fld(L, 16, 21), tt = fld(L, 22, 27),
                  dpdp = fld(L, 34, 39), wd = fld(L, 40, 45), ws = fld(L, 46, 51);
            if (std::isnan(p) || p < 2000) continue;
            if (std::isnan(tt)) continue;                 // need thermo for CAPE
            P.push_back(p);
            H.push_back(std::isnan(gph) ? NAN : gph);
            T.push_back(tt / 10 + 273.15f);
            D.push_back(std::isnan(dpdp) ? NAN : tt / 10 - dpdp / 10 + 273.15f);
            float u = NAN, v = NAN;
            if (!std::isnan(wd) && !std::isnan(ws)) {
                u = -(ws / 10) * std::sin(wd * M_PI / 180);
                v = -(ws / 10) * std::cos(wd * M_PI / 180);
            }
            U.push_back(u); V.push_back(v);
        }
        int N = P.size();
        // fill any missing heights hypsometrically, winds by carry (compute
        // tolerates; CAPE only needs P/H/T/D)
        for (int i = 0; i < N; ++i) {
            if (std::isnan(H[i])) H[i] = (i ? H[i-1] : 0) + 100;
            if (std::isnan(D[i])) D[i] = T[i] - 30;
            if (std::isnan(U[i])) { U[i] = i ? U[i-1] : 0; V[i] = i ? V[i-1] : 0; }
        }
        if (N >= 10) {
            std::vector<float> out(64), a(N), b(N), c(N);
            if (compute_sounding(P.data(), H.data(), T.data(), D.data(),
                                 U.data(), V.data(), N, out.data(),
                                 a.data(), b.data(), c.data()) == 0) {
                float ec = out[42], sh = out[43];
                if (ec > -8888 || sh > -8888)
                    printf("%d %02d %.1f %.3f\n", year, month,
                           ec > -8888 ? ec : NAN, sh > -8888 ? sh : NAN);
            }
        }
    };

    while (std::getline(*in, line)) {
        if (line.size() > gid.size() + 1 && line[0] == '#' &&
            line.compare(1, gid.size(), gid) == 0) {
            flush();
            block.clear();
            year = std::stoi(line.substr(13, 4));
            month = std::stoi(line.substr(18, 2));
            nlev = std::stoi(line.substr(32, 4));
            (void)nlev;
        } else if (!line.empty() && line[0] != '#') {
            block.push_back(line);
        }
    }
    flush();
    return 0;
}
