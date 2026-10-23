// Gridded ECAPE: the HRRR's CONUS domain (ecape.yml) and the South America viewer (scripts/models/samerica.py).
//
// Reads the float32 scratch written by fetch_hrrr.py or samerica.py, shape (NVAR, NLEV, ny, nx)
// with NVAR = PRES, HGT, TMP, SPFH, UGRD, VGRD, and runs SHARPlib's
// entrainment_cape() (Peters et al. 2023) on every column.
//
// It links skewt_wasm.cpp - the same wrapper the browser Skew-T runs through
// WebAssembly - so a point on this map and a click in the sounding explorer come
// from identical code, not merely the same library.
//
// Why native rather than driving the existing WASM from node: measured, the two
// are within 10% per column (WASM is compiled C++, so there is no interpreter to
// escape). The wins here are that OpenMP parallelises trivially over columns,
// and that nothing has to marshal ~2.3 GB of profiles between processes.
//
// Build:  see build.sh
// Usage:  ecape_grid <stem>        # reads <stem>.json + <stem>.f32
//                                  # writes <stem>_ecape.f32 + <stem>_ecape.json
//
// Two switches in <stem>.json, both off unless set to 1 (the HRRR job sets neither, so its columns and output are
// exactly as before):
//   "compact"  pressure-level input (scripts/models/samerica.py): level 0 is the surface, and a level whose pressure,
//              temperature or height is not finite (below the ground) is left out of that column instead of skipping
//              the whole column. A column needs its surface and at least 5 levels.
//   "kin"      also writes, per column, the 0-6 km bulk shear vector (u, v), storm-relative helicity 0-1 and 0-3 km
//              for the Bunkers right-moving storm, and precipitable water - the same SHARPlib calls the sounding
//              explorer's compute_sounding() makes for those numbers.

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <SHARPlib/layer.h>
#include <SHARPlib/params/convective.h>
#include <SHARPlib/thermo.h>
#include <SHARPlib/winds.h>

extern "C" int compute_sounding(const float*, const float*, const float*,
                                const float*, const float*, const float*, int,
                                float*, float*, float*, float*);
extern "C" int out_size();
extern "C" void set_ecape_screen(float);
extern "C" int compute_ecape4(const float*, const float*, const float*, const float*, const float*, const float*, int,
                              float*, float*, float*);

// skewt_wasm.cpp out[] indices we keep (see its header comment).
enum { I_ML_CAPE = 5, I_MU_CAPE = 10, I_ECAPE_MU = 42, I_ECAPE_ML = 45 };
static const float MISSING = -9999.0f;

// Output fields, in write order; "kin" appends the K_ fields after them.
enum { O_ECAPE_ML = 0, O_ECAPE_MU, O_MLCAPE, O_MUCAPE, O_NOUT };
enum { K_SHR6_U = 0, K_SHR6_V, K_SRH1, K_SRH3, K_PWAT, K_NOUT };

// The "kin" numbers of one column (surface first, n levels), MISSING where SHARPlib finds no answer or refuses the
// layer (it throws on a degenerate one).
static void kinematics(const float* P, const float* H, const float* D, const float* U, const float* V, int n,
                       float* mixr, float* k) {
    for (int i = 0; i < K_NOUT; ++i) k[i] = MISSING;
    try {
        const sharp::HeightLayer l6(0.0f, 6000.0f);
        const sharp::WindComponents shr = sharp::wind_shear(l6, H, U, V, n);
        k[K_SHR6_U] = shr.u;
        k[K_SHR6_V] = shr.v;
        const sharp::WindComponents rm = sharp::storm_motion_bunkers(P, H, U, V, n, l6, l6, false, false);
        if (rm.u != sharp::MISSING && rm.v != sharp::MISSING) {
            k[K_SRH1] = sharp::helicity(sharp::HeightLayer(0.0f, 1000.0f), rm, H, U, V, n);
            k[K_SRH3] = sharp::helicity(sharp::HeightLayer(0.0f, 3000.0f), rm, H, U, V, n);
        }
        for (int i = 0; i < n; ++i) mixr[i] = sharp::mixratio(P[i], D[i]);
        k[K_PWAT] = sharp::precipitable_water(sharp::PressureLayer(P[0], P[n - 1]), P, mixr, n);
    } catch (...) {
    }
    for (int i = 0; i < K_NOUT; ++i)
        if (!std::isfinite(k[i]) || k[i] == sharp::MISSING) k[i] = MISSING;
}

// --- minimal JSON scraping (the metadata file is ours, so this need not be a
// general parser - just pull the few scalars the kernel needs) -----------------
static long json_int(const std::string& s, const char* key, long dflt) {
    std::string k = std::string("\"") + key + "\"";
    size_t p = s.find(k);
    if (p == std::string::npos) return dflt;
    p = s.find(':', p);
    if (p == std::string::npos) return dflt;
    return std::strtol(s.c_str() + p + 1, nullptr, 10);
}

// Specific humidity -> dewpoint. SHARPlib wants dewpoint in K; HRRR native
// carries SPFH, so convert through vapour pressure (Bolton 1980 inverse).
static inline float dewpoint_from_q(float q, float p_pa) {
    if (!(q > 0.0f) || !(p_pa > 0.0f)) return NAN;
    const float w = q / (1.0f - q);                    // mixing ratio
    float e = p_pa * w / (0.621981f + w);              // vapour pressure (Pa)
    if (e < 1.0f) e = 1.0f;                            // keep the log finite
    const float l = std::log(e / 611.2f);
    return 243.5f * l / (17.67f - l) + 273.15f;        // K
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: ecape_grid <stem>\n");
        return 2;
    }
    const std::string stem = argv[1];

    // --- metadata ---------------------------------------------------------
    std::string meta;
    {
        FILE* f = std::fopen((stem + ".json").c_str(), "rb");
        if (!f) { std::perror("open meta"); return 2; }
        char buf[65536];
        size_t n = std::fread(buf, 1, sizeof(buf) - 1, f);
        buf[n] = 0; meta = buf; std::fclose(f);
    }
    const int nlev = (int)json_int(meta, "nlev", 50);
    const int ny   = (int)json_int(meta, "ny", 0);
    const int nx   = (int)json_int(meta, "nx", 0);
    const bool compact = json_int(meta, "compact", 0) == 1;
    const bool kin = json_int(meta, "kin", 0) == 1;
    const int nvar = 6;
    if (ny <= 0 || nx <= 0) { std::fprintf(stderr, "bad grid in meta\n"); return 2; }

    // --- map the scratch --------------------------------------------------
    const std::string f32 = stem + ".f32";
    int fd = ::open(f32.c_str(), O_RDONLY);
    if (fd < 0) { std::perror("open scratch"); return 2; }
    struct stat st{};
    if (::fstat(fd, &st) != 0) { std::perror("fstat"); return 2; }
    const size_t want = (size_t)nvar * nlev * ny * nx * sizeof(float);
    if ((size_t)st.st_size != want) {
        std::fprintf(stderr, "scratch is %lld bytes, expected %zu\n",
                     (long long)st.st_size, want);
        return 2;
    }
    const float* base = (const float*)::mmap(nullptr, want, PROT_READ, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) { std::perror("mmap"); return 2; }

    // var v, level k, row j, col i  ->  ((v*nlev + k)*ny + j)*nx + i
    const size_t plane = (size_t)ny * nx;
    auto VAR = [&](int v, int k) { return base + ((size_t)v * nlev + k) * plane; };
    enum { V_PRES = 0, V_HGT, V_TMP, V_SPFH, V_UGRD, V_VGRD };

    // --- level orientation ------------------------------------------------
    // SHARPlib wants the profile ordered surface -> top (pressure decreasing).
    // HRRR's hybrid numbering has flipped between conventions before, so detect
    // it from the data at a mid-domain point rather than assuming.
    bool flip = false;
    {
        const size_t mid = (size_t)(ny / 2) * nx + nx / 2;
        const float p_first = VAR(V_PRES, 0)[mid];
        const float p_last  = VAR(V_PRES, nlev - 1)[mid];
        flip = (p_first < p_last);          // level 0 is aloft -> reverse
        std::printf("  level order: %s (p[0]=%.0f hPa, p[%d]=%.0f hPa)\n",
                    flip ? "top-down, reversing" : "surface-up",
                    p_first / 100.0f, nlev - 1, p_last / 100.0f);
    }

    const int nfield = O_NOUT + (kin ? K_NOUT : 0);
    std::vector<float> out(plane * nfield, MISSING);
    const int nout = out_size();

    long long ok = 0, bad = 0;
    const bool full = std::getenv("ECAPE_FULL") && std::string(std::getenv("ECAPE_FULL")) == "1";
    if (const char* sk = std::getenv("ECAPE_SCREEN_K")) set_ecape_screen((float)std::atof(sk));   // tests; < 0 = off
    std::printf("  kernel: %s\n", full ? "full explorer analysis (ECAPE_FULL=1)" : "compute_ecape4 (ML/MU CAPE + ECAPE only, stable-column screen)");
#ifdef _OPENMP
    std::printf("  OpenMP: %d threads\n", omp_get_max_threads());
#endif
    std::fflush(stdout);

#pragma omp parallel reduction(+ : ok, bad)
    {
        // Per-thread scratch: one column plus SHARPlib's parcel buffers.
        std::vector<float> P(nlev), H(nlev), T(nlev), D(nlev), U(nlev), V(nlev);
        std::vector<float> o(nout), a(nlev), b(nlev), c(nlev);
        float kv[K_NOUT];

#pragma omp for schedule(dynamic, 8)
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const size_t idx = (size_t)j * nx + i;
                bool good = true;
                int n = 0;                          // levels in this column (all of them unless compact)
                for (int k = 0; k < nlev && good; ++k) {
                    const int kk = flip ? (nlev - 1 - k) : k;
                    const float p = VAR(V_PRES, kk)[idx];
                    const float t = VAR(V_TMP, kk)[idx];
                    const float q = VAR(V_SPFH, kk)[idx];
                    const float h = VAR(V_HGT, kk)[idx];
                    if (!std::isfinite(p) || !std::isfinite(t) ||
                        !std::isfinite(h) || p <= 0.0f) {
                        if (compact && k > 0) continue;   // a level below the ground: not in this column
                        good = false; break;
                    }
                    P[n] = p; T[n] = t; H[n] = h;
                    D[n] = dewpoint_from_q(q, p);
                    if (!std::isfinite(D[n])) D[n] = t - 40.0f;   // bone dry
                    if (D[n] > t) D[n] = t;                       // no supersaturation
                    U[n] = VAR(V_UGRD, kk)[idx];
                    V[n] = VAR(V_VGRD, kk)[idx];
                    if (!std::isfinite(U[n])) U[n] = 0.0f;
                    if (!std::isfinite(V[n])) V[n] = 0.0f;
                    ++n;
                }
                if (!good || n < 5) { ++bad; continue; }
                // SHARPlib requires strictly increasing height; HRRR native is
                // already monotonic but a flat pair would abort the parcel walk.
                for (int k = 1; k < n; ++k)
                    if (!(H[k] > H[k - 1])) H[k] = H[k - 1] + 0.1f;

                if (kin) {
                    kinematics(P.data(), H.data(), D.data(), U.data(), V.data(), n, a.data(), kv);
                    for (int f = 0; f < K_NOUT; ++f) out[(O_NOUT + f) * plane + idx] = kv[f];
                }
                // the four numbers only (2026-10-04); ECAPE_FULL=1 runs the explorer's whole analysis for comparison
                const int rc = full ? compute_sounding(P.data(), H.data(), T.data(), D.data(), U.data(), V.data(), n,
                                                       o.data(), a.data(), b.data(), c.data())
                                    : compute_ecape4(P.data(), H.data(), T.data(), D.data(), U.data(), V.data(), n,
                                                     o.data(), b.data(), c.data());
                if (rc != 0) { ++bad; continue; }
                auto keep = [&](int src) {
                    const float v = o[src];
                    // A non-convective column has ECAPE of zero, not "missing" -
                    // clamping to 0 keeps the map's zero contour meaningful.
                    return (std::isfinite(v) && v > MISSING) ? (v < 0.0f ? 0.0f : v) : 0.0f;
                };
                // Do NOT clamp ECAPE to CAPE. The ratio legitimately exceeds 1:
                // Peters' formulation nets a storm-relative kinetic-energy gain
                // against the entrainment loss, so ratio > 1 means the inflow
                // more than pays for the mixing. That regime is the most
                // interesting thing this field can show, and clamping would
                // erase it. (See skewt/methodology.html, "Parcels table".)
                out[O_ECAPE_ML * plane + idx] = keep(I_ECAPE_ML);
                out[O_ECAPE_MU * plane + idx] = keep(I_ECAPE_MU);
                out[O_MLCAPE   * plane + idx] = keep(I_ML_CAPE);
                out[O_MUCAPE   * plane + idx] = keep(I_MU_CAPE);
                ++ok;
            }
        }
    }

    ::munmap((void*)base, want);
    ::close(fd);

    const std::string of = stem + "_ecape.f32";
    FILE* fo = std::fopen(of.c_str(), "wb");
    if (!fo) { std::perror("open output"); return 2; }
    std::fwrite(out.data(), sizeof(float), out.size(), fo);
    std::fclose(fo);

    FILE* fj = std::fopen((stem + "_ecape.json").c_str(), "wb");
    std::fprintf(fj,
        "{\n  \"fields\": [\"ecape_ml\", \"ecape_mu\", \"mlcape\", \"mucape\"%s],\n"
        "  \"shape\": [%d, %d, %d],\n  \"dtype\": \"float32\",\n"
        "  \"order\": \"C\",\n  \"columns_ok\": %lld,\n  \"columns_skipped\": %lld\n}\n",
        kin ? ", \"shr6_u\", \"shr6_v\", \"srh1\", \"srh3\", \"pwat\"" : "", nfield, ny, nx, ok, bad);
    std::fclose(fj);

    std::printf("  columns: %lld computed, %lld skipped -> %s\n",
                ok, bad, of.c_str());
    return 0;
}
