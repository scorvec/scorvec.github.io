// Gridded ECAPE over the HRRR CONUS domain.
//
// Reads the float32 scratch written by fetch_hrrr.py, shape (NVAR, NLEV, ny, nx)
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

#include <cmath>
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

extern "C" int compute_sounding(const float*, const float*, const float*,
                                const float*, const float*, const float*, int,
                                float*, float*, float*, float*);
extern "C" int out_size();

// skewt_wasm.cpp out[] indices we keep (see its header comment).
enum { I_ML_CAPE = 5, I_MU_CAPE = 10, I_ECAPE_MU = 42, I_ECAPE_ML = 45 };
static const float MISSING = -9999.0f;

// Output fields, in write order.
enum { O_ECAPE_ML = 0, O_ECAPE_MU, O_MLCAPE, O_MUCAPE, O_NOUT };

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

    std::vector<float> out(plane * O_NOUT, MISSING);
    const int nout = out_size();

    long long ok = 0, bad = 0;
#ifdef _OPENMP
    std::printf("  OpenMP: %d threads\n", omp_get_max_threads());
#endif
    std::fflush(stdout);

#pragma omp parallel reduction(+ : ok, bad)
    {
        // Per-thread scratch: one column plus SHARPlib's parcel buffers.
        std::vector<float> P(nlev), H(nlev), T(nlev), D(nlev), U(nlev), V(nlev);
        std::vector<float> o(nout), a(nlev), b(nlev), c(nlev);

#pragma omp for schedule(dynamic, 8)
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const size_t idx = (size_t)j * nx + i;
                bool good = true;
                for (int k = 0; k < nlev && good; ++k) {
                    const int kk = flip ? (nlev - 1 - k) : k;
                    const float p = VAR(V_PRES, kk)[idx];
                    const float t = VAR(V_TMP, kk)[idx];
                    const float q = VAR(V_SPFH, kk)[idx];
                    const float h = VAR(V_HGT, kk)[idx];
                    if (!std::isfinite(p) || !std::isfinite(t) ||
                        !std::isfinite(h) || p <= 0.0f) { good = false; break; }
                    P[k] = p; T[k] = t; H[k] = h;
                    D[k] = dewpoint_from_q(q, p);
                    if (!std::isfinite(D[k])) D[k] = t - 40.0f;   // bone dry
                    if (D[k] > t) D[k] = t;                       // no supersaturation
                    U[k] = VAR(V_UGRD, kk)[idx];
                    V[k] = VAR(V_VGRD, kk)[idx];
                    if (!std::isfinite(U[k])) U[k] = 0.0f;
                    if (!std::isfinite(V[k])) V[k] = 0.0f;
                }
                if (!good) { ++bad; continue; }
                // SHARPlib requires strictly increasing height; HRRR native is
                // already monotonic but a flat pair would abort the parcel walk.
                for (int k = 1; k < nlev; ++k)
                    if (!(H[k] > H[k - 1])) H[k] = H[k - 1] + 0.1f;

                const int rc = compute_sounding(P.data(), H.data(), T.data(), D.data(),
                                                U.data(), V.data(), nlev,
                                                o.data(), a.data(), b.data(), c.data());
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
        "{\n  \"fields\": [\"ecape_ml\", \"ecape_mu\", \"mlcape\", \"mucape\"],\n"
        "  \"shape\": [%d, %d, %d],\n  \"dtype\": \"float32\",\n"
        "  \"order\": \"C\",\n  \"columns_ok\": %lld,\n  \"columns_skipped\": %lld\n}\n",
        O_NOUT, ny, nx, ok, bad);
    std::fclose(fj);

    std::printf("  columns: %lld computed, %lld skipped -> %s\n",
                ok, bad, of.c_str());
    return 0;
}
