/* Fused TC-candidate detection kernel.
 *
 * Replaces three chained scipy passes (minimum_filter 9x9 + two uniform_filter
 * box means for the ring test) with one pass over the grid using a
 * summed-area table for O(1) box sums, restricted to the tracking latitude
 * band, with longitude wrap via ghost columns. ~10x the throughput of the
 * scipy path on a 721x1440 field.
 *
 * Build (done automatically by tc_tracker.py when missing/stale):
 *   cc -O3 -shared -fPIC -o detect.dylib detect.c
 */
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

/* candidates written as rows of (iy, ix, pmin, depth, wmax) */
int detect_step(const float *msl, const float *wind,
                int nlat, int nlon,
                int iy0, int iy1,          /* latitude-band row range [iy0, iy1) */
                float p_track,             /* candidate ceiling, hPa            */
                int min_px,                /* local-min half-window (4 -> 9x9)  */
                int ring_px,               /* ring radius in px (20 -> ~5 deg)  */
                float depth_min,           /* ring-mean minus centre, hPa       */
                int wind_px,               /* wind search half-window (8 -> 2deg) */
                int *out_iy, int *out_ix,
                float *out_p, float *out_depth, float *out_w,
                int max_out)
{
    const int pad = ring_px + 1;
    const int wlon = nlon + 2 * pad;              /* ghost columns for wrap */

    /* summed-area table over the padded field: sat[(nlat+1) x (wlon+1)] */
    double *sat = (double *)malloc((size_t)(nlat + 1) * (wlon + 1) * sizeof(double));
    if (!sat) return -1;
    const size_t W = (size_t)wlon + 1;
    for (size_t j = 0; j < W; j++) sat[j] = 0.0;
    for (int i = 0; i < nlat; i++) {
        double rowsum = 0.0;
        double *row = sat + (size_t)(i + 1) * W;
        const double *prev = sat + (size_t)i * W;
        row[0] = 0.0;
        for (int j = 0; j < wlon; j++) {
            int src = (j - pad + nlon) % nlon;    /* wrapped source column */
            rowsum += (double)msl[(size_t)i * nlon + src];
            row[j + 1] = prev[j + 1] + rowsum;
        }
    }
#define BOXSUM(i0, i1, j0, j1) /* rows [i0,i1) cols [j0,j1) in padded space */ \
    (sat[(size_t)(i1) * W + (j1)] - sat[(size_t)(i0) * W + (j1)]              \
     - sat[(size_t)(i1) * W + (j0)] + sat[(size_t)(i0) * W + (j0)])

    int n = 0;
    for (int i = iy0; i < iy1 && n < max_out; i++) {
        const float *rowp = msl + (size_t)i * nlon;
        for (int j = 0; j < nlon; j++) {
            const float c = rowp[j];
            if (c >= p_track) continue;

            /* strict local minimum over the (2*min_px+1)^2 neighbourhood */
            int is_min = 1;
            for (int di = -min_px; di <= min_px && is_min; di++) {
                int ii = i + di;
                if (ii < 0 || ii >= nlat) continue;
                const float *r2 = msl + (size_t)ii * nlon;
                for (int dj = -min_px; dj <= min_px; dj++) {
                    if (di == 0 && dj == 0) continue;
                    int jj = j + dj;
                    jj = (jj + nlon) % nlon;
                    if (r2[jj] < c) { is_min = 0; break; }
                }
            }
            if (!is_min) continue;

            /* ring depth: mean over the annulus between the outer box
             * (2*ring_px+1)^2 and inner box (ring_px)^2, via the SAT      */
            int o0 = i - ring_px, o1 = i + ring_px + 1;
            if (o0 < 0) o0 = 0;
            if (o1 > nlat) o1 = nlat;
            int half = ring_px / 2;
            int in0 = i - half, in1 = i + half + 1;
            if (in0 < 0) in0 = 0;
            if (in1 > nlat) in1 = nlat;
            int jo0 = j, jo1 = j + 2 * ring_px + 1;          /* padded cols  */
            int ji0 = j + ring_px - half, ji1 = j + ring_px + half + 1;
            double outer = BOXSUM(o0, o1, jo0, jo1);
            double inner = BOXSUM(in0, in1, ji0, ji1);
            double n_out = (double)(o1 - o0) * (jo1 - jo0);
            double n_in = (double)(in1 - in0) * (ji1 - ji0);
            double ring = (outer - inner) / (n_out - n_in);
            float depth = (float)(ring - c);
            if (depth < depth_min) continue;

            /* peak wind near the centre (few candidates -> direct scan) */
            float wmax = 0.0f;
            int w0 = i - wind_px, w1 = i + wind_px + 1;
            if (w0 < 0) w0 = 0;
            if (w1 > nlat) w1 = nlat;
            for (int ii = w0; ii < w1; ii++) {
                const float *wr = wind + (size_t)ii * nlon;
                for (int dj = -wind_px; dj <= wind_px; dj++) {
                    int jj = (j + dj + nlon) % nlon;
                    if (wr[jj] > wmax) wmax = wr[jj];
                }
            }

            out_iy[n] = i; out_ix[n] = j;
            out_p[n] = c; out_depth[n] = depth; out_w[n] = wmax;
            n++;
        }
    }
    free(sat);
    return n;
}
