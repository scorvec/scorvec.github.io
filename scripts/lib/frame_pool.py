"""Draw a loop's frames in parallel. Forked workers inherit the caller's data, so only each frame's small job tuple
crosses to them; matplotlib and cartopy then draw one figure at a time in each process.

    import frame_pool as FP
    FP.share(full=full)             # what the frame function reads (FP.DATA["full"]), inherited by the workers
    FP.draw(frame, jobs)            # frame(job) for every job, a module-level function; its results in job order

FRAME_PROCS bounds the workers (default: every core); 1 draws in this process. Run the caller with its numerical
libraries single-threaded (OMP_NUM_THREADS=1 etc., as scripts/lib/parallel.sh sets them): the workers inherit the
parent's thread pools, and N workers each spinning N BLAS threads drowned in contention. The first frame is always drawn here,
before any worker starts, so whatever it downloads once (cartopy's Natural Earth files) is on disk for them: workers
fetching the same shapefile at once corrupted it.
"""

from __future__ import annotations

import multiprocessing as mp
import os

DATA: dict = {}


def share(**data) -> None:
    DATA.clear()
    DATA.update(data)


def procs() -> int:
    return int(os.environ.get("FRAME_PROCS", "0")) or os.cpu_count() or 1


def draw(frame, jobs: list) -> list:
    if not jobs:
        return []
    first = frame(jobs[0])
    n = min(procs(), len(jobs) - 1)
    if n <= 1:
        return [first, *(frame(j) for j in jobs[1:])]
    with mp.get_context("fork").Pool(n) as pool:
        return [first, *pool.map(frame, jobs[1:], chunksize=1)]
