"""Ion-mobility watershed segmentation of DIA MS2 isolation windows.

For each (frame, DIA isolation window), the peaks form vertical bands in ion
mobility (1/K0): all fragments of a precursor share its mobility. This module
segments a window's peaks into discrete, data-driven mobility bands so each
peptide's fragments are grouped coherently, and rejects chemical-noise bands
(broad, uniform-intensity). It replaces fixed-width IM binning for MS2.

Method (ported and validated in timsTOFRunner/im_density_ratio_random.py):
  1. Profile: on an evenly spaced mobility grid, take the MAX peak intensity
     within +/- kernel_width/2 of each grid center; subtract the median over
     occupied centers -> centered profile (>0 == above-median signal).
  2. Split into contiguous above-median regions.
  3. Relative watershed per region: detect every local maximum (edge-padded so
     boundary spikes are seen), then greedily merge adjacent apexes whose
     connecting valley does not drop by >= drop_frac of the SMALLER apex.
     Band boundaries fall at the valley minima between surviving apexes.
  4. Keep a band only if it holds >= min_peaks peaks AND their intensity
     coefficient of variation (std/mean) >= cv_min (uniform == noise).
  5. Peaks outside every kept band are dropped.
"""

import time

import numpy as np
import numba as nb
from scipy.signal import find_peaks

# --- Tunable parameters (final values from the prototype) --------------------
KERNEL_WIDTH = 0.00908   # absolute sliding-max window full-width in 1/K0
DROP_FRAC = 0.5          # valley must drop this frac of the smaller apex to split
MIN_PEAKS = 20           # min actual peaks in a band to keep it
CV_MIN = 0.6             # min intensity coefficient of variation to keep a band
GRID_OVERSAMPLE = 10     # grid spacing = KERNEL_WIDTH / GRID_OVERSAMPLE

# --- Lightweight per-step profiling ------------------------------------------
# Summed across all windows in a load; logged once by the caller.
TIMINGS = {
    "profile_build": 0.0,
    "region_split": 0.0,
    "watershed_merge": 0.0,
    "band_filter": 0.0,
    "spectrum_construct": 0.0,  # filled in by the loader
    "n_windows": 0,
    "n_bands_kept": 0,
}


def reset_timings():
    for k in TIMINGS:
        TIMINGS[k] = 0.0
    TIMINGS["n_windows"] = 0
    TIMINGS["n_bands_kept"] = 0


def format_timings():
    t = TIMINGS
    seg = (t["profile_build"] + t["region_split"]
           + t["watershed_merge"] + t["band_filter"])
    total = seg + t["spectrum_construct"]

    def pct(x):
        return f"{x:.2f}s ({100 * x / total:.0f}%)" if total > 0 else f"{x:.2f}s"

    return (
        f"IM watershed: {t['n_windows']} windows, "
        f"{t['n_bands_kept']} bands kept in {total:.2f}s\n"
        f"  profile_build {pct(t['profile_build'])}   "
        f"region_split {pct(t['region_split'])}   "
        f"watershed_merge {pct(t['watershed_merge'])}\n"
        f"  band_filter {pct(t['band_filter'])}   "
        f"spectrum_construct {pct(t['spectrum_construct'])}"
    )


def _segment_region(prof, r0, r1, drop_frac):
    """Split one contiguous signal region by relative valley depth.

    Find every local maximum (padding the ends so edge spikes are detected),
    then greedily dissolve any adjacent pair whose connecting valley does not
    drop by >= drop_frac of the SMALLER apex's height. Returns (start, end,
    apex) index tuples in grid coordinates.
    """
    seg = np.concatenate(([-np.inf], prof[r0:r1 + 1], [-np.inf]))
    ap, _ = find_peaks(seg, prominence=0)
    ap = [r0 + int(a) - 1 for a in ap]
    if not ap:
        ap = [r0 + int(np.argmax(prof[r0:r1 + 1]))]
    changed = True
    while len(ap) > 1 and changed:
        changed = False
        for i in range(len(ap) - 1):
            a, b = ap[i], ap[i + 1]
            v = a + int(np.argmin(prof[a:b + 1]))
            small = min(prof[a], prof[b])
            if small <= 0 or (small - prof[v]) / small < drop_frac:
                ap.pop(i if prof[a] < prof[b] else i + 1)
                changed = True
                break
    ap.sort()
    edges = [r0] + [ap[i] + int(np.argmin(prof[ap[i]:ap[i + 1] + 1]))
                    for i in range(len(ap) - 1)] + [r1]
    return [(edges[k], edges[k + 1], ap[k]) for k in range(len(ap))]


def segment_window(mob, intens, kernel_width=KERNEL_WIDTH, drop_frac=DROP_FRAC,
                   min_peaks=MIN_PEAKS, cv_min=CV_MIN,
                   grid_oversample=GRID_OVERSAMPLE):
    """Segment one (frame, DIA window)'s peaks into ion-mobility bands.

    Parameters
    ----------
    mob, intens : np.ndarray
        Mobility (1/K0) and intensity of every peak in the window (any order).

    Returns
    -------
    list of (im_lo, im_hi, peak_idx)
        One entry per kept band: mobility bounds and an int array of indices
        into the input arrays for peaks in that band. Empty list if none.
    """
    TIMINGS["n_windows"] += 1
    mob = np.asarray(mob, dtype=np.float64)
    intens = np.asarray(intens, dtype=np.float64)
    if mob.size < min_peaks:
        return []

    order = np.argsort(mob)
    mob_s = mob[order]
    int_s = intens[order]
    mob_min, mob_max = mob_s[0], mob_s[-1]
    half = kernel_width / 2.0
    if mob_max - mob_min <= kernel_width:
        return []

    # --- 1. sliding-max profile on an adaptive grid --------------------------
    t0 = time.perf_counter()
    spacing = kernel_width / grid_oversample
    n_grid = max(3, int(np.ceil((mob_max - mob_min - kernel_width) / spacing)) + 1)
    centers = np.linspace(mob_min + half, mob_max - half, n_grid)
    lefts = np.searchsorted(mob_s, centers - half, side="left")
    rights = np.searchsorted(mob_s, centers + half, side="right")
    prof = np.zeros(n_grid, dtype=np.float64)
    counts = rights - lefts
    for i in range(n_grid):
        if counts[i] > 0:
            prof[i] = int_s[lefts[i]:rights[i]].max()
    occupied = counts > 0
    if not occupied.any():
        return []
    prof = prof - np.median(prof[occupied])
    TIMINGS["profile_build"] += time.perf_counter() - t0

    # --- 2. contiguous above-median regions ----------------------------------
    t0 = time.perf_counter()
    idx = np.flatnonzero(prof > 0)
    if idx.size == 0:
        TIMINGS["region_split"] += time.perf_counter() - t0
        return []
    regions = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    TIMINGS["region_split"] += time.perf_counter() - t0

    # --- 3. relative watershed per region ------------------------------------
    t0 = time.perf_counter()
    bands = []
    for reg in regions:
        bands += _segment_region(prof, reg[0], reg[-1], drop_frac)
    TIMINGS["watershed_merge"] += time.perf_counter() - t0

    # --- 4. filter bands by peak count and intensity CV ----------------------
    t0 = time.perf_counter()
    kept = []
    for i0, i1, _apex in bands:
        im_lo = float(centers[i0])
        im_hi = float(centers[i1])
        lo = np.searchsorted(mob_s, im_lo, side="left")
        hi = np.searchsorted(mob_s, im_hi, side="right")
        n = hi - lo
        if n < min_peaks:
            continue
        vals = int_s[lo:hi]
        mean = vals.mean()
        if mean <= 0 or vals.std() / mean < cv_min:
            continue
        kept.append((im_lo, im_hi, order[lo:hi]))
    TIMINGS["band_filter"] += time.perf_counter() - t0
    TIMINGS["n_bands_kept"] += len(kept)
    return kept
