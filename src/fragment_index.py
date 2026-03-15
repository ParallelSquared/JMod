"""
Fragment ion index for fast candidate pre-filtering in DIA spectral fitting.

Builds a nominal-mass-binned index of fragment ions with u16 fractional m/z
encoding. Queries count fragment matches per precursor within a DIA isolation
window and RT tolerance, returning candidates with >= atleast_m matches.

Uses numba JIT compilation for the inner query loop to avoid Python overhead.
"""

import numpy as np
from numba import njit


@njit(inline='always')
def _searchsorted_left_f32(arr, lo, hi, val):
    """Binary search for leftmost insertion point of val in arr[lo:hi]."""
    while lo < hi:
        mid = (lo + hi) >> 1
        if arr[mid] < val:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(inline='always')
def _searchsorted_right_f32(arr, lo, hi, val):
    """Binary search for rightmost insertion point of val in arr[lo:hi]."""
    while lo < hi:
        mid = (lo + hi) >> 1
        if arr[mid] <= val:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit
def _query_partition_jit(flat_prec_mz, flat_deficit, flat_prec_idx,
                         bin_offsets, bin_lengths, min_nominal,
                         counter, ion_nominals, ion_deficits, ion_tols,
                         win_lo, win_hi):
    """JIT-compiled inner loop: count fragment matches per precursor.

    For each observed ion, looks up the nominal mass bin, binary-searches for
    precursors within the DIA isolation window, and increments the counter for
    precursors whose fragment deficit matches within tolerance.
    """
    n_bins = len(bin_offsets)

    for i in range(len(ion_nominals)):
        nominal = ion_nominals[i]
        query_deficit = ion_deficits[i]
        tol_u16 = ion_tols[i]

        # Query primary nominal bin
        bin_idx = nominal - min_nominal
        if 0 <= bin_idx < n_bins:
            length = bin_lengths[bin_idx]
            if length > 0:
                offset = bin_offsets[bin_idx]
                lo = _searchsorted_left_f32(flat_prec_mz, offset, offset + length, win_lo)
                hi = _searchsorted_right_f32(flat_prec_mz, offset, offset + length, win_hi)
                for j in range(lo, hi):
                    diff = flat_deficit[j] - query_deficit
                    if diff < 0:
                        diff = -diff
                    if diff <= tol_u16:
                        counter[flat_prec_idx[j]] += 1

        # Handle 1 Da boundary wrapping — low side
        if query_deficit - tol_u16 < 0:
            wrapped_deficit = 65536 + (query_deficit - tol_u16)
            wrapped_tol = -(query_deficit - tol_u16)
            bin_idx_lo = (nominal - 1) - min_nominal
            if 0 <= bin_idx_lo < n_bins:
                length = bin_lengths[bin_idx_lo]
                if length > 0:
                    offset = bin_offsets[bin_idx_lo]
                    lo = _searchsorted_left_f32(flat_prec_mz, offset, offset + length, win_lo)
                    hi = _searchsorted_right_f32(flat_prec_mz, offset, offset + length, win_hi)
                    for j in range(lo, hi):
                        diff = flat_deficit[j] - wrapped_deficit
                        if diff < 0:
                            diff = -diff
                        if diff <= wrapped_tol:
                            counter[flat_prec_idx[j]] += 1

        # Handle 1 Da boundary wrapping — high side
        if query_deficit + tol_u16 > 65535:
            wrapped_deficit = (query_deficit + tol_u16) - 65536
            wrapped_tol = (query_deficit + tol_u16) - 65535
            bin_idx_hi = (nominal + 1) - min_nominal
            if 0 <= bin_idx_hi < n_bins:
                length = bin_lengths[bin_idx_hi]
                if length > 0:
                    offset = bin_offsets[bin_idx_hi]
                    lo = _searchsorted_left_f32(flat_prec_mz, offset, offset + length, win_lo)
                    hi = _searchsorted_right_f32(flat_prec_mz, offset, offset + length, win_hi)
                    for j in range(lo, hi):
                        diff = flat_deficit[j] - wrapped_deficit
                        if diff < 0:
                            diff = -diff
                        if diff <= wrapped_tol:
                            counter[flat_prec_idx[j]] += 1


class _Partition:
    """One RT-range slice of the library, with flat arrays for numba access."""
    __slots__ = ('rt_lo', 'rt_hi', 'n_precursors', 'precursor_global_idx',
                 'precursor_rt', 'flat_prec_mz', 'flat_deficit', 'flat_prec_idx',
                 'bin_offsets', 'bin_lengths', 'min_nominal', 'counter')

    def __init__(self, rt_lo, rt_hi, n_precursors, precursor_global_idx,
                 precursor_rt, flat_prec_mz, flat_deficit, flat_prec_idx,
                 bin_offsets, bin_lengths, min_nominal):
        self.rt_lo = rt_lo
        self.rt_hi = rt_hi
        self.n_precursors = n_precursors
        self.precursor_global_idx = precursor_global_idx  # uint32[n_precursors]
        self.precursor_rt = precursor_rt                  # float32[n_precursors]
        self.flat_prec_mz = flat_prec_mz    # float32 — all bins concatenated, sorted within each bin
        self.flat_deficit = flat_deficit      # int32 — deficit values (int32 for numba arithmetic)
        self.flat_prec_idx = flat_prec_idx  # int32 — precursor indices (int32 for numba)
        self.bin_offsets = bin_offsets        # int32[n_bins] — offset into flat arrays
        self.bin_lengths = bin_lengths        # int32[n_bins] — length of each bin
        self.min_nominal = min_nominal        # int32 — minimum nominal mass
        self.counter = np.zeros(n_precursors, dtype=np.int32)  # int32 workspace


class FragmentIndex:
    """Fragment ion index for fast candidate pre-filtering."""

    def __init__(self, mz_tol_ppm: float):
        self.partitions: list[_Partition] = []
        self.mz_tol_ppm = mz_tol_ppm

    @classmethod
    def build(cls, library, all_keys, rt_mz, mz_tol_ppm,
              prec_mz_offset=0.0, max_frags_per_partition=312_000):
        """Build a FragmentIndex from a spectrum library.

        Args:
            library: SpectrumLibraryStore — library[key]['spectrum'] is n×2 (mz, intensity),
                     library[key]['top_n'] is array of indices into spectrum.
            all_keys: list of keys into library (tuples of (mod_seq, charge, ...)).
            rt_mz: ndarray shape (len(all_keys), 2) — col 0 = calibrated RT, col 1 = calibrated precursor m/z.
            mz_tol_ppm: float — fragment m/z tolerance in ppm.
            prec_mz_offset: float — offset applied to precursor m/z (e.g. -decoy_mz_offset for decoys).
            max_frags_per_partition: int — max fragments per partition (~312K for L3 cache fit).
        """
        idx = cls(mz_tol_ppm)

        n = len(all_keys)
        if n == 0:
            return idx

        # Count fragments per precursor (top_n only)
        frag_counts = np.array([len(library[all_keys[i]]['top_n']) for i in range(n)], dtype=np.int32)

        # Sort by calibrated RT
        rt_order = np.argsort(rt_mz[:, 0])

        # Determine partition boundaries based on cumulative fragment count
        cumulative = np.cumsum(frag_counts[rt_order])
        partition_starts = [0]
        last_cut = 0
        for i in range(n):
            if cumulative[i] - (cumulative[last_cut - 1] if last_cut > 0 else 0) > max_frags_per_partition:
                partition_starts.append(i)
                last_cut = i
        partition_ends = partition_starts[1:] + [n]

        for p_start, p_end in zip(partition_starts, partition_ends):
            p_indices = rt_order[p_start:p_end]
            n_prec = len(p_indices)

            prec_rts = rt_mz[p_indices, 0].astype(np.float32)
            prec_mzs = (rt_mz[p_indices, 1] + prec_mz_offset).astype(np.float32)

            # Gather all fragment data
            all_frag_mz = []
            all_frag_prec_mz = []
            all_frag_prec_idx = []

            for local_idx in range(n_prec):
                global_idx = p_indices[local_idx]
                key = all_keys[global_idx]
                spectrum = library[key]['spectrum']
                top_n = library[key]['top_n']
                frag_mzs = spectrum[top_n, 0]

                all_frag_mz.append(frag_mzs)
                all_frag_prec_mz.append(np.full(len(frag_mzs), prec_mzs[local_idx], dtype=np.float32))
                all_frag_prec_idx.append(np.full(len(frag_mzs), local_idx, dtype=np.int32))

            if not all_frag_mz:
                partition = _Partition(
                    rt_lo=float(prec_rts.min()), rt_hi=float(prec_rts.max()),
                    n_precursors=n_prec, precursor_global_idx=p_indices.astype(np.uint32),
                    precursor_rt=prec_rts,
                    flat_prec_mz=np.empty(0, dtype=np.float32),
                    flat_deficit=np.empty(0, dtype=np.int32),
                    flat_prec_idx=np.empty(0, dtype=np.int32),
                    bin_offsets=np.empty(0, dtype=np.int32),
                    bin_lengths=np.empty(0, dtype=np.int32),
                    min_nominal=0,
                )
                idx.partitions.append(partition)
                continue

            frag_mz_arr = np.concatenate(all_frag_mz)
            frag_prec_mz = np.concatenate(all_frag_prec_mz)
            frag_prec_idx = np.concatenate(all_frag_prec_idx)

            # Compute nominal mass and deficit
            nominal = np.floor(frag_mz_arr).astype(np.int32)
            deficit = np.round((frag_mz_arr - nominal) * 65536).astype(np.int32)

            # Build flat layout: group by nominal mass, sort within each bin by prec_mz
            min_nom = int(nominal.min())
            max_nom = int(nominal.max())
            n_bins = max_nom - min_nom + 1

            # Sort all entries by nominal mass, then by prec_mz within each nominal
            sort_order = np.lexsort((frag_prec_mz, nominal))
            nominal = nominal[sort_order]
            deficit = deficit[sort_order]
            frag_prec_mz = frag_prec_mz[sort_order]
            frag_prec_idx = frag_prec_idx[sort_order]

            # Compute bin offsets and lengths
            bin_offsets = np.zeros(n_bins, dtype=np.int32)
            bin_lengths = np.zeros(n_bins, dtype=np.int32)

            unique_nominals, counts = np.unique(nominal, return_counts=True)
            cum = 0
            for nom_val, count in zip(unique_nominals, counts):
                bin_idx = nom_val - min_nom
                bin_offsets[bin_idx] = cum
                bin_lengths[bin_idx] = count
                cum += count

            partition = _Partition(
                rt_lo=float(prec_rts.min()), rt_hi=float(prec_rts.max()),
                n_precursors=n_prec, precursor_global_idx=p_indices.astype(np.uint32),
                precursor_rt=prec_rts,
                flat_prec_mz=frag_prec_mz,
                flat_deficit=deficit,
                flat_prec_idx=frag_prec_idx,
                bin_offsets=bin_offsets,
                bin_lengths=bin_lengths,
                min_nominal=np.int32(min_nom),
            )
            idx.partitions.append(partition)

        return idx

    def query(self, dia_mz_array, win_lo, win_hi, prec_rt, rt_tol, atleast_m):
        """Return global indices of precursors with >= atleast_m fragment matches within RT window.

        Args:
            dia_mz_array: 1D array of observed ion m/z values from DIA spectrum.
            win_lo: float — lower bound of DIA isolation window (prec_mz - windowWidth/2).
            win_hi: float — upper bound of DIA isolation window (prec_mz + windowWidth/2).
            prec_rt: float — retention time of the DIA spectrum.
            rt_tol: float — RT tolerance for candidate filtering.
            atleast_m: int — minimum fragment match count.

        Returns:
            np.ndarray of uint32 global indices into all_keys/rt_mz.
        """
        active = [p for p in self.partitions
                  if p.rt_hi >= prec_rt - rt_tol and p.rt_lo <= prec_rt + rt_tol]

        win_lo_f32 = np.float32(win_lo)
        win_hi_f32 = np.float32(win_hi)

        # Precompute per-ion nominal, deficit, tol
        ion_nominals = np.floor(dia_mz_array).astype(np.int32)
        ion_deficits = np.round((dia_mz_array - ion_nominals) * 65536).astype(np.int32)
        ion_tols = np.ceil(ion_nominals * self.mz_tol_ppm * 1e-6 * 65536).astype(np.int32)

        candidates = []
        for p in active:
            if len(p.flat_prec_mz) == 0:
                continue

            p.counter[:] = 0

            _query_partition_jit(
                p.flat_prec_mz, p.flat_deficit, p.flat_prec_idx,
                p.bin_offsets, p.bin_lengths, p.min_nominal,
                p.counter, ion_nominals, ion_deficits, ion_tols,
                win_lo_f32, win_hi_f32,
            )

            # Phase 2: collect fragment-passing precursors, then RT filter
            passing = np.where(p.counter >= atleast_m)[0]
            if len(passing) > 0:
                rt_ok = np.abs(p.precursor_rt[passing] - prec_rt) < rt_tol
                candidates.extend(p.precursor_global_idx[passing[rt_ok]])

        return np.array(candidates, dtype=np.uint32) if candidates else np.empty(0, dtype=np.uint32)
