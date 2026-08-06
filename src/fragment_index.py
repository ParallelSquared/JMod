"""
Fragment ion index for fast candidate pre-filtering in DIA spectral fitting.

Builds a nominal-mass-binned index of fragment ions with u16 fractional m/z
encoding. Queries count fragment matches per precursor within a DIA isolation
window and RT tolerance, returning candidates with >= atleast_m matches.

Uses numba JIT compilation for the inner query loop to avoid Python overhead.
"""

#  Copyright (c) 2026 Parallel Squared Technology Institute
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  """

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


@njit(nogil=True)
def _query_all_partitions_jit(
        # DIA observed fragment m/z values
        dia_mz_array, mz_tol_ppm,
        # DIA isolation window bounds
        win_lo, win_hi,
        # Query parameters
        prec_rt, rt_tol, atleast_m,
        # Partition metadata (one entry per partition)
        part_rt_lo, part_rt_hi, part_n_precursors, part_min_nominal,
        # Fragment data — concatenated across partitions
        all_flat_prec_mz, all_flat_deficit, all_flat_prec_idx,
        flat_offsets,       # int64[n_partitions+1] — into fragment arrays
        # Bin metadata — concatenated across partitions
        all_bin_offsets, all_bin_lengths,
        bin_meta_offsets,   # int64[n_partitions+1] — into bin arrays
        # Precursor metadata — concatenated across partitions
        all_prec_global_idx, all_prec_rt,
        prec_offsets,       # int64[n_partitions+1] — into precursor arrays
        # Output buffer (pre-allocated, max possible size)
        out_indices):
    """Full query across all partitions in one nogil pass.

    Computes ion nominals, deficits, and tolerances internally to avoid
    GIL-holding numpy calls in the Python wrapper.

    Returns the number of results written to out_indices.
    """
    # Precompute per-ion nominal, deficit, tol (replaces 3 numpy calls in Python)
    n_ions = len(dia_mz_array)
    ion_nominals = np.empty(n_ions, dtype=np.int32)
    ion_deficits = np.empty(n_ions, dtype=np.int32)
    ion_tols = np.empty(n_ions, dtype=np.int32)
    ppm_factor = mz_tol_ppm * 1e-6 * 65536.0
    for i in range(n_ions):
        nom = int(np.floor(dia_mz_array[i]))
        ion_nominals[i] = nom
        ion_deficits[i] = int(np.round((dia_mz_array[i] - nom) * 65536.0))
        # np.ceil equivalent for positive values
        tol_f = nom * ppm_factor
        tol_i = int(tol_f)
        if tol_f > tol_i:
            tol_i += 1
        ion_tols[i] = tol_i

    n_partitions = len(part_rt_lo)
    rt_lo_bound = prec_rt - rt_tol
    rt_hi_bound = prec_rt + rt_tol
    n_results = 0

    for p in range(n_partitions):
        # Partition RT overlap check
        if part_rt_hi[p] < rt_lo_bound or part_rt_lo[p] > rt_hi_bound:
            continue

        # Skip empty partitions
        frag_start = flat_offsets[p]
        frag_end = flat_offsets[p + 1]
        if frag_start == frag_end:
            continue

        n_prec = part_n_precursors[p]
        min_nominal = part_min_nominal[p]
        prec_start = prec_offsets[p]

        # Bin arrays for this partition
        bin_start = bin_meta_offsets[p]
        bin_end = bin_meta_offsets[p + 1]
        n_bins = bin_end - bin_start

        # Fragment arrays for this partition (views via offset)
        p_flat_prec_mz = all_flat_prec_mz[frag_start:frag_end]
        p_flat_deficit = all_flat_deficit[frag_start:frag_end]
        p_flat_prec_idx = all_flat_prec_idx[frag_start:frag_end]
        p_bin_offsets = all_bin_offsets[bin_start:bin_end]
        p_bin_lengths = all_bin_lengths[bin_start:bin_end]

        # Zero counter for this partition (stack-allocated via array)
        counter = np.zeros(n_prec, dtype=np.int32)

        # ── Inner query loop (same logic as _query_partition_jit) ──
        for i in range(n_ions):
            nominal = ion_nominals[i]
            query_deficit = ion_deficits[i]
            tol_u16 = ion_tols[i]

            # Query primary nominal bin
            bin_idx = nominal - min_nominal
            if 0 <= bin_idx < n_bins:
                length = p_bin_lengths[bin_idx]
                if length > 0:
                    offset = p_bin_offsets[bin_idx]
                    lo = _searchsorted_left_f32(p_flat_prec_mz, offset, offset + length, win_lo)
                    hi = _searchsorted_right_f32(p_flat_prec_mz, offset, offset + length, win_hi)
                    for j in range(lo, hi):
                        diff = p_flat_deficit[j] - query_deficit
                        if diff < 0:
                            diff = -diff
                        if diff <= tol_u16:
                            counter[p_flat_prec_idx[j]] += 1

            # Handle 1 Da boundary wrapping — low side
            if query_deficit - tol_u16 < 0:
                wrapped_deficit = 65536 + (query_deficit - tol_u16)
                wrapped_tol = -(query_deficit - tol_u16)
                bin_idx_lo = (nominal - 1) - min_nominal
                if 0 <= bin_idx_lo < n_bins:
                    length = p_bin_lengths[bin_idx_lo]
                    if length > 0:
                        offset = p_bin_offsets[bin_idx_lo]
                        lo = _searchsorted_left_f32(p_flat_prec_mz, offset, offset + length, win_lo)
                        hi = _searchsorted_right_f32(p_flat_prec_mz, offset, offset + length, win_hi)
                        for j in range(lo, hi):
                            diff = p_flat_deficit[j] - wrapped_deficit
                            if diff < 0:
                                diff = -diff
                            if diff <= wrapped_tol:
                                counter[p_flat_prec_idx[j]] += 1

            # Handle 1 Da boundary wrapping — high side
            if query_deficit + tol_u16 > 65535:
                wrapped_deficit = (query_deficit + tol_u16) - 65536
                wrapped_tol = (query_deficit + tol_u16) - 65535
                bin_idx_hi = (nominal + 1) - min_nominal
                if 0 <= bin_idx_hi < n_bins:
                    length = p_bin_lengths[bin_idx_hi]
                    if length > 0:
                        offset = p_bin_offsets[bin_idx_hi]
                        lo = _searchsorted_left_f32(p_flat_prec_mz, offset, offset + length, win_lo)
                        hi = _searchsorted_right_f32(p_flat_prec_mz, offset, offset + length, win_hi)
                        for j in range(lo, hi):
                            diff = p_flat_deficit[j] - wrapped_deficit
                            if diff < 0:
                                diff = -diff
                            if diff <= wrapped_tol:
                                counter[p_flat_prec_idx[j]] += 1

        # ── Collect passing precursors with RT filter ──
        for k in range(n_prec):
            if counter[k] >= atleast_m:
                p_rt = all_prec_rt[prec_start + k]
                if p_rt - prec_rt < rt_tol and prec_rt - p_rt < rt_tol:
                    out_indices[n_results] = all_prec_global_idx[prec_start + k]
                    n_results += 1

    return n_results


class FragmentIndex:
    """Fragment ion index for fast candidate pre-filtering."""

    def __init__(self, mz_tol_ppm: float):
        self.partitions: list = []
        self.mz_tol_ppm = mz_tol_ppm
        # Flattened arrays (populated by _finalize)
        self._finalized = False

    @classmethod
    def build(cls, library, all_keys, rt_mz, mz_tol_ppm,
              max_frags_per_partition=312_000):
        """Build a FragmentIndex from a spectrum library.

        Args:
            library: SpectrumLibraryStore — library[key]['spectrum'] is n×2 (mz, intensity),
                     library[key]['top_n'] is array of indices into spectrum.
            all_keys: list of keys into library (tuples of (mod_seq, charge, ...)).
            rt_mz: ndarray shape (len(all_keys), 2) — col 0 = calibrated RT, col 1 = calibrated precursor m/z.
                   Any precursor m/z offsets (e.g. for decoys) should be pre-applied.
            mz_tol_ppm: float — fragment m/z tolerance in ppm.
            max_frags_per_partition: int — max fragments per partition (~312K for L3 cache fit).
        """
        idx = cls(mz_tol_ppm)

        n = len(all_keys)
        if n == 0:
            idx._finalize()
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
            prec_mzs = rt_mz[p_indices, 1].astype(np.float32)

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
                idx.partitions.append({
                    'rt_lo': float(prec_rts.min()), 'rt_hi': float(prec_rts.max()),
                    'n_precursors': n_prec,
                    'precursor_global_idx': p_indices.astype(np.uint32),
                    'precursor_rt': prec_rts,
                    'flat_prec_mz': np.empty(0, dtype=np.float32),
                    'flat_deficit': np.empty(0, dtype=np.int32),
                    'flat_prec_idx': np.empty(0, dtype=np.int32),
                    'bin_offsets': np.empty(0, dtype=np.int32),
                    'bin_lengths': np.empty(0, dtype=np.int32),
                    'min_nominal': np.int32(0),
                })
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

            idx.partitions.append({
                'rt_lo': float(prec_rts.min()), 'rt_hi': float(prec_rts.max()),
                'n_precursors': n_prec,
                'precursor_global_idx': p_indices.astype(np.uint32),
                'precursor_rt': prec_rts,
                'flat_prec_mz': frag_prec_mz,
                'flat_deficit': deficit,
                'flat_prec_idx': frag_prec_idx,
                'bin_offsets': bin_offsets,
                'bin_lengths': bin_lengths,
                'min_nominal': np.int32(min_nom),
            })

        idx._finalize()
        return idx

    def _finalize(self):
        """Pack partition data into concatenated arrays for the JIT query kernel."""
        n_parts = len(self.partitions)
        if n_parts == 0:
            self.part_rt_lo = np.empty(0, dtype=np.float32)
            self.part_rt_hi = np.empty(0, dtype=np.float32)
            self.part_n_precursors = np.empty(0, dtype=np.int32)
            self.part_min_nominal = np.empty(0, dtype=np.int32)
            self.all_flat_prec_mz = np.empty(0, dtype=np.float32)
            self.all_flat_deficit = np.empty(0, dtype=np.int32)
            self.all_flat_prec_idx = np.empty(0, dtype=np.int32)
            self.flat_offsets = np.zeros(1, dtype=np.int64)
            self.all_bin_offsets = np.empty(0, dtype=np.int32)
            self.all_bin_lengths = np.empty(0, dtype=np.int32)
            self.bin_meta_offsets = np.zeros(1, dtype=np.int64)
            self.all_prec_global_idx = np.empty(0, dtype=np.uint32)
            self.all_prec_rt = np.empty(0, dtype=np.float32)
            self.prec_offsets = np.zeros(1, dtype=np.int64)
            self._total_precursors = 0
            self._finalized = True
            return

        # Partition-level scalar metadata
        self.part_rt_lo = np.array([p['rt_lo'] for p in self.partitions], dtype=np.float32)
        self.part_rt_hi = np.array([p['rt_hi'] for p in self.partitions], dtype=np.float32)
        self.part_n_precursors = np.array([p['n_precursors'] for p in self.partitions], dtype=np.int32)
        self.part_min_nominal = np.array([p['min_nominal'] for p in self.partitions], dtype=np.int32)

        # Concatenate fragment arrays with offset table
        frag_arrays_mz = [p['flat_prec_mz'] for p in self.partitions]
        frag_arrays_def = [p['flat_deficit'] for p in self.partitions]
        frag_arrays_idx = [p['flat_prec_idx'] for p in self.partitions]
        self.all_flat_prec_mz = np.concatenate(frag_arrays_mz) if any(len(a) > 0 for a in frag_arrays_mz) else np.empty(0, dtype=np.float32)
        self.all_flat_deficit = np.concatenate(frag_arrays_def) if any(len(a) > 0 for a in frag_arrays_def) else np.empty(0, dtype=np.int32)
        self.all_flat_prec_idx = np.concatenate(frag_arrays_idx) if any(len(a) > 0 for a in frag_arrays_idx) else np.empty(0, dtype=np.int32)
        self.flat_offsets = np.zeros(n_parts + 1, dtype=np.int64)
        for i in range(n_parts):
            self.flat_offsets[i + 1] = self.flat_offsets[i] + len(frag_arrays_mz[i])

        # Concatenate bin arrays with offset table
        bin_arrays_off = [p['bin_offsets'] for p in self.partitions]
        bin_arrays_len = [p['bin_lengths'] for p in self.partitions]
        self.all_bin_offsets = np.concatenate(bin_arrays_off) if any(len(a) > 0 for a in bin_arrays_off) else np.empty(0, dtype=np.int32)
        self.all_bin_lengths = np.concatenate(bin_arrays_len) if any(len(a) > 0 for a in bin_arrays_len) else np.empty(0, dtype=np.int32)
        self.bin_meta_offsets = np.zeros(n_parts + 1, dtype=np.int64)
        for i in range(n_parts):
            self.bin_meta_offsets[i + 1] = self.bin_meta_offsets[i] + len(bin_arrays_off[i])

        # Concatenate precursor arrays with offset table
        prec_arrays_gidx = [p['precursor_global_idx'] for p in self.partitions]
        prec_arrays_rt = [p['precursor_rt'] for p in self.partitions]
        self.all_prec_global_idx = np.concatenate(prec_arrays_gidx)
        self.all_prec_rt = np.concatenate(prec_arrays_rt)
        self.prec_offsets = np.zeros(n_parts + 1, dtype=np.int64)
        for i in range(n_parts):
            self.prec_offsets[i + 1] = self.prec_offsets[i] + len(prec_arrays_gidx[i])

        self._total_precursors = int(self.prec_offsets[-1])
        self._finalized = True

        # Free the partition dicts — all data is in flat arrays now
        self.partitions = None

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
        if self._total_precursors == 0:
            return np.empty(0, dtype=np.uint32)

        # Output buffer — worst case: all precursors pass
        out_indices = np.empty(self._total_precursors, dtype=np.uint32)

        n_results = _query_all_partitions_jit(
            dia_mz_array, self.mz_tol_ppm,
            np.float32(win_lo), np.float32(win_hi),
            prec_rt, rt_tol, atleast_m,
            self.part_rt_lo, self.part_rt_hi, self.part_n_precursors, self.part_min_nominal,
            self.all_flat_prec_mz, self.all_flat_deficit, self.all_flat_prec_idx,
            self.flat_offsets,
            self.all_bin_offsets, self.all_bin_lengths,
            self.bin_meta_offsets,
            self.all_prec_global_idx, self.all_prec_rt,
            self.prec_offsets,
            out_indices,
        )

        return out_indices[:n_results]
