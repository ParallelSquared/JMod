#  Copyright (c) 2026. Parallel Squared Technology Institute
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

import numpy as np
import os
from src.utils.frag_encoding import encode_frag_name, encode_frag_names, encode_frag_columns, decode_frag_names
import re


# ---------------------------------------------------------------------------
# Field name constants — used by _EntryView to route reads/writes
# ---------------------------------------------------------------------------
_SCALAR_STR_FIELDS = frozenset({
    'mod_seq', 'seq', 'protein_group', 'protein_name', 'genes', 'UniprotID',
})
_SCALAR_FLOAT_FIELDS = frozenset({
    'prec_mz', 'prec_z', 'iRT', 'IonMob',
})
_ALL_SCALAR_FIELDS = _SCALAR_STR_FIELDS | _SCALAR_FLOAT_FIELDS

_STANDARD_RESIDUES = frozenset("ACDEFGHIJKLMNOPQRSTUVWY")

# Version stamp for the .npz binary cache. Bump whenever parsing semantics
# change so stale caches are re-parsed instead of silently loaded.
STORE_VERSION = 1


class StaleStoreCacheError(ValueError):
    """Raised by :meth:`SpectrumLibraryStore.load` when the .npz cache was
    written by a different (or unstamped) parser version."""

_ALL_KNOWN_FIELDS = _ALL_SCALAR_FIELDS | frozenset({
    'spectrum', 'ordered_frags', 'ordered_frag_codes', 'frag_intensities',
    'frags', 'top_n', 'parent_idx', 'spec_frags',
})


def _ensure_frag_codes(value):
    """Convert fragment names to int32 codes if they are strings."""
    arr = np.asarray(value)
    if arr.dtype == np.int32:
        return arr
    # object or string dtype — encode from strings
    return encode_frag_names(arr)


class _TargetView:
    """Lightweight proxy that restricts iteration to target entries only.

    Shares all underlying numpy arrays with the parent store — no copies.
    ``__getitem__`` delegates to the full store so any key (including decoy)
    still works for direct lookups.

    ``copy.deepcopy`` returns a target-only ``SpectrumLibraryStore`` so that
    ``MZRTfit`` can safely mutate the copy without affecting the combined store.
    """

    __slots__ = ('_store', '_target_keys')

    def __init__(self, store):
        self._store = store
        # Build ordered list of target keys (indices < n_targets)
        self._target_keys = [
            k for k, idx in store.key_to_idx.items()
            if idx < store.n_targets
        ]

    def __len__(self):
        return len(self._target_keys)

    def __iter__(self):
        return iter(self._target_keys)

    def __contains__(self, key):
        return key in self._store.key_to_idx and self._store.key_to_idx[key] < self._store.n_targets

    def __getitem__(self, key):
        return self._store[key]

    def __setitem__(self, key, value):
        self._store[key] = value

    def keys(self):
        return iter(self._target_keys)

    def values(self):
        store = self._store
        for key in self._target_keys:
            yield store[key]

    def items(self):
        store = self._store
        for key in self._target_keys:
            yield key, store[key]

    def get(self, key, default=None):
        if key in self:
            return self._store[key]
        return default

    def to_diann_df(self):
        """Export only target entries as a DIA-NN-format Polars DataFrame."""
        return self._store.to_diann_df(n=self._store.n_targets)

    def __getattr__(self, name):
        """Delegate attribute access to the underlying store for methods
        not explicitly overridden (e.g. resolve_indices)."""
        return getattr(self._store, name)

    def __deepcopy__(self, memo):
        """Return a target-only SpectrumLibraryStore (independent copy)."""
        import copy
        s = self._store
        n = s.n_targets
        # Target-only key_to_idx
        k2i = {k: idx for k, idx in s.key_to_idx.items() if idx < n}
        # Slice spectrum data for target entries
        target_spec_total = int(s.spectrum_lengths[:n].sum())
        target_frag_total = int(s.frag_lengths[:n].sum())
        target_topn_total = int(s.top_n_lengths[:n].sum())
        return SpectrumLibraryStore(
            key_to_idx=copy.deepcopy(k2i, memo),
            mod_seq=s.mod_seq[:n].copy(),
            seq=s.seq[:n].copy(),
            prec_mz=s.prec_mz[:n].copy(),
            prec_z=s.prec_z[:n].copy(),
            iRT=s.iRT[:n].copy(),
            ion_mob=s.ion_mob[:n].copy(),
            protein_group=s.protein_group[:n].copy(),
            protein_name=s.protein_name[:n].copy(),
            genes=s.genes[:n].copy(),
            uniprot_id=s.uniprot_id[:n].copy(),
            spectrum_mz=s.spectrum_mz[:target_spec_total].copy(),
            spectrum_int=s.spectrum_int[:target_spec_total].copy(),
            spectrum_offsets=s.spectrum_offsets[:n].copy(),
            spectrum_lengths=s.spectrum_lengths[:n].copy(),
            frag_names_data=s.frag_names_data[:target_spec_total].copy(),
            frag_data=s.frag_data[:target_frag_total].copy(),
            frag_keys_data=s.frag_keys_data[:target_frag_total].copy(),
            frag_offsets=s.frag_offsets[:n].copy(),
            frag_lengths=s.frag_lengths[:n].copy(),
            top_n_data=s.top_n_data[:target_topn_total].copy(),
            top_n_offsets=s.top_n_offsets[:n].copy(),
            top_n_lengths=s.top_n_lengths[:n].copy(),
            parent_idx=s.parent_idx[:n].copy(),
        )

# Canonical column name -> priority-ordered tuple of accepted input spellings.
# Canonical names match to_diann_df's output columns. Resolution is
# select-with-alias, not rename: alias lists overlap (ProteinID feeds both
# ProteinName and ProteinID when ProteinName is absent), so one source column
# may legitimately populate more than one canonical field.
#
# Known input/output asymmetries, preserved deliberately:
# - input prefers FragmentNumber, output emits FragmentSeriesNumber
# - input prefers Tr_recalibrated, output emits RT
_LIBRARY_COLUMN_ALIASES = {
    "Decoy":                ("Decoy",),
    "ModifiedPeptide":      ("ModifiedPeptide", "ModifiedSequence", "Modified.Sequence"),
    "StrippedPeptide":      ("StrippedPeptide", "PeptideSequence", "Stripped.Sequence"),
    "PrecursorCharge":      ("PrecursorCharge", "Precursor.Charge"),
    "PrecursorMz":          ("PrecursorMz", "Precursor.Mz"),
    "RT":                   ("Tr_recalibrated", "RT", "iRT"),
    "IonMobility":          ("IonMobility", "IM"),
    "ProteinGroup":         ("ProteinGroup", "Protein.Group"),
    "ProteinName":          ("ProteinName", "Protein.Names", "ProteinID", "ProteinId"),
    "ProteinID":            ("ProteinID", "UniprotID", "Protein.Ids"),
    "Genes":                ("Genes", "GeneName"),
    "FragmentType":         ("FragmentType", "Fragment.Type"),
    "FragmentSeriesNumber": ("FragmentNumber", "FragmentSeriesNumber", "Fragment.Series.Number"),
    "FragmentCharge":       ("FragmentCharge", "Fragment.Charge"),
    "FragmentLossType":     ("FragmentLossType", "Fragment.Loss.Type"),
    "FragmentMz":           ("FragmentMz", "ProductMz", "Product.Mz"),
    "RelativeIntensity":    ("RelativeIntensity", "LibraryIntensity", "Relative.Intensity"),
}


def _scan_library_tsv(path):
    import polars as pl
    # infer_schema=False reads every column as Utf8 and
    # missing_utf8_is_empty_string keeps empty cells as "" rather than null,
    # together mirroring csv.DictReader
    return pl.scan_csv(path, separator="\t", infer_schema=False,
                       missing_utf8_is_empty_string=True)


def _scan_library_parquet(path):
    import polars as pl
    return pl.scan_parquet(path)


# Fragment-row formats: extension -> callable returning a polars LazyFrame.
# A new format only supplies bytes-to-frame; alias resolution is shared.
# (.blib carries spectra rather than fragment rows and keeps its own SQLite
# path in SpectrumLibraryStore.from_blib.)
_LIBRARY_READERS = {
    "tsv": _scan_library_tsv,
    "parquet": _scan_library_parquet,
}


def _load_library_frame(spec_lib_file, file_type):
    """Read a library file into a DataFrame with canonical column names.

    Only columns matching a known alias are read from disk. Returns
    ``(df, found)`` where ``found`` maps canonical name -> the source column
    it was resolved from. The same source column may back several canonical
    columns (see _LIBRARY_COLUMN_ALIASES).
    """
    import polars as pl
    reader = _LIBRARY_READERS.get(file_type)
    if reader is None:
        raise ValueError(f"Unsupported spectral library format: {file_type}")
    spec_lib_lazy_frame = reader(spec_lib_file)
    present = spec_lib_lazy_frame.collect_schema().names()
    found = {}
    exprs = []
    for canonical, aliases in _LIBRARY_COLUMN_ALIASES.items():
        for name in aliases:
            if name in present:
                found[canonical] = name
                exprs.append(pl.col(name).alias(canonical))
                break
    return spec_lib_lazy_frame.select(exprs).collect(), found

class SpectrumLibraryStore:
    """Columnar store for spectral library data.

    Replaces dict[tuple, dict[str, Any]] with struct-of-arrays backed by
    numpy, eliminating millions of Python dicts from the GC heap.

    Downstream code accesses entries through ``store[key]`` which returns
    a lightweight ``_EntryView`` proxy that behaves like a dict.

    **Important**: ``frags`` and ``spectrum`` are stored independently.
    ``spectrum`` may contain isotope-expanded peaks while ``frags`` retains
    the original fragment dict data.
    """

    __slots__ = (
        'key_to_idx',
        # scalar columns – strings (object arrays)
        'mod_seq', 'seq', 'protein_group', 'protein_name', 'genes', 'uniprot_id',
        # scalar columns – float64
        'prec_mz', 'prec_z', 'iRT', 'ion_mob',
        # concatenated variable-length spectrum data (may include isotopes)
        'spectrum_mz', 'spectrum_int', 'spectrum_offsets', 'spectrum_lengths',
        'frag_names_data',
        # concatenated variable-length original frags data (independent of spectrum)
        'frag_data', 'frag_keys_data', 'frag_offsets', 'frag_lengths',
        # concatenated variable-length top_n data
        'top_n_data', 'top_n_offsets', 'top_n_lengths',
        # parent_idx (int64 array, -1 for targets, parent's index for decoys)
        'parent_idx',
        # target/decoy tracking
        'n_targets', 'n_decoys', 'is_decoy',
    )

    def __init__(
        self,
        key_to_idx,
        mod_seq, seq, prec_mz, prec_z, iRT, ion_mob,
        protein_group, protein_name, genes, uniprot_id,
        spectrum_mz, spectrum_int, spectrum_offsets, spectrum_lengths,
        frag_names_data,
        frag_data, frag_keys_data, frag_offsets, frag_lengths,
        top_n_data, top_n_offsets, top_n_lengths,
        parent_idx,
        n_targets=None, n_decoys=None, is_decoy=None,
        parent_key=None,  # deprecated, ignored
    ):
        self.key_to_idx = key_to_idx
        self.mod_seq = mod_seq
        self.seq = seq
        self.prec_mz = prec_mz
        self.prec_z = prec_z
        self.iRT = iRT
        self.ion_mob = ion_mob
        self.protein_group = protein_group
        self.protein_name = protein_name
        self.genes = genes
        self.uniprot_id = uniprot_id
        self.spectrum_mz = spectrum_mz
        self.spectrum_int = spectrum_int
        self.spectrum_offsets = spectrum_offsets
        self.spectrum_lengths = spectrum_lengths
        self.frag_names_data = frag_names_data
        self.frag_data = frag_data
        self.frag_keys_data = frag_keys_data
        self.frag_offsets = frag_offsets
        self.frag_lengths = frag_lengths
        self.top_n_data = top_n_data
        self.top_n_offsets = top_n_offsets
        self.top_n_lengths = top_n_lengths
        self.parent_idx = parent_idx
        # Target/decoy tracking — defaults to all-target
        n = len(key_to_idx)
        self.n_targets = n_targets if n_targets is not None else n
        self.n_decoys = n_decoys if n_decoys is not None else 0
        self.is_decoy = is_decoy if is_decoy is not None else np.zeros(n, dtype=bool)

    # ------------------------------------------------------------------
    # Backward-compat property for spectrum_data
    # ------------------------------------------------------------------

    @property
    def target_decoy_ratio(self):
        """Ratio of n_targets / n_decoys, for FDR correction."""
        if self.n_decoys == 0:
            return float('inf')
        return self.n_targets / self.n_decoys

    def target_view(self):
        """Return a lightweight proxy that only exposes target entries.

        Shares underlying arrays — no copies.  Used to restrict the
        preliminary search to targets only.
        """
        return _TargetView(self)

    @property
    def spectrum_data(self):
        """Reconstruct (N,2) array for backward compatibility. Returns a copy."""
        return np.stack([self.spectrum_mz, self.spectrum_int], axis=1)

    @spectrum_data.setter
    def spectrum_data(self, value):
        """Accept (N,2) assignment for backward compatibility."""
        self.spectrum_mz = np.ascontiguousarray(value[:, 0])
        self.spectrum_int = np.ascontiguousarray(value[:, 1])

    # ------------------------------------------------------------------
    # Internal accessors
    # ------------------------------------------------------------------

    def get_spectrum(self, idx):
        """Return (n_peaks, 2) float64 array for entry *idx*."""
        off = self.spectrum_offsets[idx]
        length = self.spectrum_lengths[idx]
        return np.stack([self.spectrum_mz[off:off + length],
                         self.spectrum_int[off:off + length]], axis=1)

    def get_ordered_frags(self, idx):
        """Return 1-D object array of decoded fragment name strings for entry *idx*."""
        off = self.spectrum_offsets[idx]
        length = self.spectrum_lengths[idx]
        return decode_frag_names(self.frag_names_data[off:off + length])

    def get_frag_intensities(self, idx):
        """Return 1-D float64 array of fragment intensities for entry *idx*."""
        off = self.spectrum_offsets[idx]
        length = self.spectrum_lengths[idx]
        return self.spectrum_int[off:off + length]

    def get_ordered_frag_codes(self, idx):
        """Return 1-D int32 array of packed fragment name codes for entry *idx*.

        Use vectorized helpers from ``frag_encoding`` (is_b_ion, get_index,
        is_isotope, etc.) to query these codes without decoding to strings.
        """
        off = self.spectrum_offsets[idx]
        length = self.spectrum_lengths[idx]
        return self.frag_names_data[off:off + length]

    def get_frags(self, idx):
        """Reconstruct the ``frags`` dict on-the-fly from original frag data.

        Keys are decoded back to strings for backward compatibility with
        code that does string-based lookups (e.g. hyperscore_b_y).
        """
        off = self.frag_offsets[idx]
        length = self.frag_lengths[idx]
        if length == 0:
            return {}
        peaks = self.frag_data[off:off + length]
        codes = self.frag_keys_data[off:off + length]
        names = decode_frag_names(codes)
        return {str(names[i]): [float(peaks[i, 0]), float(peaks[i, 1])] for i in range(length)}

    def get_top_n(self, idx):
        """Return 1-D int32 array of top-N indices for entry *idx*."""
        off = self.top_n_offsets[idx]
        length = self.top_n_lengths[idx]
        return self.top_n_data[off:off + length]

    def set_spectrum(self, idx, spectrum_array, ordered_frags=None):
        """Replace spectrum data for entry *idx*.

        ``ordered_frags`` can be int32 codes or string names (auto-encoded).
        Does NOT modify the original frags — use ``set_frags`` for that.
        """
        spectrum_array = np.asarray(spectrum_array, dtype=np.float64)
        old_len = self.spectrum_lengths[idx]
        new_len = len(spectrum_array)
        if new_len == old_len:
            off = self.spectrum_offsets[idx]
            self.spectrum_mz[off:off + new_len] = spectrum_array[:, 0]
            self.spectrum_int[off:off + new_len] = spectrum_array[:, 1]
            if ordered_frags is not None:
                self.frag_names_data[off:off + new_len] = _ensure_frag_codes(ordered_frags)
        else:
            new_off = len(self.spectrum_mz)
            self.spectrum_mz = np.concatenate(
                [self.spectrum_mz, np.ascontiguousarray(spectrum_array[:, 0])]
            )
            self.spectrum_int = np.concatenate(
                [self.spectrum_int, np.ascontiguousarray(spectrum_array[:, 1])]
            )
            if ordered_frags is not None:
                self.frag_names_data = np.concatenate(
                    [self.frag_names_data, _ensure_frag_codes(ordered_frags)]
                )
            else:
                self.frag_names_data = np.concatenate(
                    [self.frag_names_data, np.zeros(new_len, dtype=np.int32)]
                )
            self.spectrum_offsets[idx] = new_off
            self.spectrum_lengths[idx] = new_len

    def set_top_n(self, idx, top_n_array):
        """Replace top_n data for entry *idx*."""
        top_n_array = np.asarray(top_n_array, dtype=np.int32)
        new_off = len(self.top_n_data)
        self.top_n_data = np.concatenate([self.top_n_data, top_n_array])
        self.top_n_offsets[idx] = new_off
        self.top_n_lengths[idx] = len(top_n_array)

    def bulk_set_top_n(self, n):
        """Compute and set top_n for all entries in a single pass."""
        all_top_n = []
        offsets = np.empty(len(self.key_to_idx), dtype=np.int64)
        lengths = np.empty(len(self.key_to_idx), dtype=np.int32)
        cursor = 0
        for idx in range(len(self.key_to_idx)):
            spec = self.get_spectrum(idx)
            if spec is not None and len(spec) > 0:
                tn = np.argsort(-spec[:, 1])[:n].astype(np.int32)
            else:
                tn = np.empty(0, dtype=np.int32)
            all_top_n.append(tn)
            offsets[idx] = cursor
            lengths[idx] = len(tn)
            cursor += len(tn)
        self.top_n_data = np.concatenate(all_top_n) if all_top_n else np.empty(0, dtype=np.int32)
        self.top_n_offsets = offsets
        self.top_n_lengths = lengths

    def set_frags(self, idx, frags_dict):
        """Replace frags for entry *idx* from a plain dict.

        Updates both the original frag arrays AND the spectrum/ordered_frags.
        """
        from src.utils.misc_functions import frag_to_peak
        spectrum, ordered_frags = frag_to_peak(frags_dict, return_frags=True)
        self.set_spectrum(idx, spectrum, ordered_frags)
        self._set_frag_arrays(idx, frags_dict)

    def _set_frag_arrays(self, idx, frags_dict):
        """Update the original frag storage arrays."""
        keys = list(frags_dict.keys())
        vals = np.array(list(frags_dict.values()), dtype=np.float64)
        codes = encode_frag_names(keys)
        n_frags = len(keys)

        new_off = len(self.frag_data)
        self.frag_data = np.concatenate(
            [self.frag_data, vals], axis=0
        ) if n_frags > 0 else self.frag_data
        self.frag_keys_data = np.concatenate(
            [self.frag_keys_data, codes]
        ) if n_frags > 0 else self.frag_keys_data
        self.frag_offsets[idx] = new_off
        self.frag_lengths[idx] = n_frags

    # ------------------------------------------------------------------
    # Batch accessors — bypass _EntryView for bulk reads in hot loops
    # ------------------------------------------------------------------

    def resolve_indices(self, keys):
        """Convert an iterable of keys to a list of internal integer indices."""
        k2i = self.key_to_idx
        return [k2i[k] for k in keys]

    def get_spectra_batch(self, indices):
        """Return list of (n_peaks, 2) spectrum arrays for internal indices."""
        smz, si, so, sl = self.spectrum_mz, self.spectrum_int, self.spectrum_offsets, self.spectrum_lengths
        return [np.stack([smz[so[i]:so[i] + sl[i]], si[so[i]:so[i] + sl[i]]], axis=1) for i in indices]

    def get_top_n_batch(self, indices):
        """Return list of int32 top-N index arrays for internal indices."""
        td, to, tl = self.top_n_data, self.top_n_offsets, self.top_n_lengths
        return [td[to[i]:to[i] + tl[i]] for i in indices]

    def get_frag_codes_batch(self, indices):
        """Return list of int32 frag code arrays for internal indices."""
        fd, so, sl = self.frag_names_data, self.spectrum_offsets, self.spectrum_lengths
        return [fd[so[i]:so[i] + sl[i]] for i in indices]

    def get_scalar_batch(self, indices, field):
        """Return values for a scalar field at given internal indices.

        For string fields returns a list; for float fields returns a numpy slice.
        """
        _field_map = {
            'mod_seq': self.mod_seq,
            'seq': self.seq,
            'protein_group': self.protein_group,
            'protein_name': self.protein_name,
            'genes': self.genes,
            'UniprotID': self.uniprot_id,
            'prec_mz': self.prec_mz,
            'prec_z': self.prec_z,
            'iRT': self.iRT,
            'IonMob': self.ion_mob,
        }
        arr = _field_map[field]
        return [arr[i] for i in indices]

    # ------------------------------------------------------------------
    # Dict-like interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.key_to_idx)

    def __contains__(self, key):
        return key in self.key_to_idx

    def __iter__(self):
        return iter(self.key_to_idx)

    def __getitem__(self, key):
        idx = self.key_to_idx[key]
        return _EntryView(self, idx)

    def __setitem__(self, key, value):
        """Allow ``store[key] = dict_entry`` for compatibility."""
        if isinstance(value, _EntryView) and value._store is self:
            self.key_to_idx[key] = value._idx
            return
        if key in self.key_to_idx:
            idx = self.key_to_idx[key]
            self._write_entry(idx, value)
        else:
            idx = self._append_entry(value)
            self.key_to_idx[key] = idx

    def __delitem__(self, key):
        del self.key_to_idx[key]

    def keys(self):
        return self.key_to_idx.keys()

    def values(self):
        for key in self.key_to_idx:
            yield _EntryView(self, self.key_to_idx[key])

    def items(self):
        for key, idx in self.key_to_idx.items():
            yield key, _EntryView(self, idx)

    def get(self, key, default=None):
        if key in self.key_to_idx:
            return _EntryView(self, self.key_to_idx[key])
        return default

    def setdefault(self, key, default=None):
        if key in self.key_to_idx:
            return _EntryView(self, self.key_to_idx[key])
        if default is None:
            default = {}
        self[key] = default
        return self[key]

    def pop(self, key, *args):
        if key in self.key_to_idx:
            view = _EntryView(self, self.key_to_idx[key])
            del self.key_to_idx[key]
            return view
        if args:
            return args[0]
        raise KeyError(key)

    def update(self, other):
        for key, value in other.items():
            self[key] = value

    # ------------------------------------------------------------------
    # Internal mutation helpers
    # ------------------------------------------------------------------

    def _write_entry(self, idx, entry_dict):
        """Overwrite slot *idx* from a plain dict."""
        for field in _SCALAR_STR_FIELDS:
            arr = self._str_array_for(field)
            if field in entry_dict:
                arr[idx] = entry_dict[field]
        for field in _SCALAR_FLOAT_FIELDS:
            arr = self._float_array_for(field)
            if field in entry_dict:
                val = entry_dict[field]
                arr[idx] = np.nan if val is None else float(val)

        if 'parent_idx' in entry_dict:
            self.parent_idx[idx] = entry_dict['parent_idx']

        if 'frags' in entry_dict:
            self.set_frags(idx, entry_dict['frags'])
            # If spectrum is ALSO provided separately (e.g., isotope-expanded),
            # overwrite spectrum after frags
            if 'spectrum' in entry_dict:
                self.set_spectrum(
                    idx,
                    entry_dict['spectrum'],
                    entry_dict.get('ordered_frags'),
                )
        elif 'spectrum' in entry_dict:
            self.set_spectrum(
                idx,
                entry_dict['spectrum'],
                entry_dict.get('ordered_frags'),
            )

        if 'top_n' in entry_dict:
            self.set_top_n(idx, entry_dict['top_n'])

    def _append_entry(self, entry_dict):
        """Append a new entry from a plain dict, return its index."""
        idx = len(self.prec_mz)

        # Grow scalar arrays by 1
        for field in _SCALAR_STR_FIELDS:
            arr = self._str_array_for(field)
            new_arr = np.empty(idx + 1, dtype=object)
            new_arr[:idx] = arr
            new_arr[idx] = entry_dict.get(field, '')
            self._set_str_array(field, new_arr)

        for field in _SCALAR_FLOAT_FIELDS:
            arr = self._float_array_for(field)
            new_arr = np.empty(idx + 1, dtype=np.float64)
            new_arr[:idx] = arr
            val = entry_dict.get(field)
            new_arr[idx] = np.nan if val is None else float(val)
            self._set_float_array(field, new_arr)

        # parent_idx
        new_pi = np.empty(idx + 1, dtype=np.int64)
        new_pi[:idx] = self.parent_idx
        new_pi[idx] = entry_dict.get('parent_idx', -1)
        self.parent_idx = new_pi

        # spectrum offsets / lengths
        self.spectrum_offsets = np.append(self.spectrum_offsets, len(self.spectrum_mz))
        self.spectrum_lengths = np.append(self.spectrum_lengths, np.int32(0))

        # frag offsets / lengths
        self.frag_offsets = np.append(self.frag_offsets, len(self.frag_data))
        self.frag_lengths = np.append(self.frag_lengths, np.int32(0))

        # top_n offsets / lengths
        self.top_n_offsets = np.append(self.top_n_offsets, len(self.top_n_data))
        self.top_n_lengths = np.append(self.top_n_lengths, np.int32(0))

        # Now write spectrum / frags / top_n
        if 'frags' in entry_dict:
            self.set_frags(idx, entry_dict['frags'])
            if 'spectrum' in entry_dict:
                self.set_spectrum(
                    idx,
                    entry_dict['spectrum'],
                    entry_dict.get('ordered_frags'),
                )
        elif 'spectrum' in entry_dict:
            self.set_spectrum(
                idx,
                entry_dict['spectrum'],
                entry_dict.get('ordered_frags'),
            )

        if 'top_n' in entry_dict:
            self.set_top_n(idx, entry_dict['top_n'])

        return idx

    def _str_array_for(self, field):
        _map = {
            'mod_seq': 'mod_seq', 'seq': 'seq',
            'protein_group': 'protein_group', 'protein_name': 'protein_name',
            'genes': 'genes', 'UniprotID': 'uniprot_id',
        }
        return getattr(self, _map[field])

    def _set_str_array(self, field, arr):
        _map = {
            'mod_seq': 'mod_seq', 'seq': 'seq',
            'protein_group': 'protein_group', 'protein_name': 'protein_name',
            'genes': 'genes', 'UniprotID': 'uniprot_id',
        }
        setattr(self, _map[field], arr)

    def _float_array_for(self, field):
        _map = {
            'prec_mz': 'prec_mz', 'prec_z': 'prec_z',
            'iRT': 'iRT', 'IonMob': 'ion_mob',
        }
        return getattr(self, _map[field])

    def _set_float_array(self, field, arr):
        _map = {
            'prec_mz': 'prec_mz', 'prec_z': 'prec_z',
            'iRT': 'iRT', 'IonMob': 'ion_mob',
        }
        setattr(self, _map[field], arr)

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------

    def __deepcopy__(self, memo):
        """Support ``copy.deepcopy(store)``."""
        import copy
        return SpectrumLibraryStore(
            key_to_idx=copy.deepcopy(self.key_to_idx, memo),
            mod_seq=self.mod_seq.copy(),
            seq=self.seq.copy(),
            prec_mz=self.prec_mz.copy(),
            prec_z=self.prec_z.copy(),
            iRT=self.iRT.copy(),
            ion_mob=self.ion_mob.copy(),
            protein_group=self.protein_group.copy(),
            protein_name=self.protein_name.copy(),
            genes=self.genes.copy(),
            uniprot_id=self.uniprot_id.copy(),
            spectrum_mz=self.spectrum_mz.copy(),
            spectrum_int=self.spectrum_int.copy(),
            spectrum_offsets=self.spectrum_offsets.copy(),
            spectrum_lengths=self.spectrum_lengths.copy(),
            frag_names_data=self.frag_names_data.copy(),
            frag_data=self.frag_data.copy(),
            frag_keys_data=self.frag_keys_data.copy(),
            frag_offsets=self.frag_offsets.copy(),
            frag_lengths=self.frag_lengths.copy(),
            top_n_data=self.top_n_data.copy(),
            top_n_offsets=self.top_n_offsets.copy(),
            top_n_lengths=self.top_n_lengths.copy(),
            parent_idx=self.parent_idx.copy(),
            n_targets=self.n_targets,
            n_decoys=self.n_decoys,
            is_decoy=self.is_decoy.copy(),
        )

    def __copy__(self):
        """Support ``copy.copy(store)`` — same as shallow_copy."""
        return self.shallow_copy()

    def build_key_index(self):
        """Rebuild ``key_to_idx`` from ``mod_seq`` and ``prec_z`` arrays."""
        self.key_to_idx = {}
        for i in range(len(self.mod_seq)):
            key = (self.mod_seq[i], self.prec_z[i])
            self.key_to_idx[key] = i

    def shallow_copy(self):
        """Return a new store sharing spectrum data but with independent
        scalar arrays for fields that may be mutated (iRT, parent_idx)."""
        return SpectrumLibraryStore(
            key_to_idx=dict(self.key_to_idx),
            mod_seq=self.mod_seq,
            seq=self.seq,
            prec_mz=self.prec_mz,
            prec_z=self.prec_z,
            iRT=self.iRT.copy(),
            ion_mob=self.ion_mob,
            protein_group=self.protein_group,
            protein_name=self.protein_name,
            genes=self.genes,
            uniprot_id=self.uniprot_id,
            spectrum_mz=self.spectrum_mz,
            spectrum_int=self.spectrum_int,
            spectrum_offsets=self.spectrum_offsets.copy(),
            spectrum_lengths=self.spectrum_lengths.copy(),
            frag_names_data=self.frag_names_data,
            frag_data=self.frag_data,
            frag_keys_data=self.frag_keys_data,
            frag_offsets=self.frag_offsets.copy(),
            frag_lengths=self.frag_lengths.copy(),
            top_n_data=self.top_n_data,
            top_n_offsets=self.top_n_offsets.copy(),
            top_n_lengths=self.top_n_lengths.copy(),
            parent_idx=self.parent_idx.copy(),
            n_targets=self.n_targets,
            n_decoys=self.n_decoys,
            is_decoy=self.is_decoy,
        )

    # ------------------------------------------------------------------
    # Serialization (npz binary cache)
    # ------------------------------------------------------------------

    def save(self, path):
        """Save all arrays to a ``.npz`` file."""
        np.savez(
            path,
            mod_seq=self.mod_seq,
            seq=self.seq,
            prec_mz=self.prec_mz,
            prec_z=self.prec_z,
            iRT=self.iRT,
            ion_mob=self.ion_mob,
            protein_group=self.protein_group,
            protein_name=self.protein_name,
            genes=self.genes,
            uniprot_id=self.uniprot_id,
            spectrum_mz=self.spectrum_mz,
            spectrum_int=self.spectrum_int,
            spectrum_offsets=self.spectrum_offsets,
            spectrum_lengths=self.spectrum_lengths,
            frag_names_data=self.frag_names_data,
            frag_data=self.frag_data,
            frag_keys_data=self.frag_keys_data,
            frag_offsets=self.frag_offsets,
            frag_lengths=self.frag_lengths,
            top_n_data=self.top_n_data,
            top_n_offsets=self.top_n_offsets,
            top_n_lengths=self.top_n_lengths,
            parent_idx=self.parent_idx,
            n_targets=np.array(self.n_targets),
            n_decoys=np.array(self.n_decoys),
            is_decoy=self.is_decoy,
            store_version=np.array(STORE_VERSION),
        )

    @classmethod
    def load(cls, path):
        """Load from a ``.npz`` file, rebuild key index.

        Handles both the current format (with separate frag arrays) and
        older caches (without them) by falling back to spectrum data.
        Also handles older caches with object-dtype frag name arrays
        by re-encoding them as int32.
        """
        data = np.load(path, allow_pickle=True)

        cached_version = int(data['store_version']) if 'store_version' in data else None
        if cached_version != STORE_VERSION:
            raise StaleStoreCacheError(
                f"Cache {path} has store_version={cached_version}, "
                f"expected {STORE_VERSION}; re-parse the library."
            )

        # Handle older caches that lack separate frag arrays
        if 'frag_data' in data:
            frag_data = data['frag_data']
            frag_keys_data = data['frag_keys_data']
            frag_offsets = data['frag_offsets']
            frag_lengths = data['frag_lengths']
        else:
            # Fall back: use spectrum data as frag data (pre-isotope state)
            if 'spectrum_data' in data:
                frag_data = data['spectrum_data']
            else:
                frag_data = np.stack([data['spectrum_mz'], data['spectrum_int']], axis=1)
            frag_keys_data = data['frag_names_data']
            frag_offsets = data['spectrum_offsets']
            frag_lengths = data['spectrum_lengths']

        # Load spectrum_mz / spectrum_int (new format) or fall back to spectrum_data (old format)
        if 'spectrum_mz' in data:
            spectrum_mz = data['spectrum_mz']
            spectrum_int = data['spectrum_int']
        else:
            sd = data['spectrum_data']
            spectrum_mz = np.ascontiguousarray(sd[:, 0]) if len(sd) > 0 else np.empty(0, dtype=np.float64)
            spectrum_int = np.ascontiguousarray(sd[:, 1]) if len(sd) > 0 else np.empty(0, dtype=np.float64)

        # Re-encode old object-dtype caches to int32
        frag_names_data = data['frag_names_data']
        if frag_names_data.dtype == object and len(frag_names_data) > 0:
            frag_names_data = encode_frag_names(frag_names_data)
        elif frag_names_data.dtype != np.int32:
            frag_names_data = frag_names_data.astype(np.int32)

        if frag_keys_data.dtype == object and len(frag_keys_data) > 0:
            frag_keys_data = encode_frag_names(frag_keys_data)
        elif frag_keys_data.dtype != np.int32:
            frag_keys_data = frag_keys_data.astype(np.int32)

        # Handle target/decoy fields (may be absent in old caches)
        n_total = len(data['mod_seq'])
        if 'n_targets' in data:
            n_targets = int(data['n_targets'])
            n_decoys = int(data['n_decoys'])
            is_decoy_arr = data['is_decoy']
        else:
            n_targets = n_total
            n_decoys = 0
            is_decoy_arr = np.zeros(n_total, dtype=bool)

        store = cls(
            key_to_idx={},
            mod_seq=data['mod_seq'],
            seq=data['seq'],
            prec_mz=data['prec_mz'],
            prec_z=data['prec_z'],
            iRT=data['iRT'],
            ion_mob=data['ion_mob'],
            protein_group=data['protein_group'],
            protein_name=data['protein_name'],
            genes=data['genes'],
            uniprot_id=data['uniprot_id'],
            spectrum_mz=spectrum_mz,
            spectrum_int=spectrum_int,
            spectrum_offsets=data['spectrum_offsets'],
            spectrum_lengths=data['spectrum_lengths'],
            frag_names_data=frag_names_data,
            frag_data=frag_data,
            frag_keys_data=frag_keys_data,
            frag_offsets=frag_offsets,
            frag_lengths=frag_lengths,
            top_n_data=data['top_n_data'],
            top_n_offsets=data['top_n_offsets'],
            top_n_lengths=data['top_n_lengths'],
            parent_idx=data['parent_idx'] if 'parent_idx' in data else np.full(n_total, -1, dtype=np.int64),
            n_targets=n_targets,
            n_decoys=n_decoys,
            is_decoy=is_decoy_arr,
        )
        store.build_key_index()
        return store

    # ------------------------------------------------------------------
    # Factory: from existing dict-of-dicts
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, python_lib):
        """Convert an existing ``dict[tuple, dict]`` library to columnar form."""
        n = len(python_lib)
        if n == 0:
            return cls._empty()

        keys = list(python_lib.keys())

        # Pre-allocate scalar lists
        mod_seq_list = []
        seq_list = []
        prec_mz_list = []
        prec_z_list = []
        iRT_list = []
        ion_mob_list = []
        protein_group_list = []
        protein_name_list = []
        genes_list = []
        uniprot_id_list = []
        parent_idx_list = []

        # Variable-length accumulators for spectrum
        all_spec_peaks = []
        all_spec_frag_names = []
        spec_offsets = []
        spec_lengths = []
        spec_cursor = 0

        # Variable-length accumulators for original frags
        all_frag_peaks = []
        all_frag_keys = []
        frag_offsets = []
        frag_lengths = []
        frag_cursor = 0

        # top_n
        top_n_all = []
        top_n_offsets = []
        top_n_lengths = []
        top_n_cursor = 0

        key_to_idx = {}

        for i, key in enumerate(keys):
            entry = python_lib[key]
            key_to_idx[key] = i

            mod_seq_list.append(entry.get('mod_seq', key[0] if isinstance(key, tuple) else ''))
            seq_list.append(entry.get('seq', ''))
            prec_mz_list.append(float(entry['prec_mz']) if 'prec_mz' in entry else np.nan)
            prec_z_list.append(float(entry['prec_z']) if 'prec_z' in entry else (float(key[1]) if isinstance(key, tuple) else np.nan))

            irt = entry.get('iRT')
            iRT_list.append(np.nan if irt is None else float(irt))

            imob = entry.get('IonMob')
            ion_mob_list.append(np.nan if imob is None else float(imob))

            protein_group_list.append(entry.get('protein_group', ''))
            protein_name_list.append(entry.get('protein_name', ''))
            genes_list.append(entry.get('genes', ''))
            uniprot_id_list.append(entry.get('UniprotID', ''))
            parent_idx_list.append(entry.get('parent_idx', -1))

            # Original frags data
            frags = entry.get('frags')
            frag_offsets.append(frag_cursor)
            if frags:
                frag_keys_arr = encode_frag_names(list(frags.keys()))
                frag_vals_arr = np.array(list(frags.values()), dtype=np.float64)
                n_frags = len(frag_keys_arr)
                all_frag_keys.append(frag_keys_arr)
                all_frag_peaks.append(frag_vals_arr)
                frag_lengths.append(n_frags)
                frag_cursor += n_frags
            else:
                frag_lengths.append(0)

            # Spectrum data
            spec_offsets.append(spec_cursor)
            if 'spectrum' in entry and entry['spectrum'] is not None:
                spec = np.asarray(entry['spectrum'], dtype=np.float64)
                n_peaks = len(spec)
                all_spec_peaks.append(spec)
                if 'ordered_frags' in entry and entry['ordered_frags'] is not None:
                    of_raw = entry['ordered_frags']
                    # Accept both pre-encoded int32 and string arrays
                    if hasattr(of_raw, 'dtype') and of_raw.dtype == np.int32:
                        all_spec_frag_names.append(of_raw)
                    else:
                        all_spec_frag_names.append(encode_frag_names(of_raw))
                elif frags:
                    from src.utils.misc_functions import frag_to_peak
                    _, of = frag_to_peak(frags, return_frags=True)
                    all_spec_frag_names.append(encode_frag_names(of))
                else:
                    all_spec_frag_names.append(np.zeros(n_peaks, dtype=np.int32))
            elif frags:
                from src.utils.misc_functions import frag_to_peak
                spec, of = frag_to_peak(frags, return_frags=True)
                n_peaks = len(spec)
                all_spec_peaks.append(spec)
                all_spec_frag_names.append(encode_frag_names(of))
            else:
                n_peaks = 0
            spec_lengths.append(n_peaks)
            spec_cursor += n_peaks

            # Top N data
            top_n_offsets.append(top_n_cursor)
            if 'top_n' in entry and entry['top_n'] is not None:
                tn = np.asarray(entry['top_n'], dtype=np.int32)
                top_n_all.append(tn)
                top_n_lengths.append(len(tn))
                top_n_cursor += len(tn)
            else:
                top_n_lengths.append(0)

        # Build numpy arrays
        if all_spec_peaks:
            _sd = np.concatenate(all_spec_peaks, axis=0)
            spectrum_mz = np.ascontiguousarray(_sd[:, 0])
            spectrum_int = np.ascontiguousarray(_sd[:, 1])
        else:
            spectrum_mz = np.empty(0, dtype=np.float64)
            spectrum_int = np.empty(0, dtype=np.float64)
        frag_names_data = np.concatenate(all_spec_frag_names) if all_spec_frag_names else np.empty(0, dtype=np.int32)
        frag_data = np.concatenate(all_frag_peaks, axis=0) if all_frag_peaks else np.empty((0, 2), dtype=np.float64)
        frag_keys_data = np.concatenate(all_frag_keys) if all_frag_keys else np.empty(0, dtype=np.int32)
        top_n_data = np.concatenate(top_n_all) if top_n_all else np.empty(0, dtype=np.int32)

        # TODO: Sort library entries by calibrated RT before building spectrum_data.
        #       This aligns physical memory layout with the RT-ordered access pattern
        #       from fragment index queries, improving L3 cache locality when multiple
        #       threads read nearby RT neighborhoods simultaneously.

        return cls(
            key_to_idx=key_to_idx,
            mod_seq=np.array(mod_seq_list, dtype=object),
            seq=np.array(seq_list, dtype=object),
            prec_mz=np.array(prec_mz_list, dtype=np.float64),
            prec_z=np.array(prec_z_list, dtype=np.float64),
            iRT=np.array(iRT_list, dtype=np.float64),
            ion_mob=np.array(ion_mob_list, dtype=np.float64),
            protein_group=np.array(protein_group_list, dtype=object),
            protein_name=np.array(protein_name_list, dtype=object),
            genes=np.array(genes_list, dtype=object),
            uniprot_id=np.array(uniprot_id_list, dtype=object),
            spectrum_mz=spectrum_mz,
            spectrum_int=spectrum_int,
            spectrum_offsets=np.array(spec_offsets, dtype=np.int64),
            spectrum_lengths=np.array(spec_lengths, dtype=np.int32),
            frag_names_data=frag_names_data,
            frag_data=frag_data,
            frag_keys_data=frag_keys_data,
            frag_offsets=np.array(frag_offsets, dtype=np.int64),
            frag_lengths=np.array(frag_lengths, dtype=np.int32),
            top_n_data=top_n_data,
            top_n_offsets=np.array(top_n_offsets, dtype=np.int64),
            top_n_lengths=np.array(top_n_lengths, dtype=np.int32),
            parent_idx=np.array(parent_idx_list, dtype=np.int64),
        )

    # ------------------------------------------------------------------
    # Factory: build decoy store from target + worker results
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Factory: columnar decoy generation (no worker pool, no dicts)
    # ------------------------------------------------------------------

    @staticmethod
    def _decoy_sequences_and_token_masses(target_store, rules, tag):
        """Per-precursor Python pass: decoy sequence strings, collision
        removal, and flat per-token prefix/suffix mass sums.

        prefix[k] = masses of first k+1 tokens summed left-to-right (the
        exact fast_mass accumulation order, so b ions are bit-identical);
        suffix[k] = last k+1 tokens (y ions, float-noise-level differences).
        """
        from src.utils.parse_peptides import (
            parse_peptide, decoy_permutation, change_seq, extract_mod,
        )
        from pyteomics.mass import std_aa_mass
        from src import config
        from src.logger import logger
        from array import array
        from types import SimpleNamespace
        import tqdm

        all_keys = list(target_store.keys())
        N = len(all_keys)
        existing_keys = set(target_store.key_to_idx.keys())
        seen_decoy_keys = set()
        tag_re = re.compile(f"(\\({tag.name}.*?\\))") if tag else None
        close_d = {"[": "]", "(": ")"}
        token_cache = {}  # stripped token -> (aa base mass, summed mod mass)

        def token_masses(tok):
            cached = token_cache.get(tok)
            if cached is None:
                mods = [m.strip(m[0] + close_d[m[0]]) for m in extract_mod(tok)]
                mod_mass = sum([config.diann_mods[j] for j in mods if j in config.diann_mods])
                cached = (std_aa_mass[tok[0]], mod_mass)
                token_cache[tok] = cached
            return cached

        n_target_collisions = 0
        n_decoy_collisions = 0
        valid_indices = []          # target index per surviving decoy
        decoy_mod_seqs = []
        decoy_keys = []
        token_lengths = []
        prefix_aa = array('d'); suffix_aa = array('d')
        prefix_mod = array('d'); suffix_mod = array('d')
        prefix_tag = array('d'); suffix_tag = array('d')

        for i in tqdm.tqdm(range(N), desc="Generating decoys", miniters=10000):
            mod_seq_i = target_store.mod_seq[i]
            new_seq = change_seq(mod_seq_i, rules, tag=tag)
            decoy_key = (new_seq, *all_keys[i][1:])
            if decoy_key in existing_keys:
                n_target_collisions += 1
                continue
            if decoy_key in seen_decoy_keys:
                n_decoy_collisions += 1
                continue
            seen_decoy_keys.add(decoy_key)

            tokens = parse_peptide(mod_seq_i)
            if tag:
                tag_lists = [tag_re.findall(t) for t in tokens]
                stripped = [tag_re.sub("", t) for t in tokens]
            else:
                stripped = tokens
            perm = decoy_permutation(stripped, rules)
            new_tokens = [stripped[j] for j in perm]
            L = len(new_tokens)

            if tag:
                # N-terminal tag pinned to position 0; others follow their
                # residue (same rule as change_seq)
                n_term_tag = tag_lists[0][:1]
                residue_tags_at_0 = tag_lists[0][1:]
                new_tags = [tag_lists[j] if j > 0 else residue_tags_at_0 for j in perm]
                new_tags[0] = new_tags[0] + n_term_tag
                tag_mass = [
                    sum([tag.mass_dict[t.strip("()")] for t in grp
                         if t.strip("()") in tag.mass_dict])
                    for grp in new_tags
                ]

            valid_indices.append(i)
            decoy_mod_seqs.append(new_seq)
            decoy_keys.append(decoy_key)
            token_lengths.append(L)

            aa = [0.0] * L
            mod = [0.0] * L
            for j, tok in enumerate(new_tokens):
                aa[j], mod[j] = token_masses(tok)

            pa = pm = pt = 0.0
            for j in range(L):
                pa += aa[j]; prefix_aa.append(pa)
                pm += mod[j]; prefix_mod.append(pm)
                if tag:
                    pt += tag_mass[j]; prefix_tag.append(pt)
            sa = sm = st = 0.0
            for j in range(L - 1, -1, -1):
                sa += aa[j]; suffix_aa.append(sa)
                sm += mod[j]; suffix_mod.append(sm)
                if tag:
                    st += tag_mass[j]; suffix_tag.append(st)

        if n_target_collisions > 0 or n_decoy_collisions > 0:
            logger.info(
                f"Decoy collision removal: {n_target_collisions} matched target keys, "
                f"{n_decoy_collisions} duplicate decoy keys discarded ({len(valid_indices)} decoys kept)"
            )

        M = len(valid_indices)
        token_lengths = np.array(token_lengths, dtype=np.int64)
        token_offsets = np.zeros(M, dtype=np.int64)
        if M > 1:
            token_offsets[1:] = np.cumsum(token_lengths[:-1])
        return SimpleNamespace(
            all_keys=all_keys,
            valid_indices=np.array(valid_indices, dtype=np.intp),
            mod_seqs=decoy_mod_seqs,
            keys=decoy_keys,
            token_offsets=token_offsets,
            prefix_aa=np.frombuffer(prefix_aa, dtype=np.float64),
            suffix_aa=np.frombuffer(suffix_aa, dtype=np.float64),
            prefix_mod=np.frombuffer(prefix_mod, dtype=np.float64),
            suffix_mod=np.frombuffer(suffix_mod, dtype=np.float64),
            prefix_tag=np.frombuffer(prefix_tag, dtype=np.float64) if tag else None,
            suffix_tag=np.frombuffer(suffix_tag, dtype=np.float64) if tag else None,
        )

    @staticmethod
    def _decoy_fragment_mz(target_store, decoys, tag):
        """Vectorized decoy fragment m/z: gather codes/intensities from the
        target's arrays (decoy fragment names are the target's) and recompute
        masses in place, preserving fast_mass + convert_frags' exact op order:

            ((sum + H2O) + delta_ion + z*H+)/z  -  loss/z  +  (tag_sum + mod_sum)/z
        """
        from pyteomics.mass import std_ion_comp, nist_mass, calculate_mass
        from src.utils.frag_encoding import get_ion_type, get_index, get_charge, get_loss

        H2O = nist_mass['H'][0][0] * 2 + nist_mass['O'][0][0]
        PROTON = nist_mass['H+'][0][0]
        ion_delta = {
            it: sum(nist_mass[el][0][0] * num for el, num in std_ion_comp[it].items())
            for it in ('b', 'y')
        }
        # Loss codes 0-3 are the only encodable losses (frag_encoding._INT_TO_LOSS)
        loss_table = np.array([0.0, calculate_mass('H2O'), calculate_mass('NH3'),
                               calculate_mass('H3PO4')], dtype=np.float64)

        M = len(decoys.valid_indices)
        d_frag_counts = target_store.frag_lengths[decoys.valid_indices].astype(np.int64)
        total_frags = int(d_frag_counts.sum())
        gather = np.arange(total_frags, dtype=np.int64)
        if M > 0:
            gather -= np.repeat(np.concatenate(([0], np.cumsum(d_frag_counts[:-1]))), d_frag_counts)
            gather += np.repeat(target_store.frag_offsets[decoys.valid_indices], d_frag_counts)

        codes = target_store.frag_keys_data[gather]
        intensities = target_store.frag_data[gather, 1]
        del gather

        ion = get_ion_type(codes)
        if total_frags > 0 and not np.all((ion == 0) | (ion == 1)):
            raise ValueError("Decoy generation only supports b/y fragment ions")
        is_b = ion == 0
        del ion

        frag_prec = np.repeat(np.arange(M, dtype=np.int64), d_frag_counts)
        # index into the [prefix | suffix] concatenations: prefix half for
        # b ions, suffix half for y ions
        tok_idx = decoys.token_offsets[frag_prec] + (get_index(codes).astype(np.int64) - 1)
        tok_idx[~is_b] += len(decoys.prefix_aa)

        z = get_charge(codes).astype(np.float64)
        m = np.concatenate([decoys.prefix_aa, decoys.suffix_aa])[tok_idx]
        m += H2O
        m[is_b] += ion_delta['b']
        m[~is_b] += ion_delta['y']
        tmp = z * PROTON
        m += tmp
        m /= z
        np.take(loss_table, get_loss(codes), out=tmp)
        tmp /= z
        m -= tmp
        tmp = np.concatenate([decoys.prefix_mod, decoys.suffix_mod])[tok_idx]
        if tag:
            tag_sum = np.concatenate([decoys.prefix_tag, decoys.suffix_tag])[tok_idx]
            tag_sum += tmp
            tmp = tag_sum
        tmp /= z
        m += tmp
        return d_frag_counts, codes, intensities, m, frag_prec

    @classmethod
    def from_target_with_decoys(cls, target_store, rules, tag=None):
        """Build a combined target+decoy store columnar-style.

        Decoy sequences come from the same decoy_permutation / change_seq
        logic as the legacy worker path, so the generated decoy strings and
        collision removal are identical. Fragment codes and intensities are
        gathered from the target's arrays; only m/z is recomputed (b ions
        bit-identical, y ions within float noise). Targets occupy [0, N),
        decoys [N, N+M).

        CONSUMES target_store: each large target array is dropped right
        after being copied into the combined store, so peak memory is roughly
        one large array over the combined size instead of target + combined.
        The input store is unusable afterward; every caller rebinds.
        """
        import tqdm

        decoys = cls._decoy_sequences_and_token_masses(target_store, rules, tag)
        all_keys = decoys.all_keys
        valid_indices = decoys.valid_indices
        N = len(all_keys)
        M = len(valid_indices)

        progress = tqdm.tqdm(total=3, desc="Building decoy store", unit="stage", leave=False)

        def _stage(name):
            progress.set_postfix_str(name)
            progress.update(1)

        d_frag_counts, codes, intensities, d_frag_mz, frag_prec = \
            cls._decoy_fragment_mz(target_store, decoys, tag)
        d_frag_lengths = d_frag_counts.astype(np.int32)
        total_frags = len(codes)
        _stage("computed decoy m/z")

        # --- Assemble combined store ---
        key_to_idx = {}
        for i, key in enumerate(all_keys):
            key_to_idx[key] = i
        for j, dkey in enumerate(decoys.keys):
            key_to_idx[dkey] = N + j

        decoy_mod_seq_arr = np.array(decoys.mod_seqs, dtype=object)
        decoy_seq_arr = np.array(
            [re.sub(r'\(.*?\)', '', s) if '(' in s else s for s in decoys.mod_seqs],
            dtype=object)

        mod_seq = np.concatenate([target_store.mod_seq, decoy_mod_seq_arr])
        seq = np.concatenate([target_store.seq, decoy_seq_arr])
        prec_mz = np.concatenate([target_store.prec_mz, target_store.prec_mz[valid_indices]])
        prec_z = np.concatenate([target_store.prec_z, target_store.prec_z[valid_indices]])
        iRT = np.concatenate([target_store.iRT, target_store.iRT[valid_indices]])
        ion_mob = np.concatenate([target_store.ion_mob, target_store.ion_mob[valid_indices]])
        protein_group = np.concatenate([target_store.protein_group, target_store.protein_group[valid_indices]])
        protein_name = np.concatenate([target_store.protein_name, target_store.protein_name[valid_indices]])
        genes = np.concatenate([target_store.genes, target_store.genes[valid_indices]])
        uniprot_id = np.concatenate([target_store.uniprot_id, target_store.uniprot_id[valid_indices]])

        parent_idx = np.full(N + M, -1, dtype=np.int64)
        parent_idx[N:] = valid_indices

        # Frag arrays (target order preserved per decoy); final arrays are
        # allocated once and filled by slice to avoid intermediate copies
        target_total_frag = int(target_store.frag_lengths.sum()) if N > 0 else 0
        d_frag_offsets = np.empty(M, dtype=np.int64)
        if M > 0:
            d_frag_offsets[0] = target_total_frag
            if M > 1:
                np.cumsum(d_frag_counts[:-1], out=d_frag_offsets[1:])
                d_frag_offsets[1:] += target_total_frag
        frag_offsets = np.concatenate([target_store.frag_offsets, d_frag_offsets])
        frag_lengths = np.concatenate([target_store.frag_lengths, d_frag_lengths])
        frag_keys_data = np.concatenate([target_store.frag_keys_data, codes])
        target_store.frag_keys_data = None
        frag_data = np.empty((target_total_frag + total_frags, 2), dtype=np.float64)
        frag_data[:target_total_frag] = target_store.frag_data
        frag_data[target_total_frag:, 0] = d_frag_mz
        frag_data[target_total_frag:, 1] = intensities
        target_store.frag_data = None
        _stage("assembled arrays")

        # Spectrum: per-decoy stable mz sort via one global lexsort, matching
        # frag_to_peak's kind="stable" (lexsort is stable, so equal m/z keeps
        # target frag order)
        perm_spec = np.lexsort((d_frag_mz, frag_prec))
        del frag_prec
        target_total_spec = int(target_store.spectrum_lengths.sum()) if N > 0 else 0
        d_spec_offsets = d_frag_offsets - target_total_frag + target_total_spec
        spectrum_offsets = np.concatenate([target_store.spectrum_offsets, d_spec_offsets])
        spectrum_lengths = np.concatenate([target_store.spectrum_lengths, d_frag_lengths])

        total_spec = target_total_spec + total_frags
        spectrum_mz = np.empty(total_spec, dtype=np.float64)
        spectrum_mz[:target_total_spec] = target_store.spectrum_mz
        np.take(d_frag_mz, perm_spec, out=spectrum_mz[target_total_spec:])
        target_store.spectrum_mz = None
        del d_frag_mz
        spectrum_int = np.empty(total_spec, dtype=np.float64)
        spectrum_int[:target_total_spec] = target_store.spectrum_int
        np.take(intensities, perm_spec, out=spectrum_int[target_total_spec:])
        target_store.spectrum_int = None
        del intensities
        frag_names_data = np.empty(total_spec, dtype=np.int32)
        frag_names_data[:target_total_spec] = target_store.frag_names_data
        np.take(codes, perm_spec, out=frag_names_data[target_total_spec:])
        target_store.frag_names_data = None
        _stage("sorted decoy spectra")
        progress.close()

        # Top-N: target top_n + empty for decoys (recomputed by bulk_set_top_n)
        top_n_data = target_store.top_n_data.copy()
        top_n_offsets = np.concatenate([
            target_store.top_n_offsets, np.full(M, len(top_n_data), dtype=np.int64)])
        top_n_lengths = np.concatenate([
            target_store.top_n_lengths, np.zeros(M, dtype=np.int32)])

        is_decoy = np.zeros(N + M, dtype=bool)
        is_decoy[N:] = True

        return cls(
            key_to_idx=key_to_idx,
            mod_seq=mod_seq,
            seq=seq,
            prec_mz=prec_mz,
            prec_z=prec_z,
            iRT=iRT,
            ion_mob=ion_mob,
            protein_group=protein_group,
            protein_name=protein_name,
            genes=genes,
            uniprot_id=uniprot_id,
            spectrum_mz=spectrum_mz,
            spectrum_int=spectrum_int,
            spectrum_offsets=spectrum_offsets,
            spectrum_lengths=spectrum_lengths,
            frag_names_data=frag_names_data,
            frag_data=frag_data,
            frag_keys_data=frag_keys_data,
            frag_offsets=frag_offsets,
            frag_lengths=frag_lengths,
            top_n_data=top_n_data,
            top_n_offsets=top_n_offsets,
            top_n_lengths=top_n_lengths,
            parent_idx=parent_idx,
            n_targets=N,
            n_decoys=M,
            is_decoy=is_decoy,
        )

    # ------------------------------------------------------------------
    # Factory: build tagged store from target + mass tag
    # ------------------------------------------------------------------

    @classmethod
    def from_tagged(cls, target_store, tag, source_channel=None):
        """Build a tagged SpectrumLibraryStore by pre-allocating arrays.

        For each entry in *target_store*, creates M copies (one per tag
        channel), with modified ``mod_seq``, ``prec_mz``, and fragment
        m/z values.  No deepcopy, no intermediate dicts.

        Parameters
        ----------
        target_store : SpectrumLibraryStore
            The untagged library. CONSUMED: its large arrays are released
            as soon as they are no longer needed, so the M-channel output
            does not coexist with a full copy of the input. Every caller
            rebinds.
        tag : massTag
            Mass tag with ``channel_names``, ``channel_masses``,
            ``rules``, ``name``.
        """
        from src.utils.parse_peptides import parse_peptide
        from src.mass_tags import get_tag_pos
        from src.utils.frag_encoding import get_ion_type, get_index, get_charge
        from src.logger import logger
        import tqdm

        logger.info(f"Building tagged library (pre-allocated) with tag: {tag.name}")

        # Tagged spectra are rebuilt from frag_data below; the input's
        # spectrum arrays are never read, so release them before the
        # M-channel output is allocated
        target_store.spectrum_mz = None
        target_store.spectrum_int = None
        target_store.frag_names_data = None

        if source_channel:
            source_channel_mass = tag.mass_dict[source_channel]
        else:
            source_channel_mass = 0


        N = len(target_store.key_to_idx)
        M = tag.n_channels

        # --- Phase 1: Pre-compute per-entry tag info ---
        total_target_frag = len(target_store.frag_data)
        frag_n_tags = np.empty(total_target_frag, dtype=np.float64)
        n_tag_sites = np.empty(N, dtype=np.float64)
        tagged_templates = np.empty(N, dtype=object)

        # Pre-compute frag charges for all frags at once
        if total_target_frag > 0:
            all_frag_charges = get_charge(
                target_store.frag_keys_data
            ).astype(np.float64)
        else:
            all_frag_charges = np.empty(0, dtype=np.float64)

        if source_channel is None:
            logger.info("Computing tag positions")
        for i in tqdm.tqdm(range(N)):
            mod_seq_str = target_store.mod_seq[i]
            peptide = (
                mod_seq_str
                if isinstance(mod_seq_str, str)
                else "".join(mod_seq_str)
            )

            if source_channel is not None:
                src_channel_name = source_channel.replace(tag.name + "-", "")
                strip_annotation = "(" + tag.name + "-" + src_channel_name + ")"
                # Relabel to generic tag.name, preserving exact positions
                tagged_templates[i] = peptide.replace(strip_annotation, "(" + tag.name + ")")
                
                # Derive tag positions from what's actually in the sequence
                split_peptide = parse_peptide(tagged_templates[i])
                actual_tag_mask = np.array([res.count("(" + tag.name + ")") for res in split_peptide], dtype=int)
                n_tag_sites[i] = actual_tag_mask.sum()
                num_tags_n = np.cumsum(actual_tag_mask)
                num_tags_c = np.cumsum(actual_tag_mask[::-1])
            else:
                split_peptide = parse_peptide(peptide)
                all_tag_pos, additional_tag_masses = get_tag_pos(split_peptide, tag.rules)
                n_tag_sites[i] = len(all_tag_pos)
                num_tags_n = np.cumsum(additional_tag_masses, dtype=int)
                num_tags_c = np.cumsum(additional_tag_masses[::-1], dtype=int)
                for pos in all_tag_pos:
                    split_peptide[pos] += "(" + tag.name + ")"
                tagged_templates[i] = "".join(split_peptide)

            foff = int(target_store.frag_offsets[i])
            flen = int(target_store.frag_lengths[i])
            if flen > 0:
                codes = target_store.frag_keys_data[foff:foff + flen]
                ion_types = get_ion_type(codes)
                indices = get_index(codes)
                local = np.empty(flen, dtype=np.float64)
                # N-terminal ions: b=0, a=2, c=3
                n_term = (
                    (ion_types == 0) | (ion_types == 2) | (ion_types == 3)
                )
                c_term = ~n_term  # y=1, x=4, z=5
                if n_term.any():
                    local[n_term] = num_tags_n[
                        indices[n_term] - 1
                    ].astype(np.float64)
                if c_term.any():
                    local[c_term] = num_tags_c[
                        indices[c_term] - 1
                    ].astype(np.float64)
                frag_n_tags[foff:foff + flen] = local

        # --- Phase 2: Pre-compute keys and filter collisions ---
        # Peptides with zero tag sites produce identical keys across channels.
        # Deduplicate so key_to_idx and arrays stay in sync.
        idx_to_key = {v: k for k, v in target_store.key_to_idx.items()}

        # valid: list of (entry_idx, channel_idx, new_seq, orig_charge)
        valid = []
        seen_keys = set()
        n_collisions = 0
        for i in range(N):
            orig_charge = idx_to_key[i][1]
            for c in range(M):
                tag_n = tag.channel_names[c]
                replacement = tag.name + "-" + str(tag_n)
                new_seq = tagged_templates[i].replace(
                    tag.name, replacement
                )
                key = (new_seq, orig_charge)
                if key in seen_keys:
                    n_collisions += 1
                else:
                    seen_keys.add(key)
                    valid.append((i, c, new_seq, orig_charge))

        V = len(valid)
        if n_collisions > 0:
            logger.info(
                f"Tag collision removal: {n_collisions} duplicate tagged keys "
                f"discarded ({V} entries kept)"
            )

        # --- Phase 3: Pre-allocate output arrays ---
        # Compute total frag length for valid entries only
        total_out_frag = sum(
            int(target_store.frag_lengths[i]) for i, _, _, _ in valid
        )

        # Scalar arrays
        out_mod_seq = np.empty(V, dtype=object)
        out_seq = np.empty(V, dtype=object)
        out_prec_mz = np.empty(V, dtype=np.float64)
        out_prec_z = np.empty(V, dtype=np.float64)
        out_iRT = np.empty(V, dtype=np.float64)
        out_ion_mob = np.empty(V, dtype=np.float64)
        out_protein_group = np.empty(V, dtype=object)
        out_protein_name = np.empty(V, dtype=object)
        out_genes = np.empty(V, dtype=object)
        out_uniprot_id = np.empty(V, dtype=object)
        out_parent_idx = np.full(V, -1, dtype=np.int64)
        # Map (source_index, channel) → output_index for resolving parent indices
        source_channel_to_out_idx = {}

        # Variable-length arrays
        out_spectrum_mz = np.empty(total_out_frag, dtype=np.float64)
        out_spectrum_int = np.empty(total_out_frag, dtype=np.float64)
        out_frag_names_data = np.empty(total_out_frag, dtype=np.int32)
        out_frag_data = np.empty((total_out_frag, 2), dtype=np.float64)
        out_frag_keys_data = np.empty(total_out_frag, dtype=np.int32)
        out_spec_offsets = np.empty(V, dtype=np.int64)
        out_spec_lengths = np.empty(V, dtype=np.int32)
        out_frag_offsets = np.empty(V, dtype=np.int64)
        out_frag_lengths = np.empty(V, dtype=np.int32)

        # Fill scalar arrays and build key_to_idx
        key_to_idx = {}
        for out_idx, (i, c, new_seq, orig_charge) in enumerate(valid):
            tag_mass = tag.channel_masses[c]
            out_mod_seq[out_idx] = new_seq
            out_seq[out_idx] = target_store.seq[i]
            out_prec_mz[out_idx] = (
                target_store.prec_mz[i]
                + (tag_mass - source_channel_mass) * n_tag_sites[i] / target_store.prec_z[i]
            )
            out_prec_z[out_idx] = target_store.prec_z[i]
            out_iRT[out_idx] = target_store.iRT[i]
            out_ion_mob[out_idx] = target_store.ion_mob[i]
            out_protein_group[out_idx] = target_store.protein_group[i]
            out_protein_name[out_idx] = target_store.protein_name[i]
            out_genes[out_idx] = target_store.genes[i]
            out_uniprot_id[out_idx] = target_store.uniprot_id[i]
            key_to_idx[(new_seq, orig_charge)] = out_idx
            source_channel_to_out_idx[(i, c)] = out_idx

            # Resolve parent_idx: find the parent target's output index in the same channel
            old_parent = target_store.parent_idx[i] if hasattr(target_store, 'parent_idx') else -1
            if old_parent >= 0:
                out_parent_idx[out_idx] = source_channel_to_out_idx.get((old_parent, c), -1)


        # --- Phase 4: Fill variable-length arrays ---
        logger.info("Tagging library")
        cursor = 0
        for out_idx, (i, c, _, _) in enumerate(tqdm.tqdm(valid)):
            foff = int(target_store.frag_offsets[i])
            flen = int(target_store.frag_lengths[i])

            out_spec_offsets[out_idx] = cursor
            out_spec_lengths[out_idx] = flen
            out_frag_offsets[out_idx] = cursor
            out_frag_lengths[out_idx] = flen

            if flen > 0:
                src_mz = target_store.frag_data[foff:foff + flen, 0]
                src_int = target_store.frag_data[foff:foff + flen, 1]
                src_keys = target_store.frag_keys_data[foff:foff + flen]
                local_n_tags = frag_n_tags[foff:foff + flen]
                local_charges = all_frag_charges[foff:foff + flen]

                tag_mass = tag.channel_masses[c]
                new_mz = src_mz + ((tag_mass - source_channel_mass) * local_n_tags / local_charges)

                # Frag data (original ordering)
                out_frag_data[cursor:cursor + flen, 0] = new_mz
                out_frag_data[cursor:cursor + flen, 1] = src_int
                out_frag_keys_data[cursor:cursor + flen] = src_keys

                # Spectrum data (sorted by m/z)
                order = np.argsort(new_mz)
                out_spectrum_mz[cursor:cursor + flen] = new_mz[order]
                out_spectrum_int[cursor:cursor + flen] = src_int[order]
                out_frag_names_data[cursor:cursor + flen] = src_keys[order]

            cursor += flen

        target_store.frag_data = None
        target_store.frag_keys_data = None

        # Top-N: empty (recomputed downstream when needed)
        out_top_n_data = np.empty(0, dtype=np.int32)
        out_top_n_offsets = np.zeros(V, dtype=np.int64)
        out_top_n_lengths = np.zeros(V, dtype=np.int32)

        # Propagate target/decoy info
        out_is_decoy = np.array(
            [target_store.is_decoy[i] for i, _, _, _ in valid],
            dtype=bool,
        )
        out_n_targets = int(np.sum(~out_is_decoy))
        out_n_decoys = int(np.sum(out_is_decoy))

        return cls(
            key_to_idx=key_to_idx,
            mod_seq=out_mod_seq,
            seq=out_seq,
            prec_mz=out_prec_mz,
            prec_z=out_prec_z,
            iRT=out_iRT,
            ion_mob=out_ion_mob,
            protein_group=out_protein_group,
            protein_name=out_protein_name,
            genes=out_genes,
            uniprot_id=out_uniprot_id,
            spectrum_mz=out_spectrum_mz,
            spectrum_int=out_spectrum_int,
            spectrum_offsets=out_spec_offsets,
            spectrum_lengths=out_spec_lengths,
            frag_names_data=out_frag_names_data,
            frag_data=out_frag_data,
            frag_keys_data=out_frag_keys_data,
            frag_offsets=out_frag_offsets,
            frag_lengths=out_frag_lengths,
            top_n_data=out_top_n_data,
            top_n_offsets=out_top_n_offsets,
            top_n_lengths=out_top_n_lengths,
            parent_idx=out_parent_idx,
            n_targets=out_n_targets,
            n_decoys=out_n_decoys,
            is_decoy=out_is_decoy,
        )

    # ------------------------------------------------------------------
    # Export: vectorized DIA-NN-format Polars DataFrame
    # ------------------------------------------------------------------

    def to_diann_df(self, n=None):
        """Convert to a one-row-per-fragment Polars DataFrame (DIA-NN format).

        Uses a join strategy to avoid numpy fancy indexing on object arrays
        (which is extremely slow for large libraries).

        Parameters
        ----------
        n : int, optional
            Number of entries to export (from the start). Defaults to all.
        """
        import polars as pl
        from src.utils.frag_encoding import (
            get_ion_type, get_index, get_charge, get_loss,
        )

        N = n if n is not None else len(self)
        frag_lens = self.frag_lengths[:N].astype(np.intp)
        total_frags = int(frag_lens.sum())

        # -- Build gather index (handles potential gaps in frag_data) --
        entry_idx = np.repeat(np.arange(N, dtype=np.intp), frag_lens)
        offsets_expanded = np.repeat(self.frag_offsets[:N], frag_lens)
        within_group = np.arange(total_frags, dtype=np.intp) - np.repeat(
            np.cumsum(frag_lens) - frag_lens, frag_lens
        )
        gather_idx = offsets_expanded + within_group

        # -- Fragment-level columns from frag_data / frag_keys_data --
        frag_mz = self.frag_data[gather_idx, 0]
        frag_int = self.frag_data[gather_idx, 1]
        codes = self.frag_keys_data[gather_idx]

        # Decode packed int32 codes into separate fields
        _ION_NAMES = pl.Series(values=['b', 'y', 'a', 'c', 'x', 'z'])
        _LOSS_NAMES = pl.Series(values=['noloss', 'H2O', 'NH3', 'H3PO4'])

        # -- Gather index as a polars series for .gather() calls --
        entry_idx_pl = pl.Series(values=entry_idx, dtype=pl.UInt32)

        # -- Precursor string columns: build polars series once at N level,
        #    then expand via gather (all in Rust, no Python object arrays) --
        mod_seq = pl.Series("ModifiedPeptide", self.mod_seq[:N]).gather(entry_idx_pl)
        seq = pl.Series("StrippedPeptide", self.seq[:N]).gather(entry_idx_pl)
        protein_group = pl.Series("ProteinGroup", self.protein_group[:N]).gather(entry_idx_pl)
        protein_name = pl.Series("ProteinName", self.protein_name[:N]).gather(entry_idx_pl)
        genes = pl.Series("Genes", self.genes[:N]).gather(entry_idx_pl)
        uniprot_id = pl.Series("ProteinID", self.uniprot_id[:N]).gather(entry_idx_pl)

        # -- Numeric precursor columns: numpy repeat is fast on contiguous arrays --
        prec_mz = self.prec_mz[:N][entry_idx]
        prec_z = self.prec_z[:N][entry_idx]
        iRT = self.iRT[:N][entry_idx]
        ion_mob = self.ion_mob[:N][entry_idx]

        # -- Fragment-level columns --
        ion_type_codes = get_ion_type(codes)
        loss_codes = get_loss(codes)

        return pl.DataFrame({
            "ModifiedPeptide": mod_seq,
            "StrippedPeptide": seq,
            "PrecursorCharge": prec_z.astype(np.int32),
            "RT": iRT,
            "IonMobility": ion_mob,
            "PrecursorMz": prec_mz,
            "FragmentMz": frag_mz,
            "RelativeIntensity": frag_int,
            "FragmentType": _ION_NAMES.gather(ion_type_codes.astype(np.uint32)),
            "FragmentCharge": get_charge(codes).astype(np.int32),
            "FragmentSeriesNumber": get_index(codes).astype(np.int32),
            "FragmentLossType": _LOSS_NAMES.gather(loss_codes.astype(np.uint32)),
            "ProteinID": uniprot_id,
            "ProteinGroup": protein_group,
            "ProteinName": protein_name,
            "Genes": genes,
        })

    # ------------------------------------------------------------------
    # Factory: direct TSV parser
    # ------------------------------------------------------------------
    @classmethod
    def from_tsv(cls, spec_lib_file):
        from src.logger import logger
        logger.info("using: SpectrumLibraryStore.from_tsv")
        return cls._read_from_tsv_or_parquet(spec_lib_file, "tsv")

    @classmethod
    def from_parquet(cls, spec_lib_file):
        from src.logger import logger
        logger.info("using: SpectrumLibraryStore.from_parquet")
        return cls._read_from_tsv_or_parquet(spec_lib_file, "parquet")

    @classmethod
    def _read_from_tsv_or_parquet(cls, spec_lib_file, file_type):
        """Parse a DIA-NN / FragPipe TSV or Parquet directly into columnar form."""
        from src.logger import logger
        import polars as pl
        import time
        import tqdm

        t_start = time.perf_counter()
        df, found = _load_library_frame(spec_lib_file, file_type)
        logger.info(f"Parsing spectral library: {df.height:,} fragment rows")

        progress = tqdm.tqdm(total=7, desc="Parsing library", unit="stage", leave=False)

        def _stage(name):
            progress.set_postfix_str(name)
            progress.update(1)

        def _raise(message):
            from src.utils.gui_utils import send_raise_to_TK
            send_raise_to_TK(f"ValueError - {message}")
            raise ValueError(message)

        def _require(canonical, message):
            if canonical not in found and df.height > 0:
                _raise(message)

        def _utf8(canonical):
            return pl.col(canonical).cast(pl.Utf8)

        def _str_col(canonical):
            expr = _utf8(canonical) if canonical in found else pl.lit("")
            return expr.fill_null("")

        decoy_precursors = set()
        invalid_precursors = set()

        if df.height > 0:
            _require("ModifiedPeptide", "Unknown ModifiedPeptide column")
            _require("StrippedPeptide", "Unknown StrippedPeptide column")
            _require("PrecursorCharge", "SpecLib Charge Cannot be Converted to Float")

            # ModifiedPeptide with wrapping underscores stripped; decoy/invalid
            # bookkeeping uses this pre-N-terminal-move form
            df = df.with_columns(
                _utf8("ModifiedPeptide").str.strip_chars("_").alias("_mod_pep_raw")
            )
            try:
                df = df.with_columns(pl.col("PrecursorCharge").cast(pl.Float64).alias("_prec_z"))
            except Exception:
                _raise("SpecLib Charge Cannot be Converted to Float")

            # Decoy rows: same accepted spellings as str(x).strip() in ("1","1.0","True")
            if "Decoy" in found:
                decoy_dtype = df.schema["Decoy"]
                if decoy_dtype == pl.Boolean:
                    decoy_mask = pl.col("Decoy")
                elif decoy_dtype.is_numeric():
                    decoy_mask = pl.col("Decoy").cast(pl.Float64) == 1.0
                else:
                    decoy_mask = _utf8("Decoy").str.strip_chars().is_in(["1", "1.0", "True"])
                decoy_mask = decoy_mask.fill_null(False)
            else:
                decoy_mask = pl.lit(False)

            decoy_precursors = set(
                df.filter(decoy_mask).select(["_mod_pep_raw", "_prec_z"]).unique().rows()
            )
            df = df.filter(~decoy_mask)

            ## Check for non-valid AAs
            x_mask = _utf8("StrippedPeptide").str.contains("X", literal=True).fill_null(False)
            invalid_precursors |= set(
                df.filter(x_mask).select(["_mod_pep_raw", "_prec_z"]).unique().rows()
            )
            df = df.filter(~x_mask)

            ## Move DIANN N-terminal tags/modifications behind the first AA:
            ## (tag)C(UniMod:4)SQAPVYGR → C(UniMod:4)(tag)SQAPVYGR
            df = df.with_columns(
                pl.col("_mod_pep_raw").str.replace(
                    r'^((?:\([^)]*\))+)([A-Z])((?:\([^)]*\))*)(.*)$', "${2}${3}${1}${4}"
                ).alias("_mod_pep")
            )

            # Skip precursors carrying residues with no defined mass rather than
            # letting them reach decoy generation, where fast_mass raises.
            nonstd_mask = _utf8("StrippedPeptide").str.contains(
                "[^" + "".join(sorted(_STANDARD_RESIDUES)) + "]"
            ).fill_null(False)
            invalid_precursors |= set(
                df.filter(nonstd_mask).select(["_mod_pep", "_prec_z"]).unique().rows()
            )
            df = df.filter(~nonstd_mask)
            _stage("filtered decoy/invalid rows")

        if df.height > 0:
            _require("RT", "Unknown retention time column")
            _require("PrecursorMz", "Unknown PrecursorMz column")
            for canonical in ("FragmentType", "FragmentSeriesNumber", "FragmentCharge",
                              "FragmentMz", "RelativeIntensity"):
                _require(canonical, f"Unknown {canonical} column")

            rt = pl.col("RT")
            if df.schema["RT"] == pl.Utf8:
                rt = pl.when(rt == "").then(None).otherwise(rt)

            pmz = pl.col("PrecursorMz")
            if df.schema["PrecursorMz"] == pl.Utf8:
                pmz = pl.when(pmz == "").then(None).otherwise(pmz)

            if "IonMobility" in found:
                im = pl.col("IonMobility")
                if df.schema["IonMobility"] == pl.Utf8:
                    im = pl.when(im == "").then(None).otherwise(im)
                im = im.cast(pl.Float64)
                # in DIANN IM = 0.0 if experiment does not have IM
                im = pl.when(im == 0.0).then(None).otherwise(im)
            else:
                im = pl.lit(None, dtype=pl.Float64)

            genes_base = _utf8("Genes") if "Genes" in found else pl.lit("")
            ## standardize JMod and DIANN speclibs: empty Genes -> '""'
            genes = pl.when(genes_base == "").then(pl.lit('""')).otherwise(genes_base).fill_null("")

            # Fragment key, e.g. "y4-H2O_1". Loss strings are not enumerated:
            # anything other than unknown/noloss/"" is carried into the key
            # verbatim, so fragments differing only in loss type stay distinct.
            loss_base = _utf8("FragmentLossType").fill_null("") if "FragmentLossType" in found else pl.lit("")
            loss = pl.when(loss_base.is_in(["unknown", "noloss", ""])) \
                     .then(pl.lit("")).otherwise(pl.lit("-") + loss_base)
            frag_key = (_utf8("FragmentType") + _utf8("FragmentSeriesNumber")
                        + loss + pl.lit("_") + _utf8("FragmentCharge"))

            df = df.with_columns(
                _utf8("StrippedPeptide").alias("_seq"),
                rt.cast(pl.Float64).alias("_iRT"),
                pmz.cast(pl.Float64).alias("_prec_mz"),
                im.alias("_ion_mob"),
                _str_col("ProteinGroup").alias("_protein_group"),
                _str_col("ProteinName").alias("_protein_name"),
                genes.alias("_genes"),
                _str_col("ProteinID").alias("_uniprot_id"),
                frag_key.alias("_frag_key"),
                pl.col("FragmentMz").cast(pl.Float64).alias("_frag_mz"),
                pl.col("RelativeIntensity").cast(pl.Float64).alias("_frag_int"),
                # Key components, kept for arithmetic fragment-code encoding
                _utf8("FragmentType").alias("_frag_ion"),
                pl.col("FragmentSeriesNumber").cast(pl.Int64).alias("_frag_idx"),
                loss_base.alias("_frag_loss"),
                pl.col("FragmentCharge").cast(pl.Int64).alias("_frag_z"),
            )

            _stage("built typed columns")

            # RT and PrecursorMz are mandatory for every precursor (IM is optional)
            n_null_rt = df["_iRT"].null_count()
            if n_null_rt > 0:
                _raise(f"Library retention time has {n_null_rt} missing values")
            n_null_pmz = df["_prec_mz"].null_count()
            if n_null_pmz > 0:
                _raise(f"Library PrecursorMz has {n_null_pmz} missing values")

            # Precursor-level fields come from each precursor's first row;
            # maintain_order preserves first-appearance precursor ordering
            prec_df = df.group_by(["_mod_pep", "_prec_z"], maintain_order=True).agg(
                pl.col("_seq").first(),
                pl.col("_iRT").first(),
                pl.col("_ion_mob").first(),
                pl.col("_protein_group").first(),
                pl.col("_protein_name").first(),
                pl.col("_genes").first(),
                pl.col("_uniprot_id").first(),
                pl.col("_prec_mz").first(),
            )
            _stage("grouped precursors")
            # Duplicate fragment keys (identical type+number+loss+charge):
            # dict-update semantics are first-appearance position, last-seen value
            frag_df = df.group_by(["_mod_pep", "_prec_z", "_frag_key"], maintain_order=True).agg(
                pl.col("_frag_mz").last(),
                pl.col("_frag_int").last(),
                # equal keys imply equal components
                pl.col("_frag_ion").first(),
                pl.col("_frag_idx").first(),
                pl.col("_frag_loss").first(),
                pl.col("_frag_z").first(),
            )

            prec_df = prec_df.with_row_index("_pidx")
            # Contiguous per-precursor fragment blocks in precursor order; the
            # sort is stable so within-precursor first-appearance order survives
            frag_df = frag_df.join(
                prec_df.select(["_mod_pep", "_prec_z", "_pidx"]),
                on=["_mod_pep", "_prec_z"],
                how="left",
            ).sort("_pidx", maintain_order=True)
            _stage("deduplicated fragments")

        if len(decoy_precursors) > 0:
            logger.info(f"{len(decoy_precursors)} decoy precursors removed from input library")

        if len(invalid_precursors) > 0:
            example = sorted(invalid_precursors)[0][0]
            logger.info(f"{len(invalid_precursors)} precursors with non-standard residues "
                        f"removed from input library (e.g. {example})")

        # Second pass: convert to columnar arrays
        if df.height == 0:
            progress.close()
            return cls._empty()

        n = prec_df.height
        mod_seq_list = prec_df["_mod_pep"].to_list()
        prec_z_list = prec_df["_prec_z"].to_list()
        key_to_idx = {uid: i for i, uid in enumerate(zip(mod_seq_list, prec_z_list))}

        mod_seq_arr = np.array(mod_seq_list, dtype=object)
        seq_arr = np.array(prec_df["_seq"].to_list(), dtype=object)
        prec_mz_arr = prec_df["_prec_mz"].to_numpy().astype(np.float64, copy=False)
        prec_z_arr = prec_df["_prec_z"].to_numpy().astype(np.float64, copy=False)
        iRT_arr = prec_df["_iRT"].to_numpy().astype(np.float64, copy=False)
        # nulls (no IM) become NaN
        ion_mob_arr = prec_df["_ion_mob"].to_numpy().astype(np.float64, copy=False)
        protein_group_arr = np.array(prec_df["_protein_group"].to_list(), dtype=object)
        protein_name_arr = np.array(prec_df["_protein_name"].to_list(), dtype=object)
        genes_arr = np.array(prec_df["_genes"].to_list(), dtype=object)
        uniprot_id_arr = np.array(prec_df["_uniprot_id"].to_list(), dtype=object)
        parent_idx_arr = np.full(n, -1, dtype=np.int64)
        _stage("built precursor arrays")

        # Original frags: frag_df is already one contiguous block per precursor
        frag_lengths_arr = (
            frag_df.group_by("_pidx", maintain_order=True).len()["len"]
            .to_numpy().astype(np.int32)
        )
        frag_offsets_arr = np.zeros(n, dtype=np.int64)
        frag_offsets_arr[1:] = np.cumsum(frag_lengths_arr[:-1], dtype=np.int64)

        frag_keys_data = encode_frag_columns(
            frag_df["_frag_ion"].to_list(),
            frag_df["_frag_idx"].to_numpy(),
            frag_df["_frag_loss"].to_list(),
            frag_df["_frag_z"].to_numpy(),
        )
        frag_mz = frag_df["_frag_mz"].to_numpy().astype(np.float64, copy=False)
        frag_int = frag_df["_frag_int"].to_numpy().astype(np.float64, copy=False)
        frag_data = np.column_stack((frag_mz, frag_int))
        _stage("encoded fragment codes")

        # Spectrum: one global stable sort by (precursor, m/z) replaces the
        # per-precursor frag_to_peak argsorts; equal-m/z ties keep
        # first-appearance order, matching frag_to_peak's kind="stable"
        pidx = frag_df["_pidx"].to_numpy()
        perm = np.lexsort((frag_mz, pidx))
        spectrum_mz = np.ascontiguousarray(frag_mz[perm])
        spectrum_int = np.ascontiguousarray(frag_int[perm])
        frag_names_data = frag_keys_data[perm]
        spec_offsets = frag_offsets_arr.copy()
        spec_lengths = frag_lengths_arr.copy()
        _stage("sorted spectra")
        progress.close()

        logger.info(f"Parsed {n:,} precursors / {len(frag_keys_data):,} fragments "
                    f"in {time.perf_counter() - t_start:.1f}s")

        return cls(
            key_to_idx=key_to_idx,
            mod_seq=mod_seq_arr,
            seq=seq_arr,
            prec_mz=prec_mz_arr,
            prec_z=prec_z_arr,
            iRT=iRT_arr,
            ion_mob=ion_mob_arr,
            protein_group=protein_group_arr,
            protein_name=protein_name_arr,
            genes=genes_arr,
            uniprot_id=uniprot_id_arr,
            spectrum_mz=spectrum_mz,
            spectrum_int=spectrum_int,
            spectrum_offsets=spec_offsets,
            spectrum_lengths=spec_lengths,
            frag_names_data=frag_names_data,
            frag_data=frag_data,
            frag_keys_data=frag_keys_data,
            frag_offsets=frag_offsets_arr,
            frag_lengths=frag_lengths_arr,
            top_n_data=np.empty(0, dtype=np.int32),
            top_n_offsets=np.zeros(n, dtype=np.int64),
            top_n_lengths=np.zeros(n, dtype=np.int32),
            parent_idx=parent_idx_arr,
        )

    # ------------------------------------------------------------------
    # Factory: direct blib parser
    # ------------------------------------------------------------------

    @classmethod
    def from_blib(cls, spec_lib_file):
        """Parse a .blib SQLite file directly into columnar form."""
        import sqlite3
        import struct
        import zlib
        import pandas as pd

        python_lib = {}
        sql_lib = sqlite3.connect(spec_lib_file)
        Precursors = pd.read_sql("SELECT * FROM RefSpectra", sql_lib)

        for i in range(len(Precursors)):
            precID = str(Precursors["id"][i])
            precKey = (Precursors["peptideModSeq"][i], Precursors["precursorCharge"][i])
            NumPeaks = pd.read_sql("SELECT numPeaks FROM RefSpectra WHERE id = " + precID, sql_lib)['numPeaks'][0]

            SpectrumMZ = pd.read_sql("SELECT peakMZ FROM RefSpectraPeaks WHERE RefSpectraID = " + precID, sql_lib)['peakMZ'][0]
            SpectrumIntensities = pd.read_sql("SELECT peakIntensity FROM RefSpectraPeaks WHERE RefSpectraID = " + precID, sql_lib)['peakIntensity'][0]

            spectrum = None
            if len(SpectrumMZ) == 8 * NumPeaks and len(SpectrumIntensities) == 4 * NumPeaks:
                mzs = struct.unpack('d' * NumPeaks, SpectrumMZ)
                ints = struct.unpack('f' * NumPeaks, SpectrumIntensities)
                spectrum = np.array((mzs, ints)).T
            elif len(SpectrumIntensities) == 4 * NumPeaks:
                mzs = struct.unpack('d' * NumPeaks, zlib.decompress(SpectrumMZ))
                ints = struct.unpack('f' * NumPeaks, SpectrumIntensities)
                spectrum = np.array((mzs, ints)).T
            elif len(SpectrumMZ) == 8 * NumPeaks:
                mzs = struct.unpack('d' * NumPeaks, SpectrumMZ)
                ints = struct.unpack('f' * NumPeaks, zlib.decompress(SpectrumIntensities))
                spectrum = np.array((mzs, ints)).T
            elif len(zlib.decompress(SpectrumMZ)) == 8 * NumPeaks and len(zlib.decompress(SpectrumIntensities)) == 4 * NumPeaks:
                mzs = struct.unpack('d' * NumPeaks, zlib.decompress(SpectrumMZ))
                ints = struct.unpack('f' * NumPeaks, zlib.decompress(SpectrumIntensities))
                spectrum = np.array((mzs, ints)).T

            if spectrum is not None:
                python_lib[precKey] = {
                    'spectrum': spectrum,
                    'prec_mz': Precursors['precursorMZ'][i],
                    'iRT': Precursors['retentionTime'][i],
                }

        sql_lib.close()
        return cls.from_dict(python_lib)

    # ------------------------------------------------------------------
    # Empty store helper
    # ------------------------------------------------------------------

    @classmethod
    def _empty(cls):
        n = 0
        return cls(
            key_to_idx={},
            mod_seq=np.empty(n, dtype=object),
            seq=np.empty(n, dtype=object),
            prec_mz=np.empty(n, dtype=np.float64),
            prec_z=np.empty(n, dtype=np.float64),
            iRT=np.empty(n, dtype=np.float64),
            ion_mob=np.empty(n, dtype=np.float64),
            protein_group=np.empty(n, dtype=object),
            protein_name=np.empty(n, dtype=object),
            genes=np.empty(n, dtype=object),
            uniprot_id=np.empty(n, dtype=object),
            spectrum_mz=np.empty(0, dtype=np.float64),
            spectrum_int=np.empty(0, dtype=np.float64),
            spectrum_offsets=np.empty(n, dtype=np.int64),
            spectrum_lengths=np.empty(n, dtype=np.int32),
            frag_names_data=np.empty(0, dtype=np.int32),
            frag_data=np.empty((0, 2), dtype=np.float64),
            frag_keys_data=np.empty(0, dtype=np.int32),
            frag_offsets=np.empty(n, dtype=np.int64),
            frag_lengths=np.empty(n, dtype=np.int32),
            top_n_data=np.empty(0, dtype=np.int32),
            top_n_offsets=np.empty(n, dtype=np.int64),
            top_n_lengths=np.empty(n, dtype=np.int32),
            parent_idx=np.full(n, -1, dtype=np.int64),
        )
    
    def relabel_tag(self, library_tag_name, source_channel):
        """Replace a placeholder tag annotation (e.g. "(tag)") with a concrete
        source channel annotation (e.g. "(PSMtag_5plex-d0)") throughout this store.

        Updates both ``mod_seq`` values and the ``key_to_idx`` keys so they stay
        in sync. Mutates in place and returns self.
        """
        old_annotation = "(" + library_tag_name + ")"
        new_annotation = "(" + source_channel + ")"

        for i in range(len(self.mod_seq)):
            old_seq = self.mod_seq[i]
            if old_annotation in old_seq:
                self.mod_seq[i] = old_seq.replace(old_annotation, new_annotation)

        new_key_to_idx = {}
        for (mod_seq_key, charge), idx in self.key_to_idx.items():
            if old_annotation in mod_seq_key:
                mod_seq_key = mod_seq_key.replace(old_annotation, new_annotation)
            new_key_to_idx[(mod_seq_key, charge)] = idx
        self.key_to_idx = new_key_to_idx

        return self


class _EntryView:
    """Lightweight proxy that makes ``store[key]`` behave like a dict.

    Uses ``__slots__`` so it has no ``__dict__`` and is not GC-tracked.
    """

    __slots__ = ('_store', '_idx')

    def __init__(self, store, idx):
        object.__setattr__(self, '_store', store)
        object.__setattr__(self, '_idx', idx)

    # ---- dict-like read ----

    def __getitem__(self, field):
        store = self._store
        idx = self._idx

        if field == 'spectrum':
            return store.get_spectrum(idx)
        if field == 'ordered_frags':
            return store.get_ordered_frags(idx)
        if field == 'ordered_frag_codes':
            return store.get_ordered_frag_codes(idx)
        if field == 'frag_intensities':
            return store.get_frag_intensities(idx)
        if field == 'frags':
            return store.get_frags(idx)
        if field == 'top_n':
            return store.get_top_n(idx)
        if field == 'parent_idx':
            return int(store.parent_idx[idx])
        if field == 'spec_frags':
            return None

        # Scalar string fields
        if field == 'mod_seq':
            return store.mod_seq[idx]
        if field == 'seq':
            return store.seq[idx]
        if field == 'protein_group':
            return store.protein_group[idx]
        if field == 'protein_name':
            return store.protein_name[idx]
        if field == 'genes':
            return store.genes[idx]
        if field == 'UniprotID':
            return store.uniprot_id[idx]

        # Scalar float fields
        if field == 'prec_mz':
            return float(store.prec_mz[idx])
        if field == 'prec_z':
            return float(store.prec_z[idx])
        if field == 'iRT':
            val = store.iRT[idx]
            return None if np.isnan(val) else float(val)
        if field == 'IonMob':
            val = store.ion_mob[idx]
            return None if np.isnan(val) else float(val)

        raise KeyError(field)

    # ---- dict-like write ----

    def __setitem__(self, field, value):
        store = self._store
        idx = self._idx

        if field == 'spectrum':
            store.set_spectrum(idx, value)
            return
        if field == 'ordered_frags':
            off = store.spectrum_offsets[idx]
            length = store.spectrum_lengths[idx]
            value = _ensure_frag_codes(value)
            if len(value) == length:
                store.frag_names_data[off:off + length] = value
            else:
                store.set_spectrum(idx, store.get_spectrum(idx), value)
            return
        if field == 'frags':
            store.set_frags(idx, value)
            return
        if field == 'top_n':
            store.set_top_n(idx, value)
            return
        if field == 'parent_idx':
            store.parent_idx[idx] = value
            return

        # Scalar string fields
        if field == 'mod_seq':
            store.mod_seq[idx] = value
            return
        if field == 'seq':
            store.seq[idx] = value
            return
        if field == 'protein_group':
            store.protein_group[idx] = value
            return
        if field == 'protein_name':
            store.protein_name[idx] = value
            return
        if field == 'genes':
            store.genes[idx] = value
            return
        if field == 'UniprotID':
            store.uniprot_id[idx] = value
            return

        # Scalar float fields
        if field == 'prec_mz':
            store.prec_mz[idx] = float(value)
            return
        if field == 'prec_z':
            store.prec_z[idx] = float(value)
            return
        if field == 'iRT':
            store.iRT[idx] = np.nan if value is None else float(value)
            return
        if field == 'IonMob':
            store.ion_mob[idx] = np.nan if value is None else float(value)
            return

        raise KeyError(f"Unknown field: {field}")

    # ---- dict-like membership / iteration ----

    def __contains__(self, field):
        if field in _ALL_KNOWN_FIELDS:
            if field == 'IonMob':
                return not np.isnan(self._store.ion_mob[self._idx])
            if field == 'UniprotID':
                val = self._store.uniprot_id[self._idx]
                return val is not None and val != ''
            if field == 'spec_frags':
                return False
            if field == 'top_n':
                return self._store.top_n_lengths[self._idx] > 0
            if field == 'parent_idx':
                return self._store.parent_idx[self._idx] >= 0
            return True
        return False

    def keys(self):
        """Return field names present for this entry."""
        result = ['mod_seq', 'seq', 'prec_mz', 'prec_z', 'iRT',
                  'frags', 'spectrum', 'ordered_frags']
        if not np.isnan(self._store.ion_mob[self._idx]):
            result.append('IonMob')
        pg = self._store.protein_group[self._idx]
        if pg is not None and pg != '':
            result.append('protein_group')
        pn = self._store.protein_name[self._idx]
        if pn is not None and pn != '':
            result.append('protein_name')
        g = self._store.genes[self._idx]
        if g is not None and g != '':
            result.append('genes')
        uid = self._store.uniprot_id[self._idx]
        if uid is not None and uid != '':
            result.append('UniprotID')
        if self._store.top_n_lengths[self._idx] > 0:
            result.append('top_n')
        if self._store.parent_idx[self._idx] >= 0:
            result.append('parent_idx')
        return result

    def get(self, field, default=None):
        try:
            return self[field]
        except KeyError:
            return default

    def setdefault(self, field, default=None):
        if field in self:
            return self[field]
        self[field] = default
        return default

    def __iter__(self):
        return iter(self.keys())

    def __repr__(self):
        return f"_EntryView(idx={self._idx}, mod_seq={self._store.mod_seq[self._idx]})"

    def __deepcopy__(self, memo):
        """Return a plain dict deep copy — prevents deepcopy from cloning the whole store."""
        import copy
        return copy.deepcopy(dict(self), memo)
