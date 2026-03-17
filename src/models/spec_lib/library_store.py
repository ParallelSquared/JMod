"""
This Source Code Form is subject to the terms of the Oxford Nanopore
Technologies, Ltd. Public License, v. 1.0.  Full licence can be found
at https://github.com/ParallelSquared/JMod/blob/main/LICENSE.txt
"""
import numpy as np
import os
from src.utils.frag_encoding import encode_frag_name, encode_frag_names, decode_frag_names


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
_ALL_KNOWN_FIELDS = _ALL_SCALAR_FIELDS | frozenset({
    'spectrum', 'ordered_frags', 'ordered_frag_codes', 'frag_intensities',
    'frags', 'top_n', 'parent_key', 'spec_frags',
})


def _ensure_frag_codes(value):
    """Convert fragment names to int32 codes if they are strings."""
    arr = np.asarray(value)
    if arr.dtype == np.int32:
        return arr
    # object or string dtype — encode from strings
    return encode_frag_names(arr)


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
        'spectrum_data', 'spectrum_offsets', 'spectrum_lengths',
        'frag_names_data',
        # concatenated variable-length original frags data (independent of spectrum)
        'frag_data', 'frag_keys_data', 'frag_offsets', 'frag_lengths',
        # concatenated variable-length top_n data
        'top_n_data', 'top_n_offsets', 'top_n_lengths',
        # parent_key (object array, None when not set)
        'parent_key',
    )

    def __init__(
        self,
        key_to_idx,
        mod_seq, seq, prec_mz, prec_z, iRT, ion_mob,
        protein_group, protein_name, genes, uniprot_id,
        spectrum_data, spectrum_offsets, spectrum_lengths,
        frag_names_data,
        frag_data, frag_keys_data, frag_offsets, frag_lengths,
        top_n_data, top_n_offsets, top_n_lengths,
        parent_key,
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
        self.spectrum_data = spectrum_data
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
        self.parent_key = parent_key

    # ------------------------------------------------------------------
    # Internal accessors
    # ------------------------------------------------------------------

    def get_spectrum(self, idx):
        """Return (n_peaks, 2) float64 view for entry *idx*."""
        off = self.spectrum_offsets[idx]
        length = self.spectrum_lengths[idx]
        return self.spectrum_data[off:off + length]

    def get_ordered_frags(self, idx):
        """Return 1-D object array of decoded fragment name strings for entry *idx*."""
        off = self.spectrum_offsets[idx]
        length = self.spectrum_lengths[idx]
        return decode_frag_names(self.frag_names_data[off:off + length])

    def get_frag_intensities(self, idx):
        """Return 1-D float64 array of fragment intensities for entry *idx*."""
        off = self.spectrum_offsets[idx]
        length = self.spectrum_lengths[idx]
        return self.spectrum_data[off:off + length, 1]

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
        old_len = self.spectrum_lengths[idx]
        new_len = len(spectrum_array)
        if new_len == old_len:
            off = self.spectrum_offsets[idx]
            self.spectrum_data[off:off + new_len] = spectrum_array
            if ordered_frags is not None:
                self.frag_names_data[off:off + new_len] = _ensure_frag_codes(ordered_frags)
        else:
            new_off = len(self.spectrum_data)
            self.spectrum_data = np.concatenate(
                [self.spectrum_data, np.asarray(spectrum_array, dtype=np.float64)], axis=0
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
        sd, so, sl = self.spectrum_data, self.spectrum_offsets, self.spectrum_lengths
        return [sd[so[i]:so[i] + sl[i]] for i in indices]

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

        if 'parent_key' in entry_dict:
            self.parent_key[idx] = entry_dict['parent_key']

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

        # parent_key
        new_pk = np.empty(idx + 1, dtype=object)
        new_pk[:idx] = self.parent_key
        new_pk[idx] = entry_dict.get('parent_key')
        self.parent_key = new_pk

        # spectrum offsets / lengths
        self.spectrum_offsets = np.append(self.spectrum_offsets, len(self.spectrum_data))
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
            spectrum_data=self.spectrum_data.copy(),
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
            parent_key=self.parent_key.copy(),
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
        scalar arrays for fields that may be mutated (iRT, parent_key)."""
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
            spectrum_data=self.spectrum_data,
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
            parent_key=self.parent_key.copy(),
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
            spectrum_data=self.spectrum_data,
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
            parent_key=self.parent_key,
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

        # Handle older caches that lack separate frag arrays
        if 'frag_data' in data:
            frag_data = data['frag_data']
            frag_keys_data = data['frag_keys_data']
            frag_offsets = data['frag_offsets']
            frag_lengths = data['frag_lengths']
        else:
            # Fall back: use spectrum data as frag data (pre-isotope state)
            frag_data = data['spectrum_data']
            frag_keys_data = data['frag_names_data']
            frag_offsets = data['spectrum_offsets']
            frag_lengths = data['spectrum_lengths']

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
            spectrum_data=data['spectrum_data'],
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
            parent_key=data['parent_key'],
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
        parent_key_list = []

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
            parent_key_list.append(entry.get('parent_key'))

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
        spectrum_data = np.concatenate(all_spec_peaks, axis=0) if all_spec_peaks else np.empty((0, 2), dtype=np.float64)
        frag_names_data = np.concatenate(all_spec_frag_names) if all_spec_frag_names else np.empty(0, dtype=np.int32)
        frag_data = np.concatenate(all_frag_peaks, axis=0) if all_frag_peaks else np.empty((0, 2), dtype=np.float64)
        frag_keys_data = np.concatenate(all_frag_keys) if all_frag_keys else np.empty(0, dtype=np.int32)
        top_n_data = np.concatenate(top_n_all) if top_n_all else np.empty(0, dtype=np.int32)

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
            spectrum_data=spectrum_data,
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
            parent_key=np.array(parent_key_list, dtype=object),
        )

    # ------------------------------------------------------------------
    # Factory: direct TSV parser
    # ------------------------------------------------------------------

    @classmethod
    def from_tsv(cls, spec_lib_file):
        """Parse a DIA-NN / FragPipe TSV directly into columnar form."""
        import csv
        from src.utils.misc_functions import frag_to_peak
        from src.logger import logger

        logger.info("using: SpectrumLibraryStore.from_tsv")

        # First pass: accumulate into per-precursor lists
        precursor_order = []
        precursor_data = {}

        with open(spec_lib_file, newline="") as tsv_file:
            csv_reader = csv.DictReader(tsv_file, delimiter="\t")

            for row in csv_reader:
                # Resolve ModifiedPeptide
                if "ModifiedPeptide" in row:
                    mod_pep = row["ModifiedPeptide"].strip("_")
                elif "ModifiedSequence" in row:
                    mod_pep = row["ModifiedSequence"].strip("_")
                else:
                    from src.utils.gui_utils import send_raise_to_TK
                    send_raise_to_TK("ValueError - Unknown ModifiedPeptide Column")
                    raise ValueError("Unknown ModifiedPeptide column")

                charge = float(row["PrecursorCharge"])
                unique_id = (mod_pep, charge)

                if unique_id not in precursor_data:
                    precursor_order.append(unique_id)
                    if "StrippedPeptide" in row:
                        seq = row["StrippedPeptide"]
                    else:
                        seq = row["PeptideSequence"]

                    if "Tr_recalibrated" in row:
                        rt = row["Tr_recalibrated"]
                    elif "RT" in row:
                        rt = row["RT"]
                    elif "iRT" in row:
                        rt = row["iRT"]
                    else:
                        from src.utils.gui_utils import send_raise_to_TK
                        send_raise_to_TK("ValueError - Unknown Retention Time Column")
                        raise ValueError("Unknown retention time column")

                    iRT = np.nan if rt == "" else float(rt)

                    ion_mob = np.nan
                    if "IonMobility" in row and row["IonMobility"] != "":
                        ion_mob = float(row["IonMobility"])

                    protein_group = row.get("ProteinGroup", "")
                    if "ProteinName" in row:
                        protein_name = row["ProteinName"]
                    elif "ProteinID" in row:
                        protein_name = row["ProteinID"]
                    elif "ProteinId" in row:
                        protein_name = row["ProteinId"]
                    else:
                        protein_name = ""
                    genes_val = row.get("Genes", "")
                    if not genes_val and "GeneName" in row:
                        genes_val = row["GeneName"]
                    uniprot_id = row.get("UniprotID", "")

                    precursor_data[unique_id] = {
                        'mod_seq': mod_pep,
                        'seq': seq,
                        'prec_mz': float(row["PrecursorMz"]),
                        'prec_z': charge,
                        'iRT': iRT,
                        'ion_mob': ion_mob,
                        'protein_group': protein_group or "",
                        'protein_name': protein_name or "",
                        'genes': genes_val or "",
                        'uniprot_id': uniprot_id or "",
                        'frags': {},
                    }

                # Build fragment key
                loss = ""
                if "FragmentLossType" in row:
                    loss = str(row["FragmentLossType"])
                    if loss in ["unknown", "noloss", ""]:
                        loss = ""
                    else:
                        loss = "-" + loss

                if "FragmentNumber" in row:
                    frag_type = str(row["FragmentType"]) + str(row["FragmentNumber"]) + loss + "_" + str(row["FragmentCharge"])
                else:
                    frag_type = str(row["FragmentType"]) + str(row["FragmentSeriesNumber"]) + loss + "_" + str(row["FragmentCharge"])

                if "FragmentMz" in row:
                    frag_mz = float(row["FragmentMz"])
                    frag_int = float(row["RelativeIntensity"])
                else:
                    frag_mz = float(row["ProductMz"])
                    frag_int = float(row["LibraryIntensity"])

                precursor_data[unique_id]['frags'][frag_type] = [frag_mz, frag_int]

        # Second pass: convert to columnar arrays
        n = len(precursor_order)
        mod_seq_arr = np.empty(n, dtype=object)
        seq_arr = np.empty(n, dtype=object)
        prec_mz_arr = np.empty(n, dtype=np.float64)
        prec_z_arr = np.empty(n, dtype=np.float64)
        iRT_arr = np.empty(n, dtype=np.float64)
        ion_mob_arr = np.empty(n, dtype=np.float64)
        protein_group_arr = np.empty(n, dtype=object)
        protein_name_arr = np.empty(n, dtype=object)
        genes_arr = np.empty(n, dtype=object)
        uniprot_id_arr = np.empty(n, dtype=object)
        parent_key_arr = np.empty(n, dtype=object)

        all_spec_peaks = []
        all_spec_frag_names = []
        spec_offsets = np.empty(n, dtype=np.int64)
        spec_lengths = np.empty(n, dtype=np.int32)
        spec_cursor = 0

        all_frag_peaks = []
        all_frag_keys = []
        frag_offsets_arr = np.empty(n, dtype=np.int64)
        frag_lengths_arr = np.empty(n, dtype=np.int32)
        frag_cursor = 0

        key_to_idx = {}

        for i, uid in enumerate(precursor_order):
            pdata = precursor_data[uid]
            key_to_idx[uid] = i

            mod_seq_arr[i] = pdata['mod_seq']
            seq_arr[i] = pdata['seq']
            prec_mz_arr[i] = pdata['prec_mz']
            prec_z_arr[i] = pdata['prec_z']
            iRT_arr[i] = pdata['iRT']
            ion_mob_arr[i] = pdata['ion_mob']
            protein_group_arr[i] = pdata['protein_group']
            protein_name_arr[i] = pdata['protein_name']
            genes_arr[i] = pdata['genes']
            uniprot_id_arr[i] = pdata['uniprot_id']
            parent_key_arr[i] = None

            # Original frags
            frags = pdata['frags']
            frag_offsets_arr[i] = frag_cursor
            frag_keys_arr = encode_frag_names(list(frags.keys()))
            frag_vals_arr = np.array(list(frags.values()), dtype=np.float64)
            n_frags = len(frag_keys_arr)
            frag_lengths_arr[i] = n_frags
            if n_frags > 0:
                all_frag_keys.append(frag_keys_arr)
                all_frag_peaks.append(frag_vals_arr)
            frag_cursor += n_frags

            # Convert frags to spectrum
            spec, ordered_frags = frag_to_peak(frags, return_frags=True)
            n_peaks = len(spec)
            spec_offsets[i] = spec_cursor
            spec_lengths[i] = n_peaks
            all_spec_peaks.append(spec)
            all_spec_frag_names.append(encode_frag_names(ordered_frags))
            spec_cursor += n_peaks

        spectrum_data = np.concatenate(all_spec_peaks, axis=0) if all_spec_peaks else np.empty((0, 2), dtype=np.float64)
        frag_names_data = np.concatenate(all_spec_frag_names) if all_spec_frag_names else np.empty(0, dtype=np.int32)
        frag_data = np.concatenate(all_frag_peaks, axis=0) if all_frag_peaks else np.empty((0, 2), dtype=np.float64)
        frag_keys_data = np.concatenate(all_frag_keys) if all_frag_keys else np.empty(0, dtype=np.int32)

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
            spectrum_data=spectrum_data,
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
            parent_key=parent_key_arr,
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
            spectrum_data=np.empty((0, 2), dtype=np.float64),
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
            parent_key=np.empty(n, dtype=object),
        )


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
        if field == 'parent_key':
            return store.parent_key[idx]
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
        if field == 'parent_key':
            store.parent_key[idx] = value
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
            if field == 'parent_key':
                return self._store.parent_key[self._idx] is not None
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
        if self._store.parent_key[self._idx] is not None:
            result.append('parent_key')
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
