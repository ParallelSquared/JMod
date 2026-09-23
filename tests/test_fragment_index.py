import os

import numpy as np

from src.fragment_index import FragmentIndex
from src.models.spec_lib.library_store import SpectrumLibraryStore

from tests.models.spec_lib.test_library_snapshots import FIXTURE_DIR


def _library():
    """Four-precursor library with an rt_mz that keeps every precursor apart in RT."""
    store = SpectrumLibraryStore.from_tsv(os.path.join(FIXTURE_DIR, "library_edgecases.tsv"))
    store.finalize_spectra()
    store.bulk_set_top_n(10)
    keys = list(store)
    rt_mz = np.column_stack([np.arange(len(keys)) * 10.0,
                             np.asarray(store.prec_mz, dtype=np.float64)])
    return store, keys, rt_mz


def _query_with_own_fragments(index, store, rt_mz, i):
    """Query the index with precursor i's own fragments, at its m/z and RT."""
    frag_mz = store.get_spectrum(i)[:, 0].astype(np.float64)
    mz, rt = rt_mz[i, 1], rt_mz[i, 0]
    return index.query(frag_mz, mz - 1.0, mz + 1.0, rt, 1.0, 1).tolist()


class TestBuildInclude:
    def test_without_include_every_precursor_is_indexed(self):
        store, keys, rt_mz = _library()
        index = FragmentIndex.build(store, keys, rt_mz, 10)
        assert sorted(index.all_prec_global_idx.tolist()) == list(range(len(keys)))

    def test_excluded_precursors_are_left_out(self):
        store, keys, rt_mz = _library()
        include = np.array([True, False, True, False])
        index = FragmentIndex.build(store, keys, rt_mz, 10, include=include)
        assert sorted(index.all_prec_global_idx.tolist()) == [0, 2]

    def test_query_returns_library_index_of_included_precursor(self):
        # With 0 and 1 excluded, precursor 2 is the first one indexed; the query
        # must still report it as 2, its position in the library
        store, keys, rt_mz = _library()
        include = np.array([False, False, True, True])
        index = FragmentIndex.build(store, keys, rt_mz, 10, include=include)
        assert _query_with_own_fragments(index, store, rt_mz, 2) == [2]

    def test_query_never_returns_excluded_precursor(self):
        store, keys, rt_mz = _library()
        include = np.array([False, True, True, True])
        index = FragmentIndex.build(store, keys, rt_mz, 10, include=include)
        assert 0 not in _query_with_own_fragments(index, store, rt_mz, 0)

    def test_all_excluded_gives_empty_index(self):
        store, keys, rt_mz = _library()
        include = np.zeros(len(keys), dtype=bool)
        index = FragmentIndex.build(store, keys, rt_mz, 10, include=include)
        assert _query_with_own_fragments(index, store, rt_mz, 0) == []
