"""Class-blind top-k detagged-sequence cap for the main search."""
import numpy as np

from src.spectral_fitting import (
    _detag_sequence, _detagged_group_stats, _combined_top_sequences,
    _subset_candidates,
)


def test_detag_sequence():
    assert _detag_sequence("P(PSMtag_9plex-0)EPTIDEK(PSMtag_9plex-0)", "PSMtag_9plex") == "PEPTIDEK"
    assert _detag_sequence("PEPTIDEK", "PSMtag_9plex") == "PEPTIDEK"
    assert _detag_sequence("C(UniMod:4)K(PSMtag_9plex-3)", "PSMtag_9plex") == "C(UniMod:4)K"


def _mk(keys, matched_counts, matched_ints):
    n = len(keys)
    pep_cand_list, lib_peaks_matched = [], []
    rows, cols, vals, offs = [], [], [], [0]
    for j, (nm, mi) in enumerate(zip(matched_counts, matched_ints)):
        n_peaks = max(nm, 1) + 1
        peaks = np.zeros((n_peaks, 2))
        mask = np.zeros(n_peaks, dtype=bool)
        mask[:nm] = True
        peaks[:nm, 1] = mi / max(nm, 1)
        pep_cand_list.append(peaks)
        lib_peaks_matched.append(mask)
        for r in range(nm):
            rows.append(r); cols.append(j); vals.append(1.0)
        offs.append(len(rows))
    subset_args = dict(
        peaks_in_dia=list(range(n)), pep_cand=keys,
        pep_cand_loc=[np.arange(2)] * n, pep_cand_list=pep_cand_list,
        flat_rows=np.array(rows, np.int32), flat_cols=np.array(cols, np.int32),
        flat_vals=np.array(vals, np.float64), flat_offsets=np.array(offs, np.int32),
        norm_intensities=[np.ones(2)] * n, lib_peaks_matched=lib_peaks_matched,
        ms1_error=np.arange(n, dtype=float), passing=np.arange(n),
        prec_im=[0.0] * n)
    return pep_cand_list, lib_peaks_matched, subset_args


def test_combined_pool_is_class_blind():
    # targets: A (5 matches), B (2 matches); decoys: D1 (4 matches), D2 (1)
    t_list, t_matched, _ = _mk([("A(T-0)K", 2.0), ("B(T-0)K", 2.0)], [5, 2], [9, 2])
    d_list, d_matched, _ = _mk([("D1K", 2.0), ("D2K", 2.0)], [4, 1], [7, 1])
    ref = _detagged_group_stats([("A(T-0)K", 2.0), ("B(T-0)K", 2.0)], t_list, t_matched, "T")
    dec = _detagged_group_stats([("D1K", 2.0), ("D2K", 2.0)], d_list, d_matched, "T")
    keep_ref, keep_dec = _combined_top_sequences(2, ref, dec)
    # one pool of 2: A (5) and D1 (4) win; B and D2 lose regardless of class
    assert keep_ref == [0]
    assert keep_dec == [0]


def test_channels_travel_together_and_ties_break_on_intensity():
    keys = [("A(T-0)K", 2.0), ("A(T-1)K", 2.0),
            ("B(T-0)K", 2.0),
            ("C(T-0)K", 2.0), ("C(T-1)K", 2.0)]
    plist, matched, _ = _mk(keys, [5, 2, 3, 3, 1], [10, 1, 5, 8, 1])
    groups = _detagged_group_stats(keys, plist, matched, "T")
    keep_ref, keep_dec = _combined_top_sequences(2, groups, {})
    # A (5 matches) then C (3 matches, intensity 8 beats B's 5); all channels kept
    assert keep_ref == [0, 1, 3, 4]
    assert keep_dec == []


def test_no_cap_when_under_k():
    keys = [("AK", 2.0), ("BK", 2.0)]
    plist, matched, _ = _mk(keys, [2, 1], [1, 1])
    groups = _detagged_group_stats(keys, plist, matched, None)
    assert _combined_top_sequences(5, groups, {}) is None


def test_subset_rebuilds_flats():
    keys = [("AK", 2.0), ("BK", 2.0), ("CK", 2.0)]
    _, _, args = _mk(keys, [2, 3, 1], [1, 1, 1])
    out = _subset_candidates([0, 2], **args)
    assert out[1] == [("AK", 2.0), ("CK", 2.0)]
    assert out[5].tolist() == [0, 0, 1]          # renumbered cols
    assert out[7].tolist() == [0, 2, 3]          # offsets
    assert list(out[11]) == [0, 2]               # passing keeps original ids


def test_subset_identity():
    keys = [("AK", 2.0)]
    _, _, args = _mk(keys, [2], [1])
    out = _subset_candidates([0], **args)
    assert out[4] is args["flat_rows"]
