"""Top-k detagged candidate cap for the main search."""
import numpy as np

from src.spectral_fitting import _detag_sequence, _cap_to_top_detagged


def test_detag_sequence():
    assert _detag_sequence("P(PSMtag_9plex-0)EPTIDEK(PSMtag_9plex-0)", "PSMtag_9plex") == "PEPTIDEK"
    assert _detag_sequence("PEPTIDEK", "PSMtag_9plex") == "PEPTIDEK"
    assert _detag_sequence("C(UniMod:4)K(PSMtag_9plex-3)", "PSMtag_9plex") == "C(UniMod:4)K"


def _mk_inputs(keys, matched_counts, matched_ints):
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
    return dict(
        peaks_in_dia=list(range(n)), pep_cand=keys,
        pep_cand_loc=[np.arange(2)] * n, pep_cand_list=pep_cand_list,
        flat_rows=np.array(rows, np.int32), flat_cols=np.array(cols, np.int32),
        flat_vals=np.array(vals, np.float64), flat_offsets=np.array(offs, np.int32),
        norm_intensities=[np.ones(2)] * n, lib_peaks_matched=lib_peaks_matched,
        ms1_error=np.arange(n, dtype=float), passing=np.arange(n),
        prec_im=[0.0] * n)


def test_cap_keeps_best_sequences_and_channels(monkeypatch):
    import src.config as config
    class FakeTag: name = "T"
    monkeypatch.setattr(config, "tag", FakeTag, raising=False)

    keys = [("A(T-0)K", 2.0), ("A(T-1)K", 2.0),   # seq A: best 5 matches
            ("B(T-0)K", 2.0),                     # seq B: 3 matches
            ("C(T-0)K", 2.0), ("C(T-1)K", 2.0)]   # seq C: 3 matches, higher int
    out = _cap_to_top_detagged(2, **_mk_inputs(
        keys, matched_counts=[5, 2, 3, 3, 1], matched_ints=[10, 1, 5, 8, 1]))
    kept_keys = out[1]
    # top-2 sequences: A (5 matches) and C (3 matches, int 8 > B's 5);
    # ALL channel copies of kept sequences survive
    assert kept_keys == [("A(T-0)K", 2.0), ("A(T-1)K", 2.0),
                         ("C(T-0)K", 2.0), ("C(T-1)K", 2.0)]
    # columns renumbered consecutively, offsets consistent
    flat_cols, flat_offsets = out[5], out[7]
    assert flat_cols.tolist() == [0]*5 + [1]*2 + [2]*3 + [3]*1
    assert flat_offsets.tolist() == [0, 5, 7, 10, 11]
    # passing subset preserves original candidate indices
    assert out[12] == [0.0]*4
    assert list(out[11]) == [0, 1, 3, 4]


def test_cap_noop_when_under_k(monkeypatch):
    import src.config as config
    monkeypatch.setattr(config, "tag", None, raising=False)
    inputs = _mk_inputs([("AK", 2.0), ("BK", 2.0)], [2, 1], [1.0, 1.0])
    out = _cap_to_top_detagged(5, **inputs)
    assert out[1] == [("AK", 2.0), ("BK", 2.0)]
    assert out[4] is inputs["flat_rows"]
