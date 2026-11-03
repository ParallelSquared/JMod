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

"""'rev_tag' decoys keep the mass tag at its positions, so each b and y ion
carries the same number of tags in target and decoy."""

import random
import re
from unittest.mock import Mock

import pytest

from src.utils.parse_peptides import change_seq, decoy_permutation, parse_peptide, split_nterm


@pytest.fixture
def tag():
    t = Mock()
    t.name = "mTRAQ"
    return t


def _tagged_positions(seq, tag_name):
    """Body positions carrying the tag, and whether the N-terminus does."""
    nterm, body = split_nterm(seq)
    toks = parse_peptide(body)
    return ([i for i, t in enumerate(toks) if f"({tag_name}" in t],
            f"({tag_name}" in nterm, len(toks))


def _tags_per_ion(seq, tag_name):
    """Tag count on every b ion (prefix) and y ion (suffix)."""
    pos, nterm, n = _tagged_positions(seq, tag_name)
    b = [int(nterm) + sum(1 for p in pos if p < k) for k in range(1, n)]
    y = [sum(1 for p in pos if p >= n - k) for k in range(1, n)]
    return b, y


@pytest.mark.parametrize("target, expected", [
    # penultimate tagged K: stays put (under 'rev' it moves to the N-terminus)
    ("(mTRAQ)PEPTIDEK(mTRAQ)K(mTRAQ)", "(mTRAQ)EDITPEPK(mTRAQ)K(mTRAQ)"),
    ("(mTRAQ)PEPTIDEK(mTRAQ)R", "(mTRAQ)EDITPEPK(mTRAQ)R"),
    # internal tagged K keeps its index; the untagged residues reverse around it
    ("(mTRAQ)PEK(mTRAQ)PTIDER", "(mTRAQ)EDK(mTRAQ)ITPEPR"),
    # N-terminal tagged K
    ("(mTRAQ)K(mTRAQ)LIQNELR", "(mTRAQ)K(mTRAQ)LENQILR"),
    # other mods travel with their residue, as under 'rev'
    ("(mTRAQ)C(UniMod:4)EPTIDEK(mTRAQ)", "(mTRAQ)EDITPEC(UniMod:4)K(mTRAQ)"),
])
def test_rev_tag_examples(tag, target, expected):
    assert change_seq(target, "rev_tag", tag=tag) == expected


def test_rev_moves_a_penultimate_tag_to_the_n_terminus(tag):
    """The case rev_tag exists for: under 'rev' the tag changes ion series."""
    t = "(mTRAQ)PEPTIDEK(mTRAQ)K(mTRAQ)"
    d = change_seq(t, "rev", tag=tag)
    assert d == "(mTRAQ)K(mTRAQ)EDITPEPK(mTRAQ)"
    assert _tags_per_ion(d, "mTRAQ") != _tags_per_ion(t, "mTRAQ")


def test_every_ion_carries_the_same_number_of_tags(tag):
    rng = random.Random(0)
    aas = "ACDEFGHILMNPQRSTVWY"
    for _ in range(500):
        n = rng.randint(4, 20)
        body = [rng.choice(aas + "KK") for _ in range(n - 1)] + [rng.choice("KR")]
        seq = "(mTRAQ)" + "".join(a + ("(mTRAQ)" if a == "K" else "") for a in body)
        decoy = change_seq(seq, "rev_tag", tag=tag)
        assert _tags_per_ion(decoy, "mTRAQ") == _tags_per_ion(seq, "mTRAQ"), (seq, decoy)
        # same residue composition: it is a permutation
        strip = lambda s: re.sub(r"\(.*?\)", "", s)
        assert sorted(strip(decoy)) == sorted(strip(seq))


def test_without_a_tag_rev_tag_is_rev():
    for seq in ["PEPTIDEK", "PEPTIDEKK", "C(UniMod:4)EPTIDER", "KLIQNELR"]:
        assert change_seq(seq, "rev_tag") == change_seq(seq, "rev")
        toks = parse_peptide(seq)
        assert decoy_permutation(toks, "rev_tag") == decoy_permutation(toks, "rev")


def test_columnar_decoys_match_change_seq_and_keep_tag_masses():
    """The library's columnar decoy builder gives the same decoy sequences as
    change_seq, and its decoy fragment m/z differ from the target's only by
    residue masses -- never by a tag mass."""
    from src import mass
    from src.mass_tags import read_json_to_massTag
    from src.models.spec_lib.library_store import SpectrumLibraryStore
    from src.models.spec_lib.spec_lib import create_decoy_lib
    from tests.models.spec_lib.test_decoy_differential import _entry

    one = read_json_to_massTag("tests/MassTags", "test_one_channel.json")
    t = "(test_one_channel-0)"
    tm = one.mass_dict["test_one_channel-0"]
    b = lambda s, n_tags: mass.fast_mass(sequence=s, ion_type='b', charge=1) + n_tags * tm
    y = lambda s, n_tags: mass.fast_mass(sequence=s, ion_type='y', charge=1) + n_tags * tm
    seqs = {
        # penultimate tagged K
        f"{t}PEPTIDEK{t}K{t}": ("PEPTIDEKK", {'b3_1': [b("PEP", 1), 1.0], 'b5_1': [b("PEPTI", 1), 0.8],
                                              'y2_1': [y("KK", 2), 0.9], 'y4_1': [y("DEKK", 2), 0.5]}),
        # internal tagged K
        f"{t}PEK{t}PTIDER": ("PEKPTIDER", {'b2_1': [b("PE", 1), 0.7], 'b4_1': [b("PEKP", 2), 1.0],
                                            'y3_1': [y("DER", 0), 0.6], 'y6_1': [y("PTIDER", 0), 0.4],
                                            'y7_1': [y("KPTIDER", 1), 0.3]}),
    }
    d = {(s, 2.0): _entry(s, plain, 500.0, 2.0, frags) for s, (plain, frags) in seqs.items()}
    combined = create_decoy_lib(SpectrumLibraryStore.from_dict(d), rules="rev_tag", tag=one)

    for s, (plain, frags) in seqs.items():
        dseq = change_seq(s, "rev_tag", tag=one)
        assert (dseq, 2.0) in combined, f"decoy {dseq} missing"
        dplain = re.sub(r"\(.*?\)", "", dseq)
        dfrags = combined[(dseq, 2.0)]["frags"]
        for code, (mz, _) in frags.items():
            kind, k = code[0], int(code[1:code.index("_")])
            if kind == "b":
                residue_shift = (mass.fast_mass(sequence=dplain[:k], ion_type='b', charge=1)
                                 - mass.fast_mass(sequence=plain[:k], ion_type='b', charge=1))
            else:
                residue_shift = (mass.fast_mass(sequence=dplain[-k:], ion_type='y', charge=1)
                                 - mass.fast_mass(sequence=plain[-k:], ion_type='y', charge=1))
            assert dfrags[code][0] == pytest.approx(mz + residue_shift, abs=1e-4), (s, code)
