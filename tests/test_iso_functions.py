import pytest
import numpy as np
import sys, os
from pyteomics import mass
from pyteomics.auxiliary.structures import PyteomicsError


from src.iso_functions import get_seq_comp, precursor_isotopes
from brainpy._c.isotopic_distribution import TheoreticalPeak as Peak
from src.mass_tags import massTag, read_json_to_massTag
import src.iso_functions as iso
import copy




class Test_get_seq_comp():

    ##Tags are ignored by get_seq_comp. They are handled by precursor_isotopes downstream, but maybe not all things that call it?
    
    def test_no_tag(self):
        split_seq = ['A', 'A', 'A', 'A', 'A', 'D', 'L', 'A', 'N', 'R']
        ion_type = "M"
        seq_comp = get_seq_comp(split_seq, ion_type)
        assert seq_comp == mass.Composition({'H': 66, 'C': 38, 'O': 14, 'N': 14})


    def test_one_tag(self):
        split_seq = ['A(PSMtag_5plex-0)', 'A', 'A', 'A', 'A', 'D', 'L', 'A', 'N', 'R']
        ion_type = "M"
        seq_comp = get_seq_comp(split_seq, ion_type)
        assert seq_comp == mass.Composition({'H': 66, 'C': 38, 'O': 14, 'N': 14})

    def test_two_tags(self):
        split_seq = ['A(PSMtag_5plex-0)', 'A', 'A', 'A', 'A', 'D', 'L', 'A', 'N', 'R(PSMtag_5plex-0)']
        ion_type = "M"
        seq_comp = get_seq_comp(split_seq, ion_type)
        assert seq_comp == mass.Composition({'H': 66, 'C': 38, 'O': 14, 'N': 14})

    def test_one_aa(self):
        seq_comp = get_seq_comp(["A"], "M")
        assert seq_comp == mass.Composition({'H': 7, 'C': 3, 'O': 2, 'N': 1})

    def test_y_ion(self):
        split_seq = ["A", "C", "K"]
        ion_type = "y"
        seq_comp = get_seq_comp(split_seq, ion_type)
        assert seq_comp == mass.Composition({'H': 24, 'C': 12, 'O': 4, 'N': 4, 'S': 1})

    def test_carbamidomethylation(self):
        seq_comp = get_seq_comp(["A", "C(Unimod:4)", "K"], "M")
        assert seq_comp == mass.Composition({'H': 27, 'C': 14, 'O': 5, 'N': 5, 'S': 1})
    
    def test_phosphorylation(self):
        seq_comp = get_seq_comp(["A", "S(Unimod:21)", "K"], "M")
        assert seq_comp == mass.Composition({'H': 25, 'C': 12, 'O': 8, 'N': 4, 'P': 1})

    def test_weird_things_from_parse_peptide_tests(self):
        seq_comp = get_seq_comp(["A(+15.99)", "S[+80]", "K"], "M")
        assert seq_comp == mass.Composition({'H': 24, 'C': 12, 'O': 5, 'N': 4})

    def test_tag_before_mod(self):
        split_seq = ["M(PSMtag_5plex-0)(Unimod:35)", "E", "A", "T", "S", "T", "I", "C", "K"]
        ion_type = "M"
        seq_comp = get_seq_comp(split_seq, ion_type)
        assert seq_comp == mass.Composition({'H': 70, 'C': 39, 'S': 2, 'O': 16, 'N': 10})

    def test_mod_before_tag(self):
        split_seq = ["M(Unimod:35)(PSMtag_5plex-0)", "E", "A", "T", "S", "T", "I", "C", "K"]
        ion_type = "M"
        seq_comp = get_seq_comp(split_seq, ion_type)
        assert seq_comp == mass.Composition({'H': 70, 'C': 39, 'S': 2, 'O': 16, 'N': 10})

    def test_two_mods(self):
        seq_comp = get_seq_comp(["A(Unimod:35)", "S(Unimod:21)", "K"], "M")
        assert seq_comp == mass.Composition({'H': 25, 'C': 12, 'O': 9, 'N': 4, 'P': 1})



class Test_precursor_isotopes():

    tag = read_json_to_massTag("src/MassTags/", "PSMtag_5plex.json")

    tag_no_comps = copy.deepcopy(tag)
    tag_no_comps.channel_comp = None

    def compare_outputs(self, output, expected_output):
        assert len(output) == len(expected_output)
        for peak_a, peak_b in list(zip(output, expected_output)):
            assert np.isclose(peak_a.mz, peak_b.mz, atol=1e-6)
            assert np.isclose(peak_a.intensity, peak_b.intensity, atol=1e-6)
            assert peak_a.charge == peak_b.charge

    def test_stripped_seq(self):
        output = precursor_isotopes("MEATSTICK", 2, None, 2)
        expected_output = [Peak(mz=492.230453, intensity=0.672087, charge=2), Peak(mz=492.731859, intensity=0.327913, charge=2)]
        self.compare_outputs(output, expected_output)

    def test_five_isos(self):
        output = precursor_isotopes("MEATSTICK", 2, None, 5)
        expected_output = [Peak(mz=492.230453, intensity=0.548792, charge=2), Peak(mz=492.731859, intensity=0.267757, charge=2), Peak(mz=493.231317, intensity=0.130020, charge=2), Peak(mz=493.731776, intensity=0.041806, charge=2), Peak(mz=494.231870, intensity=0.011625, charge=2)]
        self.compare_outputs(output, expected_output)

    def test_untag_seq_with_tag(self):
        output = precursor_isotopes("MEATSTICK", 2, self.tag, 2)
        expected_output = [Peak(mz=492.230453, intensity=0.672087, charge=2), Peak(mz=492.731859, intensity=0.327913, charge=2)]
        self.compare_outputs(output, expected_output)

    def test_tag_seq_no_tag(self):
        output = precursor_isotopes("M(PSMtag_5plex-0)EATSTICK", 2, None, 2)
        expected_output = [Peak(mz=492.230453, intensity=0.672087, charge=2), Peak(mz=492.731859, intensity=0.327913, charge=2)]
        self.compare_outputs(output, expected_output)

    def test_one_tag(self):
        output = precursor_isotopes("M(PSMtag_5plex-0)EATSTICK", 2, self.tag, 2)
        expected_output = [Peak(mz=646.288499, intensity=0.590711, charge=2), Peak(mz=646.789957, intensity=0.409289, charge=2)]
        self.compare_outputs(output, expected_output)

    def test_two_tags(self):
        output = precursor_isotopes("M(PSMtag_5plex-0)EATSTICK(PSMtag_5plex-0)", 2, self.tag, 2)
        expected_output = [Peak(mz=800.346546, intensity=0.526913, charge=2), Peak(mz=800.848031, intensity=0.473087, charge=2)]
        self.compare_outputs(output, expected_output)

    def test_different_channel(self):
        output = precursor_isotopes("M(PSMtag_5plex-4)EATSTICK", 2, self.tag, 2)
        expected_output = [Peak(mz=648.293977, intensity=0.598493, charge=2), Peak(mz=648.795427, intensity=0.401507, charge=2)]
        self.compare_outputs(output, expected_output)

    def test_absent_channel(self):
        with pytest.raises(KeyError) as exc_info:
            precursor_isotopes("M(PSMtag_5plex-2)EATSTICK", 2, self.tag, 2)
        assert "'2'" in str(exc_info.value)
        
    def test_stripped_seq_z3(self):
        output = precursor_isotopes("MEATSTICK", 3, None, 2)
        expected_output = [Peak(mz=328.489394, intensity=0.672087, charge=3), Peak(mz=328.823665, intensity=0.327913, charge=3)]
        self.compare_outputs(output, expected_output)

    def test_one_tag_z3(self):
        output = precursor_isotopes("M(PSMtag_5plex-0)EATSTICK", 3, self.tag, 2)
        expected_output = [Peak(mz=431.194758, intensity=0.590711, charge=3), Peak(mz=431.529063, intensity=0.409289, charge=3)]
        self.compare_outputs(output, expected_output)

    def test_two_different_tags(self):
        output = precursor_isotopes("M(PSMtag_5plex-0)EATSTICK(PSMtag_5plex-4)", 3, self.tag, 2)
        expected_output = [Peak(mz=535.237108, intensity=0.533096, charge=3), Peak(mz=535.571428, intensity=0.466904, charge=3)]
        self.compare_outputs(output, expected_output)

    def test_oxidation(self):
        output = precursor_isotopes("M(Unimod:35)EATSTICK", 2, self.tag, 2)
        expected_output = [Peak(mz=500.227911, intensity=0.671915, charge=2), Peak(mz=500.729317, intensity=0.328085, charge=2)]
        self.compare_outputs(output, expected_output)

    def test_mod_and_tag(self):
        output = precursor_isotopes("M(Unimod:35)EATSTICK(PSMtag_5plex-0)", 2, self.tag, 2)
        expected_output = [Peak(mz=654.285957, intensity=0.590578, charge=2), Peak(mz=654.787415, intensity=0.409422, charge=2)]
        self.compare_outputs(output, expected_output)

    def test_tag_before_mod(self):
        output = precursor_isotopes("M(PSMtag_5plex-0)(Unimod:35)EATSTICK", 2, self.tag, 2)
        expected_output = [Peak(mz=654.285957, intensity=0.590578, charge=2), Peak(mz=654.787415, intensity=0.409422, charge=2)]
        self.compare_outputs(output, expected_output)

    def test_mod_before_tag(self):
        output = precursor_isotopes("M(Unimod:35)(PSMtag_5plex-0)EATSTICK", 2, self.tag, 2)
        expected_output = [Peak(mz=654.285957, intensity=0.590578, charge=2), Peak(mz=654.787415, intensity=0.409422, charge=2)]
        self.compare_outputs(output, expected_output)

    def test_decoy_peptide(self):
        output = precursor_isotopes("Decoy_MEATSTICK", 2, None, 2)
        expected_output = [Peak(mz=492.230453, intensity=0.672087, charge=2), Peak(mz=492.731859, intensity=0.327913, charge=2)]
        self.compare_outputs(output, expected_output)

    def test_decoy_peptide_explicit(self):
        output = precursor_isotopes("Decoy_MEATSTICK", 2, None, 2, decoys=True)
        expected_output = [Peak(mz=492.230453, intensity=0.672087, charge=2), Peak(mz=492.731859, intensity=0.327913, charge=2)]
        self.compare_outputs(output, expected_output)

    def test_decoy_peptide_decoys_off(self):
        with pytest.raises(PyteomicsError, match="Unknown label: ecoy_M"):
            precursor_isotopes("Decoy_MEATSTICK", 2, None, 2, decoys=False)

    def test_tag_channel_comp_is_none(self):
        output = precursor_isotopes("M(PSMtag_5plex-0)EATSTICK", 2, self.tag_no_comps, 2)
        expected_output = [Peak(mz=492.230453, intensity=0.672087, charge=2), Peak(mz=492.731859, intensity=0.327913, charge=2)]
        self.compare_outputs(output, expected_output)


