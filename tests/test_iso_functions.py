import pytest
import numpy as np
import sys, os
from pyteomics import mass
from pyteomics.auxiliary.structures import PyteomicsError

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))



from src.iso_functions import get_seq_comp, precursor_isotopes, fragment_seq, gen_isotopes_dict, iso_library, iso_library_multi
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

    def test_phospho_loss(self):
        seq_comp = get_seq_comp(["A(Unimod:35)", "S(Unimod:21)", "K"], "M", "H3PO4")
        assert seq_comp == mass.Composition({'H': 22, 'C': 12, 'O': 5, 'N': 4})

    def test_H2O_loss(self):
        seq_comp = get_seq_comp(["A(Unimod:35)", "S(Unimod:21)", "K"], "M", "H2O")
        assert seq_comp == mass.Composition({'H': 23, 'C': 12, 'O': 8, 'N': 4, 'P': 1})

    def test_NH3_loss(self):
        seq_comp = get_seq_comp(["A(Unimod:35)", "S(Unimod:21)", "K"], "M", "NH3")
        assert seq_comp == mass.Composition({'H': 22, 'C': 12, 'O': 9, 'N': 3, 'P': 1})



class Test_precursor_isotopes():

    tag = read_json_to_massTag("tests/MassTags/", "PSMtag_5plex.json")

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
        with pytest.raises(PyteomicsError, match="No information for ecoy_M"):
            precursor_isotopes("Decoy_MEATSTICK", 2, None, 2, decoys=False)

    def test_tag_channel_comp_is_none(self):
        output = precursor_isotopes("M(PSMtag_5plex-0)EATSTICK", 2, self.tag_no_comps, 2)
        expected_output = [Peak(mz=492.230453, intensity=0.672087, charge=2), Peak(mz=492.731859, intensity=0.327913, charge=2)]
        self.compare_outputs(output, expected_output)


class Test_fragment_seq():

    def compare_outputs(self, oseq, olist, eseq, elist):
        assert oseq == eseq
        for a, b in zip(olist, elist):
            assert a == b

    def test_b_ion(self):
        oseq, olist = fragment_seq("PEPTIDEK", "b5_2")
        eseq = ["P", "E", "P", "T", "I"]
        elist = ["b", 5, "", "2"]
        self.compare_outputs(oseq, olist, eseq, elist)

    def test_c_ion(self):
        oseq, olist = fragment_seq("PEPTIDEK", "c5_2")
        eseq = ["P", "E", "P", "T", "I"]
        elist = ["c", 5, "", "2"]
        self.compare_outputs(oseq, olist, eseq, elist)

    def test_y_ion(self):
        oseq, olist = fragment_seq("PEPTIDEK", "y3_1")
        eseq = ["D", "E", "K"]
        elist = ["y", 3, "", "1"]
        self.compare_outputs(oseq, olist, eseq, elist)

    def test_bad_ion(self):
        with pytest.raises(ValueError):
            oseq, olist = fragment_seq("PEPTIDEK", "Y3_1")

    def test_tag_as_list(self):
        oseq, olist = fragment_seq(["P", "E", "P", "T", "I", "D", "E", "K"], "b5_2")
        eseq = ["P", "E", "P", "T", "I"]
        elist = ["b", 5, "", "2"]
        self.compare_outputs(oseq, olist, eseq, elist)

    def test_assertion_error(self):
        with pytest.raises(AssertionError):
            oseq, olist = fragment_seq("PEPTIDEK", "b20_2")

    def test_neutral_loss(self):
        oseq, olist = fragment_seq("PEPTIDEK", "y3-H2O_1")
        eseq = ["D", "E", "K"]
        elist = ["y", 3, "H2O", "1"]
        self.compare_outputs(oseq, olist, eseq, elist)

    def test_one_tag(self):
        oseq, olist = fragment_seq("P(PSMtag_5plex-0)EPTIDEK", "b5_2")
        eseq = ["P(PSMtag_5plex-0)", "E", "P", "T", "I"]
        elist = ["b", 5, "", "2"]
        self.compare_outputs(oseq, olist, eseq, elist)

    def test_mod(self):
        oseq, olist = fragment_seq("PEPT(Unimod:21)IDEK", "b5_2")
        eseq = ["P", "E", "P", "T(Unimod:21)", "I"]
        elist = ["b", 5, "", "2"]
        self.compare_outputs(oseq, olist, eseq, elist)


class Test_gen_isotopes_dict():

    def compare_outputs(self, output, expected):
        assert output['frags'] == expected['frags']
        assert np.allclose(output['spectrum'], expected['spectrum'])
        assert np.array_equal(output['ordered_frags'], expected['ordered_frags'])

    def test_no_pep_tag(self):
        new_library = {}
        key = ("PEPTIDEK", 2.0)
        frags = {
            "y1_1": [mass.fast_mass(sequence="K", ion_type='y'), 0.5],
            "y2_1": [mass.fast_mass(sequence="EK", ion_type='y'), 0.6],
            "b2_1": [mass.fast_mass(sequence="PE", ion_type='b'), 1],
            "b5_2": [mass.fast_mass(sequence="PEPTI", ion_type='b'), 0.4],
            "b5-H2O_2": [mass.fast_mass(sequence="PEPTI", ion_type='b')-mass.calculate_mass(formula='H2O'), 0.1],
        }
        new_library[key] = {}
        new_library[key]["frags"] = frags
        tag = read_json_to_massTag("tests/MassTags/", "PSMtag_5plex.json")
        n_iso = 3

        new_library[key]["spectrum"],new_library[key]["ordered_frags"] = gen_isotopes_dict(key[0],frags,tag,n_iso)

        new_library_expected = {}
        new_library_expected[key] = {
            "frags": frags,
            "spectrum": np.array([[1.46105528e+02, 5.00000000e-01],
                                [1.47108335e+02, 3.72865011e-02],
                                [1.48110245e+02, 3.26301469e-03],
                                [2.26095357e+02, 1.00000000e+00],
                                [2.27098373e+02, 1.18597767e-01],
                                [2.28100371e+02, 1.46540604e-02],
                                [2.75148121e+02, 6.00000000e-01],
                                [2.76151024e+02, 8.05516867e-02],
                                [2.77153048e+02, 1.11737793e-02],
                                [5.19269299e+02, 1.00000000e-01],
                                [5.19770806e+02, 2.95581666e-02],
                                [5.20272081e+02, 5.65728474e-03],
                                [5.37279863e+02, 4.00000000e-01],
                                [5.37781372e+02, 1.18477047e-01],
                                [5.38282626e+02, 2.35234112e-02]]),
            "ordered_frags": np.array(['y1_1', 'y1_1_iso1', 'y1_1_iso2', 'b2_1', 'b2_1_iso1', 'b2_1_iso2',
                                        'y2_1', 'y2_1_iso1', 'y2_1_iso2', 'b5-H2O_2', 'b5-H2O_2_iso1',
                                        'b5-H2O_2_iso2', 'b5_2', 'b5_2_iso1', 'b5_2_iso2'], dtype='<U13')
        }

        self.compare_outputs(new_library[key], new_library_expected[key])


    def test_no_tag(self):
        new_library = {}
        key = ("PEPTIDEK", 2.0)
        frags = {
            "y1_1": [mass.fast_mass(sequence="K", ion_type='y'), 0.5],
            "y2_1": [mass.fast_mass(sequence="EK", ion_type='y'), 0.6],
            "b2_1": [mass.fast_mass(sequence="PE", ion_type='b'), 1],
            "b5_2": [mass.fast_mass(sequence="PEPTI", ion_type='b'), 0.4],
            "b5-H2O_2": [mass.fast_mass(sequence="PEPTI", ion_type='b')-mass.calculate_mass(formula='H2O'), 0.1],
        }
        new_library[key] = {}
        new_library[key]["frags"] = frags
        tag = None
        n_iso = 3

        new_library[key]["spectrum"],new_library[key]["ordered_frags"] = gen_isotopes_dict(key[0],frags,tag,n_iso)

        new_library_expected = {}
        new_library_expected[key] = {
            "frags": frags,
            "spectrum": np.array([[1.46105528e+02, 5.00000000e-01],
                                [1.47108335e+02, 3.72865011e-02],
                                [1.48110245e+02, 3.26301469e-03],
                                [2.26095357e+02, 1.00000000e+00],
                                [2.27098373e+02, 1.18597767e-01],
                                [2.28100371e+02, 1.46540604e-02],
                                [2.75148121e+02, 6.00000000e-01],
                                [2.76151024e+02, 8.05516867e-02],
                                [2.77153048e+02, 1.11737793e-02],
                                [5.19269299e+02, 1.00000000e-01],
                                [5.19770806e+02, 2.95581666e-02],
                                [5.20272081e+02, 5.65728474e-03],
                                [5.37279863e+02, 4.00000000e-01],
                                [5.37781372e+02, 1.18477047e-01],
                                [5.38282626e+02, 2.35234112e-02]]),
            "ordered_frags": np.array(['y1_1', 'y1_1_iso1', 'y1_1_iso2', 'b2_1', 'b2_1_iso1', 'b2_1_iso2',
                                        'y2_1', 'y2_1_iso1', 'y2_1_iso2', 'b5-H2O_2', 'b5-H2O_2_iso1',
                                        'b5-H2O_2_iso2', 'b5_2', 'b5_2_iso1', 'b5_2_iso2'], dtype='<U13')
        }

        self.compare_outputs(new_library[key], new_library_expected[key])


    def test_one_tag(self):
        new_library = {}
        tag = read_json_to_massTag("tests/MassTags/", "PSMtag_5plex.json")
        key = ("P(PSMtag_5plex-0)EPTIDEK", 2.0)
        frags = {
            "y1_1": [mass.fast_mass(sequence="K", ion_type='y'), 0.5],
            "y2_1": [mass.fast_mass(sequence="EK", ion_type='y'), 0.6],
            "b2_1": [mass.fast_mass(sequence="PE", ion_type='b')+tag.mass, 1],
            "b5_2": [mass.fast_mass(sequence="PEPTI", ion_type='b')+tag.mass, 0.4],
            "b5-H2O_2": [mass.fast_mass(sequence="PEPTI", ion_type='b')+tag.mass-mass.calculate_mass(formula='H2O'), 0.1],
        }
        new_library[key] = {}
        new_library[key]["frags"] = frags
        n_iso = 3

        new_library[key]["spectrum"],new_library[key]["ordered_frags"] = gen_isotopes_dict(key[0],frags,tag,n_iso)

        new_library_expected = {}
        new_library_expected[key] = {
            "frags": frags,
            "spectrum": np.array([[1.46105528e+02, 5.00000000e-01],
                                    [1.47108335e+02, 3.72865011e-02],
                                    [1.48110245e+02, 3.26301469e-03],
                                    [2.75148121e+02, 6.00000000e-01],
                                    [2.76151024e+02, 8.05516867e-02],
                                    [2.77153048e+02, 1.11737793e-02],
                                    [5.34211449e+02, 1.00000000e+00],
                                    [5.35214557e+02, 3.23570461e-01],
                                    [5.36217222e+02, 6.50687578e-02],
                                    [8.27385391e+02, 1.00000000e-01],
                                    [8.27886928e+02, 5.00554360e-02],
                                    [8.28388326e+02, 1.43264411e-02],
                                    [8.45395956e+02, 4.00000000e-01],
                                    [8.45897494e+02, 2.00466125e-01],
                                    [8.46398881e+02, 5.82501283e-02]]),
            "ordered_frags": np.array(['y1_1', 'y1_1_iso1', 'y1_1_iso2', 'y2_1', 'y2_1_iso1', 'y2_1_iso2',
                                        'b2_1', 'b2_1_iso1', 'b2_1_iso2', 'b5-H2O_2', 'b5-H2O_2_iso1',
                                        'b5-H2O_2_iso2', 'b5_2', 'b5_2_iso1', 'b5_2_iso2'], dtype='<U13')
        }

        self.compare_outputs(new_library[key], new_library_expected[key])

    def test_two_tags(self):
        new_library = {}
        tag = read_json_to_massTag("tests/MassTags/", "PSMtag_5plex.json")
        key = ("P(PSMtag_5plex-0)EPTIDEK(PSMtag_5plex-0)", 2.0)
        frags = {
            "y1_1": [mass.fast_mass(sequence="K", ion_type='y')+tag.mass, 0.5],
            "y2_1": [mass.fast_mass(sequence="EK", ion_type='y')+tag.mass, 0.6],
            "b2_1": [mass.fast_mass(sequence="PE", ion_type='b')+tag.mass, 1],
            "b5_2": [mass.fast_mass(sequence="PEPTI", ion_type='b')+tag.mass, 0.4],
            "b5-H2O_2": [mass.fast_mass(sequence="PEPTI", ion_type='b')+tag.mass-mass.calculate_mass(formula='H2O'), 0.1],
        }
        new_library[key] = {}
        new_library[key]["frags"] = frags
        n_iso = 3

        new_library[key]["spectrum"],new_library[key]["ordered_frags"] = gen_isotopes_dict(key[0],frags,tag,n_iso)

        new_library_expected = {}
        new_library_expected[key] = {
            "frags": frags,
            "spectrum": np.array([[4.54221620e+02, 5.00000000e-01],
                                    [4.55224686e+02, 1.39772848e-01],
                                    [4.56227338e+02, 2.39584261e-02],
                                    [5.34211449e+02, 1.00000000e+00],
                                    [5.35214557e+02, 3.23570461e-01],
                                    [5.36217222e+02, 6.50687578e-02],
                                    [5.83264213e+02, 6.00000000e-01],
                                    [5.84267272e+02, 2.03535303e-01],
                                    [5.85269897e+02, 4.33479117e-02],
                                    [8.27385391e+02, 1.00000000e-01],
                                    [8.27886928e+02, 5.00554360e-02],
                                    [8.28388326e+02, 1.43264411e-02],
                                    [8.45395956e+02, 4.00000000e-01],
                                    [8.45897494e+02, 2.00466125e-01],
                                    [8.46398881e+02, 5.82501283e-02]]),
            "ordered_frags": np.array(['y1_1', 'y1_1_iso1', 'y1_1_iso2', 'b2_1', 'b2_1_iso1', 'b2_1_iso2',
                                        'y2_1', 'y2_1_iso1', 'y2_1_iso2', 'b5-H2O_2', 'b5-H2O_2_iso1',
                                        'b5-H2O_2_iso2', 'b5_2', 'b5_2_iso1', 'b5_2_iso2'], dtype='<U13')
        }

        self.compare_outputs(new_library[key], new_library_expected[key])

    def test_mod(self):
        new_library = {}
        key = ("PEPT(Unimod:21)IDEK", 2.0)
        frags = {
            "y1_1": [mass.fast_mass(sequence="K", ion_type='y'), 0.5],
            "y2_1": [mass.fast_mass(sequence="EK", ion_type='y'), 0.6],
            "b2_1": [mass.fast_mass(sequence="PE", ion_type='b'), 1],
            "b5_2": [mass.fast_mass(sequence="PEPTI", ion_type='b')+mass.calculate_mass(formula='H3PO4'), 0.4],
            "b5-H2O_2": [mass.fast_mass(sequence="PEPTI", ion_type='b')+mass.calculate_mass(formula='H3PO4')-mass.calculate_mass(formula='H2O'), 0.1],
        }
        new_library[key] = {}
        new_library[key]["frags"] = frags
        tag = None
        n_iso = 3

        new_library[key]["spectrum"],new_library[key]["ordered_frags"] = gen_isotopes_dict(key[0],frags,tag,n_iso)

        new_library_expected = {}
        new_library_expected[key] = {
            "frags": frags,
            "spectrum": np.array([[1.46105528e+02, 5.00000000e-01],
                                    [1.47108335e+02, 3.72865011e-02],
                                    [1.48110245e+02, 3.26301469e-03],
                                    [2.26095357e+02, 1.00000000e+00],
                                    [2.27098373e+02, 1.18597767e-01],
                                    [2.28100371e+02, 1.46540604e-02],
                                    [2.75148121e+02, 6.00000000e-01],
                                    [2.76151024e+02, 8.05516867e-02],
                                    [2.77153048e+02, 1.11737793e-02],
                                    [6.17246194e+02, 1.00000000e-01],
                                    [6.17747704e+02, 2.96839456e-02],
                                    [6.18248917e+02, 6.31101747e-03],
                                    [6.35256758e+02, 4.00000000e-01],
                                    [6.35758270e+02, 1.18980163e-01],
                                    [6.36259466e+02, 2.61386496e-02]]),
            "ordered_frags": np.array(['y1_1', 'y1_1_iso1', 'y1_1_iso2', 'b2_1', 'b2_1_iso1', 'b2_1_iso2',
                                        'y2_1', 'y2_1_iso1', 'y2_1_iso2', 'b5-H2O_2', 'b5-H2O_2_iso1',
                                        'b5-H2O_2_iso2', 'b5_2', 'b5_2_iso1', 'b5_2_iso2'], dtype='<U13')
        }

        self.compare_outputs(new_library[key], new_library_expected[key])

    def test_dif_iso_num(self):
        new_library = {}
        key = ("PEPTIDEK", 2.0)
        frags = {
            "y1_1": [mass.fast_mass(sequence="K", ion_type='y'), 0.5],
            "y2_1": [mass.fast_mass(sequence="EK", ion_type='y'), 0.6],
            "b2_1": [mass.fast_mass(sequence="PE", ion_type='b'), 1],
            "b5_2": [mass.fast_mass(sequence="PEPTI", ion_type='b'), 0.4],
            "b5-H2O_2": [mass.fast_mass(sequence="PEPTI", ion_type='b')-mass.calculate_mass(formula='H2O'), 0.1],
        }
        new_library[key] = {}
        new_library[key]["frags"] = frags
        tag = None
        n_iso = 2

        new_library[key]["spectrum"],new_library[key]["ordered_frags"] = gen_isotopes_dict(key[0],frags,tag,n_iso)

        new_library_expected = {}
        new_library_expected[key] = {
            "frags": frags,
            "spectrum": np.array([[1.46105528e+02, 5.00000000e-01],
                                [1.47108335e+02, 3.72865011e-02],
                                [2.26095357e+02, 1.00000000e+00],
                                [2.27098373e+02, 1.18597767e-01],
                                [2.75148121e+02, 6.00000000e-01],
                                [2.76151024e+02, 8.05516867e-02],
                                [5.19269299e+02, 1.00000000e-01],
                                [5.19770806e+02, 2.95581666e-02],
                                [5.37279863e+02, 4.00000000e-01],
                                [5.37781372e+02, 1.18477047e-01]]),
            "ordered_frags": np.array(['y1_1', 'y1_1_iso1', 'b2_1', 'b2_1_iso1',
                                        'y2_1', 'y2_1_iso1', 'b5-H2O_2', 'b5-H2O_2_iso1',
                                        'b5_2', 'b5_2_iso1',], dtype='<U13')
        }

        self.compare_outputs(new_library[key], new_library_expected[key])

class Test_iso_library():

    def compare_outputs(self, output, expected):
        assert output['frags'] == expected['frags']
        assert np.allclose(output['spectrum'], expected['spectrum'])
        assert np.array_equal(output['ordered_frags'], expected['ordered_frags'])

    def test_it(self):
        tag = read_json_to_massTag("tests/MassTags/", "PSMtag_5plex.json")
        new_library = {}

        keys = []
        all_frags = []

        #no tag
        key = ("PEPTIDEK", 2.0)
        keys.append(key)
        frags = {
            "y1_1": [mass.fast_mass(sequence="K", ion_type='y'), 0.5],
            "y2_1": [mass.fast_mass(sequence="EK", ion_type='y'), 0.6],
            "b2_1": [mass.fast_mass(sequence="PE", ion_type='b'), 1],
            "b5_2": [mass.fast_mass(sequence="PEPTI", ion_type='b'), 0.4],
            "b5-H2O_2": [mass.fast_mass(sequence="PEPTI", ion_type='b')-mass.calculate_mass(formula='H2O'), 0.1],
        }
        all_frags.append(frags)
        new_library[key] = {}
        new_library[key]["frags"] = frags

        #one tag
        key = ("P(PSMtag_5plex-0)EPTIDEK", 2.0)
        keys.append(key)
        frags = {
            "y1_1": [mass.fast_mass(sequence="K", ion_type='y'), 0.5],
            "y2_1": [mass.fast_mass(sequence="EK", ion_type='y'), 0.6],
            "b2_1": [mass.fast_mass(sequence="PE", ion_type='b')+tag.mass, 1],
            "b5_2": [mass.fast_mass(sequence="PEPTI", ion_type='b')+tag.mass, 0.4],
            "b5-H2O_2": [mass.fast_mass(sequence="PEPTI", ion_type='b')+tag.mass-mass.calculate_mass(formula='H2O'), 0.1],
        }
        all_frags.append(frags)
        new_library[key] = {}
        new_library[key]["frags"] = frags

        #two tags
        key = ("P(PSMtag_5plex-0)EPTIDEK(PSMtag_5plex-0)", 2.0)
        keys.append(key)
        frags = {
            "y1_1": [mass.fast_mass(sequence="K", ion_type='y')+tag.mass, 0.5],
            "y2_1": [mass.fast_mass(sequence="EK", ion_type='y')+tag.mass, 0.6],
            "b2_1": [mass.fast_mass(sequence="PE", ion_type='b')+tag.mass, 1],
            "b5_2": [mass.fast_mass(sequence="PEPTI", ion_type='b')+tag.mass, 0.4],
            "b5-H2O_2": [mass.fast_mass(sequence="PEPTI", ion_type='b')+tag.mass-mass.calculate_mass(formula='H2O'), 0.1],
        }
        all_frags.append(frags)
        new_library[key] = {}
        new_library[key]["frags"] = frags

        #mod
        key = ("PEPT(Unimod:21)IDEK", 2.0)
        keys.append(key)
        frags = {
            "y1_1": [mass.fast_mass(sequence="K", ion_type='y'), 0.5],
            "y2_1": [mass.fast_mass(sequence="EK", ion_type='y'), 0.6],
            "b2_1": [mass.fast_mass(sequence="PE", ion_type='b'), 1],
            "b5_2": [mass.fast_mass(sequence="PEPTI", ion_type='b')+mass.calculate_mass(formula='H3PO4'), 0.4],
            "b5-H2O_2": [mass.fast_mass(sequence="PEPTI", ion_type='b')+mass.calculate_mass(formula='H3PO4')-mass.calculate_mass(formula='H2O'), 0.1],
        }
        all_frags.append(frags)
        new_library[key] = {}
        new_library[key]["frags"] = frags

        n_iso = 3

        new_library_output = iso_library(new_library,tag,n_iso)

        new_library_expected = {
            ("PEPTIDEK", 2.0): {
            "frags": all_frags[0],
            "spectrum": np.array([[1.46105528e+02, 5.00000000e-01],
                                [1.47108335e+02, 3.72865011e-02],
                                [1.48110245e+02, 3.26301469e-03],
                                [2.26095357e+02, 1.00000000e+00],
                                [2.27098373e+02, 1.18597767e-01],
                                [2.28100371e+02, 1.46540604e-02],
                                [2.75148121e+02, 6.00000000e-01],
                                [2.76151024e+02, 8.05516867e-02],
                                [2.77153048e+02, 1.11737793e-02],
                                [5.19269299e+02, 1.00000000e-01],
                                [5.19770806e+02, 2.95581666e-02],
                                [5.20272081e+02, 5.65728474e-03],
                                [5.37279863e+02, 4.00000000e-01],
                                [5.37781372e+02, 1.18477047e-01],
                                [5.38282626e+02, 2.35234112e-02]]),
            "ordered_frags": np.array(['y1_1', 'y1_1_iso1', 'y1_1_iso2', 'b2_1', 'b2_1_iso1', 'b2_1_iso2',
                                        'y2_1', 'y2_1_iso1', 'y2_1_iso2', 'b5-H2O_2', 'b5-H2O_2_iso1',
                                        'b5-H2O_2_iso2', 'b5_2', 'b5_2_iso1', 'b5_2_iso2'], dtype='<U13')
        },

            ("P(PSMtag_5plex-0)EPTIDEK", 2.0): {
            "frags": all_frags[1],
            "spectrum": np.array([[1.46105528e+02, 5.00000000e-01],
                                    [1.47108335e+02, 3.72865011e-02],
                                    [1.48110245e+02, 3.26301469e-03],
                                    [2.75148121e+02, 6.00000000e-01],
                                    [2.76151024e+02, 8.05516867e-02],
                                    [2.77153048e+02, 1.11737793e-02],
                                    [5.34211449e+02, 1.00000000e+00],
                                    [5.35214557e+02, 3.23570461e-01],
                                    [5.36217222e+02, 6.50687578e-02],
                                    [8.27385391e+02, 1.00000000e-01],
                                    [8.27886928e+02, 5.00554360e-02],
                                    [8.28388326e+02, 1.43264411e-02],
                                    [8.45395956e+02, 4.00000000e-01],
                                    [8.45897494e+02, 2.00466125e-01],
                                    [8.46398881e+02, 5.82501283e-02]]),
            "ordered_frags": np.array(['y1_1', 'y1_1_iso1', 'y1_1_iso2', 'y2_1', 'y2_1_iso1', 'y2_1_iso2',
                                        'b2_1', 'b2_1_iso1', 'b2_1_iso2', 'b5-H2O_2', 'b5-H2O_2_iso1',
                                        'b5-H2O_2_iso2', 'b5_2', 'b5_2_iso1', 'b5_2_iso2'], dtype='<U13')
        },

            ("P(PSMtag_5plex-0)EPTIDEK(PSMtag_5plex-0)", 2.0): {
            "frags": all_frags[2],
            "spectrum": np.array([[4.54221620e+02, 5.00000000e-01],
                                    [4.55224686e+02, 1.39772848e-01],
                                    [4.56227338e+02, 2.39584261e-02],
                                    [5.34211449e+02, 1.00000000e+00],
                                    [5.35214557e+02, 3.23570461e-01],
                                    [5.36217222e+02, 6.50687578e-02],
                                    [5.83264213e+02, 6.00000000e-01],
                                    [5.84267272e+02, 2.03535303e-01],
                                    [5.85269897e+02, 4.33479117e-02],
                                    [8.27385391e+02, 1.00000000e-01],
                                    [8.27886928e+02, 5.00554360e-02],
                                    [8.28388326e+02, 1.43264411e-02],
                                    [8.45395956e+02, 4.00000000e-01],
                                    [8.45897494e+02, 2.00466125e-01],
                                    [8.46398881e+02, 5.82501283e-02]]),
            "ordered_frags": np.array(['y1_1', 'y1_1_iso1', 'y1_1_iso2', 'b2_1', 'b2_1_iso1', 'b2_1_iso2',
                                        'y2_1', 'y2_1_iso1', 'y2_1_iso2', 'b5-H2O_2', 'b5-H2O_2_iso1',
                                        'b5-H2O_2_iso2', 'b5_2', 'b5_2_iso1', 'b5_2_iso2'], dtype='<U13')
        },

            ("PEPT(Unimod:21)IDEK", 2.0): {
            "frags": all_frags[3],
            "spectrum": np.array([[1.46105528e+02, 5.00000000e-01],
                                    [1.47108335e+02, 3.72865011e-02],
                                    [1.48110245e+02, 3.26301469e-03],
                                    [2.26095357e+02, 1.00000000e+00],
                                    [2.27098373e+02, 1.18597767e-01],
                                    [2.28100371e+02, 1.46540604e-02],
                                    [2.75148121e+02, 6.00000000e-01],
                                    [2.76151024e+02, 8.05516867e-02],
                                    [2.77153048e+02, 1.11737793e-02],
                                    [6.17246194e+02, 1.00000000e-01],
                                    [6.17747704e+02, 2.96839456e-02],
                                    [6.18248917e+02, 6.31101747e-03],
                                    [6.35256758e+02, 4.00000000e-01],
                                    [6.35758270e+02, 1.18980163e-01],
                                    [6.36259466e+02, 2.61386496e-02]]),
            "ordered_frags": np.array(['y1_1', 'y1_1_iso1', 'y1_1_iso2', 'b2_1', 'b2_1_iso1', 'b2_1_iso2',
                                        'y2_1', 'y2_1_iso1', 'y2_1_iso2', 'b5-H2O_2', 'b5-H2O_2_iso1',
                                        'b5-H2O_2_iso2', 'b5_2', 'b5_2_iso1', 'b5_2_iso2'], dtype='<U13')
        },

        }

        for key in keys:
            self.compare_outputs(new_library_output[key], new_library_expected[key])


    def test_dif_num_iso(self):
        tag = read_json_to_massTag("tests/MassTags/", "PSMtag_5plex.json")
        new_library = {}

        keys = []
        all_frags = []

        #no tag
        key = ("PEPTIDEK", 2.0)
        keys.append(key)
        frags = {
            "y1_1": [mass.fast_mass(sequence="K", ion_type='y'), 0.5],
            "y2_1": [mass.fast_mass(sequence="EK", ion_type='y'), 0.6],
            "b2_1": [mass.fast_mass(sequence="PE", ion_type='b'), 1],
            "b5_2": [mass.fast_mass(sequence="PEPTI", ion_type='b'), 0.4],
            "b5-H2O_2": [mass.fast_mass(sequence="PEPTI", ion_type='b')-mass.calculate_mass(formula='H2O'), 0.1],
        }
        all_frags.append(frags)
        new_library[key] = {}
        new_library[key]["frags"] = frags

        #one tag
        key = ("P(PSMtag_5plex-0)EPTIDEK", 2.0)
        keys.append(key)
        frags = {
            "y1_1": [mass.fast_mass(sequence="K", ion_type='y'), 0.5],
            "y2_1": [mass.fast_mass(sequence="EK", ion_type='y'), 0.6],
            "b2_1": [mass.fast_mass(sequence="PE", ion_type='b')+tag.mass, 1],
            "b5_2": [mass.fast_mass(sequence="PEPTI", ion_type='b')+tag.mass, 0.4],
            "b5-H2O_2": [mass.fast_mass(sequence="PEPTI", ion_type='b')+tag.mass-mass.calculate_mass(formula='H2O'), 0.1],
        }
        all_frags.append(frags)
        new_library[key] = {}
        new_library[key]["frags"] = frags

        #two tags
        key = ("P(PSMtag_5plex-0)EPTIDEK(PSMtag_5plex-0)", 2.0)
        keys.append(key)
        frags = {
            "y1_1": [mass.fast_mass(sequence="K", ion_type='y')+tag.mass, 0.5],
            "y2_1": [mass.fast_mass(sequence="EK", ion_type='y')+tag.mass, 0.6],
            "b2_1": [mass.fast_mass(sequence="PE", ion_type='b')+tag.mass, 1],
            "b5_2": [mass.fast_mass(sequence="PEPTI", ion_type='b')+tag.mass, 0.4],
            "b5-H2O_2": [mass.fast_mass(sequence="PEPTI", ion_type='b')+tag.mass-mass.calculate_mass(formula='H2O'), 0.1],
        }
        all_frags.append(frags)
        new_library[key] = {}
        new_library[key]["frags"] = frags

        #mod
        key = ("PEPT(Unimod:21)IDEK", 2.0)
        keys.append(key)
        frags = {
            "y1_1": [mass.fast_mass(sequence="K", ion_type='y'), 0.5],
            "y2_1": [mass.fast_mass(sequence="EK", ion_type='y'), 0.6],
            "b2_1": [mass.fast_mass(sequence="PE", ion_type='b'), 1],
            "b5_2": [mass.fast_mass(sequence="PEPTI", ion_type='b')+mass.calculate_mass(formula='H3PO4'), 0.4],
            "b5-H2O_2": [mass.fast_mass(sequence="PEPTI", ion_type='b')+mass.calculate_mass(formula='H3PO4')-mass.calculate_mass(formula='H2O'), 0.1],
        }
        all_frags.append(frags)
        new_library[key] = {}
        new_library[key]["frags"] = frags

        n_iso = 2

        new_library_output = iso_library(new_library,tag,n_iso)

        new_library_expected = {
            ("PEPTIDEK", 2.0): {
            "frags": all_frags[0],
            "spectrum": np.array([[1.46105528e+02, 5.00000000e-01],
                                [1.47108335e+02, 3.72865011e-02],
                                [2.26095357e+02, 1.00000000e+00],
                                [2.27098373e+02, 1.18597767e-01],
                                [2.75148121e+02, 6.00000000e-01],
                                [2.76151024e+02, 8.05516867e-02],
                                [5.19269299e+02, 1.00000000e-01],
                                [5.19770806e+02, 2.95581666e-02],
                                [5.37279863e+02, 4.00000000e-01],
                                [5.37781372e+02, 1.18477047e-01]]),
            "ordered_frags": np.array(['y1_1', 'y1_1_iso1', 'b2_1', 'b2_1_iso1',
                                        'y2_1', 'y2_1_iso1', 'b5-H2O_2', 'b5-H2O_2_iso1',
                                        'b5_2', 'b5_2_iso1'], dtype='<U13')
        },

            ("P(PSMtag_5plex-0)EPTIDEK", 2.0): {
            "frags": all_frags[1],
            "spectrum": np.array([[1.46105528e+02, 5.00000000e-01],
                                    [1.47108335e+02, 3.72865011e-02],
                                    [2.75148121e+02, 6.00000000e-01],
                                    [2.76151024e+02, 8.05516867e-02],
                                    [5.34211449e+02, 1.00000000e+00],
                                    [5.35214557e+02, 3.23570461e-01],
                                    [8.27385391e+02, 1.00000000e-01],
                                    [8.27886928e+02, 5.00554360e-02],
                                    [8.45395956e+02, 4.00000000e-01],
                                    [8.45897494e+02, 2.00466125e-01]]),
            "ordered_frags": np.array(['y1_1', 'y1_1_iso1', 'y2_1', 'y2_1_iso1',
                                        'b2_1', 'b2_1_iso1', 'b5-H2O_2', 'b5-H2O_2_iso1',
                                        'b5_2', 'b5_2_iso1'], dtype='<U13')
        },

            ("P(PSMtag_5plex-0)EPTIDEK(PSMtag_5plex-0)", 2.0): {
            "frags": all_frags[2],
            "spectrum": np.array([[4.54221620e+02, 5.00000000e-01],
                                    [4.55224686e+02, 1.39772848e-01],
                                    [5.34211449e+02, 1.00000000e+00],
                                    [5.35214557e+02, 3.23570461e-01],
                                    [5.83264213e+02, 6.00000000e-01],
                                    [5.84267272e+02, 2.03535303e-01],
                                    [8.27385391e+02, 1.00000000e-01],
                                    [8.27886928e+02, 5.00554360e-02],
                                    [8.45395956e+02, 4.00000000e-01],
                                    [8.45897494e+02, 2.00466125e-01]]),
            "ordered_frags": np.array(['y1_1', 'y1_1_iso1', 'b2_1', 'b2_1_iso1',
                                        'y2_1', 'y2_1_iso1', 'b5-H2O_2', 'b5-H2O_2_iso1',
                                        'b5_2', 'b5_2_iso1'], dtype='<U13')
        },

            ("PEPT(Unimod:21)IDEK", 2.0): {
            "frags": all_frags[3],
            "spectrum": np.array([[1.46105528e+02, 5.00000000e-01],
                                    [1.47108335e+02, 3.72865011e-02],
                                    [2.26095357e+02, 1.00000000e+00],
                                    [2.27098373e+02, 1.18597767e-01],
                                    [2.75148121e+02, 6.00000000e-01],
                                    [2.76151024e+02, 8.05516867e-02],
                                    [6.17246194e+02, 1.00000000e-01],
                                    [6.17747704e+02, 2.96839456e-02],
                                    [6.35256758e+02, 4.00000000e-01],
                                    [6.35758270e+02, 1.18980163e-01]]),
            "ordered_frags": np.array(['y1_1', 'y1_1_iso1', 'b2_1', 'b2_1_iso1',
                                        'y2_1', 'y2_1_iso1', 'b5-H2O_2', 'b5-H2O_2_iso1',
                                        'b5_2', 'b5_2_iso1'], dtype='<U13')
        },

        }

        for key in keys:
            self.compare_outputs(new_library_output[key], new_library_expected[key])


class Test_iso_library_multi():

    def compare_outputs(self, output, expected):
        assert output['frags'] == expected['frags']
        assert np.allclose(output['spectrum'], expected['spectrum'])
        assert np.array_equal(output['ordered_frags'], expected['ordered_frags'])

    def test_it(self):
        tag = read_json_to_massTag("tests/MassTags/", "PSMtag_5plex.json")
        new_library = {}

        keys = []
        all_frags = []

        #no tag
        key = ("PEPTIDEK", 2.0)
        keys.append(key)
        frags = {
            "y1_1": [mass.fast_mass(sequence="K", ion_type='y'), 0.5],
            "y2_1": [mass.fast_mass(sequence="EK", ion_type='y'), 0.6],
            "b2_1": [mass.fast_mass(sequence="PE", ion_type='b'), 1],
            "b5_2": [mass.fast_mass(sequence="PEPTI", ion_type='b'), 0.4],
            "b5-H2O_2": [mass.fast_mass(sequence="PEPTI", ion_type='b')-mass.calculate_mass(formula='H2O'), 0.1],
        }
        all_frags.append(frags)
        new_library[key] = {}
        new_library[key]["frags"] = frags

        #one tag
        key = ("P(PSMtag_5plex-0)EPTIDEK", 2.0)
        keys.append(key)
        frags = {
            "y1_1": [mass.fast_mass(sequence="K", ion_type='y'), 0.5],
            "y2_1": [mass.fast_mass(sequence="EK", ion_type='y'), 0.6],
            "b2_1": [mass.fast_mass(sequence="PE", ion_type='b')+tag.mass, 1],
            "b5_2": [mass.fast_mass(sequence="PEPTI", ion_type='b')+tag.mass, 0.4],
            "b5-H2O_2": [mass.fast_mass(sequence="PEPTI", ion_type='b')+tag.mass-mass.calculate_mass(formula='H2O'), 0.1],
        }
        all_frags.append(frags)
        new_library[key] = {}
        new_library[key]["frags"] = frags

        #two tags
        key = ("P(PSMtag_5plex-0)EPTIDEK(PSMtag_5plex-0)", 2.0)
        keys.append(key)
        frags = {
            "y1_1": [mass.fast_mass(sequence="K", ion_type='y')+tag.mass, 0.5],
            "y2_1": [mass.fast_mass(sequence="EK", ion_type='y')+tag.mass, 0.6],
            "b2_1": [mass.fast_mass(sequence="PE", ion_type='b')+tag.mass, 1],
            "b5_2": [mass.fast_mass(sequence="PEPTI", ion_type='b')+tag.mass, 0.4],
            "b5-H2O_2": [mass.fast_mass(sequence="PEPTI", ion_type='b')+tag.mass-mass.calculate_mass(formula='H2O'), 0.1],
        }
        all_frags.append(frags)
        new_library[key] = {}
        new_library[key]["frags"] = frags

        #mod
        key = ("PEPT(Unimod:21)IDEK", 2.0)
        keys.append(key)
        frags = {
            "y1_1": [mass.fast_mass(sequence="K", ion_type='y'), 0.5],
            "y2_1": [mass.fast_mass(sequence="EK", ion_type='y'), 0.6],
            "b2_1": [mass.fast_mass(sequence="PE", ion_type='b'), 1],
            "b5_2": [mass.fast_mass(sequence="PEPTI", ion_type='b')+mass.calculate_mass(formula='H3PO4'), 0.4],
            "b5-H2O_2": [mass.fast_mass(sequence="PEPTI", ion_type='b')+mass.calculate_mass(formula='H3PO4')-mass.calculate_mass(formula='H2O'), 0.1],
        }
        all_frags.append(frags)
        new_library[key] = {}
        new_library[key]["frags"] = frags

        n_iso = 3

        new_library_output = iso_library_multi(new_library,tag,n_iso)

        new_library_expected = {
            ("PEPTIDEK", 2.0): {
            "frags": all_frags[0],
            "spectrum": np.array([[1.46105528e+02, 5.00000000e-01],
                                [1.47108335e+02, 3.72865011e-02],
                                [1.48110245e+02, 3.26301469e-03],
                                [2.26095357e+02, 1.00000000e+00],
                                [2.27098373e+02, 1.18597767e-01],
                                [2.28100371e+02, 1.46540604e-02],
                                [2.75148121e+02, 6.00000000e-01],
                                [2.76151024e+02, 8.05516867e-02],
                                [2.77153048e+02, 1.11737793e-02],
                                [5.19269299e+02, 1.00000000e-01],
                                [5.19770806e+02, 2.95581666e-02],
                                [5.20272081e+02, 5.65728474e-03],
                                [5.37279863e+02, 4.00000000e-01],
                                [5.37781372e+02, 1.18477047e-01],
                                [5.38282626e+02, 2.35234112e-02]]),
            "ordered_frags": np.array(['y1_1', 'y1_1_iso1', 'y1_1_iso2', 'b2_1', 'b2_1_iso1', 'b2_1_iso2',
                                        'y2_1', 'y2_1_iso1', 'y2_1_iso2', 'b5-H2O_2', 'b5-H2O_2_iso1',
                                        'b5-H2O_2_iso2', 'b5_2', 'b5_2_iso1', 'b5_2_iso2'], dtype='<U13')
        },

            ("P(PSMtag_5plex-0)EPTIDEK", 2.0): {
            "frags": all_frags[1],
            "spectrum": np.array([[1.46105528e+02, 5.00000000e-01],
                                    [1.47108335e+02, 3.72865011e-02],
                                    [1.48110245e+02, 3.26301469e-03],
                                    [2.75148121e+02, 6.00000000e-01],
                                    [2.76151024e+02, 8.05516867e-02],
                                    [2.77153048e+02, 1.11737793e-02],
                                    [5.34211449e+02, 1.00000000e+00],
                                    [5.35214557e+02, 3.23570461e-01],
                                    [5.36217222e+02, 6.50687578e-02],
                                    [8.27385391e+02, 1.00000000e-01],
                                    [8.27886928e+02, 5.00554360e-02],
                                    [8.28388326e+02, 1.43264411e-02],
                                    [8.45395956e+02, 4.00000000e-01],
                                    [8.45897494e+02, 2.00466125e-01],
                                    [8.46398881e+02, 5.82501283e-02]]),
            "ordered_frags": np.array(['y1_1', 'y1_1_iso1', 'y1_1_iso2', 'y2_1', 'y2_1_iso1', 'y2_1_iso2',
                                        'b2_1', 'b2_1_iso1', 'b2_1_iso2', 'b5-H2O_2', 'b5-H2O_2_iso1',
                                        'b5-H2O_2_iso2', 'b5_2', 'b5_2_iso1', 'b5_2_iso2'], dtype='<U13')
        },

            ("P(PSMtag_5plex-0)EPTIDEK(PSMtag_5plex-0)", 2.0): {
            "frags": all_frags[2],
            "spectrum": np.array([[4.54221620e+02, 5.00000000e-01],
                                    [4.55224686e+02, 1.39772848e-01],
                                    [4.56227338e+02, 2.39584261e-02],
                                    [5.34211449e+02, 1.00000000e+00],
                                    [5.35214557e+02, 3.23570461e-01],
                                    [5.36217222e+02, 6.50687578e-02],
                                    [5.83264213e+02, 6.00000000e-01],
                                    [5.84267272e+02, 2.03535303e-01],
                                    [5.85269897e+02, 4.33479117e-02],
                                    [8.27385391e+02, 1.00000000e-01],
                                    [8.27886928e+02, 5.00554360e-02],
                                    [8.28388326e+02, 1.43264411e-02],
                                    [8.45395956e+02, 4.00000000e-01],
                                    [8.45897494e+02, 2.00466125e-01],
                                    [8.46398881e+02, 5.82501283e-02]]),
            "ordered_frags": np.array(['y1_1', 'y1_1_iso1', 'y1_1_iso2', 'b2_1', 'b2_1_iso1', 'b2_1_iso2',
                                        'y2_1', 'y2_1_iso1', 'y2_1_iso2', 'b5-H2O_2', 'b5-H2O_2_iso1',
                                        'b5-H2O_2_iso2', 'b5_2', 'b5_2_iso1', 'b5_2_iso2'], dtype='<U13')
        },

            ("PEPT(Unimod:21)IDEK", 2.0): {
            "frags": all_frags[3],
            "spectrum": np.array([[1.46105528e+02, 5.00000000e-01],
                                    [1.47108335e+02, 3.72865011e-02],
                                    [1.48110245e+02, 3.26301469e-03],
                                    [2.26095357e+02, 1.00000000e+00],
                                    [2.27098373e+02, 1.18597767e-01],
                                    [2.28100371e+02, 1.46540604e-02],
                                    [2.75148121e+02, 6.00000000e-01],
                                    [2.76151024e+02, 8.05516867e-02],
                                    [2.77153048e+02, 1.11737793e-02],
                                    [6.17246194e+02, 1.00000000e-01],
                                    [6.17747704e+02, 2.96839456e-02],
                                    [6.18248917e+02, 6.31101747e-03],
                                    [6.35256758e+02, 4.00000000e-01],
                                    [6.35758270e+02, 1.18980163e-01],
                                    [6.36259466e+02, 2.61386496e-02]]),
            "ordered_frags": np.array(['y1_1', 'y1_1_iso1', 'y1_1_iso2', 'b2_1', 'b2_1_iso1', 'b2_1_iso2',
                                        'y2_1', 'y2_1_iso1', 'y2_1_iso2', 'b5-H2O_2', 'b5-H2O_2_iso1',
                                        'b5-H2O_2_iso2', 'b5_2', 'b5_2_iso1', 'b5_2_iso2'], dtype='<U13')
        },

        }

        for key in keys:
            self.compare_outputs(new_library_output[key], new_library_expected[key])


    def test_dif_num_iso(self):
        tag = read_json_to_massTag("tests/MassTags/", "PSMtag_5plex.json")
        new_library = {}

        keys = []
        all_frags = []

        #no tag
        key = ("PEPTIDEK", 2.0)
        keys.append(key)
        frags = {
            "y1_1": [mass.fast_mass(sequence="K", ion_type='y'), 0.5],
            "y2_1": [mass.fast_mass(sequence="EK", ion_type='y'), 0.6],
            "b2_1": [mass.fast_mass(sequence="PE", ion_type='b'), 1],
            "b5_2": [mass.fast_mass(sequence="PEPTI", ion_type='b'), 0.4],
            "b5-H2O_2": [mass.fast_mass(sequence="PEPTI", ion_type='b')-mass.calculate_mass(formula='H2O'), 0.1],
        }
        all_frags.append(frags)
        new_library[key] = {}
        new_library[key]["frags"] = frags

        #one tag
        key = ("P(PSMtag_5plex-0)EPTIDEK", 2.0)
        keys.append(key)
        frags = {
            "y1_1": [mass.fast_mass(sequence="K", ion_type='y'), 0.5],
            "y2_1": [mass.fast_mass(sequence="EK", ion_type='y'), 0.6],
            "b2_1": [mass.fast_mass(sequence="PE", ion_type='b')+tag.mass, 1],
            "b5_2": [mass.fast_mass(sequence="PEPTI", ion_type='b')+tag.mass, 0.4],
            "b5-H2O_2": [mass.fast_mass(sequence="PEPTI", ion_type='b')+tag.mass-mass.calculate_mass(formula='H2O'), 0.1],
        }
        all_frags.append(frags)
        new_library[key] = {}
        new_library[key]["frags"] = frags

        #two tags
        key = ("P(PSMtag_5plex-0)EPTIDEK(PSMtag_5plex-0)", 2.0)
        keys.append(key)
        frags = {
            "y1_1": [mass.fast_mass(sequence="K", ion_type='y')+tag.mass, 0.5],
            "y2_1": [mass.fast_mass(sequence="EK", ion_type='y')+tag.mass, 0.6],
            "b2_1": [mass.fast_mass(sequence="PE", ion_type='b')+tag.mass, 1],
            "b5_2": [mass.fast_mass(sequence="PEPTI", ion_type='b')+tag.mass, 0.4],
            "b5-H2O_2": [mass.fast_mass(sequence="PEPTI", ion_type='b')+tag.mass-mass.calculate_mass(formula='H2O'), 0.1],
        }
        all_frags.append(frags)
        new_library[key] = {}
        new_library[key]["frags"] = frags

        #mod
        key = ("PEPT(Unimod:21)IDEK", 2.0)
        keys.append(key)
        frags = {
            "y1_1": [mass.fast_mass(sequence="K", ion_type='y'), 0.5],
            "y2_1": [mass.fast_mass(sequence="EK", ion_type='y'), 0.6],
            "b2_1": [mass.fast_mass(sequence="PE", ion_type='b'), 1],
            "b5_2": [mass.fast_mass(sequence="PEPTI", ion_type='b')+mass.calculate_mass(formula='H3PO4'), 0.4],
            "b5-H2O_2": [mass.fast_mass(sequence="PEPTI", ion_type='b')+mass.calculate_mass(formula='H3PO4')-mass.calculate_mass(formula='H2O'), 0.1],
        }
        all_frags.append(frags)
        new_library[key] = {}
        new_library[key]["frags"] = frags

        n_iso = 2

        new_library_output = iso_library_multi(new_library,tag,n_iso)

        new_library_expected = {
            ("PEPTIDEK", 2.0): {
            "frags": all_frags[0],
            "spectrum": np.array([[1.46105528e+02, 5.00000000e-01],
                                [1.47108335e+02, 3.72865011e-02],
                                [2.26095357e+02, 1.00000000e+00],
                                [2.27098373e+02, 1.18597767e-01],
                                [2.75148121e+02, 6.00000000e-01],
                                [2.76151024e+02, 8.05516867e-02],
                                [5.19269299e+02, 1.00000000e-01],
                                [5.19770806e+02, 2.95581666e-02],
                                [5.37279863e+02, 4.00000000e-01],
                                [5.37781372e+02, 1.18477047e-01]]),
            "ordered_frags": np.array(['y1_1', 'y1_1_iso1', 'b2_1', 'b2_1_iso1',
                                        'y2_1', 'y2_1_iso1', 'b5-H2O_2', 'b5-H2O_2_iso1',
                                        'b5_2', 'b5_2_iso1'], dtype='<U13')
        },

            ("P(PSMtag_5plex-0)EPTIDEK", 2.0): {
            "frags": all_frags[1],
            "spectrum": np.array([[1.46105528e+02, 5.00000000e-01],
                                    [1.47108335e+02, 3.72865011e-02],
                                    [2.75148121e+02, 6.00000000e-01],
                                    [2.76151024e+02, 8.05516867e-02],
                                    [5.34211449e+02, 1.00000000e+00],
                                    [5.35214557e+02, 3.23570461e-01],
                                    [8.27385391e+02, 1.00000000e-01],
                                    [8.27886928e+02, 5.00554360e-02],
                                    [8.45395956e+02, 4.00000000e-01],
                                    [8.45897494e+02, 2.00466125e-01]]),
            "ordered_frags": np.array(['y1_1', 'y1_1_iso1', 'y2_1', 'y2_1_iso1',
                                        'b2_1', 'b2_1_iso1', 'b5-H2O_2', 'b5-H2O_2_iso1',
                                        'b5_2', 'b5_2_iso1'], dtype='<U13')
        },

            ("P(PSMtag_5plex-0)EPTIDEK(PSMtag_5plex-0)", 2.0): {
            "frags": all_frags[2],
            "spectrum": np.array([[4.54221620e+02, 5.00000000e-01],
                                    [4.55224686e+02, 1.39772848e-01],
                                    [5.34211449e+02, 1.00000000e+00],
                                    [5.35214557e+02, 3.23570461e-01],
                                    [5.83264213e+02, 6.00000000e-01],
                                    [5.84267272e+02, 2.03535303e-01],
                                    [8.27385391e+02, 1.00000000e-01],
                                    [8.27886928e+02, 5.00554360e-02],
                                    [8.45395956e+02, 4.00000000e-01],
                                    [8.45897494e+02, 2.00466125e-01]]),
            "ordered_frags": np.array(['y1_1', 'y1_1_iso1', 'b2_1', 'b2_1_iso1',
                                        'y2_1', 'y2_1_iso1', 'b5-H2O_2', 'b5-H2O_2_iso1',
                                        'b5_2', 'b5_2_iso1'], dtype='<U13')
        },

            ("PEPT(Unimod:21)IDEK", 2.0): {
            "frags": all_frags[3],
            "spectrum": np.array([[1.46105528e+02, 5.00000000e-01],
                                    [1.47108335e+02, 3.72865011e-02],
                                    [2.26095357e+02, 1.00000000e+00],
                                    [2.27098373e+02, 1.18597767e-01],
                                    [2.75148121e+02, 6.00000000e-01],
                                    [2.76151024e+02, 8.05516867e-02],
                                    [6.17246194e+02, 1.00000000e-01],
                                    [6.17747704e+02, 2.96839456e-02],
                                    [6.35256758e+02, 4.00000000e-01],
                                    [6.35758270e+02, 1.18980163e-01]]),
            "ordered_frags": np.array(['y1_1', 'y1_1_iso1', 'b2_1', 'b2_1_iso1',
                                        'y2_1', 'y2_1_iso1', 'b5-H2O_2', 'b5-H2O_2_iso1',
                                        'b5_2', 'b5_2_iso1'], dtype='<U13')
        },

        }

        for key in keys:
            self.compare_outputs(new_library_output[key], new_library_expected[key])
            





def main():
    
   pass



if __name__ == "__main__":
    main()


