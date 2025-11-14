import pytest
import sys
import os
from src.mass_tags import massTag, read_json_to_massTag, refresh_tags, set_config_tag, tag_library
import json
import numpy as np
from pathlib import Path
from pyteomics import mass



class Test_read_json_to_massTag():

    def test_basic(self):
        
        output_tag = read_json_to_massTag("tests/MassTags", "test_basic.json")
        assert output_tag.name == "test_basic"
        assert output_tag.rules == "nK"
        assert output_tag.mass == 78.03437413308
        assert output_tag.delta == [0.0, 2.006709675600007]
        assert output_tag.channel_names == ["0", "2"]
        assert output_tag.channel_comp == {
                "0": {
                    "C": 5,
                    "H": 4,
                    "N": 1,
                    "O": 0,
                    "C[13]": 0,
                    "H[2]": 0,
                    "N[15]": 0,
                    "O[18]": 0
                },
                "2": {
                    "C": 3,
                    "H": 4,
                    "N": 1,
                    "O": 0,
                    "C[13]": 2,
                    "H[2]": 0,
                    "N[15]": 0,
                    "O[18]": 0
                }
            }
        
    def test_decode_error(self):
        output_tag = read_json_to_massTag("tests/MassTags", "test_decode_error.json")
        assert os.path.normpath(str(output_tag)) == os.path.normpath(r"tests/MassTags\test_decode_error.json")

    def test_no_comps(self):
        output_tag = read_json_to_massTag("tests/MassTags", "test_no_comps.json")
        assert output_tag == "test_no_comps"

    def test_bad_base_mass(self):
        output_tag = read_json_to_massTag("tests/MassTags", "test_bad_base_mass.json")
        assert output_tag == "test_bad_base_mass"

    def test_bad_channel_mass(self):
        output_tag = read_json_to_massTag("tests/MassTags", "test_bad_channel_mass.json")
        assert output_tag == "test_bad_channel_mass"

    def test_mismatched_names(self):
        output_tag = read_json_to_massTag("tests/MassTags", "test_name_mismatch.json")
        assert output_tag == "test_mismatch_name"
    
    def test_one_channel(self):
        output_tag = read_json_to_massTag("tests/MassTags", "test_one_channel.json")
        assert output_tag.delta == [0.0]

    def test_subset(self):
        output_tag = read_json_to_massTag("tests/MassTags/Subsets", "test_subset.json")
        assert output_tag.name == "test_subset"
        assert output_tag.rules == "nK"
        assert output_tag.mass == 78.03437413308
        assert output_tag.delta == [0.0, 2.006709675600007]
        assert output_tag.channel_names == ["0", "2"]

class Test_refresh_tags():

    def test_refresh_tags(self):
        available_tags = refresh_tags(Path("tests/MassTags"))
        assert list(available_tags.keys()) == ["test_basic", "test_one_channel", "test_subset"]
        assert all([type(x) == massTag for x in available_tags.values()])

class Test_set_config_tag():

    available_tags = refresh_tags(Path("tests/MassTags"))

    def test_tag_available(self):
        tag = set_config_tag(self.available_tags, "test_basic")
        assert tag.name == "test_basic"

    def test_tag_is_none(self):
        tag = set_config_tag(self.available_tags, "None")
        assert tag is None

    def test_unavailable_tag(self):
        with pytest.raises(Exception) as exc_info:
            tag = set_config_tag(self.available_tags, "test_no_comps")

class Test_mass_tag_class():

    mass_tag = massTag(rules="nK",
                        base_mass=78.03437413308,
                        delta=[0.0, 2.006709675600007],
                        channel_names=["0","2"],
                        name="test_basic",
                        compositions={
                                    "0": {
                                        "C": 5,
                                        "H": 4,
                                        "N": 1,
                                        "O": 0,
                                        "C[13]": 0,
                                        "H[2]": 0,
                                        "N[15]": 0,
                                        "O[18]": 0
                                    },
                                    "2": {
                                        "C": 3,
                                        "H": 4,
                                        "N": 1,
                                        "O": 0,
                                        "C[13]": 2,
                                        "H[2]": 0,
                                        "N[15]": 0,
                                        "O[18]": 0
                                    }
                                })

    def test_attributes(self):
        assert self.mass_tag.rules == "nK"
        assert self.mass_tag.mass == 78.03437413308
        assert self.mass_tag.delta == [0.0, 2.006709675600007]
        assert self.mass_tag.n_channels == 2
        assert self.mass_tag.channel_names == ["0", "2"]
        assert np.allclose(self.mass_tag.channel_masses, np.array([78.03437413, 80.04108381]))
        assert self.mass_tag.name == "test_basic"
        assert self.mass_tag.mass_dict == {'test_basic-0': 78.03437413308, 'test_basic-2': 80.04108380868}
        assert self.mass_tag.channel_comp == {'0': {'C': 5, 'H': 4, 'N': 1, 'O': 0, 'C[13]': 0, 'H[2]': 0, 'N[15]': 0, 'O[18]': 0}, '2': {'C': 3, 'H': 4, 'N': 1, 'O': 0, 'C[13]': 2, 'H[2]': 0, 'N[15]': 0, 'O[18]': 0}}

    def test_repr(self):
        expected = (
            "Mass Tag\n"
            "TagName: test_basic\n"
            "Base Mass: 78.03437413308\n"
            "MassDelta(s): [0.0, 2.006709675600007]\n"
            "ChannelNames: ['0', '2']\n"
            "ChannelMasses: [78.03437413 80.04108381]"
        )
        actual = repr(self.mass_tag)
        assert actual == expected

    def test_key_access(self):
        assert self.mass_tag["channel_names"] == ["0", "2"]



class Test_tag_library():

    def test_single_channel_two_tags(self): #also tests ordering of ordered frag dict
        tag = read_json_to_massTag("tests/MassTags", "test_one_channel.json")

        initial_dict = {
            ("PEPTIDEK", 2.0):
            {
                'mod_seq': "PEPTIDEK",
                'seq': "PEPTIDEK",
                'prec_mz': 464.727463520355,
                'prec_z': 2,
                'iRT': 0.57,
                'frags': {
                    'b3_1': [mass.fast_mass(sequence="PEP", ion_type='b'), 1],
                    'b4_1': [mass.fast_mass(sequence="PEPT", ion_type='b'), 0.85],
                    'y3_1': [mass.fast_mass(sequence="DEK", ion_type='y'), 0.5]
                },
                'protein_group': 'P36578',
                'protein_name': '',
                'genes': '',
                'spectrum': np.array([
                    [mass.fast_mass(sequence="PEP", ion_type='b'), 1],
                    [mass.fast_mass(sequence="PEPT", ion_type='b'), 0.85],
                    [mass.fast_mass(sequence="DEK", ion_type='y'), 0.5]
                ]),
                'ordered_frags': np.array([
                    'b3_1', 'b4_1', "y3_1"
                ], dtype='<U4')
            }
            
         }
        
        final_dict = tag_library(initial_dict, tag)
        key = final_dict[("P(test_one_channel-0)EPTIDEK(test_one_channel-0)", 2.0)]
        assert key['mod_seq'] == "P(test_one_channel-0)EPTIDEK(test_one_channel-0)"
        assert key['seq'] == "PEPTIDEK"
        assert key['prec_mz'] == 464.727463520355 + (2*tag.mass)/2
        assert key['prec_z'] == 2
        assert key['iRT'] == 0.57
        assert key['frags'] == {
                    'b3_1': [mass.fast_mass(sequence="PEP", ion_type='b')+tag.mass, 1],
                    'b4_1': [mass.fast_mass(sequence="PEPT", ion_type='b')+tag.mass, 0.85],
                    'y3_1': [mass.fast_mass(sequence="DEK", ion_type='y')+tag.mass, 0.5]
                }
        assert key['protein_group'] == 'P36578'
        assert key['protein_name'] == ''
        assert key['genes'] == ''
        assert np.allclose(key['spectrum'], np.array([
                    [mass.fast_mass(sequence="PEP", ion_type='b')+tag.mass, 1],
                    [mass.fast_mass(sequence="DEK", ion_type='y')+tag.mass, 0.5],
                    [mass.fast_mass(sequence="PEPT", ion_type='b')+tag.mass, 0.85]
                ]), atol=1e-6)
        assert np.array_equal(key['ordered_frags'], np.array(['b3_1', 'y3_1', 'b4_1'], dtype='<U4'))
        assert 'spec_frags' not in key

    def test_single_channel_one_tags(self): #test one channel, and z=3, and int in key
        tag = read_json_to_massTag("tests/MassTags", "test_one_channel.json")

        initial_dict = {
            ("PEPTIDER", 3):
            {
                'mod_seq': "PEPTIDER",
                'seq': "PEPTIDER",
                'prec_mz': 319.48702501677,
                'prec_z': 3,
                'iRT': 0.57,
                'frags': {
                    'b3_1': [mass.fast_mass(sequence="PEP", ion_type='b'), 1],
                    'b4_1': [mass.fast_mass(sequence="PEPT", ion_type='b'), 0.85],
                    'y3_1': [mass.fast_mass(sequence="DER", ion_type='y'), 0.5]
                },
                'protein_group': 'P36578',
                'protein_name': '',
                'genes': '',
                'spectrum': np.array([
                    [mass.fast_mass(sequence="PEP", ion_type='b'), 1],
                    [mass.fast_mass(sequence="PEPT", ion_type='b'), 0.85],
                    [mass.fast_mass(sequence="DER", ion_type='y'), 0.5]
                ]),
                'ordered_frags': np.array([
                    'b3_1', 'b4_1', "y3_1"
                ], dtype='<U4')
            }
            
         }
        
        final_dict = tag_library(initial_dict, tag)
        key = final_dict[("P(test_one_channel-0)EPTIDER", 3.0)]
        assert key['mod_seq'] == "P(test_one_channel-0)EPTIDER"
        assert key['seq'] == "PEPTIDER"
        assert key['prec_mz'] == 319.48702501677 + (1*tag.mass)/3
        assert key['prec_z'] == 3
        assert key['iRT'] == 0.57
        assert key['frags'] == {
                    'b3_1': [mass.fast_mass(sequence="PEP", ion_type='b')+tag.mass, 1],
                    'b4_1': [mass.fast_mass(sequence="PEPT", ion_type='b')+tag.mass, 0.85],
                    'y3_1': [mass.fast_mass(sequence="DER", ion_type='y'), 0.5]
                }
        assert key['protein_group'] == 'P36578'
        assert key['protein_name'] == ''
        assert key['genes'] == ''
        assert np.allclose(key['spectrum'], np.array([
                    [mass.fast_mass(sequence="PEP", ion_type='b')+tag.mass, 1],
                    [mass.fast_mass(sequence="DER", ion_type='y'), 0.5],
                    [mass.fast_mass(sequence="PEPT", ion_type='b')+tag.mass, 0.85]
                ]), atol=1e-6)
        assert np.array_equal(key['ordered_frags'], np.array(['b3_1', 'y3_1', 'b4_1'], dtype='<U4'))


    def test_mod(self):
        tag = read_json_to_massTag("tests/MassTags", "test_one_channel.json")
        phospho_mass = 79.966331
        initial_dict = {
            ("PEPT(Unimod:21)IDER", 3):
            {
                'mod_seq': "PEPT(Unimod:21)IDER",
                'seq': "PEPTIDER",
                'prec_mz': 319.48702501677 + phospho_mass,
                'prec_z': 3,
                'iRT': 0.57,
                'frags': {
                    'b3_1': [mass.fast_mass(sequence="PEP", ion_type='b'), 1],
                    'b4_1': [mass.fast_mass(sequence="PEPT", ion_type='b') + phospho_mass, 0.85],
                    'y3_1': [mass.fast_mass(sequence="DER", ion_type='y'), 0.5]
                },
                'protein_group': 'P36578',
                'protein_name': '',
                'genes': '',
                'spectrum': np.array([
                    [mass.fast_mass(sequence="PEP", ion_type='b'), 1],
                    [mass.fast_mass(sequence="PEPT", ion_type='b') + phospho_mass, 0.85],
                    [mass.fast_mass(sequence="DER", ion_type='y'), 0.5]
                ]),
                'ordered_frags': np.array([
                    'b3_1', 'b4_1', "y3_1"
                ], dtype='<U4')
            }
            
         }
        
        final_dict = tag_library(initial_dict, tag)
        key = final_dict[("P(test_one_channel-0)EPT(Unimod:21)IDER", 3.0)]
        assert key['mod_seq'] == "P(test_one_channel-0)EPT(Unimod:21)IDER"
        assert key['seq'] == "PEPTIDER"
        assert key['prec_mz'] == 319.48702501677 + phospho_mass + (1*tag.mass)/3
        assert key['prec_z'] == 3
        assert key['iRT'] == 0.57
        assert key['frags'] == {
                    'b3_1': [mass.fast_mass(sequence="PEP", ion_type='b')+tag.mass, 1],
                    'b4_1': [mass.fast_mass(sequence="PEPT", ion_type='b')+phospho_mass+tag.mass, 0.85],
                    'y3_1': [mass.fast_mass(sequence="DER", ion_type='y'), 0.5]
                }
        assert key['protein_group'] == 'P36578'
        assert key['protein_name'] == ''
        assert key['genes'] == ''
        assert np.allclose(key['spectrum'], np.array([
                    [mass.fast_mass(sequence="PEP", ion_type='b')+tag.mass, 1],
                    [mass.fast_mass(sequence="DER", ion_type='y'), 0.5],
                    [mass.fast_mass(sequence="PEPT", ion_type='b')+phospho_mass+tag.mass, 0.85]
                ]), atol=1e-6)
        assert np.array_equal(key['ordered_frags'], np.array(['b3_1', 'y3_1', 'b4_1'], dtype='<U4'))

    def test_two_channels(self):
        tag = read_json_to_massTag("tests/MassTags", "test_basic.json")

        initial_dict = {
            ("PEPTIDEK", 2.0):
            {
                'mod_seq': "PEPTIDEK",
                'seq': "PEPTIDEK",
                'prec_mz': 464.727463520355,
                'prec_z': 2,
                'iRT': 0.57,
                'frags': {
                    'b3_1': [mass.fast_mass(sequence="PEP", ion_type='b'), 1],
                    'b4_1': [mass.fast_mass(sequence="PEPT", ion_type='b'), 0.85],
                    'y3_1': [mass.fast_mass(sequence="DEK", ion_type='y'), 0.5]
                },
                'protein_group': 'P36578',
                'protein_name': '',
                'genes': '',
                'spectrum': np.array([
                    [mass.fast_mass(sequence="PEP", ion_type='b'), 1],
                    [mass.fast_mass(sequence="PEPT", ion_type='b'), 0.85],
                    [mass.fast_mass(sequence="DEK", ion_type='y'), 0.5]
                ]),
                'ordered_frags': np.array([
                    'b3_1', 'b4_1', "y3_1"
                ], dtype='<U4')
            }
            
         }
        
        final_dict = tag_library(initial_dict, tag)
        assert len(final_dict) == tag.n_channels*len(initial_dict)
        for i, channel in enumerate(tag.channel_names):
            key = final_dict[(f"P(test_basic-{channel})EPTIDEK(test_basic-{channel})", 2.0)]

            assert key['mod_seq'] == f"P(test_basic-{channel})EPTIDEK(test_basic-{channel})"

            assert key['seq'] == "PEPTIDEK"

            assert key['prec_mz'] == 464.727463520355 + (2*(tag.mass+tag.delta[i]))/2

            assert key['prec_z'] == 2

            assert key['iRT'] == 0.57

            assert key['frags'] == {
                        'b3_1': [mass.fast_mass(sequence="PEP", ion_type='b')+(tag.mass+tag.delta[i]), 1],
                        'b4_1': [mass.fast_mass(sequence="PEPT", ion_type='b')+(tag.mass+tag.delta[i]), 0.85],
                        'y3_1': [mass.fast_mass(sequence="DEK", ion_type='y')+(tag.mass+tag.delta[i]), 0.5]
                    }
            
            assert key['protein_group'] == 'P36578'

            assert key['protein_name'] == ''

            assert key['genes'] == ''

            assert np.allclose(key['spectrum'], np.array([
                        [mass.fast_mass(sequence="PEP", ion_type='b')+(tag.mass+tag.delta[i]), 1],
                        [mass.fast_mass(sequence="DEK", ion_type='y')+(tag.mass+tag.delta[i]), 0.5],
                        [mass.fast_mass(sequence="PEPT", ion_type='b')+(tag.mass+tag.delta[i]), 0.85]
                    ]), atol=1e-6)
            
            assert np.array_equal(key['ordered_frags'], np.array(['b3_1', 'y3_1', 'b4_1'], dtype='<U4'))

    def test_bad_fragment(self): 
        tag = read_json_to_massTag("tests/MassTags", "test_one_channel.json")

        initial_dict = {
            ("PEPTIDEK", 2.0):
            {
                'mod_seq': "PEPTIDEK",
                'seq': "PEPTIDEK",
                'prec_mz': 464.727463520355,
                'prec_z': 2,
                'iRT': 0.57,
                'frags': {
                    'k3_1': [mass.fast_mass(sequence="PEP", ion_type='b'), 1],
                    'b4_1': [mass.fast_mass(sequence="PEPT", ion_type='b'), 0.85],
                    'y3_1': [mass.fast_mass(sequence="DEK", ion_type='y'), 0.5]
                },
                'protein_group': 'P36578',
                'protein_name': '',
                'genes': '',
                'spectrum': np.array([
                    [mass.fast_mass(sequence="PEP", ion_type='b'), 1],
                    [mass.fast_mass(sequence="PEPT", ion_type='b'), 0.85],
                    [mass.fast_mass(sequence="DEK", ion_type='y'), 0.5]
                ]),
                'ordered_frags': np.array([
                    'b3_1', 'b4_1', "y3_1"
                ], dtype='<U4')
            }
            
         }
        
        with pytest.raises(ValueError) as excinfo:
            final_dict = tag_library(initial_dict, tag)

        assert "Invalid ion type" in str(excinfo.value)

    def test_spec_frags(self): #also tests ordering of ordered frag dict
        tag = read_json_to_massTag("tests/MassTags", "test_one_channel.json")

        initial_dict = {
            ("PEPTIDEK", 2.0):
            {
                'mod_seq': "PEPTIDEK",
                'seq': "PEPTIDEK",
                'prec_mz': 464.727463520355,
                'prec_z': 2,
                'iRT': 0.57,
                'frags': {
                    'b3_1': [mass.fast_mass(sequence="PEP", ion_type='b'), 1],
                    'b4_1': [mass.fast_mass(sequence="PEPT", ion_type='b'), 0.85],
                    'y3_1': [mass.fast_mass(sequence="DEK", ion_type='y'), 0.5]
                },
                'protein_group': 'P36578',
                'protein_name': '',
                'genes': '',
                'spectrum': np.array([
                    [mass.fast_mass(sequence="PEP", ion_type='b'), 1],
                    [mass.fast_mass(sequence="PEPT", ion_type='b'), 0.85],
                    [mass.fast_mass(sequence="DEK", ion_type='y'), 0.5]
                ]),
                'ordered_frags': np.array([
                    'b3_1', 'b4_1', "y3_1"
                ], dtype='<U4'),
                'spec_frags': []
            }
            
         }
        
        final_dict = tag_library(initial_dict, tag)
        key = final_dict[("P(test_one_channel-0)EPTIDEK(test_one_channel-0)", 2.0)]
        assert key['mod_seq'] == "P(test_one_channel-0)EPTIDEK(test_one_channel-0)"
        assert key['seq'] == "PEPTIDEK"
        assert key['prec_mz'] == 464.727463520355 + (2*tag.mass)/2
        assert key['prec_z'] == 2
        assert key['iRT'] == 0.57
        assert key['frags'] == {
                    'b3_1': [mass.fast_mass(sequence="PEP", ion_type='b')+tag.mass, 1],
                    'b4_1': [mass.fast_mass(sequence="PEPT", ion_type='b')+tag.mass, 0.85],
                    'y3_1': [mass.fast_mass(sequence="DEK", ion_type='y')+tag.mass, 0.5]
                }
        assert key['protein_group'] == 'P36578'
        assert key['protein_name'] == ''
        assert key['genes'] == ''
        assert np.allclose(key['spectrum'], np.array([
                    [mass.fast_mass(sequence="PEP", ion_type='b')+tag.mass, 1],
                    [mass.fast_mass(sequence="DEK", ion_type='y')+tag.mass, 0.5],
                    [mass.fast_mass(sequence="PEPT", ion_type='b')+tag.mass, 0.85]
                ]), atol=1e-6)
        assert np.array_equal(key['ordered_frags'], np.array(['b3_1', 'y3_1', 'b4_1'], dtype='<U4'))
        assert np.allclose(key['spec_frags'], np.array([
                    [mass.fast_mass(sequence="PEP", ion_type='b')+tag.mass, 1],
                    [mass.fast_mass(sequence="PEPT", ion_type='b')+tag.mass, 0.85]
                ]), atol=1e-6)

