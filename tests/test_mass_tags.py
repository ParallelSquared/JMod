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

import pytest
import sys
import os
from src.mass_tags import massTag, read_json_to_massTag, refresh_tags, set_config_tag, tag_library, resolve_decoy_channel
from src.utils.errors import JModError
import json
import numpy as np
from pathlib import Path
from pyteomics import mass
from src.models.spec_lib.library_store import SpectrumLibraryStore

def assert_frags_approx(actual, expected, rtol=3e-7):
    """Fragment dicts match keywise; m/z & intensity within f32 quantization."""
    assert set(actual) == set(expected), (set(actual) ^ set(expected))
    for k, (mz, inten) in expected.items():
        amz, aint = actual[k]
        assert amz == pytest.approx(mz, rel=rtol), (k, amz, mz)
        assert aint == pytest.approx(inten, rel=rtol), (k, aint, inten)




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
        assert os.path.normpath(str(output_tag)) == os.path.normpath(r"tests/MassTags/test_decode_error.json")

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

class Test_refresh_tags():

    def test_refresh_tags(self):
        available_tags = refresh_tags(Path("tests/MassTags"))
        assert set(available_tags.keys()) == set(["test_basic", "test_one_channel", "PSMtag_5plex"])
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

    def test_channels_select_some_of_the_tag(self):
        tag = set_config_tag(self.available_tags, "PSMtag_5plex", "0,8")
        assert tag.name == "PSMtag_5plex" and tag.channel_names == ["0", "8"]
        assert list(tag.channel_comp) == ["0", "8"]
        assert np.allclose(tag.channel_masses, self.available_tags["PSMtag_5plex"].channel_masses[[0, 2]])
        assert tag.mass_dict == self.available_tags["PSMtag_5plex"].mass_dict  # every channel, searched or not

    def test_a_channel_the_tag_does_not_have_raises(self):
        with pytest.raises(JModError, match="not channels of PSMtag_5plex"):
            set_config_tag(self.available_tags, "PSMtag_5plex", ["0", "5"])

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
        initial_store = SpectrumLibraryStore.from_dict(initial_dict)
        final_dict = tag_library(initial_store, tag)
        key = final_dict[("(test_one_channel-0)PEPTIDEK(test_one_channel-0)", 2.0)]
        assert key['mod_seq'] == "(test_one_channel-0)PEPTIDEK(test_one_channel-0)"
        assert key['seq'] == "PEPTIDEK"
        assert key['prec_mz'] == np.float32(464.727463520355 + (2*tag.mass)/2)
        assert key['prec_z'] == 2
        assert key['iRT'] == 0.57
        assert_frags_approx(key['frags'], {
                    'b3_1': [mass.fast_mass(sequence="PEP", ion_type='b')+tag.mass, 1],
                    'b4_1': [mass.fast_mass(sequence="PEPT", ion_type='b')+tag.mass, 0.85],
                    'y3_1': [mass.fast_mass(sequence="DEK", ion_type='y')+tag.mass, 0.5]
                })
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
        initial_store = SpectrumLibraryStore.from_dict(initial_dict)
        final_dict = tag_library(initial_store, tag)
        key = final_dict[("(test_one_channel-0)PEPTIDER", 3.0)]
        assert key['mod_seq'] == "(test_one_channel-0)PEPTIDER"
        assert key['seq'] == "PEPTIDER"
        assert key['prec_mz'] == np.float32(319.48702501677 + (1*tag.mass)/3)
        assert key['prec_z'] == 3
        assert key['iRT'] == 0.57
        assert_frags_approx(key['frags'], {
                    'b3_1': [mass.fast_mass(sequence="PEP", ion_type='b')+tag.mass, 1],
                    'b4_1': [mass.fast_mass(sequence="PEPT", ion_type='b')+tag.mass, 0.85],
                    'y3_1': [mass.fast_mass(sequence="DER", ion_type='y'), 0.5]
                })
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

        initial_store = SpectrumLibraryStore.from_dict(initial_dict)
        final_dict = tag_library(initial_store, tag)
        key = final_dict[("(test_one_channel-0)PEPT(Unimod:21)IDER", 3.0)]
        assert key['mod_seq'] == "(test_one_channel-0)PEPT(Unimod:21)IDER"
        assert key['seq'] == "PEPTIDER"
        assert key['prec_mz'] == np.float32(319.48702501677 + phospho_mass + (1*tag.mass)/3)
        assert key['prec_z'] == 3
        assert key['iRT'] == 0.57
        assert_frags_approx(key['frags'], {
                    'b3_1': [mass.fast_mass(sequence="PEP", ion_type='b')+tag.mass, 1],
                    'b4_1': [mass.fast_mass(sequence="PEPT", ion_type='b')+phospho_mass+tag.mass, 0.85],
                    'y3_1': [mass.fast_mass(sequence="DER", ion_type='y'), 0.5]
                })
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
        
        initial_store = SpectrumLibraryStore.from_dict(initial_dict)
        final_dict = tag_library(initial_store, tag)
        assert len(final_dict) == tag.n_channels*len(initial_dict)
        for i, channel in enumerate(tag.channel_names):
            key = final_dict[(f"(test_basic-{channel})PEPTIDEK(test_basic-{channel})", 2.0)]

            assert key['mod_seq'] == f"(test_basic-{channel})PEPTIDEK(test_basic-{channel})"

            assert key['seq'] == "PEPTIDEK"

            assert key['prec_mz'] == np.float32(464.727463520355 + (2*(tag.mass+tag.delta[i]))/2)

            assert key['prec_z'] == 2

            assert key['iRT'] == 0.57

            assert_frags_approx(key['frags'], {
                        'b3_1': [mass.fast_mass(sequence="PEP", ion_type='b')+(tag.mass+tag.delta[i]), 1],
                        'b4_1': [mass.fast_mass(sequence="PEPT", ion_type='b')+(tag.mass+tag.delta[i]), 0.85],
                        'y3_1': [mass.fast_mass(sequence="DEK", ion_type='y')+(tag.mass+tag.delta[i]), 0.5]
                    })
            
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
            initial_store = SpectrumLibraryStore.from_dict(initial_dict)
            final_dict = tag_library(initial_store, tag)

        assert "Cannot parse fragment name" in str(excinfo.value)

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
        
        initial_store = SpectrumLibraryStore.from_dict(initial_dict)
        final_dict = tag_library(initial_store, tag)
        key = final_dict[("(test_one_channel-0)PEPTIDEK(test_one_channel-0)", 2.0)]
        assert key['mod_seq'] == "(test_one_channel-0)PEPTIDEK(test_one_channel-0)"
        assert key['seq'] == "PEPTIDEK"
        assert key['prec_mz'] == np.float32(464.727463520355 + (2*tag.mass)/2)
        assert key['prec_z'] == 2
        assert key['iRT'] == 0.57
        assert_frags_approx(key['frags'], {
                    'b3_1': [mass.fast_mass(sequence="PEP", ion_type='b')+tag.mass, 1],
                    'b4_1': [mass.fast_mass(sequence="PEPT", ion_type='b')+tag.mass, 0.85],
                    'y3_1': [mass.fast_mass(sequence="DEK", ion_type='y')+tag.mass, 0.5]
                })
        assert key['protein_group'] == 'P36578'
        assert key['protein_name'] == ''
        assert key['genes'] == ''
        assert np.allclose(key['spectrum'], np.array([
                    [mass.fast_mass(sequence="PEP", ion_type='b')+tag.mass, 1],
                    [mass.fast_mass(sequence="DEK", ion_type='y')+tag.mass, 0.5],
                    [mass.fast_mass(sequence="PEPT", ion_type='b')+tag.mass, 0.85]
                ]), atol=1e-6)
        assert np.array_equal(key['ordered_frags'], np.array(['b3_1', 'y3_1', 'b4_1'], dtype='<U4'))
        # assert np.allclose(key['spec_frags'], np.array([
        #             [mass.fast_mass(sequence="PEP", ion_type='b')+tag.mass, 1],
        #             [mass.fast_mass(sequence="PEPT", ion_type='b')+tag.mass, 0.85]
        #         ]), atol=1e-6)
        assert key['spec_frags'] is None


class Test_tag_rules():

    @pytest.mark.parametrize("rules", ["nK", "R", "K", "STY", "nSTY", "Kn", "n"])
    def test_valid_rules(self, rules):
        from src.mass_tags import tag_rules_error
        assert tag_rules_error(rules) is None

    @pytest.mark.parametrize("rules", ["", None, "nk", "c", "n K", "KX", "nKK"])
    def test_invalid_rules(self, rules):
        from src.mass_tags import tag_rules_error
        assert tag_rules_error(rules) is not None

    def test_tag_json_with_bad_rules_does_not_load(self, tmp_path):
        tag_json = json.loads(Path("tests/MassTags/test_one_channel.json").read_text())
        tag_json["rules"] = "nKK"
        (tmp_path / "test_one_channel.json").write_text(json.dumps(tag_json))
        assert read_json_to_massTag(tmp_path, "test_one_channel.json") == "test_one_channel"
        tag_json["rules"] = "nSTY"
        (tmp_path / "test_one_channel.json").write_text(json.dumps(tag_json))
        assert read_json_to_massTag(tmp_path, "test_one_channel.json").rules == "nSTY"

    def test_tag_json_with_letter_channels_does_not_load(self, tmp_path):
        tag_json = json.loads(Path("tests/MassTags/test_one_channel.json").read_text())
        tag_json["channel_names"] = ["a"]
        tag_json["compositions"] = {"a": tag_json["compositions"]["0"]}
        (tmp_path / "test_one_channel.json").write_text(json.dumps(tag_json))
        assert read_json_to_massTag(tmp_path, "test_one_channel.json") == "test_one_channel"


class Test_decoy_channel():

    available_tags = refresh_tags(Path("tests/MassTags"))

    def test_the_step_is_extrapolated_even_when_the_tag_defines_that_channel(self):
        """0, 4, 8, 12 of PSMtag_5plex steps to 16, which the tag also defines --
        but the decoy channel is always the extrapolation, a different isotope
        route to the same nominal mass, 6.3 ppm from the reagent's."""
        full = self.available_tags["PSMtag_5plex"]
        tag = set_config_tag(self.available_tags, "PSMtag_5plex", "0,4,8,12").with_decoy_channel()
        assert tag.decoy_channel == "16"
        assert tag.channel_names == ["0", "4", "8", "12", "16"]
        assert tag.channel_comp["16"] == full.channel_comp["12"] + (full.channel_comp["4"]
                                                                    - full.channel_comp["0"])
        assert tag.channel_comp["16"] != full.channel_comp["16"]
        assert tag.channel_masses[-1] - full.channel_masses[-1] == pytest.approx(0.00632, abs=1e-5)
        # mass_dict's entry for that name is the extrapolated channel's from here on
        assert tag.mass_dict["PSMtag_5plex-16"] == pytest.approx(tag.channel_masses[-1])

    def test_auto_past_the_last_channel_carries_the_isotopes_of_the_step(self):
        """All 5 channels searched, so 20 has to be made: the 0 -> 4 step applied
        to 16, as a composition."""
        full = self.available_tags["PSMtag_5plex"]
        tag = full.with_decoy_channel()
        assert tag.decoy_channel == "20"
        step = full.channel_comp["4"] - full.channel_comp["0"]
        assert tag.channel_comp["20"] == full.channel_comp["16"] + step
        assert dict(tag.channel_comp["20"]) == {"C": 4, "H": 16, "O": 1,
                                                "C[13]": 14, "N[15]": 2, "O[18]": 2}
        expected = mass.calculate_mass(tag.channel_comp["20"])
        assert tag.channel_masses[-1] == pytest.approx(expected)
        assert tag.delta[-1] == pytest.approx(expected - full.mass)
        # one step past the last real channel, in mass as well as in isotopes
        assert tag.delta[-1] - full.delta[-1] == pytest.approx(full.delta[1] - full.delta[0], abs=1e-3)

    def test_the_step_comes_from_the_channels_being_searched(self):
        """0, 8 steps by 8, not by the tag's own 4."""
        tag = set_config_tag(self.available_tags, "PSMtag_5plex", "0,8").with_decoy_channel()
        assert tag.decoy_channel == "16"

    def test_a_tag_without_compositions_steps_by_mass(self):
        tag = massTag(rules="nK", base_mass=100.0, delta=[0.0, 4.0, 8.0],
                      channel_names=["0", "4", "8"], name="no_comps").with_decoy_channel()
        assert tag.decoy_channel == "12" and tag.channel_comp is None
        assert tag.channel_masses[-1] == pytest.approx(112.0)

    def test_one_channel_cannot_make_a_decoy_channel(self):
        tag = set_config_tag(self.available_tags, "PSMtag_5plex", "0")
        with pytest.raises(JModError, match="at least two channels"):
            tag.with_decoy_channel()

    def test_channels_need_not_be_evenly_spaced(self):
        """A tag whose channels are 0, 4, 6: the step is the first to the second
        (4), by name and by composition, not the spacing at the end."""
        comps = {"0": mass.Composition({"C": 6}),
                 "4": mass.Composition({"C": 2, "C[13]": 4}),
                 "6": mass.Composition({"C": 0, "C[13]": 6})}
        full = massTag(rules="K", base_mass=mass.calculate_mass(comps["0"]),
                       delta=[mass.calculate_mass(c) - mass.calculate_mass(comps["0"])
                              for c in comps.values()],
                       channel_names=["0", "4", "6"], name="uneven", compositions=comps)
        tag = full.with_decoy_channel()
        assert tag.decoy_channel == "10"                       # 6 + (4 - 0), not 6 + 2
        assert tag.channel_comp["10"] == comps["6"] + (comps["4"] - comps["0"])



class Test_resolve_decoy_channel():

    available_tags = refresh_tags(Path("tests/MassTags"))

    @pytest.mark.parametrize("setting", ["none", "None", "off", "", None])
    def test_no_decoy_channel(self, setting):
        tag = set_config_tag(self.available_tags, "PSMtag_5plex", "0,4,8,12")
        assert resolve_decoy_channel(tag, setting).decoy_channel is None

    def test_no_tag(self):
        assert resolve_decoy_channel(None, "auto") is None

    def test_auto(self):
        tag = resolve_decoy_channel(set_config_tag(self.available_tags, "PSMtag_5plex", "0,4,8"), "auto")
        assert tag.decoy_channel == "12" and tag.channel_names == ["0", "4", "8", "12"]

    def test_a_named_channel_already_being_searched(self):
        """The 9-plex runs used 2 or 14 as the empty channel, not the last one."""
        tag = resolve_decoy_channel(set_config_tag(self.available_tags, "PSMtag_5plex", "0,4,8,16"), "16")
        assert tag.decoy_channel == "16" and tag.channel_names == ["0", "4", "8", "16"]

    def test_a_named_channel_leaves_the_available_tag_alone(self):
        resolve_decoy_channel(self.available_tags["PSMtag_5plex"], "8")
        assert self.available_tags["PSMtag_5plex"].decoy_channel is None

    def test_auto_is_skipped_when_there_is_no_step_to_take(self, app_log):
        """auto is the default, so a tag searching one channel must go through it
        untouched rather than stop the experiment."""
        for tag in (self.available_tags["test_one_channel"],                    # a one-channel tag
                    set_config_tag(self.available_tags, "PSMtag_5plex", "0")):  # one channel searched
            assert resolve_decoy_channel(tag, "auto").decoy_channel is None
        assert any("No decoy channel" in r.getMessage() for r in app_log)

    def test_a_channel_not_being_searched_raises(self):
        tag = set_config_tag(self.available_tags, "PSMtag_5plex", "0,4,8")
        with pytest.raises(JModError, match="not one of the channels being searched"):
            resolve_decoy_channel(tag, "16")
