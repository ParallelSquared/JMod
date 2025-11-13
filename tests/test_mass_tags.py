import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.mass_tags import massTag, read_json_to_massTag, refresh_tags, set_config_tag
import json
import numpy as np
from pathlib import Path



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
        








    



def main():
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
    print(mass_tag)

if __name__ == "__main__":
    main()