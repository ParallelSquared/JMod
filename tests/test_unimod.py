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

import io

import pytest

from src.unimod import Unimod

_XML = b"""<?xml version="1.0" encoding="utf-8"?>
<umod:unimod xmlns:umod="http://www.unimod.org/xmlns/schema/unimod_2">
  <umod:modifications>
    <umod:mod title="Carbamidomethyl" full_name="Iodoacetamide derivative" record_id="4">
      <umod:specificity hidden="0" site="C" position="Anywhere" classification="Chemical derivative" spec_group="1"/>
      <umod:delta mono_mass="57.021464" avge_mass="57.0513" composition="H(3) C(2) N O">
        <umod:element symbol="H" number="3"/>
        <umod:element symbol="C" number="2"/>
        <umod:element symbol="N" number="1"/>
        <umod:element symbol="O" number="1"/>
      </umod:delta>
    </umod:mod>
    <umod:mod title="Label:13C(6)" full_name="13C(6) Silac label" record_id="188">
      <umod:delta mono_mass="6.020129" avge_mass="5.9559" composition="C(-6) 13C(6)">
        <umod:element symbol="C" number="-6"/>
        <umod:element symbol="13C" number="6"/>
      </umod:delta>
    </umod:mod>
  </umod:modifications>
</umod:unimod>
"""


def test_reads_masses_and_compositions():
    unimod = Unimod(io.BytesIO(_XML))
    assert [m["record_id"] for m in unimod.mods] == [4, 188]
    cam = unimod.by_id(4)
    assert cam["title"] == "Carbamidomethyl"
    assert cam["mono_mass"] == pytest.approx(57.021464)
    assert cam["composition"] == {"H": 3, "C": 2, "N": 1, "O": 1}


def test_isotopes_are_keyed_like_pyteomics():
    assert Unimod(io.BytesIO(_XML)).by_id(188)["composition"] == {"C": -6, "C[13]": 6}


def test_by_id_accepts_a_string_id():
    assert Unimod(io.BytesIO(_XML)).by_id("4")["record_id"] == 4
