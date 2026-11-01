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

"""Unimod modifications, read from a unimod.xml.

Parsed with the standard library's xml.etree: lxml (pyteomics' Unimod) has
not declared free-threading support, and importing it turns the GIL back on.

Each modification is a dict like pyteomics.mass.Unimod's, with the fields
JMod reads: record_id, title, mono_mass, and composition, the delta's
element counts keyed as pyteomics keys them ("C", or "C[13]" for 13C).
"""

import re
import xml.etree.ElementTree as ET

_NS = "{http://www.unimod.org/xmlns/schema/unimod_2}"
_ISOTOPE_SYMBOL = re.compile(r"^(\d*)(\D+)$")


def _element_key(symbol):
    """'13C' -> 'C[13]', 'C' -> 'C'."""
    isotope, element = _ISOTOPE_SYMBOL.match(symbol).groups()
    return f"{element}[{int(isotope)}]" if isotope else element


def _composition(delta):
    comp = {}
    for elem in delta.iter(f"{_NS}element"):
        key = _element_key(elem.get("symbol"))
        comp[key] = comp.get(key, 0) + int(elem.get("number"))
    return {k: v for k, v in comp.items() if v != 0}


class Unimod:
    """The modifications in a unimod.xml (a path or a binary file object)."""

    def __init__(self, source):
        root = ET.parse(source).getroot()
        self._mods = []
        self._id = {}
        for mod in root.iter(f"{_NS}mod"):
            delta = mod.find(f"{_NS}delta")
            record = {
                "record_id": int(mod.get("record_id")),
                "title": mod.get("title"),
                "full_name": mod.get("full_name"),
                "mono_mass": float(delta.get("mono_mass")),
                "avge_mass": float(delta.get("avge_mass")),
                "composition": _composition(delta),
            }
            self._id[record["record_id"]] = len(self._mods)
            self._mods.append(record)

    @property
    def mods(self):
        """Every modification, in file order."""
        return self._mods

    def by_id(self, record_id):
        """The modification with this Unimod record id."""
        return self._mods[self._id[int(record_id)]]
