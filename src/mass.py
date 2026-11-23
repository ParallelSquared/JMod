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

"""Elemental compositions and monoisotopic masses of peptides and ions.

The part of pyteomics.mass JMod uses, ported from pyteomics 5.0.2
(pyteomics/mass/mass.py and auxiliary/structures.py; Apache License 2.0,
https://github.com/levitsky/pyteomics).  Same tables, same arithmetic in the
same order, so every mass is bit-identical to pyteomics'.  pyteomics.mass
imports lxml (for its Unimod database), which turns the GIL back on.

Differences from pyteomics, all in what is accepted, not in results:
  - a sequence is one-letter residues only (no modX labels or terminal
    groups); pass it as sequence=...
  - a positional string is always a chemical formula (pyteomics tries it as
    a sequence first)
  - no charge carriers or average masses; fast_mass takes a proton charge
"""

import re

from src.mass_data import NIST_MASS

nist_mass = NIST_MASS
PROTON = "H+"

_isotope_string = re.compile(r"^((?:[A-Z][a-z+]*)|e-|e\*)(?:\[(\d+)\])?$")
_atom = r"([A-Z][a-z+]*)(?:\[(\d+)\])?([+-]?\d+)?"
_formula = re.compile(r"^({})*$".format(_atom))


def _make_isotope_string(element_name, isotope_num):
    """'C', 0 -> 'C';  'C', 13 -> 'C[13]'."""
    return element_name if isotope_num == 0 else f"{element_name}[{isotope_num}]"


def _parse_isotope_string(label):
    """'C' -> ('C', 0);  'C[13]' -> ('C', 13)."""
    element_name, num = _isotope_string.match(label).groups()
    return element_name, int(num) if num else 0


class Composition(dict):
    """Element counts, e.g. {'C': 6, 'H': 12, 'C[13]': 2}: a dict that never
    holds a zero (a count reaching 0 is removed), reads missing elements as 0,
    and can be added, subtracted and multiplied by an int.

    Composition()                          empty
    Composition({'H': 2, 'O': 1})          from element counts
    Composition('H2O') / (formula='H2O')   from a formula ("C[13]2" for isotopes)
    Composition(sequence='PEPTIDE')        a peptide with H- and -OH termini
    ion_type='b' (with any of these)       adjusted to that fragment ion type
    """

    def __init__(self, source=None, *, formula=None, sequence=None, ion_type=None):
        super().__init__()
        if sum(x is not None for x in (source, formula, sequence)) > 1:
            raise ValueError("give one of a composition, formula= or sequence=")
        if formula is not None:
            self._from_formula(formula)
        elif sequence is not None:
            self._from_sequence(sequence)
        elif isinstance(source, dict):
            self._from_composition(source)
        elif isinstance(source, str):
            self._from_formula(source)
        elif source is not None:
            raise TypeError(f"cannot make a Composition from {type(source).__name__}")
        if ion_type is not None:
            self += std_ion_comp[ion_type]

    def _from_formula(self, formula):
        if not _formula.match(formula):
            raise ValueError(f"Invalid formula: {formula}")
        for elem, isotope, number in re.findall(_atom, formula):
            self[_make_isotope_string(elem, int(isotope) if isotope else 0)] += int(number) if number else 1

    def _from_sequence(self, sequence):
        comp = {}
        for label in ("H-", *sequence, "-OH"):
            if label not in std_aa_comp:
                raise ValueError(f"No information for {label} in std_aa_comp")
            for elem, cnt in std_aa_comp[label].items():
                comp[elem] = comp.get(elem, 0) + cnt
        self._from_composition(comp)

    def _from_composition(self, comp):
        for isotope_string, num_atoms in comp.items():
            self[_make_isotope_string(*_parse_isotope_string(isotope_string))] = num_atoms

    def __missing__(self, key):
        return 0

    def __setitem__(self, key, value):
        if isinstance(value, float):
            value = int(round(value))
        elif not isinstance(value, int):
            raise TypeError(f"Only integers allowed as values in Composition, got {type(value).__name__}.")
        if value:
            super().__setitem__(key, value)
        elif key in self:
            del self[key]

    def copy(self):
        return type(self)(self)

    def __add__(self, other):
        result = self.copy()
        for elem, cnt in other.items():
            result[elem] += cnt
        return result

    def __iadd__(self, other):
        for elem, cnt in other.items():
            self[elem] += cnt
        return self

    __radd__ = __add__

    def __sub__(self, other):
        result = self.copy()
        for elem, cnt in other.items():
            result[elem] -= cnt
        return result

    def __isub__(self, other):
        for elem, cnt in other.items():
            self[elem] -= cnt
        return self

    def __rsub__(self, other):
        return (self - other) * (-1)

    def __neg__(self):
        return self * (-1)

    def __mul__(self, other):
        if not isinstance(other, int):
            raise TypeError("Cannot multiply Composition by non-integer")
        return type(self)({k: v * other for k, v in self.items()})

    __rmul__ = __mul__

    def __eq__(self, other):
        if not isinstance(other, dict):
            return False
        return {i for i in self.items() if i[1]} == {i for i in other.items() if i[1]}

    def __ne__(self, other):
        return not self == other

    __hash__ = None

    def __repr__(self):
        return f"Composition({dict.__repr__(self)})"

    def mass(self):
        """Monoisotopic mass, summed in key order as pyteomics sums it."""
        if self[PROTON]:
            raise ValueError("Composition contains protons; charge is not supported")
        mass = 0.0
        for isotope_string, amount in self.items():
            element_name, isotope_num = _parse_isotope_string(isotope_string)
            mass += amount * nist_mass[element_name][isotope_num][0]
        return mass


std_aa_comp = {
    'A':   Composition({'H': 5, 'C': 3, 'O': 1, 'N': 1}),
    'C':   Composition({'H': 5, 'C': 3, 'S': 1, 'O': 1, 'N': 1}),
    'D':   Composition({'H': 5, 'C': 4, 'O': 3, 'N': 1}),
    'E':   Composition({'H': 7, 'C': 5, 'O': 3, 'N': 1}),
    'F':   Composition({'H': 9, 'C': 9, 'O': 1, 'N': 1}),
    'G':   Composition({'H': 3, 'C': 2, 'O': 1, 'N': 1}),
    'H':   Composition({'H': 7, 'C': 6, 'N': 3, 'O': 1}),
    'I':   Composition({'H': 11, 'C': 6, 'O': 1, 'N': 1}),
    'J':   Composition({'H': 11, 'C': 6, 'O': 1, 'N': 1}),
    'K':   Composition({'H': 12, 'C': 6, 'N': 2, 'O': 1}),
    'L':   Composition({'H': 11, 'C': 6, 'O': 1, 'N': 1}),
    'M':   Composition({'H': 9, 'C': 5, 'S': 1, 'O': 1, 'N': 1}),
    'N':   Composition({'H': 6, 'C': 4, 'O': 2, 'N': 2}),
    'P':   Composition({'H': 7, 'C': 5, 'O': 1, 'N': 1}),
    'Q':   Composition({'H': 8, 'C': 5, 'O': 2, 'N': 2}),
    'R':   Composition({'H': 12, 'C': 6, 'N': 4, 'O': 1}),
    'S':   Composition({'H': 5, 'C': 3, 'O': 2, 'N': 1}),
    'T':   Composition({'H': 7, 'C': 4, 'O': 2, 'N': 1}),
    'V':   Composition({'H': 9, 'C': 5, 'O': 1, 'N': 1}),
    'W':   Composition({'C': 11, 'H': 10, 'N': 2, 'O': 1}),
    'Y':   Composition({'H': 9, 'C': 9, 'O': 2, 'N': 1}),
    'U':   Composition({'H': 5, 'C': 3, 'O': 1, 'N': 1, 'Se': 1}),
    'O':   Composition({'H': 19, 'C': 12, 'O': 2, 'N': 3}),
    'H-':  Composition({'H': 1}),
    '-OH': Composition({'O': 1, 'H': 1}),
}
"""Residue compositions: the 20 standard amino acids, J, selenocysteine (U),
pyrrolysine (O), and the H- and -OH termini."""


std_ion_comp = {
    'M':        Composition(formula=''),
    'M-H2O':    Composition(formula='H-2O-1'),
    'M-NH3':    Composition(formula='N-1H-3'),
    'a':        Composition(formula='H-2O-1' + 'C-1O-1'),
    'a+1':      Composition(formula='H-2O-1' + 'C-1O-1' + 'H1'),
    'a-H2O':    Composition(formula='H-2O-1' + 'C-1O-1' + 'H-2O-1'),
    'a-NH3':    Composition(formula='H-2O-1' + 'C-1O-1' + 'N-1H-3'),
    'b':        Composition(formula='H-2O-1'),
    'b-H2O':    Composition(formula='H-2O-1' + 'H-2O-1'),
    'b-NH3':    Composition(formula='H-2O-1' + 'N-1H-3'),
    'c':        Composition(formula='H-2O-1' + 'NH3'),
    'c-1':      Composition(formula='H-2O-1' + 'NH3' + 'H-1'),
    'c-dot':    Composition(formula='H-2O-1' + 'NH3' + 'H1'),
    'c+1':      Composition(formula='H-2O-1' + 'NH3' + 'H1'),
    'c+2':      Composition(formula='H-2O-1' + 'NH3' + 'H2'),
    'c-H2O':    Composition(formula='H-2O-1' + 'NH3' + 'H-2O-1'),
    'c-NH3':    Composition(formula='H-2O-1'),
    'x':        Composition(formula='H-2O-1' + 'CO2'),
    'x+1':      Composition(formula='H-2O-1' + 'CO2' + 'H1'),
    'x-H2O':    Composition(formula='H-2O-1' + 'CO2' + 'H-2O-1'),
    'x-NH3':    Composition(formula='H-2O-1' + 'CO2' + 'N-1H-3'),
    'y':        Composition(formula=''),
    'y-H2O':    Composition(formula='H-2O-1'),
    'y-NH3':    Composition(formula='N-1H-3'),
    'z-1':      Composition(formula='H-2O-1' + 'ON-1H-2'),
    'z':        Composition(formula='H-2O-1' + 'ON-1H-1'),
    'z-dot':    Composition(formula='H-2O-1' + 'ON-1'),
    'z+1':      Composition(formula='H-2O-1' + 'ON-1'),
    'z+2':      Composition(formula='H-2O-1' + 'ON-1H1'),
    'z+3':      Composition(formula='H-2O-1' + 'ON-1H2'),
    'z-H2O':    Composition(formula='H-2O-1' + 'ON-1H-1' + 'H-2O-1'),
    'z-NH3':    Composition(formula='H-2O-1' + 'ON-1H-1' + 'N-1H-3'),
}
"""What each fragment ion type adds to the sum of its residues' compositions."""


std_aa_mass = {
    'G': 57.02146372057,
    'A': 71.03711378471,
    'S': 87.03202840427001,
    'P': 97.05276384885,
    'V': 99.06841391299,
    'T': 101.04767846841,
    'C': 103.00918478471,
    'L': 113.08406397713001,
    'I': 113.08406397713001,
    'J': 113.08406397713001,
    'N': 114.04292744114001,
    'D': 115.02694302383001,
    'Q': 128.05857750527997,
    'K': 128.09496301399997,
    'E': 129.04259308796998,
    'M': 131.04048491299,
    'H': 137.05891185845002,
    'F': 147.06841391298997,
    'U': 150.95363508471,
    'R': 156.10111102359997,
    'Y': 163.06332853254997,
    'W': 186.07931294985997,
    'O': 237.14772686284996,
}
"""Monoisotopic residue masses."""


def calculate_mass(source=None, *, formula=None, sequence=None, ion_type=None):
    """Monoisotopic mass of a composition (dict), formula or sequence, as
    Composition takes them."""
    return Composition(source, formula=formula, sequence=sequence, ion_type=ion_type).mass()


def fast_mass(sequence, ion_type=None, charge=None):
    """Monoisotopic mass of a one-letter peptide, or of its fragment ion of
    *ion_type*; with *charge*, its m/z (protonated; |m/z| for a negative
    charge)."""
    try:
        mass = sum(std_aa_mass[i] for i in sequence)
    except KeyError as e:
        raise ValueError(f"No mass data for residue: {e.args[0]}") from None
    mass += nist_mass['H'][0][0] * 2 + nist_mass['O'][0][0]
    if ion_type:
        try:
            icomp = std_ion_comp[ion_type]
        except KeyError:
            raise ValueError(f"Unknown ion type: {ion_type}") from None
        mass += sum(nist_mass[element][0][0] * num for element, num in icomp.items())
    if charge:
        mass = (mass + nist_mass['H+'][0][0] * charge) / charge
        if charge < 0:
            mass = -mass
    return mass
