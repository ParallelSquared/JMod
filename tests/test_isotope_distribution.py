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

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import src.isotope_distribution as isotope_distribution
from src.isotope_distribution import isotopic_variants

# brainpy 1.5.19's isotopic_variants(comp, npeaks=k, charge=z) for each case,
# recorded when it was replaced: [(mz, intensity), ...]
BRAINPY = [
    ({"C": 34, "H": 53, "N": 7, "O": 15}, 2, 2,
     [(400.68725848012497, 0.7116842877283829), (401.18876448866575, 0.28831571227161723)]),
    ({"C": 34, "H": 53, "N": 7, "O": 15}, 5, 2,
     [(400.68725848012497, 0.6482643695338968), (401.18876448866575, 0.2626231977651996),
      (401.6900196765983, 0.07185891292087156), (402.19128176060815, 0.014745804451865482),
      (402.69250136850076, 0.0025077153281665814)]),
    ({"C": 63, "C[13]": 8, "H": 91, "N": 15, "O": 20}, 4, 1,
     [(1482.6905955507402, 0.4596157683358667), (1483.6935404859375, 0.34667671542703504),
      (1484.6962697676113, 0.14789437644895853), (1485.698905591969, 0.04581313978813971)]),
    ({"C": 50, "H": 80, "N": 14, "O": 16, "S": 2}, 4, 3,
     [(399.8512146486899, 0.4877337641963178), (400.1854769971623, 0.3038676903464258),
      (400.5187602014616, 0.15283515571918904), (400.85235105066045, 0.055563389738067376)]),
    ({"C": 20, "H": 30, "N": 5, "O": 9, "P": 1}, 3, 0,
     [(515.17811419214, 0.7764565201251927), (516.1810449203634, 0.18748302326050126),
      (517.1833195113828, 0.03606045661430602)]),
    ({"C": 10, "H[2]": 6, "H": 14, "N[15]": 2, "O[18]": 1}, 3, 2,
     [(98.10404642285998, 0.8966945760084517), (98.60574527257599, 0.09842789326621067),
      (99.10744868215508, 0.004877530725337541)]),
    # only fixed isotopes: no variants beyond the first peak
    ({"C[13]": 4}, 3, 1, [(53.02069581797, 1.0)]),
]


@pytest.mark.parametrize("comp, npeaks, charge, expected", BRAINPY)
def test_matches_brainpy(comp, npeaks, charge, expected):
    peaks = isotopic_variants(comp, npeaks=npeaks, charge=charge)
    assert len(peaks) == len(expected)
    for peak, (mz, intensity) in zip(peaks, expected):
        assert peak.mz == pytest.approx(mz, rel=1e-12)
        assert peak.intensity == pytest.approx(intensity, rel=1e-12)
        assert peak.charge == charge


def test_peaks_are_sorted_and_normalised():
    peaks = isotopic_variants({"C": 40, "H": 60, "N": 10, "O": 12}, npeaks=4, charge=2)
    assert [p.mz for p in peaks] == sorted(p.mz for p in peaks)
    assert sum(p.intensity for p in peaks) == pytest.approx(1.0)


def test_rejects_negative_counts_and_bad_npeaks():
    with pytest.raises(ValueError):
        isotopic_variants({"C": 10, "H": -2}, npeaks=3)
    with pytest.raises(ValueError):
        isotopic_variants({"C": 10}, npeaks=0)


def test_new_element_from_many_threads(monkeypatch):
    # Threads that all see an element for the first time each extend the
    # shared table; every one must still get the right answer
    monkeypatch.setattr(isotope_distribution, "_table",
                        isotope_distribution._build_table(["C", "H", "N", "O"]))
    comp = {"C": 30, "H": 50, "N": 8, "O": 10, "Se": 1}
    expected = [(p.mz, p.intensity, p.charge) for p in isotopic_variants(dict(comp), npeaks=4, charge=2)]
    monkeypatch.setattr(isotope_distribution, "_table",
                        isotope_distribution._build_table(["C", "H", "N", "O"]))
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda _: [(p.mz, p.intensity, p.charge) for p in isotopic_variants(dict(comp), 4, 2)],
                                range(64)))
    assert all(r == expected for r in results)
    assert np.isclose(sum(p[1] for p in expected), 1.0)
