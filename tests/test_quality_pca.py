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

"""Tests for the first-search PC1 quality score."""

import numpy as np
import pandas as pd
import pytest

from src.quality_pca import first_search_apex_pc1, first_search_pc1


def _frame(n=400, seed=0):
    """A frame with a real quality axis: one latent factor drives every feature.

    Error features (average_ppm, ppm_error_ms1, delta_best) move OPPOSITE the
    latent quality, which is what makes the sign checks below meaningful.
    """
    rng = np.random.default_rng(seed)
    q = rng.normal(size=n)  # latent quality, higher = better
    noise = lambda s=0.3: rng.normal(scale=s, size=n)

    return pd.DataFrame({
        "scribe_score": 2.0 + q + noise(),
        "hellinger_score": 2.0 + q + noise(),
        "spectral_contrast_angle": np.clip(0.6 + 0.1 * q + noise(0.05), 0.01, 0.99),
        "hyperscore": np.clip(30.0 + 8.0 * q + noise(3.0), 1.0, None),
        "matched_peaks": np.clip(12.0 + 3.0 * q + noise(1.0), 3.0, None),
        "longest_b": np.clip(4.0 + q + noise(0.5), 0.0, None),
        "longest_y": np.clip(6.0 + 1.5 * q + noise(0.5), 0.0, None),
        "longest_y_pct": np.clip(0.4 + 0.08 * q + noise(0.03), 0.0, 1.0),
        "matched_lib_pct": np.clip(50.0 + 12.0 * q + noise(5.0), 0.0, 100.0),
        "delta_best": np.clip(5.0 - 1.5 * q + noise(0.5), 0.0, None),
        "delta_next": np.clip(3.0 + q + noise(0.5), 0.0, None),
        "average_ppm": np.clip(4.0 - 1.0 * q + noise(0.4), 0.05, None),
        "ppm_error_ms1": (4.0 - q + noise(0.4)) * rng.choice([-1.0, 1.0], size=n),
        "q_latent": q,
    })


class TestFirstSearchPC1:
    def test_recovers_the_latent_quality_axis(self):
        df = _frame()
        scores = first_search_pc1(df)
        assert np.corrcoef(scores, df["q_latent"])[0, 1] > 0.9

    def test_higher_is_better(self):
        # PCA's eigenvector sign is arbitrary, so this is the check that the
        # anchoring works. If it fails, empirical_fit's percentile filter selects
        # the WORST PSMs to fit the RT calibration on.
        df = _frame()
        scores = first_search_pc1(df)
        assert np.corrcoef(scores, df["hyperscore"])[0, 1] > 0
        assert np.corrcoef(scores, df["average_ppm"])[0, 1] < 0

    def test_returns_one_score_per_row(self):
        df = _frame(n=137)
        assert first_search_pc1(df).shape == (137,)

    def test_sentinel_raises(self):
        # Imputing would move the column mean and std, shifting the loadings for
        # every PSM rather than just the affected rows.
        df = _frame()
        df.loc[df.index[:3], "scribe_score"] = -999.0
        with pytest.raises(ValueError, match="scribe_score"):
            first_search_pc1(df)

    def test_nan_raises(self):
        df = _frame()
        df.loc[df.index[0], "hyperscore"] = np.nan
        with pytest.raises(ValueError, match="non-finite or sentinel"):
            first_search_pc1(df)


def _apex_frame(n_prec=120, n_scans=7, seed=1):
    """Precursors eluting across several scans, with a planted apex per precursor.

    Quality rises and falls with distance from that scan, so the correct answer is
    known per group.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_prec):
        apex = rng.integers(0, n_scans)
        for s in range(n_scans):
            # 1.0 at the apex, falling off either side
            w = np.exp(-0.5 * ((s - apex) / 1.5) ** 2)
            noise = lambda sd: rng.normal(scale=sd)
            rows.append({
                "seq": f"PEPTIDE{p}", "z": 2, "scan": s, "is_apex": s == apex,
                "closest_peak_intensity_ms1": 1e4 * w + 50 + noise(20),
                "ms2_intensity": 1e3 * w + 20 + noise(5),
                "scribe_score": 2.0 * w + noise(0.1),
                "hellinger_score": 2.0 * w + noise(0.1),
                "spectral_contrast_angle": np.clip(0.3 + 0.5 * w + noise(0.02), 0.01, 0.99),
                "hyperscore": 10.0 + 25.0 * w + noise(1.0),
                "matched_peaks": 4.0 + 10.0 * w + noise(0.5),
                "matched_lib_pct": np.clip(20.0 + 50.0 * w + noise(2.0), 0, 100),
                "longest_b": 2.0 + 3.0 * w + noise(0.2),
                "longest_y": 3.0 + 5.0 * w + noise(0.2),
                "longest_y_pct": np.clip(0.2 + 0.4 * w + noise(0.02), 0, 1),
                "delta_best": np.clip(4.0 - 3.0 * w + noise(0.2), 0, None),
                "delta_next": np.clip(1.0 + 2.0 * w + noise(0.2), 0, None),
                "average_ppm": np.clip(5.0 - 3.0 * w + noise(0.2), 0.05, None),
                "ppm_error_ms1": (5.0 - 3.0 * w + noise(0.2)) * rng.choice([-1.0, 1.0]),
            })
    return pd.DataFrame(rows)


class TestApexPC1:
    def test_picks_the_planted_apex(self):
        df = _apex_frame()
        df["score"] = first_search_apex_pc1(df)
        picked = df.loc[df.groupby(["seq", "z"])["score"].idxmax()]
        assert picked["is_apex"].mean() > 0.9

    def test_single_scan_precursors_score_zero(self):
        # No within-group variation, and they are their group's only candidate.
        df = _apex_frame(n_prec=40, n_scans=7)
        solo = _apex_frame(n_prec=10, n_scans=1, seed=2)
        solo["seq"] = "SOLO" + solo["seq"]
        both = pd.concat([df, solo], ignore_index=True)
        scores = first_search_apex_pc1(both)
        assert np.allclose(scores[both["seq"].str.startswith("SOLO").to_numpy()], 0.0)

    def test_charge_states_are_separate_groups(self):
        df = _apex_frame(n_prec=30)
        other = _apex_frame(n_prec=30, seed=3)
        other["z"] = 3
        both = pd.concat([df, other], ignore_index=True)
        both["score"] = first_search_apex_pc1(both)
        picked = both.loc[both.groupby(["seq", "z"])["score"].idxmax()]
        assert len(picked) == 60
        assert picked["is_apex"].mean() > 0.9

    def test_higher_is_more_apex_like(self):
        df = _apex_frame()
        scores = first_search_apex_pc1(df)
        assert np.corrcoef(scores, df["closest_peak_intensity_ms1"])[0, 1] > 0
