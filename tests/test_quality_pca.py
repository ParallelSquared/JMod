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

from scipy.stats import spearmanr

from src.quality_pca import (
    MAIN_APEX_FEATURES,
    _transform,
    _within_group_zscore,
    first_search_apex_pc1,
    first_search_pc1,
    fit_within_group_pc1,

    main_apex_pc1_expr,
)


def _frame(n=400, seed=0):
    """A frame with a real quality axis: one latent factor drives every feature.

    Error features (average_ppm, ppm_error_ms1, delta_best) move OPPOSITE the
    latent quality, which is what makes the sign checks below meaningful.
    """
    rng = np.random.default_rng(seed)
    q = rng.normal(size=n)  # latent quality, higher = better
    noise = lambda s=0.3: rng.normal(scale=s, size=n)

    return pd.DataFrame({
        # Floored just above -1: log1p's domain. Real scribe_score bottoms out near
        # 0 and hellinger_score near -0.66, so this matches rather than constrains.
        "scribe_score": np.clip(2.0 + q + noise(), 0.001, None),
        "hellinger_score": np.clip(2.0 + q + noise(), -0.9, None),
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


class TestTransforms:
    def test_log1p_keeps_negatives(self):
        # It used to clip at 0, flattening a feature's whole negative tail onto one
        # value -- hellinger_score is negative for ~22% of real first-search PSMs.
        x = np.array([-0.5, 0.0, 1.0])
        assert np.allclose(_transform(x, "log1p"), np.log1p(x))

    def test_log1p_out_of_domain_raises(self):
        with pytest.raises(ValueError, match="log1p needs values > -1"):
            _transform(np.array([-1.5, 0.0]), "log1p")

    def test_abslog1p_handles_values_below_minus_one(self):
        # gof_stats, manhattan_distances and max_matched_residuals all go below -1.
        x = np.array([-5.6, -1.0, 2.0])
        assert np.allclose(_transform(x, "abslog1p"), np.log1p(np.abs(x)))


class TestMainSearchApex:
    """The main-search path fits on a sample and projects in polars."""

    @staticmethod
    def _frame(n_prec=200, n_scans=6, seed=5):
        rng = np.random.default_rng(seed)
        rows = []
        for p in range(n_prec):
            apex = rng.integers(0, n_scans)
            for s in range(n_scans):
                w = np.exp(-0.5 * ((s - apex) / 1.5) ** 2)
                nz = lambda sd: rng.normal(scale=sd)
                rows.append({
                    "seq": f"PEPTIDE{p}", "z": 2.0, "is_apex": s == apex,
                    "coeff": 1e4 * w + 10 + nz(20),
                    "hyperscore": 10.0 + 25.0 * w + nz(1.0),
                    "scribe_scores": np.clip(0.5 - 0.4 * w + nz(0.02), 1e-4, None),
                    "gof_stats": 2.0 - 4.0 * w + nz(0.2),
                    "manhattan_distances": -1.0 + 3.0 * w + nz(0.2),
                    "frac_lib_int": np.clip(0.3 + 0.5 * w + nz(0.02), 0.01, 1.0),
                    "b_counts": np.clip(2.0 + 3.0 * w + nz(0.3), 0, None),
                    "y_counts": np.clip(3.0 + 5.0 * w + nz(0.3), 0, None),
                    "num_lib": np.clip(10.0 + 5.0 * w + nz(0.3), 1, None),
                    "max_matched_residuals": 3.0 - 5.0 * w + nz(0.2),
                })
        return pd.DataFrame(rows)

    def test_polars_expression_matches_numpy(self):
        # The offline feature work was all numpy; the pipeline projects in polars.
        # Nothing else checks that those two agree.
        import polars as pl

        df = self._frame()
        pdf = pl.from_pandas(df.drop(columns=["is_apex"]))
        _, v = fit_within_group_pc1(df, MAIN_APEX_FEATURES, ["seq", "z"], "coeff", "main")

        expr = pdf.with_columns(main_apex_pc1_expr(v, ["seq", "z"]).alias("s"))["s"].to_numpy()
        Z, _ = _within_group_zscore(df, MAIN_APEX_FEATURES, ["seq", "z"], "main")

        assert np.allclose(expr, Z @ v, atol=1e-9)

    def test_picks_the_planted_apex(self):
        df = self._frame()
        Z, v = fit_within_group_pc1(df, MAIN_APEX_FEATURES, ["seq", "z"], "coeff", "main")
        df["score"] = Z @ v
        picked = df.loc[df.groupby(["seq", "z"])["score"].idxmax()]
        assert picked["is_apex"].mean() > 0.9

    def test_sampled_fit_tracks_full_fit(self):
        # The pipeline fits on a sample of precursors. The eigenvector has to be
        # stable enough under subsampling that the ranking does not move.
        df = self._frame(n_prec=600)
        Z, v_full = fit_within_group_pc1(df, MAIN_APEX_FEATURES, ["seq", "z"], "coeff", "main")

        # Whole precursors, never partial groups -- a split group would have the
        # wrong within-group mean and std.
        keep = {s for i, s in enumerate(sorted(df["seq"].unique())) if i % 5 == 0}
        sub = df[df["seq"].isin(keep)].reset_index(drop=True)
        _, v_sub = fit_within_group_pc1(sub, MAIN_APEX_FEATURES, ["seq", "z"], "coeff", "main")

        assert v_full @ v_sub > 0.99
        assert spearmanr(Z @ v_full, Z @ v_sub).statistic > 0.99

    def test_no_ms1_features(self):
        # --no_ms1_req lets MS1-less PSMs through the main search, so an MS1 column
        # would be meaningless for part of the population and bias every loading.
        names = {c for c, _ in MAIN_APEX_FEATURES}
        assert not (names & {"mz_error", "im_error", "prec_im",
                             "closest_peak_intensity_ms1", "Ms1_spec_id"})
