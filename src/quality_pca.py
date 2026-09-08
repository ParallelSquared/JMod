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

"""Unsupervised PSM-quality scores: PC1 over transformed per-PSM features.

The first search has no decoys (sagepy runs with ``generate_decoys=False``), so
there is nothing to fit a discriminant against and PC1 is the unsupervised
stand-in. Its sign is arbitrary out of the decomposition, so it is flipped to make
higher = better.

Features are transformed toward normality before z-scoring, since PCA on a skewed
column is driven by its tail. Missing or sentinel values raise rather than being
imputed -- substituting one shifts that column's mean and std, which moves the
loadings for every PSM.
"""

import numpy as np

# (column, transform). Transform is applied before z-scoring.
#   "log1p" -- counts/magnitudes with a long right tail
#   "abs"   -- signed errors where only magnitude is informative (the signed
#              value cancels: instrument offset centres targets near 0)
#   None    -- already near-normal, or bounded
FIRST_SEARCH_FEATURES = [
    ("scribe_score", None),
    ("hellinger_score", "log1p"),
    ("spectral_contrast_angle", None),
    ("hyperscore", "log1p"),
    ("matched_peaks", "log1p"),
    ("longest_b", "log1p"),
    ("longest_y", "log1p"),
    ("longest_y_pct", None),
    ("matched_lib_pct", None),
    ("delta_best", "log1p"),
    ("delta_next", "log1p"),
    ("average_ppm", "log1p"),
    ("ppm_error_ms1", "abs"),
]


def _transform(x, how):
    if how == "log1p":
        # Not clipped at 0. log1p is defined down to -1, and clipping silently
        # collapses a feature's whole negative tail onto one value -- hellinger_score
        # is negative for ~22% of first-search PSMs. Out of domain raises instead.
        if x.min() <= -1.0:
            raise ValueError(f"log1p needs values > -1, got min {x.min():.4g}; "
                             f"use abslog1p for a feature that goes below -1")
        return np.log1p(x)
    if how == "abslog1p":
        return np.log1p(np.abs(x))
    if how == "abs":
        return np.abs(x)
    if how is None:
        return x
    raise ValueError(f"unknown transform {how!r}")


def _feature_matrix(df, features, label):
    """Transformed feature matrix, validated. Raises on missing values.

    Note the -999 test runs on transformed values, so it only catches a sentinel
    that survives its transform. Today that is every case: no feature carries a
    sentinel on either search's real data.
    """
    cols = [c for c, _ in features]
    X = np.column_stack([
        _transform(df[c].to_numpy().astype(np.float64), t) for c, t in features
    ])

    bad = ~np.isfinite(X) | (X == -999.0)
    if bad.any():
        counts = {c: int(n) for c, n in zip(cols, bad.sum(axis=0)) if n}
        raise ValueError(
            f"non-finite or sentinel values in {label} features: {counts}. Fix the "
            f"producer -- dropping or imputing these rows here would move the "
            f"loadings for every PSM.")
    return X


def first_search_pc1(df):
    """Score each row of a first-search frame by its projection onto PC1.

    ``df`` is the collapsed first-search frame, one row per precursor. Returns one
    score per row, higher = better.
    """
    from sklearn.decomposition import PCA

    cols = [c for c, _ in FIRST_SEARCH_FEATURES]
    X = _feature_matrix(df, FIRST_SEARCH_FEATURES, "first-search PCA")

    Z = (X - X.mean(axis=0)) / X.std(axis=0)

    pca = PCA(n_components=1).fit(Z)
    v = pca.components_[0]
    # PC1's sign is arbitrary out of the decomposition. hyperscore is
    # unambiguously higher-is-better, so orient the axis by it.
    if v[cols.index("hyperscore")] < 0:
        v = -v

    return Z @ v


# Intensity is in here and out of FIRST_SEARCH_FEATURES: within one precursor's
# elution the apex IS where signal peaks, so it is signal rather than the
# between-precursor confound it is globally.
APEX_FEATURES = [
    ("closest_peak_intensity_ms1", "log1p"),
    ("ms2_intensity", "log1p"),
    ("scribe_score", None),
    ("hellinger_score", "log1p"),
    ("spectral_contrast_angle", None),
    ("hyperscore", "log1p"),
    ("matched_peaks", "log1p"),
    ("matched_lib_pct", None),
    ("longest_b", "log1p"),
    ("longest_y", "log1p"),
    ("longest_y_pct", None),
    ("delta_best", "log1p"),
    ("delta_next", "log1p"),
    ("average_ppm", "log1p"),
    ("ppm_error_ms1", "abs"),
]


def _within_group_zscore(df, features, group_cols, label):
    """Z-score each feature within its group.

    Returns (Z, rows-per-group broadcast to each row).
    """
    import pandas as pd

    X = _feature_matrix(df, features, label)

    cols = list(range(X.shape[1]))
    frame = pd.DataFrame(X)
    for c in group_cols:
        frame[c] = df[c].to_numpy()
    grouped = frame.groupby(list(group_cols), sort=False)

    mean = grouped[cols].transform("mean").to_numpy()
    std = grouped[cols].transform("std", ddof=0).to_numpy()
    counts = grouped[cols[0]].transform("size").to_numpy()

    # Constant within a group carries no information there: z = 0, not a divide by
    # zero. Same convention as the polars path in main_apex_pc1_expr.
    varies = std > 0
    return np.where(varies, (X - mean) / np.where(varies, std, 1.0), 0.0), counts


def fit_within_group_pc1(df, features, group_cols, anchor, label):
    """Fit PC1 on within-group z-scores and return its loadings.

    Split out from ``within_group_pc1`` because the main search fits on a sample of
    precursors and projects the full frame in polars -- the eigenvector is stable
    under subsampling, so there is no reason to fit on tens of millions of rows.
    """
    from sklearn.decomposition import PCA

    Z, counts = _within_group_zscore(df, features, group_cols, label)
    # Single-member groups are all zeros and would drag the fit toward the origin.
    pca = PCA(n_components=1).fit(Z[counts >= 2])
    v = pca.components_[0]
    if v[[c for c, _ in features].index(anchor)] < 0:
        v = -v
    return Z, v


def within_group_pc1(df, features, group_cols, anchor, label):
    """Score every row for how apex-like it is within its own group.

    Port of JrMod's ``peak_features::compute_pca_apex``. Features are z-scored
    WITHIN group before the decomposition, so PC1 describes how one precursor's
    scans differ from each other rather than how precursors differ from one
    another -- a globally-fitted PC1 is dominated by between-precursor abundance
    and says nothing about which scan is the apex.

    Generic in features/group_cols/anchor: the main search is the same problem with
    different columns. Single-member groups score 0.
    Returns one score per row, higher = more apex-like.
    """
    Z, v = fit_within_group_pc1(df, features, group_cols, anchor, label)
    return Z @ v


def first_search_apex_pc1(df):
    """Apex score for the pre-collapse first-search frame, grouped by (seq, z).

    Replaces picking the most intense MS1 scan, which uses one number and ignores
    every match feature the first search already computed. Anchored on that same
    MS1 intensity, which unambiguously peaks at the apex. The main search anchors on
    ``coeff`` instead -- ``no_ms1_req`` lets MS1-less PSMs through there.
    """
    return within_group_pc1(df, APEX_FEATURES, ["seq", "z"],
                            anchor="closest_peak_intensity_ms1",
                            label="First-search apex PC1")


# Main-search apex features. No MS1-derived column appears here: --no_ms1_req lets
# MS1-less PSMs through the main search, and a feature that is meaningless for part
# of the population would bias the loadings for all of it. Transforms follow the
# measured minima -- gof_stats, manhattan_distances and max_matched_residuals all go
# below -1, so log1p is undefined for them.
MAIN_APEX_FEATURES = [
    ("coeff", "log1p"),
    ("hyperscore", "log1p"),
    ("scribe_scores", "log1p"),
    ("gof_stats", None),
    ("manhattan_distances", "abslog1p"),
    ("frac_lib_int", "log1p"),
    ("b_counts", "log1p"),
    ("y_counts", "log1p"),
    ("num_lib", "log1p"),
    ("max_matched_residuals", "abslog1p"),
]

# Precursors sampled to fit the eigenvector, so fit cost does not grow with input
# size. Measured against a full-data fit on a 9-plex run: at this count 0.13% of
# apex picks differ, rising to 0.30% at 8k precursors and 0.86% at 800. There is no
# plateau -- accuracy improves roughly as 1/sqrt(n) -- so this is a stopping point,
# not an optimum, chosen because it is the size the loadings were validated at.
FIT_SAMPLE_PRECURSORS = 50_000


def fit_main_apex_pc1(lf, group_cols, seed):
    """Fit main-search apex PC1 on a sample of precursors; return loadings.

    ``lf`` is the un-collapsed main-search LazyFrame. The sample is taken as a hash
    predicate so polars pushes it into the parquet scan and reads row groups in file
    order -- a row-index gather would scatter reads across a multi-GB file.
    """
    import polars as pl

    cols = [c for c, _ in MAIN_APEX_FEATURES]
    keys = pl.struct(group_cols)

    # Hash buckets, sized from the actual precursor count so the sample is a fixed
    # number rather than a fixed fraction. Scans two columns; cheap next to the fit.
    n_precursors = lf.select(group_cols).unique().select(pl.len()).collect().item()
    keep = min(1000, max(1, -(-1000 * FIT_SAMPLE_PRECURSORS // max(n_precursors, 1))))

    sample = (lf
              .filter(keys.hash(seed=seed).mod(1000) < keep)
              .select(cols + list(group_cols))
              .collect())
    _, v = fit_within_group_pc1(sample, MAIN_APEX_FEATURES, group_cols,
                                anchor="coeff", label="Main-search apex PC1")
    return v


def main_apex_pc1_expr(loadings, group_cols):
    """Polars expression projecting every row onto the fitted apex PC1.

    Kept as an expression so the score is computed inside the existing lazy chain
    rather than materializing the feature matrix for a frame that can reach 37M rows.
    """
    import polars as pl

    terms = []
    for (c, how), w in zip(MAIN_APEX_FEATURES, loadings):
        # Cast first: these columns are Float32 on disk, and standardizing then
        # accumulating a 10-term weighted sum in float32 drifts ~3e-4 from the
        # float64 fit, enough to reorder scans within a tight peak.
        e = pl.col(c).cast(pl.Float64)
        if how == "log1p":
            e = e.log1p()
        elif how == "abslog1p":
            e = e.abs().log1p()
        elif how is not None:
            raise ValueError(f"unknown transform {how!r}")
        std = e.std(ddof=0).over(group_cols)
        # Constant within a group carries no information there: 0, not a divide by
        # zero. Matches the numpy path in _within_group_zscore.
        z = pl.when(std > 0).then((e - e.mean().over(group_cols)) / std).otherwise(0.0)
        terms.append(z * w)
    return pl.sum_horizontal(terms)
