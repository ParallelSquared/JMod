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
        return np.log1p(np.clip(x, 0.0, None))
    if how == "abs":
        return np.abs(x)
    if how is None:
        return x
    raise ValueError(f"unknown transform {how!r}")


def first_search_pc1(df):
    """Score each row of a first-search frame by its projection onto PC1.

    ``df`` is the collapsed first-search frame, one row per precursor. Returns one
    score per row, higher = better.
    """
    from sklearn.decomposition import PCA

    cols = [c for c, _ in FIRST_SEARCH_FEATURES]
    X = np.column_stack([
        _transform(df[c].to_numpy().astype(np.float64), t)
        for c, t in FIRST_SEARCH_FEATURES
    ])

    bad = ~np.isfinite(X) | (X == -999.0)
    if bad.any():
        counts = {c: int(n) for c, n in zip(cols, bad.sum(axis=0)) if n}
        raise ValueError(
            f"non-finite or sentinel values in PCA features: {counts}. Fix the "
            f"producer -- dropping or imputing these rows here would move the "
            f"loadings for every PSM.")

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


def within_group_pc1(df, features, gid, anchor, label):
    """Score every row for how apex-like it is within its own group (``gid``).

    Port of JrMod's ``peak_features::compute_pca_apex``. Features are z-scored
    WITHIN group before the decomposition, so PC1 describes how one precursor's
    scans differ from each other rather than how precursors differ from one
    another -- a globally-fitted PC1 is dominated by between-precursor abundance
    and says nothing about which scan is the apex.

    Generic in features/gid/anchor: the main search is the same problem with
    different columns. Single-member groups score 0 and are excluded from the fit.
    Returns one score per row, higher = more apex-like.
    """
    from sklearn.decomposition import PCA

    cols = [c for c, _ in features]
    X = np.column_stack([
        _transform(df[c].to_numpy().astype(np.float64), t) for c, t in features
    ])

    bad = ~np.isfinite(X) | (X == -999.0)
    if bad.any():
        counts_bad = {c: int(n) for c, n in zip(cols, bad.sum(axis=0)) if n}
        raise ValueError(f"non-finite or sentinel values in {label} features: {counts_bad}")

    n_groups = int(gid.max()) + 1 if len(gid) else 0
    counts = np.bincount(gid, minlength=n_groups).astype(np.float64)

    mean = np.column_stack([
        np.bincount(gid, weights=X[:, j], minlength=n_groups) / counts
        for j in range(X.shape[1])
    ])
    sq = np.column_stack([
        np.bincount(gid, weights=X[:, j] ** 2, minlength=n_groups) / counts
        for j in range(X.shape[1])
    ])
    std = np.sqrt(np.clip(sq - mean ** 2, 0.0, None))

    varies = std[gid] > 0
    Z = np.where(varies, (X - mean[gid]) / np.where(varies, std[gid], 1.0), 0.0)

    multi = counts[gid] >= 2
    pca = PCA(n_components=1).fit(Z[multi])
    v = pca.components_[0]
    if v[cols.index(anchor)] < 0:
        v = -v

    return Z @ v


def first_search_apex_pc1(df):
    """Apex score for the pre-collapse first-search frame, grouped by (seq, z).

    Replaces picking the most intense MS1 scan, which uses one number and ignores
    every match feature the first search already computed. Anchored on that same
    MS1 intensity, which unambiguously peaks at the apex. (The main search will
    need a different anchor -- ``no_ms1_req`` lets MS1-less PSMs through there.)
    """
    import pandas as pd

    # Charges are single digits, so folding z in at *64 cannot collide.
    seq_code = pd.factorize(df["seq"].to_numpy())[0].astype(np.int64)
    gid = pd.factorize(seq_code * 64 + df["z"].to_numpy().astype(np.int64))[0]

    return within_group_pc1(df, APEX_FEATURES, gid,
                            anchor="closest_peak_intensity_ms1",
                            label="First-search apex PC1")
