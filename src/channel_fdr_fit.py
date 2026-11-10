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

"""The channel-level FDR: Channel_Qvalue, from the channel features.

The decoy channel is a tag channel left empty in the sample but searched like
any other, so whatever is identified in it is a null draw.  One model is fitted
on the run's channel features: every real channel's rows the positives, the
decoy channel's the negatives.  Pooled, not one model per channel: a model of
one channel could tell it from the decoy by its mass offset alone instead of by
the evidence, and pooling dilutes that.  Each real channel is then ranked
against the whole decoy channel for its q-values -- one decoy entry per
precursor, as each real channel has, so the ratio is 1.

Semi-supervised, as the run's own precursor scoring is: after the first fit,
real-channel rows the model already calls decoy-like (PEP >= PEP_CUT, against
the decoy channel in their own channel) are
relabelled as decoys, and the model is refitted on the decoys, those, and the
confident real rows (Channel_Qvalue <= TRAIN_Q in their channel).  Every
prediction is out of fold, grouped by peptide so a peptide's channels and
charges are never split between training and test.
"""

import numpy as np
import pandas as pd

from src import config
from src.channel_fdr import IMPORTANCE_COLUMN, IMPORTANCE_SPREAD_COLUMN
from src.fdr_analysis import estimate_pep, score_model
from src.logger import logger

N_ITERATIONS = 3
TRAIN_Q = 0.01          # a real row this confident is a positive for the refit
PEP_CUT = 0.90          # a real row this decoy-like is relabelled a decoy for it
# Below this many decoy-channel rows the q-values are too coarse to mean much
# (the smallest is 1 / n), so there is no channel FDR
MIN_DECOYS = 100
# The CV groups: a peptide's channels, and its charges, in one fold
GROUP_COLUMN = "untag_seq"
# Models that take a missing value as it is; the features are full of them
_NAN_MODELS = ("xg", "rf")


def qvalues(score, is_decoy, ratio=1.0):
    """Target-decoy q-values, the estimator score_precursors uses for Qvalue:
    (1 + decoys at or above) / (targets at or above) x *ratio*, made monotone."""
    order = np.argsort(-score, kind="stable")
    decoy = is_decoy[order]
    with np.errstate(divide="ignore"):   # no target yet at the very top: inf, monotonized away
        q = (1 + np.cumsum(decoy)) / np.cumsum(~decoy) * ratio
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(q.size)
    out[order] = q
    return out


def per_channel_qvalues(channel, score, is_decoy, ratio=1.0):
    """Each real channel's q-values against the whole decoy channel; NaN for the
    decoy channel, which has none."""
    q = np.full(channel.size, np.nan)
    for ch in np.unique(channel[~is_decoy]):
        rows = np.flatnonzero((channel == ch) | is_decoy)
        in_channel = ~is_decoy[rows]
        q[rows[in_channel]] = qvalues(score[rows], is_decoy[rows], ratio)[in_channel]
    return q


def per_channel_pep(channel, score, is_decoy):
    """Each real channel's PEP against the whole decoy channel, as its q-values
    are taken; NaN for the decoy channel.  Not pooled over the real channels:
    each of them has its own null, as large as the decoy channel, so a pooled
    PEP would put several channels' false hits against one channel's decoys and
    come out several times too low."""
    pep = np.full(channel.size, np.nan)
    for ch in np.unique(channel[~is_decoy]):
        rows = np.flatnonzero((channel == ch) | is_decoy)
        in_channel = ~is_decoy[rows]
        pep[rows[in_channel]] = estimate_pep(score[rows], is_decoy[rows])[in_channel]
    return pep


def model_matrix(features, columns):
    """The model's input: *columns* of *features* as floats, less those with
    fewer than two values (all null, say, when the run has no ion mobility).
    Returns (the matrix, the columns dropped)."""
    X = features.select(columns).to_pandas().astype(float)
    dead = [c for c in X.columns if X[c].nunique(dropna=True) < 2]
    X = X.drop(columns=dead)
    infinite = X.columns[np.isinf(X.to_numpy()).any(axis=0)].tolist()
    if infinite:
        raise ValueError(f"Channel FDR: infinite values in {infinite}")
    return X, dead


def feature_importance(model):
    """Each feature's share of the model's gain: a table, one row per feature,
    highest first, of its mean (IMPORTANCE_COLUMN) and standard deviation
    (IMPORTANCE_SPREAD_COLUMN) over the CV folds' models."""
    rows = []
    for fold in model.models:
        if model.model_type == "xg":
            gain = fold.model.get_score(importance_type="gain")
            rows.append([gain.get(c, 0.0) for c in model.columns])
        else:
            rows.append(list(np.asarray(fold.model.feature_importances_, dtype=float)))
    table = pd.DataFrame(rows, columns=model.columns)
    table = table.div(table.sum(axis=1).replace(0, np.nan), axis=0)
    return (pd.DataFrame({IMPORTANCE_COLUMN: table.mean(), IMPORTANCE_SPREAD_COLUMN: table.std(ddof=0)})
            .rename_axis("feature").sort_values(IMPORTANCE_COLUMN, ascending=False))


def fit_channel_fdr(features, columns, n_iterations=N_ITERATIONS, train_q=TRAIN_Q, pep_cut=PEP_CUT,
                    model_type=None):
    """ChannelScore and Channel_Qvalue for every row of *features* (the channel
    feature table), fitted on its *columns*.

    Returns (score, q-values, feature importance), the first two aligned with
    the rows, the q-values NaN for the decoy channel's rows; or None, with a
    warning, when there are too few decoy-channel rows or no real ones.
    """
    is_decoy = features["is_decoy_channel"].to_numpy().astype(bool)
    channel = features["channel"].to_numpy().astype(float)
    if is_decoy.sum() < MIN_DECOYS or is_decoy.all():
        logger.warning(f"Channel FDR: {int(is_decoy.sum())} decoy-channel and {int((~is_decoy).sum())} "
                       f"real-channel precursors scored; needs at least {MIN_DECOYS} decoy-channel ones "
                       f"and a real one. No Channel_Qvalue")
        return None
    groups = features[GROUP_COLUMN].to_numpy()
    if pd.isna(groups).any():
        raise ValueError(f"Channel FDR: null {GROUP_COLUMN} in the channel features")

    X, dead = model_matrix(features, columns)
    if dead:
        logger.info(f"Channel FDR: {len(dead)} features with no information on this run left out: "
                    f"{', '.join(dead)}")
    model_type = model_type or config.score_model
    if model_type not in _NAN_MODELS:
        logger.warning(f"Channel FDR: the {model_type!r} model cannot take missing features; using 'xg'")
        model_type = "xg"
    y = (~is_decoy).astype(int)
    logger.info(f"Channel FDR: {int((~is_decoy).sum())} real-channel vs {int(is_decoy.sum())} "
                f"decoy-channel precursors, {X.shape[1]} features, {model_type} model")

    score = q = model = None
    for iteration in range(n_iterations):
        fitter = score_model(model_type)
        if iteration == 0:
            score = fitter.run_model(X, y, groups=groups)
        else:
            # Per channel, as the q-values (the reference script pooled it)
            mined = ~is_decoy & (per_channel_pep(channel, score, is_decoy) >= pep_cut)
            # q is per channel, so the positives are each channel's confident rows
            confident = ~is_decoy & ~mined & (q <= train_q)
            keep = is_decoy | confident | mined
            y_refit = np.where(mined, 0, y)
            if np.unique(y_refit[keep]).size < 2:
                logger.warning(f"Channel FDR: iteration {iteration + 1} would train on one class; "
                               f"keeping iteration {iteration}")
                break
            logger.info(f"Channel FDR iteration {iteration + 1}: training on {int(keep.sum())} of "
                        f"{keep.size}: {int(confident.sum())} confident real-channel, {int(mined.sum())} "
                        f"relabelled as decoys")
            score = fitter.run_model_filtered(X, y, keep, y_refit, groups=groups)
        model = fitter
        q = per_channel_qvalues(channel, score, is_decoy)
    model.columns = list(X.columns)
    return score, q, feature_importance(model)
