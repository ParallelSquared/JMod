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

import polars as pl # TODO Requires pyarrow
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
import os

import src.config as config
import src.quality_pca as quality_pca
from src.logger import logger


names = ["coeff", "spec_id", "Ms1_spec_id",
         "seq", "z", "window_mz", "rt",
         "prec_im",
         "num_lib",
         "frac_lib_int",
         "frac_dia_int",
         "mz_error",
         "rt_error",
         "frac_int_matched",
         "frac_int_pred",
         "spec_r2",
         "prec_r2",
         "prec_r2_uniq",
         "frac_int_uniq",
         "frac_int_uniq_pred",
         "hyperscore",
         "b_counts",
         "y_counts",
         "longest_y_ions",
         "scribe_scores",
         "max_unmatched_residuals",
         "max_matched_residuals",
         "gof_stats",
         "manhattan_distances",
         "fitted_spectral_contrasts",
         "frac_int_matched_pred",
         "frac_int_matched_pred_sigcoeff",
         "cosine",
         "mz",
         "tic",
         "im_error",
         "ent_norm_angle",
         "resp_ent_norm_angle",
         "frag_names",
         "frag_errors",
         "frag_mz",
         "frag_int",
         "obs_int",
         "unique_frag_mz",
         "unique_obs_int",
         "file_name",
         "protein",
         ]

dtypes  = {"coeff":np.float32,
           "spec_id":np.int32,
           "Ms1_spec_id":np.int32,
           "seq":str,
           "z":np.float32,
           "window_mz":np.float32,
           "rt":np.float32,
            "prec_im":np.float32,
            "num_lib":np.float32,
            "frac_lib_int":np.float32,
            "frac_dia_int":np.float32,
            "mz_error":np.float32,
            "rt_error":np.float32,
            "frac_int_matched":np.float32,
            "frac_int_pred":np.float32,
            "spec_r2":np.float32,
            "prec_r2":np.float32,
            "prec_r2_uniq":np.float32,
            "frac_int_uniq":np.float32,
            "frac_int_uniq_pred":np.float32,
            "hyperscore":np.float32,
            "b_counts":np.float32,
            "y_counts":np.float32,
            "longest_y_ions":np.float32,
            "scribe_scores": np.float32,
            "max_unmatched_residuals": np.float32,
            "max_matched_residuals": np.float32,
            "gof_stats": np.float32,
            "manhattan_distances": np.float32,
            "fitted_spectral_contrasts": np.float32,
            "frac_int_matched_pred":np.float32,
            "frac_int_matched_pred_sigcoeff":np.float32,
            "cosine":np.float32,
            "mz":np.float32,
            "tic":np.float32,
            "im_error":np.float32,
            "ent_norm_angle":np.float32,
            "resp_ent_norm_angle":np.float32,
            "frag_names":str,
            "frag_errors":str,
            "frag_mz":str,
            "frag_int":str,
            "obs_int":str,
            "file_name":str,
            "protein":str,
            }


# ---------------------------------------------------------------------------
# Parquet output schema — single source of truth for column names, types,
# and placeholder defaults used by fit_to_lib2 and run_jmod.
# ---------------------------------------------------------------------------

# Each entry: (column_name, polars_type, placeholder_default)
_SCHEMA_SPEC = [
    ("coeff",                       pl.Float32,             0.0),
    ("spec_id",                     pl.Int32,               0),
    ("Ms1_spec_id",                 pl.Int32,               0),
    ("seq",                         pl.Utf8,                ""),
    ("z",                           pl.Float32,             0.0),
    # -- timeplex inserts "time_channel" here at index 5 --
    ("window_mz",                   pl.Float32,             0.0),
    ("rt",                          pl.Float32,             0.0),
    ("prec_im",                     pl.Float32,             0.0),
    # 30 feature columns
    ("num_lib",                     pl.Float32,             0.0),
    ("frac_lib_int",                pl.Float32,             0.0),
    ("frac_dia_int",                pl.Float32,             0.0),
    ("mz_error",                    pl.Float32,             0.0),
    ("rt_error",                    pl.Float32,             0.0),
    ("frac_int_matched",            pl.Float32,             0.0),
    ("frac_int_pred",               pl.Float32,             0.0),
    ("spec_r2",                     pl.Float32,             0.0),
    ("prec_r2",                     pl.Float32,             0.0),
    ("prec_r2_uniq",                pl.Float32,             0.0),
    ("frac_int_uniq",               pl.Float32,             0.0),
    ("frac_int_uniq_pred",          pl.Float32,             0.0),
    ("hyperscore",                  pl.Float32,             0.0),
    ("b_counts",                    pl.Float32,             0.0),
    ("y_counts",                    pl.Float32,             0.0),
    ("longest_y_ions",              pl.Float32,             0.0),
    ("scribe_scores",               pl.Float32,             0.0),
    ("max_unmatched_residuals",     pl.Float32,             0.0),
    ("max_matched_residuals",       pl.Float32,             0.0),
    ("gof_stats",                   pl.Float32,             0.0),
    ("manhattan_distances",         pl.Float32,             0.0),
    ("fitted_spectral_contrasts",   pl.Float32,             0.0),
    ("frac_int_matched_pred",       pl.Float32,             0.0),
    ("frac_int_matched_pred_sigcoeff", pl.Float32,          0.0),
    ("cosine",                      pl.Float32,             0.0),
    ("mz",                          pl.Float32,             0.0),
    ("tic",                         pl.Float32,             0.0),
    # observed prec_im minus the aligned library IM; NaN on non-IM runs, as prec_im is
    ("im_error",                    pl.Float32,             0.0),
    # theta / sqrt(D1 - 1) over monoisotopic library fragments; lower is better
    ("ent_norm_angle",              pl.Float32,             0.0),
    # the same, against each matched peak's intensity split by responsibility
    # (the candidate's share of the fitted peak); lower is better
    ("resp_ent_norm_angle",         pl.Float32,             0.0),
    # 7 fragment list columns
    ("frag_names",                  pl.List(pl.Int32),       None),
    ("frag_errors",                 pl.List(pl.Float64),     None),
    ("frag_mz",                     pl.List(pl.Float64),     None),
    ("frag_int",                    pl.List(pl.Float64),     None),
    ("obs_int",                     pl.List(pl.Float64),     None),
    ("unique_frag_mz",             pl.List(pl.Float64),     None),
    ("unique_obs_int",             pl.List(pl.Float64),     None),
    # 2 string metadata columns
    ("file_name",                   pl.Utf8,                ""),
    ("protein",                     pl.Utf8,                ""),
    # target/decoy flag
    ("is_decoy",                    pl.Boolean,             False),
]


def get_parquet_schema(timeplex=False):
    """Return an ordered dict of {col_name: pl_dtype} for parquet output."""
    schema = {name: dtype for name, dtype, _ in _SCHEMA_SPEC}
    if timeplex:
        items = list(schema.items())
        items.insert(5, ("time_channel", pl.Float32))
        schema = dict(items)
    return schema



def find_extrema_in_nearby_scans(df, column_names, find_max_list, n_scans=3, anchor_col="apex_pc1"):
    """
    Polars equivalent of find_extrema_in_nearby_scans.

    For each precursor, finds the minimum or maximum value of a specified column from
    nearby scans (N scans before and N scans after by retention time) relative to
    the apex scan: the scan with the highest *anchor_col* (the apex PC1 the
    collapse keeps, so the nearby window and the reported apex are the same scan).

    Parameters
    ----------
    df : pl.DataFrame
        The dataframe containing PSM data.
    column_names : list of str
        The columns for which to find the extreme value from nearby scans.
    find_max_list : list of bool
        If True, find the maximum value; if False, find the minimum value.
    n_scans : int, default=3
        The number of scans before and after to consider.
    anchor_col : str, default="apex_pc1"
        The column whose per-precursor maximum is the window's centre.

    Returns
    -------
    pl.DataFrame
        The original dataframe with new '{column_name}_nearby_{max|min}' columns.
    """

    group_cols = ['seq', 'z'] + (['time_channel'] if 'time_channel' in df.collect_schema().names() else [])

    # TODO: bug -- "nearby scans" are not adjacent scans.  rt_rank below ranks only
    # this precursor's own fitted rows, so:
    #   - gaps are invisible: if the precursor was not fitted for several cycles,
    #     the next fitted row 10+ s away still counts as the neighbouring scan;
    #   - IM bands count as scans: overlapping IM bands of one frame are separate
    #     rows with the same rt, so the +/-n_scans window can be 2-3 bands of a
    #     single cycle instead of the cycles before and after.
    # The window should be on the run's cycle grid (best band per cycle, or every
    # band of each cycle), with cycles where the precursor was not fitted counted
    # as missing, not skipped.  Fixing it changes every *_nearby_* scoring
    # feature, so it needs the entrapment validation.

    # 1. Use Expression to calculate n_scans (group length)
    df = df.with_columns(
        pl.len().over(group_cols).alias("n_scans")
    )

    # 2. FIX: Get the 0-indexed rank as an integer (subtracting 1)
    #
    # Ties are broken on spec_id, not on row order.  Ordinal rank resolves ties by
    # position, and rows arrive in whatever order the parallel search batches
    # finished in, so an rt tie would rank differently between otherwise identical
    # runs.  With IM banding rt ties are common -- one frame produces several band
    # spectra sharing an rt -- which shifts the +/-n_scans window and changes every
    # *_nearby_* feature, and through them PredVal and Qvalue.  spec_id is unique
    # within a (seq, z) group, so it makes the ordering total and reproducible.
    df = df.with_columns(
        (pl.struct(["rt", "spec_id"]).rank(method="ordinal", descending=False).over(group_cols) - 1)
        .cast(pl.Int32)
        .alias("rt_rank")
    )

    # 3. Find the rank of the apex row (maximum anchor_col) within each group
    #    (ties broken on spec_id for the same reason)
    df = df.with_columns(
        pl.col("rt_rank")
        .sort_by(pl.struct([anchor_col, "spec_id"]), descending=True)
        .first()
        .over(group_cols)
        .alias("max_coeff_rank")
    )

    new_cols = []

    for column_name, find_max in zip(column_names, find_max_list):
        method = 'max' if find_max else 'min'
        nearby_col = f"{column_name}_nearby_{method}"

        # 4. Filter values to the n_scans window and calculate the extrema
        window_values_expr = (
            pl.when(
                (pl.col("rt_rank") >= (pl.col("max_coeff_rank") - n_scans)) &
                (pl.col("rt_rank") <= (pl.col("max_coeff_rank") + n_scans))
            )
            .then(pl.col(column_name))
            .otherwise(None)
        )

        extrema_expr = (
            window_values_expr
            .pipe(lambda expr: getattr(expr, method)().over(group_cols))
            .alias(nearby_col)
        )
        new_cols.append(extrema_expr)

    return df.with_columns(new_cols).drop(["rt_rank", "max_coeff_rank"])


def calculate_peak_smoothness(df, value_column='coeff', rt_column='rt', group_columns=None):
    """
    Calculates the smoothness (integrated squared second derivative) of chromatographic
    peaks for each group in the dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing chromatographic peak data
    value_column : str, default='coeff'
        The column containing intensity values to measure smoothness
    rt_column : str, default='rt'
        The column containing retention time values
    group_columns : list of str, default=None
        The columns to group by. If None, uses ['seq', 'z', 'time_channel']
        if 'time_channel' exists, otherwise ['seq', 'z']

    Returns
    -------
    pandas.DataFrame
        The original dataframe with an added 'smoothness' column containing
        the calculated smoothness value for each group
    """
    # 1. Determine grouping columns
    if group_columns is None:
        group_cols = ['seq', 'z']
        if 'time_channel' in df.collect_schema().names():
            group_cols.append('time_channel')
    else:
        group_cols = group_columns

    # Use a small constant to prevent division by zero, matching 1e-6 used in pandas function
    EPSILON = 1e-6

    # 2. Find the Apex Weight (denominator for normalization)
    # The Polars expression calculates the maximum 'value_column' within each group
    df = df.with_columns(
        pl.col(value_column).max().over(group_cols).alias("apex_weight")
    )

    # 3. Sort and calculate shift (lag/lead) values
    # To calculate the derivative, we need the values and RTs of the surrounding points.

    # The calculation needs values relative to the current row, sorted by RT:
    #   weights[i-1], weights[i+1], rts[i-1], rts[i+1]
    #
    # Sorted by row number: the main search writes its rows in spectrum order,
    # which is RT order (and spec_id order), and each spectrum's candidates in a
    # fixed order, so this is RT order with every tie broken the same way on every
    # run.  (rt alone would leave the band spectra sharing an rt, and the
    # candidates sharing a spectrum, in an arbitrary order.)
    df_with_shifts = df.sort("__parquet_row").with_columns(
        # Weights (Value Column) shifts
        pl.col(value_column).shift(1).over(group_cols).alias("w_prev"),
        pl.col(value_column).shift(-1).over(group_cols).alias("w_next"),
        # Retention Time shifts
        pl.col(rt_column).shift(1).over(group_cols).alias("rt_prev"),
        pl.col(rt_column).shift(-1).over(group_cols).alias("rt_next"),
    )

    # 4. Calculate Squared Derivative Term for each row
    df_smoothness_terms = df_with_shifts.with_columns(
        # Calculate the four potential derivative components (terms)

        # Term 1: (weights[i+1] - weights[i]) / (rts[i+1] - rts[i] + 1e-6)
        term_next=(pl.col("w_next") - pl.col(value_column)) / (pl.col("rt_next") - pl.col(rt_column) + EPSILON),

        # Term 2: (weights[i-1] - weights[i]) / (rts[i] - rts[i-1] + 1e-6)
        term_prev=(pl.col("w_prev") - pl.col(value_column)) / (pl.col(rt_column) - pl.col("rt_prev") + EPSILON),

        # Term 3 (Boundary Case i=0): (-weights[i]) / (rts[i+1] - rts[i] + 1e-6)
        term_zero_next=(-pl.col(value_column)) / (pl.col("rt_next") - pl.col(rt_column) + EPSILON),

        # Term 4 (Boundary Case i=len-1): (-weights[i]) / (rts[i] - rts[i-1] + 1e-6)
        term_last_prev=(-pl.col(value_column)) / (pl.col(rt_column) - pl.col("rt_prev") + EPSILON),
    )

    # 5. Combine terms based on position (Boundary Logic)
    # The derivative (deriv) is the sum of two terms:
    # Interior (i > 0 and i < len-1): deriv = Term_next + Term_prev
    # First (i = 0): deriv = Term_next + Term_zero_next
    # Last (i = len-1): deriv = Term_last_prev + Term_prev (note: the original code had a slight error here, using term_next instead of term_prev, but we follow the logic implied by the index usage)

    # We use pl.when/then to precisely replicate the original function's boundary logic:
    df_final = df_smoothness_terms.with_columns(
        deriv=pl.when(pl.col("w_prev").is_null())  # i = 0 (First point)
        .then(pl.col("term_next") + pl.col("term_zero_next"))
        .when(pl.col("w_next").is_null())  # i = len-1 (Last point)
        .then(pl.col("term_prev") + pl.col("term_last_prev"))
        .otherwise(pl.col("term_next") + pl.col("term_prev"))  # Interior points
    ).with_columns(
        # Normalize and Square the derivative
        squared_deriv=(pl.col("deriv") / pl.col("apex_weight")).pow(2)
    )

    # 6. Sum the squared derivatives within each group (smoothness)
    # Handle the group length <= 1 case and apex_weight == 0 case

    # Group length computed here rather than read from the n_scans column, which
    # find_extrema_in_nearby_scans adds and now runs after this (it needs apex_pc1).
    smoothness_expr = (pl.when(pl.len().over(group_cols) <= 1)
            .then(0.0)  # Length <= 1
        .when(pl.col("apex_weight") == 0)
            .then(0.0)  # Apex weight is 0
        .otherwise(
            pl.col("squared_deriv").sum().over(group_cols)  # Normal case: sum the squared derivatives
        )
    )

    # Final logic for single point (weights=1) case:
    # The single point case (smoothness = (-2 * weights[0] / apex_weight) ** 2) is equivalent
    # to the sum of squared_deriv when n_scans=1, because the boundary logic reduces to this.
    # We use the n_scans filter above to set it to 0.0, as done in the original code.

    # 7. Add the final 'smoothness' column and select the original columns plus 'smoothness'
    return df_final.with_columns(smoothness=smoothness_expr).drop(
        ["apex_weight", "w_prev", "w_next", "rt_prev", "rt_next", "term_next",
         "term_prev", "term_zero_next", "term_last_prev", "deriv", "squared_deriv"]
    )


# Rows per precursor partition of the main-search output.  Apex selection holds
# one partition's rows at a time, so this bounds its memory: ~3M rows of the
# scalar columns is a few GB.
APEX_PART_ROWS = 3_000_000


def log_mem(stage):
    """Log a pipeline stage with the process's current and peak memory."""
    import psutil
    mi = psutil.Process().memory_info()
    peak = getattr(mi, "peak_wset", None)                      # Windows
    if peak is None:
        import resource
        import sys
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform != "darwin":
            peak *= 1024                                       # Linux reports KiB
    logger.info(f"{stage} [memory {mi.rss / 1e9:.1f} GB, peak {peak / 1e9:.1f} GB]")


# The main search's per-fragment list columns: most of its bytes, and not needed
# until the kept rows (one per precursor) are known.
LIST_COLS = ["frag_names", "frag_errors", "frag_mz", "frag_int",
             "obs_int", "unique_frag_mz", "unique_obs_int"]


# TODO: give an informative error when reading these batch parquets dies with
# SIGBUS (polars memory-maps them).  On a .raw run Mono's signal handler catches
# it and prints a "Native Crash Reporting" dump that points at Mono, not at the
# real cause: the file shrank or changed while mapped (another run writing the
# same output folder), the output folder is on an external/network/cloud-synced
# drive that dropped, the disk is full, or the parquet is corrupt.  Check those
# (free space, readable footers) before the read and say which file and why.
# Also let the user designate a temp folder for these intermediates, so they can
# sit on a local disk even when the output folder is on external/network storage.
class MainSearchFiles:
    """The main search's batch parquets, read as one table in order.

    ``scan(i)`` is batch i with each row's global row number (``__parquet_row``, its
    index in the batches concatenated in order, what the former merged file's row
    index was), so per-file reads keep the global order.
    """

    def __init__(self, files):
        import pyarrow.parquet as pq
        self.files = list(files)
        counts = [pq.ParquetFile(f).metadata.num_rows for f in self.files]
        self.offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
        self.n_rows = int(self.offsets[-1])

    def scan(self, i):
        return pl.scan_parquet(self.files[i]).with_row_index("__parquet_row", offset=int(self.offsets[i]))

    def scan_all(self):
        return pl.scan_parquet(self.files).with_row_index("__parquet_row")


class MainSearchRows:
    """The main search's per-scan rows for MS1 quant, read for given precursors only.

    ``frame_for`` returns the ``cols`` of every scan of the precursors in ``rows``,
    sorted by ``keys`` then original row order: exactly those groups of the former
    all-rows frame.  Read one batch file at a time.
    """

    def __init__(self, ms, cols, keys):
        self.ms, self.cols, self.keys = ms, cols, keys

    def frame_for(self, rows):
        schema = self.ms.scan(0).collect_schema()
        want = pl.from_pandas(rows[self.keys].drop_duplicates().reset_index(drop=True))
        want = want.with_columns([pl.col(k).cast(schema[k]) for k in self.keys]).lazy()
        out = [self.ms.scan(i).select(self.cols + ["__parquet_row"]).join(want, on=self.keys, how="semi").collect()
               for i in range(len(self.ms.files))]
        frame = pl.concat(out).sort(self.keys + ["__parquet_row"]).drop("__parquet_row")
        log_mem(f"MS2 rows for MS1 quant: {frame.height:,} scans of {rows[self.keys].drop_duplicates().shape[0]:,} precursors")
        return frame


def get_large_prec(files,
                   rt_tol,
                   timeplex=False,
                   rt_sigma=None,
                   im_scale=None):
    """
    Select each precursor's apex scan from the main-search output and build the
    one-row-per-precursor frame, using Polars, converting to Pandas for external
    function calls.

    ``files`` is the main search's batch parquets, read as one table in order.  Every step is per precursor except the apex PC1 fit, which
    uses a hash sample of precursors.  The main search admits a precursor to a
    spectrum only within ``rt_tol`` of its aligned RT, so a precursor's rows span
    less than 2 * rt_tol: the run is processed in RT chunks, each reading only the
    rows within 2 * rt_tol of it, and a precursor belongs to the chunk holding its
    earliest row.  Only the kept rows (one per precursor) are ever held together;
    their order is restored to what one global pass gave: stable sort by
    apex_score over the frame in row order, keep the last row per
    precursor.

    Returns ``large_prec``, the kept-row frame, a ``MainSearchRows`` for MS1 quant
    and the kept rows' list columns.
    """
    # Only scalar columns go through the lazy chain, so the polars sort buffer
    # holds no list columns. List cols are re-fetched for the kept rows only, by
    # joining on the row number.
    ms = MainSearchFiles(files)

    # Rank each precursor's scans by how apex-like they are, replacing max-coeff.
    # The group key must match the unique() subset below, or the collapse would pick
    # an apex from a different partition than the one it keeps.
    apex_group_cols = ["seq", "z"] + (["time_channel"] if timeplex else [])

    # RT chunks of ~APEX_PART_ROWS rows, edges at RT quantiles of the rows
    rt_all = np.concatenate([ms.scan(i).select("rt").collect().to_series().to_numpy()
                             for i in range(len(ms.files))])
    file_rt = [(float(np.min(rt_all[ms.offsets[i]:ms.offsets[i + 1]])),
                float(np.max(rt_all[ms.offsets[i]:ms.offsets[i + 1]])))
               if ms.offsets[i + 1] > ms.offsets[i] else (np.inf, -np.inf)
               for i in range(len(ms.files))]
    n_chunks = max(1, -(-ms.n_rows // APEX_PART_ROWS))
    edges = np.unique(np.quantile(rt_all, np.linspace(0, 1, n_chunks + 1)[1:-1])) if n_chunks > 1 else np.array([])
    edges = np.concatenate(([-np.inf], edges, [np.inf]))
    del rt_all
    # A precursor's rows span < 2 * rt_tol (the main search's strict |lib - spec| <
    # rt_tol); the small absolute slack covers float32 rounding of the stored RT.
    span_max = 2.0 * float(rt_tol) + 1e-3
    margin = span_max + 1e-3

    log_mem(f"Apex selection: {ms.n_rows:,} rows in {len(edges) - 1} RT chunk(s); fitting apex PC1")
    loadings = quality_pca.fit_main_apex_pc1(
        ms.scan_all().drop(LIST_COLS), apex_group_cols, seed=config.RANDOM_SEED)

    def _chunk_lf(lo, hi):
        wlo, whi = lo - margin, hi + margin
        sel = [i for i, (a, b) in enumerate(file_rt) if b >= wlo and a <= whi]
        if not sel:
            return None
        lf = pl.concat([ms.scan(i).drop(LIST_COLS) for i in sel], how="vertical")
        if np.isfinite(wlo) or np.isfinite(whi):
            lf = lf.filter((pl.col("rt") >= wlo) & (pl.col("rt") <= whi))
        return (lf.with_columns(pl.col("rt").min().over(apex_group_cols).alias("__rt_min"),
                                pl.col("rt").max().over(apex_group_cols).alias("__rt_max"))
                  .filter((pl.col("__rt_min") >= lo) & (pl.col("__rt_min") < hi))
                  .with_columns((pl.col("__rt_max") - pl.col("__rt_min")).alias("__rt_span"))
                  .drop(["__rt_min", "__rt_max"]))

    kept_parts = []
    for i in range(len(edges) - 1):
        part_lf = _chunk_lf(edges[i], edges[i + 1])
        if part_lf is None:
            continue
        decoy_coeffs_lf = calculate_peak_smoothness(
            df=part_lf,
            value_column='coeff',
            rt_column='rt',
            group_columns=None
        )
        decoy_coeffs_lf = decoy_coeffs_lf.with_columns(
            quality_pca.main_apex_pc1_expr(loadings, apex_group_cols).alias("apex_pc1"))
        # Apex score: the spectral PC1 plus log priors for agreement with the predicted
        # RT (Gaussian, the RT tolerance's core sigma) and IM (Laplace, the IM accuracy
        # fit's core scale).  PC1 alone has no RT/IM term, and putting rt_error into the
        # PCA does not add one: it varies independently of the spectral features, so
        # PC1 gives it a loading of ~0.  Decoys inherit their target's RT and IM, so the
        # priors treat both alike.  NaN errors (no IM data) contribute 0.
        _rt_lp = (pl.lit(0.0) if not rt_sigma else
                  (-0.5 * (pl.col("rt_error").cast(pl.Float64) / float(rt_sigma)) ** 2).fill_nan(0.0).fill_null(0.0))
        _im_lp = (pl.lit(0.0) if not im_scale else
                  (-(pl.col("im_error").cast(pl.Float64).abs()) / float(im_scale)).fill_nan(0.0).fill_null(0.0))
        decoy_coeffs_lf = decoy_coeffs_lf.with_columns(
            _rt_lp.alias("rt_logprior"), _im_lp.alias("im_logprior"))
        decoy_coeffs_lf = decoy_coeffs_lf.with_columns(
            (pl.col("apex_pc1") + pl.col("rt_logprior") + pl.col("im_logprior")).alias("apex_score"))
        # What the priors overrode, as scoring features.  The spectral apex (max PC1)
        # and the chosen apex (max apex_score) come from the same PC1 values: for a
        # real peptide they usually agree; a null's best spectral evidence is anywhere
        # in its RT window.  Moving the apex toward the prediction would otherwise hide
        # that from rt_error / im_error at the chosen apex.  Same for targets and decoys.
        _spec = pl.struct(["apex_pc1", "spec_id"])
        _chosen = pl.struct(["apex_score", "spec_id"])
        decoy_coeffs_lf = decoy_coeffs_lf.with_columns(
            pl.col("rt_error").cast(pl.Float64).abs().fill_nan(0.0).fill_null(0.0)
            .sort_by(_spec, descending=True).first().over(apex_group_cols)
            .alias("spec_apex_rt_error"),
            pl.col("im_error").cast(pl.Float64).abs().fill_nan(0.0).fill_null(0.0)
            .sort_by(_spec, descending=True).first().over(apex_group_cols)
            .alias("spec_apex_im_error"),
            (pl.col("apex_pc1").max().over(apex_group_cols)
             - pl.col("apex_pc1").sort_by(_chosen, descending=True).first().over(apex_group_cols))
            .alias("apex_pc1_cost"),
        )
        # Nearby-scan extrema around the apex PC1 scan, the scan the collapse below
        # keeps; not around the max-coefficient scan, which is a different scan for
        # about a third of precursors.
        decoy_coeffs_lf = find_extrema_in_nearby_scans(
            decoy_coeffs_lf,
            ["manhattan_distances", "max_matched_residuals", "gof_stats", "scribe_scores",
             # TODO: evaluate whether the adjacent-scan minimum of resp_ent_norm_angle
             # helps, on single-channel and multiplexed data.
             "ent_norm_angle", "resp_ent_norm_angle"],
            [True, False, False, False, False, False],
            n_scans=2, anchor_col="apex_score")

        # maintain_order: the default sort is unstable, so apex_pc1 ties would feed
        # the unique() below in a different order per run.  apex_score and the priors
        # stay until the partitions are combined (they order the kept rows).
        sorted_decoy_coeffs_lf = decoy_coeffs_lf.sort(by="apex_score", maintain_order=True)
        if timeplex:
            # Calculate lib_rt and add column
            sorted_decoy_coeffs_lf = sorted_decoy_coeffs_lf.with_columns(
                (pl.col("rt") - pl.col("rt_error")).alias("lib_rt")
            )
        # Polars equivalent of drop_duplicates(..., keep='last'): the highest apex_score
        kept = (sorted_decoy_coeffs_lf
                .unique(subset=apex_group_cols, keep="last", maintain_order=True)
                .filter(pl.col("seq") != "0")
                .collect())
        kept_parts.append(kept)
        if kept.height and kept["__rt_span"].max() >= span_max:
            from src.utils.errors import JModError
            raise JModError(f"A precursor's main-search rows span {kept['__rt_span'].max():.3f} in RT, "
                            f"more than 2 x rt_tol ({rt_tol}): apex selection by RT chunk would miss rows")
        kept_parts[-1] = kept.drop("__rt_span")
        if (i + 1) % 10 == 0 or i + 2 == len(edges):
            log_mem(f"Apex selection: RT chunk {i + 1} of {len(edges) - 1}")

    # The kept rows in the order a single global pass leaves them: by apex_score,
    # ties in row order, the frame's order after the smoothness sort.
    _fdc_polars = (pl.concat(kept_parts)
                   .sort(["apex_score", "__parquet_row"])
                   # selection only: kept out of the scoring features
                   .drop(["rt_logprior", "im_logprior", "apex_score"]))
    del kept_parts

    # Create large_prec dictionary
    large_prec_rows = _fdc_polars.filter(pl.col("coeff") > 1)
    if timeplex:
        large_prec = {(i, j, l): k for i, j, k, l in
                      large_prec_rows.select(["seq", "z", "coeff", "time_channel"]).iter_rows()}
    else:
        large_prec = {(i, j): k for i, j, k in large_prec_rows.select(["seq", "z", "coeff"]).iter_rows()}
    del large_prec_rows

    _fdc_polars = _fdc_polars.with_row_index("__fdc_idx")

    # Re-fetch list cols for the kept rows only, joined on the row number.
    _row_keys = _fdc_polars.select(["__parquet_row", "__fdc_idx"]).lazy()
    fdc_list_cols = (pl.concat([ms.scan(i).select(["__parquet_row"] + LIST_COLS)
                                  .join(_row_keys, on="__parquet_row", how="inner")
                                  .collect()
                                for i in range(len(ms.files))])
                     .drop("__parquet_row")
                     .sort("__fdc_idx"))

    filtered_decoy_coeffs = _fdc_polars.drop("__parquet_row").to_pandas()
    del _fdc_polars
    log_mem(f"Apex selection done: {len(filtered_decoy_coeffs):,} precursors")

    # The scans of each precursor for MS1 quant (ms1_cor_channels), which reads a
    # handful of numeric columns and groups by (seq, z[, time_channel]).  Not read
    # here: MS1 quant asks for its precursors only (MainSearchRows.frame_for).
    dc_cols = ["seq", "z", "coeff", "rt", "Ms1_spec_id", "spec_id"]
    dc_sort_keys = ["seq", "z"]
    if timeplex:
        dc_cols.append("time_channel")
        dc_sort_keys.append("time_channel")
    decoy_coeffs = MainSearchRows(ms, dc_cols, dc_sort_keys)

    return large_prec, filtered_decoy_coeffs, decoy_coeffs, fdc_list_cols


def read_results(file,
                   timeplex=False):

    col_names = list(names)
    if timeplex:
        col_names.insert(5,"time_channel")
        dtypes["time_channel"] = np.float32 ## !!! need to fix
    # logger.info(col_names)
    decoy_coeffs = pd.read_csv(file,header=None,names=col_names,dtype=dtypes)

    results_folder = os.path.dirname(file)

    all_fdx = pd.read_parquet(results_folder+"/all_precursors.parquet")
    filtered_fdx = pd.read_parquet(os.path.join(os.path.dirname(results_folder), "filtered_precursors.parquet"))

    return decoy_coeffs,all_fdx,filtered_fdx



def frag_traces(g):
    traces = {}
    for idx,row in g.iterrows():
        spec_id = row.spec_id
        frag_names = row.frag_names.split(";")
        frag_mz = np.array(row.frag_mz.split(";"),dtype=np.float32)
        frag_int = np.array(row.frag_int.split(";"),dtype=np.float32)
        obs_int = np.array(row.obs_int.split(";"),dtype=np.float32)

        for i,f in enumerate(frag_names):
            traces.setdefault(f,[])
            traces[f].append([spec_id,frag_mz[i],obs_int[i]])