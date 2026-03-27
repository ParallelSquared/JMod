"""
This Source Code Form is subject to the terms of the Oxford Nanopore
Technologies, Ltd. Public License, v. 1.0.  Full licence can be found
at https://github.com/ParallelSquared/JMod/blob/main/LICENSE.txt
"""

import polars as pl # TODO Requires pyarrow
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
import os


names = ["coeff", "spec_id", "Ms1_spec_id",
         "seq", "z", "window_mz", "rt",
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
    # 27 feature columns
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
]


def get_parquet_schema(timeplex=False):
    """Return an ordered dict of {col_name: pl_dtype} for parquet output."""
    schema = {name: dtype for name, dtype, _ in _SCHEMA_SPEC}
    if timeplex:
        items = list(schema.items())
        items.insert(5, ("time_channel", pl.Float32))
        schema = dict(items)
    return schema



def find_extrema_in_nearby_scans(df, column_names, find_max_list, n_scans=3):
    """
    Polars equivalent of find_extrema_in_nearby_scans.

    For each precursor, finds the minimum or maximum value of a specified column from
    nearby scans (N scans before and N scans after by retention time) relative to
    the scan with the highest coefficient.

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

    Returns
    -------
    pl.DataFrame
        The original dataframe with new '{column_name}_nearby_{max|min}' columns.
    """

    group_cols = ['seq', 'z'] + (['time_channel'] if 'time_channel' in df.collect_schema().names() else [])

    # 1. Use Expression to calculate n_scans (group length)
    df = df.with_columns(
        pl.len().over(group_cols).alias("n_scans")
    )

    # 2. FIX: Get the 0-indexed rank as an integer (subtracting 1)
    df = df.with_columns(
        (pl.col("rt").rank(method="ordinal", descending=False).over(group_cols) - 1)
        .cast(pl.Int32)
        .alias("rt_rank")
    )

    # 3. Find the rank of the row with the maximum 'coeff' within each group
    df = df.with_columns(
        pl.col("rt_rank")
        .sort_by(pl.col("coeff"), descending=True)
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
    df_with_shifts = df.sort(rt_column).with_columns(
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

    smoothness_expr = (pl.when(pl.col("n_scans") <= 1)
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


# Mapping from NumPy dtypes to Polars dtypes
NP_TO_PL_DTYPES = {
    np.float32: pl.Float32,
    np.int32: pl.Int32,
    str: pl.Utf8,
    # Add other mappings if needed, e.g., np.float64: pl.Float64, np.int64: pl.Int64
}


def numpy_dtypes_to_polars_schema(np_dtypes_dict):
    """Converts a dictionary of column_name: np_dtype to column_name: pl_dtype."""
    pl_schema = {}
    for col, np_dtype in np_dtypes_dict.items():
        pl_schema[col] = NP_TO_PL_DTYPES.get(np_dtype, pl.Utf8)  # Default to Utf8 for safety
    return pl_schema


# --- Converted Polars Function ---

def get_large_prec(file,
                   condense_output=True,
                   timeplex=False):
    """
    Process peptide identification results to extract high-confidence precursors
    using Polars, converting to Pandas for external function calls.
    ... (docstring continues as before) ...
    """
    col_names = list(names)

    # 1. Convert NumPy dtypes to Polars schema for pl.read_csv
    pl_schema = numpy_dtypes_to_polars_schema(dtypes)

    if timeplex:
        col_names.insert(5, "time_channel")
        # Ensure time_channel is added to the Polars schema
        pl_schema["time_channel"] = pl.Float32

    # Explicitly fix string literal dtypes to override type inference
    string_literal_cols = [
        "frag_names", "frag_errors", "frag_mz", "frag_int",
        "obs_int", "unique_frag_mz", "unique_obs_int"
    ]

    decoy_coeffs_lf = pl.scan_parquet(file)

    decoy_coeffs_lf = find_extrema_in_nearby_scans(
        decoy_coeffs_lf,
        ["manhattan_distances", "max_matched_residuals", "gof_stats", "scribe_scores"],
        [True, False, False, False],
        n_scans=2)

    # Convert to Pandas for external function calls
    decoy_coeffs_lf = calculate_peak_smoothness(
        df=decoy_coeffs_lf,
        value_column='coeff',
        rt_column='rt',
        group_columns=None
    )

    # Polars equivalent of .sort_values(by="coeff")
    sorted_decoy_coeffs_lf = decoy_coeffs_lf.sort(by="coeff")

    # Polars equivalent of drop_duplicates(..., keep='last')
    if timeplex:
        # Calculate lib_rt and add column
        sorted_decoy_coeffs_lf = sorted_decoy_coeffs_lf.with_columns(
            (pl.col("rt") - pl.col("rt_error")).alias("lib_rt")
        )

        # Drop duplicates by "seq", "z", "time_channel" keeping the last (highest 'coeff')
        filtered_decoy_coeffs_lf = sorted_decoy_coeffs_lf.unique(
            subset=["seq", "z", "time_channel"],
            keep="last",
            maintain_order=True
        )

        # Filter out rows where seq is "0"
        filtered_decoy_coeffs_lf = filtered_decoy_coeffs_lf.filter(pl.col("seq") != "0")

        # Create large_prec dictionary
        large_prec_lf = filtered_decoy_coeffs_lf.filter(pl.col("coeff") > 1).select(
            ["seq", "z", "coeff", "time_channel"]
        )

        # Polars to native Python for dictionary creation
        large_prec = {
            (i, j, l): k
            for i, j, k, l in large_prec_lf.collect().iter_rows()
        }

    else:
        # Drop duplicates by "seq", "z" keeping the last (highest 'coeff')
        filtered_decoy_coeffs_lf = sorted_decoy_coeffs_lf.unique(
            subset=["seq", "z"],
            keep="last",
            maintain_order=True
        )

        # Filter out rows where seq is "0"
        filtered_decoy_coeffs_lf = filtered_decoy_coeffs_lf.filter(pl.col("seq") != "0")

        # Create large_prec dictionary
        large_prec_lf = filtered_decoy_coeffs_lf.filter(pl.col("coeff") > 1).select(
            ["seq", "z", "coeff"]
        )

        # Polars to native Python for dictionary creation
        large_prec = {
            (i, j): k
            for i, j, k in large_prec_lf.collect().iter_rows()
        }

    # Collect lazy frames
    filtered_decoy_coeffs = filtered_decoy_coeffs_lf.collect().to_pandas()
    decoy_coeffs = decoy_coeffs_lf.collect().to_pandas()

    if condense_output:
        return large_prec, filtered_decoy_coeffs
    else:
        return large_prec, filtered_decoy_coeffs, decoy_coeffs



def read_results(file,
                   timeplex=False):

    col_names = list(names)
    if timeplex:
        col_names.insert(5,"time_channel")
        dtypes["time_channel"] = np.float32 ## !!! need to fix
    # logger.info(col_names)
    decoy_coeffs = pd.read_csv(file,header=None,names=col_names,dtype=dtypes)

    results_folder = os.path.dirname(file)

    all_fdx = pd.read_csv(results_folder+"/all_IDs.csv")
    filtered_fdx = pd.read_csv(results_folder+"/filtered_IDs.csv")

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