"""
This Source Code Form is subject to the terms of the Oxford Nanopore
Technologies, Ltd. Public License, v. 1.0.  Full licence can be found
at https://github.com/ParallelSquared/JMod/blob/main/LICENSE.txt
"""

import pandas as pd
import numpy as np
import os


names = ["coeff","spec_id","Ms1_spec_id",
         "seq","z","window_mz","rt",
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
         "frag_names",
         "frag_errors",
         "frag_mz",
         "frag_int",
         "obs_int",
         "unique_frag_mz",
         "unique_obs_int",
         "file_name",
         "protein",
         "manhattan_distances_nearby_max",
         "max_matched_residuals_nearby_min",
         "gof_stats_nearby_min",
         "median_gof_stats"
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
            "frag_names":str,
            "frag_errors":str,
            "frag_mz":str,
            "frag_int":str,
            "obs_int":str,
            "file_name":str,
            "protein":str,
            "manhattan_distances_nearby_max":np.float32,
            "max_matched_residuals_nearby_min":np.float32,
            "gof_stats_nearby_min":np.float32,
            "median_gof_stats":np.float32
            }

def find_extreme_in_nearby_scans(df, column_name, n_scans=3, find_max=True):
    """
    For each precursor, finds the minimum or maximum value of a specified column from
    nearby scans (N scans before and N scans after by retention time) for
    the scan with the highest coefficient.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing PSM data with columns 'seq', 'z', 'rt', 'coeff',
        and the specified column_name.
    column_name : str
        The column for which to find the extreme value from nearby scans.
    n_scans : int, default=3
        The number of scans before and after to consider.
    find_max : bool, default=True
        If True, find the maximum value; if False, find the minimum value.

    Returns
    -------
    pandas.DataFrame
        The original dataframe with a new column '{column_name}_nearby_max' or
        '{column_name}_nearby_min' containing the extreme value of the specified 
        column from nearby scans for each precursor's best match.
    """
    # Create a copy of the dataframe
    result_df = df.copy()

    # Create a new column for the nearby extreme values
    operation_type = "max" if find_max else "min"
    nearby_col = f"{column_name}_nearby_{operation_type}"
    result_df[nearby_col] = np.zeros_like(result_df[column_name])

    # Define grouping columns based on whether time_channel is present
    group_cols = ['seq', 'z', 'time_channel'] if 'time_channel' in df.columns else ['seq', 'z']

    # For each unique precursor
    for _, group in df.groupby(group_cols):
        # Check if the column exists in the dataframe
        if column_name not in group.columns:
            continue
            
        # Find the index of the row with the highest coefficient
        max_coeff_idx = group['coeff'].idxmax()
        
        # Sort the group by retention time
        sorted_group = group.sort_values('rt')
        
        # Find the position of the max coefficient scan in the sorted list
        try:
            pos_in_sorted = sorted_group.index.get_loc(max_coeff_idx)
        except KeyError:
            # If the index is not found (should not happen), skip this group
            continue
        
        # Get indices of scans before and after the max coefficient scan
        start_pos = max(0, pos_in_sorted - n_scans)
        end_pos = min(len(sorted_group) - 1, pos_in_sorted + n_scans)
        nearby_indices = sorted_group.index[start_pos:end_pos+1]
        #print("nearby_indices ", nearby_indices ", \n")
        # Find the extreme value of the specified column in the nearby scans
        extreme_val = (
            group.loc[nearby_indices, column_name].max() if find_max else 
            group.loc[nearby_indices, column_name].min()
        )
        # Update all rows in the original dataframe for this group
        group_indices = group.index
        result_df.loc[group_indices, nearby_col] = extreme_val

        # Assign this extreme value to the row with the highest coefficient
        #result_df.loc[max_coeff_idx, nearby_col] = extreme_val Can I get rid of this line now?

    return result_df
    
def get_large_prec(file,
                   condense_output=True,
                   timeplex=False):
    """
    Process peptide identification results to extract high-confidence precursors.
    
    This function reads a CSV file containing peptide identification data, filters 
    to keep only the entry with the highest coefficient for each unique peptide 
    sequence and charge state combination, and creates a dictionary of "large 
    precursors" (those with coefficients > 1).
    
    Parameters
    ----------
    file : str
        Path to the CSV file containing peptide identification results.
    condense_output : bool, default=True
        If True, return only the large precursor dictionary and filtered dataframe.
        If False, also include the original dataframe.
    timeplex : bool, default=False
        If True, consider time_channel as part of the uniqueness criteria and 
        calculate library retention time (lib_rt).
    
    Returns
    -------
    large_prec : dict
        Dictionary where keys are tuples of (sequence, charge) or 
        (sequence, charge, time_channel) and values are the corresponding 
        coefficients (only for coefficients > 1).
    filtered_decoy_coeffs : pandas.DataFrame
        Filtered dataframe containing only the highest coefficient entry for 
        each unique precursor.
    decoy_coeffs : pandas.DataFrame, optional
        Original unfiltered dataframe. Only returned if condense_output=False.
    
    Notes
    -----
    This function appears to be part of a proteomics workflow, likely for 
    peptide-spectrum match filtering and confidence assessment. The 'large precursors' 
    represent peptide identifications with high confidence scores.
    """
    col_names = list(names)
    if timeplex:
        col_names.insert(5,"time_channel")
        dtypes["time_channel"] = np.float32 ## !!! need to fix 
    # print(col_names)
    decoy_coeffs = pd.read_csv(file,header=None,names=col_names,dtype=dtypes)
    
    decoy_coeffs = find_extreme_in_nearby_scans(decoy_coeffs, "manhattan_distances", n_scans=2, find_max=True)
    decoy_coeffs = find_extreme_in_nearby_scans(decoy_coeffs, "max_matched_residuals", n_scans=2, find_max=False)
    decoy_coeffs = find_extreme_in_nearby_scans(decoy_coeffs, "gof_stats", n_scans=2, find_max=False)
    # get dataframe
    sorted_decoy_coeffs = decoy_coeffs.sort_values(by="coeff")
    
    # create dictionay, where value is index of largest coeff for each key (seq,z)
    
    # create new dataframe using only these indices
    if timeplex:
        # names.insert(5,"time_channel")
        # dtypes["time_channel"] = np.int32
        lib_rt = sorted_decoy_coeffs["rt"]-sorted_decoy_coeffs["rt_error"]
        sorted_decoy_coeffs["lib_rt"] = lib_rt
        filtered_decoy_coeffs = sorted_decoy_coeffs.drop_duplicates(["seq","z","time_channel"], keep='last')
        filtered_decoy_coeffs = filtered_decoy_coeffs.reset_index(drop=True)
        filtered_decoy_coeffs = filtered_decoy_coeffs.drop(np.where(filtered_decoy_coeffs.seq=="0")[0])
        filtered_decoy_coeffs = filtered_decoy_coeffs.reset_index(drop=True)
        large_prec = {(i,j,l):k for i,j,k,l in zip(filtered_decoy_coeffs.seq,
                                                   filtered_decoy_coeffs.z,
                                                   filtered_decoy_coeffs.coeff,
                                                   filtered_decoy_coeffs.time_channel) if k>1}
    else:
        filtered_decoy_coeffs = sorted_decoy_coeffs.drop_duplicates(["seq","z"], keep='last')
        filtered_decoy_coeffs = filtered_decoy_coeffs.reset_index(drop=True)
        filtered_decoy_coeffs = filtered_decoy_coeffs.drop(np.where(filtered_decoy_coeffs.seq=="0")[0])
        filtered_decoy_coeffs = filtered_decoy_coeffs.reset_index(drop=True)
    
        large_prec = {(i,j):k for i,j,k in zip(filtered_decoy_coeffs.seq,filtered_decoy_coeffs.z,filtered_decoy_coeffs.coeff) if k>1}

    if condense_output:
        return large_prec,filtered_decoy_coeffs
    else:
        return large_prec,filtered_decoy_coeffs, decoy_coeffs



def read_results(file,
                   timeplex=False):
    
    col_names = list(names)
    if timeplex:
        col_names.insert(5,"time_channel")
        dtypes["time_channel"] = np.float32 ## !!! need to fix 
    # print(col_names)
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
        
        
    