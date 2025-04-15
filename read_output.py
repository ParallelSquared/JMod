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
         "protein"
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
            "protein":str
            }


def get_large_prec(file,
                   condense_output=True,
                   timeplex=False):
    
    col_names = list(names)
    if timeplex:
        col_names.insert(5,"time_channel")
        dtypes["time_channel"] = np.float32 ## !!! need to fix 
    # print(col_names)
    decoy_coeffs = pd.read_csv(file,header=None,names=col_names,dtype=dtypes)
    
    
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
        
        
    