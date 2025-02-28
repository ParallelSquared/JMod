#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 09:38:40 2024

@author: kevinmcdonnell
"""


from read_output import get_large_prec

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve,auc 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from scipy import stats
import xgboost as xgb

import numpy as np
import matplotlib.pyplot as plt
import tqdm
import Jplot as jp
import re
import os
import pandas as pd
import seaborn as sns
import pickle

from trace_fns import ms1_cor, ms1_cor_channels
from load_files import loadSpectra
from SpecLib import loadSpecLib

from mass_tags import mTRAQ, mTRAQ_02468, mTRAQ_678, tag_library
import iso_functions as iso_f
from miscFunctions import fragment_cor

import config




# lp,fdc,dc = get_large_prec(file,condense_output=False,timeplex=bool(params["timeplex"]))

# all_lp.append(lp)
# all_prec_labels.append(lp)
                       
       

### check if this processing was already done
### If so load it
### if not create it

# ID_attributes_file  = "precursor_attributes.csv"
# ID_attributes_path = results_folder+"/"+ID_attributes_file 
# if os.path.exists(ID_attributes_path):
#     fdc = pd.read_csv(ID_attributes_path)
# else:

    
# def add_attributes(fdc):
     
#     ## Add additional features
#     # X["prec_z"] = fdc["z"]
#     fdc["pep_len"] = [len(re.findall("([A-Z](?:\(.*?\))?)",i.split("_")[-1])) for i in fdc["seq"]]
#     fdc["stripped_seq"] = np.array([re.sub("\(.*?\)","",i) for i in fdc["seq"]])
#     # X["rt"] = fdc["rt"]
#     # X["coeff"] = fdc["coeff"]
#     fdc["sq_rt_error"] = np.power(fdc["rt_error"],2)
#     fdc["sq_mz_error"] = np.power(fdc["mz_error"],2)
    
    
#     return fdc


def ms1_quant(fdc,lp,dc,mass_tag,DIAspectra,mz_ppm,rt_tol,timeplex=False):
    # X = fdc.iloc[:,6:-5]
   
    print("Performing MS1 Quantitation") 
    
    if timeplex:
        all_keys = [(i,j,k) for i,j,k in zip(fdc.seq,fdc.z,fdc.time_channel)]
    else:
        all_keys = [(i,j) for i,j in zip(fdc.seq,fdc.z)]
    
        
    if mass_tag:
        fdc["untag_seq"] = [re.sub(f"(\({mass_tag.name}-\d+\))?","",peptide) for peptide in fdc["seq"]]
        group_p_corrs,group_ms1_traces,group_ms2_traces,group_iso_ratios, group_keys, group_fitted = ms1_cor_channels(DIAspectra, 
                                                                                                                        fdc, 
                                                                                                                        dc, 
                                                                                                                        mz_ppm=mz_ppm, 
                                                                                                                        rt_tol = rt_tol,
                                                                                                                        tag=mass_tag,
                                                                                                                        timeplex=timeplex
                                                                                                                        )
        
        ## create dictionary  that links keys to data so we can match the order of "fdc"
        
        linker_dict = {key:[group_idx,key_idx] for group_idx,keys in enumerate(group_keys) for key_idx,key in enumerate(keys)}
        
        p_corrs = [group_p_corrs[linker_dict[key][0]][linker_dict[key][1]] for key in all_keys]
        ms1_traces = [group_ms1_traces[linker_dict[key][0]][linker_dict[key][1]] for key in all_keys]
        ms2_traces = [group_ms2_traces[linker_dict[key][0]][linker_dict[key][1]] for key in all_keys]
        iso_ratios = [group_iso_ratios[linker_dict[key][0]][linker_dict[key][1]] for key in all_keys]
        extracted_keys = [group_keys[linker_dict[key][0]][linker_dict[key][1]] for key in all_keys]
        extracted_fitted = [group_fitted[linker_dict[key][0]][0][:,linker_dict[key][1]] for key in all_keys]
        extracted_fitted_specs = [group_fitted[linker_dict[key][0]][4] for key in all_keys]
        extracted_fitted_p = [group_fitted[linker_dict[key][0]][3] for key in all_keys]
        
        
        fdc["plexfitMS1"] = [np.max(i) for i in extracted_fitted]
        fdc["plexfitMS1_p"] = [j[np.argmax(i)].statistic  if type(j[np.argmax(i)])!=float else 0 for i,j in zip(extracted_fitted,extracted_fitted_p)]
    
        plexfittrace_idxs = [np.where([e in set(k) for e in j])[0] for i,j,k,p in zip(extracted_fitted,extracted_fitted_specs,ms2_traces,extracted_fitted_p)]
        plexfittrace = [i[j] for i,j in zip(extracted_fitted,plexfittrace_idxs)]
        plexfit_ps = [[i[k].statistic if type(i[k])!=float else 0 for k in j] for i,j in zip(extracted_fitted_p,plexfittrace_idxs)]
        # fdc["plexfitMS1_new"] = [np.max(i) for i in plexfittrace]
        fdc["plexfittrace"] = [";".join(map(str,i)) for i in plexfittrace] ###spec ids come from ms2_traces
        fdc["plexfit_ps"] = [";".join(map(str,i)) for i in plexfit_ps]
        
    else:
        fdc["untag_seq"] = fdc["seq"]
        p_corrs, ms1_traces, ms2_traces, iso_ratios = ms1_cor(DIAspectra, 
                                                                fdc, 
                                                                dc, 
                                                                mz_ppm=mz_ppm, 
                                                                rt_tol = rt_tol,
                                                                timeplex=timeplex)
        
    
    
    
    
    fdc["ms1_cor"] = [i[0] for i in p_corrs]
    
    for idx in range(config.num_iso_r):
        iso_num = idx+1
        fdc[f"iso{iso_num}_cor"] = [i[iso_num] for i in p_corrs]
    # fdc["iso1_cor"] = [i[1] for i in p_corrs]
    # fdc["iso2_cor"] = [i[2] for i in p_corrs]
    
    fdc["traceproduct"] = np.log10(fdc["ms1_cor"]*fdc["iso1_cor"]*fdc["iso2_cor"]+1e-6)
    
    # fdc["MS1_is1cor"] = [stats.pearsonr(list(i[0].values())[:10], list(i[1].values())[:10]).statistic for i in ms1_traces]
    
    
    fdc["iso_cor"] = [i[0].statistic for i in iso_ratios]
    
    fdc["MS1_Int"] = [i[2][0] for i in iso_ratios]
    fdc["MS1_Int"] = [np.linalg.lstsq(np.array(i[1])[:,np.newaxis], i[2])[0][0] for i in iso_ratios]
    
    # X[np.isnan(X)]=0 ## set nans to zero (mostly for r2 values)
    fdc["all_ms1_specs"] = [";".join(map(str,trace[0].keys())) for trace in ms1_traces]
    for i in range(config.num_iso_ms1):
        fdc[f"all_ms1_iso{i}vals"] = [";".join(map(str,trace[i].values())) for trace in ms1_traces]
    # fdc["ms2_trace"] = [";".join(map(str,trace.values())) for trace in ms2_traces]
    return fdc



class model_instance():
    def __init__(self,model_type):
        self.mode_type = model_type
        
    def predict(self,X):
        pred = self.__predict_fn__(X)
            
        if len(pred.shape)==2:
            output = pred[:,1]
        else:
            output = pred
        return output
        
        
class score_model():
    
    def __init__(self,model_type,n_splits=5,folder=None):
        self.model_type=model_type
        self.n_splits = n_splits
        self.folder = folder
                
    def run_model(self,X,y):
        if self.model_type=="rf":
            
            ### Random Forest
            def fit_model(X,y,idx=""):
                    m = model_instance(model_type=self.model_type)
                    m.model = RandomForestClassifier(n_estimators = 100, max_depth=10,n_jobs=-1)
                    m.model.fit(X,y)
                    m.__predict_fn__ = m.model.predict_proba
                    
                    if self.folder:
                        plt.subplots()
                        plt.barh(X.columns,m.model.feature_importances_)
                        plt.title("Feature Importance")
                        plt.savefig(self.folder+f"/RF{idx}_feature_importance.png",dpi=600,bbox_inches="tight")
                    
                    return m
                
            # self.model = fit_model(X,y)
            
        
        elif self.model_type=="lda":
            
            ## Linear Disriminant Analysis
            def fit_model(X,y,idx=""):
                    m = model_instance(model_type=self.model_type)
                    m.model = LinearDiscriminantAnalysis()
                    m.model.fit(X,y)
                    m.__predict_fn__ = m.model.predict_proba
                    return m
                
            # self.model = fit_model(X,y)
            
            
        elif self.model_type == "xg":
            
            ## XGBoost
            def fit_model(X,y,idx=""):
                    m = model_instance(model_type=self.model_type)
                    dTrain = xgb.DMatrix(X,y)
                    param = {'max_depth': 2, 'eta': 1, 'objective': 'binary:logistic'}
                    param['nthread'] = 4
                    param['eval_metric'] = 'auc'
                    
                    m.model = xgb.train(param, dtrain=dTrain)
                    def xg_predict(X):
                        X_convert = xgb.DMatrix(X)
                        return m.model.predict(X_convert)
                    m.__predict_fn__ = xg_predict
                    
                    if self.folder:
                        plt.subplots()
                        fi = m.model.get_score(importance_type="gain")
                        plt.barh(X.columns,[fi[i] if i in fi else 0 for i in X.columns])
                        plt.title("Feature Importance")
                        plt.savefig(self.folder+f"/XGBoost{idx}_feature_importance.png",dpi=600,bbox_inches="tight")
                    
                    
                    return m
                
            # self.model = fit_model(X,y)
        
            
        elif self.model_type == "nn":
            X = preprocessing.StandardScaler().fit(X).transform(X)
            ## Neural network
            def fit_model(X,y):
                    m = model_instance(model_type=self.model_type)
                    m.model = MLPClassifier((32,16,8,4),activation="relu")
                    m.model.fit(X,y)
                    m.__predict_fn__ = m.model.predict_proba
                    return m
                
        else:
            raise ValueError("Unsupported model type")
            
        kf = KFold(n_splits=self.n_splits,shuffle=True)
        k_orders = [i for i in kf.split(X,y)]
        rev_order = np.argsort(np.concatenate([i[1] for i in k_orders])) # collapse test sets and get order

        data_splits = [[X.iloc[i[0]],X.iloc[i[1]],y[i[0]],y[i[1]]] for i in k_orders] # put data into folds


        self.models = []
        self.predictions=[]
        model_idx=0
        for X_train, X_test, y_train, y_test in tqdm.tqdm(data_splits):
            m = fit_model(X_train,y_train,idx=model_idx)
            self.models.append(m)
            self.predictions.append(m.predict(X_test))
            model_idx+=1
            
        return np.concatenate(self.predictions)[rev_order]
    
    
    
from autogluon.tabular import TabularPredictor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
from scipy.stats import gaussian_kde

# def score_precursors_JD_works_02202025(fdc, model_type="ag", fdr_t=0.01, folder=None):
#     """
#     Parameters
#     ----------
#     fdc : pandas.DataFrame
#         All PSMs identified.
#     model_type : string [autogluon]
#                  Type of ML model used to discriminate targets and decoys.
#                  Only 'autogluon' is supported in this implementation.
#     fdr_t : float
#         False discovery rate threshold.
#     folder : str, optional
#         Folder path for saving plots.

#     Returns
#     -------
#     fdc : pandas.DataFrame
#         Updated dataframe with prediction values and Q-values.
#     """
#     assert model_type == "ag", 'model_type must be "autogluon"'
#     print("Scoring IDs")
    

#     fdc["decoy"] = np.array(["Decoy" in i for i in fdc["seq"]])
 

#     _bool = np.logical_and(~fdc["decoy"],fdc.coeff>1)

#     ## define our features and labels for the model
#     y = np.array(_bool,dtype=int)
#     fdc['y_count'] = fdc['frag_names'].str.count('y')
#     fdc['b_count'] = fdc['frag_names'].str.count('b')
#     fdc['yb_count'] = fdc['b_count'] + fdc['y_count']
#     fdc['y_minus_b_count'] = fdc['b_count'] - fdc['y_count']
#     fdc['y_fraction'] = fdc['y_count']/(fdc['b_count'] + fdc['y_count'])
#     fdc['coeff_MS1_Int_ratio'] = np.log10(fdc['coeff']+1) - np.log10(fdc['MS1_Int']+1)
#     fdc['coeff_x_MS1_Int'] = np.log10(fdc['coeff']+1) + np.log10(fdc['MS1_Int']+1)
#     fdc['abs_rt_error'] = np.abs(fdc['rt_error'])

#     fdc['mass'] = fdc['z']*fdc['mz']
#     fdc['ends_with_K'] = fdc['stripped_seq'].apply(lambda x: 1 if x.endswith('K') else 0)
    
#         # Count decoys and targets
#     num_decoys = fdc["decoy"].sum()  
#     num_targets = len(fdc) - num_decoys  
    
#     # Compute sample weights
#     decoy_weight = 1.0  # Decoys always have weight 1
#     target_weight = num_decoys / num_targets if num_targets > 0 else 1.0  # Avoid division by zero
    
#     # Assign sample weights
#     fdc = fdc.copy()  # Avoid SettingWithCopyWarning
#     fdc["sample_weight"] = fdc["decoy"].apply(lambda x: decoy_weight if x else target_weight)

#     # exclude necessary columns
#     # dat_keep = fdc[['MS1_Int',"iso_cor","traceproduct","iso1_cor","iso2_cor","ms1_cor","sq_mz_error","sq_rt_error","pep_len","mz",
#     #                "cosine","frac_int_matched_pred_sigcoeff","frac_int_matched_pred","hyperscore","frac_int_uniq_pred",
#     #                "frac_int_uniq","prec_r2","spec_r2","frac_int_matched","rt_error","mz_error","frac_dia_int","frac_lib_int",
#     #                "num_lib","rt","z","coeff","decoy","y_count","b_count","yb_count","y_minus_b_count","coeff_x_MS1_Int",
#     #                "y_fraction","coeff_MS1_Int_ratio","mass","ends_with_K","sample_weight"]]

#     dat_keep = fdc[["iso_cor","traceproduct","iso1_cor","iso2_cor","sq_mz_error","sq_rt_error","pep_len","mz",
#                    "cosine","frac_int_matched_pred_sigcoeff","frac_int_matched_pred","hyperscore","frac_int_uniq_pred",
#                    "frac_int_uniq","prec_r2","spec_r2","frac_int_matched","rt_error","mz_error","frac_dia_int","frac_lib_int",
#                    "num_lib","rt","z","coeff","decoy","y_count","b_count","yb_count","y_minus_b_count",
#                    "y_fraction","mass","ends_with_K","abs_rt_error","sample_weight"]]


#     #estimate RT densities using Gaussian KDE
#     rt_decoy = dat_keep[dat_keep['decoy'] == True]['rt']
#     rt_non_decoy = dat_keep[dat_keep['decoy'] == False]['rt']
    
#     kde_decoy = gaussian_kde(rt_decoy)
#     kde_non_decoy = gaussian_kde(rt_non_decoy)
    
#     density_decoy = kde_decoy(dat_keep['rt'])
#     density_non_decoy = kde_non_decoy(dat_keep['rt'])
    
#     # Compute the ratio of densities (decoy / non-decoy)
#     # Handle division by zero by replacing zeros with a small value
#     density_non_decoy = np.where(density_non_decoy == 0, 1e-10, density_non_decoy)
#     dat_keep['density_ratio'] = density_decoy / density_non_decoy
#     dat_keep['density_diff'] = density_decoy - density_non_decoy
    
#     _bool = np.logical_and(~dat_keep["decoy"], dat_keep["coeff"] > 1)
#     y = np.array(_bool, dtype=int)
        

    
#     X = dat_keep.drop(columns=['decoy']) 
#     X = X.fillna(0)  
    
#     y = np.array(~dat_keep["decoy"], dtype=int)
#     X['label'] = y  

#     X = X.astype({col: 'float32' for col in X.select_dtypes('float64').columns})

#     hyperparams = {
#         'NN_TORCH': {},
#         'FASTAI': {},
#         'GBM': {},
#         'CAT': {},
#     }

#     predictor = TabularPredictor(label='label', eval_metric='precision',sample_weight="sample_weight").fit(
#         train_data=X,
#         num_bag_folds=10,  #10-fold bagging
#         presets='high_quality',
#         hyperparameters = hyperparams,
#         time_limit=900 #seconds
        
#     )

#     # out of fold predictions
#     oof_predictions = predictor.predict_proba_oof()
#     output = oof_predictions.iloc[:, 1]  # Use probabilities for the positive class
    
#     model_name = "AutoGluon"


    
#     # Calculate ROC and threshold for FDR
#     fpr, tpr, _ = roc_curve(y, output)
#     score_order = np.argsort(-output)
#     orig_order = np.argsort(score_order)
#     decoy_order = fdc["decoy"][score_order]
#     frac_decoy = np.cumsum(decoy_order)/np.arange(1,len(decoy_order)+1)
#     T = output[score_order[np.searchsorted(frac_decoy, fdr_t)]]
    
#     print()
#     print("#IDs at 1% FDR:", np.sum(output > T))
#     above_t = output > T
#     fdc["PredVal"] = output
#     fdc["Qvalue"] = frac_decoy[orig_order]
    
#     # Plot results if folder is specified
#     if folder:
#         plt.subplots()
#         y_log = False
#         vals, bins, _ = plt.hist(output, 50, log=y_log, label="All")
#         plt.hist(output[y.astype(bool)], bins, alpha=.5, log=y_log, label="Targets")
#         plt.hist(output[~y.astype(bool)], bins, alpha=.5, log=y_log, label="Decoys")
#         plt.legend()
#         plt.title(model_name)
#         plt.vlines(T, 0, max(vals))
#         plt.savefig(folder + "/ModelScore.png", dpi=600, bbox_inches="tight")
#         for feat in ['rt_error', 'mz_error']:
#             func = np.array
#             plt.subplots()
#             vals, bins, _ = plt.hist(func([i for i in fdc[feat]]), 40, label="All")
#             vals, bins, _ = plt.hist(func([i for i in fdc[feat][above_t]]), bins, alpha=.5, label=">Threshold")
#             vals, bins, _ = plt.hist(func([i for i in fdc[feat][~above_t]]), bins, alpha=.5, label="<Threshold")
#             plt.xlabel(feat)
#             plt.ylabel("Frequency")
#             plt.title(model_name)
#             plt.legend()
#             plt.savefig(folder + f"/{feat}_error.png", dpi=600, bbox_inches="tight")

#     #checking feature importance from training data which should probably be ok, but test data might be better. Though, this will show what the model(s) used to fit the actual data.

#     X_sample = X.sample(frac=0.3, random_state=123) #speed up by only taking 30% of data for feature importance computations
#     feature_importance = predictor.feature_importance(X_sample)
#     feature_importance = feature_importance.sort_values(by="importance", ascending=False)
        
#     print("Feature Importances:")
#     print(feature_importance)
#     # Plot results if folder is specified
#     if folder:
#         plt.figure(figsize=(11, 8.5))
#         plt.barh(feature_importance.index, feature_importance["importance"], color='blue')
#         plt.xlabel("Importance")
#         plt.ylabel("Features")
#         plt.title("AutoGluon feature importances")
#         plt.gca().invert_yaxis()
#         plt.savefig(folder + "/Feature_Importance_AG.png", dpi=600, bbox_inches="tight")
        
#     return fdc    






# # Function to compute fraction of matched frag_int
# def compute_fraction(frag_int, unique_frag_mz, frag_mz):
#     # Convert to string and split safely
#     frag_int_list = list(map(float, str(frag_int).split(";")))
#     frag_mz_list = list(map(float, str(frag_mz).split(";")))
#     unique_frag_mz_list = set(map(float, str(unique_frag_mz).split(";")))  # Convert to set for fast lookup

#     # Sum of all fragment intensities
#     total_intensity = sum(frag_int_list)

#     # Find matching intensities
#     matched_intensity = sum(intensity for mz, intensity in zip(frag_mz_list, frag_int_list) if mz in unique_frag_mz_list)

#     # Compute fraction
#     return matched_intensity / total_intensity if total_intensity > 0 else 0

# # Apply function to DataFrame



# def score_precursors(fdc, model_type="ag", fdr_t=0.01, folder=None):
#     """
#     Perform ML-based precursor scoring for each unique time_channel if it exists,
#     otherwise, apply a different function.
    
#     Parameters
#     ----------
#     fdc : pandas.DataFrame
#         All PSMs identified.
#     model_type : string
#         Type of ML model used to discriminate targets and decoys (only 'autogluon' supported).
#     fdr_t : float
#         False discovery rate threshold.
#     folder : str, optional
#         Folder path for saving plots.

#     Returns
#     -------
#     fdc : pandas.DataFrame
#         Updated dataframe with prediction values and Q-values.
#     """
#     assert model_type == "ag", 'model_type must be "autogluon"'
#     print("Scoring IDs")
    
#     fdc["frac_lib_int_unique"] = fdc.apply(lambda row: compute_fraction(row["frag_int"], row["unique_frag_mz"], row["frag_mz"]), axis=1)

#     fdc["decoy"] = np.array(["Decoy" in i for i in fdc["seq"]])
    
#     fdc['y_count'] = fdc['frag_names'].str.count('y')
#     fdc['b_count'] = fdc['frag_names'].str.count('b')
#     fdc['yb_count'] = fdc['b_count'] + fdc['y_count']
#     fdc['y_minus_b_count'] = fdc['b_count'] - fdc['y_count']
#     fdc['y_fraction'] = fdc['y_count'] / (fdc['b_count'] + fdc['y_count'])
#     fdc['coeff_MS1_Int_ratio'] = np.log10(fdc['coeff'] + 1) - np.log10(fdc['MS1_Int'] + 1)
#     fdc['coeff_x_MS1_Int'] = np.log10(fdc['coeff'] + 1) + np.log10(fdc['MS1_Int'] + 1)
#     dat['abs_rt_error'] = np.abs(dat['rt_error']+0.0001)
#     dat['abs_mz_error'] = np.abs(dat['mz_error']+0.0000000001)
#     dat["unique_obs_int_counts"] = dat["unique_obs_int"].apply(lambda x: len(x.split(";")) if pd.notna(x) and x.strip() != "" else 0)
#     dat["obs_int_counts"] = dat["obs_int"].apply(lambda x: len(x.split(";")) if pd.notna(x) and x.strip() != "" else 0)
#     dat["ratio_uniqueObs_TotalObs"] = dat['unique_obs_int_counts']/(dat['unique_obs_int_counts']+dat['obs_int_counts'])
#     dat['mass'] = dat['z']*dat['mz']
#     fdc['ends_with_K'] = fdc['stripped_seq'].apply(lambda x: 1 if x.endswith('K') else 0)
    
#     if 'time_channel' in fdc.columns:
#         results = []
#         for time_channel in fdc['time_channel'].unique():
#             print(f"Processing time_channel: {time_channel}")
#             subset_fdc = fdc[fdc['time_channel'] == time_channel].copy()
#             subset_fdc_reset = subset_fdc.reset_index()
#             results.append(process_time_channel(subset_fdc_reset, model_type, fdr_t, folder, time_channel))
#         return pd.concat(results)
#     else:
#         return score_precursors_JD_works_02202025(fdc)




# def process_time_channel(subset_fdc, model_type, fdr_t, folder, time_channel):
#     """ Function to process each time_channel subset. """
#     num_decoys = subset_fdc["decoy"].sum()  
#     num_targets = len(subset_fdc) - num_decoys  
#     target_weight = num_decoys / num_targets if num_targets > 0 else 1.0  
    
#     subset_fdc["sample_weight"] = subset_fdc["decoy"].apply(lambda x: 1.0 if x else target_weight)
#     mean_abs_rt_error = subset_fdc['abs_rt_error'].mean()
#     subset_fdc["sample_weight"] = subset_fdc["sample_weight"]*(subset_fdc["abs_rt_error"]+mean_abs_rt_error)#*(1/dat_keep["frac_lib_int"])#*dat_keep["abs_mz_error"]

#     dat_keep = subset_fdc[['MS1_Int',#"iso_cor","cosine", these increase FDR (iso_cor and cosine)
#                 "traceproduct","iso1_cor","iso2_cor","ms1_cor","sq_mz_error","sq_rt_error","pep_len","mz",
#                "frac_int_matched_pred_sigcoeff","frac_int_matched_pred","hyperscore","frac_int_uniq_pred",
#                "frac_int_uniq","prec_r2","spec_r2","frac_int_matched","rt_error","mz_error","frac_dia_int","frac_lib_int",#"coeff_x_MS1_Int",
#                 'abs_rt_error','abs_mz_error','mass',
#                 'unique_obs_int_counts', 'frac_lib_int_unique', 'obs_int_counts', 'ratio_uniqueObs_TotalObs',
#                "num_lib","rt","z","coeff","decoy","sample_weight"]].copy()

    
#     # dat_keep = subset_fdc[["traceproduct", "iso1_cor", "iso2_cor", "sq_mz_error", "sq_rt_error", "pep_len", "mz",
#     #                        "frac_int_matched_pred_sigcoeff", "frac_int_matched_pred", "hyperscore", "frac_int_uniq_pred",
#     #                        "frac_int_uniq", "prec_r2", "spec_r2", "frac_int_matched", "rt_error", "mz_error", "frac_dia_int", "frac_lib_int",
#     #                        "num_lib", "rt", "z", "coeff", "decoy", "y_count", "b_count", "yb_count", "y_minus_b_count", 
#     #                        #"iso_cor","cosine", these increase FDR (iso_cor and cosine)
#     #                        "coeff_MS1_Int_ratio", "coeff_x_MS1_Int",
                           
#     #                        "y_fraction", "mass", "ends_with_K", "abs_rt_error","sample_weight"]].copy()
    
#     kde_decoy = gaussian_kde(dat_keep[dat_keep['decoy'] == True]['rt'])
#     kde_non_decoy = gaussian_kde(dat_keep[dat_keep['decoy'] == False]['rt'])
    
#     density_decoy = kde_decoy(dat_keep['rt'])
#     density_non_decoy = kde_non_decoy(dat_keep['rt'])
#     density_non_decoy = np.where(density_non_decoy == 0, 1e-10, density_non_decoy)
    
#     dat_keep.loc[:,'density_ratio'] = density_decoy / density_non_decoy
#     dat_keep.loc[:,'density_diff'] = density_decoy - density_non_decoy
    
#     y = np.array(~dat_keep["decoy"], dtype=int)
#     X_temp = dat_keep.drop(columns=['decoy']).fillna(0).astype({col: 'float32' for col in dat_keep.select_dtypes('float64').columns})
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     X = scaler.fit_transform(X_temp)
#     X = pd.DataFrame(X, columns=X_temp.columns)
#     X['label'] = y  

#     hyperparams = {
#         'NN_TORCH': {},
#         'FASTAI': {},
#         'GBM': {},
#         'CAT': {},
#     }
    
#     predictor = TabularPredictor(label='label', eval_metric='precision', sample_weight="sample_weight").fit(
#         train_data=X, 
#         num_bag_folds=10, 
#         presets='high_quality', 
#         hyperparameters=hyperparams, 
#         time_limit=900
#     )
    
#     oof_predictions = predictor.predict_proba_oof()
#     output = oof_predictions.iloc[:, 1]
    
#     fpr, tpr, _ = roc_curve(y, output)
#     score_order = np.argsort(-output)
#     orig_order = np.argsort(score_order)
#     decoy_order = subset_fdc["decoy"][score_order]
#     frac_decoy = np.cumsum(decoy_order)/np.arange(1,len(decoy_order)+1)
#     T = output[score_order[np.searchsorted(frac_decoy, fdr_t)]]
    
#     #print("#IDs at 1% FDR:", np.sum(output > T))
#     above_t = output > T
    
#     subset_fdc["PredVal"] = output
#     subset_fdc["Qvalue"] = frac_decoy[orig_order]
#     print("#IDs at 1% FDR: ",(subset_fdc["Qvalue"] < 0.01).sum())

#     if folder:
#         plt.subplots()
#         y_log=False
#         vals,bins,_ = plt.hist(output,50,log=y_log,label="All")
#         plt.hist(output[y.astype(bool)],bins,alpha=.5,log=y_log,label="Targets")
#         plt.hist(output[~y.astype(bool)],bins,alpha=.5,log=y_log,label="Decoys")
#         plt.legend()
#         plt.title(f"Model Score - {time_channel}")
#         plt.vlines(T,0,max(vals))
#         plt.savefig(f"{folder}/ModelScore_TP{time_channel}.png", dpi=600, bbox_inches="tight")
        
        
        
#         feat = 'rt_error'
#         func = np.array#np.log10#
#         plt.subplots()
#         vals,bins,_ = plt.hist(func([i for i in subset_fdc[feat]]),40,label="All")
#         # plt.hist([],[])
#         vals,bins,_ = plt.hist(func([i for i in subset_fdc[feat][above_t]]),bins,alpha=.5,label=">Threshold")
#         vals,bins,_ = plt.hist(func([i for i in subset_fdc[feat][~above_t]]),bins,alpha=.5,label="<Threshold")
#         plt.xlabel(feat)
#         plt.ylabel("Frequency")
#         plt.title(f"RT Error... TP{time_channel}")
#         plt.legend()
#         plt.savefig(f"{folder}/RT_Error_TP{time_channel}.png", dpi=600, bbox_inches="tight")
        
                
#         feat = 'mz_error'
#         func = np.array#np.log10#
#         plt.subplots()
#         vals,bins,_ = plt.hist(func([i for i in subset_fdc[feat]]),40,label="All")
#         # plt.hist([],[])
#         vals,bins,_ = plt.hist(func([i for i in subset_fdc[feat][above_t]]),bins,alpha=.5,label=">Threshold")
#         vals,bins,_ = plt.hist(func([i for i in subset_fdc[feat][~above_t]]),bins,alpha=.5,label="<Threshold")
#         plt.xlabel(feat)
#         plt.ylabel("Frequency")
#         plt.title(f"MZ Error... TP{time_channel}")
#         plt.legend()
#         plt.savefig(f"{folder}/MZ_Error_TP{time_channel}.png", dpi=600, bbox_inches="tight")
    
#     # # Plot results if folder is specified
#     # if folder:
#     #     plt.figure()
#     #     plt.hist(output, 50, label="All")
#     #     plt.hist(output[y.astype(bool)], alpha=0.5, label="Targets")
#     #     plt.hist(output[~y.astype(bool)], alpha=0.5, label="Decoys")
#     #     plt.legend()
#     #     plt.title(f"Model Score - {time_channel}")
#     #     plt.savefig(f"{folder}/ModelScore_TP{time_channel}.png", dpi=600, bbox_inches="tight")
            
#     #     for feat in ['rt_error', 'mz_error']:
#     #         plt.figure()
#     #         plt.hist(subset_fdc[feat], bins=40, label="All")
#     #         plt.hist(subset_fdc[feat][above_t], alpha=0.5, label=">Threshold")
#     #         plt.hist(subset_fdc[feat][~above_t], alpha=0.5, label="<Threshold")
#     #         plt.xlabel(feat)
#     #         plt.ylabel("Frequency")
#     #         plt.legend()
#     #         plt.title(f"{feat} - {time_channel}")
#     #         plt.savefig(f"{folder}/{feat}_error_TP{time_channel}.png", dpi=600, bbox_inches="tight")

#     #checking feature importance from training data which should probably be ok, but test data might be better. Though, this will show what the model(s) used to fit the actual data.

#     X_sample = X.sample(frac=0.3, random_state=123) #speed up by only taking 30% of data for feature importance computations
#     feature_importance = predictor.feature_importance(X_sample)
#     feature_importance = feature_importance.sort_values(by="importance", ascending=False)
        
#     print("Feature Importances:")
#     print(feature_importance)
#     # Plot results if folder is specified
#     if folder:
#         plt.figure(figsize=(11, 8.5))
#         plt.barh(feature_importance.index, feature_importance["importance"], color='blue')
#         plt.xlabel("Importance")
#         plt.ylabel("Features")
#         plt.title("AutoGluon feature importances")
#         plt.gca().invert_yaxis()
#         plt.savefig(folder + f"/Feature_Importance_AG_TP{time_channel}.png", dpi=600, bbox_inches="tight")
    
#     return subset_fdc






def score_precursors_JD_works_02202025(fdc, model_type="ag", fdr_t=0.01, folder=None):
    """
    Parameters
    ----------
    fdc : pandas.DataFrame
        All PSMs identified.
    model_type : string [autogluon]
                 Type of ML model used to discriminate targets and decoys.
                 Only 'autogluon' is supported in this implementation.
    fdr_t : float
        False discovery rate threshold.
    folder : str, optional
        Folder path for saving plots.

    Returns
    -------
    fdc : pandas.DataFrame
        Updated dataframe with prediction values and Q-values.
    """
    assert model_type == "ag", 'model_type must be "autogluon"'
    print("Scoring IDs")
    
    if config.args.plexDIA:
        fdc = plexDIA_feature_engineering(fdc)
    
 
    num_decoys = fdc["decoy"].sum()  
    num_targets = len(fdc) - num_decoys  
    target_weight = num_decoys / num_targets if num_targets > 0 else 1.0  
    
    fdc["sample_weight"] = fdc["decoy"].apply(lambda x: 1.0 if x else target_weight)
    mean_abs_rt_error = fdc['abs_rt_error'].mean()
    fdc["sample_weight"] = fdc["sample_weight"]*(fdc["abs_rt_error"]+mean_abs_rt_error)#*(1/dat_keep["frac_lib_int"])#*dat_keep["abs_mz_error"]

    plexDIA_features = (
        "abs_diff_rt_from_median", "diff_coeff_from_median", "diff_frac_int_uniq_pred_from_median", "diff_frac_dia_int_from_median",
        "abs_diff_mz_error_from_median", "abs_diff_frac_int_uniq_from_median", "diff_frac_lib_int_from_median",
        "num_channels_greater0_coeff", "num_channels_greater0_frac_int_uniq_pred", "num_channels_greater0_frac_dia_int", 
        "num_channels_greater0_frac_int_uniq", "num_channels_greater0_frac_lib_int",
        "channels_matched"
    )
    
    keep_features = (
       # "iso_cor", "cosine" (more IDs but higher FDR)
        "traceproduct", "iso1_cor", "iso2_cor", "sq_mz_error", "sq_rt_error", "pep_len", "mz",
        "frac_int_matched_pred_sigcoeff", "frac_int_matched_pred", "hyperscore", "frac_int_uniq_pred",
        "frac_int_uniq", "prec_r2", "spec_r2", "frac_int_matched", "rt_error", "mz_error", "frac_dia_int", "frac_lib_int",
        "num_lib", "rt", "z", "coeff", "decoy", "y_count", "b_count", "yb_count", "y_minus_b_count",
        "y_fraction", "mass", "ends_with_K", "abs_rt_error", "sample_weight",
        "coeff_MS1_Int_ratio",'unique_obs_int_counts', 'obs_int_counts','ratio_uniqueObs_TotalObs'

    )
    
    # Conditionally combine features
    combined_features = keep_features + (plexDIA_features if config.args.plexDIA else ())
    
    # Filter DataFrame based on selected features
    dat_keep = fdc[list(combined_features)]  # Convert tuple to list for indexing
    

    kde_decoy = gaussian_kde(dat_keep[dat_keep['decoy'] == True]['rt'])
    kde_non_decoy = gaussian_kde(dat_keep[dat_keep['decoy'] == False]['rt'])
    
    density_decoy = kde_decoy(dat_keep['rt'])
    density_non_decoy = kde_non_decoy(dat_keep['rt'])
    density_non_decoy = np.where(density_non_decoy == 0, 1e-10, density_non_decoy)
    dat_keep = dat_keep.copy()
    dat_keep.loc[:,'density_ratio'] = density_decoy / density_non_decoy
    dat_keep.loc[:,'density_diff'] = density_decoy - density_non_decoy
    
    y = np.array(~dat_keep["decoy"], dtype=int)
    X_temp = dat_keep.drop(columns=['decoy']).fillna(0).astype({col: 'float32' for col in dat_keep.select_dtypes('float64').columns})
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X_temp)
    X = pd.DataFrame(X, columns=X_temp.columns)
    X['label'] = y  

    hyperparams = {
        'NN_TORCH': {},
        'FASTAI': {},
        'GBM': {},
        'CAT': {},
    }
    
    predictor = TabularPredictor(label='label', eval_metric='precision', sample_weight="sample_weight").fit(
        train_data=X, 
        num_bag_folds=10, 
        presets='high_quality', 
        hyperparameters=hyperparams, 
        time_limit=900
    )
    
    oof_predictions = predictor.predict_proba_oof()
    output = oof_predictions.iloc[:, 1]
    
    fpr, tpr, _ = roc_curve(y, output)
    score_order = np.argsort(-output)
    orig_order = np.argsort(score_order)
    decoy_order = fdc["decoy"][score_order]
    frac_decoy = np.cumsum(decoy_order)/np.arange(1,len(decoy_order)+1)
    T = output[score_order[np.searchsorted(frac_decoy, fdr_t)]]
    
    #print("#IDs at 1% FDR:", np.sum(output > T))
    above_t = output > T
    
    fdc["PredVal"] = output
    fdc["Qvalue"] = frac_decoy[orig_order]
    print("#IDs at 1% FDR: ",(fdc["Qvalue"] < 0.01).sum())

    if folder:
        plt.subplots()
        y_log=False
        vals,bins,_ = plt.hist(output,50,log=y_log,label="All")
        plt.hist(output[y.astype(bool)],bins,alpha=.5,log=y_log,label="Targets")
        plt.hist(output[~y.astype(bool)],bins,alpha=.5,log=y_log,label="Decoys")
        plt.legend()
        plt.title(f"Model Score")
        plt.vlines(T,0,max(vals))
        plt.savefig(f"{folder}/ModelScore.png", dpi=600, bbox_inches="tight")
        
        
        
        feat = 'rt_error'
        func = np.array#np.log10#
        plt.subplots()
        vals,bins,_ = plt.hist(func([i for i in fdc[feat]]),40,label="All")
        # plt.hist([],[])
        vals,bins,_ = plt.hist(func([i for i in fdc[feat][above_t]]),bins,alpha=.5,label=">Threshold")
        vals,bins,_ = plt.hist(func([i for i in fdc[feat][~above_t]]),bins,alpha=.5,label="<Threshold")
        plt.xlabel(feat)
        plt.ylabel("Frequency")
        plt.title(f"RT Error")
        plt.legend()
        plt.savefig(f"{folder}/RT_Error.png", dpi=600, bbox_inches="tight")
        
                
        feat = 'mz_error'
        func = np.array#np.log10#
        plt.subplots()
        vals,bins,_ = plt.hist(func([i for i in fdc[feat]]),40,label="All")
        # plt.hist([],[])
        vals,bins,_ = plt.hist(func([i for i in fdc[feat][above_t]]),bins,alpha=.5,label=">Threshold")
        vals,bins,_ = plt.hist(func([i for i in fdc[feat][~above_t]]),bins,alpha=.5,label="<Threshold")
        plt.xlabel(feat)
        plt.ylabel("Frequency")
        plt.title(f"MZ Error")
        plt.legend()
        plt.savefig(f"{folder}/MZ_Error.png", dpi=600, bbox_inches="tight")
    
 
    X_sample = X.sample(frac=0.3, random_state=123) #speed up by only taking 30% of data for feature importance computations
    feature_importance = predictor.feature_importance(X_sample)
    feature_importance = feature_importance.sort_values(by="importance", ascending=False)
        
    print("Feature Importances:")
    print(feature_importance)
    # Plot results if folder is specified
    if folder:
        plt.figure(figsize=(11, 8.5))
        plt.barh(feature_importance.index, feature_importance["importance"], color='blue')
        plt.xlabel("Importance")
        plt.ylabel("Features")
        plt.title("AutoGluon feature importances")
        plt.gca().invert_yaxis()
        plt.savefig(folder + f"/Feature_Importance_AG.png", dpi=600, bbox_inches="tight")
    
    return fdc



def plexDIA_feature_engineering(fdc):
    # fdc["median_rt"] = fdc.groupby(["untag_prec"])['rt'].transform("median")
    # fdc.loc[fdc["channels_matched"] == 1, "median_rt"] = pd.NA
    # fdc["abs_diff_rt_from_median"] = np.abs(fdc['rt'] - fdc['median_rt'])
    # fdc["abs_diff_rt_from_median"].fillna(fdc["abs_diff_rt_from_median"].mean(), inplace=True)
    
    
    # fdc["median_frac_int_uniq_pred"] = fdc.groupby(["untag_prec"])['frac_int_uniq_pred'].transform("median")
    # fdc.loc[fdc["channels_matched"] == 1, "median_frac_int_uniq_pred"] = pd.NA
    # fdc["abs_diff_frac_int_uniq_pred_from_median"] = fdc['frac_int_uniq_pred']-fdc['median_frac_int_uniq_pred']
    # fdc["abs_diff_frac_int_uniq_pred_from_median"].fillna(fdc["abs_diff_frac_int_uniq_pred_from_median"].mean(), inplace=True)

    # pivot_df = fdc.pivot(index=["untag_prec",'decoy'], columns="channel", values="frac_int_uniq_pred")
    fdc["untag_prec"] = ["_".join([i[0],str(int(i[1]))]) for i in zip(fdc["untag_seq"],fdc["z"])]
    fdc["median_rt"] = fdc.groupby(["untag_prec"])['rt'].transform("median")
    fdc["channels_matched"] = fdc.groupby("untag_prec")["untag_prec"].transform("count")
    fdc.loc[fdc["channels_matched"] == 1, "median_rt"] = pd.NA
    fdc["abs_diff_rt_from_median"] = np.abs(fdc['rt'] - fdc['median_rt'])
    fdc["abs_diff_rt_from_median"].fillna(fdc["abs_diff_rt_from_median"].mean(), inplace=True)
    
    
    fdc["median_coeff"] = fdc.groupby(["untag_prec"])['coeff'].transform("median")
    fdc.loc[fdc["channels_matched"] == 1, "median_coeff"] = pd.NA
    fdc["diff_coeff_from_median"] = np.log10(fdc['coeff']+1) - np.log10(fdc['median_coeff']+1)
    fdc["diff_coeff_from_median"].fillna(fdc["diff_coeff_from_median"].mean(), inplace=True)
    
    
    fdc["median_frac_int_uniq_pred"] = fdc.groupby(["untag_prec"])['frac_int_uniq_pred'].transform("median")
    fdc.loc[fdc["channels_matched"] == 1, "median_frac_int_uniq_pred"] = pd.NA
    fdc["diff_frac_int_uniq_pred_from_median"] = fdc['frac_int_uniq_pred'] - fdc['median_frac_int_uniq_pred']
    fdc["diff_frac_int_uniq_pred_from_median"].fillna(fdc["diff_frac_int_uniq_pred_from_median"].mean(), inplace=True)
    
    
    fdc["median_frac_dia_int"] = fdc.groupby(["untag_prec"])['frac_dia_int'].transform("median")
    fdc.loc[fdc["channels_matched"] == 1, "median_frac_dia_int"] = pd.NA
    fdc["diff_frac_dia_int_from_median"] = fdc['frac_dia_int'] - fdc['median_frac_dia_int']
    fdc["diff_frac_dia_int_from_median"].fillna(fdc["diff_frac_dia_int_from_median"].mean(), inplace=True)
    
    
    fdc["median_mz_error"] = fdc.groupby(["untag_prec"])['mz_error'].transform("median")
    fdc.loc[fdc["channels_matched"] == 1, "median_mz_error"] = pd.NA
    fdc["abs_diff_mz_error_from_median"] = np.abs(fdc['mz_error'] - fdc['median_mz_error'])
    fdc["abs_diff_mz_error_from_median"].fillna(fdc["abs_diff_mz_error_from_median"].mean(), inplace=True)
    
    
    fdc["median_frac_int_uniq"] = fdc.groupby(["untag_prec"])['frac_int_uniq'].transform("median")
    fdc.loc[fdc["channels_matched"] == 1, "median_frac_int_uniq"] = pd.NA
    fdc["abs_diff_frac_int_uniq_from_median"] = fdc['frac_int_uniq'] - fdc['median_frac_int_uniq']
    fdc["abs_diff_frac_int_uniq_from_median"].fillna(fdc["abs_diff_frac_int_uniq_from_median"].mean(), inplace=True)
    
    
    fdc["median_frac_lib_int"] = fdc.groupby(["untag_prec"])['frac_lib_int'].transform("median")
    fdc.loc[fdc["channels_matched"] == 1, "median_frac_lib_int"] = pd.NA
    fdc["diff_frac_lib_int_from_median"] = fdc['frac_lib_int'] - fdc['median_frac_lib_int']
    fdc["diff_frac_lib_int_from_median"].fillna(fdc["diff_frac_lib_int_from_median"].mean(), inplace=True)
    
    # Count number of entries with 'frac_int_uniq_pred' > 0
    fdc['num_channels_greater0_coeff'] = fdc.groupby("untag_prec")['coeff'].transform(lambda x: (x > 0).sum())
    fdc['num_channels_greater0_frac_int_uniq_pred'] = fdc.groupby("untag_prec")['frac_int_uniq_pred'].transform(lambda x: (x > 0).sum())
    fdc['num_channels_greater0_frac_dia_int'] = fdc.groupby("untag_prec")['frac_dia_int'].transform(lambda x: (x > 0).sum())
    fdc['num_channels_greater0_frac_int_uniq'] = fdc.groupby("untag_prec")['frac_int_uniq'].transform(lambda x: (x > 0).sum())
    fdc['num_channels_greater0_frac_lib_int'] = fdc.groupby("untag_prec")['frac_lib_int'].transform(lambda x: (x > 0).sum())

    return fdc




# Function to compute fraction of matched frag_int
def compute_fraction(frag_int, unique_frag_mz, frag_mz):
    # Convert to string and split safely
    frag_int_list = list(map(float, str(frag_int).split(";")))
    frag_mz_list = list(map(float, str(frag_mz).split(";")))
    unique_frag_mz_list = set(map(float, str(unique_frag_mz).split(";")))  # Convert to set for fast lookup

    # Sum of all fragment intensities
    total_intensity = sum(frag_int_list)

    # Find matching intensities
    matched_intensity = sum(intensity for mz, intensity in zip(frag_mz_list, frag_int_list) if mz in unique_frag_mz_list)

    # Compute fraction
    return matched_intensity / total_intensity if total_intensity > 0 else 0

# Apply function to DataFrame



def score_precursors(fdc, model_type="ag", fdr_t=0.01, folder=None):
    """
    Perform ML-based precursor scoring for each unique time_channel if it exists,
    otherwise, apply a different function.
    
    Parameters
    ----------
    fdc : pandas.DataFrame
        All PSMs identified.
    model_type : string
        Type of ML model used to discriminate targets and decoys (only 'autogluon' supported).
    fdr_t : float
        False discovery rate threshold.
    folder : str, optional
        Folder path for saving plots.

    Returns
    -------
    fdc : pandas.DataFrame
        Updated dataframe with prediction values and Q-values.
    """
    assert model_type == "ag", 'model_type must be "autogluon"'
    print("Scoring IDs")
    
    fdc["frac_lib_int_unique"] = fdc.apply(lambda row: compute_fraction(row["frag_int"], row["unique_frag_mz"], row["frag_mz"]), axis=1)

    fdc["decoy"] = np.array(["Decoy" in i for i in fdc["seq"]])
    
    fdc['y_count'] = fdc['frag_names'].str.count('y')
    fdc['b_count'] = fdc['frag_names'].str.count('b')
    fdc['yb_count'] = fdc['b_count'] + fdc['y_count']
    fdc['y_minus_b_count'] = fdc['b_count'] - fdc['y_count']
    fdc['y_fraction'] = fdc['y_count'] / (fdc['b_count'] + fdc['y_count'])
    fdc['coeff_MS1_Int_ratio'] = np.log10(fdc['coeff'] + 1) - np.log10(fdc['MS1_Int'] + 1)
    fdc['coeff_x_MS1_Int'] = np.log10(fdc['coeff'] + 1) + np.log10(fdc['MS1_Int'] + 1)
    fdc['abs_rt_error'] = np.abs(fdc['rt_error']+0.0001)
    fdc['abs_mz_error'] = np.abs(fdc['mz_error']+0.0000000001)
    fdc["unique_obs_int_counts"] = fdc["unique_obs_int"].apply(lambda x: len(x.split(";")) if pd.notna(x) and x.strip() != "" else 0)
    fdc["obs_int_counts"] = fdc["obs_int"].apply(lambda x: len(x.split(";")) if pd.notna(x) and x.strip() != "" else 0)
    fdc["ratio_uniqueObs_TotalObs"] = fdc['unique_obs_int_counts']/(fdc['unique_obs_int_counts']+fdc['obs_int_counts'])
    fdc['mass'] = fdc['z']*fdc['mz']
    fdc['ends_with_K'] = fdc['stripped_seq'].apply(lambda x: 1 if x.endswith('K') else 0)
    
    if 'time_channel' in fdc.columns:
        results = []
        for time_channel in fdc['time_channel'].unique():
            print(f"Processing time_channel: {time_channel}")
            subset_fdc = fdc[fdc['time_channel'] == time_channel].copy()
            subset_fdc_reset = subset_fdc.reset_index()
            results.append(process_time_channel(subset_fdc_reset, model_type, fdr_t, folder, time_channel))
        return pd.concat(results)
    else:
        return score_precursors_JD_works_02202025(fdc)




def process_time_channel(subset_fdc, model_type, fdr_t, folder, time_channel):
    """ Function to process each time_channel subset. """

    
    if config.args.plexDIA:
        subset_fdc = plexDIA_feature_engineering(subset_fdc)
    
    num_decoys = subset_fdc["decoy"].sum()  
    num_targets = len(subset_fdc) - num_decoys  
    target_weight = num_decoys / num_targets if num_targets > 0 else 1.0  
    
    subset_fdc["sample_weight"] = subset_fdc["decoy"].apply(lambda x: 1.0 if x else target_weight)
    mean_abs_rt_error = subset_fdc['abs_rt_error'].mean()
    subset_fdc["sample_weight"] = subset_fdc["sample_weight"]*(subset_fdc["abs_rt_error"]+mean_abs_rt_error)#*(1/dat_keep["frac_lib_int"])#*dat_keep["abs_mz_error"]

    plexDIA_features = (
        "abs_diff_rt_from_median", "diff_coeff_from_median", "diff_frac_int_uniq_pred_from_median", "diff_frac_dia_int_from_median",
        "abs_diff_mz_error_from_median", "abs_diff_frac_int_uniq_from_median", "diff_frac_lib_int_from_median",
        "num_channels_greater0_coeff", "num_channels_greater0_frac_int_uniq_pred", "num_channels_greater0_frac_dia_int", 
        "num_channels_greater0_frac_int_uniq", "num_channels_greater0_frac_lib_int",
        "channels_matched"
    )
    
    keep_features = (
        # "iso_cor", "cosine" (more IDs but higher FDR)
        "traceproduct", "iso1_cor", "iso2_cor", "sq_mz_error", "sq_rt_error", "pep_len", "mz",
        "frac_int_matched_pred_sigcoeff", "frac_int_matched_pred", "hyperscore", "frac_int_uniq_pred",
        "frac_int_uniq", "prec_r2", "spec_r2", "frac_int_matched", "rt_error", "mz_error", "frac_dia_int", "frac_lib_int",
        "num_lib", "rt", "z", "coeff", "decoy", "y_count", "b_count", "yb_count", "y_minus_b_count",
        "y_fraction", "mass", "ends_with_K", "abs_rt_error", "sample_weight",
        "coeff_MS1_Int_ratio",'unique_obs_int_counts', 'obs_int_counts','ratio_uniqueObs_TotalObs'
    )
    
    # Conditionally combine features
    combined_features = keep_features + (plexDIA_features if config.args.plexDIA else ())
    
    # Filter DataFrame based on selected features
    dat_keep = subset_fdc[list(combined_features)]  # Convert tuple to list for indexing

    
    kde_decoy = gaussian_kde(dat_keep[dat_keep['decoy'] == True]['rt'])
    kde_non_decoy = gaussian_kde(dat_keep[dat_keep['decoy'] == False]['rt'])
    
    density_decoy = kde_decoy(dat_keep['rt'])
    density_non_decoy = kde_non_decoy(dat_keep['rt'])
    density_non_decoy = np.where(density_non_decoy == 0, 1e-10, density_non_decoy)
    dat_keep = dat_keep.copy()
    dat_keep.loc[:,'density_ratio'] = density_decoy / density_non_decoy
    dat_keep.loc[:,'density_diff'] = density_decoy - density_non_decoy
    
    y = np.array(~dat_keep["decoy"], dtype=int)
    X_temp = dat_keep.drop(columns=['decoy']).fillna(0).astype({col: 'float32' for col in dat_keep.select_dtypes('float64').columns})
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X_temp)
    X = pd.DataFrame(X, columns=X_temp.columns)
    X['label'] = y  

    hyperparams = {
        'NN_TORCH': {},
        'FASTAI': {},
        'GBM': {},
        'CAT': {},
    }
    
    predictor = TabularPredictor(label='label', eval_metric='precision', sample_weight="sample_weight").fit(
        train_data=X, 
        num_bag_folds=10, 
        presets='high_quality', 
        hyperparameters=hyperparams, 
        time_limit=900
    )
    
    oof_predictions = predictor.predict_proba_oof()
    output = oof_predictions.iloc[:, 1]
    
    fpr, tpr, _ = roc_curve(y, output)
    score_order = np.argsort(-output)
    orig_order = np.argsort(score_order)
    decoy_order = subset_fdc["decoy"][score_order]
    frac_decoy = np.cumsum(decoy_order)/np.arange(1,len(decoy_order)+1)
    T = output[score_order[np.searchsorted(frac_decoy, fdr_t)]]
    
    #print("#IDs at 1% FDR:", np.sum(output > T))
    above_t = output > T
    
    subset_fdc["PredVal"] = output
    subset_fdc["Qvalue"] = frac_decoy[orig_order]
    print("#IDs at 1% FDR: ",(subset_fdc["Qvalue"] < 0.01).sum())

    if folder:
        plt.subplots()
        y_log=False
        vals,bins,_ = plt.hist(output,50,log=y_log,label="All")
        plt.hist(output[y.astype(bool)],bins,alpha=.5,log=y_log,label="Targets")
        plt.hist(output[~y.astype(bool)],bins,alpha=.5,log=y_log,label="Decoys")
        plt.legend()
        plt.title(f"Model Score - {time_channel}")
        plt.vlines(T,0,max(vals))
        plt.savefig(f"{folder}/ModelScore_TP{time_channel}.png", dpi=600, bbox_inches="tight")
        
        
        
        feat = 'rt_error'
        func = np.array#np.log10#
        plt.subplots()
        vals,bins,_ = plt.hist(func([i for i in subset_fdc[feat]]),40,label="All")
        # plt.hist([],[])
        vals,bins,_ = plt.hist(func([i for i in subset_fdc[feat][above_t]]),bins,alpha=.5,label=">Threshold")
        vals,bins,_ = plt.hist(func([i for i in subset_fdc[feat][~above_t]]),bins,alpha=.5,label="<Threshold")
        plt.xlabel(feat)
        plt.ylabel("Frequency")
        plt.title(f"RT Error... TP{time_channel}")
        plt.legend()
        plt.savefig(f"{folder}/RT_Error_TP{time_channel}.png", dpi=600, bbox_inches="tight")
        
                
        feat = 'mz_error'
        func = np.array#np.log10#
        plt.subplots()
        vals,bins,_ = plt.hist(func([i for i in subset_fdc[feat]]),40,label="All")
        # plt.hist([],[])
        vals,bins,_ = plt.hist(func([i for i in subset_fdc[feat][above_t]]),bins,alpha=.5,label=">Threshold")
        vals,bins,_ = plt.hist(func([i for i in subset_fdc[feat][~above_t]]),bins,alpha=.5,label="<Threshold")
        plt.xlabel(feat)
        plt.ylabel("Frequency")
        plt.title(f"MZ Error... TP{time_channel}")
        plt.legend()
        plt.savefig(f"{folder}/MZ_Error_TP{time_channel}.png", dpi=600, bbox_inches="tight")
    
    # # Plot results if folder is specified
    # if folder:
    #     plt.figure()
    #     plt.hist(output, 50, label="All")
    #     plt.hist(output[y.astype(bool)], alpha=0.5, label="Targets")
    #     plt.hist(output[~y.astype(bool)], alpha=0.5, label="Decoys")
    #     plt.legend()
    #     plt.title(f"Model Score - {time_channel}")
    #     plt.savefig(f"{folder}/ModelScore_TP{time_channel}.png", dpi=600, bbox_inches="tight")
            
    #     for feat in ['rt_error', 'mz_error']:
    #         plt.figure()
    #         plt.hist(subset_fdc[feat], bins=40, label="All")
    #         plt.hist(subset_fdc[feat][above_t], alpha=0.5, label=">Threshold")
    #         plt.hist(subset_fdc[feat][~above_t], alpha=0.5, label="<Threshold")
    #         plt.xlabel(feat)
    #         plt.ylabel("Frequency")
    #         plt.legend()
    #         plt.title(f"{feat} - {time_channel}")
    #         plt.savefig(f"{folder}/{feat}_error_TP{time_channel}.png", dpi=600, bbox_inches="tight")

    #checking feature importance from training data which should probably be ok, but test data might be better. Though, this will show what the model(s) used to fit the actual data.

    X_sample = X.sample(frac=0.3, random_state=123) #speed up by only taking 30% of data for feature importance computations
    feature_importance = predictor.feature_importance(X_sample)
    feature_importance = feature_importance.sort_values(by="importance", ascending=False)
        
    print("Feature Importances:")
    print(feature_importance)
    # Plot results if folder is specified
    if folder:
        plt.figure(figsize=(11, 8.5))
        plt.barh(feature_importance.index, feature_importance["importance"], color='blue')
        plt.xlabel("Importance")
        plt.ylabel("Features")
        plt.title("AutoGluon feature importances")
        plt.gca().invert_yaxis()
        plt.savefig(folder + f"/Feature_Importance_AG_TP{time_channel}.png", dpi=600, bbox_inches="tight")
    
    return subset_fdc


def process_data(file,spectra,library,mass_tag=None,timeplex=False):
    
    results_folder = os.path.dirname(file)
    mz_ppm = config.opt_ms1_tol
    rt_tol = config.opt_rt_tol
    
    lp,fdc,dc = get_large_prec(file,condense_output=False,timeplex=timeplex)
    
    
    
    ## Add additional features
    # X["prec_z"] = fdc["z"]
    fdc["stripped_seq"] = np.array([re.sub("Decoy_","",re.sub("\(.*?\)","",i)) for i in fdc["seq"]])
    fdc["pep_len"] = [len(re.findall("([A-Z](?:\(.*?\))?)",re.sub("Decoy","",i))) for i in fdc["stripped_seq"]]
    # X["rt"] = fdc["rt"]
    # X["coeff"] = fdc["coeff"]
    fdc["sq_rt_error"] = np.power(fdc["rt_error"],2)
    fdc["sq_mz_error"] = np.power(fdc["mz_error"],2)
    
    
    fdx = ms1_quant(fdc, lp, dc, mass_tag, spectra, mz_ppm, rt_tol, timeplex)
    fdx = score_precursors(fdx,config.score_model,config.fdr_threshold,folder=results_folder)
    
    fdx["untag_prec"] = ["_".join([i[0],str(int(i[1]))]) for i in zip(fdx["untag_seq"],fdx["z"])]

    if timeplex:
        if mass_tag:
            tag_name = mass_tag.name
            fdx["channel"] = [str(int(t))+"_"+re.findall(f"{tag_name}-(\d+)",i)[0] for i,t in zip(fdx.seq,fdx.time_channel)]
        else:
            fdx["channel"] = fdx["time_channel"]
            
    elif mass_tag:
        tag_name = mass_tag.name
        ## mTRAQ label
        fdx["channel"] = [int(re.findall(f"{tag_name}-(\d+)",i)[0]) for i in fdx.seq]
    
    

    fdx["last_aa"] = [i[-1] for i in fdx["stripped_seq"]]
    fdx["seq_len"] = [len(i) for i in fdx["stripped_seq"]]
    
    # have possible reannotate woth fasta here
    # fdx["org"] = np.array([";".join(orgs[[i in all_fasta_seqs[j] for j in range(3)]]) for i in fdx["stripped_seq"]])
    
    
    channel_matches_counts = fdx["untag_prec"].value_counts()
    channel_matches_counts_dict = {i:j for i,j in zip(channel_matches_counts.index,channel_matches_counts)}
    fdx["channels_matched"] = [channel_matches_counts_dict[i] for i in fdx["untag_prec"]]

    
    ## save to results folder
    fdx.to_csv(results_folder+"/all_IDs.csv",index=False)
    fdx[np.logical_and(~fdx["decoy"],fdx["Qvalue"]<config.fdr_threshold)].to_csv(results_folder+"/filtered_IDs.csv",index=False)
