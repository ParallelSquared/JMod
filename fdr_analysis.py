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
from miscFunctions import fragment_cor,unstring_floats

import config


def area(x):max_idx = np.argmax(x);top_3 = x[np.maximum(0,max_idx-1):max_idx+2];return np.sum(top_3)#auc(range(len(top_3)),top_3)


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


def ms1_quant(dat,lp,dc,mass_tag,DIAspectra,mz_ppm,rt_tol,timeplex=False):
    # X = fdc.iloc[:,6:-5]
   
    print("Performing MS1 Quantitation") 
    
    fdc = dat[dat["decoy"] == False].copy().reset_index(drop=True)  #remove decoys
    
    #only quantify confident precs
    if config.args.unfiltered_quant: #this will not execute if you specificy --unfiltered_quant (inherently stored as false)
        fdc = fdc[fdc["BestChannel_Qvalue"] < 0.01].reset_index(drop=True)

    if timeplex:
        all_keys = [(i,j,k) for i,j,k in zip(fdc.seq,fdc.z,fdc.time_channel)]
    else:
        all_keys = [(i,j) for i,j in zip(fdc.seq,fdc.z)]


        
    if mass_tag:
        #fdc["untag_seq"] = [re.sub(f"(\({mass_tag.name}-\d+\))?","",peptide) for peptide in fdc["seq"]]
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
        
        fdc["plexfittrace_spec_all"] = [";".join(map(str,j)) for i,j,k,p in zip(extracted_fitted,extracted_fitted_specs,ms2_traces,extracted_fitted_p)]
        fdc["plexfittrace_all"] = [";".join(map(str,i)) for i,j,k,p in zip(extracted_fitted,extracted_fitted_specs,ms2_traces,extracted_fitted_p)]
        fdc["plexfittrace_ps_all"] = [";".join(map(str,[pi.statistic if pi==pi else np.isnan for pi in p])) for i,j,k,p in zip(extracted_fitted,extracted_fitted_specs,ms2_traces,extracted_fitted_p)]
        fdc["plex_Area"]=[area(list(map(float,fdc.plexfittrace.iloc[idx].split(";")))) for idx in range(len(fdc))]

    else:
        #fdc["untag_seq"] = fdc["seq"]
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
    
    fdc["MS1_Area"]=[auc(list(map(float,fdc.all_ms1_specs.iloc[idx].split(";"))),list(map(float,fdc.all_ms1_iso0vals.iloc[idx].split(";")))) for idx in range(len(fdc))]


        # Define selected columns that we want to merge
    selected_cols = [
        "plexfitMS1", "plexfitMS1_p", "plexfittrace", "plexfit_ps",
        "plexfittrace_spec_all", "plexfittrace_all", "plexfittrace_ps_all",
        "plex_Area", "ms1_cor", "traceproduct", "iso_cor", "MS1_Int",
        "all_ms1_specs", "MS1_Area"
    ]
    
    # Ensure we only select columns that actually exist in fdc
    existing_cols = [col for col in selected_cols if col in fdc.columns]
    
    # Perform the merge safely
    dat = dat.merge(fdc[["untag_prec", "channel"] + existing_cols], how="left", on=["untag_prec", "channel"]).fillna(0)


    return dat



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
                
    def run_model(self,X,y,sample_weight=None):
        print("test")
        if self.model_type=="rf":
            
            ### Random Forest
            def fit_model(X,y,sample_weight,idx=""):
                    m = model_instance(model_type=self.model_type)
                    m.model = RandomForestClassifier(n_estimators = 200,max_depth=config.tree_max_depth,n_jobs=-1)
                    m.model.fit(X,y,sample_weight=sample_weight)
                    m.__predict_fn__ = m.model.predict_proba

                    if self.folder:
                        feature_importance = m.model.feature_importances_
                        sorted_indices = np.argsort(feature_importance)  
                        sorted_features = np.array(X.columns)[sorted_indices]  
                        sorted_importance = feature_importance[sorted_indices]  
                    
                        fig, ax = plt.subplots(figsize=(8, len(X.columns)*0.3))                    
                        ax.barh(sorted_features, sorted_importance)
                        ax.set_title("Feature Importance")
                    
                        # Save plot
                        plt.savefig(self.folder + f"/RF{idx}_feature_importance.png", dpi=600, bbox_inches="tight")
                    
                    return m
                
            # self.model = fit_model(X,y)
            
        
        elif self.model_type=="lda":
            
            ## Linear Disriminant Analysis
            def fit_model(X,y,sample_weight,idx=""):
                    m = model_instance(model_type=self.model_type)
                    m.model = LinearDiscriminantAnalysis()
                    m.model.fit(X,y)
                    m.__predict_fn__ = m.model.predict_proba
                    return m
                
            # self.model = fit_model(X,y)
            
            
        elif self.model_type == "xg":
            
            ## XGBoost
            def fit_model(X,y,sample_weight,idx=""):
                    m = model_instance(model_type=self.model_type)
                    dTrain = xgb.DMatrix(X,y,weight=sample_weight)
                    param = {
                        'max_depth': config.tree_max_depth, 
                        'eta': .1, 
                        'objective': 'binary:logistic'}
                    # param['nthread'] = 4
                    param['eval_metric'] = 'pre'
                    
                    m.model = xgb.train(param, dtrain=dTrain,num_boost_round=50)
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
            columns = X.columns
            X = pd.DataFrame(preprocessing.StandardScaler().fit(X).transform(X),columns=columns)
            ## Neural network
            def fit_model(X,y,sample_weight,idx=""):
                    m = model_instance(model_type=self.model_type)
                    # m.model = MLPClassifier((32,16,8,4),activation="relu")
                    m.model = MLPClassifier((8,8,4),activation="relu")
                    m.model.fit(X,y)
                    m.__predict_fn__ = m.model.predict_proba
                    return m
                
        else:
            raise ValueError("Unsupported model type")
            
        kf = KFold(n_splits=self.n_splits,shuffle=True)
        k_orders = [i for i in kf.split(X,y)]
        rev_order = np.argsort(np.concatenate([i[1] for i in k_orders])) # collapse test sets and get order

        if sample_weight is not None:
            data_splits = [[X.iloc[i[0]],X.iloc[i[1]],y[i[0]],y[i[1]],sample_weight[i[0]]] for i in k_orders] # put data into folds
    
        else:
            data_splits = [[X.iloc[i[0]],X.iloc[i[1]],y[i[0]],y[i[1]],None] for i in k_orders] # put data into folds
        

        self.models = []
        self.predictions=[]
        model_idx=0
        for X_train, X_test, y_train, y_test,weights in tqdm.tqdm(data_splits):
            m = fit_model(X_train,y_train,sample_weight=weights,idx=model_idx)
            self.models.append(m)
            self.predictions.append(m.predict(X_test))
            model_idx+=1
            
        return np.concatenate(self.predictions)[rev_order]


def plex_features(fdc):

    if config.args.no_transfer:
            return fdc
    else: 
        print("Using joint information across channels for precursor scoring")
        if config.args.timeplex & config.args.plexDIA:
            fdc["median_rt"] = fdc.groupby(["untag_prec","time_channel"])['rt'].transform("median")
            fdc.loc[fdc["channels_matched"] == 1, "median_rt"] = pd.NA
            fdc["abs_diff_rt_from_median"] = np.abs(fdc['rt'] - fdc['median_rt'])
            fdc["abs_diff_rt_from_median"].fillna(fdc["abs_diff_rt_from_median"].mean(), inplace=True)
    
            fdc["median_rt_run"] = fdc.groupby(["untag_prec"])['rt'].transform("median")
            fdc.loc[fdc["channels_matched"] == 1, "median_rt_run"] = pd.NA
            fdc["abs_diff_rt_from_median_run"] = np.abs(fdc['rt'] - fdc['median_rt_run'])
            fdc["abs_diff_rt_from_median_run"].fillna(fdc["abs_diff_rt_from_median_run"].mean(), inplace=True)
    
        if config.args.timeplex and not config.args.plexDIA: #get difference in RTs between timeplexes
            fdc["median_rt_run"] = fdc.groupby(["untag_prec"])['rt'].transform("median")
            fdc.loc[fdc["channels_matched"] == 1, "median_rt_run"] = pd.NA
            fdc["abs_diff_rt_from_median_run"] = np.abs(fdc['rt'] - fdc['median_rt_run'])
            fdc["abs_diff_rt_from_median_run"].fillna(fdc["abs_diff_rt_from_median_run"].mean(), inplace=True)
    
        if config.args.plexDIA and not config.args.timeplex: #get difference in RTs wihtin a plex
            fdc["median_rt"] = fdc.groupby(["untag_prec"])['rt'].transform("median")
            fdc.loc[fdc["channels_matched"] == 1, "median_rt"] = pd.NA
            fdc["abs_diff_rt_from_median"] = np.abs(fdc['rt'] - fdc['median_rt'])
            fdc["abs_diff_rt_from_median"].fillna(fdc["abs_diff_rt_from_median"].mean(), inplace=True)
            
        
        fdc["median_abs_rt_error"] = fdc.groupby(["untag_prec"])["rt_error"].transform(lambda x: np.abs(x).median())
    
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
        
        #fdc["channels_matched_seq"] = fdc.groupby("untag_seq")["untag_seq"].transform("count")
        # fdc["summed_coeff"] = fdc.groupby("untag_prec")['coeff'].transform("sum")
        # fdc["summed_frac_int_uniq_pred"] = fdc.groupby("untag_prec")['coeff'].transform("sum")
        # fdc["summed_frac_dia_int"] = fdc.groupby("untag_prec")['frac_dia_int'].transform("sum")
        # fdc["summed_mz_error"] = fdc.groupby("untag_prec")['frac_dia_int'].transform("sum")
        # fdc["summed_frac_int_uniq"] = fdc.groupby("untag_prec")['frac_int_uniq'].transform("sum")
        # fdc["summed_frac_lib_int"] = fdc.groupby("untag_prec")['frac_lib_int'].transform("sum")

        # fdc["summed_manhattan_distances"] = fdc.groupby("untag_prec")['manhattan_distances'].transform("sum")
        # fdc["summed_gof_stats"] = fdc.groupby("untag_prec")['gof_stats'].transform("sum")
        # fdc["summed_max_matched_residuals"] = fdc.groupby("untag_prec")['max_matched_residuals'].transform("sum")
        # fdc["summed_scribe_scores"] = fdc.groupby("untag_prec")['scribe_scores'].transform("sum")

        return fdc
    
    
def score_precursors(fdc,model_type="rf",fdr_t=0.01, folder=None):
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

    assert model_type in ["lda", "rf", "xg"], 'model_type must be one of ["lda", "rf", "xg"]'
    
    print("Scoring IDs")
    

    if config.args.timeplex or config.args.plexDIA:
        fdc = plex_features(fdc)
    
    ## We consider decoys and targets with v small coeffs to be from the null distributiom
    _bool = np.logical_and(~fdc["decoy"],fdc.coeff>1)
    
    ## define our features and labels for the model
    y = np.array(_bool,dtype=int)
    
    # exclude necessary columns
    drop_colums = ['spec_id', 'Ms1_spec_id', 'seq', 'window_mz','frag_names', 'frag_errors', 'frag_mz', 'frag_int', 'obs_int', 'stripped_seq', 
                  'untag_seq', 'decoy','all_ms1_specs', 'all_ms1_iso0vals', 'all_ms1_iso1vals', 'all_ms1_iso2vals','all_ms1_iso3vals', 'all_ms1_iso4vals', 
                  'all_ms1_iso5vals','all_ms1_iso6vals','all_ms1_iso7vals',"plexfittrace","plexfit_ps","untag_prec","plexfittrace_spec_all","plexfittrace_all",
                  "plexfittrace_ps_all",
                  "unique_frag_mz", "untag_prec",
                  "channels_matched",
                  "unique_obs_int", 'MS1_Int',"MS1_Area", "iso_cor", "cosine", "traceproduct","iso1_cor","iso2_cor","ms1_cor","plexfitMS1","plexfitMS1_p","plex_Area", "untag_prec","channel","time_channel",
                  "unique_frag_mz",
                  "unique_obs_int",
                  "file_name",
                  "protein"]
    X = fdc.drop([c for c in drop_colums if c in fdc.columns], axis=1)
    # print(X.columns)
    X[np.isnan(X)]=0 ## set nans to zero (mostly for r2 values)
        
    sc_model = score_model(model_type,folder=folder)
    pred = sc_model.run_model(X, y)
    
    model_name= model_type

    ####### make sure to not have these in the model (bc they are left out in decoys) #######
       # "plexfitMS1", "plexfitMS1_p", "plexfittrace", "plexfit_ps","plexfittrace_spec_all","plexfittrace_all","plexfittrace_ps_all","plex_Area","ms1_cor","traceproduct","iso_cor","MS1_Int","all_ms1_specs","MS1_Area"
        
    ###############################################################################################
    ########################  Analysis the predictions    ######################################
    
    
    
    
    if len(pred.shape)==2:
        output = pred[:,1]
    else:
        output = pred
        
        
        
    ## Use the scores to estimate the #IDs as 1% FDR
    
    fpr, tpr, _ = roc_curve(y, output)
    # plt.subplots()
    # plt.plot(fpr,tpr)
    # print("AUC: ",np.round(auc(fpr,tpr),3))
    
    
    
    # ordered_scores = sorted(output)[::-1]
    
    ## note this is slow
    ## count down to find optimal FDR but then just use every Nth score to get a nice plot
    # fdr = []
    # threshold = []
    # interval = 1
    # for idx,s in enumerate(tqdm.tqdm(ordered_scores)):
    #     if idx%interval==0:
    #         ## SCORES IN INCREASING ORDER
    #         # fdr.append(np.sum(np.greater_equal(pred[~y.astype(bool),1],s))/np.sum(np.greater_equal(pred[:,1],s)))
            
    #         # Scores decreasing
    #         val = np.sum(np.greater_equal(output[~y.astype(bool)],s))/np.sum(np.greater_equal(output,s))
    #         fdr.append(val)
    #         if val<.01:
    #             # if threshold==[]:
    #             threshold = [ordered_scores[idx-1],s]
    #         else:
    #             interval=10
    
    
    ## FASTER VERSION OF ABOVE
    score_order = np.argsort(-output)
    orig_order = np.argsort(score_order)
    decoy_order = fdc["decoy"][score_order]
    frac_decoy = np.cumsum(decoy_order)/np.arange(1,len(decoy_order)+1)
    # plt.plot(frac_decoy)
    T = output[score_order[np.searchsorted(frac_decoy,0.01)]]

    print()
    print("#IDs at 1% FDR:", np.sum(output>T))
    
    above_t = output>T
    fdc["PredVal"] = output
    fdc["Qvalue"] = frac_decoy[orig_order]
    
    if folder:
        
        plt.subplots()
        y_log=False
        vals,bins,_ = plt.hist(output,50,log=y_log,label="All")
        plt.hist(output[y.astype(bool)],bins,alpha=.5,log=y_log,label="Targets")
        plt.hist(output[~y.astype(bool)],bins,alpha=.5,log=y_log,label="Decoys")
        plt.legend()
        plt.title(model_name+ f" - Type {config.unmatched_fit_type}")
        plt.vlines(T,0,max(vals))
        plt.savefig(folder+"/ModelScore.png",dpi=600,bbox_inches="tight")
        
        
        
        feat = 'rt_error'
        func = np.array#np.log10#
        plt.subplots()
        vals,bins,_ = plt.hist(func([i for i in fdc[feat]]),40,label="All")
        # plt.hist([],[])
        vals,bins,_ = plt.hist(func([i for i in fdc[feat][above_t]]),bins,alpha=.5,label="1%FDR")
        vals,bins,_ = plt.hist(func([i for i in fdc[feat][np.logical_and(~above_t,~fdc.decoy)]]),bins,alpha=.5,label="Low Scoring")
        vals,bins,_ = plt.hist(func([i for i in fdc[feat][fdc.decoy]]),bins,alpha=.5,label="Decoy")
        plt.xlabel(feat)
        plt.ylabel("Frequency")
        plt.title(model_name+ f" - Type {config.unmatched_fit_type}")
        plt.legend()
        plt.savefig(folder+"/RT_error.png",dpi=600,bbox_inches="tight")
        
                
        feat = 'mz_error'
        func = np.array#np.log10#
        plt.subplots()
        vals,bins,_ = plt.hist(func([i for i in fdc[feat]]),40,label="All")
        # plt.hist([],[])
        vals,bins,_ = plt.hist(func([i for i in fdc[feat][above_t]]),bins,alpha=.5,label="1%FDR")
        vals,bins,_ = plt.hist(func([i for i in fdc[feat][np.logical_and(~above_t,~fdc.decoy)]]),bins,alpha=.5,label="Low Scoring")
        vals,bins,_ = plt.hist(func([i for i in fdc[feat][fdc.decoy]]),bins,alpha=.5,label="Decoy")
        plt.xlabel(feat)
        plt.ylabel("Frequency")
        plt.title(model_name+ f" - Type {config.unmatched_fit_type}")
        plt.legend()
        plt.savefig(folder+"/mz_error.png",dpi=600,bbox_inches="tight")
    
    return fdc



def compute_protein_FDR(df):
    print("Computing Protein FDR")

  
    df["run_chan"] = df["file_name"].astype(str) + df["channel"].astype(str)
    df_seqchargeqvals = df[df["Qvalue"] < 0.01].copy().reset_index(drop=True) #filter
    df_seqchargeqvals["maxPredval"] = df_seqchargeqvals.groupby(["protein", "decoy"])["PredVal"].transform("max")
    df_seqchargeqvals = df_seqchargeqvals.drop_duplicates(subset=["protein", "decoy"]).reset_index(drop=True)
    
    # Rank by descending maxPredval and compute accum_decoys & Protein_Qvalue
    df_seqchargeqvals = df_seqchargeqvals.sort_values(by="maxPredval", ascending=False).reset_index(drop=True)
    df_seqchargeqvals["prot_rank"] = df_seqchargeqvals.index + 1  # Equivalent to row_number()
    df_seqchargeqvals["accum_decoys"] = df_seqchargeqvals["decoy"].cumsum()
    df_seqchargeqvals["Protein_Qvalue"] = df_seqchargeqvals["accum_decoys"] / df_seqchargeqvals["prot_rank"]
    
    # Filter for non-decoy proteins and select distinct protein values
    df_seqchargeqvals_distinct = (
        df_seqchargeqvals[df_seqchargeqvals["decoy"] == False]
        .drop_duplicates(subset=["protein"])
        [["protein", "Protein_Qvalue"]]
    )
    
    df = df.drop(columns=["Protein_Qvalue"], errors="ignore")
    df = df.merge(df_seqchargeqvals_distinct, on="protein", how="left")
        
    df_counts_prec = (
        df[(df["decoy"] == False) & (df["Qvalue"] < 0.01)]
        .drop_duplicates(subset=["run_chan", "untag_prec"])
        .groupby(["file_name", "channel"])
        .size()
        .reset_index(name="Precursor_IDs")
    )
    print("Number of precursors at 1% FDR:")
    print("All Channels:",np.sum(df_counts_prec.Precursor_IDs))
    print(df_counts_prec.to_string(index=False))
    

    df_counts_prots = (
        df[(df["Protein_Qvalue"] < 0.01) & (df["decoy"] == False) & (df["Qvalue"] < 0.01)]
        .drop_duplicates(subset=["run_chan", "protein"])
        .groupby(["run_chan","channel"])
        .size()
        .reset_index(name="Protein_IDs")
    )
    print("\nNumber of proteins at 1% FDR:")
    print("All Channels:",np.sum(df_counts_prots.Protein_IDs))
    print(df_counts_prots.to_string(index=False))


    
    # if config.args.plexDIA:
    #     if config.args.timeplex:
    #         df["BestChannel_Protein_Qvalue"] = df.groupby(["time_channel", "protein", "decoy"])["Protein_Qvalue"].transform("min")
    #     else:
    #         df["BestChannel_Protein_Qvalue"] = df.groupby(["file_name", "protein", "decoy"])["Protein_Qvalue"].transform("min")

    if config.args.plexDIA:
        print("\nAfter plexDIA identification propagation based on best channel Q-value:")
        
        # Compute number of precursor IDs at 1% FDR
        df_counts_prec = (
            df[(df["decoy"] == False) & (df["BestChannel_Qvalue"] < 0.01)]
            .drop_duplicates(subset=["run_chan", "untag_prec"])
            .groupby(["file_name", "channel"])
            .size()
            .reset_index(name="Precursor_IDs")
        )
        
        # Print precursor ID counts
        print("Number of precursors at 1% FDR (best channel):")
        print("All Channels:",np.sum(df_counts_prec.Precursor_IDs))
        print(df_counts_prec.to_string(index=False))
        
        # Compute number of protein IDs at 1% FDR
        df_counts_prots = (
            df[(df["Protein_Qvalue"] < 0.01) & (df["decoy"] == False) & (df["BestChannel_Qvalue"] < 0.01)]
            .drop_duplicates(subset=["run_chan", "protein"])
            .groupby(["run_chan", "channel"])
            .size()
            .reset_index(name="Protein_IDs")
        )
        
        # Print protein ID counts
        print("\nNumber of proteins at 1% FDR (best channel):")
        print("All Channels:",np.sum(df_counts_prots.Protein_IDs))
        print(df_counts_prots.to_string(index=False))


    return df



def process_data(file,spectra,library,mass_tag=None,timeplex=False):
    
    results_folder = os.path.dirname(file)
    mz_ppm = config.opt_ms1_tol
    rt_tol = config.opt_rt_tol
    
    lp,fdc,dc = get_large_prec(file,condense_output=False,timeplex=timeplex)
    
    # if "seq" not in fdc.columns:
    #     raise KeyError("Column 'seq' is missing in fdc. Check data loading step.")

    
    ## Add additional features
    # X["prec_z"] = fdc["z"]
   # print(fdc.columns)  # Ensure 'seq' is in fdc

    fdc["stripped_seq"] = np.array([re.sub("Decoy_","",re.sub("\(.*?\)","",i)) for i in fdc["seq"]])
    fdc["pep_len"] = [len(re.findall("([A-Z](?:\(.*?\))?)",re.sub("Decoy","",i))) for i in fdc["stripped_seq"]]
    # X["rt"] = fdc["rt"]
    # X["coeff"] = fdc["coeff"]
    fdc["sq_rt_error"] = np.power(fdc["rt_error"],2)
    fdc["sq_mz_error"] = np.power(fdc["mz_error"],2)

    if mass_tag:
        fdc["untag_seq"] = [re.sub(f"(\({mass_tag.name}-\d+\))?","",peptide) for peptide in fdc["seq"]]
    else:
        fdc["untag_seq"] = fdc["seq"]
    #print(fdc.columns)  # Ensure 'seq' is in fdc

       
    fdc["untag_prec"] = ["_".join([i[0],str(int(i[1]))]) for i in zip(fdc["untag_seq"],fdc["z"])]
    channel_matches_counts = fdc["untag_prec"].value_counts()
    channel_matches_counts_dict = {i:j for i,j in zip(channel_matches_counts.index,channel_matches_counts)}
    fdc["channels_matched"] = [channel_matches_counts_dict[i] for i in fdc["untag_prec"]]

    
    if timeplex:
        if mass_tag:
            tag_name = mass_tag.name
            fdc["channel"] = [str(int(t))+"_"+re.findall(f"{tag_name}-(\d+)",i)[0] for i,t in zip(fdc.seq,fdc.time_channel)]
        else:
            fdc["channel"] = fdc["time_channel"]
            
    elif mass_tag:
        tag_name = mass_tag.name
        ## mTRAQ label
        fdc["channel"] = [int(re.findall(f"{tag_name}-(\d+)",i)[0]) for i in fdc.seq]

    else: 
        fdc["channel"] = 0 #if LF

    #this was previously in ms1_quant function.. we need it for the target/decoy classification
    frag_errors = [unstring_floats(mz) for mz in fdc.frag_errors]
    median  = np.median(np.concatenate([i for i in frag_errors]))
    fdc["med_frag_error"] = [np.median(np.abs(median-i)) for i in frag_errors]

    ## What precursors are labeled as decoys
    fdc["decoy"] = np.array(["Decoy" in i for i in fdc["seq"]])

    
    minfraclib_toscore = getattr(config.args, "score_lib_frac", 0) 
    fdx_toscore = fdc[fdc['frac_lib_int'].fillna(0) >= minfraclib_toscore].reset_index(drop=True)
    
    fin = score_precursors(fdx_toscore,config.score_model,config.fdr_threshold,folder=results_folder)
    new_columns = [col for col in fin.columns if col not in fdc.columns and col not in ["untag_prec", "channel"]]
    fdx = fdc.merge(fin[["untag_prec", "channel"] + new_columns], how="left", on=["untag_prec", "channel"])

    ##fill NA's appropriately
    fdx['PredVal'] = fdx['PredVal'].fillna(0)  
    fdx['Qvalue'] = fdx['Qvalue'].fillna(1)     


    # if config.args.plexDIA:
    #     if config.args.timeplex:
    #         fdx["BestChannel_Qvalue"] = fdx.groupby(["time_channel", "untag_prec", "decoy"])["Qvalue"].transform("min") #within a plexDIA set for each timechannel
    #     else:
    #         fdx["BestChannel_Qvalue"] = fdx.groupby(["file_name", "untag_prec", "decoy"])["Qvalue"].transform("min") #within a plexDIA set
    
    if config.args.plexDIA or config.args.timeplex:
        fdx["BestChannel_Qvalue"] = fdx.groupby(["file_name", "untag_prec", "decoy"])["Qvalue"].transform("min") #within a run
    else:
        fdx["BestChannel_Qvalue"] = fdx["Qvalue"] #applies to no plex

    
    fdx_quant = ms1_quant(fdx, lp, dc, mass_tag, spectra, mz_ppm, rt_tol, timeplex)


    fdx_quant["last_aa"] = [i[-1] for i in fdx_quant["stripped_seq"]]
    fdx_quant["seq_len"] = [len(i) for i in fdx_quant["stripped_seq"]]
    
    # have possible reannotate woth fasta here
    # fdx["org"] = np.array([";".join(orgs[[i in all_fasta_seqs[j] for j in range(3)]]) for i in fdx["stripped_seq"]])
    fdx_quant = compute_protein_FDR(fdx_quant)

    
    ## save to results folder
    fdx_quant.to_csv(results_folder+"/all_IDs.csv",index=False)
    fdx_quant[np.logical_and(~fdx_quant["decoy"],fdx_quant["BestChannel_Qvalue"]<config.fdr_threshold)].to_csv(results_folder+"/filtered_IDs.csv",index=False)
