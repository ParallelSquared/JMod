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
    
    
    
def score_precursors(fdc,model_type="rf",fdr_t=0.01, folder=None):
    """
    

    Parameters
    ----------
    fdc : pandas.Dataframe
        All PSMs identified.
    model_type : string [lda, rf, xg]
                 Type of ML model used to discriminate targets and decoys
                 
    Returns
    -------
    None.

    """

    assert model_type in ["lda", "rf", "xg"], 'model_type must be one of ["lda", "rf", "xg"]'
    
    print("Scoring IDs")
    
    ## What precursors are labeled as decoys
    fdc["decoy"] = np.array(["Decoy" in i for i in fdc["seq"]])
    
    
    ## We consider decoys and targets with v small coeffs to be from the null distributiom
    _bool = np.logical_and(~fdc["decoy"],fdc.coeff>1)
    
    ## define our features and labels for the model
    y = np.array(_bool,dtype=int)
    
    # exclude necessary columns
    drop_colums = ['spec_id', 'Ms1_spec_id', 'seq', 'window_mz','frag_names', 'frag_errors', 'frag_mz', 'frag_int', 'obs_int', 'stripped_seq', 
                  'untag_seq', 'decoy','all_ms1_specs', 'all_ms1_iso0vals', 'all_ms1_iso1vals', 'all_ms1_iso2vals','all_ms1_iso3vals', 'all_ms1_iso4vals', 
                  'all_ms1_iso5vals','all_ms1_iso6vals','all_ms1_iso7vals',"plexfittrace","plexfit_ps","untag_prec",
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
    frac_decoy = 2*np.cumsum(decoy_order)/np.arange(1,len(decoy_order)+1)
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
        vals,bins,_ = plt.hist(func([i for i in fdc[feat][above_t]]),bins,alpha=.5,label=">Threshold")
        vals,bins,_ = plt.hist(func([i for i in fdc[feat][~above_t]]),bins,alpha=.5,label="<Threshold")
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
        vals,bins,_ = plt.hist(func([i for i in fdc[feat][above_t]]),bins,alpha=.5,label=">Threshold")
        vals,bins,_ = plt.hist(func([i for i in fdc[feat][~above_t]]),bins,alpha=.5,label="<Threshold")
        plt.xlabel(feat)
        plt.ylabel("Frequency")
        plt.title(model_name+ f" - Type {config.unmatched_fit_type}")
        plt.legend()
        plt.savefig(folder+"/mz_error.png",dpi=600,bbox_inches="tight")
    
    return fdc



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
