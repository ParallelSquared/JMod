#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 09:38:40 2024

@author: kevinmcdonnell
"""


from read_output import get_large_prec

from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_recall_curve, average_precision_score
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
        
        fdc["plexfittrace_spec_all"] = [";".join(map(str,j)) for i,j,k,p in zip(extracted_fitted,extracted_fitted_specs,ms2_traces,extracted_fitted_p)]
        fdc["plexfittrace_all"] = [";".join(map(str,i)) for i,j,k,p in zip(extracted_fitted,extracted_fitted_specs,ms2_traces,extracted_fitted_p)]
        fdc["plexfittrace_ps_all"] = [";".join(map(str,p)) for i,j,k,p in zip(extracted_fitted,extracted_fitted_specs,ms2_traces,extracted_fitted_p)]
        fdc["plex_Area"]=[area(list(map(float,fdc.plexfittrace.iloc[idx].split(";")))) for idx in range(len(fdc))]

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
    
    fdc["MS1_Area"]=[auc(list(map(float,fdc.all_ms1_specs.iloc[idx].split(";"))),list(map(float,fdc.all_ms1_iso0vals.iloc[idx].split(";")))) for idx in range(len(fdc))]

    frag_errors = [unstring_floats(mz) for mz in fdc.frag_errors]
    median  = np.median(np.concatenate([i for i in frag_errors]))
    fdc["med_frag_error"] = [np.median(np.abs(median-i)) for i in frag_errors]
    

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
    
    def __init__(self, model_type, n_splits=5, folder=None):
        self.model_type = model_type
        self.n_splits = n_splits
        self.folder = folder
                
    def run_model(self,X,y,protein_names):
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
            
        gfk = GroupKFold(n_splits = self.n_splits)
        groups = protein_names #Get the protein column

        #May be an odd scenario where number of proteins is less than number of CV folds 
        n_proteins = len(groups.unique())
        if n_proteins < self.n_splits:
            raise ValueError(f"Number of unique proteins ({n_proteins}) must be >= number of folds ({self.n_splits})")

        #k_orders = [i for i in kf.split(X,y)] old way
        k_orders = [i for i in gfk.split(X, y, groups=groups)]
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

class ModelInstance:
    def __init__(self, model_type):
        self.model_type = model_type
        
    def predict(self, X):
        pred = self.__predict_fn__(X)
        if len(pred.shape) == 2:
            output = pred[:, 1]
        else:
            output = pred
        return output

class ChannelScoreModel:
    def __init__(self, df, model_type, target_channels, decoy_channels, features=None, n_splits=5):
        """
        Initialize the model for cross-validation.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The DataFrame containing all the data
        model_type : str
            The type of model to use ('rf', 'xgb', or 'lda')
        target_channels : list
            List of target channel numbers
        decoy_channels : list
            List of decoy channel numbers
        features : list, optional
            List of feature column names to use
        n_splits : int
            Number of cross-validation folds
        """
        self.df = df  # Store the DataFrame as an instance variable
        self.target_channels = target_channels
        self.decoy_channels = decoy_channels
        self.features = features
        self.model_type = model_type
        self.n_splits = n_splits
        self.models = []
        print(f"Using {model_type} model with {n_splits} cross-validation splits")
        
    def run_model(self):
        """
        Train the model using cross-validation and generate out-of-fold predictions.
        
        Returns:
        --------
        predictions : numpy.ndarray
            Out-of-fold predictions
        true_labels : numpy.ndarray
            True labels corresponding to predictions
        """
        # Create feature matrix and target vector using the stored DataFrame
        X = self.df[self.features]
        y = self.df['is_decoy_channel'].astype(int)
        
        # Replace NaN values with 0
        X = X.fillna(0)
        
        # Use protein as grouping variable for cross-validation
        # Check if there are more than 100 unique protein names
        unique_proteins = self.df['protein'].nunique()
        if unique_proteins < 100:
            print(f"Only {unique_proteins} unique proteins found, which is less than 100. Switching to use 'untag_seq' for grouping in cross-validation.")
            groups = self.df['untag_seq']
        else:
            print(f"Using {unique_proteins} unique proteins for grouping in cross-validation.")
            groups = self.df['protein']

        # Initialize arrays to store predictions and true values
        n_samples = len(X)
        self.predictions = np.zeros(n_samples)
        self.true_labels = np.zeros(n_samples)
        
        # Define model fitting function based on model type
        if self.model_type == "rf":
            def fit_model(X, y):
                model = RandomForestClassifier(
                    n_estimators=100, 
                    max_depth=10,
                    n_jobs=-1)
                model.fit(X, y)
                return model
                
        elif self.model_type == "lda":
            def fit_model(X, y):
                model = LinearDiscriminantAnalysis()
                model.fit(X, y)
                return model
                
        elif self.model_type == "xgb":
            def fit_model(X, y):
                dtrain = xgb.DMatrix(X, y)
                params = {
                    'max_depth': 3, 
                    'eta': 0.3, 
                    'objective': 'binary:logistic',
                    'eval_metric': 'auc',
                    'nthread': 4
                }
                model = xgb.train(params, dtrain, num_boost_round=100)
                return model
                
            def predict_fn(model, X_test):
                dtest = xgb.DMatrix(X_test)
                return model.predict(dtest)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
            
        # Set predict function based on model type
        if self.model_type == "xgb":
            self.predict_fn = predict_fn
        else:
            self.predict_fn = lambda model, X_test: model.predict_proba(X_test)[:, 1]
        
        # Setup cross-validation with GroupKFold
        gfk = GroupKFold(n_splits=self.n_splits)
        
        # Check if we have enough groups
        n_groups = len(np.unique(groups))
        if n_groups < self.n_splits:
            print(f"Warning: Number of unique groups ({n_groups}) is less than number of splits ({self.n_splits})")
            self.n_splits = n_groups
            gfk = GroupKFold(n_splits=self.n_splits)
        
        # Get CV splits
        splits = list(gfk.split(X, y, groups=groups))
        
        # Train and predict for each fold
        print("Training models with cross-validation...")
        for fold_idx, (train_idx, test_idx) in enumerate(tqdm.tqdm(splits)):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            
            # Train model
            model = fit_model(X_train, y_train)
            self.models.append(model)
            
            # Generate predictions and store them directly
            fold_preds = self.predict_fn(model, X_test)
            self.predictions[test_idx] = fold_preds
            self.true_labels[test_idx] = y.iloc[test_idx].values
            
        return self.predictions, self.true_labels
    
    def evaluate(self):
        """
        Evaluate model performance using various metrics.
        
        Returns:
        --------
        roc_auc : float
            Area under the ROC curve
        avg_precision : float
            Average precision score
        accuracy : float
            Accuracy at optimal threshold
        """
        # Calculate ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(self.true_labels, self.predictions)
        roc_auc = auc(fpr, tpr)
        
        # Calculate PR curve and average precision
        precision, recall, pr_thresholds = precision_recall_curve(self.true_labels, self.predictions)
        avg_precision = average_precision_score(self.true_labels, self.predictions)
        
        # Find the optimal threshold using F1 score
        f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        
        # Handle case where optimal_idx is out of bounds
        if optimal_idx < len(pr_thresholds):
            optimal_threshold = pr_thresholds[optimal_idx]
        else:
            optimal_threshold = 0.5
        
        # Calculate binary predictions using the optimal threshold
        binary_preds = (self.predictions >= optimal_threshold).astype(int)
        accuracy = accuracy_score(self.true_labels, binary_preds)
        
        print("\nModel Evaluation Results:")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"Average Precision: {avg_precision:.4f}")
        print(f"Optimal Threshold: {optimal_threshold:.4f}")
        print(f"Accuracy at Optimal Threshold: {accuracy:.4f}")
        
        # Get feature importance for Random Forest model
        if self.model_type == "rf" and len(self.models) > 0 and hasattr(self, 'features'):
            feature_names = self.features
            
            # Average feature importance across all folds
            feature_importance = np.zeros(len(feature_names))
            for model in self.models:
                feature_importance += model.feature_importances_
            feature_importance /= len(self.models)
            
            # Sort features by importance
            indices = np.argsort(feature_importance)[::-1]
            top_features = [(feature_names[i], feature_importance[i]) for i in indices[:10]]
            
            print("\nTop 10 Important Features:")
            for i, (feature, importance) in enumerate(top_features):
                print(f"{i+1}. {feature}: {importance:.4f}")
                
        # Create visualization plots
        #self._plot_roc_curve(fpr, tpr, roc_auc)
        #self._plot_pr_curve(precision, recall, avg_precision)
        
        return #roc_auc, avg_precision, accuracy


def plex_features(fdc):

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
    
    ## What precursors are labeled as decoys
    fdc["decoy"] = np.array(["Decoy" in i for i in fdc["seq"]])

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
                  "unique_obs_int", 'MS1_Int',"MS1_Area", "iso_cor", "cosine", "traceproduct","iso1_cor","iso2_cor","ms1_cor",
                  "unique_frag_mz",
                  "unique_obs_int",
                  "file_name",
                  "protein"]
    # Check if there are more than 100 unique protein names
    unique_proteins = fdc['protein'].nunique()
    if unique_proteins < 100:
        print(f"Only {unique_proteins} unique proteins found, which is less than 100. Switching to use 'untag_seq' for grouping in cross-validation.")
        protein_names = fdc['untag_seq']
    else:
        print(f"Using {unique_proteins} unique proteins for grouping in cross-validation.")
        protein_names = fdc['protein']

    X = fdc.drop([c for c in drop_colums if c in fdc.columns], axis=1)
    # print(X.columns)
    X[np.isnan(X)] = 0  # set nans to zero (mostly for r2 values)
        
    sc_model = score_model(model_type, folder=folder)
    pred = sc_model.run_model(X, y, protein_names)
    
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

def score_channel_decoys(df, decoy_channels=[12], n_splits=5, model_type="rf", q_value_threshold=0.01):
    """
    Score peptide identifications based on channel decoy approach, where one channel
    (typically channel 12) is treated as a decoy channel.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing peptide identifications with channel information
    decoy_channels : list, default=[12]
        List of channel numbers to be used as decoy channels
    n_splits : int, default=5
        Number of cross-validation folds for model training
    model_type : str, default="rf"
        Model type to use for classification, options: "rf", "xgb", "lda"
    q_value_threshold : float, default=0.01
        Q-value threshold for filtering results
        
    Returns
    -------
    pandas.DataFrame
        The original DataFrame with additional columns:
        - is_decoy_channel: Binary indicator for decoy channel
        - predicted_decoy_[model_type]: Model score for being a decoy channel
        - channel_qvalue: Calculated q-value based on decoy competition
    """
    # Create binary target based on channel
    df['is_decoy_channel'] = df['channel'].isin(decoy_channels)
    
    # Print class distribution
    print(f"Channel distribution:")
    print(df['channel'].value_counts())
    print(f"\nDecoy channel(s) {decoy_channels} samples: {df['is_decoy_channel'].sum()}")
    print(f"Non-decoy channel samples: {(~df['is_decoy_channel']).sum()}")
    
    #Remove sequence decoys from channel fdr estimation
    df = df[~df['decoy']]
    # Define what constitutes a unique precursor
    precursor_columns = ['untag_prec', 'z']  # May need to change in future

    # Method 2: More efficient approach using transform
    # Create a flag for each precursor group indicating if any member passes threshold
    # May have be redundant with Jason's latest PR. 
    df['group_passes'] = df.groupby(precursor_columns)['Qvalue'].transform(
        lambda x: (x <= q_value_threshold).any()
    )

    # Filter to keep only rows where the group passes
    filtered_df = df[df['group_passes']]

    # Remove the temporary column if desired
    filtered_df = filtered_df.drop(columns=['group_passes'])

    # Print summary of filtering
    print(f"Original dataset size: {len(df)} rows")
    print(f"Filtered dataset size: {len(filtered_df)} rows")
    print(f"Kept {len(filtered_df)/len(df)*100:.1f}% of rows")

    # Count how many unique precursors were kept
    precursors_before = df.groupby(precursor_columns).ngroups
    precursors_after = filtered_df.groupby(precursor_columns).ngroups
    print(f"Unique precursors before: {precursors_before}")
    print(f"Unique precursors after: {precursors_after}")
    print(f"Kept {precursors_after/precursors_before*100:.1f}% of unique precursors")

    # Check distribution of channels in filtered dataset
    print("\nChannel distribution in filtered dataset:")
    print(filtered_df['channel'].value_counts())

    # Verify that we have complete precursor groups
    precursor_sizes_before = df.groupby(precursor_columns).size()
    precursor_sizes_after = filtered_df.groupby(precursor_columns).size()

    print("\nVerifying precursor group completeness:")
    if (precursor_sizes_before.loc[precursor_sizes_after.index] == precursor_sizes_after).all():
        print("✓ All precursor groups were kept intact")
    else:
        print("! Some precursor groups may be incomplete")

    # Print summary of filtering
    print(f"Original dataset size: {len(df)} rows")
    print(f"Filtered dataset size (Qvalue <= {q_value_threshold}): {len(filtered_df)} rows")
    print(f"Removed {len(df) - len(filtered_df)} rows ({(1 - len(filtered_df)/len(df))*100:.1f}%)")

    # Continue with the filtered DataFrame for your model training
    df = filtered_df  # Replace the original DataFrame with the filtered one

    # Remove sequence and metadata columns for modeling
    # Keep only numeric features that are useful for prediction
    # Define features to exclude from the model
    features = ['coeff', 'z','rt','num_lib','frac_lib_int',
                'mz_error','rt_error','frac_int_matched','prec_r2','prec_r2_uniq',
                'hyperscore','b_count','y_count','scribe_scores','max_unmatched_residuals','gof_stats',
                'manhattan_distances','frac_int_matched_pred','frac_int_matched_pred_sigcoeff','cosine'
                ,'plexfitMS1_p','plexfit_ps','ms1_cor','iso1_cor','iso2_cor',
                'traceproduct','iso_cor','MS1_Int','seq_len'
                ]
    # Get numeric columns only
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()

    # Filter features to only include those that exist in the DataFrame
    features = [col for col in features if col in numeric_columns]
    
    # Add boolean columns like ends_k if they exist
    if 'ends_k' in df.columns:
        features.append('ends_k')
    else:
        # Create ends_k feature if it doesn't exist
        df['ends_k'] = df['last_aa'] == 'K'
        features.append('ends_k')
    
    print(f"\nSelected {len(features)} features for model training")
    
    # Identify target channels (those that aren't in decoy_channels)
    target_channels = [ch for ch in df['channel'].unique() if ch not in decoy_channels]
    
    # Train the model and get predictions
    print(f"\n=== Training {model_type.upper()} model for channel decoy scoring ===")
    model = ChannelScoreModel(
        df=df,
        model_type=model_type,
        target_channels=target_channels, 
        decoy_channels=decoy_channels,
        features=features,
        n_splits=n_splits
    )
    
    predictions, true_labels = model.run_model()
    model.evaluate()
    
    # Add predictions to the dataframe
    df[f'predicted_decoy_{model_type}'] = predictions
    
    # Sort by model score (ascending for channel decoys)
    sorted_df = df.sort_values(by=f'predicted_decoy_{model_type}', ascending=True).reset_index(drop=True)
    
    # Calculate competition factor based on number of target channels vs decoy channels
    num_target_channels = len(target_channels)
    num_decoy_channels = len(decoy_channels)
    
    # Calculate competition factor
    competition_factor = num_target_channels / num_decoy_channels
    print(f"Competition factor: {competition_factor:.2f} ({num_target_channels} target channels / {num_decoy_channels} decoy channel(s))")
    
    # Calculate channel-level q-value with competition adjustment
    sorted_df['channel_qvalue'] = (sorted_df['is_decoy_channel'].cumsum() * competition_factor) / \
                                (sorted_df['is_decoy_channel'] == False).cumsum()
    
    # Apply monotonicity correction to ensure q-values never decrease
    min_q = float('inf')
    for i in range(len(sorted_df)-1, -1, -1):
        min_q = min(min_q, sorted_df.iloc[i]['channel_qvalue'])
        sorted_df.iloc[i, sorted_df.columns.get_loc('channel_qvalue')] = min_q
    
    # Use the sorted DataFrame as our result
    df = sorted_df
    
    # Print statistics about the q-values
    print(f"\nChannel Q-value summary statistics:")
    print(df['channel_qvalue'].describe())
    
    # Count identifications at different q-value thresholds
    for threshold in [0.01, 0.05, 0.1]:
        count = (df['channel_qvalue'] <= threshold).sum()
        percent = count/len(df)*100
        print(f"Identifications at q-value <= {threshold}: {count} ({percent:.1f}%)")
    
    # Create a filtered dataframe based on q-value threshold
    filtered_df = df[df['channel_qvalue'] <= q_value_threshold]
    print(f"\nAfter channel decoy filtering (q-value <= {q_value_threshold}):")
    print(f"Retained {len(filtered_df)} / {len(df)} identifications ({len(filtered_df)/len(df)*100:.1f}%)")
    
    return df

# Then, at the end of process_data, add:
# fdx = score_channel_decoys(fdx)
# fdx.to_csv(results_folder+"/all_IDs.csv", index=False)
# fdx[np.logical_and(~fdx["decoy"], fdx["Qvalue"] < config.fdr_threshold) & 
#     (fdx["channel_qvalue"] <= 0.01)].to_csv(results_folder+"/filtered_IDs.csv", index=False)
def process_data(file,spectra,library,mass_tag=None,timeplex=False):
    
    results_folder = os.path.dirname(file)
    mz_ppm = config.opt_ms1_tol
    rt_tol = config.opt_rt_tol
    
    lp,fdc,dc = get_large_prec(file,condense_output=False,timeplex=timeplex,
    max_features=['gof_stats','max_matched_residuals','manhattan_distances','scribe_scores'])
    
    
    
    ## Add additional features
    # X["prec_z"] = fdc["z"]
    fdc["stripped_seq"] = np.array([re.sub("Decoy_","",re.sub("\(.*?\)","",i)) for i in fdc["seq"]])
    fdc["pep_len"] = [len(re.findall("([A-Z](?:\(.*?\))?)",re.sub("Decoy","",i))) for i in fdc["stripped_seq"]]
    # X["rt"] = fdc["rt"]
    # X["coeff"] = fdc["coeff"]
    fdc["sq_rt_error"] = np.power(fdc["rt_error"],2)
    fdc["sq_mz_error"] = np.power(fdc["mz_error"],2)
    

    fdx = ms1_quant(fdc, lp, dc, mass_tag, spectra, mz_ppm, rt_tol, timeplex)

    
    fdx["untag_prec"] = ["_".join([i[0],str(int(i[1]))]) for i in zip(fdx["untag_seq"],fdx["z"])]
    channel_matches_counts = fdx["untag_prec"].value_counts()
    channel_matches_counts_dict = {i:j for i,j in zip(channel_matches_counts.index,channel_matches_counts)}
    fdx["channels_matched"] = [channel_matches_counts_dict[i] for i in fdx["untag_prec"]]
    
    fdx = score_precursors(fdx,config.score_model,config.fdr_threshold,folder=results_folder)
    

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

    
    ## save to results folder
    fdx.to_csv(results_folder+"/all_IDs.csv", index=False)

    # Check if we need to run channel decoy scoring
    run_channel_scoring = False

    if config.args.decoy_channels:
        decoy_channels = [int(channel.strip()) for channel in config.args.decoy_channels.split(',')]
    else:
        decoy_channels = []  # Default to empty list if no decoy channels provided
        
    if len(decoy_channels)>0:
        # Check if any rows in fdx match the decoy channels
        channel_column = 'channel'  
        decoy_channels_present = any(fdx[channel_column].isin(decoy_channels))
        if decoy_channels_present:
            run_channel_scoring = True
        else:
            print(f"WARNING: Decoy channels {decoy_channels} were provided but no rows with these channels were found in the data.")


    if run_channel_scoring:
        # Pass the decoy channels list to the scoring function
        fdx = score_channel_decoys(fdx, decoy_channels = decoy_channels)
        fdx.to_csv(results_folder+"/channel_IDs.csv", index=False)
        combined_filter = np.logical_and(~fdx["decoy"], fdx["Qvalue"] < config.fdr_threshold) & (fdx["channel_qvalue"] <= 0.05)
    else:
        # Skip channel decoy scoring
        fdx.to_csv(results_folder+"/channel_IDs.csv", index=False)
        combined_filter = np.logical_and(~fdx["decoy"], fdx["Qvalue"] < config.fdr_threshold)

    fdx[combined_filter].to_csv(results_folder+"/filtered_IDs.csv", index=False)



