"""
This Source Code Form is subject to the terms of the Oxford Nanopore
Technologies, Ltd. Public License, v. 1.0.  Full licence can be found
at https://github.com/ParallelSquared/JMod/blob/main/LICENSE.txt
"""



from src.utils.io.read_output import get_large_prec

from sklearn.model_selection import KFold,GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_curve,auc 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from scipy import stats
import xgboost as xgb

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tqdm
import re
import os
import pandas as pd

# from src.ms1_cor_channels import ms1_cor_channels
import src.ms1_cor_channels as ms1_mod
from src.utils.io.load_files import loadSpectra
from src.models.spec_lib.spec_lib import loadSpecLib

#from .mass_tags import mTRAQ, mTRAQ_02468, mTRAQ_678, tag_library
from src.utils.misc_functions import unstring_floats

from src import config
from src.logger import logger
from src.utils.misc_functions import p_result
from src.mass_tags import massTag
from pyteomics import mass
from numba import njit



@njit(nogil=True)
def _compute_shared_frac_jit(target_mz, target_int, other_mz_flat, mz_tol):
    """For one precursor, compute fraction of library intensity shared with other channels.

    target_mz, target_int: 1D arrays for this channel's matched library fragments
    other_mz_flat: concatenated m/z arrays from all OTHER channels of same untag_prec
    mz_tol: relative tolerance (e.g. 20e-6)

    Returns: fraction of target_int that has a match in any other channel
    """
    total_int = 0.0
    shared_int = 0.0
    for i in range(len(target_mz)):
        total_int += target_int[i]
        mz = target_mz[i]
        found = False
        for j in range(len(other_mz_flat)):
            if abs(other_mz_flat[j] - mz) / mz < mz_tol:
                found = True
                break
        if found:
            shared_int += target_int[i]
    if total_int == 0.0:
        return 0.0
    return shared_int / total_int


def _compute_frac_shared_intensity(fdc, mz_tol):
    """Compute fraction of library intensity from fragments shared across channels.

    For each untag_prec group with 2+ channels, compares fragment m/z lists
    across channels using relative PPM tolerance. Single-channel groups get -1.
    """
    result = np.full(len(fdc), -1.0)
    groups = fdc.groupby("untag_prec").indices

    for untag_prec, indices in groups.items():
        if len(indices) < 2:
            continue

        # Collect frag_mz and frag_int arrays for all rows in this group
        mz_arrays = []
        int_arrays = []
        for idx in indices:
            mz_arr = np.array(fdc["frag_mz"].iloc[idx], dtype=np.float64)
            int_arr = np.array(fdc["frag_int"].iloc[idx], dtype=np.float64)
            mz_arrays.append(mz_arr)
            int_arrays.append(int_arr)

        for k, idx in enumerate(indices):
            # Build other_mz_flat from all channels except this one
            other_parts = [mz_arrays[j] for j in range(len(indices)) if j != k]
            if len(other_parts) == 0 or all(len(p) == 0 for p in other_parts):
                result[idx] = 0.0
                continue
            other_mz_flat = np.concatenate(other_parts)
            target_mz = mz_arrays[k]
            target_int = int_arrays[k]
            if len(target_mz) == 0:
                result[idx] = 0.0
                continue
            result[idx] = _compute_shared_frac_jit(target_mz, target_int, other_mz_flat, mz_tol)

    return result


def area(x, apex_idx):
    top_3 = x[max(0, apex_idx-1):apex_idx+2]
    return np.sum(top_3)



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


def ms1_quant(dat,lp,dc,mass_tag,SILAC,DIAspectra,mz_ppm,rt_tol,timeplex=False):
    # X = fdc.iloc[:,6:-5]
    fit_whole_MS1 = False
   
    logger.info("")
    logger.info("Performing MS1 Quantitation") 
    
    fdc = dat[~dat["is_decoy"]].copy().reset_index(drop=True)  #remove decoys
    
    #only quantify confident precs
    if config.args.unfiltered_quant: #this will not execute if you specificy --unfiltered_quant (inherently stored as false)
        fdc = fdc[fdc["BestChannel_Qvalue"] < 0.01].reset_index(drop=True)

    if timeplex:
        all_keys = [(i,j,k) for i,j,k in zip(fdc.seq,fdc.z,fdc.time_channel)]
    else:
        all_keys = [(i,j) for i,j in zip(fdc.seq,fdc.z)]

    if mass_tag:
        tag_to_use = mass_tag
    else:
        tag_to_use = massTag(rules="nK",
                            base_mass=0.00,
                            delta=[0],
                            channel_names=["0"],
                            name="no_tag",
                            compositions=mass.Composition())

    (
    group_p_corrs,
    group_ms1_traces,
    group_ms2_traces,
    group_iso_ratios,
    group_keys,
    group_fitted,
    new_output_dict,
    fake_fdc_dict
    ) = ms1_mod.ms1_cor_channels(DIAspectra, 
                                fdc, 
                                dc, 
                                mz_ppm=mz_ppm, 
                                rt_tol = rt_tol,
                                tag=tag_to_use,
                                SILAC=SILAC,
                                timeplex=timeplex,
                                num_iso = config.num_iso_ms1,
                                num_iso_r = config.num_iso_r,
                                additional_scans = config.additional_scans,
                                fit_whole_MS1=fit_whole_MS1                                           
                                )
    
    dat = process_ms1_quant(dat,
                            fdc,
                            all_keys,
                            group_p_corrs,
                            group_ms1_traces,
                            group_ms2_traces,
                            group_iso_ratios,
                            group_keys,
                            group_fitted,
                            new_output_dict,
                            fake_fdc_dict,
                            DIAspectra,
                            fit_whole_MS1=fit_whole_MS1
                            )

    return dat



def process_ms1_quant(dat, fdc, all_keys, group_p_corrs, group_ms1_traces, group_ms2_traces, group_iso_ratios, group_keys, group_fitted, new_output_dict, fake_fdc_dict, DIAspectra, fit_whole_MS1=False):
    
    ## create dictionary  that links keys to data so we can match the order of "fdc"
    linker_dict = {key:[group_idx,key_idx] for group_idx,keys in enumerate(group_keys) for key_idx,key in enumerate(keys)}
    
    p_corrs = [group_p_corrs[linker_dict[key][0]][linker_dict[key][1]] for key in all_keys]
    ms1_traces = [group_ms1_traces[linker_dict[key][0]][linker_dict[key][1]] for key in all_keys]
    ms2_traces = [group_ms2_traces[linker_dict[key][0]][linker_dict[key][1]] for key in all_keys]
    iso_ratios = [group_iso_ratios[linker_dict[key][0]][linker_dict[key][1]] for key in all_keys]
    extracted_keys = [group_keys[linker_dict[key][0]][linker_dict[key][1]] for key in all_keys]

    if fit_whole_MS1 is False:
        extracted_fitted = [group_fitted[linker_dict[key][0]][0][:,linker_dict[key][1]] for key in all_keys]
        extracted_fitted_specs = [group_fitted[linker_dict[key][0]][4] for key in all_keys]
        extracted_fitted_p = [group_fitted[linker_dict[key][0]][3] for key in all_keys]

    else:
        ##TODO deal with fake_fdc_dict values make sure they get properly assigned
        all_fdc_idxs = list(set(key for ms1_spec_idx in new_output_dict.keys() for key in new_output_dict[ms1_spec_idx]["mapped_pred_coeffs"].keys()))
        extracted_fitted_specs = [[] for k in all_fdc_idxs if k not in fake_fdc_dict.keys()]
        extracted_fitted_p = [[] for k in all_fdc_idxs if k not in fake_fdc_dict.keys()]
        for ms1_spec_idx in new_output_dict.keys():
            for key in new_output_dict[ms1_spec_idx]["mapped_pred_coeffs"].keys():
                if key not in fake_fdc_dict.keys():
                    extracted_fitted_specs[key].append(ms1_spec_idx)
                    extracted_fitted_p[key].append(p_result(r_sq=np.nan, p=np.nan))

        extracted_fitted = []
        for fdc_idx in all_fdc_idxs:
            if fdc_idx not in fake_fdc_dict.keys():
                fdc_coeffs = []
                for scan in extracted_fitted_specs[fdc_idx]:
                    fdc_coeffs.append(new_output_dict[scan]["mapped_pred_coeffs"][fdc_idx])
                
                extracted_fitted.append(np.array(fdc_coeffs))
    
    
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
    fdc["plexfittrace_ps_all"] = [";".join(map(str,[pi.statistic if pi==pi else np.nan for pi in p])) for i,j,k,p in zip(extracted_fitted,extracted_fitted_specs,ms2_traces,extracted_fitted_p)]
    # The voted apex and scan restriction already happened in ms1_cor_channels.
    # extracted_fitted_specs is centered on the voted apex with ± 1 scan.
    # The apex is at the center position; area() sums the full window.
    plex_areas = []
    apex_scans = []
    for idx in range(len(fdc)):
        fitted = extracted_fitted[idx]
        specs = extracted_fitted_specs[idx]
        apex_pos = len(fitted) // 2
        plex_areas.append(area(fitted, apex_pos))
        apex_scans.append(int(specs[apex_pos]))
    fdc["plex_Area"] = plex_areas
    fdc["ms1_apex_scan"] = apex_scans
       


    
    
    fdc["ms1_cor"] = [i[0] for i in p_corrs]
    
    for idx in range(config.num_iso_r):
        iso_num = idx+1
        fdc[f"iso{iso_num}_cor"] = [i[iso_num] for i in p_corrs]
    # fdc["iso1_cor"] = [i[1] for i in p_corrs]
    # fdc["iso2_cor"] = [i[2] for i in p_corrs]
    
    # Rescale correlations from [-1,1] to [0,1] before computing product
    _ms1_cor_scaled = (fdc["ms1_cor"] + 1) / 2
    _iso1_cor_scaled = (fdc["iso1_cor"] + 1) / 2
    _iso2_cor_scaled = (fdc["iso2_cor"] + 1) / 2
    fdc["traceproduct"] = np.log10(_ms1_cor_scaled * _iso1_cor_scaled * _iso2_cor_scaled + 1e-6)
    
    # fdc["MS1_is1cor"] = [stats.pearsonr(list(i[0].values())[:10], list(i[1].values())[:10]).statistic for i in ms1_traces]
    
    
    fdc["iso_cor"] = [i[0].statistic for i in iso_ratios]
    
    fdc["MS1_Int"] = [i[2][0] for i in iso_ratios]
    fdc["MS1_Int"] = [np.linalg.lstsq(np.array(i[1])[:,np.newaxis], i[2], rcond=-1)[0][0] for i in iso_ratios]
    
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
        "plex_Area", "ms1_apex_scan", "ms1_cor", "traceproduct", "iso_cor", "MS1_Int",
        "all_ms1_specs", "MS1_Area"
    ] + [f"all_ms1_iso{i}vals" for i in range(config.num_iso_ms1)]
    
    # Ensure we only select columns that actually exist in fdc
    existing_cols = [col for col in selected_cols if col in fdc.columns]
    
    # Perform the merge safely
    dat = dat.merge(fdc[["untag_prec", "channel","silac_channel"] + existing_cols], how="left", on=["untag_prec", "channel","silac_channel"]).fillna(0)


    return dat



def estimate_pep(scores, is_decoy):
    """Estimate Posterior Error Probability via isotonic regression.

    Fits a non-decreasing decoy probability curve over the score distribution.
    PEP = decoy_prob / (1 - decoy_prob), clamped to [0, 1].
    """
    order = np.argsort(-scores)  # descending
    labels = is_decoy[order].astype(float)  # decoy=1, target=0

    ir = IsotonicRegression(y_min=0, y_max=1, increasing=True)
    fitted = ir.fit_transform(np.arange(len(labels)), labels)

    pep = fitted / (1 - fitted + 1e-10)
    pep = np.clip(pep, 0, 1)

    # Map back to original order
    result = np.empty_like(pep)
    result[order] = pep
    return result


class model_instance():
    def __init__(self,model_type):
        self.mode_type = model_type
        
    def predict(self,X):
        pred = self.__predict_fn__(X)
        #First column of pred is target probabilities and second column is decoys
        #If for some reason there were no decoys in one of the training folds
        #only a single column is returned. Handle this case...
        if len(pred.shape)==2:
            if pred.shape[1]==2:
                output = pred[:,1]
            else:
                output = pred[:,0]
        else:
            output = pred
        return output
        
        
class score_model():
    
    def __init__(self,model_type,n_splits=5,folder=None):
        self.model_type=model_type
        self.n_splits = n_splits
        self.folder = folder
                
    def run_model(self,X,y,sample_weight=None,groups=None):
        # logger.info(f"{config.tree_max_depth}")
        if self.model_type=="rf":
            
            ### Random Forest
            def fit_model(X,y,sample_weight,idx=""):
                    m = model_instance(model_type=self.model_type)
                    m.model = RandomForestClassifier(n_estimators = 200,max_depth=config.tree_max_depth,n_jobs=-1, random_state=config.RANDOM_SEED)
                    m.model.fit(X,y,sample_weight=sample_weight)
                    m.__predict_fn__ = m.model.predict_proba

                    if self.folder:
                        feature_importance = m.model.feature_importances_
                        sorted_indices = np.argsort(feature_importance, kind='stable')
                        sorted_features = np.array(X.columns)[sorted_indices]  
                        sorted_importance = feature_importance[sorted_indices]  
                    
                        fig, ax = plt.subplots(figsize=(8, len(X.columns)*0.3))                    
                        ax.barh(sorted_features, sorted_importance)
                        ax.set_title("Feature Importance")
                    
                        # Save plot
                        plt.savefig(self.folder + f"/scoring/RF{idx}_feature_importance.png", dpi=600, bbox_inches="tight")
                        # For RF models, log feature importance
                        plt.close(fig)
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
                    dTrain = xgb.DMatrix(X,y,weight=sample_weight,feature_names=list(X.columns))
                    # param = {
                    #     'max_depth': config.tree_max_depth, 
                    #     'eta': .1, 
                    #     'objective': 'binary:logistic',}
                    param = {
                        # 'max_depth': config.tree_max_depth, 
                        # 'eta': .1, 
                        # 'objective': 'binary:logistic',
                        
                        'objective': 'binary:logistic',  
                         'eval_metric': 'aucpr',    
                         'eta': 0.1,
                         'max_depth': 10,          
                         'subsample': 0.8,
                         'colsample_bytree': 0.8,   
                         'tree_method': 'hist',         
                         'nthread': -1,                   
                         'seed': config.RANDOM_SEED,
                         'min_child_weight': .5
                        }

                    # param['nthread'] = 4
                    # param['eval_metric'] = 'pre'
                    
                    m.model = xgb.train(param, dtrain=dTrain,num_boost_round=500)
                    def xg_predict(X):
                        X_convert = xgb.DMatrix(X,feature_names=list(X.columns))
                        return m.model.predict(X_convert)
                    m.__predict_fn__ = xg_predict
                    
                    if self.folder:
                        os.makedirs(os.path.join(self.folder, "scoring"), exist_ok=True)
                        fi = m.model.get_score(importance_type="gain")
                        feature_importance = np.array([fi.get(c, 0) for c in X.columns])
                        sorted_indices = np.argsort(feature_importance, kind='stable')
                        sorted_features = np.array(X.columns)[sorted_indices]
                        sorted_importance = feature_importance[sorted_indices]

                        fig, ax = plt.subplots(figsize=(8, len(X.columns)*0.3))
                        ax.barh(sorted_features, sorted_importance)
                        ax.set_title("Feature Importance")
                        plt.savefig(self.folder+f"/scoring/XGBoost{idx}_feature_importance.png",dpi=600,bbox_inches="tight")
                        plt.close(fig)

                    return m
                
            # self.model = fit_model(X,y)
        
            
        elif self.model_type == "nn":
            columns = X.columns
            X = pd.DataFrame(preprocessing.StandardScaler().fit(X).transform(X),columns=columns)
            ## Neural network
            def fit_model(X,y,sample_weight,idx=""):
                    m = model_instance(model_type=self.model_type)
                    # m.model = MLPClassifier((32,16,8,4),activation="relu")
                    m.model = MLPClassifier((20,20,4),activation="relu")
                    m.model.fit(X,y)
                    m.__predict_fn__ = m.model.predict_proba
                    return m
                
        else:
            from src.utils.gui_utils import send_raise_to_TK
            send_raise_to_TK("ValueError - Unsupported Model Type")
            raise ValueError("Unsupported model type")
        
        logger.debug(f"Total samples: {len(y)}, Positive: {sum(y)}, Negative: {len(y) - sum(y)}")
        
        kf = KFold(n_splits=self.n_splits,shuffle=True, random_state = config.RANDOM_SEED)
        k_orders = [i for i in kf.split(X,y)]
        rev_order = np.argsort(np.concatenate([i[1] for i in k_orders], ), kind='stable') # collapse test sets and get order

        if groups is not None:
            unique_groups = np.unique(groups)
            if len(unique_groups) < 5:
                logger.warning(f"Warning: Only {len(unique_groups)} unique groups for 5-fold CV. Using KFold instead.")
                gfk = KFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
            else:
                gfk = GroupKFold(n_splits=5)
        
            #k_orders = [i for i in kf.split(X,y)] old way
            k_orders = [i for i in gfk.split(X, y, groups=groups)]
            rev_order = np.argsort(np.concatenate([i[1] for i in k_orders]), kind='stable') # collapse test sets and get order
            
            # permutation = np.random.permutation(len(X))
            # X_shuffled = X.iloc[permutation]
            # y_shuffled = y[permutation]
            # groups_shuffled = np.array(self.groups)[permutation]
            # k_orders = [i for i in gfk.split(X_shuffled,y_shuffled,groups=groups_shuffled)]
            # rev_order = np.argsort(np.concatenate([i[1] for i in k_orders])) # collapse test sets and get order

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

    def run_model_filtered(self, X, y, keep_mask, y_filtered, sample_weight=None, groups=None):
        """Train with per-fold filtering: CV folds on full data, filter/relabel training portion only.

        For each CV fold:
          - Take the training indices from the full dataset
          - Apply keep_mask to retain only selected training samples
          - Use y_filtered (with negative-mined labels) for training
          - Predict on the FULL test split (unfiltered, unbiased)
        """
        if self.model_type=="rf":
            def fit_model(X,y,sample_weight,idx=""):
                    m = model_instance(model_type=self.model_type)
                    m.model = RandomForestClassifier(n_estimators = 200,max_depth=config.tree_max_depth,n_jobs=-1, random_state=config.RANDOM_SEED)
                    m.model.fit(X,y,sample_weight=sample_weight)
                    m.__predict_fn__ = m.model.predict_proba
                    return m
        elif self.model_type=="lda":
            def fit_model(X,y,sample_weight,idx=""):
                    m = model_instance(model_type=self.model_type)
                    m.model = LinearDiscriminantAnalysis()
                    m.model.fit(X,y)
                    m.__predict_fn__ = m.model.predict_proba
                    return m
        elif self.model_type == "xg":
            def fit_model(X,y,sample_weight,idx=""):
                    m = model_instance(model_type=self.model_type)
                    dTrain = xgb.DMatrix(X,y,weight=sample_weight,feature_names=list(X.columns))
                    param = {
                        'objective': 'binary:logistic',
                         'eval_metric': 'aucpr',
                         'eta': 0.1,
                         'max_depth': 10,
                         'subsample': 0.8,
                         'colsample_bytree': 0.8,
                         'tree_method': 'hist',
                         'nthread': -1,
                         'seed': config.RANDOM_SEED,
                         'min_child_weight': .5
                        }
                    m.model = xgb.train(param, dtrain=dTrain,num_boost_round=500)
                    def xg_predict(X):
                        X_convert = xgb.DMatrix(X,feature_names=list(X.columns))
                        return m.model.predict(X_convert)
                    m.__predict_fn__ = xg_predict
                    return m
        elif self.model_type == "nn":
            columns = X.columns
            X = pd.DataFrame(preprocessing.StandardScaler().fit(X).transform(X),columns=columns)
            def fit_model(X,y,sample_weight,idx=""):
                    m = model_instance(model_type=self.model_type)
                    m.model = MLPClassifier((20,20,4),activation="relu")
                    m.model.fit(X,y)
                    m.__predict_fn__ = m.model.predict_proba
                    return m
        else:
            raise ValueError("Unsupported model type")

        logger.debug(f"Total samples: {len(y)}, Kept for training: {keep_mask.sum()}")

        # CV folds on the FULL dataset (same as run_model)
        kf = KFold(n_splits=self.n_splits,shuffle=True, random_state = config.RANDOM_SEED)
        k_orders = [i for i in kf.split(X,y)]

        if groups is not None:
            unique_groups = np.unique(groups)
            if len(unique_groups) < 5:
                gfk = KFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
            else:
                gfk = GroupKFold(n_splits=5)
            k_orders = [i for i in gfk.split(X, y, groups=groups)]

        rev_order = np.argsort(np.concatenate([i[1] for i in k_orders]), kind='stable')

        self.models = []
        self.predictions = []
        model_idx = 0
        for train_idx, test_idx in tqdm.tqdm(k_orders):
            # Filter training portion: keep only samples in keep_mask, use relabeled y
            fold_keep = keep_mask[train_idx]
            filtered_train_idx = train_idx[fold_keep]
            X_train_fold = X.iloc[filtered_train_idx]
            y_train_fold = y_filtered[filtered_train_idx]
            sw = sample_weight[filtered_train_idx] if sample_weight is not None else None

            m = fit_model(X_train_fold, y_train_fold, sample_weight=sw, idx=model_idx)
            self.models.append(m)
            # Predict on FULL test split (unfiltered)
            self.predictions.append(m.predict(X.iloc[test_idx]))
            model_idx += 1

        return np.concatenate(self.predictions)[rev_order]


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

    logger.info("Scoring IDs")


    ## Only decoys are negatives - all targets are positives
    y = np.array(~fdc["is_decoy"], dtype=int)

    # exclude necessary columns
    drop_colums = ['spec_id', 'Ms1_spec_id', 'seq', 'window_mz', 'frag_names', 'frag_errors', 'frag_mz', 'frag_int', 'obs_int', 'stripped_seq',
                  'untag_seq', 'is_decoy', 'all_ms1_specs', 'all_ms1_iso0vals', 'all_ms1_iso1vals', 'all_ms1_iso2vals','all_ms1_iso3vals', 'all_ms1_iso4vals',
                  'all_ms1_iso5vals','all_ms1_iso6vals','all_ms1_iso7vals',"plexfittrace","plexfit_ps","untag_prec","plexfittrace_spec_all","plexfittrace_all",
                  "plexfittrace_ps_all",
                  "unique_frag_mz", "untag_prec",
                  "channels_matched",
                  "unique_obs_int", 'MS1_Int',"MS1_Area", "iso_cor", "cosine", "traceproduct","iso1_cor","iso2_cor","ms1_cor","plexfitMS1","plexfitMS1_p","plex_Area", "untag_prec","channel","time_channel",
                  "silac_channel",
                  "unique_frag_mz",
                  "unique_obs_int",
                  "file_name",
                  "protein"]
    X = fdc.drop([c for c in drop_colums if c in fdc.columns], axis=1)

    # Compute predicted RT (library RT) from observed RT minus RT error
    if 'rt' in X.columns and 'rt_error' in X.columns:
        X['predicted_rt'] = X['rt'] - X['rt_error']
    X = X.drop(columns=[c for c in ['rt'] if c in X.columns])

    # Quantile-bin positional/intensity features into 100 bins
    QBIN_FEATURES = ['predicted_rt', 'mz', 'coeff', 'tic']
    for col in QBIN_FEATURES:
        if col in X.columns:
            vals = X[col].values.astype(float)
            valid = vals[~np.isnan(vals)]
            if len(valid) > 0:
                edges = np.quantile(valid, np.linspace(0, 1, 101))
                binned = np.searchsorted(edges, vals, side='right').astype(float)
                binned = np.clip(binned, 1, 100)
                X[col + '_qbin'] = binned
            X = X.drop(columns=[col])

    X[np.isnan(X)]=0 ## set nans to zero (mostly for r2 values)

    # Iterative training: 3 iterations with target filtering and negative mining
    n_iterations = 3
    pred = None
    fdc_qvalues = None

    for itr in range(n_iterations):
        logger.info(f"  scoring iteration {itr + 1}/{n_iterations}")

        if itr == 0:
            sc_model = score_model(model_type, folder=folder)
            pred = sc_model.run_model(X, y, groups=fdc.stripped_seq)
        else:
            pep_values = estimate_pep(pred, fdc["is_decoy"].values)

            y_filtered = y.copy()
            neg_mine_mask = (y_filtered == 1) & (pep_values >= 0.90)
            y_filtered[neg_mine_mask] = 0

            keep_mask = fdc["is_decoy"].values | (fdc_qvalues <= 0.01) | neg_mine_mask

            n_kept = keep_mask.sum()
            n_relabeled = neg_mine_mask[keep_mask].sum()
            logger.info(f" f   Kept {n_kept}/{len(keep_mask)} samples, relabeled {n_relabeled} as decoys")

            sc_model = score_model(model_type, folder=folder)
            pred = sc_model.run_model_filtered(X, y, keep_mask, y_filtered, groups=fdc.stripped_seq)

        # Compute q-values for this iteration
        score_order = np.argsort(-pred, kind='stable')
        decoy_order = fdc["is_decoy"].values[score_order]
        fdc_qvalues_ordered = (1 + np.cumsum(decoy_order)) / np.cumsum(~decoy_order) * config.target_decoy_ratio
        fdc_qvalues_ordered = np.minimum.accumulate(fdc_qvalues_ordered[::-1])[::-1]
        fdc_qvalues = np.empty_like(fdc_qvalues_ordered)
        fdc_qvalues[score_order] = fdc_qvalues_ordered

    model_name = model_type

    if len(pred.shape)==2:
        output = pred[:,1]
    else:
        output = pred

    ## Use the scores to estimate the #IDs as 1% FDR
    fpr, tpr, _ = roc_curve(y, output)

    score_order = np.argsort(-output, kind='stable')
    orig_order = np.argsort(score_order, kind='stable')
    decoy_order = fdc["is_decoy"][score_order]
    frac_decoy = (1 + np.cumsum(decoy_order)) / np.cumsum(~decoy_order) * config.target_decoy_ratio
    frac_decoy = np.minimum.accumulate(frac_decoy[::-1])[::-1]  # Monotonize: q-value = min of downstream q-values
    T = output[score_order[min(len(score_order)-1,np.searchsorted(frac_decoy,0.01))]]
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
        plt.savefig(folder+"/scoring/ModelScore.png",dpi=600,bbox_inches="tight")
        plt.close()



        feat = 'rt_error'
        func = np.array#np.log10#
        plt.subplots()
        vals,bins,_ = plt.hist(func([i for i in fdc[feat]]),40,label="All")
        # plt.hist([],[])
        vals,bins,_ = plt.hist(func([i for i in fdc[feat][above_t]]),bins,alpha=.5,label="1%FDR")
        vals,bins,_ = plt.hist(func([i for i in fdc[feat][np.logical_and(~above_t,~fdc.is_decoy)]]),bins,alpha=.5,label="Low scoring")
        vals,bins,_ = plt.hist(func([i for i in fdc[feat][fdc.is_decoy]]),bins,alpha=.5,label="Decoy")
        plt.xlabel(feat)
        plt.ylabel("Frequency")
        # plt.title(model_name+ f" - Type {config.unmatched_fit_type}")
        plt.title(model_name)
        plt.legend()
        plt.savefig(folder+"/scoring/RT_error.png",dpi=600,bbox_inches="tight")
        plt.close()


        feat = 'mz_error'
        func = np.array#np.log10#
        plt.subplots()
        vals,bins,_ = plt.hist(func([i for i in fdc[feat]]),40,label="All")
        # plt.hist([],[])
        vals,bins,_ = plt.hist(func([i for i in fdc[feat][above_t]]),bins,alpha=.5,label="1%FDR")
        vals,bins,_ = plt.hist(func([i for i in fdc[feat][np.logical_and(~above_t,~fdc.is_decoy)]]),bins,alpha=.5,label="Low scoring")
        vals,bins,_ = plt.hist(func([i for i in fdc[feat][fdc.is_decoy]]),bins,alpha=.5,label="Decoy")

        # putting a xlim so that you can see entire distribution of the mz errors better
        xmin, xmax = plt.xlim()
        max_abs = max(abs(xmin), abs(xmax))
        plt.xlim(-max_abs, max_abs)

        plt.xlabel(feat)
        plt.ylabel("Frequency")
        # plt.title(model_name+ f" - Type {config.unmatched_fit_type}")
        plt.title(model_name)
        plt.legend()
        plt.savefig(folder+"/scoring/mz_error.png",dpi=600,bbox_inches="tight")

        plt.close()
        plt.close("all")

    return fdc


def log_df(df):
    filenames = df.iloc[:, 0].unique()
    filenames = [fn.rstrip("0123456789") for fn in filenames]
    filenames = list(set(filenames))
    for fn in filenames:
        logger.info(f"File: {fn}")

    df_no_first = df.iloc[:, 1:]
    for line in df_no_first.to_string(index=False).splitlines():
        logger.info(line)

def compute_protein_FDR(df,results_folder=None):
    logger.info("")
    logger.info("Computing Protein FDR")

  
    df["run_chan"] = df["file_name"].astype(str) + df["channel"].astype(str)+ df["silac_channel"].astype(str)
    df_seqchargeqvals = df[df["Qvalue"] < 0.01].copy().reset_index(drop=True) #filter
    df_seqchargeqvals["maxPredval"] = df_seqchargeqvals.groupby(["protein", "is_decoy"])["PredVal"].transform("max")
    df_seqchargeqvals = df_seqchargeqvals.drop_duplicates(subset=["protein", "is_decoy"]).reset_index(drop=True)

    # Rank by descending maxPredval and compute accum_decoys & Protein_Qvalue
    df_seqchargeqvals = df_seqchargeqvals.sort_values(by="maxPredval", ascending=False).reset_index(drop=True)
    df_seqchargeqvals["prot_rank"] = df_seqchargeqvals.index + 1  # Equivalent to row_number()
    df_seqchargeqvals["accum_decoys"] = df_seqchargeqvals["is_decoy"].cumsum()
    df_seqchargeqvals["Protein_Qvalue"] = (1 + df_seqchargeqvals["accum_decoys"]) / (~df_seqchargeqvals["is_decoy"]).cumsum() * config.target_decoy_ratio
    df_seqchargeqvals["Protein_Qvalue"] = df_seqchargeqvals["Protein_Qvalue"].iloc[::-1].cummin().iloc[::-1]  # Monotonize: q-value = min of downstream q-values
    
    # Filter for non-decoy proteins and select distinct protein values
    df_seqchargeqvals_distinct = (
        df_seqchargeqvals[~df_seqchargeqvals["is_decoy"]]
        .drop_duplicates(subset=["protein"])
        [["protein", "Protein_Qvalue"]]
    )
    
    df = df.drop(columns=["Protein_Qvalue"], errors="ignore")
    df = df.merge(df_seqchargeqvals_distinct, on="protein", how="left")
        
    df_counts_prec = (
        df[(df["is_decoy"] == False) & (df["Qvalue"] < 0.01)]
        .drop_duplicates(subset=["run_chan", "untag_prec"])
        .groupby(["file_name", "channel", "silac_channel"])
        .size()
        .reset_index(name="Precursor_IDs")
        .sort_values("channel")
    )    
    logger.info("")
    logger.info("Number of precursors at 1% FDR:")
    logger.info(f"All Channels:{np.sum(df_counts_prec.Precursor_IDs)}")
    log_df(df_counts_prec)
    

    df_counts_prots = (
        df[(df["Protein_Qvalue"] < 0.01) & (df["is_decoy"] == False) & (df["Qvalue"] < 0.01)]
        .drop_duplicates(subset=["run_chan", "protein"])
        .groupby(["file_name", "channel", "silac_channel"])
        .size()
        .reset_index(name="Protein_IDs")
        .sort_values("channel")
        )
    logger.info("")
    logger.info("Number of proteins at 1% FDR:")
    logger.info(f"All Channels:{np.sum(df_counts_prots.Protein_IDs)}")
    log_df(df_counts_prots)

    if results_folder is not None:
        with open(results_folder+'/Summary.txt', 'a') as f:
            print("Number of precursors at 1% FDR:", file=f)
            print("All Channels:",np.sum(df_counts_prec.Precursor_IDs), file=f)
            print(df_counts_prec.to_string(index=False), file=f)
            
            print("\nNumber of proteins at 1% FDR:", file=f)
            print("All Channels:",np.sum(df_counts_prots.Protein_IDs), file=f)
            print(df_counts_prots.to_string(index=False), file=f)
    
    # if config.args.plexDIA:
    #     if config.args.timeplex:
    #         df["BestChannel_Protein_Qvalue"] = df.groupby(["time_channel", "protein", "decoy"])["Protein_Qvalue"].transform("min")
    #     else:
    #         df["BestChannel_Protein_Qvalue"] = df.groupby(["file_name", "protein", "decoy"])["Protein_Qvalue"].transform("min")

    if config.args.plexDIA:
        logger.info("")
        logger.info("After plexDIA identification propagation based on best channel Q-value:")
        
        # Compute number of precursor IDs at 1% FDR
        df_counts_prec = (
            df[(df["is_decoy"] == False) & (df["BestChannel_Qvalue"] < 0.01)]
            .drop_duplicates(subset=["run_chan", "untag_prec"])
            .groupby(["file_name", "channel", "silac_channel"])
            .size()
            .reset_index(name="Precursor_IDs")
            .sort_values("channel")
        )
        
        # Print precursor ID counts
        logger.info("")
        logger.info("Number of precursors at 1% FDR (best channel):")
        logger.info(f"All Channels:{np.sum(df_counts_prec.Precursor_IDs)}")
        log_df(df_counts_prec)
        
        # Compute number of protein IDs at 1% FDR
        df_counts_prots = (
            df[(df["Protein_Qvalue"] < 0.01) & (df["is_decoy"] == False) & (df["BestChannel_Qvalue"] < 0.01)]
            .drop_duplicates(subset=["run_chan", "protein"])
            .groupby(["file_name", "channel", "silac_channel"])
            .size()
            .reset_index(name="Protein_IDs")
            .sort_values("channel")
        )
        
        # Print protein ID counts
        logger.info("")
        logger.info("Number of proteins at 1% FDR (best channel):")
        logger.info(f"All Channels:{np.sum(df_counts_prots.Protein_IDs)}")
        log_df(df_counts_prots)
        
        if results_folder is not None:
            with open(results_folder+'/Summary.txt', 'a') as f:
                print("Number of precursors at 1% FDR (best channel):", file=f)
                print("All Channels:",np.sum(df_counts_prec.Precursor_IDs), file=f)
                print(df_counts_prec.to_string(index=False), file=f)
                
                print("\nNumber of proteins at 1% FDR (best channel):", file=f)
                print("All Channels:",np.sum(df_counts_prots.Protein_IDs), file=f)
                print(df_counts_prots.to_string(index=False), file=f)


    return df

def add_median_based_features(df, metric_columns, group_col="untag_prec", count_col="channels_matched", verbose=True):
    """
    Calculate median-based features for specified metrics across groups.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe containing the metric columns
    metric_columns : list
        List of column names to calculate medians and differences for
    group_col : str, default="untag_prec"
        Column to group by for median calculations
    count_col : str, default="channels_matched"
        Column indicating how many channels each group has
    verbose : bool, default=True
        Whether to print summary statistics
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added median and difference columns
    """
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    if verbose:
        logger.debug(f"Adding median-based features for {len(metric_columns)} metrics...")
    
    for metric_col in metric_columns:
        # Calculate median for each group
        col_name = f"median_{metric_col}"
        result_df[col_name] = result_df.groupby(group_col)[metric_col].transform("median")
        
        # Set to NA for single-channel entries
        result_df.loc[result_df[count_col] == 1, col_name] = pd.NA
        
        # Calculate difference from median
        diff_col = f"diff_{metric_col}_from_median"
        result_df[diff_col] = result_df[metric_col] - result_df[col_name]
        
        # Fill NA with mean of non-NA values
        mean_val = result_df[diff_col].mean()
        result_df[diff_col] = result_df[diff_col].fillna(mean_val)
        
        if verbose:
            logger.debug(f"  Added {diff_col} (mean for NA values: {mean_val:.5f})")
            logger.debug(f"  Summary stats: min={result_df[diff_col].min():.5f}, max={result_df[diff_col].max():.5f}, mean={result_df[diff_col].mean():.5f}")
    
    return result_df


def process_data(file,spectra,library,mass_tag=None,timeplex=False,SILAC=None,elution_fwhm=None):
    
    # results_folder = os.path.dirname(file)
    results_folder = os.path.dirname(os.path.dirname(file))
    mz_ppm = config.opt_ms1_tol
    rt_tol = config.opt_rt_tol
    
    # After loading data and adding basic features
    lp,fdc,dc = get_large_prec(file,condense_output=False,timeplex=timeplex)
    
    # Add standard features
    fdc["stripped_seq"] = np.array([re.sub("Decoy_","",re.sub("\(.*?\)","",i)) for i in fdc["seq"]])
    fdc["pep_len"] = [len(re.findall("([A-Z](?:\(.*?\))?)",re.sub("Decoy","",i))) for i in fdc["stripped_seq"]]
    fdc["sq_rt_error"] = np.power(fdc["rt_error"],2)
    fdc["sq_mz_error"] = np.power(fdc["mz_error"],2)

    # Handle untag_seq
    if mass_tag and SILAC:
        fdc["untag_seq"] = [re.sub(f"(\({SILAC.name}-\d+\))?","",re.sub(f"(\({mass_tag.name}-\d+\))?","",peptide)) for peptide in fdc["seq"]]
    elif mass_tag:
        fdc["untag_seq"] = [re.sub(f"(\({mass_tag.name}-\d+\))?","",peptide) for peptide in fdc["seq"]]
    elif SILAC:
        fdc["untag_seq"] = [re.sub(f"(\({SILAC.name}-\d+\))?","",peptide) for peptide in fdc["seq"]]
    else:
        fdc["untag_seq"] = fdc["seq"]
    #print(fdc.columns)  # Ensure 'seq' is in fdc

    # Add untag_prec and channels_matched
    fdc["untag_prec"] = ["_".join([i[0],str(int(i[1]))]) for i in zip(fdc["untag_seq"],fdc["z"])]
    
    channel_matches_counts = fdc["untag_prec"].value_counts()
    channel_matches_counts_dict = {i:j for i,j in zip(channel_matches_counts.index,channel_matches_counts)}
    fdc["channels_matched"] = [channel_matches_counts_dict[i] for i in fdc["untag_prec"]]

    # Use the helper function to add median-based features
    metrics_to_process = ["gof_stats", "scribe_scores", "max_matched_residuals", "manhattan_distances"]
    fdc = add_median_based_features(fdc, metrics_to_process)

    # Compute frac_shared_intensity: fraction of library intensity from fragments shared across channels
    fdc["frac_shared_intensity"] = _compute_frac_shared_intensity(fdc, mz_tol=config.mz_tol)

    if timeplex:
        if mass_tag:
            tag_name = mass_tag.name
            tag_channel = [re.findall(f"{tag_name}-(\d+)",i) for i in fdc.seq]
            fdc["channel"] = [str(int(t))+"_"+i[0] if len(i)>0 else str(int(t)) for i,t in zip(tag_channel,fdc.time_channel)]
        else:
            fdc["channel"] = fdc["time_channel"]
            
    elif mass_tag:
        tag_name = mass_tag.name
        ## mTRAQ label
        tag_channel = [re.findall(f"{tag_name}-(\d+)",i) for i in fdc.seq]
        fdc["channel"] = [int(i[0]) if len(i)>0 else np.nan for i in tag_channel]

    else: 
        fdc["channel"] = 0 #if LF
        
    if SILAC is not None:
        silac_channel = [re.findall(f"{SILAC.name}-(\d+)",i) for i in fdc.seq]
        fdc["silac_channel"] = [int(i[0]) if len(i)>0 else np.nan for i in silac_channel] ### Note: This needs to change for multichannel SILAC
    else:
        fdc["silac_channel"] = np.nan 
        
    #this was previously in ms1_quant function.. we need it for the target/decoy classification
    # frag_errors stored as list columns in parquet — convert to numpy arrays
    frag_errors = [np.array(x, dtype=float) if x is not None and len(x) > 0 else np.array([]) for x in fdc.frag_errors]
    non_empty = [i for i in frag_errors if len(i) > 0]
    median = np.median(np.concatenate(non_empty)) if non_empty else 0.0
    fdc["med_frag_error"] = [np.median(np.abs(median-i)) if len(i) > 0 else np.nan for i in frag_errors]

    # Fragment-ion correlation features (pairwise Pearson across MS2 scans)
    from src.fragment_correlation import compute_fragment_correlations
    corr_features = compute_fragment_correlations(
        spectra=spectra,
        library=library,
        fdc=fdc,
        fwhm=elution_fwhm,
        mz_tol=config.mz_tol,
    )
    for col in corr_features.columns:
        fdc[col] = corr_features[col].values

    minfraclib_toscore = getattr(config.args, "score_lib_frac", 0)
    fdx_toscore = fdc[fdc['frac_lib_int'].fillna(0) >= minfraclib_toscore].reset_index(drop=True)
    
    fin = score_precursors(fdx_toscore,config.score_model,config.fdr_threshold,folder=results_folder)
    new_columns = [col for col in fin.columns if col not in fdc.columns and col not in ["untag_prec", "channel"]]
    fdx = fdc.merge(fin[["untag_prec", "channel","silac_channel"] + new_columns], how="left", on=["untag_prec", "channel","silac_channel"])

    ##fill NA's appropriately
    fdx['PredVal'] = fdx['PredVal'].fillna(0)  
    fdx['Qvalue'] = fdx['Qvalue'].fillna(1)     


    # if config.args.plexDIA:
    #     if config.args.timeplex:
    #         fdx["BestChannel_Qvalue"] = fdx.groupby(["time_channel", "untag_prec", "decoy"])["Qvalue"].transform("min") #within a plexDIA set for each timechannel
    #     else:
    #         fdx["BestChannel_Qvalue"] = fdx.groupby(["file_name", "untag_prec", "decoy"])["Qvalue"].transform("min") #within a plexDIA set
    
    if config.args.plexDIA or config.args.timeplex:
        fdx["BestChannel_Qvalue"] = fdx.groupby(["file_name", "untag_prec", "is_decoy"])["Qvalue"].transform("min") #within a run
    else:
        fdx["BestChannel_Qvalue"] = fdx["Qvalue"] #applies to no plex

    
    fdx_quant = ms1_quant(fdx, lp, dc, mass_tag, SILAC, spectra, mz_ppm, rt_tol, timeplex)

    fdx_quant["last_aa"] = [i[-1] for i in fdx_quant["stripped_seq"]]
    fdx_quant["seq_len"] = [len(i) for i in fdx_quant["stripped_seq"]]
    
    # have possible reannotate woth fasta here
    # fdx["org"] = np.array([";".join(orgs[[i in all_fasta_seqs[j] for j in range(3)]]) for i in fdx["stripped_seq"]])
    fdx_quant = compute_protein_FDR(fdx_quant,results_folder=results_folder)

    logger.info("")
    logger.info(f"Saving Results to Folder - {os.path.abspath(results_folder)}")
    ## save to results folder
    fdx_quant.to_csv(results_folder+"/outputs/all_IDs.csv",index=False)
    filtered = fdx_quant[np.logical_and(~fdx_quant["is_decoy"],fdx_quant["BestChannel_Qvalue"] < config.fdr_threshold)]
    filtered.to_csv(results_folder+"/filtered_IDs.csv",index=False)
    # fdx_quant[np.logical_and(~fdx_quant["is_decoy"],fdx_quant["BestChannel_Qvalue"]<config.fdr_threshold)].to_csv(results_folder+"/filtered_IDs.csv",index=False)

    ### select minimum columns for parquet
    parquet_columns = ["stripped_seq","z","untag_prec","file_name","channel","is_decoy","Qvalue", "Protein_Qvalue","PredVal",
                       "protein",'BestChannel_Qvalue', 'plex_Area', 'seq', 'silac_channel', 'untag_seq',"rt","mz"]
    parquet_columns = [i for i in parquet_columns if i in fdx_quant.columns]
    fdx_quant[parquet_columns].to_parquet(results_folder+"/outputs/all_IDs_filtered.parquet")

    # filtered IDs with parquet columns 
    filtered[parquet_columns].to_parquet(results_folder + "/filtered_IDs_parquet_columns.parquet", index=False)