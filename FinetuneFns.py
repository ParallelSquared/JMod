"""
This Source Code Form is subject to the terms of the Oxford Nanopore
Technologies, Ltd. Public License, v. 1.0.  Full licence can be found
at https://github.com/ParallelSquared/JMod/blob/main/LICENSE.txt
"""



import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
load_model = tf.keras.models.load_model
import statsmodels.api as sm
import config
from tensorflow import keras
pd.options.display.max_columns = 1000


plt.rcParams['figure.dpi'] = 500
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12






peptide_sequence = "ACDEFGHIKLMNPQRSTVWY"   # all possible amino acids
max_sequence_length = 30    
num_amino_acids = 20      # possible amino acids

# Mapping amino acids to indices
amino_acid_to_index = {aa: i for i, aa in enumerate(peptide_sequence)}

def one_hot_encode_sequence(sequence):
    one_hot_sequence = np.zeros((max_sequence_length, num_amino_acids), dtype=np.float32)
    for i, aa in enumerate(sequence[:max_sequence_length]):
        aa_index = amino_acid_to_index.get(aa.upper(), None)
        if aa_index is not None:
            one_hot_sequence[i, aa_index] = 1.0
    return one_hot_sequence



def create_model_data(grouped_df,seq_name = 'PeptideSequence', rt_name="RT"):
    ## maybe chenge this to split the indices by train/test #!!! ???

    # Encode peptide sequences
    X_peptide = [one_hot_encode_sequence(seq) for seq in grouped_df[seq_name]]
    
    # Convert the list of arrays into a numpy array for model input
    X = np.array(X_peptide)
    Y = grouped_df[rt_name].values #THIS IS THE EMPIRICAL RT'S WHICH WERE CONVERTED TO iRT'S
    # Y_RT = grouped_df['RT'].values   #THIS IS THE EMPIRICAL RT'S
    # Y_RT_original_lib = grouped_df['Tr_recalibrated'].values #THIS IS THE ORIGINAL MODEL'S iRT'S... I.E. HOW IT WOULD BE WITHOUT REFINEMENT
    
    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=123)
    
    print(f"Training data size: {X_train.shape}, Test data size: {X_test.shape}")
    
    return X_train, X_test, Y_train, Y_test




#def load_existing_models(path):
#        
#    # Load the models for prediction without compiling
#    models = []
#    num_models = 5  # We have 5 models
#    
#    for i in range(num_models):
#        model = load_model(path+str(i), compile=False)
#        # model = tf.keras.layers.TFSMLayer(f"/Volumes/Lab/KMD/JD_RT_copy/iRT_model_mTRAQ_09042024_{i}", call_endpoint="serving_default")
#        models.append(model)
#        
#    return models

def load_existing_models(model_path):
    """
    Load pre-trained CNN models using Keras 3 compatible approach.
    
    This function loads saved models using keras.layers.TFSMLayer instead of the 
    traditional keras.models.load_model approach. This adaptation is necessary because 
    Keras 3 no longer supports loading legacy TensorFlow SavedModel format directly. 
    Instead, it loads them as inference-only layers via TFSMLayer.
    
    Parameters
    ----------
    model_path : str
        Base path to the saved models. The function will append indices (0, 1, 2)
        to this path to load multiple models.
    
    Returns
    -------
    list
        A list of loaded model objects (TFSMLayer instances) that can be used for inference.
        May contain fewer than 3 models if some failed to load.
    
    Notes
    -----
    - In Keras 3, only V3 .keras files and legacy H5 format (.h5) are supported by load_model().
    - TFSMLayer provides a way to load legacy SavedModel format for inference-only use.
    - The models loaded this way cannot be further trained, only used for inference.
    - The 'serving_default' endpoint is assumed - adjust if your models use a different endpoint.

    """

    models = []
    for i in range(3):  # Assuming 3 models
        try:
            # Use TFSMLayer for TensorFlow SavedModel format
            model = keras.layers.TFSMLayer(
                model_path + str(i), 
                call_endpoint='serving_default'
            )
            models.append(model)
        except Exception as e:
            print(f"Failed to load model {i}: {e}")
    return models



def train_models(models,train_data,results_folder=None):
        
    X_train, Y_train = train_data
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Compile and fine-tune each model with new data
    all_history = []
    for i, model in enumerate(models):
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mae')
        history = model.fit(np.array(X_train), Y_train, epochs=50, batch_size=24, validation_split=0.1, callbacks=[early_stopping],verbose=1)
        all_history.append(history)
        if results_folder:        
            model.save(results_folder+f'/iRT_updated_model{i}')
        
    return models, all_history


## convert rts to the same scale as the model to make fine tuning easier
def scale_rt(rt,min_max):
    assert len(min_max)==2
    rt = np.array(rt)
    scale_factor = (min_max[1]-min_max[0])/(np.max(rt)-np.min(rt))
    rt = rt - np.min(rt)
    rt = rt * scale_factor
    rt = rt + min_max[0]
    return rt
    
    

def fine_tune_rt(grouped_df,
                 # model_path='/Volumes/Lab/KMD/JD_RT_copy/iRT_model_mTRAQ_09042024_',
                 # model_path = '/Volumes/Lab/KMD/JD_RT_copy/CNN/iRT_updated_model', # better option #!!! Move to config?
                 # model_path = "/Volumes/Lab/KMD/JD_RT_copy/CNN/timeplex_lf/iRT_updated_model", ### trained from single LF timeplex
                 qc_plots = False,
                 results_path=None,
                 tag=None):
    
    print(f"{len(grouped_df)} peptides considered for fine tuning")
    
    if tag is None:
        tag=config.tag
    
    if tag is None:
        model_path = "/Volumes/Lab/JD/Predictions/CNN/iRT_CNN_model_LF_09182024_"
        
    elif tag.name=="mTRAQ":
        model_path = "/Volumes/Lab/JD/Predictions/CNN/iRT_CNN_model_mTRAQ_09182024_"
        
    elif "diethyl" in tag.name:
        # model_path = "/Volumes/Lab/KMD/FineTuning/DE_bulk3plex/iRT_updated_model"
        model_path = "/Volumes/Lab/KMD/FineTuning/DE_bulk_thenFDX016/iRT_updated_model"
        
    elif "tag6" in tag.name:
        # model_path = "/Volumes/Lab/JD/Predictions/CNN/iRT_TransferLearning_Tag6_updated_"
        #model_path = "/Volumes/Lab/KMD/FineTuning/tag6/iRT_CNN_model_tag6_05052025_"
        model_path = "/Users/nathanwamsley/Data/JMOD_TESTS/iRT_CNN_model_tag6_05052025_"
        
    else:
        raise ValueError("Unknown label")
        
    
    data_split = X_train, X_test, Y_train, Y_test = create_model_data(grouped_df,seq_name='Stripped.Sequence')

    models = load_existing_models(model_path)
    
    # Perform LOESS regression on the filtered data

    # Perform predictions based on model type
    # Need to update models so that all are copatible with the new Keras 3 approach
    # see 'load_existing_models' function docstring above 
    X_test_array = np.array(X_test)
    model_outputs = []
    
    models = load_existing_models(model_path)
    
    # Define fallback RT conversion function
    def create_fallback_converters():
        from scipy.interpolate import interp1d
        
        if 'iRT' in grouped_df.columns and 'RT' in grouped_df.columns:
            lib_rts = grouped_df['iRT'].values
            obs_rts = grouped_df['RT'].values
            
            valid_indices = ~np.isnan(lib_rts) & ~np.isnan(obs_rts)
            lib_rts = lib_rts[valid_indices]
            obs_rts = obs_rts[valid_indices]
            
            if len(lib_rts) > 3:
                rt_converter = interp1d(lib_rts, obs_rts, bounds_error=False, fill_value="extrapolate")
                def model_to_obs(rts):
                    return rt_converter(rts)
                    
                rev_converter = interp1d(obs_rts, lib_rts, bounds_error=False, fill_value="extrapolate")
                def obs_to_model(rts):
                    return rev_converter(rts)
            else:
                model_to_obs = lambda x: x
                obs_to_model = lambda x: x
        else:
            print("Warning: 'iRT' or 'RT' columns not found in dataframe")
            model_to_obs = lambda x: x
            obs_to_model = lambda x: x
        
        return model_to_obs, obs_to_model
    
    for model in models:
        try:
            # First get the raw output
            if hasattr(model, 'predict'):
                print("Using model.predict()")
                raw_pred = model.predict(X_test_array)
            else:
                print("Calling model directly")
                raw_pred = model(X_test_array)
            
            # Debug information
            print(f"Raw prediction type: {type(raw_pred)}")
            if isinstance(raw_pred, dict):
                print(f"Dictionary keys: {raw_pred.keys()}")
            
            # Process the prediction based on its type
            if isinstance(raw_pred, dict):
                # Try different strategies to extract values from dictionary
                if 'output_0' in raw_pred:
                    pred = raw_pred['output_0']
                    print(f"Using 'output_0' key, value type: {type(pred)}")
                elif 'predictions' in raw_pred:
                    pred = raw_pred['predictions'] 
                    print(f"Using 'predictions' key, value type: {type(pred)}")
                elif 'outputs' in raw_pred:
                    pred = raw_pred['outputs']
                    print(f"Using 'outputs' key, value type: {type(pred)}")
                else:
                    # Last resort: use the first value that looks like an array
                    print("Searching for array-like values in dictionary")
                    found_array = False
                    for key, value in raw_pred.items():
                        if isinstance(value, (np.ndarray, list)) and len(np.asarray(value)) > 0:
                            pred = value
                            print(f"Using key '{key}', value type: {type(pred)}")
                            found_array = True
                            break
                    
                    if not found_array:
                        print("Could not find usable array in dictionary")
                        print(f"Dictionary contents: {raw_pred}")
                        continue  # Skip this model
            else:
                pred = raw_pred
            
            # Convert to numpy array and ensure correct shape
            pred = np.asarray(pred)
            print(f"Prediction shape: {pred.shape}")
            
            if len(pred.shape) > 1 and pred.shape[1] == 1:
                pred = pred.flatten()
                print(f"Flattened shape: {pred.shape}")
            
            # Final sanity check - ensure values are numeric
            if not np.issubdtype(pred.dtype, np.number):
                print(f"Warning: Prediction has non-numeric dtype: {pred.dtype}")
                continue
                
            model_outputs.append(pred)
            print(f"Successfully added prediction with shape {pred.shape}")
            
        except Exception as e:
            print(f"Error during prediction with model: {e}")
            continue
    
    if not model_outputs:
        print("No valid predictions from any model")
        model_to_obs, obs_to_model = create_fallback_converters()
        return data_split, None, model_to_obs
        
    #orig_predictions = np.mean([model.predict(np.array(X_test)) for model in models],axis=0)
    ###!!!  NB: Fit/training is very much dependent on the loess fit; Make sure it's monotonic and nott overfit also (edit "frac")
    lowess = sm.nonparametric.lowess
    loess_result = lowess(orig_predictions.flatten(), Y_test, frac=0.1)
    
    def obs_to_model(rts):
        return np.interp(rts, loess_result[:, 0], loess_result[:, 1])
    
    def model_to_obs(rts):
        order = np.argsort(loess_result[:, 1])
        return np.interp(rts, loess_result[order, 1], loess_result[order, 0])

    # def obs_to_model(rts):
    #     return (rts-rts.min())/(rts.max()-rts.min())
    

    if len(grouped_df)>config.FT_minimum:
        ### Fine Tune models
        print("Fine Tuning RTs")
        models,history = train_models(models,[X_train,obs_to_model(Y_train)],results_folder=results_path)
    else:
        print("Not enough IDs for fine tuning")
        print("Using base model for predictions")
        
        ##use all data as validation as none was used for training
        data_split = np.concatenate([X_train,X_test],0),np.concatenate([X_train,X_test],0),np.concatenate([Y_train,Y_test],0),np.concatenate([Y_train,Y_test],0)   
        
    if qc_plots:
           
        plt.subplots()
        plt.scatter(Y_test,orig_predictions,s=1)
        # plt.scatter(Y_test,obs_to_model(Y_test),s=1)
        plt.xlabel("Observed Values")
        plt.ylabel("Base model predictions")
        if results_path:
            plt.savefig(results_path+"/RT_finetune_ObsvsOld.png",dpi=600,bbox_inches="tight")
            
        
        plt.subplots()
        plt.scatter(Y_test,obs_to_model(Y_test),s=1)
        plt.xlabel("Observed Values")
        plt.ylabel("Aligned observed Values")
        if results_path:
            plt.savefig(results_path+"/RT_finetune_ObsvsOldAlign.png",dpi=600,bbox_inches="tight")
        
        if len(grouped_df)>config.FT_minimum:
            # Add the smoothed values to a new column
            plt.subplots()
            plt.scatter(obs_to_model(Y_test),orig_predictions,s=1)
            x_lim = plt.xlim()
            y_lim = plt.ylim()
            plt.plot(x_lim,x_lim)
            plt.xlabel("Aligned observed Values")
            plt.ylabel("Old model predictions")
            if results_path:
                plt.savefig(results_path+"/RT_finetune_AlignObsvsOld.png",dpi=600,bbox_inches="tight")
            
        
            plt.subplots()
            for h in history:
                plt.plot(h.history["loss"],color="tab:blue",alpha=.5)
                plt.plot(h.history["val_loss"],color="tab:orange",alpha=.5) 
            if results_path:
                plt.savefig(results_path+"/RT_finetune_training.png",dpi=600,bbox_inches="tight")
                 
            

            predictions = np.mean([model.predict(np.array(X_test)) for model in models],axis=0)
            plt.subplots()
            plt.scatter(obs_to_model(Y_test),predictions,s=1)
            x_lim = plt.xlim()
            y_lim = plt.ylim()
            plt.plot(x_lim,x_lim)
            plt.xlabel("Aligned observed Values")
            plt.ylabel("New model predictions")
            if results_path:
                plt.savefig(results_path+"/RT_finetune_AlignObsvsNew.png",dpi=600,bbox_inches="tight")
        
    return data_split, models, model_to_obs
