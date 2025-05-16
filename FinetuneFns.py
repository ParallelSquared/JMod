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


def convert_models_to_keras3_format(model_path, output_dir=None):
    """
    Convert legacy TensorFlow SavedModel format to Keras 3-compatible format (.h5).
    
    Parameters
    ----------
    model_path : str
        Base path to the saved models. Function will append indices (0, 1, 2).
    output_dir : str, optional
        Directory to save converted models. If None, uses original directory.
    
    Returns
    -------
    list
        Paths to converted model files.
    """
    import tensorflow as tf
    import os
    
    if output_dir is None:
        output_dir = os.path.dirname(model_path)
    
    converted_paths = []
    
    for i in range(3):  # Assuming 3 models
        try:
            # First try the direct import approach
            print(f"Converting model {i}...")
            
            # Load the original model using TFSMLayer
            layer_model = tf.keras.layers.TFSMLayer(
                model_path + str(i), 
                call_endpoint='serving_default'
            )
            
            # Create a simple wrapper model that we can save in the new format
            input_shape = (30, 20)  # Adjust based on your model's expected input
            inputs = tf.keras.Input(shape=input_shape)
            outputs = layer_model(inputs)
            
            # Extract the actual tensor from dictionary output if needed
            if isinstance(outputs, dict):
                for key in outputs:
                    outputs = outputs[key]
                    break
            
            # Create a new model with the same behavior
            new_model = tf.keras.Model(inputs=inputs, outputs=outputs)
            
            # Save in H5 format (compatible with Keras 3)
            h5_path = os.path.join(output_dir, f"converted_model_{i}.h5")
            new_model.save(h5_path, save_format='h5')
            
            # Verify the model can be loaded
            test_model = tf.keras.models.load_model(h5_path)
            
            converted_paths.append(h5_path)
            print(f"Successfully converted model {i} to {h5_path}")
            
        except Exception as e:
            print(f"Error converting model {i}: {e}")
    
    return converted_paths


def load_existing_models(model_path):
    """
    Load pre-trained CNN models in a way compatible with Keras 3.
    
    This function first attempts to load models directly. If that fails due to 
    format compatibility issues, it tries to convert them to a Keras 3-compatible
    format and then load them.
    
    Parameters
    ----------
    model_path : str
        Base path to the saved models.
    
    Returns
    -------
    list
        A list of loaded model objects that can be used for inference and training.
    """
    import tensorflow as tf
    from tensorflow import keras
    import os
    
    models = []
    try_conversion = False
    
    # First try direct loading
    try:
        for i in range(3):  # Assuming 3 models
            try:
                # Try to load directly
                model = keras.models.load_model(model_path + str(i), compile=False)
                models.append(model)
                print(f"Successfully loaded model {i} directly")
            except Exception as e:
                if "File format not supported" in str(e):
                    print(f"Model {i} format not supported by Keras 3. Will try conversion.")
                    try_conversion = True
                    break
                else:
                    print(f"Error loading model {i}: {e}")
    except Exception as e:
        print(f"Error during initial model loading: {e}")
        try_conversion = True
    
    # If direct loading failed, try conversion
    if try_conversion or not models:
        print("Attempting to convert models to Keras 3-compatible format...")
        try:
            converted_paths = convert_models_to_keras3_format(model_path)
            
            # Try to load the converted models
            models = []
            for path in converted_paths:
                try:
                    model = keras.models.load_model(path, compile=False)
                    models.append(model)
                    print(f"Successfully loaded converted model from {path}")
                except Exception as e:
                    print(f"Error loading converted model from {path}: {e}")
            
            # If conversion worked, save as the new default
            if models and all(isinstance(m, keras.Model) for m in models):
                print("Successfully converted and loaded models. These will be used for future runs.")
        except Exception as e:
            print(f"Error during model conversion: {e}")
    print(f"Returning {len(models)} models of types: {[type(m).__name__ for m in models]}")
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
    print(f"Returning {len(models)} models of types: {[type(m).__name__ for m in models]}")
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
    
# model_path = "/Users/nathanwamsley/Data/JMOD_TESTS/iRT_CNN_model_tag6_05052025_"
         

def fine_tune_rt(grouped_df,
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
    print(f"Loaded {len(models)} models")
    # Perform LOESS regression on the filtered data
    ###!!!  NB: Fit/training is very much dependent on the loess fit; Make sure it's monotonic and nott overfit also (edit "frac")
    lowess = sm.nonparametric.lowess
    
    # Modified to handle both regular Keras models and TFSMLayer
    try:
        model_outputs = []
        X_test_array = np.array(X_test)
        
        for model in models:
            try:
                # Before prediction:
                print(f"Attempting prediction with model of type {type(model).__name__}")
                
                if hasattr(model, 'predict'):
                    pred = model.predict(X_test_array)
                    print(f"Used predict() method, result type: {type(pred).__name__}")
                else:
                    # TFSMLayer objects are called directly
                    print(f"Calling model directly (TFSMLayer approach)")
                    pred = model(X_test_array)
                    print(f"Direct call result type: {type(pred).__name__}")
                    
                    # Extract data from tensor dictionary if needed
                    if isinstance(pred, dict):
                        print(f"Prediction is a dictionary with keys: {list(pred.keys())}")
                        for key in pred:
                            value = pred[key]
                            print(f"Key '{key}' contains data of type {type(value).__name__}")
                            if hasattr(value, 'numpy'):
                                print(f"Converting TensorFlow tensor from key '{key}' to NumPy array")
                                pred = value.numpy()
                                print(f"Shape after conversion: {pred.shape}")
                                break
                    
                # Add shape information
                if hasattr(pred, 'shape'):
                    print(f"Prediction shape: {pred.shape}")
                else:
                    print(f"Prediction has no shape attribute, type: {type(pred)}")
                
                model_outputs.append(pred)
                print(f"Successfully added prediction to model_outputs")
            except Exception as e:
                print(f"Error predicting with model: {e}")
                print(f"Traceback: {traceback.format_exc()}")
        
        orig_predictions = np.mean(model_outputs, axis=0)
    except Exception as e:
        print(f"Error making predictions: {e}")
        # Define fallback
        def model_to_obs(rts):
            return rts
        def obs_to_model(rts):
            return rts
        return data_split, models, model_to_obs
    

    print(f"Number of model outputs collected: {len(model_outputs)}")
    for i, output in enumerate(model_outputs):
        print(f"Model output {i} type: {type(output).__name__}, ", end="")
        if hasattr(output, 'shape'):
            print(f"shape: {output.shape}")
        else:
            print(f"no shape attribute")

    if not model_outputs:
        print("No valid predictions from any model!")
    else:
        print(f"Attempting to average {len(model_outputs)} model outputs")

    print(f"Original predictions shape: {orig_predictions.shape}")
    print(f"Y_test shape: {Y_test.shape}")
    print(f"Flattened predictions shape: {orig_predictions.flatten().shape}")
    print(f"First few values of predictions: {orig_predictions.flatten()[:5]}")
    print(f"First few values of Y_test: {Y_test[:5]}")

    loess_result = lowess(orig_predictions.flatten(), Y_test, frac=0.1)
    
    def obs_to_model(rts):
        return np.interp(rts, loess_result[:, 0], loess_result[:, 1])
    
    def model_to_obs(rts):
        order = np.argsort(loess_result[:, 1])
        return np.interp(rts, loess_result[order, 1], loess_result[order, 0])

    if len(grouped_df)>config.FT_minimum:
        ### Fine Tune models
        # Check if models support training
        can_train = all(hasattr(model, 'fit') for model in models)
        if can_train:
            print("Fine Tuning RTs")
            models,history = train_models(models,[X_train,obs_to_model(Y_train)],results_folder=results_path)
        else:
            print("Models don't support fine-tuning (using TFSMLayer). Using for inference only.")
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
        
        if len(grouped_df)>config.FT_minimum and ('history' in locals() or 'history' in globals()):
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
                 
            
            # Modified to handle both model types, similar to above
            try:
                model_outputs = []
                for model in models:
                    try:
                        if hasattr(model, 'predict'):
                            pred = model.predict(np.array(X_test))
                        else:
                            pred = model(np.array(X_test))
                            if isinstance(pred, dict):
                                for key in pred:
                                    if hasattr(pred[key], 'numpy'):
                                        pred = pred[key].numpy()
                                        break
                                        
                        if hasattr(pred, 'numpy'):
                            pred = pred.numpy()
                            
                        model_outputs.append(pred)
                    except Exception as e:
                        print(f"Error predicting with model: {e}")
                        
                predictions = np.mean(model_outputs, axis=0)
                plt.subplots()
                plt.scatter(obs_to_model(Y_test),predictions,s=1)
                x_lim = plt.xlim()
                y_lim = plt.ylim()
                plt.plot(x_lim,x_lim)
                plt.xlabel("Aligned observed Values")
                plt.ylabel("New model predictions")
                if results_path:
                    plt.savefig(results_path+"/RT_finetune_AlignObsvsNew.png",dpi=600,bbox_inches="tight")
            except Exception as e:
                print(f"Error creating fine-tuned model plot: {e}")
        
    return data_split, models, model_to_obs