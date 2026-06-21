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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
load_model = tf.keras.models.load_model
import statsmodels.api as sm
import src.config as config
from tensorflow import keras
import os
from src.logger import logger

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
    
    logger.debug(f"Training data size: {X_train.shape}, Test data size: {X_test.shape}")
    
    return X_train, X_test, Y_train, Y_test


def convert_models_to_keras3_format(model_path, output_dir=None, n_models=3):
    """
    Convert legacy TensorFlow SavedModel format to Keras 3-compatible format (.h5).

    Parameters
    ----------
    model_path : str
        Base path to the saved models. Function will append indices (0..n_models-1).
    output_dir : str, optional
        Directory to save converted models. If None, uses original directory.
    n_models : int, optional
        Number of ensemble members to convert. Default 3.

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

    for i in range(n_models):
        try:
            # First try the direct import approach
            logger.info(f"Converting model {i}...")
            
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
            logger.info(f"Successfully converted model {i} to {h5_path}")
            
        except Exception as e:
            logger.warning(f"Error converting model {i}: {e}")
    
    return converted_paths


def load_existing_models(model_path, n_models=3):
    """
    Load pre-trained CNN models in a way compatible with Keras 3.

    This function first attempts to load models directly. If that fails due to
    format compatibility issues, it tries to convert them to a Keras 3-compatible
    format and then load them.

    Parameters
    ----------
    model_path : str
        Base path to the saved models.
    n_models : int, optional
        Number of ensemble members to load (indices 0..n_models-1). Default 3.

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
        for i in range(n_models):
            try:
                # Try to load directly
                model = keras.models.load_model(model_path + str(i), compile=False)
                models.append(model)
                logger.debug(f"Successfully loaded model {i} directly")
            except Exception as e:
                if "File format not supported" in str(e):
                    logger.warning(f"Model {i} format not supported by Keras 3. Will try conversion.")
                    try_conversion = True
                    break
                else:
                    logger.warning(f"Error loading model {i}: {e}")
    except Exception as e:
        logger.warning(f"Error during initial model loading: {e}")
        try_conversion = True
    
    # If direct loading failed, try conversion
    if try_conversion or not models:
        logger.debug("Attempting to convert models to Keras 3-compatible format...")
        try:
            converted_paths = convert_models_to_keras3_format(model_path, n_models=n_models)
            
            # Try to load the converted models
            models = []
            for path in converted_paths:
                try:
                    model = keras.models.load_model(path, compile=False)
                    models.append(model)
                    logger.debug(f"Successfully loaded converted model from {path}")
                except Exception as e:
                    logger.warning(f"Error loading converted model from {path}: {e}")
            
            # If conversion worked, save as the new default
            if models and all(isinstance(m, keras.Model) for m in models):
                logger.info("Successfully converted and loaded models. These will be used for future runs.")
        except Exception as e:
            logger.warning(f"Error during model conversion: {e}")
    logger.debug(f"Returning {len(models)} models of types: {[type(m).__name__ for m in models]}")
    return models

def train_models(models,train_data,results_folder=None):
        
    X_train, Y_train = train_data
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Compile and fine-tune each model with new data
    all_history = []
    for i, model in enumerate(models):
        logger.info(f"Fitting Model {i+1} of {len(models)}")
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mae')
        history = model.fit(np.array(X_train), Y_train, epochs=50, batch_size=24, validation_split=0.1, callbacks=[early_stopping],verbose=1)
        all_history.append(history)
        if results_folder:        
            try:
                model.save(results_folder+f'/first_search/fine_tuning/iRT_updated_model{i}')
            except:
                from src.utils.gui_utils import send_raise_to_TK
                send_raise_to_TK("Path Length Error. To enable long paths, use win+R and type regedit. Navigate to HKEY_LOCAL_MACHINE\ SYSTEM\CurrentControlSet\Control\FileSystem. Set LongPathsEnabled to 1 and restart computer.")
                raise ValueError("Path Length Limit Exceeded")
    logger.debug(f"Returning {len(models)} models of types: {[type(m).__name__ for m in models]}")
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
         

def rt_model_path(tag=None):
    """Resolve the pretrained iRT-CNN base path for the active tag (same mapping
    fine_tune_rt uses internally). Used by the timeplex bagged trainer."""
    current_dir = os.path.dirname(__file__)
    if tag is None:
        tag = config.tag
    if tag is None:
        name = "iRT_CNN_model_LF_09182024_"
    elif "mTRAQ" in tag.name:
        name = "iRT_CNN_model_mTRAQ_09182024_"
    elif "diethyl" in tag.name:
        name = "iRT_CNN_model_DiEthyl_11052024_"
    elif "PSMtag" in tag.name:
        name = "iRT_TransferLearning_Tag6_updated_05072025_"
    else:
        from src.utils.gui_utils import send_raise_to_TK
        logger.info(f"Tag Name: {tag.name}")
        send_raise_to_TK("ValueError - Unknown Label")
        raise ValueError("Unknown label")
    return os.path.join(current_dir, "../rt_models", name)


def fine_tune_rt(grouped_df,
                 qc_plots = False,
                 results_path=None,
                 tag=None):
    
    logger.info(f"{len(grouped_df)} peptides considered for fine tuning")
    
    current_dir = os.path.dirname(__file__)
    
    logger.debug(current_dir)
    
    
    if tag is None:
       tag=config.tag
   
    if tag is None:
        model_path = os.path.join(current_dir,"../rt_models","iRT_CNN_model_LF_09182024_")
        
    elif "mTRAQ" in tag.name:
        model_path = os.path.join(current_dir,"../rt_models","iRT_CNN_model_mTRAQ_09182024_")
        
    elif "diethyl" in tag.name:
        model_path = os.path.join(current_dir,"../rt_models","iRT_CNN_model_DiEthyl_11052024_")
        
    elif "PSMtag" in tag.name:
        model_path = os.path.join(current_dir,"../rt_models","iRT_TransferLearning_Tag6_updated_05072025_")
        
    else:
        from src.utils.gui_utils import send_raise_to_TK
        logger.info(f"Tag Name: {tag.name}")
        send_raise_to_TK("ValueError - Unknown Label")
        raise ValueError("Unknown label")
        
    
    data_split = X_train, X_test, Y_train, Y_test = create_model_data(grouped_df,seq_name='Stripped.Sequence')

    models = load_existing_models(model_path)
    logger.info(f"Loaded {len(models)} models")
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
                logger.debug(f"Attempting prediction with model of type {type(model).__name__}")
                
                if hasattr(model, 'predict'):
                    pred = model.predict(X_test_array)
                    logger.debug(f"Used predict() method, result type: {type(pred).__name__}")
                else:
                    # TFSMLayer objects are called directly
                    logger.debug(f"Calling model directly (TFSMLayer approach)")
                    pred = model(X_test_array)
                    logger.debug(f"Direct call result type: {type(pred).__name__}")
                    
                    # Extract data from tensor dictionary if needed
                    if isinstance(pred, dict):
                        logger.debug(f"Prediction is a dictionary with keys: {list(pred.keys())}")
                        for key in pred:
                            value = pred[key]
                            logger.debug(f"Key '{key}' contains data of type {type(value).__name__}")
                            if hasattr(value, 'numpy'):
                                logger.debug(f"Converting TensorFlow tensor from key '{key}' to NumPy array")
                                pred = value.numpy()
                                logger.debug(f"Shape after conversion: {pred.shape}")
                                break
                    
                # Add shape information
                if hasattr(pred, 'shape'):
                    logger.debug(f"Prediction shape: {pred.shape}")
                else:
                    logger.debug(f"Prediction has no shape attribute, type: {type(pred)}")
                
                model_outputs.append(pred)
                logger.debug(f"Successfully added prediction to model_outputs")
            except Exception as e:
                logger.warning(f"Error predicting with model: {e}")
                logger.warning(f"Traceback: {traceback.format_exc()}")
        
        orig_predictions = np.mean(model_outputs, axis=0)
    except Exception as e:
        logger.warning(f"Error making predictions: {e}")
        # Define fallback
        def model_to_obs(rts):
            return rts
        def obs_to_model(rts):
            return rts
        return data_split, models, model_to_obs
    

    logger.info(f"Number of model outputs collected: {len(model_outputs)}")
    for i, output in enumerate(model_outputs):
        if hasattr(output, 'shape'):
            logger.debug(f"Model output {i} type: {type(output).__name__} shape: {output.shape}")
        else:
            logger.debug(f"Model output {i} type: {type(output).__name__} no shape attribute")

    if not model_outputs:
        logger.warning("No valid predictions from any model!")
    else:
        logger.info(f"Attempting to average {len(model_outputs)} model outputs")

    logger.debug(f"Original predictions shape: {orig_predictions.shape}")
    logger.debug(f"Y_test shape: {Y_test.shape}")
    logger.debug(f"Flattened predictions shape: {orig_predictions.flatten().shape}")
    logger.debug(f"First few values of predictions: {orig_predictions.flatten()[:5]}")
    logger.debug(f"First few values of Y_test: {Y_test[:5]}")

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
            logger.info("Fine Tuning RTs")
            models,history = train_models(models,[X_train,obs_to_model(Y_train)],results_folder=results_path)
        else:
            logger.info("Models don't support fine-tuning (using TFSMLayer). Using for inference only.")
    else:
        logger.info("Not enough IDs for fine tuning")
        logger.info("Using base model for predictions")
        
        ##use all data as validation as none was used for training
        data_split = np.concatenate([X_train,X_test],0),np.concatenate([X_train,X_test],0),np.concatenate([Y_train,Y_test],0),np.concatenate([Y_train,Y_test],0)   
        
    if qc_plots:
           
        plt.subplots()
        plt.scatter(Y_test,orig_predictions,s=1)
        # plt.scatter(Y_test,obs_to_model(Y_test),s=1)
        plt.xlabel("Observed Values")
        plt.ylabel("Base model predictions")
        if results_path:
            plt.savefig(results_path+"/first_search/fine_tuning/RT_finetune_ObsvsOld.png",dpi=600,bbox_inches="tight")
        plt.close()
            
        
        plt.subplots()
        plt.scatter(Y_test,obs_to_model(Y_test),s=1)
        plt.xlabel("Observed Values")
        plt.ylabel("Aligned observed Values")
        if results_path:
            plt.savefig(results_path+"/first_search/fine_tuning/RT_finetune_ObsvsOldAlign.png",dpi=600,bbox_inches="tight")
        plt.close()
        
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
                plt.savefig(results_path+"/first_search/fine_tuning/RT_finetune_AlignObsvsOld.png",dpi=600,bbox_inches="tight")
            plt.close()
            
        
            plt.subplots()
            for h in history:
                plt.plot(h.history["loss"],color="tab:blue",alpha=.5)
                plt.plot(h.history["val_loss"],color="tab:orange",alpha=.5) 
            if results_path:
                plt.savefig(results_path+"/first_search/fine_tuning/RT_finetune_training.png",dpi=600,bbox_inches="tight")
            plt.close()
                 
            
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
                        logger.warning(f"Error predicting with model: {e}")
                        
                predictions = np.mean(model_outputs, axis=0)
                plt.subplots()
                plt.scatter(obs_to_model(Y_test),predictions,s=1)
                x_lim = plt.xlim()
                y_lim = plt.ylim()
                plt.plot(x_lim,x_lim)
                plt.xlabel("Aligned observed Values")
                plt.ylabel("New model predictions")
                if results_path:
                    plt.savefig(results_path+"/first_search/fine_tuning/RT_finetune_AlignObsvsNew.png",dpi=600,bbox_inches="tight")
                plt.close()
            except Exception as e:
                logger.warning(f"Error creating fine-tuned model plot: {e}")

    return data_split, models, model_to_obs


def predict_decoy_rts(decoy_seqs, models, convertor):
    """
    Predict iRT values for decoy sequences using fine-tuned CNN models.

    Parameters
    ----------
    decoy_seqs : list of str
        Decoy peptide sequences (shuffled/reversed).
    models : list
        CNN models returned from fine_tune_rt().
    convertor : callable
        LOWESS calibration function (model_to_obs) from fine_tune_rt().

    Returns
    -------
    np.ndarray
        Predicted iRT values for each decoy sequence.
    """
    logger.info(f"Predicting RTs for {len(decoy_seqs)} decoy sequences")

    # Deduplicate sequences to avoid redundant predictions
    unique_seqs = list(set(decoy_seqs))
    unique_encoded = np.array([one_hot_encode_sequence(s) for s in unique_seqs], dtype=np.float32)

    # Ensemble predictions from all models
    model_outputs = []
    for model in models:
        try:
            if hasattr(model, 'predict'):
                pred = model.predict(unique_encoded, batch_size=4096)
            else:
                pred = model(unique_encoded)
                if isinstance(pred, dict):
                    for key in pred:
                        if hasattr(pred[key], 'numpy'):
                            pred = pred[key].numpy()
                            break
            if hasattr(pred, 'numpy'):
                pred = pred.numpy()
            model_outputs.append(pred)
        except Exception as e:
            logger.warning(f"Error predicting decoy RTs with model: {e}")

    if not model_outputs:
        logger.warning("No valid predictions for decoys, returning None")
        return None

    # Average predictions and calibrate to observed scale
    unique_rts = convertor(np.mean(model_outputs, axis=0).flatten())
    seq_to_rt = dict(zip(unique_seqs, unique_rts))

    # Map back to original order
    predicted_rts = np.array([seq_to_rt[s] for s in decoy_seqs])

    logger.info(f"Updated iRT values for {len(decoy_seqs)} decoys")
    return predicted_rts


# ---------------------------------------------------------------------------
# Timeplex-only bagged, soft-weighted fine-tuning.
#
# Used exclusively by the timeplex RT-alignment path. The non-timeplex path
# (fine_tune_rt / train_models) is unchanged. Validated recipe: MAE + a 5-member
# bagged out-of-bag (OOB) ensemble + iterative two-half-Gaussian (1-PEP) soft
# weighting. All hyperparameters are function defaults (no config/module constants).
# ---------------------------------------------------------------------------
def _two_half_gauss_pep(abs_resid, iters=200):
    """Fit a narrow+wide half-Gaussian mixture to |residual| (the |r| analog of the
    2-component zero-mean GMM) and return per-point PEP = P(wide / false-positive | |r|).

    The per-point PEP is used for soft sample weighting; the component weight is NOT
    a reliable contamination estimate (the wide half-Gaussian absorbs the heavy real
    tail), so only the per-point posterior is used.
    """
    a = np.asarray(abs_resid, dtype=float)
    c = np.sqrt(2.0 / np.pi)
    s1 = max(np.percentile(a, 40), 1e-3)
    s2 = max(np.percentile(a, 95), s1 * 2)
    w = np.array([0.8, 0.2])
    for _ in range(iters):
        f1 = w[0] * c / s1 * np.exp(-a ** 2 / (2 * s1 ** 2))
        f2 = w[1] * c / s2 * np.exp(-a ** 2 / (2 * s2 ** 2))
        g2 = f2 / np.clip(f1 + f2, 1e-300, None)
        g1 = 1.0 - g2
        w = np.array([g1.mean(), g2.mean()])
        s1 = np.sqrt(max((g1 * a ** 2).sum() / max(g1.sum(), 1e-9), 1e-6))
        s2 = np.sqrt(max((g2 * a ** 2).sum() / max(g2.sum(), 1e-9), 1e-6))
        if s1 > s2:
            s1, s2 = s2, s1
            w = w[::-1]
    f1 = w[0] * c / s1 * np.exp(-a ** 2 / (2 * s1 ** 2))
    f2 = w[1] * c / s2 * np.exp(-a ** 2 / (2 * s2 ** 2))
    pep = f2 / np.clip(f1 + f2, 1e-300, None)
    return pep, {"s_narrow": float(s1), "s_wide": float(s2), "w_wide": float(w[1])}


def _freeze_backbone(model):
    """Freeze conv backbone; train only the Dense head."""
    for layer in model.layers:
        layer.trainable = isinstance(layer, keras.layers.Dense)
    return model


def bagged_finetune_channel(seqs, obs_rt, lib_seqs, model_path,
                            n_members=5, subsample=0.6, epochs=22, head_lr=1e-3,
                            patience=7, trim_start=3, ema_beta=0.8, seed=0):
    """Bagged, OOB, soft-weighted fine-tune for one timeplex channel.

    Trains an `n_members`-member ensemble; each member sees a random `subsample`
    fraction (no replacement), so every training point has an out-of-bag (OOB)
    prediction from the members that did not train on it. Each epoch the per-point
    |OOB residual| EMA is fit with a narrow+wide half-Gaussian mixture and each point's
    training weight is set to (1 - PEP). Early stopping uses OOB median-AE.

    The CNN is fine-tuned to predict the channel's OBSERVED RT directly (no LOESS
    calibration / no de-offset), so library predictions are already on the observed-RT
    scale that the downstream search consumes as rt_spls[k][seq].

    Parameters
    ----------
    seqs : array-like of str
        Stripped sequences of the (gated) training apexes for this channel.
    obs_rt : array-like of float
        Observed RT for each training apex.
    lib_seqs : list of str
        Unique library sequences to predict (the returned dict keys).
    model_path : str
        Base path to the pretrained ensemble (indices appended).
    n_members, subsample, epochs, head_lr, patience, trim_start, ema_beta, seed
        Bagged-training hyperparameters (function defaults; no config knobs).

    Returns
    -------
    (oob_resid, lib_pred)
        oob_resid : np.ndarray of held-out OOB residuals (observed - predicted) for the
                    training apexes (for the empirical-vs-fine-tuned comparison and the
                    RT-tolerance GMM).
        lib_pred  : dict {lib_seq: predicted_observed_rt} from the best-epoch ensemble.
        history   : dict with 'oob_medae' (per-epoch held-out OOB MedAE) and
                    'best_epoch' (for the fine-tune training-curve plot).
    """
    import tensorflow as tf

    rng = np.random.default_rng(seed)
    y = np.asarray(obs_rt, dtype=float)
    n = len(y)
    X = np.array([one_hot_encode_sequence(s) for s in seqs], dtype=np.float32)

    models = [_freeze_backbone(m) for m in load_existing_models(model_path, n_models=n_members)]
    n_members = len(models)
    for m in models:
        m.compile(optimizer=keras.optimizers.legacy.Adam(head_lr), loss="mae")

    # bagging membership: each member trains on a `subsample` fraction (no replacement)
    n_sub = max(1, int(round(subsample * n)))
    inbag = np.zeros((n_members, n), bool)
    for k in range(n_members):
        inbag[k, rng.choice(n, n_sub, replace=False)] = True
    oob_mask = (~inbag).astype(np.float64)
    cnt = oob_mask.sum(0)
    has = cnt > 0

    def _predict(model, arr):
        return np.asarray(model(arr, training=False)).flatten()

    def oob_predictions():
        preds = np.array([_predict(m, X) for m in models])   # (n_members, n)
        s1 = (preds * oob_mask).sum(0)
        oob = np.where(has, s1 / np.where(has, cnt, 1), preds.mean(0))
        return oob

    sample_w = np.ones(n)
    ema = None
    oob_medae_hist = []                              # per-epoch held-out OOB MedAE
    best = {"medae": np.inf, "epoch": -1, "resid": y - oob_predictions(),
            "weights": [m.get_weights() for m in models]}
    for ep in range(epochs):
        for k in range(n_members):
            ib = inbag[k]
            models[k].fit(X[ib], y[ib], sample_weight=sample_w[ib],
                          epochs=1, batch_size=24, verbose=0)
        resid = y - oob_predictions()
        medae = float(np.median(np.abs(resid)))
        oob_medae_hist.append(medae)
        a_resid = np.abs(resid)
        ema = a_resid.copy() if ema is None else ema_beta * ema + (1 - ema_beta) * a_resid
        if ep >= trim_start:
            pep, _ = _two_half_gauss_pep(ema)
            sample_w = ema_beta * sample_w + (1 - ema_beta) * (1.0 - pep)
        if medae < best["medae"] - 1e-4:
            best = {"medae": medae, "epoch": ep, "resid": resid.copy(),
                    "weights": [m.get_weights() for m in models]}
        elif ep - best["epoch"] >= patience:
            break
    logger.info(f"  bagged fine-tune: OOB MedAE={best['medae']:.4f} "
                f"(epoch {best['epoch']}, {n} apexes, {n_members} members)")

    # restore best-epoch weights and predict the library on the observed-RT scale
    for m, wts in zip(models, best["weights"]):
        m.set_weights(wts)
    lib_enc = np.array([one_hot_encode_sequence(s) for s in lib_seqs], dtype=np.float32)
    lib_rt = np.mean([m.predict(lib_enc, batch_size=4096, verbose=0).flatten()
                      for m in models], axis=0)
    lib_pred = dict(zip(lib_seqs, lib_rt))
    return best["resid"], lib_pred, {"oob_medae": oob_medae_hist,
                                     "best_epoch": best["epoch"]}