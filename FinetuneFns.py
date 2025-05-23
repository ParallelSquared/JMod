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
import os

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




def load_existing_models(path):
        
    # Load the models for prediction without compiling
    models = []
    num_models = 5  # We have 5 models
    
    for i in range(num_models):
        model = load_model(path+str(i), compile=False)
        # model = tf.keras.layers.TFSMLayer(f"/Volumes/Lab/KMD/JD_RT_copy/iRT_model_mTRAQ_09042024_{i}", call_endpoint="serving_default")
        models.append(model)
        
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
    
    current_dir = os.path.dirname(__file__)
    
    if tag is None:
        tag=config.tag
    
    if tag is None:
        model_path = os.path.join(current_dir,"rt_models","iRT_CNN_model_LF_09182024_")
        
    elif tag.name=="mTRAQ":
        model_path = os.path.join(current_dir,"rt_models","iRT_CNN_model_mTRAQ_09182024_")
        
    elif "diethyl" in tag.name:
       model_path = os.path.join(current_dir,"rt_models","iRT_CNN_model_DiEthyl_11052024_")
        
    elif "tag6" in tag.name:
        # model_path = "/Volumes/Lab/JD/Predictions/CNN/iRT_TransferLearning_Tag6_updated_"
        model_path = os.path.join(current_dir,"rt_models","iRT_TransferLearning_Tag6_updated_05072025_")
        
    else:
        raise ValueError("Unknown label")
        
    
    data_split = X_train, X_test, Y_train, Y_test = create_model_data(grouped_df,seq_name='Stripped.Sequence')

    models = load_existing_models(model_path)
    
    # Perform LOESS regression on the filtered data
    ###!!!  NB: Fit/training is very much dependent on the loess fit; Make sure it's monotonic and nott overfit also (edit "frac")
    lowess = sm.nonparametric.lowess
    
    orig_predictions = np.mean([model.predict(np.array(X_test)) for model in models],axis=0)
    
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
