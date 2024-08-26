
import config


import numpy as np
import argparse
import os
import pickle
import multiprocessing
import time 
import tqdm
import csv
from functools import partial
import pandas as pd

import load_files 
import SpecLib
import Jplot as jp

from SpectraFitting import fit_to_lib, fit_to_lib2, fit_to_lib_decoy
from rtAlignment import MZRTfit, MZRTfit_timeplex
from miscFunctions import write_to_csv
import iso_functions as iso_f
from mass_tags import tag_library
from fdr_analysis import process_data

if __name__=="__main__":
            

    print(config.args)
    
    ####  Load Libraries   ######################
    mzml_file = config.args.mzml.replace("\\","/")
    lib_file = config.args.speclib.replace("\\","/")

    
    
    spec_file_name = mzml_file.split("/")[-1].rsplit(".",1)[0]
    lib_file_name = lib_file.split("/")[-1].rsplit(".",1)[0]


    
    use_rt = "RT" if config.args.use_rt else ""
    iso = f"iso{config.num_iso_peaks}" if config.args.iso else ""
    mTRAQ = "mTRAQ" if config.args.mTRAQ else ""
    tag = config.args.tag
    is_timeplex = "timeplex" if config.args.timeplex else ""
    use_feat = ""
    dino_features=None
    feature_path = os.path.dirname(mzml_file)+"/"+spec_file_name+".features.tsv"
    if config.args.use_features and os.path.exists(feature_path):
        use_feat = "Dino"
        print("loading Dinosaur features")
        dino_features = pd.read_csv(feature_path,delimiter="\t")
    
    ms2_align = "MS2align" if config.args.ms2_align else ""
    results_folder_name = "_".join([spec_file_name,
                                    lib_file_name+"Update060824",
                                    f"{config.mz_ppm}ppm",
                                    f"{config.atleast_m}m",
                                    f"unmatch{config.unmatched_fit_type}",
                                    f"DECOY{config.args.decoy}",
                                    *list(filter(None,[ms2_align,use_rt,use_feat,iso,tag,is_timeplex]))])
    
    results_folder_path = "/Users/kevinmcdonnell/Programming/Python/Jmod/Results/"+results_folder_name
    print(results_folder_name)
    # print(config.args.tag)
    
    # stop
    results_folder_path = os.path.dirname(mzml_file) +"/" +results_folder_name
    #results_folder_path = "/Volumes/Lab/KMD/JmodResults/"+results_folder_name
    #results_folder_path = "/Users/kevinmcdonnell/Programming/Data/Results/"+results_folder_name
    results_folder_path = "add/path/here/"
    if not os.path.exists(results_folder_path):
        os.mkdir(results_folder_path)
    
    
    overall_start_time = time.time()
    
    
    ######################################################
    #### Load the data
    spectrumLibrary = SpecLib.loadSpecLib(lib_file)
    
    DIAspectra=load_files.loadSpectra(mzml_file)
    spectra_to_fit = DIAspectra.ms2scans
    
    ######################################################
    #### RT/MZ Alignment #####
    
    # rtSpl = RTfit(spectra_to_fit,spectrumLibrary,config.mz_tol)
    # rt_mz = np.array([[rtSpl(i["iRT"]), i["prec_mz"]] for i in spectrumLibrary.values()])
    # rt_mz = np.array([[i["iRT"], i["prec_mz"]] for i in spectrumLibrary.values()])
    
    if config.args.tag:
        spectrumLibrary = tag_library(spectrumLibrary,config.tag)
        mass_tag = config.tag    
    else:
        mass_tag = None
        
        
    all_keys = list(spectrumLibrary)
        
        
    if config.args.timeplex:
        funcs = MZRTfit_timeplex(DIAspectra, spectrumLibrary, pd.read_csv(feature_path,sep="\t"), config.mz_tol,results_folder=results_folder_path,
                                 ms2=config.args.ms2_align)
        rt_spls,mz_func = funcs[:2]
        
        plex_lib = {}
        rt_mz = []
        for idx in range(len(rt_spls)):
            for key in spectrumLibrary:
                plex_lib[key+(idx,)] = spectrumLibrary[key]
            rt_mz.append([[rt_spls[idx](i["iRT"]), mz_func(i["prec_mz"])] for i in spectrumLibrary.values()])
        rt_mz = np.concatenate(rt_mz)
        spectrumLibrary = plex_lib
        
    else:    
        funcs = MZRTfit(DIAspectra, spectrumLibrary, dino_features, config.mz_tol,results_folder=results_folder_path,
                        ms2=config.args.ms2_align)
        rt_spl,mz_func = funcs[:2]
        # rt_mz = np.array([[rt_spl(i["iRT"]), mz_func(i["prec_mz"],i["iRT"])] for i in spectrumLibrary.values()])
        rt_mz = np.array([[rt_spl(i["iRT"]), mz_func(i["prec_mz"])] for i in spectrumLibrary.values()])

    if config.args.ms2_align:
        ms2_func = funcs[2]
        
        for key in all_keys:
            spectrumLibrary[key]["spectrum"][:,0] = ms2_func(spectrumLibrary[key]["spectrum"][:,0])
    else:
        ms2_func=None
        
        
    if config.args.iso:
        spectrumLibrary = iso_f.iso_library(spectrumLibrary)
        
        
    ######################################################
    ### Write search params to file
    param_file = results_folder_path + "/params.txt"
    with open(param_file,"w+") as write_file:
        write_file.writelines("Args\n")
        for key,item in vars(config.args).items():
            write_file.writelines(f"{key}: {item}\n")
        
        config_exclude = ["diann_mods","argparse", "parser","args"]
        write_file.writelines("\nConfig\n")
        for key,item in config.__dict__.items():
            if key[:2] != "__" and key not in config_exclude:
                write_file.writelines(f"{key}: {item}\n")
    
    
    
    ######################################################
    ### Start the search
    
    # write dia spectra meta data
    ms2scans_info = [[i.prec_mz,i.RT,i.scan_num,*i.ms1window] for i in spectra_to_fit]
    ms2_info_path = results_folder_path+"/ms2scans.csv"
    write_to_csv(ms2scans_info,ms2_info_path)
    
    
    """
    # ##fit to just library
    start_time = time.time()
    with multiprocessing.Pool(config.numProc) as p:
        
        outputs = list(tqdm.tqdm(p.imap(partial(fit_to_lib,
                                                library=spectrumLibrary,
                                                rt_mz=rt_mz,
                                                all_keys=all_keys,
                                                dino_features=dino_features,
                                                rt_filter=use_rt,
                                                rt_tol = config.opt_rt_tol,
                                                ms1_tol = config.opt_ms1_tol),
                                        spectra_to_fit,chunksize=config.chunksize),total=len(spectra_to_fit)))

        long_outputs = [j for i in outputs for j in i]
    print(f"Fit {len(spectra_to_fit)} spectra in {round(time.time()-start_time)} sec")

    lib_search_path = results_folder_path+"/libsearch_coeffs.csv"
    write_to_csv(long_outputs,lib_search_path)
    
    # """
    # ##fit to decoy library
    start_time = time.time()
    with multiprocessing.Pool(config.numProc) as p:
        
        ## process in batches
        num_batches = 10
        num_per_batch = int(np.ceil(len(spectra_to_fit)/num_batches))
        
        for batch_idx in range(num_batches):
            batch_spectra = spectra_to_fit[batch_idx*num_per_batch:(batch_idx+1)*num_per_batch]
            
            outputs = list(tqdm.tqdm(p.imap(partial(fit_to_lib2,
                                                    library=spectrumLibrary,
                                                    rt_mz=rt_mz,
                                                    all_keys=all_keys,
                                                    dino_features=dino_features,
                                                    rt_filter=use_rt,
                                                    rt_tol = config.opt_rt_tol,
                                                    ms1_tol = config.opt_ms1_tol,
                                                    ms1_spectra=DIAspectra.ms1scans,
                                                    return_frags=False,
                                                    decoy=True),
                                            batch_spectra,chunksize=config.chunksize),total=len(batch_spectra)))
    
            long_outputs = [j for i in outputs for j in i]
            print(f"Fit {len(spectra_to_fit)} spectra in {round(time.time()-start_time)} sec")
            
            decoylib_search_path = results_folder_path+"/decoylibsearch_coeffs.csv"
            write_to_csv(long_outputs,decoylib_search_path)
    
    process_data(file=decoylib_search_path,
                 spectra=DIAspectra,
                 library=spectrumLibrary,
                 mass_tag=mass_tag,
                 timeplex=config.args.timeplex)
    # """
    
