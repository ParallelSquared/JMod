"""
This Source Code Form is subject to the terms of the Oxford Nanopore
Technologies, Ltd. Public License, v. 1.0.  Full licence can be found
at https://github.com/ParallelSquared/JMod/blob/main/LICENSE.txt
"""


import config

import dill
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
import sys
import json

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
        

    # Check if a single argument is provided and it's a JSON file
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        # Treat this as the config_json argument
        config.args.config_json = sys.argv[1]
        print(f"Using configuration file: {config.args.config_json}")

    # Load JSON configuration if specified
    if config.args.config_json:
        print(f"Loading configuration from {config.args.config_json}")
        if not config.load_config_from_json(config.args.config_json):
            print("Failed to load JSON configuration. Using command-line arguments.")

    # Print the configuration that will be used
    print("Using configuration:")
    print(config.args)
    
    ####  Load Libraries   ######################
    mzml_file = config.args.mzml.replace("\\","/")
    lib_file = config.args.speclib.replace("\\","/")

    
    
    spec_file_name = mzml_file.split("/")[-1].rsplit(".",1)[0]
    lib_file_name = lib_file.split("/")[-1].rsplit(".",1)[0]


    use_rt = "RT" if config.args.use_rt else ""
    iso = f"iso{config.num_iso_peaks}" if config.args.iso else ""
    lib_frac = f"iso{config.args.lib_frac}"
    mTRAQ = "mTRAQ" if config.args.mTRAQ else ""
    plexDIA = "plexDIA" if config.args.plexDIA else ""
    tag = config.args.tag
    is_timeplex = "timeplex" if config.args.timeplex else ""
    dummy_val = str(config.args.dummy_value) if config.args.dummy_value else ""
    use_feat = ""
    dino_features=None
    feature_path = os.path.dirname(mzml_file)+"/"+spec_file_name+".features.tsv"
    if config.args.use_features and os.path.exists(feature_path):
        use_feat = "Dino"
        print("loading Dinosaur features")
        dino_features = pd.read_csv(feature_path,delimiter="\t")
    
    ms2_align = "MS2align" if config.args.ms2_align else ""
    results_folder_name = "_".join([spec_file_name,
                                    lib_file_name+"Update130525",
                                    f"{config.mz_ppm}ppm",
                                    f"{config.atleast_m}m",
                                    f"unmatch{config.unmatched_fit_type}",
                                    f"DECOY{config.args.decoy}",
                                    f"libfrac{config.args.lib_frac}",
                                    *list(filter(None,[ms2_align,use_rt,use_feat,iso,tag,plexDIA,is_timeplex,dummy_val]))])
    
    
    print(results_folder_name)
    # print(config.args.tag)
    
    # stop
    results_folder_path = os.path.dirname(mzml_file) +"/" +results_folder_name
    results_folder_path = "/Users/nathanwamsley/Data/JMOD_TESTS/May2025/add_json_051425"
    if config.args.output_folder is not None:
        results_folder_path = config.args.output_folder +"/" +results_folder_name
        
    if not os.path.exists(results_folder_path):
        os.mkdir(results_folder_path)
    
    
    overall_start_time = time.time()
    #python run_jmod.py -r -l /Users/nathanwamsley/Data/SPEC_LIBS/JD_LF_Feb2025/LF_HY_lib.tsv -i /Users/nathanwamsley/Data/mzML/mTRAQ_Feb2025/JD0324.mzML --iso --num_iso 5

    
    ######################################################
    #### Load the data
    spectrumLibrary = SpecLib.loadSpecLib(lib_file)
    DIAspectra=load_files.loadSpectra(mzml_file)

    if config.args.test_mode:
        print(f"Running in test mode with RT range: {config.args.test_rt_min}-{config.args.test_rt_max}, m/z range: {config.args.test_mz_min}-{config.args.test_mz_max}")
        
        # Filter MS2 scans based on retention time and precursor m/z
        filtered_ms2_scans = []
        for scan in DIAspectra.ms2scans:
            if (config.args.test_rt_min <= scan.RT <= config.args.test_rt_max and 
                config.args.test_mz_min <= scan.prec_mz <= config.args.test_mz_max):
                filtered_ms2_scans.append(scan)
        
        print(f"Selected {len(filtered_ms2_scans)} out of {len(DIAspectra.ms2scans)} MS2 scans for test mode")
        DIAspectra.ms2scans = filtered_ms2_scans
        spectra_to_fit = DIAspectra.ms2scans
        
        # Pre-filter the library to speed up processing
        # Note: This is a rough filter that will be refined after RT alignment
        filtered_library = {}
        rt_tolerance = config.rt_tol * 2  # Use a wider tolerance initially
        mz_tolerance = config.mz_tol * 2
        
        for key, entry in spectrumLibrary.items():
            #if (config.args.test_rt_min - rt_tolerance <= entry["iRT"] <= config.args.test_rt_max + rt_tolerance and
            #    config.args.test_mz_min - mz_tolerance*entry["prec_mz"] <= entry["prec_mz"] <= config.args.test_mz_max + mz_tolerance*entry["prec_mz"]):
            if (config.args.test_mz_min - mz_tolerance*entry["prec_mz"] <= entry["prec_mz"] <= config.args.test_mz_max + mz_tolerance*entry["prec_mz"]):
                filtered_library[key] = entry
        
        print(f"Pre-filtered library to {len(filtered_library)} out of {len(spectrumLibrary)} entries for test mode")
        spectrumLibrary = filtered_library
    else:
        print("yo mama")
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
        
        
        
    if config.args.timeplex:
        ## now ooutputs library as we finetune RT
        funcs,spectrumLibrary = MZRTfit_timeplex(DIAspectra, spectrumLibrary, pd.read_csv(feature_path,sep="\t"), config.mz_tol,results_folder=results_folder_path,
                                 ms2=config.args.ms2_align)
        rt_spls,mz_func = funcs[:2]
        
        plex_lib = {}
        rt_mz = []
        for idx in range(len(rt_spls)):
            for key in spectrumLibrary:
                plex_lib[key+(idx,)] = spectrumLibrary[key]
            rt_mz.append([[rt_spls[idx](i["iRT"]), mz_func(i["prec_mz"],i["iRT"])] for i in spectrumLibrary.values()])
        rt_mz = np.concatenate(rt_mz)
        spectrumLibrary = plex_lib
        

        
    else:    
        funcs,spectrumLibrary = MZRTfit(DIAspectra, spectrumLibrary, dino_features, config.mz_tol,results_folder=results_folder_path,
                                        ms2=config.args.ms2_align)
        rt_spl,mz_func = funcs[:2]
        # rt_mz = np.array([[rt_spl(i["iRT"]), mz_func(i["prec_mz"],i["iRT"])] for i in spectrumLibrary.values()])
        rt_mz = np.array([[rt_spl(i["iRT"]), mz_func(i["prec_mz"],i["iRT"])] for i in spectrumLibrary.values()])


    all_keys = list(spectrumLibrary)
     
    
    
    if config.args.ms2_align:
        ms2_func = funcs[2]
        
        for key in all_keys:
            spectrumLibrary[key]["spectrum"][:,0] = ms2_func(spectrumLibrary[key]["spectrum"][:,0])
    else:
        ms2_func=None
        
        
    if config.args.iso:
        # spectrumLibrary = iso_f.iso_library(spectrumLibrary)
        spectrumLibrary = iso_f.iso_library_multi(spectrumLibrary)
        
    # with open(results_folder_path+"/slib","wb") as dill_file:
    #     slib = dill.dump(spectrumLibrary,dill_file)   
      
    print("Creating Decoy Library")
    decoy_lib = SpecLib.create_decoy_lib(spectrumLibrary,rules="rev")
    for key in spectrumLibrary:
        spectrumLibrary[key]["top_n"]=np.argsort(-spectrumLibrary[key]["spectrum"][:,1])[:config.top_n]
    for key in decoy_lib:
        decoy_lib[key]["top_n"]=np.argsort(-decoy_lib[key]["spectrum"][:,1])[:config.top_n]
    print("... Finished Decoy Library")
    
    
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
    
    # with open(results_folder_path+"/dlib","wb") as dill_file:
    #     dlib = dill.dump(decoy_lib,dill_file)   
    
    ######################################################
    ### Start the search
    print("Starting Search")
    # write dia spectra meta data
    ms2scans_info = [[i.prec_mz,i.RT,i.scan_num,*i.ms1window] for i in spectra_to_fit]
    ms2_info_path = results_folder_path+"/ms2scans.csv"
    write_to_csv(ms2scans_info,ms2_info_path)
    
    ## process in batches
    num_batches = 10
    num_per_batch = int(np.ceil(len(spectra_to_fit)/num_batches))
    # start_time = time.time()
    for batch_idx in range(num_batches):
        start_time = time.time()
        batch_spectra = spectra_to_fit[batch_idx*num_per_batch:(batch_idx+1)*num_per_batch]
        
        print(f"Fitting batch {batch_idx+1} of {num_batches}")
        
        outputs= []
        for dia_spec in tqdm.tqdm(batch_spectra):
            
            outputs.append(fit_to_lib2(dia_spec,
                            library=spectrumLibrary,
                            rt_mz=rt_mz,
                            all_keys=all_keys,
                            dino_features=None,
                            rt_filter=True,
                            rt_tol = config.opt_rt_tol,
                            ms1_tol = config.opt_ms1_tol,
                            ms1_spectra=DIAspectra.ms1scans,
                            return_frags=False,
                            decoy=True,
                            decoy_library=decoy_lib))
            
        long_outputs = [j for i in outputs for j in i]
        print(f"Fit {len(batch_spectra)} spectra in {(round(time.time()-start_time))//60} mins and {(round(time.time()-start_time))%60} sec")
        
        decoylib_search_path = results_folder_path+"/decoylibsearch_coeffs.csv"
        write_to_csv(long_outputs,decoylib_search_path)
        
    
    
    process_data(file=decoylib_search_path,
                 spectra=DIAspectra,
                 library=spectrumLibrary,
                 mass_tag=mass_tag,
                 timeplex=config.args.timeplex)
    
    # """
    