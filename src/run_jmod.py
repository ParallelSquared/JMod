
"""
This Source Code Form is subject to the terms of the Oxford Nanopore
Technologies, Ltd. Public License, v. 1.0.  Full licence can be found
at https://github.com/ParallelSquared/JMod/blob/main/LICENSE.txt
"""

# Update imports to relative imports
from . import config
import numpy as np
import os
import time 
import tqdm
import pandas as pd
import sys 
import json
import biosaur2

from .utils.io import load_files
from .utils.set_seeds import set_seeds
from .models.spec_lib import spec_lib
from .spectral_fitting import fit_to_lib2
from .rt_alignment import MZRTfit, MZRTfit_timeplex
from .utils.misc_functions import write_to_csv
from . import iso_functions as iso_f
from .mass_tags import tag_library, available_tags
from .fdr_analysis import process_data

from src.logger import logger, set_log_filepath
import logging

def main(GUI_config_json = None):
    """Main function to run JMod analysis."""

    # Check if a single argument is provided and it's a JSON file
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        # Treat this as the config_json argument
        config.args.config_json = sys.argv[1]

    if GUI_config_json:
        config.ran_from_GUI = True
        config.args.config_json = GUI_config_json

    # Load JSON configuration if specified
    if config.args.config_json:
        if not config.load_config_from_json(config.args.config_json):
            pass

    # Check if running in test mode
    if len(sys.argv) > 1 and sys.argv[1] in ['--test', '-t', 'test']:
        # Run tests instead of normal operation
        import subprocess
        
        # Remove the test argument and pass remaining args to test runner
        test_args = sys.argv[2:] if len(sys.argv) > 2 else []
        cmd = [sys.executable, "run_tests.py"] + test_args
        
        result = subprocess.run(cmd)
        sys.exit(result.returncode)

    ####  Load Libraries   ######################
    set_seeds(config.RANDOM_SEED)
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
        dino_features = pd.read_csv(feature_path,delimiter="\t")

    if config.args.use_features and not os.path.exists(feature_path):
        import subprocess
        subprocess.run(["biosaur2", mzml_file], check=True)
        use_feat = "Dino"
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
    

    results_folder_path = os.path.dirname(mzml_file) +"/" +results_folder_name
    # results_folder_path = "/Users/nathanwamsley/Data/JMOD_TESTS/May2025/add_json_timeplex_051425_01"
    if config.args.output_folder is not None:
        results_folder_path = config.args.output_folder +"/" +results_folder_name

    if not os.path.exists(results_folder_path):
        try:
            os.mkdir(results_folder_path)
        except:
            from run_jmod_from_GUI import send_raise_to_TK
            send_raise_to_TK("Path Length Error. To enable long paths, use win+R and type regedit. Navigate to HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem. Set LongPathsEnabled to 1 and restart computer.")
            raise ValueError("Path Length Limit Exceeded")
        
    if len(results_folder_path) >= 225:  ##if results path is long, check to make sure putting things in it wont break (i.e. windows with long paths enabled or different OS)
        try:
            test_path = os.path.join(results_folder_path, "a" * 250 + ".txt")
            with open(test_path, "w") as f:
                f.write("test")
            os.remove(test_path)
        except:
            from run_jmod_from_GUI import send_raise_to_TK
            send_raise_to_TK("Path Length Error. To enable long paths, use win+R and type regedit. Navigate to HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem. Set LongPathsEnabled to 1 and restart computer.")
            raise ValueError("Path Length Limit Exceeded")

    

    args_dict = vars(config.args)
    json_path = os.path.join(results_folder_path, "config.json")
    with open(json_path, "w") as f:
        json.dump(args_dict, f, indent=4)
    
    logfile_path = os.path.join(results_folder_path, "Log.log")
    set_log_filepath(logfile_path)




    logger.debug(config.args)
    ##add statements to log once results folder has been created
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        logger.info(f"Using configuration file: {config.args.config_json}")
    if config.args.config_json:
        logger.info(f"Loading configuration from {config.args.config_json}")
        if not config.load_config_from_json(config.args.config_json):
            logger.warning("Failed to load JSON configuration. Using command-line arguments.")
    if GUI_config_json:
        config.args.config_json = GUI_config_json
        logger.info(f"Loading configuration from {config.args.config_json}")
    if len(sys.argv) > 1 and sys.argv[1] in ['--test', '-t', 'test']:
        logger.info("Running JMod in test mode...")

    # Log the configuration that will be used
    logger.info("Using configuration:")
    logger.info(config.args)


    if config.args.use_features and os.path.exists(feature_path):
        logger.info("loading Dinosaur features")
    if config.args.use_features and not os.path.exists(feature_path):
        logger.info("Dinosaur feature file not found, running biosaur2")

    logger.info(f"Results will be saved to {results_folder_path}")

    
    # logger.info(config.args.tag)
    
    # stop
    
    
    
    overall_start_time = time.time()
    # python run_jmod.py -r -l /Users/nathanwamsley/Data/SPEC_LIBS/JD_LF_Feb2025/LF_HY_lib.tsv -i /Users/nathanwamsley/Data/mzML/mTRAQ_Feb2025/JD0324.mzML --iso --num_iso 5

    
    ######################################################
    #### Load the data
    spectrumLibrary = spec_lib.loadSpecLib(lib_file)
    DIAspectra=load_files.loadSpectra(mzml_file)

    if config.args.test_mode:
        logger.info(f"Running in test mode with RT range: {config.args.test_rt_min}-{config.args.test_rt_max}, m/z range: {config.args.test_mz_min}-{config.args.test_mz_max}")
        
        # Filter MS2 scans based on retention time and precursor m/z
        filtered_ms2_scans = []
        for scan in DIAspectra.ms2scans:
            if (config.args.test_rt_min <= scan.RT <= config.args.test_rt_max and 
                config.args.test_mz_min <= scan.prec_mz <= config.args.test_mz_max):
                filtered_ms2_scans.append(scan)
        
        logger.info(f"Selected {len(filtered_ms2_scans)} out of {len(DIAspectra.ms2scans)} MS2 scans for test mode")
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
        
        logger.info(f"Pre-filtered library to {len(filtered_library)} out of {len(spectrumLibrary)} entries for test mode")
        spectrumLibrary = filtered_library
    else:
        spectra_to_fit = DIAspectra.ms2scans
    ######################################################
    #### RT/MZ Alignment #####
    
    # rtSpl = RTfit(spectra_to_fit,spectrumLibrary,config.mz_tol)
    # rt_mz = np.array([[rtSpl(i["iRT"]), i["prec_mz"]] for i in spectrumLibrary.values()])
    # rt_mz = np.array([[i["iRT"], i["prec_mz"]] for i in spectrumLibrary.values()])
    
    if config.args.tag:
        # Find the tag object based on the tag name
        if config.args.tag in available_tags:
            config.tag = available_tags[config.args.tag]
            logger.info(f"Using tag: {config.tag.name}")
            spectrumLibrary = tag_library(spectrumLibrary, config.tag)
            mass_tag = config.tag
        else:
            if config.args.tag != "None":
                from run_jmod_from_GUI import send_raise_to_TK
                send_raise_to_TK(f"Error: Tag '{config.args.tag}' not found in available_tags.")
                logger.error(f"Available tags: {list(available_tags.keys())}")
                raise ValueError("Tag Not Found")
            mass_tag = None
            config.tag = None
    else:
        mass_tag = None
        config.tag = None

    if config.args.timeplex:
        ## now ooutputs library as we finetune RT
        # With this:
        if config.args.use_features and os.path.exists(feature_path):
            logger.info("Loading Dinosaur features")
            dino_features = pd.read_csv(feature_path, delimiter="\t")
            funcs, spectrumLibrary = MZRTfit_timeplex(DIAspectra, spectrumLibrary, dino_features, config.mz_tol, results_folder=results_folder_path,
                                            ms2=config.args.ms2_align)
        else:
            logger.info("Not using features")
            funcs, spectrumLibrary = MZRTfit_timeplex(DIAspectra, spectrumLibrary, None, config.mz_tol, results_folder=results_folder_path,
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
      
    logger.info("Creating Decoy Library")
    decoy_lib = spec_lib.create_decoy_lib(spectrumLibrary,rules="rev")
    for key in spectrumLibrary:
        spectrumLibrary[key]["top_n"]=np.argsort(-spectrumLibrary[key]["spectrum"][:,1])[:config.top_n]
    for key in decoy_lib:
        decoy_lib[key]["top_n"]=np.argsort(-decoy_lib[key]["spectrum"][:,1])[:config.top_n]
    logger.info("... Finished Decoy Library")
    
    
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
    logger.info("Starting Search")
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
        
        logger.info(f"Fitting batch {batch_idx+1} of {num_batches}")
        
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
        logger.info(f"Fit {len(batch_spectra)} spectra in {(round(time.time()-start_time))//60} mins and {(round(time.time()-start_time))%60} sec")
        
        decoylib_search_path = results_folder_path+"/decoylibsearch_coeffs.csv"
        write_to_csv(long_outputs,decoylib_search_path)
    


    process_data(file=decoylib_search_path,
                 spectra=DIAspectra,
                 library=spectrumLibrary,
                 mass_tag=mass_tag,
                 timeplex=config.args.timeplex)
    # """
