
"""
This Source Code Form is subject to the terms of the Oxford Nanopore
Technologies, Ltd. Public License, v. 1.0.  Full licence can be found
at https://github.com/ParallelSquared/JMod/blob/main/LICENSE.txt
"""

# Update imports to relative imports
import src.config as config
import numpy as np
import os
import time
import datetime
import tqdm
import pandas as pd
import sys
import json
import dill
import resource


def _mem_gb():
    """Return current RSS in GB (not peak) via /proc or ps fallback."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024**3)
    except ImportError:
        pass
    if sys.platform == 'darwin':
        import subprocess
        # ps reports RSS in KB on macOS
        out = subprocess.check_output(['ps', '-o', 'rss=', '-p', str(os.getpid())])
        return int(out.strip()) / (1024**2)
    # Linux: read from /proc
    with open(f'/proc/{os.getpid()}/statm') as f:
        pages = int(f.read().split()[1])
    return pages * 4096 / (1024**3)


def _peak_mem_gb():
    """Return peak (high water mark) RSS in GB."""
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == 'darwin':
        return maxrss / (1024**3)
    return maxrss / (1024**2)


def _log_mem(label):
    logger.info(f"[MEM] {label}: {_mem_gb():.2f} GB current, {_peak_mem_gb():.2f} GB peak")


def _log_mem_breakdown(*libs):
    """Log per-array memory for each (name, library) pair."""
    import gc
    gc.collect()
    for lib_name, lib in libs:
        logger.info(f"\n=== {lib_name} library ({len(lib)} entries) ===")
        total = 0
        obj_total = 0
        for attr in sorted(lib.__slots__):
            obj = getattr(lib, attr, None)
            if obj is None:
                continue
            if isinstance(obj, dict):
                size = sys.getsizeof(obj)
                logger.info(f"  {attr:30s} {size/1e9:8.3f} GB  (dict, {len(obj)} keys)")
                total += size
            elif isinstance(obj, np.ndarray):
                nbytes = obj.nbytes
                logger.info(f"  {attr:30s} {nbytes/1e9:8.3f} GB  dtype={obj.dtype}  shape={obj.shape}")
                total += nbytes
                if obj.dtype == object and len(obj) > 0:
                    # sample to estimate true size of contained objects
                    sample_size = min(10000, len(obj))
                    sample_idx = np.linspace(0, len(obj)-1, sample_size, dtype=int)
                    sample_bytes = sum(sys.getsizeof(obj[i]) for i in sample_idx if obj[i] is not None)
                    est_true = (sample_bytes / sample_size) * len(obj)
                    logger.info(f"  {attr:30s} ~{est_true/1e9:7.3f} GB  (estimated object contents)")
                    obj_total += est_true
            else:
                size = sys.getsizeof(obj)
                logger.info(f"  {attr:30s} {size/1e9:8.3f} GB  type={type(obj).__name__}")
                total += size
        logger.info(f"  {'TOTAL (array buffers)':30s} {total/1e9:8.3f} GB")
        logger.info(f"  {'TOTAL (object contents est)':30s} {obj_total/1e9:8.3f} GB")
        logger.info(f"  {'TOTAL (combined est)':30s} {(total+obj_total)/1e9:8.3f} GB")

from src.utils.io import load_files
from src.utils.set_seeds import set_seeds
from src.models.spec_lib import spec_lib
from src.spectral_fitting import fit_to_lib2, merge_spectrum_peaks
from src.rt_alignment import MZRTfit, MZRTfit_timeplex
from src.utils.misc_functions import write_to_csv
from src import iso_functions as iso_f
from src.mass_tags import tag_library, available_tags
from src.fdr_analysis import process_data

from src.logger import logger, set_log_filepath, log_exceptions
import logging

@log_exceptions
def main(GUI_config_json=None, GUI_result_queue=None):
    """Main function to run JMod analysis."""

    # Check if a single argument is provided and it's a JSON file
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        # Treat this as the config_json argument
        config.args.config_json = sys.argv[1]

    if GUI_config_json:
        config.ran_from_GUI = True
        config.error_already_handled = False
        config.GUI_result_queue = GUI_result_queue
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
    feature_path = os.path.dirname(mzml_file)+"/"+spec_file_name+".features.tsv" #TODO this breaks if you run from cd
    if config.args.use_features and os.path.exists(feature_path):
        use_feat = "Dino"
        dino_features = pd.read_csv(feature_path,delimiter="\t")

    if config.args.use_features and not os.path.exists(feature_path) and config.args.timeplex:
        import subprocess
        subprocess.run(["biosaur2", mzml_file], check=True)
        use_feat = "Dino"
        dino_features = pd.read_csv(feature_path,delimiter="\t")

    results_folder_name = spec_file_name + "_results" + "_" + dummy_val
    results_folder_name = results_folder_name.rstrip("_")

    if config.args.output_folder is not None:
        os.makedirs(config.args.output_folder, exist_ok=True)
        results_folder_path = os.path.join(config.args.output_folder, results_folder_name)
    else:
        results_folder_path = os.path.join(os.path.dirname(mzml_file), results_folder_name)

    if os.path.exists(results_folder_path):
        datestamp = str(datetime.datetime.now())
        datestamp = datestamp.split()
        datestamp = datestamp[0].replace("-", "_") + "_" + datestamp[1].split(".")[0].replace(":", "_")
        results_folder_path = results_folder_path + "_" + datestamp


    if not os.path.exists(results_folder_path):
        try:
            os.mkdir(results_folder_path)
        except FileNotFoundError as e:
            if not os.path.exists(os.path.dirname(results_folder_path)):
                from src.utils.gui_utils import send_raise_to_TK
                send_raise_to_TK(f"Error Creating Results Folder. Parent path does not exist.\nPath: {os.path.dirname(results_folder_path)}")
                raise FileNotFoundError(f"Parent Path Does Not Exist - {os.path.dirname(results_folder_path)}")
            if "[WinError 3]" in str(e) or "[WinError 206]" in str(e):
                from src.utils.gui_utils import send_raise_to_TK
                send_raise_to_TK("Path Length Error. To enable long paths, use win+R and type regedit. Navigate to HKEY_LOCAL_MACHINE\ SYSTEM\CurrentControlSet\Control\FileSystem. Set LongPathsEnabled to 1 and restart computer.")
                raise ValueError("Path Length Limit Exceeded")
        except Exception as e:
            from src.utils.gui_utils import send_raise_to_TK
            send_raise_to_TK(f"Error Creating Results Folder. Please check the path is valid.\nPath: {results_folder_path}\nError: \n{str(e)}")
            raise e

        
    if len(results_folder_path) >= 225:  ##if results path is long, check to make sure putting things in it wont break (i.e. windows with long paths enabled or different OS)
        try:
            test_path = os.path.join(results_folder_path, "a" * 250 + ".txt")
            with open(test_path, "w") as f:
                f.write("test")
            os.remove(test_path)
        except FileNotFoundError as e:
            if "[WinError 3]" in str(e) or "[WinError 206]" in str(e):
                from src.utils.gui_utils import send_raise_to_TK
                send_raise_to_TK("Path Length Error. To enable long paths, use win+R and type regedit. Navigate to HKEY_LOCAL_MACHINE\ SYSTEM\CurrentControlSet\Control\FileSystem. Set LongPathsEnabled to 1 and restart computer.")
                raise ValueError("Path Length Limit Exceeded")
        except Exception as e:
            from src.utils.gui_utils import send_raise_to_TK
            send_raise_to_TK(f"Error Creating Results Folder. Please check the path is valid.\nPath: {results_folder_path}\nError:\n{str(e)}")
            raise e

    

    args_dict = vars(config.args)
    json_path = os.path.join(results_folder_path, "config.json")
    with open(json_path, "w") as f:
        json.dump(args_dict, f, indent=4)
    
    logfile_path = os.path.join(results_folder_path, "Log.log")
    set_log_filepath(logfile_path)
    config.results_folder_path = results_folder_path



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
        logger.info(f"Loading configuration from GUI")
    if len(sys.argv) > 1 and sys.argv[1] in ['--test', '-t', 'test']:
        logger.info("Running JMod in test mode...")

    # Log the configuration that will be used
    logger.info("Using configuration:")
    logger.info(config.args)
    logger.info("")


    if config.args.use_features and os.path.exists(feature_path) and config.args.timeplex:
        logger.info("loading Dinosaur features")
    if config.args.use_features and not os.path.exists(feature_path) and config.args.timeplex:
        logger.info("Dinosaur feature file not found, running biosaur2")

    
    overall_start_time = time.time()
    # python run_jmod.py -r -l /Users/nathanwamsley/Data/SPEC_LIBS/JD_LF_Feb2025/LF_HY_lib.tsv -i /Users/nathanwamsley/Data/mzML/mTRAQ_Feb2025/JD0324.mzML --iso --num_iso 5
    logger.info(f"Results will be saved to {os.path.abspath(results_folder_path)}")

    
    ######################################################
    #### Load the data
    spectrumLibrary = spec_lib.loadSpecLib(lib_file)
    _log_mem("after loading library")
    DIAspectra=load_files.loadSpectra(mzml_file)
    _log_mem("after loading spectra")

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
    
    if config.args.SILAC:
        # Find the tag object based on the tag name
        if config.args.SILAC in available_tags:
            config.SILAC = available_tags[config.args.SILAC]
            logger.info(f"Using SILAC: {config.SILAC.name} - {config.SILAC.n_channels} channels")
            spectrumLibrary = tag_library(spectrumLibrary, config.SILAC)
            SILAC = config.SILAC
        else:
            if config.args.SILAC != "None":
                from src.utils.gui_utils import send_raise_to_TK
                send_raise_to_TK(f"Error: SILAC '{config.args.SILAC}' not found in available_tags.")
                logger.error(f"Available tags: {list(available_tags.keys())}")
                raise ValueError("Tag Not Found")
            SILAC = None
            config.SILAC = None
    else:
        SILAC = None
        config.SILAC = None
        
    if config.args.tag:
        # Find the tag object based on the tag name
        if config.args.tag in available_tags:
            config.tag = available_tags[config.args.tag]
            logger.info(f"Using tag: {config.tag.name} - {config.tag.n_channels} channels")
            spectrumLibrary = tag_library(spectrumLibrary, config.tag)
            mass_tag = config.tag
        else:
            if config.args.tag != "None":
                from src.utils.gui_utils import send_raise_to_TK
                send_raise_to_TK(f"Error: Tag '{config.args.tag}' not found in available_tags.")
                logger.error(f"Available tags: {list(available_tags.keys())}")
                raise ValueError("Tag Not Found")
            mass_tag = None
            config.tag = None
    else:
        mass_tag = None
        config.tag = None

    _log_mem("before RT alignment")
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
                                        ms2=config.args.ms2_align, mass_tag=mass_tag, SILAC=SILAC)
        rt_spl,mz_func = funcs[:2]
        # rt_mz = np.array([[rt_spl(i["iRT"]), mz_func(i["prec_mz"],i["iRT"])] for i in spectrumLibrary.values()])
        rt_mz = np.array([[rt_spl(i["iRT"]), mz_func(i["prec_mz"],i["iRT"])] for i in spectrumLibrary.values()]) # TODO QQQXXX Input should have at least 1 dimension, got scalar array(-16.) instead


    _log_mem("after RT alignment")

    ## Merge peaks in spectra
    for spec in DIAspectra.ms1scans:
        merge_spectrum_peaks(spec, config.opt_ms1_tol)

    for spec in DIAspectra.ms2scans:
        merge_spectrum_peaks(spec, config.mz_tol)

    spectra_to_fit = DIAspectra.ms2scans
    _log_mem("after merge_spectrum_peaks")

    all_keys = list(spectrumLibrary)
     


    if config.args.ms2_align:
        ms2_func = funcs[2]
        
        for key in all_keys:
            spectrumLibrary[key]["spectrum"][:,0] = ms2_func(spectrumLibrary[key]["spectrum"][:,0])
    else:
        ms2_func=None


    _log_mem("before isotope generation")
    if config.args.iso:
        # spectrumLibrary = iso_f.iso_library(spectrumLibrary,
        #                                            tag=config.tag,
        #                                            n_iso=config.num_iso_peaks)
        spectrumLibrary = iso_f.iso_library_multi(spectrumLibrary,
                                                  tag=config.tag,
                                                  n_iso=config.num_iso_peaks)
        _log_mem("after isotope generation")

    # with open(results_folder_path+"/slib","wb") as dill_file:
    #     slib = dill.dump(spectrumLibrary,dill_file)

    logger.info("Creating Decoy Library")
    decoy_lib = spec_lib.create_decoy_lib(spectrumLibrary,rules="rev",tag=config.tag,n_iso=config.num_iso_peaks)
    _log_mem("after decoy library creation")
    spectrumLibrary.bulk_set_top_n(config.top_n)
    decoy_lib.bulk_set_top_n(config.top_n)
    logger.info("Finished Decoy Library")
    _log_mem("after bulk_set_top_n")

    _log_mem_breakdown(("target", spectrumLibrary), ("decoy", decoy_lib))

    # ---- Audit known large objects ----
    import gc as _gc
    _gc.collect()
    logger.info("\n=== Memory audit of known locals ===")
    # DIAspectra: sum up the numpy arrays inside each Spectrum
    _spec_bytes = 0
    for _s in DIAspectra.ms1scans + DIAspectra.ms2scans:
        for _attr in ('mz', 'intens'):
            _arr = getattr(_s, _attr, None)
            if _arr is not None:
                _spec_bytes += _arr.nbytes
    _n_scans = len(DIAspectra.ms1scans) + len(DIAspectra.ms2scans)
    logger.info(f"  DIAspectra arrays:     {_spec_bytes/1e9:.2f} GB  ({_n_scans:,} scans)")
    logger.info(f"  DIAspectra obj est:   ~{_n_scans * 500 /1e9:.2f} GB  ({_n_scans:,} × ~500B overhead)")
    if dino_features is not None:
        logger.info(f"  dino_features:         {dino_features.memory_usage(deep=True).sum()/1e9:.2f} GB")
    else:
        logger.info(f"  dino_features:         None")
    logger.info(f"  rt_mz:                 {rt_mz.nbytes/1e9:.4f} GB")
    logger.info(f"  all_keys:              {sys.getsizeof(all_keys)/1e9:.4f} GB")
    # Check what the closures inside funcs might be holding
    for _i, _f in enumerate(funcs):
        _closure_bytes = 0
        if hasattr(_f, '__closure__') and _f.__closure__:
            for _cell in _f.__closure__:
                try:
                    _obj = _cell.cell_contents
                    _closure_bytes += sys.getsizeof(_obj)
                    if hasattr(_obj, 'nbytes'):
                        _closure_bytes += _obj.nbytes
                except ValueError:
                    pass
            logger.info(f"  funcs[{_i}] closure:     ~{_closure_bytes/1e9:.4f} GB  ({len(_f.__closure__)} cells)")
        else:
            logger.info(f"  funcs[{_i}]:              no closure")
    # Count all numpy arrays on the heap
    _np_bytes = 0
    _np_count = 0
    for _obj in _gc.get_objects():
        if isinstance(_obj, np.ndarray):
            _np_bytes += _obj.nbytes
            _np_count += 1
    logger.info(f"  All numpy arrays:      {_np_bytes/1e9:.2f} GB  ({_np_count:,} arrays)")
    logger.info(f"  Current RSS:           {_mem_gb():.2f} GB")
    logger.info(f"  Gap (RSS - numpy):     {_mem_gb() - _np_bytes/1e9:.2f} GB")
    del _spec_bytes, _n_scans, _np_bytes, _np_count, _closure_bytes

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
    logger.info("")
    logger.info("Starting Main Search")
    # write dia spectra meta data
    ms2scans_info = [[i.prec_mz,i.RT,i.scan_num,*i.ms1window] for i in spectra_to_fit]
    ms2_info_path = results_folder_path+"/ms2scans.csv"
    write_to_csv(ms2scans_info,ms2_info_path)
    
    ## process in batches
    num_batches = 10
    num_per_batch = int(np.ceil(len(spectra_to_fit)/num_batches))

    # Line profiler setup — profile the first batch only, then dump stats
    from line_profiler import LineProfiler
    from src.spectral_fitting import (huber_nnls_irls, _lasso_nnls, hyperscore2,
                                       get_features, get_scribe, get_residuals,
                                       gof_stat, get_manhattan_distance,
                                       get_closest_ms1, create_entries,
                                       unmatched_peaks, max_matched_residual)
    from src.utils.misc_functions import window_width, hyperscore_b_y, longest_y, cosim
    lp = LineProfiler()
    lp.add_function(fit_to_lib2)
    lp.add_function(huber_nnls_irls)
    lp.add_function(_lasso_nnls)
    lp.add_function(hyperscore2)
    lp.add_function(get_features)
    lp.add_function(get_scribe)
    lp.add_function(get_residuals)
    lp.add_function(gof_stat)
    lp.add_function(get_manhattan_distance)
    lp.add_function(get_closest_ms1)
    lp.add_function(create_entries)
    lp.add_function(unmatched_peaks)
    lp.add_function(max_matched_residual)
    lp.add_function(window_width)
    lp.add_function(hyperscore_b_y)
    lp.add_function(longest_y)
    lp.add_function(cosim)
    fit_to_lib2_profiled = lp(fit_to_lib2)

    for batch_idx in range(num_batches):
        start_time = time.time()
        batch_spectra = spectra_to_fit[batch_idx*num_per_batch:(batch_idx+1)*num_per_batch]

        logger.info(f"Fitting batch {batch_idx+1} of {num_batches}")
        _log_mem(f"batch {batch_idx+1} start")

        outputs= []

        # Use profiled version for batch 5 only
        _fit_fn = fit_to_lib2_profiled if batch_idx == 4 else fit_to_lib2

        for i, dia_spec in enumerate(tqdm.tqdm(batch_spectra)):
            result = _fit_fn(dia_spec,
                                 library=spectrumLibrary,
                                 rt_mz=rt_mz,
                                 all_keys=all_keys,
                                 dino_features=None,
                                 rt_filter=True,
                                 rt_tol=config.opt_rt_tol,
                                 ms1_tol=config.opt_ms1_tol,
                                 mz_tol=config.mz_tol,
                                 ms1_spectra=DIAspectra.ms1scans,
                                 return_frags=False,
                                 decoy=True,
                                 decoy_library=decoy_lib,
                                 output_folder=results_folder_path)
            if result:
                outputs.append(result)

        # Dump line profiler stats after batch 5
        if batch_idx == 4:
            lp_output_path = results_folder_path + "/line_profiler_stats.txt"
            with open(lp_output_path, "w") as lp_file:
                lp.print_stats(stream=lp_file)
            logger.info(f"Line profiler stats written to {lp_output_path}")

        long_outputs = [j for i in outputs for j in i]
        logger.info(f"Fit {len(batch_spectra)} spectra in {(round(time.time()-start_time))//60} mins and {(round(time.time()-start_time))%60} sec")

        decoylib_search_path = results_folder_path+"/decoylibsearch_coeffs.csv"
        write_mode = "w" if batch_idx==0 else "a"
        write_to_csv(long_outputs, decoylib_search_path, write_mode)

    process_data(file=decoylib_search_path,
                 spectra=DIAspectra,
                 library=spectrumLibrary,
                 mass_tag=mass_tag,
                 SILAC=SILAC,
                 timeplex=config.args.timeplex)
    # """