
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

from src.utils.io import load_files, file_reader
from src.utils.set_seeds import set_seeds
from src.models.spec_lib import spec_lib
from src.spectral_fitting import fit_to_lib2, merge_spectrum_peaks
from src.rt_alignment import MZRTfit, MZRTfit_timeplex
from src.utils.misc_functions import write_to_csv
import polars as pl
import pyarrow.parquet as pq
from src.utils.io.read_output import get_parquet_schema
from src import iso_functions as iso_f
from src.mass_tags import tag_library, available_tags
from src.fdr_analysis import process_data
from src.finetune_funs import predict_decoy_rts

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

    # TODO: validate all config.args values against default_dict['values'] lists here
    ####  Load Libraries   ######################
    set_seeds(config.RANDOM_SEED)
    mzml_file = config.args.mzml.replace("\\","/")
    lib_file = config.args.speclib.replace("\\","/")

    
    
    spec_file_name = mzml_file.split("/")[-1].rsplit(".",1)[0]
    lib_file_name = lib_file.split("/")[-1].rsplit(".",1)[0]


    use_rt = "RT" if config.args.use_rt else ""
    iso = f"iso{config.args.num_iso}" if config.args.iso else ""
    lib_frac = f"iso{config.args.lib_frac}"
    mTRAQ = "mTRAQ" if config.args.mTRAQ else ""
    plexDIA = "plexDIA" if config.args.plexDIA else ""
    tag = config.args.tag
    is_timeplex = "timeplex" if config.args.timeplex else ""
    dummy_val = str(config.args.dummy_value) if config.args.dummy_value else ""
    dino_features = None

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
            os.mkdir(os.path.join(results_folder_path, "first_search"))
            os.mkdir(os.path.join(results_folder_path, "first_search/fine_tuning"))
            os.mkdir(os.path.join(results_folder_path, "scoring"))
            os.mkdir(os.path.join(results_folder_path, "outputs"))
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
    json_path = os.path.join(results_folder_path, "outputs/config.json")
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


    overall_start_time = time.time()
    # python run_jmod.py -r -l /Users/nathanwamsley/Data/SPEC_LIBS/JD_LF_Feb2025/LF_HY_lib.tsv -i /Users/nathanwamsley/Data/mzML/mTRAQ_Feb2025/JD0324.mzML --iso --num_iso 5
    logger.info(f"Results will be saved to {os.path.abspath(results_folder_path)}")

    
    ######################################################
    #### Load the data
    spectrumLibrary = spec_lib.loadSpecLib(lib_file)
    DIAspectra=file_reader.loadSpectra(mzml_file)

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
        mz_tolerance = (config.args.ppm * 1e-6) * 2
        
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
    #### Generate decoys (before tagging/isotopes so they apply to both)
    logger.info("Creating Decoy Library")
    spectrumLibrary = spec_lib.create_decoy_lib(spectrumLibrary, rules=config.args.decoy)
    config.target_decoy_ratio = spectrumLibrary.target_decoy_ratio
    logger.info(f"Combined library: {spectrumLibrary.n_targets} targets, "
                f"{spectrumLibrary.n_decoys} decoys "
                f"(ratio={config.target_decoy_ratio:.4f})")
    # TODO: use target_decoy_ratio to correct FDR calculation

    ######################################################
    #### Tagging #####

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

    ######################################################
    #### RT/MZ Alignment (initial search uses target entries only) #####

    target_view = spectrumLibrary.target_view()

    if config.args.timeplex:
        funcs, updated_targets, elution_fwhm = MZRTfit_timeplex(
            DIAspectra, target_view, (config.args.ppm * 1e-6),
            results_folder=results_folder_path,
            ms2=config.args.ms2_align,
            mass_tag=mass_tag, SILAC=SILAC,
        )
        # Timeplex path doesn't compute elution SD yet — use the historical default.
        vote_sigma = 1.0

        # Propagate updated iRT from aligned targets back to combined store
        for key in updated_targets:
            idx = spectrumLibrary.key_to_idx[key]
            spectrumLibrary.iRT[idx] = updated_targets[key]["iRT"]
        # Copy updated iRT to decoy entries from their parent targets
        for i in range(spectrumLibrary.n_targets, len(spectrumLibrary)):
            parent = spectrumLibrary.parent_idx[i]
            if parent >= 0:
                spectrumLibrary.iRT[i] = spectrumLibrary.iRT[parent]
        del updated_targets

        rt_spls,mz_func = funcs[:2]

        # Per-(entry, channel) search RT. Empirical: iRT->RT curves. Per-channel CNN:
        # rt_spls is K dicts {target_seq: RT_k}; targets look up by sequence and decoys
        # inherit their parent target's per-channel RT (same rule as the iRT inheritance
        # above), so decoys stay co-located with their targets for FDR.
        n_lib, K = len(spectrumLibrary), len(rt_spls)
        per_ch = np.empty((n_lib, K))
        if callable(rt_spls[0]):
            for i in range(n_lib):
                per_ch[i] = [rt_spls[k](spectrumLibrary.iRT[i]) for k in range(K)]
        else:
            for i in range(spectrumLibrary.n_targets):
                per_ch[i] = [rt_spls[k][spectrumLibrary.seq[i]] for k in range(K)]
            for i in range(spectrumLibrary.n_targets, n_lib):
                parent = spectrumLibrary.parent_idx[i]
                if parent >= 0:
                    per_ch[i] = per_ch[parent]

        plex_lib = {}
        rt_mz = []
        for idx in range(K):
            for key in spectrumLibrary:
                plex_lib[key+(idx,)] = spectrumLibrary[key]
            rt_mz.append([[per_ch[i, idx], mz_func(e["prec_mz"], e["iRT"])]
                          for i, e in enumerate(spectrumLibrary.values())])
        rt_mz = np.concatenate(rt_mz)
        
        from src.models.spec_lib.library_store import SpectrumLibraryStore
        spectrumLibrary = SpectrumLibraryStore.from_dict(plex_lib)
        del plex_lib

    else:
        funcs, updated_targets, rt_models_data, elution_fwhm, vote_sigma = MZRTfit(
            DIAspectra, target_view, dino_features, (config.args.ppm * 1e-6),
            results_folder=results_folder_path,
            ms2=config.args.ms2_align, mass_tag=mass_tag, SILAC=SILAC,
            return_rt_models=config.args.predict_decoys,
        )

        # Propagate updated iRT from aligned targets back to combined store
        for key in updated_targets:
            idx = spectrumLibrary.key_to_idx[key]
            spectrumLibrary.iRT[idx] = updated_targets[key]["iRT"]
        # Copy updated iRT to decoy entries from their parent targets (default)
        for i in range(spectrumLibrary.n_targets, len(spectrumLibrary)):
            parent = spectrumLibrary.parent_idx[i]
            if parent >= 0:
                spectrumLibrary.iRT[i] = spectrumLibrary.iRT[parent]

        # Predict independent RTs for decoy sequences using CNN
        if config.args.predict_decoys and rt_models_data is not None:
            models, convertor = rt_models_data
            decoy_seqs = [spectrumLibrary.seq[i] for i in range(spectrumLibrary.n_targets, len(spectrumLibrary))]
            predicted_rts = predict_decoy_rts(decoy_seqs, models, convertor)
            if predicted_rts is not None:
                for i, rt in enumerate(predicted_rts):
                    spectrumLibrary.iRT[spectrumLibrary.n_targets + i] = rt
            del models, convertor
        elif config.args.predict_decoys:
            logger.warning("Decoy RT prediction requested but no RT models available (using empirical RT?)")

        del updated_targets

        rt_spl,mz_func = funcs[:2]
        # Build rt_mz for ALL entries (target + decoy)
        rt_mz = np.array([[rt_spl(spectrumLibrary.iRT[i]), mz_func(spectrumLibrary.prec_mz[i], spectrumLibrary.iRT[i])]
                          for i in range(len(spectrumLibrary))])
        # Apply decoy m/z offset to decoy entries
        rt_mz[spectrumLibrary.n_targets:, 1] -= config.decoy_mz_offset

    del target_view

    ## Merge peaks in spectra (MS2 only; MS1 peaks are summed within
    ## tolerance during matrix construction in fit_channel_isotopes_numba)
    for spec in DIAspectra.ms2scans:
        merge_spectrum_peaks(spec, (config.args.ppm * 1e-6))

    spectra_to_fit = DIAspectra.ms2scans

    all_keys = list(spectrumLibrary)


    if config.args.ms2_align:
        ms2_func = funcs[2]

        for key in all_keys:
            spectrumLibrary[key]["spectrum"][:,0] = ms2_func(spectrumLibrary[key]["spectrum"][:,0])
    else:
        ms2_func=None


    if config.args.iso:
        spectrumLibrary = iso_f.iso_library_multi(spectrumLibrary,
                                                  tag=config.tag,
                                                  n_iso=config.args.num_iso)

    spectrumLibrary.bulk_set_top_n(config.top_n)
    logger.info("Finished Library Setup")

    # Build fragment index (single unified index for targets + decoys)
    if not config.args.timeplex:
        from src.fragment_index import FragmentIndex
        logger.info("Building fragment ion index")
        frag_index = FragmentIndex.build(spectrumLibrary, all_keys, rt_mz, config.args.ppm)
        logger.info("Fragment index built")
    else:
        frag_index = None

    ######################################################
    ### Write search params to file
    param_file = results_folder_path + "/outputs/params.txt"
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
    # ms2_info_path = results_folder_path+"/ms2scans.csv"
    # write_to_csv(ms2scans_info,ms2_info_path)
    
    ## process in batches
    num_batches = 10
    num_per_batch = int(np.ceil(len(spectra_to_fit)/num_batches))

    from concurrent.futures import ThreadPoolExecutor, as_completed

    n_threads = 3
    logger.info(f"Using {n_threads} threads for main search")

    # Precompute MS1 RT array once (shared across all threads, read-only)
    _ms1_rt = np.array([s.RT for s in DIAspectra.ms1scans])

    # Precompute IM-bin MS1 lookup if IM data is present
    if len(DIAspectra.ms1scans) > 0 and DIAspectra.ms1scans[0].im_lo is not None:
        from collections import defaultdict
        _im_bin_ms1_tmp = defaultdict(lambda: ([], []))
        for i, s in enumerate(DIAspectra.ms1scans):
            _im_bin_ms1_tmp[(s.im_lo, s.im_hi)][0].append(s.RT)
            _im_bin_ms1_tmp[(s.im_lo, s.im_hi)][1].append(i)
        _im_bin_ms1 = {}
        for key, (rts, idxs) in _im_bin_ms1_tmp.items():
            rt_arr = np.array(rts)
            idx_arr = np.array(idxs, dtype=int)
            order = np.argsort(rt_arr)
            _im_bin_ms1[key] = (rt_arr[order], idx_arr[order])
        logger.info(f"Built IM-bin MS1 lookup with {len(_im_bin_ms1)} bins")
    else:
        _im_bin_ms1 = None

    _pl_schema = get_parquet_schema(timeplex=config.args.timeplex)
    _pa_schema = pl.DataFrame(schema=_pl_schema).to_arrow().schema
    _BUFFER_SIZE = 1000  # results to buffer before flushing to disk

    # Measure CPU utilization across the search to assess GIL contention
    import psutil as _psutil
    _search_proc = _psutil.Process(os.getpid())
    _search_proc.cpu_percent()  # prime the measurement
    _search_wall_t0 = time.time()

    for batch_idx in range(num_batches):
        start_time = time.time()
        batch_spectra = spectra_to_fit[batch_idx*num_per_batch:(batch_idx+1)*num_per_batch]

        logger.info(f"Fitting batch {batch_idx+1} of {num_batches}")

        batch_parquet_path = results_folder_path + f"/decoylibsearch_coeffs_batch{batch_idx}.parquet"
        n_results = 0

        # ``with`` guarantees ``writer.close()`` even when a worker crashes or
        # a network write fails — otherwise the OS handle (and the server-side
        # SMB/NFS oplock on network mounts like Synology) leaks and the file
        # stays "Resource busy" until the next mount cycle.
        with pq.ParquetWriter(batch_parquet_path, _pa_schema) as writer:
            buffer = []
            with ThreadPoolExecutor(max_workers=n_threads) as pool:
                futures = {pool.submit(fit_to_lib2, dia_spec,
                                       library=spectrumLibrary,
                                       rt_mz=rt_mz,
                                       all_keys=all_keys,
                                       dino_features=None,
                                       rt_filter=True,
                                       rt_tol=config.opt_rt_tol,
                                       ms1_tol=config.opt_ms1_tol,
                                       mz_tol=(config.args.ppm * 1e-6),
                                       ms1_spectra=DIAspectra.ms1scans,
                                       return_frags=False,
                                       decoy=True,
                                       output_folder=results_folder_path,
                                       frag_index=frag_index,
                                       ms1_rt=_ms1_rt,
                                       im_bin_ms1=_im_bin_ms1): i
                          for i, dia_spec in enumerate(batch_spectra)}

                for f in tqdm.tqdm(as_completed(futures), total=len(futures)):
                    result = f.result()
                    if result:
                        buffer.extend(result)
                        n_results += len(result)
                        if len(buffer) >= _BUFFER_SIZE:
                            _col_data = {col: [row[i] for row in buffer]
                                         for i, col in enumerate(_pl_schema)}
                            writer.write_table(pl.DataFrame(_col_data, schema=_pl_schema).to_arrow())
                            buffer.clear()

            # Flush remaining buffered results
            if buffer:
                _col_data = {col: [row[i] for row in buffer]
                             for i, col in enumerate(_pl_schema)}
                writer.write_table(pl.DataFrame(_col_data, schema=_pl_schema).to_arrow())
                buffer.clear()
        logger.info(f"Fit {len(batch_spectra)} spectra in {(round(time.time()-start_time))//60} mins and {(round(time.time()-start_time))%60} sec")
        logger.info(f"Batch {batch_idx+1}: {n_results} results written")

    # Report CPU utilization for GIL contention assessment
    _search_wall = time.time() - _search_wall_t0
    _search_cpu = _search_proc.cpu_percent()
    logger.info(f"[CPU] Search wall time: {_search_wall:.1f}s, "
                f"CPU: {_search_cpu:.0f}%, "
                f"Effective cores: {_search_cpu/100:.1f}/{n_threads}")

    # Free large objects no longer needed after search (keep spectrumLibrary
    # alive for fragment correlation features computed inside process_data).
    del frag_index, _ms1_rt, spectra_to_fit
    del rt_mz, all_keys, funcs, dino_features
    import gc as _gc2
    _gc2.collect()

    # Merge batch parquets into single file (streaming, one batch at a time)
    import glob as _glob
    batch_files = sorted(_glob.glob(results_folder_path + "/decoylibsearch_coeffs_batch*.parquet"))
    decoylib_search_path = results_folder_path + "/outputs/decoylibsearch_coeffs.parquet"
    merge_writer = None
    try:
        for bf in batch_files:
            table = pq.read_table(bf)
            if merge_writer is None:
                merge_writer = pq.ParquetWriter(decoylib_search_path, table.schema)
            merge_writer.write_table(table)
            del table
    finally:
        # Guarantee close even on read/write failure mid-merge — same Synology
        # oplock leak story as the batch writer above.
        if merge_writer is not None:
            merge_writer.close()
    for bf in batch_files:
        os.remove(bf)
    process_data(file=decoylib_search_path,
                 spectra=DIAspectra,
                 library=spectrumLibrary,
                 mass_tag=mass_tag,
                 SILAC=SILAC,
                 timeplex=config.args.timeplex,
                 elution_fwhm=elution_fwhm,
                 vote_sigma=vote_sigma)
    del spectrumLibrary
    _gc2.collect()
    # """