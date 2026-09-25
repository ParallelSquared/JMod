
#  Copyright (c) 2026 Parallel Squared Technology Institute
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

# Update imports to relative imports
import src.config as config
import numpy as np
import os
import time
import tqdm
import pandas as pd
import json
import gc

from src.utils.io import file_reader
from src.utils.set_seeds import set_seeds
from src.models.spec_lib import spec_lib
from src.spectral_fitting import fit_to_lib2
from src.rt_alignment import MZRTfit, MZRTfit_timeplex, aligned_library_im
from src.utils.misc_functions import datestamped
import polars as pl
import pyarrow.parquet as pq
from src.utils.io.read_output import get_parquet_schema
from src import iso_functions as iso_f
from src.mass_tags import tag_library, available_tags
from src.fdr_analysis import process_data
from src.finetune_funs import predict_decoy_rts
from src.utils.gui_utils import load_settings, save_settings
from src.models.run_state import RunState
from src.utils.errors import JModError, report_error, mark_run_failed
from src.multi_run import combine_runs

from src.logger import logger, set_log_filepath


def main(GUI_config_json=None):
    """Run JMod on every mass spec file in the configuration."""
    try:
        run_experiment(GUI_config_json)
    except Exception as e:
        report_error(e, "JMod stopped")


def run_experiment(GUI_config_json=None):
    """Set up the experiment, build the library once, and run every mass spec file."""
    config.setup(GUI_config_json)
    experiment_dir = _start_experiment_log()
    _write_experiment_config(experiment_dir)
    run_files = config.args.mzml

    bruker_sdk_path = _prepare_readers(run_files)
    set_seeds(config.RANDOM_SEED)
    mass_tag, SILAC = resolve_tags()

    #### Build the library once.
    spectrumLibrary = build_library(config.args.speclib, experiment_dir, mass_tag, SILAC)

    failed_runs = []
    completed_run_folders = {}
    for run_idx, run_file in enumerate(run_files, start=1):
        logger.info("")
        logger.info(f"Run {run_idx} of {len(run_files)}", extra={"highlight": True})
        logger.info(run_file, extra={"highlight": True})
        runState = RunState()
        runState.file_name = run_file
        try:
            process_run(runState, spectrumLibrary, mass_tag, SILAC, bruker_sdk_path)
        except Exception as e:
            report_error(e, f"Run {run_idx} of {len(run_files)} failed ({run_file})")
            mark_run_failed(getattr(runState, "results_folder", None))
            failed_runs.append(run_file)
        else:
            logger.info(f"Run {run_idx} of {len(run_files)} finished")
            completed_run_folders[run_idx] = runState.results_folder
        # A failed run's spectra are only released once its traceback is gone
        gc.collect()

    logger.info("")
    logger.info(f"{len(run_files) - len(failed_runs)} of {len(run_files)} files completed successfully. "
                f"Output at {os.path.abspath(experiment_dir)}", extra={"highlight": True})
    if failed_runs:
        logger.error(f"{len(failed_runs)} run(s) failed, see above: {', '.join(failed_runs)}")

    # The global q-value counts targets and decoys like the per-run one, over
    # the whole library
    n_decoys = spectrumLibrary.n_decoys
    target_decoy_ratio = spectrumLibrary.n_targets / n_decoys if n_decoys else float('inf')
    del spectrumLibrary
    gc.collect()

    if len(completed_run_folders) >= 1:
        combine_runs(completed_run_folders, experiment_dir, target_decoy_ratio, config.fdr_threshold)


def _prepare_readers(run_files):
    """Load the vendor libraries the data files need, once for the experiment.

    Thermo's RawFileReader when any file is a .raw.  For any .d, the Bruker SDK
    path: a CLI arg wins and is persisted, otherwise the stored setting is used.
    It is only needed for a .d with no peaks.parquet yet, where it supplies the
    calibration for centroiding.  Returns the Bruker SDK path, or None.
    """
    if any(f.lower().endswith(".raw") for f in run_files):
        settings = load_settings()

        if config.args.rawfilereader_path is not None:
            reader_path = config.args.rawfilereader_path
            settings["rawfilereader_path"] = reader_path
            save_settings(settings)
        else:
            reader_path = settings["rawfilereader_path"]

        file_reader.load_rawfilereader(reader_path)

    if any(f.rstrip("/").lower().endswith(".d") for f in run_files):
        return file_reader.resolve_bruker_setting(config.args.bruker_sdk_path)
    return None


def _start_experiment_log():
    """Create the experiment folder, open the log there, and record the configuration.

    Experiment-level files -- the log, and a memory-mapped library -- go in the
    output folder, or next to the first data file when there is none.  Returns
    that folder.
    """
    if config.args.output_folder is not None:
        experiment_dir = config.args.output_folder
    else:
        experiment_dir = os.path.dirname(config.args.mzml[0].replace("\\","/")) or "."
    os.makedirs(experiment_dir, exist_ok=True)

    logfile_path = datestamped(os.path.join(experiment_dir, "JMod_log.log"))
    set_log_filepath(logfile_path)

    if config.ran_from_GUI:
        logger.info(f"Loaded configuration from GUI")
    elif config.args.config_json:
        logger.info(f"Loaded configuration from {config.args.config_json}")
        overrides = {k: v for k, v in config.cli_args.items() if k != "config_json"}
        if overrides:
            logger.info(f"Command-line options overriding the config JSON: {overrides}")

    # Log the configuration that will be used
    logger.info("Using configuration:")
    logger.info(config.args)
    logger.info("")
    logger.info(f"{len(config.args.mzml)} file(s) to run")
    logger.info(f"Log writing to {os.path.abspath(logfile_path)}")
    return experiment_dir


def _write_experiment_config(experiment_dir):
    """Record the experiment's configuration, with every data file, next to its log.
    Datestamped if the name is taken, so an earlier experiment's record is kept.
    """
    json_path = datestamped(os.path.join(experiment_dir, "JMod_config.json"))
    with open(json_path, "w") as f:
        json.dump(vars(config.args), f, indent=4)
    logger.info(f"Configuration written to {os.path.abspath(json_path)}")


def process_run(runState, spectrumLibrary, mass_tag, SILAC, bruker_sdk_path):
    """Search, score and quantify one data file against the shared library.

    Everything made here belongs to this run and is released when it returns:
    the spectra, the run's calibration (RunState, rt_mz, the window mask) and,
    on timeplex, the per-channel copy of the library.  The shared library is
    only read.  *runState* arrives holding the run's file_name; the results
    folder and the fitted values are added to it here.
    """
    # Every run starts from the same seed, so its results do not depend on its
    # position in the list
    set_seeds(config.RANDOM_SEED)
    mzml_file = runState.file_name.replace("\\","/")
    if not os.path.exists(mzml_file):
        raise JModError(f"Data file not found: {runState.file_name}")

    runState.results_folder = _create_results_folder(mzml_file)
    _write_run_config(runState)
    logger.info(f"Results will be saved to {os.path.abspath(runState.results_folder)}")
    DIAspectra = _load_spectra(mzml_file, bruker_sdk_path)

    funcs, target_iRT, rt_models_data, im_spl, elution_fwhm, vote_sigma = first_search(
        DIAspectra, spectrumLibrary, mass_tag, SILAC, runState.results_folder, runState)

    # On timeplex this is a per-channel copy of the library; otherwise it is the
    # shared library itself - unchanged (calibration is in rt_mz, there is also in_window mask).
    # The shared library stays untouched for the next run.
    searchLibrary, rt_mz, in_window = calibrate_library(
        spectrumLibrary, funcs, target_iRT, rt_models_data, im_spl,
        DIAspectra.ms2scans, runState)
    del funcs, target_iRT, rt_models_data, im_spl

    decoylib_search_path = main_search(DIAspectra, searchLibrary, rt_mz, in_window,
                                       runState.results_folder, runState)
    del rt_mz, in_window
    gc.collect()

    score_and_report(decoylib_search_path, DIAspectra, searchLibrary,
                     mass_tag, SILAC, elution_fwhm, vote_sigma, runState)
    del searchLibrary, DIAspectra
    gc.collect()


def _load_features(mzml_file):
    """Dinosaur/biosaur2 MS1 features for the timeplex first search.

    Reads <file>.features.tsv next to the data file, running biosaur2 to make it
    when it is missing.  Returns None when --use_features is off.
    """
    if not config.args.use_features:
        logger.info("Not using features")
        return None
    spec_file_name = mzml_file.split("/")[-1].rsplit(".",1)[0]
    feature_path = os.path.dirname(mzml_file)+"/"+spec_file_name+".features.tsv" #TODO this breaks if you run from cd
    if not os.path.exists(feature_path):
        logger.info("Dinosaur feature file not found, running biosaur2")
        import subprocess
        subprocess.run(["biosaur2", mzml_file], check=True)
    logger.info("Loading Dinosaur features")
    return pd.read_csv(feature_path,delimiter="\t")


def _create_results_folder(mzml_file):
    """Create the run's results folder and its subfolders; return its path.

    Named after the data file (plus --dummy_value), in the output folder or
    next to the data file, with a datestamp added if the name is taken.
    """
    spec_file_name = mzml_file.split("/")[-1].rsplit(".",1)[0]
    dummy_val = str(config.args.dummy_value) if config.args.dummy_value else ""
    results_folder_name = spec_file_name + "_results" + "_" + dummy_val
    results_folder_name = results_folder_name.rstrip("_")

    if config.args.output_folder is not None:
        os.makedirs(config.args.output_folder, exist_ok=True)
        results_folder_path = os.path.join(config.args.output_folder, results_folder_name)
    else:
        results_folder_path = os.path.join(os.path.dirname(mzml_file), results_folder_name)

    results_folder_path = datestamped(results_folder_path)


    if not os.path.exists(results_folder_path):
        try:
            os.mkdir(results_folder_path)
            os.mkdir(os.path.join(results_folder_path, "first_search"))
            os.mkdir(os.path.join(results_folder_path, "first_search/fine_tuning"))
            os.mkdir(os.path.join(results_folder_path, "scoring"))
            os.mkdir(os.path.join(results_folder_path, "outputs"))
        except FileNotFoundError as e:
            if not os.path.exists(os.path.dirname(results_folder_path)):
                raise JModError(f"Error Creating Results Folder. Parent path does not exist.\nPath: {os.path.dirname(results_folder_path)}") from e
            if "[WinError 3]" in str(e) or "[WinError 206]" in str(e):
                raise JModError("Path Length Error. To enable long paths, use win+R and type regedit. Navigate to HKEY_LOCAL_MACHINE\ SYSTEM\CurrentControlSet\Control\FileSystem. Set LongPathsEnabled to 1 and restart computer.") from e
            raise JModError(f"Error Creating Results Folder. Please check the path is valid.\nPath: {results_folder_path}\nError: \n{str(e)}") from e
        except Exception as e:
            raise JModError(f"Error Creating Results Folder. Please check the path is valid.\nPath: {results_folder_path}\nError: \n{str(e)}") from e


    if len(results_folder_path) >= 225:  ##if results path is long, check to make sure putting things in it wont break (i.e. windows with long paths enabled or different OS)
        try:
            test_path = os.path.join(results_folder_path, "a" * 250 + ".txt")
            with open(test_path, "w") as f:
                f.write("test")
            os.remove(test_path)
        except FileNotFoundError as e:
            if "[WinError 3]" in str(e) or "[WinError 206]" in str(e):
                raise JModError("Path Length Error. To enable long paths, use win+R and type regedit. Navigate to HKEY_LOCAL_MACHINE\ SYSTEM\CurrentControlSet\Control\FileSystem. Set LongPathsEnabled to 1 and restart computer.") from e
            raise JModError(f"Error Creating Results Folder. Please check the path is valid.\nPath: {results_folder_path}\nError:\n{str(e)}") from e
        except Exception as e:
            raise JModError(f"Error Creating Results Folder. Please check the path is valid.\nPath: {results_folder_path}\nError:\n{str(e)}") from e
    return results_folder_path


def _write_run_config(runState):
    # This run's record of the configuration, naming only its own data file
    args_dict = dict(vars(config.args), mzml=runState.file_name)
    json_path = os.path.join(runState.results_folder, "outputs/config.json")
    with open(json_path, "w") as f:
        json.dump(args_dict, f, indent=4)


def _load_spectra(mzml_file, bruker_sdk_path):
    """Load the run's spectra; with --test_mode, keep only the test RT/m/z range."""
    DIAspectra=file_reader.loadSpectra(mzml_file, bruker_sdk_path=bruker_sdk_path)

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
        del filtered_ms2_scans
    return DIAspectra


def resolve_tags():
    """Look up the mass tag and SILAC label named in the configuration.

    Returns (mass_tag, SILAC); each is None when not in use.  Also sets
    config.tag / config.SILAC, which other modules read.  Raises when a name is
    given that is not an available tag.
    """
    ## TODO Running SILAC + a Tag without an untagged library is probably not currently functional
    if config.args.SILAC:
        # Find the tag object based on the tag name
        if config.args.SILAC in available_tags:
            config.SILAC = available_tags[config.args.SILAC]
            logger.info(f"Using SILAC: {config.SILAC.name} - {config.SILAC.n_channels} channels")
            SILAC = config.SILAC
        else:
            if config.args.SILAC != "None":
                raise JModError(f"SILAC '{config.args.SILAC}' not found in available tags: "
                                f"{list(available_tags.keys())}")
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
            mass_tag = config.tag
        else:
            if config.args.tag != "None":
                raise JModError(f"Tag '{config.args.tag}' not found in available tags: "
                                f"{list(available_tags.keys())}")
            mass_tag = None
            config.tag = None
    else:
        mass_tag = None
        config.tag = None

    return mass_tag, SILAC


def build_library(lib_file, work_dir, mass_tag, SILAC):
    """Load the spectral library and build the target + decoy search library.

    Adds decoys, tags every entry with *mass_tag* and *SILAC* (from
    resolve_tags), expands isotopes (--iso) or finalizes the spectra, sets
    top_n, and freezes the result.  Nothing here depends on a run, and nothing
    after it writes to the library: per-run changes live in
    calibrate_library's return values.  Large non-iso libraries are
    memory-mapped under *work_dir*.
    """
    spectrumLibrary, library_tag_bool, source_channel_mass, library_tag_name = spec_lib.loadSpecLib(lib_file)

    if config.args.test_mode:
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

    # A pre-tagged library: its tag becomes the closest channel of mass_tag
    if mass_tag and library_tag_bool:
        diffs = np.abs(mass_tag.channel_masses - source_channel_mass)
        closest_idx = int(np.argmin(diffs))
        closest_channel_name = mass_tag.channel_names[closest_idx]
        closest_channel_mass = mass_tag.channel_masses[closest_idx]
        mass_diff = closest_channel_mass - source_channel_mass
        source_channel = mass_tag.name + "-" + str(closest_channel_name)
        logger.info(f"Tag found in library: {source_channel}. (mass difference: {mass_diff:.6f} Da)")
        spectrumLibrary.relabel_tag(library_tag_name, source_channel)
    else:
        source_channel = None

    ######################################################
    #### Generate decoys (before tagging/isotopes so they apply to both)
    logger.info("Creating Decoy Library")
    # if "diann_tagged" in lib_file_name:
    if library_tag_bool:
        lib_gen_tag = mass_tag
    else:
        lib_gen_tag = None
    spectrumLibrary = spec_lib.create_decoy_lib(spectrumLibrary, rules=config.args.decoy, tag=lib_gen_tag)
    logger.info(f"Combined library: {spectrumLibrary.n_targets} targets, "
                f"{spectrumLibrary.n_decoys} decoys")

    ######################################################
    #### Tagging #####

    if mass_tag:
        spectrumLibrary = tag_library(spectrumLibrary, mass_tag, source_channel=source_channel)
    if SILAC:
        spectrumLibrary = tag_library(spectrumLibrary, SILAC)

    ######################################################
    #### Isotopes (after tagging: the tag changes each fragment's composition)
    if config.args.iso:
        # iso_library_multi rebuilds spectra from frags and discards the
        # sort permutation -- it is the finalizer on iso runs.  The frags stay
        # monoisotopic, and first_search reads them back through
        # monoisotopic_targets()
        spectrumLibrary = iso_f.iso_library_multi(spectrumLibrary,
                                                  tag=mass_tag,
                                                  n_iso=config.args.num_iso)
    else:
        # Large libraries: back the six big fragment/spectrum arrays with
        # read-only memory maps so cold pages evict to SSD instead of
        # churning the memory compressor
        spectrumLibrary.finalize_spectra(
            memmap_dir=os.path.join(work_dir, "library_mmap"))

    spectrumLibrary.bulk_set_top_n(config.top_n)
    spectrumLibrary.freeze()
    logger.info("Finished Library Setup")

    return spectrumLibrary


def first_search(DIAspectra, spectrumLibrary, mass_tag, SILAC, results_folder_path, runState):
    """Fit this run's calibration.

    Searches the monoisotopic target spectra to fit the RT, precursor m/z and
    IM alignments and each target's aligned iRT, then re-bands the MS2 spectra
    on the fitted IM precision.  Reads the library but does not change it;
    calibrate_library applies the result.  Sets the fitted tolerances on *runState*
    (opt_rt_tol, opt_ms1_tol, opt_im_precision, opt_im_accuracy).

    Returns (funcs, target_iRT, rt_models_data, im_spl, elution_fwhm, vote_sigma).
    ``funcs`` starts with the RT spline (one per time channel on timeplex) and
    the m/z function; ``target_iRT`` holds the aligned iRT of every target,
    indexed like the library; ``rt_models_data`` is None unless decoy RTs are
    to be predicted; ``im_spl`` maps library IM to observed 1/K0, or is None
    when no IM alignment was fitted.
    """
    ######################################################
    #### RT/MZ Alignment (initial search uses monoisotopic target entries only) #####

    target_view = spectrumLibrary.monoisotopic_targets()

    if config.args.timeplex:
        # Only the timeplex first search uses MS1 features: it starts from them
        dino_features = _load_features(runState.file_name.replace("\\","/"))
        funcs, updated_targets, elution_fwhm = MZRTfit_timeplex(DIAspectra, target_view, dino_features, (config.args.ppm * 1e-6), results_folder=results_folder_path,
                                        ms2=config.args.ms2_align, runState=runState)
        del dino_features
        # Timeplex path doesn't compute elution SD yet — use the historical default.
        vote_sigma = 1.0
        rt_models_data = None
        im_spl = None

    else:
        # MZRTfit takes MS1 features but no longer uses them
        funcs, updated_targets, rt_models_data, elution_fwhm, vote_sigma, im_spl = MZRTfit(
            DIAspectra, target_view, None, (config.args.ppm * 1e-6),
            results_folder=results_folder_path,
            ms2=config.args.ms2_align, mass_tag=mass_tag, SILAC=SILAC,
            return_rt_models=config.args.predict_decoys, runState=runState,
        )

    del target_view

    # The aligned iRT of every target; the first search ran on the targets in
    # library order, so its iRT column lines up with the library's targets
    target_iRT = updated_targets.iRT
    del updated_targets

    ## Re-band MS2 on the fitted IM precision.  The bands built at load time use
    ## a hardcoded width, chosen before anything about this run's mobility
    ## resolution was known; the preliminary search has now fitted the precision,
    ## so redraw them to match.  Width is 4 x precision -- the band spans
    ## +/-2*precision, deliberately wider than the match gate so a precursor's
    ## mobility profile stays inside one spectrum rather than being split across
    ## bands; the tight gating happens later, within the band.  No-op on non-IM
    ## data (reband_ms2 self-guards on the retained un-banded peaks, which only
    ## the .d path stores).
    ## Nothing may hold a reference to the old band list across this call:
    ## reband_ms2 clears DIAspectra.ms2scans, and an alias would keep every old
    ## band spectrum alive while the new, larger set is allocated -- on a large .d
    ## that is an extra ~18 GB held for no reason.
    if DIAspectra.has_ion_mobility:
        file_reader.reband_ms2(DIAspectra, 4.0 * runState.opt_im_precision)

    return funcs, target_iRT, rt_models_data, im_spl, elution_fwhm, vote_sigma


def calibrate_library(spectrumLibrary, funcs, target_iRT, rt_models_data, im_spl, ms2scans, runState):
    """Map the frozen library into this run's coordinates.

    Every per-run change to what the search sees of the library happens here,
    and none of it is written back into the library:

    - iRT: the aligned iRT of each target, copied to its decoys, or predicted
      for them with --predict_decoys.  Only used to build rt_mz.
    - rt_mz: each entry's RT, precursor m/z and 1/K0 (via *im_spl*) in this
      run's observed coordinates.
    - in_window: the entries some isolation window of this run can select;
      main_search leaves the rest out of the fragment index.
    - runState.target_decoy_ratio: targets over decoys among those entries.

    Timeplex searches the library once per time channel, so on that path the
    returned library is a new per-channel store, not the one passed in.

    Returns (spectrumLibrary, rt_mz, in_window).
    """
    n_entries = len(spectrumLibrary)
    n_targets = spectrumLibrary.n_targets

    # Per-run iRT: the aligned targets, and each decoy its parent's value
    iRT = spectrumLibrary.iRT.copy()
    iRT[:n_targets] = target_iRT
    parents = spectrumLibrary.parent_idx[n_targets:]
    has_parent = parents >= 0
    iRT[n_targets:][has_parent] = iRT[parents[has_parent]]

    if config.args.timeplex:
        rt_spls,mz_func = funcs[:2]

        plex_lib = {}
        rt_mz = []
        for idx in range(len(rt_spls)):
            for key in spectrumLibrary:
                plex_lib[key+(idx,)] = spectrumLibrary[key]
            rt_mz.append([[rt_spls[idx](iRT[i]), mz_func(spectrumLibrary.prec_mz[i], iRT[i])]
                          for i in range(n_entries)])
        rt_mz = np.concatenate(rt_mz)
        # Column 2 as in the standard path.  This path replicates the library once
        # per plex, so the aligned IM has to be tiled to match row-for-row.
        _plex_im = aligned_library_im(spectrumLibrary, im_spl)
        rt_mz = np.column_stack([rt_mz, np.tile(_plex_im, len(rt_spls))])

        from src.models.spec_lib.library_store import SpectrumLibraryStore
        search_library = SpectrumLibraryStore.from_dict(plex_lib)
        del plex_lib

    else:
        # Predict independent RTs for decoy sequences using CNN
        if config.args.predict_decoys and rt_models_data is not None:
            models, convertor = rt_models_data
            decoy_seqs = [spectrumLibrary.seq[i] for i in range(n_targets, n_entries)]
            predicted_rts = predict_decoy_rts(decoy_seqs, models, convertor)
            if predicted_rts is not None:
                iRT[n_targets:n_targets + len(predicted_rts)] = predicted_rts
            del models, convertor
        elif config.args.predict_decoys:
            logger.warning("Decoy RT prediction requested but no RT models available (using empirical RT?)")

        rt_spl,mz_func = funcs[:2]
        # Build rt_mz for ALL entries (target + decoy)
        rt_mz = np.array([[rt_spl(iRT[i]), mz_func(spectrumLibrary.prec_mz[i], iRT[i])]
                          for i in range(n_entries)])
        # Column 2: the library's IM mapped onto observed 1/K0 by the alignment.
        # Left all-NaN when the library carries no IM or the alignment did not
        # fit, which leaves every downstream IM gate inert.  Decoys keep their
        # parent's IM unchanged -- unlike m/z there is no decoy offset, since
        # shifting it would reject decoys systematically and break FDR.
        rt_mz = np.column_stack([rt_mz, aligned_library_im(spectrumLibrary, im_spl)])
        # Apply decoy m/z offset to decoy entries
        rt_mz[n_targets:, 1] -= config.decoy_mz_offset
        search_library = spectrumLibrary

    del iRT

    # # TODO: --ms2_align is broken and needs fixing.  MZRTfit stopped returning
    # # the MS2 m/z function in 1d252a01 (its return is commented out), so
    # # funcs[2] raises IndexError.  Restoring it is not enough: get_spectrum
    # # returns a fresh np.stack, so the assignment below writes into a temporary
    # # copy and changes nothing, and on iso runs the expanded spectra come from
    # # frags, which it never touches.
    if config.args.ms2_align:
        logger.warning("MS2 Align in not currently supported. --ms2_align ignored")
        # ms2_func = funcs[2]
        # for key in list(search_library):
        #     search_library[key]["spectrum"][:,0] = ms2_func(search_library[key]["spectrum"][:,0])

    # Precursors no isolation window of this run can select are never
    # candidates.  Tested on the calibrated m/z, so each tag channel and decoy
    # is judged on its own m/z.
    in_window = spec_lib.in_windows(rt_mz[:, 1], ms2scans)
    _n_out = int((~in_window).sum())
    if _n_out:
        logger.info(f"Window filter: {_n_out:,} of {len(in_window):,} library "
                    f"precursors are outside every isolation window of this run")

    # FDR's target/decoy ratio, over the entries this run can select.  Timeplex
    # rows repeat the library once per channel at the same m/z, so the first
    # n_entries rows stand for every channel.
    _in = in_window[:n_entries]
    n_t = int(_in[:n_targets].sum())
    n_d = int(_in[n_targets:].sum())
    runState.target_decoy_ratio = n_t / n_d if n_d else float('inf')
    logger.info(f"Searchable: {n_t} targets, {n_d} decoys "
                f"(ratio={runState.target_decoy_ratio:.4f})")

    return search_library, rt_mz, in_window


def main_search(DIAspectra, spectrumLibrary, rt_mz, in_window, results_folder_path, runState):
    """Fit every MS2 spectrum against the calibrated library.

    Builds the fragment index, writes the search parameters, fits the spectra
    in batches, and merges the batch parquets into
    ``outputs/decoylibsearch_coeffs.parquet``.  Returns that path.
    """
    spectra_to_fit = DIAspectra.ms2scans
    all_keys = list(spectrumLibrary)

    # Build fragment index (single unified index for targets + decoys)
    if not config.args.timeplex:
        from src.fragment_index import FragmentIndex
        logger.info("Building fragment ion index")
        # Entries outside this run's isolation windows are left out
        frag_index = FragmentIndex.build(spectrumLibrary, all_keys, rt_mz, config.args.ppm,
                                         include=in_window)
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

        write_file.writelines("\nRun\n")
        for key,item in runState.as_dict().items():
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

    from concurrent.futures import ThreadPoolExecutor

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

    # Spectra submitted per chunk.  Bounds how many results are held at once.
    _CHUNK = 100

    # The IM candidate gate needs an aligned library IM; without one the IM
    # column of rt_mz is all NaN and the gate stays off
    _im_accuracy = runState.opt_im_accuracy if np.isfinite(rt_mz[:, 2]).any() else None

    # Constant across every spectrum and every batch, so build the kwargs once.
    _fit_kwargs = dict(library=spectrumLibrary,
                       rt_mz=rt_mz,
                       all_keys=all_keys,
                       dino_features=None,
                       rt_filter=True,
                       rt_tol=runState.opt_rt_tol,
                       ms1_tol=runState.opt_ms1_tol,
                       im_tol=runState.opt_im_precision,
                       im_accuracy=_im_accuracy,
                       file_name=runState.file_name,
                       mz_tol=(config.args.ppm * 1e-6),
                       ms1_spectra=DIAspectra.ms1scans,
                       return_frags=False,
                       decoy=True,
                       output_folder=results_folder_path,
                       frag_index=frag_index,
                       ms1_rt=_ms1_rt,
                       im_bin_ms1=_im_bin_ms1)

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
            # Submit in chunks and drain each before submitting the next.
            # Holding every future for the whole batch (the previous dict, keyed
            # by an index nothing read) kept each completed future -- and so its
            # result rows -- alive until the batch ended, so flushing ``buffer``
            # freed nothing and peak memory grew with the task count.  ~190k
            # tasks was enough to exhaust the machine.
            with ThreadPoolExecutor(max_workers=n_threads) as pool, \
                    tqdm.tqdm(total=len(batch_spectra)) as bar:
                for _start in range(0, len(batch_spectra), _CHUNK):
                    futures = [pool.submit(fit_to_lib2, dia_spec, **_fit_kwargs)
                               for dia_spec in batch_spectra[_start:_start + _CHUNK]]
                    # Drain in submission order, not completion order, so rows are
                    # written in spectrum order every run.  Scoring depends on row
                    # order (apex_pc1 ties, CV folds), so completion order made
                    # repeat runs of the same file give different IDs.
                    for f in futures:
                        result = f.result()
                        bar.update(1)
                        if result:
                            buffer.extend(result)
                            n_results += len(result)
                            if len(buffer) >= _BUFFER_SIZE:
                                _col_data = {col: [row[i] for row in buffer]
                                             for i, col in enumerate(_pl_schema)}
                                writer.write_table(pl.DataFrame(_col_data, schema=_pl_schema).to_arrow())
                                buffer.clear()
                    futures.clear()

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
    # _fit_kwargs holds the frag index too, so it has to go for the index to.
    del _fit_kwargs, frag_index, _ms1_rt, _im_bin_ms1, spectra_to_fit, all_keys
    gc.collect()

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

    return decoylib_search_path


def score_and_report(decoylib_search_path, DIAspectra, spectrumLibrary,
                     mass_tag, SILAC, elution_fwhm, vote_sigma, runState):
    """Select apex scans, score and FDR-filter, quantify, and write the reports."""
    logger.info("Selecting apex scans and scoring")
    process_data(file=decoylib_search_path,
                 spectra=DIAspectra,
                 library=spectrumLibrary,
                 mass_tag=mass_tag,
                 SILAC=SILAC,
                 timeplex=config.args.timeplex,
                 elution_fwhm=elution_fwhm,
                 vote_sigma=vote_sigma,
                 ms1_tol=runState.opt_ms1_tol,
                 rt_tol=runState.opt_rt_tol,
                 im_tol=runState.opt_im_precision,
                 target_decoy_ratio=runState.target_decoy_ratio)
