# JMod

**JMod is an open and flexible software for increasing the throughput of sensitive proteomics by supporting multiplexing in the mass and time domains.**

###
## Reference

**JMod: Joint modeling of mass spectra for empowering multiplexed DIA proteomics**

Kevin McDonnell, Daniel J. Geiszler, Nathan Wamsley, Jason Derks, Sarah Sipe, Zachary A. Cohen, Lina Kozhaya Warinner, Maddy Yeh, Eunice Koo,  Andrew Leduc, Theodore J. Zwang, Harrison Specht,  Nikolai Slavov
*bioRxiv* 2025.05.22.655512; doi: [10.1101/2025.05.22.655512](https://doi.org/10.1101/2025.05.22.655512)

###
## Table of Contents
- [Setting up JMod](#setting-up-jmod)
  - [Windows Setup](#windows-setup)
  - [Linux/MacOS Setup](#linuxmacos-setup)
  - [Thermo Raw File Support](#thermo-raw-file-support)
- [Running a Search](#running-a-search)
  - [File Conversion](#file-conversion)
  - [Library Structure](#library-structure)
  - [Graphical User Interface (GUI)](#running-jmod-with-the-graphical-user-interface-gui)
  - [Command Line Interface (CLI)](#running-jmod-with-the-command-line-interface-cli)
- [Output Files](#output-files)

###
## Setting up JMod

### Windows Setup

<details>
<summary><strong> Setup Steps </strong>
</summary>

JMod has a `.bat` executable that is only compatible with Windows computers. Follow the instructions below to launch the JMod GUI via the `.bat`.

1. Download the JMod repository. The most recent release of JMod can be downloaded [here.](https://github.com/ParallelSquared/JMod/releases/tag/v2.0.0)

2. Navigate to the JMod directory. Inside that directory is a `launch.bat` file. Double-click on the file to open the JMod GUI.
    - If this is the first time the computer is setting up a UV environment, it might take a few minutes to download all dependencies and packages.
3. The JMod GUI should open in a new window.

If you would like to set up the UV environment with the command line, please follow the instructions below in [Linux/MacOS Setup](#linuxmacos-setup).

</details>


### Linux/MacOS Setup

<details>
<summary><strong> Setup Steps </strong>
</summary>

1. Download the JMod repository. The most recent release of JMod can be downloaded [here.](https://github.com/ParallelSquared/JMod/releases/tag/v1.0.0)


2. It is recommended to use the UV package manager when running JMod. To set up a UV environment for JMod, run the following command:

    ```pip install uv``` 
 
    or [via wget/curl](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer) if it is not already installed. 

3. Open a new terminal and navigate to the JMod directory. The directory contains a `pyproject.toml` that lists all required packages and dependencies. Run the following command to install the environment: 

    ```uv sync --python 3.11```

4. You can now launch the JMod GUI with ```uv run python run_jmod_from_GUI.py``` or run a search using the command line with ```uv run python run_jmod.py <args>```.

</details>

<!-- TODO: review this -->

### Thermo Raw File Support

<details>
<summary><strong> Setup Steps </strong>
</summary>

JMod supports direct processing of Thermo `.raw` files using Thermo's RawFileReader libraries. Thermo's RawFileReader requires .NET Core Framework 4.7.2, 4.4, 4.8.1, or 8.x and newer on Windows or Mono 6.12 or newer on Linux. It does not work with .NET Core 2.x or 3.x on Windows.

To enable `.raw` file support, please download the latest RawFileReader release, which can be found [here](https://pnnl-comp-mass-spec.github.io/Thermo-Raw-File-Reader/).

When using the JMod GUI, you will be prompted to point to the `netstandard2.0` directory. If running JMod with the command line, use `--rawfilereader_path path/netstandard2.0`. Both of these options will save this path to `data/settings.json` which will be used in future runs unless `--rawfilereader_path` is specified.

If running JMod on Linux/MacOS, `mono` will need to be downloaded. This can be done with the following command:

`brew install mono`

If homebrew is not installed, it can be installed with the following command:

`/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`


Thermo RawFileReader is developed and distributed by Pacific Northwest National Laboratory (PNNL) and Thermo Fisher Scientific and is licensed and distributed separately from JMod.

</details>

###
## Running a Search

To run a JMod search, both a spectrum file (either `.raw` or `.mzML`) and a .tsv spectral library are required. If you would like to convert `.raw` files to `.mzML` to run with JMod, please follow the instructions below.

### File Conversion 

JMod currently supports `.mzML` and `.raw` files. Direct support for `.d` files will be added in future releases. To convert `.raw` files to `.mzML` files, please make sure the data is centroided. This can be done with MSConvert with the command `--filter peakPicking true 1-`

###
### Library Structure  

<details>
<summary><strong> Description of Library Columns </strong></summary>


| Column | Definition | Req/Opt |
|---|---|---|
| ModifiedPeptide | Peptide sequence including modifications | Req |
| StrippedPeptide | Peptide sequence not including modifications | Req |
| PrecursorCharge | Charge of the precursor | Req |
| RT | Retention time of the precursor | Req |
| PrecursorMz | m/z of the precursor | Req |
| FragmentMz | m/z of the fragment | Req |
| RelativeIntensity | Intensity of the fragment | Req |
| FragmentType | Fragment type (b, y) | Req |
| FragmentCharge | Fragment charge (1, 2, etc.) | Req |
| FragmentSeriesNumber | Fragment series number (12, 9, 5, etc.) | Req |
| FragmentLossType | Fragment loss type (noloss, NH3, H2O) | Opt |
| ProteinID | All proteins that contain the fragment's precursor | Opt |
| ProteinGroup | Inferred proteins | Req |
| ProteinName | Names of proteins in ProteinGroup | Opt |
| Genes | Gene names mapping to proteins in ProteinGroup | Opt |
| IonMobility | Ion mobility of the precursor | Opt |

</details>

 An example library with the required columns can also be found [here](/data/filtered_library.tsv).

###
### Running JMod with the Graphical User Interface (GUI)

![alt text](/Help/JMod_GUI.jpeg "JMod GUI Image")

The GUI can be launched with the `launch_JMod.bat` file on Windows computers. 

If not on a Windows computer, the GUI can be launched with the following command:

```
uv run python path/to/run_jmod_from_GUI.py 
```

More detailed instructions on how to run JMod from the GUI can be found
[here.](/Help/JMod_Tutorial.pdf)


###
### Running JMod with the Command Line Interface (CLI) 

JMod can be run through the command line with various search parameters. An example command is shown below:


```
uv run python path/to/run_jmod.py -l path/to/library.tsv -i path/to/file_to_search.mzML
```

To search multiple files with one library, repeat `-i` or give a folder of data files with `--mzml_folder`:

```
uv run python path/to/run_jmod.py -l path/to/library.tsv -i path/to/file_1.mzML -i path/to/file_2.mzML
uv run python path/to/run_jmod.py -l path/to/library.tsv --mzml_folder path/to/data_folder
```

Some commonly used search parameters are listed below. A more extensive list of commands can be found [here](/Help/commands.pdf).

<details>
<summary><strong> Search Parameters </strong></summary>

```
-i, --mzml
  Input file in mzML format. Repeat to search multiple files
--mzml_folder
  Search every .mzML, .raw and .d file in this folder
-l, --speclib
  Spectrum library in DIANN output format (must be .tsv or .parquet)
-o, --output_folder
  Specify an output folder to send search results
  default = location of files being searched
-m --atleast_m
  Required number of fragments matched from top N fragments (N=10)
  default = 3
-p --ppm
  MS2 matching tolerance in parts per million.
  default = 10
--iso
  Use MS2 isotopes in search.
  default = False
--num_iso
  Number of MS2 isotopes to consider if using them
  default = 2
--apex_jitter
  Allow center scan for MS1 quant to slide this many scans if intensity is monotonically increasing.
  default = 0
--free_apex
  Allow channels to be quantified in different scans. For example, useful for non-coeluting channels caused by deuterium.
  default = False
--additional_scans
  If free-apex; Each channel apex is selected within this number of scans from the plex group center. 
  default = 0
--timeplex
  Use timePlex mode for search
  default = False
--num_timeplex
  Number of time offset injections for timePlex search
  default = 0
-t, --threads
  Number of threads to be used for the search
  default = 10
--tag
  Tag used in the experiment, if any. See mass_tags.py for details.
  default = None
--use_emp_rt
  Force use of library retention time for alignment.
  default = False
--user_rt_tol
  Force use of provided retention time tolerance.
  default = False
--rt_tol
  User provided retention time tolerance.
--no_ms1_req
  Don't require observation of an MS1 peak for consideration in the search.
  default = False
--ms1_ppm
  User provided MS1 ppm error tolerance.
--mbr
  Match between runs. Requires two or more files.
  default = False
--make_library
  Build the match between runs library from a finished search's results folder, without searching.
--combine_results
  Combine the results of a finished search's results folder again, without searching.
--iso_workers
  Number of processes used to generate isotopes.
  default = 3
--rawfilereader_path
  Path to the ThermoRawFileParser if using .raw files
--bruker_sdk_path
  Path to the Bruker tdf-sdk if using .d timsTOF files
  ```

</details>


####
JMod can also be run using a configuration file. Each JMod search produces its own configuration file which can be used to initialize other searches. An example configuration can be found in ```data/default_config.json```, and a sample command can be found here:

```
uv run python path/to/run_jmod.py --config_json path/to/config.json
```

Options given on the command line override those in the configuration file.


<details>
<summary><strong> Running JMod with Sample Demo Data </strong></summary>

We have provided a small .mzML file and a small library to run a quick JMod search to check that all dependencies and environment variables are working properly. The raw file and library can be found in data/test_mode_filtered.mzML and data/filtered_library.tsv respectively. This quick search can be run on the command line with the following command:

```
cd path/to/JMod-Main

uv run python run_jmod.py -i data/test_mode_filtered.mzML -l data/filtered_library.tsv
```

</details>


###
## Output Files


JMod produces multiple output files. Each search writes them to a new `JMod_Results` folder in the output folder (`JMod_Results_<text>` with `-z <text>`), with one results folder per searched file. Below is a brief description of the main outputs alongside an example directory structure. A more comprehensive description of each output file can be found [here.](/Help/outputs.pdf)

- ```combined_filtered_IDs.parquet```: IDs from all searched files filtered at 1% FDR, with global q-values
- ```experiment_results/```: Plots comparing the searched files
- ```JMod_config.json```: Configuration file for this current search
- ```JMod_log.log```: Log of the search
- ```filtered_IDs.parquet```: IDs filtered at 1% FDR with select columns
- ```filtered_IDs.csv```: IDs filtered at 1% FDR with extended columns
- ```Summary.txt```: Summary of precursor & protein identifications

With `--mbr`, the first search of each file is in `first_pass/` and the match between runs library is in `mbr_library/`.

A comprehensive list of all output columns and descriptions for each one can be found below:

<details>
<summary><strong> JMod Output Columns </strong></summary>

 | Column | Description |
|---|---|
| `coeff` | Coefficient estimated for the precursor |
| `spec_id` | MS2 spectrum ID where the precursor was found |
| `Ms1_spec_id` | Closest MS1 spectrum to the MS2 spectrum |
| `seq` | Precursor sequence |
| `z` | Precursor charge |
| `window_mz` | Center of the MS2 isolation window |
| `rt` | Retention time of the MS2 spectrum |
| `num_lib` | Number of library fragments matched |
| `frac_lib_int` | Fraction of the library intensity matched |
| `frac_dia_int` | Fraction of the observed intensity matched to the precursor |
| `mz_error` | Relative MS1 m/z error |
| `rt_error` | Retention time error (empirical RT - library RT) |
| `frac_int_matched` | Fraction of the total spectrum intensity with any match (any precursor) |
| `frac_int_pred` | Fraction of the total spectrum intensity predicted by all precursors |
| `spec_r2` | Pearson correlation for the predictions and observed spectrum |
| `prec_r2` | Pearson correlation for the predicted precursor and matched spectrum peaks |
| `prec_r2_uniq` | Pearson correlation for the predicted precursor and uniquely matched peaks |
| `frac_int_uniq` | Fraction of library intensity uniquely matched |
| `frac_int_uniq_pred` | Fraction of total spectrum intensity uniquely predicted |
| `hyperscore` | Sum of matched intensities times factorial(#b ions) times factorial(#y ions) |
| `b_counts` | Number of b ions matched |
| `y_counts` | Number of y ions matched |
| `longest_y_ions` | Index of largest y ion matched |
| `scribe_scores` | Scribe score as described [here](https://pubs.acs.org/doi/abs/10.1021/acs.jproteome.2c00672) |
| `max_unmatched_residuals` | Maximum residual for unmatched peaks, normalized and log-transformed |
| `max_matched_residuals` | Maximum residual for matched peaks, normalized and log-transformed |
| `gof_stats` | Goodness-of-fit score for each precursor (log2 of residuals/fitted) |
| `manhattan_distances` | Manhattan distances between precursor peaks and observed peaks (normalized and log-transformed) |
| `fitted_spectral_contrasts` | Spectral contrast angles for each precursor |
| `frac_int_matched_pred` | Fraction of the matched spectrum intensity predicted by all precursors |
| `frac_int_matched_pred_sigcoeff` | Fraction of spectrum intensity predicted by precursors with coeff > 1 |
| `cosine` | Similarity between predicted spectra and observed intensities |
| `mz` | Precursor m/z |
| `tic` | Total ion current |
| `file_name` | Name of the file searched |
| `protein` | Protein associated with the precursor |
| `is_decoy` | Is this a decoy peptide |
| `n_scans` | Number of scans where that precursor was identified |
| `manhattan_distances_nearby_max` | Maximum manhattan_distances of nearby scans |
| `max_matched_residuals_nearby_min` | Minimum max_matched_residuals of nearby scans |
| `gof_stats_nearby_min` | Minimum gof_stats of nearby scans |
| `scribe_scores_nearby_min` | Minimum scribe_scores of nearby scans |
| `smoothness` | The smoothness of coeffs for that precursor |
| `stripped_seq` | Amino acid sequence without tags or mods |
| `pep_len` | Peptide sequence length |
| `sq_rt_error` | Square root of the rt_error |
| `sq_mz_error` | Square root of the mz_error |
| `untag_seq` | Peptide sequence without the tag (contains mods) |
| `untag_prec` | Untag_seq and charge combined |
| `channels_matched` | Number of channels from the plex set matched |
| `median_[feature]_stats` | Median of [feature] across channels |
| `diff_[feature]_from_median` | Difference between channel [feature] and the median [feature] value across channels |
| `frac_shared_intensity` | Fraction of shared intensity between channels |
| `channel` | Channel name of the precursor |
| `silac_channel` | SILAC channel |
| `med_frag_error` | Median absolute m/z error of the matched fragments |
| `n_corr_scans` | Number of scans used in correlation calculation |
| `n_corr_frags` | Number of fragments used in correlation calculation |
| `mean_frag_corr` | Mean correlation between fragments |
| `median_frag_corr` | Median fragment correlations between fragments for the same precursor |
| `max_frag_corr` | Maximum fragment correlations between fragments for the same precursor |
| `min_frag_corr` | Minimum fragment correlation between fragments for the same precursor |
| `std_frag_corr` | Standard deviation of fragment correlations |
| `mean_top3_frag_corr` | Mean of the top 3 fragment correlations |
| `frac_corr_above_0p5` | Fraction of fragments with correlation above 50% |
| `mean_frag_mean_corr` | *(no description provided)* |
| `top_[n]_frag_mean_corr` | Mean of top [n] fragment correlations |
| `top_[n]_frag_sum_corr` | Sum of top [n] fragment correlations |
| `top_[n]_pair_corr` | Top [n] correlations between fragments |
| `mean_prec_frag_corr` | Mean precursor to fragment correlation |
| `max_prec_frag_corr` | Maximum precursor to fragment correlation |
| `PredVal` | Raw score from the target-decoy classifier (1 is target) |
| `Qvalue` | Q-value of the precursor |
| `BestChannel_Qvalue` | Best q-value of the precursor plex set |
| `plexfitMS1` | Fitted MS1 coefficient at max |
| `plexfitMS1_p` | Correlation of theoretical MS1 isotope distribution with observed peaks |
| `plexfittrace` | Fitted MS1 coefficients of elution trace |
| `plexfit_ps` | Correlation of theoretical MS1 isotope distribution with observed peaks for each fit in elution trace |
| `plexfittrace_spec_all` | All MS1 spectra used for MS1 fitting |
| `plexfittrace_all` | All fitted MS1 coefficients |
| `plexfittrace_ps_all` | Correlation of theoretical MS1 isotope distribution with observed peaks for each fit |
| `plex_Area` | Area under fitted MS1 coefficients |
| `ms1_apex_scan` | MS1 apex scan selected for MS quantitation |
| `ms1_cor` | Correlation between MS2 coefficients and monoisotopic trace |
| `traceproduct` | Product of MS1xMS1 correlations |
| `iso_cor` | Correlation between theoretical and observed MS1 isotopes |
| `MS1_Int` | Observed intensity of MS1 monoisotopic peak at max |
| `all_ms1_specs` | MS1 spectra searched for monoisotopic peak |
| `MS1_Area` | Area under elution trace of MS1 monoisotopic trace |
| `all_ms1_iso[n]vals` | Observed MS1 intensities for the [n]th isotopic peak (0 being the monoisotopic peak) |
| `last_aa` | Last amino acid in the peptide sequence |
| `seq_len` | Length of the amino acid sequence |
| `run_chan` | Sample run and channel concatenated |
| `Protein_Qvalue` | Protein q-value |
| `frag_names` | Names of fragments matched |
| `frag_errors` | m/z errors of fragments matched |
| `frag_mz` | Theoretical m/z of fragments matched |
| `frag_int` | Library intensity of fragments matched |
| `obs_int` | Observed intensity of fragments matched |
| `unique_frag_mz` | Fragments that uniquely matched a peak |
| `unique_obs_int` | Intensities for fragments that uniquely matched a peak |

</details>

####

### Structure of Results Directory:

```text
JMod_Results

├── combined_filtered_IDs.parquet
├── JMod_config.json
├── JMod_log.log
├── experiment_results/
│   └── [experiment_plots].png
└── [file]_results/
    ├── filtered_IDs.parquet
    ├── filtered_IDs.csv
    ├── Summary.txt
    ├── first_search/
    │   └── firstSearch.tsv
    ├── outputs/
    │   ├── all_IDs_filtered.parquet
    │   ├── all_IDs.csv
    │   ├── config.json
    │   ├── decoylibsearch_coeffs.parquet
    │   └── params.txt
    ├── scoring/
    └────── [scoring_plots].png

```


