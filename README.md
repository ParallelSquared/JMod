# JMod
**JMod is an open and flexible software for increasing the throughput of sensitive proteomics by supporting multiplexing in the mass and time domains**

## Abstract 
The throughput of mass spectrometry (MS) proteomics can be increased substantially by multiplexing that enables parallelization of data acquisition. Such parallelization in the mass domain (plexDIA) and the time domain (timePlex) increases the density of mass spectra and the overlap between ions originating from different precursors, potentially complicating their analysis. To enhance sequence identification and quantification from such spectra, we developed an open source software for Joint Modeling of mass spectra: JMod. It uses the intrinsic structure in the spectra and explicitly models overlapping peaks as linear superpositions of their components. This modeling enabled performing 9-plexDIA using 2 Da offset PSMtags by deconvolving the resulting overlapping isotopic envelopes in both MS1 and MS2 space. The results demonstrate 9-fold higher throughput with preserved quantitative accuracy and coverage depth. This support for smaller mass offsets increases multiplexing capacity and thus proteomic throughput for a given plexDIA tag, and we demonstrate this generalizability with diethyl labeling. By supporting enhanced decoding of DIA spectra multiplexed in the mass and time domains, JMod provides an open and flexible software that enables increasing the throughput of sensitive proteomics.

------


## Reference 

**JMod: Joint modeling of mass spectra for empowering multiplexed DIA proteomics**
Kevin McDonnell, Nathan Wamsley, Jason Derks, Sarah Sipe, Maddy Yeh, Harrison Specht, Nikolai Slavov
*bioRxiv* 2025.05.22.655512; doi: [10.1101/2025.05.22.655512](https://doi.org/10.1101/2025.05.22.655512)

------ 

## Deployment  

### Table of Contents
- [Environment Setup](#environment-setup)
  - [Linux/MacOS](#linuxmacos)
  - [Windows](#windows)
  - [Sample Search (Test Mode)](#sample-search-test-mode)
- [Outputs](#outputs)
- [Running a Search](#running-a-search)
  - [File Conversion](#file-conversion)
  - [With Command Line Args](#with-command-line-args)
  - [With JSON Configuration File](#with-json-configuration-file)
- [Search Parameters](#search-parameters)
- [Running Tests](#running-tests)


### Environment Setup
#### Linux/MacOS
To set up a Conda environment to run JMod in, please download the `data/jmod_env.yml` file.

In a dedicated terminal, type:  
1. ```conda env create -f jmod_env.yml -n $env_name```
2. ```conda activate $env_name```


#### Windows
If on a Windows machine, please use the `data/jmod_env_windows.yml` file to set up your environment. Set up will follow the same steps as Linux/MacOS.

#### Sample Search (Test Mode)
To ensure that all packages and the environment were set up correctly, JMod can be run in test mode with the `test_mode.json` file located in the `data/` directory. Running JMod in test mode performs a full search on a subset of a small file with a small spectral library, allowing for a quick search and check that all functions are functioning properly.

The search can be run with the following command:
```
python run_jmod.py --config data/test_mode.json
```
If the conda environment was configured properly, JMod should start by printing the run configurations and proceed to run to completion.

### Outputs

- `filtered_IDs.csv` – list of precursor identifications that are FDR filtered
- `all_IDs.csv` – list of peptide identifications that are not FDR filtered
- `decoylibsearch_coeffs.csv` – list of JMod MS2 precursor coefficients in each
- `spectrumfirstSearch.csv` – Precursor IDs from the preliminary search
- `params.txt` – list of parameters used in the JMod search
- `summary.txt` – summary of the precursor and protein IDs from the search

### Running a Search

#### File Conversion
JMod supports .mzML files. When converting files to .mzML, the data should be centroided. This can be done with MSconvert through the command `--filter peakPicking true 1-`

#### With Command Line Args
```
python path/to/run_jmod.py -l path/to/library.tsv -i path/to/file_to_search.mzML
```
#### With JSON Configuration File 
Example `config.json` in ./data/default_config.json
```
python path/to/run_jmod.py --config_json path/to/config.json
```

### Search Parameters
```
-i, --mzml
  Input file in mzML format
-l, --speclib
  Spectrum library in DIANN output format (must be .tsv)
-r, --use_rt
  Use retention time filtering during search
  default = False
-f, --use_features
  Use peptide-like features in the preliminary search. mzML file must have associated Dinosaur/Biosaur output
  default = True
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
```

### Running Tests
```
python run_tests.py -c 
```