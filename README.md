<!-- TODO: Add a logo/image here? -->

# JMod

**JMod is an open and flexible software for increasing the throughput of sensitive proteomics by supporting multiplexing in the mass and time domains.**


## Reference

**JMod: Joint modeling of mass spectra for empowering multiplexed DIA proteomics**

Kevin McDonnell, Nathan Wamsley, Jason Derks, Sarah Sipe, Maddy Yeh, Harrison Specht, Nikolai Slavov
*bioRxiv* 2025.05.22.655512; doi: [10.1101/2025.05.22.655512](https://doi.org/10.1101/2025.05.22.655512)

<!-- TODO: Boil abstract down to bullet points and insert here -->

## Table of Contents
- [Setting up JMod](#environment-setup)
  - [Windows Batch Script](#setting-up-jmod-with-windows-gui-batch-script)
  - [Command Line Setup](#setting-up-a-jmod-uv-environment-via-the-command-line)
- [Running a Search](#running-a-search)
  - [File Conversion](#file-conversion)
  - [Command Line Args](#command-line-args)
  - [JSON Configuration File](#json-configuration-file)
  - [Graphic User Interface (GUI)](#graphic-user-interface-gui)
- [Output Files](#output-files)
- [Reference](#reference)

## Setting up JMod

### Setting up JMod with Windows GUI Batch Script

This `.bat` executable is only compatible with Windows computers. If your computer is running MacOS/Linux, please scroll down to the command line UV environment installation to set up and run JMod.

1. Download the JMod repository. The most recent release of JMod can be downloaded [here.](https://github.com/ParallelSquared/JMod/releases/tag/v1.0.0)

<!-- TODO: Add the J icon to the .bat file -->

2. Navigate to the JMod directory. Inside that directory is a `launch.bat` file. Double-click on the file to open the JMod GUI.
    - If this is the first time the computer is setting up a UV environment, it might take a few minutes to download all dependencies and packages.
3. The JMod GUI should open in a new window, looking something like this:

![alt text](/Help/JMod_GUI.jpeg "JMod GUI Image")


### Setting up a JMod UV Environment via the Command Line

1. Download the JMod repository. The most recent release of JMod can be downloaded [here.](https://github.com/ParallelSquared/JMod/releases/tag/v1.0.0)

2. It is recommended to use the UV package manager when running JMod. To set up a UV environment for JMod, please install UV with either ```pip install uv``` or [via wget/curl](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer) if it is not already installed. 

3. Open a new terminal window and navigate to the JMod directory. The directory contains a `pyproject.toml` that lists all required packages and dependencies. Run the following command to install the environment: ```uv sync --python 3.11```

4. Use the command ```source .venv/bin/activate``` to activate the UV environment. You should see the environment name (JMod-Main) appear at the beginning of the terminal prompt.
    - If you are on a Windows computer, run ```.venv/Scripts/activate``` to activate the UV environment.

5. You can now launch the JMod GUI with ```python run_jmod_from_GUI.py``` or run a search using the command line with ```python run_jmod.py <args>```.


## Running a Search

To run a JMod search, both a .mzML spectrum file and a .tsv spectral library are required.

### File Conversion 

JMod currently supports `.mzML` files. Direct support for `.d` and `.raw` files will be added in future releases. In the meantime, please convert `.raw` files to `.mzML` files. When converting files to `.mzML`, the data should be centroided. This can be done with MSConvert with the command `--filter peakPicking true 1-`

###
### Library Structure  

JMod requires specific library columns in a .tsv for searches to run successfully. An example library with the required columns can be found [here](/data/filtered_library.tsv).

<!-- TODO: Image of the library headings (?) or something similar -->

###
### Command Line Interface (CLI) & Graphic User Interface (GUI)

JMod can be run via either a CLI or a GUI. Tutorials on how to do both can be found below.

###
### Command Line Interface (CLI) 

JMod can be run through the command line with various search parameters. An example command is shown below:
```
uv run python path/to/run_jmod.py -l path/to/library.tsv -i path/to/file_to_search.mzML
```

Some commonly used search parameters are listed below. A more extensive list of commands can be found [here](/Help/commands.pdf).

<details>
<summary><strong> Select Parameters </strong></summary>

```
-i, --mzml
  Input file in mzML format
-l, --speclib
  Spectrum library in DIANN output format (must be .tsv)
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

</details>


####
JMod can also be run with a preset configuration file. That configuration file will include the path to the raw file and the library alongside other preset search parameters. An example configuration can be found in ```data/default_config.json```, and a sample command can be found here:

```
python path/to/run_jmod.py --config_json path/to/config.json
```


 
### Graphic User Interface (GUI)

JMod can be run through a GUI. To launch the GUI, use:

```
python path/to/run_jmod_from_GUI.py 
```

More detailed instructions on how to run JMod from the GUI can be found
[here.](/Help/JMod_Tutorial.pdf)

###
### Output Files

<!-- TODO: Fix wording -->

JMod produces multiple output files. Below is a brief description of the main outputs alongside an example directory structure. A more comprehensive description of each output file can be found [here.](/Help/outputs.pdf)

- ```filtered_IDs.parquet```: IDs filtered at 1% FDR with select columns
- ```filtered_IDs.csv```: IDs filtered at 1% FDR with extended columns
- ```config.json```: Configuration file for this current search. If the search ever needs to be repeated, this configuration file can be used in lieu of inputting all search parameters again
- ```Log.log```: Log of the search. If there are any errors or warnings during the search, they will be printed out here
- ```Summary.log```: Log showing summary statistics of precursor & protein identification

####

```text
search_results_directory

├── first_search/
│   └── firstSearch.tsv
├── outputs/
│   ├── all_IDs_filtered.parquet
│   ├── all_IDs.csv
│   ├── decoylibsearch_coeffs.parquet
│   └── params.txt
├── scoring/
├── filtered_IDs.parquet
├── filtered_IDs.csv
├── config.json
├── Log.log
└── Summary.txt
```


<!-- TODO: config moved out of outputs + called filtered_IDs.parquet -->
