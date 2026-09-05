# JMod

**JMod is an open and flexible software for increasing the throughput of sensitive proteomics by supporting multiplexing in the mass and time domains.**

###
## Reference

**JMod: Joint modeling of mass spectra for empowering multiplexed DIA proteomics**

Kevin McDonnell, Nathan Wamsley, Jason Derks, Sarah Sipe, Maddy Yeh, Harrison Specht, Nikolai Slavov
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

1. Download the JMod repository. The most recent release of JMod can be downloaded [here.](https://github.com/ParallelSquared/JMod/releases/tag/v2.0.0)


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

 An example library with the required columns can be found [here](/data/filtered_library.tsv).

###
### Running JMod with the Graphical User Interface (GUI)

![alt text](/Help/JMod_GUI.jpeg "JMod GUI Image")

The GUI can be launched with the `launch_JMod.bat` file on Windows computers. 

If not on a Windows computer, the GUI can be launched with the following command:

```
uv run python path/to/run_jmod_from_GUI.py 
```

<!-- TODO: rename the .bat file to launch_JMod.bat // wrap this in a .exe with the icon-->

More detailed instructions on how to run JMod from the GUI can be found
[here.](/Help/JMod_Tutorial.pdf)


###
### Running JMod with the Command Line Interface (CLI) 

JMod can be run through the command line with various search parameters. An example command is shown below:


```
uv run python path/to/run_jmod.py -l path/to/library.tsv -i path/to/file_to_search.mzML
```

Some commonly used search parameters are listed below. A more extensive list of commands can be found [here](/Help/commands.pdf).

<details>
<summary><strong> Common Search Parameters </strong></summary>

```
-i, --mzml
  Input file in mzML format
-l, --speclib
  Spectrum library in DIANN output format (must be .tsv or .parquet)
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
JMod can also be run using a configuration file. Each JMod search produces its own configuration file which can be used to initialize other searches. An example configuration can be found in ```data/default_config.json```, and a sample command can be found here:

```
uv run python path/to/run_jmod.py --config_json path/to/config.json
```


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


JMod produces multiple output files. Below is a brief description of the main outputs alongside an example directory structure. A more comprehensive description of each output file can be found [here.](/Help/outputs.pdf)

- ```filtered_IDs.parquet```: IDs filtered at 1% FDR with select columns
- ```filtered_IDs.csv```: IDs filtered at 1% FDR with extended columns
- ```config.json```: Configuration file for this current search
- ```Log.log```: Log of the search
- ```Summary.txt```: Summary of precursor & protein identifications

####

```text
search_results_directory

├── filtered_IDs.parquet
├── filtered_IDs.csv
├── config.json
├── Log.log
├── Summary.txt
├── first_search/
│   └── firstSearch.tsv
├── outputs/
│   ├── all_IDs_filtered.parquet
│   ├── all_IDs.csv
│   ├── decoylibsearch_coeffs.parquet
│   └── params.txt
├── scoring/
└────── [scoring_plots].png

```


