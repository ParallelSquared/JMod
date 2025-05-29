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

Running a search:
```
python run_jmod.py -l path/to/library.tsv -i path/to/file_to_search.mzml
```

Parameters:
```
-i, --mzml
  Input file in mzml format
-l, --speclib
  Spectrum library in DIANN output format (must be .tsv)
-r, --use_rt
  Use retention time filtering during search
  default = False
-f, --use_features
  Use peptide-like features in the preliminary search. Mzml file must have associated Dinosaur/Biosaur output
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
  Force use of provided rentention time tolerance.
  default = False
--rt_tol
  User provided retention time tolerance.
--no_ms1_req
  Dont't require observation of an MS1 peak for consideration in the search.
  default = False
--ms1_ppm
  User provided MS1 ppm error tolerance.
```


