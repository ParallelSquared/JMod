# JMod
JMod is an open and flexible software for increasing the throughput os sensitive proteomics, supporting multiplexing in the mass and time domains


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


