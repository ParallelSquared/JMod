# jmod
Spectrum-centric search engine

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
  Use features as a filter in the search. Mzml file must have associated Dinosaur/Biosaur output
  default = False
-m --atleast_m
  Required number of fragments matched from top N fragments (N=10)
  default = 3
-t --threads
  How many threads the search uses
  default = 10
-p --ppm
  MS2 matching tolerance in parts per million.
  default = 20
--iso
  Use MS2 isotopes in search.
  default = False
--num_iso
  Number of MS2 isotopes to consider if using them
  default = 2
--unmatched
  What way to model unmatched intensity (see config file for more details)
  default = a
--decoy
  How to create decoy peptides. Options: rev / diann
  default = rev
--tag
  Tag used in the experiment, if any. See mass_tags.py for details
  default = None

```


