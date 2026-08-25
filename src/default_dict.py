
## Defines the defaults and other parameters for JMod. These are used to initialize JMod both when running from the command line 
## and when running from GUI

#when changing default values, also think about changing Preset JSONS

"""
'name': {                         # Long flag without dashes
    'flags': ('-n', '--name'),    # Command line flags
    'default': default,           # Default Value
    'takes_value': Bool,          # True if the argument takes a value
    'in_GUI': Bool,               # True if the argument should be included in the GUI
    'type': '',                   # Type of the argument ('int', 'float', 'str') if takes_value is True, else ''
    'widget': '',                 # GUI widget type ('checkbutton', 'entry', 'dropdown') '' if not in GUI
    'tk_handle': None             # None. This is dynamically updated from run_jmod_from_GUI.py with the tk.IntVar, tk.StringVar, tk.BooleanVar, or ttk.Entry once it is initialized
    'special_upload': False       # False for all not in GUI and most in GUI. Function to add them to config_args_dict if needed that is passed the tk_handle for the current var and a mapping of default dict keys to their tk_handle
    'values': 'any'               # List of possible values ([] or 'any' or '+' or '+<x') if they are restricted by more than int, float, str, etc. 'any' if not restricted '+' is positive numbers only '+<x' is positive numbers less than or equal to x, and [] for any specific requirements
    }
"""  
default_dict = {
    'mzml': {
        'flags': ('-i', '--mzml'),  ## mzml is treated a bit specially in GUI due to running multiple files in a row so check if you change anything here
        'default': None,
        'takes_value': True,
        'in_GUI': True,
        'type': 'str',
        'widget': 'entry',
        'tk_handle': None,
        'special_upload': False,  #mzml is special, but hard coded specially in GUI
        'values': 'any'
    },
    'diaPASEF': {
        'flags': ('-d', '--diaPASEF'),
        'default': False,
        'takes_value': False,
        'in_GUI': False,
        'type': '',
        'widget': '',
        'tk_handle': None,
        'special_upload': False,  #diaPASEF is special, but hard coded specially in GUI because it relies on mzml
        'values': 'any'
    },
    'speclib': {  #hard coded a little because it is required to run JMod
        'flags': ('-l', '--speclib'),
        'default': None,
        'takes_value': True,
        'in_GUI': True,    
        'type': 'str',
        'widget': 'entry',
        'tk_handle': None,
        'special_upload': False,
        'values': 'any'
    },
    'use_rt': {
        'flags': ('-r', '--use_rt'),
        'default': True,
        'takes_value': False,
        'in_GUI': False,
        'type': '',
        'widget': '',
        'tk_handle': None,
        'special_upload': False,
        'values': 'any'
    },
    'use_features': {
        'flags': ('-f', '--use_features'),
        'default': True,
        'takes_value': False,
        'in_GUI': False,
        'type': '',
        'widget': '',
        'tk_handle': None,
        'special_upload': False,
        'values': 'any'
    },
    'atleast_m': {
        'flags': ('-m','--atleast_m'),
        'default': 3,
        'takes_value': True,
        'in_GUI': True,
        'type': 'int',
        'widget': 'dropdown',
        'tk_handle': None,
        'special_upload': False,
        'values': '+<10'
    },
    'apex_jitter': {
        'flags': ('--apex_jitter',),
        'default': 0,
        'takes_value': True,
        'in_GUI': True,
        'type': 'int',
        'widget': 'dropdown',
        'tk_handle': None,
        'special_upload': False,
        'values': '+'
    },
    'free_apex': {
        'flags': ('--free_apex',),
        'default': False,
        'takes_value': False,
        'in_GUI': False,
        'type': 'bool',
        'widget': 'checkbox',
        'tk_handle': None,
        'special_upload': False,
        'values': "+"
    },
    'additional_scans': {
        'flags': ('--additional_scans',),
        'default': 0,
        'takes_value': True,
        'in_GUI': False,
        'type': 'int',
        'widget': 'dropdown',
        'tk_handle': None,
        'special_upload': False,
        'values': '+'
    },
    ## store_false: passing --rt_norm (or the readable alias --no_rt_norm)
    ## turns the correction OFF, the same convention as --unfiltered_quant.
    ## MS2 quant restricted to fragments whose m/z differs between plexDIA
    ## channels. Under mTRAQ (rules "nK") a y-ion of an R-terminating peptide
    ## carries no tag, so all channels share that peak and the observed
    ## intensity is a superposition -- measured 99.6% of y-ions on JD0435, and
    ## R-terminating precursors are 46.9% of IDs, and they keep only ~36% of
    ## their library intensity in channel-resolved peaks.
    ## Validated against JMod's own coeff on K-terminating precursors, where
    ## nothing is superposed so the two must agree: r = 0.905 in log2. The gap
    ## from 1.0 is the joint deconvolution against co-isolated competitors,
    ## which this deliberately skips (it is what makes the refit ~1.5 min).
    ## store_false, as with --rt_norm.
    'ms2_channel_unique': {
        'flags': ('--ms2_channel_unique', '--no_ms2_channel_unique'),
        'default': True,
        'takes_value': False,
        'in_GUI': False,
        'type': '',
        'widget': 'checkbox',
        'tk_handle': None,
        'special_upload': False,
        'values': 'any'
    },
    ## Move the channel-unique MS2 refit BEFORE scoring, ungated, and add the
    ## cross-channel empirical-library features.  This is what makes those
    ## columns usable as scoring features (a value that exists only for
    ## confident targets predicts the label) and lets the RT normalizer build
    ## its MS2 curve from the uncompressed quantity.  Measured on JD0413:
    ## +3.84% precursors at 1% FDR, ~3.2 min added to a 31.5 min search.
    ## store_false: passing --no_ms2_prescoring turns it OFF.
    'ms2_prescoring': {
        'flags': ('--ms2_prescoring', '--no_ms2_prescoring'),
        'default': True,
        'takes_value': False,
        'in_GUI': False,
        'type': '',
        'widget': 'checkbox',
        'tk_handle': None,
        'special_upload': False,
        'values': 'any'
    },
    ## Report coeff_Area and coeff_ChannelUnique_Area: the MS2 coefficient
    ## summed over a window centred on the apex the whole plex group shares --
    ## the analogue of plex_Area.  Measured neutral on fold-change accuracy and
    ## slightly worse on precision than the single-scan values, and roughly
    ## doubles the pre-scoring stage, so it is separately switchable.
    ## store_false: passing --no_ms2_areas turns it OFF.
    'ms2_areas': {
        'flags': ('--ms2_areas', '--no_ms2_areas'),
        'default': True,
        'takes_value': False,
        'in_GUI': False,
        'type': '',
        'widget': 'checkbox',
        'tk_handle': None,
        'special_upload': False,
        'values': 'any'
    },
    ## Chunks the pre-scoring pass joins the held-back fragment columns in.
    ## More chunks = lower peak memory, marginally more overhead.
    ## How the cross-channel library features treat fragments whose m/z is
    ## shared by two co-isolated channels of the same precursor (mTRAQ leaves
    ## an R-terminating peptide's y-ions superposed).
    ##   'proportional' (default) keeps them, dividing each peak among the
    ##      channels in proportion to the quants fitted on their UNIQUE
    ##      fragments.  The peptide's true fragment intensity cancels out of
    ##      that ratio, so no library value is needed.
    ##   'drop' is the previous behaviour: delete them and renormalise the
    ##      survivors (an R-terminating precursor then keeps ~36% of its
    ##      library intensity).
    ## Measured on JD0413_re: proportional gives 9,955 distinct precursors at
    ## 1% FDR against 9,902 for drop (+0.54%), with rows, best-channel and AUC
    ## all up.  One run, one seed, no error bar.
    'ms2_shared_frags': {
        'flags': ('--ms2_shared_frags',),
        'default': 'proportional',
        'takes_value': True,
        'in_GUI': False,
        'type': 'str',
        'widget': 'dropdown',
        'tk_handle': None,
        'special_upload': False,
        'values': ['proportional', 'drop']
    },
    'ms2_prescoring_chunks': {
        'flags': ('--ms2_prescoring_chunks',),
        'default': 12,
        'takes_value': True,
        'in_GUI': False,
        'type': 'int',
        'widget': 'dropdown',
        'tk_handle': None,
        'special_upload': False,
        'values': '+'
    },
    ## Only FDR-passing, non-decoy rows are refitted -- the refit re-reads the
    ## MS2 spectra and is the expensive part.
    'ms2_channel_unique_qvalue': {
        'flags': ('--ms2_channel_unique_qvalue',),
        'default': 0.01,
        'takes_value': True,
        'in_GUI': False,
        'type': 'float',
        'widget': '',
        'tk_handle': None,
        'special_upload': False,
        'values': '+'
    },
    ## Do not attempt the isotope correction when fewer than this many channels
    ## of the precursor were co-isolated. OFF (0) by default: stratified on
    ## JD0588 the correction looked harmful below 7 channels (0.651 -> 0.619
    ## against 0.756 -> 0.888 above it), but that rests on n=139, and switching
    ## it on costs 0.006 compression on the population that can actually be
    ## measured -- complete-sample-set precursors are channel-rich and sit above
    ## the threshold anyway. Available for data with many sparse groups.
    'ms2_channel_unique_min_siblings': {
        'flags': ('--ms2_channel_unique_min_siblings',),
        'default': 0, 'takes_value': True, 'in_GUI': False,
        'type': 'int', 'widget': 'entry', 'tk_handle': None,
        'special_upload': False, 'values': 'any'
    },
    ## Anchor the common apex with an intensity-weighted median of the
    ## channels' RTs instead of a plain median. A dim channel localises its
    ## apex badly, and on JD1187 the channels of one precursor disagree by a
    ## median of 0.33 min (p90 1.7) because each sits in its own 2.0 m/z window
    ## and the 250-window cycle is long. Measured there: precursors whose
    ## channels agree on RT quantify at 0.719 compression against 0.263 for
    ## those that do not -- the anchor dominates everything else on that file.
    'ms2_weighted_common_apex': {
        'flags': ('--ms2_weighted_common_apex',),
        'default': True, 'takes_value': False, 'in_GUI': False,
        'type': '', 'widget': '', 'tk_handle': None,
        'special_upload': False, 'values': 'any'
    },
    'ms2_area_adjacent_scans': {
        # Scans either side of the apex summed into the *_Area columns (and the
        # common-apex area).  1 measured best on JD1187 (median per-precursor
        # error 1.206 -> 1.122 at no cost in slope); 0 is better on wide-window
        # data like JD0588, where every extra scan hurts both metrics.  Exposed
        # because no single value serves both acquisition geometries.
        'flags': ('--ms2_area_adjacent_scans',),
        'default': 1, 'takes_value': True, 'in_GUI': False,
        'type': 'int', 'widget': 'entry', 'tk_handle': None,
        'special_upload': False, 'values': '+'
    },
    'ms2_fragment_purity_diag': {
        'flags': ('--ms2_fragment_purity_diag',),
        'default': False,
        'takes_value': False,
        'in_GUI': False,
        'type': '',
        'widget': '',
        'tk_handle': None,
        'special_upload': False,
        'values': 'any'
    },
    'ms2_channel_unique_isotope_n': {
        'flags': ('--ms2_channel_unique_isotope_n',),
        'default': 5,
        'takes_value': True,
        'in_GUI': False,
        'type': 'int',
        'widget': '',
        'tk_handle': None,
        'special_upload': False,
        'values': '+'
    },
    'rt_norm': {
        'flags': ('--rt_norm', '--no_rt_norm'),
        'default': True,
        'takes_value': False,
        'in_GUI': False,
        'type': '',
        'widget': 'checkbox',
        'tk_handle': None,
        'special_upload': False,
        'values': 'any'
    },
    'rt_norm_method': {
        'flags': ('--rt_norm_method',),
        'default': 'loess_cap',
        'takes_value': True,
        'in_GUI': False,
        'type': 'str',
        'widget': 'dropdown',
        'tk_handle': None,
        'special_upload': False,
        'values': ['rolling_linear', 'rolling_median', 'rolling_mean', 'loess_n',
                   'rt_window', 'loess', 'tic', 'tic_rolling', 'none']
    },
    'rt_norm_window': {
        'flags': ('--rt_norm_window',),
        'default': 400,
        'takes_value': True,
        'in_GUI': False,
        'type': 'int',
        'widget': 'entry',
        'tk_handle': None,
        'special_upload': False,
        'values': '+'
    },
    'rt_norm_rt_window': {
        'flags': ('--rt_norm_rt_window',),
        'default': 2.0,
        'takes_value': True,
        'in_GUI': False,
        'type': 'float',
        'widget': '',
        'tk_handle': None,
        'special_upload': False,
        'values': '+'
    },
    'rt_norm_loess_frac': {
        'flags': ('--rt_norm_loess_frac',),
        'default': 0.05,
        'takes_value': True,
        'in_GUI': False,
        'type': 'float',
        'widget': '',
        'tk_handle': None,
        'special_upload': False,
        'values': '+<1'
    },
    'rt_norm_qvalue': {
        'flags': ('--rt_norm_qvalue',),
        'default': 0.05,
        'takes_value': True,
        'in_GUI': False,
        'type': 'float',
        'widget': '',
        'tk_handle': None,
        'special_upload': False,
        'values': '+<1'
    },
    'rt_norm_plex_agg': {
        'flags': ('--rt_norm_plex_agg',),
        'default': 'mean',
        'takes_value': True,
        'in_GUI': False,
        'type': 'str',
        'widget': '',
        'tk_handle': None,
        'special_upload': False,
        'values': ['mean', 'geomean', 'median']
    },
    'rt_norm_top_frac': {
        'flags': ('--rt_norm_top_frac',),
        'default': 1.0,
        'takes_value': True,
        'in_GUI': False,
        'type': 'float',
        'widget': '',
        'tk_handle': None,
        'special_upload': False,
        'values': '+<1'
    },
    'rt_norm_min_span': {
        'flags': ('--rt_norm_min_span',),
        'default': 50,
        'takes_value': True,
        'in_GUI': False,
        'type': 'int',
        'widget': '',
        'tk_handle': None,
        'special_upload': False,
        'values': '+'
    },
    'rt_norm_min_prec': {
        'flags': ('--rt_norm_min_prec',),
        'default': 50,
        'takes_value': True,
        'in_GUI': False,
        'type': 'int',
        'widget': '',
        'tk_handle': None,
        'special_upload': False,
        'values': '+'
    },
    'rt_norm_two_stage': {
        'flags': ('--rt_norm_two_stage', '--no_rt_norm_two_stage'),
        'default': True,
        'takes_value': False,
        'in_GUI': False,
        'type': '',
        'widget': '',
        'tk_handle': None,
        'special_upload': False,
        'values': 'any'
    },
    'rt_norm_run_stage': {
        'flags': ('--rt_norm_run_stage',),
        'default': 'auto',
        'takes_value': True,
        'in_GUI': False,
        'type': 'str',
        'widget': '',
        'tk_handle': None,
        'special_upload': False,
        'values': ['auto', 'always', 'never']
    },
    'rt_norm_mz_stage': {
        'flags': ('--rt_norm_mz_stage',),
        'default': False,
        'takes_value': False,
        'in_GUI': False,
        'type': '',
        'widget': '',
        'tk_handle': None,
        'special_upload': False,
        'values': 'any'
    },
    'rt_norm_stage_order': {
        'flags': ('--rt_norm_stage_order',),
        'default': 'unpaired_first',
        'takes_value': True,
        'in_GUI': False,
        'type': 'str',
        'widget': '',
        'tk_handle': None,
        'special_upload': False,
        'values': ['unpaired_first', 'paired_first',
                   'unpaired_paired_unpaired']
    },
    'rt_norm_center_scope': {
        'flags': ('--rt_norm_center_scope',),
        'default': 'channel',
        'takes_value': True,
        'in_GUI': False,
        'type': 'str',
        'widget': '',
        'tk_handle': None,
        'special_upload': False,
        'values': ['channel', 'global']
    },
    ## Floor on the quantity for a row to define the reference curve. JMod writes
    ## 0.001 into a trace position with no observed signal, so a bare > 0 test
    ## lets those placeholders set the local centre. Every row is still
    ## corrected; this only decides which rows are trusted to fit the curve.
    'rt_norm_min_intensity': {
        'flags': ('--rt_norm_min_intensity',),
        'default': 1.0,
        'takes_value': True,
        'in_GUI': False,
        'type': 'float',
        'widget': '',
        'tk_handle': None,
        'special_upload': False,
        'values': '+'
    },
    'rt_norm_inplace': {
        'flags': ('--rt_norm_inplace',),
        'default': False,
        'takes_value': False,
        'in_GUI': False,
        'type': '',
        'widget': '',
        'tk_handle': None,
        'special_upload': False,
        'values': 'any'
    },
    'no_rt_norm_plot': {
        'flags': ('--no_rt_norm_plot',),
        'default': False,
        'takes_value': False,
        'in_GUI': False,
        'type': '',
        'widget': '',
        'tk_handle': None,
        'special_upload': False,
        'values': 'any'
    },
    'threads': {
        'flags': ('-t', '--threads'),
        'default': 10,
        'takes_value': True,
        'in_GUI': False,
        'type': 'int',
        'widget': '',
        'tk_handle': None,
        'special_upload': False,
        'values': '+'
    },
    'ppm': {
        'flags': ('-p', '--ppm'),
        'default': 10,
        'takes_value': True,
        'in_GUI': True,
        'type': 'float',
        'widget': 'entry',
        'tk_handle': None,
        'special_upload': False,
        'values': '+'
    },
    'output_folder': {
        'flags': ('-o', '--output_folder'),
        'default': None,
        'takes_value': True,
        'in_GUI': True,
        'type': 'str',
        'widget': 'entry',
        'tk_handle': None,
        'special_upload': False,
        'values': 'any'
    },
    'ms2_align': {
        'flags': ('--ms2_align',),
        'default': False,
        'takes_value': False,
        'in_GUI': False,
        'type': '',
        'widget': '',
        'tk_handle': None,
        'special_upload': False,
        'values': 'any'
    },
    'timeplex': {  ##timeplex has some hard coded elements in GUI because it is determined indirectly
        'flags': ('--timeplex',),
        'default': False,
        'takes_value': False,
        'in_GUI': False,
        'type': '',
        'widget': '',
        'tk_handle': None,
        'special_upload': lambda handle, handles: int(handles['num_timeplex'].get()) > 1,
        'values': 'any'
    },
    'num_timeplex': {
        'flags': ('--num_timeplex',),
        'default': 0,
        'takes_value': True,
        'in_GUI': True,
        'type': 'int',
        'widget': 'dropdown',
        'tk_handle': None,
        'special_upload': False,
        'values': '+'
    },
    'iso': {
        'flags': ('--iso',),
        'default': False,
        'takes_value': False,
        'in_GUI': True,
        'type': '',
        'widget': 'checkbutton',
        'tk_handle': None,
        'special_upload': False,
        'values': 'any'
    },
    'unmatched': {
        'flags': ('--unmatched',),
        'default': 'c',
        'takes_value': True,
        'in_GUI': False,
        'type': 'str',
        'widget': '',
        'tk_handle': None,
        'special_upload': False,
        'values': ['a', 'b', 'c']
    },
    'decoy': {
        'flags': ('--decoy',),
        'default': 'rev',
        'takes_value': True,
        'in_GUI': False,
        'type': 'str',
        'widget': '',
        'tk_handle': None,
        'special_upload': False,
        'values': ['rev', 'rev_nc', 'shuffle']
    },
    'mTRAQ': {
        'flags': ('--mTRAQ',),
        'default': False,
        'takes_value': False,
        'in_GUI': False,
        'type': '',
        'widget': '',
        'tk_handle': None,
        'special_upload': False,
        'values': 'any'
    },
    'tag': {
        'flags': ('--tag',),
        'default': 'None',
        'takes_value': True,
        'in_GUI': True,
        'type': 'str',
        'widget': 'dropdown',
        'tk_handle': None,
        'special_upload': False,
        'values': None ## this will be set dynamically when the taglist is read into run_jmod_from_GUI. values does not affect command line reading
    },
    'num_iso': {
        'flags': ('--num_iso',),
        'default': 2,
        'takes_value': True,
        'in_GUI': True,
        'type': 'int',
        'widget': 'dropdown',
        'tk_handle': None,
        'special_upload': False,
        'values': '+'
    },
    'pp_file': {
        'flags': ('--pp_file',),
        'default': "",
        'takes_value': True,
        'in_GUI': False,
        'type': 'str',
        'widget': '',
        'tk_handle': None,
        'special_upload': False,
        'values': 'any'
    },
    'timspeak_file': {
        'flags': ('--timspeak_file',),
        'default': "",
        'takes_value': True,
        'in_GUI': False,
        'type': 'str',
        'widget': '',
        'tk_handle': None,
        'special_upload': False,
        'values': 'any'
    },
    'lib_frac': {
        'flags': ('--lib_frac',),
        'default': 0.5,
        'takes_value': True,
        'in_GUI': True,
        'type': 'float',
        'widget': 'entry',
        'tk_handle': None,
        'special_upload': False,
        'values': '+<1'
    },
    'dummy_value': {
        'flags': ('-z', '--dummy_value'),
        'default': None,
        'takes_value': True,
        'in_GUI': False,
        'type': 'str',
        'widget': '',
        'tk_handle': None,
        'special_upload': False,
        'values': 'any'
    },
    'plexDIA': { ##plexDIA has some hard coded elements in GUI because it is determined indirectly
        'flags': ('--plexDIA',),
        'default': False,
        'takes_value': False,
        'in_GUI': False,
        'type': '',
        'widget': '',
        'tk_handle': None,
        'special_upload': lambda handle, handles: (int(handles['num_timeplex'].get()) > 1 or handles['tag'].get() != "None"),
        'values': 'any'
    },
    'SILAC': {
        'flags': ('--SILAC',),
        'default': None,
        'takes_value': True,
        'in_GUI': False,
        'type': 'str',
        'widget': '',
        'tk_handle': None,
        'special_upload': False,
        'values': 'any'
    },
    'use_emp_rt': {
        'flags': ('--use_emp_rt',),
        'default': False,
        'takes_value': False,
        'in_GUI': True,
        'type': '',
        'widget': 'checkbutton',
        'tk_handle': None,
        'special_upload': False,
        'values': 'any'
    },
    'predict_decoys': {
        'flags': ('--predict_decoys',),
        'default': False,
        'takes_value': False,
        'in_GUI': False,
        'type': '',
        'widget': 'checkbutton',
        'tk_handle': None,
        'special_upload': False,
        'values': 'any'
    },
    'unfiltered_quant': {
        'flags': ('--unfiltered_quant',),
        'default': True,
        'takes_value': False,
        'in_GUI': False,
        'type': '',
        'widget': '',
        'tk_handle': None,
        'special_upload': False,
        'values': 'any'
    },
    'user_rt_tol': {
        'flags': ('--user_rt_tol',),
        'default': False,
        'takes_value': False,
        'in_GUI': True,
        'type': '',
        'widget': 'checkbutton',
        'tk_handle': None,
        'special_upload': False,
        'values': 'any'
    },
    'rt_tol': {
        'flags': ('--rt_tol',),
        'default': 0.5,
        'takes_value': True,
        'in_GUI': True,
        'type': 'float',
        'widget': 'entry',
        'tk_handle': None,
        'special_upload': lambda handle, handles: (float(handle.get()) if handles['user_rt_tol'].get() else 0.5),
        'values': '+'
    },
    'initial_percentile': {
        'flags': ('--initial_percentile',),
        'default': 50,
        'takes_value': True,
        'in_GUI': False,
        'type': 'float',
        'widget': '',
        'tk_handle': None,
        'special_upload': False,
        'values': '+<100'
    },
    'user_percentile': {
        'flags': ('--user_percentile',),
        'default': False,
        'takes_value': False,
        'in_GUI': False,
        'type': '',
        'widget': '',
        'tk_handle': None,
        'special_upload': False,
        'values': 'any'
    },
    'no_ms1_req': {
        'flags': ('--no_ms1_req',),
        'default': True,
        'takes_value': False,
        'in_GUI': False,
        'type': '',
        'widget': '',
        'tk_handle': None,
        'special_upload': False,
        'values': 'any'
    },
    'ms1_ppm': {
        'flags': ('--ms1_ppm',),
        'default': 0,
        'takes_value': True,
        'in_GUI': True,
        'type': 'float',
        'widget': 'entry',
        'tk_handle': None,
        'special_upload': False,
        'values': '+'
    },
    'config_json': {
        'flags': ('--config_json',),
        'default': None,
        'takes_value': True,
        'in_GUI': False,
        'type': 'str',
        'widget': '',
        'tk_handle': None,
        'special_upload': False,
        'values': 'any'
    },
    'test_mode': {
        'flags': ('--test_mode',),
        'default': False,
        'takes_value': False,
        'in_GUI': False,
        'type': '',
        'widget': '',
        'tk_handle': None,
        'special_upload': False,
        'values': 'any'
    },
    'test_rt_min': {
        'flags': ('--test_rt_min',),
        'default': 0,
        'takes_value': True,
        'in_GUI': False,
        'type': 'float',
        'widget': '',
        'tk_handle': None,
        'special_upload': False,
        'values': '+'
    },
    'test_rt_max': {
        'flags': ('--test_rt_max',),
        'default': 10,
        'takes_value': True,
        'in_GUI': False,
        'type': 'float',
        'widget': '',
        'tk_handle': None,
        'special_upload': False,
        'values': '+'
    },
    'test_mz_min': {
        'flags': ('--test_mz_min',),
        'default': 400,
        'takes_value': True,
        'in_GUI': False,
        'type': 'float',
        'widget': '',
        'tk_handle': None,
        'special_upload': False,
        'values': '+'
    },
    'test_mz_max': {
        'flags': ('--test_mz_max',),
        'default': 800,
        'takes_value': True,
        'in_GUI': False,
        'type': 'float',
        'widget': '',
        'tk_handle': None,
        'special_upload': False,
        'values': '+'
    }
}