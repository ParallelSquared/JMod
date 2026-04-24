
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
'score_lib_frac': {
        'flags': ('--score_lib_frac',),
        'default': 0.5,
        'takes_value': True,
        'in_GUI': False,
        'type': 'float',
        'widget': 'entry',
        'tk_handle': None,
        'special_upload': False,
        'values': '+<1'
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