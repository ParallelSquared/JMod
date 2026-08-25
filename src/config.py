"""
This Source Code Form is subject to the terms of the Oxford Nanopore
Technologies, Ltd. Public License, v. 1.0.  Full licence can be found
at https://github.com/ParallelSquared/JMod/blob/main/LICENSE.txt
"""
import argparse
import json
from src.default_dict import default_dict
import sys

from src.logger import logger

parser = argparse.ArgumentParser(
                    prog='Jmod',
                    description='What the program does',
                    epilog='Text at the bottom of help')

for key, value in default_dict.items():
    if value['takes_value'] is False:
        parser.add_argument(*value['flags'], action='store_true' if value['default'] is False else 'store_false')
    else:
        parser.add_argument(*value['flags'], default=value['default'], type=int if value['type'] == 'int' else float if value['type'] == 'float' else str)

if any("ipykernel_launcher" in arg or "jupyter" in arg or "pytest" in arg for arg in sys.argv):
    # running inside Jupyter → pass empty list
    args = parser.parse_args([])
else:
    args = parser.parse_args()
#if __name__ == "__main__":
#    args = parser.parse_args()
#else:
#    # When imported, create args with defaults
#    args = argparse.Namespace()
#    # Set all the default values here

# args.use_rt=True
# args.use_features=True
# logger.info(args)

RANDOM_SEED = 42

ran_from_GUI = False
error_already_handled = False
GUI_result_queue = None


ms1_ppm = 20
ms1_tol = ms1_ppm*10**(-6)

rt_tol = 9
im_tol = 0.5
im_merge_tol = 0.005
rt_width = 1.5


top_n_pasef = 10
atleast_m_pasef = 3

max_window_offset = None


chunksize = 10
batch_size = 1000


# rt_filtering = False
# feature_filtering = False

# How many spectra to fit RT alignment
n_most_intense = 400
n_most_intense_features = 10000

rt_percentile = .95

opt_rt_tol = rt_tol # set as default to start
opt_ms1_tol = ms1_tol 
opt_im_tol = im_tol

max_num_prelim_search = 1e5


# protein_column = 'protein_name'
protein_column = 'protein_group'


### the minimum number of IDs necessary for fine-tuning
FT_minimum = 1e3

### minimum ms1 tolerance; set to 0 to ignore
min_ms1_tol = 0#3e-6

min_iso_intensity = 1e-3 ## derived empirically

## how many isotope pearsonR corr to collect
num_iso_r = 2

## how many isotope traces to collect
num_iso_ms1 = 6

## how much to offset the decoy prec mz
decoy_mz_offset = 0


### moving average window MS1 trace
smoothing_window = 3

## min number of scans???


### test
test_var = "test_1"

### filters for picking spectra to fit
top_n = 10
"""
3 fit_types:
    a: All summed unmatched intensities are fit to a single zero intensity "obs peak"
    b: Summed unmatched intensities of each precursor are fit to their own zero intensity "obs peak"
    c: Each unmatched peak is fit to its own zero intensity "obs peak"

"""


score_model = "xg"
"""
3 options:
    rf: Random Forest
    lda: Linear Discriminant Analyisis
    xg: XgBoost
"""

tree_max_depth = 20

fdr_threshold = 0.01

target_decoy_ratio = 1 # added to get the tests working


#############################################


diann_mods = {
"UniMod:4" : 57.021464 ,
"Carbamidomethyl (C)" : 57.021464 ,
"Carbamidomethyl" : 57.021464 ,
"CAM" : 57.021464 ,
"+57" : 57.021464 ,
"+57.0" : 57.021464 ,
"UniMod:26" : 39.994915 ,
"PCm" : 39.994915 ,
"UniMod:5" : 43.005814 ,
"Carbamylation (KR)" : 43.005814 ,
"+43" : 43.005814 ,
"+43.0" : 43.005814 ,
"CRM" : 43.005814 ,
"UniMod:7" : 0.984016 ,
"Deamidation (NQ)" : 0.984016 ,
"Deamidation" : 0.984016 ,
"Dea" : 0.984016 ,
"+1" : 0.984016 ,
"+1.0" : 0.984016 ,
"UniMod:35" : 15.994915 ,
"Oxidation (M)" : 15.994915 ,
"Oxidation" : 15.994915 ,
"Oxi" : 15.994915 ,
"+16" : 15.994915 ,
"+16.0" : 15.994915 ,
"Oxi" : 15.994915 ,
"UniMod:1" : 42.010565 ,
"Acetyl (Protein N-term)" : 42.010565 ,
"+42" : 42.010565 ,
"+42.0" : 42.010565 ,
"UniMod:255" : 28.0313 ,
"AAR" : 28.0313 ,
"UniMod:254" : 26.01565 ,
"AAS" : 26.01565 ,
"UniMod:122" : 27.994915 ,
"Frm" : 27.994915 ,
"UniMod:1301" : 128.094963 ,
"+1K" : 128.094963 ,
"UniMod:1288" : 156.101111 ,
"+1R" : 156.101111 ,
"UniMod:27" : -18.010565 ,
"PGE" : -18.010565 ,
"UniMod:28" : -17.026549 ,
"PGQ" : -17.026549 ,
"UniMod:526" : -48.003371 ,
"DTM" : -48.003371 ,
"UniMod:325" : 31.989829 ,
"2Ox" : 31.989829 ,
"UniMod:342" : 15.010899 ,
"Amn" : 15.010899 ,
"UniMod:1290" : 114.042927 ,
"2CM" : 114.042927 ,
"UniMod:359" : 13.979265 ,
"PGP" : 13.979265 ,
"UniMod:30" : 21.981943 ,
"NaX" : 21.981943 ,
"UniMod:401" : -2.015650 ,
"-2H" : -2.015650 ,
"UniMod:528" : 14.999666 ,
"MDe" : 14.999666 ,
"UniMod:385" : -17.026549 ,
"dAm" : -17.026549 ,
"UniMod:23" : -18.010565 ,
"Dhy" : -18.010565 ,
"UniMod:129" : 125.896648 ,
"Iod" : 125.896648 ,
"Phosphorylation (ST)" : 79.966331 ,
"UniMod:21" : 79.966331 ,
"+80" : 79.966331 ,
"+80.0" : 79.966331 ,
"UniMod:259" : 8.014199, 
"Lys8" : 8.014199, 
"UniMod:267" : 10.008269, 
"Arg10" : 10.008269, 
"UniMod:268" : 6.013809, 
"UniMod:269" : 10.027228,
"C13-6":6.0201290268
};


# Function to load configuration from JSON
def load_config_from_json(json_path):
    """Load configuration parameters from a JSON file."""
    try:
        with open(json_path, 'r') as f:
            config_data = json.load(f)
        
        # Update args with values from JSON. Downstream code reads
        # ``config.args.X`` directly, so there is no module-level alias to
        # re-sync — mutating the Namespace is enough.
        for key, value in config_data.items():
            if hasattr(args, key):
                setattr(args, key, value)

        # Set additional config variables if present
        if 'additional_config' in config_data:
            for key, value in config_data['additional_config'].items():
                if key in globals() and not key.startswith('__'):
                    globals()[key] = value
                
        return True
    except Exception as e:
        logger.warning(f"Error loading JSON configuration: {e}")
        return False
        