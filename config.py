import argparse



parser = argparse.ArgumentParser(
                    prog='Jmod',
                    description='What the program does',
                    epilog='Text at the bottom of help')

parser.add_argument('-i','--mzml')
parser.add_argument('-d','--diaPASEF')
parser.add_argument('-l','--speclib')      
parser.add_argument('-r', '--use_rt', action='store_true')
parser.add_argument('-f', '--use_features', action='store_false')
parser.add_argument('-m', '--atleast_m', default=3, type=int)
parser.add_argument('-t', '--threads', default=10, type=int)
parser.add_argument('-p', '--ppm', default=20, type=float)
parser.add_argument('-o','--output_folder', default=None)   
parser.add_argument('--ms2_align', action='store_true')
parser.add_argument('--timeplex', action='store_true')
parser.add_argument('--num_timeplex', default=0, type=int)
parser.add_argument('--iso', action='store_true')
parser.add_argument('--unmatched', default="c", type=str)
parser.add_argument('--decoy', default="rev", type=str)
parser.add_argument('--mTRAQ', action='store_true')
parser.add_argument('--tag', default="", type=str)
parser.add_argument('--num_iso', default=2, type=float)
parser.add_argument('--pp_file', default="", type=str)
parser.add_argument('--timspeak_file', default="", type=str)
parser.add_argument('--lib_frac', default=.5, type=float)
parser.add_argument('-z','--dummy_value', type=str)
parser.add_argument('--plexDIA', action='store_true')
parser.add_argument('--use_emp_rt', action='store_true') #use original library RTs
parser.add_argument('--unfiltered_quant', action='store_false') #by default it will only do MS1 quant on precursors with best channel qvalue < 0.01
parser.add_argument('--score_lib_frac', default=.5, type=float) #minimum frac_lib_int that a precursor must have to score
parser.add_argument('--user_rt_tol', action='store_true') 
parser.add_argument('--rt_tol', default=.5, type=float)
parser.add_argument('--initial_percentile', default=50, type=float)
parser.add_argument('--user_percentile', action='store_true') 
parser.add_argument('--no_ms1_req', action='store_false') 
parser.add_argument("--ms1_ppm",default=0, type=float)



args = parser.parse_args()

# args.mzml = "/Volumes/Lab/Quant/CC20170118_SAM_Specter_Ecolidigest_DIA_01.mzML"
# args.speclib = "/Volumes/Lab/Quant/SpecLibs/EcoliSpPrositLib.msp.tsv"
# args.mzml = "/Users/kevinmcdonnell/Programming/Data/2023_10_05_lf-dia_375pg_ce24.mzml"
# args.use_rt=True
# args.use_features=True
# args.mzml = "/Users/kevinmcdonnell/Programming/Data/2023_08_28_LF-DIA_E480.mzML"
# args.speclib = "/Users/kevinmcdonnell/Programming/Data/SpecLibs/BrukerHumanPrositLib.msp.tsv"
# args.speclib = "/Users/kevinmcdonnell/Programming/Data/SpecLibs/s6Thermo_prosit.msp.tsv"
# args.speclib = "/Users/kevinmcdonnell/Programming/Data/SpecLibs/HBthermo_PrositFrags.tsv"
# args.speclib = "/Users/kevinmcdonnell/Programming/Data/SpecLibs/Human_Bruker_Library_PrositFrags.tsv"
# args.speclib = "/Users/kevinmcdonnell/Programming/Data/SpecLibs/ecoli_hb_thermo_prosit.tsv"
# args.speclib = "/Users/kevinmcdonnell/Programming/Data/SpecLibs/HBthermo_PrositPepsNCE27.msp.tsv"
# args.speclib = "/Users/kevinmcdonnell/Programming/Data/SpecLibs/HBthermo_PrositPepsNCE39.msp.tsv"
# args.speclib = "/Users/kevinmcdonnell/Programming/Data/SpecLibs/HBthermo_PrositPepsNCE45.msp.tsv"
# args.speclib = "/Users/kevinmcdonnell/Programming/Data/SpecLibs/HBthermo_PrositPepsNCE51.msp.tsv"
# print(args)


mz_ppm = args.ppm
mz_tol = mz_ppm*10**(-6)
ms1_ppm = 20
ms1_tol = ms1_ppm*10**(-6)

num_timeplex = args.num_timeplex

rt_tol = 9
im_tol = 0.5
im_merge_tol = 0.005
rt_width = 1.5


top_n_pasef = 10
atleast_m_pasef = 3

max_window_offset = None

numProc = args.threads


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



protein_column = 'protein_name'


### the minimum number of IDs necessary for fine-tuning
FT_minimum = 1e3

### minimum ms1 tolerance; set to 0 to ignore
min_ms1_tol = 3e-6

num_iso_peaks = args.num_iso
min_iso_intensity = 1e-3 ## derived empirically

## how many isotope pearsonR corr to collect
num_iso_r = 2

## how many isotope traces to collect
num_iso_ms1 = 6

## how much to offset the decoy prec mz
decoy_mz_offset = 0


### moving average window MS1 trace
smoothing_window = 3

### additional scans beyond MS1 peak found
additional_scans = 0

## min number of scans???


### test
test_var = "test_1"

### filters for picking spectra to fit
frac_lib_matched = args.lib_frac
match_ms1 = args.no_ms1_req
top_n = 10
atleast_m = args.atleast_m




## How to deal with unmatched intensity:
unmatched_fit_type = args.unmatched
"""
3 fit_types:
    a: All summed unmatched intensities are fit to a single zero intensity "obs peak"
    b: Summed unmatched intensities of each precursor are fit to their own zero intensity "obs peak"
    c: Each unmatched peak is fit to its own zero intensity "obs peak"

"""


score_model = "rf"
"""
3 options:
    rf: Random Forest
    lda: Linear Discriminant Analyisis
    xg: XgBoost
"""

tree_max_depth = None

fdr_threshold = 0.01


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
"UniMod:269" : 10.027228
};