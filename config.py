import argparse



parser = argparse.ArgumentParser(
                    prog='Jmod',
                    description='What the program does',
                    epilog='Text at the bottom of help')

parser.add_argument('-i','--mzml')
parser.add_argument('-d','--diaPASEF')
parser.add_argument('-l','--speclib')      
parser.add_argument('-r', '--use_rt', action='store_true')
parser.add_argument('-f', '--use_features', action='store_true')
parser.add_argument('-m', '--atleast_m', default=3, type=int)
parser.add_argument('-t', '--threads', default=10, type=int)
parser.add_argument('-p', '--ppm', default=20, type=float)
parser.add_argument('--ms2_align', action='store_true')
parser.add_argument('--timeplex', action='store_true')
parser.add_argument('--iso', action='store_true')
parser.add_argument('--unmatched', default="a", type=str)
parser.add_argument('--decoy', default="rev", type=str)
parser.add_argument('--mTRAQ', action='store_true')
parser.add_argument('--tag', default="", type=str)
parser.add_argument('--num_iso', default=2, type=float)
parser.add_argument('--pp_file', default="", type=str)


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
ms1_ppm = 10
ms1_tol = ms1_ppm*10**(-6)

num_timeplex = None

rt_tol = 3
im_tol = 0.5
im_merge_tol = 0.005
rt_width = 1.5

top_n = 10
atleast_m = args.atleast_m

top_n_pasef = 10
atleast_m_pasef = 1

max_window_offset = None

numProc = args.threads


chunksize = 10
batch_size = 1000


# rt_filtering = False
# feature_filtering = False

# How many spectra to fit RT alignment
n_most_intense = 400
n_most_intense_features = 10000

opt_rt_tol = rt_tol # set as default to start
opt_ms1_tol = ms1_tol 
opt_im_tol = im_tol



num_iso_peaks = args.num_iso
min_iso_intensity = 1e-3 ## not derived meeaningfully. Just looked at the minimum value from a library

decoy_mz_offset = 20


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

fdr_threshold = 0.01


#############################################
##### This causes a circular import

# #### tags here

# from mass_tags import massTag
# mTRAQ = massTag(rules = "nK",
#             base_mass=140.0949630177,
#             delta = 4.0070994,
#             channel_names = ["0","4","8"],
#             name = "mTRAQ")

# if args.mTRAQ:
#     tag = mTRAQ

# else: 
#     tag = None

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
