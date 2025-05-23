"""
This Source Code Form is subject to the terms of the Oxford Nanopore
Technologies, Ltd. Public License, v. 1.0.  Full licence can be found
at https://github.com/ParallelSquared/JMod/blob/main/LICENSE.txt
"""

from fdr_analysis import process_data
import os
import re
from load_files import loadSpectra
from SpecLib import loadSpecLib
from mass_tags import tag_library, mTRAQ, mTRAQ_02468, mTRAQ_678, diethyl_6plex,available_tags
import config
import iso_functions as iso_f
import argparse


def run():
    
    # parser = argparse.ArgumentParser()
    
    # parser.add_argument("file")
    # args = vars(parser.parse_args())
    
    file = config.args.pp_file
    
    # file = "/Users/kevinmcdonnell/Programming/Data/Results/JD0081_DiannPlexHYE.mspUpdate060824_20ppm_3m_unmatchc_DECOYrev_RT_iso5.0_timeplex/decoylibsearch_coeffs.csv"
    
    results_folder = os.path.dirname(file)
    
    params = {}
    with open(results_folder+"/params.txt","r") as read_file:
        
        line = read_file.readline()
        while line:
            if ": " in line:
                if re.match("tag: TagName",line):
                    line = re.sub("tag: ","",line)
                i,j = re.split(": ",line)
                params[i]=j.strip()
            line = read_file.readline()
            
    spec_file = params["mzml"]
    
    # spec_file = "/Users/kevinmcdonnell/Programming/Data/9plex/2024-03-21_Sciex_3-plex_678_100ng.mzML"
    # spec_file = "/Volumes/Lab/KMD/Data/9plex/2024-03-21_Sciex_9-plex_LF_A_100ng.mzML"
    
    
    DIAspectra = loadSpectra(spec_file)
    
    lib_file = params["speclib"]
    # lib_file = "/Volumes/Lab/KMD/Data/mTRAQ_Bulk/Diann/Non_red_alk_500ng_v2_40_2x_orig_E480_RAW/NEUlibSearchLibrary_PrositFrags.tsv"
    library = loadSpecLib(lib_file)
    
    if "TagName" in params:
        if  params["TagName"] in available_tags:
            mass_tag = available_tags[params["TagName"]]
        else:
            raise ValueError("Unknown Tag")
        config.tag = mass_tag
        
        # library = tag_library(library,mass_tag)
        
    else:
        # raise ValueError
        mass_tag = None
    
    
    
    if params["iso"]:
        config.num_iso_peaks = int(float(params["num_iso_peaks"]))
        # library = iso_f.iso_library(library)
    
    unmatched_type = re.search("unmatch([abc])",file.split("/")[-2])[1].upper()
    
    
    process_data(file=file, 
                 spectra=DIAspectra, 
                 library=library, 
                 mass_tag=mass_tag,
                 timeplex=eval(params["timeplex"]))




if __name__=="__main__":
    
    run()