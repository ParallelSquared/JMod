"""
This Source Code Form is subject to the terms of the Oxford Nanopore
Technologies, Ltd. Public License, v. 1.0.  Full licence can be found
at https://github.com/ParallelSquared/JMod/blob/main/LICENSE.txt
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
import re

from read_output import names, dtypes
import miscFunctions as mf
from mass_tags import mTRAQ, mTRAQ_02468, mTRAQ_678, tag_library, diethyl_6plex
import config

from FigureFunctions import load_files, plot_channel_numbers, merge_and_norm,compare_channels,plot_jaccard_matrix,get_abs_ratio_errors



theoretical_ratios = [65/65,20/5,15/30]
theoretical_yeast_amounts = {"A":6,"B":12,"C":18,"D":24}
# theoretical_human_amounts = {"A":50,"B":50,"C":50,"D":50}
theoretical_human_amounts = {"A":100,"B":100,"C":100,"D":100}

channels = [0,2,4,6,8,10]
channel_mix = ["A","C","B","D","A","C","D","B"]
tag_name = "plexDIA"#"diethyl_6plex"
channel_mix_key = {i:j for i,j in zip(channels,channel_mix)}
get_dcs = True
get_dcs = False

### load in data 1

fdx_files1 = ["/Volumes/Lab/KMD/Results/2024-08-25_SS_DE-3plex_335ng_1_Spectromine-lib-DE_untagUpdate220325_20ppm_3m_unmatchc_DECOYrev_libfrac0.5_RT_Dino_iso3.0_diethyl_3plex_plexDIA_lastUpdateNoMS1/filtered_IDs.csv",
              "/Volumes/Lab/KMD/Results/2024-08-25_SS_DE-3plex_335ng_2_Spectromine-lib-DE_untagUpdate220325_20ppm_3m_unmatchc_DECOYrev_libfrac0.5_RT_Dino_iso3.0_diethyl_3plex_plexDIA_lastUpdateNoMS1/filtered_IDs.csv",
              "/Volumes/Lab/KMD/Results/2024-08-25_SS_DE-3plex_335ng_3_Spectromine-lib-DE_untagUpdate220325_20ppm_3m_unmatchc_DECOYrev_libfrac0.5_RT_Dino_iso3.0_diethyl_3plex_plexDIA_lastUpdateNoMS1/filtered_IDs.csv",
                            ]

title1 = "3-plex"
# title1 = "4 Da"
fdx1, dc1 = load_files(fdx_files1)




fdx_files2 = ["/Volumes/Lab/KMD/Results/2024-08-25_SS_DE-6plex_1000ng_1_Spectromine-lib-DE_untagUpdate220325_20ppm_3m_unmatchc_DECOYrev_libfrac0.5_RT_Dino_iso3.0_diethyl_6plex_plexDIA_lastUpdateNoMS1/filtered_IDs.csv",
              "/Volumes/Lab/KMD/Results/2024-08-25_SS_DE-6plex_1000ng_2_Spectromine-lib-DE_untagUpdate220325_20ppm_3m_unmatchc_DECOYrev_libfrac0.5_RT_Dino_iso3.0_diethyl_6plex_plexDIA_lastUpdateNoMS1/filtered_IDs.csv",
              "/Volumes/Lab/KMD/Results/2024-08-25_SS_DE-6plex_1000ng_3_Spectromine-lib-DE_untagUpdate220325_20ppm_3m_unmatchc_DECOYrev_libfrac0.5_RT_Dino_iso3.0_diethyl_6plex_plexDIA_lastUpdateNoMS1/filtered_IDs.csv"]


title2 = "6-plex"
# title2 = "2 Da"


fdx2, dc2 = load_files(fdx_files2)


from pathlib import Path
fdx1["Run"] = [Path(i).stem for i in fdx1.file_name]
fdx2["Run"] = [Path(i).stem for i in fdx2.file_name]


## tag6 2da vs 4 da
file_groups1 = [list(np.unique(fdx1.Run))]
file_names1 = ["4Da"]
file_groups2 = [list(np.unique(fdx2.Run))]
file_names2 = ["2da"]


fdx_merged1 = merge_and_norm(["coeff","plex_Area"], fdx1,file_names1, file_groups1)
fdx_merged2 = merge_and_norm(["coeff","plex_Area"], fdx2,file_names2, file_groups2)


plot_channel_numbers(fdx1, fdx2, title1, title2, channels,summary_type="pr")


compare_channels(fdx_merged1,
                    fdx_merged2,
                    title1,
                    title2,
                    "merged_coeff",
                    channels,
                    channel_mix,
                    theoretical_yeast_amounts,
                    idx1_1=0,
                    idx1_2=2,
                    idx2_1 = 0,
                    idx2_2 = 2,
                    scatter = False,box=True)



plot_jaccard_matrix(fdx1, channels[::2], channel_mix,colormap="Wistia")



### compare MS1 and MS2 lys Arg for MS1 and MS2
all_temp_dfs = [get_abs_ratio_errors(fdx,
                                     channel_mix_key,"merged_coeff",
                                     plexes=np.unique(fdx["channel"]),
                                     theoretical_yeast_amounts=theoretical_yeast_amounts,
                                     conditions = ["Qvalue","BestChannel","num_b"],
                                     plex_name="channel") for fdx in [fdx_merged1,fdx_merged2]]+\
                [get_abs_ratio_errors(fdx,
                                     channel_mix_key,"merged_plex_Area",
                                     plexes=np.unique(fdx["channel"]),
                                     theoretical_yeast_amounts=theoretical_yeast_amounts,
                                     conditions = ["Qvalue","BestChannel","num_b"],
                                     plex_name="channel") for fdx in [fdx_merged1,fdx_merged2]]


file_names = ["4Da MS2","2Da MS2","4Da MS1","2Da MS1"]
da = ["4Da","2Da","4Da","2Da"]
da = ["4Da 3-plex","2Da 6-plex","4Da 3-plex","2Da 6-plex"]
ms = ["MS2","MS2","MS1","MS1"]