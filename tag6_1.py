#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 14:47:51 2025

@author: kevinmcdonnell
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import re

from read_output import names, dtypes
import miscFunctions as mf
from mass_tags import mTRAQ, mTRAQ_02468, mTRAQ_678, tag_library, diethyl_6plex
import config

from FigureFunctions import load_files, plot_channel_numbers, merge_and_norm,\
    compare_channels,plot_jaccard_matrix,get_abs_ratio_errors,plot_venns,\
        plot_channel_numbers2

stop

theoretical_ratios = [65/65,20/5,15/30]
theoretical_yeast_amounts = {"A":6,"B":12,"C":18,"D":24}
# theoretical_human_amounts = {"A":50,"B":50,"C":50,"D":50}
theoretical_human_amounts = {"A":100,"B":100,"C":100,"D":100}

channels = [0,2,4,6,8,10,12,14,16]
channel_mix = ["D","A","B","C","D","C","A","B","C"]
tag_name_tag6 = "plexDIA"#"diethyl_6plex"
channel_mix_key = {i:j for i,j in zip(channels,channel_mix)}

get_dcs = True
get_dcs = False

### load in data 1

fdx_files1 = ["/Volumes/Lab/KMD/Results/2025-03-20_HY_T6d0-2ng_40win-DIA_1_hs_tag6_predlib_JDRT_480_1000_jmodUpdate220325_20ppm_3m_unmatchc_DECOYrev_libfrac0.5_RT_Dino_tag6/filtered_IDs.csv",
                            ]

fdx_files1 = ["/Volumes/Lab/KMD/Results/2025-03-14_MY-HR_LF-DIA_20min_200pg_1_LF_HY_libUpdate220325_20ppm_3m_unmatchc_DECOYrev_libfrac0.5_RT_Dino_iso3.0/filtered_IDs.csv",
                "/Volumes/Lab/KMD/Results/2025-03-20_HY_T6-200pg_5plex_20win-DIA_3_hs_tag6_predlib_jmodUpdate220325_20ppm_3m_unmatchc_DECOYrev_libfrac0.5_RT_Dino_iso3.0_tag6_5plex_plexDIA/filtered_IDs.csv",
                "/Volumes/Lab/KMD/Results/2025-03-20_HY_T6-200pg_9plex_20win-DIA_3_hs_tag6_predlib_jmodUpdate220325_20ppm_3m_unmatchc_DECOYrev_libfrac0.5_RT_Dino_iso3.0_tag6_9plex_plexDIA/filtered_IDs.csv",]


## The following are placeholder files
fdx_files1 = ["/Volumes/Lab/KMD/Results/2025-03-20_HY_T6-200pg_5plex_20win-DIA_3_hs_tag6_predlib_jmodUpdate220325_20ppm_3m_unmatchc_DECOYrev_libfrac0.5_RT_Dino_iso3.0_tag6_5plex_plexDIA/filtered_IDs.csv",
              "/Volumes/Lab/KMD/Results/2025-03-20_HY_T6-200pg_9plex_20win-DIA_3_hs_tag6_predlib_jmodUpdate220325_20ppm_3m_unmatchc_DECOYrev_libfrac0.5_RT_Dino_iso3.0_tag6_9plex_plexDIA/filtered_IDs.csv",
              "/Volumes/Lab/KMD/Results/2025-03-20_HY_T6-200pg_5plex_20win-DIA_1_hs_tag6_predlib_jmodUpdate220325_20ppm_3m_unmatchc_DECOYrev_libfrac0.5_RT_Dino_iso3.0_tag6_5plex_plexDIA/filtered_IDs.csv",
              "/Volumes/Lab/KMD/Results/2025-03-20_HY_T6-200pg_9plex_20win-DIA_1_hs_tag6_predlib_jmodUpdate220325_20ppm_3m_unmatchc_DECOYrev_libfrac0.5_RT_Dino_iso3.0_tag6_9plex_plexDIA/filtered_IDs.csv",
              ]



title1 = "3-plex"
# title1 = "4 Da"
fdx1, dc1 = load_files(fdx_files1)



fdx1["Run"] = [Path(i).stem for i in fdx1.file_name]
# fdx2["Run"] = [Path(i).stem for i in fdx2.file_name]

group_keys = ["5plex","9plex"]
groups = [[i for i in list(np.unique(fdx1.Run)) if key in i] for key in group_keys]

plot_channel_numbers(fdx1, fdx1, title1, title1, channels,summary_type="protei",
                     # groups = [[i] for i in list(np.unique(fdx1.Run))],
                     groups=groups,
                     
                     )

plot_channel_numbers2([fdx1[np.isin(fdx1.Run, group)] for group in groups],
                      titles = [group_keys],
                      channels=channels)

import matplotlib
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#077DEC","#FFFFFF","#EC7607"])

plot_jaccard_matrix(fdx1[np.isin(fdx1.Run, groups[1])], channels, channel_mix,colormap=cmap,alpha=1,min_val=.6)
plt.colorbar()


### ven diagrams from replicates

plot_venns(fdx1,groups)

fdxA = fdx1[np.isin(fdx1.Run, groups[0])]
fdxB = fdx1[np.isin(fdx1.Run, groups[1])]


