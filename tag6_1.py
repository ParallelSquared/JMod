"""
This Source Code Form is subject to the terms of the Oxford Nanopore
Technologies, Ltd. Public License, v. 1.0.  Full licence can be found
at https://github.com/ParallelSquared/JMod/blob/main/LICENSE.txt
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
        plot_channel_numbers2, compute_all_comparisons, errors_boxplot,add_new_quant



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
fdx_files_d0 = ["/Volumes/Lab/KMD/Results/2025-03-20_HY_T6d0-200pg_20win-DIA_1_hs_tag6_shortpep_JDRT_480_1000_250k_jmodUpdate220325_20ppm_3m_unmatchc_DECOYrev_libfrac0.5_RT_Dino_tag6/filtered_IDs.csv",
                "/Volumes/Lab/KMD/Results/2025-03-20_HY_T6d0-200pg_20win-DIA_2_hs_tag6_shortpep_JDRT_480_1000_250k_jmodUpdate220325_20ppm_3m_unmatchc_DECOYrev_libfrac0.5_RT_Dino_tag6/filtered_IDs.csv"]
fdx_files_5p = ["/Volumes/Lab/KMD/Results/2025-03-20_HY_T6-200pg_5plex_20win-DIA_3_hs_tag6_predlib_jmodUpdate220325_20ppm_3m_unmatchc_DECOYrev_libfrac0.5_RT_Dino_iso3.0_tag6_5plex_plexDIA/filtered_IDs.csv",
              "/Volumes/Lab/KMD/Results/2025-03-20_HY_T6-200pg_5plex_20win-DIA_1_hs_tag6_predlib_jmodUpdate220325_20ppm_3m_unmatchc_DECOYrev_libfrac0.5_RT_Dino_iso3.0_tag6_5plex_plexDIA/filtered_IDs.csv"]
fdx_files_9p = ["/Volumes/Lab/KMD/Results/2025-03-20_HY_T6-200pg_9plex_20win-DIA_3_hs_tag6_predlib_jmodUpdate220325_20ppm_3m_unmatchc_DECOYrev_libfrac0.5_RT_Dino_iso3.0_tag6_9plex_plexDIA/filtered_IDs.csv",
                "/Volumes/Lab/KMD/Results/2025-03-20_HY_T6-200pg_9plex_20win-DIA_1_hs_tag6_predlib_jmodUpdate220325_20ppm_3m_unmatchc_DECOYrev_libfrac0.5_RT_Dino_iso3.0_tag6_9plex_plexDIA/filtered_IDs.csv"]              

fdx_files_lf = ["/Volumes/Lab/KMD/Results/2025-03-20_HY_LF-200pg_20win-DIA_3_LF_HY_libUpdate220325_20ppm_3m_unmatchc_DECOYrev_libfrac0.5_RT_Dino_iso3.0/filtered_IDs.csv"]

fdx_files_d0 =["/Volumes/Lab/KMD/Results/2025-03-20_HY_T6d0-200pg_20win-DIA_1_hs_tag6_shortpep_JDRT_480_1000_250k_jmodUpdate220325_20ppm_3m_unmatchc_DECOYrev_libfrac0.5_RT_Dino_tag6/filtered_IDs.csv",
               "/Volumes/Lab/KMD/Results/2025-03-20_HY_T6d0-200pg_20win-DIA_2_hs_tag6_shortpep_JDRT_480_1000_250k_jmodUpdate220325_20ppm_3m_unmatchc_DECOYrev_libfrac0.5_RT_Dino_tag6/filtered_IDs.csv",
               "/Volumes/Lab/KMD/Results/2025-03-20_HY_T6d0-200pg_20win-DIA_3_hs_tag6_shortpep_JDRT_480_1000_250k_jmodUpdate220325_20ppm_3m_unmatchc_DECOYrev_libfrac0.5_RT_Dino_tag6/filtered_IDs.csv",]
fdx_files_5p = ["/Volumes/Lab/KMD/Results/2025-03-20_HY_T6-200pg_5plex_20win-DIA_1_hs_tag6_shortpep_JDRT_480_1000_250k_jmodUpdate220325_20ppm_3m_unmatchc_DECOYrev_libfrac0.5_RT_Dino_iso3.0_tag6_9plex_plexDIA/filtered_IDs.csv",
                # "/Volumes/Lab/KMD/Results/2025-03-20_HY_T6-200pg_5plex_20win-DIA_2_hs_tag6_shortpep_JDRT_480_1000_250k_jmodUpdate220325_20ppm_3m_unmatchc_DECOYrev_libfrac0.5_RT_Dino_iso3.0_tag6_9plex_plexDIA/filtered_IDs.csv",
                "/Volumes/Lab/KMD/Results/2025-03-20_HY_T6-200pg_5plex_20win-DIA_3_hs_tag6_shortpep_JDRT_480_1000_250k_jmodUpdate220325_20ppm_3m_unmatchc_DECOYrev_libfrac0.5_RT_Dino_iso3.0_tag6_9plex_plexDIA/filtered_IDs.csv"]




fdx_files_d0d4 = ["/Volumes/Lab/KMD/Results/2025-03-20_HY_T6-2ng_2plex-d0d2_40win-DIA_1_hs_tag6_predlib_JDRT_480_1000_subset_jmodUpdate130525_10ppm_3m_unmatchc_DECOYrev_libfrac0.5_RT_Dino_iso3.0_tag6_d0d4_plexDIA/filtered_IDs.csv",
                  "/Volumes/Lab/KMD/Results/2025-03-20_HY_T6-2ng_2plex-d0d2_40win-DIA_2_hs_tag6_predlib_JDRT_480_1000_subset_jmodUpdate130525_10ppm_3m_unmatchc_DECOYrev_libfrac0.5_RT_Dino_iso3.0_tag6_d0d4_plexDIA/filtered_IDs.csv",
                  "/Volumes/Lab/KMD/Results/2025-03-20_HY_T6-2ng_2plex-d0d2_40win-DIA_3_hs_tag6_predlib_JDRT_480_1000_subset_jmodUpdate130525_10ppm_3m_unmatchc_DECOYrev_libfrac0.5_RT_Dino_iso3.0_tag6_d0d4_plexDIA/filtered_IDs.csv"
                    ]

fdx_files_d0d2 = ["/Volumes/Lab/KMD/Results/2025-03-20_HY_T6-2ng_2plex-d0d4_40win-DIA_1_hs_tag6_predlib_JDRT_480_1000_subset_jmodUpdate130525_10ppm_3m_unmatchc_DECOYrev_libfrac0.5_RT_Dino_iso3.0_tag6_d0d2_plexDIA/filtered_IDs.csv",
                  "/Volumes/Lab/KMD/Results/2025-03-20_HY_T6-2ng_2plex-d0d4_40win-DIA_2_20250321100702_hs_tag6_predlib_JDRT_480_1000_subset_jmodUpdate130525_10ppm_3m_unmatchc_DECOYrev_libfrac0.5_RT_Dino_iso3.0_tag6_d0d2_plexDIA/filtered_IDs.csv",
                  "/Volumes/Lab/KMD/Results/2025-03-20_HY_T6-2ng_2plex-d0d4_40win-DIA_3_hs_tag6_predlib_JDRT_480_1000_subset_jmodUpdate130525_10ppm_3m_unmatchc_DECOYrev_libfrac0.5_RT_Dino_iso3.0_tag6_d0d2_plexDIA/filtered_IDs.csv"
                  ]


fdx_files_lf = ["/Volumes/Lab/KMD/Results/2025-03-20_HY_LF-200pg_20win-DIA_3_LF_HY_libUpdate130525_10ppm_3m_unmatchc_DECOYrev_libfrac0.5_RT_Dino_iso3.0/filtered_IDs.csv"]

fdx_files_d0 =["/Volumes/Lab/KMD/Results/2025-03-20_HY_T6d0-200pg_20win-DIA_1_hs_tag6_predlib_JDRT_480_1000_subset_jmodUpdate130525_10ppm_3m_unmatchc_DECOYrev_libfrac0.5_RT_Dino_tag6/filtered_IDs.csv",
               "/Volumes/Lab/KMD/Results/2025-03-20_HY_T6d0-200pg_20win-DIA_2_hs_tag6_predlib_JDRT_480_1000_subset_jmodUpdate130525_10ppm_3m_unmatchc_DECOYrev_libfrac0.5_RT_Dino_tag6/filtered_IDs.csv",
               "/Volumes/Lab/KMD/Results/2025-03-20_HY_T6d0-200pg_20win-DIA_3_hs_tag6_predlib_JDRT_480_1000_subset_jmodUpdate130525_10ppm_3m_unmatchc_DECOYrev_libfrac0.5_RT_Dino_tag6/filtered_IDs.csv"
               ]

fdx_files_5p = ["/Volumes/Lab/KMD/Results/2025-03-20_HY_T6-200pg_5plex_20win-DIA_1_hs_tag6_predlib_JDRT_480_1000_subset_jmodUpdate130525_10ppm_3m_unmatchc_DECOYrev_libfrac0.5_RT_Dino_iso3.0_tag6_5plex_plexDIA/filtered_IDs.csv",
                "/Volumes/Lab/KMD/Results/2025-03-20_HY_T6-200pg_5plex_20win-DIA_2_hs_tag6_predlib_JDRT_480_1000_subset_jmodUpdate130525_10ppm_3m_unmatchc_DECOYrev_libfrac0.5_RT_Dino_iso3.0_tag6_5plex_plexDIA/filtered_IDs.csv",
                "/Volumes/Lab/KMD/Results/2025-03-20_HY_T6-200pg_5plex_20win-DIA_3_hs_tag6_predlib_JDRT_480_1000_subset_jmodUpdate130525_10ppm_3m_unmatchc_DECOYrev_libfrac0.5_RT_Dino_iso3.0_tag6_5plex_plexDIA/filtered_IDs.csv"
                ]


fdx_files_9p=["/Volumes/Lab/KMD/Results/2025-03-20_HY_T6-200pg_9plex_20win-DIA_1_hs_tag6_predlib_JDRT_480_1000_subset_jmodUpdate130525_10ppm_3m_unmatchc_DECOYrev_libfrac0.5_RT_Dino_iso3.0_tag6_9plex_plexDIA/filtered_IDs.csv",
              "/Volumes/Lab/KMD/Results/2025-03-20_HY_T6-200pg_9plex_20win-DIA_2_hs_tag6_predlib_JDRT_480_1000_subset_jmodUpdate130525_10ppm_3m_unmatchc_DECOYrev_libfrac0.5_RT_Dino_iso3.0_tag6_9plex_plexDIA/filtered_IDs.csv",
              "/Volumes/Lab/KMD/Results/2025-03-20_HY_T6-200pg_9plex_20win-DIA_3_hs_tag6_predlib_JDRT_480_1000_subset_jmodUpdate130525_10ppm_3m_unmatchc_DECOYrev_libfrac0.5_RT_Dino_iso3.0_tag6_9plex_plexDIA/filtered_IDs.csv"]






title1 = "3-plex"
# title1 = "4 Da"
# fdx1, dc1 = load_files(fdx_files1)
fdxs = [load_files(fdx_files,channels,channel_mix)[0] for fdx_files in [fdx_files_lf,
                                                                        fdx_files_d0,
                                                                     fdx_files_5p,
                                                                     fdx_files_9p]]


fdxs = [load_files(fdx_files,channels,channel_mix)[0] for fdx_files in [fdx_files_d0d4,
                                                                        fdx_files_d0d2]]


# fdxs_orig = fdxs
# fdxs[0].iloc[0].coeff=4.074932
# fdx1["Run"] = [Path(i).stem for i in fdx1.file_name]
# # fdx2["Run"] = [Path(i).stem for i in fdx2.file_name]
# fdx1["mix"] = [channel_mix[channels.index(i)] for i in fdx1.channel]


group_keys = ["LF","$\Delta$0","5plex","9plex"]

group_keys = ["$\Delta$0/$\Delta$4","$\Delta$0/$\Delta$2"]

groups = [[i for i in list(np.unique(fdx.Run))] for fdx in fdxs]

# plot_channel_numbers(fdx1, fdx1, title1, title1, channels,summary_type="protei",
#                      # groups = [[i] for i in list(np.unique(fdx1.Run))],
#                      groups=groups,
                     
#                      )

plot_channel_numbers2(fdxs,
                      titles = group_keys,
                      channels=channels,
                      summary_type="protein",
                      width = .2,
                      offset_size=.22,
                      fig_size=(10,5))
plt.grid()

## single channel
plot_channel_numbers2(fdxs,
                      titles = group_keys,
                      channels=channels,
                      summary_type="protein",
                      width = .7,
                      offset_size=0,
                      fig_size=(5,5),
                      plot_delta=True,
                      conf_lists=[[0,1,0.01]])
# plt.grid()

import matplotlib
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#077DEC","#FFFFFF","#EC7607"])

plot_jaccard_matrix(fdxs[2], channels[::2], channel_mix,colormap=cmap,alpha=1,min_val=.6,j_type="protein")
plt.colorbar(label="Data Completenes")

# plot_jaccard_matrix(fdx2, channels, channel_mix,colormap=cmap,alpha=1,min_val=.6,j_type="protein")

### ven diagrams from replicates

# plot_venns(fdx1,groups)

# fdxA = fdx1[np.isin(fdx1.Run, groups[0])]
# fdxB = fdx1[np.isin(fdx1.Run, groups[1])]



# file_groups1 = [list(np.unique(fdx1.Run))]
# file_names1 = ["4Da"]
# file_groups2 = [list(np.unique(fdx2.Run))]
# file_names2 = ["2da"]


# fdx_merged1 = merge_and_norm(["coeff","plex_Area"], fdx1,file_names1, file_groups1)
# fdx_merged2 = merge_and_norm(["coeff","plex_Area"], fdx2,file_names2, file_groups2)


# fdxs = [fdx1[np.isin(fdx1.Run, group)] for group in groups]

fdxs = [add_new_quant(fdx) for fdx in fdxs]

start_idx = 1
fdxs_merged = [merge_and_norm(["plex_Area","new_plexfitMS1","new_plexfitMS1_p"], fdx,file_names, [file_groups]) for fdx,file_names,file_groups in zip(fdxs[start_idx:],group_keys[start_idx:],groups[start_idx:])]


compute_all_comparisons(fdxs_merged[1],#[fdxs_merged[2].plexfitMS1_p>.8],
                            quant_name="merged_new_plexfitMS1_p",
                           # quant_name="merged_plex_Area",
                           channels=channels[::2],
                           channel_mix=channel_mix,
                           theoretical_yeast_amounts=theoretical_yeast_amounts,
                           plot_type = "allm",
                           # Q_value_limit=.01,
                           # BestChannel_Qvalue_limit=0.005,
                           # at_least_b=3
                           )
plt.grid()



fdx_m = merge_and_norm(["plex_Area","new_plexfitMS1"],fdx4,"9plex",[list(np.unique(fdx4.Run))])
compute_all_comparisons(fdx_m[fdx_m.Qvalue<.01],
                           # quant_name="merged_new_plexfitMS1",
                           quant_name="merged_plex_Area",
                           channels=channels,
                           channel_mix=channel_mix,
                           theoretical_yeast_amounts=theoretical_yeast_amounts,
                           plot_type = "allm",
                           Q_value_limit=.01,
                           # BestChannel_Qvalue_limit=0.005,
                           at_least_b=0
                           )


fdx2 = add_new_quant(fdx2)
fdx2 = merge_and_norm(["plex_Area","new_plexfitMS1"],fdx2,"9plex",[list(np.unique(fdx2.Run))])
compute_all_comparisons(fdx2[fdx2.plexfitMS1_p>.8],
                            # quant_name="coeff",
                             # quant_name="merged_new_plexfitMS1",
                            quant_name="merged_plex_Area",
                           channels=channels,
                           channel_mix=channel_mix,
                           theoretical_yeast_amounts=theoretical_yeast_amounts,
                           plot_type = "allm",
                           Q_value_limit=.01,
                           # BestChannel_Qvalue_limit=0.005,
                           at_least_b=3
                           )

### compare MS1 and MS2 lys Arg for MS1 and MS2
all_temp_dfs = [get_abs_ratio_errors(fdx,
                                     channel_mix_key,"merged_coeff",
                                     plexes=np.unique(fdx["channel"]),
                                     theoretical_yeast_amounts=theoretical_yeast_amounts,
                                     conditions = ["Qvalue","BestChannel","num_b"],
                                     plex_name="channel") for fdx in fdxs_merged]+\
                [get_abs_ratio_errors(fdx,
                                     channel_mix_key,"merged_plex_Area",
                                     plexes=np.unique(fdx["channel"]),
                                     theoretical_yeast_amounts=theoretical_yeast_amounts,
                                     conditions = ["Qvalue","BestChannel","num_b"],
                                     plex_name="channel") for fdx in fdxs_merged]


file_names = ["4Da MS2","2Da MS2","4Da MS1","2Da MS1"]
da = ["4Da","2Da","4Da","2Da"]
# da = ["4Da 3-plex","2Da 6-plex","4Da 3-plex","2Da 6-plex"]
ms = ["MS2","MS2","MS1","MS1"]

errors_boxplot(all_temp_dfs,
                   file_names,da,ms
                   )




#### plot d0d4 vs d0d2
compare_channels(fdxs_merged[2], fdxs_merged[2], 
                 title1=group_keys[0], title2=group_keys[1], 
                 quant_name="merged_new_plexfitMS1", 
                 channels=channels, 
                 channel_mix=channel_mix, 
                 theoretical_yeast_amounts=theoretical_yeast_amounts,
                 scatter=True,
                 Q_value_limit=.01,
                 # fig_dims = [[7,1],[1,2,2]],
                 idx1_1=2,
                 idx1_2=0,
                 idx2_1=1,
                 idx2_2=0,
                 
                 )