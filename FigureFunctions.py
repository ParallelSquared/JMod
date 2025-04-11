#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 17:24:06 2025

@author: kevinmcdonnell
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import gridspec
import os
import re
from itertools import combinations,combinations_with_replacement, product
from read_output import names, dtypes
import miscFunctions as mf
from mass_tags import mTRAQ, mTRAQ_02468, mTRAQ_678, tag_library, diethyl_6plex
import config
from scipy.stats import gaussian_kde

from upsetplot import generate_counts
from upsetplot import plot
from upsetplot import from_contents
from upsetplot import UpSet
from matplotlib_venn import venn2, venn3
from sklearn.metrics import auc
from pathlib import Path

from functools import reduce

import seaborn as sb

import tqdm


logic_and = np.logical_and.reduce
def intersect_all(*arrs): return reduce(np.intersect1d,arrs)



colours = ["tab:blue","tab:orange","tab:green","tab:red",
'tab:purple',
'tab:brown',
'tab:pink',
'tab:gray',
'tab:olive',
'tab:cyan']

plt.rcParams['figure.dpi'] = 500
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['axes.titlesize'] = 15


from loadFasta import all_fasta_seqs, all_protein_names
# all_fasta_seqs = all_fasta_seqs

labels = ["Human",
          "Ecoli",
          "Yeast"]

orgs =np.array(['Human', 'Ecoli', 'Yeast'])
    
org_colors = ["tab:blue","tab:orange","tab:green"]
org_colors = ["tab:blue","tab:orange","tab:red"]


def load_files(fdx_files1,channels,channel_mix,get_dcs=False):
    
    fdxs = []
    dcs = []
    for idx,fdx_file in enumerate(fdx_files1):
        print(f"file {idx+1} of {len(fdx_files1)}")
        fdx_n = pd.read_csv(fdx_file)
        if "channel" not in fdx_n.columns:
            fdx_n["channel"] = idx
        fdx_n["file_idx"] = idx
        fdxs.append(fdx_n)
        
        if get_dcs:
            dc_path = os.path.dirname(fdx_file)+"/decoylibsearch_coeffs.csv"
            dc_n = pd.read_csv(dc_path,header=None,names=names,dtype=dtypes)
            if "channel" not in dc_n.columns:
                dc_n["channel"] = idx
            dc_n["file_idx"] = idx
            dcs.append(dc_n)
    
    
    ### Assume they are all of the same type (Tag)
    params = {}
    with open(os.path.dirname(fdx_file)+"/params.txt","r") as read_file:
        
        line = read_file.readline()
        while line:
            if ": " in line:
                if re.match("tag: TagName",line):
                    line = re.sub("tag: ","",line)
                i,j = re.split(": ",line)
                params[i]=j.strip()
            line = read_file.readline()
    
    if "TagName" in params:
        if params["TagName"]=="mTRAQ_678":
            mass_tag1 = mTRAQ_678
            tag_name1 = "mTRAQ"
            
        elif params["TagName"]=="mTRAQ_02468":
            mass_tag1 = mTRAQ_02468
            tag_name1 = "mTRAQ"
            
        elif params["TagName"]=="mTRAQ":
            mass_tag1 = mTRAQ
            tag_name1 = "mTRAQ"
        
        elif params["TagName"]=="diethyl_6plex":
            mass_tag1 = diethyl_6plex
            tag_name1 = "DE"
        
        tag_name1 = "plexDIA"
    else:
        # raise ValueError
        mass_tag1 = None
        tag_name1 = "LF"
    
    fdx1 = pd.concat(fdxs,ignore_index=True)
    dc1 = None
    if get_dcs:
        dc1 = pd.concat(dcs,ignore_index=True)
    # fdx1["tag"] = tag_name1
    
    fdx1["org"] = np.array([";".join(orgs[[i in all_fasta_seqs[j] for j in range(3)]]) for i in fdx1["stripped_seq"]])
    fdx1["Run"] = [Path(i).stem for i in fdx1.file_name]
    # fdx2["Run"] = [Path(i).stem for i in fdx2.file_name]
    fdx1["mix"] = [channel_mix[channels.index(i)] for i in fdx1.channel]
    if "num_b" not in fdx1.columns:
        fdx1["num_b"] = fdx1.frag_names.str.count("b")
    
    return fdx1,dc1



def compute_new_quant(group,scan_width=3,use="max"):
    if "plexfittrace_spec_all" not in group.columns:
        return np.zeros(len(group))
    ## just take the first as they should be the same
    spectra =  np.array([*map(float,group.plexfittrace_spec_all.iloc[0].split(";"))])
    ## below works for np.isnan but will not generalize
    pearson_stats =  np.array([float(i) if "n" not in i else np.nan for i in group.plexfittrace_ps_all.iloc[0].split(";")])
    ms1_plex = np.sum(np.stack([np.array([float(i) if "n" not in i else np.nan for i in vals.split(";")]) for vals in group.plexfittrace_all],1),1)
    
    if np.all(np.isnan(pearson_stats)):
        return group.plexfitMS1
    elif use=="pearson":
        pearson_stats[np.isnan(pearson_stats)]=0
        largest_stat_idx =  np.nanargmax(mf.moving_average(pearson_stats,3))
    else:
        ms1_plex[np.isnan(ms1_plex)]=0
        largest_stat_idx =  np.nanargmax(mf.moving_average(ms1_plex,3))
        
    # return group.plexfittrace_all.transform(lambda x: float(x.split(";")[largest_stat_idx]))
    return group.plexfittrace_all.transform(lambda x: np.sum(list(map(float,x.split(";")[max(0,largest_stat_idx-scan_width):min(len(pearson_stats),largest_stat_idx+scan_width)]))))
    
def add_new_quant(fdx1):
    new_quant_column = fdx1.groupby(["file_idx","untag_prec"],as_index=False).apply(lambda x: compute_new_quant(x))
    fdx1["new_plexfitMS1"] = new_quant_column.reset_index(level=0, drop=True)
    return fdx1


def get_counts(fdx1,
               summary_type = "protein",
                at_least_b = 0,
                Q_value_limit = 0.05,
                BestChannel_Qvalue_limit = 0.01):
    
    if summary_type == "protein":
        return fdx1[np.logical_and.reduce([fdx1.BestChannel_Qvalue<BestChannel_Qvalue_limit,
                                            fdx1.Protein_Qvalue<.01,
                                            fdx1.Qvalue<Q_value_limit
                                            ])].groupby(["Run","channel"]).agg({'protein': pd.Series.nunique})["protein"]
        
        
    else:
        return fdx1[fdx1["Qvalue"]<Q_value_limit].groupby(["Run","channel"])["untag_prec"].count()
    
        
    
def plot_channel_numbers(fdx1,fdx2,
                         title1,
                         title2,
                         channels,
                         groups = None,
                         summary_type = "protein",
                         at_least_b = 0,
                         Q_value_limit = 0.05,
                         BestChannel_Qvalue_limit = 0.01):
    
    counts = get_counts(fdx1,summary_type,at_least_b,Q_value_limit,BestChannel_Qvalue_limit)
    counts_d = get_counts(fdx1,summary_type,at_least_b,Q_value_limit,)
    print("test3")
    
    
    
    # file_groups = [[0,4],[3,7],[2,6],[1,5]]
    if groups:
        # file_groups = [["DE_benchmark_008","DE_benchmark_009","DE_benchmark_011"],
        #                ["DE_benchmark_007","DE_benchmark_010","DE_benchmark_012"],
        #                ["DE_benchmark_002","DE_benchmark_003","DE_benchmark_005"],
        #                ["DE_benchmark_001","DE_benchmark_004","DE_benchmark_006"],
        #                ]
        # file_names = ["14Win","8Win","4Win","2Win"]
        file_groups = groups
        file_names = [f"G{i}" for i in range(len(file_groups))]
        width = 0.6
    else:
        # file_groups = [["2024-08-25_SS_DE-3plex_335ng_1","2024-08-25_SS_DE-3plex_335ng_2","2024-08-25_SS_DE-3plex_335ng_3"]]
        # # file_groups = [["2024-08-25_SS_DE-3plex_335ng_1"]]
        # group2 = ["2024-08-25_SS_DE-6plex_1000ng_1","2024-08-25_SS_DE-6plex_1000ng_2","2024-08-25_SS_DE-6plex_1000ng_3"]
        # file_names = ["Bulk"]
        # width = 0.4
        
        file_groups = [list(np.unique(fdx1.Run))]
        group2 = list(np.unique(fdx2.Run))
        file_names2 = ["2da"]
        file_names = ["Bulk"]
    
    
    channel_colours = colours
    # channel_colours = ["black","grey","darkgrey","lightgrey","whitesmoke","white"]
    alpha=.5
    vals = []
    d_vals = []
    fig, ax = plt.subplots(figsize=(5,5))
    for idx,group in enumerate(file_groups):
        # break
        file_stack = np.stack([counts.loc[i] for i in group if i in counts.index],0).reshape(len(set(counts[group].index.get_level_values("channel"))),sum([i in counts.index for i in group]))
        means = np.mean(file_stack,1)
        std_devs = np.std(file_stack,1)
        vals.append(means)
        ### below for across windows
        if groups:
            plt.bar(np.array([idx]*(len(set(counts[group].index.get_level_values("channel"))))),
                    means,
                    width=width,
                    bottom=np.append([0],np.cumsum(means)[:-1]),
                    yerr=std_devs,
                    color=np.array(channel_colours)[np.array(counts[group].index.get_level_values("channel")/2,dtype=int)],
                    alpha=alpha)
            # [plt.text(i,j,str(int(k)),horizontalalignment='center',verticalalignment='center') for i,j,k in zip(np.array([idx]*len(set(counts.index.get_level_values("channel")))),
            #                                     np.append([0],np.cumsum(means)[:-1])+np.array(means)/2,
            #                                     means)]
        
        ### below for single file
        else:
            plt.bar([0]*len(set(counts.index.get_level_values("channel")))
                    ,means,
                    width=width,
                    bottom=np.append([0],np.cumsum(means)[:-1]),
                    yerr=std_devs,
                    color=[channel_colours[c//2] for c in list(counts.index.get_level_values("channel"))],
                    alpha=alpha,edgecolor="black")
            [plt.text(i,j,str(int(k)),horizontalalignment='center',verticalalignment='center') for i,j,k in zip([0]*len(set(counts.index.get_level_values("channel"))),
                                            np.append([0],np.cumsum(means)[:-1])+np.array(means)/2,
                                            means)]
        if groups:
            continue
            file_stack = np.stack([counts_d.loc[i] for i in group if i in counts_d.index],0).reshape(len(set(counts_d.index.get_level_values("channel"))),sum([i in counts_d.index for i in group]))
        else:
            file_stack = np.stack([counts_d.loc[i] for i in group2 if i in counts_d.index],0).reshape(len(set(counts_d.index.get_level_values("channel"))),sum([i in counts_d.index for i in group2]))
        
        means = np.mean(file_stack,1)
        std_devs = np.std(file_stack,1)
        d_vals.append(means)
        ### below for across windows
        if groups:
            plt.bar(np.array([idx]*len(set(counts.index.get_level_values("channel"))))+ width/1.9,means,width=width,bottom=np.append([0],np.cumsum(means)[:-1]),yerr=std_devs,color=colours[:len(set(counts.index.get_level_values("channel")))],alpha=alpha)
            
        ### below for single file
        else:
            plt.bar([1]*len(set(counts_d.index.get_level_values("channel"))),
                    means,width=width,
                    bottom=np.append([0],np.cumsum(means)[:-1]),
                    yerr=std_devs,
                    color=[channel_colours[c//2] for c in list(counts_d.index.get_level_values("channel"))],
                    alpha=alpha,edgecolor="black")
            [plt.text(i,j,str(int(k)),horizontalalignment='center',verticalalignment='center') for i,j,k in zip([1]*len(set(counts_d.index.get_level_values("channel"))),
                                             np.append([0],np.cumsum(means)[:-1])+np.array(means)/2,
                                             means)]
    
    # plt.xticks(range(idx+1),file_names)
    # xtick_labels = [f"  Jmod | DIANN\n{cat}" for cat in file_names]
    xtick_labels = file_names
    if groups:
        plt.xticks(range(len(file_names)),xtick_labels)
    
    ## for single file
    else:
        plt.xticks([0,1],[title1,title2])
        # plt.xticks([0,1],["3-plex","6-plex"])
    
    ax.spines[['right', 'top']].set_visible(False)
    if summary_type == "protein":
        plt.ylabel("Number of Protein Data Points")
    else:
        plt.ylabel("Number of Precursors")
        
    
    import matplotlib.patches as mpatches
    
    legend_patches = [
        mpatches.Patch(facecolor=channel_colours[i],edgecolor="black", label=f"$\Delta${channels[i]}",alpha=alpha) for i in range(len(channels))]
    
    # Add the custom legend
    if groups:
        loc="upper right"
    else:
        loc="upper left"
    ax.legend(handles=legend_patches, title="Channel", loc=loc)
    
    # plt.yticks(np.linspace(0,8000,11),np.linspace(0,8000,11))

def descending_greyscale(n):
    return [(i/255, i/255, i/255) for i in range(255, -1, -255//(n-1))][:n]

def descending_greyscale_hex(n):
    return [f'#{i:02x}{i:02x}{i:02x}' for i in range(255, -1, -255//(n-1))][:n]

# Example usage: get 10 descending grey colors
# greys = descending_greyscale(10)
# print(greys)

def plot_channel_numbers2(fdxs,
                         titles,
                         channels,
                         groups = None,
                         summary_type = "protein",
                         at_least_b = 0,
                         Q_value_limit = 0.05,
                         BestChannel_Qvalue_limit = 0.01,
                         width = 0.1,
                         offset_size = .11,
                         fig_size=(5,5)):

    # fdxA = fdx1[np.isin(fdx1.Run, groups[0])]
    # fdxB = fdx1[np.isin(fdx1.Run, groups[1])]

    

    # file_groups1 = [list(np.unique(fdxA.Run))]
    # file_groups2 = [list(np.unique(fdxB.Run))]
    # file_names1 = ["2da"]
    # file_names2 = ["Bulk"]
 
    # counts1 = get_counts(fdxA,summary_type,at_least_b,Q_value_limit,BestChannel_Qvalue_limit)
    # counts2 = get_counts(fdxB,summary_type,at_least_b,Q_value_limit,BestChannel_Qvalue_limit)
    
    
    offset = -offset_size
    fig, ax = plt.subplots(figsize=fig_size)
    for b,q,b in [[0,1,0.01],
                      [0,0.05,0.01],
                      [0,0.01,0.01]]:
        # counts1 = get_counts(fdxA,summary_type,b,q,b)
        # counts2 = get_counts(fdxB,summary_type,b,q,b)
        
        # print("test3")
        
        
        # all_counts = [counts1,counts2]
        # all_files = [file_groups1,file_groups2]
        all_counts = [get_counts(fdx,summary_type,b,q,b) for fdx in fdxs]
        all_files = [[list(np.unique(fdx.Run))] for fdx in fdxs]
        channel_colours = colours
        channel_colours = descending_greyscale(10)
        # channel_colours = ["black","grey","darkgrey","lightgrey","whitesmoke","white"]
        alpha=.5
        vals = []
        d_vals = []
        
        pos = 0
        for counts,file_groups in zip(all_counts,all_files):
            for idx,group in enumerate(file_groups):
                # break
                file_stack = np.stack([counts.loc[i] for i in group if i in counts.index],0).reshape(len(set(counts[group].index.get_level_values("channel"))),sum([i in counts.index for i in group]))
                means = np.mean(file_stack,1)
                std_devs = np.std(file_stack,1)
                vals.append(means)
                ### below for across windows
                
                plt.bar(np.array([idx]*(len(set(counts[group].index.get_level_values("channel")))))+pos+offset,
                        means,
                        width=width,
                        bottom=np.append([0],np.cumsum(means)[:-1]),
                        yerr=std_devs,
                        color=[tuple(i) for i in np.array(channel_colours)[np.array(sorted(set(counts[group].index.get_level_values("channel")/2)),dtype=int)]],
                        edgecolor="black",
                        alpha=alpha)
            pos+=1
        offset+=offset_size
    ax.set_xticks(range(len(all_counts)),titles)
    ax.spines[['right', 'top']].set_visible(False)
    if summary_type == "protein":
        plt.ylabel("Number of Protein Data Points")
    else:
        plt.ylabel("Number of Precursors")
    
    
        
def merge_and_norm(orig_quant_name,fdx,file_names,file_groups,plex_filter=-np.inf):
    merged_data = []
    for idx,(win,files) in enumerate(zip(file_names,file_groups)):
        # break
        quant_greater_zero = (fdx[orig_quant_name]>0).to_numpy() if type(orig_quant_name)==str else np.logical_and.reduce([(fdx[qn]>0).to_numpy() for qn in orig_quant_name])
        subset = fdx[np.logical_and.reduce([fdx['Run'].isin(files).to_numpy(),(fdx["plexfitMS1_p"]>plex_filter).to_numpy(),quant_greater_zero])].copy()  # Filter by file group
        subset['WindowNum'] = win  # Add group name
        subset["file_idx"] = idx
        if type(orig_quant_name)==str:
            quant_names = [orig_quant_name]
        else:
            quant_names = orig_quant_name
        for qn in quant_names:
            subset["channel_median_"+qn] = subset.groupby(["file_idx","channel"])[qn].transform(lambda x:  np.power(2,np.nanmedian(np.log2(x))))
            subset["Normalized_"+qn] = subset.groupby(["file_idx","channel"])[qn].transform(lambda x:np.log2(x) - np.nanmedian(np.log2(x)))
            subset["merged_"+qn] = subset.groupby(["untag_prec","channel"])["Normalized_"+qn].transform(lambda x: np.median(x))
        subset = subset.drop_duplicates(["untag_prec","channel"],keep="last").reset_index(drop=True)
        # median_values = subset.groupby(["untag_prec","channel","last_aa","org"])[orig_quant_name].median().reset_index()
        
        merged_data.append(subset)
        
    return pd.concat(merged_data)


def get_intersect(  fdxA,
                    fdxB,
                    quant_name,
                    conditions = [],
                    file_idx1 = None,
                    file_idx2 = None,
                    channels=None,
                    mix=None,
                    at_least_b = 3,
                    Q_value_limit = 0.05,
                    BestChannel_Qvalue_limit = 0.01,
                    last_aa = "R"):
    
    if channels is not None:
        channelA,channelB = channels
    elif mix is not None:
        mixA,mixB = mix
    else:
        raise ValueError("No channel specified")
    
    intersect1 = intersect_all(fdxA["untag_prec"][logic_and([fdxA["channel"]==channelA if channels is not None else np.ones_like(fdxA["channel"],dtype=bool),
                                                             fdxA["mix"]==mixA if mix is not None else np.ones_like(fdxA["channel"],dtype=bool),
                                                               fdxA["file_idx"]==file_idx1 if file_idx1 is not None else np.ones_like(fdxA["channel"],dtype=bool),
                                                                fdxA["plexfitMS1_p"]>.75 if "pearson_fit" in conditions and "plexfitMS1_p" in fdxA.columns else np.ones_like(fdxA["channel"],dtype=bool),
                                                                fdxA["last_aa"]==last_aa if "last_aa" in conditions else np.ones_like(fdxA["channel"],dtype=bool),
                                                                fdxA.Qvalue<Q_value_limit if "Qvalue" in conditions else np.ones_like(fdxA["channel"],dtype=bool),
                                                                fdxA.BestChannel_Qvalue<BestChannel_Qvalue_limit if "BestChannel" in conditions else np.ones_like(fdxA["channel"],dtype=bool),
                                                                fdxA.num_b>=at_least_b if "num_b" in conditions else np.ones_like(fdxA["channel"],dtype=bool),
                                                                fdxA[quant_name]>0
                                                                
                                                              ])],
                                 fdxB["untag_prec"][logic_and([fdxB["channel"]==channelB if channels is not None else np.ones_like(fdxA["channel"],dtype=bool),
                                                               fdxB["mix"]==mixA if mix is not None else np.ones_like(fdxB["channel"],dtype=bool),
                                                               fdxB["file_idx"]==file_idx2 if file_idx2 is not None else np.ones_like(fdxB["channel"],dtype=bool),
                                                                fdxB["plexfitMS1_p"]>.75 if "pearson_fit" in conditions  and "plexfitMS1_p" in fdxB.columns else np.ones_like(fdxB["channel"],dtype=bool),
                                                                fdxB["last_aa"]==last_aa if "last_aa" in conditions else np.ones_like(fdxB["channel"],dtype=bool),
                                                                fdxB.Qvalue<Q_value_limit if "Qvalue" in conditions else np.ones_like(fdxB["channel"],dtype=bool),
                                                                fdxB.BestChannel_Qvalue<BestChannel_Qvalue_limit if "BestChannel" in conditions else np.ones_like(fdxB["channel"],dtype=bool),
                                                                fdxB.num_b>=at_least_b if "num_b" in conditions else np.ones_like(fdxB["channel"],dtype=bool),
                                                                fdxB[quant_name]>0
                                                              ])])
    
    return intersect1




def compare_channels(fdxA,
                    fdxB,
                    title1,
                    title2,
                    quant_name,
                    channels,
                    channel_mix,
                    theoretical_yeast_amounts,
                    file_idx1_1=0,
                    file_idx1_2=0,
                    idx1_1=0,
                    idx1_2=2,
                    
                    file_idx2_1=0,
                    file_idx2_2=0,
                    idx2_1 = 0,
                    idx2_2 = 1,
                    sub_title1 = "",
                    sub_title2 = "",
                    scatter = True,
                    box=False,
                    last_aa = "R",
                    conditions = [
                                    # "pearson_fit",
                                    # "last_aa",
                                    "Qvalue",
                                    "BestChannel",
                                    "num_b"
                                  ],
                    at_least_b = 3,
                    Q_value_limit = 0.05,
                    BestChannel_Qvalue_limit = 0.01,
                    ):

    long_title1 = title1+sub_title1
    long_title2 = title2+sub_title2
    
    
    logic_and = np.logical_and.reduce
    def intersect_all(*arrs): return reduce(np.intersect1d,arrs)
    
    
    
    
    
   
    # dummy_bool = np.ones_like(fdx["channel"],dtype=bool)
    
    intersect1 = intersect_all(fdxA["untag_prec"][logic_and([fdxA["channel"]==channels[idx1_1],
                                                               fdxA["file_idx"]==file_idx1_1,
                                                                fdxA["plexfitMS1_p"]>.75 if "pearson_fit" in conditions and "plexfitMS1_p" in fdxA.columns else np.ones_like(fdxA["channel"],dtype=bool),
                                                                fdxA["last_aa"]==last_aa if "last_aa" in conditions else np.ones_like(fdxA["channel"],dtype=bool),
                                                                fdxA.Qvalue<Q_value_limit if "Qvalue" in conditions else np.ones_like(fdxA["channel"],dtype=bool),
                                                                fdxA.BestChannel_Qvalue<BestChannel_Qvalue_limit if "BestChannel" in conditions else np.ones_like(fdxA["channel"],dtype=bool),
                                                                fdxA.num_b>=at_least_b if "num_b" in conditions else np.ones_like(fdxA["channel"],dtype=bool),
                                                                
                                                                # ["C" not in i or "M" not in i for i in fdxA.untag_seq],
                                                                # ["C"  in i for i in fdxA.untag_seq],
                                                                # np.log10(fdxA.coeff)>4
                                                                # np.abs(fdxA.rt_error)<.5
                                                                # fdxA.prop==1,
                                                              ])],
                                 fdxA["untag_prec"][logic_and([fdxA["channel"]==channels[idx1_2],
                                                               fdxA["file_idx"]==file_idx1_2,
                                                                fdxA["plexfitMS1_p"]>.75 if "pearson_fit" in conditions  and "plexfitMS1_p" in fdxA.columns else np.ones_like(fdxA["channel"],dtype=bool),
                                                                fdxA["last_aa"]==last_aa if "last_aa" in conditions else np.ones_like(fdxA["channel"],dtype=bool),
                                                                fdxA.Qvalue<Q_value_limit if "Qvalue" in conditions else np.ones_like(fdxA["channel"],dtype=bool),
                                                                fdxA.BestChannel_Qvalue<BestChannel_Qvalue_limit if "BestChannel" in conditions else np.ones_like(fdxA["channel"],dtype=bool),
                                                                fdxA.num_b>=at_least_b if "num_b" in conditions else np.ones_like(fdxA["channel"],dtype=bool),
                                                                
                                                                # ["C" not in i or "M" not in i for i in fdxA.untag_seq],
                                                                # ["C"  in i for i in fdxA.untag_seq],
                                                                # np.log10(fdxA.coeff)>4
                                                                # np.log10(fdxA.coeff)>4
                                                                # np.abs(fdxA.rt_error)<.5
                                                                # fdxA.prop==1,
                                                              ])])
    
    intersect2 = intersect_all(fdxB["untag_prec"][logic_and([fdxB["channel"]==channels[idx2_1],
                                                            fdxB["file_idx"]==file_idx2_1,
                                                              fdxB["plexfitMS1_p"]>.75 if "pearson_fit" in conditions and "plexfitMS1_p" in fdxB.columns else np.ones_like(fdxB["channel"],dtype=bool),
                                                              fdxB["last_aa"]==last_aa if "last_aa" in conditions else np.ones_like(fdxB["channel"],dtype=bool),
                                                               fdxB.Qvalue<Q_value_limit if "Qvalue" in conditions else np.ones_like(fdxB["channel"],dtype=bool),
                                                               fdxB.BestChannel_Qvalue<BestChannel_Qvalue_limit if "BestChannel" in conditions else np.ones_like(fdxB["channel"],dtype=bool),
                                                               fdxB.num_b>=at_least_b if "num_b" in conditions else np.ones_like(fdxB["channel"],dtype=bool),
                                                               
                                                               # ["C" not in i or "M" not in i for i in fdxB.untag_seq],
                                                               # ["C"  in i for i in fdxB.untag_seq],
                                                               # np.log10(fdxA.coeff)>4
                                                               # np.log10(fdxB.coeff)>4
                                                               # np.abs(fdxB.rt_error)<.5
                                                              # fdxB.prop==1,
                                                            ])],
                                fdxB["untag_prec"][logic_and([fdxB["channel"]==channels[idx2_2],
                                                             fdxB["file_idx"]==file_idx2_2,
                                                               fdxB["plexfitMS1_p"]>.75 if "pearson_fit" in conditions and "plexfitMS1_p" in fdxB.columns else np.ones_like(fdxB["channel"],dtype=bool),
                                                               fdxB["last_aa"]==last_aa if "last_aa" in conditions else np.ones_like(fdxB["channel"],dtype=bool),
                                                                fdxB.Qvalue<Q_value_limit if "Qvalue" in conditions else np.ones_like(fdxB["channel"],dtype=bool),
                                                                fdxB.BestChannel_Qvalue<BestChannel_Qvalue_limit if "BestChannel" in conditions else np.ones_like(fdxB["channel"],dtype=bool),
                                                                fdxB.num_b>=at_least_b if "num_b" in conditions else np.ones_like(fdxB["channel"],dtype=bool),
                                                                
                                                                # ["C" not in i or "M" not in i for i in fdxB.untag_seq],
                                                                # ["C"  in i for i in fdxB.untag_seq],
                                                                # np.log10(fdxA.coeff)>4
                                                                # np.log10(fdxB.coeff)>4
                                                                # np.abs(fdxB.rt_error)<.5
                                                               # fdxB.prop==1,
                                                             ])])
                                                              
                                                              
    shared_keys = intersect_all(intersect1,intersect2)
    
    ## show intersection
    shared_keys_set = set(shared_keys)
    sk_bool1 = np.array([i in shared_keys_set for i in fdxA["untag_prec"]]) ## faster
    sk_bool2 = np.array([i in shared_keys_set for i in fdxB["untag_prec"]]) ## faster
    
    ### show all
    intersect1_set = set(intersect1)
    intersect2_set = set(intersect2)
    # sk_bool1 = np.array([i in intersect1_set for i in fdxA["untag_prec"]]) ## faster
    # sk_bool2 = np.array([i in intersect2_set for i in fdxB["untag_prec"]]) ## faster
    
    
    # sk_bool1 = np.array([i in intersect1_set and i not in intersect2_set for i in fdxA["untag_prec"]]) ## faster
    # sk_bool2 = np.array([i in intersect2_set and i not in intersect1_set for i in fdxB["untag_prec"]]) ## faster
    
    
    
    ###!!! need check here or have it earlier to ensure same ratio for all!!!
    theoretical_ratios = [1,np.nan,theoretical_yeast_amounts[channel_mix[idx2_1]]/theoretical_yeast_amounts[channel_mix[idx2_2]]]
    
    # theoretical_ratios = [1, np.nan,8] ## for timeplex
    
    
    
    
    plot_width = 15
    fig = plt.figure(figsize=(plot_width, plot_width/4))
    # gs = gridspec.GridSpec(1, 5, width_ratios=[1, 3,3,.5, 0.5], 
    #                        # wspace=(plt.gcf().get_size_inches()[0])*.02
    #                        )
    if scatter:
        dims = [[7,1],[1,3,3]]
        rotation = 90
        wspace=None
    else:
        dims = [[6,2],[2,2,2]]
        rotation=45
        wspace=None
        
        ## for narrow bars and hist
        dims = [[5,3],[1,2,2]]
        rotation = 90
        wspace=.5 
    gs = gridspec.GridSpec(1,2,width_ratios=dims[0],wspace=0.05)
    gs1 = gridspec.GridSpecFromSubplotSpec(1, 3,width_ratios=dims[1],wspace=wspace,subplot_spec =gs[0])
    gs2 = gridspec.GridSpecFromSubplotSpec(1, 2,subplot_spec =gs[1])
    
    titles = [title1 ,title2,"Both"]
    if "last_aa" in conditions:
        titles = [i+f"_{last_aa}" for i in titles]
    ax_bar = fig.add_subplot(gs1[0])
    ax_bar.bar(range(len(titles)),[len(intersect1),len(intersect2),len(shared_keys)])
    ax_bar.set_xticks(range(len(titles)),titles)
    ax_bar.set_xticks(range(len(titles)),["\n".join(re.split("\W",i)) for i in titles])
    ax_bar.set_xticks(range(len(titles)),titles,rotation=rotation)
    ax_bar.set_ylabel(f"Precursor Ratios ({channel_mix[idx2_1]}/{channel_mix[idx2_2]})")
    plot_offset = 0
    quant_names = ["merged_MS1_Area","merged_plex_Area"]
    # quant_names = ["plex_Area","Ms1.Area"]
    for idx1,idx2,fdx,channels,sk_bool, file_idx1,file_idx2 in [
        [idx1_1,idx1_2,fdxA,channels,sk_bool1,file_idx1_1,file_idx1_2],
        [idx2_1,idx2_2,fdxB,channels,sk_bool2,file_idx2_1,file_idx2_2]]:
        # break
    
        # if "plexfitMS1" in fdx.columns:
        #     quant_name = "plexfitMS1"
        # else:
        #     quant_name = "MS1_Int"
        # # quant_name = "MS1_Area"
        # # quant_name="log_plex_Area"
        # quant_name = quant_names[plot_offset]
        # # quant_name = "merged_quant"
        # # quant_name = "merged_coeff"
        # # quant_name = "merged_plexfitMS1" 
        # quant_name = "merged_plex_Area"
        # # quant_name = "MS1_Area"
        # quant_name = "merged_new_plexfitMS1"
        
        orig_quant_name = re.sub("merged_","",quant_name)
        o_i = 0
        offset=0
        max_val=0
        x_lims = [15,31]
        x_lims = np.percentile(np.log2(fdx[np.logical_and(np.logical_or(fdx.file_idx==file_idx1,fdx.file_idx==file_idx2),fdx[orig_quant_name]>0)][orig_quant_name]),(5,99.9))
        y_lim = 6
        offsets=[0,0]
        dfs = []
        
        max_kde = 0
        
        # ax_hist = fig.add_subplot(gs[1])
        ax_scatter = fig.add_subplot(gs1[1+plot_offset])
        ax_box = fig.add_subplot(gs2[plot_offset])
        # ax_hist.sharey(ax_scatter)
        ax_box.sharey(ax_scatter)
        # ax_scatter.set_zorder(ax_box.get_zorder())
        num_orgs = 2
        for o_i in [0,2]:
            
            
            # i1 = fdx[logic_and([fdx["channel"]==channels[idx1],sk_bool,fdx["org"]==orgs[o_i]])]
            # i2 = fdx[logic_and([fdx["channel"]==channels[idx2],sk_bool,fdx["org"]==orgs[o_i]])]
            i1 = fdx[logic_and([fdx["channel"]==channels[idx1],sk_bool,fdx["org"]==orgs[o_i],fdx["file_idx"]==file_idx1])]
            i2 = fdx[logic_and([fdx["channel"]==channels[idx2],sk_bool,fdx["org"]==orgs[o_i],fdx["file_idx"]==file_idx2])]
            
            i1 = i1.sort_values(by="untag_prec")
            i2 = i2.sort_values(by="untag_prec")
            
            if "merged" in quant_name:
                log_ratios = i1[quant_name].to_numpy()-i2[quant_name].to_numpy()
            else:
                log_ratios = np.log2(np.divide(i1[quant_name].to_numpy(),i2[quant_name].to_numpy()))
                log_ratios = np.log2(np.divide(i1[quant_name].to_numpy()+1,i2[quant_name].to_numpy()+1))
            i1["log_ratios"] = log_ratios
            i2["log_ratios"] = log_ratios
            
            # rt_diffs = np.subtract(i2.rt.to_numpy(),i1.rt.to_numpy())
            # i1["rt_diffs"] = rt_diffs
            # i2["rt_diffs"] = rt_diffs
        
            
            if o_i==0:
                offset = np.nanmedian(log_ratios)
            
            
            
            # ax_hist.hist(log_ratios-offset, bins=np.linspace(-y_lim,y_lim,60), orientation='horizontal', color=org_colors[o_i],alpha=.5,density=True,label=orgs[o_i])
            
            
           
            if scatter:
                ax_scatter.scatter(np.log2(i1[orig_quant_name]), log_ratios-offset, color=org_colors[o_i], 
                                   alpha=.2,s=10,edgecolor="none",label=orgs[o_i])
                ax_scatter.hlines(y=np.log2(theoretical_ratios)[o_i],xmin=x_lims[0],xmax=x_lims[1]+120,
                                  colors=org_colors[o_i],clip_on=True,zorder=0,linestyles="--")
                ax_scatter.set_xlim(*x_lims)
                ax_scatter.set_xlabel(f"log$_{2}$({channel_mix[idx2_1]})")
                if plot_offset==0:
                    ax_scatter.legend(markerscale=5)
            else:
                kde=gaussian_kde(log_ratios-offset)
                bins = np.linspace(-y_lim,y_lim,60)
                # ax_hist.plot(kde(bins),bins,color=org_colors[o_i])
                ax_scatter.fill_betweenx(bins,0,kde(bins),alpha=.5,color=org_colors[o_i],label=orgs[o_i])
                max_kde = max(max(kde(bins)),max_kde)
                ax_scatter.hlines(y=np.log2(theoretical_ratios)[o_i],xmin=0,xmax=120,
                                  colors=org_colors[o_i],clip_on=True,zorder=0,linestyles="--")
                ax_scatter.set_xlim(0,max_kde*1.1)
                ax_scatter.set_xlabel(f"Density")
                
                if plot_offset==0:
                    ax_scatter.legend()
            
            if box:
                ax_box.boxplot((log_ratios-offset)[~np.isnan(log_ratios)],positions=[o_i],labels=[orgs[o_i]], 
                               widths=1, vert=True, patch_artist=True, boxprops=dict(facecolor=org_colors[o_i]),
                               showfliers=True, 
                               flierprops = dict(marker='o', markerfacecolor=org_colors[o_i], markersize=1,
                                                 linestyle='none',markeredgecolor=org_colors[o_i]),
                               notch=True,zorder=100,medianprops={"color":"black"})
                ax_box.hlines(y=np.log2(theoretical_ratios)[o_i],xmin=-.5,xmax=num_orgs+.5,colors=org_colors[o_i],clip_on=False,zorder=0,linestyles="--")
                
            else:
                kde=gaussian_kde(log_ratios-offset)
                bins = np.linspace(-y_lim,y_lim,60)
                # ax_hist.plot(kde(bins),bins,color=org_colors[o_i])
                ax_box.fill_betweenx(bins,0,kde(bins),alpha=.5,color=org_colors[o_i],label=orgs[o_i])
                max_kde = max(max(kde(bins)),max_kde)
                ax_box.hlines(y=np.log2(theoretical_ratios)[o_i],xmin=0,xmax=120,
                                  colors=org_colors[o_i],clip_on=True,zorder=0,linestyles="--")
                ax_box.set_xlim(0,max_kde*1.1)
                # ax_box.set_xlabel(f"Density")
            
            
            plt.ylim(-y_lim,y_lim)
            # ax_scatter.set_xlim(np.min(np.log2(i1[quant_name]))*.95,np.max(np.log2(i1[quant_name]))*1.05)
            # ax_scatter.set_xlim(*x_lims)
            
            dfs.append([i1,i2])
            
            
        ax_scatter.spines['top'].set_visible(False)
        ax_scatter.spines['right'].set_visible(False)
        # ax_hist.set_xlabel("Density")
        # ax_hist.spines['top'].set_visible(False)
        # ax_hist.spines['right'].set_visible(False)
        # ax_hist.spines['left'].set_visible(False)
        # ax_hist.yaxis.set_tick_params(labelleft=False) 
        # ax_hist.yaxis.set_tick_params(left=False)
        ax_bar.spines['top'].set_visible(False)
        ax_bar.spines['right'].set_visible(False)
        ax_box.spines['top'].set_visible(False)
        ax_box.spines['right'].set_visible(False)
        ax_box.spines['left'].set_visible(False)
        ax_box.spines['bottom'].set_visible(False)
        ax_box.yaxis.set_tick_params(labelleft=False) 
        ax_box.yaxis.set_tick_params(left=False)
        ax_box.xaxis.set_tick_params(labelbottom=False) 
        ax_box.xaxis.set_tick_params(bottom=False)
        
        # plt.xlim(-4,4)
        rect = plt.Rectangle(
                (0, 1.0),  # (x, y) position relative to the subplot
                1,  # Full width
                0.1,  # Height of the box
                edgecolor="black",
                transform=ax_scatter.transAxes, facecolor='lightgrey', alpha=0.6, clip_on=False, zorder=1
            )
        ax_scatter.add_patch(rect)
        
        rect = plt.Rectangle(
                (0, 1.0),  # (x, y) position relative to the subplot
                1,  # Full width
                0.1,  # Height of the box
                edgecolor="black",
                transform=ax_box.transAxes, facecolor='lightgrey', alpha=0.6, clip_on=False, zorder=1
            )
        ax_box.add_patch(rect)
        ax_scatter.set_title([long_title1,long_title2][plot_offset])
        # ax_scatter.set_title(titles[plot_offset])#,bbox={'facecolor':'0.8', 'pad':5})
        ax_box.set_title(titles[plot_offset])
        
        if plot_offset==0:
            ax_scatter.set_ylabel(f"log$_{2}$({channel_mix[idx2_1]}/{channel_mix[idx2_2]})")
        # ax_scatter.set_xlabel(f"log({channel_mix2[idx2_1]})")
        # ax_scatter.set_title(f"MS"+("2" if quant_name=="coeff" else "1") + " Quant" + f" - d{channels_de[idx1_de]}/d{channels_de[idx2_de]}" +(f", End:{last_aa}" if 'last_aa' in conditions else ''))
        
      
        
        plot_offset+=1
        
    









#############################################################################
##############   Jaccard Similarity ############################################
#############################################################################

def highlight_cell(x,y, ax=None, **kwargs):
    ### From: https://stackoverflow.com/questions/56654952/how-to-draw-cell-borders-in-imshow
    rect = plt.Rectangle((x-.5, y-.5), 1,1, fill=False, **kwargs)
    ax = ax or plt.gca()
    ax.add_patch(rect)
    return rect


### precursor Jaccard
def precursor_jaccard_matrix(fdx,
                             channels,
                             quant_name = "plexfitMS1"):
    
    # quant_name = "coeff"
    
    ### filter the data by below
    filter_column = "BestChannel_Qvalue"
    
    # fdx_long = fdx.pivot(index=["untag_seq","z","org"],columns="channel",values=[quant_name,"rt","plexfitMS1_p"])
    fdx_long = fdx[fdx[filter_column]<.01].pivot(index=["untag_seq","z","org"],columns=["file_idx","channel"],values=[quant_name,"rt","plexfitMS1_p"])
    fdx_long.iloc[np.where(fdx_long==0)] = np.nan
    
    file_idxs = np.unique(fdx.file_idx)
    file_idx = 0
    file_idx
    idx1=0
    idx2=1
    jaccard_matrix = np.zeros((len(channels)*len(file_idxs),len(channels)*len(file_idxs)))*np.nan
    # jaccard_matrix[len(channels)-1-np.arange(len(channels)),np.arange(len(channels))] = 1
    
    # [file_idx1,idx1],[file_idx2,idx2] = [0,2],[1,2]
    # jaccard_matrix[(len(file_idxs)-file_idx1)*len(channels)-1-channels.index(idx1),(file_idx2)*len(channels)+channels.index(idx2)] = 1
    
    
    # for idx1,idx2 in combinations_with_replacement(channels,2):
    for [file_idx1,idx1],[file_idx2,idx2] in list(combinations_with_replacement(list(product(file_idxs,channels)),2)):
        # break
        # q1 = fdx_long[quant_name][idx1]
        # q2 = fdx_long[quant_name][idx2]
        q1 = fdx_long[quant_name][file_idx1][idx1]
        q2 = fdx_long[quant_name][file_idx2][idx2]
        both_absent = np.sum(np.logical_and(np.isnan(q1),np.isnan(q2)))
        jaccard_matrix[(len(file_idxs)-list(file_idxs).index(file_idx1))*len(channels)-1-channels.index(idx1),(list(file_idxs).index(file_idx2))*len(channels)+channels.index(idx2)] = 1-((np.sum(np.isnan(q1*q2))-both_absent)/(len(q1*q2)-both_absent)) 
    
    return jaccard_matrix


#### protein Jaccard
def protein_jaccard_matrix(fdx,
                             channels,
                             quant_name = "plexfitMS1"):
    # create df with channel and protein
    protein_df = fdx[["channel","protein",quant_name]][np.logical_and(fdx.Protein_Qvalue<.01 , fdx.BestChannel_Qvalue<.01)]
    protein_df= protein_df.drop_duplicates(["channel","protein"],keep="last")
    protein_df = protein_df[[";" not in i for i in protein_df.protein]]## remove proteins that shared a peptide e.g. proteinA;proteinB
    pdf_long = protein_df.pivot(index=["protein"],columns="channel",values=[quant_name])
    
    file_idxs = np.unique(fdx.file_idx)
    
    idx1=0
    idx2=1
    jaccard_matrix = np.zeros((len(channels),len(channels)))*np.nan
    # jaccard_matrix[len(channels)-1-np.arange(len(channels)),np.arange(len(channels))] = 1
    for idx1,idx2 in combinations_with_replacement(channels,2):
        q1 = pdf_long[quant_name][idx1]
        q2 = pdf_long[quant_name][idx2]
        both_absent = np.sum(np.logical_and(np.isnan(q1),np.isnan(q2)))
        jaccard_matrix[len(channels)-1-channels.index(idx1),channels.index(idx2)] = 1-((np.sum(np.isnan(q1*q2))-both_absent)/(len(q1*q2)-both_absent)) 
        
    jaccard_matrix = np.zeros((len(channels)*len(file_idxs),len(channels)*len(file_idxs)))*np.nan
    # jaccard_matrix[len(channels)-1-np.arange(len(channels)),np.arange(len(channels))] = 1
    
    
    ### Multi file version
    # create df with channel and protein
    protein_df = fdx[["channel","protein",quant_name,"file_idx"]]
    protein_df= protein_df.drop_duplicates(["channel","protein","file_idx"],keep="last")
    protein_df = protein_df[[";" not in i for i in protein_df.protein]]## remove proteins that shared a peptide e.g. proteinA;proteinB
    pdf_long = protein_df.pivot(index=["protein"],columns=["file_idx","channel"],values=[quant_name])
    
    
    # for idx1,idx2 in combinations_with_replacement(channels,2):
    for [file_idx1,idx1],[file_idx2,idx2] in list(combinations_with_replacement(list(product(file_idxs,channels)),2)):
        # break
        # q1 = fdx_long[quant_name][idx1]
        # q2 = fdx_long[quant_name][idx2]
        q1 = pdf_long[quant_name][file_idx1][idx1]
        q2 = pdf_long[quant_name][file_idx2][idx2]
        both_absent = np.sum(np.logical_and(np.isnan(q1),np.isnan(q2)))
        jaccard_matrix[(len(file_idxs)-list(file_idxs).index(file_idx1))*len(channels)-1-channels.index(idx1),(list(file_idxs).index(file_idx2))*len(channels)+channels.index(idx2)] = 1-((np.sum(np.isnan(q1*q2))-both_absent)/(len(q1*q2)-both_absent)) 

    return jaccard_matrix



## NB, imshow reverses the y axis from neg on top to pos on bottom
def plot_jaccard_matrix(fdx,
                        channels,
                        channel_mix,
                        j_type="protein",
                        colormap="copper",
                        alpha=1,
                        min_val = .5):
    
    if j_type=="protein":
        jaccard_matrix = protein_jaccard_matrix(fdx, channels)
    else:
        jaccard_matrix = precursor_jaccard_matrix(fdx, channels)
        
    file_idxs = np.unique(fdx.file_idx)
    
    ax = plt.subplot(111)
    # file_idxs = [0,1,2,3]#np.unique(fdx.file_idx)
    # ax2 = ax.twinx()
    plt.imshow(jaccard_matrix,cmap=colormap,alpha=alpha)
    plt.clim(min_val,1)
    ax.set_xticks(np.arange(len(channels)*len(file_idxs)),labels=channel_mix[:len(channels)]*len(file_idxs))
    plt.yticks(np.arange(len(channels)*len(file_idxs)),labels=channel_mix[:len(channels)][::-1]*len(file_idxs))
    ax.xaxis.set_ticks_position('bottom')
    # plt.grid()
    ax.spines[['right', 'top']].set_visible(False)
    for file_idx in range(len(file_idxs)):
        # ax2=ax.secondary_xaxis(-.1,transform=ax.transData)
        # ax2.set_xticks([file_idx*len(channels)+(len(channels)-1)/2],labels=[f"Rep {file_idx+1}"])
        # ax2.set_xlim(0,1)
        
        # ax3=ax.secondary_yaxis(-.1,transform=ax.transData)
        # ax3.set_yticks([(len(file_idxs)-file_idx)*len(channels)-(len(channels)-1)/2],labels=[f"Rep {file_idx+1}"])
    
    
        buffer=.2
        trans = ax.get_xaxis_transform()
        ax.plot([len(channels)*file_idx+buffer-.5,
                  len(channels)*(file_idx+1)-buffer-.5],
                 # [(len(file_idxs))*len(channels)+1.2,(len(file_idxs))*len(channels)+1.2],
                 [-.1,-.1],
                  clip_on=False,zorder=100,
                  transform=trans,color="black",
                  linewidth=.5)
        
        ax.text(x=file_idx*len(channels)+(len(channels)-1)/2,#file_idx*len(channels)+(len(channels)-1)/2,
                y=-.11,#(len(file_idxs))*len(channels)+1.2,
                s=f"Rep {file_idx+1}",
                transform=trans,color="black",
                ha="center",va="top")
        
        
        trans = ax.get_yaxis_transform()
        ax.plot([-.1,-.1],
                [len(channels)*file_idx+buffer-.5,
                  len(channels)*(file_idx+1)-buffer-.5],
                  clip_on=False,zorder=100,
                  transform=trans,color="black",
                  linewidth=.5)
        
        ax.text(y=(len(file_idxs)-1-file_idx)*len(channels)+(len(channels)-1)/2,
                x=-.11,
                s=f"Rep {file_idx+1}",
                ha="right",va="center",
                rotation=90,
                transform=trans)
        
    # highlight_cell(10, 3)
    
    for [file_idx1,idx1],[file_idx2,idx2] in list(combinations_with_replacement(list(product(file_idxs,channels)),2)):
        plt.text((list(file_idxs).index(file_idx2))*len(channels)+channels.index(idx2),
                 (len(file_idxs)-list(file_idxs).index(file_idx1))*len(channels)-1-channels.index(idx1),
                 s=np.array(np.round(jaccard_matrix[(len(file_idxs)-list(file_idxs).index(file_idx1))*len(channels)-1-channels.index(idx1),
                                                    (list(file_idxs).index(file_idx2))*len(channels)+channels.index(idx2)]*100),
                            dtype=np.int32),
                 ha="center",va="center",size=6)
        highlight_cell((len(file_idxs)-list(file_idxs).index(file_idx1))*len(channels)-1-channels.index(idx1),(list(file_idxs).index(file_idx2))*len(channels)+channels.index(idx2))
        
        
        
#### Venn diagrams

def plot_venns(fdx1,group,title="",names=None,level="protein"):
        
    if level=="protein":
        protein_df = fdx1[["channel","protein","coeff"]][np.logical_and(fdx1.Protein_Qvalue<.01 , fdx1.BestChannel_Qvalue<.01)]
        
    assert len(sets)==len(names)
    
    plt.subplots()
    if len(sets)==2:
        venn2(sets,names)
    elif len(sets)==3:
        venn3(sets,names)
    else:
        raise ValueError("Set number must be 2 or 3")
    plt.title("Jmod")
        
def get_abs_ratio_errors(fdx,channel_mix_key,quant_name,plexes, 
                         theoretical_yeast_amounts,
                         plex_name="file_idx",conditions=[],
                         at_least_b = 3,
                         Q_value_limit = 0.05,
                         BestChannel_Qvalue_limit = 0.01):
    
    """
    

    Parameters
    ----------
    fdx : TYPE
        fdx as above.
    channel_mix_key : TYPE
        dict where plex channels map to teortical amounts 
    plex_name : TYPE, optional
        What coloum to perform the comparison over. 
        "channel" for plexDIA, "file_idx" for LF
        The default is "file_idx".

    Returns
    -------
    None.

    """
    # plexes = np.unique(fdx[plex_name])
    channel_bools = {i:fdx[plex_name]==i for i in plexes}
    all_temp_dfs = []
    file_temp_dfs = []
    for idx1,idx2 in tqdm.tqdm(list(combinations(plexes,2))):
        # break
        # print(f"{channel_mix[idx1]}/{channel_mix[idx2]}")
        # lower_score = T
        # quant_name = "MS1_Area"
        
        cond_bool = np.logical_and.reduce([
                                            fdx.Qvalue<Q_value_limit if "Qvalue" in conditions else np.ones_like(fdx["channel"],dtype=bool),
                                            fdx.BestChannel_Qvalue<BestChannel_Qvalue_limit if "BestChannel" in conditions else np.ones_like(fdx["channel"],dtype=bool),
                                            fdx.num_b>=at_least_b if "num_b" in conditions else np.ones_like(fdx["channel"],dtype=bool)])
        
        shared_keys = np.intersect1d(fdx["untag_prec"][logic_and([fdx[plex_name]==idx1,
                                                                    # fdx["plexfitMS1_p"]>.75, 
                                                                   # fdx.Qvalue<0.01,
                                                                   ~fdx.decoy,
                                                                   cond_bool
                                                                  ])],
                                     fdx["untag_prec"][logic_and([fdx[plex_name]==idx2,
                                                                    # fdx["plexfitMS1_p"]>.75,
                                                                   # fdx.Qvalue<0.01,
                                                                   ~fdx.decoy,
                                                                   cond_bool
                                                                  ])])
        
        shared_keys_set = set(shared_keys)
        sk_bool = np.array([i in shared_keys_set for i in fdx["untag_prec"]]) ## faster
        
        theoretical_ratios = [1,np.nan,theoretical_yeast_amounts[channel_mix_key[idx1]]/theoretical_yeast_amounts[channel_mix_key[idx2]]]
    
        o_i = 0
        offset=0
        max_val=0
        temp_dfs = []
        for o_i in [0,2]:
            
            # i1 = fdx[logic_and([channel_bools[idx1],sk_bool,fdx["org"]==orgs[o_i]])]
            # i2 = fdx[logic_and([channel_bools[idx2],sk_bool,fdx["org"]==orgs[o_i]])]
            i1 = fdx[logic_and([channel_bools[idx1],sk_bool,fdx["org"]==orgs[o_i],fdx[plex_name]==idx1,cond_bool])]
            i2 = fdx[logic_and([channel_bools[idx2],sk_bool,fdx["org"]==orgs[o_i],fdx[plex_name]==idx2,cond_bool])]
            
            i1 = i1.sort_values(by="untag_prec")
            i2 = i2.sort_values(by="untag_prec")
            
            if np.min(i1[quant_name])<0:## log scale
                log_ratios = i1[quant_name].to_numpy()-i2[quant_name].to_numpy()
            else:
                log_ratios = np.log2(np.divide(i1[quant_name].to_numpy(),i2[quant_name].to_numpy()))
            if o_i==0:
                offset = np.nanmedian(log_ratios)
    
            temp_df = pd.DataFrame({"abs_error":np.abs(np.log2(theoretical_ratios)[o_i]-(log_ratios-offset)),
                                    "untag_prec":i1["untag_prec"],
                                    "org":[orgs[o_i] for i in log_ratios],
                                    "plexes":["/".join([str(idx1),str(idx2)]) for i in log_ratios]
                                    # "tag":[tag_name for i in log_ratios]
                                    })
            file_temp_dfs.append(temp_df)
            # vals,bins,_ = plt.hist(log_ratios-offset,np.linspace(-6,6,100),label=orgs[o_i],color=org_colors[o_i],alpha=.5,
            #                         # density=True
                                    # )
            
        # break
    ratio_error_df = pd.concat(file_temp_dfs,ignore_index=True)
    
    return ratio_error_df
        

def errors_boxplot(all_temp_dfs,
                   file_names,da,ms
                   ):
    all_data = pd.concat([df.assign(file_name=f,da=d,ms=m) for df,f,d,m in zip(all_temp_dfs,file_names,da,ms)],ignore_index=True)
    
    all_data.columns
    
    
    all_data["C_term"] = ["R" if i[-3]=="R" else "K" for i in all_data["untag_prec"]]
    # all_data["file_name"] = [file_namer[file_runs[i]] for i in all_data.file_idx]
    
    # ax = plt.subplot(111)
    
    
    medianprops={"linewidth":2.5}
    ax = sb.boxplot(all_data,x="file_name",y="abs_error",hue="C_term",hue_order=["K","R"],
                    fliersize=0,showcaps=False,widths=.3,medianprops=medianprops)
    max_val = 2.5
    # ax = sb.violinplot(all_data[all_data["abs_error"]<max_val],x="file_name",y="abs_error",
    #                     hue="C_term",cut=0)
    plt.ylim(-.1,max_val)
    plt.ylabel("|log$_2$ Precursor Ratio Error|")
    legend = ax.legend(loc="upper right")
    plt.xlabel("")
    # plt.title("DE 6-plex 8 ng")
    
    # Hide the right and top spines
    ax.spines[['right', 'top']].set_visible(False)
    
    
    ### add medians
    
    medians = all_data.groupby(["file_name", "C_term"])["abs_error"].median().reset_index()
    
    # Get the x positions of each box
    box_positions = {category: i for i, category in enumerate(all_data["file_name"].unique())}
    
    for i, row in medians.iterrows():
        x_pos = box_positions[row["file_name"]]  # Base x position
        # Adjust x position based on hue
        x_offset = -0.2 if row["C_term"] == "K" else 0.2
        ax.text(x_pos + x_offset, row["abs_error"] + .06, f"{row['abs_error']:.2f}", 
                ha='center', fontsize=8, fontweight='bold', color='black')
    
    for patch in ax.patches:
        patch.set_alpha(0.5) 
    
    for legend_handle in legend.legendHandles:
        legend_handle.set_alpha(0.5)  
        
        

#### distributions of different combinations


# fdxA["mix"] = [channel_mix1[channels1.index(i)] for i in fdxA.channel]
# fdxB["mix"] = [channel_mix2[channels2.index(i)] for i in fdxB.channel]
# fdx["mix"] = [channel_mix[channels.index(i)] for i in fdx.channel]
# fdx2["mix"] = [channel_mix[channels.index(i)] for i in fdx2.channel]
# diann_report["mix"]=  [channel_mix[channels.index(i)] for i in diann_report.channel]
# list(combinations(np.unique(channel_mix1), 2))

# file_idx1=0
# file_idx2=0
# quant_name = "coeff"
# quant_name = "plexfitMS1"
# quant_name = "merged_coeff"
# quant_name = "plex_Area"
# quant_name = "new_plexfitMS1"
# quant_name = "merged_new_plexfitMS1"
# # quant_name = "merged_plex_Area"

# channels = channels2[::2]
# channel_mix =channel_mix1



def compute_all_comparisons(fdx,
                            quant_name,
                            channels,
                            channel_mix,
                            theoretical_yeast_amounts,
                            plot_type = "all",
                            conditions=['Qvalue', 'BestChannel',"num_b"],
                            at_least_b = 3,
                            Q_value_limit = 0.05,
                            BestChannel_Qvalue_limit = 0.01,
                            name=""
                            ):

# quant_name = "plex_Area"
# intersect1 = jpf.get_intersect(fdxA, fdxA,  
#                                conditions, 
#                                file_idx1=file_idx1, file_idx2=file_idx2,
#                                 channels=[0,2],
#                                # mix=["A","B"]
#                                )

# intersect2 = jpf.get_intersect(fdxA, fdxA, conditions, channels, 0, 3, file_idx1, file_idx2)
# shared_keys = intersect_all(intersect1,intersect2)

    
    all_ratios = []
    mix_values={i:[] for i in list(combinations_with_replacement(np.unique(channel_mix), 2))}
    
    scale=1.3
    if plot_type=="all":
        fig,ax = plt.subplots(figsize=(7*scale,1*scale))
    hist_labels = []
    for idx,chan in enumerate(list(combinations(np.unique(channels), 2))):
        # break
        
        intersect1 = get_intersect(fdx, fdx,
                                       conditions=['Qvalue', 'BestChannel',"num_b"], 
                                       # file_idx1=file_idx1, file_idx2=file_idx2,
                                        channels=chan,
                                        quant_name=quant_name,
                                       # mix=["A","B"]
                                        at_least_b = at_least_b,
                                        Q_value_limit = Q_value_limit,
                                        BestChannel_Qvalue_limit = BestChannel_Qvalue_limit
                                       )
        # intersect1 = intersect_all(get_intersect(fdx, fdx,
        #                                conditions=['Qvalue', 'BestChannel',"num_b"], 
        #                                file_idx1=file_idx1, file_idx2=file_idx2,
        #                                 channels=chan,
        #                                 quant_name=quant_name,
        #                                # mix=["A","B"]
        #                                 Q_value_limit=.01
        #                                ),
        #             get_intersect(fdx1, fdx1,
        #                                conditions=['Qvalue', 'BestChannel',"num_b"], 
        #                                file_idx1=file_idx1, file_idx2=file_idx2,
        #                                 channels=chan,
        #                                 quant_name=quant_name,
        #                                # mix=["A","B"]
        #                                 Q_value_limit=.01
        #                                ))
        shared_keys_set = set(intersect1)
        sk_bool = np.array([i in shared_keys_set for i in fdx["untag_prec"]]) ## faster
        
        org_offset = 0
        offset=0
        y_lim = 4
        for o_i in [0,2]:
            # break
            
            # i1 = fdx[logic_and([fdx["channel"]==channels[idx1],sk_bool,fdx["org"]==orgs[o_i]])]
            # i2 = fdx[logic_and([fdx["channel"]==channels[idx2],sk_bool,fdx["org"]==orgs[o_i]])]
            # i1 = fdx[logic_and([fdx["channel"]==chan[0],sk_bool,fdx["org"]==orgs[o_i],fdx["file_idx"]==file_idx1])]
            # i2 = fdx[logic_and([fdx["channel"]==chan[1],sk_bool,fdx["org"]==orgs[o_i],fdx["file_idx"]==file_idx2])]
            i1 = fdx[logic_and([fdx["channel"]==chan[0],sk_bool,fdx["org"]==orgs[o_i]])]
            i2 = fdx[logic_and([fdx["channel"]==chan[1],sk_bool,fdx["org"]==orgs[o_i]])]
            
            i1 = i1.sort_values(by="untag_prec")
            i2 = i2.sort_values(by="untag_prec")
            # if np.min(i1[quant_name])<0:## log scale
            if "merged" in quant_name:
                # print("log")
                log_ratios = i1[quant_name].to_numpy()-i2[quant_name].to_numpy()
            else:
                # print("no log")
                log_ratios = np.log2(np.divide(i1[quant_name].to_numpy(),i2[quant_name].to_numpy()))
            # log_ratios = i1[quant_name].to_numpy()-i2[quant_name].to_numpy()
            i1["log_ratios"] = log_ratios
            i2["log_ratios"] = log_ratios
            
            if o_i==0:
                offset = np.nanmedian(log_ratios)
            
            if o_i==2:
                # plt.boxplot(log_ratios-offset,positions=[idx+org_offset/2],
                #             widths=.4, vert=True, patch_artist=True, boxprops=dict(facecolor=org_colors[o_i]),
                #             showfliers=False, 
                #             flierprops = dict(marker='o', markerfacecolor=org_colors[o_i], markersize=1,
                #                               linestyle='none',markeredgecolor=org_colors[o_i]),
                #             notch=True,zorder=100,medianprops={"color":"black"})
                
                sign = 1 if tuple(sorted([i1.mix.iloc[0],i2.mix.iloc[0]]))==tuple(([i1.mix.iloc[0],i2.mix.iloc[0]])) else -1
                mix_values[tuple(sorted([i1.mix.iloc[0],i2.mix.iloc[0]]))]+=list(sign*(log_ratios-offset))
                
                if plot_type == "all":
                    kde=gaussian_kde(log_ratios-offset)
                    bins = np.linspace(-y_lim,y_lim,60)
                    # ax_hist.plot(kde(bins),bins,color=org_colors[o_i])
                    plt.fill_betweenx(bins,idx,idx+kde(bins),alpha=.5,color=org_colors[o_i],label=orgs[o_i])
                # org_offset+=1
                    
                    t_ratio = np.log2(theoretical_yeast_amounts[i1.mix.iloc[0]]/theoretical_yeast_amounts[i2.mix.iloc[0]])
                    plt.hlines(t_ratio,idx,idx+1)
                    # plt.hlines(np.mean(log_ratios)-offset,idx,idx+1,colors="r")
                    hist_labels.append(str(chan[0])+"/"+str(chan[1])+"\n"+i1.mix.iloc[0]+"/"+i2.mix.iloc[0])
                    # hist_labels.append(str(chan[0])+"/"+str(chan[1]))
      
    if plot_type=="all":
        plt.xticks(np.arange(len(hist_labels)),hist_labels,fontsize=4)
        plt.xlabel("Mixture Comparison",fontsize=5)
        plt.title(f"Jmod Quant {name} (all)",fontsize =7)
        # plt.hlines(np.unique(all_ratios),0,len(all_ratios))
    
    else:
        counter_size = 0.9
        only_A = False
        y_lim = 4
        tick_offset = .1
        count = 0
        counts = []
        hist_labels = []
        scale=1.3
        fig,ax = plt.subplots(figsize=(7*scale,1*scale))
        for idx,comb in enumerate(mix_values):
            if mix_values[comb]==[]:
                continue
            if only_A:
                if comb[0]!="A":
                    continue
            kde=gaussian_kde(mix_values[comb])
            bins = np.linspace(-y_lim,y_lim,60)
            # ax_hist.plot(kde(bins),bins,color=org_colors[o_i])
            plt.fill_betweenx(bins,count,count+ kde(bins),alpha=.5,color=org_colors[o_i],label=orgs[o_i]+" ratio" if count==0 else "")
            t_ratio = np.log2(theoretical_yeast_amounts[comb[0]]/theoretical_yeast_amounts[comb[1]])
            end_of_line = min(counter_size*.95,(max(kde(bins)))*1.2)
            # end_of_line = .9#
            plt.hlines(t_ratio,count,count+ end_of_line,label="Theoretical Ratio" if count==0 else "")
            hist_labels.append(comb[0]+"/"+comb[1])
            counts.append(count)
            count+=counter_size#
            # count+=end_of_line*1.1
        # plt.xticks(range(len(hist_labels)),hist_labels)
        plt.xticks(np.array(counts)+tick_offset,hist_labels)
        plt.legend(bbox_to_anchor=(1.2, 1.2))
        plt.ylabel("Log$_2$ Ratios")
        ax.spines[['right', 'top']].set_visible(False)


