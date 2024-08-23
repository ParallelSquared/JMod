import numpy as np
import pandas as pd
import re
import time
import pickle
import matplotlib.pyplot as plt
import multiprocessing
import tqdm
from functools import partial
from pyteomics import mass 

from SpecLib import load_tsv_speclib,load_tsv_lib, loadSpecLib
import load_files 
import SpecLib
import Jplot as jp
import config
import iso_functions as iso_f

from SpectraFitting import fit_to_lib, fit_to_pasef, fit_timspeak
from scipy.interpolate import LSQUnivariateSpline as spline
from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline
from scipy.optimize import isotonic_regression
from statistics import quantiles
from miscFunctions import within_tol
from scipy import signal
from scipy.optimize import curve_fit
from scipy import stats
import warnings
import dill
import itertools 
import h5py
# import alphatims.bruker
from mass_tags import tag_library, mTRAQ,mTRAQ_678, mTRAQ_02468

from miscFunctions import feature_list_mz, feature_list_rt, createTolWindows, within_tol,moving_average, \
    closest_ms1spec, closest_peak_diff,split_frag_name


# spectra = spectra
# spec_lib=spec_lib
# n=400

# stop

# mz_ppm = 10
# mz_tol = mz_ppm*10**(-6)

# all_spectra = spectra.ms2scans
# dia_data = [[spec.peak_list().T,spec.prec_mz,float(spec.RT),idx] for idx,spec in enumerate(all_spectra)]
# dia_data = [[dia_data[i][0],dia_data[i][1],dia_data[i][2],dia_data[i][3],float(dia_data[i+1][1]) - float(dia_data[i][1])] 
#             for i in range(len(dia_data)-1)]

# max_window_precmz = max([i[1] for i in dia_data])+max([i[4] for i in dia_data])
# max_window_offset =  max([i[4] for i in dia_data])

# dia_spectra = load_files.loadSpectra("/Volumes/Lab/Quant/CC20170118_SAM_Specter_Ecolidigest_DIA_01.mzML")
# librarySpectra = load_tsv_lib("/Volumes/Lab/Quant/SpecLibs/EcoliSpPrositLib.msp.tsv")
# # librarySpectra = load_tsv_lib("/Volumes/Lab/Quant/SpecLibs/EcoliSpPrositLib.msp.tsv")
# dino_features = pd.read_csv("/Volumes/Lab/Quant/CC20170118_SAM_Specter_Ecolidigest_DIA_01.features.tsv",sep="\t")


# dia_spectra = load_files.loadSpectra("/Volumes/Lab/Quant/ThermoTest2/K562_8000pg_1_ms2_bgrad.mzML")
# # librarySpectra = load_tsv_lib("/Users/kevinmcdonnell/Programming/Data/SpecLibs/HBthermo_prosit.msp.tsv")
# dino_features = pd.read_csv("/Volumes/Lab/Quant/ThermoTest2/K562_8000pg_1_ms2_bgrad.features.tsv",sep="\t")


# dia_spectra = load_files.loadSpectra("/Volumes/Lab/Quant/windows/K562_8000pg_15windows.mzML")
# # librarySpectra = load_tsv_lib("/Users/kevinmcdonnell/Programming/Data/SpecLibs/HBthermo_prosit.msp.tsv")
# dino_features = pd.read_csv("/Volumes/Lab/Quant/windows/K562_8000pg_15windows.features.tsv",sep="\t")


# dia_spectra = load_files.loadSpectra("/Volumes/Lab/KMD/Data/Windows/K562_8000pg_8windows.mzML")
# librarySpectra = loadSpecLib("/Users/kevinmcdonnell/Programming/Data/SpecLibs/tims_library_dec_23_PrositFrags.tsv")
# dino_features = pd.read_csv("/Volumes/Lab/KMD/Data/Windows/K562_8000pg_8windows.features.tsv",sep="\t")

# dia_spectra = load_files.SpectrumFile("/Users/kevinmcdonnell/Programming/Data/2023_08_28_LF-DIA_E480.mzML")
# # librarySpectra = load_tsv_lib("/Users/kevinmcdonnell/Programming/Data/SpecLibs/HBthermo_prosit.msp.tsv")
# dino_features = pd.read_csv("/Users/kevinmcdonnell/Programming/Data/2023_08_28_LF-DIA_E480.features.tsv",sep="\t")

# dia_spectra = load_files.SpectrumFile("/Users/kevinmcdonnell/Programming/Data/Neo_timePlex_loadcol2_K562_col1_K562.mzML")
# dino_features = pd.read_csv("/Users/kevinmcdonnell/Programming/Data/Neo_timePlex_loadcol2_K562_col1_K562.features.tsv",sep="\t")

# dia_spectra = load_files.SpectrumFile("/Volumes/Lab/KMD/timeplex/02062024_col3_LOAD_RUN123_50ng_re2.mzML")
# dino_features = pd.read_csv("/Volumes/Lab/KMD/timeplex/02062024_col3_LOAD_RUN123_50ng_re2.features.tsv",sep="\t")

# dia_spectra = load_files.SpectrumFile("/Volumes/Lab/KMD/2023-11-21_timeplex_loadC3_RUN_20240109152104.mzML")
# dino_features = pd.read_csv("/Volumes/Lab/KMD/2023-11-21_timeplex_loadC3_RUN_20240109152104.features.tsv",sep="\t")

# dia_spectra = load_files.SpectrumFile("/Volumes/Lab/KMD/timeplex/2023-01_16_timeplex_loadC3_RUN_re.mzML")
# dino_features = pd.read_csv("/Volumes/Lab/KMD/timeplex/2023-01_16_timeplex_loadC3_RUN_re.features.tsv",sep="\t")

# dia_spectra = load_files.SpectrumFile("/Volumes/Lab/KMD/timeplex/2023-01_16_timeplex_loadC3_RUN_re_closer.mzML")
# dino_features = pd.read_csv("/Volumes/Lab/KMD/timeplex/2023-01_16_timeplex_loadC3_RUN_re_closer.features.tsv",sep="\t")

# dia_spectra = load_files.SpectrumFile("/Volumes/Lab/KMD/Data/mTRAQ_Bulk/2024_02_23_MY_pDIA_Non_Red_Alk_500ng_v2-40_2x_orig_E480.mzML")
# dino_features = pd.read_csv("/Volumes/Lab/KMD/Data/mTRAQ_Bulk/2024_02_23_MY_pDIA_Non_Red_Alk_500ng_v2-40_2x_orig_E480.features.tsv",sep="\t")

# dia_spectra = load_files.SpectrumFile("/Volumes/Lab/KMD/Data/mTRAQ_Bulk/2024_02_23_MY_pDIA_Non_Red_Alk_500ng_v1-25_1x_orig_E480.mzML")

# dia_spectra = load_files.loadSpectra("/Volumes/Lab/KMD/Data/9plex/2024-03-21_Sciex_3-plex_678_100ng.mzML")
# dia_spectra = load_files.loadSpectra("/Volumes/Lab/KMD/Data/9plex/2024-03-21_Sciex_5-plex_01458_100ng.mzML")

# dia_spectra = load_files.loadSpectra("/Volumes/Lab/KMD/timeplex/4plex/20240611_allcols_LF_load_and_run_2p4kV.mzML")
# dino_features = pd.read_csv("/Volumes/Lab/KMD/timeplex/4plex/20240611_allcols_LF_load_and_run_2p4kV.features.tsv",sep="\t")

# dia_spectra = load_files.loadSpectra("/Volumes/Lab/KMD/timeplex/JD_timeplex_2col_IO_25cm_4ngeachLF_MS_andAux_PS_re_perfectly_equaldistance_45minGrad_rep2.mzML")
# dino_features = pd.read_csv("/Volumes/Lab/KMD/timeplex/JD_timeplex_2col_IO_25cm_4ngeachLF_MS_andAux_PS_re_perfectly_equaldistance_45minGrad_rep2.features.tsv",sep="\t")

# dia_spectra  = load_files.SpectrumFile("/Volumes/Lab/KMD/timeplex/JD_timeplex_2col_IO_25cm_4ngeachLF_MS_andAux_PS_re_perfectly_equaldistance_45minGrad_rep1.mzML")
# dino_features = pd.read_csv("/Volumes/Lab/KMD/timeplex/JD_timeplex_2col_IO_25cm_4ngeachLF_MS_andAux_PS_re_perfectly_equaldistance_45minGrad_rep1.features.tsv",sep="\t")


# diann_report = pd.read_csv("/Volumes/Lab/KMD/Data/mTRAQ_Bulk/Diann/Non_red_alk_500ng_v2_40_2x_orig_E480_RAW/report.tsv",sep="\t")
# diann_rt = {(re.sub("\(mTRAQ-[nK]-\d\)","",i),j):k for i,j,k in zip(diann_report["Modified.Sequence"],diann_report["Precursor.Charge"],diann_report["RT"])}
# for i in spec_lib:
#     spec_lib[i]["iRT"] = diann_rt[i]
# librarySpectra = loadSpecLib("/Users/kevinmcdonnell/Programming/Data/SpecLibs/tims_library_dec_23_PrositFrags.tsv")
# dia_spectra = load_files.SpectrumFile("/Volumes/Lab/KMD/Data/2024-02-05_QC_plex_DIA_e480.mzMl")
# dino_features = pd.read_csv("/Volumes/Lab/KMD/Data/2024-02-05_QC_plex_DIA_e480.features.tsv",sep="\t")
# # spec_lib = loadSpecLib("/Users/kevinmcdonnell/Programming/Data/SpecLibs/tims_library_dec_23_PrositFrags.tsv")
# spec_lib = loadSpecLib("/Volumes/Lab/KMD/Data/mTRAQ_Bulk/Diann/Non_red_alk_500ng_v2_40_2x_orig_E480_RAW/NEUlibSearchLibrary_PrositFrags.tsv")

# spec_lib = loadSpecLib("SpectronautUntagChan04.tsv")
# spec_lib = loadSpecLib("SpectronautUntagChan04_obs.tsv")
# spec_lib = loadSpecLib("/Volumes/Lab/KMD/Data/mTRAQ_Bulk/Diann/Non_red_alk_500ng_v2_40_2x_orig_E480_RAW/NEUlibSearchLibrary_PrositFrags_iTRAQ.tsv")
# spec_lib = loadSpecLib("/Volumes/Lab/KMD/DIANN_Searches/PredictedmTRAQ/library.tsv")
# config.tag = mTRAQ_678
# librarySpectra = tag_library(spec_lib,config.tag)
###librarySpectra = iso_f.iso_library(librarySpectra)

# dia_spectra = load_files.loadSpectra("/Volumes/Lab/KMD/Data/9plex/2024-07-02_SS_1Da-3plex_500pg_45min_4win.mzML")
# dino_features =         pd.read_csv("/Volumes/Lab/KMD/Data/9plex/2024-07-02_SS_1Da-3plex_500pg_45min_4win.features.tsv",sep="\t")
# dia_spectra = load_files.loadSpectra("/Volumes/Lab/KMD/Data/9plex/2024-07-02_SS_1Da-3plex_750pg_45min_4win.mzML")
# dino_features =         pd.read_csv("/Volumes/Lab/KMD/Data/9plex/2024-07-02_SS_1Da-3plex_750pg_45min_4win.features.tsv",sep="\t")
# dia_spectra = load_files.loadSpectra("/Volumes/Lab/KMD/Data/9plex/2024-07-02_SS_1Da-3plex_250pg_80min_v1-45.mzML")
# dino_features =         pd.read_csv("/Volumes/Lab/KMD/Data/9plex/2024-07-02_SS_1Da-3plex_250pg_80min_v1-45.features.tsv",sep="\t")
# dia_spectra = load_files.loadSpectra("/Volumes/Lab/KMD/Data/9plex/2024-07-02_SS_1Da-3plex_250pg_45min_4win.mzML")
# dino_features =         pd.read_csv("/Volumes/Lab/KMD/Data/9plex/2024-07-02_SS_1Da-3plex_250pg_45min_4win.features.tsv",sep="\t")

# dia_spectra = load_files.loadSpectra("/Volumes/Lab/KMD/Data/9plex/2024-07-12_SS_1Da-3plex_QCmeth_500pg.mzML")
# dino_features =         pd.read_csv("/Volumes/Lab/KMD/Data/9plex/2024-07-12_SS_1Da-3plex_QCmeth_500pg.features.tsv",sep="\t")
# dia_spectra = load_files.loadSpectra("/Volumes/Lab/KMD/Data/9plex/2024-07-12_SS_1Da-3plex_QCmeth_5ng.mzML")
# dino_features =         pd.read_csv("/Volumes/Lab/KMD/Data/9plex/2024-07-12_SS_1Da-3plex_QCmeth_5ng.features.tsv",sep="\t")
# dia_spectra = load_files.loadSpectra("/Volumes/Lab/KMD/Data/9plex/2024-07-12_SS_1Da-3plex_QCmeth_2ng.mzML")
# dino_features =         pd.read_csv("/Volumes/Lab/KMD/Data/9plex/2024-07-12_SS_1Da-3plex_QCmeth_2ng.features.tsv",sep="\t")
# dia_spectra = load_files.loadSpectra("/Volumes/Lab/KMD/Data/9plex/2024-07-12_SS_1Da-3plex_QCmeth_1ng_1.mzML")
# dino_features =         pd.read_csv("/Volumes/Lab/KMD/Data/9plex/2024-07-12_SS_1Da-3plex_QCmeth_1ng_1.features.tsv",sep="\t")
# dia_spectra = load_files.loadSpectra("/Volumes/Lab/KMD/Data/9plex/2024-07-12_SS_1Da-3plex_QCmeth_1ng_2.mzML")
# dino_features =         pd.read_csv("/Volumes/Lab/KMD/Data/9plex/2024-07-12_SS_1Da-3plex_QCmeth_1ng_2.features.tsv",sep="\t")
# dia_spectra = load_files.loadSpectra("/Volumes/Lab/KMD/Data/9plex/2024-07-12_SS_1Da-3plex_QCmeth_500pg.mzML")
# dino_features =         pd.read_csv("/Volumes/Lab/KMD/Data/9plex/2024-07-12_SS_1Da-3plex_QCmeth_500pg.features.tsv",sep="\t")



# library = librarySpectra
# librarySpectra = library
# diann_lib =  load_tsv_speclib("/Volumes/Lab/KMD/Data/mTRAQ_Bulk/Diann/Non_red_alk_500ng_v2_40_2x_orig_E480_RAW/NEUlibSearchLibrary.tsv")
# diann_lib = load_tsv_speclib("/Volumes/Lab/KMD/Data/mTRAQ_Bulk/Diann/Non_red_alk_500ng_v2_40_2x_orig_E480_RAW/NEUlibSearchLibrary_PrositFilter.tsv")
# librarySpectra=diann_lib

# stripped_lib = {(library[i]["seq"],i[1]):library[i] for i in library if "(mTRAQ-0)" in i[0]}
# stripped_diann = {(diann_lib[i]["seq"],i[1]):diann_lib[i] for i in diann_lib}
# dino_features =pd.read_csv("/Volumes/Lab/KMD/timeplex/2023-01_18_CytochromeC_col3_load_andRUN_8ngLFHuman.features.tsv",sep="\t")
# # ## e240 with low ms1 error
# dia_spectra = load_files.SpectrumFile("/Users/kevinmcdonnell/Programming/Data/2023_10_02_QC_LF_DIA_2_E240.mzML")
# dino_features = pd.read_csv("/Users/kevinmcdonnell/Programming/Data/2023_10_02_QC_LF_DIA_2_E240.features.tsv",sep="\t")

# files = ["/Users/kevinmcdonnell/Programming/Data/2023_10_02_QC_LF_DIA_2_E240.features.tsv",
#          "/Users/kevinmcdonnell/Programming/Data/Neo_timePlex_loadcol2_K562_col1_K562.features.tsv",
#          "/Volumes/Lab/KMD/2023-11-21_timeplex_loadC3_RUN_20240109152104.features.tsv"]
# for idx,file in enumerate(files):
#     dino_features = pd.read_csv(file,sep="\t")
#     # bins = np.linspace(3,10,30)
#     # plt.hist(np.log10(dino_features.intensitySum),bins,alpha=.5,label=f"{idx} plex")
#     bins = np.linspace(-10,10,30)
#     plt.hist(dino_features.fwhm,bins,alpha=.5,label=f"{idx+1} plex")
# plt.legend()
# librarySpectra = load_tsv_speclib("/Users/kevinmcdonnell/Programming/Data/SpecLibs/HBthermo_PrositFrags.tsv")
# librarySpectra = load_tsv_speclib("/Users/kevinmcdonnell/Programming/Data/SpecLibs/Human_Bruker_Library_PrositFrags.tsv")

### timsTOF 
"""
librarySpectra = load_tsv_speclib("/Users/kevinmcdonnell/Programming/Data/SpecLibs/tims_library_dec_23.tsv")

bruker_d_folder_name = "/Users/kevinmcdonnell/Programming/Data/2023_10_23_QC_LF_DIA_timsTOF_Slot2-20_1_893.d/"
data = alphatims.bruker.TimsTOF(bruker_d_folder_name)

file = "/Users/kevinmcdonnell/Programming/Python/2023_10_23_QC_LF_DIA_timsTOF_cfgfl_yaml_output_windows.hdf"
hdf_root = h5py.File(file, "r")

df = pd.DataFrame()
for key in hdf_root["clustering"]["as_dataframe"].keys():
    df[key]= hdf_root["clustering"]["as_dataframe"][key]

fragment_indices = hdf_root["ms2"]["fragments"]["cluster_pointers"]
fragment_clusters = df.iloc[fragment_indices]

fragment_clusters.loc[:,"quad_low_mz_values"] = np.array(data.as_dataframe(np.array(fragment_clusters.apex_pointer,dtype=int)).quad_low_mz_values)
fragment_clusters.loc[:,"quad_high_mz_values"] = np.array(data.as_dataframe(np.array(fragment_clusters.apex_pointer,dtype=int)).quad_high_mz_values)


#"""


# key_mz = {i:librarySpectra[i]["prec_mz"] for i in librarySpectra}
# diann_key_mz = {i:diannLibrarySpectra[i]["prec_mz"] for i in diannLibrarySpectra}

# shared_keys = set(key_mz).intersection(set(diann_key_mz))

# dia_spectra = DIAspectra
# librarySpectra=spectrumLibrary#spec_lib#
"""
def RTfit(dia_spectra,librarySpectra,mz_tol):
    
    print(f"Fitting the {config.n_most_intense} most intense spectra")
        
    totalIC = np.array([np.sum(i.intens) for i in dia_spectra])
    
    top_n = np.argsort(-totalIC)[:config.n_most_intense]
    all_keys = list(librarySpectra)
    rt_mz = np.array([[i["iRT"], i["prec_mz"]] for i in librarySpectra.values()])
    
    # input_data = [(idx, 
    #                 dia_spectra[idx],
    #                 librarySpectra,
    #                 mz_tol,
    #                 22.013000488281023,
    #                 10,
    #                 5) for idx in top_n]
    
    output_rts =[]
    dia_rt = []
    lib_rt = []
    top_n_spectra = [dia_spectra[i] for i in top_n]
    with multiprocessing.Pool(config.numProc) as p:
        fit_outputs = list(tqdm.tqdm(p.imap(partial(fit_to_lib,
                                                    library=librarySpectra,
                                                    rt_mz=rt_mz,
                                                    all_keys=all_keys,
                                                    dino_features=None,
                                                    rt_filter=False),
                                            top_n_spectra,chunksize=len(top_n_spectra)//config.numProc),total=len(top_n_spectra)))
        
    output=[]
    for idx,i in enumerate(top_n):
        # fit_output = fit_to_lib(inputs)
        fit_output = fit_to_lib(dia_spectra[i],
                                library=librarySpectra,
                                rt_mz=rt_mz,
                                all_keys=all_keys,
                                dino_features=None,
                                rt_filter=False)
    # for idx,fit_output in enumerate(fit_outputs):    
        if fit_output[0][0]!=0:
            lib_rt.append([librarySpectra[(i[2],i[3])]["iRT"] for i in fit_output])
            dia_rt.append(top_n_spectra[idx].RT)
            output.append(fit_output)
            max_id = np.argmax([i[0] for i in fit_output])
            
    def max_coeff_rt(outputs):
        max_id = np.argmax([i[0] for i in outputs])
        # if outputs[0][0]==0:
        #     return np.nan
        # else:
        return librarySpectra[(outputs[max_id][2],outputs[max_id][3])]["iRT"]
    
    
    output_rts = np.array([max_coeff_rt(i) for i in output])
    
    ## 2 step fitting
    sorted_idxs = np.argsort(output_rts)
    sort_rts = np.array(output_rts)[sorted_idxs]
    sort_dia_rts = np.array(dia_rt)[sorted_idxs]
    knots = quantiles(sort_rts,n=2)
    spl = spline(sort_rts,sort_dia_rts,knots)
    
    # find outliers and remove
    _bool = abs(spl(sort_rts)-sort_dia_rts)<20
    spl2 = spline(sort_rts[_bool],sort_dia_rts[_bool],knots)
    
    plt.scatter(sort_rts,sort_dia_rts)
    plt.scatter(sort_rts,spl(sort_rts))
    # plt.show()
    
    return spl2
"""
# p = np.poly1d(np.polyfit(dia_rt,output_rts,6))
# plt.scatter(diann_report.RT,diann_report.iRT,s=.1)
# plt.scatter(dia_rt,p(dia_rt))
# plt.scatter(dia_rt,output_rts,s=5)

# # ##opp way around
# p = np.poly1d(np.polyfit(output_rts,dia_rt,6))
# plt.scatter(diann_report.iRT,diann_report.RT,s=.1)

# plt.scatter(x,spl2(x),s=5)
# plt.scatter(output_rts,p2(output_rts))
# plt.scatter(output_rts,dia_rt,s=5)

# x = np.linspace(min(output_rts),max(output_rts),100)
# plt.scatter(x-10,p2(x),s=5)
# plt.scatter(x+10,p2(x),s=5)

# sorted_idxs = np.argsort(output_rts)
# sort_rts = np.array(output_rts)[sorted_idxs]
# knots = quantiles(sort_rts,n=2)
# spl = spline(sort_rts,np.array(dia_rt)[sorted_idxs],knots)

# _bool = abs(spl(sort_rts)-np.array(dia_rt)[sorted_idxs])<20
# spl2 = spline(sort_rts[_bool],np.array(dia_rt)[sorted_idxs][_bool],knots)

# # plt.scatter(sort_rts,np.array(dia_rt)[sorted_idxs])
# # plt.scatter(sort_rts,spl(sort_rts))
# # plt.scatter(sort_rts[_bool],np.array(dia_rt)[sorted_idxs][_bool])

# plt.scatter(diann_report.iRT,diann_report.RT,s=.1)

# plt.scatter(x,spl2(x),s=5)
                            

def twostepfit(x,y,n_knots=2,z=None,k1=1):
    if z is None:
        z= np.ones_like(x)
    y_exists = np.isfinite(y)
    x_exists = np.isfinite(x)*y_exists
    x=np.array(x)[x_exists]
    y=np.array(y)[x_exists]
    z=np.array(z)[x_exists]
    y_range = np.max(y)-np.min(y)
    sorted_idxs = np.argsort(x)
    sort_x = np.array(x)[sorted_idxs]
    sort_y = np.array(y)[sorted_idxs]
    sort_z = np.array(z)[sorted_idxs]
    knots = quantiles(sort_x,n=n_knots)
    spl = spline(sort_x,sort_y,knots,w=sort_z,k=k1)
    # plt.scatter(x,y,s=1)
    # plt.scatter(x,spl(x),s=1)
    # find outliers and remove; points over 1/4 of the y range away from prediction
    _bool = abs(spl(sort_x)-sort_y)<(y_range/4)
    spl2 = spline(sort_x[_bool],sort_y[_bool],knots,w=sort_z[_bool])
    # plt.scatter(sort_x[_bool],sort_y[_bool],c=np.log10(sort_z[_booxl]),s=1)
    # plt.scatter(sort_x[_bool],spl2(sort_x[_bool]),s=1)
    # plt.scatter(x,y,s=1)
    # plt.scatter(x,spl2(x),s=1)
    return spl2

def threestepfit(x,y,n_knots=2,z=None,k1=1):
    if z is None:
        z= np.ones_like(x)
    y_exists = np.isfinite(y)
    x_exists = np.isfinite(x)*y_exists
    x=np.array(x)[x_exists]
    y=np.array(y)[x_exists]
    z=np.array(z)[x_exists]
    y_range = np.max(y)-np.min(y)
    sorted_idxs = np.argsort(x)
    sort_x = np.array(x)[sorted_idxs]
    sort_y = np.array(y)[sorted_idxs]
    sort_z = np.array(z)[sorted_idxs]
    knots = quantiles(sort_x,n=n_knots)
    spl = spline(sort_x,sort_y,knots,w=sort_z,k=1)
    # poly = np.polyfit(sort_x, sort_y, w=sort_z, deg=5)
    # sort_x+=np.arange(len(sort_x))*1e-7
    # spl  = InterpolatedUnivariateSpline(sort_x,sort_y,w=np.log10(sort_z),k=5)
    # plt.plot(sort_x,np.polyval(poly, sort_x))
    # plt.scatter(x,y,s=1)
    # plt.scatter(x,spl(x),s=1)
    # find outliers and remove; points over 1/4 of the y range away from prediction
    _bool = abs(spl(sort_x)-sort_y)<(y_range/4)
    # knots = quantiles(sort_x,n=4)
    spl2 = spline(sort_x[_bool],sort_y[_bool],knots,w=sort_z[_bool])
    # spl2 = UnivariateSpline(sort_x,sort_y)
    # plt.scatter(sort_x[_bool],sort_y[_bool],s=1)
    # plt.scatter(x,spl2(x),s=1)
    
    _bool = abs(spl2(sort_x)-sort_y)<(y_range/8)
    
    # knots = quantiles(sort_x,n=n_knots)
    spl3 = spline(sort_x[_bool],sort_y[_bool],knots,w=sort_z[_bool])
    # plt.scatter(sort_x[_bool],sort_y[_bool],s=1)
    # plt.scatter(sort_x[_bool],spl3(sort_x[_bool]),s=1)
    
    return spl3


from numpy.polynomial import legendre as leg
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

def sgd_fit(x,y,z=None,deg=2,penalty="l1"):
    if z is None:
        z= np.ones_like(x)
    y_exists = np.isfinite(y)
    x_exists = np.isfinite(x)*y_exists
    x=np.array(x)[x_exists]
    y=np.array(y)[x_exists]
    z=np.array(z)[x_exists]
    
    sorted_idxs = np.argsort(x)
    sort_x = np.array(x)[sorted_idxs]
    sort_y = np.array(y)[sorted_idxs]
    sort_z = np.array(z)[sorted_idxs]
    
    
    sort_x = StandardScaler().fit_transform(sort_x[:,np.newaxis])
    V=leg.legvander(sort_x.flatten(),deg)
    
    
    sgdr = SGDRegressor(loss="huber",max_iter=10000,alpha=0.0001,penalty=penalty,learning_rate="adaptive")
    sgdr.fit(V, sort_y)
    
    # plt.scatter(sort_x,sort_y,s=1)
    # plt.scatter(sort_x,sgdr.predict(V),s=1)
    # plt.yscale("log")
    # plt.ylim(0,70)
    
    def spl(x_vals):
        V_vals = leg.legvander(x_vals,deg)
        return sgdr.predict(V_vals)
    
    return spl

def initstepfit(x,y,n_knots=2,z=None,k1=1):
    ### like above but initial guess is just a straight line from [min_x,min_] to [max_x,max_y]
    if z is None:
        z= np.ones_like(x)
    y_exists = np.isfinite(y)
    x_exists = np.isfinite(x)*y_exists
    x=np.array(x)[x_exists]
    y=np.array(y)[x_exists]
    z=np.array(z)[x_exists]
    y_range = np.max(y)-np.min(y)
    x_range = np.max(x)-np.min(x)
    sorted_idxs = np.argsort(x)
    sort_x = np.array(x)[sorted_idxs]
    sort_y = np.array(y)[sorted_idxs]
    sort_z = np.array(z)[sorted_idxs]
    knots = quantiles(sort_x,n=n_knots)
    # spl = spline(sort_x,sort_y,knots,w=sort_z,k=1)
    # plt.scatter(x,y,s=1)
    # plt.scatter(x,spl(x),s=1)
    # plt.plot(x,((y_range/x_range)*x)+min(y)-((y_range/x_range)*min(x)))
    _bool = np.abs((((y_range/x_range)*sort_x)+min(y)-((y_range/x_range)*min(x)))-sort_y)<(y_range/4)
    # plt.scatter(sort_x[_bool],sort_y[_bool],s=1)
    # find outliers and remove; points over 1/4 of the y range away from prediction
    # _bool = np.abs(spl(sort_x)-sort_y)<(y_range/4)
    # knots = quantiles(sort_x,n=4)
    spl2 = spline(sort_x[_bool],sort_y[_bool],knots,w=sort_z[_bool])
    # spl2 = UnivariateSpline(sort_x,sort_y)
    # plt.scatter(sort_x[_bool],sort_y[_bool],s=1)
    # plt.scatter(x,spl2(x),s=1)
    
    _bool = np.abs(spl2(sort_x)-sort_y)<(y_range/8)
    
    # knots = quantiles(sort_x,n=n_knots)
    spl3 = spline(sort_x[_bool],sort_y[_bool],knots,w=sort_z[_bool])
    # plt.scatter(sort_x[_bool],sort_y[_bool],s=1)
    # plt.scatter(sort_x[_bool],spl3(sort_x[_bool]),s=1)
    
    return spl3

# from scipy.interpolate import interp1d
# import statsmodels.api as sm
# x = output_rts
# y = dia_rt

# all_id_rt = [[(i[j][2],i[j][3]),i[j][5],i[j][18]] for i in output for j in range(len(i)) if i[j][0]>10 and i[j][18]>15]
# all_lib_rts = np.array([librarySpectra[i[0]]["iRT"] for i in all_id_rt])

# x = all_lib_rts
# y = [i[1] for i in all_id_rt]

# plt.scatter(x,y,s=1)

# lowess = sm.nonparametric.lowess(y, x, frac=.75)

# # unpack the lowess smoothed points to their values
# lowess_x = list(zip(*lowess))[0]
# lowess_y = list(zip(*lowess))[1]

# # run scipy's interpolation. There is also extrapolation I believe
# f = interp1d(lowess_x, lowess_y, bounds_error=False)

# xnew = [i/10. for i in range(800)]
# xnew = np.linspace(min(x),max(x),100)

# # this this generate y values for our xvalues by our interpolator
# # it will MISS values outsite of the x window (less than 3, greater than 33)
# # There might be a better approach, but you can run a for loop
# #and if the value is out of the range, use f(min(lowess_x)) or f(max(lowess_x))
# ynew = f(xnew)


# plt.scatter(x, y,s=1)
# # plt.plot(lowess_x, lowess_y, '*')
# plt.plot(xnew, ynew, '-')


def get_diff(mz,peaks,window,tol):
    
    log_diff = within_tol(mz, peaks, atol=0, rtol=tol) 
    idxs = np.where(log_diff[...,0])[0]
    
    
    # if a match
    if idxs.size > 0:
        
        # in case of multiple matches
        # select idx with smallest error
        closest_idx = idxs[np.argmin(log_diff[idxs,1])]
        
        return (peaks[closest_idx]-mz)/mz
    
    else:
        return np.nan
            

def lin_func(x,a,b,c):
    return (a*np.array(x[0]))+(b*np.array(x[1]))+c

def curve_param(x,y,z,func=lin_func):
    # z are predictions
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    
    z_exists = np.isfinite(z) # some diffs may be nans
    parameters, covariance = curve_fit(func, [x[z_exists],y[z_exists] ], z[z_exists])
    return parameters

def contains_peak(window,mz):
    return window[0]<mz<window[1]

def max_intens(peaks,window):
    
    window_peaks = peaks[np.logical_and(peaks[:,0]>window[0],peaks[:,0]<window[1])]
                                        
    if len(window_peaks)>0:
        return window_peaks[np.argmax(window_peaks[:,1])]
    else:
        return np.zeros(2)
    
# # find largest peak in each spectrum
# def get_largest(dia_spectra):
    
#     ms1spectra = dia_spectra.ms1scans
#     ms2spectra = dia_spectra.ms2scans
#     all_windows = [tuple(i.ms1window) for i in ms2spectra]
#     set_windows = sorted(set(all_windows))
#     all_windows = np.stack(all_windows)
#     min_mz = np.min(all_windows[:,0])
#     max_mz = np.max(all_windows[:,1])
    
#     ms2_ms1_ratio = round(len(ms2spectra)/len(ms1spectra))
#     mzs=[]
#     for idx,spec in enumerate(ms1spectra[:100]):
#         _bool = np.logical_and(spec.mz>min_mz,spec.mz<max_mz)
#         max_intens_idx = np.argmax(spec.intens[_bool])
#         max_intens =  spec.intens[_bool][max_intens_idx]
#         mz_of_max = spec.mz[_bool][max_intens_idx]
#         mzs.append(mz_of_max)
#         # ms2_idxs = np.arange(idx*ms2_ms1_ratio,(idx+1)*ms2_ms1_ratio)
#         # windows = [contains_peak(ms2spectra[i].ms1window,mz_of_max) for i in ms2_idxs]
#         # np.where(windows)[0][0]
        
# find largest peak in each spectrum
def get_largest(dia_spectra):
    
    ms1spectra = dia_spectra.ms1scans
    ms2spectra = dia_spectra.ms2scans
    all_windows = [tuple(i.ms1window) for i in ms2spectra]
    set_windows = sorted(set(all_windows))
    all_windows = np.stack(all_windows)
    min_mz = np.min(all_windows[:,0])
    max_mz = np.max(all_windows[:,1])
    
    # num_windows if non-overalpping
    ms2_ms1_ratio = round(len(ms2spectra)/len(ms1spectra))
    peaks_per_window = config.n_most_intense//ms2_ms1_ratio
   
    mp_window_mp_spec = np.array([np.append(np.pad([max_intens(i.peak_list().T,window) for window in all_windows[idx*ms2_ms1_ratio:(idx+1)*ms2_ms1_ratio]],((0,ms2_ms1_ratio),(0,0)))[:ms2_ms1_ratio],
                                            np.ones(ms2_ms1_ratio)[:,np.newaxis]*idx,1) 
                                  for idx,i in enumerate(ms1spectra)])
    
    top_window = np.argsort(-mp_window_mp_spec[:,:,1],axis=0)[:peaks_per_window]
    top_mzs = np.array([i[j] for i,j in zip(np.transpose(mp_window_mp_spec,[1,0,2]),top_window.T)])
    top_mzs[:,:,2] = (top_mzs[:,:,2]*ms2_ms1_ratio)+np.arange(ms2_ms1_ratio)[:,np.newaxis]
    return top_mzs.reshape(-1,3)

# def spectra_largest_features(dia_spectra,features):
    
    
def closest_feature(mz,rt,dino_features,rt_tol,mz_tol):
    
    _bool = np.logical_and(dino_features.rtStart<(rt+rt_tol),(rt-rt_tol)<dino_features.rtEnd)
    
    feature_mzs = np.array(dino_features.mz[_bool])
    
    log_diff = within_tol(mz, feature_mzs, atol=0, rtol=mz_tol) 
    idxs = np.where(log_diff[...,0])[0]
    
    
    # if a match
    if idxs.size > 0:
        
        # in case of multiple matches
        # select idx with smallest error
        closest_idx = idxs[np.argmin(log_diff[idxs,1])]
        
        return (feature_mzs[closest_idx]-mz)/mz
    
    else:
        return np.nan
        
def closest_feature2(mz,rt,dino_features,rt_tol,mz_tol):
    
    _bool = np.logical_and(dino_features.rtStart<(rt+rt_tol),(rt-rt_tol)<dino_features.rtEnd)
    
    feature_mzs = np.array(dino_features.mz[_bool])
    
    log_diff = within_tol(mz, feature_mzs, atol=0, rtol=mz_tol) 
    idxs = np.where(log_diff[...,0])[0]
    
    
    # if a match
    if idxs.size > 0:
        
        # in case of multiple matches
        # select idx with smallest error
        closest_idx = idxs[np.argmin(log_diff[idxs,1])]
        
        return feature_mzs[closest_idx]
    
    else:
        return np.nan
    
def get_tol(x):
    x= x[np.isfinite(x)]
    max_scale = np.max(x)
    min_scale = np.min(x)
    bin_size = (max_scale-min_scale)/300 # found to work well but not optimised
    bins = np.arange(min_scale,max_scale+bin_size,bin_size)
    vals,bins = np.histogram(x,bins)
    smooth_vals = moving_average(vals, 10) # found to work well but not optimised
    _,peak=signal.find_peaks(smooth_vals,height=max(smooth_vals),width=1,rel_height=.5)
    return(peak["widths"][0]*bin_size*2) # twice fwhm

def closest_spec(dia_rt_mzwin,mz,rt):
    
    _bool = np.logical_and(dia_rt_mzwin[:,1]<mz,mz<dia_rt_mzwin[:,2])
    contender_idxs = np.where(_bool)
    contenders = dia_rt_mzwin[_bool]
    closest_idx = contender_idxs[0][np.argmin(np.abs(contenders[:,0]-rt))]
    return closest_idx
    # try:
    #     closest_idx = contender_idxs[0][np.argmin(np.abs(contenders[:,0]-rt))]
    #     return closest_idx
    # except:
    #     print(mz,rt)
    #     return 0
    

def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / 2 / stddev) ** 2)

def fwhm(stddev):
    return 2 * np.sqrt(2 * np.log(2)) * stddev

# NB: Need to make the folowing changes to this code
## if there is a background uniform distribution, it will not fit the gaussian well
# therefore we can subtract the min val in all bins fram all bins and then fit
### This seems to work for some data but need to robustly test
def fit_gaussian(data,init_std=None):
    
    data = np.array(data)
    data = data[~np.isnan(data)]
    # Create a histogram
    hist, bin_edges = np.histogram(data, bins=50, density=True)
    
    ### Need to test
    # background = np.min(hist)
    # hist-=background
    
    # Find peaks in the histogram
    # peaks, _ = signal.find_peaks(hist, height=0.01, distance=10)
    peaks, _ = signal.find_peaks(hist, height=max(hist)*0.5, distance=10)
    
    # Find the highest peak
    highest_peak_index = np.argmax(hist[peaks])
    highest_peak_height = hist[peaks][highest_peak_index]
    highest_peak_x = bin_edges[peaks][highest_peak_index]
    
    # Calculate the width of the highest peak using Gaussian fit
    # split bins in 2 to get x values
    x_data = (bin_edges[:-1] + bin_edges[1:]) / 2
    y_data = hist
    
    if init_std is None:
        init_std = 2*np.subtract(*bin_edges[1::-1])
    
    fit_params, _ = curve_fit(gaussian, x_data, y_data, p0=[highest_peak_height, highest_peak_x, init_std])
    
    return fit_params#, background





##################################################################################################################################
##################################################################################################################################
##################################################################################################################################


                                        



def MZRTfit(dia_spectra,librarySpectra,dino_features,mz_tol,ms1=False,results_folder=None,ms2=False):
    ## for testing
    # mz_tol,ms1,results_folder,ms2 = (config.ms1_tol,False,None,False) 
    # here spectra are both ms1 and ms2 
    
    scans_per_cycle = round(len(dia_spectra.ms2scans)/len(dia_spectra.ms1scans))
    print("Intitial search")
    # print(f"Fitting the {config.n_most_intense} most intense spectra")
    
    ms1spectra = dia_spectra.ms1scans
    ms2spectra = dia_spectra.ms2scans
    
    ms1_rt = np.array([i.RT for i in ms1spectra])
    
    totalIC = np.array([np.sum(i.intens) for i in ms2spectra])
    
    num_partition = 10
    split_size = int(np.ceil(len(totalIC)/num_partition))
    split_tic = [totalIC[i*split_size:(i+1)*split_size] for i in range(num_partition)]
    split_top_n = [(np.argsort(-tics)+(idx*split_size))[:(config.n_most_intense//num_partition)] for idx,tics in enumerate(split_tic)]
    top_n = np.concatenate(split_top_n)
    
    # top_n = np.argsort(-totalIC)[:config.n_most_intense]
    top_n_ms1 = top_n//scans_per_cycle
    all_keys = list(librarySpectra)
    rt_mz = np.array([[i["iRT"], i["prec_mz"]] for i in librarySpectra.values()])

    top_n_spectra = [ms2spectra[i] for i in top_n]
    
    # # with multiprocessing.Pool(config.numProc) as p:
    # #     fit_outputs = list(tqdm.tqdm(p.imap(partial(fit_to_lib,
    # #                                                 library=librarySpectra,
    # #                                                 rt_mz=rt_mz,
    # #                                                 all_keys=all_keys,
    # #                                                 dino_features=None,
    # #                                                 rt_filter=False),
    # #                                         top_n_spectra,chunksize=len(top_n_spectra)//config.numProc),total=len(top_n_spectra)))
        
   
    
    # """
    
    
    ### redefine "top_n_spectra" to evenly span Rt and m/z
    np.random.seed(0)
    top_n = np.random.choice(np.arange(len(ms2spectra)),config.n_most_intense,replace=False)
    
    
    fit_outputs=[]
    
    frags = []
    for idx in tqdm.trange(len(top_n)):
        if ms2:
            fit_output,frag_errors = fit_to_lib(top_n_spectra[idx],
                                    library=librarySpectra,
                                    rt_mz=rt_mz,
                                    all_keys=all_keys,
                                    dino_features=None,
                                    rt_filter=False,
                                    return_frags=True,
                                    ms1_spectra = ms1spectra
                                    )
            frags.append(frag_errors)
        else:
            fit_output = fit_to_lib(top_n_spectra[idx],
                                    library=librarySpectra,
                                    rt_mz=rt_mz,
                                    all_keys=all_keys,
                                    dino_features=None,
                                    rt_filter=False,
                                    return_frags=False,
                                    ms1_spectra = ms1spectra
                                    )
        fit_outputs.append(fit_output)
    fit_outputs1=fit_outputs
    # """
    #################################################################################
    """
    all_dia_rt = [i.RT for i in ms2spectra]
    all_dia_windows = np.array([i.ms1window for i in ms2spectra])
    lowest_mz = np.min(all_dia_windows,0)[0] # assume window span is constant over time
    largest_mz = np.max(all_dia_windows,0)[1]
    mz_bins = np.linspace(lowest_mz,largest_mz,6)
    
    ## remove charge 1+ features
    dino_features = dino_features[dino_features["charge"]!=1]
    dino_features = dino_features.reset_index(drop=True)
    sorted_features = np.argsort(-np.array(dino_features.intensityApex))
    sorted_mz = dino_features.mz[sorted_features]
    large_feature_indices = sorted_features[np.array(np.logical_and(sorted_mz>lowest_mz,sorted_mz<largest_mz))][:config.n_most_intense_features] 
    
    sorted_feature_mz_bins = [sorted_features[np.logical_and(sorted_mz>mz_bins[i],sorted_mz<mz_bins[i+1])] for i in range(len(mz_bins)-1)]
    large_feature_indices = [j for i in sorted_feature_mz_bins for j in i[:(config.n_most_intense_features//(len(mz_bins)-1))]]
    
    lf_rt = np.array(dino_features.rtApex[large_feature_indices])
    lf_mz = np.array(dino_features.mz[large_feature_indices])
    print("Finding correct spectra")
    # lf_spectra = [np.argmin(np.abs(np.array(all_dia_rt)-i)) for i in lf_rt]
    dia_rt_mzwin = np.array([[i.RT,*i.ms1window] for i in ms2spectra])
    lf_spectra = [closest_spec(dia_rt_mzwin,i,j) for i,j in zip(lf_mz,lf_rt)] 
    # mz_int_n = get_largest(dia_spec,tra)
    fit_outputs2=[]
    frags = []
    for idx in tqdm.trange(len(lf_spectra)):
        if ms2:
            fit_output,frag_errors = fit_to_lib(ms2spectra[int(lf_spectra[idx])],
                                                library=librarySpectra,
                                                rt_mz=rt_mz,
                                                all_keys=all_keys,
                                                dino_features=None,
                                                rt_filter=False,
                                                ms1_mz=lf_mz[idx],
                                                return_frags = True,
                                                ms1_spectra = ms1spectra
                                                )
            
            frags.append(frag_errors)
        else:
            
            fit_output = fit_to_lib(ms2spectra[int(lf_spectra[idx])],
                                    library=librarySpectra,
                                    rt_mz=rt_mz,
                                    all_keys=all_keys,
                                    dino_features=None,
                                    rt_filter=False,
                                    ms1_mz=lf_mz[idx],
                                    ms1_spectra = ms1spectra
                                    )
        fit_outputs2.append(fit_output)
        
    fit_outputs = fit_outputs2
    top_n_spectra = [ms2spectra[i] for i in lf_spectra]
    # """
     ########################################################################
     
     
    dia_rt = []
    lib_rt = []
    output=[]
    max_ids=[]
    lc_frags_errors=[]
    lc_frags=[]
    for idx,fit_output in enumerate(fit_outputs):    
        if fit_output[0][0]!=0:
            lib_rt.append([librarySpectra[(i[2],i[3])]["iRT"] for i in fit_output])
            dia_rt.append(top_n_spectra[idx].RT)
            output.append(fit_output)
            max_id = np.argmax([i[0] for i in fit_output])
            max_ids.append(max_id)
            if ms2:
                lc_frags_errors.append(frags[idx][0][max_id])
                lc_frags.append(frags[idx][1][max_id])
        
    # max_ids = [np.argmax([i[0] for i in j]) for j in output]
    ms1windows = [i.ms1window for i in top_n_spectra]
    id_keys = [(i[j][2],i[j][3]) for i,j in zip(output,max_ids)]
    id_mzs = [librarySpectra[i]["prec_mz"] for i in id_keys]
    
    # plt.hist(np.log10([i[j][0] for i,j in zip(output,max_ids)]),np.arange(1,9,.3))
    # plt.xlabel("log10(Coefficients)")
    # plt.ylabel("Frequency")
    
    # plt.scatter(dino_features.mz,dino_features.rtApex,s=.1)
    # plt.ylabel("Retention time")
    # plt.xlabel("m/z")
    
    min_int = 100#np.median([j[0] for i in output for j in i])
    
    all_id_rt = [[(i[j][2],i[j][3]),i[j][5]] for i in output for j in range(len(i)) if i[j][0]>min_int]
    all_coeff = [i[j][0] for i in output for j in range(len(i)) if i[j][0]>min_int]
    all_id_mzs = [librarySpectra[i[0]]["prec_mz"] for i in all_id_rt]
    
    all_hyper = [i[j][18] for i in output for j in range(len(i)) if i[j][0]>min_int]
    
    def max_coeff_rt(outputs):
        max_id = np.argmax([i[0] for i in outputs])
        # if outputs[0][0]==0:
        #     return np.nan
        # else:
        return librarySpectra[(outputs[max_id][2],outputs[max_id][3])]["iRT"]
    
    
    output_rts = np.array([max_coeff_rt(i) for i in output])
    output_coeff = np.array([i[j][0] for i,j in zip(output,max_ids)])
    output_hyper = np.array([i[j][18] for i,j in zip(output,max_ids)])
    all_lib_rts = np.array([librarySpectra[i[0]]["iRT"] for i in all_id_rt])
    
    
    # plt.scatter(all_lib_rts,[i[1] for i in all_id_rt],label="Original_RT",s=1)
    
    ## 2 step fitting
    # rt_spl = threestepfit(output_rts,dia_rt,1)
    # rt_spl = threestepfit(output_rts,dia_rt,1,z=np.log10(output_coeff))
    # rt_spl = threestepfit(output_rts,dia_rt,1,z=output_hyper)
    rt_spl = threestepfit(all_lib_rts,[i[1] for i in all_id_rt],1,z=all_hyper)
    # rt_spl = threestepfit(output_rts,dia_rt,1,z=output_coeff)
    # rt_spl = threestepfit(all_lib_rts,[i[1] for i in all_id_rt],1,z=all_coeff)
    # rt_spl = initstepfit(output_rts,dia_rt,1,z=np.log10(output_coeff))
    # rt_spl = initstepfit(all_lib_rts,[i[1] for i in all_id_rt],1,z=np.log10(all_coeff))
    rt_spl = initstepfit(all_lib_rts,[i[1] for i in all_id_rt],1,z=all_hyper)
    # rt_spl = sgd_fit(output_rts,dia_rt)
    
    # plt.scatter(output_rts,dia_rt,label="Original_RT",s=1)#0,c=output_hyper)
    # plt.scatter(output_rts,rt_spl(output_rts),label="Predicted_RT",s=1)
    # plt.xlabel("Library RT");plt.ylabel("Observed RT");
    
    # plt.scatter(all_lib_rts,[i[1] for i in all_id_rt],label="Original_RT",s=1)
    # plt.scatter(all_lib_rts,rt_spl(all_lib_rts),label="Predicted_RT",s=1)
    
    converted_rt = rt_spl(output_rts)
    
    rt_amplitude, rt_mean, rt_stddev = fit_gaussian(dia_rt-converted_rt)
    # rt_spl = twostepfit(all_lib_rts,[i[1] for i in all_id_rt]) # does not work
    
    # mz_spl = twostepfit(id_mzs, diffs)
    # mz_spl = twostepfit(output_rts, diffs)
    
    ## old version (failed)
    # diffs = [get_diff(mz, ms1spectra[ms1_idx].mz, window, mz_tol) for  mz,ms1_idx,window in zip(id_mzs,top_n_ms1,ms1windows)]
    
    
    # all_diffs = [closest_feature(all_id_mzs[i],all_id_rt[i][1],dino_features,1,20*1e-6) for i in range(len(all_id_mzs))]
    # all_coeffs = [closest_feature(all_id_mzs[i],all_id_rt[i][1],dino_features,1,10*1e-6) for i in range(len(all_id_mzs))]
    
    
    resp_ms1scans = [closest_ms1spec(dia_rt[i], ms1_rt) for i in range(len(dia_rt))]
    diffs = [closest_peak_diff(mz, ms1spectra[i].mz) for i,mz in zip(resp_ms1scans,id_mzs)]
    
    # diffs = [closest_feature(id_mzs[i],dia_rt[i],dino_features,0,10*1e-6) for i in range(len(id_mzs))]
    
    # feature_mz = [closest_feature2(id_mzs[i],converted_rt[i],dino_features,1,30*1e-6) for i in range(len(id_mzs))]
    mz_spl = twostepfit(id_mzs,diffs,1)
    # mz_spl = twostepfit(id_mzs,diffs,1,z=np.log10(output_coeff))
    
    # plt.scatter(all_id_mzs,[i for i in all_diffs],label="Original_RT",s=1)


    def mz_func(mz):
        return mz+(mz_spl(mz)*mz)
    
    mz_amplitude, mz_mean, mz_stddev = fit_gaussian(diffs-mz_spl(id_mzs))
    
    
    ### MS2 alignment
    if ms2:
        all_frag_errors = np.concatenate(lc_frags_errors)
        all_frags = np.concatenate(lc_frags)
        ms2_spl = twostepfit(all_frags,all_frag_errors,1)
        def ms2_func(mz):
            return mz+(ms2_spl(mz)*mz)
        
        ms2_amplitude, ms2_mean, ms2_stddev = fit_gaussian(all_frag_errors-ms2_spl(all_frags))
    
    # plt.hist(diffs,50)
    # plt.subplots()
    # plt.scatter(id_mzs,np.array(diffs)*1e6,s=1)
    # plt.xlabel("m/z")
    # plt.ylabel("m/z error (ppm)")
    # plt.subplots()
    # plt.scatter(dia_rt,np.array(diffs)*1e6,s=1)
    # plt.xlabel("RT")
    # plt.ylabel("m/z error (ppm)")
    
    # ## 2D plane fitting
    # function = lin_func
    # parameters = curve_param(output_rts, id_mzs, diffs,func=function)
    
    # def mz_func(mz,rt):
    #     return mz+(function([rt,mz],*parameters)*mz)
    
    # set optimised rt tol to 95th percentile of search (assumes a few outliers)
    # buffer = 1.2
    # config.opt_rt_tol = np.round(np.sort(np.abs(dia_rt-rt_spl(output_rts)))[int(config.n_most_intense*.95)]*buffer,5) 
    
    # new_rt_tol = get_tol(dia_rt-rt_spl(output_rts))
    new_rt_tol = 4*np.abs(rt_stddev) 
    print(f"Optimsed RT tolerance: {new_rt_tol}")
    config.opt_rt_tol = new_rt_tol
    
    # set optimised ms2 tol
    # is_real = ~np.isnan(diffs)
    # buffer = 1.2
    # config.opt_ms1_tol = np.round(
    #                         np.sort(
    #                             np.abs(((np.array(id_mzs)+np.array(diffs)*id_mzs)-mz_func(id_mzs, output_rts))/id_mzs
    #                                    )[is_real])[int(sum(is_real)*.95)]*buffer,6+5)#6 for 1e-6 the 5 decimal places

    # new_ms1_tol = get_tol(((np.array(id_mzs)+np.array(diffs)*id_mzs)-mz_func(id_mzs, output_rts))/id_mzs)
    # new_ms1_tol = get_tol(diffs-mz_spl(id_mzs))
    new_ms1_tol = 4*mz_stddev
    print(f"Optimsed ms1 tolerance: {new_ms1_tol}")
    
    # config.opt_ms1_tol  = new_ms1_tol
    
    if ms2:
        new_ms2_tol = 4*ms2_stddev
        config.opt_ms2_tol  = new_ms2_tol
    
    if results_folder is not None:
        
        ### Save functions
        with open(results_folder+"/rt_spl","wb") as dill_file:
            dill.dump(rt_spl,dill_file)
            
        with open(results_folder+"/mz_func","wb") as dill_file:
            dill.dump(mz_func,dill_file)
        
        if ms2:
            with open(results_folder+"/ms2_func","wb") as dill_file:
                dill.dump(ms2_func,dill_file)
            
        ##plot RT alignment
        plt.subplots()
        plt.scatter(output_rts,dia_rt,label="Original_RT",s=1)
        plt.scatter(output_rts,rt_spl(output_rts),label="Predicted_RT",s=1)
        # plt.legend()
        plt.xlabel("Library RT")
        plt.ylabel("Observed RT")
        # plt.show()
        plt.savefig(results_folder+"/RTfit.png",dpi=600,bbox_inches="tight")
        
        
        plt.subplots()
        plt.scatter(output_rts,dia_rt-rt_spl(output_rts),label="Original_RT",s=1)
        plt.plot([min(output_rts),max(output_rts)],[0,0],color="r",linestyle="--",alpha=.5)
        plt.plot([min(output_rts),max(output_rts)],[config.opt_rt_tol,config.opt_rt_tol],color="g",linestyle="--",alpha=.5)
        plt.plot([min(output_rts),max(output_rts)],[-config.opt_rt_tol,-config.opt_rt_tol],color="g",linestyle="--",alpha=.5)
        # plt.scatter(output_rts,rt_spl(output_rts),label="Predicted_RT",s=1)
        # plt.legend()
        plt.xlabel("Library RT")
        plt.ylabel("RT Residuals")
        # plt.show()
        plt.savefig(results_folder+"/RtResidual.png",dpi=600,bbox_inches="tight")
        
        
        plt.subplots()
        vals,bins,_ = plt.hist(dia_rt-rt_spl(output_rts),100)
        plt.vlines([-config.opt_rt_tol,config.opt_rt_tol],0,max(vals),color="r")
        plt.vlines([-4*rt_stddev,4*rt_stddev],0,max(vals),color="g")
        plt.text(config.opt_rt_tol,max(vals),np.round(config.opt_rt_tol,2))
        plt.xlabel("RT difference")
        plt.ylabel("Frequency")
        # plt.show()
        plt.savefig(results_folder+"/RTdiff.png",dpi=600,bbox_inches="tight")
        
        
        ##plot mz alignment
        plt.subplots()
        plt.scatter(id_mzs,diffs,label="Original_MZ",s=1)
        plt.scatter(id_mzs,mz_spl(id_mzs),label="Predicted_MZ",s=1)
        # plt.legend()
        plt.xlabel("m/z")
        plt.ylabel("m/z difference (relative)")
        # plt.show()
        plt.savefig(results_folder+"/MZfit.png",dpi=600,bbox_inches="tight")
        
        
        ## plot mz alignment
        plt.subplots()
        plt.hist(diffs,100)
        # plt.hist(((np.array(id_mzs)+np.array(diffs)*id_mzs)-mz_func(id_mzs, output_rts))/id_mzs,100,alpha=.5)
        # plt.hist(((np.array(id_mzs)+np.array(diffs)*id_mzs)-mz_spl(id_mzs))/id_mzs,100,alpha=.5)
        plt.hist(diffs-mz_spl(id_mzs),100,alpha=.5)
        plt.vlines([-config.opt_ms1_tol,config.opt_ms1_tol],0,50,color="r")
        # plt.vlines([-4*mz_stddev,4*mz_stddev],0,50,color="g")
        plt.text(config.opt_ms1_tol,50,f"{np.round(1e6*config.opt_ms1_tol,2)} ppm")
        plt.xlabel("m/z difference (relative)")
        plt.ylabel("Frequency")
        # plt.show()
        plt.savefig(results_folder+"/MZdiff.png",dpi=600,bbox_inches="tight")
    
    
        if ms2:
            ##plot mz alignment
            plt.subplots()
            plt.scatter(all_frags,all_frag_errors,label="Original_MS2",s=1)
            plt.scatter(all_frags,ms2_spl(all_frags),label="Predicted_MS2",s=1)
            # plt.legend()
            plt.xlabel("m/z")
            plt.ylabel("m/z difference (relative)")
            # plt.show()
            plt.savefig(results_folder+"/MS2fit.png",dpi=600,bbox_inches="tight")
            
            
            ## plot mz alignment
            plt.subplots()
            plt.hist(all_frag_errors,100)
            # plt.hist(((np.array(id_mzs)+np.array(diffs)*id_mzs)-mz_func(id_mzs, output_rts))/id_mzs,100,alpha=.5)
            # plt.hist(((np.array(id_mzs)+np.array(diffs)*id_mzs)-mz_spl(id_mzs))/id_mzs,100,alpha=.5)
            plt.hist(all_frag_errors-ms2_spl(all_frags),100,alpha=.5)
            plt.vlines([-config.opt_ms2_tol,config.opt_ms2_tol],0,50,color="r")
            # plt.vlines([-4*mz_stddev,4*mz_stddev],0,50,color="g")
            plt.text(config.opt_ms2_tol,50,np.round(config.opt_ms2_tol,2))
            plt.xlabel("m/z difference (relative)")
            plt.ylabel("Frequency")
            # plt.show()
            plt.savefig(results_folder+"/MS2diff.png",dpi=600,bbox_inches="tight")
    # plt.scatter(id_mzs,diffs)
    # plt.scatter(output_rts,diffs)
    
    if ms2:
        return (rt_spl, mz_func, ms2_func)
    else:
        return (rt_spl, mz_func)




###################################################################################################
###################################################################################################
###################################################################################################

colours = ["tab:blue","tab:orange","tab:green","tab:red",
'tab:purple',
'tab:brown',
'tab:pink',
'tab:gray',
'tab:olive',
'tab:cyan']


def MZRTfit_timeplex(dia_spectra,librarySpectra,dino_features,mz_tol,ms1=False,results_folder=None,ms2=False):
    ## for testing
    # mz_tol,ms1,results_folder,ms2 = (config.ms1_tol,False,None,False)
    # here spectra are both ms1 and ms2 
    
    scans_per_cycle = round(len(dia_spectra.ms2scans)/len(dia_spectra.ms1scans))
    print("Intitial search")
    # print(f"Fitting the {config.n_most_intense} most intense spectra")
    
    ms1spectra = dia_spectra.ms1scans
    ms2spectra = dia_spectra.ms2scans
    
    ms1_rt = np.array([i.RT for i in ms1spectra])
    
    totalIC = np.array([np.sum(i.intens) for i in ms2spectra])
    
    top_n = np.argsort(-totalIC)[:config.n_most_intense]
    top_n_ms1 = top_n//scans_per_cycle
    all_keys = list(librarySpectra)
    rt_mz = np.array([[i["iRT"], i["prec_mz"]] for i in librarySpectra.values()])

    
    
    # top_n_spectra = [ms2spectra[i] for i in top_n]
    # # with multiprocessing.Pool(config.numProc) as p:
    # #     fit_outputs = list(tqdm.tqdm(p.imap(partial(fit_to_lib,
    # #                                                 library=librarySpectra,
    # #                                                 rt_mz=rt_mz,
    # #                                                 all_keys=all_keys,
    # #                                                 dino_features=None,
    # #                                                 rt_filter=False),
    # #                                         top_n_spectra,chunksize=len(top_n_spectra)//config.numProc),total=len(top_n_spectra)))
        
   
    
    # fit_outputs=[]
    # for idx in tqdm.trange(len(top_n)):
    #     fit_output = fit_to_lib(top_n_spectra[idx],
    #                             library=librarySpectra,
    #                             rt_mz=rt_mz,
    #                             all_keys=all_keys,
    #                             dino_features=None,
    #                             rt_filter=False,
    #                             )
    #     fit_outputs.append(fit_output)
    # fit_outputs1=fit_outputs
    
    #################################################################################
    all_dia_rt = [i.RT for i in ms2spectra]
    all_dia_windows = np.array([i.ms1window for i in ms2spectra])
    lowest_mz = np.min(all_dia_windows,0)[0] # assume window span is constant over time
    largest_mz = np.max(all_dia_windows,0)[1]
    mz_bins = np.linspace(lowest_mz,largest_mz,6)
    
    sorted_features = np.argsort(-np.array(dino_features.intensityApex))
    sorted_mz = dino_features.mz[sorted_features]
    large_feature_indices = sorted_features[np.array(np.logical_and(sorted_mz>lowest_mz,sorted_mz<largest_mz))][:config.n_most_intense_features] 
    
    sorted_feature_mz_bins = [sorted_features[np.logical_and(sorted_mz>mz_bins[i],sorted_mz<mz_bins[i+1])] for i in range(len(mz_bins)-1)]
    large_feature_indices = [j for i in sorted_feature_mz_bins for j in i[:(config.n_most_intense_features//(len(mz_bins)-1))]]
    
    lf_rt = np.array(dino_features.rtApex[large_feature_indices])
    lf_mz = np.array(dino_features.mz[large_feature_indices])
    print("Finding correct spectra")
    # lf_spectra = [np.argmin(np.abs(np.array(all_dia_rt)-i)) for i in lf_rt]
    dia_rt_mzwin = np.array([[i.RT,*i.ms1window] for i in ms2spectra])
    lf_spectra = [closest_spec(dia_rt_mzwin,i,j) for i,j in zip(lf_mz,lf_rt)] 
    # mz_int_n = get_largest(dia_spec,tra)
    fit_outputs2=[]
    frags = []
    for idx in tqdm.trange(len(lf_spectra)):
        if ms2:
            fit_output,frag_errors = fit_to_lib(ms2spectra[int(lf_spectra[idx])],
                                                library=librarySpectra,
                                                rt_mz=rt_mz,
                                                all_keys=all_keys,
                                                dino_features=None,
                                                rt_filter=False,
                                                ms1_mz=lf_mz[idx],
                                                return_frags = True,
                                                ms1_spectra = ms1spectra
                                                )
            
            frags.append(frag_errors)
        else:
            
            fit_output = fit_to_lib(ms2spectra[int(lf_spectra[idx])],
                                    library=librarySpectra,
                                    rt_mz=rt_mz,
                                    all_keys=all_keys,
                                    dino_features=None,
                                    rt_filter=False,
                                    ms1_mz=lf_mz[idx],
                                    ms1_spectra = ms1spectra
                                    )
        fit_outputs2.append(fit_output)
        
    fit_outputs = fit_outputs2
    top_n_spectra = [ms2spectra[i] for i in lf_spectra]
     ########################################################################
     
     
    dia_rt = []
    lib_rt = []
    output=[]
    max_ids=[]
    lc_frags_errors=[]
    lc_frags=[]
    for idx,fit_output in enumerate(fit_outputs):    
        if fit_output[0][0]!=0:
            lib_rt.append([librarySpectra[(i[2],i[3])]["iRT"] for i in fit_output])
            dia_rt.append(top_n_spectra[idx].RT)
            output.append(fit_output)
            max_id = np.argmax([i[0] for i in fit_output])
            max_ids.append(max_id)
            if ms2:
                lc_frags_errors.append(frags[idx][0][max_id])
                lc_frags.append(frags[idx][1][max_id])
        
    # max_ids = [np.argmax([i[0] for i in j]) for j in output]
    ms1windows = [i.ms1window for i in top_n_spectra]
    id_keys = [(i[j][2],i[j][3]) for i,j in zip(output,max_ids)]
    id_mzs = [librarySpectra[i]["prec_mz"] for i in id_keys]
    
    
    all_id_rt = [[(i[j][2],i[j][3]),i[j][5]] for i in output for j in range(len(i)) if i[j][0]>10]
    all_coeff = [i[j][0] for i in output for j in range(len(i)) if i[j][0]>10]
    all_id_mzs = [librarySpectra[i[0]]["prec_mz"] for i in all_id_rt]
    
    
    def max_coeff_rt(outputs):
        max_id = np.argmax([i[0] for i in outputs])
        # if outputs[0][0]==0:
        #     return np.nan
        # else:
        return librarySpectra[(outputs[max_id][2],outputs[max_id][3])]["iRT"]
    
    
    output_rts = np.array([max_coeff_rt(i) for i in output])
    output_hyper = np.array([i[j][18] for i,j in zip(output,max_ids)])
    output_seqs = np.array([i[j][2:4] for i,j in zip(output,max_ids)])
    all_lib_rts = np.array([librarySpectra[i[0]]["iRT"] for i in all_id_rt])
    
    
    
    ## find keys that appear more than once
    multiples = []
    multiples_idxs = []
    num_multiples = []
    searched = set()
    for key in set(id_keys):
        if key in searched:
            continue
        else:
            key_pos = np.where([i==key for i in id_keys])[0]
            if len(key_pos)>1:
                multiple_rts = np.array([dia_rt[i] for i in key_pos])
                order = np.argsort(multiple_rts)
                multiples.append(multiple_rts[order])
                multiples_idxs.append(key_pos[order])
                num_multiples.append(len(key_pos))
            searched.update(key)
    
    if config.num_timeplex is None:
        timeplex = stats.mode(num_multiples).mode
    else:
        timeplex = config.num_timeplex
        
    # while it may be nice to know, we are assuming that this is not constant and therfore not necessary to know
    time_diffs = np.concatenate([np.diff(i) for i in multiples if len(i)==timeplex])
    # plt.hist(time_diffs,np.linspace(-1,5,40))
    # plt.xlabel("TimePLEX offset")
    
    # plt.scatter([i[0] for i in multiples if len(i)==timeplex],time_diffs,s=1)
    # plt.ylabel("TimePLEX offset")
    # plt.xlabel("T0 RT")
    
    
    # t1 = np.array([[dia_rt[i[0]],output_rts[i[0]]] for i in multiples_idxs])
    # t2 = np.array([[dia_rt[i[1]],output_rts[i[1]]] for i in multiples_idxs])
    
    rt_spls = []
    t_vals = []
    t_seqs = []
    converted_rts = []
    gaussian_fits = []
    for idx in range(timeplex):
        lib_rt_range = [np.percentile(rt_mz[:,0],5),np.percentile(rt_mz[:,0],95)]
        t1 = np.array([[dia_rt[i[idx]],output_rts[i[idx]],output_hyper[i[idx]]] for i in multiples_idxs if len(i)==timeplex and output_rts[i[idx]]>lib_rt_range[0] and output_rts[i[idx]]<lib_rt_range[1]])
        t1_s = [output_seqs[i[idx]] for i in multiples_idxs if len(i)==timeplex and output_rts[i[idx]]>lib_rt_range[0] and output_rts[i[idx]]<lib_rt_range[1]]
        # t1 = np.array([[dia_rt[i[idx]],output_rts[i[idx]],output_hyper[i[idx]]] for i in multiples_idxs if len(i)==timeplex])
        rt_spl = threestepfit(t1[:,1],t1[:,0],1,t1[:,2])
        rt_spls.append(rt_spl)
        t_vals.append(t1)
        t_seqs.append(t1_s)
        
        converted_rt = rt_spl(t1[:,1])
        converted_rts.append(converted_rt)
        gaussian_fits.append(fit_gaussian(t1[:,0]-converted_rt))
    
    # ## just use T0
    # export_df = pd.DataFrame({"obs_rt":np.concatenate([t_vals[0][:,0],t_vals[1][:,0]]),
    #                           "lib_rt":np.concatenate([t_vals[0][:,1],t_vals[1][:,1]]),
    #                           "seq":[i[0] for i in t_seqs[0]]+[i[0] for i in t_seqs[1]],
    #                           "charge":[i[1] for i in t_seqs[0]]+[i[1] for i in t_seqs[1]]})
    # export_df.to_csv("/Volumes/Lab/KMD/For_JD/AllFeatures.csv")
    
    
    ## combined gausian fit
    rt_amplitude, rt_mean, rt_stddev = fit_gaussian(np.concatenate([t[:,0]-c_rt for t,c_rt in zip(t_vals,converted_rts)]))
    
    ## NB: Only for timeplex=2
    ## computes differences between the fit lines of both plexes
    prediction_diffs = np.abs(rt_spls[1](t_vals[1][:,1])-rt_spls[0](t_vals[1][:,1]))

    #####  Assume that the mz error is independent of timeplex
    resp_ms1scans = [closest_ms1spec(dia_rt[i], ms1_rt) for i in range(len(dia_rt))]
    diffs = [closest_peak_diff(mz, ms1spectra[i].mz) for i,mz in zip(resp_ms1scans,id_mzs)]
    
    mz_spl = twostepfit(id_mzs,diffs,1)
    
    
    def mz_func(mz):
        return mz+(mz_spl(mz)*mz)
    
    mz_amplitude, mz_mean, mz_stddev = fit_gaussian(diffs-mz_spl(id_mzs))
    
    
    ### MS2 alignment
    if ms2:
        all_frag_errors = np.concatenate(lc_frags_errors)
        all_frags = np.concatenate(lc_frags)
        ms2_spl = twostepfit(all_frags,all_frag_errors,1)
        def ms2_func(mz):
            return mz+(ms2_spl(mz)*mz)
        
        ms2_amplitude, ms2_mean, ms2_stddev = fit_gaussian(all_frag_errors-ms2_spl(all_frags))
    
   
    # new_rt_tol = get_tol(dia_rt-rt_spl(output_rts))
    new_rt_tol = 4*rt_stddev
    print(f"Optimsed RT tolerance: {new_rt_tol}")
    
    # ## ensure there is no overlap
    # obs_rt_range = [min(dia_rt),max(dia_rt)]
    # ## range that captures middle 90% of library
    # lib_rt_range = [np.percentile(rt_mz[:,0],5),np.percentile(rt_mz[:,0],95)]
    # sample_rts = np.linspace(lib_rt_range[0],lib_rt_range[1],100)
    # # plt.scatter(sample_rts,rt_spls[0](sample_rts),s=1)
    # # plt.scatter(sample_rts,rt_spls[1](sample_rts),s=1)
    # ## differnce in 
    # model_diffs = np.abs(rt_spls[0](sample_rts)-rt_spls[1](sample_rts))
    # rt_tol_spl = InterpolatedUnivariateSpline(sample_rts,model_diffs)
    # # plt.plot(rt_spls[0](sample_rts),model_diffs)
    # # plt.scatter(rt_spls[1](t_vals[1][:,1]),np.ones_like(rt_spls[1](t_vals[1][:,1]))*new_rt_tol*2,s=1)
    
    # def rt_tol_fn(obs_rt):
    #     return np.maximum(np.minimum(new_rt_tol,(rt_tol_spl(lib_rt)/2)*.99),0)
    
    # config.rt_tol_spl = rt_tol_fn
    
    
    if new_rt_tol>np.median(time_diffs)/2:
        print("Warning; Library RTs overlapping")
        new_rt_tol = (np.min(prediction_diffs)/2)*.99 # ensure no overlap
        print(f"Reseting tolerance to {new_rt_tol}")
    
    
    
    config.opt_rt_tol = new_rt_tol
    
    
    
    # set optimised ms2 tol
    # is_real = ~np.isnan(diffs)
    # buffer = 1.2
    # config.opt_ms1_tol = np.round(
    #                         np.sort(
    #                             np.abs(((np.array(id_mzs)+np.array(diffs)*id_mzs)-mz_func(id_mzs, output_rts))/id_mzs
    #                                    )[is_real])[int(sum(is_real)*.95)]*buffer,6+5)#6 for 1e-6 the 5 decimal places

    # new_ms1_tol = get_tol(((np.array(id_mzs)+np.array(diffs)*id_mzs)-mz_func(id_mzs, output_rts))/id_mzs)
    # new_ms1_tol = get_tol(diffs-mz_spl(id_mzs))
    new_ms1_tol = 4*mz_stddev
    print(f"Optimsed ms1 tolerance: {new_ms1_tol}")
    
    config.opt_ms1_tol  = new_ms1_tol
    
    if ms2:
        new_ms2_tol = 4*ms2_stddev
        config.opt_ms2_tol  = new_ms2_tol
    
    if results_folder is not None:
        
        ### Save functions
        for idx in range(timeplex):
            with open(results_folder+f"/rt_spl{idx}","wb") as dill_file:
                dill.dump(rt_spls[idx],dill_file)
            
        with open(results_folder+"/mz_func","wb") as dill_file:
            dill.dump(mz_func,dill_file)
        
        if ms2:
            with open(results_folder+"/ms2_func","wb") as dill_file:
                dill.dump(ms2_func,dill_file)
            
        ##plot RT alignment
        plt.subplots()
        for idx in range(timeplex):
            plt.scatter(t_vals[idx][:,1],t_vals[idx][:,0],s=1,c=colours[idx],alpha=.2)
            plt.scatter(t_vals[idx][:,1],rt_spls[idx](t_vals[idx][:,1]),s=1,label=f"T{str(idx)}",c=colours[idx])
            plt.scatter(t_vals[idx][:,1],rt_spls[idx](t_vals[idx][:,1])+config.opt_rt_tol,s=.1,c=colours[idx],alpha=.1)
            plt.scatter(t_vals[idx][:,1],rt_spls[idx](t_vals[idx][:,1])-config.opt_rt_tol,s=.1,c=colours[idx],alpha=.1)
            # plt.scatter(t_vals[idx][:,1],rt_spls[idx](t_vals[idx][:,1])+config.rt_tol_spl(t_vals[idx][:,1]),s=.1,c=colours[idx],alpha=.1)
            # plt.scatter(t_vals[idx][:,1],rt_spls[idx](t_vals[idx][:,1])-config.rt_tol_spl(t_vals[idx][:,1]),s=.1,c=colours[idx],alpha=.1)
        plt.legend(markerscale=10)
        plt.xlabel("Library RT")
        plt.ylabel("Observed RT")
        # plt.show()
        plt.savefig(results_folder+"/RTfit.png",dpi=600,bbox_inches="tight")
        
        plt.subplots()
        for idx in range(timeplex):
            vals,bins,_ =plt.hist(t_vals[idx][:,0]-rt_spls[idx](t_vals[idx][:,1]),100,alpha=.5,label=f"T{str(idx)}")
            # rt_stddev = gaussian_fits[idx][-1]
        x_scale = np.diff(plt.xlim())[0]
        plt.vlines([-config.opt_rt_tol,config.opt_rt_tol],0,max(vals),color="r")
        plt.text(config.opt_rt_tol+x_scale/100,max(vals)*.8,np.round(config.opt_rt_tol,2))
        plt.legend()  
        plt.xlabel("RT difference")
        plt.ylabel("Frequency") 
        # plt.show()
        plt.savefig(results_folder+"/RTdiff.png",dpi=600,bbox_inches="tight")
        
        
        plt.subplots()
        for idx in range(timeplex):
            if idx==1:
                offset = prediction_diffs
            else:
                offset = 0
            vals,bins,_ =plt.hist(t_vals[idx][:,0]-rt_spls[idx](t_vals[idx][:,1])+offset,100,alpha=.5,label=f"T{str(idx)}")
            # rt_stddev = gaussian_fits[idx][-1]
            plt.vlines([-config.opt_rt_tol+np.median(offset),config.opt_rt_tol+np.median(offset)],0,max(vals),color="r")
        x_scale = np.diff(plt.xlim())[0]
        # plt.vlines([-config.opt_rt_tol,config.opt_rt_tol],0,max(vals),color="r")
        plt.text(config.opt_rt_tol+x_scale/100,max(vals)*.8,np.round(config.opt_rt_tol,2))
        plt.legend()  
        plt.xlabel("RT difference")
        plt.ylabel("Frequency") 
        # plt.show()
        
        fig, ax = plt.subplots(nrows = timeplex)        
        for idx,row in enumerate(ax):
            row.scatter(t_vals[idx][:,1],t_vals[idx][:,0]-rt_spls[idx](t_vals[idx][:,1]),label="Original_RT",s=.1)
            row.plot([min(t_vals[idx][:,1]),max(t_vals[idx][:,1])],[0,0],color="r",linestyle="--",alpha=.5)
            row.plot([min(t_vals[idx][:,1]),max(t_vals[idx][:,1])],[config.opt_rt_tol,config.opt_rt_tol],color="g",linestyle="--",alpha=.5)
            row.plot([min(t_vals[idx][:,1]),max(t_vals[idx][:,1])],[-config.opt_rt_tol,-config.opt_rt_tol],color="g",linestyle="--",alpha=.5)
            row.set_ylabel(f"RT Residuals (T{idx})")
        # plt.scatter(output_rts,rt_spl(output_rts),label="Predicted_RT",s=1)
        # plt.legend()
        plt.xlabel("Library RT")
        # plt.ylabel("RT Residuals")
        # plt.show()
        plt.savefig(results_folder+"/RtResidual.png",dpi=600,bbox_inches="tight")
        
        
        ##plot mz alignment
        plt.subplots()
        plt.scatter(id_mzs,diffs,label="Original_MZ",s=1)
        plt.scatter(id_mzs,mz_spl(id_mzs),label="Predicted_MZ",s=1)
        # plt.legend()
        plt.xlabel("m/z")
        plt.ylabel("m/z difference (relative)")
        # plt.show()
        plt.savefig(results_folder+"/MZfit.png",dpi=600,bbox_inches="tight")
        
        
        ## plot mz alignment
        plt.subplots()
        plt.hist(diffs,100)
        # plt.hist(((np.array(id_mzs)+np.array(diffs)*id_mzs)-mz_func(id_mzs, output_rts))/id_mzs,100,alpha=.5)
        # plt.hist(((np.array(id_mzs)+np.array(diffs)*id_mzs)-mz_spl(id_mzs))/id_mzs,100,alpha=.5)
        plt.hist(diffs-mz_spl(id_mzs),100,alpha=.5)
        plt.vlines([-config.opt_ms1_tol,config.opt_ms1_tol],0,50,color="r")
        # plt.vlines([-4*mz_stddev,4*mz_stddev],0,50,color="g")
        plt.text(config.opt_ms1_tol,50,f"{np.round(1e6*config.opt_ms1_tol,2)} ppm")
        plt.xlabel("m/z difference (relative)")
        plt.ylabel("Frequency")
        # plt.show()
        plt.savefig(results_folder+"/MZdiff.png",dpi=600,bbox_inches="tight")
    
    
        if ms2:
            ##plot mz alignment
            plt.subplots()
            plt.scatter(all_frags,all_frag_errors,label="Original_MS2",s=1)
            plt.scatter(all_frags,ms2_spl(all_frags),label="Predicted_MS2",s=1)
            # plt.legend()
            plt.xlabel("m/z")
            plt.ylabel("m/z difference (relative)")
            # plt.show()
            plt.savefig(results_folder+"/MS2fit.png",dpi=600,bbox_inches="tight")
            
            
            ## plot mz alignment
            plt.subplots()
            plt.hist(all_frag_errors,100)
            # plt.hist(((np.array(id_mzs)+np.array(diffs)*id_mzs)-mz_func(id_mzs, output_rts))/id_mzs,100,alpha=.5)
            # plt.hist(((np.array(id_mzs)+np.array(diffs)*id_mzs)-mz_spl(id_mzs))/id_mzs,100,alpha=.5)
            plt.hist(all_frag_errors-ms2_spl(all_frags),100,alpha=.5)
            plt.vlines([-config.opt_ms2_tol,config.opt_ms2_tol],0,50,color="r")
            # plt.vlines([-4*mz_stddev,4*mz_stddev],0,50,color="g")
            plt.text(config.opt_ms2_tol,50,np.round(config.opt_ms2_tol,2))
            plt.xlabel("m/z difference (relative)")
            plt.ylabel("Frequency")
            # plt.show()
            plt.savefig(results_folder+"/MS2diff.png",dpi=600,bbox_inches="tight")
    # plt.scatter(id_mzs,diffs)
    # plt.scatter(output_rts,diffs)
    
    if ms2:
        return (rt_spls, mz_func, ms2_func)
    else:
        return (rt_spls, mz_func)






###################################################################################################
###################################################################################################
###################################################################################################
# import alphatims.bruker
# data = alphatims.bruker.TimsTOF("/Users/kevinmcdonnell/Programming/Data/2023_10_23_QC_LF_DIA_timsTOF_Slot2-20_1_893.d/")
# librarySpectra = load_tsv_speclib("/Users/kevinmcdonnell/Programming/Data/SpecLibs/tims_library_dec_23.tsv")

# diann_file = "/Volumes/Lab/KMD/timsLibrary/report.tsv"
# report = pd.read_csv(diann_file,delimiter="\t")

# for idx,key  in enumerate(zip(report["Modified.Sequence"],report["Precursor.Charge"])):
#     if key in librarySpectra:
#         librarySpectra[key]["iRT"] = report["RT"][idx]
#         librarySpectra[key]["IonMob"] = report["IM"][idx]
        
# prec_index_dict = {i:j for i,j in zip(data.fragment_frames.Frame,data.fragment_frames.Precursor)}
# mz_tol= 20e-6

def align_pasef(data,librarySpectra,prec_index_dict,mz_tol,im_merge_tol = None, ms1=False,results_folder=None):
    
    print("Intitial search")
    
    
    ## temp: Move these to config file
    num_frames_to_check = 40
    num_scans_to_check = 20

    num_frames = data.frame_max_index
    num_scans = np.mean(data.frames.NumScans.to_numpy(),dtype=int)
    
    frames_to_check = np.array(((np.arange(num_frames_to_check)+.5)/num_frames_to_check)*num_frames,dtype=int)
    scans_to_check = np.array(((np.arange(num_scans_to_check)+.5)/num_scans_to_check)*num_scans,dtype=int)
    
    # check if ms1 only scan => Move to next scan
    for i,frame in enumerate(frames_to_check):
        if frame%data.precursor_max_index==1:
            frames_to_check[i]+=1
            

    all_keys = list(librarySpectra)
    rt_mz_im = np.array([[i["iRT"], i["prec_mz"], i["IonMob"]] for i in librarySpectra.values()])

    frame_scan_pairs = [(f,s) for f in frames_to_check for s in scans_to_check]
    
    fit_outputs = []
    for frame,scan in tqdm.tqdm(frame_scan_pairs):
    

        fit_output = fit_to_pasef(frame_scan=[frame,scan],
                                  AllData=data,
                                  library=librarySpectra,
                                  rt_mz_im=rt_mz_im,
                                  all_keys=all_keys,
                                  prec_index_dict=prec_index_dict,
                                  rt_filter=False,
                                  im_merge_tol=im_merge_tol)
        
        fit_outputs.append(fit_output)
        
    
     
    dia_rt = []
    lib_rt = []
    dia_im = []
    lib_im = []
    output=[]
    max_ids=[]
    for idx,fit_output in enumerate(fit_outputs):    
        if fit_output[0][0]!=0:
            lib_rt.append([librarySpectra[(i[2],i[3])]["iRT"] for i in fit_output])
            dia_rt.append(data.rt_values[frame_scan_pairs[idx][0]])
            lib_im.append([librarySpectra[(i[2],i[3])]["IonMob"] for i in fit_output])
            dia_im.append(data.mobility_values[frame_scan_pairs[idx][1]])
            output.append(fit_output)
            max_ids.append(np.argmax([i[0] for i in fit_output]))
    
    # max_ids = [np.argmax([i[0] for i in j]) for j in output]
    # ms1windows = [i.ms1window for i in top_n_spectra]
    id_keys = [(i[j][2],i[j][3]) for i,j in zip(output,max_ids)]
    id_mzs = [librarySpectra[i]["prec_mz"] for i in id_keys]
        
    def max_coeff_val(outputs,val="iRT"):
        max_id = np.argmax([i[0] for i in outputs])
        # if outputs[0][0]==0:
        #     return np.nan
        # else:
        return librarySpectra[(outputs[max_id][2],outputs[max_id][3])][val]
    
    
    output_rts = np.array([max_coeff_val(i) for i in output])

    rt_spl = threestepfit(output_rts,dia_rt,1)
    
    converted_rt = rt_spl(output_rts)
    
    rt_amplitude, rt_mean, rt_stddev = fit_gaussian(dia_rt-converted_rt)
    
    new_rt_tol = 4*rt_stddev
    print(f"Optimsed RT tolerance: {new_rt_tol}")
    config.opt_rt_tol = new_rt_tol
    
    output_ims = np.array([max_coeff_val(i,"IonMob") for i in output])
    im_spl = threestepfit(output_ims,dia_im,1)
    converted_im = im_spl(output_ims)
    
    im_amplitude, im_mean, im_stddev = fit_gaussian(dia_im-converted_im)
    new_im_tol = 4*im_stddev
    print(f"Optimsed IM tolerance: {new_im_tol}")
    config.opt_im_tol = new_im_tol
    
    
    
    if results_folder is not None:
        
        
        ### Save functions
        with open(results_folder+"/rt_spl","wb") as dill_file:
            dill.dump(rt_spl,dill_file)
            
        with open(results_folder+"/im_spl","wb") as dill_file:
            dill.dump(im_spl,dill_file)
        
        # with open(results_folder+"/mz_func","wb") as dill_file:
        #     dill.dump(mz_func,dill_file)
        
        # if ms2:
        #     with open(results_folder+"/ms2_func","wb") as dill_file:
        #         dill.dump(ms2_func,dill_file)
        
        ##plot RT alignment
        plt.subplots()
        plt.scatter(output_rts,dia_rt,label="Original_RT",s=1)
        plt.scatter(output_rts,rt_spl(output_rts),label="Predicted_RT",s=1)
        plt.legend()
        plt.xlabel("Library RT")
        plt.ylabel("Observed RT")
        # plt.show()
        plt.savefig(results_folder+"/RTfit.png",dpi=600,bbox_inches="tight")
        
        plt.subplots()
        vals,bins,_,=plt.hist(dia_rt-rt_spl(output_rts),100)
        plt.vlines([-config.opt_rt_tol,config.opt_rt_tol],0,max(vals),color="r")
        # plt.vlines([-4*rt_stddev,4*rt_stddev],0,50,color="g")
        plt.text(config.opt_rt_tol,max(vals),np.round(config.opt_rt_tol,2))
        plt.xlabel("RT difference")
        plt.ylabel("Frequency")
        # plt.show()
        plt.savefig(results_folder+"/RTdiff.png",dpi=600,bbox_inches="tight")
        
        
        ##plot IM alignment
        plt.subplots()
        plt.scatter(output_ims,dia_im,label="Original_IM",s=1)
        plt.scatter(output_ims,im_spl(output_ims),label="Predicted_IM",s=1)
        plt.legend()
        plt.xlabel("Library IM")
        plt.ylabel("Observed IM")
        # plt.show()
        plt.savefig(results_folder+"/imfit.png",dpi=600,bbox_inches="tight")
        
        plt.subplots()
        vals,bins,_,=plt.hist(dia_im-im_spl(output_ims),100)
        plt.vlines([-config.opt_im_tol,config.opt_im_tol],0,max(vals),color="r")
        # plt.vlines([-4*im_stddev,4*im_stddev],0,50,color="g")
        plt.text(config.opt_im_tol,max(vals),np.round(config.opt_im_tol,3))
        plt.xlabel("IM difference")
        plt.ylabel("Frequency")
        # plt.show()
        plt.savefig(results_folder+"/imdiff.png",dpi=600,bbox_inches="tight")
        
    return (rt_spl, im_spl)

def align_timspeak(fragment_clusters,librarySpectra,prec_index_dict,mz_tol,im_merge_tol = None, ms1=False,results_folder=None):
    
    rt_width = config.rt_width
    all_keys = list(librarySpectra)
    rt_mz_im = np.array([[i["iRT"], i["prec_mz"], i["IonMob"]] for i in librarySpectra.values()])
    
    num_scans = 100
    
    rt_range = [np.floor(np.min(fragment_clusters["rt_weighted_average"])),
                np.ceil(np.max(fragment_clusters["rt_weighted_average"]))]
   
    quad_mz_pairs = sorted(set([(i,j) for i,j in zip(fragment_clusters.quad_low_mz_values,fragment_clusters.quad_high_mz_values) if i>0]))# remove ms1 peaks

    
    rt_quad_pairs = [(rt,quad_pair) for rt in range(int(rt_range[0]),int(rt_range[1])) for quad_pair in quad_mz_pairs]
    
    np.random.seed(0)
    random_idxs = np.random.choice(range(len(rt_quad_pairs)),num_scans,replace=False)
    
    rt_quad_pairs_random = [rt_quad_pairs[i] for i in random_idxs]
    
    fit_outputs = []
    for rt_quad in tqdm.tqdm(rt_quad_pairs_random):
    
        
        fit_output = fit_timspeak(rt_quad,
                                  library=librarySpectra,
                                  rt_mz_im=rt_mz_im,
                                  all_keys=all_keys,
                                  fragment_clusters=fragment_clusters,
                                  rt_width=rt_width,
                                  rt_tol=None,
                                  mz_tol=2e-5,
                                  im_tol=0.05)
        fit_outputs.append(fit_output)
        
    
    
    return




"""
    
    ##########################################################################
    sorted_idxs = np.argsort(output_rts)
    sort_rts = np.array(output_rts)[sorted_idxs]
    sort_dia_rts = np.array(dia_rt)[sorted_idxs]
    knots = quantiles(sort_rts,n=2)
    spl = spline(sort_rts,sort_dia_rts,knots)
    
    # find outliers and remove
    _bool = abs(spl(sort_rts)-sort_dia_rts)<20
    spl2 = spline(sort_rts[_bool],sort_dia_rts[_bool],knots)
    
    plt.scatter(sort_rts,sort_dia_rts,label="Orginal_RT")
    plt.scatter(sort_rts,rt_spl(sort_rts),label="Predicted_RT") 
    plt.xlabel("Library RT")
    plt.ylabel("Observed RT")
    plt.legend()
    
    vals,bins,_ = plt.hist(sort_rts-sort_dia_rts,100,label="Orginal_RT")
    plt.hist(rt_spl(sort_rts)-sort_dia_rts,bins,alpha=.5,label="Predicted_RT")
    plt.xlabel("Difference between lib and obs RT")
    plt.ylabel("Frequency")
    plt.legend()
    # plt.show()
    
    x=id_mzs
    y=diffs
    y_exists = np.isfinite(y)
    x_exists = np.isfinite(x)*y_exists
    x=np.array(x)[x_exists]
    y=np.array(y)[x_exists]
    y_range = np.max(y)-np.min(y)
    sorted_idxs = np.argsort(x)
    sort_x = np.array(x)[sorted_idxs]
    sort_y = np.array(y)[sorted_idxs]
    knots = quantiles(sort_x,n=2)
    spl = spline(sort_x,sort_y,knots)
    
    plt.scatter(np.array(id_mzs)+diffs,diffs)
    plt.xlabel("Observed m/z")
    plt.ylabel("Difference to observed MS1 peak")
    
    plt.scatter(dia_rt,diffs)
    plt.xlabel("Observed RT")
    plt.ylabel("Difference to observed MS1 peak")
    
    # find outliers and remove; points over 1/4 of the y range away from prediction
    _bool = abs(spl(sort_x)-sort_y)<(y_range/4)
    spl2 = spline(sort_x[_bool],sort_y[_bool],knots)
    
    mz_spl = twostepfit(id_mzs, diffs)
    plt.scatter(id_mzs,diffs)
    plt.scatter(sort_x[_bool],sort_y[_bool])
    plt.scatter(id_mzs,spl(id_mzs))
    plt.scatter(id_mzs,spl2(id_mzs))
    
    plt.scatter(output_rts,id_mzs,diffs)
    plt.scatter(output_rts,mz_spl(output_rts))
    

def func(x,a,b,c):
    return (a*np.array(x[0]))+(b*np.array(x[1]))+c

x=np.array(output_rts)
y=np.array(id_mzs)
z=np.array(diffs)
z_exists = np.isfinite(z)

parameters, covariance = curve_fit(func, [x[z_exists],y[z_exists] ], z[z_exists])

model_x_data = np.linspace(min(x), max(x), 30)
model_y_data = np.linspace(min(y), max(y), 30)
X, Y = np.meshgrid(model_x_data, model_y_data)
# calculate Z coordinate array
Z = func(np.array([X, Y]), *parameters)
fig = plt.figure()
# setup 3d object
#ax = Axes3D(fig)
ax=fig.add_subplot(111,projection='3d')
# plot surface
# ax.plot_surface(X, Y, Z)
ax.scatter(x,y,z,c=z)
ax.set_xlabel("RT")
ax.set_ylabel("m/z")
ax.set_zlabel("m/z Difference")


z_pred = func([x,y],*parameters)
vals,bins,_=plt.hist(diffs,100)
plt.hist(z_pred,bins,alpha=.5)

diffs = [get_diff(mz, ms1spectra[ms1_idx].mz, window, mz_tol) for  mz,ms1_idx,window in zip(id_mzs,top_n_ms1,ms1windows)]


original_mz = np.array(y[z_exists])+(z[z_exists]*y[z_exists])
pred_diff = func([x[z_exists],y[z_exists]],*parameters)*y[z_exists]
pred_mz = y[z_exists]+pred_diff
pred_mz = mz_func(y[z_exists],x[z_exists])
# pred_mz = original_mz-(func([x[z_exists],y[z_exists]],*parameters)*y[z_exists])
# plt.scatter(original_mz,pred_mz)
vals,bins,_=plt.hist(diffs,50,label="Original")
# plt.hist(pred_mz-y[z_exists],bins,alpha=.5,label="Predicted")
plt.hist((original_mz-pred_mz)/pred_mz,bins,alpha=.5,label="Predicted")
plt.xlabel("m/z difference")
plt.ylabel("Frequency")
plt.legend()

# """