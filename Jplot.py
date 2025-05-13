"""
This Source Code Form is subject to the terms of the Oxford Nanopore
Technologies, Ltd. Public License, v. 1.0.  Full licence can be found
at https://github.com/ParallelSquared/JMod/blob/main/LICENSE.txt
"""


import subprocess
import numpy as np
from pyteomics import mzml, auxiliary
import os
import matplotlib.pyplot as plt


plt.rcParams['figure.dpi'] = 500
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12


def within_tol(x, y, atol, rtol):
    x = np.asanyarray(x)
    y = np.asanyarray(y)
    diff = x-y
    logic = np.less_equal(abs(diff), atol + rtol * np.abs(y))
    log_dif = np.zeros((*logic.shape,2))
    log_dif[...,0] = logic
    log_dif[...,1] = diff
    return log_dif 

# def plot_mz(mz=None,intensity=None,peak_list=None,show=False,col=None,alpha=None):
#     if mz is None and peak_list is None:
#         print("enter valid peaks")
#         return None
#     if peak_list is not None:
#         if len(peak_list)==2:
#             mz=peak_list[0]
#             intensity=peak_list[1]
#         else:
#             mz,intensity = [i for i in zip(*peak_list)]
#     max_intens = max(intensity)
#     plt.vlines(mz, 0, intensity,colors=col,alpha=alpha)
#     plt.ylim(0,max_intens*1.1)
#     plt.xlabel("m/z")
#     plt.ylabel("Intensity")
    
#     if show:
#         plt.show()

def plot_mz(peak_list=None,show=False,col=None,alpha=None,axis=True,norm=False):
    
    if  peak_list is None:
        print("enter valid peaks")
        return None
    if peak_list is not None:
        if len(peak_list)==2:
            mz=peak_list[0]
            intensity=peak_list[1]
        else:
            mz,intensity = [i for i in zip(*peak_list)]
    max_intens = max(intensity)
    if norm:
        plt.vlines(mz, 0, intensity/max_intens,colors=col,alpha=alpha)
    else:
        plt.vlines(mz, 0, intensity,colors=col,alpha=alpha)
    if axis:
        plt.ylim(0,max_intens*1.1)
        plt.xlabel("m/z")
        plt.ylabel("Intensity")
        
    if show:
        plt.show()
        
    
        
        
def plot_mz_inv(peak_list=None,show=False,col=None,alpha=None,axis=True,norm=False):
    
    if  peak_list is None:
        print("enter valid peaks")
        return None
    if peak_list is not None:
        if len(peak_list)==2:
            mz=peak_list[0]
            intensity=peak_list[1]
        else:
            mz,intensity = [i for i in zip(*peak_list)]
    max_intens = max(intensity)
    if norm:
        plt.vlines(mz, -intensity/max_intens, 0,colors=col,alpha=alpha)
    else:
        plt.vlines(mz, -intensity, 0,colors=col,alpha=alpha)
    if axis:
        plt.ylim(max_intens*1.1,0)
        plt.xlabel("m/z")
        plt.ylabel("Intensity")
        
    if show:
        plt.show()
    
def mirror_plot(peak_list1,peak_list2,show=False,col=None,alpha=None):
        
    plot_mz(peak_list1,axis=False,norm=True)
    plot_mz_inv(peak_list2,axis=False,norm=True)
    xlim = plt.xlim()
    plt.hlines(0,xlim[0],xlim[1],colors="black",linewidth=.5)
    
    plt.xlabel("m/z")
    plt.ylabel("Intensity")

def plot_trace(rts=None,ints=None):
    if rts is None:
        return
    
    plt.plot(rts,ints)
    
# ms_spectra = spectra.ms1scans

##### unifinished!!!
def filter_spectrum(spectrum,mz=None,rt=None,mz_tol=0.05,rt_tol=1):
    peak_list = np.stack(spectrum.mz,spectrum.intens,1)
    if mz is not None:
        if not hasattr(mz, "__len__"):
            
            return [[spectrum.rt,spectrum.mz]]

def get_trace(mz,spectra,tol = 0.05):
    
    return

#########

def plot_mz_rt(mz_list,rt_list,int_list=None,t=False):
    x,y = [rt_list,mz_list] if t else [mz_list,rt_list]
    plt.scatter(x,y,s=.01,c=int_list)
    
# plot ms1 feature
def plot_feature(spectra,mz,rt,mz_tol=5,rt_tol=1,t=False):
    
    mz_list=[]
    rt_list=[]
    int_list=[]
    for spec in spectra:
        current_rt=spec.RT
        if abs(rt-current_rt)<rt_tol:
            for _mz,i in zip(spec.mz,spec.intens):
                if abs(mz-_mz)<mz_tol:
                    mz_list.append(_mz)
                    rt_list.append(current_rt)
                    int_list.append(i)
    plot_mz_rt(mz_list,rt_list,int_list,t=t)
    

def plot_mz_rt_all(spectra,t=False):
    
    mz_list=[]
    rt_list=[]
    int_list=[]
    for spec in spectra:
        current_rt=spec.RT
        
        for _mz,i in zip(spec.mz,spec.intens):
            
            mz_list.append(_mz)
            rt_list.append(current_rt)
            int_list.append(i)
    print(f"Plotting {len(mz_list)} points")   
    plot_mz_rt(mz_list,rt_list,int_list,t=t)
    
    
colours = ['tab:blue',
 'tab:orange',
 'tab:green',
 'tab:red',
 'tab:purple',
 'tab:brown',
 'tab:pink',
 'tab:gray',
 'tab:olive',
 'tab:cyan']