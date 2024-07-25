import numpy as np
import csv
import re
from pyteomics import mass
from scipy import signal
from scipy.optimize import curve_fit
import config


def feature_list_rt(DinoDF,rt,rt_tol): 
    # _bool=np.logical_and(DinoDF["rtStart"]-rt_tol<rt,DinoDF["rtEnd"]+rt_tol>rt)
    _bool=np.abs(DinoDF["rtApex"]-rt)<rt_tol
    return(DinoDF[_bool])
    
# def feature_list_mz(DinoDF,window): 
#     _bool=np.logical_and(DinoDF["mz"]>window[0],DinoDF["mz"]<window[1])
def feature_list_mz(DinoDF,window_mz,window_width): 
    _bool=np.abs(DinoDF["mz"]-window_mz)<(window_width/2)
    return(DinoDF[_bool])
    
    

def filter_lib(dia_spectrum,library,rt_tol=.5):
    
    rt = dia_spectrum.RT
    mz_window = dia_spectrum.ms1window
    # get rt and prec mz vector
    rt_mz = np.array([[i["iRT"], i["prec_mz"]] for i in library])
    
    _bool = np.logical_and(rt_mz[:,0]>rt-rt_tol,rt_mz[:,0]<rt+rt_tol)
    
def window_width(spec):
    w1,w2 = spec.ms1window
    return w2-w1
    

def createTolWindows(positions,tolerance):
    sorted_positions = np.sort(positions)
    abs_tols=tolerance*sorted_positions
    l_bound = sorted_positions-abs_tols
    u_bound = sorted_positions+abs_tols
    diffs = np.diff(sorted_positions)<2*abs_tols[1:]
    l_bound = np.append(l_bound[:1],l_bound[1:][~diffs])
    u_bound = np.append(u_bound[:-1][~diffs],u_bound[-1:])
    # windowEdges = np.sort(np.concatenate((l_bound,u_bound)))
    windowEdges = np.empty((l_bound.size + u_bound.size,), dtype=u_bound.dtype)
    windowEdges[::2]=l_bound
    windowEdges[1::2]=u_bound
    return windowEdges
    


def write_to_csv(data,filepath,colnames=None):
    
    with open(filepath,"a", newline='') as write_file:
        writer = csv.writer(write_file)
        if colnames is not None:
            writer.writerow(colnames)
        writer.writerows(data)
        
        
        
def within_tol(x, y, atol, rtol):
    x = np.asanyarray(x)
    y = np.asanyarray(y)
    diff = x-y
    logic = np.less_equal(abs(diff), atol + rtol * np.abs(y))
    log_dif = np.zeros((*logic.shape,2))
    log_dif[...,0] = logic
    log_dif[...,1] = diff
    return log_dif 
    

def get_diff(mz,peaks,tol):
    
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
    
    
            
def ms1_error(dia_ms1,lib_mzs,tol):
    
    diffs = np.array([get_diff(mz, dia_ms1, tol) for mz in lib_mzs])
    return diffs




def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w

# def moving_average(a, n=3):
#     a = np.array(a, dtype=np.float32)
#     ret = np.cumsum(a,)
#     ret[n:] = ret[n:] - ret[:-n]
#     return ret[n - 1:] / n


def moving_auc(x, w, dx):
    return np.convolve(x, np.ones(w), 'same') * dx

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
      
        

def frag_to_peak(frag_dict):
    peaks = np.array(list(frag_dict.values()))
    order = np.argsort(peaks[:,0])
    return peaks[order]




def closest_ms1spec(ms2rt,ms1rt):
    """
    Parameters
    ----------
    ms2rt : float
        Single rt from ms2 spectrum in question.
    ms1rt : float
        numpy array of RTs for all MS2 spectra.

    Returns
    -------
    closest_idx : int
        Index of cosest MS1 spectrum in RT space.

    """
    closest_idx = np.argmin(np.abs(ms1rt-ms2rt))
    return closest_idx

def closest_peak_diff(mz,spec_mz_list,max_diff=2e-5):
    all_diffs = spec_mz_list-mz
    smallest_diff = all_diffs[np.argmin(np.abs(all_diffs))]/mz # rel diff
    if np.abs(smallest_diff)<max_diff:
        return smallest_diff
    else: 
        return np.nan
    
    

## split up the fragment name (b/y)(-loss)(frag index)_charge
def split_frag_name(ion_type):
    frag_name,frag_z = ion_type.split("_")
    loss_check = frag_name.split("-")
    loss = ""
    if len(loss_check)>1:
        frag_name,loss = loss_check
    frag_type = frag_name[0]
    frag_idx = int(frag_name[1:])
    
    return frag_type,frag_idx,loss,frag_z
    
########### Decoy functions

diann_rules = {
                'G':'L',
                 'A':'L',
                 'V':'L',
                 'L':'V',
                 'I':'V',
                 'F':'L',
                 'M':'L',
                 'P':'L',
                 'W':'L',
                 'S':'T',
                 'C':'S',
                 'T':'S',
                 'Y':'S',
                 'H':'S',
                 'K':'L',
                 'R':'L',
                 'Q':'N',
                 'E':'D',
                 'N':'Q',
                 'D':'E'
                 }

def change_seq(seq,rules):
    # seq: list of AAs
    # frags: dictionary of frags
    
    if type(seq)==str:
        seq = re.findall("[A-Z](?<!\([A-Z])",seq)
    else:
        seq = [re.sub("\(.*\)","",aa) for aa in seq]
        
    if rules=="diann":
        new_seq = "".join([diann_rules[aa] for aa in seq])
    elif rules=="rev":
        new_seq = "".join(seq[:-1][::-1]+seq[-1:])
    
    return new_seq

def change_seq(seq,rules):
    # seq: list of AAs
    # frags: dictionary of frags
    # re.findall("([A-Z](?:\(.*?\))?)",peptide)
    if type(seq)==str:
        seq = re.findall("([A-Z](?:\(.*?\))?)",seq)
    else:
        seq = [re.sub("\(.*\)","",aa) for aa in seq]
        
    if rules=="diann":
        new_seq = "".join([diann_rules[aa] for aa in seq])
    elif rules=="rev":
        new_seq = "".join(seq[:-1][::-1]+seq[-1:])
    
    return new_seq


def convert_prec_mz(seq,z):
    
    split_seq = re.findall("([A-Z](?:\(.*?\))?)",seq)
    
    mods = [re.findall("\((.*?)\)",i) for i in split_seq]
    
    ## assume AA is the first 
    unmod_seq = [i[0] for i in split_seq]
    
    unmod_mass = mass.fast_mass(unmod_seq,charge=z)
    
    if config.tag:
    	tag_masses = sum([config.tag.mass_dict[j] for i in mods for j in i if j in config.tag.mass_dict])
    else: 
    	tag_masses = 0
        
    return unmod_mass+(tag_masses)/z
    

def convert_frags_orig(seq,frags,rules):
    
    new_seq = change_seq(seq=seq,rules=rules)
    
    new_frags = {}
    
    for frag in frags:
        ion,charge = frag.split("_")
        ion_type = ion[0]
        ion_nmr = int(ion[1:])
        
        if ion_type=="b":
            mz = mass.fast_mass(new_seq[:ion_nmr],ion_type,int(charge))
        
        if ion_type=="y":
            mz = mass.fast_mass(new_seq[ion_nmr:],ion_type,int(charge))

        new_frags[frag] = [mz, frags[frag][1]]

    return new_frags


# def convert_frags(seq,frags,rules=diann_rules):
    
#     new_seq = change_seq(seq=seq,rules=rules)
    
#     new_frags = {}
    
#     for frag in frags:
#         ion,charge = frag.split("_")
#         ion_type = ion[0]
#         ion_nmr_loss = ion[1:]
#         ion_nmr_loss_split = ion_nmr_loss.split("-")
#         ion_nmr = int(ion_nmr_loss_split[0])
#         loss = 0
#         if len(ion_nmr_loss_split)>1:
#             ion_loss = ion_nmr_loss[1]
#             loss = mass.calculate_mass(ion_loss)
#         if ion_type=="b":
#             mz = mass.fast_mass(new_seq[:ion_nmr],ion_type,int(charge)) - loss
        
#         if ion_type=="y":
#             mz = mass.fast_mass(new_seq[-ion_nmr:],ion_type,int(charge)) - loss

#         new_frags[frag] = [mz, frags[frag][1]]

#     return new_frags





## update to work for mTRAQ
## Note: unsure if this works for modifications
####  TO DO:   Need to add tag masses to larger dict with modifications included 

def convert_frags(seq,frags,rules=diann_rules):
    
    new_seq = change_seq(seq=seq,rules=rules)    
    
    split_seq = re.findall("([A-Z](?:\(.*?\))?)",new_seq)
    
    mods = [re.findall("\((.*?)\)",i) for i in split_seq]
    
    ## assume AA is the first 
    unmod_seq = [i[0] for i in split_seq]
    
    if config.tag:
    	tag_masses = [sum([config.tag.mass_dict[j]  for j in i if j in config.tag.mass_dict]) for i in mods]
    else:
    	tag_masses = [0 for i in mods]
        
    new_frags = {}
    
    for frag in frags:
        ion,charge = frag.split("_")
        ion_type = ion[0]
        ion_nmr_loss = ion[1:]
        ion_nmr_loss_split = ion_nmr_loss.split("-")
        ion_nmr = int(ion_nmr_loss_split[0])
        loss = 0
        if len(ion_nmr_loss_split)>1:
            ion_loss = ion_nmr_loss[1]
            try:
                loss = mass.calculate_mass(ion_loss)
            except:
                loss=0
        if ion_type=="b":
            mz = mass.fast_mass(unmod_seq[:ion_nmr],ion_type,int(charge)) - loss + sum(tag_masses[:ion_nmr])
        
        if ion_type=="y":
            mz = mass.fast_mass(unmod_seq[-ion_nmr:],ion_type,int(charge)) - loss + sum(tag_masses[-ion_nmr:])

        new_frags[frag] = [mz, frags[frag][1]]

    return new_frags


def hyperscore(frag_list,matches):
    num_b = sum(["b" in i for i,j in zip(frag_list,matches) if j])
    num_y = sum(["y" in i for i,j in zip(frag_list,matches) if j])
    dp = np.sum(frag_to_peak(frag_list)[:,1][matches])
    return max(0,np.log(dp*np.math.factorial(num_b)*np.math.factorial(num_y)))

def hyperscore2(frag_list,matches):
    num_b = sum(["b" in i for i,j in zip(frag_list,matches) if j])
    num_y = sum(["y" in i for i,j in zip(frag_list,matches) if j])
    dp = np.sum(frag_to_peak(frag_list)[:,1][matches])
    return max(0,np.log(dp*np.math.factorial(num_b)*np.math.factorial(num_y)))



def cosim(x,y):
    assert len(x)==len(y)
    x = np.squeeze(x)
    y = np.squeeze(y)
    
    return np.dot(x,y)/(np.sqrt(np.sum(np.power(x,2)))*np.sqrt(np.sum(np.power(y,2))))




import warnings

from scipy import stats

def calculate_corr(key,library,fragment_clusters,quad_mz_pairs,rt_tol,mz_tol=20e-6):

    precursor = library[key]
    rt = precursor["iRT"]
    prec_mz = precursor["prec_mz"]
    low_quad,high_quad = quad_mz_pairs[np.argmin(1/(quad_mz_pairs[:,0]-prec_mz))]
    im = precursor['IonMob']

    frags = precursor["spectrum"][:,0]
    # print(f"IM:{precursor['IonMob']}")
    peaks = fragment_clusters[np.logical_and(np.abs(fragment_clusters.rt_weighted_average-(rt*60))<rt_tol,
                                             fragment_clusters.quad_low_mz_values==low_quad)]
    # plt.scatter(peaks.mz_weighted_average,peaks.im_weighted_average,s=1)

    summed_intensities = []
    im_diffs = []
    
    for fr_idx,frag in enumerate(frags):
        matched_peaks = peaks[np.abs((peaks.mz_weighted_average-frag)/frag)<mz_tol][["im_weighted_average","summed_intensity"]]
        # print(matched_peaks)
        if len(matched_peaks)>0:
            peak_idx = np.argmin(np.abs(im-matched_peaks["im_weighted_average"]))
            summed_intensities.append(matched_peaks["summed_intensity"].iloc[peak_idx])
            im_diffs.append(im-matched_peaks["im_weighted_average"].iloc[peak_idx])
        else:
            summed_intensities.append(0)
            im_diffs.append(10)
            # missing_values.append(precursor["spectrum"][fr_idx,0])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p_corr = stats.pearsonr(summed_intensities,precursor["spectrum"][:,1]).statistic
    
    return [p_corr,summed_intensities,im_diffs,precursor["spectrum"]]




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




def frag_to_peak(frag_dict,return_frags=False):
    peaks = np.array(list(frag_dict.values()))
    order = np.argsort(peaks[:,0])
    if return_frags:
        ordered_frags = np.array(list(frag_dict.keys()))[order]
        return peaks[order],ordered_frags
    else:
        return peaks[order]


non_specific_frags = ["b1","b2","y1","y2","y3"]
def specific_frags(frag_dict,non_spec =non_specific_frags):
    peaks = []
    for frag in frag_dict:
        frag_type,frag_idx,loss,frag_z = split_frag_name(frag)
        if frag_type+str(frag_idx) not in non_spec:
            peaks.append(frag_dict[frag])
    
    peaks = np.array(peaks)
    order = np.argsort(peaks[:,0])
    return peaks[order]
            

def ordered_frags(frag_dict):
    peaks = np.array(list(frag_dict.values()))
    order = np.argsort(peaks[:,0])
    frags = list(frag_dict.keys())
    return {frags[i]:peaks[i] for i in order}
