"""
This Source Code Form is subject to the terms of the Oxford Nanopore
Technologies, Ltd. Public License, v. 1.0.  Full licence can be found
at https://github.com/ParallelSquared/JMod/blob/main/LICENSE.txt
"""

import numpy as np
import numpy.typing as npt
import csv
import re
from pyteomics import mass
import src.config as config
import pandas as pd 
from src.logger import logger
def feature_list_rt(DinoDF,rt,rt_tol): 
    """
    Description:

    Parameters:
    ----------
    DinoDF : type?
        description...
    rt : type?
        description...
    rt_tol : type?
        description...

    Returns:
    -------
    description...
    
    Notes:
    ------
    __make_docstring__

    import pprint
    print("DinoDF:")
    pprint.pprint(DinoDF)
    print("Type:", type(DinoDF))
    print("Structure Example:", list(DinoDF.items())[:1])  # show one entry if possible
    print()
    Force error after debugging output
    raise RuntimeError("Intentional debug stop after printing input information.")
    """

    # _bool=np.logical_and(DinoDF["rtStart"]-rt_tol<rt,DinoDF["rtEnd"]+rt_tol>rt)
    _bool=np.abs(DinoDF["rtApex"]-rt)<rt_tol

    return(DinoDF[_bool])
    
# def feature_list_mz(DinoDF,window): 
#     _bool=np.logical_and(DinoDF["mz"]>window[0],DinoDF["mz"]<window[1])
def feature_list_mz(DinoDF,window_mz,window_width): 
    """
    Description:

    Parameters:
    ----------
    DinoDF : type?
        description...
    window_mz : type?
        description...
    window_width : type?
        description...

    Returns:
    -------
    description...
    
    Notes:
    ------
    __make_docstring__

    import pprint
    
    print("var_name:")
    pprint.pprint(var_name)
    print("Type:", type(var_name))
    print("Structure Example:", list(var_name.items())[:1])  # show one entry if possible
    print()
    Force error after debugging output
    raise RuntimeError("Intentional debug stop after printing input information.")
    """
    _bool=np.abs(DinoDF["mz"]-window_mz)<(window_width/2)
    return(DinoDF[_bool])
    
def window_width(spec):
    """
    Description:

    Parameters:
    ----------
    spec : type?
        description...

    Returns:
    -------
    description...
    
    Notes:
    ------
    __make_docstring__

    import pprint
    
    print("var_name:")
    pprint.pprint(var_name)
    print("Type:", type(var_name))
    print("Structure Example:", list(var_name.items())[:1])  # show one entry if possible
    print()
    Force error after debugging output
    raise RuntimeError("Intentional debug stop after printing input information.")
    """
    w1,w2 = spec.ms1window
    return w2-w1
    

def createTolWindows(positions,tolerance):
    """
    Description:

    Parameters:
    ----------
    positions : type?
        description...
    tolerance : type?
        description...

    Returns:
    -------
    description...
    
    Notes:
    ------
    __make_docstring__

    import pprint
    
    print("var_name:")
    pprint.pprint(var_name)
    print("Type:", type(var_name))
    print("Structure Example:", list(var_name.items())[:1])  # show one entry if possible
    print()
    Force error after debugging output
    raise RuntimeError("Intentional debug stop after printing input information.")
    """ 
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
    


def write_to_csv(data: list[list],filepath: str,colnames: list[str] = None):
    """
    Description:

    Parameters:
    ----------
    data : list[list[float/int/complex]]
        Row major matrix of numbers as list  
    filepath : string
        Where to write the csv file?
    colnames : list[str]
        List of column names. Must correspond to the number of columns in 'data'

    Returns:
    -------
    None. Writes the data to a CSV at filepath with 'colnames' as the header if not None

    """
    with open(filepath,"a", newline='') as write_file:
        writer = csv.writer(write_file)
        if colnames is not None:
            if len(colnames)==len(data[0]):
                writer.writerow(colnames)
            else:
                logger.warning("WARNING: Column names do not match data columns. Skipping header row.")
        writer.writerows(data)
        
        
        
def within_tol(x: list[float], y: list[float], atol: float, rtol: float) -> npt.NDArray[np.float64]:
    """
    Compare two lists of floats element-wise to determine if each pair of values 
    is within a specified absolute and relative tolerance.

    Parameters:
    ----------
    x : list[float]
        First list of float values.
    y : list[float]
        Second list of float values to compare against.
    atol : float
        Absolute tolerance.
    rtol : float
        Relative tolerance.

    Returns:
    -------
    np.ndarray
        A NumPy array of shape (n, 2) where:
        - Column 0 contains boolean values indicating whether the corresponding
          elements in x and y are within the specified tolerance.
        - Column 1 contains the raw differences (x - y) for each pair.
    
    Notes:
    ------
    Potentially performance issues 
    """
    x = np.asanyarray(x)
    y = np.asanyarray(y)
    diff = x - y
    logic = np.less_equal(abs(diff), atol + rtol * np.abs(y))
    log_dif = np.zeros((*logic.shape, 2))
    log_dif[..., 0] = logic
    log_dif[..., 1] = diff
    return log_dif
    

def get_diff(mz,peaks,tol):
    """
    Description:

    Parameters:
    ----------
    mz : type?
        description...
    peaks : type?
        description...
    tol : type?
        description...

    Returns:
    -------
    description...
    
    Notes:
    ------
    __make_docstring__

    import pprint
    
    print("var_name:")
    pprint.pprint(var_name)
    print("Type:", type(var_name))
    print("Structure Example:", list(var_name.items())[:1])  # show one entry if possible
    print()
    Force error after debugging output
    raise RuntimeError("Intentional debug stop after printing input information.")
    """
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

#@profile
# def closest_peak_diff(mz,spec_mz_list,max_diff=2e-5):
#     all_diffs = spec_mz_list-mz
#     smallest_diff = all_diffs[np.argmin(np.abs(all_diffs))]/mz # rel diff
#     if np.abs(smallest_diff)<max_diff:
#         return smallest_diff
#     else: 
#         return np.nan
# #@profile   
def closest_peak_diff(mz: np.float64,spec_mz_list: npt.NDArray[np.float64],max_diff: float=2e-5):
    """
    Description:

    Parameters:
    ----------
    mz : np.float64
        m/z of the query peak 
    peaks : npt.NDArray[np.float64]
        sorted list of m/z values from the spectrum (ascending)
    max_diff : the mass tolerance for the peak search
        the mass tolerance for the peak search

    Returns:
    -------
    np.float64
        Smallest relative difference between the query m/z and the nearest peak in the spectrum `peaks`. 
        Returns np.nan if no match within the max_diff tolerance 
    """

    # all_diffs = spec_mz_list-mz
    # smallest_diff = all_diffs[np.argmin(np.abs(all_diffs))]/mz # rel diff
    order_idx = np.searchsorted(spec_mz_list, mz)
    
    # Handle edge cases for indices at the bounds
    if order_idx == 0:
        mz_diff = (spec_mz_list[0]-mz)/mz
    elif order_idx == len(spec_mz_list):
        mz_diff = (spec_mz_list[-1]-mz)/mz
    else:
        # Compare the closest values on both sides of the searchsorted index
        left_idx = order_idx - 1
        right_idx = order_idx
        
        # Find the closest value between the two neighboring indices
        left_diff = spec_mz_list[left_idx] - mz
        right_diff = spec_mz_list[right_idx] - mz
        if abs(left_diff) < abs(right_diff):
            mz_diff = left_diff/mz
        else:
            mz_diff = right_diff/mz
    if (-max_diff)<=mz_diff<=max_diff:
        return mz_diff
    else: 
        return np.nan
    


def hyperscore_b_y(frag_list: dict[str, list[float]],matches: npt.NDArray[np.bool_]) -> tuple[float, int, int]:
    """
    Description:
        Given a list of fragments, their theoretical m/z, theoretical intensities, and a list indicating
        which fragments matched peaks in the spectrum, calcualte the number of matched b and y ions as
        well as the hyperscore. 
    Parameters:
    ----------
    frag_list : dict[str, list[float]]
        Dictionary mapping fragment names for a precursor to a list pair containng the m/z ratio and relative intensity.
    matches : npt.NDArray[np.bool_]
        Same length as the keys in the dictionary. In order of smallest to largest fragment by m/z. Indicates which of the fragments in the frag_list matched peaks
        in the spectrum. 

    Returns:
    -------
    hyperscore, matched b-ions, matched y-ions, tuple[float, int, int]
    
    Example:
    --------
    >>> frag_list= {
        'b3_1': [244.09279699588, 0.25249937],
        'b4_1': [372.15137450116, 0.45614848],
        'b5_1': [485.23543847829, 0.27464142],
        'b6_1': [542.25690219886, 0.07701804],
        'y4_1': [472.2514228956, 0.31066182],
        'y5_1': [529.27288661617, 1.0],
        'y6_1': [642.3569505933, 0.50503737],
        'y7_1': [770.41552809858, 0.14570464],
        'y8_1': [827.43699181915, 0.3309042],
        'y9_1': [956.47958490712, 0.054985035]
        }
    >>> matches = array([ True,  True,  True,  True,  True, False,  True,  True,  True,
       False])
    Notes:
    ------
    Not an efficient implementation. 'frag_to_peak' allocates an array to get the sorted indices of the 
    fragments by their m/z so that they correspond to matches. 
    Potential fix: each frag list could have three elements, the last is the 'match' indicator. Should also be 
    a dict of tuples and not lists. 
    """
    #num_b = sum(["b" in i for i,j in zip(frag_list,matches) if j])
    #num_y = sum(["y" in i for i,j in zip(frag_list,matches) if j])
    #dp = np.sum(frag_to_peak(frag_list)[:,1][matches])
    #return max(0,np.log(dp*np.math.factorial(num_b)*np.math.factorial(num_y))), num_b, num_y
    num_b, num_y = 0, 0
    for (i, j) in zip(frag_list, matches):
        if j:
            if "b" in i:
                num_b += 1
            elif "y" in i:
                num_y += 1

    #Not efficient to have to allocate and sort every time. 
    dp = np.sum(frag_to_peak(frag_list)[:,1][matches])
    return max(0,np.log(dp*np.math.factorial(num_b)*np.math.factorial(num_y))), num_b, num_y

def longest_y(frag_list: dict[str, list[float]],matches: npt.NDArray[np.bool_]) -> int:
    """
    Description:
        Given a list of fragments, gets the index of the longest y ion that matched peaks in the spectrum.

    Parameters:
    ----------
    frag_list : dict[str, list[float]]
        Dictionary mapping fragment names for a precursor to a list pair containng the m/z ratio and relative intensity.
    matches : npt.NDArray[np.bool_]
        Same length as the keys in the dictionary. In order of smallest to largest fragment by m/z. Indicates which of the fragments in the frag_list matched peaks
        in the spectrum. 

    Returns:
    -------
    Index of longest y ion 
    
    Example:
    --------
    >>> frag_list= {
        'b3_1': [244.09279699588, 0.25249937],
        'b4_1': [372.15137450116, 0.45614848],
        'b5_1': [485.23543847829, 0.27464142],
        'b6_1': [542.25690219886, 0.07701804],
        'y4_1': [472.2514228956, 0.31066182],
        'y5_1': [529.27288661617, 1.0],
        'y6_1': [642.3569505933, 0.50503737],
        'y7_1': [770.41552809858, 0.14570464],
        'y8_1': [827.43699181915, 0.3309042],
        'y9_1': [956.47958490712, 0.054985035]
        }
    >>> matches = array([ True,  True,  True,  True,  True, False,  True,  True,  True,
       False])

    Notes:
    ------
    This only works if the dictionary is sorted to match the order of matches. Need to keep the two data strutures together. Ideally the 'matched'
    field is another value in the dictionary. Otherwise it's possible the 'frag_list' and 'matches' won't be in the same order. See 'frag_to_peak' function.
    """
    def extract_fragment_number(fragment_name):
        # Check if fragment_name is a string
        if not isinstance(fragment_name, str):
            return 0  # Return 0 for non-string inputs
            
        # Pattern matches 'b' or 'y' followed by digits, optionally followed by underscore and more characters
        pattern = r"^[by](\d+)(?:[_-].*)?$"
        match = re.match(pattern, fragment_name)
        if match:
            # Convert the captured digits to an integer
            return int(match.group(1))
        return 0
    
    max_y = 0
    for frag_name, match_value in zip(frag_list, matches):
        if match_value and frag_name.startswith('y'):
            # Extract the fragment number from the fragment name
            frag_number = extract_fragment_number(frag_name)
            if frag_number > max_y:
                max_y = frag_number

    return max_y 

def cosim(x: npt.NDArray[np.float64],y: npt.NDArray[np.float64]) -> np.float64:
    """
    Description:
        Calculates the cosine similarity between two vectors x and y.
    Parameters:
    ----------
    x : type?
        description...
   y : type?
        description...

    Returns:
    -------
    description...
    
    Notes:
    ------
    """
    assert len(x)==len(y)
    x = np.squeeze(x)
    y = np.squeeze(y)
    return np.dot(x,y)/(np.sqrt(np.sum(np.power(x,2)))*np.sqrt(np.sum(np.power(y,2))))





import warnings

from scipy import stats



def frag_to_peak(frag_dict: dict[str, list[float]],return_frags: bool=False): #->npt.NDArray[np.float64]
    """
    Description:
        Sorts the values (m/z and intensities) rows of the frag_dict by their m/z and returns
        a matrix with a row for each fragment and columns for the m/z and intensity.
    Parameters:
    ----------
    frag_list : dict[str, list[float]]
        Dictionary mapping fragment names for a precursor to a list pair containng the m/z ratio and relative intensity.
    return_frags : bool
        Wheather to also return the dictionary keys sortedy by their values (m/z)

    Returns:
    -------
    peaks : np.ndarray
        A 2D numpy array with the first column as m/z values and the second column as intensities, sorted by m/z.
    
    Example:
    --------
    >>> {
        'b3_1': [329.16081698605, 0.25633228],
        'b3_1_iso1': [330.1637526155811, 0.05175954048391056],
        'b4_1': [458.20341007402, 0.12788501],
        'b4_1_iso1': [459.20636583855736, 0.033455156427240874],
        'y3_1': [401.28708012833, 0.13112698],
        'y3_1_iso1': [402.2898720399767, 0.029145210435823944],
        'y4_1': [564.35040866088, 0.40015215],
        'y4_1_iso1': [565.3533188802813, 0.13007301732202506],
        'y5_1': [693.39300174885, 0.723387],
        'y5_1_iso1': [694.395929600629, 0.278314979784522],
        'y6_1': [764.43011553356, 1.0],
        'y6_1_iso1': [765.4330309212257, 0.4217951826932419],
        'y7_1': [835.46722931827, 0.840592],
        'y7_1_iso1': [836.4701342550052, 0.38570703236196757]
    }
    >>> frag_to_peak(frag_list)
    array([[3.29160817e+02, 2.56332280e-01],
       [3.30163753e+02, 5.17595405e-02],
       [4.01287080e+02, 1.31126980e-01],
       [4.02289872e+02, 2.91452104e-02],
       [4.58203410e+02, 1.27885010e-01],
       [4.59206366e+02, 3.34551564e-02],
       [5.64350409e+02, 4.00152150e-01],
       [5.65353319e+02, 1.30073017e-01],
       [6.93393002e+02, 7.23387000e-01],
       [6.94395930e+02, 2.78314980e-01],
       [7.64430116e+02, 1.00000000e+00],
       [7.65433031e+02, 4.21795183e-01],
       [8.35467229e+02, 8.40592000e-01],
       [8.36470134e+02, 3.85707032e-01]])
    Notes:
    ------
    Not an efficient implementation. 'frag_to_peak' allocates an array to get the sorted indices of the 
    fragments by their m/z so that they correspond to matches. 
    Potential fix: each frag list could have three elements, the last is the 'match' indicator. Should also be 
    a dict of tuples and not lists. 
    """
    peaks = np.array(list(frag_dict.values()))
    order = np.argsort(peaks[:,0])
    if return_frags:
        ordered_frags = np.array(list(frag_dict.keys()))[order]
        return peaks[order],ordered_frags
    else:
        return peaks[order]

def specific_frags(frag_dict: dict[str, list[float]],non_spec: list[str] =["b1","b2","y1","y2","y3"]) -> npt.NDArray[np.float64]:
    """
    Description:
        Removes fragments from the fragment dictionary that are 'non-specific' (e.g., b1, b2, y1, y2, y3). That is,
    they are likely to have interference from other peptides. 

    Parameters:
    ----------
    frag_list : dict[str, list[float]]
        Dictionary mapping fragment names for a precursor to a list pair containng the m/z ratio and relative intensity.
    non_spec : list[str]
        Fragments that are likely to be non-specific 

    Returns:
    -------
    frags : dict[str, list[float]]
        Filtered 'frag_dict' from input 
    
    Example:
    --------
    >>> frag_list = {
        'b3_1': [329.16081698605, 0.25633228],
        'b3_1_iso1': [330.1637526155811, 0.05175954048391056],
        'b4_1': [458.20341007402, 0.12788501],
        'b4_1_iso1': [459.20636583855736, 0.033455156427240874],
    }
    >>> specific_frags(frag_list, non_spec=["b1","b2","y1","y2","y3"]))
    [
    [458.20341007402, 0.12788501],
    [459.20636583855736, 0.033455156427240874],
    ]
    """
    peaks = []
    for frag in frag_dict:
        frag_type,frag_idx,loss,frag_z = split_frag_name(frag)
        if frag_type+str(frag_idx) not in non_spec:
            peaks.append(frag_dict[frag])
    
    peaks = np.array(peaks)
    order = np.argsort(peaks[:,0])
    return peaks[order]
            

def ordered_frags(frag_dict):
    """
    Description:
        Sorts the dictionary of fragments by the m/z values  

    Parameters:
    ----------
    frag_list : dict[str, list[float]]
        Dictionary mapping fragment names for a precursor to a list pair containng the m/z ratio and relative intensity.

    Returns:
    -------
    frags : dict[str, list[float]]
        Sorted 'frag_dict' from input 
    
    Example:
    --------

    >>> frag_list = {
        'b4_1': [458.20341007402, 0.12788501],
        'b4_1_iso1': [459.20636583855736, 0.033455156427240874],
        'b3_1': [329.16081698605, 0.25633228],
        'b3_1_iso1': [330.1637526155811, 0.05175954048391056],
    }
    >>> ordered_frags(frag_list)
    {
        'b3_1': [329.16081698605, 0.25633228],
        'b3_1_iso1': [330.1637526155811, 0.05175954048391056],
        'b4_1': [458.20341007402, 0.12788501],
        'b4_1_iso1': [459.20636583855736, 0.033455156427240874],
    }
    """
    peaks = np.array(list(frag_dict.values()))
    order = np.argsort(peaks[:,0])
    frags = list(frag_dict.keys())
    return {frags[i]:peaks[i] for i in order}
### source: https://stackoverflow.com/questions/62817623/improve-speed-of-scipy-pearson-correlation-for-many-pairwise-calculations
from scipy.special import betainc

# import cupy as xp
# from cupyx.scipy.special import betainc

class p_result:
    def __init__(self,r_sq,p=0):
        self.statistic = r_sq
        self.pvalue=p
    def __repr__(self):
        return f"PearsonRResult(statistic={self.statistic}, pvalue={self.pvalue})"
    
## source https://cancerdatascience.org/blog/posts/pearson-correlation/
def np_pearson_cor(x: list[float], y: list[float]) -> p_result:
    """
    Notes:
    ------
    Inputs should be as numpy arrays 
    """
    assert len(x)==len(y)
    n=len(x)
    x_np = np.array(x)
    y_np = np.array(y)
    xv = x_np - x_np.mean(axis=0)
    yv = y_np - y_np.mean(axis=0)
    xvss = (xv * xv).sum(axis=0)
    yvss = (yv * yv).sum(axis=0)
    
    if xvss==0 or yvss==0:
        return p_result(np.nan,1)
    result = np.matmul(xv.transpose(), yv) / np.sqrt(np.multiply(xvss, yvss))
    
    ## from scipy## TOO SLOW
    ab = n/2 - 1
    # dist = stats.beta(ab, ab, loc=-1, scale=2)
    # p = 2*dist.cdf(-abs(result))
    p= 2*(1-betainc(ab, ab, (abs(result) + 1)/2)) 
    
    # bound the values to -1 to 1 in the event of precision issues
    return p_result(np.maximum(np.minimum(result, 1.0), -1.0),p)



def string_floats(arr,delim=";"):
    assert type(delim)==str
    return delim.join(map(str,arr))

def unstring_floats(string,delim=";"):
    assert type(string)==str
    assert type(delim)==str
    return np.array([*map(float,string.split(delim))])



from sklearn.ensemble import RandomForestClassifier
def fit_model(data):
    rf = RandomForestClassifier(n_estimators = 100, max_depth=10)
    rf.fit(*data)
    return rf



from scipy.interpolate import interp1d
import statsmodels.api as sm


def fragment_cor(df: pd.core.frame.DataFrame,didx: int,fn: str="cos") -> np.float64:
    """
    Description:
        The data frame 'df' has a row for each psm. The columns 'obs_int' and 'frag_int' contain the observed and
        theoretical fragment intensities for peaks that matched the spectrum. These are in a ';' delimited list in string format. 
        parses the strings into a vector and gets cosine similarity or pearson correlation between the two vectors.

    Parameters:
    ----------
    df: pd.core.frame.DataFrame
        DataFrame of psms that contains the columns 'frag_names', 'obs_int', and 'frag_int'.
    didx : int
        Row of the DataFrame that contains the psms 
    fn : str
        Function to use for the correlation. Either 'cos' for cosine similarity or 'pearson' for pearson correlation.

    Returns:
    -------
    Index of longest y ion 
    
    Example:
    --------
    >>> pprint.pprint(df.loc[:,['frag_names', 'obs_int', 'frag_int']].iloc[didx])
        frag_names              b3_1;b4_1;y4_1;b5_1;y5_1;y6_1;y7_1;y8_1
        obs_int       5595.63916015625;1993.73876953125;15022.671875...
        frag_int      0.25249937;0.45614848;0.31066182;0.27464142;1....

    Notes:
    ------
    Parsing strings is slow 
    """
    #d1={i[0]:i[1] for i in zip(df.iloc[didx].frag_names.split(";"),unstring_floats(df.iloc[didx].obs_int))}    
    #d2={i[0]:i[1] for i in zip(df.iloc[didx].frag_names.split(";"),unstring_floats(df.iloc[didx].frag_int))}
        # Find the column indices based on their names
    try:
        frag_names_idx = df.columns.get_loc("frag_names")
        obs_int_idx = df.columns.get_loc("obs_int")
        frag_int_idx = df.columns.get_loc("frag_int")
        
        # Use positions instead of names
        d1 = {i[0]:i[1] for i in zip(df.iloc[didx, frag_names_idx].split(";"), 
                                     unstring_floats(df.iloc[didx, obs_int_idx]))}
        d2 = {i[0]:i[1] for i in zip(df.iloc[didx, frag_names_idx].split(";"), 
                                     unstring_floats(df.iloc[didx, frag_int_idx]))}
        
        # Rest of function...
    except (KeyError, AttributeError):
        # Return default value if columns not found
        return 0
    # include_frags = {i:j for i,j in d2.items() if j>.05}
    shared_d = set(d1).intersection(set(d2))
    # shared_d = {i for i in shared_d for j in include_frags if j  in i}
    if fn=="cos":
        return cosim(np.array([d1[i] for i in shared_d]),np.array([d2[i] for i in shared_d]))
    else: 
        return np_pearson_cor(np.array([d1[i] for i in shared_d]),np.array([d2[i] for i in shared_d])).statistic
    
    
