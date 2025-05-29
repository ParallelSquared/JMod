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
from scipy import signal
from scipy.optimize import curve_fit
import config
import line_profiler

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
                println("WARNING: Column names do not match data columns. Skipping header row.")
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
    
def parse_peptide(seq: str) -> list[str]:
    """Parse a peptide sequence into individual amino acids with their modifications.
    
    This function takes a peptide sequence string that may contain modifications
    in brackets (parentheses or square brackets) and splits it into a list where
    each element is an amino acid, potentially with its associated modifications.
    Modifications are kept attached to their preceding amino acid.
    
    Parameters
    ----------
    seq : str
        A peptide sequence string, potentially containing modifications in 
        parentheses () or square brackets []. 
        Examples: "PEPTIDE", "PEP(+79.97)TIDE", "K[+42]PEPTIDE"
    
    Returns
    -------
    list[str]
        A list where each element is a single amino acid (single letter code)
        potentially followed by its modification(s) in brackets.
        
    Examples
    --------
    >>> parse_peptide("PEPTIDE")
    ['P', 'E', 'P', 'T', 'I', 'D', 'E']
    
    >>> parse_peptide("PEP(+79.97)TIDE")
    ['P', 'E', 'P(+79.97)', 'T', 'I', 'D', 'E']
    
    >>> parse_peptide("K(mTRAQ)PEPTIDE(+15.99)R")
    ['K(mTRAQ)', 'P', 'E', 'P', 'T', 'I', 'D', 'E(+15.99)', 'R']
    
    >>> parse_peptide("C[+57.02]PEPTIDE")
    ['C[+57.02]', 'P', 'E', 'P', 'T', 'I', 'D', 'E']
    
    >>> parse_peptide("")
    []
    
    Notes
    -----
    - Modifications must be properly closed with matching brackets
    - Nested brackets are handled correctly
    - If a sequence starts with a modification (edge case), it's added as a standalone element
    - Both parentheses () and square brackets [] are supported for modifications
    """
    ### extract all [] or () from seq with preceding AA
    close_d = {"[": "]", "(": ")"}
    new_seq = []
    s_idx = 0
    current = ""

    while s_idx < len(seq):
        s = seq[s_idx]

        if s in "[(":
            # Handle modifications in brackets
            opener = s
            bracket_content = s
            s_idx += 1
            while s_idx < len(seq) and seq[s_idx] != close_d[opener]:
                bracket_content += seq[s_idx]
                s_idx += 1
            bracket_content += seq[s_idx]  # Add closing bracket

            if current:  
                current += bracket_content  # Attach modification to the preceding letter
            else:  
                new_seq.append(bracket_content)  # Handle edge cases where the sequence starts with a modification

        else:
            if current:
                new_seq.append(current)  # Save previous character/modification
            current = s  # Start a new character

        s_idx += 1

    if current:  # Append the last residue
        new_seq.append(current)

    return new_seq
    

def extract_mod(AA: str) -> list[str]:
    """Extract all modifications from an amino acid string.
    
    This function takes a string representing an amino acid with potential
    modifications in brackets (parentheses or square brackets) and returns
    a list of all modifications found, including their enclosing brackets.
    
    Parameters
    ----------
    AA : str
        A string representing an amino acid with potential modifications.
        Can be just an amino acid letter (e.g., "K") or an amino acid
        with modifications in parentheses or square brackets.
        Examples: "K", "K(mTRAQ)", "C[+57.02]", "K(mTRAQ)(+42)"
    
    Returns
    -------
    list[str]
        A list of all modifications found in the input string, where each
        modification includes its enclosing brackets. Returns an empty list
        if no modifications are found.
        
    Examples
    --------
    >>> extract_mod("K")
    []
    
    >>> extract_mod("K(mTRAQ)")
    ['(mTRAQ)']
    
    >>> extract_mod("E(+15.99)")
    ['(+15.99)']
    
    >>> extract_mod("C[+57.02]")
    ['[+57.02]']
    
    >>> extract_mod("K(mTRAQ)(+42)")
    ['(mTRAQ)', '(+42)']
    
    >>> extract_mod("S(Phospho)[+80]")
    ['(Phospho)', '[+80]']
    
    Notes
    -----
    - The function preserves the bracket type (parentheses or square brackets)
    - Multiple modifications on the same amino acid are returned as separate elements
    - The function handles nested brackets correctly using a bracket counter
    - Modifications are returned in the order they appear in the input
    - The amino acid letter itself is not included in the output
    
    See Also
    --------
    parse_peptide : Parse entire peptide sequences with modifications
    """
    close_d = {"[":"]","(":")"}
    mods = []
    s_idx=0
    current= ""
    b_count = 0
    while s_idx<len(AA):
        s= AA[s_idx]
        if s_idx==0 and s not in "[(":
            pass
        elif s_idx!=0 and s not in "[(":
            mods.append(current)
            current= s
        
        elif s in "[(":
            b_count +=1
            opener = s
            current+=s
            while s!=close_d[opener] and b_count!=0:
                if s==close_d[opener]:
                    b_count-=1
                elif s in "[(":
                    b_count+=1
                s_idx+=1
                s= AA[s_idx]
                current+=s
            mods.append(current)
            current= ""
        s_idx+=1
    return mods
        
## split up the fragment name (b/y)(-loss)(frag index)_charge
def split_frag_name(ion_type: str) -> tuple[str, int, str, str]:
    """Split a fragment ion name into its component parts.
    
    This function parses mass spectrometry fragment ion notation and extracts
    the ion type, fragment index, optional neutral loss, and charge state.
    Fragment names follow the format: (ion_type)(index)(-loss)_(charge)
    where the loss component is optional.
    
    Parameters
    ----------
    ion_type : str
        A fragment ion name in standard mass spectrometry notation.
        Format: (b/y/a/c/x/z)(index)(-loss)_(charge)
        Examples: "b5_2", "y10-H2O_1", "b3-NH3_2", "y7_1"
    
    Returns
    -------
    tuple[str, int, str, str]
        A tuple containing four elements:
        - frag_type (str): The ion series type (e.g., 'b', 'y', 'a', 'c', 'x', 'z')
        - frag_idx (int): The fragment index number
        - loss (str): The neutral loss notation (e.g., 'H2O', 'NH3'), empty string if no loss
        - frag_z (str): The charge state as a string (e.g., '1', '2', '3')
    
    Examples
    --------
    >>> split_frag_name("b5_2")
    ('b', 5, '', '2')
    
    >>> split_frag_name("y10-H2O_1")
    ('y', 10, 'H2O', '1')
    
    >>> split_frag_name("b3-NH3_2")
    ('b', 3, 'NH3', '2')
    
    >>> split_frag_name("a12_3")
    ('a', 12, '', '3')
    
    >>> split_frag_name("y7-H3PO4_1")
    ('y', 7, 'H3PO4', '1')
    
    Notes
    -----
    - The function expects the charge to be separated by an underscore '_'
    - Neutral losses are indicated by a hyphen '-' after the fragment index
    - Common neutral losses include H2O (water), NH3 (ammonia), H3PO4 (phosphoric acid)
    - The function supports standard ion series: a, b, c (N-terminal) and x, y, z (C-terminal)
    - The charge state is returned as a string, not an integer
    
    Raises
    ------
    ValueError
        If the input string doesn't contain an underscore to separate charge state
    
    See Also
    --------
    fragment_seq : Uses split_frag_name to determine fragment sequences
    convert_frags : Processes fragment information for decoy generation
    """
    frag_name,frag_z = ion_type.split("_")
    loss_check = frag_name.split("-")
    loss = ""
    if len(loss_check)>1:
        frag_name,loss = loss_check
    frag_type = frag_name[0]
    frag_idx = int(frag_name[1:])
    
    return frag_type,frag_idx,loss,frag_z
    
########### Decoy functions
#Store somewhere easier to find. 
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

def change_seq(seq: str, rules: str) -> str:
    """Modifies a peptide sequence to create a complementary decoy sequence.
    Uses either the sequence reversal method or the DIA-NN rules for sequence
    'mutation' 

    Parameters
    ----------
    seq : str
        The original peptide sequence 
    rules : str
        Method to modify string (must either be 'rev' or 'diann')

    Returns
    -------
    str
        The modified sequence 
    
    Example
    -------
    >>> change_seq("PEPTIDE", "diann")
    "LDLSVED"
    >>> change_seq("PEPTIDE", "rev")
    "LDLSVED"
    """
    # seq: list of AAs
    # frags: dictionary of frags
    # re.findall("([A-Z](?:\(.*?\))?)",peptide)
    if type(seq)==str:
        seq = parse_peptide(seq)
    # else:
    #     seq = [re.sub("\(.*\)","",aa) for aa in seq]\
        
    if config.tag:   
       tags = [re.findall(f"(\({config.tag.name}.*?\))",i) for i in seq]
       seq = [re.sub(f"(\({config.tag.name}.*?\))","",i) for i in seq]
    else:
        tags = [[] for i in seq]
        
    #mods = [extract_mod(i) for i in seq]
    ## assume AA is the first 
    #untag_seq = [i[0] for i in seq]
    
    if rules=="diann":
        new_split_seq = [diann_rules[aa] for aa in seq]
    elif rules=="rev":
        new_split_seq = seq[:-1][::-1]+seq[-1:]
    else:
        raise ValueError("Unavailable rules selected")
    # elif rules==None:
    #     new_seq = "".join(seq)
    
    new_seq = "".join([i+"".join(j) for i,j in zip(new_split_seq,tags)])
    
    return new_seq


def convert_prec_mz(seq: str, z: int = 1, mass_dict: dict[str, float] = {}) -> float:
    """Gets the precursors m/z for a peptide sequence given the
    amino acid sequence and the charge state. Accepts sequences with modifications
    in '(' brackets  

    Parameters
    ----------
    seq : str
        The original peptide sequence 
    z : int
        The charge state of the peptide (default is 1)
    mass_dict : dict[str, float], optional
        A dictionary mapping modification tags (e.g., 'Phospho', 'mTRAQ-0') to their mass shifts.
        Defaults to empty dict.

    Returns
    -------
    float
        The unmodified m/z ratio of the peptide 
    
    Example
    -------
    >>> change_seq("PEPTIDE", 2)
    "LDLSVED"
    >>> change_seq("PEPTIDE", 3)
    "LDLSVED"
    """
        
    #Identifies modifications within the sequence 
    split_seq = re.findall("([A-Z](?:\(.*?\))?)",seq)
    mods = [re.findall("\((.*?)\)",i) for i in split_seq]
    
    ## assume AA is the first. P(+79.97) or K(mTRAQ-0) for example 
    unmod_seq = [i[0] for i in split_seq]
    
    unmod_mass = mass.fast_mass(unmod_seq,charge=z)
    
    # if config.tag:
    #	tag_masses = sum([config.tag.mass_dict[j] for i in mods for j in i if j in config.tag.mass_dict])
    #else: 
    # 	tag_masses = 0
    
    # Compute total mass of the modifications in the provided mass_dict
    tag_masses = sum(
        mass_dict.get(j, 0.0) for sublist in mods for j in sublist
    )

    return unmod_mass+(tag_masses)/z

## update to work for mTRAQ
## Note: unsure if this works for modifications
####  TO DO:   Need to add tag masses to larger dict with modifications included 
#@profile
def convert_frags(seq: str,frags: dict[str, list[float]],rules: str = diann_rules) -> dict[str, list[float]]:
    """Gets the precursors m/z for a peptide sequence given the
    amino acid sequence and the charge state. Accepts sequences with modifications
    in '(' brackets  

    Parameters
    ----------
    seq : str
        The original peptide sequence 
    frags : dict[str, list[float]]
        Maps the fragment names (e.g., 'b3_2') to a list containing the m/z ratio and intensity.
        Fragment names have format (b/y)(frag index)_charge, e.g., 'b3_2' for b ion 3 with charge 2.
    rules : str
        Method to modify string (must either be 'rev' or 'diann')


    Returns
    -------
    float
        The unmodified m/z ratio of the peptide 
    
    Example
    -------
    >>> seq = 'AAAEQAISVR'
    >>> frags = {
                    'b3_1': [214.1186178209, 0.48869178],
                    'b4_1': [343.16121090887, 0.29596102],
                    'b5_1': [471.21978841415, 0.13230422],
                    'b6_1': [542.25690219886, 0.1957216],
                    'y3_1': [361.21939449133, 0.51684165],
                    'y4_1': [474.30345846846, 0.22480424],
                    'y5_1': [545.34057225317, 0.35331511],
                    'y6_1': [673.39914975845, 0.5639957],
                    'y7_1': [802.44174284642, 1.0],
                    'y8_1': [873.47885663113, 0.8503218],
                    'y9_1': [944.51597041584, 0.49269193]
                }
    >>> convert_frags(seq, frags, "rev")
    {
    'b3_1': [300.19178276116, 0.48869178],
    'b4_1': [371.22889654587004, 0.29596102],
    'b5_1': [499.28747405115, 0.13230422],
    'b6_1': [628.3300671391199, 0.1957216],
    'y3_1': [317.19317974349, 0.51684165],
    'y4_1': [388.2302935282, 0.22480424],
    'y5_1': [517.2728866161699, 0.35331511],
    'y6_1': [645.3314641214499, 0.5639957],
    'y7_1': [716.3685779061599, 1.0],
    'y8_1': [829.4526418832899, 0.8503218],
    'y9_1': [916.4846702875599, 0.49269193]
    }
    """

    new_seq = change_seq(seq=seq,rules=rules)    
    
    split_seq = parse_peptide(new_seq)
    
    if config.tag:   
       #tags = [re.findall(f"(\({config.tag.name}.*?\))",i) for i in seq]
       #seq = [re.sub(f"(\({config.tag.name}.*?\))","",i) for i in seq]
       tags = [[t.strip("()") for t in re.findall(f"(\({config.tag.name}.*?\))",i)] for i in split_seq]
       split_seq = [re.sub(f"(\({config.tag.name}.*?\))","",i) for i in split_seq]

    else:
        tags = [[] for i in seq]
        
    close_d = {"[":"]","(":")"}
    mods = [[m.strip(m[0]+close_d[m[0]]) for m in extract_mod(i)] for i in split_seq]
    mod_masses = [sum([config.diann_mods[j]  for j in i if j in config.diann_mods]) for i in mods]
    ## assume AA is the first 
    unmod_seq = [i[0] for i in split_seq]
    
    if config.tag:
    	tag_masses = [sum([config.tag.mass_dict[j]  for j in i if j in config.tag.mass_dict]) for i in tags]
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
            ion_loss = ion_nmr_loss_split[1]
            try:
                loss = mass.calculate_mass(ion_loss)
            except:
                loss=0
        if ion_type=="b":
            mz = mass.fast_mass(unmod_seq[:ion_nmr],ion_type,int(charge)) - (loss/float(charge)) + ((sum(tag_masses[:ion_nmr]) +sum(mod_masses[:ion_nmr]))/float(charge))
        
        if ion_type=="y":
            mz = mass.fast_mass(unmod_seq[-ion_nmr:],ion_type,int(charge)) - (loss/float(charge)) + ((sum(tag_masses[-ion_nmr:]) +sum(mod_masses[-ion_nmr:]))/float(charge))  

        new_frags[frag] = [mz, frags[frag][1]]

    #import pprint
    #print("new_frags:")
    #pprint.pprint(new_frags)
    #print("Type:", type(new_frags))
    #print("Structure Example:", list(new_frags.items())[:1])  # show one entry if possible
    #print()
    # Force error after debugging output
    #raise RuntimeError("Intentional debug stop after printing input information.")

    return new_frags

def hyperscore_b_y(frag_list: dict[str, list[float]],matches: npt.NDArray[np.bool_]) -> tuple[float, int, int]:
    """
    Description:

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


def longest_y(frag_list, matches):
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
    
    # Extract fragment numbers from matching fragments
    fragment_numbers = [extract_fragment_number(frag_name) for frag_name, match_value in zip(frag_list, matches) if match_value]
    
    # Handle the case where there are no matches
    if not fragment_numbers:
        return 0
        
    return np.max(fragment_numbers)

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





# def frag_to_peak(frag_dict):
#     peaks = np.array(list(frag_dict.values()))
#     order = np.argsort(peaks[:,0])
#     return peaks[order]

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




def convert_mq_frag(frg):
    c = re.findall("\((\d)\+\)",frg)
    if len(c)>0:
        charge= c[0]
        frg = re.sub(f"\({c[0]}\+\)","",frg)
    else:
        charge = "1"
    # charge = "1" if len(c)==0 else c[0]
        
    return frg+"_"+charge


def SpectralAngle(x,y):
    x /= np.sqrt(np.sum(np.power(x,2)))
    y /= np.sqrt(np.sum(np.power(y,2)))
    
    return 1- 2*(np.arccos(np.dot(x,y))/np.pi)


def cosim(x,y):
    assert len(x)==len(y)
    x = np.squeeze(x)
    y = np.squeeze(y)
    
    return np.dot(x,y)/(np.sqrt(np.sum(np.power(x,2)))*np.sqrt(np.sum(np.power(y,2))))



### source: https://stackoverflow.com/questions/62817623/improve-speed-of-scipy-pearson-correlation-for-many-pairwise-calculations
from scipy.special import betainc

# import cupy as xp
# from cupyx.scipy.special import betainc

def pearsonr2(x, y):
    # Assumes inputs are DataFrames and computation is to be performed
    # pairwise between columns. We convert to arrays and reshape so calculation
    # is performed according to normal broadcasting rules along the last axis.
    x = np.asarray(x).T[:, np.newaxis, :]
    y = np.asarray(y).T
    n = x.shape[-1]

    # Compute Pearson correlation coefficient. We can't use `cov` or `corrcoef`
    # because they want to compute everything pairwise between rows of a
    # stacked x and y.
    xm = x.mean(axis=-1, keepdims=True)
    ym = y.mean(axis=-1, keepdims=True)
    cov = np.sum((x - xm) * (y - ym), axis=-1)/(n-1)
    sx = np.std(x, ddof=1, axis=-1)
    sy = np.std(y, ddof=1, axis=-1)
    rho = cov/(sx * sy)

    # Compute the two-sided p-values. See documentation of scipy.stats.pearsonr.
    ab = n/2 - 1
    x = (abs(rho) + 1)/2
    p = 2*(1-betainc(ab, ab, x))
    return rho, p

class p_result:
    def __init__(self,r_sq,p=0):
        self.statistic = r_sq
        self.pvalue=p
    def __repr__(self):
        return f"PearsonRResult(statistic={self.statistic}, pvalue={self.pvalue})"
## source https://cancerdatascience.org/blog/posts/pearson-correlation/
def np_pearson_cor(x, y):
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

def lowess_fit(x,y,frac=.2, it=3):

    # plt.scatter(x,y,s=1)
    
    lowess = sm.nonparametric.lowess(y, x, frac=frac,it=it)
    
    # unpack the lowess smoothed points to their values
    lowess_x = list(zip(*lowess))[0]
    lowess_y = list(zip(*lowess))[1]
    
    # run scipy's interpolation. There is also extrapolation I believe
    f = interp1d(lowess_x, lowess_y, bounds_error=False,fill_value=(min(lowess_y),max(lowess_y)))
    
    return f


def fragment_cor(df,didx,fn="cos"):
    
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
    
    
