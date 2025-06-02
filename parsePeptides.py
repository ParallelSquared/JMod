"""
This Source Code Form is subject to the terms of the Oxford Nanopore
Technologies, Ltd. Public License, v. 1.0.  Full licence can be found
at https://github.com/ParallelSquared/JMod/blob/main/LICENSE.txt
"""
#######
#Parse peptide strings, fragments and modifications from spectral libraries 
import re
from pyteomics import mass
import config

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



