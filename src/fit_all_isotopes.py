import pickle
import numpy as np
from scipy import optimize, stats

import sys
import os
import time
import re
from fast_merge import merge_intensities as merge_intensities_rust
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from memory_profiler import profile


pkl_path = r"C:\Users\zcohe\Jmod\JMod_Profiling\Output\Line_Profiler_fit_mTRAQ\fit_results.pkl"
run_from = "python"


def merge_intensities(dia_spectrum, mz_ppm):
    merged_coords_idxs = np.searchsorted(dia_spectrum[:,0]+mz_ppm*dia_spectrum[:,0],dia_spectrum[:,0])
    
    # what are the first mz of these peak groups
    merged_coords = dia_spectrum[np.unique(merged_coords_idxs),0]
    # merged_intensities = np.zeros(len((merged_coords_idxs)))
    # for j,val in zip(merged_coords_idxs,dia_spectrum[:,1]):
    #     merged_intensities[j]+=val
    # merged_intensities = merged_intensities[merged_intensities!=0]
    merged_intensities = np.bincount(merged_coords_idxs, weights=dia_spectrum[:,1])
    merged_intensities = merged_intensities[merged_intensities != 0]
    
    #update spectrum to new values (note mz remains first in group as this will eventually be rounded)
    return np.array((merged_coords,merged_intensities)).transpose()




@profile
def main():
    print("running main with ", run_from)
    with open(pkl_path, "rb") as f:
        pkl_data = pickle.load(f)
    all_outputs = pkl_data["all_outputs"]
    all_scans_to_search = pkl_data["all_scans_to_search"]
    all_ms1_spectra = pkl_data["all_ms1_spectra"]
    all_ms1_spec_idxs = pkl_data["all_ms1_spec_idxs"]
    all_group_isos = pkl_data["all_group_isos"]
    all_mz_ppm = pkl_data["all_mz_ppm"]
    new_function_outputs = []
    print("finished reading pickle")


    
    t1 = time.time()

    max_len = max(len(s.mz) for group in all_ms1_spectra for s in group)
    dia_buffer = np.zeros((max_len, 2), dtype=np.float64)

    for i, ms1_spectra in enumerate(all_ms1_spectra):
        for ms1_spec_idx in all_scans_to_search[i]:
            spec = ms1_spectra[np.where(all_ms1_spec_idxs[i]==ms1_spec_idx)[0][0]]
            pred_coeff, obs_peaks, fit_matrix = fit_mTRAQ_isotopes_fast(spec,all_group_isos[i],all_mz_ppm[i], dia_buffer)
            new_function_outputs.append((pred_coeff, obs_peaks, fit_matrix))
    t2 = time.time()

    print(f"Finished fitting mTRAQ isotopes in {t2-t1} seconds")
    
    
    compare_output_to_gold(new_function_outputs, all_outputs)

def compare_output_to_gold(new_function_outputs, all_outputs):
    def all_equal(a_list, b_list):
        return all(np.array_equal(a, b) for a, b in zip(a_list, b_list))
    tol = 1e-9
    def arrays_all_close(list1, list2):
        return all(np.allclose(a, b, rtol=tol, atol=tol) for a, b in zip(list1, list2))
    
    new_pred_coeff = [x[0] for x in new_function_outputs]
    new_obs_peaks = [x[1] for x in new_function_outputs]
    new_fit_matrix = [x[2] for x in new_function_outputs]
    old_pred_coeff = [x[0] for x in all_outputs]
    old_obs_peaks = [x[1] for x in all_outputs]
    old_fit_matrix = [x[2] for x in all_outputs]

    if all_equal(new_pred_coeff, old_pred_coeff) and all_equal(new_obs_peaks, old_obs_peaks) and all_equal(new_fit_matrix, old_fit_matrix):
        print("Exactly the same")
        return
    
    if arrays_all_close(new_pred_coeff, old_pred_coeff) and arrays_all_close(new_obs_peaks, old_obs_peaks) and arrays_all_close(new_fit_matrix, old_fit_matrix):
        print("Arrays All Close")
        return
    
    if len(new_function_outputs) == len(all_outputs):
        print(f"Same Length: {len(new_function_outputs)}")
    else:
        print(f"Different Lengths: new: {len(new_function_outputs)} old: {len(all_outputs)}")


    if np.all(new_pred_coeff == old_pred_coeff):
        print("New_pred_coeff == old_pred_coeff")
    if np.all(new_obs_peaks == old_obs_peaks):
        print("new_obs_peaks == old_obs_peaks")
    if np.all(new_fit_matrix == old_fit_matrix):
        print("new_fit_matrix == old_fit_matrix")


class DummySpec:
    def __init__(self, mz, intens):
        self.mz=mz
        self.intens=intens

    def peak_list(self):
        return(np.array([self.mz,self.intens]))

    
class DummyPeak:
    def __init__(self, mz, intensity):
        self.mz = float(mz)
        self.intensity = float(intensity)


def test_fit_mTRAQ_isotopes():
    #Basic Test
    dia_spec = DummySpec(
        mz=np.array([50.0, 60.0, 100.0, 200.0]),
        intens=np.array([0.12, 0.18, 0.6, 0.1])
    )
    all_iso = [
        [
        DummyPeak(mz=50.0, intensity=0.12),
        DummyPeak(mz=60.0, intensity=0.18),
        DummyPeak(mz=100.0, intensity=0.6),
        DummyPeak(mz=200.0, intensity=0.1)
        ]
    ]
    mz_ppm = 1e-6

    max_len = len(dia_spec.mz)
    dia_buffer = np.zeros((max_len, 2), dtype=np.float64)

    pred_coeff, obs_peaks, fit_matrix = fit_mTRAQ_isotopes_fast(dia_spec, all_iso, mz_ppm, dia_buffer)
    pred_coeff_expected = np.array([1])
    obs_peaks_expected = np.array([0.12, 0.18, 0.6, 0.1, 0])
    fit_matrix_expected = np.array([
        [0.12],
        [0.18],
        [0.6],
        [0.1],
        [0.0]
    ])

    np.testing.assert_array_almost_equal(pred_coeff, pred_coeff_expected, decimal=6)
    np.testing.assert_array_almost_equal(obs_peaks, obs_peaks_expected, decimal=6)
    np.testing.assert_array_almost_equal(fit_matrix, fit_matrix_expected, decimal=6)


    #1 peptide 5 plex 4 Da spacing no noise

    import math
    num_atoms = 60
    p_heavy = 0.01
    z = 2
    channel_multiplier = [1, 2, 1.5, 3, 0.5]
    mz_intensity_dict = {}
    per_channel_iso_intensity_dict = {}
    for channel in range(0, 5):
        per_channel_iso_intensity_dict[channel] = {}
        num_atoms_undoped = 60 - (4*channel)
        for k in range(0, 5):
            rel_intens = math.comb(num_atoms_undoped, k) * (p_heavy**k) * ((1 - p_heavy)**(num_atoms_undoped - k))
            intens = rel_intens * channel_multiplier[channel]
            mz = (num_atoms + (4*channel) + k + z)/z
            per_channel_iso_intensity_dict[channel][mz] = rel_intens
            if mz not in mz_intensity_dict:
                mz_intensity_dict[mz] = 0
            mz_intensity_dict[mz] += intens
    
    sum_intensity = sum(mz_intensity_dict.values())
    normalized_mz_intensity_dict = {k: v/sum_intensity for k, v in mz_intensity_dict.items()}
            


    dia_spec = DummySpec(
        mz=np.array(list(normalized_mz_intensity_dict.keys())),
        intens=np.array(list(normalized_mz_intensity_dict.values()))
    )

    all_iso = [[DummyPeak(mz=mz, intensity=i) for mz, i in per_channel_iso_intensity_dict[channel].items()] for channel in per_channel_iso_intensity_dict.keys()]
    mz_ppm = 1e-6

    max_len = len(dia_spec.mz)
    dia_buffer = np.zeros((max_len, 2), dtype=np.float64)

    pred_coeff, obs_peaks, fit_matrix = fit_mTRAQ_isotopes_fast(dia_spec, all_iso, mz_ppm, dia_buffer)

    pred_coeff_expected = np.array([0.12502356, 0.25004712, 0.18753534, 0.37507069, 0.06251178])
    obs_peaks_expected = [
    6.84074722e-02, 4.14590741e-02, 1.23539665e-02, 2.41255912e-03,
    1.42774404e-01, 8.05648481e-02, 2.23791245e-02, 4.06893172e-03,
    1.11746745e-01, 5.84092183e-02, 1.50447987e-02, 2.53279439e-03,
    2.31840837e-01, 1.12255727e-01, 2.66465614e-02, 4.12707684e-03,
    4.06397816e-02, 1.78536870e-02, 3.87731585e-03, 5.48307292e-04,
    5.67691894e-05, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
    0.00000000e+00, 0.00000000e+00
    ]
    fit_matrix_expected = np.array([
    [0.54715664, 0., 0., 0., 0.],
    [0.33161009, 0., 0., 0., 0.],
    [0.09881311, 0., 0., 0., 0.],
    [0.01929684, 0., 0., 0., 0.],
    [0.00277757, 0.5696012, 0., 0., 0.],
    [0., 0.32219866, 0., 0., 0.],
    [0., 0.08949963, 0., 0., 0.],
    [0., 0.01627266, 0., 0., 0.],
    [0., 0.00217791, 0.59296645, 0., 0.],
    [0., 0., 0.31145712, 0., 0.],
    [0., 0., 0.0802238, 0., 0.],
    [0., 0., 0.01350569, 0., 0.],
    [0., 0., 0.00167116, 0.61729014, 0.],
    [0., 0., 0., 0.29929219, 0.],
    [0., 0., 0., 0.07104411, 0.],
    [0., 0., 0., 0.01100346, 0.],
    [0., 0., 0., 0.00125039, 0.6426116],
    [0., 0., 0., 0., 0.28560516],
    [0., 0., 0., 0., 0.06202536],
    [0., 0., 0., 0., 0.00877126],
    [0., 0., 0., 0., 0.00090814],
    [0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0.]
    ])


    np.testing.assert_array_almost_equal(pred_coeff, pred_coeff_expected, decimal=6)
    np.testing.assert_array_almost_equal(obs_peaks, obs_peaks_expected, decimal=6)
    np.testing.assert_array_almost_equal(fit_matrix, fit_matrix_expected, decimal=6)

    print("fit_mTRAQ_Isotopes Tests Passing")

def fit_mTRAQ_isotopes_fast(spec,all_iso,mz_ppm, dia_buffer):
    """
    
    ### spec is an ms1 spectrum
    #### all_iso is a list of the mTRAQ isotopes 
    e.g.
    [[Peak(mz=661.011960, intensity=0.352935, charge=3),
      Peak(mz=661.346233, intensity=0.335236, charge=3),
      Peak(mz=661.680188, intensity=0.192931, charge=3)]
     ...]
    
    mz_ppm is the relative mz tolerance e.g. 5.6e-6
    
    """
    ### spec is an ms1 spectrum
    #### all_iso is a list of the mTRAQ isotopes 
    
    
    #ms1_iso_patterns2 = np.array([[[i.mz,i.intensity] for i in isotope] for isotope in all_iso])

    flat = np.array([(p.mz, p.intensity) for iso in all_iso for p in iso], dtype=np.float64)
    ms1_iso_patterns = flat.reshape(len(all_iso), len(all_iso[0]), 2)

    
    #dia_spectrum2 = np.stack(spec.peak_list(),1)

    #dia_spectrum3 = np.array(spec.peak_list(), dtype=np.float64).T

    mz = spec.mz
    intens = spec.intens
    n = mz.shape[0]

    dia_buffer[:n, 0] = mz
    dia_buffer[:n, 1] = intens
    dia_spectrum = dia_buffer[:n]




    
    
    ### we only need to conseider the part of the spectrum that falls within the isotopic envelopes of the channels
    # min_isotope2 = min([j.mz for i in all_iso for j in i])-1
    # max_isotope2 = max([j.mz for i in all_iso for j in i])+1

    min_isotope = ms1_iso_patterns[:, :, 0].min() - 1
    max_isotope = ms1_iso_patterns[:, :, 0].max() + 1


    # dia_spectrum2 = dia_spectrum[np.logical_and(dia_spectrum[:,0]>min_isotope,dia_spectrum[:,0]<max_isotope)]

    mz = dia_spectrum[:, 0]
    lo = np.searchsorted(mz, min_isotope, side="right")
    hi = np.searchsorted(mz, max_isotope, side="left")
    dia_spectrum = dia_spectrum[lo:hi]

    
    dia_spectrum = merge_intensities(dia_spectrum, mz_ppm)
    # print(dia_spectrum)
    
    #get window edge positions each side of peaks in observed spectra (NB the tolerance is now about the first peak in the group not the middile)
    # centroid_breaks2 = np.concatenate((dia_spectrum[:,0]-mz_ppm*dia_spectrum[:,0],dia_spectrum[:,0]+mz_ppm*dia_spectrum[:,0]))
    # centroid_breaks2 = np.sort(centroid_breaks2)

    mz = dia_spectrum[:, 0]
    offsets = mz_ppm * mz
    centroid_breaks = np.empty(mz.size * 2, dtype=mz.dtype)
    centroid_breaks[:mz.size] = mz - offsets
    centroid_breaks[mz.size:] = mz + offsets
    centroid_breaks.sort()

 

    #bin_centers = np.mean(np.stack((centroid_breaks[::2],centroid_breaks[1::2]),1),1)
    
    #ref_coords2 = [np.searchsorted(centroid_breaks,M[:,0]) for M in ms1_iso_patterns]

    all_mz = ms1_iso_patterns[:,:,0].ravel()   # flatten all m/z values
    ref_coords_flat = np.searchsorted(centroid_breaks, all_mz)
    ref_coords = ref_coords_flat.reshape(ms1_iso_patterns.shape[0], -1)

    
    
    # lib_peaks_matched2 = [j%2==1 for j in ref_coords]

    lib_peaks_matched = (ref_coords % 2 == 1).tolist()




    #ref_spec_row_indices_split2 = [np.int32(((i[j]+1)/2)-1) for i,j in zip(ref_coords,lib_peaks_matched)] # NB these are floats

    ref_spec_row_indices_split = [(((rc + 1) // 2) - 1).astype(np.int32)[mask]for rc, mask in zip(ref_coords, lib_peaks_matched)]



    #num_lib_peaks_matched2 = np.array([np.sum(i) for i in lib_peaks_matched]) #f1
    num_lib_peaks_matched = np.fromiter((sum(i) for i in lib_peaks_matched), dtype=np.int32)


    ref_spec_col_indices_split = [np.array([idx]*i,dtype=np.int32) for idx,i in zip(range(len(ref_coords)),num_lib_peaks_matched)] 


    #ref_spec_values_split2 = [i[:,1][j] for i,j in zip(ms1_iso_patterns,lib_peaks_matched)]
    ref_spec_values_split = [ms1_iso_patterns[idx, :, 1][mask] for idx, mask in enumerate(lib_peaks_matched)]


    
    
    lib_coefficients = np.zeros(len(ref_coords))
    dia_spec_int = []
    matrix = []
    if any([i.size>0 for i in ref_spec_row_indices_split]):
        
        ref_spec_row_indices = np.concatenate(ref_spec_row_indices_split)
        ref_spec_col_indices = np.concatenate(ref_spec_col_indices_split)
        ref_spec_values = np.concatenate(ref_spec_values_split)
        # what peaks from the spectrum are matched by library peps
        unique_row_idxs = [int(i) for i in set(ref_spec_row_indices)]
        unique_row_idxs.sort()
        
        dia_spec_int = dia_spectrum[unique_row_idxs,1]
        
        lower_limit=1e-10
        last_row = max(unique_row_idxs)
        
        #### Type B
        not_dia_col_indices = np.arange(len(ref_coords))
        not_dia_row_indices = [last_row+1]*len(not_dia_col_indices)+not_dia_col_indices
        # not_dia_values2 = np.array([np.sum([ms1_iso_patterns[:,:,1][idx][peak_idx] for peak_idx in range(len(ms1_iso_patterns[:,:,1][idx])) if ref_coords[idx][peak_idx]%2==0])
        #                           for idx in range(len(ref_coords))])
                                  
        
        ref_coords_arr = np.array(ref_coords)
        ms1_intensities = ms1_iso_patterns[:, :, 1]  # shape: (num_peptides, num_peaks)

        mask = (ref_coords_arr % 2 == 0)
        not_dia_values = (ms1_intensities * mask).sum(axis=1)
         
        
        
        # sparse_row_indices2 = np.append(ref_spec_row_indices,not_dia_row_indices)
        # sparse_col_indices2 = np.append(ref_spec_col_indices,not_dia_col_indices)
        # sparse_values2 = np.append(ref_spec_values,not_dia_values)

        sparse_row_indices = np.concatenate([ref_spec_row_indices, not_dia_row_indices])
        sparse_col_indices = np.concatenate([ref_spec_col_indices, not_dia_col_indices])
        sparse_values = np.concatenate([ref_spec_values, not_dia_values])

        
        # some dia peaks are not matched and are therefore ignored
        # below ranks the rows by number therefore removing missing rows

        #sparse_row_indices2 = stats.rankdata(sparse_row_indices,method="dense").astype(int)-1

        unique_vals, new_indices = np.unique(sparse_row_indices, return_inverse=True)
        sparse_row_indices = new_indices.astype(np.int32)
        
        max_row = np.max(sparse_row_indices)+1 # plus 1 for indexing
        max_col = np.max(sparse_col_indices)+1
        matrix = np.zeros((max_row,max_col))
        matrix[sparse_row_indices,sparse_col_indices] = sparse_values
        
        dia_spec_int = np.append(dia_spec_int,[0]*(matrix.shape[0]-dia_spec_int.shape[0])) 
        
        # Generate sparse matrix from data
        # sparse_lib_matrix = sparse.coo_matrix((sparse_values,(sparse_row_indices,sparse_col_indices)))
        # dia_spec_int = np.append(dia_spec_int,[0]*(sparse_lib_matrix.shape[0]-dia_spec_int.shape[0])) 
        
        # Fit lib spectra to observed spectra
        # fit_results = sparse_nnls.lsqnonneg(sparse_lib_matrix,dia_spec_int,{"show_progress":False})
        # lib_coefficients = np.array(fit_results['x']).flatten()
        
        ### NOT Non-Negative!!
        # matrix = np.array(sparse_lib_matrix.todense())
        # lib_coefficients = np.linalg.lstsq(matrix, dia_spec_int)[0]
        lib_coefficients, residuals = optimize.nnls(matrix, dia_spec_int)
        
    return lib_coefficients, dia_spec_int,  matrix

def fit_mTRAQ_isotopes_slow(spec,all_iso,mz_ppm):
    """
    
    ### spec is an ms1 spectrum
    #### all_iso is a list of the mTRAQ isotopes 
    e.g.
    [[Peak(mz=661.011960, intensity=0.352935, charge=3),
      Peak(mz=661.346233, intensity=0.335236, charge=3),
      Peak(mz=661.680188, intensity=0.192931, charge=3)]
     ...]
    
    mz_ppm is the relative mz tolerance e.g. 5.6e-6
    
    """
    ### spec is an ms1 spectrum
    #### all_iso is a list of the mTRAQ isotopes 
    
    
    ms1_iso_patterns = np.array([[[i.mz,i.intensity] for i in isotope] for isotope in all_iso])
    
    dia_spectrum = np.stack(spec.peak_list(),1)

    
    
    ### we only need to conseider the part of the spectrum that falls within the isotopic envelopes of the channels
    min_isotope = min([j.mz for i in all_iso for j in i])-1
    max_isotope = max([j.mz for i in all_iso for j in i])+1

    dia_spectrum = dia_spectrum[np.logical_and(dia_spectrum[:,0]>min_isotope,dia_spectrum[:,0]<max_isotope)]
    
    dia_spectrum = merge_intensities(dia_spectrum, mz_ppm)
    
    # print(dia_spectrum)
    
    #get window edge positions each side of peaks in observed spectra (NB the tolerance is now about the first peak in the group not the middile)
    centroid_breaks = np.concatenate((dia_spectrum[:,0]-mz_ppm*dia_spectrum[:,0],dia_spectrum[:,0]+mz_ppm*dia_spectrum[:,0]))
    centroid_breaks = np.sort(centroid_breaks)
    bin_centers = np.mean(np.stack((centroid_breaks[::2],centroid_breaks[1::2]),1),1)
    
    ref_coords = [np.searchsorted(centroid_breaks,M[:,0]) for M in ms1_iso_patterns]
    
    lib_peaks_matched = [j%2==1 for j in ref_coords]
    ref_spec_row_indices_split = [np.int32(((i[j]+1)/2)-1) for i,j in zip(ref_coords,lib_peaks_matched)] # NB these are floats
    num_lib_peaks_matched = np.array([np.sum(i) for i in lib_peaks_matched]) #f1

    ref_spec_col_indices_split = [np.array([idx]*i,dtype=np.int32) for idx,i in zip(range(len(ref_coords)),num_lib_peaks_matched)] 
    ref_spec_values_split = [i[:,1][j] for i,j in zip(ms1_iso_patterns,lib_peaks_matched)]
    
    
    lib_coefficients = np.zeros(len(ref_coords))
    dia_spec_int = []
    matrix = []
    if any([i.size>0 for i in ref_spec_row_indices_split]):
        
        ref_spec_row_indices = np.concatenate(ref_spec_row_indices_split)
        ref_spec_col_indices = np.concatenate(ref_spec_col_indices_split)
        ref_spec_values = np.concatenate(ref_spec_values_split)
        # what peaks from the spectrum are matched by library peps
        unique_row_idxs = [int(i) for i in set(ref_spec_row_indices)]
        unique_row_idxs.sort()
        
        dia_spec_int = dia_spectrum[unique_row_idxs,1]
        
        lower_limit=1e-10
        last_row = max(unique_row_idxs)
        
        #### Type B
        not_dia_col_indices = np.arange(len(ref_coords))
        not_dia_row_indices = [last_row+1]*len(not_dia_col_indices)+not_dia_col_indices
        not_dia_values = np.array([np.sum([ms1_iso_patterns[:,:,1][idx][peak_idx] for peak_idx in range(len(ms1_iso_patterns[:,:,1][idx])) if ref_coords[idx][peak_idx]%2==0])
                                  for idx in range(len(ref_coords))])
        
        
        
        sparse_row_indices = np.append(ref_spec_row_indices,not_dia_row_indices)
        sparse_col_indices = np.append(ref_spec_col_indices,not_dia_col_indices)
        sparse_values = np.append(ref_spec_values,not_dia_values)
        
        # some dia peaks are not matched and are therefore ignored
        # below ranks the rows by number therefore removing missing rows
        sparse_row_indices = stats.rankdata(sparse_row_indices,method="dense").astype(int)-1
        
        max_row = np.max(sparse_row_indices)+1 # plus 1 for indexing
        max_col = np.max(sparse_col_indices)+1
        matrix = np.zeros((max_row,max_col))
        matrix[sparse_row_indices,sparse_col_indices] = sparse_values
        
        dia_spec_int = np.append(dia_spec_int,[0]*(matrix.shape[0]-dia_spec_int.shape[0])) 
        
        # Generate sparse matrix from data
        # sparse_lib_matrix = sparse.coo_matrix((sparse_values,(sparse_row_indices,sparse_col_indices)))
        # dia_spec_int = np.append(dia_spec_int,[0]*(sparse_lib_matrix.shape[0]-dia_spec_int.shape[0])) 
        
        # Fit lib spectra to observed spectra
        # fit_results = sparse_nnls.lsqnonneg(sparse_lib_matrix,dia_spec_int,{"show_progress":False})
        # lib_coefficients = np.array(fit_results['x']).flatten()
        
        ### NOT Non-Negative!!
        # matrix = np.array(sparse_lib_matrix.todense())
        # lib_coefficients = np.linalg.lstsq(matrix, dia_spec_int)[0]
        lib_coefficients, residuals = optimize.nnls(matrix, dia_spec_int)
        
    return lib_coefficients, dia_spec_int,  matrix
    


        
if __name__ == "__main__":
    main()
    #test_fit_mTRAQ_isotopes()
