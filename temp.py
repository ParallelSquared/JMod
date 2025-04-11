def fit_to_lib2(dia_spec,library,rt_mz,all_keys,dino_features=None,rt_filter=False,ms1_mz=None,
               ms1_spectra = None,
               rt_tol = config.rt_tol,
               ms1_tol = config.ms1_tol,
               mz_tol = config.mz_tol,
               return_frags = False,
               decoy=False,
               decoy_library=None):
    # spec_idx,dia_spec,library = inputs
    
    spec_idx=dia_spec.scan_num
    
    # mz_tol = config.mz_tol
    # rt_tol = min(config.rt_tol,config.opt_rt_tol)
    # ms1_tol = min(config.ms1_tol,config.opt_ms1_tol)
    top_n=config.top_n
    atleast_m=config.atleast_m
    spec = dia_spec#spectra.ms2scans[spec_idx]
    dia_spectrum = np.stack(spec.peak_list(),1)
    prec_mz = spec.prec_mz
    prec_rt = spec.RT
    # spec_idx = spec.id
    
    windowWidth = window_width(dia_spec)
    
    
    if ms1_spectra is not None:
        ms1_spec = get_closest_ms1(prec_rt,ms1_spectra)
    
    
    lib_coefficients = []
   
    if ms1_mz:
        _bool = (np.abs(rt_mz[:,1]-ms1_mz)/ms1_mz)<ms1_tol
        
    else:
        if rt_filter:
            _bool = np.logical_and(np.abs(rt_mz[:,1]-prec_mz)<(windowWidth/2),np.abs(rt_mz[:,0]-prec_rt)<rt_tol)
        else:
            _bool = np.abs(rt_mz[:,1]-prec_mz)<(windowWidth/2)
            
    window_idxs = np.where(_bool)[0]        
        
    
    mass_window_candidates = [all_keys[i] for i in window_idxs] 
    candidate_peaks = [library[i]['spectrum'] for i in mass_window_candidates]
    
    
    ###### Process dia spectrum 
    
    # what are the first indices of peaks grouped by tolerance
    merged_coords_idxs = np.searchsorted(dia_spectrum[:,0]+mz_tol*dia_spectrum[:,0],dia_spectrum[:,0])
    
    # what are the first mz of these peak groups
    merged_coords = dia_spectrum[np.unique(merged_coords_idxs),0]
    # print(merged_coords)
    
    
    # NB - should we not sum the intensities?????
    # merged_intensities = [np.mean(dia_spectrum[np.where(merged_coords_idxs==i)[0],1]) for i in np.unique(merged_coords_idxs)]
    merged_intensities = np.zeros(len((merged_coords_idxs)))
    for j,val in zip(merged_coords_idxs,dia_spectrum[:,1]):
        merged_intensities[j]+=val
    #merged_intensities = [np.mean(dia_spectrum[merged_coords_idxs==i,1]) for i in np.unique(merged_coords_idxs)]
    merged_intensities = merged_intensities[merged_intensities!=0]
    
    #update spectrum to new values (note mz remains first in group as this will eventually be rounded)
    dia_spectrum = np.array((merged_coords,merged_intensities)).transpose()
    # print(dia_spectrum)
    
    #get window edge positions each side of peaks in observed spectra (NB the tolerance is now about the first peak in the group not the middile)
    centroid_breaks = np.concatenate((dia_spectrum[:,0]-mz_tol*dia_spectrum[:,0],dia_spectrum[:,0]+mz_tol*dia_spectrum[:,0]))
    centroid_breaks = np.sort(centroid_breaks)
    bin_centers = np.mean(np.stack((centroid_breaks[::2],centroid_breaks[1::2]),1),1)
    
    # lib_idx=0
    # M = candidate_peaks[lib_idx]
    # ref_coords = [np.searchsorted(centroid_breaks,M[:,0]) for M in candidate_peaks]
    # top_ten = [np.searchsorted(centroid_breaks,M[np.argsort(-M[:,1])[0:min(top_n,M.shape[0])],0]) for M in candidate_peaks]
    
    ## Filter precursors based on resp. MS1 peak
    # ms1_peak = ~np.isnan([closest_peak_diff(mz,ms1_spec.mz) for mz in rt_mz[window_idxs,1]])
    
    # does the top ten peaks fall between centroid breaks? i.e. odd numbers (%2==1), 
    # ref_peaks_in_dia = [i for i in range(len(candidate_peaks)) if len([a for a in top_ten[i] if a%2 ==1])>atleast_m]
    # prop_ref_peaks_in_dia = [len([a for a in top_ten[i] if a%2 ==1])/candidate_peaks[i].shape[0] for i in range(len(candidate_peaks))]
    
    # all_norm_intensities = [M[:,1]/sum(M[:,1]) for M in candidate_peaks]
    # ref_peaks_in_dia = [i for i in range(len(candidate_peaks)) if np.sum(all_norm_intensities[i][(ref_coords[i]%2)==1])>0.5 and np.sum(top_ten[i]%2)>atleast_m and ms1_peak[i]]
    # ref_peaks_in_dia = [i for i in range(len(candidate_peaks)) if  np.sum(top_ten[i]%2)>atleast_m and ms1_peak[i]]
    
    # print(len(ref_peaks_in_dia))
    
    # filter database further to those that match the required num peaks
    # ref_pep_cand_loc = [ref_coords[i] for i in ref_peaks_in_dia]
    # ref_pep_cand_list = [candidate_peaks[i] for i in ref_peaks_in_dia]
    # # ref_pep_cand = [candidate_lib[i]["seq"] for i in ref_peaks_in_dia] # Nb this is modified seq!!
    # ref_pep_cand = [mass_window_candidates[i] for i in ref_peaks_in_dia] # Nb this is modified seq!!
    
    # norm_intensities = [M[:,1]/sum(M[:,1]) for M in ref_pep_cand_list]


    ########## Update
    # lib peaks that match
    # lib_peaks_matched = [j%2==1 for j in ref_pep_cand_loc]
    
    # # name these something different so can be accessed later
    # ref_spec_row_indices_split = [np.int32(((i[j]+1)/2)-1) for i,j in zip(ref_pep_cand_loc,lib_peaks_matched)] # NB these are floats
    # num_lib_peaks_matched = np.array([np.sum(i) for i in lib_peaks_matched]) #f1
    # ref_spec_col_indices_split = [np.array([idx]*i) for idx,i in zip(range(len(ref_pep_cand)),num_lib_peaks_matched)] 
    # ref_spec_values_split = [ints[i] for ints,i in zip(norm_intensities,lib_peaks_matched)]
    
    
    top_n_idxs = [library[i]['top_n'] for i in mass_window_candidates]
    
    
    spec_frags = None
    if "spec_frags" in library[all_keys[0]].keys():
        spec_frags = [library[i]['spec_frags'] for i in mass_window_candidates]
        
    ref_peaks_in_dia,\
    ref_pep_cand,\
    ref_pep_cand_loc,\
    ref_pep_cand_list,\
    ref_spec_row_indices_split,\
        ref_spec_col_indices_split,\
        ref_spec_values_split, \
        norm_intensities, \
        lib_peaks_matched, \
        ref_ms1_error = create_entries(centroid_breaks=centroid_breaks, 
                                        candidate_peaks=candidate_peaks, 
                                        mass_window_candidates=mass_window_candidates, 
                                        top_n=top_n, 
                                        atleast_m=atleast_m, 
                                        prec_mzs=rt_mz[:,1][window_idxs], 
                                        ms1_spec=ms1_spec,
                                        ms1_tol=ms1_tol,
                                        top_n_idxs=top_n_idxs)

    
    ### Generate eqivalent Decoy spectra
    if decoy:
        mass_window_decoy_candidates = [("Decoy_"+i[0],*i[1:]) for i in mass_window_candidates] 
        # print("old")
        # converted_seqs = [change_seq(i[0],config.args.decoy) for i in mass_window_candidates]
        # decoy_mz = np.array([convert_prec_mz(i, z=j[1]) for i,j in zip(converted_seqs, mass_window_candidates)])
        # if config.args.decoy=="rev": ## this will have the same mz as many correct mathces and therefore a really good ms1 isotope corr
        #     decoy_mz -= config.decoy_mz_offset
        # ## NB: Below needs to change to ibcorporate iso frags!!
        # converted_frags = [convert_frags(i[0], library[i]["frags"],config.args.decoy) for i in mass_window_candidates]
        # decoy_sorted_frags = [sorted(converted_frags[i],key = lambda x: converted_frags[i][x][0]) for i in range(len(converted_frags))]
        # if config.args.iso:
        #     candidate_decoy_peaks = [gen_isotopes(i,j) for i,j in zip(converted_seqs,converted_frags)]
        # else:
        #     candidate_decoy_peaks = [frag_to_peak(i) for i in converted_frags]
        
        # ## if using decoy_library
        # print("new")
        converted_frags = [decoy_library[i]["frags"] for i in mass_window_candidates]
        decoy_sorted_frags = [decoy_library[i]["ordered_frags"] for i in mass_window_candidates]
        candidate_decoy_peaks = [decoy_library[i]["spectrum"] for i in mass_window_candidates]
        # decoy_mz = np.array([decoy_library[i]["prec_mz"] for i in mass_window_candidates])
        decoy_mz = rt_mz[:,1][window_idxs] - config.decoy_mz_offset
    
        decoy_top_n_idxs = [decoy_library[i]['top_n'] for i in mass_window_candidates]
        
        decoy_spec_frags = None
        # if "spec_frags" in library[all_keys[0]].keys():
        #     decoy_spec_frags = [specific_frags(i) for i in converted_frags]
        
        # ## Decoy equiv
        # decoy_coords = [np.searchsorted(centroid_breaks,M[:,0]) for M in candidate_decoy_peaks]
        # top_ten_decoy = [np.searchsorted(centroid_breaks,M[np.argsort(-M[:,1])[0:min(top_n,M.shape[0])],0]) for M in candidate_decoy_peaks]
        # # decoy_peaks_in_dia = [i for i in range(len(candidate_decoy_peaks)) if len([a for a in top_ten_decoy[i] if a%2 ==1])>atleast_m]
        # all_norm_decoy_intensities = [M[:,1]/sum(M[:,1]) for M in candidate_decoy_peaks]
        # decoy_ms1_peak = ~np.isnan([closest_peak_diff(mz,ms1_spec.mz) for mz in decoy_mz])
        # # decoy_peaks_in_dia = [i for i in range(len(candidate_peaks)) if np.sum(all_norm_decoy_intensities[i][(decoy_coords[i]%2)==1])>0.5 and np.sum(top_ten_decoy[i]%2)>atleast_m and decoy_ms1_peak[i]]
        # decoy_peaks_in_dia = [i for i in range(len(candidate_peaks)) if np.sum(top_ten_decoy[i]%2)>atleast_m and decoy_ms1_peak[i]]
        
        # decoy_pep_cand_loc = [decoy_coords[i] for i in decoy_peaks_in_dia]
        # decoy_pep_cand_list = [candidate_decoy_peaks[i] for i in decoy_peaks_in_dia]
        # decoy_pep_cand = [mass_window_decoy_candidates[i] for i in decoy_peaks_in_dia] # Nb this is modified seq!!
        
        # norm_decoy_intensities = [M[:,1]/sum(M[:,1]) for M in decoy_pep_cand_list]
        
        # decoy_lib_peaks_matched = [j%2==1 for j in decoy_pep_cand_loc]
        
        # decoy_spec_row_indices_split = [np.int32(((i[j]+1)/2)-1) for i,j in zip(decoy_pep_cand_loc,decoy_lib_peaks_matched)] # NB these are floats
        # num_decoy_peaks_matched = np.array([np.sum(i) for i in decoy_lib_peaks_matched]) #f1
        # decoy_spec_col_indices_split = [np.array([idx]*i,dtype=int) for idx,i in zip(range(len(decoy_pep_cand)),num_decoy_peaks_matched)] 
        # decoy_spec_values_split = [ints[i] for ints,i in zip(norm_decoy_intensities,decoy_lib_peaks_matched)]
        
        
    
        decoy_peaks_in_dia,\
        decoy_pep_cand,\
        decoy_pep_cand_loc,\
        decoy_pep_cand_list,\
        decoy_spec_row_indices_split,\
            decoy_spec_col_indices_split,\
                decoy_spec_values_split, \
                    norm_decoy_intensities, \
                        decoy_lib_peaks_matched, \
                            decoy_ms1_error = create_entries(centroid_breaks=centroid_breaks, 
                                                                candidate_peaks=candidate_decoy_peaks, 
                                                                mass_window_candidates=mass_window_decoy_candidates, 
                                                                top_n=top_n, 
                                                                atleast_m=atleast_m, 
                                                                prec_mzs=decoy_mz, 
                                                                ms1_spec=ms1_spec,
                                                                ms1_tol=ms1_tol,
                                                                spec_frags=decoy_spec_frags,
                                                                top_n_idxs=decoy_top_n_idxs)
       
    frag_errors = []
    lib_frag_mz = []
    decoy_col_offset = 0
    
    if len(ref_spec_row_indices_split)>0 and len(ref_spec_col_indices_split)>0 and len(ref_spec_values_split)>0:
        
        #### concatenate the matrix values
        ref_spec_row_indices = np.concatenate(ref_spec_row_indices_split)
        ref_spec_col_indices = np.concatenate(ref_spec_col_indices_split)
        ref_spec_values = np.concatenate(ref_spec_values_split)
        
        frag_errors = [np.array(bin_centers[ref_spec_row_indices_split[i]]-ref_pep_cand_list[i][:,0][lib_peaks_matched[i]])/bin_centers[ref_spec_row_indices_split[i]] for i in range(len(lib_peaks_matched))]
        lib_frag_mz = [ref_pep_cand_list[i][:,0][lib_peaks_matched[i]] for i in range(len(lib_peaks_matched))]
        lib_frag_int = [ref_pep_cand_list[i][:,1][lib_peaks_matched[i]] for i in range(len(lib_peaks_matched))]
        obs_frag_int = [dia_spectrum[ref_spec_row_indices_split[i],1] for i in range(len(ref_spec_row_indices_split))]
        frag_names = [library[i]["ordered_frags"][j] for i,j in zip(ref_pep_cand,lib_peaks_matched)]
        
        decoy_col_offset = np.max(ref_spec_col_indices)+1 
        
    else:
        ref_spec_row_indices=np.array([],dtype=int)
        ref_spec_col_indices=np.array([],dtype=int)
        ref_spec_values=np.array([],dtype=int)
        frag_errors = []#np.array([],dtype=float)
        lib_frag_mz = []#np.array([],dtype=float)
        lib_frag_int = []
        obs_frag_int = []
        frag_names = []
        
        
    if decoy and len(decoy_spec_row_indices_split)>0:
        decoy_spec_row_indices = np.concatenate(decoy_spec_row_indices_split)
        decoy_spec_col_indices = np.concatenate(decoy_spec_col_indices_split)+decoy_col_offset
        decoy_spec_values = np.concatenate(decoy_spec_values_split)
        decoy_frag_errors = [np.array(bin_centers[decoy_spec_row_indices_split[i]]-decoy_pep_cand_list[i][:,0][decoy_lib_peaks_matched[i]])/bin_centers[decoy_spec_row_indices_split[i]] for i in range(len(decoy_lib_peaks_matched))]
        decoy_lib_frag_mz = [decoy_pep_cand_list[i][:,0][decoy_lib_peaks_matched[i]] for i in range(len(decoy_lib_peaks_matched))]
        decoy_lib_frag_int = [decoy_pep_cand_list[i][:,1][decoy_lib_peaks_matched[i]] for i in range(len(decoy_lib_peaks_matched))]
        decoy_obs_frag_int = [dia_spectrum[decoy_spec_row_indices_split[i],1] for i in range(len(decoy_spec_row_indices_split))]
        decoy_frag_names =  [decoy_sorted_frags[i][decoy_lib_peaks_matched[idx]] for idx,i in enumerate(decoy_peaks_in_dia)]
    else:
        decoy_spec_row_indices_split=[] ## needs to be improved
        decoy_spec_values_split=[] ## needs to be improved
        decoy_spec_row_indices=np.array([],dtype=int)
        decoy_spec_col_indices=np.array([],dtype=int)
        decoy_spec_values=np.array([],dtype=int)
        decoy_frag_errors = []#np.array([],dtype=float)
        decoy_lib_frag_mz = []#np.array([],dtype=float)
        decoy_lib_frag_int = []
        decoy_obs_frag_int = []
        decoy_frag_names = []
        
    if len(decoy_spec_row_indices_split)>0 or len(ref_spec_row_indices_split)>0:
        # what peaks from the spectrum are matched by library peps
        unique_row_idxs = np.unique(np.concatenate((ref_spec_row_indices,decoy_spec_row_indices)))
        unique_row_idxs = np.array(np.sort(unique_row_idxs),dtype=int)
        
        
        # # find peaks that are bot matched in dia spectrum
        # ref_peaks_not_in_dia = np.array([idx for loc_list in ref_pep_cand_loc for idx in range(len(loc_list)) if loc_list[idx]%2==0])
        # # get col indices (will just be one for each)
        # not_dia_col_indices = np.arange(len(ref_pep_cand))
        # num_rows = max(unique_row_idxs)
        # # row indices always the last row (num peaks+1)
        # not_dia_row_indices = [num_rows+1]*len(not_dia_col_indices)
        # # sum peak intensities not in dia spectrum
        # not_dia_values = np.array([np.sum([norm_intensities[idx][peak_idx] for peak_idx in range(len(norm_intensities[idx])) if ref_pep_cand_loc[idx][peak_idx]%2==0])
        #                           for idx in range(len(norm_intensities))])
       
        if len(ref_spec_row_indices_split)>0:
            not_dia_row_indices,not_dia_col_indices,not_dia_values = unmatched_peaks(norm_intensities=norm_intensities,
                                                                                     pep_cand_loc=ref_pep_cand_loc,
                                                                                     last_row=max(unique_row_idxs)+1,
                                                                                     fit_type=config.unmatched_fit_type)
        else:
            not_dia_row_indices=np.array([],dtype=np.int32)
            not_dia_col_indices=np.array([],dtype=np.int32)
            not_dia_values=np.array([],dtype=np.int32)
            
        if decoy and len(decoy_spec_row_indices_split)>0:
            decoy_not_dia_row_indices,decoy_not_dia_col_indices,decoy_not_dia_values = unmatched_peaks(norm_intensities=norm_decoy_intensities,
                                                                                                         pep_cand_loc=decoy_pep_cand_loc,
                                                                                                         last_row=max(not_dia_row_indices,default=max(unique_row_idxs)+1), # if all ref are mathched the initial list is empty
                                                                                                         fit_type=config.unmatched_fit_type)
        else:
            decoy_not_dia_row_indices=np.array([],dtype=np.int32)
            decoy_not_dia_col_indices=np.array([],dtype=np.int32)
            decoy_not_dia_values=np.array([],dtype=np.int32)
            
        ref_sparse_row_indices = np.append(ref_spec_row_indices,not_dia_row_indices)
        ref_sparse_col_indices = np.append(ref_spec_col_indices,not_dia_col_indices)
        ref_sparse_values = np.append(ref_spec_values,not_dia_values)
        
        decoy_sparse_row_indices = np.append(decoy_spec_row_indices,decoy_not_dia_row_indices)
        decoy_sparse_col_indices = np.append(decoy_spec_col_indices,decoy_not_dia_col_indices+decoy_col_offset)
        decoy_sparse_values = np.append(decoy_spec_values,decoy_not_dia_values)
        
        
        sparse_row_indices = np.concatenate((ref_sparse_row_indices,decoy_sparse_row_indices))
        sparse_col_indices = np.concatenate((ref_sparse_col_indices,decoy_sparse_col_indices))
        sparse_values = np.concatenate((ref_sparse_values,decoy_sparse_values))
        
        # some dia peaks are not matched and are therefore ignored
        # below ranks the rows by number therefore removing missing rows
        new_row_indices = stats.rankdata(sparse_row_indices,method="dense").astype(int)-1
        peak_idx_convertor = {i:j for i,j in zip(sparse_row_indices,new_row_indices)}
        sparse_row_indices =new_row_indices
        
        # Generate sparse matrix from data
        sparse_lib_matrix = sparse.coo_matrix((sparse_values,(sparse_row_indices,sparse_col_indices)))
        
        
        dia_spec_int = dia_spectrum[unique_row_idxs,1]
        
        # add another term to penalise additional lib peaks
        dia_spec_int = np.append(dia_spec_int,[0]*(sparse_lib_matrix.shape[0]-dia_spec_int.shape[0])) 
        
        # Fit lib spectra to observed spectra
        fit_results = sparse_nnls.lsqnonneg(sparse_lib_matrix,dia_spec_int,{"show_progress":False})
        lib_coefficients = fit_results['x']
        
        
        ####################################
        features = get_features(rt_mz[window_idxs[ref_peaks_in_dia]],
                                ref_spec_values_split,
                                ref_spec_row_indices_split,
                                ref_spec_col_indices_split,
                                decoy_spec_values_split,
                                decoy_spec_row_indices_split,
                                decoy_spec_col_indices_split,
                                ref_peaks_in_dia,
                                dia_spectrum,
                                prec_rt,
                                window_idxs,
                                dia_spec_int,
                                lib_coefficients,
                                sparse_lib_matrix,
                                sparse_row_indices,
                                sparse_col_indices,
                                lib_peaks_matched,
                                ref_pep_cand,
                                (ref_spec_row_indices_split+decoy_spec_row_indices_split),
                                (ref_spec_values_split+decoy_spec_values_split),
                                [library[i]["frags"] for i in ref_pep_cand],
                                ref_ms1_error)
        
        single_matched_rows = np.where(np.sum(sparse_lib_matrix>0,1)==1)[0]
        
        # new_row_indices_split = [[peak_idx_convertor[j] for j in i] for i in ref_spec_row_indices_split]
        unique_row_indices_split = [[peak_idx_convertor[j] in single_matched_rows for j in i] for i in ref_spec_row_indices_split]
        unique_frags = [i[j] for i,j in zip(lib_frag_mz,unique_row_indices_split)]
        unique_frags_int = [i[j] for i,j in zip(obs_frag_int,unique_row_indices_split)]
        
        ####################################
        if decoy:
            decoy_features = get_features(np.stack([rt_mz[window_idxs[decoy_peaks_in_dia],0],decoy_mz[decoy_peaks_in_dia]],1),
                                          decoy_spec_values_split,
                                            decoy_spec_row_indices_split,
                                            decoy_spec_col_indices_split,
                                            ref_spec_values_split,
                                            ref_spec_row_indices_split,
                                            ref_spec_col_indices_split,
                                            decoy_peaks_in_dia,
                                            dia_spectrum,
                                            prec_rt,
                                            window_idxs,
                                            dia_spec_int,
                                            lib_coefficients,
                                            sparse_lib_matrix,
                                            sparse_row_indices,
                                            sparse_col_indices,
                                            decoy_lib_peaks_matched,
                                            decoy_pep_cand,
                                            (ref_spec_row_indices_split+decoy_spec_row_indices_split),
                                            (ref_spec_values_split+decoy_spec_values_split),
                                            [converted_frags[i] for i in decoy_peaks_in_dia],
                                            decoy_ms1_error)
        
            # new_row_indices_split = [[peak_idx_convertor[j] for j in i] for i in decoy_spec_row_indices_split]
            unique_row_indices_split_decoy = [[peak_idx_convertor[j] in single_matched_rows for j in i] for i in decoy_spec_row_indices_split]
            unique_frags_decoy = [i[j] for i,j in zip(decoy_lib_frag_mz,unique_row_indices_split_decoy)]
            unique_frags_int_decoy = [i[j] for i,j in zip(decoy_obs_frag_int,unique_row_indices_split_decoy)]
                
        ####################################
            
    #Select non-zero coeffs
    # Note: many coeffs are non-zero but essentially zero!! Perhaps set less than 1e-7??
    non_zero_coeffs = [c for c in lib_coefficients if c!=0]
    non_zero_coeffs_idxs = [i for i,c in enumerate(lib_coefficients) if c!=0]
    # print(f"N: {len(lib_coefficients)}, {len(non_zero_coeffs)}")
    if config.args.timeplex:
        output = [[0,spec_idx,ms1_spec.scan_num,0,0,-1,prec_mz,prec_rt,*np.zeros(len(names)-7)]]
    else:
        output = [[0,spec_idx,ms1_spec.scan_num,0,0,prec_mz,prec_rt,*np.zeros(len(names)-7)]]
    
    if len(non_zero_coeffs)>0:
        lib_spec_ids = [ref_pep_cand[i] for i in range(len(ref_pep_cand)) if lib_coefficients[i] != 0]
        if decoy:
            updated_decoy_offset = int(max(ref_sparse_col_indices))+1 if len(ref_sparse_col_indices)>0 else 0
            decoy_spec_ids = [decoy_pep_cand[i] for i in range(len(decoy_pep_cand)) if lib_coefficients[updated_decoy_offset+i] != 0]
        
            all_spec_ids = lib_spec_ids+decoy_spec_ids
            all_features = np.concatenate((features,decoy_features))
            all_ms2_frags = [[";".join(map(str,j)) for j in i] for i in zip(frag_names+decoy_frag_names,
                                                                            frag_errors+decoy_frag_errors,
                                                                            lib_frag_mz+decoy_lib_frag_mz,
                                                                            lib_frag_int+decoy_lib_frag_int,
                                                                            obs_frag_int+decoy_obs_frag_int,
                                                                            unique_frags+unique_frags_decoy,
                                                                            unique_frags_int+unique_frags_int_decoy)]
            
            
        else:
            all_spec_ids = lib_spec_ids
            all_features = features
            all_ms2_frags = [[";".join(map(str,j)) for j in i] for i in zip(frag_names,
                                                                            frag_errors,
                                                                            lib_frag_mz,
                                                                            lib_frag_int,
                                                                            obs_frag_int,
                                                                            unique_frags,
                                                                            unique_frags_int)]
            
        return_prot = config.protein_column in library[next(iter(library))]
        
        if config.args.timeplex:
            output = [[non_zero_coeffs[i],
                       spec_idx,
                       ms1_spec.scan_num,
                       all_spec_ids[i][0],
                       all_spec_ids[i][1],
                       all_spec_ids[i][2],
                       prec_mz,
                       prec_rt,
                       *all_features[j],
                       *all_ms2_frags[j],
                       config.args.mzml,
                       library[(re.sub("Decoy_","",all_spec_ids[i][0]),all_spec_ids[i][1],all_spec_ids[i][2])][config.protein_column] if return_prot else "NA" ]
                       for i,j in zip(range(len(non_zero_coeffs)),non_zero_coeffs_idxs)]
        
        else:
            
            output = [[non_zero_coeffs[i],
                       spec_idx,
                       ms1_spec.scan_num,
                       all_spec_ids[i][0],
                       all_spec_ids[i][1],
                       prec_mz,
                       prec_rt,
                       *all_features[j],
                       *all_ms2_frags[j],
                       config.args.mzml,
                       library[(re.sub("Decoy_","",all_spec_ids[i][0]),all_spec_ids[i][1])][config.protein_column] if return_prot else "NA" ]
                       for i,j in zip(range(len(non_zero_coeffs)),non_zero_coeffs_idxs)]
            
        # lib_spec_ids = [ref_pep_cand[i] for i in range(len(ref_pep_cand)) if lib_coefficients[i] != 0]
        # output = [[non_zero_coeffs[i],spec_idx,lib_spec_ids[i][0],lib_spec_ids[i][1],prec_mz,prec_rt,*features[j]] for i,j in zip(range(len(non_zero_coeffs)),non_zero_coeffs_idxs)]
    
    if return_frags:
        return output, [frag_errors,lib_frag_mz]
    else:
        return output


#taken from get features 
#ref spec values and row indices are arguments to get features. T
frac_lib_intensity = [np.sum(i) for i in ref_spec_values_split] # all ints sum to 1 so these give frac
frac_dia_intensity = [np.sum(dia_spectrum[i,1])/tic for i in ref_spec_row_indices_split]
frac_dia_intensity_pred = [(i*c)/j for i,j,c in zip(frac_lib_intensity,frac_dia_intensity,lib_coefficients)]

    