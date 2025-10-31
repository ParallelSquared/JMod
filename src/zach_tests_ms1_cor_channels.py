import pytest
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.ms1_cor_channels import get_seqs_and_mzs, get_other_channels, minmax_spec_window, get_ms2_vals, build_ms2_interpolator, compute_isotopes, get_trace_int_numba, get_isotope_traces_vectorized, fit_channel_isotopes_numba, fill_scan_values
from src.mass_tags import massTag, read_json_to_massTag
from src.utils.io.load_files import SpectrumFile, Spectrum
from brainpy._c.isotopic_distribution import TheoreticalPeak
import pandas as pd
import math
from brainpy._c.isotopic_distribution import TheoreticalPeak as Peak
import copy



class Test_get_other_channels():

    def compare_outputs(self, output_dict, channel_dict):
        assert output_dict.keys() == channel_dict.keys()
        output_seqs = [item[0] for item in output_dict.values()]
        output_mzs = np.array([item[1] for item in output_dict.values()])
        channel_seqs = [item[0] for item in channel_dict.values()]
        channel_mzs = np.array([item[1] for item in channel_dict.values()])
        assert output_seqs == channel_seqs
        np.testing.assert_array_almost_equal(output_mzs, channel_mzs, decimal=6)

    def test_one_tag(self):
        channel_dict = get_other_channels(('A(PSMtag_5plex-0)AAAADLANR', 2.0), 626.31433, read_json_to_massTag("src\\MassTags\\", "PSMtag_5plex.json"))
        output_dict = {'PSMtag_5plex-0': ['A(PSMtag_5plex-0)AAAADLANR', 626.31433], 'PSMtag_5plex-4': ['A(PSMtag_5plex-4)AAAADLANR', 628.3198080300001], 'PSMtag_5plex-8': ['A(PSMtag_5plex-8)AAAADLANR', 630.32774935], 'PSMtag_5plex-12': ['A(PSMtag_5plex-12)AAAADLANR', 632.331299055], 'PSMtag_5plex-16': ['A(PSMtag_5plex-16)AAAADLANR', 634.33361711]}
        self.compare_outputs(output_dict, channel_dict)

    def test_two_tags(self):
        channel_dict = get_other_channels(('A(PSMtag_5plex-0)AAAADLANR(PSMtag_5plex-0)', 2.0), 780.372376195, read_json_to_massTag("src\\MassTags\\", "PSMtag_5plex.json"))
        output_dict = {'PSMtag_5plex-0': ['A(PSMtag_5plex-0)AAAADLANR(PSMtag_5plex-0)', 780.372376195], 'PSMtag_5plex-4': ['A(PSMtag_5plex-4)AAAADLANR(PSMtag_5plex-4)', 784.38333225104], 'PSMtag_5plex-8': ['A(PSMtag_5plex-8)AAAADLANR(PSMtag_5plex-8)', 788.3992148974], 'PSMtag_5plex-12': ['A(PSMtag_5plex-12)AAAADLANR(PSMtag_5plex-12)', 792.4063143042], 'PSMtag_5plex-16': ['A(PSMtag_5plex-16)AAAADLANR(PSMtag_5plex-16)', 796.41095041584]}
        self.compare_outputs(output_dict, channel_dict)

    def test_unimod_with_channel_name(self):
        channel_dict = get_other_channels(('A(PSMtag_5plex-4)AAAAC(UniMod:4)DLANR', 2.0), 708.4005400300001, read_json_to_massTag("src\\MassTags\\", "PSMtag_5plex.json"))
        output_dict = {'PSMtag_5plex-0': ['A(PSMtag_5plex-0)AAAAC(UniMod:4)DLANR', 706.3950620019801], 'PSMtag_5plex-4': ['A(PSMtag_5plex-4)AAAAC(UniMod:4)DLANR', 708.4005400300001], 'PSMtag_5plex-8': ['A(PSMtag_5plex-8)AAAAC(UniMod:4)DLANR', 710.4084813531802], 'PSMtag_5plex-12': ['A(PSMtag_5plex-12)AAAAC(UniMod:4)DLANR', 712.4120310565801], 'PSMtag_5plex-16': ['A(PSMtag_5plex-16)AAAAC(UniMod:4)DLANR', 714.4143491124001]}
        self.compare_outputs(output_dict, channel_dict)

    def test_two_channels(self):
        with pytest.raises(AssertionError):
            channel_dict = get_other_channels(('A(PSMtag_5plex-4)AAAADLANR(PSMtag_5plex-0)', 2.0), 780.372376195, read_json_to_massTag("src\\MassTags\\", "PSMtag_5plex.json"))
    
    def test_channel_not_in_tag(self):
        with pytest.raises(AssertionError):
            channel_dict = get_other_channels(('A(PSMtag_5plex-2)AAAADLANR', 2.0), 780.372376195, read_json_to_massTag("src\\MassTags\\", "PSMtag_5plex.json"))

    


class Test_get_seqs_and_mzs():
    
    def get_fdc_group(self, add_timeplex=None):
        df = pd.DataFrame({
        "mz": [500.123, 502.130, 504.135, 506.140, 340.1, 560.3],
        "z": [2, 2, 2, 2, 3, 2],
        "seq": [
            "A(PSMtag_5plex-0)AAAADLANR",
            "A(PSMtag_5plex-4)AAAADLANR",
            "A(PSMtag_5plex-8)AAAADLANR",
            "A(PSMtag_5plex-12)AAAADLANR",
            "A(PSMtag_5plex-12)AAAADLANR",
            "Another Sequence"
        ],
        "untag_seq": [
            "AAAAADLANR",
            "AAAAADLANR",
            "AAAAADLANR",
            "AAAAADLANR",
            "AAAAADLANR",
            "Another Sequence"
        ],
        "coeff": [5345.0, 6631.24, 5739.2, 8761.4, 10001, 55673.2],
        "Ms1_spec_id": [11983, 11971, 11983, 11983, 11982, 12002],
        "rt": [18.1, 18.0, 18.1, 18.0, 18.0, 18.3]
        })

        if add_timeplex:
            df["time_channel"] = add_timeplex
            fdc_group = df.groupby(["untag_seq","z","time_channel"])
        else:
            fdc_group = df.groupby(["untag_seq","z"])

        return fdc_group
    
    def compare_outputs(self, x, y):
        for (a, b) in list(zip(x, y)):
            assert a == b
   

    def test_plexDIA(self):
    
        prec_seqs, prec_mzs, prec_z, prec_rt, top_ms1_spec_idx, largest_coeff_scans, time_channel = get_seqs_and_mzs(self.get_fdc_group(), False, read_json_to_massTag("src\\MassTags\\", "PSMtag_5plex.json"), ('AAAAADLANR', 2.0))
        x = [prec_seqs, prec_mzs, prec_z, prec_rt, top_ms1_spec_idx, largest_coeff_scans, time_channel]
        output_seqs = ('A(PSMtag_5plex-0)AAAADLANR', 'A(PSMtag_5plex-4)AAAADLANR', 'A(PSMtag_5plex-8)AAAADLANR', 'A(PSMtag_5plex-12)AAAADLANR', 'A(PSMtag_5plex-16)AAAADLANR') 
        output_mzs = (500.1230309454, 502.12850897342, 504.13645029659995, 506.14, 508.14231805582)
        output_z = 2.0
        output_rt = 18.0
        output_top_idx = 11983
        output_largest_coeff_scans = [11983, 11971, 11983, 11983]
        output_time_channel = None
        y = [output_seqs, output_mzs, output_z, output_rt, output_top_idx, output_largest_coeff_scans, output_time_channel]
        self.compare_outputs(x, y)

    def test_plexDIA_timeplex(self):
        prec_seqs, prec_mzs, prec_z, prec_rt, top_ms1_spec_idx, largest_coeff_scans, time_channel = get_seqs_and_mzs(self.get_fdc_group(add_timeplex=[1, 2, 1, 1, 1, 1]), True, read_json_to_massTag("src\\MassTags\\", "PSMtag_5plex.json"), ('AAAAADLANR', 2.0, 1))
        x = [prec_seqs, prec_mzs, prec_z, prec_rt, top_ms1_spec_idx, largest_coeff_scans, time_channel]
        output_seqs = ('A(PSMtag_5plex-0)AAAADLANR', 'A(PSMtag_5plex-4)AAAADLANR', 'A(PSMtag_5plex-8)AAAADLANR', 'A(PSMtag_5plex-12)AAAADLANR', 'A(PSMtag_5plex-16)AAAADLANR') 
        output_mzs = (500.1230309454, 502.12850897342, 504.13645029659995, 506.14, 508.14231805582)
        output_z = 2.0
        output_rt = 18.0
        output_top_idx = 11983
        output_largest_coeff_scans = [11983, 11983, 11983]
        output_time_channel = 1
        y = [output_seqs, output_mzs, output_z, output_rt, output_top_idx, output_largest_coeff_scans, output_time_channel]
        self.compare_outputs(x, y)


class Test_minmax_spec_window():

    def test_basic(self):
        largest_coeff_scans = [20, 21, 25]

        all_spectra = SpectrumFile() 
        all_spectra.ms1scans = []
        all_spectra.ms2scans = [] 
        all_spectra.scan_pos = {}
        for scan_num in range(60): 
            spec = Spectrum()
            spec.scan_num = scan_num
            all_spectra.ms1scans.append(spec)
            all_spectra.scan_pos[scan_num] = [1, len(all_spectra.ms1scans)-1]

        ms1_spectra = all_spectra.ms1scans
        ms1_spec_idxs = np.array([i.scan_num for i in ms1_spectra])
        window_half_width = 10

        all_scans, spectra_subset = minmax_spec_window(largest_coeff_scans, ms1_spec_idxs, ms1_spectra, all_spectra, window_half_width)

        assert all_scans == [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
        assert [spec.scan_num for spec in spectra_subset] == all_scans

    def test_at_edges(self):
        largest_coeff_scans = [5, 6, 7]

        all_spectra = SpectrumFile() 
        all_spectra.ms1scans = []
        all_spectra.ms2scans = [] 
        all_spectra.scan_pos = {}
        for scan_num in range(9): 
            spec = Spectrum()
            spec.scan_num = scan_num
            all_spectra.ms1scans.append(spec)
            all_spectra.scan_pos[scan_num] = [1, len(all_spectra.ms1scans)-1]

        ms1_spectra = all_spectra.ms1scans
        ms1_spec_idxs = np.array([i.scan_num for i in ms1_spectra])
        window_half_width = 10

        all_scans, spectra_subset = minmax_spec_window(largest_coeff_scans, ms1_spec_idxs, ms1_spectra, all_spectra, window_half_width)

        assert all_scans == [0, 1, 2, 3, 4, 5, 6, 7, 8]
        assert [spec.scan_num for spec in spectra_subset] == all_scans


class Test_fit_channel_isotopes_numba():

    def move_peaks_above(self, peak_num, cutoff):
        if peak_num > cutoff:
            return 0.01
        else:
            return 0
        
    def fivePlex_fourDa_spacing(self):
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

        return normalized_mz_intensity_dict, per_channel_iso_intensity_dict, z

    def test_5plex_no_noise_or_missed_peaks(self):
        normalized_mz_intensity_dict, per_channel_iso_intensity_dict, z = self.fivePlex_fourDa_spacing()
                
        dia_spec = Spectrum()
        dia_spec.mz = np.array(list(normalized_mz_intensity_dict.keys()))
        dia_spec.intens = np.array(list(normalized_mz_intensity_dict.values()))

        all_iso = [[TheoreticalPeak(mz=mz, intensity=i, charge=z) for mz, i in per_channel_iso_intensity_dict[channel].items()] for channel in per_channel_iso_intensity_dict.keys()]
        mz_ppm = 1e-6
        pred_coeff, obs_peaks, fit_matrix = fit_channel_isotopes_numba(dia_spec, all_iso, mz_ppm)

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

    def test_5plex_small_shift_no_missed_peaks(self):
        normalized_mz_intensity_dict, per_channel_iso_intensity_dict, z = self.fivePlex_fourDa_spacing()
                
        dia_spec = Spectrum()
        dia_spec.mz = np.array(list(normalized_mz_intensity_dict.keys())) + 1e-5
        dia_spec.intens = np.array(list(normalized_mz_intensity_dict.values()))

        all_iso = [[TheoreticalPeak(mz=mz, intensity=i, charge=z) for mz, i in per_channel_iso_intensity_dict[channel].items()] for channel in per_channel_iso_intensity_dict.keys()]
        mz_ppm = 1e-6
        pred_coeff, obs_peaks, fit_matrix = fit_channel_isotopes_numba(dia_spec, all_iso, mz_ppm)

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

    def test_5plex_noise_peaks_above_below_and_in(self):
        normalized_mz_intensity_dict, per_channel_iso_intensity_dict, z = self.fivePlex_fourDa_spacing()
                
        dia_spec = Spectrum()
        mzs = np.array(list(normalized_mz_intensity_dict.keys()))
        intenss = np.array(list(normalized_mz_intensity_dict.values()))

        dia_spec.mz = np.concatenate([np.array([30.5]), mzs[:6], np.array([np.mean(mzs[5:7])]), mzs[6:], np.array([41.5])])
        dia_spec.intens = np.concatenate([np.array([100]), intenss[:6], np.array([np.mean(intenss[5:7])]), intenss[6:], np.array([100])])

        all_iso = [[TheoreticalPeak(mz=mz, intensity=i, charge=z) for mz, i in per_channel_iso_intensity_dict[channel].items()] for channel in per_channel_iso_intensity_dict.keys()]
        mz_ppm = 1e-6
        pred_coeff, obs_peaks, fit_matrix = fit_channel_isotopes_numba(dia_spec, all_iso, mz_ppm)

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

    def test_no_matched_peaks(self):
        normalized_mz_intensity_dict, per_channel_iso_intensity_dict, z = self.fivePlex_fourDa_spacing()
                
        dia_spec = Spectrum()
        dia_spec.mz = np.array(list(normalized_mz_intensity_dict.keys())) + 0.01
        dia_spec.intens = np.array(list(normalized_mz_intensity_dict.values()))

        all_iso = [[TheoreticalPeak(mz=mz, intensity=i, charge=z) for mz, i in per_channel_iso_intensity_dict[channel].items()] for channel in per_channel_iso_intensity_dict.keys()]
        mz_ppm = 1e-6
        pred_coeff, obs_peaks, fit_matrix = fit_channel_isotopes_numba(dia_spec, all_iso, mz_ppm)

        pred_coeff_expected = np.array([0, 0, 0, 0, 0])
        obs_peaks_expected = [0, 0, 0, 0, 0]
        fit_matrix_expected = np.array([
            [0.99965425, 0., 0., 0., 0.],
            [0., 0.99975006, 0., 0., 0.],
            [0., 0., 0.99982422, 0., 0.],
            [0., 0., 0., 0.99988029, 0.],
            [0., 0., 0., 0., 0.99992152]
        ])

        ## These are what fit mTRAQ isotopes returned
        # obs_peaks_expected = []  
        # fit_matrix_expected = []

        np.testing.assert_array_almost_equal(pred_coeff, pred_coeff_expected, decimal=6)
        np.testing.assert_array_almost_equal(obs_peaks, obs_peaks_expected, decimal=6)
        np.testing.assert_array_almost_equal(fit_matrix, fit_matrix_expected, decimal=6)

    def test_5plex_half_matched_peaks(self):
        normalized_mz_intensity_dict, per_channel_iso_intensity_dict, z = self.fivePlex_fourDa_spacing()
                
        dia_spec = Spectrum()
        dia_spec.mz = np.array([key + (i%2)/100 for i, key in enumerate(normalized_mz_intensity_dict.keys())])
        dia_spec.intens = np.array(list(normalized_mz_intensity_dict.values()))

        all_iso = [[TheoreticalPeak(mz=mz, intensity=i, charge=z) for mz, i in per_channel_iso_intensity_dict[channel].items()] for channel in per_channel_iso_intensity_dict.keys()]
        mz_ppm = 1e-6
        pred_coeff, obs_peaks, fit_matrix = fit_channel_isotopes_numba(dia_spec, all_iso, mz_ppm)

        pred_coeff_expected = np.array([0.08964466, 0.18621283, 0.14516634, 0.30031454, 0.05187122])
        obs_peaks_expected = [
            6.84074722e-02, 1.23539665e-02, 1.42774404e-01, 2.23791245e-02,
            1.11746745e-01, 1.50447987e-02, 2.31840837e-01, 2.66465614e-02,
            4.06397816e-02, 3.87731585e-03, 5.67691894e-05, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00
        ]
        fit_matrix_expected = np.array([
            [0.54715664, 0., 0., 0., 0.],
            [0.09881311, 0., 0., 0., 0.],
            [0.00277757, 0.5696012, 0., 0., 0.],
            [0., 0.08949963, 0., 0., 0.],
            [0., 0.00217791, 0.59296645, 0., 0.],
            [0., 0., 0.0802238, 0., 0.],
            [0., 0., 0.00167116, 0.61729014, 0.],
            [0., 0., 0., 0.07104411, 0.],
            [0., 0., 0., 0.00125039, 0.6426116],
            [0., 0., 0., 0., 0.06202536],
            [0., 0., 0., 0., 0.00090814],
            [0.35090692, 0., 0., 0., 0.],
            [0., 0.33847132, 0., 0., 0.],
            [0., 0., 0.32496281, 0., 0.],
            [0., 0., 0., 0.31029565, 0.],
            [0., 0., 0., 0., 0.29437642]
        ])

        np.testing.assert_array_almost_equal(pred_coeff, pred_coeff_expected, decimal=6)
        np.testing.assert_array_almost_equal(obs_peaks, obs_peaks_expected, decimal=6)
        np.testing.assert_array_almost_equal(fit_matrix, fit_matrix_expected, decimal=6)

    def test_2_channels_matched_peaks(self):
        normalized_mz_intensity_dict, per_channel_iso_intensity_dict, z = self.fivePlex_fourDa_spacing()
                
        dia_spec = Spectrum()
        dia_spec.mz = np.array([key + self.move_peaks_above(i, 6) for i, key in enumerate(normalized_mz_intensity_dict.keys())])
        dia_spec.intens = np.array(list(normalized_mz_intensity_dict.values()))

        all_iso = [[TheoreticalPeak(mz=mz, intensity=i, charge=z) for mz, i in per_channel_iso_intensity_dict[channel].items()] for channel in per_channel_iso_intensity_dict.keys()]
        mz_ppm = 1e-6
        pred_coeff, obs_peaks, fit_matrix = fit_channel_isotopes_numba(dia_spec, all_iso, mz_ppm)

        pred_coeff_expected = np.array([0.1250243, 0.24985216, 0.0, 0.0, 0.0])
        obs_peaks_expected = [
            0.06840747, 0.04145907, 0.01235397, 0.00241256, 
            0.1427744, 0.08056485, 0.02237912, 0.0, 0.0,
            0.0, 0.0, 0.0      
        ]
        fit_matrix_expected = np.array([
            [0.54715664,  0.0,        0.0,         0.0,         0.0       ],
            [0.33161009,  0.0,        0.0,         0.0,         0.0       ],
            [0.09881311,  0.0,        0.0,         0.0,         0.0       ],
            [0.01929684,  0.0,        0.0,         0.0,         0.0       ],
            [0.00277757,  0.5696012,  0.0,         0.0,         0.0       ],
            [0.0,         0.32219866, 0.0,         0.0,         0.0       ],
            [0.0,         0.08949963, 0.0,         0.0,         0.0       ],
            [0.0,         0.0,        0.0,         0.0,         0.0       ],
            [0.0,         0.01845057, 0.0,         0.0,         0.0       ],
            [0.0,         0.0,        0.99982422,  0.0,         0.0       ],
            [0.0,         0.0,        0.0,         0.99988029,  0.0       ],
            [0.0,         0.0,        0.0,         0.0,         0.99992152]
        ])

        np.testing.assert_array_almost_equal(pred_coeff, pred_coeff_expected, decimal=6)
        np.testing.assert_array_almost_equal(obs_peaks, obs_peaks_expected, decimal=6)
        np.testing.assert_array_almost_equal(fit_matrix, fit_matrix_expected, decimal=6)


class Test_get_ms2_vals():
    def initialize_data(self):
        prec_seq = 'A(PSMtag_5plex-0)AAAADLANR'
        prec_z = 2
        prec_rt = 2.1
        time_channel = None
        timeplex = False
        rt_tol = 0.01
        prec_mz = 626.31433
        ms2_spec_idxs = np.array([1 * i for i in range(1, 101) if i%10 != 1]) #leaves ms1_spec_idxs 1, 11, ... 91
        ms2_rt = np.array([0.1 * x for x in ms2_spec_idxs])
        window_min = 300
        window_max = 900
        number_of_bins = 9
        bin_size = (window_max - window_min) / 9
        bottom_of_window_set = [window_min + bin_size*i for i in range (0,number_of_bins)]
        top_of_window_set = [window_min+0.000001 + (bin_size*(i+1)) for i in range (0,number_of_bins)]
        windows = np.array([x for pair in zip(bottom_of_window_set, top_of_window_set) for x in pair])
        bottom_of_window, top_of_window = [], []
        while len(bottom_of_window) < len(ms2_spec_idxs):
            bottom_of_window += bottom_of_window_set
            top_of_window += top_of_window_set
        bottom_of_window = np.array(bottom_of_window)
        top_of_window = np.array(top_of_window)
        assert len(ms2_spec_idxs) == len(ms2_rt) == len(bottom_of_window) == len(top_of_window)

        return prec_seq, prec_z, prec_rt, time_channel, timeplex, ms2_rt, rt_tol, prec_mz, bottom_of_window, top_of_window, ms2_spec_idxs

    def define_windows(self):
        window_min = 300
        window_max = 900
        number_of_bins = 9
        bin_size = (window_max - window_min) / 9
        bottom_of_window_set = [window_min + bin_size*i for i in range (0,number_of_bins)]
        top_of_window_set = [window_min+0.000001 + (bin_size*(i+1)) for i in range (0,number_of_bins)]
        windows = np.array([x for pair in zip(bottom_of_window_set, top_of_window_set) for x in pair])
        return windows

    def define_ms2_spec_ids(self, ms1_spec_ids, mzs):
        ref_coords = np.searchsorted(self.define_windows(), mzs) 
        bin_coords = (ref_coords-1)//2
        ms2_spec_idxs = ((np.array(ms1_spec_ids) // 10) * 10) + bin_coords + 2
        return ms2_spec_idxs

    def get_grouped_decoy_coeffs(   self,
                                    mzs = [626.31433, 626.31433, 626.31433, 626.31433, 628.31433, 630.31433, 634.31433, 634.31433, 560.3],
                                    zs = [2, 2, 2, 2, 2, 2, 2, 2, 2],
                                    seqs = [
                                        "A(PSMtag_5plex-0)AAAADLANR",
                                        "A(PSMtag_5plex-0)AAAADLANR",
                                        "A(PSMtag_5plex-0)AAAADLANR",
                                        "A(PSMtag_5plex-0)AAAADLANR",
                                        "A(PSMtag_5plex-4)AAAADLANR",
                                        "A(PSMtag_5plex-8)AAAADLANR",
                                        "A(PSMtag_5plex-16)AAAADLANR",
                                        "A(PSMtag_5plex-16)AAAADLANR",
                                        "Another Sequence"
                                        ],
                                    coeffs = [1000, 2000, 3000, 4000, 6631.24, 5739.2, 10001, 8761.4, 55673.2],
                                    ms1_spec_ids = [1, 11, 21, 31, 1, 1, 61, 71, 1],
                                    spec_ids = [None, None, None, None, None, None, None, None, None],
                                    rts = [0.1, 1.1, 2.1, 3.1, 0.1, 0.1, 6.1, 7.1, 0.1],
                                    add_timeplex=None
                                    ):
        
        if all(x is None for x in spec_ids):
            spec_ids = self.define_ms2_spec_ids(ms1_spec_ids, mzs)
        df = pd.DataFrame({
        "mz": mzs,
        "z": zs,
        "seq": seqs,
        "coeff": coeffs,
        "Ms1_spec_id": ms1_spec_ids,
        "spec_id": spec_ids,
        "rt": rts
        })

        if add_timeplex:
            df["time_channel"] = add_timeplex
            grouped_decoy_coeffs = df.groupby(["seq","z","time_channel"])
        else:
            grouped_decoy_coeffs = df.groupby(["seq","z"])

        return grouped_decoy_coeffs

    
    def compare_outputs(self, output, expected):
        assert output[1] == expected[1]
        assert np.allclose(np.array(list(output[0].keys())), np.array(list(expected[0].keys())))
        assert np.allclose(np.array(list(output[0].values())), np.array(list(expected[0].values())))
        assert output[2] == expected[2]
    
    def test_one_tag(self):
        grouped_decoy_coeffs = self.get_grouped_decoy_coeffs()
        prec_seq, prec_z, prec_rt, time_channel, timeplex, ms2_rt, rt_tol, prec_mz, bottom_of_window, top_of_window, ms2_spec_idxs = self.initialize_data()
        
        ms2_vals, highest_ranked_spec, channel_key = get_ms2_vals(prec_seq, prec_z, prec_rt, time_channel, timeplex, grouped_decoy_coeffs, ms2_rt, rt_tol, prec_mz, bottom_of_window, top_of_window, ms2_spec_idxs)
        output = (ms2_vals, highest_ranked_spec, channel_key) 
        expected = (
            {6: 1000.0, 16: 2000.0, 26: 3000.0, 36: 4000.0},
            31,
            ('A(PSMtag_5plex-0)AAAADLANR', 2)
                    )
        self.compare_outputs(output, expected)

    def test_two_tags(self):
        seqs = [
                "A(PSMtag_5plex-0)AAAADLANR(PSMtag_5plex-0)",
                "A(PSMtag_5plex-0)AAAADLANR(PSMtag_5plex-0)",
                "A(PSMtag_5plex-0)AAAADLANR(PSMtag_5plex-0)",
                "A(PSMtag_5plex-0)AAAADLANR(PSMtag_5plex-0)",
                "A(PSMtag_5plex-4)AAAADLANR(PSMtag_5plex-4)",
                "A(PSMtag_5plex-8)AAAADLANR(PSMtag_5plex-8)",
                "A(PSMtag_5plex-12)AAAADLANR(PSMtag_5plex-12)",
                "A(PSMtag_5plex-12)AAAADLANR(PSMtag_5plex-12)",
                "Another Sequence"
                ]
        grouped_decoy_coeffs = self.get_grouped_decoy_coeffs(seqs=seqs)
        prec_seq, prec_z, prec_rt, time_channel, timeplex, ms2_rt, rt_tol, prec_mz, bottom_of_window, top_of_window, ms2_spec_idxs = self.initialize_data()
        prec_seq = "A(PSMtag_5plex-0)AAAADLANR(PSMtag_5plex-0)"

        ms2_vals, highest_ranked_spec, channel_key = get_ms2_vals(prec_seq, prec_z, prec_rt, time_channel, timeplex, grouped_decoy_coeffs, ms2_rt, rt_tol, prec_mz, bottom_of_window, top_of_window, ms2_spec_idxs)
        output = (ms2_vals, highest_ranked_spec, channel_key) 
        expected = (
            {6: 1000.0, 16: 2000.0, 26: 3000.0, 36: 4000.0},
            31,
            ('A(PSMtag_5plex-0)AAAADLANR(PSMtag_5plex-0)', 2)
                    )
        self.compare_outputs(output, expected)

    def test_no_tag(self):
        seqs = [
                "AAAAADLANR",
                "AAAAADLANR",
                "AAAAADLANR",
                "AAAAADLANR",
                "A(PSMtag_5plex-4)AAAADLANR(PSMtag_5plex-4)",
                "A(PSMtag_5plex-8)AAAADLANR(PSMtag_5plex-8)",
                "A(PSMtag_5plex-12)AAAADLANR(PSMtag_5plex-12)",
                "A(PSMtag_5plex-12)AAAADLANR(PSMtag_5plex-12)",
                "Another Sequence"
                ]
        grouped_decoy_coeffs = self.get_grouped_decoy_coeffs(seqs=seqs)
        prec_seq, prec_z, prec_rt, time_channel, timeplex, ms2_rt, rt_tol, prec_mz, bottom_of_window, top_of_window, ms2_spec_idxs = self.initialize_data()
        prec_seq = "AAAAADLANR"

        ms2_vals, highest_ranked_spec, channel_key = get_ms2_vals(prec_seq, prec_z, prec_rt, time_channel, timeplex, grouped_decoy_coeffs, ms2_rt, rt_tol, prec_mz, bottom_of_window, top_of_window, ms2_spec_idxs)
        output = (ms2_vals, highest_ranked_spec, channel_key) 
        expected = (
            {6: 1000.0, 16: 2000.0, 26: 3000.0, 36: 4000.0},
            31,
            ('AAAAADLANR', 2)
                    )
        self.compare_outputs(output, expected)

    def test_another_channel(self):
        grouped_decoy_coeffs = self.get_grouped_decoy_coeffs()
        prec_seq, prec_z, prec_rt, time_channel, timeplex, ms2_rt, rt_tol, prec_mz, bottom_of_window, top_of_window, ms2_spec_idxs = self.initialize_data()
        prec_seq = "A(PSMtag_5plex-16)AAAADLANR"
        prec_mz = 634.31433

        ms2_vals, highest_ranked_spec, channel_key = get_ms2_vals(prec_seq, prec_z, prec_rt, time_channel, timeplex, grouped_decoy_coeffs, ms2_rt, rt_tol, prec_mz, bottom_of_window, top_of_window, ms2_spec_idxs)
        output = (ms2_vals, highest_ranked_spec, channel_key) 
        expected = (
            {67: 10001, 77: 8761.4},
            61,
            ('A(PSMtag_5plex-16)AAAADLANR', 2)
                    )
        self.compare_outputs(output, expected)

    def test_missing_inner_val(self):
        grouped_decoy_coeffs = self.get_grouped_decoy_coeffs(
                                    mzs = [626.31433, 626.31433, 626.31433, 628.31433, 630.31433, 634.31433, 634.31433, 560.3],
                                    zs = [2, 2, 2, 2, 2, 2, 3, 2],
                                    seqs = [
                                        "A(PSMtag_5plex-0)AAAADLANR",
                                        "A(PSMtag_5plex-0)AAAADLANR",
                                        "A(PSMtag_5plex-0)AAAADLANR",
                                        "A(PSMtag_5plex-4)AAAADLANR",
                                        "A(PSMtag_5plex-8)AAAADLANR",
                                        "A(PSMtag_5plex-16)AAAADLANR",
                                        "A(PSMtag_5plex-16)AAAADLANR",
                                        "Another Sequence"
                                        ],
                                    coeffs = [1000, 3000, 4000, 6631.24, 5739.2, 10001, 8761.4, 55673.2],
                                    ms1_spec_ids = [1, 21, 31, 1, 1, 1, 61, 71],
                                    spec_ids = [None, None, None, None, None, None, None, None],
                                    rts = [0.1, 2.1, 3.1, 0.1, 0.1, 0.1, 6.1, 7.1]
                                    )
        
        prec_seq, prec_z, prec_rt, time_channel, timeplex, ms2_rt, rt_tol, prec_mz, bottom_of_window, top_of_window, ms2_spec_idxs = self.initialize_data()
        
        ms2_vals, highest_ranked_spec, channel_key = get_ms2_vals(prec_seq, prec_z, prec_rt, time_channel, timeplex, grouped_decoy_coeffs, ms2_rt, rt_tol, prec_mz, bottom_of_window, top_of_window, ms2_spec_idxs)
        output = (ms2_vals, highest_ranked_spec, channel_key) 
        expected = (
            {6: 1000.0, 16: 0.001, 26: 3000.0, 36: 4000.0},
            31,
            ('A(PSMtag_5plex-0)AAAADLANR', 2)
                    )
        self.compare_outputs(output, expected)

    def test_high_rt_tol(self):
        grouped_decoy_coeffs = self.get_grouped_decoy_coeffs()
        prec_seq, prec_z, prec_rt, time_channel, timeplex, ms2_rt, rt_tol, prec_mz, bottom_of_window, top_of_window, ms2_spec_idxs = self.initialize_data()
        rt_tol = 2

        ms2_vals, highest_ranked_spec, channel_key = get_ms2_vals(prec_seq, prec_z, prec_rt, time_channel, timeplex, grouped_decoy_coeffs, ms2_rt, rt_tol, prec_mz, bottom_of_window, top_of_window, ms2_spec_idxs)
        output = (ms2_vals, highest_ranked_spec, channel_key) 
        expected = (
            {6: 1000.0, 16: 2000.0, 26: 3000.0, 36: 4000.0, 46: 0.001},
            31,
            ('A(PSMtag_5plex-0)AAAADLANR', 2)
                    )
        self.compare_outputs(output, expected)

    def test_key_not_in_GDC(self):
        grouped_decoy_coeffs = self.get_grouped_decoy_coeffs()
        prec_seq, prec_z, prec_rt, time_channel, timeplex, ms2_rt, rt_tol, prec_mz, bottom_of_window, top_of_window, ms2_spec_idxs = self.initialize_data()
        prec_seq = "adhkdskdhhf"

        ms2_vals, highest_ranked_spec, channel_key = get_ms2_vals(prec_seq, prec_z, prec_rt, time_channel, timeplex, grouped_decoy_coeffs, ms2_rt, rt_tol, prec_mz, bottom_of_window, top_of_window, ms2_spec_idxs)
        output = (ms2_vals, highest_ranked_spec, channel_key) 
        expected = (
            {0:0},
            None,
            ("adhkdskdhhf", 2)
                    )
        self.compare_outputs(output, expected)

    def test_highest_ranked_spec_tie(self):  ##This doesn't necessarily need to be the tie behavior
        grouped_decoy_coeffs = self.get_grouped_decoy_coeffs(coeffs = [1000, 2000, 4000, 4000, 6631.24, 5739.2, 10001, 8761.4, 55673.2])
        prec_seq, prec_z, prec_rt, time_channel, timeplex, ms2_rt, rt_tol, prec_mz, bottom_of_window, top_of_window, ms2_spec_idxs = self.initialize_data()

        ms2_vals, highest_ranked_spec, channel_key = get_ms2_vals(prec_seq, prec_z, prec_rt, time_channel, timeplex, grouped_decoy_coeffs, ms2_rt, rt_tol, prec_mz, bottom_of_window, top_of_window, ms2_spec_idxs)
        output = (ms2_vals, highest_ranked_spec, channel_key) 
        expected = (
            {6: 1000.0, 16: 2000.0, 26: 4000.0, 36: 4000.0},
            21,
            ('A(PSMtag_5plex-0)AAAADLANR', 2)
                    )
        self.compare_outputs(output, expected)

    def test_charge_is_three(self):  
        grouped_decoy_coeffs = self.get_grouped_decoy_coeffs(zs = [3, 3, 3, 3, 3, 3, 3, 3, 3])
        prec_seq, prec_z, prec_rt, time_channel, timeplex, ms2_rt, rt_tol, prec_mz, bottom_of_window, top_of_window, ms2_spec_idxs = self.initialize_data()
        prec_z = 3

        ms2_vals, highest_ranked_spec, channel_key = get_ms2_vals(prec_seq, prec_z, prec_rt, time_channel, timeplex, grouped_decoy_coeffs, ms2_rt, rt_tol, prec_mz, bottom_of_window, top_of_window, ms2_spec_idxs)
        output = (ms2_vals, highest_ranked_spec, channel_key) 
        expected = (
            {6: 1000.0, 16: 2000.0, 26: 3000.0, 36: 4000.0},
            31,
            ('A(PSMtag_5plex-0)AAAADLANR', 3)
                    )
        self.compare_outputs(output, expected)

    def test_mixed_charges(self):  
        grouped_decoy_coeffs = self.get_grouped_decoy_coeffs(zs = [2, 3, 2, 3, 3, 3, 3, 3, 3])
        prec_seq, prec_z, prec_rt, time_channel, timeplex, ms2_rt, rt_tol, prec_mz, bottom_of_window, top_of_window, ms2_spec_idxs = self.initialize_data()
        prec_z = 3

        ms2_vals, highest_ranked_spec, channel_key = get_ms2_vals(prec_seq, prec_z, prec_rt, time_channel, timeplex, grouped_decoy_coeffs, ms2_rt, rt_tol, prec_mz, bottom_of_window, top_of_window, ms2_spec_idxs)
        output = (ms2_vals, highest_ranked_spec, channel_key) 
        expected = (
            {16: 2000.0, 26: 0.001, 36: 4000.0},
            31,
            ('A(PSMtag_5plex-0)AAAADLANR', 3)
                    )
        self.compare_outputs(output, expected)

    def test_timeplex(self):
        grouped_decoy_coeffs = self.get_grouped_decoy_coeffs(add_timeplex= [3, 3, 3, 3, 3, 3, 3, 3, 3])
        prec_seq, prec_z, prec_rt, time_channel, timeplex, ms2_rt, rt_tol, prec_mz, bottom_of_window, top_of_window, ms2_spec_idxs = self.initialize_data()
        timeplex = True
        time_channel = 3

        ms2_vals, highest_ranked_spec, channel_key = get_ms2_vals(prec_seq, prec_z, prec_rt, time_channel, timeplex, grouped_decoy_coeffs, ms2_rt, rt_tol, prec_mz, bottom_of_window, top_of_window, ms2_spec_idxs)
        output = (ms2_vals, highest_ranked_spec, channel_key) 
        expected = (
            {6: 1000.0, 16: 2000.0, 26: 3000.0, 36: 4000.0},
            31,
            ('A(PSMtag_5plex-0)AAAADLANR', 2, 3)
                    )
        self.compare_outputs(output, expected)

    def test_mixed_timeplexes(self):  
        grouped_decoy_coeffs = self.get_grouped_decoy_coeffs(add_timeplex = [2, 3, 2, 3, 3, 3, 3, 3, 3])
        prec_seq, prec_z, prec_rt, time_channel, timeplex, ms2_rt, rt_tol, prec_mz, bottom_of_window, top_of_window, ms2_spec_idxs = self.initialize_data()
        timeplex = True
        time_channel = 3

        ms2_vals, highest_ranked_spec, channel_key = get_ms2_vals(prec_seq, prec_z, prec_rt, time_channel, timeplex, grouped_decoy_coeffs, ms2_rt, rt_tol, prec_mz, bottom_of_window, top_of_window, ms2_spec_idxs)
        output = (ms2_vals, highest_ranked_spec, channel_key) 
        expected = (
            {16: 2000.0, 26: 0.001, 36: 4000.0},
            31,
            ('A(PSMtag_5plex-0)AAAADLANR', 2, 3)
                    )
        self.compare_outputs(output, expected)

class Test_build_ms2_interpolator():

    def test_make_interpolator(self):
        ms2_vals = {0: 0.001, 1: 10, 2: 20, 3: 30, 4:20, 5:10, 6:0.001}
        f = build_ms2_interpolator(ms2_vals)

        assert np.isclose(f(1.5), 15.0)
        assert np.isclose(f(4.5), 15.0)
        assert np.isclose(f(2.5), 25.0)
        assert np.isclose(f(3.5), 25.0)
        assert np.isclose(f(0), 0.001)
        assert np.isclose(f(6), 0.001)
        assert np.isnan(f(-1))
        assert np.isnan(f(7))

class Test_compute_isotopes():

    tag = read_json_to_massTag("src\\MassTags\\", "PSMtag_5plex.json")

    tag_no_comp = copy.deepcopy(tag)
    tag_no_comp.channel_comp = None

    def compare_outputs(self, output, expected_output):
        assert len(output) == len(expected_output)
        for peak_a, peak_b in list(zip(output, expected_output)):
            assert np.isclose(peak_a.mz, peak_b.mz, atol=1e-6)
            assert np.isclose(peak_a.intensity, peak_b.intensity, atol=1e-6)
            assert peak_a.charge == peak_b.charge

    def test_compute_isotopes_no_tag(self):
        prec_seq = "PEPTIDEK"
        prec_mz = 464.734739987125
        prec_z = 2
        num_iso = 6
        tag = self.tag

        isotopes = compute_isotopes(prec_seq, prec_mz, prec_z, num_iso, tag)
        expected = [Peak(mz=464.734740, intensity=0.600835, charge=2), Peak(mz=465.236229, intensity=0.287848, charge=2), Peak(mz=465.737521, intensity=0.087264, charge=2), Peak(mz=466.238790, intensity=0.019787, charge=2), Peak(mz=466.740027, intensity=0.003680, charge=2), Peak(mz=467.241248, intensity=0.000586, charge=2)]
        self.compare_outputs(isotopes, expected)

    def test_compute_isotopes_one_tag(self):
        prec_seq = "P(PSMtag_5plex-0)EPTIDEK"
        prec_mz = 618.792786
        prec_z = 2
        num_iso = 6
        tag = self.tag

        isotopes = compute_isotopes(prec_seq, prec_mz, prec_z, num_iso, tag)
        expected = [Peak(mz=618.792786, intensity=0.487119, charge=2), Peak(mz=619.294302, intensity=0.333215, charge=2), Peak(mz=619.795684, intensity=0.131298, charge=2), Peak(mz=620.297017, intensity=0.037846, charge=2), Peak(mz=620.798315, intensity=0.008791, charge=2), Peak(mz=621.299592, intensity=0.001731, charge=2)]
        self.compare_outputs(isotopes, expected)

    def test_compute_isotopes_two_tags(self):
        prec_seq = "P(PSMtag_5plex-0)EPTIDEK(PSMtag_5plex-0)"
        prec_mz = 772.850832
        prec_z = 2
        num_iso = 6
        tag = self.tag

        isotopes = compute_isotopes(prec_seq, prec_mz, prec_z, num_iso, tag)
        expected = [Peak(mz=772.850832, intensity=0.395076, charge=2), Peak(mz=773.352363, intensity=0.351233, charge=2), Peak(mz=773.853795, intensity=0.172197, charge=2), Peak(mz=774.355175, intensity=0.060560, charge=2), Peak(mz=774.856520, intensity=0.016947, charge=2), Peak(mz=775.357839, intensity=0.003987, charge=2)]
        self.compare_outputs(isotopes, expected)

    def test_compute_isotopes_two_tags_and_mod(self):
        prec_seq = "P(PSMtag_5plex-0)EPT(Unimod:21)IDEK(PSMtag_5plex-0)"
        prec_mz = 812.833998
        prec_z = 2
        num_iso = 6
        tag = self.tag

        isotopes = compute_isotopes(prec_seq, prec_mz, prec_z, num_iso, tag)
        expected = [Peak(mz=812.833998, intensity=0.392212, charge=2), Peak(mz=813.335530, intensity=0.349179, charge=2), Peak(mz=813.836951, intensity=0.173805, charge=2), Peak(mz=814.338319, intensity=0.062487, charge=2), Peak(mz=814.839653, intensity=0.017961, charge=2), Peak(mz=815.340962, intensity=0.004355, charge=2)]
        self.compare_outputs(isotopes, expected)

    def test_change_iso_num(self):
        prec_seq = "PEPTIDEK"
        prec_mz = 310.1589188136733
        prec_z = 3
        num_iso = 2
        tag = self.tag

        isotopes = compute_isotopes(prec_seq, prec_mz, prec_z, num_iso, tag)
        expected = [Peak(mz=310.158919, intensity=0.676096, charge=3), Peak(mz=310.493245, intensity=0.323904, charge=3)]
        self.compare_outputs(isotopes, expected)

    def test_fake_channel(self):
        with pytest.raises(KeyError) as exc_info:
            compute_isotopes("P(PSMtag_5plex-2)EPTIDEK", 100, 2, 2, self.tag)
        assert "'2'" in str(exc_info.value)

    def test_fake_tag_name(self):
        prec_seq = "P(TagChannel2)EPTIDEK"
        prec_mz = 464.734739987125
        prec_z = 2
        num_iso = 6
        tag = self.tag

        isotopes = compute_isotopes(prec_seq, prec_mz, prec_z, num_iso, tag)
        expected = [Peak(mz=464.734740, intensity=0.600835, charge=2), Peak(mz=465.236229, intensity=0.287848, charge=2), Peak(mz=465.737521, intensity=0.087264, charge=2), Peak(mz=466.238790, intensity=0.019787, charge=2), Peak(mz=466.740027, intensity=0.003680, charge=2), Peak(mz=467.241248, intensity=0.000586, charge=2)]
        self.compare_outputs(isotopes, expected)

    def test_one_tag_no_comps(self):
        prec_seq = "P(PSMtag_5plex-0)EPTIDEK"
        prec_mz = 618.792786
        prec_z = 2
        num_iso = 6
        tag = self.tag_no_comp

        isotopes = compute_isotopes(prec_seq, prec_mz, prec_z, num_iso, tag)
        expected = [Peak(mz=618.792786, intensity=0.600835, charge=2), Peak(mz=619.294302, intensity=0.287848, charge=2), Peak(mz=619.795684, intensity=0.087264, charge=2), Peak(mz=620.297017, intensity=0.019787, charge=2), Peak(mz=620.798315, intensity=0.003680, charge=2), Peak(mz=621.299592, intensity=0.000586, charge=2)]
        self.compare_outputs(isotopes, expected)

    def test_two_tags_no_comps(self):
        prec_seq = "P(PSMtag_5plex-0)EPTIDEK(PSMtag_5plex-0)"
        prec_mz = 618.792786
        prec_z = 2
        num_iso = 6
        tag = self.tag_no_comp

        isotopes = compute_isotopes(prec_seq, prec_mz, prec_z, num_iso, tag)
        expected = [Peak(mz=618.792786, intensity=0.600835, charge=2), Peak(mz=619.294302, intensity=0.287848, charge=2), Peak(mz=619.795684, intensity=0.087264, charge=2), Peak(mz=620.297017, intensity=0.019787, charge=2), Peak(mz=620.798315, intensity=0.003680, charge=2), Peak(mz=621.299592, intensity=0.000586, charge=2)]
        self.compare_outputs(isotopes, expected)

    def test_different_channel(self):
        prec_seq = "P(PSMtag_5plex-4)EPTIDEK"
        prec_mz = 620.798264
        prec_z = 2
        num_iso = 2
        tag = self.tag

        isotopes = compute_isotopes(prec_seq, prec_mz, prec_z, num_iso, tag)
        expected = [Peak(mz=620.798264, intensity=0.601670, charge=2), Peak(mz=621.299775, intensity=0.398330, charge=2)]
        self.compare_outputs(isotopes, expected)

class Test_get_trace_int_numba():

    mz_ppm = 1e-6
    min_int = 0.001

    def test_only_matched_peaks(self):
        dia_spec = Spectrum()
        dia_spec.mz = np.array([100, 101, 102])
        dia_spec.intens = np.array([1000, 2000, 3000])
        isotopes = [Peak(mz=100, intensity=0.6, charge=2), Peak(mz=101, intensity=0.3, charge=2), Peak(mz=102, intensity=0.1, charge=2)]

        iso_ints = get_trace_int_numba(dia_spec.mz, dia_spec.intens, np.array([isotope.mz for isotope in isotopes]), self.mz_ppm, self.min_int)
        expected = np.array([1000, 2000, 3000])
        assert np.allclose(iso_ints, expected)

    def test_empty_spec(self):
        dia_spec = Spectrum()
        dia_spec.mz = np.array([])
        dia_spec.intens = np.array([])
        isotopes = [Peak(mz=100, intensity=0.6, charge=2), Peak(mz=101, intensity=0.3, charge=2), Peak(mz=102, intensity=0.1, charge=2)]

        iso_ints = get_trace_int_numba(dia_spec.mz, dia_spec.intens, np.array([isotope.mz for isotope in isotopes]), self.mz_ppm, self.min_int)
        expected = np.array([0.001, 0.001, 0.001])
        assert np.allclose(iso_ints, expected)

    def test_no_matched(self):
        dia_spec = Spectrum()
        dia_spec.mz = np.array([100.5, 101.5, 102.5])
        dia_spec.intens = np.array([1000, 2000, 3000])
        isotopes = [Peak(mz=100, intensity=0.6, charge=2), Peak(mz=101, intensity=0.3, charge=2), Peak(mz=102, intensity=0.1, charge=2)]

        iso_ints = get_trace_int_numba(dia_spec.mz, dia_spec.intens, np.array([isotope.mz for isotope in isotopes]), self.mz_ppm, self.min_int)
        expected = np.array([0.001, 0.001, 0.001])
        assert np.allclose(iso_ints, expected)

    def test_noisy(self):
        dia_spec = Spectrum()
        dia_spec.mz = np.array([100, 100.5, 101, 101.5, 102, 102.5])
        dia_spec.intens = np.array([1000, 1, 2000, 2, 3000, 3])
        isotopes = [Peak(mz=100, intensity=0.6, charge=2), Peak(mz=101, intensity=0.3, charge=2), Peak(mz=102, intensity=0.1, charge=2)]

        iso_ints = get_trace_int_numba(dia_spec.mz, dia_spec.intens, np.array([isotope.mz for isotope in isotopes]), self.mz_ppm, self.min_int)
        expected = np.array([1000, 2000, 3000])
        assert np.allclose(iso_ints, expected)

    def test_peaks_within_tol(self):
        dia_spec = Spectrum()
        dia_spec.mz = np.array([100+100*1e-7, 101+101*1e-7, 102+102*1e-7])
        dia_spec.intens = np.array([1000, 2000, 3000])
        isotopes = [Peak(mz=100, intensity=0.6, charge=2), Peak(mz=101, intensity=0.3, charge=2), Peak(mz=102, intensity=0.1, charge=2)]

        iso_ints = get_trace_int_numba(dia_spec.mz, dia_spec.intens, np.array([isotope.mz for isotope in isotopes]), self.mz_ppm, self.min_int)
        expected = np.array([1000, 2000, 3000])
        assert np.allclose(iso_ints, expected)

    def test_peaks_outside_tol(self):
        dia_spec = Spectrum()
        dia_spec.mz = np.array([100+100*1e-5, 101+101*1e-5, 102+102*1e-5])
        dia_spec.intens = np.array([1000, 2000, 3000])
        isotopes = [Peak(mz=100, intensity=0.6, charge=2), Peak(mz=101, intensity=0.3, charge=2), Peak(mz=102, intensity=0.1, charge=2)]

        iso_ints = get_trace_int_numba(dia_spec.mz, dia_spec.intens, np.array([isotope.mz for isotope in isotopes]), self.mz_ppm, self.min_int)
        expected = np.array([0.001, 0.001, 0.001])
        assert np.allclose(iso_ints, expected)

    def test_peaks_in_and_outside_tol(self):
        dia_spec = Spectrum()
        dia_spec.mz = np.array([100+100*1e-5, 101+101*1e-7, 102+102*1e-5])
        dia_spec.intens = np.array([1000, 2000, 3000])
        isotopes = [Peak(mz=100, intensity=0.6, charge=2), Peak(mz=101, intensity=0.3, charge=2), Peak(mz=102, intensity=0.1, charge=2)]

        iso_ints = get_trace_int_numba(dia_spec.mz, dia_spec.intens, np.array([isotope.mz for isotope in isotopes]), self.mz_ppm, self.min_int)
        expected = np.array([0.001, 2000, 0.001])
        assert np.allclose(iso_ints, expected)

    def test_two_peaks_within_tol(self):
        dia_spec = Spectrum()
        dia_spec.mz = np.array([100+100*1e-5, 101+101*1e-8, 101+101*1e-7, 102+102*1e-5])
        dia_spec.intens = np.array([1000, 2000, 1, 3000])
        isotopes = [Peak(mz=100, intensity=0.6, charge=2), Peak(mz=101, intensity=0.3, charge=2), Peak(mz=102, intensity=0.1, charge=2)]

        iso_ints = get_trace_int_numba(dia_spec.mz, dia_spec.intens, np.array([isotope.mz for isotope in isotopes]), self.mz_ppm, self.min_int)
        expected = np.array([0.001, 2000, 0.001])
        assert np.allclose(iso_ints, expected)

class Test_get_isotope_traces_vectorized():

    def compare_outputs(self, output, expected):
        assert len(output) == len(expected)
        for o, e in zip(output, expected):
            assert list(o.keys()) == list(e.keys())
            assert np.allclose(list(o.values()), list(e.values()))


    def test_all_peaks_match(self):
        dia_spec_list = []
        for i, x in enumerate([1, 11, 21], start=1):
            dia_spec = Spectrum()
            dia_spec.mz = np.array([100, 101, 102, 103])
            factor = 2-np.abs(2-i)
            dia_spec.intens = np.array([5000*factor, 3000*factor, 1500*factor, 500*factor])
            dia_spec.scan_num = x
            dia_spec_list.append(dia_spec)

        isotopes = [Peak(mz=100, intensity=0.5, charge=2), Peak(mz=101, intensity=0.3, charge=2), Peak(mz=102, intensity=0.15, charge=2), Peak(mz=103, intensity=0.05, charge=2)]
        all_isotope_traces = get_isotope_traces_vectorized(isotopes, 1e-6, dia_spec_list)
        expected = [{1: 5000.0, 11: 10000.0, 21: 5000.0}, {1: 3000.0, 11: 6000.0, 21: 3000.0}, {1: 1500.0, 11: 3000.0, 21: 1500.0}, {1: 500.0, 11: 1000.0, 21: 500.0}]
        self.compare_outputs(all_isotope_traces, expected)


    def test_all_peaks_match_within_tolerance(self):
        dia_spec_list = []
        for i, x in enumerate([1, 11, 21], start=1):
            dia_spec = Spectrum()
            dia_spec.mz = np.array([100+100*1e-7, 101+101*1e-7, 102+102*1e-7, 103+103*1e-7])
            factor = 2-np.abs(2-i)
            dia_spec.intens = np.array([5000*factor, 3000*factor, 1500*factor, 500*factor])
            dia_spec.scan_num = x
            dia_spec_list.append(dia_spec)

        isotopes = [Peak(mz=100, intensity=0.5, charge=2), Peak(mz=101, intensity=0.3, charge=2), Peak(mz=102, intensity=0.15, charge=2), Peak(mz=103, intensity=0.05, charge=2)]
        all_isotope_traces = get_isotope_traces_vectorized(isotopes, 1e-6, dia_spec_list)
        expected = [{1: 5000.0, 11: 10000.0, 21: 5000.0}, {1: 3000.0, 11: 6000.0, 21: 3000.0}, {1: 1500.0, 11: 3000.0, 21: 1500.0}, {1: 500.0, 11: 1000.0, 21: 500.0}]
        self.compare_outputs(all_isotope_traces, expected)

    def test_no_peaks_match_within_tolerance(self):
        dia_spec_list = []
        for i, x in enumerate([1, 11, 21], start=1):
            dia_spec = Spectrum()
            dia_spec.mz = np.array([100+100*1e-5, 101+101*1e-5, 102+102*1e-5, 103+103*1e-5])
            factor = 2-np.abs(2-i)
            dia_spec.intens = np.array([5000*factor, 3000*factor, 1500*factor, 500*factor])
            dia_spec.scan_num = x
            dia_spec_list.append(dia_spec)

        isotopes = [Peak(mz=100, intensity=0.5, charge=2), Peak(mz=101, intensity=0.3, charge=2), Peak(mz=102, intensity=0.15, charge=2), Peak(mz=103, intensity=0.05, charge=2)]
        all_isotope_traces = get_isotope_traces_vectorized(isotopes, 1e-6, dia_spec_list)
        expected = [{1: 0.001, 11: 0.001, 21: 0.001}, {1: 0.001, 11: 0.001, 21: 0.001}, {1: 0.001, 11: 0.001, 21: 0.001}, {1: 0.001, 11: 0.001, 21: 0.001}]
        self.compare_outputs(all_isotope_traces, expected)

    def test_one_scan_empty(self):
        dia_spec_list = []
        for i, x in enumerate([1, 11, 21], start=1):
            dia_spec = Spectrum()
            dia_spec.mz = np.array([100, 101, 102, 103])
            factor = 2-np.abs(2-i)
            dia_spec.intens = np.array([5000*factor, 3000*factor, 1500*factor, 500*factor])
            dia_spec.scan_num = x
            dia_spec_list.append(dia_spec)

        dia_spec_list[2].mz = np.array([])
        dia_spec_list[2].intens = np.array([])

        isotopes = [Peak(mz=100, intensity=0.5, charge=2), Peak(mz=101, intensity=0.3, charge=2), Peak(mz=102, intensity=0.15, charge=2), Peak(mz=103, intensity=0.05, charge=2)]
        all_isotope_traces = get_isotope_traces_vectorized(isotopes, 1e-6, dia_spec_list)
        expected = [{1: 5000.0, 11: 10000.0, 21: 0.001}, {1: 3000.0, 11: 6000.0, 21: 0.001}, {1: 1500.0, 11: 3000.0, 21: 0.001}, {1: 500.0, 11: 1000.0, 21: 0.001}]
        self.compare_outputs(all_isotope_traces, expected)

    def test_one_peak_missing(self):
        dia_spec_list = []
        for i, x in enumerate([1, 11, 21], start=1):
            dia_spec = Spectrum()
            dia_spec.mz = np.array([100, 101.5, 102, 103])
            factor = 2-np.abs(2-i)
            dia_spec.intens = np.array([5000*factor, 3000*factor, 1500*factor, 500*factor])
            dia_spec.scan_num = x
            dia_spec_list.append(dia_spec)

        isotopes = [Peak(mz=100, intensity=0.5, charge=2), Peak(mz=101, intensity=0.3, charge=2), Peak(mz=102, intensity=0.15, charge=2), Peak(mz=103, intensity=0.05, charge=2)]
        all_isotope_traces = get_isotope_traces_vectorized(isotopes, 1e-6, dia_spec_list)
        expected = [{1: 5000.0, 11: 10000.0, 21: 5000.0}, {1: 0.001, 11: 0.001, 21: 0.001}, {1: 1500.0, 11: 3000.0, 21: 1500.0}, {1: 500.0, 11: 1000.0, 21: 500.0}]
        self.compare_outputs(all_isotope_traces, expected)


class Test_fill_scan_values():

    def test_it(self):
        ms2_vals = {
                    102: 0.001,
                    112: 100,
                    122: 200,
                    132: 100,
                    142: 0.001
                    }
        
        ms1_vals = {
                    101: 0.001,
                    111: 0.001,
                    121: 0.001,
                    131: 0.001,
                    141: 0.001
                    }
        
        all_scans = [101, 111, 121, 131, 141]

        interp_func = build_ms2_interpolator(ms2_vals)
        all_ms2_vals = fill_scan_values(all_scans, interp_func, ms1_vals)
        expected = {101: np.nan, 111: 90.0001, 121: 190.0, 131: 110.0, 141: 10.000900000000001}

        assert list(all_ms2_vals.keys()) == list(expected.keys())
        assert np.allclose(list(all_ms2_vals.values()), list(expected.values()), equal_nan=True)








def main():
    pass


if __name__ == "__main__":
    main()

