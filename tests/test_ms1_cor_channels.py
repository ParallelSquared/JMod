import pytest
import numpy as np
import sys, os
os.environ['NUMBA_DISABLE_JIT'] = '1'
from src.ms1_cor_channels import get_seqs_and_mzs, get_other_channels, minmax_spec_window, get_ms2_vals, build_ms2_interpolator, compute_isotopes, get_trace_int_numba, get_isotope_traces_vectorized, fit_channel_isotopes_numba, fill_scan_values, get_ms1_index_of_max, moving_average, get_ms1_peak, filter_all_scans, compute_ms1_ms2_cors, select_scans_to_search, fit_isotopes_and_score, ms1_cor_channels, get_matrix_to_fit_numba
from src.mass_tags import massTag, read_json_to_massTag
from src.utils.io.load_files import SpectrumFile, Spectrum
from brainpy._c.isotopic_distribution import TheoreticalPeak
import pandas as pd
import math
from brainpy._c.isotopic_distribution import TheoreticalPeak as Peak
import copy
from collections import namedtuple
PearsonRResult = namedtuple("PearsonRResult", ["statistic", "pvalue"])
from src.utils.misc_functions import p_result



class DummySpectrumFile:
    def __init__(self, scans):
        self.scan_pos = {}
        self.ms1scans = []
        self.ms2scans = []
        for scan in scans:
            if scan["ms level"] == 1:
                spec = DummySpectrum(scan)
                self.ms1scans.append(spec)
                self.scan_pos[spec.scan_num] = [scan["ms level"],len(self.ms1scans)-1]
            if scan["ms level"] == 2:
                spec = DummySpectrum(scan)
                self.ms2scans.append(spec)
                self.scan_pos[spec.scan_num] = [scan["ms level"],len(self.ms2scans)-1]
                
    
    def get_by_idx(self,idx):
        level, level_idx = self.scan_pos[idx]
        if level==1:
            return self.ms1scans[level_idx]
        elif level==2:
            return self.ms2scans[level_idx]
        
class DummySpectrum():
    def __init__(self,scan=None):
        self.level=None
        self.RT=None
        self.mz=None
        self.intens=None

        if scan:
            self.get_vals(scan)

    def get_vals(self,scan):
        self.scan_num = scan["scan_num"]
        self.level=scan["ms level"]
        self.RT = scan['RT']
        self.mz = scan["m/z array"]
        self.intens = scan["intensity array"]
        if self.level==2:
            isolationWindow = scan["precursorList"]["precursor"][0]["isolationWindow"]
            self.ms1window = isolationWindow["isolation window target m/z"]+np.array([-1,1])*[isolationWindow['isolation window lower offset'],isolationWindow['isolation window upper offset']]
    
    def peak_list(self):
            return(np.array([self.mz,self.intens]))
    
class Test_get_other_channels():

    def compare_outputs(self, expected_list, actual_list):
        """Compare expected and actual lists of [seq, mz] pairs."""
        assert len(expected_list) == len(actual_list), f"Length mismatch: {len(expected_list)} vs {len(actual_list)}"
        expected_seqs = [item[0] for item in expected_list]
        expected_mzs = np.array([item[1] for item in expected_list])
        actual_seqs = [item[0] for item in actual_list]
        actual_mzs = np.array([item[1] for item in actual_list])
        assert expected_seqs == actual_seqs, f"Sequences mismatch: {expected_seqs} vs {actual_seqs}"
        np.testing.assert_array_almost_equal(expected_mzs, actual_mzs, decimal=6)

    def test_one_tag(self):
        result = get_other_channels(('A(PSMtag_5plex-0)AAAADLANR', 2.0), 626.31433, [read_json_to_massTag("src/MassTags", "PSMtag_5plex.json")])
        expected = [
            ['A(PSMtag_5plex-0)AAAADLANR', 626.31433],
            ['A(PSMtag_5plex-4)AAAADLANR', 628.3198080300001],
            ['A(PSMtag_5plex-8)AAAADLANR', 630.32774935],
            ['A(PSMtag_5plex-12)AAAADLANR', 632.331299055],
            ['A(PSMtag_5plex-16)AAAADLANR', 634.33361711]
        ]
        self.compare_outputs(expected, result)

    def test_two_tags(self):
        result = get_other_channels(('A(PSMtag_5plex-0)AAAADLANR(PSMtag_5plex-0)', 2.0), 780.372376195, [read_json_to_massTag("tests/MassTags/", "PSMtag_5plex.json")])
        expected = [
            ['A(PSMtag_5plex-0)AAAADLANR(PSMtag_5plex-0)', 780.372376195],
            ['A(PSMtag_5plex-4)AAAADLANR(PSMtag_5plex-4)', 784.38333225104],
            ['A(PSMtag_5plex-8)AAAADLANR(PSMtag_5plex-8)', 788.3992148974],
            ['A(PSMtag_5plex-12)AAAADLANR(PSMtag_5plex-12)', 792.4063143042],
            ['A(PSMtag_5plex-16)AAAADLANR(PSMtag_5plex-16)', 796.41095041584]
        ]
        self.compare_outputs(expected, result)

    def test_unimod_with_channel_name(self):
        result = get_other_channels(('A(PSMtag_5plex-4)AAAAC(UniMod:4)DLANR', 2.0), 708.4005400300001, [read_json_to_massTag("tests/MassTags/", "PSMtag_5plex.json")])
        expected = [
            ['A(PSMtag_5plex-0)AAAAC(UniMod:4)DLANR', 706.3950620019801],
            ['A(PSMtag_5plex-4)AAAAC(UniMod:4)DLANR', 708.4005400300001],
            ['A(PSMtag_5plex-8)AAAAC(UniMod:4)DLANR', 710.4084813531802],
            ['A(PSMtag_5plex-12)AAAAC(UniMod:4)DLANR', 712.4120310565801],
            ['A(PSMtag_5plex-16)AAAAC(UniMod:4)DLANR', 714.4143491124001]
        ]
        self.compare_outputs(expected, result)

    def test_two_channels(self):
        # Test with mismatched channels - the function now handles this without raising
        # It should return results where all tag occurrences get the same channel
        result = get_other_channels(('A(PSMtag_5plex-4)AAAADLANR(PSMtag_5plex-0)', 2.0), 780.372376195, [read_json_to_massTag("tests/MassTags/", "PSMtag_5plex.json")])
        # Function now returns 5 channel combinations (replacing ALL tag occurrences with same channel)
        assert len(result) == 5

    def test_channel_not_in_tag(self):
        # Test with invalid channel - now raises KeyError instead of AssertionError
        with pytest.raises(KeyError):
            get_other_channels(('A(PSMtag_5plex-2)AAAADLANR', 2.0), 780.372376195, [read_json_to_massTag("tests/MassTags/", "PSMtag_5plex.json")])

    


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

        prec_seqs, prec_mzs, prec_z, prec_rt, top_ms1_spec_idx, largest_coeff_scans, time_channel = get_seqs_and_mzs(self.get_fdc_group(), False, read_json_to_massTag("tests/MassTags/", "PSMtag_5plex.json"), ('AAAAADLANR', 2.0), None)
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
        prec_seqs, prec_mzs, prec_z, prec_rt, top_ms1_spec_idx, largest_coeff_scans, time_channel = get_seqs_and_mzs(self.get_fdc_group(add_timeplex=[1, 2, 1, 1, 1, 1]), True, read_json_to_massTag("tests/MassTags/", "PSMtag_5plex.json"), ('AAAAADLANR', 2.0, 1), None)
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

    tag = read_json_to_massTag("tests/MassTags/", "PSMtag_5plex.json")

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

class Test_MS1_index_of_max():

    def test_empty(self):
        output = get_ms1_index_of_max({0:0}, 1, 2)
        assert output == 1
    def test_one_value(self):
        output = get_ms1_index_of_max({0.1:0.1}, 1, 2)
        assert output == 2
    def test_normal(self):
        output = get_ms1_index_of_max({11835: 100, 11857: 200}, 1, 2)
        assert output == 2

class Test_moving_average():

    def test_real_example(self):
        input = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 16807.875, 11393.623046875, 17849.171875, 23156.712890625, 16398.578125, 11658.19140625, 7275.765625, 7272.23583984375, 15155.2353515625, 6253.7275390625, 6313.9365234375, 0.001, 0.001]
        output = moving_average(input)
        expected = np.array([
                            5.00000000e-04, 7.50000000e-04, 1.00000000e-03, 1.00000000e-03,
                            1.00000000e-03, 1.00000000e-03, 1.00000000e-03, 1.00000000e-03,
                            4.20196950e+03, 7.05037501e+03, 1.15126677e+04, 1.73018457e+04,
                            1.71995215e+04, 1.72656636e+04, 1.46223120e+04, 1.06511927e+04,
                            1.03403571e+04, 8.98924109e+03, 8.74878381e+03, 6.93072510e+03,
                            3.14191652e+03, 1.57848463e+03
                        ])
        assert np.allclose(output, expected)


class Test_get_ms1_peak():

    def test_simple(self):
        dict_keys = [1, 11, 21, 31, 41, 51, 61]
        dict_values = [0.001, 0.001, 100, 200, 100, 0.001, 0.001]
        ms1_index_of_max = 31
        ms1_peak_idx, ms1_peak_edge_idxs = get_ms1_peak(dict_keys, moving_average(dict_values), ms1_index_of_max, 0)
        assert ms1_peak_idx == 31
        assert np.allclose(ms1_peak_edge_idxs, np.array([1, 61]))

    def test_empty(self):
        dict_keys = [0.0]
        dict_values = [0.0]
        ms1_index_of_max = 31
        ms1_peak_idx, ms1_peak_edge_idxs = get_ms1_peak(dict_keys, moving_average(dict_values), ms1_index_of_max, 0)
        assert ms1_peak_idx == 0.0
        assert np.allclose(ms1_peak_edge_idxs, np.array([0.0, 0.0]))
    
    def test_two_peaks(self):
        dict_keys = [1, 11, 21, 31, 41, 51, 61, 71, 81, 91]
        dict_values = [0.001, 50, 200, 50, 0.001, 0.001, 50, 200, 50, 0.001]
        ms1_index_of_max = 31
        ms1_peak_idx, ms1_peak_edge_idxs = get_ms1_peak(dict_keys, moving_average(dict_values), ms1_index_of_max, 0)
        assert ms1_peak_idx == 21
        assert np.allclose(ms1_peak_edge_idxs, np.array([1, 51]))

    def test_two_peaks_second_peak(self):
        dict_keys = [1, 11, 21, 31, 41, 51, 61, 71, 81, 91]
        dict_values = [0.001, 50, 200, 50, 0.001, 0.001, 50, 200, 50, 0.001]
        ms1_index_of_max = 71
        ms1_peak_idx, ms1_peak_edge_idxs = get_ms1_peak(dict_keys, moving_average(dict_values), ms1_index_of_max, 0)
        assert ms1_peak_idx == 71
        assert np.allclose(ms1_peak_edge_idxs, np.array([1, 91]))

    def test_more_additional_scans(self):
        dict_keys = [1, 11, 21, 31, 41, 51, 61, 71, 81, 91]
        dict_values = [0.001, 50, 200, 50, 0.001, 0.001, 50, 200, 50, 0.001]
        ms1_index_of_max = 31
        ms1_peak_idx, ms1_peak_edge_idxs = get_ms1_peak(dict_keys, moving_average(dict_values), ms1_index_of_max, 2)
        assert ms1_peak_idx == 21
        assert np.allclose(ms1_peak_edge_idxs, np.array([1, 71]))

class Test_filter_all_scans():

    def compare_outputs(self, x, y):
        assert x[0] == y[0]
        assert list(x[1].keys()) == list(y[1].keys())
        assert list(x[1].values()) == list(y[1].values())
        assert list(x[3].keys()) == list(y[3].keys())
        assert list(x[3].values()) == list(y[3].values())
        assert len(x[2]) == len(y[2])
        for a, b in zip(x[2], y[2]):
            assert list(a.keys()) == list(b.keys())
            assert list(a.values()) == list(b.values())

    def test_all_inclusive(self):
        all_scans = [1, 11, 21, 31, 41, 51, 61, 71]
        ms1_peak_edge_idxs = np.array([1, 71])
        all_ms1_vals = {x: x*100 for x in all_scans}
        all_iso_vals = [{x: x*b for x in all_scans} for b in [2, 3, 4]]
        all_ms2_vals = {x: x*50 for x in all_scans}

        channel_scans_out, all_ms1_vals_out, all_iso_vals_out, all_ms2_vals_out = filter_all_scans(all_scans, ms1_peak_edge_idxs, all_ms1_vals, all_iso_vals, all_ms2_vals)
        x = (channel_scans_out, all_ms1_vals_out, all_iso_vals_out, all_ms2_vals_out)

        expected_channel_scans_out = [1, 11, 21, 31, 41, 51, 61, 71]
        expected_all_ms1_vals = all_ms1_vals = {x: x*100 for x in all_scans}
        expected_all_iso_vals = [{x: x*b for x in all_scans} for b in [2, 3, 4]]
        expected_all_ms2_vals = {x: x*50 for x in all_scans}
        y = (expected_channel_scans_out, expected_all_ms1_vals, expected_all_iso_vals, expected_all_ms2_vals)

        self.compare_outputs(x, y)

    def test_middle_only(self):
        all_scans = [1, 11, 21, 31, 41, 51, 61, 71]
        ms1_peak_edge_idxs = np.array([11, 61])
        all_ms1_vals = {x: x*100 for x in all_scans}
        all_iso_vals = [{x: x*b for x in all_scans} for b in [2, 3, 4]]
        all_ms2_vals = {x: x*50 for x in all_scans}

        channel_scans_out, all_ms1_vals_out, all_iso_vals_out, all_ms2_vals_out = filter_all_scans(all_scans, ms1_peak_edge_idxs, all_ms1_vals, all_iso_vals, all_ms2_vals)
        x = (channel_scans_out, all_ms1_vals_out, all_iso_vals_out, all_ms2_vals_out)

        expected_channel_scans_out = [11, 21, 31, 41, 51, 61]
        expected_all_ms1_vals = all_ms1_vals = {x: x*100 for x in expected_channel_scans_out}
        expected_all_iso_vals = [{x: x*b for x in expected_channel_scans_out} for b in [2, 3, 4]]
        expected_all_ms2_vals = {x: x*50 for x in expected_channel_scans_out}
        y = (expected_channel_scans_out, expected_all_ms1_vals, expected_all_iso_vals, expected_all_ms2_vals)

        self.compare_outputs(x, y)



class Test_compute_ms1_ms2_cors():

    def compare_results(self, output, expected):
        assert len(output) == len(expected)
        for o, e in zip(output, expected):
            if isinstance(e, PearsonRResult):
                assert np.allclose([o.statistic, o.pvalue], [e.statistic, e.pvalue])
            else:
                assert np.allclose(o, e)

    def test_simple(self):
        channel_scans = [1, 11, 21, 31, 41, 51, 61, 71]
        all_ms1_vals = {x: x*100 for x in channel_scans}
        all_iso_vals = [{x: x*b for x in channel_scans} for b in [2, 3, 4, 5, 6]]
        all_ms2_vals = {x: x*50 for x in channel_scans}
        num_iso_r = 2
        isotopes = [Peak(mz=100, intensity=0.5, charge=2), Peak(mz=100.5, intensity=0.3, charge=2), Peak(mz=101, intensity=0.1, charge=2), Peak(mz=101.5, intensity=0.05, charge=2), Peak(mz=102, intensity=0.03, charge=2), Peak(mz=102.5, intensity=0.02, charge=2)]

        all_pearson_to_append, iso_ratios_to_append = compute_ms1_ms2_cors(all_ms2_vals, all_ms1_vals, all_iso_vals, num_iso_r, channel_scans, isotopes)
        expected_pearson = [1.0, 1.0, 1.0]
        expected_iso_ratios = [PearsonRResult(statistic=0.8269433745794761, pvalue=0.04233149195743158), [0.5, 0.3, 0.1, 0.05, 0.03, 0.02], [7100, 142, 213, 284, 355, 426]]

        self.compare_results(all_pearson_to_append, expected_pearson)
        self.compare_results(iso_ratios_to_append, expected_iso_ratios)

    def test_ms2_is_dif(self):
        rng = np.random.default_rng(42)
        channel_scans = [1, 11, 21, 31, 41, 51, 61, 71]
        all_ms1_vals = {x: x*100 for x in channel_scans}
        all_iso_vals = [{x: x*b for x in channel_scans} for b in [2, 3, 4, 5, 6]]
        all_ms2_vals = {x: x*50 for x in channel_scans}
        all_ms2_vals = {k: v+rng.normal(0, 1000) for k, v in all_ms2_vals.items()}
        num_iso_r = 2
        isotopes = [Peak(mz=100, intensity=0.5, charge=2), Peak(mz=100.5, intensity=0.3, charge=2), Peak(mz=101, intensity=0.1, charge=2), Peak(mz=101.5, intensity=0.05, charge=2), Peak(mz=102, intensity=0.03, charge=2), Peak(mz=102.5, intensity=0.02, charge=2)]

        all_pearson_to_append, iso_ratios_to_append = compute_ms1_ms2_cors(all_ms2_vals, all_ms1_vals, all_iso_vals, num_iso_r, channel_scans, isotopes)
        expected_pearson = [0.7054342857572004, 0.7054342857572006, 0.7054342857572004]
        expected_iso_ratios = [PearsonRResult(statistic=0.8269433745794761, pvalue=0.04233149195743158), [0.5, 0.3, 0.1, 0.05, 0.03, 0.02], [7100, 142, 213, 284, 355, 426]]

        self.compare_results(all_pearson_to_append, expected_pearson)
        self.compare_results(iso_ratios_to_append, expected_iso_ratios)

    def test_ms1_is_dif(self):
        rng = np.random.default_rng(42)
        channel_scans = [1, 11, 21, 31, 41, 51, 61, 71]
        all_ms1_vals = {x: x*100 for x in channel_scans}
        all_ms1_vals = {k: v+rng.normal(0, 1000) for k, v in all_ms1_vals.items()}
        all_iso_vals = [{x: x*b for x in channel_scans} for b in [2, 3, 4, 5, 6]]
        all_ms2_vals = {x: x*50 for x in channel_scans}
        num_iso_r = 2
        isotopes = [Peak(mz=100, intensity=0.5, charge=2), Peak(mz=100.5, intensity=0.3, charge=2), Peak(mz=101, intensity=0.1, charge=2), Peak(mz=101.5, intensity=0.05, charge=2), Peak(mz=102, intensity=0.03, charge=2), Peak(mz=102.5, intensity=0.02, charge=2)]

        all_pearson_to_append, iso_ratios_to_append = compute_ms1_ms2_cors(all_ms2_vals, all_ms1_vals, all_iso_vals, num_iso_r, channel_scans, isotopes)
        expected_pearson = [0.9109972512052685, 1.0, 1.0]
        expected_iso_ratios = [PearsonRResult(statistic=0.8260818133361278, pvalue=0.04274100520281188), [0.5, 0.3, 0.1, 0.05, 0.03, 0.02], [6783.757407656418, 142, 213, 284, 355, 426]]

        self.compare_results(all_pearson_to_append, expected_pearson)
        self.compare_results(iso_ratios_to_append, expected_iso_ratios)

    def test_change_num_iso_r(self):
        channel_scans = [1, 11, 21, 31, 41, 51, 61, 71]
        all_ms1_vals = {x: x*100 for x in channel_scans}
        all_iso_vals = [{x: x*b for x in channel_scans} for b in [2, 3, 4, 5, 6]]
        all_ms2_vals = {x: x*50 for x in channel_scans}
        num_iso_r = 4
        isotopes = [Peak(mz=100, intensity=0.5, charge=2), Peak(mz=100.5, intensity=0.3, charge=2), Peak(mz=101, intensity=0.1, charge=2), Peak(mz=101.5, intensity=0.05, charge=2), Peak(mz=102, intensity=0.03, charge=2), Peak(mz=102.5, intensity=0.02, charge=2)]

        all_pearson_to_append, iso_ratios_to_append = compute_ms1_ms2_cors(all_ms2_vals, all_ms1_vals, all_iso_vals, num_iso_r, channel_scans, isotopes)
        expected_pearson = [1.0, 1.0, 1.0, 1.0, 1.0]
        expected_iso_ratios = [PearsonRResult(statistic=0.8269433745794761, pvalue=0.04233149195743158), [0.5, 0.3, 0.1, 0.05, 0.03, 0.02], [7100, 142, 213, 284, 355, 426]]

        self.compare_results(all_pearson_to_append, expected_pearson)
        self.compare_results(iso_ratios_to_append, expected_iso_ratios)

class Test_select_scans_to_search():

    def test_all_overlap(self):
        top_ms1_spec_idx = 31
        all_scans = [1, 11, 21, 31, 41, 51, 61, 71]
        all_channel_scans = [[1, 11, 21], [31, 41, 51], [61, 71]]
        window_half_width = 2
        scans_to_search = select_scans_to_search(top_ms1_spec_idx, all_scans, all_channel_scans, window_half_width)
        expected = np.array([1, 11, 21, 31, 41, 51, 61, 71])
        assert np.allclose(scans_to_search, expected)

    def test_not_all_overlap(self):
        top_ms1_spec_idx = 31
        all_scans = [1, 11, 21, 31, 41, 51, 61, 71]
        all_channel_scans = [[11, 21], [31, 41, 51], [61, 71]]
        window_half_width = 2
        scans_to_search = select_scans_to_search(top_ms1_spec_idx, all_scans, all_channel_scans, window_half_width)
        expected = np.array([11, 21, 31, 41, 51, 61, 71])
        assert np.allclose(scans_to_search, expected)


class Test_fit_isotopes_and_score():

    

    def test_all_intensity_matched(self):
        ms1_spectra = [
            Spectrum(),
            Spectrum(),
            Spectrum()
        ]
        normalized_mz_intensity_dict, per_channel_iso_intensity_dict, z = Test_fit_channel_isotopes_numba().fivePlex_fourDa_spacing()
        ms1_spectra[0].mz = np.array([100, 101, 102])
        ms1_spectra[0].intens = np.array([1000, 500, 200])
        ms1_spectra[0].scan_num = 11
        ms1_spectra[1].mz = np.array(list(normalized_mz_intensity_dict.keys()))
        ms1_spectra[1].intens = np.array(list(normalized_mz_intensity_dict.values()))
        ms1_spectra[1].scan_num = 21
        ms1_spectra[2].mz = np.array([100, 101, 102])
        ms1_spectra[2].intens = np.array([1200, 600, 250])
        ms1_spectra[2].scan_num = 31

        ms1_spec_idxs = np.array([11, 21, 31])
        ms1_spec_idx = 21
        mz_ppm = 1e-6
        group_iso = [[TheoreticalPeak(mz=mz, intensity=i, charge=z) for mz, i in per_channel_iso_intensity_dict[channel].items()] for channel in per_channel_iso_intensity_dict]

        pred_coeff, obs_peaks, fit_matrix, fit_cor = fit_isotopes_and_score(ms1_spectra, ms1_spec_idxs, ms1_spec_idx, group_iso, mz_ppm)
        p_result_expected = PearsonRResult(statistic=1.0, pvalue=0.0)
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
        assert np.allclose([fit_cor.statistic, fit_cor.pvalue], [p_result_expected.statistic, p_result_expected.pvalue])

    def test_different_spec_idx(self):
        ms1_spectra = [
            Spectrum(),
            Spectrum(),
            Spectrum()
        ]
        normalized_mz_intensity_dict, per_channel_iso_intensity_dict, z = Test_fit_channel_isotopes_numba().fivePlex_fourDa_spacing()
        ms1_spectra[1].mz = np.array([100, 101, 102])
        ms1_spectra[1].intens = np.array([1000, 500, 200])
        ms1_spectra[1].scan_num = 11
        ms1_spectra[0].mz = np.array(list(normalized_mz_intensity_dict.keys()))
        ms1_spectra[0].intens = np.array(list(normalized_mz_intensity_dict.values()))
        ms1_spectra[0].scan_num = 21
        ms1_spectra[2].mz = np.array([100, 101, 102])
        ms1_spectra[2].intens = np.array([1200, 600, 250])
        ms1_spectra[2].scan_num = 31

        ms1_spec_idxs = np.array([11, 21, 31])
        ms1_spec_idx = 11
        mz_ppm = 1e-6
        group_iso = [[TheoreticalPeak(mz=mz, intensity=i, charge=z) for mz, i in per_channel_iso_intensity_dict[channel].items()] for channel in per_channel_iso_intensity_dict]

        pred_coeff, obs_peaks, fit_matrix, fit_cor = fit_isotopes_and_score(ms1_spectra, ms1_spec_idxs, ms1_spec_idx, group_iso, mz_ppm)
        p_result_expected = PearsonRResult(statistic=1.0, pvalue=0.0)
        pred_coeff_expected = np.array([0.12502356, 0.25004712, 0.18753534, 0.37507069, 0.06251178])

        np.testing.assert_array_almost_equal(pred_coeff, pred_coeff_expected, decimal=6)
        assert np.allclose([fit_cor.statistic, fit_cor.pvalue], [p_result_expected.statistic, p_result_expected.pvalue])

    def test_some_intensity_matched(self):
        ms1_spectra = [
            Spectrum(),
            Spectrum(),
            Spectrum()
        ]
        normalized_mz_intensity_dict, per_channel_iso_intensity_dict, z = Test_fit_channel_isotopes_numba().fivePlex_fourDa_spacing()
        ms1_spectra[0].mz = np.array([100, 101, 102])
        ms1_spectra[0].intens = np.array([1000, 500, 200])
        ms1_spectra[0].scan_num = 11
        ms1_spectra[1].mz = np.array(list(normalized_mz_intensity_dict.keys()))
        ms1_spectra[1].intens = np.array(list(normalized_mz_intensity_dict.values()))
        ms1_spectra[1].intens[:10] += 700
        ms1_spectra[1].scan_num = 21
        ms1_spectra[2].mz = np.array([100, 101, 102])
        ms1_spectra[2].intens = np.array([1200, 600, 250])
        ms1_spectra[2].scan_num = 31

        ms1_spec_idxs = np.array([11, 21, 31])
        ms1_spec_idx = 21
        mz_ppm = 1e-6
        group_iso = [[TheoreticalPeak(mz=mz, intensity=i, charge=z) for mz, i in per_channel_iso_intensity_dict[channel].items()] for channel in per_channel_iso_intensity_dict]

        pred_coeff, obs_peaks, fit_matrix, fit_cor = fit_isotopes_and_score(ms1_spectra, ms1_spec_idxs, ms1_spec_idx, group_iso, mz_ppm)
        p_result_expected = PearsonRResult(statistic=0.71961104232076, pvalue=3.4203284732692296e-05)
        assert np.allclose([fit_cor.statistic, fit_cor.pvalue], [p_result_expected.statistic, p_result_expected.pvalue])

    def test_no_intensity_matched(self):
        ms1_spectra = [
            Spectrum(),
            Spectrum(),
            Spectrum()
        ]
        normalized_mz_intensity_dict, per_channel_iso_intensity_dict, z = Test_fit_channel_isotopes_numba().fivePlex_fourDa_spacing()
        ms1_spectra[0].mz = np.array([100, 101, 102])
        ms1_spectra[0].intens = np.array([1000, 500, 200])
        ms1_spectra[0].scan_num = 11
        ms1_spectra[1].mz = np.array([100, 101, 102])
        ms1_spectra[1].intens = np.array([1000, 500, 200])
        ms1_spectra[1].scan_num = 21
        ms1_spectra[2].mz = np.array([100, 101, 102])
        ms1_spectra[2].intens = np.array([1200, 600, 250])
        ms1_spectra[2].scan_num = 31

        ms1_spec_idxs = np.array([11, 21, 31])
        ms1_spec_idx = 21
        mz_ppm = 1e-6
        group_iso = [[TheoreticalPeak(mz=mz, intensity=i, charge=z) for mz, i in per_channel_iso_intensity_dict[channel].items()] for channel in per_channel_iso_intensity_dict]

        pred_coeff, obs_peaks, fit_matrix, fit_cor = fit_isotopes_and_score(ms1_spectra, ms1_spec_idxs, ms1_spec_idx, group_iso, mz_ppm)
        p_result_expected = PearsonRResult(statistic=np.nan, pvalue=1)
        assert np.allclose([fit_cor.statistic, fit_cor.pvalue], [p_result_expected.statistic, p_result_expected.pvalue], equal_nan=True)

class Test_ms1_cor_channels():

    def compare_pearsons(self, all_group_pearson, group_pearson_out):
        not_all_equal = False
        for i, item in enumerate(group_pearson_out):
            for j, x in enumerate(item):
                y = all_group_pearson[i][j]
                if not np.allclose(x, y, equal_nan=True):
                    not_all_equal = True
        assert not not_all_equal

    def compare_ms1(self, all_ms1, all_ms1_out):
        not_all_equal = False
        for i, item in enumerate(all_ms1_out):
            for j, x in enumerate(item):
                y = all_ms1[i][j]
                for l, m in enumerate(x):
                    z = y[l]
                    if not m == z:
                        not_all_equal = True
        assert not not_all_equal

    def compare_coeff(self, all_coeff, all_coeff_out):
        not_all_equal = False
        for i, item in enumerate(all_coeff_out):
            for j, x in enumerate(item):
                y = all_coeff[i][j]
                if not x.keys() == y.keys():
                    not_all_equal = True
                
                elif not np.allclose(list(x.values()), list(y.values()), equal_nan=True):
                    not_all_equal = True
        assert not not_all_equal

    def compare_iso(self, all_iso, all_iso_out):
        not_all_equal = False
        for i, item in enumerate(all_iso):
            for j, x in enumerate(item):
                y = all_iso_out[i][j]
                for l, m in enumerate(x):
                    z = y[l]
                    if type(m) != p_result:
                        if not np.allclose(m, z, equal_nan=True):
                            not_all_equal = True
                    else:
                        if not (np.isclose(m.statistic, z.statistic, equal_nan=True) and np.isclose(m.pvalue, z.pvalue, equal_nan=True)):
                            not_all_equal = True
        assert not not_all_equal


    def compare_group_keys(self, all_group_keys, all_group_keys_out):
        not_all_equal = False
        for i, item in enumerate(all_group_keys_out):
            if not item == all_group_keys[i]:
                not_all_equal = True
        assert not not_all_equal

    def compare_fitted(self, all_fitted, all_fitted_out):
        not_all_equal = False
        skipped = False
        for i, item in enumerate(all_fitted):
            for j, x in enumerate(item):
                y = all_fitted_out[i][j]
                for l, m in enumerate(x):
                    try:
                        z = y[l]
                    except:
                        skipped = True
                        continue
                    if type(m) == list:
                        m = np.array(m)
                    if type(z) == list:
                        z = np.array(z)
                    if type(m) == np.ndarray:
                        if m.shape != z.shape:
                            print("shapes not equal")
                            not_all_equal = True
                        elif not np.allclose(m, z, equal_nan=True):
                            print("values not equal")
                            not_all_equal = True
                    elif type(m) == p_result or type(m) == PearsonRResult:
                        if not (type(z) == p_result or type(z) == PearsonRResult):
                            print("type mismatch: expected p_result or PearsonRResult")
                            not_all_equal = True
                            continue
                        elif not (np.isclose(m.statistic, z.statistic, equal_nan=True) and np.isclose(m.pvalue, z.pvalue, equal_nan=True)):
                            print("p_result values not equal")
                            not_all_equal = True
                    elif type(z) == p_result or type(z) == PearsonRResult:
                        print("type mismatch: expected non-p_result")
                        not_all_equal = True
                    else:
                        if not np.isclose(m, z, equal_nan=True):
                            print("scalar values not equal")
                            not_all_equal = True
        assert not skipped
        assert not not_all_equal

    def compare_output_dict(self, new_output_dict, new_output_dict_out):
        assert new_output_dict == new_output_dict_out

    def compare_fake_fdc_dict(self, fake_fdc_dict, fake_fdc_dict_out):
        assert fake_fdc_dict == fake_fdc_dict_out

        
    def get_spectra_and_dfs_for_testing(self, test_timeplex=False):
        
        peptide = 'A(PSMtag_5plex-0)AAAADLANR'
        ms1_spec_idxs = [101, 111, 121, 131, 141]
        ms2_spec_idxs = list(sorted(set(range(102, 151)) - set(ms1_spec_idxs)))
        spec_idxs = list(range(101, 151))
        rt_list = [0.1 * scan for scan in spec_idxs]
        prec_mz = 626.31433

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

        scans = []
        for i, spec_idx in enumerate(spec_idxs):
            scan = {}
            scan["scan_num"] = spec_idx
            scan['RT'] = rt_list[i]
            if spec_idx in ms1_spec_idxs:
                scan["ms level"] = 1
                scan["m/z array"] = np.array([626.31433, 626.31433+0.5, 626.31433+1])
                scan["intensity array"] = np.array([1500, 1000, 500])

            elif spec_idx in ms2_spec_idxs:
                scan["ms level"] = 2
                scan["m/z array"] = np.array([1, 2, 3])
                scan["intensity array"] = np.array([1, 2, 3])
                if spec_idx % 10 == 6:
                    pass
                else:
                    pass

                idx_in_ms2 = (spec_idx+-2) % 10
                bottom = bottom_of_window[idx_in_ms2]
                top = top_of_window[idx_in_ms2]
                target_mz = (bottom + top) / 2
                lower_offset = target_mz - bottom
                upper_offset = top - target_mz

                scan["precursorList"] = {
                    "precursor": [
                        {
                            "isolationWindow": {
                                "isolation window target m/z": target_mz,
                                "isolation window lower offset": lower_offset,
                                "isolation window upper offset": upper_offset,
                            }
                        }
                    ]
                }
            scans.append(scan)

        fake_spec_file = DummySpectrumFile(scans)

        base_df = pd.DataFrame({
            "z": [2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2],
            "seq": [
                "A(PSMtag_5plex-0)AAAADLANR",
                "A(PSMtag_5plex-0)AAAADLANR",
                "A(PSMtag_5plex-0)AAAADLANR",
                "A(PSMtag_5plex-0)AAAADLANR",
                "A(PSMtag_5plex-0)AAAADLANR",
                "A(PSMtag_5plex-0)AAAADLANR",
                "A(PSMtag_5plex-4)AAAADLANR",
                "A(PSMtag_5plex-8)AAAADLANR",
                "A(PSMtag_5plex-12)AAAADLANR",
                "A(PSMtag_5plex-12)AAAADLANR",
                "P(PSMtag_5plex-0)EPTIDEK(PSMtag_5plex-0)"
            ],
            "coeff": [100, 200, 300, 200, 100, 9999, 100, 200, 300, 200, 100],
            "Ms1_spec_id": [101, 111, 121, 131, 141, 121, 101, 111, 121, 131, 141],
            "rt": [0.1* x for x in [101, 111, 121, 131, 141, 121, 101, 111, 121, 131, 141]]
            })
        
        if test_timeplex:
            base_df["time_channel"] = [1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1]

        fdc = base_df.copy(deep=True)
        dc = base_df.copy(deep=True)

        fdc["mz"] = [626.31433, 626.31433, 626.31433, 626.31433, 626.31433, 417.87622, 628.31433, 630.31433, 634.31433, 634.31433, 560.3]
        fdc["untag_seq"] = [
                "AAAAADLANR",
                "AAAAADLANR",
                "AAAAADLANR",
                "AAAAADLANR",
                "AAAAADLANR",
                "AAAAADLANR",
                "AAAAADLANR",
                "AAAAADLANR",
                "AAAAADLANR",
                "AAAAADLANR",
                "PEPTIDEK"]

        dc["spec_id"] = [106, 116, 126, 136, 146, 126, 106, 116, 126, 136, 146]
        dc["frag_errors"] = [
            "0.1;0.2;0.15",
            "0.2;0.25;0.3",
            "0.05;0.1;0.2",
            "0.2;0.15;0.25",
            "0.1;0.05;0.2",
            "0.3;0.35;0.25",
            "0.1;0.2",
            "0.05;0.1",
            "0.1;0.2;0.3",
            "0.15;0.2",
            "0.05;0.1"
        ]
        dc["rt_error"] = [0.01, 0.02, -0.01, 0.02, 0.01, 0.05, -0.01, 0.02, 0.03, -0.01, 0.0]
        dc["mz_error"] = [0.01, -0.02, 0.0, 0.01, -0.01, 0.03, 0.0, 0.01, -0.02, 0.0, 0.0]

        return fake_spec_file, fdc, dc

    def test_no_timeplex(self):
        all_spectra, filtered_decoy_coeffs, decoy_coeffs = self.get_spectra_and_dfs_for_testing()
        mz_ppm = 1e-6
        rt_tol = 0.05
        tag = read_json_to_massTag("tests/MassTags/", "PSMtag_5plex.json")
        SILAC = None
        timeplex = False
        num_iso = 6
        num_iso_r = 2
        additional_scans = 0

        all_group_pearson, all_ms1, all_coeff, all_iso, all_group_keys, all_fitted, new_output_dict, fake_fdc_dict = ms1_cor_channels(all_spectra, filtered_decoy_coeffs, decoy_coeffs, mz_ppm, rt_tol, tag, SILAC, timeplex, num_iso, num_iso_r, additional_scans)

        assert list(decoy_coeffs["untag_seq"]) == ['AAAAADLANR', 'AAAAADLANR', 'AAAAADLANR', 'AAAAADLANR', 'AAAAADLANR', 'AAAAADLANR', 'AAAAADLANR', 'AAAAADLANR', 'AAAAADLANR', 'AAAAADLANR', 'PEPTIDEK']
        assert list(decoy_coeffs["untag_prec"]) == ['AAAAADLANR_2', 'AAAAADLANR_2', 'AAAAADLANR_2', 'AAAAADLANR_2', 'AAAAADLANR_2', 'AAAAADLANR_3', 'AAAAADLANR_2', 'AAAAADLANR_2', 'AAAAADLANR_2', 'AAAAADLANR_2', 'PEPTIDEK_2']
        assert np.allclose(list(decoy_coeffs["med_frag_error"]), [0.05000000000000002, 0.04999999999999999, 0.1, 0.04999999999999999, 0.1, 0.09999999999999998, 0.05, 0.125, 0.09999999999999998, 0.02500000000000001, 0.125], atol=1e-6)
        assert list(decoy_coeffs["abs_rt_error"]) == [0.01, 0.02, 0.01, 0.02, 0.01, 0.05, 0.01, 0.02, 0.03, 0.01, 0.0]
        assert list(decoy_coeffs["abs_mz_error"]) == [0.01, 0.02, 0.0, 0.01, 0.01, 0.03, 0.0, 0.01, 0.02, 0.0, 0.0]
        group_pearson_out = [[[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]], [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]], [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]]]
        all_ms1_out = [[[{101: 1500.0, 111: 1500.0, 121: 1500.0, 131: 1500.0, 141: 1500.0}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}], [{101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}], [{101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}], [{101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}], [{101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}]], [[{101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}], [{101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}], [{101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}], [{101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}], [{101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}]], [[{101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}], [{101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}], [{101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}], [{101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}], [{101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}]]]
        all_coeff_out = [[{101: np.nan, 111: 150.0, 121: 250.0, 131: 250.0, 141: 150.0}, {101: np.nan, 111: np.nan, 121: np.nan, 131: np.nan, 141: np.nan}, {101: np.nan, 111: np.nan, 121: np.nan, 131: np.nan, 141: np.nan}, {101: np.nan, 111: np.nan, 121: np.nan, 131: 250.0, 141: np.nan}, {101: np.nan, 111: np.nan, 121: np.nan, 131: np.nan, 141: np.nan}], [{101: np.nan, 111: np.nan, 121: np.nan, 131: np.nan, 141: np.nan}, {101: np.nan, 111: np.nan, 121: np.nan, 131: np.nan, 141: np.nan}, {101: np.nan, 111: np.nan, 121: np.nan, 131: np.nan, 141: np.nan}, {101: np.nan, 111: np.nan, 121: np.nan, 131: np.nan, 141: np.nan}, {101: np.nan, 111: np.nan, 121: np.nan, 131: np.nan, 141: np.nan}], [{101: np.nan, 111: np.nan, 121: np.nan, 131: np.nan, 141: np.nan}, {101: np.nan, 111: np.nan, 121: np.nan, 131: np.nan, 141: np.nan}, {101: np.nan, 111: np.nan, 121: np.nan, 131: np.nan, 141: np.nan}, {101: np.nan, 111: np.nan, 121: np.nan, 131: np.nan, 141: np.nan}, {101: np.nan, 111: np.nan, 121: np.nan, 131: np.nan, 141: np.nan}]]
        all_iso_out = [[[PearsonRResult(statistic=0.7867351937891747, pvalue=0.06337297444937429), [0.49102465892481517, 0.33391659224774267, 0.12903065384183116, 0.03627745516676849, 0.008187980120663317, 0.0015626596981793016], [1500.0, 0.001, 0.001, 0.001, 0.001, 0.001]], [PearsonRResult(statistic=np.nan, pvalue=1), [0.5029008133959499, 0.33092281183441424, 0.12377061960678473, 0.033686532632517606, 0.007359805270184388, 0.0013594172601491194], [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]], [PearsonRResult(statistic=np.nan, pvalue=1), [0.5350960061854384, 0.3175872635736647, 0.11137947517388755, 0.028818144103447473, 0.0060415378723211255, 0.0010775730912408093], [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]], [PearsonRResult(statistic=np.nan, pvalue=1), [0.5546465564478869, 0.309167791034079, 0.10402739294370679, 0.02596905922069484, 0.005275150307925415, 0.0009140500457070699], [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]], [PearsonRResult(statistic=np.nan, pvalue=1), [0.5640507087431275, 0.30603367186170966, 0.10006209773255949, 0.0242639967330577, 0.004784990258982495, 0.0008045346705631915], [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]]], [[PearsonRResult(statistic=np.nan, pvalue=1), [0.49102465892481517, 0.33391659224774267, 0.12903065384183116, 0.03627745516676849, 0.008187980120663317, 0.0015626596981793016], [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]], [PearsonRResult(statistic=np.nan, pvalue=1), [0.5029008133959499, 0.33092281183441424, 0.12377061960678473, 0.033686532632517606, 0.007359805270184388, 0.0013594172601491194], [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]], [PearsonRResult(statistic=np.nan, pvalue=1), [0.5350960061854384, 0.3175872635736647, 0.11137947517388755, 0.028818144103447473, 0.0060415378723211255, 0.0010775730912408093], [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]], [PearsonRResult(statistic=np.nan, pvalue=1), [0.5546465564478869, 0.309167791034079, 0.10402739294370679, 0.02596905922069484, 0.005275150307925415, 0.0009140500457070699], [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]], [PearsonRResult(statistic=np.nan, pvalue=1), [0.5640507087431275, 0.30603367186170966, 0.10006209773255949, 0.0242639967330577, 0.004784990258982495, 0.0008045346705631915], [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]]], [[PearsonRResult(statistic=np.nan, pvalue=1), [0.39507620528511905, 0.3512325726627763, 0.17219722005062268, 0.0605595509130631, 0.01694736169978692, 0.00398708938863182], [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]], [PearsonRResult(statistic=np.nan, pvalue=1), [0.414363030153363, 0.35013680485679677, 0.16318127391223858, 0.05455887818878705, 0.014514399295965628, 0.0032456135928489605], [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]], [PearsonRResult(statistic=np.nan, pvalue=1), [0.46904177829954585, 0.33582135179986045, 0.13973706503559785, 0.04266923603080598, 0.010525047187108276, 0.00220552164708163], [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]], [PearsonRResult(statistic=np.nan, pvalue=1), [0.5039089626278065, 0.32440260823695743, 0.12557265897855013, 0.03602764939007652, 0.008411800357601374, 0.001676320409008026], [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]], [PearsonRResult(statistic=np.nan, pvalue=1), [0.5211244218188595, 0.3200080979442071, 0.11805483136401036, 0.03227415201992785, 0.007176930251734072, 0.0013615666012610549], [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]]]]  
        all_group_keys_out = [[('A(PSMtag_5plex-0)AAAADLANR', 2), ('A(PSMtag_5plex-4)AAAADLANR', 2), ('A(PSMtag_5plex-8)AAAADLANR', 2), ('A(PSMtag_5plex-12)AAAADLANR', 2), ('A(PSMtag_5plex-16)AAAADLANR', 2)], [('A(PSMtag_5plex-0)AAAADLANR', 3), ('A(PSMtag_5plex-4)AAAADLANR', 3), ('A(PSMtag_5plex-8)AAAADLANR', 3), ('A(PSMtag_5plex-12)AAAADLANR', 3), ('A(PSMtag_5plex-16)AAAADLANR', 3)], [('P(PSMtag_5plex-0)EPTIDEK(PSMtag_5plex-0)', 2), ('P(PSMtag_5plex-4)EPTIDEK(PSMtag_5plex-4)', 2), ('P(PSMtag_5plex-8)EPTIDEK(PSMtag_5plex-8)', 2), ('P(PSMtag_5plex-12)EPTIDEK(PSMtag_5plex-12)', 2), ('P(PSMtag_5plex-16)EPTIDEK(PSMtag_5plex-16)', 2)]]
        all_fitted_out = [[np.array([[1472.59946548,    0.        ,    0.        ,    0.        ,
           0.        ],
       [1472.59946548,    0.        ,    0.        ,    0.        ,
           0.        ],
       [1472.59946548,    0.        ,    0.        ,    0.        ,
           0.        ],
       [1472.59946548,    0.        ,    0.        ,    0.        ,
           0.        ],
       [1472.59946548,    0.        ,    0.        ,    0.        ,
           0.        ]]), [np.array([1500.,    0.,    0.,    0.,    0.,    0.]), np.array([1500.,    0.,    0.,    0.,    0.,    0.]), np.array([1500.,    0.,    0.,    0.,    0.,    0.]), np.array([1500.,    0.,    0.,    0.,    0.,    0.]), np.array([1500.,    0.,    0.,    0.,    0.,    0.])], [np.array([[0.49102466, 0.        , 0.        , 0.        , 0.        ],
       [0.50897534, 0.        , 0.        , 0.        , 0.        ],
       [0.        , 1.        , 0.        , 0.        , 0.        ],
       [0.        , 0.        , 1.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 1.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 1.        ]]), np.array([[0.49102466, 0.        , 0.        , 0.        , 0.        ],    
       [0.50897534, 0.        , 0.        , 0.        , 0.        ],
       [0.        , 1.        , 0.        , 0.        , 0.        ],
       [0.        , 0.        , 1.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 1.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 1.        ]]), np.array([[0.49102466, 0.        , 0.        , 0.        , 0.        ],    
       [0.50897534, 0.        , 0.        , 0.        , 0.        ],
       [0.        , 1.        , 0.        , 0.        , 0.        ],
       [0.        , 0.        , 1.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 1.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 1.        ]]), np.array([[0.49102466, 0.        , 0.        , 0.        , 0.        ],    
       [0.50897534, 0.        , 0.        , 0.        , 0.        ],
       [0.        , 1.        , 0.        , 0.        , 0.        ],
       [0.        , 0.        , 1.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 1.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 1.        ]]), np.array([[0.49102466, 0.        , 0.        , 0.        , 0.        ],    
       [0.50897534, 0.        , 0.        , 0.        , 0.        ],
       [0.        , 1.        , 0.        , 0.        , 0.        ],
       [0.        , 0.        , 1.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 1.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 1.        ]])], [PearsonRResult(statistic=0.6152773434275145, pvalue=0.1935455906543273), PearsonRResult(statistic=0.6152773434275145, pvalue=0.1935455906543273), PearsonRResult(statistic=0.6152773434275145, pvalue=0.1935455906543273), PearsonRResult(statistic=0.6152773434275145, pvalue=0.1935455906543273), PearsonRResult(statistic=0.6152773434275145, pvalue=0.1935455906543273)], np.array([101, 111, 121, 131, 141])], [np.array([[0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.]]), [np.array([0., 0., 0., 0., 0.]), np.array([0., 0., 0., 0., 0.]), np.array([0., 0., 0., 0., 0.]), np.array([0., 0., 0., 0., 0.]), np.array([0., 0., 0., 0., 0.])], [np.array([[1., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0.],
       [0., 0., 1., 0., 0.],
       [0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 1.]]), np.array([[1., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0.],
       [0., 0., 1., 0., 0.],
       [0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 1.]]), np.array([[1., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0.],
       [0., 0., 1., 0., 0.],
       [0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 1.]]), np.array([[1., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0.],
       [0., 0., 1., 0., 0.],
       [0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 1.]]), np.array([[1., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0.],
       [0., 0., 1., 0., 0.],
       [0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 1.]])], [PearsonRResult(statistic=np.nan, pvalue=1), PearsonRResult(statistic=np.nan, pvalue=1), PearsonRResult(statistic=np.nan, pvalue=1), PearsonRResult(statistic=np.nan, pvalue=1), PearsonRResult(statistic=np.nan, pvalue=1)], np.array([101, 111, 121, 131, 141])], [np.array([[0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.]]), [np.array([0., 0., 0., 0., 0.]), np.array([0., 0., 0., 0., 0.]), np.array([0., 0., 0., 0., 0.]), np.array([0., 0., 0., 0., 0.]), np.array([0., 0., 0., 0., 0.])], [np.array([[1., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0.],
       [0., 0., 1., 0., 0.],
       [0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 1.]]), np.array([[1., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0.],
       [0., 0., 1., 0., 0.],
       [0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 1.]]), np.array([[1., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0.],
       [0., 0., 1., 0., 0.],
       [0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 1.]]), np.array([[1., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0.],
       [0., 0., 1., 0., 0.],
       [0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 1.]]), np.array([[1., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0.],
       [0., 0., 1., 0., 0.],
       [0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 1.]])], [PearsonRResult(statistic=np.nan, pvalue=1), PearsonRResult(statistic=np.nan, pvalue=1), PearsonRResult(statistic=np.nan, pvalue=1), PearsonRResult(statistic=np.nan, pvalue=1), PearsonRResult(statistic=np.nan, pvalue=1)], np.array([101, 111, 121, 131, 141])]]
        new_output_dict_out = {}
        fake_fdc_dict_out = {}
        self.compare_pearsons(all_group_pearson, group_pearson_out)
        self.compare_ms1(all_ms1, all_ms1_out)
        self.compare_coeff(all_coeff, all_coeff_out)
        self.compare_iso(all_iso, all_iso_out)
        self.compare_group_keys(all_group_keys, all_group_keys_out)
        self.compare_fitted(all_fitted, all_fitted_out)
        self.compare_output_dict(new_output_dict, new_output_dict_out)
        self.compare_fake_fdc_dict(fake_fdc_dict, fake_fdc_dict_out)



    def test_timeplex(self):
        all_spectra, filtered_decoy_coeffs, decoy_coeffs = self.get_spectra_and_dfs_for_testing(test_timeplex=True)
        mz_ppm = 1e-6
        rt_tol = 0.05
        tag = read_json_to_massTag("tests/MassTags/", "PSMtag_5plex.json")
        SILAC = None
        timeplex = True
        num_iso = 6
        num_iso_r = 2
        additional_scans = 0

        all_group_pearson, all_ms1, all_coeff, all_iso, all_group_keys, all_fitted, new_output_dict, fake_fdc_dict = ms1_cor_channels(all_spectra, filtered_decoy_coeffs, decoy_coeffs, mz_ppm, rt_tol, tag, SILAC, timeplex, num_iso, num_iso_r, additional_scans)

        assert list(decoy_coeffs["untag_seq"]) == ['AAAAADLANR', 'AAAAADLANR', 'AAAAADLANR', 'AAAAADLANR', 'AAAAADLANR', 'AAAAADLANR', 'AAAAADLANR', 'AAAAADLANR', 'AAAAADLANR', 'AAAAADLANR', 'PEPTIDEK']
        assert list(decoy_coeffs["untag_prec"]) == ['AAAAADLANR_2', 'AAAAADLANR_2', 'AAAAADLANR_2', 'AAAAADLANR_2', 'AAAAADLANR_2', 'AAAAADLANR_3', 'AAAAADLANR_2', 'AAAAADLANR_2', 'AAAAADLANR_2', 'AAAAADLANR_2', 'PEPTIDEK_2']
        assert np.allclose(list(decoy_coeffs["med_frag_error"]), [0.05000000000000002, 0.04999999999999999, 0.1, 0.04999999999999999, 0.1, 0.09999999999999998, 0.05, 0.125, 0.09999999999999998, 0.02500000000000001, 0.125], atol=1e-6)
        assert list(decoy_coeffs["abs_rt_error"]) == [0.01, 0.02, 0.01, 0.02, 0.01, 0.05, 0.01, 0.02, 0.03, 0.01, 0.0]
        assert list(decoy_coeffs["abs_mz_error"]) == [0.01, 0.02, 0.0, 0.01, 0.01, 0.03, 0.0, 0.01, 0.02, 0.0, 0.0]
        group_pearson_out = [[[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]], [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]], [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]], [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]]]
        all_ms1_out = [[[{101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}], [{101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}], [{101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}], [{101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}], [{101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}]], [[{101: 1500.0, 111: 1500.0, 121: 1500.0, 131: 1500.0, 141: 1500.0}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}], [{101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}], [{101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}], [{101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}], [{101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}]], [[{101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}], [{101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}], [{101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}], [{101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}], [{101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}]], [[{101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}], [{101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}], [{101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}], [{101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}], [{101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}, {101: 0.001, 111: 0.001, 121: 0.001, 131: 0.001, 141: 0.001}]]]
        all_coeff_out = [[{101: np.nan, 111: 150.0, 121: 100.0005, 131: 100.0005, 141: 150.0}, {101: np.nan, 111: np.nan, 121: np.nan, 131: np.nan, 141: np.nan}, {101: np.nan, 111: np.nan, 121: np.nan, 131: np.nan, 141: np.nan}, {101: np.nan, 111: np.nan, 121: np.nan, 131: 88.88944444444445, 141: np.nan}, {101: np.nan, 111: np.nan, 121: np.nan, 131: np.nan, 141: np.nan}], [{101: np.nan, 111: np.nan, 121: np.nan, 131: np.nan, 141: np.nan}, {101: np.nan, 111: np.nan, 121: np.nan, 131: np.nan, 141: np.nan}, {101: np.nan, 111: np.nan, 121: np.nan, 131: np.nan, 141: np.nan}, {101: np.nan, 111: np.nan, 121: np.nan, 131: np.nan, 141: np.nan}, {101: np.nan, 111: np.nan, 121: np.nan, 131: np.nan, 141: np.nan}], [{101: np.nan, 111: np.nan, 121: np.nan, 131: np.nan, 141: np.nan}, {101: np.nan, 111: np.nan, 121: np.nan, 131: np.nan, 141: np.nan}, {101: np.nan, 111: np.nan, 121: np.nan, 131: np.nan, 141: np.nan}, {101: np.nan, 111: np.nan, 121: np.nan, 131: np.nan, 141: np.nan}, {101: np.nan, 111: np.nan, 121: np.nan, 131: np.nan, 141: np.nan}], [{101: np.nan, 111: np.nan, 121: np.nan, 131: np.nan, 141: np.nan}, {101: np.nan, 111: np.nan, 121: np.nan, 131: np.nan, 141: np.nan}, {101: np.nan, 111: np.nan, 121: np.nan, 131: np.nan, 141: np.nan}, {101: np.nan, 111: np.nan, 121: np.nan, 131: np.nan, 141: np.nan}, {101: np.nan, 111: np.nan, 121: np.nan, 131: np.nan, 141: np.nan}]]
        all_iso_out = [[[PearsonRResult(statistic=np.nan, pvalue=1), [0.49102465892481517, 0.33391659224774267, 0.12903065384183116, 0.03627745516676849, 0.008187980120663317, 0.0015626596981793016], [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]], [PearsonRResult(statistic=np.nan, pvalue=1), [0.5029008133959499, 0.33092281183441424, 0.12377061960678473, 0.033686532632517606, 0.007359805270184388, 0.0013594172601491194], [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]], [PearsonRResult(statistic=np.nan, pvalue=1), [0.5350960061854384, 0.3175872635736647, 0.11137947517388755, 0.028818144103447473, 0.0060415378723211255, 0.0010775730912408093], [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]], [PearsonRResult(statistic=np.nan, pvalue=1), [0.5546465564478869, 0.309167791034079, 0.10402739294370679, 0.02596905922069484, 0.005275150307925415, 0.0009140500457070699], [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]], [PearsonRResult(statistic=np.nan, pvalue=1), [0.5640507087431275, 0.30603367186170966, 0.10006209773255949, 0.0242639967330577, 0.004784990258982495, 0.0008045346705631915], [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]]], [[PearsonRResult(statistic=0.7867351937891747, pvalue=0.06337297444937429), [0.49102465892481517, 0.33391659224774267, 0.12903065384183116, 0.03627745516676849, 0.008187980120663317, 0.0015626596981793016], [1500.0, 0.001, 0.001, 0.001, 0.001, 0.001]], [PearsonRResult(statistic=np.nan, pvalue=1), [0.5029008133959499, 0.33092281183441424, 0.12377061960678473, 0.033686532632517606, 0.007359805270184388, 0.0013594172601491194], [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]], [PearsonRResult(statistic=np.nan, pvalue=1), [0.5350960061854384, 0.3175872635736647, 0.11137947517388755, 0.028818144103447473, 0.0060415378723211255, 0.0010775730912408093], [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]], [PearsonRResult(statistic=np.nan, pvalue=1), [0.5546465564478869, 0.309167791034079, 0.10402739294370679, 0.02596905922069484, 0.005275150307925415, 0.0009140500457070699], [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]], [PearsonRResult(statistic=np.nan, pvalue=1), [0.5640507087431275, 0.30603367186170966, 0.10006209773255949, 0.0242639967330577, 0.004784990258982495, 0.0008045346705631915], [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]]], [[PearsonRResult(statistic=np.nan, pvalue=1), [0.49102465892481517, 0.33391659224774267, 0.12903065384183116, 0.03627745516676849, 0.008187980120663317, 0.0015626596981793016], [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]], [PearsonRResult(statistic=np.nan, pvalue=1), [0.5029008133959499, 0.33092281183441424, 0.12377061960678473, 0.033686532632517606, 0.007359805270184388, 0.0013594172601491194], [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]], [PearsonRResult(statistic=np.nan, pvalue=1), [0.5350960061854384, 0.3175872635736647, 0.11137947517388755, 0.028818144103447473, 0.0060415378723211255, 0.0010775730912408093], [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]], [PearsonRResult(statistic=np.nan, pvalue=1), [0.5546465564478869, 0.309167791034079, 0.10402739294370679, 0.02596905922069484, 0.005275150307925415, 0.0009140500457070699], [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]], [PearsonRResult(statistic=np.nan, pvalue=1), [0.5640507087431275, 0.30603367186170966, 0.10006209773255949, 0.0242639967330577, 0.004784990258982495, 0.0008045346705631915], [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]]], [[PearsonRResult(statistic=np.nan, pvalue=1), [0.39507620528511905, 0.3512325726627763, 0.17219722005062268, 0.0605595509130631, 0.01694736169978692, 0.00398708938863182], [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]], [PearsonRResult(statistic=np.nan, pvalue=1), [0.414363030153363, 0.35013680485679677, 0.16318127391223858, 0.05455887818878705, 0.014514399295965628, 0.0032456135928489605], [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]], [PearsonRResult(statistic=np.nan, pvalue=1), [0.46904177829954585, 0.33582135179986045, 0.13973706503559785, 0.04266923603080598, 0.010525047187108276, 0.00220552164708163], [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]], [PearsonRResult(statistic=np.nan, pvalue=1), [0.5039089626278065, 0.32440260823695743, 0.12557265897855013, 0.03602764939007652, 0.008411800357601374, 0.001676320409008026], [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]], [PearsonRResult(statistic=np.nan, pvalue=1), [0.5211244218188595, 0.3200080979442071, 0.11805483136401036, 0.03227415201992785, 0.007176930251734072, 0.0013615666012610549], [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]]]]
        all_group_keys_out = [[('A(PSMtag_5plex-0)AAAADLANR', 2, 1), ('A(PSMtag_5plex-4)AAAADLANR', 2, 1), ('A(PSMtag_5plex-8)AAAADLANR', 2, 1), ('A(PSMtag_5plex-12)AAAADLANR', 2, 1), ('A(PSMtag_5plex-16)AAAADLANR', 2, 1)], [('A(PSMtag_5plex-0)AAAADLANR', 2, 2), ('A(PSMtag_5plex-4)AAAADLANR', 2, 2), ('A(PSMtag_5plex-8)AAAADLANR', 2, 2), ('A(PSMtag_5plex-12)AAAADLANR', 2, 2), ('A(PSMtag_5plex-16)AAAADLANR', 2, 2)], [('A(PSMtag_5plex-0)AAAADLANR', 3, 1), ('A(PSMtag_5plex-4)AAAADLANR', 3, 1), ('A(PSMtag_5plex-8)AAAADLANR', 3, 1), ('A(PSMtag_5plex-12)AAAADLANR', 3, 1), ('A(PSMtag_5plex-16)AAAADLANR', 3, 1)], [('P(PSMtag_5plex-0)EPTIDEK(PSMtag_5plex-0)', 2, 1), ('P(PSMtag_5plex-4)EPTIDEK(PSMtag_5plex-4)', 2, 1), ('P(PSMtag_5plex-8)EPTIDEK(PSMtag_5plex-8)', 2, 1), ('P(PSMtag_5plex-12)EPTIDEK(PSMtag_5plex-12)', 2, 1), ('P(PSMtag_5plex-16)EPTIDEK(PSMtag_5plex-16)', 2, 1)]]
        all_fitted_out = [[np.array([[0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.]]), [np.array([0., 0., 0., 0., 0.]), np.array([0., 0., 0., 0., 0.]), np.array([0., 0., 0., 0., 0.]), np.array([0., 0., 0., 0., 0.]), np.array([0., 0., 0., 0., 0.])], [np.array([[1., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0.],
       [0., 0., 1., 0., 0.],
       [0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 1.]]), np.array([[1., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0.],
       [0., 0., 1., 0., 0.],
       [0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 1.]]), np.array([[1., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0.],
       [0., 0., 1., 0., 0.],
       [0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 1.]]), np.array([[1., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0.],
       [0., 0., 1., 0., 0.],
       [0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 1.]]), np.array([[1., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0.],
       [0., 0., 1., 0., 0.],
       [0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 1.]])], [PearsonRResult(statistic=np.nan, pvalue=1), PearsonRResult(statistic=np.nan, pvalue=1), PearsonRResult(statistic=np.nan, pvalue=1), PearsonRResult(statistic=np.nan, pvalue=1), PearsonRResult(statistic=np.nan, pvalue=1)], np.array([101, 111, 121, 131, 141])], [np.array([[1472.59946548,    0.        ,    0.        ,    0.        ,
           0.        ],
       [1472.59946548,    0.        ,    0.        ,    0.        ,
           0.        ],
       [1472.59946548,    0.        ,    0.        ,    0.        ,
           0.        ],
       [1472.59946548,    0.        ,    0.        ,    0.        ,
           0.        ],
       [1472.59946548,    0.        ,    0.        ,    0.        ,
           0.        ]]), [np.array([1500.,    0.,    0.,    0.,    0.,    0.]), np.array([1500.,    0.,    0.,    0.,    0.,    0.]), np.array([1500.,    0.,    0.,    0.,    0.,    0.]), np.array([1500.,    0.,    0.,    0.,    0.,    0.]), np.array([1500.,    0.,    0.,    0.,    0.,    0.])], [np.array([[0.49102466, 0.        , 0.        , 0.        , 0.        ],
       [0.50897534, 0.        , 0.        , 0.        , 0.        ],
       [0.        , 1.        , 0.        , 0.        , 0.        ],
       [0.        , 0.        , 1.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 1.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 1.        ]]), np.array([[0.49102466, 0.        , 0.        , 0.        , 0.        ],    
       [0.50897534, 0.        , 0.        , 0.        , 0.        ],
       [0.        , 1.        , 0.        , 0.        , 0.        ],
       [0.        , 0.        , 1.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 1.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 1.        ]]), np.array([[0.49102466, 0.        , 0.        , 0.        , 0.        ],    
       [0.50897534, 0.        , 0.        , 0.        , 0.        ],
       [0.        , 1.        , 0.        , 0.        , 0.        ],
       [0.        , 0.        , 1.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 1.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 1.        ]]), np.array([[0.49102466, 0.        , 0.        , 0.        , 0.        ],    
       [0.50897534, 0.        , 0.        , 0.        , 0.        ],
       [0.        , 1.        , 0.        , 0.        , 0.        ],
       [0.        , 0.        , 1.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 1.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 1.        ]]), np.array([[0.49102466, 0.        , 0.        , 0.        , 0.        ],    
       [0.50897534, 0.        , 0.        , 0.        , 0.        ],
       [0.        , 1.        , 0.        , 0.        , 0.        ],
       [0.        , 0.        , 1.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 1.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 1.        ]])], [PearsonRResult(statistic=0.6152773434275145, pvalue=0.1935455906543273), PearsonRResult(statistic=0.6152773434275145, pvalue=0.1935455906543273), PearsonRResult(statistic=0.6152773434275145, pvalue=0.1935455906543273), PearsonRResult(statistic=0.6152773434275145, pvalue=0.1935455906543273), PearsonRResult(statistic=0.6152773434275145, pvalue=0.1935455906543273)], np.array([101, 111, 121, 131, 141])], [np.array([[0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.]]), [np.array([0., 0., 0., 0., 0.]), np.array([0., 0., 0., 0., 0.]), np.array([0., 0., 0., 0., 0.]), np.array([0., 0., 0., 0., 0.]), np.array([0., 0., 0., 0., 0.])], [np.array([[1., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0.],
       [0., 0., 1., 0., 0.],
       [0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 1.]]), np.array([[1., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0.],
       [0., 0., 1., 0., 0.],
       [0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 1.]]), np.array([[1., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0.],
       [0., 0., 1., 0., 0.],
       [0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 1.]]), np.array([[1., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0.],
       [0., 0., 1., 0., 0.],
       [0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 1.]]), np.array([[1., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0.],
       [0., 0., 1., 0., 0.],
       [0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 1.]])], [PearsonRResult(statistic=np.nan, pvalue=1), PearsonRResult(statistic=np.nan, pvalue=1), PearsonRResult(statistic=np.nan, pvalue=1), PearsonRResult(statistic=np.nan, pvalue=1), PearsonRResult(statistic=np.nan, pvalue=1)], np.array([101, 111, 121, 131, 141])], [np.array([[0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0.]]), [np.array([0., 0., 0., 0., 0.]), np.array([0., 0., 0., 0., 0.]), np.array([0., 0., 0., 0., 0.]), np.array([0., 0., 0., 0., 0.]), np.array([0., 0., 0., 0., 0.])], [np.array([[1., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0.],
       [0., 0., 1., 0., 0.],
       [0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 1.]]), np.array([[1., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0.],
       [0., 0., 1., 0., 0.],
       [0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 1.]]), np.array([[1., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0.],
       [0., 0., 1., 0., 0.],
       [0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 1.]]), np.array([[1., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0.],
       [0., 0., 1., 0., 0.],
       [0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 1.]]), np.array([[1., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0.],
       [0., 0., 1., 0., 0.],
       [0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 1.]])], [PearsonRResult(statistic=np.nan, pvalue=1), PearsonRResult(statistic=np.nan, pvalue=1), PearsonRResult(statistic=np.nan, pvalue=1), PearsonRResult(statistic=np.nan, pvalue=1), PearsonRResult(statistic=np.nan, pvalue=1)], np.array([101, 111, 121, 131, 141])]]
        new_output_dict_out = {}
        fake_fdc_dict_out = {}
        self.compare_pearsons(all_group_pearson, group_pearson_out)
        self.compare_ms1(all_ms1, all_ms1_out)
        self.compare_coeff(all_coeff, all_coeff_out)
        self.compare_iso(all_iso, all_iso_out)
        self.compare_group_keys(all_group_keys, all_group_keys_out)
        self.compare_fitted(all_fitted, all_fitted_out)
        self.compare_output_dict(new_output_dict, new_output_dict_out)
        self.compare_fake_fdc_dict(fake_fdc_dict, fake_fdc_dict_out)


class Test_get_matrix_to_fit_numba():  #this is tested more thoroughly in the tests for fit_channel_isotopes_numba

    def test_very_quick(self):
        ms1_iso_patterns = np.array([[[3.10000000e+01, 5.47156642e-01],
                                    [3.15000000e+01, 3.31610086e-01],
                                    [3.20000000e+01, 9.88131065e-02],
                                    [3.25000000e+01, 1.92968356e-02],
                                    [3.30000000e+01, 2.77757482e-03]],

                                [[3.30000000e+01, 5.69601202e-01],
                                    [3.35000000e+01, 3.22198660e-01],
                                    [3.40000000e+01, 8.94996278e-02],
                                    [3.45000000e+01, 1.62726596e-02],
                                    [3.50000000e+01, 2.17790646e-03]],

                                [[3.50000000e+01, 5.92966446e-01],
                                    [3.55000000e+01, 3.11457123e-01],
                                    [3.60000000e+01, 8.02238045e-02],
                                    [3.65000000e+01, 1.35056910e-02],
                                    [3.70000000e+01, 1.67115873e-03]],

                                [[3.70000000e+01, 6.17290141e-01],
                                    [3.75000000e+01, 2.99292190e-01],
                                    [3.80000000e+01, 7.10441056e-02],
                                    [3.85000000e+01, 1.10034642e-02],
                                    [3.90000000e+01, 1.25039366e-03]],

                                [[3.90000000e+01, 6.42611602e-01],
                                    [3.95000000e+01, 2.85605156e-01],
                                    [4.00000000e+01, 6.20253623e-02],
                                    [4.05000000e+01, 8.77126335e-03],
                                    [4.10000000e+01, 9.08135852e-04]]])
        group_lengths = np.array([5, 5, 5, 5, 5])
        dia_spectrum = np.array([[3.10000000e+01, 6.84074722e-02],
                                [3.15000000e+01, 4.14590741e-02],
                                [3.20000000e+01, 1.23539665e-02],
                                [3.25000000e+01, 2.41255912e-03],
                                [3.30000000e+01, 1.42774404e-01],
                                [3.35000000e+01, 8.05648481e-02],
                                [3.40000000e+01, 2.23791245e-02],
                                [3.45000000e+01, 4.06893172e-03],
                                [3.50000000e+01, 1.11746745e-01],
                                [3.55000000e+01, 5.84092183e-02],
                                [3.60000000e+01, 1.50447987e-02],
                                [3.65000000e+01, 2.53279439e-03],
                                [3.70000000e+01, 2.31840837e-01],
                                [3.75000000e+01, 1.12255727e-01],
                                [3.80000000e+01, 2.66465614e-02],
                                [3.85000000e+01, 4.12707684e-03],
                                [3.90000000e+01, 4.06397816e-02],
                                [3.95000000e+01, 1.78536870e-02],
                                [4.00000000e+01, 3.87731585e-03],
                                [4.05000000e+01, 5.48307292e-04],
                                [4.10000000e+01, 5.67691894e-05]])
        len_all_iso = 5
        mz_ppm = 1e-6
        dense_matrix, dia_spec_int = get_matrix_to_fit_numba(ms1_iso_patterns, group_lengths, dia_spectrum, len_all_iso, mz_ppm)
        dense_matrix_expected = np.array([[0.54715664, 0.        , 0.        , 0.        , 0.        ],
                                            [0.33161009, 0.        , 0.        , 0.        , 0.        ],
                                            [0.09881311, 0.        , 0.        , 0.        , 0.        ],
                                            [0.01929684, 0.        , 0.        , 0.        , 0.        ],
                                            [0.00277757, 0.5696012 , 0.        , 0.        , 0.        ],
                                            [0.        , 0.32219866, 0.        , 0.        , 0.        ],
                                            [0.        , 0.08949963, 0.        , 0.        , 0.        ],
                                            [0.        , 0.01627266, 0.        , 0.        , 0.        ],
                                            [0.        , 0.00217791, 0.59296645, 0.        , 0.        ],
                                            [0.        , 0.        , 0.31145712, 0.        , 0.        ],
                                            [0.        , 0.        , 0.0802238 , 0.        , 0.        ],
                                            [0.        , 0.        , 0.01350569, 0.        , 0.        ],
                                            [0.        , 0.        , 0.00167116, 0.61729014, 0.        ],
                                            [0.        , 0.        , 0.        , 0.29929219, 0.        ],
                                            [0.        , 0.        , 0.        , 0.07104411, 0.        ],
                                            [0.        , 0.        , 0.        , 0.01100346, 0.        ],
                                            [0.        , 0.        , 0.        , 0.00125039, 0.6426116 ],
                                            [0.        , 0.        , 0.        , 0.        , 0.28560516],
                                            [0.        , 0.        , 0.        , 0.        , 0.06202536],
                                            [0.        , 0.        , 0.        , 0.        , 0.00877126],
                                            [0.        , 0.        , 0.        , 0.        , 0.00090814],
                                            [0.        , 0.        , 0.        , 0.        , 0.        ],
                                            [0.        , 0.        , 0.        , 0.        , 0.        ],
                                            [0.        , 0.        , 0.        , 0.        , 0.        ],
                                            [0.        , 0.        , 0.        , 0.        , 0.        ],
                                            [0.        , 0.        , 0.        , 0.        , 0.        ]])
        dia_spec_int_expected = np.array([6.84074722e-02, 4.14590741e-02, 1.23539665e-02, 2.41255912e-03,
                                        1.42774404e-01, 8.05648481e-02, 2.23791245e-02, 4.06893172e-03,
                                        1.11746745e-01, 5.84092183e-02, 1.50447987e-02, 2.53279439e-03,
                                        2.31840837e-01, 1.12255727e-01, 2.66465614e-02, 4.12707684e-03,
                                        4.06397816e-02, 1.78536870e-02, 3.87731585e-03, 5.48307292e-04,
                                        5.67691894e-05, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                        0.00000000e+00, 0.00000000e+00])
        
        assert np.allclose(dense_matrix, dense_matrix_expected, atol=1e-6)
        assert np.allclose(dia_spec_int, dia_spec_int_expected, atol=1e-6)





def main():
    
    instance = Test_fit_channel_isotopes_numba()
    instance.test_5plex_no_noise_or_missed_peaks()



if __name__ == "__main__":
    main()

