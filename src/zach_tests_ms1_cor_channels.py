import pytest
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.ms1_cor_channels import get_seqs_and_mzs, get_other_channels, minmax_spec_window, fit_mTRAQ_isotopes, fit_channel_isotopes_numba
from src.mass_tags import massTag, read_json_to_massTag
from src.utils.io.load_files import SpectrumFile, Spectrum
from brainpy._c.isotopic_distribution import TheoreticalPeak

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

    


import pandas as pd
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







    


def main():
    normalized_mz_intensity_dict, per_channel_iso_intensity_dict, z = Test_fit_channel_isotopes_numba.fivePlex_fourDa_spacing()

    def move_peaks_above(peak_num, cutoff):
        if peak_num > cutoff:
            return 0.01
        else:
            return 0
                
    dia_spec = Spectrum()
    dia_spec.mz = np.array([key + move_peaks_above(i, 6) for i, key in enumerate(normalized_mz_intensity_dict.keys())])
    dia_spec.intens = np.array(list(normalized_mz_intensity_dict.values()))

    all_iso = [[TheoreticalPeak(mz=mz, intensity=i, charge=z) for mz, i in per_channel_iso_intensity_dict[channel].items()] for channel in per_channel_iso_intensity_dict.keys()]
    mz_ppm = 1e-6
    pred_coeff, obs_peaks, fit_matrix = fit_mTRAQ_isotopes(dia_spec, all_iso, mz_ppm)

    print(pred_coeff)
    print(obs_peaks)
    print(fit_matrix)


if __name__ == "__main__":
    main()

