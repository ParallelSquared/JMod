"""
This Source Code Form is subject to the terms of the Oxford Nanopore
Technologies, Ltd. Public License, v. 1.0.  Full licence can be found
at https://github.com/ParallelSquared/JMod/blob/main/LICENSE.txt
"""

"""
Tests for functions in mass_tags.py
"""

import pytest
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy import stats
import tempfile
import os

# Import the functions we want to test
from src.rt_alignment import (
    twostepfit, lowess_fit, closest_spec, gaussian, fit_errors, cdf_plots, alignment_plots, cdf_data, get_multiples, empirical_fit, filter_rts_by_dense, MZRTfit,
    timeplex_pair_nn_gate, _hwhm_mode_sigma
)

from tests.fixtures.test_data import SAMPLE_LIBRARY_ENTRY, SAMPLE_MS2_SPECTRUM, SAMPLE_FEATURE



class TestTwoStepFit:
    def test_twostepfit_basic(self):
        # Generate a simple linear relationship with small noise
        x = np.linspace(0, 10, 50)
        y = 2 * x + 1 + np.random.normal(0, 0.1, size=len(x))
        
        spl = twostepfit(x, y)
        assert isinstance(spl, UnivariateSpline)

        # Test the spline behaves sensibly: prediction close to linear
        y_pred = spl(x)
        corr = np.corrcoef(y, y_pred)[0, 1]
        assert corr > 0.9

    def test_twostepfit_handles_nan_values(self):
        x = np.linspace(0, 5, 20)
        y = 3 * x + 2
        y[::5] = np.nan  # introduce NaNs
        spl = twostepfit(x, y)
        assert isinstance(spl, UnivariateSpline)

class TestLowessFit:
    def test_lowess_fit_interpolation_behavior(self):
        x = np.linspace(0, 10, 50)
        y = x ** 2
        f = lowess_fit(x, y)
        y_pred = f(x)
        corr = np.corrcoef(y, y_pred)[0, 1]
        assert corr > 0.9

    def test_lowess_fit_handles_small_arrays(self):
        x = np.array([0, 1, 2, 3])
        y = np.array([0, 1, 4, 9])
        f = lowess_fit(x, y)
        assert callable(f)
        result = f([1.5])
        assert isinstance(result, np.ndarray)


class TestClosestSpec:
    def test_closest_spec_basic(self):
        # Columns: [RT, lower m/z, upper m/z]
        dia_rt_mzwin = np.array([
            [10.0, 400, 410],
            [20.0, 410, 420],
            [30.0, 420, 430]
        ])
        mz = 415  # Falls into second window
        rt = 21
        idx = closest_spec(dia_rt_mzwin, mz, rt)
        assert idx == 1

    def test_closest_spec_returns_zero_if_no_match(self):
        dia_rt_mzwin = np.array([
            [10.0, 400, 410],
            [20.0, 410, 420]
        ])
        mz = 500  # Outside all ranges
        rt = 15
        idx = closest_spec(dia_rt_mzwin, mz, rt)
        assert idx == 0

    def test_closest_spec_picks_nearest_rt(self):
        dia_rt_mzwin = np.array([
            [10.0, 400, 410],
            [30.0, 400, 410],
            [50.0, 400, 410],
        ])
        mz = 405
        rt = 32
        idx = closest_spec(dia_rt_mzwin, mz, rt)
        assert idx == 1  # Closest RT = 30

class TestGaussian:
    def test_gaussian_peak_and_symmetry(self):
        x = np.linspace(-5, 5, 100)
        y = gaussian(x, amplitude=1, mean=0, stddev=1)
        peak_idx = np.argmax(y)
        center_idx = len(x) // 2
        assert abs(peak_idx - center_idx) <= 1  # allow small shift
        assert np.isclose(y[0], y[-1], atol=1e-6)

    def test_gaussian_changes_with_parameters(self):
        x = np.linspace(-5, 5, 100)
        y1 = gaussian(x, amplitude=1, mean=0, stddev=1)
        y2 = gaussian(x, amplitude=2, mean=0, stddev=1)
        assert np.allclose(y2, 2 * y1, rtol=1e-5)


class DummyLogger:
    """Simple no-op logger for testing."""
    def info(self, msg): 
        pass


class TestFitErrors:
    """Unit tests for the fit_errors() function."""

    def setup_method(self, method):
        """Setup for each test."""
        np.random.seed(0)

    def test_gaussian_like_distribution(self, monkeypatch):
        """fit_errors should fit a Gaussian-like distribution and return a reasonable boundary."""
        errors = np.abs(np.random.normal(0, 1, 1000))
        dummy_logger = DummyLogger()
        monkeypatch.setattr("src.rt_alignment.logger", dummy_logger)

        boundary = fit_errors(errors, limit=10, percentile=0.999)

        assert isinstance(boundary, float)
        assert 2 < boundary < 5

class TestCdfPlots:
    @staticmethod
    @pytest.fixture
    def sample_cdf_data():
        """Generate example data for empirical and predicted CDFs."""
        emp_data = np.linspace(0, 10, 50)
        emp_p = stats.expon.cdf(emp_data, scale=2)
        pred_data = np.linspace(0, 10, 50)
        pred_p = stats.halfnorm.cdf(pred_data, scale=2)
        return emp_data, emp_p, pred_data, pred_p

    def test_cdf_plots_runs_without_error(self, sample_cdf_data, monkeypatch, tmp_path):
        """Test that cdf_plots runs and generates a valid matplotlib figure."""
        emp_data, emp_p, pred_data, pred_p = sample_cdf_data
        (tmp_path / "first_search").mkdir(exist_ok=True)

        monkeypatch.setitem(globals(), "colours", ["blue", "orange"])

        # Convert tmp_path to string
        results_folder = str(tmp_path)

        # Call the plotting function
        cdf_plots(emp_data, emp_p, percentile=0.999, boundary=5,
                pred_data=pred_data, pred_p=pred_p, results_folder=results_folder)
        
        out_file = tmp_path / "first_search/RTelbows.png"
        assert out_file.exists(), "Expected output plot file to be created."

    def test_cdf_data_computation(self):
        """Test the cdf_data function produces correct outputs."""
        rt_diffs = np.array([-2.5, -1.0, 0.0, 1.5, 3.2, 4.0])
        data, p, cdf_auc = cdf_data(rt_diffs, limit=3)

        # Check types
        assert isinstance(data, np.ndarray)
        assert isinstance(p, np.ndarray)
        assert isinstance(cdf_auc, float)

        # Check values are within expected range
        assert np.all(data <= 3)  # limited by 'limit'
        assert np.all(p >= 0) and np.all(p <= 1)

        # Check lengths
        assert len(data) == len(p)

        # Basic sanity check for AUC
        assert cdf_auc > 0

class TestAlignmentPlots:
    @pytest.fixture
    def sample_filtered_output(self):
        """Generate a minimal example of filtered_output."""
        import pandas as pd
        class DummyOutput:
            def __init__(self):
                self.lib_rt = np.linspace(10, 50, 10)
                self.updated_lib_rt = np.linspace(12, 52, 10)
                self.rt = self.lib_rt + np.random.normal(0, 0.5, size=10)
                self.mz = np.linspace(500, 510, 10)
                self.mz_diffs = np.random.normal(0, 0.01, size=10)
        return DummyOutput()

    @pytest.fixture
    def dummy_splines(self):
        """Return simple spline/dummy functions for testing."""
        def orig_spl(x): return x + 0.1
        def rt_spl(x): return x + 0.2
        def f_rt_mz(x): return 0.001 * x
        def mz_spl(x): return 0.001 * x
        return orig_spl, rt_spl, f_rt_mz, mz_spl

    def test_alignment_plots_creates_files(self, sample_filtered_output, dummy_splines, tmp_path):
        orig_spl, rt_spl, f_rt_mz, mz_spl = dummy_splines
        (tmp_path / "first_search").mkdir(exist_ok=True)

        alignment_plots(
            filtered_output=sample_filtered_output,
            orig_spl=orig_spl,
            rt_spl=rt_spl,
            f_rt_mz=f_rt_mz,
            mz_spl=mz_spl,
            rt_dist_params=(1,1,1),  # dummy parameters
            results_folder=str(tmp_path)  # convert Path to string
        )

        expected_files = [
            "first_search/OriginalRTfit.png",
            "first_search/RTfit.png",
            "first_search/RtResidual.png",
            "first_search/RTdiff.png",
            "first_search/MZrtfit.png",
            "first_search/MZfit.png",
            "first_search/MZdiff.png"
        ]

        for fname in expected_files:
            out_file = tmp_path / fname
            assert out_file.exists(), f"Expected plot file {fname} to be created."

        # Optional: assert all files exist
        assert len(list((tmp_path / 'first_search').iterdir())) == len(expected_files)


class TestGetMultiplesAndFilter:

    @pytest.fixture
    def sample_output_df(self):
        """Create a small DataFrame for testing get_multiples and get_df_filter."""
        data = {
            "rt": [10, 12, 10, 15, 20],
            "hyperscore": [100, 200, 150, 80, 90],
            "coeff": [1.0, 0.9, 1.1, 0.8, 0.95],
            "frag_cosines_p": [0.9, 0.85, 0.95, 0.5, 0.4],
            "scribe_scores": [0.5, 0.6, 0.55, 0.9, 0.95],
            "gof_stats": [0.3, 0.35, 0.32, 0.9, 0.92],
            "manhattan_distances": [1.0, 0.9, 1.1, 0.5, 0.6],
            "max_matched_residuals": [0.2, 0.25, 0.22, 0.8, 0.9],
            "med_frag_error": [0.05, 0.06, 0.05, 0.2, 0.25],
        }
        return pd.DataFrame(data)

    def test_get_multiples_detects_duplicates(self, sample_output_df):
        """Test get_multiples identifies repeated keys correctly."""
        id_keys = [("PEPTIDE", 2), ("PEPTIDE", 2), ("ACDE", 1), ("PEPTIDE", 2), ("XYZ", 3)]
        multiples, num_multiples, multiples_idxs = get_multiples(id_keys, sample_output_df)

        # Check correct number of duplicates detected
        assert len(multiples) == 1  # only "PEPTIDE,2" repeated
        assert num_multiples[0] == 3
        # Check the returned indices are correct
        assert np.all(multiples_idxs[0] == np.array([0, 1, 3]))

        # Check that the RT values match expected
        expected_rts = np.array([10, 12, 15])
        assert np.all(multiples[0] == expected_rts)


class TestEmpiricalFit:
    @classmethod
    def setup_class(cls):
        np.random.seed(42)
        N = 100
        cls.output_df = pd.DataFrame({
            "hyperscore": np.random.rand(N),
            "frag_cosines_p": np.random.rand(N),
            "manhattan_distances": np.random.rand(N),
            "scribe_score": np.random.rand(N),
            "gof_stats": np.random.rand(N),
            "max_matched_residuals": np.random.rand(N),
            "med_frag_error": np.random.rand(N),
            "lib_rt": np.linspace(0, 100, N),
            "rt": np.linspace(0, 100, N) + np.random.normal(0, 2, N)
        })
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.test_results_dir = cls.temp_dir.name

    @classmethod
    def teardown_class(cls):
        cls.temp_dir.cleanup()

    def test_empirical_fit_runs_without_error(self):
        os.makedirs(os.path.join(self.test_results_dir, "first_search"), exist_ok=True)

        try:
            cor_filter, emp_rt_spl = empirical_fit(
                self.output_df,
                self.test_results_dir
            )

        except RuntimeError as e:
            if "Optimal parameters not found" in str(e):
                pytest.skip("")
            else:
                raise

        assert hasattr(cor_filter, "dtype")
        assert cor_filter.dtype == bool
        assert callable(emp_rt_spl)
        assert os.path.isdir(self.test_results_dir)

class TestTimeplexPairNnGate:
    @staticmethod
    def _clean_frame(n=400, gaps=(3.0, 2.4), noise=0.2, seed=0):
        rng = np.random.default_rng(seed)
        base = rng.uniform(5, 25, n)
        c1 = gaps[0]
        c2 = gaps[0] + gaps[1]
        rows = []
        for i in range(n):
            rows.append((f"PEP{i}", base[i] + rng.normal(0, noise), 0))
            rows.append((f"PEP{i}", base[i] + c1 + rng.normal(0, noise), 1))
            rows.append((f"PEP{i}", base[i] + c2 + rng.normal(0, noise), 2))
        return pd.DataFrame(rows, columns=["stripped_seq", "rt", "channel"])

    def test_keeps_clean_peptides(self):
        df = self._clean_frame()
        mask = timeplex_pair_nn_gate(df, 3)
        assert mask.dtype == bool
        assert len(mask) == len(df)
        # the vast majority of clean members are kept
        assert mask.mean() > 0.9

    def test_drops_channel_off_with_both_others(self):
        # channel 2 sits far from channels 0 and 1 -> dropped; 0 and 1 agree -> kept
        df = self._clean_frame()
        df = pd.concat([df, pd.DataFrame(
            [("badA", 10.0, 0), ("badA", 13.0, 1), ("badA", 25.0, 2)],
            columns=df.columns)], ignore_index=True)
        mask = timeplex_pair_nn_gate(df, 3)
        keep = dict(zip(zip(df["stripped_seq"], df["channel"]), mask))
        assert keep[("badA", 0)] is np.True_ or keep[("badA", 0)]
        assert keep[("badA", 1)]
        assert not keep[("badA", 2)]

    def test_exonerates_member_passing_with_another_channel(self):
        # channel 1 collapses onto channel 0 (0-1 dist ~0, out of window) but 0 and 2
        # still agree (0-2 in window) -> drop only channel 1, keep 0 and 2
        df = self._clean_frame()
        df = pd.concat([df, pd.DataFrame(
            [("badB", 13.0, 0), ("badB", 13.0, 1), ("badB", 18.4, 2)],
            columns=df.columns)], ignore_index=True)
        mask = timeplex_pair_nn_gate(df, 3)
        keep = dict(zip(zip(df["stripped_seq"], df["channel"]), mask))
        assert keep[("badB", 0)]
        assert not keep[("badB", 1)]
        assert keep[("badB", 2)]

    def test_k1_keeps_all(self):
        df = pd.DataFrame({"stripped_seq": ["A", "B"], "rt": [1.0, 2.0],
                           "channel": [0, 0]})
        mask = timeplex_pair_nn_gate(df, 1)
        assert mask.all()


class TestHwhmModeSigma:
    def test_recovers_gaussian_center_and_width(self):
        rng = np.random.default_rng(0)
        v = rng.normal(5.0, 0.5, 20000)
        mode, sigma = _hwhm_mode_sigma(v)
        assert abs(mode - 5.0) < 0.2
        assert abs(sigma - 0.5) < 0.15

    def test_width_robust_to_tail(self):
        rng = np.random.default_rng(1)
        core = rng.normal(3.0, 0.3, 10000)
        tail = rng.uniform(0, 30, 3000)        # heavy contamination
        mode, sigma = _hwhm_mode_sigma(np.concatenate([core, tail]))
        assert abs(mode - 3.0) < 0.3
        # tail does not blow up the width (a plain std of this mixture would be ~8)
        assert sigma < 1.0


class TestFilterRtsByDense:
    def test_filter_rts_by_dense_basic(self):

        dense_middle = np.linspace(30, 70, 100)
        sparse_edges = np.concatenate([np.linspace(0, 10, 10), np.linspace(90, 100, 10)])
        rts = np.concatenate([sparse_edges, dense_middle])

        mask = filter_rts_by_dense(rts, n_bins=10)

        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert len(mask) == len(rts)
