"""
This Source Code Form is subject to the terms of the Oxford Nanopore
Technologies, Ltd. Public License, v. 1.0.  Full licence can be found
at https://github.com/ParallelSquared/JMod/blob/main/LICENSE.txt
"""

"""
Integration tests for run_jmod.py main entry point.

These tests run the full pipeline once with test data and verify outputs.
Mark with @pytest.mark.slow to skip in quick test runs.
"""

import json
import os
import shutil
import tempfile

import pandas as pd
import pytest


# Path to test data
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
TEST_MZML = os.path.join(DATA_DIR, 'test_mode_filtered.mzML')
TEST_LIBRARY = os.path.join(DATA_DIR, 'filtered_library.tsv')
TEST_CONFIG = os.path.join(DATA_DIR, 'test_mode.json')


@pytest.fixture(scope="class")
def pipeline_results(request):
    """
    Run the JMod pipeline once and share results across all tests in the class.

    This fixture:
    1. Creates a temporary output directory
    2. Runs main() once
    3. Provides the output directory and results folder to all tests
    4. Cleans up after all tests complete
    """
    # Create temporary output directory
    temp_dir = tempfile.mkdtemp(prefix='jmod_test_')

    # Load and modify config
    with open(TEST_CONFIG, 'r') as f:
        config = json.load(f)

    config['mzml'] = os.path.abspath(TEST_MZML)
    config['speclib'] = os.path.abspath(TEST_LIBRARY)
    config['output_folder'] = temp_dir
    config['test_mode'] = False

    # Write temporary config file
    temp_config_path = os.path.join(temp_dir, 'test_config.json')
    with open(temp_config_path, 'w') as f:
        json.dump(config, f)

    # Run the pipeline
    from src.run_jmod import main
    import src.config as config_module

    config_module.args.config_json = temp_config_path

    exit_code = None
    try:
        main()
    except SystemExit as e:
        exit_code = e.code

    # Find results folder
    results_folders = [d for d in os.listdir(temp_dir)
                      if os.path.isdir(os.path.join(temp_dir, d))]
    results_folder = os.path.join(temp_dir, results_folders[0]) if results_folders else None

    # Provide results to tests
    yield {
        'output_dir': temp_dir,
        'config_path': temp_config_path,
        'results_folder': results_folder,
        'exit_code': exit_code,
    }

    # Cleanup after all tests
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.slow
class TestJModIntegration:
    """Integration tests for the full JMod pipeline."""

    def test_main_runs_without_error(self, pipeline_results):
        """Test that main() completes without raising exceptions."""
        exit_code = pipeline_results['exit_code']
        if exit_code is not None and exit_code != 0:
            pytest.fail(f"main() exited with code {exit_code}")

    def test_results_folder_created(self, pipeline_results):
        """Test that a results folder was created."""
        results_folder = pipeline_results['results_folder']
        assert results_folder is not None, "No results folder created"
        assert os.path.isdir(results_folder), "Results folder is not a directory"
        assert not os.path.basename(results_folder).startswith('run_failed'), \
            "Pipeline run failed (results folder prefixed with 'run_failed')"

    def test_output_files_created(self, pipeline_results):
        """Test that expected output files are created."""
        results_folder = pipeline_results['results_folder']
        assert results_folder is not None, "No results folder"

        expected_files = [
            # 'ms2scans.csv',
            'decoylibsearch_coeffs.parquet',
            'all_IDs.csv',
            'filtered_IDs.csv',
            'params.txt'
        ]

        for filename in expected_files:
            filepath = os.path.join(results_folder, filename)
            assert os.path.exists(filepath), f"Expected output file not found: {filename}"

    def test_output_files_have_content(self, pipeline_results):
        """Test that output files are non-empty and have expected structure."""
        results_folder = pipeline_results['results_folder']
        assert results_folder is not None, "No results folder"

        # Check all_IDs.csv has expected columns
        all_ids_path = os.path.join(results_folder, 'all_IDs.csv')
        assert os.path.exists(all_ids_path), "all_IDs.csv not found"

        df = pd.read_csv(all_ids_path)
        assert len(df) > 0, "all_IDs.csv is empty"

        # Check for key columns
        expected_columns = ['seq', 'z', 'coeff']
        for col in expected_columns:
            assert col in df.columns, f"Expected column '{col}' not in all_IDs.csv"

    def test_filtered_ids_subset_of_all_ids(self, pipeline_results):
        """Test that filtered_IDs is a subset of all_IDs."""
        results_folder = pipeline_results['results_folder']
        assert results_folder is not None, "No results folder"

        all_ids_path = os.path.join(results_folder, 'all_IDs.csv')
        filtered_ids_path = os.path.join(results_folder, 'filtered_IDs.csv')

        assert os.path.exists(all_ids_path), "all_IDs.csv not found"
        assert os.path.exists(filtered_ids_path), "filtered_IDs.csv not found"

        all_ids = pd.read_csv(all_ids_path)
        filtered_ids = pd.read_csv(filtered_ids_path)

        # Filtered should have fewer or equal rows
        assert len(filtered_ids) <= len(all_ids), \
            "filtered_IDs has more rows than all_IDs"

    def test_no_decoys_in_filtered_ids(self, pipeline_results):
        """Test that filtered_IDs contains no decoy peptides."""
        results_folder = pipeline_results['results_folder']
        assert results_folder is not None, "No results folder"

        filtered_ids_path = os.path.join(results_folder, 'filtered_IDs.csv')
        assert os.path.exists(filtered_ids_path), "filtered_IDs.csv not found"

        filtered_ids = pd.read_csv(filtered_ids_path)

        if 'decoy' in filtered_ids.columns:
            decoy_count = filtered_ids['decoy'].sum()
            assert decoy_count == 0, \
                f"filtered_IDs contains {decoy_count} decoy peptides"

        if 'seq' in filtered_ids.columns:
            decoy_seqs = filtered_ids['seq'].str.startswith('Decoy_').sum()
            assert decoy_seqs == 0, \
                f"filtered_IDs contains {decoy_seqs} sequences starting with 'Decoy_'"


@pytest.fixture(scope="class")
def silac_pipeline_results(request):
    """
    Run the JMod pipeline with SILAC K_6C13 tagging and share
    results across all tests in the class.
    """
    temp_dir = tempfile.mkdtemp(prefix='jmod_test_silac_')

    with open(TEST_CONFIG, 'r') as f:
        config = json.load(f)

    config['mzml'] = os.path.abspath(TEST_MZML)
    config['speclib'] = os.path.abspath(TEST_LIBRARY)
    config['output_folder'] = temp_dir
    config['test_mode'] = False
    config['SILAC'] = 'K_6C13'

    temp_config_path = os.path.join(temp_dir, 'test_config.json')
    with open(temp_config_path, 'w') as f:
        json.dump(config, f)

    from src.run_jmod import main
    import src.config as config_module

    config_module.args.config_json = temp_config_path

    exit_code = None
    try:
        main()
    except SystemExit as e:
        exit_code = e.code

    results_folders = [d for d in os.listdir(temp_dir)
                      if os.path.isdir(os.path.join(temp_dir, d))]
    results_folder = os.path.join(temp_dir, results_folders[0]) if results_folders else None

    yield {
        'output_dir': temp_dir,
        'config_path': temp_config_path,
        'results_folder': results_folder,
        'exit_code': exit_code,
    }

    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.slow
class TestSILACIntegration:
    """Integration tests for SILAC K_6C13 tagging pipeline."""

    def test_main_runs_without_error(self, silac_pipeline_results):
        """Test that main() completes with SILAC K_6C13 tagging."""
        exit_code = silac_pipeline_results['exit_code']
        if exit_code is not None and exit_code != 0:
            pytest.fail(f"main() exited with code {exit_code}")

    def test_results_folder_created(self, silac_pipeline_results):
        """Test that a results folder was created."""
        results_folder = silac_pipeline_results['results_folder']
        assert results_folder is not None, "No results folder created"
        assert os.path.isdir(results_folder), "Results folder is not a directory"
        assert not os.path.basename(results_folder).startswith('run_failed'), \
            "Pipeline run failed (results folder prefixed with 'run_failed')"

    def test_output_files_created(self, silac_pipeline_results):
        """Test that expected output files are created."""
        results_folder = silac_pipeline_results['results_folder']
        assert results_folder is not None, "No results folder"

        expected_files = [
           # 'ms2scans.csv',
            'decoylibsearch_coeffs.parquet',
            'all_IDs.csv',
            'filtered_IDs.csv',
            'params.txt'
        ]

        for filename in expected_files:
            filepath = os.path.join(results_folder, filename)
            assert os.path.exists(filepath), f"Expected output file not found: {filename}"
