"""
Integration tests for run_jmod.py main entry point.

These tests run the full pipeline once with test data and verify outputs.
Mark with @pytest.mark.slow to skip in quick test runs.
"""

#  Copyright (c) 2026 Parallel Squared Technology Institute
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import json
import os
import re
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest


# Path to test data
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
TEST_MZML = os.path.join(DATA_DIR, 'test_mode_filtered.mzML')
TEST_LIBRARY = os.path.join(DATA_DIR, 'filtered_library.tsv')
TEST_CONFIG = os.path.join(DATA_DIR, 'test_mode.json')


def _find_run_folders(output_dir):
    """The runs' results folders in *output_dir*, sorted by name.

    Found by name -- <data file>_results, with a datestamp when the name was
    taken, and a run_failed_ prefix if the run failed -- since the output
    folder also holds the experiment's own folders, such as experiment_results/.
    """
    stem = os.path.splitext(os.path.basename(TEST_MZML))[0]
    pattern = re.compile(rf"(run_failed_)?{re.escape(stem)}_results(_\d{{4}}(_\d\d){{5}})?")
    return sorted(os.path.join(output_dir, d) for d in os.listdir(output_dir)
                  if os.path.isdir(os.path.join(output_dir, d)) and pattern.fullmatch(d))


def _find_run_folder(output_dir):
    """The run's results folder in *output_dir* (a one-file experiment), or None."""
    folders = _find_run_folders(output_dir)
    return folders[0] if folders else None


def _run_experiment(config, temp_dir):
    """Write *config* as a config JSON in *temp_dir* and run JMod on it.

    config.args and config's module settings outlive an experiment, so they
    are reset to the defaults first and restored afterwards: no fixture's
    settings (SILAC, --mbr, additional_config) carry over into the next
    fixture, or into the unit tests that run after these.  Returns (the config
    JSON's path, the exit code or None).
    """
    temp_config_path = os.path.join(temp_dir, 'test_config.json')
    with open(temp_config_path, 'w') as f:
        json.dump(config, f)

    from src.run_jmod import main
    import src.config as config_module

    saved_args = dict(vars(config_module.args))
    saved_settings = dict(vars(config_module))
    vars(config_module.args).update(vars(config_module.parser.parse_args([])))
    config_module.args.config_json = temp_config_path
    exit_code = None
    try:
        main()
    except SystemExit as e:
        exit_code = e.code
    finally:
        vars(config_module.args).clear()
        vars(config_module.args).update(saved_args)
        vars(config_module).update(saved_settings)
    return temp_config_path, exit_code


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

    temp_config_path, exit_code = _run_experiment(config, temp_dir)

    # Find results folder
    # Everything the experiment makes goes in its JMod_Results folder
    experiment_dir = os.path.join(temp_dir, 'JMod_Results')
    results_folder = _find_run_folder(experiment_dir)

    # Provide results to tests
    yield {
        'output_dir': experiment_dir,
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
            'outputs/decoylibsearch_coeffs.parquet',
            'outputs/all_IDs.csv',
            'filtered_IDs.csv',
            'outputs/params.txt'
        ]

        for filename in expected_files:
            filepath = os.path.join(results_folder, filename)
            assert os.path.exists(filepath), f"Expected output file not found: {filename}"

    def test_output_files_have_content(self, pipeline_results):
        """Test that output files are non-empty and have expected structure."""
        results_folder = pipeline_results['results_folder']
        assert results_folder is not None, "No results folder"

        # Check all_IDs.csv has expected columns
        all_ids_path = os.path.join(results_folder, 'outputs/all_IDs.csv')
        assert os.path.exists(all_ids_path), "all_IDs.csv not found"

        df = pd.read_csv(all_ids_path)
        assert len(df) > 0, "outputs/all_IDs.csv is empty"

        # Check for key columns
        expected_columns = ['seq', 'z', 'coeff']
        for col in expected_columns:
            assert col in df.columns, f"Expected column '{col}' not in all_IDs.csv"

    def test_filtered_ids_subset_of_all_ids(self, pipeline_results):
        """Test that filtered_IDs is a subset of all_IDs."""
        results_folder = pipeline_results['results_folder']
        assert results_folder is not None, "No results folder"

        all_ids_path = os.path.join(results_folder, 'outputs/all_IDs.csv')
        filtered_ids_path = os.path.join(results_folder, 'filtered_IDs.csv')

        assert os.path.exists(all_ids_path), "outputs/all_IDs.csv not found"
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

        if 'is_decoy' in filtered_ids.columns:
            decoy_count = filtered_ids['is_decoy'].sum()
            assert decoy_count == 0, \
                f"filtered_IDs contains {decoy_count} decoy peptides"

        if 'seq' in filtered_ids.columns:
            decoy_seqs = filtered_ids['seq'].str.startswith('Decoy_').sum()
            assert decoy_seqs == 0, \
                f"filtered_IDs contains {decoy_seqs} sequences starting with 'Decoy_'"

    def test_experiment_results_combine_the_run(self, pipeline_results):
        """The experiment's combined table holds this one run's IDs."""
        combined_path = os.path.join(pipeline_results['output_dir'], 'combined_filtered_IDs.parquet')
        assert os.path.isfile(combined_path), "combined_filtered_IDs.parquet not found"
        assert os.path.isdir(os.path.join(pipeline_results['output_dir'], 'experiment_results'))
        combined = pd.read_parquet(combined_path)
        assert len(combined) > 0, "The combined table is empty"
        assert set(combined['run_idx']) == {1}


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

    temp_config_path, exit_code = _run_experiment(config, temp_dir)

    # Everything the experiment makes goes in its JMod_Results folder
    experiment_dir = os.path.join(temp_dir, 'JMod_Results')
    results_folder = _find_run_folder(experiment_dir)

    yield {
        'output_dir': experiment_dir,
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
            'outputs/decoylibsearch_coeffs.parquet',
            'outputs/all_IDs.csv',
            'filtered_IDs.csv',
            'outputs/params.txt'
        ]

        for filename in expected_files:
            filepath = os.path.join(results_folder, filename)
            assert os.path.exists(filepath), f"Expected output file not found: {filename}"


@pytest.fixture(scope="class")
def mbr_pipeline_results(request):
    """
    Run one --mbr experiment on the same data file twice, against the whole
    test library (a smaller one gives the first pass no IDs to build the MBR
    library from), and share the results across all tests in the class.
    """
    temp_dir = tempfile.mkdtemp(prefix='jmod_test_mbr_')

    with open(TEST_CONFIG, 'r') as f:
        config = json.load(f)

    config['mzml'] = [os.path.abspath(TEST_MZML)] * 2
    config['speclib'] = os.path.abspath(TEST_LIBRARY)
    config['output_folder'] = temp_dir
    config['test_mode'] = False
    config['mbr'] = True

    _run_experiment(config, temp_dir)
    experiment_dir = os.path.join(temp_dir, 'JMod_Results')

    yield {
        'output_dir': experiment_dir,
        'first_pass_dir': os.path.join(experiment_dir, 'first_pass'),
        'mbr_dir': os.path.join(experiment_dir, 'mbr_library'),
    }

    shutil.rmtree(temp_dir, ignore_errors=True)


def _completed_run_folders(parent_dir):
    """The run folders in *parent_dir*, checked to be two runs that completed."""
    run_folders = _find_run_folders(parent_dir)
    assert len(run_folders) == 2, f"Expected two results folders in {parent_dir}, found {run_folders}"
    failed = [f for f in run_folders if os.path.basename(f).startswith('run_failed')]
    assert not failed, f"Runs failed: {failed}"
    return run_folders


@pytest.mark.slow
class TestMBRIntegration:
    """--mbr on the same data file twice: a first pass in first_pass/, an MBR
    library of its IDs in mbr_library/, then the final pass under the normal
    names.

    Both runs are the same file, which also tests determinism and run order:
    in each pass the two runs must give the same IDs, so nothing about a run
    depends on the run before it.  And their RTs agree exactly: the reference
    run is run 1 (ties go to the earlier run), and the MBR library's RTs must
    be run 1's first-pass RTs.
    """

    def test_first_pass_goes_in_first_pass(self, mbr_pipeline_results):
        first_pass_dir = mbr_pipeline_results['first_pass_dir']
        _completed_run_folders(first_pass_dir)
        assert os.path.isfile(os.path.join(first_pass_dir, 'combined_filtered_IDs.parquet'))

    def test_first_pass_runs_give_the_same_ids(self, mbr_pipeline_results):
        run_folders = _completed_run_folders(mbr_pipeline_results['first_pass_dir'])
        n_ids = [len(pd.read_parquet(os.path.join(f, 'filtered_IDs.parquet'))) for f in run_folders]
        assert n_ids[0] > 0, "The first pass identified nothing"
        assert n_ids[0] == n_ids[1], f"IDs differ between the two first-pass runs of the same file: {n_ids}"

    def test_mbr_library_and_its_plots_are_written(self, mbr_pipeline_results):
        mbr_dir = mbr_pipeline_results['mbr_dir']
        for name in ('mbrlib.parquet', 'library_size_by_run.png'):
            assert os.path.isfile(os.path.join(mbr_dir, name)), f"{name} not in mbr_library/"
        assert os.listdir(os.path.join(mbr_dir, 'lowess')) == ['run_2.png']  # run 1 is the reference

    def test_mbr_library_holds_the_first_pass_ids(self, mbr_pipeline_results):
        library = pd.read_parquet(os.path.join(mbr_pipeline_results['mbr_dir'], 'mbrlib.parquet'))
        first_pass = pd.read_parquet(os.path.join(mbr_pipeline_results['first_pass_dir'],
                                                  'combined_filtered_IDs.parquet'))
        library_precursors = set(library['ModifiedPeptide'] + '_' + library['PrecursorCharge'].astype(str))
        assert len(library_precursors) > 0, "The MBR library is empty"
        assert library_precursors == set(first_pass['untag_prec'])

    def test_mbr_library_rts_are_the_reference_runs(self, mbr_pipeline_results):
        library = pd.read_parquet(os.path.join(mbr_pipeline_results['mbr_dir'], 'mbrlib.parquet'))
        first_pass = pd.read_parquet(os.path.join(mbr_pipeline_results['first_pass_dir'],
                                                  'combined_filtered_IDs.parquet'))
        library_rt = (library.assign(untag_prec=library['ModifiedPeptide'] + '_'
                                     + library['PrecursorCharge'].astype(str))
                      .groupby('untag_prec')['RT'].first())
        run_1_rt = first_pass[first_pass['run_idx'] == 1].set_index('untag_prec')['rt']
        assert np.allclose(library_rt[run_1_rt.index], run_1_rt, rtol=1e-6)

    def test_final_pass_keeps_the_normal_names(self, mbr_pipeline_results):
        run_folders = _completed_run_folders(mbr_pipeline_results['output_dir'])
        n_ids = [len(pd.read_parquet(os.path.join(f, 'filtered_IDs.parquet'))) for f in run_folders]
        assert n_ids[0] > 0, "The final pass identified nothing"
        assert n_ids[0] == n_ids[1], f"IDs differ between the two final runs of the same file: {n_ids}"
