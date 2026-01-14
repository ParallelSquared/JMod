"""
This Source Code Form is subject to the terms of the Oxford Nanopore
Technologies, Ltd. Public License, v. 1.0.  Full licence can be found
at https://github.com/ParallelSquared/JMod/blob/main/LICENSE.txt
"""

import re
from collections import defaultdict

import numpy as np
import polars as pl
from scipy.optimize import curve_fit


def _gaussian(x: np.ndarray, amplitude: float, mu: float, sigma: float) -> np.ndarray:
    """Gaussian function for curve fitting."""
    return amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _create_modified_peptide_id(stripped_seq: str, modifications: str) -> str:
    """
    Create a unique identifier for modified peptides.

    Args:
        stripped_seq: Bare peptide sequence without modifications
        modifications: String representation of modification masses

    Returns:
        Unique identifier combining sequence and rounded modification masses
    """
    mod_str = str(modifications).replace('\n', ' ').replace('  ', ' ')
    masses = re.findall(r'[-+]?\d*\.?\d+', mod_str)
    rounded_masses = tuple(round(float(m), 2) for m in masses if float(m) != 0)
    return f"{stripped_seq}_{rounded_masses}"


def _find_adjacent_clusters(
    scans_rts: list[tuple[int, float]],
    max_scan_gap: int
) -> list[list[tuple[int, float]]]:
    """
    Find clusters of adjacent scans.

    Args:
        scans_rts: List of (scan_number, retention_time) tuples
        max_scan_gap: Maximum scan gap to consider as "adjacent"

    Returns:
        List of clusters, each cluster is a list of (scan, rt) tuples
    """
    if len(scans_rts) == 0:
        return []

    # Aggregate RTs for duplicate scans
    scan_to_rts: dict[int, list[float]] = defaultdict(list)
    for scan, rt in scans_rts:
        scan_to_rts[scan].append(rt)

    unique_scans = [(scan, np.mean(rts)) for scan, rts in scan_to_rts.items()]
    sorted_data = sorted(unique_scans, key=lambda x: x[0])

    clusters: list[list[tuple[int, float]]] = []
    current_cluster = [sorted_data[0]]

    for i in range(1, len(sorted_data)):
        scan, rt = sorted_data[i]
        prev_scan = current_cluster[-1][0]

        if scan - prev_scan <= max_scan_gap:
            current_cluster.append((scan, rt))
        else:
            clusters.append(current_cluster)
            current_cluster = [(scan, rt)]

    clusters.append(current_cluster)
    return clusters


def _compute_scan_gap_mode(df: pl.DataFrame) -> int:
    """
    Compute the mode of scan gaps between consecutive PSMs for the same peptide.

    This automatically detects the MS duty cycle.

    Args:
        df: DataFrame with columns: file_id, scan, mod_peptide

    Returns:
        Mode of scan gap distribution (the MS duty cycle)
    """
    all_gaps: list[int] = []

    # Group by file_id and mod_peptide
    grouped = df.group_by(['file_id', 'mod_peptide']).agg([
        pl.col('scan').sort()
    ])

    for row in grouped.iter_rows(named=True):
        scans = row['scan']
        if len(scans) < 2:
            continue

        # Compute gaps between consecutive scans
        gaps = np.diff(scans)
        all_gaps.extend(gaps)

    if len(all_gaps) == 0:
        return 25  # Default fallback

    # Find mode
    unique_gaps, counts = np.unique(all_gaps, return_counts=True)
    mode_idx = np.argmax(counts)
    mode_gap = int(unique_gaps[mode_idx])

    return mode_gap


def calculate_elution_width(df: pl.DataFrame) -> tuple[float, float]:
    """
    Calculate FWHM and sigma of elution profile from PSM data.

    This function:
    1. Extracts scan numbers from spec_name
    2. Creates modified peptide identifiers
    3. Auto-detects the MS duty cycle (scan gap mode)
    4. Clusters adjacent scans for each peptide
    5. Centers and overlays all RT profiles
    6. Fits a Gaussian to compute FWHM and sigma

    Args:
        df: Polars DataFrame with columns:
            - file_id: MS run identifier
            - spec_name: Spectrum name containing 'scan=N'
            - rt: Retention time in minutes
            - stripped_seq: Peptide sequence without modifications
            - modifications: Modification masses string
            - z: Charge state

    Returns:
        Tuple of (fwhm, sigma) in seconds

    Raises:
        ValueError: If required columns are missing or no valid clusters found
    """
    required_cols = {'file_id', 'spec_name', 'rt', 'stripped_seq', 'modifications', 'z'}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Step 1: Extract scan numbers from spec_name
    df = df.with_columns(
        pl.col('spec_name').str.extract(r'scan=(\d+)', 1).cast(pl.Int64).alias('scan')
    )

    # Drop rows with missing values
    df = df.drop_nulls(subset=['scan', 'rt', 'stripped_seq'])

    if df.height == 0:
        raise ValueError("No valid PSMs after filtering")

    # Step 2: Create modified peptide identifiers
    # Convert to Python for the UDF
    mod_peptide_ids = [
        _create_modified_peptide_id(seq, mods)
        for seq, mods in zip(
            df['stripped_seq'].to_list(),
            df['modifications'].to_list()
        )
    ]
    df = df.with_columns(pl.Series('mod_peptide', mod_peptide_ids))

    # Step 3: Compute scan gap mode (auto-detect MS duty cycle)
    scan_gap_mode = _compute_scan_gap_mode(df)
    max_scan_gap = scan_gap_mode + 5

    # Step 4 & 5: Cluster adjacent scans and collect centered RTs
    centered_rts: list[float] = []

    # Group by file_id, mod_peptide, and charge state
    grouped = df.group_by(['file_id', 'mod_peptide', 'z']).agg([
        pl.col('scan'),
        pl.col('rt')
    ])

    for row in grouped.iter_rows(named=True):
        scans = row['scan']
        rts = row['rt']

        if len(scans) < 2:
            continue

        # Create list of (scan, rt) tuples
        scans_rts = list(zip(scans, rts))

        # Find adjacent clusters
        clusters = _find_adjacent_clusters(scans_rts, max_scan_gap)

        # Keep largest cluster
        largest_cluster = max(clusters, key=len)

        if len(largest_cluster) >= 2:
            cluster_rts = np.array([rt for _, rt in largest_cluster])
            # Center the RTs and convert to seconds
            centered = (cluster_rts - np.mean(cluster_rts))
            centered_rts.extend(centered)

    if len(centered_rts) == 0:
        raise ValueError("No valid clusters found for FWHM calculation")

    centered_rts_arr = np.array(centered_rts)

    # Step 6: Fit Gaussian to pooled distribution
    hist, bin_edges = np.histogram(centered_rts_arr, bins=100, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    try:
        p0 = [hist.max(), 0, np.std(centered_rts_arr)]
        popt, _ = curve_fit(_gaussian, bin_centers, hist, p0=p0, maxfev=5000)
        sigma = abs(popt[2])
    except RuntimeError:
        # Fallback to std-based estimate if curve fitting fails
        sigma = np.std(centered_rts_arr)

    fwhm = 2.355 * sigma

    return fwhm, sigma


if __name__ == '__main__':
    # Example usage / verification
    import sys

    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = 'firstSearch_clean.tsv'

    print(f"Loading {input_file}...")
    df = pl.read_csv(input_file, separator='\t')
    print(f"Loaded {df.height} PSMs")

    print("\nCalculating elution width...")
    fwhm, sigma = calculate_elution_width(df)

    print(f"\nResults:")
    print(f"  Sigma: {sigma:.2f} sec")
    print(f"  FWHM:  {fwhm:.2f} sec")
