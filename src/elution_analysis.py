"""
This Source Code Form is subject to the terms of the Oxford Nanopore
Technologies, Ltd. Public License, v. 1.0.  Full licence can be found
at https://github.com/ParallelSquared/JMod/blob/main/LICENSE.txt
"""

from collections import defaultdict

import numpy as np
import polars as pl
from scipy.optimize import curve_fit


def _gaussian(x: np.ndarray, amplitude: float, mu: float, sigma: float) -> np.ndarray:
    """Gaussian function for curve fitting."""
    return amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


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

    This automatically detects the MS duty cycle. When IM-binned data produces
    many scans at the same RT, duplicates are collapsed per unique RT first so
    that the gap reflects the true RT-cycle spacing.

    Args:
        df: DataFrame with columns: file_id, scan, rt, seq

    Returns:
        Mode of scan gap distribution (the MS duty cycle)
    """
    all_gaps: list[int] = []

    # Group by file_id and seq, collecting both scan and rt
    grouped = df.group_by(['file_id', 'seq']).agg([
        pl.col('scan').sort(),
        pl.col('rt'),
    ])

    for row in grouped.iter_rows(named=True):
        scans = np.array(row['scan'])
        rts = np.array(row['rt'])

        if len(scans) < 2:
            continue

        # Sort by scan number
        order = np.argsort(scans)
        scans = scans[order]
        rts = rts[order]

        # Deduplicate by unique RT (collapse IM-bin duplicates at the same RT)
        _, unique_idx = np.unique(rts, return_index=True)
        unique_idx.sort()
        scans = scans[unique_idx]

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


def calculate_elution_width(df: pl.DataFrame) -> tuple[float, float, pl.DataFrame]:
    """
    Calculate FWHM and sigma of elution profile from PSM data.

    This function:
    1. Extracts scan numbers from spec_name
    2. Auto-detects the MS duty cycle (scan gap mode)
    3. Clusters adjacent scans for each peptide
    4. Centers and overlays all RT profiles (intensity-weighted)
    5. Fits a Gaussian to compute FWHM and sigma
    6. Adds cluster_size column to the dataframe

    Args:
        df: Polars DataFrame with columns:
            - file_id: MS run identifier
            - spec_name: Spectrum name containing 'scan=N'
            - rt: Retention time in minutes
            - z: Charge state
            - seq: Modified peptide sequence
            - closest_peak_intensity_ms1: MS1 intensity for weighting

    Returns:
        Tuple of (fwhm, sigma, df) where df has a new 'cluster_size' column

    Raises:
        ValueError: If required columns are missing or no valid clusters found
    """
    required_cols = {'file_id', 'spec_name', 'rt', 'z', 'seq', 'closest_peak_intensity_ms1'}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Extract scan numbers from spec_name
    df = df.with_columns(
        pl.col('spec_name').str.extract(r'scan=(\d+)', 1).cast(pl.Int64).alias('scan')
    )

    # Drop rows with missing values
    df = df.drop_nulls(subset=['scan', 'rt', 'seq'])

    if df.height == 0:
        raise ValueError("No valid PSMs after filtering")

    # Compute scan gap mode (auto-detect MS duty cycle)
    scan_gap_mode = _compute_scan_gap_mode(df)
    max_scan_gap = scan_gap_mode + 1 # extra scan just in case

    # Step 4 & 5: Cluster adjacent scans and collect centered RTs
    centered_rts: list[float] = []
    centered_intensities: list[float] = []

    # Track cluster sizes per (seq, z)
    cluster_sizes: dict[tuple[str, int], int] = {}

    # Group by file_id, seq, and charge state
    grouped = df.group_by(['file_id', 'seq', 'z']).agg([
        pl.col('scan'),
        pl.col('rt'),
        pl.col('closest_peak_intensity_ms1')
    ])

    for row in grouped.iter_rows(named=True):
        scans = np.array(row['scan'])
        rts = np.array(row['rt'])
        intensities = np.array(row['closest_peak_intensity_ms1'])
        key = (row['seq'], row['z'])

        # Collapse IM-bin duplicates: keep highest intensity per unique RT
        unique_rts_vals, inv = np.unique(rts, return_inverse=True)
        if len(unique_rts_vals) < len(rts):
            # Multiple scans at same RT (IM bins) — pick best intensity per RT
            best_scans = np.empty(len(unique_rts_vals), dtype=scans.dtype)
            best_ints = np.full(len(unique_rts_vals), -np.inf)
            best_rts = unique_rts_vals
            for j in range(len(scans)):
                uid = inv[j]
                if intensities[j] > best_ints[uid]:
                    best_ints[uid] = intensities[j]
                    best_scans[uid] = scans[j]
            scans = best_scans
            rts = best_rts
            intensities = best_ints

        if len(scans) < 2:
            cluster_sizes[key] = max(cluster_sizes.get(key, 0), 1)
            continue

        # Create mapping from scan to intensity
        scan_to_intensity = {s: i for s, i in zip(scans, intensities)}

        # Create list of (scan, rt) tuples
        scans_rts = list(zip(scans, rts))

        # Find adjacent clusters
        clusters = _find_adjacent_clusters(scans_rts, max_scan_gap)

        # Keep largest cluster
        largest_cluster = max(clusters, key=len)
        cluster_sizes[key] = max(cluster_sizes.get(key, 0), len(largest_cluster))

        if len(largest_cluster) >= 2:
            cluster_rts = np.array([rt for _, rt in largest_cluster])
            cluster_ints = np.array([scan_to_intensity[scan] for scan, _ in largest_cluster])
            # Center the RTs
            centered = (cluster_rts - np.mean(cluster_rts))
            centered_rts.extend(centered)
            centered_intensities.extend(cluster_ints)

    if len(centered_rts) == 0:
        raise ValueError("No valid clusters found for FWHM calculation")

    centered_rts_arr = np.array(centered_rts)
    centered_intensities_arr = np.array(centered_intensities)

    # Step 6: Fit Gaussian to pooled distribution (intensity-weighted)
    hist, bin_edges = np.histogram(centered_rts_arr, bins=100, weights=centered_intensities_arr, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    try:
        p0 = [hist.max(), 0, np.std(centered_rts_arr)]
        popt, _ = curve_fit(_gaussian, bin_centers, hist, p0=p0, maxfev=5000)
        sigma = abs(popt[2])
    except RuntimeError:
        # Fallback to std-based estimate if curve fitting fails
        sigma = np.std(centered_rts_arr)

    fwhm = 2.355 * sigma

    # Step 7: Add cluster_size column to dataframe
    df = df.with_columns(
        pl.struct(['seq', 'z'])
        .map_elements(lambda r: cluster_sizes.get((r['seq'], r['z']), 0), return_dtype=pl.Int64)
        .alias('cluster_size')
    )

    return fwhm, sigma, df


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
