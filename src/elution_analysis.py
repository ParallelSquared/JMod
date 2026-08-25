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


def _peak_fwhm(rts, ints):
    """FWHM of a single elution peak by half-maximum crossing.

    Linearly interpolates where the profile crosses half its apex on each side.
    Returns NaN unless the peak is bracketed on BOTH sides, so a truncated
    elution (cut off by the edge of its scan cluster) is skipped rather than
    reported as narrow.
    """
    order = np.argsort(rts)
    r, y = np.asarray(rts, float)[order], np.asarray(ints, float)[order]
    if r.size < 3:
        return np.nan
    i = int(np.argmax(y))
    ymax = y[i]
    if not np.isfinite(ymax) or ymax <= 0:
        return np.nan
    half = 0.5 * ymax
    left = right = None
    for j in range(i, 0, -1):
        if y[j - 1] <= half <= y[j]:
            left = (r[j - 1] if y[j] == y[j - 1]
                    else r[j - 1] + (half - y[j - 1]) * (r[j] - r[j - 1]) / (y[j] - y[j - 1]))
            break
    for j in range(i, r.size - 1):
        if y[j + 1] <= half <= y[j]:
            right = (r[j + 1] if y[j] == y[j + 1]
                     else r[j] + (half - y[j]) * (r[j + 1] - r[j]) / (y[j + 1] - y[j]))
            break
    if left is None or right is None:
        return np.nan
    return float(right - left)


def calculate_elution_width(df: pl.DataFrame, n_channels: int | None = None):
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

    # Per-time-channel accumulation. A timePlex peptide elutes once per channel, so
    # its clusters ranked by RT ARE the channels (the same rank-order argument used
    # for channel assignment everywhere else). Only peptides seen in exactly
    # n_channels clusters are used, so the rank-to-channel mapping is unambiguous.
    per_ch_rts: list[list[float]] = [[] for _ in range(n_channels or 0)]
    per_ch_ints: list[list[float]] = [[] for _ in range(n_channels or 0)]
    # FWHM measured on each INDIVIDUAL elution peak, then taken as a median per
    # channel. Fitting one gaussian to all of a channel's peaks overlaid instead
    # returns a blend of narrow and broad peaks that matches no actual peak.
    per_ch_fwhms: list[list[float]] = [[] for _ in range(n_channels or 0)]

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

        if n_channels:
            _use = [c for c in clusters if len(c) >= 2]
            if len(_use) == n_channels:
                _use.sort(key=lambda c: float(np.mean([rt for _, rt in c])))
                for _k, _c in enumerate(_use):
                    _crt = np.array([rt for _, rt in _c], dtype=float)
                    _cin = np.array([scan_to_intensity[sc] for sc, _ in _c], dtype=float)
                    per_ch_rts[_k].extend(_crt - np.mean(_crt))
                    per_ch_ints[_k].extend(_cin)
                    _w = _peak_fwhm(_crt, _cin)
                    if np.isfinite(_w):
                        per_ch_fwhms[_k].append(_w)

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

    per_ch_sigma = None
    if n_channels:
        per_ch_sigma = []
        for _k in range(n_channels):
            _r = np.asarray(per_ch_rts[_k], dtype=float)
            _w = np.asarray(per_ch_ints[_k], dtype=float)
            if _r.size < 20 or not np.isfinite(_r).any():
                per_ch_sigma.append(float(sigma))   # too thin to fit; pooled value
                continue
            try:
                _h, _e = np.histogram(_r, bins=100, weights=_w, density=True)
                _bc = (_e[:-1] + _e[1:]) / 2
                _popt, _ = curve_fit(_gaussian, _bc, _h,
                                     p0=[_h.max(), 0, np.std(_r)], maxfev=5000)
                _s = abs(_popt[2])
            except Exception:
                _s = float(np.std(_r))
            per_ch_sigma.append(float(_s) if np.isfinite(_s) and _s > 0 else float(sigma))

        per_ch_fwhm = []
        for _k in range(n_channels):
            _v = np.asarray(per_ch_fwhms[_k], dtype=float)
            _v = _v[np.isfinite(_v) & (_v > 0)]
            # too few well-formed peaks to trust a median -> this channel's fitted
            # sigma converted to a width, rather than a value from another channel
            per_ch_fwhm.append(float(np.median(_v)) if _v.size >= 20
                               else float(2.355 * per_ch_sigma[_k]))

    # Step 7: Add cluster_size column to dataframe
    df = df.with_columns(
        pl.struct(['seq', 'z'])
        .map_elements(lambda r: cluster_sizes.get((r['seq'], r['z']), 0), return_dtype=pl.Int64)
        .alias('cluster_size')
    )

    if n_channels:
        return fwhm, sigma, df, {"sigma": per_ch_sigma,
                                 "fwhm": per_ch_fwhm,
                                 "n_peaks": [len(v) for v in per_ch_fwhms]}
    return fwhm, sigma, df


def collapse_to_cluster_apices(df: pl.DataFrame, group_cols=('file_id', 'seq', 'z')) -> pl.DataFrame:
    """
    Collapse each peptide's per-spectrum PSMs into one apex row per elution.

    Uses the same adjacent-scan clustering as :func:`calculate_elution_width`,
    but instead of keeping only the largest cluster it emits one representative
    row per cluster: the highest-MS1-intensity PSM in that cluster (the apex).

    A peptide eluting N times (e.g. once per timeplex window) therefore yields N
    rows — one per elution — which is the "multiple RT entries per (seq, z)"
    structure the timeplex aligner expects. peppy_sage's first search produces a
    cluster of consecutive-scan matches per elution rather than a single row, so
    this step reconstructs that structure before ``get_multiples``.

    Args:
        df: Polars DataFrame with columns: file_id, spec_name, rt, z, seq,
            closest_peak_intensity_ms1 (the peppy_sage first-search output).
        group_cols: columns defining one peptide. Pass ('file_id', 'stripped_seq')
            to collapse co-eluting charge states / modforms of the same bare
            peptide into a single apex per elution, so each (peptide, channel)
            yields one row.

    Returns:
        Polars DataFrame containing a subset of ``df``'s rows (one apex per
        elution cluster), with all original columns preserved.
    """
    required_cols = {'spec_name', 'rt', 'seq', 'closest_peak_intensity_ms1', *group_cols}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Extract scan numbers and tag each row so clusters can be mapped back.
    df = df.with_columns(
        pl.col('spec_name').str.extract(r'scan=(\d+)', 1).cast(pl.Int64).alias('scan')
    )
    df = df.drop_nulls(subset=['scan', 'rt', 'seq'])
    df = df.with_row_index('__row')

    if df.height == 0:
        return df.drop(['scan', '__row'])

    max_scan_gap = _compute_scan_gap_mode(df) + 1  # extra scan just in case

    grouped = df.group_by(list(group_cols)).agg([
        pl.col('scan'),
        pl.col('rt'),
        pl.col('closest_peak_intensity_ms1'),
        pl.col('__row'),
    ])

    apex_rows: list[int] = []
    for row in grouped.iter_rows(named=True):
        scans = np.array(row['scan'])
        rts = np.array(row['rt'])
        intensities = np.array(row['closest_peak_intensity_ms1'])
        rows = np.array(row['__row'])

        if len(scans) == 1:
            apex_rows.append(int(rows[0]))
            continue

        # Highest-intensity source row per scan (collapses IM-bin duplicates),
        # used to map each cluster back to a single apex row.
        best_row_for_scan: dict[int, int] = {}
        best_int_for_scan: dict[int, float] = {}
        for s, i, r in zip(scans, intensities, rows):
            if s not in best_int_for_scan or i > best_int_for_scan[s]:
                best_int_for_scan[s] = i
                best_row_for_scan[s] = int(r)

        clusters = _find_adjacent_clusters(list(zip(scans, rts)), max_scan_gap)
        for cluster in clusters:
            apex_scan = max((s for s, _ in cluster), key=lambda s: best_int_for_scan[s])
            apex_rows.append(best_row_for_scan[apex_scan])

    return df.filter(pl.col('__row').is_in(apex_rows)).drop(['scan', '__row'])


def rt_cluster_ids(df: pl.DataFrame, group_cols, rt_gap: float) -> pl.DataFrame:
    """Vectorized RT-gap clustering. Adds ``__cid`` (1-based cluster index within
    each group): a new cluster starts whenever the RT gap to the previous row in
    the group exceeds ``rt_gap``. Elutions are seconds-wide and timeplex windows
    are minutes apart, so this cleanly separates the windows without the slow
    scan-adjacency Python loop (and without its duty-cycle harmonic fragmentation).
    """
    gcols = list(group_cols)
    df = df.sort([*gcols, 'rt'])
    df = df.with_columns(pl.col('rt').diff().over(gcols).alias('__gap'))
    df = df.with_columns(
        ((pl.col('__gap').is_null()) | (pl.col('__gap') > rt_gap)).cast(pl.Int64).alias('__brk'))
    df = df.with_columns(pl.col('__brk').cum_sum().over(gcols).alias('__cid'))
    return df.drop(['__gap', '__brk'])


def _count_exactly_k(df: pl.DataFrame, K: int, rt_gap: float, group_cols) -> int:
    """Number of peptides whose RT clusters total exactly K (summed across files)."""
    gcols = list(group_cols)
    u = rt_cluster_ids(df.select([*gcols, 'rt']).unique(), gcols, rt_gap)
    per = (u.group_by(gcols).agg(pl.col('__cid').max().alias('n'))
           .group_by('stripped_seq').agg(pl.col('n').sum().alias('n')))
    return int((per['n'] == K).sum())


def select_timeplex_alignment_set(df: pl.DataFrame, K: int, rt_gap: float,
                                  scribe_percentiles=None):
    """Build the timeplex RT-alignment input set, fully vectorized.

    Sweeps scribe-score cutoffs and picks the one that **maximizes the number of
    peptides seen in exactly K RT clusters** (K = n_timeplex). At that cutoff it
    RT-clusters the surviving rows, takes the **apex** (max MS1 intensity) of each
    cluster, keeps peptides with exactly K clusters, and assigns ``channel`` 0..K-1
    by RT rank within each peptide.

    Replaces the slow scan-adjacency collapse + scribe filter + completeness
    assignment on the timeplex path only.

    Required columns: ``file_id, stripped_seq, rt, scribe_score,
    closest_peak_intensity_ms1`` (all original columns are preserved on output).

    Returns
    -------
    apex : pl.DataFrame
        One apex row per (peptide, channel), with an added ``channel`` column.
    cutoff : float
        The chosen scribe-score cutoff.
    n_pep : int
        Number of peptides with exactly K clusters at that cutoff.
    """
    if scribe_percentiles is None:
        # p is "keep the top p% by scribe", so LARGER p = MORE LENIENT. The sweep
        # used to start at 50, i.e. it could never consider anything more lenient
        # than the median even when that would yield more COMPLETE peptides --
        # and completeness across all K channels, not confidence, is what this set
        # is short of (JD0319: only 1,947 peptides made exactly-K, against 14,444
        # fed to the no-timeplex fine-tune). The selection criterion is unchanged
        # (maximize the count of exactly-K peptides), so adding lenient options can
        # only help: if loosening starts splitting peptides into >K noise clusters,
        # the exactly-K count falls and the sweep rejects it on its own.
        # p=100 = no scribe filter at all. Added because the sweep chose the
        # most lenient option available (95 -> scribe >= -0.2561 against a floor
        # of -0.6511) and was therefore PINNED AT THE EDGE of the range, which
        # usually means the optimum lies past it. If 100 wins, the scribe cut was
        # never the useful constraint here; if the exactly-K count falls at 100,
        # the sweep rejects it unaided.
        # p=100 removed: it won the sweep's exactly-K criterion but LOST on IDs
        # (56,868 vs 59,670 at p95), because dropping the upstream scribe cut made
        # empirical_fit's own stepping cut harder downstream -- net FEWER training
        # peptides (5,107 vs 7,247). The sweep is blind to that, so cap it at 95.
        scribe_percentiles = [95, 90, 80, 70, 60,
                              50, 40, 30, 25, 20, 15, 12.5, 10, 8, 6.25, 5]
    group_cols = ('file_id', 'stripped_seq')
    scribe = df['scribe_score'].to_numpy()

    best_cut, best_n, best_p = None, -1, None
    for p in scribe_percentiles:
        cut = float(np.quantile(scribe, 1 - p / 100))
        n_k = _count_exactly_k(df.filter(pl.col('scribe_score') >= cut), K, rt_gap, group_cols)
        if n_k > best_n:
            best_cut, best_n, best_p = cut, n_k, p

    # build the apex set at the chosen cutoff
    sel = df.filter(pl.col('scribe_score') >= best_cut)
    sel = sel.with_columns(pl.col('closest_peak_intensity_ms1').fill_null(0.0).alias('__ms1i'))
    sel = rt_cluster_ids(sel, group_cols, rt_gap)
    apex = (sel.sort('__ms1i', descending=True)
            .group_by([*group_cols, '__cid'], maintain_order=True).first())

    # cluster count per peptide (summed across files)
    nc = (apex.group_by(list(group_cols)).agg(pl.len().alias('__nc'))
          .group_by('stripped_seq').agg(pl.col('__nc').sum().alias('__nc')))
    complete = nc.filter(pl.col('__nc') == K).select('stripped_seq')

    full = apex.join(complete, on='stripped_seq', how='inner')
    # channel by RT rank within each peptide (unambiguous only when complete)
    full = full.with_columns(
        (pl.col('rt').rank('ordinal').over('stripped_seq').cast(pl.Int32) - 1).alias('channel'))

    # Partial (2-of-K) recovery REMOVED. Channel assignment here is by RT RANK,
    # which is exact only when all K clusters are present. With fewer, the
    # assignment is degenerate: if the inter-channel gaps are similar, (0,1) and
    # (1,2) imply equally consistent solutions and no scoring can separate them.
    # On JD0319 the gaps differ only 38% (5.00 vs 6.90) with heavily overlapping
    # spreads, so the margin test rejected 71% of candidates (1,568 assigned /
    # 3,864 ambiguous) -- and disabling recovery entirely changed IDs by 15
    # (60,950 vs 60,965), i.e. nothing. It carried a mis-assignment risk for no
    # measurable benefit, so requiring all K is both simpler and structurally
    # unambiguous.

    n_pep = full.select('stripped_seq').n_unique()
    full = full.drop([c for c in ['__cid', '__ms1i'] if c in full.columns])
    return full, best_cut, n_pep


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
