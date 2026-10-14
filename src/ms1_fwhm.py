"""
Per-precursor MS1 elution peak width (``--ms1_fwhm``).

For every confident precursor (non-decoy, BestChannel_Qvalue < 0.01) the MS1
elution profile is rebuilt from the same per-scan isotope-envelope fit ("plex
fit") that ``plex_Area`` sums over its +/-1-scan apex window, starting from the
same voted apex scan. The profile is grown outward scan by scan until every
channel has returned to baseline, and a Gaussian is fitted to it over RT. The
reported width is FWHM = 2*sqrt(2 ln 2)*sigma, in the units of the spectra RT
(minutes for mzML input), in the column ``MS1_FWHM``. The fit's R² goes in
``MS1_FWHM_R2``; widths are reported whatever their R², so filter on it
downstream if needed.

The stored ``plexfittrace_all`` cannot be reused: it holds only the 3 scans
around the apex, which is less than one peak width at typical MS1 cycle times.

This module only reads. Nothing it computes feeds scoring, FDR or
quantification; the caller attaches the result as two columns immediately
before the results are written.
"""
import warnings

import numpy as np
import tqdm
from pyteomics import mass
from scipy.optimize import curve_fit

import src.config as config
from src.logger import logger
from src.mass_tags import massTag
from src.ms1_cor_channels import (get_seqs_and_mzs, minmax_spec_window,
                                  compute_isotopes, fit_isotopes_and_score)

FWHM_COL = "MS1_FWHM"
R2_COL = "MS1_FWHM_R2"
FWHM_QVALUE = 0.01        # rows with BestChannel_Qvalue below this get a width
SIGMA_TO_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))
VOTE_WINDOW = 10          # ms1_cor_channels' window_half_width; the apex vote must match it
EDGE_FRAC = 0.05          # a side ends after two consecutive scans <= this fraction of the max
VALLEY_RISE = 0.10        # ...or at a sub-half-max valley followed by a rise of this fraction
MIN_PEAK_PTS = 3          # nonzero points in the peak (plus at least one baseline point)


def _gauss(t, a, mu, sigma):
    return a * np.exp(-0.5 * ((t - mu) / sigma) ** 2)


def _voted_apex(tag_group, all_scans, vote_sigma):
    """The group's apex MS1 scan, voted exactly as ms1_cor_channels votes it."""
    all_scans_arr = np.asarray(all_scans)
    positions = np.arange(len(all_scans_arr), dtype=np.float64)
    votes = np.zeros(len(all_scans_arr), dtype=np.float64)
    for seq in tag_group["seq"].unique():
        ch_rows = tag_group[tag_group["seq"] == seq]
        best_row = ch_rows.loc[ch_rows["coeff"].idxmax()]
        s_idx = float(np.searchsorted(all_scans_arr, int(best_row["Ms1_spec_id"])))
        votes += float(best_row["coeff"]) * np.exp(-0.5 * ((positions - s_idx) / vote_sigma) ** 2)
    return int(all_scans_arr[int(np.argmax(votes))])


def _grow_profile(ms1_spectra, ms1_spec_idxs, scans, apex_pos, group_iso, mz_ppm):
    """Plex-fit intensity of every channel, fitted scan by scan outward from
    ``apex_pos`` until each channel has had two consecutive scans at or below
    EDGE_FRAC of its running maximum (or the scan list ends).

    Returns (lo, hi, coeffs) with coeffs of shape (hi - lo + 1, n_channels).
    """
    cache = {}

    def coeff(i):
        if i not in cache:
            cache[i] = np.asarray(fit_isotopes_and_score(
                ms1_spectra, ms1_spec_idxs, scans[i], group_iso, mz_ppm)[0], dtype=float)
        return cache[i]

    peak = coeff(apex_pos).copy()
    ends = []
    for step in (-1, 1):
        i = apex_pos
        low_run = np.zeros(peak.shape, dtype=int)
        while 0 <= i + step < len(scans):
            i += step
            c = coeff(i)
            peak = np.maximum(peak, c)
            low_run = np.where(c <= EDGE_FRAC * peak, low_run + 1, 0)
            if np.all(low_run >= 2):
                break
        ends.append(i)
    lo, hi = ends
    return lo, hi, np.vstack([coeff(i) for i in range(lo, hi + 1)])


def _peak_side(y, apex, step, ymax):
    """Index where the peak ends walking from ``apex`` in direction ``step``, or
    None if the profile runs out before the peak is resolved on that side.

    The side ends at the first of two consecutive scans at or below EDGE_FRAC of
    the apex, or at a valley: a local minimum below half the apex followed by a
    rise of more than VALLEY_RISE of the apex (the start of a neighbouring peak).
    Lone dropouts (NaN, see _gaussian_fwhm) are stepped over.
    """
    n = len(y)
    i = apex
    prev = ymax
    while True:
        j = i + step
        while 0 <= j < n and np.isnan(y[j]):
            j += step
        if not 0 <= j < n:
            return None
        if y[j] <= EDGE_FRAC * ymax:
            k = j + step
            while 0 <= k < n and np.isnan(y[k]):
                k += step
            if not 0 <= k < n or y[k] <= EDGE_FRAC * ymax:
                return j
        if prev < 0.5 * ymax and y[j] > prev + VALLEY_RISE * ymax:
            return i
        prev = y[j]
        i = j


_NO_FIT = (np.nan, np.nan)


def _gaussian_fwhm(t, y, start):
    """(FWHM, R²) of a Gaussian fitted to the peak of ``y`` (over RT ``t``) that
    contains or is reached uphill from index ``start``; (NaN, NaN) when there is
    no resolved peak there or its fitted width is pinned to a bound. R² is over the
    fitted points and is not used as a filter."""
    y = np.asarray(y, dtype=float).copy()
    n = len(y)
    if n < 3:
        return _NO_FIT
    # a lone zero between two non-zero scans is a missed detection, not signal
    pos = y > 0
    lone = np.zeros(n, dtype=bool)
    lone[1:-1] = (~pos[1:-1]) & pos[:-2] & pos[2:]
    y[lone] = np.nan

    apex = int(start)
    if np.isnan(y[apex]):
        apex = apex - 1 if (apex > 0 and not np.isnan(y[apex - 1])) else apex + 1
    while True:                                   # walk uphill to the local maximum
        left = y[apex - 1] if apex > 0 and not np.isnan(y[apex - 1]) else -np.inf
        right = y[apex + 1] if apex < n - 1 and not np.isnan(y[apex + 1]) else -np.inf
        if max(left, right) <= y[apex]:
            break
        apex = apex - 1 if left > right else apex + 1
    ymax = y[apex]
    if not np.isfinite(ymax) or ymax <= 0:
        return _NO_FIT

    lo = _peak_side(y, apex, -1, ymax)
    hi = _peak_side(y, apex, +1, ymax)
    if lo is None or hi is None:                  # never returned to baseline
        return _NO_FIT
    tt, yy = t[lo:hi + 1], y[lo:hi + 1]
    ok = ~np.isnan(yy)
    tt, yy = tt[ok], yy[ok]
    if np.count_nonzero(yy > 0) < MIN_PEAK_PTS or len(yy) < MIN_PEAK_PTS + 1:
        return _NO_FIT

    cycle = float(np.median(np.diff(t)))
    span = float(tt[-1] - tt[0])
    if not (np.isfinite(cycle) and cycle > 0 and span > 0):
        return _NO_FIT
    s_lo, s_hi = 0.25 * cycle, span
    s0 = float(np.clip(np.count_nonzero(yy >= 0.5 * ymax) * cycle / SIGMA_TO_FWHM,
                       s_lo * 1.01, s_hi * 0.99))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            (a, mu, sigma), _ = curve_fit(
                _gauss, tt, yy, p0=(ymax, t[apex], s0),
                bounds=([0.0, tt[0], s_lo], [5.0 * ymax, tt[-1], s_hi]),
                maxfev=2000)
    except (RuntimeError, ValueError):
        return _NO_FIT
    # a width pinned to its bounds is not a measured width
    if not (s_lo * 1.01 < sigma < s_hi * 0.99):
        return _NO_FIT
    resid = yy - _gauss(tt, a, mu, sigma)
    ss_tot = float(np.sum((yy - yy.mean()) ** 2))
    if ss_tot <= 0:
        return _NO_FIT
    return float(SIGMA_TO_FWHM * sigma), 1.0 - float(np.sum(resid ** 2)) / ss_tot


def compute_ms1_fwhm(dat, spectra, mz_ppm, mass_tag=None, SILAC=None, timeplex=False,
                     vote_sigma=1.0, elution_fwhm=None):
    """Gaussian MS1 FWHM and fit R² for each confident row of ``dat``.

    Returns {(untag_prec, channel, silac_channel): (fwhm, r2)} for the non-decoy rows
    with BestChannel_Qvalue < FWHM_QVALUE ((NaN, NaN) where no peak could be fitted).
    """
    fdc = dat[(~dat["is_decoy"].astype(bool)) &
              (dat["BestChannel_Qvalue"] < FWHM_QVALUE)].reset_index(drop=True)
    if len(fdc) == 0:
        return {}

    if mass_tag:
        tag = mass_tag
    else:   # same placeholder ms1_quant uses for label-free data
        tag = massTag(rules="nK", base_mass=0.00, delta=[0], channel_names=["0"],
                      name="no_tag", compositions=mass.Composition())

    ms1_spectra = spectra.ms1scans
    ms1_spec_idxs = np.array([s.scan_num for s in ms1_spectra])
    ms1_rt = {s.scan_num: float(s.RT) for s in ms1_spectra}
    all_rt = np.sort(np.fromiter(ms1_rt.values(), dtype=float))
    cycle = float(np.median(np.diff(all_rt))) if len(all_rt) > 1 else 0.0
    # cap on how far a profile may grow each side of its apex; the profile normally
    # stops much earlier, once the peak is back at baseline
    try:
        ef = float(np.nanmedian(np.atleast_1d(np.asarray(elution_fwhm, dtype=float))))
    except (TypeError, ValueError):
        ef = np.nan
    half_window = max(0.5, 6.0 * ef) if np.isfinite(ef) and ef > 0 else 1.0
    max_scans = int(min(200, max(VOTE_WINDOW, np.ceil(half_window / cycle)))) if cycle > 0 else VOTE_WINDOW
    logger.info("")
    logger.info(f"MS1 FWHM (--ms1_fwhm): Gaussian fit to the plex-fit MS1 profile of "
                f"{len(fdc):,} rows (BestChannel_Qvalue < {FWHM_QVALUE}); profiles grow up to "
                f"{max_scans} MS1 scans ({max_scans * cycle:.2f} min) each side of the apex")

    group_cols = ["untag_seq", "z", "time_channel"] if timeplex else ["untag_seq", "z"]
    fdc_group = fdc.groupby(group_cols)
    num_iso = config.num_iso_ms1
    fwhm_by_key = {}
    for key in tqdm.tqdm(list(fdc_group.groups)):
        tag_group = fdc_group.get_group(key)
        prec_seqs, prec_mzs, prec_z, _rt, _top, largest_coeff_scans, time_channel = \
            get_seqs_and_mzs(fdc_group, timeplex, tag, key, SILAC)
        vote_scans, _ = minmax_spec_window(largest_coeff_scans, ms1_spec_idxs, ms1_spectra,
                                           spectra, VOTE_WINDOW)
        voted_apex = _voted_apex(tag_group, vote_scans, vote_sigma)
        scans, _ = minmax_spec_window(largest_coeff_scans, ms1_spec_idxs, ms1_spectra,
                                      spectra, max_scans)
        scans = np.asarray(scans)
        apex_pos = int(np.searchsorted(scans, voted_apex))
        lo = max(0, apex_pos - max_scans)
        scans = scans[lo:apex_pos + max_scans + 1]
        apex_pos -= lo

        group_iso = [compute_isotopes(s, m, prec_z, num_iso, [tag, SILAC])
                     for s, m in zip(prec_seqs, prec_mzs)]
        p_lo, p_hi, coeffs = _grow_profile(ms1_spectra, ms1_spec_idxs, scans, apex_pos,
                                           group_iso, mz_ppm)
        t = np.array([ms1_rt[s] for s in scans[p_lo:p_hi + 1]])
        for ci, prec_seq in enumerate(prec_seqs):
            ch_key = (prec_seq, prec_z, time_channel) if timeplex else (prec_seq, prec_z)
            fwhm_by_key[ch_key] = _gaussian_fwhm(t, coeffs[:, ci], apex_pos - p_lo)

    row_keys = (zip(fdc["seq"], fdc["z"], fdc["time_channel"]) if timeplex
                else zip(fdc["seq"], fdc["z"]))
    out = {}
    for rk, up, ch, sc in zip(row_keys, fdc["untag_prec"], fdc["channel"], fdc["silac_channel"]):
        out[(up, ch, sc)] = fwhm_by_key.get(tuple(rk), _NO_FIT)
    vals = np.array(list(out.values()), dtype=float).reshape(-1, 2)
    ok = np.isfinite(vals[:, 0])
    w, r2 = vals[ok, 0], vals[ok, 1]
    logger.info(f"MS1 FWHM: fitted {int(ok.sum()):,}/{len(vals):,} rows"
                + (f"; median {np.median(w):.4f}, IQR "
                   f"{np.percentile(w, 25):.4f}-{np.percentile(w, 75):.4f}; median R² "
                   f"{np.median(r2):.3f}, {int((r2 < 0.5).sum()):,} with R² < 0.5"
                   if ok.any() else ""))
    return out


def add_ms1_fwhm(dat, spectra, mz_ppm, mass_tag=None, SILAC=None, timeplex=False,
                 vote_sigma=1.0, elution_fwhm=None):
    """``dat`` with two extra columns, MS1_FWHM and MS1_FWHM_R2 (NaN for decoys and
    for rows with BestChannel_Qvalue >= 0.01)."""
    fwhm = compute_ms1_fwhm(dat, spectra, mz_ppm, mass_tag=mass_tag, SILAC=SILAC,
                            timeplex=timeplex, vote_sigma=vote_sigma,
                            elution_fwhm=elution_fwhm)
    confident = ((~dat["is_decoy"].astype(bool)) &
                 (dat["BestChannel_Qvalue"] < FWHM_QVALUE)).to_numpy()
    fits = [fwhm.get(k, _NO_FIT) if ok else _NO_FIT for ok, k in
            zip(confident, zip(dat["untag_prec"], dat["channel"], dat["silac_channel"]))]
    dat[FWHM_COL] = [f for f, _ in fits]
    dat[R2_COL] = [r for _, r in fits]
    return dat
