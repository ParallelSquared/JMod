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

"""Channel-level evidence: whether a particular tag channel is there, not just
the peptide.

A plexDIA set is one peptide searched once per channel.  The precursor-level
score says the peptide is in the run; it says much less about which channels it
is in, because every channel of a set shares most of its fragments.  What
separates the channels is the fragments that *do not*: a fragment spanning a tag
site carries the tag, so its m/z moves from channel to channel, while the rest
are identical in every channel and are matched by all of them equally.

Everything here is built on that split, decided by chemistry (where the searched
sequence carries a tag, and which residues an ion spans), never by comparing m/z.

MS1 adds the other half: ms1_quant fits every channel of a set jointly, scan by
scan, so each channel has its own coefficient trace across the set's MS1 fit
window.  A channel that is really there elutes as a peak there; an empty one is
flat or noise.  Counts are monoisotopic: the number of isotope peaks a library
entry has is set by --num_iso, not by evidence.  Intensities come both ways.
"""

import os

import numpy as np
import pandas as pd
import polars as pl

from src.logger import logger
from src.utils.frag_encoding import get_index, get_ion_type, is_isotope
from src.utils.parse_peptides import parse_peptide

# b, a and c ions span residues 1..i from the N-terminus; y, x and z span the
# last i residues.  The codes are frag_encoding's _ION_TYPE_TO_INT
_NTERM_IONS = (0, 2, 3)

FEATURE_FOLDER = "channel_fdr"
FEATURE_FILE = "channel_features.parquet"
FEATURE_IMPORTANCE_FILE = "feature_importance.csv"
# Each feature's share of the model's gain, over the CV folds: mean and spread
IMPORTANCE_COLUMN = "share_of_gain"
IMPORTANCE_SPREAD_COLUMN = "share_of_gain_sd"
# The channel FDR's outputs, written into the feature table after the fit
SCORE_COLUMNS = ["ChannelScore", "Channel_Qvalue"]

# Carried into the feature table so it stands on its own: the row's identity and
# the set-level score the rows were kept by.  The row's own score is PredVal, a
# feature; Qvalue is not carried, being a coarser step function of it.  These are
# the table's only columns that are not features
_ID_COLUMNS = ["file_name", "seq", "untag_seq", "untag_prec", "channel", "time_channel",
               "is_decoy_channel", "BestChannel_Qvalue"]

# ms1_quant's outputs the MS1 features read: this channel's plex-fit coefficient
# and the joint fit's quality at every scan of the set's fit window (";"-joined),
# the window's scans, this channel's MS1 apex in it, and the MS1 scan of the MS2
# apex.  Read, not written: the traces stay in all_IDs
_MS1_COLUMNS = ["plexfittrace_all", "plexfittrace_ps_all", "plexfittrace_spec_all",
                "ms1_apex_scan", "Ms1_spec_id"]
# A Gaussian fit to fewer points than this says nothing
MIN_MS1_POINTS = 5

# Per-row features JMod already computes, carried in as they are, apart from
# their placeholders (_PLACEHOLDERS).  Abundance (coeff, plex_Area, MS1_Area,
# MS1_Int, plexfitMS1) and channel identity (channel, window_mz, mz, prec_im,
# silac_channel, run_chan, channels_matched) are left out on purpose: abundance
# is what the experiment measures, and identity is the label.
_CORRELATION_SUMMARIES = ["mean_frag_corr", "median_frag_corr", "max_frag_corr", "min_frag_corr",
                          "std_frag_corr", "mean_top3_frag_corr", "frac_corr_above_0p5",
                          "mean_frag_mean_corr"]
_RANKED_CORRELATIONS = [f"top{i}_{kind}" for kind in ("frag_mean_corr", "frag_sum_corr", "pair_corr")
                        for i in range(1, 11)]
_SET_MEDIANS = ["median_gof_stats", "median_scribe_scores", "median_max_matched_residuals",
                "median_manhattan_distances"]
_FROM_SET_MEDIANS = ["diff_gof_stats_from_median", "diff_scribe_scores_from_median",
                     "diff_max_matched_residuals_from_median", "diff_manhattan_distances_from_median"]
_CARRIED_FEATURES = [
    # the run's own score for the row (the discriminant behind Qvalue), and the
    # precursor's charge (also part of its identity, with seq)
    "PredVal", "z",
    # the precursor's scans and errors, and where it eluted
    "n_scans", "smoothness", "apex_pc1", "hyperscore", "mz_error", "rt_error", "im_error", "rt",
    # fragment elution correlations
    "n_corr_frags", *_CORRELATION_SUMMARIES, *_RANKED_CORRELATIONS,
    "mean_prec_frag_corr", "max_prec_frag_corr",
    # the apex spectrum's match to the library
    "gof_stats", "scribe_scores", "max_matched_residuals", "manhattan_distances",
    "fitted_spectral_contrasts", "cosine", "frac_shared_intensity",
    "frac_int_matched", "frac_int_pred", "frac_int_uniq", "frac_int_uniq_pred",
    "frac_int_matched_pred", "frac_int_matched_pred_sigcoeff", "frac_lib_int", "frac_dia_int",
    "b_counts", "y_counts", "longest_y_ions", "med_frag_error",
    "manhattan_distances_nearby_max", "max_matched_residuals_nearby_min",
    "gof_stats_nearby_min", "scribe_scores_nearby_min",
    # MS1.  gauss_r2_joint: one Gaussian fit to this channel's MS1 plex-fit trace
    # and its MS2 coefficients on the same cycles (fdr_analysis.gauss_r2_joint).
    # MS1_quality is built from it, and must never be carried itself: it exists
    # only for filtered_IDs, which leave the decoy channel out, so it is missing
    # for exactly the decoy channel's rows -- the label itself.  plexfitMS1_p:
    # the joint MS1 plex fit's quality at this channel's largest coefficient, which
    # needs no Gaussian (unlike plexfit_ps_at_apex's 5 points)
    "ms1_cor", "iso_cor", "traceproduct", "gauss_r2_joint", "plexfitMS1_p",
    # the plex set: each statistic's median over the set's channels, and this
    # channel's distance from it
    *_SET_MEDIANS, *_FROM_SET_MEDIANS,
]
# Read for the placeholder rules only, not written
_RULE_COLUMNS = ["channels_matched", "plexfitMS1"]
# Carried under another name.  JMod's cosine is the apex scan's whole fit --
# observed against predicted at the peaks of every candidate with a coefficient
# above 1, one value for the scan -- not this channel's match to its library,
# which is cosine_all
_RENAMED = {"cosine": "scan_fit_cosine"}
# The last residue (JMod's, off the stripped sequence), written as last_aa_code:
# its place in _AMINO_ACIDS, -1 for anything else.  An ordinal, as the reference
# had it: a tree can still split off any one residue
_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# Where JMod writes a value for something undefined.  ms1_quant merges its
# outputs back with fillna(0) over every column of the frame, so anything NaN
# before it is 0 after it; the rest are JMod's own markers.  Left in, a
# placeholder reads as a real value (0 is the best traceproduct), and every
# comparison between channels made with it is wrong.  {column: the rows where it
# is a placeholder}, each checked against a real run: the conditions hold
# exactly where they key off another column.  Evaluated together, on the values
# as JMod wrote them.
_PLACEHOLDERS = {
    # Pairwise summaries, undefined below two correlatable fragments.  Their
    # other 0s are real: the std of one pair, no pair above 0.5
    **{c: pl.col("n_corr_frags") < 2 for c in _CORRELATION_SUMMARIES},
    # Ranked slots: -1 = fewer than i fragments, or pairs
    **{c: pl.col(c) == -1 for c in _RANKED_CORRELATIONS},
    # A Pearson r is never exactly 0: the precursor was not seen in MS2, or the
    # MS1 trace was flat.  traceproduct is log10(product of rescaled
    # correlations + 1e-6), in [-6, 4.3e-7], so an exact 0 is always fillna's
    **{c: pl.col(c) == 0 for c in ("mean_prec_frag_corr", "max_prec_frag_corr",
                                   "ms1_cor", "iso_cor", "traceproduct")},
    # Undefined for a single-scan precursor, which is exactly where they are 0
    **{c: pl.col("n_scans") == 1 for c in ("smoothness", "apex_pc1")},
    # -1: a set with one channel, so nothing to share
    "frac_shared_intensity": pl.col("frac_shared_intensity") == -1,
    # A set with one channel has no median to measure against
    **{c: pl.col("channels_matched") == 1 for c in _SET_MEDIANS + _FROM_SET_MEDIANS},
    # NaN before fillna: no MS1 peak within tolerance (without --require_ms1),
    # no ion mobility (every row of a run without it), no fragment errors
    **{c: pl.col(c) == 0 for c in ("mz_error", "im_error", "med_frag_error")},
    # No MS1 plex fit for the channel: its largest coefficient is 0, so there is
    # no apex and argmax picks the window's first scan, whose quality says nothing
    # about this channel.  Its own 0 is fillna's (a real one is a Pearson r, never
    # exactly 0), and only happens with the first
    "plexfitMS1_p": (pl.col("plexfitMS1") == 0) | (pl.col("plexfitMS1_p") == 0),
}


def null_placeholders(frame):
    """*frame* with every _PLACEHOLDERS value made null.  The masks are all
    evaluated on the incoming values, so a rule keyed off another column sees
    that column before it is nulled."""
    return frame.with_columns(pl.when(mask).then(None).otherwise(pl.col(c)).alias(c)
                              for c, mask in _PLACEHOLDERS.items())


def ms2_cycles(scan_numbers, windows):
    """The DIA cycle of every MS2 scan, as a table (spec_id, cycle): the k-th scan
    of an isolation window is cycle k, counting from 0.

    Counted per window, not from the MS1 scans, because the two need not line up:
    a method can put two MS1 scans in each cycle (MS1 every half cycle), and then
    a precursor's window comes round every second MS1 scan.  *windows* is each
    scan's isolation window (spectrum.ms1window).  Not checked on ion-mobility
    data, where one m/z window can be visited more than once a cycle at
    different mobilities.
    """
    scan_numbers = np.asarray(scan_numbers, dtype=np.int64)
    cycles = np.empty(len(scan_numbers), dtype=np.int64)
    seen = {}
    for i in np.argsort(scan_numbers, kind="stable"):
        window = tuple(windows[i])
        cycles[i] = seen.get(window, 0)
        seen[window] = cycles[i] + 1
    return pl.DataFrame({"spec_id": scan_numbers, "cycle": cycles})


def longest_consecutive_scans(scan_coeffs, scan_cycles, keys):
    """The longest run of consecutive DIA cycles in which each precursor has an
    MS2 coefficient, as {key tuple: run length}.

    *scan_coeffs* is the run's MS2 coefficients, one row per precursor and MS2
    scan (spec_id); JMod writes no zero coefficients, so a missing cycle is a
    gap.  *scan_cycles* is ms2_cycles' table.  A precursor in two overlapping
    windows is seen twice in one cycle, which counts once.
    """
    runs = (scan_coeffs.select(keys + [pl.col("spec_id").cast(pl.Int64)])
            .join(scan_cycles, on="spec_id", how="inner")
            .select(keys + ["cycle"]).unique()
            .sort(keys + ["cycle"])
            # constant along a run of consecutive cycles, new after each gap
            .with_columns(run=pl.col("cycle") - pl.int_range(pl.len()).over(keys))
            .group_by(keys + ["run"]).agg(pl.len().alias("length"))
            .group_by(keys).agg(pl.col("length").max()))
    return {tuple(row[:-1]): row[-1] for row in runs.iter_rows()}


def tag_sites(seq, tag_name):
    """Where the search put *tag_name* on the tagged sequence *seq*: (the 0-based
    residues carrying a tag, the number of tags, the number of residues).

    Read off the sequence's own annotations, as SpectrumLibraryStore.from_tagged
    does for a pre-tagged library when it computes each fragment's m/z shift --
    what the search did, not what the tag's rules predict.  The two differ for a
    library tagged other than by the rules.  parse_peptide attaches an N-terminal
    annotation to the first residue, so the N-terminus is site 0, and an
    N-terminal K under "nK" is one site carrying two tags.
    """
    marker = f"({tag_name}-"
    counts = [residue.count(marker) for residue in parse_peptide(seq)]
    return [i for i, n in enumerate(counts) if n], sum(counts), len(counts)


def channel_specific_mask(codes, n_residues, first_site, last_site):
    """True for each fragment of *codes* that spans a tag site, so that its m/z
    moves between channels; False for the fragments every channel shares.

    An N-terminal ion of index i covers residues 0..i-1, so it carries a tag iff
    the first site is below i.  A C-terminal ion of index i covers the last i
    residues, so it carries one iff the last site is at or past n_residues - i.
    """
    ion, idx = get_ion_type(codes), get_index(codes)
    return np.where(np.isin(ion, _NTERM_IONS),
                    idx >= first_site + 1,
                    idx >= n_residues - last_site)


def _cosine(observed, library):
    """Cosine of the observed intensities against the library's, over the
    library's whole fragment list: a fragment the library predicts and the run
    did not match counts as a zero, which is the point of scoring it this way.

    0 when the library has fragments of this class and none were matched -- a
    real value, and for the channel-specific class strong evidence against the
    channel.  NaN only when the library has no fragments of the class at all
    (e.g. no channel-specific ones), where the cosine is undefined.
    """
    library_norm = np.linalg.norm(library)
    if library.size == 0 or library_norm == 0:
        return np.nan
    observed_norm = np.linalg.norm(observed)
    if observed_norm == 0:
        return 0.0
    return float(np.dot(observed, library) / (observed_norm * library_norm))


def _ratio(numerator, denominator):
    """*numerator* over *denominator*, NaN where the denominator is zero: the
    fraction of nothing is not nothing, and a 0 here would read as a real low
    value to the model."""
    return float(numerator) / denominator if denominator else np.nan


def first_of_each(codes, values):
    """*codes* and *values* with every repeat of a fragment identity dropped, the
    first kept, and whether there was any.  A library can list one identity twice
    at different intensities; then the identity has no single intensity, and
    counting it twice would count one fragment as two."""
    _, first = np.unique(codes, return_index=True)
    if first.size == codes.size:
        return codes, values, False
    first = np.sort(first)
    return codes[first], values[first], True


def _row_features(lib_codes, lib_int, matched_codes, matched_obs,
                  n_residues, first_site, last_site):
    """Every per-row feature for one precursor-channel, as a dict.

    *lib_codes* / *lib_int* are the library entry's whole fragment list;
    *matched_codes* / *matched_obs* are the fragments matched at the apex scan,
    a subset of it.
    """
    lib_mono = ~is_isotope(lib_codes)
    lib_chan = channel_specific_mask(lib_codes, n_residues, first_site, last_site)
    n_lib_chan = int((lib_chan & lib_mono).sum())
    n_lib_shared = int((~lib_chan & lib_mono).sum())

    obs_mono = ~is_isotope(matched_codes)
    obs_chan = channel_specific_mask(matched_codes, n_residues, first_site, last_site)
    n_chan = int((obs_chan & obs_mono).sum())
    n_shared = int((~obs_chan & obs_mono).sum())

    # The observed intensity of every library fragment, zero where it was not
    # matched, so the cosines can be taken over the library's whole list
    order = np.argsort(matched_codes, kind="stable")
    sorted_codes, sorted_obs = matched_codes[order], matched_obs[order]
    at = np.clip(np.searchsorted(sorted_codes, lib_codes), 0, max(sorted_codes.size - 1, 0))
    found = (sorted_codes[at] == lib_codes) if sorted_codes.size else np.zeros(lib_codes.size, bool)
    obs_of_lib = np.where(found, sorted_obs[at] if sorted_obs.size else 0.0, 0.0)

    chan_int = float(matched_obs[obs_chan].sum())
    shared_int = float(matched_obs[~obs_chan].sum())
    chan_int_mono = float(matched_obs[obs_chan & obs_mono].sum())
    shared_int_mono = float(matched_obs[~obs_chan & obs_mono].sum())
    # The typical matched fragment of each class, so the count is out of it;
    # undefined for a class with nothing matched
    chan_median, shared_median = (float(np.median(matched_obs[side & obs_mono])) if (side & obs_mono).any()
                                  else np.nan for side in (obs_chan, ~obs_chan))

    cosine_chan = _cosine(obs_of_lib[lib_chan & lib_mono], lib_int[lib_chan & lib_mono])
    cosine_all = _cosine(obs_of_lib[lib_mono], lib_int[lib_mono])
    return {
        # counts, monoisotopic peaks only
        "n_chan_frags": n_chan,
        "n_shared_frags": n_shared,
        "n_lib_chan_frags": n_lib_chan,
        "n_lib_shared_frags": n_lib_shared,
        # how much of what could be matched was, monoisotopic
        "frac_chan_frags_matched": _ratio(n_chan, n_lib_chan),
        "frac_shared_frags_matched": _ratio(n_shared, n_lib_shared),
        "frac_frags_matched": _ratio(n_chan + n_shared, n_lib_chan + n_lib_shared),
        # How much of the row's evidence is channel-specific.  A fraction, not
        # the ratio channel-specific / shared: the same ordering (f = r / (1 + r)),
        # so a tree sees the same splits, but bounded and finite when nothing is
        # shared, so nothing needs transforming or clamping.  (The reference
        # script used a clamped log2 of the ratio.)  0 / 0 is NaN.  Fractions
        # only, never the intensities themselves: those are the channel's
        # abundance, which the experiment measures
        "chan_frag_fraction": _ratio(n_chan, n_chan + n_shared),                        # mono
        "chan_intensity_fraction": _ratio(chan_int, chan_int + shared_int),             # incl. isotopes
        "chan_mono_intensity_fraction": _ratio(chan_int_mono, chan_int_mono + shared_int_mono),  # mono
        # The same of the median matched fragment, mono: per fragment, apart from
        # how many matched
        "chan_median_intensity_fraction": _ratio(chan_median, chan_median + shared_median),
        # observed against library over the library's whole list, monoisotopic
        "cosine_chan": cosine_chan,
        "cosine_all": cosine_all,
        "cosine_chan_minus_all": cosine_chan - cosine_all,
    }


# A row whose library entry is missing: every feature unknown, counts included
# (a 0 would read as "nothing could be matched", which is a real value)
_EMPTY_ROW = dict.fromkeys(_row_features(np.zeros(0, np.int32), np.zeros(0), np.zeros(0, np.int32),
                                         np.zeros(0), 1, 0, 0))


def frag_error_features(codes, errors, n_residues, first_site, last_site, centre):
    """How far the matched fragments' mass errors sit from the run's calibration
    (*centre*), channel-specific and shared apart: the median |error - centre|
    of each, as med_frag_error takes it over all of them, monoisotopic only.

    A channel that is not there matches its channel-specific fragments by chance,
    anywhere in the tolerance window, while its shared fragments are the real
    channels' peaks.  The difference is given too because the comparison is the
    evidence -- a precursor's errors move together -- and a tree cannot take it.
    Null where a class has no matched fragment.
    """
    usable = ~is_isotope(codes) & np.isfinite(errors)
    chan = channel_specific_mask(codes, n_residues, first_site, last_site)
    deviation = np.abs(errors - centre)
    chan_error, shared_error = (float(np.median(deviation[usable & side])) if (usable & side).any() else np.nan
                                for side in (chan, ~chan))
    return {"chan_frag_error": chan_error,
            "shared_frag_error": shared_error,
            "chan_minus_shared_frag_error": chan_error - shared_error}


_EMPTY_ERRORS = dict.fromkeys(["chan_frag_error", "shared_frag_error", "chan_minus_shared_frag_error"])


def _peptide_context(sites, n_tags, n_residues):
    """The peptide's own tagging: the same in every channel of a set, but it sets
    how much the channels can differ at all.

    The tag's position is given as a distance from each end, counted from the end
    that end's fragment ladder is numbered from, so each is the number of tag-free
    ions on that ladder: b1..b<first_tag_from_n> and y1..y<last_tag_from_c> are
    the same in every channel.  last_tag_from_c == 0 is a tagged C-terminal
    residue, when every y ion moves between channels.
    """
    return {"n_tags": n_tags,
            "pep_len": n_residues,
            "first_tag_from_n": sites[0],
            "last_tag_from_c": n_residues - 1 - sites[-1]}


def _trace(value):
    """A ";"-joined ms1_quant trace as floats, or None where there is none:
    ms1_quant merges its outputs in with fillna(0), so a row with no MS1 fit
    window has a 0 there, not a trace."""
    if not isinstance(value, str) or value == "":
        return None
    return np.array(value.split(";"), dtype=float)


_MS1_FEATURES = ["ms1_trace_nonzero", "plexfit_gauss_r2", "plexfit_sigma", "plexfit_trace_cv",
                 "plexfit_ps_at_apex", "plexfit_ps_weighted", "ms1_ms2_apex_delta"]


def ms1_features(coeff_trace, ps_trace, spec_trace, ms1_apex_scan, ms1_spec_id,
                 min_points=MIN_MS1_POINTS):
    """The shape of one channel's MS1 plex-fit coefficient trace, as a dict.

    *coeff_trace* is the channel's coefficient at each scan of the set's MS1 fit
    window, *ps_trace* the joint fit's quality there (the same for every channel
    of the set), *spec_trace* the window's scans; the three are ms1_quant's
    ";"-joined strings, or its 0 where it fitted no window.

    - ms1_trace_nonzero: whether the channel has any positive coefficient
    - plexfit_gauss_r2, plexfit_sigma: a Gaussian put on the trace by its
      coefficient-weighted moments (sigma in MS1 cycles), and how well it fits
    - plexfit_trace_cv: the trace's coefficient of variation
    - plexfit_ps_at_apex, plexfit_ps_weighted: the joint fit's quality at this
      channel's own apex, and weighted by this channel's coefficients -- the
      quality itself is per set, so only these selections of it can tell the
      channels apart
    - ms1_ms2_apex_delta: this channel's MS1 apex minus the MS1 scan of its MS2
      apex, in scan numbers

    All null where ms1_quant fitted no window: its 0s there are placeholders,
    not an empty trace.  The shape features need min_points points and a
    positive coefficient somewhere.  The window is a contiguous run of the
    run's MS1 scans (ms1_cor_channels.minmax_spec_window, then a slice of it),
    so point i is cycle i.
    """
    out = dict.fromkeys(_MS1_FEATURES, np.nan)
    c, p = _trace(coeff_trace), _trace(ps_trace)
    if c is None or _trace(spec_trace) is None:
        return out
    if p is not None and p.size != c.size:
        raise ValueError(f"MS1 fit quality has {p.size} points for a trace of {c.size}")
    p = np.zeros(0) if p is None else p
    out["ms1_ms2_apex_delta"] = float(ms1_apex_scan) - float(ms1_spec_id)

    w = np.clip(c, 0, None)
    out["ms1_trace_nonzero"] = float(w.sum() > 0)
    if c.size < min_points or w.sum() <= 0:
        return out

    x = np.arange(c.size, dtype=float)
    mu = (w * x).sum() / w.sum()
    sigma = float(np.sqrt(max((w * (x - mu) ** 2).sum() / w.sum(), 1e-12)))
    g = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
    amplitude = (g @ c) / (g @ g) if (g @ g) > 0 else 0.0
    ss_tot = ((c - c.mean()) ** 2).sum()
    out["plexfit_gauss_r2"] = (float(np.clip(1.0 - ((c - amplitude * g) ** 2).sum() / ss_tot, 0.0, 1.0))
                               if ss_tot > 0 else np.nan)
    out["plexfit_sigma"] = sigma
    out["plexfit_trace_cv"] = float(w.std() / w.mean()) if w.mean() > 0 else np.nan

    finite = np.isfinite(p)
    if finite.any():
        apex = int(np.argmax(c))
        out["plexfit_ps_at_apex"] = float(p[apex]) if finite[apex] else np.nan
        weights = w * finite
        if weights.sum() > 0:
            out["plexfit_ps_weighted"] = float((weights * np.nan_to_num(p)).sum() / weights.sum())
    return out


def build_channel_features(ids, list_cols, scan_coeffs, scan_cycles, library, mass_tag, fdr_threshold):
    """The channel-evidence table for the rows of *ids* worth scoring.

    Those are the target rows of every plex set that passed the run's FDR
    (BestChannel_Qvalue, which is one value for a whole set, so a set is kept or
    dropped together).  The decoy channel's rows are kept: they are the null the
    channel-level FDR is measured against.

    *list_cols* holds the apex scan's matched fragments (read straight from
    polars rather than merged into the frame, which is why this runs before that
    merge); *scan_coeffs* the run's MS2 coefficients per scan and *scan_cycles*
    each MS2 scan's DIA cycle (ms2_cycles), for longest_consecutive_scans.  Returns a polars DataFrame,
    one row per scored precursor-channel.
    """
    missing = [c for c in _MS1_COLUMNS + _CARRIED_FEATURES + _RULE_COLUMNS + ["last_aa"]
               if c not in ids.columns]
    if missing:
        raise KeyError(f"Channel features need these columns of the run's IDs; missing {missing}")
    keep = (~ids["is_decoy"].to_numpy()) & (ids["BestChannel_Qvalue"].to_numpy() < fdr_threshold)
    scored = ids.loc[keep, [c for c in _ID_COLUMNS if c in ids.columns] + _CARRIED_FEATURES
                           + _RULE_COLUMNS + _MS1_COLUMNS + ["last_aa", "__fdc_idx"]]
    # Where each precursor is tagged, read off its sequence (tag_sites).  An
    # untagged one (no K under a K-only tag) was searched once: it has no channels
    # to tell apart, so it is not scored
    sites = [tag_sites(seq, mass_tag.name) for seq in scored["seq"]]
    tagged = np.array([n_tags > 0 for _, n_tags, _ in sites], dtype=bool)
    if not tagged.all():
        logger.info(f"Channel features: {int((~tagged).sum())} untagged precursors left out")
        scored = scored[tagged]
        sites = [s for s, keep_it in zip(sites, tagged) if keep_it]
    if scored.empty:
        logger.warning("No precursors passed the run's FDR: no channel features")
        return pl.DataFrame()

    # maintain_order: polars' join is otherwise free to reorder, which would hand
    # each row another precursor's fragments
    frags = (pl.from_pandas(scored[["__fdc_idx"]])
             .join(list_cols.select(["__fdc_idx", "frag_names", "obs_int", "frag_errors"]),
                   on="__fdc_idx", how="left", maintain_order="left"))
    if len(frags) != len(scored):
        raise ValueError(f"{len(frags)} fragment rows for {len(scored)} scored rows: "
                         f"__fdc_idx is not unique in the fragment list columns")
    matched_codes = frags["frag_names"].to_list()
    matched_obs = frags["obs_int"].to_list()
    matched_errors = frags["frag_errors"].to_list()
    # The run's fragment calibration: the median error over every precursor's
    # matched fragments, targets and decoys, as med_frag_error takes it
    all_errors = list_cols.select(pl.col("frag_errors").explode(empty_as_null=False).drop_nulls())["frag_errors"]
    centre = float(all_errors.median()) if all_errors.len() else 0.0

    # One library entry per row; a precursor the library no longer has (it is
    # keyed on the tagged sequence) leaves that row's fragment features null
    keys = list(zip(scored["seq"], scored["z"].astype(float)))
    lib_idx = [library.key_to_idx.get(k) for k in keys]
    n_missing = sum(i is None for i in lib_idx)
    if n_missing:
        logger.warning(f"{n_missing} of {len(lib_idx)} scored precursors are not in the library: "
                       f"their fragment features are null")

    # A precursor is one per time channel on timeplex
    scan_keys = ["seq", "z"] + (["time_channel"] if "time_channel" in scan_coeffs.columns else [])
    longest = longest_consecutive_scans(scan_coeffs, scan_cycles, scan_keys)
    longest = [longest.get((seq, float(z), *rest)) for seq, z, *rest
               in zip(*(scored[k] for k in scan_keys))]

    ms1 = [ms1_features(*row) for row in zip(*(scored[c] for c in _MS1_COLUMNS))]
    last_aa_code = [_AMINO_ACIDS.find(aa) if isinstance(aa, str) and len(aa) == 1 else -1
                    for aa in scored["last_aa"]]

    rows, errors, context = [], [], []
    n_repeated = 0
    for idx, seq, (residues, n_tags, n_residues), codes, obs, err in zip(
            lib_idx, scored["seq"], sites, matched_codes, matched_obs, matched_errors):
        context.append(_peptide_context(residues, n_tags, n_residues))
        if codes is None:
            rows.append(dict(_EMPTY_ROW))
            errors.append(dict(_EMPTY_ERRORS))
            continue
        codes = np.asarray(codes, dtype=np.int32)
        obs, err = np.asarray(obs, dtype=float), np.asarray(err, dtype=float)
        if not codes.size == obs.size == err.size:
            raise ValueError(f"{seq}: {codes.size} matched fragment names but {obs.size} "
                             f"observed intensities and {err.size} mass errors")
        codes, kept, _ = first_of_each(codes, np.column_stack([obs, err]))
        obs, err = kept[:, 0], kept[:, 1]
        # needs only the matched fragments, so not the library entry either
        errors.append(frag_error_features(codes, err, n_residues, residues[0], residues[-1], centre))
        if idx is None:
            rows.append(dict(_EMPTY_ROW))
            continue
        lib_codes, lib_int, repeated = first_of_each(
            np.asarray(library.get_ordered_frag_codes(idx), dtype=np.int32),
            np.asarray(library.get_frag_intensities(idx), dtype=float))
        n_repeated += repeated
        rows.append(_row_features(lib_codes, lib_int, codes, obs,
                                  n_residues, residues[0], residues[-1]))
    if n_repeated:
        logger.warning(f"{n_repeated} scored precursors list a fragment identity more than once "
                       f"in the library: the first is used")

    carried = null_placeholders(
        pl.from_pandas(scored.drop(columns=["__fdc_idx", "last_aa"] + _MS1_COLUMNS).reset_index(drop=True))
    ).drop(_RULE_COLUMNS).rename(_RENAMED)
    # hstack: one row each, so unequal heights are an error, not padding
    return (carried
            .hstack([pl.Series("longest_consecutive_scans", longest, dtype=pl.Int64)])
            .hstack(pl.DataFrame(context))
            .hstack([pl.Series("last_aa_code", last_aa_code, dtype=pl.Int64)])
            .hstack(pl.DataFrame(rows, infer_schema_length=None))
            .hstack(pl.DataFrame(errors, schema={name: pl.Float64 for name in _EMPTY_ERRORS}))
            .hstack(pl.DataFrame(ms1, schema={name: pl.Float64 for name in _MS1_FEATURES})))


def model_columns(features):
    """The columns of the channel feature table the channel FDR is fitted on:
    every one but the identity columns and its own outputs."""
    return [c for c in features.columns if c not in _ID_COLUMNS + SCORE_COLUMNS]


def run_channel_fdr(ids, list_cols, scan_coeffs, spectra, library, mass_tag, results_folder, fdr_threshold,
                    plots="off", workers=1):
    """The channel-level FDR for one run, into <results_folder>/channel_fdr/.

    Builds the channel feature table (build_channel_features), fits the model
    on it (channel_fdr_fit.fit_channel_fdr), adds its ChannelScore and
    Channel_Qvalue to the table, and writes:
      - FEATURE_FILE: the table
      - FEATURE_IMPORTANCE_FILE: each feature's share of the model's gain
      - the best channel by PredVal, IDs vs Channel_Qvalue, the IDs retained
        at *fdr_threshold* and the feature importance
        (channel_fdr_plots.plot_channel_fdr)
      - unless *plots* is "off", every column's ECDF too (in *workers*
        processes), and
        with "KR" every plot but the importance split by the C-terminal
        residue

    Returns the table, for channel_qvalues to put Channel_Qvalue onto the run's
    IDs.  Without a decoy channel there is nothing to measure the channels
    against (label free, a tag searching one channel, or --decoy_channel none),
    so it does nothing and returns None; likewise when the apex scan's
    fragments were not kept apart (*list_cols* None).
    """
    if mass_tag is None or mass_tag.decoy_channel is None or list_cols is None:
        return None
    scan_cycles = ms2_cycles([spectrum.scan_num for spectrum in spectra.ms2scans],
                             [spectrum.ms1window for spectrum in spectra.ms2scans])
    features = build_channel_features(ids, list_cols, scan_coeffs, scan_cycles, library, mass_tag,
                                      fdr_threshold)
    folder = os.path.join(results_folder, FEATURE_FOLDER)
    os.makedirs(folder, exist_ok=True)
    importance = None
    if features.height:
        from src.channel_fdr_fit import fit_channel_fdr
        fitted = fit_channel_fdr(features, model_columns(features))
        if fitted is None:
            features = features.with_columns(pl.lit(None, pl.Float64).alias(c) for c in SCORE_COLUMNS)
        else:
            score, q, importance = fitted
            # the decoy channel has no q-value: null, not NaN
            features = features.with_columns(pl.Series("ChannelScore", score, dtype=pl.Float64),
                                             pl.Series("Channel_Qvalue", q, dtype=pl.Float64).fill_nan(None))
            write_feature_importance(importance, folder)
    path = os.path.join(folder, FEATURE_FILE)
    features.write_parquet(path)
    logger.info(f"Channel features for {len(features)} precursor-channels written to "
                f"{os.path.abspath(path)}")
    if features.height:
        from src.channel_fdr_plots import plot_channel_fdr
        ecdfs = str(plots).lower() != "off"
        plot_channel_fdr(features, folder, importance=importance, split=plots if ecdfs else "all",
                         fdr_threshold=fdr_threshold, ecdfs=ecdfs, workers=workers)
    return features


def write_feature_importance(importance, folder):
    """*importance* (channel_fdr_fit.feature_importance's table) to
    *folder*/FEATURE_IMPORTANCE_FILE."""
    importance.to_csv(os.path.join(folder, FEATURE_IMPORTANCE_FILE))


def read_feature_importance(folder):
    """write_feature_importance's file back as its table, or None if there is none."""
    path = os.path.join(folder, FEATURE_IMPORTANCE_FILE)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, index_col="feature")


def channel_qvalues(ids, features):
    """Copies Channel_Qvalue from the channel feature table onto the run's IDs.

    *ids* is every row of the run (pandas: fdx_quant); *features* only the rows
    the channel FDR scored, so this is a lookup of each *ids* row in
    *features*: by its tagged sequence and charge (and time channel, on
    timeplex), which name one precursor-channel in a run -- a left join.
    Returns one value per *ids* row, in its order and with its index: the row's
    Channel_Qvalue if it was scored, null if not (decoys, the untagged, the
    sets that failed the run's FDR, the decoy channel, which has none) or if
    nothing was fitted.  A precursor-channel listed twice in *features* is an
    error, as the lookup would be ambiguous.
    """
    if features is None or "Channel_Qvalue" not in features.columns:
        return pd.Series(np.nan, index=ids.index)
    keys = ["seq", "z"] + (["time_channel"] if "time_channel" in features.columns else [])
    q = features.select(keys + ["Channel_Qvalue"]).to_pandas()
    if q.duplicated(keys).any():
        raise ValueError(f"Channel FDR: the channel features repeat a precursor-channel ({', '.join(keys)})")
    q["z"] = q["z"].astype(ids["z"].dtype)
    # a left merge keeps the left frame's rows, in their order
    merged = ids[keys].merge(q, on=keys, how="left")
    return pd.Series(merged["Channel_Qvalue"].to_numpy(dtype=float), index=ids.index)
