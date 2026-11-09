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
sequence carries a tag, and which residues an ion spans), never by comparing m/z.  Counts are monoisotopic: the number of isotope peaks a library
entry has is set by --num_iso, not by evidence.  Intensities come both ways.
"""

import os

import numpy as np
import polars as pl

from src.logger import logger
from src.utils.frag_encoding import get_index, get_ion_type, is_isotope
from src.utils.parse_peptides import parse_peptide

# b, a and c ions span residues 1..i from the N-terminus; y, x and z span the
# last i residues.  The codes are frag_encoding's _ION_TYPE_TO_INT
_NTERM_IONS = (0, 2, 3)

FEATURE_FOLDER = "channel_fdr"
FEATURE_FILE = "channel_features.parquet"

# Carried into the feature table so it stands on its own: the row's identity and
# the scores the channel-level FDR is built on top of
_ID_COLUMNS = ["file_name", "seq", "untag_seq", "untag_prec", "z", "channel",
               "is_decoy_channel", "Qvalue", "BestChannel_Qvalue", "PredVal"]


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
        # intensity sums, with and without the isotope peaks
        "chan_frag_int_sum": chan_int,
        "shared_frag_int_sum": shared_int,
        "chan_frag_int_mono_sum": chan_int_mono,
        "shared_frag_int_mono_sum": shared_int_mono,
        # How much of the row's evidence is channel-specific.  A fraction, not
        # the ratio channel-specific / shared: the same ordering (f = r / (1 + r)),
        # so a tree sees the same splits, but bounded and finite when nothing is
        # shared, so nothing needs transforming or clamping.  (The reference
        # script used a clamped log2 of the ratio.)  0 / 0 is NaN.
        "chan_frag_fraction": _ratio(n_chan, n_chan + n_shared),                        # mono
        "chan_intensity_fraction": _ratio(chan_int, chan_int + shared_int),             # incl. isotopes
        "chan_mono_intensity_fraction": _ratio(chan_int_mono, chan_int_mono + shared_int_mono),  # mono
        # observed against library over the library's whole list, monoisotopic
        "cosine_chan": cosine_chan,
        "cosine_all": cosine_all,
        "cosine_chan_minus_all": cosine_chan - cosine_all,
    }


# A row whose library entry is missing: every feature unknown, counts included
# (a 0 would read as "nothing could be matched", which is a real value)
_EMPTY_ROW = dict.fromkeys(_row_features(np.zeros(0, np.int32), np.zeros(0), np.zeros(0, np.int32),
                                         np.zeros(0), 1, 0, 0))


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


def build_channel_features(ids, list_cols, library, mass_tag, fdr_threshold):
    """The channel-evidence table for the rows of *ids* worth scoring.

    Those are the target rows of every plex set that passed the run's FDR
    (BestChannel_Qvalue, which is one value for a whole set, so a set is kept or
    dropped together).  The decoy channel's rows are kept: they are the null the
    channel-level FDR is measured against.

    *list_cols* holds the apex scan's matched fragments (read straight from
    polars rather than merged into the frame, which is why this runs before that
    merge).  Returns a polars DataFrame, one row per scored precursor-channel.
    """
    keep = (~ids["is_decoy"].to_numpy()) & (ids["BestChannel_Qvalue"].to_numpy() < fdr_threshold)
    scored = ids.loc[keep, [c for c in _ID_COLUMNS if c in ids.columns] + ["__fdc_idx"]]
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
             .join(list_cols.select(["__fdc_idx", "frag_names", "obs_int"]),
                   on="__fdc_idx", how="left", maintain_order="left"))
    if len(frags) != len(scored):
        raise ValueError(f"{len(frags)} fragment rows for {len(scored)} scored rows: "
                         f"__fdc_idx is not unique in the fragment list columns")
    matched_codes = frags["frag_names"].to_list()
    matched_obs = frags["obs_int"].to_list()

    # One library entry per row; a precursor the library no longer has (it is
    # keyed on the tagged sequence) leaves that row's fragment features null
    keys = list(zip(scored["seq"], scored["z"].astype(float)))
    lib_idx = [library.key_to_idx.get(k) for k in keys]
    n_missing = sum(i is None for i in lib_idx)
    if n_missing:
        logger.warning(f"{n_missing} of {len(lib_idx)} scored precursors are not in the library: "
                       f"their fragment features are null")

    rows, context = [], []
    n_repeated = 0
    for idx, seq, (residues, n_tags, n_residues), codes, obs in zip(
            lib_idx, scored["seq"], sites, matched_codes, matched_obs):
        context.append(_peptide_context(residues, n_tags, n_residues))
        if idx is None or codes is None:
            rows.append(dict(_EMPTY_ROW))
            continue
        codes, obs = np.asarray(codes, dtype=np.int32), np.asarray(obs, dtype=float)
        if codes.size != obs.size:
            raise ValueError(f"{seq}: {codes.size} matched fragment names but "
                             f"{obs.size} observed intensities")
        lib_codes, lib_int, repeated = first_of_each(
            np.asarray(library.get_ordered_frag_codes(idx), dtype=np.int32),
            np.asarray(library.get_frag_intensities(idx), dtype=float))
        codes, obs, _ = first_of_each(codes, obs)
        n_repeated += repeated
        rows.append(_row_features(lib_codes, lib_int, codes, obs,
                                  n_residues, residues[0], residues[-1]))
    if n_repeated:
        logger.warning(f"{n_repeated} scored precursors list a fragment identity more than once "
                       f"in the library: the first is used")

    return pl.concat([pl.from_pandas(scored.drop(columns=["__fdc_idx"]).reset_index(drop=True)),
                      pl.DataFrame(context), pl.DataFrame(rows, infer_schema_length=None)],
                     how="horizontal")


def write_channel_features(ids, list_cols, library, mass_tag, results_folder, fdr_threshold):
    """build_channel_features, written to <results_folder>/channel_fdr/.

    Does nothing without a decoy channel: the features exist for the
    channel-level FDR, which has nothing to measure against without one (label
    free, a tag searching one channel, or --decoy_channel none).  Also nothing
    when the apex scan's fragments were not kept apart (*list_cols* None).
    """
    if mass_tag is None or mass_tag.decoy_channel is None or list_cols is None:
        return None
    features = build_channel_features(ids, list_cols, library, mass_tag, fdr_threshold)
    folder = os.path.join(results_folder, FEATURE_FOLDER)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, FEATURE_FILE)
    features.write_parquet(path)
    logger.info(f"Channel features for {len(features)} precursor-channels written to "
                f"{os.path.abspath(path)}")
    return features
