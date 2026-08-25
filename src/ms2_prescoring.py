"""
This Source Code Form is subject to the terms of the Oxford Nanopore
Technologies, Ltd. Public License, v. 1.0.  Full licence can be found
at https://github.com/ParallelSquared/JMod/blob/main/LICENSE.txt
"""

"""
Channel-unique MS2 quant and empirical-library features, computed BEFORE scoring.

Both of these were previously produced after ``score_precursors``:

* ``coeff_ChannelUnique`` -- fitted only for rows already past
  ``BestChannel_Qvalue < q_gate``
* the cross-channel empirical-library features -- not produced at all

Two reasons to move them earlier and drop the gate:

1. **They are scoring features.**  A column that exists only for confident
   targets predicts the label perfectly.  To be usable by the scorer they must
   exist for every candidate, decoys included.
2. **``coeff_ChannelUnique`` is the better MS2 normalization primary.**
   ``coeff`` is ratio-compressed (0.65 of the designed fold change on JD0413
   against 0.97), and the paired stage of the RT normalizer measures precisely
   the deviation compression attenuates, so a compressed primary under-corrects.
   Estimating the curve from ``coeff_ChannelUnique`` instead cut the
   across-time-channel RT drift of ``coeff`` by 22% and of
   ``coeff_ChannelUnique`` by 16%, and improved yeast accuracy for both.
   Ungated, it covers 99.0% of the normalizer's reference set against 91.0%
   gated.

The obstacle is memory, not logic.  ``get_large_prec`` deliberately holds the
per-fragment list columns out of the working frame (``fdc_list_cols``, keyed by
``__fdc_idx``) and merges them back only after protein FDR, so the heavy phase
runs on a slim frame -- that guard is what keeps large searches inside RAM.
Merging them early to reach the fragments would undo it.

So this joins the list columns back **one precursor-chunk at a time**, computes
everything that needs fragments, and returns a small scalar frame to merge into
``fdc``.  Peak extra memory is one chunk of fragment lists, not the whole table.
Chunking is by a hash of ``untag_prec`` so that every channel of a precursor
lands in the same chunk -- the channel-unique mask and the empirical library
both need the full plex group together.
"""
import hashlib
import os

import numpy as np
import pandas as pd

try:
    from src.logger import logger
except Exception:  # pragma: no cover
    import logging
    logger = logging.getLogger(__name__)

IDX = "__fdc_idx"
LIST_COLS = ("frag_names", "frag_mz", "frag_int")
# "mz" and "z" are the precursor m/z and charge: the isotope-crosstalk
# correction needs them to work out how much of a contaminating channel's
# envelope the isolation window actually admits (see
# ms2_isotope_xtalk.isolated_envelope_fraction). They are cheap scalars, and
# leaving them out silently disables that truncation -- the slim frame simply
# would not carry them and the correction falls back to assuming the whole
# envelope got in.
SCALARS = ("untag_prec", "time_channel", "spec_id", "window_mz", "rt",
           "frac_lib_int", "is_decoy", "BestChannel_Qvalue", "seq", "mz", "z")


def _modules():
    """The two workers, importable in-package or by file path."""
    def _load(name):
        try:
            return __import__(f"src.{name}", fromlist=["*"])
        except Exception:
            import importlib.util
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                name + ".py")
            spec = importlib.util.spec_from_file_location(name, path)
            mod = importlib.util.module_from_spec(spec)
            # register under the bare name: ms2_common_apex and
            # ms2_xchannel_lib fall back to `from ms2_unique_quant import ...`
            # when the src package is not importable
            import sys as _sys
            _sys.modules.setdefault(name, mod)
            spec.loader.exec_module(mod)
            return mod
    return (_load("ms2_unique_quant"), _load("ms2_xchannel_lib"),
            _load("ms2_common_apex"))


def add_prescoring_ms2(fdc: pd.DataFrame, fdc_list_cols=None, spectra=None,
                       plexDIA: bool = True, ppm: float = 10.0,
                       n_chunks: int = 12, mz_tol: float = 0.01,
                       want_xchannel: bool = True,
                       want_areas: bool = True,
                       n_adjacent: int = 1,
                       shared_frag_mode: str = "proportional",
                       tag=None, isotope_correct: bool = True,
                       n_iso: int = 5,
                       mz_tol_da: float = 0.015,
                       purity_diag: bool = False,
                       min_siblings: int = 0,
                       weighted_apex: bool = True) -> pd.DataFrame:
    """Add ``coeff_ChannelUnique`` and the ``xchan_*`` features to ``fdc``.

    ``fdc`` is the slim working frame; ``fdc_list_cols`` the polars frame the
    fragment lists were held back in.  Returns ``fdc`` with the new scalar
    columns, having never held more than one chunk of fragment lists at a time.
    """
    muq, xch, cap = _modules()

    if spectra is None or fdc_list_cols is None or IDX not in fdc.columns:
        logger.warning("pre-scoring MS2 features skipped: need spectra, the "
                       "held-back list columns and __fdc_idx")
        return fdc
    try:
        import polars as pl
    except Exception:
        logger.warning("pre-scoring MS2 features skipped: polars unavailable")
        return fdc

    have = set(fdc_list_cols.columns)
    if not set(LIST_COLS).issubset(have):
        logger.warning(f"pre-scoring MS2 features skipped: list frame lacks "
                       f"{sorted(set(LIST_COLS) - have)}")
        return fdc

    scal = [c for c in SCALARS if c in fdc.columns]
    if "untag_prec" not in scal:
        logger.warning("pre-scoring MS2 features skipped: no untag_prec")
        return fdc

    # chunk by precursor so a plex group is never split
    uniq = fdc["untag_prec"].astype(str).unique()
    bucket = {p: hashlib.md5(p.encode()).digest()[0] % max(n_chunks, 1)
              for p in uniq}
    chunk_of = fdc["untag_prec"].astype(str).map(bucket).to_numpy()

    lazy = (fdc_list_cols.lazy() if hasattr(fdc_list_cols, "lazy")
            else fdc_list_cols)
    parts = []
    for c in range(max(n_chunks, 1)):
        rows = np.flatnonzero(chunk_of == c)
        if rows.size == 0:
            continue
        sub = fdc.iloc[rows][[IDX] + scal].copy()
        keys = sub[IDX].to_numpy()
        frag = (lazy.filter(pl.col(IDX).is_in(keys))
                    .select([IDX, *LIST_COLS]).collect().to_pandas())
        sub = sub.merge(frag, on=IDX, how="left")
        del frag

        # ONE pass.  ``add_xchannel_features`` performs the channel-unique fit
        # itself (it is the sibling weight) and now emits it, so calling
        # ``add_channel_unique_coeff`` as well would rebuild the same masks and
        # re-extract the same spectra to recompute a number already in hand.
        sub = xch.add_xchannel_features(sub, spectra=spectra, ppm=ppm,
                                        mz_tol=mz_tol,
                                        shared_frag_mode=shared_frag_mode,
                                        tag=tag, isotope_correct=isotope_correct,
                                        n_iso=n_iso, mz_tol_da=mz_tol_da,
                                        purity_diag=purity_diag,
                                        min_siblings=min_siblings,
                                        )

        # MS2 areas: the coefficient summed over a window centred on the apex
        # the whole plex group shares -- the analogue of plex_Area.  Done here
        # rather than in a post-scoring pass because this chunk already holds
        # the fragment lists; it still costs its own extraction per scan of the
        # window, so it roughly doubles this stage.
        if want_areas:
            sub = cap.add_common_apex_coeff(sub, spectra=spectra, plexDIA=plexDIA,
                                            ppm=ppm, n_adjacent=n_adjacent,
                                            mz_tol=mz_tol, q_gate=None,
                                            tag=tag,
                                            isotope_correct=isotope_correct,
                                            n_iso=n_iso, mz_tol_da=mz_tol_da,
                                            min_siblings=min_siblings,
                                            weighted_apex=weighted_apex,
)
        keep = ([IDX, xch.CU_QUANT] + list(xch.CU_FEATURES)
                + (list(xch.PURITY_DIAG) + list(xch.PURITY_SWEEP)
                   if purity_diag else [])
                + (list(xch.FEATURES) if want_xchannel else [])
                + ([cap.ALL_APEX, cap.ALL_AREA, cap.CU_APEX, cap.CU_AREA]
                   if want_areas else []))
        parts.append(sub[[k for k in keep if k in sub.columns]].copy())
        del sub

    if not parts:
        return fdc
    add = pd.concat(parts, ignore_index=True)
    del parts
    new = [c for c in add.columns if c != IDX and c in fdc.columns]
    if new:                       # never silently overwrite an existing column
        fdc = fdc.drop(columns=new)
    out = fdc.merge(add, on=IDX, how="left")
    n_cu = int(np.isfinite(pd.to_numeric(out.get(xch.CU_QUANT),
                                         errors="coerce")).sum())
    logger.info(f"pre-scoring MS2: {xch.CU_QUANT} fitted for {n_cu:,} of "
                f"{len(out):,} candidate rows (ungated, decoys included)"
                + (f"; {len(xch.FEATURES)} empirical-library features"
                   if want_xchannel else "")
                + ("; MS2 areas (coeff_Area, coeff_ChannelUnique_Area)"
                   if want_areas else ""))
    return out
