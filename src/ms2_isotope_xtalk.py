"""
This Source Code Form is subject to the terms of the Oxford Nanopore
Technologies, Ltd. Public License, v. 1.0.  Full licence can be found
at https://github.com/ParallelSquared/JMod/blob/main/LICENSE.txt
"""

"""
Isotope-envelope crosstalk between plexDIA channels, for MS2 channel-unique
quant.

mTRAQ (and the 9-plex PSMtag ladder built the same way) encodes each channel
step as a small number of 13C/15N/18O substitutions, chosen so the mass step
lands close to a whole number of natural isotope spacings.  For the 9-plex tag
the per-step delta is 2.0067 Da -- almost exactly 2 x the 13C isotope spacing
(1.00336 Da).  So channel N's own M+2 isotope peak sits within ~1 mDa of
channel (N+1)'s monoisotopic target m/z: no realistic resolving power tells
them apart, and a plain ppm-window sum at channel (N+1)'s target silently
integrates channel N's isotope tail along with it.

``ms2_unique_quant.py`` already restricts itself to fragments whose *nominal*
m/z differs across channels (``channel_unique_masks``), which correctly
excludes fragments superposed at the SAME m/z (the R-terminating-y-ion case).
It has no model for fragments that are nominally distinct but whose isotope
envelopes overlap -- this module adds that.

Approach
--------
``deconvolve_fragment_group`` solves one fragment key (e.g. "b3_1") jointly
across its sibling channels -- useful as a building block and for testing,
but on its own it means each channel's final ``coeff_ChannelUnique`` comes
from several small, independent per-key systems recombined through
``jmod_coefficient``'s separate weighted-sum step. A channel's TRUE
abundance is one quantity that should explain ALL of its channel-unique
fragments at once, the same way ``ms1_cor_channels.fit_channel_isotopes_numba``
fits every channel's whole isotope-expanded pattern in one joint solve at
the MS1 level -- but MS2's channels do not share one scan the way MS1's
isotope orders of a single precursor do (different plexDIA channels
generally apex at different RTs), so a single joint NNLS spanning every
channel's own rows either evaluates most of them at the wrong scan (their
real signal reads as near-zero noise there) or, restricted to just one
channel's rows, is underdetermined (one equation cannot pin down both that
channel's and a contaminator's abundance at once). Both were tried; both
were wrong (see git history / earlier revisions of this file for exactly
how, if useful context for a future change here).

``cascading_channel_fit`` is what's actually used: channels are processed
in ascending mass order (see the call sites in ``ms2_unique_quant.py`` and
``ms2_xchannel_lib.py``), so by the time channel N is fitted, every
lower-mass sibling that could contaminate it has ALREADY been fitted from
ITS OWN best scan -- a known constant, not a free unknown. Contamination is
then simple subtraction (each contaminator's own coefficient times its own
isotope-decay fraction at the aligned order) followed by the ORIGINAL
closed-form ``jmod_coefficient`` on the residual. A channel with no eligible
lower-mass contaminator reduces EXACTLY to ``jmod_coefficient`` on the raw
extraction (verified by test) -- this differs from the pre-existing fit only
by removing contamination that was never physically part of the channel's
own signal, not by changing the underlying formula.

A channel need not have independently matched a given fragment to be
included as a contamination source -- see ``channel_label_from_seq``; every
channel's theoretical target position for any key is derivable from a
single matched sibling's own position plus the tag's exact per-channel
mass.

The isotope pattern for a given (fragment composition, channel, charge) is
deterministic, so it is cached -- the same fragment key recurs across the
group's several rows (RT-adjacent scans, area variants, etc).
"""
import re
from functools import reduce

import numpy as np
from scipy.optimize import nnls

from brainpy import isotopic_variants

from src.iso_functions import fragment_seq, get_seq_comp
from src.utils.frag_encoding import decode_frag_name

_ISO_SUFFIX = re.compile(r"_iso\d+$")

# TEMPORARY instrumentation: why does the cascade decline to subtract?
from collections import defaultdict as _dd
_S = _dd(float)


def dump_stats(log=None):
    n = max(_S["calls"], 1.0)
    p = max(_S["positions"], 1.0)
    lines = [
        "=== cascading_channel_fit instrumentation ===",
        f"  calls                          {_S['calls']:,.0f}",
        f"  mean rows in same-scan group   {_S['group_rows_total']/n:.2f}",
        f"  ladder rungs below channel     {_S['ladder_rungs_total']/n:.2f}",
        f"     resolved from a fitted row  {_S['rung_from_fit']:,.0f}",
        f"  fragment positions examined    {_S['positions']:,.0f}",
        f"  (j,pos) pairs considered       {_S['pairs_considered']:,.0f}",


        f"     skipped OUT OF TOLERANCE    {_S['skip_tolerance']:,.0f}"
        + (f"  (mean min|dm| = "
           f"{_S['skip_tol_mindiff']/max(_S['skip_tolerance'],1):.4f} Da)"
           if _S['skip_tolerance'] else ""),
        f"     skipped no-lib              {_S['skip_no_lib']:,.0f}",
        f"     APPLIED                     {_S['applied']:,.0f}"
        + (f"  (mean isotope order {_S['applied_order']/max(_S['applied'],1):.2f})"
           if _S['applied'] else ""),
        f"  position target from sibling   {_S['pos_from_sibling']:,.0f}",
        f"  position derived from mass     {_S['pos_derived']:,.0f}",
        f"  total intensity subtracted     {100*_S['subtracted']/max(_S['obs_total'],1):.3f}% "
        f"of observed",
    ]
    out = "\n".join(lines)
    if log is not None:
        log.info("\n" + out)
    else:
        print(out)
    return out


def normalize_frag_name(name):
    """A fragment "name" in the pre-scoring (ungated) candidate frame is
    sometimes still the raw packed int32 code as a string (e.g. "16537"
    instead of "b12_1_iso1") -- ``fdc_list_cols`` stores it that way for
    memory, and only the final report decodes it. ``channel_unique_masks``
    and ``jmod_coefficient`` never need to parse the string, so this was
    invisible to them; ``fragment_seq``/``split_frag_name`` do, and choke on
    a bare number. Decode it back to the real name so downstream parsing
    sees the same string the final report would have."""
    if isinstance(name, str) and name.lstrip("-").isdigit():
        return decode_frag_name(int(name))
    return name


def base_frag_key(name):
    """Strip a "_isoN" suffix, so an already isotope-expanded library entry
    (e.g. from ``iso_functions.gen_isotopes_dict``) collapses to the same key
    as its monoisotopic sibling."""
    return _ISO_SUFFIX.sub("", name)


def seq_for_label(mod_seq, tag, label):
    """Retag a modified sequence onto a different channel.

    A channel that JMod never proposed as a candidate still has its peptide in
    the spectrum, and its isotope envelope depends on the tag's
    channel-specific composition -- so to model it as a contamination source
    we need its modified sequence, which is just this peptide's with every tag
    annotation moved to that channel.
    """
    return re.sub(f"{re.escape(tag.name)}-\\d+", f"{tag.name}-{label}", mod_seq)


def tag_ladder(tag):
    """[(label, mass)] for every channel of the tag, ascending by mass."""
    out = []
    for key, m in tag.mass_dict.items():
        if not key.startswith(f"{tag.name}-"):
            continue
        lbl = key[len(tag.name) + 1:]
        if lbl.isdigit():
            out.append((lbl, float(m)))
    return sorted(out, key=lambda t: t[1])


def channel_label_from_seq(mod_seq, tag):
    """The channel label a tagged sequence belongs to, e.g. "8" from
    ``"L(PSMtag_9plex-8)DLPDEEK(PSMtag_9plex-8)"``.

    Every tag annotation in one peptide's sequence carries the same channel,
    so the first match is enough. Needed to derive a SIBLING channel's own
    theoretical fragment position without requiring that channel to have
    independently matched the fragment itself -- see
    ``deconvolve_fragment_group``'s docstring for why that matters.
    """
    m = re.search(f"{re.escape(tag.name)}-(\\d+)", mod_seq)
    if m is None:
        raise ValueError(f"no {tag.name!r} channel annotation in {mod_seq!r}")
    return m.group(1)


def fragment_isotope_pattern(mod_seq, base_ion_type, tag, n_iso=5, cache=None):
    """Theoretical isotope envelope of one fragment, one tagged channel.

    Parameters
    ----------
    mod_seq : str
        Full modified peptide sequence, tag annotations included, e.g.
        ``"E(PSMtag_9plex-4)MAAAPPGR"``.
    base_ion_type : str
        Fragment name without any "_iso" suffix, e.g. ``"b3_1"``.
    tag : massTag
        Needed for ``tag.name`` and ``tag.channel_comp``.
    n_iso : int
        Number of isotope orders to compute (0 = monoisotopic only).
    cache : dict or None
        Keyed by (composition string, charge, n_iso); isotope patterns are a
        pure function of elemental composition, so this is safe to share
        across an entire run.

    Returns
    -------
    mz_offset : np.ndarray, shape (n_iso,)
        Isotope order i's m/z minus the monoisotopic m/z (offset, not
        absolute, so the caller adds it to whatever m/z the library/report
        actually recorded -- keeps this consistent under small calibration
        differences between the two).
    rel_int : np.ndarray, shape (n_iso,)
        Isotope order i's intensity relative to order 0 (rel_int[0] == 1).
    """
    split_frag_seq, frag_info = fragment_seq(mod_seq, base_ion_type)
    frag_type, _frag_idx, loss, frag_z = frag_info
    frag_z = int(frag_z)
    frag_comp = get_seq_comp(split_frag_seq, frag_type,
                             neutral_loss=(loss or None))

    tags = [t for aa in split_frag_seq
            for t in re.findall(f"\\(({tag.name}.*?)\\)", aa)]
    if tag.channel_comp is not None and tags:
        tag_comp = reduce(lambda x, y: x + y,
                          [tag.channel_comp[re.findall(f"{tag.name}-(\\d+)", t)[0]]
                           for t in tags])
        frag_comp = frag_comp + tag_comp

    key = None
    if cache is not None:
        key = (str(sorted(frag_comp.items())), frag_z, n_iso)
        hit = cache.get(key)
        if hit is not None:
            return hit

    isotopes = isotopic_variants(frag_comp, npeaks=n_iso, charge=frag_z)
    mono = isotopes[0]
    mz_offset = np.array([p.mz - mono.mz for p in isotopes])
    rel_int = np.array([p.intensity / mono.intensity for p in isotopes])

    if cache is not None:
        cache[key] = (mz_offset, rel_int)
    return mz_offset, rel_int


def amplitude_from_coeff(coeff, lib, mask, frac_lib_int):
    """Per-library-unit amplitude ``A`` behind a ``jmod_coefficient`` result.

    To subtract what channel j contributes at some m/z we need A_j, defined by
    ``observed_j ~= A_j * lib_j`` -- NOT its coefficient. The two are not
    proportional by ``total/f``, because ``jmod_coefficient`` carries an
    unmatched-library penalty row that has no counterpart in the physical
    signal. Writing ``S2 = sum(lib[matched & mask]^2)`` and
    ``total = sum(lib)``, substituting ``obs = A*lib`` into the closed form
    gives

        coeff = A * total * f * S2 / ( f^2 * S2 + (1-f)^2 * total^2 )

    so inverting for A means multiplying by the bracket, not just by f/total:

        A = coeff * ( f^2 * S2 + (1-f)^2 * total^2 ) / ( total * f * S2 )

    Treating ``A = coeff * f / total`` (which is what this module did before)
    is only correct when ``f == 1``. At the real median ``frac_lib_int`` of
    0.74 it underestimates A by ~2.4x, at the 25th percentile by ~8x, and at
    the 5th by ~45x -- so contamination was subtracted at a small fraction of
    its true size and the bleed survived almost untouched.
    """
    lib = np.asarray(lib, dtype=float)
    ok = np.isfinite(lib) & (lib > 0)
    if not ok.any() or not np.isfinite(coeff):
        return np.nan
    total = float(np.sum(lib[ok]))
    use = ok if mask is None else (ok & np.asarray(mask, dtype=bool))
    if total <= 0 or not use.any():
        return np.nan
    s2 = float(np.sum(lib[use] ** 2))
    f = (float(frac_lib_int)
         if np.isfinite(frac_lib_int) and 0 < frac_lib_int <= 1 else 1.0)
    denom = total * f * s2
    if denom <= 0:
        return np.nan
    return float(coeff * (f * f * s2 + (1.0 - f) ** 2 * total * total) / denom)


_PROTON = 1.007276466
_C13_STEP = 1.0033548
_AVERAGINE = dict(C=4.9384, H=7.7583, N=1.3577, O=1.4773, S=0.0417)
_AVERAGINE_MASS = 111.1254


def isolated_envelope_fraction(prec_mz, prec_z, window_lo, window_hi,
                               frag_mz, frag_z, prec_mz_contaminator=None,
                               order=None, cache=None, max_k=24):
    """Fraction of a contaminating channel's reachable material the isolation
    window actually admits.

    To put ``order`` extra neutrons on the target's fragment, the CONTAMINATOR's
    precursor must itself carry at least that many, and that isotopologue must
    be inside the window that was fragmented. Its isotopologue q sits at
    ``prec_mz_contaminator + q * 1.00335/z``, so the admitted set is

        q >= order            (cannot give the fragment more than it has)
        window_lo <= mz_q < window_hi

    and what survives is ``sum_k c_k`` over ``k = q - order``, where ``c`` is
    the isotope envelope of the fragment's COMPLEMENT (the rest of the
    peptide) -- because ``p_q = sum_r f_r c_(q-r)``.

    BOTH bounds matter, and the lower one is the one that bites:

      1-tag peptide (R-terminating): channels are one tag step apart, and that
      step IS 2 x 13C, so the contaminator's q=2 isotopologue lands exactly on
      the TARGET's own precursor -- inside the window by construction. Nearly
      everything is admitted (measured 0.915 in a 2.0 m/z window).

      2-tag peptide (K-terminating): channels are TWO tag steps apart, so the
      contaminator's q=2 isotopologue lands only halfway to the target --
      0.0034 m/z BELOW the window's lower edge, and excluded. Only q>=3
      survives, and q=2 was the dominant term. Admitted drops to 0.279, so
      assuming full admission over-subtracts by 3.6x.

    That is why K-terminating peptides degraded under the correction in
    JD1187's 2.0 m/z windows while improving in JD0588's 18 m/z ones: with a
    wide window both isotopologues are comfortably inside and the distinction
    never arises.

    Simplification kept from before: the window is treated as a hard cut. Real
    quadrupole transmission rolls off, so this slightly over-corrects at the
    edges; a measured transmission profile is the obvious refinement.

    Returns 1.0 whenever the inputs are unusable, so a missing window can only
    fall back to the previous behaviour.
    """
    try:
        z = int(prec_z)
        r = 0 if order is None else int(order)
        if z < 1 or not np.isfinite(prec_mz):
            return 1.0
        if window_hi is None or not np.isfinite(window_hi):
            return 1.0
        d = _C13_STEP / z
        # where the contaminator's own monoisotopic precursor sits. Falling
        # back to "its q=order isotopologue is at the target" reproduces the
        # 1-tag geometry, which is what the old signature assumed.
        pc = (float(prec_mz) - r * d if prec_mz_contaminator is None
              else float(prec_mz_contaminator))
        q_hi = int(np.floor((float(window_hi) - pc) / d + 1e-9))
        q_lo = r
        if window_lo is not None and np.isfinite(window_lo):
            q_lo = max(r, int(np.ceil((float(window_lo) - pc) / d - 1e-9)))
    except (TypeError, ValueError, ZeroDivisionError):
        return 1.0
    k_lo, k_hi = q_lo - r, q_hi - r
    if k_hi < k_lo or k_hi < 0:
        return 0.0
    k_lo = max(0, k_lo)
    if k_lo == 0 and k_hi >= max_k:
        return 1.0
    try:
        prec_neutral = (float(prec_mz) - _PROTON) * z
        frag_neutral = (float(frag_mz) - _PROTON) * int(frag_z)
    except (TypeError, ValueError):
        return 1.0
    comp_mass = prec_neutral - frag_neutral
    if not np.isfinite(comp_mass) or comp_mass <= 0:
        return 1.0 if k_lo == 0 else 0.0
    key = (int(comp_mass // 50), k_lo, min(k_hi, max_k))
    if cache is not None and key in cache:
        return cache[key]
    n = max(1, int(round(comp_mass / _AVERAGINE_MASS)))
    comp = {e: max(1, int(round(v * n))) for e, v in _AVERAGINE.items()}
    peaks = isotopic_variants(comp, npeaks=min(max_k, k_hi + 6), charge=0)
    inten = np.array([p.intensity for p in peaks], dtype=float)
    tot = inten.sum()
    val = 1.0 if tot <= 0 else float(inten[k_lo:k_hi + 1].sum() / tot)
    val = min(1.0, max(0.0, val))
    if cache is not None:
        cache[key] = val
    return val




def frag_composition(mod_seq, base_ion_type, tag):
    """Elemental composition of a fragment, tag included."""
    split_frag_seq, frag_info = fragment_seq(mod_seq, base_ion_type)
    frag_type, _i, loss, frag_z = frag_info
    comp = get_seq_comp(split_frag_seq, frag_type, neutral_loss=(loss or None))
    tags = [t for aa in split_frag_seq
            for t in re.findall(f"\\(({tag.name}.*?)\\)", aa)]
    if tag.channel_comp is not None and tags:
        comp = comp + reduce(lambda x, y: x + y,
                             [tag.channel_comp[
                                 re.findall(f"{tag.name}-(\\d+)", t)[0]]
                              for t in tags])
    return comp, int(frag_z)


def _pattern_for_label(mod_seq, frag_name, tag, label, n_iso, cache):
    """Isotope envelope of one fragment as it would appear in channel
    ``label``, derived from any sibling's sequence by retagging."""
    return fragment_isotope_pattern(seq_for_label(mod_seq, tag, label),
                                    base_frag_key(frag_name), tag,
                                    n_iso=n_iso, cache=cache)


def deconvolve_fragment_group(mod_seqs, base_ion_type, target_mz, observed,
                              tag, n_iso=5, mz_tol_da=0.015, cache=None):
    """Joint isotope deconvolution of one fragment key across sibling channels.

    Parameters
    ----------
    mod_seqs : sequence of str, length K
        One modified sequence per channel present for this fragment key (its
        OWN tag annotation must be embedded, e.g. row 0's "(PSMtag_9plex-0)",
        row 1's "(PSMtag_9plex-4)", ...).
    base_ion_type : str
        Shared fragment name, "_iso" suffix already stripped, e.g. "b3_1".
    target_mz : array-like, length K
        Each channel's own (monoisotopic) target m/z for this fragment.  A
        channel need not have independently MATCHED this fragment for its
        entry to be valid -- since every channel shares the same peptide
        backbone, an unmatched channel's target is just as derivable from
        any matched sibling's position plus the tag's known per-channel mass
        (``channel_label_from_seq`` + ``tag.mass_dict``).  Restricting siblings
        to ones that happened to match the SAME fragment independently
        undercounts real contamination sources -- a channel not being listed
        as "matched" doesn't mean no signal reached the detector there, only
        that it fell under this row's own match threshold.
    observed : array-like, length K
        Observed intensity already extracted at each channel's own
        ``target_mz``, from the SAME MS2 scan (the deconvolution is only
        physically meaningful within one scan -- mixing scans would compare
        signal from different points in time).
    tag : massTag
    n_iso : int
        Highest isotope order to consider a channel's envelope out to.
        Contamination beyond a handful of channel-steps is negligible; the
        default covers 2 steps of the 9-plex ladder (order 4) with margin.
    mz_tol_da : float
        Tolerance for matching a contaminating isotope peak to a sibling's
        target m/z.  Deliberately an ABSOLUTE m/z tolerance, not ppm: the
        gap between a channel step and the nearest natural isotope spacing
        it approximates is a near-constant few mDa (it shrinks with charge,
        not with fragment mass -- e.g. ~5 mDa at z=1, ~1 mDa at z=3, measured
        on the 9-plex tag), so a ppm window sized for peak-matching noise
        would silently miss the true alignment on small, low-m/z fragments.
    cache : dict or None
        Passed through to ``fragment_isotope_pattern``.

    Returns
    -------
    np.ndarray, length K
        Per-channel intensity net of the isotope contamination it received
        from earlier (lower-mass) channels in the group -- the drop-in
        replacement for ``observed`` in the existing per-channel fit.
    """
    target_mz = np.asarray(target_mz, dtype=float)
    observed = np.asarray(observed, dtype=float)
    k = len(target_mz)
    if k == 0:
        return observed
    if k == 1:
        return observed

    order = np.argsort(target_mz)
    patterns = [fragment_isotope_pattern(mod_seqs[i], base_ion_type, tag,
                                         n_iso=n_iso, cache=cache)
                for i in range(k)]

    A = np.zeros((k, k))
    for row in range(k):
        m = order[row]
        for col in range(k):
            j = order[col]
            if j > m:
                continue  # a channel cannot contaminate an earlier one
            mz_off, rel_int = patterns[j]
            predicted_mz = target_mz[j] + mz_off
            diff = np.abs(predicted_mz - target_mz[m])
            hit = np.argmin(diff)
            if diff[hit] <= mz_tol_da:
                A[row, col] = rel_int[hit]

    b = observed[order]
    coeffs, _ = nnls(A, b)

    out = np.empty(k)
    out[order] = coeffs
    return out


_WARNED_BAD_KEYS = set()


def joint_channel_fit(group_rows, seqs, names, mzs, libs, masks, fracs,
                      extract_fn, tag, n_iso=5, mz_tol_da=0.015, cache=None,
):
    """Solve ALL channels of a plex group simultaneously, at ONE scan.

    ``cascading_channel_fit`` walks the mass ladder and subtracts each
    contaminator using its OWN already-fitted coefficient. That is sequential,
    so an error in an early channel propagates into every later one and
    compounds -- measured on JD1187, d12 (7th of 9, and the only C channel
    whose predecessor is also C) ends up over-subtracted by ~0.7 log2, giving a
    d10/d12 gap of +0.859 where doing nothing at all would give +0.154.

    Here every channel is a free unknown in one non-negative least squares:

        rows     distinct observed m/z positions (peaks that coincide within
                 tolerance are ONE row -- that is the whole point, since
                 channel N's M+2 IS channel N+1's monoisotopic peak)
        columns  one per channel
        A[p, j]  what channel j contributes at position p per unit coefficient,
                 = (lib_j,k / total_j) * frac_j * rel_j,k[order]
        plus one penalty row per channel carrying (1 - frac_j) against 0, the
        same unmatched-library penalty jmod_coefficient applies, so the result
        is on the SAME scale as every other coeff column.

    No channel's estimate feeds another's, so there is nothing to propagate.
    Requires all channels to be read at one scan, which is what the common
    apex provides -- without it a joint solve would evaluate most channels at
    the wrong point on their elution profile, which is why this was not
    possible before.
    """
    from src.utils.parse_peptides import split_frag_name

    rows = [r for r in group_rows if masks.get(r) is not None
            and np.asarray(masks[r]).any()]
    if not rows:
        return {}
    lab = {}
    for r in rows:
        try:
            lab[r] = channel_label_from_seq(seqs[r], tag)
        except Exception:
            return {}
    scale = {}
    for r in rows:
        lib_r = np.asarray(libs[r], dtype=float)
        ok = np.isfinite(lib_r) & (lib_r > 0)
        tot = float(np.sum(lib_r[ok])) if ok.any() else 0.0
        f = fracs[r]
        f = float(f) if np.isfinite(f) and 0 < f <= 1 else 1.0
        scale[r] = (tot, f)

    # candidate positions: every channel's own masked, monoisotopic fragments
    cand = []
    for r in rows:
        nm, mz, m_r = names[r], mzs[r], np.asarray(masks[r], bool)
        for k in np.nonzero(m_r)[0]:
            if k < len(nm) and "_iso" not in str(nm[k]):
                cand.append((float(mz[k]), base_frag_key(str(nm[k]))))
    if not cand:
        return {}
    # merge positions that coincide -- a peak is one observation
    cand.sort()
    pos = []
    for mzv, key in cand:
        if pos and abs(pos[-1][0] - mzv) <= mz_tol_da:
            pos[-1][1].add(key)
        else:
            pos.append([mzv, {key}])
    targets = np.array([p[0] for p in pos], dtype=float)
    obs = np.asarray(extract_fn(targets), dtype=float)
    if obs.size != targets.size:
        return {}

    # Solve for the physical AMPLITUDE (obs ~= sum_j A_j * lib_j * rel), with
    # no penalty row, then convert to coeff units afterwards. Putting the
    # (1-f) penalty into the joint system does not work: each column also
    # carries that channel's isotope satellites, so it sums to ~1.55*f rather
    # than f, and the penalty then has the wrong relative weight -- measured,
    # that alone produced 1.09-1.57 log2 errors at realistic frac_lib_int.
    # Converting afterwards reproduces jmod_coefficient exactly when there is
    # nothing to deconvolve.
    n_p, n_c = len(pos), len(rows)
    A = np.zeros((n_p, n_c), dtype=float)
    b = np.nan_to_num(obs, nan=0.0)
    for jc, r in enumerate(rows):
        nm, mz, lib_r = names[r], mzs[r], np.asarray(libs[r], dtype=float)
        m_r = np.asarray(masks[r], bool)
        tot, f = scale[r]
        if tot <= 0:
            continue
        for k in np.nonzero(m_r)[0]:
            if k >= len(nm) or "_iso" in str(nm[k]):
                continue
            key = base_frag_key(str(nm[k]))
            try:
                off, rel = fragment_isotope_pattern(seqs[r], key, tag,
                                                    n_iso=n_iso, cache=cache)
            except Exception:
                continue
            a_k = lib_r[k]
            for q in range(len(off)):
                m_q = float(mz[k]) + float(off[q])
                ip = int(np.argmin(np.abs(targets - m_q)))
                if abs(targets[ip] - m_q) > mz_tol_da:
                    continue
                rq = float(rel[q])
                A[ip, jc] += a_k * rq
    keep = A.any(axis=0)
    if not keep.any():
        return {}
    try:
        x, _ = nnls(A[:, keep], b)
    except Exception:
        return {}
    out, t = {}, 0
    for jc, r in enumerate(rows):
        if not keep[jc]:
            continue
        amp = float(x[t]); t += 1
        tot, f = scale[r]
        lib_r = np.asarray(libs[r], dtype=float)
        okm = np.isfinite(lib_r) & (lib_r > 0) & np.asarray(masks[r], bool)
        s2 = float(np.sum(lib_r[okm] ** 2))
        den = f * f * s2 + (1.0 - f) ** 2 * tot * tot
        out[r] = (amp * tot * f * s2 / den) if den > 0 else np.nan
    return out


def cascading_channel_fit(i, group_rows, seqs, names, mzs, libs, masks, fracs,
                          known_coeffs, extract_fn, tag, n_iso=5,
                          mz_tol_da=0.015, cache=None, prec_mz=None,
                          prec_z=None, window_hi=None, window_lo=None, win_cache=None,
                          return_obs=False,
                          min_siblings=0,):
    """``coeff_ChannelUnique`` for channel ``i``, with isotope-envelope
    contamination from lower-mass sibling channels subtracted before the fit,
    using each sibling's OWN already-computed coefficient.

    Why cascading rather than one joint solve for the whole group: a
    contaminating channel's true abundance is only known from ITS OWN
    fragments at ITS OWN best scan -- mixing every channel's rows into one
    NNLS anchored at channel ``i``'s scan (an earlier version of this
    function) either evaluates siblings at the wrong point in time (their
    real signal reads as near-zero noise there, silently biasing the shared
    solution) or, if their own rows are left out entirely, leaves the system
    underdetermined: one equation (channel i's contaminated fragment) cannot
    pin down two unknowns (channel i's and the contaminator's true
    abundance) at once.

    The fix used throughout this module already computes channels' own
    values in ascending mass order (see the call site): by the time channel
    ``i`` is processed, every lower-mass sibling that could contaminate it
    has already been fitted from ITS OWN scan, so its coefficient is a KNOWN
    constant here, not a free unknown. This turns the correction back into
    simple subtraction followed by the ORIGINAL closed-form
    ``jmod_coefficient``, rather than a joint least-squares solve:

        corrected_obs_k = raw_obs_k - sum_j( known_coeffs[j] * a_j(k, order) )
        coeff_i = jmod_coefficient(lib_i, corrected_obs, frac_i, mask)

    where ``a_j(k, order) = (lib_k / total_j) * frac_j * rel_int_at_order``
    is contaminating channel j's OWN normalization (its own total library
    intensity and frac_lib_int -- NOT channel i's), so the subtracted amount
    is exactly what channel j's fit says it should be contributing at that
    isotope order, in the SAME units ``jmod_coefficient`` already reports.

    A channel with no eligible lower-mass contaminator in ``known_coeffs``
    (the very first channel in a group, or one whose alignment falls outside
    ``mz_tol_da``) subtracts nothing and this reduces EXACTLY to
    ``jmod_coefficient`` on the raw extraction -- verified by test.

    Parameters
    ----------
    i : int
        The channel (row index) to compute ``coeff_ChannelUnique`` for.
    group_rows : sequence of int
        Every channel co-isolated in the one scan/precursor group.
    seqs, names, mzs, libs, fracs : mapping row -> array/str/float
        Per-row modified sequence, fragment name array (already normalized
        via ``normalize_frag_name``), fragment m/z array, fragment
        library-relative-intensity array (channel i's and each
        contaminator's FULL library, not just channel-unique fragments --
        needed to reproduce each channel's own ``total``), and
        ``frac_lib_int``.
    masks : mapping row -> bool array
        ``channel_unique_masks`` output per row.
    known_coeffs : mapping row -> float
        Already-computed ``coeff_ChannelUnique`` for every row processed so
        far in this group's ascending-mass walk (so, always lower mass than
        ``i``). A row not present is treated as contributing no
        contamination -- either it hasn't been reached yet (which by
        construction only happens for rows at or above ``i``'s own mass,
        never a real contamination source for ``i``), or its own fit
        failed.
    extract_fn : callable(np.ndarray) -> np.ndarray
        Observed intensity at arbitrary m/z, from channel ``i``'s OWN
        resolved scan.
    tag : massTag
    n_iso, mz_tol_da, cache : see ``deconvolve_fragment_group``.

    Returns
    -------
    float or nan
    """
    from src.utils.parse_peptides import split_frag_name
    from src.ms2_unique_quant import jmod_coefficient

    mask_i = masks[i]
    if not mask_i.any():
        return (np.nan, None, None) if return_obs else np.nan

    label = {r: channel_label_from_seq(seqs[r], tag) for r in group_rows}
    mass = {r: tag.mass_dict[f"{tag.name}-{label[r]}"] for r in group_rows}

    # every sibling's own matched (base-key) position, for deriving a
    # not-independently-matched contaminator's position from channel i's own
    sib_pos = {}
    sib_lib = {}
    for r in group_rows:
        nm, mz, mask = names[r], mzs[r], masks[r]
        lib_r = np.asarray(libs[r], dtype=float)
        for pos in np.nonzero(mask)[0]:
            if "_iso" not in nm[pos]:
                sib_pos[(r, nm[pos])] = mz[pos]
                if pos < lib_r.size:
                    sib_lib[(r, nm[pos])] = lib_r[pos]

    def own_total_frac(r):
        lib_r = np.asarray(libs[r], dtype=float)
        ok = np.isfinite(lib_r) & (lib_r > 0)
        total_r = float(np.sum(lib_r[ok])) if ok.any() else 0.0
        f = fracs[r]
        frac_r = float(f) if np.isfinite(f) and 0 < f <= 1 else 1.0
        return total_r, frac_r

    amp_cache = {}

    def own_amplitude(r, c_r):
        """Channel r's per-library-unit amplitude, i.e. obs_r ~= A_r * lib_r."""
        if r not in amp_cache:
            amp_cache[r] = amplitude_from_coeff(c_r, libs[r], masks.get(r),
                                                fracs[r])
        return amp_cache[r]

    def pattern(r, k):
        return fragment_isotope_pattern(seqs[r], k, tag, n_iso=n_iso, cache=cache)

    nm_i, mz_i, lib_i = names[i], mzs[i], np.asarray(libs[i], dtype=float)
    obs = extract_fn(mz_i)
    # Kept so a caller can evaluate the fit at any subtraction scale alpha
    # without re-extracting: jmod_coefficient is LINEAR in the observation and
    # the correction enters as obs(alpha) = raw - alpha*(raw - corrected), so
    #     coeff(alpha) = coeff(0) + alpha * (coeff(1) - coeff(0))
    # exactly. Two evaluations pin the whole line -- no sweep needed.
    obs_raw0 = obs.copy()
    total_i, _ = own_total_frac(i)

    # Usable fragment positions, with the charge each one needs for its
    # per-channel m/z shift. Resolved once: every ladder rung reuses them.
    usable, zs = [], []
    for pos in np.nonzero(mask_i)[0]:
        if "_iso" in nm_i[pos]:
            continue  # own-channel evidence only, as in the search library
        k = nm_i[pos]
        try:
            _, _, _, frag_z_str = split_frag_name(k)
            zs.append(int(frag_z_str))
            usable.append(pos)
        except Exception as e:
            if k not in _WARNED_BAD_KEYS:
                _WARNED_BAD_KEYS.add(k)
                from src.logger import logger
                logger.warning(f"cascading_channel_fit: skipping fragment "
                               f"key {k!r} ({type(e).__name__}: {e})")
    if not usable:
        c0 = jmod_coefficient(lib_i, obs, fracs[i], mask=mask_i)
        return (c0, obs, obs_raw0) if return_obs else c0
    usable = np.asarray(usable)
    zs = np.asarray(zs, dtype=float)

    # Too few co-isolated siblings to estimate contamination reliably. Measured
    # on JD0588: with 7-9 channels present the correction takes compression
    # 0.756 -> 0.888, but with only 5-6 it goes 0.651 -> 0.619, i.e. WORSE than
    # not correcting -- the contaminator amplitudes are too poorly determined
    # and the subtraction adds bias instead of removing it. Backing off is
    # strictly safer than guessing.
    if min_siblings and len(group_rows) < int(min_siblings):
        _S["skipped_sparse_group"] += 1
        c0 = jmod_coefficient(lib_i, obs, fracs[i], mask=mask_i)
        return (c0, obs, obs_raw0) if return_obs else c0

    own_label = channel_label_from_seq(seqs[i], tag)
    own_mass = float(tag.mass_dict[f"{tag.name}-{own_label}"])
    # tags on the whole peptide -- 1 for an R-terminating peptide, 2 for a
    # K-terminating one. This is what sets the PRECURSOR spacing between
    # channels, and therefore whether a contaminator's low isotopologues fall
    # inside the target's isolation window at all.
    n_tags_prec = len(re.findall(f"{re.escape(tag.name)}-\\d+", seqs[i])) or 1
    row_of_label = {label[r]: r for r in group_rows}

    # A contamination source does NOT have to be a fitted row. Its peptide is
    # in the spectrum whether or not JMod proposed it as a candidate, so
    # restricting the cascade to rows present in the frame silently drops the
    # dominant M+2 term whenever the immediate predecessor was not identified
    # -- measured on JD0588, only 6.07 of 9 channels are present per group and
    # 29% of channels had no usable contaminator at all. Walk the tag's OWN
    # ladder instead, and where a rung has no fitted row, estimate its
    # amplitude straight from this spectrum at its derived positions.
    max_steps = max(1, (n_iso - 1) // 2)     # rung s aligns at isotope order 2s
    rungs = [(lbl, m) for lbl, m in tag_ladder(tag) if m < own_mass - 1e-9]
    rungs = rungs[-max_steps:]               # nearest rungs below, ascending

    _S["calls"] += 1
    _S["group_rows_total"] += len(group_rows)
    _S["ladder_rungs_total"] += len(rungs)
    _sub_before = float(np.sum(obs[mask_i]))

    amps = {}          # label -> per-library-unit amplitude on THIS spectrum
    for lbl, m_l in rungs:
        j = row_of_label.get(lbl)
        c_j = known_coeffs.get(j) if j is not None else None
        if c_j is not None and np.isfinite(c_j) and c_j > 0:
            # Fitted row: use its own coefficient, which was solved over its
            # whole fragment set at its own best scan -- more robust than a
            # re-fit here, and the behaviour already validated.
            a = own_amplitude(j, c_j)
            if np.isfinite(a) and a > 0:
                # that amplitude is calibrated against row j's OWN library
                # normalisation, so pairing it with channel i's library value
                # needs the ratio of the two totals
                total_j, _ = own_total_frac(j)
                scale = (total_j / total_i
                         if total_i > 0 and total_j > 0 else 1.0)
                amps[lbl] = (a, scale, j)
                _S["rung_from_fit"] += 1
                continue
        # No fitted row for this rung: skip it.  A spectrum-estimated
        # 'ladder' rung was tried and removed -- with the fitted-row path
        # working it changed 94% of cells not at all and every scorable
        # channel to three decimals, so it bought nothing for an extra
        # code path and a silent-failure surface.
        continue

    for n, pos in enumerate(usable):
        target_m = mz_i[pos]
        _S["positions"] += 1
        for lbl, (a_l, scale_l, row_l) in amps.items():
            m_l = float(tag.mass_dict[f"{tag.name}-{lbl}"])
            src = target_m + (m_l - own_mass) / zs[n]
            mz_off, rel_int = _pattern_for_label(seqs[i], nm_i[pos], tag, lbl,
                                                 n_iso, cache)
            diff = np.abs((src + mz_off) - target_m)
            hit = int(np.argmin(diff))
            if diff[hit] > mz_tol_da:
                _S["skip_tolerance"] += 1
                _S["skip_tol_mindiff"] += float(diff[hit])
                continue
            # prefer that channel's own recorded library intensity for this
            # fragment; fall back to rescaling channel i's
            lib_kj = (sib_lib.get((row_l, nm_i[pos]))
                      if row_l is not None else None)
            if lib_kj is None or not np.isfinite(lib_kj) or lib_kj <= 0:
                lib_kj = lib_i[pos] * scale_l
            # A narrow isolation window admits only part of the contaminating
            # channel's reachable envelope -- see isolated_envelope_fraction.
            admit = 1.0
            if prec_mz is not None and window_hi is not None:
                try:
                    pc = float(prec_mz) + (n_tags_prec * (m_l - own_mass)
                                           / float(prec_z))
                except (TypeError, ValueError, ZeroDivisionError):
                    pc = None
                admit = isolated_envelope_fraction(
                    prec_mz, prec_z, window_lo, window_hi, target_m, zs[n],
                    prec_mz_contaminator=pc, order=hit, cache=win_cache)
            rel_h = rel_int[hit]
            drop = a_l * lib_kj * rel_h * admit
            _S["applied"] += 1
            _S["applied_order"] += hit
            _S["subtracted"] += min(float(drop), float(obs[pos]))
            obs[pos] = max(0.0, obs[pos] - drop)

    _S["obs_total"] += _sub_before
    coeff = jmod_coefficient(lib_i, obs, fracs[i], mask=mask_i)
    return (coeff, obs, obs_raw0) if return_obs else coeff
