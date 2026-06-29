"""
Engineered scoring features for JMod precursor identification.

Validated 2026-06-19 on JD1189 (timeplex 3ch x PSMtag_9plex): adding these to the
score_precursors feature set lifts precursors at 1% FDR by +4.5% (own) / +8.1%
(best-channel) in the full production pipeline. This 15-feature LEAN set retains
~90% of the gain of the larger experimental set.

Three sources:
  - rank / competition + elution shape  -> from the un-condensed search parquet
    (within-scan coeff rank aggregated across each precursor's scans; coeff-weighted
    elution width). The condensed fdc keeps one scan per precursor, so these need the
    full 50M-row file.
  - fragment-error + spectral entropy   -> from the fragment list columns
    (per-fragment ppm errors and observed intensities).
  - scalar transforms                   -> from fdc scalars.

Called once from process_data() right before compute_fragment_correlations().
The names are NOT in score_precursors' drop-list, so they feed the model directly.
"""
import numpy as np
import pandas as pd
import polars as pl

# the lean, non-redundant winners
ENGINEERED_FEATURE_NAMES = [
    "rank_min", "normrank_min", "rank_primary", "coeffsum_mean", "n_scans_fit",
    "pk_rt_sd", "pk_rt_sd_vs_fwhm",
    "n_frag", "frag_err_sd", "frag_err_absmean", "log_sum_inv_abs_err", "spec_entropy_obs",
    "log_coeff", "log_tic", "longy_over_len",
    # elution-coherence + mass-stability set (validated 2026-06-22 on JD1189: +1,788 own /
    # +3,897 best-channel @1% FDR ON TOP of the lean-15; the productive axes are temporal
    # coherence of the elution peak + mass-accuracy stability across the profile).
    "by_series_corr", "b_coeff_corr", "y_coeff_corr", "rvn_trim", "rvnnorm_trim",
    "frag_rt_disp", "fit_data_apex_off", "masserr_coh_w",
]

_GAUSS_FWHM_TO_SD = 2.3548200450309493  # 2*sqrt(2*ln2)


def _rank_elution_features(file, timeplex, elution_fwhm):
    """Within-scan coeff rank + coeff-weighted elution width, per precursor.
    Reads only the columns it needs from the 50M-row search parquet."""
    keys = ["seq", "z", "time_channel"] if timeplex else ["seq", "z"]
    lf = pl.scan_parquet(file).select(["coeff", "spec_id", "rt"] + keys)
    lf = lf.with_columns([
        pl.col("coeff").rank("min", descending=True).over("spec_id").alias("_rk"),
        pl.col("coeff").sum().over("spec_id").alias("_ssum"),
        pl.len().over("spec_id").alias("_sn"),
    ])
    lf = lf.with_columns([
        (pl.col("coeff") / pl.when(pl.col("_ssum") > 0).then(pl.col("_ssum")).otherwise(1.0)).alias("_cfs"),
        pl.when(pl.col("_sn") > 1).then((pl.col("_rk") - 1) / (pl.col("_sn") - 1)).otherwise(0.0).alias("_nrk"),
    ])
    agg = lf.group_by(keys).agg([
        pl.col("_rk").min().alias("rank_min"),
        pl.col("_nrk").min().alias("normrank_min"),
        pl.col("_rk").sort_by("coeff", descending=True).first().alias("rank_primary"),
        pl.col("_cfs").mean().alias("coeffsum_mean"),
        pl.len().alias("n_scans_fit"),
        pl.col("coeff").sum().alias("_w"),
        (pl.col("rt") * pl.col("coeff")).sum().alias("_rtw"),
        (pl.col("rt") ** 2 * pl.col("coeff")).sum().alias("_rt2w"),
    ]).collect()
    g = agg.to_pandas()
    w = g["_w"].replace(0, np.nan)
    rtm = g["_rtw"] / w
    g["pk_rt_sd"] = np.sqrt(np.clip(g["_rt2w"] / w - rtm ** 2, 0, None))
    fwhm = elution_fwhm if (elution_fwhm and np.isfinite(elution_fwhm) and elution_fwhm > 0) else 0.0550
    g["pk_rt_sd_vs_fwhm"] = g["pk_rt_sd"] / (fwhm / _GAUSS_FWHM_TO_SD)
    return g.drop(columns=["_w", "_rtw", "_rt2w"]), keys


def _fragment_features(fdc_list_cols):
    """Fragment-error stats + observed spectral entropy, per kept precursor (__fdc_idx)."""
    fe = pl.col("frag_errors")
    pos = fe.list.eval(pl.when(pl.element().abs() > 0).then(pl.element().abs()).otherwise(None)).list.drop_nulls()
    obs = pl.col("obs_int")
    feat = fdc_list_cols.select([
        pl.col("__fdc_idx"),
        fe.list.len().alias("n_frag"),
        fe.list.std().alias("frag_err_sd"),
        fe.list.eval(pl.element().abs()).list.mean().alias("frag_err_absmean"),
        (pos.list.eval(1.0 / pl.element()).list.sum()).log10().alias("log_sum_inv_abs_err"),
        obs.list.sum().alias("_So"),
        obs.list.eval(pl.when(pl.element() > 0).then(pl.element() * pl.element().log()).otherwise(0.0)).list.sum().alias("_Solx"),
    ]).to_pandas()
    So = feat["_So"].values
    feat["spec_entropy_obs"] = np.where(
        So > 1e-12, np.log(np.clip(So, 1e-12, None)) - feat["_Solx"].values / np.where(So > 0, So, 1.0), 0.0)
    return feat.drop(columns=["_So", "_Solx"])


_NUM_MAX_CACHE = {}


def _num_max(m):
    """Max sum-of-squared consecutive rank-diffs for a length-m permutation (RVN scan-max)."""
    if m in _NUM_MAX_CACHE:
        return _NUM_MAX_CACHE[m]
    from itertools import permutations
    if m < 2:
        v = np.nan
    elif m <= 8:
        b = 0
        for p in permutations(range(1, m + 1)):
            b = max(b, int((np.diff(p) ** 2).sum()))
        v = b
    else:  # m>8: zigzag lower bound + random search
        lo, hi, zz = 1, m, []
        while lo <= hi:
            zz.append(lo); lo += 1
            if lo <= hi:
                zz.append(hi); hi -= 1
        b = int((np.diff(zz) ** 2).sum())
        rng = np.random.default_rng(0)
        for _ in range(3000):
            b = max(b, int((np.diff(rng.permutation(np.arange(1, m + 1))) ** 2).sum()))
        v = b
    _NUM_MAX_CACHE[m] = v
    return v


def _rvn_trim(c):
    """von-Neumann ratio on a coeff elution profile, apex window trimmed at RELATIVE 0.1*apex.
    Returns (rvn /expected, rvnnorm /scan-max). Low = smooth/coherent (target-like)."""
    a = np.asarray(c, float); m = len(a)
    if m < 3:
        return np.nan, np.nan
    ap = int(a.argmax()); thr = 0.1 * a[ap]; l = ap; r = ap
    while l - 1 >= 0 and a[l - 1] >= thr:
        l -= 1
    while r + 1 < len(a) and a[r + 1] >= thr:
        r += 1
    w = a[l:r + 1]; mm = len(w)
    if mm < 3:
        return np.nan, np.nan
    rk = w.argsort().argsort() + 1
    num = int((np.diff(rk) ** 2).sum())
    nmax = _num_max(mm)
    return num / (mm * (mm * mm - 1) / 12.0), (num / nmax if nmax and nmax > 0 else np.nan)


def _pear(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b)); a, b = a[m], b[m]
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _apexrt(v, rt):
    v = np.asarray(v, float)
    if v.size == 0 or np.all(np.isnan(v)):
        return np.nan
    return rt[int(np.nanargmax(v))]


def _coelution_features(file, timeplex):
    """8 validated elution-coherence + mass-stability features from the full search parquet.
    Per precursor (rt-ordered scans): b/y-series obs-intensity co-elution, each series vs coeff,
    max-normalised trimmed RVN smoothness, per-fragment apex-RT dispersion, fit-vs-data apex
    offset, and coeff-weighted mass-error coherence. All "lower = target-like" except the corrs."""
    keys = ["seq", "z", "time_channel"] if timeplex else ["seq", "z"]
    pf = pl.scan_parquet(file)
    total = pf.select(pl.len()).collect().item()

    # --- per-scan reductions (b/y mean obs-int, total int, mean frag-error) via chunked explode ---
    CHUNK = 5_000_000; parts = []
    for off in range(0, total, CHUNK):
        base = (pf.slice(off, CHUNK)
                .select(keys + ["rt", "coeff", "frag_names", "obs_int", "frag_errors"])
                .with_row_index("_ri"))
        red = (base.select(["_ri", "frag_names", "obs_int", "frag_errors"])
               .explode(["frag_names", "obs_int", "frag_errors"])
               .with_columns((pl.col("frag_names") & 7).alias("_ion"))
               .group_by("_ri").agg([
                   pl.col("obs_int").filter(pl.col("_ion") == 0).mean().alias("_b"),
                   pl.col("obs_int").filter(pl.col("_ion") == 1).mean().alias("_y"),
                   pl.col("obs_int").sum().alias("_tot"),
                   pl.col("frag_errors").mean().alias("_me")]))
        parts.append(base.select(["_ri"] + keys + ["rt", "coeff"]).join(red, on="_ri").drop("_ri").collect())
    prof = (pl.concat(parts).lazy().group_by(keys).agg([
        pl.col("rt").sort_by("rt").alias("rt"),
        pl.col("_b").sort_by("rt").alias("b"), pl.col("_y").sort_by("rt").alias("y"),
        pl.col("_tot").sort_by("rt").alias("tot"), pl.col("coeff").sort_by("rt").alias("c"),
        pl.col("_me").sort_by("rt").alias("me")]).collect())
    del parts
    g = prof.select(keys).to_pandas()
    RT = prof["rt"].to_list(); B = prof["b"].to_list(); Y = prof["y"].to_list()
    TOT = prof["tot"].to_list(); C = prof["c"].to_list(); ME = prof["me"].to_list()
    del prof
    n = len(RT)
    bys = np.full(n, np.nan); bcc = np.full(n, np.nan); ycc = np.full(n, np.nan)
    rvt = np.full(n, np.nan); rnt = np.full(n, np.nan); fdo = np.full(n, np.nan); mcw = np.full(n, np.nan)
    for i in range(n):
        rt = np.asarray(RT[i], float)
        if len(rt) < 3:
            continue
        c = C[i]
        bys[i] = _pear(B[i], Y[i]); bcc[i] = _pear(B[i], c); ycc[i] = _pear(Y[i], c)
        rvt[i], rnt[i] = _rvn_trim(c)
        fdo[i] = abs(_apexrt(c, rt) - _apexrt(TOT[i], rt))
        me = np.asarray(ME[i], float); w = np.asarray(c, float); mk = ~np.isnan(me)
        if mk.sum() >= 3 and w[mk].sum() > 0:
            ww = w[mk]; mm = me[mk]; wm = np.sum(ww * mm) / ww.sum()
            mcw[i] = np.sqrt(np.sum(ww * (mm - wm) ** 2) / ww.sum())
    g["by_series_corr"] = bys; g["b_coeff_corr"] = bcc; g["y_coeff_corr"] = ycc
    g["rvn_trim"] = rvt; g["rvnnorm_trim"] = rnt; g["fit_data_apex_off"] = fdo; g["masserr_coh_w"] = mcw

    # --- per-fragment apex-RT dispersion (streaming explode -> group by fragment identity) ---
    fr = (pf.select(keys + ["rt", "obs_int", "frag_names"]).explode(["frag_names", "obs_int"])
          .group_by(keys + ["frag_names"]).agg([
              pl.col("rt").sort_by("obs_int", descending=True).first().alias("_aprt"),
              pl.col("obs_int").max().alias("_w")]))
    dq = fr.group_by(keys).agg([pl.col("_aprt"), pl.col("_w")])
    try:
        dprof = dq.collect(engine="streaming")
    except TypeError:
        dprof = dq.collect(streaming=True)
    dg = dprof.select(keys).to_pandas()
    AP = dprof["_aprt"].to_list(); W = dprof["_w"].to_list()
    del dprof
    disp = np.full(len(AP), np.nan)
    for i in range(len(AP)):
        ap = np.asarray(AP[i], float); w = np.asarray(W[i], float); mk = (w > 0) & (~np.isnan(ap))
        if mk.sum() >= 2:
            ap = ap[mk]; w = w[mk]; wm = np.sum(w * ap) / w.sum()
            disp[i] = np.sqrt(np.sum(w * (ap - wm) ** 2) / w.sum())
    dg["frag_rt_disp"] = disp
    return g.merge(dg, on=keys, how="left"), keys


def add_engineered_features(fdc, fdc_list_cols, file, timeplex=True, elution_fwhm=None):
    """Merge the 23 validated engineered features onto fdc and return it.

    fdc must already carry ``coeff``, ``tic``, ``longest_y_ions``, ``pep_len`` and
    (for timeplex) ``time_channel`` plus the ``__fdc_idx`` row key produced by
    get_large_prec. Memory note: the rank/elution pass scans the full search
    parquet (~50M rows); for very large runs alongside loaded spectra+library this
    can be the binding constraint (run get_large_prec/scoring staged if so).
    """
    rank_pk, keys = _rank_elution_features(file, timeplex, elution_fwhm)
    for k in keys:
        if k in fdc.columns:
            rank_pk[k] = rank_pk[k].astype(fdc[k].dtype)
    fdc = fdc.merge(rank_pk, on=keys, how="left")

    coel, _ = _coelution_features(file, timeplex)
    for k in keys:
        if k in fdc.columns:
            coel[k] = coel[k].astype(fdc[k].dtype)
    fdc = fdc.merge(coel, on=keys, how="left")

    frag = _fragment_features(fdc_list_cols)
    fdc = fdc.merge(frag, on="__fdc_idx", how="left")

    fdc["log_coeff"] = np.log10(np.clip(fdc["coeff"].values.astype(float), 1e-9, None))
    fdc["log_tic"] = np.log10(np.clip(fdc["tic"].values.astype(float), 1e-9, None))
    fdc["longy_over_len"] = fdc["longest_y_ions"].values / np.clip(fdc["pep_len"].values.astype(float), 1, None)
    return fdc
