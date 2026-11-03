"""The nearby-scan extrema are centred on the apex PC1 scan, not the max-coefficient scan."""
import polars as pl

from src.utils.io.read_output import find_extrema_in_nearby_scans


def test_window_is_centred_on_the_apex_pc1_scan():
    # 9 scans of one precursor; max coeff at scan 1, max apex_pc1 at scan 7
    df = pl.DataFrame({
        "seq": ["P"] * 9, "z": [2.0] * 9,
        "rt": [float(i) for i in range(9)], "spec_id": list(range(9)),
        "coeff": [1, 50, 1, 1, 1, 1, 1, 5, 1],
        "apex_pc1": [0, 1, 0, 0, 0, 0, 0, 9, 0],
        "val": [100, 100, 100, 100, 100, 3, 4, 5, 6],
    }).lazy()
    out = find_extrema_in_nearby_scans(df, ["val"], [False], n_scans=2).collect()
    # +-2 scans around scan 7 = scans 5..8 -> min 3 (around scan 1 it would be 100)
    assert out["val_nearby_min"].unique().to_list() == [3]


def test_anchor_can_still_be_the_coefficient():
    df = pl.DataFrame({
        "seq": ["P"] * 5, "z": [2.0] * 5, "rt": [0.0, 1, 2, 3, 4], "spec_id": [0, 1, 2, 3, 4],
        "coeff": [9, 1, 1, 1, 1], "apex_pc1": [0, 0, 0, 0, 9], "val": [1, 2, 3, 4, 5],
    }).lazy()
    out = find_extrema_in_nearby_scans(df, ["val"], [True], n_scans=1, anchor_col="coeff").collect()
    assert out["val_nearby_max"].unique().to_list() == [2]


def test_smoothness_runs_before_the_nearby_features():
    """get_large_prec computes smoothness before n_scans exists."""
    from src.utils.io.read_output import calculate_peak_smoothness
    df = pl.DataFrame({"seq": ["P"] * 3 + ["Q"], "z": [2.0] * 4, "rt": [0.0, 1, 2, 0],
                       "spec_id": [0, 1, 2, 3], "coeff": [1.0, 4.0, 1.0, 2.0]}).lazy()
    out = calculate_peak_smoothness(df).collect()
    assert "smoothness" in out.columns
    assert out.filter(pl.col("seq") == "Q")["smoothness"].to_list() == [0.0]
    assert out.filter(pl.col("seq") == "P")["smoothness"][0] > 0


def test_apex_score_prefers_the_scan_near_the_prediction(tmp_path):
    """With the RT prior, the apex moves from a stronger far scan to the near one."""
    import numpy as np
    from src.utils.io.read_output import get_large_prec, get_parquet_schema
    schema = get_parquet_schema(timeplex=False)
    rows = []
    # one precursor, two clusters: a strong interferer 60 s off, the real peak at 0
    for k, (rt_err, scale) in enumerate([(-60, 3.0), (-61, 2.5), (-59, 2.5), (0, 1.0), (1, 0.8), (-1, 0.8)]):
        r = {c: 0.0 for c in schema}
        r.update(seq="PEPTIDEK", z=2.0, spec_id=k, Ms1_spec_id=0, rt=1000.0 + k, coeff=100 * scale,
                 hyperscore=10 * scale, scribe_scores=0.1 / scale, gof_stats=-scale,
                 manhattan_distances=scale, frac_lib_int=0.2 * scale, b_counts=2 * scale,
                 y_counts=3 * scale, num_lib=5 * scale, max_matched_residuals=-scale,
                 rt_error=float(rt_err), im_error=0.0, file_name="f", protein="P", is_decoy=False,
                 frag_names=[], frag_errors=[], frag_mz=[], frag_int=[], obs_int=[],
                 unique_frag_mz=[], unique_obs_int=[], ent_norm_angle=0.1)
        rows.append(r)
    # padding precursors so the PCA has groups to fit on
    rng = np.random.default_rng(0)
    for p in range(60):
        for k in range(5):
            r = dict(rows[0]); w = np.exp(-0.5 * (k - 2) ** 2)
            r.update(seq=f"PAD{p}", spec_id=100 + 5 * p + k, rt=float(k), coeff=10 + 100 * w + rng.normal(),
                     hyperscore=5 + 10 * w, rt_error=0.0, gof_stats=-w, max_matched_residuals=-w)
            rows.append(r)
    pl.DataFrame(rows, schema=schema).write_parquet(tmp_path / "decoylibsearch_coeffs.parquet")
    f = str(tmp_path / "decoylibsearch_coeffs.parquet")
    pick = lambda **kw: get_large_prec(f, condense_output=False, **kw)[1].set_index("seq").loc["PEPTIDEK", "rt_error"]
    assert pick() == -60.0                        # spectral PC1 alone: the interferer
    assert abs(pick(rt_sigma=20.0)) <= 1.0        # with the RT prior: the real peak
    scans = pl.read_parquet(tmp_path / "apex_pc1_scans.parquet")
    assert {"apex_pc1", "rt_logprior", "im_logprior", "apex_score"} <= set(scans.columns)


def test_override_features_record_the_spectral_apex(tmp_path):
    """spec_apex_rt_error / apex_pc1_cost describe what the RT prior overrode."""
    import numpy as np
    from src.utils.io.read_output import get_large_prec, get_parquet_schema
    schema = get_parquet_schema(timeplex=False)
    base = {c: 0.0 for c in schema}
    base.update(seq="PEPTIDEK", z=2.0, Ms1_spec_id=0, file_name="f", protein="P", is_decoy=False,
                frag_names=[], frag_errors=[], frag_mz=[], frag_int=[], obs_int=[],
                unique_frag_mz=[], unique_obs_int=[], ent_norm_angle=0.1, im_error=0.0)
    rows = []
    for k, (rt_err, sc) in enumerate([(-60, 3.0), (-61, 2.5), (-59, 2.5), (0, 1.0), (1, 0.8), (-1, 0.8)]):
        r = dict(base); r.update(spec_id=k, rt=1000.0 + k, coeff=100 * sc, hyperscore=10 * sc,
                                 scribe_scores=0.1 / sc, gof_stats=-sc, manhattan_distances=sc,
                                 frac_lib_int=0.2 * sc, b_counts=2 * sc, y_counts=3 * sc, num_lib=5 * sc,
                                 max_matched_residuals=-sc, rt_error=float(rt_err))
        rows.append(r)
    rng = np.random.default_rng(0)
    for p in range(60):
        for k in range(5):
            r = dict(rows[0]); w = np.exp(-0.5 * (k - 2) ** 2)
            r.update(seq=f"PAD{p}", spec_id=100 + 5 * p + k, rt=float(k), coeff=10 + 100 * w + rng.normal(),
                     hyperscore=5 + 10 * w, rt_error=0.0, gof_stats=-w, max_matched_residuals=-w)
            rows.append(r)
    pl.DataFrame(rows, schema=schema).write_parquet(tmp_path / "decoylibsearch_coeffs.parquet")
    out = get_large_prec(str(tmp_path / "decoylibsearch_coeffs.parquet"), condense_output=False,
                         rt_sigma=20.0)[1].set_index("seq").loc["PEPTIDEK"]
    assert abs(out["rt_error"]) <= 1.0                 # chosen apex: the real peak
    assert out["spec_apex_rt_error"] == 60.0           # spectral apex: the interferer
    assert out["apex_pc1_cost"] > 0
