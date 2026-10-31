# RT alignment: model choice and open questions

Notes from the 2026-10-30 session.  Every `TODO` that says
"See RT_ALIGNMENT_NOTES.md" refers to a section here.

## What changed in MZRTfit (untested on real data)

Choosing between fine-tuned and empirical RTs:

- **Before:** the fine-tuned model was scored on its held-out peptides.  The
  empirical spline was scored on every `cor_filter` row, including the rows it
  was fitted on.  Residuals at or above `limit = 3` were dropped first.  `3` has
  no units: on Bruker `.d` data (`Frames.Time`, seconds) it meant 3 s, and only
  ~3% of first-search residuals fell inside it.  On mzML/Thermo data (minutes)
  it meant 3 min.
- **Now:** both models are scored on the same held-out peptides, the ones
  `JmodRTPredictor.calibrate` already sets aside.
  - The empirical spline is refitted without those peptides, with the same
    `fast_modal_lowess` settings as `empirical_fit`.
  - Each held-out peptide's empirical prediction is the median over its rows,
    matching how its observed RT in `holdout` is the median over its charge
    states.
  - There is no cutoff.  Both CDFs run to just past the largest error of either
    model, so each AUC is approximately (end - mean |error|).  The model with
    the lower mean |error| on the same peptides wins.
- Charge-state ambiguity filter: `np.ptp(rts) < 1` became `< fwhm` (the run's
  elution width).
- Plot-only uses of `limit` in `MZRTfit` are now untrimmed: the no-fine-tuning
  CDF and the MAD lines in the CDF figure.  The MAD lines used to filter on
  signed residuals.

Not changed: the RT tolerance (`boundary` = GMM on the chosen model's
residuals), the log lines, and timeplex.

Checked so far: compiles; the 41 tests that import `rt_alignment` pass; on
synthetic data each AUC is within 0.1 of (end - mean |error|) and the choice
goes to the lower-error model.  **No test runs `MZRTfit` end to end.  Needs a
real run**, ideally on both a `.d` and an mzML file.  Expect the choice to change
on `.d` data.

## Open questions

1. **Empirical boundary is in-sample** (`rt_alignment.py`, empirical branch).
   When empirical wins, `boundary` comes from a GMM on `all_emp_diffs`, which
   are residuals on the rows the spline was fitted on.  When fine-tuned wins it
   comes from held-out residuals.  Should the empirical boundary also use
   held-out residuals?  That would change the RT tolerance, not just the choice.

2. **Timeplex still assumes minutes** (`MZRTfit_timeplex`):
   - `limit = 3` drives the same CDF comparison *and* `fit_errors`, which sets
     the RT tolerance.
   - `offset_tolerance = 1  ## 1 minute`.

   Timeplex doesn't call `empirical_fit`, so it needs its own data-derived scale.

3. **`fit_errors` filters signed errors** (`errors < limit`, not
   `abs(errors) < limit`), so every negative error passes the cutoff.  It then
   fits half-normal and exponential CDFs as if the errors were absolute.  Check
   whether its callers pass |errors|.

4. **`config.rt_tol = 9`** carries an implicit unit.  It's passed as `rt_tol` to
   `fit_to_lib` in `fit_with_features` (rt_alignment.py, ~line 500) with
   `rt_filter=False`, so it looks unused there.  Confirm and remove if dead.
   `run_jmod.py`'s test-mode `rt_tolerance = config.rt_tol * 2` is only read in
   a commented-out line.

5. **`--test_rt_min` / `--test_rt_max` defaults (0 and 10)** look like minutes.
   They're user-set, but the defaults imply a unit.

6. **`Library IM from <model> for N entries ...` log line**
   (`aligned_library_im`).  RT has no counterpart.  Keep or remove?

## Housekeeping

- `tests/fixtures/library_edgecases.tsv_store.npz` is tracked but stamped
  `STORE_VERSION` 6 (now 7).  Nothing reads it; the tests write their own caches
  to a temporary directory.  Delete it?
- `tests/test_config.py::TestCommandLineOverrides::test_only_typed_options_are_collected`
  fails on HEAD too (`no_ms1_req` comes back `True`, expected `False`).
  Pre-existing, unrelated.
- `finalized_phase3/decoy_edgecases_shuffle.snapshot.pkl` was regenerated from
  current code in 42fb6c6, because its input peptide changed (LIONELK ->
  LIQNELK).  For that one case the finalize bit-identity check now compares
  against today's output, not the original phase-3 arrays.
