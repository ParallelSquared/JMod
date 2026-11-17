# Branch `splitwin_ms1env`

Based on `dev`. Everything new is opt-in: without the new flags, results match `dev`. On a test-mode slice of a 1 ng run, the flags-off output (IDs and search coefficients) is identical to `dev`.

## What it adds

| Change | Flag(s) | Files |
|---|---|---|
| Joint fit of adjacent DIA windows, for precursors whose isotope envelope is split across two isolation windows. Window edges are measured from each run. | `--split_win` (needs `--iso`); `--split_min_frac` 0.1, `--split_max_dt` 10, `--split_transmission` measure/box | `src/split_windows.py` (new), `src/spectral_fitting.py`, `src/run_jmod.py`, `src/default_dict.py` |
| MS1 isotope-envelope scoring features: M0..M3 vs the theoretical envelope, in the nearest MS1 scan and ±2 scans | `--ms1_envelope`; `--ms1_envelope_scans` 2 | `src/ms1_envelope.py` (new), `src/spectral_fitting.py`, `src/run_jmod.py`, `src/utils/io/read_output.py` |
| Libraries with a-ions: decoy generation used to stop with "only supports b/y fragment ions"; decoy a_k is now decoy b_k − CO | none | `src/models/spec_lib/library_store.py`, `src/split_windows.py` |
| `--test_mode` crash fix: its library pre-filter built a plain dict, which has no `relabel_tag`; it now uses `subset_entries` | none | `src/run_jmod.py` |

## How to run

```
python run_jmod.py -i RUN.mzML -l LIBRARY.parquet -o OUTDIR \
  -m 3 --iso --num_iso 3 --lib_frac 0.2 -p 10 --use_emp_rt --decoy rev --unmatched c -t 8 \
  --split_win --ms1_envelope
```

Add `--tag NAME` for mass-tagged samples. Check the log for these lines, which show the features ran:

- `--split_win: measured window transmission: ...` (or a warning that it fell back to the nominal box)
- `--split_win: N windows (measured transmission); templates for N entries ...`
- `--split_win: N of N MS2 scans have the window above acquired within ...`
- `--ms1_envelope: theoretical envelopes for N entries ...`

`--split_win` is skipped, with a warning, for ion-mobility data, overlapping windows and runs without `--iso`.

## Results

Two 1 ng runs on an Orbitrap Astral, 2 Th windows from 600 to 1300 m/z: one sample labelled with a new mass tag (`new_tag`), one label-free (LF). Each was searched with a 100 ng DIA-NN empirical library of the same sample type. The searches ran on JMod 2.1.0 with the same code, `--no_ms1_req` (no MS1 requirement, the default on `dev`) and `--iso --num_iso 3`. Precursors / proteins at 1% FDR:

| | new_tag | LF |
|---|---|---|
| 2.1.0 | 63,834 / 5,185 | 62,938 / 5,856 |
| `--split_win` | 67,793 / 5,350 (+6.2%) | 64,956 / 6,024 (+3.2%) |
| `--ms1_envelope` | 65,510 / 5,194 (+2.6%) | 64,076 / 5,955 (+1.8%) |
| `--split_win --ms1_envelope` | 68,956 / 5,305 (+8.0%) | 66,671 / 6,052 (+5.9%) |
| same, library without y1 ions | **74,685 / 5,528 (+17.0%)** | — |

### Entrapment (LF)

The LF run was also searched with a human + yeast + Arabidopsis library. The sample is human only, so yeast, Arabidopsis and other plant hits are false.

- At decoy q < 1%, 2.1.0 gives 45,327 human precursors and `--split_win --ms1_envelope` gives 47,987 (+5.9%).
- Entrapment FDP (combined estimator) does not rise: Arabidopsis-only 0.48% for 2.1.0 and 0.47% with the new flags; all entrapment species 1.40% and 1.34%. So the extra IDs are not false hits.
- At matched all-entrapment FDP, the gain is +6.5% at 1% FDP (44,239 vs 47,110 human precursors) and +5.6% at 2% (46,003 vs 48,596).

There is no entrapment check for `new_tag` yet, because there is no `new_tag` entrapment library.

### `--split_win` settings tried

| | new_tag | LF |
|---|---|---|
| Joint fit only when the window above is within 0.2 s, `--split_min_frac` 0.4, box transmission | 66,232 / 5,250 | 64,385 / 5,984 |
| Always the nearest scan of the window above, box transmission | 68,090 / 5,264 | 65,089 / 5,976 |
| Always the nearest scan, measured transmission, `--split_min_frac` 0.1 (the defaults) | 67,793 / 5,350 | 64,956 / 6,024 |

`--num_iso 4` was slightly worse than 3 (new_tag 67,612, LF 64,649).

### Fragment ions in the library (new_tag, `--split_win --ms1_envelope`)

| Library | Precursors / proteins | vs full library |
|---|---|---|
| Full | 68,956 / 5,305 | |
| Without b1 | 70,329 / 5,383 | +2.0% |
| **Without y1** | **74,685 / 5,528** | **+8.3%** |
| Without b1 and y1 | 73,734 / 5,555 | +6.9% |
| Without y1 and y2 | 72,425 / 5,543 | +5.0% |
| With a2 ions added | 68,547 / 5,364 | −0.6% |

- **y1 is the problem.** Every `new_tag` y1 is one of two ions, R's y1 (175.12) or the tagged K's y1, so a y1 match says almost nothing about which peptide it is. Yet y1 is the most intense library fragment for 61% of `new_tag` precursors and carries 20% of the library's fragment intensity (12.3% in the LF library).
- Without y1 the gain is larger for K-terminal precursors (+10.4%) than R-terminal ones (+6.4%). Dropping y2 as well costs IDs (K +5.2%, R +5.1%), so y2 is informative. Dropping b1, b2, y1 and y2 gave 67,704 precursors, below the full library.
- For LF, dropping b1 and y1 does not help: +0.6% with `--split_win` alone, −0.6% with both flags.
- **a-ions.** a2 was added at 0.124 × b2, the a2/b2 ratio measured in `new_tag` spectra. It adds 0.86% of the library intensity and changes IDs by −0.6%, so a-ions are not worth adding.
- **FDR proxy (no entrapment yet).** 11.4% of the precursors gained without y1 were accepted by DIA-NN at 100 ng only at q 0.01–0.05. Any set of marginal IDs shows the same share (11.0% for the IDs that 2.1.0 finds and `--split_win --ms1_envelope` does not), against about 4% among all 1 ng IDs. So the gained IDs look like ordinary marginal IDs, but this is not a substitute for entrapment.

## Why `--split_win`

With narrow DIA windows, a precursor whose M0 sits near the top of its window has the rest of its isotope envelope isolated and fragmented in the next window up, in the same cycle. JMod searches each precursor only in the window holding M0, with a fragment-isotope template that assumes every precursor isotope was isolated.

In the 1 ng runs above, the M0 window isolates on average 75% of a `new_tag` precursor's envelope and 83% of an LF one's. About 20% of precursors have only M0 isolated. Those are identified 4–7 points less often at 1 ng than other precursors of the same charge and M0 share, and `new_tag` precursors lose more. Heavy precursors, such as mass-tagged peptides, are hit hardest because less of their envelope is in M0.

## What `--split_win` does

Each MS2 scan (window A) is fitted jointly with the scan of the window directly above it (window B) when B's scan was acquired within `--split_max_dt` seconds of A's. The default, 10 s, always uses the nearest scan of the window above.

The window above is not always acquired next to A. The Astral methods above interleave each cycle as two sweeps from 600 to 1300 m/z. 178 of 349 `new_tag` window pairs (131 of 259 LF) are acquired back to back, within 50 ms. The rest come half a cycle later, about 1.5 s for `new_tag` and 1.1 s for LF, against peak widths of roughly 3 s. The pattern is fixed for each pair. Pairing with the nearest scan whatever the gap beat a 0.2 s limit (see the settings table). With a smaller `--split_max_dt`, pairs further apart are fitted alone, with window-truncated templates.

- **Joint spectrum.** B's peaks are appended with m/z × 1000. Matching stays ppm-exact, and B's peaks can never match A's.
- **Precursors with M0 in A, one column each.** In A, the template keeps the fragment isotopes allowed by the precursor isotopes A isolated. When B isolates at least `--split_min_frac` of the precursor's envelope, the column also gets a B part, with the fragment isotopes allowed by the precursor isotopes B isolated. The coefficient is a single value fitted to both scans.
  - Template intensity of fragment isotope k when precursor isotopes j in [a, b] are isolated: L_fk × P(complement heavy count in [a−k, b−k]).
  - L_fk is the library's untruncated isotope pattern from JMod's `--iso` expansion.
  - The complement is the precursor minus the fragment. Exact C/H/N/O/S compositions are used, including mods and tags.
- **Candidates that pass only through B.** A precursor with M0 in A is also a candidate when it passes the fragment-index gate in B's spectrum (only when its B part is modelled).
- **Competitors, fitted but not reported.**
  - Precursors with M0 in the window below A whose B part lands in A, with their window-above template.
  - Precursors with M0 in B, with their own window-B template.
- **Output.** Only precursors with M0 in A are reported, so the output keeps one row per precursor per scan with the same columns.
- **Features.** `frag_mz` lists are unscaled. Window-B peaks carry spare frag-code bit 21 and are left out of the b/y counts, `hyperscore` and `longest_y_ions`, so those features still count each fragment once.
- **Precursors without a B part** still get the window-truncated A template. For a precursor whose whole envelope is in A, that template equals the library template.

The truncation factor assumes the library's fragment intensities are not already truncated by isolation. That holds for the 100 ng DIA-NN libraries used here: within a precursor, library intensities do not track the complement factor (slope +0.00 LF, +0.07 `new_tag`).

### Measured window transmission (`--split_transmission measure`, the default)

Before building templates, each run fits its own isolation-window transmission. `box` uses the nominal window instead.

- **Measurement.** Bright MS1 isotope clusters: M0, M1 and M2 at z=2 or 3, no M−1 peak, and M1/M0 consistent with a peptide of that mass. In the nearest-in-RT MS2 scan of every window near a cluster, unfragmented isotopologues survive equally, so (MS2_p/MS2_r)/(MS1_p/MS1_r) = T(o_p)/T(o_r).
- **Fit.** Cell medians over (o_r, o_p) are fitted with T = Φ((o−l)/wl)·Φ((h−o)/wh), without assuming where the flat top is, so a displaced passband shows up. The MS2-vs-MS1 m/z offset is tracked locally along the run.
- **Quality checks.** At least 2,000 ratios, at least 60 cells, median log error ≤ 0.7, and a passband at least 1 Th wide. Otherwise it falls back to the box.
- **Templates.** Window factors become sums over precursor isotopes weighted by transmission: L_fk × Σ_j τ_W(j) P_complement(j−k). The envelope fractions f_A and f_B can add up to more than 1, because an overlap between windows is transmitted in both.

Measured on the two runs above (2 Th windows, so the nominal edges are ±1 Th):

| Run | Lower edge | Upper edge |
|---|---|---|
| new_tag | −0.98 Th, width 0.09 | +1.09 Th, width 0.21 |
| LF | −0.81 Th, sharp | +1.10 Th, width 0.17 |

Both are stable across m/z thirds. The LF passband sits about 0.15 Th high. In both runs the soft upper edge makes adjacent windows overlap by about 0.2–0.3 Th.

Not yet handled: only the window above is fitted jointly. When the passband is displaced upwards, as for LF, M0 within about 0.2 Th of a window's lower edge is mostly transmitted by the window below.

## `--ms1_envelope`

Adds seven scoring features. For each candidate, M0..M3 in the nearest MS1 scan and in ±`--ms1_envelope_scans` scans are compared with the exact theoretical envelope:

- in the nearest scan: isotopes found (`ms1env_n`), cosine (`ms1env_cos`), log summed intensity (`ms1env_logint`) and intensity-weighted ppm error (`ms1env_ppm`);
- over the nearby scans: the best of n, cosine and log summed intensity (`ms1env_n_nb`, `ms1env_cos_nb`, `ms1env_logint_nb`).

Without the flag the output schema is unchanged.

## a-ion libraries

Decoy a_k is the decoy's own b_k minus CO, from the same prefix sums, so b/y decoy masses are unchanged. `--split_win` uses a-ion compositions of b − CO.

## Tests

- `python tests/test_split_windows.py`: window factors and envelope fractions against a brute-force exact calculation (heavy-tagged and CAM peptides, z=2 and z=3, box and measured transmission), and the joint-library layout.
- `python tests/test_ms1_envelope.py`: theoretical envelopes and the features on synthetic MS1 scans.
- `pytest tests/test_aions_decoy.py`: decoy a2 = decoy b2 − CO, and decoy b/y m/z identical with and without a2 rows, on the bundled `data/filtered_library.tsv`.
