# Notebook audit — Fall 2026

Audit of `text_embeddings_FA26.ipynb` (Google Drive version last edited 2026-10-30, 78 cells) and the
changes made for this repository. The aim was to fix problems and remove repetition **without changing the
analysis or its results**, and to make the notebook easy to follow for someone new to it.

## How "no change in results" was checked

`tests/smoke_test.py` runs every cell on synthetic data (random vectors, fake sentences, same file formats) in four
configurations: Sources with human selection, Sources with machine selection, Sources with S merged into P, and
PEPPER. The original and the cleaned notebook were both run, and these tables were compared for **exact**
equality: `results`, `full_corpus_results`, `calibration`, `main_cutoffs`, `n_subcentroids`, `selection_results`,
`selection_summary`, `near_best`, `df_similarities`. **All 36 comparisons were identical.**

What changed visually: the "Human bank" strategy is now purple in every figure (it was near-black in some and purple
in others), and the machine-mode similarity figures now use the same layout as the human-mode ones.

## Safety and privacy

| # | Finding | Change |
|---|---|---|
| S1 | The saved notebook carried the owner's name and Google user id 36 times (Colab `executionInfo`), 15 output tables, and 11 figures. PEPPER sentences are not anonymized. | All outputs, execution counts, and execution metadata removed. A pre-commit hook (`nbstripout`) now strips them on every commit. |
| S2 | `local_folder` was a hard-coded personal Mac path (exposed the username, broke for anyone else). | `Path.home() / 'PEPPER' / 'colab' / 'text_embeddings_FA26'`, the same folder on the owner's Mac, with a comment on how to change it. |
| S3 | Saved embeddings are loaded with `pickle`, which can run code, and the `.pkl` files contain every sentence. | Comment: only load `.pkl` files you created; treat them like the raw data. `.gitignore` blocks `*.pkl`. |
| S4 | `HF_TOKEN` is read from the environment (good), with no guidance. | Comment: never paste a token into a cell; use an environment variable or Colab Secrets. |
| S5 | Nothing stopped data files from being committed. | `.gitignore` for `raw_data/`, `saved_embeddings/`, `.xlsx/.xls/.csv/.pkl`. Claude Code settings deny reading them and ask before shell commands that touch them. |
| S6 | `legacy/find_indices.py` contains a personal Windows path. It is already in the public git history. | Left unchanged (legacy code); noted here. |

## Errors and fragile code

| # | Finding | Change |
|---|---|---|
| E1 | `import seaborn as sns` was never used and not installed by the `%pip` cell, so a fresh local runtime **crashed in the packages cell** (confirmed by the smoke test). | Import removed. |
| E2 | The maximum sequence length `512` was typed in three places (saved-file check, model setting, saved metadata). Changing one would silently break loading. | One `max_seq_length = 512` setting. |
| E3 | `n_init=(5 if dataset == 'pepper' else 50)` was repeated three times (once inside a PEPPER-only block). | One derived `kmeans_n_init` in Setup. Same values. |
| E4 | The same strategy colors were redefined in six places, and "Human bank" had two different colors. | One palette block at the top of the graphing helpers; Human bank is purple everywhere, so it no longer looks like the black full-human reference line. |
| E5 | `similarity_scatter` depended on `codes` and `strategy_colors` defined in the previous cell. | Both figure cells now use the shared palette and `coding_scheme`. |
| E6 | Machine-mode figure branches used a different style (bold titles, "Category X" titles, re-setting spines already set globally). | Merged into one code path per figure with the human-mode style. Same data. |
| E7 | Unused variables: `exemplar_training_indices`, `exemplar_validation_indices`, `pool_sizes`; `combined_counts` and `focal_counts` were always overwritten before use. | Removed. |
| E8 | Typos: `code_occurence`, "mutally". | Fixed. |
| E9 | The "Human Exemplars" note lists counts that will go stale when the data changes. | Marked "as of the October 2026 data"; the next cell prints current counts. |
| E10 | Two markdown notes did not match the code: "Kappa Calculation" said background codes use random examples (human mode uses the human-bank subsample), and "Choose the Exemplars" described only machine-mode central/diverse. | Corrected. |

## Repetition replaced by helpers

New `analysis_helpers` cell (Helper Functions). Each helper is the original code moved, not rewritten:

- `fit_elbow_models()` and `choose_elbow_k()`: the elbow fit and rule were written out twice (`elbow` and
  `main_calibration`).
- `centroid_scores()`, `predict_codes()`, `agreement_rows()`: scoring, prediction, and κ tables were duplicated in
  `main_scoring`/`main_kappa`, `full_corpus_kappa`, and (prediction) `selection_calculation`.
- `vector_summary()`: defined twice, identically, inside `selection_similarities`.

Left as they are on purpose:

- The two out-of-fold cutoff calibrations (`main_calibration` with sklearn κ and KMeans; `selection_calculation`,
  vectorized for speed). Merging them risks changing results; comments now point from one to the other.
- Random-stream numbers (6101, 6201, 6301, 6401). Changing them changes which exemplars are drawn; now explained.

## Navigation and documentation

- Intro: quick start (which settings to change, run order, runtimes), folder layout, a linked table of contents,
  and a glossary.
- A short "what this does / what to look for" note under every main section and under subsections that only had a
  title.
- Setup split into **settings you change** and **derived settings and checks**; every setting has a comment.
- Docstrings on every helper (two plotting helpers had none), in the notebook's existing `<arg>:` style.
- Comments for the non-obvious steps: the sigmoid weighting of negatives, why the negative-aware method reuses the
  mean centroid, the cumulative-sum scoring trick, the vectorized κ formula, cutoff tie-breaking.

## Repository changes

- Old 2025 scripts moved unchanged to `legacy/` (with a README).
- `README.md` rewritten: layout, how to run, privacy rules, development setup.
- Added `requirements.txt`, `.gitignore`, `ruff.toml`, `.pre-commit-config.yaml` (nbstripout, jupytext, ruff),
  `jupytext.toml` + `text_embeddings_FA26.py` (plain-Python copy for readable diffs), `CLAUDE.md`,
  `.claude/settings.json` + `.claude/hooks/data_guard.py`, and `tests/` (synthetic data + smoke test).

## Suggestions not implemented

- A GitHub Actions workflow running `tests/smoke_test.py` on every push.
- Fine-tuning section: still a placeholder.
- The machine-mode diverse selection loop is O(n²) per code and repetition; fine at current sizes, worth vectorizing
  if the corpus grows a lot.
