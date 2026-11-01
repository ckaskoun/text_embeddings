# Text Embeddings — project notes for Claude

Research code that codes student writing with sentence embeddings (centroid method after Odden et al.).
The working notebook is `text_embeddings_FA26.ipynb`; the user edits it in Google Colab, and a copy lives in their
Google Drive (`Text Embeddings/text_embeddings_FA26/`). `legacy/` is the retired 2025 script workflow: do not edit it.

## Privacy rules (most important)

- `raw_data/`, `saved_embeddings/`, and any `.xlsx`, `.csv`, or `.pkl` file contain student writing.
  **PEPPER data is not anonymized.** Never read, print, summarize, upload, or commit these files, and never ask
  the user to paste their contents. `.claude/settings.json` denies Read on them and asks before Bash touches them.
- Never commit a notebook with outputs. Outputs and Colab metadata (`executionInfo`) contain sentence tables and the
  user's name and Google user id. The pre-commit hook (`nbstripout`) strips them; check with
  `git diff --cached` before committing.
- To test code, use the synthetic data in `tests/` (`python tests/smoke_test.py`), never the real data.

## Notebook conventions

- Headings: `## **Section**`, `### **Subsection**`, `#### **Step**`. Each `##` section starts with a short
  markdown note: what it does, what it reads, what to look for.
- Every code cell has a descriptive cell tag (`settings`, `main_kappa`, ...); keep tags and cell ids stable. The
  intro's table of contents links to cell ids with Colab `#scrollTo=<id>`.
- Settings live only in the Setup `settings` cell, split into "SETTINGS YOU CHANGE" and "DERIVED SETTINGS AND CHECKS".
  `tests/smoke_test.py` rewrites settings by matching `name = value` at the start of a line, so keep that form.
- Reusable logic goes in the Helper Functions cells (`formatting_helpers`, `counting_helper`, `analysis_helpers`,
  `graphing_helpers`). Helpers may read `coding_scheme`, `dataset`, and other Setup globals.
- Docstrings use the existing style: one summary line, then `<arg>: description` lines.
- Comments: sentence case, above the block they explain, blank line after in long cells. Explain *why*.
- Colors and markers come from the constants at the top of `graphing_helpers` (`CODE_COLOR_LIST`,
  `METHOD_COLORS`, `STRATEGY_COLORS`, `STRATEGY_MARKERS`, `NOTE_COLOR`, `GRID_COLOR`). Do not hard-code new hex values.
- Global matplotlib style is set once in the `packages` cell; do not reset spines/frames per figure.
- Random draws use `np.random.default_rng([random_seed, repetition, stream, code])`; changing a stream number
  changes the results.

## Checking a change

```bash
python tests/smoke_test.py                                   # runs every cell on fake data, 4 configurations
python tests/smoke_test.py --save-results /tmp/before ...    # save key tables to compare two versions
ruff check text_embeddings_FA26.ipynb
pre-commit run --all-files                                   # strip outputs, sync .py, lint
```

A refactor should leave the saved tables (`results`, `full_corpus_results`, `calibration`, `selection_summary`,
`near_best`, `df_similarities`) exactly equal; compare with `pandas.testing.assert_frame_equal`.

## Keeping the pair in sync

`text_embeddings_FA26.py` is the Jupytext copy of the notebook (`jupytext.toml`). Edit either file, then run
`jupytext --sync text_embeddings_FA26.ipynb` (the pre-commit hook does this too).
