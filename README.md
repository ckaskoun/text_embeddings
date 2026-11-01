## Text Embeddings

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ckaskoun/text_embeddings/blob/main/text_embeddings_FA26.ipynb)

This repository contains code based upon the Odden et al. paper and the G.E.V.I.R repository.
Only the code is included here (no data) for privacy and security reasons.

The main work is one notebook, **[`text_embeddings_FA26.ipynb`](text_embeddings_FA26.ipynb)**, which asks:

- What datasets work the best for a text embeddings workflow (with centroid/Odden et al. style method)?
- How should researchers best choose exemplars?
- What is the ideal amount of sentences to maximize agreement metrics (mainly Cohen's κ)?
- What, if any, are the effects of fine tuning the embeddings model on the exemplars? (work in progress)

The notebook opens with a quick start, a table of contents, and a glossary.

**Code is a work in progress.**

### Folder layout

```
text_embeddings_FA26.ipynb   the notebook (run this)
text_embeddings_FA26.py      the same notebook as plain Python (kept in sync by Jupytext; for reading diffs)
raw_data/                    NOT in git: sources_v2.xlsx, centroids_expanded.xlsx,
                             human_central-diverse.xlsx, exemplar_storage.xlsx, coding_master_alt_final.xlsx
saved_embeddings/            NOT in git: .pkl embedding files (these contain sentence text too)
tests/                       synthetic data + a smoke test that runs the whole notebook
legacy/                      the 2025 script workflow, kept for reference
AUDIT.md                     the Fall 2026 code audit: what changed and why
```

### Running it

- **Sources (`sources_v2`) data:** open the notebook in Colab, keep `raw_data/` and `saved_embeddings/` in the
  same Drive folder as the notebook, set `drive_folder` in Setup, then run all cells.
- **PEPPER data:** this data is **not anonymized**. Use a Colab *local runtime* (or Jupyter on your own computer)
  with the data on that computer only, and set `local_folder` in Setup.
- Packages: `pip install -r requirements.txt` (the notebook's first cell also installs what it needs).

### Data privacy

- Never commit `raw_data/`, `saved_embeddings/`, `.xlsx`, or `.pkl` files (`.gitignore` blocks them).
- Never commit a notebook with outputs: tables, figures, and Colab metadata can carry sentences and account details.
  The pre-commit hook strips outputs automatically (see below), or use *Edit > Clear all outputs* first.
- Don't paste student text or data files into online tools.

### Development setup (one time)

```bash
pip install -r requirements.txt pre-commit
pre-commit install
```

After that, every `git commit`:

- strips notebook outputs and metadata (`nbstripout`)
- keeps `text_embeddings_FA26.py` in sync with the notebook (`jupytext`)
- checks the Python for errors (`ruff`)

To check that the notebook still runs end to end on synthetic (fake) data, about 2 minutes:

```bash
pip install nbclient ipykernel
python tests/smoke_test.py
```
