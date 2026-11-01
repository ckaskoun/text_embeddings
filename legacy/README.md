# Legacy scripts (2025)

These are the original script-based workflow (adapted from the G.E.V.I.R. repository used in
Odden et al., 2024). They are kept unchanged for reference and are **no longer maintained**.

The current workflow is the notebook at the top of the repository:
[`text_embeddings_FA26.ipynb`](../text_embeddings_FA26.ipynb).

| Script | What it did |
|---|---|
| `create_embeddings.py` | Cleaned `.xlsx` files and created embeddings (Jina v4) |
| `format_data.py` | Codebook and helpers to align coded data with the G.E.V.I.R. format |
| `embedding_functions.py` | Scoring with centroids/subcentroids, KL/JS divergence, heatmaps |
| `analyze_embeddings.py` | Compared assigned scores with human-coded scores |
| `assign_codes.py` | Assigned codes to uncoded files from selected exemplar sentences |
| `dimension_analysis.py` | Compared accuracy at reduced embedding dimensions |
| `plot_embeddings.py` | 2D UMAP plots of representative sentences and subcentroids |
| `find_indices.py` | Printed sentence indices for each code |

They read data from hard-coded local paths and `.pkl` files that are not in this repository.
