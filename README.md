## **Text Embeddings**

Code based upon the Odden et al. paper and the G.E.V.I.R repository, using sentence embeddings to code student
writing. Everything is in one notebook, **[`text_embeddings_FA26.ipynb`](text_embeddings_FA26.ipynb)**, which asks:

- What datasets work the best for a text embeddings workflow (with centroid/Odden et al. style method)?
- How should researchers best choose exemplars?
- What is the ideal amount of sentences to maximize agreement metrics (mainly Cohen's κ)?
- What, if any, are the effects of fine tuning the embeddings model on the exemplars? (work in progress)

**Code is a work in progress.**

### Contents

```
text_embeddings_FA26.ipynb   the notebook
raw_data/                    Sources data (open source with the Mjaaland et al. paper)
saved_embeddings/            Saved Qwen3-Embedding-8B vectors for the Sources data
legacy/                      the earlier (2025) script version, kept for reference
```

### Running

Open the notebook in Colab or Jupyter, set `drive_folder` (Colab) or `local_folder` (your computer) in the Setup
cell to the folder containing these files, and run all cells. The first cell installs the packages it needs.

### Data privacy

The PEPPER dataset is **not anonymized** and is not included here; run it only on a local runtime with the data
on your own computer.
