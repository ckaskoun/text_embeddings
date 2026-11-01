# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] id="06679e72"
# # **Text Embeddings Analysis**
#
# This notebook will focus on a few main questions:
#
# - What datasets work the best for a text embeddings workflow (with centroid/Odden et al. style method)?
# - How should researchers best choose exemplars?
# - What is the ideal amount of sentences to maximize agreement metrics (mainly Cohen's κ)?
# - What, if any, are the effects of fine tuning the embeddings model on the exemplars? [**Work-in-progress**]
#
# ---
#
# ### **Quick Start**
#
# 1. **Pick a runtime.**
#    - `sources_v2` data: hosted Colab is fine. The Setup cells mount Google Drive.
#    - PEPPER data (`coding_master_alt_final`): use a **local runtime** (Connect > Connect to a local runtime).
#      PEPPER is **not anonymized**, so it should never be uploaded to Colab, Drive, GitHub, or an AI tool.
# 2. **Edit the settings cell in Setup** (the cell starting with `# Coding schemes`). Usually only these change:
#    - `coding_scheme`: `sources_base`, `sources_combined`, or `PEPPER_combined` (this also picks the dataset)
#    - `exemplar_selection_source`: `'human'` or `'machine'`
#    - `drive_folder` / `local_folder`: where this notebook's folder lives
#    - `use_saved`: `True` loads saved embeddings, `False` creates new ones (needs a GPU)
# 3. **Run all cells in order** (Runtime > Run all). To change dataset or settings, edit Setup and run all again.
# 4. **Runtime:** most cells take seconds. *Exemplar Analysis > Kappa Calculation* takes about **10 minutes** with
#    `n_repetitions = 250`; lower it in Setup for a quick check.
#
# The `sources_v2` data is open source with the Mjaaland et al. paper.
#
# ### **Folder Layout**
#
# The enclosing folder only needs this notebook, `raw_data/`, and `saved_embeddings/`.
# `raw_data/` should contain:
#
# - `sources_v2.xlsx`
# - `centroids_expanded.xlsx`
# - `human_central-diverse.xlsx` (Sources human central/diverse labels)
# - `exemplar_storage.xlsx`
# - `coding_master_alt_final.xlsx`
#
# `saved_embeddings/` can contain pre-embedded data as a .pkl, but not necessary (can create embeddings in this notebook).
# These `.pkl` files contain the sentence text too, so treat them like the raw data.
#
# Results and graphs stay in the notebook, save as needed. **Clear all outputs before sharing or committing**
# (Edit > Clear all outputs), since tables and metadata can contain sentences and account details.
#
# ### **Contents**
#
# | Section | What it does |
# |---|---|
# | [Setup](#scrollTo=23c0bc00) | Packages, settings, data folder checks |
# | [Helper Functions](#scrollTo=9e072838) | Small reusable functions for data, scoring, and graphs |
# | [Data](#scrollTo=7e8c3295) | Loads the coded corpus and the human exemplars |
# | [Embeddings](#scrollTo=358b6f5d) | Loads saved vectors or creates new ones |
# | [2D Projections](#scrollTo=0ee98289) | PCA (and optional UMAP) views of the corpus |
# | [Centroid Analysis](#scrollTo=56cce8c2) | Mean, KMeans, and negative-aware centroids from the full human bank, scored with κ |
# | [Exemplar Analysis](#scrollTo=05b9a312) | How κ changes with the number and type (random/central/diverse) of exemplars |
# | [Fine-Tuning](#scrollTo=649a6f52) | Future work |
#
# In Colab, the **Table of contents** panel (left sidebar) also lists every heading.
#
# ### **Glossary**
#
# - **Exemplar**: a human-chosen example sentence for a code. The full set is the **human bank**.
# - **Centroid**: the normalized average embedding of a code's exemplars. A sentence's score for a code is its
#   cosine similarity to that centroid.
# - **KMeans subcentroids**: several centers per code (count chosen with an elbow curve); a sentence's score is its
#   highest similarity to any of them.
# - **Anti-centroid** (negative-aware method): a center built from a code's negative examples; its similarity is
#   subtracted from the positive score.
# - **Central / Diverse**: exemplar sets that are typical of a code (close to its center) or spread out across it.
#   In **human** mode these come from human C/D labels; in **machine** mode they are chosen from embedding distances.
# - **Validation split**: a fixed half of the corpus, never used to choose exemplars, used to measure agreement.
# - **Cohen's κ**: agreement between the model's codes and the human codes, corrected for chance
#   (1 = perfect, 0 = chance level). **Macro κ** averages the per-code binary κ values; **multiclass κ** (Sources only)
#   treats the four codes as one label.
# - **Cutoff** (PEPPER only): the similarity a sentence needs to receive a code, since PEPPER sentences can have
#   several codes or none.

# %% [markdown] id="23c0bc00"
# ## **Setup**

# %% [markdown] id="97df359d"
# Run these cells first. The settings cell is the only one you normally edit; the next cell checks that every
# required file exists before anything else runs.
#
# The saved-embedding analysis needs `pandas`, `numpy`, `matplotlib`, `scikit-learn`, and `openpyxl`.
# Creating new embeddings also needs `torch` and `sentence-transformers`; the optional UMAP view needs `umap-learn`.

# %% id="90a96249" tags=["install_packages"]
# Run once after connecting to a new Colab or local runtime.
# Existing packages are reused; this does not upgrade them unnecessarily.

# %pip install -q numpy pandas matplotlib scikit-learn openpyxl matplotlib-inline

# %% id="c722caf7" tags=["packages"]
# Packages

from pathlib import Path
import os
import time
import pickle as pkl

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_inline.backend_inline import set_matplotlib_formats

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from IPython.display import display

# SentenceTransformer and torch are imported in the new-embeddings cell only.
# UMAP is also optional, and is imported in its own cell below.

# Keep graph fonts, borders, and legends consistent throughout the notebook.

plt.rcParams.update({
    'figure.dpi': 160,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': False,
    'figure.facecolor': 'white'
})

# For higher resolution graphs
set_matplotlib_formats('svg', 'retina')

# %% id="e41f6b34" tags=["settings"]
# ============================================================
# SETTINGS YOU CHANGE
# Edit values in this part only, then run all cells again.
# ============================================================

# Coding schemes

# Sources: L = Limitations, O = Other, P = Physical principles, S = Statistics.
# PEPPER: DA = Direct authenticity, S3 = Failure, S5 = Iteration, S7 = Decision making.
sources_base = ['L', 'O', 'P', 'S']
sources_combined = ['L', 'O', 'P']  # Merge S (Statistics) into P - not used usually, but for comparison used here
PEPPER_combined = ['DA', 'S3', 'S5', 'S7']

# Data

# Make sure you have the folder `raw_data` in the same folder as this notebook, and that it contains the following files:
# - `sources_v2.xlsx`
# - `centroids_expanded.xlsx`
# - `human_central-diverse.xlsx`
# - `exemplar_storage.xlsx`
# - `coding_master_alt_final.xlsx`

# Add the folder where this notebook is located in drive to get `sources_v2` data
drive_folder = Path('/content/drive/MyDrive/Colab Notebooks/Text Embeddings/text_embeddings_FA26')

# Add path to local computer folder to local runtime to use `PEPPER` data.
# Path.home() is your user folder (for example /Users/<you> on a Mac). Edit the rest to match your computer.
local_folder = Path.home() / 'PEPPER' / 'colab' / 'text_embeddings_FA26'

# Parameters

coding_scheme = sources_base  # Also chooses the dataset: PEPPER_combined uses PEPPER, the others use Sources
random_seed = 67
validation_fraction = .5  # Fixed evaluation set for centroid and repeated-exemplar analyses

# The maximum number of subcentroids to use for KMeans subcentroids method

max_elbow_k = 5

embedding_model = 'Qwen/Qwen3-Embedding-8B'
embedding_revision = '1d8ad4ca9b3dd8059ad90a75d4983776a23d44af'  # Fixed model version, for reproducibility
max_seq_length = 512  # Longest input, in tokens; part of the saved-embedding settings
use_prompt = False  # False keeps the instruction empty and uses a separate saved file.
use_saved = True  # False creates and saves new embeddings; it can take a while
batch_size = 256  # Lower this if memory runs out; Qwen8B also needs enough memory for its weights.
use_umap = False  # Optional extra view; PCA is used for the main figures

# Exemplar selection settings

# All repeated selections use sampling WITHOUT replacement within each code.
# n counts distinct pool rows. Original corpus rows may contain identical text.

# 'human' uses the recorded C/D labels for Sources; 'machine' uses corpus rankings.
# PEPPER has no C/D annotation file and continues to use machine selection.
exemplar_selection_source = 'human'  # 'human' or 'machine'
# Random is independent of the central/diverse choice. Samples remain within each code.
random_selection_source = 'training'  # 'training' or 'human_central_diverse'
human_selection_file = 'human_central-diverse.xlsx'  # In raw_data beside the other files
human_selection_column = 'Final'
human_exclude_validation_text = False  # Prevent exact validation text in human selection pools

n_repetitions = 250  # Repeated selections, always using the same validation set (lower for a quick test run)
exemplar_counts = list(range(1, 1001))  # Exemplar counts tested on the growth curves (capped at each pool size)
selection_zoom_max = 35  # Main plots focus on small counts; insets show the full range

central_bands = 5  # Machine central: number of distance bands, closest band used first
diverse_candidate_fraction = .05  # Machine diverse: pick from this farthest fraction at each step
kappa_tolerance = .025  # "Near-best" means within this much of the best mean κ

# ============================================================
# DERIVED SETTINGS AND CHECKS (no edits needed below)
# ============================================================

# The scheme determines which dataset to load.
if coding_scheme not in [sources_base, sources_combined, PEPPER_combined]:
    raise ValueError('Choose one of the three coding schemes above.')
dataset = 'pepper' if coding_scheme == PEPPER_combined else 'sources_v2'

# PEPPER has a smaller, multi-label bank, so KMeans uses fewer random restarts there.
kmeans_n_init = 5 if dataset == 'pepper' else 50

# Check the selection settings and fall back where PEPPER has no human C/D labels.
if exemplar_selection_source not in ['human', 'machine']:
    raise ValueError("exemplar_selection_source must be 'human' or 'machine'.")
effective_selection_source = exemplar_selection_source
if dataset == 'pepper' and exemplar_selection_source == 'human':
    effective_selection_source = 'machine'
    print('PEPPER has no human C/D labels in this Sources workbook; using machine selection.')

if random_selection_source not in ['training', 'human_central_diverse']:
    raise ValueError("random_selection_source must be 'training' or 'human_central_diverse'.")
effective_random_source = random_selection_source if dataset == 'sources_v2' else 'training'
if dataset == 'pepper' and random_selection_source != 'training':
    print('PEPPER has no human C/D bank; Random uses the corpus training partition.')
needs_human_selection_labels = (dataset == 'sources_v2' and
    (effective_selection_source == 'human' or effective_random_source == 'human_central_diverse'))

# The code columns used by convert_codes() in the helper functions.
coded_category_names = coding_scheme

# The saved-embedding file name records the model, dataset, and prompt choice.
embedding_name = f"{embedding_model.replace('/', '_')}_{dataset}"
if not use_prompt:
    embedding_name += '_no_prompt'

# Use the same instruction for corpus sentences and human exemplars.
# An empty string means no task instruction for that dataset.
embedding_prompt = (
    'Instruct: Represent student explanations of experimental uncertainty so that responses based on '
    'the same kind of causal reasoning are similar.'
    '\nQuery: '
    if dataset == 'sources_v2' else
    'Instruct: Classify student reflections about authentic research and scientific research '
    'practices.'
    '\nQuery: '
)

if not use_prompt:
    embedding_prompt = ''

# %% id="2022ecb8" tags=["runtime_files"]
# Run to automatically mount drive for `sources_v2`, will need to connect to
# a local runtime for PEPPER.
# This cell also checks that every file needed for the chosen settings exists.

hosted_colab = Path('/var/colab/hostname').exists()

if hosted_colab and dataset == 'pepper':
    raise RuntimeError(
        'PEPPER uses your computer. Choose Connect > Connect to local runtime, '
        'or open this notebook locally, then rerun Setup.'
    )

if hosted_colab:
    from google.colab import drive

    print('When prompted, mount Drive')
    drive.mount('/content/drive')
    notebook_folder = drive_folder

else:
    # An absolute path is used because a Colab local runtime need not start
    # in the folder containing the notebook. Change local_folder in Setup if you move it.
    notebook_folder = local_folder

# All data reads and newly saved embeddings use this one selected folder.

sources_data_raw = notebook_folder / 'raw_data' / 'sources_v2.xlsx'
PEPPER_data_raw = notebook_folder / 'raw_data' / 'coding_master_alt_final.xlsx'
sources_human_exemplars = notebook_folder / 'raw_data' / 'centroids_expanded.xlsx'
pepper_human_exemplars = notebook_folder / 'raw_data' / 'exemplar_storage.xlsx'
human_selection_path = notebook_folder / 'raw_data' / human_selection_file
embedding_file = notebook_folder / 'saved_embeddings' / f'{embedding_name}.pkl'

# Sources does not need the PEPPER files on Drive, and vice versa locally.

required_files = ([sources_data_raw, sources_human_exemplars] if dataset == 'sources_v2'
                  else [PEPPER_data_raw, pepper_human_exemplars])

if needs_human_selection_labels:
    required_files = required_files + [human_selection_path]

if use_saved:
    required_files = required_files + [embedding_file]

missing = [str(path) for path in required_files if not path.is_file()]

if missing:
    raise FileNotFoundError(
        'Check the selected folder and account. Missing files: ' + ', '.join(missing)
    )

print('Runtime:', 'Google-hosted Colab' if hosted_colab else 'Local computer')
print('Data folder:', notebook_folder)
print('Embedding file:', embedding_file)

# %% id="de14be16" tags=["install_embedding_packages"]
# These larger packages are needed only when creating new embeddings, or creating optional UMAP plots.
# sentence-transformers also installs torch if it is missing (hosted Colab already has it).

if not use_saved:
    # %pip install -q "sentence-transformers>=3.0.0" "transformers>=4.51.0"

if use_umap:
    # %pip install -q umap-learn


# %% [markdown] id="9e072838"
# ## **Helper Functions**
#
# Small functions used later in the notebook. Running these cells only defines the functions; nothing is
# calculated or plotted yet. Several helpers read `coding_scheme` and `dataset` from Setup, so run Setup first.

# %% [markdown] id="025a4dc9"
# ### **Data Structure Helper Functions**

# %% id="2c8124ee" tags=["formatting_helpers"]
def convert_codes(row):
    """
    Converts columns of coding into one column of a list of codes.
    <row>: For use with pandas df.
    Returns a list of codes, or ['No Code'] if none are present.
    """
    # The selected coding scheme supplies the column names.

    code_lst = [col for col in coded_category_names if row[col] == 1]

    if len(code_lst) == 0:
        code_lst.append('No Code')

    return code_lst


def add_codes_column(df):
    """Creates a new column 'Codes' to contain the list of codes."""
    df['Codes'] = df.apply(convert_codes, axis=1)

    return df


def convert_codes_column(df):
    """Converts the Sources 'updated_code' strings into lists in 'Codes'."""
    df['Codes'] = df['updated_code'].apply(
        lambda x: [code.strip() for code in x.split(',')] if pd.notna(x) else ['No Code']
    )

    return df


# %% [markdown] id="b3365e44"
# ### **Data Analysis Helper Functions**

# %% id="cd6159ec" tags=["counting_helper"]
def sum_codes(df_coded):
    """Sums occurrences of codes in the 'Codes' column."""
    code_occurrence = {}

    # Count every recorded code, including several codes on the same sentence.

    for cell in df_coded['Codes']:
        for code in cell:
            code_occurrence[code] = code_occurrence.get(code, 0) + 1

    return code_occurrence


# %% id="044e57af" tags=["analysis_helpers"]
def fit_elbow_models(vectors):
    """
    Fits KMeans with k = 1, 2, ... centers, up to max_elbow_k (or the number of distinct vectors).
    <vectors>: one embedding per row.
    Returns the list of fitted models; model k - 1 has k centers.
    """
    max_k = min(max_elbow_k, len(np.unique(vectors, axis=0)))
    models = []

    for k in range(1, max_k + 1):
        model = KMeans(n_clusters=k, n_init=kmeans_n_init, random_state=random_seed)
        model.fit(vectors)
        models.append(model)

    return models


def choose_elbow_k(inertia_values):
    """
    Chooses the number of centers from an elbow curve.
    <inertia_values>: KMeans inertia for k = 1, 2, 3, ...
    Returns the chosen k (1 when the curve is flat/straight or has fewer than three points).
    """
    # Put both axes on a 0-to-1 scale so their different units do not matter.
    # The elbow is the largest gap below the line joining the two endpoints.

    values = np.asarray(inertia_values)

    if len(values) >= 3 and values[0] - values[-1] > 1e-12:

        x = np.linspace(0, 1, len(values))
        y = (values - values[-1]) / (values[0] - values[-1])
        gap = (1 - x) - y

        if gap.max() > 1e-8:
            return int(gap.argmax()) + 1

    return 1


def centroid_scores(vectors, centers, anti_centers=None):
    """
    Scores every sentence against every code.
    <vectors>: normalized embeddings, one row per sentence.
    <centers>: {code: array of one or more normalized centers}.
    <anti_centers>: optional {code: normalized anti-centroid}, subtracted for the negative-aware method.
    Returns an array with one row per sentence and one column per code in coding_scheme.
    """
    # A subcentroid method takes the largest similarity within each code.

    scores = np.column_stack([
        (vectors @ centers[code_name].T).max(axis=1)
        for code_name in coding_scheme
    ])

    if anti_centers is not None:

        negative_scores = np.column_stack([
            vectors @ anti_centers[code_name]
            for code_name in coding_scheme
        ])

        scores = scores - negative_scores

    return scores


def predict_codes(scores, cutoffs=None):
    """
    Turns scores into 0/1 code predictions.
    Sources chooses exactly one code: the highest score.
    PEPPER can assign multiple codes, or no code: a code is predicted when its score reaches its cutoff.
    <cutoffs>: PEPPER only; one cutoff per code. A NaN (unsupported) cutoff leaves that code unknown (NaN).
    """
    if dataset == 'sources_v2':
        return np.eye(len(coding_scheme))[scores.argmax(axis=1)]

    return np.where(
        np.isfinite(cutoffs)[None, :], (scores >= cutoffs).astype(float), np.nan
    )


def agreement_rows(method, labels, predictions):
    """
    Cohen's κ between human labels and predictions, for each code and in summary.
    <labels>, <predictions>: arrays with one column per code in coding_scheme.
    Returns a list of {'method', 'code', 'kappa'} rows: one per code, 'Macro', and (Sources) 'Overall (Sources)'.
    """
    rows = []
    values = []

    for j, code_name in enumerate(coding_scheme):

        value = np.nan

        if np.isfinite(predictions[:, j]).all():
            value = cohen_kappa_score(labels[:, j], predictions[:, j])

        values.append(value)
        rows.append({'method': method, 'code': code_name, 'kappa': value})

    # Macro binary kappa is the common summary for Sources and PEPPER.

    macro = np.mean(values) if np.isfinite(values).all() else np.nan
    rows.append({'method': method, 'code': 'Macro', 'kappa': macro})

    if dataset == 'sources_v2':

        # Multiclass kappa is a separate statistic for the exclusive Sources labels.

        value = cohen_kappa_score(labels.argmax(axis=1), predictions.argmax(axis=1))
        rows.append({'method': method, 'code': 'Overall (Sources)', 'kappa': value})

    return rows


def vector_summary(vectors):
    """
    Summarizes a group of normalized vectors.
    Returns (center, within): the normalized mean direction, and the mean pairwise cosine
    similarity between different members (self-pairs excluded; NaN for a single vector).
    """
    vectors = np.asarray(vectors, dtype=np.float64)
    total = vectors.sum(axis=0)
    center = total / max(np.linalg.norm(total), 1e-12)
    n = len(vectors)

    # |sum|^2 counts every pair twice plus each vector with itself; remove the self-pairs.

    within = ((float(total @ total) - float(np.square(vectors).sum())) / (n * (n - 1))
              if n > 1 else np.nan)

    return center, within


# %% [markdown] id="caeadaed"
# ### **Graphing Helper Functions**

# %% id="99b2ef4a" tags=["graphing_helpers"]
# Shared colors and markers, so each code, method, and selection strategy looks the same in every graph.
# Colorblind-friendly (Okabe-Ito based). Codes get colors in coding_scheme order.

CODE_COLOR_LIST = ['#0072B2', '#D99A00', '#009E73', '#D55E00']
METHOD_COLORS = {'Mean centroid': '#0072B2', 'KMeans subcentroids': '#009E73',
                 'Negative-aware centroid': '#D55E00'}
STRATEGY_COLORS = {'Random': '#777777', 'Central': '#0072B2', 'Diverse': '#D55E00',
                   'Human bank': '#5B1879', 'Human (Actual)': '#5B1879'}
STRATEGY_LINESTYLES = {'Random': '-', 'Central': '--', 'Diverse': '-.', 'Human bank': ':'}
STRATEGY_MARKERS = {'Random': 'o', 'Central': 's', 'Diverse': 'D', 'Human bank': 'X'}
NOTE_COLOR = '#555555'  # Captions under the graphs
GRID_COLOR = '#D8DDE2'


def plot_projection(coords, labels, coding_scheme, title, per_code=False,
                    centroid_coords=None, negative_coords=None):
    """
    Plots an already fitted 2D projection. Does not fit PCA/UMAP again.
    <labels>: dataframe with one binary (or unknown/NaN) column per code.
    <centroid_coords>, <negative_coords>: optional dictionaries of 2D centers.
    """
    codes = list(coding_scheme)
    colors = dict(zip(codes, CODE_COLOR_LIST))
    coords = np.asarray(coords)
    positives = labels[codes].eq(1).to_numpy()
    centroid_coords = {} if centroid_coords is None else centroid_coords
    negative_coords = {} if negative_coords is None else negative_coords

    # Reuse the corpus limits in every main projection plot.
    all_points = coords
    low, high = all_points.min(axis=0), all_points.max(axis=0)
    padding = np.maximum((high - low) * 0.05, 0.01)
    panels = codes if per_code else ['All codes']
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5), squeeze=False)
    for ax, panel in zip(axes.flat, panels):
        # Grey points provide context, including sentences with unknown labels.
        background = np.ones(len(coords), dtype=bool) if per_code else positives.sum(axis=1) == 0
        if background.any():
            ax.scatter(coords[background, 0], coords[background, 1], color='lightgrey',
                       s=12, alpha=0.5, rasterized=True,
                       label='All sentences (context)' if per_code else 'No active code')
        for code in codes:
            if per_code and code != panel:
                continue
            mask = positives[:, codes.index(code)]
            # In a combined plot, multiple positive codes get their own marker.
            if not per_code:
                mask = mask & (positives.sum(axis=1) == 1)
            ax.scatter(coords[mask, 0], coords[mask, 1], color=colors[code],
                       s=16, alpha=0.7, label=code, rasterized=True)
            if code in centroid_coords:
                centers = np.asarray(centroid_coords[code]).reshape(-1, 2)
                ax.scatter(centers[:, 0], centers[:, 1], c=[colors[code]], s=180,
                           marker='X', edgecolors='black', label='_nolegend_', zorder=4)
            if code in negative_coords:
                centers = np.asarray(negative_coords[code]).reshape(-1, 2)
                ax.scatter(centers[:, 0], centers[:, 1], facecolors='none', s=140,
                           edgecolors=colors[code], marker='D', linewidths=2,
                           label='_nolegend_', zorder=5)
        if not per_code:
            multiple = positives.sum(axis=1) > 1
            if multiple.any():
                ax.scatter(coords[multiple, 0], coords[multiple, 1], c='black',
                           s=18, alpha=0.7, label='Multiple positive codes', rasterized=True)
        ax.set(xlim=(low[0] - padding[0], high[0] + padding[0]),
               ylim=(low[1] - padding[1], high[1] + padding[1]),
               xlabel='Projection dimension 1', ylabel='Projection dimension 2',
               title=f'{title}: {panel}' if per_code else title)
        # Code colors appear once; use generic markers for the prototype types.
        handles, names = ax.get_legend_handles_labels()

        if centroid_coords:
            handles.append(plt.Line2D([], [], marker='X', linestyle='none', color='black', markersize=8))
            names.append('Centroid(s)')

        if negative_coords:
            handles.append(plt.Line2D([], [], marker='D', linestyle='none', color='black',
                                      markerfacecolor='none', markersize=7))
            names.append('Anti-centroid')

        ax.legend(handles, names, fontsize=8, loc='upper center',
                  bbox_to_anchor=(.5, -.14), ncol=min(3, len(handles)))
    fig.tight_layout()
    plt.show()
    return fig


def plot_elbow(inertia, k_values, selected_k):
    """Plots the fitted elbow curves and the selected count for each code."""
    fig, axes = plt.subplots(1, len(inertia), figsize=(4 * len(inertia), 3.5), squeeze=False)
    for ax, code in zip(axes.flat, inertia):
        ax.plot(k_values[code], inertia[code], marker='o')
        chosen = selected_k[code]
        ax.axvline(chosen, color='#D55E00', linestyle='--', linewidth=1)
        ax.scatter(chosen, inertia[code][k_values[code].index(chosen)],
                   color='#D55E00', s=40, zorder=3)
        ax.set(title=f'{code}: selected k = {chosen}',
               xlabel='Number of centers', ylabel='KMeans inertia')
        positions = np.unique(np.linspace(0, len(k_values[code]) - 1,
                                              min(6, len(k_values[code])), dtype=int))
        ax.set_xticks(np.asarray(k_values[code])[positions])
        ax.grid(axis='y', alpha=0.2)
    fig.tight_layout()
    plt.show()
    return fig


def plot_kappa(results):
    """Compares validation kappa. Bars do not imply an uncertainty estimate."""
    table = results.pivot(index='code', columns='method', values='kappa')
    summaries = [c for c in ['Macro', 'Overall (Sources)'] if c in table.index]
    codes = [c for c in results['code'].drop_duplicates() if c not in summaries]
    table = table.reindex(index=summaries + codes, columns=results['method'].drop_duplicates())
    ax = table.plot.bar(figsize=(max(7, len(table) * 1.4), 4.5), width=0.8,
                        color=[METHOD_COLORS[c] for c in table.columns])
    ax.set_xticklabels([{'Macro': 'Macro\n(binary)', 'Overall (Sources)': 'Overall\n(multiclass)'}.get(c, c)
                        for c in table.index])
    ax.axhline(0, color='grey', linewidth=0.8)
    ax.set(xlabel='', ylabel='Validation κ', title='')
    ax.tick_params(axis='x', rotation=0)
    handles, names = ax.get_legend_handles_labels()
    ax.get_legend().remove()
    ax.figure.suptitle('Agreement on the fixed validation set', y=.99, fontsize=12)
    ax.figure.legend(handles, names, loc='upper center', bbox_to_anchor=(.5, .93),
                     ncol=len(table.columns), frameon=False, fontsize=8)
    ax.grid(axis='y', alpha=0.2)
    ax.figure.tight_layout(rect=(0, 0, 1, .82))
    plt.show()
    return ax.figure


def plot_selection_curves(summary, full_human, coding_scheme, zoom_max=50,
                          strategy_labels=None, selection_note='', reference_label='Full human bank: mean centroid'):
    """
    Plots validation κ as the number of exemplars grows, one panel per summary or code.
    <summary>: selection_summary rows (mean_kappa, lower, upper for each scope/strategy/count/code).
    <full_human>: the full-human-bank κ rows drawn as a dotted reference line (empty to omit).
    <zoom_max>: x-axis limit of the main panels; the small inset shows the full range.
    <strategy_labels>: legend name for each strategy. <selection_note>: extra caption text.
    Returns the two figures: all codes together, and individual codes.
    """
    strategies = [s for s in ['Random', 'Central', 'Diverse', 'Human bank'] if s in summary['strategy'].values]
    colors = STRATEGY_COLORS
    linestyles = STRATEGY_LINESTYLES
    strategy_labels = strategy_labels or {s: s for s in strategies}
    titles = {'Macro': 'All codes: macro binary κ', 'Overall (Sources)': 'All codes: multiclass κ'}
    values = np.r_[summary[['lower', 'upper']].to_numpy(dtype=float).ravel(), full_human['kappa'].to_numpy(dtype=float)]
    values = values[np.isfinite(values)]
    limits = (max(-1, min(-.04, values.min() - .03)), min(1.03, values.max() + .03))
    handles = [plt.Line2D([], [], color=colors[s], linestyle=linestyles[s], linewidth=1.8,
                          label=strategy_labels[s]) for s in strategies]
    if not full_human.empty:
        handles.append(plt.Line2D([], [], color='black', linestyle=':', linewidth=1.5, label=reference_label))

    def draw_panels(scope, codes, heading, ncols):
        nrows = int(np.ceil(len(codes) / ncols))
        height = 4.8 * nrows + .9
        fig = plt.figure(figsize=(6 * ncols, height))
        panels = fig.add_gridspec(nrows, ncols, hspace=.18, wspace=.2)
        for position, code_name in enumerate(codes):
            small_grid = panels[position // ncols, position % ncols].subgridspec(
                2, 2, height_ratios=[3.2, 1.1], hspace=.45)
            ax = fig.add_subplot(small_grid[0, :])
            inset = fig.add_subplot(small_grid[1, 1])
            frame = summary.loc[summary['scope'].eq(scope) & summary['code'].eq(code_name)]
            full_max = float(frame['count'].max())
            zoom_end = min(float(zoom_max), full_max)
            inset.axvspan(0, zoom_end, color='#626A71', alpha=.09, zorder=0)
            for strategy in strategies:
                rows = frame.loc[frame['strategy'].eq(strategy)].sort_values('count')
                x = rows['count'].to_numpy(dtype=float)
                y = rows['mean_kappa'].to_numpy(dtype=float)
                low, high = rows['lower'].to_numpy(dtype=float), rows['upper'].to_numpy(dtype=float)
                for panel in [ax, inset]:
                    panel.fill_between(x, low, high, color=colors[strategy], alpha=.12, linewidth=0)
                    panel.plot(x, y, color=colors[strategy], linestyle=linestyles[strategy], linewidth=1.8)
            human = full_human.loc[full_human['code'].eq(code_name), 'kappa']
            for panel in [ax, inset]:
                if len(human) and pd.notna(human.iloc[0]):
                    panel.axhline(human.iloc[0], color='black', linestyle=':', linewidth=1.2)
                panel.set_ylim(*limits)
                panel.grid(axis='y', color=GRID_COLOR, alpha=.6, linewidth=.6)
                panel.spines[['top', 'right']].set_visible(False)
            ax.set(xlim=(0, zoom_end), title=titles.get(code_name, code_name),
                   xlabel='Requested positive draws per code' if scope == 'combined' else f'Positive draws for {code_name}',
                   ylabel='Validation κ')
            inset.set(xlim=(0, full_max), xticks=[0, int(full_max // 2), int(full_max)])
            inset.set_title('Full range', fontsize=9, pad=3)
            inset.tick_params(labelsize=8, length=2, pad=1)
        fig.suptitle(heading, fontsize=12, y=.99)
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, 1 - .35 / height), ncol=2, fontsize=9)
        caption = 'Shading: 2.5th–97.5th percentiles across draws on the fixed validation set.'
        if 'Human bank' in strategies:
            caption += '\nAll draws use no replacement; human-bank curves stop at the common central/diverse counts.'
        elif scope == 'combined':
            caption += '\nEach code stops at its available pool size.'
        caption += '\n' + selection_note
        if scope == 'focal' and 'Human bank' in strategies:
            caption += '\nOther codes use the same human-bank subsample within each repetition.'
        fig.text(.5, .015, caption, ha='center', fontsize=9, color=NOTE_COLOR)
        fig.subplots_adjust(left=.75 / (6 * ncols), right=.98,
                            bottom=(1.1 if scope == 'focal' else 1.0) / height, top=1 - 1.2 / height)
        plt.show()
        return fig

    combined_codes = [c for c in ['Macro', 'Overall (Sources)']
                      if (summary['scope'].eq('combined') & summary['code'].eq(c)).any()]
    source = 'Human' if 'Human bank' in strategies else 'Machine'
    return (draw_panels('combined', combined_codes, source + ' sentence selection: all codes', len(combined_codes)),
            draw_panels('focal', list(coding_scheme), source + ' sentence selection: individual codes', 2))


def plot_matched_comparison(matched_summary, full_human, coding_scheme, human_budgets,
                            human_negative_budgets=None, strategy_labels=None, selection_note='',
                            reference_label='Full human bank: mean centroid'):
    """
    Compares κ for each strategy at the same per-code exemplar counts (points: mean; lines: 2.5th-97.5th percentiles).
    <matched_summary>: selection_summary rows with scope 'matched'.
    <full_human>: the fixed full-human-bank κ rows (machine mode), or empty (human mode).
    <human_budgets>: positive exemplars per code. <human_negative_budgets>: PEPPER negatives per code, or None.
    """
    strategies = [s for s in ['Random', 'Central', 'Diverse', 'Human bank'] if s in matched_summary['strategy'].values]
    colors = STRATEGY_COLORS
    markers = STRATEGY_MARKERS
    strategy_labels = strategy_labels or {s: s for s in strategies}
    metrics = [c for c in ['Macro', 'Overall (Sources)'] if c in matched_summary['code'].values]
    codes = metrics + list(coding_scheme)
    x = np.arange(len(codes), dtype=float)
    fig, ax = plt.subplots(figsize=(max(10, 1.8 * len(codes)), 5.5))
    offsets = np.linspace(-.27, .27, len(strategies)) if full_human.empty else [-.27, -.09, .09]
    for offset, strategy in zip(offsets, strategies):
        rows = matched_summary.loc[matched_summary['strategy'].eq(strategy)].set_index('code').reindex(codes)
        y = rows['mean_kappa'].to_numpy(dtype=float)
        low, high = rows['lower'].to_numpy(dtype=float), rows['upper'].to_numpy(dtype=float)
        ax.vlines(x + offset, low, high, color=colors[strategy], linewidth=1.4)
        ax.scatter(x + offset, y, color=colors[strategy], marker=markers[strategy], s=50,
                   edgecolors='white', linewidths=.7, label=strategy_labels[strategy], zorder=3)
    if not full_human.empty:
        ax.scatter(x + .27, full_human.set_index('code').reindex(codes)['kappa'], color='black', marker='X',
                   s=60, label=reference_label, zorder=3)
    names = {'Macro': 'All codes\n(macro binary κ)', 'Overall (Sources)': 'All codes\n(multiclass κ)'}
    labels = [names.get(c, f'{c}\nn = {human_budgets.get(c, "")}') for c in codes]
    if human_negative_budgets is not None:
        labels = [names.get(c, f'{c}\n{human_budgets.get(c, "")} positive\n'
                              f'{human_negative_budgets.get(c, "")} negative') for c in codes]
    ax.set_xticks(x, labels)
    ax.set(xlim=(-.6, len(codes) - .4), ylabel='Validation κ')
    ax.axhline(0, color='#999999', linewidth=.7)
    ax.grid(axis='y', color=GRID_COLOR, alpha=.6, linewidth=.6)
    ax.tick_params(axis='x', length=0, pad=9)
    fig.suptitle('Sentence selection at equal positive counts per code' if human_negative_budgets is None
                 else 'Sentence selection at the human-bank positive and negative counts', y=.99, fontsize=12)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, 1.27), ncol=2, fontsize=9)
    counts = 'Positive draws per code: ' + ', '.join(f'{c} = {human_budgets[c]}' for c in coding_scheme) + '.'
    if full_human.empty:
        detail = 'Human bank samples only bank sentences, without replacement. Intervals: 2.5th–97.5th percentiles.'
    else:
        detail = f'Full human bank: {int(full_human.sentence_count.iloc[0])} unique sentences; fixed reference, no resampling.'
    fig.text(.5, .02, counts + '\n' + detail + '\n' + selection_note, ha='center', fontsize=9, color=NOTE_COLOR)
    fig.subplots_adjust(left=.08, right=.98, bottom=.25 if human_negative_budgets is None else .30, top=.73)
    plt.show()
    return fig


# %% [markdown] id="7e8c3295"
# ## **Data**
#
# Loads the coded corpus for the chosen dataset and the human exemplar bank, and puts both into the same form:
# one 0/1 column per code (PEPPER exemplars can also be unknown, NaN). Check the printed counts against what you expect.

# %% id="4667cca6" tags=["corpus_data"]
# Load the selected dataset and put the human labels into a common form.

if dataset == 'sources_v2':
    df_data = pd.read_excel(sources_data_raw)
    sentence_column = 'response'
    df_data['original_code'] = df_data['updated_code']

    # Keep the original code so that the S-to-P combination can be checked.

    if coding_scheme == sources_combined:
        df_data['updated_code'] = df_data['updated_code'].replace({'S': 'P'})

    if not df_data['updated_code'].isin(coding_scheme).all():
        raise ValueError('Sources contains an unknown or missing updated_code.')

    for code_name in coding_scheme:
        df_data[code_name] = (df_data['updated_code'] == code_name).astype(int)

    df_data = convert_codes_column(df_data)

else:
    df_data = pd.read_excel(PEPPER_data_raw, sheet_name='MAIN')
    sentence_column = 'Sentence'

    # DA is present if any of DRR, FLS, or URR is present.
    # These corpus columns are recorded as 0 or 1.

    df_data['DA'] = df_data[['DRR', 'FLS', 'URR']].max(axis=1)
    df_data = add_codes_column(df_data)

# Save the original Excel row before removing sentences with no text.
# The header is row 1, so the first data row is row 2.

df_data['source_row'] = df_data.index + 2
has_text = (
    df_data[sentence_column].notna()
    & df_data[sentence_column].fillna('').str.strip().ne('')
)

print(f'Omitting {int((~has_text).sum())} rows without sentence text.')
df_data = df_data.loc[has_text].reset_index(drop=True)

if not df_data[coding_scheme].isin([0, 1]).all().all():
    raise ValueError('The selected corpus labels must be recorded as 0 or 1.')

print(f'{dataset}: {len(df_data):,} sentences, including repeated text.')
display(pd.Series(sum_codes(df_data), name='Human code count').to_frame())

# %% [markdown] id="3b61b116"
# ### **Human Exemplars**

# %% [markdown] id="89225b30"
# Main analyses use these exemplar counts (positive / negative), as of the October 2026 data. The next cell prints the current counts:
#
# - **Sources:** **93 unique sentences**. L: **28 / 65**; O: **31 / 62**; P: **15 / 78**; S: **19 / 74**. With S merged into P, P has **34 / 59**. Negatives are the other-code exemplars and are used only by the negative-aware method.
# - **PEPPER:** **185 unique sentences**, including **145 with positive labels** and **40 negative-only**. DA: **51 / 16**; S3: **26 / 12**; S5: **34 / 8**; S7: **36 / 4**. DA cutoff negatives are pooled from its subcodes: DRR **5**, FLS **5**, URR **6**. Mean and KMeans centroids fit positives; negatives calibrate their cutoffs.

# %% id="8e0a7750" tags=["human_data"]
# Load the human-selected examples.
# Combine repeated bank annotations by exact text; keep all corpus rows.

if dataset == 'sources_v2':
    human_raw = pd.read_excel(sources_human_exemplars)

    if coding_scheme == sources_combined:
        human_raw['code'] = human_raw['code'].replace({'S': 'P'})

    # Each Sources example has a code, so its other code columns are known zeros.

    human_labels = pd.get_dummies(human_raw['code']).astype(float)
    human_labels['text'] = human_raw['text']

    df_human = (
        human_labels.groupby('text', sort=False).max()
        .reindex(columns=coding_scheme, fill_value=0)
    )
    df_human = df_human.reset_index()

else:
    human_raw = pd.concat(
        pd.read_excel(pepper_human_exemplars, sheet_name=None).values(),
        ignore_index=True
    )

    human_code_names = {
        'Did real research': 'DRR', 'Felt like a scientist': 'FLS',
        'Understanding real research': 'URR', 'Failure': 'S3',
        'Iteration': 'S5', 'Decision making': 'S7'
    }

    # Keep the codes used in this analysis and their recorded positive/negative labels.

    human_raw = human_raw.loc[human_raw['code'].isin(human_code_names)].copy()
    human_raw['short_code'] = human_raw['code'].map(human_code_names)
    human_raw['label'] = human_raw['exemplar_type'].map(
        {'Typical': 1, 'Atypical': 1, 'Close but no': 0}
    )

    if human_raw['label'].isna().any():
        raise ValueError('Unrecognized exemplar_type in the human bank.')

    # A code that was not recorded stays unknown (NaN), rather than becoming zero.

    df_human = (
        human_raw.pivot_table(
            index='exemplar_text', columns='short_code', values='label',
            aggfunc='max', sort=False
        )
        .reindex(columns=['DRR', 'FLS', 'URR', 'S3', 'S5', 'S7'])
    )

    # Any positive subcode establishes a positive DA exemplar.
    # For DA cutoff calibration, pool the recorded negatives from its three subcodes.
    # A positive takes priority; other unrecorded labels still stay unknown.

    children = df_human[['DRR', 'FLS', 'URR']]
    df_human['DA'] = np.where(
        children.eq(1).any(axis=1), 1,
        np.where(children.eq(0).any(axis=1), 0, np.nan)
    )
    df_human = (
        df_human[coding_scheme].reset_index()
        .rename(columns={'exemplar_text': 'text'})
    )

    # Keep the child annotations in human_raw. A sentence with no known labels
    # in the selected scheme cannot be used to fit or calibrate those codes.

    df_human = df_human.loc[
        df_human[coding_scheme].notna().any(axis=1)
    ].reset_index(drop=True)

display(pd.DataFrame({
    'Positive': df_human[coding_scheme].eq(1).sum(),
    'Negative': df_human[coding_scheme].eq(0).sum(),
    'Unknown': df_human[coding_scheme].isna().sum()
}))
print(f'{len(df_human)} unique human exemplar sentences with usable labels.')

# %% [markdown] id="40f50bf2"
# ### **Human Central/Diverse Labels**
#
# `exemplar_selection_source` chooses human C/D labels or machine selection. Independently, `random_selection_source` chooses `training` (default) or `human_central_diverse`. Training Random draws from that code's examples in the entire training partition. Human C/D Random draws from the union of that code's annotated C and D bank examples, ignoring type. The optional validation-text exclusion applies to human pools. Rerun from Setup after changing either setting.
#
# Human mode reads `raw_data/human_central-diverse.xlsx`, using `Final`, `Type`, or `RM type`. Exact text and code identify examples. Central, diverse, Random, and the sampled human-bank comparison all select **without replacement** within each code and repetition. The human-bank comparison samples **only** from the original `centroids_expanded.xlsx` bank, independently of the Random source and C/D types, at the same per-code counts as the other three methods. Repeated subsampling produces a mean and 2.5th–97.5th percentile interval. Human-bank curves cover the common C/D count range. For individual-code curves, all methods share the same human-bank subsample for the other codes within each repetition.
#
# In **machine mode**, the human-bank comparison continues to use the full fixed bank without sampling. The earlier main centroid analyses also continue to use the full bank. Machine central, diverse, and Random selections, calibration-fold shuffles, and PEPPER negative selections use no replacement. PEPPER continues to use corpus selection because this C/D workbook contains Sources labels only.

# %% id="efa036fd" tags=["human_selection_labels"]
# Read the human central/diverse annotations without changing the exemplar bank.
# Match exact text and code, never Excel row order or the unnamed index column.
human_selection_annotations = pd.DataFrame()

if needs_human_selection_labels:
    typed = pd.read_excel(human_selection_path)
    required_columns = ['text', 'code', human_selection_column]
    missing_columns = [c for c in required_columns if c not in typed]
    if missing_columns:
        raise ValueError(f'Human selection file is missing columns: {missing_columns}.')

    typed = typed[required_columns].copy()
    if typed[required_columns].isna().any().any():
        raise ValueError('Human selection text, code, and selected type column must be filled in.')
    if not typed['text'].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError('Human selection text must contain nonempty sentences.')

    typed['code'] = typed['code'].astype(str).str.strip().str.upper()
    typed['selection_type'] = (
        typed[human_selection_column].astype(str).str.strip().str.casefold()
        .map({'c': 'Central', 'central': 'Central', 'd': 'Diverse', 'diverse': 'Diverse'})
    )
    if typed['selection_type'].isna().any():
        raise ValueError(f'{human_selection_column} must contain C/D or Central/Diverse.')
    if not typed['code'].isin(sources_base).all():
        raise ValueError('Human central/diverse file contains an unknown Sources code.')
    if coding_scheme == sources_combined:
        typed['code'] = typed['code'].replace({'S': 'P'})

    conflicts = typed.groupby(['text', 'code'])['selection_type'].nunique()
    if conflicts.gt(1).any():
        raise ValueError('Conflicting central/diverse types for the same text and analysis code.')
    typed = typed[['text', 'code', 'selection_type']].drop_duplicates()

    human_index = pd.Series(df_human.index, index=df_human['text'])
    typed['human_index'] = typed['text'].map(human_index)
    if typed['human_index'].isna().any():
        raise ValueError('Some typed sentences are absent from the human bank. Update the bank first.')
    typed['human_index'] = typed['human_index'].astype(int)
    for row in typed.itertuples(index=False):
        if df_human.loc[row.human_index, row.code] != 1:
            raise ValueError(f'Human selection code disagrees with the bank for {row.code}.')

    human_selection_annotations = typed.reset_index(drop=True)
    print(f'Human central/diverse labels: {human_selection_path.name}; column: {human_selection_column}.')
    display(typed.groupby(['code', 'selection_type']).size().unstack(fill_value=0)
            .reindex(coding_scheme).rename_axis(columns=None))

# %% [markdown] id="358b6f5d"
# ## **Embeddings**
#
# Every corpus sentence and human exemplar becomes one vector. With `use_saved = True` the vectors are read from
# `saved_embeddings/`; otherwise they are created with `embedding_model` (GPU needed) and saved for next time.
# All vectors are normalized, so a dot product is a cosine similarity.

# %% [markdown] id="52c6aa06"
# ### **Load Saved Embeddings or Create New Embeddings**

# %% id="5a6e0c46" tags=["embeddings"]
# Combine the text to embed once, without changing whitespace or punctuation.
# Repeated corpus rows are restored after the text lookup below.

sentences = pd.concat(
    [df_data[sentence_column], df_human['text']], ignore_index=True
).drop_duplicates().tolist()

if use_saved:
    # Load the requested model's saved vectors.

    if not embedding_file.exists():
        raise FileNotFoundError(
            f'{embedding_file.name} was not found. Check the folder, or set use_saved=False.'
        )

    with open(embedding_file, 'rb') as f:
        # Only load .pkl files you created: unpickling a file can run code.
        # The file also stores every sentence, so keep it as private as the raw data.

        saved_embeddings = pkl.load(f)

    # The saved text and vectors must use the same model and embedding settings.

    if (
        saved_embeddings['model'] != embedding_model
        or saved_embeddings['revision'] != embedding_revision
        or saved_embeddings['prompt'] != embedding_prompt
        or saved_embeddings['max_length'] != max_seq_length
    ):
        raise ValueError(
            'Saved embeddings use different model/settings. Check Setup or create new embeddings.'
        )

    # positions gives the saved-vector row for each exact sentence in sentences.
    # A position of -1 means the requested sentence was not found.

    saved_text_index = pd.Index(saved_embeddings['sentences'])
    positions = saved_text_index.get_indexer(sentences)

    if (positions < 0).any():
        raise ValueError(
            f'{int((positions < 0).sum())} sentences are missing from the saved embeddings.'
        )

    text_embeddings = np.asarray(
        saved_embeddings['embeddings'][positions], dtype=np.float32
    )
    print(f'Loaded {len(sentences):,} unique sentence embeddings from {embedding_file.name}.')
    del saved_embeddings

else:
    # Create new embeddings. HF_TOKEN is optional if already logged in.
    # Never paste a token into a cell: set it as an environment variable, or in Colab's Secrets panel.
    # Requires sentence-transformers >= 3.0 and transformers >= 4.51.

    import torch
    from sentence_transformers import SentenceTransformer

    device = (
        'cuda' if torch.cuda.is_available()
        else ('mps' if torch.backends.mps.is_available() else 'cpu')
    )
    if hosted_colab and device != 'cuda':
        raise RuntimeError(
            'For new Qwen8B embeddings, choose Runtime > Change runtime type > GPU, '
            'or set use_saved=True to load the existing embeddings.'
        )

    print(f'Embedding device: {device}. Lower batch_size in Setup if memory runs out.')

    model = SentenceTransformer(
        embedding_model, revision=embedding_revision,
        token=os.environ.get('HF_TOKEN'), device=device,
        model_kwargs={'torch_dtype': torch.float32 if device == 'cpu' else torch.float16}
    )

    model.max_seq_length = max_seq_length
    model.default_prompt_name = None
    model.eval()

    print(f'Starting embeddings for {len(sentences):,} sentences on {device}...')
    start_time = time.time()

    with torch.no_grad():
        text_embeddings = model.encode(
            sentences, prompt=embedding_prompt, batch_size=batch_size,
            normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True
        )

    print(f'Embeddings completed in {time.time() - start_time:.2f} seconds.')

    # Save the sentence lookup and model settings together with the new vectors.

    embedding_file.parent.mkdir(parents=True, exist_ok=True)

    with open(embedding_file, 'wb') as f:
        pkl.dump({
            'model': embedding_model, 'revision': embedding_revision,
            'prompt': embedding_prompt, 'max_length': max_seq_length,
            'sentences': sentences, 'embeddings': text_embeddings
        }, f)

    del model

# Normalize in float32 for cosine similarity.
# Saved vectors can have small rounding differences.

text_embeddings = np.asarray(text_embeddings, dtype=np.float32)

if not np.isfinite(text_embeddings).all() or (np.linalg.norm(text_embeddings, axis=1) == 0).any():
    raise ValueError('Embeddings must be finite, nonzero vectors.')

text_embeddings /= np.linalg.norm(text_embeddings, axis=1, keepdims=True)

# Look up each original row's vector by exact text. Duplicate sentences therefore
# get the same vector, while corpus order and repeated rows are preserved.

text_index = pd.Index(sentences)
embeddings = text_embeddings[text_index.get_indexer(df_data[sentence_column])]
human_embeddings = text_embeddings[text_index.get_indexer(df_human['text'])]

print(f'Corpus embeddings: {embeddings.shape}; human exemplar embeddings: {human_embeddings.shape}.')

# %% [markdown] id="0ee98289"
# ## **2D Projections**
#
# A picture of the embedding space: each point is a corpus sentence, colored by its human code. One PCA is fitted
# here and reused for every later projection graph, so positions are comparable across graphs.

# %% [markdown] id="e2044811"
# ### **All Data**

# %% id="4fe03803" tags=["pca_all"]
# Fit one PCA to the corpus and reuse it for every main plot and centroid.

pca = PCA(n_components=2, svd_solver='randomized', random_state=random_seed)
projection = pca.fit_transform(embeddings)

print(
    f'Variation shown: PC1 = {pca.explained_variance_ratio_[0]:.1%}, '
    f'PC2 = {pca.explained_variance_ratio_[1]:.1%}.'
)

plot_projection(projection, df_data, coding_scheme, f'{dataset}: human codes (PCA)');

# %% [markdown] id="72fb13d0"
# ### **Per Code**

# %% id="62515623" tags=["pca_per_code"]
# Use the same PCA coordinates for the separate code panels.
# Multi-coded PEPPER sentences can appear in more than one panel.

plot_projection(
    projection, df_data, coding_scheme,
    f'{dataset}: human codes (PCA)', per_code=True
);

# %% [markdown] id="2c26b2fc"
# ### **UMAP** [Optional]
#
# Optional alternate 2D projection option, set `use_umap = True` in Setup to show.
# In my experience, does not provide a better visualization than PCA (a bit more confusing/overlap).

# %% id="a5378c8e" tags=["optional_umap"]
if use_umap:
    import umap

    # Fit once, then use the same coordinates for this optional view.

    umap_model = umap.UMAP(
        n_components=2, n_neighbors=30, min_dist=.05,
        metric='cosine', random_state=random_seed
    )
    umap_projection = umap_model.fit_transform(embeddings)

    plot_projection(
        umap_projection, df_data, coding_scheme, f'{dataset}: human codes (UMAP)'
    )

# %% [markdown] id="56cce8c2"
# ## **Centroid Analysis**
#
# Builds code centroids from the **full human exemplar bank** with three methods (mean centroid, KMeans subcentroids,
# negative-aware centroid) and measures agreement (Cohen's κ) with the human codes on a fixed validation half of the corpus.

# %% [markdown] id="bb89d9be"
# ### **Validation Split**
#
# Splits the corpus once into a candidate (training) half and a fixed validation half, keeping each code's share
# the same in both. Every κ in this notebook is measured on the same validation half.

# %% id="6b92c06e" tags=["validation_split"]
# Stratify Sources by code, and PEPPER by its combination of codes.

strata = df_data[coding_scheme].astype(str).agg(''.join, axis=1)
stratum_counts = strata.value_counts()

if (strata.map(stratum_counts) < 2).any():
    # A singleton cannot appear on both sides of a stratified split.
    # Use the largest stratum for splitting only; keep its actual labels unchanged.

    strata = strata.where(strata.map(stratum_counts) >= 2, stratum_counts.idxmax())

# Split row indices once. Every method below reuses these validation rows.

training_indices, validation_indices = train_test_split(
    np.arange(len(df_data)), test_size=validation_fraction,
    stratify=strata, random_state=random_seed
)
training_indices = np.sort(training_indices)
validation_indices = np.sort(validation_indices)

validation_embeddings = embeddings[validation_indices]
validation_labels = df_data.loc[validation_indices, coding_scheme].to_numpy(dtype=float)

display(pd.DataFrame({
    'All positives': df_data[coding_scheme].sum(),
    'Candidate-pool positives': df_data.loc[training_indices, coding_scheme].sum(),
    'Validation positives': df_data.loc[validation_indices, coding_scheme].sum()
}))
print(f'{len(training_indices):,} candidate rows; {len(validation_indices):,} fixed validation rows.')

# %% [markdown] id="a3403d86"
# ### **Human Examples for the Centroids**

# %% id="5aff35f4" tags=["main_examples"]
# All centroid methods use this same full set of human examples.
# Positive examples fit the centroids. Recorded negatives remain available for cutoffs.

training_vectors = human_embeddings
training_labels = df_human[coding_scheme].to_numpy(dtype=float)
training_texts = df_human['text'].to_numpy()

selected_by_code = {}

for j, code_name in enumerate(coding_scheme):

    # Use every human positive for this code; there is no random subsampling here.

    selected_by_code[code_name] = np.flatnonzero(training_labels[:, j] == 1)

    if len(selected_by_code[code_name]) == 0:
        raise ValueError(f'No human positive examples were found for {code_name}.')

# Keep negative-only PEPPER examples as well as the positive examples.
# Missing labels stay unknown, and do not become negative training decisions.

selected_rows = np.arange(len(df_human))
selected_sentence_count = len(pd.unique(training_texts))

display(pd.DataFrame({
    'Positive examples for centroid': {c: len(rows) for c, rows in selected_by_code.items()},
    'Negative exemplars': df_human[coding_scheme].eq(0).sum()
}))

print(f'All methods use the full human bank: {selected_sentence_count} unique sentence texts.')
if dataset == 'sources_v2':
    print('Other-code exemplars provide the negatives for the negative-aware method only.')
else:
    print("Negatives calibrate cutoffs only; DA pools its three subcodes' recorded negatives.")

# Filled by the three centroid cells below: {method name: {code: centers}}.
# It lives here so rerunning one centroid cell does not erase the others.

centroids = {}

# %% [markdown] id="a2bcad4d"
# ### **Mean Centroid**
#
# #### **Centroid Determination and Graph**

# %% id="d235a488" tags=["mean_centroids"]
# Average each code's normalized positive vectors, then normalize the mean.
# The dot product with this vector gives cosine similarity to the centroid.

mean_centroids = {}

for code_name in coding_scheme:
    center = training_vectors[selected_by_code[code_name]].mean(axis=0)
    center = center / max(np.linalg.norm(center), 1e-12)
    mean_centroids[code_name] = center.reshape(1, -1)

centroids['Mean centroid'] = mean_centroids

# Transform the centers using the existing PCA; do not fit a new projection.

centroid_projection = {
    code_name: pca.transform(centers)
    for code_name, centers in mean_centroids.items()
}
plot_projection(
    projection, df_data, coding_scheme, 'Mean centroids (PCA)',
    centroid_coords=centroid_projection
);

# %% [markdown] id="f5a195c1"
# ### **KMeans Subcentroids**
#
# Some codes cover several kinds of sentence, so one average may sit between them. Each code's exemplars are split
# into up to `max_elbow_k` clusters; the elbow graph shows the count chosen for each code (dashed line).

# %% id="70dc9c33" tags=["elbow"]
# Fit the elbow curve using only the human positives for each code.
# Inertia is the sum of squared distances to the fitted cluster centers.

inertia = {}
k_values = {}
n_subcentroids = {}

for code_name in coding_scheme:

    positive_vectors = training_vectors[selected_by_code[code_name]]
    models = fit_elbow_models(positive_vectors)

    k_values[code_name] = list(range(1, len(models) + 1))
    inertia[code_name] = [model.inertia_ for model in models]

    # The elbow rule is in choose_elbow_k() (Helper Functions).
    # A flat/straight curve, or fewer than three tested counts, keeps one center.

    n_subcentroids[code_name] = choose_elbow_k(inertia[code_name])

display(pd.DataFrame({
    'Human positives': {c: len(selected_by_code[c]) for c in coding_scheme},
    'Selected subcentroids': n_subcentroids
}))

plot_elbow(inertia, k_values, n_subcentroids);

# %% [markdown] id="20664ce1"
# #### **Centroid Determination and Graph**

# %% id="6c069b88" tags=["kmeans_centroids"]
# Fit subcentroids using only that code's positive examples.
# A code's score is its largest cosine similarity to one of its subcentroids.

kmeans_centroids = {}

for code_name in coding_scheme:
    positive_vectors = training_vectors[selected_by_code[code_name]]

    # Use the count chosen above for this code; validation did not choose it.

    kmeans = KMeans(
        n_clusters=n_subcentroids[code_name],
        n_init=kmeans_n_init, random_state=random_seed
    )
    centers = kmeans.fit(positive_vectors).cluster_centers_
    centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12)
    kmeans_centroids[code_name] = centers

centroids['KMeans subcentroids'] = kmeans_centroids

# Use the same fitted PCA to show the subcentroids.

centroid_projection = {
    code_name: pca.transform(centers)
    for code_name, centers in kmeans_centroids.items()
}
plot_projection(
    projection, df_data, coding_scheme, 'KMeans subcentroids (PCA)',
    centroid_coords=centroid_projection
);

# %% [markdown] id="4558dc39"
# ### **Negative-Aware Centroids**

# %% [markdown] id="c5e5df06"
# #### **Centroid and Anti-Centroid Determination and Graph**
#
# For `sources_v2` only. PEPPER's negative exemplars are used for cutoff calibration.

# %% id="8f5ab5d9" tags=["negative_centroids"]
# Sources has known negatives: examples assigned to one of the other codes.

anti_centroids = {}

if dataset == 'sources_v2':
    for j, code_name in enumerate(coding_scheme):
        negative_rows = selected_rows[training_labels[selected_rows, j] == 0]
        negative_vectors = training_vectors[negative_rows]
        positive_center = mean_centroids[code_name][0]

        # Negatives closer to the positive centroid receive more weight.
        # The weight is a sigmoid of the cosine similarity; 8 sets how sharply it rises.

        weights = 1 / (1 + np.exp(-8 * (negative_vectors @ positive_center)))
        anti_center = np.average(negative_vectors, axis=0, weights=weights)
        anti_center /= max(np.linalg.norm(anti_center), 1e-12)
        anti_centroids[code_name] = anti_center

    # The positive part of this method is the mean centroid; the anti-centroid is subtracted when scoring.

    centroids['Negative-aware centroid'] = mean_centroids

    # Plot both sets of centers on the existing PCA coordinates.

    positive_projection = {
        code_name: pca.transform(value)
        for code_name, value in mean_centroids.items()
    }
    negative_projection = {
        code_name: pca.transform(value.reshape(1, -1))
        for code_name, value in anti_centroids.items()
    }
    plot_projection(
        projection, df_data, coding_scheme, 'Positive and negative centroids (PCA)',
        centroid_coords=positive_projection, negative_coords=negative_projection
    )

else:
    print('Negative-aware centroid comparison is only run for Sources in this notebook.')

# %% [markdown] id="28b11aeb"
# ### **Kappa**
#
# Scores each validation sentence against every code, turns the scores into predicted codes, and compares them with
# the human codes using Cohen's κ.

# %% [markdown] id="bc47d57d"
# #### **Similarity Scores**

# %% id="75d969b9" tags=["main_scoring"]
# Score with the full-dimensional centroids already fitted above.
# Each method gives one score per validation sentence and code (see centroid_scores()).

main_scores = {}

for method, centers in centroids.items():

    anti_centers = anti_centroids if method == 'Negative-aware centroid' else None
    main_scores[method] = centroid_scores(validation_embeddings, centers, anti_centers)

# %% [markdown] id="4833f07c"
# #### **PEPPER Cutoffs**
#
# Need to determine a cutoff for PEPPER to account for non-mutually exclusive coding and `No Code`.

# %% id="03483035" tags=["main_calibration"]
# PEPPER only: find each code's similarity cutoff from the human bank, never from validation sentences.
# Each human example is scored by centroids fitted WITHOUT it (5-fold out-of-fold scores), then the
# cutoff with the best κ against the known human decisions is kept. Sources needs no cutoff (highest score wins).
# The Exemplar Analysis repeats this calculation, vectorized, for every draw (see Kappa Calculation).

main_cutoffs = {}
calibration_rows = []

for method in centroids:

    cutoffs = np.zeros(len(coding_scheme))

    if dataset == 'pepper':

        for j, code_name in enumerate(coding_scheme):

            positives = selected_by_code[code_name]
            known_rows = selected_rows[np.isfinite(training_labels[selected_rows, j])]

            cutoff = np.nan
            rule = 'Unsupported: fewer than two fitted positives'
            fold_subcentroids = []

            if len(positives) >= 2:

                # Spread positive examples over folds, keeping a positive in every fit.

                n_folds = min(5, len(positives))
                folds = np.full(len(training_vectors), -1)
                fold_rng = np.random.default_rng(random_seed + j)

                shuffled = fold_rng.permutation(positives)
                folds[shuffled] = np.arange(len(shuffled)) % n_folds

                for value in [0, 1]:

                    other_rows = np.setdiff1d(
                        known_rows[training_labels[known_rows, j] == value], positives
                    )

                    shuffled = fold_rng.permutation(other_rows)
                    folds[shuffled] = np.arange(len(shuffled)) % n_folds

                # Predict each fold using only the remaining human positives.

                oof_scores = np.full(len(training_vectors), np.nan)

                for fold in range(n_folds):

                    fit_rows = positives[folds[positives] != fold]
                    held_rows = known_rows[folds[known_rows] == fold]

                    if method == 'KMeans subcentroids':

                        # Choose the elbow again using this fold's fit examples only.
                        # Held-out human examples must not help choose the cluster count.

                        # Use the same normalized endpoint-gap rule as the main elbow.

                        fold_models = fit_elbow_models(training_vectors[fit_rows])
                        chosen_k = choose_elbow_k([model.inertia_ for model in fold_models])

                        fitted = fold_models[chosen_k - 1].cluster_centers_.copy()
                        fold_subcentroids.append(chosen_k)

                    else:
                        fitted = training_vectors[fit_rows].mean(axis=0, keepdims=True)
                        fold_subcentroids.append(1)

                    fitted /= np.maximum(np.linalg.norm(fitted, axis=1, keepdims=True), 1e-12)

                    oof_scores[held_rows] = (
                        training_vectors[held_rows] @ fitted.T
                    ).max(axis=1)

                # Find the cutoff from known human decisions only.
                # Candidates are 0 and 15 quantiles of the out-of-fold scores; ties go to the cutoff closest to 0.

                values = oof_scores[known_rows]
                truth = training_labels[known_rows, j]

                if np.isfinite(values).all():

                    if (truth == 0).any():

                        candidates = np.unique(np.r_[
                            0., np.quantile(values, np.linspace(0, 1, 15))
                        ])

                        quality = np.array([
                            cohen_kappa_score(truth, (values >= t).astype(int))
                            for t in candidates
                        ])

                        best = np.flatnonzero(np.isclose(quality, np.nanmax(quality)))
                        cutoff = candidates[best[np.argmin(np.abs(candidates[best]))]]
                        rule = 'Out-of-fold κ'

                    else:

                        cutoff = np.quantile(values, .1)
                        rule = 'Positive-only: 10th percentile of out-of-fold similarity'

            cutoffs[j] = cutoff

            calibration_rows.append({
                'method': method, 'code': code_name, 'cutoff': cutoff, 'rule': rule,
                'known_positive': int((training_labels[known_rows, j] == 1).sum()),
                'known_negative': int((training_labels[known_rows, j] == 0).sum()),
                'Fold subcentroids': fold_subcentroids
            })

    main_cutoffs[method] = cutoffs

calibration = pd.DataFrame(calibration_rows)

if not calibration.empty:
    display(calibration)

# %% [markdown] id="87b26385"
# #### **Predictions and Agreement**

# %% id="c0764935" tags=["main_kappa"]
# Predict codes for the validation sentences and compare them with the human codes (Cohen's κ).
# predict_codes() and agreement_rows() are in Helper Functions.

kappa_rows = []
main_predictions = {}  # Kept for inspection: {method: 0/1 prediction array}

for method, scores in main_scores.items():

    main_predictions[method] = predict_codes(scores, main_cutoffs[method])
    kappa_rows += agreement_rows(method, validation_labels, main_predictions[method])

results = pd.DataFrame(kappa_rows)
results['sentence_count'] = selected_sentence_count
results['selected_row_count'] = len(selected_rows)

display(results.pivot(index='code', columns='method', values='kappa').round(3))

print(f'Each method used {selected_sentence_count} human sentence texts, '
      f'or {selected_sentence_count / len(df_data):.2%} of the corpus size.')
print(f'κ is evaluated on {len(validation_indices):,} fixed validation sentences; '
      'these are separate from the exemplar counts above.')

plot_kappa(results);

# %% [markdown] id="2ab6b049"
# ### **Full-Data Centroid Summary**
#
# The same κ on the whole coded corpus, for comparison with the validation numbers above.

# %% id="fe8ffc3d" tags=["full_corpus_kappa"]
# The same centroids and cutoffs, scored on the whole coded corpus instead of the validation half.
# This includes the sentences that were also human exemplars (see the overlap table below).

full_corpus_labels = df_data[coding_scheme].to_numpy(dtype=float)
full_corpus_rows = []
full_corpus_predictions = {}  # Kept for inspection: {method: 0/1 prediction array}

for method, centers in centroids.items():

    anti_centers = anti_centroids if method == 'Negative-aware centroid' else None
    scores = centroid_scores(embeddings, centers, anti_centers)

    full_corpus_predictions[method] = predict_codes(scores, main_cutoffs[method])
    full_corpus_rows += agreement_rows(method, full_corpus_labels, full_corpus_predictions[method])

full_corpus_results = pd.DataFrame(full_corpus_rows)

display(full_corpus_results.pivot(index='code', columns='method', values='kappa').round(3))

# Report exact-text overlap without silently filtering or changing the populations.

human_texts = set(df_human['text'])
exact_human_matches = df_data[sentence_column].isin(human_texts)

display(pd.DataFrame({
    'Evaluation': [f'Shared {validation_fraction:.0%} validation', 'Full coded corpus'],
    'Sentences': [len(validation_indices), len(df_data)],
    'Exact human-exemplar text matches retained': [
        int(exact_human_matches.iloc[validation_indices].sum()),
        int(exact_human_matches.sum())
    ]
}))

# %% [markdown] id="05b9a312"
# ## **Exemplar Analysis**
#
# Asks how many exemplars are needed, and which kind works best. Random, central, and diverse exemplar sets (and, in
# human mode, subsamples of the human bank) are drawn `n_repetitions` times at each count in `exemplar_counts`, and
# each draw is scored with the same validation κ as above. This is the slow part of the notebook (about 10 minutes at 250 repetitions).

# %% [markdown] id="fe286ea3"
# ### **Validation Split**
#
# Uses the same validation split as the **Centroid Analysis** section.

# %% id="33b40504" tags=["selection_setup"]
# Reuse the existing split for every selection method and graph.
# Exemplars are only ever drawn from the training rows (or the human bank); κ always uses validation_indices.
human_mode = effective_selection_source == 'human'
selection_strategies = ['Random', 'Central', 'Diverse'] + (['Human bank'] if human_mode else [])
human_budgets = {c: len(rows) for c, rows in selected_by_code.items()}
full_human = results.loc[results['method'] == 'Mean centroid'].copy()
validation_texts = set(df_data.loc[validation_indices, sentence_column])
human_reference_overlap = int(df_human['text'].isin(validation_texts).sum())

# Corpus rows and bank examples share one vector lookup, but keep separate pools.
candidate_rows = training_indices[
    df_data.loc[training_indices, coding_scheme].eq(1).any(axis=1).to_numpy()
]
selection_labels = df_data.loc[candidate_rows, coding_scheme].to_numpy(dtype=float)
selection_embeddings = embeddings[candidate_rows]
selection_texts = df_data.loc[candidate_rows, sentence_column].to_numpy()
corpus_pools = {
    c: np.flatnonzero(selection_labels[:, j] == 1) for j, c in enumerate(coding_scheme)
}

if needs_human_selection_labels:
    human_offset = len(candidate_rows)
    selection_embeddings = np.vstack([selection_embeddings, human_embeddings])
    selection_labels = np.vstack([selection_labels, df_human[coding_scheme].to_numpy(dtype=float)])
    selection_texts = np.r_[selection_texts, df_human['text'].to_numpy()]
    eligible_human = np.ones(len(df_human), dtype=bool)
    if human_exclude_validation_text:
        eligible_human &= ~df_human['text'].isin(validation_texts).to_numpy()
    human_bank_pools = {
        c: human_offset + np.flatnonzero(df_human[c].eq(1).to_numpy() & eligible_human)
        for c in coding_scheme
    }
    typed_pools = {}
    for c in coding_scheme:
        rows = human_selection_annotations.loc[
            human_selection_annotations['code'].eq(c), 'human_index'
        ].to_numpy(dtype=int)
        typed_pools[c] = human_offset + np.unique(rows[eligible_human[rows]])

# Candidate pools for each strategy: {strategy: {code: row positions in selection_embeddings}}.

if human_mode:
    random_pools = corpus_pools if effective_random_source == 'training' else typed_pools
    selection_pools = {'Random': random_pools, 'Human bank': human_bank_pools}
    for strategy in ['Central', 'Diverse']:
        selection_pools[strategy] = {}
        for c in coding_scheme:
            rows = human_selection_annotations.loc[
                human_selection_annotations['code'].eq(c)
                & human_selection_annotations['selection_type'].eq(strategy), 'human_index'
            ].to_numpy(dtype=int)
            selection_pools[strategy][c] = human_offset + np.sort(rows[eligible_human[rows]])
    strategy_labels = {
        'Random': 'Random (training data)' if effective_random_source == 'training' else 'Random (human C/D bank)',
        'Central': 'Human central', 'Diverse': 'Human diverse',
        'Human bank': 'Human bank (subsampled)'
    }
    human_reference_label = strategy_labels['Human bank']
    selection_note = f'Human labels: {human_selection_column}. Random: ' + ('training data.' if effective_random_source == 'training' else 'human C/D bank.')
    overlap_note = ('Validation text excluded from human pools.' if human_exclude_validation_text
                    else 'Validation-text overlap retained in human pools.')
    print(f'{int((~eligible_human).sum())} bank sentences excluded for validation-text overlap.')
else:
    # Machine mode retains its original corpus draws and fixed full-bank reference.
    selection_pools = {s: corpus_pools for s in selection_strategies}
    strategy_labels = {'Random': 'Random', 'Central': 'Machine central', 'Diverse': 'Machine diverse'}
    if effective_random_source == 'human_central_diverse':
        selection_pools['Random'] = typed_pools
        strategy_labels['Random'] = 'Random (human C/D bank)'
    human_reference_label = 'Full human bank: mean centroid'
    if human_reference_overlap:
        human_reference_label += ' (overlap retained)'
    selection_note = ('Machine central/diverse. Random: ' +
                      ('training data.' if effective_random_source == 'training' else 'human C/D bank.'))
    overlap_note = f'Full-bank reference retains {human_reference_overlap} validation-text matches.'

# Matched budgets: the per-code count every strategy can supply (at most the full human positive count).

strategy_pool_sizes = {
    s: {c: len(rows) for c, rows in pools.items()} for s, pools in selection_pools.items()
}
if any(n == 0 for sizes in strategy_pool_sizes.values() for n in sizes.values()):
    raise ValueError('An active code has no eligible examples. Check the annotations and validation exclusions.')
matched_budgets = {
    c: min([human_budgets[c]] + [strategy_pool_sizes[s][c] for s in selection_strategies])
    for c in coding_scheme
}
if not human_mode and effective_random_source == 'training' and matched_budgets != human_budgets:
    raise ValueError('Too few corpus candidates to match the full human bank.')

# PEPPER also matches the human bank's negative counts, drawn from corpus training negatives.

human_negative_budgets = None
negative_pools = {}
if dataset == 'pepper':
    human_negative_budgets = {c: int(df_human[c].eq(0).sum()) for c in coding_scheme}
    negative_pools = {
        c: training_indices[df_data.loc[training_indices, c].eq(0).to_numpy()]
        for c in coding_scheme
    }
    if any(human_negative_budgets[c] > len(negative_pools[c]) for c in coding_scheme):
        raise ValueError('Too few candidate negatives to match the human bank.')
    display(pd.DataFrame({
        'Corpus negative candidates': {c: len(rows) for c, rows in negative_pools.items()},
        'Full human negative count': human_negative_budgets
    }))

display(pd.DataFrame({
    **{strategy_labels[s] + ' available': sizes for s, sizes in strategy_pool_sizes.items()},
    'Matched positive draws per code': matched_budgets,
    'Full human positive count': human_budgets
}))
selection_note += ' Without replacement.'
print(selection_note, overlap_note)
if human_mode:
    print('Human bank: subsampling without replacement at the same per-code counts as the other methods.')
    print('Every selected row appears once within each per-code draw. Repetitions may select the same subset.')

# Precompute cosine similarities once; the repeated draws below only index into these.

selection_similarity = selection_embeddings @ selection_embeddings.T
validation_similarity = validation_embeddings @ selection_embeddings.T
selection_validation_overlap = pd.Series(selection_texts).isin(validation_texts).to_numpy()
corpus_similarity = {}
central_distance = {}
if not human_mode:
    corpus_similarity = {
        c: selection_similarity[np.ix_(rows, rows)] for c, rows in corpus_pools.items()
    }
    for c, rows in corpus_pools.items():
        center = selection_embeddings[rows].mean(axis=0)
        center /= max(np.linalg.norm(center), 1e-12)
        central_distance[c] = 1 - selection_embeddings[rows] @ center

# Human-bank curves cover the common central/diverse count range.
# The direct comparison uses matched_budgets for all four methods.
curve_limits = {s: sizes.copy() for s, sizes in strategy_pool_sizes.items()}
if human_mode:
    curve_limits['Human bank'] = matched_budgets.copy()
combined_counts_by_strategy = {}
focal_counts_by_strategy = {}
for s, sizes in curve_limits.items():
    combined_counts_by_strategy[s] = sorted(set(
        [n for n in exemplar_counts if 0 < n <= max(sizes.values())] + list(sizes.values())
    ))
    focal_counts_by_strategy[s] = {
        c: sorted(set([n for n in exemplar_counts if 0 < n <= sizes[c]]
                      + [matched_budgets[c], sizes[c]])) for c in coding_scheme
    }

# %% [markdown] id="5e7bd88a"
# ### **Central-Random-Diverse Exemplars**

# %% [markdown] id="89158692"
# #### **Choose the Exemplars**
#
# - **Random**: random selection (from the training data or the human C/D bank, set by `random_selection_source`).
#
# In **human** mode, central and diverse are the human-labelled C and D exemplars, in random order.
# In **machine** mode, they are chosen from the corpus training rows:
#
# - **Central**: `central_bands` equal groups sorted by distance from mean, then a random selection from the closest band, moving to the next when out of the first, etc.
# - **Diverse**: Randomly pick a starting exemplar, then pick the next randomly from the top 5% (`diverse_candidate_fraction`) most different cosine similarity.

# %% id="6fdb7e32" tags=["selection_orders"]
# Store one nested order per code, method, and repetition.
# Keeping the orders also lets the combined and individual experiments reuse the same draws.
# Taking the first n rows of an order gives the n-exemplar draw, so larger draws contain smaller ones.
# Each random generator is seeded with [random_seed, repetition, stream, code]. The stream numbers
# (6101 central, 6201 diverse, 6301 PEPPER negatives, 6401 human bank) only keep the draws independent;
# changing them changes which exemplars are drawn.

selection_orders = []
matched_negative_selections = []
start_time = time.time()

for repetition in range(n_repetitions):

    orders = {strategy: {} for strategy in selection_strategies}

    for j, (code_name, pool) in enumerate(corpus_pools.items()):

        if effective_selection_source == 'human':
            # Human categories have no within-category ranking: randomize only within the type.
            for strategy, seed_offset in [('Random', 0), ('Central', 6101), ('Diverse', 6201)]:
                seed = ([random_seed, repetition, j] if strategy == 'Random'
                        else [random_seed, repetition, seed_offset, j])
                rng = np.random.default_rng(seed)
                orders[strategy][code_name] = rng.permutation(selection_pools[strategy][code_name])
            # Subsample the original bank without replacement, independently of C/D types.
            rng = np.random.default_rng([random_seed, repetition, 6401, j])
            orders['Human bank'][code_name] = rng.choice(
                selection_pools['Human bank'][code_name], size=matched_budgets[code_name], replace=False
            )
            continue

        # Random: a shuffled list without replacement.

        rng = np.random.default_rng([random_seed, repetition, j])
        orders['Random'][code_name] = rng.permutation(selection_pools['Random'][code_name])

        # Machine central: rank by proximity, then shuffle inside central_bands contiguous bands.

        rng = np.random.default_rng([random_seed, repetition, 6101, j])
        ties = rng.permutation(len(pool))
        ranked = ties[np.argsort(central_distance[code_name][ties], kind='stable')]
        bands = np.array_split(ranked, central_bands)

        orders['Central'][code_name] = pool[
            np.concatenate([rng.permutation(band) for band in bands])
        ]

        # Diverse: update the distance to the nearest example already selected.

        rng = np.random.default_rng([random_seed, repetition, 6201, j])
        local_similarity = corpus_similarity[code_name]

        acquired = np.zeros(len(pool), dtype=bool)
        nearest = np.full(len(pool), -np.inf)
        diverse_order = []

        priority = np.empty(len(pool), dtype=int)
        priority[rng.permutation(len(pool))] = np.arange(len(pool))
        picked = int(rng.integers(len(pool)))

        for _ in range(len(pool)):

            diverse_order.append(picked)
            acquired[picked] = True
            nearest = np.maximum(nearest, local_similarity[:, picked])

            remaining = np.flatnonzero(~acquired)

            if not len(remaining):
                break

            distance = 1 - nearest[remaining]
            n_farthest = max(1, int(np.ceil(diverse_candidate_fraction * len(remaining))))
            cutoff = np.partition(distance, len(distance) - n_farthest)[len(distance) - n_farthest]

            # Random tie priority keeps the farthest-5% set the requested size.

            farther = remaining[distance > cutoff]
            tied = remaining[distance == cutoff]
            boundary = tied[np.argsort(priority[tied])][:n_farthest - len(farther)]

            picked = int(rng.choice(np.r_[farther, boundary], replace=False))

        orders['Diverse'][code_name] = pool[diverse_order]

    # For the PEPPER human-count comparison, hold the same random negatives
    # fixed across the three positive-selection methods within this repetition.

    negatives = {}

    if dataset == 'pepper':
        for j, code_name in enumerate(coding_scheme):
            rng = np.random.default_rng([random_seed, repetition, 6301, j])
            negatives[code_name] = rng.choice(
                negative_pools[code_name], size=human_negative_budgets[code_name],
                replace=False
            )

    matched_negative_selections.append(negatives)
    selection_orders.append(orders)

    if (repetition + 1) % max(1, n_repetitions // 10) == 0:
        print(f'Example orders: {repetition + 1}/{n_repetitions}.')

example_orders = selection_orders[0]

print(f'Example orders completed in {time.time() - start_time:.1f} seconds.')

# %% [markdown] id="771505c6"
# #### **Kappa Calculation**
#
# The **combined** curve grows every code together. Each code stops at its own available count.
#
# The **individual-code** curves grow only that code. The other codes keep the same background examples
# (human-bank subsample in human mode, random in machine mode) at their matched counts for that repetition.
#
# PEPPER cutoffs are recalibrated for every draw with the same out-of-fold method as *PEPPER Cutoffs* above.
# This cell is the slow one; progress prints every 10%.

# %% id="328f3036" tags=["selection_calculation"]
selection_rows = []
selection_calibration_rows = []
start_time = time.time()

for repetition, orders in enumerate(selection_orders):

    # These background examples stay fixed across all individual-code curves in this draw.

    background_strategy = 'Human bank' if human_mode else 'Random'
    background = {c: orders[background_strategy][c][:matched_budgets[c]] for c in coding_scheme}

    # Prefix scores avoid repeated large matrix sums for dense growth curves.
    # Each prefix contains distinct selected rows, including the human-bank prefixes.
    # A mean-centroid score is (v · sum of chosen vectors) / |sum of chosen vectors|, so running
    # (cumulative) sums give the score for every exemplar count in one pass.
    score_cache = {}
    if human_mode:
        for strategy in selection_strategies:
            score_cache[strategy] = {}
            for c in coding_scheme:
                order = orders[strategy][c][:max(combined_counts_by_strategy[strategy][-1],
                                               focal_counts_by_strategy[strategy][c][-1])]
                gram = selection_similarity[np.ix_(order, order)].astype(np.float64)
                increments = np.diag(gram) + 2 * np.triu(gram, 1).sum(axis=0)
                lengths = np.sqrt(np.maximum(np.cumsum(increments), 1e-12))
                score_cache[strategy][c] = np.cumsum(validation_similarity[:, order], axis=1,
                                                   dtype=np.float64) / lengths[None, :]
    matched_negatives = matched_negative_selections[repetition]

    for experiment in ['Combined'] + coding_scheme + ['Human counts']:

        if experiment == 'Combined':
            scope = 'combined'

        elif experiment == 'Human counts':
            scope = 'matched'
            counts = [0]  # A separate per-code count vector; not a scalar curve position.

        else:
            scope = 'focal'

        for strategy in selection_strategies:

            if scope == 'combined':
                counts = combined_counts_by_strategy[strategy]
            elif scope == 'focal':
                counts = focal_counts_by_strategy[strategy][experiment]

            for count in counts:

                # Choose the requested positive examples for this experiment.

                if scope == 'combined':
                    chosen = {c: orders[strategy][c][:min(count, curve_limits[strategy][c])]
                              for c in coding_scheme}

                elif scope == 'matched':
                    chosen = {c: orders[strategy][c][:matched_budgets[c]] for c in coding_scheme}

                else:
                    chosen = background.copy()
                    chosen[experiment] = orders[strategy][experiment][:count]

                # Sorting keeps floating-point summation consistent without changing the selections.

                chosen = {c: np.sort(rows) for c, rows in chosen.items()}
                used = np.unique(np.concatenate(list(chosen.values())))
                if dataset == 'pepper' and scope == 'matched':

                    # Positive indices are local to the positive-candidate array.
                    # Negative indices already refer to the original corpus rows.

                    used_corpus_rows = np.unique(np.r_[
                        candidate_rows[used], np.concatenate(list(matched_negatives.values()))
                    ])
                    sentence_count = df_data.loc[used_corpus_rows, sentence_column].nunique()
                    selected_row_count = len(used_corpus_rows)
                    known_decisions = sum(
                        len(chosen[c]) + len(matched_negatives[c]) for c in coding_scheme
                    )

                else:
                    sentence_count = len(pd.unique(selection_texts[used]))
                    selected_row_count = len(used)
                    known_decisions = int(np.isfinite(selection_labels[used]).sum())

                scores = np.empty((len(validation_indices), len(coding_scheme)))
                cutoffs = np.full(len(coding_scheme), np.nan)

                for j, code_name in enumerate(coding_scheme):

                    positives = chosen[code_name]
                    if human_mode:
                        score_strategy = (background_strategy if scope == 'focal' and code_name != experiment
                                          else strategy)
                        scores[:, j] = score_cache[score_strategy][code_name][:, len(positives) - 1]
                    else:
                        length = np.sqrt(max(float(
                            selection_similarity[np.ix_(positives, positives)].sum()
                        ), 1e-12))
                        scores[:, j] = validation_similarity[:, positives].sum(axis=1) / length

                    # A PEPPER individual-code curve needs only that code's binary cutoff.

                    if dataset != 'pepper' or (scope == 'focal' and code_name != experiment):
                        continue

                    if scope == 'matched':

                        # Use exactly the human positive AND negative counts for this code.
                        # Extra labels on the other selected sentences are not used here.

                        calibration_vectors = np.vstack([
                            selection_embeddings[positives], embeddings[matched_negatives[code_name]]
                        ])
                        calibration_similarity = calibration_vectors @ calibration_vectors.T
                        calibration_labels = np.r_[
                            np.ones(len(positives)), np.zeros(len(matched_negatives[code_name]))
                        ]
                        calibration_positives = np.arange(len(positives))
                        known = np.arange(len(calibration_labels))

                    else:

                        # Keep the original calibration for all main growth curves.

                        calibration_similarity = selection_similarity
                        calibration_labels = selection_labels[:, j]
                        calibration_positives = positives
                        known = used[np.isfinite(calibration_labels[used])]

                    cutoff = np.nan
                    rule = 'Unsupported: fewer than two fitted positives'

                    if len(calibration_positives) >= 2:

                        # The same out-of-fold mean-centroid calculation used in the main analysis.

                        n_folds = min(5, len(calibration_positives))
                        fold_rng = np.random.default_rng(random_seed + j)
                        folds = np.full(len(calibration_labels), -1)

                        shuffled = fold_rng.permutation(calibration_positives)
                        folds[shuffled] = np.arange(len(shuffled)) % n_folds

                        for value in [0, 1]:

                            other = np.setdiff1d(known[calibration_labels[known] == value], calibration_positives)
                            shuffled = fold_rng.permutation(other)
                            folds[shuffled] = np.arange(len(shuffled)) % n_folds

                        oof = np.full(len(calibration_labels), np.nan)

                        for fold in range(n_folds):

                            fit = calibration_positives[folds[calibration_positives] != fold]
                            held = known[folds[known] == fold]

                            length = np.sqrt(max(float(
                                calibration_similarity[np.ix_(fit, fit)].sum()
                            ), 1e-12))

                            oof[held] = calibration_similarity[np.ix_(held, fit)].sum(axis=1) / length

                        # Pick the cutoff using selected labels, never validation labels.

                        values = oof[known]
                        truth = calibration_labels[known]

                        if np.isfinite(values).all():

                            if (truth == 0).any():

                                candidates = np.unique(np.r_[
                                    0., np.quantile(values, np.linspace(0, 1, 15))
                                ])

                                predicted = values[:, None] >= candidates[None, :]
                                observed = (predicted == truth[:, None]).mean(axis=0)

                                true_fraction = truth.mean()
                                predicted_fraction = predicted.mean(axis=0)
                                chance = (true_fraction * predicted_fraction
                                          + (1 - true_fraction) * (1 - predicted_fraction))

                                quality = (observed - chance) / (1 - chance)
                                best = np.flatnonzero(np.isclose(quality, np.nanmax(quality)))

                                cutoff = candidates[best[np.argmin(np.abs(candidates[best]))]]
                                rule = 'Out-of-fold κ'

                            else:

                                cutoff = np.quantile(values, .1)
                                rule = 'Positive-only: 10th percentile'

                    cutoffs[j] = cutoff

                    selection_calibration_rows.append({
                        'repetition': repetition, 'scope': scope, 'focal_code': experiment,
                        'strategy': strategy, 'count': count, 'code': code_name,
                        'cutoff': cutoff, 'rule': rule,
                        'known_positive': int((calibration_labels[known] == 1).sum()),
                        'known_negative': int((calibration_labels[known] == 0).sum())
                    })

                # Predict and calculate the same Cohen's kappa as above.
                # Array arithmetic keeps the many repeated curves quick to calculate.
                # κ = (observed agreement - chance agreement) / (1 - chance agreement),
                # the same value as cohen_kappa_score, computed for all codes at once.

                predictions = predict_codes(scores, cutoffs)

                observed = (predictions == validation_labels).mean(axis=0)
                true_fraction = validation_labels.mean(axis=0)
                predicted_fraction = predictions.mean(axis=0)
                chance = (true_fraction * predicted_fraction
                          + (1 - true_fraction) * (1 - predicted_fraction))

                kappas = (observed - chance) / (1 - chance)

                if scope == 'focal':
                    metrics = {experiment: kappas[coding_scheme.index(experiment)]}

                else:

                    metrics = dict(zip(coding_scheme, kappas))
                    metrics['Macro'] = np.mean(kappas) if np.isfinite(kappas).all() else np.nan

                    if dataset == 'sources_v2':

                        observed = (predictions.argmax(axis=1) == validation_labels.argmax(axis=1)).mean()
                        chance = (true_fraction * predicted_fraction).sum()

                        metrics['Overall (Sources)'] = (observed - chance) / (1 - chance)

                for code_name, value in metrics.items():

                    selection_rows.append({
                        'selection_source': effective_selection_source,
                        'random_source': effective_random_source,
                        'positive_draws': sum(len(rows) for rows in chosen.values()),
                        'repetition': repetition, 'scope': scope, 'strategy': strategy,
                        'count': count, 'code': code_name, 'kappa': value,
                        'sentence_count': sentence_count, 'selected_row_count': selected_row_count,
                        'known_decisions': known_decisions,
                        'validation_text_matches': int(selection_validation_overlap[used].sum()),
                        **{f'n_{c}': len(chosen[c]) for c in coding_scheme}
                    })

    if (repetition + 1) % max(1, n_repetitions // 10) == 0:
        print(f'Kappa calculations: {repetition + 1}/{n_repetitions}.')

selection_results = pd.DataFrame(selection_rows)
selection_calibration = pd.DataFrame(selection_calibration_rows)

print(f'Kappa curves completed in {time.time() - start_time:.1f} seconds.')

# %% [markdown] id="8c26fa5c"
# #### **Combine Results**

# %% id="eab8e318" tags=["selection_summary"]
# Aggregate curve points and the separate full-human-count comparison.

selection_summary = selection_results.groupby(
    ['selection_source', 'random_source', 'scope', 'strategy', 'count', 'code'], as_index=False
).agg(
    mean_kappa=('kappa', 'mean'),
    lower=('kappa', lambda x: x.quantile(.025)),
    upper=('kappa', lambda x: x.quantile(.975)),
    supported_draws=('kappa', 'count'),
    mean_sentences=('sentence_count', 'mean'),
    mean_selected_rows=('selected_row_count', 'mean'),
    mean_known_decisions=('known_decisions', 'mean'),
    mean_positive_draws=('positive_draws', 'mean'),
    mean_validation_text_matches=('validation_text_matches', 'mean')
)

selection_summary['fraction_of_corpus'] = selection_summary['mean_sentences'] / len(df_data)

matched_summary = selection_summary.loc[selection_summary['scope'] == 'matched'].copy()

# Actual per-code counts are retained for every draw, including capped combined points.

count_columns = [f'n_{c}' for c in coding_scheme]

display(selection_results.loc[
    (selection_results['repetition'] == 0) & (selection_results['code'] == 'Macro'),
    ['scope', 'strategy', 'count', *count_columns, 'sentence_count', 'selected_row_count']
])

# %% [markdown] id="ad281392"
# ### **Graphs**
#
# Curves show mean validation κ against the number of exemplars (shading: 2.5th–97.5th percentiles across draws).
# The projection graph shows one matched-count draw of each strategy on the corpus PCA.

# %% id="8f8863b6" tags=["selection_figures"]
comparison_reference = full_human.iloc[0:0] if human_mode else full_human

plot_selection_curves(
    selection_summary,
    comparison_reference,
    coding_scheme,
    zoom_max=selection_zoom_max, strategy_labels=strategy_labels,
    selection_note=selection_note, reference_label=human_reference_label
);

# %% id="9e24f95e" tags=["selection_projection"]
# Show the central/random/diverse examples at the same counts as the direct comparison.
selection_projection = pca.transform(selection_embeddings)
fig, axes = plt.subplots(int(np.ceil(len(coding_scheme) / 2)), 2,
                         figsize=(12, 4 * int(np.ceil(len(coding_scheme) / 2))))
axes = np.asarray(axes).ravel()
for ax, c in zip(axes, coding_scheme):
    pool = np.unique(np.concatenate([selection_pools[s][c] for s in selection_strategies]))
    ax.scatter(selection_projection[pool, 0], selection_projection[pool, 1],
               color=GRID_COLOR, s=14, alpha=.5)
    for s in selection_strategies:
        rows = example_orders[s][c][:matched_budgets[c]]
        ax.scatter(selection_projection[rows, 0], selection_projection[rows, 1],
                   color=STRATEGY_COLORS[s], marker=STRATEGY_MARKERS[s], s=45,
                   alpha=.75, edgecolors='white', linewidths=.5, label=strategy_labels[s])
    ax.set(title=f'{c}: {matched_budgets[c]} positives per strategy', xlabel='PC1', ylabel='PC2')
for ax in axes[len(coding_scheme):]:
    ax.set_visible(False)
handles, names = axes[0].get_legend_handles_labels()
fig.suptitle('Central / random / diverse exemplars: one matched-count draw', y=.99, fontsize=12)
fig.legend(handles, names, loc='upper center', bbox_to_anchor=(.5, .955), ncol=2, fontsize=9)
fig.text(.5, .02, selection_note + '\nPCA fitted on the corpus; pale points show the candidate pools. Each method selects distinct rows; methods may select the same sentence.',
         ha='center', fontsize=9, color=NOTE_COLOR)
fig.subplots_adjust(top=.84, bottom=.12, hspace=.32, wspace=.2)
plt.show()

# %% [markdown] id="c0afd6d4"
# ### **Compare Against Human Exemplars**
#
# Each Central-Random-Diverse method gets the same number of sentences as the human exemplars.

# %% id="f73b7743" tags=["matched_comparison"]
# Sources matches positive counts. PEPPER also matches the recorded negative
# counts used for cutoff calibration, with common random negatives in each draw.

plot_matched_comparison(
    matched_summary, comparison_reference, coding_scheme, matched_budgets, human_negative_budgets,
    strategy_labels=strategy_labels, selection_note=selection_note,
    reference_label=human_reference_label
);

display(matched_summary.loc[
    matched_summary['code'] == 'Macro',
    ['strategy', 'mean_kappa', 'lower', 'upper', 'mean_selected_rows',
     'mean_sentences', 'fraction_of_corpus', 'mean_known_decisions',
     'mean_validation_text_matches']
].round(3))

display(pd.Series(matched_budgets, name='Matched positive count per strategy').to_frame())
print(selection_note)

if human_mode:
    print('Human-bank comparison is resampled within the original bank at the matched per-code counts.')
    print('Draws contain distinct pool rows; mean_sentences reports distinct sentence texts.')
else:
    print(f'Full human bank: {len(df_human)} unique sentences; '
          f'{int(df_human[coding_scheme].notna().sum().sum())} recorded label decisions.')

if not selection_calibration.empty:

    display(selection_calibration.loc[
        selection_calibration['scope'] == 'matched'
    ].groupby(['strategy', 'code', 'rule'], as_index=False).agg(
        draws=('cutoff', 'size'),
        mean_cutoff=('cutoff', 'mean'),
        mean_known_positive=('known_positive', 'mean'),
        mean_known_negative=('known_negative', 'mean')
    ))

# %% [markdown] id="SBy-8M2J0ai_"
# #### **Cosine Similarity of Centroid Variations**

# %% id="Bn1TZKlA0nRV" tags=["selection_similarities"]
# How similar is each strategy's centroid to the human centroid, and how spread out is each draw?
# vector_summary() (Helper Functions) gives a draw's centroid and its mean within-group similarity.

if human_mode:
    # All metrics reuse the same matched samples without replacement used for kappa.

    similarity_rows = []
    for repetition, orders in enumerate(selection_orders):
        for c in coding_scheme:
            bank_draw = orders['Human bank'][c][:matched_budgets[c]]
            reference_center, _ = vector_summary(selection_embeddings[bank_draw])
            for strategy in selection_strategies:
                chosen = orders[strategy][c][:matched_budgets[c]]
                center, within = vector_summary(selection_embeddings[chosen])
                similarity_rows.append({
                    'selection_source': effective_selection_source, 'random_source': effective_random_source,
                    'Repetition': repetition, 'Strategy': strategy, 'Code': c,
                    'Selected positives': len(chosen),
                    'Unique sentence texts': len(pd.unique(selection_texts[chosen])),
                    'Cosine Similarity to Human Centroid': float(center @ reference_center),
                    'Within-Group Similarity': within
                })
    matched_kappa = selection_results.loc[selection_results['scope'].eq('matched'),
        ['repetition', 'strategy', 'code', 'kappa']].rename(columns={
            'repetition': 'Repetition', 'strategy': 'Strategy', 'code': 'Code', 'kappa': 'Validation Kappa'
        })
    df_scatter = pd.DataFrame(similarity_rows).merge(
        matched_kappa, on=['Repetition', 'Strategy', 'Code'], validate='one_to_one')
    assert len(df_scatter) == len(selection_orders) * len(selection_strategies) * len(coding_scheme)
    df_similarities = df_scatter.groupby(['Code', 'Strategy'], as_index=False).agg(**{
        'Similarity to Human Centroid (Mean)': ('Cosine Similarity to Human Centroid', 'mean'),
        'Similarity lower': ('Cosine Similarity to Human Centroid', lambda x: x.quantile(.025)),
        'Similarity upper': ('Cosine Similarity to Human Centroid', lambda x: x.quantile(.975)),
        'Within-Group Similarity (Mean)': ('Within-Group Similarity', 'mean'),
        'Within lower': ('Within-Group Similarity', lambda x: x.quantile(.025)),
        'Within upper': ('Within-Group Similarity', lambda x: x.quantile(.975)),
        'Mean unique sentences': ('Unique sentence texts', 'mean')
    })
    print('Human-centroid alignment uses that repetition\'s matched human-bank subsample as the reference.')
    print('Human-bank draws come only from centroids_expanded; all four methods use the same per-code counts.')
    display(df_similarities.sort_values(['Code', 'Strategy']).round(4))

else:
    # Reuse the matched kappa draws: the same source, type pools, indices, and counts.
    # The reference is the full human bank's centroid for each code.

    repetition_scatter_data = []
    human_similarity_rows = []
    for code_name in coding_scheme:
        human_center, human_within = vector_summary(training_vectors[selected_by_code[code_name]])
        human_similarity_rows.append({
            'Code': code_name, 'Strategy': 'Human (Actual)',
            'Similarity to Human Centroid (Mean)': 1.,
            'Within-Group Similarity (Mean)': human_within
        })
        for repetition, orders in enumerate(selection_orders):
            for strategy in selection_strategies:
                chosen = orders[strategy][code_name][:matched_budgets[code_name]]
                center, within = vector_summary(selection_embeddings[chosen])
                repetition_scatter_data.append({
                    'selection_source': effective_selection_source,
                    'Repetition': repetition, 'Strategy': strategy, 'Code': code_name,
                    'Selected positives': len(chosen),
                    'Cosine Similarity to Human Centroid': float(center @ human_center),
                    'Within-Group Similarity': within
                })

    matched_kappa = selection_results.loc[selection_results['scope'].eq('matched'),
        ['repetition', 'strategy', 'code', 'kappa']].rename(columns={
            'repetition': 'Repetition', 'strategy': 'Strategy', 'code': 'Code', 'kappa': 'Validation Kappa'
        })
    df_scatter = pd.DataFrame(repetition_scatter_data).merge(
        matched_kappa, on=['Repetition', 'Strategy', 'Code'], validate='one_to_one'
    )
    assert len(df_scatter) == len(selection_orders) * len(selection_strategies) * len(coding_scheme)
    df_similarities = df_scatter.groupby(['Code', 'Strategy'], as_index=False).agg(**{
        'Similarity to Human Centroid (Mean)': ('Cosine Similarity to Human Centroid', 'mean'),
        'Within-Group Similarity (Mean)': ('Within-Group Similarity', 'mean')
    })
    df_similarities = pd.concat([df_similarities, pd.DataFrame(human_similarity_rows)], ignore_index=True)
    print(selection_note)
    print('Similarity and scatter graphs use the equal per-code counts shown in the kappa comparison.')
    display(df_similarities.sort_values(['Code', 'Strategy']).round(4))

# %% [markdown] id="e3c4-zJ1kkCf"
# #### **Similarity Comparisons**

# %% id="HeseGHwKkq8z" tags=["similarity_figures"]
# Bar charts of the similarity table above: alignment to the human centroid (left)
# and how tightly each strategy's exemplars cluster (right).
# Human mode compares with each draw's matched human-bank subsample (whiskers: 2.5th-97.5th percentiles).
# Machine mode compares with the full human bank, which also appears as its own bar on the right.

codes = list(coding_scheme)
x = np.arange(len(codes))

if human_mode:
    panels = [
        (['Random', 'Central', 'Diverse'], 'Similarity to Human Centroid (Mean)',
         ('Similarity lower', 'Similarity upper'), 'Alignment to matched human-bank centroid'),
        (selection_strategies, 'Within-Group Similarity (Mean)',
         ('Within lower', 'Within upper'), 'Within-group pairwise cosine similarity')
    ]
    legend_strategies = selection_strategies
    caption = 'Whiskers: 2.5th–97.5th percentiles across matched draws. All methods use no replacement.'

else:
    panels = [
        (['Random', 'Central', 'Diverse'], 'Similarity to Human Centroid (Mean)',
         None, 'Alignment to full human-bank centroid'),
        (['Random', 'Central', 'Diverse', 'Human (Actual)'], 'Within-Group Similarity (Mean)',
         None, 'Within-group pairwise cosine similarity')
    ]
    legend_strategies = ['Random', 'Central', 'Diverse', 'Human (Actual)']
    caption = 'Bars: means across matched draws. All methods use no replacement.'

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

for ax, (strategies, mean, interval, title) in zip(axes, panels):
    width = .72 / len(strategies)

    for j, strategy in enumerate(strategies):
        rows = df_similarities.loc[df_similarities.Strategy.eq(strategy)].set_index('Code').reindex(codes)
        positions = x + (j - (len(strategies) - 1) / 2) * width
        ax.bar(positions, rows[mean], width, color=STRATEGY_COLORS[strategy], edgecolor='white', linewidth=.5)

        if interval is not None:
            ax.vlines(positions, rows[interval[0]], rows[interval[1]], color='#111111', linewidth=1)

    ax.set(title=title, ylabel='Cosine similarity', xticks=x, xticklabels=codes)
    ax.grid(axis='y', color=GRID_COLOR, alpha=.5, linewidth=.6)

# 'Human (Actual)' (machine mode) is the full human bank, labelled like the reference elsewhere.

handles = [plt.Rectangle((0, 0), 1, 1, color=STRATEGY_COLORS[s]) for s in legend_strategies]
fig.suptitle(selection_note, fontsize=10)
fig.legend(handles, [strategy_labels.get(s, human_reference_label) for s in legend_strategies],
           loc='lower center', bbox_to_anchor=(.5, .03), ncol=2, fontsize=9)
fig.text(.5, .015, caption, ha='center', fontsize=8, color=NOTE_COLOR)
fig.tight_layout(rect=[0, .18, 1, .90])
plt.show()

# %% id="oX1fwDgyp8Yy" tags=["similarity_scatter"]
# One point per matched draw: does a draw that is closer to the human centroid (or more tightly
# clustered) also get a higher validation κ? Reuses the exact matched-count selections and κ values above.
# In machine mode, the star marks the full human bank (its within-group similarity and κ).

if human_mode:
    scatter_plots = [
        ('Cosine Similarity to Human Centroid', 'Alignment to human-bank subsample vs. performance',
         'Cosine similarity to matched human-bank centroid', ['Random', 'Central', 'Diverse']),
        ('Within-Group Similarity', 'Within-group similarity vs. performance',
         'Within-group pairwise cosine similarity', selection_strategies)
    ]

else:
    scatter_plots = [
        ('Cosine Similarity to Human Centroid', 'Alignment to human centroid vs. performance',
         'Cosine similarity to full human-bank centroid', ['Random', 'Central', 'Diverse']),
        ('Within-Group Similarity', 'Within-group similarity vs. performance',
         'Within-group pairwise cosine similarity', ['Random', 'Central', 'Diverse'])
    ]

for metric, title, xlabel, strategies in scatter_plots:
    show_human_star = not human_mode and metric == 'Within-Group Similarity'

    nrows = int(np.ceil(len(coding_scheme) / 2))
    fig, axes = plt.subplots(nrows, 2, figsize=(10, 4 * nrows), sharey=True)
    axes = np.asarray(axes).ravel()

    for ax, c in zip(axes, coding_scheme):
        for strategy in strategies:
            rows = df_scatter.loc[df_scatter.Code.eq(c) & df_scatter.Strategy.eq(strategy)]
            ax.scatter(rows[metric], rows['Validation Kappa'], color=STRATEGY_COLORS[strategy],
                       alpha=.3, s=15, edgecolors='none')

        if show_human_star:
            human_similarity = df_similarities.loc[
                df_similarities.Code.eq(c) & df_similarities.Strategy.eq('Human (Actual)'),
                'Within-Group Similarity (Mean)'
            ].iloc[0]
            human_kappa = full_human.loc[full_human['code'].eq(c), 'kappa'].iloc[0]
            ax.scatter(human_similarity, human_kappa, color=STRATEGY_COLORS['Human (Actual)'],
                       marker='*', s=150, edgecolors='black', linewidths=1.2, zorder=5)

        ax.set(title=c, xlabel=xlabel, ylabel='Validation κ')
        ax.grid(True, linestyle=':', alpha=.35)
        ax.tick_params(labelsize=8)
        ax.xaxis.label.set_size(8)

    for ax in axes[len(coding_scheme):]:
        ax.set_visible(False)

    handles = [plt.Line2D([], [], marker='o', linestyle='', color=STRATEGY_COLORS[s], markersize=5)
               for s in strategies]
    names = [strategy_labels[s] for s in strategies]

    if show_human_star:
        handles.append(plt.Line2D([], [], marker='*', linestyle='', markersize=11,
                                  color=STRATEGY_COLORS['Human (Actual)'], markeredgecolor='black'))
        names.append(human_reference_label)

    fig.suptitle(title, fontsize=12, y=.99)
    fig.text(.5, .945, selection_note, ha='center', fontsize=9)
    fig.legend(handles, names, loc='lower center', ncol=2 if len(handles) == 4 else 3, fontsize=9)
    fig.tight_layout(rect=[0, .08, 1, .91])
    plt.show()

# %% [markdown] id="774651cd"
# ### **Within Tolerance Best Kappa**
#
# Find the earliest kappa value within `kappa_tolerance` of the best observed mean.

# %% id="b5aa1ca3" tags=["near_best"]
print(f'Near-best tolerance: {kappa_tolerance:.2f} κ.')

near_best_rows = []

curves = selection_summary.loc[selection_summary['scope'] != 'matched']

for (scope, strategy, code_name), group in curves.groupby(['scope', 'strategy', 'code']):

    # Individual-code results come from the focal experiment, not the combined curve.

    if scope == 'combined' and code_name in coding_scheme:
        continue

    group = group.loc[group['mean_kappa'].notna()].sort_values('count')

    if group.empty:
        continue

    best_kappa = group['mean_kappa'].max()
    first = group.loc[group['mean_kappa'] >= best_kappa - kappa_tolerance].iloc[0]

    near_best_rows.append({
        'selection_source': effective_selection_source,
        'scope': scope, 'strategy': strategy_labels[strategy], 'code': code_name,
        'best_observed_kappa': best_kappa, 'near_best_requested_count': int(first['count']),
        'kappa_at_count': first['mean_kappa'], 'mean_unique_sentences': first['mean_sentences'],
        'fraction_of_corpus': first['fraction_of_corpus']
    })

near_best = pd.DataFrame(near_best_rows)

display(near_best.round(3))

# %% [markdown] id="649a6f52"
# ## **Fine-Tuning** [Future Work]
#
# Fine-tuning on the model using the selected central, random, or diverse exemplars.
#
# *No code yet: this section is a placeholder.*
