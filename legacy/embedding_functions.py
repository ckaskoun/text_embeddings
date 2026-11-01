"""
Functions used in the analysis code.
"""

# Hide Kmeans memory leak warnings
import warnings
warnings.filterwarnings("ignore",
                        category=UserWarning,
                        module=r"sklearn\.cluster\._kmeans")

# Imports
import numpy as np
import pandas as pd
import string
import seaborn as sns
import matplotlib.patches as mpatches
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

# Main function for calcuating embeddings scores from already coded data (manual or random).
# Creates subcentroids for more accurate scoring.
def embedding_score(
    embedding_matrix: np.ndarray,
    alpha: float,
    centroid_indices: list[list[int]],
    metric: str,
    scaling: str,
    rounding: bool = False,
    assignment_matrix: np.ndarray | None = None,
    k_per_cat: list[int | None] | None = None,  # per-category subcentroid counts (None ⇒ use all reps)
    random_state: int = 42,
) -> np.ndarray:
    """
    Score each row of `embedding_matrix` (or `assignment_matrix` if provided) against
    per-category subcentroids computed with KMeans (cosine-aware when metric='cosine').

    - metric: 'cosine' | 'euclidean' | 'cityblock'
    - scaling:
        * 'exponential'  => w = exp(-alpha * d)
        * 'power'        => w = d ** (-alpha), with safe floor on d
    - k_per_cat: list of ints/None, same length as centroid_indices.
                 If None: keep all reps (dedup, order-preserving).
                 If 1: mean center. If >1: exact KMeans k.
    - Returns: (n_samples, n_categories) row-normalized scores.
    """
    assert scaling in ("exponential", "power")
    assert metric in ("cosine", "euclidean", "cityblock")

    def _l2norm_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        n = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.maximum(n, eps)

    def _unique_rows_preserve_order(a: np.ndarray) -> np.ndarray:
        if a.size == 0:
            return a
        a_c = np.ascontiguousarray(a)
        view = a_c.view([('', a_c.dtype)] * a_c.shape[1])
        _, idx = np.unique(view, return_index=True)
        return a[idx]

    def _subcentroids_kmeans(reps: np.ndarray, k: int | None) -> np.ndarray:
        if reps.size == 0:
            return reps
        if k is None:
            return _unique_rows_preserve_order(reps)
        n, _ = reps.shape
        if n < k:
            return _unique_rows_preserve_order(reps)
        if k == 1:
            return reps.mean(axis=0, keepdims=True).astype(reps.dtype, copy=False)
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10, algorithm="lloyd")
        km.fit(reps)
        return km.cluster_centers_.astype(reps.dtype, copy=False)

    # Choose which matrix to score; keep a separate base matrix for subcentroiding
    base_matrix = np.asarray(embedding_matrix, dtype=np.float64)
    score_matrix = base_matrix if assignment_matrix is None else np.asarray(assignment_matrix, dtype=np.float64)

    # Cosine geometry ⇒ normalize both; for Euclidean/Cityblock leave unnormalized
    if metric == "cosine":
        base_matrix  = _l2norm_rows(base_matrix)
        score_matrix = _l2norm_rows(score_matrix)

    n_samples = score_matrix.shape[0]
    n_cats = len(centroid_indices)
    if k_per_cat is None:
        k_per_cat = [None] * n_cats

    score = np.zeros((n_samples, n_cats), dtype=np.float64)

    for i, idxs in enumerate(centroid_indices):
        reps = base_matrix[idxs]
        centers = _subcentroids_kmeans(reps, k=k_per_cat[i])

        if centers.size == 0:
            continue

        distances = cdist(score_matrix, centers, metric=metric)

        if scaling == "exponential":
            weights = np.exp(-alpha * distances)
        else:  # 'power'
            distances = np.maximum(distances, 1e-12)
            weights = distances ** (-alpha)

        score[:, i] = weights.mean(axis=1)

    # Row-normalize safely
    row_sums = np.maximum(score.sum(axis=1, keepdims=True), 1e-12)
    score = score / row_sums

    if rounding:
        score = np.where(score <= 0.01, 0.0, score)
        row_sums = np.maximum(score.sum(axis=1, keepdims=True), 1e-12)
        score = score / row_sums

    return score

# functions for calculating the KL divergence and JS divergence
def kl_divergence(p, q):
    p = np.where(p == 0, 1e-10, p)
    q = np.where(q == 0, 1e-10, q)
    return np.sum(p * np.log2(p / q), axis = 1)

def js_divergence(p, q):
    m = 0.5*(p+q)
    return 0.5*(kl_divergence(p, m) + kl_divergence(q, m))

# Function for creating a heatmap of the embedding scores, Its heavily modified
# to plot our exact data.
def plot_heatmap(fig, axs, scores_list, title_list, legend_patches, cmap):
    """This function is for plotting heatmaps for 5 articles and their corresponding
    scores. One can either pass in just embedding scores, or both LDA and embedding scores.
    The function will then plot the heatmaps for the scores, and also calculate the JS divergence
    between the two scores if both are passed in.

    Args:
        fig (matplotlib fig): The figure to plot the heatmaps on
        axs (matplotlib axs): The axs to plot the heatmaps on, len 5
        scores_list (list): A list of the scores to plot
        title_list (list): Title for each of the subplots
        legend_patches (matplotlib legend): Legend to plot
        cmap (matplotlib cmap): colors for heatmap

    Returns:
        _type_: _description_
    """
    assert len(scores_list) == len(title_list) == len(axs)

    # For naming the subplots
    alphabet = list(string.ascii_lowercase)

    # Creating text on left side of the heatmap
    y_tl = False
    if len(scores_list[0]) == 2:
        y_tl = ["Coded",'EMB']

    for scores, title, ax, letter in zip(scores_list, title_list, axs, alphabet):
        # Plot the heatmap
        gx = sns.heatmap(scores, cmap=cmap, annot=True, yticklabels=y_tl,
                         xticklabels=range(1,len(scores[0])+1), vmin=0, vmax=1,
                         ax=ax, cbar=False, fmt='.2f')

        # Formatting the ticks
        gx.set_yticklabels(gx.get_yticklabels(), fontsize=9)

        # Add the JS divergence to the right of each heatmap if both coded and embedding scores are passed in
        if len(scores) == 2:
            JS_div = js_divergence(scores[0].reshape(1, -1), scores[1].reshape(1, -1))
            ax.text(8.1, 1.5, f'JS: {JS_div[0]:.2f}',
                    fontsize=9, rotation=-90, fontweight='bold')

        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.text(-0.3, -0.2, f"({letter})", fontsize=11, fontweight='bold')

    # Create a colorbar
    # cbar_ax = fig.add_axes([1.1, 0.05, 0.075, 0.64])
    # cbar = fig.colorbar(gx.collections[0], cax=cbar_ax)
    # cbar.outline.set_visible(False)
    # cbar.ax.tick_params(labelsize=8)
    # cbar.set_label('', rotation=270, fontsize=8, labelpad=10)

    # Create fig legend
    fig.subplots_adjust(hspace=.7,bottom=0.1)
    fig.legend(handles=legend_patches,
               loc='upper left',
               bbox_to_anchor=(0.05, 0.075),
               frameon=False,
               prop={'size':10,'weight':'bold'},
               ncol=3)

    return fig, axs
