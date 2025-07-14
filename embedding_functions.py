"""
Functions used in the analysis code <text_embeddings.py>.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sc
import math
import string
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib.ticker import PercentFormatter
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

def embedding_score(embedding_matrix, alpha, centroid_indices, metric, scaling, rounding=False):
    """Calculates the scores for a given embedding matrix (n, d)
        and a list of indices for each centroid (k), listed together in
        centroid_indices.

    Args:
        embedding_matrix np.array: The embedding matrix, shape (n, d)
        alpha (float): The scaling parameter
        centroid_indices (list): A list of lists of indices of the centroids
        metric (str): The metric used for calculating the distance
        scaling (str): The scaling used for scaling the distances
        rounding (bool, optional): If True scores are rounded. Defaults to False.

    Returns:
        np.array: The embedding scores, shape (n, k)
    """

    # Check that the input is valid
    assert scaling in ["exponential", "power"]
    assert metric in ["cosine", "euclidean", "cityblock"]

    # Normalizing the embedding matrix
    embedding_matrix /= np.linalg.norm(embedding_matrix, axis=1)[:,None]

    # Calculating centroid vectors
    centroid = np.zeros((len(centroid_indices), len(embedding_matrix[0])))
    for i in range(len(centroid_indices)):
        embeddings_temp = np.array([embedding_matrix[index] for index in centroid_indices[i]])
        centroid[i] = np.mean(embeddings_temp, axis=0)

    # Calculating distances using scipy.spatial.distance.cdist function
    distances = sc.spatial.distance.cdist(embedding_matrix, centroid, metric=metric)

    # Scale and invert the distances according to the specified scaling
    if scaling == "exponential":
        score = 1/np.exp(alpha*distances)
    if scaling == "power":
        score = 1/(distances**alpha)

    # L1 norm the scores to get a probability distribution
    score /= score.sum(axis=1)[:,None]

    # Round the scores to 2 decimals if specified
    if rounding:
        # Set all values below 0.01 to 0
        score[score <=0.01] = 0
        score /= score.sum(axis=1)[:,None]

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
            ax.text(3.1, 1.5, f'JS: {JS_div[0]:.2f}',
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
               ncol=1)

    return fig, axs
