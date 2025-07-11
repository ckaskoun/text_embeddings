# Basic imports
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import scipy as sc
import math

# Special imports for plots
import string
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib.ticker import PercentFormatter
import matplotlib.colors as mcolors

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

    # new functions for calculating the KL divergence and JS divergence
    def kl_divergence(p, q):
        p = np.where(p == 0, 1e-10, p)
        q = np.where(q == 0, 1e-10, q)
        return np.sum(p * np.log2(p / q), axis = 1)


    def js_divergence(p, q):
        m = 0.5*(p+q)
        return 0.5*(kl_divergence(p, m) + kl_divergence(q, m))
