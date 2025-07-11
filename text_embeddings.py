# Basic imports
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import scipy as sc
import math
import create_embeddings as ct
import format_data as dt
from embedding_functions import kl_divergence, js_divergence

# Special imports for plots
import string
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib.ticker import PercentFormatter
import matplotlib.colors as mcolors

dict_embeddings = ct.embed_files("C:/Users/carlo/OneDrive/Documents/coded_memos") # Local files for testing and privacy
df_embeddings = dt.combine_embedded(dict_embeddings)

dict_codes = dt.format_coded("C:/Users/carlo/OneDrive/Documents/coded_memos") # Local files for testing and privacy
df_codes = dt.combine_coded(dict_codes)
df_codes_weighted = dt.weight_coded(df_codes, 'category')

# Create a matrix with all the embeddings
embedding_matrix = np.vstack(df_embeddings["Embeddings"].values)

# Create a list of the categories
coded_category_names = list(df_codes_weighted.columns[4:])

# Create a matrix of the hand-coded weights
# Shape will be (n_sentences, n_categories)
coded_scores = np.vstack(df_codes_weighted[coded_category_names].values).astype(float)

# Number of articles to be included in calculating the centroid
sum_categories = dt.sum_codes(df_codes_weighted)
min_code = float('inf')
for k in sum_categories:
    if sum_categories[k] < min_code:
        min_code = sum_categories[k]

n_sentences_for_centroid = min_code

# Sort indices along categories after human scores
sorted_indices = np.argsort(coded_scores, axis=0)

# Reverse the slice to have max values at the top
# Transpose to get the same shape (n_categories, n_sentences_for_centroid)
centroid_indices = sorted_indices[-n_sentences_for_centroid:][::-1].T

# We also matches this with a color list, such that we get the same colors for the same categories
# through the first part of the analysis
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"] # Temporarily a fixed number to match 'category' level
category_NameColor_dict = dict(zip(coded_category_names, colors))

# The uniform benchmark - Calculated by generating a matrix of 0.1
zero_p_one_matrix = np.zeros((180,4)) + 0.1
zero_p_one_score_unrounded = np.sum(js_divergence(zero_p_one_matrix, coded_scores))

print("Sum of JS divergence of uniform scores (aka 0.1 for each topic) against coded scores:")
print(zero_p_one_score_unrounded)
