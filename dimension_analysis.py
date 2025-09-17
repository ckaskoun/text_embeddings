"""
Code to analyze the importance of maintaining high-dimensional vectors.
"""

# Imports (held over from analyze_embeddings.py)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sc
import math
import create_embeddings as ct
import format_data as dt
from embedding_functions import kl_divergence, js_divergence, embedding_score, plot_heatmap
import string
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib.ticker import PercentFormatter, FixedLocator, FuncFormatter
import matplotlib.colors as mcolors
from scipy.stats import kendalltau, pearsonr, spearmanr
import pickle as pkl

# Create a dataframe of embeddings for comparison to human-coded scores

# If embeddings already saved in pickle
with open("coded_embeddings.pkl", "rb") as f:
    dict_embeddings2048 = pkl.load(f)
df_embeddings2048 = dt.combine_embedded(dict_embeddings2048)

# Continue with all other dimensions
with open("coded_embeddings1024.pkl", "rb") as f:
    dict_embeddings1024 = pkl.load(f)
df_embeddings1024 = dt.combine_embedded(dict_embeddings1024)

with open("coded_embeddings512.pkl", "rb") as f:
    dict_embeddings512 = pkl.load(f)
df_embeddings512 = dt.combine_embedded(dict_embeddings512)

with open("coded_embeddings256.pkl", "rb") as f:
    dict_embeddings256 = pkl.load(f)
df_embeddings256 = dt.combine_embedded(dict_embeddings256)

with open("coded_embeddings128.pkl", "rb") as f:
    dict_embeddings128 = pkl.load(f)
df_embeddings128 = dt.combine_embedded(dict_embeddings128)

# dict_embeddings2048 = ct.embed_files("") # Path to coded files goes here
# with open("coded_embeddings.pkl", "wb") as f:
#     pkl.dump(dict_embeddings2048, f)
# df_embeddings2048 = dt.combine_embedded(dict_embeddings2048)
#
# # Different dataframes for each dimension
# dict_embeddings1024 = ct.embed_files("", red_dim=1024) # Path to coded files goes here
# with open("coded_embeddings1024.pkl", "wb") as f:
#     pkl.dump(dict_embeddings1024, f)
# df_embeddings1024 = dt.combine_embedded(dict_embeddings1024)
#
# dict_embeddings512 = ct.embed_files("", red_dim=512) # Path to coded files goes here
# with open("coded_embeddings512.pkl", "wb") as f:
#     pkl.dump(dict_embeddings512, f)
# df_embeddings512 = dt.combine_embedded(dict_embeddings512)
#
# dict_embeddings256 = ct.embed_files("", red_dim=256) # Path to coded files goes here
# with open("coded_embeddings256.pkl", "wb") as f:
#     pkl.dump(dict_embeddings256, f)
# df_embeddings256 = dt.combine_embedded(dict_embeddings256)
#
# dict_embeddings128 = ct.embed_files("", red_dim=128) # Path to coded files goes here
# with open("coded_embeddings128.pkl", "wb") as f:
#     pkl.dump(dict_embeddings128, f)
# df_embeddings128 = dt.combine_embedded(dict_embeddings128)

# Create a dataframe of the human-coded scores
dict_codes = dt.format_coded("") # Path to coded files goes here
df_codes = dt.combine_coded(dict_codes)
df_codes_weighted = dt.weight_coded(df_codes, 'topic') # 'category' for highest level of coding

# Create a matrix with all the embeddings for each dimension
embedding_matrix2048 = np.vstack(df_embeddings2048["Embeddings"].values)
embedding_matrix1024 = np.vstack(df_embeddings1024["Embeddings"].values)
embedding_matrix512 = np.vstack(df_embeddings512["Embeddings"].values)
embedding_matrix256 = np.vstack(df_embeddings256["Embeddings"].values)
embedding_matrix128 = np.vstack(df_embeddings128["Embeddings"].values)

embedding_matrices = [embedding_matrix2048, embedding_matrix1024, embedding_matrix512,
                      embedding_matrix256, embedding_matrix128]

# Create a list of the categories (category level)
# coded_category_names = list(df_codes_weighted.columns[4:5]) + list(df_codes_weighted.columns[6:]) # Remove redundant Indirect Authenticity category

# Option for 'topic' level of coding
coded_category_names = ( list(df_codes_weighted.columns[4:7]) + list(df_codes_weighted.columns[11:12])
                 + list(df_codes_weighted.columns[13:]) ) # Remove redundant categories

# Create a matrix of the hand-coded weights
# Shape will be (n_sentences, n_categories)
coded_scores = np.vstack(df_codes_weighted[coded_category_names].values)
coded_scores = coded_scores.astype(float)

# Option of 'category' level of coding
# centroid_idxs = {
#     'Direct Authenticity': [287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 402, 403, 463, 464, 465, 466, 467, 468, 469, 470, 478, 479],
#     'Research Components': [310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 397, 398, 399, 400, 401, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 471, 472, 473, 474, 475, 476, 477],
#     'No Code': [0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 23, 24, 25, 26, 27, 28, 30, 32, 33, 35, 36, 37, 38, 39, 42, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 75, 80, 81, 82, 83, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 110, 111, 112, 113, 114, 117, 118, 119, 124, 125, 126, 127, 128, 129, 132, 133, 134, 135, 136, 137, 138, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 184, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 198, 199, 201, 202, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 221, 224, 225, 226, 227, 228, 229, 230, 231, 232, 234, 235, 236, 237, 238, 239, 240, 241, 246, 247, 248, 251, 254, 255, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 272, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462]
# }

# Option for 'topic' level of coding
centroid_idxs = {
    'Did real research': [287, 288, 289, 290, 291, 292, 293, 294, 479],
    'Felt like a scientist': [295, 296, 297, 298, 299, 478],
    'Understanding real research': [300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 463, 464, 465, 466, 467, 468, 469, 470],
    'Failure': [310, 311, 312, 313, 314, 315, 316, 471, 472, 473, 474],
    'Iteration': [317, 318, 319, 320, 321, 322],
    'Scientific practices': [323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424],
    'Decision making': [346, 347, 348, 349, 350, 351, 397, 398, 399, 400, 401, 475, 476, 477],
    'No Code': [0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 23, 24, 25, 26, 27, 28, 30, 32, 33, 35, 36, 37, 38, 39, 42, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 75, 80, 81, 82, 83, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 110, 111, 112, 113, 114, 117, 118, 119, 124, 125, 126, 127, 128, 129, 132, 133, 134, 135, 136, 137, 138, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 184, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 198, 199, 201, 202, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 221, 224, 225, 226, 227, 228, 229, 230, 231, 232, 234, 235, 236, 237, 238, 239, 240, 241, 246, 247, 248, 251, 254, 255, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 272, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462]
}

centroid_indices = [centroid_idxs[category] for category in coded_category_names]

# Plot showing how J-S divergence changes with number of dimensions in embeddings
scaling = "exponential"
metric = "cosine"
alpha_range = np.linspace(0, 400, 200)

# x: embedding dimensionality; y: best JS divergence at that dimensionality; c: best alpha
x = np.zeros(len(embedding_matrices))
y = np.zeros(len(embedding_matrices))
c = np.zeros(len(embedding_matrices))

for i, E in enumerate(embedding_matrices):
    x[i] = E.shape[1]  # dimension on x-axis
    temp = np.zeros(len(alpha_range))

    for j, alph in enumerate(alpha_range):
        scores = embedding_score(E, alph, centroid_indices, metric, scaling)
        temp[j] = np.sum(js_divergence(coded_scores, scores))

    y[i] = temp.min()
    c[i] = alpha_range[temp.argmin()]

fig = plt.figure(dpi=500)
# Create a plot with the number of articles on the x-axis and the JS divergence on the y-axis
plt.scatter(x, y, c=c, cmap="viridis")
plt.plot(x, y, label="JS divergence")
plt.colorbar(label="alpha")
plt.xlabel("Embedding dimension")
plt.ylabel("JS divergence")
plt.title("JS divergence by embedding dimension")
plt.xscale('log', base=2)

ax = plt.gca()
ax.xaxis.set_major_locator(FixedLocator(sorted(x)))
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{int(v):,}"))

plt.legend()
plt.tight_layout()
plt.savefig('DimensionChange.png', bbox_inches='tight')
plt.show()
