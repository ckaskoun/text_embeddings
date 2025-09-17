"""
File for assigning new embeddings to files based on a manual selection of
sentences from already coded files.
"""

# Imports
import create_embeddings as ct
import format_data as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from embedding_functions import js_divergence, embedding_score
import pickle as pkl

# Create a dataframe of hand-coded sentences for calculation of alpha
dict_codes = dt.format_coded("") # Path to coded files goes here
df_codes = dt.combine_coded(dict_codes)
df_codes_weighted = dt.weight_coded(df_codes, 'category') # 'category' for highest level of coding

# Embed cand-coded sentences for centroid calculation

# If embeddings already saved in pickle
# with open("coded_embeddings.pkl", "rb") as f:
#     dict_code_embed = pkl.load(f)

# If need new embeddings
dict_code_embed = ct.embed_files("") # Path to coded files goes here
with open("coded_embeddings.pkl", "wb") as f:
    pkl.dump(dict_code_embed, f)

df_code_embed = dt.combine_embedded(dict_code_embed)

# Create a matrix with the embeddings
code_embed_matrix = np.vstack(df_code_embed["Embeddings"].values)

# Create a list of the categories
coded_category_names = list(df_codes_weighted.columns[4:5]) + list(df_codes_weighted.columns[6:]) # Remove redundant Indirect Authenticity category

# Option for 'topic' level of coding
# coded_category_names = ( list(df_codes_weighted.columns[4:7]) + list(df_codes_weighted.columns[11:12])
#                  + list(df_codes_weighted.columns[13:]) ) # Remove redundant categories

# Scores for later alpha comparison
coded_scores = np.vstack(df_codes_weighted[coded_category_names].values)
coded_scores = coded_scores.astype(float)

# Manual selection of representative sentences for centroid calculation
centroid_idxs = {
    'Direct Authenticity': [287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 402, 403, 463, 464, 465, 466, 467, 468, 469, 470, 478, 479],
    'Research Components': [310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 397, 398, 399, 400, 401, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 471, 472, 473, 474, 475, 476, 477],
    'No Code': [0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 23, 24, 25, 26, 27, 28, 30, 32, 33, 35, 36, 37, 38, 39, 42, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 75, 80, 81, 82, 83, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 110, 111, 112, 113, 114, 117, 118, 119, 124, 125, 126, 127, 128, 129, 132, 133, 134, 135, 136, 137, 138, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 184, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 198, 199, 201, 202, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 221, 224, 225, 226, 227, 228, 229, 230, 231, 232, 234, 235, 236, 237, 238, 239, 240, 241, 246, 247, 248, 251, 254, 255, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 272, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462]
}

# Option for 'topic' level of coding
# centroid_idxs = {
#     'Did real research': [287, 288, 289, 290, 291, 292, 293, 294, 479],
#     'Felt like a scientist': [295, 296, 297, 298, 299, 478],
#     'Understanding real research': [300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 463, 464, 465, 466, 467, 468, 469, 470],
#     'Failure': [310, 311, 312, 313, 314, 315, 316, 471, 472, 473, 474],
#     'Iteration': [317, 318, 319, 320, 321, 322],
#     'Scientific practices': [323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424],
#     'Decision making': [346, 347, 348, 349, 350, 351, 397, 398, 399, 400, 401, 475, 476, 477],
#     'No Code': [0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 23, 24, 25, 26, 27, 28, 30, 32, 33, 35, 36, 37, 38, 39, 42, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 75, 80, 81, 82, 83, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 110, 111, 112, 113, 114, 117, 118, 119, 124, 125, 126, 127, 128, 129, 132, 133, 134, 135, 136, 137, 138, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 184, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 198, 199, 201, 202, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 221, 224, 225, 226, 227, 228, 229, 230, 231, 232, 234, 235, 236, 237, 238, 239, 240, 241, 246, 247, 248, 251, 254, 255, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 272, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462]
# }

# build a Python list of lists
centroid_indices = [centroid_idxs[category] for category in coded_category_names]

# Function for manually determining alpha from graph. Only needs to be used when
# sentences for centroids are changed, then is optional (as the result stays the same).

# # The uniform benchmark - Calculated by generating a matrix of 0.1
# zero_p_one_matrix = np.zeros((480,8)) + 0.1
# zero_p_one_score_unrounded = np.sum(js_divergence(zero_p_one_matrix, coded_scores))
#
# # The Random benchmark - Calculated by generating random vectors
# n = 2000 # Number of times we create random vectors
# JS_random_unrounded = np.zeros(n) # Array to store the JS scores for each n-run
#
# for i in range(n):
#     # Generate random vectors and normalize them (read sum to 1)
#     random_vector = np.random.rand(480,8)
#     random_vector = random_vector/np.sum(random_vector, axis=1)[:,None]
#
#     # Loop over all articles and calculate the
#     # JS divergence between the random vector and the LDA scores:
#     JS_random_unrounded[i]= np.sum(js_divergence(coded_scores, random_vector))
#
# # Define the different scalings and metrics
# scalings = ["exponential", "power"]
# metrics = ["cityblock", "euclidean", "cosine"]
#
# # Different alpha and plot changes in JS divergence sum
# alpha_list = [(np.linspace(0, 15, 101),         # Exponential + Manhatten
#                np.linspace(0, 170, 101),        # Exponential + Euclidean
#                np.linspace(0, 170, 101)),       # Exponential + Cosine
#             (np.linspace(0, 30, 51),            # Power + Manhatten
#              np.linspace(0, 30, 51),            # Power + Euclidean
#              np.linspace(0, 30, 101)),          # Power + Cosine
#             ]
#
# # Initialize the figure
# fig, axs = plt.subplots(len(scalings), 1, figsize=(16,4*len(scalings)), sharey=True)
#
# # Define the colors for the different metrics
# color = plt.cm.ocean(np.linspace(0,0.8,len(metrics)))
#
# # Loop over the different scalings
# for i, scaling in enumerate(scalings):
#
#     # Create a linspace for plotting the horizontal lines (y1 and y2)
#     nr = 1000
#     stop = alpha_list[i][2][-1]
#     linspace = np.linspace(0, stop, nr)
#
#     # The benchmark scores
#     y1 = JS_random_unrounded.mean() + np.zeros(nr) # + np.zeros(nr) to get the
#     y2 = zero_p_one_score_unrounded + np.zeros(nr) # right shape for plotting
#
#     #plotting y1 and y2 as horizontal lines
#     axs[i].plot(linspace, y1, "--",
#                 color="black", linewidth=2,
#                 label=f"Mean of random scores = {y1[0]:.2f}")
#     axs[i].plot(linspace, y2, "--",
#                 color="gray", linewidth=2,
#                 label=f"Uniform scores = {y2[0]:.2f}")
#
#     # Loop over the different metrics
#     for j, metric in enumerate(metrics):
#         alpha = alpha_list[i][j]
#         y = np.zeros(len(alpha))
#
#         # Calculate the JS divergence for each alpha
#         for k, alp in enumerate(alpha):
#             # Calculate the embedding scores for each alpha, metric and scaling
#             embedding_scores = embedding_score(code_embed_matrix, alp, centroid_indices,
#                                                metric, scaling, rounding=False)
#
#             # Assigning the sum of js-divergence for every article to the y-array
#             y[k] = np.sum(js_divergence(embedding_scores, coded_scores))
#
#         # Plot the JS divergence over alpha for each metric and scaling
#         axs[i].plot(alpha, y , "-o", label=metric.capitalize(), color=color[j], ms=3)
#
#         # Plotting the minimum of each curve:
#         min_index = np.nanargmin(y)
#         min_alpha = alpha[min_index]
#         min_y = y[min_index]
#
#         # Just for formatting the legend of the minimum
#         label_format = "({:.0f}, {:.2f})" if min_alpha % 1 == 0 else "({:.1f}, {:.2f})"
#         label = label_format.format(min_alpha, min_y)
#
#         # Plotting the minimum as a scatter point
#         axs[i].scatter(min_alpha, min_y, label=label, color=color[j], marker="o", s=100)
#
#     # Watch out for the y-axis limits as they are static
#     axs[i].set_ylim(25, 210)
#     # Formatting the plots
#     axs[i].set_xlabel(r"Scaling parameter $\alpha$", fontsize=16, fontweight='bold')
#     axs[i].set_title(f"{scaling.capitalize()} scaling", fontsize=20, fontweight='bold')
#     axs[i].spines[['right', "top"]].set_visible(False)
#     axs[i].set_xlim(-0.005*stop,stop)
#     axs[i].tick_params(axis='both', which='major', labelsize=16)
#     axs[i].legend(fontsize=14, loc="right", frameon=False, bbox_to_anchor=(1.475, 0.4))
#
# fig.tight_layout(w_pad=3, h_pad=3)
# fig.subplots_adjust(left=.1, right=.7)
# fig.supylabel(r"JS-divergence", fontsize=16, x=.03, y=0.5, fontweight='bold')
# plt.savefig('ScalingComparison.png')
# plt.show()

# Embed the files to assign codes to

# If embeddings already saved in pickle
# with open("new_embeddings.pkl", "rb") as f:
#     dict_assign = pkl.load(f)

# If need new embeddings
dict_assign = ct.embed_files("") # Path to uncoded files goes here
with open("new_embeddings.pkl", "wb") as f:
    pkl.dump(dict_code_embed, f)

df_assign = dt.combine_embedded(dict_assign)

# Store in matrix
assign_matrix = np.vstack(df_assign["Embeddings"].values)

# Set the parameters for the embedding scores to use in the analysis
alpha = 100 # Found from graphing
scaling = "exponential"
metric = "cosine" # Makes sense in a higher dimensional space

# Calculate the embedding scores
embedding_scores_8 = embedding_score(code_embed_matrix, alpha, centroid_indices,
                                      metric, scaling, rounding=False, assignment_matrix=assign_matrix)

# Add the embedding scores to the DataFrame in the same order as the coded scores
for i, category in enumerate(coded_category_names):
    df_assign[str(category)] = embedding_scores_8[:,i]

save_name = 'enter_name' # Need to manually change name to avoid overwrites if changing parameters
df_assign.to_excel(f'{save_name}.xlsx', index=False)
print(f'Saved coded memos to {save_name}.xlsx.')
