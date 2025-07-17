"""
Graphs and analysis of text embeddings. Compares assigned scores to human-coded
scores. Adapted from the G.E.V.I.R. repository used in the Odden et al. (2024)
paper.
"""

# Imports
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
from matplotlib.ticker import PercentFormatter
import matplotlib.colors as mcolors
from scipy.stats import kendalltau, pearsonr, spearmanr

# Create a dataframe of embeddings for comparison to human-coded scores
dict_embeddings = ct.embed_files("") # Path to coded files goes here
df_embeddings = dt.combine_embedded(dict_embeddings)

# Create a dataframe of the human-coded scores
dict_codes = dt.format_coded("") # Path to coded files goes here
df_codes = dt.combine_coded(dict_codes)
df_codes_weighted = dt.weight_coded(df_codes, 'category') # 'category' for highest level of coding

# Create a matrix with all the embeddings
embedding_matrix = np.vstack(df_embeddings["Embeddings"].values)

# Create a list of the categories
coded_category_names = list(df_codes_weighted.columns[4:5]) + list(df_codes_weighted.columns[6:]) # Remove redundant Indirect Authenticity category

# Create a matrix of the hand-coded weights
# Shape will be (n_sentences, n_categories)
coded_scores = np.vstack(df_codes_weighted[coded_category_names].values)
coded_scores = coded_scores.astype(float)

# Number of articles to be included in calculating the centroid. This analysis
# uses a random selection of the highest human-scored codes, and must have the
# same amount for each category. Comment out for manual option.
sum_categories = dt.sum_codes(df_codes_weighted)
min_code = float('inf')
for k in sum_categories:
    if sum_categories[k] < min_code:
        min_code = sum_categories[k]

n_sentences_for_centroid = min_code # Gives maximum amount for a symmetric sample size

# Sort indices along categories after human scores
sorted_indices = np.argsort(coded_scores, axis=0)

# Reverse the slice to have max values at the top
# Transpose to get the same shape (n_categories, n_sentences_for_centroid)
centroid_indices = sorted_indices[-n_sentences_for_centroid:][::-1].T

# This is another option, where you can manually choose the sentences for each category
# (can be different lengths)

# centroid_idxs = {
#     'Direct Authenticity': [287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 402, 403],
#     'Research Components': [310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 397, 398, 399, 400, 401, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417],
#     'No Code': [0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 23, 24, 25, 26, 27, 28, 30, 32, 33, 35, 36, 37, 38, 39, 42, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 75, 80, 81, 82, 83, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 110, 111, 112, 113, 114, 117, 118, 119, 124, 125, 126, 127, 128, 129, 132, 133, 134, 135, 136, 137, 138, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 184, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 198, 199, 201, 202, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 221, 224, 225, 226, 227, 228, 229, 230, 231, 232, 234, 235, 236, 237, 238, 239, 240, 241, 246, 247, 248, 251, 254, 255, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 272, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396]
# }
#
# centroid_indices = [centroid_idxs[category] for category in coded_category_names]

# We also matches this with a color list, such that we get the same colors for the same categories
# through the first part of the analysis
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"] # Fixed number of colors to match 'category' level
category_NameColor_dict = dict(zip(coded_category_names, colors))

# Creationg of a graph of J-S divergence for different scalings
# The uniform benchmark - Calculated by generating a matrix of 0.1
zero_p_one_matrix = np.zeros((330,3)) + 0.1
zero_p_one_score_unrounded = np.sum(js_divergence(zero_p_one_matrix, coded_scores))

# The Random benchmark - Calculated by generating random vectors
n = 2000 # Number of times we create random vectors
JS_random_unrounded = np.zeros(n) # Array to store the JS scores for each n-run

for i in range(n):
    # Generate random vectors and normalize them (read sum to 1)
    random_vector = np.random.rand(330,3)
    random_vector = random_vector/np.sum(random_vector, axis=1)[:,None]

    # Loop over all articles and calculate the
    # JS divergence between the random vector and the LDA scores:
    JS_random_unrounded[i]= np.sum(js_divergence(coded_scores, random_vector))

# Define the different scalings and metrics
scalings = ["exponential", "power"]
metrics = ["cityblock", "euclidean", "cosine"]

# Different alpha values for the different scalings and metrics as
# some scalings and metrics are more sensitive to alpha than others
alpha_list = [(np.linspace(0, 20, 101),         # Exponential + Manhatten
               np.linspace(0, 400, 101),        # Exponential + Euclidean
               np.linspace(0, 400, 101)),       # Exponential + Cosine
            (np.linspace(0, 100, 51),            # Power + Manhatten
             np.linspace(0, 100, 51),            # Power + Euclidean
             np.linspace(0, 100, 101)),          # Power + Cosine
            ]

# Initialize the figure
fig, axs = plt.subplots(len(scalings), 1, figsize=(16,4*len(scalings)), sharey=True)

# Define the colors for the different metrics
color = plt.cm.ocean(np.linspace(0,0.8,len(metrics)))

# Loop over the different scalings
for i, scaling in enumerate(scalings):

    # Create a linspace for plotting the horizontal lines (y1 and y2)
    nr = 1000
    stop = alpha_list[i][2][-1]
    linspace = np.linspace(0, stop, nr)

    # The benchmark scores
    y1 = JS_random_unrounded.mean() + np.zeros(nr) # + np.zeros(nr) to get the
    y2 = zero_p_one_score_unrounded + np.zeros(nr) # right shape for plotting

    #plotting y1 and y2 as horizontal lines
    axs[i].plot(linspace, y1, "--",
                color="black", linewidth=2,
                label=f"Mean of random scores = {y1[0]:.2f}")
    axs[i].plot(linspace, y2, "--",
                color="gray", linewidth=2,
                label=f"Uniform scores = {y2[0]:.2f}")

    # Loop over the different metrics
    for j, metric in enumerate(metrics):
        alpha = alpha_list[i][j]
        y = np.zeros(len(alpha))

        # Calculate the JS divergence for each alpha
        for k, alp in enumerate(alpha):
            # Calculate the embedding scores for each alpha, metric and scaling
            embedding_scores = embedding_score(embedding_matrix, alp, centroid_indices,
                                               metric, scaling, rounding=False)

            # Assigning the sum of js-divergence for every article to the y-array
            y[k] = np.sum(js_divergence(embedding_scores, coded_scores))

        # Plot the JS divergence over alpha for each metric and scaling
        axs[i].plot(alpha, y , "-o", label=metric.capitalize(), color=color[j], ms=3)

        # Plotting the minimum of each curve:
        min_index = np.nanargmin(y)
        min_alpha = alpha[min_index]
        min_y = y[min_index]

        # Just for formatting the legend of the minimum
        label_format = "({:.0f}, {:.2f})" if min_alpha % 1 == 0 else "({:.1f}, {:.2f})"
        label = label_format.format(min_alpha, min_y)

        # Plotting the minimum as a scatter point
        axs[i].scatter(min_alpha, min_y, label=label, color=color[j], marker="o", s=100)

    # Watch out for the y-axis limits as they are static
    axs[i].set_ylim(60, 165)
    # Formatting the plots
    axs[i].set_xlabel(r"Scaling parameter $\alpha$", fontsize=16, fontweight='bold')
    axs[i].set_title(f"{scaling.capitalize()} scaling", fontsize=20, fontweight='bold')
    axs[i].spines[['right', "top"]].set_visible(False)
    axs[i].set_xlim(-0.005*stop,stop)
    axs[i].tick_params(axis='both', which='major', labelsize=16)
    axs[i].legend(fontsize=14, loc="right", frameon=False, bbox_to_anchor=(1.475, 0.4))

fig.tight_layout(w_pad=3, h_pad=3)
fig.subplots_adjust(left=.1, right=.7)
fig.supylabel(r"JS-divergence", fontsize=16, x=.03, y=0.5, fontweight='bold')
plt.savefig('ScalingComparison.png')
plt.show()

# Set the parameters for the embedding scores to use in for the analysis.
# Must be manually set, and can be determined optimally from the graph.
alpha = 84 # Found from calculation and graphing earlier
scaling = "exponential"
metric = "cosine" # Makes the most sense in a higher dimensional space

# Calculate the embedding scores
# Name arbitary, but 3 for now to represent '3' categories
embedding_scores_3 = embedding_score(embedding_matrix, alpha, centroid_indices,
                                      metric, scaling, rounding=False)

# Add the embedding scores to the DataFrame in the same order as the coded scores
for i, category in enumerate(coded_category_names):
    df_embeddings[str(category)] = embedding_scores_3[:,i]

# Array to store the JS divergence
JS_divergence_array = js_divergence(embedding_scores_3, coded_scores)

# Plotting the distribution of the JS divergence
plt.hist(JS_divergence_array, bins=50)
#median
plt.plot([np.median(JS_divergence_array), np.median(JS_divergence_array)], [0, 20],
         "--", color="gray", label="Median")
#mean
plt.plot([np.mean(JS_divergence_array), np.mean(JS_divergence_array)], [0, 20],
         "--", color="black", label="Mean")
plt.xlabel("JS divergence")
plt.ylabel("Number of sentences")
plt.legend()
plt.savefig('SentenceDivergence.png')
plt.show()

# Plot showing how J-S divergence changes with number of sentences per centroid.
# This is where the earlier theoretical limit came from.
number_of_articles_arr = np.arange(1, 19)
scaling = "exponential"
metric = "cosine"
alpha_range = np.linspace(0, 400, 101)

x = np.zeros(len(number_of_articles_arr))
y = np.zeros(len(number_of_articles_arr))
c = np.zeros(len(number_of_articles_arr))

# Sort indices along categories after Coded scores
sorted_indices = np.argsort(coded_scores, axis=0)

for i, nr in enumerate(number_of_articles_arr):
    x[i] = nr
    # Reverse the slice to have max values at the top
    # Transpose to get the same shape (n_categories, n_articles_for_centroid)
    centroid_indices = sorted_indices[-nr:][::-1].T

    temp = np.zeros(len(alpha_range))
    #calculate y[i] using old method
    for j, alph in enumerate(alpha_range):
        scores = embedding_score(embedding_matrix, alph, centroid_indices, metric, scaling)
        temp[j] = np.sum(js_divergence(coded_scores, scores))

    y[i] = np.min(temp)
    c[i] = alpha_range[np.argmin(temp)]

fig = plt.figure(dpi=500)
# Create a plot with the number of articles on the x-axis and the JS divergence on the y-axis
plt.scatter(x, y, c=c, cmap="viridis")
plt.plot(x, y, label="JS divergence")
plt.colorbar(label="alpha")
plt.xlabel("Number of sentences")
plt.ylabel("JS divergence")
plt.title("JS divergence by # of sentences in centroid")
plt.legend()
plt.savefig('DivergenceChange.png')
plt.show()

# 2 Plots to represent the scores given in the 1st, 2nd, and 3rd highest categories
# for the memos.
# Choosing which scores we want to plot, and also given them a title
scores_title = {"Coded": coded_scores, "Embedding scores": embedding_scores_3}

# Initialize the figure
fig, axs = plt.subplots(len(scores_title), 1, figsize=(20,20), sharex=True, sharey=False)

colors = ["C0", "C1", "C3"] #primary, secondary, tertiary colors
labels=["Primary Topic", "Secondary Topic", "Tertiary Topic"]

# Create handles for the legend
handles = [mpatches.Patch(color=colors[0], label=labels[0]),
           mpatches.Patch(color=colors[1], alpha=0.5, label=labels[1]),
           mpatches.Patch(color=colors[2], alpha=0.5, label=labels[2])]

# Loop over the different scores
for i, title in enumerate(scores_title):

    # Calculate the primary, secondary and tertiary topic scores
    primary_topic_score = np.max(scores_title[title], axis=1)
    secondary_topic_score = np.partition(scores_title[title], -2, axis=1)[:, -2]
    tertiary_topic_score = np.partition(scores_title[title], -3, axis=1)[:, -3]

    # Plotting the histograms of the scores
    axs[i].hist(primary_topic_score, bins=50, color=colors[0])
    axs[i].hist(secondary_topic_score, bins=50, alpha=0.5, color=colors[1])
    axs[i].hist(tertiary_topic_score, bins=50, alpha=0.5, color=colors[2])

    # Formatting the plots
    axs[i].set_title(title, fontsize=30, fontweight='bold')
    axs[i].set_xlabel("Topic Score", fontsize=24, labelpad=20, fontweight='bold')
    axs[i].set_ylabel("Count", fontsize=24, fontweight='bold')
    axs[i].spines['right'].set_visible(False)
    axs[i].spines['top'].set_visible(False)
    axs[i].tick_params(which='both', bottom=True, top=False, labelbottom=True, labelsize=20)
    axs[i].set_xlim(left=0)
    axs[i].legend(handles=handles, labels=labels, fontsize=24,
                  loc='upper center', bbox_to_anchor=(0.89, 0.95))

# Removing the y tick at 0, since sharey=True, it will be removed for both subplots
axs[0].set_yticks(axs[0].get_yticks()[1:])

fig.tight_layout(pad=6.0)
plt.savefig('TopicDistributions.png')
plt.show()

# Sns violin plot
sns.set_theme(style="whitegrid")
s1 = np.max(coded_scores, axis=1)
s2 = np.partition(coded_scores, -2, axis=1)[:, -2]
s3 = np.partition(coded_scores, -3, axis=1)[:, -3]

p1 = np.max(embedding_scores_3, axis=1)
p2 = np.partition(embedding_scores_3, -2, axis=1)[:, -2]
p3 = np.partition(embedding_scores_3, -3, axis=1)[:, -3]

scores = np.concatenate([s1, s2, s3, p1, p2, p3])
type_ = np.repeat(["Coded", "Embedding"], len(s1)*3)
topic = np.concatenate([np.repeat(["Primary Topic"], len(s1)),
                        np.repeat(["Secondary Topic"], len(s2)),
                        np.repeat(["Tertiary Topic"], len(s3)),
                        np.repeat(["Primary Topic"], len(p1)),
                        np.repeat(["Secondary Topic"], len(p2)),
                        np.repeat(["Tertiary Topic"], len(p3))])
df_violin = pd.DataFrame({"Scores": scores, "Type": type_, "Topic": topic})

fig, ax = plt.subplots(figsize=(14, 7), dpi=500)

color_lda = "skyblue"
color_emb = (0.9, 0.3, 0.4)
my_pal = {"Coded": color_lda, "Embedding": color_emb}

ax = sns.violinplot(data=df_violin, y="Scores", x="Topic", hue="Type", split=True, ax=ax, inner="quart", gap=0.1, inner_kws={"alpha":0.8, "linewidth":2.5}, cut=0, palette=my_pal)

# Adjusting ticks
labels = ["Primary Topic", "Secondary Topic", "Tertiary Topic"]
ax.set_xticks(labels)
ax.set_xticklabels(labels,fontsize=20, fontweight='bold')
ax.tick_params(axis='y', which='major', labelsize=16)

# Fix labels
ax.set(xlabel=None)
ax.set_ylabel("Topic Scores", fontsize=20, fontweight='bold')

# Add legend
ax.legend(fontsize=20)

plt.savefig('TopicDistributions2.png')
plt.show()
sns.reset_defaults()

# Kendelltau correlation coefficient, for accuracy assessment.
for i, (name, color) in enumerate(category_NameColor_dict.items()):
    # Get the embedding scores and LDA scores for category i
    x = coded_scores[:, i]
    y = embedding_scores_3[:, i]

    # Here you can change to either pearsonr, spearmanr or kendalltau
    r, p = kendalltau(x, y)
    print(f"{name}: r = {r:.2f}, p = {p:g}")
    # r is the correlation coefficient
    # p is the p-value

# Plot heatmaps of scores given for specific sections of the embeddings
# Create color map for heatmap plots
colors = [(0.95, 0.95, 1), (0.5, 0.75, 1), (0.4, 0.5, 0.9),
            (0.3, 0.4, 0.8), (0.2, 0.3, 0.7), (0.1, 0.15, 0.5)]
cmap = mcolors.LinearSegmentedColormap.from_list("custom", colors, N=256)
# Legend patches is the categories for the coded scores
legend_patches = [mpatches.Patch(color='white', label=(f'{i+1}: {category}')) for i, category in enumerate(coded_category_names)]

# Create the inputs for the plot_heatmap function
scores_list = []
title_list = []

# List of indices of the articles that are closest to the mean
heatmap_indices = np.abs(JS_divergence_array - np.mean(JS_divergence_array)).argsort()[:5]

# Filling in the scores_list and title_list, for the plot_heatmap function
for i, index in enumerate(heatmap_indices):
    scores_list.append([coded_scores[index, :], embedding_scores_3[index, :]])
    title = f'{df_codes_weighted["File"][index]} ({df_codes_weighted["Lines"][index]}) \n {df_codes_weighted["Sentences"][index][:120]}'
    if len(df_codes_weighted["Sentences"][index]) > 120:
        title = title + '...'
    title_list.append(title)

# Using the heatmap function to plot the heatmap
fig, axs = plt.subplots(5,1, figsize=(12, 14))
fig, axs = plot_heatmap(fig, axs, scores_list, title_list, legend_patches, cmap)
plt.savefig('HeatmapMean.png')
plt.show()

# Take the JS divergence array and argsort it to get the indices of the sorted array
# List of five articles with the highest JS divergence
heatmap_indices = np.argsort(JS_divergence_array)[-5:][::-1] # [::-1] is to flip the lis

# Create the inputs for the plot_heatmap function
scores_list = []
title_list = []

# Filling in the scores_list and title_list, for the plot_heatmap function
for i, index in enumerate(heatmap_indices):
    scores_list.append([coded_scores[index, :], embedding_scores_3[index, :]])
    title = f'{df_codes_weighted["File"][index]} ({df_codes_weighted["Lines"][index]}) \n {df_codes_weighted["Sentences"][index][:120]}'
    if len(df_codes_weighted["Sentences"][index]) > 120:
        title = title + '...'
    title_list.append(title)


# Use the heatmap function to plot the heatmap
fig, axs = plt.subplots(5,1, figsize=(12, 14))
fig, axs = plot_heatmap(fig, axs, scores_list, title_list, legend_patches, cmap)
plt.savefig('HeatmapMaximum.png')
plt.show()

# Create new inputs for the plot_heatmap function for a median sample
scores_list = []
title_list = []

# List of indices of the articles that are closest to the median
heatmap_indices = np.abs(JS_divergence_array - np.median(JS_divergence_array)).argsort()[:5]

# Filling in the scores_list and title_list, for the plot_heatmap function
for i, index in enumerate(heatmap_indices):
    scores_list.append([coded_scores[index, :], embedding_scores_3[index, :]])
    title = f'{df_codes_weighted["File"][index]} ({df_codes_weighted["Lines"][index]}) \n {df_codes_weighted["Sentences"][index][:120]}'
    if len(df_codes_weighted["Sentences"][index]) > 120:
        title = title + '...'
    title_list.append(title)

# Using the heatmap function to plot the heatmap
fig, axs = plt.subplots(5,1, figsize=(12, 14))
fig, axs = plot_heatmap(fig, axs, scores_list, title_list, legend_patches, cmap)
plt.savefig('HeatmapMedian.png')
plt.show()
