"""
2D UMAP plotting of centroid representative sentences and subcentroids.
"""

# Hide Kmeans memory leak warnings
import warnings
warnings.filterwarnings("ignore",
                        category=UserWarning,
                        module=r"sklearn\.cluster\._kmeans")

import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import umap
import format_data as dt
from typing import Optional

# Load coded embeddings saved to pkl
with open("coded_embeddings.pkl", "rb") as f:
    dict_embeddings = pkl.load(f)

df_embeddings = dt.combine_embedded(dict_embeddings)
embedding_matrix = np.vstack(df_embeddings["Embeddings"].values)

# Normalize for Kmeans
def _l2norm_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, eps)

embedding_matrix_norm = _l2norm_rows(embedding_matrix)

# Manually chosen categories and indices
coded_category_names = ['Direct Authenticity', 'Research Components', 'No Code']

# # For 'topic' level
# coded_category_names = ['Did real research', 'Felt like a scientist', 'Understanding real research',
#                         'Failure', 'Iteration', 'Scientific practices',
#                         'Decision making', 'No Code']

# For 'category' level
centroid_idxs = {
    'Direct Authenticity': [287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 402, 403, 463, 464, 465, 466, 467, 468, 469, 470, 478, 479],
    'Research Components': [310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 397, 398, 399, 400, 401, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 471, 472, 473, 474, 475, 476, 477],
    'No Code': [0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 23, 24, 25, 26, 27, 28, 30, 32, 33, 35, 36, 37, 38, 39, 42, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 75, 80, 81, 82, 83, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 110, 111, 112, 113, 114, 117, 118, 119, 124, 125, 126, 127, 128, 129, 132, 133, 134, 135, 136, 137, 138, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 184, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 198, 199, 201, 202, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 221, 224, 225, 226, 227, 228, 229, 230, 231, 232, 234, 235, 236, 237, 238, 239, 240, 241, 246, 247, 248, 251, 254, 255, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 272, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462]
}

# # For 'topic' level
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

centroid_indices = [centroid_idxs[cat] for cat in coded_category_names]

# Combine representative sentences and corresponding labels (on normalized space)
combined_reps = []
categories = []
for category, idxs in zip(coded_category_names, centroid_indices):
    combined_reps.append(embedding_matrix_norm[idxs])
    categories.extend([category] * len(idxs))

combined_reps = np.vstack(combined_reps)   # (M, D) normalized
categories = np.array(categories)          # (M,)

# Keep colors consistent
palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
cat_colors = {cat: palette[i % len(palette)] for i, cat in enumerate(coded_category_names)}

# Choose number of subcentroids per topic (None = use all reps)
n_reps_per_topic = {
    'Direct Authenticity': 2,
    'Research Components': 4,
    'No Code': 6,
}

# # For 'topic' level
# n_reps_per_topic = {
#     'Did real research': 1,
#     'Felt like a scientist': 1,
#     'Understanding real research': 1,
#     'Failure': 1,
#     'Iteration': 1,
#     'Scientific practices': 2,
#     'Decision making': 1,
#     'No Code': 6,
# }

def _unique_rows_preserve_order(a: np.ndarray) -> np.ndarray:
    if a.size == 0:
        return a
    a_c = np.ascontiguousarray(a)
    view = a_c.view([('', a_c.dtype)] * a_c.shape[1])
    _, idx = np.unique(view, return_index=True)
    return a[idx]

# Helper function for finding subcentroids
def compute_subcentroids(reps: np.ndarray,
                         k: Optional[int] | None,
                         random_state: int = 42) -> np.ndarray:
    """
    Behavior:
      - If reps is empty: return (0, D).
      - If k is None: return reps, deduped, preserving first-seen order.
      - If k == 1: return the mean (fast, deterministic).
      - If k > 1 and N >= k: return EXACTLY k centers via KMeans.
      - If N < k: return order-preserving unique reps.

    Returns:
        np.ndarray of shape (num_centers, D). May be empty if reps is empty.
    """
    if not isinstance(reps, np.ndarray) or reps.ndim != 2:
        raise ValueError("`reps` must be a 2D numpy array of shape (N, D).")

    n, d = reps.shape
    if n == 0:
        return np.empty((0, d), dtype=reps.dtype)

    if k is None:
        return _unique_rows_preserve_order(reps)

    if n < k:
        return _unique_rows_preserve_order(reps)

    if k == 1:
        return reps.mean(axis=0, keepdims=True).astype(reps.dtype, copy=False)

    km = KMeans(n_clusters=k, random_state=random_state, n_init=10, algorithm="lloyd")
    km.fit(reps)
    return km.cluster_centers_.astype(reps.dtype, copy=False)

# Find subcentroids for each topic
s_centroids = {}
for category, idxs in zip(coded_category_names, centroid_indices):
    reps = embedding_matrix_norm[idxs]
    k = n_reps_per_topic.get(category, None)
    centers = compute_subcentroids(reps, k=k, random_state=42)
    s_centroids[category] = centers

# Reduce representative sentences using UMAP
umap2 = umap.UMAP(
    n_components=2,
    n_neighbors=5,
    min_dist=0.05,
    metric='cosine',
    random_state=17,
    init='spectral'
)
UMAP2_reps = umap2.fit_transform(combined_reps)

# Transform subcentroids for each category
s_centroids_2d = {}
for category, centers in s_centroids.items():
    if centers is not None and centers.size:
        s_centroids_2d[category] = umap2.transform(centers)
    else:
        s_centroids_2d[category] = np.empty((0, 2))

# Plot 2D UMAP
plt.figure(figsize=(12, 12))

# to avoid duplicate legend entries, track which labels we've added
added = set()

for category in coded_category_names:
    color = cat_colors[category]
    mask = (categories == category)

    # reps (points)
    label_reps = f"{category}"
    if label_reps not in added:
        plt.scatter(UMAP2_reps[mask, 0], UMAP2_reps[mask, 1],
                    s=14, alpha=0.85, color=color, label=label_reps)
        added.add(label_reps)
    else:
        plt.scatter(UMAP2_reps[mask, 0], UMAP2_reps[mask, 1],
                    s=14, alpha=0.85, color=color)

    # subcentroids (X markers)
    centers2d = s_centroids_2d.get(category, np.empty((0, 2)))
    if centers2d.size:
        label_sub = f"{category} subcentroid(s)"
        # background "stroke"
        plt.scatter(
            centers2d[:, 0], centers2d[:, 1],
            marker='x', s=130, linewidths=6.2, color='black',
            label=None,
            zorder=4
        )
        # foreground color
        plt.scatter(
            centers2d[:, 0], centers2d[:, 1],
            marker='x', s=90, linewidths=3.6, color=color,
            label=None if label_sub in added else label_sub,
            zorder=5
        )
        added.add(label_sub)

plt.title("2D UMAP: representative sentences per category and subcentroids")
plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
plt.legend(markerscale=1.2, frameon=True)
plt.tight_layout()
plt.savefig('umap.png')
plt.show()
