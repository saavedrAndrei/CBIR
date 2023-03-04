import os
import numpy as np
from skimage.io import imread
from skimage.feature import local_binary_pattern
from scipy.spatial.distance import cdist

def cbir_search(query_paths, k):
    # Step 1: Data collection
    dataset_path = './database/c_index/'
    dataset_filenames = os.listdir(dataset_path)

    # Step 2: Feature extraction
    lbp_radius = 3
    lbp_n_points = 8 * lbp_radius
    lbp_histogram_size = 256
    lbp_features = []

    for i, filename in enumerate(dataset_filenames):
        image = imread(os.path.join(dataset_path, filename), as_gray=True)
        lbp = local_binary_pattern(image, lbp_n_points, lbp_radius, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=lbp_histogram_size, range=(0, lbp_histogram_size))
        lbp_features.append(hist)

    lbp_features = np.array(lbp_features)

    # Step 3: Indexing
    # Create an inverted file to index the LBP features
    inverted_file = {}
    for i, hist in enumerate(lbp_features):
        for j in range(lbp_histogram_size):
            if hist[j] > 0:
                if j not in inverted_file:
                    inverted_file[j] = []
                inverted_file[j].append(i)

    # Step 4: Query processing
    search_results = []
    for query_path in query_paths:
        query_image = imread(query_path, as_gray=True)
        query_lbp = local_binary_pattern(query_image, lbp_n_points, lbp_radius, method="uniform")
        query_hist, _ = np.histogram(query_lbp.ravel(), bins=lbp_histogram_size, range=(0, lbp_histogram_size))

        # Compute the EMD distance between the query histogram and the histograms in the dataset
        matching_indices = set()
        for j in range(lbp_histogram_size):
            if query_hist[j] > 0:
                if j in inverted_file:
                    matching_indices.update(inverted_file[j])

        matching_features = lbp_features[list(matching_indices)]
        distances = cdist(matching_features, query_hist.reshape(1, -1), metric="emd")
        sorted_indices = np.argsort(distances.ravel())
        results_indices = [matching_indices[i] for i in sorted_indices[:k]]
        search_results.append([dataset_filenames[i] for i in results_indices])

    return search_results

query_paths = ["database/search/apple_0_airplane_arriving.png"]
k = 10
search_results = cbir_search(query_paths, k)