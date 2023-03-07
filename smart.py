import cv2
import os
import pandas as pd
import numpy as np
from functools import reduce
from scipy.stats import wasserstein_distance

# Define grid and cell sizes
grid_size = (4, 4)
cell_size = (30, 30)


# Define helper functions
def read_images(path):
    return [{'path': os.path.join(path, filename), 'img': cv2.imread(os.path.join(path, filename))} for filename in
            os.listdir(path)]


def calculate_histograms(image, grid_size, cell_size):
    histograms = []
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            cell = image[i * cell_size[0]:(i + 1) * cell_size[0], j * cell_size[1]:(j + 1) * cell_size[1], :]
            hist = cv2.calcHist([cell], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            histograms.extend(hist)
    return histograms


def compare_histograms(search_des, index_des):
    return cv2.compareHist(np.array(index_des), np.array(search_des), cv2.HISTCMP_CORREL)


def process_image(dataset, grid_size, cell_size):
    return [{'path': img['path'], 'descriptors': calculate_histograms(img['img'], grid_size, cell_size)} for img in
            dataset]


def process_search_images(dataset_search, dataset_index, grid_size, cell_size):
    return [{'search_path': img['path'], 'search_descriptors': calculate_histograms(img['img'], grid_size, cell_size),
             'index': dataset_index} for img in dataset_search]


def find_top_scores(search_image):
    top_scores = []
    for index_image in search_image['index']:
        score = compare_histograms(search_image['search_descriptors'], index_image['descriptors'])
        top_scores.append({'Index Image Path': index_image['path'], 'Similarity Score': score})
    top_scores_df = pd.DataFrame(top_scores).sort_values(by='Similarity Score', ascending=False).head(20)
    top_scores_df.insert(0, 'Search Image Path', search_image['search_path'])
    top_scores_df.insert(2, 'Search Image', f'<img src="{search_image["search_path"]}">')
    top_scores_df.insert(3, 'Index Image', top_scores_df['Index Image Path'].apply(lambda x: f'<img src="{x}">'))
    return top_scores_df[['Search Image', 'Index Image', 'Similarity Score']]


# Define dataset paths
dataset_index_path = './database/c_index/'
dataset_search_path = './database/search/'

# Read images and calculate descriptors
dataset_index = read_images(dataset_index_path)
dataset_search = read_images(dataset_search_path)
dataset_index_descriptors = process_image(dataset_index, grid_size, cell_size)
dataset_search_descriptors = process_search_images(dataset_search, dataset_index_descriptors, grid_size, cell_size)

# Find similarity scores and concatenate dataframes
dfs = list(map(find_top_scores, dataset_search_descriptors))
df = reduce(lambda left, right: pd.concat([left, right], ignore_index=True), dfs)

# Write dataframe to html file
with open('output_smart.html', 'w') as f:
    f.write(df.to_html(index=False, escape=False))
