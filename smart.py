import cv2
import os
import pandas as pd
from cv2 import xfeatures2d
import numpy as np

dataset_index_path = './database/c_index/'
dataset_index = []
for filename in os.listdir(dataset_index_path):
    img_index = cv2.imread(os.path.join(dataset_index_path, filename))
    dataset_index.append({'path': os.path.join(dataset_index_path, filename), 'img': img_index})

dataset_search_path = './database/search/'
dataset_search = []
for filename in os.listdir(dataset_search_path):
    img_search = cv2.imread(os.path.join(dataset_search_path, filename))
    dataset_search.append({'path': os.path.join(dataset_search_path, filename), 'img': img_search})

grid_size = (8, 8)
cell_size = (15, 15)

dataset_search_descriptors = []
for img in dataset_search:
    descriptor_search = []
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            cell = img['img'][i*cell_size[0]:(i+1)*cell_size[0], j*cell_size[1]:(j+1)*cell_size[1], :]
            hist = cv2.calcHist([cell], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            descriptor_search.extend(hist)
    dataset_search_descriptors.append(descriptor_search)

dataset_index_descriptors = []
for img in dataset_index:
    descriptor_index = []
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            cell = img['img'][i*cell_size[0]:(i+1)*cell_size[0], j*cell_size[1]:(j+1)*cell_size[1], :]
            hist = cv2.calcHist([cell], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            descriptor_index.extend(hist)
    dataset_index_descriptors.append(descriptor_index)

similarity_scores = []
dfs = []
for i, search_des in enumerate(dataset_search_descriptors):
    search_image_path = dataset_search[i]['path']
    search_image_name = os.path.splitext(os.path.basename(search_image_path))[0]
    top_scores = []
    for j, index_des in enumerate(dataset_index_descriptors):
        index_image_path = dataset_index[j]['path']
        index_image_name = os.path.splitext(os.path.basename(index_image_path))[0]
        score = cv2.compareHist(np.array(index_des), np.array(search_des), cv2.HISTCMP_INTERSECT)
        top_scores.append({'Index Image Path': index_image_path, 'Similarity Score': score})
    top_scores_df = pd.DataFrame(top_scores).sort_values(by='Similarity Score', ascending=False).head(20)
    top_scores_df.insert(0, 'Search Image Path', search_image_path)
    top_scores_df.insert(2, 'Search Image', f'<img src="{search_image_path}">')
    top_scores_df.insert(3, 'Index Image', top_scores_df['Index Image Path'].apply(lambda x: f'<img src="{x}">'))
    top_scores_df = top_scores_df[['Search Image', 'Index Image', 'Similarity Score']]
    dfs.append(top_scores_df)

df = pd.concat(dfs, ignore_index=True)
# Write dataframe to html file
with open('output_smart.html', 'w') as f:
    f.write(df.to_html(index=False, escape=False))