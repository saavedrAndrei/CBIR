from functools import partial
from typing import List, Dict, Tuple
import os
import numpy as np
import cv2
import pandas as pd


def read_images(path: str) -> List[Dict[str, any]]:
    return [
        {'path': os.path.join(path, filename), 'img': cv2.imread(os.path.join(path, filename))}
        for filename in os.listdir(path)
    ]


def detect_and_compute(sift: cv2.SIFT, img_dict: Dict[str, any]) -> Dict[str, any]:
    # Convert the image to HSV color space
    hsv_img = cv2.cvtColor(img_dict['img'], cv2.COLOR_BGR2HSV)
    # Compute the grid-based color histogram descriptor
    rows = 4
    cols = 4
    hist = []
    for i in range(rows):
        for j in range(cols):
            # Define the region of interest (ROI) for the current cell
            cell_h = hsv_img.shape[0] // rows
            cell_w = hsv_img.shape[1] // cols
            roi = hsv_img[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            # Compute the color histogram for the ROI
            hist_cell = cv2.calcHist([roi], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
            # Normalize the histogram
            hist_cell = cv2.normalize(hist_cell, hist_cell).flatten()
            # Add the histogram of the current cell to the descriptor
            hist.extend(hist_cell)
    # Convert the descriptor to a NumPy array
    des = np.array(hist)
    return {'path': img_dict['path'], 'des': des}


def match_images(bf: cv2.FlannBasedMatcher, search_des: Dict[str, any],
                 index_descriptors: List[Dict[str, any]]) -> pd.DataFrame:
    top_scores = []
    for index_des in index_descriptors:
        index_img = cv2.imread(index_des['path'])
        search_img = cv2.imread(search_des['path'])

        # Compute the color histogram descriptors
        index_hist = cv2.calcHist([index_img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        search_hist = cv2.calcHist([search_img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

        # Normalize the histograms
        cv2.normalize(index_hist, index_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(search_hist, search_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        # Compute the distance between the histograms
        dist = cv2.compareHist(search_hist, index_hist, cv2.HISTCMP_CHISQR)

        top_scores.append({'Index Image Path': index_des['path'], 'Similarity Score': dist})
    top_scores_df = pd.DataFrame(top_scores).sort_values(by='Similarity Score', ascending=True).head(20)
    top_scores_df.insert(0, 'Search Image Path', search_des['path'])
    top_scores_df.insert(2, 'Search Image', f'<img src="{search_des["path"]}">')
    top_scores_df.insert(3, 'Index Image', top_scores_df['Index Image Path'].apply(lambda x: f'<img src="{x}">'))
    top_scores_df = top_scores_df[['Search Image', 'Index Image', 'Similarity Score']]
    return top_scores_df


def main():
    dataset_index_path = './database/c_index/'
    dataset_search_path = './database/search/'
    sift = cv2.xfeatures2d.SIFT_create()
    bf = cv2.FlannBasedMatcher()
    dataset_index = read_images(dataset_index_path)
    dataset_search = read_images(dataset_search_path)

    dataset_search_descriptors = list(map(partial(detect_and_compute, sift), dataset_search))
    dataset_index_descriptors = list(map(partial(detect_and_compute, sift), dataset_index))

    match_images_partial = partial(match_images, bf)
    similarity_scores = list(map(match_images_partial, dataset_search_descriptors, [dataset_index_descriptors] * len(dataset_search_descriptors)))

    df = pd.concat(similarity_scores, ignore_index=True)
    with open('output_smart_sift.html', 'w') as f:
        f.write(df.to_html(index=False, escape=False))


if __name__ == '__main__':
    main()