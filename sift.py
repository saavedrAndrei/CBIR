import cv2 as cv
import os
import pandas as pd

index_dir_path = os.path.abspath('./database/c_index/') 
search_dir_path = os.path.abspath('./database/search/')
search_dir = os.listdir(search_dir_path)
index_dir = os.listdir(index_dir_path)

def get_sift_image_descriptors(search_img, idx_img):
    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # Find keypoints and compute descriptors with SIFT
    _, search_des_sift = sift.detectAndCompute(search_img,None)
    _, idx_des_sift = sift.detectAndCompute(idx_img,None)
    return search_des_sift, idx_des_sift 

def get_similarity_from_desc(approach, search_desc, idx_desc):
    if approach == 'sift' or approach == 'orb_sift':
        # BFMatcher with euclidean distance
        bf = cv.BFMatcher()
    else:
        # BFMatcher with hamming distance
        bf = cv.BFMatcher(cv.NORM_HAMMING)
    matches = bf.match(search_desc, idx_desc)
    # Distances between search and index features that match
    distances = [m.distance for m in matches]
    # Distance between search and index images
    distance = sum(distances) / len(distances)
    # If distance == 0 -> similarity = 1
    similarity = 1 / (1 + distance)
    return similarity
    
def get_ranking_from_desc(approach):
    dfs = []
    for search_img in search_dir:
        df = pd.DataFrame(columns=['search_image', 'index_image', 'similarity_score'])
        similarities = []
        index_images = []
        search_image_tag = f'<img src="./database/search/{search_img}" width="100px"/>'
        for idx_img in index_dir:
           # Read images in gray scale
           search = cv.imread(os.path.join(search_dir_path, search_img) , cv.IMREAD_GRAYSCALE)
           idx = cv.imread(os.path.join(index_dir_path, idx_img), cv.IMREAD_GRAYSCALE)
           if approach == 'sift':
               img_descriptors = get_sift_image_descriptors(search,idx)
           similarities.append(get_similarity_from_desc(approach, img_descriptors[0], img_descriptors[1]))
           index_images.append(f'<img src="./database/index/{idx_img}" width="100px"/>')
        df['search_image'] = [search_image_tag] * len(index_dir)
        df['index_image'] = index_images
        df['similarity_score'] = similarities
        df = df.sort_values(by='similarity_score', ascending=False)
        # Select top 20 matches for every search image
        df = df.head(20)
        # Insert image tags into HTML output
        df['search_image'] = df['search_image'].apply(lambda x: f'{x}')
        df['index_image'] = df['index_image'].apply(lambda x: f'{x}')
        dfs.append(df)
    # Build ranking with similarity scores and images for all search images
    ranking = pd.concat(dfs).reset_index(drop=True)
    with open('output_sift.html', 'w') as f:
        f.write(ranking.to_html(index=False, escape=False))


if __name__ == "__main__":
    sift_ranking = get_ranking_from_desc('sift')