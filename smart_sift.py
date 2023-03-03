import cv2
import os
import pandas as pd

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

sift = cv2.xfeatures2d.SIFT_create()
bf = cv2.BFMatcher()

dataset_search_descriptors = []
for img in dataset_search:
    kp, des = sift.detectAndCompute(img['img'], None)
    dataset_search_descriptors.append({'path': img['path'], 'des': des})

dataset_index_descriptors = []
for img in dataset_index:
    kp, des = sift.detectAndCompute(img['img'], None)
    dataset_index_descriptors.append({'path': img['path'], 'des': des})

similarity_scores = []
dfs = []
for i, search_des in enumerate(dataset_search_descriptors):
    search_image_path = search_des['path']
    search_image_name = os.path.splitext(os.path.basename(search_image_path))[0]
    top_scores = []
    for j, index_des in enumerate(dataset_index_descriptors):
        index_image_path = index_des['path']
        index_image_name = os.path.splitext(os.path.basename(index_image_path))[0]
        matches = bf.knnMatch(search_des['des'], index_des['des'], k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        score = len(good_matches)
        top_scores.append({'Index Image Path': index_image_path, 'Similarity Score': score})
    top_scores_df = pd.DataFrame(top_scores).sort_values(by='Similarity Score', ascending=False).head(20)
    top_scores_df.insert(0, 'Search Image Path', search_image_path)
    top_scores_df.insert(2, 'Search Image', f'<img src="{search_image_path}">')
    top_scores_df.insert(3, 'Index Image', top_scores_df['Index Image Path'].apply(lambda x: f'<img src="{x}">'))
    top_scores_df = top_scores_df[['Search Image', 'Index Image', 'Similarity Score']]
    dfs.append(top_scores_df)

df = pd.concat(dfs, ignore_index=True)
# Write dataframe to html file
with open('output_smart_sift.html', 'w') as f:
    f.write(df.to_html(index=False, escape=False))