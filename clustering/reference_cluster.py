from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import numpy as np
import cv2
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from clustering.crop_img import crop_image_by_center
from clustering.img_size_EQ import resize_and_pad

def process_team_clustering(frame, track_results, reference_centroids):
    cluster_result = track_results.tolist()
    RGB_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cropped_imgs = crop_image_by_center(RGB_image, cluster_result)
    resized_imgs = resize_and_pad(cropped_imgs)
    
    images_arr = np.empty((0, 150, 50, 3))
    for img in resized_imgs:
        #display(img)
        image_arr = np.array(img)
        images_arr = np.concatenate((images_arr, np.expand_dims(image_arr, axis=0)), axis=0)
    
    flat_imgs = [img.flatten() for img in images_arr]
    kmeans = KMeans(n_clusters=3, random_state=0).fit(flat_imgs)
    
    if reference_centroids is None:
        reference_centroids = kmeans.cluster_centers_
        team_list = kmeans.labels_
    else:
        team_list = remap_labels(kmeans, flat_imgs, reference_centroids)
    
    for i in range(len(team_list)):
        cluster_result[i].append(team_list[i])
    
    return cluster_result, reference_centroids, cropped_imgs

def remap_labels(kmeans, X, reference_centroids):
    distances = cdist(X, reference_centroids)
    new_labels = np.argmin(distances, axis=1)
    return new_labels



