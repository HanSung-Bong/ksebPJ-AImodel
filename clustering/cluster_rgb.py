from sklearn.cluster import KMeans
import numpy as np
import cv2
import warnings
warnings.filterwarnings('ignore')

# 이미지 rgb 형식으로 리스트에 저장
def KMEANS_cls(input_images, k):
  images_arr=np.empty((0,150,50,3))
  for img in input_images:
    image_arr=np.array(img)
    images_arr=np.concatenate((images_arr, np.expand_dims(image_arr, axis=0)),axis=0)
  
  #저장된 4차원 리스트를 2차원으로 변경
  flattend_images = [img.reshape(1,-1) for img in images_arr]
  all_imgs = np.vstack(flattend_images)

  #Kmenas Clustering 적용
  kmeans = KMeans(n_clusters = k).fit(all_imgs)

  #결과 라벨을 반환
  team_ls = kmeans.labels_
  return team_ls


