from sklearn.cluster import KMeans
import numpy as np
import cv2

# 이미지 rgb 형식으로 리스트에 저장
def KMEANS_cls(input_images, k)
  imgs = []
  images=input_images
  for i in range(len(images)):
    image = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
    if image is not None:
        imgs.append(image)
    else:
        print("Failed to load image(rgb)")

  #저장된 4차원 리스트를 2차원으로 변경
  flattend_images = [img.reshape(1,-1) for img in imgs]
  all_imgs = np.vstack(flattend_images)

  #Kmenas Clustering 적용
  kmeans = KMeans(n_clusters = k).fit(all_imgs)

  #결과 라벨을 반환
  team_ls = kmeans.labels_
  return team_ls


