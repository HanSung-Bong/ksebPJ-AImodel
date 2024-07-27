import cv2
from clustering.crop_img import  crop_image_by_center
from clustering.cluster_rgb import KMEANS_cls
from clustering.img_size_EQ import resize_and_pad


def team_recognition(frame, tracking_result, size = (50,250), k=3):
    #선수 인식 데이터 불러오기 및 crop
    data_list = tracking_result.tolist()
    RGB_image=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cropped_imgs = crop_image_by_center(RGB_image, data_list)
    #print("Crop Finished")
    
    #이미지 크기 조정
    resized_imgs = resize_and_pad(cropped_imgs)
    #print("Resize Finished")
    
    # 이미지 클러스터링
    team_list = KMEANS_cls(resized_imgs, k)
    for i in range(len(team_list)):
        data_list[i].append(team_list[i])
    return cropped_imgs, data_list

        
    