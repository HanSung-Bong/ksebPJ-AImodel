import cv2
from crop_img import  crop_image_by_center
from cluster_rgb import KMEANS_cls
from img_size_EQ import resize_and_pad


def team_recognition(input_img_path, tracking_result, size = (50,250), k=3):
    #선수 인식 데이터 불러오기 및 crop
    data_list = tracking_result.tolist()
    image = cv2.imread(input_img_path, cv2.IMREAD_COLOR)
    cropped_imgs = crop_image_by_center(image, data_list)
    print("Crop Finished")
    
    #이미지 크기 조정
    resized_imgs = resize_and_pad(cropped_imgs)
    print("Resize Finished")
    
    # 이미지 클러스터링
    team_list = KMEANS_cls(resized_imgs, k)
    for i in range(len(team_list)):
        data_list[i].append(team_list[i])
    return data_list

        
    