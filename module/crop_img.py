from PIL import Image
import cv2

def crop_image_by_center(input_image, data_list):
    # 이미지의 가로, 세로 길이를 가져옵니다    
    image=input_image
    img=Image.fromarray(image)
    cropped_imgs = []

    for i in range(len(data_list)):
        if data_list[i][0]==0:
            crop_width = data_list[i][1]
            crop_height = data_list[i][2]
            box_width = data_list[i][3]
            box_height = data_list[i][4]

            # 중심을 기준으로 자를 영역을 계산합니다
            left = crop_width - (box_width/2)
            upper = crop_height - (box_height/2)
            right = crop_width + (box_width/2)
            lower = crop_height + (box_height/2)
            crop_area = (left, upper, right, lower)
            
            # 자른 이미지를 리스트에 저장
            try:
                cropped_image = img.crop(crop_area)
                cropped_imgs.append(cropped_image)
            except:
                print(f"잘라내기를 실패했습니다: {crop_area}")
        else:
            continue
    
    return cropped_imgs
