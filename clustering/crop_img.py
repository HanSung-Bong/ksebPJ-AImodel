from PIL import Image

def crop_image_by_center(input_image, data_list):
    # 이미지의 가로, 세로 길이를 가져옵니다    
    image=input_image
    img_width, img_height = image.size
    cropped_imgs = []

    for i in range(len(data_list)):
        if data_list[j][0]=='0':
            crop_width = float(data_list[j][1])* img_width
            crop_height = float(data_list[j][2])*img_height
            box_width = float(data_list[j][3])*img_width
            box_height = float(data_list[j][4])*img_height

            # 중심을 기준으로 자를 영역을 계산합니다
            left = crop_width - (box_width/2)
            upper = crop_height - (box_height/2)
            right = crop_width + (box_width/2)
            lower = crop_height + (box_height/2)
            crop_area = (left, upper, right, lower)
            
            
            # 자른 이미지를 리스트에 저장
            try:
                cropped_image = image.crop(crop_area)
                cropped_imgs.append(cropped_image)
            except:
                print(f"잘라내기를 실패했습니다: {crop_area}")
        else:
            continue
    return cropped_imgs
