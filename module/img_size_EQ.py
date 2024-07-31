from PIL import Image

def resize_and_pad(input_images, size=(50,150)):
    resized_imgs = []
    for i in range(len(input_images)):
        image = input_images[i]
        # 원본 이미지의 비율 유지하며 리사이즈
        resized_img = image.resize(size, Image.Resampling.LANCZOS)
        resized_imgs.append(resized_img)
    return resized_imgs