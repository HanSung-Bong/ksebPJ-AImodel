from ultralytics import YOLO
#from clustering.team_rec import team_recognition
from clustering.crop_img import  crop_image_by_center
from clustering.cluster_rgb import KMEANS_cls
from clustering.img_size_EQ import resize_and_pad


import yt_dlp as youtube_dl
import requests
import ffmpeg
import sys
import cv2
import numpy as np
import argparse

def get_youtube_stream_url(url, format='best'):
    ydl_opts = {'format': format}
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=False)
        video_url = info_dict['url']
    return video_url

# 프레임 단위로 영상 스트리밍 (ffmpeg 사용)
def stream_video_from_url(stream_url, width=1920, height=1080):
    process = (
        ffmpeg
        .input(stream_url)
        .filter('scale', width, height)  # scale 필터를 사용하여 크기 조정
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .run_async(pipe_stdout=True)
    )
    while True:
        in_bytes = process.stdout.read(width * height * 3)
        if not in_bytes:
            break
        frame = (
            np
            .frombuffer(in_bytes, np.uint8)
            .reshape([height, width, 3])
        )
        yield frame

#Parser 정의
parser = argparse.ArgumentParser(description="YOLOv8 Tracking")
parser.add_argument("--url", type=str, required=True, help="YouTube video URL to process")
parser.add_argument("--format", type=str, default="best", help="Video format (quality) to download")


def run(args):
    # Tracking 모델 로드
    tracking_model = YOLO("player_det_best_v1.pt")
    #OCR 모델 로드
    ocr_model=YOLO('ocr_best_8n_v1.pt')
    # 유튜브 URL 설정
    stream_url = get_youtube_stream_url(args.url, args.format)


    frame_num=1
    
    # 스트리밍 영상 처리 및 YOLOv8 추적
    for frame in stream_video_from_url(stream_url):
        track_result = tracking_model.track(frame, persist=True, tracker='bytetrack.yaml', save=True)
        #Tracking 결과값 np 배열로 변환
        track_results=np.array([[]])
        cls=track_result[0].boxes.cls.int().cpu().numpy()
        boxes=track_result[0].boxes.xywh.int().cpu().numpy()
        track_ids=track_result[0].boxes.id.int().cpu().numpy()
        cls_2d=np.reshape(cls, (-1,1))
        boxes_2d=np.reshape(boxes, (-1,4)) 
        track_ids_2d=np.reshape(track_ids, (-1,1))
        track_results=np.hstack((cls_2d, boxes_2d, track_ids_2d))
        
        ############################## ID 가 동일한 경우, OCR 과정은 반복해서 진행하지 않도록###############################
        ######################다만, 등번호가 인식되지 않았거나 Confidence가 일정이하 혹은 특정 조건인 경우######################
        ############################ 등번호를 재인식 하고, 기존의 결과와 비교 후 수정#########################################
        
        #선수 팀 클러스터링
        #선수 인식 데이터 불러오기 및 crop
        cluster_result = track_results.tolist() 
        RGB_image=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cropped_imgs = crop_image_by_center(RGB_image, cluster_result)
        #print("Crop Finished")
        #이미지 크기 조정
        resized_imgs = resize_and_pad(cropped_imgs)
        #print("Resize Finished")
        # 이미지 클러스터링
        k=3
        team_list = KMEANS_cls(resized_imgs, k)
        for i in range(len(team_list)):
            cluster_result[i].append(team_list[i])
        
        
        #result_history를 활용해서, 조건 추가.
        #OCR 결과 result list에 추가 및 POST
        iteration=0
        for crop_img in cropped_imgs:
            crop_img_arr=np.array(crop_img)
            ocr_results = ocr_model.predict(crop_img_arr)
            jersy_num_arr = ocr_results[0].boxes.cls.cpu().numpy()
            print(jersy_num_arr)
            if jersy_num_arr.shape==(2,):
                jersy_num=int(''.join(map(lambda x: str(int(x)), jersy_num_arr)))
            else:
                jersy_num=map(lambda x: int(x), jersy_num_arr) #숫자가 한개만 인식된 경우.(jersy_num_arr이 어떻게 출력되는지 확인)
            cluster_result[iteration].append(jersy_num) #Confidence값도 같이 저장해서, 등번호를 재인식할지 여부를 판단해야 됨.
            iteration+=1
        frame_num+=1
        final_result=cluster_result
        result_history=final_result
        
        #requests.get(URL, final_result)

if __name__=="__main__":
    opt = parser.parse_args() 
    run(opt)
