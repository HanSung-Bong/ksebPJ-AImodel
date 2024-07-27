from collections import defaultdict
from ultralytics import YOLO
from clustering.team_rec import team_recognition

import yt_dlp as youtube_dl

import ffmpeg
import sys
import cv2
import numpy as np
import argparse

def get_youtube_stream_url(url):
    ydl_opts = {'format': 'best'}
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=False)
        video_url = info_dict['url']
    return video_url

# 프레임 단위로 영상 스트리밍 
def stream_video_from_url(stream_url):
    process = (
        ffmpeg
        .input(stream_url)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .run_async(pipe_stdout=True)
    )
    width, height = 640, 360  # 원하는 해상도로 설정
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
parser.add_argument("--url", type=str, required=True, help="Youtube video URL to process.")


def run(arg):
    # Tracking 모델 로드
    tracking_model = YOLO("player_det_best_v1.pt")

    # 유튜브 URL 설정
    url = arg.url
    stream_url = get_youtube_stream_url(url)

    frame=1
    # 스트리밍 영상 처리 및 YOLOv8 추적
    for frame in stream_video_from_url(stream_url):
        results = tracking_model.track(frame, persist=True, tracker='bytetrack.yaml')
        #Tracking 결과값 np 배열로 변환
        track_results=np.array([[]])
        cls=results[0].boxes.cls.numpy()
        boxes=results[0].boxes.xywh.numpy()
        track_ids=results[0].boxes.id.numpy()
        cls_2d=np.reshape(cls, (-1,1))
        boxes_2d=np.reshape(boxes, (-1,4)) 
        track_ids_2d=np.reshape(track_ids, (-1,1))
        track_results=np.hstack((cls_2d, boxes_2d, track_ids_2d))
        
        
        ############################## ID 가 동일한 경우, 아래의 과정은 반복해서 진행하지 않도록###############################
        ######################다만, 등번호가 인식되지 않았거나 Confidence가 일정이하 혹은 특정 조건인 경우######################
        ############################ 등번호를 재인식 하고, 기존의 결과와 비교 후 수정#########################################
        
    
        #선수 팀 클러스터링
        cropped_imgs, cluster_results=team_recognition(frame, track_results, k=3)    
        print(cluster_results)
        #OCR 모델 로드
        ocr_model=YOLO('ocr_best_8n.pt')
        #OCR 결과 result list에 추가 및 POST
        for crop_img in cropped_imgs:
            crop_img_arr=np.array(crop_img)
            ocr_results = ocr_model.predict(crop_img_arr)
            for ocr_result in ocr_results:
                jersy_num_arr = ocr_result[0].boxes.cls.numpy()
                if jersy_num_arr
        frame+=1


if __name__=="__main__":
    opt = parser.parse_args()
    run(opt)
