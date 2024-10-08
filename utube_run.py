import argparse
import yt_dlp 
import ffmpeg
import numpy as np
import cv2
import json
import asyncio
from queue import Queue
from ultralytics import YOLO
import requests
from PIL import Image
import torch    

#from module.save_list2txt import save_to_textfile
from module.ocrNmatching import id_num_matching
from module.reference_cluster import process_team_clustering


def get_youtube_stream_url(url):
    ydl_opts = {
        'format': 'bestvideo[height<=1080]',
        'quiet': True,
        'no_warnings': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        # 가장 높은 화질의 비디오 스트림 URL을 가져옴
        formats = info.get('formats', [info])
        best_format = max(formats, key=lambda f: f.get('height', 0) if f.get('height') is not None and f.get('height', 0) <= 1080 else 0)
        return best_format['url']


def notify_invalid_url(url, error_message):
    server_url = 'http://example.com/notify_invalid_url'  # 서버 URL을 여기에 입력
    payload = {'url': url, 'error': error_message}
    try:
        response = requests.post(server_url, json=payload)
        response.raise_for_status()
        print("Invalid URL notification sent successfully")
    except requests.exceptions.RequestException as e:
        print(f"Failed to send invalid URL notification: {e}")
        
def send_list_to_server(data_list):
    url = "https://808e-112-170-16-151.ngrok-free.app/api/receive_list"
    payload = {'data': data_list}
    print('1')
    try:
        # POST 요청 보내기
        response = requests.post(url, json=payload)
        print('2')
        response.raise_for_status()  # 오류가 발생하면 예외 발생
        print('3')
        # 서버 응답 처리
        print("Data sent successfully. Server responded with:", response.status_code)
        return response.json()  # 서버의 JSON 응답 반환
    except requests.exceptions.RequestException as e:
        print(f"Failed to send data: {e}")
        return None


async def stream_video_from_url(stream_url, frame_queue, width=1920, height=1080):
    process = (
        ffmpeg
        .input(stream_url)
        .filter('scale', width, height)
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
        await frame_queue.put(frame)

async def process_frames_with_yolo(tracking_yolo_model, ocr_yolo_model, frame_queue, save_path):
    frame_num = 0
    match_dic = {}
    reference_centroids = None
    while True:
        frame = await frame_queue.get()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results = tracking_yolo_model.track(frame_bgr, imgsz=1920, tracker='bytetrack.yaml', persist=True, show_conf=False, save=True)
        for track_result in results:
            track_results = np.array([[]])
            cls = track_result.boxes.cls.int().cpu().numpy()    
            boxes = track_result.boxes.xywh.int().cpu().numpy()
            track_ids = track_result.boxes.id.int().cpu().numpy()
            cls_2d = np.reshape(cls, (-1, 1))
            boxes_2d = np.reshape(boxes, (-1, 4))
            track_ids_2d = np.reshape(track_ids, (-1, 1))
            track_results = np.hstack((cls_2d, boxes_2d, track_ids_2d))
            cluster_result, reference_centroids, cropped_imgs = process_team_clustering(frame, track_results, reference_centroids)
            # print(cluster_result)
            match_dic, ocr_results = id_num_matching(ocr_yolo_model, cluster_result, match_dic, cropped_imgs)
            final_result_ls = [[int(item) for item in sublist] for sublist in ocr_results]
            final_result=[]
            for player_info in final_result_ls:
                player_info.append(frame_num)
                final_result.append(player_info)
            print(final_result)
            response = send_list_to_server(final_result)
            if response:    
                print("server response: ", response)   
            #save_to_textfile(final_result,f'txt/{frame_num:03d}.txt')
        frame_num += 1
        frame_queue.task_done()

async def main(youtube_url, tracking_yolo_model, ocr_yolo_model, save_path, width=1920, height=1080):
    frame_queue = asyncio.Queue(maxsize=10)
    try:
        stream_url = get_youtube_stream_url(youtube_url)
    except Exception as e:
        print(f"Failed to get stream URL: {e}")
        return

    stream_task = asyncio.create_task(stream_video_from_url(stream_url, frame_queue, width, height))
    process_task = asyncio.create_task(process_frames_with_yolo(tracking_yolo_model, ocr_yolo_model, frame_queue, save_path))

    await asyncio.gather(stream_task, process_task)

def run(args):
    tracking_yolo_model = YOLO('player_det_best_v1.pt').to('cuda')
    ocr_yolo_model = YOLO('8n_best_v2.pt').to('cuda')
    youtube_url = args.url
    save_path = args.path
    asyncio.run(main(youtube_url, tracking_yolo_model, ocr_yolo_model, save_path))

if __name__ == "__main__":
    opt = parser.parse_args()
    run(opt)
