
import argparse
#Parser 정의
parser = argparse.ArgumentParser(description="YOLOv8 Tracking")
parser.add_argument("--url", type=str, required=True, help="YouTube video URL to process")
parser.add_argument("--format", type=str, default="best", help="Video format (quality) to download")


import yt_dlp as youtube_dl
import ffmpeg
import numpy as np
import cv2
import asyncio
from queue import Queue
import nest_asyncio
from ultralytics import YOLO
import requests

from clustering.reference_cluster import process_team_clustering

# nest_asyncio 적용
nest_asyncio.apply()

def get_youtube_stream_url(url, format='best'):
    ydl_opts = {'format': format}
    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=False)
            video_url = info_dict['url']
        return video_url
    except Exception as e:
        notify_invalid_url(url, str(e))
        raise

def notify_invalid_url(url, error_message):
    server_url = 'http://example.com/notify_invalid_url'  # 서버 URL을 여기에 입력
    payload = {'url': url, 'error': error_message}
    try:
        response = requests.post(server_url, json=payload)
        response.raise_for_status()
        print("Invalid URL notification sent successfully")
    except requests.exceptions.RequestException as e:
        print(f"Failed to send invalid URL notification: {e}")

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

async def process_frames_with_yolo(tracking_yolo_model, ocr_yolo_model, frame_queue):
    frame_num = 0
    reference_centroids = None
    while True:
        frame = await frame_queue.get()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results = tracking_yolo_model.track(frame_bgr, imgsz=1920, tracker='bytetrack.yaml', persist=True, save=True)
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
            #print(cluster_result)
            index=0
            for cropped_img in cropped_imgs:
              ocr_results=ocr_yolo_model(cropped_img, imgsz=150)
              try:
                nums = ocr_results[0].boxes.cls.cpu().numpy()
                print(nums.shape)
                if nums.shape==(1,):
                  cluster_result[index].append(int(nums.item()))
                  index+=1
                elif nums.shape==(2,):
                  cluster_result[index].append(int(''.join(map(str,nums))))
                  index+=1
                else:
                  cluster_result[index].append(int(100))
                  index+=1
                  continue
              except:
                cluster_result[index].append(int(100))
                index+=1
            print(cluster_result)
        frame_num += 1
        print(frame_num)
        frame_queue.task_done()

async def main(youtube_url, tracking_yolo_model, ocr_yolo_model, width=1920, height=1080):
    frame_queue = asyncio.Queue(maxsize=10)
    try:
        stream_url = get_youtube_stream_url(youtube_url)
    except Exception as e:
        print(f"Failed to get stream URL: {e}")
        return

    stream_task = asyncio.create_task(stream_video_from_url(stream_url, frame_queue, width, height))
    process_task = asyncio.create_task(process_frames_with_yolo(tracking_yolo_model, ocr_yolo_model, frame_queue))

    await asyncio.gather(stream_task, process_task)

def run(args):
    tracking_yolo_model = YOLO('player_det_best_v1.pt')
    ocr_yolo_model = YOLO('ocr_best_8n_v1.pt')
    youtube_url = args.url
    asyncio.run(main(youtube_url, tracking_yolo_model,ocr_yolo_model))


if __name__=="__main__":
    opt = parser.parse_args() 
    run(opt)
