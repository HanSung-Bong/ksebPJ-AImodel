import cv2
from ultralytics import YOLO
import numpy as np    

from module.reference_cluster import process_team_clustering
from module.ocrNmatching import id_num_matching

video_path='video_120.mp4'
tracking_yolo_model = YOLO('player_det_best_v1.pt')
ocr_yolo_model = YOLO('8x_100_best.pt')
cap=cv2.VideoCapture(video_path)

frame_num = 0
match_dic={}
reference_centroids = None

while cap.isOpened():   
    ret, frame = cap.read()
    if not ret:
        break
    
    results = tracking_yolo_model.track(frame, imgsz=1920, tracker='bytetrack.yaml', persist=True, save=True, show_conf=False)
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
        match_dic, ocr_results = id_num_matching(ocr_yolo_model, cluster_result, match_dic, cropped_imgs)
        final_result= [[int(item) for item in sublist] for sublist in ocr_results]
        print(frame_num) #나중에 ocr 까지 끝난 결과와 함께 전송
        print(final_result)
    frame_num += 1

cap.release()
