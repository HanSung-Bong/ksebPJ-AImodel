import cv2
import numpy as np
# import requests
from functools import partial
import torch

from boxmot.utils.checks import RequirementsChecker
from boxmot.detectors import get_yolo_inferer

from parsing import parse_opt
from init_tracker import on_predict_start
# from clustering.team_rec import team_recognition

checker = RequirementsChecker()
checker.check_packages(('ultralytics @ git+https://github.com/mikel-brostrom/ultralytics.git', ))  # install

from ultralytics import YOLO


@torch.no_grad()
def run(args):

    yolo = YOLO(
        args.yolo_model if 'yolov8' in str(args.yolo_model) else 'yolov8n.pt',
    )

    results = yolo.track(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        agnostic_nms=args.agnostic_nms,
        show=False,
        stream=True,
        device=args.device,
        show_conf=args.show_conf,
        save_txt=args.save_txt,
        show_labels=args.show_labels,
        save=args.save,
        verbose=args.verbose,
        exist_ok=args.exist_ok,
        project=args.project,
        name=args.name,
        classes=args.classes,
        imgsz=args.imgsz,
        vid_stride=args.vid_stride,
        line_width=args.line_width
    )

    yolo.add_callback('on_predict_start', partial(on_predict_start, persist=True))

    if 'yolov8' not in str(args.yolo_model):
        # replace yolov8 model
        m = get_yolo_inferer(args.yolo_model)
        model = m(
            model=args.yolo_model,
            device=yolo.predictor.device,
            args=yolo.predictor.args
        )
        yolo.predictor.model = model

    # store custom args in predictor
    yolo.predictor.custom_args = args
    
    for r in results:
        img = yolo.predictor.trackers[0].plot_results(r.orig_img, args.show_trajectories)
        if args.show is True:   
            cv2.imshow('BoxMOT', img)     
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') or key == ord('q'):
                break
    

    for result in results:
        frame_results=np.array([[]])
        cls=result.boxes.cls.numpy()
        boxes=result.boxes.xywh.numpy()
        track_ids=result.boxes.id.numpy()
        
        cls_2d=np.reshape(cls, (-1,1))
        boxes_2d=np.reshape(boxes, (-1,4))
        track_ids_2d=np.reshape(track_ids, (-1,1))
        frame_results=np.hstack((cls_2d, boxes_2d, track_ids_2d))
        # requests.post("http://localhost:9999", body={"frame_results" : frame_results}, headers={"Authorization" : "Bearer askdfj;aksj;flkajs;dkfj;asd"})
        input_frame_path = #영상에서 프레임 추출 혹은 영상으로 변환 전 프레임 PATH
        cluster_results=team_recognition(input_frame_path, frame_results, k=3)

    ###plotting results###




if __name__ == "__main__":
    opt = parse_opt()
    run(opt)
