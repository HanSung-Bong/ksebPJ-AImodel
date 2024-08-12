import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import numpy as np
import asyncio
import nest_asyncio
import aiohttp
import ffmpeg
import cv2
import argparse
import torch
from ultralytics import YOLO
from module.reference_cluster import process_team_clustering
from module.img_size_EQ import resize_and_pad
from module.only_matching import id_num_matching
from utube_run import get_youtube_stream_url, stream_video_from_url
from utils import ocr_preprocess, ocr_postprocess, load_engine, allocate_buffers

# CUDA 컨텍스트와 스트림 설정
def setup_cuda():
    cuda.init()
    device = cuda.Device(0)
    context = device.make_context()
    stream1 = cuda.Stream()
    stream2 = cuda.Stream()
    return context, stream1, stream2

def run_yolo_track(det_model, frame, stream):
    with torch.cuda.stream(stream):
        results=det_model.track(frame, imgsz=1920, tracker='bytetrack.yaml', persist=True)
        for track_result in results:
            cls = track_result.boxes.cls.int().cpu().numpy()    
            boxes = track_result.boxes.xywh.int().cpu().numpy()
            track_ids = track_result.boxes.id.int().cpu().numpy()
            cls_2d = np.reshape(cls, (-1, 1))
            boxes_2d = np.reshape(boxes, (-1, 4))
            track_ids_2d = np.reshape(track_ids, (-1, 1))
            track_results = np.hstack((cls_2d, boxes_2d, track_ids_2d))
    return track_results.tolist()
            
def run_engine(engine_context, inputs, outputs, bindings, stream):
    #input: host->device
    for inp, binding in zip(inputs, bindings[:len(inputs)]):
        cuda.memcpy_htod_async(binding, inp.host, stream)
    #infernce
    try:
        engine_context.execute_async_v3(stream_handle=stream.handle)
    except Exception as e:
        print(f"Erro Occured during inference: {e}")
    #ouput: device->host
    for out, binding in zip(outputs, bindings[len(inputs):]):
        cuda.memcpy_htod_async(out.host, binding, stream)
    stream.synchronize()
    return outputs


async def send_results_to_server(results):
    url = "http://example.com/api/upload"
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json={"results": results}) as response:
            return response.status

async def process_frame(det_model, ocr_engine, ocr_engine_context, stream1, stream2, frame_queue, result_queue):

    ocr_inputs, ocr_outputs, ocr_bindings=allocate_buffers(ocr_engine)
    match_dict={}
    # 각 텐서 주소 설정
    for i in range(ocr_engine.num_io_tensors):
        tensor_name = ocr_engine.get_tensor_name(i)
        ocr_engine.set_tensor_address(tensor_name, ocr_bindings[i])

    prev_clusters, prev_imgs = None, None, None
    while True:
        frame = await frame_queue.get()
        
        # 선수 추적 모델과 클러스터링 실행 (스트림 1)
        track_result= run_yolo_track(det_model, frame, stream1)
        cluster_result, reference_centroids, cropped_imgs = process_team_clustering( frame, track_result, reference_centroids)

        # 이전 프레임에 대해 A 모델 실행 (스트림 2)
        if prev_clusters:
            imgs = resize_and_pad(prev_imgs, size=(160,160))
            input_tensor=ocr_preprocess(imgs)
            ocr_inputs[0].host=input_tensor
            ocr_output = run_engine(ocr_engine_context, ocr_inputs, ocr_outputs, ocr_bindings, stream2)
            ocr_results=ocr_postprocess(ocr_output, len(imgs)) ##ocr_results = [[class_id, score]]
            final_result=id_num_matching(ocr_results, prev_clusters, match_dict)
            await result_queue.put(final_result)  # put_nowait 대신 put 사용
        
        # 다음 프레임으로 전달
        prev_clusters, prev_imgs =  cluster_result, cropped_imgs


async def main(youtube_url):
    nest_asyncio.apply() 
    context, stream1, stream2 = setup_cuda()
    frame_queue = asyncio.Queue(maxsize=10)
    result_queue = asyncio.Queue(maxsize=10)
    stream_url = get_youtube_stream_url(youtube_url)
    
    det_model=YOLO("det_v1.mdoel")
    ocr_engine=load_engine("ocr_v2.engine")
    ocr_engine_context=ocr_engine.create_execution_context()
    ocr_engine_context.set_input_shape("images", [16, 3, 160, 160])
    
    # 비디오 스트리밍 비동기 실행
    video_task = asyncio.create_task(stream_video_from_url(stream_url, frame_queue))

    # 프레임 처리 비동기 실행
    processing_task = asyncio.create_task(process_frame(det_model, ocr_engine, ocr_engine_context, stream1, stream2, frame_queue, result_queue))

    try:
        while True:
            # A 모델의 결과를 서버로 비동기 전송
            a_results = await result_queue.get()
            status = await send_results_to_server(a_results)
            print(f"A 모델 결과 서버 전송 상태 코드: {status}")

    except asyncio.CancelledError:
        # 작업 취소 시 리소스 해제
        video_task.cancel()
        processing_task.cancel()
        await video_task
        await processing_task

    finally:
        context.pop()
        context.detach()


parser = argparse.ArgumentParser(description="YOLOv8 Tracking")
parser.add_argument("--url", type=str, required=True, help="YouTube video URL to process")

if __name__ == "__main__":
    opt=parser.parese_args()
    asyncio.run(main())

