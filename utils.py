import pycuda.driver as cuda
from cuda import cudart
import tensorrt as trt
import cv2
import numpy as np
import torch
from types import SimpleNamespace

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def load_engine(engine_file_path):
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
          print("load engine failed")
    return engine

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []

    for i in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(i)
        size = trt.volume(engine.get_tensor_shape(tensor_name))
        dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))

        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        bindings.append(int(device_mem))

        if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
        #print(f"Tensor Name: {tensor_name}, Size: {size}, Dtype: {dtype}")
    return inputs, outputs, bindings


def ocr_preprocess(images):
    input_tensors=[]
    for image in images:
        image=np.array(image)/255.0
        image = np.transpose(image,(2, 0, 1)).astype(np.float32)
        input_tensors.append(image)
    input_tensors=np.ascontiguousarray(input_tensors)
    return input_tensors

def ocr_postprocess(ocr_results, batch, prev_track, prev_cluster):
    outputs = ocr_results[0].host.reshape(batch,14,525).transpose(0,2,1)
    results=[]
    for output in outputs:
        rows = output.shape[0]
        boxes = []
        scores = []
        class_ids=[]
        result={"cls": [], "boxes": [], "conf" : []}
        # Iterate through output to collect bounding boxes, confidence scores, and class IDs
        for i in range(rows):
            classes_scores = output[i][4:]
            max_score=np.amax(classes_scores)
            if max_score >= 0.3:
                box = [
                    output[i][0] - (0.5 * output[i][2]),
                    output[i][1] - (0.5 * output[i][3]),
                    output[i][2],
                    output[i][3],
                ]
                boxes.append(box)
                scores.append(max_score)
                class_ids.append(np.argmax(classes_scores))
        # Apply NMS (Non-maximum suppression)
        result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.3, 0.45)
        for index in result_boxes:
            result["cls"].append(class_ids[index])
            result["boxes"].append([boxes[index]])
            result["conf"].append(scores[index])
        results.append(result)
    results=SimpleNamespace(**results)
    return results