
import pycuda.driver as cuda
from cuda import cudart
import tensorrt as trt
import pycuda.autoinit
import cv2
import numpy as np
import torch
import time


def check_device_data(binding, host_shape, dtype, stream):
    # 디바이스 메모리에서 데이터를 호스트로 복사하기 위한 버퍼 생성
    host_buffer = cuda.pagelocked_empty(host_shape, dtype)

    # 디바이스 메모리에서 호스트 메모리로 데이터 복사
    cuda.memcpy_dtoh_async(host_buffer, binding, stream)

    # 스트림 동기화 (모든 비동기 작업이 완료될 때까지 기다림)
    stream.synchronize()

    # 복사된 데이터를 반환
    return host_buffer

def clean_up():
    cuda.Context.pop()
    cuda.Context.get_current().detach()

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

def do_inference(engine, context, bindings, stream, inputs, outputs):
    # 입력 데이터를 디바이스로 복사
    for inp, binding in zip(inputs, bindings[:len(inputs)]):
        cuda.memcpy_htod_async(binding, inp.host, stream)
    #디바이스에 들어간 입력 데이터 다시 확인
    #device_data = check_device_data(bindings[0], input_tensors.shape, np.float32, stream)
    #print("Device Data:", device_data.shape)

    # 각 텐서 주소 설정
    for i in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(i)
        context.set_tensor_address(tensor_name, bindings[i])

    # 추론 실행
    try:
        start=time.time()
        context.execute_async_v3(stream_handle=stream.handle)
        end=time.time()
        print(end-start)
    except Exception as e:
        print(f"추론 실행 중 오류 발생: {e}")

    # 출력 데이터를 호스트로 복사
    for out, binding in zip(outputs, bindings[len(inputs):]):
        cuda.memcpy_dtoh_async(out.host, binding, stream)

    stream.synchronize()  # 스트림 동기화

    return outputs


cuda.init()
device = cuda.Device(0)
context1 = device.make_context() #CUDA context 생성
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
stream = cuda.Stream() #CUDA stream 생성

engine1 = load_engine("ocr_batch816_fp16.engine")
if engine1 is None:
  print("not loaded")

context1.push()
engine_context1 = engine1.create_execution_context() #trt context 생성

input_tensors=[]
for i in range(1,5):
  image_path = f'image/ex{i}.jpg'
  image = cv2.imread(image_path)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image = cv2.resize(image, (160, 160))
  image=np.array(image)/255.0
  image = np.transpose(image,(2, 0, 1)).astype(np.float32)
  input_tensors.append(image)

input_tensors=np.ascontiguousarray(input_tensors)

print("input_tensor_shape: ", input_tensors.shape)

# 사용할 최적화 프로파일 선택 (예: 첫 번째 프로파일)
profile_index = 0
engine_context1.set_optimization_profile_async(profile_index, stream.handle)


input_shape = [4, 3, 160, 160]
engine_context1.set_input_shape("images", input_shape)
print(engine_context1.get_tensor_shape("images"))

batch_size = 4
inputs, outputs, bindings = allocate_buffers(engine1, batch_size)
inputs[0].host = input_tensors


output = do_inference(engine1, engine_context1, bindings, stream, inputs, outputs)

#print("입력 호스트 데이터:", inputs[0].host[:10])
print("출력 호스트 데이터 (첫 10개 요소):", output[0].host[:10])  # 출력 데이터의 첫 10개 요소를 출력


context1.pop()
