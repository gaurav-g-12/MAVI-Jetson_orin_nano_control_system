import os
import cv2
import time
import torch
import onnx
import pyclipper
import numpy as np
from PIL import Image
import pycuda.autoinit
import tensorrt as trt
import threading
from typing import Tuple
from engine_builder import build_engine
import pycuda.driver as cuda
from shapely.geometry import Polygon
from torchvision.transforms import transforms


class PARSEQ:
    def __init__(self, onnx_model_path, tensorrt_engine_path, engine_precision, img_size, batch_size):
        self.onnx_model_path = onnx_model_path
        self.tensorrt_engine_path = tensorrt_engine_path
        self.engine_precision = engine_precision
        self.img_size = img_size
        self.batch_size = batch_size

        self.cfx = cuda.Device(0).make_context()
        self.stream = cuda.Stream()


        self.TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(self.TRT_LOGGER, '')
        runtime = trt.Runtime(self.TRT_LOGGER)

        # build_engine(onnx_model_path, tensorrt_engine_path, engine_precision, img_size, batch_size)
        x = time.time()
        with open(self.tensorrt_engine_path, "rb") as f: 
            buf = f.read()
            self.engine = runtime.deserialize_cuda_engine(buf)    
        self.context = self.engine.create_execution_context()

        self.host_inputs  = []
        self.cuda_inputs  = []
        self.host_outputs = []
        self.cuda_outputs = []
        self.bindings = []

        for binding in self.engine:
            shape = self.engine.get_binding_shape(binding)
            size = trt.volume(shape) * self.engine.max_batch_size
            host_mem = cuda.pagelocked_empty(size, np.float32)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(cuda_mem))
            if self.engine.binding_is_input(binding):
                self.host_inputs.append(host_mem)
                self.cuda_inputs.append(cuda_mem)
            else:
                self.host_outputs.append(host_mem)
                self.cuda_outputs.append(cuda_mem)
        y = time.time()
        # print('paseq_in'*100, y-x)

    # pre-processing
    def get_transform(self, img_size: Tuple[int], augment: bool = False, rotation: int = 0):
    
        transforms_list = []
        transforms_list.extend([
        transforms.Resize(img_size, transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
        ])
        return transforms.Compose(transforms_list)

    def preprocess(self, image):
        image = self.get_transform((32,128))(image).unsqueeze(0).numpy()
        return image


    # post-processing
    def postprocess(self, result):
        output = torch.tensor(result)
        output = output.float()
        output = output.softmax(-1)
        
        res = output.topk(1, dim=-1)[1]
        result = []
        for val in res.squeeze(-1)[0]:
            if val == 0:
                break
            result.append(val.item())
        charset = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
        results = ''.join([charset[i-1] for i in result])
        score = 1
        output = output.max(-1)
        for i in range(len(results)):
            score *= output[0][0][i]
            
        return results, score



    def infer_one_image(self, image):
        threading.Thread.__init__(self)
        self.cfx.push()
        
        x = time.time()
        image = self.preprocess(image)
        image = np.asarray(image, dtype=np.float32)
        np.copyto(self.host_inputs[0], image.ravel())
        y = time.time()
        # print('parseq_pre', y-x)
        
        x = time.time()
        cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)
        self.context.execute_async(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.host_outputs[0], self.cuda_outputs[0], self.stream)
        self.stream.synchronize()
        y = time.time()
        # print('parseq_model', y-x)
        
        x = time.time()
        output_data = torch.Tensor(self.host_outputs[0]).reshape(self.engine.max_batch_size, 26, 95)
        pred, score = self.postprocess(output_data)
        y = time.time()
        # print('parseq_model', y-x)
        
        self.cfx.pop()
        return pred, score
    
    def destory(self):
        self.cfx.pop()


def main():

    onnx_model_path='/home/vision/MAVI/TensorRT/onnx_files/parseqResnet50.onnx'
    tensorrt_engine_path='/home/vision/MAVI/TensorRT/engines/parseq_resnet50.engine'
    engine_precision='FP16' 
    img_size=[3, 32, 128]
    batch_size=1
    image_dir = '/home/vision/MAVI/TensorRT/dataset/crops/'
    
    parseq = PARSEQ(onnx_model_path, tensorrt_engine_path, engine_precision, img_size, batch_size)

    for image_name in os.listdir(image_dir):
        
        image_path = os.path.join(image_dir, image_name)
        image = Image.open(image_path).convert("RGB")

        # x = time.time()
        pred, score = parseq.infer_one_image(image)
        # y = time.time()
        # print(y-x)


if __name__ == '__main__':
    main()