import torch
import cv2
import onnx
import os
import time
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from torchvision import transforms as T
from shapely.geometry import Polygon
import tensorrt as trt
import pyclipper
from PIL import Image
import torchvision
import threading
from engine_builder import build_engine


class RT_DETR:
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
                

    def preprocess(self, image):
        img = cv2.resize(image, (640,640))
        img = img.astype(np.float32) / 255.0
        input_img = np.transpose(img, [2, 0, 1])
        image = input_img[np.newaxis, :, :, :]
        return image
    
    def postprocess(self, results):
        results = results[0]
        # bbox_pred=paddle.to_tensor(bbox_pred)
        # bbox_num=paddle.to_tensor(bbox_num)
        scores = results[:, 1]
        boxes = results[:, 2:]
        lables = results[:, 0]

        keep = results[:, 1]>0.6

        boxes = boxes[keep]
        scores = scores[keep]
        lables = lables[keep]

        return boxes, scores, lables
    

    def infer_one_image(self, image):
        threading.Thread.__init__(self)
        self.cfx.push()

        x = time.time()
        image = self.preprocess(image)
        image = np.asarray(image, dtype=np.float32)
        np.copyto(self.host_inputs[0], image.ravel())
        y = time.time()
        # print('dter_pre', y-x)

        x = time.time()
        cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)
        self.context.execute_async(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.host_outputs[0], self.cuda_outputs[0], self.stream)
        cuda.memcpy_dtoh_async(self.host_outputs[1], self.cuda_outputs[1], self.stream)
        self.stream.synchronize()
        y = time.time()
        # print('dter_model', y-x)


        x = time.time()
        results = torch.Tensor(self.host_outputs[1]).reshape(1, -1, 6)
        boxes, scores, labels = self.postprocess(results)
        y = time.time()
        # print('dter_model', y-x)
        
        self.cfx.pop()
        return boxes, scores, labels
    

    def save_image_with_bbox(self, image, pred_box, scores, labels, save_path):
        label_background_color = (125, 175, 75)
        label_text_color = (255, 255, 255) 
        LABELS_dict = {'0':'person', '1000': 'signboard', '19': 'cow', '15': 'dog'}
        COLOR_dict = {'0':(0,255,0), '100': (255,0,0), '19': (0,0,255), '15': (255,255,0)}

        image = cv2.resize(image, (640,640))

        for box, label, score in zip(pred_box, labels, scores):
            label = str(int(label))
            confidence_score = int(score*100)

            if label=='0' or label=='100' or label=='19' or label=='15':
                box[0] = (int(box[0]))
                box[1] = (int(box[1]))
                box[2] = (int(box[2]))
                box[3] = (int(box[3]))

                image = cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), COLOR_dict[label], 2)

                label_text = LABELS_dict[label] + '(' + str(confidence_score) + '%)'
                label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                label_left = box[0]
                label_top = box[1] - label_size[1]
                if (label_top < 1):
                    label_top = 1
                label_right = label_left + label_size[0]
                label_bottom = label_top + label_size[1]

                cv2.rectangle(image, (int(label_left - 1), int(label_top - 1)), (int(label_right + 1), int(label_bottom + 1)), label_background_color, -1)
                cv2.putText(image, label_text, (int(label_left), int(label_bottom)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text_color, 1)
        
        cv2.imwrite(save_path, image)



def main():
    
    onnx_model_path="/home/vision/MAVI/TensorRT/onnx_files/rtdetr_r50_test_1.onnx"
    tensorrt_engine_path='/home/gaurav_t/scratch/TensorRT/engines/rtdetr_19_op16.engine'
    engine_precision='FP16'
    img_size=[3, 640, 640]
    batch_size=1
    image_dir = '/home/gaurav_t/scratch/TensorRT/dataset/object detection'
    save_dir = '/home/vision/MAVI/TensorRT/dataset/output/onj/'

    rt_detr = RT_DETR(onnx_model_path, tensorrt_engine_path, engine_precision, img_size, batch_size)

    for image_name in sorted(os.listdir(image_dir)):
        image_path = image_dir + image_name
        save_path = save_dir + image_name
        
        image = cv2.imread(image_path)
        boxes, scores, labels = rt_detr.infer_one_image(image)

        rt_detr.save_image_with_bbox(image, boxes, scores, labels, save_path)       


if __name__ == '__main__':
    main()