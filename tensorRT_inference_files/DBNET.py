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


class DBNET:
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

        build_engine(onnx_model_path, tensorrt_engine_path, engine_precision, img_size, batch_size)

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


    # pre-processing
    def preprocess(self, image):
        if image is None:
            raise Exception('Image not found!')
        
        frame = cv2.resize(image, (640, 640)).astype('float32')  

        RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793])
        frame -= RGB_MEAN
        frame /= 255.
        frame = frame.transpose(2,0,1)
        frame = np.expand_dims(frame, axis=0)
        return frame

    # post-processing
    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int32), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int32), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int32), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def unclip(self, box, unclip_ratio=1.5):
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
            bounding_box = cv2.minAreaRect(contour)
            points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

            index_1, index_2, index_3, index_4 = 0, 1, 2, 3
            if points[1][1] > points[0][1]:
                index_1 = 0
                index_4 = 1
            else:
                index_1 = 1
                index_4 = 0
            if points[3][1] > points[2][1]:
                index_2 = 2
                index_3 = 3
            else:
                index_2 = 3
                index_3 = 2

            box = [points[index_1], points[index_2],
                points[index_3], points[index_4]]
            return box, min(bounding_box[1])

    def postprocess(self, pred, _bitmap, dest_width, dest_height):

            bitmap = _bitmap[0]  
            pred = pred[0]

            height, width = bitmap.shape
            boxes = []
            scores = []

            contours, _ = cv2.findContours(
                (bitmap * 255).astype(np.uint8),
                cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
            for contour in contours[:1000]:
                epsilon = 0.002 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                points = approx.reshape((-1, 2))
                
                if points.shape[0] < 4:
                    continue
                score = self.box_score_fast(pred, points.reshape(-1, 2))

                if 0.7 > score:
                    continue

                if points.shape[0] > 2:
                    box = self.unclip(points, unclip_ratio=2.0)
                    if len(box) > 1:
                        continue
                else:
                    continue
                box = box.reshape(-1, 2)
            
                if box.shape[0] <=1 :
                    continue
            
                min_size = 0
                _, sside = self.get_mini_boxes(box.reshape((-1, 1, 2)))

            
                if sside < min_size + 2:
                    continue

                if not isinstance(dest_width, int):
                    dest_width = dest_width.item()
                    dest_height = dest_height.item()

                box[:, 0] = np.clip(
                    np.round(box[:, 0] / width * dest_width), 0, dest_width)
                box[:, 1] = np.clip(
                    np.round(box[:, 1] / height * dest_height), 0, dest_height)

                boxes.append(box.tolist())
                scores.append(score)
                
            return boxes, scores

    def crop_minAreaRect(self, img, rect):
        angle = rect[2]
        rows, cols = img.shape[0], img.shape[1]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        img_rot = cv2.warpAffine(img, M, (cols, rows))
        # rotate bounding box
        rect0 = (rect[0], rect[1], 0.0)
        box = cv2.boxPoints(rect)
        pts = np.int0(cv2.transform(np.array([box]), M))[0]
        pts[pts < 0] = 0
        # crop
        img_crop = img_rot[pts[1][1]:pts[0][1],
                    pts[1][0]:pts[2][0]]
        return img_crop

    def infer_one_image(self, image):
        threading.Thread.__init__(self)
        self.cfx.push()

        x = time.time()
        image = self.preprocess(image)
        image = np.asarray(image, dtype=np.float32)
        np.copyto(self.host_inputs[0], image.ravel())
        y = time.time()
        # print('dbnet_pre', y-x)
        
        x = time.time()
        cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)
        self.context.execute_async(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.host_outputs[0], self.cuda_outputs[0], self.stream)
        self.stream.synchronize()
        y = time.time()
        # print('dbnet+_model', y-x)

        x = time.time()
        output_data = torch.Tensor(self.host_outputs[0]).reshape(self.engine.max_batch_size, 640, 640)
        
        one_image_output = output_data.numpy()
        _, height, width = one_image_output.shape
        segmentation = one_image_output > 0.3
        boxes, score = self.postprocess(one_image_output, segmentation, height, width)
        y = time.time()
        # print('dbnet_post', y-x)

        self.cfx.pop()
        return boxes, score

    def destory(self):
        self.cfx.pop()


    def save_image_with_bbox(self, image, boxes, scores, save_path):
        
        image = cv2.resize(image, (640, 640))
        
        for box, score in zip(boxes, scores):
            confidence_score = score*100

            box = np.array(box, np.int32)
            box = box.reshape((-1,1,2))
            image = cv2.polylines(image, [box], isClosed=True, color=(0,255,0), thickness=2)
        
        cv2.imwrite(save_path, image)

def main():

    onnx_model_path = '/home/vision/MAVI/TensorRT/onnx_files/DBNET.onnx'
    tensorrt_engine_path = '/home/vision/MAVI/TensorRT/engines/DB_NET.engine'
    engine_precision = 'FP16'
    img_size = [3, 640, 640]
    batch_size = 1
    image_dir = '/home/vision/MAVI/TensorRT/dataset/signboard/28-01-2023'
    save_dir = '/home/vision/MAVI/TensorRT/dataset/output/sign/'
    
    dbnet = DBNET(onnx_model_path, tensorrt_engine_path, engine_precision, img_size, batch_size)

    for image_name in os.listdir(image_dir):

        image_path = os.path.join(image_dir, image_name)
        save_path = os.path.join(save_dir, image_name)

        # x = time.time()
        image = cv2.imread(image_path)
        boxes, scores = dbnet.infer_one_image(image)
        # y = time.time()
        # print(y-x)    

        dbnet.save_image_with_bbox(image, boxes, scores, save_path)
       

if __name__ == '__main__':
    main()