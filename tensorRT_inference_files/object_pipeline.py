from rt_detr import RT_DETR
import os
import cv2
import numpy as np
from PIL import Image
import time
import multiprocessing as mp



class init_object_pipeline:

    def __init__(self):
        self.save_dir = '/home/vision/MAVI/TensorRT/pipekine_output/dino'
        self.count = 0

        self.object_detection()


    def object_detection(self):
        onnx_model_path="/home/vision/MAVI/TensorRT/onnx_files/rtdetr_r50_test_1.onnx"
        tensorrt_engine_path='/home/vision/MAVI/TensorRT/engines/rtdetr_19_op16.engine'
        engine_precision='FP16'
        img_size=[3, 640, 640]
        batch_size=1

        self.rt_detr = RT_DETR(onnx_model_path, tensorrt_engine_path, engine_precision, img_size, batch_size)


    def infer_one_frame(self, frame):

        self.count +=1
        image_name = str(self.count)+'.jpg'
        save_path = os.path.join(self.save_dir, image_name)
        
        results = {}
        obj_boxes, obj_scores, obj_labels = self.rt_detr.infer_one_image(frame)
        
        self.rt_detr.save_image_with_bbox(frame, obj_boxes, obj_scores, obj_labels, save_path)
        results = {'obj_boxes': obj_boxes, 'obj_lables': obj_labels, 'obj_scores': obj_scores}
        return results

    def start_inferenceing_obj(self, obj_frame_queue, obj_results_queue):

        while(1):

            frame = obj_frame_queue.get()
            results = self.infer_one_frame(frame)
            obj_results_queue.put(results)
            