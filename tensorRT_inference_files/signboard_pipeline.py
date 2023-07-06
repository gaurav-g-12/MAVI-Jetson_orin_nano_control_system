from DBNET import DBNET
from parseq import PARSEQ
import os
import cv2
import numpy as np
from PIL import Image
import time
import multiprocessing as mp

class init_signboard_pipeline:
    def __init__(self):
        self.save_dir = '/home/vision/MAVI/TensorRT/pipekine_output/dbnet'
        self.count = 0

        self.text_recognisition()
        self.text_detection()

    def text_detection(self):
        onnx_model_path = '/home/vision/MAVI/TensorRT/onnx_files/DBNET.onnx'
        tensorrt_engine_path = '/home/vision/MAVI/TensorRT/engines/DB_NET.engine'
        engine_precision = 'FP16'
        img_size = [3, 640, 640]
        batch_size = 1

        self.dbnet = DBNET(onnx_model_path, tensorrt_engine_path, engine_precision, img_size, batch_size)
        
    
    def text_recognisition(self):
        onnx_model_path='/home/vision/MAVI/TensorRT/onnx_files/parseqResnet50.onnx'
        tensorrt_engine_path='/home/vision/MAVI/TensorRT/engines/parseq_resnet50.engine'
        engine_precision='FP16'
        img_size=[3, 32, 128]
        batch_size=1
        
        self.parseq = PARSEQ(onnx_model_path, tensorrt_engine_path, engine_precision, img_size, batch_size)


    # this is a function to convert dbnet bbox preds to image crops
    def ploy_crop_to_rectangular_crop(self, box, image, dbnet):
        box = np.array(box).astype(np.int32).reshape(-1, 2)
        min_area_rect = cv2.minAreaRect(box)
        crop_img = dbnet.crop_minAreaRect(image, min_area_rect) 
        if crop_img.shape[0] > crop_img.shape[1]:
            crop_img = cv2.rotate(crop_img, cv2.ROTATE_90_CLOCKWISE)    
        frame_input = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_input).convert('RGB')

        return frame_pil
    

    def infer_one_frame(self, frame):
        
        self.count += 1
        image_name = str(self.count) + '.jpg'
        save_image_path = os.path.join(self.save_dir, image_name)

        # this list will store parseq prediction for all the crops in one frame
        txt_results = []
        
        sign_boxes, sign_scores = self.dbnet.infer_one_image(frame)

        self.dbnet.save_image_with_bbox(frame, sign_boxes, sign_scores, save_image_path)
        image = cv2.resize(frame, (640, 640))
        
        for i, box in enumerate(sign_boxes):
            frame_pil = self.ploy_crop_to_rectangular_crop(box, image, self.dbnet)

            pred, score = self.parseq.infer_one_image(frame_pil)
            # this will be a dict that will store pred and score for each crop
            crop_result = {'txt_pred': pred, 'score': score}
            txt_results.append(crop_result)

        return txt_results

    
    def start_inferenceing_txt(self, txt_frame_queue, txt_results_queue):   
        while(1):
            
            frame = txt_frame_queue.get()
            results = self.infer_one_frame(frame)
            txt_results_queue.put(results)