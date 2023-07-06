from DBNET import DBNET
from parseq import PARSEQ
from DINO import DINO
import os
import cv2
import numpy as np
from PIL import Image
import time
import multiprocessing as mp

class initialize_models:
    def __init__(self):
        pass
    
    def signboard_detection(self):
        onnx_model_path = '/home/gaurav_t/scratch/TensorRT/onnx_files/DBNET.onnx'
        tensorrt_engine_path = '/home/gaurav_t/scratch/TensorRT/engines/DB_NET.engine'
        engine_precision = 'FP16'
        img_size = [3, 640, 640]
        batch_size = 1
        
        dbnet = DBNET(onnx_model_path, tensorrt_engine_path, engine_precision, img_size, batch_size)
        return dbnet
    
    def signboard_recognisition(self):
        onnx_model_path='/home/gaurav_t/scratch/TensorRT/onnx_files/parseqResnet50.onnx'
        tensorrt_engine_path='/home/gaurav_t/scratch/TensorRT/engines/parseq_resnet50.engine'
        engine_precision='FP16'
        img_size=[3, 32, 128]
        batch_size=1
        
        parseq = PARSEQ(onnx_model_path, tensorrt_engine_path, engine_precision, img_size, batch_size)
        return parseq

    def object_detection(self):
        onnx_model_path="/home/gaurav_t/scratch/TensorRT/onnx_files/DINO_base_800_800.onnx"
        tensorrt_engine_path='/home/gaurav_t/scratch/TensorRT/engines/DINO_800_800.engine'
        engine_precision='FP16'
        img_size=[3, 800, 800]
        batch_size=1
        
        dino = DINO(onnx_model_path, tensorrt_engine_path, engine_precision, img_size, batch_size)
        return dino
    
    def ploy_crop_to_rectangular_crop(self, box, image, dbnet):
        box = np.array(box).astype(np.int32).reshape(-1, 2)
        min_area_rect = cv2.minAreaRect(box)
        crop_img = dbnet.crop_minAreaRect(image, min_area_rect) 
        if crop_img.shape[0] > crop_img.shape[1]:
            crop_img = cv2.rotate(crop_img, cv2.ROTATE_90_CLOCKWISE)    
        frame_input = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_input).convert('RGB')

        return frame_pil

    
def main():

    model_init = initialize_models()
    dbnet = model_init.signboard_detection()
    parseq = model_init.signboard_recognisition()
    dino = model_init.object_detection()

    image_dir = '/home/gaurav_t/scratch/TensorRT/dataset/signboard/28-01-2023'
    signboard_save_dir = '/home/gaurav_t/scratch/TensorRT/dataset/output/sign'
    object_save_dir = '/home/gaurav_t/scratch/TensorRT/dataset/output/obj'
    signboard_reco_save_dir = '/home/gaurav_t/scratch/TensorRT/dataset/output/sign_reco'

    if not os.path.exists(signboard_save_dir):
        os.mkdir(signboard_save_dir)

    if not os.path.exists(object_save_dir):
        os.mkdir(object_save_dir)
    
    if not os.path.exists(signboard_reco_save_dir):
        os.mkdir(signboard_reco_save_dir)

     
    p1 = mp.Process(target=dbnet.infer_one_image)
    p2 = mp.Process(target=dino.infer_one_image)
    p1.start()
    p2.start()

    for image_name in sorted(os.listdir(image_dir)):
        print(f'{image_name} pred are : ', end='\t')
       
        image_path = os.path.join(image_dir, image_name)
        sign_save_path = os.path.join(signboard_save_dir, image_name)
        obj_save_path = os.path.join(object_save_dir, image_name)
        txt_file_name = image_name[:-4]+'.txt'
        sign_reco_save_path = os.path.join(signboard_reco_save_dir, txt_file_name)

        sign_boxes, sign_scores = dbnet.infer_one_image(image_path)

        obj_boxes, obj_scores, obj_labels = dino.infer_one_image(image_path)

        image = cv2.imread(image_path)
        image = cv2.resize(image, (640, 640))

        pred_file = open(sign_reco_save_path, 'w')

        for i, box in enumerate(sign_boxes):
            frame_pil = model_init.ploy_crop_to_rectangular_crop(box, image, dbnet)

            pred, score = parseq.infer_one_image(frame_pil)
            
            if score > 0.7:
                pred_file.write(pred + '\t' + str(float(score*100)) + '\n')

        dbnet.save_image_with_bbox(image_path, sign_boxes, sign_scores, sign_save_path)
        dino.save_image_with_bbox(image_path, obj_boxes, obj_scores, obj_labels, obj_save_path)
        pred_file.close()

    

if __name__ == '__main__':
    main()