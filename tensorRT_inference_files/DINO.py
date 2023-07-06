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


class DINO:
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
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        t = T.Compose([
                    torchvision.transforms.Resize((800,800)),
                    normalize,
                ])
        img = Image.fromarray(image).convert("RGB")
        batch_image = t(img).unsqueeze(0)
        return batch_image
    
    def postprocess(self, pred_logits, pred_boxes):
        prob=pred_logits.sigmoid()

        topk_values, topk_indexes=torch.topk(prob.view(pred_boxes.shape[0],-1),10,1)

        scores=topk_values
        topk_boxes=torch.div(topk_indexes, pred_logits.shape[2], rounding_mode='floor')
        labels=topk_indexes % pred_logits.shape[2]
        boxes=torch.gather(pred_boxes,1,topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        cx,cy,w,h=boxes.unbind(-1)
        new_bbox = [(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)]
        pred_box=torch.stack(new_bbox,dim=-1)

        pred_box[:,0::2]*=800
        pred_box[:,1::2]*=800

        keep = scores>0.42
        pred_box = pred_box[keep]
        scores = scores[keep]
        labels = labels[keep]

        return pred_box, scores, labels
    

    def infer_one_image(self, image):
        threading.Thread.__init__(self)
        self.cfx.push()

        image = self.preprocess(image)
        image = np.asarray(image, dtype=np.float32)
        np.copyto(self.host_inputs[0], image.ravel())

        x = time.time()
        cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)
        self.context.execute_async(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.host_outputs[0], self.cuda_outputs[0], self.stream)
        cuda.memcpy_dtoh_async(self.host_outputs[1], self.cuda_outputs[1], self.stream)
        self.stream.synchronize()
        y = time.time()
        
        pred_logits = torch.Tensor(self.host_outputs[0]).reshape(self.engine.max_batch_size, 900, -1)
        pred_boxes = torch.Tensor(self.host_outputs[1]).reshape(self.engine.max_batch_size, 900, -1)

        pred_box, scores, labels = self.postprocess(pred_logits, pred_boxes)
        
        self.cfx.pop()
        return pred_box, scores, labels
    

    def save_image_with_bbox(self, image, pred_box, scores, labels, save_path):
        label_background_color = (125, 175, 75)
        label_text_color = (255, 255, 255) 
        LABELS_dict = {'1':'person', '12': 'signboard', '21': 'cow', '18': 'dog'}
        COLOR_dict = {'1':(0,255,0), '12': (255,0,0), '21': (0,0,255), '18': (255,255,0)}

        image = cv2.resize(image, (640,480))

        for box, label, score in zip(pred_box, labels, scores):
            label = str(int(label))
            confidence_score = int(score*100)

            if label=='1' or label=='12' or label=='21' or label=='18':
                box[0] = int(int(box[0]) * 640/800)
                box[1] = int(int(box[1]) * 480/800)
                box[2] = int(int(box[2]) * 640/800)
                box[3] = int(int(box[3]) * 480/800)

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
    
    onnx_model_path="/home/vision/MAVI/TensorRT/onnx_files/DINO_base_800_800.onnx"
    tensorrt_engine_path='/home/gaurav_t/scratch/TensorRT/engines/DINO_800_800.engine'
    engine_precision='FP16'
    img_size=[3, 800, 800]
    batch_size=1
    image_dir = '/home/vision/MAVI/TensorRT/dataset/signboard/28-01-2023/'
    save_dir = '/home/vision/MAVI/TensorRT/dataset/output/obj/'

    dino = DINO(onnx_model_path, tensorrt_engine_path, engine_precision, img_size, batch_size)

    for image_name in sorted(os.listdir(image_dir)):
        image_path = image_dir + image_name
        save_path = save_dir + image_name
        
        image = cv2.imread(image_path)
        pred_box, scores, labels = dino.infer_one_image(image)

        dino.save_image_with_bbox(image, pred_box, scores, labels, save_path)       


if __name__ == '__main__':
    main()