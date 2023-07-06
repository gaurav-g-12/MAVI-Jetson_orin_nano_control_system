import cv2
import time
import os 

class Camera:
    def __init__(self):
        self.num_of_frames = 0
        self.dimention = (480, 640)
        # self.cap = cv2.VideoCapture(0)
        self.save_dir = ''
        self.camera_start = True
        self.image_dir = '/home/vision/MAVI/TensorRT/dataset/signboard/28-01-2023'

    
    def start(self, obj_queue, txt_queue):
        
        while(1):
            # ret, frame = self.cap.read()
            for image_name in os.listdir(self.image_dir):
                
                image_path = os.path.join(self.image_dir, image_name)
                
                frame = cv2.imread(image_path)
                # frame = cv2.resize(frame, self.dimention)

                if(obj_queue.qsize() > 10):
                    
                    obj_queue.queue.clear()
                
                if(txt_queue.qsize() > 10):
                    
                    txt_queue.queue.clear()
                    
                obj_queue.put(frame)
                txt_queue.put(frame)
                
    
        # self.cap.release()
    

    def stop(self):
        self.camera_start = False
