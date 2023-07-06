import cv2
import time
import threading
from PIL import Image
import tensorrt as trt
import multiprocessing as mp
import queue as Q

from camera import Camera
from mobile import Mobile
from signboard_pipeline import init_signboard_pipeline
from object_pipeline import init_object_pipeline


class myThread(threading.Thread):
   def __init__(self, func, args):
      threading.Thread.__init__(self)
      self.func = func
      self.args = args

   def run(self):
      print ("Starting")
      self.func(*self.args)
      print ("Exiting")


if __name__ == '__main__':

   cam = Camera()
   mob = Mobile()
   sign_pipe = init_signboard_pipeline()
   obj_pipe = init_object_pipeline()
   
   # input frame queues 
   obj_frame_queue = Q.Queue()
   txt_frame_queue = Q.Queue()

   # output results queue
   obj_results_queue = Q.Queue()
   txt_results_queue = Q.Queue()

   cam_thread = myThread(cam.start, [obj_frame_queue, txt_frame_queue])

   mob_thread = myThread(mob.start, [obj_results_queue, txt_results_queue])

   txt_thread = myThread(sign_pipe.start_inferenceing_txt, [txt_frame_queue, txt_results_queue])

   obj_thread = myThread(obj_pipe.start_inferenceing_obj, [obj_frame_queue, obj_results_queue])


   cam_thread.start()
   mob_thread.start()
   txt_thread.start()
   obj_thread.start()

   # thread1.join()
   # thread2.join()

   # print('*'*1000)

   # parseq.destory();
   # dbnet.destory();

