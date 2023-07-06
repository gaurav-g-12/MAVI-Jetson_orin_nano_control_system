import time

class Mobile:
    def __init__(self):
        pass
    
    def start(self, obj_result_queue, txt_results_queue):
        while(1):
            obj_res = obj_result_queue.get()
            txt_res = txt_results_queue.get()

            # print('!'*100)
            # print(obj_res)
            # print(txt_res)
        
