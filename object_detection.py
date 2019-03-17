# from darkflow.net.build import TFNet 
import cv2 
import sys, os
import time
#print(os.getcwd())
# the path below should be change according to the path of darknet
sys.path.append('/home/nvidia/Documents/darknet/')

import darknet as dn
import pdb

os.chdir('/home/nvidia/Documents/darknet')
class tfnet():


    def __init__(self):
        start = time.time()
        self.net = dn.load_net(b"./bike_and_bicycle/yolov3-tiny.cfg", "./bike_and_bicycle/dataset/backup/yolov3-tiny_final.weights", 0)
        self.meta = dn.load_meta(b"./bike_and_bicycle/voc.data")
        print(time.time() - start)
        # self.tfnet = TFNet(options)
        # self.results = None
        
    """process result, return two bboxes, one is persons, the other is bicycles"""
    def result_process(self, results):
        # change according to the output of yolov3
        bboxes_person = []
        bboxes_bicycle = []
        for result in results:
            bbox = (result[2][0]-result[2][2]/2, result[2][1]-result[2][3]/2, 
                    result[2][0]+result[2][2]/2, result[2][1]+result[2][3]/2)
            if result[0] == 'person':
                bboxes_person.append(bbox)
            else:
                bboxes_bicycle.append(bbox)
        return bboxes_person, bboxes_bicycle

    def object_detection(self, original_img):
        temp = b'tmp/temp.png'
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(temp, original_img)
        results = dn.detect(self.net, self.meta, temp)

        return self.result_process(results)
