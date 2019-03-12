# from darkflow.net.build import TFNet 
import cv2 
import sys, os
import time
#print(os.getcwd())
# the path below should be change according to the path of darknet
sys.path.append(os.path.join('/home/spl/chuyutensor/darknet/','python/'))

import darknet as dn
import pdb

class tfnet():
    options = {"pbLoad": "built_graph/yolo-2c.pb", "metaLoad": "built_graph/yolo-2c.meta", "threshold": 0.1}

    def __init__(self, options = options):
        start = time.time()
        self.net = dn.load_net(b"cfg/yolov3.cfg", b"yolov3.weights", 0)
        self.meta = dn.load_meta(b"cfg/coco.data")
        print(time.time() - start)
        # self.tfnet = TFNet(options)
        self.results = None
        
    """process result, return two bboxes, one is persons, the other is bicycles"""
    def result_process(self):
        # change according to the output of yolov3
        bboxes_person = []
        bboxes_bicycle = []
        for result in self.results:
            bbox = (result['topleft']['x'], result['topleft']['y'], 
                    result['bottomright']['x'], result['bottomright']['y'])
            if result['label'] == 'person':
                bboxes_person.append(bbox)
            else:
                bboxes_bicycle.append(bbox)
        return bboxes_person, bboxes_bicycle

    def object_detection(self, original_img):
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        self.rseults = dn.detect(self.net, self.meta, b"data/bedroom.jpg")
        # self.results = self.tfnet.return_predict(original_img)
        return self.result_process()
