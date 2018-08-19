from darkflow.net.build import TFNet 
import cv2 


class tfnet():
    options = {"pbLoad": "built_graph/yolo-2c.pb", "metaLoad": "built_graph/yolo-2c.meta", "threshold": 0.1}

    def __init__(self, options = options):
        self.tfnet = TFNet(options)
        self.results = None
        
    def result_transform(self):
        bboxes = []
        for result in self.results:
            bbox = (result['topleft']['x'], result['topleft']['y'], 
                    result['bottomright']['x'] - result['topleft']['x'], 
                    result['bottomright']['y'] - result['topleft']['y'])
            bboxes.append(bbox)
        return bboxes

    def object_detection(self, original_img):
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        self.results = self.tfnet.return_predict(original_img)
        return self.result_transform()