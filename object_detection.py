from darkflow.net.build import TFNet 
import cv2 


class tfnet():
    options = {"pbLoad": "built_graph/yolo-2c.pb", "metaLoad": "built_graph/yolo-2c.meta", "threshold": 0.1}

    def __init__(self, options = options):
        self.tfnet = TFNet(options)
        self.results = None
        
    """process result, return two bboxes, one is persons, the other is bicycles"""
    def result_process(self):
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
        self.results = self.tfnet.return_predict(original_img)
        return self.result_process()