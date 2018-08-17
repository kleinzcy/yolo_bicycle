import matplotlib.pyplot as plt 
import numpy as np 
from darkflow.net.build import TFNet 
import cv2 
import pprint as pp


# model为模型的配置文件，load为模型的参数文件，需要自己训练更改。cfg与bin都在yolo_bicycle文件目录下。

# 输入为原图片（并不是地址），在原图片上标注预测物体的位置。


def boxing(original_img, predictions):
        newImage = np.copy(original_img)

        for result in predictions:
            top_x = result['topleft']['x']
            top_y = result['topleft']['y']
            
            btm_x = result['bottomright']['x']
            btm_y = result['bottomright']['y']

            confidence = result['confidence']
            label = result['label'] + " " + str(round(confidence, 3))

            if confidence > 0.3:
                newImage = cv2.rectangle(newImage, (top_x, top_y), (btm_x, btm_y), (255,0,0), 3)
                newImage = cv2.putText(newImage, label, (top_x, top_y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.8, (0, 230, 0), 1, cv2.LINE_AA)         
        return newImage


def result_transform(results):
    bboxes = []
    for result in results:
        bbox = (result['topleft']['x'], result['topleft']['y'], 
                result['bottomright']['x'] - result['topleft']['x'], 
                result['bottomright']['y'] - result['topleft']['y'])
        bboxes.append(bbox)
    return bboxes


def object_detection(original_img):
    options = {"pbLoad": "built_graph/yolo-2c.pb", "metaLoad": "built_graph/yolo-2c.meta", "threshold": 0.1}
    tfnet = TFNet(options)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    results = tfnet.return_predict(original_img)
    return result_transform(results)

