
# coding: utf-8

# 导入相关库，其中TFNet为darkflow文件中的。

# In[1]:


import matplotlib.pyplot as plt 
import numpy as np 
from darkflow.net.build import TFNet 
import cv2 
import pprint as pp


# model为模型的配置文件，load为模型的参数文件，需要自己训练更改。cfg与bin都在yolo_bicycle文件目录下。

# 输入为原图片（并不是地址），在原图片上标注预测物体的位置。

# In[3]:


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


# In[ ]:


def result_person(results):
    for result in results:
        if result['label'] == 'person' and result['confidence'] > 0.5:
            bbox = (result['topleft']['x'], result['topleft']['y'], 
                    result['bottomright']['x'] - result['topleft']['x'], 
                    result['bottomright']['y'] - result['topleft']['y'])
            return bbox


# In[12]:


def object_detection(original_img):
    options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.1}
    tfnet = TFNet(options)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    results = tfnet.return_predict(original_img)
    return result_person(results)

    """fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(original_img)

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.imshow(boxing(original_img, results))"""

