# YOLO_Bicycle

This work is our scientific research.

## Problem

Our work is hard to apply to complex scene.

Refer to [this](https://www.learnopencv.com/object-tracking-using-opencv-cpp-python/) to learn more about tracker API in opencv.

## Environment

`NOTE`:pip or pip3 depend on your environment, and you can install tensorflow through anaconda, referring to [this](https://www.tensorflow.org/install/install_linux)

* Tensorflow installation

  pip3 install tensorflow, refer to [this](https://www.tensorflow.org/install/install_linux#installing_with_native_pip) for more details.

* Opncv and opencv-contrib installation

  pip3 install opencv-contrib-python, refer to [this](https://pypi.org/project/opencv-python/).

  Learn more about the difference bewteen opencv and opencv-contrib, refer to [this](https://github.com/opencv/opencv_contrib). 

* Darknet installation

  Refer to [this](https://github.com/thtrieu/darkflow).

* File

  Download yolo.weights file from [here](https://pan.baidu.com/s/1vRT3Iwb5KONtWo85rzUvJg), password pk5v. And place the file in /bin/.
  
* Update .pb files

  [Here](https://pan.baidu.com/s/1zqU9fOcEnhMvqOukv8juFQ) password 27yw.
  
## Coding
  
The kernel code is in tracker.py  is friendly to read.

judge.py is used to determined the statement of the current frame

object_detection.py is used to detect person and bicycle for current frame by yolo

overlap_ratio.py is used to count the coincidence and match the bike to the person with the most overlap
  
if you want to run this item, you just need enter the code in the terminal:

`python tracker.py` or `python demo.py`

Our project logic diagram is as [follow](http://t.cn/RFUAEVI)

Pull request if you fix the problem or find some bugs.
