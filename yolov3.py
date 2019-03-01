import sys, os
import time
#print(os.getcwd())
# the path below should be change according to the path of darknet
sys.path.append(os.path.join('/home/spl/chuyutensor/darknet/','python/'))

import darknet as dn
import pdb

dn.set_gpu(0)
os.chdir('/home/spl/chuyutensor/darknet')

start = time.time()
net = dn.load_net(b"cfg/yolov3.cfg", b"yolov3.weights", 0)
meta = dn.load_meta(b"cfg/coco.data")
print(time.time() - start)

r = dn.detect(net, meta, b"data/bedroom.jpg")
print(r)
