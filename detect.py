## python 深度学习目标跟踪检测代码
## 用法: 在Windows命令行输入
#        python detect.py FileName
## 对指定的图片文件或图像数据进行目标检测
## 2018-4-13

import sys
sys.path.append('.')
import time
import tensorflow as tf
import numpy as np
from PIL import Image

# 模型位置
PATH_TO_CKPT = '/tmp/output_graph.pb'
OUTPUT_LAYER = 'final_result:0'

# 初始化图
detection_graph = tf.Graph()
with detection_graph.as_default():
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        od_graph_def = tf.GraphDef()
        od_graph_def.ParseFromString(fid.read())
        tf.import_graph_def(od_graph_def, name='')

detection_graph.as_default()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(graph=detection_graph, config=config)

# 检测图像数据
def test_src(src):
    image_np = np.array(src).astype(np.uint8)
    image_np_expanded = np.expand_dims(image_np, axis=0)
    
    softmax_tensor = detection_graph.get_tensor_by_name(OUTPUT_LAYER)
    predictions = sess.run(softmax_tensor,
        feed_dict={'ExpandDims:0': image_np_expanded})

    return predictions

# 检测图片文件
def test_image(path):
    try:
        image = Image.open(path)
        predictions = test_src(image)
        print('predictions.type =', type(predictions))
        print('predictions.shape =', predictions.shape)
        print('predictions =', predictions)
    except IOError:
        print('IOError: File is not accessible.')

# 激活GPU
test_image('image.jpg')

if __name__ == '__main__':
  test_image('image.jpg' if (1 == len(sys.argv)) else sys.argv[1])
