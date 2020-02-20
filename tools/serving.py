import cv2
import numpy as np

# from src.label 				import Label, lwrite
from os.path 				import splitext, basename, isdir
# from src.utils 				import crop_region, image_files_from_folder

import tensorflow as tf
import os

from tensorflow.python.platform import gfile


def main():
    model = TF_Model('data/vehicle-detector/yolo-voc')
    return model

class TF_Model:
    def __init__(self, path):
        weights = 'data/vehicle-detector/yolo-voc.weights'
        netcfg  = 'data/vehicle-detector/yolo-voc.cfg'
        dataset = 'data/vehicle-detector/voc.data'
        # self.threshold = threshold
        sess = tf.Session()

        with tf.io.gfile.GFile(path + '.pb', 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')
            graph_nodes=[n for n in graph_def.node]
            self.names = []
            for t in graph_nodes:
               self.names.append(t.name)
            # print(names)
        saver = tf.compat.v1.train.import_meta_graph(path + '.ckpt.meta')
        # saver = tf.compat.v1.train.Saver(tf.global_variables())
        saver.restore(sess, path + '.ckpt')
        sess.run(tf.compat.v1.global_variables_initializer())
        self.session = sess

        # 需要先复原变量
        # print(sess.run('b:0'))
        # # 1
        #
        # # 输入
        # input_x = sess.graph.get_tensor_by_name('x:0')
        # input_y = sess.graph.get_tensor_by_name('y:0')
        #
        # op = sess.graph.get_tensor_by_name('op_to_store:0')
        #
        # ret = sess.run(op,  feed_dict={input_x: 5, input_y: 5})
        # print(ret)


    def __call__(self, x):
        return


if __name__ == '__main__':
    model = main()
