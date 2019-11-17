# -*- coding:utf-8 -*-
"""
@version: 01
@author:erichym
@license: Apache Licence 
@file: mytest.py 
@time: 2018/12/08
@contact: yongminghe_eric@qq.com
@software: PyCharm
"""


import numpy as np
from utils import LoadImage,Huber
import glob
import tensorflow as tf
from PIL import Image
from tensorflow.python.platform import gfile

T_in=7

def get_x(path):
    dir_frames=glob.glob(path+"*.png")
    dir_frames.sort()
    frames=[]
    for f in dir_frames:
        frames.append(LoadImage(f))
    frames = np.asarray(frames) # print(frames.shape) (20, 100, 115, 3)
    frames_padded = np.lib.pad(frames, pad_width=((T_in // 2, T_in // 2), (0, 0), (0, 0), (0, 0)), mode='constant') # print(frames_padded.shape) (26, 100, 115, 3)
    return frames,frames_padded

def get_y(path):
    dir_frames = glob.glob(path+"*.png")
    dir_frames.sort()
    frames = []
    for f in dir_frames:
        frames.append(LoadImage(f))
    frames = np.asarray(frames)
    return frames

x_path='./data/x_test_data4x/'
y_path='./data/y_test_data/'
x_data,x_data_padded=get_x(x_path) # print(x_data_padded.shape) (26, 100, 115, 3)
y_data=get_y(y_path) # print(y_data.shape) (20, 400, 460, 3)

y_true=[]
for i in range(len(y_data)):
    y_true.append(y_data[i][np.newaxis,np.newaxis,:,:,:]) # print(yy[1].shape) (1, 1, 400, 460, 3)
y_true=np.asarray(y_true)
y_data=y_true

with tf.Graph().as_default():
    output_graph_def = tf.GraphDef()
    output_graph_path = './model/My_Duf_2.pb'

    with gfile.FastGFile(output_graph_path,"rb") as f:
        output_graph_def.ParseFromString(f.read())
        # fix nodes
        for node in output_graph_def.node:
            if node.op == 'RefSwitch':
                node.op = 'Switch'
            elif node.op == 'AssignSub':
                node.op = 'Sub'
                if 'use_locking' in node.attr: del node.attr['use_locking']
            elif node.op == 'AssignAdd':
                node.op='Add'
                if 'use_locking' in node.attr: del node.attr['use_locking']
        _ = tf.import_graph_def(output_graph_def,name="")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        input = sess.graph.get_tensor_by_name("L_in:0")
        output = sess.graph.get_tensor_by_name("out_H:0")
        is_train=sess.graph.get_tensor_by_name('is_train:0')

        total_loss=0
        for j in range(x_data.shape[0]):
            in_L = x_data_padded[j:j + T_in]  # select T_in frames
            in_L = in_L[np.newaxis, :, :, :, :]
            y_out = sess.run(output, feed_dict={input: in_L, is_train: False})
            Image.fromarray(np.around(y_out[0, 0] * 255).astype(np.uint8)).save('./result_test/{:05}.png'.format(j))

            cost = Huber(y_true=y_data[j], y_pred=y_out, delta=0.01)
            loss = sess.run(cost)
            total_loss = total_loss+loss
            print('this single test cost: {:.7f}'.format(loss))

        avg_test_loss=total_loss/x_data.shape[0]
        print("avg test cost: {:.7f}".format(avg_test_loss))
