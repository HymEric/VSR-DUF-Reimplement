# -*- coding:utf-8 -*-
""" 
@version: 01
@author:erichym
@license: Apache Licence 
@file: mytrain.py 
@time: 2018/12/08
@contact: yongminghe_eric@qq.com
@software: PyCharm
"""
import tensorflow as tf
from utils import BatchNorm,Conv3D,DynFilter3D,depth_to_space_3D,Huber, LoadImage
import numpy as np
import glob
from tensorflow.python.framework import graph_util

# Size of input temporal radius
T_in = 7
# Upscaling factor
R = 4

def freeze_graph(check_point_folder,model_folder,pb_name):
    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(check_point_folder)
    input_checkpoint = checkpoint.model_checkpoint_path

    # We precise the file fullname of our freezed graph
    output_graph=model_folder+'/'+pb_name
    # Before exporting our graph, we need to precise what is our output node
    # this variables is plural, because you can have multiple output nodes
    output_node_names = "out_H"
    list_str =[]

    # We clear the devices, to allow TensorFlow to control on the loading where it wants operations to be calculated
    clear_devices = True

    # We import the meta graph and retrive a Saver
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

    # We retrieve the protobuf graph definition
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    # We start a session and restore the graph weights
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)
        # fix batch norm nodes
        for node in input_graph_def.node:
            if node.op == 'RefSwitch':
                node.op = 'Switch'
                for index in range(len(node.input)):
                    if 'moving_' in node.input[index]:
                        node.input[index] = node.input[index] + '/read'
            elif node.op == 'AssignSub':
                node.op = 'Sub'
                if 'use_locking' in node.attr: del node.attr['use_locking']
        # We use a built-in TF helper to export variables to constant
        output_graph_def = graph_util.convert_variables_to_constants(
            sess,
            input_graph_def,
            output_node_names.split(",")  # We split on comma for convenience
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

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

"""
train datasets
"""
x_train_path='./data/x_train_data4x/'
y_train_path='./data/y_train_data/'
x_train_data,x_train_data_padded=get_x(x_train_path) # print(x_data_padded.shape) (26, 100, 115, 3)
y_train_data=get_y(y_train_path) # print(y_data.shape) (20, 400, 460, 3)

y_true=[]
for i in range(len(y_train_data)):
    y_true.append(y_train_data[i][np.newaxis,np.newaxis,:,:,:]) # print(yy[1].shape) (1, 1, 400, 460, 3)
y_true=np.asarray(y_true)
y_train_data=y_true

"""
valid datasets
"""
x_valid_path='./data/x_valid_data4x/'
y_valid_path='./data/y_valid_data/'
x_valid_data,x_valid_data_padded=get_x(x_valid_path) # print(x_data_padded.shape) (26, 100, 115, 3)
y_valid_data=get_y(y_valid_path) # print(y_data.shape) (20, 400, 460, 3)

y_true=[]
for i in range(len(y_valid_data)):
    y_true.append(y_valid_data[i][np.newaxis,np.newaxis,:,:,:]) # print(yy[1].shape) (1, 1, 400, 460, 3)
y_true=np.asarray(y_true)
y_valid_data=y_true

# Gaussian kernel for downsampling
def gkern(kernlen=13, nsig=1.6):
    import scipy.ndimage.filters as fi
    # create nxn zeros
    inp = np.zeros((kernlen, kernlen))
    # set element at the middle to one, a dirac delta
    inp[kernlen // 2, kernlen // 2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    return fi.gaussian_filter(inp, nsig)

h = gkern(13, 1.6)  # 13 and 1.6 for x4
h = h[:, :, np.newaxis, np.newaxis].astype(np.float32)

# Network
H_out_true=tf.placeholder(tf.float32,shape=(1,1,None,None,3),name='H_out_true') 

is_train = tf.placeholder(tf.bool, shape=[],name='is_train') # Phase ,scalar

# L_ = DownSample(H_in, h, R)
L =  tf.placeholder(tf.float32, shape=[None, T_in, None, None, 3],name='L_in')

# build model
stp = [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]]
sp = [[0, 0], [0, 0], [1, 1], [1, 1], [0, 0]]
# [1, 3, 3, 3, 64] [filter_depth, filter_height, filter_width, in_channels,out_channels]
x = Conv3D(tf.pad(L, sp, mode='CONSTANT'), [1, 3, 3, 3, 64], [1, 1, 1, 1, 1], 'VALID', name='conv1')

F = 64
G = 32
for r in range(3): 
    t = BatchNorm(x, is_train, name='Rbn' + str(r + 1) + 'a')
    t = tf.nn.relu(t)
    t = Conv3D(t, [1, 1, 1, F, F], [1, 1, 1, 1, 1], 'VALID', name='Rconv' + str(r + 1) + 'a')

    t = BatchNorm(t, is_train, name='Rbn' + str(r + 1) + 'b')
    t = tf.nn.relu(t)
    t = Conv3D(tf.pad(t, stp, mode='CONSTANT'), [3, 3, 3, F, G], [1, 1, 1, 1, 1], 'VALID',
               name='Rconv' + str(r + 1) + 'b')

    x = tf.concat([x, t], 4)
    F += G
for r in range(3, 6):
    t = BatchNorm(x, is_train, name='Rbn' + str(r + 1) + 'a')
    t = tf.nn.relu(t)
    t = Conv3D(t, [1, 1, 1, F, F], [1, 1, 1, 1, 1], 'VALID', name='Rconv' + str(r + 1) + 'a')

    t = BatchNorm(t, is_train, name='Rbn' + str(r + 1) + 'b')
    t = tf.nn.relu(t)
    t = Conv3D(tf.pad(t, sp, mode='CONSTANT'), [3, 3, 3, F, G], [1, 1, 1, 1, 1], 'VALID',
               name='Rconv' + str(r + 1) + 'b')

    x = tf.concat([x[:, 1:-1], t], 4)
    F += G

# sharen section
x = BatchNorm(x, is_train, name='fbn1')
x = tf.nn.relu(x)
x = Conv3D(tf.pad(x, sp, mode='CONSTANT'), [1, 3, 3, 256, 256], [1, 1, 1, 1, 1], 'VALID', name='conv2')
x = tf.nn.relu(x)

# R
r = Conv3D(x, [1, 1, 1, 256, 256], [1, 1, 1, 1, 1], 'VALID', name='rconv1')
r = tf.nn.relu(r)
r = Conv3D(r, [1, 1, 1, 256, 3 * 16], [1, 1, 1, 1, 1], 'VALID', name='rconv2')

# F
f = Conv3D(x, [1, 1, 1, 256, 512], [1, 1, 1, 1, 1], 'VALID', name='fconv1')
f = tf.nn.relu(f)
f = Conv3D(f, [1, 1, 1, 512, 1 * 5 * 5 * 16], [1, 1, 1, 1, 1], 'VALID', name='fconv2')

ds_f = tf.shape(f)
f = tf.reshape(f, [ds_f[0], ds_f[1], ds_f[2], ds_f[3], 25, 16])
f = tf.nn.softmax(f, dim=4)

Fx=f
Rx =r

x=L
x_c = []
for c in range(3):
    t = DynFilter3D(x[:, T_in // 2:T_in // 2 + 1, :, :, c], Fx[:, 0, :, :, :, :], [1, 5, 5])  # [B,H,W,R*R]
    t = tf.depth_to_space(t, R)  # [B,H*R,W*R,1]
    x_c += [t]
x = tf.concat(x_c, axis=3)  # [B,H*R,W*R,3] Tensor("concat_9:0", shape=(?, ?, ?, 3), dtype=float32)

x = tf.expand_dims(x, axis=1) # Tensor("ExpandDims_3:0", shape=(?, 1, ?, ?, 3), dtype=float32)
Rx = depth_to_space_3D(Rx, R)  # [B,1,H*R,W*R,3] Tensor("Reshape_6:0", shape=(?, ?, ?, ?, ?), dtype=float32)
x += Rx # Tensor("add_18:0", shape=(?, ?, ?, ?, 3), dtype=float32) 

out_H=tf.clip_by_value(x,0,1,name='out_H')

cost=Huber(y_true=H_out_true,y_pred=out_H,delta=0.01)

learning_rate=0.001
learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32,name='learning_rate')
learning_rate_decay_op = learning_rate.assign(learning_rate * 0.9)
optimizer=tf.train.AdamOptimizer(learning_rate).minimize(cost)

# total train epochs
num_epochs=100

# Session
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
saver = tf.train.Saver()
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    # tf.global_variables_initializer().run()
    for global_step in range(num_epochs):
        if global_step!=0 & np.mod(global_step,10)==0:
            sess.run(learning_rate_decay_op)
        total_train_loss = 0
        total_valid_loss = 0
        print("-------------------------- Epoch {:3d} ----------------------------".format(global_step))
        for i in range(5):
            print("---------- optimize sess.run start ----------")
            for j in range(x_train_data.shape[0]):
                in_L = x_train_data_padded[j:j + T_in]  # select T_in frames
                in_L = in_L[np.newaxis, :, :, :, :]
                sess.run(optimizer,feed_dict={H_out_true:y_train_data[j],L:in_L,is_train: True})
                print("optimize:"+ str(i)+" "+str(j) +" finished.")
        print("---------- train cost sess.run start -----------")
        for j in range(x_train_data.shape[0]):
            in_L = x_train_data_padded[j:j + T_in]  # select T_in frames
            in_L = in_L[np.newaxis, :, :, :, :]
            train_loss = sess.run(cost, feed_dict={H_out_true: y_train_data[j], L: in_L, is_train: True})
            total_train_loss = total_train_loss + train_loss
            # print('this single train cost: {:.7f}'.format(train_loss))
            print("train cost :" + str(i) + " " + str(j) + " finished.")
        for j in range(x_valid_data.shape[0]):
            in_L = x_valid_data_padded[j:j + T_in]  # select T_in frames
            in_L = in_L[np.newaxis, :, :, :, :]
            valid_loss = sess.run(cost, feed_dict={H_out_true: y_valid_data[j], L: in_L, is_train: True})
            total_valid_loss = total_valid_loss + valid_loss
            # print('this single valid cost: {:.7f}'.format(valid_loss))
            print("valid cost :" + str(i) + " " + str(j) + " finished.")
        avg_train_loss=total_train_loss/16/x_train_data.shape[0]
        avg_valid_loss=total_valid_loss/16/x_valid_data.shape[0]
        print("Epoch - {:2d}, avg loss on train set: {:.7f}, avg loss on valid set: {:.7f}.".format(global_step, avg_train_loss,avg_valid_loss))

        if global_step==0:
            with open('./logs/pb_graph_log.txt', 'w') as f:
                f.write(str(sess.graph_def)) 
            var_list = tf.global_variables()
            with open('./logs/global_variables_log.txt','w') as f:
                f.write(str(var_list)) 

        tf.train.write_graph(sess.graph_def, '.', './checkpoint/duf_'+str(global_step)+'.pbtxt')
        saver.save(sess, save_path="./checkpoint/duf",global_step=global_step)
        freeze_graph(check_point_folder='./checkpoint/',model_folder='./model',pb_name='My_Duf_'+str(global_step)+'.pb')
