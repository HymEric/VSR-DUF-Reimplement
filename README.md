## Introdction
It is a re-implementation of paper named "Deep Video Super-Resolution Network Using Dynamic Upsampling Filters Without Explicit Motion Compensation" called VSR-DUF model. There are both training codes and test codes about VSR-DUF based tensorflow.  

If you are interested in Image Super-rerolution(ISR) or Video Super-resolution(VSR), you maybe know there is paper author github: [https://github.com/yhjo09/VSR-DUF/](https://github.com/yhjo09/VSR-DUF/ "VSR-DUF")  about VSR-DUF, but it's only can be tested with your Low-resolution frames and can't be trained. Therefor, I try to re-implement this paper based above github repository. I hope this will be helpful for you. 

If you think it is useful, please star it. Thank you.

## Note
If your environment is different with mine,you might encounter the error of large image size e.g. 480x270 in reference stage where the output has a black-border (0 value border). Unfortunately, I still havn't find the solution to address it. But this is a simple way to handle it by dividing the matrix before *tf.matmul* in *DynFilter3D* function in utils.py. Thanks to **beichengding** for his observation and solution.
```
def DynFilter3D(x, F, filter_size):
    '''
    3D Dynamic filtering
    input x: (b, t, h, w)
          F: (b, h, w, tower_depth, output_depth)
          filter_shape (ft, fh, fw)
    '''
    # make tower
    filter_localexpand_np = np.reshape(np.eye(np.prod(filter_size), np.prod(filter_size)), (filter_size[1], filter_size[2], filter_size[0], np.prod(filter_size)))
    filter_localexpand = tf.Variable(filter_localexpand_np, trainable=False, dtype='float32',name='filter_localexpand') 
    x = tf.transpose(x, perm=[0,2,3,1])   # [  1, 270, 480,  1] -> [  1, 270, 480,  1] 
    x_localexpand = tf.nn.conv2d(x, filter_localexpand, [1,1,1,1], 'SAME') # b, h, w, 1*5*5   result([  1, 270, 480,  25])
    x_localexpand = tf.expand_dims(x_localexpand, axis=3)  # b, h, w, 1, 1*5*5

    ##########   modification  ##############
    #   divide the large matrix into two small matrixs before tf.matmul operation

    num = 240                                        # e.g. num=240 free to change
    xl = x_localexpand[:,:,:num,:,:]                 #  left
    xr = x_localexpand[:,:,num:,:,:]                 #  right
    fl = F[:,:,:num,:,:]                             #  left
    fr = F[:,:,num:,:,:]                             #  right
    outl = tf.matmul(xl, fl) 
    outr = tf.matmul(xr, fr) 
    out = tf.concat([outl, outr], axis = 2)          #  cancat

    ##########   modification  ##############
    out = tf.squeeze(out, axis=3) # b, h, w, R*R

    return out
```

## Environments
TensorFlow:1.8.0  
pillow:5.3.0  
numpy:1.15.4

## Data-preparation

### The directory tree
![](https://github.com/HymEric/VSR-DUF-Reimplement/blob/master/tree%20png/folder%20tree.png)

### The data tree
![](https://github.com/HymEric/VSR-DUF-Reimplement/blob/master/tree%20png/data%20tree.png)

#### Data folder
Low-resolution:  
**./data/x_train_data4x/:** The scaled frames by 4x which will be used in train stage  
**./data/x_valid_data4x/:** The scaled frames by 4x which will be used in valid stage  
**./data/x_test_data4x/:** The scaled frames by 4x which will be used in test stage

Original-resolution:  
**./data/y_train_data/:** The HR frames which will be used in train stage coresponde to **./data/x_train_data4x/**  
**./data/y_valid_data/:** The HR frames which will be used in valid stage coresponde to **./data/x_valid_data4x/**

#### Another folders

Results of output:  
**./result_test/:** The output frames after VSR-DUF processing

**./checkpoint/:** save ckpt  
**./model/:** save pb model  
**./logs/:** save graph and variables  



## Attention

Before run train and test, your should prepare your datasets and check your environments if you want to run it successfully.   
Please, remind that the performence will be determined by your selected training datasets. Good training datasets good performence!   
Recommend: try to use public video dataset like [http://toflow.csail.mit.edu](http://toflow.csail.mit.edu), even though the performance may not be exactly the same as in the paper

And when you are runing mytest.py, you maybe need to modify it as you need to choose a trined .pb model to test.

## Run
**Training: python mytrain.py  
Testing: python mytest.py**

If you want to improve it or change something according your requirements, just feel free to modify it!

## Other:
This is another repository about ISR/VSR [Latest-development-of-ISR-VSR.](https://github.com/HymEric/latest-development-of-ISR-VSR), it's updating with the conference...

## Author
EricHym (Yongming He)   
Interests: CV and Deep Learning  
If you have or find any problems, this is my email: [yongminghe_eric@qq.com](yongminghe_eric@qq.com). And I'm glad to reply it.
Thanks. 
