## Introdction
It is a re-implementation of paper named "Deep Video Super-Resolution Network Using Dynamic Upsampling Filters Without Explicit Motion Compensation" called VSR-DUF model. There are both training codes and test codes about VSR-DUF based tensorflow.  

If you are interested in Image Super-rerolution(ISR) or Video Super-resolution(VSR), you maybe know there is paper author github: [https://github.com/yhjo09/VSR-DUF/](https://github.com/yhjo09/VSR-DUF/ "VSR-DUF")  about VSR-DUF, but it's only can be tested with your Low-resolution frames and can't be trained. Therefor, I try to re-implement this paper based above github repository. I hope this will be helpful for you. 

If you think it is useful, please star it. Thank you.

## Todo
[.] The error of image size in reference stage with a specific training modification

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

If you want to improve it or change something acording your requirements, just feel free to modify it!

## Other:
This is another repository about ISR/VSR [Latest-development-of-ISR-VSR.](https://github.com/HymEric/latest-development-of-ISR-VSR), it's updating with the conference...

## Author
EricHym (Yongming He)   
Interests: CV and Deep Learning  
If you have or find any problems, this is my email: [yongminghe_eric@qq.com](yongminghe_eric@qq.com). And I'm glad to reply it.
Thanks. 
