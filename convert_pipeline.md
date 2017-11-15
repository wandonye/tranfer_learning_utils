

```python
from __future__ import division, print_function
from keras import backend as K
from keras.layers import *
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.convolutional import Convolution2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
from google.protobuf import text_format

from scipy.misc import imresize
import matplotlib.pyplot as plt
import numpy as np
import os
import re
```

    Using TensorFlow backend.



```python
import caffe
from caffe_utils import *
from extra_layers import *
```


```python
OUTPUT_DIR = "./data"

caffe.set_mode_cpu()
net = caffe.Net('VGG_FACE_deploy_3v2_2_gpu.prototxt', 
                '_iter_6829.caffemodel', caffe.TEST)



# write out weight matrices and bias vectors
caffe_weights = {}
for k, v in net.params.items():
    print(k, v[0].data.shape, v[1].data.shape)
    np.save(os.path.join(OUTPUT_DIR, "W_{:s}.npy".format(k)), v[0].data)
    caffe_weights["W_"+k] = v[0].data
    np.save(os.path.join(OUTPUT_DIR, "b_{:s}.npy".format(k)), v[1].data)
    caffe_weights["b_"+k] = v[1].data

```

    conv1_1 (64, 3, 3, 3) (64,)
    conv1_2 (64, 64, 3, 3) (64,)
    conv2_1 (128, 64, 3, 3) (128,)
    conv2_2 (128, 128, 3, 3) (128,)
    conv3_1 (256, 128, 3, 3) (256,)
    conv3_2 (256, 256, 3, 3) (256,)
    conv3_3 (256, 256, 3, 3) (256,)
    conv4_1 (512, 256, 3, 3) (512,)
    conv4_2 (512, 512, 3, 3) (512,)
    conv4_3 (512, 512, 3, 3) (512,)
    conv5_1 (512, 512, 3, 3) (512,)
    conv5_2 (512, 512, 3, 3) (512,)
    conv5_3 (512, 512, 3, 3) (512,)
    fc6 (4096, 25088) (4096,)
    fc7 (4096, 4096) (4096,)
    fc8_train (2, 4096) (2,)



```python
# layer names and output shapes
for layer_name, blob in net.blobs.items():
    print(layer_name, blob.data.shape)
```


```python
def transform_conv_weight(W):
    # for non FC layers, do this because Keras does convolution vs Caffe correlation
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W[i, j] = np.rot90(W[i, j], 2)
    return W

def transform_fc_weight(W):
    return W.T
```


```python
keras_weights = {}
for k in caffe_weights:
    if k[0]=='W':
        if k[2:6]=='conv':
            keras_weights[k] = np.rollaxis(np.rollaxis(transform_conv_weight(caffe_weights[k]),
                                                       1,4),0,4)
        if k[2:4]=='fc':
            keras_weights[k] = transform_fc_weight(caffe_weights[k])
    else:
        keras_weights[k] = caffe_weights[k]
    print(k,keras_weights[k].shape)
```

    W_conv5_3 (3, 3, 512, 512)
    W_fc8_train (4096, 2)
    b_conv2_1 (128,)
    b_conv3_2 (256,)
    W_conv2_2 (3, 3, 128, 128)
    b_conv2_2 (128,)
    b_conv4_3 (512,)
    W_conv5_1 (3, 3, 512, 512)
    b_conv3_1 (256,)
    b_fc7 (4096,)
    W_conv3_3 (3, 3, 256, 256)
    b_conv5_3 (512,)
    W_conv2_1 (3, 3, 64, 128)
    W_conv4_3 (3, 3, 512, 512)
    W_conv5_2 (3, 3, 512, 512)
    W_conv3_1 (3, 3, 128, 256)
    b_conv1_2 (64,)
    W_conv1_2 (3, 3, 64, 64)
    b_conv4_1 (512,)
    W_conv4_2 (3, 3, 512, 512)
    W_conv4_1 (3, 3, 256, 512)
    b_conv5_1 (512,)
    W_conv1_1 (3, 3, 3, 64)
    W_fc6 (25088, 4096)
    b_conv1_1 (64,)
    W_conv3_2 (3, 3, 256, 256)
    W_fc7 (4096, 4096)
    b_conv4_2 (512,)
    b_conv3_3 (256,)
    b_fc6 (4096,)
    b_conv5_2 (512,)
    b_fc8_train (2,)



```python
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
```


```python
def VGG_16(weights):
    model = Sequential()
    model.add(Conv2D(64, (3, 3),input_shape=(224,224,3), 
                     weights=(weights['W_conv1_1'],weights['b_conv1_1']), 
                     activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, (3, 3), 
                     weights=(weights['W_conv1_2'],weights['b_conv1_2']),                      
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3),
                     weights=(weights['W_conv2_1'],weights['b_conv2_1']),                      
                     activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), 
                     weights=(weights['W_conv2_2'],weights['b_conv2_2']),                      
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), 
                     weights=(weights['W_conv3_1'],weights['b_conv3_1']),                      
                     activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), 
                    weights=(weights['W_conv3_2'],weights['b_conv3_2']),                      
                    activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), 
                     weights=(weights['W_conv3_3'],weights['b_conv3_3']),                      
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), 
                     weights=(weights['W_conv4_1'],weights['b_conv4_1']),                      
                     activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3),
                     weights=(weights['W_conv4_2'],weights['b_conv4_2']),                      
                     activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3),
                     weights=(weights['W_conv4_3'],weights['b_conv4_3']),                      
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), 
                     weights=(weights['W_conv5_1'],weights['b_conv5_1']),                      
                     activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), 
                     weights=(weights['W_conv5_2'],weights['b_conv5_2']),                      
                     activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), 
                     weights=(weights['W_conv5_3'],weights['b_conv5_3']),                      
                     activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, 
                    weights=(weights['W_fc6'],weights['b_fc6']), 
                    activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096,
                    weights=(weights['W_fc7'],weights['b_fc7']), 
                    activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, 
                    weights=(weights['W_fc8_train'],weights['b_fc8_train']), 
                    activation='softmax'))

    return model
```


```python
model = VGG_16(keras_weights)
```


```python
print(model.summary())
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_1 (Conv2D)            (None, 222, 222, 64)      1792      
    _________________________________________________________________
    zero_padding2d_1 (ZeroPaddin (None, 224, 224, 64)      0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 222, 222, 64)      36928     
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 111, 111, 64)      0         
    _________________________________________________________________
    zero_padding2d_2 (ZeroPaddin (None, 113, 113, 64)      0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 111, 111, 128)     73856     
    _________________________________________________________________
    zero_padding2d_3 (ZeroPaddin (None, 113, 113, 128)     0         
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 111, 111, 128)     147584    
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 55, 55, 128)       0         
    _________________________________________________________________
    zero_padding2d_4 (ZeroPaddin (None, 57, 57, 128)       0         
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 55, 55, 256)       295168    
    _________________________________________________________________
    zero_padding2d_5 (ZeroPaddin (None, 57, 57, 256)       0         
    _________________________________________________________________
    conv2d_6 (Conv2D)            (None, 55, 55, 256)       590080    
    _________________________________________________________________
    zero_padding2d_6 (ZeroPaddin (None, 57, 57, 256)       0         
    _________________________________________________________________
    conv2d_7 (Conv2D)            (None, 55, 55, 256)       590080    
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 27, 27, 256)       0         
    _________________________________________________________________
    zero_padding2d_7 (ZeroPaddin (None, 29, 29, 256)       0         
    _________________________________________________________________
    conv2d_8 (Conv2D)            (None, 27, 27, 512)       1180160   
    _________________________________________________________________
    zero_padding2d_8 (ZeroPaddin (None, 29, 29, 512)       0         
    _________________________________________________________________
    conv2d_9 (Conv2D)            (None, 27, 27, 512)       2359808   
    _________________________________________________________________
    zero_padding2d_9 (ZeroPaddin (None, 29, 29, 512)       0         
    _________________________________________________________________
    conv2d_10 (Conv2D)           (None, 27, 27, 512)       2359808   
    _________________________________________________________________
    max_pooling2d_4 (MaxPooling2 (None, 13, 13, 512)       0         
    _________________________________________________________________
    zero_padding2d_10 (ZeroPaddi (None, 15, 15, 512)       0         
    _________________________________________________________________
    conv2d_11 (Conv2D)           (None, 13, 13, 512)       2359808   
    _________________________________________________________________
    zero_padding2d_11 (ZeroPaddi (None, 15, 15, 512)       0         
    _________________________________________________________________
    conv2d_12 (Conv2D)           (None, 13, 13, 512)       2359808   
    _________________________________________________________________
    zero_padding2d_12 (ZeroPaddi (None, 15, 15, 512)       0         
    _________________________________________________________________
    conv2d_13 (Conv2D)           (None, 13, 13, 512)       2359808   
    _________________________________________________________________
    zero_padding2d_13 (ZeroPaddi (None, 15, 15, 512)       0         
    _________________________________________________________________
    max_pooling2d_5 (MaxPooling2 (None, 7, 7, 512)         0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 25088)             0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 4096)              102764544 
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 4096)              0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 4096)              16781312  
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 4096)              0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 2)                 8194      
    =================================================================
    Total params: 134,268,738
    Trainable params: 134,268,738
    Non-trainable params: 0
    _________________________________________________________________
    None



```python
model.save('keras_model_rot90.h5')  # creates a HDF5 file 'my_model.h5'
```


```python

```
