# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 17:16:56 2017

@author: vasir
"""

#from sklearn.datasets import load_digits
#from sklearn.model_selection import train_test_split
#from matplotlib.pylab import plt
#digits=load_digits()
#X,Y=digits.data,digits.target
#X_train,Y_train,X_test,Y_test=train_test_split(X,Y,test_size=0.5)
##plt.figure(1, figsize=(3, 3))
#plt.imshow(digits.images[-1], cmap=plt.cm.gray_r, interpolation='nearest')
#plt.show()

import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Flatten,Dropout,Dense
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.datasets import mnist
from keras import backend as K
K.set_image_dim_ordering('tf')
s=5
np.random.seed(s)
(X_train,Y_train),(X_test,Y_test)=mnist.load_data()
X_train=X_train.reshape(X_train.shape[0],1,28,28).astype('float32')
X_test=X_test.reshape(X_test.shape[0],1,28,28).astype('float32')

#normalizing inputs
X_train=X_train/255
X_test=X_test/255
#one-hot encoding
Y_train=np_utils.to_categorical(Y_train)
Y_test=np_utils.to_categorical(Y_test)
num_classes=Y_train.shape[1]

#Basic Convoulutional model
def baseline_model():
    model=Sequential()
    model.add(Conv2D(32,(5,5),input_shape=(1,28,28),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(num_classes,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model
model=baseline_model()
model.fit(X_train,Y_train,validation_data=(X_test,Y_test),epoch=10,batch_size=32)
score = model.evaluate(X_test, Y_test, verbose=0)
print("Baseline error %.2f%%" %(100-score[1]*100))
