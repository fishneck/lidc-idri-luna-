import numpy as np
from keras.applications import vgg16,resnet50,inception_v3
import pandas as pd
import matplotlib.pyplot as plt
from time import time

import keras
from keras.layers import Dense, Dropout, Activation, Conv2D,GlobalAveragePooling2D
from keras.models import Sequential,Model
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras.optimizers import SGD


freeze = 8


def nn_vgg16():
    model_vgg16 = vgg16.VGG16(weights='imagenet',
                              include_top=True,  ## 是否保留顶层的3个全连接网络
                              input_shape=(224, 224, 3),  ## 输入层的尺寸
                              )
    model_vgg16.layers.pop()  # defaults to last
    model_vgg16.outputs = [model_vgg16.layers[-1].output]
    model_vgg16.layers[-1].outbound_nodes = []

    final_vgg16 = Sequential()
    for layer in model_vgg16.layers:
        final_vgg16.add(layer)
    final_vgg16.add(Dense(1024, activation='relu'))
    final_vgg16.add(Dense(6, activation='softmax'))

    for layer in final_vgg16.layers[:-4]:
        layer.trainable = False
    #final_vgg16.summary()
    final_vgg16.compile(optimizer='adamax', loss='categorical_crossentropy', metrics=['accuracy'])
    return final_vgg16


def nn_resnet50():
    model_resnet50 = resnet50.ResNet50(weights='imagenet',
                                       include_top=False,  ## 是否保留顶层的3个全连接网络
                                       input_shape=(224, 224, 3),  ## 输入层的尺寸
                                       )
    # model_resnet50.summary()
    x = model_resnet50.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(6, activation='softmax')(x)

    final_resnet50 = Model(inputs=model_resnet50.input, outputs=predictions)

    for layer in final_resnet50.layers[:-5]:
        layer.trainable = False
    final_resnet50.compile(optimizer='adamax', loss='categorical_crossentropy', metrics=['accuracy'])
    return  final_resnet50


def nn_inception_v3():
    model_inception_v3 = inception_v3.InceptionV3(weights='imagenet',
                                                  include_top=False,  ## 是否保留顶层的3个全连接网络
                                                  input_shape=(299, 299, 3),  ## 输入层的尺寸
                                                  )
    # add a global spatial average pooling layer
    x = model_inception_v3.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(6, activation='softmax')(x)

    # this is the model we will train
    final_inception_v3 = Model(inputs=model_inception_v3.input, outputs=predictions)
    for layer in final_inception_v3.layers[:-4]:
        layer.trainable = False

    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    final_inception_v3.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return final_inception_v3


'''
History = final_vgg16.fit(train_X[:-1000], train_y_[:-1000],
                    batch_size=32,
                    epochs=3,
                    shuffle=True,
                    validation_data=(train_X[-1000:], train_y_[-1000:]),
#                     verbose=0,
                   )

final_vgg16.save('final_vgg16.h5')
'''