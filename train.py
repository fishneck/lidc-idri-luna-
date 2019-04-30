import models,config,os
import numpy as np
from keras.applications import vgg16,resnet50,inception_v3
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import cv2
import matplotlib.pyplot as plt

cw ={}
X = []
Y = []

inception_size = (299,299)
vgg_size = (224,224)
resnet_size = (224,224)

#with open(config.path_to_error_file, 'a') as f:
#    f.write("---Fail to load npy:" + file + '\n')
def train_inception():
    for i in range(6):
        for file in os.listdir(config.path_to_cls_npy + str(i) + '/'):
            try:
                img = np.load(config.path_to_cls_npy + str(i) + '/' + file)
                #print(img)
                img = cv2.resize(img, inception_size, interpolation=cv2.INTER_CUBIC)
                img = np.stack((img,) * 3, axis=-1)
                X.append(img.tolist())
                Y.append(i)


            except EOFError:
                img = np.load(config.path_to_cls_npy + str(i) + '/' + file)
                print(len(X)+'fff')
                img = cv2.resize(img, inception_size, interpolation=cv2.INTER_CUBIC)
                img = np.stack((img,) * 3, axis=-1)
                X.append(img.tolist())
                Y.append(i)
                if len(X)==935:
                    print(config.path_to_cls_npy + str(i) + '/' + file)
        cw[i] = len(os.listdir(config.path_to_cls_npy + str(i) + '/'))

    trainX = np.array(X)
    trainY = np.array(Y)
    for i in range(6):
        cw[i] = round(len(trainY) / cw[i])
    val_num = round(0.1 * len(trainY))
    trainY = to_categorical(trainY, 6)

    print(np.shape(trainY))
    print(trainY)
    model = models.nn_inception_v3()

    trainX = inception_v3.preprocess_input(trainX)
    History = model.fit(trainX[:-val_num], trainY[:-val_num],
                        batch_size=2048,
                        epochs=5,
                        shuffle=True,
                        validation_data=(trainX[-val_num:], trainY[-val_num:]),
                        class_weight=cw,
                        )
    model.save('inception_v3.h5')
    return History

History  = train_inception()
fig = plt.figure()
plt.plot(History.history['acc'])
plt.plot(History.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='lower left')
#
fig.savefig('performance.png')
