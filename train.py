import models,config,os
import numpy as np
import pandas as pd
from keras.applications import vgg16,resnet50,inception_v3
from keras.utils.training_utils import multi_gpu_model
from keras.utils import to_categorical
import cv2,random,utils
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

cw ={}
X = []
Y = []

inception_size = (299,299)
vgg_size = (224,224)
resnet_size = (224,224)
neg_num = 8000
neg_count = 0
batchsize=128
#utils.save_file2csv(config.path_to_cls_npy, "all.csv")
with open("all.txt",'r') as f:
    content = f.readlines()
pathlist =[]
for i in range(len(content)):
    if content[i].split('//')[1].split('/')[0] != '0':
        pathlist.append(content[i].strip('\n'))
    elif neg_count < neg_num:
        print(neg_count)
        pathlist.append(content[i].strip('\n'))
        neg_count += 1
print(len(pathlist))
with open("classweight.txt",'r') as f:
    weights = f.readlines()
for i in range(len(weights)):
    if i == 0:
        cw[i] = round(len(pathlist) / neg_count)
    else:
        cw[i] = round(len(pathlist) / int(weights[i].strip('\n')))
#print(pathlist)
print(cw)

def load_img_from_np(paths,size,color = 3):
    imgs = []
    tags = []
    for i in range(len(paths)):
        try:
            img = np.load(paths[i])
            # print(img)
            img = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
            img = np.stack((img,) * color, axis=-1)
            imgs.append(img.tolist())
            tags.append(paths[i].split('//')[1].split('/')[0])

        except EOFError:
            img = np.load(paths[i])
            # print(img)
            img = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
            img = np.stack((img,) * color, axis=-1)
            imgs.append(img.tolist())
            tags.append(i)
    imgs = np.array(imgs)
    tags = np.array(tags)
    return imgs,tags


def get_batch(img_path, batch_size, size, preprocess ,mode, color = 3,tratio = 0.8, vratio = 0.2):
    #  mode="Train"  or "Val" or "test"
    #  tratio train-ration,vratio val-ratio, sum=1
    while 1:
        for i in range(0,len(img_path)-batch_size,batch_size):
            if mode == "Train":
                x, y = load_img_from_np(img_path[i:i + round(tratio*batch_size)], size, color)
            elif mode == "Val":
                x, y = load_img_from_np(img_path[i + round(tratio*batch_size) : i + round((tratio+vratio)*batch_size)], size, color)
            else:
                x, y = load_img_from_np(img_path[i + round((tratio+vratio)*batch_size) : i + batch_size], size, color)
        y = to_categorical(y, 6)
        if preprocess == "inception":
            x = inception_v3.preprocess_input(x)
            yield ({'input_1': x}, {'dense_2': y})
        elif preprocess == "vgg":
            x = vgg16.preprocess_input(x)
            yield ({'input_1': x}, {'dense_2': y})
        elif preprocess == "resnet":
            x = resnet50.preprocess_input(x)
            yield ({'input_1':x},{'dense_2':y})

def train_session(model_type):
    if model_type=='inception_v3':
        model = models.nn_inception_v3()
        History = model.fit_generator(
            generator=get_batch(pathlist, batchsize, inception_size, preprocess="inception", mode="Train"),
            steps_per_epoch=100,
            epochs=20,
            shuffle=True,
            validation_data=get_batch(pathlist, batchsize, inception_size, preprocess="inception", mode="Val"),
            validation_steps=100,
            max_queue_size=10,
            workers=4,
            class_weight=cw,
        )
    elif model_type=='resnet50':
        model = models.nn_resnet50()

        History = model.fit_generator(
            generator=get_batch(pathlist, batchsize, resnet_size, preprocess="resnet", mode="Train"),
            steps_per_epoch=100,
            epochs=20,
            shuffle=True,
            validation_data=get_batch(pathlist, batchsize, resnet_size, preprocess="resnet", mode="Val"),
            validation_steps=100,
            max_queue_size=10,
            workers=4,
            class_weight=cw,
        )
    elif model_type == 'vgg16':
        model = models.nn_vgg16()
        History = model.fit_generator(
            generator=get_batch(pathlist, batchsize, vgg_size, preprocess="vgg", mode="Train"),
            steps_per_epoch=100,
            epochs=20,
            shuffle=True,
            validation_data=get_batch(pathlist, batchsize, vgg_size, preprocess="vgg", mode="Val"),
            validation_steps=100,
            max_queue_size=10,
            workers=4,
            class_weight=cw,
        )

    return History


History = train_session('resnet50')



fig = plt.figure()
#print(History.history.keys())

plt.plot(History.history['binary_accuracy'])
plt.plot(History.history['val_binary_accuracy'])
plt.title('model binary accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
fig.savefig('evaluation1.png')

fig = plt.figure()
plt.plot(History.history['categorical_accuracy'])
plt.plot(History.history['val_categorical_accuracy'])
plt.title('model categorical accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
fig.savefig('evaluation2.png')

fig = plt.figure()
plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='lower left')
#
fig.savefig('performance.png')
