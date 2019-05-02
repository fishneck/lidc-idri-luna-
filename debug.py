import os
import numpy as np
import matplotlib.pyplot as plt

with open("all.txt",'r') as f:
    content = f.readlines()
pathlist =[]
neg_num = 8000
neg_count = 0
for i in range(len(content)):
    if neg_count < neg_num and content[i].split('//')[1].split('/')[0] == '0':
        print(neg_count)
        pathlist.append(content[i].strip('\n'))
        neg_count += 1

trainX = np.load("trainX.npy")
trainY = np.load("trainY.npy")
print(np.shape(trainY))

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

neg_img,neg_tag = load_img_from_np(pathlist,(299,299),color = 3)

trainX = np.concatenate(trainX,neg_img,axis=0)
trainY = np.concatenate(trainY,neg_tag,axis=0)