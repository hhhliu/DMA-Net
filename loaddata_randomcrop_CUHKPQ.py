#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 21:50:56 2017

@author: liuhuihui
"""


import numpy as np
import h5py
import random
from keras.preprocessing import image
import os

outsize=256

def rangeCrop(path):
    img=image.load_img(path)
    (width,height)=img.size
    if width >= outsize & height >= outsize:
        x=random.randint(0,width-outsize)
        y=random.randint(0,height-outsize)
        new_img=img.crop((x,y,x+outsize,y+outsize))       
    else:
        new_img=img.resize((outsize,outsize))
        
    data=image.img_to_array(new_img)
    return data 

  
image_trainhigh_dir = '/home/liuhuihui/now_work/dataset/CUHKPQ/test_high/'
image_trainlow_dir = '/home/liuhuihui/now_work/dataset/CUHKPQ/test_low/'  

n_trainhigh=2262
n_trainlow=6580
n_train=n_trainhigh+n_trainlow
labels_trainhigh=np.ones((n_trainhigh,1))
labels_trainlow=np.zeros((n_trainlow,1))
labels_train=np.concatenate((labels_trainhigh,labels_trainlow),axis=0)

# channel last format following Tensorflow
trainhigh_shape = (n_trainhigh, 256, 256, 3)
trainlow_shape = (n_trainlow, 256, 256, 3)
train_shape=(n_train,256,256,3)

mean = np.zeros(train_shape[1:], np.float32)
n_images = float(n_train)
print n_images

dset_train =np.zeros((n_train,256,256,3),dtype=np.uint8)
idx=0
for root,dirs,files in os.walk(image_trainhigh_dir):
    for file in files:
        print idx+1
        addr=os.path.join(root,file)
        data=rangeCrop(addr)     
        dset_train[idx,:] = data 
        idx+=1
        mean += data / n_images

    
for root,dirs,files in os.walk(image_trainlow_dir):
    for file in files:
        print idx+1
        addr=os.path.join(root,file)
        data=rangeCrop(addr)     
        dset_train[idx,:] = data 
        idx+=1
        mean += data / n_images

print dset_train.shape
        
train_file = h5py.File('/home/liuhuihui/CUHKPQ_DMA/test/test_data1.hdf5', 'w')


perm = range(n_train)
random.seed(702)
random.shuffle(perm)
dset_train=dset_train[perm]
labels_train=labels_train[perm]

train_file.create_dataset("images", data = dset_train)
train_file.create_dataset("labels", data = labels_train)
train_file.create_dataset("mean", data = mean)
train_file.close()

