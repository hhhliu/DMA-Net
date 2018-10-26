#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 11:28:53 2017

@author: liuhuihui
"""

import numpy as np
import h5py
import random
from keras.preprocessing import image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

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
    
image_dir = '/home/liuhuihui/AVA/'
label_path = '/home/liuhuihui/AVAset/labels.txt'

indices=np.load('/home/liuhuihui/AVAset/indices.npy')
labels=np.load('/home/liuhuihui/AVAset/labels.npy')

# divide the samples into 80% train and 20% test
train_indices = indices[:int(0.8*len(indices))]
train_labels = labels[:int(0.8*len(labels))]

test_indices = indices[int(0.8*len(indices)):]
test_labels = labels[int(0.8*len(labels)):]

# channel last format following Tensorflow
train_shape = (len(train_indices), 256, 256, 3)
test_shape = (len(test_indices), 256, 256, 3)

print train_shape[1:]

mean = np.zeros(train_shape[1:], np.float32)
n_images = float(len(train_indices) + len(test_indices))
print n_images

# creat training data and save images
train_file = h5py.File('/home/liuhuihui/DMA_Net/randomCrop/train_data.hdf5', 'w')
dset = train_file.create_dataset("images", train_shape, np.uint8)
for idx, fid in enumerate(train_indices):
    if (idx + 1) % 1000 == 0:
        print 'Train data: {}/{}'.format(idx + 1, len(train_indices))
    addr = image_dir + fid + '.jpg'
    data=rangeCrop(addr)     
    dset[idx,:] = data 
    mean += data / n_images

# creat testing data and save images
test_file = h5py.File('/home/liuhuihui/DMA_Net/randomCrop/test_data.hdf5', 'w')
dset = test_file.create_dataset("images", test_shape, np.uint8)
for idx, fid in enumerate(test_indices):
    if (idx + 1) % 1000 == 0:
        print 'Test data: {}/{}'.format(idx + 1, len(test_indices))
    addr = image_dir + fid + '.jpg'
    data=rangeCrop(addr)
    dset[idx,:] = data
    mean += data / n_images

# save labels
train_file.create_dataset("labels", data = train_labels)
test_file.create_dataset("labels", data = test_labels)

# save mean value
train_file.create_dataset("mean", data = mean)
test_file.create_dataset("mean", data = mean)

train_file.close()
test_file.close()