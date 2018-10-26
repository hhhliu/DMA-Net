#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 22:28:43 2017

@author: liuhuihui
"""

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation,Flatten,Lambda,Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import numpy as np
from keras.utils import np_utils
from keras.optimizers import SGD,Adam,Adadelta
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint,EarlyStopping,LearningRateScheduler
import matplotlib.pyplot as plt
import h5py
from keras.layers import Merge
from keras.layers.merge import Concatenate
import random
from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
import os
import scipy.io as sio
os.environ['CUDA_VISIBLE_DEVICES']='0'
    
def build_net(input_shape):
    
    model=Sequential()
    
    model.add(Conv2D(64,(11,11),strides=(2,2), data_format = 'channels_last', input_shape = input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001))
    
    model.add(Conv2D(64,(5,5),padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001))

    model.add(Conv2D(64,(3,3),strides=(1,1)))
    model.add(Activation('relu'))
    
    model.add(Conv2D(64,(3,3),strides=(1,1)))
    model.add(Activation('relu'))

    model.add(Flatten())
    
    model.add(Dense(1000))
    model.add(Activation('relu'))
    
    model.add(Dense(256))
    model.add(Activation('relu'))
    
    return model


def generate_sequences(n_batches, images1,images2,images3,images4,images5,labels,mean1,mean2,mean3,mean4,mean5,idxs):
    # generate batches of samples    
    while True:
        for bid in xrange(0, n_batches):
            if bid == n_batches - 1:
                batch_idxs = idxs[bid * batch_size:]
            else:
                batch_idxs = idxs[bid * batch_size: (bid + 1) * batch_size]
                
            batch_length=len(batch_idxs)
            X1= np.zeros((batch_length,224,224,3),np.float32)
            X2= np.zeros((batch_length,224,224,3),np.float32)
            X3= np.zeros((batch_length,224,224,3),np.float32)
            X4= np.zeros((batch_length,224,224,3),np.float32)
            X5= np.zeros((batch_length,224,224,3),np.float32)

            y = labels[batch_idxs]
            Y = np_utils.to_categorical(y, 2)
            
            count=0
            # for every image of a batch
            for i in batch_idxs:
                xx1 = images1[i, ...].astype(np.float32)
                xx1 -= mean1
                xx2 = images2[i, ...].astype(np.float32)
                xx2 -= mean2
                xx3 = images3[i, ...].astype(np.float32)
                xx3 -= mean3
                xx4 = images4[i, ...].astype(np.float32)
                xx4 -= mean4
                xx5 = images5[i, ...].astype(np.float32)
                xx5 -= mean5
                
                offset_x=random.randint(0,31)
                offset_y=random.randint(0,31)
                xx1=xx1[offset_x:offset_x+224,offset_y:offset_y+224,:]
                xx2=xx2[offset_x:offset_x+224,offset_y:offset_y+224,:]
                xx3=xx3[offset_x:offset_x+224,offset_y:offset_y+224,:]
                xx4=xx4[offset_x:offset_x+224,offset_y:offset_y+224,:]
                xx5=xx5[offset_x:offset_x+224,offset_y:offset_y+224,:]
                
                X1[count,...]=xx1
                X2[count,...]=xx2
                X3[count,...]=xx3
                X4[count,...]=xx4
                X5[count,...]=xx5
                count+=1    
            
            yield [X1,X2,X3,X4,X5],Y
            
def statistics_layer(xx):
    print xx.shape
    x_min=tf.reduce_min(xx,axis=1)
    print x_min.shape
    x_max=tf.reduce_max(xx,axis=1)
    x_sum=tf.reduce_sum(xx,axis=1)
    x_mean=tf.reduce_mean(xx,axis=1)
    x_sta=tf.concat([x_min,x_max,x_sum,x_mean],1)
    print x_sta.shape
    return x_sta
    
if __name__ == '__main__':

    input_shape = (224, 224, 3)
    num_classes = 2
    model1 = build_net(input_shape)
    model1.load_weights('/home/liuhuihui/CUHKPQ_DMA/ava/model_ini.h5',by_name=True)
    model2 = build_net(input_shape)
    model2.load_weights('/home/liuhuihui/CUHKPQ_DMA/ava/model_ini.h5',by_name=True)
    model3 = build_net(input_shape)
    model3.load_weights('/home/liuhuihui/CUHKPQ_DMA/ava/model_ini.h5',by_name=True)
    model4 = build_net(input_shape)
    model4.load_weights('/home/liuhuihui/CUHKPQ_DMA/ava/model_ini.h5',by_name=True)
    model5 = build_net(input_shape)
    model5.load_weights('/home/liuhuihui/CUHKPQ_DMA/ava/model_ini.h5',by_name=True)
    
    merged=Merge(layers=[model1,model2,model3,model4,model5],mode='concat',concat_axis=1)
    model=Sequential()
    model.add(merged)
    model.add(Reshape((5,256),input_shape=(1280,)))
    print 'merged'
    model.add(Lambda(statistics_layer,output_shape=None))
#    model.add(Flatten())
    model.add(Activation('relu'))
    model.add(Dense(2))
    model.add(Activation('softmax')) 
    
    sgd = SGD(lr=0.001, momentum=0.9, decay=1e-5, nesterov=True)
    model.compile(optimizer=sgd,loss='binary_crossentropy', metrics=['accuracy'])
    
    data_format = 'channels_last'
    batch_size = 64
    nb_epoch = 150
    validation_ratio = 0.1
    # training
    path_train1='/home/liuhuihui/DMA_Net/randomCrop/train_data1.hdf5'
    path_train2='/home/liuhuihui/DMA_Net/randomCrop/train_data2.hdf5'
    path_train3='/home/liuhuihui/DMA_Net/randomCrop/train_data3.hdf5'
    path_train4='/home/liuhuihui/DMA_Net/randomCrop/train_data4.hdf5'
    path_train5='/home/liuhuihui/DMA_Net/randomCrop/train_data5.hdf5'
    with h5py.File(path_train1, 'r') as train_file1,h5py.File(path_train2, 'r') as train_file2,h5py.File(path_train3, 'r') as train_file3,h5py.File(path_train4, 'r') as train_file4,h5py.File(path_train5, 'r') as train_file5:
        print 'enter'
        images1 = train_file1['images']
        labels = train_file1['labels']
        mean1= train_file1['mean'][...]
       
        idxs = range(len(images1))
        train_idxs = idxs[: int(len(images1) * (1 - validation_ratio))]
        validation_idxs = idxs[int(len(images1) * (1 - validation_ratio)) :]

        images2 = train_file2['images']
        mean2 = train_file2['mean'][...]
                              
        images3 = train_file3['images']
        mean3 = train_file3['mean'][...]
        
        images4 = train_file4['images']
        mean4 = train_file4['mean'][...]
        
        images5 = train_file5['images']
        mean5 = train_file5['mean'][...]
    
        n_train_batches = len(train_idxs) // batch_size
        n_remainder = len(train_idxs) % batch_size
        if n_remainder:
            n_train_batches = n_train_batches + 1
        train_generator = generate_sequences(n_train_batches,images1,images2,images3,images4,images5,labels,mean1,mean2,mean3,mean4,mean5,train_idxs)

        n_validation_batches = len(validation_idxs) // batch_size
        n_remainder = len(validation_idxs) % batch_size
        if n_remainder:
            n_validation_batches = n_validation_batches + 1
        validation_generator = generate_sequences(n_train_batches,images1,images2,images3,images4,images5,labels,mean1,mean2,mean3,mean4,mean5,train_idxs)

        
        ReduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10,
                 verbose=1, mode='min', epsilon=1e-4, cooldown=0, min_lr=0)
        
        Checkpoint = ModelCheckpoint(filepath='/home/liuhuihui/DMA_Net/data_avatest/weight_dma2.h5', monitor='val_loss', verbose=1,
                 save_best_only=True, save_weights_only=False,mode='min', period=1)
        
        earlyStop = EarlyStopping(monitor='val_loss',patience=15,verbose=1,mode='min')
#        
        history = model.fit_generator(generator = train_generator,
                            validation_data = validation_generator,
                            steps_per_epoch = n_train_batches,
                            validation_steps = n_validation_batches,
                            epochs = nb_epoch,
                            callbacks=[ReduceLR,Checkpoint,earlyStop])
    
    path_test1='/home/liuhuihui/DMA_Net/randomCrop/test_data1.hdf5'
    path_test2='/home/liuhuihui/DMA_Net/randomCrop/test_data2.hdf5'
    path_test3='/home/liuhuihui/DMA_Net/randomCrop/test_data3.hdf5'
    path_test4='/home/liuhuihui/DMA_Net/randomCrop/test_data4.hdf5'
    path_test5='/home/liuhuihui/DMA_Net/randomCrop/test_data5.hdf5'
    with h5py.File(path_test1, 'r') as test_file1,h5py.File(path_test2, 'r') as test_file2,h5py.File(path_test3, 'r') as test_file3,h5py.File(path_test4, 'r') as test_file4,h5py.File(path_test5, 'r') as test_file5:
        print 'enter'
        images1_test = test_file1['images']
        labels_test = test_file1['labels']
        
        idxs = range(len(images1_test))
        test_idxs = idxs[: int(len(images1_test))]
        
        images2_test = test_file2['images']
        images3_test = test_file3['images']          
        images4_test = test_file4['images']
        images5_test = test_file5['images']
                                    
        # testing sample generator
        n_test_batches = len(test_idxs) // batch_size
        n_remainder = len(test_idxs) % batch_size
        if n_remainder:
            n_test_batches = n_test_batches + 1
        test_generator = generate_sequences(n_test_batches,images1_test,images2_test,images3_test,images4_test,images5_test,labels_test,mean1,mean2,mean3,mean4,mean5,test_idxs)    

        predicted_test = model.predict_generator(generator=test_generator,steps=n_test_batches,verbose=1)      
        sio.savemat('/home/liuhuihui/DMA_Net/data_avatest/predicted_dma.mat',{'predicted':predicted_test})
        np.save('/home/liuhuihui/DMA_Net/data_avatest/predicted_dma.npy',predicted_test)
        
        score_test =model.evaluate_generator(generator=test_generator,steps=n_test_batches)
        sio.savemat('/home/liuhuihui/DMA_Net/data_avatest/score_dma.mat',{'score':score_test})
        np.save('/home/liuhuihui/DMA_Net/data_avatest/score_dam.npy',score_test)
