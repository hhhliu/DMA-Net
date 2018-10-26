#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 20:08:09 2017

@author: liuhuihui
"""

import numpy as np
from keras.preprocessing import image
import h5py
import scipy.io as sio


path='/home/liuhuihui/DMA_Net/data_cuhkpq/predicted_test.npy'
predict_test=np.load(path)

predict=predict_test[:,1]
np.save('/home/liuhuihui/DMA_Net/data_cuhkpq/cuhkpq_scores.npy',predict)
sio.savemat('/home/liuhuihui/DMA_Net/data_cuhkpq/cuhkpq_scores.mat',{'cuhkpq_scores':predict})