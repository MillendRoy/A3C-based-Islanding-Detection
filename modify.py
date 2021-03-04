# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 10:28:28 2021

@author: Millend
"""
import pandas as pd
dataset=pd.read_csv('overall.csv')
for i in range(dataset.shape[0]):
    if (dataset['OUTCOME'][i]=='ISLAND'):
        dataset['OUTCOME'][i]=1
    else:
        dataset['OUTCOME'][i]=0

import numpy as np
data=dataset.iloc[:,:].values
np.random.shuffle(data)

datax=pd.DataFrame(data=data[:,:], columns=['F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F13','F14','F15','F16','F17','F18','F19','F20','F21','F22','OUTCOME'])
datax.to_csv('islanding_data.csv')