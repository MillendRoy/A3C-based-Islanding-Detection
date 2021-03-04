# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 18:38:25 2021

@author: Millend
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class Enviro():
    def __init__(self):
        self.dataset=pd.read_excel('example_classification.xlsx')
        self.X=self.dataset.iloc[:,:-1].values
        self.y=self.dataset.iloc[:,-1].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, stratify= self.y, test_size = 0.25, random_state = 0)
        sc = StandardScaler()
        self.X_train = sc.fit_transform(self.X_train)
        self.X_test = sc.transform(self.X_test)

        self.action_space = len(np.unique(self.y))
        self.observation_space = self.X.shape[1]
        self.seed()
        self.state=None
        
    def seed(self,seed=None):
        np.random.seed(seed)
        
    def train_reset(self,seedx=None):
        self.seed(seedx)
        no=np.random.randint(self.X_train.shape[0])
        self.state = self.X_train[no]
        return np.array(self.state),no
        '''self.state=self.X[0]
        return np.array(self.state)'''
    
    def test_reset(self,seedx=None):
        self.seed(seedx)
        no=np.random.randint(self.X_test.shape[0])
        self.state = self.X_test[no]
        return np.array(self.state),no
        '''self.state=self.X[0]
        return np.array(self.state)'''

    def train_step(self, action,i,ix):
        self.state=self.X_train[(i+1)%(self.X_train.shape[0])]
        if((ix-1)==-1):
            z=self.X_train.shape[0]-1
        else:
            z=ix-1
        if (i==z):
            done=True
        else:
            done=False
    
        #ro=(list(self.y).count(0))/(list(self.y).count(1))
        if (action==0 and self.y_train[i]==0):
            reward = 1.0
        elif (action==1 and self.y_train[i]==1):
            reward = 1.0
        elif (action==0 and self.y_train[i]==1):
            reward = 0
        elif (action==1 and self.y_train[i]==0):
            reward = 0
            
        return np.array(self.state), reward, done, {}
    
    def test_step(self, action,i,ix):
        self.state=self.X_test[(i+1)%(self.X_test.shape[0])]
        if((ix-1)==-1):
            z=self.X_test.shape[0]-1
        else:
            z=ix-1
        if (i==z):
            done=True
        else:
            done=False
    
        #ro=(list(self.y).count(0))/(list(self.y).count(1))
        if (action==0 and self.y_test[i]==0):
            reward = 1.0
        elif (action==1 and self.y_test[i]==1):
            reward = 1.0
        elif (action==0 and self.y_test[i]==1):
            reward = 0
        elif (action==1 and self.y_test[i]==0):
            reward = 0
            
        return np.array(self.state), reward, done, {}