# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 21:15:03 2022

@author: PO KAI
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

def gradicent(lr,b,x,c,y,err):
    p = 1/(1+np.exp(-(np.dot(b,x.T)+c)))
    b = b - lr*np.mean((p - y).T*x,axis=0)
    c = c - lr*np.mean(p - y)
    err.append(np.mean(-(y*np.log(p)+(1-y)*np.log(1-p))))
    return b,c,err

def accuracy(b,x,c,y):
  y_pred=1/(1+np.exp(-(np.dot(b,x.T)+c)))
  y_pred[y_pred>=0.5]=1   
  y_pred[y_pred<0.5]=0
  correct=len(y_pred[y_pred==y])
  return (correct/len(y.T))*100   

def predict(w,x):
    b,c=w
    return 1/(1+np.exp(-(np.dot(b,x.T)+c)))

            
if __name__ == '__main__':
    data = pd.read_csv('./dataset/pushup.csv')
    data['class'] = data['class'].replace(['up','down'],[1,0])
    #shuffle
    data = data.sample(frac=1).reset_index(drop=True)
    
    l = data.shape[0]
    train_data = data.iloc[:int(l*0.7)]
    test_data = data.iloc[int(l*0.7):]
    
    train_data = train_data.sample(frac=1).reset_index(drop=True)
    
    x_train = train_data.drop(['class'],axis=1)
    y_train = train_data['class']
    x_test = test_data.drop(['class'],axis=1)
    y_test = test_data['class']
    
    y_train = y_train.astype('float64')
    
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)
    
    
    error = []
    acc = []
    b = np.random.randn(1,132)
    c = random.random()
    lr = 0.5
    epochs = 150
    
    for i in range(epochs):
      b,c,error = gradicent(lr,b,x_train,c,y_train,error)
      print("epoch : ",i,"\nError : ",error[-1],"\nAccuracy : ",accuracy(b,x_train,c,y_train))
      acc.append(accuracy(b,x_train,c,y_train))
     
      
    plt.plot(acc)
    plt.savefig('resource/opt/acc.png')
    plt.close()
    plt.plot(error)
    plt.savefig('resource/opt/plt.png')
    plt.close()
    
    np.savetxt('weight.txt',np.append(b,c))