# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 09:12:43 2019

@author: Administrator
"""
import numpy as np
import pickle,os 

def load_CIFAR_batch(filename):
    """load single batch of cifar"""
    with open(filename,'rb') as f:
        data_dict=pickle.load(f,encoding='bytes')#反序列化
        images=data_dict[b'data']
        labels=data_dict[b'labels']
        #把原始数据结构调整为：BCWH
        images=images.reshape(10000,3,32,32)
        #tensorflow处理图像数据的结构：BWHC
        images=images.transpose(0,2,3,1)
        labels=np.array(labels)
        return images,labels

def load_CIFAR_data(data_dir):
    """load CIFAR data"""
    images_train=[]
    labels_train=[]
    for i in range(5):
        f=os.path.join(data_dir,'data_batch_%d' %(i+1))
        print('loading',f)
        #调用load_CIFAR_batch()获取批量的图像及其对应的标签
        image_batch,label_batch=load_CIFAR_batch(f)
        images_train.append(image_batch)
        labels_train.append(label_batch)
        del image_batch,label_batch
    Xtrain=np.concatenate(images_train)
    Ytrain=np.concatenate(labels_train)
    Xtest,Ytest=load_CIFAR_batch(os.path.join(data_dir,'test_batch')) 
    print('finished loading CIFAR-10 data')
    return Xtrain,Ytrain,Xtest,Ytest

data_dir='data/cifar-10-batches-py'
Xtrain,Ytrain,Xtest,Ytest=load_CIFAR_data(data_dir)
        
        
        
        
        
        
        
        