# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 13:15:40 2019

@author: Administrator
"""
from sklearn.preprocessing import OneHotEncoder
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
from time import time
import importData

tf.reset_default_graph()

data_dir='data/cifar-10-batches-py'
Xtrain,Ytrain,Xtest,Ytest=importData.load_CIFAR_data(data_dir)
#数据预处理
Xtrain_normalize=Xtrain.astype('float32')/255.0
Xtest_normalize=Xtest.astype('float32')/255.0
encoder=OneHotEncoder(sparse=False)
yy=[[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]]
encoder.fit(yy)
Ytrain_reshape=Ytrain.reshape(-1,1)
Ytrain_onehot=encoder.transform(Ytrain_reshape)
Ytest_reshape=Ytest.reshape(-1,1)
Ytest_onehot=encoder.transform(Ytest_reshape)

def weight(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.1),name='W')

def bias(shape):
    return tf.Variable(tf.constant(0.1,shape=shape),name='b')
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#输入层
#32x32图像，通道为3（RGB）
with tf.name_scope('input_layer'):
    x=tf.placeholder('float',shape=[None,32,32,3],name='x')
    
#第1个卷积层
#输入通道：3，输出通道：32，卷积后图像尺寸不变，还是32x32
with tf.name_scope('conv_1'):
    W1=weight([3,3,3,32])#[k_width,k_height,input_chn,output_chn]
    b1=bias([32])#与output_chn一致
    conv_1=conv2d(x,W1)+b1
    conv_1=tf.nn.relu(conv_1)

#第1个池化层
#将32x32图像缩小为16x16,池化不改变通道数量
with tf.name_scope('pool_1'):
    pool_1=max_pool_2x2(conv_1)

#第2个卷积层
#输入通道：32，输出通道：64，卷积后图像尺寸不变，依然是16x16
with tf.name_scope('conv_2'):
    W2=weight([3,3,32,64])
    b2=bias([64])
    conv_2= conv2d(pool_1,W2)+b2
    conv_2=tf.nn.relu(conv_2)
        
#第2个池化层
#将16x16图像缩小为8x8，池化不改变通道数量，因此依然是64个
with tf.name_scope('pool_2'):
    pool_2=max_pool_2x2(conv_2)
    
#全连接层
#将第2个池化层的64个8x8的图像转换为一维的向量，长度为64x8x8=4096
#128个神经元
with tf.name_scope('fc'):
    W3=weight([4096,128])
    b3=bias([128])
    flat=tf.reshape(pool_2,[-1,4096])
    h=tf.nn.relu(tf.matmul(flat,W3)+b3)
    h_dropout=tf.nn.dropout(h,keep_prob=0.8)#使训练时一些神经元不激活，防止过拟合

#输出层
#输出层共有10个神经元，对应到0-9这十个类别
with tf.name_scope('output_layer'):
    W4=weight([128,10])
    b4=bias([10])
    pred=tf.nn.softmax(tf.matmul(h_dropout,W4)+b4)
    
with tf.name_scope('optimizer'):
    y=tf.placeholder('float',shape=[None,10],name='label')
    loss_function=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
    optimizer=tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss_function)
   
with tf.name_scope('evaluation'):
    correct_prediction=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,'float'))
    
#启动会话
train_epochs=200
batch_size=50
total_batch=int(len(Xtrain)/batch_size)
epoch_list=[];accuracy_list=[];loss_list=[];
epoch=tf.Variable(0,name='epoch',trainable=False)
startTime=time()
sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)

#设置检查点存储目录
ckpt_dir='CIFAR10_log/'
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
    
saver=tf.train.Saver(max_to_keep=1)

#如果有检查点文件，读取最新的检查点文件，恢复各种变量值
ckpt=tf.train.latest_checkpoint(ckpt_dir)
if ckpt!=None:
    saver.restore(sess,ckpt)#加载所有参数
    #从这里开始就可以直接使用模型进行预测，或者接着继续训练模型
else:
    print('Training from scratch.')
    
#获取训练参数
start=sess.run(epoch)
print('Training starts from {} epoch.'.format(start+1))
    
def get_train_batch(number,batch_size):
    return Xtrain_normalize[number*batch_size:(number+1)*batch_size],\
           Ytrain_onehot[number*batch_size:(number+1)*batch_size]

for ep in range(start,train_epochs):
    for i in range(total_batch):
        batch_x,batch_y=get_train_batch(i,batch_size)
        sess.run(optimizer,feed_dict={x:batch_x,y:batch_y})
        if i%100 ==0:
            print('Step {}'.format(i),'finished')
            
    loss,acc=sess.run([loss_function,accuracy],feed_dict={x:batch_x,y:batch_y})
    epoch_list.append(ep+1)
    loss_list.append(loss)
    accuracy_list.append(acc)
    
    print('Train epoch:','%02d'%(sess.run(epoch)+1),'Loss=','{:.6f}'.format(loss),'Accuracy=',acc)
    
    saver.save(sess,ckpt_dir+'CIFAR10_cnn_model.ckpt',global_step=ep+1)
    sess.run(epoch.assign(ep+1))

duration=time()-startTime
print('Train finished takes:'+ str(duration))
    
#可视化损失值
import matplotlib.pyplot as plt
fig=plt.gcf()
fig.set_size_inches(4,2)
plt.plot(epoch_list,loss_list,label='loss')
plt.ylabel('loss');plt.xlabel("epoch");plt.legend(['loss'],loc='upper right') 
plt.show()  

#可视化准确率
plt.plot(epoch_list,accuracy_list,label='accuracy')
fig=plt.gcf()
fig.set_size_inches(4,2)
plt.ylim(0.1,1)
plt.ylabel('accuracy');plt.xlabel("epoch");plt.legend(['accuracy'],loc='upper right')
plt.show()   
    
#计算测试集上的准确率
test_total_batch=int(len(Xtest_normalize)/batch_size)
test_acc_sum=0.0
for i in range(test_total_batch):
    test_image_batch=Xtest_normalize[i*batch_size:(i+1)*batch_size]
    test_label_batch=Ytest_onehot[i*batch_size:(i+1)*batch_size]
    test_batch_acc=sess.run(accuracy,feed_dict={x:test_image_batch,y:test_label_batch})
    test_acc_sum +=test_batch_acc
test_acc=float(test_acc_sum/test_total_batch)
print('Test accuracy:{:.6f}'.format(test_acc))

#使用模型进行预测
test_pred=sess.run(pred,feed_dict={x:Xtest_normalize[:10]})
prediction_result=sess.run(tf.argmax(test_pred,1))
import visualization
visualization.plot_images_labels_prediction(Xtest,Ytest,prediction_result,0,10)













