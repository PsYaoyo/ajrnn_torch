# @Author:Ps_Y
# -*- coding = utf-8 -*-
# @Time : 2021-11-15 16:25
# @File : test.py
# @Software : PyCharm
import torch
from torch import nn
import grumodel
from torch.nn import functional as F
import utils
from torch import nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib
import os
import numpy as np
import argparse

##############---Config---################
Missing_value = 128.0
GPU = ''
train_data_filename = './results/data/50words/50words_TRAIN_20.csv'
test_data_filename = './results/data/50words/50words_TEST_20.csv'
num_steps = 0
class_num= 0
batch_size = 20
epoch = 10
lamda_D = 1
G_epoch = 5
hidden_size = 100
layer_num = 1
learning_rate =1e-4
lamda = 1
D_epoch = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


##############---Data---################
print ('Loading data && Transform data--------------------')
print (train_data_filename)
train_data, train_label = utils.load_data(train_data_filename)
# print('train_data:',train_data.shape,train_data) #(450,270)
# print('train_label:',train_label.shape,train_label) #(450,)
#For univariate
num_steps = train_data.shape[1] #序列长度T 270
input_dimension_size = 1  #输入维度
print("nums_steps:",num_steps)

train_label, num_classes = utils.transfer_labels(train_label) #1--50  ->  0--49
class_num = num_classes #50


print ('Train Label:', np.unique(train_label))
print ('Train data completed-------------')

test_data, test_labels = utils.load_data(test_data_filename)
# print('test_labels',test_labels.shape,test_labels) #455

test_label, test_classes = utils.transfer_labels(test_labels)
# print('test_label:',test_label.shape,test_label) #455
print ('Test data completed-------------')


##############---GRU---################
class GRUModel(nn.Module):
    def __init__(self,input_size,hidden_size):
        super().__init__()
        self.gru = nn.GRUCell(input_size,hidden_size)
        self.linear1 = nn.Linear(hidden_size,class_num)
        self.W = torch.nn.Parameter(torch.normal(mean=0.0,std=0.1,size=[hidden_size,input_dimension_size]))
        self.bias = torch.nn.Parameter(torch.tensor([0.1],dtype=torch.float32))
        self.h0 = torch.nn.Parameter(torch.zeros((batch_size,hidden_size)))


    def forward(self,input):
        outputs = list()
        for time_step in range(num_steps):
            if time_step == 0:
                ht = self.gru(input[time_step,:,:],self.h0)
                outputs.append(ht)
            else:
                comparison = input[time_step, :, :] == torch.tensor(Missing_value,dtype=torch.float32)
                x_hat = torch.mm(outputs[time_step-1],self.W) + self.bias
                current_input = torch.where(comparison, x_hat, input[time_step,:,:])
                ht = self.gru(current_input,ht)
                outputs.append(ht)

                label_target_hidden_output = outputs[-1]#最后一个时间步经过rnn后的ht  20,100
                prediction_target_hidden_output = outputs[:-1] #除去最后一个，前面t-1个ht
                prediction_hidden_output =torch.reshape( input = torch.concat(prediction_target_hidden_output,axis = 1),shape = [-1,hidden_size] ) #5380*100
                # print(prediction_hidden_output.shape) 5380,100
                prediction = torch.add(torch.mm(prediction_hidden_output,self.W),self.bias) #5380,1
                label_logits = self.linear1(label_target_hidden_output) #20,50
                # label_logits = F.softmax(label_logits,dim=1)

        return outputs,prediction,label_logits


##############---Discriminator---################
class Discriminator(nn.Module):
    def __init__(self,x):
        super().__init__()
        self.linear1 = nn.Linear(x.shape[1], x.shape[1])
        self.linear2 = nn.Linear(x.shape[1], int(x.shape[1])//2)
        self.linear3 = nn.Linear(int(x.shape[1])//2, x.shape[1])

    def forward(self,x):
        x1 = torch.tanh(self.linear1(x.to(torch.float32)))
        x2 = torch.tanh(self.linear2(x1))
        predict_mask = torch.sigmoid(self.linear3(x2))
        return predict_mask

##############---Loss---################
class MyCrossEntropy(torch.nn.Module):
    def __init__(self, scale=1.0):
        super(MyCrossEntropy, self).__init__()
        # self.device = device
        # self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels, tag):
        if tag == 'softmax':
            pred = F.log_softmax(pred, dim=1)
        else:
            pred = F.logsigmoid(pred)
        # label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        nce = -1 * torch.sum(labels * pred, dim=1)
        return self.scale * nce.mean()

'''Train'''
for i in range(epoch):
    for input, prediction_target, mask, label_target, _, batch_need_label in utils.next_batch(batch_size, train_data, train_label, True,input_dimension_size,num_steps, Trainable = True):
        input = torch.tensor(input,dtype=torch.float32)
        prediction_target = torch.tensor(prediction_target)
        mask = torch.tensor(mask)
        label_target = torch.tensor(label_target,dtype=torch.float32)

        # print(input[1,:,:])
        # print(input)

        gru = GRUModel(1,hidden_size)
        # cost = nn.CrossEntropyLoss() #内部自带log_softmax
        cost = MyCrossEntropy()
        optimizer_cls = torch.optim.Adam(gru.parameters(),lr=learning_rate)
        outputs, prediction, label_logits= gru(input)

        reg_loss = 0.0
        for i, var in enumerate(gru.parameters()):
            if i < 2:
                reg_loss += torch.norm(var,p=2)
        # print(reg_loss)
        # print(outputs[-1])
        # print(prediction.shape)  #5380,1
        # print("logits:",label_logits,"target:",label_target) #20*50  ok
        loss_cls = cost(label_logits,label_target,'softmax')
        # print(loss_cls)
        # print(label_target.shape)  #20*50

        prediction_targets =torch.reshape(input=prediction_target, shape=[-1, input_dimension_size])
        # print(prediction_targets) #5380 1  ok
        masks = torch.reshape(input=mask, shape=[-1, input_dimension_size])
        # print(masks)  ok
        loss_imp = torch.mean(torch.square((prediction_targets - prediction) * masks)) / batch_size  #MSE_loss imputation
        # print(loss_imp)  ok
        correct_predictions = torch.argmax(label_logits, dim=1) == torch.argmax(label_target, dim=1)  #bool 当前位置分类正确的坐标 logits里取最大，坐标相等acc
        accuracy = correct_predictions.float() #float32
        # print(accuracy) ok

        prediction = torch.reshape(input=prediction, shape=[-1, (num_steps - 1) * input_dimension_size]) #20,269
        M = torch.reshape(input=mask, shape=[-1, (num_steps - 1) * input_dimension_size])
        # print(M.shape) 20*269
        prediction_target = torch.reshape(input=prediction_targets, shape=[-1, (num_steps - 1) * input_dimension_size])
        real_pre = prediction * (1 - M) + prediction_target * M #20,269
        real_pre = torch.reshape(input=real_pre,shape=[batch_size,(num_steps-1)*input_dimension_size])
        # print(real_pre.shape) ok
        D = Discriminator(real_pre)
        predict_M = D(real_pre)
        # print(predict_M.shape)  20*269
        predict_M = torch.reshape(input=predict_M, shape=[-1, (num_steps - 1) * input_dimension_size])
        # print(predict_M.shape)  20*269
        predict_M = torch.sigmoid(predict_M)
        # print(predict_M)  ok
        # print(cost2(predict_M,M))
        D_loss = torch.mean(cost(predict_M, M, 'sigmoid'))
        # print(D_loss)
        G_loss = torch.mean(cost(predict_M, (1 - M), 'sigmoid') * (1 - M))
        # print(G_loss)
        optimizer_adv = torch.optim.Adam(D.parameters(), lr=learning_rate)
        batch_loss = loss_cls + lamda * loss_imp + 1e-4 * reg_loss #正则项
        total_loss = batch_loss + lamda_D * G_loss

        optimizer_adv.zero_grad()
        optimizer_cls.zero_grad()


        batch_loss.backward()


        optimizer_cls.step()
        optimizer_adv.step()

        # print(total_loss)
        total_l = []
        total_train_acc = []
        total_l.append(batch_loss.tolist())
        total_train_acc.append(accuracy.tolist())
    print("Loss: ", np.mean(total_l),"Train acc:", np.mean(np.array(total_train_acc).reshape(-1)))




















