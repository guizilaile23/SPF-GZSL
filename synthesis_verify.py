##################
### 文件名：finall_GZSL_AWA2_AVE1
### 创建时间： 2019.4.25 19:53
### 版本： V2
###
### 说明： add the average acc calculate method
### 结果：
###
##################

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset,DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

import os
import time
import copy
import random

import Assistant

import scipy.io as sio
from Similary_Choice import Similary_choice_finall

##########################################################################
#################    Result saved setting       ##########################
##########################################################################

dataset = '/home/lee/Desktop/Datasets/1_ZSL/PS/CUB'

image_embedding = 'res101'
class_embedding = 'att_splits'

Image_Mat = sio.loadmat(dataset + "/" + image_embedding + ".mat")
Split_Mat = sio.loadmat(dataset + "/" + class_embedding + ".mat")

### 属性
Attribute = Split_Mat['original_att'].T

NUM_OF_CLASS,NUM_OF_MIDDLE_LAYER = Attribute.shape

Attribute_all = Attribute

a_max = Attribute_all.max()
a_min = Attribute_all.min()
Attribute_all = (Attribute_all-a_min)/(a_max - a_min)

### 所有类别特征和标号
Features_all = Image_Mat['features'].T
Labels_all   = Image_Mat['labels'].astype(int).squeeze() - 1

###可见类序号和不可见类序号
train_seen_sequence_num  = Split_Mat['trainval_loc'].squeeze() - 1
test_seen_sequence_num   = Split_Mat['test_seen_loc'].squeeze() - 1
test_unseen_sequence_num = Split_Mat['test_unseen_loc'].squeeze() - 1

Train_Seen_Labels     = Labels_all[train_seen_sequence_num].astype(int)  # train_label
Test_Unseen_Labels   = Labels_all[test_unseen_sequence_num].astype(int)  # test_label


SEEN_class_dict = np.unique(Train_Seen_Labels)
UNSEEN_class_dict = np.unique(Test_Unseen_Labels)  # test_id


def Compose_Net(x, rule_data):
    out = np.zeros([1, NUM_OF_MIDDLE_LAYER])

    for i in range(len(rule_data)):
        out = out + (x[i] * rule_data[i])

    return out

Similary_dict = Similary_choice_finall(Attribute,SEEN_class_dict,5)
Similary_dict.similarity_show(title='original')

Attribute_syn = np.zeros(Attribute.shape,dtype=np.float32)

for i in range(NUM_OF_CLASS):
    if i in SEEN_class_dict:
        Attribute_syn[i] = Attribute_all[i]
    else:
        similar_info = Similary_dict.get_similar_info(i)
        similar_sample = similar_info.ms_class  ### 获取相似样本类别号
        similar_value = similar_info.ms_value
        similar_value /= similar_value.sum()

        input_a = Attribute_all[similar_sample]

        Attribute_syn[i] = Compose_Net(input_a, similar_value)


Similary_dict_s = Similary_choice_finall(Attribute_syn, SEEN_class_dict, 5)

Similary_dict_s.similarity_show(title='synthesis')

original_similar = Similary_dict.get_similarity()
synthesis_similar = Similary_dict_s.get_similarity()

np.save('original_similar.npy',original_similar)
np.save('synthesis_similar.npy',synthesis_similar)

np.save('attri.npy',Attribute_all)
np.save('attri-s.npy',Attribute_syn)










