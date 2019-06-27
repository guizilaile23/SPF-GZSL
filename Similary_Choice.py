import torch
from scipy import spatial
import matplotlib.pyplot as plt
import numpy as np
import heapq

def distance_compute(x1,x2):

    result = x1-x2

    return result

class Similar_Info:
    def __init__(self,similar_item_num,attribute_length):

        self.ms_class = np.zeros(similar_item_num)
        self.mu_class = np.zeros(similar_item_num)
        self.ms_value = np.zeros(similar_item_num)
        self.mu_value = np.zeros(similar_item_num)
        self.similar_attribute = np.zeros([similar_item_num,attribute_length])
        self.unlikely_attribute = np.zeros([similar_item_num,attribute_length])

class Similary_choice:

    ##################################################################################
    def __init__(self,attr_file_name, global_label_dir, visiable_dicr_dir ):

        ###全局label   存成一个字典，名字：label
        global_label = {}
        for item in open(global_label_dir):
            item = item.split()
            name = item[1]
            g_label = int(item[0]) - 1
            global_label[name] = g_label
        # print(global_label)

        ###  可见类 存成一个字典， 全局label：1
        visiable_dict = np.zeros(shape=40, dtype=int)
        i = 0
        for item in open(visiable_dicr_dir):
            item = item.replace('\n', '')
            visiable_dict[i] = global_label[item]
            i = i + 1
        # print(visiable_dict)

        ## 读取属性信息
        self.attribute = np.loadtxt(attr_file_name, dtype=np.float64)  #读取到的属性信息
        ## 属性向量归一化
        # a_mean = self.attribute.mean()
        # a_max = self.attribute.max()
        # a_min = self.attribute.min()
        # self.attribute = (self.attribute-a_mean)/(a_max - a_min)

        ## 相关参数初始化
        self.max_similar_num = 5

        class_num = self.attribute.shape[0]                    #类别数量  AWA2：50类
        self.attr_num = self.attribute.shape[1]                     #属性条目数量  85条
        similar_metric = np.zeros([class_num,class_num])   #相似矩阵
        self.similar_metric_norm = np.zeros([class_num,class_num]) #归一化相似矩阵


        self.ms_attribute = np.zeros([class_num,self.max_similar_num,self.attr_num])
        self.mu_attribute = np.zeros([class_num,self.max_similar_num,self.attr_num])
        self.ms_class = np.zeros(shape=[class_num,self.max_similar_num] ,dtype=int)
        self.mu_class = np.zeros(shape=[class_num,self.max_similar_num] ,dtype=int)
        self.ms_value = np.zeros(shape=[class_num,self.max_similar_num] )
        self.mu_value = np.zeros(shape=[class_num,self.max_similar_num] )

        ## 计算相似度矩阵 compute the similariy met
        for i in range(class_num):      #计算距离
            vector_b_1 = self.attribute[i]
            for j in range(class_num):
                vector_b_2 = self.attribute[j]
                # similar_metric[i][j] = np.sqrt(np.sum(np.square(vector_b_1 - vector_b_2)))
                similar_metric[i][j] = spatial.distance.cosine(vector_b_1,vector_b_2)
        ### normalization
        similar_metric = 1-similar_metric/similar_metric.max()

        self.similar_metric_norm = similar_metric.copy()  #归一化后得到相似性矩阵

        ### 计算最相似样本与最不相似样本
        ###    vector为第i个类别与其他所有类的相似性关系，需要找到最大的5个，即得到一个array，50*5，表示第i个类的5个最大相似样本
        for i in range(class_num):
            vector = similar_metric[i]
            vector_mean = vector.mean()
            vector[i] = 0
            for j in range(self.max_similar_num):  #循环查找5次，且必须在可见类中查找
                max = 0
                max_position = 0
                for k in visiable_dict:   #在可见类样本中，查找最相似的类别
                    if max <= vector[k]:
                        max = vector[k]
                        max_position = k

                vector[max_position] = 0
                self.ms_class[i][j] = max_position
                self.ms_value[i][j] = max
                self.ms_attribute[i][j] = self.attribute[k]

    ########################################################################################

    def similarity_show(self,title = None):
        plt.figure()
        plt.imshow(self.similar_metric_norm)
        plt.pause(0.001)

    def get_attribute(self,index):
        return self.attribute[index]


    def get_similar_info(self,index):

        item = Similar_Info(self.max_similar_num,self.attr_num)
        item.ms_class = self.ms_class[index]
        item.ms_value = self.ms_value[index]
        item.similar_attribute = self.ms_attribute[index]

        return item



class Similary_choice_finall:

    ##################################################################################
    def __init__(self,Attribute, Seen_class_dict, num_of_similar ):

        ###  可见类 存成一个字典， 全局label：1
        visiable_dict = Seen_class_dict
        # print(visiable_dict)

        ## 读取属性信息
        self.attribute = Attribute  #读取到的属性信息
        ## 属性向量归一化
        a_mean = self.attribute.mean()
        a_max = self.attribute.max()
        a_min = self.attribute.min()
        self.attribute = (self.attribute-a_min)/(a_max - a_min)

        ## 相关参数初始化
        self.max_similar_num = num_of_similar

        class_num = self.attribute.shape[0]                    #类别数量
        self.attr_num = self.attribute.shape[1]                     #属性条目数量
        similar_metric = np.zeros([class_num,class_num])   #相似矩阵
        self.similar_metric_norm = np.zeros([class_num,class_num]) #归一化相似矩阵


        self.ms_attribute = np.zeros([class_num,self.max_similar_num,self.attr_num])
        self.mu_attribute = np.zeros([class_num,self.max_similar_num,self.attr_num])
        self.ms_class = np.zeros(shape=[class_num,self.max_similar_num] ,dtype=int)
        self.mu_class = np.zeros(shape=[class_num,self.max_similar_num] ,dtype=int)
        self.ms_value = np.zeros(shape=[class_num,self.max_similar_num] )
        self.mu_value = np.zeros(shape=[class_num,self.max_similar_num] )

        ## 计算相似度矩阵 compute the similariy met
        for i in range(class_num):      #计算距离
            vector_b_1 = self.attribute[i]
            for j in range(class_num):
                vector_b_2 = self.attribute[j]
                # similar_metric[i][j] = np.sqrt(np.sum(np.square(vector_b_1 - vector_b_2)))
                similar_metric[i][j] = spatial.distance.euclidean(vector_b_1,vector_b_2)
        ### normalization
        similar_metric = 1-similar_metric/similar_metric.max()

        self.similar_metric_norm = similar_metric.copy()  #归一化后得到相似性矩阵

        ### 计算最相似样本与最不相似样本
        ###    vector为第i个类别与其他所有类的相似性关系，需要找到最大的5个，即得到一个array，50*5，表示第i个类的5个最大相似样本
        for i in range(class_num):
            vector = similar_metric[i]
            vector_mean = vector.mean()
            vector[i] = 0
            for j in range(self.max_similar_num):  #循环查找5次，且必须在可见类中查找
                max = 0
                max_position = 0
                for k in visiable_dict:   #在可见类样本中，查找最相似的类别
                    if max <= vector[k]:
                        max = vector[k]
                        max_position = k

                vector[max_position] = 0
                self.ms_class[i][j] = max_position
                self.ms_value[i][j] = max
                self.ms_attribute[i][j] = self.attribute[k]

    ########################################################################################

    def similarity_show(self,title = None):
        plt.figure(figsize=[6,6])
        plt.title(title)
        plt.imshow(self.similar_metric_norm)
        plt.pause(0.001)

    def get_attribute(self,index):
        return self.attribute[index]

    def get_similarity(self):

        return self.similar_metric_norm

    def get_similar_info(self,index):

        item = Similar_Info(self.max_similar_num,self.attr_num)
        item.ms_class = self.ms_class[index]
        item.ms_value = self.ms_value[index]
        item.similar_attribute = self.ms_attribute[index]

        return item