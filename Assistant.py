import torch

import time
import matplotlib.pyplot as plt
import time
import os
import copy

import numpy as np
import random


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


# class Curve:
#     def __init__(self, length,item_num,name_list):
#         self.length = length
#         self.item_num = item_num
#         self.name_list = name_list
#         self.curve = {}
#         self.color_dict = {}
#         self.colors = ['black','gray','lightcoral','brown','maroon','red','darksalmon','coral'
#         ,'orangered','darkorange','tan','goldenrod','gold','olive','yellow'
#         ,'olivedrab','green','seagreen','aquamarine','turquoise','cyan','deepskyblue'
#         ,'cornflowerblue','navy','blue','blueviolet','violet','purple','fuchsia'
#         ,'deeppink','hotpink']
#
#         for i in self.name_list:
#             self.curve[i] = np.zeros(self.length)
#             self.color_dict[i] = random.choice(self.colors)
#
#     # def save_txt(self, name):
#     #     result_txt = np.zeros([self.num, 4])
#     #     result_txt[:, 0] = self.train_loss
#     #     result_txt[:, 1] = self.val_loss
#     #     result_txt[:, 2] = self.train_acc
#     #     result_txt[:, 3] = self.val_acc
#     #     np.savetxt(name, result_txt, fmt='%.4f')
#
#     def display(self,title,x_name = None  ,y_name = None,):
#         plt.figure(1)
#         plt.cla()
#         plt.title(title)
#         plt.xlabel('epoch num')
#         plt.ylabel('loss value value')
#         for i in self.name_list:
#             plt.plot(self.curve[i], self.color_dict[i], label=i)
#
#         plt.legend()
#         # plt.show()
#         plt.pause(0.01)


class loss_curve:
    def __init__(self, total_num):
        self.num = total_num
        self.train_loss = np.zeros(self.num)
        self.val_loss = np.zeros(self.num)
        self.train_acc = np.zeros(self.num)
        self.val_acc = np.zeros(self.num)

    def save_txt(self, name):
        result_txt = np.zeros([self.num, 4])
        result_txt[:, 0] = self.train_loss
        result_txt[:, 1] = self.val_loss
        result_txt[:, 2] = self.train_acc
        result_txt[:, 3] = self.val_acc
        np.savetxt(name, result_txt, fmt='%.4f')

    def display(self):
        plt.figure(1)
        plt.cla()
        plt.title('trainning curve')
        plt.xlabel('epoch num')
        plt.ylabel('loss value value')
        plt.plot(self.train_loss, 'r', label='train loss')
        plt.plot(self.val_loss, 'deeppink', label='val loss')
        plt.plot(self.train_acc, 'g', label='train accuracy')
        plt.plot(self.val_acc, 'greenyellow', label='val accuracy')
        plt.legend()
        # plt.show()
        plt.pause(0.01)


class loss_curve_for_71:
    def __init__(self, total_num = 1):
        if total_num == None:
            total_num = 1
        self.num = total_num
        self.train_loss = np.zeros(self.num)
        self.train_acc = np.zeros(self.num)

        self.test_seen_loss = np.zeros(self.num)
        self.test_unseen_loss = np.zeros(self.num)

        self.test_seen_acc = np.zeros(self.num)
        self.test_seen_ave_acc = np.zeros(self.num)
        self.test_unseen_acc = np.zeros(self.num)
        self.test_unseen_ave_acc = np.zeros(self.num)

        self.harmonic_mean = np.zeros(self.num)
        self.harmonic_mean_ave = np.zeros(self.num)

    def save_txt(self, name):
        result_txt = np.zeros([self.num, 10])
        result_txt[:, 0] = self.train_loss
        result_txt[:, 1] = self.train_acc

        result_txt[:, 2] = self.test_seen_loss
        result_txt[:, 3] = self.test_seen_acc
        result_txt[:, 4] = self.test_seen_ave_acc

        result_txt[:, 5] = self.test_unseen_loss
        result_txt[:, 6] = self.test_unseen_acc
        result_txt[:, 7] = self.test_unseen_ave_acc

        result_txt[:, 8] = self.harmonic_mean
        result_txt[:, 9] = self.harmonic_mean_ave

        np.savetxt(name, result_txt, fmt='%.4f')

    def read_txt(self, name):

        # print('fuck this shit')
        result_txt = np.loadtxt(name)
        self.train_loss = result_txt[:,0]
        self.train_acc = result_txt[:,1]

        self.test_seen_loss = result_txt[:, 2]
        self.test_seen_acc =result_txt[:, 3]
        self.test_seen_ave_acc = result_txt[:, 4]

        self.test_unseen_loss = result_txt[:, 5]
        self.test_unseen_acc = result_txt[:, 6]
        self.test_unseen_ave_acc = result_txt[:, 7]

        self.harmonic_mean = result_txt[:, 8]
        self.harmonic_mean_ave = result_txt[:, 9]


    def display_all(self, title1,title2,saved,save_name):
        plt.figure(num=1,figsize=[20,8])
        plt.cla()

        plt.xlabel('epoch num')

        plt.subplot(121)
        plt.title(title1)
        plt.ylabel('loss value value')
        plt.grid(True)
        plt.plot(self.train_loss, color='deeppink',linestyle=':',marker='.', label='train loss')
        plt.plot(self.test_seen_loss, 'blue', linestyle=':',marker='.',label='seen_loss')
        plt.plot(self.test_unseen_loss, 'green',linestyle=':',marker='.', label='unseen_loss')
        plt.legend()
        plt.subplot(122)
        plt.title(title2)
        plt.ylim((0, 1))
        plt.yticks(np.linspace(0, 1, 21))
        plt.ylabel('acc')
        plt.grid(True)

        plt.plot(self.train_acc, 'deeppink', label='train_acc')

        plt.plot(self.test_seen_acc, 'blue', label='hit1 seen_acc %f' % self.test_seen_acc.max())
        plt.plot(self.test_seen_ave_acc, 'blue', linestyle=':', marker='o',
                 label='seen_ave_acc %f' % self.test_seen_ave_acc.max())

        plt.plot(self.test_unseen_acc, 'green', label='hit1 unseen_acc %f' % self.test_unseen_acc.max())
        plt.plot(self.test_unseen_ave_acc, 'green', linestyle=':', marker='o',
                 label='unseen_ave_acc %f' % self.test_unseen_ave_acc.max())

        plt.plot(self.harmonic_mean, 'purple', label='harmonic mean %f' % self.harmonic_mean.max())
        plt.plot(self.harmonic_mean_ave, 'purple', linestyle=':', marker='o',
                 label='harmonic mean ave %f' % self.harmonic_mean_ave.max())

        plt.legend()
        # plt.show()
        if saved == True:
            plt.savefig(save_name)
        plt.pause(0.01)




#####################################################################################################################################

def get_time_for_name():
    time_now = time.localtime()
    mon = time_now.tm_mon
    if mon <10:
        mon = '0'+str(mon)
    else:
        mon = str(mon)
    date = time_now.tm_mday
    if date <10:
        date = '0'+str(date)
    else:
        date = str(date)

    hour = time_now.tm_hour
    if hour < 10:
        hour = '0' + str(hour)
    else:
        hour = str(hour)
    minit = time_now.tm_min
    if minit < 10:
        minit = '0' + str(minit)
    else:
        minit = str(minit)
    sec = time_now.tm_sec
    if sec < 10:
        sec = '0' + str(sec)
    else:
        sec = str(sec)

    time_for_name = mon+date+'--'+hour+'-'+minit+'-'+sec

    return time_for_name

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)

    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def show_train_sample_siamese(data, label):
    images_so_far = 0
    img_0 = data[0]
    img_1 = data[1]
    labels = label

    sample_number = len(labels)
    for i in range(sample_number):

        images_so_far += 1
        ax = plt.subplot(1, sample_number, i+1)
        ax.axis('off')
        ax.set_title('labels: {} '.format(labels[i]))
        # ax.set_title('predicted: {} \n True_label: {}'.format(class_names[preds[j]],class_names[labels[j]]))
        plt.subplots_adjust(wspace =0.4,hspace=0.4)
        imshow(img_0.data[i])

        ax = plt.subplot(2, sample_number, i+sample_number+1)
        ax.axis('off')
        ax.set_title('labels: {} '.format(labels[i]))
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        imshow(img_1.data[i])

def show_sample_classification(data, label):
    images_so_far = 0
    img = data
    labels = label

    sample_number = len(labels)
    for i in range(sample_number):

        images_so_far += 1
        ax = plt.subplot(2, sample_number//2, i+1)
        ax.axis('off')
        ax.set_title('labels: {} '.format(labels[i]))
        # ax.set_title('predicted: {} \n True_label: {}'.format(class_names[preds[j]],class_names[labels[j]]))
        plt.subplots_adjust(wspace =0.4,hspace=0.4)
        imshow(img.data[i])




def visualize_model(model, num_images=25, data_set=None, name=None):
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    class_name = name

    for i, data in enumerate(data_set):
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()

        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images // 5, 5, images_so_far)
            ax.axis('off')
            if preds[j] == labels[j]:
                color = 'blue'
            else:
                color = 'red'
            ax.set_title('Predicted : {} \nTrue_label: {}'.format(class_name[preds[j]], class_name[labels[j]]),
                         color=color)
            plt.subplots_adjust(wspace=0.4, hspace=0.4)
            imshow(inputs.cpu().data[j])

            if images_so_far == num_images:
                return


class Similar_Info:
    def __init__(self,similar_item_num,attribute_length):
        self.ms_class = np.zeros(similar_item_num)
        self.mu_class = np.zeros(similar_item_num)
        self.ms_value = np.zeros(similar_item_num)
        self.mu_value = np.zeros(similar_item_num)
        self.similar_attribute = np.zeros([similar_item_num,attribute_length])
        self.unlikely_attribute = np.zeros([similar_item_num,attribute_length])
#
class Attritube_sample:

    ##################################################################################
    def __init__(self,attr_file_name, global_label_dir, visiable_dicr_dir ):
        self.num = 50
        self.max_similar_num = 5
        self.max = 0
        self.min = 0
        self.max_position = 0
        self.min_position = 0

        ###全局label   存成一个字典，名字：label
        global_label = {}
        for item in  open(global_label_dir):
            item = item.split()
            name = item[1]
            g_label = int(item[0]) - 1
            global_label[name] = g_label
        print(global_label)

        ###  可见类 存成一个字典， 全局label：1
        visiable_dicr = np.zeros(shape=40,dtype=int)
        i = 0
        for item in open(visiable_dicr_dir):
            item = item.replace('\n','')
            visiable_dicr[i] = global_label[item]
            i = i+1
        print(visiable_dicr)


        self.attribute = np.loadtxt(attr_file_name, dtype=np.float64)  #读取到的属性信息

        a_mean = self.attribute.mean()
        a_max = self.attribute.max()
        a_min = self.attribute.min()
        self.attribute = (self.attribute-a_mean)/(a_max - a_min)

        self.class_num = self.attribute.shape[0]                    #类别数量  AWA2：50类
        self.attr_num = self.attribute.shape[1]                     #属性条目数量  85条
        self.similar_metric = np.zeros([self.class_num,self.class_num])   #相似矩阵
        self.similar_metric_norm = np.zeros([self.class_num,self.class_num]) #归一化相似矩阵

        self.vector_b_1 = np.zeros(self.attr_num)
        self.vector_b_2 = np.zeros(self.attr_num)
        self.vector = np.zeros(self.class_num)

        self.ms_attribute = np.zeros([self.class_num,self.max_similar_num,self.attr_num])
        self.mu_attribute = np.zeros([self.class_num,self.max_similar_num,self.attr_num])
        self.ms_class = np.zeros(shape=[self.class_num,self.max_similar_num] ,dtype=int)
        self.mu_class = np.zeros(shape=[self.class_num,self.max_similar_num] ,dtype=int)
        self.ms_value = np.zeros(shape=[self.class_num,self.max_similar_num] )
        self.mu_value = np.zeros(shape=[self.class_num,self.max_similar_num] )
        ## compute the similariy met
        for i in range(self.class_num):      #计算距离
            self.vector_b_1 = self.attribute[i]

            for j in range(self.class_num):
                self.vector_b_2 = self.attribute[j]
                self.similar_metric[i][j] = np.sqrt(np.sum(np.square(self.vector_b_1 - self.vector_b_2)))

                if self.similar_metric[i][j] > self.max:
                    self.max = self.similar_metric[i][j]
        ### normalization
        for i in range(self.class_num):    #归一化
            for j in range(self.class_num):
                self.similar_metric[i][j] = 1 - (self.similar_metric[i][j] / self.max)
                # similar_result[i*10:i*10+9][j*10:j*10+9] = int(similar_metric[i][j])
        self.similar_metric_norm = self.similar_metric.copy()  #归一化后得到相似性矩阵

        ### 计算最相似样本与最不相似样本
        for i in range(self.class_num):
            self.vector = self.similar_metric[i]
            self.vector_mean = self.vector.mean()
            self.vector[i] = self.vector_mean
            print('vector mean:{:.6f}'.format(self.vector_mean))
            for j in range(self.max_similar_num):  #循环查找5次
                self.max = 0
                self.max_position = 0
                for k in visiable_dicr:   #在49个其他类中，查找最相似的类别
                # for k in range(self.class_num):   #在49个其他类中，查找最相似的类别
                    if self.max <= self.vector[k]:
                        self.max = self.vector[k]
                        self.max_position = k

                self.vector[self.max_position] = self.vector_mean
                self.ms_class[i][j] = self.max_position
                self.ms_value[i][j] = self.max
                self.ms_attribute[i][j] = self.attribute[k]

            for j in range(self.max_similar_num):  # 循环查找5次，找最小值
                self.min = 1
                self.min_position = 0
                for k in visiable_dicr:  # 在49个其他类中，查找最不相似的类别
                # for k in range(self.class_num):  # 在49个其他类中，查找最不相似的类别
                    if self.min >= self.vector[k]:
                        self.min = self.vector[k]
                        self.min_position = k

                self.vector[self.min_position] = self.vector_mean
                self.mu_class[i][j] = self.min_position    #保存类别i的5个最不相似样本的类别号
                self.mu_value[i][j] = self.min              #保存类别i的5个最不相似样本的距离
                self.mu_attribute[i][j] = self.attribute[k] #保存类别i的5个最不相似样本的属性信息

    ########################################################################################

    def similarity_show(self,title = None):
        plt.figure()
        plt.imshow(self.similar_metric_norm)
        plt.pause(1)

    def get_attribute(self,index):
        return self.attribute[index]


    def get_similar_info(self,index):

        item = Similar_Info(self.max_similar_num,self.attr_num)
        item.ms_class = self.ms_class[index]
        item.ms_value = self.ms_value[index]
        item.similar_attribute = self.ms_attribute[index]

        item.mu_class = self.mu_class[index]
        item.mu_value = self.mu_value[index]
        item.unlikely_attribute = self.mu_attribute[index]
        return item