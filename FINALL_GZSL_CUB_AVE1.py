##################
### 文件名：FINALL_GZSL_CUB_AVE
### 创建时间： 2019.4.27
### 版本：
###
### 说明： ADD average acc calculate
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
import scipy.io as sio
import Assistant
from Similary_Choice import Similary_choice_finall

##########################################################################
#################    Result saved setting       ##########################
##########################################################################
yita_list = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 0.4, 0.4, 0.4, 0.4, 0.4 ])

for jjj in range(len(yita_list)):
    time_stamp = Assistant.get_time_for_name()

    save_name = 'FINALL_GZSL_CUB--'+time_stamp

    log_dir = '/home/lee/Desktop/Code/GZSL_FINALLL_VERSION/result/CUB-best_result/'

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    log_name_html = log_dir+save_name+'.html'
    log_name_fig =  log_dir+save_name+'.png'
    log_lossacc = log_dir+save_name+'.txt'
    save_Enet_name = log_dir+save_name+'Embedding.pth'
    save_Cnet_name = log_dir+save_name+'Classify.pth'

    print(log_name_html)
    print(log_name_fig)

    ##########################################################################
    #################   1/  HYPER PARAMETERS        ##########################
    ##########################################################################
    optim_method = 'Adam'
    scheduler_method = 'SetpLR'

    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        device = 'cuda'
        torch.cuda.empty_cache()
    else :
        device ='cpu'

    best_acc = 0  # best val accuracy
    LR = 0.0001   #fixed
    LR_C = 0.001
    GAMMA = 0.1
    WEIGHT_DECAY = 1e-3
    WEIGHT_DECAY_C = 1e-5

    if scheduler_method == 'SetpLR':
        STEP_SIZE = 50
    elif scheduler_method=='MultiStepLR':
        STEP_SIZE = [100,400,700]

    BATCH_IN_EPOCH = 15
    EPOCH_NUM = 100

    BATCH_SIZE_TRAIN = 500
    BATCH_SIZE_VAL   = 500

    # yita = 0.3
    yita = yita_list[jjj]

    NUM_OF_SIMILAR = 3
    NUM_OF_MIDDLE_LAYER = 312
    LAMBDA = 1
    AMPLIFY = 2

    debug_info = ' loop yita ' + str(yita)

    dis_f = 0

    print('==================================================')
    print('Start time:',time_stamp)
    print('Config:  ')
    print('Optim:',optim_method)
    print('LR:          ',LR       ,'  ')
    print('LR_C:        ',LR_C       ,'  ')
    print('GAMMA:       ',GAMMA    ,'  ')
    print('WEIGHT_DECAY:',WEIGHT_DECAY ,'  ')
    print('WEIGHT_DECAY_classify:',WEIGHT_DECAY_C ,'  ')
    print('STEP_SIZE   :',STEP_SIZE    ,'  ')
    print('BATCH_SIZE_TRAIN:',BATCH_SIZE_TRAIN ,'  ')
    print('BATCH_SIZE_VAL  :',BATCH_SIZE_VAL   ,'  ')
    print('BATCH_IN_EPOCH  :',BATCH_IN_EPOCH   ,'  ')
    print('EPOCH_NUM       :',EPOCH_NUM    ,'  ')
    print('YITA            :',yita    ,'  ')
    print('NUM_SIMILAR     :',NUM_OF_SIMILAR    ,'  ')
    print('LAMBDA          :',LAMBDA    ,'  ')
    print('AMPLIFY         :',AMPLIFY    ,'  ')
    print()
    print(debug_info)
    print('=================================================')

    ########################################################################################
    ####################   Dataset    ######################################################
    ########################################################################################
    print('===>>Prepairing data<<===')

    dataset = '/home/lee/Desktop/Datasets/1_ZSL/PS/CUB'
    image_embedding = 'res101'
    class_embedding = 'att_splits'

    Image_Mat = sio.loadmat(dataset + "/" + image_embedding + ".mat")
    Split_Mat = sio.loadmat(dataset + "/" + class_embedding + ".mat")

    ### 属性
    Attribute = Split_Mat['att'].T

    NUM_OF_CLASS , NUM_OF_MIDDLE_LAYER = Attribute.shape

    Attribute_all = Attribute
    a_mean = Attribute_all.mean()
    a_max = Attribute_all.max()
    a_min = Attribute_all.min()

    Attribute_all = AMPLIFY* ((Attribute_all-a_min)/(a_max - a_min) -0.5 )

    Attribute_all = torch.from_numpy(Attribute_all).float()

    ### 所有类别特征和标号
    Features_all = Image_Mat['features'].T
    Labels_all   = Image_Mat['labels'].astype(int).squeeze() - 1

    ###可见类序号和不可见类序号
    train_seen_sequence_num  = Split_Mat['trainval_loc'].squeeze() - 1
    test_seen_sequence_num   = Split_Mat['test_seen_loc'].squeeze() - 1
    test_unseen_sequence_num = Split_Mat['test_unseen_loc'].squeeze() - 1


    Train_Seen_Features   = Features_all[train_seen_sequence_num]  # train_features
    Train_Seen_Labels     = Labels_all[train_seen_sequence_num].astype(int)  # train_label

    Test_Seen_Features = Features_all[test_seen_sequence_num]  # test_seen_feature
    Test_Seen_Labels   = Labels_all[test_seen_sequence_num].astype(int)  # test_seen_label

    Test_Unseen_Features = Features_all[test_unseen_sequence_num]  # test_feature
    Test_Unseen_Labels   = Labels_all[test_unseen_sequence_num].astype(int)  # test_label


    SEEN_class_dict = np.unique(Train_Seen_Labels)
    UNSEEN_class_dict = np.unique(Test_Unseen_Labels)  # test_id

    label_to_indices_SEEN = {label: np.where(Labels_all == label)[0] for label in SEEN_class_dict}
    label_to_indices_UNSEEN  = {label: np.where(Labels_all == label)[0] for label in UNSEEN_class_dict}

    ########################################################################
    ## 制作数据集   array转tensor   tensor转dataset   dataset转dataloader  经验证正确
    Features_all = torch.from_numpy(Features_all)
    Labels_all = torch.from_numpy(Labels_all.reshape(Labels_all.shape[0]))

    Test_Seen_Feature = torch.from_numpy(Test_Seen_Features)
    Test_Seen_Labels= torch.from_numpy(Test_Seen_Labels.reshape(Test_Seen_Labels.shape[0]))

    Test_Unseen_Features = torch.from_numpy(Test_Unseen_Features)
    Test_Unseen_Labels= torch.from_numpy(Test_Unseen_Labels.reshape(Test_Unseen_Labels.shape[0]))



    TEST_set_seen = TensorDataset(Test_Seen_Feature,Test_Seen_Labels)
    TEST_set_unseen = TensorDataset(Test_Unseen_Features,Test_Unseen_Labels)
            #dataloader 只用于验证
    TEST_loader_seen = DataLoader(TEST_set_seen,batch_size=BATCH_SIZE_VAL,shuffle=True)
    TEST_loader_unseen = DataLoader(TEST_set_unseen,batch_size=BATCH_SIZE_VAL,shuffle=True)


    Similary_dict = Similary_choice_finall(Attribute,SEEN_class_dict,NUM_OF_SIMILAR)
    Similary_dict.similarity_show()
    ##########################################################################
    ###########       Network architecture and optimizer        ##############
    ##########################################################################

    class EmbeddingNet(nn.Module):
        def __init__(self,num_of_embedding):
            super(EmbeddingNet, self).__init__()
            self.embed = nn.Sequential(
                nn.Linear(in_features=2048, out_features=1024, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=1024, out_features=512, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=512, out_features=num_of_embedding, bias=True),
            )

        def forward(self, x):

            x = self.embed(x)
            return x


    class ClassifyNet(nn.Module):
        """docstring for RelationNetwork"""
        def __init__(self,innput_num ,output_num):
            super(ClassifyNet, self).__init__()

            self.fc = nn.Linear(innput_num,output_num)

        def forward(self,x):
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x


    def Compose_Net( x , rule_data):

        out = torch.zeros([1,NUM_OF_MIDDLE_LAYER])
        out = out.cuda()
        for i in range(len(rule_data)):
            out = out + (x[i] *rule_data[i])

        return out

    torch.cuda.empty_cache()

    Embedding_Net = EmbeddingNet(NUM_OF_MIDDLE_LAYER)
    # Compose_Net = ComposeNet()
    Classify_Net = ClassifyNet(NUM_OF_MIDDLE_LAYER, NUM_OF_CLASS)

    # nn.init.xavier_uniform(Embedding_Net.embed.weight.data)
    # nn.init.xavier_uniform(Classify_Net.fc.weight.data)

    Embedding_Net = Embedding_Net.cuda()
    Classify_Net = Classify_Net.cuda()

    criterion = nn.CrossEntropyLoss()
    mse  = nn.MSELoss()


    if optim_method == 'SGD':
        EN_optimizer = optim.SGD(Embedding_Net.parameters(), lr=LR,momentum=0.9, weight_decay=WEIGHT_DECAY)
        CN_optimizer = optim.SGD(Classify_Net.parameters(), lr=LR_C, momentum=0.9,  weight_decay=WEIGHT_DECAY_C)

    elif optim_method == 'Adam':
        EN_optimizer = optim.Adam(Embedding_Net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        CN_optimizer = optim.Adam(Classify_Net.parameters(), lr=LR_C,  weight_decay=WEIGHT_DECAY_C)
    else:
        print('Error, no optim method select')
        plt.waitforbuttonpress()

    if scheduler_method == 'SetpLR':
        EN_scheduler = optim.lr_scheduler.StepLR(EN_optimizer, step_size=STEP_SIZE, gamma=GAMMA)
        CN_scheduler = optim.lr_scheduler.StepLR(CN_optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    elif scheduler_method=='MultiStepLR':
        EN_scheduler = optim.lr_scheduler.MultiStepLR(EN_optimizer, milestones= STEP_SIZE, gamma=GAMMA)
        CN_scheduler = optim.lr_scheduler.MultiStepLR(CN_optimizer, milestones=STEP_SIZE, gamma=GAMMA)
    else:
        print('Error, no Scheduler method select')
        plt.waitforbuttonpress()
    ##########################################################################
    ###################       Training and validate        ###################
    ##########################################################################



    best_acc_seen = 0
    best_acc_ave_seen = 0
    best_seen_poisiton = 0
    best_seen_ave_poisiton = 0
    best_acc_unseen = 0
    best_acc_ave_unseen = 0
    best_unseen_ave_poisiton = 0
    best_H_mean = 0
    best_H_mean_ave = 0
    best_h_poisiton = 0
    best_h_poisiton_ave = 0

    last_20_seen_acc = 0
    last_20_unseen_acc = 0
    last_20_h_mean = 0


    curve = Assistant.loss_curve_for_71(EPOCH_NUM)

    seen_number = int(yita*BATCH_SIZE_TRAIN)
    unseen_number = BATCH_SIZE_TRAIN - seen_number
    for current_epoch_num in range(EPOCH_NUM):
        ##################################################################
        ######## Training Stage  #########################################
        EN_scheduler.step()
        CN_scheduler.step()

        train_loss = 0
        train_loss_classify = 0
        train_loss_attribute = 0
        train_correct = 0
        train_total = 0

        Embedding_Net.train()
        Classify_Net.train()

        train_labels = np.zeros(BATCH_SIZE_TRAIN)
        train_inputs = np.zeros((BATCH_SIZE_TRAIN, 1, NUM_OF_MIDDLE_LAYER))
        ##  训练
        for batch_num in range(BATCH_IN_EPOCH):
            start_time = time.time()
            # zero the parameter gradients
            EN_optimizer.zero_grad()
            CN_optimizer.zero_grad()

            ## 训练阶段，随机选择一组label值，然后按这些值随机在其对应的样本序号字典里抽取样本
            # train_labels = np.random.randint(0,49,(BATCH_SIZE_TRAIN))
            t_l_seen = np.random.choice(SEEN_class_dict, seen_number)
            t_l_unseen = np.random.choice(UNSEEN_class_dict, unseen_number)
            train_labels = np.hstack([t_l_seen,t_l_unseen])
            np.random.shuffle(train_labels)
            # print('Label generated:',train_labels)
            ## 嵌入网络的中间输出结果，为85维向量，可扩展为属性回归器

            Attribute_label = Attribute_all[train_labels]
            Attribute_label = Attribute_label.cuda()

            middle_embed = torch.Tensor(BATCH_SIZE_TRAIN,1,NUM_OF_MIDDLE_LAYER)
            middle_embed = middle_embed.cuda()

            for i in range(BATCH_SIZE_TRAIN):  ## 对随机选择的各个类进行逐个计算
                item = train_labels[i]

                if item in SEEN_class_dict:  ##可见类，直接送入嵌入网络
                    # print('Choosed seen label: %d' % (item))
                    sample_seq_num = random.choice(label_to_indices_SEEN[item])

                    input_v = Features_all[sample_seq_num]
                    input_v = input_v.cuda()
                    out = Embedding_Net(input_v.float())

                else:   ## 未见类，选择相似样本列表
                    similar_info = Similary_dict.get_similar_info(item)
                    similar_sample = similar_info.ms_class   ### 获取相似样本类别号
                    similar_value  = similar_info.ms_value
                    similar_value /= similar_value.sum()

                    out_u = torch.zeros([NUM_OF_SIMILAR,NUM_OF_MIDDLE_LAYER])
                    out_u = out_u.cuda()
                    for j in range(NUM_OF_SIMILAR):
                        if similar_sample[j] in UNSEEN_class_dict:
                            print('Picking up unseen classes, program will be killed')
                            print('Picking up unseen classes, program will be killed')
                            print('Picking up unseen classes, program will be killed')
                            print('Picking up unseen classes, program will be killed')
                            plt.waitforbuttonpress()
                        sample_seq_num = random.choice(label_to_indices_SEEN[similar_sample[j]])
                        input_v = Features_all[sample_seq_num]
                        input_v = input_v.cuda()

                        out_u[j] = Embedding_Net(input_v.float())

                    out = Compose_Net(out_u,similar_value)

                    # out = out_u
                middle_embed[i] = out
            ## forward compute

            loss_attribute = LAMBDA * mse(middle_embed, Attribute_label)

            outputs = Classify_Net(middle_embed)

            train_labels = torch.from_numpy(train_labels).cuda()
            loss_classify = criterion(outputs, train_labels)

            loss  = loss_attribute + loss_classify

            loss.backward()
            # if current_epoch_num < 500:
            EN_optimizer.step()
            CN_optimizer.step()

            train_loss += loss.item()
            train_loss_classify += loss_classify
            train_loss_attribute += loss_attribute

            _, predicted = outputs.max(1)
            train_total += train_labels.size(0)
            train_correct += predicted.eq(train_labels).sum().item()

            batch_time = time.time() - start_time

            # print('===============================')
            # print('predeicted', predicted)
            # print('labels    ', train_labels)
            # print('correct num of this epoch', predicted.eq(train_labels).sum().item())
            # print('===============================')

            print(
                'Training| Epoch: %d | Batch: %d | Loss: %.3f | Loss_Attribute: %.3f | Loss_classify: %.3f | Acc: %.3f | Time: %s'
                % (current_epoch_num, batch_num, train_loss / (batch_num + 1), train_loss_attribute / (batch_num + 1),
                   train_loss_classify / (batch_num + 1),
                   100. * train_correct / train_total, Assistant.format_time(batch_time)))

        train_acc = train_correct / train_total

        curve.train_loss[current_epoch_num] = train_loss / (batch_num + 1)
        curve.train_acc[current_epoch_num] = train_acc
        ################################################################################################################
        #### test stage

        with torch.no_grad():
            Embedding_Net.eval()
            Classify_Net.eval()

            ##############   Seen class test    #############################################################################
            start_time = time.time()

            S_test_loss = 0
            S_test_loss_a = 0
            S_test_loss_c = 0
            S_test_correct = 0
            S_test_total = 0

            seen_acc = 0
            seen_acc_ave = 0

            ave_pc_total = np.zeros(NUM_OF_CLASS, dtype=np.uint32)
            ave_pc_correct = np.zeros(NUM_OF_CLASS, dtype=np.uint32)
            ave_pc_accpc = np.zeros(NUM_OF_CLASS, dtype=np.float32)

            ##  可见类样本验证
            for batch_num, (S_inputs_test, S_test_labels) in enumerate(TEST_loader_seen):

                Attribute_label = Attribute_all[S_test_labels]
                Attribute_label = Attribute_label.cuda()
                S_inputs_test = S_inputs_test.cuda()
                S_test_labels = S_test_labels.cuda()

                middle = Embedding_Net(S_inputs_test.float())
                S_outputs_test = Classify_Net(middle)

                loss_attribute = LAMBDA * mse(middle,Attribute_label)
                loss_classify = criterion(S_outputs_test, S_test_labels)
                loss = loss_attribute + loss_classify
                S_test_loss += loss.item()
                S_test_loss_a += loss_attribute.item()
                S_test_loss_c += loss_classify.item()

                _, predicted = S_outputs_test.max(1)
                S_test_total += S_test_labels.size(0)
                S_test_correct += predicted.eq(S_test_labels).sum().item()

                for i in range(len(S_test_labels)):
                    class_number = S_test_labels[i].item()
                    ave_pc_total[class_number] += 1
                    if S_test_labels[i].item() == predicted[i].item():
                        ave_pc_correct[class_number] += 1
            seen_acc_ave = 0
            for i in SEEN_class_dict:
                ave_pc_accpc[i] = ave_pc_correct[i] / ave_pc_total[i]
                seen_acc_ave += ave_pc_accpc[i]
            seen_acc_ave = seen_acc_ave / len(SEEN_class_dict)

            seen_loss = S_test_loss / (batch_num + 1)
            seen_acc = S_test_correct / S_test_total

            curve.test_seen_loss[current_epoch_num] = seen_loss
            curve.test_seen_acc[current_epoch_num] = seen_acc
            curve.test_seen_ave_acc[current_epoch_num] = seen_acc_ave

            batch_time = time.time() - start_time
            print(
                'Testing Seen classes:  | Epoch: %d | batch num: %d | Loss: %.3f | Loss_Attribute: %.3f | Loss_classify: %.3f | H1-Acc: %.3f | AVE-Acc: %.3f | Time: %s'
                % (current_epoch_num, batch_num, S_test_loss / (batch_num + 1),
                   S_test_loss_a / (batch_num + 1),
                   S_test_loss_c / (batch_num + 1),
                   seen_acc * 100,
                   seen_acc_ave * 100,
                   Assistant.format_time(batch_time)))

            ##############   Unseen class test    #############################################################################
            ##############   Unseen class test    #############################################################################
            start_time = time.time()

            Z_test_loss = 0
            Z_test_loss_a = 0
            Z_test_loss_c = 0
            Z_test_correct = 0
            Z_test_total = 0

            unseen_acc = 0
            unseen_acc_ave = 0
            ave_pc_total = np.zeros(NUM_OF_CLASS, dtype=np.uint32)
            ave_pc_correct = np.zeros(NUM_OF_CLASS, dtype=np.uint32)
            ave_pc_accpc = np.zeros(NUM_OF_CLASS, dtype=np.float32)

            ## 不可见类样本验证
            for batch_num, (Z_inputs_test, Z_test_labels) in enumerate(TEST_loader_unseen):

                Attribute_label = Attribute_all[Z_test_labels]
                Attribute_label = Attribute_label.cuda()
                Z_inputs_test = Z_inputs_test.cuda()
                Z_test_labels = Z_test_labels.cuda()
                middle = Embedding_Net(Z_inputs_test.float())
                Z_outputs_test = Classify_Net(middle)

                loss_attribute = LAMBDA * mse(middle, Attribute_label)
                loss_classify = criterion(Z_outputs_test, Z_test_labels)
                loss = loss_attribute + loss_classify

                Z_test_loss += loss.item()
                Z_test_loss_a += loss_attribute.item()
                Z_test_loss_c += loss_classify.item()


                _, predicted = Z_outputs_test.max(1)
                Z_test_total += Z_test_labels.size(0)
                Z_test_correct += predicted.eq(Z_test_labels).sum().item()

                # calculate average acc
                for i in range(len(Z_test_labels)):
                    class_number = Z_test_labels[i].item()
                    ave_pc_total[class_number] += 1
                    if Z_test_labels[i].item() == predicted[i].item():
                        ave_pc_correct[class_number] += 1

            unseen_acc_ave = 0
            for i in UNSEEN_class_dict:
                ave_pc_accpc[i] = ave_pc_correct[i] / ave_pc_total[i]
                unseen_acc_ave += ave_pc_accpc[i]
            unseen_acc_ave = unseen_acc_ave / len(UNSEEN_class_dict)

            unseen_acc = Z_test_correct / Z_test_total

            h_mean = (2.0 * unseen_acc * seen_acc) / (seen_acc + unseen_acc)
            h_mean_ave = (2.0 * unseen_acc_ave * seen_acc_ave) / (seen_acc_ave + unseen_acc_ave)

            curve.test_unseen_loss[current_epoch_num] = Z_test_loss / (batch_num + 1)
            curve.test_unseen_acc[current_epoch_num] = Z_test_correct / Z_test_total
            curve.test_unseen_ave_acc[current_epoch_num] = unseen_acc_ave

            curve.harmonic_mean[current_epoch_num] = h_mean
            curve.harmonic_mean_ave[current_epoch_num] = h_mean_ave

            batch_time = time.time() - start_time
            print(
                'Testing Unseen classes:| Epoch: %d | batch num: %d | Loss: %.3f | Loss_Attribute: %.3f | Loss_classify: %.3f | H1-Acc: %.3f | H mean: %.3f | AVE-Acc: %.3f | AVE-H mean: %.3f |  Time: %s'
                % (current_epoch_num, batch_num, Z_test_loss / (batch_num + 1),
                   Z_test_loss_a / (batch_num + 1),
                   Z_test_loss_c / (batch_num + 1),
                   unseen_acc * 100,
                   h_mean * 100,
                   unseen_acc_ave * 100,
                   h_mean_ave * 100,
                   Assistant.format_time(batch_time)))
            print()

            ## 转存可见类的最高准确率及其位置
            if best_acc_seen < seen_acc:
                best_acc_seen = seen_acc
                best_seen_poisiton = current_epoch_num
            ## 转存不可见类的最高准确率及其位置
            if best_acc_unseen < unseen_acc:
                best_acc_unseen = unseen_acc
                best_unseen_poisiton = current_epoch_num
            ## 转存不可见类的最高准确率及其位置
            if best_H_mean < h_mean:
                best_H_mean = h_mean
                best_h_poisiton = current_epoch_num

            ## 转存可见类的最高准确率及其位置
            if best_acc_ave_seen < seen_acc_ave:
                best_acc_ave_seen = seen_acc_ave
                best_seen_ave_poisiton = current_epoch_num
            ## 转存不可见类的最高准确率及其位置
            if best_acc_ave_unseen < unseen_acc_ave:
                best_acc_ave_unseen = unseen_acc_ave
                best_unseen_ave_poisiton = current_epoch_num
            ## 转存不可见类的最高准确率及其位置
            if best_H_mean_ave < h_mean_ave:
                best_H_mean_ave = h_mean_ave
                best_h_poisiton_ave = current_epoch_num

        # 显示结果
        if dis_f > 0:
            if current_epoch_num % dis_f == 0:
                curve.display_all(title1='Loss curve Epoch:%d'%(current_epoch_num+1),
                          title2='Acc curve Epoch:%d'%(current_epoch_num+1) ,
                          saved=False,save_name=log_name_fig)

    print('===>> Finish trainning <<===')

    curve.display_all(title1='Loss curve ', title2='Acc curve ',
                          saved=True, save_name=log_name_fig)
    curve.save_txt(log_lossacc)



    #############################################################################################################
    ###    Display the result         ##################################
    #############################################################################################################


    last_20_h_mean = curve.harmonic_mean[EPOCH_NUM-21:EPOCH_NUM-1].mean()
    last_20_h_mean_ave = curve.harmonic_mean_ave[EPOCH_NUM-21:EPOCH_NUM-1].mean()

    last_20_seen_acc = curve.test_seen_acc[EPOCH_NUM-21:EPOCH_NUM-1].mean()
    last_20_seen_acc_ave = curve.test_seen_ave_acc[EPOCH_NUM-21:EPOCH_NUM-1].mean()

    last_20_unseen_acc = curve.test_unseen_acc[EPOCH_NUM-21:EPOCH_NUM-1].mean()
    last_20_unseen_acc_ave = curve.test_unseen_ave_acc[EPOCH_NUM-21:EPOCH_NUM-1].mean()

    print()
    print()
    print()
    print('==================================================')
    print('Start time:',time_stamp)
    print('Config:  ')
    print('Optim:',optim_method)
    print('LR:          ',LR       ,'  ')
    print('LR_C:        ',LR_C       ,'  ')
    print('LAMBDA:      ',LAMBDA       ,'  ')
    print('GAMMA:       ',GAMMA    ,'  ')
    print('WEIGHT_DECAY:',WEIGHT_DECAY ,'  ')
    print('WEIGHT_DECAY_classify:',WEIGHT_DECAY_C ,'  ')
    print('STEP_SIZE   :',STEP_SIZE    ,'  ')
    print('BATCH_SIZE_TRAIN:',BATCH_SIZE_TRAIN ,'  ')
    print('BATCH_SIZE_VAL  :',BATCH_SIZE_VAL   ,'  ')
    print('BATCH_IN_EPOCH  :',BATCH_IN_EPOCH   ,'  ')
    print('EPOCH_NUM       :',EPOCH_NUM    ,'  ')
    print('YITA            :',yita    ,'  ')
    print('NUM_SIMILAR     :',NUM_OF_SIMILAR    ,'  ')
    print()
    print('Best S :' , best_acc_seen,  '| Pos: ',best_seen_poisiton,\
                    '| U ACC:', curve.test_unseen_acc[best_seen_poisiton],
                    '| H mean:',curve.harmonic_mean[best_seen_poisiton]
          )
    print('Best U :' , best_acc_unseen,'| Pos: ',best_unseen_poisiton , \
                    '| S ACC:', curve.test_seen_acc[best_unseen_poisiton],
                    '| H mean:',curve.harmonic_mean[best_unseen_poisiton]
          )
    print('Best H :' , best_H_mean,    '| Pos: ',best_h_poisiton , \
                    '| S ACC:', curve.test_seen_acc[best_h_poisiton],
                    '| U ACC:', curve.test_unseen_acc[best_h_poisiton]
          )
    print()
    print('Best S AVE :' , best_acc_ave_seen,  '| Pos: ',best_seen_ave_poisiton,\
                        '| U ACC:', curve.test_unseen_ave_acc[best_seen_ave_poisiton],
                        '| H mean:',curve.harmonic_mean_ave[best_seen_ave_poisiton]
          )
    print('Best U AVE  :' , best_acc_ave_unseen,'| Pos: ',best_unseen_ave_poisiton , \
                        '| S ACC:', curve.test_seen_ave_acc[best_unseen_ave_poisiton],
                        '| H mean:',curve.harmonic_mean_ave[best_unseen_ave_poisiton]
          )
    print('Best H AVE  :' , best_H_mean_ave,    '| Pos: ',best_h_poisiton_ave , \
                        '| S ACC:', curve.test_seen_ave_acc[best_h_poisiton_ave],
                        '| U ACC:', curve.test_unseen_ave_acc[best_h_poisiton_ave]
          )
    print()

    print('Last 20 epoch S acc mean:', last_20_seen_acc_ave)
    print('Last 20 epoch U acc mean:', last_20_unseen_acc_ave)
    print('Last 20 epoch H  mean   :', last_20_h_mean_ave)

    print('Last 20 epoch S acc mean:', last_20_seen_acc)
    print('Last 20 epoch U acc mean:', last_20_unseen_acc)
    print('Last 20 epoch H  mean   :', last_20_h_mean)
    print(debug_info)
    print('==================================================')

    log_file = open(log_name_html,'w')

    log_file.write('Start time:'+time_stamp+'<br />')
    log_file.write('Config'+'<br />')
    log_file.write('Optim:'+optim_method+'<br />')
    log_file.write('scheduler:'+scheduler_method+'<br />')
    log_file.write('<br />')
    log_file.write('LR:          '+str(LR)+'<br />')
    log_file.write('LR_C:        '+str(LR_C)+'<br />')
    log_file.write('LAMBDA:      '+str(LAMBDA)+'<br />')
    log_file.write('GAMMA:       '+str(GAMMA)+'<br />')
    log_file.write('WEIGHT_DECAY:'+str(WEIGHT_DECAY)+'<br />')
    log_file.write('WEIGHT_DECAY_c:'+str(WEIGHT_DECAY_C)+'<br />')
    log_file.write('<br />')
    log_file.write('STEP_SIZE   :'+str(STEP_SIZE)+'<br />')
    log_file.write('BATCH_SIZE_TRAIN:'+str(BATCH_SIZE_TRAIN)+'<br />')
    log_file.write('BATCH_SIZE_VAL  :'+str(BATCH_SIZE_VAL)+'<br />')
    log_file.write('BATCH_IN_EPOCH  :'+str(BATCH_IN_EPOCH)+'<br />')
    log_file.write('EPOCH_NUM       :'+str(EPOCH_NUM)+'<br />')
    log_file.write('<br />')
    log_file.write('YITA            :'+str(yita)+'<br />')
    log_file.write('NUM_SIMILAR     :'+str(NUM_OF_SIMILAR)+' <br />')
    log_file.write('AMPLIFY         :'+str(AMPLIFY)+' <br />')
    log_file.write('<br />')
    log_file.write(debug_info + '<br />')
    log_file.write('<br />')

    log_file.write('Best S :' +str(best_acc_seen)+'| Pos: '+str(best_seen_poisiton) )
    log_file.write('| U ACC:' +str(curve.test_unseen_acc[best_seen_poisiton]) )
    log_file.write('| H mean:'+str(curve.harmonic_mean[best_seen_poisiton])+'<br />')

    log_file.write('Best U :' +str(best_acc_unseen)+'| Pos: '+str(best_unseen_poisiton))
    log_file.write('| S ACC:' +str(curve.test_seen_acc[best_unseen_poisiton]) )
    log_file.write('| H mean:'+str(curve.harmonic_mean[best_unseen_poisiton]) +'<br />')

    log_file.write('Best H :' +str(best_H_mean)+    '| Pos: '+str(best_h_poisiton) )
    log_file.write('| S ACC:' +str(curve.test_seen_acc[best_h_poisiton]) )
    log_file.write('| U ACC:' +str(curve.test_unseen_acc[best_h_poisiton]) +'<br />')


    log_file.write('Best S AVE :' +str(best_acc_ave_seen)+'| Pos: '+str(best_seen_ave_poisiton) )
    log_file.write('| U ACC AVE :' +str(curve.test_unseen_ave_acc[best_seen_ave_poisiton]) )
    log_file.write('| H mean AVE :'+str(curve.harmonic_mean_ave[best_seen_ave_poisiton])+'<br />')

    log_file.write('Best U AVE :' +str(best_acc_ave_unseen)+'| Pos: '+str(best_unseen_ave_poisiton))
    log_file.write('| S ACC AVE :' +str(curve.test_seen_ave_acc[best_unseen_ave_poisiton]) )
    log_file.write('| H mean AVE :'+str(curve.harmonic_mean_ave[best_unseen_ave_poisiton]) +'<br />')

    log_file.write('Best H AVE :' +str(best_H_mean_ave)+    '| Pos: '+str(best_h_poisiton_ave) )
    log_file.write('| S ACC AVE :' +str(curve.test_seen_ave_acc[best_h_poisiton_ave]) )
    log_file.write('| U ACC AVE :' +str(curve.test_unseen_ave_acc[best_h_poisiton_ave]) +'<br />')

    log_file.write('<br />')
    log_file.write('Last 20 epoch S acc mean:'+str(last_20_seen_acc)+'<br />')
    log_file.write('Last 20 epoch U acc mean:'+str(last_20_unseen_acc)+'<br />')
    log_file.write('Last 20 epoch H  mean   :'+str(last_20_h_mean)+'<br />')
    log_file.write('Last 20 epoch S acc AVE mean:'+str(last_20_seen_acc_ave)+'<br />')
    log_file.write('Last 20 epoch U acc AVE mean:'+str(last_20_unseen_acc_ave)+'<br />')
    log_file.write('Last 20 epoch H  AVE mean   :'+str(last_20_h_mean_ave)+'<br />')
    log_file.write('<br />')

    log_file.write('<img src = "'+log_name_fig+'">')
    log_file.close()

    torch.save(Embedding_Net, save_Enet_name)
    torch.save(Classify_Net, save_Cnet_name)

    ##########################################################################
    #########################       Main        ##############################
    ##########################################################################










