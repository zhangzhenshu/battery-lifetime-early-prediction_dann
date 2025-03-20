import torch
import torch.nn as nn
from torch.autograd import Function
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from scipy.io import loadmat
import math
import copy
import random
import matplotlib.pyplot as plt
import os
import sys
# from model_CNN import CNN
from model_dann import DANNModel

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

usegpu = torch.cuda.is_available()

def normalization(data):
    min = data.min(1)
    max = data.max(1)
    range = max - min
    m = data.shape[0]
    normData = data - np.tile(min, (1, m))
    normData = normData / np.tile(range, (1, m))
    return normData

def MAPE(true, pred):
    diff = np.abs(np.array(true)-np.array(pred))
    return np.mean(diff / true)

# soure data prepare
# load data
path_s = 'Data/deltaq124.mat'
data_s = loadmat(path_s)
deltaQ_s = data_s['deltaq']
deltaQ_s = np.transpose(deltaQ_s)
path_s = 'Data/eol124.mat'
life_s = loadmat(path_s)
life_s = life_s['bat_label']

valid_index_s = [122,113,109,105,104,93,79,67,	60,	55,	54,	48,	29,	24,	23,	21,	19,	14,	5,	2]
test_index_s = [121,107,106,89,84,	81,	75,	74,	70,	58,	50,	43,	41,	40,	33,	32,	20,	18,	12,	10]

no_train_index_s=valid_index_s+test_index_s
no_train_index_s.sort(reverse=True)
valid_index_s.sort(reverse=True)
test_index_s.sort(reverse=True)

trainingData_s = []
trainingLabel_s = []

for i in range(len(deltaQ_s)):
    trainingData_s.append(deltaQ_s[i,:].reshape(1,1000))
    trainingLabel_s.append(life_s[i])

for i, del_index in enumerate(no_train_index_s):
    del trainingData_s[del_index]
    del trainingLabel_s[del_index]


valData_s = []
valLabel_s = []

for i, index in enumerate(valid_index_s):
    valData_s.append(deltaQ_s[index,:].reshape(1,1000))
    valLabel_s.append(life_s[index])

# data normalization
for i in range(len(trainingData_s)):
    tmp = trainingData_s[i]
    trainingData_s[i] = normalization(tmp)


for i in range(len(valData_s)):
    tmp = valData_s[i]
    valData_s[i] = normalization(tmp)

# data augmentation
num = len(trainingData_s)
for n in range(20):
    for i in range(num):
        noise = np.random.normal(0, 1e-2, [1, 1000])
        traininData_n_s = trainingData_s[i] + noise
        trainingData_s.append(traininData_n_s)
        trainingLabel_s.append(trainingLabel_s[i])

for n in range(20):
    for i in range(num):
        noise = np.random.normal(0, 2e-2, [1, 1000])
        traininData_n_s = trainingData_s[i] + noise
        trainingData_s.append(traininData_n_s)
        trainingLabel_s.append(trainingLabel_s[i])

trainingData_s = np.array(trainingData_s, dtype=np.float32)
valData_s = np.array(valData_s, dtype=np.float32)

trainingLabel_s = np.array(trainingLabel_s, dtype=np.float32)
valLabel_s = np.array(valLabel_s, dtype=np.float32)

BATCH_SIZE = 128
# numpy to tensor
trainingData_s = TensorDataset(torch.from_numpy(trainingData_s),
                            torch.from_numpy(trainingLabel_s))
valData_s = TensorDataset(torch.from_numpy(valData_s),
                            torch.from_numpy(valLabel_s))

# tensor to DataLoader
train_dataloader_s = DataLoader(trainingData_s, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_dataloader_s = DataLoader(valData_s, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

# target data prepare
# load data
path_t = 'Data/deltaq45.mat'
data_t = loadmat(path_t)
deltaQ_t = data_t['deltaq']
deltaQ_t = np.transpose(deltaQ_t)
path_t = 'Data/eol45.mat'
life_t = loadmat(path_t)
life_t = life_t['eol']

valid_index_t=[42,40,33,31,30,18,15,9]
test_index_t=[34,28,27,21,17,16,12,4]

no_train_index_t=valid_index_t+test_index_t
no_train_index_t.sort(reverse=True)
valid_index_t.sort(reverse=True)
test_index_t.sort(reverse=True)

trainingData_t = []
trainingLabel_t = []

for i in range(len(deltaQ_t)):
    trainingData_t.append(deltaQ_t[i,:].reshape(1,1000))
    trainingLabel_t.append(life_t[i])

for i, del_index in enumerate(no_train_index_t):
    del trainingData_t[del_index]
    del trainingLabel_t[del_index]


valData_t = []
valLabel_t = []

for i, index in enumerate(valid_index_t):
    valData_t.append(deltaQ_t[index,:].reshape(1,1000))
    valLabel_t.append(life_t[index])

# data normalization
for i in range(len(trainingData_t)):
    tmp = trainingData_t[i]
    trainingData_t[i] = normalization(tmp)


for i in range(len(valData_t)):
    tmp = valData_t[i]
    valData_t[i] = normalization(tmp)

# data augmentation
num = len(trainingData_t)
for n in range(20):
    for i in range(num):
        noise = np.random.normal(0, 1e-2, [1, 1000])
        traininData_n_t = trainingData_t[i] + noise
        trainingData_t.append(traininData_n_t)
        trainingLabel_t.append(trainingLabel_t[i])

for n in range(20):
    for i in range(num):
        noise = np.random.normal(0, 2e-2, [1, 1000])
        traininData_n_t = trainingData_t[i] + noise
        trainingData_t.append(traininData_n_t)
        trainingLabel_t.append(trainingLabel_t[i])

trainingData_t = np.array(trainingData_t, dtype=np.float32)
valData_t = np.array(valData_t, dtype=np.float32)

trainingLabel_t = np.array(trainingLabel_t, dtype=np.float32)
valLabel_t = np.array(valLabel_t, dtype=np.float32)

BATCH_SIZE = 128
# numpy to tensor
trainingData_t = TensorDataset(torch.from_numpy(trainingData_t),
                            torch.from_numpy(trainingLabel_t))
valData_t = TensorDataset(torch.from_numpy(valData_t),
                            torch.from_numpy(valLabel_t))
# tensor to DataLoader
train_dataloader_t = DataLoader(trainingData_t, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_dataloader_t = DataLoader(valData_t, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

print('Data input finished')

# for loop in range(20):

    ######### training stage ############

precision = 1e-8

if usegpu:
    model=DANNModel().cuda()
else:
    model=DANNModel()

loss_class = nn.MSELoss()
loss_domain = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

for p in model.parameters():
    p.requires_grad = True

epoch_num = 300

best_loss_t, best_loss_s = float('inf'), float('inf')
 
for epoch in range(epoch_num):
    print('epoch: {} / {}'.format(epoch+1, epoch_num))
    print('-' *20)

    len_dataloader = min(len(train_dataloader_s), len(train_dataloader_t))
    data_source_iter = iter(train_dataloader_s)
    data_target_iter = iter(train_dataloader_t)

    running_loss, running_loss_s_l, running_loss_t_d, running_loss_s_d, total_num = 0.0, 0.0, 0.0, 0.0, 0.0
    model.train()

    for i in range(len_dataloader):

        p = float(i + epoch * len_dataloader) / epoch_num / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # training model using source data
        data_source = data_source_iter.__next__()

        data_s, label_s = data_source

        model.zero_grad()
        batchsize = len(label_s)
        domain_label = torch.zeros(batchsize).long()

        if usegpu:
            data_s = data_s.cuda()
            label_s = label_s.cuda()
            domain_label = domain_label.cuda()

        class_output, domain_output = model(input_data=data_s, alpha=alpha)
        err_s_label = loss_class(class_output, label_s)
        err_s_domain = loss_domain(domain_output, domain_label)

        # training model using target data
        data_target = data_target_iter.__next__()
        data_t, _ = data_target
        batchsize = len(data_t)

        domain_label = torch.ones(batchsize).long()

        if usegpu:
            data_t = data_t.cuda()
            domain_label = domain_label.cuda()

        _, domain_output = model(input_data=data_t, alpha=alpha)
        err_t_domain = loss_domain(domain_output, domain_label)
        err = err_t_domain + err_s_domain + err_s_label

        running_loss_s_l += err_s_label
        running_loss_s_d += err_s_domain
        running_loss_t_d += err_t_domain

        running_loss += err

        err.backward()
        optimizer.step()

        sys.stdout.write('\r epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f, alpha:%f' \
              % (epoch, i + 1, len_dataloader, err_s_label.data.cpu().numpy(),
                 err_s_domain.data.cpu().numpy(), err_t_domain.data.cpu().item(), alpha))
        sys.stdout.flush()

    Loss = running_loss / (i+1)
    running_loss_s_l = running_loss_s_l/(i+1)
    running_loss_s_d = running_loss_s_d/(i+1)
    running_loss_t_d = running_loss_t_d/(i+1)

    # print('Train: Loss:{:.4f}'.format(Loss))
    # print('Train: RMSE Loss: {:.4f}'.format(math.sqrt(Loss)))
    print('\nTrain:   S_L_Loss: {:.4f}   S_D_Loss: {:.4f}  T_D_Loss:  {:.4f}  Loss: {:.4f}  RMSE Loss:{:.4f}'
        .format(running_loss_s_l, running_loss_s_d, running_loss_t_d, Loss, math.sqrt(Loss)))

######### valing stage ############

    # source data validate
    running_loss_s = 0.0
    model.eval()
    for i, (data, label) in enumerate(val_dataloader_s):
        if usegpu:
            data = data.cuda()
            label = label.cuda()

        pred_label, _ = model(input_data = data, alpha = 0)
        # print(pred_label.shape)
        loss = loss_class(pred_label, label)
        running_loss_s += loss.item()
    Loss_s = running_loss_s / (i+1)
    print('Val: Loss: {:.4f}'.format(Loss_s))
    print('Val: RMSE Loss: {:.4f}'.format(math.sqrt(Loss_s)))

    #target data validate
    running_loss_t = 0.0
    model.eval()
    for i, (data, label) in enumerate(val_dataloader_t):
        if usegpu:
            data = data.cuda()
            label = label.cuda()

        pred_label,_ = model(input_data = data, alpha = 0)
        loss = loss_class(pred_label, label)
        running_loss_t += loss.item()
    Loss_t = running_loss_t / (i+1)
    print('Val: Loss: {:.4f}'.format(Loss_t))
    print('Val: RMSE Loss: {:.4f}'.format(math.sqrt(Loss_t)))


    if Loss_t < best_loss_t:
        best_loss_s = Loss_s
        best_loss_t = Loss_t
        best_weight = copy.deepcopy(model.state_dict())
        best_epoch = epoch +1
        print('New best validation Loss: {:.4f}'.format(Loss))

    print('The best epoch is:', best_epoch)

torch.save(best_weight, r'saved_model\CNN_fix_dann_train.pkl')


