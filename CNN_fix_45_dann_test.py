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
from model_dann import DANNModel

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

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

# load data
path = 'Data/deltaq45.mat'
data = loadmat(path)
deltaQ = data['deltaq']
deltaQ = np.transpose(deltaQ)
path = 'Data/eol45.mat'
life = loadmat(path)
life = life['eol']

valid_index=[42,40,33,31,30,18,15,9]
test_index=[34,28,27,21,17,16,12,4]

no_train_index=valid_index+test_index
no_train_index.sort(reverse=True)
valid_index.sort(reverse=True)
test_index.sort(reverse=True)


trainingData = []
trainingLabel = []

for i in range(len(deltaQ)):
    trainingData.append(deltaQ[i,:].reshape(1,1000))
    trainingLabel.append(life[i])

for i, del_index in enumerate(no_train_index):
    del trainingData[del_index]
    del trainingLabel[del_index]


valData = []
valLabel = []

for i, index in enumerate(valid_index):
    valData.append(deltaQ[index,:].reshape(1,1000))
    valLabel.append(life[index])

testingData = []
testingLabel = []

for i, index in enumerate(test_index):
    testingData.append(deltaQ[index,:].reshape(1,1000))
    testingLabel.append(life[index])


# data normalization
for i in range(len(trainingData)):
    tmp = trainingData[i]
    trainingData[i] = normalization(tmp)


for i in range(len(valData)):
    tmp = valData[i]
    valData[i] = normalization(tmp)

for i in range(len(testingData)):
    tmp = testingData[i]
    testingData[i] = normalization(tmp)

# data augmentation
num = len(trainingData)
for n in range(20):
    for i in range(num):
        noise = np.random.normal(0, 1e-2, [1, 1000])
        traininData_n = trainingData[i] + noise
        trainingData.append(traininData_n)
        trainingLabel.append(trainingLabel[i])

# print(len(trainingData))

for n in range(20):
    for i in range(num):
        noise = np.random.normal(0, 2e-2, [1, 1000])
        traininData_n = trainingData[i] + noise
        trainingData.append(traininData_n)
        trainingLabel.append(trainingLabel[i])

trainingData = np.array(trainingData, dtype=np.float32)
valData = np.array(valData, dtype=np.float32)
testingData = np.array(testingData, dtype=np.float32)

trainingLabel = np.array(trainingLabel, dtype=np.float32)
valLabel = np.array(valLabel, dtype=np.float32)
testingLabel = np.array(testingLabel, dtype=np.float32)


BATCH_SIZE = 128
# numpy to tensor
trainingData = TensorDataset(torch.from_numpy(trainingData),
                            torch.from_numpy(trainingLabel))
valData = TensorDataset(torch.from_numpy(valData),
                            torch.from_numpy(valLabel))
# tensor to DataLoader
train_dataloader = DataLoader(trainingData, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_dataloader = DataLoader(valData, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)


print('Data input finished')
maes=[]
rmses=[]
mapes=[]

pred_labels=[]

alphas=[1,10,100,1000]

for alpha in alphas:

    for loop in range(20):

        ######### training stage ############

        precision = 1e-8

        if usegpu:
            net=DANNModel().cuda()
        else:
            net=DANNModel()

        net.load_state_dict(torch.load('saved_model\CNN_fix_dann_train_am_%d_%d.pkl'%(alpha,loop),map_location=torch.device('cpu')))

        criterion = nn.MSELoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        initial_CNN = net.parameters

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min',
                                                        factor=0.1,patience=10,
                                                        verbose=True, eps=precision)

        
        net.eval()
        print('Start Testing...')

        data = testingData

        if usegpu:
            input_tensor = torch.from_numpy(data).cuda()
        else:
            input_tensor = torch.from_numpy(data)

        pred_label,_ = net(input_data=input_tensor,alpha=0)
        pred_label = pred_label.cpu()
        pred_label = pred_label.detach().numpy()

        pred_label = pred_label.squeeze()

        label = testingLabel
        label = label.squeeze()
    # 
        print("MAE:", mean_absolute_error(label, pred_label))
        print("RMSE:", math.sqrt(mean_squared_error(label, pred_label)))
        print("MAPE:", MAPE(label, np.transpose(pred_label)))
        print('--------------------------------------------')
        maes.append(mean_absolute_error(label, pred_label))
        rmses.append(math.sqrt(mean_squared_error(label, pred_label)))
        mapes.append(MAPE(label, np.transpose(pred_label)))

        # pred_labels.append(pred_label)
        
    np.savetxt('CNN45_mae_%d.txt'%alpha, maes, fmt='%.4f',newline='\n')
    np.savetxt('CNN45_rmse_%d.txt'%alpha, rmses, fmt='%.4f',newline='\n')
    np.savetxt('CNN45_mape_%d.txt'%alpha, mapes, fmt='%.4f',newline='\n')
    # np.savetxt('CNN45_pred_label_%d.txt'%alpha, pred_labels, fmt='%.4f',newline='\n')

