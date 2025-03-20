import torch
import torch.nn as nn
from torch.autograd import Function
import torch.optim as optim
from torchinfo import summary
from functions import ReverseLayerF

NUM_ConV1 = 64
NUM_ConV2 = 128
NUM_ConV3 = 256
OUTPUT_SIZE1 = 118
OUTPUT_SIZE2 = 22
OUTPUT_SIZE3 = 10
FEATURE_SIZE1 = 320


class AttentionMechanism(nn.Module):
    def __init__(self,inplanes):
        super(AttentionMechanism, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # self.ecaconv = nn.Conv1d(1,1,kernel_size=1,stride=1,padding=0,bias=False)
        self.fc1 = nn.Linear(in_features=inplanes, out_features=inplanes)
        self.fc2 = nn.Linear(in_features=inplanes, out_features=inplanes)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print(x.shape)
        avg_out = self.relu(self.fc1(self.avg_pool(x).view(x.size(0), 1, x.size(1))))
        max_out = self.relu(self.fc2(self.max_pool(x).view(x.size(0), 1, x.size(1))))
        # print(avg_out.shape)
        out = avg_out + max_out
        out = out.view(out.size(0), out.size(2), 1)
        return self.sigmoid(out)

class MAM_MA(nn.Module):
    def __init__(self,inplane1,inplane2):
        super(MAM_MA, self).__init__()
        self.ca = AttentionMechanism(inplane1)
        self.sa = AttentionMechanism(inplane2)

    def forward(self, x):
        out_ca = x * self.ca(x)
        out_ca=out_ca.permute(0,2,1)
        out_sa = out_ca * self.sa(out_ca)
        out_sa = out_sa.permute(0,2,1)
        return out_sa

class DANNModel(nn.Module):

    def __init__(self):
        super(DANNModel, self).__init__()

        self.feature = nn.Sequential(
            nn.Conv1d(1,NUM_ConV1, kernel_size=64, stride=8),
            nn.BatchNorm1d(NUM_ConV1),
            nn.PReLU(NUM_ConV1),
            nn.Dropout(0.2),
            MAM_MA(NUM_ConV1,OUTPUT_SIZE1),

            nn.Conv1d(NUM_ConV1, NUM_ConV2, kernel_size=32, stride=4),
            nn.BatchNorm1d(NUM_ConV2),
            nn.PReLU(NUM_ConV2),
            nn.Dropout(0.2),
            MAM_MA(NUM_ConV2,OUTPUT_SIZE2),

            nn.Conv1d(NUM_ConV2,NUM_ConV3, kernel_size=4, stride=2),
            nn.BatchNorm1d(NUM_ConV3),
            nn.PReLU(NUM_ConV3),
            nn.Dropout(0.2),
            MAM_MA(NUM_ConV3,OUTPUT_SIZE3)
        )
        self.class_classifier = nn.Sequential(
            nn.Linear(NUM_ConV3 * OUTPUT_SIZE3, FEATURE_SIZE1),
            nn.BatchNorm1d(FEATURE_SIZE1),
            nn.PReLU(FEATURE_SIZE1),
            nn.Dropout(0.3),

            nn.Linear(FEATURE_SIZE1, FEATURE_SIZE1),
            nn.BatchNorm1d(FEATURE_SIZE1),
            nn.PReLU(FEATURE_SIZE1),
            nn.Dropout(0.3),

            nn.Linear(FEATURE_SIZE1, 1)
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(NUM_ConV3 * OUTPUT_SIZE3, FEATURE_SIZE1),
            nn.BatchNorm1d(FEATURE_SIZE1),
            nn.PReLU(FEATURE_SIZE1),
            nn.Dropout(0.3),
            nn.Linear(FEATURE_SIZE1, 2),
            # nn.LogSoftmax(dim=1)           
        )

    def forward(self, input_data, alpha):
        feature = self.feature(input_data)
        feature = feature.view(-1, NUM_ConV3 * OUTPUT_SIZE3)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output
            


if __name__ =='__main__':

    daan=DANNModel()

    print(summary(daan,input_size=(1,1,1000)))

