import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .networkUtil import *


class convBlock(nn.Module):
    def __init__(self, indim=2, iConvNum = 5, f=64):
        super(convBlock, self).__init__()
        self.Relu = nn.ReLU()
        self.conv1 = nn.Conv2d(indim,f,3,padding = 1)
        convList = []
        for i in range(1, iConvNum-1):
            tmpConv = nn.Conv2d(f,f,3,padding = 1)
            convList.append(tmpConv)
        self.layerList = nn.ModuleList(convList)

        self.conv2 = nn.Conv2d(f,indim,3,padding = 1)
    
    def forward(self, x1):
        x2 = self.conv1(x1)
        x2 = self.Relu(x2)
        for conv in self.layerList:
            x2 = conv(x2)
            x2 = self.Relu(x2)
        x3 = self.conv2(x2)
        
        x4 = x3 + x1
        
        return x4

class DC_CNN(nn.Module):
    def __init__(self, d = 5, c = 5, fNum = 32, isFastmri=False):
        super(DC_CNN, self).__init__()
        templayerList = []
        for i in range(c):
            tmpConv = convBlock(2, d, fNum)
            tmpDF = dataConsistencyLayer_fastmri(isFastmri=isFastmri)
            templayerList.append(tmpConv)
            templayerList.append(tmpDF)
        self.layerList = nn.ModuleList(templayerList)
        
    def forward(self, x1, y, mask):
        xt = x1
        flag = True
        for layer in self.layerList:
            if(flag):
                xt = layer(xt)
                flag = False
            else:
                xt = layer(xt, y, mask)
                flag = True
        
        return xt


class DC_CNN_multicoil(nn.Module):
    def __init__(self, indim=30, d = 5, c = 5, fNum = 32, isFastmri=True, isMulticoil=True):
        super(DC_CNN_multicoil, self).__init__()
        templayerList = []
        for i in range(c):
            tmpConv = convBlock(indim, d, fNum)
            tmpDF = dataConsistencyLayer_fastmri_m(isFastmri=isFastmri, isMulticoil=isMulticoil)
            templayerList.append(tmpConv)
            templayerList.append(tmpDF)
        self.layerList = nn.ModuleList(templayerList)
        
    def forward(self, x1, y, mask):
        xt = x1
        flag = True
        for layer in self.layerList:
            if(flag):
                xt = layer(xt)
                flag = False
            else:
                xt = layer(xt, y, mask)
                flag = True
        
        return xt
