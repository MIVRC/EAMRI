import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from .networkUtil import *

class dilatedConvBlock(nn.Module):
    def __init__(self, iConvNum = 3):
        super(dilatedConvBlock, self).__init__()
        self.LRelu = nn.LeakyReLU()
        convList = []
        for i in range(1, iConvNum+1):
            tmpConv = nn.Conv2d(32,32,3,padding = i, dilation = i)
            convList.append(tmpConv)
        self.layerList = nn.ModuleList(convList)
    
    def forward(self, x1):
        x2 = x1
        for conv in self.layerList:
            x2 = conv(x2)
            x2 = self.LRelu(x2)
        
        return x2
    
class RDN_recursionUnit(nn.Module):
    def __init__(self, convNum = 3, recursiveTime = 3, inChannel = 2):
        super(RDN_recursionUnit, self).__init__()
        self.rTime = recursiveTime
        self.LRelu = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(inChannel,32,3,padding = 1)
        self.dilateBlock = dilatedConvBlock(convNum)
        self.conv2 = nn.Conv2d(32,2,3,padding = 1)
        
    def forward(self, x1):
        x2 = self.conv1(x1)
        x2 = self.LRelu(x2)
        xt = x2
        for i in range(self.rTime):
            x3 = self.dilateBlock(xt)
            xt = x3+x2
        x4 = self.conv2(xt)
        x4 = self.LRelu(x4)
        x5 = x4+x1
        
        return x5
    
class dataFidelityUnit(nn.Module):
    def __init__(self, initLamda = 10e6):
        super(dataFidelityUnit, self).__init__()
        self.normalized = True
        self.lamda = initLamda
    
    def forward(self, xin, y, mask):
        # y: aliased image 
        # x1: reconstructed image
        # mask: sampling mask
        mask = mask.reshape(mask.shape[0],mask.shape[1],mask.shape[2],1)
        
        xin_c = xin
        xin_c = xin_c.permute(0,2,3,1)
        
        xin_f = torch.fft(xin_c,2, normalized=self.normalized)
        xGT_f = y
        
        xout_f = xin_f + (- xin_f + xGT_f) * iScale * mask

        xout = torch.ifft(xout_f,2, normalized=self.normalized)
        xout = xout.permute(0,3,1,2)
        
        return xout

class RDN_complex(nn.Module):
    def __init__(self, b=5, d=3, r=3, xin1 = 2, dcLayer = 'DF', isFastmri=False):
        super(RDN_complex, self).__init__()
        templayerList = []
        for i in range(b):
            if(i==0):
                tmpConv = RDN_recursionUnit(d,r,inChannel=xin1)
            else:
                tmpConv = RDN_recursionUnit(d,r)
            if(dcLayer=='DF'):
                tmpDF = dataFidelityUnit()
            else:
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
        
   
#=====================================================================
# multicoil

class dilatedConvBlock_m(nn.Module):
    def __init__(self, iConvNum = 3, inChannel=32):
        super(dilatedConvBlock_m, self).__init__()
        self.LRelu = nn.LeakyReLU()
        convList = []
        for i in range(1, iConvNum+1):
            tmpConv = nn.Conv2d(inChannel,inChannel,3,padding = i, dilation = i)
            convList.append(tmpConv)
        self.layerList = nn.ModuleList(convList)
    
    def forward(self, x1):
        x2 = x1
        for conv in self.layerList:
            x2 = conv(x2)
            x2 = self.LRelu(x2)
        
        return x2
 


class RDN_recursionUnit_m(nn.Module):
    def __init__(self, convNum = 3, recursiveTime = 3, inChannel = 2, midChannel=32):
        super(RDN_recursionUnit_m, self).__init__()
        self.rTime = recursiveTime
        self.LRelu = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(inChannel,midChannel,3,padding = 1)
        self.dilateBlock = dilatedConvBlock_m(convNum, midChannel)
        self.conv2 = nn.Conv2d(midChannel,inChannel,3,padding = 1)
        
    def forward(self, x1):
        x2 = self.conv1(x1)
        x2 = self.LRelu(x2)
        xt = x2
        for i in range(self.rTime):
            x3 = self.dilateBlock(xt)
            xt = x3+x2
        x4 = self.conv2(xt)
        x4 = self.LRelu(x4)
        x5 = x4+x1
        
        return x5
 


class RDN_multicoil(nn.Module):
    def __init__(self, b=5, d=3, r=3, xin1=30, midChannel=64, isFastmri=True, isMulticoil=True):
        super(RDN_multicoil, self).__init__()
        templayerList = []
        for i in range(b):
            tmpConv = RDN_recursionUnit_m(d,r,xin1,midChannel)
            tmpDF = dataConsistencyLayer_fastmri_m(isFastmri=isFastmri, isMulticoil=isMulticoil)
            templayerList.append(tmpConv)
            templayerList.append(tmpDF)

        self.layerList = nn.ModuleList(templayerList)
   
    def forward(self, x1, y, mask, sens_map=None):
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
    
