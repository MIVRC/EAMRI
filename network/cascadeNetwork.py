import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .srrfn_model import *
from .networkUtil import *
import pdb


def conv1d(in_channels,out_channels):
    conv = nn.Conv2d(in_channels, out_channels,kernel_size=1)
    return conv


class convBlock(nn.Module):
    def __init__(self, ioChannel = 2, iConvNum = 5, f=32, dilationLayer = False):
        super(convBlock, self).__init__()
        if(isinstance(ioChannel,int)):
            self.inChannel = ioChannel
            self.outChannel = ioChannel
        else:
            self.inChannel = ioChannel[0]
            self.outChannel = ioChannel[1]
        dilate = 1
        if(dilationLayer):
            dilateMulti = 2
        else:
            dilateMulti = 1
        self.Relu = nn.ReLU()
        self.conv1 = nn.Conv2d(self.inChannel,f,3,padding = 1)
        convList = []
        for i in range(1, iConvNum-1):
            tmpConv = nn.Conv2d(f,f,3,padding = dilate, dilation = dilate)
            dilate = dilate * dilateMulti
            convList.append(tmpConv)
        self.layerList = nn.ModuleList(convList)
        self.conv2 = nn.Conv2d(f,self.outChannel,3,padding = 1)
    
    def forward(self, x1):
        x2 = self.conv1(x1)
        x2 = self.Relu(x2)
        for conv in self.layerList:
            x2 = conv(x2)
            x2 = self.Relu(x2)
        x3 = self.conv2(x2)
        #x3 = self.Relu(x3)
        
        x4 = x3 + x1[:,0:self.outChannel]
        
        return x4

class denseConv(nn.Module):
    def __init__(self, inChannel=16, kernelSize=3, growthRate=16, layer=4, inceptionLayer = False, dilationLayer = False, activ = 'ReLU', useOri = False):
        super(denseConv, self).__init__()
        dilate = 1
        if(dilationLayer):
            dilateMulti = 2
        else:
            dilateMulti = 1
        pad = int((kernelSize-1)/2)
        #self.denselayer = layer-2
        self.denselayer = layer
        templayerList = []
        for i in range(0, self.denselayer):
            if(useOri):
                tempLayer = denseBlockLayer_origin(inChannel=inChannel+growthRate*i, outChannel=growthRate, kernelSize=kernelSize, bottleneckChannel=inChannel, dilateScale=dilate, activ=activ)
            else:
                tempLayer = denseBlockLayer(inChannel+growthRate*i, growthRate, kernelSize, inceptionLayer, dilate, activ)
            dilate = dilate * dilateMulti
            templayerList.append(tempLayer)
        self.layerList = nn.ModuleList(templayerList)
            
    def forward(self,x):
        for i in range(0, self.denselayer):
            tempY = self.layerList[i](x)
            x = torch.cat((x, tempY), 1)
            
        return x


class subDenseNet(nn.Module):
    '''
    ioChannel[0] = 2 for complex 
    '''
    def __init__(self, ioChannel = 2, fNum = 16, growthRate = 16, layer = 3, dilate = False, activation = 'ReLU', useOri = False, transition = 0, useSE = False, residual = True):
        super(subDenseNet, self).__init__()
        if(isinstance(ioChannel,int)):
            self.inChannel = ioChannel
            self.outChannel = ioChannel
        else:
            self.inChannel = ioChannel[0]
            self.outChannel = ioChannel[1]

        self.transition = transition
        self.useSE = useSE
        self.residual = residual
        self.inConv = nn.Conv2d(self.inChannel, fNum, 3, padding = 1) # (16,2,3,3)
        if(activation == 'LeakyReLU'):
            self.activ = nn.LeakyReLU()
        elif(activation == 'ReLU'):
            self.activ = nn.ReLU()
        self.denseConv = denseConv(fNum, 3, growthRate, layer - 2, dilationLayer = dilate, activ = activation, useOri = useOri)
        if(self.useSE):
            self.se = SELayer(fNum+growthRate*(layer-2),256,256)
        if(transition>0):
            #assert transition==0.5, "transition has to be 0.5 for debug"
            self.transitionLayer = transitionLayer(fNum+growthRate*(layer-2), transition, activ = activation)
            #self.outConv = nn.Conv2d(int((fNum+growthRate*(layer-2))*transition), inChannel, 3, padding = 1)
            self.outConv = convLayer(int((fNum+growthRate*(layer-2))*transition), self.outChannel, activ = activation)
        else:
            #self.outConv = nn.Conv2d(fNum+growthRate*(layer-2), inChannel, 3, padding = 1)
            self.outConv = convLayer(fNum+growthRate*(layer-2), self.outChannel, activ = activation)

    def forward(self, x):
        x2 = self.inConv(x) #[16,2,3,3]
        #x2 = self.activ(x2)
        x2 = self.denseConv(x2)
        if(self.useSE):
            x2 = self.se(x2)
        if(self.transition>0):
            x2 = self.transitionLayer(x2)
            #x2 = self.activ(x2)
        x2 = self.outConv(x2)
        if(self.residual):
            x2 = x2+x[:,:self.outChannel]

        return x2

#===================================================
class subUnet(nn.Module):
    def __init__(self, ioChannel = 2, fNum = 16, growthRate = 16, layer = 3, dilate = False, activation = 'ReLU', useOri = False, transition = 0.5, useSE = False, residual = True):
        super(subUnet, self).__init__()
        if(isinstance(ioChannel,int)):
            self.inChannel = ioChannel
            self.outChannel = ioChannel
        else:
            self.inChannel = ioChannel[0]
            self.outChannel = ioChannel[1]
        #self.transition = transition
        self.useSE = useSE
        self.residual = residual
        self.inConv = nn.Conv2d(self.inChannel, fNum, 3, padding = 1)

        self.e1_inter = denseConv(fNum, 3, growthRate, layer, dilationLayer = dilate, activ = activation, useOri = useOri)
        self.e1_tr = convLayer(fNum+growthRate*layer, fNum, activ = activation, kernelSize = 1)
        self.e1_ds = nn.MaxPool2d(kernel_size = 2)

        self.e2_inter = denseConv(fNum, 3, growthRate, layer, dilationLayer = dilate, activ = activation, useOri = useOri)
        self.e2_tr = convLayer(fNum+growthRate*layer, fNum, activ = activation, kernelSize = 1)
        self.e2_ds = nn.MaxPool2d(kernel_size = 2)

        self.m_inter = denseConv(fNum, 3, growthRate, layer, dilationLayer = dilate, activ = activation, useOri = useOri)
        self.m_tr = convLayer(fNum+growthRate*layer, fNum, activ = activation, kernelSize = 1)
        self.m_us = nn.ConvTranspose2d(fNum, fNum, 2, 2)

        self.d2_cat = convLayer(2 * fNum, fNum, activ = activation, kernelSize = 1)
        self.d2_inter = denseConv(fNum, 3, growthRate, layer, dilationLayer = dilate, activ = activation, useOri = useOri)
        self.d2_tr = convLayer(fNum+growthRate*layer, fNum, activ = activation, kernelSize = 1)
        self.d2_us = nn.ConvTranspose2d(fNum, fNum, 2, 2)

        self.d1_cat = convLayer(2 * fNum, fNum, activ = activation, kernelSize = 1)
        self.d1_inter = denseConv(fNum, 3, growthRate, layer, dilationLayer = dilate, activ = activation, useOri = useOri)
        self.d1_tr = convLayer(fNum+growthRate*layer, fNum, activ = activation, kernelSize = 1)

        #self.outConv = nn.Conv2d(fNum, self.outChannel, 3, padding = 1)
        self.outConv = convLayer(fNum, self.outChannel, activ = activation, kernelSize = 3)

    def forward(self, x):
        fin = self.inConv(x)
        f2 = self.e1_tr(self.e1_inter(fin))
        f2d = self.e1_ds(f2)

        f3 = self.e2_tr(self.e2_inter(f2d))
        f3d = self.e2_ds(f3)

        f4 = self.m_tr(self.m_inter(f3d))
        f4u = self.m_us(f4)

        f4c = torch.cat([f4u,f3],1)

        f5 = self.d2_cat(f4c)
        f5 = self.d2_tr(self.d2_inter(f5))
        f5u = self.d2_us(f5)

        f5c = torch.cat([f5u,f2],1)

        f6 = self.d1_cat(f5c)
        f6 = self.d1_tr(self.d1_inter(f6))

        y = self.outConv(f6)

        return y



class CN_Dense(nn.Module):
    def __init__(self, inChannel = 1, d = 5, c = 5, fNum = 16, growthRate = 16, dilate = False, activation = 'ReLU', useOri = False, transition = 0, trick = 0, globalDense = False, useSE = False, globalResSkip = False, subnetType = 'Dense', isFastmri=False):
        super(CN_Dense, self).__init__()
        self.globalSkip = globalDense
        self.globalResSkip = globalResSkip
        if(isinstance(trick,int)):
            trickList = [trick] * c
        else:
            assert len(trick) == c, 'Different length of c and trick'
            trickList = trick
        if(isinstance(subnetType,str)):
            subnetType = [subnetType] * c
        else:
            assert len(subnetType) == c, 'Different length of c and subnetType'
        subNetClassList = []
        for netType in subnetType:
            if(netType == 'Dense' or netType == 'd'):
                subNetClass = subDenseNet
            elif(netType == 'Unet' or netType == 'u'):
                subNetClass = subUnet
            else:
                assert False, "no such subnetType:" + subnetType
            subNetClassList.append(subNetClass)
        templayerList = []
        for i in range(c): # for each subnetwork
            if(self.globalSkip):
                tmpSubNet = subNetClassList[i]((inChannel*(i+1),inChannel), fNum, growthRate, d, dilate, activation, useOri, transition, useSE)
            else:
                if(isinstance(inChannel,tuple)):
                    if(i==0): # for the first subnetwork
                        tmpSubNet = subNetClassList[i](inChannel, fNum, growthRate, d, dilate, activation, useOri, transition, useSE)
                    else: # for the other subnetwork
                        tmpSubNet = subNetClassList[i](inChannel[1], fNum, growthRate, d, dilate, activation, useOri, transition, useSE)
                else:
                    tmpSubNet = subNetClassList[i](inChannel, fNum, growthRate, d, dilate, activation, useOri, transition, useSE)

            tmpDF = dataConsistencyLayer_static(trick = trickList[i], isFastmri=isFastmri)
            templayerList.append(tmpSubNet)
            templayerList.append(tmpDF)

        if(globalResSkip):
            templayerList[-2].residual = False
        self.layerList = nn.ModuleList(templayerList)
        
    def forward(self, x1, y, mask): # (image, kspace, mask)
        xin = x1
        flag = True
        for layer in self.layerList:
            if(flag):
                xt = layer(xin)
                flag = False
            else:
                xt = layer(xt, y, mask) # tdn layer
                flag = True
                if(self.globalSkip):
                    xin = torch.cat([xt,xin],1)
                else:
                    xin = xt
        if(self.globalResSkip):
            xt = xt + x1
        
        return xt



