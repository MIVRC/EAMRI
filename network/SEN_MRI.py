import torch
import torch.nn as nn
from .srrfn_model import *
from .networkUtil import *
import pdb


def conv1d(in_channels,out_channels):
    conv = nn.Conv2d(in_channels, out_channels,kernel_size=1)
    return conv

class WAM(nn.Module):
    """
    input: (b,c,h,w)
    output: (b,c,h,w)
    """
    def __init__(self, inFeat=2, midFeat=64):
        super(WAM,self).__init__()
        self.head = nn.Conv2d(2*inFeat,midFeat,kernel_size=1)
        self.act = nn.ReLU()
        self.tail = nn.Conv2d(midFeat, inFeat,kernel_size=1)

    def forward(self,x1,x2):
        """
        x1: (8,2,256,256)
        """
        x = torch.cat([x1,x2],dim=1)
        x = self.head(x) 
        x = self.act(x)
        x = self.tail(x)
        return x


class TDCFM(nn.Module):
    def __init__(self,inChannel=2, fmChannel=64,kernel_size=3, act=nn.ReLU(True), n_resblocks=5, isFastmri=False, isMulticoil=False):
        super(TDCFM,self).__init__()

        self.dc1 = dataConsistencyLayer_fastmri_m(isFastmri=isFastmri, isMulticoil=isMulticoil)
        self.head = conv1d(inChannel, fmChannel)
        self.dam = ResidualGroup(fmChannel, kernel_size, act=act, res_scale=1, n_resblocks=n_resblocks) 
        self.tail = conv1d(fmChannel, inChannel)
        self.dc2 = dataConsistencyLayer_fastmri_m(isFastmri=isFastmri, isMulticoil=isMulticoil)


    def forward(self,x,y,mask):

        x1 = self.dc1(x, y, mask) 
        x1 = self.head(x1)
        x1 = self.dam(x1) 
        x1 = self.tail(x1) # decrease channel
        x2 = self.dc2(x1,y,mask)
        return x2


class WADCFM(nn.Module):
    """
    WADCFM module
    """
    def __init__(self, inChannel = 2, wChannel = 32, fmChannel=64, kernel_size = 3, n_resblocks=5, act= nn.ReLU(True), type_tdcfm=0, isFastmri=False, isMulticoil=False):

        """
        WFD + FMRDN(DC + FM + DC)
        wChannel : middle channel in wfd module

        """
        super(WADCFM,self).__init__()
        self.wfd = WAM(inChannel, wChannel)
        if type_tdcfm == 0:
            self.tdcfm = TDCFM(inChannel, fmChannel, kernel_size, act, n_resblocks, isFastmri=isFastmri, isMulticoil=isMulticoil)
        else:
            raise NotImplementedError("please specify the correct type for tdcfm")

        self.last_hidden = None

    def forward(self, x, y, mask):
       
        if self.should_reset:
            self.last_hidden = torch.zeros(x.size()).cuda()
            self.last_hidden.copy_(x)
            self.should_reset = False

        x1 = self.wfd(x,self.last_hidden) # fuse with prev input
        x2 = self.tdcfm(x1, y, mask)
        self.last_hidden = x2
        return x2


    def reset_state(self):
        self.should_reset = True
        self.last_hidden = None



class WAS(nn.Module):
    "SEN-MRI"
    def __init__(self, inChannel = 2, wChannel = 12, fmChannel = 12, skChannel = 12, c=3, nmodule=3, M=2, r=4, L= 6, kernel_size = 3, n_resblocks=5, act= nn.ReLU(True), type_tdcfm=0, isFastmri=False, isMulticoil=False):
        """
        inChannel = 2
        subgroups = 5 
        kernel_size = 3
        n_resblocks = 10
        """
        super(WAS, self).__init__()
        self.recur_times = c
        layers = []
        for _ in range(nmodule):
            layers.append(\
                    WADCFM(inChannel=inChannel, \
                          wChannel=wChannel,\
                          fmChannel=fmChannel,\
                          kernel_size = kernel_size,\
                          n_resblocks = n_resblocks, \
                          act = act,\
                          type_tdcfm = type_tdcfm,\
                          isFastmri = isFastmri,\
                          isMulticoil=isMulticoil)
                    )

        # ===========================
        # skmodule
        self.layerList = nn.ModuleList(layers)
        self.skconv = ASIM(inChannel, skChannel, M, r,1, L) 
        self.sktail = dataConsistencyLayer_fastmri_m(isFastmri=isFastmri, isMulticoil=isMulticoil)
        self.sktail1 = dataConsistencyLayer_fastmri_m(isFastmri=isFastmri, isMulticoil=isMulticoil)
        self.last_hidden = None

    def forward(self, x, y, mask):
    
        self._reset_state()
        x1 = x 
        outs = []

        for ii in range(self.recur_times):
            for layer in self.layerList:
                x1 = layer(x1,y,mask)

            outs.append(x1)
            if ii == 0:
                self.last_hidden = x1
            elif ii == 1:
                x2 = self.skconv(self.last_hidden, x1)
                self.last_hidden = self.sktail(x2,y,mask) # dc consistency
            elif ii == 2:
                x2 = self.skconv(self.last_hidden, x1)
                self.last_hidden = self.sktail1(x2,y,mask) # dc consistency

        outs.append(self.last_hidden) 
        return outs


    def _reset_state(self):
        self.last_hidden = None
        for layer in self.layerList:
            layer.reset_state()








