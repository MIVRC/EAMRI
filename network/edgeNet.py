"""
U-net shaped edgeNet
multi-scale edge model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from .networkUtil import *
from .cascadeNetwork import denseConv


def paramNumber(net):
    params = list(net.parameters())
    k = 0
    for i in params:
        #print('ddd: '+str(list(i.size())))
        l = 1
        for j in i.size():
            l *= j
        #print('aaa: '+ str(l))
        k = k + l
    
    return k


class DAM(nn.Module):
    '''
    basic DAM module 
    '''
    def __init__(self, inChannel = 2, outChannel = 32, fNum = 16, growthRate = 16, layer = 3, dilate = True, activation = 'ReLU', useOri = True, transition = 0.5, residual = True):
        super(DAM, self).__init__()
        self.inChannel = inChannel
        self.outChannel = outChannel
        self.transition = transition
        self.residual = residual
        self.inConv = nn.Conv2d(self.inChannel, fNum, 3, padding = 1) # (16,2,3,3)
        if(activation == 'LeakyReLU'):
            self.activ = nn.LeakyReLU()
        elif(activation == 'ReLU'):
            self.activ = nn.ReLU()
        
        self.denseConv = denseConv(fNum, 3, growthRate, layer - 2, dilationLayer = dilate, activ = activation, useOri = useOri)
        self.transitionLayer = transitionLayer(fNum+growthRate*(layer-2), transition, activ = activation)
        self.outConv = convLayer(int((fNum+growthRate*(layer-2))*transition), self.outChannel, activ = activation)


    def forward(self, x):
        """
        x1 is the input from the simpleRes, i.e edge 
        """
        x2 = self.inConv(x) #[16,16,3,3]
        x2 = self.denseConv(x2) #(8,64,256,256)
        x2 = self.transitionLayer(x2) #(8,32,256,256)
        x2 = self.outConv(x2)
        return x2




class EAM(nn.Module):
    def __init__(self, C, C1, C2):
        """
        EAM module : simple dot product attention 
        C: input channel of image
        C1: input channel of edge
        C2: output channel
        """
        super(EAM, self).__init__()
        self.conv1 = nn.Conv2d(C1, C2, kernel_size=1, bias=True) 
        self.conv2 = nn.Conv2d(C, C2, kernel_size=1, bias=True) 
        self.conv3 = nn.Conv2d(C, C2, kernel_size=1, bias=True)
        self.outChannel = C2

    def forward(self, x, e):
        """
        input:
            x: image, (B, C, H, W)
            e: edge, (B, C1, H, W)
        return:
            out: (B, C2, H, W)
        """

        H, W = x.shape[2], x.shape[3]
        Q = self.conv1(e).reshape(-1, H, W) # B*C2, H, W
        V = self.conv2(x).reshape(-1, H, W) # B*C2, H, W
        K = self.conv3(x).reshape(-1, H, W) # B*C2, H, W
        
        score = torch.bmm(Q, V.transpose(1,2)) # B*C2, H, H
        attn = F.softmax(score.reshape(-1, H), dim=1).reshape(-1, H, H)
        ctx = torch.bmm(attn, K) # B*C2, H, W
        ctx = ctx.reshape(-1, self.outChannel, H, W) # B, C2, H, W

        return ctx



class EAM2(nn.Module):
    def __init__(self, C, C1, C2):
        """
        EAM2 module
        C: input channel of image
        C1: input channel of edge
        C2: output channel
        """
        super(EAM2, self).__init__()
        self.conv1 = nn.Conv2d(C+C1, C2, kernel_size=1, bias=True)

    def forward(self, x, e):
        """
        input:
            x: image, (B, C, H, W)
            e: edge, (B, C1, H, W)
        return:
            out: (B, C2, H, W)
        """
        x1 = torch.cat([x,e], dim=1) #(B, C+C1, H, W)
        x1 = self.conv1(x1) #(B, C2, H, W)
        return x1






class CALayer(nn.Module):
    """
    channel attentoion layer
    channel: number of input channel
    scale: channel downscale

    input: B, C, H, W
    output: B, C, H, W
    """
    def __init__(self, channel, scale=16):
        super(CALayer, self).__init__()
        assert scale > 0, "zero scal in calyer, pls check"
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel//scale, 1, padding=0, bias=True),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(channel//scale, channel, 1, padding=0, bias=True),
                nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class simpleFuse(nn.Module):
    def __init__(self, c, G0, G):
        """
        simple fusion module using RDB, CALayer and data consistency
        c: number of RDB blocks
        G0: input Channel
        G: adding dimension in RDB
        """
        super(simpleFuse, self).__init__()
        self.conv = RDB(c, G0, G) 
        self.calayer = CALayer(G0, G0//4)
        self.downconv = nn.Conv2d(G0, 2, kernel_size=1, bias=True)
        self.dc = dataConsistencyLayer_fastmri()


    def forward(self, x, x1, y, m):
        """
        input:
            x: input image (B, G0, H, W)
            x1: guided image (after eam) (B, G0, H, W)
            y : subsampled kspace (B, H, W, 2)
            m: mask
        output:
            (B, 2, H, W)
        """ 
        d = x - x1
        x2 = x1 + self.conv(d)
        x2 = self.calayer(x2) #(B, G0, H, W)
        x2 = self.downconv(x2) #(B, 2, H, W)
        out = self.dc(x2, y, m) #(B, 2, H, W)
        return out



class simpleFuse2(nn.Module):
    def __init__(self, indim, outdim):
        """
        simple fusion module using 1*1 conv and dc
        c: number of RDB blocks
        G0: input Channel
        G: adding dimension in RDB
        """
        super(simpleFuse2, self).__init__()
        self.conv = nn.Conv2d(2*indim, outdim, kernel_size=1, bias=True)
        self.dc = dataConsistencyLayer_fastmri()

    def forward(self, x, x1, y, m):
        """
        input:
            x: input image (B, G0, H, W)
            x1: guided image (after eam) (B, G0, H, W)
            y : subsampled kspace (B, H, W, 2)
            m: mask
        output:
            (B, 2, H, W)
        """ 
        x2 = torch.cat([x,x1],dim=1)
        x2 = self.conv(x2) #(B, 2, H, W)
        out = self.dc(x2, y, m) #(B, 2, H, W)
        return out



class simpleFuse3(nn.Module):
    def __init__(self, c, G0, G):
        """
        simple fusion module using RDB, CALayer and data consistency
        c: number of RDB blocks
        G0: input Channel
        G: adding dimension in RDB
        """
        super(simpleFuse3, self).__init__()
        self.conv = RDB(c, 2*G0, G) 
        self.calayer = CALayer(2*G0, G0//4)
        self.downconv = nn.Conv2d(2*G0, 2, kernel_size=1, bias=True)
        self.dc = dataConsistencyLayer_fastmri()


    def forward(self, x, x1, y, m):
        """
        input:
            x: input image (B, G0, H, W)
            x1: guided image (after eam) (B, G0, H, W)
            y : subsampled kspace (B, H, W, 2)
            m: mask
        output:
            (B, 2, H, W)
        """ 
        x2 = torch.cat([x,x1], dim=1) #(B, 2* G0, H, W)
        x2 = self.conv(x2) #(B, 2*G0, H, W)
        x2 = self.calayer(x2) #(B, 2*G0, H, W)
        x2 = self.downconv(x2) #(B, 2, H, W)
        out = self.dc(x2, y, m) #(B, 2, H, W)
        return out



# baseline

class edgeNet_var3_rdg_eam2_fuse3_rdg_edge(nn.Module):
    def __init__(self, indim, middim, outdim, num_RDB):
        """
        edgeNet_var3 with new feature extract and eam2
        use 3 RDB for head feature extraction
        input:
            indim: input image channel, default 2
            middim: mid image channel, default 32
            outdim: output image channel, default 2
            num_RDB: number of RDB blocks, default 2

        """ 
        super(edgeNet_var3_rdg_eam2_fuse3_rdg, self).__init__()
        self.im_conv0 = nn.Conv2d(indim, middim, kernel_size=3, padding=1, bias=True) 
        self.edge_conv0 = nn.Conv2d(indim, middim, kernel_size=3, padding=1, bias=True) 

        self.im_head = RDG(4, middim, middim//4, 4) 
        self.edge_head = RDG(4, middim, middim//4, 4) 
        
        self.conv_33 = nn.Conv2d(middim, middim, kernel_size=3, dilation=1, padding = 1)
        self.conv_55 = nn.Conv2d(middim, middim, kernel_size=3, dilation=2, padding = 2)
        self.conv_77 = nn.Conv2d(middim, middim, kernel_size=3, dilation=3, padding = 3)

        self.econv_33 = nn.Conv2d(middim, middim, kernel_size=3, dilation=1, padding = 1)
        self.econv_55 = nn.Conv2d(middim, middim, kernel_size=3, dilation=2, padding = 2)
        self.econv_77 = nn.Conv2d(middim, middim, kernel_size=3, dilation=3, padding = 3)

        # eam module
        self.eam1 = EAM2(middim, middim, middim)
        self.eam2 = EAM2(2, middim, middim)
        self.eam3 = EAM2(2, middim, middim)

        self.fuse1 = simpleFuse3(num_RDB, middim, middim//4)
        self.fuse2 = simpleFuse3(num_RDB, middim, middim//4)
        self.fuse3 = simpleFuse3(num_RDB, middim, middim//4)

        self.recon = RDG(4, middim, middim//4, 4) 
        self.downconv = nn.Conv2d(middim, outdim, kernel_size=1, bias=True)
        
        self.dc = dataConsistencyLayer_fastmri()

    def forward(self, x, e0,  y, m):
        """
        input:
            x: input image (B, 2, H, W)
            y: subsampled k-space (B, H, W, 2)
            m: mask
        """ 

        x1 = self.im_conv0(x)   #(B, middim, H, W)
        e1 = self.edge_conv0(e0)  #(B, middim, H, W)

        # feature extraction
        x1 = self.im_head(x1)
        e1 = self.edge_head(e1)

        # multi-scale convolution of image
        x33 = self.conv_33(x1) #(B, middim, H, W)
        x55 = self.conv_55(x1) #(B, middim, H, W)
        x77_1 = self.conv_77(x1) #(B, middim, H, W)

        # multi-scale convolution of edge
        e33 = self.econv_33(e1) #(B, middim, H, W)
        e55 = self.econv_55(e1) #(B, middim, H, W)
        e77 = self.econv_77(e1) #(B, middim, H, W)

        xe1 = self.eam1(x77_1, e77) #(B, middim, H, W)
        x55_1 = self.fuse1(xe1, x55, y, m) #(B, 2, H, W)
        
        xe2 = self.eam2(x55_1,e55) #(B, middim, H, W)
        x33_1 = self.fuse2(xe2,x33, y, m) #(B, 2, H, W)

        xe3 = self.eam3(x33_1, e33) #(B, middim, H, W)
        x_out = self.recon(xe3)
        x_out = self.downconv(x_out)
        x_out = self.dc(x_out, y, m)

        return x_out






if __name__ == '__main__':

    x = torch.randn(8,2,320,320)
    e = torch.randn(8,2,320,320)
    y = torch.randn(8,320,320,2)
    m = torch.randn(8,1,320,1)
    net = edgeNet_var2(2, 32, 2, 4)
    out = net(x,y,m)

    print(out.shape) #(8,2,320,320)
    print(paramNumber(net)) # (83816)
     
