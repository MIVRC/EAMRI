"""
modified at 23/05/22

"""


import sys
sys.path.insert(0,'..')
import torch
from torch import nn
from fastmri.data import transforms_simple as T
from .RS_attention_edge_0523 import RFB 
from torch.nn import functional as F
import pdb
import numpy as np
from .networkUtil import *

# image net
class myRDG(nn.Module):
    def __init__(self, C, G0, G1, n_RDB):
        """
        input:
            C: number of conv in RDB
            G0: input_channel 
            G1: middle_channel 
            n_RDB: number of RDBs 
        """
        super(myRDG, self).__init__()
        self.n_RDB = n_RDB
        RDBs = []
        self.head = nn.Conv2d(G0, G1, kernel_size=3, padding=1, bias=True) 
        for i in range(n_RDB):
            RDBs.append(RDB(C, G1, G1//4))
        self.RDB = nn.Sequential(*RDBs)
        self.conv = nn.Conv2d(G1*n_RDB, G0, kernel_size=3, stride=1, padding=1, bias=True)


    def forward(self, x):
        """
        input: 
            x (B, G0, H, W)
        output:
            out (B, G1, H, W)
        """
        buffer = self.head(x)
        temp = []
        for i in range(self.n_RDB):
            buffer = self.RDB[i](buffer)
            temp.append(buffer)
        buffer_cat = torch.cat(temp, dim=1)
        out = self.conv(buffer_cat) #(B, G0, H, W)

        return out


# transformer net

class transBlock(nn.Module):
    """
        edgeNet using transformer block
        1 encoder + RFB + 1 decoder
    Args:
        in_channels: input channels of the image, default 2
        out_channels: output channels of the image, default 2
        nf: middle channels in encoder, RFB and decoder
        depth: depthof RPTL (how many transformer block), defualt 6


    """

    def __init__(self, in_channels=2, out_channels=1, nf=48, down_scale=2, img_size=320, num_head=4, depth=2, window_size=8, mlp_ratio=2.,):

        super(transBlock, self).__init__()

        # down scale the image before transformer block
        if down_scale == 2:
            kernel1, stride1 = 3, 1
            kernel2, stride2 = 4, 2
        elif down_scale == 1:
            kernel1, stride1 = 3, 1
            kernel2, stride2 = 3, 1
        else:
            exit('Error: unrecognized down_scale')


        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=nf, kernel_size=kernel1, stride=stride1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=kernel2, stride=stride2, padding=1, bias=True),
        ) 

        self.RFB = RFB(img_size, nf, depth, num_head, window_size, mlp_ratio, down=True, down_scale=down_scale)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=nf, out_channels=out_channels, kernel_size=kernel1, stride=stride1, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )


    def forward(self, x):
        """
            x: image 
        """
        x1 = self.encoder(x) #(8,64,320,320)
        x2 = self.RFB(x1)
        xout = self.decoder(x2)
        
        return xout


class edgeAcct(nn.Module):

    def __init__(self, in_dim, out_dim):
        """
        in_dim: default 2
        out_dim: default 16
        """
        super(edgeAcct, self).__init__()

        self.query_proj = nn.Conv2d(in_channels=in_dim, out_channels=out_dim , kernel_size=1)
        self.key_proj = nn.Conv2d(in_channels=in_dim, out_channels=out_dim , kernel_size=1)
        self.value_proj = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1)


    def forward(self, x, e):
        """
        x: (B, 2, H, W)  
        e: (B, 1, H, W)
        """

        m_batchsize, C, width, height = x.size()

        x_norm = (x ** 2).sum(dim=1)
        x_norm = torch.unsqueeze(x_norm, 1)
        x1 = torch.cat([x_norm, e], dim=1) # B * C * H * W

        Q = self.query_proj(x1).view(m_batchsize, -1, width * height) # B * out_dim * (W * H)
        K = self.key_proj(x1).view(m_batchsize, -1, width * height) # B * out_dim * (W * H)
        V = self.value_proj(x).view(m_batchsize, -1, width * height) # B * C * (W * H)

        attention = torch.bmm(Q.permute(0, 2, 1), K) # B * (W * H) * (W * H)
        attention = self.softmax(attention)

        self_attetion = torch.bmm(V, attention) # B * C * (W * H)
        out = self_attetion.view(m_batchsize, C, width, height) # B * C * W * H

        return out #(B, C, H, W)
       


class simpleFuse(nn.Module):
    def __init__(self, indim, middim = 6, outdim=2):
        """
        simple fusion module using 1*1 conv and dc
        indim: input_channel
        outdim: output_channel, default=2
        """
        super(simpleFuse, self).__init__()
        self.conv1 = nn.Conv2d(indim, middim, kernel_size=3, padding=1, bias=True)
        self.acct = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv2d(middim, outdim, kernel_size=3, padding=1, bias=True)

    def forward(self, x, x1):
        """
        input:
            x: input image (B, G0, H, W)
            x1: guided image (after eam) (B, G0, H, W)
        output:
            (B, 2, H, W)
        """ 
        x2 = torch.cat([x,x1],dim=1)
        x2 = self.conv1(x2) #(B, 2, H, W)
        x2 = self.acct(x2) #(B, 2, H, W)
        xout = self.conv2(x2)

        return xout


class simpleFuse_debug(nn.Module):
    def __init__(self, indim, middim = 6, outdim=2):
        """
        simple fusion module using 1*1 conv and dc
        indim: input_channel
        outdim: output_channel, default=2
        """
        super(simpleFuse_debug, self).__init__()
        self.conv1 = nn.Conv2d(indim, middim, kernel_size=3, padding=1, bias=True)
        self.acct = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv2d(middim, outdim, kernel_size=3, padding=1, bias=True)
        self.acct1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, x1):
        """
        input:
            x: input image (B, G0, H, W)
            x1: guided image (after eam) (B, G0, H, W)
        output:
            (B, 2, H, W)
        """ 
        return x 

#=======================================================

class subNet(nn.Module):
    def __init__(self, indim=2, fNum=16, growthRate=16, layer=3, dilate=True, activation='ReLU', useOri=True, transition=0.5, residual=True, isFastmri=True):
        
        super(subNet, self).__init__()
        self.b1 = DAM(indim, fNum, growthRate, layer, dilate, activation, useOri, transition, residual)
        self.b2 = DAM(indim, fNum, growthRate, layer, dilate, activation, useOri, transition, residual)
        self.b3 = DAM(indim, fNum, growthRate, layer, dilate, activation, useOri, transition, residual)
        self.dc = dataConsistencyLayer_fastmri(isFastmri=isFastmri)


    def forward(self,x, y, m):

        x1 = self.b1(x)
        x1 = self.dc(x1, y, m)

        x1 = self.b2(x1)
        x1 = self.dc(x1, y, m)

        x1 = self.b3(x1)
        x1 = self.dc(x1, y, m)

        return x1

class subNet1(nn.Module):
    def __init__(self, indim=2, fNum=16, growthRate=16, layer=3, dilate=True, activation='ReLU', useOri=True, transition=0.5, residual=True, isFastmri=True, n_DAM=3):
        
        super(subNet1, self).__init__()
        layers = []
        for _ in range(n_DAM):
            layers.append(DAM(indim, fNum, growthRate, layer, dilate, activation, useOri, transition, residual))
            layers.append(dataConsistencyLayer_fastmri(isFastmri=isFastmri))
        self.layers = nn.ModuleList(layers)


    def forward(self,x, y, m):

        x1 = x
        for i, layer in enumerate(self.layers):
            if i % 2 == 0:
                x1 = layer(x1)
            else:
                x1 = layer(x1, y, m)

        return x1





class convTranNet_0523(nn.Module):
    """
    conv + transformer block
    """
    def __init__(self, img_size=320, C=3, G0=2, G1=16, n_RDB=4, nf=36, num_head=6, depth=6, window_size=8, isFastmri=False):

        """
        input:
            C: number of conv in RDB, default 3
            G0: input_channel, default 2
            G1: output_channel, default 2
            n_RDB: number of RDBs 

        """
        super(convTranNet_0523, self).__init__()
       
        self.net1 = subNet(isFastmri=isFastmri)
        self.net2 = subNet(isFastmri=isFastmri)

        self.edgeNet = transBlock(in_channels=2, out_channels=1, nf=nf, down_scale=1, img_size=img_size, num_head=num_head, depth=depth, window_size=window_size, mlp_ratio=2.)

        self.fuse1 = simpleFuse(3, 12, 2) 
        self.fuse2 = simpleFuse(3, 12, 2) 

        self.dc = dataConsistencyLayer_fastmri(isFastmri=isFastmri)

    def forward(self, x0, _, y, m): # (image, kspace, mask)
        """
        input:
            x0: (B, 2, H, W) zero-filled image
            e0: (B, 2, H, W) edge of zim
            y: under-sampled kspace 
            m: mask

        """
        # first stage
        x1 = self.net1(x0, y, m) #(B, 2, H, W)
        e1 = self.edgeNet(x0) 
        x1 = self.fuse1(x1, e1)
        x2 = self.dc(x1,y,m)

        # second stage
        x3 = self.net2(x2, y, m) #(B, 2, H, W)
        e2 = self.edgeNet(x2) 
        x3 = self.fuse2(x3, e2)
        xout = self.dc(x3,y,m)

        return (e1,e2,xout)




class convTranNet_0523_var1(nn.Module):
    """
    conv + transformer block
    """
    def __init__(self, img_size=320, C=3, G0=2, G1=16, n_RDB=4, nf=36, num_head=6, depth=6, window_size=8, isFastmri=False):

        """
        input:
            C: number of conv in RDB, default 3
            G0: input_channel, default 2
            G1: output_channel, default 2
            n_RDB: number of RDBs 

        """
        super(convTranNet_0523_var1, self).__init__()
       
        self.net1 = subNet(isFastmri=isFastmri)
        self.net2 = subNet(isFastmri=isFastmri)
        self.net3 = transBlock(in_channels=2, out_channels=2, nf=nf, down_scale=1, img_size=img_size, num_head=num_head, depth=depth, window_size=window_size, mlp_ratio=2.)

        self.dc = dataConsistencyLayer_fastmri(isFastmri=isFastmri)

    def forward(self, x0, y, m): # (image, kspace, mask)
        """
        input:
            x0: (B, 2, H, W) zero-filled image
            e0: (B, 2, H, W) edge of zim
            y: under-sampled kspace 
            m: mask

        """
        # first stage
        x1 = self.net1(x0, y, m) #(B, 2, H, W)

        # second stage
        x1 = self.net2(x1, y, m) #(B, 2, H, W)
        
        res = self.net3(x1)

        xout = self.dc(x1 + res,y,m)

        return xout


class convTranNet_0523_var2(nn.Module):
    """
    conv + transformer block
    """
    def __init__(self, img_size=320, C=3, G0=2, G1=16, n_RDB=4, nf=36, num_head=6, depth=6, window_size=8, isFastmri=False):

        """
        input:
            C: number of conv in RDB, default 3
            G0: input_channel, default 2
            G1: output_channel, default 2
            n_RDB: number of RDBs 

        """
        super(convTranNet_0523_var2, self).__init__()
       
        self.net1 = subNet(isFastmri=isFastmri)
        self.net2 = subNet(isFastmri=isFastmri)
        self.net3 = transBlock(in_channels=2, out_channels=2, nf=nf, down_scale=1, img_size=img_size, num_head=num_head, depth=depth, window_size=window_size, mlp_ratio=2.)

        self.dc = dataConsistencyLayer_fastmri(isFastmri=isFastmri)

    def forward(self, x0, y, m): # (image, kspace, mask)
        """
        input:
            x0: (B, 2, H, W) zero-filled image
            e0: (B, 2, H, W) edge of zim
            y: under-sampled kspace 
            m: mask

        """
        # first stage
        x1 = self.net1(x0, y, m) #(B, 2, H, W)

        # second stage
        x1 = self.net2(x1, y, m) #(B, 2, H, W)
        
        x1 = self.net3(x1)

        xout = self.dc(x1,y,m)

        return xout


class convTranNet_0523_var3(nn.Module):
    """
    12 DAM + transformer block
    """
    def __init__(self, img_size=320, C=3, G0=2, G1=16, n_RDB=4, nf=36, num_head=6, depth=6, window_size=8, n_DAM=6, isFastmri=False):

        """
        input:
            C: number of conv in RDB, default 3
            G0: input_channel, default 2
            G1: output_channel, default 2
            n_RDB: number of RDBs 

        """
        super(convTranNet_0523_var3, self).__init__()
       
        self.net1 = subNet1(isFastmri=isFastmri, n_DAM=n_DAM)
        self.net2 = subNet1(isFastmri=isFastmri, n_DAM=n_DAM)

        self.edgeNet = transBlock(in_channels=2, out_channels=1, nf=nf, down_scale=1, img_size=img_size, num_head=num_head, depth=depth, window_size=window_size, mlp_ratio=2.)

        self.fuse = simpleFuse(3, 12, 2) 
        self.dc = dataConsistencyLayer_fastmri(isFastmri=isFastmri)

    def forward(self, x0, _, y, m): # (image, kspace, mask)
        """
        input:
            x0: (B, 2, H, W) zero-filled image
            e0: (B, 2, H, W) edge of zim
            y: under-sampled kspace 
            m: mask

        """
        # first stage
        x1 = self.net1(x0, y, m) #(B, 2, H, W)
        e1 = self.edgeNet(x0) 
        x1 = self.fuse(x1, e1)
        x2 = self.dc(x1,y,m)

        # second stage
        x3 = self.net2(x2, y, m) #(B, 2, H, W)
        e2 = self.edgeNet(x2) 
        x3 = self.fuse(x3, e2)
        xout = self.dc(x3,y,m)

        return (e1,e2,xout)



class convTranNet_0523_var4(nn.Module):
    """
    12 DAM + transformer block
    """
    def __init__(self, img_size=320, C=3, G0=2, G1=16, n_RDB=4, nf=36, num_head=6, depth=6, window_size=8, n_DAM=3, isFastmri=False):

        """
        input:
            C: number of conv in RDB, default 3
            G0: input_channel, default 2
            G1: output_channel, default 2
            n_RDB: number of RDBs 

        """
        super(convTranNet_0523_var4, self).__init__()
       
        self.net1 = subNet1(isFastmri=isFastmri, n_DAM=n_DAM)
        self.net2 = subNet1(isFastmri=isFastmri, n_DAM=n_DAM)
        self.net3 = subNet1(isFastmri=isFastmri, n_DAM=n_DAM)
        self.net4 = subNet1(isFastmri=isFastmri, n_DAM=n_DAM)

        self.edgeNet = transBlock(in_channels=2, out_channels=1, nf=nf, down_scale=1, img_size=img_size, num_head=num_head, depth=depth, window_size=window_size, mlp_ratio=2.)

        self.fuse = simpleFuse(3, 12, 2) 

        self.dc = dataConsistencyLayer_fastmri(isFastmri=isFastmri)

    def forward(self, x1, _, y, m): # (image, kspace, mask)
        """
        input:
            x1: (B, 2, H, W) zero-filled image
            e0: (B, 2, H, W) edge of zim
            y: under-sampled kspace 
            m: mask

        """
        # first stage
        x2 = self.net1(x1, y, m) #(B, 2, H, W)
        e1 = self.edgeNet(x1) 
        x1 = self.fuse(x2, e1)
        x1 = self.dc(x1,y,m)

        # second stage
        x2 = self.net2(x1, y, m) #(B, 2, H, W)
        e2 = self.edgeNet(x1) 
        x1 = self.fuse(x2, e2)
        x1 = self.dc(x1,y,m)

        # third stage
        x2 = self.net3(x1, y, m) #(B, 2, H, W)
        e3 = self.edgeNet(x1) 
        x1 = self.fuse(x2, e3)
        x1 = self.dc(x1,y,m)

        # fourth stage
        x2 = self.net4(x1, y, m) #(B, 2, H, W)
        e4 = self.edgeNet(x1) 
        x1 = self.fuse(x2, e4)
        x1 = self.dc(x1,y,m)

        return (e1,e2,e3,e4,x1)





class convTranNet_0523_baseline(nn.Module):
    """
    pure cnn baseline
    """
    def __init__(self, img_size=320, C=3, G0=2, G1=16, n_RDB=4, nf=36, num_head=6, depth=6, window_size=8, isFastmri=False):

        """
        input:
            C: number of conv in RDB, default 3
            G0: input_channel, default 2
            G1: output_channel, default 2
            n_RDB: number of RDBs 

        """
        super(convTranNet_0523_baseline, self).__init__()
       
        self.net1 = subNet(isFastmri=isFastmri)
        self.net2 = subNet(isFastmri=isFastmri)
        self.net3 = subNet(isFastmri=isFastmri)
        self.net4 = subNet(isFastmri=isFastmri)


    def forward(self, x0, y, m): # (image, kspace, mask)
        """
        input:
            x0: (B, 2, H, W) zero-filled image
            e0: (B, 2, H, W) edge of zim
            y: under-sampled kspace 
            m: mask

        """
        # first stage
        x1 = self.net1(x0, y, m) #(B, 2, H, W)

        # second stage
        x1 = self.net2(x1, y, m) #(B, 2, H, W)

        x1 = self.net3(x1, y, m) #(B, 2, H, W)

        x1 = self.net4(x1, y, m) #(B, 2, H, W)

        return x1



