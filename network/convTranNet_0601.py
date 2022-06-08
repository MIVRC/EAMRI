"""
modified at 01/06/22

"""


import sys
sys.path.insert(0,'..')
import torch
from torch import nn
from fastmri.data import transforms_simple as T
#from .RS_attention_edge_0523 import RFB 
from torch.nn import functional as F
import pdb
import numpy as np
from einops import rearrange
from .networkUtil import *


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        #if isinstance(normalized_shape, numbers.Integral):
        normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x).contiguous()), h, w).contiguous()




class edgeBlock(nn.Module):
    def __init__(self, inFeat=1, outFeat=16, ks=3):
        super(edgeBlock,self).__init__()
        self.conv1 = nn.Conv2d(inFeat, outFeat, kernel_size=3, padding = 1, stride=1) 
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(outFeat, 1, kernel_size=1) 
        self.act2 = nn.ReLU()

    def forward(self, x):
        
        x1 = self.conv1(x)
        x2 = self.act1(x1)
        x3 = self.conv2(x2)
        return self.act2(x3)


class edgeBlock_debug(nn.Module):
    def __init__(self, inFeat=2, midFeat = 16, outFeat=1, ks=3):
        super(edgeBlock_debug,self).__init__()

        self.bn = nn.BatchNorm2d(inFeat)
        self.act = nn.ReLU()
        self.conv = nn.Conv2d(inFeat, midFeat, kernel_size=1) 

        self.bn1 = nn.BatchNorm2d(midFeat)
        self.act1 = nn.ReLU()
        self.conv1 = nn.Conv2d(midFeat, outFeat, kernel_size=3, padding=1, stride=1) 



    def forward(self, x):
        
        x1 = self.bn(x)
        x1 = self.act(x1)
        x1 = self.conv(x1)

        x1 = self.bn1(x1)
        x1 = self.act1(x1)
        x1 = self.conv1(x1)

        return x1



class imhead(nn.Module):
    """
    norm + 1*1 conv + 3*3 dconv
    """
    
    def __init__(self, C, C1, layernorm_type= 'BiasFree', bias=False):
        """
        C: input channel 
        C1: output channel after 1*1 conv

        """
        super(imhead,self).__init__()
        self.norm = LayerNorm(C, layernorm_type)
        self.conv1 = nn.Conv2d(C, C1, kernel_size=1, bias=bias) 
        self.conv2 = nn.Conv2d(C1, C1, kernel_size=3, stride=1, padding=1, groups=C1, bias=bias)

    def forward(self, x):
        """
        x: (B, C, H, W)
        """ 
        x1 = self.norm(x) #(B, C, H, W)
        x1 = self.conv1(x1) # (B, C1, H, W)
        x1 = self.conv2(x1)  #(B, C1, H, W)

        return x1


class attFuse_0601(nn.Module):
    def __init__(self, C, C1, C2, num_heads, bias):
        """
        C: input channel of image 
        C1: input channel of edge

        C2 : output channel of imhead
        """

        super(attFuse_0601, self).__init__()

        self.conv1 = nn.Conv2d(C+C1, C2, kernel_size=3, padding=1, stride=1, bias=True)
        self.imhead = imhead(C2, 3* C2)
        
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.project_out = nn.Conv2d(C2, C, kernel_size=1, bias=bias)
        self.out = nn.Conv2d(C2, C, kernel_size=1, bias=bias)

    def forward(self, x, e):
        """
        x: input image (B, C, H, W)
        e: input edge (B, C1, H, W)
        """
        _, _, H, W = x.shape    

        xe = torch.cat([x, e], dim=1) #(B, C+C1, H, W)
        xe = self.conv1(xe) #(B, C2, H, W)
        qkv = self.imhead(xe) #(B, 3*C2, H, W)

        q, k, v = qkv.chunk(3, dim=1)
   
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
       
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=H, w=W)

        out = self.project_out(out) #(B, C, H, W)
        
        return x + out #(B, C, H, W)



class attFuse_0601_var1(nn.Module):
    def __init__(self, C, C2, num_heads, bias):
        """
        C: input channel of image 
        C1: input channel of edge

        C2 : output channel of imhead
        """

        super(attFuse_0601_var1, self).__init__()

        self.conv1 = nn.Conv2d(C, C2, kernel_size=3, padding=1, stride=1, bias=True)
        self.imhead = imhead(C2, 3* C2)
        
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.project_out = nn.Conv2d(C2, C, kernel_size=1, bias=bias)
        self.out = nn.Conv2d(C2, C, kernel_size=1, bias=bias)

    def forward(self, x):
        """
        x: input image (B, C, H, W)
        """
        _, _, H, W = x.shape    

        xe = self.conv1(x) #(B, C2, H, W)
        qkv = self.imhead(xe) #(B, 3*C2, H, W)

        q, k, v = qkv.chunk(3, dim=1)
   
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
       
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=H, w=W)

        out = self.project_out(out) #(B, C, H, W)
        
        return x + out #(B, C, H, W)



class attFuse_debug(nn.Module):
    def __init__(self, C, C1, C2, num_heads, bias):
        """
        C: input channel of image 
        C1: input channel of edge
        C2 : output channel of imhead
        """

        super(attFuse_debug, self).__init__()

    def forward(self, x, e):
        """
        x: input image (B, C, H, W)
        e: input edge (B, C1, H, W)
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


class convTranNet_0601(nn.Module):
    """
    12 DAM + transformer block
    """
    def __init__(self, img_size=320, indim=2, outdim=12, num_head=4, n_DAM=3, isFastmri=False): 
        """
        input:
            C: number of conv in RDB, default 3
            G0: input_channel, default 2
            G1: output_channel, default 2
            n_RDB: number of RDBs 

        """
        super(convTranNet_0601, self).__init__()
       
        self.net1 = subNet1(isFastmri=isFastmri, n_DAM=n_DAM)
        self.net2 = subNet1(isFastmri=isFastmri, n_DAM=n_DAM)
        self.net3 = subNet1(isFastmri=isFastmri, n_DAM=n_DAM)
        self.net4 = subNet1(isFastmri=isFastmri, n_DAM=n_DAM)

        self.edgeNet = edgeBlock(inFeat=2) # simple edge block

        self.fuse1 = attFuse_0601(indim, 1, outdim, num_head, bias=False) 
        self.fuse2 = attFuse_0601(indim, 1, outdim, num_head, bias=False) 
        self.fuse3 = attFuse_0601(indim, 1, outdim, num_head, bias=False) 
        self.fuse4 = attFuse_0601(indim, 1, outdim, num_head, bias=False) 

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
        x1 = self.fuse1(x2, e1)
        x1 = self.dc(x1,y,m)

        # second stage
        x2 = self.net2(x1, y, m) #(B, 2, H, W)
        e2 = self.edgeNet(x1) 
        x1 = self.fuse2(x2, e2)
        x1 = self.dc(x1,y,m)

        # third stage
        x2 = self.net3(x1, y, m) #(B, 2, H, W)
        e3 = self.edgeNet(x1) 
        x1 = self.fuse3(x2, e3)
        x1 = self.dc(x1,y,m)

        # fourth stage
        x2 = self.net4(x1, y, m) #(B, 2, H, W)
        e4 = self.edgeNet(x1) 
        x1 = self.fuse4(x2, e4)
        x1 = self.dc(x1,y,m)

        return (e1,e2,e3,e4,x1)


class convTranNet_0601_debug(nn.Module):
    """
    12 DAM + transformer block
    """
    def __init__(self, img_size=320, indim=2, outdim=12, num_head=4, n_DAM=3, isFastmri=False): 
        """
        input:
            C: number of conv in RDB, default 3
            G0: input_channel, default 2
            G1: output_channel, default 2
            n_RDB: number of RDBs 

        """
        super(convTranNet_0601_debug, self).__init__()
      
        self.net1 = subNet1(isFastmri=isFastmri, n_DAM=n_DAM)
        self.net2 = subNet1(isFastmri=isFastmri, n_DAM=n_DAM)
        self.net3 = subNet1(isFastmri=isFastmri, n_DAM=n_DAM)
        self.net4 = subNet1(isFastmri=isFastmri, n_DAM=n_DAM)
        
        self.edgeNet = edgeBlock(inFeat=2) # simple edge block

        self.fuse1 = attFuse_debug(indim, 1, outdim, num_head, bias=False) 
        self.fuse2 = attFuse_debug(indim, 1, outdim, num_head, bias=False) 
        self.fuse3 = attFuse_debug(indim, 1, outdim, num_head, bias=False) 
        self.fuse4 = attFuse_debug(indim, 1, outdim, num_head, bias=False) 

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
        x1 = self.fuse1(x2, e1)
        #x1 = self.dc(x1,y,m)

        # second stage
        x2 = self.net2(x1, y, m) #(B, 2, H, W)
        e2 = self.edgeNet(x1) 
        x1 = self.fuse2(x2, e2)
        #x1 = self.dc(x1,y,m)

        # third stage
        x2 = self.net3(x1, y, m) #(B, 2, H, W)
        e3 = self.edgeNet(x1) 
        x1 = self.fuse3(x2, e3)
        #x1 = self.dc(x1,y,m)

        # fourth stage
        x2 = self.net4(x1, y, m) #(B, 2, H, W)
        e4 = self.edgeNet(x1) 
        x1 = self.fuse4(x2, e4)
        #x1 = self.dc(x1,y,m)

        return (e1, e2,e3,e4, x1)


class convTranNet_0601_debug2(nn.Module):
    """
    12 DAM + transformer block
    """
    def __init__(self, img_size=320, indim=2, outdim=12, num_head=4, n_DAM=3, isFastmri=False): 
        """
        input:
            C: number of conv in RDB, default 3
            G0: input_channel, default 2
            G1: output_channel, default 2
            n_RDB: number of RDBs 

        """
        super(convTranNet_0601_debug2, self).__init__()
       
        self.net1 = subNet1(isFastmri=isFastmri, n_DAM=n_DAM)
        self.net2 = subNet1(isFastmri=isFastmri, n_DAM=n_DAM)
        self.net3 = subNet1(isFastmri=isFastmri, n_DAM=n_DAM)
        self.net4 = subNet1(isFastmri=isFastmri, n_DAM=n_DAM)

        self.fuse1 = attFuse_debug(indim, 1, outdim, num_head, bias=False) 
        self.fuse2 = attFuse_debug(indim, 1, outdim, num_head, bias=False) 
        self.fuse3 = attFuse_debug(indim, 1, outdim, num_head, bias=False) 
        self.fuse4 = attFuse_debug(indim, 1, outdim, num_head, bias=False) 

        self.dc = dataConsistencyLayer_fastmri(isFastmri=isFastmri)

    def forward(self, x1, y, m): # (image, kspace, mask)
        """
        input:
            x1: (B, 2, H, W) zero-filled image
            e0: (B, 2, H, W) edge of zim
            y: under-sampled kspace 
            m: mask

        """

        # first stage
        x2 = self.net1(x1, y, m) #(B, 2, H, W)
        x1 = self.fuse1(x2, x2)
        #x1 = self.dc(x1,y,m)

        # second stage
        x2 = self.net2(x1, y, m) #(B, 2, H, W)
        x1 = self.fuse2(x2, x2)
        #x1 = self.dc(x1,y,m)

        # third stage
        x2 = self.net3(x1, y, m) #(B, 2, H, W)
        x1 = self.fuse3(x2, x2)
        #x1 = self.dc(x1,y,m)

        # fourth stage
        x2 = self.net4(x1, y, m) #(B, 2, H, W)
        x1 = self.fuse4(x2, x2)
        #x1 = self.dc(x1,y,m)

        return x1






class convTranNet_0601_var1(nn.Module):
    """
    12 DAM + transformer block
    """
    def __init__(self, img_size=320, indim=2, outdim=12, num_head=4, n_DAM=3, isFastmri=False): 
        """
        input:
            C: number of conv in RDB, default 3
            G0: input_channel, default 2
            G1: output_channel, default 2
            n_RDB: number of RDBs 

        """
        super(convTranNet_0601_var1, self).__init__()
       
        self.net1 = subNet1(isFastmri=isFastmri, n_DAM=n_DAM)
        self.net2 = subNet1(isFastmri=isFastmri, n_DAM=n_DAM)
        self.net3 = subNet1(isFastmri=isFastmri, n_DAM=n_DAM)
        self.net4 = subNet1(isFastmri=isFastmri, n_DAM=n_DAM)

        self.fuse1 = attFuse_0601_var1(indim, outdim, num_head, bias=False) 
        self.fuse2 = attFuse_0601_var1(indim, outdim, num_head, bias=False) 
        self.fuse3 = attFuse_0601_var1(indim, outdim, num_head, bias=False) 
        self.fuse4 = attFuse_0601_var1(indim, outdim, num_head, bias=False) 

        self.dc = dataConsistencyLayer_fastmri(isFastmri=isFastmri)

    def forward(self, x1, y, m): # (image, kspace, mask)
        """
        input:
            x1: (B, 2, H, W) zero-filled image
            e0: (B, 2, H, W) edge of zim
            y: under-sampled kspace 
            m: mask

        """

        # first stage
        x2 = self.net1(x1, y, m) #(B, 2, H, W)
        x1 = self.fuse1(x2)
        x1 = self.dc(x1,y,m)

        # second stage
        x2 = self.net2(x1, y, m) #(B, 2, H, W)
        x1 = self.fuse2(x2)
        x1 = self.dc(x1,y,m)

        # third stage
        x2 = self.net3(x1, y, m) #(B, 2, H, W)
        x1 = self.fuse3(x2)
        x1 = self.dc(x1,y,m)

        # fourth stage
        x2 = self.net4(x1, y, m) #(B, 2, H, W)
        x1 = self.fuse4(x2)
        x1 = self.dc(x1,y,m)

        return x1





