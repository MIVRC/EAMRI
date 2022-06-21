"""
modified at 09/06/22
"""

import sys
sys.path.insert(0,'..')
import torch
from torch import nn
from fastmri.data import transforms_simple as T
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


#======================================================

class im_extractor(nn.Module):
    """
    DAM + DC
    """
    def __init__(self, indim=2, fNum=16, growthRate=16, layer=3, dilate=True, activation='ReLU', useOri=True, transition=0.5, residual=True, isFastmri=True, n_DAM=3):
        
        super(im_extractor, self).__init__()
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



class edgeBlock(nn.Module):
    def __init__(self, inFeat=1, outFeat=16, ks=3):
        super(edgeBlock,self).__init__()

        self.conv1 = nn.Conv2d(inFeat, outFeat, kernel_size=3, padding = 1, stride=1) 
        self.bn1 = nn.BatchNorm2d(outFeat)
        self.act1 = nn.ReLU()
        
        # decode part
        self.conv2 = nn.Conv2d(outFeat, inFeat, kernel_size=1, stride=1) 
        self.bn2 = nn.BatchNorm2d(inFeat)
        self.act2 = nn.ReLU()


    def forward(self, x):

        # encode
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.act1(x1)

        # decode
        x2 = self.conv2(x1)
        x2 = self.bn2(x2)
        x2 = self.act2(x2)

        return (x1,x2)

#============================== #============================== #============================== #============================== #============================== #==============================

class attHead(nn.Module):
    """
    norm + 1*1 conv + 3*3 dconv
    """
    
    def __init__(self, C, C1, layernorm_type= 'BiasFree', bias=False):
        """
        C: input channel 
        C1: output channel after 1*1 conv

        """
        super(attHead,self).__init__()
        self.norm = LayerNorm(C, layernorm_type)
        self.conv1 = nn.Conv2d(C, C1, kernel_size=1, bias=bias) 
        self.conv2 = nn.Conv2d(C1, C1, kernel_size=3, stride=1, padding=1, groups=C1, bias=bias)

    def forward(self, x):
        """
        x: (B, C, H, W)
        """ 
        #x1 = self.norm(x) #(B, H, W, C)
        x1 = self.conv1(x) # (B, C1, H, W)
        x1 = self.conv2(x1)  #(B, C1, H, W)

        return x1



class EEM(nn.Module):
    def __init__(self, C, C1, C2, num_heads, bias):
        """
        edge enhance module (EEM)
        C: input channel of image 
        C1: input channel of edge
        C2 : output channel of imhead
        """

        super(EEM, self).__init__()

        self.imhead = attHead(C, 2 * C2)
        self.ehead = attHead(C1, C2)
        self.num_heads = num_heads
        self.a1 = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.project_out = nn.Conv2d(C2, C, kernel_size=1, bias=bias)
        self.out = nn.Conv2d(C2, C, kernel_size=1, bias=bias)


    def forward(self, x, e):
        """
        x: input image (B, C, H, W)
        e: input edge (B, C1, H, W)
        """
        _, _, H, W = x.shape    

        q1 = self.imhead(x) #(B, C2, H, W)
        k_eg = self.ehead(e) #(B,2 * C2, H, W)

        # split into q, k, v 
        q_im, v_im = q1.chunk(2, dim=1)

        # reshape
        q_im = rearrange(q_im, 'b (head c) h w -> b head c (h w)', head=self.num_heads) 
        k_eg = rearrange(k_eg, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_im = rearrange(v_im, 'b (head c) h w -> b head c (h w)', head=self.num_heads) #(B, head, C, H*W)

        q_im = torch.nn.functional.normalize(q_im, dim=-1)
        k_eg = torch.nn.functional.normalize(k_eg, dim=-1)
        
        attn = (q_im @ k_eg.transpose(-2, -1)) * self.a1 #(B, head, C, C)
        attn = attn.softmax(dim=-1)

        out = (attn @ v_im)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=H, w=W)

        out = self.project_out(out) #(B, C, H, W)
        xout = x + out
        return xout #(B, C, H, W)



class IEM(nn.Module):
    def __init__(self, C, C2, num_heads, bias):
        """
        image enhancement module (IEM)
        C: input channel of image 
        C2 : output channel of imhead
        """

        super(IEM, self).__init__()

        self.imhead = attHead(C, 3 * C2)
        self.num_heads = num_heads
        self.a1 = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.project_out = nn.Conv2d(C2, C, kernel_size=1, bias=bias)
        self.out = nn.Conv2d(C2, C, kernel_size=1, bias=bias)


    def forward(self, x):
        """
        x: input image (B, C, H, W)
        e: input edge (B, C1, H, W)
        """
        _, _, H, W = x.shape    

        q1 = self.imhead(x) #(B, 2 * C2, H, W)

        # split into q, k, v 
        q_im, k_im, v_im = q1.chunk(3, dim=1)

        # reshape
        q_im = rearrange(q_im, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_im = rearrange(k_im, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_im = rearrange(v_im, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_im = torch.nn.functional.normalize(q_im, dim=-1)
        k_im = torch.nn.functional.normalize(k_im, dim=-1)
        
        attn = (q_im @ k_im.transpose(-2, -1)) * self.a1
        attn = attn.softmax(dim=-1)

        out = (attn @ v_im)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=H, w=W)

        out = self.project_out(out) #(B, C, H, W)
        xout = x + out
        
        return xout #(B, C, H, W)


class attBlock(nn.Module):
    def __init__(self, C, C1, C2, num_heads, bias):
        super(attBlock, self).__init__()
        
        self.b1 = EEM(C, C1, C2, num_heads, bias) 
        self.b2 = IEM(C, C2, num_heads, bias) 

    def forward(self, x, e):

        # EEM

        #x1 = self.b2(x)
        x1 = self.b1(x, e)

        return x1


class net_0621_var3(nn.Module):
    """
    12 DAM + transformer block
    """
    def __init__(self, img_size=320, indim=2, edgeFeat=16, outdim=12, num_head=4, n_DAM=3, isFastmri=False): 
        """
        input:
            C: number of conv in RDB, default 3
            G0: input_channel, default 2
            G1: output_channel, default 2
            n_RDB: number of RDBs 

        """
        super(net_0621_var3, self).__init__()
        
        # image module
        self.net1 = im_extractor(isFastmri=isFastmri, n_DAM=n_DAM)
        self.net2 = im_extractor(isFastmri=isFastmri, n_DAM=n_DAM)
        self.net3 = im_extractor(isFastmri=isFastmri, n_DAM=n_DAM)
        self.net4 = im_extractor(isFastmri=isFastmri, n_DAM=n_DAM)

        # edge module
        self.edgeNet = edgeBlock(inFeat=2, outFeat=edgeFeat) # simple edge block

        # fuse module
        self.fuse1 = attBlock(indim, edgeFeat, outdim, num_head, bias=False) 
        self.fuse2 = attBlock(indim, edgeFeat, outdim, num_head, bias=False) 
        self.fuse3 = attBlock(indim, edgeFeat, outdim, num_head, bias=False) 
        self.fuse4 = attBlock(indim, edgeFeat, outdim, num_head, bias=False) 

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
        (e1, e1_d) = self.edgeNet(x1) 
        x1 = self.fuse1(x2, e1)
        x1 = self.dc(x1,y,m)

        # second stage
        x2 = self.net2(x1, y, m) #(B, 2, H, W)
        (e2, e2_d) = self.edgeNet(x1) 
        x1 = self.fuse2(x2, e2)
        x1 = self.dc(x1,y,m)

        # third stage
        x2 = self.net3(x1, y, m) #(B, 2, H, W)
        (e3, e3_d) = self.edgeNet(x1) 
        x1 = self.fuse3(x2, e3)
        x1 = self.dc(x1,y,m)

        # fourth stage
        x2 = self.net4(x1, y, m) #(B, 2, H, W)
        (e4, e4_d) = self.edgeNet(x1) 
        x1 = self.fuse4(x2, e4)
        x1 = self.dc(x1,y,m)

        return (e1_d,e2_d,e3_d,e4_d,x1)




