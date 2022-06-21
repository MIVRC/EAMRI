"""
modified at 29/05/22

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
        self.conv1 = nn.Conv2d(C, C1, kernel_size=1, bias=False) 
        self.conv2 = nn.Conv2d(C1, C1, kernel_size=3, stride=1, padding=1, groups=C1, bias=bias)

    def forward(self, x):
        """
        x: (B, C, H, W)
        """ 
        x1 = self.norm(x) #(B, C, H, W)
        x1 = self.conv1(x1) # (B, C1, H, W)
        x1 = self.conv2(x1)  #(B, C1, H, W)

        return x1

class attFuse1(nn.Module):
    def __init__(self, C, C1, C2, num_heads, bias):
        """
        C: input channel of image 
        C1: input channel of edge
        C2 : output channel of imhead
        """

        super(attFuse1, self).__init__()

        self.imhead = imhead(C, C2)
        self.ehead = imhead(C1, 2 * C2)
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.project_out = nn.Conv2d(C2, C, kernel_size=1, bias=bias)
        self.out = nn.Conv2d(C, 2, kernel_size=1, bias=bias)

    def forward(self, x, e):
        """
        x: input image (B, C, H, W)
        e: input edge (B, C1, H, W)
        """
        _, _, H, W = x.shape    

        q_image = self.imhead(x) #(B, C2, H, W)
        e1 = self.ehead(e)  #(B, 2* C2, H, W)
        k_edge, v_edge = e1.chunk(2, dim=1)
   
        q_image = rearrange(q_image, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_edge = rearrange(k_edge, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_edge = rearrange(v_edge, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_image = torch.nn.functional.normalize(q_image, dim=-1)
        k_edge = torch.nn.functional.normalize(k_edge, dim=-1)
        
        attn = (q_image @ k_edge.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v_edge)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=H, w=W)

        out = self.project_out(out) #(B, C, H, W)
        xout = self.out(x + out)

        return xout #(B, C, H, W)

class attFuse2(nn.Module):
    def __init__(self, C, C1, C2, num_heads, bias):
        """
        C: input channel of image 
        C1: input channel of edge
        C2 : output channel of imhead
        """

        super(attFuse2, self).__init__()

        self.imhead = imhead(C, C2)
        self.ehead = imhead(C1, 2 * C2)
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

        q_image = self.imhead(x) #(B, C2, H, W)
        e1 = self.ehead(e)  #(B, 2* C2, H, W)
        k_edge, v_edge = e1.chunk(2, dim=1)
   
        q_image = rearrange(q_image, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_edge = rearrange(k_edge, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_edge = rearrange(v_edge, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_image = torch.nn.functional.normalize(q_image, dim=-1)
        k_edge = torch.nn.functional.normalize(k_edge, dim=-1)
        
        attn = (q_image @ k_edge.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v_edge)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=H, w=W)

        out = self.project_out(out) #(B, C, H, W)

        return out #(B, C, H, W)





class attFuse(nn.Module):
    def __init__(self, C, C1, C2, num_heads, bias):
        """
        C: input channel of image 
        C1: input channel of edge
        C2 : output channel of imhead
        """

        super(attFuse, self).__init__()

        self.imhead = imhead(C, C2)
        self.ehead = imhead(C1, 2 * C2)
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

        q_image = self.imhead(x) #(B, C2, H, W)
        e1 = self.ehead(e)  #(B, 2* C2, H, W)
        k_edge, v_edge = e1.chunk(2, dim=1)
   
        q_image = rearrange(q_image, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_edge = rearrange(k_edge, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_edge = rearrange(v_edge, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_image = torch.nn.functional.normalize(q_image, dim=-1)
        k_edge = torch.nn.functional.normalize(k_edge, dim=-1)
        
        attn = (q_image @ k_edge.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v_edge)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=H, w=W)

        out = self.project_out(out) #(B, C, H, W)
        xout = x + out

        return xout #(B, C, H, W)




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



class convTranNet_0529(nn.Module):
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
        super(convTranNet_0529, self).__init__()
       
        self.net1 = subNet1(isFastmri=isFastmri, n_DAM=n_DAM)
        self.net2 = subNet1(isFastmri=isFastmri, n_DAM=n_DAM)
        self.net3 = subNet1(isFastmri=isFastmri, n_DAM=n_DAM)
        self.net4 = subNet1(isFastmri=isFastmri, n_DAM=n_DAM)

        self.edgeNet = edgeBlock(inFeat=2) # simple edge block

        self.fuse = attFuse(indim, 1, outdim, num_head, bias=False) 

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



class convTranNet_0529_debug(nn.Module):
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
        super(convTranNet_0529_debug, self).__init__()
       
        self.net1 = subNet1(isFastmri=isFastmri, n_DAM=n_DAM)
        self.net2 = subNet1(isFastmri=isFastmri, n_DAM=n_DAM)

        self.edgeNet = edgeBlock(inFeat=2) # simple edge block

        self.fuse = attFuse(indim, 1, outdim, num_head, bias=False) 

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

        return (e1,e2, x1)


class convTranNet_0529_var1(nn.Module):
    """
    12 DAM + transformer block
    """
    def __init__(self, img_size=320, indim=2, fNum=16, outdim=32, num_head=4, n_DAM=3, isFastmri=False): 
        """
        input:
            indim: input dimmension


        """
        super(convTranNet_0529_var1, self).__init__()
       
        self.net1 = DAM1(indim, fNum, growthRate=16, layer=5, dilate=True, activation='ReLU', useOri=True, transition=0.5)

        self.net2 = DAM1(indim, fNum, growthRate=16, layer=5, dilate=True, activation='ReLU', useOri=True, transition=0.5)

        self.net3 = DAM1(indim, fNum, growthRate=16, layer=5, dilate=True, activation='ReLU', useOri=True, transition=0.5)

        self.net4 = DAM1(indim, fNum, growthRate=16, layer=5, dilate=True, activation='ReLU', useOri=True, transition=0.5)

        self.edgeNet = edgeBlock(2, fNum) # simple edge block

        self.fuse = attFuse1(fNum, fNum, outdim, num_head, bias=False) 

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
        x2 = self.net1(x1) #(B, 16, H, W)
        e1 = self.edgeNet(x1)  #(B, 16, H, W)
        x1 = self.fuse(x2, e1)
        x1 = self.dc(x1,y,m)

        # second stage
        x2 = self.net2(x1) #(B, 2, H, W)
        e2 = self.edgeNet(x1) 
        x1 = self.fuse(x2, e2)
        x1 = self.dc(x1,y,m)

        # third stage
        x2 = self.net3(x1) #(B, 2, H, W)
        e3 = self.edgeNet(x1) 
        x1 = self.fuse(x2, e3)
        x1 = self.dc(x1,y,m)

        # fourth stage
        x2 = self.net4(x1) #(B, 2, H, W)
        e4 = self.edgeNet(x1) 
        x1 = self.fuse(x2, e4)
        x1 = self.dc(x1,y,m)

        return (e1,e2,e3,e4,x1)



class convTranNet_0529_var2(nn.Module):
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
        super(convTranNet_0529_var2, self).__init__()
       
        self.net1 = subNet1(isFastmri=isFastmri, n_DAM=n_DAM)
        self.net2 = subNet1(isFastmri=isFastmri, n_DAM=n_DAM)
        self.net3 = subNet1(isFastmri=isFastmri, n_DAM=n_DAM)
        self.net4 = subNet1(isFastmri=isFastmri, n_DAM=n_DAM)

        self.edgeNet = edgeBlock(2) # simple edge block

        self.fuse = attFuse2(indim, 1, outdim, num_head, bias=False) 

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



class convTranNet_0529_var3(nn.Module):
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
        super(convTranNet_0529_var3, self).__init__()
       
        self.net1 = subNet1(isFastmri=isFastmri, n_DAM=n_DAM)
        self.net2 = subNet1(isFastmri=isFastmri, n_DAM=n_DAM)
        self.net3 = subNet1(isFastmri=isFastmri, n_DAM=n_DAM)
        self.net4 = subNet1(isFastmri=isFastmri, n_DAM=n_DAM)

        self.edgeNet = edgeBlock(inFeat=2) # simple edge block

        self.fuse = attFuse(indim, 1, outdim, num_head, bias=False) 

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
        r1 = self.dc(x1,y,m)

        # second stage
        x2 = self.net2(r1, y, m) #(B, 2, H, W)
        e2 = self.edgeNet(r1) 
        x1 = self.fuse(x2, e2)
        r2 = self.dc(x1,y,m)

        # third stage
        x2 = self.net3(r2, y, m) #(B, 2, H, W)
        e3 = self.edgeNet(r2) 
        x1 = self.fuse(x2, e3)
        r3 = self.dc(x1,y,m)

        # fourth stage
        x2 = self.net4(r3, y, m) #(B, 2, H, W)
        e4 = self.edgeNet(r3) 
        x1 = self.fuse(x2, e4)
        xout = self.dc(x1,y,m)

        return (e1,e2,e3,e4,r1,r2,r3,xout)



