"""
modified at 09/06/22
"""

import sys
sys.path.insert(0,'..')
import torch
from torch import nn
import torch.utils.checkpoint as checkpoint
from fastmri.data import transforms_simple as T
from torch.nn import functional as F
import pdb
import numpy as np
from .networkUtil import *



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
 


class rdn_convBlock(nn.Module):
    def __init__(self, convNum = 3, recursiveTime = 3, inChannel = 24, midChannel=32, isFastmri=False, isMulticoil=False):
        super(rdn_convBlock, self).__init__()
        self.rTime = recursiveTime
        self.LRelu = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(inChannel,midChannel,3,padding = 1)
        self.dilateBlock = dilatedConvBlock_m(convNum, midChannel)
        self.conv2 = nn.Conv2d(midChannel,inChannel,3,padding = 1)
        self.dc = dataConsistencyLayer_fastmri_m(isFastmri=isFastmri, isMulticoil=isMulticoil)
        
    def forward(self, x1, y, m):
        x2 = self.conv1(x1)
        x2 = self.LRelu(x2)
        xt = x2
        for i in range(self.rTime):
            x3 = self.dilateBlock(xt)
            xt = x3+x2
        x4 = self.conv2(xt)
        x4 = self.LRelu(x4)
        x5 = x4+x1
        
        xout = self.dc(x5, y, m)

        return xout




class convBlock(nn.Module):
    """
    DAM + DC
    """
    def __init__(self, 
                indim=2, 
                fNum=16, 
                growthRate=16, 
                layer=3, 
                dilate=True, 
                activation='ReLU', 
                useOri=True, 
                transition=0.5, 
                residual=True, 
                isFastmri=True, 
                n_DAM=3, 
                num_iter=3,
                isMulticoil=False):
        
        super(convBlock, self).__init__()
        layers = []
        self.n_DAM = n_DAM
        self.num_iter = num_iter

        for _ in range(n_DAM):
            layers.append(DAM(indim, fNum, growthRate, layer, dilate, activation, useOri, transition, residual))
            layers.append(dataConsistencyLayer_fastmri_m(isFastmri=isFastmri, isMulticoil=isMulticoil))

        self.layers = nn.ModuleList(layers)

    def forward(self, x1, y, m):
        
        for _ in range(self.num_iter):
            for i, layer in enumerate(self.layers):
                if i % 2 == 0: 
                    x1 = layer(x1)
                else:     
                    x1 = layer(x1, y, m) # dc layer
        
        return x1



class MSRB(nn.Module):
    def __init__(self, n_feats):
        """
        n_feats: input dimension
        """
        super(MSRB, self).__init__()

        kernel_size_1 = 3
        kernel_size_2 = 5
        
        # stage 1
        self.conv_3_1 = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1, bias=True)
        self.conv_5_1 = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=2, bias=True, dilation=2)
        self.fuse1 = nn.Conv2d(2*n_feats, n_feats, kernel_size=1, bias=True) 

        # stage 2
        self.conv_3_2 = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1, bias=True)
        self.conv_5_2 = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=2, dilation=2, bias=True) # 7*7 conv

        self.confusion = nn.Conv2d(n_feats * 2, n_feats, 1, padding=0, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        input_1 = x
        output_3_1 = self.conv_3_1(input_1) # 3*3 conv
        output_5_1 = self.conv_5_1(input_1) #3*3 conv
        input_2 = self.relu(torch.cat([output_3_1, output_5_1], 1))
        input_2 = self.fuse1(input_2) # 1*1 conv
        
        output_3_2 = self.conv_3_2(input_2)
        output_5_2 = self.conv_5_2(input_2)

        input_3 = self.relu(torch.cat([output_3_2, output_5_2], 1))
        output = self.confusion(input_3)

        output += x

        return output


class Edge_Net(nn.Module):
    def __init__(self, indim, hiddim, conv=default_conv, n_MSRB=3):
        
        super(Edge_Net, self).__init__()
        
        kernel_size = 3
        self.n_MSRB = n_MSRB 
       
        # head
        #self.norm = LayerNorm(indim//2, 'Bias_Free')
        modules_head = [conv(2, hiddim, kernel_size)] #3*3 conv
        # body
        modules_body = nn.ModuleList()
        for i in range(n_MSRB):
            modules_body.append(MSRB(hiddim))

        # tail
        modules_tail = [nn.Conv2d(hiddim* (self.n_MSRB + 1), hiddim, 1, padding=0, stride=1), conv(hiddim, 1, kernel_size)]

        self.Edge_Net_head = nn.Sequential(*modules_head)
        self.Edge_Net_body = nn.Sequential(*modules_body)
        self.Edge_Net_tail = nn.Sequential(*modules_tail)

    def forward(self, x, sens_map=None):

        """
        x: (B, C, H, W)
        """
        # turn into real valued
        B, C, H, W = x.shape
        x = x.reshape(B, -1, H, W, 2) #(B, C//2, H, W, 2)
        x = T.reduce_operator(x, sens_map, dim=1) #(B, H, W, 2)
        x = x.permute(0,3,1,2) #(B,2,H,W)

        x = self.Edge_Net_head(x) #(B, hiddim, H, W)
        res = x

        MSRB_out = []
        for i in range(self.n_MSRB):
            x = self.Edge_Net_body[i](x)
            MSRB_out.append(x)
        MSRB_out.append(res)

        res = torch.cat(MSRB_out,1) #(B, hiddim*(self.n_MSRB+1), H, W)
        # decode
        x = self.Edge_Net_tail(res)

        return x


#============================== #============================== #============================== #============================== 

class attHead_image(nn.Module):
    """
    norm + 1*1 conv + 3*3 dconv
    """
    
    def __init__(self, C, C1, layernorm_type= 'BiasFree', bias=False):
        """
        C: input channel 
        C1: output channel after 1*1 conv

        """
        super(attHead_image,self).__init__()
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


class attHead_edge(nn.Module):
    """
    norm + 3*3 dconv
    """
    
    def __init__(self, C, C1, layernorm_type= 'BiasFree', bias=False):
        """
        C: input channel 
        C1: output channel after 1*1 conv

        """
        super(attHead_edge,self).__init__()
        #self.norm = LayerNorm(C, layernorm_type)
        self.conv = nn.Conv2d(C, C1, kernel_size=3, stride=1, padding=1, groups=1, bias=bias)

    def forward(self, x):
        """
        x: (B, C, H, W)
        """ 
        x1 = self.conv(x)  #(B, C1, H, W)

        return x1



class EEM(nn.Module):
    def __init__(self, C, C1, C2, num_heads, bias):
        """
        edge enhance module (EEM)
        C: input channel of image 
        C1: input channel of edge
        C2 : output channel of imhead/ehead
        """

        super(EEM, self).__init__()

        self.imhead = attHead_image(2, 2 * C2)
        self.ehead = attHead_edge(1, C2)
        self.num_heads = num_heads
         
        self.a1 = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.project_out = nn.Conv2d(C2, 2, kernel_size=1, bias=bias)
        

    def edge_att(self, x, e, sens_map=None):
        """
        edge attention
        x: input image (B, C, H, W)
        e: input edge (B, C1, H, W)
        """
        B, C, H, W = x.shape    
        x = x.reshape(B, -1, H, W, 2) #(B, C//2, H, W, 2)
        x = T.reduce_operator(x, sens_map, dim=1) #(B, H, W, 2)
        
        x = x.permute(0,3,1,2) #(B, 2, H, W) #single coil data

        #=================================
        # high freq, edge
        # split into q, k, v 

        q1 = self.imhead(x) #(B, 2*C2, H, W) 
        k_eg = self.ehead(e) #(B, C2, H, W) 
        q_im, v_im = q1.chunk(2, dim=1)

        q_im = rearrange(q_im, 'b (head c) h w -> b head c (h w)', head=self.num_heads) 
        k_eg = rearrange(k_eg, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_im = rearrange(v_im, 'b (head c) h w -> b head c (h w)', head=self.num_heads) #(B, head, C, H*W)

        q_im = torch.nn.functional.normalize(q_im, dim=-1)
        k_eg = torch.nn.functional.normalize(k_eg, dim=-1)
        
        attn = (q_im @ k_eg.transpose(-2, -1)) * self.a1 #(B, head, C, C)
        attn = attn.softmax(dim=-1)
        out = (attn @ v_im)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=H, w=W)

        out = self.project_out(out) #(B, 2, H, W)
        out = x + out

        return out.permute(0,2,3,1).contiguous()


    def forward(self, x, e, sens_map=None):
        
        B, _, H, W = x.shape

        out = self.edge_att(x,e, sens_map) #(B, H, W, 2)
        xout = T.expand_operator(out, sens_map, dim=1) #(B, coils, H, W, 2)
        xout = xout.reshape(B, -1, H, W) #(B, 2*coils, H, W)
        
        return xout #(B, C, H, W)



class attBlock(nn.Module):
    def __init__(self, C, C1, C2, num_heads, bias, isFastmri, isMulticoil):
        super(attBlock, self).__init__()
        
        self.b1 = EEM(C, C1, C2, num_heads, bias) 
        self.dc = dataConsistencyLayer_fastmri_m(isFastmri=isFastmri, isMulticoil=isMulticoil)

    def forward(self, x, e, y, m, sens_map=None):

        x1 = self.b1(x, e, sens_map)
        x1 = self.dc(x1, y, m)
        
        return x1


class eamri_0719_var2(nn.Module):
    """
    12 DAM + transformer block
    """
    def __init__(self, 
                indim=2, 
                edgeFeat=16, 
                attdim=4, 
                num_head=4, 
                growthRates=[16,16,16,16, 16],
                fNums=[16,16,16,16, 16],
                n_DAMs=[1,1,3,1,1], 
                layers=[4,4,4,4,4], 
                num_iters=[3,3,1,3,3], 
                n_MSRB=1,
                isFastmri=False, 
                isMulticoil=False): 
        """
        input:
            C: number of conv in RDB, default 3
            G0: input_channel, default 2
            G1: output_channel, default 2
            n_RDB: number of RDBs 

        """
        super(eamri_0719_var2, self).__init__()
        
        # image module
        self.net1 = rdn_convBlock(inChannel=indim, midChannel=fNums[0], recursiveTime=num_iters[0], isMulticoil=isMulticoil, isFastmri=isFastmri)
        self.net2 = rdn_convBlock(inChannel=indim, midChannel=fNums[1], recursiveTime=num_iters[1], isMulticoil=isMulticoil, isFastmri=isFastmri)
        self.net3 = rdn_convBlock(inChannel=indim, midChannel=fNums[2], recursiveTime=num_iters[2], isMulticoil=isMulticoil, isFastmri=isFastmri)
        self.net4 = rdn_convBlock(inChannel=indim, midChannel=fNums[3], recursiveTime=num_iters[3], isMulticoil=isMulticoil, isFastmri=isFastmri)
        self.net5 = rdn_convBlock(inChannel=indim, midChannel=fNums[4], recursiveTime=num_iters[4], isMulticoil=isMulticoil, isFastmri=isFastmri)

        # edge module
        self.edgeNet = Edge_Net(indim=indim, hiddim=edgeFeat, n_MSRB=n_MSRB) # simple edge block

        # edgeatt module
        self.fuse1 = attBlock(indim, 1, attdim, num_head, bias=False, isFastmri=isFastmri, isMulticoil=isMulticoil) 
        self.fuse2 = attBlock(indim, 1, attdim, num_head, bias=False, isFastmri=isFastmri, isMulticoil=isMulticoil) 
        self.fuse3 = attBlock(indim, 1, attdim, num_head, bias=False, isFastmri=isFastmri, isMulticoil=isMulticoil) 
        self.fuse4 = attBlock(indim, 1, attdim, num_head, bias=False, isFastmri=isFastmri, isMulticoil=isMulticoil) 
       

    def forward(self, x1, y, m, sens_map=None): # (image, kspace, mask)
        """
        input:
            x1: (B, 2, H, W) zero-filled image
            e0: (B, 2, H, W) edge of zim
            y: under-sampled kspace 
            m: mask

        """

        # image stem
        x1 = self.net1(x1, y, m) #(B, 2, H, W)

        # second stage
        x2 = self.net2(x1, y, m) #(B, 2, H, W)
        e2 = self.edgeNet(x1, sens_map) 
        x1 = self.fuse1(x2, e2, y, m, sens_map)

        # third stage
        x2 = self.net3(x1, y, m) #(B, 2, H, W)
        e3 = self.edgeNet(x1, sens_map) 
        x1 = self.fuse2(x2, e3, y, m, sens_map)

        # fourth stage
        x2 = self.net4(x1, y, m) #(B, 2, H, W)
        e4 = self.edgeNet(x1,sens_map) 
        x1 = self.fuse3(x2, e4, y, m, sens_map)

        # fourth stage
        x2 = self.net5(x1, y, m) #(B, 2, H, W)
        e5 = self.edgeNet(x1, sens_map) 
        x1 = self.fuse4(x2, e5, y, m, sens_map)

        return [e2,e3,e4,e5,x1]





