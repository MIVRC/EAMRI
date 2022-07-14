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

    def forward(self,x1, y, m):

        for _ in range(self.num_iter):
            for i, layer in enumerate(self.layers):
                if i < self.n_DAM: 
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
        modules_head = [conv(indim, hiddim, kernel_size)] #3*3 conv

        # body
        modules_body = nn.ModuleList()
        for i in range(n_MSRB):
            modules_body.append(MSRB(hiddim))

        # tail
        modules_tail = [nn.Conv2d(hiddim* (self.n_MSRB + 1), hiddim, 1, padding=0, stride=1), conv(hiddim, 1, kernel_size)]

        self.Edge_Net_head = nn.Sequential(*modules_head)
        self.Edge_Net_body = nn.Sequential(*modules_body)
        self.Edge_Net_tail = nn.Sequential(*modules_tail)

    def forward(self, x):
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







class edgeBlock(nn.Module):
    def __init__(self, inFeat=1, outFeat=16, ks=3):
        super(edgeBlock,self).__init__()
        
        # encode
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

#============================== #============================== #============================== #============================== 

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
        x1 = self.norm(x) #(B, H, W, C)
        x1 = self.conv1(x) # (B, C1, H, W)
        x1 = self.conv2(x1)  #(B, C1, H, W)
        return x1


class attHead1(nn.Module):
    """
    norm + 3*3 dconv
    """
    
    def __init__(self, C, C1, layernorm_type= 'BiasFree', bias=False):
        """
        C: input channel 
        C1: output channel after 1*1 conv

        """
        super(attHead1,self).__init__()
        self.norm = LayerNorm(C, layernorm_type)
        self.conv = nn.Conv2d(C, C1, kernel_size=3, stride=1, padding=1, groups=1, bias=bias)

    def forward(self, x):
        """
        x: (B, C, H, W)
        """ 
        x1 = self.norm(x) #(B, H, W, C)
        x1 = self.conv(x1)  #(B, C1, H, W)

        return x1


class attHead2(nn.Module):
    """
    norm + 3*3 dconv
    """
    
    def __init__(self, C, C1, layernorm_type= 'BiasFree', bias=False):
        """
        C: input channel 
        C1: output channel after 1*1 conv

        """
        super(attHead2,self).__init__()
        self.norm = LayerNorm(C, layernorm_type)

    def forward(self, x):
        """
        x: (B, C, H, W)
        """ 
        x1 = self.norm(x) #(B, H, W, C)
        #x1 = self.conv(x1)  #(B, C1, H, W)

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
        self.ehead = attHead1(C1, C2)
        self.num_heads = num_heads
         
        self.a1 = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.a2 = nn.Parameter(torch.ones(num_heads, 1, 1))

        # ==========================
        # img att
        self.im_dim = 12
        #self.imhead1 = attHead2(C,self.im_dim)
        self.ws = 8 # window size
        self.im_qkv = nn.Linear(C, self.im_dim*3, bias=bias)
        self.im_proj = nn.Linear(self.im_dim, self.im_dim)


        # ==========================
        # final
        self.project_out = nn.Conv2d(C2, C, kernel_size=1, bias=bias)
        

    def edge_att(self, x, e):
        """
        edge attention
        x: input image (B, C, H, W)
        e: input edge (B, C1, H, W)
        """
        B, C, H, W = x.shape    

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

        return out 


    def forward(self, x, e):

        out_eg = self.edge_att(x,e)
        out = self.project_out(out_eg) #(B, C, H, W)
        xout = x + out
        
        return xout #(B, C, H, W)



class attBlock(nn.Module):
    def __init__(self, C, C1, C2, num_heads, bias, isFastmri, isMulticoil):
        super(attBlock, self).__init__()
        
        self.b1 = EEM(C, C1, C2, num_heads, bias) 
        self.dc = dataConsistencyLayer_fastmri_m(isFastmri=isFastmri, isMulticoil=isMulticoil)

    def forward(self, x, e, y, m):

        x1 = self.b1(x, e)
        x1 = self.dc(x1, y, m)
        
        return x1


class eamri_0714(nn.Module):
    """
    12 DAM + transformer block
    """
    def __init__(self, 
                indim=2, 
                edgeFeat=16, 
                attdim=4, 
                num_head=4, 
                fNums=[16,16,16,16],
                n_DAMs=[1,1,3,1], 
                layers=[4,4,4,4], 
                num_iters=[3,3,1,3], 
                isFastmri=False, 
                isMulticoil=False): 
        """
        input:
            C: number of conv in RDB, default 3
            G0: input_channel, default 2
            G1: output_channel, default 2
            n_RDB: number of RDBs 

        """
        super(eamri_0714, self).__init__()
        
        # image module
        self.net1 = convBlock(indim=indim, fNum=fNums[0], n_DAM=n_DAMs[0], layer=layers[0], num_iter=num_iters[0], isMulticoil=isMulticoil, isFastmri=isFastmri)
        self.net2 = convBlock(indim=indim, fNum=fNums[1], n_DAM=n_DAMs[1], layer=layers[1], num_iter=num_iters[1], isMulticoil=isMulticoil, isFastmri=isFastmri)
        self.net3 = convBlock(indim=indim, fNum=fNums[2], n_DAM=n_DAMs[2], layer=layers[2], num_iter=num_iters[2], isMulticoil=isMulticoil, isFastmri=isFastmri)
        self.net4 = convBlock(indim=indim, fNum=fNums[3], n_DAM=n_DAMs[3], layer=layers[3], num_iter=num_iters[3], isMulticoil=isMulticoil, isFastmri=isFastmri)

        # edge module
        self.edgeNet = Edge_Net(indim=indim, hiddim=edgeFeat, n_MSRB=1) # simple edge block

        # edgeatt module
        self.fuse = attBlock(indim, 1, attdim, num_head, bias=False, isFastmri=isFastmri, isMulticoil=isMulticoil) 
       

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
        e2 = self.edgeNet(x1) 
        x1 = checkpoint.checkpoint(self.fuse, x2, e2, y, m)

        # third stage
        x2 = self.net3(x1, y, m) #(B, 2, H, W)
        e3 = self.edgeNet(x1) 
        x1 = checkpoint.checkpoint(self.fuse, x2, e3, y, m)

        # fourth stage
        x2 = self.net4(x1, y, m) #(B, 2, H, W)
        e4 = self.edgeNet(x1) 
        x1 = checkpoint.checkpoint(self.fuse, x2, e4, y, m)

        return [e2,e3,e4,x1]





