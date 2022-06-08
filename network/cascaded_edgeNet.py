"""
cascaded edge net
"""

import torch
import torch.nn as nn
from .cascadeNetwork import denseConv
from .networkUtil import *
import pdb


class edgeBlock(nn.Module):
    def __init__(self, inFeat, outFeat=16, ks=3):
        super(edgeBlock,self).__init__()
        self.name = 'edge'
        self.conv1 = nn.Conv2d(inFeat, outFeat, ks, padding = 1) 
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(outFeat, inFeat, ks, padding = 1) 

    def forward(self, x):
        
        x1 = self.conv1(x)
        x2 = self.act1(x1)
        x3 = self.conv2(x2)
        return x + x3





class DAM(nn.Module):
    '''
    basic DAM module 
    '''
    def __init__(self, inChannel = 2, fNum = 16, growthRate = 16, layer = 3, dilate = False, activation = 'ReLU', useOri = False, transition = 0, residual = True, is_edge = False):
        super(DAM, self).__init__()
        self.name = 'dam'
        self.addEdge = is_edge
        self.inChannel = inChannel
        self.outChannel = inChannel
        self.transition = transition
        self.residual = residual
        self.inConv = nn.Conv2d(self.inChannel, fNum, 3, padding = 1) # (16,2,3,3)
        if(activation == 'LeakyReLU'):
            self.activ = nn.LeakyReLU()
        elif(activation == 'ReLU'):
            self.activ = nn.ReLU()
        self.denseConv = denseConv(fNum, 3, growthRate, layer - 2, dilationLayer = dilate, activ = activation, useOri = useOri)
        
        if(transition>0):
            self.transitionLayer = transitionLayer(fNum+growthRate*(layer-2), transition, activ = activation)
            self.outConv = convLayer(int((fNum+growthRate*(layer-2))*transition), self.outChannel, activ = activation)
        else:
            self.outConv = convLayer(fNum+growthRate*(layer-2), self.outChannel, activ = activation)


    def forward(self, x, edge=None):
        """
        x1 is the input from the simpleRes, i.e edge 
        """

        x2 = self.inConv(x) #[16,16,3,3]
        x2 = self.denseConv(x2) #(8,64,256,256)
        x2 = self.transitionLayer(x2) #(8,32,256,256)
        x2 = self.outConv(x2) #(8,2,256,256)
        if(self.residual):
            x2 = x2+x[:,:self.outChannel]

        # add edge information
        if self.addEdge:
            x2 = x2 + edge 

        return x2


class DAM1(nn.Module):
    '''
    DAM variant 1
    '''
    def __init__(self, inChannel = 2, fNum = 16, growthRate = 16, layer = 3, dilate = False, activation = 'ReLU', useOri = False, transition = 0, residual = True, is_edge = False):
        super(DAM1, self).__init__()
        self.name = 'dam'
        self.addEdge = is_edge
        self.inChannel = inChannel
        self.outChannel = inChannel
        self.transition = transition
        self.residual = residual
        self.inConv = nn.Conv2d(self.inChannel, fNum, 3, padding = 1) # (16,2,3,3)
        if(activation == 'LeakyReLU'):
            self.activ = nn.LeakyReLU()
        elif(activation == 'ReLU'):
            self.activ = nn.ReLU()
        self.denseConv = denseConv(fNum, 3, growthRate, layer - 2, dilationLayer = dilate, activ = activation, useOri = useOri)
        
        if(transition>0):
            self.transitionLayer = transitionLayer(fNum+growthRate*(layer-2), transition, activ = activation)
            self.outConv = convLayer(int((fNum+growthRate*(layer-2))*transition), self.outChannel, activ = activation)
        else:
            self.outConv = convLayer(fNum+growthRate*(layer-2), self.outChannel, activ = activation)

        self.edgeFuse = nn.Conv2d(2*self.inChannel, self.inChannel, 1)

    def forward(self, x, edge):
        """
        x1 is the input from the simpleRes, i.e edge 
        """

        x2 = self.inConv(x) #[16,16,3,3]
        x2 = self.denseConv(x2) #(8,64,256,256)
        x2 = self.transitionLayer(x2) #(8,32,256,256)
        x2 = self.outConv(x2) #(8,2,256,256)
        if(self.residual):
            x2 = x2+x[:,:self.outChannel]

        # add edge information
        if self.addEdge:
            x2 = self.edgeFuse(torch.cat((x2,edge),dim=1))

        return x2


#===========================
# adapted from seanet
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class MSRB(nn.Module):
    def __init__(self, n_feats, kernel_size_1 = 3, kernel_size_2 = 5, conv=default_conv):
        super(MSRB, self).__init__()

        #n_feats = 64

        self.conv_3_1 = conv(n_feats, n_feats, kernel_size_1)
        self.conv_3_2 = conv(n_feats * 2, n_feats * 2, kernel_size_1)
        self.conv_5_1 = conv(n_feats, n_feats, kernel_size_2)
        self.conv_5_2 = conv(n_feats * 2, n_feats * 2, kernel_size_2)
        self.confusion = nn.Conv2d(n_feats * 4, n_feats, 1, padding=0, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        input_1 = x
        output_3_1 = self.relu(self.conv_3_1(input_1))
        output_5_1 = self.relu(self.conv_5_1(input_1))
        input_2 = torch.cat([output_3_1, output_5_1], 1)
        output_3_2 = self.relu(self.conv_3_2(input_2))
        output_5_2 = self.relu(self.conv_5_2(input_2))
        input_3 = torch.cat([output_3_2, output_5_2], 1)
        output = self.confusion(input_3)
        output += x
        return output


class edgeBlock1(nn.Module):
    # adopted from edgenet of seanet
    def __init__(self, inFeat=2, nFeat =64, ks=3, n_blocks=1, conv=default_conv):
        super(edgeBlock1, self).__init__()
        
        act = nn.ReLU(True)
        self.n_blocks = n_blocks
        
        modules_head = [conv(inFeat, nFeat, ks)] # increase channel

        modules_body = nn.ModuleList()
        for i in range(n_blocks):
            modules_body.append(MSRB(n_feats=nFeat))

        modules_tail = [nn.Conv2d(nFeat * (self.n_blocks + 1), inFeat, 1, padding=0, stride=1)]

        self.Edge_Net_head = nn.Sequential(*modules_head)
        self.Edge_Net_body = nn.Sequential(*modules_body)
        self.Edge_Net_tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.Edge_Net_head(x)
        res = x

        MSRB_out = []
        for i in range(self.n_blocks):
            x = self.Edge_Net_body[i](x)
            MSRB_out.append(x)
        MSRB_out.append(res)

        res = torch.cat(MSRB_out,1)
        x = self.Edge_Net_tail(res)
        return x 



#================================================ 
# main body
class edgeNet(nn.Module):
    def __init__(self, inChannel = 2, d = 5, c = 5, fNum = 16, growthRate = 16, ebChannel=64, dilate = True, activation = 'ReLU', useOri = True, 
                transition = 0.5, trick = 2, globalDense = False, globalResSkip = False, subnetType = 'Dense'):
        """
        recommand model: inChannel=2, dilate = True, useOri=Truem transition=0.5, trick=2, c=5
        inChannel: dimension of the input channel 
        c: number of sub networks
        fNum:
        trick:
        """
        super(edgeNet, self).__init__()
        layers = [] 
        self.dam1 = DAM(inChannel, fNum, growthRate, d, dilate, activation, useOri, transition, residual=True, is_edge= 1)
        self.dam2 = DAM(inChannel, fNum, growthRate, d, dilate, activation, useOri, transition, residual=True, is_edge= 1)
        self.dam3 = DAM(inChannel, fNum, growthRate, d, dilate, activation, useOri, transition, residual=True, is_edge= 1)
        self.dam4 = DAM(inChannel, fNum, growthRate, d, dilate, activation, useOri, transition, residual=True, is_edge= 1)
        self.dam5 = DAM(inChannel, fNum, growthRate, d, dilate, activation, useOri, transition, residual=True, is_edge= 1)

        self.dc1 = DC(trick = 2)
        self.dc2 = DC(trick = 2)
        self.dc3 = DC(trick = 2)
        self.dc4 = DC(trick = 2)
        self.dc5 = DC(trick = 2)

        self.eb1 = edgeBlock(inChannel, ebChannel)
        self.eb2 = edgeBlock(inChannel, ebChannel)
        self.eb3 = edgeBlock(inChannel, ebChannel)
        self.eb4 = edgeBlock(inChannel, ebChannel)
        self.eb5 = edgeBlock(inChannel, ebChannel)

        self.edgeGet = Get_gradient()
        self.lam = 0.9

    def forward(self, x1, y, mask): # (image, kspace, mask)

        e0 = self.edgeGet(x1)

        # first stage
        e0 = self.eb1(e0)
        r1 = self.dam1(x1,e0) 
        z1 = self.dc1(r1,y,mask)
        e1 = self.lam * self.edgeGet(z1) + e0
        
        # second stage
        e1 = self.eb2(e1)
        r2 = self.dam2(z1,e1) 
        z2 = self.dc2(r2,y,mask)
        e2 = self.lam * self.edgeGet(z2) + e1
 
        # third stage
        e2 = self.eb3(e2)
        r3 = self.dam3(z2,e2) 
        z3 = self.dc3(r3,y,mask)
        e3 = self.lam * self.edgeGet(z3) + e2
 
        # fourth stage
        e3 = self.eb4(e3)
        r4 = self.dam4(z3,e3) 
        z4 = self.dc4(r4,y,mask)
        e4 = self.lam * self.edgeGet(z4) + e3
 
        # fifth stage
        e4 = self.eb5(e4)
        r5 = self.dam5(z4,e4) 
        z5 = self.dc5(r5,y,mask)
        e5 = self.lam * self.edgeGet(z5) + e4

        return [e5, z5]


# edgeNet variant 1
class edgeNet_var1(nn.Module):
    def __init__(self, inChannel = 2, d = 5, c = 5, fNum = 16, growthRate = 16, ebChannel=64, dilate = True, activation = 'ReLU', useOri = True, 
                transition = 0.5, trick = 2, globalDense = False, globalResSkip = False, subnetType = 'Dense'):
        """
        recommand model: inChannel=2, dilate = True, useOri=Truem transition=0.5, trick=2, c=5
        inChannel: dimension of the input channel 
        c: number of sub networks
        fNum:
        trick:
        """
        super(edgeNet_var1, self).__init__()
        layers = [] 
        tmp = inChannel
        self.dam1 = DAM(inChannel, fNum, growthRate, d, dilate, activation, useOri, transition, residual=True, is_edge= 1)
        self.dam2 = DAM(inChannel, fNum, growthRate, d, dilate, activation, useOri, transition, residual=True, is_edge= 1)
        self.dam3 = DAM(inChannel, fNum, growthRate, d, dilate, activation, useOri, transition, residual=True, is_edge= 1)
        self.dam4 = DAM(inChannel, fNum, growthRate, d, dilate, activation, useOri, transition, residual=True, is_edge= 1)
        self.dam5 = DAM(inChannel, fNum, growthRate, d, dilate, activation, useOri, transition, residual=True, is_edge= 1)

        self.dc1 = DC(trick = 2)
        self.dc2 = DC(trick = 2)
        self.dc3 = DC(trick = 2)
        self.dc4 = DC(trick = 2)
        self.dc5 = DC(trick = 2)

        self.eb1 = edgeBlock(inChannel, ebChannel)
        self.eb2 = edgeBlock(inChannel, ebChannel)
        self.eb3 = edgeBlock(inChannel, ebChannel)
        self.eb4 = edgeBlock(inChannel, ebChannel)
        self.eb5 = edgeBlock(inChannel, ebChannel)

        self.edgeGet = Get_gradient()
        self.lam = 0.9

    def forward(self, x1, y, mask): # (image, kspace, mask)

        e0 = self.edgeGet(x1)

        # first stage
        e0 = self.eb1(e0)
        r1 = self.dam1(x1,e0) 
        z1 = self.dc1(r1,y,mask)
        e1 = self.lam * self.edgeGet(z1) + e0
        
        # second stage
        e1 = self.eb2(e1)
        r2 = self.dam2(z1,e1) 
        z2 = self.dc2(r2,y,mask)
        e2 = self.lam * self.edgeGet(z2) + e1
 
        # third stage
        e2 = self.eb3(e2)
        r3 = self.dam3(z2,e2) 
        z3 = self.dc3(r3,y,mask)
        e3 = self.lam * self.edgeGet(z3) + e2
 
        # fourth stage
        e3 = self.eb4(e3)
        r4 = self.dam4(z3,e3) 
        z4 = self.dc4(r4,y,mask)
        e4 = self.lam * self.edgeGet(z4) + e3
 
        # fifth stage
        e4 = self.eb5(e4)
        r5 = self.dam5(z4,e4) 
        z5 = self.dc5(r5,y,mask)
        e5 = self.lam * self.edgeGet(z5) + e4

        return [e1, e2, e3, e4, e5, z5] # return all the edges




# edgeNet variant 2
class edgeNet_var2(nn.Module):
    """
    dam 1 + edgeBlock
    """
    def __init__(self, inChannel = 2, d = 5, c = 5, fNum = 16, growthRate = 16, ebChannel=64, dilate = True, activation = 'ReLU', useOri = True, 
                transition = 0.5, trick = 2, globalDense = False, globalResSkip = False, subnetType = 'Dense'):
        """
        recommand model: inChannel=2, dilate = True, useOri=Truem transition=0.5, trick=2, c=5
        inChannel: dimension of the input channel 
        c: number of sub networks
        fNum:
        trick:
        """
        super(edgeNet_var2, self).__init__()
        layers = [] 
        tmp = inChannel
        self.dam1 = DAM1(inChannel, fNum, growthRate, d, dilate, activation, useOri, transition, residual=True, is_edge= 1)
        self.dam2 = DAM1(inChannel, fNum, growthRate, d, dilate, activation, useOri, transition, residual=True, is_edge= 1)
        self.dam3 = DAM1(inChannel, fNum, growthRate, d, dilate, activation, useOri, transition, residual=True, is_edge= 1)
        self.dam4 = DAM1(inChannel, fNum, growthRate, d, dilate, activation, useOri, transition, residual=True, is_edge= 1)
        self.dam5 = DAM1(inChannel, fNum, growthRate, d, dilate, activation, useOri, transition, residual=True, is_edge= 1)

        self.dc1 = DC(trick = 2)
        self.dc2 = DC(trick = 2)
        self.dc3 = DC(trick = 2)
        self.dc4 = DC(trick = 2)
        self.dc5 = DC(trick = 2)

        self.eb1 = edgeBlock(inChannel, ebChannel)
        self.eb2 = edgeBlock(inChannel, ebChannel)
        self.eb3 = edgeBlock(inChannel, ebChannel)
        self.eb4 = edgeBlock(inChannel, ebChannel)
        self.eb5 = edgeBlock(inChannel, ebChannel)

        self.edgeGet = Get_gradient()
        self.lam = 0.9

    def forward(self, x1, y, mask): # (image, kspace, mask)

        e0 = self.edgeGet(x1)

        # first stage
        e0 = self.eb1(e0)
        r1 = self.dam1(x1,e0) 
        z1 = self.dc1(r1,y,mask)
        e1 = self.edgeGet(z1) + self.lam * e0
        
        # second stage
        e1 = self.eb2(e1)
        r2 = self.dam2(z1,e1) 
        z2 = self.dc2(r2,y,mask)
        e2 = self.edgeGet(z2) + self.lam * e1
 
        # third stage
        e2 = self.eb3(e2)
        r3 = self.dam3(z2,e2) 
        z3 = self.dc3(r3,y,mask)
        e3 = self.edgeGet(z3) + self.lam * e2
 
        # fourth stage
        e3 = self.eb4(e3)
        r4 = self.dam4(z3,e3) 
        z4 = self.dc4(r4,y,mask)
        e4 = self.edgeGet(z4) + self.lam * e3
 
        # fifth stage
        e4 = self.eb5(e4)
        r5 = self.dam5(z4,e4) 
        z5 = self.dc5(r5,y,mask)
        e5 = self.edgeGet(z5) + self.lam* e4

        return [e1, e2, e3, e4, e5, z5] # return all the edges


# edgeNet variant 2
class edgeNet_var21(nn.Module):
    """
    original c5
    dam 1 (do not add edge back)
    edge loss is just guided
    """
    def __init__(self, inChannel = 2, d = 5, c = 5, fNum = 16, growthRate = 16, ebChannel=64, dilate = True, activation = 'ReLU', useOri = True, 
                transition = 0.5, trick = 2, globalDense = False, globalResSkip = False, subnetType = 'Dense'):
        """
        recommand model: inChannel=2, dilate = True, useOri=Truem transition=0.5, trick=2, c=5
        inChannel: dimension of the input channel 
        c: number of sub networks
        fNum:
        trick:
        """
        super(edgeNet_var21, self).__init__()
        layers = [] 
        tmp = inChannel
        self.dam1 = DAM1(inChannel, fNum, growthRate, d, dilate, activation, useOri, transition, residual=True, is_edge= 0)
        self.dam2 = DAM1(inChannel, fNum, growthRate, d, dilate, activation, useOri, transition, residual=True, is_edge= 0)
        self.dam3 = DAM1(inChannel, fNum, growthRate, d, dilate, activation, useOri, transition, residual=True, is_edge= 0)
        self.dam4 = DAM1(inChannel, fNum, growthRate, d, dilate, activation, useOri, transition, residual=True, is_edge= 0)
        self.dam5 = DAM1(inChannel, fNum, growthRate, d, dilate, activation, useOri, transition, residual=True, is_edge= 0)

        self.dc1 = DC(trick = 2)
        self.dc2 = DC(trick = 2)
        self.dc3 = DC(trick = 2)
        self.dc4 = DC(trick = 2)
        self.dc5 = DC(trick = 2)

        self.eb1 = edgeBlock(inChannel, ebChannel)
        self.eb2 = edgeBlock(inChannel, ebChannel)
        self.eb3 = edgeBlock(inChannel, ebChannel)
        self.eb4 = edgeBlock(inChannel, ebChannel)
        self.eb5 = edgeBlock(inChannel, ebChannel)

        self.edgeGet = Get_gradient()
        self.lam = 0.9

    def forward(self, x1, y, mask): # (image, kspace, mask)

        e0 = self.edgeGet(x1)

        # first stage
        e0 = self.eb1(e0)
        r1 = self.dam1(x1,e0) 
        z1 = self.dc1(r1,y,mask)
        e1 = self.lam * self.edgeGet(z1) + e0
        
        # second stage
        e1 = self.eb2(e1)
        r2 = self.dam2(z1,e1) 
        z2 = self.dc2(r2,y,mask)
        e2 = self.lam * self.edgeGet(z2) + e1
 
        # third stage
        e2 = self.eb3(e2)
        r3 = self.dam3(z2,e2) 
        z3 = self.dc3(r3,y,mask)
        e3 = self.lam * self.edgeGet(z3) + e2
 
        # fourth stage
        e3 = self.eb4(e3)
        r4 = self.dam4(z3,e3) 
        z4 = self.dc4(r4,y,mask)
        e4 = self.lam * self.edgeGet(z4) + e3
 
        # fifth stage
        e4 = self.eb5(e4)
        r5 = self.dam5(z4,e4) 
        z5 = self.dc5(r5,y,mask)
        e5 = self.lam * self.edgeGet(z5) + e4

        return [e1, e2, e3, e4, e5, z5] # return all the edges



#==============================
#baseline

class simple_edgeBlock(nn.Module):
    def __init__(self, indim, middim):
        """
        indim: input channel

        """
        super(simple_edgeBlock,self).__init__()
        self.conv1 = nn.Conv2d(indim, middim, kernel_size=3, padding = 1, bias=True) 
        self.conv2 = nn.Conv2d(middim, indim, kernel_size=1, padding = 0, bias=True) 
        self.act1 = nn.ReLU()

    def forward(self, x):
        
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        out = self.act1(x2)
        
        return out




class simpleFuse(nn.Module):
    def __init__(self, indim, outdim=2):
        """
        simple fusion module using 1*1 conv and dc
        indim: input_channel
        outdim: output_channel, default=2
        """
        super(simpleFuse, self).__init__()
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


class casEdgeNet(nn.Module):
    """
    casEdgeNet
    """
    def __init__(self, indim, middim):
        """
        input:
            indim: input_channel,default 2
            middim: mid channel, default 32

        """
        super(casEdgeNet, self).__init__()
        
        self.im_block1 = RDG1(4, indim, middim, middim//4, 3)
        self.im_block2 = RDG1(4, indim, middim, middim//4, 3)
        self.im_block3 = RDG1(4, indim, middim, middim//4, 3)
        self.im_block4 = RDG1(4, indim, middim, middim//4, 3)
        self.im_block5 = RDG1(4, indim, middim, middim//4, 3)

        self.edge_block1 = RDG2(4, 1, middim, middim//4,3)
        self.edge_block2 = RDG2(4, 1, middim, middim//4,3)
        self.edge_block3 = RDG2(4, 1, middim, middim//4,3)
        self.edge_block4 = RDG2(4, 1, middim, middim//4,3)
        self.edge_block5 = RDG2(4, 1, middim, middim//4,3)

        self.fuse1 = simpleFuse(middim)
        self.fuse2 = simpleFuse(middim)
        self.fuse3 = simpleFuse(middim)
        self.fuse4 = simpleFuse(middim)
        self.fuse5 = simpleFuse(middim)

        self.lam = 0.9

    def forward(self, x0, e0, y, m): # (image, kspace, mask)
        """
        input:
            x0: (B, 2, H, W) zero-filled image
            e0: (B, 1, H, W) edge of zim
            y: under-sampled kspace 
            m: mask

        """

        # first stage
        x1 = self.im_block1(x0) #(B, middim, H, W)
        ef1, eo1 = self.edge_block1(e0) #(B, mid, H, W) and (B, 1, H, W)
        x1 = self.fuse1(x1,ef1,y,m) #(B,2,H,W)

        # second stage
        x2 = self.im_block2(x1) #(B, middim, H, W)
        ef2, eo2 = self.edge_block2(eo1) #(B, 2, H, W)
        x2 = self.fuse2(x2,ef2,y,m) #(B,2,H,W)

        # third stage
        x3 = self.im_block3(x2) #(B, 2, H, W)
        ef3, eo3 = self.edge_block3(eo2) #(B, 2, H, W)
        x3 = self.fuse3(x3,ef3,y,m) #(B,2,H,W)

        # fourth stage
        x4 = self.im_block4(x3) #(B, 2, H, W)
        ef4, eo4 = self.edge_block4(eo3) #(B, 2, H, W)
        x4 = self.fuse4(x4,ef4,y,m) #(B,2,H,W)

        # fifth stage
        x5 = self.im_block5(x4) #(B, 2, H, W)
        ef5, eo5 = self.edge_block5(eo4) #(B, 2, H, W)
        x5 = self.fuse5(x5,ef5,y,m) #(B,2,H,W)

        return [eo1, eo2, eo3, eo4, eo5, x5] # return all the edges


class net_0426(nn.Module):
    """
    casEdgeNet without edge block
    """
    def __init__(self, indim, middim):
        """
        input:
            indim: input_channel,default 2
            middim: mid channel, default 32

        """
        super(net_0426, self).__init__()
        
        self.im_block1 = RDG1(4, indim, indim, middim//4, 6)
        self.im_block2 = RDG1(4, indim, indim, middim//4, 6)
        self.im_block3 = RDG1(4, indim, indim, middim//4, 6)
        self.im_block4 = RDG1(4, indim, indim, middim//4, 6)
        self.im_block5 = RDG1(4, indim, indim, middim//4, 6)

        self.dc1 = dataConsistencyLayer_fastmri()
        self.dc2 = dataConsistencyLayer_fastmri()
        self.dc3 = dataConsistencyLayer_fastmri()
        self.dc4 = dataConsistencyLayer_fastmri()
        self.dc5 = dataConsistencyLayer_fastmri()


    def forward(self, x0, y, m): # (image, kspace, mask)
        """
        input:
            x0: (B, 2, H, W) zero-filled image
            e0: (B, 2, H, W) edge of zim
            y: under-sampled kspace 
            m: mask

        """

        # first stage
        x1 = self.im_block1(x0) #(B, 2, H, W)
        x1 = self.dc1(x1, y, m) #(B, 2, H, W)

        x1 = self.im_block2(x1) #(B, 2, H, W)
        x1 = self.dc2(x1, y, m) #(B, 2, H, W)

        x1 = self.im_block3(x1) #(B, 2, H, W)
        x1 = self.dc3(x1, y, m) #(B, 2, H, W)

        x1 = self.im_block4(x1) #(B, 2, H, W)
        x1 = self.dc4(x1, y, m) #(B, 2, H, W)

        x1 = self.im_block5(x1) #(B, 2, H, W)
        x1 = self.dc5(x1, y, m) #(B, 2, H, W)

        return x1



