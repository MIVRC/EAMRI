import sys
sys.path.append('..')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from einops import rearrange
from typing import List, Optional, Tuple
import numbers
from torch.nn.parameter import Parameter
from einops import rearrange
from fastmri.data import transforms_simple as T
import pdb



#=================================================================
# adapted from seanet
def default_conv(in_channels, out_channels, kernel_size, dilate=1, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=dilate, dilation=dilate, bias=bias) 



class Get_gradient(nn.Module):
    def __init__(self):
        super(Get_gradient, self).__init__()
        kernel_v = [[0, -1, 0], 
                    [0, 0, 0], 
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0], 
                    [-1, 0, 1], 
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False)
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False)

    def forward(self, x):
        """
        input: 
            x (B, H, W) or (B, 2, H, W)
        output:
            x (B, H, W) or (B, 2, H, W)

        """
        if len(x.shape) == 3:
            x0_v = F.conv2d(x.unsqueeze(1), self.weight_v, padding = 1)
            x0_h = F.conv2d(x.unsqueeze(1), self.weight_h, padding = 1)
            x = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2)) + 0.
        elif len(x.shape) == 4:
            assert x.shape[1] == 2, "invalid input when extract gradient"
            x0 = x[:, 0]
            x1 = x[:, 1]
            x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding = 1)
            x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding = 1)
            x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding = 1)
            x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding = 1)
            x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2)) + 0.
            x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2)) + 0.
            x = torch.cat([x0, x1], dim=1)
        else:
            raise NotImplementedError
    
        return x



##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
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








#======================================
# RDB
class one_conv(nn.Module):
    """
    input: B, G0, H, W
    output: B, G0+G, H, W
    """
    def __init__(self, G0, G):
        super(one_conv, self).__init__()
        self.conv = nn.Conv2d(G0, G, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
    def forward(self, x):
        output = self.relu(self.conv(x))
        return torch.cat((x, output), dim=1)


class RDB(nn.Module):
    def __init__(self, C, G0, G):
        """
        residual dense block
        C: number of conv
        G0: input channel
        G: adding channel 

        input: B, G0, H, W
        output: B, G0, H, W
        """

        super(RDB, self).__init__()
        convs = []
        for i in range(C):
            convs.append(one_conv(G0+i*G, G))
        self.conv = nn.Sequential(*convs)
        self.LFF = nn.Conv2d(G0+C*G, G0, kernel_size=1, stride=1, padding=0, bias=True)
    def forward(self, x):
        out = self.conv(x)
        lff = self.LFF(out)
        return lff + x




class RDG(nn.Module):
    def __init__(self, C, G0, G, n_RDB):
        """
        C: number of conv in RDB
        G0: input_channel 
        G: adding channel
        n_RDB: number of RDBs 
        """
        super(RDG, self).__init__()
        self.n_RDB = n_RDB
        RDBs = []
        for i in range(n_RDB):
            RDBs.append(RDB(C, G0, G))
        self.RDB = nn.Sequential(*RDBs)
        self.conv = nn.Conv2d(G0*n_RDB, G0, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        buffer = x
        temp = []
        for i in range(self.n_RDB):
            buffer = self.RDB[i](buffer)
            temp.append(buffer)
        buffer_cat = torch.cat(temp, dim=1)
        out = self.conv(buffer_cat)
        return out 



#==========================================
# sknet
class SKConv_re(nn.Module):
    def __init__(self, in_feat=2, mid_feat=32, WH=0, M=2, G=8, r=16 ,stride=1, L=16):
        """ Constructor, my revised skconv
        Args:
            in_feat: input channel dimensionality, we have 2 here 
            mid_feat: output channel dimensionality, we have 2 here 
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs. we have 2 branch 
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv_re, self).__init__()
        self.M = M
        self.in_feat = in_feat
        self.mid_feat = mid_feat
        self.split_convs = nn.ModuleList([])
        self.head1 = nn.Conv2d(in_feat, mid_feat, 1)
        self.head2 = nn.Conv2d(in_feat, mid_feat, 1)

        for i in range(M): # for each branch
            self.split_convs.append(
                    nn.Sequential(
                        nn.Conv2d(mid_feat, mid_feat, kernel_size=3, stride=stride, padding=1+i, dilation=1+i, bias=False), # use 3*3 kernel
                        nn.BatchNorm2d(mid_feat),
                        nn.ReLU(inplace=False)
            ))

        d = max(int(mid_feat/r), L) # max(32/16, 16) = 16
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(nn.Conv2d(mid_feat, d, kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(d),
                                nn.ReLU(inplace=False))
 
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Conv2d(d,mid_feat,kernel_size=1,stride=1) 
            )
        self.softmax = nn.Softmax(dim=1)
        self.tail = nn.Conv2d(mid_feat,in_feat,1)

    def forward(self, x1, x2):
        """
        x1: the first recurrent branch
        x2: the second recurrent branch
        """
       
        batch_size = x1.shape[0]
        # increase channels using 1*1 kernel
        fea1 = self.head1(x1) # (8,32,256,256)
        fea2 = self.head2(x2) # (8.32,256,256)

        feats = []
        # split, conv
        for i, conv in enumerate(self.split_convs): # using 3*3 conv
            if i == 0:
                fea = conv(fea1) #(8,32,256,256)
            else:
                fea = conv(fea2)
            feats.append(fea)
        
        feas = torch.cat(feats, dim=1) #(8,2,32,256,256)
        feas = feas.view(batch_size, self.M, self.mid_feat, feas.shape[2], feas.shape[3]) # (8,2,32,256,256)
        
        # add the splits
        fea_U = torch.sum(feas, dim=1) # (8,32,256,256)
        fea_S = self.gap(fea_U) #(8,32,256,256)
        fea_Z = self.fc(fea_S) # (8,d,256,256)
        
        attention_vectors = [fc(fea_Z) for fc in self.fcs] # d -> 32 
        attention_vectors = torch.cat(attention_vectors,dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.M, self.mid_feat, 1, 1) # (8,2,32,1,1)
        attention_vectors = self.softmax(attention_vectors) 
        
        fea_V = (feas * attention_vectors).sum(dim=1)
        res = self.tail(fea_V)

        return res 


class ASIM(nn.Module):
    def __init__(self, in_feat=2, mid_feat=32, M=2, r=16 ,stride=1, L=16):
        """ Constructor, my revised skconv
        Args:
            in_feat: input channel dimensionality, we have 2 here 
            mid_feat: output channel dimensionality, we have 2 here 
            M: the number of branchs. we have 2 branch 
            r: the reduction radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(ASIM, self).__init__()
        self.M = M
        self.in_feat = in_feat
        self.mid_feat = mid_feat
        self.split_convs = nn.ModuleList([])

        for i in range(M): # for each branch
            self.split_convs.append(
                    nn.Sequential(
                        nn.Conv2d(in_feat, mid_feat, kernel_size=1, bias=False), # use 1*1 kernel
                        nn.BatchNorm2d(mid_feat),
                        nn.ReLU(inplace=False)
            ))

        d = max(int(mid_feat/r), L) # max(32/16, 16) = 16
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(nn.Conv2d(mid_feat, d, kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(d),
                                nn.ReLU(inplace=False))
 
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Conv2d(d,mid_feat,kernel_size=1,stride=1) 
            )
        self.softmax = nn.Softmax(dim=1)
        self.tail = nn.Conv2d(mid_feat,in_feat,1)

    def forward(self, x1, x2):
        """
        x1: the first recurrent branch
        x2: the second recurrent branch
        """
       
        batch_size = x1.shape[0]
        feats = []
        # split, conv
        for i, conv in enumerate(self.split_convs): # using 3*3 conv
            if i == 0:
                fea = conv(x1) #(8,32,256,256)
            else:
                fea = conv(x2)
            feats.append(fea)
        
        feas = torch.cat(feats, dim=1) #(8,2,32,256,256)
        feas = feas.view(batch_size, self.M, self.mid_feat, feas.shape[2], feas.shape[3]) # (8,2,32,256,256)
        
        # add the splits
        fea_U = torch.sum(feas, dim=1) # (8,32,256,256)
        fea_S = self.gap(fea_U) #(8,32,1,1)
        fea_Z = self.fc(fea_S) # (8,d,1,1)
        
        attention_vectors = [fc(fea_Z) for fc in self.fcs] # d -> 32 
        attention_vectors = torch.cat(attention_vectors,dim=1) # (8,64,1,1)
        attention_vectors = attention_vectors.view(batch_size, self.M, self.mid_feat, 1, 1) # (8,2,32,1,1)
        attention_vectors = self.softmax(attention_vectors) # (8,2,32,1,1)
        
        fea_V = (feas * attention_vectors).sum(dim=1)
        res = self.tail(fea_V)

        return res 






class SKUnit(nn.Module):
    def __init__(self, in_features, out_features, WH, M, G, r, mid_features=None, stride=1, L=32):
        """ Constructor
        Args:
            in_features: input channel dimensionality.
            out_features: output channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            mid_features: the channle dim of the middle conv with stride not 1, default out_features/2.
            stride: stride.
            L: the minimum dim of the vector z in paper.
        """
        super(SKUnit, self).__init__()
        if mid_features is None:
            mid_features = int(out_features/2)
        self.feas = nn.Sequential(
            SKConv_re(mid_features, WH, M, G, r, stride=stride, L=L),
            nn.BatchNorm2d(mid_features),
            nn.Conv2d(mid_features, out_features, 1, stride=1),
            nn.BatchNorm2d(out_features)
        )
        if in_features == out_features: # when dim not change, in could be added diectly to out
            self.shortcut = nn.Sequential()
        else: # when dim not change, in should also change dim to be added to out
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_features, out_features, 1, stride=stride),
                nn.BatchNorm2d(out_features)
            )
    
    def forward(self, x):
        fea = self.feas(x)
        return fea + self.shortcut(x)



#==========================================
# feedback block

def activation(act_type='relu', inplace=True, slope=0.2, n_prelu=1):
    act_type = act_type.lower()
    layer = None
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=slope)
    else:
        raise NotImplementedError('[ERROR] Activation layer [%s] is not implemented!'%act_type)
    return layer

def norm(n_feature, norm_type='bn'):
    norm_type = norm_type.lower()
    layer = None
    if norm_type =='bn':
        layer = nn.BatchNorm2d(n_feature)
    else:
        raise NotImplementedError('[ERROR] Normalization layer [%s] is not implemented!'%norm_type)
    return layer

def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None

    layer = None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('[ERROR] Padding layer [%s] is not implemented!'%pad_type)
    return layer


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('[ERROR] %s.sequential() does not support OrderedDict'%sys.modules[__name__])
        else:
            return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module:
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def DeconvBlock(in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, padding=0, \
                act_type='relu', norm_type='bn', pad_type='zero', mode='CNA'):
    assert (mode in ['CNA', 'NAC']), '[ERROR] Wrong mode in [%s]!'%sys.modules[__name__]

    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=bias)

    if mode == 'CNA':
        act = activation(act_type) if act_type else None
        n = norm(out_channels, norm_type) if norm_type else None
        return sequential(p, deconv, n, act)
    elif mode == 'NAC':
        act = activation(act_type, inplace=False) if act_type else None
        n = norm(in_channels, norm_type) if norm_type else None
        return sequential(n, act, p, deconv)

def ConvBlock(in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, valid_padding=True, padding=0,\
              act_type='relu', norm_type='bn', pad_type='zero', mode='CNA'):
    assert (mode in ['CNA', 'NAC']), '[ERROR] Wrong mode in [%s]!'%sys.modules[__name__]

    if valid_padding:
        padding = get_valid_padding(kernel_size, dilation)
    else:
        pass
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)

    if mode == 'CNA':
        act = activation(act_type) if act_type else None
        n = norm(out_channels, norm_type) if norm_type else None
        return sequential(p, conv, n, act)
    elif mode == 'NAC':
        act = activation(act_type, inplace=False) if act_type else None
        n = norm(in_channels, norm_type) if norm_type else None
        return sequential(n, act, p, conv)

def get_valid_padding(kernel_size, dilation):
    """
    Padding value to remain feature size.
    """
    kernel_size = kernel_size + (kernel_size-1)*(dilation-1)
    padding = (kernel_size-1) // 2
    return padding

class FeedbackBlock(nn.Module):
    def __init__(self, num_features, num_groups, upscale_factor, act_type, norm_type):
        super(FeedbackBlock, self).__init__()
        if upscale_factor == 2:
            stride = 2
            padding = 2
            kernel_size = 6
        elif upscale_factor == 3:
            stride = 3
            padding = 2
            kernel_size = 7
        elif upscale_factor == 4:
            stride = 4
            padding = 2
            kernel_size = 8
        elif upscale_factor == 8:
            stride = 8
            padding = 2
            kernel_size = 12

        self.num_groups = num_groups

        self.compress_in = ConvBlock(2*num_features, num_features,
                                     kernel_size=1,
                                     act_type=act_type, norm_type=norm_type)

        self.upBlocks = nn.ModuleList()
        self.downBlocks = nn.ModuleList()
        self.uptranBlocks = nn.ModuleList()
        self.downtranBlocks = nn.ModuleList()
        
        # upBlocks : deconv + deconv + deconv + ... + deconv
        # uptranBlocks : conv + conv + ... + conv (total num_groups - 1 conv)
        # downBlocks : conv  + conv + conv + ... + conv
        # downtranBlocks: conv + conv + ... conv 
        for idx in range(self.num_groups): 
            self.upBlocks.append(DeconvBlock(num_features, num_features,
                                             kernel_size=kernel_size, stride=stride, padding=padding,
                                             act_type=act_type, norm_type=norm_type))
            self.downBlocks.append(ConvBlock(num_features, num_features,
                                             kernel_size=kernel_size, stride=stride, padding=padding,
                                             act_type=act_type, norm_type=norm_type, valid_padding=False))
            if idx > 0:
                self.uptranBlocks.append(ConvBlock(num_features*(idx+1), num_features,
                                                   kernel_size=1, stride=1,
                                                   act_type=act_type, norm_type=norm_type))
                self.downtranBlocks.append(ConvBlock(num_features*(idx+1), num_features,
                                                     kernel_size=1, stride=1,
                                                     act_type=act_type, norm_type=norm_type))

        self.compress_out = ConvBlock(num_groups*num_features, num_features,
                                      kernel_size=1,
                                      act_type=act_type, norm_type=norm_type)

        self.should_reset = True
        self.last_hidden = None

    def forward(self, x):
        if self.should_reset:
            #self.last_hidden = torch.zeros(x.size()).cuda()

            self.last_hidden = torch.zeros(x.size())
            self.last_hidden.copy_(x)
            self.should_reset = False

        x = torch.cat((x, self.last_hidden), dim=1)
        x = self.compress_in(x)

        lr_features = [] # low resolution features
        hr_features = [] # high resolution features
        lr_features.append(x)

        for idx in range(self.num_groups):
            LD_L = torch.cat(tuple(lr_features), 1)    # when idx == 0, lr_features == [x]
            if idx > 0:
                LD_L = self.uptranBlocks[idx-1](LD_L)
            LD_H = self.upBlocks[idx](LD_L)

            hr_features.append(LD_H)

            LD_H = torch.cat(tuple(hr_features), 1)
            if idx > 0:
                LD_H = self.downtranBlocks[idx-1](LD_H)
            LD_L = self.downBlocks[idx](LD_H)

            lr_features.append(LD_L)

        del hr_features
        output = torch.cat(tuple(lr_features[1:]), 1)   # leave out input x, i.e. lr_features[0]
        output = self.compress_out(output)

        self.last_hidden = output

        return output

    def reset_state(self):
        self.should_reset = True





#==========================================
class denseConv(nn.Module):
    def __init__(self, 
            inChannel=16, 
            kernelSize=3, 
            growthRate=16, 
            layer=4, 
            inceptionLayer = False, 
            dilationLayer = False, 
            activ = 'ReLU', 
            useOri = False):

        super(denseConv, self).__init__()
        dilate = 1
        if(dilationLayer):
            dilateMulti = 2
        else:
            dilateMulti = 1
        pad = int((kernelSize-1)/2)
        self.denselayer = layer
        templayerList = []
        for i in range(0, self.denselayer): #0,1
            if(useOri):
                tempLayer = denseBlockLayer_origin(inChannel=inChannel+growthRate*i, 
                                            outChannel=growthRate,
                                            kernelSize=kernelSize,
                                            bottleneckChannel=inChannel,
                                            dilateScale=dilate, activ=activ)
            else:
                tempLayer = new_denseBlockLayer(indim=inChannel+growthRate*i, middim=inChannel, outdim=inChannel+growthRate*i)


            dilate = dilate * dilateMulti
            templayerList.append(tempLayer)
        self.layerList = nn.ModuleList(templayerList)
            
    def forward(self,x):
        for i in range(0, self.denselayer):
            tempY = self.layerList[i](x)
            x = torch.cat((x, tempY), 1)
            
        return x.contiguous()




class DAM(nn.Module):
    '''
    basic DAM module 
    '''
    def __init__(self, inChannel = 2, 
                        fNum = 16, 
                        growthRate = 16, 
                        layer = 3, 
                        dilate = False, 
                        activation = 'ReLU', 
                        useOri = False, 
                        transition = 0, 
                        residual = True):


        super(DAM, self).__init__()
        self.inChannel = inChannel
        self.outChannel = inChannel
        self.transition = transition
        self.residual = residual

        self.inConv = nn.Conv2d(self.inChannel, fNum, 3, padding = 1) # (16,2,3,3)

        if(activation == 'LeakyReLU'):
            self.activ = nn.LeakyReLU()
        elif(activation == 'ReLU'):
            self.activ = nn.ReLU()
        elif(activation == 'GELU'):
            self.activ = nn.GELU()

        self.denseConv = denseConv(inChannel=fNum, \
                                    kernelSize=3,\
                                    growthRate=growthRate, \
                                    layer = layer - 2, \
                                    dilationLayer = dilate,\
                                    activ = activation, \
                                    useOri = useOri)
       

        if(transition>0):
            self.transitionLayer = transitionLayer(inChannel=fNum+growthRate*(layer-2), \
                                                    compressionRate = transition, \
                                                    activ = activation)

            self.outConv = convLayer(int((fNum+growthRate*(layer-2))*transition), self.outChannel, activ = activation)

        else:
            self.outConv = convLayer(fNum+growthRate*(layer-2), self.outChannel, activ = activation)


    def forward(self, x):

        x2 = self.inConv(x) #[16,16,3,3]
        x2 = self.denseConv(x2) #(8,64,256,256)
        x2 = self.transitionLayer(x2) #(8,32,256,256)
        x2 = self.outConv(x2) #(8,2,256,256)
    
        if(self.residual):
            x2 = x2+x[:,:self.outChannel]

        return x2







# Dense Block in DAM
class denseBlockLayer(nn.Module):
    def __init__(self,inChannel=64, outChannel=64, kernelSize=3, inception = False, dilateScale = 1, activ = 'ReLU'):
        super(denseBlockLayer, self).__init__()
        self.useInception = inception

        if(self.useInception):
            self.conv1 = nn.Conv2d(inChannel,outChannel,3,padding = 1)
            self.conv2 = nn.Conv2d(inChannel,outChannel,5,padding = 2)
            self.conv3 = nn.Conv2d(inChannel,outChannel,7,padding = 3)
            if(activ == 'LeakyReLU'):
                self.relu = nn.LeakyReLU()
            else:
                self.relu = nn.ReLU()
            self.conv4 = nn.Conv2d(outChannel*3,outChannel,1,padding = 0)
            #self.relu2 = nn.ReLU()
        else:
            pad = int(dilateScale * (kernelSize - 1) / 2)
            
            self.conv = nn.Conv2d(inChannel,outChannel,kernelSize,padding = pad, dilation = dilateScale)
            if(activ == 'LeakyReLU'):
                self.relu = nn.LeakyReLU()
            else:
                self.relu = nn.ReLU()
            
    def forward(self,x):
        if(self.useInception):
            y2 = x
            y3_1 = self.conv1(y2)
            y3_2 = self.conv1(y2)
            y3_3 = self.conv1(y2)
            y4 = torch.cat((y3_1,y3_2,y3_3),1)
            y4 = self.relu(y4)
            y5 = self.conv4(y4)
            y_ = self.relu(y5)
        else:
            y2 = self.conv(x)
            y_ = self.relu(y2)
            
            
        return y_
    
class denseBlock(nn.Module):
    def __init__(self, inChannel=64, outChannel=64, kernelSize=3, growthRate=16, layer=4, inceptionLayer = False, dilationLayer = False, activ = 'ReLU'):
        super(denseBlock, self).__init__()
        dilate = 1
        if(dilationLayer):
            dilateMulti = 2
        else:
            dilateMulti = 1
        pad = int((kernelSize-1)/2)
        self.layer = layer
        templayerList = []
        for i in range(0, layer):
            tempLayer = denseBlockLayer(inChannel+growthRate*i,growthRate,kernelSize,inceptionLayer,dilate,activ)
            dilate = dilate * dilateMulti
            templayerList.append(tempLayer)
        self.layerList = nn.ModuleList(templayerList)
        self.outputLayer = denseBlockLayer(inChannel+growthRate*layer,outChannel,kernelSize,inceptionLayer,1,activ)
        self.bn = nn.BatchNorm2d(outChannel)
            
    def forward(self,x):
        for i in range(0, self.layer):
            tempY = self.layerList[i](x)
            x = torch.cat((x, tempY), 1)
        y = self.outputLayer(x)
        y = self.bn(y)
            
        return y
    
class SELayer(nn.Module):
    def __init__(self,channel,height,width):
        super(SELayer, self).__init__()
        self.inShape = [channel,height,width]
        self.globalPooling = nn.AvgPool2d((self.inShape[1],self.inShape[2]))
        self.fc1 = nn.Conv2d(self.inShape[0],int(self.inShape[0]/2),1)
        self.fc2 = nn.Conv2d(int(self.inShape[0]/2),self.inShape[0],1)
    
    def forward(self,x):
        v1 = self.globalPooling(x)
        v2 = self.fc1(v1)
        v3 = self.fc2(v2)
        
        f = x*v3
        
        return f
   

class denseBlockLayer_origin(nn.Module):
    """
    bn + relu + conv + bn + relu + conv
    """
    def __init__(self,inChannel=64, outChannel=64, kernelSize = 3, bottleneckChannel = 64, dilateScale = 1, activ = 'ReLU'):
        super(denseBlockLayer_origin, self).__init__()
        
        pad = int((kernelSize-1)/2)

        self.bn = nn.BatchNorm2d(inChannel)
        if(activ == 'LeakyReLU'):
            self.relu = nn.LeakyReLU()
        else:
            self.relu = nn.ReLU()
        self.conv = nn.Conv2d(inChannel,bottleneckChannel,1, bias=False)  # 1*1 conv
        self.bn2 = nn.BatchNorm2d(bottleneckChannel)
        if(activ == 'LeakyReLU'):
            self.relu2 = nn.LeakyReLU()
        else:
            self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(bottleneckChannel,outChannel,kernelSize,padding = dilateScale,dilation=dilateScale) # #*3 conv
        
            
    def forward(self,x):
        
        x1 = self.bn(x) #(B, indim, H, W)
        x1 = self.relu(x1)
        x1 = self.conv(x1) #(B, middim, H, W)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)
        y = self.conv2(x1) #(B, outdim, H, W)
            
        return y





class denseBlock_origin(nn.Module):
    def __init__(self, inChannel=64, kernelSize=3, growthRate=16, layer=4, bottleneckMulti = 4, dilationLayer = False, activ = 'ReLU'):
        super(denseBlock_origin, self).__init__()
        dilate = 1
        if(dilationLayer):
            dilateMulti = 2
        else:
            dilateMulti = 1
        pad = int((kernelSize-1)/2)
        self.layer = layer
        templayerList = []
        for i in range(0, layer):
            tempLayer = denseBlockLayer_origin(inChannel+growthRate*i,growthRate,kernelSize,bottleneckMulti*growthRate, dilate, activ)
            dilate = dilate * dilateMulti
            templayerList.append(tempLayer)
        self.layerList = nn.ModuleList(templayerList)
            
    def forward(self,x):
        for i in range(0, self.layer):
            tempY = self.layerList[i](x)
            x = torch.cat((x, tempY), 1)
            
        return x
   

class transitionLayer(nn.Module):
    def __init__(self, inChannel = 64, compressionRate = 0.5, activ = 'ReLU'):
        super(transitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(inChannel)
        if(activ == 'LeakyReLU'):
            self.relu = nn.LeakyReLU()
        else:
            self.relu = nn.ReLU()
        self.conv = nn.Conv2d(inChannel,int(inChannel*compressionRate),1)
        
    def forward(self,x):
        x1 = self.bn(x)
        x2 = self.relu(x1)
        y = self.conv(x2) # 1*1 conv
        
        return y



class convLayer(nn.Module):
    def __init__(self, inChannel = 64, outChannel = 64, activ = 'ReLU', kernelSize = 3):
        super(convLayer, self).__init__()
        self.bn = nn.BatchNorm2d(inChannel)
        pad = int((kernelSize-1)/2)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(inChannel,outChannel,kernelSize,padding = pad)
        
    def forward(self,x):
        x1 = self.bn(x)
        x2 = self.relu(x1)
        y = self.conv(x2)
        return y
   



def fft_fastmri(input):
    # fft using fastmri api
    """
    input (N, 2, H, W)
    """

    input = input.permute(0, 2, 3, 1) #(N, H, W, 2)
    input = T.fft2(input) 
    # (N, W, H, 2) -> (N, 2, W, H)
    input = input.permute(0, 3, 1, 2)
    return input

def ifft_fastmri(input):
    input = input.permute(0, 2, 3, 1)
    input = T.ifft2(input) 

    # (N, D, W, H, 2) -> (N, 2, D, W, H)
    input = input.permute(0, 3, 1, 2)

    return input




class dataConsistencyLayer_fastmri(nn.Module):
    """
    one step data consistency layer 
    using fastmri fft api
    """
    def __init__(self, initLamda = 1, isStatic = True, isFastmri=True):
        super(dataConsistencyLayer_fastmri, self).__init__()
        self.normalized = True #norm == 'ortho'
        self.lamda = Parameter(torch.Tensor(1))
        self.lamda.data.uniform_(0, 1)
        self.isStatic = isStatic
        self.isFastmri = isFastmri
    
    def forward(self, xin, y, mask):
        if(self.isStatic):
            iScale = 1
        else:
            iScale = self.lamda/(1+self.lamda)

        if(len(xin.shape)==4):
            if(xin.shape[1]==1):
                emptyImag = torch.zeros_like(xin)
                xin_c = torch.cat([xin,emptyImag],1).permute(0,2,3,1).contiguous()
            else:
                xin_c = xin.permute(0,2,3,1).contiguous() #(bs,h,w,2)

        elif(len(xin.shape)==5):
            if(xin.shape[1]==1):
                emptyImag = torch.zeros_like(xin)
                xin_c = torch.cat([xin,emptyImag],1).permute(0,2,3,4,1).contiguous()
            else:
                xin_c = xin.permute(0,2,3,4,1).contiguous()

            mask = mask.reshape(mask.shape[0],mask.shape[1],mask.shape[2],mask.shape[3],1)
        else:
            assert False, "xin shape length has to be 4(2d) or 5(3d)"
       
        if self.isFastmri: # fastmri
            xin_f = T.fft2(xin_c,normalized=self.normalized)
            xGT_f = y
            xout_f = xin_f + (- xin_f + xGT_f) * iScale * mask
            xout = T.ifft2(xout_f) 

        else: # for cc359 or cardiac
            xin_f = T.fft(xin_c,2, normalized=self.normalized)
            xGT_f = y
            xout_f = xin_f + (- xin_f + xGT_f) * iScale * mask
            xout = T.ifft(xout_f,2, normalized=self.normalized)

        if(len(xin.shape)==4):
            xout = xout.permute(0,3,1,2)
        else:
            xout = xout.permute(0,4,1,2,3)
        if(xin.shape[1]==1):
            xout = torch.sqrt(xout[:,0:1]*xout[:,0:1]+xout[:,1:2]*xout[:,1:2])

        return xout.contiguous()



class dataConsistencyLayer_fastmri_m(nn.Module):
    """
    multi coil 
    one step data consistency layer 
    using fastmri fft api
    """
    def __init__(self, initLamda = 1, isStatic = True, isFastmri=True, isMulticoil=False):
        super(dataConsistencyLayer_fastmri_m, self).__init__()
        self.normalized = True #norm == 'ortho'
        self.lamda = Parameter(torch.Tensor(1))
        self.lamda.data.uniform_(0, 1)
        self.isStatic = isStatic
        self.isFastmri = isFastmri
        self.isMulticoil = isMulticoil 
    
    def forward(self, xin, y, mask):
        if(self.isStatic):
            iScale = 1
        else:
            iScale = self.lamda/(1+self.lamda)

        #=============================================
        # reshape
        #=============================================
        N = len(xin)
        if(len(xin.shape)==4): #(B,C,H,W)
            if(xin.shape[1]==1):
                emptyImag = torch.zeros_like(xin)
                xin_c = torch.cat([xin,emptyImag],1).permute(0,2,3,1).contiguous()
            else:
                if not self.isMulticoil:
                    xin = xin.permute(0,2,3,1).contiguous() #(bs,h,w,2)
        
        # dynamic MRI
        elif(len(xin.shape)==5): 
            assert True, "DC layer: not implement dynamic mri"
            if(xin.shape[1]==1):
                emptyImag = torch.zeros_like(xin)
                xin_c = torch.cat([xin,emptyImag],1).permute(0,2,3,4,1).contiguous()
            else:
                xin_c = xin.permute(0,2,3,4,1).contiguous()

            mask = mask.reshape(mask.shape[0],mask.shape[1],mask.shape[2],mask.shape[3],1)
        else:
            assert False, "xin shape length has to be 4(2d) or 5(3d)"
        
        #=============================================
        # singlecoil 
        #=============================================
        if not self.isMulticoil: 
            if self.isFastmri: # fastmri
                xin_f = T.fft2(xin,normalized=self.normalized, shift=True)
                xGT_f = y
                xout_f = xin_f + (- xin_f + xGT_f) * iScale * mask
                xout = T.ifft2(xout_f, shift=True) 

            else: # for cc359 or cardiac
                xin_f = T.fft(xin, 2, normalized=self.normalized)
                xGT_f = y
                xout_f = xin_f + (- xin_f + xGT_f) * iScale * mask
                xout = T.ifft(xout_f,2, normalized=self.normalized)

            if(len(xin.shape)==4):
                xout = xout.permute(0,3,1,2) #(B, 2, H, W)
            else:
                assert True, "DC layer: not implement dynamic mri"
                xout = xout.permute(0,4,1,2,3)
            if(xin.shape[1]==1):
                xout = torch.sqrt(xout[:,0:1]*xout[:,0:1]+xout[:,1:2]*xout[:,1:2])

        #=============================================
        # multicoil 
        #=============================================
        else: #xin_c (B,C,H,W)
            B, C, H, W = xin.shape
            xin_c = xin.reshape(N,-1,H,W,2) 
            if self.isFastmri:
                xin_f = T.fft2(xin_c,normalized=self.normalized, shift=True)
            else:
                xin_f = T.fft2(xin_c,normalized=self.normalized, shift=False)

            xGT_f = y
            xout_f = xin_f + (- xin_f + xGT_f) * iScale * mask
            if self.isFastmri:
                xout = T.ifft2(xout_f, normalized=self.normalized, shift=True) 
            else:
                xout = T.ifft2(xout_f, normalized=self.normalized, shift=False) 

            xout = xout.reshape(N,-1,H,W)

        return xout.contiguous()



class DC_multicoil(nn.Module):
    """
    data consistency layer for multicoil
    """
    def __init__(self, initLamda = 1, isStatic = True, shift=False):
        super(DC_multicoil, self).__init__()
        self.normalized = True #norm == 'ortho'
        self.lamda = Parameter(torch.Tensor(1))
        self.lamda.data.uniform_(0, 1)
        self.isStatic = isStatic
        self.shift = shift 
    
    def forward(self, xin, y, mask, sens_map):
        """
        xin: (B, 2, H, W)
        """
        assert xin.shape[1] == 2, "dc layer the last dimension of input x should be greater than 2"
        iScale = 1
        
        xin = T.expand_operator(xin.permute(0,2,3,1), sens_map, dim=1) #(B, coils, H, W, 2)
        xin_f = T.fft2(xin,normalized=self.normalized, shift=self.shift)
        xGT_f = y
        xout_f = xin_f + (- xin_f + xGT_f) * iScale * mask
        xout = T.ifft2(xout_f, normalized=self.normalized, shift=self.shift) 
        xout = T.reduce_operator(xout, sens_map, dim=1) #(B, H, W, 2)
        
        return xout.permute(0,3,1,2).contiguous()




class dataConsistencyLayer_static(nn.Module):
    def __init__(self, initLamda = 1, trick = 0, dynamic = False, conv = None, isFastmri=False):
        super(dataConsistencyLayer_static, self).__init__()
        self.normalized = True 
        self.trick = trick
        self.isFastmri = isFastmri
        tempConvList = []
        if(self.trick in [3,4]):
            if(conv is None):
                if(dynamic):
                    conv = nn.Conv3d(4,2,1,padding=0)
                else:
                    conv = nn.Conv2d(4,2,1,padding=0)
            tempConvList.append(conv)
        self.trickConvList = nn.ModuleList(tempConvList)

    def dc_operate(self, xin, y, mask):
        iScale = 1
        if(len(xin.shape)==4):
            if(xin.shape[1]==1):
                emptyImag = torch.zeros_like(xin)
                xin_c = torch.cat([xin,emptyImag],1).permute(0,2,3,1)
            else:
                xin_c = xin.permute(0,2,3,1) # (bs,h,w,c)
            mask = mask.reshape(mask.shape[0],mask.shape[1],mask.shape[2],1)
        
        elif(len(xin.shape)==5):
            if(xin.shape[1]==1):
                emptyImag = torch.zeros_like(xin)
                xin_c = torch.cat([xin,emptyImag],1).permute(0,2,3,4,1)
            else:
                xin_c = xin.permute(0,2,3,4,1)
            mask = mask.reshape(mask.shape[0],mask.shape[1],mask.shape[2],mask.shape[3],1)
        else:
            assert False, "xin shape length has to be 4(2d) or 5(3d)"
       
        # fastmri version
        if self.isFastmri:
            xin_f = T.fft2(xin_c,normalized=self.normalized)
            xGT_f = y
            xout_f = xin_f + (- xin_f + xGT_f) * iScale * mask
            xout = T.ifft2(xout_f) 
        else:
            xin_f = torch.fft(xin_c,2, normalized=self.normalized)
            xGT_f = y
            xout_f = xin_f + (- xin_f + xGT_f) * iScale * mask
            xout = torch.ifft(xout_f,2, normalized=self.normalized)

        if(len(xin.shape)==4):
            xout = xout.permute(0,3,1,2)
        else:
            xout = xout.permute(0,4,1,2,3)

        if(xin.shape[1]==1):
            xout = torch.sqrt(xout[:,0:1]*xout[:,0:1]+xout[:,1:2]*xout[:,1:2])
        
        return xout
    
    def forward(self, xin, y, mask):
        xt = xin
        if(self.trick == 1):
            xt = abs4complex(xt)
            xt = self.dc_operate(xt, y, mask)
        elif(self.trick == 2):
            xt = self.dc_operate(xt, y, mask)
            xt = abs4complex(xt)
            xt = self.dc_operate(xt, y, mask)
        elif(self.trick == 3):
            xdc1 = self.dc_operate(xt, y, mask)
            xt = abs4complex(xt)
            xdc2 = self.dc_operate(xt, y, mask)
            xdc = torch.cat([xdc1,xdc2],1)
            xt = self.trickConvList[0](xdc)
        elif(self.trick == 4):
            xdc1 = self.dc_operate(xt, y, mask)
            xabs = abs4complex(xdc1)
            xdc2 = self.dc_operate(xabs, y, mask)
            xdc = torch.cat([xdc1,xdc2],1)
            xt = self.trickConvList[0](xdc)

        elif(self.trick == 5): # 5 dc
            xt = self.dc_operate(xt, y, mask)
            xt = abs4complex(xt)
            xt = self.dc_operate(xt, y, mask)
            xt = abs4complex(xt)
            xt = self.dc_operate(xt, y, mask)
            xt = abs4complex(xt)
            xt = self.dc_operate(xt, y, mask)
            xt = abs4complex(xt)
            xt = self.dc_operate(xt, y, mask)
            xt = abs4complex(xt)

        else:
            xt = self.dc_operate(xt, y, mask)

        return xt
        

#===================================================
class dilatedConvBlock(nn.Module):
    def __init__(self, iConvNum = 3, inChannel=32):
        super(dilatedConvBlock, self).__init__()
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
 

class sensNet(nn.Module):
    def __init__(self, convNum = 3, recursiveTime = 3, inChannel = 24, midChannel=32):
        super(sensNet, self).__init__()
        self.rTime = recursiveTime
        self.LRelu = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(inChannel,midChannel,3,padding = 1)
        self.dilateBlock = dilatedConvBlock(convNum, midChannel)
        self.conv2 = nn.Conv2d(midChannel,inChannel,3,padding = 1)


    def complex_to_chan_dim(self, x):
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    def chan_complex_to_last_dim(self, x):
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()


    def forward(self, x1):

        x1 = self.complex_to_chan_dim(x1) 

        x2 = self.conv1(x1)
        x2 = self.LRelu(x2)
        xt = x2
        for i in range(self.rTime):
            x3 = self.dilateBlock(xt)
            xt = x3+x2
        x4 = self.conv2(xt)
        x4 = self.LRelu(x4)
        x5 = x4+x1
        
        x5 = self.chan_complex_to_last_dim(x5)
        return x5



class SensitivityModel(nn.Module):
    """
    Model for learning sensitivity estimation from k-space data.
    This model applies an IFFT to multichannel k-space data and then a U-Net
    to the coil images to estimate coil sensitivities. It can be used with the
    end-to-end variational network.
    """

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
        mask_center: bool = True,
        shift:bool = False,
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
            mask_center: Whether to mask center of k-space for sensitivity map
                calculation.
        """
        super(SensitivityModel, self).__init__()
        self.mask_center = mask_center
        self.shift = shift
        
        self.norm_unet = sensNet(convNum=3, recursiveTime=1, inChannel=2, midChannel=8) 

    def chans_to_batch_dim(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        b, c, h, w, comp = x.shape 

        return x.view(b * c, 1, h, w, comp), b

    def batch_chans_to_chan_dim(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        bc, _, h, w, comp = x.shape
        c = bc // batch_size

        return x.view(batch_size, c, h, w, comp)

    def divide_root_sum_of_squares(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, coils, H, W, 2)
        assert x.shape[-1] == 2, "the last dimension of input should be 2"
        tmp = T.root_sum_of_squares(T.root_sum_of_squares(x, dim=4), dim=1).unsqueeze(-1).unsqueeze(1)
        
        return x/tmp
        #return T.safe_divide(x, tmp).cuda()


    def get_pad_and_num_low_freqs(
        self, mask: torch.Tensor, num_low_frequencies: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if num_low_frequencies is None or num_low_frequencies == 0:
            # get low frequency line locations and mask them out
            # mask: [B, 1, H, W, 1]
            squeezed_mask = mask[:, 0, 0, :, 0].to(torch.int8)
            cent = squeezed_mask.shape[1] // 2
            # running argmin returns the first non-zero
            left = torch.argmin(squeezed_mask[:, :cent].flip(1), dim=1)
            right = torch.argmin(squeezed_mask[:, cent:], dim=1)
            num_low_frequencies_tensor = torch.max(
                2 * torch.min(left, right), torch.ones_like(left)
            )  # force a symmetric center unless 1
        else:
            num_low_frequencies_tensor = num_low_frequencies * torch.ones(
                mask.shape[0], dtype=mask.dtype, device=mask.device
            )

        pad = (mask.shape[-2] - num_low_frequencies_tensor + 1) // 2

        return pad, num_low_frequencies_tensor

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        num_low_frequencies: Optional[int] = None,
    ) -> torch.Tensor:

        if self.mask_center:
            pad, num_low_freqs = self.get_pad_and_num_low_freqs(
                mask, num_low_frequencies
            )
            masked_kspace = T.batched_mask_center(
                masked_kspace, pad, pad + num_low_freqs
            )

        # convert to image space
        # images: [96, 1, 218, 179, 2]
        # batches: 8
        images, batches = self.chans_to_batch_dim(T.ifft2(masked_kspace, shift=self.shift))
        
        # estimate sensitivities
        return self.divide_root_sum_of_squares(self.batch_chans_to_chan_dim(self.norm_unet(images), batches))







