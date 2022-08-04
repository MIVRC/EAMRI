import torch
import torch.nn as nn
from fastmri.data import transforms_simple as T
import pdb


class dataConsistencyTerm(nn.Module):
    """
    DCB
    """
    def __init__(self, shift=False, noise_lvl=None):
        super(dataConsistencyTerm, self).__init__()
        self.noise_lvl = noise_lvl
        if noise_lvl is not None:
            self.noise_lvl = torch.nn.Parameter(torch.Tensor([noise_lvl]))
        self.shift = shift

    def perform(self, x, k0, mask, sensitivity):

        """
        k    - input in k-space
        k0   - initially sampled elements in k-space
        mask - corresponding nonzero location
        """
        #x = complex_multiply(x[...,0].unsqueeze(1), x[...,1].unsqueeze(1), sensitivity[...,0], sensitivity[...,1])

        x = T.expand_operator(x, sensitivity, dim=1)
        k = T.fft2(x, shift=self.shift)
        v = self.noise_lvl
        
        if v is not None: # noisy case
            # out = (1 - mask) * k + mask * (k + v * k0) / (1 + v)
            out = (1 - mask) * k + mask * (v * k + (1 - v) * k0) 
        else:  # noiseless case
            out = (1 - mask) * k + mask * k0
    
        # ### backward op ### #
        x = T.ifft2(out, shift=self.shift)
      
        '''
        Sx = T.complex_multiply(x[...,0], x[...,1], 
                                sensitivity[...,0], 
                               -sensitivity[...,1]).sum(dim=1)     
        '''
        Sx = T.reduce_operator(x, sensitivity, dim=1)

        '''
        SS = T.complex_multiply(sensitivity[...,0], 
                                sensitivity[...,1], 
                                sensitivity[...,0], 
                               -sensitivity[...,1]).sum(dim=1)
        '''
        SS = T.complex_mul(sensitivity, sensitivity).sum(dim=1)

        return Sx, SS

    
class weightedAverageTerm(nn.Module):

    def __init__(self, para=None):
        super(weightedAverageTerm, self).__init__()
        self.para = para
        if para is not None:
            self.para = torch.nn.Parameter(torch.Tensor([para]))

    def perform(self, cnn, Sx, SS):
        
        x = self.para*cnn + (1 - self.para)*Sx
        return x



class cnn_layer(nn.Module):
    
    def __init__(self, hiddim=72):
        super(cnn_layer, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(2, hiddim, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hiddim, hiddim, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hiddim, hiddim, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hiddim, hiddim, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hiddim, 2,  3, padding=1, bias=True)
        )     
        
    def forward(self, x):
        
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        return x
    
    
class vsnet(nn.Module):
    
    def __init__(self, alfa=0.1, beta=0.1, cascades=5, hiddim=72, shift=False, crop=False):
        super(vsnet, self).__init__()
        
        self.cascades = cascades 
        conv_blocks = []
        dc_blocks = []
        wa_blocks = []
        
        for i in range(cascades):
            conv_blocks.append(cnn_layer(hiddim=hiddim)) 
            dc_blocks.append(dataConsistencyTerm(shift=shift, noise_lvl=alfa)) 
            wa_blocks.append(weightedAverageTerm(para=beta)) 
        
        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.dc_blocks = nn.ModuleList(dc_blocks)
        self.wa_blocks = nn.ModuleList(wa_blocks)
        self.crop = crop
        

    def forward(self, x, k, m, c):
        """
        x: (B, coils, H, W, 2)
        """ 
        x = T.reduce_operator(x, c, dim=1)

        for i in range(self.cascades):
            x_cnn = self.conv_blocks[i](x)
            Sx, SS = self.dc_blocks[i].perform(x, k, m, c)
            x = self.wa_blocks[i].perform(x + x_cnn, Sx, SS)

        x = x.permute(0,3,1,2).contiguous()
        if self.crop: 
            x = T.center_crop(x, (320,320))

        return x
