import os
import sys
from thop import profile
import torch
from network import getNet
import pdb
import warnings;warnings.filterwarnings("ignore")

model = getNet('net_0707_var2')
#model = getNet('Unet_dc_fastmri')
#model = getNet('edge')

x = torch.randn(1,2,256, 256)
y = torch.randn(1,256,256,2)
mask = torch.randn(1,256,256,1)
macs, params = profile(model, inputs=(x,None,y,mask))

#macs, params = profile(model, inputs=(x,))

print("model flops {}G".format(macs/1e9))
print("model param {}K".format(params/1e3))

