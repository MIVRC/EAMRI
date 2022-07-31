import os
import sys
from thop import profile
import torch
from network import getNet
import pdb
import warnings;warnings.filterwarnings("ignore")

model = getNet('e2evarnet_var1')
#model = getNet('Unet_dc_fastmri')
#model = getNet('edge')

x = torch.randn(1,12,218,170, 2)
y = torch.randn(1,12, 218,170,2)
mask = torch.randn(1,1,218,170,1)
mask = mask.type(torch.uint8)
macs, params = profile(model, inputs=(x,y,mask))

#macs, params = profile(model, inputs=(x,))

print("model flops {}G".format(macs/1e9))
print("model param {}K".format(params/1e3))

