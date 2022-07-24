import os
import sys
from thop import profile
import torch
from network import getNet
import pdb
import warnings;warnings.filterwarnings("ignore")

model = getNet('eamri_')
#model = getNet('Unet_dc_fastmri')
#model = getNet('edge')

x = torch.randn(1,24,218,170)
y = torch.randn(1,218,170,2)
mask = torch.randn(1,218,170,1)
sens_map = torch.randn(1,12,218,170,2)
macs, params = profile(model, inputs=(x,y,mask,sens_map))

#macs, params = profile(model, inputs=(x,))

print("model flops {}G".format(macs/1e9))
print("model param {}K".format(params/1e3))

