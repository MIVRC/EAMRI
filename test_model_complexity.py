import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
from thop import profile
import torch
import time
from network import getNet
import pdb
import warnings;warnings.filterwarnings("ignore")

#model = getNet('recurvarnet')
#model = getNet('eamri_0722')
#model = getNet('unet')
#model = getNet('vsnet')

model = getNet('eamri_0722_n1')
#model = torch.nn.DataParallel(model)
#model = getNet('edge')

x = torch.randn(1,12,218,170, 2)
y = torch.randn(1,12,218,170,2)
sens_map = torch.randn(1,12,218,170,2)
mask = torch.randn(1,1,218,170,1)
mask = mask.type(torch.uint8)
macs, params = profile(model, inputs=(x,y,mask))

#macs, params = profile(model, inputs=(x,))

print("model flops {}G".format(macs/1e9))
print("model param {}K".format(params/1e3))

'''
t1 = time.time()
for _ in range(10):
    #outs = model(x, y,mask, sens_map)
    outs = model(x, y,mask)

t2 = time.time()
print("elasped time for 10 iters {}".format((t2-t1)/10))
'''
