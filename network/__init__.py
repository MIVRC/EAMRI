import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from .networkUtil import *
from .CNN import Unet_dc
from .unet_fastmri import UnetModel
from .RDN_complex import RDN_complex
from .DC_CNN import DC_CNN
from .cascadeNetwork import CN_Dense
from .SEN_MRI import WAS 
from .zero_filled_model import ZF
from .md_recon import MRIReconstruction as mdr

#from .cascaded_edgeNet import casEdgeNet, net_0426 
#from .Recurrent_Transformer import ReconFormer
#from .casFormer import casFormer
#from .edgeFormer_0413 import edgeFormer_0413 

#from .convTranNet_0523 import convTranNet_0523, convTranNet_0523_baseline, convTranNet_0523_var1, convTranNet_0523_var2, convTranNet_0523_var3, convTranNet_0523_var4
#from .convTranNet_0529 import convTranNet_0529, convTranNet_0529_var1, convTranNet_0529_var2, convTranNet_0529_debug, convTranNet_0529_var3

#from .convTranNet_0531 import convTranNet_0531, convTranNet_0531_var1, convTranNet_0531_var2, convTranNet_0531_debug

#from .convTranNet_0601 import convTranNet_0601, convTranNet_0601_debug, convTranNet_0601_var1, convTranNet_0601_debug2

from .net_0621 import net_0621
#from .net_0621_var1 import net_0621_var1
#from .net_0621_var2 import net_0621_var2
#from .net_0621_var3 import net_0621_var3
#from .net_0621_var4 import net_0621_var4

from .net_0622 import net_0622
from .net_0622_var1 import net_0622_var1
from .net_0626 import net_0626

from .net_0702 import net_0702
from .net_0702_var1 import net_0702_var1
from .net_0702_var2 import net_0702_var2
from .net_0702_var3 import net_0702_var3 
from .net_0702_var4 import net_0702_var4
from .net_0702_var5 import net_0702_var5
from .net_0706 import net_0706, Edge_Net
from .net_0705 import net_0705
from .net_0705_var2 import net_0705_var2
from .net_0705_var3 import net_0705_var3

from .net_0707 import net_0707
from .net_0707_var1 import net_0707_var1
from .net_0707_var2 import net_0707_var2


def getScheduler(optimizer, config):

    schedulerType = config['train']['scheduler']
    if(schedulerType == 'lr'):
        scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=int(config['train']['lr_scheduler_stepsize']))
    else:
        assert False,"Wrong scheduler type"
        
    return scheduler 


def getOptimizer(param, optimizerType, LR, weightDecay = 0):

    if(optimizerType == 'RMSprop'):
        optimizer = torch.optim.RMSprop(param, lr=LR)
    elif(optimizerType == 'Adam'):
        optimizer = torch.optim.Adam(param, lr=LR, weight_decay = 1e-7) #weight decay for DC_CNN
    elif(optimizerType == 'SGD'):
        optimizer = torch.optim.SGD(param, lr=LR, momentum = 0.9) #sgd + momentum
    else:
        assert False,"Wrong optimizer type"
    return optimizer

# main function to getNet
def getNet(netType):

    #===========DC_CNN============
    if(netType == 'DCCNN'):
        return DC_CNN(isFastmri=False)
    elif(netType == 'DCCNN_fastmri'):
        return DC_CNN(isFastmri=True)

    elif(netType == 'DCCNN_f16'):
        return DC_CNN(fNum=16)
    elif(netType == 'DCCNN_f64'):
        return DC_CNN(fNum=64)
    
    #===========RDN===============
    elif(netType == 'RDN_complex_DC'):
        return RDN_complex(dcLayer = 'DC', isFastmri=False)
    elif(netType == 'RDN_complex_DC_fastmri'):
        return RDN_complex(dcLayer = 'DC', isFastmri=True)

    #===========cascade===========
    elif(netType == 'cddntdc'): 
        return CN_Dense(2,c=5,dilate=True, useOri = True, transition=0.5, trick = 2, isFastmri=False)
    elif(netType == 'cddntdc_fastmri'): 
        return CN_Dense(2,c=5,dilate=True, useOri = True, transition=0.5, trick = 2, isFastmri=True)

    #===========SEN-MRI===========
    elif(netType == 'wasnet'):
        return WAS(inChannel = 2, wChannel = 12, fmChannel = 12, skChannel = 12, c=3, nmodule=3, M=2, r=4, L= 16, kernel_size = 3, n_resblocks=5, isFastmri=False) 
    elif(netType == 'wasnet_fastmri'):
        return WAS(inChannel = 2, wChannel = 12, fmChannel = 12, skChannel = 12, c=3, nmodule=3, M=2, r=4, L= 16, kernel_size = 3, n_resblocks=5, isFastmri=True) 
    elif(netType == 'wasnet_big_fastmri'):
        return WAS(inChannel = 2, wChannel = 12, fmChannel = 12, skChannel = 12, c=3, nmodule=5, M=2, r=4, L= 16, kernel_size = 3, n_resblocks=5, isFastmri=True) 
    elif(netType == 'wasnet_big'):
        return WAS(inChannel = 2, wChannel = 12, fmChannel = 12, skChannel = 12, c=3, nmodule=5, M=2, r=4, L= 16, kernel_size = 3, n_resblocks=5, isFastmri=False) 
    
    elif(netType == 'wasnet_fastmri_multicoil'):
        return WAS(inChannel = 30, wChannel = 48, fmChannel = 12, skChannel = 12, c=3, nmodule=3, M=2, r=4, L= 16, kernel_size = 3, n_resblocks=5, isFastmri=True, isMulticoil=True) 

    #===========Unet============
    elif(netType == 'vanillaCNN'):
        return vanillaCNN()
    elif(netType == 'Unet_dc'):
        return Unet_dc(isFastmri=False)
    elif(netType == 'Unet_dc_fastmri'):
        return Unet_dc(isFastmri=True)
    elif(netType == 'Unet_fastmri_real'):
        return UnetModel(in_chans=1, out_chans=1, chans=32, num_pool_layers=4, drop_prob=0.0)

    elif(netType == 'Unet_dc_multicoil'):
        return Unet_dc(indim=30, isFastmri=True, isMulticoil=True)

    #===========mdr============
    elif (netType == 'mdr'):
        return mdr(isFastmri=False)
    elif (netType == 'mdr_fastmri'):
        return mdr(isFastmri=True)

    #===========cascaded_edgeNet============
    elif (netType == 'casEdgeNet'):
        return casEdgeNet(2,32)
    elif (netType == 'net_0426'):
        return net_0426(2,64)

    #===========Recon============
    elif (netType == 'recon'):
        return ReconFormer(in_channels=2, out_channels=2, num_ch=(96,48,24), num_iter=5, down_scales=(2,1,1.5), img_size=320, num_heads=(6,6,6), depths=(2,1,1), window_sizes=(8,8,8), mlp_ratio=2., resi_connection='1conv', use_checkpoint=(False,False, True, True, False, False))
    elif (netType == 'casFormer'):
        return casFormer(in_channels=2, out_channels=2, num_ch=(96,48,48,24,24), down_scales=(2,1,1,1,1.5), img_size=320, num_heads=(6,6,6,6,6), depths=(2,2,1,1,1), window_sizes=(8,8,8,8,8), mlp_ratio=2., resi_connection='1conv', use_checkpoint=(False,False, True, True, False, False))
    
    elif (netType == 'edgeFormer'):
        return edgeFormer_0413(in_channels=2, out_channels=2, num_ch=(36,48,48,96), down_scales=(2,1,1,1.5), img_size=320, num_heads=(6,6,6,6), depths=(3,3,9,3), window_sizes=(8,8,8,8), mlp_ratio=2., resi_connection='1conv', use_checkpoint=(False,False, False, False))


    #===========convTransNet============
    elif (netType == 'convTranNet_0523_fastmri'):
        return convTranNet_0523(img_size=320, C=3, G0=2, G1=16, n_RDB=4, nf=36, num_head=6, depth=6, window_size=8, isFastmri=True)

    elif (netType == 'convTranNet_0523'):
        return convTranNet_0523(img_size=256, C=6, G0=2, G1=32, n_RDB=6, nf=36, num_head=3, depth=2, window_size=8, isFastmri=False)


    elif (netType == 'convTranNet_0523_var1'):
        return convTranNet_0523_var1(img_size=256, C=6, G0=2, G1=32, n_RDB=6, nf=36, num_head=3, depth=2, window_size=8, isFastmri=False)


    elif (netType == 'convTranNet_0523_var2'):
        return convTranNet_0523_var2(img_size=256, C=6, G0=2, G1=32, n_RDB=6, nf=36, num_head=3, depth=2, window_size=8, isFastmri=False)


    elif (netType == 'convTranNet_0523_var3'):
        return convTranNet_0523_var3(img_size=256, C=6, G0=2, G1=32, n_RDB=6, nf=36, num_head=3, depth=2, window_size=8, n_DAM=6, isFastmri=False)

    elif (netType == 'convTranNet_0523_var4'):
        return convTranNet_0523_var4(img_size=256, C=6, G0=2, G1=32, n_RDB=6, nf=36, num_head=3, depth=2, window_size=8, n_DAM=3, isFastmri=False)


    elif (netType == 'convTranNet_0523_baseline'):
        return convTranNet_0523_baseline(img_size=256, C=6, G0=2, G1=32, n_RDB=6, nf=36, num_head=3, depth=2, window_size=8, isFastmri=False)

    elif (netType == 'convTranNet_0529'):
        return convTranNet_0529(img_size=256, indim=2, outdim=12, num_head=4, n_DAM=3, isFastmri=False)


    elif (netType == 'convTranNet_0529_debug'):
        return convTranNet_0529_debug(img_size=256, indim=2, outdim=12, num_head=4, n_DAM=3, isFastmri=False)

    elif (netType == 'convTranNet_0529_ex2'):
        return convTranNet_0529(img_size=256, indim=2, outdim=64, num_head=8, n_DAM=3, isFastmri=False)


    elif (netType == 'convTranNet_0529_var1'):
        return convTranNet_0529_var1(img_size=256, indim=2, fNum=16, outdim=32, num_head=4, isFastmri=False)

    elif (netType == 'convTranNet_0529_var2'):
        return convTranNet_0529_var2(img_size=256, indim=2, outdim=12, num_head=4, n_DAM=3, isFastmri=False)

    elif (netType == 'convTranNet_0529_var3'):
        return convTranNet_0529_var3(img_size=256, indim=2, outdim=64, num_head=8, n_DAM=3, isFastmri=False)

    elif (netType == 'convTranNet_0531'):
        return convTranNet_0531(img_size=256, indim=2, outdim=12, num_head=4, n_DAM=3, isFastmri=False)

    elif (netType == 'convTranNet_0531_debug'):
        return convTranNet_0531_debug(img_size=256, indim=2, outdim=12, num_head=4, n_DAM=3, isFastmri=False)


    elif (netType == 'convTranNet_0531_ex2'):
        return convTranNet_0531(img_size=256, indim=2, outdim=12, num_head=1, n_DAM=3, isFastmri=False)


    elif (netType == 'convTranNet_0531_var1'):
        return convTranNet_0531_var1(img_size=256, indim=2, outdim=12, num_head=4, n_DAM=3, isFastmri=False)

    elif (netType == 'convTranNet_0531_var2'):
        return convTranNet_0531_var2(img_size=256, indim=2, outdim=12, num_head=4, n_DAM=3, isFastmri=False)


    # =========================================================
    elif (netType == 'convTranNet_0601'):
        return convTranNet_0601(img_size=256, indim=2, outdim=12, num_head=4, n_DAM=3, isFastmri=False)

    elif (netType == 'convTranNet_0601_fastmri'):
        return convTranNet_0601(img_size=320, indim=2, outdim=12, num_head=4, n_DAM=3, isFastmri=True)


    elif (netType == 'net_0601_var1'):
        return net_0601_var1(img_size=256, indim=2, outdim=12, num_head=4, n_DAM=3, isFastmri=False)

    elif (netType == 'convTranNet_0601_debug'):
        return convTranNet_0601_debug(img_size=256, indim=2, outdim=12, num_head=4, n_DAM=3, isFastmri=False)


    elif (netType == 'convTranNet_0601_debug2'):
        return convTranNet_0601_debug2(img_size=256, indim=2, outdim=12, num_head=4, n_DAM=3, isFastmri=False)

    
    # =========================================================
    elif (netType == 'net_0617'):
        return net_0617(img_size=256, indim=2, edgeFeat=16, outdim=32, num_head=4, n_DAM=3, isFastmri=False)

    elif (netType == 'net_0617_var1'):
        return net_0617_var1(img_size=256, indim=2, edgeFeat=32, outdim=32, num_head=4, n_DAM=3, isFastmri=False)

    elif (netType == 'net_0617_var2'):
        return net_0617_var2(img_size=256, indim=2, edgeFeat=32, outdim=32, num_head=4, n_DAM=3, isFastmri=False)


    # =========================================================
    elif (netType == 'net_0621'):
        return net_0621(img_size=256, indim=2, edgeFeat=32, outdim=32, num_head=4, n_DAM=3, isFastmri=False)

    elif (netType == 'net_0621_var1'):
        return net_0621_var1(img_size=256, indim=2, edgeFeat=32, outdim=32, num_head=4, n_DAM=3, isFastmri=False)

    elif (netType == 'net_0621_var2'):
        return net_0621_var2(img_size=256, indim=2, edgeFeat=32, outdim=32, num_head=4, n_DAM=3, isFastmri=False)

    elif (netType == 'net_0621_var3'):
        return net_0621_var3(img_size=256, indim=2, edgeFeat=32, outdim=32, num_head=4, n_DAM=3, isFastmri=False)

    elif (netType == 'net_0621_var4'):
        return net_0621_var4(img_size=256, indim=2, edgeFeat=32, outdim=32, num_head=4, n_DAM=3, isFastmri=False)


    # =========================================================
    elif (netType == 'net_0622'):
        return net_0622(img_size=256, indim=2, edgeFeat=32, outdim=32, num_head=4, n_DAM=1, isFastmri=False)

    elif (netType == 'net_0622_var1'):
        return net_0622_var1(img_size=256, indim=2, edgeFeat=32, outdim=32, num_head=4, n_DAM=1, isFastmri=False)

    # =========================================================
    elif (netType == 'net_0626'):
        return net_0626(img_size=256, indim=2, edgeFeat=32, outdim=32, num_head=4, n_DAM=3, isFastmri=False)


    # =========================================================
    elif (netType == 'net_0702'):
        return net_0702(img_size=256, indim=2, edgeFeat=32, outdim=32, num_head=4, n_DAM=3, isFastmri=False)
    elif (netType == 'net_0702_var1'):
        return net_0702_var1(img_size=256, indim=2, edgeFeat=32, outdim=32, num_head=4, n_DAM=3, isFastmri=False)
    elif (netType == 'net_0702_var2'):
        return net_0702_var2(img_size=256, indim=2, edgeFeat=32, outdim=32, num_head=4, n_DAM=3, isFastmri=False)
    elif (netType == 'net_0702_var3'):
        return net_0702_var3(img_size=256, indim=2, edgeFeat=32, outdim=32, num_head=4, n_DAM=3, isFastmri=False)
    elif (netType == 'net_0702_var4'):
        return net_0702_var4(img_size=256, indim=2, edgeFeat=32, outdim=32, num_head=4, n_DAM=3, isFastmri=False)
    elif (netType == 'net_0702_var5'):
        return net_0702_var5(img_size=256, indim=2, edgeFeat=32, outdim=32, num_head=4, n_DAM=3, isFastmri=False)
   
    #==================
    # test
    elif (netType == 'net_0706'):
        return net_0706(img_size=256, indim=2, edgeFeat=12, attdim=8, n_DAMs=[1,1,3,1], num_head=4, layers=[4,4,4,4], isFastmri=False)
    elif (netType == 'net_0706_var1'):
        return net_0706(img_size=256, indim=2, edgeFeat=12, attdim=8, n_DAMs=[2,2,6,2], num_head=4, layers=[4,4,4,4], isFastmri=False)
    elif (netType == 'net_0706_var2'):
        return net_0706(img_size=256, indim=2, edgeFeat=12, attdim=8, n_DAMs=[2,2,6,2], num_head=4, layers=[3,3,4,3], isFastmri=False)

    # =========================================================
    elif (netType == 'net_0707'):
        return net_0707(img_size=256, indim=2, edgeFeat=12, attdim=8, n_DAMs=[1,1,3,1], num_head=4, layers=[3,3,4,3], num_iters=[3,3,1,3], isFastmri=False)
    elif (netType == 'net_0707_var1'):
        return net_0707_var1(img_size=256, indim=2, edgeFeat=12, attdim=8, n_DAMs=[1,1,1,1], num_head=4, layers=[3,4,4,4], num_iters=[1,5,5,5], isFastmri=False)
    elif (netType == 'net_0707_var2'):
        return net_0707_var2(img_size=256, indim=2, edgeFeat=12, attdim=16, n_DAMs=[1,5,5,5], num_head=4, fNums=[64,64,64,64], num_iters=[1,1,1,1], isFastmri=False)


    elif (netType == 'edge'):
        return Edge_Net(indim=2, hiddim=8, n_MSRB=2)

    # =========================================================
    elif (netType == 'net_0705'):
        return net_0705(img_size=256, indim=2, convDim=16, edgeFeat=8, attdim=32, growthRate=8, DAM_denseLayer=5, num_head=4, n_MSRB=2, n_DAM=3, isFastmri=False)
    elif (netType == 'net_0705_var1'):
        return net_0705(img_size=256, indim=2, convDim=16, edgeFeat=8, attdim=32, growthRate=8, DAM_denseLayer=3, num_head=4, n_MSRB=1, n_DAM=3, isFastmri=False)
    elif (netType == 'net_0705_var3'):
        return net_0705_var3(img_size=256, indim=2, convDim=16, edgeFeat=8, attdim=32, growthRate=16, DAM_denseLayer=3, num_head=4, n_MSRB=1, n_DAM=3, isFastmri=False)
    elif (netType == 'net_0705_var2'):
        return net_0705_var2(img_size=256, indim=2, convDim=16, expand=2, edgeFeat=8, attdim=32, DAM_denseLayer=1, num_head=4, n_MSRB=2, n_DAM=1, isFastmri=False)


    # =========================================================
    else:
        assert False,"Wrong net type"


def getLoss(lossType):
    if(lossType == 'mse'):
        return torch.nn.MSELoss()
    elif(lossType == 'mae'):
        return torch.nn.L1Loss()
    else:
        assert False,"Wrong loss type"



