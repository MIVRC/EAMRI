import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from .networkUtil import *
from .Unet import Unet_dc, unet_multicoil_cc359
from .unet_fastmri import UnetModel
from .RDN_complex import RDN_complex, RDN_multicoil
from .DC_CNN import DC_CNN, DC_CNN_multicoil
from .cascadeNetwork import CN_Dense
from .SEN_MRI import WAS 
from .zero_filled_model import ZF
from .md_recon import MRIReconstruction as mdr

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
#from .net_0707_var2 import net_0707_var2
from .net_0707_var3 import net_0707_var3

from .recurrentvarnet import RecurrentVarNet
from .eamri_0714 import eamri_0714
from .eamri_0714_debug import eamri_0714_debug
from .eamri_0715 import eamri_0715
from .eamri_0715_var1 import eamri_0715_var1
from .eamri_0715_var2 import eamri_0715_var2
from .eamri_0716 import eamri_0716
from .eamri_0716_var1 import eamri_0716_var1
from .eamri_0716_var2 import eamri_0716_var2
from .eamri_0718 import eamri_0718
from .eamri_0719 import eamri_0719
from .eamri_0719_var1 import eamri_0719_var1
from .eamri_0719_var2 import eamri_0719_var2
from .eamri_0719_debug import eamri_0719_debug
from .eamri_0719_debug3 import eamri_0719_debug3
from .eamri_0719_debug4 import eamri_0719_debug4
from .eamri_0721 import eamri_0721
from .eamri_0721_var1 import eamri_0721_var1
from .eamri_0721_var2 import eamri_0721_var2
from .eamri_0722 import eamri_0722
from .eamri_0722_var1 import eamri_0722_var1
from .eamri_0722_var2 import eamri_0722_var2
from .eamri_0722_var4 import eamri_0722_var4
from .eamri_0722_var3 import eamri_0722_var3

from .e2evarnet import VarNet
from .kikinet import KIKINet 
from .vsnet import vsnet



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
    elif(netType == 'DCCNN_fastmri_multicoil'):
        return DC_CNN_multicoil(indim=30, fNum=96, isFastmri=True, isMulticoil=True)
    elif(netType == 'DCCNN_cc359_multicoil'):
        return DC_CNN_multicoil(indim=24, fNum=96, isFastmri=False, isMulticoil=True)
   
    #===========RDN===============
    elif(netType == 'RDN_complex_DC'):
        return RDN_complex(dcLayer = 'DC', isFastmri=False)
    elif(netType == 'RDN_complex_DC_fastmri'):
        return RDN_complex(dcLayer = 'DC', isFastmri=True)
    elif(netType == 'RDN_cc359_multicoil'):
        return RDN_multicoil(xin1=24,midChannel=96,isFastmri=False)

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
    elif(netType == 'unet'): #cc359 multicoil
        return unet_multicoil_cc359(indim=2, shift=False, img_pad=True)
    elif(netType == 'unet_fastmri'): #cc359 multicoil
        return unet_multicoil_cc359(indim=2, shift=True, img_pad=False)

    elif(netType == 'Unet_dc'):
        return Unet_dc(isFastmri=False)
    elif(netType == 'Unet_dc_fastmri'):
        return Unet_dc(isFastmri=True)
    elif(netType == 'unet_24chans'): #dev
        return Unet_dc(indim=24, isFastmri=False, isMulticoil=True)
    elif(netType == 'Unet_dc_multicoil_fastmri'): #dev
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

    # =========================================================
    elif (netType == 'net_0707_var3'):
        return net_0707_var3(img_size=256, indim=2, edgeFeat=12, attdim=8, n_DAMs=[1,1,1,1], num_head=4, layers=[3,4,4,4], num_iters=[1,5,5,5], isFastmri=False)
    elif (netType == 'net_0707_var3_fastmri'):
        return net_0707_var3(img_size=320, indim=2, edgeFeat=12, attdim=8, n_DAMs=[1,1,1,1], num_head=4, layers=[3,4,4,4], num_iters=[1,5,5,5], isFastmri=True)




    # =========================================================
    elif (netType == 'eamri_0714_cc359'):
        return eamri_0714(indim=2, edgeFeat=12, attdim=48, n_DAMs=[1,1,1,1], growthRates=[16,16,16,16], num_head=4, layers=[3,5,5,5], num_iters=[1,1,1,1], fNums=[16,16,16,16], n_MSRB=1, isFastmri=False, isMulticoil=False)

    elif (netType == 'eamri_0714_cc359_multicoil'):
        return eamri_0714(indim=24, edgeFeat=32, attdim=48, n_DAMs=[1,1,1,1], growthRates=[32,32,32,32], num_head=4, layers=[3,5,5,5], num_iters=[1,5,5,5], fNums=[32,32,32,32], n_MSRB=2, isFastmri=False, isMulticoil=True)

    elif (netType == 'eamri_debug'):
        return eamri_0714_debug(indim=24, edgeFeat=32, attdim=48, n_DAMs=[1,1,1,1], growthRates=[32,32,32,32], num_head=4, layers=[3,5,5,5], num_iters=[1,5,5,5], fNums=[32,32,32,32], n_MSRB=2, isFastmri=False, isMulticoil=True)


    elif (netType == 'eamri_0715'):
        return eamri_0715(indim=24, edgeFeat=32, attdim=48, n_DAMs=[1,1,1,1], growthRates=[32,32,32,32], num_head=4, layers=[3,5,5,5], num_iters=[1,1,1,1], fNums=[32,32,32,32], n_MSRB=2, isFastmri=False, isMulticoil=True)

    elif (netType == 'eamri_0715_var1'):
        return eamri_0715_var1(indim=24, edgeFeat=32, attdim=48, n_DAMs=[1,1,1,1], growthRates=[32,32,32,32], num_head=4, layers=[3,5,5,5], num_iters=[1,1,1,1], fNums=[32,32,32,32], n_MSRB=2, isFastmri=False, isMulticoil=True)

    elif (netType == 'eamri_0715_var2'):
        return eamri_0715_var2(indim=24, edgeFeat=32, attdim=48, n_DAMs=[1,1,1,1], growthRates=[32,32,32,32], num_head=4, layers=[3,5,5,5], num_iters=[1,1,1,1], fNums=[32,32,32,32], n_MSRB=2, isFastmri=False, isMulticoil=True)

    elif (netType == 'eamri_0715_var4'):
        return eamri_0715_var1(indim=24, edgeFeat=32, attdim=48, n_DAMs=[1,3,3,3], growthRates=[32,32,32,32], num_head=4, layers=[3,5,5,5], num_iters=[1,5,5,5], fNums=[32,32,32,32], n_MSRB=1, isFastmri=False, isMulticoil=True)

    elif (netType == 'eamri_0716'):
        return eamri_0716(indim=24, edgeFeat=32, attdim=48, n_DAMs=[1,1,1,1], growthRates=[32,32,32,32], num_head=4, layers=[3,5,5,5], num_iters=[1,5,5,5], fNums=[32,32,32,32], n_MSRB=2, isFastmri=False, isMulticoil=True)

    elif (netType == 'eamri_0716_var1'):
        return eamri_0716_var1(indim=24, edgeFeat=32, attdim=48, n_DAMs=[1,1,1,1], growthRates=[32,32,32,32], num_head=4, layers=[3,5,5,5], num_iters=[1,5,5,5], fNums=[72,72,96,96], n_MSRB=2, isFastmri=False, isMulticoil=True)

    elif (netType == 'eamri_0716_var2'):
        return eamri_0716_var2(indim=24, edgeFeat=32, attdim=64, n_DAMs=[1,1,1,1,1], growthRates=[32,32,32,32,32], num_head=8, layers=[3,5,5,5,5], num_iters=[1,3,4,4,4], fNums=[48,72,72,96,96], n_MSRB=2, isFastmri=False, isMulticoil=True)

    elif (netType == 'eamri_0716_var3'):
        return eamri_0716_var1(indim=24, edgeFeat=48, attdim=48, n_DAMs=[1,1,1,1], growthRates=[32,32,32,32], num_head=4, layers=[3,5,5,5], num_iters=[3,5,5,5], fNums=[72,72,96,96], n_MSRB=2, isFastmri=False, isMulticoil=True)

    elif (netType == 'eamri_0716_var4'):
        return eamri_0716_var2(indim=24, edgeFeat=32, attdim=64, n_DAMs=[1,1,1,1,1], growthRates=[32,32,32,32,32], num_head=8, layers=[3,5,5,5,5], num_iters=[3,3,3,3,3], fNums=[96,96,96,96,96], n_MSRB=2, isFastmri=False, isMulticoil=True)
    
    elif (netType == 'eamri_0718'):
        return eamri_0718(indim=24, edgeFeat=32, attdim=32, n_DAMs=[1,1,1,1,1], growthRates=[32,32,32,32,32], num_head=4, layers=[3,5,5,5,5], num_iters=[1,3,4,4,4], fNums=[48,72,72,96,96], n_MSRB=2, isFastmri=False, isMulticoil=True)
    elif (netType == 'eamri_0718_var1'):
        return eamri_0718(indim=24, edgeFeat=32, attdim=32, n_DAMs=[1,1,1,1,1], growthRates=[32,32,32,32,32], num_head=4, layers=[3,5,5,5,5], num_iters=[1,5,5,5,5], fNums=[48,96,96,96,96], n_MSRB=2, isFastmri=False, isMulticoil=True)

    elif (netType == 'eamri_0719'):
        return eamri_0719(indim=24, edgeFeat=32, attdim=32, n_DAMs=[1,1,1,1,1], growthRates=[32,32,32,32,32], num_head=4, layers=[3,5,5,5,5], num_iters=[1,3,4,4,4], fNums=[48,72,72,96,96], n_MSRB=2, isFastmri=False, isMulticoil=True)

    elif (netType == 'eamri_0719_var1'):
        return eamri_0719_var1(indim=24, edgeFeat=32, attdim=32, n_DAMs=[1,1,1,1,1], growthRates=[32,32,32,32,32], num_head=4, layers=[3,5,5,5,5], num_iters=[1,5,5,5,4], fNums=[48,96,96,96,96], n_MSRB=2, isFastmri=False, isMulticoil=True)

    elif (netType == 'eamri_0719_var2'):
        return eamri_0719_var2(indim=24, edgeFeat=32, attdim=32, n_DAMs=[1,1,1,1,1], growthRates=[32,32,32,32,32], num_head=4, layers=[3,5,5,5,5], num_iters=[1,5,5,5,4], fNums=[48,96,96,96,96], n_MSRB=2, isFastmri=False, isMulticoil=True)

    elif (netType == 'eamri_0719_debug'):
        return eamri_0719_debug(indim=24, edgeFeat=32, attdim=32, n_DAMs=[1,1,1,1,1], growthRates=[32,32,32,32,32], num_head=4, layers=[3,5,5,5,5], num_iters=[1,3,4,4,4], fNums=[48,72,72,96,96], n_MSRB=2, isFastmri=False, isMulticoil=True)
   
    elif (netType == 'eamri_0719_debug2'):
        return eamri_0719_debug(indim=24, edgeFeat=32, attdim=32, n_DAMs=[1,1,1,1,1], growthRates=[32,32,32,32,32], num_head=4, layers=[3,5,5,5,5], num_iters=[1,3,4,4,4], fNums=[48,72,72,96,96], n_MSRB=2, isFastmri=False, isMulticoil=True)


    elif (netType == 'eamri_0719_debug3'):
        return eamri_0719_debug3(indim=24, edgeFeat=32, attdim=32, n_DAMs=[1,1,1,1,1], growthRates=[32,32,32,32,32], num_head=4, layers=[3,5,5,5,5], num_iters=[1,1,1,1,1], fNums=[48,72,72,96,96], n_MSRB=2, isFastmri=False, isMulticoil=True)

    elif (netType == 'eamri_0719_debug4'):
        return eamri_0719_debug4(indim=24, edgeFeat=32, attdim=32, n_DAMs=[1,1,1,1,1], growthRates=[32,32,32,32,32], num_head=4, layers=[3,5,5,5,5], num_iters=[1,3,4,4,4], fNums=[48,72,72,96,96], n_MSRB=2, isFastmri=False, isMulticoil=True)

    elif (netType == 'eamri_0719_debug5'):
        return eamri_0719_debug(indim=24, edgeFeat=32, attdim=32, n_DAMs=[1,1,1,1,1], growthRates=[32,32,32,32,32], num_head=4, layers=[3,5,5,5,5], num_iters=[1,1,1,1,1], fNums=[48,72,72,96,96], n_MSRB=2, isFastmri=False, isMulticoil=True)

    elif (netType == 'eamri_0721'):
        return eamri_0721(indim=24, edgeFeat=32, attdim=32, n_DAMs=[1,1,1,1,1], growthRates=[32,32,32,32,32], num_head=4, layers=[3,5,5,5,5], num_iters=[1,1,1,1,1], fNums=[48,72,72,96,96], n_MSRB=2, isFastmri=False, isMulticoil=True)
 
    elif (netType == 'eamri_0721_var1'):
        return eamri_0721_var1(indim=24, edgeFeat=32, attdim=32, n_DAMs=[1,1,1,1,1], growthRates=[32,32,32,32,32], num_head=4, layers=[3,5,5,5,5], num_iters=[1,1,1,1,1], fNums=[48,72,96,96,96], n_MSRB=3, isFastmri=False, isMulticoil=True)

    elif (netType == 'eamri_0721_var2'):
        return eamri_0721_var2(indim=2, edgeFeat=24, attdim=32, n_DAMs=[1,1,1,1,1], growthRates=[32,32,32,32,32], num_head=4, layers=[3,5,5,5,5], num_iters=[1,1,1,1,1], fNums=[48,72,96,96,96], n_MSRB=3, isFastmri=False, isMulticoil=True)

    # =========================================
    # final model for eamri
    elif (netType == 'eamri_0722'): # cc359
        return eamri_0722(indim=2, edgeFeat=24, attdim=32, num_head=4, num_iters=[1,3,3,3,3], fNums=[48,96,96,96,96], n_MSRB=3, shift=False)
    elif (netType == 'eamri_0722_fastmri'): # fastmri
        return eamri_0722(indim=2, edgeFeat=24, attdim=32, num_head=4, num_iters=[1,3,3,3,3], fNums=[48,96,96,96,96], n_MSRB=3, shift=True)
    elif (netType == 'eamri_0722_var1'): #ablation model without edge
        return eamri_0722_var1(indim=2, edgeFeat=24, attdim=32, num_head=4, num_iters=[1,3,3,3,3], fNums=[48,96,96,96,96], n_MSRB=3)
    elif (netType == 'eamri_0722_var2'): # cc359
        return eamri_0722_var2(indim=2, edgeFeat=24, attdim=32, num_head=4, num_iters=[1,3,3,3,3], fNums=[48,96,96,96,96], n_MSRB=3, shift=False)
    elif (netType == 'eamri_0722_var4'): # cc359
        return eamri_0722_var4(indim=2, edgeFeat=24, attdim=32, num_head=4, num_iters=[1,3,3,3,3], fNums=[48,96,96,96,96], n_MSRB=3, shift=False)
    elif (netType == 'eamri_0722_var5'): # cc359
        return eamri_0722(indim=2, edgeFeat=24, attdim=32, num_head=4, num_iters=[1,3,3,3,3], fNums=[48,96,96,96,96], n_MSRB=1, shift=False)
    elif (netType == 'eamri_0722_var6'): # cc359
        return eamri_0722(indim=2, edgeFeat=24, attdim=32, num_head=4, num_iters=[1,3,3,3,3], fNums=[48,96,96,96,96], n_MSRB=5, shift=False)

    elif (netType == 'eamri_0722_var3'):
        return eamri_0722_var3(indim=2, edgeFeat=24, attdim=32, num_head=4, num_iters=[1,3,3,3,3], fNums=[48,96,96,96,96], n_MSRB=3, shift=False)


    # =========================================================
    elif (netType == 'eamri_0722_sc'):
        return eamri_0722_sc(indim=2, edgeFeat=12, attdim=8, num_head=4, num_iters=[3,3,3,3,3], fNums=[32,32,32,32,32], nMSRB=1)


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
    # recurvarnet

    elif (netType == 'recurvarnet'):
        return RecurrentVarNet(in_channels= 2, num_steps= 3, recurrent_hidden_channels= 96, recurrent_num_layers= 4, no_parameter_sharing= True, learned_initializer= True, initializer_initialization= 'sense', initializer_channels= (32, 32, 64, 64), initializer_dilations=(1, 1, 2, 4), initializer_multiscale= 3, normalized= False, shift= False)

    elif (netType == 'recurvarnet_big'):
        return RecurrentVarNet(in_channels= 2, num_steps= 10, recurrent_hidden_channels= 64, recurrent_num_layers= 4, no_parameter_sharing= True, learned_initializer= True, initializer_initialization= 'sense', initializer_channels= (32, 32, 64, 64), initializer_dilations=(1, 1, 2, 4), initializer_multiscale= 3, normalized= False, shift= False)
    elif (netType == 'recurvarnet_var1'):
        return RecurrentVarNet(in_channels= 2, num_steps= 8, recurrent_hidden_channels= 48, recurrent_num_layers= 4, no_parameter_sharing= True, learned_initializer= True, initializer_initialization= 'sense', initializer_channels= (16, 16, 48, 48), initializer_dilations=(1, 1, 2, 4), initializer_multiscale= 3, normalized= False, shift= False)
    elif (netType == 'recurvarnet_fastmri'):
        return RecurrentVarNet(in_channels= 2, num_steps= 3, recurrent_hidden_channels= 96, recurrent_num_layers= 4, no_parameter_sharing= True, learned_initializer= True, initializer_initialization= 'sense', initializer_channels= (32, 32, 64, 64), initializer_dilations=(1, 1, 2, 4), initializer_multiscale= 3, normalized= False, shift= True)

    # =========================================================
    # e2evarnet

    elif (netType == 'e2evarnet'):
        return VarNet(num_cascades=5, sens_chans=4, sens_pools=4, chans=8, pools=4, mask_center=True, shift=False)
    elif (netType == 'e2evarnet_fastmri'):
        return VarNet(num_cascades=5, sens_chans=4, sens_pools=4, chans=8, pools=4, mask_center=True, shift=True)
    elif (netType == 'e2evarnet_big'):
        return VarNet(num_cascades=12, sens_chans=4, sens_pools=4, chans=18, pools=4, mask_center=True, shift=False)

    elif (netType == 'e2evarnet_var1'):
        return VarNet(num_cascades=8, sens_chans=4, sens_pools=4, chans=8, pools=4, mask_center=True, shift=False)


    # =========================================================
    elif (netType == 'kikinet'):
        return KIKINet(image_model_architecture= "MWCNN", kspace_model_architecture= "UNET", num_iter=5,image_mwcnn_hidden_channels = 16, image_mwcnn_num_scales = 4, image_mwcnn_bias = True, image_mwcnn_batchnorm = False, image_unet_num_filters = 8, image_unet_num_pool_layers = 4, image_unet_dropout_probability = 0.0, kspace_conv_hidden_channels = 16, kspace_conv_n_convs = 4, kspace_conv_batchnorm = False, kspace_didn_hidden_channels = 64, kspace_didn_num_dubs = 6, kspace_didn_num_convs_recon = 9, kspace_unet_num_filters = 8, kspace_unet_num_pool_layers = 4, kspace_unet_dropout_probability = 0.0, shift = False)

    # =========================================================
    elif (netType == 'vsnet'):
        return vsnet(alfa=0.1, beta=0.1, cascades=5, hiddim=96, shift=False)
    elif (netType == 'vsnet_fastmri'):
        return vsnet(alfa=0.1, beta=0.1, cascades=5, hiddim=96, shift=True, crop=True)
    elif (netType == 'vsnet_var1'):
        return vsnet(alfa=0.1, beta=0.1, cascades=8, hiddim=72, shift=False)
    else:
        assert False,"Wrong net type"


def getLoss(lossType):
    if(lossType == 'mse'):
        return torch.nn.MSELoss()
    elif(lossType == 'mae'):
        return torch.nn.L1Loss()
    else:
        assert False,"Wrong loss type"



