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

from .recurrentvarnet import RecurrentVarNet
from .EAMRI import EAMRI
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

    # =========================================
    # final model for eamri
    # eamri
    elif (netType == 'EAMRI'): # cc359
        return EAMRI(indim=2, edgeFeat=24, attdim=32, num_head=4, num_iters=[1,3,3,3,3], fNums=[48,96,96,96,96], n_MSRB=3, shift=False)
    elif (netType == 'EAMRI_fastmri'): # fastmri
        return EAMRI(indim=2, edgeFeat=24, attdim=32, num_head=4, num_iters=[1,3,3,3,3], fNums=[48,96,96,96,96], n_MSRB=3, shift=True)
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

    elif (netType == 'eamri_0722_sc'):
        return eamri_0722_sc(indim=2, edgeFeat=12, attdim=8, num_head=4, num_iters=[3,3,3,3,3], fNums=[32,32,32,32,32], nMSRB=1)

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

    elif (netType == 'e2evarnet_var1_fastmri'):
        return VarNet(num_cascades=8, sens_chans=4, sens_pools=4, chans=8, pools=4, mask_center=True, shift=True)

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



