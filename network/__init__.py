import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from .networkUtil import *
from .EAMRI import EAMRI


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
# you can add your own network here
def getNet(netType):

    # EAMRI
    if (netType == 'EAMRI'): # cc359
        return EAMRI(indim=2, edgeFeat=24, attdim=32, num_head=4, num_iters=[1,3,3,3,3], fNums=[48,96,96,96,96], n_MSRB=3, shift=False)
    elif (netType == 'EAMRI_fastmri'): # fastmri
        return EAMRI(indim=2, edgeFeat=24, attdim=32, num_head=4, num_iters=[1,3,3,3,3], fNums=[48,96,96,96,96], n_MSRB=3, shift=True)

    else:
        assert False,"Wrong net type"


def getLoss(lossType):
    if(lossType == 'mse'):
        return torch.nn.MSELoss()
    elif(lossType == 'mae'):
        return torch.nn.L1Loss()
    else:
        assert False,"Wrong loss type"



