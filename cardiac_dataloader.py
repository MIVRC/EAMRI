"""
data loader for cardiac dataset (DICOM version)
"""

import torch
import csv
from datetime import datetime as dt
import numpy as np
from PIL import Image
from tqdm import tqdm 
from torch.utils.data import Dataset
import scipy.io as sio
import random
import os
import h5py
import pdb

# path of the mask
fakeRandomPath = 'mask/mask_r30k_29.mat' # 30%
fakeRandomPath_15 = 'mask/mask_r10k_15.mat' # 15%
fakeRandomPath_10 = 'mask/mask_r4k_10.mat' # 10%
fakeRandomPath_5 = 'mask/mask_r4k_5.mat' # 5%
fakeRandomPath_FastMRI_25 = 'mask/mask320_r10k_25.mat' # 25% 


def subsampling_mask(srcImg,offset=0, mode = "default", mi = None):
    mask = np.zeros_like(srcImg)
    mask[mi==1,:] = 1 
    return mask

def kspace_subsampling(srcImg, mask):
    '''
    return subF has shape[...,2], without permute
    '''
    y = srcImg 
    if srcImg.shape[0] == 2:
        y = srcImg.permute(1,2,0) #(H, W, 2)
    
    mask = mask.reshape(mask.shape[0],mask.shape[1],1)
  
    xGT_f = torch.fft(y,2, normalized=True)
    subF = xGT_f * mask
    return subF


    
class SliceData_cardiac(Dataset):
    def __init__(self, root, samplingMode, isTrain=0):
        """
        isTrain = 1 or 0
        mode = type of input data
        samplingMode:str = sampling ratio of the mask
        staticRandom = true
        """
        
        if(isTrain == 1):
            iRange = range(1,31)
        else:
            iRange = range(31,34)

        if isinstance(samplingMode, list):
            samplingMode = samplingMode[0]
    
        if(samplingMode == 30): # 30%
            mDic = sio.loadmat(fakeRandomPath) 
            miList = mDic['RAll']
        elif(samplingMode == 15): # we use this
            mDic = sio.loadmat(fakeRandomPath_15)
            miList = mDic['RAll']
        elif(samplingMode == 10):
            mDic = sio.loadmat(fakeRandomPath_10)
            miList = mDic['RAll']
        elif(samplingMode == 5):
            mDic = sio.loadmat(fakeRandomPath_5)
            miList = mDic['RAll']
        else:
            raise NotImplementedError("Cardiac dataset: do not have samplingMode {}".format(samplingMode))
        
        self.zimList = []
        self.yList = []
        self.subfList = []
        self.mList = []
        self.meanList = []
        self.stdList = []
        self.maxList = []
        self.fnameList = []
        self.sliceList = []
        index = 0
        tList = range(1,21)
        
        for i in tqdm(iRange):
            for z in range(0,5): #0,4
                for t in tList: #1,20
                    filename = os.path.join(root, 'mr_heart_p%02dt%02dz%d.png' %(i, t, z))
                    im = Image.open(filename)
                    im_np = np.array(im).astype(np.float32)/255.
                  
                    randI = index
                    mi = miList[:,randI] # shape (256,)
                    mask = subsampling_mask(im_np, 0, 'fakeRandom', mi) # mask with selected row all 1
                   
                    m = np.fft.ifftshift(mask)  
                    m = torch.from_numpy(m)
                    im_tor = torch.from_numpy(im_np)

                    # subF
                    y = torch.zeros(2,256,256)
                    y[0] = im_tor
                    subF = kspace_subsampling(y,m) # subsampled kspace (256,256,2)

                    # zim 
                    zim = torch.ifft(subF,2, normalized=True) 
                    zim = zim.permute(2,0,1)
                    m = m.unsqueeze(-1)
                    
                    self.zimList.append(zim) # zim
                    self.yList.append(im_tor) # target
                    self.subfList.append(subF) #masked kspace
                    self.mList.append(m) # mask
                    self.meanList.append(torch.tensor([0]))
                    self.stdList.append(torch.tensor([1]))
                    self.maxList.append(1)
                    self.fnameList.append(filename)
                    self.sliceList.append(t)
                    index += 1


    def __getitem__(self, index):
        i = index

        zim = self.zimList[i]
        target = self.yList[i]
        subF = self.subfList[i]
        mask = self.mList[i]
        mean = self.meanList[i]
        std = self.stdList[i]
        maxv = self.maxList[i]
        fname = self.fnameList[i]
        slice = self.sliceList[i]
        
        return zim, target, subF, mask, mean, std, maxv, fname, slice

    def __len__(self):
        return len(self.yList)



