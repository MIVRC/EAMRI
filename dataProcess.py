# main file for data process
# support loading three type of dataset
# only support Cartesian masking now 

import csv
import cv2
from datetime import datetime as dt
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
import torch.utils.data as data
import scipy.io as sio
import random
import os
import h5py
import pdb
from util.imageUtil import *
# fastmri api
#from fastmri.fftc import ifft2c_old as ifft2c
#from fastmri.data import transforms_simple as T
#from fastmri.data.transforms_simple import to_tensor, apply_mask, complex_center_crop, center_crop, normalize
#from fastmri.data.subsample import RandomMaskFunc as RM

# data type in the mask
# dictionary with '__header__', '__version__', '__globals', RALL
# RALL: np.array (320, 10000) for fastmri or (256,30000) for other dataset

# path of the mask
fakeRandomPath = 'mask/mask_r30k_29.mat' # 30%
fakeRandomPath_15 = 'mask/mask_r10k_15.mat' # 15%
fakeRandomPath_10 = 'mask/mask_r4k_10.mat' # 10%
fakeRandomPath_5 = 'mask/mask_r4k_5.mat' # 5%
fakeRandomPath_FastMRI_25 = 'mask/mask320_r10k_25.mat' # 25% 

# path of the original data
pathFastMRI_train = '/home/ET/hanhui/opendata/fastmri_knee_singlecoil_dataset/singlecoil_train/'
pathFastMRI_eval = '/home/ET/hanhui/opendata/fastmri_knee_singlecoil_dataset/singlecoil_val/'
pathBrainMRI_train = '/home/ET/hanhui/opendata/Brain_MRI/myTrain/'
pathBrainMRI_eval = '/home/ET/hanhui/opendata/Brain_MRI/Val/'


def generateDatasetName(configs):
    # generate dataset name
    datasetName = ""
    for index in range(len(configs)):
        if(configs[index]==""):
            continue
        datasetName += str(configs[index])
        datasetName += '_'

    return datasetName[:-1]


def getDataloader(dataType, isTrain = 1, batchSize = 1, seed = 0, num_workers = 0):
    """
    generate pytorch dataloader
    """

    #-----------------------
    # determine dataset type
    dataType = dataType.lower()
    if('fastmri' in dataType):
        dataset = "FastMRI"
    elif('brain' in dataType):
        dataset = 'BrainMRI'
    elif('cardiac' in dataType):
        dataset = "cardiac"
    else:
        raise NotImplementedError("Unkown dataset. Only support fastmri/brain/cardiac")

    #-----------------------
    # use partial or full data
    if('reduce' in dataType):
        reduceMode = "reduce"
    else:
        reduceMode = "full"

    #-----------------------
    # type of data: complex or real
    if('complex' in dataType):
        dataMode = 'complex'
    else:
        dataMode = 'abs'

    #-----------------------
    # type of sampling
    if('static' in dataType):  
        staticSampling = True
        staticPrefix = "static"
    else:
        staticSampling = False
        staticPrefix = ""

    #-----------------------
    # ratio of the mask
    #----------
    # the first four type is Cartesian masking
    if('random' in dataType) or ('rand30' in dataType): # rand30(cardiac) or rand25(fastmri)
        samplingMode = 'random'
    elif(('r15' in dataType) or ('rand15' in dataType)):
        samplingMode = 'rand15'
    elif(('r10' in dataType) or ('rand10' in dataType)):
        samplingMode = 'rand10'
    elif(('r5' in dataType) or ('rand5' in dataType)):
        samplingMode = 'rand5'
    #----------
    elif('fakeRandom' in dataType):
        samplingMode = 'fakeRandom'
    elif('nolattice' in dataType):
        samplingMode = 'nolattice'
    elif('lattice8' in dataType):
        samplingMode = 'lattice8'
    else:
        samplingMode = 'lattice'

    #-----------------------
    datasetName = generateDatasetName([dataset,dataMode,staticPrefix+samplingMode,isTrain,reduceMode])
    print('#Generating dataset:'+datasetName)

    #=============================
    # generate pytorch dataset
    if(dataset == 'cardiac'):  
        dataset = genCardiac(isTrain, dataMode, samplingMode, reduceMode, staticSampling)
        isdrop = False
    elif(dataset == 'FastMRI'):
        dataset = genFastMRI(isTrain, dataMode, samplingMode, reduceMode, staticSampling)  # dev
        #dataset = genFastMRI_fastmri(isTrain, reduceMode, 0.1, 4, 1)  # dev
        isdrop = False
    elif(dataset == 'BrainMRI'):
        dataset = genBrainMRI(isTrain, dataMode, samplingMode, reduceMode, staticSampling) 
        isdrop = True
    else:
        assert False,"wrong dataset type"
        
    def _init_func(work_id):
        np.random.seed(work_id+seed)
  
    # generate dataloader
    # if train, then shuffle data, if test, not shuffle data

    print("start loading data") 
    data_loader = data.DataLoader(dataset, batch_size=batchSize, shuffle=isTrain, num_workers=num_workers, pin_memory=True, worker_init_fn=_init_func, drop_last=isdrop) # True for fastmri and False for cardiac
    datasize = len(dataset) # 3000
    return data_loader,datasize

    
class genCardiac(data.Dataset):
    def __init__(self, isTrain = 1, mode = 'abs', samplingMode = 'default', reduceMode = "", staticRandom = False):
        """
        isTrain = 1 or 0
        mode = type of input data
        samplingMode:str = sampling ratio of the mask
        reduceMode: str, use partial or full data
        staticRandom = true
        """
        
        if(isTrain == 1):
            iRange = range(1,31)
        else:
            iRange = range(31,34)

        if(samplingMode == 'random'): # 30%
            mDic = sio.loadmat(fakeRandomPath) 
            miList = mDic['RAll']
        elif(samplingMode == 'rand15'): # we use this
            mDic = sio.loadmat(fakeRandomPath_15)
            miList = mDic['RAll']
        elif(samplingMode == 'rand10'):
            mDic = sio.loadmat(fakeRandomPath_10)
            miList = mDic['RAll']
        elif(samplingMode == 'rand5'):
            mDic = sio.loadmat(fakeRandomPath_5)
            miList = mDic['RAll']

        self.yList = []
        self.mList = []
        self.meanList = []
        self.stdList = []
        self.mode = mode
        offset = 0
        index = 0

        assert mode in ['complex','abs'], "data mode only support abs or complex now"

        if(reduceMode == "reduce"):
            tList = [1,5,15,20]
        else:
            tList = range(1,21)
        
        if('nolattice_offset' in samplingMode):
            offset = int(samplingMode[-1])

        # for the 30 patients
        for i in tqdm(iRange):
            for z in range(0,5): #0,4
                for t in tList: #1,20
                    filename = "data/cardiac_ktz/mr_heart_p%02dt%02dz%d.png" % (i,t,z)
                    im = Image.open(filename)
                    im_np = np.array(im).astype(np.float32)/255.
                  
                    # cartisan sampling
                    if(samplingMode == 'fakeRandom' or ('rand' in samplingMode)):
                        if(staticRandom): # whether random choose mask
                            randI = index
                        else:
                            randI = random.randrange(miList.shape[1]) # randomness

                        mi = miList[:,randI] # shape (256,)
                        mask = subsampling_mask(im_np, 0, 'fakeRandom', mi) # mask with selected row all 1
                    # non cartisan sampling(not test yet)
                    elif(samplingMode == 'lattice8'):
                        mask = subsampling_mask(im_np, offset, 'lattice8')
                    else:
                        mask = subsampling_mask(im_np, offset)
                    
                    m = np.fft.ifftshift(mask)  
                    y = np.zeros((1,256,256))

                    if(mode == 'abs'):
                        y = np.zeros((1,256,256))
                    else: # complex mode
                        y = np.zeros((2,256,256))

                    y[0] = im_np

                    self.yList.append(y)
                    self.mList.append(m) 
                    self.meanList.append(torch.tensor([0]))
                    self.stdList.append(torch.tensor([1]))
                    
                    if(samplingMode == 'nolattice'):
                        pass
                    else:
                        offset = (offset+1)%4

                    index += 1

    def __getitem__(self, index):
        i = index
        label = self.yList[i]
        mask = self.mList[i]
        mean = self.meanList[i]
        std = self.stdList[i]
        
        return label, mask, mean, std

    def __len__(self):
        return len(self.yList)


class genFastMRI(data.Dataset):
    """
    f keys:
        kspace: (34,640,372), complex
        reconstruction_esc: (34,320,320)
        reconstruction_rss: (34,320,320)
    """

    def __init__(self, isTrain = 'train', mode = 'abs', samplingMode = 'default', reduceMode = "", staticRandom = False):

        if(samplingMode == 'random'):
            mDic = sio.loadmat(fakeRandomPath_FastMRI_25) # mask
            miList = mDic['RAll']
        else:
            raise NotImplementedError("FastMRI can only accept random25 mask now")

        if(reduceMode == "reduce"): # only use part of training data
            isReduce = True
            pathDir = pathFastMRI_train 
            if(isTrain):
                ir_start=0
                ir_end=140
            else:
                ir_start=140
                ir_end=150
        else: # 
            isReduce = False
            if(isTrain):
                pathDir = pathFastMRI_train
            else:
                pathDir = pathFastMRI_eval

        if(os.path.exists(pathDir)):
            listF = os.listdir(pathDir)
            listF = sorted(listF)
        else:
            assert False, "no such path:" + pathDir

        self.yList = []
        self.mList = []
        self.meanList = []
        self.varList = []
        
        assert mode in ['complex','abs'], "real mode is abandoned"
        self.mode = mode
        offset = 0
        SIZE = 320
        
        if('nolattice_offset' in samplingMode):
            offset = int(samplingMode[-1])
        
        for index in trange(len(listF)): # for each file
           
            if(isReduce):
                if(index<ir_start or index>=ir_end):
                    continue

            filename = listF[index]
            f = h5py.File(pathDir + filename, 'r')
            esc = np.array(f['reconstruction_esc']) #(number of slices, height, width)

            for j in range(esc.shape[0]): # for each slice
            
                # skip the first 10
                if j < 10: continue

                im_np = esc[j].astype(np.float32) # (height, width)
                im_np, mean, std = im_normalize_re(im_np)
                im_np = im_np.clip(-6,6)
                
                if(samplingMode == 'fakeRandom' or ('rand' in samplingMode)):
                    if(staticRandom):
                        randI = index
                    else:
                        randI = random.randrange(miList.shape[1]-1) # miList : (320, 10000)
                   
                    mi = miList[:,randI]
                    mask = subsampling_mask(im_np, 0, 'fakeRandom', mi)

                else:
                    assert False, "fastMRI can only accept Cartisian mask only"

                m = np.fft.ifftshift(mask) 
                y = np.zeros((1,SIZE,SIZE))

                if(mode == 'abs'):
                    y = np.zeros((1,SIZE,SIZE))
                else:
                    y = np.zeros((2,SIZE,SIZE))
                
                y[0] = im_np 
               
                self.yList.append(y)
                self.mList.append(m)
                self.meanList.append(mean)
                self.varList.append(std)
                
            f.close()


    def __getitem__(self, index):
        i = index
        label = self.yList[i]
        mask = self.mList[i]
        mean = self.meanList[i]
        std = self.varList[i]
        
        return label, mask, mean, std

    def __len__(self):
        return len(self.yList)


class genBrainMRI(data.Dataset):

    def __init__(self, isTrain = True, mode = 'complex', samplingMode = 'default', reduceMode = "", staticRandom = False):

        if(samplingMode == 'random'): # 30%
            mDic = sio.loadmat(fakeRandomPath) 
            miList = mDic['RAll']
        elif(samplingMode == 'rand15'): # we use this
            mDic = sio.loadmat(fakeRandomPath_15)
            miList = mDic['RAll']
        elif(samplingMode == 'rand10'):
            mDic = sio.loadmat(fakeRandomPath_10)
            miList = mDic['RAll']
        elif(samplingMode == 'rand5'):
            mDic = sio.loadmat(fakeRandomPath_5)
            miList = mDic['RAll']

        
        if(reduceMode == "reduce"): # only use part of training data + val data
            isReduce = True
            pathDir = pathBrainMRI_train 
            if(isTrain):
                ir_start=0
                ir_end=28
            else:
                ir_start=28
                ir_end=31
        else: # 
            isReduce = False
            if(isTrain):
                pathDir = pathBrainMRI_train
            else:
                pathDir = pathBrainMRI_eval

        if(os.path.exists(pathDir)):
            listF = os.listdir(pathDir)
            listF = sorted(listF)
        else:
            assert False, "no such path:" + pathDir

        self.yList = []
        self.mList = []
        self.meanList = []
        self.varList = []

        assert mode in ['complex','abs'], "real mode is abandoned"
        self.mode = mode
        offset = 0
        SIZE = 256
        
        if('nolattice_offset' in samplingMode):
            offset = int(samplingMode[-1])

        for index in trange(len(listF)): # for each file
            if(isReduce):
                if(index<ir_start or index>=ir_end):
                    continue

            filename = listF[index]
            kspace = torch.from_numpy(np.load(pathDir + filename)) #(number of slices, h, w, 2)
            if len(kspace) < 100: continue 
            data = imgFromSubF_pytorch(kspace,False) # return absolute image

            for j in range(data.shape[0]): # for each slice
                
                if j < 30 or j > 140: continue

                im_np = data[j].numpy() #(height, width)
                im_np = im_np[0]
               
                im_np, mean, std = im_normalize_re(im_np)  # dev
                im_np = im_np.clip(-6,6)

                if(samplingMode == 'fakeRandom' or ('rand' in samplingMode)):
                    if(staticRandom):
                        randI = index
                    else:
                        randI = random.randrange(miList.shape[1]-1) # miList : (320, 10000)
                   
                    mi = miList[:,randI]
                    mask = subsampling_mask(im_np, 0, 'fakeRandom', mi)
                else:
                    assert False, "brainMRI can only accept Cartisian mask only"

                m = np.fft.ifftshift(mask) # what happend here?
                y = np.zeros((1,SIZE,SIZE))
                
                if(mode == 'abs'):
                    y = np.zeros((1,SIZE,SIZE))
                elif(mode == 'complex'):
                    y = np.zeros((2,SIZE,SIZE))
                else:
                    assert False,"real mode is abandoned"
                
                y[0] = im_np

                self.yList.append(y)
                self.mList.append(m)
                self.meanList.append(mean)
                self.varList.append(std)


    def __getitem__(self, index):
        i = index
        label = self.yList[i]
        mask = self.mList[i]
        mean = self.meanList[i]
        std = self.varList[i]
        
        return label, mask, mean, std

    def __len__(self):
        return len(self.yList)



'''
class genFastMRI_fastmri(data.Dataset):
    """
    support fastmri api
    f keys:
        kspace: (34,640,372), complex
        reconstruction_esc: (34,320,320)
        reconstruction_rss: (34,320,320)
    """

    def __init__(self, isTrain = True, reduceMode = "", centerFrac = 0, acceleration = 0, seed = False):

        mask_func = RM(center_fractions=[centerFrac], accelerations=[acceleration])
        # use all the data or portion data
        if(reduceMode == "reduce"): # only use validation data
            isReduce = True
            pathDir = pathFastMRI_train # use validation datset
            if(isTrain):
                ir_start=0
                ir_end=180
            else:
                ir_start=180
                ir_end=199
        else: # 
            isReduce = False
            if(isTrain):
                pathDir = pathFastMRI_train
            else:
                pathDir = pathFastMRI_eval
        if(os.path.exists(pathDir)):
            listF = os.listdir(pathDir)
        else:
            assert False, "no such path:" + pathDir

        self.xList = [] # zero-filled images
        self.yList = [] # gt
        self.kList = [] # masked kspace
        self.meanList = [] 
        self.varList = []
            
        index = 0
        SIZE = 320
        crop_size = (SIZE,SIZE)
        
        for filename in tqdm(listF): # for each file
            if(isReduce):
                if(index<ir_start or index>=ir_end):
                    index += 1
                    continue

            f = h5py.File(pathDir + filename, 'r')
            kspace_all = to_tensor(np.array(f['kspace'])) #(34,640,372,2) : (number of slices, height, width, 2), 2 is channel
            target_all = to_tensor(np.array(f['reconstruction_esc']))  #(34, 320, 320)

            for j in range(kspace_all.shape[0]): # for each slice
                print("process {}".format(j))
                kspace = kspace_all[j]
                target = target_all[j]
                masked_kspace, mask = apply_mask(kspace, mask_func, seed)
                zimage = ifft2c(masked_kspace) # zero-filled image
                assert zimage.shape[-2] >= crop_size[0], "get improper image size for the zero-filled image"
               
                # normalize zim
                zimage = complex_center_crop(zimage,crop_size)
                zimage = complex_abs(zimage)
                zimage, mean, std = im_normalize_re(zimage)
                zimage = zimage.clamp(-6,6)

                # normalize target : what is target here reconstruction_ess
                #target = center_crop(target, crop_size)
                assert target.shape[1] == SIZE, "non matched target size"
                target = normalize(target, mean, std, eps=1e-11)
                target = target.clamp(-6, 6) 

                self.yList.append(target)
                self.xList.append(zimage)
                self.kList.append(masked_kspace)
                self.meanList.append(mean)
                self.varList.append(std)
                
                index += 1

            f.close()

    def __getitem__(self, index):
        
        i = index
        inputs = self.xList[i]
        label = self.yList[i]
        mask = self.kList[i]
        mean = self.meanList[i]
        std = self.varList[i]

        
        return inputs, label, mask, mean, std

    def __len__(self):
        return len(self.yList)

    

'''

if __name__ == '__main__':
   
    #datatype =  'cardiac_complex_static_rand15'
    datatype =  'fastmri_complex_static_random_reduce'
    #datatype =  'brain_complex_static_rand15_reduce'
    _,dataSize1 = getDataloader(dataType = datatype, isTrain = 1, batchSize = 1)
    print(dataSize1) # 3000




