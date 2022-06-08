"""
modified on 20210921
try to seperate core and main
add mean and std for fastmri and brain
"""

import os
import sys
import random
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
from torch.nn import MSELoss as mse
import torch.optim
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
import numpy as np
from PIL import Image
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
from util import imshow, img2f, f2img, kspace_subsampling, Recorder, paramNumber, kspace_subsampling_pytorch, imgFromSubF_pytorch
from network import getNet, getLoss, getOptimizer
from dataProcess import getDataloader
import cv2
import configparser
import warnings
warnings.filterwarnings("ignore")
import pdb


class Core():
    def __init__(self,configName, isEvaluate = False):

        configPath = 'config/' + configName + '.ini'
        self.config = configparser.ConfigParser()
        self.config.read(configPath)

        self.model_name = configName
        
        # init random
        self.seed = int(self.config['general']['SEED'])
        self._init_random(self.seed)
        print("#SEED:"+str(self.seed))

        # [general]
        self.dataType = self.config['general']['dataType']
        self.netType = self.config['general']['netType']
        self.modelPath = self.config['general']['path']
        self.num_workers = int(self.config['general']['num_workers']) 
        self.opt = self.config['train']['optimizer']
        self.mode = 'complex' if 'complex' in self.dataType else 'abs' 
        self.useCuda = self.config.getboolean('general','useCuda')
        self.needParallel = self.config.getboolean('general','needParallel')
        self.dtype = torch.cuda.FloatTensor if self.useCuda else torch.FloatTensor

        print("#Data Type:"+self.dataType)
        print("#Model Path:"+self.modelPath)
        print("#Create network:"+self.netType)
        print("#Mode:"+self.mode)

        # adjust range for psnr
        self.scaleVal = 0 if 'cardiac' in self.dataType.lower() else 1
        if(self.scaleVal):
            print("data range for PSNR: 0~12 (estimated)")
        else:
            print("data range for PSNR: 0~1")


        # [train]
        self.batchSize = int(self.config['train']['batchSize'])
        self.epoch = int(self.config['train']['epoch'])
        self.LR = float(self.config['train']['lr'])
        self.lossLambda = float(self.config['train']['lossLambda'])
       
        # [log]
        self.saveEpoch = int(self.config['log']['saveEpoch'])
        self.maxSaved = int(self.config['log']['maxSaved'])
       
        # device
        listDeviceStr = self.config['general']['device']
        listDevice = []
        for i in range(len(listDeviceStr)):
            if(listDeviceStr[i] == '1'):
                listDevice.append(str(i))
            devicesStr=','.join(listDevice)

        assert len(listDevice) > 0,"No device is selected"
        os.environ["CUDA_VISIBLE_DEVICES"] = devicesStr
        
        # net 
        self.net = getNet(self.netType).type(self.dtype)
        if(self.needParallel):
            print('parallel device:'+str(listDevice))
            self.net = nn.DataParallel(self.net, device_ids = list(range(len(listDevice)))).type(self.dtype)
        paramNum = paramNumber(self.net)

        # dataloader
        if(not isEvaluate):
            self.trainloader,self.trainsetSize = getDataloader(self.dataType, 1, self.batchSize, self.seed, self.num_workers)
        
        self.testloader,self.testsetSize = getDataloader(self.dataType, 0, 1, self.seed, self.num_workers)
 
        # recorder
        self.record = Recorder(self.modelPath,self.saveEpoch*self.maxSaved)

        # do not train
        if(not isEvaluate):
            self.config.write(open(self.record.rootPath+"/config.ini","w"))
            self.record.logNet(self.net)

        self.record.log("#Number of network parameters:%d"%paramNum)

        # optimizer
        self.optimizer = getOptimizer(self.net.parameters(), self.opt, self.LR)

        schedulerStep = 200
        self.scheduler = StepLR(self.optimizer,step_size=schedulerStep,gamma=0.5) 
        
        print('#Optimizer: '+ self.opt + ' LR = %.2e weightDecay = %.2e'%(self.LR,1e-7))
        print('#Scheduler step: {}'.format(schedulerStep))

        self.ckp_epoch = 0 
        self.loss_forward = mse()
        self.loss_list = [] 
        self.bestScore = -99999

    def train(self, need_evaluate = True):

        epoch = self.epoch
        reco = self.record
        msg = "start training: epoch = %d"%(epoch)
        reco.log(msg)
        dataset_size = len(self.trainloader)

        t1 = time.time()
        for j in range(1,epoch + 1): # for each epoch
            self.net.train()
            i = 0
            total_loss = 0.0
            for label, mask, _, _ in self.trainloader: # label: (8,2,256,256), mask: (8,256, 256) 8 is batch_size

                for param in self.net.parameters():
                    param.grad = None

                netLabel = Variable(label).type(self.dtype)
                mask_var = Variable(mask).type(self.dtype) #(8,256,256)
                subF = kspace_subsampling_pytorch(netLabel,mask_var) # subsampled kspace (8,256,256,2)
                complexFlag = (self.mode == 'complex') 
                netInput = imgFromSubF_pytorch(subF,complexFlag) # zero-filled reconstruction (8,2,256,256)

                netOutput = self.net(netInput, subF, mask_var)
                if type(netOutput) != list: 
                    loss = self.loss_forward(netOutput,netLabel)
                else:
                    loss = 0.0
                    for kk, tmp1 in enumerate(netOutput):
                        loss_tmp = self.loss_forward(tmp1,netLabel)
                        if kk < len(netOutput) - 1:
                            loss += self.lossLambda * loss_tmp
                        else:
                            loss += loss_tmp # for the last one, add full loss
                    
                loss.backward()
                total_loss = total_loss + loss.item()*label.shape[0]
                self.optimizer.step()
                i += label.shape[0]
                print ('Epoch %05d [%04d/%04d] loss %.8f ' % (j+self.ckp_epoch,i,self.trainsetSize,loss.item()),'\r', end='') # print batch loss


            reco.log_train(j+self.ckp_epoch,total_loss/i) # log the training process

            #======================
            # testing
            #======================
            
            if j % self.saveEpoch == 0:
                print ('Epoch %05d [%04d/%04d] loss %.8f time %.5f SAVED' % (j+self.ckp_epoch,i,self.trainsetSize,total_loss/self.trainsetSize, float((time.time() - t1)/self.saveEpoch)))
                if need_evaluate:
                    l,p1,p2,s1,s2 = self.validation()
                    reco.log_valid(j+self.ckp_epoch,l,p1,p2,s1,s2)
                    reco.log("psnr = %.2f|%.2f, ssim = %.4f|%.4f, loss = %.8f"%(p1,p2,s1,s2,total_loss/self.trainsetSize))
                    
                    # save the best result
                    if p1 > self.bestScore:
                        self.bestScore = p1
                        reco.write_to_file(self.net.state_dict(), 2) # save to the best model
                    
                reco.write_to_file(self.net.state_dict(), 0) # normal save
                t1 = time.time()

            self.scheduler.step()
            print('[lr %.8f]' % self.scheduler.get_lr()[0],'\r',end='')

        self.ckp_epoch += epoch
        reco.write_to_file(self.net.state_dict(), 1)
   
    
    @classmethod
    def calPsnrSsim(cls,label,img1,img2, isScale):

        if not isScale: 
            img1 = np.clip(img1,0,1)
            img2 = np.clip(img2,0,1)

        psnr1 = psnr(label,img1,data_range=label.max())
        psnr2 = psnr(label,img2,data_range=label.max())
        ssim1 = ssim(label,img1,data_range=label.max())
        ssim2 = ssim(label,img2,data_range=label.max())

        return psnr1, psnr2, ssim1, ssim2 



    def testValue(self,mode,label,mask, mean, std):
        
        result = self.test(mode,label,mask, mean, std)
        return result['loss'],result['psnr1'],result['psnr2'],result['ssim1'],result['ssim2']


    # main function to calculate metrics
    def test(self,mode,label,mask, mean, std):

        mean = mean.numpy()[0]
        std = std.numpy()[0]

        y = label.numpy() 
        netLabel = Variable(label).type(self.dtype)
        
        mask_var = Variable(mask).type(self.dtype)
        subF = kspace_subsampling_pytorch(netLabel,mask_var)
        complexFlag = (mode == 'complex')
        netInput = imgFromSubF_pytorch(subF,complexFlag) #(1,2,256,256) 

        y2=y[0,0] #(1,256,256) take the first channel, real part
        y2 = y2 * std + mean
        
        netOutput = self.net(netInput, subF, mask_var)
        
        if type(netOutput) == list:
            netOutput = netOutput[-1] #take the final result as output

        loss = self.loss_forward(netOutput,netLabel) # take the final result as output
        netOutput_np = netOutput.cpu().data.numpy() #(1,2,256,256), for img1
        netOutput_abs = abs(netOutput_np[0,0] + netOutput_np[0,1]*1j) # (256, 256) # for img2

        netOutput_np = netOutput_np * std + mean
        netOutput_np1 = netOutput_abs * std + mean # dev (256,256)
        
        img1 = netOutput_np[0,0].astype('float64') # take the real part
        img2 = netOutput_np1.astype('float64')

        if len(img1.shape) == 3:
            img1 = img1[0]

        if len(img2.shape) == 3:
            img2 = img2[0]
       
        psnrBefore, psnrAfter, ssimBefore, ssimAfter = self.calPsnrSsim(y2,img1,img2,self.scaleVal)
        return {"loss":loss.item(),"psnr1":psnrBefore,"psnr2":psnrAfter,"ssim1":ssimBefore,"ssim2":ssimAfter,"result1":img1,"result2":img2,'label':y2}
  

    def eval_single_branch(self,mode,label,mask, mean, std):
        """
        # output psnr and ssim for different branchs
        """

        y = label.numpy() 
        mean = mean.numpy()[0]
        std = std.numpy()[0]

        netLabel = Variable(label).type(self.dtype)
        mask_var = Variable(mask).type(self.dtype)
        subF = kspace_subsampling_pytorch(netLabel,mask_var)
        complexFlag = (mode == 'complex')
        netInput = imgFromSubF_pytorch(subF,complexFlag) #(1,2,256,256) 

        y2 = y[0,0]
        y2 = y2 * std + mean
        
        netOutputs = self.net(netInput, subF, mask_var)

        loss_branch = []
        psnrb_branch = []
        psnra_branch = []
        ssimb_branch = []
        ssima_branch = []

        if type(netOutputs) == list:
            
            for netOutput in netOutputs:
                loss = self.loss_forward(netOutput,netLabel) # take the final result as output
                netOutput_np = netOutput.cpu().data.numpy() #(1,2,256,256)
                netOutput_np = netOutput_np * std + mean

                img1 = netOutput_np[0,0].astype('float64') # take the real part
                if(netOutput_np.shape[1]==2):
                    netOutput_np = abs(netOutput_np[:,0:1]+netOutput_np[:,1:2]*1j) # take the abs
                img2 = netOutput_np[0].astype('float64') 
 
                if len(img1.shape) == 3:
                    img1 = img1[0]
                if len(img2.shape) == 3:
                    img2 = img2[0]

                psnrBefore, psnrAfter, ssimBefore, ssimAfter = self.calPsnrSsim(y2,img1,img2,self.scaleVal)
                
                loss_branch.append(loss.item())
                psnrb_branch.append(psnrBefore.item())
                psnra_branch.append(psnrAfter.item())
                
                ssimb_branch.append(ssimBefore.item())
                ssima_branch.append(ssimAfter.item())

        return {"loss":loss_branch,"psnr1":psnrb_branch,"psnr2":psnra_branch,"ssim1":ssimb_branch,"ssim2":ssima_branch}

    def validation(self,returnList = False):
        
        self.net.eval()
        i = 0
        totalLoss = 0
        lpsnr1 = []
        lpsnr2 = []
        lssim1 = []
        lssim2 = []
        psnr1 = psnr2 = ssim1 = ssim2 = 0
       
        for label, mask, mean, std in self.testloader:
            loss0,psnrA,psnrB,ssimA,ssimB = self.testValue(self.mode, label,mask, mean, std)
            totalLoss += loss0
            psnr1 += psnrA
            psnr2 += psnrB
            ssim1 += ssimA
            ssim2 += ssimB
            lpsnr1.append(psnrA)
            lpsnr2.append(psnrB)
            lssim1.append(ssimA)
            lssim2.append(ssimB)
            i+=1
            print ('Evaluating %04d psnr(before|after) =  %.2f|%.2f ssim = %.4f|%.4f' % (i, psnrA, psnrB, ssimA, ssimB), '\r', end='')
        print ('Evaluating %04d psnr(before|after) =  %.2f|%.2f ssim = %.4f|%.4f Done' % (i, psnr1/i, psnr2/i, ssim1/i, ssim2/i))
        if(returnList):
            return lpsnr1,lpsnr2,lssim1,lssim2
        return totalLoss/i, psnr1/i, psnr2/i, ssim1/i, ssim2/i

    def validation_branch(self):
        self.net.eval()
        totalLoss = 0
        psnr1 = psnr2 = ssim1 = ssim2 = 0
        N = len(self.testloader)
        for label, mask, mean, std in self.testloader: # batchsize=1
            result = self.eval_single_branch(self.mode, label, mask,mean,std)
            totalLoss += np.array(result['loss']) # np.array([br1,br2,br3,br4])
            psnr1 += np.array(result['psnr1'])
            psnr2 += np.array(result['psnr2'])
            ssim1 += np.array(result['ssim1'])
            ssim2 += np.array(result['ssim2'])

        totalLoss /= N
        psnr1 /= N
        psnr2 /= N
        ssim1 /= N
        ssim2 /= N

        print(totalLoss)  
        print(psnr1)  
        print(psnr2)  
        print(ssim1)  
        print(ssim2)  


    # plot the results
    def plot_results(self, save_root):
        
        self.net.eval()
        idx = 0

        # mapping btw model names
        model_remap = {'Unet_dc':'U-Net', 
                    'DCCNN':'DCCNN',
                    'CN_dOri_c5_complex_tr_trick2':'CDDNTDC',
                    'RDN_complex_DC':'RDN',
                    'mdr':'MDR',
                    'wasnet':'SEN-MRI',
                    'wasnet_ab7':'SEN-MRI_ab7',
                    'wasnet_ab6':'SEN-MRI_ab6',
                    'wasnet_5b':'SEN-MRI_5B'}
        
        cm = plt.cm.jet
        
        for mode, label, mask, mean, std in self.testloader:
      
            print("Processing {}".format(idx))
            im_root = os.path.join(save_root, 'test_image{}'.format(idx))
            if not os.path.exists(im_root):
                os.makedirs(im_root)

            mean = mean.numpy()[0]
            std = std.numpy()[0]
            
            idx += 1
            y = label.numpy() 
            y=y[0,0,:,:] #(256,256) take the first channel
            
            netLabel = Variable(label).type(self.dtype)
            mask_var = Variable(mask).type(self.dtype)
            subF = kspace_subsampling_pytorch(netLabel,mask_var)
            complexFlag = (self.mode == 'complex')
            netInput = imgFromSubF_pytorch(subF,complexFlag) #(1,2,256,256) 

            netOutput = self.net(netInput, subF, mask_var)
            if type(netOutput) == list:
                netOutput = netOutput[-1] #take the final result as output

            netOutput_np = netOutput.cpu().data.numpy() #(1,2,256,256)
            img1 = netOutput_np[0,0,:,:].astype('float64') # take the real part
         
            zim = netInput.cpu().numpy()[0,0] # take the first part
            
            # transform back using mean and std
            y = y * std + mean
            img1 = img1*std + mean
            zim = zim*std + mean

            # calculate psnr
            im_psnr = np.round(psnr(y, img1, data_range=y.max()), decimals=2)
            zim_psnr = np.round(psnr(y, zim, data_range=y.max()), decimals=2)

            res = np.abs(y-img1)/(y.max()+1e-11)
            res[res < 0.02] = 0
            zim_res = np.abs(y-zim)/(y.max()+1e-11)
            zim_res[zim_res < 0.02] = 0

            # normalize
            y = cv2.normalize(y, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
            img1 = cv2.normalize(img1, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
            zim = cv2.normalize(zim, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)

            res = cv2.normalize(res, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
            zim_res = cv2.normalize(zim_res, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)

            norm = plt.Normalize(vmin=res.min(),vmax=res.max())
            res1 = cm(norm(res))

            norm = plt.Normalize(vmin=zim_res.min(),vmax=zim_res.max())
            zim_res1= cm(norm(zim_res))


            # convert to uint8
            y = np.uint8(y)
            img1 = np.uint8(img1)
            zim = np.uint8(zim)
            mask_np = np.fft.fftshift(mask[0].numpy())
        
            # save to image
            y = Image.fromarray(y)  
            img1 = Image.fromarray(img1)
            zim = Image.fromarray(zim)
          
            y.save(os.path.join(im_root, 'gt.png'))
            img1.save(os.path.join(im_root, '{}_pred_PSNR{}.png'.format(model_remap[self.netType], im_psnr)))
            zim.save(os.path.join(im_root, 'zim_PSNR{}.png'.format(zim_psnr)))

            plt.imsave(os.path.join(im_root, '{}_res.png'.format(model_remap[self.netType])), res1)
            plt.imsave(os.path.join(im_root,'zim_res.png'), zim_res1)
            
            plt.imsave(os.path.join(im_root, 'mask.png'), mask_np,cmap='gray')




    def plot_results_to_one_folder(self, save_root):
        
        """
        plot result to a folder for only one model
        """

        self.net.eval()
        idx = 0

        model_remap = {'Unet_dc':'U-Net', 
                    'DCCNN':'DCCNN',
                    'CN_dOri_c5_complex_tr_trick2':'CDDNTDC',
                    'RDN_complex_DC':'RDN',
                    'mdr':'MDR',
                    'wasnet':'SEN-MRI',
                    'wasnet_ab7':'SEN-MRI_ab7',
                    'wasnet_ab6':'SEN-MRI_ab6'}
        
        cm = plt.get_cmap('viridis')
        im_root = save_root
        if not os.path.exists(im_root):
            os.makedirs(im_root)

        gt_root = os.path.join(im_root,'gt')
        if not os.path.exists(gt_root):
            os.makedirs(gt_root)

        for label, mask, mean, std in self.testloader:
      
            print("Processing {}".format(idx))
            mean = mean.numpy()[0]
            std = std.numpy()[0]
            
            idx += 1
            y = label.numpy() 
            y=y[0,0,:,:] #(256,256) take the first channel
            
            netLabel = Variable(label).type(self.dtype)
            mask_var = Variable(mask).type(self.dtype)

            subF = kspace_subsampling_pytorch(netLabel,mask_var)
            complexFlag = (self.mode == 'complex')
            netInput = imgFromSubF_pytorch(subF,complexFlag) #(1,2,256,256) 

            netOutput = self.net(netInput, subF, mask_var)
            if type(netOutput) == list:
                netOutput = netOutput[-1] #take the final result as output

            netOutput_np = netOutput.cpu().data.numpy() #(1,2,256,256)
            img1 = netOutput_np[0,0,:,:].astype('float64') # take the real part
         
            zim = netInput.cpu().numpy()[0,0] # take the first part
            
            # transform back using mean and std
            y = y * std + mean
            img1 = img1*std + mean
            zim = zim*std + mean

            res = np.abs(y-img1)/(y.max()+1e-11)
            res[res < 0.05] = 0

            zim_res = np.abs(y-zim)/(y.max()+1e-11)
            zim_res[zim_res < 0.05] = 0

            # calculate psnr
            im_psnr = np.round(psnr(y, img1, data_range=y.max()), decimals=2)
            zim_psnr = np.round(psnr(y, zim, data_range=y.max()), decimals=2)

            # normalize
            y = cv2.normalize(y, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
            img1 = cv2.normalize(img1, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
            zim = cv2.normalize(zim, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)

            res = cv2.normalize(res, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
            zim_res = cv2.normalize(zim_res, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)

            # convert to uint8
            y = np.uint8(y)
            img1 = np.uint8(img1)
            zim = np.uint8(zim)
            res = np.uint8(cm(res)*255)
            zim_res = np.uint8(cm(zim_res)*255)
            mask_np = mask[0].numpy()

            # save to image
            y = Image.fromarray(y)  
            img1 = Image.fromarray(img1)
            zim = Image.fromarray(zim)
            res = Image.fromarray(res)
            zim_res = Image.fromarray(zim_res)
            
            y.save(os.path.join(gt_root, 'im_{}.png'.format(idx)))
            img1.save(os.path.join(im_root, 'im_{}.png'.format(idx)))




    def cal_metric_zim(self):
        """
        calculate psnr/ssim for zero-filled image
        """ 
        self.net.eval()
        idx = 0
        psnrs = []
        ssims = []
        for mode, label, mask, mean, std in self.testloader:
      
            print("Processing {}".format(idx))
            mean = mean.numpy()[0]
            std = std.numpy()[0]
            
            idx += 1
            y = label.numpy() 
            y=y[0,0,:,:] #(256,256) take the first channel
            
            netLabel = Variable(label).type(self.dtype)
            if(self.mode != 'inNetDC'):
                assert False, 'only for inNetDC mode'
            else:
                mask_var = Variable(mask).type(self.dtype)

                subF = kspace_subsampling_pytorch(netLabel,mask_var)
                complexFlag = (mode[0] == 'complex')
                netInput = imgFromSubF_pytorch(subF,complexFlag) #(1,2,256,256) 

            netOutput = self.net(netInput, subF, mask_var)
            if type(netOutput) == list:
                netOutput = netOutput[-1] #take the final result as output

            netOutput_np = netOutput.cpu().data.numpy() #(1,2,256,256)
            img1 = netOutput_np[0,0,:,:].astype('float64') # take the real part
         
            zim = netInput.cpu().numpy()[0,0] # take the first part
            
            # transform back using mean and std
            y = y * std + mean
            img1 = img1*std + mean
            zim = zim*std + mean

            # calculate psnr
            zim_psnr = np.round(psnr(y, zim, data_range=y.max()), decimals=2)
            zim_ssim =  np.round(ssim(y, zim, data_range=y.max()), decimals=2)

            psnrs.append(zim_psnr) 
            ssims.append(zim_ssim) 

        print(np.average(psnrs))
        print(np.average(ssims))

    def loadCkpt(self, expectedEpoch, loadType):
        """
        loadType: 
            0: normal 
            1: checked
            2: best
        """
        self.record.load_from_file(self.net, expectedEpoch, loadType, False)
        self.ckp_epoch = expectedEpoch


    def _init_random(self,SEED):

        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        np.random.seed(SEED)
        random.seed(SEED)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministric = True
        torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    pass






