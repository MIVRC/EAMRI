# main file for logger

import datetime
import torch
import scipy.io as sio
import os
import numpy as np
import pdb

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def deleteFile(path):
    if(os.path.exists(path)):
        os.remove(path)

def saveNet(param, saveDir, epochNum, epochOffset, checkpoint=0):
    """
    main function to save model weight. Three types with different postfix can be saved: "CHECKED", "BEST" and "saved"
    saveDir: save dir for model weight
    epochNum: current epoch number of training
    epochOffset: determine which file to be deleted
    checkpoint: save type
    """
    tmpDir = saveDir
    tmpPath = tmpDir+"/saved"+"_"+str(epochNum)+".pkl"
    if checkpoint == 1: # save checkpoint
        tmpPath = tmpDir+"/CHECKED_saved"+"_"+str(epochNum)+".pkl"
    elif checkpoint == 2: # save the best results
        names= [ele for ele in os.listdir(tmpDir) if 'BEST' in ele]
        if len(names):
            assert len(names) == 1, "many BEST appear in the model path!"
            name = names[0] 
            prePath = os.path.join(tmpDir,name)
            deleteFile(prePath)

        tmpPath = tmpDir+"/BEST_saved"+"_"+str(epochNum)+".pkl"

    oldNum = epochNum - epochOffset
    if oldNum > 0:
        oldPath = tmpDir+"/saved"+"_"+str(oldNum)+".pkl" 
        deleteFile(oldPath)

    torch.save(param,tmpPath)


def saveCheckpoints(param, saveDir, epochNum, epochOffset, checkpoint=False):
    tmpDir = saveDir
    tmpPath = tmpDir+"/saved"+"_"+str(epochNum)+".pth"
    if(checkpoint):
        tmpPath = tmpDir+"/CHECKED_saved"+"_"+str(epochNum)+".pth"

    # remove old files
    files = [ele for ele in os.listdir(tmpDir) if 'CHECKED' not in ele]
    if len(files) > epochOffset:
        oldPath = os.path.join(tmpDir, files[-1])
        os.remove(oldPath)

    torch.save(param,tmpPath)
 


def loadNet(net, loadDir, epochNum, loadType):
    tmpDir = loadDir
    if loadType == 0:
        tmpPath = tmpDir+"/saved"+"_"+str(epochNum)+".pkl"
    elif loadType == 1:
        tmpPath = tmpDir+"/CHECKED_saved"+"_"+str(epochNum)+".pkl"
    elif loadType == 2:
        tmpPath = tmpDir+"/BEST_saved"+"_"+str(epochNum)+".pkl"
    else:
        raise NotImplementedError("loadNet do not have correct loadType!")

    net.load_state_dict(torch.load(tmpPath))

class Recorder():
    def __init__(self, path, offsetEpoch):
        """
        path:
        offsetEpoch:
        """
        
        self.trained_epoch = 0 
        self.save_offset = offsetEpoch # for delete old weight file
        self.trainRecord = {"epoch":[], "trainLoss":[]}
        self.validRecord = {"epoch":[], "validLoss":[], "PSNR":[], "PSNRafter":[], "SSIM":[], "SSIMafter":[]}
        self.rootPath = path
        self.logPath = self.rootPath+'/log'
        self.weightPath = self.rootPath+'/weight'
        mkdir(self.logPath)
        mkdir(self.weightPath)

    
    def log(self,msg):
        """
        log msg in the log file
        """
        print(msg)
        f = open(self.logPath+"/generalLog.txt","a+")
        timestamp = datetime.datetime.now().strftime("[%m-%d  %H:%M:%S] ")
        f.write(timestamp + msg + "\n")
        f.close()


    def log_train(self,epoch0,loss0):
        self.trainRecord["epoch"].append(epoch0)
        self.trained_epoch = epoch0
        self.trainRecord["trainLoss"].append(loss0)
    
    def log_valid(self,epoch0,loss0,psnr0,psnrafter0,ssim0,ssimafter0):
        self.validRecord["epoch"].append(epoch0)
        self.validRecord["validLoss"].append(loss0)
        self.validRecord["PSNR"].append(psnr0)
        self.validRecord["PSNRafter"].append(psnrafter0)
        self.validRecord["SSIM"].append(ssim0)
        self.validRecord["SSIMafter"].append(ssimafter0)
       
    def logTrainLoss(self,msg):
        # record the training loss
        f = open(self.logPath+"/trainLossLog.txt","a+")
        timestamp = datetime.datetime.now().strftime("[%m-%d  %H:%M:%S] ")
        f.write(timestamp + "loss: " + msg + "\n")
        f.close()

            
    def logNet(self,net):
        f = open(self.logPath+"/netInfo.txt","w")
        timestamp = datetime.datetime.now().strftime("[%m-%d  %H:%M:%S] ")
        f.write(timestamp + str(net) + "\n")
        f.close()
        
    def write_to_file(self,param,isCheckpoint): 
        sio.savemat(self.logPath+"/trainLog.mat",self.trainRecord)
        sio.savemat(self.logPath+"/validLog.mat",self.validRecord)
        f = open(self.logPath+"/generalLog.txt","a+")
        timestamp = datetime.datetime.now().strftime("[%m-%d  %H:%M:%S] ")
        saveNet(param, self.weightPath, self.trained_epoch, self.save_offset, isCheckpoint) 
        if isCheckpoint != 2:
            f.write(timestamp + "file updated epoch:%05d\n" % (self.trained_epoch))
        f.close()


    def write_checkpoints(self,checkpoint,isCheckpoint, is_resume = 0, starti = 0, endi = 0):
        if is_resume:
            sio.savemat(self.logPath+"/trainLog_{}_{}.mat".format(starti, endi),self.trainRecord)
            sio.savemat(self.logPath+"/validLog_{}_{}.mat".format(starti, endi),self.validRecord)
        else:
            sio.savemat(self.logPath+"/trainLog.mat",self.trainRecord)
            sio.savemat(self.logPath+"/validLog.mat",self.validRecord)

        f = open(self.logPath+"/generalLog.txt","a+")
        timestamp = datetime.datetime.now().strftime("[%m-%d  %H:%M:%S] ")
        saveCheckpoints(checkpoint, self.weightPath, self.trained_epoch, self.save_offset, isCheckpoint)
        f.write(timestamp + "file updated epoch:%05d\n" % (self.trained_epoch))
        f.close()
 


    def load_from_file(self, net, expectedEpoch, loadType='normal', inTrain = True):
        if(expectedEpoch == 0):
            return
        tmpTrainRecord = sio.loadmat(self.logPath+"/trainLog.mat")
        tmpValidRecord = sio.loadmat(self.logPath+"/validLog.mat")
        lastRecord1 = np.max(tmpTrainRecord['epoch'])
        lastRecord2 = np.max(tmpValidRecord['epoch'])
        print(">>>Log Data Loaded!")
        print(">>>[Log] last/target epoch:%d/%d"%(lastRecord1,expectedEpoch))

        if(expectedEpoch in tmpTrainRecord["epoch"]):
            for key in self.trainRecord.keys():
                self.trainRecord[key] = tmpTrainRecord[key][0].tolist()
            index = self.trainRecord["epoch"].index(expectedEpoch)
            for key in self.trainRecord.keys():
                #self.trainRecord[key] = tmpTrainRecord[key][0].tolist()
                self.trainRecord[key] = self.trainRecord[key][:index+1]
        else:
            assert False, "No expectedEpoch in trainRecord"

        if(expectedEpoch in tmpValidRecord["epoch"]):
            for key in self.validRecord.keys():
                self.validRecord[key] = tmpValidRecord[key][0].tolist()
            index = self.validRecord["epoch"].index(expectedEpoch)
            for key in self.validRecord.keys():
                #self.validRecord[key] = tmpValidRecord[key][0].tolist()
                self.validRecord[key] = self.validRecord[key][:index+1]
        else:
            assert False, "No expectedEpoch in validRecord"
        
        loadNet(net, self.weightPath, expectedEpoch, loadType)
        self.trained_epoch = expectedEpoch
        if(inTrain):
            self.log(".mat file load from epoch:%05d\n" % (expectedEpoch))
