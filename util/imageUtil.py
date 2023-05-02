import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
from PIL import Image
import os
import torch.nn as nn
import torch.nn.functional as F
import pdb


def Get_sobel(target):
    
    ddepth = cv2.CV_16S
    scale = 1
    delta = 0
    grad_x = cv2.Sobel(target, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(target, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return grad


def Get_canny(target):
    """
    gaussian blur 
    """    
    blurred_img = cv2.blur(target,ksize=(5,5)).astype(np.uint8)
    med_val = np.median(target) 
    lower = int(max(0 ,0.5*med_val))
    upper = int(min(255,1.5*med_val))
    edges = cv2.Canny(image=blurred_img, threshold1=lower,threshold2=upper) 

    return edges


def Get_prewitt(gt):

    kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    img_prewittx = cv2.filter2D(gt, -1, kernelx)
    img_prewitty = cv2.filter2D(gt, -1, kernely)

    return np.sqrt(img_prewittx**2 + img_prewitty**2)



def complex_abs(data):
    """
    Compute the absolute value of a complex valued input tensor.

    Args:
        data (torch.Tensor): A complex valued tensor, where the size of the final dimension
            should be 2.

    Returns:
        torch.Tensor: Absolute value of data
    """
    assert data.size(-1) == 2
    return (data ** 2).sum(dim=-1).sqrt()



def imshow(img,mode='pil',vmax=0,overlap=False):
    if(mode == 'cv'):
        b,g,r = cv2.split(img)
        im_np = cv2.merge([r,g,b])
        plt.imshow(im_np)
    elif(mode == 'pil'):
        if(vmax!=0):
            plt.imshow(img,vmax=vmax)
        else:
            plt.imshow(img)
    elif(mode == 'g'):
        plt.imshow(img,'gray')
    elif(mode == 'b'):
        plt.imshow(img,'binary')
    elif(mode == 'c'):
        if('complex' in str(img.dtype)):
            showImg = img
        else:
            showImg = fc2c(img)
        plt.imshow(abs(showImg),'gray')
    elif(mode == 'k'):
        draw_kspace_img(img)
    else:
        assert False,"wrong mode"
    
    plt.axis('off')
    if(~overlap):
        plt.show()

def fc2c(img):
    '''
    from fake_complex np.array, shape = [2,...] to complex np.array
    '''
    assert img.shape[0]==2,"the first dimension of img should be 2"
    outImg = img[0]+img[1]*1j

    return outImg


def draw_kspace_img(img):
    s1 = np.log(np.abs(img))
    plt.imshow(s1,'gray')
    
def img2f(img):
    f = np.fft.fft2(img, norm='ortho')
    f = np.fft.fftshift(f)
    
    return f

def f2img(f):
    f = np.fft.ifftshift(f)
    img = np.fft.ifft2(f, norm='ortho')
    
    return img

def kspace_subsampling(srcImg,offset = 0,mode="default",mi=None):
    mask = np.zeros_like(srcImg)
    if(mode == "default"):
        assert offset<4 , "offset out of range"
        mask[offset::4,:] = 1
        mask[122:134,:] = 1
    elif(mode == "lattice8"):
        assert offset<8 , "offset out of range"
        mask[offset::8,:] = 1
        mask[125:131,:] = 1
    elif(mode == "fakeRandom"):
        mask[mi==1,:] = 1
    else:
        assert False, "wrong subsampling mode"

    srcF = np.fft.fft2(srcImg, norm='ortho')
    srcF = np.fft.fftshift(srcF)
    tarF = srcF*mask
    
    return tarF,mask

def subsampling_mask(srcImg,offset=0, mode = "default", mi = None):
    mask = np.zeros_like(srcImg)
    if(mode == "default"):
        assert offset<4 , "offset out of range"
        mask[offset::4,:] = 1
        mask[122:134,:] = 1
    elif(mode == "lattice8"):
        assert offset<8 , "offset out of range"
        mask[offset::8,:] = 1
        mask[125:131,:] = 1
    elif(mode == "fakeRandom"): # cartesian sampling
        mask[mi==1,:] = 1 
    else:
        assert False, "wrong subsampling mode"
    
    return mask

def addZoomIn(img, x0 = 87, y0 = 135,offsetX=32,offsetY=0,scale=3,border = 0):
    if(len(img.shape)==3):
        im1 = np.zeros_like(img)
        im1 = img.copy()
    else:
        im1 = np.zeros((img.shape[0],img.shape[1],3))
        for i in range(3):
            im1[:,:,i] = img
    if(offsetY==0):
        offsetY = offsetX
    imzoomin = im1[y0:y0+offsetY,x0:x0+offsetX]
    imzoomin = cv2.resize(imzoomin,((offsetY*scale,offsetX*scale)))
    cv2.rectangle(im1,(x0,y0),(x0+offsetX,y0+offsetY),(255,0,0),1)
    im1[256-offsetY*scale:,256-offsetX*scale:] = imzoomin
    if(border>0):
        im1[-offsetX*scale-border:-offsetX*scale,256-offsetX*scale-border:] = (0,0,0)
        im1[256-offsetY*scale-border:,-offsetX*scale-border:-offsetX*scale] = (0,0,0)
    
    return im1

def saveIm(img,path):
    b,g,r = cv2.split(img) 
    img2 = cv2.merge([r,g,b])
    cv2.imwrite(path,img2)
    
def rgb2bgr(img):
    b,g,r = cv2.split(img) 
    img2 = cv2.merge([r,g,b])
    
    return img2

def kspace_subsampling_pytorch(srcImg,mask):
    '''
    return subF has shape[...,2], without permute
    '''
    y = srcImg # (bs, 2, 256, 256)
    if(len(y.shape)==4):
        if(y.shape[1]==1): # for real image
            emptyImag = torch.zeros_like(y)
            xGT_c = torch.cat([y,emptyImag],1).permute(0,2,3,1) 
        else:
            xGT_c = y.permute(0,2,3,1) # (bs, 256, 256, 2)
        mask = mask.reshape(mask.shape[0],mask.shape[1],mask.shape[2],1)
    elif(len(y.shape)==5):
        if(y.shape[1]==1):
            emptyImag = torch.zeros_like(y)
            xGT_c = torch.cat([y,emptyImag],1).permute(0,2,3,4,1)
        else:
            xGT_c = y.permute(0,2,3,4,1)
        mask = mask.reshape(mask.shape[0],mask.shape[1],mask.shape[2],mask.shape[3],1)
    else:
        assert False, "srcImg shape length has to be 4(2d) or 5(3d)"
  
    xGT_f = torch.fft(xGT_c,2, normalized=True)
    #xGT_f = torch.fft.fft2(xGT_c, norm='forward')
    subF = xGT_f * mask

    # if(len(y.shape)==4):
    #         subF = subF.permute(0,3,1,2)
    #     else:
    #         subF = subF.permute(0,4,1,2,3)

    return subF


def imgFromSubF_pytorch(subF,returnComplex=False):
    subIm = torch.ifft(subF,2, normalized=True) # sig_dim = 2
    if(len(subIm.shape)==4): 
        subIm = subIm.permute(0,3,1,2)
    else:
        subIm = subIm.permute(0,4,1,2,3)

    if(returnComplex):
        return subIm
    else:
        subIm = torch.sqrt(subIm[:,0:1]*subIm[:,0:1]+subIm[:,1:2]*subIm[:,1:2])
        return subIm





#=================================
# active contour loss
def acLoss(y_true, y_pred): 
    """
    y_true : (8,2,256,256)
    """

    """
    length 
    """
    x = y_pred[:,:,1:,:] - y_pred[:,:,:-1,:] # horizontal and vertical directions 
    y = y_pred[:,:,:,1:] - y_pred[:,:,:,:-1]

    delta_x = x[:,:,1:,:-2]**2
    delta_y = y[:,:,:-2,1:]**2
    delta_u = torch.abs(delta_x + delta_y) 

    lenth = torch.mean(torch.sqrt(delta_u + 0.00000001)) # equ.(11) in the paper

    """
    region term
    """

    C_1 = torch.ones((256, 256)).cuda()
    C_2 = torch.zeros((256, 256)).cuda()
   
    region_in = torch.abs(torch.mean(y_pred[:,0,:,:] * ((y_true[:,0,:,:] - C_1)**2) ) ) # equ.(12) in the paper
    region_out = torch.abs(torch.mean( (1-y_pred[:,0,:,:]) * ((y_true[:,0,:,:] - C_2)**2) )) # equ.(12) in the paper

    lambdaP = 1 # lambda parameter could be various.
    mu = 1 # mu parameter could be various.
    
    return lenth + lambdaP * (mu * region_in + region_out) 






#=================================



def im_normalize(im, eps=0.):
    mean = im.mean()
    std = im.std()
    return (im - mean) / (std + eps)

def im_normalize_re(im, eps=0.):
    mean = im.mean()
    std = im.std()
    return (im - mean) / (std + eps), mean, std



def getLosses(file1):
    losses = []
    with open(file1,'r') as f:
        lines = f.readlines()
        for line in lines:
            if "loss" in line:
                val = float(line.split(':')[3].strip(''))
                losses.append(val)
        return losses 

def getPsnr_SSIM(file1,mode='after'):
    psnr = []
    ssim =[]
    with open(file1,'r') as f:
        lines = f.readlines()
        for line in lines:
            if "psnr" in line:
                splits = line.split('=')
                tmp_psnr = splits[1].strip(' ssim').split('|')
                tmp_ssim = splits[2].split('|')
                
                tmp_psnr[1] = tmp_psnr[1].strip(',') 
                tmp_ssim[1] = tmp_ssim[1].strip(', loss ') 
                
                idx = 0 if(mode=='before') else 1
                psnr.append(float(tmp_psnr[idx]))
                ssim.append(float(tmp_ssim[idx]))
                
        return psnr, ssim


def plotFromFile(labels, folders, ifloss=0, ifpsnr=0, ifssim=0):
    "suppot train loss, psnr and ssim"

    colors = ['r','b','g','c','y']
    num1 = len(folders) # number of line
    num2 = ifloss + ifpsnr + ifssim # number of subplot 

    with plt.style.context('bmh'):
        font = {"color": "darkred", "size": 13, "family" : "serif"}
    fig, axs = plt.subplots(1, num2, figsize=(10, 5))
    pos = 0
    # the first subplot : loss if any 
    if ifloss:
        print("plot train loss")
        for label, folder,color in zip(labels,folders,colors): # for each model
            filename = os.path.join(folder,'trainLossLog.txt')
            yy = getLosses(filename)
            xx = list(range(len(yy)))
            if num2 == 1:
                plt.plot(xx, yy, label=label, rasterized=True, color=color, linewidth=1.5)
            else:
                axs[pos].plot(xx, yy, label=label, rasterized=True, color=color, linewidth=1.5)

        if num2 == 1:
            plt.ylim([0.0, 0.002])
            plt.xlabel('Number of backprops')
            plt.ylabel('loss')
            plt.title('train loss')
            plt.legend(labels = labels, loc="lower right")
            plt.show()

        else:
            axs[pos].set_ylim([0.0, 0.002])
            axs[pos].set_xlabel('Number of backprops')
            axs[pos].set_ylabel('loss')
            axs[pos].set_title('train loss')
            axs[pos].legend(loc="lower right")
            axs[pos].tick_params(axis='both')
            pos += 1

    # the second subplot : psnr if any 
    if ifpsnr:
        print("plot test psnr")
        for label, folder,color in zip(labels,folders,colors):
            filename = os.path.join(folder,'generalLog.txt')
            psnr, ssim = getPsnr_SSIM(filename)
            xx = list(range(len(psnr)))
            axs[pos].plot(xx, psnr, label=label, rasterized=True, 
                    color=color, linewidth=1.5)
         
        axs[pos].set_ylim([32, 36])
        axs[pos].set_xlabel('epoch')
        axs[pos].set_ylabel('psnr')
        axs[pos].set_title('test psnr')
        axs[pos].legend(loc="lower right")
        axs[pos].tick_params(axis='both')
        pos += 1

    if ifssim:
        print("plot test ssim")
        for label, folder,color in zip(labels,folders,colors):
            filename = os.path.join(folder,'generalLog.txt')
            _ , ssim = getPsnr_SSIM(filename)
            xx = list(range(len(ssim)))
            axs[pos].plot(xx, ssim, label=label, rasterized=True, 
                    color=color, linewidth=1.5)
        
        axs[pos].set_ylim([0.920, 0.975])
        axs[pos].set_xlabel('epoch')
        axs[pos].set_ylabel('ssim')
        axs[pos].set_title('test ssim')
        axs[pos].legend(loc="lower right")
        axs[pos].tick_params(axis='both')

    plt.tight_layout()
    root = '/home/hanhui/CSMRI_0325/result/sampling_rate_15/'
    fig.savefig(os.path.join(root,"result.png"))


if __name__ == '__main__':

    labels = ['default','resgroup']
    folders = [ 

            '/home/hanhui/CSMRI_0325/result/sampling_rate_15/CN_dOri_c5_complex_tr_trick2_1000epoch/log' ,
            '/home/hanhui/CSMRI_0325/result/sampling_rate_15/CN_Dense_resgroup/log' ,

            ]

    plotFromFile(labels, folders,ifloss=1, ifpsnr=1, ifssim=1)
    print('done')


