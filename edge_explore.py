import os
import cv2
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch
from matplotlib import pyplot as plt
import pdb



def plot_and_save(data, name):
    plt.imsave(name, data, cmap='gray') 


def im2tensor(data):

    temp = torch.from_numpy(data)
    temp1 = torch.zeros(1,2,320,320)
    temp1[0,0,:,:] = temp
    return temp1


class Get_gradient(nn.Module):
    def __init__(self):
        super(Get_gradient, self).__init__()
        kernel_v = [[0, -1, 0], 
                    [0, 0, 0], 
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0], 
                    [-1, 0, 1], 
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False)
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False)

    def forward(self, x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding = 1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding = 1)
        x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding = 1)
        x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding = 1)
        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2)) + 0. 
        x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2)) + 0. 
        x = torch.cat([x0, x1], dim=1)
        return x




def Get_sobel(gt):

    ddepth = cv2.CV_16S
    scale = 1
    delta = 0
    grad_x = cv2.Sobel(target*1e6, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(target*1e6, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return grad





if __name__ == '__main__':

    #fname = '/home/ET/hanhui/opendata/fastmri_knee_singlecoil_dataset/singlecoil_train/file1001475.h5' 

    fid = 1000356
    frame = 25

    fname = '/home/ET/hanhui/opendata/fastmri_knee_singlecoil_dataset/singlecoil_val/file{}.h5'.format(fid)
    root = './images/edge_explore_{}'.format(fid)
    if not os.path.exists(root):
        os.mkdir(root)

    edgeExtract = Get_gradient()

    # read data
    with h5py.File(fname, 'r') as data:
        kspace = h5py.File(fname,'r')['kspace']
        target = data['reconstruction_esc'][frame] #(320,320)

    
    target_tor = im2tensor(target) # (1,2,320,320)

    # soft edge
    gt_se = edgeExtract(target_tor) #(1,2,320,320)
    gt_se = gt_se[0,0,:,:].numpy()

    # sobel edge  
    gt_sobel = Get_sobel(target)

    plot_and_save(target, os.path.join(root,'frame{}_gt.png'.format(frame)))
    plot_and_save(gt_se, os.path.join(root,'frame{}_gt_se.png'.format(frame)))
    plot_and_save(gt_sobel, os.path.join(root, 'frame{}_gt_sobel.png'.format(frame)))

    '''
    # blur target and then save soft edge and sobel
    for kernel in [3,5,7,9]:
        for sigma in [0,0.5,1,2,4,6,9]: 
            gt_blur = cv2.GaussianBlur(target, (kernel,kernel), sigma) 
            gt_blur_tor = im2tensor(gt_blur)
            gt_blur_se = edgeExtract(gt_blur_tor)[0,0,:,:].numpy()
            gt_blur_sobel = Get_sobel(gt_blur)

            plot_and_save(gt_blur_se, os.path.join(root,'frame{}_gt_kernel{}sigma{}_blur_se.png'.format(frame,kernel,sigma)))
            plot_and_save(gt_blur_sobel, os.path.join(root, 'frame{}_gt_kernel{}sigma{}_blur_sobel.png'.format(frame, kernel, sigma)))

    '''




