import os
import cv2
import numpy as np
import time
import math
import h5py
import matplotlib
matplotlib.use('Agg')
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch
from matplotlib import pyplot as plt
import pdb
from scipy import ndimage



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


def c2(r, r0, t):
    return math.exp(-((r-r0)/float(t))**2)



def Get_sobel(gt, scale, delta):

    ddepth = cv2.CV_16S
    grad_x = cv2.Sobel(target*1e6, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(target*1e6, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return grad


def median_filter_ex():

    # patient id and frame name 
    fid = 1002382
    frame = 30
    scale = 1
    delta = 0

    fname = '/home/ET/hanhui/opendata/fastmri_knee_singlecoil_dataset/singlecoil_val/file{}.h5'.format(fid)
    root = './images/edge_explore_{}_scale{}_delta{}'.format(fid, scale, delta)
    if not os.path.exists(root):
        os.makedirs(root)

    # read data
    with h5py.File(fname, 'r') as data:
        kspace = h5py.File(fname,'r')['kspace']
        target = data['reconstruction_esc'][frame] #(320,320)

    # apply meadian filter
    target_med = ndimage.median_filter(target, 3) 

    gt_sobel = Get_sobel(target,scale, delta)
    gt_sobel_med = Get_sobel(target_med, scale, delta)


    plot_and_save(target, os.path.join(root,'frame{}_gt.png'.format(frame)))
    plot_and_save(gt_sobel, os.path.join(root, 'frame{}_gt_sobel.png'.format(frame)))
    plot_and_save(target_med, os.path.join(root,'frame{}_gt_med.png'.format(frame)))
    plot_and_save(gt_sobel_med, os.path.join(root, 'frame{}_gt_sobel_med.png'.format(frame)))



def meadian_filter_susan_op_ex():

    # too slow
    # patient id and frame name 

    fid = 1002382
    frame = 30

    fname = '/home/ET/hanhui/opendata/fastmri_knee_singlecoil_dataset/singlecoil_val/file{}.h5'.format(fid)
    root = './images/edge_explore_{}_susan'.format(fid)
    if not os.path.exists(root):
        os.makedirs(root)

    # read data
    with h5py.File(fname, 'r') as data:
        kspace = h5py.File(fname,'r')['kspace']
        target = data['reconstruction_esc'][frame] #(320,320)

    pixels = target.copy()
    pixels = cv2.normalize(target*1e6,  pixels, 0, 255, cv2.NORM_MINMAX)
    t = 10
    r = 3.4
    md = int(math.ceil(r*2))
    m = [[0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0]]
    
    count = 37
    g = 3.0*count/4.0
    n = [[0 for x in range(320)] for x in range(320)]

    t1 = time.time()
    for x in range(320):
        for y in range(320):
            for xr in range(md):
                    for yr in range(md):
                        xx = int(x-r+xr)
                        yy = int(y-r+yr)
                        if m[xr][yr] == 1 and xx>=0 and xx<320 and yy>=0 and yy<320:
                            cdif = c2(pixels[xx, yy], pixels[x,y], t)
                            n[x][y] += cdif
    
    for x in range(320):
        for y in range(320):
            if n[x][y] < g:
                n[x][y] = g - n[x][y]
            else:
                n[x][y] = 0

    
    for x in range(320):
        for y in range(320):
            if n[x][y] != 0:
                pixels[x,y] = int(n[x][y] * 255/g)
            else:
                pixels[x,y] = 0

    print("finish time {}".format(time.time() - t1))


    plot_and_save(target, os.path.join(root,'frame{}_gt.png'.format(frame)))
    plot_and_save(pixels, os.path.join(root, 'frame{}_gt_susan.png'.format(frame)))








if __name__ == '__main__':

    #fname = '/home/ET/hanhui/opendata/fastmri_knee_singlecoil_dataset/singlecoil_train/file1001475.h5' 

    meadian_filter_susan_op_ex()



