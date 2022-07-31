"sample code for fastmri training and evaluating"

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
import time
import cv2
import logging
from tqdm import tqdm
import pathlib
import argparse
import shutil
import random
import numpy as np
import torch
from torch.nn import functional as F
from network import getNet, getLoss, getOptimizer, Get_gradient
from util import paramNumber 
from PIL import Image
import matplotlib.pyplot as plt
from dataloader import getDataloader, dataFormat, handle_output 
from model_test import test_save_result_per_slice, test_save_result_per_volume 
import warnings
warnings.filterwarnings("ignore")
import pdb


def create_logger(args, mode):
    
    if not os.path.exists(args.exp_dir):
        os.mkdir(args.exp_dir)
    filename = os.path.join(args.exp_dir, "train.log")

    logging.basicConfig(filename=filename,
                        format='%(asctime)s %(message)s', 
                        filemode=mode,
                        level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger


def train_epoch(args, epoch, model, data_loader, optimizer, logger):
    
    model.train()
    avg_loss = 0.
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(data_loader)

    for iter, data in enumerate(data_loader):
    
        optimizer.zero_grad()

        # input: (B, C, H, W)
        # target : (B, H, W)
        # subF: (B, H, W, 2)
        # mask_var: (B, 1, H, W, 1)
        
        zf = data['zf']
        gt = data['gt']
        subF = data['subF']
        mask_var = data['mask']

        assert subF.shape[-1] == 2, "last dimension of input kspace should be 2"

        zf = zf.to(args.device, dtype=torch.float)
        gt = gt.to(args.device, dtype=torch.float)
        subF = subF.to(args.device, dtype=torch.float)
        mask_var = mask_var.to(args.device,dtype=torch.uint8)

        #=================
        # predict
        #=================
        if 'edge' in args.dataMode: # complex_edge, train with edge model
            assert 'gt_edge' in data.keys(),"Training dataloader do not have gt_edge"
            gt_edge = data['gt_edge'].to(args.device, dtype=torch.float)
        
        if args.use_sens_map:
            sens_map = data['sens_map'].to(args.device, dtype=torch.float)            
            output = model(zf, subF, mask_var, sens_map)
        else:
            output = model(zf, subF, mask_var)
  

        #=================
        # loss
        #=================
        nn = 1
        if not isinstance(output, list): 
            output = dataFormat(output)
            loss = F.l1_loss(output, gt)
        else: # list of outputs
            nn = len(output)
            loss = 0.
            for ii, ele in enumerate(output):
                ele= dataFormat(ele)
                # edge model
                if 'edge' in args.dataMode:
                    if ii < len(output)-1:
                        assert len(ele.shape)==3, 'invalid edge output'
                        loss += F.l1_loss(ele, gt_edge)
                    else:
                        loss += F.l1_loss(ele, gt)
                # not edge model
                else:
                    loss += F.l1_loss(ele, gt)

        loss.backward()
        optimizer.step()
        avg_loss = (0.99 * avg_loss + 0.01 * loss.item() if iter > 0 else loss.item())/nn
        
        if iter % args.report_interval == 0:
            logger.debug(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} Avg Loss = {avg_loss:.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
        start_iter = time.perf_counter()
        
    return avg_loss, time.perf_counter() - start_epoch




def save_model(args, exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_dev_loss': best_dev_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')



def load_model(args, checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    model = build_model(args)
    if args.data_parallel:
        model = torch.nn.DataParallel(model)

    model.load_state_dict(checkpoint['model'])

    optimizer = build_optim(args, model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint, model, optimizer



def build_model(args):
    model = getNet(args.netType).to(args.device)
    return model


def build_optim(args, params):
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay = 1e-7) 
    return optimizer



def visualize(args, model, data_loader):
    """
    visualize  
    """
    print("visualizing")
    model.eval()

    with torch.no_grad():
        for iter, data in tqdm(enumerate(data_loader)):
            if iter % args.jump == 0: # every 10 iter
                
                #===========================   
                # prepare data
                #===========================   

                fname = data['fname']
                slice = data['slice_id']

                for idx in range(len(fname)):
                    select = "{}-{}".format(fname[idx].split('.')[0], slice[idx])
                    #if select not in ["e15197s3_P53760-133", "e15197s3_P53760-127", "e14531s6_P68096-205", "e14531s6_P68096-127", "e16673s13_P31744-205"]:
                        #continue

                input = data['zf'].to(args.device, dtype=torch.float)
                gt = data['gt'].to(args.device, dtype=torch.float)
                subF = data['subF'].to(args.device, dtype=torch.float)
                mask_var = data['mask'].to(args.device, dtype=torch.uint8)
                maxval = data['maxval'].to(args.device, dtype=torch.float)
                mean = data['mean'].to(args.device, dtype=torch.float)
                std = data['std'].to(args.device, dtype=torch.float)
                
                if args.use_sens_map:
                    sens_map = data['sens_map'].to(args.device, dtype=torch.float)            
 
                #===========================   
                # predict              
                #===========================   
                if args.dev == 1:
                    outputs = input
                else:
                    if args.use_sens_map:
                        outputs = model(input, subF, mask_var, sens_map)
                    else:
                        outputs = model(input, subF, mask_var)
                
                if type(outputs) == list:
                    pred = outputs[-1]
                else:
                    pred = outputs
                
                #===========================   
                # store edge outputs
                #===========================   

                edges = [] #(e1, e2, e3, gt_edge)
                if 'edge' in args.dataMode: # edge model
                    assert type(outputs) is list, "outputs should at least have 2 elements for edge models"
                    for ii in range(len(outputs)-1):
                        tmp = dataFormat(outputs[ii]).detach().cpu().data.numpy()
                        edges.append(tmp)
                    edges.append(data['gt_edge'])

                N_edge = len(edges)

                #===========================   
                # normalize back data
                #===========================   

                if args.dataName == 'fastmri':
                    if 'complex' in args.dataMode: 
                        input = dataFormat(input) /1e6
                        pred = dataFormat(pred) / 1e6
                        gt = gt/1e6

                    elif args.dataMode == 'real':
                        std1 = std.view(-1,1,1)
                        mean1 = mean.view(-1,1,1)
                        input = dataFormat(input) * std1 + mean1
                        pred  =  dataFormat(pred) * std1 +mean1
                        gt = gt* std + mean

                elif args.dataName == 'cc359':

                    if args.challenge == 'singlecoil':
                        input = dataFormat(input) * 1e5
                        pred = dataFormat(pred) * 1e5 
                        gt = gt * 1e5

                    else: #multicoil
                        input = dataFormat(input) * maxval.view(-1,1,1)
                        pred = dataFormat(pred) * maxval.view(-1,1,1) 
                        gt = gt * maxval.view(-1,1,1)

                else:
                    raise NotImplementedError('Please provide correct dataset name: fastmri or cc359')
                
                #===========================   
                # transform into numpy
                #===========================   
                
                gt_np = gt.detach().cpu().data.numpy()
                pred_np = pred.detach().cpu().data.numpy()
                input_np = input.detach().cpu().data.numpy()
                mask_np = mask_var.detach().cpu().data.numpy()  #(B,1,W,1)
                res_np = 5 * (np.abs(gt_np - pred_np) / gt_np.max())
                zim_res_np = 5 * (np.abs(gt_np- input_np) / gt_np.max())

                #===========================   
                # plot
                #===========================   
                
                N = len(gt_np)
                for idx in range(N):
                    select = "{}-{}".format(fname[idx].split('.')[0], slice[idx])
                    #if select not in ["e15197s3_P53760-133", "e15197s3_P53760-127", "e14531s6_P68096-205", "e14531s6_P68096-127", "e16673s13_P31744-205"]:
                        #continue

                    plt.imsave(os.path.join(args.im_root, '{}-{}_gt.png'.format(fname[idx].split('.')[0], slice[idx])), gt_np[idx], cmap='gray' )
                    plt.imsave(os.path.join(args.im_root, '{}-{}_pred.png'.format(fname[idx].split('.')[0], slice[idx])), pred_np[idx], cmap='gray' )
                    plt.imsave(os.path.join(args.im_root, '{}-{}_zf.png'.format(fname[idx].split('.')[0], slice[idx])), input_np[idx], cmap='gray' )
                    plt.imsave(os.path.join(args.im_root, '{}-{}_res.png'.format(fname[idx].split('.')[0], slice[idx])), res_np[idx], cmap='viridis')
                    plt.imsave(os.path.join(args.im_root, '{}-{}_zim_res.png'.format(fname[idx].split('.')[0], slice[idx])), zim_res_np[idx], cmap='viridis')
                    
                    # plot mask
                    if len(mask_np.shape) == 5:
                        plt.imsave(os.path.join(args.im_root, '{}-{}_mask.png'.format(fname[idx].split('.')[0], slice[idx])), mask_np[idx,0,:,:,0], cmap='gray')
                    else:
                        plt.imsave(os.path.join(args.im_root, '{}-{}_mask.png'.format(fname[idx].split('.')[0], slice[idx])), mask_np[idx,:,:,0], cmap='gray')
                        
                    # plot edge
                    if N_edge > 0:
                        for kk in range(N_edge):
                            if kk < N_edge - 1:
                                plt.imsave(os.path.join(args.im_root, '{}-{}_pred_edge{}.png'.format(fname[idx].split('.')[0], slice[idx], kk)), edges[kk][idx], cmap='gray' )
                            else:
                                plt.imsave(os.path.join(args.im_root, '{}-{}_gt_edge.png'.format(fname[idx].split('.')[0], slice[idx])), edges[kk][idx], cmap='gray' )




def main(args):

    # create folder
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    args.im_root = os.path.join(args.exp_dir, 'images')
    if not os.path.exists(args.im_root):
        os.mkdir(args.im_root)

    # =======================================
    # load model
    # =======================================
    if (args.resume == 1) or (args.is_evaluate == 1): 

        if args.resume == 1: # safeguard
            args.is_evaluate = 0

        assert args.dev == 0, "args.dev must be 0 when resume or test mode"
        logger = create_logger(args, 'a')
        logger.debug("loading model. Resume: {}, Evaluate: {}".format(args.resume, args.is_evaluate))

        if 'wasnet' in args.netType or 'edge' in args.netType:
            checkpoint, model, optimizer = load_model(args, os.path.join(args.exp_dir, 'best_model.pt'))
        else:
            checkpoint, model, optimizer = load_model(args, os.path.join(args.exp_dir, 'model.pt'))

        best_dev_loss = checkpoint['best_dev_loss']
        start_epoch = checkpoint['epoch']
        assert start_epoch <= args.num_epochs, "model already finish training, do not resume"
        del checkpoint

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)

    else:
        logger = create_logger(args, 'w')
        if not args.dev:
            model = build_model(args)
            if args.data_parallel:
                model = torch.nn.DataParallel(model)
            optimizer = build_optim(args, model.parameters())
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)
            param = paramNumber(model)
            logger.debug(model)
            logger.debug("model parameters : {}".format(param))


        else:
            model = None

        best_dev_loss = 1e9
        start_epoch = 0
        logger.debug(args)


    # =======================================
    # load dataloader 
    # =======================================
    train_loader, dev_loader = getDataloader(args.dataName, args.dataMode, args.batchSize, [args.center_fractions], [args.accer], args.resolution, args.train_root, args.valid_root, args.sample_rate, args.challenge, args.use_sens_map)


    # =======================================
    # training
    # =======================================
    if (not args.is_evaluate) and (not args.dev):
        logger.debug("start training")
        for epoch in range(start_epoch, args.num_epochs):
            scheduler.step(epoch)

            train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, logger)
            dev_loss, dev_rmse, dev_psnr, dev_ssim ,dev_time = test_save_result_per_volume(model, dev_loader, args)

            is_new_best = dev_loss < best_dev_loss
            best_dev_loss = min(best_dev_loss, dev_loss)
            save_model(args, args.exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best)
            logger.debug(
                f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
                f'DevLoss = {dev_loss:.4g} DevRMSE = {dev_rmse:.4g} DevPSNR = {dev_psnr:.4g} DevSSIM = {dev_ssim:.4g} TrainTime = {train_time:.4f}s DevTime = {dev_time:.4f}s',
            )

    # ====================================
    # evaluating mode
    # ====================================
    else:
        logger.debug("Start evaluating (without training)")
        #dev_loss, dev_rmse, dev_psnr, dev_ssim ,dev_time = test_save_result_per_volume(model, dev_loader, args)
        #logger.debug(f'Epoch = [{start_epoch:4d}] DevLoss = {dev_loss:.4g} DevRMSE = {dev_rmse:.4g} DevPSNR = {dev_psnr:.4g} DevSSIM = {dev_ssim:.4g} DevTime = {dev_time:.4f}s')
        if args.dev != 1:
            visualize(args , model, dev_loader)
        



def create_arg_parser_fastmri():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=43) 
    parser.add_argument('--dev', type=int, default=0, help='for development test') 
    parser.add_argument('--train_root', type=str, help='path to store the train data') 
    parser.add_argument('--valid_root', type=str, help='path to store the train data')
    parser.add_argument('--dataName', type=str, help='name of the dataset. fastmri/cc359', default='fastmri')
    parser.add_argument('--challenge', type=str, help='challenge. singlecoil/multicoil', default='singlecoil')
    parser.add_argument('--dataMode', type=str, help="data mode for input data, real/complex", default='complex') 
    parser.add_argument('--resolution', type=int, help="resolution of data. 320 for fastmri or 256 for cc359", default=320)
    parser.add_argument('--netType', type=str) 
    parser.add_argument('--exp-dir', type=pathlib.Path, default='./results/', help='Path to store the results')
    parser.add_argument('--num-epochs', type=int, default=80, help='Number of training epochs')
    parser.add_argument('--jump', type=int, default=10, help='how many jump to plot images')
    parser.add_argument('--sample_rate', type=float, help="Sample rate", default=1.)
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--lr-step-size', type=int, default=20, help='Period of learning rate decay')
    parser.add_argument('--batchSize', type=int, default=8) 
    parser.add_argument('--accer', type=int, default=4) 
    parser.add_argument('--center_fractions', type=float, default=0.08) 
    parser.add_argument('--report_interval', type=int, default=100) 
    parser.add_argument('--display_interval', type=int, default=10) 
    parser.add_argument('--resume', type=int, default=0, help="resume training") 
    parser.add_argument('--is_evaluate', type=int, default=0, help="train or test") 
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='Multiplicative factor of learning rate decay')
    parser.add_argument('--weight-decay', type=float, default=0.,
                        help='Strength of weight decay regularization')
    parser.add_argument('--data-parallel', action='store_true',default=True,
                        help='If set, use multiple GPUs using data parallelism')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Which device to train on. Set to "cuda" to use the GPU')
    parser.add_argument('--server', type=str, default='cluster', help='Which device to train on.')
    parser.add_argument('--use_sens_map', type=int, default=0, help='Whether to calculate sensitivity map.')

    return parser.parse_args()


if __name__ == '__main__':
    
    args = create_arg_parser_fastmri()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    if args.server == 'cluster':
        if args.challenge == 'singlecoil':
            if args.dataName == 'fastmri':
                args.train_root = '/project/Math/hanhui/opendata/fastmri_knee_singlecoil_dataset/singlecoil_train/' 
                args.valid_root = '/project/Math/hanhui/opendata/fastmri_knee_singlecoil_dataset/singlecoil_val/' 

            elif args.dataName == 'cc359':
                args.train_root = '/project/Math/hanhui/opendata/CC-359_single_coil/Train/' 
                args.valid_root = '/project/Math/hanhui/opendata/CC-359_single_coil/Val/' 

        elif args.challenge == 'multicoil':
            if args.dataName == 'fastmri':
                args.train_root = '/project/Math/hanhui/opendata/fastmri_knee_multicoil_dataset/multicoil_train/' 
                args.valid_root = '/project/Math/hanhui/opendata/fastmri_knee_multicoil_dataset/multicoil_val/' 
            
            elif args.dataName == 'cc359':
                args.train_root = '/project/Math/hanhui/opendata/CC-359_multi_coil/Train/' 
                args.valid_root = '/project/Math/hanhui/opendata/CC-359_multi_coil/Val/' 


    elif args.server == 'ai':
        if args.challenge == 'singlecoil':
            if args.dataName == 'fastmri':
                args.train_root = '/home/ET/hanhui/opendata/fastmri_knee_singlecoil_dataset/singlecoil_train/' 
                args.valid_root = '/home/ET/hanhui/opendata/fastmri_knee_singlecoil_dataset/singlecoil_val/' 

            elif args.dataName == 'cc359':
                args.train_root = '/home/ET/hanhui/opendata/CC-359_single_coil/Train/' 
                args.valid_root = '/home/ET/hanhui/opendata/CC-359_single_coil/Val/' 

        elif args.challenge == 'multicoil':
            if args.dataName == 'fastmri':
                args.train_root = '/home/ET/hanhui/opendata/fastmri_knee_multicoil_dataset/multicoil_train_pd/' 
                args.valid_root = '/home/ET/hanhui/opendata/fastmri_knee_multicoil_dataset/multicoil_val_pd/' 

            elif args.dataName == 'cc359':
                args.train_root = '/home/ET/hanhui/opendata/CC-359_multi_coil/Train/' 
                args.valid_root = '/home/ET/hanhui/opendata/CC-359_multi_coil/Val/' 


    if args.accer == 4:
        args.center_fractions = 0.08
    elif args.accer == 8:
        args.center_fractions = 0.04

    main(args)
    

