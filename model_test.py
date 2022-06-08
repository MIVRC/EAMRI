import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import defaultdict
import numpy as np
from fastmri import evaluate
from dataloader import handle_output, fastmri_format
import h5py
from network import Get_gradient
from tqdm import tqdm
from skimage import measure
import pdb


def mse(gt, pred):
    """ Compute Mean Squared Error (MSE) """
    return np.mean((gt - pred) ** 2)


def nmse(pred, gt):
    """ Compute Normalized Mean Squared Error (NMSE) """
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2


def psnr(pred,gt, data_range):
    """ Compute Peak Signal to Noise Ratio metric (PSNR) """
    assert pred.shape[0] == 1, "calculate psnr, input batch size should be 1"
    return measure.compare_psnr(gt, pred, data_range = data_range)


def ssim(pred,gt,data_range):
    """ Compute Structural Similarity Index Metric (SSIM). """
    assert gt.shape[0] == 1, "calculate psnr, input batch size should be 1"
    return measure.compare_ssim(gt[0], pred[0], data_range = data_range )


def test_save_result_per_slice(model, data_loader, args):
    
    """
    calculate metrics per slide
    """
        
    print("evaluating validation data")
    model.eval()
    test_logs =[]
    total_loss = []
    total_psnr = []
    total_ssim = []
    start = time.perf_counter()
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(data_loader)):
            input, target, subF, mask_val, mean, std, maxval, fname, slice = batch
            input = input.to(args.device)
            target = target.to(args.device)
            subF = subF.to(args.device)
            mask_val = mask_val.to(args.device)
            output = model(input, subF, mask_val)
            output = handle_output(output, 'test')
            mean = mean.unsqueeze(1).unsqueeze(2).to(args.device)
            std = std.unsqueeze(1).unsqueeze(2).to(args.device)
            
            #input = fastmri_format(input) * std
            if args.dataMode == 'complex':
                output =  fastmri_format(output) /1e6
                target = target /1e6

            elif args.dataMode == 'real':
                output =  fastmri_format(output) * std + mean
                target = target * std + mean

            output = output.detach().cpu().numpy()             
            target = target.detach().cpu().numpy()
            maxval = maxval.numpy()
            
            tmp_psnr = psnr(output, target, maxval)
            tmp_ssim = ssim(output, target, maxval)
            tmp_nmse = nmse(output, target)

            total_loss.append(tmp_nmse)
            total_psnr.append(tmp_psnr)
            total_ssim.append(tmp_ssim)

    return np.mean(total_loss), np.mean(total_psnr), np.mean(total_ssim), time.perf_counter()-start 




def test_save_result_per_volume(model, data_loader, args):
    """
    calculate metrics per volume
    """ 
    model.eval()
    test_logs =[]
    start = time.perf_counter()
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(data_loader)):
            input, target, subF, mask_val, mean, std, maxval, fname, slice = batch

            input = input.to(args.device, dtype=torch.float)
            target = target.to(args.device, dtype=torch.float)
            subF = subF.to(args.device, dtype=torch.float)
            mask_val = mask_val.to(args.device, dtype=torch.float)

            output = model(input, subF, mask_val)
            output = handle_output(output, 'test')
            mean = mean.unsqueeze(1).unsqueeze(2).to(args.device, dtype=torch.float)
            std = std.unsqueeze(1).unsqueeze(2).to(args.device, dtype=torch.float)
           
            if args.dataName == 'fastmri':

                if args.dataMode == 'complex':
                    input = fastmri_format(input) /1e6
                    if args.dev == 1:
                        output = input
                    else:
                        output =  fastmri_format(output) /1e6
                    target = target /1e6

                elif args.dataMode == 'real':
                    input = input * std + mean
                    output =  fastmri_format(output) * std + mean
                    target = target * std + mean

            elif args.dataName == 'cc359':
                if args.dataMode == 'complex':
                    input = fastmri_format(input) * 1e5
                    if args.dev == 1:
                        output = input 
                    else:
                        output =  fastmri_format(output) * 1e5
                    target = target * 1e5
            else:
                raise NotImplementedError('Please provide correct dataset name: fastmri or cc359')

            test_loss = F.l1_loss(output, target)
            test_logs.append({
                'fname': fname,
                'slice': slice,
                'maxval': maxval,
                'output': output.cpu().detach().numpy(),
                'target': target.cpu().detach().numpy(),
                'input': input.cpu().detach().numpy(),
                'loss': test_loss.cpu().detach().numpy(),
            })


        losses = []
        outputs = defaultdict(list)
        targets = defaultdict(list)
        inputs = defaultdict(list)
        maxvals = defaultdict(list) # store max val of volume

        for log in test_logs:
            losses.append(log['loss'])
            for i, (fname, slice) in enumerate(zip(log['fname'], log['slice'])):
                outputs[fname].append((slice, log['output'][i]))
                targets[fname].append((slice, log['target'][i]))
                inputs[fname].append((slice, log['input'][i]))
                maxvals[fname].append((slice, log['maxval'][i]))
                
        
        metrics = dict(val_loss=losses, nmse=[], rmse=[], ssim=[], psnr=[])
        outputs_save = defaultdict(list)
        targets_save = defaultdict(list)

        '''
        for fname in outputs:
            outputs_save[fname] = np.stack([out for _, out in sorted(outputs[fname])])
            targets_save[fname] = np.stack([tgt for _, tgt in sorted(targets[fname])])
            inputs[fname] = np.stack([tgt for _, tgt in sorted(inputs[fname])])
        '''

        for fname in outputs:
            output = np.stack([out for _, out in sorted(outputs[fname])])
            target = np.stack([tgt for _, tgt in sorted(targets[fname])])
            maxval_volume = np.stack([temp for _, temp in sorted(maxvals[fname])])[0]

            if args.dataName == 'cc359':
                maxval_volume = None

            metrics['nmse'].append(evaluate.nmse(target, output))
            metrics['rmse'].append(evaluate.rmse(target, output))
            metrics['ssim'].append(evaluate.ssim(target, output, maxval_volume))
            metrics['psnr'].append(evaluate.psnr(target, output, maxval_volume))

        metrics = {metric: np.mean(values) for metric, values in metrics.items()}
        torch.cuda.empty_cache()
        return metrics['nmse'], metrics['rmse'], metrics['psnr'], metrics['ssim'], time.perf_counter()-start 


        #save_reconstructions(outputs_save, args.save_dir /'reconstructions')
        #save_reconstructions(targets_save, args.save_dir /'gt')
        #save_reconstructions(inputs, args.save_dir /'input')



def test_save_result_per_volume_edge(model, data_loader, args):
    """
    calculate metrics per volume for edge model
    """ 
    model.eval()
    test_logs =[]
    start = time.perf_counter()
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(data_loader)):
            
            input, zim_edge, target, _, subF, mask_val, mean, std, maxval, fname, slice = batch

            #zim_edge = torch.unsqueeze(zim_edge, 1).contiguous() # dev

            input = input.to(args.device, dtype=torch.float)
            zim_edge = zim_edge.to(args.device, dtype=torch.float)
            target = target.to(args.device, dtype=torch.float)
            subF = subF.to(args.device, dtype=torch.float)
            mask_val = mask_val.to(args.device, dtype=torch.float)

            output = model(input, zim_edge, subF, mask_val)
            output = handle_output(output, 'test')

            mean = mean.unsqueeze(1).unsqueeze(2).to(args.device)
            std = std.unsqueeze(1).unsqueeze(2).to(args.device)
            
            if args.dataName == 'fastmri':
                if args.dataMode == 'complex' or args.dataMode == 'complex_edge':
                    input = fastmri_format(input) /1e6
                    output =  fastmri_format(output) /1e6
                    target = target /1e6
                elif args.dataMode == 'real':
                    input = input * std + mean
                    output =  fastmri_format(output) * std + mean
                    target = target * std + mean

            elif args.dataName == 'cc359':
                if args.dataMode == 'complex' or args.dataMode == 'complex_edge':
                    input = fastmri_format(input) * 1e5
                    if args.dev == 1:
                        output = input 
                    else:
                        output =  fastmri_format(output) * 1e5
                    target = target * 1e5
                else:
                    raise NotImplementedError("Do not have real dataMode for cc359 dataset!")

            test_loss = F.l1_loss(output, target)
            test_logs.append({
                'fname': fname,
                'slice': slice,
                'maxval': maxval,
                'output': output.cpu().detach().numpy(),
                'target': target.cpu().detach().numpy(),
                'input': input.cpu().detach().numpy(),
                'loss': test_loss.cpu().detach().numpy(),
            })


        losses = []
        outputs = defaultdict(list)
        targets = defaultdict(list)
        inputs = defaultdict(list)
        maxvals = defaultdict(list) # store max val of volume

        for log in test_logs:
            losses.append(log['loss'])
            for i, (fname, slice) in enumerate(zip(log['fname'], log['slice'])):
                outputs[fname].append((slice, log['output'][i]))
                targets[fname].append((slice, log['target'][i]))
                inputs[fname].append((slice, log['input'][i]))
                maxvals[fname].append((slice, log['maxval'][i]))
                
        
        metrics = dict(val_loss=losses, nmse=[], ssim=[], psnr=[])
        outputs_save = defaultdict(list)
        targets_save = defaultdict(list)

        for fname in outputs:
            output = np.stack([out for _, out in sorted(outputs[fname])])
            target = np.stack([tgt for _, tgt in sorted(targets[fname])])
            maxval_volume = np.stack([temp for _, temp in sorted(maxvals[fname])])[0]

            if args.dataName == 'cc359':
                maxval_volume = None
            
            metrics['nmse'].append(evaluate.nmse(target, output))
            metrics['ssim'].append(evaluate.ssim(target, output, maxval_volume))
            metrics['psnr'].append(evaluate.psnr(target, output, maxval_volume))

        metrics = {metric: np.mean(values) for metric, values in metrics.items()}
        torch.cuda.empty_cache()
        return metrics['nmse'], 0, metrics['psnr'], metrics['ssim'], time.perf_counter()-start 



def save_reconstructions(reconstructions, out_dir):
    """
    Saves the reconstructions from a model into h5 files that is appropriate for submission
    to the leaderboard.
    Args:
        reconstructions (dict[str, np.array]): A dictionary mapping input filenames to
            corresponding reconstructions (of shape num_slices x height x width).
        out_dir (pathlib.Path): Path to the output directory where the reconstructions
            should be saved.
    """
    out_dir.mkdir(exist_ok=True, parents=True)
    for fname, recons in reconstructions.items():
        with h5py.File(out_dir / fname, 'w') as f:
            f.create_dataset('reconstruction', data=recons)








