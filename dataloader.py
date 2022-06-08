# dataloader for fastmri dataset
import os
import logging
import pathlib
import random
import shutil
import time
import h5py
import cv2
import pdb
import numpy as np
import torch
from util import Get_sobel
from torch.utils.data import DataLoader
from fastmri.data import transforms_simple as transforms
from torch.utils.data import Dataset
from cardiac_dataloader import SliceData_cardiac


class MaskFunc:
    """
    MaskFunc creates a sub-sampling mask of a given shape.
    The mask selects a subset of columns from the input k-space data. If the k-space data has N
    columns, the mask picks out:
        1. N_low_freqs = (N * center_fraction) columns in the center corresponding to
           low-frequencies
        2. The other columns are selected uniformly at random with a probability equal to:
           prob = (N / acceleration - N_low_freqs) / (N - N_low_freqs).
    This ensures that the expected number of columns selected is equal to (N / acceleration)
    It is possible to use multiple center_fractions and accelerations, in which case one possible
    (center_fraction, acceleration) is chosen uniformly at random each time the MaskFunc object is
    called.
    For example, if accelerations = [4, 8] and center_fractions = [0.08, 0.04], then there
    is a 50% probability that 4-fold acceleration with 8% center fraction is selected and a 50%
    probability that 8-fold acceleration with 4% center fraction is selected.
    """

    def __init__(self, center_fractions, accelerations):
        """
        Args:
            center_fractions (List[float]): Fraction of low-frequency columns to be retained.
                If multiple values are provided, then one of these numbers is chosen uniformly
                each time.
            accelerations (List[int]): Amount of under-sampling. This should have the same length
                as center_fractions. If multiple values are provided, then one of these is chosen
                uniformly each time. An acceleration of 4 retains 25% of the columns, but they may
                not be spaced evenly.
        """
        if len(center_fractions) != len(accelerations):
            raise ValueError('Number of center fractions should match number of accelerations')

        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.rng = np.random.RandomState()

    def __call__(self, shape, seed=None):
        """
        Args:
            shape (iterable[int]): The shape of the mask to be created. The shape should have
                at least 3 dimensions. Samples are drawn along the second last dimension.
            seed (int, optional): Seed for the random number generator. Setting the seed
                ensures the same mask is generated each time for the same shape.
        Returns:
            torch.Tensor: A mask of the specified shape.
        """
        if len(shape) < 3:
            raise ValueError('Shape should have 3 or more dimensions')

        self.rng.seed(seed)
        num_cols = shape[-2]

        choice = self.rng.randint(0, len(self.accelerations))
        center_fraction = self.center_fractions[choice]
        acceleration = self.accelerations[choice]

        # Create the mask
        num_low_freqs = int(round(num_cols * center_fraction))
        prob = (num_cols / acceleration - num_low_freqs) / (num_cols - num_low_freqs)
        mask = self.rng.uniform(size=num_cols) < prob
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad:pad + num_low_freqs] = True

        # Reshape the mask
        mask_shape = [1 for _ in shape]
        mask_shape[-2] = num_cols
        mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

        return mask



class SliceData_fastmri(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices for fastmri
    """

    def __init__(self, root, transform, challenge, sample_rate=1, skip_head=0):
        """
        Args:
            root (pathlib.Path): Path to the dataset.
            transform (callable): A callable object that pre-processes the raw data into
                appropriate form. The transform function should take 'kspace', 'target',
                'attributes', 'filename', and 'slice' as inputs. 'target' may be null
                for test data.
            challenge (str): "singlecoil" or "multicoil" depending on which challenge to use.
            sample_rate (float, optional): A float between 0 and 1. This controls what fraction of the volumes should be loaded.

            skip_head: whether skip the first few bad images
        """
        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')

        self.transform = transform
        self.recons_key = 'reconstruction_esc' if challenge == 'singlecoil' \
            else 'reconstruction_rss'

        self.examples = []
        files = list(pathlib.Path(root).iterdir())
        if sample_rate < 1:
            random.shuffle(files)
            num_files = round(len(files) * sample_rate)
            files = files[:num_files]

        for fname in sorted(files):
            kspace = h5py.File(fname, 'r')['kspace'] #(num_slices, height, width)
            num_slices = kspace.shape[0]
            if skip_head:
                begin = num_slices // 2 - 8
            else:
                begin = 0
            self.examples += [(fname, slice) for slice in range(begin, num_slices)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice = self.examples[i]
        with h5py.File(fname, 'r') as data:
            kspace = data['kspace'][slice]
            target = data[self.recons_key][slice] if self.recons_key in data else None
            return self.transform(kspace, target, data.attrs, fname.name, slice)




class SliceData_cc359(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices for brainMRI
    """

    def __init__(self, root, transform, sample_rate=1, skip_head=0):
        """
        Args:
            root (pathlib.Path): Path to the dataset.
            transform (callable): A callable object that pre-processes the raw data into
                appropriate form. The transform function should take 'kspace', 'target',
                'attributes', 'filename', and 'slice' as inputs. 'target' may be null
                for test data.
            challenge (str): "singlecoil" or "multicoil" depending on which challenge to use.
            sample_rate (float, optional): A float between 0 and 1. This controls what fraction
                of the volumes should be loaded.
        """
        self.transform = transform
        self.examples = []
        files = list(pathlib.Path(root).iterdir())
        if sample_rate < 1:
            random.shuffle(files)
            num_files = round(len(files) * sample_rate)
            files = files[:num_files]

        for fname in sorted(files):
            im = np.load(fname) #(num_slices, height, width, 2)
            num_slices = im.shape[0]
            if skip_head:
                begin = num_slices // 2 - 8
            else:
                begin = 0
            self.examples += [(fname, slice) for slice in range(begin, num_slices)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice = self.examples[i]
        im = np.load(fname)
        kspace = im[slice]
        maxval = 0
        return self.transform(kspace, maxval, fname.name, slice)




class DataTransform_real_fastmri:
    """
    Data Transformer for training fastmri single-coild real-valued image
    """

    def __init__(self, mask_func, resolution, which_challenge, use_seed=True):
        """
        Args:
            mask_func (common.subsample.MaskFunc): A function that can create a mask of
                appropriate shape.
            resolution (int): Resolution of the image.
            which_challenge (str): Either "singlecoil" or "multicoil" denoting the dataset.
            use_seed (bool): If true, this class computes a pseudo random number generator seed
                from the filename. This ensures that the same mask is used for all the slices of
                a given volume every time.
        """
        if which_challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')
        self.mask_func = mask_func
        self.resolution = resolution
        self.which_challenge = which_challenge
        self.use_seed = use_seed

    def __call__(self, kspace, target, attrs, fname, slice):
        """
        Args:
            kspace (numpy.array): Input k-space of shape (num_coils, rows, cols, 2) for multi-coil
                data or (rows, cols, 2) for single coil data.
            target (numpy.array): Target image
            attrs (dict): Acquisition related information stored in the HDF5 object.
            fname (str): File name
            slice (int): Serial number of the slice.
        Returns:
            (tuple): tuple containing:
                image (torch.Tensor): Zero-filled input image.
                target (torch.Tensor): Target image converted to a torch Tensor.
                mean (float): Mean value used for normalization.
                std (float): Standard deviation value used for normalization.
                norm (float): L2 norm of the entire volume.
        """

        seed = None if not self.use_seed else tuple(map(ord, fname))

        # target
        target = transforms.to_tensor(target)
        kspace = transforms.to_tensor(kspace)

        # get masked kspace
        masked_kspace, mask = transforms.apply_mask(kspace, self.mask_func, seed)

        # get zim
        zim = transforms.ifft2(masked_kspace)
        zim = transforms.complex_center_crop(zim, (self.resolution, self.resolution))
        zim = transforms.complex_abs(zim)

        if self.which_challenge == 'multicoil':
            zim = transforms.root_sum_of_squares(zim)

        zim, mean, std = transforms.normalize_instance(zim, eps=1e-11)
        zim= zim.clamp(-6, 6)

        # normalize target
        target = transforms.normalize(target, mean, std, eps=1e-11)
        target = target.clamp(-6, 6)

        return zim.unsqueeze(0), target, zim, zim, mean, std, attrs['max'].astype(np.float32), fname, slice



class DataTransform_complex_fastmri:
    """
    Data Transformer for training fastmri complex-valued image
    """

    def __init__(self, mask_func, resolution, which_challenge, use_seed=True):
        """
        Args:
            mask_func (common.subsample.MaskFunc): A function that can create a mask of
                appropriate shape.
            resolution (int): Resolution of the image.
            which_challenge (str): Either "singlecoil" or "multicoil" denoting the dataset.
            use_seed (bool): If true, this class computes a pseudo random number generator seed
                from the filename. This ensures that the same mask is used for all the slices of
                a given volume every time.
        """
        if which_challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')
        self.mask_func = mask_func
        self.resolution = resolution
        self.which_challenge = which_challenge
        self.use_seed = use_seed

    def __call__(self, kspace, target, attrs, fname, slice):
        """
        Args:
            kspace (numpy.array): Input k-space of shape (num_coils, rows, cols, 2) for multi-coil
                data or (rows, cols, 2) for single coil data.
            target (numpy.array): Target image
            attrs (dict): Acquisition related information stored in the HDF5 object.
            fname (str): File name
            slice (int): Serial number of the slice.
        Returns:
            (tuple): tuple containing:
                image (torch.Tensor): Zero-filled input image.
                target (torch.Tensor): Target image converted to a torch Tensor.
                mean (float): Mean value used for normalization.
                std (float): Standard deviation value used for normalization.
                norm (float): L2 norm of the entire volume.
        """
        seed = None if not self.use_seed else tuple(map(ord, fname))

        # full kspace
        full_kspace = transforms.to_tensor(kspace)

        # target
        target = transforms.to_tensor(target)
        
        # crop kspace
        kspace_crop = transforms.fft2(transforms.complex_center_crop(transforms.ifft2(full_kspace), (self.resolution, self.resolution)))
       
        # scaling
        kspace_crop *= 1e6
        masked_kspace, mask = transforms.apply_mask(kspace_crop, self.mask_func, seed)

        # zero-filled
        zim = transforms.ifft2(masked_kspace)

        zim_abs = transforms.complex_abs(zim)
        _, mean, std = transforms.normalize_instance(zim_abs, eps=1e-11)

        # scaling
        target = target * 1e6
      
        # reshape
        zim = zim.permute(2,0,1)

        return zim, target, masked_kspace, mask, mean, std, attrs['max'].astype(np.float32), fname, slice




class DataTransform_complex_fastmri_recon:
    """
    adapt from reconformer
    Data Transformer for training fastmri complex-valued image
    """

    def __init__(self, mask_func, resolution, which_challenge, use_seed=True):
        """
        Args:
            mask_func (common.subsample.MaskFunc): A function that can create a mask of
                appropriate shape.
            resolution (int): Resolution of the image.
            which_challenge (str): Either "singlecoil" or "multicoil" denoting the dataset.
            use_seed (bool): If true, this class computes a pseudo random number generator seed
                from the filename. This ensures that the same mask is used for all the slices of
                a given volume every time.
        """
        if which_challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')
        self.mask_func = mask_func
        self.resolution = resolution
        self.which_challenge = which_challenge
        self.use_seed = use_seed

    def __call__(self, kspace, target, attrs, fname, slice):
        """
        Args:
            kspace (numpy.array): Input k-space of shape (num_coils, rows, cols, 2) for multi-coil
                data or (rows, cols, 2) for single coil data.
            target (numpy.array): Target image
            attrs (dict): Acquisition related information stored in the HDF5 object.
            fname (str): File name
            slice (int): Serial number of the slice.
        Returns:
            (tuple): tuple containing:
                image (torch.Tensor): Zero-filled input image.
                target (torch.Tensor): Target image converted to a torch Tensor.
                mean (float): Mean value used for normalization.
                std (float): Standard deviation value used for normalization.
                norm (float): L2 norm of the entire volume.
        """
        seed = None if not self.use_seed else tuple(map(ord, fname))

        # full kspace
        full_kspace = transforms.to_tensor(kspace)

        # target
        target = transforms.to_tensor(target)
        
        # crop kspace
        kspace_crop = transforms.fft2(transforms.complex_center_crop(transforms.ifft2(full_kspace), (self.resolution, self.resolution)))
       
        # scaling
        masked_kspace, mask = transforms.apply_mask(kspace_crop, self.mask_func, seed)

        # zero-filled
        zim = transforms.ifft2(masked_kspace)

        zim_abs = transforms.complex_abs(zim)
        _, mean, std = transforms.normalize_instance(zim_abs, eps=1e-11)

        # scaling
        zim /= mean 
        target = target/mean
        masked_kspace /= mean
      
        # reshape
        zim = zim.permute(2,0,1)

        return zim, target, masked_kspace, mask, mean, mean, attrs['max'].astype(np.float32), fname, slice








class DataTransform_complex_fastmri_multicoil:
    """
    Data Transformer for training fastmri complex-valued image
    """
    def __init__(self, mask_func, resolution, which_challenge, use_seed=True):
        """
        Args:
            mask_func (common.subsample.MaskFunc): A function that can create a mask of
                appropriate shape.
            resolution (int): Resolution of the image.
            which_challenge (str): Either "singlecoil" or "multicoil" denoting the dataset.
            use_seed (bool): If true, this class computes a pseudo random number generator seed
                from the filename. This ensures that the same mask is used for all the slices of
                a given volume every time.
        """
        if which_challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')
        self.mask_func = mask_func
        self.resolution = resolution
        self.which_challenge = which_challenge
        self.use_seed = use_seed

    def __call__(self, kspace, target, attrs, fname, slice):
        """
        Args:
            kspace (numpy.array): Input k-space of shape (num_coils, rows, cols, 2) for multi-coil
                data or (rows, cols, 2) for single coil data.
            target (numpy.array): Target image
            attrs (dict): Acquisition related information stored in the HDF5 object.
            fname (str): File name
            slice (int): Serial number of the slice.
        Returns:
            (tuple): tuple containing:
                image (torch.Tensor): Zero-filled input image.
                target (torch.Tensor): Target image converted to a torch Tensor.
                mean (float): Mean value used for normalization.
                std (float): Standard deviation value used for normalization.
                norm (float): L2 norm of the entire volume.
        """
        seed = None if not self.use_seed else tuple(map(ord, fname))

        # full kspace
        full_kspace = transforms.to_tensor(kspace) #(num_coils, rows, cols, 2)

        # target
        target = transforms.to_tensor(target) #(rows, cols)
        
        # crop kspace
        kspace_crop = transforms.fft2(transforms.complex_center_crop(transforms.ifft2(full_kspace), (self.resolution, self.resolution)))
       
        # scaling
        kspace_crop *= 1e6
        masked_kspace, mask = transforms.apply_mask(kspace_crop, self.mask_func, seed) #(num_coils, rows, cols, 2), (1,1,cols,1)

        # zero-filled
        zim = transforms.ifft2(masked_kspace) #(num_coils, rows, cols, 2)

        zim_abs = transforms.complex_abs(zim)
        _, mean, std = transforms.normalize_instance(zim_abs, eps=1e-11)

        # scaling
        target = target * 1e6
      
        # stacking
        zim = zim.reshape(-1,self.resolution,self.resolution) #(num_coils, rows, cols)

        return zim, target, masked_kspace, mask, mean, std, attrs['max'].astype(np.float32), fname, slice




class DataTransform_complex_fastmri_edge:
    """
    Data Transformer for training fastmri edge
    provide different edge extract for different gaussian smoothing  
    """

    def __init__(self, mask_func, resolution, which_challenge, use_seed=True):
        """
        Args:
            mask_func (common.subsample.MaskFunc): A function that can create a mask of
                appropriate shape.
            resolution (int): Resolution of the image.
            which_challenge (str): Either "singlecoil" or "multicoil" denoting the dataset.
            use_seed (bool): If true, this class computes a pseudo random number generator seed
                from the filename. This ensures that the same mask is used for all the slices of
                a given volume every time.
        """
        if which_challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')
        self.mask_func = mask_func
        self.resolution = resolution
        self.which_challenge = which_challenge
        self.use_seed = use_seed

    def __call__(self, kspace, target, attrs, fname, slice):
        """
        Args:
            kspace (numpy.array): Input k-space of shape (num_coils, rows, cols, 2) for multi-coil
                data or (rows, cols, 2) for single coil data.
            target (numpy.array): Target image
            attrs (dict): Acquisition related information stored in the HDF5 object.
            fname (str): File name
            slice (int): Serial number of the slice.
        Returns:
            (tuple): tuple containing:
                image (torch.Tensor): Zero-filled input image.
                target (torch.Tensor): Target image converted to a torch Tensor.
                mean (float): Mean value used for normalization.
                std (float): Standard deviation value used for normalization.
                norm (float): L2 norm of the entire volume.
        """
        seed = None if not self.use_seed else tuple(map(ord, fname))

        # full kspace
        full_kspace = transforms.to_tensor(kspace)

        # target
        target = transforms.to_tensor(target)
        
        # crop kspace
        kspace_crop = transforms.fft2(transforms.complex_center_crop(transforms.ifft2(full_kspace), (self.resolution, self.resolution)))
       
        # scaling
        kspace_crop *= 1e6
        masked_kspace, mask = transforms.apply_mask(kspace_crop, self.mask_func, seed)

        # zero-filled
        zim = transforms.ifft2(masked_kspace)

        zim_abs = transforms.complex_abs(zim)
        _, mean, std = transforms.normalize_instance(zim_abs, eps=1e-11)

        # scaling
        target = target * 1e6 #(320,320)
        
        # true edge
        gt_edge = torch.from_numpy(Get_sobel(target.numpy())) # 1 channel
        gt_edge = gt_edge / gt_edge.max() # normalize
        gt_edge = gt_edge * (gt_edge > 0.2)


        # zim edge
        zim_edge = torch.from_numpy(Get_sobel(zim_abs.numpy()))

        # reshape
        zim = zim.permute(2,0,1)
        zim_edge = zim_edge.unsqueeze(0)

        return zim, zim_edge, target, gt_edge, masked_kspace, mask, mean, std, attrs['max'].astype(np.float32), fname, slice


class DataTransform_complex_cc359:
    """
    Data Transformer for CC-359
    """

    def __init__(self, mask_func, resolution=256, use_seed=True):
        """
        Args:
            mask_func (common.subsample.MaskFunc): A function that can create a mask of
                appropriate shape.
            resolution (int): Resolution of the image.
            which_challenge (str): Either "singlecoil" or "multicoil" denoting the dataset.
            use_seed (bool): If true, this class computes a pseudo random number generator seed
                from the filename. This ensures that the same mask is used for all the slices of
                a given volume every time.
        """
        self.mask_func = mask_func
        self.resolution = resolution
        self.use_seed = use_seed

    def __call__(self, target, maxval, fname, slice):
        """
        Args:
            target (numpy.array): full-sampled kspace, complex-valued (slice, H, W, 2)
            maxval(float): useless, default 0
            fname (str): File name
            slice (int): Serial number of the slice.
        Returns:
            (tuple): tuple containing:
                image (torch.Tensor): Zero-filled input image.
                target (torch.Tensor): Target image converted to a torch Tensor, 1 channel
                mean (float): Mean value used for normalization.
                std (float): Standard deviation value used for normalization.
                norm (float): L2 norm of the entire volume.
        """
        seed = None if not self.use_seed else tuple(map(ord, fname))

        # full kspace
        kspace = transforms.to_tensor(target) #uncentered kspace
        kspace = transforms.fftshift(kspace, dim=(-3,-2))

        # scaling
        kspace /= 1e5

        # target
        target = transforms.ifftshift(transforms.ifft2(kspace))
        target = transforms.complex_abs(target)
       
        # masked kspace
        masked_kspace, mask = transforms.apply_mask(kspace, self.mask_func, seed)
        
        # zero-filled
        zim = transforms.ifftshift(transforms.ifft2(masked_kspace))

        zim_abs = transforms.complex_abs(zim)
        _, mean, std = transforms.normalize_instance(zim_abs, eps=1e-11)

        zim = zim.permute(2,0,1)

        return zim, target, masked_kspace, mask, mean, std, maxval, fname, slice




class DataTransform_complex_cc359_edge:
    """
    Data Transformer for CC-359
    """

    def __init__(self, mask_func, resolution=256, use_seed=True):
        """
        Args:
            mask_func (common.subsample.MaskFunc): A function that can create a mask of
                appropriate shape.
            resolution (int): Resolution of the image.
            which_challenge (str): Either "singlecoil" or "multicoil" denoting the dataset.
            use_seed (bool): If true, this class computes a pseudo random number generator seed
                from the filename. This ensures that the same mask is used for all the slices of
                a given volume every time.
        """
        self.mask_func = mask_func
        self.resolution = resolution
        self.use_seed = use_seed

    def __call__(self, target, maxval, fname, slice):
        """
        Args:
            target (numpy.array): full-sampled kspace, complex-valued (slice, H, W, 2)
            maxval(float): useless, default 0
            fname (str): File name
            slice (int): Serial number of the slice.
        Returns:
            (tuple): tuple containing:
                image (torch.Tensor): Zero-filled input image.
                target (torch.Tensor): Target image converted to a torch Tensor, 1 channel
                mean (float): Mean value used for normalization.
                std (float): Standard deviation value used for normalization.
                norm (float): L2 norm of the entire volume.
        """
        seed = None if not self.use_seed else tuple(map(ord, fname))

        # full kspace
        kspace = transforms.to_tensor(target) #uncentered kspace
        kspace = transforms.fftshift(kspace, dim=(-3,-2))

        # scaling
        kspace /= 1e5

        # target
        target = transforms.ifftshift(transforms.ifft2(kspace))
        
        target = transforms.complex_abs(target)
     
         
        # true edge
        gt_edge = torch.from_numpy(Get_sobel(target.numpy())) # 1 channel
        gt_edge = gt_edge / gt_edge.max() # normalize

        # masked kspace
        masked_kspace, mask = transforms.apply_mask(kspace, self.mask_func, seed)
        
        # zero-filled
        zim = transforms.ifftshift(transforms.ifft2(masked_kspace))

        zim_abs = transforms.complex_abs(zim)
        _, mean, std = transforms.normalize_instance(zim_abs, eps=1e-11)

        # zim edge
        zim_edge = torch.from_numpy(Get_sobel(zim_abs.numpy()))
        
        zim = zim.permute(2,0,1)

        return zim, zim_edge, target, gt_edge, masked_kspace, mask, mean, std, maxval, fname, slice








def create_datasets(dataName, dataMode, train_root, valid_root, center_fractions, 
                accelerations, resolution, sample_rate,challenge='singlecoil'):

    """
    create dataset, support cardiac, fastmri or cc359 dataset
    input: 
        dataName(str): fastmri or cc359
        dataMode(str): data type of input data. Real/complex
        train_root(str): path of the training data
        valid_root(str): path of the validation data
        center_fractions(list):
        accelerations(list):
        resolution(int): 320 for fastmri or 256 for cc359
        sample_rate(float):

    """
    assert dataName in ['cardiac','fastmri', 'cc359'], "Only support cardiac/fastmri/cc359 dataset"

    if dataName == 'cardiac':

        train_data = SliceData_cardiac(train_root, accelerations, isTrain=1)
        dev_data = SliceData_cardiac(valid_root, accelerations)

    else:
        # use fastmri masking function
        train_mask = MaskFunc(center_fractions, accelerations)
        dev_mask = MaskFunc(center_fractions, accelerations)
        if dataName == 'fastmri':
            if challenge == 'singlecoil':
                if dataMode == 'real':
                    dt = DataTransform_real_fastmri # for unet
                elif dataMode == 'complex':
                    dt = DataTransform_complex_fastmri
                elif dataMode == 'complex_edge':
                    dt = DataTransform_complex_fastmri_edge
                else:
                    raise NotImplementedError("Only support real/complex/complex_edge dataMode in fastmri dataloader")
            elif challenge == 'multicoil':
                dt = DataTransform_complex_fastmri_multicoil

            train_data = SliceData_fastmri(
                root=train_root, 
                transform=dt(train_mask, resolution, challenge),
                sample_rate=sample_rate,
                challenge=challenge,
                skip_head=1
            )
            dev_data = SliceData_fastmri(
                root=valid_root, 
                transform=dt(dev_mask, resolution, challenge, use_seed=True),
                sample_rate=sample_rate,
                challenge=challenge,
                skip_head=0
            )
        
        elif dataName == 'cc359':
            if dataMode == 'complex':
                dt = DataTransform_complex_cc359
            elif dataMode == 'complex_edge':
                dt = DataTransform_complex_cc359_edge
            else:
                raise NotImplementedError("Only support real/complex/complex_edge dataMode in fastmri dataloader")

            train_data = SliceData_cc359(
                root=train_root, 
                transform=dt(train_mask),
                sample_rate=sample_rate,
            )
            dev_data = SliceData_cc359(
                root=valid_root, 
                transform=dt(dev_mask, use_seed=True),
                sample_rate=sample_rate,
            )


    return dev_data, train_data







def fastmri_format(x): 
    if x.shape[1] == 1:
        x = x.squeeze(1)
    elif x.shape[1] == 2: # take modules
        x = ((x** 2).sum(dim=1) + 0.0).sqrt()
    else: #multi-coil
        bs = len(x)
        x = x.reshape(bs,-1,320,320,2) #(B,C,H,W,2)
        x = ((x**2).sum(dim=-1) + 0.0).sqrt() #(B,C,H,W)
        x = ((x**2).sum(dim=1) + 0.0).sqrt()

    return x



def getDataloader(dataName, dataMode, batch_size, center_fractions, accelerations, resolution, train_root, valid_root, sample_rate=1, challenge='singlecoil'): 

    dev_data, train_data = create_datasets(dataName, dataMode, train_root, valid_root, center_fractions, accelerations, resolution, sample_rate, challenge)

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    if dataName != 'cardiac':
        dev_loader = DataLoader(
            dataset=dev_data,
            batch_size=batch_size,
            num_workers=8,
            pin_memory=True,
        )
    else:
        dev_loader = DataLoader(
            dataset=dev_data,
            batch_size=1,
            num_workers=8,
            pin_memory=True,
        )

    return train_loader, dev_loader 





def handle_output(data, mode):
  
    assert mode in ['train', 'test'], "please provide correct mode for handle output"
    if mode == 'test':
        if isinstance(data, list) or isinstance(data, tuple): # take the last element
            data = data[-1]

    return data


if __name__ == '__main__':
    
    pass
