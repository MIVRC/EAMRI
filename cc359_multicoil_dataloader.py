import torch
import h5py
import glob
import numpy as np
from torch.utils.data import Dataset
from fastmri.data import transforms_simple as T
from typing import List, Tuple
import pdb


class MaskFunc:
    """
    adapt from dataloader.py
    """
    def __init__(self, center_fractions, accelerations):
        if len(center_fractions) != len(accelerations):
            raise ValueError('Number of center fractions should match number of accelerations')

        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.rng = np.random.RandomState()

    def __call__(self, shape, fname):
        if len(shape) < 3:
            raise ValueError('Shape should have 3 or more dimensions')
        
        seed = tuple(map(ord, fname))
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

        mask_prod = np.zeros((1,num_cols,1)) 
        mask_prod[0,:,0] = mask
        temp = np.ones(shape)
        temp = temp * mask_prod

        return temp#(H,W,1)




class SliceData_cc359_multicoil(Dataset):
    """Generates image-domain data for Keras models during training and testing.
    Performs iFFT to yield zero-filled images as input data with fully-sampled references as the target."""

    def __init__(
        self,
        root: str,
        crop: Tuple[int],
        center_fractions: int,
        accelerations: int,
        shuffle: bool,
        is_train:bool,
        dataMode:str, 
    ):
        """Constructor for DataGenerator.
        :param root: root to store the data
        :type list_IDs: List[str]
        :param dim: Spatial dimension of images,
        :type dim: Tuple[int]
        :param under_masks: Numpy mask to simulate under-sampling of k-space.
            See ./Data/poisson_sampling/*.npy for masks.
        :type under_masks: np.ndarray
        :param crop: Tuple containing slices to crop from volumes. Ie., (30, 30) crops the first and last 30 slices from
            volume used to train
        :type crop: Tuple[int]
        :param batch_size: Batch size to generate data in.
        :type batch_size: int
        :param n_channels: Number of channels (coils*2) in the data provided in the list_IDs param.
            eg., n_channels = 24 for track 01 data (12 real, 12 imaginary channels)
        :type n_channels: int
        :param nslices: Number of slices per volume, defaults to 256
        :type nslices: int, optional
        :param shuffle: Whether or not to shuffle data, defaults to True.
        :type shuffle: bool, optional
        """
        self.list_IDs = glob.glob((root+"/*.h5").__str__())
        self.dim = (218,170)
        
        self.mask_func = MaskFunc(center_fractions, accelerations)

        self.crop = crop  # Remove slices with no or little anatomy
        self.batch_size = 1
        self.dataMode = dataMode
        self.n_channels = 24
        self.nslices = 256
        self.is_train = is_train
        self.shuffle = shuffle
        self.nsamples = len(self.list_IDs) * (self.nslices - self.crop[0] - self.crop[1])
        self.on_epoch_end()

    def __len__(self) -> int:
        """Denotes the number of batches per epoch"""
        return int(np.floor(self.nsamples / self.batch_size))

    def __getitem__(self, index: int) -> Tuple[np.ndarray]:
        """Get batch at index"
        :param index: Index to retrieve batch
        :type index: int
        :return: X,y tuple of zero-filled inputs and fully-sampled reconstructions.
            Shape of X and y is [batch_size, dim[0], dim[1], n_channels]
        :rtype: Tuple[np.ndarray]
        """
        # Generate indexes of the batch
        batch_indexes = self.indexes[index]  

        file_id = batch_indexes // (self.nslices - self.crop[0] - self.crop[1])
        file_slice = self.crop[0] + batch_indexes % (self.nslices - self.crop[0] - self.crop[1])
        fname = self.list_IDs[file_id]
        
        if self.is_train:
            mask =  self.mask_func((218,170,1), fname) # use different seed for training
        else:
            mask =  self.mask_func((218,170,1), 'test') # use fix seed for testing 
        
        if 'edge' in self.dataMode:
            assert True, "not implemented"
        else:
            zim, target, masked_kspace, mask, mean, std, norm, fname, file_slice = self.__data_generation_new(fname, file_slice, batch_indexes, mask)

        return zim, target, masked_kspace, mask, mean, std, norm, fname, file_slice 


    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(self.nsamples)
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    
    def __data_generation(self, fname, file_slice, batch_indexes, mask):
        """Generates data containing batch_size samples
        :param batch_indexes: Ndarray containing indices to generate in this batch.
        :type batch_indexes: np.ndarray
        :return: X,y tuple containing zero-filled under-sampled and fully-sampled data, respectively.
            Shape X and y is [batch_size, dim[0], dim[1], n_channels]
        :rtype: Tuple[np.ndarray]
        """
        # Initialization
        zim = np.zeros((self.n_channels, self.dim[0], self.dim[1])) #(24, H, W) 
        masked_kspace = np.zeros((2, self.dim[0], self.dim[1], self.n_channels//2)) #(2, H, W, 12)
        
        # load_data
        with h5py.File(fname, "r") as f:
            data = f["kspace"]
            if data.shape[2] == self.dim[1]:
                kspace = data[file_slice]
            else: #exceeds 170
                idx = int((data.shape[2] - self.dim[1]) / 2)
                kspace = data[file_slice, :, idx:-idx, :]
     
        # kspace: (H,W,24)

        aux = np.fft.ifft2(kspace[:, :, ::2] + 1j * kspace[:, :, 1::2], axes=(0, 1))
        target = torch.from_numpy(np.abs(aux)+0.) #(H, W, C)
        target = ((target**2).sum(axis=-1) + 0.0).sqrt() #(H, W)

        # mask kspace 
        temp = kspace * mask #(H, W, 24) mask:(H,W,1)
        masked_kspace[0,:,:,:] = temp[:,:,::2]
        masked_kspace[1,:,:,:] = temp[:,:,1::2]
        
        # zero-filled
        aux2 = np.fft.ifft2(temp[:,:,::2] + 1j * temp[:,:,1::2], axes=(0, 1)) #(218, 170, 12) np.complex
        zim[::2, :, :] = np.transpose(aux2.real, (2,0,1))
        zim[1::2, :, :] = np.transpose(aux2.imag, (2,0,1))

        # normalize
        norm = np.abs(aux2).max() * np.ones((1,1,1)) #(1,1,1)
        zim = torch.from_numpy(zim/norm)
        target = target / norm[:,:,0]
        masked_kspace = torch.from_numpy(masked_kspace / norm)
        mask = torch.unsqueeze(torch.from_numpy(mask),0) #(1, H, W, 1)
        fname = fname.split('/')[-1]
        
        return zim, target, masked_kspace.permute(3,1,2,0), mask, 0, 1, norm, fname, file_slice


    def __data_generation_new(self, fname, file_slice, batch_indexes, mask):
        """Generates data containing batch_size samples
        :param batch_indexes: Ndarray containing indices to generate in this batch.
        :type batch_indexes: np.ndarray
        :return: X,y tuple containing zero-filled under-sampled and fully-sampled data, respectively.
            Shape X and y is [batch_size, dim[0], dim[1], n_channels]
        :rtype: Tuple[np.ndarray]
        """
        # Initialization
        masked_kspace = np.zeros((2, self.dim[0], self.dim[1], self.n_channels//2)) #(2, H, W, 12)
        full_kspace = np.zeros((2, self.dim[0], self.dim[1], self.n_channels//2)) #(2, H, W, 12)
        
        # load_data
        with h5py.File(fname, "r") as f:
            data = f["kspace"]
            if data.shape[2] == self.dim[1]:
                kspace = data[file_slice]
            else: #exceeds 170
                idx = int((data.shape[2] - self.dim[1]) / 2)
                kspace = data[file_slice, :, idx:-idx, :]
     
        # target: (H,W)
        full_kspace[0,:,:,:] = kspace[:,:,::2]
        full_kspace[1,:,:,:] = kspace[:,:,1::2]
        full_kspace = torch.from_numpy(full_kspace).permute(3,1,2,0).contiguous() #(12, H, W, 2)
        target = T.ifft2(full_kspace, shift=False) #(12, H, W, 2)
        target = (target**2).sum(dim=-1).sum(dim=0).sqrt() + 0. #(H, W)

        # mask kspace 
        temp = kspace * mask #(H, W, 24) mask:(H,W,1)
        masked_kspace[0,:,:,:] = temp[:,:,::2]
        masked_kspace[1,:,:,:] = temp[:,:,1::2]
        masked_kspace = torch.from_numpy(masked_kspace).permute(3,1,2,0).contiguous() #(12, H, W, 2)
        
        # zero-filled
        zim = T.ifft2(masked_kspace, shift=False) #(12, H, W, 2)
        norm = (zim**2).sum(dim=-1).sqrt().max().item()

        
        # normalize
        zim = zim/norm
        zim = zim.reshape(-1, self.dim[0], self.dim[1]) # stacking
        target = target / norm
        masked_kspace = masked_kspace / norm
        mask = torch.unsqueeze(torch.from_numpy(mask),0) #(1, H, W, 1)
        fname = fname.split('/')[-1]
        
        return zim, target, masked_kspace, mask, 0, 1, norm, fname, file_slice




    def __data_generation_edge(self, batch_indexes, mask):
        """Generates data containing batch_size samples
        :param batch_indexes: Ndarray containing indices to generate in this batch.
        :type batch_indexes: np.ndarray
        :return: X,y tuple containing zero-filled under-sampled and fully-sampled data, respectively.
            Shape X and y is [batch_size, dim[0], dim[1], n_channels]
        :rtype: Tuple[np.ndarray]
        """
        # Initialization
        kspace = np.zeros((self.batch_size, self.dim[0], self.dim[1], self.n_channels)) #kspace data
        target = np.zeros((self.batch_size, self.dim[0], self.dim[1], self.n_channels)) #gt image
        zim = np.zeros((self.batch_size, self.dim[0], self.dim[1], self.n_channels)) #gt image
        #mask = np.zeros((self.batch_size, self.dim[0], self.dim[1]))

        # Generate data
        for ii in range(batch_indexes.shape[0]):
            # Store sample
            file_id = batch_indexes[ii] // (self.nslices - self.crop[0] - self.crop[1])
            file_slice = batch_indexes[ii] % (self.nslices - self.crop[0] - self.crop[1])
            # Load data
            with h5py.File(self.list_IDs[file_id], "r") as f:
                data = f["kspace"]
                if data.shape[2] == self.dim[1]:
                    kspace[ii, :, :, :] = data[self.crop[0] + file_slice]
                else: #exceeds 170
                    idx = int((data.shape[2] - self.dim[1]) / 2)
                    kspace[ii, :, :, :] = data[self.crop[0] + file_slice, :, idx:-idx, :]
        
        aux = np.fft.ifft2(kspace[:, :, :, ::2] + 1j * kspace[:, :, :, 1::2], axes=(1, 2))
        target[:, :, :, ::2] = aux.real
        target[:, :, :, 1::2] = aux.imag
      
        # masking
        masked_kspace = kspace * mask
        
        aux2 = np.fft.ifft2(masked_kspace[:, :, :, ::2] + 1j * masked_kspace[:, :, :, 1::2], axes=(1, 2))
        zim[:, :, :, ::2] = aux2.real
        zim[:, :, :, 1::2] = aux2.imag
        norm = np.abs(aux2).max(axis=(1, 2, 3), keepdims=True)  

        # normalize
        target = target / norm  
        masked_kspace = masked_kspace / norm  
        

        return zim, target, masked_kspace, norm 
