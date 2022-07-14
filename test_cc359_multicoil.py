import torch
import h5py
import numpy as np
import pdb
import glob
import matplotlib.pyplot as plt
from fastmri.data import transforms_simple as T
from fastmri.data.transforms_simple import EstimateSensitivityMap 
import warnings;warnings.filterwarnings("ignore")


DATA_PATH = "/home/ET/hanhui/opendata/CC-359_multi_coil/Train/*.h5"
center = [0.08]
accer = [4]
resolution = 320


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



train = glob.glob(DATA_PATH.__str__())
train = train[10]
with h5py.File(train, 'r') as f:
    kspace = f['kspace'][100]

full_kspace = np.zeros((2, 218, 170, 12)) #(2, H, W, 12)

full_kspace[0,:,:,:] = kspace[:,:,::2]
full_kspace[1,:,:,:] = kspace[:,:,1::2]
full_kspace = torch.from_numpy(full_kspace).permute(3,1,2,0).contiguous() #(12, H, W, 2)

# sens map
esmap = EstimateSensitivityMap(gaussian_sigma=0.3)
sens_map = esmap(full_kspace) #(coil, [slice], h, w, 2)
sens_map = (sens_map**2).sum(dim=-1) #(coil, [slice], h, w)

for i in range(len(sens_map)):
    plt.imsave('sens_map_{}.png'.format(i), sens_map[i].numpy(), cmap='gray' )


target0 = T.ifft2(full_kspace, shift=False) #(12, H, W, 2)
target_abs = (target0**2).sum(dim=-1).sqrt() #(12,H,W)

for i in range(len(target_abs)):
    plt.imsave('coil_{}.png'.format(i), target_abs[i].numpy(), cmap='gray' )


target0 = (target0**2).sum(dim=-1).sum(dim=0).sqrt() + 0. #(H, W)

aux = np.fft.ifft2(kspace[:, :, ::2] + 1j * kspace[:, :, 1::2], axes=(0, 1))
target = torch.from_numpy(np.abs(aux)+0.) #(H, W, C)
target = ((target**2).sum(axis=-1) + 0.0).sqrt() #(H, W)
#target = torch.unsqueeze(target, 0) #(1, H, W)

# masking
mask_func = MaskFunc([0.08], [4])
mask =  mask_func((218,170,1), 'test') # use different seed for training

temp = kspace * mask #(H, W, C)
masked_kspace = np.zeros((2, 218, 170, 12)) #(2, H, W, 12)
masked_kspace[0,:,:,:] = temp[:,:,::2]
masked_kspace[1,:,:,:] = temp[:,:,1::2]
masked_kspace = torch.from_numpy(masked_kspace).permute(3,1,2,0).contiguous() #(12, H, W, 2)


# zero-filled 1
aux2 = np.fft.ifft2(temp[:,:,::2] + 1j * temp[:,:,1::2], axes=(0, 1)) #(218, 170, 12) np.complex
zim1 = np.sqrt((np.abs(aux2)**2).sum(axis=-1))

# zero-filled 2
print(masked_kspace.shape)
aux22 = T.ifft2(masked_kspace, shift=False) #(12, H, W, 2)
zim2 = (aux22 ** 2).sum(dim=-1).sum(dim=0).sqrt()

plt.imsave('cc359_m_gt.png', target.numpy(), cmap='gray' )
plt.imsave('cc359_m_gt2.png', target0.numpy(), cmap='gray' )
plt.imsave('cc359_m_zim1.png', zim1, cmap='gray' )
plt.imsave('cc359_m_zim2.png', zim2, cmap='gray' )

