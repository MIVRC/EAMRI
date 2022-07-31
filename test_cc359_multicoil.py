import torch
import h5py
import numpy as np
import pdb
import glob
import cv2 as cv
import torch.nn as nn
import matplotlib.pyplot as plt
from fastmri.data import transforms_simple as T
from fastmri.data.transforms_simple import EstimateSensitivityMap 
from util import Get_sobel 
import warnings;warnings.filterwarnings("ignore")


DATA_PATH = "/home/ET/hanhui/opendata/CC-359_multi_coil/Train/*.h5"
center = [0.08]
accer = [4]
resolution = 320

'''
class SensitivityModel(nn.Module):
    """
    Model for learning sensitivity estimation from k-space data.
    This model applies an IFFT to multichannel k-space data and then a U-Net
    to the coil images to estimate coil sensitivities. It can be used with the
    end-to-end variational network.
    """

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
        mask_center: bool = True,
        shift:bool = False,
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
            mask_center: Whether to mask center of k-space for sensitivity map
                calculation.
        """
        super(SensitivityModel, self).__init__()
        self.mask_center = mask_center
        self.shift = shift
        
        self.norm_unet = sensNet(convNum=3, recursiveTime=1, inChannel=2, midChannel=8) 

    def chans_to_batch_dim(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        b, c, h, w, comp = x.shape 

        return x.view(b * c, 1, h, w, comp), b

    def batch_chans_to_chan_dim(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        bc, _, h, w, comp = x.shape
        c = bc // batch_size

        return x.view(batch_size, c, h, w, comp)

    def divide_root_sum_of_squares(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, coils, H, W, 2)
        assert x.shape[-1] == 2, "the last dimension of input should be 2"
        tmp = T.root_sum_of_squares(T.root_sum_of_squares(x, dim=4), dim=1).unsqueeze(-1).unsqueeze(1)
        
        return x/tmp
        #return T.safe_divide(x, tmp).cuda()


    def get_pad_and_num_low_freqs(
        self, mask: torch.Tensor, num_low_frequencies: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if num_low_frequencies is None or num_low_frequencies == 0:
            # get low frequency line locations and mask them out
            # mask: [B, 1, H, W, 1]
            squeezed_mask = mask[:, 0, 0, :, 0].to(torch.int8)
            cent = squeezed_mask.shape[1] // 2
            # running argmin returns the first non-zero
            left = torch.argmin(squeezed_mask[:, :cent].flip(1), dim=1)
            right = torch.argmin(squeezed_mask[:, cent:], dim=1)
            num_low_frequencies_tensor = torch.max(
                2 * torch.min(left, right), torch.ones_like(left)
            )  # force a symmetric center unless 1
        else:
            num_low_frequencies_tensor = num_low_frequencies * torch.ones(
                mask.shape[0], dtype=mask.dtype, device=mask.device
            )

        pad = (mask.shape[-2] - num_low_frequencies_tensor + 1) // 2

        return pad, num_low_frequencies_tensor

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        num_low_frequencies: Optional[int] = None,
    ) -> torch.Tensor:
        
        if self.mask_center:
            pad, num_low_freqs = self.get_pad_and_num_low_freqs(
                mask, num_low_frequencies
            )
            masked_kspace = T.batched_mask_center(
                masked_kspace, pad, pad + num_low_freqs
            )

        # convert to image space
        # images: [96, 1, 218, 179, 2]
        # batches: 8
        images, batches = self.chans_to_batch_dim(T.ifft2(masked_kspace, shift=self.shift))
        
        # estimate sensitivities
        return self.divide_root_sum_of_squares(self.batch_chans_to_chan_dim(self.norm_unet(images), batches))


'''



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


ks_abs = (full_kspace ** 2).sum(dim=-1).sqrt()
ks_abs = np.log(1 + ks_abs)
for i in range(12):
    plt.imsave('gt_kspace_{}.png'.format(i), ks_abs[i], cmap='gray' )





'''
#===============================================
# sens map
#===============================================
esmap = EstimateSensitivityMap(gaussian_sigma=0.3)
sens_map = esmap(full_kspace) #(coil, [slice], h, w, 2)
sens_map_abs = (sens_map**2).sum(dim=-1) #(coil, [slice], h, w)

for i in range(len(sens_map)):
    plt.imsave('sens_map_{}.png'.format(i), sens_map_abs[i].numpy(), cmap='gray' )


#===============================================
# coil image
#===============================================
target0 = T.ifft2(full_kspace, shift=False) #(12, H, W, 2)
target_abs = (target0**2).sum(dim=-1).sqrt() #(12,H,W)
for i in range(len(target_abs)):
    plt.imsave('coil_{}.png'.format(i), target_abs[i].numpy(), cmap='gray' )

'''

#===============================================
# gt
#===============================================
#target0 = (target0**2).sum(dim=-1).sum(dim=0).sqrt() + 0. #(H, W)
#plt.imsave('gt.png', target0.numpy(), cmap='gray' )


'''
#===============================================
# edge
#===============================================
target_rss = (target_abs ** 2).sum(dim=0).sqrt().numpy() #(H,W)
n_target_rss = (255*(target_rss - np.min(target_rss))/np.ptp(target_rss))
gt_edge = torch.from_numpy(Get_sobel(n_target_rss)) # (H, W)
gt_edge = gt_edge / gt_edge.max()
plt.imsave('edge.png', gt_edge.numpy(), cmap='gray' )


aux = np.fft.ifft2(kspace[:, :, ::2] + 1j * kspace[:, :, 1::2], axes=(0, 1))
target = torch.from_numpy(np.abs(aux)+0.) #(H, W, C)
target = ((target**2).sum(axis=-1) + 0.0).sqrt() #(H, W)
#target = torch.unsqueeze(target, 0) #(1, H, W)

'''

#===============================================
# mask kspace
#===============================================
mask_func = MaskFunc([0.08], [4])
mask =  mask_func((218,170,1), 'test') # use different seed for training

temp = kspace * mask #(H, W, C)
masked_kspace = np.zeros((2, 218, 170, 12)) #(2, H, W, 12)
masked_kspace[0,:,:,:] = temp[:,:,::2]
masked_kspace[1,:,:,:] = temp[:,:,1::2]
masked_kspace = torch.from_numpy(masked_kspace).permute(3,1,2,0).contiguous() #(12, H, W, 2)
ks_abs = (masked_kspace ** 2).sum(dim=-1).sqrt().numpy()
ks_abs = np.log(1 + ks_abs)

for i in range(12):
    plt.imsave('kspace_{}.png'.format(i), ks_abs[i], cmap='gray' )


#===============================================
# zero-filled 1
#===============================================
aux2 = np.fft.ifft2(temp[:,:,::2] + 1j * temp[:,:,1::2], axes=(0, 1)) #(218, 170, 12) np.complex
zim1 = np.sqrt((np.abs(aux2)**2).sum(axis=-1))

zf_ks = np.fft.fft2(aux2) #(218, 170, 12)
zfks_abs = np.abs(zf_ks)
zfks_abs = np.log(1 + zfks_abs)

for i in range(12):
    plt.imsave('zf_kspace_{}.png'.format(i), zfks_abs[:,:,i], cmap='gray' )



'''
#===============================================
# zero-filled 2
print(masked_kspace.shape)
aux22 = T.ifft2(masked_kspace, shift=False) #(12, H, W, 2)
zim2 = (aux22 ** 2).sum(dim=-1).sum(dim=0).sqrt()
#===============================================


#===============================================
# reduce operator
#===============================================

tmp = T.reduce_operator(aux22, sens_map) #(H, W, 2)
tmp_abs = (tmp**2).sum(dim=-1).sqrt()
plt.imsave('reduce_output.png', tmp_abs, cmap='gray' )


#===============================================
# expand operator
#===============================================
tmp_expand = T.expand_operator(tmp, sens_map)



plt.imsave('cc359_m_gt.png', target.numpy(), cmap='gray' )
plt.imsave('cc359_m_gt2.png', target0.numpy(), cmap='gray' )
plt.imsave('cc359_m_zim1.png', zim1, cmap='gray' )
plt.imsave('cc359_m_zim2.png', zim2, cmap='gray' )

'''
