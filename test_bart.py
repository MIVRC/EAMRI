import torch
import numpy as np
from fastmri.data import transforms_simple as T 
import matplotlib.pyplot as plt
import h5py
import bart
import pdb


fname = '/home/ET/hanhui/opendata/fastmri_knee_multicoil_dataset/multicoil_train/file1000999.h5' 

#fname = '/home/ET/hanhui/opendata/CC-359_multi_coil/Train/e14089s3_P53248.7.h5' 

with h5py.File(fname, 'r') as data:
    kspace = data['kspace'][20] #(15, 640, 368)


ks = T.to_tensor(kspace) #(15, 640, 368, 2)
ks_crop = T.fft2(T.complex_center_crop(T.ifft2(ks, shift=True), (320, 320)), shift=True) #(15, 320, 320, 2)


'''
# gt
target0 = T.ifft2(full_kspace.permute(3,1,2,0).contiguous(), shift=False) #(12, H, W, 2)
target_abs = (target0**2).sum(dim=-1).sqrt() #(12,H,W)
for i in range(len(target_abs)):
    plt.imsave('coil_{}.png'.format(i), target_abs[i].numpy(), cmap='gray' )
'''
# zf
zim = T.ifft2(ks_crop)
zim = T.complex_abs(zim) #(15, 320, 320)
for i in range(zim.shape[0]):
    plt.imsave('coil_{}.png'.format(i), zim[i, :,:], cmap='gray')

# zf
#aux2 = np.fft.ifft2(kspace1, axes=(0, 1)) #(218, 170, 12) np.complex
#zim1 = np.sqrt((np.abs(aux2)**2).sum(axis=-1))

# sens map !!!!!
ks1 = ks_crop[:,:,:,0] + 1j*ks_crop[:,:,:,1] #(15, 320, 320)
ks1 = ks1.permute(1,2,0).contiguous()

sens_maps = bart.bart(1, "ecalib -m 1 -r 26", ks1.numpy()[None, ...])
#sen_maps_tor = torch.from_numpy(sens_maps)
#sen_maps_tor = T.fftshift(sen_maps_tor, dim=(-3,-2)) #shift sens_map


# plot 
#plt.imsave('bart_zf.png', zim1, cmap='gray')
temp = np.abs(sens_maps) #(H, W, coils)
for i in range(temp.shape[-1]):
    plt.imsave('bart_{}.png'.format(i), temp[0, :,:,i], cmap='gray')

