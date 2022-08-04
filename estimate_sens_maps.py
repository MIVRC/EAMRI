import sys, os
#sys.path.insert(0, './bart-0.6.00/python')
#os.environ["OMP_NUM_THREADS"] = "3"
#os.environ["TOOLBOX_PATH"]    = './bart-0.6.00'
#sys.path.append('./bart-0.6.00/python')

import glob
import numpy as np
from tqdm import tqdm
import multiprocessing
import h5py
import torch
import click
from bart import bart
from fastmri.data import transforms_simple as T 

'''
@click.command()
@click.option('--input-dir', default=, help='directory with raw data ')
@click.option('--output-dir', default=, help='output directory for maps')
'''

def main0(input_dir, output_dir):
    
    # estimate sensitivity map for cc359 dataset

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_list = sorted(glob.glob(input_dir + '/*.h5'))
    for file in tqdm(file_list):
        basename = os.path.basename( file ) 
        output_name = os.path.join( output_dir, basename )
        if os.path.exists( output_name ):
            continue
        with h5py.File(file, 'r') as data:
            #num_slices = int(data.attrs['num_slices'])
            kspace = np.array( data['kspace'] ) #(slices, H, W, 24)
            s1 = kspace.shape
            s_maps = np.zeros((s1[0], s1[1], s1[2], 12), dtype = np.complex64)
            num_slices = kspace.shape[0]
            num_coils = kspace.shape[1]
            for slice_idx in range( num_slices ):
                gt_ksp = kspace[slice_idx] #(H, W, 24)
                gt_ksp1 = gt_ksp[:,:,::2] + 1j*gt_ksp[:,:,1::2] #(H, W, 12)
                s_maps_ind = bart(1, 'ecalib -m1 -r26', gt_ksp1[None,...]) #(1, H, W, 12)
                s_maps_ind = T.fftshift(torch.from_numpy(s_maps_ind), dim=(-3, -2)).numpy()
                s_maps[ slice_idx ] = s_maps_ind

            h5 = h5py.File( output_name, 'w' )
            h5.create_dataset( 'sens_maps', data = s_maps )
            h5.close()
        

def main_fastmri_crop(input_dir, output_dir):
    
    # estimate sensitivity map for fastmri dataset
    # crop the dataset first

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_list = sorted(glob.glob(input_dir + '/*.h5'))
    for file in tqdm(file_list):
        basename = os.path.basename( file ) 
        output_name = os.path.join( output_dir, basename )
        if os.path.exists( output_name ):
            continue
        with h5py.File(file, 'r') as data:
            kspace = torch.from_numpy(np.array( data['kspace'])) #(slices, 15, H, W)
            s1 = kspace.shape
            s_maps = np.zeros((s1[0], 320, 320, 15), dtype = np.complex64) #(slices, 320, 320, 15)
            num_slices = kspace.shape[0]
            num_coils = kspace.shape[1]
            for slice_idx in range( num_slices ):
                gt_ksp = kspace[slice_idx] #(15, H, W)

                # crop
                ks = T.to_tensor(gt_ksp) #(15, 640, 368, 2)
                ks_crop = T.fft2(T.complex_center_crop(T.ifft2(ks, shift=True), (320, 320)), shift=True) #(15, 320, 320, 2)
                ks1 = ks_crop[:,:,:,0] + 1j*ks_crop[:,:,:,1] #(15, 320, 320)
                ks1 = ks1.permute(1,2,0).contiguous() #(320, 320, 15)
                
                s_maps_ind = bart(1, 'ecalib -m1 -r26', ks1.numpy()[None,...]) #(1, H, W, 15)
                s_maps[ slice_idx ] = s_maps_ind

            h5 = h5py.File( output_name, 'w' )
            h5.create_dataset( 'sens_maps', data = s_maps )
            h5.close()
 

def main_fastmri_nocrop(file1):
    
    # estimate sensitivity map for fastmri dataset

    #file_list = sorted(glob.glob(input_dir + '/*.h5'))
    #for file in tqdm(file_list):

    output_dir = '/home/ET/hanhui/opendata/fastmri_knee_multicoil_dataset/multicoil_val_pd_sensitivity_no_crop/' 
    basename = os.path.basename(file1) 

    print("processing file {}".format(basename))
    output_name = os.path.join(output_dir, basename)

    if not os.path.exists( output_name ):
        with h5py.File(file1, 'r') as data:
            kspace = torch.from_numpy(np.array(data['kspace'])).permute(0,2,3,1).contiguous() #(slices, H, W, 15)
            s1 = kspace.shape
            s_maps = np.zeros(s1, dtype = np.complex64) #(slices, H, W, 15)
            num_slices = s1[0]
            for slice_idx in range( num_slices ):
                gt_ksp = kspace[slice_idx] #(H, W, 15)
                s_maps_ind = bart(1, 'ecalib -m1 -r26', gt_ksp.numpy()[None,...]) #(1, H, W, 15)
                s_maps[ slice_idx ] = s_maps_ind

            h5 = h5py.File( output_name, 'w' )
            h5.create_dataset( 'sens_maps', data = s_maps )
            h5.close()
 

if __name__ == '__main__':
    '''
    # old
    input_dir = '/home/ET/hanhui/opendata/fastmri_knee_multicoil_dataset/multicoil_train_pd/' 
    output_dir = '/home/ET/hanhui/opendata/fastmri_knee_multicoil_dataset/multicoil_train_pd_sensitivity_no_crop/' 
    main_fastmri(input_dir, output_dir)
    '''

    input_dir = '/home/ET/hanhui/opendata/fastmri_knee_multicoil_dataset/multicoil_val_pd/' 
    output_dir = '/home/ET/hanhui/opendata/fastmri_knee_multicoil_dataset/multicoil_val_pd_sensitivity_no_crop/' 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_list = sorted(glob.glob(input_dir + '/*.h5'))

    with multiprocessing.Pool(8) as pool:
        pool.map(main_fastmri_nocrop, file_list)

