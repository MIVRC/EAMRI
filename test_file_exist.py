"""
test fasmri multicoil files exist on the list
"""

import os
import pandas as pd
import pdb
import h5py
import shutil

type1 = 'val'
from_root= '/home/ET/hanhui/opendata/fastmri_knee_multicoil_dataset/multicoil_{}_less_sensitivity/'.format(type1)

to_root1= '/home/ET/hanhui/opendata/fastmri_knee_multicoil_dataset/multicoil_{}_pd_sensitivity/'.format(type1)
to_root2= '/home/ET/hanhui/opendata/fastmri_knee_multicoil_dataset/multicoil_{}_pdfs_sensitivity/'.format(type1)


if not os.path.exists(to_root1):
    os.mkdir(to_root1)

if not os.path.exists(to_root2):
    os.mkdir(to_root2)

file1 = '/home/ET/hanhui/opendata/fastmri_knee_multicoil_dataset/{}_split.csv'.format(type1)

df = pd.read_csv(file1, header=None)
cnt = 0

for idx, row in df.iterrows():
    
    print("process {} ".format(idx))
    from1 = os.path.join(from_root, row[0]+'.h5') #pd
    from2 = os.path.join(from_root, row[1]+'.h5') #pdfs

    dst1 = os.path.join(to_root1, row[0]+'.h5') #pd
    dst2 = os.path.join(to_root2, row[1]+'.h5') #pdfs

    shutil.copyfile(from1, dst1)
    shutil.copyfile(from2, dst2)

