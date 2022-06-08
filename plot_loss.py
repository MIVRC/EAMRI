from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pdb


def helper(root,scale=1.0):
    
    file1 = open(root,'r')
    lines = file1.readlines()
    vals = []
    for line in tqdm(lines):
        if 'loss' in line:
            splits = line.split(',')
            val = float(splits[-1].split('=')[-1])
            vals.append(val/scale)

    return vals


if __name__ == '__main__':

    root = './result/fastmri/default_c5_fastmri_reduce_r25_1000eps/log/generalLog.txt' 
    root1 = './result/fastmri/was_r25_1000eps/log/generalLog.txt' 
    c5 = np.array(helper(root))
    was = np.array(helper(root1, 4.0))
    was = was[:91]

    x = (1 + np.arange(len(c5))) * 10
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x,c5,label='c5')
    ax.plot(x,was,label='was')
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')
    plt.legend(loc='upper right')

    fig.tight_layout()
    fig.savefig('fa_test.png')
