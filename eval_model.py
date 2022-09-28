"""
support visdom
support scheduler
support random seed
support dsn

[20210920] 
seperate core class from main.py
in core.py, try to test different branch
"""
import os
import sys
#from core import Core
from core_fastmri import Core
import warnings
warnings.filterwarnings("ignore")
import pdb

# get seed
assert len(sys.argv)>1, "Need config name"
configName = sys.argv[1]
epoch = int(sys.argv[2])
folder = sys.argv[3]
# random number
if __name__ == '__main__':

    save_root = './images/' + folder
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    # load model type: normal(checkpoint) or best
    #loadType = 'normal'
    loadType = 0
    c = Core(configName, True)
    c.loadCkpt(epoch,loadType)

    if 0:
        # eval the test dataset and calculate psnr/ssim
        result = c.validation() 
        print(result)

    if 0:
        # eval the test dataset for each iteration branch
        c.validation_branch() 

    if 0:
        # plot the predictions for all models and save images into the same folder
        c.plot_results(save_root)
   
    if 1:
        # save to a single folder for each model
        save_root1 = os.path.join(save_root,configName)
        if not os.path.exists(save_root1):
            os.mkdir(save_root1)
        c.plot_results_to_one_folder(save_root1)

    if 0:
        # cal metric for zero-filled image
        c.cal_metric_zim()
