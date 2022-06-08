# SEN-MRI
This is the official code hub for the SEN-MRI

# Requirements
- Python==3.6.6
- numpy==1.14.3
- opencv-python==3.4.1.15
- scipy==1.1.0
- pytorch == 1.7.0+cu110
- matplotlib==2.2.2
- scikit-image == 0.15.0
- h5py
- visdom

# How to Train
1. Prepare data.
2. Create an Initialization File in `config/`, named like `CONFIGNAME.ini`. You can also make a copy of `default.ini` and edit it.
3. run `python run_model.py CONFIGNAME`. Notice that `config/` and `.ini` will be added automatically.

# Prepare Data
1 Cardiac dataset: The original data is established from the work Alexander et al. Details can be found in the paper. 
You can download the original data from [Here](http://jtl.lassonde.yorku.ca/software/datasets/).

For us, we use the converted png images provided by [Here](https://github.com/tinyRattar/CDDNwithTDC_storage/tree/master/data/pngFormat), and the convert code is [Here](https://github.com/tinyRattar/CDDNwithTDC_storage/blob/master/data/saveAsPng.m).

Although there are 4480 frames, we only use 3300 frames(100 frames/patient). Preparing for training, you should:
1. Download the png-format data.
2. Put the data in `./data/cardiac_ktz/`.

2 Brain dataset: this dataset is establisded by Souza et al. you can download it from [Here](https://sites.google.com/view/calgary-campinas-dataset/download?authuser=0)

3 In our training process, we pre-generate a quantity of random sampling masks in the `mask/`, named like `mask_rAMOUT_SAMPLINGRATE.mat`. These masks will be applied in the constructor of dataset. 

# How to Conifg
## General
1. dataType: dataType_dataName_samplingRate_samplingMode_dataPartial. For example, complex_cardiac_rand15_static_reduce_. Note that the program determines the keyword in string way, so you can shuffle the order of the word. 
2. netType: The network for MRI reconstruction. All the options can be found in the function [`getNet`](https://github.com/tinyRattar/CSMRI_0325/blob/b5a8cec01b98a2be0c313dfe403488582c7fced2/network/__init__.py#L31). 
3. useCuda: use `True` for cuda
4. needParallel: use `True` if you want to train with multi gpu devices. We recommend to choose `True` even if only one devices is available.
5. device: 1 for use and 0 for not. E.g., you want to use the 2nd and the 3rd gpu devices , you should write `0110` here. (more or less devices is acceptable)
6. lossType: The loss function. `mse` or `mae`. 
7. path: The saving path for the log file and the trained weights.
8. SEED: random seed
9. num_workers: num of workers for pytorch dataloader


## Train
1. epoch: Epoch for training
2. batchSize: Batch-size for during training.
3. lr: Learning rate.
4. optimizer: Check [`getOptimizer`](https://github.com/tinyRattar/CSMRI_0325/blob/b5a8cec01b98a2be0c313dfe403488582c7fced2/network/__init__.py#L15).
5. weightDecay: It only work if you use `Adam_wd` in Optimizer above. Remember `Adam_DC_DCNN` and `Adam_RDN` will use the pre-defined weight decay.

## Log
1. saveEpoch: The result will be logged and saved per `SaveEpoch` epoches.
2. maxSaved: Only last `MaxSaved` weights will be reversed. Earlier ones will be removed automatically.


# How to evaluate/plot?
1. Please use the if statement in eval_model.py. For the other useless if statements, please use 0 to comment them.
2. Please run the code 'python eval_model.py CONFIGNAME epochNum'. Please make sure that epochNum is the same as the model weights stored in your model folder.


# How to load the model
Use the function [`loadCkpt`](https://github.com/tinyRattar/CSMRI_0325/blob/b5a8cec01b98a2be0c313dfe403488582c7fced2/core.py#L196) instead if you want to load the record. 
For example:
```python
c1 = core.core('PATH_TO_RESULT/config.ini', True) # True for not loading training dataset.
c1.loadCkpt(1000, True) # True for checked weight.
```

