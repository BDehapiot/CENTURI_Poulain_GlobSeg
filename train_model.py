#%% Imports

import random
import numpy as np
from skimage import io 
from pathlib import Path
import albumentations as A
import matplotlib.pyplot as plt
from joblib import Parallel, delayed 
from stardist import gputools_available
from stardist.models import Config2D, StarDist2D

from functions import norm_data, split_data, augment_data

#%% Inputs

qlow = 0.001
qhigh = 0.999
split = 0.25

#%% Paths & import

for path in Path('data', 'train').iterdir(): 

    if 'crop' in path.name and 'mask' in path.name:   
            
        raw_path = Path('data', 'train', path.name.replace('_mask', ''))
        
        if 'raw' not in globals(): 
            raw = io.imread(raw_path)
            mask = io.imread(path)   
            
        else:
            raw = np.dstack((raw, io.imread(raw_path)))
            mask = np.dstack((mask, io.imread(path))) 
            
raw = np.rollaxis(raw, 2)      
mask = np.rollaxis(mask, 2) 

#%% Configure StarDist2D

conf = Config2D (
    axes='YXC',
    backbone='unet',
    grid=(2, 2),
    n_channel_in=1,
    n_channel_out=33,
    n_dim=2,
    n_rays=32,
    net_conv_after_unet=128,
    net_input_shape=(None, None, 1),
    net_mask_shape=(None, None, 1),
    train_background_reg=0.0001,
    train_batch_size=32,
    train_checkpoint='weights_best.h5',
    train_checkpoint_epoch='weights_now.h5',
    train_checkpoint_last='weights_last.h5',
    train_completion_crop=32,
    train_dist_loss='mae',
    train_epochs=32,
    train_foreground_only=0.9,
    train_learning_rate=0.0003,
    train_loss_weights=(1, 0.2),
    train_n_val_patches=None,
    train_patch_size=(128, 128),
    train_reduce_lr={'factor': 0.5, 'min_delta': 0, 'patience': 40},
    train_shape_completion=False,
    train_steps_per_epoch=64,
    train_tensorboard=True,
    unet_activation='relu',
    unet_batch_norm=False,
    unet_dropout=0.0,
    unet_kernel_size=(3, 3),
    unet_last_activation='relu',
    unet_n_conv_per_depth=2,
    unet_n_depth=3,
    unet_n_filter_base=32,
    unet_pool=(2, 2),
    unet_prefix='',
    use_gpu=True and gputools_available()
    )

model = StarDist2D(
    conf, 
    name='stardist', 
    basedir='models'
    )
               
#%% Execute

# Normalize data
raw = norm_data(raw, qlow, qhigh)

# Split data
raw_trn, mask_trn, raw_val, mask_val = split_data(raw, mask, split)

# Augment data
operations = A.Compose([
    A.VerticalFlip(p=0.5),              
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.Transpose(p=0.5),
    A.GridDistortion(p=0.5),
    ])

raw_trn, mask_trn = augment_data(
    raw_trn, mask_trn, operations, iterations=raw_trn.shape[0]*100, parallel=False
    )

# Train model
model.train(
    raw_trn, mask_trn, validation_data=(raw_val, mask_val)
    )

# train_log()
