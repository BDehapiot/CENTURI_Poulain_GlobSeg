#%% Imports

import napari
import numpy as np
from skimage import io 
from pathlib import Path
import matplotlib.pyplot as plt
from joblib import Parallel, delayed 

#%% Inputs

qlow = 0.001
qhigh = 0.999
split = 0.25

#%% Create paths & import data

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

# # Check imports
# viewer = napari.Viewer()
# viewer.add_image(raw[5])
# viewer.add_labels(mask[5])
               
#%% Normalize data

def norm_data(raw, qlow, qhigh):
    
    """
    Description
    
    Parameters
    ----------
    raw : ndarray
        Description
    qlow : float
        Description
    qhigh : float
        Description
    
    Returns
    -------
    raw_trn : ndarray or list of ndarray
        Description
    
    Raises
    ------
    """
    
    raw = raw.astype('float')
    
    for i, img in enumerate(raw):
        
        # Get lower and higher threshold
        tlow = np.quantile(img, qlow)
        thigh = np.quantile(img, qhigh)
        
        # Normalize image
        img = (img - tlow) / (thigh - tlow)
        img[img > 1] = 1
        img[img < 0] = 0
        
        # Update raw
        raw[i,...] = img
        
    return raw             

''' ----------------------------------------------------------------------- '''

raw = norm_data(raw, qlow, qhigh)

#%% Split training & validation data

import random

def split_data(raw, mask, split):
    
    """
    Description
    
    Parameters
    ----------
    raw : ndarray
        Description
    mask : ndarray
        Description
    split : int
        Description
    
    Returns
    -------
    raw_trn : ndarray
        Description
    mask_trn : ndarray
        Description
    raw_val : ndarray
        Description
    mask_val : ndarray
        Description
    
    Raises
    ------
    """
    
    nI = raw.shape[0]
    
    # Define index
    idx = random.sample(range(0, nI), nI)
    trn_idx = idx[0:int(nI*(1-split))]
    val_idx = idx[-(nI-len(trn_idx)):]
    
    # Extract data    
    raw_trn = raw[trn_idx,...]
    mask_trn = mask[trn_idx,...]
    raw_val = raw[val_idx,...]
    mask_val = mask[val_idx,...]

    return raw_trn, mask_trn, raw_val, mask_val

''' ----------------------------------------------------------------------- '''

raw_trn, mask_trn, raw_val, mask_val = split_data(raw, mask, split)

#%% Augment data

import albumentations as A

def augment_data(raw, mask, operations, iterations=256, parallel=True):
    
    """
    Description
    
    Parameters
    ----------
    raw : ndarray
        Description
    mask : ndarray
        Description
    split : int
        Description
    
    Returns
    -------
    raw_trn : ndarray
        Description
    mask_trn : ndarray
        Description
    raw_val : ndarray
        Description
    mask_val : ndarray
        Description
    
    Raises
    ------
    """
    
    # Nested function ---------------------------------------------------------
    
    def _augment_data(raw, mask, operations):
        
        rand = random.randint(0, raw.shape[0]-1)
        outputs = operations(image=raw[rand,:,:], mask=mask[rand,:,:])
            
        raw_aug = outputs['image']
        mask_aug = outputs['mask']
        
        return raw_aug, mask_aug
    
    # Main function -----------------------------------------------------------
        
    if parallel:
 
        # Run _augment_data (parallel)
        output_list = Parallel(n_jobs=-1)(
            delayed(_augment_data)(
                raw,
                mask,
                operations
                )
            for i in range(iterations)
            )
            
    else:
            
        # Run _augment_data
        output_list = [_augment_data(
                raw,
                mask,
                operations
                ) 
            for i in range(iterations)
            ]

    # Extract outputs
    raw_aug = np.stack([arrays[0] for arrays in output_list], axis=0)
    mask_aug = np.stack([arrays[1] for arrays in output_list], axis=0)
    
    return raw_aug, mask_aug

''' ----------------------------------------------------------------------- '''

# Define operations 
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

raw_val, mask_val = augment_data(
    raw_val, mask_val, operations, iterations=raw_val.shape[0]*100, parallel=False
    )

#%% Configure and train StarDist2D

from stardist import gputools_available
from stardist.models import Config2D, StarDist2D

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
    train_epochs=8,
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

model.train(
    raw_trn, 
    mask_trn, 
    validation_data=(raw_val, mask_val)
    )

#%%

raw_val_pred = np.zeros_like(mask_val)
for i, temp_raw in enumerate(raw_val):
    raw_val_pred[i,...] = model.predict_instances(
        temp_raw,
        prob_thresh=0.3,
        nms_thresh=0.3,    
        n_tiles=model._guess_n_tiles(temp_raw),
        )[0]
    
viewer = napari.Viewer()
viewer.add_image(raw_val)
viewer.add_labels(raw_val_pred)


#%%

img = io.imread(
    Path('data', '20mbar_end_AV_20171019_162451_substack(1-900-5)_t0.tif')    
    )

img = norm_data(img, qlow, qhigh)

img_pred = model.predict_instances(
    img,
    prob_thresh=0.3,
    nms_thresh=0.3,    
    n_tiles=model._guess_n_tiles(img),
    )[0]

viewer = napari.Viewer()
viewer.add_image(img)
viewer.add_labels(img_pred)
