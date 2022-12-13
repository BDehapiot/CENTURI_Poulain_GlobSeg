#%% Imports

import random
import numpy as np
from joblib import Parallel, delayed 
from skimage.restoration import rolling_ball

#%%

def process_data(raw, radius=0, parallel=True):
    
    """
    Description
    
    Parameters
    ----------
    raw : ndarray
        Description
        
    radius : int
        Description
    
    Returns
    -------
    process : ndarray
        Description
    
    Raises
    ------
    """
    
    # Nested function ---------------------------------------------------------
    
    def _process_data(temp):
    
        # Subtract background
        process =  temp - rolling_ball(temp, radius=radius)
        
        return process
    
    # Main function -----------------------------------------------------------
        
    # Subtract mean projection
    temp = raw - np.mean(raw, axis=0)
    
    if radius > 0:
    
        if parallel:
     
            # Run _process_data (parallel)
            output_list = Parallel(n_jobs=-1)(
                delayed(_process_data)(
                    img
                    )
                for img in temp
                )
                
        else:
                
            # Run _process_data
            output_list = [_process_data(
                    img
                    ) 
                for img in temp
                ]
    
        # Extract outputs
        process = np.stack([arrays for arrays in output_list], axis=0)
        
    else:   
            
        process = temp
            
    return process

#%% norm_data

def norm_data(raw, qlow=0.001, qhigh=0.999):
    
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

#%% split_data

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

#%% augment_data

def augment_data(raw, mask, operations, iterations=256, parallel=True):
    
    """
    Description
    
    Parameters
    ----------
    raw : ndarray
        Description
        
    mask : ndarray
        Description
        
    operations : compose object (see albumentation)
        Description
        
    iterations : int
        Description
            
    parallel : bool
        Description
    
    Returns
    -------
    raw_aug : ndarray
        Description
        
    mask_aug : ndarray
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
