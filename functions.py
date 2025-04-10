#%% Imports -------------------------------------------------------------------

import random
import numpy as np
from joblib import Parallel, delayed 
from skimage.restoration import rolling_ball

#%% Function : process_data() -------------------------------------------------

def process_data(arr, radius=0):
        
    def _process_data(temp):
        return temp - rolling_ball(temp, radius=radius)

    # Subtract mean projection
    prp = arr - np.mean(arr, axis=0)
    
    # Subtract background
    if radius > 0:    
        outputs = Parallel(n_jobs=-1)(
            delayed(_process_data)(img) 
            for img in prp
            ) 
        prp = np.stack([data for data in outputs])
            
    return prp

#%% Function : norm_data() ----------------------------------------------------

def norm_data(arr, qlow=0.001, qhigh=0.999):
    
    arr = arr.astype("float32")
    
    for i, img in enumerate(arr):
        
        # Get lower and higher threshold
        low = np.quantile(img, qlow)
        hgh = np.quantile(img, qhigh)
        
        # Normalize image
        img = (img - low) / (hgh - low)
        img[img > 1] = 1
        img[img < 0] = 0
        
        # Update arr
        arr[i,...] = img
        
    return arr           

#%% Function : split_data() ---------------------------------------------------

def split_data(arr, mask, split):
        
    nI = arr.shape[0]
    
    # Define indexes
    idx = random.sample(range(0, nI), nI)
    trn_idx = idx[0:int(nI*(1-split))]
    val_idx = idx[-(nI-len(trn_idx)):]
    
    # Extract data    
    arr_trn = arr[trn_idx,...]
    mask_trn = mask[trn_idx,...]
    arr_val = arr[val_idx,...]
    mask_val = mask[val_idx,...]

    return arr_trn, mask_trn, arr_val, mask_val

#%% Function : augment_data() -------------------------------------------------

def augment_data(arr, mask, operations, iterations=256):

    def _augment_data(arr, mask, operations):
        rand = random.randint(0, arr.shape[0]-1)
        outputs = operations(image=arr[rand,:,:], mask=mask[rand,:,:])
        arr_aug = outputs['image']
        mask_aug = outputs['mask']
        return arr_aug, mask_aug
    
    # Augment data
    outputs = Parallel(n_jobs=-1)(
        delayed(_augment_data)(arr, mask, operations)
        for i in range(iterations)
        )
    arr_aug = np.stack([data[0] for data in outputs], axis=0)
    mask_aug = np.stack([data[1] for data in outputs], axis=0)
    
    return arr_aug, mask_aug
