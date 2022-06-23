#%% Imports

import numpy as np
from skimage import io 
from pathlib import Path
import matplotlib.pyplot as plt
from joblib import Parallel, delayed 

#%% Inputs

qlow = 0.001
qhigh = 0.999
split = 0.25

#%% Create paths & open data

for path in Path('data').iterdir(): 

    if 'crop' in path.stem and 'labels' not in path.stem:   
        
        labels_path = Path('data', path.stem + '_labels.tif')
        
        if 'raw' not in globals(): 
            raw = io.imread(path)
            labels = io.imread(labels_path)    
        else:
            raw = np.dstack((raw, io.imread(path)))
            labels = np.dstack((labels, io.imread(labels_path))) 
            
raw = np.rollaxis(raw, 2)      
labels = np.rollaxis(labels, 2)   
               
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

def split_data(raw, labels, split):
    
    """
    Description
    
    Parameters
    ----------
    raw : ndarray
        Description
    labels : ndarray
        Description
    split : int
        Description
    
    Returns
    -------
    raw_trn : ndarray
        Description
    labels_trn : ndarray
        Description
    raw_val : ndarray
        Description
    labels_val : ndarray
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
    labels_trn = labels[trn_idx,...]
    raw_val = raw[val_idx,...]
    labels_val = labels[val_idx,...]

    return raw_trn, labels_trn, raw_val, labels_val

''' ----------------------------------------------------------------------- '''

raw_trn, labels_trn, raw_val, labels_val = split_data(raw, labels, split)

#%% Augment data

import napari
import albumentations as A

def augment_data(raw, labels, operations, iterations=256, parallel=True):
    
    """
    Description
    
    Parameters
    ----------
    raw : ndarray
        Description
    labels : ndarray
        Description
    split : int
        Description
    
    Returns
    -------
    raw_trn : ndarray
        Description
    labels_trn : ndarray
        Description
    raw_val : ndarray
        Description
    labels_val : ndarray
        Description
    
    Raises
    ------
    """
    
    # Nested function ---------------------------------------------------------
    
    def _augment_data(raw, labels, operations):
        
        rand = random.randint(0, raw.shape[0]-1)
        outputs = operations(image=raw[rand,:,:], labels=labels[rand,:,:])
            
        raw_aug = outputs['image']
        labels_aug = outputs['labels']
        
        return raw_aug, labels_aug
    
    # Main function -----------------------------------------------------------
        
    if parallel:
 
        # Run _augment_data (parallel)
        output_list = Parallel(n_jobs=-1)(
            delayed(_augment_data)(
                raw,
                labels,
                operations
                )
            for i in range(iterations)
            )
            
    else:
            
        # Run _augment_data
        output_list = [_augment_data(
                raw,
                labels,
                operations
                ) 
            for i in range(iterations)
            ]

    # Extract outputs
    raw_aug = np.stack([arrays[0] for arrays in output_list], axis=0)
    labels_aug = np.stack([arrays[1] for arrays in output_list], axis=0)
    
    return raw_aug, labels_aug

''' ----------------------------------------------------------------------- '''

# Define operations 
operations = A.Compose([
    A.VerticalFlip(p=0.5),              
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.Transpose(p=0.5),
    A.GridDistortion(p=0.5),
    ])

raw_trn_aug, labels_trn_aug = augment_data(
    raw_trn, labels_trn, operations, iterations=1000, parallel=False
    )

viewer = napari.Viewer()
viewer.add_image(raw_trn_aug)
viewer.add_labels(labels_trn_aug)
