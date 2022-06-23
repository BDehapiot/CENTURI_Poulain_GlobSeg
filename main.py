#%% Imports

import napari
import numpy as np
from skimage import io 
from pathlib import Path
import matplotlib.pyplot as plt
from joblib import Parallel, delayed 

#%%

file_name = '20mbar_end_AV_20171019_162451_substack(1-900-5).tif'
# file_name = '35mbar_end_05_20170613_151622_substack(1-900-5).tif'

#%% Initialize

# create paths
root_path = Path.cwd()
data_path = Path(root_path / 'data' )
file_path = Path(root_path / 'data' / file_name)

# open data
stack = io.imread(file_path) 

#%% Pre-process

from skimage.restoration import rolling_ball

def preprocess(stack):
    
    # convert to float
    preprocessed = stack.astype('float').copy()
    
    # subtract mean projection
    mean_proj = np.mean(preprocessed, axis=0)
    preprocessed -= mean_proj
    
    # subtract background (rolling_ball)
    for i, img in enumerate(preprocessed):
        
        preprocessed[i,...] = img - rolling_ball(img, radius=2)

    return preprocessed
        
preprocessed = preprocess(stack)
io.imsave(data_path / 'preprocessed.tif', preprocessed.astype('uint8'), check_contrast=False) 

#%% Create images for training

if file_name == '20mbar_end_AV_20171019_162451_substack(1-900-5).tif':
    y_crop = 5; x_crop = 90
  
if file_name == '35mbar_end_05_20170613_151622_substack(1-900-5).tif':
    y_crop = 60; x_crop = 280
   
t_crop = np.arange(0,stack.shape[0],stack.shape[0]//5)   

for t in t_crop: 
    
    img = stack[t,
        y_crop:y_crop+128,
        x_crop:x_crop+128
        ]
    
    io.imsave(
        data_path / f'{file_path.stem}_crop({t:03}-{y_crop:03}-{x_crop:03}).tif', 
        img, check_contrast=False
        )

#%%

# viewer = napari.view_image(preprocessed)
        