#%% Imports

import skvideo.io 
import numpy as np
from skimage import io 
from pathlib import Path

from functions import range_uint8, process_data

#%% Inputs

dir_path = '/home/bdehapiot/Projects/CENTURI_Poulain_GlobSeg/data/movies/'
mov_name  = 'ML30_outlet_donneurFYY_Temp267_x20_DeltaP20mBars_vid0002.wmv'

#%% Open raw movie

raw = skvideo.io.vread(dir_path + mov_name)[...,0]   

#%%
io.imsave(dir_path + mov_name.replace('wmv', 'tif'), raw, check_contrast=False)
