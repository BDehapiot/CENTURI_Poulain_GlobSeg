#%% Imports

import numpy as np
from skimage import io 
from pathlib import Path

#%% Inputs

raw_name = '20mbar_end_AV_20171019_162451_substack(1-900-5).tif'
# file_name = '35mbar_end_05_20170613_151622_substack(1-900-5).tif'

sample_size = 50
img_size = 128

#%%

raw =  io.imread(Path('data', raw_name))

nI = raw.shape[0]
nY = raw.shape[1]
nX = raw.shape[2]

#%%

randI = np.random.randint(0, nI, size=sample_size)
randY = np.random.randint(0, nY-img_size, size=sample_size)
randX = np.random.randint(0, nX-img_size, size=sample_size)

rand_img = np.zeros((sample_size, img_size, img_size))
for i, y, x in zip(randI, randY, randX):
    rand_img[i,...] = raw[i,y:y+img_size,x:x+img_size]