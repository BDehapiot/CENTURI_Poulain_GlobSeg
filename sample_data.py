#%% Imports

import numpy as np
from skimage import io 
from pathlib import Path

from functions import range_uint8, process_data

#%% Inputs

sample_size = 50 # number of random cropped images 
seed = 1 # seed for random indexes
crop_size = 128 # size of random cropped images 
radius = 3 # radius for rolling ball background subtration (0 = deactivate)
   
#%% Open & process raw data

data = []
for path in Path('data', 'raw').iterdir():   
    raw = io.imread(path)   
    process = process_data(raw, radius=radius, parallel=True)
    data.append((raw, process, raw.shape, path))

#%% Generate random cropped images (seeded)

np.random.seed(seed)

# Generate random indexes
randR = np.random.randint(0, len(data), size=sample_size)
randI = np.zeros_like(randR)
randY = np.zeros_like(randR)
randX = np.zeros_like(randR)

for i in range(randR.shape[0]):    
    
    randI[i] = np.random.randint(0, data[randR[i]][0].shape[0])
    
    if data[randR[i]][0].shape[1] > crop_size:
        
        randY[i] = np.random.randint(0, data[randR[i]][0].shape[1]-crop_size)
        
    else:
        
        randY[i] = 0
        
    if data[randR[i]][0].shape[2] > crop_size:
        
        randX[i] = np.random.randint(0, data[randR[i]][0].shape[2]-crop_size)    
        
    else:
        
        randX[i] = 0

#%% Extract & save cropped data

for r, i, y, x in zip(randR, randI, randY, randX):
    crop = data[r][1][i,y:y+crop_size,x:x+crop_size]
    path = Path('data', 'train', 
        f'{data[r][3].stem}_crop({r:02}-{i:04}-{y:04}-{x:04}).tif'
        )    
    io.imsave(path, range_uint8(crop, int_range=0.99), check_contrast=False)