#%% Imports

import napari
import numpy as np
from skimage import io 
from pathlib import Path


#%% Inputs

sample_size = 50
crop_size = 128

#%% Process data
    
from joblib import Parallel, delayed 
from skimage.restoration import rolling_ball

def process_data(raw, radius=2, parallel=True):
    
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
            
    return process
    
#%% Open & process raw data

data = []
for path in Path('data', 'raw').iterdir():   
    raw = io.imread(path)   
    process = process_data(raw, radius=2, parallel=True)
    data.append((raw, process, raw.shape, path))

#%% Generate random cropped images (seeded)

np.random.seed(1)

# Generate random indexes
randR = np.random.randint(0, len(data), size=sample_size)
randI = np.zeros_like(randR)
randY = np.zeros_like(randR)
randX = np.zeros_like(randR)
for i in range(randR.shape[0]):    
    randI[i] = np.random.randint(0, data[randR[i]][0].shape[0])
    randY[i] = np.random.randint(0, data[randR[i]][0].shape[1]-crop_size)
    randX[i] = np.random.randint(0, data[randR[i]][0].shape[2]-crop_size)    

#%%

# Fuse both loops !!!!!

# Extract cropped data
data_crop = []
for r, i, y, x in zip(randR, randI, randY, randX):
    data_crop.append((
        data[r][0][i,y:y+crop_size,x:x+crop_size],
        data[r][1][i,y:y+crop_size,x:x+crop_size],
        (r, i, y, x),
        f'{data[r][3].stem}_crop({r:03}-{i:03}-{y:03}-{x:03})' # name
        ))    

# Save data

for i in range(len(data_crop)):
    
    io.imsave(Path(
        'data', 'train', data_crop[i][3] + '.tif'
        ), data_crop[i][1], check_contrast=False
        )

        
#%%

viewer = napari.Viewer()
viewer.add_image(data_crop[0][1])
